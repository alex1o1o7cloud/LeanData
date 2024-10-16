import Mathlib

namespace NUMINAMATH_CALUDE_championship_outcomes_l1284_128484

def number_of_competitors : ℕ := 5
def number_of_events : ℕ := 3

theorem championship_outcomes :
  (number_of_competitors ^ number_of_events : ℕ) = 125 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_l1284_128484


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l1284_128462

-- Define the plane and points
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P]
variable (O A B C : P)

-- Define the non-collinearity condition
def noncollinear (O A B C : P) : Prop :=
  ¬ (∃ (a b c : ℝ), a • (A - O) + b • (B - O) + c • (C - O) = 0 ∧ (a, b, c) ≠ (0, 0, 0))

-- State the theorem
theorem vector_ratio_theorem (h_noncollinear : noncollinear P O A B C)
  (h_eq : A - O - 4 • (B - O) + 3 • (C - O) = 0) :
  ‖A - B‖ / ‖C - A‖ = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l1284_128462


namespace NUMINAMATH_CALUDE_sin_value_at_pi_over_four_l1284_128451

theorem sin_value_at_pi_over_four 
  (φ : Real) 
  (ω : Real)
  (h1 : (- 4 : Real) / 5 = Real.cos φ ∧ (3 : Real) / 5 = Real.sin φ)
  (h2 : (2 * Real.pi) / ω = Real.pi)
  (h3 : ω > 0) :
  Real.sin ((2 : Real) * Real.pi / 4 + φ) = - (4 : Real) / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_value_at_pi_over_four_l1284_128451


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1284_128443

theorem nested_fraction_evaluation :
  1 / (3 - 1 / (2 - 1 / (3 - 1 / (2 - 1 / 2)))) = 11 / 26 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1284_128443


namespace NUMINAMATH_CALUDE_movie_theater_deal_l1284_128429

/-- Movie theater deal problem -/
theorem movie_theater_deal (deal_price : ℝ) (ticket_price : ℝ) (savings : ℝ)
  (h1 : deal_price = 20)
  (h2 : ticket_price = 8)
  (h3 : savings = 2) :
  let popcorn_price := ticket_price - 3
  let total_normal_price := deal_price + savings
  let drink_price := (total_normal_price - ticket_price - popcorn_price) * (2/3)
  drink_price - popcorn_price = 1 := by sorry

end NUMINAMATH_CALUDE_movie_theater_deal_l1284_128429


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1284_128476

/-- A geometric sequence is a sequence where the ratio between successive terms is constant. -/
def IsGeometricSequence (a b c : ℝ) : Prop := b ^ 2 = a * c

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The discriminant of a quadratic function ax^2 + bx + c -/
def Discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_no_real_roots
  (a b c : ℝ)
  (h_geom : IsGeometricSequence a b c)
  (h_a_nonzero : a ≠ 0) :
  (∀ x : ℝ, QuadraticFunction a b c x ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1284_128476


namespace NUMINAMATH_CALUDE_jackie_cosmetics_purchase_l1284_128471

/-- The cost of a bottle of lotion -/
def lotion_cost : ℚ := 6

/-- The number of bottles of lotion purchased -/
def lotion_quantity : ℕ := 3

/-- The amount needed to reach the free shipping threshold -/
def additional_amount : ℚ := 12

/-- The free shipping threshold -/
def free_shipping_threshold : ℚ := 50

/-- The cost of a bottle of shampoo or conditioner -/
def shampoo_conditioner_cost : ℚ := 10

theorem jackie_cosmetics_purchase :
  2 * shampoo_conditioner_cost + lotion_cost * lotion_quantity + additional_amount = free_shipping_threshold := by
  sorry

end NUMINAMATH_CALUDE_jackie_cosmetics_purchase_l1284_128471


namespace NUMINAMATH_CALUDE_pizza_cost_per_piece_l1284_128427

/-- 
Given that Luigi bought 4 pizzas for $80 and each pizza was cut into 5 pieces,
prove that each piece of pizza costs $4.
-/
theorem pizza_cost_per_piece 
  (num_pizzas : ℕ) 
  (total_cost : ℚ) 
  (pieces_per_pizza : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : total_cost = 80) 
  (h3 : pieces_per_pizza = 5) : 
  total_cost / (num_pizzas * pieces_per_pizza : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_pizza_cost_per_piece_l1284_128427


namespace NUMINAMATH_CALUDE_ab_equals_zero_l1284_128473

theorem ab_equals_zero (a b : ℝ) 
  (h1 : (2 : ℝ) ^ a = (2 : ℝ) ^ (2 * (b + 1)))
  (h2 : (7 : ℝ) ^ b = (7 : ℝ) ^ (a - 2)) : 
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_ab_equals_zero_l1284_128473


namespace NUMINAMATH_CALUDE_largest_smallest_divisible_by_165_l1284_128479

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧  -- 7-digit number
  (n % 165 = 0) ∧  -- divisible by 165
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6] →
    (∃! i : ℕ, i < 7 ∧ (n / 10^i) % 10 = d)  -- each digit appears exactly once

theorem largest_smallest_divisible_by_165 :
  (∀ n : ℕ, is_valid_number n → n ≤ 6431205) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ 1042635) ∧
  is_valid_number 6431205 ∧
  is_valid_number 1042635 :=
sorry

end NUMINAMATH_CALUDE_largest_smallest_divisible_by_165_l1284_128479


namespace NUMINAMATH_CALUDE_equal_positive_integers_l1284_128453

theorem equal_positive_integers (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ k : ℕ, b^n + n = k * (a^n + n)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_positive_integers_l1284_128453


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1284_128424

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1284_128424


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_planes_l1284_128482

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Statement 1
theorem perpendicular_line_to_plane 
  (α : Plane) (a l₁ l₂ : Line) :
  contains α l₁ → 
  contains α l₂ → 
  intersect l₁ l₂ → 
  perpendicular a l₁ → 
  perpendicular a l₂ → 
  perpendicularLP a α :=
sorry

-- Statement 6
theorem perpendicular_planes 
  (α β : Plane) (b : Line) :
  contains β b → 
  perpendicularLP b α → 
  perpendicularPP β α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_perpendicular_planes_l1284_128482


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l1284_128477

/-- Represents the number of types of each clothing item -/
def num_types : ℕ := 8

/-- Represents the number of colors available -/
def num_colors : ℕ := 8

/-- Calculates the total number of outfit combinations -/
def total_combinations : ℕ := num_types^4

/-- Calculates the number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Theorem: The number of valid outfit choices is 4088 -/
theorem valid_outfit_choices : 
  total_combinations - same_color_outfits = 4088 := by sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l1284_128477


namespace NUMINAMATH_CALUDE_fourth_number_proof_l1284_128404

theorem fourth_number_proof : ∃ x : ℕ, 9548 + 7314 = 3362 + x ∧ x = 13500 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l1284_128404


namespace NUMINAMATH_CALUDE_club_members_count_l1284_128437

theorem club_members_count :
  ∃! n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 ∧ n = 226 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l1284_128437


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1284_128458

/-- A regular polygon with exterior angle 20° and side length 10 has perimeter 180 -/
theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) : 
  n > 2 →
  exterior_angle = 20 →
  side_length = 10 →
  n * exterior_angle = 360 →
  n * side_length = 180 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1284_128458


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_numbers_l1284_128447

theorem largest_divisor_of_consecutive_even_numbers : ∃ (m : ℕ), 
  (∀ (n : ℕ), (2*n) * (2*n + 2) * (2*n + 4) % m = 0) ∧ 
  (∀ (k : ℕ), k > m → ∃ (n : ℕ), (2*n) * (2*n + 2) * (2*n + 4) % k ≠ 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_numbers_l1284_128447


namespace NUMINAMATH_CALUDE_max_value_f_range_of_a_l1284_128496

-- Define the function f(x) = x^2 - 2ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Theorem 1: Maximum value of f(x) when a = 1 and x ∈ [-1, 2]
theorem max_value_f (x : ℝ) (h : x ∈ Set.Icc (-1) 2) : 
  f 1 x ≤ 5 :=
sorry

-- Theorem 2: Range of a when f(x) ≥ a for x ∈ [-1, +∞)
theorem range_of_a (a : ℝ) :
  (∀ x ≥ -1, f a x ≥ a) ↔ a ∈ Set.Icc (-3) 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_range_of_a_l1284_128496


namespace NUMINAMATH_CALUDE_no_real_solutions_l1284_128409

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (3 * x^2) / (x - 2) - (3 * x + 9) / 4 + (5 - 9 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1284_128409


namespace NUMINAMATH_CALUDE_ellipse_equation_l1284_128425

/-- Given two ellipses C1 and C2, where C1 is defined by x²/4 + y² = 1,
    C2 has the same eccentricity as C1, and the minor axis of C2 is
    the same as the major axis of C1, prove that the equation of C2 is
    y²/16 + x²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let C1 := {(x, y) | x^2/4 + y^2 = 1}
  let e1 := Real.sqrt (1 - (2^2)/(4^2))  -- eccentricity of C1
  let C2 := {(x, y) | ∃ (a : ℝ), a > 2 ∧ y^2/a^2 + x^2/4 = 1 ∧ Real.sqrt (1 - (2^2)/(a^2)) = e1}
  ∀ (x y : ℝ), (x, y) ∈ C2 ↔ y^2/16 + x^2/4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1284_128425


namespace NUMINAMATH_CALUDE_path_count_on_grid_l1284_128445

/-- The number of distinct paths on a 6x5 grid from upper left to lower right corner -/
def number_of_paths : ℕ := 126

/-- The number of right moves required to reach the right edge of a 6x5 grid -/
def right_moves : ℕ := 5

/-- The number of down moves required to reach the bottom edge of a 6x5 grid -/
def down_moves : ℕ := 4

/-- The total number of moves (right + down) required to reach the bottom right corner -/
def total_moves : ℕ := right_moves + down_moves

theorem path_count_on_grid : 
  number_of_paths = Nat.choose total_moves right_moves :=
by sorry

end NUMINAMATH_CALUDE_path_count_on_grid_l1284_128445


namespace NUMINAMATH_CALUDE_arrangementsWithConstraintFor5_l1284_128439

/-- The number of ways to arrange n distinct objects with one object required to be before another -/
def arrangementsWithConstraint (n : ℕ) : ℕ :=
  Nat.factorial n / 2

/-- The theorem stating that for 5 objects, the number of arrangements with one object required to be before another is 60 -/
theorem arrangementsWithConstraintFor5 :
  arrangementsWithConstraint 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arrangementsWithConstraintFor5_l1284_128439


namespace NUMINAMATH_CALUDE_acid_solution_dilution_l1284_128486

theorem acid_solution_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 15) / 100 * (m + x)) → x = 15 * m / (m - 15) := by
  sorry

end NUMINAMATH_CALUDE_acid_solution_dilution_l1284_128486


namespace NUMINAMATH_CALUDE_correct_factorization_l1284_128485

theorem correct_factorization (m : ℤ) : m^3 + m = m * (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1284_128485


namespace NUMINAMATH_CALUDE_erased_numbers_theorem_l1284_128467

def sumBetween (a b : ℕ) : ℕ := (b - a - 1) * (a + b) / 2

def sumOutside (a b : ℕ) : ℕ := (2018 * 2019) / 2 - sumBetween a b - a - b

theorem erased_numbers_theorem (a b : ℕ) (ha : a = 673) (hb : b = 1346) :
  2 * sumBetween a b = sumOutside a b := by
  sorry

end NUMINAMATH_CALUDE_erased_numbers_theorem_l1284_128467


namespace NUMINAMATH_CALUDE_prove_A_equals_five_l1284_128410

/-- Given that 14A and B73 are three-digit numbers, 14A + B73 = 418, and A and B are single digits, prove that A = 5 -/
theorem prove_A_equals_five (A B : ℕ) : 
  (100 ≤ 14 * A) ∧ (14 * A < 1000) ∧  -- 14A is a three-digit number
  (100 ≤ B * 100 + 73) ∧ (B * 100 + 73 < 1000) ∧  -- B73 is a three-digit number
  (14 * A + B * 100 + 73 = 418) ∧  -- 14A + B73 = 418
  (A < 10) ∧ (B < 10) →  -- A and B are single digits
  A = 5 := by sorry

end NUMINAMATH_CALUDE_prove_A_equals_five_l1284_128410


namespace NUMINAMATH_CALUDE_rectangle_triangle_count_l1284_128400

theorem rectangle_triangle_count (n m : ℕ) (hn : n = 6) (hm : m = 7) :
  n.choose 2 * m + m.choose 2 * n = 231 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_count_l1284_128400


namespace NUMINAMATH_CALUDE_down_payment_calculation_l1284_128480

theorem down_payment_calculation (purchase_price : ℝ) 
  (monthly_payment : ℝ) (num_payments : ℕ) (interest_rate : ℝ) :
  purchase_price = 118 →
  monthly_payment = 10 →
  num_payments = 12 →
  interest_rate = 0.15254237288135593 →
  ∃ (down_payment : ℝ),
    down_payment + (monthly_payment * num_payments) = 
      purchase_price * (1 + interest_rate) ∧
    down_payment = 16 :=
by sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l1284_128480


namespace NUMINAMATH_CALUDE_no_k_exists_product_minus_one_is_power_l1284_128493

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k odd prime numbers -/
def productFirstKOddPrimes (k : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number k such that the product of the first k odd prime numbers minus 1 is an exact power of a natural number greater than one -/
theorem no_k_exists_product_minus_one_is_power :
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstKOddPrimes k - 1 = a^n :=
sorry

end NUMINAMATH_CALUDE_no_k_exists_product_minus_one_is_power_l1284_128493


namespace NUMINAMATH_CALUDE_average_of_numbers_l1284_128495

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125831.9 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1284_128495


namespace NUMINAMATH_CALUDE_combined_travel_time_l1284_128414

/-- 
Given a car that takes 4.5 hours to reach station B, and a train that takes 2 hours longer 
than the car to travel the same distance, the combined time for both to reach station B is 11 hours.
-/
theorem combined_travel_time (car_time train_time : ℝ) : 
  car_time = 4.5 → 
  train_time = car_time + 2 → 
  car_time + train_time = 11 := by
sorry

end NUMINAMATH_CALUDE_combined_travel_time_l1284_128414


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1284_128434

def indistinguishable_distributions (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem five_balls_three_boxes : 
  indistinguishable_distributions 5 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1284_128434


namespace NUMINAMATH_CALUDE_log_base_three_squared_l1284_128433

theorem log_base_three_squared (m : ℝ) (b : ℝ) (h : 3^m = b) : 
  Real.log b / Real.log (3^2) = m / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_base_three_squared_l1284_128433


namespace NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l1284_128406

theorem four_numbers_product_sum_prime :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧
  Nat.Prime (a * b + c * d) ∧
  Nat.Prime (a * c + b * d) ∧
  Nat.Prime (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l1284_128406


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1284_128413

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 2 * x₁ - 15 = 0) →
  (3 * x₂^2 - 2 * x₂ - 15 = 0) →
  (x₁^2 + x₂^2 = 94/9) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1284_128413


namespace NUMINAMATH_CALUDE_base10_231_to_base6_l1284_128474

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- Checks if a list of digits is a valid base 6 representation -/
def isValidBase6 (l : List ℕ) : Prop :=
  l.all (· < 6) ∧ l ≠ []

theorem base10_231_to_base6 :
  let base6 := toBase6 231
  isValidBase6 base6 ∧ base6 = [1, 0, 2, 3] :=
sorry

end NUMINAMATH_CALUDE_base10_231_to_base6_l1284_128474


namespace NUMINAMATH_CALUDE_right_triangle_side_length_right_triangle_side_length_proof_l1284_128463

/-- Given a right triangle with hypotenuse length 13 and one non-hypotenuse side length 12,
    the length of the other side is 5. -/
theorem right_triangle_side_length : ℝ → ℝ → ℝ → Prop :=
  fun hypotenuse side1 side2 =>
    hypotenuse = 13 ∧ side1 = 12 ∧ side2 * side2 + side1 * side1 = hypotenuse * hypotenuse →
    side2 = 5

/-- Proof of the theorem -/
theorem right_triangle_side_length_proof : right_triangle_side_length 13 12 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_right_triangle_side_length_proof_l1284_128463


namespace NUMINAMATH_CALUDE_lauren_earnings_l1284_128430

/-- Represents the earnings for a single day --/
structure DayEarnings where
  commercial_rate : ℝ
  subscription_rate : ℝ
  commercial_views : ℕ
  subscriptions : ℕ

/-- Calculates the total earnings for a single day --/
def day_total (d : DayEarnings) : ℝ :=
  d.commercial_rate * d.commercial_views + d.subscription_rate * d.subscriptions

/-- Represents the earnings for the weekend --/
structure WeekendEarnings where
  merchandise_sales : ℝ
  merchandise_rate : ℝ

/-- Calculates the total earnings for the weekend --/
def weekend_total (w : WeekendEarnings) : ℝ :=
  w.merchandise_sales * w.merchandise_rate

/-- Represents Lauren's earnings for the entire period --/
structure PeriodEarnings where
  monday : DayEarnings
  tuesday : DayEarnings
  weekend : WeekendEarnings

/-- Calculates the total earnings for the entire period --/
def period_total (p : PeriodEarnings) : ℝ :=
  day_total p.monday + day_total p.tuesday + weekend_total p.weekend

/-- Theorem stating that Lauren's total earnings for the period equal $140.00 --/
theorem lauren_earnings :
  let p : PeriodEarnings := {
    monday := {
      commercial_rate := 0.40,
      subscription_rate := 0.80,
      commercial_views := 80,
      subscriptions := 20
    },
    tuesday := {
      commercial_rate := 0.50,
      subscription_rate := 1.00,
      commercial_views := 100,
      subscriptions := 27
    },
    weekend := {
      merchandise_sales := 150,
      merchandise_rate := 0.10
    }
  }
  period_total p = 140
:= by sorry

end NUMINAMATH_CALUDE_lauren_earnings_l1284_128430


namespace NUMINAMATH_CALUDE_starship_age_conversion_l1284_128454

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_number (octal : List Nat) : Nat :=
  octal.enum.foldr (fun (i, digit) acc => acc + octal_to_decimal digit * (8^i)) 0

theorem starship_age_conversion :
  octal_to_decimal_number [6, 7, 2, 4] = 3540 := by
  sorry

end NUMINAMATH_CALUDE_starship_age_conversion_l1284_128454


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l1284_128499

/-- Proves that the average speed of a triathlete is 0.125 miles per minute
    given specific conditions for running and swimming. -/
theorem triathlete_average_speed
  (run_distance : ℝ)
  (swim_distance : ℝ)
  (run_speed : ℝ)
  (swim_speed : ℝ)
  (h1 : run_distance = 3)
  (h2 : swim_distance = 3)
  (h3 : run_speed = 10)
  (h4 : swim_speed = 6) :
  (run_distance + swim_distance) / ((run_distance / run_speed + swim_distance / swim_speed) * 60) = 0.125 := by
  sorry

#check triathlete_average_speed

end NUMINAMATH_CALUDE_triathlete_average_speed_l1284_128499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1284_128455

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_divisibility
  (a : ℕ → ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_div : ∀ n : ℕ, 2005 ∣ a n * a (n + 31)) :
  ∀ n : ℕ, 2005 ∣ a n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1284_128455


namespace NUMINAMATH_CALUDE_system_solution_l1284_128419

theorem system_solution : 
  let x : ℚ := 25 / 31
  let y : ℚ := -11 / 31
  (3 * x + 4 * y = 1) ∧ (7 * x - y = 6) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1284_128419


namespace NUMINAMATH_CALUDE_bicycle_discount_l1284_128411

theorem bicycle_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 ∧ 
  discount1 = 0.4 ∧ 
  discount2 = 0.25 → 
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_discount_l1284_128411


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1284_128464

theorem smallest_n_for_inequality : ∀ n : ℕ, n ≥ 5 → 2^n > n^2 ∧ ∀ k : ℕ, k < 5 → 2^k ≤ k^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1284_128464


namespace NUMINAMATH_CALUDE_class_test_probability_l1284_128428

theorem class_test_probability (p_first p_second p_both : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.25)
  (h3 : p_both = 0.20) :
  1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_probability_l1284_128428


namespace NUMINAMATH_CALUDE_smallest_n_value_l1284_128407

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of invisible cubes when three faces are shown -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Theorem stating the smallest possible value of N -/
theorem smallest_n_value (d : BlockDimensions) : 
  invisibleCubes d = 143 → totalCubes d ≥ 336 ∧ ∃ d', invisibleCubes d' = 143 ∧ totalCubes d' = 336 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1284_128407


namespace NUMINAMATH_CALUDE_balanced_domino_config_exists_l1284_128456

/-- A domino configuration on an n × n board. -/
structure DominoConfig (n : ℕ) where
  /-- The number of dominoes in the configuration. -/
  num_dominoes : ℕ
  /-- Predicate that the configuration is balanced. -/
  is_balanced : Prop

/-- The minimum number of dominoes needed for a balanced configuration. -/
def min_dominoes (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 * n / 3 else 2 * n

/-- Theorem stating the existence of a balanced configuration and the minimum number of dominoes needed. -/
theorem balanced_domino_config_exists (n : ℕ) (h : n ≥ 3) :
  ∃ (config : DominoConfig n), config.is_balanced ∧ config.num_dominoes = min_dominoes n :=
by sorry

end NUMINAMATH_CALUDE_balanced_domino_config_exists_l1284_128456


namespace NUMINAMATH_CALUDE_train_passing_tree_l1284_128422

/-- Proves that a train 280 meters long, traveling at 72 km/hr, will take 14 seconds to pass a tree. -/
theorem train_passing_tree (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) :
  train_length = 280 ∧ 
  train_speed_kmh = 72 →
  time = train_length / (train_speed_kmh * (5/18)) ∧ 
  time = 14 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_tree_l1284_128422


namespace NUMINAMATH_CALUDE_division_problem_l1284_128416

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 2944) (h2 : quotient = 40) (h3 : remainder = 64) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 72 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l1284_128416


namespace NUMINAMATH_CALUDE_compare_expressions_compare_square_roots_l1284_128497

-- Problem 1
theorem compare_expressions (x y : ℝ) : x^2 + y^2 + 1 > 2*(x + y - 1) := by
  sorry

-- Problem 2
theorem compare_square_roots (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) :
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_compare_square_roots_l1284_128497


namespace NUMINAMATH_CALUDE_unique_symmetric_solutions_l1284_128401

theorem unique_symmetric_solutions (a b α β : ℝ) :
  (α * β = a ∧ α + β = b) →
  (∀ x y : ℝ, x * y = a ∧ x + y = b ↔ (x = α ∧ y = β) ∨ (x = β ∧ y = α)) :=
sorry

end NUMINAMATH_CALUDE_unique_symmetric_solutions_l1284_128401


namespace NUMINAMATH_CALUDE_john_travel_distance_l1284_128402

/-- Calculates the total distance traveled given a constant speed and two driving periods -/
def totalDistance (speed : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  speed * (time1 + time2)

/-- Proves that the total distance traveled is 225 miles -/
theorem john_travel_distance :
  let speed := 45
  let time1 := 2
  let time2 := 3
  totalDistance speed time1 time2 = 225 := by
sorry

end NUMINAMATH_CALUDE_john_travel_distance_l1284_128402


namespace NUMINAMATH_CALUDE_point_on_line_l1284_128469

/-- Given that the point (x, -3) lies on the straight line joining (2, 10) and (6, 2) in the xy-plane, prove that x = 8.5 -/
theorem point_on_line (x : ℝ) :
  (∃ t : ℝ, t ∈ (Set.Icc 0 1) ∧
    x = 2 * (1 - t) + 6 * t ∧
    -3 = 10 * (1 - t) + 2 * t) →
  x = 8.5 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1284_128469


namespace NUMINAMATH_CALUDE_parabola_property_l1284_128489

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_property (p : Parabola) :
  p.y_at (-3) = 4 →  -- vertex at (-3, 4)
  p.y_at (-2) = 7 →  -- passes through (-2, 7)
  3 * p.a + 2 * p.b + p.c = 76 := by
  sorry

end NUMINAMATH_CALUDE_parabola_property_l1284_128489


namespace NUMINAMATH_CALUDE_number_problem_l1284_128492

theorem number_problem (N : ℝ) :
  (4 / 5) * N = (N / (4 / 5)) - 27 → N = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1284_128492


namespace NUMINAMATH_CALUDE_cubic_extrema_l1284_128418

-- Define a cubic function
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d

-- Define the derivative of the cubic function
def cubic_derivative (a b c : ℝ) : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem cubic_extrema (a b c d : ℝ) :
  let f := cubic_function a b c d
  let f' := cubic_derivative (3*a) (2*b) c
  (∀ x, x * f' x = 0 ↔ x = 0 ∨ x = 2 ∨ x = -2) →
  (∀ x, f x ≤ f (-2)) ∧ (∀ x, f 2 ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_cubic_extrema_l1284_128418


namespace NUMINAMATH_CALUDE_vertex_x_coordinate_is_one_l1284_128459

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem vertex_x_coordinate_is_one 
  (a b c : ℝ) 
  (h1 : quadratic a b c 0 = 3)
  (h2 : quadratic a b c 2 = 3)
  (h3 : quadratic a b c 4 = 11) :
  ∃ k : ℝ, quadratic a b c x = a * (x - 1)^2 + k := by
sorry


end NUMINAMATH_CALUDE_vertex_x_coordinate_is_one_l1284_128459


namespace NUMINAMATH_CALUDE_jack_waiting_time_l1284_128442

/-- The total waiting time for Jack's trip to Canada -/
def total_waiting_time (customs_hours : ℕ) (quarantine_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  customs_hours + quarantine_days * hours_per_day

/-- Theorem stating that Jack's total waiting time is 356 hours -/
theorem jack_waiting_time :
  total_waiting_time 20 14 24 = 356 := by
  sorry

end NUMINAMATH_CALUDE_jack_waiting_time_l1284_128442


namespace NUMINAMATH_CALUDE_N_divisible_by_1980_l1284_128450

/-- The number formed by concatenating all two-digit numbers from 19 to 80 inclusive -/
def N : ℕ := sorry

/-- N is divisible by 1980 -/
theorem N_divisible_by_1980 : 1980 ∣ N := by sorry

end NUMINAMATH_CALUDE_N_divisible_by_1980_l1284_128450


namespace NUMINAMATH_CALUDE_projectile_trajectory_area_l1284_128465

theorem projectile_trajectory_area (v₀ g : ℝ) (h₁ : v₀ > 0) (h₂ : g > 0) :
  let v := fun t => v₀ + t * v₀  -- v varies from v₀ to 2v₀
  let x := fun t => (v t)^2 / (2 * g)
  let y := fun t => (v t)^2 / (4 * g)
  let area := ∫ t in (0)..(1), y (v t) * (x (v 1) - x (v 0))
  area = 3 * v₀^4 / (8 * g^2) :=
by sorry

end NUMINAMATH_CALUDE_projectile_trajectory_area_l1284_128465


namespace NUMINAMATH_CALUDE_puzzle_pieces_left_l1284_128438

theorem puzzle_pieces_left (total_pieces : ℕ) (num_boys : ℕ) (reyn_pieces : ℕ) : 
  total_pieces = 300 →
  num_boys = 3 →
  reyn_pieces = 25 →
  total_pieces - (reyn_pieces + 2 * reyn_pieces + 3 * reyn_pieces) = 150 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_pieces_left_l1284_128438


namespace NUMINAMATH_CALUDE_josephs_total_cards_l1284_128466

/-- The number of cards in a standard deck -/
def cards_per_deck : ℕ := 52

/-- The number of decks Joseph has -/
def josephs_decks : ℕ := 4

/-- Theorem: Joseph has 208 cards in total -/
theorem josephs_total_cards : 
  josephs_decks * cards_per_deck = 208 := by
  sorry

end NUMINAMATH_CALUDE_josephs_total_cards_l1284_128466


namespace NUMINAMATH_CALUDE_min_value_theorem_l1284_128472

/-- Given positive real numbers a, b, c, x, y, z satisfying certain conditions,
    the minimum value of a specific function is 1/2 -/
theorem min_value_theorem (a b c x y z : ℝ) 
    (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
    (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
    (eq1 : b * z + c * y = a)
    (eq2 : a * z + c * x = b)
    (eq3 : a * y + b * x = c) :
    (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 →
      x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ 
      x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)) →
    x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_min_value_theorem_l1284_128472


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1284_128460

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ x^2 + x = 210 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1284_128460


namespace NUMINAMATH_CALUDE_amanda_earnings_l1284_128440

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Number of appointments on Monday -/
def monday_appointments : ℕ := 5

/-- Duration of each Monday appointment in hours -/
def monday_appointment_duration : ℝ := 1.5

/-- Duration of Tuesday appointment in hours -/
def tuesday_appointment_duration : ℝ := 3

/-- Number of appointments on Thursday -/
def thursday_appointments : ℕ := 2

/-- Duration of each Thursday appointment in hours -/
def thursday_appointment_duration : ℝ := 2

/-- Duration of Saturday appointment in hours -/
def saturday_appointment_duration : ℝ := 6

/-- Total earnings for the week -/
def total_earnings : ℝ :=
  hourly_rate * (monday_appointments * monday_appointment_duration +
                 tuesday_appointment_duration +
                 thursday_appointments * thursday_appointment_duration +
                 saturday_appointment_duration)

theorem amanda_earnings : total_earnings = 410 := by
  sorry

end NUMINAMATH_CALUDE_amanda_earnings_l1284_128440


namespace NUMINAMATH_CALUDE_video_game_lives_l1284_128449

theorem video_game_lives (initial_lives hard_part_lives next_level_lives : ℝ) :
  initial_lives + hard_part_lives + next_level_lives =
  initial_lives + (hard_part_lives + next_level_lives) :=
by
  sorry

-- Example usage
def tiffany_game (initial_lives hard_part_lives next_level_lives : ℝ) : ℝ :=
  initial_lives + hard_part_lives + next_level_lives

#eval tiffany_game 43.0 14.0 27.0

end NUMINAMATH_CALUDE_video_game_lives_l1284_128449


namespace NUMINAMATH_CALUDE_function_properties_l1284_128412

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + 2*a*x

def g (a b : ℝ) (x : ℝ) : ℝ := 3*a^2 * Real.log x + b

-- State the theorem
theorem function_properties (a b : ℝ) :
  (a > 0) →
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    f a x₀ = g a b x₀ ∧ 
    (deriv (f a)) x₀ = (deriv (g a b)) x₀ ∧
    a = Real.exp 1) →
  (b = -(Real.exp 1)^2 / 2) ∧
  (∀ x > 0, f a x ≥ g a b x - b) →
  (0 < a ∧ a ≤ Real.exp ((5:ℝ)/6)) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1284_128412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l1284_128468

/-- Given an arithmetic sequence {a_n} where a_1 + a_7 + a_13 = 4π, 
    prove that tan(a_2 + a_12) = -√3 -/
theorem arithmetic_sequence_tangent (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 1 + a 7 + a 13 = 4 * Real.pi →                  -- given condition
  Real.tan (a 2 + a 12) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l1284_128468


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1284_128436

theorem inequality_and_equality_condition (p q : ℝ) (hp : 0 < p) (hq : p < q)
  (α β γ δ ε : ℝ) (hα : p ≤ α ∧ α ≤ q) (hβ : p ≤ β ∧ β ≤ q)
  (hγ : p ≤ γ ∧ γ ≤ q) (hδ : p ≤ δ ∧ δ ≤ q) (hε : p ≤ ε ∧ ε ≤ q) :
  (α + β + γ + δ + ε) * (1/α + 1/β + 1/γ + 1/δ + 1/ε) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ∧
  ((α + β + γ + δ + ε) * (1/α + 1/β + 1/γ + 1/δ + 1/ε) = 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 ↔
   ((α = p ∧ β = p ∧ γ = q ∧ δ = q ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = p ∧ δ = q ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = q ∧ δ = p ∧ ε = q) ∨
    (α = p ∧ β = q ∧ γ = q ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = p ∧ γ = p ∧ δ = q ∧ ε = q) ∨
    (α = q ∧ β = p ∧ γ = q ∧ δ = p ∧ ε = q) ∨
    (α = q ∧ β = p ∧ γ = q ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = q ∧ γ = p ∧ δ = p ∧ ε = q) ∨
    (α = q ∧ β = q ∧ γ = p ∧ δ = q ∧ ε = p) ∨
    (α = q ∧ β = q ∧ γ = q ∧ δ = p ∧ ε = p))) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1284_128436


namespace NUMINAMATH_CALUDE_fraction_equality_l1284_128490

theorem fraction_equality (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hab : a - b * (1 / a) ≠ 0) : 
  (a^2 - 1/b^2) / (b^2 - 1/a^2) = a^2 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1284_128490


namespace NUMINAMATH_CALUDE_percent_increase_decrease_l1284_128405

theorem percent_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 100) (hM : M > 0) :
  (M * (1 + p/100) * (1 - q/100) > M) ↔ (p > 100*q / (100 - q)) := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_decrease_l1284_128405


namespace NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l1284_128481

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_product_and_sum_l1284_128481


namespace NUMINAMATH_CALUDE_not_divides_2007_l1284_128408

theorem not_divides_2007 : ¬(2007 ∣ (2009^3 - 2009)) := by sorry

end NUMINAMATH_CALUDE_not_divides_2007_l1284_128408


namespace NUMINAMATH_CALUDE_box_weight_problem_l1284_128475

theorem box_weight_problem (a b c : ℝ) 
  (ha : a > 40) (hb : b > 40) (hc : c > 40)
  (hab : a + b = 132) (hbc : b + c = 135) (hca : c + a = 137) :
  a + b + c = 202 := by
sorry

end NUMINAMATH_CALUDE_box_weight_problem_l1284_128475


namespace NUMINAMATH_CALUDE_pascal_identity_l1284_128403

theorem pascal_identity (n k : ℕ) (h1 : k ≤ n) (h2 : ¬(n = 0 ∧ k = 0)) : 
  Nat.choose n k = Nat.choose (n - 1) k + Nat.choose (n - 1) (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_pascal_identity_l1284_128403


namespace NUMINAMATH_CALUDE_square_difference_l1284_128421

theorem square_difference : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by sorry

end NUMINAMATH_CALUDE_square_difference_l1284_128421


namespace NUMINAMATH_CALUDE_sum_of_roots_l1284_128491

theorem sum_of_roots (k c x₁ x₂ : ℝ) (h_distinct : x₁ ≠ x₂) 
  (h₁ : 2 * x₁^2 - k * x₁ = 2 * c) (h₂ : 2 * x₂^2 - k * x₂ = 2 * c) : 
  x₁ + x₂ = k / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1284_128491


namespace NUMINAMATH_CALUDE_bottle_caps_per_box_l1284_128420

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) 
  (h1 : total_caps = 316) (h2 : num_boxes = 79) :
  total_caps / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_per_box_l1284_128420


namespace NUMINAMATH_CALUDE_stating_special_words_count_l1284_128488

/-- The number of possible letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering -/
def word_length : ℕ := 6

/-- 
  Calculates the number of six-letter words where:
  1) The first and last letters are the same
  2) The second and fifth letters are the same
-/
def count_special_words : ℕ := alphabet_size ^ 4

/-- 
  Theorem stating that the number of special six-letter words
  (as defined in the problem) is equal to 26^4
-/
theorem special_words_count : 
  count_special_words = 456976 := by sorry

end NUMINAMATH_CALUDE_stating_special_words_count_l1284_128488


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1284_128417

theorem complex_modulus_problem (z : ℂ) (h : Complex.I * z = (1 - 2 * Complex.I)^2) : 
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1284_128417


namespace NUMINAMATH_CALUDE_min_value_of_f_l1284_128494

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1284_128494


namespace NUMINAMATH_CALUDE_mystery_compound_is_nh4_l1284_128483

/-- Represents the atomic weight of an element -/
structure AtomicWeight where
  value : ℝ
  positive : value > 0

/-- Represents a chemical compound -/
structure Compound where
  molecularWeight : ℝ
  nitrogenCount : ℕ
  otherElementCount : ℕ
  otherElementWeight : AtomicWeight

/-- The atomic weight of nitrogen -/
def nitrogenWeight : AtomicWeight :=
  { value := 14.01, positive := by norm_num }

/-- The atomic weight of hydrogen -/
def hydrogenWeight : AtomicWeight :=
  { value := 1.01, positive := by norm_num }

/-- The compound in question -/
def mysteryCompound : Compound :=
  { molecularWeight := 18,
    nitrogenCount := 1,
    otherElementCount := 4,
    otherElementWeight := hydrogenWeight }

/-- Theorem stating that the mystery compound must be NH₄ -/
theorem mystery_compound_is_nh4 :
  ∀ (c : Compound),
    c.molecularWeight = 18 →
    c.nitrogenCount = 1 →
    c.otherElementWeight.value * c.otherElementCount + nitrogenWeight.value = c.molecularWeight →
    c = mysteryCompound :=
  sorry

end NUMINAMATH_CALUDE_mystery_compound_is_nh4_l1284_128483


namespace NUMINAMATH_CALUDE_proposition_2_proposition_3_l1284_128444

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- Define the given conditions
variable (m n a b : Line) (α β : Plane)
variable (h_mn_distinct : m ≠ n)
variable (h_αβ_distinct : α ≠ β)
variable (h_a_perp_α : perpendicularLP a α)
variable (h_b_perp_β : perpendicularLP b β)

-- State the theorems to be proved
theorem proposition_2 
  (h_m_parallel_a : parallel m a)
  (h_n_parallel_b : parallel n b)
  (h_α_perp_β : perpendicularPP α β) :
  perpendicular m n :=
sorry

theorem proposition_3
  (h_m_parallel_α : parallelLP m α)
  (h_n_parallel_b : parallel n b)
  (h_α_parallel_β : parallelPP α β) :
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_proposition_2_proposition_3_l1284_128444


namespace NUMINAMATH_CALUDE_angle_CDB_measure_l1284_128487

-- Define the figure
structure Figure where
  -- Regular pentagon
  pentagon_angle : ℝ
  pentagon_angle_eq : pentagon_angle = 108

  -- Equilateral triangle
  triangle_angle : ℝ
  triangle_angle_eq : triangle_angle = 60

  -- Shared side
  shared_side : ℝ

  -- Angle CDB
  angle_CDB : ℝ

-- Theorem statement
theorem angle_CDB_measure (fig : Figure) : fig.angle_CDB = 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_CDB_measure_l1284_128487


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_evaluate_expression_2_l1284_128431

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by sorry

-- Problem 2
theorem simplify_and_evaluate_expression_2 (x y : ℝ) :
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = x * y^2 + x * y := by sorry

theorem evaluate_expression_2 :
  let x : ℝ := -3
  let y : ℝ := -2
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = -6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_evaluate_expression_2_l1284_128431


namespace NUMINAMATH_CALUDE_bob_jacket_purchase_percentage_l1284_128426

/-- Calculates the percentage of the suggested retail price that Bob paid for a jacket -/
theorem bob_jacket_purchase_percentage (P : ℝ) (P_pos : P > 0) : 
  let marked_price := P * (1 - 0.4)
  let bob_price := marked_price * (1 - 0.4)
  bob_price / P = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_bob_jacket_purchase_percentage_l1284_128426


namespace NUMINAMATH_CALUDE_math_problem_solution_l1284_128452

theorem math_problem_solution :
  ∀ (S₁ S₂ S₃ S₁₂ S₁₃ S₂₃ S₁₂₃ : ℕ),
  S₁ + S₂ + S₃ + S₁₂ + S₁₃ + S₂₃ + S₁₂₃ = 100 →
  S₁ + S₁₂ + S₁₃ + S₁₂₃ = 60 →
  S₂ + S₁₂ + S₂₃ + S₁₂₃ = 60 →
  S₃ + S₁₃ + S₂₃ + S₁₂₃ = 60 →
  (S₁ + S₂ + S₃) - S₁₂₃ = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_math_problem_solution_l1284_128452


namespace NUMINAMATH_CALUDE_cylinder_height_difference_l1284_128415

theorem cylinder_height_difference (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_difference_l1284_128415


namespace NUMINAMATH_CALUDE_sixth_power_to_third_power_l1284_128448

theorem sixth_power_to_third_power (x : ℝ) (h : 728 = x^6 + 1/x^6) : 
  x^3 + 1/x^3 = Real.sqrt 730 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_to_third_power_l1284_128448


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l1284_128435

/-- Two vectors in ℝ² are orthogonal if their dot product is zero -/
def orthogonal (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The first vector -/
def v : ℝ × ℝ := (3, 4)

/-- The second vector -/
def w (x : ℝ) : ℝ × ℝ := (x, -7)

/-- The theorem stating that the vectors are orthogonal when x = 28/3 -/
theorem vectors_orthogonal : orthogonal v (w (28/3)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l1284_128435


namespace NUMINAMATH_CALUDE_hexagram_arrangement_exists_and_unique_l1284_128478

def Hexagram := Fin 7 → Fin 7

def is_valid_arrangement (h : Hexagram) : Prop :=
  (∀ i : Fin 7, ∃! j : Fin 7, h j = i) ∧
  (h 0 + h 1 + h 3 = 12) ∧
  (h 0 + h 2 + h 4 = 12) ∧
  (h 1 + h 2 + h 5 = 12) ∧
  (h 3 + h 4 + h 5 = 12) ∧
  (h 0 + h 6 + h 5 = 12) ∧
  (h 1 + h 6 + h 4 = 12) ∧
  (h 2 + h 6 + h 3 = 12)

theorem hexagram_arrangement_exists_and_unique :
  ∃! h : Hexagram, is_valid_arrangement h :=
sorry

end NUMINAMATH_CALUDE_hexagram_arrangement_exists_and_unique_l1284_128478


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1284_128457

theorem complex_fraction_equality (a b : ℝ) : 
  (1 + I : ℂ) / (1 - I) = a + b * I → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1284_128457


namespace NUMINAMATH_CALUDE_abc_inequality_l1284_128423

theorem abc_inequality (a b c : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) (hc : -1 < c ∧ c < 1) :
  a * b * c + 2 > a + b + c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1284_128423


namespace NUMINAMATH_CALUDE_donuts_distribution_l1284_128446

/-- Calculates the number of donuts each student who likes donuts receives -/
def donuts_per_student (total_donuts : ℕ) (total_students : ℕ) (donut_liking_ratio : ℚ) : ℚ :=
  total_donuts / (total_students * donut_liking_ratio)

/-- Proves that given 4 dozen donuts distributed among 80% of 30 students, 
    each student who likes donuts receives 2 donuts -/
theorem donuts_distribution : 
  donuts_per_student (4 * 12) 30 (4/5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_donuts_distribution_l1284_128446


namespace NUMINAMATH_CALUDE_largest_n_for_inequality_l1284_128461

theorem largest_n_for_inequality (z : ℕ) (h : z = 9) :
  ∃ n : ℕ, (27 ^ z > 3 ^ n ∧ ∀ m : ℕ, m > n → 27 ^ z ≤ 3 ^ m) ∧ n = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_inequality_l1284_128461


namespace NUMINAMATH_CALUDE_largest_difference_l1284_128432

def A : ℕ := 3 * 1003^1004
def B : ℕ := 1003^1004
def C : ℕ := 1002 * 1003^1003
def D : ℕ := 3 * 1003^1003
def E : ℕ := 1003^1003
def F : ℕ := 1003^1002

def P : ℕ := A - B
def Q : ℕ := B - C
def R : ℕ := C - D
def S : ℕ := D - E
def T : ℕ := E - F

theorem largest_difference :
  P > max Q (max R (max S T)) := by sorry

end NUMINAMATH_CALUDE_largest_difference_l1284_128432


namespace NUMINAMATH_CALUDE_aku_birthday_cookies_l1284_128441

/-- Given the number of friends, packages, and cookies per package, 
    calculate the number of cookies each child will eat. -/
def cookies_per_child (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  (packages * cookies_per_package) / (friends + 1)

/-- Theorem stating that under the given conditions, each child will eat 15 cookies. -/
theorem aku_birthday_cookies : 
  cookies_per_child 4 3 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_aku_birthday_cookies_l1284_128441


namespace NUMINAMATH_CALUDE_unique_polynomial_composition_l1284_128498

/-- The polynomial P(x) = x^2 - x satisfies P(P(x)) = (x^2 - x + 1) P(x) and is the only nonconstant polynomial solution. -/
theorem unique_polynomial_composition (x : ℝ) : ∃! P : ℝ → ℝ, 
  (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) ∧ 
  (a ≠ 0 ∨ b ≠ 0) ∧
  (∀ x, P (P x) = (x^2 - x + 1) * P x) ∧
  P = fun x ↦ x^2 - x := by
  sorry

end NUMINAMATH_CALUDE_unique_polynomial_composition_l1284_128498


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l1284_128470

/-- Given a cylinder with volume 72π, prove the volumes of a cone and sphere with related dimensions. -/
theorem cylinder_cone_sphere_volumes (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  π * r^2 * h = 72 * π → 
  (1/3 : ℝ) * π * r^2 * h = 24 * π ∧ 
  (4/3 : ℝ) * π * (h/2)^3 = 12 * r * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l1284_128470
