import Mathlib

namespace NUMINAMATH_CALUDE_range_of_f_l2994_299408

noncomputable def f (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f :
  Set.range f = {-3 * Real.pi / 4, Real.pi / 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2994_299408


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2994_299476

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (2 * y - 3)) = Real.sqrt 6 → y = 14 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2994_299476


namespace NUMINAMATH_CALUDE_business_partnership_problem_l2994_299428

/-- A business partnership problem -/
theorem business_partnership_problem 
  (a_investment : ℕ) 
  (total_duration : ℕ) 
  (b_join_time : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) 
  (h1 : a_investment = 3500)
  (h2 : total_duration = 12)
  (h3 : b_join_time = 8)
  (h4 : profit_ratio_a = 2)
  (h5 : profit_ratio_b = 3) : 
  ∃ (b_investment : ℕ), 
    b_investment = 1575 ∧ 
    (a_investment * total_duration) / (b_investment * (total_duration - b_join_time)) = 
    profit_ratio_a / profit_ratio_b :=
sorry

end NUMINAMATH_CALUDE_business_partnership_problem_l2994_299428


namespace NUMINAMATH_CALUDE_boxes_left_l2994_299436

/-- The number of boxes Jerry started with -/
def initial_boxes : ℕ := 10

/-- The number of boxes Jerry sold -/
def sold_boxes : ℕ := 5

/-- Theorem: Jerry has 5 boxes left after selling -/
theorem boxes_left : initial_boxes - sold_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_left_l2994_299436


namespace NUMINAMATH_CALUDE_vectors_perpendicular_distance_AB_l2994_299409

-- Define the line and parabola
def line (x y : ℝ) : Prop := y = x - 2
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define points A and B as intersections
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define O as the origin
def O : ℝ × ℝ := (0, 0)

-- Vector from O to A
def OA : ℝ × ℝ := (A.1 - O.1, A.2 - O.2)

-- Vector from O to B
def OB : ℝ × ℝ := (B.1 - O.1, B.2 - O.2)

-- Theorem 1: OA ⊥ OB
theorem vectors_perpendicular : OA.1 * OB.1 + OA.2 * OB.2 = 0 := by sorry

-- Theorem 2: |AB| = 2√10
theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_distance_AB_l2994_299409


namespace NUMINAMATH_CALUDE_floor_neg_three_point_seven_l2994_299404

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem: The floor of -3.7 is -4 -/
theorem floor_neg_three_point_seven :
  floor (-3.7) = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_three_point_seven_l2994_299404


namespace NUMINAMATH_CALUDE_sum_unit_digit_not_two_l2994_299457

theorem sum_unit_digit_not_two (n : ℕ) : (n * (n + 1) / 2) % 10 ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_unit_digit_not_two_l2994_299457


namespace NUMINAMATH_CALUDE_big_bottle_volume_proof_l2994_299447

/-- The volume of a big bottle of mango juice in ounces -/
def big_bottle_volume : ℝ := 30

/-- The cost of a big bottle in pesetas -/
def big_bottle_cost : ℝ := 2700

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℝ := 6

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℝ := 600

/-- The amount saved by buying a big bottle instead of equivalent small bottles in pesetas -/
def savings : ℝ := 300

theorem big_bottle_volume_proof :
  big_bottle_volume = 30 :=
by sorry

end NUMINAMATH_CALUDE_big_bottle_volume_proof_l2994_299447


namespace NUMINAMATH_CALUDE_transformation_result_l2994_299445

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def rotate_x_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, -z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180
    |> reflect_yz
    |> reflect_xz
    |> rotate_x_180
    |> reflect_yz

theorem transformation_result :
  transform initial_point = (-2, 2, 2) := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l2994_299445


namespace NUMINAMATH_CALUDE_complement_union_M_N_l2994_299474

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l2994_299474


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l2994_299478

theorem arctan_sum_special_case (a b : ℝ) :
  a = 2/3 →
  (a + 1) * (b + 1) = 3 →
  Real.arctan a + Real.arctan b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l2994_299478


namespace NUMINAMATH_CALUDE_two_integers_sum_squares_product_perfect_square_l2994_299485

/-- There exist two integers less than 10 whose sum of squares plus their product is a perfect square. -/
theorem two_integers_sum_squares_product_perfect_square :
  ∃ a b : ℤ, a < 10 ∧ b < 10 ∧ ∃ k : ℤ, a^2 + b^2 + a*b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_squares_product_perfect_square_l2994_299485


namespace NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l2994_299496

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (cards_per_rank_suit : Nat)

/-- A standard deck has 52 cards, 13 ranks, 4 suits, and 1 card per rank per suit -/
def standard_deck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4
  , cards_per_rank_suit := 1
  }

/-- The probability of drawing a King first and a Queen second from a standard deck -/
def prob_king_queen (d : Deck) : ℚ :=
  (d.suits : ℚ) / d.cards * (d.suits : ℚ) / (d.cards - 1)

/-- Theorem: The probability of drawing a King first and a Queen second from a standard deck is 4/663 -/
theorem prob_king_queen_standard_deck : 
  prob_king_queen standard_deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l2994_299496


namespace NUMINAMATH_CALUDE_car_robot_ratio_l2994_299434

theorem car_robot_ratio : 
  ∀ (tom_michael_robots : ℕ) (bob_robots : ℕ),
    tom_michael_robots = 9 →
    bob_robots = 81 →
    (bob_robots : ℚ) / tom_michael_robots = 9 := by
  sorry

end NUMINAMATH_CALUDE_car_robot_ratio_l2994_299434


namespace NUMINAMATH_CALUDE_anniversary_count_l2994_299469

def founding_year : Nat := 1949
def current_year : Nat := 2015

theorem anniversary_count :
  current_year - founding_year = 66 := by sorry

end NUMINAMATH_CALUDE_anniversary_count_l2994_299469


namespace NUMINAMATH_CALUDE_finite_common_terms_l2994_299402

/-- Two sequences of natural numbers with specific recurrence relations have only finitely many common terms -/
theorem finite_common_terms 
  (a b : ℕ → ℕ) 
  (ha : ∀ n : ℕ, n ≥ 1 → a (n + 1) = n * a n + 1)
  (hb : ∀ n : ℕ, n ≥ 1 → b (n + 1) = n * b n - 1) :
  Set.Finite {n : ℕ | ∃ m : ℕ, a n = b m} :=
sorry

end NUMINAMATH_CALUDE_finite_common_terms_l2994_299402


namespace NUMINAMATH_CALUDE_sample_correlation_strength_theorem_l2994_299420

/-- Sample correlation coefficient -/
def sample_correlation_coefficient (data : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Strength of linear relationship -/
def linear_relationship_strength (r : ℝ) : ℝ :=
  sorry

theorem sample_correlation_strength_theorem (data : Set (ℝ × ℝ)) :
  let r := sample_correlation_coefficient data
  ∀ s : ℝ, s ∈ Set.Icc (-1 : ℝ) 1 →
    linear_relationship_strength r = linear_relationship_strength (abs r) ∧
    (abs r > abs s → linear_relationship_strength r > linear_relationship_strength s) :=
  sorry

end NUMINAMATH_CALUDE_sample_correlation_strength_theorem_l2994_299420


namespace NUMINAMATH_CALUDE_lawn_length_is_80_l2994_299403

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  road_width : ℝ
  travel_cost_per_sqm : ℝ
  total_travel_cost : ℝ

/-- Calculates the area of the roads on the lawn -/
def road_area (l : LawnWithRoads) : ℝ :=
  l.road_width * l.length + l.road_width * (l.width - l.road_width)

/-- Theorem stating the length of the lawn given specific conditions -/
theorem lawn_length_is_80 (l : LawnWithRoads) 
    (h1 : l.width = 60)
    (h2 : l.road_width = 10)
    (h3 : l.travel_cost_per_sqm = 5)
    (h4 : l.total_travel_cost = 6500)
    (h5 : l.total_travel_cost = l.travel_cost_per_sqm * road_area l) :
  l.length = 80 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_80_l2994_299403


namespace NUMINAMATH_CALUDE_distribute_3_4_l2994_299433

/-- The number of ways to distribute n distinct objects into m distinct containers -/
def distribute (n m : ℕ) : ℕ := m^n

/-- Theorem: Distributing 3 distinct objects into 4 distinct containers results in 64 ways -/
theorem distribute_3_4 : distribute 3 4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_distribute_3_4_l2994_299433


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2994_299438

theorem fraction_decomposition (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (x^2 - 2*x + 5) / (x^3 - x) = (-5)/x + (6*x - 2) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2994_299438


namespace NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l2994_299451

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The perimeter of a quadrilateral -/
def Quadrilateral.perimeter (q : Quadrilateral) : ℝ :=
  q.a + q.b + q.c + q.d

/-- Given an equilateral triangle ABC with side length 4 and a right isosceles triangle DBE
    with DB = EB = 1 cut from it, the perimeter of the remaining quadrilateral ACED is 10 + √2 -/
theorem remaining_quadrilateral_perimeter :
  let abc : Triangle := { a := 4, b := 4, c := 4 }
  let dbe : Triangle := { a := 1, b := 1, c := Real.sqrt 2 }
  let aced : Quadrilateral := { a := 4, b := 3, c := Real.sqrt 2, d := 3 }
  aced.perimeter = 10 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l2994_299451


namespace NUMINAMATH_CALUDE_winner_percentage_approx_62_l2994_299458

/-- Represents an election with two candidates -/
structure Election :=
  (total_votes : ℕ)
  (winner_votes : ℕ)
  (margin : ℕ)

/-- Calculates the percentage of votes for the winner -/
def winner_percentage (e : Election) : ℚ :=
  (e.winner_votes : ℚ) / (e.total_votes : ℚ) * 100

/-- Theorem stating the winner's percentage in the given election -/
theorem winner_percentage_approx_62 (e : Election) 
  (h1 : e.winner_votes = 837)
  (h2 : e.margin = 324)
  (h3 : e.total_votes = e.winner_votes + (e.winner_votes - e.margin)) :
  ∃ (p : ℚ), abs (winner_percentage e - p) < 1 ∧ p = 62 := by
  sorry

#eval winner_percentage { total_votes := 1350, winner_votes := 837, margin := 324 }

end NUMINAMATH_CALUDE_winner_percentage_approx_62_l2994_299458


namespace NUMINAMATH_CALUDE_cookies_per_bag_l2994_299465

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 703) (h2 : num_bags = 37) :
  total_cookies / num_bags = 19 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l2994_299465


namespace NUMINAMATH_CALUDE_baker_sales_difference_l2994_299466

theorem baker_sales_difference (cakes_made pastries_made cakes_sold pastries_sold : ℕ) :
  cakes_made = 157 →
  pastries_made = 169 →
  cakes_sold = 158 →
  pastries_sold = 147 →
  cakes_sold - pastries_sold = 11 := by
sorry

end NUMINAMATH_CALUDE_baker_sales_difference_l2994_299466


namespace NUMINAMATH_CALUDE_equation_solutions_l2994_299482

open Real

-- Define the tangent function
noncomputable def tg (x : ℝ) : ℝ := tan x

-- Define the equation
def equation (x : ℝ) : Prop := tg x + tg (2*x) + tg (3*x) + tg (4*x) = 0

-- Define the set of solutions
def solution_set : Set ℝ := {0, π/7.2, π/5, π/3.186, π/2.5, -π/7.2, -π/5, -π/3.186, -π/2.5}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2994_299482


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2994_299429

theorem no_positive_integer_solution (p : ℕ) (x y : ℕ) (hp : p > 3) (hp_prime : Nat.Prime p) (hx : p ∣ x) :
  ¬(x^2 - 1 = y^p) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2994_299429


namespace NUMINAMATH_CALUDE_switch_circuit_probability_l2994_299426

theorem switch_circuit_probability (P_A P_AB : ℝ) 
  (h1 : P_A = 1/2) 
  (h2 : P_AB = 1/5) : 
  P_AB / P_A = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_switch_circuit_probability_l2994_299426


namespace NUMINAMATH_CALUDE_multiplier_is_three_l2994_299491

theorem multiplier_is_three (n : ℝ) (h1 : 3 * n = (20 - n) + 20) (h2 : n = 10) : 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l2994_299491


namespace NUMINAMATH_CALUDE_at_most_one_integer_root_l2994_299453

theorem at_most_one_integer_root (k : ℝ) :
  ∃! (n : ℤ), (n : ℝ)^3 - 24*(n : ℝ) + k = 0 ∨
  ∀ (m : ℤ), (m : ℝ)^3 - 24*(m : ℝ) + k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_integer_root_l2994_299453


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_18_l2994_299425

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 18

theorem first_year_after_2020_with_digit_sum_18 :
  ∀ year : ℕ, isValidYear year → year ≥ 2799 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_digit_sum_18_l2994_299425


namespace NUMINAMATH_CALUDE_max_c_value_max_c_attainable_l2994_299448

theorem max_c_value (c a b : ℕ+) (h1 : c ≤ 2017) 
  (h2 : 2^(a:ℕ) * 5^(b:ℕ) = (a^3 + a^2 + a + 1) * c) : c ≤ 1000 := by
  sorry

theorem max_c_attainable : ∃ (c a b : ℕ+), c = 1000 ∧ c ≤ 2017 ∧ 
  2^(a:ℕ) * 5^(b:ℕ) = (a^3 + a^2 + a + 1) * c := by
  sorry

end NUMINAMATH_CALUDE_max_c_value_max_c_attainable_l2994_299448


namespace NUMINAMATH_CALUDE_binary_101110_equals_octal_56_l2994_299418

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 101110₂ -/
def binary_101110 : List Bool := [false, true, true, true, true, false]

theorem binary_101110_equals_octal_56 :
  decimal_to_octal (binary_to_decimal binary_101110) = [6, 5] :=
by sorry

end NUMINAMATH_CALUDE_binary_101110_equals_octal_56_l2994_299418


namespace NUMINAMATH_CALUDE_correct_calculation_l2994_299493

theorem correct_calculation (x : ℝ) (h : 2 * x = 22) : 20 * x + 3 = 223 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2994_299493


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l2994_299444

theorem lcm_hcf_relation (a b : ℕ) (ha : a = 210) (hlcm : Nat.lcm a b = 2310) :
  Nat.gcd a b = a * b / Nat.lcm a b :=
by sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l2994_299444


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2994_299452

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 4 = 0) : 
  (a^2 - 3) * (a + 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2994_299452


namespace NUMINAMATH_CALUDE_shenzhen_metro_growth_l2994_299492

/-- Represents the passenger growth of Shenzhen Metro Line 11 -/
theorem shenzhen_metro_growth (x : ℝ) : 
  (1.2 : ℝ) * (1 + x)^2 = 1.75 ↔ 
  120 * (1 + x)^2 = 175 := by sorry

#check shenzhen_metro_growth

end NUMINAMATH_CALUDE_shenzhen_metro_growth_l2994_299492


namespace NUMINAMATH_CALUDE_probability_two_red_two_green_l2994_299484

theorem probability_two_red_two_green (total_red : ℕ) (total_green : ℕ) (drawn : ℕ) : 
  total_red = 10 → total_green = 8 → drawn = 4 →
  (Nat.choose total_red 2 * Nat.choose total_green 2) / Nat.choose (total_red + total_green) drawn = 7 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_two_green_l2994_299484


namespace NUMINAMATH_CALUDE_overtime_pay_is_3_20_l2994_299480

/-- Calculates the overtime pay rate given the following conditions:
  * Regular week has 5 working days
  * Regular working hours per day is 8
  * Regular pay rate is 2.40 rupees per hour
  * Total earnings in 4 weeks is 432 rupees
  * Total hours worked in 4 weeks is 175
-/
def overtime_pay_rate (
  regular_days_per_week : ℕ)
  (regular_hours_per_day : ℕ)
  (regular_pay_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours : ℕ) : ℚ :=
by
  sorry

/-- Theorem stating that the overtime pay rate is 3.20 rupees per hour -/
theorem overtime_pay_is_3_20 :
  overtime_pay_rate 5 8 (240/100) 432 175 = 320/100 :=
by
  sorry

end NUMINAMATH_CALUDE_overtime_pay_is_3_20_l2994_299480


namespace NUMINAMATH_CALUDE_min_sum_max_product_l2994_299416

theorem min_sum_max_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * b = 1 → a + b ≥ 2) ∧ (a + b = 1 → a * b ≤ 1/4) := by sorry

end NUMINAMATH_CALUDE_min_sum_max_product_l2994_299416


namespace NUMINAMATH_CALUDE_letter_count_theorem_l2994_299449

structure LetterCounts where
  china : ℕ
  italy : ℕ
  india : ℕ

def january : LetterCounts := { china := 6, italy := 8, india := 4 }
def february : LetterCounts := { china := 9, italy := 5, india := 7 }

def percentageChange (old new : ℕ) : ℚ :=
  (new - old : ℚ) / old * 100

def tripleCount (count : LetterCounts) : LetterCounts :=
  { china := 3 * count.china,
    italy := 3 * count.italy,
    india := 3 * count.india }

def totalLetters (a b c : LetterCounts) : ℕ :=
  a.china + a.italy + a.india +
  b.china + b.italy + b.india +
  c.china + c.italy + c.india

theorem letter_count_theorem :
  percentageChange january.china february.china = 50 ∧
  percentageChange january.italy february.italy = -37.5 ∧
  percentageChange january.india february.india = 75 ∧
  totalLetters january february (tripleCount january) = 93 := by
  sorry

end NUMINAMATH_CALUDE_letter_count_theorem_l2994_299449


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2994_299488

/-- Proves that a 25% reduction in oil price resulting in 5 kg more for Rs. 900 leads to a reduced price of Rs. 45 per kg -/
theorem oil_price_reduction (original_price : ℝ) (original_quantity : ℝ) : 
  (original_quantity * original_price = 900) →
  ((original_quantity + 5) * (0.75 * original_price) = 900) →
  (0.75 * original_price = 45) :=
by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2994_299488


namespace NUMINAMATH_CALUDE_gcd_count_for_product_180_l2994_299405

theorem gcd_count_for_product_180 : 
  ∃ (S : Finset ℕ), 
    (∀ a b : ℕ, a > 0 → b > 0 → Nat.gcd a b * Nat.lcm a b = 180 → 
      Nat.gcd a b ∈ S) ∧ 
    (∀ n ∈ S, ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ Nat.gcd a b * Nat.lcm a b = 180 ∧ 
      Nat.gcd a b = n) ∧
    Finset.card S = 7 :=
by sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_180_l2994_299405


namespace NUMINAMATH_CALUDE_prime_divisibility_l2994_299424

theorem prime_divisibility (p a b : ℕ) : 
  Prime p → 
  p ≠ 3 → 
  a > 0 → 
  b > 0 → 
  p ∣ (a + b) → 
  p^2 ∣ (a^3 + b^3) → 
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2994_299424


namespace NUMINAMATH_CALUDE_friend_payment_is_five_l2994_299468

/-- The cost per person when splitting a restaurant bill -/
def cost_per_person (num_friends : ℕ) (hamburger_price : ℚ) (num_hamburgers : ℕ)
  (fries_price : ℚ) (num_fries : ℕ) (soda_price : ℚ) (num_sodas : ℕ)
  (spaghetti_price : ℚ) (num_spaghetti : ℕ) : ℚ :=
  (hamburger_price * num_hamburgers + fries_price * num_fries +
   soda_price * num_sodas + spaghetti_price * num_spaghetti) / num_friends

/-- Theorem: Each friend pays $5 when splitting the bill equally -/
theorem friend_payment_is_five :
  cost_per_person 5 3 5 (6/5) 4 (1/2) 5 (27/10) 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_payment_is_five_l2994_299468


namespace NUMINAMATH_CALUDE_cos_72_minus_cos_144_eq_zero_l2994_299406

theorem cos_72_minus_cos_144_eq_zero : 
  Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_minus_cos_144_eq_zero_l2994_299406


namespace NUMINAMATH_CALUDE_roots_expression_l2994_299487

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) : 
  (α^2 + p*α - 2 = 0) → 
  (β^2 + p*β - 2 = 0) → 
  (γ^2 + q*γ - 3 = 0) → 
  (δ^2 + q*δ - 3 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) - 2*q + 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l2994_299487


namespace NUMINAMATH_CALUDE_equation_one_solution_l2994_299400

theorem equation_one_solution (x : ℝ) : 3 * x * (x - 1) = 1 - x → x = 1 ∨ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2994_299400


namespace NUMINAMATH_CALUDE_frood_game_theorem_l2994_299479

/-- Sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Points earned from eating n froods -/
def eating_points (n : ℕ) : ℕ := 12 * n

/-- The least number of froods for which dropping them earns more points than eating them -/
def least_froods : ℕ := 24

theorem frood_game_theorem :
  least_froods = 24 ∧
  (∀ n : ℕ, n < least_froods → triangular_number n ≤ eating_points n) ∧
  triangular_number least_froods > eating_points least_froods :=
by sorry

end NUMINAMATH_CALUDE_frood_game_theorem_l2994_299479


namespace NUMINAMATH_CALUDE_stratified_sampling_total_l2994_299490

theorem stratified_sampling_total (sample_size : ℕ) (model_a_count : ℕ) (total_model_b : ℕ) :
  sample_size = 80 →
  model_a_count = 50 →
  total_model_b = 1800 →
  (sample_size - model_a_count) * 60 = total_model_b →
  sample_size * 60 = 4800 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_total_l2994_299490


namespace NUMINAMATH_CALUDE_people_speaking_both_languages_l2994_299435

/-- Given a group of people with specified language abilities, calculate the number who speak both languages. -/
theorem people_speaking_both_languages 
  (total : ℕ) 
  (latin : ℕ) 
  (french : ℕ) 
  (neither : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_neither : neither = 6) :
  latin + french - (total - neither) = 9 := by
sorry

end NUMINAMATH_CALUDE_people_speaking_both_languages_l2994_299435


namespace NUMINAMATH_CALUDE_decreasing_condition_passes_through_origin_l2994_299417

/-- Given linear function y = (2-k)x - k^2 + 4 -/
def y (k x : ℝ) : ℝ := (2 - k) * x - k^2 + 4

/-- y decreases as x increases iff k > 2 -/
theorem decreasing_condition (k : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y k x₁ > y k x₂) ↔ k > 2 :=
sorry

/-- The graph passes through the origin iff k = -2 -/
theorem passes_through_origin (k : ℝ) :
  y k 0 = 0 ↔ k = -2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_condition_passes_through_origin_l2994_299417


namespace NUMINAMATH_CALUDE_weight_measurement_l2994_299407

def weight_set : List Nat := [2, 5, 15]

def heaviest_weight (weights : List Nat) : Nat :=
  weights.sum

def different_weights (weights : List Nat) : Finset Nat :=
  sorry

theorem weight_measurement (weights : List Nat := weight_set) :
  (heaviest_weight weights = 22) ∧
  (different_weights weights).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_weight_measurement_l2994_299407


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2994_299475

theorem complex_expression_simplification :
  let c : ℂ := 3 + 2*I
  let d : ℂ := -2 - I
  3*c + 4*d = 1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2994_299475


namespace NUMINAMATH_CALUDE_egg_container_problem_l2994_299443

theorem egg_container_problem (num_containers : ℕ) 
  (front_pos back_pos left_pos right_pos : ℕ) :
  num_containers = 28 →
  front_pos + back_pos = 34 →
  left_pos + right_pos = 5 →
  (num_containers * ((front_pos + back_pos - 1) * (left_pos + right_pos - 1))) = 3696 :=
by sorry

end NUMINAMATH_CALUDE_egg_container_problem_l2994_299443


namespace NUMINAMATH_CALUDE_quadratic_functions_equality_l2994_299467

/-- Given a quadratic function f(x) = x² + bx + 8 with b ≠ 0 and two distinct real roots x₁ and x₂,
    and a quadratic function g(x) with quadratic coefficient 1 and roots x₁ + 1/x₂ and x₂ + 1/x₁,
    prove that if g(1) = f(1), then g(1) = -8. -/
theorem quadratic_functions_equality (b : ℝ) (x₁ x₂ : ℝ) :
  b ≠ 0 →
  x₁ ≠ x₂ →
  (∀ x, x^2 + b*x + 8 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ c d : ℝ, ∀ x, (x - (x₁ + 1/x₂)) * (x - (x₂ + 1/x₁)) = x^2 + c*x + d) →
  (1^2 + b*1 + 8 = 1^2 + c*1 + d) →
  1^2 + c*1 + d = -8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_functions_equality_l2994_299467


namespace NUMINAMATH_CALUDE_roll_less_than_5_most_likely_l2994_299494

-- Define the probability of an event on a fair die
def prob (n : ℕ) : ℚ := n / 6

-- Define the events
def roll_6 : ℚ := prob 1
def roll_more_than_4 : ℚ := prob 2
def roll_less_than_4 : ℚ := prob 3
def roll_less_than_5 : ℚ := prob 4

-- Theorem statement
theorem roll_less_than_5_most_likely :
  roll_less_than_5 > roll_6 ∧
  roll_less_than_5 > roll_more_than_4 ∧
  roll_less_than_5 > roll_less_than_4 :=
sorry

end NUMINAMATH_CALUDE_roll_less_than_5_most_likely_l2994_299494


namespace NUMINAMATH_CALUDE_popcorn_selling_price_l2994_299450

/-- Calculate the selling price per bag of popcorn -/
theorem popcorn_selling_price 
  (cost_price : ℝ) 
  (num_bags : ℕ) 
  (total_profit : ℝ) 
  (h1 : cost_price = 4)
  (h2 : num_bags = 30)
  (h3 : total_profit = 120) : 
  (cost_price * num_bags + total_profit) / num_bags = 8 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_selling_price_l2994_299450


namespace NUMINAMATH_CALUDE_parabola_decreasing_for_positive_x_l2994_299473

theorem parabola_decreasing_for_positive_x (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) :
  -x₂^2 + 3 < -x₁^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_decreasing_for_positive_x_l2994_299473


namespace NUMINAMATH_CALUDE_complex_cube_sum_ratio_l2994_299495

theorem complex_cube_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 3*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 48 := by
sorry

end NUMINAMATH_CALUDE_complex_cube_sum_ratio_l2994_299495


namespace NUMINAMATH_CALUDE_insurance_covers_80_percent_l2994_299446

/-- Represents the medication and insurance scenario for Tom --/
structure MedicationScenario where
  pills_per_day : ℕ
  doctor_visits_per_year : ℕ
  cost_per_visit : ℕ
  cost_per_pill : ℕ
  total_annual_payment : ℕ

/-- Calculates the percentage of medication cost covered by insurance --/
def insurance_coverage_percentage (scenario : MedicationScenario) : ℚ :=
  let total_pills := scenario.pills_per_day * 365
  let medication_cost := total_pills * scenario.cost_per_pill
  let doctor_cost := scenario.doctor_visits_per_year * scenario.cost_per_visit
  let total_cost := medication_cost + doctor_cost
  let insurance_coverage := total_cost - scenario.total_annual_payment
  (insurance_coverage : ℚ) / (medication_cost : ℚ) * 100

/-- Tom's specific medication scenario --/
def tom_scenario : MedicationScenario :=
  { pills_per_day := 2
  , doctor_visits_per_year := 2
  , cost_per_visit := 400
  , cost_per_pill := 5
  , total_annual_payment := 1530 }

/-- Theorem stating that the insurance covers 80% of Tom's medication cost --/
theorem insurance_covers_80_percent :
  insurance_coverage_percentage tom_scenario = 80 := by
  sorry

end NUMINAMATH_CALUDE_insurance_covers_80_percent_l2994_299446


namespace NUMINAMATH_CALUDE_positive_divisors_of_90_l2994_299419

theorem positive_divisors_of_90 : Finset.card (Nat.divisors 90) = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_divisors_of_90_l2994_299419


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2994_299472

-- Problem 1
theorem problem_1 (m : ℝ) :
  let p := ∀ x, |x| + |x - 1| > m → x ∈ Set.univ
  let q := ∀ x y, x < y → (-(5 - 2*m))^x > (-(5 - 2*m))^y
  (p ∨ q) ∧ ¬(p ∧ q) → 1 ≤ m ∧ m < 2 := by sorry

-- Problem 2
theorem problem_2 (a b c d : ℝ) :
  a > b ∧ b > c ∧ c > d ∧ d > 0 ∧ a + d = b + c →
  Real.sqrt d + Real.sqrt a < Real.sqrt b + Real.sqrt c := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2994_299472


namespace NUMINAMATH_CALUDE_unique_s_value_l2994_299411

theorem unique_s_value : ∃! s : ℝ, ∀ x : ℝ, 
  (3 * x^2 - 4 * x + 8) * (5 * x^2 + s * x + 15) = 
  15 * x^4 - 29 * x^3 + 87 * x^2 - 60 * x + 120 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_s_value_l2994_299411


namespace NUMINAMATH_CALUDE_sum_plus_difference_l2994_299456

theorem sum_plus_difference (a b c : ℝ) (h : c = a + b + 5.1) : c = 48.9 :=
  by sorry

#check sum_plus_difference 20.2 33.8 48.9

end NUMINAMATH_CALUDE_sum_plus_difference_l2994_299456


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2994_299440

theorem complex_equation_solution (z : ℂ) :
  Complex.I * z = 4 + 3 * Complex.I → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2994_299440


namespace NUMINAMATH_CALUDE_left_placement_equals_100a_plus_b_l2994_299463

/-- A single-digit number is a natural number from 0 to 9 -/
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A two-digit number is a natural number from 10 to 99 -/
def TwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The three-digit number formed by placing a to the left of b -/
def LeftPlacement (a b : ℕ) : ℕ := 100 * a + b

theorem left_placement_equals_100a_plus_b (a b : ℕ) 
  (ha : SingleDigit a) (hb : TwoDigit b) : 
  LeftPlacement a b = 100 * a + b := by
  sorry

end NUMINAMATH_CALUDE_left_placement_equals_100a_plus_b_l2994_299463


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2994_299427

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l2994_299427


namespace NUMINAMATH_CALUDE_average_lunchmeat_price_l2994_299431

def joan_bologna_weight : ℝ := 3
def joan_bologna_price : ℝ := 2.80
def grant_pastrami_weight : ℝ := 2
def grant_pastrami_price : ℝ := 1.80

theorem average_lunchmeat_price :
  let total_weight := joan_bologna_weight + grant_pastrami_weight
  let total_cost := joan_bologna_weight * joan_bologna_price + grant_pastrami_weight * grant_pastrami_price
  total_cost / total_weight = 2.40 := by
sorry

end NUMINAMATH_CALUDE_average_lunchmeat_price_l2994_299431


namespace NUMINAMATH_CALUDE_bouquet_cost_45_l2994_299477

/-- The cost of a bouquet of lilies, given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  30 * (n : ℚ) / 18

theorem bouquet_cost_45 : bouquet_cost 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_45_l2994_299477


namespace NUMINAMATH_CALUDE_no_integer_points_between_A_and_B_l2994_299413

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- The line passing through points A(2,3) and B(50,305) -/
def line_AB (p : IntPoint) : Prop :=
  (p.y - 3) * (50 - 2) = (p.x - 2) * (305 - 3)

/-- A point is strictly between A and B -/
def between_A_and_B (p : IntPoint) : Prop :=
  2 < p.x ∧ p.x < 50

theorem no_integer_points_between_A_and_B :
  ¬ ∃ p : IntPoint, line_AB p ∧ between_A_and_B p :=
sorry

end NUMINAMATH_CALUDE_no_integer_points_between_A_and_B_l2994_299413


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l2994_299454

/-- An isosceles right triangle with given properties -/
structure IsoscelesRightTriangle where
  -- The length of the equal sides
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- Condition that the area is equal to half the square of the leg
  area_eq : area = (1/2) * leg^2

/-- The main theorem -/
theorem isosceles_right_triangle_hypotenuse_length 
  (t : IsoscelesRightTriangle) (h : t.area = 25) : 
  t.leg * Real.sqrt 2 = 10 := by
  sorry

#check isosceles_right_triangle_hypotenuse_length

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_length_l2994_299454


namespace NUMINAMATH_CALUDE_v_3003_equals_3_l2994_299483

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- Default case for inputs not in the table

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| n + 1 => g (v n)

-- Theorem to prove
theorem v_3003_equals_3 : v 3003 = 3 := by
  sorry

end NUMINAMATH_CALUDE_v_3003_equals_3_l2994_299483


namespace NUMINAMATH_CALUDE_magazine_selection_count_l2994_299430

def total_magazines : ℕ := 8
def literature_magazines : ℕ := 3
def math_magazines : ℕ := 5
def magazines_to_select : ℕ := 3

theorem magazine_selection_count :
  (Nat.choose math_magazines magazines_to_select) +
  (Nat.choose math_magazines (magazines_to_select - 1)) +
  (Nat.choose math_magazines (magazines_to_select - 2)) +
  (if literature_magazines ≥ magazines_to_select then 1 else 0) = 26 := by
  sorry

end NUMINAMATH_CALUDE_magazine_selection_count_l2994_299430


namespace NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l2994_299498

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds : 
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧ 
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l2994_299498


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2994_299410

/-- The surface area of a cylinder with base radius 1 and volume 2π is 6π. -/
theorem cylinder_surface_area (r h : ℝ) : 
  r = 1 → π * r^2 * h = 2*π → 2*π*r*h + 2*π*r^2 = 6*π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2994_299410


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2994_299432

theorem sufficient_not_necessary (x y : ℝ) :
  (x < y ∧ y < 0 → x^2 > y^2) ∧
  ∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2994_299432


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l2994_299401

/-- Given an ellipse x²/m² + y²/n² = 1 passing through point P(a, b),
    prove that the minimum value of m + n is (a²/³ + b²/³)¹/³ -/
theorem ellipse_min_sum (a b m n : ℝ) (hm : m > 0) (hn : n > 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hab : abs a ≠ abs b)
  (h_ellipse : a^2 / m^2 + b^2 / n^2 = 1) :
  ∃ (min_sum : ℝ), min_sum = (a^(2/3) + b^(2/3))^(1/3) ∧
    ∀ (m' n' : ℝ), m' > 0 → n' > 0 → a^2 / m'^2 + b^2 / n'^2 = 1 →
      m' + n' ≥ min_sum :=
sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l2994_299401


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l2994_299459

theorem solution_set_reciprocal_inequality (x : ℝ) : 
  (1 / x > 2) ↔ (0 < x ∧ x < 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l2994_299459


namespace NUMINAMATH_CALUDE_water_current_speed_l2994_299471

/-- Given a swimmer's speed in still water and their performance against a current,
    calculate the speed of the water current. -/
theorem water_current_speed 
  (still_water_speed : ℝ) 
  (distance_against_current : ℝ) 
  (time_against_current : ℝ) 
  (h1 : still_water_speed = 4)
  (h2 : distance_against_current = 6)
  (h3 : time_against_current = 2)
  : ℝ :=
  by
  -- The speed of the water current is 1 km/h
  sorry

#check water_current_speed

end NUMINAMATH_CALUDE_water_current_speed_l2994_299471


namespace NUMINAMATH_CALUDE_solution_set_for_f_l2994_299422

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem solution_set_for_f
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a (2/a) > f a (3/a)) :
  ∀ x, f a (1 - 1/x) > 1 ↔ 1 < x ∧ x < 1/(1-a) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_for_f_l2994_299422


namespace NUMINAMATH_CALUDE_systematic_sample_result_l2994_299461

/-- Systematic sampling function -/
def systematicSample (populationSize sampleSize : ℕ) (firstSelected : ℕ) : ℕ → ℕ :=
  fun n => firstSelected + (n - 1) * (populationSize / sampleSize)

theorem systematic_sample_result 
  (populationSize sampleSize firstSelected : ℕ) 
  (h1 : populationSize = 800) 
  (h2 : sampleSize = 50) 
  (h3 : firstSelected = 11) 
  (h4 : firstSelected ≤ 16) :
  ∃ n : ℕ, 33 ≤ n ∧ n ≤ 48 ∧ systematicSample populationSize sampleSize firstSelected n = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_result_l2994_299461


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2994_299421

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2994_299421


namespace NUMINAMATH_CALUDE_probability_of_selecting_specific_pair_l2994_299464

/-- Given a box of shoes with the following properties:
    - There are 20 pairs of shoes (40 shoes in total)
    - Each pair has a unique design
    Prove that the probability of randomly selecting both shoes
    from a specific pair (pair A) is 1/780. -/
theorem probability_of_selecting_specific_pair (total_shoes : Nat) (total_pairs : Nat)
    (h1 : total_shoes = 40)
    (h2 : total_pairs = 20)
    (h3 : total_shoes = 2 * total_pairs) :
  (1 : ℚ) / total_shoes * (1 : ℚ) / (total_shoes - 1) = 1 / 780 := by
  sorry

#check probability_of_selecting_specific_pair

end NUMINAMATH_CALUDE_probability_of_selecting_specific_pair_l2994_299464


namespace NUMINAMATH_CALUDE_sulfuric_acid_mixture_l2994_299442

/-- Proves that mixing 42 liters of 2% sulfuric acid solution with 18 liters of 12% sulfuric acid solution results in a 60-liter solution containing 5% sulfuric acid. -/
theorem sulfuric_acid_mixture :
  let solution1_volume : ℝ := 42
  let solution1_concentration : ℝ := 0.02
  let solution2_volume : ℝ := 18
  let solution2_concentration : ℝ := 0.12
  let total_volume : ℝ := solution1_volume + solution2_volume
  let total_acid : ℝ := solution1_volume * solution1_concentration + solution2_volume * solution2_concentration
  let final_concentration : ℝ := total_acid / total_volume
  total_volume = 60 ∧ final_concentration = 0.05 := by
  sorry

#check sulfuric_acid_mixture

end NUMINAMATH_CALUDE_sulfuric_acid_mixture_l2994_299442


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l2994_299460

-- Define the parabola
def Parabola := ℝ → ℝ

-- Define the properties of the parabola
axiom parabola_increasing (f : Parabola) : ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂
axiom parabola_decreasing (f : Parabola) : ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂

-- Define the points on the parabola
def A (f : Parabola) := f (-2)
def B (f : Parabola) := f 1
def C (f : Parabola) := f 3

-- State the theorem
theorem parabola_point_ordering (f : Parabola) : B f < C f ∧ C f < A f := by sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l2994_299460


namespace NUMINAMATH_CALUDE_rosa_phone_calls_l2994_299455

theorem rosa_phone_calls (total_pages : ℝ) (pages_this_week : ℝ) 
  (h1 : total_pages = 18.8) 
  (h2 : pages_this_week = 8.6) : 
  total_pages - pages_this_week = 10.2 := by
sorry

end NUMINAMATH_CALUDE_rosa_phone_calls_l2994_299455


namespace NUMINAMATH_CALUDE_gcd_diff_is_square_l2994_299437

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k^2 := by sorry

end NUMINAMATH_CALUDE_gcd_diff_is_square_l2994_299437


namespace NUMINAMATH_CALUDE_inverse_proportion_l2994_299499

/-- Given that x is inversely proportional to y, prove that when x = 5 for y = -4, 
    then x = 2 for y = -10 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) 
    (h1 : x * y = k)  -- x is inversely proportional to y
    (h2 : 5 * (-4) = k)  -- x = 5 when y = -4
    : x = 2 ∧ y = -10 → x * y = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2994_299499


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l2994_299497

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l2994_299497


namespace NUMINAMATH_CALUDE_chosen_number_proof_l2994_299481

theorem chosen_number_proof :
  ∀ x : ℝ, (x / 6 - 15 = 5) → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l2994_299481


namespace NUMINAMATH_CALUDE_n_to_b_equals_sixteen_l2994_299439

-- Define n and b
def n : ℝ := 2 ^ (1 / 4)
def b : ℝ := 16.000000000000004

-- Theorem statement
theorem n_to_b_equals_sixteen : n ^ b = 16 := by
  sorry

end NUMINAMATH_CALUDE_n_to_b_equals_sixteen_l2994_299439


namespace NUMINAMATH_CALUDE_project_completion_theorem_l2994_299486

theorem project_completion_theorem (a b c x y z : ℝ) 
  (ha : a / x = 1 / y + 1 / z)
  (hb : b / y = 1 / x + 1 / z)
  (hc : c / z = 1 / x + 1 / y)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1 := by
sorry


end NUMINAMATH_CALUDE_project_completion_theorem_l2994_299486


namespace NUMINAMATH_CALUDE_repeating_base_representation_l2994_299414

theorem repeating_base_representation (k : ℕ) : 
  k > 0 ∧ (12 : ℚ) / 65 = (3 * k + 1 : ℚ) / (k^2 - 1) → k = 17 :=
by sorry

end NUMINAMATH_CALUDE_repeating_base_representation_l2994_299414


namespace NUMINAMATH_CALUDE_anns_shopping_trip_l2994_299462

/-- Calculates the cost of each top in Ann's shopping trip -/
def cost_per_top (total_spent : ℚ) (num_shorts : ℕ) (price_shorts : ℚ) 
  (num_shoes : ℕ) (price_shoes : ℚ) (num_tops : ℕ) : ℚ :=
  let total_shorts := num_shorts * price_shorts
  let total_shoes := num_shoes * price_shoes
  let total_tops := total_spent - total_shorts - total_shoes
  total_tops / num_tops

/-- Proves that the cost per top is $5 given the conditions of Ann's shopping trip -/
theorem anns_shopping_trip : 
  cost_per_top 75 5 7 2 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_anns_shopping_trip_l2994_299462


namespace NUMINAMATH_CALUDE_max_value_fraction_l2994_299412

theorem max_value_fraction (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (k * x + y)^2 / (x^2 + k * y^2) ≤ k + 1 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (k * x + y)^2 / (x^2 + k * y^2) = k + 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2994_299412


namespace NUMINAMATH_CALUDE_prime_factor_count_l2994_299423

theorem prime_factor_count (p : ℕ) : 
  (26 : ℕ) + p + (2 : ℕ) = (33 : ℕ) → p = (5 : ℕ) := by
  sorry

#check prime_factor_count

end NUMINAMATH_CALUDE_prime_factor_count_l2994_299423


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2994_299441

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 24 →
  b = 18 →
  c^2 = a^2 + b^2 →
  c = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2994_299441


namespace NUMINAMATH_CALUDE_correct_swap_l2994_299415

def swap_values (m n : ℕ) : ℕ × ℕ := 
  let s := m
  let m' := n
  let n' := s
  (m', n')

theorem correct_swap : 
  ∀ (m n : ℕ), swap_values m n = (n, m) := by
  sorry

end NUMINAMATH_CALUDE_correct_swap_l2994_299415


namespace NUMINAMATH_CALUDE_a_n_bounds_l2994_299489

variable (n : ℕ+)

noncomputable def a : ℕ → ℚ
  | 0 => 1/2
  | k + 1 => a k + (1/n) * (a k)^2

theorem a_n_bounds : 1 - 1/n < a n n ∧ a n n < 1 := by sorry

end NUMINAMATH_CALUDE_a_n_bounds_l2994_299489


namespace NUMINAMATH_CALUDE_repeating_block_11_13_l2994_299470

def decimal_expansion (n d : ℕ) : List ℕ :=
  sorry

def is_repeating_block (l : List ℕ) (block : List ℕ) : Prop :=
  sorry

theorem repeating_block_11_13 :
  ∃ (block : List ℕ),
    block.length = 6 ∧
    is_repeating_block (decimal_expansion 11 13) block ∧
    ∀ (smaller_block : List ℕ),
      smaller_block.length < 6 →
      ¬ is_repeating_block (decimal_expansion 11 13) smaller_block :=
by sorry

end NUMINAMATH_CALUDE_repeating_block_11_13_l2994_299470
