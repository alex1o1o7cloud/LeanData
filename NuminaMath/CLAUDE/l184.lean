import Mathlib

namespace NUMINAMATH_CALUDE_wendy_run_distance_l184_18409

/-- The distance Wendy walked in miles -/
def walked_distance : ℝ := 9.166666666666666

/-- The additional distance Wendy ran compared to what she walked in miles -/
def additional_run_distance : ℝ := 10.666666666666666

/-- The total distance Wendy ran in miles -/
def total_run_distance : ℝ := walked_distance + additional_run_distance

theorem wendy_run_distance : total_run_distance = 19.833333333333332 := by
  sorry

end NUMINAMATH_CALUDE_wendy_run_distance_l184_18409


namespace NUMINAMATH_CALUDE_investment_strategy_optimal_l184_18478

/-- Represents the maximum interest earned from a two-rate investment strategy --/
def max_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (max_at_rate1 : ℝ) : ℝ :=
  rate1 * max_at_rate1 + rate2 * (total_investment - max_at_rate1)

/-- Theorem stating the maximum interest earned under given conditions --/
theorem investment_strategy_optimal (total_investment : ℝ) (rate1 rate2 : ℝ) (max_at_rate1 : ℝ)
    (h1 : total_investment = 25000)
    (h2 : rate1 = 0.07)
    (h3 : rate2 = 0.12)
    (h4 : max_at_rate1 = 11000) :
    max_interest total_investment rate1 rate2 max_at_rate1 = 2450 := by
  sorry

#eval max_interest 25000 0.07 0.12 11000

end NUMINAMATH_CALUDE_investment_strategy_optimal_l184_18478


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l184_18416

theorem fraction_to_zero_power (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a / b : ℚ) ^ (0 : ℤ) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l184_18416


namespace NUMINAMATH_CALUDE_no_lions_present_l184_18423

theorem no_lions_present (total : ℕ) (tigers monkeys : ℕ) : 
  tigers = 7 * (total - tigers) →
  monkeys = (total - monkeys) / 7 →
  tigers + monkeys = total →
  ∀ other : ℕ, other ≤ total - (tigers + monkeys) → other = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_lions_present_l184_18423


namespace NUMINAMATH_CALUDE_total_unique_photos_l184_18400

/-- Represents the number of photographs taken by Octavia -/
def octavia_photos : ℕ := 36

/-- Represents the number of Octavia's photographs framed by Jack -/
def jack_framed_octavia : ℕ := 24

/-- Represents the number of photographs framed by Jack that were taken by other photographers -/
def jack_framed_others : ℕ := 12

/-- Theorem stating the total number of unique photographs either framed by Jack or taken by Octavia -/
theorem total_unique_photos : 
  (octavia_photos + (jack_framed_octavia + jack_framed_others) - jack_framed_octavia) = 48 := by
  sorry


end NUMINAMATH_CALUDE_total_unique_photos_l184_18400


namespace NUMINAMATH_CALUDE_weights_division_condition_l184_18460

/-- A function that checks if a set of weights from 1 to n grams can be divided into three equal mass piles -/
def canDivideWeights (n : ℕ) : Prop :=
  ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧
                         a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
                         (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)

/-- The theorem stating the condition for when weights can be divided into three equal mass piles -/
theorem weights_division_condition (n : ℕ) (h : n > 3) :
  canDivideWeights n ↔ n % 3 = 0 ∨ n % 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_weights_division_condition_l184_18460


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l184_18450

theorem tank_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l184_18450


namespace NUMINAMATH_CALUDE_power_of_negative_product_l184_18411

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l184_18411


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l184_18419

theorem mod_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ 47635 % 21 = n ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l184_18419


namespace NUMINAMATH_CALUDE_prime_relative_frequency_l184_18420

/-- The number of natural numbers considered -/
def total_numbers : ℕ := 4000

/-- The number of prime numbers among the first 4000 natural numbers -/
def prime_count : ℕ := 551

/-- The relative frequency of prime numbers among the first 4000 natural numbers -/
def relative_frequency : ℚ := prime_count / total_numbers

theorem prime_relative_frequency :
  relative_frequency = 551 / 4000 :=
by sorry

end NUMINAMATH_CALUDE_prime_relative_frequency_l184_18420


namespace NUMINAMATH_CALUDE_abs_neg_a_eq_five_implies_a_eq_plus_minus_five_l184_18403

theorem abs_neg_a_eq_five_implies_a_eq_plus_minus_five (a : ℝ) :
  |(-a)| = 5 → (a = 5 ∨ a = -5) := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_a_eq_five_implies_a_eq_plus_minus_five_l184_18403


namespace NUMINAMATH_CALUDE_plane_equation_proof_l184_18447

def plane_equation (w : ℝ × ℝ × ℝ) (s t : ℝ) : Prop :=
  w = (2 + 2*s - 3*t, 4 - 2*s, 1 - s + 3*t)

theorem plane_equation_proof :
  ∃ (A B C D : ℤ),
    (∀ x y z : ℝ, (∃ s t : ℝ, plane_equation (x, y, z) s t) ↔ A * x + B * y + C * z + D = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 2 ∧ B = -1 ∧ C = 2 ∧ D = -2 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l184_18447


namespace NUMINAMATH_CALUDE_prob_even_sum_is_half_l184_18456

/-- Represents a wheel with a given number of sections and even sections -/
structure Wheel where
  total_sections : ℕ
  even_sections : ℕ

/-- Calculates the probability of getting an even sum when spinning two wheels -/
def prob_even_sum (wheel1 wheel2 : Wheel) : ℚ :=
  let p1_even := wheel1.even_sections / wheel1.total_sections
  let p2_even := wheel2.even_sections / wheel2.total_sections
  p1_even * p2_even + (1 - p1_even) * (1 - p2_even)

/-- The main theorem stating that the probability of getting an even sum
    when spinning the two given wheels is 1/2 -/
theorem prob_even_sum_is_half :
  let wheel1 : Wheel := { total_sections := 5, even_sections := 2 }
  let wheel2 : Wheel := { total_sections := 4, even_sections := 2 }
  prob_even_sum wheel1 wheel2 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_even_sum_is_half_l184_18456


namespace NUMINAMATH_CALUDE_fraction_decimal_difference_l184_18493

theorem fraction_decimal_difference : 
  2/3 - 0.66666667 = 1/(3 * 10^8) := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_difference_l184_18493


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l184_18477

theorem sqrt_product_equality : 
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l184_18477


namespace NUMINAMATH_CALUDE_larger_interior_angle_measure_l184_18430

/-- A circular pavilion constructed with congruent isosceles trapezoids -/
structure CircularPavilion where
  /-- The number of trapezoids in the pavilion -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of a typical trapezoid in degrees -/
  larger_interior_angle : ℝ
  /-- Assertion that the bottom sides of the two end trapezoids are horizontal -/
  horizontal_bottom_sides : Prop

/-- Theorem stating the measure of the larger interior angle in a circular pavilion with 12 trapezoids -/
theorem larger_interior_angle_measure (p : CircularPavilion) 
  (h1 : p.num_trapezoids = 12)
  (h2 : p.horizontal_bottom_sides) :
  p.larger_interior_angle = 97.5 := by
  sorry

end NUMINAMATH_CALUDE_larger_interior_angle_measure_l184_18430


namespace NUMINAMATH_CALUDE_f_inequality_l184_18422

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem f_inequality (a b c : ℝ) (h1 : a > 0) (h2 : ∀ x, f a b c (1 - x) = f a b c (1 + x)) :
  ∀ x, f a b c (2^x) > f a b c (3^x) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l184_18422


namespace NUMINAMATH_CALUDE_inequality_solution_l184_18428

theorem inequality_solution : ∀ x : ℕ+, 
  (2 * x.val + 9 ≥ 3 * (x.val + 2)) ↔ (x.val = 1 ∨ x.val = 2 ∨ x.val = 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l184_18428


namespace NUMINAMATH_CALUDE_sandys_initial_fish_count_l184_18486

theorem sandys_initial_fish_count (initial_fish current_fish bought_fish : ℕ) : 
  current_fish = initial_fish + bought_fish →
  current_fish = 32 →
  bought_fish = 6 →
  initial_fish = 26 := by
sorry

end NUMINAMATH_CALUDE_sandys_initial_fish_count_l184_18486


namespace NUMINAMATH_CALUDE_perimeter_equality_l184_18471

/-- The perimeter of a rectangle -/
def rectangle_perimeter (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height)

/-- The perimeter of a figure composed of two rectangles sharing one edge -/
def composite_perimeter (width1 : ℕ) (height1 : ℕ) (width2 : ℕ) (height2 : ℕ) (shared_edge : ℕ) : ℕ :=
  rectangle_perimeter width1 height1 + rectangle_perimeter width2 height2 - 2 * shared_edge

theorem perimeter_equality :
  rectangle_perimeter 4 3 = composite_perimeter 2 3 3 2 3 := by
  sorry

#eval rectangle_perimeter 4 3
#eval composite_perimeter 2 3 3 2 3

end NUMINAMATH_CALUDE_perimeter_equality_l184_18471


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_count_l184_18466

/-- Represents the number of ferris wheel rides -/
def ferris_wheel_rides : ℕ := sorry

/-- Represents the number of bumper car rides -/
def bumper_car_rides : ℕ := 4

/-- Represents the cost of each ride in tickets -/
def cost_per_ride : ℕ := 7

/-- Represents the total number of tickets used -/
def total_tickets : ℕ := 63

/-- Theorem stating that the number of ferris wheel rides is 5 -/
theorem ferris_wheel_rides_count : ferris_wheel_rides = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_count_l184_18466


namespace NUMINAMATH_CALUDE_semicircle_radius_l184_18487

theorem semicircle_radius (width length : ℝ) (h1 : width = 3) (h2 : length = 8) :
  let rectangle_area := width * length
  let semicircle_radius := Real.sqrt (2 * rectangle_area / Real.pi)
  semicircle_radius = Real.sqrt (48 / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l184_18487


namespace NUMINAMATH_CALUDE_factorization_of_3x2_minus_12y2_l184_18468

theorem factorization_of_3x2_minus_12y2 (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_3x2_minus_12y2_l184_18468


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l184_18441

theorem quadratic_equation_solution (x : ℝ) : 9 * x^2 - 4 = 0 ↔ x = 2/3 ∨ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l184_18441


namespace NUMINAMATH_CALUDE_roses_difference_l184_18445

theorem roses_difference (initial_roses : ℕ) (thrown_away : ℕ) (final_roses : ℕ)
  (h1 : initial_roses = 21)
  (h2 : thrown_away = 34)
  (h3 : final_roses = 15) :
  thrown_away - (thrown_away + final_roses - initial_roses) = 6 := by
  sorry

end NUMINAMATH_CALUDE_roses_difference_l184_18445


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l184_18479

theorem opposite_of_negative_seven : 
  (-(- 7 : ℤ)) = (7 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l184_18479


namespace NUMINAMATH_CALUDE_exam_scores_l184_18494

theorem exam_scores (total_items : Nat) (marion_score : Nat) (marion_ella_relation : Nat) :
  total_items = 40 →
  marion_score = 24 →
  marion_score = marion_ella_relation + 6 →
  ∃ (ella_score : Nat),
    marion_score = ella_score / 2 + 6 ∧
    ella_score = total_items - 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_l184_18494


namespace NUMINAMATH_CALUDE_complex_product_real_imag_equal_l184_18434

theorem complex_product_real_imag_equal (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_equal_l184_18434


namespace NUMINAMATH_CALUDE_correct_categorization_l184_18454

-- Define the given numbers
def numbers : List ℚ := [-2/9, -9, -301, -314/100, 2004, 0, 22/7]

-- Define the sets
def fractions : Set ℚ := {x | x ∈ numbers ∧ x ≠ 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def negative_fractions : Set ℚ := {x | x ∈ fractions ∧ x < 0}
def integers : Set ℚ := {x | x ∈ numbers ∧ ∃ (n : ℤ), x = n}
def positive_integers : Set ℚ := {x | x ∈ integers ∧ x > 0}
def positive_rationals : Set ℚ := {x | x ∈ numbers ∧ x > 0 ∧ ∃ (a b : ℤ), b > 0 ∧ x = a / b}

-- State the theorem
theorem correct_categorization :
  fractions = {-2/9, 22/7} ∧
  negative_fractions = {-2/9} ∧
  integers = {-9, -301, 2004, 0} ∧
  positive_integers = {2004} ∧
  positive_rationals = {2004, 22/7} :=
sorry

end NUMINAMATH_CALUDE_correct_categorization_l184_18454


namespace NUMINAMATH_CALUDE_equation_solution_l184_18444

theorem equation_solution : ∃! x : ℚ, (5 * x / (x + 3) - 3 / (x + 3) = 1 / (x + 3)) ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l184_18444


namespace NUMINAMATH_CALUDE_f_composition_of_two_l184_18413

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x - 1

-- State the theorem
theorem f_composition_of_two : f (f 2) = 1481 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_two_l184_18413


namespace NUMINAMATH_CALUDE_alissa_presents_l184_18474

/-- Given that Ethan has 31 presents and Alissa has 22 more presents than Ethan,
    prove that Alissa has 53 presents. -/
theorem alissa_presents (ethan_presents : ℕ) (alissa_extra : ℕ) :
  ethan_presents = 31 → alissa_extra = 22 → ethan_presents + alissa_extra = 53 :=
by sorry

end NUMINAMATH_CALUDE_alissa_presents_l184_18474


namespace NUMINAMATH_CALUDE_total_tickets_sold_l184_18443

/-- Proves the total number of tickets sold given ticket prices, total receipts, and number of senior citizen tickets --/
theorem total_tickets_sold (adult_price senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  adult_price = 25 →
  senior_price = 15 →
  total_receipts = 9745 →
  senior_tickets = 348 →
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    adult_tickets + senior_tickets = 529 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l184_18443


namespace NUMINAMATH_CALUDE_no_base_for_630_four_digits_odd_final_l184_18483

theorem no_base_for_630_four_digits_odd_final : ¬ ∃ b : ℕ, 
  2 ≤ b ∧ 
  b^3 ≤ 630 ∧ 
  630 < b^4 ∧ 
  (630 % b) % 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_base_for_630_four_digits_odd_final_l184_18483


namespace NUMINAMATH_CALUDE_rotation_to_second_quadrant_l184_18433

/-- Given a complex number z = (-1+3i)/i, prove that rotating the point A 
    corresponding to z counterclockwise by 2π/3 radians results in a point B 
    in the second quadrant. -/
theorem rotation_to_second_quadrant (z : ℂ) : 
  z = (-1 + 3*Complex.I) / Complex.I → 
  let A := z
  let θ := 2 * Real.pi / 3
  let B := Complex.exp (Complex.I * θ) * A
  (B.re < 0 ∧ B.im > 0) := by sorry

end NUMINAMATH_CALUDE_rotation_to_second_quadrant_l184_18433


namespace NUMINAMATH_CALUDE_inequality_solution_l184_18439

theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) ≥ 2 * x ↔ x ≤ 0 ∨ (1 < x ∧ x ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l184_18439


namespace NUMINAMATH_CALUDE_magnitude_sum_of_vectors_l184_18421

/-- Given two plane vectors a and b, prove that |a + b| = √5 under specific conditions -/
theorem magnitude_sum_of_vectors (a b : ℝ × ℝ) : 
  a = (1, 1) → 
  ‖b‖ = 1 → 
  Real.cos (Real.pi / 4) * ‖a‖ * ‖b‖ = a.fst * b.fst + a.snd * b.snd →
  ‖a + b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_sum_of_vectors_l184_18421


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l184_18426

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^7) = 0.0000752 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l184_18426


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l184_18461

/-- Given a function f(x) = ax² - bx + ln x where a and b are real numbers,
    and g(x) = f(x) - ax² = -bx + ln x has two distinct zeros x₁ and x₂,
    prove that x₁x₂ > e² -/
theorem zeros_product_greater_than_e_squared (a b x₁ x₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : -b * x₁ + Real.log x₁ = 0) 
  (h₃ : -b * x₂ + Real.log x₂ = 0) : 
  x₁ * x₂ > Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l184_18461


namespace NUMINAMATH_CALUDE_prize_distributions_count_l184_18417

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 7

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := num_bowlers - 1

/-- The number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- The total number of possible prize distributions -/
def total_distributions : ℕ := outcomes_per_game ^ num_games

/-- Theorem stating that the number of possible prize distributions is 64 -/
theorem prize_distributions_count :
  total_distributions = 64 := by sorry

end NUMINAMATH_CALUDE_prize_distributions_count_l184_18417


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l184_18449

theorem unique_solution_quadratic (k : ℝ) :
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ (k = 0 ∨ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l184_18449


namespace NUMINAMATH_CALUDE_fib_formula_l184_18440

/-- The golden ratio φ, defined as the positive solution of x² = x + 1 -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The negative solution φ' of x² = x + 1 -/
noncomputable def φ' : ℝ := (1 - Real.sqrt 5) / 2

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: The nth Fibonacci number is given by (φⁿ - φ'ⁿ) / √5 -/
theorem fib_formula (n : ℕ) : (fib n : ℝ) = (φ ^ n - φ' ^ n) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fib_formula_l184_18440


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l184_18457

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := 300

/-- The number of raisin cookies Helen baked the day before yesterday -/
def raisin_cookies_day_before : ℕ := 280

/-- The difference in raisin cookies between yesterday and the day before -/
def raisin_cookie_difference : ℕ := raisin_cookies_yesterday - raisin_cookies_day_before

theorem helen_raisin_cookies : raisin_cookie_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_helen_raisin_cookies_l184_18457


namespace NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l184_18410

/-- The number of marbles in a dozen -/
def marbles_per_dozen : ℕ := 12

/-- The number of dozens of red marbles Jessica has -/
def jessica_dozens : ℕ := 3

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 144

/-- The ratio of Sandy's red marbles to Jessica's red marbles -/
def marble_ratio : ℚ := sandy_marbles / (jessica_dozens * marbles_per_dozen)

theorem sandy_jessica_marble_ratio :
  marble_ratio = 4 := by sorry

end NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l184_18410


namespace NUMINAMATH_CALUDE_cookies_taken_theorem_l184_18499

/-- Calculates the number of cookies taken out in 6 days given the initial count,
    remaining count after 10 days, and assuming equal daily consumption. -/
def cookies_taken_in_six_days (initial_count : ℕ) (remaining_count : ℕ) : ℕ :=
  let total_taken := initial_count - remaining_count
  let daily_taken := total_taken / 10
  6 * daily_taken

/-- Theorem stating that given 150 initial cookies and 45 remaining after 10 days,
    the number of cookies taken in 6 days is 63. -/
theorem cookies_taken_theorem :
  cookies_taken_in_six_days 150 45 = 63 := by
  sorry

#eval cookies_taken_in_six_days 150 45

end NUMINAMATH_CALUDE_cookies_taken_theorem_l184_18499


namespace NUMINAMATH_CALUDE_computer_price_increase_l184_18496

theorem computer_price_increase (d : ℝ) : 
  (d * 1.3 = 377) → (2 * d = 580) := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l184_18496


namespace NUMINAMATH_CALUDE_children_playing_both_sports_l184_18492

theorem children_playing_both_sports 
  (total : ℕ) 
  (tennis : ℕ) 
  (squash : ℕ) 
  (neither : ℕ) 
  (h1 : total = 38) 
  (h2 : tennis = 19) 
  (h3 : squash = 21) 
  (h4 : neither = 10) : 
  tennis + squash - (total - neither) = 12 := by
sorry

end NUMINAMATH_CALUDE_children_playing_both_sports_l184_18492


namespace NUMINAMATH_CALUDE_tracy_candies_l184_18481

theorem tracy_candies : ∃ (initial : ℕ) (brother_took : ℕ), 
  initial % 20 = 0 ∧ 
  1 ≤ brother_took ∧ 
  brother_took ≤ 6 ∧
  (3 * initial) / 5 - 40 - brother_took = 4 ∧
  initial = 80 := by sorry

end NUMINAMATH_CALUDE_tracy_candies_l184_18481


namespace NUMINAMATH_CALUDE_only_B_is_difference_of_squares_l184_18465

-- Define the difference of squares formula
def difference_of_squares (a b : ℝ) : ℝ := a^2 - b^2

-- Define the expressions
def expr_A (x : ℝ) : ℝ := (x - 2) * (x + 1)
def expr_B (x y : ℝ) : ℝ := (x + 2*y) * (x - 2*y)
def expr_C (x y : ℝ) : ℝ := (x + y) * (-x - y)
def expr_D (x : ℝ) : ℝ := (-x + 1) * (x - 1)

-- Theorem stating that only expr_B fits the difference of squares formula
theorem only_B_is_difference_of_squares :
  (∃ (a b : ℝ), expr_B x y = difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_A x ≠ difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_C x y ≠ difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_D x ≠ difference_of_squares a b) :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_difference_of_squares_l184_18465


namespace NUMINAMATH_CALUDE_bill_drew_140_lines_l184_18459

/-- The number of lines drawn for a given shape --/
def lines_for_shape (num_shapes : ℕ) (sides_per_shape : ℕ) : ℕ :=
  num_shapes * sides_per_shape

/-- The total number of lines drawn by Bill --/
def total_lines : ℕ :=
  let triangles := lines_for_shape 12 3
  let squares := lines_for_shape 8 4
  let pentagons := lines_for_shape 4 5
  let hexagons := lines_for_shape 6 6
  let octagons := lines_for_shape 2 8
  triangles + squares + pentagons + hexagons + octagons

theorem bill_drew_140_lines : total_lines = 140 := by
  sorry

end NUMINAMATH_CALUDE_bill_drew_140_lines_l184_18459


namespace NUMINAMATH_CALUDE_smallest_number_l184_18404

theorem smallest_number (S : Set ℤ) (h1 : S = {1, 0, -2, -3}) :
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l184_18404


namespace NUMINAMATH_CALUDE_derivative_of_linear_function_l184_18482

theorem derivative_of_linear_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => 2 * x
  (deriv y) x = 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_linear_function_l184_18482


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l184_18414

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l184_18414


namespace NUMINAMATH_CALUDE_locus_of_equal_power_l184_18405

/-- Given two non-concentric circles in a plane, the locus of points with equal power
    relative to both circles is a straight line. -/
theorem locus_of_equal_power (R₁ R₂ a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, ((x + a)^2 + y^2 - R₁^2 = (x - a)^2 + y^2 - R₂^2) ↔ (x = k) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_equal_power_l184_18405


namespace NUMINAMATH_CALUDE_D_72_eq_27_l184_18489

/-- 
D(n) represents the number of ways to write a positive integer n as a product of 
integers greater than 1, where the order matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- 
factorizations(n) represents the list of all valid factorizations of n,
where each factorization is a list of integers greater than 1.
-/
def factorizations (n : ℕ+) : List (List ℕ+) := sorry

/-- 
is_valid_factorization(n, factors) checks if the given list of factors
is a valid factorization of n according to the problem's conditions.
-/
def is_valid_factorization (n : ℕ+) (factors : List ℕ+) : Prop :=
  factors.all (· > 1) ∧ factors.prod = n

theorem D_72_eq_27 : D 72 = 27 := by sorry

end NUMINAMATH_CALUDE_D_72_eq_27_l184_18489


namespace NUMINAMATH_CALUDE_inequality_proof_l184_18451

theorem inequality_proof (r p q : ℝ) (hr : r > 0) (hp : p > 0) (hq : q > 0) (h : p^2 * r > q^2 * r) :
  1 > -q/p := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l184_18451


namespace NUMINAMATH_CALUDE_dans_music_store_spending_l184_18406

def clarinet_cost : ℚ := 130.30
def songbook_cost : ℚ := 11.24

theorem dans_music_store_spending :
  clarinet_cost + songbook_cost = 141.54 := by sorry

end NUMINAMATH_CALUDE_dans_music_store_spending_l184_18406


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l184_18464

-- Define an even function that is increasing on [0,+∞)
def is_even_and_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x < y → f x < f y)

-- State the theorem
theorem even_increasing_inequality (f : ℝ → ℝ) 
  (h : is_even_and_increasing_on_nonneg f) : 
  f π > f (-3) ∧ f (-3) > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l184_18464


namespace NUMINAMATH_CALUDE_quadratic_always_negative_l184_18429

theorem quadratic_always_negative (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (m - 1) * x + (m - 1) < 0) ↔ m < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_negative_l184_18429


namespace NUMINAMATH_CALUDE_min_operations_to_measure_88_l184_18435

/-- Represents the state of the puzzle -/
structure PuzzleState where
  barrel : ℕ
  vessel7 : ℕ
  vessel5 : ℕ

/-- Represents a pouring operation -/
inductive PourOperation
  | FillFrom7 : PourOperation
  | FillFrom5 : PourOperation
  | EmptyTo7 : PourOperation
  | EmptyTo5 : PourOperation
  | Pour7To5 : PourOperation
  | Pour5To7 : PourOperation

/-- Applies a single pouring operation to a puzzle state -/
def applyOperation (state : PuzzleState) (op : PourOperation) : PuzzleState :=
  sorry

/-- Checks if a sequence of operations is valid and results in the target state -/
def isValidSequence (initialState : PuzzleState) (targetBarrel : ℕ) (ops : List PourOperation) : Bool :=
  sorry

/-- Theorem: The minimum number of operations to measure 88 quarts is 17 -/
theorem min_operations_to_measure_88 :
  ∃ (ops : List PourOperation),
    ops.length = 17 ∧
    isValidSequence (PuzzleState.mk 108 0 0) 88 ops ∧
    ∀ (other_ops : List PourOperation),
      isValidSequence (PuzzleState.mk 108 0 0) 88 other_ops →
      other_ops.length ≥ 17 :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_measure_88_l184_18435


namespace NUMINAMATH_CALUDE_job_completion_time_l184_18462

/-- Given workers p and q, where p can complete a job in 4 days and q's daily work rate
    is one-third of p's, prove that p and q working together can complete the job in 3 days. -/
theorem job_completion_time (p q : ℝ) 
    (hp : p = 1 / 4)  -- p's daily work rate
    (hq : q = 1 / 3 * p) : -- q's daily work rate relative to p
    1 / (p + q) = 3 := by
  sorry


end NUMINAMATH_CALUDE_job_completion_time_l184_18462


namespace NUMINAMATH_CALUDE_min_distance_MN_l184_18424

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := parabola x y

-- Define a line passing through F(0,1)
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the intersection of a line with the parabola
def intersection_line_parabola (k : ℝ) (x y : ℝ) : Prop :=
  point_on_parabola x y ∧ line_through_F k x y

-- Define the line AO (or BO)
def line_AO (x₁ y₁ : ℝ) (x y : ℝ) : Prop := y = (y₁/x₁) * x

-- Define the intersection of line AO (or BO) with line l
def intersection_AO_l (x₁ y₁ : ℝ) (x y : ℝ) : Prop :=
  line_AO x₁ y₁ x y ∧ line_l x y

-- The main theorem
theorem min_distance_MN :
  ∃ (min_dist : ℝ),
    min_dist = 8 * Real.sqrt 2 / 5 ∧
    ∀ (k : ℝ) (x₁ y₁ x₂ y₂ xM yM xN yN : ℝ),
      intersection_line_parabola k x₁ y₁ →
      intersection_line_parabola k x₂ y₂ →
      intersection_AO_l x₁ y₁ xM yM →
      intersection_AO_l x₂ y₂ xN yN →
      Real.sqrt ((xM - xN)^2 + (yM - yN)^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_MN_l184_18424


namespace NUMINAMATH_CALUDE_divisibility_property_l184_18408

theorem divisibility_property (a b c d : ℤ) 
  (h : (a - c) ∣ (a * b + c * d)) : 
  (a - c) ∣ (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l184_18408


namespace NUMINAMATH_CALUDE_wall_height_breadth_ratio_l184_18495

/-- Proves that the ratio of height to breadth of a wall with given dimensions is 5:1 -/
theorem wall_height_breadth_ratio :
  ∀ (h b l : ℝ),
    b = 0.4 →
    l = 8 * h →
    ∃ (n : ℝ), h = n * b →
    l * b * h = 12.8 →
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_wall_height_breadth_ratio_l184_18495


namespace NUMINAMATH_CALUDE_furniture_fraction_l184_18472

theorem furniture_fraction (original_savings tv_cost : ℚ) 
  (h1 : original_savings = 500)
  (h2 : tv_cost = 100) : 
  (original_savings - tv_cost) / original_savings = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_furniture_fraction_l184_18472


namespace NUMINAMATH_CALUDE_max_profit_at_100_l184_18453

-- Define the cost function
def C (x : ℕ) : ℚ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

-- Define the profit function
def L (x : ℕ) : ℚ :=
  if x < 80 then -(1/3) * x^2 + 40 * x - 250
  else 1200 - (x + 10000 / x)

-- Theorem statement
theorem max_profit_at_100 :
  ∀ x : ℕ, x > 0 → L x ≤ 1000 ∧ L 100 = 1000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_100_l184_18453


namespace NUMINAMATH_CALUDE_solve_for_y_l184_18463

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 4*x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l184_18463


namespace NUMINAMATH_CALUDE_sequence_inequality_l184_18425

theorem sequence_inequality (n : ℕ) (a : ℝ) (seq : ℕ → ℝ) 
  (h1 : seq 1 = a)
  (h2 : seq n = a)
  (h3 : ∀ k ∈ Finset.range (n - 2), seq (k + 2) ≤ (seq (k + 1) + seq (k + 3)) / 2) :
  ∀ k ∈ Finset.range n, seq (k + 1) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l184_18425


namespace NUMINAMATH_CALUDE_probability_two_even_toys_l184_18497

def number_of_toys : ℕ := 21
def number_of_even_toys : ℕ := 10

theorem probability_two_even_toys :
  let p := (number_of_even_toys / number_of_toys) * ((number_of_even_toys - 1) / (number_of_toys - 1))
  p = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_even_toys_l184_18497


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l184_18490

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (face_cards : Nat)
  (hearts : Nat)
  (face_hearts : Nat)

/-- Calculates the probability of drawing a face card, then any heart, then a face card -/
def probability_face_heart_face (d : Deck) : Rat :=
  let first_draw := d.face_cards / d.total_cards
  let second_draw := (d.hearts - d.face_hearts) / (d.total_cards - 1)
  let third_draw := (d.face_cards - 1) / (d.total_cards - 2)
  first_draw * second_draw * third_draw

/-- Standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , face_cards := 12
  , hearts := 13
  , face_hearts := 3 }

theorem probability_in_standard_deck :
  probability_face_heart_face standard_deck = 1320 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l184_18490


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l184_18402

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 8) (h₃ : a₃ = 13) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 73 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l184_18402


namespace NUMINAMATH_CALUDE_consecutive_square_roots_l184_18401

theorem consecutive_square_roots (n : ℕ) (h : Real.sqrt n = 3) :
  Real.sqrt (n + 1) = 3 + Real.sqrt 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_square_roots_l184_18401


namespace NUMINAMATH_CALUDE_bridge_length_l184_18455

/-- The length of a bridge given train parameters -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 215 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l184_18455


namespace NUMINAMATH_CALUDE_quinary1234_equals_octal302_l184_18488

/-- Converts a quinary (base-5) number to decimal (base-10) --/
def quinaryToDecimal (q : ℕ) : ℕ := sorry

/-- Converts a decimal (base-10) number to octal (base-8) --/
def decimalToOctal (d : ℕ) : ℕ := sorry

/-- The quinary representation of 1234 --/
def quinary1234 : ℕ := 1234

/-- The octal representation of 302 --/
def octal302 : ℕ := 302

theorem quinary1234_equals_octal302 : 
  decimalToOctal (quinaryToDecimal quinary1234) = octal302 := by sorry

end NUMINAMATH_CALUDE_quinary1234_equals_octal302_l184_18488


namespace NUMINAMATH_CALUDE_combined_efficiency_l184_18438

-- Define the variables
def ray_efficiency : ℚ := 50
def tom_efficiency : ℚ := 10
def ray_distance : ℚ := 50
def tom_distance : ℚ := 100

-- Define the theorem
theorem combined_efficiency :
  let total_distance := ray_distance + tom_distance
  let ray_fuel := ray_distance / ray_efficiency
  let tom_fuel := tom_distance / tom_efficiency
  let total_fuel := ray_fuel + tom_fuel
  total_distance / total_fuel = 150 / 11 :=
by sorry

end NUMINAMATH_CALUDE_combined_efficiency_l184_18438


namespace NUMINAMATH_CALUDE_union_M_N_l184_18415

def M : Set ℤ := {x | |x| < 2}
def N : Set ℤ := {-2, -1, 0}

theorem union_M_N : M ∪ N = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l184_18415


namespace NUMINAMATH_CALUDE_log_y_equals_negative_two_l184_18470

theorem log_y_equals_negative_two (y : ℝ) : 
  y = (Real.log 3 / Real.log 27) ^ (Real.log 81 / Real.log 3) → 
  Real.log y / Real.log 9 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_y_equals_negative_two_l184_18470


namespace NUMINAMATH_CALUDE_fifteenth_even_multiple_of_5_l184_18436

/-- A function that returns the nth positive integer that is both even and a multiple of 5 -/
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

/-- The 15th positive integer that is both even and a multiple of 5 is 150 -/
theorem fifteenth_even_multiple_of_5 : evenMultipleOf5 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_even_multiple_of_5_l184_18436


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l184_18458

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define symmetry about a vertical line
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  IsEven (fun x ↦ f (x + 1)) → SymmetricAboutLine f 1 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l184_18458


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l184_18431

theorem roots_sum_reciprocal_cubes (r s : ℂ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_cubes_l184_18431


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l184_18476

theorem quadratic_two_distinct_roots : ∃ x y : ℝ, x ≠ y ∧ 
  x^2 + 2*x - 5 = 0 ∧ y^2 + 2*y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l184_18476


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l184_18473

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * q) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l184_18473


namespace NUMINAMATH_CALUDE_proposition_q_false_iff_a_lt_2_l184_18407

theorem proposition_q_false_iff_a_lt_2 (a : ℝ) :
  (¬∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_proposition_q_false_iff_a_lt_2_l184_18407


namespace NUMINAMATH_CALUDE_two_workers_better_l184_18491

/-- Represents the number of production lines -/
def num_lines : ℕ := 3

/-- Represents the probability of failure for each production line -/
def failure_prob : ℚ := 1/3

/-- Represents the monthly salary of each maintenance worker -/
def worker_salary : ℕ := 10000

/-- Represents the monthly profit of a production line with no failure -/
def profit_no_failure : ℕ := 120000

/-- Represents the monthly profit of a production line with failure and repair -/
def profit_with_repair : ℕ := 80000

/-- Represents the monthly profit of a production line with failure and no repair -/
def profit_no_repair : ℕ := 0

/-- Calculates the expected profit with a given number of maintenance workers -/
def expected_profit (num_workers : ℕ) : ℚ :=
  sorry

/-- Theorem stating that the expected profit with 2 workers is greater than with 1 worker -/
theorem two_workers_better :
  expected_profit 2 > expected_profit 1 := by
  sorry

end NUMINAMATH_CALUDE_two_workers_better_l184_18491


namespace NUMINAMATH_CALUDE_smaller_number_proof_l184_18442

theorem smaller_number_proof (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) :
  min x y = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l184_18442


namespace NUMINAMATH_CALUDE_intense_goblet_points_difference_l184_18412

/-- The number of teams in the tournament -/
def num_teams : ℕ := 10

/-- The number of points awarded for a win -/
def win_points : ℕ := 4

/-- The number of points awarded for a tie -/
def tie_points : ℕ := 2

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 1

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- The minimum total points possible in the tournament -/
def min_total_points : ℕ := num_teams * (num_teams - 1) * loss_points

theorem intense_goblet_points_difference :
  max_total_points - min_total_points = 90 := by
  sorry

end NUMINAMATH_CALUDE_intense_goblet_points_difference_l184_18412


namespace NUMINAMATH_CALUDE_partner_C_profit_share_l184_18475

-- Define the investment ratios and time periods
def investment_ratio_A : ℚ := 4
def investment_ratio_B : ℚ := 1
def investment_ratio_C : ℚ := 16/3
def investment_ratio_D : ℚ := 2
def investment_ratio_E : ℚ := 2/3

def time_period_A : ℚ := 6
def time_period_B : ℚ := 9
def time_period_C : ℚ := 12
def time_period_D : ℚ := 8
def time_period_E : ℚ := 10

def total_profit : ℚ := 220000

-- Define the theorem
theorem partner_C_profit_share :
  let total_share := investment_ratio_A * time_period_A +
                     investment_ratio_B * time_period_B +
                     investment_ratio_C * time_period_C +
                     investment_ratio_D * time_period_D +
                     investment_ratio_E * time_period_E
  let C_share := (investment_ratio_C * time_period_C / total_share) * total_profit
  ∃ ε > 0, |C_share - 49116.32| < ε :=
by sorry

end NUMINAMATH_CALUDE_partner_C_profit_share_l184_18475


namespace NUMINAMATH_CALUDE_base3_addition_theorem_l184_18498

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else convert (m / 3) ((m % 3) :: acc)
    convert n []

theorem base3_addition_theorem :
  let a := base3ToDecimal [2]
  let b := base3ToDecimal [0, 2, 1]
  let c := base3ToDecimal [1, 2, 2]
  let d := base3ToDecimal [2, 1, 1, 1]
  let e := base3ToDecimal [2, 2, 0, 1]
  let sum := a + b + c + d + e
  decimalToBase3 sum = [1, 0, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base3_addition_theorem_l184_18498


namespace NUMINAMATH_CALUDE_yield_prediction_80kg_l184_18446

/-- Predicts the rice yield based on the amount of fertilizer applied. -/
def predict_yield (x : ℝ) : ℝ := 5 * x + 250

/-- Theorem stating that when 80 kg of fertilizer is applied, the predicted yield is 650 kg. -/
theorem yield_prediction_80kg : predict_yield 80 = 650 := by
  sorry

end NUMINAMATH_CALUDE_yield_prediction_80kg_l184_18446


namespace NUMINAMATH_CALUDE_product_digit_sum_l184_18427

/-- The first 101-digit number -/
def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

/-- The second 101-digit number -/
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

/-- Function to get the hundreds digit of a number -/
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

/-- Function to get the tens digit of a number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem product_digit_sum :
  hundreds_digit (number1 * number2) + tens_digit (number1 * number2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l184_18427


namespace NUMINAMATH_CALUDE_inequality_solution_l184_18484

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) ↔ 
  (x < 0 ∨ (1 < x ∧ x < 2) ∨ 2 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l184_18484


namespace NUMINAMATH_CALUDE_final_value_one_fourth_l184_18448

theorem final_value_one_fourth (x : ℝ) : 
  (1 / 4) * ((5 * x + 3) - 1) = (5 * x) / 4 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_final_value_one_fourth_l184_18448


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l184_18452

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 6
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1296 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l184_18452


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l184_18467

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (2*m + 1, -1/2)
  let b : ℝ × ℝ := (2*m, 1)
  are_parallel a b → m = -1/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l184_18467


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l184_18469

theorem quadratic_equation_solution (h : 108 * (3/4)^2 + 61 = 145 * (3/4) - 7) :
  ∃ x : ℚ, x ≠ 3/4 ∧ 108 * x^2 + 61 = 145 * x - 7 ∧ x = 68/81 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l184_18469


namespace NUMINAMATH_CALUDE_socorro_multiplication_time_l184_18437

/-- The time spent on multiplication problems each day, given the total training time,
    number of training days, and daily time spent on division problems. -/
def time_on_multiplication (total_hours : ℕ) (days : ℕ) (division_minutes : ℕ) : ℕ :=
  ((total_hours * 60) - (days * division_minutes)) / days

/-- Theorem stating that Socorro spends 10 minutes each day on multiplication problems. -/
theorem socorro_multiplication_time :
  time_on_multiplication 5 10 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_socorro_multiplication_time_l184_18437


namespace NUMINAMATH_CALUDE_parabola_c_value_l184_18485

/-- A parabola with equation y = ax^2 + bx + c, vertex (-1, -2), and passing through (-2, -1) has c = -1 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a*x^2 + b*x + c) →  -- Equation of the parabola
  (-2 = a*(-1)^2 + b*(-1) + c) →      -- Vertex condition
  (-1 = a*(-2)^2 + b*(-2) + c) →      -- Point condition
  c = -1 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l184_18485


namespace NUMINAMATH_CALUDE_polynomial_remainder_l184_18418

theorem polynomial_remainder (x : ℝ) : 
  (x^15 + 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l184_18418


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l184_18480

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l184_18480


namespace NUMINAMATH_CALUDE_correct_time_allocation_l184_18432

/-- Represents the time allocation for different tasks -/
structure TimeAllocation where
  clientCalls : ℕ
  accounting : ℕ
  reports : ℕ
  meetings : ℕ

/-- Calculates the time allocation based on a given ratio and total time -/
def calculateTimeAllocation (ratio : List ℚ) (totalTime : ℕ) : TimeAllocation :=
  sorry

/-- Checks if the calculated time allocation is correct -/
def isCorrectAllocation (allocation : TimeAllocation) : Prop :=
  allocation.clientCalls = 383 ∧
  allocation.accounting = 575 ∧
  allocation.reports = 767 ∧
  allocation.meetings = 255

/-- Theorem stating that the calculated time allocation for the given ratio and total time is correct -/
theorem correct_time_allocation :
  let ratio := [3, 4.5, 6, 2]
  let totalTime := 1980
  let allocation := calculateTimeAllocation ratio totalTime
  isCorrectAllocation allocation ∧ 
  allocation.clientCalls + allocation.accounting + allocation.reports + allocation.meetings = totalTime :=
by sorry

end NUMINAMATH_CALUDE_correct_time_allocation_l184_18432
