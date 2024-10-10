import Mathlib

namespace convex_polygon_with_arithmetic_angles_l1965_196562

/-- A convex polygon with interior angles forming an arithmetic sequence,
    where the smallest angle is 100° and the largest angle is 140°, has exactly 6 sides. -/
theorem convex_polygon_with_arithmetic_angles (n : ℕ) : 
  n ≥ 3 → -- convex polygon has at least 3 sides
  (∃ (a d : ℝ), 
    a = 100 ∧ -- smallest angle
    a + (n - 1) * d = 140 ∧ -- largest angle
    ∀ i : ℕ, i < n → a + i * d ≥ 0 ∧ a + i * d ≤ 180) → -- all angles are between 0° and 180°
  (n : ℝ) * (100 + 140) / 2 = 180 * (n - 2) →
  n = 6 :=
sorry

end convex_polygon_with_arithmetic_angles_l1965_196562


namespace triangle_consecutive_integers_l1965_196522

theorem triangle_consecutive_integers (n : ℕ) :
  let a := n - 1
  let b := n
  let c := n + 1
  let s := (a + b + c) / 2
  let area := n + 2
  (area : ℝ)^2 = s * (s - a) * (s - b) * (s - c) → n = 4 :=
by sorry

end triangle_consecutive_integers_l1965_196522


namespace birthday_count_theorem_l1965_196545

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a birthday -/
structure Birthday where
  month : ℕ
  day : ℕ

def startDate : Date := ⟨2012, 12, 26⟩

def dadBirthday : Birthday := ⟨5, 1⟩
def chunchunBirthday : Birthday := ⟨7, 1⟩

def daysToCount : ℕ := 2013

/-- Counts the number of birthdays between two dates -/
def countBirthdays (start : Date) (days : ℕ) (birthday : Birthday) : ℕ :=
  sorry

theorem birthday_count_theorem :
  countBirthdays startDate daysToCount dadBirthday +
  countBirthdays startDate daysToCount chunchunBirthday = 11 :=
by sorry

end birthday_count_theorem_l1965_196545


namespace coat_price_calculation_l1965_196518

/-- Calculates the total selling price of a coat given its original price, discount percentage, and tax percentage. -/
def totalSellingPrice (originalPrice discount tax : ℚ) : ℚ :=
  let salePrice := originalPrice * (1 - discount)
  salePrice * (1 + tax)

/-- Theorem stating that the total selling price of a coat with original price $120, 30% discount, and 15% tax is $96.60. -/
theorem coat_price_calculation :
  totalSellingPrice 120 (30 / 100) (15 / 100) = 966 / 10 := by
  sorry

#eval totalSellingPrice 120 (30 / 100) (15 / 100)

end coat_price_calculation_l1965_196518


namespace degree_of_5m2n3_l1965_196596

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : ℕ) (n : ℕ) : ℕ := m + n

/-- The monomial 5m^2n^3 has degree 5. -/
theorem degree_of_5m2n3 : degree_of_monomial 2 3 = 5 := by sorry

end degree_of_5m2n3_l1965_196596


namespace sum_and_ratio_to_difference_l1965_196573

theorem sum_and_ratio_to_difference (x y : ℝ) : 
  x + y = 520 → x / y = 0.75 → y - x = 74 := by
  sorry

end sum_and_ratio_to_difference_l1965_196573


namespace complex_modulus_problem_l1965_196506

theorem complex_modulus_problem (z : ℂ) : z = (3 - Complex.I) / (1 + 2 * Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l1965_196506


namespace quadratic_complex_roots_l1965_196564

theorem quadratic_complex_roots (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ a b : ℝ, b ≠ 0 ∧ (∀ x : ℂ, x^2 + p*x + q = 0 → x = Complex.mk a b ∨ x = Complex.mk a (-b))) →
  (∀ x : ℂ, x^2 + p*x + q = 0 → x.re = 1/2) →
  p = 1 := by sorry

end quadratic_complex_roots_l1965_196564


namespace product_of_special_ratio_numbers_l1965_196563

theorem product_of_special_ratio_numbers (x y : ℝ) 
  (h : ∃ (k : ℝ), k > 0 ∧ x - y = k ∧ x + y = 2*k ∧ x^2 * y^2 = 18*k) : 
  x * y = 16 := by
sorry

end product_of_special_ratio_numbers_l1965_196563


namespace Z_in_first_quadrant_l1965_196587

/-- A complex number is in the first quadrant if its real and imaginary parts are both positive. -/
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

/-- The sum of two complex numbers (5+4i) and (-1+2i) -/
def Z : ℂ := Complex.mk 5 4 + Complex.mk (-1) 2

/-- Theorem: Z is located in the first quadrant of the complex plane -/
theorem Z_in_first_quadrant : is_in_first_quadrant Z := by
  sorry

end Z_in_first_quadrant_l1965_196587


namespace water_bottle_capacity_l1965_196540

/-- The capacity of a water bottle in milliliters -/
def bottle_capacity : ℕ := 12800

/-- The volume of the smaller cup in milliliters -/
def small_cup : ℕ := 250

/-- The volume of the larger cup in milliliters -/
def large_cup : ℕ := 600

/-- The number of times water is scooped with the smaller cup -/
def small_cup_scoops : ℕ := 20

/-- The number of times water is scooped with the larger cup -/
def large_cup_scoops : ℕ := 13

/-- Conversion factor from milliliters to liters -/
def ml_to_l : ℚ := 1 / 1000

theorem water_bottle_capacity :
  (bottle_capacity : ℚ) * ml_to_l = 12.8 ∧
  bottle_capacity = small_cup * small_cup_scoops + large_cup * large_cup_scoops :=
sorry

end water_bottle_capacity_l1965_196540


namespace desired_interest_percentage_l1965_196524

theorem desired_interest_percentage 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_value : ℝ) 
  (h1 : face_value = 56) 
  (h2 : dividend_rate = 0.09) 
  (h3 : market_value = 42) : 
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

end desired_interest_percentage_l1965_196524


namespace consecutive_ones_count_l1965_196546

def a : ℕ → ℕ
  | 0 => 1  -- We define a(0) = 1 to simplify the recursion
  | 1 => 2
  | 2 => 3
  | (n + 3) => a (n + 2) + a (n + 1)

theorem consecutive_ones_count : 
  (2^8 : ℕ) - a 8 = 201 :=
sorry

end consecutive_ones_count_l1965_196546


namespace water_depth_l1965_196574

theorem water_depth (ron_height dean_height water_depth : ℕ) : 
  ron_height = 13 →
  dean_height = ron_height + 4 →
  water_depth = 15 * dean_height →
  water_depth = 255 := by
sorry

end water_depth_l1965_196574


namespace chocolate_cookie_percentage_l1965_196560

/-- Calculates the percentage of chocolate in cookies given the initial ingredients and leftover chocolate. -/
theorem chocolate_cookie_percentage
  (dough : ℝ)
  (initial_chocolate : ℝ)
  (leftover_chocolate : ℝ)
  (h_dough : dough = 36)
  (h_initial : initial_chocolate = 13)
  (h_leftover : leftover_chocolate = 4) :
  (initial_chocolate - leftover_chocolate) / (dough + initial_chocolate - leftover_chocolate) * 100 = 20 := by
  sorry

end chocolate_cookie_percentage_l1965_196560


namespace survivor_quitter_probability_survivor_quitter_probability_proof_l1965_196511

/-- The probability that both quitters are from the same tribe in a Survivor game -/
theorem survivor_quitter_probability : ℚ :=
  let total_participants : ℕ := 32
  let tribe_size : ℕ := 16
  let num_quitters : ℕ := 2

  -- The probability that both quitters are from the same tribe
  15 / 31

/-- Proof of the survivor_quitter_probability theorem -/
theorem survivor_quitter_probability_proof :
  survivor_quitter_probability = 15 / 31 := by
  sorry

end survivor_quitter_probability_survivor_quitter_probability_proof_l1965_196511


namespace complex_magnitude_problem_l1965_196570

theorem complex_magnitude_problem (z : ℂ) : z = (2 - Complex.I) / (1 + 2 * Complex.I) → Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l1965_196570


namespace valid_pairings_count_l1965_196532

def num_colors : ℕ := 5

def total_pairings (n : ℕ) : ℕ := n * n

def same_color_pairings (n : ℕ) : ℕ := n

theorem valid_pairings_count :
  total_pairings num_colors - same_color_pairings num_colors = 20 := by
  sorry

end valid_pairings_count_l1965_196532


namespace triangular_prism_surface_area_l1965_196519

/-- The surface area of a triangular-based prism created by vertically cutting a rectangular prism -/
theorem triangular_prism_surface_area (l w h : ℝ) (h_l : l = 3) (h_w : w = 5) (h_h : h = 12) :
  let front_area := l * h
  let side_area := l * w
  let triangle_area := w * h / 2
  let back_diagonal := Real.sqrt (w^2 + h^2)
  let back_area := l * back_diagonal
  front_area + side_area + 2 * triangle_area + back_area = 150 :=
sorry

end triangular_prism_surface_area_l1965_196519


namespace symmetry_condition_range_on_interval_range_positive_l1965_196537

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2 - a) * x - 2 * a

-- Theorem for symmetry condition
theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, f a (1 + x) = f a (1 - x)) → a = 4 :=
sorry

-- Theorem for range on [0,4] when a = 4
theorem range_on_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → -9 ≤ f 4 x ∧ f 4 x ≤ -5) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 4 ∧ f 4 x = -9 ∧ f 4 y = -5) :=
sorry

-- Theorem for the range of x when f(x) > 0
theorem range_positive (a : ℝ) (x : ℝ) :
  (a = -2 → (f a x > 0 ↔ x ≠ -2)) ∧
  (a > -2 → (f a x > 0 ↔ x < -2 ∨ x > a)) ∧
  (a < -2 → (f a x > 0 ↔ -2 < x ∧ x < a)) :=
sorry

end symmetry_condition_range_on_interval_range_positive_l1965_196537


namespace one_is_not_prime_and_not_composite_l1965_196558

-- Define the properties of natural numbers based on their divisors
def has_only_one_divisor (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

-- Theorem to prove
theorem one_is_not_prime_and_not_composite : 
  ¬(is_prime 1 ∧ ¬is_composite 1) :=
sorry

end one_is_not_prime_and_not_composite_l1965_196558


namespace a_investment_is_400_l1965_196536

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  a_investment : ℝ
  b_investment : ℝ
  total_profit : ℝ
  a_profit : ℝ
  b_investment_time : ℝ
  total_time : ℝ

/-- Theorem stating that given the conditions, A's investment was $400 -/
theorem a_investment_is_400 (scenario : InvestmentScenario) 
  (h1 : scenario.b_investment = 200)
  (h2 : scenario.total_profit = 100)
  (h3 : scenario.a_profit = 80)
  (h4 : scenario.b_investment_time = 6)
  (h5 : scenario.total_time = 12)
  (h6 : scenario.a_investment * scenario.total_time / 
        (scenario.b_investment * scenario.b_investment_time) = 
        scenario.a_profit / (scenario.total_profit - scenario.a_profit)) :
  scenario.a_investment = 400 := by
  sorry


end a_investment_is_400_l1965_196536


namespace quadratic_inequality_solution_l1965_196534

theorem quadratic_inequality_solution (a c : ℝ) :
  (∀ x : ℝ, (a * x^2 + 2 * x + c < 0) ↔ (x < -1 ∨ x > 2)) →
  a + c = 2 :=
by sorry

end quadratic_inequality_solution_l1965_196534


namespace expression_simplification_l1965_196584

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((x + 3*y)^2 - (x + y)*(x - y)) / (2*y) = 3*x + 5*y := by
  sorry

end expression_simplification_l1965_196584


namespace geometric_sequence_cos_ratio_l1965_196568

open Real

/-- Given an arithmetic sequence {a_n} with first term a₁ and common difference d,
    where 0 < d < 2π, if {cos a_n} forms a geometric sequence,
    then the common ratio of {cos a_n} is -1. -/
theorem geometric_sequence_cos_ratio
  (a₁ : ℝ) (d : ℝ) (h_d : 0 < d ∧ d < 2 * π)
  (h_geom : ∀ n : ℕ, n ≥ 1 → cos (a₁ + n * d) / cos (a₁ + (n - 1) * d) =
                           cos (a₁ + d) / cos a₁) :
  ∀ n : ℕ, n ≥ 1 → cos (a₁ + (n + 1) * d) / cos (a₁ + n * d) = -1 := by
sorry

end geometric_sequence_cos_ratio_l1965_196568


namespace equal_money_after_transfer_l1965_196501

/-- Given that Ann has $777 and Bill has $1,111, prove that if Bill gives $167 to Ann, 
    they will have equal amounts of money. -/
theorem equal_money_after_transfer (ann_initial : ℕ) (bill_initial : ℕ) (transfer : ℕ) : 
  ann_initial = 777 →
  bill_initial = 1111 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer :=
by
  sorry

#check equal_money_after_transfer

end equal_money_after_transfer_l1965_196501


namespace sum_of_real_cube_roots_of_64_l1965_196517

theorem sum_of_real_cube_roots_of_64 :
  ∃ (x : ℝ), x^3 = 64 ∧ (∀ y : ℝ, y^3 = 64 → y = x) ∧ x = 4 := by
  sorry

end sum_of_real_cube_roots_of_64_l1965_196517


namespace wall_ratio_l1965_196591

/-- Given a wall with specific dimensions, prove the ratio of its length to height --/
theorem wall_ratio (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  volume = w * h * l →
  w = 6.999999999999999 →
  volume = 86436 →
  l / h = 7 := by
sorry

end wall_ratio_l1965_196591


namespace congruence_system_solution_l1965_196531

theorem congruence_system_solution (x : ℤ) :
  (9 * x + 3) % 15 = 6 →
  x % 5 = 2 →
  x % 5 = 2 := by
sorry

end congruence_system_solution_l1965_196531


namespace first_day_price_is_four_l1965_196555

/-- Represents the sales data for a pen store over three days -/
structure PenSales where
  price1 : ℝ  -- Price per pen on the first day
  quantity1 : ℝ  -- Number of pens sold on the first day

/-- The revenue is the same for all three days given the pricing and quantity changes -/
def sameRevenue (s : PenSales) : Prop :=
  s.price1 * s.quantity1 = (s.price1 - 1) * (s.quantity1 + 100) ∧
  s.price1 * s.quantity1 = (s.price1 + 2) * (s.quantity1 - 100)

/-- The price on the first day is 4 yuan -/
theorem first_day_price_is_four (s : PenSales) (h : sameRevenue s) : s.price1 = 4 := by
  sorry

end first_day_price_is_four_l1965_196555


namespace seongjun_has_500_ttakji_l1965_196585

/-- The number of ttakji Seongjun has -/
def seongjun_ttakji : ℕ := sorry

/-- The number of ttakji Seunga has -/
def seunga_ttakji : ℕ := 100

/-- The relationship between Seongjun's and Seunga's ttakji -/
axiom ttakji_relationship : (3 / 4 : ℚ) * seongjun_ttakji - 25 = 7 * (seunga_ttakji - 50)

theorem seongjun_has_500_ttakji : seongjun_ttakji = 500 := by sorry

end seongjun_has_500_ttakji_l1965_196585


namespace hockey_league_games_l1965_196530

theorem hockey_league_games (n : ℕ) (games_per_pair : ℕ) : n = 10 ∧ games_per_pair = 4 →
  (n * (n - 1) / 2) * games_per_pair = 180 := by
  sorry

end hockey_league_games_l1965_196530


namespace polynomial_standard_form_l1965_196598

theorem polynomial_standard_form :
  ∀ (a b x : ℝ),
  (a - b) * (a + b) * (a^2 + a*b + b^2) * (a^2 - a*b + b^2) = a^6 - b^6 ∧
  (x - 1)^3 * (x + 1)^2 * (x^2 + 1) * (x^2 + x + 1) = x^9 - x^7 - x^8 - x^5 + x^4 + x^3 + x^2 - 1 ∧
  (x^4 - x^2 + 1) * (x^2 - x + 1) * (x^2 + x + 1) = x^8 + x^4 + 1 :=
by
  sorry

end polynomial_standard_form_l1965_196598


namespace crayons_per_box_l1965_196521

theorem crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) (h1 : total_crayons = 56) (h2 : num_boxes = 8) :
  total_crayons / num_boxes = 7 := by
  sorry

end crayons_per_box_l1965_196521


namespace smallest_upper_bound_for_sum_of_square_roots_l1965_196569

theorem smallest_upper_bound_for_sum_of_square_roots :
  ∃ (M : ℝ), (∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 →
    Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
    Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ M) ∧
  (M = 4 / Real.sqrt 3) ∧
  (∀ (M' : ℝ), M' < M →
    ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
      Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
      Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > M') :=
by sorry

end smallest_upper_bound_for_sum_of_square_roots_l1965_196569


namespace factor_expression_l1965_196583

theorem factor_expression (x : ℝ) : x * (x + 3) + (x + 3) = (x + 1) * (x + 3) := by
  sorry

end factor_expression_l1965_196583


namespace oldest_babysat_prime_age_l1965_196553

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def max_babysit_age (current_age : ℕ) (start_age : ℕ) (stop_age : ℕ) (years_since_stop : ℕ) : ℕ :=
  min (stop_age / 2 + years_since_stop) (current_age - 1)

def satisfies_babysit_criteria (age : ℕ) (max_age : ℕ) (gap : ℕ) : Prop :=
  age ≤ max_age ∧ ∃ n : ℕ, n ≤ gap ∧ max_age - n = age

theorem oldest_babysat_prime_age :
  ∀ (current_age : ℕ) (start_age : ℕ) (stop_age : ℕ) (years_since_stop : ℕ),
    current_age = 32 →
    start_age = 20 →
    stop_age = 22 →
    years_since_stop = 10 →
    ∃ (oldest_age : ℕ),
      is_prime oldest_age ∧
      oldest_age = 19 ∧
      satisfies_babysit_criteria oldest_age (max_babysit_age current_age start_age stop_age years_since_stop) 1 ∧
      ∀ (age : ℕ),
        is_prime age →
        satisfies_babysit_criteria age (max_babysit_age current_age start_age stop_age years_since_stop) 1 →
        age ≤ oldest_age :=
by sorry

end oldest_babysat_prime_age_l1965_196553


namespace exam_results_l1965_196571

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 34)
  (h2 : failed_english = 44)
  (h3 : failed_both = 22) :
  100 - (failed_hindi + failed_english - failed_both) = 44 := by
  sorry

end exam_results_l1965_196571


namespace crow_eating_time_l1965_196590

/-- Given a constant eating rate where 1/5 of the nuts are eaten in 8 hours,
    prove that it takes 10 hours to eat 1/4 of the nuts. -/
theorem crow_eating_time (eating_rate : ℝ → ℝ) (h1 : eating_rate (8 : ℝ) = 1/5) 
    (h2 : ∀ t1 t2 : ℝ, eating_rate (t1 + t2) = eating_rate t1 + eating_rate t2) : 
    ∃ t : ℝ, eating_rate t = 1/4 ∧ t = 10 := by
  sorry


end crow_eating_time_l1965_196590


namespace largest_prime_factor_of_1755_l1965_196552

theorem largest_prime_factor_of_1755 : ∃ p : Nat, Nat.Prime p ∧ p ∣ 1755 ∧ ∀ q : Nat, Nat.Prime q → q ∣ 1755 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1755_l1965_196552


namespace existence_of_solutions_l1965_196554

theorem existence_of_solutions (k : ℕ) (a : ℕ) (n : Fin k → ℕ) 
  (h1 : ∀ i, a > 0 ∧ n i > 0)
  (h2 : ∀ i j, i ≠ j → Nat.gcd (n i) (n j) = 1)
  (h3 : ∀ i, a ^ (n i) % (n i) = 1)
  (h4 : ∀ i, ¬(n i ∣ a - 1)) :
  ∃ (S : Finset ℕ), S.card ≥ 2^(k+1) - 2 ∧ 
    (∀ x ∈ S, x > 1 ∧ a^x % x = 1) :=
by sorry

end existence_of_solutions_l1965_196554


namespace median_formulas_l1965_196527

/-- Triangle with sides a, b, c and medians ma, mb, mc -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ma : ℝ
  mb : ℝ
  mc : ℝ

/-- Theorem: Median formula and sum of squares of medians -/
theorem median_formulas (t : Triangle) :
  t.ma^2 = (2*t.b^2 + 2*t.c^2 - t.a^2) / 4 ∧
  t.ma^2 + t.mb^2 + t.mc^2 = 3*(t.a^2 + t.b^2 + t.c^2) / 4 := by
  sorry

end median_formulas_l1965_196527


namespace solve_equation_l1965_196516

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 := by
  sorry

end solve_equation_l1965_196516


namespace unique_rectangle_existence_l1965_196589

theorem unique_rectangle_existence (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∃! (x y : ℝ), 0 < x ∧ x < y ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4 := by
  sorry

end unique_rectangle_existence_l1965_196589


namespace curve_transformation_l1965_196597

/-- Given a curve C in a plane rectangular coordinate system, 
    prove that its equation is 50x^2 + 72y^2 = 1 after an expansion transformation. -/
theorem curve_transformation (x y x' y' : ℝ) : 
  (x' = 5*x ∧ y' = 3*y) →  -- Transformation equations
  (2*x'^2 + 8*y'^2 = 1) →  -- Equation of transformed curve
  (50*x^2 + 72*y^2 = 1)    -- Equation of original curve C
  := by sorry

end curve_transformation_l1965_196597


namespace scientific_notation_32000000_l1965_196550

theorem scientific_notation_32000000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 32000000 = a * (10 : ℝ) ^ n ∧ a = 3.2 ∧ n = 7 :=
by sorry

end scientific_notation_32000000_l1965_196550


namespace set_intersection_theorem_l1965_196567

def M : Set ℝ := {x | |x| < 3}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem set_intersection_theorem : M ∩ N = {x | 1 < x ∧ x < 3} := by sorry

end set_intersection_theorem_l1965_196567


namespace after_two_right_turns_l1965_196507

/-- Represents a position in the square formation -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- The size of the square formation -/
def formationSize : Nat := 9

/-- Counts the number of people in front of a given position -/
def peopleInFront (pos : Position) : Nat :=
  formationSize - pos.row - 1

/-- Performs a right turn on a position -/
def rightTurn (pos : Position) : Position :=
  ⟨formationSize - pos.col + 1, pos.row⟩

/-- The main theorem to prove -/
theorem after_two_right_turns 
  (initialPos : Position)
  (h1 : peopleInFront initialPos = 2)
  (h2 : peopleInFront (rightTurn initialPos) = 4) :
  peopleInFront (rightTurn (rightTurn initialPos)) = 6 := by
  sorry

end after_two_right_turns_l1965_196507


namespace y_decreases_as_x_increases_l1965_196542

def tensor (m n : ℝ) : ℝ := -m * n + n

theorem y_decreases_as_x_increases :
  let f : ℝ → ℝ := λ x ↦ tensor x 2
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end y_decreases_as_x_increases_l1965_196542


namespace largest_valid_n_l1965_196538

/-- A coloring of integers from 1 to 14 using two colors -/
def Coloring := Fin 14 → Bool

/-- Check if a coloring satisfies the condition for a given k -/
def valid_for_k (c : Coloring) (k : Nat) : Prop :=
  ∃ (i j i' j' : Fin 14),
    i < j ∧ j - i = k ∧ c i = c j ∧
    i' < j' ∧ j' - i' = k ∧ c i' ≠ c j'

/-- A coloring is valid up to n if it satisfies the condition for all k from 1 to n -/
def valid_coloring (c : Coloring) (n : Nat) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → valid_for_k c k

/-- The main theorem: 11 is the largest n for which a valid coloring exists -/
theorem largest_valid_n :
  (∃ c : Coloring, valid_coloring c 11) ∧
  (∀ c : Coloring, ¬valid_coloring c 12) := by
  sorry

end largest_valid_n_l1965_196538


namespace sin_alpha_terminal_side_l1965_196556

/-- Given a point P on the terminal side of angle α with coordinates (3a, 4a) where a < 0, prove that sin α = -4/5 -/
theorem sin_alpha_terminal_side (a : ℝ) (α : ℝ) (h : a < 0) :
  let P : ℝ × ℝ := (3 * a, 4 * a)
  (P.1 = 3 * a ∧ P.2 = 4 * a) → Real.sin α = -4/5 :=
by sorry

end sin_alpha_terminal_side_l1965_196556


namespace function_is_identity_l1965_196525

/-- A function satisfying the given functional equation for all positive real numbers -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (z + 1) * f (x + y) = f (x * f z + y) + f (y * f z + x)

/-- The main theorem: if f satisfies the equation, then f(x) = x for all positive real numbers -/
theorem function_is_identity {f : ℝ → ℝ} (hf : SatisfiesEquation f) :
    ∀ x : ℝ, x > 0 → f x = x := by
  sorry

end function_is_identity_l1965_196525


namespace age_of_b_l1965_196549

/-- Given three people a, b, and c, with their ages represented as natural numbers. -/
def problem (a b c : ℕ) : Prop :=
  -- The average age of a, b, and c is 26 years
  (a + b + c) / 3 = 26 ∧
  -- The average age of a and c is 29 years
  (a + c) / 2 = 29 →
  -- The age of b is 20 years
  b = 20

/-- Theorem stating that under the given conditions, the age of b must be 20 years -/
theorem age_of_b (a b c : ℕ) : problem a b c := by
  sorry

end age_of_b_l1965_196549


namespace all_points_enclosable_l1965_196599

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that checks if three points can be enclosed in a circle of radius 1 -/
def enclosableInUnitCircle (p q r : Point) : Prop :=
  ∃ (center : Point), (center.x - p.x)^2 + (center.y - p.y)^2 ≤ 1 ∧
                      (center.x - q.x)^2 + (center.y - q.y)^2 ≤ 1 ∧
                      (center.x - r.x)^2 + (center.y - r.y)^2 ≤ 1

/-- The main theorem -/
theorem all_points_enclosable (n : ℕ) (points : Fin n → Point)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → enclosableInUnitCircle (points i) (points j) (points k)) :
  ∃ (center : Point), ∀ (i : Fin n), (center.x - (points i).x)^2 + (center.y - (points i).y)^2 ≤ 1 :=
sorry

end all_points_enclosable_l1965_196599


namespace bridge_length_l1965_196509

/-- The length of a bridge given train parameters --/
theorem bridge_length
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 100)
  (h2 : crossing_time = 26.997840172786177)
  (h3 : train_speed_kmph = 36) :
  ∃ (bridge_length : ℝ),
    bridge_length = 169.97840172786177 :=
by
  sorry

#check bridge_length

end bridge_length_l1965_196509


namespace square_difference_l1965_196561

theorem square_difference (n : ℝ) : 
  let m : ℝ := 4 * n + 3
  m^2 - 8 * m * n + 16 * n^2 = 9 := by
  sorry

end square_difference_l1965_196561


namespace bill_caroline_age_ratio_l1965_196523

/-- Given the ages of Bill and Caroline, prove their age ratio -/
theorem bill_caroline_age_ratio :
  ∀ (bill_age caroline_age : ℕ),
  bill_age = 17 →
  bill_age + caroline_age = 26 →
  ∃ (n : ℕ), bill_age = n * caroline_age - 1 →
  (bill_age : ℚ) / caroline_age = 17 / 9 := by
  sorry

end bill_caroline_age_ratio_l1965_196523


namespace condition_a_sufficient_not_necessary_l1965_196535

theorem condition_a_sufficient_not_necessary :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end condition_a_sufficient_not_necessary_l1965_196535


namespace epidemic_supplies_theorem_l1965_196580

-- Define the prices of type A and B supplies
def price_A : ℕ := 16
def price_B : ℕ := 4

-- Define the conditions
axiom condition1 : 60 * price_A + 45 * price_B = 1140
axiom condition2 : 45 * price_A + 30 * price_B = 840

-- Define the total units and budget
def total_units : ℕ := 600
def total_budget : ℕ := 8000

-- Define the function to calculate the maximum number of type A units
def max_type_A : ℕ :=
  (total_budget - price_B * total_units) / (price_A - price_B)

-- Theorem to prove
theorem epidemic_supplies_theorem :
  price_A = 16 ∧ price_B = 4 ∧ max_type_A = 466 :=
sorry

end epidemic_supplies_theorem_l1965_196580


namespace inverse_of_complex_l1965_196593

theorem inverse_of_complex (z : ℂ) (h : z = (1 : ℝ) / 2 + (Real.sqrt 3 / 2) * I) : 
  z⁻¹ = (1 : ℝ) / 2 - (Real.sqrt 3 / 2) * I := by
  sorry

end inverse_of_complex_l1965_196593


namespace student_ratio_proof_l1965_196575

theorem student_ratio_proof (m n : ℕ) (a b : ℝ) (α β : ℝ) 
  (h1 : α = 3 / 4)
  (h2 : β = 19 / 20)
  (h3 : a = α * b)
  (h4 : a = β * (a * m + b * n) / (m + n)) :
  m / n = 8 / 9 := by
  sorry

end student_ratio_proof_l1965_196575


namespace distance_inequality_l1965_196514

/-- Given five points A, B, C, D, E on a plane, 
    the sum of distances AB + CD + DE + EC 
    is less than or equal to 
    the sum of distances AC + AD + AE + BC + BD + BE -/
theorem distance_inequality (A B C D E : EuclideanSpace ℝ (Fin 2)) :
  dist A B + dist C D + dist D E + dist E C ≤ 
  dist A C + dist A D + dist A E + dist B C + dist B D + dist B E := by
  sorry

end distance_inequality_l1965_196514


namespace fiftieth_term_is_199_l1965_196557

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftieth_term_is_199 :
  arithmetic_sequence 3 4 50 = 199 := by
  sorry

end fiftieth_term_is_199_l1965_196557


namespace nested_radical_twenty_l1965_196579

theorem nested_radical_twenty (x : ℝ) (h : x > 0) (eq : x = Real.sqrt (20 + x)) : x = 5 := by
  sorry

end nested_radical_twenty_l1965_196579


namespace angle_inequality_theorem_l1965_196581

theorem angle_inequality_theorem (θ : Real) : 
  (π / 2 < θ ∧ θ < π) ↔ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ + x * (1 - x) - (1 - x)^3 * Real.sin θ < 0) ∧
  (∀ φ : Real, 0 ≤ φ ∧ φ ≤ 2*π ∧ φ ≠ θ → 
    ∃ y : Real, 0 ≤ y ∧ y ≤ 1 ∧ y^3 * Real.cos φ + y * (1 - y) - (1 - y)^3 * Real.sin φ ≥ 0) :=
by sorry

end angle_inequality_theorem_l1965_196581


namespace equation_one_solutions_l1965_196541

theorem equation_one_solutions (x : ℝ) :
  x - 2 = 4 * (x - 2)^2 ↔ x = 2 ∨ x = 9/4 := by
sorry

end equation_one_solutions_l1965_196541


namespace john_writing_speed_l1965_196577

/-- The number of books John writes -/
def num_books : ℕ := 3

/-- The number of pages in each book -/
def pages_per_book : ℕ := 400

/-- The number of days it takes John to write the books -/
def total_days : ℕ := 60

/-- The number of pages John writes per day -/
def pages_per_day : ℕ := (num_books * pages_per_book) / total_days

theorem john_writing_speed : pages_per_day = 20 := by
  sorry

end john_writing_speed_l1965_196577


namespace triangle_division_possible_l1965_196595

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  area : Nat

/-- Represents the entire triangle -/
structure Triangle where
  parts : List TrianglePart

/-- The sum of numbers in a triangle part -/
def sumPart (part : TrianglePart) : Nat :=
  part.numbers.sum

/-- The total sum of all numbers in the triangle -/
def totalSum (triangle : Triangle) : Nat :=
  triangle.parts.map sumPart |>.sum

/-- Check if all parts have equal sums -/
def equalSums (triangle : Triangle) : Prop :=
  ∀ i j, i < triangle.parts.length → j < triangle.parts.length → 
    sumPart (triangle.parts.get ⟨i, by sorry⟩) = sumPart (triangle.parts.get ⟨j, by sorry⟩)

/-- Check if all parts have different areas -/
def differentAreas (triangle : Triangle) : Prop :=
  ∀ i j, i < triangle.parts.length → j < triangle.parts.length → i ≠ j → 
    (triangle.parts.get ⟨i, by sorry⟩).area ≠ (triangle.parts.get ⟨j, by sorry⟩).area

/-- The main theorem -/
theorem triangle_division_possible : 
  ∃ (t : Triangle), totalSum t = 63 ∧ t.parts.length = 3 ∧ equalSums t ∧ differentAreas t :=
sorry

end triangle_division_possible_l1965_196595


namespace solution_set_of_inequality_l1965_196515

theorem solution_set_of_inequality (x : ℝ) : 
  (x - 1) / (x + 3) < 0 ↔ -3 < x ∧ x < 1 := by sorry

end solution_set_of_inequality_l1965_196515


namespace road_graveling_cost_l1965_196592

/-- Calculate the cost of graveling two intersecting roads on a rectangular lawn. -/
theorem road_graveling_cost
  (lawn_length : ℕ)
  (lawn_width : ℕ)
  (road_width : ℕ)
  (gravel_cost_per_sqm : ℕ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 40)
  (h3 : road_width = 10)
  (h4 : gravel_cost_per_sqm = 3) :
  lawn_length * road_width + lawn_width * road_width - road_width * road_width * gravel_cost_per_sqm = 3900 :=
by sorry

end road_graveling_cost_l1965_196592


namespace calculation_proofs_l1965_196565

theorem calculation_proofs :
  (1 - 2^2 / (1/5) * 5 - (-10)^2 - |(-3)| = -123) ∧
  ((-1)^2023 + (-5) * ((-2)^3 + 2) - (-4)^2 / (-1/2) = 61) := by
  sorry

end calculation_proofs_l1965_196565


namespace school_study_sample_size_l1965_196533

/-- Represents a collection of student report cards -/
structure ReportCardCollection where
  total : Nat
  selected : Nat
  h_selected_le_total : selected ≤ total

/-- Defines the sample size of a report card collection -/
def sampleSize (collection : ReportCardCollection) : Nat :=
  collection.selected

/-- Theorem stating that for the given scenario, the sample size is 100 -/
theorem school_study_sample_size :
  ∀ (collection : ReportCardCollection),
    collection.total = 1000 →
    collection.selected = 100 →
    sampleSize collection = 100 := by
  sorry

end school_study_sample_size_l1965_196533


namespace product_zero_in_special_set_l1965_196544

theorem product_zero_in_special_set (n : ℕ) (h : n = 1997) (S : Finset ℝ) 
  (hcard : S.card = n)
  (hsum : ∀ x ∈ S, (S.sum id - x) ∈ S) :
  S.prod id = 0 :=
sorry

end product_zero_in_special_set_l1965_196544


namespace solution_set_f_geq_6_min_value_f_min_value_a_plus_2b_l1965_196547

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for the solution set of f(x) ≥ 6
theorem solution_set_f_geq_6 :
  {x : ℝ | f x ≥ 6} = {x : ℝ | x ≥ 1 ∨ x ≤ -2} :=
sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (m : ℝ), m = 4 ∧ ∀ x, f x ≥ m :=
sorry

-- Theorem for the minimum value of a + 2b
theorem min_value_a_plus_2b :
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a*b + a + 2*b = 4 →
  a + 2*b ≥ 2*Real.sqrt 5 - 2 :=
sorry

end solution_set_f_geq_6_min_value_f_min_value_a_plus_2b_l1965_196547


namespace root_product_simplification_l1965_196512

theorem root_product_simplification (a : ℝ) (ha : 0 < a) :
  (a ^ (1 / Real.sqrt a)) * (a ^ (1 / 3)) = a ^ (5 / 6) :=
by sorry

end root_product_simplification_l1965_196512


namespace hockey_league_games_l1965_196548

/-- The number of games played in a hockey league --/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a league with 12 teams, where each team plays 4 games against every other team, 
    the total number of games played is 264. --/
theorem hockey_league_games :
  total_games 12 4 = 264 := by
  sorry

end hockey_league_games_l1965_196548


namespace hayleys_friends_l1965_196513

def total_stickers : ℕ := 72
def stickers_per_friend : ℕ := 8

theorem hayleys_friends :
  total_stickers / stickers_per_friend = 9 :=
by sorry

end hayleys_friends_l1965_196513


namespace borrowed_amount_with_interest_l1965_196551

/-- Calculates the total amount to be returned given a borrowed amount and an interest rate. -/
def totalAmount (borrowed : ℝ) (interestRate : ℝ) : ℝ :=
  borrowed * (1 + interestRate)

/-- Proves that given a borrowed amount of $100 and an agreed increase of 10%, 
    the total amount to be returned is $110. -/
theorem borrowed_amount_with_interest : 
  totalAmount 100 0.1 = 110 := by
  sorry

end borrowed_amount_with_interest_l1965_196551


namespace diagonals_bisect_if_equal_areas_l1965_196502

/-- A quadrilateral in a 2D plane. -/
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

/-- The area of a triangle given its vertices. -/
noncomputable def triangleArea {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C : V) : ℝ := sorry

/-- Statement that a line segment divides a quadrilateral into two equal areas. -/
def dividesEquallyBy {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) (P Q : V) : Prop :=
  triangleArea q.A P Q + triangleArea Q P q.D = triangleArea q.B P Q + triangleArea Q P q.C

/-- The intersection point of the diagonals of a quadrilateral. -/
noncomputable def diagonalIntersection {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : V := sorry

/-- Statement that a point is the midpoint of a line segment. -/
def isMidpoint {V : Type*} [AddCommGroup V] [Module ℝ V] (M A B : V) : Prop :=
  2 • M = A + B

theorem diagonals_bisect_if_equal_areas {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  dividesEquallyBy q q.A q.C → dividesEquallyBy q q.B q.D →
  let E := diagonalIntersection q
  isMidpoint E q.A q.C ∧ isMidpoint E q.B q.D :=
sorry

end diagonals_bisect_if_equal_areas_l1965_196502


namespace highest_result_l1965_196528

def alice_calc (x : ℕ) : ℕ := x * 3 - 2 + 3

def bob_calc (x : ℕ) : ℕ := (x - 3) * 3 + 4

def carla_calc (x : ℕ) : ℕ := x * 3 + 4 - 2

theorem highest_result (start : ℕ) (h : start = 12) : 
  carla_calc start > alice_calc start ∧ carla_calc start > bob_calc start := by
  sorry

end highest_result_l1965_196528


namespace circle_through_origin_l1965_196576

theorem circle_through_origin (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 3*x + m + 1 = 0 → (x = 0 ∧ y = 0)) → m = -1 :=
by sorry

end circle_through_origin_l1965_196576


namespace union_when_a_is_3_union_equals_B_iff_l1965_196572

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
def B : Set ℝ := Set.Ioo 1 5

-- Statement 1: When a = 3, A ∪ B = (1, 7]
theorem union_when_a_is_3 : 
  A 3 ∪ B = Set.Ioc 1 7 := by sorry

-- Statement 2: A ∪ B = B if and only if a ∈ (2, √7)
theorem union_equals_B_iff (a : ℝ) : 
  A a ∪ B = B ↔ a ∈ Set.Ioo 2 (Real.sqrt 7) := by sorry

end union_when_a_is_3_union_equals_B_iff_l1965_196572


namespace third_generation_tail_length_l1965_196539

/-- The tail length increase factor between generations -/
def increase_factor : ℝ := 1.25

/-- The initial tail length of the first generation in centimeters -/
def initial_length : ℝ := 16

/-- The tail length of a given generation -/
def tail_length (generation : ℕ) : ℝ :=
  initial_length * (increase_factor ^ generation)

/-- Theorem: The tail length of the third generation is 25 cm -/
theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end third_generation_tail_length_l1965_196539


namespace total_time_spent_on_pictures_johns_total_time_is_34_hours_l1965_196526

/-- Calculates the total time spent on drawing and coloring pictures. -/
theorem total_time_spent_on_pictures 
  (num_pictures : ℕ) 
  (drawing_time : ℝ) 
  (coloring_time_reduction : ℝ) : ℝ :=
  let coloring_time := drawing_time * (1 - coloring_time_reduction)
  let time_per_picture := drawing_time + coloring_time
  num_pictures * time_per_picture

/-- Proves that John spends 34 hours on all pictures given the conditions. -/
theorem johns_total_time_is_34_hours : 
  total_time_spent_on_pictures 10 2 0.3 = 34 := by
  sorry

end total_time_spent_on_pictures_johns_total_time_is_34_hours_l1965_196526


namespace calculation_correctness_l1965_196508

theorem calculation_correctness : 
  (4 + (-2) = 2) ∧ 
  (-2 - (-1.5) = -0.5) ∧ 
  (-(-4) + 4 = 8) ∧ 
  (|-6| + |2| ≠ 4) :=
by sorry

end calculation_correctness_l1965_196508


namespace negative_x_squared_to_fourth_negative_x_squared_y_cubed_l1965_196500

-- Problem 1
theorem negative_x_squared_to_fourth (x : ℝ) : (-x^2)^4 = x^8 := by sorry

-- Problem 2
theorem negative_x_squared_y_cubed (x y : ℝ) : (-x^2*y)^3 = -x^6*y^3 := by sorry

end negative_x_squared_to_fourth_negative_x_squared_y_cubed_l1965_196500


namespace hyperbola_equation_l1965_196504

/-
  Define the hyperbola equation
-/
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-
  Define the asymptote equation
-/
def has_asymptote (x y : ℝ) : Prop :=
  y = 2 * x

/-
  Define the parabola equation
-/
def is_parabola (x y : ℝ) : Prop :=
  y^2 = 20 * x

/-
  State the theorem
-/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, is_hyperbola a b x y ∧ has_asymptote x y) →
  (∃ x y : ℝ, is_parabola x y ∧ 
    ((x - 5)^2 + y^2 = a^2 + b^2 ∨ (x + 5)^2 + y^2 = a^2 + b^2)) →
  a^2 = 5 ∧ b^2 = 20 :=
by sorry

end hyperbola_equation_l1965_196504


namespace valid_numbers_count_l1965_196505

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = 2 * c ∧
    b = (a + c) / 2

theorem valid_numbers_count :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_valid_number n) ∧
    s.card = 3 :=
sorry

end valid_numbers_count_l1965_196505


namespace optimal_price_for_profit_l1965_196503

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 400

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 10) * sales_volume x

-- State the theorem
theorem optimal_price_for_profit :
  ∃ x : ℝ, 
    x > 0 ∧ 
    profit x = 2160 ∧ 
    ∀ y : ℝ, y > 0 ∧ profit y = 2160 → sales_volume x ≤ sales_volume y := by
  sorry

end optimal_price_for_profit_l1965_196503


namespace fish_to_rice_value_l1965_196586

/-- Represents the exchange rate between fish and bread -/
def fish_to_bread : ℚ := 3 / 5

/-- Represents the exchange rate between bread and rice -/
def bread_to_rice : ℚ := 5 / 2

/-- Theorem stating that one fish is worth 3/2 bags of rice -/
theorem fish_to_rice_value : 
  (fish_to_bread * bread_to_rice)⁻¹ = 3 / 2 := by sorry

end fish_to_rice_value_l1965_196586


namespace largest_c_for_negative_three_in_range_l1965_196520

-- Define the function g(x)
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), (∃ (x : ℝ), g c' x = -3) → c' ≤ c) ∧
  (∃ (x : ℝ), g 1 x = -3) :=
sorry

end largest_c_for_negative_three_in_range_l1965_196520


namespace simplify_exponent_division_l1965_196529

theorem simplify_exponent_division (x : ℝ) (h : x ≠ 0) : x^6 / x^2 = x^4 := by
  sorry

end simplify_exponent_division_l1965_196529


namespace no_intersection_l1965_196594

/-- The function representing y = |3x + 6| -/
def f (x : ℝ) : ℝ := |3 * x + 6|

/-- The function representing y = -|4x - 3| -/
def g (x : ℝ) : ℝ := -|4 * x - 3|

/-- Theorem stating that there are no intersection points between f and g -/
theorem no_intersection :
  ¬∃ (x : ℝ), f x = g x :=
sorry

end no_intersection_l1965_196594


namespace arithmetic_sequence_remainder_l1965_196566

theorem arithmetic_sequence_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 3 →
  d = 8 →
  aₙ = 347 →
  aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ) / 2) % 8 = 4 :=
by sorry

end arithmetic_sequence_remainder_l1965_196566


namespace two_same_color_points_at_unit_distance_l1965_196588

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to color points
def colorPoint : Point → Color := sorry

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem two_same_color_points_at_unit_distance :
  ∃ (p1 p2 : Point), colorPoint p1 = colorPoint p2 ∧ distance p1 p2 = 1 := by
  sorry

end two_same_color_points_at_unit_distance_l1965_196588


namespace factor_x_squared_minus_64_l1965_196559

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_x_squared_minus_64_l1965_196559


namespace tank_capacity_proof_l1965_196578

/-- The total capacity of a water tank in liters. -/
def tank_capacity : ℝ := 120

/-- The difference in water volume between 70% full and 40% full, in liters. -/
def volume_difference : ℝ := 36

/-- Theorem stating the total capacity of the tank given the volume difference between two fill levels. -/
theorem tank_capacity_proof :
  tank_capacity * (0.7 - 0.4) = volume_difference :=
sorry

end tank_capacity_proof_l1965_196578


namespace coefficient_x3_in_product_l1965_196543

theorem coefficient_x3_in_product : 
  let p1 : Polynomial ℤ := 2 * X^4 + 3 * X^3 - 4 * X^2 + 2
  let p2 : Polynomial ℤ := X^3 - 8 * X + 3
  (p1 * p2).coeff 3 = 41 := by sorry

end coefficient_x3_in_product_l1965_196543


namespace find_c_value_l1965_196582

theorem find_c_value (a b c : ℝ) 
  (eq1 : 2 * a + 3 = 5)
  (eq2 : b - a = 1)
  (eq3 : c = 2 * b) : 
  c = 4 := by sorry

end find_c_value_l1965_196582


namespace right_triangle_sides_l1965_196510

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h1 : a > 0) (h2 : d > 0) (h3 : k > 0) :
  a = 3 ∧ d = 1 ∧ k = 2 →
  (a + k * d) ^ 2 = a ^ 2 + (a + d) ^ 2 :=
by sorry

end right_triangle_sides_l1965_196510
