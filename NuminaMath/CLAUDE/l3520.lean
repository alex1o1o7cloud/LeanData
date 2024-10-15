import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3520_352049

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + y^2 + z^2) * ((x^2 + y^2 + z^2)^2 - (x*y + y*z + z*x)^2) ≥ 
  (x + y + z)^2 * ((x^2 + y^2 + z^2) - (x*y + y*z + z*x))^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3520_352049


namespace NUMINAMATH_CALUDE_intersecting_line_theorem_l3520_352072

/-- Given points A and B, and a line y = ax intersecting segment AB at point C,
    prove that if AC = 2CB, then a = 1 -/
theorem intersecting_line_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, 4)
  ∃ (C : ℝ × ℝ), 
    (C.2 = a * C.1) ∧  -- C is on the line y = ax
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B) ∧  -- C is on segment AB
    ((C.1 - A.1, C.2 - A.2) = (2 * (B.1 - C.1), 2 * (B.2 - C.2)))  -- AC = 2CB
    → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_line_theorem_l3520_352072


namespace NUMINAMATH_CALUDE_range_of_a_l3520_352040

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(B a ⊆ A)) ↔ (1/2 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3520_352040


namespace NUMINAMATH_CALUDE_smallest_five_times_decrease_five_times_decrease_form_no_twelve_times_decrease_divisible_by_k_condition_l3520_352023

def is_valid_number (N : ℕ) : Prop :=
  ∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n)

theorem smallest_five_times_decrease (N : ℕ) :
  is_valid_number N →
  (∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 5 * (N % 10^n)) →
  N ≥ 25 :=
sorry

theorem five_times_decrease_form (N : ℕ) :
  is_valid_number N →
  (∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 5 * (N % 10^n)) →
  ∃ (m : ℕ), N = 12 * 10^m ∨ N = 24 * 10^m ∨ N = 36 * 10^m ∨ N = 48 * 10^m :=
sorry

theorem no_twelve_times_decrease (N : ℕ) :
  is_valid_number N →
  ¬(∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ N = 12 * (N % 10^n)) :=
sorry

theorem divisible_by_k_condition (k : ℕ) :
  (∃ (N : ℕ), is_valid_number N ∧ 
   ∃ (x n : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ N = x * 10^n + (N % 10^n) ∧ k ∣ (N % 10^n)) ↔
  ∃ (x a b : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ a + b > 0 ∧ k = x * 2^a * 5^b :=
sorry

end NUMINAMATH_CALUDE_smallest_five_times_decrease_five_times_decrease_form_no_twelve_times_decrease_divisible_by_k_condition_l3520_352023


namespace NUMINAMATH_CALUDE_particle_movement_ways_l3520_352024

/-- The number of distinct ways a particle can move on a number line -/
def distinct_ways (total_steps : ℕ) (final_distance : ℕ) : ℕ :=
  Nat.choose total_steps ((total_steps + final_distance) / 2) +
  Nat.choose total_steps ((total_steps - final_distance) / 2)

/-- Theorem stating the number of distinct ways for the given conditions -/
theorem particle_movement_ways :
  distinct_ways 10 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_particle_movement_ways_l3520_352024


namespace NUMINAMATH_CALUDE_intern_distribution_l3520_352036

/-- The number of ways to distribute n intern teachers to k freshman classes,
    with each class having at least 1 intern -/
def distribution_plans (n k : ℕ) : ℕ :=
  if n ≥ k then (n - k + 1) else 0

/-- Theorem: There are 4 ways to distribute 5 intern teachers to 4 freshman classes,
    with each class having at least 1 intern -/
theorem intern_distribution : distribution_plans 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_intern_distribution_l3520_352036


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3520_352017

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 1005)
  (h2 : x + 1005 * Real.sin y = 1003)
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) :
  x + y = 1005 + 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3520_352017


namespace NUMINAMATH_CALUDE_equal_cost_mileage_l3520_352092

/-- Represents the cost function for a truck rental company -/
structure RentalCompany where
  baseCost : ℝ
  costPerMile : ℝ

/-- Calculates the total cost for a given mileage -/
def totalCost (company : RentalCompany) (miles : ℝ) : ℝ :=
  company.baseCost + company.costPerMile * miles

/-- Theorem: The mileage at which all three companies have the same cost is 150 miles, 
    and this common cost is $85.45 -/
theorem equal_cost_mileage 
  (safety : RentalCompany)
  (city : RentalCompany)
  (metro : RentalCompany)
  (h1 : safety.baseCost = 41.95 ∧ safety.costPerMile = 0.29)
  (h2 : city.baseCost = 38.95 ∧ city.costPerMile = 0.31)
  (h3 : metro.baseCost = 44.95 ∧ metro.costPerMile = 0.27) :
  ∃ (m : ℝ), 
    m = 150 ∧ 
    totalCost safety m = totalCost city m ∧
    totalCost city m = totalCost metro m ∧
    totalCost safety m = 85.45 :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_mileage_l3520_352092


namespace NUMINAMATH_CALUDE_election_vote_count_l3520_352053

theorem election_vote_count (votes : List Nat) : 
  votes = [195, 142, 116, 90] →
  votes.length = 4 →
  votes[0]! = 195 →
  votes[0]! - votes[1]! = 53 →
  votes[0]! - votes[2]! = 79 →
  votes[0]! - votes[3]! = 105 →
  votes.sum = 543 := by
sorry

end NUMINAMATH_CALUDE_election_vote_count_l3520_352053


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3520_352082

theorem sqrt_fraction_simplification :
  Real.sqrt (7^2 + 24^2) / Real.sqrt (49 + 16) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3520_352082


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3520_352074

theorem sin_2theta_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/5) : 
  Real.sin (2 * θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3520_352074


namespace NUMINAMATH_CALUDE_burger_cost_is_five_l3520_352091

/-- The cost of a burger meal -/
def burger_meal_cost : ℝ := 9.50

/-- The cost of a kid's meal -/
def kids_meal_cost : ℝ := 5

/-- The cost of french fries -/
def fries_cost : ℝ := 3

/-- The cost of a soft drink -/
def drink_cost : ℝ := 3

/-- The cost of a kid's burger -/
def kids_burger_cost : ℝ := 3

/-- The cost of kid's french fries -/
def kids_fries_cost : ℝ := 2

/-- The cost of a kid's juice box -/
def kids_juice_cost : ℝ := 2

/-- The amount saved by buying meals instead of individual items -/
def savings : ℝ := 10

theorem burger_cost_is_five (burger_cost : ℝ) : burger_cost = 5 :=
  by sorry

end NUMINAMATH_CALUDE_burger_cost_is_five_l3520_352091


namespace NUMINAMATH_CALUDE_probability_of_twin_primes_l3520_352083

/-- The set of prime numbers not exceeding 30 -/
def primes_le_30 : Finset Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

/-- A pair of primes (p, q) is considered a twin prime pair if q = p + 2 -/
def is_twin_prime_pair (p q : Nat) : Prop :=
  p ∈ primes_le_30 ∧ q ∈ primes_le_30 ∧ q = p + 2

/-- The set of twin prime pairs among primes not exceeding 30 -/
def twin_prime_pairs : Finset (Nat × Nat) :=
  {(3, 5), (5, 7), (11, 13), (17, 19)}

theorem probability_of_twin_primes :
  (twin_prime_pairs.card : Rat) / (Nat.choose primes_le_30.card 2 : Rat) = 4 / 45 :=
sorry

end NUMINAMATH_CALUDE_probability_of_twin_primes_l3520_352083


namespace NUMINAMATH_CALUDE_midpoint_xy_product_l3520_352077

/-- Given that C = (3, 5) is the midpoint of AB, where A = (1, 8) and B = (x, y), prove that xy = 10 -/
theorem midpoint_xy_product (x y : ℝ) : 
  (3 : ℝ) = (1 + x) / 2 ∧ (5 : ℝ) = (8 + y) / 2 → x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_xy_product_l3520_352077


namespace NUMINAMATH_CALUDE_evaluate_expression_l3520_352030

theorem evaluate_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  -(6 * (Real.sqrt 2 - Real.sqrt 6 - Real.sqrt 10 - Real.sqrt 14)) / 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3520_352030


namespace NUMINAMATH_CALUDE_median_is_5_probability_l3520_352025

def classCount : ℕ := 9
def selectedClassCount : ℕ := 5
def medianClassNumber : ℕ := 5

def probabilityMedianIs5 : ℚ :=
  (Nat.choose 4 2 * Nat.choose 4 2) / Nat.choose classCount selectedClassCount

theorem median_is_5_probability :
  probabilityMedianIs5 = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_median_is_5_probability_l3520_352025


namespace NUMINAMATH_CALUDE_constant_magnitude_l3520_352012

theorem constant_magnitude (z₁ z₂ : ℂ) (h₁ : Complex.abs z₁ = 5) 
  (h₂ : ∀ θ : ℝ, z₁^2 - z₁ * z₂ * Complex.sin θ + z₂^2 = 0) : 
  Complex.abs z₂ = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_magnitude_l3520_352012


namespace NUMINAMATH_CALUDE_problem_solution_l3520_352015

theorem problem_solution (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t)
  (h2 : y = 5 * t + 6)
  (h3 : x = -2) :
  y = 37 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3520_352015


namespace NUMINAMATH_CALUDE_ninth_term_value_l3520_352087

/-- An arithmetic sequence {aₙ} with sum Sₙ of first n terms -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ S n = n / 2 * (2 * a 1 + (n - 1) * d)

theorem ninth_term_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a S)
  (h_S8 : S 8 = 4 * a 1)
  (h_a7 : a 7 = -2)
  : a 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_value_l3520_352087


namespace NUMINAMATH_CALUDE_new_cards_count_l3520_352021

def cards_per_page : ℕ := 3
def pages_used : ℕ := 6
def old_cards : ℕ := 10

theorem new_cards_count :
  (cards_per_page * pages_used) - old_cards = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_new_cards_count_l3520_352021


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_one_l3520_352080

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem subset_implies_a_leq_one (a : ℝ) : A ⊆ B a → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_one_l3520_352080


namespace NUMINAMATH_CALUDE_cos_equality_problem_l3520_352038

theorem cos_equality_problem (m : ℤ) : 
  0 ≤ m ∧ m ≤ 180 → (Real.cos (m * π / 180) = Real.cos (1234 * π / 180) ↔ m = 154) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l3520_352038


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_l3520_352039

theorem seeds_per_flowerbed (total_seeds : ℕ) (num_flowerbeds : ℕ) 
  (h1 : total_seeds = 45)
  (h2 : num_flowerbeds = 9)
  (h3 : total_seeds % num_flowerbeds = 0) :
  total_seeds / num_flowerbeds = 5 := by
sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_l3520_352039


namespace NUMINAMATH_CALUDE_cubic_fraction_equality_l3520_352005

theorem cubic_fraction_equality : 
  let a : ℚ := 7
  let b : ℚ := 6
  let c : ℚ := 1
  (a^3 + b^3) / (a^2 - a*b + b^2 + c) = 559 / 44 := by sorry

end NUMINAMATH_CALUDE_cubic_fraction_equality_l3520_352005


namespace NUMINAMATH_CALUDE_symmetric_difference_eq_zero_three_l3520_352003

-- Define the function f
def f (n : ℕ) : ℕ := 2 * n + 1

-- Define the sets P and Q
def P : Set ℕ := {1, 2, 3, 4, 5}
def Q : Set ℕ := {3, 4, 5, 6, 7}

-- Define sets A and B
def A : Set ℕ := {n : ℕ | f n ∈ P}
def B : Set ℕ := {n : ℕ | f n ∈ Q}

-- State the theorem
theorem symmetric_difference_eq_zero_three :
  (A ∩ (Set.univ \ B)) ∪ (B ∩ (Set.univ \ A)) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_eq_zero_three_l3520_352003


namespace NUMINAMATH_CALUDE_max_tan_MPN_l3520_352078

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4/25
def C2 (x y θ : ℝ) : Prop := (x - 3 - Real.cos θ)^2 + (y - Real.sin θ)^2 = 1/25

-- Define a point P on C2
def P_on_C2 (x y θ : ℝ) : Prop := C2 x y θ

-- Define tangent points M and N on C1
def tangent_points (xm ym xn yn : ℝ) : Prop := C1 xm ym ∧ C1 xn yn

-- Define the angle MPN
def angle_MPN (xp yp xm ym xn yn : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_tan_MPN :
  ∃ (xp yp θ xm ym xn yn : ℝ),
    P_on_C2 xp yp θ ∧
    tangent_points xm ym xn yn ∧
    (∀ (xp' yp' θ' xm' ym' xn' yn' : ℝ),
      P_on_C2 xp' yp' θ' →
      tangent_points xm' ym' xn' yn' →
      Real.tan (angle_MPN xp yp xm ym xn yn) ≥ Real.tan (angle_MPN xp' yp' xm' ym' xn' yn')) ∧
    Real.tan (angle_MPN xp yp xm ym xn yn) = 4 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_max_tan_MPN_l3520_352078


namespace NUMINAMATH_CALUDE_eight_balls_distribution_l3520_352011

/-- The number of ways to distribute n distinct balls into 3 boxes,
    where box i contains at least i balls -/
def distribution_count (n : ℕ) : ℕ := sorry

/-- Theorem stating that there are 2268 ways to distribute 8 distinct balls
    into 3 boxes numbered 1, 2, and 3, where each box i contains at least i balls -/
theorem eight_balls_distribution : distribution_count 8 = 2268 := by sorry

end NUMINAMATH_CALUDE_eight_balls_distribution_l3520_352011


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l3520_352066

theorem hyperbola_minimum_value (x y : ℝ) :
  x^2 / 4 - y^2 = 1 →
  3 * x^2 - 2 * x * y ≥ 6 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l3520_352066


namespace NUMINAMATH_CALUDE_stream_speed_l3520_352016

/-- Proves that the speed of a stream is 19 kmph given the conditions of the rowing problem -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) : 
  boat_speed = 57 →
  upstream_time = 2 * downstream_time →
  (boat_speed - 19) * (boat_speed + 19) = boat_speed^2 :=
by
  sorry

#eval (57 : ℝ) - 19 -- Expected output: 38
#eval (57 : ℝ) + 19 -- Expected output: 76
#eval (57 : ℝ)^2    -- Expected output: 3249
#eval 38 * 76       -- Expected output: 2888

end NUMINAMATH_CALUDE_stream_speed_l3520_352016


namespace NUMINAMATH_CALUDE_linear_system_solution_l3520_352089

theorem linear_system_solution (x y a : ℝ) : 
  (3 * x + y = a) → 
  (x - 2 * y = 1) → 
  (2 * x + 3 * y = 2) → 
  (a = 3) := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l3520_352089


namespace NUMINAMATH_CALUDE_ball_distribution_count_l3520_352034

/-- Represents a valid distribution of balls into boxes -/
structure BallDistribution where
  x : ℕ
  y : ℕ
  z : ℕ
  sum_eq_7 : x + y + z = 7
  ordered : x ≥ y ∧ y ≥ z

/-- The number of ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes -/
def distributionCount : ℕ := sorry

theorem ball_distribution_count : distributionCount = 8 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_count_l3520_352034


namespace NUMINAMATH_CALUDE_greatest_number_of_teams_l3520_352033

theorem greatest_number_of_teams (num_girls num_boys : ℕ) 
  (h_girls : num_girls = 40)
  (h_boys : num_boys = 32) :
  (∃ k : ℕ, k > 0 ∧ k ∣ num_girls ∧ k ∣ num_boys ∧ 
    ∀ m : ℕ, m > 0 → m ∣ num_girls → m ∣ num_boys → m ≤ k) ↔ 
  Nat.gcd num_girls num_boys = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_of_teams_l3520_352033


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3520_352085

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_binomial_expansion : 
  (Finset.sum (Finset.range 9) (fun k => Nat.choose 8 k * 3^(8-k) * 2^k * (if k = 4 then 1 else 0))) = 90720 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3520_352085


namespace NUMINAMATH_CALUDE_total_cookie_sales_l3520_352096

/-- Represents the sales data for Robyn and Lucy's cookie selling adventure -/
structure CookieSales where
  /-- Sales in the first neighborhood -/
  neighborhood1 : Nat × Nat
  /-- Sales in the second neighborhood -/
  neighborhood2 : Nat × Nat
  /-- Sales in the third neighborhood -/
  neighborhood3 : Nat × Nat
  /-- Total sales in the first park -/
  park1_total : Nat
  /-- Total sales in the second park -/
  park2_total : Nat

/-- Theorem stating the total number of packs sold by Robyn and Lucy -/
theorem total_cookie_sales (sales : CookieSales)
  (h1 : sales.neighborhood1 = (15, 12))
  (h2 : sales.neighborhood2 = (23, 15))
  (h3 : sales.neighborhood3 = (17, 16))
  (h4 : sales.park1_total = 25)
  (h5 : ∃ x y : Nat, x = 2 * y ∧ x + y = sales.park1_total)
  (h6 : sales.park2_total = 35)
  (h7 : ∃ x y : Nat, y = x + 5 ∧ x + y = sales.park2_total) :
  (sales.neighborhood1.1 + sales.neighborhood1.2 +
   sales.neighborhood2.1 + sales.neighborhood2.2 +
   sales.neighborhood3.1 + sales.neighborhood3.2 +
   sales.park1_total + sales.park2_total) = 158 := by
  sorry

end NUMINAMATH_CALUDE_total_cookie_sales_l3520_352096


namespace NUMINAMATH_CALUDE_odd_cube_minus_odd_divisible_by_24_l3520_352057

theorem odd_cube_minus_odd_divisible_by_24 (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^3 - (2*n + 1) = 24 * k := by
sorry

end NUMINAMATH_CALUDE_odd_cube_minus_odd_divisible_by_24_l3520_352057


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3520_352084

theorem absolute_value_inequality (x y : ℝ) (h : x * y < 0) : 
  |x + y| < |x - y| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3520_352084


namespace NUMINAMATH_CALUDE_quarter_count_l3520_352069

/-- Given a sum of $3.35 consisting of quarters and dimes, with a total of 23 coins, 
    prove that the number of quarters is 7. -/
theorem quarter_count (total_value : ℚ) (total_coins : ℕ) (quarter_value dime_value : ℚ) 
  (h1 : total_value = 335/100)
  (h2 : total_coins = 23)
  (h3 : quarter_value = 25/100)
  (h4 : dime_value = 1/10)
  : ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧ 
    quarters * quarter_value + dimes * dime_value = total_value ∧
    quarters = 7 :=
by sorry

end NUMINAMATH_CALUDE_quarter_count_l3520_352069


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3520_352006

theorem rectangle_area_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (x + 3) * (y - 1) = x * y ∧
  (x - 3) * (y + 1.5) = x * y ∧
  x * y = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3520_352006


namespace NUMINAMATH_CALUDE_three_digit_number_sum_l3520_352001

theorem three_digit_number_sum (a b c : ℕ) : 
  (100 * a + 10 * b + c) % 5 = 0 →
  a = 2 * b →
  a * b * c = 40 →
  a + b + c = 11 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_sum_l3520_352001


namespace NUMINAMATH_CALUDE_exam_average_proof_l3520_352007

theorem exam_average_proof :
  let group1_count : ℕ := 15
  let group1_average : ℚ := 75/100
  let group2_count : ℕ := 10
  let group2_average : ℚ := 95/100
  let total_count : ℕ := group1_count + group2_count
  
  (group1_count * group1_average + group2_count * group2_average) / total_count = 83/100 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_proof_l3520_352007


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_distance_l3520_352032

theorem hyperbola_asymptote_distance (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ d : ℝ, d = (2 * b) / Real.sqrt (4 + b^2) ∧ d = Real.sqrt 2) →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_distance_l3520_352032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3520_352079

/-- Given an arithmetic sequence {a_n} with non-zero common difference d,
    if a_2 + a_3 = a_6, then (a_1 + a_2) / (a_3 + a_4 + a_5) = 1/3 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 + a 3 = a 6) :
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3520_352079


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_9_l3520_352009

theorem no_linear_term_implies_m_equals_9 (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x - 3) * (3 * x + m) = a * x^2 + b) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_9_l3520_352009


namespace NUMINAMATH_CALUDE_parallel_line_divides_equally_l3520_352093

-- Define the shaded area
def shaded_area : ℝ := 10

-- Define the distance from MO to the parallel line
def distance_from_MO : ℝ := 2.6

-- Define the function that calculates the area above the parallel line
def area_above (d : ℝ) : ℝ := sorry

-- Theorem statement
theorem parallel_line_divides_equally :
  area_above distance_from_MO = shaded_area / 2 := by sorry

end NUMINAMATH_CALUDE_parallel_line_divides_equally_l3520_352093


namespace NUMINAMATH_CALUDE_bmw_sales_count_l3520_352054

-- Define the total number of cars sold
def total_cars : ℕ := 300

-- Define the percentages of non-BMW cars sold
def volkswagen_percent : ℚ := 10/100
def toyota_percent : ℚ := 25/100
def acura_percent : ℚ := 20/100

-- Define the theorem
theorem bmw_sales_count :
  let non_bmw_percent : ℚ := volkswagen_percent + toyota_percent + acura_percent
  let bmw_percent : ℚ := 1 - non_bmw_percent
  (bmw_percent * total_cars : ℚ) = 135 := by sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l3520_352054


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3520_352062

theorem smallest_integer_with_given_remainders : ∃ M : ℕ,
  (M > 0) ∧
  (M % 4 = 3) ∧
  (M % 5 = 4) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (∀ n : ℕ, n > 0 ∧
    n % 4 = 3 ∧
    n % 5 = 4 ∧
    n % 6 = 5 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 → n ≥ M) ∧
  M = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3520_352062


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3520_352056

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 13) = 15 → x = 212 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3520_352056


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l3520_352071

theorem sum_of_possible_x_values (x z : ℝ) 
  (h1 : (x - z)^2 = 100) 
  (h2 : (z - 12)^2 = 36) : 
  ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 - z)^2 = 100 ∧ 
    (x2 - z)^2 = 100 ∧ 
    (x3 - z)^2 = 100 ∧ 
    (x4 - z)^2 = 100 ∧ 
    (z - 12)^2 = 36 ∧
    x1 + x2 + x3 + x4 = 48 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l3520_352071


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l3520_352029

theorem delta_max_success_ratio 
  (charlie_day1_score charlie_day1_total : ℕ)
  (charlie_day2_score charlie_day2_total : ℕ)
  (delta_day1_score delta_day1_total : ℕ)
  (delta_day2_score delta_day2_total : ℕ)
  (h1 : charlie_day1_score = 200)
  (h2 : charlie_day1_total = 360)
  (h3 : charlie_day2_score = 160)
  (h4 : charlie_day2_total = 240)
  (h5 : delta_day1_score > 0)
  (h6 : delta_day2_score > 0)
  (h7 : delta_day1_total + delta_day2_total = 600)
  (h8 : delta_day1_total ≠ 360)
  (h9 : (delta_day1_score : ℚ) / delta_day1_total < (charlie_day1_score : ℚ) / charlie_day1_total)
  (h10 : (delta_day2_score : ℚ) / delta_day2_total < (charlie_day2_score : ℚ) / charlie_day2_total)
  (h11 : (charlie_day1_score + charlie_day2_score : ℚ) / (charlie_day1_total + charlie_day2_total) = 3/5) :
  (delta_day1_score + delta_day2_score : ℚ) / (delta_day1_total + delta_day2_total) ≤ 166/600 :=
by sorry


end NUMINAMATH_CALUDE_delta_max_success_ratio_l3520_352029


namespace NUMINAMATH_CALUDE_parabola_parameters_correct_l3520_352048

/-- Two parabolas with common focus and passing through two points -/
structure TwoParabolas where
  F : ℝ × ℝ
  P₁ : ℝ × ℝ
  P₂ : ℝ × ℝ
  h₁ : F = (2, 2)
  h₂ : P₁ = (4, 2)
  h₃ : P₂ = (-2, 5)

/-- The parameters of the two parabolas -/
def parabola_parameters (tp : TwoParabolas) : ℝ × ℝ :=
  (2, 3.6)

/-- Theorem stating that the parameters of the two parabolas are 2 and 3.6 -/
theorem parabola_parameters_correct (tp : TwoParabolas) :
  parabola_parameters tp = (2, 3.6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameters_correct_l3520_352048


namespace NUMINAMATH_CALUDE_race_distance_l3520_352013

theorem race_distance (race_length : ℝ) (gap : ℝ) : 
  race_length > 0 → 
  gap > 0 → 
  gap < race_length → 
  let v1 := race_length
  let v2 := race_length - gap
  let v3 := (race_length - gap) * ((race_length - gap) / race_length)
  (race_length - v3) = 19 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3520_352013


namespace NUMINAMATH_CALUDE_ten_cuts_eleven_pieces_l3520_352065

/-- The number of pieces resulting from cutting a log -/
def num_pieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: 10 cuts on a log result in 11 pieces -/
theorem ten_cuts_eleven_pieces : num_pieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ten_cuts_eleven_pieces_l3520_352065


namespace NUMINAMATH_CALUDE_equipment_productivity_increase_l3520_352008

/-- Represents the productivity increase factor of the equipment -/
def productivity_increase : ℝ := 4

/-- Represents the time taken by the first worker to complete the job -/
def first_worker_time : ℝ := 8

/-- Represents the time taken by the second worker to complete the job -/
def second_worker_time : ℝ := 5

/-- Represents the setup time for the second worker -/
def setup_time : ℝ := 2

/-- Represents the time after which the second worker processes as many parts as the first worker -/
def equal_parts_time : ℝ := 1

theorem equipment_productivity_increase :
  (∃ (r : ℝ),
    r > 0 ∧
    r * first_worker_time = productivity_increase * r * (second_worker_time - setup_time) ∧
    r * (setup_time + equal_parts_time) = productivity_increase * r * equal_parts_time) :=
by
  sorry

#check equipment_productivity_increase

end NUMINAMATH_CALUDE_equipment_productivity_increase_l3520_352008


namespace NUMINAMATH_CALUDE_f_3_range_l3520_352035

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem f_3_range (a b : ℝ) :
  (-1 ≤ f a b 1 ∧ f a b 1 ≤ 2) →
  (1 ≤ f a b 2 ∧ f a b 2 ≤ 3) →
  -3 ≤ f a b 3 ∧ f a b 3 ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_f_3_range_l3520_352035


namespace NUMINAMATH_CALUDE_three_digit_prime_not_divisor_of_permutation_l3520_352052

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_permutation (a b : ℕ) : Prop :=
  ∃ (x y z : ℕ), a = 100*x + 10*y + z ∧ b = 100*y + 10*z + x ∨
                  a = 100*x + 10*y + z ∧ b = 100*z + 10*x + y ∨
                  a = 100*x + 10*y + z ∧ b = 100*y + 10*x + z ∨
                  a = 100*x + 10*y + z ∧ b = 100*z + 10*y + x ∨
                  a = 100*x + 10*y + z ∧ b = 100*x + 10*z + y

theorem three_digit_prime_not_divisor_of_permutation (p : ℕ) (h1 : is_three_digit p) (h2 : Nat.Prime p) :
  ∀ n : ℕ, is_permutation p n → ¬(n % p = 0) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_prime_not_divisor_of_permutation_l3520_352052


namespace NUMINAMATH_CALUDE_compared_same_type_as_reference_l3520_352047

/-- Two expressions are of the same type if they have the same variables with the same exponents -/
def same_type (e1 e2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ a b, ∃ k : ℚ, e1 a b = k * e2 a b

/-- The reference expression a^2 * b -/
def reference (a b : ℕ) : ℚ := (a^2 : ℚ) * b

/-- The expression to be compared: -2/5 * b * a^2 -/
def compared (a b : ℕ) : ℚ := -(2/5 : ℚ) * b * (a^2 : ℚ)

/-- Theorem stating that the compared expression is of the same type as the reference -/
theorem compared_same_type_as_reference : same_type compared reference := by
  sorry

end NUMINAMATH_CALUDE_compared_same_type_as_reference_l3520_352047


namespace NUMINAMATH_CALUDE_pie_shop_earnings_l3520_352044

def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

theorem pie_shop_earnings : 
  price_per_slice * slices_per_pie * number_of_pies = 180 := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_earnings_l3520_352044


namespace NUMINAMATH_CALUDE_conference_games_l3520_352094

/-- Calculates the total number of games in a sports conference season -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * teams_per_division * (teams_per_division - 1) / 2 * intra_division_games
  let inter_division_total := total_teams * (total_teams - teams_per_division) / 2 * inter_division_games
  intra_division_total + inter_division_total

/-- The theorem stating the total number of games in the specific conference setup -/
theorem conference_games : total_games 18 9 3 2 = 378 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_l3520_352094


namespace NUMINAMATH_CALUDE_min_c_value_l3520_352075

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < 2*b) (h3 : 2*b < c)
  (h4 : ∃! (x y : ℝ), 3*x + y = 3000 ∧ y = |x - a| + |x - b| + |x - 2*b| + |x - c|) :
  c ≥ 502 ∧ ∃ (a' b' : ℕ+), a' < b' ∧ b' < 2*b' ∧ 2*b' < 502 ∧
    ∃! (x y : ℝ), 3*x + y = 3000 ∧ y = |x - a'| + |x - b'| + |x - 2*b'| + |x - 502| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3520_352075


namespace NUMINAMATH_CALUDE_sum_of_integers_l3520_352070

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3520_352070


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l3520_352099

theorem cube_root_equation_solutions :
  ∀ x : ℝ, (x^(1/3) = 15 / (8 - x^(1/3))) ↔ (x = 27 ∨ x = 125) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l3520_352099


namespace NUMINAMATH_CALUDE_square_equals_double_product_l3520_352018

theorem square_equals_double_product (a : ℤ) (b : ℝ) : 
  0 ≤ b → b < 1 → a^2 = 2*b*(a + b) → b = 0 ∨ b = (-1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_double_product_l3520_352018


namespace NUMINAMATH_CALUDE_minimize_y_l3520_352022

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^3 + (x - b)^3

/-- The theorem stating that (a+b)/2 minimizes y -/
theorem minimize_y (a b : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b ≥ y x a b ∧ x = (a + b) / 2 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l3520_352022


namespace NUMINAMATH_CALUDE_age_problem_l3520_352067

theorem age_problem (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : 4 * a = 3 * b) :
  a + b = 35 := by sorry

end NUMINAMATH_CALUDE_age_problem_l3520_352067


namespace NUMINAMATH_CALUDE_fred_took_233_marbles_l3520_352081

/-- The number of black marbles Fred took from Sara -/
def marbles_taken (initial_black_marbles remaining_black_marbles : ℕ) : ℕ :=
  initial_black_marbles - remaining_black_marbles

/-- Proof that Fred took 233 black marbles from Sara -/
theorem fred_took_233_marbles :
  marbles_taken 792 559 = 233 := by
  sorry

end NUMINAMATH_CALUDE_fred_took_233_marbles_l3520_352081


namespace NUMINAMATH_CALUDE_value_of_m_l3520_352090

theorem value_of_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 4*x + m
  let g : ℝ → ℝ := λ x => x^2 - 2*x + 2*m
  3 * f 3 = g 3 → m = 12 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l3520_352090


namespace NUMINAMATH_CALUDE_remainder_three_pow_244_mod_5_l3520_352042

theorem remainder_three_pow_244_mod_5 : 3^244 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_pow_244_mod_5_l3520_352042


namespace NUMINAMATH_CALUDE_cat_arrangements_l3520_352027

def number_of_arrangements (n : ℕ) : ℕ := Nat.factorial n

theorem cat_arrangements : number_of_arrangements 3 = 6 := by sorry

end NUMINAMATH_CALUDE_cat_arrangements_l3520_352027


namespace NUMINAMATH_CALUDE_complex_power_to_rectangular_l3520_352063

theorem complex_power_to_rectangular : 
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.I * Complex.sin (Real.pi / 6)) ^ 4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_to_rectangular_l3520_352063


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l3520_352055

theorem inequality_not_always_hold (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ c : ℝ, ¬(a * c > b * c) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l3520_352055


namespace NUMINAMATH_CALUDE_slope_of_line_l3520_352050

/-- The slope of a line represented by the equation 4x + 5y = 20 is -4/5 -/
theorem slope_of_line (x y : ℝ) : 4 * x + 5 * y = 20 → (y - 4) / x = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l3520_352050


namespace NUMINAMATH_CALUDE_custom_op_result_l3520_352098

def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

theorem custom_op_result : custom_op (custom_op 8 6) 12 = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l3520_352098


namespace NUMINAMATH_CALUDE_complex_number_problem_l3520_352068

theorem complex_number_problem (z : ℂ) (m n : ℝ) :
  (z.re > 0) →
  (Complex.abs z = 2 * Real.sqrt 5) →
  ((1 + 2 * Complex.I) * z).re = 0 →
  (z ^ 2 + m * z + n = 0) →
  (z = 4 + 2 * Complex.I ∧ m = -8 ∧ n = 20) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3520_352068


namespace NUMINAMATH_CALUDE_kims_candy_bars_l3520_352076

/-- Calculates the number of weeks passed given the number of candy bars saved, 
    the number of candy bars received per week, and the number of weeks between eating candy bars. -/
def weeks_passed (candy_bars_saved : ℕ) (candy_bars_per_week : ℕ) (weeks_between_eating : ℕ) : ℕ :=
  let candy_bars_saved_per_cycle := candy_bars_per_week * weeks_between_eating - 1
  candy_bars_saved / candy_bars_saved_per_cycle * weeks_between_eating

/-- Theorem stating that given the conditions from Kim's candy bar problem, 
    the number of weeks passed is 16. -/
theorem kims_candy_bars : 
  weeks_passed 28 2 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_kims_candy_bars_l3520_352076


namespace NUMINAMATH_CALUDE_backyard_area_l3520_352010

/-- Represents a rectangular backyard with specific walking conditions -/
structure Backyard where
  length : ℝ
  width : ℝ
  length_condition : 25 * length = 1000
  perimeter_condition : 10 * (2 * (length + width)) = 1000

/-- The area of a backyard with the given conditions is 400 square meters -/
theorem backyard_area (b : Backyard) : b.length * b.width = 400 := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l3520_352010


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3520_352097

theorem complex_magnitude_product : Complex.abs ((7 - 24 * Complex.I) * (3 + 4 * Complex.I)) = 125 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3520_352097


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l3520_352058

theorem relay_race_arrangements (total_students : Nat) (boys : Nat) (girls : Nat) 
  (selected_students : Nat) (selected_boys : Nat) (selected_girls : Nat) : 
  total_students = 8 →
  boys = 6 →
  girls = 2 →
  selected_students = 4 →
  selected_boys = 3 →
  selected_girls = 1 →
  (Nat.choose girls selected_girls) * 
  (Nat.choose boys selected_boys) * 
  selected_boys * 
  (Nat.factorial (selected_students - 1)) = 720 := by
sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l3520_352058


namespace NUMINAMATH_CALUDE_cos_eq_neg_mul_sin_at_beta_l3520_352051

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := |cos x| - k * x

theorem cos_eq_neg_mul_sin_at_beta
  (k : ℝ) (hk : k > 0)
  (α β : ℝ) (hα : α > 0) (hβ : β > 0) (hαβ : α < β)
  (hzeros : ∀ x, x > 0 → f k x = 0 ↔ x = α ∨ x = β)
  : cos β = -β * sin β :=
sorry

end NUMINAMATH_CALUDE_cos_eq_neg_mul_sin_at_beta_l3520_352051


namespace NUMINAMATH_CALUDE_monotonicity_intervals_no_increasing_intervals_l3520_352064

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + a^2

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1/x + 2*x - 2*a

theorem monotonicity_intervals (x : ℝ) (h : x > 0) :
  let a := 2
  (f_deriv a x > 0 ↔ (x < (2 - Real.sqrt 2) / 2 ∨ x > (2 + Real.sqrt 2) / 2)) ∧
  (f_deriv a x < 0 ↔ ((2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2)) :=
sorry

theorem no_increasing_intervals (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f_deriv a x ≤ 0) ↔ a ≥ 19/6 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_no_increasing_intervals_l3520_352064


namespace NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_l3520_352037

/-- Represents the percentage of students that are freshmen -/
def freshman_percentage : ℝ := 80

/-- Represents the percentage of freshmen enrolled in liberal arts -/
def liberal_arts_percentage : ℝ := 60

/-- Represents the percentage of liberal arts freshmen who are psychology majors -/
def psychology_percentage : ℝ := 50

/-- Theorem stating the percentage of students who are freshmen psychology majors in liberal arts -/
theorem freshmen_psych_liberal_arts_percentage :
  (freshman_percentage / 100) * (liberal_arts_percentage / 100) * (psychology_percentage / 100) * 100 = 24 := by
  sorry


end NUMINAMATH_CALUDE_freshmen_psych_liberal_arts_percentage_l3520_352037


namespace NUMINAMATH_CALUDE_sum_minimized_at_6_l3520_352019

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = -11
  sum_of_4th_and_6th : a 4 + a 6 = -6

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- The value of n that minimizes the sum of first n terms -/
def minimizing_n (seq : ArithmeticSequence) : ℕ :=
  6

theorem sum_minimized_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n_terms seq (minimizing_n seq) ≤ sum_n_terms seq n :=
sorry

end NUMINAMATH_CALUDE_sum_minimized_at_6_l3520_352019


namespace NUMINAMATH_CALUDE_equation_solution_l3520_352020

theorem equation_solution :
  ∃! x : ℤ, 45 - (28 - (x - (15 - 17))) = 56 :=
by
  -- The unique solution is x = 19
  use 19
  constructor
  · -- Prove that x = 19 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3520_352020


namespace NUMINAMATH_CALUDE_price_after_two_reductions_l3520_352095

/-- Represents the relationship between the initial price, reduction percentage, and final price after two reductions. -/
theorem price_after_two_reductions 
  (initial_price : ℝ) 
  (reduction_percentage : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 2) 
  (h2 : 0 ≤ reduction_percentage ∧ reduction_percentage < 1) :
  final_price = initial_price * (1 - reduction_percentage)^2 :=
by sorry

#check price_after_two_reductions

end NUMINAMATH_CALUDE_price_after_two_reductions_l3520_352095


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l3520_352046

/-- Reflects a point about the line y = x -/
def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Translates a point vertically by a given amount -/
def translate_vertical (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

/-- The main theorem -/
theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -7) →
  (translate_vertical (reflect_about_y_eq_x center) 4) = (-7, 7) := by
  sorry


end NUMINAMATH_CALUDE_circle_reflection_translation_l3520_352046


namespace NUMINAMATH_CALUDE_rain_probability_l3520_352088

theorem rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.35)
  (h2 : p_sunday = 0.45)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.6425 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3520_352088


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3520_352073

-- Define the inequality
def inequality (x m : ℝ) : Prop :=
  (x + 2) / 2 ≥ (2 * x + m) / 3 + 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ 8

-- Theorem statement
theorem inequality_solution_implies_m_value :
  (∀ x, inequality x m ↔ solution_set x) → 2^m = (1 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3520_352073


namespace NUMINAMATH_CALUDE_jenny_chocolate_squares_count_l3520_352014

/-- The number of chocolate squares Jenny ate -/
def jenny_chocolate_squares (mike_chocolate_squares : ℕ) : ℕ :=
  3 * mike_chocolate_squares + 5

/-- The number of candies Mike's friend ate -/
def mikes_friend_candies (mike_candies : ℕ) : ℕ :=
  mike_candies - 10

/-- The number of candies Jenny ate -/
def jenny_candies (mikes_friend_candies : ℕ) : ℕ :=
  2 * mikes_friend_candies

theorem jenny_chocolate_squares_count 
  (mike_chocolate_squares : ℕ) 
  (mike_candies : ℕ) 
  (h1 : mike_chocolate_squares = 20) 
  (h2 : mike_candies = 20) :
  jenny_chocolate_squares mike_chocolate_squares = 65 := by
  sorry

#check jenny_chocolate_squares_count

end NUMINAMATH_CALUDE_jenny_chocolate_squares_count_l3520_352014


namespace NUMINAMATH_CALUDE_expression_value_l3520_352031

theorem expression_value : 
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20) = -1 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3520_352031


namespace NUMINAMATH_CALUDE_min_value_of_ab_l3520_352086

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 8) :
  a * b ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l3520_352086


namespace NUMINAMATH_CALUDE_samson_activity_solution_l3520_352000

/-- Represents the utility function for Samson's activities -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := math * frisbee

/-- Represents the total hours spent on activities -/
def totalHours (math : ℝ) (frisbee : ℝ) : ℝ := math + frisbee

theorem samson_activity_solution :
  ∃ (t : ℝ),
    (utility (10 - t) t = utility (t + 5) (4 - t)) ∧
    (totalHours (10 - t) t ≥ 8) ∧
    (totalHours (t + 5) (4 - t) ≥ 8) ∧
    (t ≥ 0) ∧
    (∀ (s : ℝ),
      (utility (10 - s) s = utility (s + 5) (4 - s)) ∧
      (totalHours (10 - s) s ≥ 8) ∧
      (totalHours (s + 5) (4 - s) ≥ 8) ∧
      (s ≥ 0) →
      s = t) ∧
    t = 0 :=
by sorry

end NUMINAMATH_CALUDE_samson_activity_solution_l3520_352000


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_l3520_352026

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  side : ℝ

/-- The diagonal of an isosceles trapezoid -/
def diagonal (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specified isosceles trapezoid is 13 units -/
theorem isosceles_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := { base1 := 24, base2 := 12, side := 13 }
  diagonal t = 13 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_l3520_352026


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3520_352060

theorem smallest_number_with_given_remainders : ∃ (x : ℕ), 
  x > 0 ∧
  x % 6 = 2 ∧ 
  x % 5 = 3 ∧ 
  x % 7 = 4 ∧
  ∀ (y : ℕ), y > 0 → y % 6 = 2 → y % 5 = 3 → y % 7 = 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3520_352060


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3520_352045

def M : Set ℝ := {x | |x - 1| > 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_of_M_and_N : M ∩ N = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3520_352045


namespace NUMINAMATH_CALUDE_function_equivalence_l3520_352004

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_equivalence : 
  (∀ x : ℝ, f (2 * x) = 6 * x - 1) → 
  (∀ x : ℝ, f x = 3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l3520_352004


namespace NUMINAMATH_CALUDE_sara_added_hundred_pencils_l3520_352028

/-- The number of pencils Sara placed in the drawer -/
def pencils_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Sara added 100 pencils to the drawer -/
theorem sara_added_hundred_pencils :
  pencils_added 115 215 = 100 := by sorry

end NUMINAMATH_CALUDE_sara_added_hundred_pencils_l3520_352028


namespace NUMINAMATH_CALUDE_donnas_earnings_proof_l3520_352061

/-- Calculates Donna's total earnings over 7 days based on her work schedule --/
def donnas_weekly_earnings (dog_walking_rate : ℚ) (dog_walking_hours : ℚ) 
  (card_shop_rate : ℚ) (card_shop_hours : ℚ) (card_shop_days : ℕ)
  (babysitting_rate : ℚ) (babysitting_hours : ℚ) : ℚ :=
  (dog_walking_rate * dog_walking_hours * 7) + 
  (card_shop_rate * card_shop_hours * card_shop_days) + 
  (babysitting_rate * babysitting_hours)

theorem donnas_earnings_proof : 
  donnas_weekly_earnings 10 2 12.5 2 5 10 4 = 305 := by
  sorry

end NUMINAMATH_CALUDE_donnas_earnings_proof_l3520_352061


namespace NUMINAMATH_CALUDE_long_letter_time_ratio_l3520_352002

/-- Represents the letter writing schedule and times for Steve --/
structure LetterWriting where
  days_between_letters : ℕ
  regular_letter_time : ℕ
  time_per_page : ℕ
  long_letter_time : ℕ
  total_pages_per_month : ℕ

/-- Calculates the ratio of time spent per page for the long letter compared to a regular letter --/
def time_ratio (lw : LetterWriting) : ℚ :=
  let regular_letters_per_month := 30 / lw.days_between_letters
  let pages_per_regular_letter := lw.regular_letter_time / lw.time_per_page
  let regular_letter_pages := regular_letters_per_month * pages_per_regular_letter
  let long_letter_pages := lw.total_pages_per_month - regular_letter_pages
  let long_letter_time_per_page := lw.long_letter_time / long_letter_pages
  long_letter_time_per_page / lw.time_per_page

/-- Theorem stating that the ratio of time spent per page for the long letter compared to a regular letter is 2:1 --/
theorem long_letter_time_ratio (lw : LetterWriting) 
  (h1 : lw.days_between_letters = 3)
  (h2 : lw.regular_letter_time = 20)
  (h3 : lw.time_per_page = 10)
  (h4 : lw.long_letter_time = 80)
  (h5 : lw.total_pages_per_month = 24) : 
  time_ratio lw = 2 := by
  sorry


end NUMINAMATH_CALUDE_long_letter_time_ratio_l3520_352002


namespace NUMINAMATH_CALUDE_groceries_expense_l3520_352059

def monthly_salary : ℕ := 20000
def savings_percentage : ℚ := 1/10
def savings_amount : ℕ := 2000
def rent : ℕ := 5000
def milk : ℕ := 1500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 2500

theorem groceries_expense (h1 : savings_amount = monthly_salary * savings_percentage) 
  (h2 : savings_amount = 2000) : 
  monthly_salary - (rent + milk + education + petrol + miscellaneous + savings_amount) = 6500 := by
  sorry

end NUMINAMATH_CALUDE_groceries_expense_l3520_352059


namespace NUMINAMATH_CALUDE_bills_double_pay_threshold_l3520_352041

/-- Proves that Bill starts getting paid double after 40 hours -/
theorem bills_double_pay_threshold (base_rate : ℝ) (double_rate : ℝ) (total_hours : ℝ) (total_pay : ℝ)
  (h1 : base_rate = 20)
  (h2 : double_rate = 2 * base_rate)
  (h3 : total_hours = 50)
  (h4 : total_pay = 1200) :
  ∃ x : ℝ, x = 40 ∧ base_rate * x + double_rate * (total_hours - x) = total_pay :=
by
  sorry

end NUMINAMATH_CALUDE_bills_double_pay_threshold_l3520_352041


namespace NUMINAMATH_CALUDE_total_hours_worked_l3520_352043

/-- 
Given a person works 8 hours per day for 4 days, 
prove that the total number of hours worked is 32.
-/
theorem total_hours_worked (hours_per_day : ℕ) (days_worked : ℕ) : 
  hours_per_day = 8 → days_worked = 4 → hours_per_day * days_worked = 32 := by
sorry

end NUMINAMATH_CALUDE_total_hours_worked_l3520_352043
