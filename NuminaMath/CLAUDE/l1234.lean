import Mathlib

namespace NUMINAMATH_CALUDE_chord_sum_l1234_123461

/-- Definition of the circle --/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 20 = 0

/-- The point (1, -1) lies on the circle --/
axiom point_on_circle : circle_equation 1 (-1)

/-- Definition of the longest chord length --/
def longest_chord_length : ℝ := sorry

/-- Definition of the shortest chord length --/
def shortest_chord_length : ℝ := sorry

/-- Theorem: The sum of the longest and shortest chord lengths is 18 --/
theorem chord_sum :
  longest_chord_length + shortest_chord_length = 18 := by sorry

end NUMINAMATH_CALUDE_chord_sum_l1234_123461


namespace NUMINAMATH_CALUDE_envelope_ratio_l1234_123446

theorem envelope_ratio (blue_envelopes : ℕ) (yellow_diff : ℕ) (total_envelopes : ℕ)
  (h1 : blue_envelopes = 14)
  (h2 : yellow_diff = 6)
  (h3 : total_envelopes = 46) :
  ∃ (green_envelopes yellow_envelopes : ℕ),
    yellow_envelopes = blue_envelopes - yellow_diff ∧
    green_envelopes = 3 * yellow_envelopes ∧
    blue_envelopes + yellow_envelopes + green_envelopes = total_envelopes ∧
    green_envelopes / yellow_envelopes = 3 := by
  sorry

end NUMINAMATH_CALUDE_envelope_ratio_l1234_123446


namespace NUMINAMATH_CALUDE_power_difference_equals_eight_l1234_123483

theorem power_difference_equals_eight : 4^2 - 2^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equals_eight_l1234_123483


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l1234_123440

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 3| + |x - 5| ≥ 4} = {x : ℝ | x ≤ 2 ∨ x ≥ 6} := by sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l1234_123440


namespace NUMINAMATH_CALUDE_bookcase_weight_excess_l1234_123436

theorem bookcase_weight_excess :
  let bookcase_limit : ℝ := 80
  let hardcover_count : ℕ := 70
  let hardcover_weight : ℝ := 0.5
  let textbook_count : ℕ := 30
  let textbook_weight : ℝ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℝ := 6
  let total_weight := hardcover_count * hardcover_weight +
                      textbook_count * textbook_weight +
                      knickknack_count * knickknack_weight
  total_weight - bookcase_limit = 33 := by
sorry

end NUMINAMATH_CALUDE_bookcase_weight_excess_l1234_123436


namespace NUMINAMATH_CALUDE_biff_wifi_cost_l1234_123478

/-- Proves the hourly cost of WiFi for Biff to break even on a 3-hour bus trip -/
theorem biff_wifi_cost (ticket : ℝ) (snacks : ℝ) (headphones : ℝ) (hourly_rate : ℝ) 
  (trip_duration : ℝ) :
  ticket = 11 →
  snacks = 3 →
  headphones = 16 →
  hourly_rate = 12 →
  trip_duration = 3 →
  ∃ (wifi_cost : ℝ),
    wifi_cost = 2 ∧
    trip_duration * hourly_rate = ticket + snacks + headphones + trip_duration * wifi_cost :=
by sorry

end NUMINAMATH_CALUDE_biff_wifi_cost_l1234_123478


namespace NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l1234_123416

-- Define an acute triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

-- State the theorem
theorem angle_measure_in_acute_triangle (t : AcuteTriangle) :
  (t.b^2 + t.c^2 - t.a^2) * Real.tan t.A = t.b * t.c → t.A = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l1234_123416


namespace NUMINAMATH_CALUDE_production_increase_l1234_123432

/-- Calculates the number of units produced today given the previous average, 
    number of days, and new average including today's production. -/
def units_produced_today (prev_avg : ℝ) (prev_days : ℕ) (new_avg : ℝ) : ℝ :=
  (new_avg * (prev_days + 1)) - (prev_avg * prev_days)

/-- Proves that given the conditions, the number of units produced today is 90. -/
theorem production_increase (prev_avg : ℝ) (prev_days : ℕ) (new_avg : ℝ) 
  (h1 : prev_avg = 60)
  (h2 : prev_days = 5)
  (h3 : new_avg = 65) :
  units_produced_today prev_avg prev_days new_avg = 90 := by
  sorry

#eval units_produced_today 60 5 65

end NUMINAMATH_CALUDE_production_increase_l1234_123432


namespace NUMINAMATH_CALUDE_committee_probability_l1234_123412

/-- The probability of selecting exactly 2 boys in a 5-person committee
    chosen randomly from a group of 30 members (12 boys and 18 girls) -/
theorem committee_probability (total : Nat) (boys : Nat) (girls : Nat) (committee_size : Nat) :
  total = 30 →
  boys = 12 →
  girls = 18 →
  committee_size = 5 →
  (Nat.choose boys 2 * Nat.choose girls 3 : ℚ) / Nat.choose total committee_size = 26928 / 71253 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1234_123412


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1234_123426

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (count1 : ℕ)
  (avg1 : ℚ)
  (count2 : ℕ)
  (avg2 : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_count1 : count1 = 2)
  (h_avg1 : avg1 = 4.4)
  (h_count2 : count2 = 2)
  (h_avg2 : avg2 = 3.85) :
  let sum_all := total * avg_all
  let sum1 := count1 * avg1
  let sum2 := count2 * avg2
  let remaining := total - count1 - count2
  let sum_remaining := sum_all - sum1 - sum2
  (sum_remaining / remaining : ℚ) = 3.6 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1234_123426


namespace NUMINAMATH_CALUDE_integer_triplet_solution_l1234_123454

theorem integer_triplet_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 - 2*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_triplet_solution_l1234_123454


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l1234_123498

open Real

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => log x / log 2) x = 1 / (x * log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l1234_123498


namespace NUMINAMATH_CALUDE_inequality_proof_l1234_123472

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / x + 1 / y ≥ 4 / (x + y)) ∧
  (1 / x + 1 / y + 1 / z ≥ 2 / (x + y) + 2 / (y + z) + 2 / (z + x)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1234_123472


namespace NUMINAMATH_CALUDE_papa_worms_correct_l1234_123458

/-- The number of worms Papa bird caught -/
def papa_worms (babies : ℕ) (worms_per_baby_per_day : ℕ) (days : ℕ) 
  (mama_caught : ℕ) (stolen : ℕ) (mama_needs : ℕ) : ℕ :=
  babies * worms_per_baby_per_day * days - ((mama_caught - stolen) + mama_needs)

theorem papa_worms_correct : 
  papa_worms 6 3 3 13 2 34 = 9 := by
  sorry

end NUMINAMATH_CALUDE_papa_worms_correct_l1234_123458


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l1234_123467

theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧
  (8 - x) / x = 8 / 6 ∧
  (6 - y) / y = 8 / 10 →
  x / y = 36 / 35 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l1234_123467


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero_l1234_123497

theorem sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero (π : ℝ) : 
  |Real.sqrt 2 - 1| + (π - 1)^0 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero_l1234_123497


namespace NUMINAMATH_CALUDE_intersection_A_B_l1234_123469

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | 2 * x^2 - 9 * x + 9 ≤ 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1234_123469


namespace NUMINAMATH_CALUDE_only_one_true_proposition_l1234_123496

theorem only_one_true_proposition :
  (∃! n : Fin 4, 
    (n = 0 → (∀ a b : ℝ, a > b ↔ a^2 > b^2)) ∧
    (n = 1 → (∀ a b : ℝ, a > b ↔ a^3 > b^3)) ∧
    (n = 2 → (∀ a b : ℝ, a > b → |a| > |b|)) ∧
    (n = 3 → (∀ a b c : ℝ, a * c^2 ≤ b * c^2 → a > b))) :=
by sorry

end NUMINAMATH_CALUDE_only_one_true_proposition_l1234_123496


namespace NUMINAMATH_CALUDE_g_composition_of_six_l1234_123406

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 2 * x + 1

theorem g_composition_of_six : g (g (g (g 6))) = 23 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_six_l1234_123406


namespace NUMINAMATH_CALUDE_carol_invitation_packs_l1234_123408

/-- The number of friends Carol is sending invitations to -/
def num_friends : ℕ := 12

/-- The number of invitations in each pack -/
def invitations_per_pack : ℕ := 4

/-- The number of packs Carol bought -/
def num_packs : ℕ := num_friends / invitations_per_pack

theorem carol_invitation_packs : num_packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_carol_invitation_packs_l1234_123408


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1234_123457

/-- The radius of the inscribed circle in a right-angled triangle -/
theorem inscribed_circle_radius_right_triangle (a b c r : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  r = (a * b) / (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1234_123457


namespace NUMINAMATH_CALUDE_mall_sales_optimal_profit_l1234_123484

/-- Represents the selling prices and profit calculation for products A and B --/
structure ProductSales where
  cost_price : ℝ
  price_a : ℝ
  price_b : ℝ
  sales_a : ℝ → ℝ
  sales_b : ℝ → ℝ
  profit : ℝ → ℝ

/-- The theorem statement based on the given problem --/
theorem mall_sales_optimal_profit (s : ProductSales) : 
  s.cost_price = 20 ∧ 
  20 * s.price_a + 10 * s.price_b = 840 ∧ 
  10 * s.price_a + 15 * s.price_b = 660 ∧
  s.sales_a 0 = 40 ∧
  (∀ m, s.sales_a m = s.sales_a 0 + 10 * m) ∧
  (∀ m, s.price_a - m ≥ s.price_b) ∧
  (∀ m, s.profit m = (s.price_a - m - s.cost_price) * s.sales_a m + (s.price_b - s.cost_price) * s.sales_b m) →
  s.price_a = 30 ∧ 
  s.price_b = 24 ∧ 
  (∃ m, s.sales_a m = s.sales_b m ∧ 
       s.profit m = 810 ∧ 
       ∀ n, s.profit n ≤ s.profit m) := by
  sorry

end NUMINAMATH_CALUDE_mall_sales_optimal_profit_l1234_123484


namespace NUMINAMATH_CALUDE_sum_of_squares_l1234_123402

/-- Given a matrix N with the specified structure, prove that if N^T N = I, then x^2 + y^2 + z^2 = 47/120 -/
theorem sum_of_squares (x y z : ℝ) : 
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![0, 3*y, 2*z; 2*x, y, -z; 2*x, -y, z]
  (N.transpose * N = 1) → x^2 + y^2 + z^2 = 47/120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1234_123402


namespace NUMINAMATH_CALUDE_total_tips_proof_l1234_123430

/-- Calculates the total tips earned over three days given the tips per customer,
    customer counts for Friday and Sunday, and that Saturday's count is 3 times Friday's. -/
def total_tips (tips_per_customer : ℕ) (friday_customers : ℕ) (sunday_customers : ℕ) : ℕ :=
  let saturday_customers := 3 * friday_customers
  tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

/-- Proves that the total tips earned over three days is $296 -/
theorem total_tips_proof : total_tips 2 28 36 = 296 := by
  sorry

end NUMINAMATH_CALUDE_total_tips_proof_l1234_123430


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1234_123441

theorem fixed_point_of_linear_function (k : ℝ) : 
  2 = k * 1 - k + 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1234_123441


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1234_123462

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1234_123462


namespace NUMINAMATH_CALUDE_all_propositions_true_l1234_123409

theorem all_propositions_true (a b : ℝ) :
  (a > b → a * |a| > b * |b|) ∧
  (a * |a| > b * |b| → a > b) ∧
  (a ≤ b → a * |a| ≤ b * |b|) ∧
  (a * |a| ≤ b * |b| → a ≤ b) :=
sorry

end NUMINAMATH_CALUDE_all_propositions_true_l1234_123409


namespace NUMINAMATH_CALUDE_cubic_log_relationship_l1234_123423

theorem cubic_log_relationship (x : ℝ) :
  (x^3 < 27 → Real.log x / Real.log (1/3) > -1) ∧
  ¬(Real.log x / Real.log (1/3) > -1 → x^3 < 27) :=
sorry

end NUMINAMATH_CALUDE_cubic_log_relationship_l1234_123423


namespace NUMINAMATH_CALUDE_total_amount_calculation_l1234_123425

theorem total_amount_calculation (part1 : ℝ) (part2 : ℝ) (total_interest : ℝ) :
  part1 = 1500.0000000000007 →
  part1 * 0.05 + part2 * 0.06 = 135 →
  part1 + part2 = 2500.000000000000 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l1234_123425


namespace NUMINAMATH_CALUDE_quadratic_sum_l1234_123491

/-- Given a quadratic x^2 - 20x + 36 that can be written as (x + b)^2 + c,
    prove that b + c = -74 -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 36 = (x + b)^2 + c) → b + c = -74 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1234_123491


namespace NUMINAMATH_CALUDE_percentage_error_multiplication_l1234_123400

theorem percentage_error_multiplication : 
  let correct_factor : ℚ := 5 / 3
  let incorrect_factor : ℚ := 3 / 5
  let percentage_error := (correct_factor - incorrect_factor) / correct_factor * 100
  percentage_error = 64 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_multiplication_l1234_123400


namespace NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l1234_123443

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l1234_123443


namespace NUMINAMATH_CALUDE_inequality_proof_l1234_123418

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 / (a^2 + a*b + b^2) + b^3 / (b^2 + b*c + c^2) + c^3 / (c^2 + c*a + a^2) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1234_123418


namespace NUMINAMATH_CALUDE_function_decomposition_l1234_123413

/-- A function is α-periodic if f(x + α) = f(x) for all x -/
def Periodic (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x, f (x + α) = f x

/-- A function is linear if f(x) = ax for some constant a -/
def Linear (f : ℝ → ℝ) : Prop :=
  ∃ a, ∀ x, f x = a * x

theorem function_decomposition (f : ℝ → ℝ) (α β : ℝ) (hα : α ≠ 0)
    (h : ∀ x, f (x + α) = f x + β) :
    ∃ (g h : ℝ → ℝ), Periodic g α ∧ Linear h ∧ ∀ x, f x = g x + h x := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l1234_123413


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1234_123417

theorem inequality_solution_set (x : ℝ) : 
  x^2 - |x - 1| - 1 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1234_123417


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1234_123403

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def B : Set ℝ := {x | 2 < x ∧ x < 5}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem set_operations_and_range (a : ℝ) : 
  (A ∪ B = {x | 2 < x ∧ x ≤ 9}) ∧ 
  (B ∩ C a = ∅ → a ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1234_123403


namespace NUMINAMATH_CALUDE_symmetric_center_of_translated_cosine_l1234_123428

theorem symmetric_center_of_translated_cosine : 
  let f (x : ℝ) := Real.cos (2 * x + π / 4)
  let g (x : ℝ) := f (x - π / 4)
  ∃ (k : ℤ), g ((k : ℝ) * π / 2 + 3 * π / 8) = g (-(k : ℝ) * π / 2 - 3 * π / 8) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_center_of_translated_cosine_l1234_123428


namespace NUMINAMATH_CALUDE_lottery_solution_l1234_123414

def lottery_numbers (A B C D E : ℕ) : Prop :=
  -- Define the five numbers
  let AB := 10 * A + B
  let BC := 10 * B + C
  let CA := 10 * C + A
  let CB := 10 * C + B
  let CD := 10 * C + D
  -- Conditions
  (1 ≤ A) ∧ (A < B) ∧ (B < C) ∧ (C < 9) ∧ (B < D) ∧ (D ≤ 9) ∧
  (AB < BC) ∧ (BC < CA) ∧ (CA < CB) ∧ (CB < CD) ∧
  (AB + BC + CA + CB + CD = 100 * B + 10 * C + C) ∧
  (CA * BC = 1000 * B + 100 * B + 10 * E + C) ∧
  (CA * CD = 1000 * E + 100 * C + 10 * C + D)

theorem lottery_solution :
  ∃! (A B C D E : ℕ), lottery_numbers A B C D E ∧ A = 1 ∧ B = 2 ∧ C = 8 ∧ D = 5 ∧ E = 6 := by
  sorry

end NUMINAMATH_CALUDE_lottery_solution_l1234_123414


namespace NUMINAMATH_CALUDE_lindas_trip_length_l1234_123490

theorem lindas_trip_length :
  ∀ (total_length : ℚ),
  (1 / 4 : ℚ) * total_length + 30 + (1 / 6 : ℚ) * total_length = total_length →
  total_length = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lindas_trip_length_l1234_123490


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1234_123424

theorem constant_term_binomial_expansion :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, k > 0 ∧ k ≤ n + 1 ∧
  (∀ r : ℕ, r ≥ 0 ∧ r ≤ n →
    (Nat.choose n r * (1 : ℚ)) = 0 ∨ (2 * r = n → k = r + 1)) →
  k = 6 ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1234_123424


namespace NUMINAMATH_CALUDE_simplify_sqrt_144000_l1234_123422

theorem simplify_sqrt_144000 : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_144000_l1234_123422


namespace NUMINAMATH_CALUDE_average_mark_is_35_l1234_123447

/-- The average mark obtained by candidates in an examination. -/
def average_mark (total_marks : ℕ) (num_candidates : ℕ) : ℚ :=
  total_marks / num_candidates

/-- Theorem stating that the average mark is 35 given the conditions. -/
theorem average_mark_is_35 :
  average_mark 4200 120 = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_is_35_l1234_123447


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1234_123429

-- Define the distance function
def s (t : ℝ) : ℝ := 3 * t^2 + t

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 6 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1234_123429


namespace NUMINAMATH_CALUDE_cos_A_minus_B_l1234_123407

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1.5) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_minus_B_l1234_123407


namespace NUMINAMATH_CALUDE_square_difference_601_599_l1234_123435

theorem square_difference_601_599 : (601 : ℤ)^2 - (599 : ℤ)^2 = 2400 := by sorry

end NUMINAMATH_CALUDE_square_difference_601_599_l1234_123435


namespace NUMINAMATH_CALUDE_intersection_condition_l1234_123405

/-- The parabola equation: x = -3y^2 - 4y + 10 -/
def parabola (x y : ℝ) : Prop := x = -3 * y^2 - 4 * y + 10

/-- The line equation: x = k -/
def line (x k : ℝ) : Prop := x = k

/-- The condition for exactly one intersection point -/
def unique_intersection (k : ℝ) : Prop :=
  ∃! y, parabola k y

theorem intersection_condition (k : ℝ) :
  unique_intersection k ↔ k = 34 / 3 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l1234_123405


namespace NUMINAMATH_CALUDE_optimal_production_time_l1234_123466

def shaping_time : ℕ := 15
def firing_time : ℕ := 30
def total_items : ℕ := 75
def total_workers : ℕ := 13

def production_time (shaping_workers : ℕ) (firing_workers : ℕ) : ℕ :=
  let shaping_rounds := (total_items + shaping_workers - 1) / shaping_workers
  let firing_rounds := (total_items + firing_workers - 1) / firing_workers
  max (shaping_rounds * shaping_time) (firing_rounds * firing_time)

theorem optimal_production_time :
  ∃ (shaping_workers firing_workers : ℕ),
    shaping_workers + firing_workers = total_workers ∧
    ∀ (s f : ℕ), s + f = total_workers →
      production_time shaping_workers firing_workers ≤ production_time s f ∧
      production_time shaping_workers firing_workers = 325 :=
by sorry

end NUMINAMATH_CALUDE_optimal_production_time_l1234_123466


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_measure_l1234_123476

/-- The measure of one interior angle of a regular decagon in degrees. -/
def regular_decagon_interior_angle : ℝ := 144

/-- Theorem: The measure of one interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle_measure :
  regular_decagon_interior_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_interior_angle_measure_l1234_123476


namespace NUMINAMATH_CALUDE_swap_values_l1234_123449

theorem swap_values (a b : ℕ) (ha : a = 8) (hb : b = 17) :
  ∃ c : ℕ, (c = b) ∧ (b = a) ∧ (a = c) ∧ (a = 17 ∧ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_swap_values_l1234_123449


namespace NUMINAMATH_CALUDE_no_power_of_two_solution_l1234_123474

theorem no_power_of_two_solution : ¬∃ (a b c k : ℕ), 
  a + b + c = 1001 ∧ 27 * a + 14 * b + c = 2^k :=
by sorry

end NUMINAMATH_CALUDE_no_power_of_two_solution_l1234_123474


namespace NUMINAMATH_CALUDE_treats_per_day_l1234_123473

def treat_cost : ℚ := 1 / 10
def total_cost : ℚ := 6
def days_in_month : ℕ := 30

theorem treats_per_day :
  (total_cost / treat_cost) / days_in_month = 2 := by sorry

end NUMINAMATH_CALUDE_treats_per_day_l1234_123473


namespace NUMINAMATH_CALUDE_min_value_theorem_l1234_123486

theorem min_value_theorem (y₁ y₂ y₃ : ℝ) (h_pos₁ : y₁ > 0) (h_pos₂ : y₂ > 0) (h_pos₃ : y₃ > 0)
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 120) :
  y₁^2 + 4 * y₂^2 + 9 * y₃^2 ≥ 14400 / 29 ∧
  (∃ (y₁' y₂' y₃' : ℝ), y₁'^2 + 4 * y₂'^2 + 9 * y₃'^2 = 14400 / 29 ∧
    2 * y₁' + 3 * y₂' + 4 * y₃' = 120 ∧ y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1234_123486


namespace NUMINAMATH_CALUDE_jessica_quarters_l1234_123453

/-- Calculates the number of quarters Jessica has after her sister borrows some. -/
def quarters_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Jessica has 5 quarters remaining. -/
theorem jessica_quarters : quarters_remaining 8 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_l1234_123453


namespace NUMINAMATH_CALUDE_percentage_b_grades_l1234_123419

def scores : List Nat := [92, 81, 68, 88, 82, 63, 79, 70, 85, 99, 59, 67, 84, 90, 75, 61, 87, 65, 86]

def is_b_grade (score : Nat) : Bool :=
  80 ≤ score ∧ score ≤ 84

def count_b_grades (scores : List Nat) : Nat :=
  scores.filter is_b_grade |>.length

theorem percentage_b_grades : 
  (count_b_grades scores : Rat) / (scores.length : Rat) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_b_grades_l1234_123419


namespace NUMINAMATH_CALUDE_hat_cloak_color_probability_l1234_123477

/-- The number of possible hat colors for sixth-graders -/
def num_hat_colors : ℕ := 2

/-- The number of possible cloak colors for seventh-graders -/
def num_cloak_colors : ℕ := 3

/-- The total number of possible color combinations -/
def total_combinations : ℕ := num_hat_colors * num_cloak_colors

/-- The number of combinations where hat and cloak colors are different -/
def different_color_combinations : ℕ := num_hat_colors * (num_cloak_colors - 1)

/-- The probability of hat and cloak colors being different -/
def prob_different_colors : ℚ := different_color_combinations / total_combinations

theorem hat_cloak_color_probability :
  prob_different_colors = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_hat_cloak_color_probability_l1234_123477


namespace NUMINAMATH_CALUDE_triangle_30_60_90_divisible_l1234_123420

/-- A triangle with angles 30°, 60°, and 90° -/
structure Triangle30_60_90 where
  -- We define the triangle using its angles
  angle1 : Real
  angle2 : Real
  angle3 : Real
  angle1_eq : angle1 = 30
  angle2_eq : angle2 = 60
  angle3_eq : angle3 = 90
  sum_angles : angle1 + angle2 + angle3 = 180

/-- A representation of three equal triangles -/
structure ThreeEqualTriangles where
  -- We define three triangles and their equality
  triangle1 : Triangle30_60_90
  triangle2 : Triangle30_60_90
  triangle3 : Triangle30_60_90
  equality12 : triangle1 = triangle2
  equality23 : triangle2 = triangle3

/-- Theorem stating that a 30-60-90 triangle can be divided into three equal triangles -/
theorem triangle_30_60_90_divisible (t : Triangle30_60_90) : 
  ∃ (et : ThreeEqualTriangles), True :=
sorry

end NUMINAMATH_CALUDE_triangle_30_60_90_divisible_l1234_123420


namespace NUMINAMATH_CALUDE_train_distance_l1234_123464

/-- Proves that a train traveling at a rate of 1 mile per 1.5 minutes will cover 40 miles in 60 minutes -/
theorem train_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 1 / 1.5 → time = 60 → distance = rate * time → distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l1234_123464


namespace NUMINAMATH_CALUDE_function_extrema_l1234_123404

open Real

theorem function_extrema (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := fun x => (sin x + a) / sin x
  ∃ (m : ℝ), (∀ x, 0 < x → x < π → f x ≥ m) ∧
  (∀ M : ℝ, ∃ x, 0 < x ∧ x < π ∧ f x > M) := by
  sorry

end NUMINAMATH_CALUDE_function_extrema_l1234_123404


namespace NUMINAMATH_CALUDE_chessboard_zero_condition_l1234_123427

/-- Represents a chessboard with natural numbers -/
def Chessboard (m n : ℕ) := Fin m → Fin n → ℕ

/-- Sums the numbers on black squares of a chessboard -/
def sumBlack (board : Chessboard m n) : ℕ := sorry

/-- Sums the numbers on white squares of a chessboard -/
def sumWhite (board : Chessboard m n) : ℕ := sorry

/-- Represents an allowed move on the chessboard -/
def allowedMove (board : Chessboard m n) (i j : Fin m) (k l : Fin n) (value : ℤ) : Chessboard m n := sorry

/-- Predicate to check if all numbers on the board are zero -/
def allZero (board : Chessboard m n) : Prop := ∀ i j, board i j = 0

/-- Predicate to check if a board can be reduced to all zeros using allowed moves -/
def canReduceToZero (board : Chessboard m n) : Prop := sorry

theorem chessboard_zero_condition {m n : ℕ} (board : Chessboard m n) :
  canReduceToZero board ↔ sumBlack board = sumWhite board := by sorry

end NUMINAMATH_CALUDE_chessboard_zero_condition_l1234_123427


namespace NUMINAMATH_CALUDE_element_in_set_l1234_123479

theorem element_in_set : 
  let M : Set ℕ := {0, 1, 2}
  let a : ℕ := 0
  a ∈ M :=
by sorry

end NUMINAMATH_CALUDE_element_in_set_l1234_123479


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1234_123452

theorem quadratic_equation_root_zero (m : ℝ) : 
  (m - 1 ≠ 0) →
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) →
  ((m - 1) * 0^2 + 2 * 0 + m^2 - 1 = 0) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l1234_123452


namespace NUMINAMATH_CALUDE_triangle_inequality_l1234_123480

theorem triangle_inequality (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a₁ ≥ 0 → b₁ ≥ 0 → c₁ ≥ 0 →
  a₂ ≥ 0 → b₂ ≥ 0 → c₂ ≥ 0 →
  a + b > c → b + c > a → c + a > b →
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1234_123480


namespace NUMINAMATH_CALUDE_horner_method_innermost_polynomial_l1234_123401

def f (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

def horner_v1 (a : ℝ) : ℝ := 1 * a + 1

theorem horner_method_innermost_polynomial :
  horner_v1 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_innermost_polynomial_l1234_123401


namespace NUMINAMATH_CALUDE_photo_ratio_theorem_l1234_123471

/-- Represents the number of photos in various scenarios --/
structure PhotoCounts where
  initial : ℕ  -- Initial number of photos in the gallery
  firstDay : ℕ  -- Number of photos taken on the first day
  secondDay : ℕ  -- Number of photos taken on the second day
  final : ℕ  -- Final number of photos in the gallery

/-- Theorem stating the ratio of first day photos to initial gallery photos --/
theorem photo_ratio_theorem (p : PhotoCounts) 
  (h1 : p.initial = 400)
  (h2 : p.secondDay = p.firstDay + 120)
  (h3 : p.final = 920)
  (h4 : p.final = p.initial + p.firstDay + p.secondDay) :
  p.firstDay * 2 = p.initial := by
  sorry

#check photo_ratio_theorem

end NUMINAMATH_CALUDE_photo_ratio_theorem_l1234_123471


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1234_123475

theorem cone_lateral_surface_area 
  (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) (S : ℝ) :
  r = 3 →
  V = 12 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  S = Real.pi * r * l →
  S = 15 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1234_123475


namespace NUMINAMATH_CALUDE_basketball_game_scores_l1234_123489

/-- Represents the quarterly scores of a team -/
structure QuarterlyScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an arithmetic sequence -/
def is_arithmetic_sequence (s : QuarterlyScores) : Prop :=
  ∃ d : ℕ, d > 0 ∧ 
    s.q2 = s.q1 + d ∧
    s.q3 = s.q2 + d ∧
    s.q4 = s.q3 + d

/-- Checks if the scores form a geometric sequence -/
def is_geometric_sequence (s : QuarterlyScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧
    s.q2 = s.q1 * r ∧
    s.q3 = s.q2 * r ∧
    s.q4 = s.q3 * r

/-- The main theorem -/
theorem basketball_game_scores 
  (tigers lions : QuarterlyScores)
  (h1 : tigers.q1 = lions.q1)  -- Tied at the end of first quarter
  (h2 : is_arithmetic_sequence tigers)
  (h3 : is_geometric_sequence lions)
  (h4 : (tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4) + 2 = 
        (lions.q1 + lions.q2 + lions.q3 + lions.q4))  -- Lions won by 2 points
  (h5 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 100)
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 100)
  : tigers.q1 + tigers.q2 + lions.q1 + lions.q2 = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l1234_123489


namespace NUMINAMATH_CALUDE_class_representative_count_l1234_123485

theorem class_representative_count (male_students female_students : ℕ) :
  male_students = 26 → female_students = 24 →
  male_students + female_students = 50 :=
by sorry

end NUMINAMATH_CALUDE_class_representative_count_l1234_123485


namespace NUMINAMATH_CALUDE_soccer_lineup_combinations_l1234_123481

def num_goalkeepers : ℕ := 3
def num_defenders : ℕ := 5
def num_midfielders : ℕ := 8
def num_forwards : ℕ := 4

theorem soccer_lineup_combinations : 
  num_goalkeepers * num_defenders * num_midfielders * (num_forwards * (num_forwards - 1)) = 1440 :=
by sorry

end NUMINAMATH_CALUDE_soccer_lineup_combinations_l1234_123481


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1234_123431

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + c = (x + a)^2) → c = 5625 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1234_123431


namespace NUMINAMATH_CALUDE_four_holes_when_unfolded_l1234_123456

/-- Represents a rectangular sheet of paper -/
structure Paper :=
  (width : ℝ)
  (height : ℝ)
  (holes : List (ℝ × ℝ))

/-- Represents the state of the paper after folding -/
inductive FoldState
  | Unfolded
  | DiagonalFold
  | HalfFold
  | FinalFold

/-- Represents a folding operation -/
def fold (p : Paper) (state : FoldState) : Paper :=
  sorry

/-- Represents the operation of punching a hole -/
def punchHole (p : Paper) (x : ℝ) (y : ℝ) : Paper :=
  sorry

/-- Represents the unfolding operation -/
def unfold (p : Paper) : Paper :=
  sorry

/-- The main theorem to prove -/
theorem four_holes_when_unfolded (p : Paper) :
  let p1 := fold p FoldState.DiagonalFold
  let p2 := fold p1 FoldState.HalfFold
  let p3 := fold p2 FoldState.FinalFold
  let p4 := punchHole p3 (p.width / 2) (p.height / 2)
  let final := unfold p4
  final.holes.length = 4 :=
sorry

end NUMINAMATH_CALUDE_four_holes_when_unfolded_l1234_123456


namespace NUMINAMATH_CALUDE_broken_line_coverage_coin_covers_broken_line_l1234_123450

/-- A closed broken line in a 2D plane -/
structure ClosedBrokenLine where
  points : Set (ℝ × ℝ)
  is_closed : True  -- Placeholder for the closed property
  length : ℝ

/-- Theorem: Any closed broken line of length 5 can be covered by a circle of radius 1.25 -/
theorem broken_line_coverage (L : ClosedBrokenLine) (h : L.length = 5) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ L.points → dist center p ≤ 1.25 := by
  sorry

/-- Corollary: A coin with diameter > 2.5 can cover a 5 cm closed broken line -/
theorem coin_covers_broken_line (L : ClosedBrokenLine) (h : L.length = 5) 
  (coin_diameter : ℝ) (hd : coin_diameter > 2.5) :
  ∃ (center : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ L.points → dist center p ≤ coin_diameter / 2 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_coverage_coin_covers_broken_line_l1234_123450


namespace NUMINAMATH_CALUDE_stock_decrease_duration_l1234_123495

/-- The number of bicycles the stock decreases each month -/
def monthly_decrease : ℕ := 2

/-- The number of months from January 1 to September 1 -/
def months_jan_to_sep : ℕ := 8

/-- The total decrease in bicycles from January 1 to September 1 -/
def total_decrease : ℕ := 18

/-- The number of months the stock has been decreasing -/
def months_decreasing : ℕ := 1

theorem stock_decrease_duration :
  monthly_decrease * months_decreasing + monthly_decrease * months_jan_to_sep = total_decrease :=
by sorry

end NUMINAMATH_CALUDE_stock_decrease_duration_l1234_123495


namespace NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l1234_123421

/-- The time taken to complete a job when three people work together, given their individual completion times. -/
theorem job_completion_time (t1 t2 t3 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) (h3 : t3 > 0) :
  1 / (1 / t1 + 1 / t2 + 1 / t3) = (t1 * t2 * t3) / (t2 * t3 + t1 * t3 + t1 * t2) :=
by sorry

/-- The specific case of the job completion time for the given problem. -/
theorem specific_job_completion_time :
  1 / (1 / 15 + 1 / 20 + 1 / 25 : ℝ) = 300 / 47 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_specific_job_completion_time_l1234_123421


namespace NUMINAMATH_CALUDE_swimmer_speed_ratio_l1234_123465

theorem swimmer_speed_ratio :
  ∀ (v₁ v₂ : ℝ),
    v₁ > v₂ →
    v₁ > 0 →
    v₂ > 0 →
    (v₁ + v₂) * 3 = 12 →
    (v₁ - v₂) * 6 = 12 →
    v₁ / v₂ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_ratio_l1234_123465


namespace NUMINAMATH_CALUDE_decimal_units_count_l1234_123410

theorem decimal_units_count :
  (∃ n : ℕ, n * (1 / 10 : ℚ) = (19 / 10 : ℚ) ∧ n = 19) ∧
  (∃ m : ℕ, m * (1 / 100 : ℚ) = (8 / 10 : ℚ) ∧ m = 80) :=
by sorry

end NUMINAMATH_CALUDE_decimal_units_count_l1234_123410


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1234_123499

theorem absolute_value_inequality (x : ℝ) : 
  (|5 - x| < 6) ↔ (-1 < x ∧ x < 11) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1234_123499


namespace NUMINAMATH_CALUDE_max_min_sum_of_quadratic_expression_l1234_123415

theorem max_min_sum_of_quadratic_expression (a b : ℝ) 
  (h : a^2 + a*b + b^2 = 3) : 
  let f := fun (x y : ℝ) => x^2 - x*y + y^2
  ∃ (M m : ℝ), (∀ x y, f x y ≤ M ∧ m ≤ f x y) ∧ M + m = 10 := by
sorry

end NUMINAMATH_CALUDE_max_min_sum_of_quadratic_expression_l1234_123415


namespace NUMINAMATH_CALUDE_alex_total_marbles_l1234_123459

/-- The number of marbles each person has -/
structure MarbleCount where
  lorin_black : ℕ
  jimmy_yellow : ℕ
  alex_black : ℕ
  alex_yellow : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.lorin_black = 4 ∧
  m.jimmy_yellow = 22 ∧
  m.alex_black = 2 * m.lorin_black ∧
  m.alex_yellow = m.jimmy_yellow / 2

/-- The theorem stating that Alex has 19 marbles in total -/
theorem alex_total_marbles (m : MarbleCount) 
  (h : marble_problem m) : m.alex_black + m.alex_yellow = 19 := by
  sorry


end NUMINAMATH_CALUDE_alex_total_marbles_l1234_123459


namespace NUMINAMATH_CALUDE_eagle_count_theorem_l1234_123487

/-- The total number of unique types of eagles across all sections of the mountain -/
def total_unique_eagles (lower middle upper overlapping : ℕ) : ℕ :=
  lower + middle + upper - overlapping

/-- Theorem stating that the total number of unique types of eagles is 32 -/
theorem eagle_count_theorem (lower middle upper overlapping : ℕ) 
  (h1 : lower = 12)
  (h2 : middle = 8)
  (h3 : upper = 16)
  (h4 : overlapping = 4) :
  total_unique_eagles lower middle upper overlapping = 32 := by
  sorry

end NUMINAMATH_CALUDE_eagle_count_theorem_l1234_123487


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l1234_123463

theorem largest_integer_for_negative_quadratic : 
  ∃ n : ℤ, n^2 - 11*n + 28 < 0 ∧ 
  ∀ m : ℤ, m^2 - 11*m + 28 < 0 → m ≤ n ∧ 
  n = 6 := by sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l1234_123463


namespace NUMINAMATH_CALUDE_even_quadratic_sum_l1234_123492

/-- A quadratic function f(x) = ax^2 + bx defined on [-1, 2] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The property of f being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The theorem stating that if f is even, then a + b = 1/3 -/
theorem even_quadratic_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f a b x = f a b (-x)) →
  a + b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_even_quadratic_sum_l1234_123492


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1234_123433

theorem no_solution_for_equation : ¬∃ x : ℝ, 1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1234_123433


namespace NUMINAMATH_CALUDE_projection_theorem_l1234_123444

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ := sorry

theorem projection_theorem (Q : Plane) :
  project (7, 1, 8) Q = (6, 3, 2) →
  project (6, 2, 9) Q = (9/2, 5, 9/2) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l1234_123444


namespace NUMINAMATH_CALUDE_divide_decimals_l1234_123451

theorem divide_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_divide_decimals_l1234_123451


namespace NUMINAMATH_CALUDE_definite_integral_x_x_squared_sin_x_l1234_123438

theorem definite_integral_x_x_squared_sin_x : 
  ∫ x in (-1)..1, (x + x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_x_squared_sin_x_l1234_123438


namespace NUMINAMATH_CALUDE_relay_station_problem_l1234_123494

theorem relay_station_problem (x : ℝ) (h : x > 3) : 
  (∃ (slow_speed fast_speed : ℝ),
    slow_speed > 0 ∧ 
    fast_speed > 0 ∧
    fast_speed = 2 * slow_speed ∧
    900 / (x + 1) = slow_speed ∧
    900 / (x - 3) = fast_speed) ↔ 
  2 * (900 / (x + 1)) = 900 / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_relay_station_problem_l1234_123494


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1234_123439

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 20 * x + c = 0) →  -- exactly one solution
  (a + c = 29) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 4 ∧ c = 25) := by              -- conclusion
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1234_123439


namespace NUMINAMATH_CALUDE_math_is_90_average_l1234_123482

/-- Represents the scores in three subjects -/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- Represents the conditions given in the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.physics = 80 ∧
  (s.physics + s.chemistry + s.mathematics) / 3 = 80 ∧
  (s.physics + s.chemistry) / 2 = 70 ∧
  ∃ x, (s.physics + x) / 2 = 90 ∧ (x = s.chemistry ∨ x = s.mathematics)

/-- Theorem stating that mathematics is the subject averaging 90 with physics -/
theorem math_is_90_average (s : Scores) (h : satisfiesConditions s) :
  (s.physics + s.mathematics) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_math_is_90_average_l1234_123482


namespace NUMINAMATH_CALUDE_delta_value_l1234_123445

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ + 5 → Δ = -17 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l1234_123445


namespace NUMINAMATH_CALUDE_may_largest_drop_l1234_123411

/-- Represents the months in the first half of the year -/
inductive Month
| January
| February
| March
| April
| May
| June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | .January  => -1.25
  | .February => 2.75
  | .March    => -0.75
  | .April    => 1.50
  | .May      => -3.00
  | .June     => -1.00

/-- Definition of a price drop -/
def is_price_drop (x : ℝ) : Prop := x < 0

/-- The month with the largest price drop -/
def largest_drop (m : Month) : Prop :=
  ∀ n : Month, is_price_drop (price_change n) →
    price_change n ≥ price_change m

theorem may_largest_drop :
  largest_drop Month.May :=
sorry

end NUMINAMATH_CALUDE_may_largest_drop_l1234_123411


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1234_123493

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1234_123493


namespace NUMINAMATH_CALUDE_p_value_l1234_123442

theorem p_value (p : ℝ) : (∀ x : ℝ, (x - 1) * (x + 2) = x^2 + p*x - 2) → p = 1 := by
  sorry

end NUMINAMATH_CALUDE_p_value_l1234_123442


namespace NUMINAMATH_CALUDE_complex_subtraction_l1234_123468

theorem complex_subtraction (a b : ℂ) (h1 : a = 4 - 2*I) (h2 : b = 3 + 2*I) :
  a - 2*b = -2 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1234_123468


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1234_123434

/-- The circle equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The line equation ax + y - 5 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  a*x + y - 5 = 0

/-- The chord length of the intersection is 4 -/
def chord_length_is_4 (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2

theorem line_circle_intersection (a : ℝ) :
  chord_length_is_4 a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1234_123434


namespace NUMINAMATH_CALUDE_evelyn_family_without_daughters_l1234_123470

/-- Represents the family structure of Evelyn and her descendants -/
structure EvelynFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_daughters : ℕ
  daughters_per_mother : ℕ

/-- The actual family structure of Evelyn -/
def evelyn_family : EvelynFamily :=
  { daughters := 8,
    granddaughters := 36 - 8,
    daughters_with_daughters := (36 - 8) / 7,
    daughters_per_mother := 7 }

/-- The number of Evelyn's daughters and granddaughters who have no daughters -/
def women_without_daughters (f : EvelynFamily) : ℕ :=
  (f.daughters - f.daughters_with_daughters) + f.granddaughters

theorem evelyn_family_without_daughters :
  women_without_daughters evelyn_family = 32 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_family_without_daughters_l1234_123470


namespace NUMINAMATH_CALUDE_no_real_solutions_l1234_123455

theorem no_real_solutions :
  ¬∃ (x : ℝ), Real.sqrt (x + 16) - 8 / Real.sqrt (x + 16) + 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1234_123455


namespace NUMINAMATH_CALUDE_min_marked_price_for_profit_l1234_123437

theorem min_marked_price_for_profit (num_sets : ℕ) (purchase_price : ℝ) (discount_rate : ℝ) (desired_profit : ℝ) :
  let marked_price := (desired_profit + num_sets * purchase_price) / (num_sets * (1 - discount_rate))
  marked_price ≥ 200 ∧ 
  num_sets * (1 - discount_rate) * marked_price - num_sets * purchase_price ≥ desired_profit ∧
  ∀ x < marked_price, num_sets * (1 - discount_rate) * x - num_sets * purchase_price < desired_profit :=
by sorry

end NUMINAMATH_CALUDE_min_marked_price_for_profit_l1234_123437


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1234_123460

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (a 7 + a 9 = 16) ∧
  (a 4 = 4)

/-- Theorem: For the given arithmetic sequence, a_12 = 12 -/
theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1234_123460


namespace NUMINAMATH_CALUDE_line_intersects_curve_l1234_123448

/-- Given real numbers a and b where ab ≠ 0, the line ax - y + b = 0 intersects
    the curve bx² + ay² = ab. -/
theorem line_intersects_curve (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (x y : ℝ), (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_curve_l1234_123448


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1234_123488

theorem intersection_of_lines :
  ∃! (x y : ℚ), (8 * x - 5 * y = 10) ∧ (3 * x + 2 * y = 16) ∧ 
  (x = 100 / 31) ∧ (y = 98 / 31) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1234_123488
