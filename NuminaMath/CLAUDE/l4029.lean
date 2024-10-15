import Mathlib

namespace NUMINAMATH_CALUDE_exponential_functional_equation_l4029_402959

theorem exponential_functional_equation 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : 
  ∀ x y : ℝ, (fun x => a^x) x * (fun x => a^x) y = (fun x => a^x) (x + y) :=
by sorry

end NUMINAMATH_CALUDE_exponential_functional_equation_l4029_402959


namespace NUMINAMATH_CALUDE_leftover_value_is_seven_l4029_402962

/-- Calculates the value of leftover coins after pooling and rolling --/
def leftover_value (james_quarters james_dimes rebecca_quarters rebecca_dimes : ℕ) 
  (quarters_per_roll dimes_per_roll : ℕ) : ℚ :=
  let total_quarters := james_quarters + rebecca_quarters
  let total_dimes := james_dimes + rebecca_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

theorem leftover_value_is_seven :
  leftover_value 50 80 170 340 40 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_is_seven_l4029_402962


namespace NUMINAMATH_CALUDE_fraction_equality_l4029_402974

theorem fraction_equality : (1000^2 : ℚ) / (252^2 - 248^2) = 500 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l4029_402974


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l4029_402914

theorem polynomial_root_problem (a b : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 - 3 * Complex.I : ℂ) ^ 3 + a * (2 - 3 * Complex.I : ℂ) ^ 2 - 2 * (2 - 3 * Complex.I : ℂ) + b = 0 →
  a = -1/4 ∧ b = 195/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l4029_402914


namespace NUMINAMATH_CALUDE_min_handshakes_for_35_people_l4029_402913

/-- Represents a handshake graph for a conference. -/
structure ConferenceHandshakes where
  people : ℕ
  min_handshakes_per_person : ℕ
  total_handshakes : ℕ

/-- The minimum number of handshakes for a conference with given parameters. -/
def min_handshakes (c : ConferenceHandshakes) : ℕ := c.total_handshakes

/-- Theorem stating the minimum number of handshakes for the specific conference scenario. -/
theorem min_handshakes_for_35_people : 
  ∀ c : ConferenceHandshakes, 
  c.people = 35 → 
  c.min_handshakes_per_person = 3 → 
  min_handshakes c = 51 := by
  sorry


end NUMINAMATH_CALUDE_min_handshakes_for_35_people_l4029_402913


namespace NUMINAMATH_CALUDE_min_odd_integers_l4029_402960

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 30)
  (sum_abcd : a + b + c + d = 45)
  (sum_all : a + b + c + d + e + f = 62) :
  ∃ (odd_count : ℕ), 
    odd_count ≥ 2 ∧ 
    (∃ (odd_integers : Finset ℤ), 
      odd_integers.card = odd_count ∧
      odd_integers ⊆ {a, b, c, d, e, f} ∧
      ∀ x ∈ odd_integers, Odd x) ∧
    ∀ (other_odd_count : ℕ),
      other_odd_count < odd_count →
      ¬∃ (other_odd_integers : Finset ℤ),
        other_odd_integers.card = other_odd_count ∧
        other_odd_integers ⊆ {a, b, c, d, e, f} ∧
        ∀ x ∈ other_odd_integers, Odd x :=
sorry

end NUMINAMATH_CALUDE_min_odd_integers_l4029_402960


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l4029_402963

theorem unique_root_in_interval (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l4029_402963


namespace NUMINAMATH_CALUDE_carriage_sharing_equation_correct_l4029_402901

/-- Represents the problem of "multiple people sharing a carriage" --/
def carriage_sharing_problem (x : ℕ) : Prop :=
  -- When 3 people share 1 carriage, 2 carriages are left empty
  (x / 3 : ℚ) + 2 = 
  -- When 2 people share 1 carriage, 9 people are left without a carriage
  ((x - 9) / 2 : ℚ)

/-- The equation (x/3) + 2 = (x-9)/2 correctly represents the carriage sharing problem --/
theorem carriage_sharing_equation_correct (x : ℕ) : 
  carriage_sharing_problem x ↔ (x / 3 : ℚ) + 2 = ((x - 9) / 2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_carriage_sharing_equation_correct_l4029_402901


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l4029_402947

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 * x - 3| < 5} = Set.Ioo (-1 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l4029_402947


namespace NUMINAMATH_CALUDE_carnival_rides_l4029_402923

theorem carnival_rides (total_time hours roller_coaster_time tilt_a_whirl_time giant_slide_time : ℕ) 
  (roller_coaster_rides tilt_a_whirl_rides : ℕ) : 
  total_time = hours * 60 →
  roller_coaster_time = 30 →
  tilt_a_whirl_time = 60 →
  giant_slide_time = 15 →
  hours = 4 →
  roller_coaster_rides = 4 →
  tilt_a_whirl_rides = 1 →
  (total_time - (roller_coaster_rides * roller_coaster_time + tilt_a_whirl_rides * tilt_a_whirl_time)) / giant_slide_time = 4 :=
by sorry

end NUMINAMATH_CALUDE_carnival_rides_l4029_402923


namespace NUMINAMATH_CALUDE_douglas_county_x_votes_l4029_402909

/-- The percentage of votes Douglas won in county X -/
def douglas_votes_x : ℝ := 74

/-- The ratio of voters in county X to county Y -/
def voter_ratio : ℝ := 2

/-- The percentage of total votes Douglas won in both counties -/
def douglas_total_percent : ℝ := 66

/-- The percentage of votes Douglas won in county Y -/
def douglas_votes_y : ℝ := 50.00000000000002

theorem douglas_county_x_votes :
  let total_votes := voter_ratio + 1
  let douglas_total_votes := douglas_total_percent / 100 * total_votes
  let douglas_y_votes := douglas_votes_y / 100
  douglas_votes_x / 100 * voter_ratio + douglas_y_votes = douglas_total_votes :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_x_votes_l4029_402909


namespace NUMINAMATH_CALUDE_smallest_with_eight_divisors_l4029_402916

/-- A function that returns the number of distinct positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- Proposition: 24 is the smallest positive integer with exactly eight distinct positive divisors -/
theorem smallest_with_eight_divisors :
  (∀ m : ℕ, m > 0 → m < 24 → numDivisors m ≠ 8) ∧ numDivisors 24 = 8 := by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_divisors_l4029_402916


namespace NUMINAMATH_CALUDE_village_foods_customers_l4029_402920

/-- The number of customers per month for Village Foods --/
def customers_per_month (lettuce_cost tomato_cost total_cost_per_customer total_sales : ℚ) : ℚ :=
  total_sales / total_cost_per_customer

/-- Theorem: Village Foods gets 500 customers per month --/
theorem village_foods_customers :
  customers_per_month 2 2 4 2000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_village_foods_customers_l4029_402920


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4029_402904

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem parallel_vectors_x_value :
  let p : ℝ × ℝ := (2, -3)
  let q : ℝ × ℝ := (x, 6)
  are_parallel p q → x = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4029_402904


namespace NUMINAMATH_CALUDE_fraction_equality_l4029_402977

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 3) 
  (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4029_402977


namespace NUMINAMATH_CALUDE_equation_has_real_root_l4029_402925

theorem equation_has_real_root (M : ℝ) : ∃ x : ℝ, x = M^2 * (x - 1) * (x - 2) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l4029_402925


namespace NUMINAMATH_CALUDE_sail_pressure_velocity_l4029_402917

/-- The pressure-area-velocity relationship for a boat sail -/
theorem sail_pressure_velocity 
  (k : ℝ) 
  (A₁ A₂ V₁ V₂ P₁ P₂ : ℝ) 
  (h1 : P₁ = k * A₁ * V₁^2) 
  (h2 : P₂ = k * A₂ * V₂^2) 
  (h3 : A₁ = 2) 
  (h4 : V₁ = 20) 
  (h5 : P₁ = 5) 
  (h6 : A₂ = 4) 
  (h7 : P₂ = 20) : 
  V₂ = 20 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sail_pressure_velocity_l4029_402917


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_300_l4029_402986

theorem modular_inverse_13_mod_300 :
  ∃ (x : ℕ), x < 300 ∧ (13 * x) % 300 = 1 :=
by
  use 277
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_300_l4029_402986


namespace NUMINAMATH_CALUDE_five_distinct_naturals_product_1000_l4029_402966

theorem five_distinct_naturals_product_1000 :
  ∃ (a b c d e : ℕ), a * b * c * d * e = 1000 ∧
                     a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                     c ≠ d ∧ c ≠ e ∧
                     d ≠ e :=
by
  use 1, 2, 4, 5, 25
  sorry

end NUMINAMATH_CALUDE_five_distinct_naturals_product_1000_l4029_402966


namespace NUMINAMATH_CALUDE_sock_pairs_combinations_l4029_402989

/-- Given 7 pairs of socks, proves that the number of ways to choose 2 socks 
    from different pairs is 84. -/
theorem sock_pairs_combinations (n : ℕ) (h : n = 7) : 
  (2 * n * (2 * n - 2)) / 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_combinations_l4029_402989


namespace NUMINAMATH_CALUDE_f_neg_two_eq_one_fourth_l4029_402987

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_neg_two_eq_one_fourth :
  f (-2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_one_fourth_l4029_402987


namespace NUMINAMATH_CALUDE_grace_total_pennies_l4029_402982

/-- The value of a dime in pennies -/
def dime_value : ℕ := 10

/-- The value of a coin in pennies -/
def coin_value : ℕ := 5

/-- The number of dimes Grace has -/
def grace_dimes : ℕ := 10

/-- The number of coins Grace has -/
def grace_coins : ℕ := 10

/-- The total value of Grace's dimes and coins in pennies -/
def total_value : ℕ := grace_dimes * dime_value + grace_coins * coin_value

theorem grace_total_pennies : total_value = 150 := by sorry

end NUMINAMATH_CALUDE_grace_total_pennies_l4029_402982


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4029_402967

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 11 :=
by
  -- The unique solution is y = 3.5
  use 3.5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4029_402967


namespace NUMINAMATH_CALUDE_salary_increase_l4029_402985

theorem salary_increase (S : ℝ) (savings_rate_year1 savings_rate_year2 savings_ratio : ℝ) :
  savings_rate_year1 = 0.10 →
  savings_rate_year2 = 0.06 →
  savings_ratio = 0.6599999999999999 →
  ∃ (P : ℝ), 
    savings_rate_year2 * S * (1 + P / 100) = savings_ratio * (savings_rate_year1 * S) ∧
    P = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l4029_402985


namespace NUMINAMATH_CALUDE_fruit_rate_proof_l4029_402957

/-- The rate per kg for both apples and mangoes -/
def R : ℝ := 70

/-- The weight of apples purchased in kg -/
def apple_weight : ℝ := 8

/-- The weight of mangoes purchased in kg -/
def mango_weight : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 1190

theorem fruit_rate_proof :
  apple_weight * R + mango_weight * R = total_paid :=
by sorry

end NUMINAMATH_CALUDE_fruit_rate_proof_l4029_402957


namespace NUMINAMATH_CALUDE_fraction_transformation_impossibility_l4029_402928

theorem fraction_transformation_impossibility : ¬∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_impossibility_l4029_402928


namespace NUMINAMATH_CALUDE_odd_prime_divisor_property_l4029_402956

theorem odd_prime_divisor_property (n : ℕ+) : 
  (∀ d : ℕ+, d ∣ n → (d + 1) ∣ (n + 1)) ↔ Nat.Prime n.val ∧ n.val % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_divisor_property_l4029_402956


namespace NUMINAMATH_CALUDE_quadratic_properties_l4029_402926

theorem quadratic_properties (a b c m : ℝ) : 
  a < 0 →
  -2 < m →
  m < -1 →
  a * 1^2 + b * 1 + c = 0 →
  a * m^2 + b * m + c = 0 →
  b < 0 ∧ 
  a + b + c = 0 ∧ 
  a * (m + 1) - b + c > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l4029_402926


namespace NUMINAMATH_CALUDE_fixed_point_of_line_family_l4029_402973

theorem fixed_point_of_line_family (k : ℝ) : 
  (3 * k - 1) * (2 / 7) + (k + 2) * (1 / 7) - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_family_l4029_402973


namespace NUMINAMATH_CALUDE_inequality_proof_l4029_402919

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^3 + b^3) / 2 ≥ ((a^2 + b^2) / 2) * ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4029_402919


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l4029_402951

/-- An arithmetic sequence {aₙ} where a₂ = 2 and a₃ = 4 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 3 = 4 ∧ ∀ n : ℕ, a (n + 1) - a n = a 3 - a 2

theorem tenth_term_of_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 :=
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l4029_402951


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l4029_402997

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l4029_402997


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l4029_402907

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 150 = longer_base
  midline_ratio_condition : (shorter_base + (shorter_base + 150) / 2) / 
    ((shorter_base + 150 + (shorter_base + 150)) / 2) = 3 / 4
  equal_area_condition : ∃ h₁ : ℝ, 
    2 * (1/2 * h₁ * (shorter_base + equal_area_segment)) = 
    1/2 * height * (shorter_base + longer_base)

/-- The main theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  ⌊t.equal_area_segment^2 / 150⌋ = 300 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l4029_402907


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l4029_402980

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l4029_402980


namespace NUMINAMATH_CALUDE_juggler_balls_l4029_402994

theorem juggler_balls (total_jugglers : ℕ) (total_balls : ℕ) 
  (h1 : total_jugglers = 378) 
  (h2 : total_balls = 2268) 
  (h3 : total_balls % total_jugglers = 0) : 
  total_balls / total_jugglers = 6 := by
  sorry

end NUMINAMATH_CALUDE_juggler_balls_l4029_402994


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l4029_402953

/-- A bus driver's compensation problem -/
theorem bus_driver_compensation 
  (regular_rate : ℝ) 
  (overtime_rate_increase : ℝ) 
  (max_regular_hours : ℕ) 
  (total_compensation : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate_increase = 0.75)
  (h3 : max_regular_hours = 40)
  (h4 : total_compensation = 864) :
  ∃ (total_hours : ℕ), 
    total_hours = 48 ∧ 
    (↑max_regular_hours * regular_rate + 
     (↑total_hours - ↑max_regular_hours) * (regular_rate * (1 + overtime_rate_increase)) = 
     total_compensation) :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l4029_402953


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sign_l4029_402906

theorem quadratic_coefficient_sign 
  (a b c : ℝ) 
  (h1 : a + b + c < 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sign_l4029_402906


namespace NUMINAMATH_CALUDE_quadratic_residue_characterization_l4029_402995

theorem quadratic_residue_characterization (a b c : ℕ+) :
  (∀ (p : ℕ) (hp : Prime p) (n : ℤ), 
    (∃ (m : ℤ), n ≡ m^2 [ZMOD p]) → 
    (∃ (k : ℤ), (a.val : ℤ) * n^2 + (b.val : ℤ) * n + (c.val : ℤ) ≡ k^2 [ZMOD p])) ↔
  (∃ (d e : ℤ), (a : ℤ) = d^2 ∧ (b : ℤ) = 2*d*e ∧ (c : ℤ) = e^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_residue_characterization_l4029_402995


namespace NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l4029_402975

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  Real.sin x - Real.cos x = Real.sqrt 2 →
  x = 3 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l4029_402975


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l4029_402932

theorem same_solution_implies_c_value (x c : ℚ) : 
  (3 * x + 5 = 1) ∧ (c * x - 8 = -5) → c = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l4029_402932


namespace NUMINAMATH_CALUDE_fraction_equality_l4029_402937

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4029_402937


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l4029_402998

-- Define the number of balls
def num_balls : ℕ := 10

-- Define the probability of a ball being in its original position after two transpositions
def prob_original_position : ℚ := 18 / 25

-- Theorem statement
theorem expected_balls_in_original_position :
  (num_balls : ℚ) * prob_original_position = 72 / 10 := by
sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l4029_402998


namespace NUMINAMATH_CALUDE_unique_postage_arrangements_l4029_402988

/-- Represents the quantity of stamps for each denomination -/
def stamp_quantities : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

/-- Represents the denominations of stamps available -/
def stamp_denominations : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

/-- The target postage amount -/
def target_postage : Nat := 12

/-- A function to calculate the number of unique arrangements -/
noncomputable def count_arrangements (quantities : List Nat) (denominations : List Nat) (target : Nat) : Nat :=
  sorry  -- Implementation details omitted

/-- Theorem stating that there are 82 unique arrangements -/
theorem unique_postage_arrangements :
  count_arrangements stamp_quantities stamp_denominations target_postage = 82 := by
  sorry

#check unique_postage_arrangements

end NUMINAMATH_CALUDE_unique_postage_arrangements_l4029_402988


namespace NUMINAMATH_CALUDE_wall_bricks_count_l4029_402952

theorem wall_bricks_count (x : ℝ) 
  (h1 : x > 0)  -- Ensure positive number of bricks
  (h2 : (x / 8 + x / 12 - 15) > 0)  -- Ensure positive combined rate
  (h3 : 6 * (x / 8 + x / 12 - 15) = x)  -- Equation from working together for 6 hours
  : x = 360 := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l4029_402952


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_l4029_402969

/-- The marginal function of f -/
def marginal (f : ℕ → ℝ) : ℕ → ℝ := fun x ↦ f (x + 1) - f x

/-- The revenue function -/
def R : ℕ → ℝ := fun x ↦ 300 * x - 2 * x^2

/-- The cost function -/
def C : ℕ → ℝ := fun x ↦ 50 * x + 300

/-- The profit function -/
def p : ℕ → ℝ := fun x ↦ R x - C x

/-- The marginal profit function -/
def Mp : ℕ → ℝ := marginal p

theorem profit_and_marginal_profit (x : ℕ) (h : 1 ≤ x ∧ x ≤ 100) :
  p x = -2 * x^2 + 250 * x - 300 ∧
  Mp x = 248 - 4 * x ∧
  (∃ y : ℕ, 1 ≤ y ∧ y ≤ 100 ∧ p y = 7512 ∧ ∀ z : ℕ, 1 ≤ z ∧ z ≤ 100 → p z ≤ p y) ∧
  (Mp 1 = 244 ∧ ∀ z : ℕ, 1 < z ∧ z ≤ 100 → Mp z ≤ Mp 1) :=
by sorry

#check profit_and_marginal_profit

end NUMINAMATH_CALUDE_profit_and_marginal_profit_l4029_402969


namespace NUMINAMATH_CALUDE_irrational_root_theorem_l4029_402992

theorem irrational_root_theorem (a : ℝ) :
  (¬ (∃ (q : ℚ), a = q)) →
  (∃ (s p : ℤ), a + (a^3 - 6*a) = s ∧ a*(a^3 - 6*a) = p) →
  (a = -1 - Real.sqrt 2 ∨
   a = -Real.sqrt 5 ∨
   a = 1 - Real.sqrt 2 ∨
   a = -1 + Real.sqrt 2 ∨
   a = Real.sqrt 5 ∨
   a = 1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_irrational_root_theorem_l4029_402992


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l4029_402927

/-- Represents the volume ratio of two tanks given specific oil transfer conditions -/
theorem tank_volume_ratio (tank1 tank2 : ℚ) : 
  tank1 > 0 → 
  tank2 > 0 → 
  (3/4 : ℚ) * tank1 = (2/5 : ℚ) * tank2 → 
  tank1 / tank2 = 8/15 := by
  sorry

#check tank_volume_ratio

end NUMINAMATH_CALUDE_tank_volume_ratio_l4029_402927


namespace NUMINAMATH_CALUDE_statement_analysis_l4029_402934

theorem statement_analysis (m n : ℝ) : 
  (∀ m n, m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0) ∧ 
  (∀ m n, m > 0 ∧ n > 0 → m + n > 0) ∧ 
  (∃ m n, m + n > 0 ∧ ¬(m > 0 ∧ n > 0)) :=
by sorry

end NUMINAMATH_CALUDE_statement_analysis_l4029_402934


namespace NUMINAMATH_CALUDE_square_lake_area_l4029_402984

/-- Represents a square lake with a given boat speed and crossing times -/
structure SquareLake where
  boat_speed : ℝ  -- Speed of the boat in miles per hour
  length_time : ℝ  -- Time to cross the length in hours
  width_time : ℝ  -- Time to cross the width in hours

/-- Calculates the area of a square lake based on boat speed and crossing times -/
def lake_area (lake : SquareLake) : ℝ :=
  (lake.boat_speed * lake.length_time) * (lake.boat_speed * lake.width_time)

/-- Theorem: The area of the specified square lake is 100 square miles -/
theorem square_lake_area :
  let lake := SquareLake.mk 10 2 (1/2)
  lake_area lake = 100 := by
  sorry


end NUMINAMATH_CALUDE_square_lake_area_l4029_402984


namespace NUMINAMATH_CALUDE_squirrel_acorns_l4029_402961

theorem squirrel_acorns :
  let num_squirrels : ℕ := 5
  let acorns_needed : ℕ := 130
  let acorns_to_collect : ℕ := 15
  let acorns_per_squirrel : ℕ := acorns_needed - acorns_to_collect
  num_squirrels * acorns_per_squirrel = 575 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l4029_402961


namespace NUMINAMATH_CALUDE_jacket_price_l4029_402996

theorem jacket_price (jacket_count : ℕ) (shorts_count : ℕ) (pants_count : ℕ) 
  (shorts_price : ℚ) (pants_price : ℚ) (total_spent : ℚ) :
  jacket_count = 3 → 
  shorts_count = 2 →
  pants_count = 4 →
  shorts_price = 6 →
  pants_price = 12 →
  total_spent = 90 →
  ∃ (jacket_price : ℚ), 
    jacket_price * jacket_count + shorts_price * shorts_count + pants_price * pants_count = total_spent ∧
    jacket_price = 10 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_l4029_402996


namespace NUMINAMATH_CALUDE_delegates_without_badges_l4029_402930

theorem delegates_without_badges (total : ℕ) (pre_printed : ℚ) (break_fraction : ℚ) (hand_written : ℚ) 
  (h_total : total = 100)
  (h_pre_printed : pre_printed = 1/5)
  (h_break : break_fraction = 3/7)
  (h_hand_written : hand_written = 2/9) :
  ↑total - (↑total * pre_printed).floor - 
  ((↑total - (↑total * pre_printed).floor) * break_fraction).floor - 
  (((↑total - (↑total * pre_printed).floor) - ((↑total - (↑total * pre_printed).floor) * break_fraction).floor) * hand_written).floor = 36 :=
by sorry

end NUMINAMATH_CALUDE_delegates_without_badges_l4029_402930


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l4029_402991

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition "a = 0" is necessary but not sufficient for a complex number z = a + bi to be purely imaginary. -/
theorem a_zero_necessary_not_sufficient :
  (∀ z : ℂ, is_purely_imaginary z → z.re = 0) ∧
  ¬(∀ z : ℂ, z.re = 0 → is_purely_imaginary z) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l4029_402991


namespace NUMINAMATH_CALUDE_base_sum_22_l4029_402933

def F₁ (R : ℕ) : ℚ := (4*R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5*R + 4) / (R^2 - 1)

theorem base_sum_22 (R₁ R₂ : ℕ) : 
  (F₁ R₁ = 0.454545 ∧ F₂ R₁ = 0.545454) →
  (F₁ R₂ = 3 / 10 ∧ F₂ R₂ = 7 / 10) →
  R₁ + R₂ = 22 := by sorry

end NUMINAMATH_CALUDE_base_sum_22_l4029_402933


namespace NUMINAMATH_CALUDE_camel_distribution_count_l4029_402900

def is_valid_camel_distribution (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 99 ∧
  ∀ k : ℕ, k ≤ 62 → k + min (62 - k) (n - k) ≥ (100 + n) / 2

theorem camel_distribution_count :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_camel_distribution n) ∧ S.card = 72 :=
sorry

end NUMINAMATH_CALUDE_camel_distribution_count_l4029_402900


namespace NUMINAMATH_CALUDE_count_quads_with_perimeter_36_l4029_402912

/-- A convex cyclic quadrilateral with integer sides --/
structure ConvexCyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  convex : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c
  cyclic : a * c = b * d

/-- The set of all convex cyclic quadrilaterals with perimeter 36 --/
def QuadsWithPerimeter36 : Set ConvexCyclicQuad :=
  {q : ConvexCyclicQuad | q.a + q.b + q.c + q.d = 36}

/-- Counts the number of distinct quadrilaterals in the set --/
def CountDistinctQuads (s : Set ConvexCyclicQuad) : ℕ :=
  sorry

theorem count_quads_with_perimeter_36 :
  CountDistinctQuads QuadsWithPerimeter36 = 1026 := by
  sorry

end NUMINAMATH_CALUDE_count_quads_with_perimeter_36_l4029_402912


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l4029_402905

theorem consecutive_non_primes (k : ℕ+) : ∃ n : ℕ, ∀ i : ℕ, i < k → ¬ Nat.Prime (n + i) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l4029_402905


namespace NUMINAMATH_CALUDE_frog_safety_probability_l4029_402918

/-- Represents the probability of the frog reaching stone 14 safely when starting from stone n -/
def safe_probability (n : ℕ) : ℚ := sorry

/-- The total number of stones -/
def total_stones : ℕ := 15

/-- The probability of jumping backwards from stone n -/
def back_prob (n : ℕ) : ℚ := (n + 1) / total_stones

/-- The probability of jumping forwards from stone n -/
def forward_prob (n : ℕ) : ℚ := 1 - back_prob n

theorem frog_safety_probability :
  0 < 2 ∧ 2 < 14 →
  (∀ n : ℕ, 0 < n ∧ n < 14 →
    safe_probability n = back_prob n * safe_probability (n - 1) +
                         forward_prob n * safe_probability (n + 1)) →
  safe_probability 0 = 0 →
  safe_probability 14 = 1 →
  safe_probability 2 = 85 / 256 :=
sorry

end NUMINAMATH_CALUDE_frog_safety_probability_l4029_402918


namespace NUMINAMATH_CALUDE_segments_AB_CD_parallel_l4029_402931

-- Define points in 2D space
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, -1)
def C : ℝ × ℝ := (0, 4)
def D : ℝ × ℝ := (2, -4)

-- Define a function to check if two segments are parallel
def are_parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  let v1 := (p2.1 - p1.1, p2.2 - p1.2)
  let v2 := (q2.1 - q1.1, q2.2 - q1.2)
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

-- Theorem statement
theorem segments_AB_CD_parallel :
  are_parallel A B C D := by
  sorry

end NUMINAMATH_CALUDE_segments_AB_CD_parallel_l4029_402931


namespace NUMINAMATH_CALUDE_triangle_folding_theorem_l4029_402970

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A folding method is a function that takes a triangle and produces a set of fold lines -/
def FoldingMethod := Triangle → Set (ℝ × ℝ → ℝ × ℝ)

/-- The result of applying a folding method to a triangle -/
structure FoldedObject where
  original : Triangle
  foldLines : Set (ℝ × ℝ → ℝ × ℝ)
  thickness : ℕ

/-- A folding method is valid if it produces a folded object with uniform thickness -/
def isValidFolding (method : FoldingMethod) : Prop :=
  ∀ t : Triangle, ∃ fo : FoldedObject, 
    fo.original = t ∧ 
    fo.foldLines = method t ∧ 
    fo.thickness = 2020

/-- The main theorem: there exists a valid folding method for any triangle -/
theorem triangle_folding_theorem : ∃ (method : FoldingMethod), isValidFolding method := by
  sorry

end NUMINAMATH_CALUDE_triangle_folding_theorem_l4029_402970


namespace NUMINAMATH_CALUDE_paper_cutting_l4029_402978

theorem paper_cutting (k : ℕ) : 
  (¬ ∃ (n m : ℕ), 1 + 7 * n + 11 * m = 60) ∧
  (k > 60 → ∃ (n m : ℕ), 1 + 7 * n + 11 * m = k) := by
  sorry

end NUMINAMATH_CALUDE_paper_cutting_l4029_402978


namespace NUMINAMATH_CALUDE_sum_a_b_eq_neg_four_l4029_402929

theorem sum_a_b_eq_neg_four (a b : ℝ) (h : |1 - 2*a + b| + 2*a = -a^2 - 1) : 
  a + b = -4 := by sorry

end NUMINAMATH_CALUDE_sum_a_b_eq_neg_four_l4029_402929


namespace NUMINAMATH_CALUDE_equation_solution_l4029_402936

theorem equation_solution : 
  ∀ x : ℝ, 
    (9 / (Real.sqrt (x - 5) - 10) + 
     2 / (Real.sqrt (x - 5) - 5) + 
     8 / (Real.sqrt (x - 5) + 5) + 
     15 / (Real.sqrt (x - 5) + 10) = 0) ↔ 
    (x = 14 ∨ x = 1335 / 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4029_402936


namespace NUMINAMATH_CALUDE_z_value_l4029_402915

theorem z_value (x : ℝ) (z : ℝ) (h1 : 3 * x = 0.75 * z) (h2 : x = 20) : z = 80 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l4029_402915


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l4029_402954

theorem cubic_roots_relation (p q r : ℝ) (u v w : ℝ) : 
  (∀ x, x^3 + 5*x^2 + 6*x - 7 = (x - p) * (x - q) * (x - r)) →
  (∀ x, x^3 + u*x^2 + v*x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p))) →
  w = 37 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l4029_402954


namespace NUMINAMATH_CALUDE_solve_for_a_l4029_402979

-- Define the equations as functions of x
def eq1 (x : ℝ) : Prop := 6 * (x + 8) = 18 * x
def eq2 (a x : ℝ) : Prop := 6 * x - 2 * (a - x) = 2 * a + x

-- State the theorem
theorem solve_for_a : ∃ (a : ℝ), ∃ (x : ℝ), eq1 x ∧ eq2 a x ∧ a = 7 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l4029_402979


namespace NUMINAMATH_CALUDE_line_intercept_sum_l4029_402949

/-- Given a line 3x + 5y + c = 0 where the sum of its x-intercept and y-intercept is 16, prove that c = -30 -/
theorem line_intercept_sum (c : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + c = 0 ∧ x + y = 16) → c = -30 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l4029_402949


namespace NUMINAMATH_CALUDE_plant_arrangement_l4029_402958

theorem plant_arrangement (basil_count : Nat) (tomato_count : Nat) :
  basil_count = 6 →
  tomato_count = 3 →
  (Nat.factorial (basil_count + 1)) * (Nat.factorial tomato_count) = 30240 :=
by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_l4029_402958


namespace NUMINAMATH_CALUDE_sports_enthusiasts_difference_l4029_402941

theorem sports_enthusiasts_difference (total : ℕ) (basketball : ℕ) (football : ℕ)
  (h_total : total = 46)
  (h_basketball : basketball = 23)
  (h_football : football = 29) :
  basketball - (basketball + football - total) = 17 :=
by sorry

end NUMINAMATH_CALUDE_sports_enthusiasts_difference_l4029_402941


namespace NUMINAMATH_CALUDE_line_equation_to_slope_intercept_l4029_402948

/-- Given a line equation, prove it can be expressed in slope-intercept form --/
theorem line_equation_to_slope_intercept :
  ∀ (x y : ℝ),
  3 * (x + 2) - 4 * (y - 8) = 0 →
  y = (3 / 4) * x + (19 / 2) :=
by
  sorry

#check line_equation_to_slope_intercept

end NUMINAMATH_CALUDE_line_equation_to_slope_intercept_l4029_402948


namespace NUMINAMATH_CALUDE_total_spending_is_correct_l4029_402924

def lunch_cost : ℚ := 50.50
def dessert_cost : ℚ := 8.25
def beverage_cost : ℚ := 3.75
def lunch_discount : ℚ := 0.10
def dessert_tax : ℚ := 0.07
def beverage_tax : ℚ := 0.05
def lunch_tip : ℚ := 0.20
def other_tip : ℚ := 0.15

def discounted_lunch : ℚ := lunch_cost * (1 - lunch_discount)
def taxed_dessert : ℚ := dessert_cost * (1 + dessert_tax)
def taxed_beverage : ℚ := beverage_cost * (1 + beverage_tax)

def lunch_tip_amount : ℚ := discounted_lunch * lunch_tip
def other_tip_amount : ℚ := (taxed_dessert + taxed_beverage) * other_tip

def total_spending : ℚ := discounted_lunch + taxed_dessert + taxed_beverage + lunch_tip_amount + other_tip_amount

theorem total_spending_is_correct : total_spending = 69.23 := by sorry

end NUMINAMATH_CALUDE_total_spending_is_correct_l4029_402924


namespace NUMINAMATH_CALUDE_candy_bar_calculation_l4029_402965

theorem candy_bar_calculation :
  let f : ℕ := 12
  let b : ℕ := f + 6
  let j : ℕ := 10 * (f + b)
  (40 : ℚ) / 100 * (j ^ 2 : ℚ) = 36000 := by sorry

end NUMINAMATH_CALUDE_candy_bar_calculation_l4029_402965


namespace NUMINAMATH_CALUDE_solution_pairs_count_l4029_402971

theorem solution_pairs_count : 
  let equation := λ (x y : ℕ) => 4 * x + 7 * y = 600
  ∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => equation p.1 p.2) (Finset.product (Finset.range 601) (Finset.range 601))).card ∧ n = 22 := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_count_l4029_402971


namespace NUMINAMATH_CALUDE_B_inverse_proof_l4029_402940

variable (A B : Matrix (Fin 2) (Fin 2) ℚ)

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; 3, 4]

theorem B_inverse_proof :
  A⁻¹ = A_inv →
  B * A = 1 →
  B⁻¹ = !![(-2), 1; (3/2), (-1/2)] := by sorry

end NUMINAMATH_CALUDE_B_inverse_proof_l4029_402940


namespace NUMINAMATH_CALUDE_absolute_value_equality_l4029_402993

theorem absolute_value_equality (x : ℚ) :
  (|x + 3| = |x - 4|) ↔ (x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l4029_402993


namespace NUMINAMATH_CALUDE_alice_favorite_number_l4029_402908

def is_favorite_number (n : ℕ) : Prop :=
  30 ≤ n ∧ n ≤ 70 ∧
  n % 7 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 10 + n % 10) % 4 = 0

theorem alice_favorite_number :
  ∀ n : ℕ, is_favorite_number n ↔ n = 35 := by
  sorry

end NUMINAMATH_CALUDE_alice_favorite_number_l4029_402908


namespace NUMINAMATH_CALUDE_three_double_derivative_l4029_402945

-- Define the derivative operation
noncomputable def derive (f : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the given equation as a property
axiom equation (q : ℝ) : derive (λ x => x) q = 3 * q - 3

-- State the theorem
theorem three_double_derivative : derive (derive (λ x => x)) 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_three_double_derivative_l4029_402945


namespace NUMINAMATH_CALUDE_average_increment_l4029_402972

theorem average_increment (a b c : ℝ) (h : (a + b + c) / 3 = 8) :
  ((a + 1) + (b + 2) + (c + 3)) / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_increment_l4029_402972


namespace NUMINAMATH_CALUDE_incorrect_transformation_l4029_402942

theorem incorrect_transformation (a b c : ℝ) : 
  (a = b) → ¬(∀ c, a / c = b / c) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l4029_402942


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l4029_402946

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem f_derivative_at_one : 
  deriv f 1 = 2 * Real.log 2 - 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l4029_402946


namespace NUMINAMATH_CALUDE_monthly_income_of_P_l4029_402999

/-- Given the average monthly incomes of three people, prove the monthly income of P. -/
theorem monthly_income_of_P (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_of_P_l4029_402999


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_with_conditions_l4029_402911

theorem greatest_three_digit_number_with_conditions : ∃ n : ℕ, 
  (n ≤ 999 ∧ n ≥ 100) ∧ 
  (∃ k : ℕ, n = 7 * k + 2) ∧ 
  (∃ m : ℕ, n = 6 * m + 4) ∧
  (∀ x : ℕ, (x ≤ 999 ∧ x ≥ 100) → 
    (∃ a : ℕ, x = 7 * a + 2) → 
    (∃ b : ℕ, x = 6 * b + 4) → 
    x ≤ n) ∧
  n = 994 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_with_conditions_l4029_402911


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_l4029_402944

structure Line3D where
  -- Placeholder for 3D line properties

structure Plane3D where
  -- Placeholder for 3D plane properties

def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

theorem perpendicular_planes_parallel (m : Line3D) (α β : Plane3D) :
  perpendicular m α → perpendicular m β → parallel α β := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_l4029_402944


namespace NUMINAMATH_CALUDE_exponential_function_characterization_l4029_402950

/-- A function f is exponential if it satisfies f(x+y) = f(x)f(y) for all x and y -/
def IsExponential (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

/-- A function f is monotonically increasing if f(x) ≤ f(y) whenever x ≤ y -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem exponential_function_characterization (f : ℝ → ℝ) 
  (h_exp : IsExponential f) (h_mono : MonoIncreasing f) :
  ∃ a : ℝ, a > 1 ∧ (∀ x, f x = a^x) := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_characterization_l4029_402950


namespace NUMINAMATH_CALUDE_spiders_in_room_l4029_402902

theorem spiders_in_room (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 4 := by
sorry

end NUMINAMATH_CALUDE_spiders_in_room_l4029_402902


namespace NUMINAMATH_CALUDE_square_area_difference_l4029_402903

theorem square_area_difference (small_side large_side : ℝ) 
  (h1 : small_side = 4)
  (h2 : large_side = 9)
  (h3 : small_side < large_side) : 
  large_side^2 - small_side^2 = 65 := by
sorry

end NUMINAMATH_CALUDE_square_area_difference_l4029_402903


namespace NUMINAMATH_CALUDE_mother_ate_five_cookies_l4029_402921

def total_cookies : ℕ := 30
def charlie_cookies : ℕ := 15
def father_cookies : ℕ := 10

def mother_cookies : ℕ := total_cookies - (charlie_cookies + father_cookies)

theorem mother_ate_five_cookies : mother_cookies = 5 := by
  sorry

end NUMINAMATH_CALUDE_mother_ate_five_cookies_l4029_402921


namespace NUMINAMATH_CALUDE_preimage_of_two_one_l4029_402943

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * p.1 + p.2, p.1 - 2 * p.2)

/-- Theorem stating that (1, 0) is the pre-image of (2, 1) under f -/
theorem preimage_of_two_one :
  f (1, 0) = (2, 1) ∧ ∀ p : ℝ × ℝ, f p = (2, 1) → p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_two_one_l4029_402943


namespace NUMINAMATH_CALUDE_distance_AA_l4029_402976

/-- Two unit circles intersecting at X and Y with distance 1 between them -/
structure IntersectingCircles where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  dist_X_Y : dist X Y = 1

/-- Point C on one circle with tangents to the other circle -/
structure TangentPoint (ic : IntersectingCircles) where
  C : ℝ × ℝ
  on_circle : (∃ center, dist center C = 1 ∧ (center = ic.X ∨ center = ic.Y))
  A : ℝ × ℝ  -- Point where tangent CA touches the other circle
  B : ℝ × ℝ  -- Point where tangent CB touches the other circle
  is_tangent_A : ∃ center, dist center A = 1 ∧ center ≠ C
  is_tangent_B : ∃ center, dist center B = 1 ∧ center ≠ C

/-- A' is the point where CB intersects the first circle again -/
def A' (ic : IntersectingCircles) (tp : TangentPoint ic) : ℝ × ℝ :=
  sorry  -- Definition of A' based on the given conditions

/-- The main theorem to prove -/
theorem distance_AA'_is_sqrt3 (ic : IntersectingCircles) (tp : TangentPoint ic) :
  dist tp.A (A' ic tp) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_distance_AA_l4029_402976


namespace NUMINAMATH_CALUDE_sequence_gcd_is_one_l4029_402990

theorem sequence_gcd_is_one (n : ℕ+) : 
  let a : ℕ+ → ℕ := fun k => 100 + 2 * k^2
  Nat.gcd (a n) (a (n + 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_gcd_is_one_l4029_402990


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l4029_402983

def number_of_cages : ℕ := 15
def empty_cages : ℕ := 3
def number_of_chickens : ℕ := 3
def number_of_dogs : ℕ := 3
def number_of_cats : ℕ := 6

def arrangement_count : ℕ := Nat.choose number_of_cages empty_cages * 
                              Nat.factorial 3 * 
                              Nat.factorial number_of_chickens * 
                              Nat.factorial number_of_dogs * 
                              Nat.factorial number_of_cats

theorem animal_arrangement_count : arrangement_count = 70761600 := by
  sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l4029_402983


namespace NUMINAMATH_CALUDE_average_increase_theorem_l4029_402955

/-- Represents a cricket player's batting statistics -/
structure CricketStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (stats : CricketStats) (newInningsRuns : ℕ) : ℚ :=
  (stats.totalRuns + newInningsRuns) / (stats.innings + 1)

theorem average_increase_theorem (initialStats : CricketStats) :
  initialStats.innings = 9 →
  newAverage initialStats 200 = initialStats.average + 8 →
  newAverage initialStats 200 = 128 := by
  sorry


end NUMINAMATH_CALUDE_average_increase_theorem_l4029_402955


namespace NUMINAMATH_CALUDE_AC_length_l4029_402939

/-- A right triangle with a circle passing through its altitude --/
structure RightTriangleWithCircle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- ABC is a right triangle with right angle at A
  right_angle_at_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- AH is perpendicular to BC
  AH_perpendicular_BC : (H.1 - A.1) * (C.1 - B.1) + (H.2 - A.2) * (C.2 - B.2) = 0
  -- A circle passes through A, H, X, and Y
  circle_passes : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2
  -- X is on AB
  X_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  Y_on_AC : ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- Given lengths
  AX_length : ((X.1 - A.1)^2 + (X.2 - A.2)^2)^(1/2 : ℝ) = 5
  AY_length : ((Y.1 - A.1)^2 + (Y.2 - A.2)^2)^(1/2 : ℝ) = 6
  AB_length : ((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2 : ℝ) = 9

/-- The main theorem --/
theorem AC_length (t : RightTriangleWithCircle) : 
  ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)^(1/2 : ℝ) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_AC_length_l4029_402939


namespace NUMINAMATH_CALUDE_hike_consumption_ratio_l4029_402981

/-- Proves the ratio of food to water consumption given hiking conditions --/
theorem hike_consumption_ratio 
  (initial_water : ℝ) 
  (initial_food : ℝ) 
  (initial_gear : ℝ)
  (water_rate : ℝ) 
  (time : ℝ) 
  (final_weight : ℝ) :
  initial_water = 20 →
  initial_food = 10 →
  initial_gear = 20 →
  water_rate = 2 →
  time = 6 →
  final_weight = 34 →
  ∃ (food_rate : ℝ), 
    final_weight = initial_water - water_rate * time + 
                   initial_food - food_rate * time + 
                   initial_gear ∧
    food_rate / water_rate = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hike_consumption_ratio_l4029_402981


namespace NUMINAMATH_CALUDE_pau_total_chicken_l4029_402910

/-- Calculates the total number of chicken pieces Pau eats given the initial orders and a second round of ordering. -/
theorem pau_total_chicken (kobe_order : ℝ) (pau_multiplier : ℝ) (pau_extra : ℝ) (shaq_extra_percent : ℝ) : 
  kobe_order = 5 →
  pau_multiplier = 2 →
  pau_extra = 2.5 →
  shaq_extra_percent = 0.5 →
  2 * (pau_multiplier * kobe_order + pau_extra) = 25 := by
  sorry

end NUMINAMATH_CALUDE_pau_total_chicken_l4029_402910


namespace NUMINAMATH_CALUDE_complement_of_α_l4029_402968

-- Define a custom type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle α
def α : Angle := ⟨25, 39⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let total_minutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨total_minutes / 60, total_minutes % 60⟩

-- Theorem statement
theorem complement_of_α :
  complement α = ⟨64, 21⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_α_l4029_402968


namespace NUMINAMATH_CALUDE_fraction_equivalence_l4029_402938

theorem fraction_equivalence : 
  ∃ (n : ℚ), (3 + n) / (5 + n) = 9 / 11 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l4029_402938


namespace NUMINAMATH_CALUDE_max_xyz_value_l4029_402922

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + 2 * z = (x + z) * (y + z))
  (h2 : x + y + 2 * z = 2) :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  a * b + 2 * c = (a + c) * (b + c) →
  a + b + 2 * c = 2 →
  x * y * z ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_max_xyz_value_l4029_402922


namespace NUMINAMATH_CALUDE_clock_time_after_hours_l4029_402935

theorem clock_time_after_hours (current_time hours_passed : ℕ) : 
  current_time = 2 → 
  hours_passed = 3467 → 
  (current_time + hours_passed) % 12 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_clock_time_after_hours_l4029_402935


namespace NUMINAMATH_CALUDE_difference_of_squares_l4029_402964

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4029_402964
