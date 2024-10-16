import Mathlib

namespace NUMINAMATH_CALUDE_smallest_upper_bound_for_triangle_ratio_l873_87392

/-- Triangle sides -/
structure Triangle :=
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (triangle_ineq_ab : c < a + b)
  (triangle_ineq_bc : a < b + c)
  (triangle_ineq_ca : b < c + a)
  (a_neq_b : a ≠ b)

/-- The smallest upper bound for (a² + b²) / c² in any triangle with unequal sides -/
theorem smallest_upper_bound_for_triangle_ratio :
  ∃ N : ℝ, (∀ t : Triangle, (t.a^2 + t.b^2) / t.c^2 < N) ∧
           (∀ ε > 0, ∃ t : Triangle, N - ε < (t.a^2 + t.b^2) / t.c^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_for_triangle_ratio_l873_87392


namespace NUMINAMATH_CALUDE_inequality_proof_l873_87399

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 1/2) : 
  (1 - a + c) / (Real.sqrt c * (Real.sqrt a + 2 * Real.sqrt b)) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l873_87399


namespace NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l873_87377

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_imply_parallel_lines 
  (a b : Line) (α β : Plane) 
  (ha : subset a α) (hb : subset b α) :
  parallel α β → (parallel α β ∧ parallel α β) := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_imply_parallel_lines_l873_87377


namespace NUMINAMATH_CALUDE_sample_capacity_l873_87380

/-- Given a sample divided into groups, prove that the total sample capacity is 144
    when one group has a frequency of 36 and a frequency rate of 0.25. -/
theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) : 
  frequency = 36 → frequency_rate = 1/4 → n = frequency / frequency_rate → n = 144 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l873_87380


namespace NUMINAMATH_CALUDE_subtracted_value_l873_87305

theorem subtracted_value (x : ℝ) (h1 : (x - 5) / 7 = 7) : 
  ∃ y : ℝ, (x - y) / 10 = 4 ∧ y = 14 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l873_87305


namespace NUMINAMATH_CALUDE_divisors_of_720_l873_87385

theorem divisors_of_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_720_l873_87385


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l873_87354

theorem quadratic_inequality_solution (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (0 < c ∧ c < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l873_87354


namespace NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5_l873_87373

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if four consecutive natural numbers are all prime, false otherwise -/
def fourConsecutivePrimes (a b c d : ℕ) : Prop := 
  isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1

theorem smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5 :
  ∃ (a b c d : ℕ),
    fourConsecutivePrimes a b c d ∧
    a > 10 ∧
    (a + b + c + d) % 5 = 0 ∧
    (a + b + c + d = 60) ∧
    ∀ (w x y z : ℕ),
      fourConsecutivePrimes w x y z →
      w > 10 →
      (w + x + y + z) % 5 = 0 →
      (w + x + y + z) ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_four_consecutive_primes_above_10_divisible_by_5_l873_87373


namespace NUMINAMATH_CALUDE_m_range_l873_87381

theorem m_range (m : ℝ) : 
  (2 * 3 - m > 4) ∧ (2 * 2 - m ≤ 4) → 0 ≤ m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l873_87381


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l873_87370

theorem polynomial_division_remainder :
  ∀ (R : Polynomial ℤ) (Q : Polynomial ℤ),
    (Polynomial.degree R < 2) →
    (x^101 : Polynomial ℤ) = (x^2 - 3*x + 2) * Q + R →
    R = 2^101 * (x - 1) - (x - 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l873_87370


namespace NUMINAMATH_CALUDE_Q_equals_set_l873_87382

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem Q_equals_set : Q = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_Q_equals_set_l873_87382


namespace NUMINAMATH_CALUDE_average_weight_increase_l873_87334

theorem average_weight_increase (initial_count : ℕ) (original_weight replaced_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  original_weight = 101 →
  (original_weight - replaced_weight) / initial_count = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l873_87334


namespace NUMINAMATH_CALUDE_supplementary_angle_measure_l873_87387

-- Define the angle x
def x : ℝ := 10

-- Define the complementary angle
def complementary_angle (x : ℝ) : ℝ := 90 - x

-- Define the supplementary angle
def supplementary_angle (x : ℝ) : ℝ := 180 - x

-- Theorem statement
theorem supplementary_angle_measure :
  (x / complementary_angle x = 1 / 8) →
  supplementary_angle x = 170 := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_measure_l873_87387


namespace NUMINAMATH_CALUDE_cos_symmetry_l873_87303

theorem cos_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.cos (2 * x + π / 3)
  let symmetry_point := π / 3
  f (symmetry_point + x) = f (symmetry_point - x) := by
  sorry

#check cos_symmetry

end NUMINAMATH_CALUDE_cos_symmetry_l873_87303


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l873_87372

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) 
  (h1 : total_packages = 9) 
  (h2 : total_pieces = 135) : 
  total_pieces / total_packages = 15 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l873_87372


namespace NUMINAMATH_CALUDE_rationalize_denominator_l873_87364

theorem rationalize_denominator :
  (2 * Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 5 + Real.sqrt 3) = (3 * Real.sqrt 15 - 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l873_87364


namespace NUMINAMATH_CALUDE_Q_iff_a_in_open_interval_P_or_Q_and_not_P_and_Q_iff_a_in_union_l873_87340

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a + 1) + y^2 / (a - 2) = 1 ∧ (a + 1) * (a - 2) < 0

-- Theorem 1
theorem Q_iff_a_in_open_interval (a : ℝ) : Q a ↔ a ∈ Set.Ioo (-1) 2 := by sorry

-- Theorem 2
theorem P_or_Q_and_not_P_and_Q_iff_a_in_union (a : ℝ) : 
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ∈ Set.Ioo 1 2 ∪ Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_Q_iff_a_in_open_interval_P_or_Q_and_not_P_and_Q_iff_a_in_union_l873_87340


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l873_87310

theorem base_2_representation_of_123 : 
  ∃ (b : List Bool), 
    (b.length = 7) ∧ 
    (b = [true, true, true, true, false, true, true]) ∧
    (Nat.ofDigits 2 (b.map (fun x => if x then 1 else 0)) = 123) := by
  sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l873_87310


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l873_87398

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), (1052 + y) % 23 = 0 → y ≥ x) ∧ 
  (1052 + x) % 23 = 0 := by
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l873_87398


namespace NUMINAMATH_CALUDE_ninth_term_is_18_l873_87375

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  first_term : a 1 = 1 / 2
  condition : a 2 * a 8 = 2 * a 5 + 3

/-- The 9th term of the geometric sequence is 18 -/
theorem ninth_term_is_18 (seq : GeometricSequence) : seq.a 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_18_l873_87375


namespace NUMINAMATH_CALUDE_wire_length_is_250_meters_l873_87379

-- Define the density of copper
def copper_density : Real := 8900

-- Define the volume of wire bought by Chek
def wire_volume : Real := 0.5e-3

-- Define the diagonal of the wire's square cross-section
def wire_diagonal : Real := 2e-3

-- Theorem to prove
theorem wire_length_is_250_meters :
  let cross_section_area := (wire_diagonal ^ 2) / 2
  let wire_length := wire_volume / cross_section_area
  wire_length = 250 := by sorry

end NUMINAMATH_CALUDE_wire_length_is_250_meters_l873_87379


namespace NUMINAMATH_CALUDE_boys_in_class_l873_87308

theorem boys_in_class (total : ℕ) (girls_more : ℕ) (boys : ℕ) 
  (h1 : total = 485) 
  (h2 : girls_more = 69) 
  (h3 : total = boys + (boys + girls_more)) : 
  boys = 208 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l873_87308


namespace NUMINAMATH_CALUDE_AB_value_l873_87393

/- Define the points and circles -/
variable (A O B O' : ℝ)
variable (C C' : Set ℝ)

/- Define the radii of the circles -/
def r : ℝ := 2015
def r' : ℝ := 2016

/- Define the properties of the points and circles -/
axiom aligned : A < O ∧ O < B ∧ B < O'
axiom circle_C : C = {x : ℝ | (x - O)^2 ≤ r^2}
axiom circle_C' : C' = {x : ℝ | (x - O')^2 ≤ r'^2}

/- A and B are intersection points of common tangents -/
axiom tangent_points : ∃ (t₁ t₂ : ℝ → ℝ), 
  (t₁ A ∈ C ∧ t₁ A ∈ C') ∧ (t₂ B ∈ C ∧ t₂ B ∈ C') ∧
  (∀ x, x ≠ A → t₁ x ∉ C ∧ t₁ x ∉ C') ∧
  (∀ x, x ≠ B → t₂ x ∉ C ∧ t₂ x ∉ C')

/- AB, AO, and AO' are integers -/
axiom AB_integer : ∃ n : ℕ, B - A = n
axiom AO_integer : ∃ m : ℕ, O - A = m
axiom AO'_integer : ∃ k : ℕ, O' - A = k

/- AB is less than 10^7 -/
axiom AB_bound : B - A < 10^7

/- The theorem to prove -/
theorem AB_value : B - A = 8124480 :=
sorry

end NUMINAMATH_CALUDE_AB_value_l873_87393


namespace NUMINAMATH_CALUDE_old_lamp_height_is_one_foot_l873_87357

-- Define the height of the new lamp
def new_lamp_height : ℝ := 2.3333333333333335

-- Define the difference in height between the new and old lamps
def height_difference : ℝ := 1.3333333333333333

-- Theorem to prove
theorem old_lamp_height_is_one_foot :
  new_lamp_height - height_difference = 1 := by sorry

end NUMINAMATH_CALUDE_old_lamp_height_is_one_foot_l873_87357


namespace NUMINAMATH_CALUDE_donation_is_45_l873_87311

/-- The total donation to the class funds given the number of stuffed animals and selling prices -/
def total_donation (barbara_stuffed_animals : ℕ) (barbara_price : ℚ) (trish_price : ℚ) : ℚ :=
  let trish_stuffed_animals := 2 * barbara_stuffed_animals
  barbara_stuffed_animals * barbara_price + trish_stuffed_animals * trish_price

/-- Proof that the total donation is $45 given the specific conditions -/
theorem donation_is_45 :
  total_donation 9 2 (3/2) = 45 := by
  sorry

#eval total_donation 9 2 (3/2)

end NUMINAMATH_CALUDE_donation_is_45_l873_87311


namespace NUMINAMATH_CALUDE_die_roll_probability_l873_87335

theorem die_roll_probability : 
  let p_two : ℚ := 1 / 6  -- probability of rolling a 2
  let p_not_two : ℚ := 5 / 6  -- probability of not rolling a 2
  let num_rolls : ℕ := 5  -- number of rolls
  let num_twos : ℕ := 4  -- number of 2s we want
  
  -- probability of rolling exactly four 2s in first four rolls and not a 2 in last roll
  p_two ^ num_twos * p_not_two = 5 / 7776 :=
by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l873_87335


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l873_87350

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  let V (radius : ℝ) := (4 / 3) * Real.pi * radius ^ 3
  V (2 * r) = 8 * V r :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l873_87350


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l873_87366

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2/a + 3/b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l873_87366


namespace NUMINAMATH_CALUDE_origami_distribution_l873_87371

theorem origami_distribution (total_papers : ℝ) (num_cousins : ℝ) (papers_per_cousin : ℝ) : 
  total_papers = 48.0 →
  num_cousins = 6.0 →
  total_papers = num_cousins * papers_per_cousin →
  papers_per_cousin = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_origami_distribution_l873_87371


namespace NUMINAMATH_CALUDE_overtime_switches_l873_87302

/-- Represents a worker assembling light switches -/
structure Worker where
  payment : ℕ
  switches : ℕ

/-- The problem setup -/
def overtime_problem : Prop :=
  ∃ (worker1 worker2 worker3 : Worker),
    let total_payment := 4700
    let total_hours := 5
    let minutes_per_switch_worker2 := 4
    -- Worker 1 conditions
    worker1.payment = 2000 ∧
    -- Worker 2 conditions
    worker2.switches = (total_hours * 60) / minutes_per_switch_worker2 ∧
    -- Worker 3 conditions
    worker3.payment = worker2.payment - 300 ∧
    -- Total payment condition
    worker1.payment + worker2.payment + worker3.payment = total_payment ∧
    -- Proportional payment condition
    worker1.payment * worker2.switches = worker2.payment * worker1.switches ∧
    worker2.payment * worker3.switches = worker3.payment * worker2.switches ∧
    -- Total switches assembled
    worker1.switches + worker2.switches + worker3.switches = 235

/-- The theorem to be proved -/
theorem overtime_switches : overtime_problem := by sorry

end NUMINAMATH_CALUDE_overtime_switches_l873_87302


namespace NUMINAMATH_CALUDE_sin_plus_sin_sqrt2_not_periodic_l873_87390

/-- The function x ↦ sin x + sin (√2 x) is not periodic -/
theorem sin_plus_sin_sqrt2_not_periodic :
  ¬ ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, Real.sin x + Real.sin (Real.sqrt 2 * x) = Real.sin (x + p) + Real.sin (Real.sqrt 2 * (x + p)) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_sin_sqrt2_not_periodic_l873_87390


namespace NUMINAMATH_CALUDE_polynomial_integer_values_l873_87339

/-- A polynomial of degree 3 with real coefficients -/
def Polynomial3 := ℝ → ℝ

/-- Predicate to check if a number is an integer -/
def IsInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- Main theorem: If a polynomial of degree 3 takes integer values at four consecutive integers,
    then it takes integer values at all integers -/
theorem polynomial_integer_values (P : Polynomial3) (i : ℤ) 
  (h1 : IsInteger (P i))
  (h2 : IsInteger (P (i + 1)))
  (h3 : IsInteger (P (i + 2)))
  (h4 : IsInteger (P (i + 3))) :
  ∀ n : ℤ, IsInteger (P n) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_values_l873_87339


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l873_87397

/-- A convex polygon with the sum of all angles except two equal to 3240° has 22 sides. -/
theorem convex_polygon_sides (n : ℕ) (sum_except_two : ℝ) : 
  sum_except_two = 3240 → (∃ (a b : ℝ), 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 
    180 * (n - 2) = sum_except_two + a + b) → n = 22 :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l873_87397


namespace NUMINAMATH_CALUDE_find_multiplier_l873_87345

theorem find_multiplier (N S D : ℕ) (h1 : N = 220040) (h2 : S = 555 + 445) (h3 : D = 555 - 445) :
  ∃ x : ℕ, N = S * x * D + 40 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_multiplier_l873_87345


namespace NUMINAMATH_CALUDE_jeremy_gives_two_watermelons_l873_87395

/-- The number of watermelons Jeremy gives to his dad each week. -/
def watermelons_given_to_dad (total_watermelons : ℕ) (weeks_lasted : ℕ) (eaten_per_week : ℕ) : ℕ :=
  (total_watermelons / weeks_lasted) - eaten_per_week

/-- Theorem stating that Jeremy gives 2 watermelons to his dad each week. -/
theorem jeremy_gives_two_watermelons :
  watermelons_given_to_dad 30 6 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_gives_two_watermelons_l873_87395


namespace NUMINAMATH_CALUDE_f_neg_one_equals_five_l873_87333

-- Define the function f
def f (x : ℝ) : ℝ := (1 - x)^2 + 1

-- State the theorem
theorem f_neg_one_equals_five : f (-1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_equals_five_l873_87333


namespace NUMINAMATH_CALUDE_prob_sum_equals_seven_ninths_l873_87342

def total_balls : ℕ := 9
def black_balls : ℕ := 5
def white_balls : ℕ := 4

def P_A : ℚ := black_balls / total_balls
def P_B_given_A : ℚ := white_balls / (total_balls - 1)

theorem prob_sum_equals_seven_ninths :
  (P_A * P_B_given_A) + P_B_given_A = 7 / 9 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_equals_seven_ninths_l873_87342


namespace NUMINAMATH_CALUDE_range_of_f_l873_87362

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f : {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l873_87362


namespace NUMINAMATH_CALUDE_complement_probability_l873_87306

theorem complement_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_complement_probability_l873_87306


namespace NUMINAMATH_CALUDE_katie_has_more_games_l873_87356

/-- The number of games Katie has -/
def katie_games : ℕ := 81

/-- The number of games Katie's friends have -/
def friends_games : ℕ := 59

/-- The difference in games between Katie and her friends -/
def game_difference : ℕ := katie_games - friends_games

theorem katie_has_more_games : game_difference = 22 := by
  sorry

end NUMINAMATH_CALUDE_katie_has_more_games_l873_87356


namespace NUMINAMATH_CALUDE_smallest_n_correct_l873_87394

/-- The smallest positive integer n such that n^2 + 4 has at least four distinct prime factors -/
def smallest_n : ℕ := 179

/-- A function that returns the number of distinct prime factors of a natural number -/
def num_distinct_prime_factors (m : ℕ) : ℕ := sorry

theorem smallest_n_correct :
  (∀ k : ℕ, k < smallest_n → num_distinct_prime_factors (k^2 + 4) < 4) ∧
  num_distinct_prime_factors (smallest_n^2 + 4) ≥ 4 := by sorry

end NUMINAMATH_CALUDE_smallest_n_correct_l873_87394


namespace NUMINAMATH_CALUDE_city_tax_solution_l873_87344

/-- Represents the number of residents and tax paid in a city. -/
structure CityTax where
  residents : ℕ
  taxPerResident : ℕ

/-- Calculates the total tax paid by the city. -/
def totalTax (c : CityTax) : ℕ := c.residents * c.taxPerResident

/-- Checks if the given CityTax satisfies the conditions of the problem. -/
def satisfiesConditions (c : CityTax) : Prop :=
  (c.residents + 3000) * (c.taxPerResident - 10) = c.residents * c.taxPerResident ∧
  (c.residents - 1000) * (c.taxPerResident + 10) = c.residents * c.taxPerResident

/-- Theorem stating the solution to the problem. -/
theorem city_tax_solution :
  ∃ (c : CityTax), satisfiesConditions c ∧ 
    c.residents = 3000 ∧ 
    c.taxPerResident = 20 ∧
    totalTax c = 60000 := by
  sorry


end NUMINAMATH_CALUDE_city_tax_solution_l873_87344


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_format_l873_87367

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits in a Time24 -/
def sumOfDigitsTime24 (t : Time24) : ℕ :=
  sumOfDigits t.hours.val + sumOfDigits t.minutes.val

/-- The theorem stating the maximum sum of digits in a 24-hour format display -/
theorem max_sum_of_digits_24hour_format :
  ∃ (max : ℕ), ∀ (t : Time24), sumOfDigitsTime24 t ≤ max ∧
  ∃ (t' : Time24), sumOfDigitsTime24 t' = max ∧ max = 24 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_format_l873_87367


namespace NUMINAMATH_CALUDE_painted_rectangle_ratio_l873_87338

/-- Given a rectangle with length 2s and width s, and a paint brush of width w,
    if half the area of the rectangle is painted when the brush is swept along both diagonals,
    then the ratio of the length of the rectangle to the brush width is 6. -/
theorem painted_rectangle_ratio (s w : ℝ) (h_pos_s : 0 < s) (h_pos_w : 0 < w) :
  w^2 + 2*(s-w)^2 = s^2 → (2*s) / w = 6 := by sorry

end NUMINAMATH_CALUDE_painted_rectangle_ratio_l873_87338


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l873_87301

/-- Proves the distance covered by a boat traveling downstream -/
theorem boat_downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (time : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 5)
  (h3 : time = 4) :
  boat_speed * time + stream_speed * time = 84 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l873_87301


namespace NUMINAMATH_CALUDE_unique_number_exists_l873_87347

theorem unique_number_exists : ∃! n : ℕ, 
  let sum := 2615 + 3895
  let diff := 3895 - 2615
  n / sum = 3 * diff ∧ n % sum = 65 := by
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l873_87347


namespace NUMINAMATH_CALUDE_unique_special_function_l873_87378

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f ((x + 2) + 2) = f (x + 2)) ∧  -- g(x) = f(x+2) is even
  (∀ x, x ∈ Set.Icc 0 2 → f x = x)  -- f(x) = x for x ∈ [0, 2]

/-- There exists a unique function satisfying the special_function conditions -/
theorem unique_special_function : ∃! f : ℝ → ℝ, special_function f :=
sorry

end NUMINAMATH_CALUDE_unique_special_function_l873_87378


namespace NUMINAMATH_CALUDE_fraction_irreducible_l873_87391

theorem fraction_irreducible (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l873_87391


namespace NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l873_87322

theorem integral_exp_plus_2x_equals_e :
  ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_integral_exp_plus_2x_equals_e_l873_87322


namespace NUMINAMATH_CALUDE_favorite_movies_total_duration_l873_87355

/-- Given the durations of four people's favorite movies with specific relationships,
    prove that the total duration of all movies is 76 hours. -/
theorem favorite_movies_total_duration
  (michael_duration : ℝ)
  (joyce_duration : ℝ)
  (nikki_duration : ℝ)
  (ryn_duration : ℝ)
  (h1 : joyce_duration = michael_duration + 2)
  (h2 : nikki_duration = 3 * michael_duration)
  (h3 : ryn_duration = 4/5 * nikki_duration)
  (h4 : nikki_duration = 30) :
  joyce_duration + michael_duration + nikki_duration + ryn_duration = 76 := by
sorry


end NUMINAMATH_CALUDE_favorite_movies_total_duration_l873_87355


namespace NUMINAMATH_CALUDE_production_cost_decrease_rate_l873_87358

theorem production_cost_decrease_rate : ∃ x : ℝ, 
  (400 * (1 - x)^2 = 361) ∧ (x = 0.05) := by sorry

end NUMINAMATH_CALUDE_production_cost_decrease_rate_l873_87358


namespace NUMINAMATH_CALUDE_f_sum_positive_l873_87384

/-- The function f(x) = x + x³ -/
def f (x : ℝ) : ℝ := x + x^3

/-- Theorem: For x₁, x₂ ∈ ℝ with x₁ + x₂ > 0, f(x₁) + f(x₂) > 0 -/
theorem f_sum_positive (x₁ x₂ : ℝ) (h : x₁ + x₂ > 0) : f x₁ + f x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l873_87384


namespace NUMINAMATH_CALUDE_mangoes_harvested_l873_87389

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) 
  (h1 : neighbors = 8)
  (h2 : mangoes_per_neighbor = 35)
  (h3 : ∃ (total : ℕ), total / 2 = neighbors * mangoes_per_neighbor) :
  ∃ (total : ℕ), total = 560 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_harvested_l873_87389


namespace NUMINAMATH_CALUDE_rational_numbers_include_zero_not_only_positive_and_negative_rationals_l873_87312

theorem rational_numbers_include_zero : ∃ (x : ℚ), x ≠ 0 ∧ x ≥ 0 ∧ x ≤ 0 := by
  sorry

theorem not_only_positive_and_negative_rationals : 
  ¬(∀ (x : ℚ), x > 0 ∨ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_include_zero_not_only_positive_and_negative_rationals_l873_87312


namespace NUMINAMATH_CALUDE_special_right_triangle_pair_theorem_l873_87341

/-- Two right triangles with specific properties -/
structure SpecialRightTrianglePair where
  /-- The length of the common leg -/
  x : ℝ
  /-- The length of the other leg of T₁ -/
  y : ℝ
  /-- The length of the hypotenuse of T₁ -/
  w : ℝ
  /-- The length of the other leg of T₂ -/
  v : ℝ
  /-- The length of the hypotenuse of T₂ -/
  z : ℝ
  /-- Area of T₁ is 3 -/
  area_t1 : x * y / 2 = 3
  /-- Area of T₂ is 4 -/
  area_t2 : x * v / 2 = 4
  /-- Hypotenuse of T₁ is twice the length of the hypotenuse of T₂ -/
  hypotenuse_relation : w = 2 * z
  /-- Pythagorean theorem for T₁ -/
  pythagorean_t1 : x^2 + y^2 = w^2
  /-- Pythagorean theorem for T₂ -/
  pythagorean_t2 : x^2 + v^2 = z^2

/-- The square of the product of the third sides is 2304/25 -/
theorem special_right_triangle_pair_theorem (t : SpecialRightTrianglePair) :
  (t.y * t.v)^2 = 2304/25 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_pair_theorem_l873_87341


namespace NUMINAMATH_CALUDE_smallest_d_for_divisibility_l873_87313

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (d : ℕ) : ℕ := 5 + 4 + 7 + d + 0 + 6

def number (d : ℕ) : ℕ := 547000 + d * 1000 + 6

theorem smallest_d_for_divisibility :
  ∃ (d : ℕ), d = 2 ∧ 
  is_divisible_by_3 (number d) ∧ 
  ∀ (k : ℕ), k < d → ¬is_divisible_by_3 (number k) := by
  sorry

#check smallest_d_for_divisibility

end NUMINAMATH_CALUDE_smallest_d_for_divisibility_l873_87313


namespace NUMINAMATH_CALUDE_cylinder_in_sphere_surface_area_difference_l873_87329

/-- Given a sphere of radius R and an inscribed cylinder with maximum lateral surface area,
    the difference between the sphere's surface area and the cylinder's lateral surface area is 2πR². -/
theorem cylinder_in_sphere_surface_area_difference (R : ℝ) (h : R > 0) :
  ∃ (cylinder_lsa : ℝ),
    (∀ (other_cylinder_lsa : ℝ), cylinder_lsa ≥ other_cylinder_lsa) →
    4 * π * R^2 - cylinder_lsa = 2 * π * R^2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_in_sphere_surface_area_difference_l873_87329


namespace NUMINAMATH_CALUDE_infinitely_many_composite_f_increasing_l873_87317

/-- The number of positive divisors of a natural number -/
def tau (a : ℕ) : ℕ := (Nat.divisors a).card

/-- The function f(n) as defined in the problem -/
def f (n : ℕ) : ℕ := tau (Nat.factorial n) - tau (Nat.factorial (n - 1))

/-- A composite number -/
def Composite (n : ℕ) : Prop := ¬Nat.Prime n ∧ n > 1

/-- The main theorem to be proved -/
theorem infinitely_many_composite_f_increasing :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
  (∀ n ∈ S, Composite n ∧ 
    (∀ m : ℕ, m < n → f m < f n)) := by sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_f_increasing_l873_87317


namespace NUMINAMATH_CALUDE_expression_equality_l873_87353

theorem expression_equality : 
  Real.sqrt 3 * Real.tan (30 * π / 180) - (1 / 2)⁻¹ + Real.sqrt 8 - |1 - Real.sqrt 2| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l873_87353


namespace NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_100_l873_87388

/-- A function that checks if all digits of a natural number are unique -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 9 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_nine_with_unique_digits_mod_100 :
  M % 9 = 0 ∧ has_unique_digits M ∧ (∀ k : ℕ, k % 9 = 0 → has_unique_digits k → k ≤ M) →
  M % 100 = 81 := by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_100_l873_87388


namespace NUMINAMATH_CALUDE_x_value_in_equation_l873_87300

theorem x_value_in_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + 6 * x * y = x^3 + x * y^2) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_equation_l873_87300


namespace NUMINAMATH_CALUDE_tribe_leadership_combinations_l873_87349

def tribe_size : ℕ := 12
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem tribe_leadership_combinations :
  (tribe_size) *
  (tribe_size - num_chiefs) *
  (tribe_size - num_chiefs - 1) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs) num_inferior_officers_per_chief) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs - num_inferior_officers_per_chief) num_inferior_officers_per_chief) = 221760 :=
by sorry

end NUMINAMATH_CALUDE_tribe_leadership_combinations_l873_87349


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_l873_87307

/-- Given points A, B, C, D, E, and F in a coordinate plane, where:
    A is at (0,8), B at (0,0), C at (10,0)
    D is the midpoint of AB
    E is the midpoint of BC
    F is the intersection of lines AE and CD
    Prove that the sum of F's coordinates is 6 -/
theorem intersection_coordinate_sum :
  let A : ℝ × ℝ := (0, 8)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (10, 0)
  let D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let m_AE : ℝ := (E.2 - A.2) / (E.1 - A.1)
  let b_AE : ℝ := A.2 - m_AE * A.1
  let m_CD : ℝ := (D.2 - C.2) / (D.1 - C.1)
  let b_CD : ℝ := C.2 - m_CD * C.1
  let F : ℝ × ℝ := ((b_CD - b_AE) / (m_AE - m_CD), m_AE * ((b_CD - b_AE) / (m_AE - m_CD)) + b_AE)
  F.1 + F.2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_intersection_coordinate_sum_l873_87307


namespace NUMINAMATH_CALUDE_x_zero_sufficient_not_necessary_l873_87337

theorem x_zero_sufficient_not_necessary :
  (∃ x : ℝ, x = 0 → x^2 - 2*x = 0) ∧
  (∃ x : ℝ, x^2 - 2*x = 0 ∧ x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_zero_sufficient_not_necessary_l873_87337


namespace NUMINAMATH_CALUDE_train_distance_theorem_l873_87328

/-- Represents the distance traveled by the second train -/
def x : ℝ := 400

/-- The speed of the first train in km/hr -/
def speed1 : ℝ := 50

/-- The speed of the second train in km/hr -/
def speed2 : ℝ := 40

/-- The additional distance traveled by the first train compared to the second train -/
def additional_distance : ℝ := 100

/-- The total distance between the starting points of the two trains -/
def total_distance : ℝ := x + (x + additional_distance)

theorem train_distance_theorem :
  speed1 > 0 ∧ speed2 > 0 ∧ 
  x / speed2 = (x + additional_distance) / speed1 →
  total_distance = 900 :=
sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l873_87328


namespace NUMINAMATH_CALUDE_plan1_greater_loss_l873_87319

/-- Probability of minor flooding -/
def p_minor : ℝ := 0.2

/-- Probability of major flooding -/
def p_major : ℝ := 0.05

/-- Cost of building a protective wall -/
def wall_cost : ℝ := 4000

/-- Loss due to major flooding -/
def major_flood_loss : ℝ := 30000

/-- Loss due to minor flooding in Plan 2 -/
def minor_flood_loss : ℝ := 15000

/-- Expected loss for Plan 1 -/
def expected_loss_plan1 : ℝ := major_flood_loss * p_major + wall_cost * p_minor + wall_cost

/-- Expected loss for Plan 2 -/
def expected_loss_plan2 : ℝ := major_flood_loss * p_major + minor_flood_loss * p_minor

/-- Theorem stating that the expected loss of Plan 1 is greater than the expected loss of Plan 2 -/
theorem plan1_greater_loss : expected_loss_plan1 > expected_loss_plan2 :=
  sorry

end NUMINAMATH_CALUDE_plan1_greater_loss_l873_87319


namespace NUMINAMATH_CALUDE_tan_a_values_l873_87352

theorem tan_a_values (a : Real) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_values_l873_87352


namespace NUMINAMATH_CALUDE_segment_division_l873_87309

/-- Given a segment of length a, prove that dividing it into n equal parts
    results in each part having a length of a/(n+1). -/
theorem segment_division (a : ℝ) (n : ℕ) (h : 0 < n) :
  ∃ (x : ℝ), x = a / (n + 1) ∧ n * x = a :=
by sorry

end NUMINAMATH_CALUDE_segment_division_l873_87309


namespace NUMINAMATH_CALUDE_tangent_slope_at_4_l873_87316

def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 1

theorem tangent_slope_at_4 : 
  (deriv f) 4 = -8 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_4_l873_87316


namespace NUMINAMATH_CALUDE_first_four_seeds_l873_87361

/-- Represents a row in the random number table -/
def RandomTableRow := List Nat

/-- The random number table -/
def randomTable : List RandomTableRow := [
  [78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
  [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
  [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
  [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
  [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]
]

/-- The starting position in the random number table -/
def startPosition : Nat × Nat := (2, 5)

/-- The total number of seeds -/
def totalSeeds : Nat := 850

/-- Function to get the next valid seed number -/
def getNextValidSeed (table : List RandomTableRow) (pos : Nat × Nat) (maxSeed : Nat) : Option (Nat × (Nat × Nat)) :=
  sorry

/-- Theorem stating that the first 4 valid seed numbers are 390, 737, 220, and 372 -/
theorem first_four_seeds :
  let seedNumbers := [390, 737, 220, 372]
  ∃ (pos1 pos2 pos3 pos4 : Nat × Nat),
    getNextValidSeed randomTable startPosition totalSeeds = some (seedNumbers[0], pos1) ∧
    getNextValidSeed randomTable pos1 totalSeeds = some (seedNumbers[1], pos2) ∧
    getNextValidSeed randomTable pos2 totalSeeds = some (seedNumbers[2], pos3) ∧
    getNextValidSeed randomTable pos3 totalSeeds = some (seedNumbers[3], pos4) :=
  sorry

end NUMINAMATH_CALUDE_first_four_seeds_l873_87361


namespace NUMINAMATH_CALUDE_perfect_square_condition_l873_87360

theorem perfect_square_condition (x y : ℕ) :
  (∃ (n : ℕ), (x + y)^2 + 3*x + y + 1 = n^2) ↔ x = y :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l873_87360


namespace NUMINAMATH_CALUDE_max_bent_strips_achievable_14_bent_strips_max_bent_strips_is_14_l873_87376

/-- Represents a 3x3x3 cube --/
structure Cube :=
  (side_length : ℕ := 3)

/-- Represents a 3x1 strip --/
structure Strip :=
  (length : ℕ := 3)
  (width : ℕ := 1)

/-- Represents the configuration of strips on the cube --/
structure CubeConfiguration :=
  (cube : Cube)
  (total_strips : ℕ := 18)
  (bent_strips : ℕ)
  (flat_strips : ℕ)

/-- The theorem stating the maximum number of bent strips --/
theorem max_bent_strips (config : CubeConfiguration) : config.bent_strips ≤ 14 :=
by sorry

/-- The theorem stating that 14 bent strips is achievable --/
theorem achievable_14_bent_strips : ∃ (config : CubeConfiguration), config.bent_strips = 14 ∧ config.flat_strips = 4 :=
by sorry

/-- The main theorem combining the above results --/
theorem max_bent_strips_is_14 : 
  (∀ (config : CubeConfiguration), config.bent_strips ≤ 14) ∧
  (∃ (config : CubeConfiguration), config.bent_strips = 14) :=
by sorry

end NUMINAMATH_CALUDE_max_bent_strips_achievable_14_bent_strips_max_bent_strips_is_14_l873_87376


namespace NUMINAMATH_CALUDE_gilbert_cricket_ratio_l873_87369

/-- The number of crickets Gilbert eats per week at 90°F -/
def crickets_90 : ℕ := 4

/-- The total number of weeks -/
def total_weeks : ℕ := 15

/-- The fraction of time the temperature is 90°F -/
def temp_90_fraction : ℚ := 4/5

/-- The total number of crickets eaten over the entire period -/
def total_crickets : ℕ := 72

/-- The number of crickets Gilbert eats per week at 100°F -/
def crickets_100 : ℕ := 8

theorem gilbert_cricket_ratio :
  crickets_100 / crickets_90 = 2 :=
sorry

end NUMINAMATH_CALUDE_gilbert_cricket_ratio_l873_87369


namespace NUMINAMATH_CALUDE_tangent_line_sum_l873_87326

open Real

/-- Given a function f: ℝ → ℝ with a tangent line at x = 2 described by the equation 2x - y - 3 = 0,
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f 2 → 2 * x - y - 3 = 0 ↔ y = 2 * x - 3) :
  f 2 + deriv f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l873_87326


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l873_87336

/-- Represents the number of ways to arrange plants under lamps -/
def plant_arrangements : ℕ := 49

/-- The number of basil plants -/
def num_basil : ℕ := 3

/-- The number of aloe plants -/
def num_aloe : ℕ := 1

/-- The number of lamp colors -/
def num_lamp_colors : ℕ := 3

/-- The number of lamps per color -/
def lamps_per_color : ℕ := 2

/-- The total number of lamps -/
def total_lamps : ℕ := num_lamp_colors * lamps_per_color

/-- The total number of plants -/
def total_plants : ℕ := num_basil + num_aloe

theorem plant_arrangement_count :
  (num_basil = 3) →
  (num_aloe = 1) →
  (num_lamp_colors = 3) →
  (lamps_per_color = 2) →
  (total_plants ≤ total_lamps) →
  (plant_arrangements = 49) := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l873_87336


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l873_87365

theorem set_inclusion_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {1, 2, a}
  let B : Set ℝ := {1, a^2 - a}
  A ⊇ B → a = -1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l873_87365


namespace NUMINAMATH_CALUDE_shorties_eating_today_l873_87374

/-- Represents the number of shorties who eat donuts every day -/
def daily_eaters : ℕ := 6

/-- Represents the number of shorties who eat donuts every other day -/
def bi_daily_eaters : ℕ := 8

/-- Represents the number of shorties who ate donuts yesterday -/
def yesterday_eaters : ℕ := 11

/-- Theorem stating that the number of shorties who will eat donuts today is 9 -/
theorem shorties_eating_today : 
  ∃ (today_eaters : ℕ), today_eaters = 9 ∧
  today_eaters = daily_eaters + (bi_daily_eaters - (yesterday_eaters - daily_eaters)) :=
by
  sorry


end NUMINAMATH_CALUDE_shorties_eating_today_l873_87374


namespace NUMINAMATH_CALUDE_union_complement_equals_B_l873_87314

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equals_B : B ∪ (U \ A) = B := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_B_l873_87314


namespace NUMINAMATH_CALUDE_donna_has_40_bananas_l873_87330

/-- The number of bananas Donna has -/
def donnas_bananas (total : ℕ) (dawn_lydia_diff : ℕ) (lydias_bananas : ℕ) : ℕ :=
  total - (lydias_bananas + dawn_lydia_diff + lydias_bananas)

/-- Theorem: Donna has 40 bananas given the specified conditions -/
theorem donna_has_40_bananas :
  donnas_bananas 200 40 60 = 40 := by sorry

end NUMINAMATH_CALUDE_donna_has_40_bananas_l873_87330


namespace NUMINAMATH_CALUDE_even_function_implies_f_2_eq_3_l873_87331

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_f_2_eq_3 :
  (∀ x : ℝ, f a x = f a (-x)) → f a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_f_2_eq_3_l873_87331


namespace NUMINAMATH_CALUDE_consumption_decrease_l873_87359

theorem consumption_decrease (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  let new_price := 1.4 * original_price
  let new_budget := 1.12 * (original_price * original_quantity)
  let new_quantity := new_budget / new_price
  new_quantity / original_quantity = 0.8 := by sorry

end NUMINAMATH_CALUDE_consumption_decrease_l873_87359


namespace NUMINAMATH_CALUDE_parallel_tangents_and_function_inequality_l873_87315

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2 * Real.log x

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*a*x - (2*a + 1) + 2/x

theorem parallel_tangents_and_function_inequality (a : ℝ) (h_a : a > 0) :
  (f_deriv a 1 = f_deriv a 3 → a = 1/12) ∧
  (∀ x₁ ∈ Set.Ioc 0 2, ∃ x₂ ∈ Set.Ioc 0 2, f a x₁ < g x₂) ↔ 
  (0 < a ∧ a ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_parallel_tangents_and_function_inequality_l873_87315


namespace NUMINAMATH_CALUDE_cubic_root_sum_l873_87320

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  (a*b)/c + (b*c)/a + (c*a)/b = 49/6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l873_87320


namespace NUMINAMATH_CALUDE_scientific_notation_of_2410000_l873_87368

theorem scientific_notation_of_2410000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2410000 = a * (10 : ℝ) ^ n ∧ a = 2.41 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2410000_l873_87368


namespace NUMINAMATH_CALUDE_both_balls_prob_at_least_one_ball_prob_l873_87304

/-- The probability space for the ball experiment -/
structure BallProbSpace where
  /-- The probability of ball A falling into the box -/
  prob_A : ℝ
  /-- The probability of ball B falling into the box -/
  prob_B : ℝ
  /-- The probability of ball A falling into the box is 1/2 -/
  hA : prob_A = 1/2
  /-- The probability of ball B falling into the box is 1/3 -/
  hB : prob_B = 1/3
  /-- The events A and B are independent -/
  indep : ∀ {p : ℝ} {q : ℝ}, p = prob_A → q = prob_B → p * q = prob_A * prob_B

/-- The probability that both ball A and ball B fall into the box is 1/6 -/
theorem both_balls_prob (space : BallProbSpace) : space.prob_A * space.prob_B = 1/6 := by
  sorry

/-- The probability that at least one of ball A and ball B falls into the box is 2/3 -/
theorem at_least_one_ball_prob (space : BallProbSpace) :
  1 - (1 - space.prob_A) * (1 - space.prob_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_both_balls_prob_at_least_one_ball_prob_l873_87304


namespace NUMINAMATH_CALUDE_xyz_inequality_l873_87327

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^3 + y^3 + z^3 ≥ x*y + y*z + x*z := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_l873_87327


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l873_87332

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (x + 1/x) ≤ Real.sqrt 15 ∧ ∃ y : ℝ, y + 1/y = Real.sqrt 15 ∧ 13 = y^2 + 1/y^2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l873_87332


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l873_87325

-- Define the initial solution parameters
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.35
def target_concentration : ℝ := 0.50

-- Define the amount of pure alcohol to be added
def added_alcohol : ℝ := 1.8

-- Theorem statement
theorem alcohol_concentration_proof :
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := initial_alcohol + added_alcohol
  (final_alcohol / final_volume) = target_concentration := by
  sorry


end NUMINAMATH_CALUDE_alcohol_concentration_proof_l873_87325


namespace NUMINAMATH_CALUDE_andrea_jim_age_sum_l873_87323

theorem andrea_jim_age_sum : 
  ∀ (A J x y : ℕ),
  A = J + 29 →                   -- Andrea is 29 years older than Jim
  A - x + J - x = 47 →           -- Sum of their ages x years ago was 47
  J - y = 2 * (J - x) →          -- Jim's age y years ago was twice his age x years ago
  A = 3 * (J - y) →              -- Andrea's current age is three times Jim's age y years ago
  A + J = 79 :=                  -- The sum of their current ages is 79
by
  sorry

end NUMINAMATH_CALUDE_andrea_jim_age_sum_l873_87323


namespace NUMINAMATH_CALUDE_negative_three_times_inequality_l873_87324

theorem negative_three_times_inequality (m n : ℝ) (h : m > n) : -3*m < -3*n := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_inequality_l873_87324


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_on_interval_f_max_on_interval_l873_87383

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for monotonically decreasing intervals
theorem f_monotone_decreasing :
  (∀ x < -1, (f' x) < 0) ∧ (∀ x > 3, (f' x) < 0) :=
sorry

-- Theorem for minimum value on [-2, 2]
theorem f_min_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y ∧ f x = -7 :=
sorry

-- Theorem for maximum value on [-2, 2]
theorem f_max_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≤ f x ∧ f x = 20 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_f_min_on_interval_f_max_on_interval_l873_87383


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l873_87363

theorem opposite_of_negative_three : 
  ∃ x : ℤ, x + (-3) = 0 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l873_87363


namespace NUMINAMATH_CALUDE_square_difference_equality_l873_87386

theorem square_difference_equality : 1013^2 - 991^2 - 1007^2 + 997^2 = 24048 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l873_87386


namespace NUMINAMATH_CALUDE_triangle_area_is_eight_l873_87348

-- Define the slopes and intersection point
def slope1 : ℝ := -1
def slope2 : ℝ := 3
def intersection : ℝ × ℝ := (1, 3)

-- Define the lines
def line1 (x : ℝ) : ℝ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℝ) : ℝ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℝ) : Prop := x - y = 2

-- Define the points of the triangle
def pointA : ℝ × ℝ := intersection
def pointB : ℝ × ℝ := (-1, -3)  -- Intersection of line2 and line3
def pointC : ℝ × ℝ := (3, 1)    -- Intersection of line1 and line3

-- Theorem statement
theorem triangle_area_is_eight :
  let area := (1/2) * abs (
    pointA.1 * (pointB.2 - pointC.2) +
    pointB.1 * (pointC.2 - pointA.2) +
    pointC.1 * (pointA.2 - pointB.2)
  )
  area = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_eight_l873_87348


namespace NUMINAMATH_CALUDE_biased_coin_probability_l873_87396

theorem biased_coin_probability (p : ℝ) (n : ℕ) (h_p : p = 3/4) (h_n : n = 4) :
  1 - p^n = 175/256 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l873_87396


namespace NUMINAMATH_CALUDE_cos_11pi_3_plus_tan_neg_3pi_4_l873_87346

theorem cos_11pi_3_plus_tan_neg_3pi_4 :
  Real.cos (11 * π / 3) + Real.tan (-3 * π / 4) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_11pi_3_plus_tan_neg_3pi_4_l873_87346


namespace NUMINAMATH_CALUDE_kaleb_second_half_score_l873_87351

/-- 
Given that Kaleb scored 43 points in the first half of a trivia game and 66 points in total,
this theorem proves that he scored 23 points in the second half.
-/
theorem kaleb_second_half_score 
  (first_half_score : ℕ) 
  (total_score : ℕ) 
  (h1 : first_half_score = 43)
  (h2 : total_score = 66) :
  total_score - first_half_score = 23 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_second_half_score_l873_87351


namespace NUMINAMATH_CALUDE_binomial_product_l873_87343

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 7) = 4 * x^2 - 25 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l873_87343


namespace NUMINAMATH_CALUDE_dereks_age_l873_87318

/-- Given that Charlie's age is four times Derek's age, Emily is five years older than Derek,
    and Charlie and Emily are twins, prove that Derek is 5/3 years old. -/
theorem dereks_age (charlie emily derek : ℝ)
    (h1 : charlie = 4 * derek)
    (h2 : emily = derek + 5)
    (h3 : charlie = emily) :
    derek = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dereks_age_l873_87318


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l873_87321

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + 3 - k

-- Define the condition for two distinct real roots
def has_two_distinct_roots (k : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ quadratic α k = 0 ∧ quadratic β k = 0

-- Define the relationship between k and the roots
def root_relationship (k α β : ℝ) : Prop :=
  k^2 = α * β + 3*k

-- Theorem statement
theorem quadratic_root_theorem (k : ℝ) :
  has_two_distinct_roots k → (∃ α β : ℝ, root_relationship k α β) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l873_87321
