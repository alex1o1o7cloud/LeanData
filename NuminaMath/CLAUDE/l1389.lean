import Mathlib

namespace NUMINAMATH_CALUDE_four_digit_number_with_zero_removal_l1389_138903

/-- Represents a four-digit number with one digit being zero -/
structure FourDigitNumberWithZero where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_less_than_10 : a < 10
  b_less_than_10 : b < 10
  c_less_than_10 : c < 10
  d_less_than_10 : d < 10
  has_one_zero : (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b = 0 ∧ c ≠ 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b ≠ 0 ∧ c = 0 ∧ d ≠ 0) ∨
                 (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d = 0)

/-- The value of the four-digit number -/
def value (n : FourDigitNumberWithZero) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The value of the number after removing the zero -/
def valueWithoutZero (n : FourDigitNumberWithZero) : Nat :=
  if n.a = 0 then 100 * n.b + 10 * n.c + n.d
  else if n.b = 0 then 100 * n.a + 10 * n.c + n.d
  else if n.c = 0 then 100 * n.a + 10 * n.b + n.d
  else 100 * n.a + 10 * n.b + n.c

theorem four_digit_number_with_zero_removal (n : FourDigitNumberWithZero) :
  (value n = 9 * valueWithoutZero n) → (value n = 2025 ∨ value n = 6075) := by
  sorry


end NUMINAMATH_CALUDE_four_digit_number_with_zero_removal_l1389_138903


namespace NUMINAMATH_CALUDE_carla_counting_theorem_l1389_138991

theorem carla_counting_theorem (ceiling_tiles : ℕ) (books : ℕ) 
  (h1 : ceiling_tiles = 38) 
  (h2 : books = 75) : 
  ceiling_tiles * 2 + books * 3 = 301 := by
  sorry

end NUMINAMATH_CALUDE_carla_counting_theorem_l1389_138991


namespace NUMINAMATH_CALUDE_perimeter_is_120_inches_l1389_138947

/-- The perimeter of a figure formed by cutting an equilateral triangle from a square and rotating it -/
def rotated_triangle_perimeter (square_side : ℝ) (triangle_side : ℝ) : ℝ :=
  3 * square_side + 3 * triangle_side

/-- Theorem: The perimeter of the new figure is 120 inches -/
theorem perimeter_is_120_inches :
  let square_side := (20 : ℝ)
  let triangle_side := (20 : ℝ)
  rotated_triangle_perimeter square_side triangle_side = 120 := by
  sorry

#eval rotated_triangle_perimeter 20 20

end NUMINAMATH_CALUDE_perimeter_is_120_inches_l1389_138947


namespace NUMINAMATH_CALUDE_complex_number_relation_l1389_138997

theorem complex_number_relation (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 8)
  (eq5 : s + t + u = 4) :
  s * t * u = 10 := by
sorry


end NUMINAMATH_CALUDE_complex_number_relation_l1389_138997


namespace NUMINAMATH_CALUDE_josh_pencils_calculation_l1389_138972

/-- The number of pencils Josh had initially -/
def initial_pencils : ℕ := 142

/-- The number of pencils Josh gave away -/
def pencils_given_away : ℕ := 31

/-- The number of pencils Josh is left with -/
def remaining_pencils : ℕ := initial_pencils - pencils_given_away

theorem josh_pencils_calculation : remaining_pencils = 111 := by
  sorry

end NUMINAMATH_CALUDE_josh_pencils_calculation_l1389_138972


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l1389_138912

theorem least_integer_in_ratio (a b c : ℕ+) : 
  (a : ℝ) + (b : ℝ) + (c : ℝ) = 90 →
  (b : ℝ) = 3 * (a : ℝ) →
  (c : ℝ) = 5 * (a : ℝ) →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l1389_138912


namespace NUMINAMATH_CALUDE_division_problem_l1389_138964

theorem division_problem (A : ℕ) (h : 23 = A * 3 + 2) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1389_138964


namespace NUMINAMATH_CALUDE_faucet_filling_time_l1389_138952

/-- Given that four faucets can fill a 120-gallon tub in 5 minutes,
    prove that two faucets can fill a 60-gallon tub in 5 minutes. -/
theorem faucet_filling_time 
  (tub_capacity : ℝ) 
  (filling_time : ℝ) 
  (faucet_count : ℕ) :
  tub_capacity = 120 ∧ 
  filling_time = 5 ∧ 
  faucet_count = 4 →
  ∃ (new_tub_capacity : ℝ) (new_faucet_count : ℕ),
    new_tub_capacity = 60 ∧
    new_faucet_count = 2 ∧
    (new_tub_capacity / new_faucet_count) / (tub_capacity / faucet_count) * filling_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_faucet_filling_time_l1389_138952


namespace NUMINAMATH_CALUDE_max_value_of_f_l1389_138980

def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem max_value_of_f :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1389_138980


namespace NUMINAMATH_CALUDE_group_size_from_circular_arrangements_l1389_138913

/-- The number of ways to arrange k people from a group of n people around a circular table. -/
def circularArrangements (n k : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: If there are 144 ways to seat 5 people around a circular table from a group of n people, then n = 7. -/
theorem group_size_from_circular_arrangements (n : ℕ) 
  (h : circularArrangements n 5 = 144) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_group_size_from_circular_arrangements_l1389_138913


namespace NUMINAMATH_CALUDE_license_plate_count_l1389_138982

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letter positions in the license plate -/
def num_letter_positions : ℕ := 3

/-- The number of digit positions in the license plate -/
def num_digit_positions : ℕ := 3

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_letters ^ num_letter_positions * num_digits ^ num_digit_positions

theorem license_plate_count : total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1389_138982


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1389_138904

theorem complex_number_quadrant (z : ℂ) : z * (1 - Complex.I) = (1 + 2 * Complex.I) * Complex.I →
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1389_138904


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1389_138958

def A : Set ℤ := {0, 1}
def B : Set ℤ := {0, -1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1389_138958


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1389_138933

theorem infinite_geometric_series_ratio (a S : ℝ) (h1 : a = 400) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 21 / 25 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1389_138933


namespace NUMINAMATH_CALUDE_bicycle_trip_speed_l1389_138949

/-- Proves that given a total distance of 400 km, with the first 100 km traveled at 20 km/h
    and an average speed of 16 km/h for the entire trip, the speed for the remainder of the trip is 15 km/h. -/
theorem bicycle_trip_speed (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ)
  (h1 : total_distance = 400)
  (h2 : first_part_distance = 100)
  (h3 : first_part_speed = 20)
  (h4 : average_speed = 16) :
  let remainder_distance := total_distance - first_part_distance
  let total_time := total_distance / average_speed
  let first_part_time := first_part_distance / first_part_speed
  let remainder_time := total_time - first_part_time
  remainder_distance / remainder_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_trip_speed_l1389_138949


namespace NUMINAMATH_CALUDE_divisible_by_three_l1389_138909

def five_digit_number (n : Nat) : Nat :=
  52000 + n * 100 + 48

theorem divisible_by_three (n : Nat) : 
  n < 10 → (five_digit_number n % 3 = 0 ↔ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l1389_138909


namespace NUMINAMATH_CALUDE_negative_abs_equal_l1389_138966

theorem negative_abs_equal : -|5| = -|-5| := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_equal_l1389_138966


namespace NUMINAMATH_CALUDE_machine_working_time_l1389_138974

theorem machine_working_time 
  (total_shirts : ℕ) 
  (production_rate : ℕ) 
  (num_malfunctions : ℕ) 
  (malfunction_fix_time : ℕ) 
  (h1 : total_shirts = 360)
  (h2 : production_rate = 4)
  (h3 : num_malfunctions = 2)
  (h4 : malfunction_fix_time = 5) :
  (total_shirts / production_rate) + (num_malfunctions * malfunction_fix_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_l1389_138974


namespace NUMINAMATH_CALUDE_mica_pasta_purchase_l1389_138943

theorem mica_pasta_purchase (pasta_price : ℝ) (beef_price : ℝ) (sauce_price : ℝ) 
  (quesadilla_price : ℝ) (total_budget : ℝ) :
  pasta_price = 1.5 →
  beef_price = 8 →
  sauce_price = 2 →
  quesadilla_price = 6 →
  total_budget = 15 →
  (total_budget - (beef_price * 0.25 + sauce_price * 2 + quesadilla_price)) / pasta_price = 2 :=
by
  sorry

#check mica_pasta_purchase

end NUMINAMATH_CALUDE_mica_pasta_purchase_l1389_138943


namespace NUMINAMATH_CALUDE_remainder_not_always_same_l1389_138919

theorem remainder_not_always_same (a b : ℕ) :
  (3 * a + b) % 10 = (3 * b + a) % 10 →
  ¬(a % 10 = b % 10) :=
by sorry

end NUMINAMATH_CALUDE_remainder_not_always_same_l1389_138919


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1389_138908

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x^2 - (a^2 - 2*a - 15)*x + (a - 1) = 0 ∧ 
               y^2 - (a^2 - 2*a - 15)*y + (a - 1) = 0 ∧ 
               x = -y) → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1389_138908


namespace NUMINAMATH_CALUDE_jelly_cost_l1389_138948

theorem jelly_cost (N B J : ℕ) (h1 : N = 15) 
  (h2 : 6 * B * N + 7 * J * N = 315) 
  (h3 : B > 0) (h4 : J > 0) : 
  7 * J * N / 100 = 315 / 100 := by
  sorry

end NUMINAMATH_CALUDE_jelly_cost_l1389_138948


namespace NUMINAMATH_CALUDE_amount_per_bulb_is_fifty_cents_l1389_138951

/-- The amount Jane earned for planting flower bulbs -/
def total_earned : ℚ := 75

/-- The number of tulip bulbs Jane planted -/
def tulip_bulbs : ℕ := 20

/-- The number of daffodil bulbs Jane planted -/
def daffodil_bulbs : ℕ := 30

/-- The number of iris bulbs Jane planted -/
def iris_bulbs : ℕ := tulip_bulbs / 2

/-- The number of crocus bulbs Jane planted -/
def crocus_bulbs : ℕ := 3 * daffodil_bulbs

/-- The total number of bulbs Jane planted -/
def total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs

/-- The amount paid per bulb -/
def amount_per_bulb : ℚ := total_earned / total_bulbs

theorem amount_per_bulb_is_fifty_cents : amount_per_bulb = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_amount_per_bulb_is_fifty_cents_l1389_138951


namespace NUMINAMATH_CALUDE_technicians_avg_salary_is_900_l1389_138935

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary_all : ℕ
  num_technicians : ℕ
  avg_salary_non_tech : ℕ

/-- Calculates the average salary of technicians given workshop data -/
def avg_salary_technicians (w : Workshop) : ℕ :=
  let total_salary := w.total_workers * w.avg_salary_all
  let non_tech_workers := w.total_workers - w.num_technicians
  let non_tech_salary := non_tech_workers * w.avg_salary_non_tech
  let tech_salary := total_salary - non_tech_salary
  tech_salary / w.num_technicians

/-- Theorem stating that the average salary of technicians is 900 given the workshop conditions -/
theorem technicians_avg_salary_is_900 (w : Workshop) 
  (h1 : w.total_workers = 20)
  (h2 : w.avg_salary_all = 750)
  (h3 : w.num_technicians = 5)
  (h4 : w.avg_salary_non_tech = 700) :
  avg_salary_technicians w = 900 := by
  sorry

#eval avg_salary_technicians ⟨20, 750, 5, 700⟩

end NUMINAMATH_CALUDE_technicians_avg_salary_is_900_l1389_138935


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1389_138901

/-- Proves that given a train of length 75 meters, which crosses a bridge in 7.5 seconds
    and a lamp post on the bridge in 2.5 seconds, the length of the bridge is 150 meters. -/
theorem bridge_length_calculation (train_length : ℝ) (bridge_crossing_time : ℝ) (lamppost_crossing_time : ℝ)
  (h1 : train_length = 75)
  (h2 : bridge_crossing_time = 7.5)
  (h3 : lamppost_crossing_time = 2.5) :
  let bridge_length := (train_length * bridge_crossing_time / lamppost_crossing_time) - train_length
  bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1389_138901


namespace NUMINAMATH_CALUDE_total_jokes_over_two_saturdays_l1389_138994

/-- 
Given that Jessy told 11 jokes and Alan told 7 jokes on the first Saturday,
and they both double their jokes on the second Saturday, prove that the
total number of jokes told over both Saturdays is 54.
-/
theorem total_jokes_over_two_saturdays 
  (jessy_first : ℕ) 
  (alan_first : ℕ) 
  (h1 : jessy_first = 11) 
  (h2 : alan_first = 7) : 
  jessy_first + alan_first + 2 * jessy_first + 2 * alan_first = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_jokes_over_two_saturdays_l1389_138994


namespace NUMINAMATH_CALUDE_optimal_tax_theorem_l1389_138957

/-- Market model with linear demand and supply functions, and a per-unit tax --/
structure MarketModel where
  -- Demand function: Qd = a - bP
  a : ℝ
  b : ℝ
  -- Supply function: Qs = cP + d
  c : ℝ
  d : ℝ
  -- Elasticity ratio at equilibrium
  elasticity_ratio : ℝ
  -- Tax amount
  tax : ℝ
  -- Producer price after tax
  producer_price : ℝ

/-- Finds the optimal tax rate and resulting revenue for a given market model --/
def optimal_tax_and_revenue (model : MarketModel) : ℝ × ℝ :=
  sorry

/-- The main theorem stating the optimal tax and revenue for the given market conditions --/
theorem optimal_tax_theorem (model : MarketModel) :
  model.a = 688 ∧
  model.b = 4 ∧
  model.elasticity_ratio = 1.5 ∧
  model.tax = 90 ∧
  model.producer_price = 64 →
  optimal_tax_and_revenue model = (54, 10800) :=
sorry

end NUMINAMATH_CALUDE_optimal_tax_theorem_l1389_138957


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l1389_138965

theorem triangle_abc_problem (a b c : ℝ) (A B C : ℝ) :
  A = π / 6 →
  (1 + Real.sqrt 3) * c = 2 * b →
  b * a * Real.cos C = 1 + Real.sqrt 3 →
  C = π / 4 ∧ a = Real.sqrt 2 ∧ b = 1 + Real.sqrt 3 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l1389_138965


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l1389_138960

theorem solution_set_implies_a_range (a : ℝ) : 
  (∃ P : Set ℝ, (∀ x ∈ P, (x + 1) / (x + a) < 2) ∧ 1 ∉ P) → 
  a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l1389_138960


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1389_138955

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1389_138955


namespace NUMINAMATH_CALUDE_no_k_for_always_negative_quadratic_l1389_138984

theorem no_k_for_always_negative_quadratic :
  ¬ ∃ k : ℝ, ∀ x : ℝ, x^2 - (k + 4) * x + k - 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_k_for_always_negative_quadratic_l1389_138984


namespace NUMINAMATH_CALUDE_adoption_time_l1389_138963

def initial_puppies : ℕ := 3
def new_puppies : ℕ := 3
def adoption_rate : ℕ := 3

theorem adoption_time :
  (initial_puppies + new_puppies) / adoption_rate = 2 :=
sorry

end NUMINAMATH_CALUDE_adoption_time_l1389_138963


namespace NUMINAMATH_CALUDE_equation_solution_l1389_138956

theorem equation_solution : ∃ x : ℝ, 
  (0.5^3 - x^3 / 0.5^2 + 0.05 + 0.1^2 = 0.4) ∧ 
  (abs (x + 0.378) < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1389_138956


namespace NUMINAMATH_CALUDE_at_most_one_true_l1389_138996

theorem at_most_one_true (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_true_l1389_138996


namespace NUMINAMATH_CALUDE_polynomial_identity_l1389_138925

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ x : ℝ, P (x^3 - 2) = P x^3 - 2) ↔ (∀ x : ℝ, P x = x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1389_138925


namespace NUMINAMATH_CALUDE_stating_external_diagonals_inequality_invalid_external_diagonals_l1389_138934

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- 
Theorem stating that for a valid set of external diagonal lengths of a right regular prism,
the sum of squares of the two smaller lengths is greater than or equal to 
the square of the largest length.
-/
theorem external_diagonals_inequality (d : ExternalDiagonals) : d.a ^ 2 + d.b ^ 2 ≥ d.c ^ 2 := by
  sorry

/-- 
Proves that {6, 8, 11} cannot be the lengths of external diagonals of a right regular prism
-/
theorem invalid_external_diagonals : 
  ¬∃ (d : ExternalDiagonals), d.a = 6 ∧ d.b = 8 ∧ d.c = 11 := by
  sorry

end NUMINAMATH_CALUDE_stating_external_diagonals_inequality_invalid_external_diagonals_l1389_138934


namespace NUMINAMATH_CALUDE_buoy_radius_l1389_138978

/-- The radius of a buoy given the dimensions of the hole it leaves -/
theorem buoy_radius (hole_width : ℝ) (hole_depth : ℝ) : 
  hole_width = 30 → hole_depth = 10 → ∃ r : ℝ, r = 16.25 ∧ 
  ∃ x : ℝ, x^2 + (hole_width/2)^2 = (x + hole_depth)^2 ∧ r = x + hole_depth :=
by sorry

end NUMINAMATH_CALUDE_buoy_radius_l1389_138978


namespace NUMINAMATH_CALUDE_price_decrease_calculation_l1389_138931

/-- The original price of an article before a price decrease -/
def original_price : ℝ := 421.05

/-- The percentage of the original price after the decrease -/
def percentage_after_decrease : ℝ := 0.76

/-- The price of the article after the decrease -/
def price_after_decrease : ℝ := 320

/-- Theorem stating that the original price is correct given the conditions -/
theorem price_decrease_calculation :
  price_after_decrease = percentage_after_decrease * original_price := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_calculation_l1389_138931


namespace NUMINAMATH_CALUDE_x_value_proof_l1389_138900

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (144 : ℝ)^(1/7) :=
by sorry

end NUMINAMATH_CALUDE_x_value_proof_l1389_138900


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_difference_two_l1389_138923

def a (n : ℕ) : ℝ := 2 * (n + 1) + 3

theorem arithmetic_sequence_with_difference_two :
  ∀ n : ℕ, a (n + 1) - a n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_difference_two_l1389_138923


namespace NUMINAMATH_CALUDE_probability_two_odd_numbers_l1389_138987

/-- A fair eight-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8 

/-- The set of odd numbers on an eight-sided die -/
def OddNumbers : Finset ℕ := Finset.filter (fun x => x % 2 = 1) EightSidedDie

/-- The probability of an event occurring when rolling two fair eight-sided dice -/
def probability (event : Finset (ℕ × ℕ)) : ℚ :=
  event.card / (EightSidedDie.card * EightSidedDie.card)

/-- The event of rolling two odd numbers -/
def TwoOddNumbers : Finset (ℕ × ℕ) :=
  Finset.product OddNumbers OddNumbers

theorem probability_two_odd_numbers : probability TwoOddNumbers = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_odd_numbers_l1389_138987


namespace NUMINAMATH_CALUDE_parallel_iff_m_eq_four_l1389_138928

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (-((3 * m - 4) / 4) = -(m / 2))

/-- The condition for parallelism is equivalent to m = 4 -/
theorem parallel_iff_m_eq_four :
  ∀ m : ℝ, parallel_lines m ↔ m = 4 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_m_eq_four_l1389_138928


namespace NUMINAMATH_CALUDE_sqrt_50_minus_sqrt_48_approx_0_14_l1389_138979

theorem sqrt_50_minus_sqrt_48_approx_0_14 (ε : ℝ) (h : ε > 0) :
  ∃ δ > 0, |Real.sqrt 50 - Real.sqrt 48 - 0.14| < δ ∧ δ < ε :=
by sorry

end NUMINAMATH_CALUDE_sqrt_50_minus_sqrt_48_approx_0_14_l1389_138979


namespace NUMINAMATH_CALUDE_conditions_for_a_and_b_l1389_138976

/-- Given a system of equations, prove the conditions for a and b -/
theorem conditions_for_a_and_b (a b x y : ℝ) : 
  (x^2 + x*y + y^2 - y = 0) →
  (a * x^2 + b * x * y + x = 0) →
  ((a + 1)^2 = 4*(b + 1) ∧ b ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_conditions_for_a_and_b_l1389_138976


namespace NUMINAMATH_CALUDE_units_digit_of_squares_l1389_138930

theorem units_digit_of_squares (n : ℕ) : 
  (n ≥ 10 ∧ n ≤ 99) → 
  (n % 10 = 2 ∨ n % 10 = 7) → 
  (n^2 % 10 ≠ 2 ∧ n^2 % 10 ≠ 6 ∧ n^2 % 10 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_squares_l1389_138930


namespace NUMINAMATH_CALUDE_line_direction_vector_l1389_138918

def point := ℝ × ℝ

-- Define the two points on the line
def p1 : point := (-3, 4)
def p2 : point := (2, -1)

-- Define the direction vector type
def direction_vector := ℝ × ℝ

-- Function to calculate the direction vector between two points
def calc_direction_vector (p q : point) : direction_vector :=
  (q.1 - p.1, q.2 - p.2)

-- Function to scale a vector
def scale_vector (v : direction_vector) (s : ℝ) : direction_vector :=
  (s * v.1, s * v.2)

-- Theorem statement
theorem line_direction_vector : 
  ∃ (a : ℝ), calc_direction_vector p1 p2 = scale_vector (a, 2) (-5/2) ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1389_138918


namespace NUMINAMATH_CALUDE_gcd_divisibility_and_multiple_l1389_138986

theorem gcd_divisibility_and_multiple (a b n : ℕ) (h : a ≠ 0) :
  let d := Nat.gcd a b
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧
  ∀ c : ℕ, c > 0 → Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by sorry

end NUMINAMATH_CALUDE_gcd_divisibility_and_multiple_l1389_138986


namespace NUMINAMATH_CALUDE_eighth_power_sum_l1389_138970

theorem eighth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^8 + b^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_eighth_power_sum_l1389_138970


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l1389_138985

theorem pyramid_height_equals_cube_volume (cube_edge : Real) (pyramid_base : Real) (pyramid_height : Real) : 
  cube_edge = 6 →
  pyramid_base = 12 →
  (1 / 3) * pyramid_base^2 * pyramid_height = cube_edge^3 →
  pyramid_height = 4.5 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l1389_138985


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1389_138921

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℤ)
  (h_arithmetic : ArithmeticSequence a)
  (h_21st : a 21 = 26)
  (h_22nd : a 22 = 30) :
  a 5 = -38 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1389_138921


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l1389_138926

/-- The width of the painting in inches -/
def painting_width : ℝ := 20

/-- The height of the painting in inches -/
def painting_height : ℝ := 30

/-- The frame width on the sides in inches -/
def frame_side_width : ℝ := 5

/-- The frame width on the top and bottom in inches -/
def frame_top_bottom_width : ℝ := 3 * frame_side_width

/-- The area of the painting in square inches -/
def painting_area : ℝ := painting_width * painting_height

/-- The area of the framed painting in square inches -/
def framed_painting_area : ℝ := (painting_width + 2 * frame_side_width) * (painting_height + 2 * frame_top_bottom_width)

/-- The width of the framed painting in inches -/
def framed_painting_width : ℝ := painting_width + 2 * frame_side_width

/-- The height of the framed painting in inches -/
def framed_painting_height : ℝ := painting_height + 2 * frame_top_bottom_width

theorem framed_painting_ratio :
  framed_painting_area = 2 * painting_area ∧
  framed_painting_width / framed_painting_height = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l1389_138926


namespace NUMINAMATH_CALUDE_pedros_daughters_l1389_138999

/-- The number of ice cream flavors available -/
def num_flavors : ℕ := 12

/-- The number of scoops in each child's combo -/
def scoops_per_combo : ℕ := 3

/-- The total number of scoops ordered for each flavor -/
def scoops_per_flavor : ℕ := 2

structure Family where
  num_boys : ℕ
  num_girls : ℕ

/-- Pedro's family satisfies the given conditions -/
def is_valid_family (f : Family) : Prop :=
  f.num_boys > 0 ∧ 
  f.num_girls > f.num_boys ∧
  (f.num_boys + f.num_girls) * scoops_per_combo = num_flavors * scoops_per_flavor ∧
  ∃ (boys_flavors girls_flavors : Finset ℕ), 
    boys_flavors.card = (3 * f.num_boys) / 2 ∧
    girls_flavors.card = (3 * f.num_girls) / 2 ∧
    boys_flavors ∩ girls_flavors = ∅ ∧
    boys_flavors ∪ girls_flavors = Finset.range num_flavors

theorem pedros_daughters (f : Family) (h : is_valid_family f) : f.num_girls = 6 :=
sorry

end NUMINAMATH_CALUDE_pedros_daughters_l1389_138999


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l1389_138993

theorem opposite_of_negative_one_half : 
  ∃ (x : ℚ), x + (-1/2) = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l1389_138993


namespace NUMINAMATH_CALUDE_complex_equal_modulus_unequal_square_exists_l1389_138961

theorem complex_equal_modulus_unequal_square_exists : 
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equal_modulus_unequal_square_exists_l1389_138961


namespace NUMINAMATH_CALUDE_S_is_valid_set_l1389_138939

-- Define the set of numbers greater than √2
def S : Set ℝ := {x : ℝ | x > Real.sqrt 2}

-- Theorem stating that S is a valid set
theorem S_is_valid_set : 
  (∀ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y → x ≠ y) ∧  -- Elements are distinct
  (∀ x y, x ∈ S ∧ y ∈ S → y ∈ S ∧ x ∈ S) ∧  -- Elements are unordered
  (∀ x, x ∈ S ↔ x > Real.sqrt 2)  -- Elements are determined
  := by sorry

end NUMINAMATH_CALUDE_S_is_valid_set_l1389_138939


namespace NUMINAMATH_CALUDE_diamond_four_three_l1389_138988

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a^2 + a*b - b^3

-- Theorem statement
theorem diamond_four_three : diamond 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_diamond_four_three_l1389_138988


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1389_138967

theorem hyperbola_asymptotes (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (2 * Real.sqrt (-1/m) = 4) →
  (∀ x y : ℝ, y = 2*x ∨ y = -2*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1389_138967


namespace NUMINAMATH_CALUDE_only_B_on_x_axis_l1389_138992

def point_A : ℝ × ℝ := (-2, -3)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (0, 3)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis :
  is_on_x_axis point_B ∧
  ¬is_on_x_axis point_A ∧
  ¬is_on_x_axis point_C ∧
  ¬is_on_x_axis point_D :=
by sorry

end NUMINAMATH_CALUDE_only_B_on_x_axis_l1389_138992


namespace NUMINAMATH_CALUDE_surface_is_cone_l1389_138907

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of the surface --/
def surfaceEquation (p : Point3D) (a b c d θ : ℝ) : Prop :=
  (p.x - a)^2 + (p.y - b)^2 + (p.z - c)^2 = (d * Real.cos θ)^2

/-- The set of points satisfying the equation --/
def surfaceSet (a b c d : ℝ) : Set Point3D :=
  {p : Point3D | ∃ θ, surfaceEquation p a b c d θ}

/-- Definition of a cone --/
def isCone (S : Set Point3D) : Prop :=
  ∃ v : Point3D, ∃ axis : Point3D → Point3D → Prop,
    ∀ p ∈ S, ∃ r θ : ℝ, r ≥ 0 ∧ 
      p = Point3D.mk (v.x + r * Real.cos θ) (v.y + r * Real.sin θ) (v.z + r)

theorem surface_is_cone (d : ℝ) (h : d > 0) :
  isCone (surfaceSet 0 0 0 d) := by
  sorry

end NUMINAMATH_CALUDE_surface_is_cone_l1389_138907


namespace NUMINAMATH_CALUDE_zoe_chocolate_sales_l1389_138975

/-- Given a box of chocolate bars, calculate the money made from selling a certain number of bars. -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Prove that Zoe made $42 by selling all but 6 bars from a box of 13 bars, each costing $6. -/
theorem zoe_chocolate_sales : money_made 13 6 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_zoe_chocolate_sales_l1389_138975


namespace NUMINAMATH_CALUDE_a_eq_4_neither_sufficient_nor_necessary_l1389_138981

/-- Two lines in the real plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line l₁: ax + 8y - 8 = 0 -/
def l1 (a : ℝ) : Line :=
  { a := a, b := 8, c := -8 }

/-- The second line l₂: 2x + ay - a = 0 -/
def l2 (a : ℝ) : Line :=
  { a := 2, b := a, c := -a }

/-- The main theorem stating that a = 4 is neither sufficient nor necessary for parallelism -/
theorem a_eq_4_neither_sufficient_nor_necessary :
  ∃ a : ℝ, a ≠ 4 ∧ parallel (l1 a) (l2 a) ∧
  ∃ b : ℝ, b = 4 ∧ ¬parallel (l1 b) (l2 b) :=
sorry

end NUMINAMATH_CALUDE_a_eq_4_neither_sufficient_nor_necessary_l1389_138981


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l1389_138929

/-- Represents the dimensions of a rectangular prism --/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes in a rectangular prism given its dimensions --/
def cubesInPrism (d : PrismDimensions) : ℕ := d.width * d.length * d.height

/-- Theorem: The number of cubes not touching tin foil in the given prism is 128 --/
theorem cubes_not_touching_foil (prism_width : ℕ) (inner_prism : PrismDimensions) : 
  prism_width = 10 →
  inner_prism.width = 2 * inner_prism.length →
  inner_prism.width = 2 * inner_prism.height →
  inner_prism.width ≤ prism_width - 2 →
  cubesInPrism inner_prism = 128 := by
  sorry

#check cubes_not_touching_foil

end NUMINAMATH_CALUDE_cubes_not_touching_foil_l1389_138929


namespace NUMINAMATH_CALUDE_point_on_line_line_slope_is_one_line_equation_correct_l1389_138940

/-- A line passing through the point (1, 3) with slope 1 -/
def line (x y : ℝ) : Prop := x - y + 2 = 0

/-- The point (1, 3) lies on the line -/
theorem point_on_line : line 1 3 := by sorry

/-- The slope of the line is 1 -/
theorem line_slope_is_one :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → line x₁ y₁ → line x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = 1 := by sorry

/-- The equation x - y + 2 = 0 represents the unique line passing through (1, 3) with slope 1 -/
theorem line_equation_correct :
  ∀ (x y : ℝ), (x - y + 2 = 0) ↔ (∃ (m b : ℝ), m = 1 ∧ y = m * (x - 1) + 3) := by sorry

end NUMINAMATH_CALUDE_point_on_line_line_slope_is_one_line_equation_correct_l1389_138940


namespace NUMINAMATH_CALUDE_pascal_row10_sums_l1389_138968

/-- Represents a row in Pascal's Triangle -/
def PascalRow (n : ℕ) := Fin (n + 1) → ℕ

/-- The 10th row of Pascal's Triangle -/
def row10 : PascalRow 10 := sorry

/-- Sum of elements in a Pascal's Triangle row -/
def row_sum (n : ℕ) (row : PascalRow n) : ℕ := sorry

/-- Sum of squares of elements in a Pascal's Triangle row -/
def row_sum_of_squares (n : ℕ) (row : PascalRow n) : ℕ := sorry

theorem pascal_row10_sums :
  (row_sum 10 row10 = 2^10) ∧
  (row_sum_of_squares 10 row10 = 183756) := by sorry

end NUMINAMATH_CALUDE_pascal_row10_sums_l1389_138968


namespace NUMINAMATH_CALUDE_midpoint_fraction_l1389_138905

theorem midpoint_fraction : ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d = (3 : ℚ) / 4 / 2 + (5 : ℚ) / 6 / 2 ∧ n = 19 ∧ d = 24 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l1389_138905


namespace NUMINAMATH_CALUDE_journey_length_l1389_138973

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total +        -- First part (dirt road)
  30 +                         -- Second part (highway)
  (1 / 7 : ℚ) * total =        -- Third part (city street)
  total →                      -- Sum of all parts equals total
  total = 840 / 17 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l1389_138973


namespace NUMINAMATH_CALUDE_range_of_x_l1389_138938

theorem range_of_x (x : ℝ) : (1 + 2*x ≤ 8 + 3*x) → x ≥ -7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1389_138938


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1389_138937

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 1 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1389_138937


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l1389_138911

/-- The quadratic function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The range of x values we're considering -/
def range : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }

theorem max_value_of_quadratic :
  ∃ (m : ℝ), m = 36 ∧ ∀ x ∈ range, f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l1389_138911


namespace NUMINAMATH_CALUDE_profit_per_meter_l1389_138950

/-- Given a cloth sale scenario, calculate the profit per meter. -/
theorem profit_per_meter
  (meters_sold : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : meters_sold = 60)
  (h2 : total_selling_price = 8400)
  (h3 : cost_price_per_meter = 128) :
  (total_selling_price - meters_sold * cost_price_per_meter) / meters_sold = 12 := by
sorry


end NUMINAMATH_CALUDE_profit_per_meter_l1389_138950


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1389_138906

def circle1_center : ℝ × ℝ := (-32, 42)
def circle2_center : ℝ × ℝ := (0, 0)
def circle1_radius : ℝ := 52
def circle2_radius : ℝ := 3

theorem circles_externally_tangent :
  let d := Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)
  d = circle1_radius + circle2_radius := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1389_138906


namespace NUMINAMATH_CALUDE_farmer_land_area_l1389_138915

theorem farmer_land_area : ∃ (total : ℚ),
  total > 0 ∧
  total / 3 + total / 4 + total / 5 + 26 = total ∧
  total = 120 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_area_l1389_138915


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l1389_138916

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l1389_138916


namespace NUMINAMATH_CALUDE_twenty_men_handshakes_l1389_138971

/-- The maximum number of handshakes without cyclic handshakes for n people -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 20 men, the maximum number of handshakes without cyclic handshakes is 190 -/
theorem twenty_men_handshakes :
  max_handshakes 20 = 190 := by
  sorry

#eval max_handshakes 20  -- This will evaluate to 190

end NUMINAMATH_CALUDE_twenty_men_handshakes_l1389_138971


namespace NUMINAMATH_CALUDE_susan_remaining_moves_l1389_138990

/-- Represents the board game with 100 spaces -/
def BoardGame := 100

/-- Susan's movements over 7 turns -/
def susanMoves : List ℤ := [15, 2, 20, 0, 2, 0, 12]

/-- The total distance Susan has moved -/
def totalDistance : ℤ := susanMoves.sum

/-- Theorem: Susan needs to move 49 more spaces to reach the end -/
theorem susan_remaining_moves : BoardGame - totalDistance = 49 := by
  sorry

end NUMINAMATH_CALUDE_susan_remaining_moves_l1389_138990


namespace NUMINAMATH_CALUDE_student_selection_probability_l1389_138989

theorem student_selection_probability (b g o : ℝ) : 
  b + g + o = 1 →  -- total probability
  b > 0 ∧ g > 0 ∧ o > 0 →  -- probabilities are positive
  b = (1/2) * o →  -- boy probability is half of other
  g = o - b →  -- girl probability is difference between other and boy
  b = 1/4 :=  -- ratio of boys to total is 1/4
by sorry

end NUMINAMATH_CALUDE_student_selection_probability_l1389_138989


namespace NUMINAMATH_CALUDE_total_volume_is_716_l1389_138927

/-- The volume of a cube with side length s -/
def cube_volume (s : ℝ) : ℝ := s ^ 3

/-- The number of cubes Carl has -/
def carl_cubes : ℕ := 8

/-- The side length of Carl's cubes -/
def carl_side_length : ℝ := 3

/-- The number of cubes Kate has -/
def kate_cubes : ℕ := 4

/-- The side length of Kate's cubes -/
def kate_side_length : ℝ := 5

/-- The total volume of all cubes -/
def total_volume : ℝ :=
  (carl_cubes : ℝ) * cube_volume carl_side_length +
  (kate_cubes : ℝ) * cube_volume kate_side_length

theorem total_volume_is_716 : total_volume = 716 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_is_716_l1389_138927


namespace NUMINAMATH_CALUDE_admission_price_is_12_l1389_138977

/-- The admission price for the aqua park. -/
def admission_price : ℝ := sorry

/-- The price of the tour. -/
def tour_price : ℝ := 6

/-- The number of people in the first group (who take the tour). -/
def group1_size : ℕ := 10

/-- The number of people in the second group (who only pay admission). -/
def group2_size : ℕ := 5

/-- The total earnings of the aqua park. -/
def total_earnings : ℝ := 240

/-- Theorem stating that the admission price is $12 given the conditions. -/
theorem admission_price_is_12 :
  (group1_size : ℝ) * (admission_price + tour_price) + (group2_size : ℝ) * admission_price = total_earnings →
  admission_price = 12 := by
  sorry

end NUMINAMATH_CALUDE_admission_price_is_12_l1389_138977


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1389_138936

theorem rectangle_measurement_error (L W : ℝ) (L_excess W_deficit : ℝ) 
  (h1 : L_excess = 1.20)  -- 20% excess on first side
  (h2 : W_deficit > 0)    -- deficit percentage is positive
  (h3 : W_deficit < 1)    -- deficit percentage is less than 100%
  (h4 : L_excess * (1 - W_deficit) = 1.08)  -- 8% error in area
  : W_deficit = 0.10 :=   -- 10% deficit on second side
by sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1389_138936


namespace NUMINAMATH_CALUDE_share_difference_after_tax_l1389_138942

/-- Represents the share ratios for p, q, r, and s respectively -/
def shareRatios : Fin 4 → ℕ
  | 0 => 3
  | 1 => 7
  | 2 => 12
  | 3 => 5

/-- Represents the tax rates for p, q, r, and s respectively -/
def taxRates : Fin 4 → ℚ
  | 0 => 1/10
  | 1 => 15/100
  | 2 => 1/5
  | 3 => 1/4

/-- The difference between p and q's shares after tax deduction -/
def differenceAfterTax : ℚ := 2400

theorem share_difference_after_tax :
  let x : ℚ := differenceAfterTax / (shareRatios 1 * (1 - taxRates 1) - shareRatios 0 * (1 - taxRates 0))
  let qShare : ℚ := shareRatios 1 * x * (1 - taxRates 1)
  let rShare : ℚ := shareRatios 2 * x * (1 - taxRates 2)
  abs (rShare - qShare - 2695.38) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_after_tax_l1389_138942


namespace NUMINAMATH_CALUDE_female_students_count_l1389_138945

/-- Represents the class configuration described in the problem -/
structure ClassConfiguration where
  total_students : Nat
  male_students : Nat
  (total_ge_male : total_students ≥ male_students)

/-- The number of students called by the kth student -/
def students_called (k : Nat) : Nat := k + 2

/-- The theorem statement -/
theorem female_students_count (c : ClassConfiguration) 
  (h1 : c.total_students = 42)
  (h2 : ∀ k, k ≤ c.male_students → students_called k ≤ c.total_students)
  (h3 : students_called c.male_students = c.total_students / 2) :
  c.total_students - c.male_students = 23 := by
  sorry


end NUMINAMATH_CALUDE_female_students_count_l1389_138945


namespace NUMINAMATH_CALUDE_nabla_neg_five_neg_seven_l1389_138917

def nabla (a b : ℝ) : ℝ := a * b + a - b

theorem nabla_neg_five_neg_seven : nabla (-5) (-7) = 37 := by
  sorry

end NUMINAMATH_CALUDE_nabla_neg_five_neg_seven_l1389_138917


namespace NUMINAMATH_CALUDE_range_of_m_l1389_138914

theorem range_of_m (p : ℝ → Prop) (m : ℝ) 
  (h1 : ∀ x, p x ↔ x^2 + 2*x - m > 0)
  (h2 : ¬ p 1)
  (h3 : p 2) :
  3 ≤ m ∧ m < 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1389_138914


namespace NUMINAMATH_CALUDE_chord_equation_l1389_138920

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a circle with center (0,0) and radius 3 -/
def isOnCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 9

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m p q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

/-- Checks if a line passes through a point -/
def linePassesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem chord_equation (p q : Point) (m : Point) :
  isOnCircle p ∧ isOnCircle q ∧ isMidpoint m p q ∧ m.x = 1 ∧ m.y = 2 →
  ∃ l : Line, l.a = 1 ∧ l.b = 2 ∧ l.c = -5 ∧ linePassesThrough l p ∧ linePassesThrough l q :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l1389_138920


namespace NUMINAMATH_CALUDE_milk_container_percentage_difference_l1389_138902

/-- Given a scenario where milk is transferred between containers, this theorem proves
    the percentage difference between the quantity in one container and the original capacity. -/
theorem milk_container_percentage_difference
  (total_milk : ℝ)
  (transfer_amount : ℝ)
  (h_total : total_milk = 1216)
  (h_transfer : transfer_amount = 152)
  (h_equal_after_transfer : ∃ (b c : ℝ), b + c = total_milk ∧ b + transfer_amount = c - transfer_amount) :
  ∃ (b : ℝ), (total_milk - b) / total_milk * 100 = 56.25 := by
  sorry

#eval (1216 - 532) / 1216 * 100  -- Should output approximately 56.25

end NUMINAMATH_CALUDE_milk_container_percentage_difference_l1389_138902


namespace NUMINAMATH_CALUDE_spinner_probability_l1389_138944

def spinner_A : Finset ℕ := {1, 2, 3}
def spinner_B : Finset ℕ := {2, 3, 4}

def is_multiple_of_four (n : ℕ) : Bool :=
  n % 4 = 0

def total_outcomes : ℕ :=
  (spinner_A.card) * (spinner_B.card)

def favorable_outcomes : ℕ :=
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (1 + b))).card +
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (2 + b))).card +
  (spinner_A.card) * (spinner_B.filter (λ b => is_multiple_of_four (3 + b))).card

theorem spinner_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_spinner_probability_l1389_138944


namespace NUMINAMATH_CALUDE_multiples_properties_l1389_138983

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) ∧ 
  (∃ q : ℤ, a - b = 2 * q) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l1389_138983


namespace NUMINAMATH_CALUDE_neighborhood_households_l1389_138946

theorem neighborhood_households (no_car_no_bike : ℕ) (car_and_bike : ℕ) (total_with_car : ℕ) (bike_only : ℕ) :
  no_car_no_bike = 11 →
  car_and_bike = 18 →
  total_with_car = 44 →
  bike_only = 35 →
  no_car_no_bike + car_and_bike + (total_with_car - car_and_bike) + bike_only = 90 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_households_l1389_138946


namespace NUMINAMATH_CALUDE_garden_flowers_l1389_138922

theorem garden_flowers (roses tulips : ℕ) (percent_not_roses : ℚ) (total daisies : ℕ) : 
  roses = 25 →
  tulips = 40 →
  percent_not_roses = 3/4 →
  total = roses + tulips + daisies →
  (total : ℚ) * (1 - percent_not_roses) = roses →
  daisies = 35 :=
by sorry

end NUMINAMATH_CALUDE_garden_flowers_l1389_138922


namespace NUMINAMATH_CALUDE_may_has_greatest_percentage_difference_l1389_138910

/-- Represents the sales data for a single month --/
structure MonthSales where
  drummers : ℕ
  bugles : ℕ
  flutes : ℕ

/-- Calculates the percentage difference for a given month's sales --/
def percentageDifference (sales : MonthSales) : ℚ :=
  let max := max sales.drummers (max sales.bugles sales.flutes)
  let min := min sales.drummers (min sales.bugles sales.flutes)
  (max - min : ℚ) / min * 100

/-- Sales data for each month --/
def januarySales : MonthSales := ⟨5, 4, 6⟩
def februarySales : MonthSales := ⟨6, 5, 6⟩
def marchSales : MonthSales := ⟨6, 6, 6⟩
def aprilSales : MonthSales := ⟨7, 5, 8⟩
def maySales : MonthSales := ⟨3, 5, 4⟩

/-- Theorem: May has the greatest percentage difference in sales --/
theorem may_has_greatest_percentage_difference :
  percentageDifference maySales > percentageDifference januarySales ∧
  percentageDifference maySales > percentageDifference februarySales ∧
  percentageDifference maySales > percentageDifference marchSales ∧
  percentageDifference maySales > percentageDifference aprilSales :=
by sorry


end NUMINAMATH_CALUDE_may_has_greatest_percentage_difference_l1389_138910


namespace NUMINAMATH_CALUDE_map_distance_conversion_l1389_138969

/-- Given a map scale where 1 inch represents 500 meters, 
    this theorem proves that a line segment of 7.25 inches 
    on the map represents 3625 meters in reality. -/
theorem map_distance_conversion 
  (scale : ℝ) 
  (map_length : ℝ) 
  (h1 : scale = 500) 
  (h2 : map_length = 7.25) : 
  map_length * scale = 3625 := by
sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l1389_138969


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l1389_138962

theorem circle_diameter_ratio (D C : Real) (h1 : D = 20) 
  (h2 : C > 0) (h3 : C < D) 
  (h4 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4) : 
  C = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l1389_138962


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l1389_138995

-- Define the systems of equations
def system1 (x y : ℚ) : Prop :=
  x - y = 3 ∧ 3 * x - 8 * y = 14

def system2 (x y : ℚ) : Prop :=
  3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33

-- Theorem for the first system
theorem solution_system1 :
  ∃ x y : ℚ, system1 x y ∧ x = 2 ∧ y = -1 :=
by sorry

-- Theorem for the second system
theorem solution_system2 :
  ∃ x y : ℚ, system2 x y ∧ x = 6 ∧ y = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l1389_138995


namespace NUMINAMATH_CALUDE_intersection_inequality_solution_l1389_138941

/-- Given two lines y = 3x + a and y = -2x + b that intersect at a point with x-coordinate -5,
    the solution set of the inequality 3x + a < -2x + b is {x ∈ ℝ | x < -5}. -/
theorem intersection_inequality_solution (a b : ℝ) :
  (∃ y, 3 * (-5) + a = y ∧ -2 * (-5) + b = y) →
  (∀ x, 3 * x + a < -2 * x + b ↔ x < -5) :=
by sorry

end NUMINAMATH_CALUDE_intersection_inequality_solution_l1389_138941


namespace NUMINAMATH_CALUDE_hcf_problem_l1389_138932

theorem hcf_problem (a b : ℕ+) (h1 : Nat.lcm a b % 11 = 0) 
  (h2 : Nat.lcm a b % 12 = 0) (h3 : max a b = 480) : Nat.gcd a b = 40 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1389_138932


namespace NUMINAMATH_CALUDE_raccoon_stall_time_l1389_138924

/-- Proves that the time both locks together stall the raccoons is 60 minutes -/
theorem raccoon_stall_time : ∀ (t1 t2 t_both : ℕ),
  t1 = 5 →
  t2 = 3 * t1 - 3 →
  t_both = 5 * t2 →
  t_both = 60 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_stall_time_l1389_138924


namespace NUMINAMATH_CALUDE_probability_is_one_third_l1389_138998

/-- A rectangle in the xy-plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability that a randomly chosen point (x,y) from the given rectangle satisfies x > 2y --/
def probability_x_gt_2y (rect : Rectangle) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle := {
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 1
  h_x := by norm_num
  h_y := by norm_num
}

theorem probability_is_one_third :
  probability_x_gt_2y problem_rectangle = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l1389_138998


namespace NUMINAMATH_CALUDE_day_relationship_l1389_138954

/-- Represents days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the weekday of a given YearDay -/
def weekday_of_yearday : YearDay → Weekday := sorry

/-- Theorem stating the relationship between the given days and their weekdays -/
theorem day_relationship (N : Int) :
  (weekday_of_yearday ⟨N, 250⟩ = Weekday.Wednesday) →
  (weekday_of_yearday ⟨N + 1, 150⟩ = Weekday.Wednesday) →
  (weekday_of_yearday ⟨N - 1, 50⟩ = Weekday.Saturday) :=
by sorry

end NUMINAMATH_CALUDE_day_relationship_l1389_138954


namespace NUMINAMATH_CALUDE_last_digit_of_tower_of_power_l1389_138959

def tower_of_power (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | n + 1 => 2^(tower_of_power n)

theorem last_digit_of_tower_of_power :
  tower_of_power 2007 % 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_tower_of_power_l1389_138959


namespace NUMINAMATH_CALUDE_function_inequality_implies_unique_a_l1389_138953

theorem function_inequality_implies_unique_a :
  ∀ (a : ℝ),
  (∀ (x : ℝ), Real.exp x + a * (x^2 - x) - Real.cos x ≥ 0) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_unique_a_l1389_138953
