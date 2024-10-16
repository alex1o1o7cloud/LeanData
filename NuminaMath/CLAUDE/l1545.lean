import Mathlib

namespace NUMINAMATH_CALUDE_smaller_circle_area_l1545_154547

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  -- Centers of the smaller and larger circles
  S : ℝ × ℝ
  L : ℝ × ℝ
  -- Radii of the smaller and larger circles
  r_small : ℝ
  r_large : ℝ
  -- Point P from which tangents are drawn
  P : ℝ × ℝ
  -- Points of tangency on the circles
  A : ℝ × ℝ
  B : ℝ × ℝ
  -- The circles are externally tangent
  externally_tangent : dist S L = r_small + r_large
  -- PAB is a common tangent
  tangent_line : dist P A = dist A B
  -- A is on the smaller circle, B is on the larger circle
  on_circles : dist S A = r_small ∧ dist L B = r_large
  -- Length condition
  length_condition : dist P A = 4 ∧ dist A B = 4

/-- The area of the smaller circle in the TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : Real.pi * tc.r_small ^ 2 = 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_smaller_circle_area_l1545_154547


namespace NUMINAMATH_CALUDE_number_plus_ten_l1545_154540

theorem number_plus_ten (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_ten_l1545_154540


namespace NUMINAMATH_CALUDE_problem_solution_l1545_154510

theorem problem_solution : 
  (Real.sqrt 27 + Real.sqrt 2 * Real.sqrt 6 + Real.sqrt 20 - 5 * Real.sqrt (1/5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  ((Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) + (Real.sqrt 3 - 2) = Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1545_154510


namespace NUMINAMATH_CALUDE_cost_per_person_is_correct_l1545_154591

def item1_base_cost : ℝ := 40
def item1_tax_rate : ℝ := 0.05
def item1_discount_rate : ℝ := 0.10

def item2_base_cost : ℝ := 70
def item2_tax_rate : ℝ := 0.08
def item2_coupon : ℝ := 5

def item3_base_cost : ℝ := 100
def item3_tax_rate : ℝ := 0.06
def item3_discount_rate : ℝ := 0.10

def num_people : ℕ := 3

def calculate_item1_cost : ℝ := 
  let cost_after_tax := item1_base_cost * (1 + item1_tax_rate)
  cost_after_tax * (1 - item1_discount_rate)

def calculate_item2_cost : ℝ := 
  item2_base_cost * (1 + item2_tax_rate) - item2_coupon

def calculate_item3_cost : ℝ := 
  let cost_after_tax := item3_base_cost * (1 + item3_tax_rate)
  cost_after_tax * (1 - item3_discount_rate)

def total_cost : ℝ := 
  calculate_item1_cost + calculate_item2_cost + calculate_item3_cost

theorem cost_per_person_is_correct : 
  total_cost / num_people = 67.93 := by sorry

end NUMINAMATH_CALUDE_cost_per_person_is_correct_l1545_154591


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1545_154597

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter (a b c : ℝ) :
  IsoscelesTriangle a b c →
  a + b + c = 30 →
  ((2 * (a + b + c) / 5 = a ∧ (a + b + c) / 5 = b ∧ (a + b + c) / 5 = c) ∨
   (a = 8 ∧ b = 11 ∧ c = 11) ∨
   (a = 8 ∧ b = 8 ∧ c = 14) ∨
   (a = 11 ∧ b = 8 ∧ c = 11) ∨
   (a = 14 ∧ b = 8 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1545_154597


namespace NUMINAMATH_CALUDE_percentage_increase_l1545_154542

theorem percentage_increase (B C : ℝ) (h1 : C = B - 30) : 
  let A := 3 * B
  100 * (A - C) / C = 200 + 9000 / C := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1545_154542


namespace NUMINAMATH_CALUDE_jan_cindy_age_difference_l1545_154514

def age_difference (cindy_age jan_age marcia_age greg_age : ℕ) : Prop :=
  (cindy_age = 5) ∧
  (jan_age > cindy_age) ∧
  (marcia_age = 2 * jan_age) ∧
  (greg_age = marcia_age + 2) ∧
  (greg_age = 16) ∧
  (jan_age - cindy_age = 2)

theorem jan_cindy_age_difference :
  ∃ (cindy_age jan_age marcia_age greg_age : ℕ),
    age_difference cindy_age jan_age marcia_age greg_age := by
  sorry

end NUMINAMATH_CALUDE_jan_cindy_age_difference_l1545_154514


namespace NUMINAMATH_CALUDE_train_speed_conversion_l1545_154509

theorem train_speed_conversion (speed_kmph : ℝ) (speed_ms : ℝ) : 
  speed_kmph = 216 → speed_ms = 60 → speed_kmph * 1000 / 3600 = speed_ms := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l1545_154509


namespace NUMINAMATH_CALUDE_third_runner_time_l1545_154560

/-- A relay race with 4 runners -/
structure RelayRace where
  mary : ℝ
  susan : ℝ
  third : ℝ
  tiffany : ℝ

/-- The conditions of the relay race -/
def validRelayRace (race : RelayRace) : Prop :=
  race.mary = 2 * race.susan ∧
  race.susan = race.third + 10 ∧
  race.tiffany = race.mary - 7 ∧
  race.mary + race.susan + race.third + race.tiffany = 223

/-- The theorem stating that the third runner's time is 30 seconds -/
theorem third_runner_time (race : RelayRace) 
  (h : validRelayRace race) : race.third = 30 := by
  sorry

end NUMINAMATH_CALUDE_third_runner_time_l1545_154560


namespace NUMINAMATH_CALUDE_contractor_absent_days_l1545_154539

/-- Represents the contract details and outcome -/
structure ContractDetails where
  totalDays : ℕ
  paymentPerDay : ℚ
  finePerDay : ℚ
  totalReceived : ℚ

/-- Calculates the number of absent days given the contract details -/
def absentDays (c : ContractDetails) : ℚ :=
  (c.totalDays * c.paymentPerDay - c.totalReceived) / (c.paymentPerDay + c.finePerDay)

/-- Theorem stating that given the specific contract details, the number of absent days is 10 -/
theorem contractor_absent_days :
  let c : ContractDetails := {
    totalDays := 30,
    paymentPerDay := 25,
    finePerDay := 15/2,
    totalReceived := 425
  }
  absentDays c = 10 := by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l1545_154539


namespace NUMINAMATH_CALUDE_unrepaired_road_not_thirty_percent_l1545_154582

/-- Represents the percentage of road repaired in the first phase -/
def first_phase_repair : ℝ := 0.4

/-- Represents the percentage of remaining road repaired in the second phase -/
def second_phase_repair : ℝ := 0.3

/-- Represents the total length of the road in meters -/
def total_road_length : ℝ := 200

/-- Theorem stating that the unrepaired portion of the road is not 30% -/
theorem unrepaired_road_not_thirty_percent :
  let remaining_after_first := 1 - first_phase_repair
  let repaired_in_second := remaining_after_first * second_phase_repair
  let total_repaired := first_phase_repair + repaired_in_second
  total_repaired ≠ 0.7 := by sorry

end NUMINAMATH_CALUDE_unrepaired_road_not_thirty_percent_l1545_154582


namespace NUMINAMATH_CALUDE_least_number_of_cookies_l1545_154588

theorem least_number_of_cookies (n : ℕ) : n ≥ 208 →
  (n % 6 = 4 ∧ n % 5 = 3 ∧ n % 8 = 6 ∧ n % 9 = 7) →
  n = 208 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_cookies_l1545_154588


namespace NUMINAMATH_CALUDE_inequality_proof_l1545_154556

theorem inequality_proof (a b : ℝ) (ha : |a| ≤ Real.sqrt 3) (hb : |b| ≤ Real.sqrt 3) :
  Real.sqrt 3 * |a + b| ≤ |a * b + 3| := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1545_154556


namespace NUMINAMATH_CALUDE_license_plate_combinations_l1545_154599

/-- The number of letters in the English alphabet -/
def alphabet_count : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants (including Y) -/
def consonant_count : ℕ := alphabet_count - vowel_count

/-- The number of digits -/
def digit_count : ℕ := 10

/-- The number of possible four-character license plates -/
def license_plate_count : ℕ := consonant_count * vowel_count * consonant_count * digit_count

theorem license_plate_combinations :
  license_plate_count = 22050 := by sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l1545_154599


namespace NUMINAMATH_CALUDE_cake_slices_left_l1545_154570

def cake_problem (first_cake_slices second_cake_slices : ℕ)
  (first_cake_friend_fraction second_cake_friend_fraction : ℚ)
  (family_fraction : ℚ)
  (first_cake_alex_eats second_cake_alex_eats : ℕ) : Prop :=
  let first_remaining_after_friends := first_cake_slices - (first_cake_slices * first_cake_friend_fraction).floor
  let second_remaining_after_friends := second_cake_slices - (second_cake_slices * second_cake_friend_fraction).floor
  let first_remaining_after_family := first_remaining_after_friends - (first_remaining_after_friends * family_fraction).floor
  let second_remaining_after_family := second_remaining_after_friends - (second_remaining_after_friends * family_fraction).floor
  let first_final := max 0 (first_remaining_after_family - first_cake_alex_eats)
  let second_final := max 0 (second_remaining_after_family - second_cake_alex_eats)
  first_final + second_final = 2

theorem cake_slices_left :
  cake_problem 8 12 (1/4) (1/3) (1/2) 3 2 :=
by sorry

end NUMINAMATH_CALUDE_cake_slices_left_l1545_154570


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1545_154525

theorem linear_equation_solution : 
  ∃ x : ℝ, (2 / 3 : ℝ) * x - 2 = 4 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1545_154525


namespace NUMINAMATH_CALUDE_marbles_taken_correct_l1545_154544

/-- The number of green marbles Mike took from Dan -/
def marbles_taken (initial_green : ℕ) (remaining_green : ℕ) : ℕ :=
  initial_green - remaining_green

/-- Theorem stating that the number of marbles Mike took is the difference between
    Dan's initial and remaining green marbles -/
theorem marbles_taken_correct (initial_green : ℕ) (remaining_green : ℕ) 
    (h : initial_green ≥ remaining_green) :
  marbles_taken initial_green remaining_green = initial_green - remaining_green :=
by
  sorry

#eval marbles_taken 32 9  -- Should output 23

end NUMINAMATH_CALUDE_marbles_taken_correct_l1545_154544


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1545_154565

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1545_154565


namespace NUMINAMATH_CALUDE_derivative_sin_pi_third_l1545_154569

theorem derivative_sin_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) :
  deriv f (π / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_pi_third_l1545_154569


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l1545_154530

/-- Calculates the gain percent given the cost price and selling price ratio -/
theorem chocolate_gain_percent 
  (cost_price selling_price : ℝ) 
  (h : 81 * cost_price = 45 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l1545_154530


namespace NUMINAMATH_CALUDE_max_naive_number_with_divisible_ratio_l1545_154576

def is_naive_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 % 10 = n % 10 + 6) ∧
  (n / 100 % 10 = n / 10 % 10 + 2)

def P (n : ℕ) : ℕ :=
  3 * (n / 1000 % 10 + n / 100 % 10) + n / 10 % 10 + n % 10

def Q (n : ℕ) : ℕ :=
  n / 1000 % 10 - 5

theorem max_naive_number_with_divisible_ratio :
  ∃ (m : ℕ), is_naive_number m ∧ 
             (∀ n, is_naive_number n → P n / Q n % 10 = 0 → n ≤ m) ∧
             (P m / Q m % 10 = 0) ∧
             m = 9313 :=
sorry

end NUMINAMATH_CALUDE_max_naive_number_with_divisible_ratio_l1545_154576


namespace NUMINAMATH_CALUDE_multiply_18396_9999_l1545_154584

theorem multiply_18396_9999 : 18396 * 9999 = 183941604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_18396_9999_l1545_154584


namespace NUMINAMATH_CALUDE_tims_kittens_l1545_154583

theorem tims_kittens (initial_kittens : ℕ) : 
  (initial_kittens > 0) →
  (initial_kittens * 2 / 3 * 3 / 5 = 12) →
  initial_kittens = 30 := by
sorry

end NUMINAMATH_CALUDE_tims_kittens_l1545_154583


namespace NUMINAMATH_CALUDE_compatible_polygons_exist_l1545_154532

/-- A simple polygon is a polygon that does not intersect itself. -/
def SimplePolygon : Type := sorry

/-- Two simple polygons are compatible if there exists a positive integer k such that 
    each polygon can be partitioned into k congruent polygons similar to the other one. -/
def compatible (P Q : SimplePolygon) : Prop := sorry

/-- The number of sides of a simple polygon. -/
def num_sides (P : SimplePolygon) : ℕ := sorry

theorem compatible_polygons_exist (m n : ℕ) (hm : Even m) (hn : Even n) 
  (hm_ge_4 : m ≥ 4) (hn_ge_4 : n ≥ 4) : 
  ∃ (P Q : SimplePolygon), num_sides P = m ∧ num_sides Q = n ∧ compatible P Q := by
  sorry

end NUMINAMATH_CALUDE_compatible_polygons_exist_l1545_154532


namespace NUMINAMATH_CALUDE_white_balls_count_l1545_154504

theorem white_balls_count (total : ℕ) (red : ℕ) (prob : ℚ) : 
  red = 8 → 
  prob = 2/5 → 
  prob = red / total → 
  total - red = 12 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l1545_154504


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_g_l1545_154537

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - 2

-- Define the function g
def g (x : ℝ) : ℝ := |x + 3| - |2*x - 1| - 2

-- Theorem 1: Solution set of f(x) < |x-1|
theorem solution_set_f (x : ℝ) : f x < |x - 1| ↔ x < 0 := by sorry

-- Theorem 2: Maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_g_l1545_154537


namespace NUMINAMATH_CALUDE_fermat_coprime_l1545_154586

/-- The n-th Fermat number -/
def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Fermat numbers are pairwise coprime -/
theorem fermat_coprime : ∀ i j : ℕ, i ≠ j → Nat.gcd (fermat i) (fermat j) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fermat_coprime_l1545_154586


namespace NUMINAMATH_CALUDE_solution_of_system_l1545_154536

/-- Given a system of equations with four distinct real numbers a₁, a₂, a₃, a₄,
    prove that the solution is x₁ = 1 / (a₄ - a₁), x₂ = 0, x₃ = 0, x₄ = 1 / (a₄ - a₁) -/
theorem solution_of_system (a₁ a₂ a₃ a₄ : ℝ) 
    (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ),
    (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
    (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
    (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
    (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1) ∧
    x₁ = 1 / (a₄ - a₁) ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / (a₄ - a₁) := by
  sorry


end NUMINAMATH_CALUDE_solution_of_system_l1545_154536


namespace NUMINAMATH_CALUDE_quadratic_rational_root_parity_l1545_154517

theorem quadratic_rational_root_parity (a b c : ℤ) (h_a : a ≠ 0) :
  (∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) →
  (Even b ∨ Even c) →
  ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_parity_l1545_154517


namespace NUMINAMATH_CALUDE_product_of_integers_l1545_154520

theorem product_of_integers (A B C D : ℕ+) : 
  (A : ℝ) + (B : ℝ) + (C : ℝ) + (D : ℝ) = 40 →
  (A : ℝ) + 3 = (B : ℝ) - 3 ∧ 
  (A : ℝ) + 3 = (C : ℝ) * 3 ∧ 
  (A : ℝ) + 3 = (D : ℝ) / 3 →
  (A : ℝ) * (B : ℝ) * (C : ℝ) * (D : ℝ) = 2666.25 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l1545_154520


namespace NUMINAMATH_CALUDE_existence_of_p_and_q_l1545_154502

theorem existence_of_p_and_q : ∃ (p q : ℝ), 
  ((p - 1)^2 - 4*q > 0) ∧ 
  ((p + 1)^2 - 4*q > 0) ∧ 
  (p^2 - 4*q ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_existence_of_p_and_q_l1545_154502


namespace NUMINAMATH_CALUDE_diaz_future_age_l1545_154549

/-- Given that 40 less than 10 times Diaz's age is 20 more than 10 times Sierra's age,
    and Sierra is currently 30 years old, prove that Diaz will be 56 years old 20 years from now. -/
theorem diaz_future_age :
  ∀ (diaz_age sierra_age : ℕ),
  sierra_age = 30 →
  10 * diaz_age - 40 = 10 * sierra_age + 20 →
  diaz_age + 20 = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_diaz_future_age_l1545_154549


namespace NUMINAMATH_CALUDE_am_gm_strict_inequality_l1545_154580

theorem am_gm_strict_inequality {a b : ℝ} (ha : a > b) (hb : b > 0) :
  Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_am_gm_strict_inequality_l1545_154580


namespace NUMINAMATH_CALUDE_can_capacity_is_eight_litres_l1545_154598

/-- Represents the contents and capacity of a can containing a mixture of milk and water. -/
structure Can where
  initial_milk : ℝ
  initial_water : ℝ
  capacity : ℝ

/-- Proves that the capacity of the can is 8 litres given the specified conditions. -/
theorem can_capacity_is_eight_litres (can : Can)
  (h1 : can.initial_milk / can.initial_water = 1 / 5)
  (h2 : (can.initial_milk + 2) / can.initial_water = 3 / 5)
  (h3 : can.capacity = can.initial_milk + can.initial_water + 2) :
  can.capacity = 8 := by
  sorry

#check can_capacity_is_eight_litres

end NUMINAMATH_CALUDE_can_capacity_is_eight_litres_l1545_154598


namespace NUMINAMATH_CALUDE_calculation_proof_l1545_154506

theorem calculation_proof : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1545_154506


namespace NUMINAMATH_CALUDE_tennis_ball_order_l1545_154579

theorem tennis_ball_order (white yellow : ℕ) : 
  white = yellow →                        -- Initially equal number of white and yellow balls
  (white : ℚ) / (yellow + 70 : ℚ) = 8/13 →  -- Ratio after error
  white + yellow = 224 :=                 -- Total number of balls ordered
by sorry

end NUMINAMATH_CALUDE_tennis_ball_order_l1545_154579


namespace NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l1545_154566

/-- A function is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (fun x ↦ f x + g x)) ∧
  (∃ f g : ℝ → ℝ, ¬(IsEven f ∧ IsEven g) ∧ IsEven (fun x ↦ f x + g x)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l1545_154566


namespace NUMINAMATH_CALUDE_set_membership_implies_x_values_l1545_154572

theorem set_membership_implies_x_values (x : ℝ) :
  let A : Set ℝ := {2, 4, x^2 - x}
  6 ∈ A → x = 3 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_x_values_l1545_154572


namespace NUMINAMATH_CALUDE_max_students_distribution_l1545_154551

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 2010) (h_pencils : pencils = 1050) : 
  (∃ (students : ℕ), students > 0 ∧ 
    pens % students = 0 ∧ 
    pencils % students = 0 ∧ 
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) → 
  (∃ (max_students : ℕ), max_students = Nat.gcd pens pencils) :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l1545_154551


namespace NUMINAMATH_CALUDE_ferry_problem_l1545_154548

/-- Ferry problem -/
theorem ferry_problem (v_p v_q : ℝ) (d_p d_q : ℝ) (t_p t_q : ℝ) :
  v_p = 8 →
  d_q = 3 * d_p →
  v_q = v_p + 1 →
  t_q = t_p + 5 →
  d_p = v_p * t_p →
  d_q = v_q * t_q →
  t_p = 3 := by
  sorry

#check ferry_problem

end NUMINAMATH_CALUDE_ferry_problem_l1545_154548


namespace NUMINAMATH_CALUDE_average_of_last_three_l1545_154568

theorem average_of_last_three (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 62 →
  ((list.take 4).sum / 4 : ℝ) = 55 →
  ((list.drop 4).sum / 3 : ℝ) = 71 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_average_of_last_three_l1545_154568


namespace NUMINAMATH_CALUDE_floor_power_minus_n_even_l1545_154596

theorem floor_power_minus_n_even (n : ℕ+) : 
  ∃ (u : ℝ), u > 0 ∧ ∀ (n : ℕ+), Even (⌊u^(n : ℝ)⌋ - n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_floor_power_minus_n_even_l1545_154596


namespace NUMINAMATH_CALUDE_dot_product_result_l1545_154594

theorem dot_product_result :
  let a : ℝ × ℝ := (2 * Real.sin (35 * π / 180), 2 * Real.cos (35 * π / 180))
  let b : ℝ × ℝ := (Real.cos (5 * π / 180), -Real.sin (5 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_result_l1545_154594


namespace NUMINAMATH_CALUDE_remainder_13_pow_2011_mod_100_l1545_154518

theorem remainder_13_pow_2011_mod_100 : 13^2011 % 100 = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2011_mod_100_l1545_154518


namespace NUMINAMATH_CALUDE_simplify_expression_l1545_154554

theorem simplify_expression (m : ℝ) : m^2 - m*(m-3) = 3*m := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1545_154554


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1545_154516

theorem sum_of_fractions : 
  (1/10 : ℚ) + (2/10 : ℚ) + (3/10 : ℚ) + (4/10 : ℚ) + (5/10 : ℚ) + 
  (6/10 : ℚ) + (7/10 : ℚ) + (8/10 : ℚ) + (9/10 : ℚ) + (90/10 : ℚ) = 
  (27/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1545_154516


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1545_154545

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 1 - a 3 = -3) :
  a 4 = -8 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1545_154545


namespace NUMINAMATH_CALUDE_expression_value_l1545_154561

theorem expression_value : 
  let a : ℝ := 1.69
  let b : ℝ := 1.73
  let c : ℝ := 0.48
  1 / (a^2 - a*c - a*b + b*c) + 2 / (b^2 - a*b - b*c + a*c) + 1 / (c^2 - a*c - b*c + a*b) = 20 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1545_154561


namespace NUMINAMATH_CALUDE_equation_solution_set_l1545_154501

theorem equation_solution_set : 
  {(x, y) : ℕ × ℕ | 3 * x^2 + 2 * 9^y = x * (4^(y + 1) - 1)} = 
  {(2, 1), (3, 1), (3, 2), (18, 2)} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_set_l1545_154501


namespace NUMINAMATH_CALUDE_jerome_trail_time_l1545_154592

/-- The time it takes Jerome to run the trail -/
def jerome_time : ℝ := 6

/-- The time it takes Nero to run the trail -/
def nero_time : ℝ := 3

/-- Jerome's running speed in MPH -/
def jerome_speed : ℝ := 4

/-- Nero's running speed in MPH -/
def nero_speed : ℝ := 8

/-- Theorem stating that Jerome's time to run the trail is 6 hours -/
theorem jerome_trail_time : jerome_time = 6 := by sorry

end NUMINAMATH_CALUDE_jerome_trail_time_l1545_154592


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1545_154515

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  2 * b = a + c →  -- arithmetic sequence
  (c * a) * (b * c) = (a * b) * (a * b) →  -- geometric sequence
  a + b + c = 15 →
  a = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1545_154515


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1545_154546

theorem inequality_solution_set (x : ℝ) : 6 + 5*x - x^2 > 0 ↔ -1 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1545_154546


namespace NUMINAMATH_CALUDE_sum_zero_in_2x2_square_l1545_154571

/-- Given a 2x2 square with numbers a, b, c, d that are pairwise distinct,
    with the sum of numbers in the first row equal to the sum of numbers in the second row,
    and the product of numbers in the first column equal to the product of numbers in the second column,
    prove that the sum of all four numbers is zero. -/
theorem sum_zero_in_2x2_square (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (row_sum : a + b = c + d)
  (col_prod : a * c = b * d) :
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_zero_in_2x2_square_l1545_154571


namespace NUMINAMATH_CALUDE_range_of_a_l1545_154589

-- Define the set of valid values for a
def ValidA : Set ℝ :=
  {x | x > -1 ∧ x ≠ -5/6 ∧ x ≠ (1 + Real.sqrt 21) / 4 ∧ x ≠ (1 - Real.sqrt 21) / 4 ∧ x ≠ -7/8}

-- State the theorem
theorem range_of_a (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (eq1 : b^2 + c^2 = 2*a^2 + 16*a + 14) 
  (eq2 : b*c = a^2 - 4*a - 5) : 
  a ∈ ValidA :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1545_154589


namespace NUMINAMATH_CALUDE_calculate_expression_l1545_154507

theorem calculate_expression : (28 * (9 + 2 - 5)) * 3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1545_154507


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_l1545_154563

theorem polynomial_nonnegative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_l1545_154563


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1545_154522

/-- A parabola y = ax^2 + 8 is tangent to the line y = 2x + 3 if and only if a = 1/5 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 8 = 2 * x + 3) ↔ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1545_154522


namespace NUMINAMATH_CALUDE_program_output_l1545_154581

def program (initial_A initial_B : Int) : (Int × Int × Int) :=
  let A₁ := if initial_A < 0 then -initial_A else initial_A
  let B₁ := initial_B * initial_B
  let A₂ := A₁ + B₁
  let C := A₂ - 2 * B₁
  let A₃ := A₂ / C
  let B₂ := B₁ * C + 1
  (A₃, B₂, C)

theorem program_output : program (-6) 2 = (5, 9, 2) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l1545_154581


namespace NUMINAMATH_CALUDE_largest_difference_l1545_154521

def A : ℕ := 3 * 2010^2011
def B : ℕ := 2010^2011
def C : ℕ := 2009 * 2010^2010
def D : ℕ := 3 * 2010^2010
def E : ℕ := 2010^2010
def F : ℕ := 2010^2009

theorem largest_difference : 
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) := by sorry

end NUMINAMATH_CALUDE_largest_difference_l1545_154521


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1545_154555

/-- A line with slope -3 passing through (3,0) has y-intercept (0,9) -/
theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) :
  m = -3 →
  x₀ = 3 →
  y₀ = 0 →
  ∃ (b : ℝ), ∀ (x y : ℝ), y = m * (x - x₀) + y₀ → y = m * x + b →
  b = 9 ∧ 9 = m * 0 + b :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1545_154555


namespace NUMINAMATH_CALUDE_election_lead_probability_l1545_154553

theorem election_lead_probability (total votes_A votes_B : ℕ) 
  (h_total : total = votes_A + votes_B)
  (h_A_more : votes_A > votes_B) :
  let prob := (votes_A - votes_B) / total
  prob = 1 / 43 :=
by sorry

end NUMINAMATH_CALUDE_election_lead_probability_l1545_154553


namespace NUMINAMATH_CALUDE_derivative_of_cos_squared_l1545_154500

theorem derivative_of_cos_squared (x : ℝ) : 
  deriv (λ x : ℝ => (1 + Real.cos (2 * x))^2) x = -4 * Real.sin (2 * x) - 2 * Real.sin (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_cos_squared_l1545_154500


namespace NUMINAMATH_CALUDE_cone_ratio_l1545_154534

/-- For a cone with a central angle of 120° in its unfolded lateral surface,
    the ratio of its base radius to its slant height is 1/3 -/
theorem cone_ratio (r : ℝ) (l : ℝ) (h : r > 0) (h' : l > 0) :
  2 * Real.pi * r = 2 * Real.pi / 3 * l → r / l = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_ratio_l1545_154534


namespace NUMINAMATH_CALUDE_sara_paycheck_l1545_154511

/-- Sara's paycheck calculation --/
theorem sara_paycheck (weeks : ℕ) (hours_per_week : ℕ) (hourly_rate : ℚ) (tire_cost : ℚ) :
  weeks = 2 →
  hours_per_week = 40 →
  hourly_rate = 11.5 →
  tire_cost = 410 →
  (weeks * hours_per_week : ℚ) * hourly_rate - tire_cost = 510 :=
by sorry

end NUMINAMATH_CALUDE_sara_paycheck_l1545_154511


namespace NUMINAMATH_CALUDE_average_weight_proof_l1545_154590

theorem average_weight_proof (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 33 →
  (a + b) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_average_weight_proof_l1545_154590


namespace NUMINAMATH_CALUDE_abc_inequalities_l1545_154593

theorem abc_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
  ((1 + a) * (1 + b) * (1 + c) ≥ 8) ∧ 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 1/a + 1/b + 1/c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l1545_154593


namespace NUMINAMATH_CALUDE_cubic_difference_even_iff_sum_even_l1545_154585

theorem cubic_difference_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end NUMINAMATH_CALUDE_cubic_difference_even_iff_sum_even_l1545_154585


namespace NUMINAMATH_CALUDE_sqrt_2m_minus_n_equals_sqrt_2_l1545_154523

theorem sqrt_2m_minus_n_equals_sqrt_2 (m n : ℝ) : 
  (2 * m + n = 8 ∧ 2 * n - m = 1) → Real.sqrt (2 * m - n) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2m_minus_n_equals_sqrt_2_l1545_154523


namespace NUMINAMATH_CALUDE_lock_settings_count_l1545_154527

/-- The number of digits on each dial of the lock -/
def numDigits : ℕ := 10

/-- The number of dials on the lock -/
def numDials : ℕ := 4

/-- Calculates the number of different settings for the lock -/
def lockSettings : ℕ := numDigits * (numDigits - 1) * (numDigits - 2) * (numDigits - 3)

/-- Theorem stating that the number of different settings for the lock is 5040 -/
theorem lock_settings_count : lockSettings = 5040 := by
  sorry

end NUMINAMATH_CALUDE_lock_settings_count_l1545_154527


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l1545_154595

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- The ratio of the area of triangle ABC to the area of triangle ADC is 4:1
  area_ratio : ab / cd = 4
  -- The sum of AB and CD is 250
  sum_sides : ab + cd = 250

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC 
to the area of triangle ADC is 4:1, and AB + CD = 250 cm, then AB = 200 cm.
-/
theorem trapezoid_side_length (t : Trapezoid) : t.ab = 200 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l1545_154595


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l1545_154573

/-- Theorem: Number of adult tickets sold in a movie theater --/
theorem adult_tickets_sold (adult_price child_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100) : 
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧ 
    adult_tickets = 500 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l1545_154573


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l1545_154559

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l1545_154559


namespace NUMINAMATH_CALUDE_no_intersection_l1545_154505

-- Define the two functions
def f (x : ℝ) : ℝ := |3*x + 6|
def g (x : ℝ) : ℝ := -|4*x - 3|

-- Theorem statement
theorem no_intersection :
  ¬ ∃ (x y : ℝ), f x = y ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l1545_154505


namespace NUMINAMATH_CALUDE_steve_height_l1545_154578

/-- Converts feet and inches to total inches -/
def feet_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Calculates final height after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  feet_to_inches initial_feet initial_inches + growth

theorem steve_height :
  final_height 5 6 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_steve_height_l1545_154578


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1545_154512

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1545_154512


namespace NUMINAMATH_CALUDE_total_cupcakes_is_52_l1545_154524

/-- Represents the number of cupcakes ordered by the mum -/
def total_cupcakes : ℕ := 52

/-- Represents the number of vegan cupcakes -/
def vegan_cupcakes : ℕ := 24

/-- Represents the number of non-vegan cupcakes containing gluten -/
def non_vegan_gluten_cupcakes : ℕ := 28

/-- States that half of all cupcakes are gluten-free -/
axiom half_gluten_free : total_cupcakes / 2 = total_cupcakes - (vegan_cupcakes / 2 + non_vegan_gluten_cupcakes)

/-- States that half of vegan cupcakes are gluten-free -/
axiom half_vegan_gluten_free : vegan_cupcakes / 2 = total_cupcakes / 2 - non_vegan_gluten_cupcakes

/-- Theorem: The total number of cupcakes is 52 -/
theorem total_cupcakes_is_52 : total_cupcakes = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_cupcakes_is_52_l1545_154524


namespace NUMINAMATH_CALUDE_expected_total_rainfall_l1545_154529

/-- Represents the weather conditions for each day -/
structure DailyWeather where
  sun_prob : Real
  rain_3in_prob : Real
  rain_8in_prob : Real

/-- Calculates the expected rainfall for a single day -/
def expected_daily_rainfall (w : DailyWeather) : Real :=
  w.sun_prob * 0 + w.rain_3in_prob * 3 + w.rain_8in_prob * 8

/-- The weather forecast for the week -/
def weather_forecast : DailyWeather :=
  { sun_prob := 0.3
  , rain_3in_prob := 0.4
  , rain_8in_prob := 0.3 }

/-- The number of days in the forecast -/
def num_days : Nat := 5

/-- Theorem: The expected total rainfall for the week is 18 inches -/
theorem expected_total_rainfall :
  (expected_daily_rainfall weather_forecast) * num_days = 18 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rainfall_l1545_154529


namespace NUMINAMATH_CALUDE_notebook_cost_l1545_154513

theorem notebook_cost (notebook_cost pen_cost : ℝ) 
  (total_cost : notebook_cost + pen_cost = 4.50)
  (cost_difference : notebook_cost = pen_cost + 3) : 
  notebook_cost = 3.75 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l1545_154513


namespace NUMINAMATH_CALUDE_k_lower_bound_l1545_154552

open Real

theorem k_lower_bound (k : ℝ) (hk : k > 0) :
  (∃ x₀ : ℝ, x₀ ≥ 1 ∧ exp x₀ + exp (-x₀) ≤ k * (-x₀^2 + 3*x₀)) →
  k > (exp 1 + exp (-1)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_k_lower_bound_l1545_154552


namespace NUMINAMATH_CALUDE_taehyung_average_problems_l1545_154562

/-- The average number of problems solved per day -/
def average_problems_per_day (total_problems : ℕ) (num_days : ℕ) : ℚ :=
  (total_problems : ℚ) / (num_days : ℚ)

/-- Theorem stating that the average number of problems solved per day is 23 -/
theorem taehyung_average_problems :
  average_problems_per_day 161 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_average_problems_l1545_154562


namespace NUMINAMATH_CALUDE_dolls_distribution_l1545_154558

theorem dolls_distribution (total_dolls : ℕ) (defective_dolls : ℕ) (num_stores : ℕ) : 
  total_dolls = 40 → defective_dolls = 4 → num_stores = 4 →
  (total_dolls - defective_dolls) / num_stores = 9 := by
  sorry

end NUMINAMATH_CALUDE_dolls_distribution_l1545_154558


namespace NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_rate_l1545_154575

/-- Calculates the interest rate of an additional investment given initial investment conditions and total annual income. -/
theorem additional_investment_rate (initial_investment : ℝ) (initial_rate : ℝ) (total_rate : ℝ) (total_income : ℝ) : ℝ :=
  let additional_investment := (total_income - initial_investment * total_rate) / (total_rate - initial_rate)
  let additional_income := total_income - initial_investment * initial_rate
  additional_income / additional_investment

/-- Proves that the interest rate of the additional investment is approximately 6.13% given the specified conditions. -/
theorem barbata_investment_rate : 
  let initial_investment : ℝ := 2200
  let initial_rate : ℝ := 0.05
  let total_rate : ℝ := 0.06
  let total_income : ℝ := 1099.9999999999998
  abs (additional_investment_rate initial_investment initial_rate total_rate total_income - 0.0613) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_additional_investment_rate_barbata_investment_rate_l1545_154575


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l1545_154574

theorem subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) 
  (S : Finset ℕ) (hS : S.card = n) (hS_subset : ∀ x ∈ S, x ∈ Finset.range (2*n)) :
  ∃ T : Finset ℕ, T ⊆ S ∧ (2*n) ∣ (T.sum id) :=
sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l1545_154574


namespace NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l1545_154564

theorem angle_with_supplement_four_times_complement (x : ℝ) :
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l1545_154564


namespace NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l1545_154526

/-- Given a 2x2 matrix B with real entries such that B^4 = 0, prove that B^3 = 0 -/
theorem nilpotent_matrix_cube_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l1545_154526


namespace NUMINAMATH_CALUDE_unit_rectangle_coverage_l1545_154587

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle parallel to the axes -/
structure Rectangle where
  left : ℝ
  bottom : ℝ
  width : ℝ
  height : ℝ

/-- The theorem stating that 1821 points can be arranged to cover all unit-area rectangles in a 15x15 square -/
theorem unit_rectangle_coverage : ∃ (points : Finset Point),
  (points.card = 1821) ∧ 
  (∀ p : Point, p ∈ points → p.x ≥ 0 ∧ p.x ≤ 15 ∧ p.y ≥ 0 ∧ p.y ≤ 15) ∧
  (∀ r : Rectangle, 
    r.left ≥ 0 ∧ r.left + r.width ≤ 15 ∧ 
    r.bottom ≥ 0 ∧ r.bottom + r.height ≤ 15 ∧
    r.width * r.height = 1 →
    ∃ p : Point, p ∈ points ∧ 
      p.x ≥ r.left ∧ p.x ≤ r.left + r.width ∧
      p.y ≥ r.bottom ∧ p.y ≤ r.bottom + r.height) := by
  sorry


end NUMINAMATH_CALUDE_unit_rectangle_coverage_l1545_154587


namespace NUMINAMATH_CALUDE_product_of_hash_operations_l1545_154503

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + a / b

-- Theorem statement
theorem product_of_hash_operations :
  let x := hash 8 3
  let y := hash 5 4
  x * y = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_product_of_hash_operations_l1545_154503


namespace NUMINAMATH_CALUDE_license_plate_count_l1545_154550

def license_plate_options : ℕ :=
  let first_char_options := 5  -- 3, 5, 6, 8, 9
  let second_char_options := 3 -- B, C, D
  let other_char_options := 4  -- 1, 3, 6, 9
  first_char_options * second_char_options * other_char_options * other_char_options * other_char_options

theorem license_plate_count : license_plate_options = 960 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1545_154550


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1545_154533

/-- The probability of shooter A hitting the target -/
def P_A : ℝ := 0.9

/-- The probability of shooter B hitting the target -/
def P_B : ℝ := 0.8

/-- The probability of both A and B hitting the target -/
def P_both : ℝ := P_A * P_B

/-- The probability of at least one of A and B hitting the target -/
def P_at_least_one : ℝ := 1 - (1 - P_A) * (1 - P_B)

theorem shooting_probabilities :
  P_both = 0.72 ∧ P_at_least_one = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1545_154533


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l1545_154531

/-- Two lines in the form y = mx + b are parallel if and only if they have the same slope m -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} : 
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) → m1 = m2

/-- Given that the line ax + 2y + 1 = 0 is parallel to x + y - 2 = 0, prove that a = 2 -/
theorem parallel_lines_a_equals_two (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 1 = 0 ↔ x + y - 2 = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l1545_154531


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1545_154528

theorem inequality_equivalence (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ 7 / 3 < x ∧ x < 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1545_154528


namespace NUMINAMATH_CALUDE_total_money_is_8915_88_l1545_154577

-- Define the amounts of money for Sam, Billy, and Lila
def sam_money : ℝ := 750.50
def billy_money : ℝ := 4.5 * sam_money - 345.25
def lila_money : ℝ := 2.25 * (billy_money - sam_money)

-- Define the total amount of money
def total_money : ℝ := sam_money + billy_money + lila_money

-- Theorem to prove
theorem total_money_is_8915_88 : 
  ∀ (sam billy lila : ℝ),
  sam = sam_money →
  billy = billy_money →
  lila = lila_money →
  sam + billy + lila = 8915.88 :=
by sorry

end NUMINAMATH_CALUDE_total_money_is_8915_88_l1545_154577


namespace NUMINAMATH_CALUDE_playground_children_count_l1545_154508

theorem playground_children_count :
  let boys : ℕ := 27
  let girls : ℕ := 35
  boys + girls = 62 := by sorry

end NUMINAMATH_CALUDE_playground_children_count_l1545_154508


namespace NUMINAMATH_CALUDE_rhombus_area_l1545_154538

/-- A rhombus with side length √113 and diagonals differing by 10 units has area (√201)² - 5√201 -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 113 →
  d₂ = d₁ + 10 →
  d₁ * d₂ = 4 * s^2 →
  (1/2) * d₁ * d₂ = (Real.sqrt 201)^2 - 5 * Real.sqrt 201 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1545_154538


namespace NUMINAMATH_CALUDE_sophia_bus_time_l1545_154541

def sophia_schedule : Prop :=
  let leave_home : Nat := 8 * 60 + 15  -- 8:15 AM in minutes
  let catch_bus : Nat := 8 * 60 + 45   -- 8:45 AM in minutes
  let class_duration : Nat := 55
  let num_classes : Nat := 5
  let lunch_break : Nat := 45
  let club_activities : Nat := 3 * 60  -- 3 hours in minutes
  let arrive_home : Nat := 17 * 60 + 30  -- 5:30 PM in minutes

  let total_away_time : Nat := arrive_home - leave_home
  let school_activities_time : Nat := num_classes * class_duration + lunch_break + club_activities
  let bus_time : Nat := total_away_time - school_activities_time

  bus_time = 25

theorem sophia_bus_time : sophia_schedule := by
  sorry

end NUMINAMATH_CALUDE_sophia_bus_time_l1545_154541


namespace NUMINAMATH_CALUDE_smallest_integer_980_divisors_l1545_154557

theorem smallest_integer_980_divisors (n m k : ℕ) : 
  (∀ i < n, (Nat.divisors i).card ≠ 980) →
  (Nat.divisors n).card = 980 →
  n = m * 4^k →
  ¬(4 ∣ m) →
  (∀ j, j < n → (Nat.divisors j).card = 980 → j = n) →
  m + k = 649 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_980_divisors_l1545_154557


namespace NUMINAMATH_CALUDE_punch_mixture_l1545_154519

theorem punch_mixture (total_volume : ℕ) (lemonade_parts : ℕ) (extra_cranberry_parts : ℕ) :
  total_volume = 72 →
  lemonade_parts = 3 →
  extra_cranberry_parts = 18 →
  lemonade_parts + (lemonade_parts + extra_cranberry_parts) = total_volume →
  lemonade_parts + extra_cranberry_parts = 21 := by
  sorry

#check punch_mixture

end NUMINAMATH_CALUDE_punch_mixture_l1545_154519


namespace NUMINAMATH_CALUDE_kats_required_score_l1545_154567

/-- Given Kat's first two test scores and desired average, calculate the required score on the third test --/
theorem kats_required_score (score1 score2 desired_avg : ℚ) (h1 : score1 = 95/100) (h2 : score2 = 80/100) (h3 : desired_avg = 90/100) :
  ∃ score3 : ℚ, (score1 + score2 + score3) / 3 ≥ desired_avg ∧ score3 = 95/100 :=
by sorry

end NUMINAMATH_CALUDE_kats_required_score_l1545_154567


namespace NUMINAMATH_CALUDE_min_value_of_f_l1545_154535

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / (x - 1)

theorem min_value_of_f :
  ∃ (x_min : ℝ), x_min > 1 ∧
  (∀ (x : ℝ), x > 1 → f x ≥ f x_min) ∧
  f x_min = 2 * Real.sqrt 3 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1545_154535


namespace NUMINAMATH_CALUDE_white_balls_count_l1545_154543

/-- Represents a box of balls with white and black colors. -/
structure BallBox where
  total : ℕ
  white : ℕ
  black : ℕ
  sum_correct : white + black = total
  white_condition : ∀ (n : ℕ), n ≥ 12 → n.choose white > 0
  black_condition : ∀ (n : ℕ), n ≥ 20 → n.choose black > 0

/-- Theorem stating that a box with 30 balls satisfying the given conditions has 19 white balls. -/
theorem white_balls_count (box : BallBox) (h_total : box.total = 30) : box.white = 19 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l1545_154543
