import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l4003_400308

theorem right_triangle_consecutive_legs (a : ℕ) :
  let b := a + 1
  let c := Real.sqrt (a^2 + b^2)
  c^2 = 2*a^2 + 2*a + 1 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_legs_l4003_400308


namespace NUMINAMATH_CALUDE_sheila_work_hours_l4003_400359

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hourly_rate : ℕ
  weekly_earnings : ℕ
  tue_thu_hours : ℕ
  mon_wed_fri_hours : ℕ

/-- Theorem stating that given Sheila's work conditions, she works 24 hours on Mon, Wed, Fri -/
theorem sheila_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.hourly_rate = 7)
  (h2 : schedule.weekly_earnings = 252)
  (h3 : schedule.tue_thu_hours = 6 * 2)
  (h4 : schedule.weekly_earnings = 
        schedule.hourly_rate * (schedule.tue_thu_hours + schedule.mon_wed_fri_hours)) :
  schedule.mon_wed_fri_hours = 24 := by
  sorry


end NUMINAMATH_CALUDE_sheila_work_hours_l4003_400359


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l4003_400397

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 20*x^2 + 400) * (x^2 - 20) = x^6 - 8000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l4003_400397


namespace NUMINAMATH_CALUDE_inscribed_tangent_circle_exists_l4003_400323

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a circle is inscribed in an angle -/
def isInscribed (c : Circle) (a : Angle) : Prop := sorry

/-- Predicate to check if two circles are tangent -/
def isTangent (c1 c2 : Circle) : Prop := sorry

/-- Theorem stating that given an angle and a circle, there exists an inscribed circle tangent to the given circle -/
theorem inscribed_tangent_circle_exists (a : Angle) (c : Circle) :
  ∃ (inscribed_circle : Circle), isInscribed inscribed_circle a ∧ isTangent inscribed_circle c := by
  sorry

end NUMINAMATH_CALUDE_inscribed_tangent_circle_exists_l4003_400323


namespace NUMINAMATH_CALUDE_number_problem_l4003_400346

theorem number_problem (x : ℝ) : (x / 4 + 15 = 27) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4003_400346


namespace NUMINAMATH_CALUDE_gcd_442872_312750_l4003_400350

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_442872_312750_l4003_400350


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4003_400341

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > 1) ↔ (∃ x : ℕ, x^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4003_400341


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l4003_400378

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 20) → cows = 10 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l4003_400378


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4003_400381

theorem sqrt_equation_solution (s : ℝ) : 
  Real.sqrt (3 * Real.sqrt (s - 3)) = (8 - s) ^ (1/4) → s = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4003_400381


namespace NUMINAMATH_CALUDE_derivative_cos_at_pi_12_l4003_400386

/-- Given a function f(x) = cos(2x + π/3), prove that its derivative at x = π/12 is -2. -/
theorem derivative_cos_at_pi_12 (f : ℝ → ℝ) (h : ∀ x, f x = Real.cos (2 * x + π / 3)) :
  deriv f (π / 12) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_at_pi_12_l4003_400386


namespace NUMINAMATH_CALUDE_job_completion_time_l4003_400319

/-- The number of days it takes A to complete the job alone -/
def days_for_A : ℚ := 12

/-- The rate at which A completes the job -/
def rate_A : ℚ := 1 / days_for_A

/-- The rate at which B completes the job -/
def rate_B : ℚ := 1 / 3 - rate_A

/-- The rate at which C completes the job -/
def rate_C : ℚ := 1 / 3 - rate_A

theorem job_completion_time :
  (rate_A + rate_B = 1 / 3) ∧
  (rate_B + rate_C = 1 / 2) ∧
  (rate_A + rate_C = 1 / 3) →
  days_for_A = 12 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l4003_400319


namespace NUMINAMATH_CALUDE_solution_set_l4003_400322

def system_solution (x : ℝ) : Prop :=
  x / 3 ≥ -1 ∧ 3 * x + 4 < 1

theorem solution_set : ∀ x : ℝ, system_solution x ↔ -3 ≤ x ∧ x < -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_l4003_400322


namespace NUMINAMATH_CALUDE_composite_difference_22_l4003_400377

def sequence_a : ℕ → ℕ
  | 0 => sorry  -- Initial value, not specified in the problem
  | 1 => sorry  -- Initial value, not specified in the problem
  | (n + 2) => sequence_a (n + 1) * sequence_a n + 1

theorem composite_difference_22 (n : ℕ) (h : n > 10) :
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ sequence_a n - 22 = k * m := by
  sorry

end NUMINAMATH_CALUDE_composite_difference_22_l4003_400377


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4003_400301

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 11 + 7 * Complex.I → z = 3 + 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4003_400301


namespace NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l4003_400353

/-- The sum of the tens digit and the ones digit of (3+4)^17 in integer form is 7 -/
theorem sum_of_digits_3_plus_4_pow_17 : 
  let n : ℕ := (3 + 4)^17
  let tens_digit : ℕ := (n / 10) % 10
  let ones_digit : ℕ := n % 10
  tens_digit + ones_digit = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l4003_400353


namespace NUMINAMATH_CALUDE_sum_of_possible_values_l4003_400383

theorem sum_of_possible_values (e f : ℚ) : 
  (2 * |2 - e| = 5 ∧ |3 * e + f| = 7) → 
  (∃ e₁ f₁ e₂ f₂ : ℚ, 
    (2 * |2 - e₁| = 5 ∧ |3 * e₁ + f₁| = 7) ∧
    (2 * |2 - e₂| = 5 ∧ |3 * e₂ + f₂| = 7) ∧
    e₁ + f₁ + e₂ + f₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_values_l4003_400383


namespace NUMINAMATH_CALUDE_least_cans_for_given_volumes_l4003_400334

/-- The least number of cans required to pack drinks -/
def leastCans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd := Nat.gcd (Nat.gcd maaza pepsi) sprite
  (maaza / gcd) + (pepsi / gcd) + (sprite / gcd)

/-- Theorem stating the least number of cans required for given volumes -/
theorem least_cans_for_given_volumes :
  leastCans 40 144 368 = 69 := by
  sorry

end NUMINAMATH_CALUDE_least_cans_for_given_volumes_l4003_400334


namespace NUMINAMATH_CALUDE_triangle_properties_l4003_400385

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = t.a * Real.cos t.C + (Real.sqrt 3 / 3) * t.a * Real.sin t.C)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b * t.c = 6) : 
  t.A = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4003_400385


namespace NUMINAMATH_CALUDE_blithe_toy_count_l4003_400339

/-- The number of toys Blithe has after losing some and finding some -/
def finalToyCount (initial lost found : ℕ) : ℕ :=
  initial - lost + found

/-- Theorem: Given Blithe's initial toy count, the number of toys lost, and the number of toys found,
    the final toy count is equal to the initial count minus the lost toys plus the found toys -/
theorem blithe_toy_count : finalToyCount 40 6 9 = 43 := by
  sorry

end NUMINAMATH_CALUDE_blithe_toy_count_l4003_400339


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l4003_400343

/-- Given a geometric sequence with common ratio 3, prove that a_3 = 3 if S_3 + S_4 = 53/3 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 3 * a n) →  -- Geometric sequence with common ratio 3
  (∀ n, S n = (a 1 * (3^n - 1)) / 2) →  -- Sum formula for geometric sequence
  S 3 + S 4 = 53 / 3 →  -- Given condition
  a 3 = 3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_a3_l4003_400343


namespace NUMINAMATH_CALUDE_keaton_yearly_earnings_l4003_400337

/-- Represents Keaton's farm earnings -/
def farm_earnings (orange_harvest_interval : ℕ) (orange_price : ℕ) (apple_harvest_interval : ℕ) (apple_price : ℕ) : ℕ :=
  let months_in_year := 12
  let orange_harvests := months_in_year / orange_harvest_interval
  let apple_harvests := months_in_year / apple_harvest_interval
  orange_harvests * orange_price + apple_harvests * apple_price

/-- Keaton's yearly earnings from his farm of oranges and apples -/
theorem keaton_yearly_earnings :
  farm_earnings 2 50 3 30 = 420 :=
by sorry

end NUMINAMATH_CALUDE_keaton_yearly_earnings_l4003_400337


namespace NUMINAMATH_CALUDE_cos_sum_equals_one_l4003_400352

theorem cos_sum_equals_one (x : ℝ) (h : Real.cos (x - Real.pi / 6) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_equals_one_l4003_400352


namespace NUMINAMATH_CALUDE_smallest_class_size_l4003_400366

/-- Represents the number of students in a physical education class. -/
def class_size (x : ℕ) : ℕ := 5 * x + 3

/-- Theorem stating the smallest possible class size satisfying the given conditions. -/
theorem smallest_class_size :
  ∀ n : ℕ, class_size n > 50 → class_size 10 ≤ class_size n :=
by
  sorry

#eval class_size 10  -- Should output 53

end NUMINAMATH_CALUDE_smallest_class_size_l4003_400366


namespace NUMINAMATH_CALUDE_continuous_n_times_iff_odd_l4003_400304

/-- A function that takes every real value exactly n times. -/
def ExactlyNTimes (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ∧ (∃ (S : Finset ℝ), S.card = n ∧ ∀ x : ℝ, f x = y ↔ x ∈ S)

/-- Main theorem: A continuous function that takes every real value exactly n times exists if and only if n is odd. -/
theorem continuous_n_times_iff_odd (n : ℕ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ExactlyNTimes f n) ↔ Odd n :=
sorry


end NUMINAMATH_CALUDE_continuous_n_times_iff_odd_l4003_400304


namespace NUMINAMATH_CALUDE_table_tennis_play_time_l4003_400318

/-- Represents the table tennis playing scenario -/
structure TableTennis where
  total_students : ℕ
  playing_students : ℕ
  total_time : ℕ
  num_tables : ℕ
  play_time_per_student : ℕ

/-- The theorem statement -/
theorem table_tennis_play_time 
  (tt : TableTennis) 
  (h1 : tt.total_students = 6)
  (h2 : tt.playing_students = 4)
  (h3 : tt.total_time = 210)
  (h4 : tt.num_tables = 2)
  (h5 : tt.total_students % tt.playing_students = 0)
  (h6 : tt.play_time_per_student * tt.total_students = tt.total_time * tt.num_tables) :
  tt.play_time_per_student = 140 := by
  sorry


end NUMINAMATH_CALUDE_table_tennis_play_time_l4003_400318


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l4003_400364

/-- Given a line L₁: Ax + By + C = 0 and a point P₀(x₀, y₀), 
    the line L₂ passing through P₀ and perpendicular to L₁ 
    has the equation Bx - Ay - Bx₀ + Ay₀ = 0 -/
theorem perpendicular_line_equation (A B C x₀ y₀ : ℝ) :
  let L₁ := fun (x y : ℝ) ↦ A * x + B * y + C = 0
  let P₀ := (x₀, y₀)
  let L₂ := fun (x y : ℝ) ↦ B * x - A * y - B * x₀ + A * y₀ = 0
  (∀ x y, L₂ x y ↔ (x - x₀) * B = (y - y₀) * A) ∧
  (∀ x₁ y₁ x₂ y₂, L₁ x₁ y₁ ∧ L₁ x₂ y₂ → (x₂ - x₁) * B = -(y₂ - y₁) * A) ∧
  L₂ x₀ y₀ :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l4003_400364


namespace NUMINAMATH_CALUDE_stamp_collection_theorem_l4003_400357

/-- The face value of Xiaoming's stamps in jiao -/
def xiaoming_stamp_value : ℕ := 16

/-- The face value of Xiaoliang's stamps in jiao -/
def xiaoliang_stamp_value : ℕ := 2

/-- The number of stamps Xiaoming exchanges -/
def xiaoming_exchange_count : ℕ := 2

/-- The ratio of Xiaoliang's stamps to Xiaoming's before exchange -/
def pre_exchange_ratio : ℕ := 5

/-- The ratio of Xiaoliang's stamps to Xiaoming's after exchange -/
def post_exchange_ratio : ℕ := 3

/-- The total number of stamps Xiaoming and Xiaoliang have -/
def total_stamps : ℕ := 168

theorem stamp_collection_theorem :
  let xiaoming_initial := xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value
  let xiaoming_final := xiaoming_initial + xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value - xiaoming_exchange_count
  let xiaoliang_initial := pre_exchange_ratio * xiaoming_initial
  let xiaoliang_final := xiaoliang_initial - xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value + xiaoming_exchange_count
  (xiaoliang_final = post_exchange_ratio * xiaoming_final) →
  (xiaoming_initial + xiaoliang_initial = total_stamps) := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_theorem_l4003_400357


namespace NUMINAMATH_CALUDE_identical_digit_divisibility_l4003_400303

/-- A natural number with identical digits that is divisible by a given divisor -/
structure IdenticalDigitNumber (divisor : ℕ) where
  /-- The number of digits -/
  num_digits : ℕ
  /-- The digit used (1-9) -/
  digit : Fin 9
  /-- The number is divisible by the given divisor -/
  divisible : (digit.val + 1) * (10^num_digits - 1) / 9 % divisor = 0

/-- The set of possible number of digits for numbers with identical digits divisible by 7 -/
def divisible_by_7 : Set ℕ :=
  {n : ℕ | ∃ k : ℕ, n = 6 * k}

/-- The set of possible number of digits for numbers with identical digits divisible by 43 -/
def divisible_by_43 : Set ℕ :=
  {n : ℕ | ∃ m : ℕ, n = 21 * m}

/-- The set of possible number of digits for numbers with identical digits divisible by 41 -/
def divisible_by_41 : Set ℕ :=
  {n : ℕ | ∃ p : ℕ, n = 5 * p}

/-- The set of possible number of digits for numbers with identical digits divisible by 301 -/
def divisible_by_301 : Set ℕ :=
  {n : ℕ | ∃ r : ℕ, n = 42 * r}

/-- The set of possible number of digits for numbers with identical digits divisible by 3 -/
def divisible_by_3 : Set ℕ :=
  {n : ℕ | ∃ s : ℕ, n = 3 * s}

/-- The set of possible number of digits for numbers with identical digits divisible by 21 -/
def divisible_by_21 : Set ℕ :=
  {n : ℕ | (∃ s : ℕ, n = 3 * s) ∨ (∃ k : ℕ, n = 6 * k)}

theorem identical_digit_divisibility :
  ∀ (d : ℕ) (n : IdenticalDigitNumber d),
    d = 7 → n.num_digits ∈ divisible_by_7 ∧
    d = 43 → n.num_digits ∈ divisible_by_43 ∧
    d = 41 → n.num_digits ∈ divisible_by_41 ∧
    d = 301 → n.num_digits ∈ divisible_by_301 ∧
    d = 3 → n.num_digits ∈ divisible_by_3 ∧
    d = 21 → n.num_digits ∈ divisible_by_21 :=
by sorry

#check identical_digit_divisibility

end NUMINAMATH_CALUDE_identical_digit_divisibility_l4003_400303


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l4003_400362

-- Define the package details
def package_A_price : ℝ := 10
def package_A_months : ℕ := 6
def package_A_discount : ℝ := 0.10

def package_B_price : ℝ := 12
def package_B_months : ℕ := 9
def package_B_discount : ℝ := 0.15

-- Define the tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the function to calculate the total cost
def total_cost (package_A_price package_A_months package_A_discount
                package_B_price package_B_months package_B_discount
                sales_tax_rate : ℝ) : ℝ :=
  let package_A_total := package_A_price * package_A_months
  let package_B_total := package_B_price * package_B_months
  let package_A_discounted := package_A_total * (1 - package_A_discount)
  let package_B_discounted := package_B_total * (1 - package_B_discount)
  let package_A_tax := package_A_total * sales_tax_rate
  let package_B_tax := package_B_total * sales_tax_rate
  package_A_discounted + package_A_tax + package_B_discounted + package_B_tax

-- Theorem statement
theorem total_cost_is_correct :
  total_cost package_A_price package_A_months package_A_discount
             package_B_price package_B_months package_B_discount
             sales_tax_rate = 159.24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l4003_400362


namespace NUMINAMATH_CALUDE_pat_has_42_cookies_l4003_400374

-- Define the given conditions
def candy : ℕ := 63
def brownies : ℕ := 21
def family_members : ℕ := 7
def dessert_per_person : ℕ := 18

-- Define the total dessert needed
def total_dessert : ℕ := family_members * dessert_per_person

-- Define the number of cookies
def cookies : ℕ := total_dessert - (candy + brownies)

-- Theorem to prove
theorem pat_has_42_cookies : cookies = 42 := by
  sorry

end NUMINAMATH_CALUDE_pat_has_42_cookies_l4003_400374


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4003_400390

/-- Arithmetic sequence with first term 4 and common difference 2 -/
def a (n : ℕ) : ℕ := 4 + 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℕ := n * (2 * 4 + (n - 1) * 2) / 2

/-- The proposition to be proved -/
theorem arithmetic_sequence_problem :
  ∃ (k : ℕ), k > 0 ∧ S k - a (k + 5) = 44 ∧ k = 7 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4003_400390


namespace NUMINAMATH_CALUDE_wendy_distance_difference_l4003_400324

theorem wendy_distance_difference (ran walked : ℝ) 
  (h1 : ran = 19.83) (h2 : walked = 9.17) : 
  ran - walked = 10.66 := by sorry

end NUMINAMATH_CALUDE_wendy_distance_difference_l4003_400324


namespace NUMINAMATH_CALUDE_solve_system_l4003_400333

theorem solve_system (x y : ℤ) 
  (h1 : x + y = 270) 
  (h2 : x - y = 200) : 
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l4003_400333


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l4003_400329

/-- The set of points equidistant from a fixed point and a line forms a parabola -/
theorem trajectory_is_parabola (x y : ℝ) : 
  (∃ (C : ℝ × ℝ), C.1 = x ∧ C.2 = y ∧ 
    (C.1^2 + (C.2 - 3)^2)^(1/2) = |C.2 + 3|) →
  ∃ (a : ℝ), y = (1 / (4 * a)) * x^2 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l4003_400329


namespace NUMINAMATH_CALUDE_machine_selling_price_l4003_400361

/-- Calculates the selling price of a machine given its costs and profit percentage -/
def calculate_selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 28500 Rs -/
theorem machine_selling_price :
  calculate_selling_price 13000 5000 1000 50 = 28500 := by
  sorry

#eval calculate_selling_price 13000 5000 1000 50

end NUMINAMATH_CALUDE_machine_selling_price_l4003_400361


namespace NUMINAMATH_CALUDE_magnitude_of_c_l4003_400348

/-- Given vectors a and b, if there exists a vector c satisfying certain conditions, then the magnitude of c is 2√5. -/
theorem magnitude_of_c (a b c : ℝ × ℝ) : 
  a = (-1, 2) →
  b = (3, -6) →
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -1/2 →
  c.1 * (-1) + c.2 * 8 = 5 →
  Real.sqrt (c.1^2 + c.2^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_c_l4003_400348


namespace NUMINAMATH_CALUDE_convex_separation_equivalence_l4003_400382

-- Define the type for a compact convex set in ℝ²
def CompactConvexSet : Type := Set (Real × Real)

-- Define the property of a set being compact and convex
def is_compact_convex (S : CompactConvexSet) : Prop := sorry

-- Define the convex hull operation
def conv_hull (S T : CompactConvexSet) : CompactConvexSet := sorry

-- Define the property of a line separating two sets
def separates (L : Set (Real × Real)) (S T : CompactConvexSet) : Prop := sorry

-- Define the property of a line intersecting a set
def intersects (L : Set (Real × Real)) (S : CompactConvexSet) : Prop := sorry

-- The main theorem
theorem convex_separation_equivalence 
  (A B C : CompactConvexSet) 
  (hA : is_compact_convex A) 
  (hB : is_compact_convex B) 
  (hC : is_compact_convex C) : 
  (∀ L : Set (Real × Real), ¬(intersects L A ∧ intersects L B ∧ intersects L C)) ↔ 
  (∃ LA LB LC : Set (Real × Real), 
    separates LA A (conv_hull B C) ∧ 
    separates LB B (conv_hull A C) ∧ 
    separates LC C (conv_hull A B)) := by
  sorry

end NUMINAMATH_CALUDE_convex_separation_equivalence_l4003_400382


namespace NUMINAMATH_CALUDE_teaspoons_per_tablespoon_l4003_400306

/-- Given the following definitions:
  * One cup contains 480 grains of rice
  * Half a cup is 8 tablespoons
  * One teaspoon contains 10 grains of rice
  Prove that there are 3 teaspoons in one tablespoon -/
theorem teaspoons_per_tablespoon 
  (grains_per_cup : ℕ) 
  (tablespoons_per_half_cup : ℕ) 
  (grains_per_teaspoon : ℕ) 
  (h1 : grains_per_cup = 480)
  (h2 : tablespoons_per_half_cup = 8)
  (h3 : grains_per_teaspoon = 10) : 
  (grains_per_cup / 2) / tablespoons_per_half_cup / grains_per_teaspoon = 3 :=
by sorry

end NUMINAMATH_CALUDE_teaspoons_per_tablespoon_l4003_400306


namespace NUMINAMATH_CALUDE_fourth_number_l4003_400332

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → n ≤ 10 → a n = a (n - 1) + a (n - 2)

theorem fourth_number (a : ℕ → ℕ) (h : sequence_property a) 
  (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_l4003_400332


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4003_400351

/-- Given r and s are solutions to the equation 3x^2 - 5x + 2 = 0,
    prove that (9r^3 - 9s^3)(r - s)^{-1} = 19 -/
theorem cubic_equation_solution (r s : ℝ) 
  (h1 : 3 * r^2 - 5 * r + 2 = 0)
  (h2 : 3 * s^2 - 5 * s + 2 = 0)
  (h3 : r ≠ s) : 
  (9 * r^3 - 9 * s^3) / (r - s) = 19 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4003_400351


namespace NUMINAMATH_CALUDE_angela_action_figures_l4003_400398

theorem angela_action_figures (initial : ℕ) (sold_fraction : ℚ) (given_fraction : ℚ) : 
  initial = 24 →
  sold_fraction = 1 / 4 →
  given_fraction = 1 / 3 →
  initial - (initial * sold_fraction).floor - ((initial - (initial * sold_fraction).floor) * given_fraction).floor = 12 := by
sorry

end NUMINAMATH_CALUDE_angela_action_figures_l4003_400398


namespace NUMINAMATH_CALUDE_solutions_to_quartic_equation_l4003_400355

theorem solutions_to_quartic_equation :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end NUMINAMATH_CALUDE_solutions_to_quartic_equation_l4003_400355


namespace NUMINAMATH_CALUDE_quadratic_equation_m_l4003_400388

/-- Given that (m+3)x^(m^2-7) + mx - 2 = 0 is a quadratic equation in x, prove that m = 3 -/
theorem quadratic_equation_m (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 3) * x^(m^2 - 7) + m * x - 2 = a * x^2 + b * x + c) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_l4003_400388


namespace NUMINAMATH_CALUDE_bob_sandwich_options_l4003_400313

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents whether turkey is available. -/
def has_turkey : Prop := True

/-- Represents whether roast beef is available. -/
def has_roast_beef : Prop := True

/-- Represents whether Swiss cheese is available. -/
def has_swiss_cheese : Prop := True

/-- Represents whether rye bread is available. -/
def has_rye_bread : Prop := True

/-- Represents the number of sandwiches with turkey and Swiss cheese. -/
def turkey_swiss_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and roast beef. -/
def rye_roast_beef_combos : ℕ := num_cheeses

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : 
  num_breads * num_meats * num_cheeses - turkey_swiss_combos - rye_roast_beef_combos = 199 :=
sorry

end NUMINAMATH_CALUDE_bob_sandwich_options_l4003_400313


namespace NUMINAMATH_CALUDE_red_box_position_l4003_400315

theorem red_box_position (n : ℕ) (initial_position : ℕ) (h1 : n = 45) (h2 : initial_position = 29) :
  n + 1 - initial_position = 17 := by
sorry

end NUMINAMATH_CALUDE_red_box_position_l4003_400315


namespace NUMINAMATH_CALUDE_sheilas_weekly_earnings_l4003_400372

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Sheila's work hours for each day -/
def workHours (d : Day) : ℕ :=
  match d with
  | Day.Monday => 8
  | Day.Tuesday => 6
  | Day.Wednesday => 8
  | Day.Thursday => 6
  | Day.Friday => 8
  | Day.Saturday => 0
  | Day.Sunday => 0

/-- Sheila's hourly wage -/
def hourlyWage : ℚ := 14

/-- Calculates daily earnings -/
def dailyEarnings (d : Day) : ℚ :=
  hourlyWage * (workHours d)

/-- Calculates weekly earnings -/
def weeklyEarnings : ℚ :=
  (dailyEarnings Day.Monday) +
  (dailyEarnings Day.Tuesday) +
  (dailyEarnings Day.Wednesday) +
  (dailyEarnings Day.Thursday) +
  (dailyEarnings Day.Friday) +
  (dailyEarnings Day.Saturday) +
  (dailyEarnings Day.Sunday)

/-- Theorem: Sheila's weekly earnings are $504 -/
theorem sheilas_weekly_earnings : weeklyEarnings = 504 := by
  sorry

end NUMINAMATH_CALUDE_sheilas_weekly_earnings_l4003_400372


namespace NUMINAMATH_CALUDE_root_one_implies_m_three_l4003_400344

theorem root_one_implies_m_three (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 2 = 0 ∧ x = 1) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_one_implies_m_three_l4003_400344


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l4003_400373

theorem sin_cos_equation_solution (x : Real) : 
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x)^4 - (Real.cos x)^4 = 1 / (Real.cos x) - 1 / (Real.sin x) ↔ 
  x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l4003_400373


namespace NUMINAMATH_CALUDE_divisor_of_p_l4003_400302

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 80 < Nat.gcd s p)
  (h5 : Nat.gcd s p < 120) :
  7 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_p_l4003_400302


namespace NUMINAMATH_CALUDE_annual_production_after_five_years_l4003_400321

/-- Given an initial value, growth rate, and time span, calculate the final value after compound growth -/
def compound_growth (initial_value : ℝ) (growth_rate : ℝ) (time_span : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ time_span

/-- Theorem: The annual production after 5 years with a given growth rate -/
theorem annual_production_after_five_years 
  (a : ℝ) -- initial production in 2005
  (x : ℝ) -- annual growth rate
  : 
  compound_growth a x 5 = a * (1 + x)^5 := by
  sorry

end NUMINAMATH_CALUDE_annual_production_after_five_years_l4003_400321


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l4003_400365

theorem min_perimeter_triangle (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  let min_perimeter := Real.sqrt (2 * (a^2 + b^2))
  ∀ c d : ℝ, (c ≥ 0) → (d ≥ 0) → 
    Real.sqrt ((a - c)^2 + b^2) + Real.sqrt ((d - c)^2 + d^2) + Real.sqrt ((a - d)^2 + (b - d)^2) ≥ min_perimeter :=
by sorry


end NUMINAMATH_CALUDE_min_perimeter_triangle_l4003_400365


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l4003_400358

theorem triangle_trigonometric_identities (A B C : ℝ) 
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 
    4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2)) ∧
  (Real.tan A + Real.tan B + Real.tan C = 
    Real.tan A * Real.tan B * Real.tan C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l4003_400358


namespace NUMINAMATH_CALUDE_fraction_power_four_l4003_400328

theorem fraction_power_four : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_fraction_power_four_l4003_400328


namespace NUMINAMATH_CALUDE_absolute_value_calculation_l4003_400368

theorem absolute_value_calculation : |-2| - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_l4003_400368


namespace NUMINAMATH_CALUDE_parallel_vector_sum_diff_l4003_400387

/-- Given two 2D vectors a and b, if a + b is parallel to a - b, then the first component of a is -4/3. -/
theorem parallel_vector_sum_diff (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = 2 ∧ b = (2, -3) →
  (∃ k : ℝ, k ≠ 0 ∧ (a + b) = k • (a - b)) →
  m = -4/3 := by sorry

end NUMINAMATH_CALUDE_parallel_vector_sum_diff_l4003_400387


namespace NUMINAMATH_CALUDE_smallest_k_with_given_remainders_l4003_400360

theorem smallest_k_with_given_remainders : ∃! k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 8 = 1 ∧
  k % 4 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 4 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_given_remainders_l4003_400360


namespace NUMINAMATH_CALUDE_set_operations_l4003_400312

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ C)) = {6, 8, 10}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l4003_400312


namespace NUMINAMATH_CALUDE_square_sum_roots_l4003_400392

theorem square_sum_roots (a b c : ℝ) (h : a ≠ 0) : 
  let roots_sum := -b / a
  (x^2 + b*x + c = 0) → roots_sum^2 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_roots_l4003_400392


namespace NUMINAMATH_CALUDE_sum_of_digits_63_l4003_400363

theorem sum_of_digits_63 : 
  let n : ℕ := 63
  let tens : ℕ := n / 10
  let ones : ℕ := n % 10
  tens - ones = 3 →
  tens + ones = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_63_l4003_400363


namespace NUMINAMATH_CALUDE_marie_keeps_remainder_l4003_400317

/-- The number of lollipops Marie keeps for herself -/
def lollipops_kept (total_lollipops : ℕ) (num_friends : ℕ) : ℕ :=
  total_lollipops % num_friends

/-- The total number of lollipops Marie has -/
def total_lollipops : ℕ := 75 + 132 + 9 + 315

/-- The number of friends Marie has -/
def num_friends : ℕ := 13

theorem marie_keeps_remainder :
  lollipops_kept total_lollipops num_friends = 11 := by
  sorry

end NUMINAMATH_CALUDE_marie_keeps_remainder_l4003_400317


namespace NUMINAMATH_CALUDE_equation_a_l4003_400340

theorem equation_a (a : ℝ) (x : ℝ) : 
  (x + a) * (x + 2*a) * (x + 3*a) * (x + 4*a) = 3*a^4 ↔ 
  x = (-5*a + a*Real.sqrt 37)/2 ∨ x = (-5*a - a*Real.sqrt 37)/2 :=
sorry


end NUMINAMATH_CALUDE_equation_a_l4003_400340


namespace NUMINAMATH_CALUDE_square_difference_forty_thirtynine_l4003_400320

theorem square_difference_forty_thirtynine : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_forty_thirtynine_l4003_400320


namespace NUMINAMATH_CALUDE_computation_proof_l4003_400384

theorem computation_proof : 55 * 1212 - 15 * 1212 = 48480 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l4003_400384


namespace NUMINAMATH_CALUDE_no_solutions_exist_l4003_400310

theorem no_solutions_exist : ¬∃ (a b : ℕ), 2019 * a^2018 = 2017 + b^2016 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l4003_400310


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4003_400345

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((3 + i) / i) = -3 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4003_400345


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4003_400330

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Theorem statement
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_condition : a 2 * a 5 < 0) :
  a 1 * a 2 * a 3 * a 4 > 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l4003_400330


namespace NUMINAMATH_CALUDE_flour_calculation_l4003_400370

/-- Given a sugar to flour ratio and an amount of sugar, calculate the required amount of flour -/
def flour_amount (sugar_flour_ratio : ℚ) (sugar_amount : ℚ) : ℚ :=
  sugar_amount / sugar_flour_ratio

theorem flour_calculation (sugar_amount : ℚ) :
  sugar_amount = 50 →
  flour_amount (10 / 1) sugar_amount = 5 := by
sorry

end NUMINAMATH_CALUDE_flour_calculation_l4003_400370


namespace NUMINAMATH_CALUDE_fraction_problem_l4003_400356

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * F * (2/5) * N = 15)
  (h2 : (40/100) * N = 180) : 
  F = 2/3 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l4003_400356


namespace NUMINAMATH_CALUDE_don_bottles_from_shop_C_l4003_400309

/-- The number of bottles Don buys from Shop A -/
def bottles_from_A : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def bottles_from_B : ℕ := 180

/-- The total number of bottles Don is capable of buying -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop C -/
def bottles_from_C : ℕ := total_bottles - (bottles_from_A + bottles_from_B)

theorem don_bottles_from_shop_C :
  bottles_from_C = 220 :=
by sorry

end NUMINAMATH_CALUDE_don_bottles_from_shop_C_l4003_400309


namespace NUMINAMATH_CALUDE_lettuce_purchase_proof_l4003_400376

/-- Calculates the total pounds of lettuce bought given the costs of green and red lettuce and the price per pound. -/
def total_lettuce_pounds (green_cost red_cost price_per_pound : ℚ) : ℚ :=
  (green_cost + red_cost) / price_per_pound

/-- Proves that given the specified costs and price per pound, the total pounds of lettuce is 7. -/
theorem lettuce_purchase_proof :
  let green_cost : ℚ := 8
  let red_cost : ℚ := 6
  let price_per_pound : ℚ := 2
  total_lettuce_pounds green_cost red_cost price_per_pound = 7 := by
sorry

#eval total_lettuce_pounds 8 6 2

end NUMINAMATH_CALUDE_lettuce_purchase_proof_l4003_400376


namespace NUMINAMATH_CALUDE_binary_1101100_equals_108_l4003_400300

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101100_equals_108 :
  binary_to_decimal [false, false, true, true, false, true, true] = 108 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101100_equals_108_l4003_400300


namespace NUMINAMATH_CALUDE_Q_has_exactly_one_negative_root_l4003_400316

def Q (x : ℝ) : ℝ := x^7 + 5*x^5 + 5*x^4 - 6*x^3 - 2*x^2 - 10*x + 12

theorem Q_has_exactly_one_negative_root :
  ∃! x : ℝ, x < 0 ∧ Q x = 0 :=
sorry

end NUMINAMATH_CALUDE_Q_has_exactly_one_negative_root_l4003_400316


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4003_400338

theorem inequality_equivalence (x : ℝ) : 3 - 1 / (3 * x + 2) < 5 ↔ x < -7/6 ∨ x > -2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4003_400338


namespace NUMINAMATH_CALUDE_red_crayons_count_l4003_400396

/-- Proves that the number of red crayons is 11 given the specified conditions. -/
theorem red_crayons_count (orange_boxes : Nat) (orange_per_box : Nat)
  (blue_boxes : Nat) (blue_per_box : Nat) (total_crayons : Nat) :
  orange_boxes = 6 → orange_per_box = 8 →
  blue_boxes = 7 → blue_per_box = 5 →
  total_crayons = 94 →
  total_crayons - (orange_boxes * orange_per_box + blue_boxes * blue_per_box) = 11 := by
  sorry

end NUMINAMATH_CALUDE_red_crayons_count_l4003_400396


namespace NUMINAMATH_CALUDE_matrix_determinant_equality_l4003_400326

theorem matrix_determinant_equality 
  (A B : Matrix (Fin 4) (Fin 4) ℝ) 
  (h1 : A * B = B * A) 
  (h2 : Matrix.det (A^2 + A*B + B^2) = 0) : 
  Matrix.det (A + B) + 3 * Matrix.det (A - B) = 6 * Matrix.det A + 6 * Matrix.det B := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_equality_l4003_400326


namespace NUMINAMATH_CALUDE_box_height_is_five_l4003_400380

-- Define the box dimensions and cube properties
def box_length : ℝ := 8
def box_width : ℝ := 15
def cube_volume : ℝ := 10
def min_cubes : ℕ := 60

-- Define the theorem
theorem box_height_is_five :
  let total_volume := (min_cubes : ℝ) * cube_volume
  let height := total_volume / (box_length * box_width)
  height = 5 := by sorry

end NUMINAMATH_CALUDE_box_height_is_five_l4003_400380


namespace NUMINAMATH_CALUDE_fishing_competition_result_l4003_400336

/-- The number of days in the fishing competition -/
def competition_days : ℕ := 5

/-- The number of fishes caught per day by the first person -/
def fishes_per_day_1 : ℕ := 6

/-- The number of fishes caught per day by the second person -/
def fishes_per_day_2 : ℕ := 4

/-- The number of fishes caught per day by the third person -/
def fishes_per_day_3 : ℕ := 8

/-- The total number of fishes caught by the team throughout the competition -/
def total_fishes : ℕ := competition_days * (fishes_per_day_1 + fishes_per_day_2 + fishes_per_day_3)

theorem fishing_competition_result : total_fishes = 90 := by
  sorry

end NUMINAMATH_CALUDE_fishing_competition_result_l4003_400336


namespace NUMINAMATH_CALUDE_solution_conditions_l4003_400394

-- Define the variables
variable (a b x y z : ℝ)

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  (a > 0) ∧ (abs b < a) ∧ (a < Real.sqrt 2 * abs b) ∧ ((3 * a^2 - b^2) * (3 * b^2 - a^2) > 0)

-- Define the equations
def equations (a b x y z : ℝ) : Prop :=
  (x + y + z = a) ∧ (x^2 + y^2 + z^2 = b^2) ∧ (x * y = z^2)

-- Define the property of distinct positive numbers
def distinct_positive (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

-- Theorem statement
theorem solution_conditions (a b x y z : ℝ) :
  equations a b x y z → (conditions a b ↔ distinct_positive x y z) := by
  sorry

end NUMINAMATH_CALUDE_solution_conditions_l4003_400394


namespace NUMINAMATH_CALUDE_print_time_with_rate_change_l4003_400393

/-- Represents the printing scenario with given parameters -/
structure PrintingScenario where
  num_presses : ℕ
  initial_time : ℝ
  new_time : ℝ
  num_papers : ℕ

/-- Calculates the time taken to print papers given a printing scenario -/
def time_to_print (s : PrintingScenario) : ℝ :=
  s.new_time

/-- Theorem stating that the time to print remains the same as the new_time 
    when the printing rate changes but the number of presses remains constant -/
theorem print_time_with_rate_change (s : PrintingScenario) 
  (h1 : s.num_presses = 35)
  (h2 : s.initial_time = 15)
  (h3 : s.new_time = 21)
  (h4 : s.num_papers = 500000) :
  time_to_print s = s.new_time := by
  sorry


end NUMINAMATH_CALUDE_print_time_with_rate_change_l4003_400393


namespace NUMINAMATH_CALUDE_square_root_of_25_l4003_400311

theorem square_root_of_25 : ∃ (x y : ℝ), x^2 = 25 ∧ y^2 = 25 ∧ x = 5 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_25_l4003_400311


namespace NUMINAMATH_CALUDE_square_area_error_l4003_400347

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x * 1.12
  let actual_area := x^2
  let calculated_area := measured_side^2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 25.44 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l4003_400347


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4003_400305

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4003_400305


namespace NUMINAMATH_CALUDE_unique_odd_number_with_eight_multiples_l4003_400331

theorem unique_odd_number_with_eight_multiples : 
  ∃! x : ℕ, 
    x % 2 = 1 ∧ 
    x > 0 ∧
    (∃ S : Finset ℕ, 
      S.card = 8 ∧
      (∀ y ∈ S, 
        y < 80 ∧ 
        y % 2 = 1 ∧
        ∃ k m : ℕ, k > 0 ∧ m % 2 = 1 ∧ y = k * x * m) ∧
      (∀ y : ℕ, 
        y < 80 → 
        y % 2 = 1 → 
        (∃ k m : ℕ, k > 0 ∧ m % 2 = 1 ∧ y = k * x * m) → 
        y ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_unique_odd_number_with_eight_multiples_l4003_400331


namespace NUMINAMATH_CALUDE_orange_selling_loss_l4003_400314

/-- Calculates the percentage loss when selling oranges at a given rate per rupee,
    given the rate that would result in a 44% gain. -/
def calculate_loss_percentage (loss_rate : ℚ) (gain_rate : ℚ) (gain_percentage : ℚ) : ℚ :=
  let cost_price := 1 / (gain_rate * (1 + gain_percentage))
  let loss := cost_price - 1 / loss_rate
  (loss / cost_price) * 100

/-- The percentage loss when selling oranges at 36 per rupee is approximately 4.17%,
    given that selling at 24 per rupee results in a 44% gain. -/
theorem orange_selling_loss : 
  let loss_rate : ℚ := 36
  let gain_rate : ℚ := 24
  let gain_percentage : ℚ := 44 / 100
  let calculated_loss := calculate_loss_percentage loss_rate gain_rate gain_percentage
  ∃ ε > 0, abs (calculated_loss - 4.17) < ε ∧ ε < 0.01 :=
sorry

end NUMINAMATH_CALUDE_orange_selling_loss_l4003_400314


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4003_400325

/-- The function f(x) = a^(x-1) + 3 passes through the point (1, 4) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l4003_400325


namespace NUMINAMATH_CALUDE_min_sum_abcd_l4003_400354

theorem min_sum_abcd (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) :
  a + b + c + d ≥ 108 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abcd_l4003_400354


namespace NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l4003_400379

/-- The function f(x) with parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The largest value of c such that -2 is in the range of f(x) -/
theorem largest_c_for_negative_two_in_range :
  ∃ (c_max : ℝ), 
    (∃ (x : ℝ), f c_max x = -2) ∧ 
    (∀ (c : ℝ), c > c_max → ¬∃ (x : ℝ), f c x = -2) ∧
    c_max = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l4003_400379


namespace NUMINAMATH_CALUDE_team_selection_count_l4003_400399

/-- The number of ways to select a team of 8 students from 10 boys and 12 girls, with at least 4 girls -/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (min_girls : ℕ) : ℕ :=
  (Nat.choose total_girls min_girls * Nat.choose total_boys (team_size - min_girls)) +
  (Nat.choose total_girls (min_girls + 1) * Nat.choose total_boys (team_size - min_girls - 1)) +
  (Nat.choose total_girls (min_girls + 2) * Nat.choose total_boys (team_size - min_girls - 2)) +
  (Nat.choose total_girls (min_girls + 3) * Nat.choose total_boys (team_size - min_girls - 3)) +
  (Nat.choose total_girls (min_girls + 4))

theorem team_selection_count :
  select_team 10 12 8 4 = 245985 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l4003_400399


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4003_400307

theorem least_addition_for_divisibility :
  ∃ (x : ℕ), x = 4 ∧ 
  (28 ∣ (1056 + x)) ∧ 
  (∀ (y : ℕ), y < x → ¬(28 ∣ (1056 + y))) :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4003_400307


namespace NUMINAMATH_CALUDE_fraction_inequality_l4003_400349

theorem fraction_inequality (a : ℝ) : a > 1 → (2*a + 1)/(a - 1) > 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l4003_400349


namespace NUMINAMATH_CALUDE_square_with_tens_digit_seven_l4003_400391

theorem square_with_tens_digit_seven (n : ℕ) (h : n > 0) :
  (n^2 / 10) % 10 = 7 → n^2 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_with_tens_digit_seven_l4003_400391


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_and_angle_l4003_400342

/-- Given an arc length of 4 cm corresponding to a central angle of 2 radians,
    the area of the sector formed by this central angle is 4 cm^2. -/
theorem sector_area_from_arc_length_and_angle (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = 2) :
  let r := s / θ
  (1 / 2) * r^2 * θ = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_and_angle_l4003_400342


namespace NUMINAMATH_CALUDE_better_fit_for_lower_rss_l4003_400375

/-- Represents a model with its residual sum of squares -/
structure Model where
  rss : ℝ

/-- Definition of a better fit model -/
def better_fit (m1 m2 : Model) : Prop := m1.rss < m2.rss

theorem better_fit_for_lower_rss (model1 model2 : Model) 
  (h1 : model1.rss = 152.6) 
  (h2 : model2.rss = 159.8) : 
  better_fit model1 model2 := by
  sorry

#check better_fit_for_lower_rss

end NUMINAMATH_CALUDE_better_fit_for_lower_rss_l4003_400375


namespace NUMINAMATH_CALUDE_diamond_calculation_l4003_400371

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a + 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5)) = 89 / 420 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l4003_400371


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4003_400335

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Complex.abs z = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4003_400335


namespace NUMINAMATH_CALUDE_prime_iff_divisibility_condition_l4003_400369

theorem prime_iff_divisibility_condition (n : ℕ) (h : n ≥ 2) :
  Prime n ↔ ∀ d : ℕ, d > 1 → d ∣ n → (d^2 + n) ∣ (n^2 + d) :=
sorry

end NUMINAMATH_CALUDE_prime_iff_divisibility_condition_l4003_400369


namespace NUMINAMATH_CALUDE_weight_of_b_l4003_400395

/-- Given the average weights of different combinations of people, prove the weight of person b. -/
theorem weight_of_b (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : (b + c + d) / 3 = 47) :
  b = 31 := by
  sorry


end NUMINAMATH_CALUDE_weight_of_b_l4003_400395


namespace NUMINAMATH_CALUDE_cyclist_distance_theorem_l4003_400327

/-- A cyclist travels in a straight line for two minutes. -/
def cyclist_travel (v1 v2 : ℝ) : ℝ := v1 * 60 + v2 * 60

/-- The theorem states that a cyclist traveling at 2 m/s for the first minute
    and 4 m/s for the second minute covers a total distance of 360 meters. -/
theorem cyclist_distance_theorem :
  cyclist_travel 2 4 = 360 := by sorry

end NUMINAMATH_CALUDE_cyclist_distance_theorem_l4003_400327


namespace NUMINAMATH_CALUDE_factorization_proof_l4003_400367

theorem factorization_proof (x y : ℝ) : 75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l4003_400367


namespace NUMINAMATH_CALUDE_baxter_earnings_l4003_400389

structure School where
  name : String
  students : ℕ
  days : ℕ
  bonus : ℚ

def total_student_days (schools : List School) : ℕ :=
  schools.foldl (fun acc s => acc + s.students * s.days) 0

def total_bonus (schools : List School) : ℚ :=
  schools.foldl (fun acc s => acc + s.students * s.bonus) 0

theorem baxter_earnings (schools : List School) 
  (h_schools : schools = [
    ⟨"Ajax", 5, 4, 0⟩, 
    ⟨"Baxter", 3, 6, 5⟩, 
    ⟨"Colton", 6, 8, 0⟩
  ])
  (h_total_paid : 920 = (total_student_days schools) * (daily_wage : ℚ) + total_bonus schools)
  (daily_wage : ℚ) :
  ∃ (baxter_earnings : ℚ), baxter_earnings = 204.42 ∧ 
    baxter_earnings = 3 * 6 * daily_wage + 3 * 5 :=
by sorry

end NUMINAMATH_CALUDE_baxter_earnings_l4003_400389
