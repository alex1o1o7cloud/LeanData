import Mathlib

namespace NUMINAMATH_CALUDE_car_repair_cost_proof_l3566_356618

/-- Calculates the total cost for a car repair given the hourly rate, hours worked per day,
    number of days worked, and cost of parts. -/
def total_repair_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_worked + parts_cost

/-- Proves that given the specified conditions, the total cost for the car's owner is $9220. -/
theorem car_repair_cost_proof :
  total_repair_cost 60 8 14 2500 = 9220 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_proof_l3566_356618


namespace NUMINAMATH_CALUDE_enclosure_blocks_l3566_356602

/-- Calculates the number of blocks required for a rectangular enclosure --/
def blocks_required (length width height : ℕ) : ℕ :=
  let external_volume := length * width * height
  let internal_length := length - 2
  let internal_width := width - 2
  let internal_height := height - 2
  let internal_volume := internal_length * internal_width * internal_height
  external_volume - internal_volume

/-- Proves that the number of blocks required for the given dimensions is 598 --/
theorem enclosure_blocks : blocks_required 15 13 6 = 598 := by
  sorry

end NUMINAMATH_CALUDE_enclosure_blocks_l3566_356602


namespace NUMINAMATH_CALUDE_distance_between_points_l3566_356631

theorem distance_between_points (a : ℝ) : 
  let A : ℝ × ℝ := (a, -2)
  let B : ℝ × ℝ := (0, 3)
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 7^2) → (a = 2 * Real.sqrt 6 ∨ a = -2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3566_356631


namespace NUMINAMATH_CALUDE_fuel_station_service_cost_l3566_356603

/-- Fuel station problem -/
theorem fuel_station_service_cost
  (fuel_cost_per_liter : Real)
  (num_minivans : Nat)
  (num_trucks : Nat)
  (total_cost : Real)
  (minivan_tank_capacity : Real)
  (truck_tank_multiplier : Real)
  (h1 : fuel_cost_per_liter = 0.70)
  (h2 : num_minivans = 4)
  (h3 : num_trucks = 2)
  (h4 : total_cost = 395.4)
  (h5 : minivan_tank_capacity = 65)
  (h6 : truck_tank_multiplier = 2.2)
  : (total_cost - (fuel_cost_per_liter * 
      (num_minivans * minivan_tank_capacity + 
       num_trucks * (minivan_tank_capacity * truck_tank_multiplier)))) / 
    (num_minivans + num_trucks) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_service_cost_l3566_356603


namespace NUMINAMATH_CALUDE_cricket_run_rate_problem_l3566_356698

/-- Calculates the required run rate for the remaining overs in a cricket game -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_problem :
  required_run_rate 50 10 (32/10) 272 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_problem_l3566_356698


namespace NUMINAMATH_CALUDE_circle_delta_area_l3566_356634

/-- Circle δ with points A and B -/
structure Circle_delta where
  center : ℝ × ℝ
  radius : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Conditions for the circle δ -/
def circle_conditions (δ : Circle_delta) : Prop :=
  δ.A = (2, 9) ∧ 
  δ.B = (10, 5) ∧
  (δ.A.1 - δ.center.1)^2 + (δ.A.2 - δ.center.2)^2 = δ.radius^2 ∧
  (δ.B.1 - δ.center.1)^2 + (δ.B.2 - δ.center.2)^2 = δ.radius^2

/-- Tangent lines intersection condition -/
def tangent_intersection (δ : Circle_delta) : Prop :=
  ∃ x : ℝ, 
    let slope_AB := (δ.B.2 - δ.A.2) / (δ.B.1 - δ.A.1)
    let perp_slope := -1 / slope_AB
    let midpoint := ((δ.A.1 + δ.B.1) / 2, (δ.A.2 + δ.B.2) / 2)
    perp_slope * (x - midpoint.1) + midpoint.2 = 0

/-- Theorem stating the area of circle δ -/
theorem circle_delta_area (δ : Circle_delta) 
  (h1 : circle_conditions δ) (h2 : tangent_intersection δ) : 
  π * δ.radius^2 = 83.44 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_delta_area_l3566_356634


namespace NUMINAMATH_CALUDE_hockey_championship_wins_l3566_356658

theorem hockey_championship_wins (total_matches : ℕ) (total_points : ℤ) 
  (win_points loss_points : ℤ) (h_total_matches : total_matches = 38) 
  (h_total_points : total_points = 60) (h_win_points : win_points = 12) 
  (h_loss_points : loss_points = 5) : 
  ∃! wins : ℕ, ∃ losses draws : ℕ,
    wins + losses + draws = total_matches ∧ 
    wins * win_points - losses * loss_points = total_points ∧
    losses > 0 := by
  sorry

#check hockey_championship_wins

end NUMINAMATH_CALUDE_hockey_championship_wins_l3566_356658


namespace NUMINAMATH_CALUDE_maries_socks_l3566_356620

theorem maries_socks (x y z : ℕ) : 
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≥ 1 →
  y ≥ 1 →
  z ≥ 1 →
  x = 11 := by
sorry

end NUMINAMATH_CALUDE_maries_socks_l3566_356620


namespace NUMINAMATH_CALUDE_good_number_ending_8_has_9_before_l3566_356649

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

def ends_with_8 (n : ℕ) : Prop :=
  n % 10 = 8

def second_to_last_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem good_number_ending_8_has_9_before :
  ∀ n : ℕ, is_good n → ends_with_8 n → second_to_last_digit n = 9 := by
  sorry

end NUMINAMATH_CALUDE_good_number_ending_8_has_9_before_l3566_356649


namespace NUMINAMATH_CALUDE_intersecting_circles_m_plus_c_l3566_356673

/-- Two circles intersect at points A and B, with the centers of the circles lying on a line. -/
structure IntersectingCircles where
  m : ℝ
  c : ℝ
  A : ℝ × ℝ := (1, 3)
  B : ℝ × ℝ := (m, -1)
  centers_line : ℝ → ℝ := fun x ↦ x + c

/-- The value of m+c for the given intersecting circles configuration is 3. -/
theorem intersecting_circles_m_plus_c (circles : IntersectingCircles) : 
  circles.m + circles.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_intersecting_circles_m_plus_c_l3566_356673


namespace NUMINAMATH_CALUDE_smallest_bob_number_l3566_356600

def alice_number : ℕ := 24

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ),
    bob_number > 0 ∧
    has_all_prime_factors alice_number bob_number ∧
    (∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → bob_number ≤ m) ∧
    bob_number = 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l3566_356600


namespace NUMINAMATH_CALUDE_max_handshakes_specific_gathering_l3566_356694

/-- Represents a gathering of people and their handshakes. -/
structure Gathering where
  people : Nat
  restricted_people : Nat
  max_handshakes_per_person : Nat
  max_handshakes_for_restricted : Nat

/-- Calculates the maximum number of handshakes in a gathering. -/
def max_handshakes (g : Gathering) : Nat :=
  sorry

/-- Theorem stating the maximum number of handshakes for the specific gathering. -/
theorem max_handshakes_specific_gathering :
  let g : Gathering := {
    people := 30,
    restricted_people := 5,
    max_handshakes_per_person := 29,
    max_handshakes_for_restricted := 10
  }
  max_handshakes g = 325 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_specific_gathering_l3566_356694


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3566_356665

theorem fractional_inequality_solution_set (x : ℝ) :
  (x + 5) / (1 - x) ≤ 0 ↔ (x ≤ -5 ∨ x > 1) ∧ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l3566_356665


namespace NUMINAMATH_CALUDE_sum_base6_series_l3566_356635

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : ℕ) : ℕ := sorry

/-- Sum of arithmetic series in base 10 -/
def sumArithmeticSeries (a l n : ℕ) : ℕ := n * (a + l) / 2

theorem sum_base6_series :
  let first := base6To10 3
  let last := base6To10 100
  let n := last - first + 1
  base10To6 (sumArithmeticSeries first last n) = 3023 :=
by sorry

end NUMINAMATH_CALUDE_sum_base6_series_l3566_356635


namespace NUMINAMATH_CALUDE_max_product_digits_l3566_356644

theorem max_product_digits : ∀ a b : ℕ,
  10000 ≤ a ∧ a < 100000 →
  1000 ≤ b ∧ b < 10000 →
  a * b < 1000000000 := by
sorry

end NUMINAMATH_CALUDE_max_product_digits_l3566_356644


namespace NUMINAMATH_CALUDE_square_difference_equality_l3566_356670

theorem square_difference_equality : 1012^2 - 992^2 - 1008^2 + 996^2 = 16032 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3566_356670


namespace NUMINAMATH_CALUDE_monomial_exponents_sum_l3566_356663

/-- Two monomials are like terms if their variables have the same exponents -/
def are_like_terms (a b c d : ℕ) : Prop :=
  a = c ∧ b = d

theorem monomial_exponents_sum (m n : ℕ) : 
  are_like_terms 5 (2*n) m 4 → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_sum_l3566_356663


namespace NUMINAMATH_CALUDE_power_inequality_l3566_356651

theorem power_inequality (x y : ℝ) (h : x^2013 + y^2013 > x^2012 + y^2012) :
  x^2014 + y^2014 > x^2013 + y^2013 :=
by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3566_356651


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l3566_356629

/-- Given the conversion rates between knicks, knacks, and knocks, 
    prove that 30 knocks are equal to 20 knicks. -/
theorem knicks_knacks_knocks_conversion :
  ∀ (knicks knacks knocks : ℚ),
    (5 * knicks = 3 * knacks) →
    (2 * knacks = 5 * knocks) →
    (30 * knocks = 20 * knicks) :=
by
  sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l3566_356629


namespace NUMINAMATH_CALUDE_geometric_sequence_max_value_l3566_356628

theorem geometric_sequence_max_value (a b c d : ℝ) : 
  (∃ r : ℝ, a * r = b ∧ b * r = c ∧ c * r = d) →  -- geometric sequence condition
  (∀ x : ℝ, Real.log (x + 2) - x ≤ c) →           -- maximum value condition
  (Real.log (b + 2) - b = c) →                    -- maximum occurs at x = b
  a * d = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_value_l3566_356628


namespace NUMINAMATH_CALUDE_parabolas_intersection_l3566_356685

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15
def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 11

-- Define the intersection points
def intersection_points : Set ℝ := {x | parabola1 x = parabola2 x}

-- Theorem statement
theorem parabolas_intersection :
  ∃ (x1 x2 : ℝ), x1 ∈ intersection_points ∧ x2 ∈ intersection_points ∧
  x1 = (3 + Real.sqrt 61) / 2 ∧ x2 = (3 - Real.sqrt 61) / 2 :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l3566_356685


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l3566_356627

/-- The number of days it takes for one cow to eat one bag of husk -/
def days_for_one_cow_one_bag (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) : ℚ :=
  (num_days : ℚ) * (num_cows : ℚ) / (num_bags : ℚ)

/-- Theorem stating that one cow will eat one bag of husk in 36 days -/
theorem one_cow_one_bag_days :
  days_for_one_cow_one_bag 60 75 45 = 36 := by
  sorry

#eval days_for_one_cow_one_bag 60 75 45

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l3566_356627


namespace NUMINAMATH_CALUDE_square_area_problem_l3566_356642

theorem square_area_problem : 
  ∀ x : ℚ, 
  (5 * x - 20 : ℚ) = (25 - 2 * x : ℚ) → 
  (5 * x - 20 : ℚ) > 0 →
  (5 * x - 20 : ℚ)^2 = 7225 / 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_problem_l3566_356642


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3566_356671

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
noncomputable def ω : ℂ := sorry

/-- The property that ω^2 + ω + 1 = 0 -/
axiom ω_property : ω^2 + ω + 1 = 0

/-- The polynomial x^104 + Ax^3 + Bx -/
def polynomial (A B : ℝ) (x : ℂ) : ℂ := x^104 + A * x^3 + B * x

/-- The divisibility condition -/
def is_divisible (A B : ℝ) : Prop :=
  polynomial A B ω = 0

theorem polynomial_divisibility (A B : ℝ) :
  is_divisible A B → A + B = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3566_356671


namespace NUMINAMATH_CALUDE_hajis_mother_sales_l3566_356630

/-- Haji's mother's sales problem -/
theorem hajis_mother_sales (tough_week_sales : ℕ) (total_sales : ℕ) :
  tough_week_sales = 800 →
  total_sales = 10400 →
  ∃ (good_weeks : ℕ),
    good_weeks * (2 * tough_week_sales) + 3 * tough_week_sales = total_sales ∧
    good_weeks = 5 :=
by sorry

end NUMINAMATH_CALUDE_hajis_mother_sales_l3566_356630


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_exists_l3566_356641

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a division of an equilateral triangle into smaller equilateral triangles -/
structure TriangleDivision where
  original : EquilateralTriangle
  num_divisions : ℕ
  side_lengths : Finset ℝ
  all_positive : ∀ l ∈ side_lengths, l > 0

/-- Theorem stating that there exists a division of an equilateral triangle into 2011 smaller equilateral triangles with only two different side lengths -/
theorem equilateral_triangle_division_exists : 
  ∃ (div : TriangleDivision), div.num_divisions = 2011 ∧ div.side_lengths.card = 2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_exists_l3566_356641


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3566_356680

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem tangent_point_coordinates (a : ℝ) :
  (∀ x, f_deriv a (-x) = -f_deriv a x) →
  ∃ x₀ y₀, f a x₀ = y₀ ∧ f_deriv a x₀ = 3/2 →
  x₀ = Real.log 2 ∧ y₀ = 5/2 := by sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3566_356680


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_13_18_l3566_356660

theorem smallest_divisible_by_15_13_18 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 13 ∣ n ∧ 18 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 13 ∣ m → 18 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_13_18_l3566_356660


namespace NUMINAMATH_CALUDE_defective_probability_l3566_356633

/-- The probability of a randomly chosen unit being defective in a factory with two machines --/
theorem defective_probability (total_output : ℝ) (machine_a_output : ℝ) (machine_b_output : ℝ)
  (machine_a_defective_rate : ℝ) (machine_b_defective_rate : ℝ) :
  machine_a_output = 0.4 * total_output →
  machine_b_output = 0.6 * total_output →
  machine_a_defective_rate = 9 / 1000 →
  machine_b_defective_rate = 1 / 50 →
  (machine_a_output / total_output) * machine_a_defective_rate +
  (machine_b_output / total_output) * machine_b_defective_rate = 0.0156 := by
  sorry


end NUMINAMATH_CALUDE_defective_probability_l3566_356633


namespace NUMINAMATH_CALUDE_percentage_prefer_x_is_zero_l3566_356639

def total_employees : ℕ := 200
def relocated_to_x : ℚ := 30 / 100
def relocated_to_y : ℚ := 70 / 100
def prefer_y : ℚ := 40 / 100
def max_satisfied : ℕ := 140

theorem percentage_prefer_x_is_zero :
  ∃ (prefer_x : ℚ),
    prefer_x ≥ 0 ∧
    prefer_x + prefer_y = 1 ∧
    (prefer_x * total_employees).floor + (prefer_y * total_employees).floor ≤ max_satisfied ∧
    prefer_x = 0 := by sorry

end NUMINAMATH_CALUDE_percentage_prefer_x_is_zero_l3566_356639


namespace NUMINAMATH_CALUDE_number_divisibility_l3566_356682

theorem number_divisibility (x : ℝ) : (x / 6) * 12 = 18 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l3566_356682


namespace NUMINAMATH_CALUDE_lenny_pens_percentage_l3566_356683

theorem lenny_pens_percentage (total_boxes : ℕ) (pens_per_box : ℕ) (remaining_pens : ℕ) : 
  total_boxes = 20 →
  pens_per_box = 5 →
  remaining_pens = 45 →
  ∃ (percentage : ℚ),
    percentage = 40 ∧
    (3/4 : ℚ) * ((total_boxes * pens_per_box : ℚ) - percentage) / 100 * (total_boxes * pens_per_box) = remaining_pens :=
by sorry

end NUMINAMATH_CALUDE_lenny_pens_percentage_l3566_356683


namespace NUMINAMATH_CALUDE_work_time_proof_l3566_356655

/-- Represents the time taken by A to complete the work alone -/
def time_A : ℝ := 6

/-- Represents the time taken by B to complete the work alone -/
def time_B : ℝ := 8

/-- Represents the time taken by A, B, and C together to complete the work -/
def time_ABC : ℝ := 3

/-- Represents A's share of the payment -/
def share_A : ℝ := 300

/-- Represents B's share of the payment -/
def share_B : ℝ := 225

/-- Represents C's share of the payment -/
def share_C : ℝ := 75

/-- Represents the total payment for the work -/
def total_payment : ℝ := 600

theorem work_time_proof :
  (1 / time_A + 1 / time_B + (share_C / share_A) / time_A = 1 / time_ABC) ∧
  (share_A / share_B = 4 / 3) ∧
  (share_A / share_C = 4) ∧
  (share_A + share_B + share_C = total_payment) →
  time_A = 6 := by sorry

end NUMINAMATH_CALUDE_work_time_proof_l3566_356655


namespace NUMINAMATH_CALUDE_division_of_fractions_l3566_356636

theorem division_of_fractions : (3 : ℚ) / 7 / 4 = 3 / 28 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3566_356636


namespace NUMINAMATH_CALUDE_triangle_inequality_fraction_l3566_356690

theorem triangle_inequality_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : (a + b) / (a + b + c) > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_fraction_l3566_356690


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3566_356632

theorem sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ (∃ a : ℝ, a ≤ 1 ∧ 1 / a < 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3566_356632


namespace NUMINAMATH_CALUDE_arctan_sum_problem_l3566_356681

theorem arctan_sum_problem (a b : ℝ) : 
  a = 1/3 → 
  (a + 2) * (b + 2) = 15 → 
  Real.arctan a + Real.arctan b = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_problem_l3566_356681


namespace NUMINAMATH_CALUDE_A_n_nonempty_finite_l3566_356637

/-- The set A_n for a positive integer n -/
def A_n (n : ℕ+) : Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ k : ℕ, (Real.sqrt (p.1^2 + p.2 + n) + Real.sqrt (p.2^2 + p.1 + n) : ℝ) = k}

/-- Theorem stating that A_n is non-empty and finite for any positive integer n -/
theorem A_n_nonempty_finite (n : ℕ+) : Set.Nonempty (A_n n) ∧ Set.Finite (A_n n) := by
  sorry

end NUMINAMATH_CALUDE_A_n_nonempty_finite_l3566_356637


namespace NUMINAMATH_CALUDE_seaweed_for_fires_l3566_356613

theorem seaweed_for_fires (total_seaweed livestock_feed : ℝ)
  (h1 : total_seaweed = 400)
  (h2 : livestock_feed = 150)
  (h3 : livestock_feed = 0.75 * (1 - fire_percentage / 100) * total_seaweed) :
  fire_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_for_fires_l3566_356613


namespace NUMINAMATH_CALUDE_stratified_sampling_school_a_l3566_356669

theorem stratified_sampling_school_a (total_sample : ℕ) 
  (school_a : ℕ) (school_b : ℕ) (school_c : ℕ) : 
  total_sample = 90 → 
  school_a = 3600 → 
  school_b = 5400 → 
  school_c = 1800 → 
  (school_a * total_sample) / (school_a + school_b + school_c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_school_a_l3566_356669


namespace NUMINAMATH_CALUDE_total_rain_time_l3566_356616

/-- Given rain durations over three days, prove the total rain time -/
theorem total_rain_time (first_day_start : Nat) (first_day_end : Nat)
  (h1 : first_day_end - first_day_start = 10)
  (h2 : ∃ second_day_duration : Nat, second_day_duration = (first_day_end - first_day_start) + 2)
  (h3 : ∃ third_day_duration : Nat, third_day_duration = 2 * (first_day_end - first_day_start + 2)) :
  ∃ total_duration : Nat, total_duration = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_rain_time_l3566_356616


namespace NUMINAMATH_CALUDE_matrix_inverse_zero_if_singular_l3566_356677

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 8; -2, -4]

theorem matrix_inverse_zero_if_singular :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_zero_if_singular_l3566_356677


namespace NUMINAMATH_CALUDE_deepak_current_age_l3566_356691

-- Define the ratio of Rahul to Deepak's age
def age_ratio : ℚ := 4 / 3

-- Define Rahul's age after 4 years
def rahul_future_age : ℕ := 32

-- Define the number of years in the future for Rahul's age
def years_in_future : ℕ := 4

-- Theorem to prove Deepak's current age
theorem deepak_current_age :
  ∃ (rahul_age deepak_age : ℕ),
    (rahul_age : ℚ) / deepak_age = age_ratio ∧
    rahul_age + years_in_future = rahul_future_age ∧
    deepak_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_deepak_current_age_l3566_356691


namespace NUMINAMATH_CALUDE_complex_magnitude_l3566_356664

theorem complex_magnitude (z : ℂ) (h : z * (1 - 2*I) = 3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3566_356664


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l3566_356611

/-- Proves that given a distance that takes 2 hours 45 minutes to walk and 40 minutes to run at 16.5 kmph, the walking speed is 4 kmph. -/
theorem walking_speed_calculation (distance : ℝ) : 
  distance / (2 + 45 / 60) = 4 → distance / (40 / 60) = 16.5 → distance / (2 + 45 / 60) = 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l3566_356611


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3566_356646

/-- The surface area of a sphere circumscribing a right square prism -/
theorem circumscribed_sphere_surface_area (a h : ℝ) (ha : a = 2) (hh : h = 3) :
  let R := (1 / 2 : ℝ) * Real.sqrt (h^2 + 2 * a^2)
  4 * Real.pi * R^2 = 17 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3566_356646


namespace NUMINAMATH_CALUDE_gnuff_tutoring_time_l3566_356676

/-- Proves that given Gnuff's tutoring rates and total amount paid, the number of minutes tutored is 18 --/
theorem gnuff_tutoring_time (flat_rate per_minute_rate total_paid : ℕ) : 
  flat_rate = 20 → 
  per_minute_rate = 7 → 
  total_paid = 146 → 
  (total_paid - flat_rate) / per_minute_rate = 18 := by
sorry

end NUMINAMATH_CALUDE_gnuff_tutoring_time_l3566_356676


namespace NUMINAMATH_CALUDE_circle_equation_l3566_356606

/-- The equation of a circle passing through point P(2,5) with center C(8,-3) -/
theorem circle_equation (x y : ℝ) : 
  let P : ℝ × ℝ := (2, 5)
  let C : ℝ × ℝ := (8, -3)
  (x - C.1)^2 + (y - C.2)^2 = (P.1 - C.1)^2 + (P.2 - C.2)^2 ↔ 
  (x - 8)^2 + (y + 3)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3566_356606


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l3566_356601

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ+) (h1 : Nat.gcd A B = 23)
  (h2 : Nat.lcm A B = 23 * X * 16) (h3 : A = 368) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l3566_356601


namespace NUMINAMATH_CALUDE_set_operations_l3566_356668

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | -3 ≤ x ∧ x < -1}) ∧
  (A ∪ B = {x | x ≤ 4 ∨ x > 5}) ∧
  ((Set.compl A) ∩ (Set.compl B) = {x | 4 < x ∧ x ≤ 5}) :=
sorry

end NUMINAMATH_CALUDE_set_operations_l3566_356668


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3566_356608

theorem quadratic_factorization (c d : ℤ) : 
  (∀ x, 25*x^2 - 85*x - 90 = (5*x + c) * (5*x + d)) → c + 2*d = -24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3566_356608


namespace NUMINAMATH_CALUDE_power_product_simplification_l3566_356674

theorem power_product_simplification :
  (5 / 3 : ℚ) ^ 2023 * (6 / 10 : ℚ) ^ 2022 = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_power_product_simplification_l3566_356674


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3566_356695

theorem power_of_two_equality (y : ℤ) : (1 / 8 : ℚ) * (2 ^ 40 : ℚ) = (2 : ℚ) ^ y → y = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3566_356695


namespace NUMINAMATH_CALUDE_triangle_altitude_sum_square_l3566_356643

theorem triangle_altitude_sum_square (a b c : ℕ) : 
  (∃ (h_a h_b h_c : ℝ), h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ 
    h_a = (2 * (a * h_a / 2)) / a ∧ 
    h_b = (2 * (b * h_b / 2)) / b ∧ 
    h_c = (2 * (c * h_c / 2)) / c ∧ 
    h_a = h_b + h_c) → 
  ∃ (k : ℚ), a^2 + b^2 + c^2 = k^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_sum_square_l3566_356643


namespace NUMINAMATH_CALUDE_shoe_size_for_given_length_xiao_gang_shoe_size_l3566_356689

/-- A linear function representing the relationship between shoe size and foot length. -/
def shoe_size_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

/-- Theorem stating the shoe size for a given foot length based on the given conditions. -/
theorem shoe_size_for_given_length (k b : ℝ) :
  shoe_size_function k b 23 = 36 →
  shoe_size_function k b 26 = 42 →
  shoe_size_function k b 24.5 = 39 := by
  sorry

/-- Corollary: Xiao Gang's shoe size -/
theorem xiao_gang_shoe_size (k b : ℝ) :
  shoe_size_function k b 23 = 36 →
  shoe_size_function k b 26 = 42 →
  ∃ y : ℝ, y = shoe_size_function k b 24.5 ∧ y = 39 := by
  sorry

end NUMINAMATH_CALUDE_shoe_size_for_given_length_xiao_gang_shoe_size_l3566_356689


namespace NUMINAMATH_CALUDE_reciprocal_roots_imply_a_eq_neg_one_l3566_356638

theorem reciprocal_roots_imply_a_eq_neg_one (a : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
    x^2 + (a-1)*x + a^2 = 0 ∧ 
    y^2 + (a-1)*y + a^2 = 0 ∧ 
    x*y = 1) → 
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_imply_a_eq_neg_one_l3566_356638


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3566_356604

theorem midpoint_trajectory (a b : ℝ) : 
  a^2 + b^2 = 1 → ∃ x y : ℝ, x = a ∧ y = b/2 ∧ x^2 + 4*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3566_356604


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l3566_356687

-- Define the cone's properties
def base_radius : ℝ := 6

-- Define the theorem
theorem cone_generatrix_length :
  ∀ (generatrix : ℝ),
  (2 * Real.pi * base_radius = Real.pi * generatrix) →
  generatrix = 12 := by
sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l3566_356687


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3566_356672

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  ∃ k, n = k * (lcm 3 (lcm 4 (lcm 5 (lcm 6 7)))) + 1

theorem smallest_valid_number : 
  is_valid_number 61 ∧ ∀ m, is_valid_number m → m ≥ 61 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3566_356672


namespace NUMINAMATH_CALUDE_ellipse_minimum_value_l3566_356666

theorem ellipse_minimum_value (x y : ℝ) :
  x > 0 → y > 0 → x^2 / 16 + y^2 / 12 = 1 →
  x / (4 - x) + 3 * y / (6 - y) ≥ 4 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 / 16 + y₀^2 / 12 = 1 ∧
    x₀ / (4 - x₀) + 3 * y₀ / (6 - y₀) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minimum_value_l3566_356666


namespace NUMINAMATH_CALUDE_parabola_equation_theorem_l3566_356662

/-- Define a parabola by its focus and directrix -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → ℝ

/-- The equation of a parabola in general form -/
def parabola_equation (a b c d e f : ℤ) (x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0

/-- The given parabola -/
def given_parabola : Parabola :=
  { focus := (4, 4),
    directrix := λ x y => 4 * x + 8 * y - 32 }

/-- Theorem stating the equation of the given parabola -/
theorem parabola_equation_theorem :
  ∃ (a b c d e f : ℤ),
    (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | parabola_equation a b c d e f p.1 p.2} ↔ 
      (x - given_parabola.focus.1)^2 + (y - given_parabola.focus.2)^2 = 
      (given_parabola.directrix x y)^2 / (4^2 + 8^2)) ∧
    a > 0 ∧
    Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)) (Int.natAbs e)) (Int.natAbs f) = 1 ∧
    a = 16 ∧ b = -64 ∧ c = 64 ∧ d = -128 ∧ e = -256 ∧ f = 768 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_theorem_l3566_356662


namespace NUMINAMATH_CALUDE_largest_n_for_product_2210_l3566_356675

/-- An arithmetic sequence with integer terms -/
def ArithmeticSeq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_2210 :
  ∀ a b : ℕ → ℕ,
  ArithmeticSeq a → ArithmeticSeq b →
  a 1 = 1 → b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 2210) →
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 2210) → m ≤ 170) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2210_l3566_356675


namespace NUMINAMATH_CALUDE_parabola_equation_l3566_356610

/-- A parabola with the same shape and orientation as y = -2x^2 + 2 and vertex (4, -2) -/
structure Parabola where
  shape_coeff : ℝ
  vertex : ℝ × ℝ
  shape_matches : shape_coeff = -2
  vertex_coords : vertex = (4, -2)

/-- The analytical expression of the parabola -/
def parabola_expression (p : Parabola) (x : ℝ) : ℝ :=
  p.shape_coeff * (x - p.vertex.1)^2 + p.vertex.2

theorem parabola_equation (p : Parabola) :
  ∀ x, parabola_expression p x = -2 * (x - 4)^2 - 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3566_356610


namespace NUMINAMATH_CALUDE_orange_distribution_l3566_356622

/-- The number of ways to distribute distinct oranges to sons. -/
def distribute_oranges (num_oranges : ℕ) (num_sons : ℕ) : ℕ :=
  (num_sons.choose num_oranges) * num_oranges.factorial

/-- Theorem: The number of ways to distribute 5 distinct oranges to 8 sons is 6720. -/
theorem orange_distribution :
  distribute_oranges 5 8 = 6720 := by
  sorry

#eval distribute_oranges 5 8

end NUMINAMATH_CALUDE_orange_distribution_l3566_356622


namespace NUMINAMATH_CALUDE_fraction_power_five_l3566_356621

theorem fraction_power_five : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_five_l3566_356621


namespace NUMINAMATH_CALUDE_sum_three_consecutive_terms_l3566_356661

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_three_consecutive_terms
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 21) :
  a 4 + a 5 + a 6 = 63 :=
sorry

end NUMINAMATH_CALUDE_sum_three_consecutive_terms_l3566_356661


namespace NUMINAMATH_CALUDE_f_properties_l3566_356640

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def has_max_value (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ c

-- State the theorem
theorem f_properties (hsym : symmetric_about_origin f)
                     (hinc : increasing_on f 3 7)
                     (hmax : has_max_value f 3 7 5) :
  increasing_on f (-7) (-3) ∧ has_max_value f (-7) (-3) (-5) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3566_356640


namespace NUMINAMATH_CALUDE_range_of_a_l3566_356686

/-- A function that is monotonically increasing on an interval -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

/-- The definition of a hyperbola -/
def IsHyperbola (a : ℝ) : Prop :=
  2 * a^2 - 3 * a - 2 < 0

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (MonotonicallyIncreasing f 1 2) ∧
  (IsHyperbola a) →
  -1/2 < a ∧ a ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3566_356686


namespace NUMINAMATH_CALUDE_recruitment_plans_count_l3566_356626

/-- Represents the daily installation capacity of workers -/
structure WorkerCapacity where
  skilled : ℕ
  new : ℕ

/-- Represents a recruitment plan -/
structure RecruitmentPlan where
  skilled : ℕ
  new : ℕ

/-- Checks if a recruitment plan is valid given the constraints -/
def isValidPlan (plan : RecruitmentPlan) : Prop :=
  1 < plan.skilled ∧ plan.skilled < 8 ∧ 0 < plan.new

/-- Checks if a recruitment plan can complete the task -/
def canCompleteTask (capacity : WorkerCapacity) (plan : RecruitmentPlan) : Prop :=
  15 * (capacity.skilled * plan.skilled + capacity.new * plan.new) = 360

/-- Main theorem to prove -/
theorem recruitment_plans_count 
  (capacity : WorkerCapacity)
  (h1 : 2 * capacity.skilled + capacity.new = 10)
  (h2 : 3 * capacity.skilled + 2 * capacity.new = 16) :
  ∃! (plans : Finset RecruitmentPlan), 
    plans.card = 4 ∧ 
    (∀ plan ∈ plans, isValidPlan plan ∧ canCompleteTask capacity plan) ∧
    (∀ plan, isValidPlan plan ∧ canCompleteTask capacity plan → plan ∈ plans) :=
  sorry

end NUMINAMATH_CALUDE_recruitment_plans_count_l3566_356626


namespace NUMINAMATH_CALUDE_green_bean_to_onion_ratio_l3566_356623

def potato_count : ℕ := 2
def carrot_to_potato_ratio : ℕ := 6
def onion_to_carrot_ratio : ℕ := 2
def green_bean_count : ℕ := 8

def carrot_count : ℕ := potato_count * carrot_to_potato_ratio
def onion_count : ℕ := carrot_count * onion_to_carrot_ratio

theorem green_bean_to_onion_ratio :
  (green_bean_count : ℚ) / onion_count = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_green_bean_to_onion_ratio_l3566_356623


namespace NUMINAMATH_CALUDE_permutation_calculation_l3566_356659

/-- Definition of permutation notation -/
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem stating that A₆² A₄² equals 360 -/
theorem permutation_calculation : A 6 2 * A 4 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_permutation_calculation_l3566_356659


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3566_356619

/-- Given four positive numbers in sequence where the first is 4 and the last is 16,
    with two numbers inserted between them such that the first three form a geometric progression
    and the last three form a harmonic progression, prove that the sum of the inserted numbers is 8. -/
theorem inserted_numbers_sum (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧  -- x and y are positive
  (∃ r : ℝ, r > 0 ∧ x = 4 * r ∧ y = 4 * r^2) ∧  -- geometric progression
  2 / y = 1 / x + 1 / 16 →  -- harmonic progression
  x + y = 8 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3566_356619


namespace NUMINAMATH_CALUDE_sum_lower_bound_l3566_356614

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l3566_356614


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3566_356615

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | x < -2 or x > 3}, prove the following statements -/
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ x < -2 ∨ x > 3) :
  (a > 0) ∧
  (a + b + c < 0) ∧
  (∀ x, c*x^2 - b*x + a < 0 ↔ x < -1/3 ∨ x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3566_356615


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3566_356617

theorem opposite_of_negative_two :
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3566_356617


namespace NUMINAMATH_CALUDE_sequence_decreasing_equivalence_l3566_356667

def IsDecreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) < a n

theorem sequence_decreasing_equivalence (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) ↔ IsDecreasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_decreasing_equivalence_l3566_356667


namespace NUMINAMATH_CALUDE_birthday_1200th_day_l3566_356684

/-- Given a person born on a Monday, their 1200th day of life will fall on a Thursday. -/
theorem birthday_1200th_day : 
  ∀ (birth_day : Nat), 
  birth_day % 7 = 1 →  -- Monday is represented as 1 (1-based indexing for days of week)
  (birth_day + 1200) % 7 = 5  -- Thursday is represented as 5
  := by sorry

end NUMINAMATH_CALUDE_birthday_1200th_day_l3566_356684


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l3566_356697

theorem complex_magnitude_proof : Complex.abs (3/4 + 3*I) = (Real.sqrt 153)/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l3566_356697


namespace NUMINAMATH_CALUDE_harold_marbles_distribution_l3566_356652

def marble_distribution (total_marbles : ℕ) (kept_marbles : ℕ) (num_friends : ℕ) : ℕ :=
  (total_marbles - kept_marbles) / num_friends

theorem harold_marbles_distribution :
  marble_distribution 100 20 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_harold_marbles_distribution_l3566_356652


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3566_356656

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 31) 
  (eq2 : 4 * a + 3 * b = 35) : 
  a + b = 68 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3566_356656


namespace NUMINAMATH_CALUDE_carla_marbles_l3566_356654

/-- The number of marbles Carla bought -/
def marbles_bought (initial final : ℕ) : ℕ := final - initial

/-- Proof that Carla bought 134 marbles -/
theorem carla_marbles : marbles_bought 53 187 = 134 := by
  sorry

end NUMINAMATH_CALUDE_carla_marbles_l3566_356654


namespace NUMINAMATH_CALUDE_race_finish_time_difference_l3566_356699

theorem race_finish_time_difference :
  ∀ (total_runners : ℕ) 
    (fast_runners : ℕ) 
    (slow_runners : ℕ) 
    (fast_time : ℝ) 
    (total_time : ℝ),
  total_runners = fast_runners + slow_runners →
  total_runners = 8 →
  fast_runners = 5 →
  fast_time = 8 →
  total_time = 70 →
  ∃ (slow_time : ℝ),
    total_time = fast_runners * fast_time + slow_runners * slow_time ∧
    slow_time - fast_time = 2 :=
by sorry

end NUMINAMATH_CALUDE_race_finish_time_difference_l3566_356699


namespace NUMINAMATH_CALUDE_least_four_digit_number_with_conditions_l3566_356653

/-- A function that checks if a number has all different digits -/
def hasDifferentDigits (n : ℕ) : Prop := sorry

/-- A function that checks if a number includes the digit 5 -/
def includesFive (n : ℕ) : Prop := sorry

/-- A function that checks if a number is divisible by all of its digits -/
def divisibleByAllDigits (n : ℕ) : Prop := sorry

theorem least_four_digit_number_with_conditions :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    hasDifferentDigits n ∧
    includesFive n ∧
    divisibleByAllDigits n →
    1536 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_number_with_conditions_l3566_356653


namespace NUMINAMATH_CALUDE_glasses_purchase_price_l3566_356648

/-- The purchase price of the glasses in yuan -/
def purchase_price : ℝ := 80

/-- The selling price after the initial increase -/
def increased_price (x : ℝ) : ℝ := 10 * x

/-- The selling price after applying the discount -/
def discounted_price (x : ℝ) : ℝ := 0.5 * increased_price x

/-- The profit made from selling the glasses -/
def profit (x : ℝ) : ℝ := discounted_price x - 20 - x

theorem glasses_purchase_price :
  profit purchase_price = 300 :=
sorry

end NUMINAMATH_CALUDE_glasses_purchase_price_l3566_356648


namespace NUMINAMATH_CALUDE_josh_marbles_remaining_l3566_356624

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

theorem josh_marbles_remaining : initial_marbles - lost_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_remaining_l3566_356624


namespace NUMINAMATH_CALUDE_vertical_distance_traveled_l3566_356657

/-- Calculate the total vertical distance traveled in a week -/
theorem vertical_distance_traveled (story : Nat) (trips_per_day : Nat) (feet_per_story : Nat) (days_in_week : Nat) : 
  story = 5 → trips_per_day = 3 → feet_per_story = 10 → days_in_week = 7 →
  2 * story * feet_per_story * trips_per_day * days_in_week = 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_distance_traveled_l3566_356657


namespace NUMINAMATH_CALUDE_albert_running_laps_l3566_356645

theorem albert_running_laps 
  (total_distance : ℕ) 
  (track_length : ℕ) 
  (laps_run : ℕ) 
  (h1 : total_distance = 99)
  (h2 : track_length = 9)
  (h3 : laps_run = 6) :
  (total_distance / track_length) - laps_run = 5 :=
by
  sorry

#eval (99 / 9) - 6  -- This should output 5

end NUMINAMATH_CALUDE_albert_running_laps_l3566_356645


namespace NUMINAMATH_CALUDE_completing_square_proof_l3566_356678

theorem completing_square_proof (x : ℝ) : 
  (x^2 - 8*x + 5 = 0) ↔ ((x - 4)^2 = 11) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_proof_l3566_356678


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3566_356650

theorem division_remainder_problem (a b q r : ℕ) 
  (h1 : a - b = 1335)
  (h2 : a = 1584)
  (h3 : a = q * b + r)
  (h4 : q = 6)
  (h5 : r < b) :
  r = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3566_356650


namespace NUMINAMATH_CALUDE_two_machines_half_hour_copies_l3566_356605

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.copies_per_minute * minutes

/-- Theorem: Two copy machines working together for half an hour will produce 3300 copies -/
theorem two_machines_half_hour_copies :
  let machine1 : CopyMachine := ⟨35⟩
  let machine2 : CopyMachine := ⟨75⟩
  let half_hour : ℕ := 30
  (copies_made machine1 half_hour) + (copies_made machine2 half_hour) = 3300 :=
by
  sorry

end NUMINAMATH_CALUDE_two_machines_half_hour_copies_l3566_356605


namespace NUMINAMATH_CALUDE_top_books_sold_l3566_356696

/-- The number of "TOP" books sold last week -/
def top_books : ℕ := 13

/-- The price of a "TOP" book in dollars -/
def top_price : ℕ := 8

/-- The price of an "ABC" book in dollars -/
def abc_price : ℕ := 23

/-- The number of "ABC" books sold last week -/
def abc_books : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books in dollars -/
def earnings_difference : ℕ := 12

theorem top_books_sold : 
  top_books * top_price - abc_books * abc_price = earnings_difference := by
  sorry

end NUMINAMATH_CALUDE_top_books_sold_l3566_356696


namespace NUMINAMATH_CALUDE_fifteenth_digit_of_sum_l3566_356607

def decimal_rep_1_9 : ℚ := 1 / 9
def decimal_rep_1_11 : ℚ := 1 / 11

def sum_of_reps : ℚ := decimal_rep_1_9 + decimal_rep_1_11

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_of_sum :
  nth_digit_after_decimal sum_of_reps 15 = 1 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_of_sum_l3566_356607


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_term_l3566_356688

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the exponent of x in the general term
def exponent (r : ℕ) : ℚ := 9 - (3 / 2) * r

theorem coefficient_of_x_cubed_term :
  ∃ (r : ℕ), exponent r = 3 ∧ binomial 9 r = 126 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_term_l3566_356688


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3566_356625

theorem solution_set_inequality (x : ℝ) : 
  (((1 - x) / (x + 1) ≤ 0) ↔ (x ∈ Set.Iic (-1) ∪ Set.Ici 1)) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3566_356625


namespace NUMINAMATH_CALUDE_smallest_positive_root_l3566_356612

theorem smallest_positive_root (b c d : ℝ) (hb : |b| ≤ 1) (hc : |c| ≤ 3) (hd : |d| ≤ 2) :
  ∃ (s : ℝ), s > 0 ∧ s^3 + b*s^2 + c*s + d = 0 ∧
  ∀ (x : ℝ), x > 0 ∧ x^3 + b*x^2 + c*x + d = 0 → s ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_l3566_356612


namespace NUMINAMATH_CALUDE_triangle_problem_l3566_356692

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : Real :=
  v.x * w.x + v.y * w.y

variable (ABC : Triangle)

/-- Vector m as defined in the problem -/
def m : Vector2D :=
  { x := Real.cos (ABC.A - ABC.B),
    y := Real.sin (ABC.A - ABC.B) }

/-- Vector n as defined in the problem -/
def n : Vector2D :=
  { x := Real.cos ABC.B,
    y := -Real.sin ABC.B }

/-- Main theorem capturing the problem statement and its solution -/
theorem triangle_problem (h1 : dot_product (m ABC) (n ABC) = -3/5)
                         (h2 : ABC.a = 4 * Real.sqrt 2)
                         (h3 : ABC.b = 5) :
  Real.sin ABC.A = 4/5 ∧
  ABC.B = π/4 ∧
  -(ABC.c * Real.cos ABC.B) = -Real.sqrt 2 / 2 :=
sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l3566_356692


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3566_356609

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 360 →
  a + b + c + d ≤ 66 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3566_356609


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l3566_356693

/-- Given vectors AB, BC, and AD, prove that if A, C, and D are collinear, then m = -2/3 -/
theorem collinear_points_m_value 
  (AB BC AD : ℝ × ℝ)
  (h1 : AB = (7, 6))
  (h2 : BC = (-3, m))
  (h3 : AD = (-1, 2*m))
  (h4 : ∃ k : ℝ, k ≠ 0 ∧ AB + BC = k • AD) :
  m = -2/3 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l3566_356693


namespace NUMINAMATH_CALUDE_sin_neg_pi_l3566_356647

theorem sin_neg_pi : Real.sin (-π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_pi_l3566_356647


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l3566_356679

/-- The number of individuals to be removed from a population to make it divisible by a given sample size -/
def individualsToRemove (populationSize sampleSize : ℕ) : ℕ :=
  populationSize - sampleSize * (populationSize / sampleSize)

/-- Theorem stating that 4 individuals should be removed from a population of 3,204 to make it divisible by 80 -/
theorem systematic_sampling_removal :
  individualsToRemove 3204 80 = 4 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l3566_356679
