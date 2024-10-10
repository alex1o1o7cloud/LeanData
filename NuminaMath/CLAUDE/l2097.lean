import Mathlib

namespace even_function_with_range_l2097_209704

/-- Given a function f(x) = (x + a)(bx - a) where a and b are real constants,
    if f is an even function and its range is [-4, +∞),
    then f(x) = x^2 - 4 -/
theorem even_function_with_range (a b : ℝ) :
  (∀ x, (x + a) * (b * x - a) = ((-(x : ℝ)) + a) * (b * (-x) - a)) →
  (∀ y ≥ -4, ∃ x, (x + a) * (b * x - a) = y) →
  (∀ x, (x + a) * (b * x - a) = x^2 - 4) :=
by sorry

end even_function_with_range_l2097_209704


namespace fourth_quadrant_trig_simplification_l2097_209736

/-- For an angle α in the fourth quadrant, 
    cos α √((1 - sin α) / (1 + sin α)) + sin α √((1 - cos α) / (1 + cos α)) = cos α - sin α -/
theorem fourth_quadrant_trig_simplification (α : Real) 
  (h_fourth_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  Real.cos α - Real.sin α := by
sorry

end fourth_quadrant_trig_simplification_l2097_209736


namespace no_mn_divisibility_l2097_209706

theorem no_mn_divisibility : ¬∃ (m n : ℕ+), 
  (m.val * n.val ∣ 3^m.val + 1) ∧ (m.val * n.val ∣ 3^n.val + 1) := by
  sorry

end no_mn_divisibility_l2097_209706


namespace best_method_for_pedestrian_phone_use_data_l2097_209735

/-- Represents a data collection method -/
structure DataCollectionMethod where
  name : String
  target_group : String
  is_random : Bool

/-- Represents the characteristics of a good data collection method -/
structure MethodCharacteristics where
  is_representative : Bool
  is_extensive : Bool

/-- Defines the criteria for evaluating a data collection method -/
def evaluate_method (method : DataCollectionMethod) : MethodCharacteristics :=
  { is_representative := method.is_random && method.target_group = "pedestrians on roadside",
    is_extensive := method.is_random && method.target_group = "pedestrians on roadside" }

/-- The theorem stating that randomly distributing questionnaires to pedestrians on the roadside
    is the most representative and extensive method for collecting data on pedestrians
    walking on the roadside while looking down at their phones -/
theorem best_method_for_pedestrian_phone_use_data :
  let method := { name := "Random questionnaires to roadside pedestrians",
                  target_group := "pedestrians on roadside",
                  is_random := true : DataCollectionMethod }
  let evaluation := evaluate_method method
  evaluation.is_representative ∧ evaluation.is_extensive :=
by
  sorry


end best_method_for_pedestrian_phone_use_data_l2097_209735


namespace negation_of_existence_proposition_l2097_209770

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end negation_of_existence_proposition_l2097_209770


namespace system_solution_l2097_209799

theorem system_solution (x y b : ℚ) : 
  5 * x - 2 * y = b →
  3 * x + 4 * y = 3 * b →
  y = 3 →
  b = 13 / 2 := by
sorry

end system_solution_l2097_209799


namespace square_root_identity_polynomial_identity_square_root_polynomial_l2097_209712

theorem square_root_identity (n : ℕ) : 
  Real.sqrt ((n - 1) * (n + 1) + 1) = n :=
sorry

theorem polynomial_identity (n : ℕ) : 
  (n * (n + 3) + 1)^2 = n * (n + 1) * (n + 2) * (n + 3) + 1 :=
sorry

theorem square_root_polynomial (n : ℕ) : 
  Real.sqrt (n * (n + 1) * (n + 2) * (n + 3) + 1) = n * (n + 3) :=
sorry

end square_root_identity_polynomial_identity_square_root_polynomial_l2097_209712


namespace min_value_part1_min_value_part2_l2097_209724

-- Part 1
theorem min_value_part1 (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) + 6 ≥ 9 ∧
  (x + 4 / (x + 1) + 6 = 9 ↔ x = 1) :=
sorry

-- Part 2
theorem min_value_part2 (x : ℝ) (h : x > 1) :
  (x^2 + 8) / (x - 1) ≥ 8 ∧
  ((x^2 + 8) / (x - 1) = 8 ↔ x = 4) :=
sorry

end min_value_part1_min_value_part2_l2097_209724


namespace three_sqrt_two_bounds_l2097_209727

theorem three_sqrt_two_bounds : 4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 := by
  sorry

end three_sqrt_two_bounds_l2097_209727


namespace sum_of_angles_l2097_209720

-- Define the angles as real numbers
variable (A B C D F G EDC ECD : ℝ)

-- Define the conditions
variable (h1 : A + B + C + D = 360) -- ABCD is a quadrilateral
variable (h2 : G + F = EDC + ECD)   -- Given condition

-- Theorem statement
theorem sum_of_angles : A + B + C + D + F + G = 360 := by
  sorry

end sum_of_angles_l2097_209720


namespace motorbike_distance_theorem_l2097_209732

/-- Given two motorbikes traveling the same distance, with speeds of 60 km/h and 64 km/h
    respectively, and the slower bike taking 1 hour more than the faster bike,
    prove that the distance traveled is 960 kilometers. -/
theorem motorbike_distance_theorem (distance : ℝ) (time_slower : ℝ) (time_faster : ℝ) :
  (60 * time_slower = distance) →
  (64 * time_faster = distance) →
  (time_slower = time_faster + 1) →
  distance = 960 := by
  sorry

end motorbike_distance_theorem_l2097_209732


namespace cos_four_theta_value_l2097_209769

theorem cos_four_theta_value (θ : ℝ) 
  (h : ∑' n, (Real.cos θ)^(2*n) = 9/2) : 
  Real.cos (4*θ) = -31/81 := by
  sorry

end cos_four_theta_value_l2097_209769


namespace question_one_l2097_209729

theorem question_one (a : ℝ) (h : a^2 + a = 3) : 2*a^2 + 2*a + 2023 = 2029 := by
  sorry


end question_one_l2097_209729


namespace inequality_proof_l2097_209753

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by
  sorry

end inequality_proof_l2097_209753


namespace volleyball_tournament_l2097_209719

theorem volleyball_tournament (n : ℕ) : n > 0 → 2 * (n.choose 2) = 56 → n = 8 := by
  sorry

end volleyball_tournament_l2097_209719


namespace marias_quarters_l2097_209798

/-- Represents the number of coins of each type in Maria's piggy bank -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Calculates the total value in dollars given a CoinCount -/
def totalValue (coins : CoinCount) : ℚ :=
  0.1 * coins.dimes + 0.25 * coins.quarters + 0.05 * coins.nickels

/-- The problem statement -/
theorem marias_quarters (initialCoins : CoinCount) (finalTotal : ℚ) : 
  initialCoins.dimes = 4 → 
  initialCoins.quarters = 4 → 
  initialCoins.nickels = 7 → 
  finalTotal = 3 →
  ∃ (addedQuarters : ℕ), 
    totalValue { dimes := initialCoins.dimes,
                 quarters := initialCoins.quarters + addedQuarters,
                 nickels := initialCoins.nickels } = finalTotal ∧
    addedQuarters = 5 := by
  sorry


end marias_quarters_l2097_209798


namespace total_molecular_weight_l2097_209716

-- Define atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Fe : ℝ := 55.845
def atomic_weight_S : ℝ := 32.07
def atomic_weight_Mn : ℝ := 54.938

-- Define molecular weights
def molecular_weight_K2Cr2O7 : ℝ :=
  2 * atomic_weight_K + 2 * atomic_weight_Cr + 7 * atomic_weight_O

def molecular_weight_Fe2SO43 : ℝ :=
  2 * atomic_weight_Fe + 3 * (atomic_weight_S + 4 * atomic_weight_O)

def molecular_weight_KMnO4 : ℝ :=
  atomic_weight_K + atomic_weight_Mn + 4 * atomic_weight_O

-- Define the theorem
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 +
  3 * molecular_weight_Fe2SO43 +
  5 * molecular_weight_KMnO4 = 3166.658 :=
by sorry

end total_molecular_weight_l2097_209716


namespace inequality_solutions_l2097_209771

theorem inequality_solutions :
  (∀ x : ℝ, (|x + 1| / |x + 2| ≥ 1) ↔ (x ≤ -3/2 ∧ x ≠ -2)) ∧
  (∀ a x : ℝ,
    (a * (x - 1) / (x - 2) > 1) ↔
    ((a > 1 ∧ (x > 2 ∨ x < (a - 2) / (a - 1))) ∨
     (a = 1 ∧ x > 2) ∨
     (0 < a ∧ a < 1 ∧ 2 < x ∧ x < (a - 2) / (a - 1)) ∨
     (a < 0 ∧ (a - 2) / (a - 1) < x ∧ x < 2))) ∧
  (∀ x : ℝ, ¬(0 * (x - 1) / (x - 2) > 1)) :=
by sorry

end inequality_solutions_l2097_209771


namespace seventh_term_of_geometric_sequence_l2097_209709

/-- Given a geometric sequence with first term 5 and second term 1/5, 
    the seventh term of the sequence is 1/48828125. -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℚ := 5
  let a₂ : ℚ := 1/5
  let r : ℚ := a₂ / a₁
  let n : ℕ := 7
  let a_n : ℚ := a₁ * r^(n-1)
  a_n = 1/48828125 := by sorry

end seventh_term_of_geometric_sequence_l2097_209709


namespace cost_price_calculation_l2097_209790

/-- 
Given a selling price and a profit percentage, calculate the cost price.
-/
theorem cost_price_calculation 
  (selling_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : selling_price = 1800) 
  (h2 : profit_percentage = 20) :
  selling_price / (1 + profit_percentage / 100) = 1500 :=
by sorry

end cost_price_calculation_l2097_209790


namespace wire_cutting_l2097_209765

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (longer_part : ℝ) : 
  total_length = 180 ∧ difference = 32 → longer_part = 106 :=
by
  sorry

end wire_cutting_l2097_209765


namespace vector_subtraction_l2097_209774

/-- Given two planar vectors a and b, prove that a - 2b equals the expected result. -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (5, 3)) (h2 : b = (1, -2)) :
  a - 2 • b = (3, 7) := by
  sorry

end vector_subtraction_l2097_209774


namespace simplified_fourth_root_sum_l2097_209780

theorem simplified_fourth_root_sum (a b : ℕ+) :
  (2^6 * 5^2 : ℝ)^(1/4) = a * b^(1/4) → a + b = 102 := by
  sorry

end simplified_fourth_root_sum_l2097_209780


namespace no_solution_exists_l2097_209725

theorem no_solution_exists : ¬∃ (a b c d : ℕ), 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧ 
  71 * a + 72 * b + 73 * c + 74 * d = 2014 :=
sorry

end no_solution_exists_l2097_209725


namespace remainder_problem_l2097_209734

theorem remainder_problem (d : ℤ) (r : ℤ) 
  (h1 : d > 1)
  (h2 : 1250 % d = r)
  (h3 : 1890 % d = r)
  (h4 : 2500 % d = r) :
  d - r = 10 := by
sorry

end remainder_problem_l2097_209734


namespace tangent_line_right_triangle_l2097_209701

/-- Given a line ax + by + c = 0 (a, b, c ≠ 0) tangent to the circle x² + y² = 1,
    the triangle with side lengths |a|, |b|, and |c| is a right triangle. -/
theorem tangent_line_right_triangle (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
    (h_tangent : ∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 = 1) :
  a^2 + b^2 = c^2 := by
sorry

end tangent_line_right_triangle_l2097_209701


namespace geometric_sequence_common_ratio_l2097_209788

theorem geometric_sequence_common_ratio 
  (a : ℝ) (term2 term3 term4 : ℝ) :
  a = 12 ∧ 
  term2 = -18 ∧ 
  term3 = 27 ∧ 
  term4 = -40.5 ∧ 
  term2 = a * r ∧ 
  term3 = a * r^2 ∧ 
  term4 = a * r^3 →
  r = -3/2 :=
by sorry

end geometric_sequence_common_ratio_l2097_209788


namespace total_tires_is_101_l2097_209741

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of tires on a bicycle -/
def bicycle_tires : ℕ := 2

/-- The number of tires on a pickup truck -/
def pickup_truck_tires : ℕ := 4

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of cars Juan saw -/
def cars_seen : ℕ := 15

/-- The number of bicycles Juan saw -/
def bicycles_seen : ℕ := 3

/-- The number of pickup trucks Juan saw -/
def pickup_trucks_seen : ℕ := 8

/-- The number of tricycles Juan saw -/
def tricycles_seen : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 
  cars_seen * car_tires + 
  bicycles_seen * bicycle_tires + 
  pickup_trucks_seen * pickup_truck_tires + 
  tricycles_seen * tricycle_tires

theorem total_tires_is_101 : total_tires = 101 := by
  sorry

end total_tires_is_101_l2097_209741


namespace rectangle_area_from_perimeter_and_diagonal_l2097_209760

/-- The area of a rectangle given its perimeter and diagonal -/
theorem rectangle_area_from_perimeter_and_diagonal (p d : ℝ) (hp : p > 0) (hd : d > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * (x + y) = p ∧ x^2 + y^2 = d^2 ∧
  x * y = (p^2 - 4 * d^2) / 8 := by
  sorry

#check rectangle_area_from_perimeter_and_diagonal

end rectangle_area_from_perimeter_and_diagonal_l2097_209760


namespace pizza_combinations_l2097_209728

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  1 + n + n.choose 2 = 37 := by
  sorry

end pizza_combinations_l2097_209728


namespace complex_power_approximation_l2097_209742

/-- The complex number (2 + i)/(2 - i) raised to the power of 600 is approximately equal to -0.982 - 0.189i -/
theorem complex_power_approximation :
  let z : ℂ := (2 + Complex.I) / (2 - Complex.I)
  ∃ (ε : ℝ) (hε : ε > 0), Complex.abs (z^600 - (-0.982 - 0.189 * Complex.I)) < ε :=
by sorry

end complex_power_approximation_l2097_209742


namespace equation_equivalence_l2097_209745

theorem equation_equivalence :
  ∃ (m n p : ℤ), ∀ (a b x y : ℝ),
    (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
    ((a^m*x - a^n) * (a^p*y - a^3) = a^5*b^5) :=
by sorry

end equation_equivalence_l2097_209745


namespace complex_quadratic_modulus_l2097_209751

theorem complex_quadratic_modulus (z : ℂ) : z^2 - 8*z + 40 = 0 → Complex.abs z = 2 * Real.sqrt 10 := by
  sorry

end complex_quadratic_modulus_l2097_209751


namespace certain_number_divisor_of_factorial_l2097_209785

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem certain_number_divisor_of_factorial :
  ∃! (n : ℕ), n > 0 ∧ (factorial 15) % (n^6) = 0 ∧ (factorial 15) % (n^7) ≠ 0 :=
by sorry

end certain_number_divisor_of_factorial_l2097_209785


namespace units_digit_of_8129_power_1351_l2097_209776

theorem units_digit_of_8129_power_1351 : 8129^1351 % 10 = 9 := by
  sorry

end units_digit_of_8129_power_1351_l2097_209776


namespace cube_roots_of_unity_sum_l2097_209713

theorem cube_roots_of_unity_sum (i : ℂ) :
  i^2 = -1 →
  let x : ℂ := (-1 + i * Real.sqrt 3) / 2
  let y : ℂ := (-1 - i * Real.sqrt 3) / 2
  x^6 + y^6 = 2 := by sorry

end cube_roots_of_unity_sum_l2097_209713


namespace sum_of_fractions_l2097_209793

theorem sum_of_fractions : (1 : ℚ) / 12 + (1 : ℚ) / 15 = (3 : ℚ) / 20 := by
  sorry

end sum_of_fractions_l2097_209793


namespace tangent_slope_angle_l2097_209775

noncomputable def f (x : ℝ) : ℝ := -(Real.sqrt 3 / 3) * x^3 + 2

theorem tangent_slope_angle (x : ℝ) : 
  let f' := deriv f
  let slope := f' 1
  let angle := Real.arctan (-slope)
  x = 1 → angle = 2 * Real.pi / 3 := by sorry

end tangent_slope_angle_l2097_209775


namespace calculate_fraction_product_l2097_209718

theorem calculate_fraction_product : 
  let mixed_number : ℚ := 3 + 3/4
  let decimal_one : ℚ := 0.2
  let whole_number : ℕ := 135
  let decimal_two : ℚ := 5.4
  ((mixed_number * decimal_one) / whole_number) * decimal_two = 0.03 := by
sorry

end calculate_fraction_product_l2097_209718


namespace total_buses_is_816_l2097_209777

/-- Represents the bus schedule for different types of days -/
structure BusSchedule where
  weekday : Nat
  saturday : Nat
  sunday_holiday : Nat

/-- Calculates the total number of buses in a month -/
def total_buses_in_month (schedule : BusSchedule) (public_holidays : Nat) : Nat :=
  let weekdays := 20 - public_holidays
  let saturdays := 4
  let sundays_holidays := 4 + public_holidays
  weekdays * schedule.weekday + saturdays * schedule.saturday + sundays_holidays * schedule.sunday_holiday

/-- The bus schedule for the given problem -/
def problem_schedule : BusSchedule :=
  { weekday := 36
  , saturday := 24
  , sunday_holiday := 12 }

/-- Theorem stating that the total number of buses in the month is 816 -/
theorem total_buses_is_816 :
  total_buses_in_month problem_schedule 2 = 816 := by
  sorry

end total_buses_is_816_l2097_209777


namespace difference_of_products_l2097_209707

theorem difference_of_products : 20132014 * 20142013 - 20132013 * 20142014 = 10000 := by
  sorry

end difference_of_products_l2097_209707


namespace solve_travel_problem_l2097_209787

def travel_problem (train_distance : ℝ) : Prop :=
  let bus_distance := train_distance / 2
  let cab_distance := bus_distance / 3
  let total_distance := train_distance + bus_distance + cab_distance
  (train_distance = 300) → (total_distance = 500)

theorem solve_travel_problem : travel_problem 300 := by
  sorry

end solve_travel_problem_l2097_209787


namespace choose_four_from_fifteen_l2097_209783

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end choose_four_from_fifteen_l2097_209783


namespace part1_part2_part3_l2097_209715

-- Define the system of equations
def system (x y m : ℝ) : Prop :=
  2 * x + y = 4 * m ∧ x + 2 * y = 2 * m + 1

-- Part 1
theorem part1 (x y m : ℝ) :
  system x y m → x + y = 1 → m = 1/3 := by sorry

-- Part 2
theorem part2 (x y m : ℝ) :
  system x y m → -1 ≤ x - y ∧ x - y ≤ 5 → 0 ≤ m ∧ m ≤ 3 := by sorry

-- Part 3
theorem part3 (m : ℝ) :
  0 ≤ m ∧ m ≤ 3 →
  (0 ≤ m ∧ m ≤ 3/2 → |m+2| + |2*m-3| = 5 - m) ∧
  (3/2 < m ∧ m ≤ 3 → |m+2| + |2*m-3| = 3*m - 1) := by sorry

end part1_part2_part3_l2097_209715


namespace trig_expression_value_cos_2α_minus_π_4_l2097_209752

/- For the first problem -/
theorem trig_expression_value (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 := by
  sorry

/- For the second problem -/
theorem cos_2α_minus_π_4 (α : Real) (h1 : Real.sin α + Real.cos α = 1/5) (h2 : 0 ≤ α ∧ α ≤ π) :
  Real.cos (2*α - π/4) = -31*Real.sqrt 2/50 := by
  sorry

end trig_expression_value_cos_2α_minus_π_4_l2097_209752


namespace calculate_expression_l2097_209796

theorem calculate_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^4 + 3*x^2 - 2*y + 2*y^2) / 6 = 22 := by
  sorry

end calculate_expression_l2097_209796


namespace square_root_range_l2097_209717

theorem square_root_range (x : ℝ) : x - 2 ≥ 0 ↔ x ≥ 2 := by
  sorry

end square_root_range_l2097_209717


namespace school_distance_l2097_209762

/-- The distance between a girl's house and school, given her travel speeds and total round trip time. -/
theorem school_distance (speed_to_school speed_from_school : ℝ) (total_time : ℝ) : 
  speed_to_school = 6 →
  speed_from_school = 4 →
  total_time = 10 →
  (1 / speed_to_school + 1 / speed_from_school) * (speed_to_school * speed_from_school / (speed_to_school + speed_from_school)) = 24 := by
  sorry

end school_distance_l2097_209762


namespace remainder_sum_l2097_209768

theorem remainder_sum (a b c : ℤ) 
  (ha : a % 90 = 84) 
  (hb : b % 120 = 114) 
  (hc : c % 150 = 144) : 
  (a + b + c) % 30 = 12 := by
sorry

end remainder_sum_l2097_209768


namespace square_root_problem_l2097_209743

theorem square_root_problem (a x : ℝ) 
  (h1 : Real.sqrt a = x + 3) 
  (h2 : Real.sqrt a = 3 * x - 11) : 
  2 * a - 1 = 199 := by
sorry

end square_root_problem_l2097_209743


namespace seven_correct_guesses_l2097_209705

/-- A guess is either a lower bound (not less than) or an upper bound (not more than) -/
inductive Guess
  | LowerBound (n : Nat)
  | UpperBound (n : Nat)

/-- The set of guesses made by the teachers -/
def teacherGuesses : List Guess := [
  Guess.LowerBound 1, Guess.UpperBound 2,
  Guess.LowerBound 3, Guess.UpperBound 4,
  Guess.LowerBound 5, Guess.UpperBound 6,
  Guess.LowerBound 7, Guess.UpperBound 8,
  Guess.LowerBound 9, Guess.UpperBound 10,
  Guess.LowerBound 11, Guess.UpperBound 12
]

/-- A guess is correct if it's satisfied by the given number -/
def isCorrectGuess (x : Nat) (g : Guess) : Bool :=
  match g with
  | Guess.LowerBound n => x ≥ n
  | Guess.UpperBound n => x ≤ n

/-- The number of correct guesses for a given number -/
def correctGuessCount (x : Nat) : Nat :=
  (teacherGuesses.filter (isCorrectGuess x)).length

/-- There exists a number for which exactly 7 guesses are correct -/
theorem seven_correct_guesses : ∃ x, correctGuessCount x = 7 := by
  sorry

end seven_correct_guesses_l2097_209705


namespace complex_number_in_first_quadrant_l2097_209744

theorem complex_number_in_first_quadrant :
  let z : ℂ := (2 + I) / (1 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end complex_number_in_first_quadrant_l2097_209744


namespace order_of_logarithmic_fractions_l2097_209703

theorem order_of_logarithmic_fractions :
  let a : ℝ := (Real.log 2) / 2
  let b : ℝ := (Real.log 3) / 3
  let c : ℝ := (Real.log 5) / 5
  b > a ∧ a > c := by sorry

end order_of_logarithmic_fractions_l2097_209703


namespace compute_a_l2097_209730

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 48

-- State the theorem
theorem compute_a : 
  ∃ (a b : ℚ), f a b (-1 - 5 * Real.sqrt 3) = 0 ∧ a = 50/37 := by
  sorry

end compute_a_l2097_209730


namespace probability_seven_odd_in_ten_rolls_l2097_209755

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of rolls -/
def num_rolls : ℕ := 10

/-- The number of desired odd rolls -/
def desired_odd_rolls : ℕ := 7

/-- The probability of getting exactly 7 odd numbers in 10 rolls of a fair 6-sided die -/
theorem probability_seven_odd_in_ten_rolls :
  Nat.choose num_rolls desired_odd_rolls * prob_odd ^ desired_odd_rolls * (1 - prob_odd) ^ (num_rolls - desired_odd_rolls) = 15/128 := by
  sorry

end probability_seven_odd_in_ten_rolls_l2097_209755


namespace biased_coin_probability_l2097_209721

theorem biased_coin_probability : ∀ h : ℝ,
  0 < h ∧ h < 1 →
  (Nat.choose 6 2 : ℝ) * h^2 * (1 - h)^4 = (Nat.choose 6 3 : ℝ) * h^3 * (1 - h)^3 →
  (Nat.choose 6 4 : ℝ) * h^4 * (1 - h)^2 = 19440 / 117649 := by
  sorry

end biased_coin_probability_l2097_209721


namespace sum_divided_non_negative_l2097_209786

theorem sum_divided_non_negative (x : ℝ) :
  ((x + 6) / 2 ≥ 0) ↔ (∃ y ≥ 0, y = (x + 6) / 2) :=
by sorry

end sum_divided_non_negative_l2097_209786


namespace largest_810_triple_l2097_209778

/-- Converts a base-10 number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 8) :: aux (m / 8)
  aux n |>.reverse

/-- Converts a list of digits to its base-10 representation -/
def fromDigits (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

/-- Checks if a number is an 8-10 triple -/
def is810Triple (n : ℕ) : Prop :=
  fromDigits (toBase8 n) = 3 * n

/-- Statement: 273 is the largest 8-10 triple -/
theorem largest_810_triple : 
  (∀ m : ℕ, m > 273 → ¬ is810Triple m) ∧ is810Triple 273 :=
sorry

end largest_810_triple_l2097_209778


namespace parabola_latus_rectum_l2097_209789

/-- 
For a parabola with equation x^2 = ay and latus rectum y = 2, 
the value of a is -8.
-/
theorem parabola_latus_rectum (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- equation of parabola
  (∃ x : ℝ, x^2 = 2*a) →    -- latus rectum condition
  a = -8 := by
sorry

end parabola_latus_rectum_l2097_209789


namespace complex_fraction_simplification_l2097_209737

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i + 2) / i = 1 - 2 * i := by
sorry

end complex_fraction_simplification_l2097_209737


namespace three_valid_floor_dimensions_l2097_209756

/-- 
Represents the number of valid floor dimensions (m, n) satisfying:
1. n > m
2. (m-6)(n-6) = 12
3. m ≥ 7 and n ≥ 7
where m and n are positive integers, and the unpainted border is 2 feet wide on each side.
-/
def validFloorDimensions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let m := p.1
    let n := p.2
    n > m ∧ (m - 6) * (n - 6) = 12 ∧ m ≥ 7 ∧ n ≥ 7
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The main theorem stating that there are exactly 3 valid floor dimensions. -/
theorem three_valid_floor_dimensions : validFloorDimensions = 3 := by
  sorry

end three_valid_floor_dimensions_l2097_209756


namespace bottle_ratio_is_half_l2097_209767

/-- Represents the distribution of bottles in a delivery van -/
structure BottleDistribution where
  total : ℕ
  cider : ℕ
  beer : ℕ
  mixed : ℕ
  first_house : ℕ

/-- The ratio of bottles given to the first house to the total number of bottles -/
def bottle_ratio (d : BottleDistribution) : ℚ :=
  d.first_house / d.total

/-- Theorem stating the ratio of bottles given to the first house to the total number of bottles -/
theorem bottle_ratio_is_half (d : BottleDistribution) 
    (h1 : d.total = 180)
    (h2 : d.cider = 40)
    (h3 : d.beer = 80)
    (h4 : d.mixed = d.total - d.cider - d.beer)
    (h5 : d.first_house = 90) : 
  bottle_ratio d = 1/2 := by
  sorry

#eval bottle_ratio { total := 180, cider := 40, beer := 80, mixed := 60, first_house := 90 }

end bottle_ratio_is_half_l2097_209767


namespace ball_max_height_l2097_209733

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 - 40 * t + 50

-- State the theorem
theorem ball_max_height :
  ∃ (max : ℝ), max = 70 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end ball_max_height_l2097_209733


namespace expression_evaluation_l2097_209766

theorem expression_evaluation :
  let a : ℚ := -1
  let b : ℚ := 1/7
  (3*a^3 - 2*a*b + b^2) - 2*(-a^3 - a*b + 4*b^2) = -5 - 1/7 :=
by sorry

end expression_evaluation_l2097_209766


namespace complex_power_magnitude_l2097_209738

theorem complex_power_magnitude : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 2) ^ 6) = 1728 := by
  sorry

end complex_power_magnitude_l2097_209738


namespace cubic_derivative_value_l2097_209740

theorem cubic_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (deriv f) x₀ = 3 →
  x₀ = 1 ∨ x₀ = -1 := by
sorry

end cubic_derivative_value_l2097_209740


namespace triangle_area_is_eight_l2097_209759

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A triangle defined by three lines -/
structure Triangle where
  line1 : Line
  line2 : Line
  line3 : Line

/-- Calculate the area of a triangle given its three bounding lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { line1 := { slope := 2, intercept := 0 }
  , line2 := { slope := -2, intercept := 0 }
  , line3 := { slope := 0, intercept := 4 }
  }

theorem triangle_area_is_eight :
  triangleArea problemTriangle = 8 := by
  sorry

end triangle_area_is_eight_l2097_209759


namespace optimal_price_theorem_l2097_209748

-- Define the problem parameters
def initial_price : ℝ := 60
def initial_sales : ℝ := 300
def cost_price : ℝ := 40
def target_profit : ℝ := 6080
def price_sales_ratio : ℝ := 20

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - cost_price) * (initial_sales + price_sales_ratio * (initial_price - price))

-- State the theorem
theorem optimal_price_theorem :
  ∃ (optimal_price : ℝ),
    profit optimal_price = target_profit ∧
    optimal_price < initial_price ∧
    ∀ (p : ℝ), p < optimal_price → profit p < target_profit :=
by
  -- The proof goes here
  sorry

end optimal_price_theorem_l2097_209748


namespace sphere_volume_equals_surface_area_l2097_209700

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 :=
by
  sorry

end sphere_volume_equals_surface_area_l2097_209700


namespace least_addition_for_divisibility_l2097_209782

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m < n → ¬(23 ∣ (4499 * 17 + m))) ∧
  (23 ∣ (4499 * 17 + n)) := by
  sorry

end least_addition_for_divisibility_l2097_209782


namespace base8_4532_equals_2394_l2097_209714

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the base 8 number 4532
def base8Number : List Nat := [2, 3, 5, 4]

-- Theorem statement
theorem base8_4532_equals_2394 :
  base8ToBase10 base8Number = 2394 := by
  sorry

end base8_4532_equals_2394_l2097_209714


namespace min_value_abs_sum_l2097_209763

theorem min_value_abs_sum (x : ℝ) : 
  |x + 1| + |x - 2| + |x - 3| ≥ 4 ∧ ∃ y : ℝ, |y + 1| + |y - 2| + |y - 3| = 4 := by
  sorry

end min_value_abs_sum_l2097_209763


namespace blue_balls_count_l2097_209739

theorem blue_balls_count (black_balls : ℕ) (blue_balls : ℕ) : 
  (black_balls : ℚ) / blue_balls = 5 / 3 → 
  black_balls = 15 → 
  blue_balls = 9 := by
sorry

end blue_balls_count_l2097_209739


namespace chess_draw_probability_l2097_209731

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.4)
  (h_not_lose : p_not_lose = 0.9) :
  p_not_lose - p_win = 0.5 := by
  sorry

end chess_draw_probability_l2097_209731


namespace gcd_of_squares_l2097_209792

theorem gcd_of_squares : Nat.gcd (121^2 + 233^2 + 345^2) (120^2 + 232^2 + 346^2) = 5 := by
  sorry

end gcd_of_squares_l2097_209792


namespace sin_cos_shift_l2097_209702

/-- Given two functions f and g defined on real numbers,
    prove that they are equivalent up to a horizontal shift. -/
theorem sin_cos_shift (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x + π / 3)
  let g : ℝ → ℝ := λ x ↦ Real.cos (2 * x)
  f x = g (x - π / 12) := by
  sorry


end sin_cos_shift_l2097_209702


namespace michael_work_time_l2097_209723

/-- Given that:
    - Michael and Adam can complete a work together in 20 days
    - They work together for 18 days, then Michael stops
    - Adam completes the remaining work in 10 days
    Prove that Michael can complete the work separately in 25 days -/
theorem michael_work_time (total_work : ℝ) (michael_rate : ℝ) (adam_rate : ℝ)
  (h1 : michael_rate + adam_rate = total_work / 20)
  (h2 : 18 * (michael_rate + adam_rate) = 9 / 10 * total_work)
  (h3 : adam_rate = total_work / 100) :
  michael_rate = total_work / 25 := by
  sorry

end michael_work_time_l2097_209723


namespace systematic_sampling_methods_l2097_209764

/-- Represents a sampling method -/
inductive SamplingMethod
  | BallSelection
  | ProductInspection
  | MarketSurvey
  | CinemaAudienceSurvey

/-- Defines the characteristics of systematic sampling -/
def is_systematic_sampling (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.BallSelection => true
  | SamplingMethod.ProductInspection => true
  | SamplingMethod.MarketSurvey => false
  | SamplingMethod.CinemaAudienceSurvey => true

/-- Theorem stating which sampling methods are systematic -/
theorem systematic_sampling_methods :
  (is_systematic_sampling SamplingMethod.BallSelection) ∧
  (is_systematic_sampling SamplingMethod.ProductInspection) ∧
  (¬is_systematic_sampling SamplingMethod.MarketSurvey) ∧
  (is_systematic_sampling SamplingMethod.CinemaAudienceSurvey) :=
by sorry

end systematic_sampling_methods_l2097_209764


namespace jesse_carpet_need_l2097_209754

/-- The additional carpet needed for Jesse's room -/
def additional_carpet_needed (room_length : ℝ) (room_width : ℝ) (existing_carpet : ℝ) : ℝ :=
  room_length * room_width - existing_carpet

/-- Theorem stating the additional carpet needed for Jesse's room -/
theorem jesse_carpet_need : 
  additional_carpet_needed 11 15 16 = 149 := by
  sorry

end jesse_carpet_need_l2097_209754


namespace implication_disjunction_equivalence_l2097_209746

theorem implication_disjunction_equivalence (A B : Prop) : (A → B) ↔ (¬A ∨ B) := by
  sorry

end implication_disjunction_equivalence_l2097_209746


namespace largest_multiple_of_seven_below_neg_85_l2097_209784

theorem largest_multiple_of_seven_below_neg_85 :
  ∀ n : ℤ, 7 ∣ n ∧ n < -85 → n ≤ -91 :=
by
  sorry

end largest_multiple_of_seven_below_neg_85_l2097_209784


namespace polar_bear_trout_consumption_l2097_209758

/-- The daily fish consumption of the polar bear in buckets -/
def total_fish : ℝ := 0.6

/-- The daily salmon consumption of the polar bear in buckets -/
def salmon : ℝ := 0.4

/-- The daily trout consumption of the polar bear in buckets -/
def trout : ℝ := total_fish - salmon

theorem polar_bear_trout_consumption :
  trout = 0.2 := by sorry

end polar_bear_trout_consumption_l2097_209758


namespace mod_23_equivalence_l2097_209797

theorem mod_23_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 50238 ≡ n [ZMOD 23] ∧ n = 19 := by
  sorry

end mod_23_equivalence_l2097_209797


namespace train_cars_count_l2097_209772

/-- Calculates the number of cars in a train based on observed data -/
def train_cars (cars_observed : ℕ) (observation_time : ℕ) (total_time : ℕ) : ℕ :=
  (cars_observed * total_time) / observation_time

/-- Proves that the number of cars in the train is 112 given the observed data -/
theorem train_cars_count : train_cars 8 15 210 = 112 := by
  sorry

end train_cars_count_l2097_209772


namespace nested_fraction_evaluation_l2097_209795

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 4))) = 11 / 29 := by
  sorry

end nested_fraction_evaluation_l2097_209795


namespace phone_inventory_and_profit_optimization_l2097_209794

/-- Represents a phone model with purchase and selling prices -/
structure PhoneModel where
  purchasePrice : ℕ
  sellingPrice : ℕ

/-- Represents the inventory and financial data of a business hall -/
structure BusinessHall where
  modelA : PhoneModel
  modelB : PhoneModel
  totalSpent : ℕ
  totalProfit : ℕ

/-- Theorem stating the correct number of units purchased and maximum profit -/
theorem phone_inventory_and_profit_optimization 
  (hall : BusinessHall) 
  (hall_data : hall.modelA.purchasePrice = 3000 ∧ 
               hall.modelA.sellingPrice = 3400 ∧
               hall.modelB.purchasePrice = 3500 ∧ 
               hall.modelB.sellingPrice = 4000 ∧
               hall.totalSpent = 32000 ∧ 
               hall.totalProfit = 4400) :
  (∃ (a b : ℕ), 
    a * hall.modelA.purchasePrice + b * hall.modelB.purchasePrice = hall.totalSpent ∧
    a * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    b * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) = hall.totalProfit ∧
    a = 6 ∧ b = 4) ∧
  (∃ (x : ℕ), 
    x ≥ 10 ∧ 30 - x ≤ 2 * x ∧
    ∀ y : ℕ, y ≥ 10 → 30 - y ≤ 2 * y → 
    x * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - x) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) ≥
    y * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - y) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) ∧
    x * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - x) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) = 14000) :=
by sorry

end phone_inventory_and_profit_optimization_l2097_209794


namespace pie_eating_problem_l2097_209761

theorem pie_eating_problem (initial_stock : ℕ) (daily_portion : ℕ) (day : ℕ) :
  initial_stock = 340 →
  daily_portion > 0 →
  day > 0 →
  initial_stock = day * daily_portion + daily_portion / 4 →
  (day = 5 ∨ day = 21) :=
sorry

end pie_eating_problem_l2097_209761


namespace minimum_students_with_both_devices_l2097_209757

theorem minimum_students_with_both_devices (n : ℕ) (h1 : n % 7 = 0) (h2 : n % 6 = 0) : ∃ x : ℕ,
  x = n * 3 / 7 + n * 5 / 6 - n ∧
  x ≥ 11 ∧
  (∀ y : ℕ, y < x → ∃ m : ℕ, m > n ∧ m % 7 = 0 ∧ m % 6 = 0 ∧ y = m * 3 / 7 + m * 5 / 6 - m) :=
by sorry

#check minimum_students_with_both_devices

end minimum_students_with_both_devices_l2097_209757


namespace ellipse_hyperbola_eccentricity_l2097_209711

theorem ellipse_hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e_ellipse := Real.sqrt 3 / 2
  let c := e_ellipse * a
  let e_hyperbola := Real.sqrt ((a^2 + b^2) / a^2)
  (a^2 = b^2 + c^2) → e_hyperbola = Real.sqrt 5 / 2 := by
  sorry

end ellipse_hyperbola_eccentricity_l2097_209711


namespace sum_of_powers_mod_17_l2097_209750

theorem sum_of_powers_mod_17 :
  (∃ x : ℤ, x * 3 ≡ 1 [ZMOD 17]) →
  (∃ y : ℤ, y * 3^2 ≡ 1 [ZMOD 17]) →
  (∃ z : ℤ, z * 3^3 ≡ 1 [ZMOD 17]) →
  (∃ w : ℤ, w * 3^4 ≡ 1 [ZMOD 17]) →
  (∃ v : ℤ, v * 3^5 ≡ 1 [ZMOD 17]) →
  (∃ u : ℤ, u * 3^6 ≡ 1 [ZMOD 17]) →
  x + y + z + w + v + u ≡ 5 [ZMOD 17] :=
by sorry

end sum_of_powers_mod_17_l2097_209750


namespace equation_solution_l2097_209722

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (8 * x)^16 = (32 * x)^8 → x = 1/2 := by sorry

end equation_solution_l2097_209722


namespace woodburning_price_l2097_209747

/-- Represents the selling price of a woodburning -/
def selling_price : ℝ := 15

/-- Represents the number of woodburnings sold -/
def num_woodburnings : ℕ := 20

/-- Represents the cost of wood -/
def wood_cost : ℝ := 100

/-- Represents the total profit -/
def total_profit : ℝ := 200

/-- Theorem stating that the selling price of each woodburning is $15 -/
theorem woodburning_price : 
  selling_price * num_woodburnings - wood_cost = total_profit :=
by sorry

end woodburning_price_l2097_209747


namespace rectangle_width_decrease_l2097_209779

theorem rectangle_width_decrease (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_L := 1.5 * L
  let new_W := W * (L / new_L)
  (W - new_W) / W = 1 / 3 := by
sorry

end rectangle_width_decrease_l2097_209779


namespace work_completion_time_l2097_209749

/-- Given workers a, b, and c, and their work rates, prove that b alone completes the work in 48 days -/
theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b = 1 / 16)  -- a and b together finish in 16 days
  (h2 : a = 1 / 24)      -- a alone finishes in 24 days
  (h3 : c = 1 / 48)      -- c alone finishes in 48 days
  : b = 1 / 48 :=        -- b alone finishes in 48 days
by sorry

end work_completion_time_l2097_209749


namespace negative_double_negation_l2097_209726

theorem negative_double_negation (x : ℝ) (h : -x = 2) : -(-(-x)) = 2 := by
  sorry

end negative_double_negation_l2097_209726


namespace emma_running_time_l2097_209781

theorem emma_running_time (emma_time : ℝ) (fernando_time : ℝ) : 
  fernando_time = 2 * emma_time →
  emma_time + fernando_time = 60 →
  emma_time = 20 := by
sorry

end emma_running_time_l2097_209781


namespace cos_squared_165_minus_sin_squared_15_l2097_209773

theorem cos_squared_165_minus_sin_squared_15 :
  Real.cos (165 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_165_minus_sin_squared_15_l2097_209773


namespace direction_vector_of_line_l_l2097_209791

/-- The line l is defined by the equation (x-1)/3 = (y+1)/4 -/
def line_l (x y : ℝ) : Prop := (x - 1) / 3 = (y + 1) / 4

/-- A direction vector of a line is a vector parallel to the line -/
def is_direction_vector (v : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (t : ℝ) (x y : ℝ), l x y → l (x + t * v.1) (y + t * v.2)

/-- Prove that (3,4) is a direction vector of the line l -/
theorem direction_vector_of_line_l : is_direction_vector (3, 4) line_l := by
  sorry

end direction_vector_of_line_l_l2097_209791


namespace total_erasers_l2097_209708

theorem total_erasers (celine gabriel julian : ℕ) : 
  celine = 2 * gabriel → 
  julian = 2 * celine → 
  celine = 10 → 
  celine + gabriel + julian = 35 := by
sorry

end total_erasers_l2097_209708


namespace spinner_probability_l2097_209710

-- Define the spinner regions
inductive Region
| A
| B1
| B2
| C

-- Define the probability function
def P : Region → ℚ
| Region.A  => 3/8
| Region.B1 => 1/8
| Region.B2 => 1/4
| Region.C  => 1/4  -- This is what we want to prove

-- State the theorem
theorem spinner_probability :
  P Region.C = 1/4 :=
by
  sorry

-- Additional lemmas to help with the proof
lemma total_probability :
  P Region.A + P Region.B1 + P Region.B2 + P Region.C = 1 :=
by
  sorry

lemma b_subregions :
  P Region.B1 + P Region.B2 = 3/8 :=
by
  sorry

end spinner_probability_l2097_209710
