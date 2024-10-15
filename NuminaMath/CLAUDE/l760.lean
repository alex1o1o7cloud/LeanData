import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l760_76043

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) :
  (Finset.range 2017).sum (fun k => i^(k + 1)) = i := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l760_76043


namespace NUMINAMATH_CALUDE_elderly_people_not_well_defined_l760_76042

-- Define a structure for a potential set
structure PotentialSet where
  elements : String
  is_well_defined : Bool

-- Define the criteria for a well-defined set
def is_well_defined_set (s : PotentialSet) : Prop :=
  s.is_well_defined = true

-- Define the set of elderly people
def elderly_people : PotentialSet :=
  { elements := "All elderly people", is_well_defined := false }

-- Theorem stating that the set of elderly people is not well-defined
theorem elderly_people_not_well_defined : ¬(is_well_defined_set elderly_people) := by
  sorry

#check elderly_people_not_well_defined

end NUMINAMATH_CALUDE_elderly_people_not_well_defined_l760_76042


namespace NUMINAMATH_CALUDE_max_plumber_earnings_l760_76082

def toilet_rate : ℕ := 50
def shower_rate : ℕ := 40
def sink_rate : ℕ := 30

def job1_earnings : ℕ := 3 * toilet_rate + 3 * sink_rate
def job2_earnings : ℕ := 2 * toilet_rate + 5 * sink_rate
def job3_earnings : ℕ := 1 * toilet_rate + 2 * shower_rate + 3 * sink_rate

theorem max_plumber_earnings :
  max job1_earnings (max job2_earnings job3_earnings) = 250 := by
  sorry

end NUMINAMATH_CALUDE_max_plumber_earnings_l760_76082


namespace NUMINAMATH_CALUDE_ferry_distance_ratio_l760_76053

/-- Represents a ferry with speed and travel time -/
structure Ferry where
  speed : ℝ
  time : ℝ

/-- The problem setup -/
def ferryProblem : Prop :=
  ∃ (P Q : Ferry),
    P.speed = 8 ∧
    P.time = 2 ∧
    Q.speed = P.speed + 4 ∧
    Q.time = P.time + 2 ∧
    Q.speed * Q.time / (P.speed * P.time) = 3

/-- The theorem to prove -/
theorem ferry_distance_ratio :
  ferryProblem := by sorry

end NUMINAMATH_CALUDE_ferry_distance_ratio_l760_76053


namespace NUMINAMATH_CALUDE_sum_of_three_digit_numbers_divisible_by_37_l760_76045

/-- A function that generates all possible three-digit numbers from three digits -/
def generateThreeDigitNumbers (a b c : ℕ) : List ℕ :=
  [100*a + 10*b + c,
   100*a + 10*c + b,
   100*b + 10*a + c,
   100*b + 10*c + a,
   100*c + 10*a + b,
   100*c + 10*b + a]

/-- Theorem: The sum of all possible three-digit numbers formed from three distinct non-zero digits is divisible by 37 -/
theorem sum_of_three_digit_numbers_divisible_by_37 
  (a b c : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  37 ∣ (List.sum (generateThreeDigitNumbers a b c)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_digit_numbers_divisible_by_37_l760_76045


namespace NUMINAMATH_CALUDE_sum_of_squares_l760_76098

theorem sum_of_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_products : a * b + a * c + b * c = -3)
  (product : a * b * c = 2) :
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l760_76098


namespace NUMINAMATH_CALUDE_oranges_given_eq_difference_l760_76031

/-- The number of oranges Clarence gave to Joyce -/
def oranges_given : ℝ := sorry

/-- Clarence's initial number of oranges -/
def initial_oranges : ℝ := 5.0

/-- Clarence's remaining number of oranges -/
def remaining_oranges : ℝ := 2.0

/-- Theorem stating that the number of oranges given is equal to the difference between initial and remaining oranges -/
theorem oranges_given_eq_difference : 
  oranges_given = initial_oranges - remaining_oranges := by sorry

end NUMINAMATH_CALUDE_oranges_given_eq_difference_l760_76031


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_improvement_l760_76025

/-- Proves the increase in travel distance after modifying a car's fuel efficiency -/
theorem car_fuel_efficiency_improvement (initial_mpg : ℝ) (tank_capacity : ℝ) 
  (fuel_reduction_factor : ℝ) (h1 : initial_mpg = 28) (h2 : tank_capacity = 15) 
  (h3 : fuel_reduction_factor = 0.8) : 
  (initial_mpg / fuel_reduction_factor - initial_mpg) * tank_capacity = 84 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_improvement_l760_76025


namespace NUMINAMATH_CALUDE_grocer_banana_profit_l760_76011

/-- Represents the profit calculation for a grocer selling bananas -/
theorem grocer_banana_profit : 
  let purchase_rate : ℚ := 3 / 0.5  -- 3 pounds per $0.50
  let sell_rate : ℚ := 4 / 1        -- 4 pounds per $1.00
  let total_pounds : ℚ := 108       -- Total pounds purchased
  let cost_price := total_pounds / purchase_rate
  let sell_price := total_pounds / sell_rate
  let profit := sell_price - cost_price
  profit = 9 := by sorry

end NUMINAMATH_CALUDE_grocer_banana_profit_l760_76011


namespace NUMINAMATH_CALUDE_odd_function_property_l760_76034

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 9

-- Theorem statement
theorem odd_function_property (h1 : ∀ x, f (-x) = -f x) 
                              (h2 : g (-2) = 3) : 
  f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l760_76034


namespace NUMINAMATH_CALUDE_angle_less_iff_sin_less_l760_76086

theorem angle_less_iff_sin_less (A B : Real) (hA : 0 < A) (hB : B < π) (hAB : A + B < π) :
  A < B ↔ Real.sin A < Real.sin B := by sorry

end NUMINAMATH_CALUDE_angle_less_iff_sin_less_l760_76086


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l760_76029

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ), 
    P = 2 / 15 ∧ Q = 1 / 3 ∧ R = 0 ∧
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 5*x + 6) / ((x - 1)*(x - 4)*(x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l760_76029


namespace NUMINAMATH_CALUDE_tens_digit_of_11_pow_2045_l760_76095

theorem tens_digit_of_11_pow_2045 : ∃ k : ℕ, 11^2045 ≡ 50 + k [ZMOD 100] :=
by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_11_pow_2045_l760_76095


namespace NUMINAMATH_CALUDE_train_passing_platform_l760_76061

/-- The time taken for a train to pass a platform -/
theorem train_passing_platform (train_length platform_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 360 →
  platform_length = 390 →
  train_speed_kmh = 45 →
  (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_platform_l760_76061


namespace NUMINAMATH_CALUDE_age_difference_l760_76046

theorem age_difference (A B C : ℕ) (h1 : C = A - 17) : A + B - (B + C) = 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l760_76046


namespace NUMINAMATH_CALUDE_f_properties_l760_76073

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x > 0 then x^2 - 3
  else if x < 0 then -x^2 + 3
  else 0

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x^2 - 3) ∧  -- given condition for x > 0
  (∀ x < 0, f x = -x^2 + 3) ∧  -- prove this for x < 0
  (f 0 = 0) ∧  -- prove f(0) = 0
  ({x : ℝ | f x = 2 * x} = {-3, 0, 3}) :=  -- prove the solution set
by sorry

end NUMINAMATH_CALUDE_f_properties_l760_76073


namespace NUMINAMATH_CALUDE_damien_picked_fraction_l760_76000

/-- Proves that Damien picked 3/5 of the fruits from the trees --/
theorem damien_picked_fraction (apples plums : ℕ) (picked_fraction : ℚ) : 
  apples = 3 * plums →  -- The number of apples is three times the number of plums
  apples = 180 →  -- The initial number of apples is 180
  (1 - picked_fraction) * (apples + plums) = 96 →  -- After picking, 96 fruits remain
  picked_fraction = 3 / 5 := by
sorry


end NUMINAMATH_CALUDE_damien_picked_fraction_l760_76000


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l760_76003

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 * i) / (1 + i)
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l760_76003


namespace NUMINAMATH_CALUDE_monday_sales_proof_l760_76072

/-- Represents the daily pastry sales for a week -/
structure WeeklySales :=
  (monday : ℕ)
  (increase_per_day : ℕ)
  (days_per_week : ℕ)

/-- Calculates the total sales for the week -/
def total_sales (s : WeeklySales) : ℕ :=
  s.days_per_week * s.monday + (s.days_per_week * (s.days_per_week - 1) * s.increase_per_day) / 2

/-- Theorem: If daily sales increase by 1 for 7 days and average 5 per day, Monday's sales were 2 -/
theorem monday_sales_proof (s : WeeklySales) 
  (h1 : s.increase_per_day = 1)
  (h2 : s.days_per_week = 7)
  (h3 : total_sales s / s.days_per_week = 5) :
  s.monday = 2 := by
  sorry


end NUMINAMATH_CALUDE_monday_sales_proof_l760_76072


namespace NUMINAMATH_CALUDE_arrangement_theorem_l760_76057

def arrangement_count (n : ℕ) (zeros : ℕ) : ℕ :=
  if n = 27 ∧ zeros = 13 then 14
  else if n = 26 ∧ zeros = 13 then 105
  else 0

theorem arrangement_theorem (n : ℕ) (zeros : ℕ) :
  (n = 27 ∨ n = 26) ∧ zeros = 13 →
  arrangement_count n zeros = 
    (if n = 27 then 14 else 105) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l760_76057


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l760_76020

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧ 
  (∀ (n' : ℕ), n' > 0 → 
    (∃ (k : ℕ), 5 * n' = k^2) → 
    (∃ (m : ℕ), 3 * n' = m^3) → 
    n ≤ n') ∧
  n = 225 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l760_76020


namespace NUMINAMATH_CALUDE_ellies_calculation_l760_76026

theorem ellies_calculation (x y z : ℝ) 
  (h1 : x - (y + z) = 18) 
  (h2 : x - y - z = 6) : 
  x - y = 12 := by
sorry

end NUMINAMATH_CALUDE_ellies_calculation_l760_76026


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l760_76037

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x + a|

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l760_76037


namespace NUMINAMATH_CALUDE_elvis_album_songs_l760_76058

/-- Calculates the number of songs on Elvis' new album given the studio time constraints. -/
theorem elvis_album_songs (
  total_studio_time : ℕ
  ) (record_time : ℕ) (edit_time : ℕ) (write_time : ℕ) 
  (h1 : total_studio_time = 5 * 60)  -- 5 hours in minutes
  (h2 : record_time = 12)            -- 12 minutes to record each song
  (h3 : edit_time = 30)              -- 30 minutes to edit all songs
  (h4 : write_time = 15)             -- 15 minutes to write each song
  : ℕ := by
  
  -- The number of songs is equal to the available time for writing and recording
  -- divided by the time needed for writing and recording one song
  have num_songs : ℕ := (total_studio_time - edit_time) / (write_time + record_time)
  
  -- Prove that num_songs equals 10
  sorry

#eval (5 * 60 - 30) / (15 + 12)  -- Should evaluate to 10

end NUMINAMATH_CALUDE_elvis_album_songs_l760_76058


namespace NUMINAMATH_CALUDE_f_inequality_l760_76051

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, f (x + 1) = f (-(x + 1)))
variable (h2 : ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y)
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ < 0)
variable (h4 : x₂ > 0)
variable (h5 : x₁ + x₂ < -2)

-- State the theorem
theorem f_inequality : f (-x₁) > f (-x₂) := by sorry

end NUMINAMATH_CALUDE_f_inequality_l760_76051


namespace NUMINAMATH_CALUDE_downstream_distance_l760_76021

-- Define the given constants
def boat_speed : ℝ := 22
def stream_speed : ℝ := 5
def time_downstream : ℝ := 2

-- Define the theorem
theorem downstream_distance :
  let effective_speed := boat_speed + stream_speed
  effective_speed * time_downstream = 54 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_l760_76021


namespace NUMINAMATH_CALUDE_salon_buys_33_cans_l760_76022

/-- Represents the number of cans of hairspray a salon buys daily. -/
def salon_hairspray_cans (customers : ℕ) (cans_per_customer : ℕ) (extra_cans : ℕ) : ℕ :=
  customers * cans_per_customer + extra_cans

/-- Theorem stating that the salon buys 33 cans of hairspray daily. -/
theorem salon_buys_33_cans :
  salon_hairspray_cans 14 2 5 = 33 := by
  sorry

#eval salon_hairspray_cans 14 2 5

end NUMINAMATH_CALUDE_salon_buys_33_cans_l760_76022


namespace NUMINAMATH_CALUDE_sum_y_equals_375_l760_76054

variable (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ)

-- Define the sum of x values
def sum_x : ℝ := x₁ + x₂ + x₃ + x₄ + x₅

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.67 * x + 54.9

-- State the theorem
theorem sum_y_equals_375 
  (h_sum_x : sum_x = 150) : 
  y₁ + y₂ + y₃ + y₄ + y₅ = 375 := by
  sorry

end NUMINAMATH_CALUDE_sum_y_equals_375_l760_76054


namespace NUMINAMATH_CALUDE_a_50_equals_6_5_l760_76023

-- Define the sequence a_n
def a : ℕ → ℚ
| n => sorry

-- Theorem statement
theorem a_50_equals_6_5 : a 50 = 6/5 := by sorry

end NUMINAMATH_CALUDE_a_50_equals_6_5_l760_76023


namespace NUMINAMATH_CALUDE_power_seven_eight_mod_hundred_l760_76006

theorem power_seven_eight_mod_hundred : 7^8 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_eight_mod_hundred_l760_76006


namespace NUMINAMATH_CALUDE_ice_cream_cost_l760_76044

theorem ice_cream_cost (total_spent : ℚ) (apple_extra_cost : ℚ) 
  (h1 : total_spent = 25)
  (h2 : apple_extra_cost = 10) : 
  ∃ (ice_cream_cost : ℚ), 
    ice_cream_cost + (ice_cream_cost + apple_extra_cost) = total_spent ∧ 
    ice_cream_cost = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l760_76044


namespace NUMINAMATH_CALUDE_self_inverse_matrix_l760_76027

theorem self_inverse_matrix (c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; c, d]
  A * A = 1 → c = 7.5 ∧ d = -4 := by
sorry

end NUMINAMATH_CALUDE_self_inverse_matrix_l760_76027


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l760_76078

theorem arithmetic_expression_equality : 68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l760_76078


namespace NUMINAMATH_CALUDE_passengers_off_in_texas_l760_76038

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : ℕ
  texas_off : ℕ
  texas_on : ℕ
  nc_off : ℕ
  nc_on : ℕ
  final : ℕ

/-- Theorem stating that 48 passengers got off in Texas --/
theorem passengers_off_in_texas (fp : FlightPassengers) 
  (h1 : fp.initial = 124)
  (h2 : fp.texas_on = 24)
  (h3 : fp.nc_off = 47)
  (h4 : fp.nc_on = 14)
  (h5 : fp.final = 67)
  (h6 : fp.initial - fp.texas_off + fp.texas_on - fp.nc_off + fp.nc_on = fp.final) :
  fp.texas_off = 48 := by
  sorry


end NUMINAMATH_CALUDE_passengers_off_in_texas_l760_76038


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l760_76028

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a > r1) (hb : b > r2) :
  let d := Nat.gcd (a - r1) (b - r2)
  d = Nat.gcd a b ∧ 
  a % d = r1 ∧ 
  b % d = r2 ∧ 
  ∀ m : ℕ, m > d → (a % m = r1 ∧ b % m = r2) → False :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l760_76028


namespace NUMINAMATH_CALUDE_complex_z_value_l760_76096

def is_negative_real (z : ℂ) : Prop := ∃ (r : ℝ), r < 0 ∧ z = r

def is_purely_imaginary (z : ℂ) : Prop := ∃ (r : ℝ), z = r * Complex.I

theorem complex_z_value (z : ℂ) 
  (h1 : is_negative_real ((z - 3*Complex.I) / (z + Complex.I)))
  (h2 : is_purely_imaginary ((z - 3) / (z + 1))) :
  z = Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_z_value_l760_76096


namespace NUMINAMATH_CALUDE_largest_unorderable_number_l760_76074

def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_unorderable_number : 
  (∀ m > 43, is_orderable m) ∧ ¬(is_orderable 43) := by
  sorry

end NUMINAMATH_CALUDE_largest_unorderable_number_l760_76074


namespace NUMINAMATH_CALUDE_complete_square_m_values_l760_76092

/-- A polynomial of the form x^2 + mx + 4 can be factored using the complete square formula -/
def is_complete_square (m : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 4 = (x + a)^2

/-- If a polynomial x^2 + mx + 4 can be factored using the complete square formula,
    then m = 4 or m = -4 -/
theorem complete_square_m_values (m : ℝ) :
  is_complete_square m → m = 4 ∨ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_m_values_l760_76092


namespace NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l760_76068

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

theorem f_always_negative_iff_a_in_range :
  (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l760_76068


namespace NUMINAMATH_CALUDE_line_mn_properties_l760_76005

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the condition for the sum of vertical coordinates
def sum_of_verticals (m n : ℝ × ℝ) : Prop := m.2 + n.2 = 4

-- Define the angle condition
def angle_condition (m n : ℝ × ℝ) : Prop :=
  (m.2 / (m.1 + 2)) + (n.2 / (n.1 + 2)) = 0

-- Main theorem
theorem line_mn_properties (m n : ℝ × ℝ) :
  parabola m → parabola n → sum_of_verticals m n → angle_condition m n →
  ∃ k b : ℝ, k = 1 ∧ b = -2 ∧ ∀ x y : ℝ, y = k * x + b ↔ (x = m.1 ∧ y = m.2) ∨ (x = n.1 ∧ y = n.2) :=
sorry

end NUMINAMATH_CALUDE_line_mn_properties_l760_76005


namespace NUMINAMATH_CALUDE_bruce_total_payment_l760_76060

def grape_quantity : ℝ := 8
def grape_rate : ℝ := 70
def grape_discount : ℝ := 0.1
def mango_quantity : ℝ := 10
def mango_rate : ℝ := 55

theorem bruce_total_payment :
  let grape_cost := grape_quantity * grape_rate
  let grape_discount_amount := grape_cost * grape_discount
  let final_grape_cost := grape_cost - grape_discount_amount
  let mango_cost := mango_quantity * mango_rate
  final_grape_cost + mango_cost = 1054 := by sorry

end NUMINAMATH_CALUDE_bruce_total_payment_l760_76060


namespace NUMINAMATH_CALUDE_minoxidil_concentration_l760_76018

/-- Proves that the initial concentration of Minoxidil is 2% --/
theorem minoxidil_concentration 
  (initial_volume : ℝ) 
  (added_volume : ℝ) 
  (added_concentration : ℝ) 
  (final_volume : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 70)
  (h2 : added_volume = 35)
  (h3 : added_concentration = 0.05)
  (h4 : final_volume = 105)
  (h5 : final_concentration = 0.03)
  (h6 : final_volume = initial_volume + added_volume) :
  ∃ (initial_concentration : ℝ), 
    initial_concentration = 0.02 ∧ 
    initial_volume * initial_concentration + added_volume * added_concentration = 
    final_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_minoxidil_concentration_l760_76018


namespace NUMINAMATH_CALUDE_unique_colors_count_l760_76035

/-- The total number of unique colored pencils owned by Serenity, Jordan, and Alex -/
def total_unique_colors (serenity_colors jordan_colors alex_colors
                         serenity_jordan_shared serenity_alex_shared jordan_alex_shared
                         all_shared : ℕ) : ℕ :=
  serenity_colors + jordan_colors + alex_colors
  - (serenity_jordan_shared + serenity_alex_shared + jordan_alex_shared - 2 * all_shared)
  - all_shared

/-- Theorem stating the total number of unique colored pencils -/
theorem unique_colors_count :
  total_unique_colors 24 36 30 8 5 10 3 = 73 := by
  sorry

end NUMINAMATH_CALUDE_unique_colors_count_l760_76035


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l760_76052

theorem simple_interest_rate_calculation (P : ℝ) (R : ℝ) : 
  P * (1 + 7 * R / 100) = 7 / 6 * P → R = 100 / 49 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l760_76052


namespace NUMINAMATH_CALUDE_ab_bounds_l760_76056

theorem ab_bounds (a b c : ℝ) (h1 : a ≠ b) (h2 : c > 0)
  (h3 : a^4 - 2019*a = c) (h4 : b^4 - 2019*b = c) :
  -Real.sqrt c < a * b ∧ a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_bounds_l760_76056


namespace NUMINAMATH_CALUDE_miss_world_contest_l760_76047

-- Define the total number of girls
def total_girls : ℕ := 48

-- Define the number of girls with blue eyes and white skin
def blue_eyes_white_skin : ℕ := 12

-- Define the number of girls with light black skin
def light_black_skin : ℕ := 28

-- Define the number of girls with brown eyes
def brown_eyes : ℕ := 15

-- Define a as the number of girls with brown eyes and light black skin
def a : ℕ := sorry

-- Define b as the number of girls with white skin and brown eyes
def b : ℕ := sorry

-- Theorem to prove
theorem miss_world_contest :
  a = 7 ∧ b = 8 := by sorry

end NUMINAMATH_CALUDE_miss_world_contest_l760_76047


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l760_76009

theorem range_of_2a_minus_b (a b : ℝ) (ha : 2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l760_76009


namespace NUMINAMATH_CALUDE_jims_age_fraction_l760_76081

theorem jims_age_fraction (tom_age_5_years_ago : ℕ) (jim_age_in_2_years : ℕ) : 
  tom_age_5_years_ago = 32 →
  jim_age_in_2_years = 29 →
  ∃ f : ℚ, 
    (jim_age_in_2_years - 9 : ℚ) = f * (tom_age_5_years_ago + 2 : ℚ) + 5 ∧ 
    f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jims_age_fraction_l760_76081


namespace NUMINAMATH_CALUDE_optimal_speed_theorem_l760_76002

theorem optimal_speed_theorem (d t : ℝ) 
  (h1 : d = 45 * (t + 1/15))
  (h2 : d = 75 * (t - 1/15)) :
  d / t = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_optimal_speed_theorem_l760_76002


namespace NUMINAMATH_CALUDE_square_difference_of_even_integers_l760_76030

theorem square_difference_of_even_integers (x y : ℕ) : 
  Even x → Even y → x > y → x + y = 68 → x - y = 20 → x^2 - y^2 = 1360 :=
by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_even_integers_l760_76030


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l760_76040

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l760_76040


namespace NUMINAMATH_CALUDE_cream_cheese_price_l760_76064

-- Define variables for bagel and cream cheese prices
variable (B : ℝ) -- Price of one bag of bagels
variable (C : ℝ) -- Price of one package of cream cheese

-- Define the equations from the problem
def monday_equation : Prop := 2 * B + 3 * C = 12
def friday_equation : Prop := 4 * B + 2 * C = 14

-- Theorem statement
theorem cream_cheese_price 
  (h1 : monday_equation B C) 
  (h2 : friday_equation B C) : 
  C = 2.5 := by sorry

end NUMINAMATH_CALUDE_cream_cheese_price_l760_76064


namespace NUMINAMATH_CALUDE_library_book_sorting_l760_76077

theorem library_book_sorting (damaged : ℕ) (obsolete : ℕ) : 
  obsolete = 6 * damaged - 8 →
  damaged + obsolete = 69 →
  damaged = 11 := by
sorry

end NUMINAMATH_CALUDE_library_book_sorting_l760_76077


namespace NUMINAMATH_CALUDE_megan_total_markers_l760_76085

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := 217

/-- The number of markers Robert gave to Megan -/
def received_markers : ℕ := 109

/-- The total number of markers Megan has -/
def total_markers : ℕ := initial_markers + received_markers

theorem megan_total_markers : total_markers = 326 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_markers_l760_76085


namespace NUMINAMATH_CALUDE_function_property_l760_76094

theorem function_property (f : ℕ → ℝ) :
  f 1 = 3/2 ∧
  (∀ x y : ℕ, f (x + y) = (1 + y / (x + 1 : ℝ)) * f x + (1 + x / (y + 1 : ℝ)) * f y + x^2 * y + x * y + x * y^2) →
  ∀ x : ℕ, f x = (1/4 : ℝ) * x * (x + 1) * (2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l760_76094


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l760_76007

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - 2*x ≤ 0 → -1 ≤ x ∧ x ≤ 3) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x > 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l760_76007


namespace NUMINAMATH_CALUDE_area_STUV_l760_76071

/-- A semicircle with an inscribed square PQRS and another square STUV -/
structure SemicircleWithSquares where
  /-- The radius of the semicircle -/
  r : ℝ
  /-- The side length of the inscribed square PQRS -/
  s : ℝ
  /-- The side length of the square STUV -/
  x : ℝ
  /-- The radius is determined by the side length of PQRS -/
  h_radius : r = s * Real.sqrt 2 / 2
  /-- PQRS is inscribed in the semicircle -/
  h_inscribed : s^2 + s^2 = (2*r)^2
  /-- STUV has a vertex on the semicircle -/
  h_on_semicircle : 6^2 + x^2 = r^2

/-- The area of square STUV is 36 -/
theorem area_STUV (c : SemicircleWithSquares) : c.x^2 = 36 := by
  sorry

#check area_STUV

end NUMINAMATH_CALUDE_area_STUV_l760_76071


namespace NUMINAMATH_CALUDE_balloon_distribution_l760_76013

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) (balloons_returned : ℕ) : 
  total_balloons = 250 → 
  num_friends = 5 → 
  balloons_returned = 11 → 
  (total_balloons / num_friends) - balloons_returned = 39 := by
sorry

end NUMINAMATH_CALUDE_balloon_distribution_l760_76013


namespace NUMINAMATH_CALUDE_max_product_sum_2004_l760_76090

theorem max_product_sum_2004 :
  ∃ (a b : ℤ), a + b = 2004 ∧
  ∀ (x y : ℤ), x + y = 2004 → x * y ≤ a * b ∧
  a * b = 1004004 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_2004_l760_76090


namespace NUMINAMATH_CALUDE_sonika_deposit_l760_76070

/-- Calculates the final amount after simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem sonika_deposit :
  ∀ (P R : ℝ),
  simpleInterest P (R / 100) 3 = 10200 →
  simpleInterest P ((R + 2) / 100) 3 = 10680 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_sonika_deposit_l760_76070


namespace NUMINAMATH_CALUDE_square_difference_equals_24_l760_76099

theorem square_difference_equals_24 (x y : ℝ) (h1 : x + y = 4) (h2 : x - y = 6) :
  x^2 - y^2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_24_l760_76099


namespace NUMINAMATH_CALUDE_count_valid_s_l760_76019

def is_valid_sequence (n p q r s : ℕ) : Prop :=
  p < q ∧ q < r ∧ r < s ∧ s ≤ n ∧ 100 < p ∧
  ((q = p + 1 ∧ r = q + 1) ∨ (r = q + 1 ∧ s = r + 1) ∨ (q = p + 1 ∧ r = q + 1 ∧ s = r + 1))

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_removed (p q r s : ℕ) : ℕ := p + q + r + s

def remaining_sum (n p q r s : ℕ) : ℕ := sum_first_n n - sum_removed p q r s

def average_is_correct (n p q r s : ℕ) : Prop :=
  (remaining_sum n p q r s : ℚ) / (n - 4 : ℚ) = 89.5625

theorem count_valid_s (n : ℕ) : 
  (∃ p q r s, is_valid_sequence n p q r s ∧ average_is_correct n p q r s) →
  (∃! (valid_s : Finset ℕ), 
    (∀ s, s ∈ valid_s ↔ ∃ p q r, is_valid_sequence n p q r s ∧ average_is_correct n p q r s) ∧
    valid_s.card = 22) :=
sorry

end NUMINAMATH_CALUDE_count_valid_s_l760_76019


namespace NUMINAMATH_CALUDE_product_formula_l760_76012

theorem product_formula (a b : ℕ) :
  (100 - a) * (100 + b) = ((b + (200 - a) - 100) * 100) - a * b := by
  sorry

end NUMINAMATH_CALUDE_product_formula_l760_76012


namespace NUMINAMATH_CALUDE_second_tea_price_l760_76008

/-- Represents the price of tea varieties and their mixture --/
structure TeaPrices where
  first : ℝ
  second : ℝ
  third : ℝ
  mixture : ℝ

/-- Theorem stating the price of the second tea variety --/
theorem second_tea_price (p : TeaPrices)
  (h1 : p.first = 126)
  (h2 : p.third = 177.5)
  (h3 : p.mixture = 154)
  (h4 : p.mixture * 4 = p.first + p.second + 2 * p.third) :
  p.second = 135 := by
  sorry

end NUMINAMATH_CALUDE_second_tea_price_l760_76008


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l760_76015

/-- Given that 55 cows eat 55 bags of husk in 55 days, prove that one cow will eat one bag of husk in 55 days. -/
theorem cow_husk_consumption (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) 
  (h1 : num_cows = 55) 
  (h2 : num_bags = 55) 
  (h3 : num_days = 55) : 
  num_days = 55 := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l760_76015


namespace NUMINAMATH_CALUDE_prob_one_red_bag_with_three_red_balls_l760_76080

/-- A bag containing red and non-red balls -/
structure Bag where
  red : ℕ
  nonRed : ℕ

/-- The probability of drawing exactly one red ball in two consecutive draws with replacement -/
def probOneRedWithReplacement (b : Bag) : ℚ :=
  let totalBalls := b.red + b.nonRed
  let probRed := b.red / totalBalls
  let probNonRed := b.nonRed / totalBalls
  2 * (probRed * probNonRed)

/-- The probability of drawing exactly one red ball in two consecutive draws without replacement -/
def probOneRedWithoutReplacement (b : Bag) : ℚ :=
  let totalBalls := b.red + b.nonRed
  let probRedFirst := b.red / totalBalls
  let probNonRedSecond := b.nonRed / (totalBalls - 1)
  2 * (probRedFirst * probNonRedSecond)

theorem prob_one_red_bag_with_three_red_balls :
  let b : Bag := { red := 3, nonRed := 3 }
  probOneRedWithReplacement b = 1/2 ∧ probOneRedWithoutReplacement b = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_red_bag_with_three_red_balls_l760_76080


namespace NUMINAMATH_CALUDE_calculate_expression_l760_76055

theorem calculate_expression : 18 - (-16) / (2^3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l760_76055


namespace NUMINAMATH_CALUDE_f_has_max_iff_a_ge_e_l760_76036

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ a then Real.log x else a / x

-- Theorem statement
theorem f_has_max_iff_a_ge_e (a : ℝ) :
  (∃ (M : ℝ), ∀ (x : ℝ), x > 0 → f a x ≤ M) ↔ a ≥ Real.exp 1 := by
  sorry

end

end NUMINAMATH_CALUDE_f_has_max_iff_a_ge_e_l760_76036


namespace NUMINAMATH_CALUDE_holiday_ticket_cost_theorem_l760_76004

def holiday_ticket_cost (regular_adult_price : ℝ) : ℝ :=
  let holiday_adult_price := 1.1 * regular_adult_price
  let child_price := 0.5 * regular_adult_price
  6 * holiday_adult_price + 5 * child_price

theorem holiday_ticket_cost_theorem (regular_adult_price : ℝ) :
  4 * (1.1 * regular_adult_price) + 3 * (0.5 * regular_adult_price) = 28.80 →
  holiday_ticket_cost regular_adult_price = 44.41 := by
  sorry

end NUMINAMATH_CALUDE_holiday_ticket_cost_theorem_l760_76004


namespace NUMINAMATH_CALUDE_paul_cookie_price_l760_76076

/-- Represents a cookie baker -/
structure Baker where
  name : String
  num_cookies : ℕ
  price_per_cookie : ℚ

/-- The total amount of dough used by all bakers -/
def total_dough : ℝ := 120

theorem paul_cookie_price 
  (art paul : Baker)
  (h1 : art.name = "Art")
  (h2 : paul.name = "Paul")
  (h3 : art.num_cookies = 10)
  (h4 : paul.num_cookies = 20)
  (h5 : art.price_per_cookie = 1/2)
  (h6 : (total_dough / art.num_cookies) = (total_dough / paul.num_cookies)) :
  paul.price_per_cookie = 1/4 := by
sorry

end NUMINAMATH_CALUDE_paul_cookie_price_l760_76076


namespace NUMINAMATH_CALUDE_cube_cutting_l760_76067

theorem cube_cutting (n s : ℕ) : 
  n > s → 
  n^3 - s^3 = 152 → 
  n = 6 ∧ s = 4 := by sorry

end NUMINAMATH_CALUDE_cube_cutting_l760_76067


namespace NUMINAMATH_CALUDE_platform_length_l760_76041

theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_platform = 27) 
  (h3 : time_pole = 18) : 
  ∃ (platform_length : ℝ), platform_length = 150 ∧ 
  (train_length + platform_length) / time_platform = train_length / time_pole :=
sorry

end NUMINAMATH_CALUDE_platform_length_l760_76041


namespace NUMINAMATH_CALUDE_polynomial_identity_coefficients_l760_76097

theorem polynomial_identity_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x : ℝ, x^5 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + a₅*(x+2)^5) : 
  a₃ = 40 ∧ a₀ + a₁ + a₂ + a₄ + a₅ = -41 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_coefficients_l760_76097


namespace NUMINAMATH_CALUDE_total_letters_written_l760_76062

/-- The number of letters Nathan can write in one hour -/
def nathan_speed : ℕ := 25

/-- Jacob's writing speed relative to Nathan's -/
def jacob_relative_speed : ℕ := 2

/-- The number of hours they write together -/
def total_hours : ℕ := 10

/-- Theorem stating the total number of letters Jacob and Nathan can write together -/
theorem total_letters_written : 
  (nathan_speed + jacob_relative_speed * nathan_speed) * total_hours = 750 := by
  sorry

end NUMINAMATH_CALUDE_total_letters_written_l760_76062


namespace NUMINAMATH_CALUDE_watermelon_weight_is_reasonable_l760_76091

/-- The approximate weight of a typical watermelon in grams -/
def watermelon_weight : ℕ := 4000

/-- Predicate to determine if a given weight is a reasonable approximation for a watermelon -/
def is_reasonable_watermelon_weight (weight : ℕ) : Prop :=
  3500 ≤ weight ∧ weight ≤ 4500

/-- Theorem stating that the defined watermelon weight is a reasonable approximation -/
theorem watermelon_weight_is_reasonable : 
  is_reasonable_watermelon_weight watermelon_weight := by
  sorry

end NUMINAMATH_CALUDE_watermelon_weight_is_reasonable_l760_76091


namespace NUMINAMATH_CALUDE_megan_cupcakes_l760_76014

theorem megan_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 32)
  (h2 : packages = 6)
  (h3 : cupcakes_per_package = 6) :
  todd_ate + packages * cupcakes_per_package = 68 := by
  sorry

end NUMINAMATH_CALUDE_megan_cupcakes_l760_76014


namespace NUMINAMATH_CALUDE_count_common_divisors_l760_76050

/-- The number of positive divisors that 9240 and 10080 have in common -/
def common_divisors_count : ℕ := 32

/-- The first given number -/
def n1 : ℕ := 9240

/-- The second given number -/
def n2 : ℕ := 10080

/-- Theorem stating that the number of positive divisors that n1 and n2 have in common is equal to common_divisors_count -/
theorem count_common_divisors : 
  (Finset.filter (λ d => d ∣ n1 ∧ d ∣ n2) (Finset.range (min n1 n2 + 1))).card = common_divisors_count := by
  sorry


end NUMINAMATH_CALUDE_count_common_divisors_l760_76050


namespace NUMINAMATH_CALUDE_braking_velocities_l760_76069

/-- The displacement function representing the braking system -/
def s (t : ℝ) : ℝ := -3 * t^3 + t^2 + 20

/-- The velocity function (derivative of displacement) -/
def v (t : ℝ) : ℝ := -9 * t^2 + 2 * t

/-- Theorem stating the average and instantaneous velocities during braking -/
theorem braking_velocities :
  (∀ t ∈ Set.Icc 0 2, s t ≥ 0) →  -- Braking completes within 2 seconds
  ((s 1 - s 0) / 1 = -2) ∧        -- Average velocity in first second
  ((s 2 - s 1) / 1 = -18) ∧       -- Average velocity between 1 and 2 seconds
  (v 1 = -7)                      -- Instantaneous velocity at 1 second
:= by sorry

end NUMINAMATH_CALUDE_braking_velocities_l760_76069


namespace NUMINAMATH_CALUDE_train_speed_l760_76083

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 160)
  (h2 : bridge_length = 215)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l760_76083


namespace NUMINAMATH_CALUDE_board_traversal_paths_bound_l760_76033

/-- A piece on an n × n board that can move one step at a time (up, down, left, or right) --/
structure Piece (n : ℕ) where
  x : Fin n
  y : Fin n

/-- The number of unique paths to traverse the entire n × n board --/
def t (n : ℕ) : ℕ := sorry

/-- The theorem to be proved --/
theorem board_traversal_paths_bound {n : ℕ} (h : n ≥ 100) :
  (1.25 : ℝ) < (t n : ℝ) ^ (1 / (n^2 : ℝ)) ∧ (t n : ℝ) ^ (1 / (n^2 : ℝ)) < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_board_traversal_paths_bound_l760_76033


namespace NUMINAMATH_CALUDE_max_value_of_expression_l760_76093

theorem max_value_of_expression (x y : ℝ) (h : x^2 + y^2 ≠ 0) :
  (3*x^2 + 16*x*y + 15*y^2) / (x^2 + y^2) ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l760_76093


namespace NUMINAMATH_CALUDE_savings_ratio_l760_76016

theorem savings_ratio (initial_savings : ℝ) (final_savings : ℝ) (months : ℕ) 
  (h1 : initial_savings = 10)
  (h2 : final_savings = 160)
  (h3 : months = 5) :
  ∃ (ratio : ℝ), ratio = 2 ∧ final_savings = initial_savings * ratio ^ (months - 1) :=
sorry

end NUMINAMATH_CALUDE_savings_ratio_l760_76016


namespace NUMINAMATH_CALUDE_max_value_quadratic_inequality_l760_76032

/-- Given that the solution set of ax^2 + 2x + c ≤ 0 is {x | x = -1/a} and a > c,
    prove that the maximum value of (a-c)/(a^2+c^2) is √2/4 -/
theorem max_value_quadratic_inequality (a c : ℝ) 
    (h1 : ∀ x, a * x^2 + 2 * x + c ≤ 0 ↔ x = -1/a)
    (h2 : a > c) :
    (∀ a' c', a' > c' → (a' - c') / (a'^2 + c'^2) ≤ Real.sqrt 2 / 4) ∧ 
    (∃ a' c', a' > c' ∧ (a' - c') / (a'^2 + c'^2) = Real.sqrt 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_inequality_l760_76032


namespace NUMINAMATH_CALUDE_least_divisible_by_three_l760_76059

theorem least_divisible_by_three (x : ℕ) : (∃ y : ℕ, y > 0 ∧ 23 * y % 3 = 0) → x ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_three_l760_76059


namespace NUMINAMATH_CALUDE_only_zero_factorizable_l760_76084

/-- The polynomial we're considering -/
def poly (m : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + x + m*y - 2*m

/-- A linear factor with integer coefficients -/
def linear_factor (a b c : ℤ) (x y : ℤ) : ℤ := a*x + b*y + c

/-- Predicate to check if the polynomial can be factored into two linear factors with integer coefficients -/
def can_be_factored (m : ℤ) : Prop :=
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℤ), ∀ (x y : ℤ),
    poly m x y = linear_factor a₁ b₁ c₁ x y * linear_factor a₂ b₂ c₂ x y

theorem only_zero_factorizable :
  ∀ m : ℤ, can_be_factored m ↔ m = 0 :=
sorry

end NUMINAMATH_CALUDE_only_zero_factorizable_l760_76084


namespace NUMINAMATH_CALUDE_smallest_divisible_by_495_l760_76088

/-- Represents a number in the sequence with n digits of 5 -/
def sequenceNumber (n : ℕ) : ℕ :=
  (10^n - 1) / 9 * 5

/-- The target number we want to prove is the smallest divisible by 495 -/
def targetNumber : ℕ := sequenceNumber 18

/-- Checks if a number is in the sequence -/
def isInSequence (k : ℕ) : Prop :=
  ∃ n : ℕ, sequenceNumber n = k

theorem smallest_divisible_by_495 :
  (targetNumber % 495 = 0) ∧
  (∀ k : ℕ, k < targetNumber → isInSequence k → k % 495 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_495_l760_76088


namespace NUMINAMATH_CALUDE_water_in_bucket_l760_76066

theorem water_in_bucket (initial_amount poured_out : ℚ) 
  (h1 : initial_amount = 15/8)
  (h2 : poured_out = 9/8) : 
  initial_amount - poured_out = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l760_76066


namespace NUMINAMATH_CALUDE_anna_left_probability_l760_76063

/-- The probability that the girl on the right is lying -/
def P_right_lying : ℚ := 1/4

/-- The probability that the girl on the left is lying -/
def P_left_lying : ℚ := 1/5

/-- The event that Anna is sitting on the left -/
def A : Prop := sorry

/-- The event that both girls claim to be Brigitte -/
def B : Prop := sorry

/-- The probability of event A given event B -/
def P_A_given_B : ℚ := sorry

theorem anna_left_probability : P_A_given_B = 3/7 := by sorry

end NUMINAMATH_CALUDE_anna_left_probability_l760_76063


namespace NUMINAMATH_CALUDE_seven_lines_regions_l760_76075

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- Seven lines in a plane with no two parallel and no three concurrent -/
def seven_lines : ℕ := 7

theorem seven_lines_regions :
  regions seven_lines = 29 := by sorry

end NUMINAMATH_CALUDE_seven_lines_regions_l760_76075


namespace NUMINAMATH_CALUDE_least_prime_factor_of_8_pow_4_minus_8_pow_3_l760_76079

theorem least_prime_factor_of_8_pow_4_minus_8_pow_3 :
  Nat.minFac (8^4 - 8^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_8_pow_4_minus_8_pow_3_l760_76079


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l760_76089

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 6| = 3*x + 6) ↔ (x = 0) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l760_76089


namespace NUMINAMATH_CALUDE_vector_sum_equality_implies_same_direction_l760_76001

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def same_direction (a b : n) : Prop := ∃ (k : ℝ), k > 0 ∧ a = k • b

theorem vector_sum_equality_implies_same_direction (a b : n) 
  (ha : a ≠ 0) (hb : b ≠ 0) (h : ‖a + b‖ = ‖a‖ + ‖b‖) :
  same_direction a b := by sorry

end NUMINAMATH_CALUDE_vector_sum_equality_implies_same_direction_l760_76001


namespace NUMINAMATH_CALUDE_cake_box_width_l760_76017

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

theorem cake_box_width :
  let carton := BoxDimensions.mk 25 42 60
  let cakeBox := BoxDimensions.mk 8 W 5
  let maxBoxes := 210
  boxVolume carton = maxBoxes * boxVolume cakeBox →
  W = 7.5 := by
sorry

end NUMINAMATH_CALUDE_cake_box_width_l760_76017


namespace NUMINAMATH_CALUDE_semicircle_radius_l760_76039

theorem semicircle_radius (D E F : ℝ × ℝ) : 
  -- Triangle DEF has a right angle at D
  (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0 →
  -- Area of semicircle on DE = 12.5π
  (1/2) * Real.pi * ((E.1 - D.1)^2 + (E.2 - D.2)^2) / 4 = 12.5 * Real.pi →
  -- Arc length of semicircle on DF = 7π
  Real.pi * ((F.1 - D.1)^2 + (F.2 - D.2)^2).sqrt / 2 = 7 * Real.pi →
  -- The radius of the semicircle on EF is √74
  ((E.1 - F.1)^2 + (E.2 - F.2)^2).sqrt / 2 = Real.sqrt 74 := by
sorry

end NUMINAMATH_CALUDE_semicircle_radius_l760_76039


namespace NUMINAMATH_CALUDE_no_snow_probability_l760_76087

def probability_of_snow : ℚ := 2/3

def days : ℕ := 5

theorem no_snow_probability :
  (1 - probability_of_snow) ^ days = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l760_76087


namespace NUMINAMATH_CALUDE_mass_of_man_equals_240kg_l760_76065

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating that the mass of the man is 240 kg under given conditions. -/
theorem mass_of_man_equals_240kg 
  (boat_length : ℝ) 
  (boat_breadth : ℝ) 
  (boat_sinking : ℝ) 
  (water_density : ℝ) 
  (h1 : boat_length = 8) 
  (h2 : boat_breadth = 3) 
  (h3 : boat_sinking = 0.01) 
  (h4 : water_density = 1000) :
  mass_of_man boat_length boat_breadth boat_sinking water_density = 240 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_equals_240kg_l760_76065


namespace NUMINAMATH_CALUDE_complex_division_simplification_l760_76048

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  let z : ℂ := 5 / (2 - i)
  z = 2 + i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l760_76048


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l760_76049

/-- Given a street with parallel curbs and a crosswalk, prove the distance between stripes. -/
theorem crosswalk_stripe_distance
  (curb_distance : ℝ)
  (curb_length : ℝ)
  (stripe_length : ℝ)
  (h_curb_distance : curb_distance = 60)
  (h_curb_length : curb_length = 20)
  (h_stripe_length : stripe_length = 75) :
  curb_length * curb_distance / stripe_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l760_76049


namespace NUMINAMATH_CALUDE_complex_sum_real_imag_l760_76024

theorem complex_sum_real_imag (a : ℝ) : 
  let z : ℂ := a / (2 + Complex.I) + (2 + Complex.I) / 5
  (z.re + z.im = 1) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_real_imag_l760_76024


namespace NUMINAMATH_CALUDE_optimal_pencil_purchase_l760_76010

theorem optimal_pencil_purchase : ∀ (x y : ℕ),
  -- Cost constraint
  27 * x + 23 * y ≤ 940 →
  -- Difference constraint
  y ≤ x + 10 →
  -- Optimality
  (∀ (x' y' : ℕ), 27 * x' + 23 * y' ≤ 940 → y' ≤ x' + 10 → x' + y' ≤ x + y) →
  -- Minimizing red pencils
  (∀ (x' : ℕ), x' < x → ∃ (y' : ℕ), 27 * x' + 23 * y' ≤ 940 ∧ y' ≤ x' + 10 ∧ x' + y' < x + y) →
  x = 14 ∧ y = 24 :=
by sorry

end NUMINAMATH_CALUDE_optimal_pencil_purchase_l760_76010
