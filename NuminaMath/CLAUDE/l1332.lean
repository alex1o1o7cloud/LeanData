import Mathlib

namespace NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1332_133211

/-- Represents a system of two linear equations with two unknowns,
    where the coefficients form an arithmetic progression. -/
structure ArithmeticProgressionSystem where
  a : ℝ
  d : ℝ

/-- The solution to the system of linear equations. -/
def solution : ℝ × ℝ := (-1, 2)

/-- Checks if the given pair (x, y) satisfies the first equation of the system. -/
def satisfies_equation1 (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) : Prop :=
  sys.a * sol.1 + (sys.a + sys.d) * sol.2 = sys.a + 2 * sys.d

/-- Checks if the given pair (x, y) satisfies the second equation of the system. -/
def satisfies_equation2 (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) : Prop :=
  (sys.a + 3 * sys.d) * sol.1 + (sys.a + 4 * sys.d) * sol.2 = sys.a + 5 * sys.d

/-- Theorem stating that the solution satisfies both equations of the system. -/
theorem solution_satisfies_system (sys : ArithmeticProgressionSystem) :
  satisfies_equation1 sys solution ∧ satisfies_equation2 sys solution :=
sorry

/-- Theorem stating that the solution is unique. -/
theorem solution_is_unique (sys : ArithmeticProgressionSystem) (sol : ℝ × ℝ) :
  satisfies_equation1 sys sol ∧ satisfies_equation2 sys sol → sol = solution :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_solution_is_unique_l1332_133211


namespace NUMINAMATH_CALUDE_function_symmetry_periodicity_l1332_133229

theorem function_symmetry_periodicity 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f x = -f (2 - x)) : 
  ∀ x, f (x + 4) = f x := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_periodicity_l1332_133229


namespace NUMINAMATH_CALUDE_profit_decrease_l1332_133292

theorem profit_decrease (P : ℝ) (x : ℝ) : 
  (P + 0.10 * P) * (1 - x / 100) * 1.50 = P * 1.3200000000000003 →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_decrease_l1332_133292


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1332_133256

theorem quadratic_factorization (b k : ℝ) : 
  (∀ x, x^2 + b*x + 5 = (x - 2)^2 + k) → (b = -4 ∧ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1332_133256


namespace NUMINAMATH_CALUDE_eight_digit_number_bounds_l1332_133286

/-- Given an 8-digit number A derived from B, prove the max and min values of A -/
theorem eight_digit_number_bounds (A B : ℕ) : 
  (∃ d : ℕ, d < 10 ∧ A = d * 10^7 + (B - d) / 10) →  -- A is derived from B
  (∀ m : ℕ, m ∣ B ∧ m ∣ 12 → m = 1) →  -- B is coprime with 12
  B > 44444444 →  -- B > 44444444
  (∃ A_max A_min : ℕ, 
    (∀ A' : ℕ, (∃ B' : ℕ, (∀ m : ℕ, m ∣ B' ∧ m ∣ 12 → m = 1) ∧ 
                B' > 44444444 ∧ 
                (∃ d : ℕ, d < 10 ∧ A' = d * 10^7 + (B' - d) / 10)) 
    → A' ≤ A_max) ∧
    (∀ A' : ℕ, (∃ B' : ℕ, (∀ m : ℕ, m ∣ B' ∧ m ∣ 12 → m = 1) ∧ 
                B' > 44444444 ∧ 
                (∃ d : ℕ, d < 10 ∧ A' = d * 10^7 + (B' - d) / 10)) 
    → A' ≥ A_min) ∧
    A_max = 99999998 ∧ 
    A_min = 14444446) :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_number_bounds_l1332_133286


namespace NUMINAMATH_CALUDE_relay_race_time_difference_l1332_133264

theorem relay_race_time_difference 
  (total_time : ℕ) 
  (jen_time : ℕ) 
  (susan_time : ℕ) 
  (mary_time : ℕ) 
  (tiffany_time : ℕ) :
  total_time = 223 →
  jen_time = 30 →
  susan_time = jen_time + 10 →
  mary_time = 2 * susan_time →
  tiffany_time < mary_time →
  total_time = mary_time + susan_time + jen_time + tiffany_time →
  mary_time - tiffany_time = 7 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_time_difference_l1332_133264


namespace NUMINAMATH_CALUDE_unique_solution_l1332_133263

/-- Represents a number of the form 13xy4.5z -/
def SpecialNumber (x y z : ℕ) : ℚ :=
  13000 + 100 * x + 10 * y + 4 + 0.5 + 0.01 * z

theorem unique_solution :
  ∃! (x y z : ℕ),
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    (∃ (k : ℕ), SpecialNumber x y z = k * 792) ∧
    (45 + z) % 8 = 0 ∧
    (1 + 3 + x + y + 4 + 5 + z) % 9 = 0 ∧
    (1 - 3 + x - y + 4 - 5 + z) % 11 = 0 ∧
    SpecialNumber x y z = 13804.56 :=
by
  sorry

#eval SpecialNumber 8 0 6  -- Should output 13804.56

end NUMINAMATH_CALUDE_unique_solution_l1332_133263


namespace NUMINAMATH_CALUDE_random_event_last_third_probability_l1332_133291

/-- The probability of a random event occurring in the last third of a given time interval is 1/3 -/
theorem random_event_last_third_probability (total_interval : ℝ) (h : total_interval > 0) :
  let last_third := total_interval / 3
  (last_third / total_interval) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_random_event_last_third_probability_l1332_133291


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1332_133290

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  (total_players - throwers) % 3 = 0 →
  56 = throwers + (total_players - throwers) - (total_players - throwers) / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1332_133290


namespace NUMINAMATH_CALUDE_ellipse_line_and_fixed_circle_l1332_133230

/-- Given an ellipse C and points P, Q, and conditions for line l, prove the equation of l and that point S lies on a fixed circle. -/
theorem ellipse_line_and_fixed_circle 
  (x₀ y₀ : ℝ) 
  (hy₀ : y₀ ≠ 0)
  (hP : x₀^2/4 + y₀^2/3 = 1) 
  (Q : ℝ × ℝ)
  (hQ : Q = (x₀/4, y₀/3))
  (l : Set (ℝ × ℝ))
  (hl : ∀ M ∈ l, (M.1 - x₀) * (x₀/4) + (M.2 - y₀) * (y₀/3) = 0)
  (F : ℝ × ℝ)
  (hF : F.1 > 0 ∧ F.1^2 = 1 + F.2^2/3)  -- Condition for right focus
  (S : ℝ × ℝ)
  (hS : ∃ k, S = (4 + k * (4*y₀)/(3*x₀), k) ∧ 
             S.2 = (y₀/(x₀-1)) * (S.1 - 1)) :
  (∀ x y, (x, y) ∈ l ↔ x₀*x/4 + y₀*y/3 = 1) ∧ 
  ((S.1 - 1)^2 + S.2^2 = 36) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_line_and_fixed_circle_l1332_133230


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l1332_133262

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of a line segment with endpoints (10, π/4) and (10, 3π/4) in polar coordinates is (5√2, π/2) --/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 := by sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l1332_133262


namespace NUMINAMATH_CALUDE_water_to_dean_height_ratio_l1332_133276

-- Define the heights and water depth
def ron_height : ℝ := 14
def height_difference : ℝ := 8
def water_depth : ℝ := 12

-- Define Dean's height
def dean_height : ℝ := ron_height - height_difference

-- Theorem statement
theorem water_to_dean_height_ratio :
  (water_depth / dean_height) = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_to_dean_height_ratio_l1332_133276


namespace NUMINAMATH_CALUDE_siblings_ratio_l1332_133254

/-- Given the number of siblings for Masud, Janet, and Carlos, prove the ratio of Carlos's to Masud's siblings -/
theorem siblings_ratio (masud_siblings : ℕ) (janet_siblings : ℕ) (carlos_siblings : ℕ) : 
  masud_siblings = 60 →
  janet_siblings = 4 * masud_siblings - 60 →
  janet_siblings = carlos_siblings + 135 →
  carlos_siblings * 4 = masud_siblings * 3 := by
  sorry

#check siblings_ratio

end NUMINAMATH_CALUDE_siblings_ratio_l1332_133254


namespace NUMINAMATH_CALUDE_lowest_cost_l1332_133274

variable (x y z a b c : ℝ)

/-- The painting areas of the three rooms satisfy x < y < z -/
axiom area_order : x < y ∧ y < z

/-- The painting costs of the three colors satisfy a < b < c -/
axiom cost_order : a < b ∧ b < c

/-- The total cost function for a painting scheme -/
def total_cost (p q r : ℝ) : ℝ := p*x + q*y + r*z

/-- The theorem stating that az + by + cx is the lowest total cost -/
theorem lowest_cost : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = a ∧ q = c ∧ r = b) ∨ 
                  (p = b ∧ q = a ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ 
                  (p = c ∧ q = a ∧ r = b) ∨ (p = c ∧ q = b ∧ r = a) →
                  total_cost a b c ≤ total_cost p q r :=
by sorry

end NUMINAMATH_CALUDE_lowest_cost_l1332_133274


namespace NUMINAMATH_CALUDE_relationship_abc_l1332_133225

theorem relationship_abc (x : ℝ) (a b c : ℝ) 
  (h1 : x > Real.exp (-1)) 
  (h2 : x < 1) 
  (h3 : a = Real.log x) 
  (h4 : b = (1/2) ^ (Real.log x)) 
  (h5 : c = Real.exp (Real.log x)) : 
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1332_133225


namespace NUMINAMATH_CALUDE_problem_l1332_133267

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

theorem problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  b * Real.cos c / a = -1 := by sorry

end NUMINAMATH_CALUDE_problem_l1332_133267


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_sixth_l1332_133238

theorem units_digit_of_six_to_sixth (n : ℕ) : n = 6^6 → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_sixth_l1332_133238


namespace NUMINAMATH_CALUDE_apple_cost_graph_properties_l1332_133261

def apple_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n else 18 * n

theorem apple_cost_graph_properties :
  ∃ (f : ℕ → ℚ),
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → f n = apple_cost n) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n < 10 → f (n + 1) - f n = 20) ∧
    (∀ n : ℕ, 10 < n ∧ n < 20 → f (n + 1) - f n = 18) ∧
    (f 11 - f 10 ≠ 20 ∧ f 11 - f 10 ≠ 18) :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_graph_properties_l1332_133261


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l1332_133223

-- Define the slopes of the two lines
def slope1 : ℝ := 6
def slope2 (c : ℝ) : ℝ := 3 * c

-- Define the condition for parallel lines
def are_parallel (c : ℝ) : Prop := slope1 = slope2 c

-- Theorem statement
theorem parallel_lines_c_value :
  ∃ c : ℝ, are_parallel c ∧ c = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l1332_133223


namespace NUMINAMATH_CALUDE_bobby_total_blocks_l1332_133253

def bobby_blocks : ℕ := 2
def father_gift : ℕ := 6

theorem bobby_total_blocks :
  bobby_blocks + father_gift = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_blocks_l1332_133253


namespace NUMINAMATH_CALUDE_total_votes_calculation_l1332_133239

theorem total_votes_calculation (V : ℝ) 
  (h1 : 0.3 * V + (0.3 * V + 1760) = V) : V = 4400 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_calculation_l1332_133239


namespace NUMINAMATH_CALUDE_subtracting_and_dividing_l1332_133273

theorem subtracting_and_dividing (x : ℝ) : x = 32 → (x - 6) / 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtracting_and_dividing_l1332_133273


namespace NUMINAMATH_CALUDE_same_solution_value_of_c_l1332_133287

theorem same_solution_value_of_c : ∃ (c : ℝ), 
  (∃ (x : ℝ), 3 * x + 5 = 1 ∧ c * x + 15 = 3) → c = 9 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_value_of_c_l1332_133287


namespace NUMINAMATH_CALUDE_age_difference_proof_l1332_133270

def elder_age : ℕ := 30

theorem age_difference_proof (younger_age : ℕ) 
  (h : elder_age - 6 = 3 * (younger_age - 6)) : 
  elder_age - younger_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1332_133270


namespace NUMINAMATH_CALUDE_johnny_first_job_hours_l1332_133284

/-- Represents Johnny's work schedule and earnings --/
structure WorkSchedule where
  hourlyRate1 : ℝ
  hourlyRate2 : ℝ
  hourlyRate3 : ℝ
  hours2 : ℝ
  hours3 : ℝ
  daysWorked : ℝ
  totalEarnings : ℝ

/-- Theorem stating that given the conditions, Johnny worked 3 hours on the first job each day --/
theorem johnny_first_job_hours (schedule : WorkSchedule)
  (h1 : schedule.hourlyRate1 = 7)
  (h2 : schedule.hourlyRate2 = 10)
  (h3 : schedule.hourlyRate3 = 12)
  (h4 : schedule.hours2 = 2)
  (h5 : schedule.hours3 = 4)
  (h6 : schedule.daysWorked = 5)
  (h7 : schedule.totalEarnings = 445) :
  ∃ (x : ℝ), x = 3 ∧ 
    schedule.daysWorked * (schedule.hourlyRate1 * x + 
      schedule.hourlyRate2 * schedule.hours2 + 
      schedule.hourlyRate3 * schedule.hours3) = schedule.totalEarnings :=
by
  sorry

end NUMINAMATH_CALUDE_johnny_first_job_hours_l1332_133284


namespace NUMINAMATH_CALUDE_remainder_3_pow_1999_mod_13_l1332_133214

theorem remainder_3_pow_1999_mod_13 : 3^1999 % 13 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_3_pow_1999_mod_13_l1332_133214


namespace NUMINAMATH_CALUDE_suits_sold_is_two_l1332_133255

/-- The number of suits sold given the commission rate, shirt sales, loafer sales, and total commission earned. -/
def suits_sold (commission_rate : ℚ) (num_shirts : ℕ) (shirt_price : ℚ) (num_loafers : ℕ) (loafer_price : ℚ) (suit_price : ℚ) (total_commission : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the number of suits sold is 2 under the given conditions. -/
theorem suits_sold_is_two :
  suits_sold (15 / 100) 6 50 2 150 700 300 = 2 := by
  sorry

end NUMINAMATH_CALUDE_suits_sold_is_two_l1332_133255


namespace NUMINAMATH_CALUDE_sphere_volume_l1332_133275

theorem sphere_volume (r : ℝ) (d V : ℝ) (h : d = (16 / 9 * V) ^ (1 / 3)) (h_r : r = 1 / 3) : V = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l1332_133275


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1332_133213

/-- Given a line passing through points (1, -3) and (-1, 3), 
    prove that the sum of its slope and y-intercept is -3 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → 
    ((x = 1 ∧ y = -3) ∨ (x = -1 ∧ y = 3))) → 
  m + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1332_133213


namespace NUMINAMATH_CALUDE_linear_programming_problem_l1332_133269

theorem linear_programming_problem (x y a b : ℝ) :
  3 * x - y - 6 ≤ 0 →
  x - y + 2 ≥ 0 →
  x ≥ 0 →
  y ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 3 * x' - y' - 6 ≤ 0 → x' - y' + 2 ≥ 0 → x' ≥ 0 → y' ≥ 0 → a * x' + b * y' ≤ 12) →
  a * x + b * y = 12 →
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

end NUMINAMATH_CALUDE_linear_programming_problem_l1332_133269


namespace NUMINAMATH_CALUDE_unique_solution_l1332_133219

theorem unique_solution : ∃! x : ℚ, x * 8 / 3 - (2 + 3) * 2 = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1332_133219


namespace NUMINAMATH_CALUDE_gravel_path_cost_example_l1332_133281

/-- Calculates the cost of gravelling a path inside a rectangular plot -/
def gravel_path_cost (length width path_width gravel_cost_per_sqm : ℝ) : ℝ :=
  let total_area := length * width
  let inner_area := (length - 2 * path_width) * (width - 2 * path_width)
  let path_area := total_area - inner_area
  path_area * gravel_cost_per_sqm

/-- Theorem: The cost of gravelling the path is 425 INR -/
theorem gravel_path_cost_example : 
  gravel_path_cost 110 65 2.5 0.5 = 425 := by
sorry

end NUMINAMATH_CALUDE_gravel_path_cost_example_l1332_133281


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l1332_133252

theorem sin_pi_minus_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α + π / 3) = 3 / 5) : 
  Real.sin (π - α) = (3 + 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l1332_133252


namespace NUMINAMATH_CALUDE_max_m_value_l1332_133209

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_m_value : 
  (∃ (m : ℝ), m = 4 ∧ 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x)) ∧
  (∀ (m : ℝ), m > 4 → 
    (∀ (t : ℝ), ∃ (x : ℝ), x ∈ Set.Icc 1 m ∧ f (x + t) > x)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1332_133209


namespace NUMINAMATH_CALUDE_melanie_dimes_l1332_133210

theorem melanie_dimes (initial : ℕ) (from_dad : ℕ) (total : ℕ) : 
  initial = 7 → from_dad = 8 → total = 19 → total - (initial + from_dad) = 4 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1332_133210


namespace NUMINAMATH_CALUDE_candy_distribution_l1332_133297

/-- Candy distribution problem -/
theorem candy_distribution (tabitha stan julie carlos : ℕ) : 
  tabitha = 22 →
  stan = 13 →
  julie = tabitha / 2 →
  tabitha + stan + julie + carlos = 72 →
  carlos / stan = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1332_133297


namespace NUMINAMATH_CALUDE_sin_630_degrees_l1332_133226

theorem sin_630_degrees : Real.sin (630 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_630_degrees_l1332_133226


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l1332_133206

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), (∀ x ∈ S, ∃ c d : ℕ+, (Nat.gcd c d * Nat.lcm c d = 360 ∧ Nat.gcd c d = x)) ∧ 
                      (∀ y : ℕ, (∃ e f : ℕ+, (Nat.gcd e f * Nat.lcm e f = 360 ∧ Nat.gcd e f = y)) → y ∈ S) ∧
                      S.card = 12) := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l1332_133206


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1332_133234

theorem quadratic_equation_coefficients :
  ∀ (a b c d e f : ℝ),
  (∀ x, a * x^2 + b * x + c = d * x^2 + e * x + f) →
  (a - d = 4) →
  (b - e = -2 ∧ c - f = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1332_133234


namespace NUMINAMATH_CALUDE_range_of_f_less_than_zero_l1332_133282

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f y ≤ f x

-- State the theorem
theorem range_of_f_less_than_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : decreasing_on_nonpositive f)
  (h_f_neg_two : f (-2) = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_less_than_zero_l1332_133282


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l1332_133227

def g (x : ℝ) : ℝ := (x - 3)^2 - 7

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l1332_133227


namespace NUMINAMATH_CALUDE_sector_central_angle_l1332_133259

/-- Given a circular sector with perimeter 4 and area 1, 
    its central angle measure is 2 radians. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1332_133259


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1332_133250

theorem smallest_integer_with_given_remainders :
  ∃ (n : ℕ), n > 0 ∧
    n % 5 = 4 ∧
    n % 7 = 5 ∧
    n % 11 = 9 ∧
    n % 13 = 11 ∧
    (∀ m : ℕ, m > 0 ∧
      m % 5 = 4 ∧
      m % 7 = 5 ∧
      m % 11 = 9 ∧
      m % 13 = 11 → m ≥ n) ∧
    n = 999 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1332_133250


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l1332_133207

theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_diff = 1/6 →
  ∃ (speed_B : ℝ),
    distance / speed_B - distance / (speed_ratio * speed_B) = time_diff ∧
    speed_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_speed_problem_l1332_133207


namespace NUMINAMATH_CALUDE_parabola_uniqueness_l1332_133222

/-- A parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.tangent_slope (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem parabola_uniqueness (p : Parabola) :
  p.y_coord 1 = 1 →
  p.y_coord 2 = -1 →
  p.tangent_slope 2 = 1 →
  p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_uniqueness_l1332_133222


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l1332_133212

-- Define the points
variable (A B C D E F : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define E as the intersection of angle bisectors of ∠B and ∠C
def is_angle_bisector_intersection (E B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define F as the intersection of AB and CD
def is_line_intersection (F A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the condition AB + CD = BC
def sum_equals_side (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := 
  dist A B + dist C D = dist B C

-- Define cyclic quadrilateral
def is_cyclic (A D E F : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Theorem statement
theorem cyclic_quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_angle_bisector_intersection E B C)
  (h3 : is_line_intersection F A B C D)
  (h4 : sum_equals_side A B C D) :
  is_cyclic A D E F := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l1332_133212


namespace NUMINAMATH_CALUDE_juice_bar_spending_l1332_133278

theorem juice_bar_spending (mango_price pineapple_price pineapple_total group_size : ℕ) 
  (h1 : mango_price = 5)
  (h2 : pineapple_price = 6)
  (h3 : pineapple_total = 54)
  (h4 : group_size = 17) :
  ∃ (mango_glasses pineapple_glasses : ℕ),
    mango_glasses + pineapple_glasses = group_size ∧
    mango_glasses * mango_price + pineapple_glasses * pineapple_price = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_juice_bar_spending_l1332_133278


namespace NUMINAMATH_CALUDE_geometry_theorem_l1332_133240

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def contains (p : Plane) (l : Line) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) (l : Line) : Prop := sorry

-- State the theorem
theorem geometry_theorem 
  (m n l : Line) 
  (α β γ : Plane) 
  (hm : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (hα : α ≠ β ∧ α ≠ γ ∧ β ≠ γ) :
  (contains α m ∧ contains β n ∧ perpendicular_planes α β ∧ 
   intersection α β l ∧ perpendicular_lines m l → perpendicular_lines m n) ∧
  (parallel_planes α γ ∧ parallel_planes β γ → parallel_planes α β) := by
  sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1332_133240


namespace NUMINAMATH_CALUDE_f_contraction_implies_a_bound_l1332_133258

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x^2 + 1

-- State the theorem
theorem f_contraction_implies_a_bound
  (a : ℝ)
  (h_a_neg : a < 0)
  (h_contraction : ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    |f a x₁ - f a x₂| ≥ |x₁ - x₂|) :
  a ≤ -1/8 := by
  sorry

end NUMINAMATH_CALUDE_f_contraction_implies_a_bound_l1332_133258


namespace NUMINAMATH_CALUDE_problem_solution_l1332_133288

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem problem_solution (a : ℝ) : f a (f a 0) = 4*a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1332_133288


namespace NUMINAMATH_CALUDE_inspector_group_b_count_l1332_133204

/-- Represents the problem of determining the number of inspectors in Group B -/
theorem inspector_group_b_count : 
  ∀ (a b : ℕ) (group_b_count : ℕ),
  a > 0 → b > 0 →
  (2 * (a + 2 * b)) / 2 = (2 * (a + 5 * b)) / 3 →  -- Equation from Group A's work
  (5 * (a + 5 * b)) / (group_b_count * 5) = (2 * (a + 2 * b)) / (8 * 2) →  -- Equation comparing Group A and B's work
  group_b_count = 12 := by
    sorry


end NUMINAMATH_CALUDE_inspector_group_b_count_l1332_133204


namespace NUMINAMATH_CALUDE_committee_with_chair_count_l1332_133221

theorem committee_with_chair_count : 
  let total_students : ℕ := 8
  let committee_size : ℕ := 5
  let committee_count : ℕ := Nat.choose total_students committee_size
  let chair_choices : ℕ := committee_size
  committee_count * chair_choices = 280 := by
sorry

end NUMINAMATH_CALUDE_committee_with_chair_count_l1332_133221


namespace NUMINAMATH_CALUDE_triangle_PPB_area_l1332_133243

/-- A square with side length 10 inches -/
def square_side : ℝ := 10

/-- Point P is a vertex of the square -/
def P : ℝ × ℝ := (0, 0)

/-- Point B is on the side of the square -/
def B : ℝ × ℝ := (square_side, 0)

/-- Point Q is inside the square and 8 inches above P -/
def Q : ℝ × ℝ := (0, 8)

/-- PQ is perpendicular to PB -/
axiom PQ_perp_PB : (Q.1 - P.1) * (B.1 - P.1) + (Q.2 - P.2) * (B.2 - P.2) = 0

/-- The area of triangle PPB -/
def triangle_area : ℝ := 0.5 * square_side * 8

/-- Theorem: The area of triangle PPB is 40 square inches -/
theorem triangle_PPB_area : triangle_area = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_PPB_area_l1332_133243


namespace NUMINAMATH_CALUDE_big_sale_commission_proof_l1332_133242

/-- Calculates the commission amount for a big sale given the following conditions:
  * new_average: Matt's new average commission after the big sale
  * total_sales: Total number of sales including the big sale
  * average_increase: The amount by which the big sale raised the average commission
-/
def big_sale_commission (new_average : ℚ) (total_sales : ℕ) (average_increase : ℚ) : ℚ :=
  new_average * total_sales - (new_average - average_increase) * (total_sales - 1)

/-- Theorem stating that given Matt's new average commission is $250, he has made 6 sales,
    and the big sale commission raises his average by $150, the commission amount for the
    big sale is $1000. -/
theorem big_sale_commission_proof :
  big_sale_commission 250 6 150 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_big_sale_commission_proof_l1332_133242


namespace NUMINAMATH_CALUDE_no_inverse_implies_x_equals_five_l1332_133272

def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 5],
    ![6, 6]]

theorem no_inverse_implies_x_equals_five :
  ∀ x : ℝ, ¬(IsUnit (M x)) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_inverse_implies_x_equals_five_l1332_133272


namespace NUMINAMATH_CALUDE_half_of_a_l1332_133296

theorem half_of_a (a : ℝ) : (1 / 2) * a = a / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_of_a_l1332_133296


namespace NUMINAMATH_CALUDE_opposite_of_three_l1332_133294

theorem opposite_of_three : -(3 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l1332_133294


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l1332_133277

theorem sphere_volume_increase (r₁ r₂ : ℝ) (h : r₂ = 2 * r₁) : 
  (4 / 3) * π * r₂^3 = 8 * ((4 / 3) * π * r₁^3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l1332_133277


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_thirteen_thirds_l1332_133231

/-- Given two functions f and g, where f is linear and g(f(x)) = 4x + 2,
    prove that the sum of coefficients of f is 13/3 -/
theorem sum_of_coefficients_is_thirteen_thirds
  (f g : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = a * x + b)
  (h2 : ∀ x, g x = 3 * x - 7)
  (h3 : ∀ x, g (f x) = 4 * x + 2) :
  a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_thirteen_thirds_l1332_133231


namespace NUMINAMATH_CALUDE_quadratic_term_elimination_l1332_133220

theorem quadratic_term_elimination (m : ℝ) : 
  (∀ x : ℝ, 36 * x^2 - 3 * x + 5 - (-3 * x^3 - 12 * m * x^2 + 5 * x - 7) = 3 * x^3 - 8 * x + 12) → 
  m^3 = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_term_elimination_l1332_133220


namespace NUMINAMATH_CALUDE_dance_steps_time_l1332_133266

def time_step1 : ℕ := 30

def time_step2 (t1 : ℕ) : ℕ := t1 / 2

def time_step3 (t1 t2 : ℕ) : ℕ := t1 + t2

def total_time (t1 t2 t3 : ℕ) : ℕ := t1 + t2 + t3

theorem dance_steps_time :
  total_time time_step1 (time_step2 time_step1) (time_step3 time_step1 (time_step2 time_step1)) = 90 :=
by sorry

end NUMINAMATH_CALUDE_dance_steps_time_l1332_133266


namespace NUMINAMATH_CALUDE_mary_candy_ratio_l1332_133235

/-- The number of times Mary initially has more candy than Megan -/
def candy_ratio (megan_candy : ℕ) (mary_final_candy : ℕ) (mary_added_candy : ℕ) : ℚ :=
  (mary_final_candy - mary_added_candy : ℚ) / megan_candy

theorem mary_candy_ratio :
  candy_ratio 5 25 10 = 3 := by sorry

end NUMINAMATH_CALUDE_mary_candy_ratio_l1332_133235


namespace NUMINAMATH_CALUDE_parametric_to_equation_l1332_133215

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  constant : Point3D
  direction1 : Point3D
  direction2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if the given coefficients satisfy the required conditions -/
def validCoefficients (eq : PlaneEquation) : Prop :=
  eq.A > 0 ∧ Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.B) = 1 ∧
  Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.C) = 1 ∧
  Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.D) = 1

/-- Check if a point satisfies the plane equation -/
def satisfiesEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The main theorem to prove -/
theorem parametric_to_equation (plane : ParametricPlane) :
  ∃ (eq : PlaneEquation),
    validCoefficients eq ∧
    (∀ (s t : ℝ),
      let p : Point3D := {
        x := plane.constant.x + s * plane.direction1.x + t * plane.direction2.x,
        y := plane.constant.y + s * plane.direction1.y + t * plane.direction2.y,
        z := plane.constant.z + s * plane.direction1.z + t * plane.direction2.z
      }
      satisfiesEquation p eq) ∧
    eq.A = 2 ∧ eq.B = -5 ∧ eq.C = 2 ∧ eq.D = -7 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_equation_l1332_133215


namespace NUMINAMATH_CALUDE_garden_perimeter_l1332_133224

/-- A rectangular garden with given diagonal and area has a specific perimeter -/
theorem garden_perimeter (x y : ℝ) (h_rectangle : x > 0 ∧ y > 0) 
  (h_diagonal : x^2 + y^2 = 34^2) (h_area : x * y = 240) : 
  2 * (x + y) = 80 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1332_133224


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1332_133293

/-- Given that x and y are inversely proportional, prove that if x = 40 when y = 8, then x = 16 when y = 20 -/
theorem inverse_proportion_problem (x y : ℝ) (h : x * y = (40 : ℝ) * 8) :
  x * 20 = (40 : ℝ) * 8 → x = 16 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1332_133293


namespace NUMINAMATH_CALUDE_reverse_order_product_sum_l1332_133251

/-- Checks if two positive integers have reverse digit order -/
def are_reverse_order (a b : ℕ) : Prop := sorry

/-- Given two positive integers m and n with reverse digit order and m * n = 1446921630, prove m + n = 79497 -/
theorem reverse_order_product_sum (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : are_reverse_order m n) 
  (h4 : m * n = 1446921630) : 
  m + n = 79497 := by sorry

end NUMINAMATH_CALUDE_reverse_order_product_sum_l1332_133251


namespace NUMINAMATH_CALUDE_age_sum_in_five_years_l1332_133249

/-- Given a person (Mike) who is 30 years younger than his mom, and the sum of their ages is 70 years,
    the sum of their ages in 5 years will be 80 years. -/
theorem age_sum_in_five_years (mike_age mom_age : ℕ) : 
  mike_age = mom_age - 30 → 
  mike_age + mom_age = 70 → 
  (mike_age + 5) + (mom_age + 5) = 80 := by
sorry

end NUMINAMATH_CALUDE_age_sum_in_five_years_l1332_133249


namespace NUMINAMATH_CALUDE_complex_geometry_problem_l1332_133201

/-- A complex number z with specific properties -/
def z : ℂ :=
  sorry

/-- The condition that |z| = √2 -/
axiom z_norm : Complex.abs z = Real.sqrt 2

/-- The condition that the imaginary part of z² is 2 -/
axiom z_sq_im : Complex.im (z ^ 2) = 2

/-- The condition that z is in the first quadrant -/
axiom z_first_quadrant : Complex.re z > 0 ∧ Complex.im z > 0

/-- Point A corresponds to z -/
def A : ℂ := z

/-- Point B corresponds to z² -/
def B : ℂ := z ^ 2

/-- Point C corresponds to z - z² -/
def C : ℂ := z - z ^ 2

/-- The main theorem to be proved -/
theorem complex_geometry_problem :
  z = 1 + Complex.I ∧
  Real.cos (Complex.arg (B - A) - Complex.arg (C - B)) = -2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_complex_geometry_problem_l1332_133201


namespace NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_l1332_133200

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- Define the domain of f
def domain (x : ℝ) : Prop := x > 0

-- State the theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = 1) :
  ∃ (m b : ℝ), ∀ x y, domain x → (y = f a x) →
  (x = 2 → y = f a 2 → x - 4*y + 4*Real.log 2 - 4 = 0) :=
sorry

-- Define the interval (0, e]
def interval (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

-- State the theorem for the minimum value
theorem minimum_value (a : ℝ) :
  (a ≤ 0 → ¬∃ m, ∀ x, interval x → f a x ≥ m) ∧
  (0 < a → a < Real.exp 1 → ∃ m, m = Real.log a ∧ ∀ x, interval x → f a x ≥ m) ∧
  (a ≥ Real.exp 1 → ∃ m, m = a / Real.exp 1 ∧ ∀ x, interval x → f a x ≥ m) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_l1332_133200


namespace NUMINAMATH_CALUDE_negation_equivalence_l1332_133247

theorem negation_equivalence (a b x : ℝ) : 
  ¬(x ≠ a ∧ x ≠ b → x^2 - (a+b)*x + a*b ≠ 0) ↔ 
  (x = a ∨ x = b → x^2 - (a+b)*x + a*b = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1332_133247


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1332_133285

theorem largest_prime_factor_of_expression : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 15^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → q ≤ p) ∧
  (Nat.Prime 241 ∧ 241 ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1332_133285


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l1332_133233

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The cryptarithmetic equation ABAC + BAC = KCKDC -/
def CryptarithmeticEquation (A B C D K : Digit) : Prop :=
  1000 * A.val + 100 * B.val + 10 * A.val + C.val +
  100 * B.val + 10 * A.val + C.val =
  10000 * K.val + 1000 * C.val + 100 * K.val + 10 * D.val + C.val

/-- All digits are different -/
def AllDifferent (A B C D K : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ K ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ K ∧
  C ≠ D ∧ C ≠ K ∧
  D ≠ K

theorem cryptarithmetic_solution :
  ∃! (A B C D K : Digit),
    CryptarithmeticEquation A B C D K ∧
    AllDifferent A B C D K ∧
    A.val = 9 ∧ B.val = 5 ∧ C.val = 0 ∧ D.val = 8 ∧ K.val = 1 :=
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l1332_133233


namespace NUMINAMATH_CALUDE_min_value_theorem_l1332_133246

theorem min_value_theorem (a b : ℝ) (h1 : a * b > 0) (h2 : 2 * a + b = 5) :
  (∀ x y : ℝ, x * y > 0 ∧ 2 * x + y = 5 → 
    2 / (a + 1) + 1 / (b + 1) ≤ 2 / (x + 1) + 1 / (y + 1)) ∧
  2 / (a + 1) + 1 / (b + 1) = 9 / 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1332_133246


namespace NUMINAMATH_CALUDE_valid_n_set_l1332_133298

theorem valid_n_set (n : ℕ) : (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n ∈ ({0, 1, 2} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_valid_n_set_l1332_133298


namespace NUMINAMATH_CALUDE_total_pictures_l1332_133218

/-- The number of pictures drawn by each person and their total -/
def picture_problem (randy peter quincy susan thomas : ℕ) : Prop :=
  randy = 5 ∧
  peter = randy + 3 ∧
  quincy = peter + 20 ∧
  susan = 2 * quincy - 7 ∧
  thomas = randy ^ 3 ∧
  randy + peter + quincy + susan + thomas = 215

/-- Proof that the total number of pictures drawn is 215 -/
theorem total_pictures : ∃ randy peter quincy susan thomas : ℕ, 
  picture_problem randy peter quincy susan thomas := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_l1332_133218


namespace NUMINAMATH_CALUDE_polynomial_factor_l1332_133245

theorem polynomial_factor (x : ℝ) : ∃ (q : ℝ → ℝ), x^2 - 1 = (x + 1) * q x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l1332_133245


namespace NUMINAMATH_CALUDE_total_yen_is_correct_l1332_133216

/-- Represents the total assets of a family in various currencies and investments -/
structure FamilyAssets where
  bahamian_dollars : ℝ
  us_dollars : ℝ
  euros : ℝ
  checking_account1 : ℝ
  checking_account2 : ℝ
  savings_account1 : ℝ
  savings_account2 : ℝ
  stocks : ℝ
  bonds : ℝ
  mutual_funds : ℝ

/-- Exchange rates for different currencies to Japanese yen -/
structure ExchangeRates where
  bahamian_to_yen : ℝ
  usd_to_yen : ℝ
  euro_to_yen : ℝ

/-- Calculates the total amount of yen from all assets -/
def total_yen (assets : FamilyAssets) (rates : ExchangeRates) : ℝ :=
  assets.bahamian_dollars * rates.bahamian_to_yen +
  assets.us_dollars * rates.usd_to_yen +
  assets.euros * rates.euro_to_yen +
  assets.checking_account1 +
  assets.checking_account2 +
  assets.savings_account1 +
  assets.savings_account2 +
  assets.stocks +
  assets.bonds +
  assets.mutual_funds

/-- Theorem stating that the total amount of yen is 1,716,611 -/
theorem total_yen_is_correct (assets : FamilyAssets) (rates : ExchangeRates) :
  assets.bahamian_dollars = 5000 →
  assets.us_dollars = 2000 →
  assets.euros = 3000 →
  assets.checking_account1 = 15000 →
  assets.checking_account2 = 6359 →
  assets.savings_account1 = 5500 →
  assets.savings_account2 = 3102 →
  assets.stocks = 200000 →
  assets.bonds = 150000 →
  assets.mutual_funds = 120000 →
  rates.bahamian_to_yen = 122.13 →
  rates.usd_to_yen = 110.25 →
  rates.euro_to_yen = 128.50 →
  total_yen assets rates = 1716611 := by
  sorry

end NUMINAMATH_CALUDE_total_yen_is_correct_l1332_133216


namespace NUMINAMATH_CALUDE_complex_expression_equals_one_l1332_133271

theorem complex_expression_equals_one 
  (x y : ℂ) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0) 
  (h_equation : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2005 + (y / (x + y))^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_one_l1332_133271


namespace NUMINAMATH_CALUDE_m_value_l1332_133208

/-- The function f(x) = 4x^2 + 5x + 6 -/
def f (x : ℝ) : ℝ := 4 * x^2 + 5 * x + 6

/-- The function g(x) = 2x^3 - mx + 8, parameterized by m -/
def g (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - m * x + 8

/-- Theorem stating that if f(5) - g(5) = 15, then m = 28.4 -/
theorem m_value (m : ℝ) : f 5 - g m 5 = 15 → m = 28.4 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l1332_133208


namespace NUMINAMATH_CALUDE_circle_area_l1332_133299

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
  (∃ (center_x center_y radius : ℝ),
    ∀ (x' y' : ℝ), (x' - center_x)^2 + (y' - center_y)^2 = radius^2 ↔
    3 * x'^2 + 3 * y'^2 - 9 * x' + 12 * y' + 27 = 0) →
  (π * (1/2)^2 : ℝ) = π/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l1332_133299


namespace NUMINAMATH_CALUDE_first_discount_is_twenty_percent_l1332_133237

def initial_price : ℝ := 12000
def final_price : ℝ := 7752
def second_discount : ℝ := 0.15
def third_discount : ℝ := 0.05

def first_discount_percentage (x : ℝ) : Prop :=
  final_price = initial_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount)

theorem first_discount_is_twenty_percent :
  first_discount_percentage 20 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_is_twenty_percent_l1332_133237


namespace NUMINAMATH_CALUDE_square_root_of_x_plus_y_l1332_133232

theorem square_root_of_x_plus_y (x y : ℝ) 
  (h1 : 2*x + 7*y + 1 = 6^2) 
  (h2 : 8*x + 3*y = 5^3) : 
  (x + y).sqrt = 4 ∨ (x + y).sqrt = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_x_plus_y_l1332_133232


namespace NUMINAMATH_CALUDE_velvet_area_for_box_l1332_133203

/-- The total area of velvet needed to line the inside of a box with given dimensions -/
theorem velvet_area_for_box (long_side_length long_side_width short_side_length short_side_width top_bottom_area : ℕ) 
  (h1 : long_side_length = 8)
  (h2 : long_side_width = 6)
  (h3 : short_side_length = 5)
  (h4 : short_side_width = 6)
  (h5 : top_bottom_area = 40) :
  2 * (long_side_length * long_side_width) + 
  2 * (short_side_length * short_side_width) + 
  2 * top_bottom_area = 236 := by
  sorry

#eval 2 * (8 * 6) + 2 * (5 * 6) + 2 * 40

end NUMINAMATH_CALUDE_velvet_area_for_box_l1332_133203


namespace NUMINAMATH_CALUDE_triangle_incenter_inequality_l1332_133295

theorem triangle_incenter_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1/4 < ((a+b)*(b+c)*(c+a)) / ((a+b+c)^3) ∧ ((a+b)*(b+c)*(c+a)) / ((a+b+c)^3) ≤ 8/27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_incenter_inequality_l1332_133295


namespace NUMINAMATH_CALUDE_mozzarella_count_l1332_133202

def cheese_pack (cheddar pepperjack mozzarella : ℕ) : Prop :=
  cheddar = 15 ∧ 
  pepperjack = 45 ∧ 
  (pepperjack : ℚ) / (cheddar + pepperjack + mozzarella) = 1/2

theorem mozzarella_count : ∃ m : ℕ, cheese_pack 15 45 m ∧ m = 30 := by
  sorry

end NUMINAMATH_CALUDE_mozzarella_count_l1332_133202


namespace NUMINAMATH_CALUDE_max_min_value_l1332_133228

def f (x y : ℝ) : ℝ := x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_min_value (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) 
  (h1 : f a b = f b c) (h2 : f b c = f c a) : 
  ∃ (m : ℝ), m = 1 ∧ 
  ∀ (x y z : ℝ), x ≠ y → y ≠ z → z ≠ x → f x y = f y z → f y z = f z x →
  min (x^4 - 4*x^3 + 4*x^2) (min (y^4 - 4*y^3 + 4*y^2) (z^4 - 4*z^3 + 4*z^2)) ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_min_value_l1332_133228


namespace NUMINAMATH_CALUDE_smallest_product_is_zero_l1332_133279

def S : Set ℤ := {-8, -4, 0, 2, 5}

theorem smallest_product_is_zero :
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a * b = 0 ∧ 
  ∀ (x y : ℤ), x ∈ S → y ∈ S → a * b ≤ x * y :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_is_zero_l1332_133279


namespace NUMINAMATH_CALUDE_square_sum_from_means_l1332_133244

theorem square_sum_from_means (x y : ℝ) 
  (h_am : (x + y) / 2 = 20) 
  (h_gm : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l1332_133244


namespace NUMINAMATH_CALUDE_boat_rowing_probability_l1332_133283

theorem boat_rowing_probability : 
  let p_left1 : ℚ := 3/5  -- Probability of first left oar working
  let p_left2 : ℚ := 2/5  -- Probability of second left oar working
  let p_right1 : ℚ := 4/5  -- Probability of first right oar working
  let p_right2 : ℚ := 3/5  -- Probability of second right oar working
  
  -- Probability of both left oars failing
  let p_left_fail : ℚ := (1 - p_left1) * (1 - p_left2)
  
  -- Probability of both right oars failing
  let p_right_fail : ℚ := (1 - p_right1) * (1 - p_right2)
  
  -- Probability of all four oars failing
  let p_all_fail : ℚ := p_left_fail * p_right_fail
  
  -- Probability of being able to row the boat
  let p_row : ℚ := 1 - (p_left_fail + p_right_fail - p_all_fail)
  
  p_row = 437/625 := by sorry

end NUMINAMATH_CALUDE_boat_rowing_probability_l1332_133283


namespace NUMINAMATH_CALUDE_survey_result_l1332_133241

/-- Represents the result of a stratified sampling survey -/
structure SurveyResult where
  totalPopulation : ℕ
  sampleSize : ℕ
  physicsInSample : ℕ
  historyInPopulation : ℕ

/-- Checks if the survey result is valid based on the given conditions -/
def isValidSurvey (s : SurveyResult) : Prop :=
  s.totalPopulation = 1500 ∧
  s.sampleSize = 120 ∧
  s.physicsInSample = 80 ∧
  s.sampleSize - s.physicsInSample > 0 ∧
  s.sampleSize < s.totalPopulation

/-- Theorem stating the result of the survey -/
theorem survey_result (s : SurveyResult) (h : isValidSurvey s) :
  s.historyInPopulation = 500 := by
  sorry

#check survey_result

end NUMINAMATH_CALUDE_survey_result_l1332_133241


namespace NUMINAMATH_CALUDE_range_of_a_l1332_133205

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x^2 - 2*a*x + 2 ≥ 0) → 
  a ∈ Set.Iic (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1332_133205


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l1332_133280

-- Define the types of solids
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

-- Define a predicate for solids that can have a triangular front view
def hasTriangularFrontView (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

-- Theorem stating which solids can have a triangular front view
theorem triangular_front_view_solids :
  ∀ s : Solid, hasTriangularFrontView s ↔
    (s = Solid.TriangularPyramid ∨
     s = Solid.SquarePyramid ∨
     s = Solid.TriangularPrism ∨
     s = Solid.Cone) :=
by sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l1332_133280


namespace NUMINAMATH_CALUDE_prob_eight_odd_rolls_l1332_133217

/-- A fair twelve-sided die -/
def TwelveSidedDie : Finset ℕ := Finset.range 12

/-- The set of odd numbers on a twelve-sided die -/
def OddNumbers : Finset ℕ := TwelveSidedDie.filter (λ x => x % 2 = 1)

/-- The probability of rolling an odd number with a twelve-sided die -/
def ProbOdd : ℚ := (OddNumbers.card : ℚ) / (TwelveSidedDie.card : ℚ)

/-- The number of consecutive rolls -/
def NumRolls : ℕ := 8

theorem prob_eight_odd_rolls :
  ProbOdd ^ NumRolls = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_prob_eight_odd_rolls_l1332_133217


namespace NUMINAMATH_CALUDE_continuity_at_two_l1332_133289

def f (x : ℝ) : ℝ := -5 * x^2 - 8

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_two_l1332_133289


namespace NUMINAMATH_CALUDE_gcd_1021_2729_l1332_133257

theorem gcd_1021_2729 : Nat.gcd 1021 2729 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1021_2729_l1332_133257


namespace NUMINAMATH_CALUDE_equation_solution_l1332_133236

theorem equation_solution (a b : ℝ) (h : a ≠ 0) :
  let x : ℝ := (a^2 - b^2) / a
  x^2 + 4 * b^2 = (2 * a - x)^2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1332_133236


namespace NUMINAMATH_CALUDE_vinegar_mixture_l1332_133248

/-- Given a mixture of water and vinegar, prove the amount of vinegar used. -/
theorem vinegar_mixture (total_mixture water_fraction vinegar_fraction : ℚ) 
  (h_total : total_mixture = 27)
  (h_water : water_fraction = 3/5)
  (h_vinegar : vinegar_fraction = 5/6)
  (h_water_amount : water_fraction * 20 + vinegar_fraction * vinegar_amount = total_mixture) :
  vinegar_amount = 15 := by
  sorry

#check vinegar_mixture

end NUMINAMATH_CALUDE_vinegar_mixture_l1332_133248


namespace NUMINAMATH_CALUDE_no_tangent_lines_l1332_133265

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of two circles --/
structure TwoCircles where
  circle1 : Circle
  circle2 : Circle
  center_distance : ℝ

/-- Counts the number of tangent lines between two circles --/
def count_tangent_lines (tc : TwoCircles) : ℕ := sorry

/-- The specific configuration given in the problem --/
def problem_config : TwoCircles :=
  { circle1 := { center := (0, 0), radius := 4 }
  , circle2 := { center := (3, 0), radius := 6 }
  , center_distance := 3 }

/-- Theorem stating that the number of tangent lines is zero for the given configuration --/
theorem no_tangent_lines : count_tangent_lines problem_config = 0 := by sorry

end NUMINAMATH_CALUDE_no_tangent_lines_l1332_133265


namespace NUMINAMATH_CALUDE_addition_subtraction_ratio_l1332_133260

theorem addition_subtraction_ratio (A B : ℝ) (h : A > 0) (h' : B > 0) (h'' : A / B = 7) : 
  (A + B) / (A - B) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_addition_subtraction_ratio_l1332_133260


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l1332_133268

/-- Given 6 people in an elevator, if a 7th person weighing 133 lbs enters
    and the new average weight becomes 151 lbs, then the initial average
    weight was 154 lbs. -/
theorem elevator_weight_problem :
  ∀ (initial_average : ℝ),
  (6 * initial_average + 133) / 7 = 151 →
  initial_average = 154 :=
by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l1332_133268
