import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_equation_l2319_231964

-- Define the hyperbola
def Hyperbola (x y : ℝ) : Prop := y^2 - x^2/2 = 1

-- State the theorem
theorem hyperbola_equation :
  ∀ (c a : ℝ),
  c = Real.sqrt 3 →
  a = Real.sqrt 3 - 1 →
  ∀ (x y : ℝ),
  Hyperbola x y ↔ 
  (x^2 / (c^2 - a^2) - y^2 / c^2 = 1) ∧
  (c^2 - a^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2319_231964


namespace NUMINAMATH_CALUDE_original_nes_price_l2319_231912

/-- Calculates the original NES sale price before tax given trade-in values and final payment details --/
theorem original_nes_price
  (snes_value : ℝ)
  (snes_credit_rate : ℝ)
  (gameboy_value : ℝ)
  (gameboy_credit_rate : ℝ)
  (ps2_value : ℝ)
  (ps2_credit_rate : ℝ)
  (nes_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (cash_paid : ℝ)
  (change_received : ℝ)
  (free_game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : snes_credit_rate = 0.8)
  (h3 : gameboy_value = 50)
  (h4 : gameboy_credit_rate = 0.75)
  (h5 : ps2_value = 100)
  (h6 : ps2_credit_rate = 0.6)
  (h7 : nes_discount_rate = 0.15)
  (h8 : sales_tax_rate = 0.05)
  (h9 : cash_paid = 80)
  (h10 : change_received = 10)
  (h11 : free_game_value = 30) :
  ∃ (original_price : ℝ), abs (original_price - 289.08) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_original_nes_price_l2319_231912


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l2319_231979

/-- Given a survey with 250 total respondents and 200 preferring brand X,
    prove that the ratio of people preferring brand X to those preferring brand Y is 4:1 -/
theorem brand_preference_ratio (total : ℕ) (brand_x : ℕ) (h1 : total = 250) (h2 : brand_x = 200) :
  (brand_x : ℚ) / (total - brand_x : ℚ) = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l2319_231979


namespace NUMINAMATH_CALUDE_prices_and_min_cost_l2319_231981

/-- Represents the price of a thermometer in yuan -/
def thermometer_price : ℝ := sorry

/-- Represents the price of a barrel of disinfectant in yuan -/
def disinfectant_price : ℝ := sorry

/-- The total cost of 4 thermometers and 2 barrels of disinfectant is 400 yuan -/
axiom equation1 : 4 * thermometer_price + 2 * disinfectant_price = 400

/-- The total cost of 2 thermometers and 4 barrels of disinfectant is 320 yuan -/
axiom equation2 : 2 * thermometer_price + 4 * disinfectant_price = 320

/-- The total number of items to be purchased -/
def total_items : ℕ := 80

/-- The constraint that the number of thermometers is no less than 1/4 of the number of disinfectant -/
def constraint (m : ℕ) : Prop := m ≥ (total_items - m) / 4

/-- The cost function for m thermometers and (80 - m) barrels of disinfectant -/
def cost (m : ℕ) : ℝ := thermometer_price * m + disinfectant_price * (total_items - m)

/-- The theorem stating the unit prices and minimum cost -/
theorem prices_and_min_cost :
  thermometer_price = 80 ∧
  disinfectant_price = 40 ∧
  ∃ m : ℕ, constraint m ∧ cost m = 3840 ∧ ∀ n : ℕ, constraint n → cost m ≤ cost n :=
sorry

end NUMINAMATH_CALUDE_prices_and_min_cost_l2319_231981


namespace NUMINAMATH_CALUDE_cans_per_bag_l2319_231927

theorem cans_per_bag (total_cans : ℕ) (total_bags : ℕ) (h1 : total_cans = 42) (h2 : total_bags = 7) :
  total_cans / total_bags = 6 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l2319_231927


namespace NUMINAMATH_CALUDE_parallelogram_bisector_slope_l2319_231917

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

/-- Checks if a line passes through a given point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x

/-- Checks if a line cuts a parallelogram into two congruent polygons -/
def cutsParallelogramCongruently (l : Line) (p1 p2 p3 p4 : Point) : Prop :=
  sorry -- Definition of this property

/-- The main theorem -/
theorem parallelogram_bisector_slope :
  ∀ (l : Line),
    let p1 : Point := ⟨12, 50⟩
    let p2 : Point := ⟨12, 120⟩
    let p3 : Point := ⟨30, 160⟩
    let p4 : Point := ⟨30, 90⟩
    l.passesThrough p1 ∧
    l.passesThrough p2 ∧
    l.passesThrough p3 ∧
    l.passesThrough p4 ∧
    cutsParallelogramCongruently l p1 p2 p3 p4 →
    l.slope = 5 :=
by
  sorry

#check parallelogram_bisector_slope

end NUMINAMATH_CALUDE_parallelogram_bisector_slope_l2319_231917


namespace NUMINAMATH_CALUDE_marie_sells_40_loaves_l2319_231965

/-- The number of loaves of bread Marie sells each day. -/
def L : ℕ := sorry

/-- The cost of the cash register in dollars. -/
def cash_register_cost : ℕ := 1040

/-- The price of each loaf of bread in dollars. -/
def bread_price : ℕ := 2

/-- The number of cakes sold daily. -/
def cakes_sold : ℕ := 6

/-- The price of each cake in dollars. -/
def cake_price : ℕ := 12

/-- The daily rent in dollars. -/
def daily_rent : ℕ := 20

/-- The daily electricity cost in dollars. -/
def daily_electricity : ℕ := 2

/-- The number of days needed to pay for the cash register. -/
def days_to_pay : ℕ := 8

/-- Theorem stating that Marie sells 40 loaves of bread each day. -/
theorem marie_sells_40_loaves : L = 40 := by
  sorry

end NUMINAMATH_CALUDE_marie_sells_40_loaves_l2319_231965


namespace NUMINAMATH_CALUDE_renovation_constraint_l2319_231999

/-- Represents the constraint condition for hiring workers in a renovation project. -/
theorem renovation_constraint (x y : ℕ) : 
  (50 : ℝ) * x + (40 : ℝ) * y ≤ 2000 ↔ (5 : ℝ) * x + (4 : ℝ) * y ≤ 200 :=
by sorry

#check renovation_constraint

end NUMINAMATH_CALUDE_renovation_constraint_l2319_231999


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2319_231995

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) :
  A = 225 * Real.pi → d = 30 → A = Real.pi * (d / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2319_231995


namespace NUMINAMATH_CALUDE_maia_remaining_requests_l2319_231986

/-- Calculates the number of remaining client requests after a given number of days -/
def remaining_requests (daily_intake : ℕ) (daily_completion : ℕ) (days : ℕ) : ℕ :=
  (daily_intake - daily_completion) * days

/-- Theorem: Given Maia's work pattern, she will have 10 remaining requests after 5 days -/
theorem maia_remaining_requests :
  remaining_requests 6 4 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_maia_remaining_requests_l2319_231986


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l2319_231961

/-- Given a sphere of ice cream with radius 3 inches that melts into a cylinder
    with radius 12 inches while maintaining constant density, the height of the
    resulting cylinder is 1/4 inch. -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h →
  h = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_melted_ice_cream_height_l2319_231961


namespace NUMINAMATH_CALUDE_complex_equidistant_points_l2319_231984

theorem complex_equidistant_points : ∃! z : ℂ, 
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - Complex.I * 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_equidistant_points_l2319_231984


namespace NUMINAMATH_CALUDE_jia_test_probability_l2319_231983

/-- The probability of passing a test with given parameters -/
def test_pass_probability (total_questions n_correct_known n_selected n_to_pass : ℕ) : ℚ :=
  let favorable_outcomes := Nat.choose n_correct_known 2 * Nat.choose (total_questions - n_correct_known) 1 +
                            Nat.choose n_correct_known 3
  let total_outcomes := Nat.choose total_questions n_selected
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability of Jia passing the test -/
theorem jia_test_probability :
  test_pass_probability 10 5 3 2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jia_test_probability_l2319_231983


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l2319_231948

theorem workshop_salary_problem (total_workers : ℕ) (avg_salary : ℝ) 
  (technicians : ℕ) (technician_avg_salary : ℝ) :
  total_workers = 28 →
  avg_salary = 8000 →
  technicians = 7 →
  technician_avg_salary = 14000 →
  (total_workers * avg_salary - technicians * technician_avg_salary) / (total_workers - technicians) = 6000 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l2319_231948


namespace NUMINAMATH_CALUDE_smallest_multiple_of_nine_l2319_231940

theorem smallest_multiple_of_nine (x : ℕ) : x = 18 ↔ 
  (∃ k : ℕ, x = 9 * k) ∧ 
  (x^2 > 200) ∧ 
  (x < Real.sqrt (x^2 - 144) * 5) ∧
  (∀ y : ℕ, y < x → (∃ k : ℕ, y = 9 * k) → 
    (y^2 ≤ 200 ∨ y ≥ Real.sqrt (y^2 - 144) * 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_nine_l2319_231940


namespace NUMINAMATH_CALUDE_job_completion_time_l2319_231922

theorem job_completion_time (a_time b_time : ℕ) (remaining_fraction : ℚ) : 
  a_time = 15 → b_time = 20 → remaining_fraction = 8/15 →
  ∃ (days_worked : ℕ), days_worked = 4 ∧
    (1 - remaining_fraction) = days_worked * (1/a_time + 1/b_time) :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2319_231922


namespace NUMINAMATH_CALUDE_boat_rental_solutions_l2319_231913

theorem boat_rental_solutions :
  ∀ (x y : ℕ),
    12 * x + 5 * y = 99 →
    ((x = 2 ∧ y = 15) ∨ (x = 7 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_boat_rental_solutions_l2319_231913


namespace NUMINAMATH_CALUDE_perpendicular_lines_parameter_l2319_231966

/-- Given two lines ax + y - 1 = 0 and 4x + (a - 5)y - 2 = 0 that are perpendicular,
    prove that a = 1 -/
theorem perpendicular_lines_parameter (a : ℝ) :
  (∃ x y, a * x + y - 1 = 0 ∧ 4 * x + (a - 5) * y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂, 
    (a * x₁ + y₁ - 1 = 0 ∧ 4 * x₁ + (a - 5) * y₁ - 2 = 0) →
    (a * x₂ + y₂ - 1 = 0 ∧ 4 * x₂ + (a - 5) * y₂ - 2 = 0) →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    (a * (x₂ - x₁) + (y₂ - y₁)) * (4 * (x₂ - x₁) + (a - 5) * (y₂ - y₁)) = 0) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parameter_l2319_231966


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2319_231960

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2319_231960


namespace NUMINAMATH_CALUDE_village_population_l2319_231918

theorem village_population (population : ℕ) : 
  (96 : ℚ) / 100 * population = 23040 → population = 24000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l2319_231918


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2319_231900

def base_8_to_decimal (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : Nat,
  n < 1000 →
  n > 0 →
  base_8_to_decimal n % 7 = 0 →
  n ≤ 777 :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2319_231900


namespace NUMINAMATH_CALUDE_quadratic_polynomial_special_roots_l2319_231949

theorem quadratic_polynomial_special_roots (p q : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q
  (∃ α β : ℝ, (f α = 0 ∧ f β = 0) ∧ 
   ((α = f 0 ∧ β = f 1) ∨ (α = f 1 ∧ β = f 0))) →
  f 6 = 71/2 - p := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_special_roots_l2319_231949


namespace NUMINAMATH_CALUDE_even_function_negative_domain_l2319_231937

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_positive : ∀ x > 0, f x = x) :
  ∀ x < 0, f x = -x :=
by sorry

end NUMINAMATH_CALUDE_even_function_negative_domain_l2319_231937


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2319_231939

theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) :
  (a * b) / (c * d) = 16 / 25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2319_231939


namespace NUMINAMATH_CALUDE_highway_distance_is_4km_l2319_231935

/-- Represents the travel scenario between two points A and B -/
structure TravelScenario where
  highway_speed : ℝ
  path_speed : ℝ
  time_difference : ℝ
  distance_difference : ℝ

/-- The distance from A to B along the highway given the travel scenario -/
def highway_distance (scenario : TravelScenario) : ℝ :=
  scenario.path_speed * scenario.time_difference

/-- Theorem stating that for the given scenario, the highway distance is 4 km -/
theorem highway_distance_is_4km (scenario : TravelScenario) 
  (h1 : scenario.highway_speed = 5)
  (h2 : scenario.path_speed = 4)
  (h3 : scenario.time_difference = 1)
  (h4 : scenario.distance_difference = 6) :
  highway_distance scenario = 4 := by
  sorry

#eval highway_distance { highway_speed := 5, path_speed := 4, time_difference := 1, distance_difference := 6 }

end NUMINAMATH_CALUDE_highway_distance_is_4km_l2319_231935


namespace NUMINAMATH_CALUDE_beads_per_earring_is_five_l2319_231998

/-- The number of beads needed to make one earring given Kylie's jewelry-making activities --/
def beads_per_earring : ℕ :=
  let necklaces_monday : ℕ := 10
  let necklaces_tuesday : ℕ := 2
  let bracelets : ℕ := 5
  let earrings : ℕ := 7
  let beads_per_necklace : ℕ := 20
  let beads_per_bracelet : ℕ := 10
  let total_beads : ℕ := 325
  let beads_for_necklaces : ℕ := (necklaces_monday + necklaces_tuesday) * beads_per_necklace
  let beads_for_bracelets : ℕ := bracelets * beads_per_bracelet
  let beads_for_earrings : ℕ := total_beads - beads_for_necklaces - beads_for_bracelets
  beads_for_earrings / earrings

theorem beads_per_earring_is_five : beads_per_earring = 5 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_earring_is_five_l2319_231998


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2319_231928

theorem expand_and_simplify (x y : ℝ) : (x - 2*y)^2 - 2*y*(y - 2*x) = x^2 + 2*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2319_231928


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2319_231919

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 6) :
  a^4 + b^4 + c^4 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2319_231919


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2319_231973

theorem right_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 17 → a = 15 →
  a^2 + b^2 = c^2 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2319_231973


namespace NUMINAMATH_CALUDE_b_value_l2319_231905

theorem b_value (a b : ℚ) : 
  (let x := 2 + Real.sqrt 3
   x^3 + a*x^2 + b*x - 20 = 0) →
  b = 81 := by
sorry

end NUMINAMATH_CALUDE_b_value_l2319_231905


namespace NUMINAMATH_CALUDE_sum_of_powers_l2319_231959

theorem sum_of_powers (x : ℝ) (h1 : x^2020 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 + x^2012 + x^2011 + x^2010 +
  x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + x^2002 + x^2001 + x^2000 +
  x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 + x^1992 + x^1991 + x^1990 +
  x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + x^1982 + x^1981 + x^1980 +
  x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + x^1972 + x^1971 + x^1970 +
  -- ... (continue for all powers from 1969 to 2)
  x^2 + x + 1 - 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l2319_231959


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l2319_231971

/-- The growth factor of bacteria per cycle -/
def growth_factor : ℕ := 4

/-- The duration of one growth cycle in hours -/
def cycle_duration : ℕ := 5

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 1000

/-- The final number of bacteria -/
def final_bacteria : ℕ := 256000

/-- The number of cycles needed to reach the final bacteria count -/
def num_cycles : ℕ := 4

theorem bacteria_growth_time :
  cycle_duration * num_cycles =
    (final_bacteria / initial_bacteria).log growth_factor * cycle_duration :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l2319_231971


namespace NUMINAMATH_CALUDE_m_necessary_not_sufficient_for_n_l2319_231955

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem stating the relationship between M and N
theorem m_necessary_not_sufficient_for_n :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_m_necessary_not_sufficient_for_n_l2319_231955


namespace NUMINAMATH_CALUDE_mike_payment_l2319_231952

def medical_costs (x_ray : ℝ) (blood_tests : ℝ) (deductible : ℝ) : ℝ :=
  let mri := 3 * x_ray
  let ct_scan := 2 * mri
  let ultrasound := 0.5 * mri
  let total_cost := x_ray + mri + ct_scan + blood_tests + ultrasound
  let remaining_amount := total_cost - deductible
  let insurance_coverage := 0.8 * x_ray + 0.8 * mri + 0.7 * ct_scan + 0.5 * blood_tests + 0.6 * ultrasound
  remaining_amount - insurance_coverage

theorem mike_payment (x_ray : ℝ) (blood_tests : ℝ) (deductible : ℝ) 
  (h1 : x_ray = 250)
  (h2 : blood_tests = 200)
  (h3 : deductible = 500) :
  medical_costs x_ray blood_tests deductible = 400 := by
  sorry

end NUMINAMATH_CALUDE_mike_payment_l2319_231952


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l2319_231924

/-- Given the weights of pairs of people, prove that Abby and Damon's combined weight is 285 pounds. -/
theorem abby_and_damon_weight
  (a b c d : ℝ)  -- Weights of Abby, Bart, Cindy, and Damon
  (h1 : a + b = 260)  -- Abby and Bart's combined weight
  (h2 : b + c = 245)  -- Bart and Cindy's combined weight
  (h3 : c + d = 270)  -- Cindy and Damon's combined weight
  : a + d = 285 := by
  sorry

#check abby_and_damon_weight

end NUMINAMATH_CALUDE_abby_and_damon_weight_l2319_231924


namespace NUMINAMATH_CALUDE_smallest_in_set_l2319_231915

theorem smallest_in_set : 
  let S : Set ℤ := {0, -1, 1, 2}
  ∀ x ∈ S, -1 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_in_set_l2319_231915


namespace NUMINAMATH_CALUDE_jelly_cost_l2319_231968

theorem jelly_cost (N C J : ℕ) (h1 : N > 1) (h2 : 3 * N * C + 6 * N * J = 312) : 
  (6 * N * J : ℚ) / 100 = 0.72 := by sorry

end NUMINAMATH_CALUDE_jelly_cost_l2319_231968


namespace NUMINAMATH_CALUDE_parallelogram_bisector_slope_l2319_231904

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Checks if a line through the origin bisects a parallelogram into two congruent polygons -/
def bisects_parallelogram (m n : ℕ) (p : Parallelogram) : Prop :=
  -- This is a placeholder for the actual condition
  True

/-- The main theorem -/
theorem parallelogram_bisector_slope (p : Parallelogram) :
  p.v1 = ⟨20, 90⟩ ∧
  p.v2 = ⟨20, 228⟩ ∧
  p.v3 = ⟨56, 306⟩ ∧
  p.v4 = ⟨56, 168⟩ ∧
  bisects_parallelogram 369 76 p →
  369 / 76 = (p.v3.y - p.v1.y) / (p.v3.x - p.v1.x) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_slope_l2319_231904


namespace NUMINAMATH_CALUDE_total_carrots_l2319_231993

theorem total_carrots (sally_carrots fred_carrots : ℕ) 
  (h1 : sally_carrots = 6) 
  (h2 : fred_carrots = 4) : 
  sally_carrots + fred_carrots = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_l2319_231993


namespace NUMINAMATH_CALUDE_largest_factorial_as_product_of_four_consecutive_l2319_231978

/-- Predicate that checks if a number is expressible as the product of 4 consecutive integers -/
def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) * (x + 2) * (x + 3)

/-- Theorem stating that 6 is the largest integer n such that n! can be expressed as the product of 4 consecutive integers -/
theorem largest_factorial_as_product_of_four_consecutive :
  (6 : ℕ).factorial = 6 * 7 * 8 * 9 ∧
  ∀ n : ℕ, n > 6 → ¬(is_product_of_four_consecutive n.factorial) :=
sorry

end NUMINAMATH_CALUDE_largest_factorial_as_product_of_four_consecutive_l2319_231978


namespace NUMINAMATH_CALUDE_not_always_achievable_all_plus_l2319_231990

/-- Represents a sign in a cell of the grid -/
inductive Sign
| Plus
| Minus

/-- Represents the grid -/
def Grid := Fin 8 → Fin 8 → Sign

/-- Represents a square subgrid -/
structure Square where
  size : Nat
  row : Fin 8
  col : Fin 8

/-- Checks if a square is valid (3x3 or 4x4) -/
def Square.isValid (s : Square) : Prop :=
  (s.size = 3 ∨ s.size = 4) ∧
  s.row + s.size ≤ 8 ∧
  s.col + s.size ≤ 8

/-- Applies an operation to the grid -/
def applyOperation (g : Grid) (s : Square) : Grid :=
  sorry

/-- Checks if a grid is filled with only Plus signs -/
def isAllPlus (g : Grid) : Prop :=
  ∀ i j, g i j = Sign.Plus

/-- Main theorem: It's not always possible to achieve all Plus signs -/
theorem not_always_achievable_all_plus :
  ∃ (initial : Grid), ¬∃ (operations : List Square),
    (∀ s ∈ operations, s.isValid) →
    isAllPlus (operations.foldl applyOperation initial) :=
  sorry

end NUMINAMATH_CALUDE_not_always_achievable_all_plus_l2319_231990


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2319_231911

/-- If 4x^2 - (a-b)x + 9 is a perfect square trinomial, then 2a-2b = ±24 -/
theorem perfect_square_trinomial (a b : ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, 4*x^2 - (a-b)*x + 9 = (2*x - c)^2) →
  (2*a - 2*b = 24 ∨ 2*a - 2*b = -24) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2319_231911


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l2319_231914

/-- Represents a card in the game -/
structure Card where
  id : Nat
  deriving Repr

/-- Represents the state of the game -/
structure GameState where
  player1_cards : List Card
  player2_cards : List Card
  deriving Repr

/-- Represents the strength relationship between cards -/
def beats (card1 card2 : Card) : Bool := sorry

/-- Represents a single turn in the game -/
def play_turn (state : GameState) : GameState := sorry

/-- Represents the strategy chosen by the players -/
def strategy (state : GameState) : GameState := sorry

/-- Theorem stating that there exists a strategy to end the game -/
theorem exists_winning_strategy 
  (n : Nat) 
  (initial_state : GameState) 
  (h1 : initial_state.player1_cards.length + initial_state.player2_cards.length = n) 
  (h2 : ∀ c1 c2 : Card, c1 ≠ c2 → (beats c1 c2 ∨ beats c2 c1)) :
  ∃ (final_state : GameState), 
    (final_state.player1_cards.length = 0 ∨ final_state.player2_cards.length = 0) ∧
    (∃ k : Nat, (strategy^[k]) initial_state = final_state) :=
sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l2319_231914


namespace NUMINAMATH_CALUDE_equal_area_trapezoid_result_l2319_231947

/-- 
A trapezoid with bases differing by 150 units, where x is the length of the segment 
parallel to the bases that divides the trapezoid into two equal-area regions.
-/
structure EqualAreaTrapezoid where
  base_diff : ℝ := 150
  x : ℝ
  divides_equally : x > 0

/-- 
The greatest integer not exceeding x^2/120 for an EqualAreaTrapezoid is 3000.
-/
theorem equal_area_trapezoid_result (t : EqualAreaTrapezoid) : 
  ⌊(t.x^2 / 120)⌋ = 3000 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_trapezoid_result_l2319_231947


namespace NUMINAMATH_CALUDE_find_number_to_add_l2319_231953

theorem find_number_to_add : ∃ x : ℝ, (5 * 12) / (180 / 3) + x = 71 := by
  sorry

end NUMINAMATH_CALUDE_find_number_to_add_l2319_231953


namespace NUMINAMATH_CALUDE_gravitational_force_calculation_l2319_231909

/-- Gravitational force calculation -/
theorem gravitational_force_calculation 
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances
  (f₁ : ℝ) -- Initial force
  (h₁ : d₁ = 5000) -- Initial distance
  (h₂ : d₂ = 300000) -- New distance
  (h₃ : f₁ = 400) -- Initial force value
  (h₄ : k = f₁ * d₁^2) -- Inverse square law
  : (k / d₂^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_gravitational_force_calculation_l2319_231909


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2319_231908

/-- Proves that given the specified conditions, the train's speed is 36 kmph -/
theorem train_speed_calculation (jogger_speed : ℝ) (jogger_ahead : ℝ) (train_length : ℝ) (pass_time : ℝ) :
  jogger_speed = 9 →
  jogger_ahead = 240 →
  train_length = 120 →
  pass_time = 35.99712023038157 →
  (jogger_ahead + train_length) / pass_time * 3.6 = 36 := by
  sorry

#eval (240 + 120) / 35.99712023038157 * 3.6

end NUMINAMATH_CALUDE_train_speed_calculation_l2319_231908


namespace NUMINAMATH_CALUDE_mars_ticket_cost_after_30_years_l2319_231982

/-- The cost of a ticket to Mars after a given number of decades, 
    given an initial cost and a halving rate every decade. -/
def mars_ticket_cost (initial_cost : ℚ) (decades : ℕ) : ℚ :=
  initial_cost / (2 ^ decades)

/-- Theorem stating that the cost of a ticket to Mars after 3 decades
    is $125,000, given an initial cost of $1,000,000 and halving every decade. -/
theorem mars_ticket_cost_after_30_years 
  (initial_cost : ℚ) (h_initial : initial_cost = 1000000) :
  mars_ticket_cost initial_cost 3 = 125000 := by
  sorry

#eval mars_ticket_cost 1000000 3

end NUMINAMATH_CALUDE_mars_ticket_cost_after_30_years_l2319_231982


namespace NUMINAMATH_CALUDE_min_red_points_for_square_l2319_231951

/-- A type representing a point on a circle -/
structure CirclePoint where
  angle : ℝ
  isOnCircle : 0 ≤ angle ∧ angle < 2 * π

/-- A function that determines if a point is colored red -/
def isRed : CirclePoint → Prop := sorry

/-- A predicate that checks if four points form a square on the circle -/
def formSquare (p1 p2 p3 p4 : CirclePoint) : Prop := sorry

/-- The theorem stating the minimum number of red points needed -/
theorem min_red_points_for_square (points : Fin 100 → CirclePoint)
  (equally_spaced : ∀ i : Fin 100, points i = ⟨2 * π * i / 100, sorry⟩) :
  (∃ red_points : Finset CirclePoint,
    red_points.card = 76 ∧
    (∀ p ∈ red_points, isRed p) ∧
    (∀ red_points' : Finset CirclePoint,
      red_points'.card < 76 →
      (∀ p ∈ red_points', isRed p) →
      ¬∃ p1 p2 p3 p4, formSquare p1 p2 p3 p4 ∧ isRed p1 ∧ isRed p2 ∧ isRed p3 ∧ isRed p4)) :=
sorry

end NUMINAMATH_CALUDE_min_red_points_for_square_l2319_231951


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_with_max_base_area_l2319_231950

/-- The volume of a rectangular prism with maximum base area -/
theorem rectangular_prism_volume_with_max_base_area
  (height : ℝ)
  (base_circumference : ℝ)
  (h_height : height = 5)
  (h_base_circumference : base_circumference = 16)
  (h_base_max_area : ∀ (w l : ℝ), w + l = base_circumference / 2 → w * l ≤ (base_circumference / 4)^2) :
  height * (base_circumference / 4)^2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_with_max_base_area_l2319_231950


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2319_231958

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum sequence
  arithmetic_seq : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Main theorem about properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 6 < seq.S 7) (h2 : seq.S 7 > seq.S 8) :
  seq.d < 0 ∧ seq.S 9 < seq.S 6 ∧ ∀ n, seq.S n ≤ seq.S 7 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2319_231958


namespace NUMINAMATH_CALUDE_tom_to_ben_ratio_l2319_231942

def phillip_apples : ℕ := 40
def tom_apples : ℕ := 18
def ben_extra_apples : ℕ := 8

def ben_apples : ℕ := phillip_apples + ben_extra_apples

theorem tom_to_ben_ratio : 
  (tom_apples : ℚ) / (ben_apples : ℚ) = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_tom_to_ben_ratio_l2319_231942


namespace NUMINAMATH_CALUDE_recreation_area_tents_l2319_231934

/-- Represents the number of tents in different parts of the campsite -/
structure CampsiteTents where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : CampsiteTents) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents : 
  ∀ c : CampsiteTents, 
  c.north = 100 → 
  c.east = 2 * c.north → 
  c.center = 4 * c.north → 
  c.south = 200 → 
  total_tents c = 900 := by
  sorry


end NUMINAMATH_CALUDE_recreation_area_tents_l2319_231934


namespace NUMINAMATH_CALUDE_factorial_division_l2319_231938

theorem factorial_division :
  (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 15120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l2319_231938


namespace NUMINAMATH_CALUDE_class_size_calculation_l2319_231989

theorem class_size_calculation (E T B N : ℕ) 
  (h1 : E = 55)
  (h2 : T = 85)
  (h3 : N = 30)
  (h4 : B = 20) :
  E + T - B + N = 150 := by
  sorry

end NUMINAMATH_CALUDE_class_size_calculation_l2319_231989


namespace NUMINAMATH_CALUDE_alloy_interchange_mass_l2319_231901

theorem alloy_interchange_mass (m₁ m₂ x : ℝ) : 
  m₁ = 6 →
  m₂ = 12 →
  0 < x →
  x < m₁ →
  x < m₂ →
  x / m₁ = (m₂ - x) / m₂ →
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_alloy_interchange_mass_l2319_231901


namespace NUMINAMATH_CALUDE_common_root_divisibility_l2319_231933

theorem common_root_divisibility (a b c : ℤ) (h1 : c ≠ b) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) →
  ∃ k : ℤ, a + b + 2*c = 3*k := by
sorry

end NUMINAMATH_CALUDE_common_root_divisibility_l2319_231933


namespace NUMINAMATH_CALUDE_bakery_roll_combinations_l2319_231985

theorem bakery_roll_combinations :
  let total_rolls : ℕ := 9
  let fixed_rolls : ℕ := 6
  let remaining_rolls : ℕ := total_rolls - fixed_rolls
  let num_types : ℕ := 4
  (Nat.choose (remaining_rolls + num_types - 1) (num_types - 1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bakery_roll_combinations_l2319_231985


namespace NUMINAMATH_CALUDE_g_negative_two_equals_negative_fifteen_l2319_231988

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 7
def g (a b x : ℝ) : ℝ := f a b x + 2

-- State the theorem
theorem g_negative_two_equals_negative_fifteen 
  (a b : ℝ) (h : f a b 2 = 3) : g a b (-2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_two_equals_negative_fifteen_l2319_231988


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2319_231916

-- Define the equation
def equation (M : ℝ) : Prop := M * (M - 8) = -8

-- Theorem statement
theorem sum_of_solutions : 
  ∃ (M₁ M₂ : ℝ), equation M₁ ∧ equation M₂ ∧ M₁ + M₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2319_231916


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2319_231974

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (-2, 4)
  are_parallel a b → x = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2319_231974


namespace NUMINAMATH_CALUDE_young_inequality_l2319_231962

theorem young_inequality (p q a b : ℝ) : 
  0 < p → 0 < q → 1 / p + 1 / q = 1 → 0 < a → 0 < b →
  a * b ≤ a^p / p + b^q / q := by
  sorry

end NUMINAMATH_CALUDE_young_inequality_l2319_231962


namespace NUMINAMATH_CALUDE_min_handshakes_theorem_l2319_231946

/-- Represents the number of people at the conference -/
def num_people : ℕ := 30

/-- Represents the minimum number of handshakes per person -/
def min_handshakes_per_person : ℕ := 3

/-- Calculates the minimum number of handshakes for the given conditions -/
def min_total_handshakes : ℕ :=
  (num_people * min_handshakes_per_person) / 2

/-- Theorem stating that the minimum number of handshakes is 45 -/
theorem min_handshakes_theorem :
  min_total_handshakes = 45 := by sorry

end NUMINAMATH_CALUDE_min_handshakes_theorem_l2319_231946


namespace NUMINAMATH_CALUDE_square_between_500_600_l2319_231921

theorem square_between_500_600 : ∃ n : ℕ, 
  500 < n^2 ∧ n^2 ≤ 600 ∧ (n-1)^2 < 500 := by
  sorry

end NUMINAMATH_CALUDE_square_between_500_600_l2319_231921


namespace NUMINAMATH_CALUDE_business_profit_calculation_l2319_231976

theorem business_profit_calculation (suresh_investment ramesh_investment ramesh_profit_share : ℕ) :
  suresh_investment = 24000 →
  ramesh_investment = 40000 →
  ramesh_profit_share = 11875 →
  ∃ (total_profit : ℕ), total_profit = 19000 :=
by sorry

end NUMINAMATH_CALUDE_business_profit_calculation_l2319_231976


namespace NUMINAMATH_CALUDE_base_9_digits_of_2500_l2319_231926

/-- The number of digits in the base-9 representation of a positive integer -/
def num_digits_base_9 (n : ℕ+) : ℕ :=
  Nat.log 9 n.val + 1

/-- Theorem: The number of digits in the base-9 representation of 2500 is 4 -/
theorem base_9_digits_of_2500 : num_digits_base_9 2500 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_9_digits_of_2500_l2319_231926


namespace NUMINAMATH_CALUDE_seven_digit_integers_count_l2319_231936

/-- The number of different seven-digit integers that can be formed using the digits 1, 2, 2, 3, 3, 3, and 5 -/
def seven_digit_integers : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating that the number of different seven-digit integers
    formed using the digits 1, 2, 2, 3, 3, 3, and 5 is equal to 420 -/
theorem seven_digit_integers_count : seven_digit_integers = 420 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_integers_count_l2319_231936


namespace NUMINAMATH_CALUDE_pears_for_20_apples_l2319_231957

/-- The number of apples that cost the same as 5 oranges -/
def apples_per_5_oranges : ℕ := 10

/-- The number of oranges that cost the same as 4 pears -/
def oranges_per_4_pears : ℕ := 3

/-- The number of apples we want to find the equivalent pears for -/
def target_apples : ℕ := 20

/-- The function to calculate the number of pears equivalent to a given number of apples -/
def pears_for_apples (n : ℕ) : ℚ :=
  (n : ℚ) * 5 / apples_per_5_oranges * 4 / oranges_per_4_pears

theorem pears_for_20_apples :
  pears_for_apples target_apples = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pears_for_20_apples_l2319_231957


namespace NUMINAMATH_CALUDE_no_valid_c_l2319_231972

theorem no_valid_c : ¬ ∃ (c : ℕ+), 
  (∃ (x y : ℚ), 3 * x^2 + 7 * x + c.val = 0 ∧ 3 * y^2 + 7 * y + c.val = 0 ∧ x ≠ y) ∧ 
  (∃ (x y : ℚ), 3 * x^2 + 7 * y + c.val = 0 ∧ 3 * y^2 + 7 * y + c.val = 0 ∧ x + y > 4) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_c_l2319_231972


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l2319_231980

theorem ceiling_floor_product (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → -12 < y ∧ y < -11 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l2319_231980


namespace NUMINAMATH_CALUDE_markup_percentage_is_30_l2319_231987

/-- Represents the markup percentage applied by a merchant -/
def markup_percentage : ℝ → ℝ := sorry

/-- Represents the discount percentage applied to the marked price -/
def discount_percentage : ℝ := 10

/-- Represents the profit percentage after discount -/
def profit_percentage : ℝ := 17

/-- Theorem stating that given the conditions, the markup percentage is 30% -/
theorem markup_percentage_is_30 :
  ∀ (cost_price : ℝ),
  cost_price > 0 →
  let marked_price := cost_price * (1 + markup_percentage cost_price / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  selling_price = cost_price * (1 + profit_percentage / 100) →
  markup_percentage cost_price = 30 :=
by sorry

end NUMINAMATH_CALUDE_markup_percentage_is_30_l2319_231987


namespace NUMINAMATH_CALUDE_smallest_n_with_perfect_cube_product_l2319_231932

/-- Checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- Finds the smallest natural number n for which there exist distinct nonzero naturals a, b, c,
    such that n = a + b + c and (a + b)(b + c)(c + a) is a perfect cube -/
theorem smallest_n_with_perfect_cube_product : 
  (∃ n : ℕ, n > 0 ∧ 
    (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
      n = a + b + c ∧ 
      isPerfectCube ((a + b) * (b + c) * (c + a))) ∧
    (∀ m : ℕ, m < n →
      ¬(∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
        m = a + b + c ∧ 
        isPerfectCube ((a + b) * (b + c) * (c + a))))) ∧
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
    10 = a + b + c ∧ 
    isPerfectCube ((a + b) * (b + c) * (c + a))) :=
by
  sorry

#check smallest_n_with_perfect_cube_product

end NUMINAMATH_CALUDE_smallest_n_with_perfect_cube_product_l2319_231932


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l2319_231920

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_wrt_origin (2*a + 1) 4 1 (3*b - 1) → 2*a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l2319_231920


namespace NUMINAMATH_CALUDE_moving_points_minimum_distance_l2319_231929

/-- Two points moving along perpendicular lines towards their intersection --/
theorem moving_points_minimum_distance 
  (a b v v₁ : ℝ) (ha : a > 0) (hb : b > 0) (hv : v > 0) (hv₁ : v₁ > 0) :
  let min_distance := |b * v - a * v₁| / Real.sqrt (v^2 + v₁^2)
  let vertex_distance_diff := |a * v^2 + a * b * v * v₁| / (v^2 + v₁^2)
  let equal_speed_min_distance := |a - b| / Real.sqrt 2
  let equal_speed_time := (a + b) / (2 * v)
  let equal_speed_distance_a := (a - b) / 2
  let equal_speed_distance_b := (b - a) / 2
  ∃ (t : ℝ), 
    (∀ (s : ℝ), 
      Real.sqrt ((a - v * s)^2 + (b - v₁ * s)^2) ≥ min_distance) ∧
    (Real.sqrt ((a - v * t)^2 + (b - v₁ * t)^2) = min_distance) ∧
    (|(a - v * t) - (b - v₁ * t)| = vertex_distance_diff) ∧
    (v = v₁ → 
      min_distance = equal_speed_min_distance ∧
      t = equal_speed_time ∧
      a - v * t = equal_speed_distance_a ∧
      b - v₁ * t = equal_speed_distance_b) := by sorry

end NUMINAMATH_CALUDE_moving_points_minimum_distance_l2319_231929


namespace NUMINAMATH_CALUDE_joan_remaining_flour_l2319_231943

/-- Given a cake recipe that requires a certain amount of flour and the amount already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Theorem: Joan needs to add 4 more cups of flour. -/
theorem joan_remaining_flour :
  remaining_flour 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_flour_l2319_231943


namespace NUMINAMATH_CALUDE_problem_solution_l2319_231977

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 * b + a * b^2 = 2) :
  (a^3 + b^3 ≥ 2) ∧ ((a + b) * (a^5 + b^5) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2319_231977


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l2319_231970

/-- A line parallel to the X-axis passing through the point (3, -2) has the equation y = -2 -/
theorem line_parallel_to_x_axis (line : Set (ℝ × ℝ)) : 
  ((3 : ℝ), -2) ∈ line →  -- The line passes through the point (3, -2)
  (∀ (x y₁ y₂ : ℝ), ((x, y₁) ∈ line ∧ (x, y₂) ∈ line) → y₁ = y₂) →  -- The line is parallel to the X-axis
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = -2 :=  -- The equation of the line is y = -2
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l2319_231970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2319_231954

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + a_2 = 10 and a_4 = a_3 + 2,
    prove that a_3 + a_4 = 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 2 = 10)
  (h_diff : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2319_231954


namespace NUMINAMATH_CALUDE_row_swap_matrix_l2319_231994

theorem row_swap_matrix : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] := by
  sorry

end NUMINAMATH_CALUDE_row_swap_matrix_l2319_231994


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2319_231997

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2319_231997


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_l2319_231992

theorem fraction_is_positive_integer (p : ℕ+) :
  (↑p : ℚ) = 3 ↔ (∃ (k : ℕ+), ((4 * p + 35) : ℚ) / ((3 * p - 8) : ℚ) = ↑k) := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_l2319_231992


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2319_231925

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n = 34 * (n / 100 + (n / 10 % 10) + (n % 10))

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 4 ∧
  (∀ m : ℕ, is_valid_number m → m ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2319_231925


namespace NUMINAMATH_CALUDE_min_parcels_covers_cost_l2319_231941

/-- The minimum number of parcels Lucy must deliver to cover the cost of her motorbike -/
def min_parcels : ℕ := 750

/-- The cost of Lucy's motorbike -/
def motorbike_cost : ℕ := 6000

/-- Lucy's earnings per parcel -/
def earnings_per_parcel : ℕ := 12

/-- Lucy's fuel cost per delivery -/
def fuel_cost_per_delivery : ℕ := 4

/-- Theorem stating that min_parcels is the minimum number of parcels 
    Lucy must deliver to cover the cost of her motorbike -/
theorem min_parcels_covers_cost :
  (min_parcels * (earnings_per_parcel - fuel_cost_per_delivery) ≥ motorbike_cost) ∧
  ∀ n : ℕ, n < min_parcels → n * (earnings_per_parcel - fuel_cost_per_delivery) < motorbike_cost :=
sorry

end NUMINAMATH_CALUDE_min_parcels_covers_cost_l2319_231941


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2319_231969

theorem simplify_and_evaluate : ∀ x : ℤ, 
  -1 < x → x < 3 → x ≠ 1 → x ≠ 2 →
  (3 / (x - 1) - x - 1) * ((x - 1) / (x^2 - 4*x + 4)) = (2 + x) / (2 - x) ∧
  (0 : ℤ) ∈ {y : ℤ | -1 < y ∧ y < 3 ∧ y ≠ 1 ∧ y ≠ 2} ∧
  (2 + 0) / (2 - 0) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2319_231969


namespace NUMINAMATH_CALUDE_triangle_side_length_l2319_231923

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (h3 : B = π / 3) :
  let b := Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))
  b = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2319_231923


namespace NUMINAMATH_CALUDE_tetrakis_hexahedron_colorings_l2319_231975

/-- The number of faces in a regular tetrakis hexahedron -/
def num_faces : ℕ := 16

/-- The number of available colors -/
def num_colors : ℕ := 12

/-- The order of the rotational symmetry group of a tetrakis hexahedron -/
def symmetry_order : ℕ := 24

/-- Calculates the number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The number of distinguishable colorings of a tetrakis hexahedron -/
def distinguishable_colorings : ℕ :=
  permutations num_colors (num_faces - 1) / symmetry_order

theorem tetrakis_hexahedron_colorings :
  distinguishable_colorings = 479001600 := by
  sorry

end NUMINAMATH_CALUDE_tetrakis_hexahedron_colorings_l2319_231975


namespace NUMINAMATH_CALUDE_decimal_equals_scientific_l2319_231991

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_abs_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The given number in decimal form -/
def decimal_number : ℝ := -0.0000406

/-- The scientific notation representation of the number -/
def scientific_form : ScientificNotation := {
  coefficient := -4.06,
  exponent := -5,
  one_le_abs_coeff := by sorry
}

/-- Theorem stating that the decimal number is equal to its scientific notation representation -/
theorem decimal_equals_scientific : 
  decimal_number = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by sorry

end NUMINAMATH_CALUDE_decimal_equals_scientific_l2319_231991


namespace NUMINAMATH_CALUDE_chess_game_probability_l2319_231956

theorem chess_game_probability (p_draw p_B_win : ℝ) 
  (h_draw : p_draw = 1/2) 
  (h_B_win : p_B_win = 1/3) : 
  1 - p_draw - p_B_win = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2319_231956


namespace NUMINAMATH_CALUDE_total_acorns_formula_l2319_231910

/-- The total number of acorns for Shawna, Sheila, and Danny -/
def total_acorns (x y : ℝ) : ℝ :=
  let shawna_acorns := x
  let sheila_acorns := 5.3 * x
  let danny_acorns := sheila_acorns + y
  shawna_acorns + sheila_acorns + danny_acorns

/-- Theorem stating that the total number of acorns is 11.6x + y -/
theorem total_acorns_formula (x y : ℝ) : total_acorns x y = 11.6 * x + y := by
  sorry

end NUMINAMATH_CALUDE_total_acorns_formula_l2319_231910


namespace NUMINAMATH_CALUDE_at_least_two_consecutive_successes_l2319_231996

def probability_success : ℚ := 2 / 5

def probability_failure : ℚ := 1 - probability_success

def number_of_attempts : ℕ := 4

theorem at_least_two_consecutive_successes :
  let p_success := probability_success
  let p_failure := probability_failure
  let n := number_of_attempts
  (1 : ℚ) - (p_failure^n + n * p_success * p_failure^(n-1) + 3 * p_success^2 * p_failure^2) = 44 / 125 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_consecutive_successes_l2319_231996


namespace NUMINAMATH_CALUDE_bill_money_left_l2319_231907

/-- The amount of fool's gold Bill sells in ounces -/
def foolsGoldSold : ℕ := 8

/-- The price per ounce of fool's gold in dollars -/
def pricePerOunce : ℕ := 9

/-- The fine Bill has to pay in dollars -/
def fine : ℕ := 50

/-- The amount of money Bill is left with after selling fool's gold and paying the fine -/
def moneyLeft : ℕ := foolsGoldSold * pricePerOunce - fine

theorem bill_money_left : moneyLeft = 22 := by
  sorry

end NUMINAMATH_CALUDE_bill_money_left_l2319_231907


namespace NUMINAMATH_CALUDE_bicycle_ride_average_speed_l2319_231902

/-- Prove that given an initial ride of 8 miles at 20 mph, riding an additional 16 miles at 40 mph 
    will result in an average speed of 30 mph for the entire trip. -/
theorem bicycle_ride_average_speed 
  (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ)
  (additional_distance : ℝ) :
  initial_distance = 8 ∧ 
  initial_speed = 20 ∧ 
  second_speed = 40 ∧ 
  target_average_speed = 30 ∧
  additional_distance = 16 →
  (initial_distance + additional_distance) / 
    ((initial_distance / initial_speed) + (additional_distance / second_speed)) = 
  target_average_speed :=
by sorry

end NUMINAMATH_CALUDE_bicycle_ride_average_speed_l2319_231902


namespace NUMINAMATH_CALUDE_max_glow_count_max_glow_count_for_given_conditions_l2319_231963

/-- The maximum number of times a light can glow in a given time range -/
theorem max_glow_count (total_duration : ℕ) (glow_interval : ℕ) : ℕ :=
  (total_duration / glow_interval : ℕ)

/-- Proof that the maximum number of glows is 236 for the given conditions -/
theorem max_glow_count_for_given_conditions :
  max_glow_count 4969 21 = 236 := by
  sorry

end NUMINAMATH_CALUDE_max_glow_count_max_glow_count_for_given_conditions_l2319_231963


namespace NUMINAMATH_CALUDE_snack_packs_distribution_l2319_231931

theorem snack_packs_distribution (pretzels : ℕ) (suckers : ℕ) (kids : ℕ) : 
  pretzels = 64 → 
  suckers = 32 → 
  kids = 16 → 
  (pretzels + 4 * pretzels + suckers) / kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_snack_packs_distribution_l2319_231931


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l2319_231930

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_width :
  let carol_rect : Rectangle := { length := 5, width := 24 }
  let jordan_rect : Rectangle := { length := 2, width := 60 }
  area carol_rect = area jordan_rect → jordan_rect.width = 60 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l2319_231930


namespace NUMINAMATH_CALUDE_well_capacity_1200_gallons_l2319_231944

/-- The capacity of a well filled by two pipes -/
def well_capacity (rate1 rate2 : ℝ) (time : ℝ) : ℝ :=
  (rate1 + rate2) * time

/-- Theorem stating the capacity of the well -/
theorem well_capacity_1200_gallons (rate1 rate2 time : ℝ) 
  (h1 : rate1 = 48)
  (h2 : rate2 = 192)
  (h3 : time = 5) :
  well_capacity rate1 rate2 time = 1200 := by
  sorry

end NUMINAMATH_CALUDE_well_capacity_1200_gallons_l2319_231944


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2319_231903

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 60 →
  x ∈ S →
  y ∈ S →
  x = 50 →
  y = 65 →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - x - y) / (S.card - 2) = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2319_231903


namespace NUMINAMATH_CALUDE_log_product_eq_two_implies_x_eq_49_l2319_231906

theorem log_product_eq_two_implies_x_eq_49 
  (k x : ℝ) 
  (h : k > 0) 
  (h' : x > 0) 
  (h'' : (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 2) : 
  x = 49 := by
sorry

end NUMINAMATH_CALUDE_log_product_eq_two_implies_x_eq_49_l2319_231906


namespace NUMINAMATH_CALUDE_correlation_coefficient_comparison_l2319_231967

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def correlation_coefficient (x y : List ℝ) : ℝ := sorry

def r₁ : ℝ := correlation_coefficient X Y
def r₂ : ℝ := correlation_coefficient U V

theorem correlation_coefficient_comparison : r₂ < 0 ∧ r₁ > 0 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_comparison_l2319_231967


namespace NUMINAMATH_CALUDE_square_difference_cube_and_sixth_power_equation_l2319_231945

theorem square_difference_cube_and_sixth_power_equation :
  (∀ m : ℕ, m > 1 → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3) ∧
  (∀ x y : ℕ, x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 → x = 4 ∧ y = 63) :=
by
  sorry

#check square_difference_cube_and_sixth_power_equation

end NUMINAMATH_CALUDE_square_difference_cube_and_sixth_power_equation_l2319_231945
