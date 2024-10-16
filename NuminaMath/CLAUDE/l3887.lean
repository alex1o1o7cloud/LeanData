import Mathlib

namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3887_388797

theorem triangle_max_perimeter : 
  ∀ x y z : ℕ,
  x = 4 * y →
  z = 20 →
  (x + y > z ∧ x + z > y ∧ y + z > x) →
  ∀ a b c : ℕ,
  a = 4 * b →
  c = 20 →
  (a + b > c ∧ a + c > b ∧ b + c > a) →
  x + y + z ≤ a + b + c →
  x + y + z ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3887_388797


namespace NUMINAMATH_CALUDE_purple_tile_cost_l3887_388787

-- Define the problem parameters
def wall1_width : ℝ := 5
def wall1_height : ℝ := 8
def wall2_width : ℝ := 7
def wall2_height : ℝ := 8
def tiles_per_sqft : ℝ := 4
def turquoise_tile_cost : ℝ := 13
def savings : ℝ := 768

-- Calculate total area and number of tiles
def total_area : ℝ := wall1_width * wall1_height + wall2_width * wall2_height
def total_tiles : ℝ := total_area * tiles_per_sqft

-- Calculate costs
def turquoise_total_cost : ℝ := total_tiles * turquoise_tile_cost
def purple_total_cost : ℝ := turquoise_total_cost - savings

-- Theorem to prove
theorem purple_tile_cost : purple_total_cost / total_tiles = 11 := by
  sorry

end NUMINAMATH_CALUDE_purple_tile_cost_l3887_388787


namespace NUMINAMATH_CALUDE_joan_balloons_l3887_388710

theorem joan_balloons (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 2 → total = initial + received → total = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l3887_388710


namespace NUMINAMATH_CALUDE_blanket_rate_problem_l3887_388781

/-- Calculates the unknown rate of two blankets given the following conditions:
    - 2 blankets purchased at 100 rupees each
    - 5 blankets purchased at 150 rupees each
    - 2 blankets purchased at an unknown rate
    - The average price of all blankets is 150 rupees
-/
def unknownBlanketRate : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  λ rate1 count1 rate2 count2 avgPrice =>
    let totalCount := count1 + count2 + 2
    let knownCost := rate1 * count1 + rate2 * count2
    ∃ (unknownRate : ℕ),
      (knownCost + 2 * unknownRate) / totalCount = avgPrice ∧
      unknownRate = 200

theorem blanket_rate_problem :
  unknownBlanketRate 100 2 150 5 150 :=
sorry

end NUMINAMATH_CALUDE_blanket_rate_problem_l3887_388781


namespace NUMINAMATH_CALUDE_arithmetic_operation_l3887_388750

theorem arithmetic_operation : 3 * 14 + 3 * 15 + 3 * 18 + 11 = 152 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operation_l3887_388750


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l3887_388774

theorem greatest_integer_difference (x y : ℝ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) :
  ∃ (n : ℕ), n = ⌊y - x⌋ ∧ n ≤ 6 ∧ ∀ (m : ℕ), m = ⌊y - x⌋ → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l3887_388774


namespace NUMINAMATH_CALUDE_laundry_water_usage_l3887_388737

/-- Calculates the total water usage for a set of laundry loads -/
def total_water_usage (heavy_wash_gallons : ℕ) (regular_wash_gallons : ℕ) (light_wash_gallons : ℕ)
  (heavy_loads : ℕ) (regular_loads : ℕ) (light_loads : ℕ) (bleached_loads : ℕ) : ℕ :=
  heavy_wash_gallons * heavy_loads +
  regular_wash_gallons * regular_loads +
  light_wash_gallons * (light_loads + bleached_loads)

/-- Proves that the total water usage for the given laundry scenario is 76 gallons -/
theorem laundry_water_usage :
  total_water_usage 20 10 2 2 3 1 2 = 76 := by sorry

end NUMINAMATH_CALUDE_laundry_water_usage_l3887_388737


namespace NUMINAMATH_CALUDE_johns_next_birthday_l3887_388757

-- Define variables for ages
variable (j b a : ℝ)

-- Define the relationships between ages
def john_bob_relation (j b : ℝ) : Prop := j = 1.25 * b
def bob_alice_relation (b a : ℝ) : Prop := b = 0.5 * a
def age_sum (j b a : ℝ) : Prop := j + b + a = 37.8

-- Theorem statement
theorem johns_next_birthday 
  (h1 : john_bob_relation j b)
  (h2 : bob_alice_relation b a)
  (h3 : age_sum j b a) :
  ⌈j⌉ + 1 = 12 := by sorry

end NUMINAMATH_CALUDE_johns_next_birthday_l3887_388757


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3887_388735

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

theorem hyperbola_properties :
  ∃ (e : ℝ), eccentricity e ∧
  ∀ (x y : ℝ), hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3887_388735


namespace NUMINAMATH_CALUDE_abs_inequality_implies_a_greater_than_two_l3887_388785

theorem abs_inequality_implies_a_greater_than_two :
  (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_abs_inequality_implies_a_greater_than_two_l3887_388785


namespace NUMINAMATH_CALUDE_unique_b_for_three_integer_solutions_l3887_388724

theorem unique_b_for_three_integer_solutions :
  ∃! b : ℤ, ∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x : ℤ, x ∈ s ↔ x^2 + b*x + 5 ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_unique_b_for_three_integer_solutions_l3887_388724


namespace NUMINAMATH_CALUDE_b_investment_is_8000_l3887_388786

/-- Represents a partnership with three partners -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  a_profit : ℝ
  b_profit : ℝ

/-- The profit share is proportional to the investment -/
def profit_proportional (p : Partnership) : Prop :=
  p.a_profit / p.a_investment = p.b_profit / p.b_investment

/-- Theorem stating that given the conditions, b's investment is $8000 -/
theorem b_investment_is_8000 (p : Partnership) 
  (h1 : p.a_investment = 7000)
  (h2 : p.c_investment = 18000)
  (h3 : p.a_profit = 560)
  (h4 : p.b_profit = 880)
  (h5 : profit_proportional p) : 
  p.b_investment = 8000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_8000_l3887_388786


namespace NUMINAMATH_CALUDE_fixed_point_on_curve_l3887_388783

-- Define the curve equation
def curve_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Theorem statement
theorem fixed_point_on_curve :
  ∀ k : ℝ, k ≠ -1 → curve_equation k 1 (-3) :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_curve_l3887_388783


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l3887_388795

theorem imaginary_part_of_2_plus_i_times_i (i : ℂ) : 
  i ^ 2 = -1 → Complex.im ((2 + i) * i) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l3887_388795


namespace NUMINAMATH_CALUDE_mosquito_shadow_speed_l3887_388784

/-- The speed of a mosquito's shadow on the bottom of a water body. -/
def shadow_speed (v : ℝ) (cos_beta : ℝ) : Set ℝ :=
  {0, 2 * v * cos_beta}

/-- Theorem: Given the conditions of the mosquito problem, the speed of the shadow is either 0 m/s or 0.8 m/s. -/
theorem mosquito_shadow_speed 
  (v : ℝ) 
  (t : ℝ) 
  (h : ℝ) 
  (cos_theta : ℝ) 
  (cos_beta : ℝ) 
  (hv : v = 0.5)
  (ht : t = 20)
  (hh : h = 6)
  (hcos_theta : cos_theta = 0.6)
  (hcos_beta : cos_beta = 0.8)
  : shadow_speed v cos_beta = {0, 0.8} := by
  sorry

#check mosquito_shadow_speed

end NUMINAMATH_CALUDE_mosquito_shadow_speed_l3887_388784


namespace NUMINAMATH_CALUDE_symmetric_circles_line_l3887_388709

-- Define the circles
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 - a = 0
def C₂ (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*a*y + 3 = 0

-- Define the line
def line_l (x y : ℝ) : Prop := 2*x - 4*y + 5 = 0

-- State the theorem
theorem symmetric_circles_line (a : ℝ) :
  (∀ x y, C₁ x y a ↔ C₂ (2*x + 1) (2*y - a) a) →
  (∀ x y, line_l x y ↔ (∃ x₀ y₀, C₁ x₀ y₀ a ∧ C₂ (2*x - x₀) (2*y - y₀) a)) :=
sorry

end NUMINAMATH_CALUDE_symmetric_circles_line_l3887_388709


namespace NUMINAMATH_CALUDE_factorial_ratio_l3887_388782

theorem factorial_ratio (N : ℕ) (h : N > 1) :
  (Nat.factorial (N^2 - 1)) / ((Nat.factorial (N + 1))^2) = 
  (Nat.factorial (N - 1)) / (N + 1) :=
sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3887_388782


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3887_388762

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * Real.log 3 / Real.log 2 + y = Real.log 18 / Real.log 2

def equation2 (x y : ℝ) : Prop := (5 : ℝ)^x = 25^y

-- Theorem statement
theorem solution_satisfies_system :
  equation1 2 1 ∧ equation2 2 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3887_388762


namespace NUMINAMATH_CALUDE_test_score_combination_l3887_388761

theorem test_score_combination :
  ∀ (x y z : ℕ),
    x + y + z = 6 →
    8 * x + 2 * y = 20 →
    x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_CALUDE_test_score_combination_l3887_388761


namespace NUMINAMATH_CALUDE_problem_statement_l3887_388723

theorem problem_statement (a b c : ℝ) (h : a + b + c = 0) :
  (a = 0 ∧ b = 0 ∧ c = 0 ↔ a * b + b * c + a * c = 0) ∧
  (a * b * c = 1 ∧ a ≥ b ∧ b ≥ c → c ≤ -Real.rpow 4 (1/3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3887_388723


namespace NUMINAMATH_CALUDE_fraction_equality_l3887_388707

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x - 2*y) / (2*x - 5*y) = 3) : 
  (2*x + 5*y) / (5*x - 2*y) = 31/63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3887_388707


namespace NUMINAMATH_CALUDE_count_seven_100_to_199_l3887_388725

/-- Count of digit 7 in a number -/
def count_seven (n : ℕ) : ℕ := sorry

/-- Sum of count_seven for a range of numbers -/
def sum_count_seven (start finish : ℕ) : ℕ := sorry

theorem count_seven_100_to_199 :
  sum_count_seven 100 199 = 20 := by sorry

end NUMINAMATH_CALUDE_count_seven_100_to_199_l3887_388725


namespace NUMINAMATH_CALUDE_joan_pinball_spending_l3887_388717

/-- The amount of money in dollars represented by a half-dollar -/
def half_dollar_value : ℚ := 0.5

/-- The total amount spent in dollars given the number of half-dollars spent each day -/
def total_spent (wed thur fri : ℕ) : ℚ :=
  half_dollar_value * (wed + thur + fri : ℚ)

/-- Theorem stating that if Joan spent 4 half-dollars on Wednesday, 14 on Thursday,
    and 8 on Friday, then the total amount she spent playing pinball is $13.00 -/
theorem joan_pinball_spending :
  total_spent 4 14 8 = 13 := by sorry

end NUMINAMATH_CALUDE_joan_pinball_spending_l3887_388717


namespace NUMINAMATH_CALUDE_max_sum_of_entries_l3887_388736

def numbers : List ℕ := [1, 2, 4, 5, 7, 8]

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
  d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def sum_of_entries (a b c d e f : ℕ) : ℕ := (a + b + c) * (d + e + f)

theorem max_sum_of_entries :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    sum_of_entries a b c d e f ≤ 182 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_entries_l3887_388736


namespace NUMINAMATH_CALUDE_min_teams_for_highest_score_fewer_wins_l3887_388738

/-- Represents a soccer team --/
structure Team :=
  (id : ℕ)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

/-- Calculates the score of a team --/
def score (t : Team) : ℕ := 2 * t.wins + t.draws

/-- Represents a soccer tournament --/
structure Tournament :=
  (teams : List Team)
  (numTeams : ℕ)
  (allPlayedAgainstEachOther : Bool)

/-- Checks if a team has the highest score in the tournament --/
def hasHighestScore (t : Team) (tournament : Tournament) : Prop :=
  ∀ other : Team, other ∈ tournament.teams → score t ≥ score other

/-- Checks if a team has fewer wins than all other teams --/
def hasFewerWins (t : Team) (tournament : Tournament) : Prop :=
  ∀ other : Team, other ∈ tournament.teams → other.id ≠ t.id → t.wins < other.wins

theorem min_teams_for_highest_score_fewer_wins (n : ℕ) :
  (∃ tournament : Tournament,
    tournament.numTeams = n ∧
    tournament.allPlayedAgainstEachOther = true ∧
    (∃ t : Team, t ∈ tournament.teams ∧ 
      hasHighestScore t tournament ∧
      hasFewerWins t tournament)) →
  n ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_teams_for_highest_score_fewer_wins_l3887_388738


namespace NUMINAMATH_CALUDE_sun_city_population_l3887_388742

theorem sun_city_population (willowdale roseville sun : ℕ) : 
  willowdale = 2000 →
  roseville = 3 * willowdale - 500 →
  sun = 2 * roseville + 1000 →
  sun = 12000 := by
sorry

end NUMINAMATH_CALUDE_sun_city_population_l3887_388742


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3887_388771

/-- Given a quadratic equation (k+2)x^2 + 6x + k^2 + k - 2 = 0 where one of its roots is 0,
    prove that k = 1 -/
theorem quadratic_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0) ∧ 
  ((k + 2) * 0^2 + 6 * 0 + k^2 + k - 2 = 0) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3887_388771


namespace NUMINAMATH_CALUDE_weight_difference_E_D_l3887_388706

/-- Given the weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D -/
theorem weight_difference_E_D (w_A w_B w_C w_D w_E : ℝ) : w_E - w_D = 3 :=
  by
  have h1 : (w_A + w_B + w_C) / 3 = 84 := by sorry
  have h2 : (w_A + w_B + w_C + w_D) / 4 = 80 := by sorry
  have h3 : (w_B + w_C + w_D + w_E) / 4 = 79 := by sorry
  have h4 : w_A = 75 := by sorry
  sorry

#check weight_difference_E_D

end NUMINAMATH_CALUDE_weight_difference_E_D_l3887_388706


namespace NUMINAMATH_CALUDE_field_trip_total_l3887_388769

/-- Field trip problem -/
theorem field_trip_total (
  num_vans : ℕ) (num_minibusses : ℕ) (num_coach_buses : ℕ)
  (students_per_van : ℕ) (teachers_per_van : ℕ) (parents_per_van : ℕ)
  (students_per_minibus : ℕ) (teachers_per_minibus : ℕ) (parents_per_minibus : ℕ)
  (students_per_coach : ℕ) (teachers_per_coach : ℕ) (parents_per_coach : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_minibusses = 4)
  (h3 : num_coach_buses = 2)
  (h4 : students_per_van = 10)
  (h5 : teachers_per_van = 2)
  (h6 : parents_per_van = 1)
  (h7 : students_per_minibus = 24)
  (h8 : teachers_per_minibus = 3)
  (h9 : parents_per_minibus = 2)
  (h10 : students_per_coach = 48)
  (h11 : teachers_per_coach = 4)
  (h12 : parents_per_coach = 4) :
  (num_vans * (students_per_van + teachers_per_van + parents_per_van) +
   num_minibusses * (students_per_minibus + teachers_per_minibus + parents_per_minibus) +
   num_coach_buses * (students_per_coach + teachers_per_coach + parents_per_coach)) = 306 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_total_l3887_388769


namespace NUMINAMATH_CALUDE_original_number_proof_l3887_388759

theorem original_number_proof (x : ℝ) : 
  (1.25 * x - 0.70 * x = 22) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3887_388759


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_m_bound_l3887_388749

/-- Given that f(x) = -x^2 + mx is an increasing function on (-∞, 1], prove that m ≥ 2 -/
theorem increasing_quadratic_function_m_bound 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = -x^2 + m*x) 
  (h2 : ∀ x y, x < y → x ≤ 1 → y ≤ 1 → f x < f y) : 
  m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_m_bound_l3887_388749


namespace NUMINAMATH_CALUDE_complex_magnitude_l3887_388764

theorem complex_magnitude (z : ℂ) (h : z^2 = 4 - 3*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3887_388764


namespace NUMINAMATH_CALUDE_horner_method_proof_l3887_388776

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ :=
  horner_polynomial [5, 4, 3, 2, 1, 0] x

theorem horner_method_proof :
  f 3 = 1641 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l3887_388776


namespace NUMINAMATH_CALUDE_smallest_a_for_two_zeros_in_unit_interval_l3887_388700

theorem smallest_a_for_two_zeros_in_unit_interval :
  ∃ (a b c : ℤ), 
    a = 5 ∧
    (∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
      a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) ∧
    (∀ (a' b' c' : ℤ), a' > 0 ∧ a' < 5 →
      ¬(∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
        a' * x^2 - b' * x + c' = 0 ∧ a' * y^2 - b' * y + c' = 0)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_two_zeros_in_unit_interval_l3887_388700


namespace NUMINAMATH_CALUDE_sum_and_sum_squares_bound_equality_conditions_l3887_388734

theorem sum_and_sum_squares_bound (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := by
  sorry

theorem equality_conditions (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  (a + b + c + a^2 + b^2 + c^2 = 4) ↔ 
  ((a, b, c) = (1, 1, -1) ∨ (a, b, c) = (1, -1, 1) ∨ (a, b, c) = (-1, 1, 1)) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_sum_squares_bound_equality_conditions_l3887_388734


namespace NUMINAMATH_CALUDE_grid_rectangle_division_l3887_388741

/-- A grid rectangle with cell side length 1 cm and area 2021 cm² -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  area_eq : width * height = 2021

/-- A cut configuration for the grid rectangle -/
structure CutConfig where
  hor_cut : ℕ
  ver_cut : ℕ

/-- The four parts resulting from a cut configuration -/
def parts (rect : GridRectangle) (cut : CutConfig) : Fin 4 → ℕ
| ⟨0, _⟩ => cut.hor_cut * cut.ver_cut
| ⟨1, _⟩ => cut.hor_cut * (rect.width - cut.ver_cut)
| ⟨2, _⟩ => (rect.height - cut.hor_cut) * cut.ver_cut
| ⟨3, _⟩ => (rect.height - cut.hor_cut) * (rect.width - cut.ver_cut)
| _ => 0

/-- The theorem to be proved -/
theorem grid_rectangle_division (rect : GridRectangle) :
  ∀ (cut : CutConfig), cut.hor_cut < rect.height → cut.ver_cut < rect.width →
  ∃ (i : Fin 4), parts rect cut i ≥ 528 := by
  sorry

end NUMINAMATH_CALUDE_grid_rectangle_division_l3887_388741


namespace NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3887_388729

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 (b > 0), 
    if the distance from its foci to the asymptote is 3, 
    then the length of its imaginary axis is 6. -/
theorem hyperbola_imaginary_axis_length 
  (b : ℝ) 
  (h1 : b > 0) 
  (h2 : (b * Real.sqrt (4 + b^2)) / Real.sqrt (4 + b^2) = 3) : 
  2 * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3887_388729


namespace NUMINAMATH_CALUDE_parametric_to_regular_equation_l3887_388773

theorem parametric_to_regular_equation 
  (t : ℝ) (ht : t ≠ 0) 
  (x : ℝ) (hx : x = t + 1/t) 
  (y : ℝ) (hy : y = t^2 + 1/t^2) : 
  x^2 - y - 2 = 0 ∧ y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_regular_equation_l3887_388773


namespace NUMINAMATH_CALUDE_f_properties_l3887_388796

def f (x : ℝ) := x^3 - 3*x^2 + 6

theorem f_properties :
  (∃ (a : ℝ), IsLocalMin f a ∧ f a = 2) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 6) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 6) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, 2 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 2) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l3887_388796


namespace NUMINAMATH_CALUDE_shooter_probability_l3887_388703

theorem shooter_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.24) (h2 : p9 = 0.28) (h3 : p8 = 0.19) :
  1 - p10 - p9 = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l3887_388703


namespace NUMINAMATH_CALUDE_b_absolute_value_l3887_388701

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the polynomial g(x)
def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^5 + b * x^4 + c * x^3 + b * x + a

-- State the theorem
theorem b_absolute_value (a b c : ℤ) : 
  (g a b c (3 + i) = 0) →  -- Condition 1
  (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c) = 1) →  -- Condition 2 and 3
  (Int.natAbs b = 60) :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_b_absolute_value_l3887_388701


namespace NUMINAMATH_CALUDE_correct_value_proof_l3887_388721

theorem correct_value_proof (n : ℕ) (initial_mean correct_mean wrong_value : ℚ) 
  (h1 : n = 30)
  (h2 : initial_mean = 250)
  (h3 : correct_mean = 251)
  (h4 : wrong_value = 135) :
  ∃ (correct_value : ℚ),
    correct_value = 165 ∧
    n * correct_mean = n * initial_mean - wrong_value + correct_value :=
by
  sorry

end NUMINAMATH_CALUDE_correct_value_proof_l3887_388721


namespace NUMINAMATH_CALUDE_largest_even_multiple_of_15_under_500_l3887_388747

theorem largest_even_multiple_of_15_under_500 : ∃ n : ℕ, 
  n * 15 = 480 ∧ 
  480 % 2 = 0 ∧ 
  480 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 % 2 = 0 → m * 15 ≤ 480 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_multiple_of_15_under_500_l3887_388747


namespace NUMINAMATH_CALUDE_division_theorem_l3887_388763

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 176 → 
  divisor = 19 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3887_388763


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3887_388727

/-- A circle passing through three points -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The given circle -/
def givenCircle : Circle where
  equation := fun x y => x^2 + y^2 - 4*x - 6*y = 0

/-- Theorem stating that the given circle passes through the specified points -/
theorem circle_passes_through_points (c : Circle) :
  (c.equation 0 0) ∧ (c.equation 4 0) ∧ (c.equation (-1) 1) → c = givenCircle := by
  sorry

#check circle_passes_through_points

end NUMINAMATH_CALUDE_circle_passes_through_points_l3887_388727


namespace NUMINAMATH_CALUDE_product_sum_quotient_l3887_388746

theorem product_sum_quotient (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x * y = 9375 ∧ x + y = 400 → max x y / min x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_quotient_l3887_388746


namespace NUMINAMATH_CALUDE_mistake_correction_l3887_388718

theorem mistake_correction (a : ℤ) (h : 31 - a = 12) : 31 + a = 50 := by
  sorry

end NUMINAMATH_CALUDE_mistake_correction_l3887_388718


namespace NUMINAMATH_CALUDE_sales_growth_rate_equation_l3887_388744

/-- The average monthly growth rate of a store's sales revenue -/
def average_monthly_growth_rate (march_revenue : ℝ) (may_revenue : ℝ) : ℝ → Prop :=
  λ x => 3 * (1 + x)^2 = 3.63

theorem sales_growth_rate_equation (march_revenue may_revenue : ℝ) 
  (h1 : march_revenue = 30000)
  (h2 : may_revenue = 36300) :
  ∃ x, average_monthly_growth_rate march_revenue may_revenue x :=
by
  sorry

end NUMINAMATH_CALUDE_sales_growth_rate_equation_l3887_388744


namespace NUMINAMATH_CALUDE_smallest_common_factor_l3887_388713

theorem smallest_common_factor (n : ℕ) : n > 0 ∧ 
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 4)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 4))) →
  n = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l3887_388713


namespace NUMINAMATH_CALUDE_third_grade_sample_size_l3887_388778

/-- Calculates the number of students to be sampled from a specific grade in stratified sampling -/
def stratified_sample_size (total_students : ℕ) (sample_size : ℕ) (grade_students : ℕ) : ℕ :=
  (grade_students * sample_size) / total_students

/-- Theorem: In a stratified sampling of 65 students from a high school with 1300 total students,
    where 500 students are in the third grade, the number of students to be sampled from the
    third grade is 25. -/
theorem third_grade_sample_size :
  stratified_sample_size 1300 65 500 = 25 := by
  sorry

#eval stratified_sample_size 1300 65 500

end NUMINAMATH_CALUDE_third_grade_sample_size_l3887_388778


namespace NUMINAMATH_CALUDE_x_minus_y_equals_two_l3887_388714

theorem x_minus_y_equals_two (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_sq_eq : x^2 - y^2 = 20) : 
  x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_two_l3887_388714


namespace NUMINAMATH_CALUDE_intersection_count_l3887_388731

/-- Calculates the number of intersections between two regular polygons inscribed in a circle -/
def intersections (n m : ℕ) : ℕ := 2 * n * m

/-- The set of regular polygons inscribed in the circle -/
def polygons : Finset ℕ := {4, 6, 8, 10}

/-- The set of all pairs of polygons -/
def polygon_pairs : Finset (ℕ × ℕ) := 
  {(4, 6), (4, 8), (4, 10), (6, 8), (6, 10), (8, 10)}

theorem intersection_count :
  (polygon_pairs.sum (fun (p : ℕ × ℕ) => intersections p.1 p.2)) = 568 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l3887_388731


namespace NUMINAMATH_CALUDE_remainder_3079_div_67_l3887_388732

theorem remainder_3079_div_67 : 3079 % 67 = 64 := by sorry

end NUMINAMATH_CALUDE_remainder_3079_div_67_l3887_388732


namespace NUMINAMATH_CALUDE_complex_number_in_quadrant_IV_l3887_388754

/-- The complex number (1+i)/(1+2i) lies in Quadrant IV of the complex plane -/
theorem complex_number_in_quadrant_IV : 
  let z : ℂ := (1 + Complex.I) / (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_quadrant_IV_l3887_388754


namespace NUMINAMATH_CALUDE_find_number_l3887_388772

theorem find_number : ∃! x : ℤ, x - 254 + 329 = 695 ∧ x = 620 := by sorry

end NUMINAMATH_CALUDE_find_number_l3887_388772


namespace NUMINAMATH_CALUDE_last_digit_of_7_to_1032_l3887_388791

theorem last_digit_of_7_to_1032 : ∃ n : ℕ, 7^1032 ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_7_to_1032_l3887_388791


namespace NUMINAMATH_CALUDE_possible_values_of_d_over_a_l3887_388765

theorem possible_values_of_d_over_a (a d : ℝ) (h1 : a^2 - 6*a*d + 8*d^2 = 0) (h2 : a ≠ 0) :
  d/a = 1/2 ∨ d/a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_d_over_a_l3887_388765


namespace NUMINAMATH_CALUDE_mother_pies_per_day_l3887_388751

/-- The number of pies Eddie's mother can bake per day -/
def mother_pies : ℕ := 8

/-- The number of pies Eddie can bake per day -/
def eddie_pies : ℕ := 3

/-- The number of pies Eddie's sister can bake per day -/
def sister_pies : ℕ := 6

/-- The number of days they bake pies -/
def days : ℕ := 7

/-- The total number of pies they can bake in the given days -/
def total_pies : ℕ := 119

theorem mother_pies_per_day :
  eddie_pies * days + sister_pies * days + mother_pies * days = total_pies :=
by sorry

end NUMINAMATH_CALUDE_mother_pies_per_day_l3887_388751


namespace NUMINAMATH_CALUDE_registration_cost_per_vehicle_l3887_388712

theorem registration_cost_per_vehicle 
  (num_dirt_bikes : ℕ) 
  (cost_per_dirt_bike : ℕ) 
  (num_off_road : ℕ) 
  (cost_per_off_road : ℕ) 
  (total_cost : ℕ) 
  (h1 : num_dirt_bikes = 3)
  (h2 : cost_per_dirt_bike = 150)
  (h3 : num_off_road = 4)
  (h4 : cost_per_off_road = 300)
  (h5 : total_cost = 1825) :
  (total_cost - (num_dirt_bikes * cost_per_dirt_bike + num_off_road * cost_per_off_road)) / (num_dirt_bikes + num_off_road) = 25 := by
    sorry

end NUMINAMATH_CALUDE_registration_cost_per_vehicle_l3887_388712


namespace NUMINAMATH_CALUDE_equation_transformation_l3887_388758

theorem equation_transformation (x y : ℝ) : x = y → x - 2 = y - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3887_388758


namespace NUMINAMATH_CALUDE_f_sum_equals_three_l3887_388753

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_sum_equals_three 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_odd : is_odd_function (fun x ↦ f (x - 1))) 
  (h_f2 : f 2 = 3) : 
  f 5 + f 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_f_sum_equals_three_l3887_388753


namespace NUMINAMATH_CALUDE_lcm_852_1491_l3887_388788

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end NUMINAMATH_CALUDE_lcm_852_1491_l3887_388788


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l3887_388760

theorem ten_thousandths_place_of_5_32 : 
  ∃ (n : ℕ), (5 : ℚ) / 32 = (n * 10000 + 2) / 100000 ∧ n < 10000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l3887_388760


namespace NUMINAMATH_CALUDE_ball_volume_ratio_l3887_388719

theorem ball_volume_ratio :
  ∀ (x y z : ℝ),
    x > 0 → y > 0 → z > 0 →
    x = 3 * (y - x) →
    z - y = 3 * x →
    ∃ (k : ℝ), k > 0 ∧ x = 3 * k ∧ y = 4 * k ∧ z = 13 * k :=
by sorry

end NUMINAMATH_CALUDE_ball_volume_ratio_l3887_388719


namespace NUMINAMATH_CALUDE_product_sum_difference_l3887_388715

theorem product_sum_difference (x y : ℝ) : x * y = 23 ∧ x + y = 24 → |x - y| = 22 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_difference_l3887_388715


namespace NUMINAMATH_CALUDE_tom_initial_investment_l3887_388704

/-- Represents the investment scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_initial : ℕ  -- Tom's initial investment
  jose_investment : ℕ := 4500  -- Jose's investment
  total_profit : ℕ := 5400  -- Total profit after one year
  jose_profit : ℕ := 3000  -- Jose's share of the profit
  tom_months : ℕ := 12  -- Months Tom invested
  jose_months : ℕ := 10  -- Months Jose invested

/-- Theorem stating that Tom's initial investment was 3000 --/
theorem tom_initial_investment (shop : ShopInvestment) : shop.tom_initial = 3000 := by
  sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l3887_388704


namespace NUMINAMATH_CALUDE_triangle_properties_l3887_388722

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  -- Triangle ABC with sides a, b, c and angles A, B, C
  -- Vectors (a-b, 1) and (a-c, 2) are collinear
  (a - b) / (a - c) = 1 / 2 →
  -- Angle A is 120°
  A = 2 * π / 3 →
  -- Circumradius is 14
  R = 14 →
  -- Ratio a:b:c is 7:5:3
  ∃ (k : ℝ), a = 7 * k ∧ b = 5 * k ∧ c = 3 * k ∧
  -- Area of triangle ABC is 45√3
  1/2 * b * c * Real.sin A = 45 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3887_388722


namespace NUMINAMATH_CALUDE_candy_distribution_l3887_388728

theorem candy_distribution (n : ℕ) (total_candy : ℕ) : 
  total_candy = 120 →
  (∃ q : ℕ, total_candy = 2 * n + 2 * q) →
  n = 58 ∨ n = 60 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3887_388728


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3887_388716

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3887_388716


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l3887_388792

theorem greatest_integer_inequality : 
  (∃ (y : ℤ), (5 : ℚ) / 8 > (y : ℚ) / 15 ∧ 
    ∀ (z : ℤ), (5 : ℚ) / 8 > (z : ℚ) / 15 → z ≤ y) ∧ 
  (5 : ℚ) / 8 > (9 : ℚ) / 15 ∧ 
  (5 : ℚ) / 8 ≤ (10 : ℚ) / 15 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l3887_388792


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3887_388779

theorem unique_solution_for_exponential_equation :
  ∀ a n : ℕ+, 3^(n : ℕ) = (a : ℕ)^2 - 16 → a = 5 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3887_388779


namespace NUMINAMATH_CALUDE_victors_final_amount_l3887_388745

/-- Calculates the final amount of money Victor has after transactions -/
def final_amount (initial : ℕ) (allowance : ℕ) (additional : ℕ) (expense : ℕ) : ℕ :=
  initial + allowance + additional - expense

/-- Theorem stating that Victor's final amount is $203 -/
theorem victors_final_amount :
  final_amount 145 88 30 60 = 203 := by
  sorry

end NUMINAMATH_CALUDE_victors_final_amount_l3887_388745


namespace NUMINAMATH_CALUDE_salt_mixture_percentage_l3887_388755

theorem salt_mixture_percentage : 
  let initial_volume : ℝ := 70
  let initial_concentration : ℝ := 0.20
  let added_volume : ℝ := 70
  let added_concentration : ℝ := 0.60
  let total_volume : ℝ := initial_volume + added_volume
  let final_concentration : ℝ := (initial_volume * initial_concentration + added_volume * added_concentration) / total_volume
  final_concentration = 0.40 := by sorry

end NUMINAMATH_CALUDE_salt_mixture_percentage_l3887_388755


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l3887_388740

/-- The concentration of glucose in the solution in grams per 100 cubic centimeters -/
def glucose_concentration : ℝ := 10

/-- The volume of solution in cubic centimeters that contains 100 grams of glucose -/
def reference_volume : ℝ := 100

/-- The amount of glucose in grams poured into the container -/
def glucose_in_container : ℝ := 4.5

/-- The volume of solution poured into the container in cubic centimeters -/
def volume_poured : ℝ := 45

theorem glucose_solution_volume :
  (glucose_concentration / reference_volume) * volume_poured = glucose_in_container :=
sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l3887_388740


namespace NUMINAMATH_CALUDE_circular_platform_area_l3887_388794

/-- The area of a circular platform with a diameter of 2 yards is π square yards. -/
theorem circular_platform_area (diameter : ℝ) (h : diameter = 2) : 
  (π * (diameter / 2)^2 : ℝ) = π := by sorry

end NUMINAMATH_CALUDE_circular_platform_area_l3887_388794


namespace NUMINAMATH_CALUDE_expression_evaluation_l3887_388702

theorem expression_evaluation : (2.1 * (49.7 + 0.3) + 15 : ℝ) = 120 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3887_388702


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l3887_388780

/-- The minimum positive value of m that makes the function 
    y = √3 cos(x + m) + sin(x + m) symmetric about the y-axis is π/6. -/
theorem min_shift_for_symmetry :
  let f (x m : ℝ) := Real.sqrt 3 * Real.cos (x + m) + Real.sin (x + m)
  ∃ (m : ℝ), m > 0 ∧ 
    (∀ x, f x m = f (-x) m) ∧ 
    (∀ m' > 0, (∀ x, f x m' = f (-x) m') → m ≤ m') ∧
    m = π / 6 :=
sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l3887_388780


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3887_388711

theorem sin_2theta_value (θ : Real) (h : Real.sin (π / 4 + θ) = 1 / 3) :
  Real.sin (2 * θ) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3887_388711


namespace NUMINAMATH_CALUDE_regions_for_twenty_points_l3887_388775

/-- The number of regions created by chords in a circle --/
def num_regions (n : ℕ) : ℕ :=
  let vertices := n + (n.choose 4)
  let edges := (n * (n - 1) + 2 * (n.choose 4)) / 2
  edges - vertices + 1

/-- Theorem stating the number of regions for 20 points --/
theorem regions_for_twenty_points :
  num_regions 20 = 5036 := by
  sorry

end NUMINAMATH_CALUDE_regions_for_twenty_points_l3887_388775


namespace NUMINAMATH_CALUDE_six_integers_mean_twice_mode_l3887_388708

theorem six_integers_mean_twice_mode (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 
  x ≤ 100 ∧ y ≤ 100 ∧
  y > x ∧
  (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x →
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_six_integers_mean_twice_mode_l3887_388708


namespace NUMINAMATH_CALUDE_revenue_difference_is_164_5_l3887_388799

/-- Represents the types of fruits sold by Kevin --/
inductive Fruit
  | Grapes
  | Mangoes
  | PassionFruits

/-- Represents the pricing and quantity information for each fruit --/
structure FruitInfo where
  price : ℕ
  quantity : ℕ
  discountThreshold : ℕ
  discountRate : ℚ

/-- Calculates the revenue for a given fruit with or without discount --/
def calculateRevenue (info : FruitInfo) (applyDiscount : Bool) : ℚ :=
  let price := if applyDiscount && info.quantity > info.discountThreshold
    then info.price * (1 - info.discountRate)
    else info.price
  price * info.quantity

/-- Theorem: The difference between total revenue without and with discounts is $164.5 --/
theorem revenue_difference_is_164_5 (fruitData : Fruit → FruitInfo) 
    (h1 : fruitData Fruit.Grapes = { price := 15, quantity := 13, discountThreshold := 10, discountRate := 0.1 })
    (h2 : fruitData Fruit.Mangoes = { price := 20, quantity := 20, discountThreshold := 15, discountRate := 0.15 })
    (h3 : fruitData Fruit.PassionFruits = { price := 25, quantity := 17, discountThreshold := 5, discountRate := 0.2 })
    (h4 : (fruitData Fruit.Grapes).quantity + (fruitData Fruit.Mangoes).quantity + (fruitData Fruit.PassionFruits).quantity = 50) :
    (calculateRevenue (fruitData Fruit.Grapes) false +
     calculateRevenue (fruitData Fruit.Mangoes) false +
     calculateRevenue (fruitData Fruit.PassionFruits) false) -
    (calculateRevenue (fruitData Fruit.Grapes) true +
     calculateRevenue (fruitData Fruit.Mangoes) true +
     calculateRevenue (fruitData Fruit.PassionFruits) true) = 164.5 := by
  sorry


end NUMINAMATH_CALUDE_revenue_difference_is_164_5_l3887_388799


namespace NUMINAMATH_CALUDE_new_person_weight_l3887_388768

theorem new_person_weight (initial_total : ℝ) (h1 : initial_total > 0) : 
  let initial_avg := initial_total / 5
  let new_avg := initial_avg + 4
  let new_total := new_avg * 5
  new_total - (initial_total - 50) = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3887_388768


namespace NUMINAMATH_CALUDE_juniper_bones_theorem_l3887_388739

/-- Represents the number of bones Juniper has -/
def juniper_bones (initial : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
  initial + x - y

theorem juniper_bones_theorem (x : ℕ) (y : ℕ) :
  juniper_bones 4 x y = 8 - y :=
by
  sorry

#check juniper_bones_theorem

end NUMINAMATH_CALUDE_juniper_bones_theorem_l3887_388739


namespace NUMINAMATH_CALUDE_triangle_side_ratio_range_l3887_388798

theorem triangle_side_ratio_range (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  S > 0 →
  2 * S = a^2 - (b - c)^2 →
  3 / 5 < b / c ∧ b / c < 5 / 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_ratio_range_l3887_388798


namespace NUMINAMATH_CALUDE_three_inscribed_circles_exist_l3887_388766

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Three circles are inscribed in a larger circle --/
structure InscribedCircles where
  outer : Circle
  inner1 : Circle
  inner2 : Circle
  inner3 : Circle

/-- The property of three circles being equal --/
def equal_circles (c1 c2 c3 : Circle) : Prop :=
  c1.radius = c2.radius ∧ c2.radius = c3.radius

/-- The property of two circles being tangent --/
def tangent_circles (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- The property of a circle being inscribed in another circle --/
def inscribed_circle (outer inner : Circle) : Prop :=
  let (x1, y1) := outer.center
  let (x2, y2) := inner.center
  (x2 - x1)^2 + (y2 - y1)^2 = (outer.radius - inner.radius)^2

/-- Theorem: Three equal circles can be inscribed in a larger circle,
    such that they are tangent to each other and to the larger circle --/
theorem three_inscribed_circles_exist (outer : Circle) :
  ∃ (ic : InscribedCircles),
    ic.outer = outer ∧
    equal_circles ic.inner1 ic.inner2 ic.inner3 ∧
    tangent_circles ic.inner1 ic.inner2 ∧
    tangent_circles ic.inner2 ic.inner3 ∧
    tangent_circles ic.inner3 ic.inner1 ∧
    inscribed_circle outer ic.inner1 ∧
    inscribed_circle outer ic.inner2 ∧
    inscribed_circle outer ic.inner3 :=
  sorry

end NUMINAMATH_CALUDE_three_inscribed_circles_exist_l3887_388766


namespace NUMINAMATH_CALUDE_log_equation_solution_l3887_388756

theorem log_equation_solution (p q : ℝ) (h : 0 < p) (h' : 0 < q) :
  Real.log p + 2 * Real.log q = Real.log (2 * p + q) → p = q / (q^2 - 2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3887_388756


namespace NUMINAMATH_CALUDE_two_talent_students_l3887_388733

theorem two_talent_students (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 50 ∧
  cant_sing = 20 ∧
  cant_dance = 35 ∧
  cant_act = 15 →
  ∃ (two_talents : ℕ),
    two_talents = 30 ∧
    two_talents = (total - cant_sing) + (total - cant_dance) + (total - cant_act) - total :=
by sorry

end NUMINAMATH_CALUDE_two_talent_students_l3887_388733


namespace NUMINAMATH_CALUDE_exists_equal_area_split_line_l3887_388705

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the four circles
def circles : List Circle := [
  { center := (14, 92), radius := 5 },
  { center := (17, 76), radius := 5 },
  { center := (19, 84), radius := 5 },
  { center := (25, 90), radius := 5 }
]

-- Define a line passing through a point with a given slope
structure Line where
  point : ℝ × ℝ
  slope : ℝ

-- Function to calculate the area of a circle segment cut by a line
def circleSegmentArea (c : Circle) (l : Line) : ℝ := sorry

-- Function to calculate the total area of circle segments on one side of the line
def totalSegmentArea (cs : List Circle) (l : Line) : ℝ := sorry

-- Theorem statement
theorem exists_equal_area_split_line :
  ∃ m : ℝ, let l := { point := (17, 76), slope := m }
    totalSegmentArea circles l = (1/2) * (List.sum (circles.map (fun c => π * c.radius^2))) :=
sorry

end NUMINAMATH_CALUDE_exists_equal_area_split_line_l3887_388705


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3887_388720

theorem repeating_decimal_to_fraction :
  ∀ (a b : ℕ) (x : ℚ),
    (x = 0.4 + (31 : ℚ) / (990 : ℚ)) →
    (x = (427 : ℚ) / (990 : ℚ)) :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3887_388720


namespace NUMINAMATH_CALUDE_rosa_initial_flowers_l3887_388748

theorem rosa_initial_flowers (flowers_from_andre : ℝ) (total_flowers : ℕ) :
  flowers_from_andre = 90.0 →
  total_flowers = 157 →
  total_flowers - Int.floor flowers_from_andre = 67 := by
  sorry

end NUMINAMATH_CALUDE_rosa_initial_flowers_l3887_388748


namespace NUMINAMATH_CALUDE_triangle_area_is_36_l3887_388743

/-- The area of the triangle bounded by y = x, y = -x, and y = 6 -/
def triangle_area : ℝ := 36

/-- The line y = x -/
def line1 (x : ℝ) : ℝ := x

/-- The line y = -x -/
def line2 (x : ℝ) : ℝ := -x

/-- The line y = 6 -/
def line3 : ℝ := 6

theorem triangle_area_is_36 :
  triangle_area = 36 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_36_l3887_388743


namespace NUMINAMATH_CALUDE_shoe_promotion_savings_difference_l3887_388726

/-- Calculates the savings difference between two promotions for shoe purchases -/
theorem shoe_promotion_savings_difference : 
  let original_price : ℝ := 50
  let promotion_c_discount : ℝ := 0.20
  let promotion_d_discount : ℝ := 15
  let cost_c : ℝ := original_price + (original_price * (1 - promotion_c_discount))
  let cost_d : ℝ := original_price + (original_price - promotion_d_discount)
  cost_c - cost_d = 5 := by sorry

end NUMINAMATH_CALUDE_shoe_promotion_savings_difference_l3887_388726


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3887_388770

theorem square_sum_theorem (p q : ℝ) 
  (h1 : p * q = 9)
  (h2 : p^2 * q + q^2 * p + p + q = 70) :
  p^2 + q^2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3887_388770


namespace NUMINAMATH_CALUDE_tunnel_length_is_one_mile_l3887_388730

/-- Calculates the length of a tunnel given train and time information -/
def tunnel_length (train_length : ℝ) (train_speed : ℝ) (total_time : ℝ) (front_exit_time : ℝ) : ℝ :=
  train_speed * front_exit_time - train_length

/-- Theorem stating that under given conditions, the tunnel length is 1 mile -/
theorem tunnel_length_is_one_mile 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (total_time : ℝ) 
  (front_exit_time : ℝ)
  (h1 : train_length = 1)
  (h2 : train_speed = 30 / 60)  -- 30 miles per hour converted to miles per minute
  (h3 : total_time = 5)
  (h4 : front_exit_time = 3)  -- 5 minutes - 2 minutes
  : tunnel_length train_length train_speed total_time front_exit_time = 1 := by
  sorry

#check tunnel_length_is_one_mile

end NUMINAMATH_CALUDE_tunnel_length_is_one_mile_l3887_388730


namespace NUMINAMATH_CALUDE_distance_to_symmetry_axis_range_l3887_388793

variable (a b c : ℝ)
variable (f : ℝ → ℝ)

theorem distance_to_symmetry_axis_range 
  (ha : a > 0)
  (hf : f = fun x ↦ a * x^2 + b * x + c)
  (htangent : ∀ x₀, 0 ≤ (2 * a * x₀ + b) ∧ (2 * a * x₀ + b) ≤ 1) :
  ∃ d : Set ℝ, d = {x | 0 ≤ x ∧ x ≤ 1 / (2 * a)} ∧
    ∀ x₀, Set.Mem (|x₀ + b / (2 * a)|) d :=
by sorry

end NUMINAMATH_CALUDE_distance_to_symmetry_axis_range_l3887_388793


namespace NUMINAMATH_CALUDE_martinez_family_height_l3887_388767

def chiquita_height : ℝ := 5

def mr_martinez_height : ℝ := chiquita_height + 2

def mrs_martinez_height : ℝ := chiquita_height - 1

def son_height : ℝ := chiquita_height + 3

def combined_height : ℝ := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_height : combined_height = 24 := by
  sorry

end NUMINAMATH_CALUDE_martinez_family_height_l3887_388767


namespace NUMINAMATH_CALUDE_mrs_wonderful_class_size_l3887_388752

theorem mrs_wonderful_class_size :
  ∀ (girls : ℕ) (boys : ℕ),
  boys = girls + 3 →
  girls * girls + boys * boys + 10 + 8 = 450 →
  girls + boys = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_wonderful_class_size_l3887_388752


namespace NUMINAMATH_CALUDE_f_composition_eq_exp_l3887_388789

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^x else 3*x - 1

theorem f_composition_eq_exp (a : ℝ) :
  {a : ℝ | f (f a) = 2^(f a)} = Set.Ici (2/3) := by sorry

end NUMINAMATH_CALUDE_f_composition_eq_exp_l3887_388789


namespace NUMINAMATH_CALUDE_charlie_extra_cost_l3887_388777

/-- Charlie's cell phone plan and usage details -/
structure CellPhonePlan where
  included_data : ℕ
  extra_cost_per_gb : ℕ
  week1_usage : ℕ
  week2_usage : ℕ
  week3_usage : ℕ
  week4_usage : ℕ

/-- Calculate the extra cost for Charlie's cell phone usage -/
def calculate_extra_cost (plan : CellPhonePlan) : ℕ :=
  let total_usage := plan.week1_usage + plan.week2_usage + plan.week3_usage + plan.week4_usage
  let over_limit := if total_usage > plan.included_data then total_usage - plan.included_data else 0
  over_limit * plan.extra_cost_per_gb

/-- Theorem: Charlie's extra cost is $120.00 -/
theorem charlie_extra_cost :
  let charlie_plan : CellPhonePlan := {
    included_data := 8,
    extra_cost_per_gb := 10,
    week1_usage := 2,
    week2_usage := 3,
    week3_usage := 5,
    week4_usage := 10
  }
  calculate_extra_cost charlie_plan = 120 := by
  sorry

end NUMINAMATH_CALUDE_charlie_extra_cost_l3887_388777


namespace NUMINAMATH_CALUDE_cerulean_somewhat_green_l3887_388790

/-- The number of people surveyed -/
def total_surveyed : ℕ := 120

/-- The number of people who think cerulean is "kind of blue" -/
def kind_of_blue : ℕ := 80

/-- The number of people who think cerulean is both "kind of blue" and "somewhat green" -/
def both : ℕ := 35

/-- The number of people who think cerulean is neither "kind of blue" nor "somewhat green" -/
def neither : ℕ := 20

/-- The theorem states that the number of people who believe cerulean is "somewhat green" is 55 -/
theorem cerulean_somewhat_green : 
  total_surveyed - kind_of_blue + both = 55 :=
by sorry

end NUMINAMATH_CALUDE_cerulean_somewhat_green_l3887_388790
