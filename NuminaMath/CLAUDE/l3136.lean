import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_simplification_l3136_313695

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^2 + 7 * x - 3) - (x^2 + 5 * x - 12) = x^2 + 2 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3136_313695


namespace NUMINAMATH_CALUDE_complex_fraction_equals_minus_one_plus_i_l3136_313617

theorem complex_fraction_equals_minus_one_plus_i :
  (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_minus_one_plus_i_l3136_313617


namespace NUMINAMATH_CALUDE_jerry_water_usage_l3136_313614

/-- Calculates the total water usage for Jerry's household in July --/
def total_water_usage (drinking_cooking : ℕ) (shower_usage : ℕ) (num_showers : ℕ) 
  (pool_length : ℕ) (pool_width : ℕ) (pool_height : ℕ) : ℕ :=
  drinking_cooking + (shower_usage * num_showers) + (pool_length * pool_width * pool_height)

theorem jerry_water_usage :
  total_water_usage 100 20 15 10 10 6 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jerry_water_usage_l3136_313614


namespace NUMINAMATH_CALUDE_sqrt_three_expression_equality_l3136_313618

theorem sqrt_three_expression_equality : 
  (Real.sqrt 3 + 1)^2 - Real.sqrt 12 + 2 * Real.sqrt (1/3) = 4 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_expression_equality_l3136_313618


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3136_313626

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3136_313626


namespace NUMINAMATH_CALUDE_die_roll_probability_l3136_313668

def standard_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def roll_twice : Finset (ℕ × ℕ) :=
  standard_die.product standard_die

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 3), (2, 6)}

theorem die_roll_probability :
  (favorable_outcomes.card : ℚ) / roll_twice.card = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3136_313668


namespace NUMINAMATH_CALUDE_compute_expression_l3136_313625

theorem compute_expression : 9 * (2/3)^4 = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3136_313625


namespace NUMINAMATH_CALUDE_thirty_thousand_squared_l3136_313611

theorem thirty_thousand_squared :
  (30000 : ℕ) ^ 2 = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_thirty_thousand_squared_l3136_313611


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3136_313664

open Real

noncomputable def series_sum (n : ℕ) : ℝ := 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

theorem infinite_series_sum :
  (∑' n, series_sum n) = 1/4 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3136_313664


namespace NUMINAMATH_CALUDE_mikes_video_games_l3136_313692

theorem mikes_video_games (non_working : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) : 
  non_working = 9 → price_per_game = 5 → total_earnings = 30 →
  non_working + (total_earnings / price_per_game) = 15 :=
by sorry

end NUMINAMATH_CALUDE_mikes_video_games_l3136_313692


namespace NUMINAMATH_CALUDE_bus_driver_overtime_rate_increase_bus_driver_overtime_rate_increase_approx_l3136_313647

/-- Calculates the percentage increase in overtime rate compared to regular rate for a bus driver --/
theorem bus_driver_overtime_rate_increase 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) : ℝ :=
  let overtime_hours := total_hours - regular_hours
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  let percentage_increase := (overtime_rate - regular_rate) / regular_rate * 100
  percentage_increase

/-- The percentage increase in overtime rate is approximately 74.93% --/
theorem bus_driver_overtime_rate_increase_approx :
  ∃ ε > 0, abs (bus_driver_overtime_rate_increase 14 40 998 57.88 - 74.93) < ε :=
sorry

end NUMINAMATH_CALUDE_bus_driver_overtime_rate_increase_bus_driver_overtime_rate_increase_approx_l3136_313647


namespace NUMINAMATH_CALUDE_sean_charles_whistle_difference_l3136_313655

/-- 
Given that Sean has 45 whistles and Charles has 13 whistles, 
prove that Sean has 32 more whistles than Charles.
-/
theorem sean_charles_whistle_difference :
  ∀ (sean_whistles charles_whistles : ℕ),
    sean_whistles = 45 →
    charles_whistles = 13 →
    sean_whistles - charles_whistles = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_sean_charles_whistle_difference_l3136_313655


namespace NUMINAMATH_CALUDE_ratio_solution_set_l3136_313656

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the solution set of f(x) ≥ 0
def solution_set_f (f : ℝ → ℝ) : Set ℝ := {x | f x ≥ 0}

-- Define the solution set of g(x) ≥ 0
def solution_set_g (g : ℝ → ℝ) : Set ℝ := {x | g x ≥ 0}

-- Define the solution set of f(x)/g(x) > 0
def solution_set_ratio (f g : ℝ → ℝ) : Set ℝ := {x | f x / g x > 0}

-- State the theorem
theorem ratio_solution_set 
  (h1 : solution_set_f f = Set.Icc 1 2) 
  (h2 : solution_set_g g = ∅) : 
  solution_set_ratio f g = Set.Ioi 2 ∪ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_solution_set_l3136_313656


namespace NUMINAMATH_CALUDE_no_real_solutions_l3136_313696

theorem no_real_solutions :
  ¬∃ (x : ℝ), Real.sqrt (x + 16) - 8 / Real.sqrt (x + 16) + 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3136_313696


namespace NUMINAMATH_CALUDE_four_holes_when_unfolded_l3136_313697

/-- Represents a rectangular sheet of paper -/
structure Paper :=
  (width : ℝ)
  (height : ℝ)
  (holes : List (ℝ × ℝ))

/-- Represents the state of the paper after folding -/
inductive FoldState
  | Unfolded
  | DiagonalFold
  | HalfFold
  | FinalFold

/-- Represents a folding operation -/
def fold (p : Paper) (state : FoldState) : Paper :=
  sorry

/-- Represents the operation of punching a hole -/
def punchHole (p : Paper) (x : ℝ) (y : ℝ) : Paper :=
  sorry

/-- Represents the unfolding operation -/
def unfold (p : Paper) : Paper :=
  sorry

/-- The main theorem to prove -/
theorem four_holes_when_unfolded (p : Paper) :
  let p1 := fold p FoldState.DiagonalFold
  let p2 := fold p1 FoldState.HalfFold
  let p3 := fold p2 FoldState.FinalFold
  let p4 := punchHole p3 (p.width / 2) (p.height / 2)
  let final := unfold p4
  final.holes.length = 4 :=
sorry

end NUMINAMATH_CALUDE_four_holes_when_unfolded_l3136_313697


namespace NUMINAMATH_CALUDE_find_k_value_l3136_313601

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k) * (x + 3*k) = x^3 + k*(x^2 - 2*x - 8)) → k = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l3136_313601


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l3136_313608

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ y - a * x^2
  ∃! p : ℝ × ℝ, f p = 0 ∧ p.1 = 0 ∧ p.2 = 1 / (4 * a) :=
sorry

/-- The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ y - 4 * x^2
  ∃! p : ℝ × ℝ, f p = 0 ∧ p.1 = 0 ∧ p.2 = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l3136_313608


namespace NUMINAMATH_CALUDE_spheres_in_cone_radius_l3136_313606

/-- A right circular cone -/
structure Cone :=
  (base_radius : ℝ)
  (height : ℝ)

/-- A sphere -/
structure Sphere :=
  (radius : ℝ)

/-- Configuration of four spheres in a cone -/
structure SpheresInCone :=
  (cone : Cone)
  (sphere : Sphere)
  (tangent_to_base : Prop)
  (tangent_to_each_other : Prop)
  (tangent_to_side : Prop)

/-- Theorem stating the radius of spheres in the given configuration -/
theorem spheres_in_cone_radius 
  (config : SpheresInCone)
  (h_base_radius : config.cone.base_radius = 6)
  (h_height : config.cone.height = 15)
  (h_tangent_base : config.tangent_to_base)
  (h_tangent_each_other : config.tangent_to_each_other)
  (h_tangent_side : config.tangent_to_side) :
  config.sphere.radius = 15 / 11 :=
sorry

end NUMINAMATH_CALUDE_spheres_in_cone_radius_l3136_313606


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3136_313665

theorem probability_nine_heads_in_twelve_flips :
  let n : ℕ := 12  -- total number of flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1-p)^(n-k) = 220/4096 := by
sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l3136_313665


namespace NUMINAMATH_CALUDE_combined_distance_is_122_l3136_313622

-- Define the fuel-to-distance ratios for both cars
def car_A_ratio : Rat := 4 / 7
def car_B_ratio : Rat := 3 / 5

-- Define the amount of fuel used by each car
def car_A_fuel : ℕ := 44
def car_B_fuel : ℕ := 27

-- Function to calculate distance given fuel and ratio
def calculate_distance (fuel : ℕ) (ratio : Rat) : ℚ :=
  (fuel : ℚ) * (ratio.den : ℚ) / (ratio.num : ℚ)

-- Theorem stating the combined distance is 122 miles
theorem combined_distance_is_122 :
  (calculate_distance car_A_fuel car_A_ratio + calculate_distance car_B_fuel car_B_ratio) = 122 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_122_l3136_313622


namespace NUMINAMATH_CALUDE_fungi_at_128pm_l3136_313689

/-- The number of fungi at a given time, given an initial population and doubling time -/
def fungiBehavior (initialPopulation : ℕ) (doublingTime : ℕ) (elapsedTime : ℕ) : ℕ :=
  initialPopulation * 2 ^ (elapsedTime / doublingTime)

/-- Theorem stating the number of fungi at 1:28 p.m. given the initial conditions -/
theorem fungi_at_128pm (initialPopulation : ℕ) (doublingTime : ℕ) (elapsedTime : ℕ) :
  initialPopulation = 30 → doublingTime = 4 → elapsedTime = 28 →
  fungiBehavior initialPopulation doublingTime elapsedTime = 3840 := by
  sorry

#check fungi_at_128pm

end NUMINAMATH_CALUDE_fungi_at_128pm_l3136_313689


namespace NUMINAMATH_CALUDE_connie_marbles_left_l3136_313675

/-- Given Connie's initial number of marbles and the number she gave away,
    calculate how many marbles she has left. -/
def marblesLeft (initialMarbles gaveAway : ℕ) : ℕ :=
  initialMarbles - gaveAway

/-- Theorem stating that if Connie started with 143 marbles and gave away 73,
    she has 70 marbles left. -/
theorem connie_marbles_left : marblesLeft 143 73 = 70 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_left_l3136_313675


namespace NUMINAMATH_CALUDE_ages_problem_l3136_313650

/-- The present ages of individuals A, B, C, and D satisfy the given conditions. -/
theorem ages_problem (A B C D : ℕ) : 
  (C + 10 = 3 * (A + 10)) →  -- In 10 years, C will be 3 times as old as A
  (A = 2 * (B - 10)) →       -- A will be twice as old as B was 10 years ago
  (A = B + 12) →             -- A is now 12 years older than B
  (B = D + 5) →              -- B is 5 years older than D
  (D = C / 2) →              -- D is half the age of C
  (A = 88 ∧ B = 76 ∧ C = 142 ∧ D = 71) :=
by sorry

end NUMINAMATH_CALUDE_ages_problem_l3136_313650


namespace NUMINAMATH_CALUDE_clay_pot_flower_cost_difference_clay_pot_flower_cost_difference_proof_l3136_313621

/-- The cost difference between a clay pot and flowers -/
theorem clay_pot_flower_cost_difference : ℝ → ℝ → ℝ → Prop :=
  fun flower_cost clay_pot_cost soil_cost =>
    flower_cost = 9 ∧
    clay_pot_cost > flower_cost ∧
    soil_cost = flower_cost - 2 ∧
    flower_cost + clay_pot_cost + soil_cost = 45 →
    clay_pot_cost - flower_cost = 20

/-- Proof of the clay_pot_flower_cost_difference theorem -/
theorem clay_pot_flower_cost_difference_proof :
  ∃ (flower_cost clay_pot_cost soil_cost : ℝ),
    clay_pot_flower_cost_difference flower_cost clay_pot_cost soil_cost :=
by
  sorry

end NUMINAMATH_CALUDE_clay_pot_flower_cost_difference_clay_pot_flower_cost_difference_proof_l3136_313621


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l3136_313663

/-- Given two algebraic terms are like terms, prove that the sum of their exponents is 5 -/
theorem like_terms_exponent_sum (a b : ℝ) (m n : ℕ) : 
  (∃ (k : ℝ), k * a^(2*m) * b^3 = 5 * a^6 * b^(n+1)) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l3136_313663


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l3136_313654

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_number_with_given_hcf_lcm_factors 
  (a b : ℕ) 
  (hcf_prime : is_prime 31) 
  (hcf_val : Nat.gcd a b = 31) 
  (lcm_factors : Nat.lcm a b = 31 * 13 * 14 * 17) :
  max a b = 95914 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l3136_313654


namespace NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l3136_313648

theorem square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three :
  let x : ℝ := Real.sqrt 2 + 1
  (x^2 - 2*x + 2 : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_square_root_two_plus_one_squared_minus_two_times_plus_two_equals_three_l3136_313648


namespace NUMINAMATH_CALUDE_fraction_equivalence_and_decimal_l3136_313693

theorem fraction_equivalence_and_decimal : 
  let original : ℚ := 2 / 4
  let equiv1 : ℚ := 6 / 12
  let equiv2 : ℚ := 20 / 40
  let decimal : ℝ := 0.5
  (original = equiv1) ∧ (original = equiv2) ∧ (original = decimal) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_and_decimal_l3136_313693


namespace NUMINAMATH_CALUDE_tan_sum_special_case_l3136_313632

theorem tan_sum_special_case :
  let tan55 := Real.tan (55 * π / 180)
  let tan65 := Real.tan (65 * π / 180)
  tan55 + tan65 - Real.sqrt 3 * tan55 * tan65 = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_case_l3136_313632


namespace NUMINAMATH_CALUDE_binomial_1500_1_l3136_313681

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1500_1_l3136_313681


namespace NUMINAMATH_CALUDE_p_q_ratio_equals_ways_ratio_l3136_313673

/-- The number of balls -/
def n : ℕ := 20

/-- The number of bins -/
def k : ℕ := 4

/-- The probability of a 3-5-6-6 distribution -/
def p : ℚ := sorry

/-- The probability of a 5-5-5-5 distribution -/
def q : ℚ := sorry

/-- The number of ways to distribute n balls into k bins with a given distribution -/
def ways_to_distribute (n : ℕ) (k : ℕ) (distribution : List ℕ) : ℕ := sorry

/-- The ratio of p to q is equal to the ratio of the number of ways to achieve each distribution -/
theorem p_q_ratio_equals_ways_ratio : 
  p / q = (ways_to_distribute n k [3, 5, 6, 6] * 12) / ways_to_distribute n k [5, 5, 5, 5] := by
  sorry

end NUMINAMATH_CALUDE_p_q_ratio_equals_ways_ratio_l3136_313673


namespace NUMINAMATH_CALUDE_exactly_one_solution_l3136_313667

-- Define the function g₀
def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 150|

-- Define the function gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 1

-- Theorem statement
theorem exactly_one_solution :
  ∃! x, g 100 x = 0 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_solution_l3136_313667


namespace NUMINAMATH_CALUDE_quarterback_passes_l3136_313603

theorem quarterback_passes (left right center : ℕ) : 
  left = 12 →
  right = 2 * left →
  center = left + 2 →
  left + right + center = 50 := by
sorry

end NUMINAMATH_CALUDE_quarterback_passes_l3136_313603


namespace NUMINAMATH_CALUDE_theater_seats_l3136_313644

theorem theater_seats (first_row : ℕ) (last_row : ℕ) (total_seats : ℕ) (num_rows : ℕ) 
  (h1 : first_row = 14)
  (h2 : last_row = 50)
  (h3 : total_seats = 416)
  (h4 : num_rows = 13) :
  ∃ (additional_seats : ℕ), 
    (additional_seats = 3) ∧ 
    (last_row = first_row + (num_rows - 1) * additional_seats) ∧
    (total_seats = (num_rows * (first_row + last_row)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_theater_seats_l3136_313644


namespace NUMINAMATH_CALUDE_two_equal_real_roots_l3136_313684

def quadratic_equation (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem two_equal_real_roots (a b c : ℝ) (ha : a ≠ 0) :
  a = 4 ∧ b = -4 ∧ c = 1 →
  ∃ x : ℝ, quadratic_equation a b c x ∧
    ∀ y : ℝ, quadratic_equation a b c y → y = x :=
by
  sorry

end NUMINAMATH_CALUDE_two_equal_real_roots_l3136_313684


namespace NUMINAMATH_CALUDE_spicy_hot_noodles_count_l3136_313633

theorem spicy_hot_noodles_count (total_plates lobster_rolls seafood_noodles : ℕ) 
  (h1 : total_plates = 55)
  (h2 : lobster_rolls = 25)
  (h3 : seafood_noodles = 16) :
  total_plates - (lobster_rolls + seafood_noodles) = 14 := by
  sorry

end NUMINAMATH_CALUDE_spicy_hot_noodles_count_l3136_313633


namespace NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l3136_313671

/-- The perimeter of a rectangular garden with width 24 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is equal to 64 meters. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (garden_width playground_length playground_width garden_perimeter : ℝ) =>
    garden_width = 24 ∧
    playground_length = 16 ∧
    playground_width = 12 ∧
    garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width →
    garden_perimeter = 2 * (garden_width + (playground_length * playground_width / garden_width)) →
    garden_perimeter = 64

/-- Proof of the garden_perimeter theorem -/
theorem garden_perimeter_proof : garden_perimeter 24 16 12 64 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l3136_313671


namespace NUMINAMATH_CALUDE_trapezium_area_l3136_313690

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 10) (hh : h = 10) :
  (a + b) * h / 2 = 150 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_l3136_313690


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3136_313687

theorem line_inclination_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 2 = 0 →
  Real.arctan (1 / Real.sqrt 3) = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3136_313687


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_l3136_313602

theorem square_perimeter_from_rectangle (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 32)
  (h2 : rectangle_width = 10)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  ∃ (square_side : ℝ), 
    square_side^2 = 5 * (rectangle_length * rectangle_width) ∧ 
    4 * square_side = 160 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_l3136_313602


namespace NUMINAMATH_CALUDE_total_savings_calculation_l3136_313612

def initial_savings : ℕ := 849400
def monthly_income : ℕ := 110000
def monthly_expenses : ℕ := 58500
def months : ℕ := 5

theorem total_savings_calculation :
  initial_savings + months * monthly_income - months * monthly_expenses = 1106900 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_calculation_l3136_313612


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3136_313694

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℕ), x > 0 ∧
    (a * x).gcd (b * x) = 1 ∧
    (a * x).gcd (c * x) = 1 ∧
    (b * x).gcd (c * x) = 1 ∧
    a * x + b * x + c * x = total ∧
    min (a * x) (min (b * x) (c * x)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3136_313694


namespace NUMINAMATH_CALUDE_log_equation_solution_l3136_313680

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem log_equation_solution :
  ∀ (X Y Z : ℕ),
    X > 0 ∧ Y > 0 ∧ Z > 0 →
    is_coprime X Y ∧ is_coprime Y Z ∧ is_coprime X Z →
    X * (Real.log 3 / Real.log 100) + Y * (Real.log 5 / Real.log 100) = Z →
    X + Y + Z = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3136_313680


namespace NUMINAMATH_CALUDE_inequality_proof_l3136_313642

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 - a*b + b^2) ≥ (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3136_313642


namespace NUMINAMATH_CALUDE_no_solution_when_k_equals_five_l3136_313666

theorem no_solution_when_k_equals_five :
  ∀ x : ℝ, x ≠ 2 → x ≠ 6 → (x - 1) / (x - 2) ≠ (x - 5) / (x - 6) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_when_k_equals_five_l3136_313666


namespace NUMINAMATH_CALUDE_polynomial_value_equals_one_l3136_313683

theorem polynomial_value_equals_one (x₀ : ℂ) (h : x₀^2 + x₀ + 2 = 0) :
  x₀^4 + 2*x₀^3 + 3*x₀^2 + 2*x₀ + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equals_one_l3136_313683


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l3136_313699

/-- Two parallel lines with a given distance between them -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  distance : ℝ
  parallel : a / b = c / d
  dist_formula : distance = |c - a| / Real.sqrt (c^2 + d^2)

/-- The theorem to be proved -/
theorem parallel_lines_sum (lines : ParallelLines) 
  (h1 : lines.a = 3 ∧ lines.b = 4)
  (h2 : lines.c = 6)
  (h3 : lines.distance = 3) :
  (lines.d + lines.c = -12) ∨ (lines.d + lines.c = 48) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l3136_313699


namespace NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l3136_313631

theorem integral_3x_plus_sin_x (x : Real) : 
  ∫ x in (0)..(π/2), (3*x + Real.sin x) = 3*π^2/8 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l3136_313631


namespace NUMINAMATH_CALUDE_correct_subtraction_result_l3136_313629

/-- 
Given a subtraction problem where:
- The tens digit 7 was mistaken for 9
- The ones digit 3 was mistaken for 8
- The mistaken subtraction resulted in a difference of 76

Prove that the correct difference is 51.
-/
theorem correct_subtraction_result : 
  ∀ (original_tens original_ones mistaken_tens mistaken_ones mistaken_difference : ℕ),
  original_tens = 7 →
  original_ones = 3 →
  mistaken_tens = 9 →
  mistaken_ones = 8 →
  mistaken_difference = 76 →
  (mistaken_tens * 10 + mistaken_ones) - (original_tens * 10 + original_ones) = mistaken_difference →
  (original_tens * 10 + original_ones) - 
    ((mistaken_tens * 10 + mistaken_ones) - (original_tens * 10 + original_ones)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_result_l3136_313629


namespace NUMINAMATH_CALUDE_honeycomb_thickness_scientific_notation_l3136_313678

theorem honeycomb_thickness_scientific_notation :
  0.000073 = 7.3 * 10^(-5) := by
  sorry

end NUMINAMATH_CALUDE_honeycomb_thickness_scientific_notation_l3136_313678


namespace NUMINAMATH_CALUDE_sin_two_alpha_zero_l3136_313662

theorem sin_two_alpha_zero (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = Real.sin x - Real.cos x) 
  (h2 : f α = 1) : 
  Real.sin (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_zero_l3136_313662


namespace NUMINAMATH_CALUDE_square_difference_l3136_313685

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : 
  (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3136_313685


namespace NUMINAMATH_CALUDE_square_land_equation_l3136_313637

theorem square_land_equation (a p : ℝ) (h1 : p = 36) : 
  (∃ (s : ℝ), s > 0 ∧ a = s^2 ∧ p = 4*s) → 
  (5*a = 10*p + 45) := by
sorry

end NUMINAMATH_CALUDE_square_land_equation_l3136_313637


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3136_313600

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (x - 3) (x + 2) (2*x - 1) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3136_313600


namespace NUMINAMATH_CALUDE_inequality_implies_values_l3136_313688

theorem inequality_implies_values (a b : ℤ) 
  (h : ∀ x : ℝ, x ≤ 0 → (a * x + 2) * (x^2 + 2 * b) ≤ 0) : 
  a = 1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_values_l3136_313688


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3136_313661

theorem smallest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 40 →           -- One angle is 40°
  c = 3 * b →        -- The other two angles are in the ratio 1:3
  min a (min b c) = 35 :=  -- The smallest angle is 35°
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3136_313661


namespace NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l3136_313679

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distributeBalls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_balls_4_boxes_l3136_313679


namespace NUMINAMATH_CALUDE_exists_k_for_A_l3136_313646

theorem exists_k_for_A (n m : ℕ) (hn : n ≥ 2) (hm : m ≥ 2) :
  ∃ k : ℕ, ((n + Real.sqrt (n^2 - 4)) / 2)^m = (k + Real.sqrt (k^2 - 4)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_k_for_A_l3136_313646


namespace NUMINAMATH_CALUDE_dilation_determinant_l3136_313674

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- The determinant of a 2x2 matrix -/
def det2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

theorem dilation_determinant :
  let E := dilationMatrix 12
  det2x2 E = 144 := by sorry

end NUMINAMATH_CALUDE_dilation_determinant_l3136_313674


namespace NUMINAMATH_CALUDE_reciprocal_of_difference_l3136_313686

-- Define repeating decimals
def repeating_decimal_1 : ℚ := 1/9
def repeating_decimal_6 : ℚ := 2/3

-- State the theorem
theorem reciprocal_of_difference : (repeating_decimal_6 - repeating_decimal_1)⁻¹ = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_difference_l3136_313686


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3136_313630

/-- Calculates the total ticket sales for a theater performance --/
theorem theater_ticket_sales 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_attendance : ℕ) 
  (child_attendance : ℕ) : 
  adult_price = 8 → 
  child_price = 1 → 
  total_attendance = 22 → 
  child_attendance = 18 → 
  (total_attendance - child_attendance) * adult_price + child_attendance * child_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3136_313630


namespace NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l3136_313669

theorem cosine_sine_ratio_equals_sqrt_three : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_ratio_equals_sqrt_three_l3136_313669


namespace NUMINAMATH_CALUDE_all_points_on_line_l3136_313652

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The set of n points on the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- The property that any line through two points contains at least one more point -/
def ThreePointProperty (points : PointSet n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j →
    ∃ (l : Line), pointOnLine (points i) l ∧ pointOnLine (points j) l →
      ∃ (m : Fin n), m ≠ i ∧ m ≠ j ∧ pointOnLine (points m) l

/-- The theorem statement -/
theorem all_points_on_line (n : ℕ) (points : PointSet n) 
  (h : ThreePointProperty points) : 
  ∃ (l : Line), ∀ (i : Fin n), pointOnLine (points i) l :=
sorry

end NUMINAMATH_CALUDE_all_points_on_line_l3136_313652


namespace NUMINAMATH_CALUDE_fabric_order_calculation_l3136_313672

/-- The conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- David's waist size in inches -/
def waist_size : ℝ := 38

/-- The extra allowance for waistband sewing in centimeters -/
def waistband_allowance : ℝ := 2

/-- The total length of fabric David should order in centimeters -/
def total_fabric_length : ℝ := waist_size * inch_to_cm + waistband_allowance

theorem fabric_order_calculation :
  total_fabric_length = 98.52 :=
by sorry

end NUMINAMATH_CALUDE_fabric_order_calculation_l3136_313672


namespace NUMINAMATH_CALUDE_statement_1_false_statement_2_true_statement_3_true_statement_4_false_statement_5_true_l3136_313613

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Statement 1
theorem statement_1_false : ∃ x y z : ℝ, (heartsuit (heartsuit x y) z) ≠ (heartsuit x (heartsuit y z)) := by sorry

-- Statement 2
theorem statement_2_true : ∀ x y : ℝ, 3 * (heartsuit x y) = heartsuit (3 * x) (3 * y) := by sorry

-- Statement 3
theorem statement_3_true : ∀ x y : ℝ, heartsuit x (-y) = heartsuit (-x) y := by sorry

-- Statement 4
theorem statement_4_false : ∃ x : ℝ, heartsuit x x ≠ x := by sorry

-- Statement 5
theorem statement_5_true : ∀ x y : ℝ, heartsuit x y ≥ 0 := by sorry

end NUMINAMATH_CALUDE_statement_1_false_statement_2_true_statement_3_true_statement_4_false_statement_5_true_l3136_313613


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l3136_313636

theorem purely_imaginary_z (m : ℝ) : 
  let z : ℂ := (m - 1) * (m - 2) + (m - 2) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l3136_313636


namespace NUMINAMATH_CALUDE_train_length_calculation_l3136_313627

/-- Calculates the length of a train given its speed, the platform length, and the time taken to cross the platform. -/
theorem train_length_calculation (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 55 → 
  platform_length = 620 → 
  crossing_time = 71.99424046076314 → 
  ∃ (train_length : ℝ), (train_length ≥ 479.9 ∧ train_length ≤ 480.1) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3136_313627


namespace NUMINAMATH_CALUDE_hannah_easter_eggs_l3136_313698

theorem hannah_easter_eggs 
  (total : ℕ) 
  (helen : ℕ) 
  (hannah : ℕ) 
  (h1 : total = 63)
  (h2 : hannah = 2 * helen)
  (h3 : total = helen + hannah) : 
  hannah = 42 := by
sorry

end NUMINAMATH_CALUDE_hannah_easter_eggs_l3136_313698


namespace NUMINAMATH_CALUDE_magnitude_z_l3136_313658

open Complex

theorem magnitude_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : abs w = Real.sqrt 29) :
  abs z = (25 * Real.sqrt 29) / 29 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_l3136_313658


namespace NUMINAMATH_CALUDE_drawn_games_in_specific_tournament_l3136_313615

/-- Represents a chess tournament. -/
structure ChessTournament where
  participants : Nat
  total_matches : Nat
  wins_per_participant : Nat
  has_growing_lists : Bool

/-- Calculates the number of drawn games in a chess tournament. -/
def drawn_games (tournament : ChessTournament) : Nat :=
  tournament.total_matches - (tournament.participants * tournament.wins_per_participant)

/-- Theorem stating the number of drawn games in the specific tournament. -/
theorem drawn_games_in_specific_tournament :
  ∀ (t : ChessTournament),
    t.participants = 12 ∧
    t.total_matches = (12 * 11) / 2 ∧
    t.wins_per_participant = 1 ∧
    t.has_growing_lists = true →
    drawn_games t = 54 := by
  sorry

end NUMINAMATH_CALUDE_drawn_games_in_specific_tournament_l3136_313615


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l3136_313610

/-- Proves that erasing 33 cm from a 1 m line results in a 67 cm line -/
theorem line_length_after_erasing :
  ∀ (initial_length : ℝ) (erased_length : ℝ),
  initial_length = 1 →
  erased_length = 33 / 100 →
  (initial_length - erased_length) * 100 = 67 := by
sorry

end NUMINAMATH_CALUDE_line_length_after_erasing_l3136_313610


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3136_313657

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  sin A / a = sin B / b →
  sin B / b = sin C / c →
  2 * sin A - sin B = 2 * sin C * cos B →
  c = 2 →
  C = π / 3 ∧ ∀ x, (2 * a - b = x) → -2 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3136_313657


namespace NUMINAMATH_CALUDE_gcd_lcm_product_300_l3136_313651

theorem gcd_lcm_product_300 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 300) :
  ∃! (s : Finset ℕ), s.card = 8 ∧ ∀ d, d ∈ s ↔ ∃ (x y : ℕ+), x * y = 300 ∧ Nat.gcd x y = d :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_300_l3136_313651


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3136_313653

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := 2 * x + 6
  length > 0 ∧ width > 0 →
  length * width = 2 * (length + width) →
  x = (-3 + Real.sqrt 33) / 4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3136_313653


namespace NUMINAMATH_CALUDE_jenny_recycling_money_is_160_l3136_313639

/-- Calculates the money Jenny makes from recycling cans and bottles -/
def jenny_recycling_money : ℕ :=
let bottle_weight : ℕ := 6
let can_weight : ℕ := 2
let total_capacity : ℕ := 100
let cans_collected : ℕ := 20
let bottle_price : ℕ := 10
let can_price : ℕ := 3
let remaining_capacity : ℕ := total_capacity - (can_weight * cans_collected)
let bottles_collected : ℕ := remaining_capacity / bottle_weight
bottles_collected * bottle_price + cans_collected * can_price

theorem jenny_recycling_money_is_160 :
  jenny_recycling_money = 160 := by
sorry

end NUMINAMATH_CALUDE_jenny_recycling_money_is_160_l3136_313639


namespace NUMINAMATH_CALUDE_sum_inverse_max_min_S_l3136_313638

/-- Given real numbers x and y satisfying 4x^2 - 5xy + 4y^2 = 5,
    and S defined as x^2 + y^2, prove that the maximum and minimum
    values of S exist, and 1/S_max + 1/S_min = 8/5. -/
theorem sum_inverse_max_min_S :
  ∃ (S_max S_min : ℝ),
    (∀ x y : ℝ, 4 * x^2 - 5 * x * y + 4 * y^2 = 5 →
      let S := x^2 + y^2
      S ≤ S_max ∧ S_min ≤ S) ∧
    1 / S_max + 1 / S_min = 8 / 5 := by
  sorry

#check sum_inverse_max_min_S

end NUMINAMATH_CALUDE_sum_inverse_max_min_S_l3136_313638


namespace NUMINAMATH_CALUDE_function_properties_l3136_313691

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x + b

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 9

theorem function_properties (a b : ℝ) :
  f a b 0 = 2 →
  f' a 1 = 0 →
  (∃ (x : ℝ), f 3 2 x = f a b x) ∧
  (∀ (x : ℝ), x < -3 → (f' 3 x > 0)) ∧
  (∀ (x : ℝ), -3 < x ∧ x < 1 → (f' 3 x < 0)) ∧
  (∀ (x : ℝ), x > 1 → (f' 3 x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3136_313691


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3136_313607

/-- The foci of the hyperbola y²/16 - x²/9 = 1 are located at (0, ±5) -/
theorem hyperbola_foci : 
  ∀ (x y : ℝ), (y^2 / 16 - x^2 / 9 = 1) → 
  ∃ (c : ℝ), c = 5 ∧ ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3136_313607


namespace NUMINAMATH_CALUDE_cereal_cost_l3136_313604

theorem cereal_cost (total_spent groceries_cost milk_cost cereal_boxes banana_cost banana_count
                     apple_cost apple_count cookie_cost_multiplier cookie_boxes : ℚ) :
  groceries_cost = 25 →
  milk_cost = 3 →
  cereal_boxes = 2 →
  banana_cost = 0.25 →
  banana_count = 4 →
  apple_cost = 0.5 →
  apple_count = 4 →
  cookie_cost_multiplier = 2 →
  cookie_boxes = 2 →
  (groceries_cost - (milk_cost + banana_cost * banana_count + apple_cost * apple_count +
   cookie_cost_multiplier * milk_cost * cookie_boxes)) / cereal_boxes = 3.5 := by
sorry

end NUMINAMATH_CALUDE_cereal_cost_l3136_313604


namespace NUMINAMATH_CALUDE_range_of_a_l3136_313682

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3136_313682


namespace NUMINAMATH_CALUDE_regions_less_than_199_with_99_lines_l3136_313623

/-- The number of regions created by dividing a plane with lines -/
def num_regions (num_lines : ℕ) (all_parallel : Bool) (all_concurrent : Bool) : ℕ :=
  if all_parallel then
    num_lines + 1
  else if all_concurrent then
    2 * num_lines - 1
  else
    1 + num_lines + (num_lines.choose 2)

/-- Theorem stating the possible number of regions less than 199 when 99 lines divide a plane -/
theorem regions_less_than_199_with_99_lines :
  let possible_regions := {n : ℕ | n < 199 ∧ ∃ (parallel concurrent : Bool), 
    num_regions 99 parallel concurrent = n}
  possible_regions = {100, 198} := by
  sorry

end NUMINAMATH_CALUDE_regions_less_than_199_with_99_lines_l3136_313623


namespace NUMINAMATH_CALUDE_area_of_square_II_l3136_313628

/-- Given a square I with diagonal 3(a+b), where a and b are positive real numbers,
    the area of a square II that is three times the area of square I is equal to 27(a+b)^2/2. -/
theorem area_of_square_II (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let diagonal_I := 3 * (a + b)
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 3 * area_I
  area_II = 27 * (a + b)^2 / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_square_II_l3136_313628


namespace NUMINAMATH_CALUDE_number_of_choices_l3136_313609

-- Define the total number of subjects
def total_subjects : ℕ := 6

-- Define the number of science subjects
def science_subjects : ℕ := 3

-- Define the number of humanities subjects
def humanities_subjects : ℕ := 3

-- Define the number of subjects to be chosen
def subjects_to_choose : ℕ := 3

-- Define the minimum number of science subjects to be chosen
def min_science_subjects : ℕ := 2

-- Theorem statement
theorem number_of_choices :
  (Nat.choose science_subjects 2 * Nat.choose humanities_subjects 1) +
  (Nat.choose science_subjects 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_choices_l3136_313609


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3136_313660

def N : Matrix (Fin 4) (Fin 4) ℝ := !![3, -1, 8, 1; 4, 6, -2, 0; -9, -3, 5, 7; 1, 2, 0, -1]

def i : Fin 4 → ℝ := ![1, 0, 0, 0]
def j : Fin 4 → ℝ := ![0, 1, 0, 0]
def k : Fin 4 → ℝ := ![0, 0, 1, 0]
def l : Fin 4 → ℝ := ![0, 0, 0, 1]

theorem matrix_N_satisfies_conditions :
  N.mulVec i = ![3, 4, -9, 1] ∧
  N.mulVec j = ![-1, 6, -3, 2] ∧
  N.mulVec k = ![8, -2, 5, 0] ∧
  N.mulVec l = ![1, 0, 7, -1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3136_313660


namespace NUMINAMATH_CALUDE_problem_statement_l3136_313605

theorem problem_statement (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) :
  x^2 * y - x * y^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3136_313605


namespace NUMINAMATH_CALUDE_novelist_writing_speed_l3136_313641

/-- Calculates the effective writing speed given total words, total hours, and break hours -/
def effectiveWritingSpeed (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℕ :=
  totalWords / (totalHours - breakHours)

/-- Proves that the effective writing speed for the given conditions is 750 words per hour -/
theorem novelist_writing_speed :
  effectiveWritingSpeed 60000 100 20 = 750 := by
  sorry

end NUMINAMATH_CALUDE_novelist_writing_speed_l3136_313641


namespace NUMINAMATH_CALUDE_real_roots_condition_l3136_313620

theorem real_roots_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ 0 ≤ a ∧ a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_condition_l3136_313620


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l3136_313676

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  last_two_digits (sum_factorials 15) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l3136_313676


namespace NUMINAMATH_CALUDE_frustum_central_angle_l3136_313635

/-- Represents a frustum of a cone -/
structure Frustum where
  lateral_area : ℝ
  total_area : ℝ

/-- 
Given a frustum of a cone with lateral surface area 10π and total surface area 19π,
the central angle of the lateral surface when laid flat is 324°.
-/
theorem frustum_central_angle (f : Frustum) 
  (h1 : f.lateral_area = 10 * Real.pi)
  (h2 : f.total_area = 19 * Real.pi) : 
  ∃ (angle : ℝ), angle = 324 ∧ 
  (angle / 360) * Real.pi * ((6 * 360) / angle)^2 = f.lateral_area := by
  sorry


end NUMINAMATH_CALUDE_frustum_central_angle_l3136_313635


namespace NUMINAMATH_CALUDE_equation_solution_l3136_313619

theorem equation_solution (x : ℚ) (h : 2 * x + 1 = 8) : 4 * x + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3136_313619


namespace NUMINAMATH_CALUDE_seven_mile_taxi_cost_l3136_313649

/-- Calculates the total cost of a taxi ride given the fixed cost, cost per mile, and distance traveled. -/
def taxi_cost (fixed_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  fixed_cost + cost_per_mile * distance

/-- Theorem stating that a 7-mile taxi ride with $2.00 fixed cost and $0.30 per mile costs $4.10 -/
theorem seven_mile_taxi_cost :
  taxi_cost 2.00 0.30 7 = 4.10 := by
  sorry

end NUMINAMATH_CALUDE_seven_mile_taxi_cost_l3136_313649


namespace NUMINAMATH_CALUDE_equation_solution_l3136_313634

theorem equation_solution (x : ℝ) : 14 * x + 5 - 21 * x^2 = -2 → 6 * x^2 - 4 * x + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3136_313634


namespace NUMINAMATH_CALUDE_angle_A_is_obtuse_l3136_313640

/-- Triangle ABC with vertices A(2,1), B(-1,4), and C(5,3) -/
structure Triangle where
  A : ℝ × ℝ := (2, 1)
  B : ℝ × ℝ := (-1, 4)
  C : ℝ × ℝ := (5, 3)

/-- Calculate the squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Check if an angle is obtuse using the cosine law -/
def isObtuse (a b c : ℝ) : Prop :=
  a^2 > b^2 + c^2

theorem angle_A_is_obtuse (t : Triangle) : 
  isObtuse (squaredDistance t.B t.C) (squaredDistance t.A t.B) (squaredDistance t.A t.C) :=
sorry

end NUMINAMATH_CALUDE_angle_A_is_obtuse_l3136_313640


namespace NUMINAMATH_CALUDE_cos_equation_solution_l3136_313645

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos x - 3 * Real.cos (4 * x))^2 = 16 + Real.sin (3 * x)^2 ↔ 
  ∃ k : ℤ, x = π + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l3136_313645


namespace NUMINAMATH_CALUDE_trains_passing_time_l3136_313624

/-- Given two trains with specified characteristics, prove that they will completely pass each other in 11 seconds. -/
theorem trains_passing_time (tunnel_length : ℝ) (tunnel_time : ℝ) (bridge_length : ℝ) (bridge_time : ℝ)
  (freight_train_length : ℝ) (freight_train_speed : ℝ) :
  tunnel_length = 285 →
  tunnel_time = 24 →
  bridge_length = 245 →
  bridge_time = 22 →
  freight_train_length = 135 →
  freight_train_speed = 10 →
  ∃ (train_speed : ℝ) (train_length : ℝ),
    train_speed = (tunnel_length - bridge_length) / (tunnel_time - bridge_time) ∧
    train_length = train_speed * tunnel_time - tunnel_length ∧
    (train_length + freight_train_length) / (train_speed + freight_train_speed) = 11 :=
by sorry

end NUMINAMATH_CALUDE_trains_passing_time_l3136_313624


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l3136_313616

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x ^ 2 ≡ 0 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l3136_313616


namespace NUMINAMATH_CALUDE_f_at_two_l3136_313643

/-- The polynomial function f(x) = x^6 - 2x^5 + 3x^3 + 4x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := x^6 - 2*x^5 + 3*x^3 + 4*x^2 - 6*x + 5

/-- Theorem: The value of f(2) is 29 -/
theorem f_at_two : f 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_l3136_313643


namespace NUMINAMATH_CALUDE_equal_weight_implies_all_genuine_l3136_313659

/-- Represents a coin, which can be either genuine or counterfeit. -/
inductive Coin
| genuine
| counterfeit

/-- The total number of coins. -/
def total_coins : ℕ := 12

/-- The number of genuine coins. -/
def genuine_coins : ℕ := 9

/-- The number of counterfeit coins. -/
def counterfeit_coins : ℕ := 3

/-- A function that returns the weight of a coin. -/
def weight : Coin → ℝ
| Coin.genuine => 1
| Coin.counterfeit => 2  -- Counterfeit coins are heavier

/-- A type representing a selection of coins. -/
def CoinSelection := Fin 6 → Coin

/-- The property that all coins in a selection are genuine. -/
def all_genuine (selection : CoinSelection) : Prop :=
  ∀ i, selection i = Coin.genuine

/-- The property that the weights of two sets of coins are equal. -/
def weights_equal (selection : CoinSelection) : Prop :=
  (weight (selection 0) + weight (selection 1) + weight (selection 2)) =
  (weight (selection 3) + weight (selection 4) + weight (selection 5))

/-- The main theorem to be proved. -/
theorem equal_weight_implies_all_genuine :
  ∀ (selection : CoinSelection),
  weights_equal selection → all_genuine selection :=
by sorry

end NUMINAMATH_CALUDE_equal_weight_implies_all_genuine_l3136_313659


namespace NUMINAMATH_CALUDE_train_arrival_interval_l3136_313677

def minutes_between (h1 m1 h2 m2 : ℕ) : ℕ :=
  (h2 * 60 + m2) - (h1 * 60 + m1)

theorem train_arrival_interval (x : ℕ) : 
  x > 0 → 
  minutes_between 10 10 10 55 % x = 0 → 
  minutes_between 10 55 11 58 % x = 0 → 
  x = 9 :=
sorry

end NUMINAMATH_CALUDE_train_arrival_interval_l3136_313677


namespace NUMINAMATH_CALUDE_prism_volume_l3136_313670

/-- The volume of a right rectangular prism with face areas 15, 10, and 30 square inches is 30√5 cubic inches. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : c * a = 30) :
  a * b * c = 30 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3136_313670
