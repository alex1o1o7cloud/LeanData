import Mathlib

namespace q_satisfies_conditions_l2964_296486

/-- A cubic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (15/8) * x^3 + (5/4) * x^2 - (13/8) * x + 3

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q 0 = 3 ∧ q 1 = 5 ∧ q 2 = 13 ∧ q 3 = 41 := by
  sorry

#eval q 0
#eval q 1
#eval q 2
#eval q 3

end q_satisfies_conditions_l2964_296486


namespace imaginary_part_of_z_l2964_296472

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)) :
  z.im = 4/5 := by
  sorry

end imaginary_part_of_z_l2964_296472


namespace quadratic_equation_m_value_l2964_296494

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 5 * x + m^2 - 1 = 0) → 
  (m^2 - 1 = 0) → 
  m = -1 :=
by
  sorry

end quadratic_equation_m_value_l2964_296494


namespace square_sum_zero_implies_both_zero_l2964_296489

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l2964_296489


namespace factorization_xy_squared_minus_x_l2964_296479

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l2964_296479


namespace hoseok_flowers_left_l2964_296422

/-- Calculates the number of flowers Hoseok has left after giving some away. -/
def flowers_left (initial : ℕ) (to_minyoung : ℕ) (to_yoojeong : ℕ) : ℕ :=
  initial - (to_minyoung + to_yoojeong)

/-- Theorem stating that Hoseok has 7 flowers left after giving some away. -/
theorem hoseok_flowers_left :
  flowers_left 18 5 6 = 7 := by
  sorry

end hoseok_flowers_left_l2964_296422


namespace opposite_sign_quadratic_solution_l2964_296487

theorem opposite_sign_quadratic_solution :
  ∀ m n : ℝ,
  (|2*m + n| + Real.sqrt (3*n + 12) = 0) →
  (m = 2 ∧ n = -4) ∧
  (∀ x : ℝ, m*x^2 + 4*n*x - 2 = 0 ↔ x = 4 + Real.sqrt 17 ∨ x = 4 - Real.sqrt 17) :=
by sorry

end opposite_sign_quadratic_solution_l2964_296487


namespace bad_carrots_count_l2964_296409

/-- The number of bad carrots in Faye's garden -/
def bad_carrots (faye_carrots mother_carrots good_carrots : ℕ) : ℕ :=
  faye_carrots + mother_carrots - good_carrots

/-- Theorem: The number of bad carrots is 16 -/
theorem bad_carrots_count : bad_carrots 23 5 12 = 16 := by
  sorry

end bad_carrots_count_l2964_296409


namespace floor_ceiling_sum_l2964_296484

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end floor_ceiling_sum_l2964_296484


namespace binomial_expansion_sum_l2964_296432

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 160 := by
sorry

end binomial_expansion_sum_l2964_296432


namespace exchange_rate_theorem_l2964_296406

/-- Represents the number of boys in the group -/
def b : ℕ := sorry

/-- Represents the number of girls in the group -/
def g : ℕ := sorry

/-- Represents the exchange rate of yuan to alternative currency -/
def x : ℕ := sorry

/-- The total cost in yuan at the first disco -/
def first_disco_cost : ℕ := b * g

/-- The total cost in alternative currency at the second place -/
def second_place_cost : ℕ := (b + g) * (b + g - 1) + (b + g) + 1

/-- Theorem stating the exchange rate between yuan and alternative currency -/
theorem exchange_rate_theorem : 
  first_disco_cost * x = second_place_cost ∧ x = 5 :=
by sorry

end exchange_rate_theorem_l2964_296406


namespace triangle_perimeter_l2964_296471

theorem triangle_perimeter (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : |a + b - c| + |b + c - a| + |c + a - b| = 12) : 
  a + b + c = 12 := by
sorry

end triangle_perimeter_l2964_296471


namespace boat_downstream_speed_l2964_296475

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a boat given its speed in still water and upstream -/
def downstreamSpeed (boat : BoatSpeed) : ℝ :=
  2 * boat.stillWater - boat.upstream

theorem boat_downstream_speed 
  (boat : BoatSpeed) 
  (h1 : boat.stillWater = 8.5) 
  (h2 : boat.upstream = 4) : 
  downstreamSpeed boat = 13 := by
  sorry

end boat_downstream_speed_l2964_296475


namespace tony_age_proof_l2964_296404

/-- Represents Tony's age at the beginning of the work period -/
def initial_age : ℕ := 10

/-- Represents the number of days Tony worked -/
def work_days : ℕ := 80

/-- Represents Tony's daily work hours -/
def daily_hours : ℕ := 3

/-- Represents Tony's base hourly wage in cents -/
def base_wage : ℕ := 75

/-- Represents the age-based hourly wage increase in cents -/
def age_wage_increase : ℕ := 25

/-- Represents Tony's total earnings in cents -/
def total_earnings : ℕ := 84000

/-- Theorem stating that the given initial age satisfies the problem conditions -/
theorem tony_age_proof :
  ∃ (x : ℕ), x ≤ work_days ∧
  (daily_hours * (base_wage + age_wage_increase * initial_age) * x +
   daily_hours * (base_wage + age_wage_increase * (initial_age + 1)) * (work_days - x) =
   total_earnings) :=
sorry

end tony_age_proof_l2964_296404


namespace probability_circle_or_square_l2964_296485

-- Define the total number of figures
def total_figures : ℕ := 10

-- Define the number of circles
def num_circles : ℕ := 3

-- Define the number of squares
def num_squares : ℕ := 4

-- Define the number of triangles
def num_triangles : ℕ := 3

-- Theorem statement
theorem probability_circle_or_square :
  (num_circles + num_squares : ℚ) / total_figures = 7 / 10 := by
  sorry

end probability_circle_or_square_l2964_296485


namespace quadratic_roots_properties_l2964_296443

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : x₁^2 - x₁ - 1 = 0 ∧ x₂^2 - x₂ - 1 = 0) : 
  x₁ + x₂ = 1 ∧ x₁ * x₂ = -1 ∧ x₁^2 + x₂^2 = 3 := by
  sorry

end quadratic_roots_properties_l2964_296443


namespace second_company_base_rate_l2964_296470

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 6

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second telephone company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes used -/
def minutes_used : ℝ := 120

/-- The base rate of the second telephone company in dollars -/
def second_base_rate : ℝ := 12

theorem second_company_base_rate :
  united_base_rate + united_per_minute * minutes_used =
  second_base_rate + second_per_minute * minutes_used :=
by sorry

end second_company_base_rate_l2964_296470


namespace magic_ink_combinations_l2964_296444

/-- The number of valid combinations for a magic ink recipe. -/
def validCombinations (herbTypes : ℕ) (essenceTypes : ℕ) (incompatibleHerbs : ℕ) : ℕ :=
  herbTypes * essenceTypes - incompatibleHerbs

/-- Theorem stating that the number of valid combinations for the magic ink is 21. -/
theorem magic_ink_combinations :
  validCombinations 4 6 3 = 21 := by
  sorry

end magic_ink_combinations_l2964_296444


namespace divide_eight_by_repeating_third_l2964_296451

-- Define the repeating decimal 0.3333...
def repeating_decimal : ℚ := 1 / 3

-- Theorem statement
theorem divide_eight_by_repeating_third : 8 / repeating_decimal = 24 := by
  sorry

end divide_eight_by_repeating_third_l2964_296451


namespace price_increase_percentage_l2964_296429

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 420) :
  ((new_price - old_price) / old_price) * 100 = 40 := by
  sorry

end price_increase_percentage_l2964_296429


namespace fraction_equality_l2964_296420

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 4) 
  (h2 : r / t = 8 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 := by
  sorry

end fraction_equality_l2964_296420


namespace perfect_square_condition_l2964_296425

theorem perfect_square_condition (a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + 2*(a+4)*x + 25 = (x + k)^2) → (a = 1 ∨ a = -9) :=
by sorry

end perfect_square_condition_l2964_296425


namespace constant_for_max_n_l2964_296455

theorem constant_for_max_n (c : ℝ) : 
  (∀ n : ℤ, n ≤ 8 → c * n^2 ≤ 8100) ∧ 
  (c * 9^2 > 8100) ↔ 
  c = 126.5625 := by
sorry

end constant_for_max_n_l2964_296455


namespace range_of_g_l2964_296434

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * (x + 5)

-- State the theorem
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 1 ∧ g x = y) ↔ y ≠ 18 := by
  sorry

end range_of_g_l2964_296434


namespace biscuit_price_is_two_l2964_296418

/-- Represents the bakery order problem --/
def bakery_order (quiche_price croissant_price biscuit_price : ℚ) : Prop :=
  let quiche_count : ℕ := 2
  let croissant_count : ℕ := 6
  let biscuit_count : ℕ := 6
  let discount_rate : ℚ := 1 / 10
  let discounted_total : ℚ := 54

  let original_total : ℚ := quiche_count * quiche_price + 
                            croissant_count * croissant_price + 
                            biscuit_count * biscuit_price

  let discounted_amount : ℚ := original_total * discount_rate
  
  (original_total > 50) ∧ 
  (original_total - discounted_amount = discounted_total) ∧
  (quiche_price = 15) ∧
  (croissant_price = 3) ∧
  (biscuit_price = 2)

/-- Theorem stating that the biscuit price is $2.00 --/
theorem biscuit_price_is_two :
  ∃ (quiche_price croissant_price biscuit_price : ℚ),
    bakery_order quiche_price croissant_price biscuit_price ∧
    biscuit_price = 2 := by
  sorry

end biscuit_price_is_two_l2964_296418


namespace continuous_at_five_l2964_296410

def f (x : ℝ) : ℝ := 4 * x^2 - 2

theorem continuous_at_five :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by
  sorry

end continuous_at_five_l2964_296410


namespace jessicas_balloons_l2964_296421

/-- Given the number of blue balloons for Joan, Sally, and the total,
    prove that Jessica has 2 blue balloons. -/
theorem jessicas_balloons
  (joan_balloons : ℕ)
  (sally_balloons : ℕ)
  (total_balloons : ℕ)
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : total_balloons = 16)
  (h4 : ∃ (jessica_balloons : ℕ), joan_balloons + sally_balloons + jessica_balloons = total_balloons) :
  ∃ (jessica_balloons : ℕ), jessica_balloons = 2 ∧ joan_balloons + sally_balloons + jessica_balloons = total_balloons :=
by
  sorry

end jessicas_balloons_l2964_296421


namespace max_period_is_14_l2964_296466

/-- A function with symmetry properties and a period -/
structure SymmetricPeriodicFunction where
  f : ℝ → ℝ
  period : ℝ
  periodic : ∀ x, f (x + period) = f x
  sym_1 : ∀ x, f (1 + x) = f (1 - x)
  sym_8 : ∀ x, f (8 + x) = f (8 - x)

/-- The maximum period for a SymmetricPeriodicFunction is 14 -/
theorem max_period_is_14 (spf : SymmetricPeriodicFunction) : 
  spf.period ≤ 14 := by sorry

end max_period_is_14_l2964_296466


namespace quadratic_factorization_l2964_296460

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end quadratic_factorization_l2964_296460


namespace smallest_n_divisible_by_2016_n_193_divisible_by_2016_smallest_n_is_193_l2964_296447

theorem smallest_n_divisible_by_2016 :
  ∀ n : ℕ, n > 1 → (3 * n^3 + 2013) % 2016 = 0 → n ≥ 193 :=
by sorry

theorem n_193_divisible_by_2016 :
  (3 * 193^3 + 2013) % 2016 = 0 :=
by sorry

theorem smallest_n_is_193 :
  ∃! n : ℕ, n > 1 ∧ (3 * n^3 + 2013) % 2016 = 0 ∧
  ∀ m : ℕ, m > 1 → (3 * m^3 + 2013) % 2016 = 0 → m ≥ n :=
by sorry

end smallest_n_divisible_by_2016_n_193_divisible_by_2016_smallest_n_is_193_l2964_296447


namespace arithmetic_calculation_l2964_296488

theorem arithmetic_calculation : 4 * 6 * 9 - 18 / 3 + 2^3 = 218 := by
  sorry

end arithmetic_calculation_l2964_296488


namespace subtraction_of_like_terms_l2964_296426

theorem subtraction_of_like_terms (a : ℝ) : 4 * a - 3 * a = a := by sorry

end subtraction_of_like_terms_l2964_296426


namespace cubic_inequality_l2964_296435

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end cubic_inequality_l2964_296435


namespace ben_cards_l2964_296492

theorem ben_cards (B : ℕ) (tim_cards : ℕ) : 
  tim_cards = 20 → B + 3 = 2 * tim_cards → B = 37 := by sorry

end ben_cards_l2964_296492


namespace purely_imaginary_complex_number_l2964_296476

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end purely_imaginary_complex_number_l2964_296476


namespace chord_length_polar_circle_l2964_296412

/-- The length of the chord intercepted by the line tan θ = 1/2 on the circle ρ = 4sin θ is 16/5 -/
theorem chord_length_polar_circle (θ : Real) (ρ : Real) : 
  ρ = 4 * Real.sin θ → Real.tan θ = 1 / 2 → 
  2 * ρ * Real.sin θ = 16 / 5 := by sorry

end chord_length_polar_circle_l2964_296412


namespace f_decreasing_on_neg_reals_l2964_296445

-- Define the function
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem f_decreasing_on_neg_reals : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 0 → f x₁ > f x₂ := by
  sorry

end f_decreasing_on_neg_reals_l2964_296445


namespace horner_method_v2_l2964_296463

def f (x : ℝ) : ℝ := x^6 - 8*x^5 + 60*x^4 + 16*x^3 + 96*x^2 + 240*x + 64

def horner_v2 (a : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * a - 8
  v1 * a + 60

theorem horner_method_v2 :
  horner_v2 2 = 48 :=
by sorry

end horner_method_v2_l2964_296463


namespace paige_dresser_capacity_l2964_296438

/-- Represents the capacity of a dresser in pieces of clothing. -/
def dresser_capacity (pieces_per_drawer : ℕ) (num_drawers : ℕ) : ℕ :=
  pieces_per_drawer * num_drawers

/-- Theorem stating that a dresser with 8 drawers, each holding 5 pieces, has a total capacity of 40 pieces. -/
theorem paige_dresser_capacity :
  dresser_capacity 5 8 = 40 := by
  sorry

end paige_dresser_capacity_l2964_296438


namespace wrapping_paper_needed_l2964_296403

theorem wrapping_paper_needed (present1 : ℝ) (present2 : ℝ) (present3 : ℝ) :
  present1 = 2 →
  present2 = 3 / 4 * present1 →
  present3 = present1 + present2 →
  present1 + present2 + present3 = 7 := by
  sorry

end wrapping_paper_needed_l2964_296403


namespace jellybean_probability_l2964_296493

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 2
def yellow_jellybeans : ℕ := 5
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 3 * Nat.choose (blue_jellybeans + yellow_jellybeans) 1) /
  Nat.choose total_jellybeans picked_jellybeans = 14 / 99 :=
by sorry

end jellybean_probability_l2964_296493


namespace number_problem_l2964_296400

theorem number_problem (x : ℚ) : (x / 4) * 12 = 9 → x = 3 := by
  sorry

end number_problem_l2964_296400


namespace advertising_sales_prediction_l2964_296446

-- Define the relationship between advertising expenditure and sales revenue
def advertising_sales_relation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 4

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ :=
  6.5 * x + 17.5

-- Theorem statement
theorem advertising_sales_prediction :
  ∀ x y : ℝ, advertising_sales_relation x y →
  (linear_regression 10 = 82.5) ∧
  (∀ x : ℝ, y = linear_regression x) :=
by sorry

end advertising_sales_prediction_l2964_296446


namespace rachel_homework_l2964_296439

theorem rachel_homework (total_pages reading_pages biology_pages : ℕ) 
  (h1 : total_pages = 15)
  (h2 : reading_pages = 3)
  (h3 : biology_pages = 10) : 
  total_pages - reading_pages - biology_pages = 2 := by
sorry

end rachel_homework_l2964_296439


namespace illuminated_cube_surface_area_l2964_296417

/-- The illuminated area of a cube's surface when a cylindrical beam of light is directed along its main diagonal --/
theorem illuminated_cube_surface_area
  (a : ℝ) -- Edge length of the cube
  (ρ : ℝ) -- Radius of the cylindrical beam
  (h1 : a = Real.sqrt (2 + Real.sqrt 2)) -- Given edge length
  (h2 : ρ = Real.sqrt 2) -- Given beam radius
  (h3 : ρ > 0) -- Positive radius
  (h4 : a > 0) -- Positive edge length
  : Real.sqrt 3 * π / 2 + 3 * Real.sqrt 6 = 
    (3 : ℝ) * π * ρ^2 / 2 := by
  sorry

end illuminated_cube_surface_area_l2964_296417


namespace dividend_calculation_l2964_296427

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 8) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 141 := by
  sorry

end dividend_calculation_l2964_296427


namespace a4_square_area_l2964_296452

/-- Represents the properties of an A4 sheet of paper -/
structure A4Sheet where
  length : Real
  width : Real
  ratio_preserved : length / width = length / (2 * width)

theorem a4_square_area (sheet : A4Sheet) (h1 : sheet.length = 29.7) :
  ∃ (area : Real), abs (area - sheet.width ^ 2) < 0.05 ∧ abs (area - 441.0) < 0.05 := by
  sorry

end a4_square_area_l2964_296452


namespace complex_magnitude_product_l2964_296464

theorem complex_magnitude_product : |(7 + 6*I)*(-5 + 3*I)| = Real.sqrt 2890 := by
  sorry

end complex_magnitude_product_l2964_296464


namespace fish_sales_hours_l2964_296490

/-- The number of hours fish are sold for, given peak and low season sales rates,
    price per pack, and daily revenue difference between seasons. -/
theorem fish_sales_hours 
  (peak_rate : ℕ) 
  (low_rate : ℕ) 
  (price_per_pack : ℕ) 
  (daily_revenue_diff : ℕ) 
  (h_peak_rate : peak_rate = 6)
  (h_low_rate : low_rate = 4)
  (h_price : price_per_pack = 60)
  (h_revenue_diff : daily_revenue_diff = 1800) :
  (peak_rate - low_rate) * price_per_pack * h = daily_revenue_diff → h = 15 :=
by sorry

end fish_sales_hours_l2964_296490


namespace min_c_value_l2964_296442

theorem min_c_value (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- consecutive integers
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧  -- consecutive integers
  ∃ m : ℕ, b + c + d = m^2 ∧  -- b + c + d is a perfect square
  ∃ n : ℕ, a + b + c + d + e = n^3 ∧  -- a + b + c + d + e is a perfect cube
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' < b' ∧ b' < c' ∧ c' < d' ∧ d' < e' ∧
    b' = a' + 1 ∧ c' = b' + 1 ∧ d' = c' + 1 ∧ e' = d' + 1 ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) →
  c' ≥ c →
  c = 675 :=
sorry

end min_c_value_l2964_296442


namespace average_of_seventeen_numbers_l2964_296491

-- Define the problem parameters
def total_count : ℕ := 17
def first_nine_avg : ℚ := 56
def last_nine_avg : ℚ := 63
def ninth_number : ℚ := 68

-- Theorem statement
theorem average_of_seventeen_numbers :
  let first_nine_sum := 9 * first_nine_avg
  let last_nine_sum := 9 * last_nine_avg
  let total_sum := first_nine_sum + last_nine_sum - ninth_number
  total_sum / total_count = 59 := by
sorry

end average_of_seventeen_numbers_l2964_296491


namespace arithmetic_sequence_property_l2964_296437

/-- An increasing arithmetic sequence of integers -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a) 
  (h_prod : a 4 * a 5 = 12) : 
  a 2 * a 7 = 6 := by
  sorry

end arithmetic_sequence_property_l2964_296437


namespace unique_solution_for_absolute_value_equation_l2964_296478

theorem unique_solution_for_absolute_value_equation :
  ∃! x : ℤ, |x - 8 * (3 - 12)| - |5 - 11| = 73 :=
by
  sorry

end unique_solution_for_absolute_value_equation_l2964_296478


namespace unicorn_count_correct_l2964_296480

/-- The number of unicorns in the Enchanted Forest --/
def num_unicorns : ℕ := 6

/-- The number of flowers that bloom with each unicorn step --/
def flowers_per_step : ℕ := 4

/-- The length of the journey in kilometers --/
def journey_length : ℕ := 9

/-- The length of each unicorn step in meters --/
def step_length : ℕ := 3

/-- The total number of flowers that bloom during the journey --/
def total_flowers : ℕ := 72000

/-- Theorem stating that the number of unicorns is correct given the conditions --/
theorem unicorn_count_correct : 
  num_unicorns * flowers_per_step * (journey_length * 1000 / step_length) = total_flowers :=
by sorry

end unicorn_count_correct_l2964_296480


namespace hyperbola_asymptotes_l2964_296424

/-- The equations of the asymptotes of the hyperbola x²/16 - y²/9 = 1 -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := λ x y => x^2/16 - y^2/9 = 1
  ∃ (f g : ℝ → ℝ), (∀ x, f x = (3/4) * x) ∧ (∀ x, g x = -(3/4) * x) ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, h x y → (|x| > M → |y - f x| < ε ∨ |y - g x| < ε)) :=
by sorry

end hyperbola_asymptotes_l2964_296424


namespace prob_two_girls_prob_two_girls_five_l2964_296461

/-- The probability of selecting two girls from a club with equal numbers of boys and girls -/
theorem prob_two_girls (n : ℕ) (h : n > 0) : 
  (Nat.choose n 2) / (Nat.choose (2*n) 2) = 2 / 9 :=
sorry

/-- The specific case for a club with 5 girls and 5 boys -/
theorem prob_two_girls_five : 
  (Nat.choose 5 2) / (Nat.choose 10 2) = 2 / 9 :=
sorry

end prob_two_girls_prob_two_girls_five_l2964_296461


namespace smallest_base_perfect_square_l2964_296407

theorem smallest_base_perfect_square : ∃ (b : ℕ), 
  b > 3 ∧ 
  (∃ (n : ℕ), n^2 = 2*b + 3 ∧ n^2 < 25) ∧
  (∀ (k : ℕ), k > 3 ∧ k < b → ¬∃ (m : ℕ), m^2 = 2*k + 3 ∧ m^2 < 25) ∧
  b = 11 := by
sorry

end smallest_base_perfect_square_l2964_296407


namespace arrangements_count_l2964_296458

/-- Number of red flags -/
def red_flags : ℕ := 8

/-- Number of white flags -/
def white_flags : ℕ := 8

/-- Number of black flags -/
def black_flags : ℕ := 1

/-- Total number of flags -/
def total_flags : ℕ := red_flags + white_flags + black_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
def count_arrangements (r w b p : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of distinguishable arrangements is 315 -/
theorem arrangements_count :
  count_arrangements red_flags white_flags black_flags flagpoles = 315 :=
sorry

end arrangements_count_l2964_296458


namespace sport_participation_theorem_l2964_296433

/-- Represents the number of students who play various sports in a class -/
structure SportParticipation where
  total_students : ℕ
  basketball : ℕ
  cricket : ℕ
  baseball : ℕ
  basketball_cricket : ℕ
  cricket_baseball : ℕ
  basketball_baseball : ℕ
  all_three : ℕ

/-- Calculates the number of students who play at least one sport -/
def students_playing_at_least_one_sport (sp : SportParticipation) : ℕ :=
  sp.basketball + sp.cricket + sp.baseball - sp.basketball_cricket - sp.cricket_baseball - sp.basketball_baseball + sp.all_three

/-- Calculates the number of students who don't play any sport -/
def students_not_playing_any_sport (sp : SportParticipation) : ℕ :=
  sp.total_students - students_playing_at_least_one_sport sp

/-- Theorem stating the correct number of students playing at least one sport and not playing any sport -/
theorem sport_participation_theorem (sp : SportParticipation) 
  (h1 : sp.total_students = 40)
  (h2 : sp.basketball = 15)
  (h3 : sp.cricket = 20)
  (h4 : sp.baseball = 12)
  (h5 : sp.basketball_cricket = 5)
  (h6 : sp.cricket_baseball = 7)
  (h7 : sp.basketball_baseball = 3)
  (h8 : sp.all_three = 2) :
  students_playing_at_least_one_sport sp = 32 ∧ students_not_playing_any_sport sp = 8 := by
  sorry

end sport_participation_theorem_l2964_296433


namespace intersection_point_of_lines_l2964_296430

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 8) ∧ (6 * x + 3 * y = 21) ↔ x = 22/9 ∧ y = 19/9 := by
  sorry

end intersection_point_of_lines_l2964_296430


namespace radio_station_survey_l2964_296405

theorem radio_station_survey (total_listeners total_non_listeners female_listeners male_non_listeners : ℕ)
  (h1 : total_listeners = 160)
  (h2 : total_non_listeners = 180)
  (h3 : female_listeners = 72)
  (h4 : male_non_listeners = 88) :
  total_listeners - female_listeners = 92 :=
by
  sorry

#check radio_station_survey

end radio_station_survey_l2964_296405


namespace class_size_l2964_296401

/-- The number of students in Ms. Perez's class -/
def S : ℕ := sorry

/-- The number of students who collected 12 cans each -/
def students_12_cans : ℕ := S / 2

/-- The number of students who collected 4 cans each -/
def students_4_cans : ℕ := 13

/-- The number of students who didn't collect any cans -/
def students_0_cans : ℕ := 2

/-- The total number of cans collected -/
def total_cans : ℕ := 232

theorem class_size :
  S = 30 ∧
  S = students_12_cans + students_4_cans + students_0_cans ∧
  total_cans = students_12_cans * 12 + students_4_cans * 4 + students_0_cans * 0 :=
sorry

end class_size_l2964_296401


namespace ordered_pairs_1764_l2964_296481

/-- The number of ordered pairs of positive integers (x,y) that satisfy xy = n,
    where n has the prime factorization p₁^a₁ * p₂^a₂ * ... * pₖ^aₖ -/
def count_ordered_pairs (n : ℕ) (primes : List ℕ) (exponents : List ℕ) : ℕ :=
  sorry

theorem ordered_pairs_1764 :
  count_ordered_pairs 1764 [2, 3, 7] [2, 2, 2] = 27 :=
sorry

end ordered_pairs_1764_l2964_296481


namespace smallest_difference_l2964_296469

/-- Represents the first sequence in the table -/
def first_sequence (n : ℕ) : ℤ := 2 * n - 1

/-- Represents the second sequence in the table -/
def second_sequence (n : ℕ) : ℤ := 5055 - 5 * n

/-- The difference between the two sequences at position n -/
def difference (n : ℕ) : ℤ := (second_sequence n) - (first_sequence n)

/-- The number of terms in each sequence -/
def sequence_length : ℕ := 1010

theorem smallest_difference :
  ∃ (k : ℕ), k ≤ sequence_length ∧ difference k = 2 ∧
  ∀ (n : ℕ), n ≤ sequence_length → difference n ≥ 2 :=
sorry

end smallest_difference_l2964_296469


namespace amp_composition_l2964_296482

def amp (x : ℤ) : ℤ := 9 - x
def amp_bar (x : ℤ) : ℤ := x - 9

theorem amp_composition : amp (amp_bar 15) = 15 := by
  sorry

end amp_composition_l2964_296482


namespace smallest_number_l2964_296468

/-- Convert a number from base 6 to decimal -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number from base 4 to decimal -/
def base4ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number from base 2 to decimal -/
def base2ToDecimal (n : ℕ) : ℕ := sorry

theorem smallest_number :
  let n1 := base6ToDecimal 210
  let n2 := base4ToDecimal 1000
  let n3 := base2ToDecimal 111111
  n3 < n1 ∧ n3 < n2 := by sorry

end smallest_number_l2964_296468


namespace cone_sphere_intersection_l2964_296473

noncomputable def cone_angle (r : ℝ) (h : ℝ) : ℝ :=
  let α := Real.arcsin ((Real.sqrt 5 - 1) / 2)
  2 * α

theorem cone_sphere_intersection (r : ℝ) (h : ℝ) (hr : r > 0) (hh : h > 0) :
  let α := cone_angle r h / 2
  let sphere_radius := h / 2
  let sphere_cap_area := 4 * Real.pi * sphere_radius^2 * Real.sin α^2
  let cone_cap_area := Real.pi * (2 * sphere_radius * Real.cos α * Real.sin α) * (2 * sphere_radius * Real.cos α)
  sphere_cap_area = cone_cap_area →
  cone_angle r h = 2 * Real.arccos (Real.sqrt 5 - 2) :=
by sorry

end cone_sphere_intersection_l2964_296473


namespace unequal_grandchildren_probability_l2964_296411

def num_grandchildren : ℕ := 12

theorem unequal_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_split := Nat.choose num_grandchildren (num_grandchildren / 2)
  (total_outcomes - equal_split) / total_outcomes = 793 / 1024 := by
  sorry

end unequal_grandchildren_probability_l2964_296411


namespace song_book_cost_l2964_296449

/-- The cost of the song book given the costs of other items and the total spent --/
theorem song_book_cost (trumpet_cost music_tool_cost total_spent : ℚ) : 
  trumpet_cost = 149.16 →
  music_tool_cost = 9.98 →
  total_spent = 163.28 →
  total_spent - (trumpet_cost + music_tool_cost) = 4.14 := by
  sorry

end song_book_cost_l2964_296449


namespace find_number_l2964_296402

theorem find_number : ∃ x : ℚ, x * 9999 = 824777405 ∧ x = 82482.5 := by
  sorry

end find_number_l2964_296402


namespace h_is_correct_l2964_296465

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := -9*x^3 - x^2 - 4*x + 3

-- State the theorem
theorem h_is_correct : 
  ∀ x : ℝ, 9*x^3 + 6*x^2 - 3*x + 1 + h x = 5*x^2 - 7*x + 4 := by
  sorry

end h_is_correct_l2964_296465


namespace kitten_weight_l2964_296431

theorem kitten_weight (k r p : ℝ) 
  (total_weight : k + r + p = 38)
  (kitten_rabbit_weight : k + r = 3 * p)
  (kitten_parrot_weight : k + p = r) :
  k = 9.5 := by
sorry

end kitten_weight_l2964_296431


namespace system_solvability_l2964_296415

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x * Real.cos a + y * Real.sin a - 2 ≤ 0 ∧
  x^2 + y^2 + 6*x - 2*y - b^2 + 4*b + 6 = 0

-- Define the solution set for b
def solution_set (b : ℝ) : Prop :=
  b ≤ 4 - Real.sqrt 10 ∨ b ≥ Real.sqrt 10

-- Theorem statement
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system x y a b) ↔ solution_set b := by
  sorry

end system_solvability_l2964_296415


namespace subtraction_decimal_proof_l2964_296467

theorem subtraction_decimal_proof :
  (12.358 : ℝ) - (7.2943 : ℝ) = 5.0637 := by
  sorry

end subtraction_decimal_proof_l2964_296467


namespace identity_is_unique_solution_l2964_296498

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The main theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → ∀ x : ℝ, f x = x :=
by sorry

end identity_is_unique_solution_l2964_296498


namespace concyclic_intersecting_lines_ratio_l2964_296459

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the condition that A, B, C, D are concyclic
def concyclic (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the condition that lines (AB) and (CD) intersect at E
def intersect_at (A B C D E : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem concyclic_intersecting_lines_ratio 
  (h1 : concyclic A B C D) 
  (h2 : intersect_at A B C D E) :
  (distance A C / distance B C) * (distance A D / distance B D) = 
  distance A E / distance B E := by sorry

end concyclic_intersecting_lines_ratio_l2964_296459


namespace F_negative_sufficient_not_necessary_l2964_296419

/-- Represents a general equation of the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure GeneralEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if a GeneralEquation represents a circle -/
def is_circle (eq : GeneralEquation) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ 
    eq.D = -2 * h ∧ 
    eq.E = -2 * k ∧ 
    eq.F = h^2 + k^2 - r^2

/-- Theorem stating that F < 0 is a sufficient but not necessary condition for a circle -/
theorem F_negative_sufficient_not_necessary (eq : GeneralEquation) :
  (eq.F < 0 → is_circle eq) ∧ ¬(is_circle eq → eq.F < 0) :=
sorry

end F_negative_sufficient_not_necessary_l2964_296419


namespace student_count_last_year_l2964_296499

theorem student_count_last_year 
  (increase_rate : ℝ) 
  (current_count : ℕ) 
  (h1 : increase_rate = 0.2) 
  (h2 : current_count = 960) : 
  ℕ :=
  by
    -- Proof goes here
    sorry

#check student_count_last_year

end student_count_last_year_l2964_296499


namespace capri_sun_cost_per_pouch_l2964_296495

/-- Calculates the cost per pouch in cents -/
def cost_per_pouch (num_boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) : ℕ :=
  (total_cost_dollars * 100) / (num_boxes * pouches_per_box)

/-- Theorem: The cost per pouch is 20 cents -/
theorem capri_sun_cost_per_pouch :
  cost_per_pouch 10 6 12 = 20 := by
  sorry

end capri_sun_cost_per_pouch_l2964_296495


namespace simplify_fraction_l2964_296414

theorem simplify_fraction : (54 : ℚ) / 972 = 1 / 18 := by
  sorry

end simplify_fraction_l2964_296414


namespace complex_modulus_example_l2964_296436

theorem complex_modulus_example : Complex.abs (7/4 + 3*I) = Real.sqrt 193 / 4 := by
  sorry

end complex_modulus_example_l2964_296436


namespace table_capacity_l2964_296497

theorem table_capacity (invited : ℕ) (no_show : ℕ) (tables : ℕ) : 
  invited = 68 → no_show = 50 → tables = 6 → 
  (invited - no_show) / tables = 3 := by
  sorry

end table_capacity_l2964_296497


namespace hen_price_calculation_l2964_296477

/-- Proves that given 5 goats and 10 hens with a total cost of 2500,
    and an average price of 400 per goat, the average price of a hen is 50. -/
theorem hen_price_calculation (num_goats num_hens total_cost goat_price : ℕ)
    (h1 : num_goats = 5)
    (h2 : num_hens = 10)
    (h3 : total_cost = 2500)
    (h4 : goat_price = 400) :
    (total_cost - num_goats * goat_price) / num_hens = 50 := by
  sorry

end hen_price_calculation_l2964_296477


namespace limit_f_at_infinity_l2964_296408

noncomputable def f (x : ℝ) := (x - Real.sin x) / (x + Real.sin x)

theorem limit_f_at_infinity :
  ∀ ε > 0, ∃ N : ℝ, ∀ x ≥ N, |f x - 1| < ε :=
by
  sorry

/- Assumptions:
   1. x is a real number (implied by the use of ℝ)
   2. sin x is bounded between -1 and 1 (this is a property of sine in Mathlib)
-/

end limit_f_at_infinity_l2964_296408


namespace xyz_negative_l2964_296416

theorem xyz_negative (a b c x y z : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 := by
sorry

end xyz_negative_l2964_296416


namespace ones_digit_of_35_power_ones_digit_of_35_large_power_l2964_296483

theorem ones_digit_of_35_power (n : ℕ) : n > 0 → (35^n) % 10 = 5 := by sorry

theorem ones_digit_of_35_large_power : (35^(35*(17^17))) % 10 = 5 := by sorry

end ones_digit_of_35_power_ones_digit_of_35_large_power_l2964_296483


namespace binomial_coefficient_equality_l2964_296450

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 15 (2*x + 1) = Nat.choose 15 (x + 2)) ↔ (x = 1 ∨ x = 4) := by
  sorry

end binomial_coefficient_equality_l2964_296450


namespace double_burger_cost_l2964_296453

/-- Proves that the cost of a double burger is $1.50 given the specified conditions -/
theorem double_burger_cost (total_spent : ℚ) (total_hamburgers : ℕ) (double_burgers : ℕ) (single_burger_cost : ℚ) :
  total_spent = 70.5 ∧
  total_hamburgers = 50 ∧
  double_burgers = 41 ∧
  single_burger_cost = 1 →
  (total_spent - (total_hamburgers - double_burgers : ℚ) * single_burger_cost) / double_burgers = 1.5 := by
  sorry

end double_burger_cost_l2964_296453


namespace profit_percentage_is_25_l2964_296456

/-- 
Given that the cost price of 30 articles is equal to the selling price of 24 articles,
prove that the profit percentage is 25%.
-/
theorem profit_percentage_is_25 (C S : ℝ) (h : 30 * C = 24 * S) : 
  (S - C) / C * 100 = 25 := by
  sorry

end profit_percentage_is_25_l2964_296456


namespace hyperbola_equation_l2964_296496

/-- Represents a hyperbola in 2D space -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- Properties of a specific hyperbola -/
def hyperbola_properties (h : Hyperbola) : Prop :=
  ∃ (a b : ℝ),
    -- The center is at the origin
    h.equation 0 0 ∧
    -- The right focus is at (3,0)
    (∃ (x y : ℝ), h.equation x y ∧ x = 3 ∧ y = 0) ∧
    -- The eccentricity is 3/2
    (3 / a = 3 / 2) ∧
    -- The equation of the hyperbola
    (∀ (x y : ℝ), h.equation x y ↔ x^2 / a^2 - y^2 / b^2 = 1)

/-- Theorem: The hyperbola with given properties has the equation x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (h : Hyperbola) (hp : hyperbola_properties h) :
  ∀ (x y : ℝ), h.equation x y ↔ x^2 / 4 - y^2 / 5 = 1 := by
  sorry

end hyperbola_equation_l2964_296496


namespace add_average_score_theorem_singing_competition_scores_l2964_296462

/-- Represents a set of scores with their statistical properties -/
structure ScoreSet where
  count : ℕ
  average : ℝ
  variance : ℝ

/-- Represents the result after adding a new score -/
structure NewScoreSet where
  new_average : ℝ
  new_variance : ℝ

/-- 
Given a set of scores and a new score, calculates the new average and variance
-/
def add_score (scores : ScoreSet) (new_score : ℝ) : NewScoreSet :=
  sorry

/-- 
Theorem: Adding a score equal to the original average keeps the average the same
and reduces the variance
-/
theorem add_average_score_theorem (scores : ScoreSet) :
  let new_set := add_score scores scores.average
  new_set.new_average = scores.average ∧ new_set.new_variance < scores.variance :=
  sorry

/-- 
Application of the theorem to the specific problem
-/
theorem singing_competition_scores :
  let original_scores : ScoreSet := ⟨8, 5, 3⟩
  let new_set := add_score original_scores 5
  new_set.new_average = 5 ∧ new_set.new_variance < 3 :=
  sorry

end add_average_score_theorem_singing_competition_scores_l2964_296462


namespace eighth_term_is_eight_l2964_296428

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first six terms is 21
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  -- The seventh term is 7
  seventh_term : a + 6*d = 7

/-- The eighth term of the arithmetic sequence is 8 -/
theorem eighth_term_is_eight (seq : ArithmeticSequence) : seq.a + 7*seq.d = 8 := by
  sorry

end eighth_term_is_eight_l2964_296428


namespace fraction_problem_l2964_296423

theorem fraction_problem (x y : ℚ) 
  (h1 : y / (x - 1) = 1 / 3)
  (h2 : (y + 4) / x = 1 / 2) :
  y / x = 7 / 22 :=
by sorry

end fraction_problem_l2964_296423


namespace lines_perpendicular_to_plane_are_parallel_l2964_296474

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (l₁ l₂ : Line) (α : Plane)
  (h₁ : l₁ ≠ l₂)  -- l₁ and l₂ are non-coincident
  (h₂ : perpendicular l₁ α)
  (h₃ : perpendicular l₂ α) :
  parallel l₁ l₂ :=
sorry

end lines_perpendicular_to_plane_are_parallel_l2964_296474


namespace shoe_selection_probability_l2964_296448

/-- The number of pairs of shoes in the cabinet -/
def num_pairs : ℕ := 3

/-- The total number of shoes in the cabinet -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes that do not form a pair -/
def prob_not_pair : ℚ := 4/5

theorem shoe_selection_probability :
  (Nat.choose total_shoes selected_shoes - num_pairs) / Nat.choose total_shoes selected_shoes = prob_not_pair :=
sorry

end shoe_selection_probability_l2964_296448


namespace solution_ratio_l2964_296457

theorem solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) : c / d = -1 := by
  sorry

end solution_ratio_l2964_296457


namespace round_trip_time_l2964_296454

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water,
    the stream speed, and the total distance traveled. -/
theorem round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_distance : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : total_distance = 420) : 
  (total_distance / (boat_speed + stream_speed) + 
   total_distance / (boat_speed - stream_speed)) = 120 := by
  sorry

#check round_trip_time

end round_trip_time_l2964_296454


namespace average_weight_problem_l2964_296440

/-- Given three weights a, b, c, prove that if their average is 45,
    the average of a and b is 40, and b is 31, then the average of b and c is 43. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 31 →
  (b + c) / 2 = 43 := by
sorry

end average_weight_problem_l2964_296440


namespace min_PQ_length_l2964_296441

/-- Circle C with center (3,4) and radius 2 -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

/-- Point P is outside the circle -/
def P_outside_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 > 4

/-- Length of PQ equals distance from P to origin -/
def PQ_equals_PO (x y : ℝ) : Prop := ∃ (qx qy : ℝ), 
  circle_C qx qy ∧ (x - qx)^2 + (y - qy)^2 = x^2 + y^2

/-- Theorem: Minimum value of |PQ| is 17/2 -/
theorem min_PQ_length (x y : ℝ) : 
  circle_C x y → P_outside_circle x y → PQ_equals_PO x y → 
  ∃ (qx qy : ℝ), circle_C qx qy ∧ (x - qx)^2 + (y - qy)^2 ≥ (17/2)^2 :=
sorry

end min_PQ_length_l2964_296441


namespace rate_percent_calculation_l2964_296413

/-- Given that the simple interest on Rs. 25,000 amounts to Rs. 5,500 in 7 years,
    prove that the rate percent is equal to (5500 * 100) / (25000 * 7) -/
theorem rate_percent_calculation (principal : ℝ) (interest : ℝ) (time : ℝ) 
    (h1 : principal = 25000)
    (h2 : interest = 5500)
    (h3 : time = 7)
    (h4 : interest = principal * (rate_percent / 100) * time) :
  rate_percent = (interest * 100) / (principal * time) := by
  sorry

end rate_percent_calculation_l2964_296413
