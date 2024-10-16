import Mathlib

namespace NUMINAMATH_CALUDE_least_common_denominator_of_fractions_l887_88771

theorem least_common_denominator_of_fractions : Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7)))) = 420 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_of_fractions_l887_88771


namespace NUMINAMATH_CALUDE_harolds_marbles_l887_88757

theorem harolds_marbles (kept : ℕ) (friends : ℕ) (each_friend : ℕ) (initial : ℕ) : 
  kept = 20 → 
  friends = 5 → 
  each_friend = 16 → 
  initial = kept + friends * each_friend → 
  initial = 100 := by
sorry

end NUMINAMATH_CALUDE_harolds_marbles_l887_88757


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l887_88724

theorem complex_magnitude_product : Complex.abs (3 - 5 * Complex.I) * Complex.abs (3 + 5 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l887_88724


namespace NUMINAMATH_CALUDE_subtract_from_zero_l887_88722

theorem subtract_from_zero (x : ℚ) : 0 - x = -x := by sorry

end NUMINAMATH_CALUDE_subtract_from_zero_l887_88722


namespace NUMINAMATH_CALUDE_revenue_condition_l887_88707

def initial_price : ℝ := 50
def initial_sales : ℝ := 300
def revenue_threshold : ℝ := 15950

def monthly_revenue (x : ℝ) : ℝ := (initial_price - x) * (initial_sales + 10 * x)

theorem revenue_condition (x : ℝ) :
  monthly_revenue x ≥ revenue_threshold ↔ (x = 9 ∨ x = 11) :=
sorry

end NUMINAMATH_CALUDE_revenue_condition_l887_88707


namespace NUMINAMATH_CALUDE_hot_sauce_serving_size_l887_88769

/-- Calculates the number of ounces per serving of hot sauce -/
theorem hot_sauce_serving_size (servings_per_day : ℕ) (quart_size : ℕ) (container_reduction : ℕ) (days_lasting : ℕ) :
  servings_per_day = 3 →
  quart_size = 32 →
  container_reduction = 2 →
  days_lasting = 20 →
  (quart_size - container_reduction : ℚ) / (servings_per_day * days_lasting) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hot_sauce_serving_size_l887_88769


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l887_88729

theorem triangle_angle_measure (a b c : ℝ) : 
  a + b + c = 180 →  -- sum of angles in a triangle is 180°
  a = 40 →           -- one angle is 40°
  b = 2 * c →        -- one angle is twice the other
  c = 140 / 3 :=     -- prove that the third angle is 140/3°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l887_88729


namespace NUMINAMATH_CALUDE_no_prime_multiples_of_ten_in_range_l887_88717

theorem no_prime_multiples_of_ten_in_range : 
  ¬ ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 10000 ∧ 10 ∣ n ∧ Nat.Prime n ∧ n > 10 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_multiples_of_ten_in_range_l887_88717


namespace NUMINAMATH_CALUDE_reciprocal_of_neg_tan_60_l887_88777

theorem reciprocal_of_neg_tan_60 :
  (-(Real.tan (60 * π / 180)))⁻¹ = -((3 : ℝ).sqrt / 3) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_neg_tan_60_l887_88777


namespace NUMINAMATH_CALUDE_total_pencils_count_l887_88725

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 2

/-- The number of children -/
def number_of_children : ℕ := 8

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_child * number_of_children

theorem total_pencils_count : total_pencils = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l887_88725


namespace NUMINAMATH_CALUDE_solution_x_l887_88708

theorem solution_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_l887_88708


namespace NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l887_88773

/-- A line in a coordinate plane. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The product of a line's slope and y-intercept. -/
def slopeInterceptProduct (l : Line) : ℝ := l.slope * l.yIntercept

/-- Theorem: For a line with y-intercept -2 and slope 3, the product of its slope and y-intercept is -6. -/
theorem slope_intercept_product_specific_line :
  ∃ l : Line, l.yIntercept = -2 ∧ l.slope = 3 ∧ slopeInterceptProduct l = -6 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_product_specific_line_l887_88773


namespace NUMINAMATH_CALUDE_pencil_ratio_l887_88763

theorem pencil_ratio (jeanine_initial : ℕ) (clare : ℕ) : 
  jeanine_initial = 18 →
  clare = jeanine_initial * 2 / 3 - 3 →
  clare.gcd jeanine_initial = clare →
  clare / (clare.gcd jeanine_initial) = 1 ∧ 
  jeanine_initial / (clare.gcd jeanine_initial) = 2 := by
sorry

end NUMINAMATH_CALUDE_pencil_ratio_l887_88763


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l887_88776

theorem blue_paint_calculation (total_paint white_paint : ℕ) 
  (h1 : total_paint = 6689)
  (h2 : white_paint = 660) :
  total_paint - white_paint = 6029 :=
by sorry

end NUMINAMATH_CALUDE_blue_paint_calculation_l887_88776


namespace NUMINAMATH_CALUDE_courtyard_width_l887_88723

/-- The width of a rectangular courtyard given its length and paving stone requirements. -/
theorem courtyard_width
  (length : ℝ)
  (num_stones : ℕ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (h1 : length = 50)
  (h2 : num_stones = 165)
  (h3 : stone_length = 5/2)
  (h4 : stone_width = 2)
  : (num_stones * stone_length * stone_width) / length = 33/2 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l887_88723


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l887_88799

theorem binomial_expansion_problem (a b : ℝ) (n : ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n + 1 → Nat.choose n (k - 1) ≤ Nat.choose n 5) ∧
  (a + b = 4) →
  (n = 10) ∧
  ((4^n + 7) % 3 = 2) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l887_88799


namespace NUMINAMATH_CALUDE_round_robin_tournament_games_l887_88779

theorem round_robin_tournament_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_games_l887_88779


namespace NUMINAMATH_CALUDE_greatest_x_under_conditions_l887_88711

theorem greatest_x_under_conditions (x : ℕ) : 
  x % 4 = 0 → 
  x > 0 → 
  x^3 < 8000 → 
  x ≤ 16 ∧ 
  ∀ y : ℕ, y % 4 = 0 → y > 0 → y^3 < 8000 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_under_conditions_l887_88711


namespace NUMINAMATH_CALUDE_unique_number_exists_l887_88764

theorem unique_number_exists : ∃! x : ℝ, x > 0 ∧ 100000 * x = 5 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l887_88764


namespace NUMINAMATH_CALUDE_log_equation_solution_l887_88761

theorem log_equation_solution :
  ∀ x : ℝ, (x + 5 > 0) ∧ (2*x - 1 > 0) ∧ (3*x^2 - 11*x + 5 > 0) →
  (Real.log (x + 5) + Real.log (2*x - 1) = Real.log (3*x^2 - 11*x + 5)) ↔
  (x = 10 + 3 * Real.sqrt 10 ∨ x = 10 - 3 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l887_88761


namespace NUMINAMATH_CALUDE_divisibility_by_five_l887_88745

theorem divisibility_by_five (a b : ℕ) : 
  (5 ∣ (a * b)) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l887_88745


namespace NUMINAMATH_CALUDE_cubic_of_99999_l887_88754

theorem cubic_of_99999 :
  let N : ℕ := 99999
  N^3 = 999970000299999 := by
sorry

end NUMINAMATH_CALUDE_cubic_of_99999_l887_88754


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l887_88752

def num_red_balls : ℕ := 2
def num_white_balls : ℕ := 6

def total_balls : ℕ := num_red_balls + num_white_balls

theorem probability_of_red_ball :
  (num_red_balls : ℚ) / (total_balls : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l887_88752


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l887_88704

theorem complex_number_in_second_quadrant :
  let z : ℂ := (2 * Complex.I) / (2 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l887_88704


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_the_year_l887_88796

/-- A prime p is a Prime of the Year if there exists a positive integer n such that n^2 + 1 ≡ 0 (mod p^2007) -/
def PrimeOfTheYear (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, n > 0 ∧ (n^2 + 1) % p^2007 = 0

/-- There are infinitely many Primes of the Year -/
theorem infinitely_many_primes_of_the_year :
  ∀ N : ℕ, ∃ p : ℕ, p > N ∧ PrimeOfTheYear p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_the_year_l887_88796


namespace NUMINAMATH_CALUDE_bhupathi_amount_l887_88767

theorem bhupathi_amount (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A + B = 1210) (h4 : (4/15) * A = (2/5) * B) : B = 484 :=
by
  sorry

end NUMINAMATH_CALUDE_bhupathi_amount_l887_88767


namespace NUMINAMATH_CALUDE_nigels_initial_amount_l887_88701

theorem nigels_initial_amount 
  (olivia_initial : ℕ) 
  (ticket_price : ℕ) 
  (num_tickets : ℕ) 
  (amount_left : ℕ) 
  (h1 : olivia_initial = 112)
  (h2 : ticket_price = 28)
  (h3 : num_tickets = 6)
  (h4 : amount_left = 83) :
  olivia_initial + (ticket_price * num_tickets - (olivia_initial - amount_left)) = 251 :=
by sorry

end NUMINAMATH_CALUDE_nigels_initial_amount_l887_88701


namespace NUMINAMATH_CALUDE_max_b_over_a_l887_88716

theorem max_b_over_a (a b : ℝ) (h_a : a > 0) : 
  (∀ x : ℝ, a * Real.exp x ≥ 2 * x + b) → b / a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_b_over_a_l887_88716


namespace NUMINAMATH_CALUDE_sugar_needed_is_six_l887_88730

/-- Represents the ratios in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of sugar needed in the new recipe --/
def sugar_needed (original : RecipeRatio) (water_new : ℚ) : ℚ :=
  let flour_water_ratio_new := 2 * (original.flour / original.water)
  let flour_sugar_ratio_new := (original.flour / original.sugar) / 2
  let flour_new := flour_water_ratio_new * water_new
  flour_new / flour_sugar_ratio_new

/-- Theorem stating that the amount of sugar needed is 6 cups --/
theorem sugar_needed_is_six :
  let original := RecipeRatio.mk 8 4 3
  let water_new := 2
  sugar_needed original water_new = 6 := by
  sorry

#eval sugar_needed (RecipeRatio.mk 8 4 3) 2

end NUMINAMATH_CALUDE_sugar_needed_is_six_l887_88730


namespace NUMINAMATH_CALUDE_coefficient_sum_equals_negative_eight_l887_88743

/-- Given a polynomial equation, prove that a specific linear combination of its coefficients equals -8 -/
theorem coefficient_sum_equals_negative_eight 
  (a : Fin 9 → ℝ) 
  (h : ∀ x : ℝ, x^5 * (x+3)^3 = (a 8)*(x+1)^8 + (a 7)*(x+1)^7 + (a 6)*(x+1)^6 + 
                               (a 5)*(x+1)^5 + (a 4)*(x+1)^4 + (a 3)*(x+1)^3 + 
                               (a 2)*(x+1)^2 + (a 1)*(x+1) + (a 0)) : 
  7*(a 7) + 5*(a 5) + 3*(a 3) + (a 1) = -8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_equals_negative_eight_l887_88743


namespace NUMINAMATH_CALUDE_men_who_left_job_l887_88760

/-- Given information about tree cutting rates, prove the number of men who left the job -/
theorem men_who_left_job (initial_men : ℕ) (initial_trees : ℕ) (initial_hours : ℕ)
  (final_trees : ℕ) (final_hours : ℕ) (h1 : initial_men = 20)
  (h2 : initial_trees = 30) (h3 : initial_hours = 4)
  (h4 : final_trees = 36) (h5 : final_hours = 6) :
  ∃ (men_left : ℕ),
    men_left = 4 ∧
    (initial_trees : ℚ) / initial_hours / initial_men =
    (final_trees : ℚ) / final_hours / (initial_men - men_left) :=
by sorry

end NUMINAMATH_CALUDE_men_who_left_job_l887_88760


namespace NUMINAMATH_CALUDE_thirteenth_square_vs_first_twelve_l887_88715

def grains (k : ℕ) : ℕ := 2^k

def sum_grains (n : ℕ) : ℕ := (grains (n + 1)) - 2

theorem thirteenth_square_vs_first_twelve :
  grains 13 = sum_grains 12 + 2 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_square_vs_first_twelve_l887_88715


namespace NUMINAMATH_CALUDE_imaginary_unit_equation_l887_88720

/-- Given that i is the imaginary unit and |((a+i)/i)| = 2, prove that a = √3 where a is a positive real number. -/
theorem imaginary_unit_equation (i : ℂ) (a : ℝ) (h1 : i * i = -1) (h2 : a > 0) :
  Complex.abs ((a + i) / i) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_equation_l887_88720


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l887_88713

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : m * n > 0) :
  1/m + 1/n ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l887_88713


namespace NUMINAMATH_CALUDE_ac_circuit_current_l887_88778

def V : ℂ := 2 + 2*Complex.I
def Z : ℂ := 2 - 2*Complex.I

theorem ac_circuit_current : V = Complex.I * Z := by sorry

end NUMINAMATH_CALUDE_ac_circuit_current_l887_88778


namespace NUMINAMATH_CALUDE_alpha_range_l887_88790

noncomputable def f (α : Real) (x : Real) : Real := Real.log x + Real.tan α

theorem alpha_range (α : Real) (x₀ : Real) :
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  x₀ < 1 →
  x₀ > 0 →
  (fun x => 1 / x) x₀ = f α x₀ →
  α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_alpha_range_l887_88790


namespace NUMINAMATH_CALUDE_limit_of_exponential_l887_88795

theorem limit_of_exponential (a : ℝ) :
  (a > 1 → ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → a^x > M) ∧
  (0 < a ∧ a < 1 → ∀ ε : ℝ, ε > 0 → ∃ N : ℝ, ∀ x : ℝ, x > N → a^x < ε) :=
by sorry

end NUMINAMATH_CALUDE_limit_of_exponential_l887_88795


namespace NUMINAMATH_CALUDE_tangent_line_correct_l887_88728

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 3)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

/-- Theorem stating that the tangent line equation is correct -/
theorem tangent_line_correct :
  let (a, b) := point
  tangent_line a b ∧
  ∀ x, tangent_line x (f x) → x = a := by sorry

end NUMINAMATH_CALUDE_tangent_line_correct_l887_88728


namespace NUMINAMATH_CALUDE_max_vouchers_for_680_yuan_l887_88735

/-- Represents the shopping voucher system with a given initial cash amount -/
structure VoucherSystem where
  initial_cash : ℕ
  voucher_rate : ℚ

/-- Calculates the maximum total vouchers that can be received -/
def max_vouchers (system : VoucherSystem) : ℕ :=
  sorry

/-- The theorem stating the maximum vouchers for the given problem -/
theorem max_vouchers_for_680_yuan :
  let system : VoucherSystem := { initial_cash := 680, voucher_rate := 1/5 }
  max_vouchers system = 160 := by
  sorry

end NUMINAMATH_CALUDE_max_vouchers_for_680_yuan_l887_88735


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l887_88786

/-- Given two points (x₁, y₁) and (x₂, y₂) on opposite sides of the line 3x - 2y + a = 0,
    prove that the range of values for 'a' is -4 < a < 9 -/
theorem opposite_sides_line_range (x₁ y₁ x₂ y₂ : ℝ) (h : (3*x₁ - 2*y₁ + a) * (3*x₂ - 2*y₂ + a) < 0) :
  -4 < a ∧ a < 9 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l887_88786


namespace NUMINAMATH_CALUDE_max_constant_value_l887_88732

theorem max_constant_value (c d : ℝ) : 
  (∃ (k : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) →
  (∃ (max_k : ℝ), ∀ (k : ℝ), (∃ (c d : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) → k ≤ max_k) →
  (∃ (max_k : ℝ), ∀ (k : ℝ), (∃ (c d : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) → k ≤ max_k) ∧
  (∃ (c d : ℝ), 5 * c + (d - 12)^2 = 235 ∧ c ≤ 47) :=
by sorry

end NUMINAMATH_CALUDE_max_constant_value_l887_88732


namespace NUMINAMATH_CALUDE_jessys_reading_plan_l887_88768

/-- Jessy's reading plan problem -/
theorem jessys_reading_plan (total_pages : ℕ) (days : ℕ) (pages_per_session : ℕ) (additional_pages : ℕ)
  (h1 : total_pages = 140)
  (h2 : days = 7)
  (h3 : pages_per_session = 6)
  (h4 : additional_pages = 2) :
  ∃ (sessions : ℕ), sessions * pages_per_session * days + additional_pages * days = total_pages ∧ sessions = 3 :=
by sorry

end NUMINAMATH_CALUDE_jessys_reading_plan_l887_88768


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l887_88726

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 - Complex.I) = 3 + Complex.I) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l887_88726


namespace NUMINAMATH_CALUDE_paco_cookies_l887_88753

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 19

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := 11

/-- The number of sweet cookies Paco ate in the first round -/
def sweet_cookies_eaten_first : ℕ := 5

/-- The number of salty cookies Paco ate in the first round -/
def salty_cookies_eaten_first : ℕ := 2

/-- The difference between sweet and salty cookies eaten in the second round -/
def sweet_salty_difference : ℕ := 3

theorem paco_cookies : 
  initial_sweet_cookies = 
    (initial_sweet_cookies - sweet_cookies_eaten_first - sweet_salty_difference) + 
    sweet_cookies_eaten_first + 
    (salty_cookies_eaten_first + sweet_salty_difference) :=
by sorry

end NUMINAMATH_CALUDE_paco_cookies_l887_88753


namespace NUMINAMATH_CALUDE_min_distance_parabola_circle_l887_88758

/-- The minimum distance between a point on the parabola y^2 = x and a point on the circle (x-3)^2 + y^2 = 1 is (√11)/2 - 1 -/
theorem min_distance_parabola_circle :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (d : ℝ), d = Real.sqrt 11 / 2 - 1 ∧
    ∀ (m : ℝ × ℝ) (n : ℝ × ℝ), m ∈ parabola → n ∈ circle →
      Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_circle_l887_88758


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l887_88731

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (Complex.I - 1) / Complex.I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l887_88731


namespace NUMINAMATH_CALUDE_max_d_value_in_multiple_of_13_l887_88785

theorem max_d_value_in_multiple_of_13 :
  let is_valid : (ℕ → ℕ → Bool) :=
    fun d e => (520000 + 10000 * d + 550 + 10 * e) % 13 = 0 ∧ 
               d < 10 ∧ e < 10
  ∃ d e, is_valid d e ∧ d = 6 ∧ ∀ d' e', is_valid d' e' → d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_in_multiple_of_13_l887_88785


namespace NUMINAMATH_CALUDE_car_distance_theorem_l887_88742

/-- Represents the car's driving characteristics and total driving time -/
structure CarDriving where
  speed : ℕ              -- Speed in miles per hour
  drive_time : ℕ         -- Continuous driving time in hours
  cool_time : ℕ          -- Cooling time in hours
  total_time : ℕ         -- Total available time in hours

/-- Calculates the total distance a car can travel given its driving characteristics -/
def total_distance (car : CarDriving) : ℕ :=
  sorry

/-- Theorem stating that a car with given characteristics can travel 88 miles in 13 hours -/
theorem car_distance_theorem :
  let car := CarDriving.mk 8 5 1 13
  total_distance car = 88 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l887_88742


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l887_88747

theorem quadratic_equation_solution : ∃ x : ℝ, (10 - x)^2 = x^2 + 6 ∧ x = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l887_88747


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l887_88766

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l887_88766


namespace NUMINAMATH_CALUDE_surface_area_difference_specific_l887_88737

/-- Calculates the surface area difference when removing a cube from a rectangular solid -/
def surface_area_difference (l w h : ℝ) (cube_side : ℝ) : ℝ :=
  let original_surface_area := 2 * (l * w + l * h + w * h)
  let new_faces_area := 2 * cube_side * cube_side
  let removed_faces_area := 5 * cube_side * cube_side
  new_faces_area - removed_faces_area

/-- The surface area difference for the specific problem -/
theorem surface_area_difference_specific :
  surface_area_difference 6 5 4 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_specific_l887_88737


namespace NUMINAMATH_CALUDE_gcf_of_1260_and_1440_l887_88798

theorem gcf_of_1260_and_1440 : Nat.gcd 1260 1440 = 180 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_1260_and_1440_l887_88798


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l887_88775

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) :
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p)^3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l887_88775


namespace NUMINAMATH_CALUDE_profit_increase_l887_88789

theorem profit_increase (initial_profit : ℝ) (h : initial_profit > 0) :
  let april_profit := initial_profit * 1.2
  let may_profit := april_profit * 0.8
  let june_profit := initial_profit * 1.4399999999999999
  (june_profit / may_profit - 1) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l887_88789


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l887_88700

/-- Proves that the number of years until a man's age is twice his son's age is 2 -/
theorem mans_age_twice_sons_age (man_age son_age years : ℕ) : 
  man_age = son_age + 28 →
  son_age = 26 →
  (man_age + years) = 2 * (son_age + years) →
  years = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l887_88700


namespace NUMINAMATH_CALUDE_sum_of_numbers_l887_88765

-- Define the range of numbers
def valid_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 50

-- Define primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define Alice's number
def alice_number (a : ℕ) : Prop := valid_number a

-- Define Bob's number
def bob_number (b : ℕ) : Prop := valid_number b ∧ is_prime b

-- Alice can't determine who has the larger number
def alice_uncertainty (a b : ℕ) : Prop :=
  alice_number a → bob_number b → ¬(a > b ∨ b > a)

-- Bob can determine who has the larger number after Alice's statement
def bob_certainty (a b : ℕ) : Prop :=
  alice_number a → bob_number b → (a > b ∨ b > a)

-- 200 * Bob's number + Alice's number is a perfect square
def perfect_square_condition (a b : ℕ) : Prop :=
  alice_number a → bob_number b → ∃ k : ℕ, 200 * b + a = k * k

-- Theorem statement
theorem sum_of_numbers (a b : ℕ) :
  alice_number a →
  bob_number b →
  alice_uncertainty a b →
  bob_certainty a b →
  perfect_square_condition a b →
  a + b = 43 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l887_88765


namespace NUMINAMATH_CALUDE_cookies_and_game_cost_l887_88770

-- Define the quantities of each item
def bracelets : ℕ := 12
def necklaces : ℕ := 8
def rings : ℕ := 20

-- Define the costs to make each item
def bracelet_cost : ℚ := 1
def necklace_cost : ℚ := 2
def ring_cost : ℚ := 1/2

-- Define the selling prices of each item
def bracelet_price : ℚ := 3/2
def necklace_price : ℚ := 3
def ring_price : ℚ := 1

-- Define the target profit margin
def target_margin : ℚ := 1/2

-- Define the remaining money after purchases
def remaining_money : ℚ := 5

-- Theorem to prove
theorem cookies_and_game_cost :
  let total_cost := bracelets * bracelet_cost + necklaces * necklace_cost + rings * ring_cost
  let total_revenue := bracelets * bracelet_price + necklaces * necklace_price + rings * ring_price
  let profit := total_revenue - total_cost
  let target_profit := total_cost * target_margin
  let cost_of_purchases := profit - remaining_money
  cost_of_purchases = 43 := by sorry

end NUMINAMATH_CALUDE_cookies_and_game_cost_l887_88770


namespace NUMINAMATH_CALUDE_sweater_wool_correct_l887_88741

/-- The number of balls of wool used for a sweater -/
def sweater_wool : ℕ := 4

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for a scarf -/
def scarf_wool : ℕ := 3

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem sweater_wool_correct : 
  aaron_scarves * scarf_wool + (aaron_sweaters + enid_sweaters) * sweater_wool = total_wool := by
  sorry

end NUMINAMATH_CALUDE_sweater_wool_correct_l887_88741


namespace NUMINAMATH_CALUDE_hundred_squared_plus_201_is_composite_l887_88703

theorem hundred_squared_plus_201_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 100^2 + 201 = a * b := by
  sorry

end NUMINAMATH_CALUDE_hundred_squared_plus_201_is_composite_l887_88703


namespace NUMINAMATH_CALUDE_caesars_meal_charge_is_30_l887_88774

/-- Represents the charge per meal at Caesar's -/
def caesars_meal_charge : ℝ := sorry

/-- Caesar's room rental fee -/
def caesars_room_fee : ℝ := 800

/-- Venus Hall's room rental fee -/
def venus_room_fee : ℝ := 500

/-- Venus Hall's charge per meal -/
def venus_meal_charge : ℝ := 35

/-- Number of guests when the costs are equal -/
def num_guests : ℕ := 60

theorem caesars_meal_charge_is_30 :
  caesars_room_fee + num_guests * caesars_meal_charge =
  venus_room_fee + num_guests * venus_meal_charge →
  caesars_meal_charge = 30 := by sorry

end NUMINAMATH_CALUDE_caesars_meal_charge_is_30_l887_88774


namespace NUMINAMATH_CALUDE_pats_stick_is_30_inches_l887_88772

/-- The length of Pat's stick in inches -/
def pats_stick_length : ℝ := 30

/-- The length of the portion of Pat's stick covered in dirt, in inches -/
def covered_portion : ℝ := 7

/-- The length of Sarah's stick in inches -/
def sarahs_stick_length : ℝ := 46

/-- The length of Jane's stick in inches -/
def janes_stick_length : ℝ := 22

/-- Proves that Pat's stick is 30 inches long given the conditions -/
theorem pats_stick_is_30_inches :
  (pats_stick_length = covered_portion + (sarahs_stick_length / 2)) ∧
  (janes_stick_length = sarahs_stick_length - 24) ∧
  (janes_stick_length = 22) →
  pats_stick_length = 30 := by sorry

end NUMINAMATH_CALUDE_pats_stick_is_30_inches_l887_88772


namespace NUMINAMATH_CALUDE_remainder_sum_mod_59_l887_88746

theorem remainder_sum_mod_59 (a b c : ℕ+) 
  (ha : a ≡ 28 [ZMOD 59])
  (hb : b ≡ 34 [ZMOD 59])
  (hc : c ≡ 5 [ZMOD 59]) :
  (a + b + c) ≡ 8 [ZMOD 59] := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_59_l887_88746


namespace NUMINAMATH_CALUDE_pet_store_choices_l887_88727

def num_puppies : ℕ := 10
def num_kittens : ℕ := 8
def num_bunnies : ℕ := 12

def alice_choices : ℕ := num_kittens + num_bunnies
def bob_choices (alice_choice : ℕ) : ℕ :=
  if alice_choice ≤ num_kittens then num_puppies + num_bunnies
  else num_puppies + (num_bunnies - 1)
def charlie_choices (alice_choice bob_choice : ℕ) : ℕ :=
  num_puppies + num_kittens + num_bunnies - alice_choice - bob_choice

def total_choices : ℕ :=
  (alice_choices * bob_choices num_kittens * charlie_choices num_kittens num_puppies) +
  (alice_choices * bob_choices num_kittens * charlie_choices num_kittens num_bunnies) +
  (alice_choices * bob_choices num_bunnies * charlie_choices num_bunnies num_puppies) +
  (alice_choices * bob_choices num_bunnies * charlie_choices num_bunnies (num_bunnies - 1))

theorem pet_store_choices : total_choices = 4120 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_choices_l887_88727


namespace NUMINAMATH_CALUDE_sea_glass_collection_l887_88788

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red dorothy_total : ℕ)
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : dorothy_total = 57) :
  ∃ (rose_blue : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧
    rose_blue = 11 := by
  sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l887_88788


namespace NUMINAMATH_CALUDE_haley_video_files_l887_88762

/-- Given the initial number of music files, the number of deleted files,
    and the remaining number of files, calculate the initial number of video files. -/
def initialVideoFiles (initialMusicFiles deletedFiles remainingFiles : ℕ) : ℕ :=
  remainingFiles + deletedFiles - initialMusicFiles

theorem haley_video_files :
  initialVideoFiles 27 11 58 = 42 := by
  sorry

end NUMINAMATH_CALUDE_haley_video_files_l887_88762


namespace NUMINAMATH_CALUDE_sum_proper_divisors_256_l887_88755

theorem sum_proper_divisors_256 : 
  (Finset.filter (fun d => d ≠ 256 ∧ 256 % d = 0) (Finset.range 257)).sum id = 255 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_256_l887_88755


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l887_88781

theorem quadratic_root_in_unit_interval (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l887_88781


namespace NUMINAMATH_CALUDE_sequence_formula_l887_88721

def sequence_a (n : ℕ) : ℕ := 2^n - 1

def sum_S : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * sum_S n + n + 1

theorem sequence_formula (n : ℕ) :
  n > 0 →
  sequence_a 1 = 1 ∧
  (∀ k, k > 0 → sum_S (k + 1) = 2 * sum_S k + k + 1) →
  sequence_a n = sum_S n - sum_S (n - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l887_88721


namespace NUMINAMATH_CALUDE_constant_reciprocal_sum_parabola_l887_88718

/-- Theorem: Constant Reciprocal Sum of Squared Distances on Parabola
  Given a point P(a,0) on the x-axis and a line through P intersecting
  the parabola y^2 = 8x at points A and B, if the sum of reciprocals of
  squared distances 1/|AP^2| + 1/|BP^2| is constant for all such lines,
  then a = 4. -/
theorem constant_reciprocal_sum_parabola (a : ℝ) : 
  (∀ m : ℝ, ∃ A B : ℝ × ℝ, 
    (A.2)^2 = 8 * A.1 ∧ 
    (B.2)^2 = 8 * B.1 ∧ 
    A.1 = m * A.2 + a ∧ 
    B.1 = m * B.2 + a ∧
    (∃ k : ℝ, ∀ m : ℝ, 
      1 / ((A.1 - a)^2 + (A.2)^2) + 1 / ((B.1 - a)^2 + (B.2)^2) = k)) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_reciprocal_sum_parabola_l887_88718


namespace NUMINAMATH_CALUDE_factorization_problem_l887_88787

theorem factorization_problem (x y : ℝ) : (x - y)^2 - 2*(x - y) + 1 = (x - y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_l887_88787


namespace NUMINAMATH_CALUDE_scatter_plot_regression_role_l887_88709

/-- The role of a scatter plot in regression analysis -/
def scatter_plot_role : String :=
  "to roughly judge whether variables are linearly related"

/-- The main theorem about the role of scatter plots in regression analysis -/
theorem scatter_plot_regression_role :
  scatter_plot_role = "to roughly judge whether variables are linearly related" := by
  sorry

end NUMINAMATH_CALUDE_scatter_plot_regression_role_l887_88709


namespace NUMINAMATH_CALUDE_selling_price_for_target_profit_l887_88759

-- Define the cost price
def cost_price : ℝ := 40

-- Define the function for monthly sales volume based on selling price
def sales_volume (x : ℝ) : ℝ := 1000 - 10 * x

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

-- Theorem stating the selling prices that result in 8000 yuan profit
theorem selling_price_for_target_profit : 
  ∃ (x : ℝ), (x = 60 ∨ x = 80) ∧ profit x = 8000 := by
  sorry


end NUMINAMATH_CALUDE_selling_price_for_target_profit_l887_88759


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l887_88706

theorem average_of_three_numbers (y : ℝ) : (15 + 24 + y) / 3 = 20 → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l887_88706


namespace NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_root_l887_88751

theorem sqrt_two_plus_sqrt_three_root : ∃ x : ℝ, x = Real.sqrt 2 + Real.sqrt 3 ∧ x^4 - 10*x^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_root_l887_88751


namespace NUMINAMATH_CALUDE_not_perfect_square_for_prime_l887_88793

theorem not_perfect_square_for_prime (p : ℕ) (h : Prime p) : ¬ ∃ t : ℤ, (7 * p + 3^p - 4 : ℤ) = t^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_for_prime_l887_88793


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l887_88734

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x + Real.log (1 - x)}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the half-open interval [0, 1)
def interval_zero_one : Set ℝ := {x | 0 ≤ x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_interval : M_intersect_N = interval_zero_one := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l887_88734


namespace NUMINAMATH_CALUDE_point_inside_ellipse_l887_88792

theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) → (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_ellipse_l887_88792


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l887_88750

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (63 - 21*x - x^2 = 0) → 
  (∃ r s : ℝ, (63 - 21*r - r^2 = 0) ∧ (63 - 21*s - s^2 = 0) ∧ (r + s = -21)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l887_88750


namespace NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l887_88783

/-- The minimum positive integer n such that the expansion of (x^2 + 1/(2x^3))^n contains a constant term, where x is a positive integer. -/
def min_n_constant_term : ℕ := 5

/-- Predicate to check if the expansion of (x^2 + 1/(2x^3))^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (k : ℕ), 2 * n = 5 * k

theorem min_n_constant_term_is_correct :
  (∀ m : ℕ, m < min_n_constant_term → ¬has_constant_term m) ∧
  has_constant_term min_n_constant_term :=
sorry

end NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l887_88783


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_and_q_false_l887_88794

theorem not_p_or_q_false_implies_p_and_q_false (p q : Prop) :
  (¬(¬p ∨ q)) → ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_and_q_false_l887_88794


namespace NUMINAMATH_CALUDE_remainder_sum_l887_88748

theorem remainder_sum (c d : ℤ) (hc : c % 60 = 47) (hd : d % 45 = 14) : (c + d) % 15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l887_88748


namespace NUMINAMATH_CALUDE_first_number_value_l887_88736

theorem first_number_value (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 →
  (b + c + d) / 3 = 15 →
  d = 18 →
  a = 33 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l887_88736


namespace NUMINAMATH_CALUDE_infinite_diamond_2005_l887_88719

/-- A number is diamond 2005 if it has the form ...ab999...99999cd... -/
def is_diamond_2005 (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ) (k m : ℕ), n = a * 10^(k+m+4) + b * 10^(k+m+3) + 999 * 10^m + c * 10 + d

/-- A sequence {a_n} is bounded by C*n if for all n, a_n < C*n -/
def is_bounded_by_linear (a : ℕ → ℕ) (C : ℝ) : Prop :=
  ∀ n, (a n : ℝ) < C * n

/-- A sequence {a_n} is increasing if for all n, a_n <= a_(n+1) -/
def is_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n, a n ≤ a (n + 1)

/-- Main theorem: An increasing sequence bounded by C*n contains infinitely many diamond 2005 numbers -/
theorem infinite_diamond_2005 (a : ℕ → ℕ) (C : ℝ) 
  (h_bound : is_bounded_by_linear a C) 
  (h_incr : is_increasing a) : 
  ∀ m : ℕ, ∃ n > m, is_diamond_2005 (a n) :=
sorry

end NUMINAMATH_CALUDE_infinite_diamond_2005_l887_88719


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l887_88797

theorem smaller_number_in_ratio (n m d u x y : ℝ) : 
  0 < n → n < m → x > 0 → y > 0 → x / y = n / m → x + y + u = d → 
  min x y = n * (d - u) / (n + m) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l887_88797


namespace NUMINAMATH_CALUDE_min_value_expression_l887_88738

theorem min_value_expression (x y : ℝ) : (x^2*y - 1)^2 + (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l887_88738


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l887_88739

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a)
    (h_a1 : a 1 = 1/3)
    (h_sum : a 2 + a 5 = 4)
    (h_an : ∃ n : ℕ, a n = 33) :
  ∃ n : ℕ, a n = 33 ∧ n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l887_88739


namespace NUMINAMATH_CALUDE_compute_expression_l887_88702

theorem compute_expression : 12 * (216 / 3 + 36 / 6 + 16 / 8 + 2) = 984 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l887_88702


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l887_88784

/-- Given the total marks in mathematics and physics is 70, and chemistry score is 20 marks more than physics score, prove that the average marks in mathematics and chemistry is 45. -/
theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 70 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l887_88784


namespace NUMINAMATH_CALUDE_special_pyramid_volume_l887_88712

/-- Represents a pyramid with an equilateral triangle base and isosceles right triangle lateral faces -/
structure SpecialPyramid where
  base_side_length : ℝ
  is_equilateral_base : base_side_length > 0
  is_isosceles_right_lateral : True

/-- Calculates the volume of the special pyramid -/
def volume (p : SpecialPyramid) : ℝ :=
  sorry

/-- Theorem stating that the volume of the special pyramid with base side length 2 is √2/3 -/
theorem special_pyramid_volume :
  ∀ (p : SpecialPyramid), p.base_side_length = 2 → volume p = Real.sqrt 2 / 3 :=
  sorry

end NUMINAMATH_CALUDE_special_pyramid_volume_l887_88712


namespace NUMINAMATH_CALUDE_negative_solutions_count_l887_88714

def f (x : ℤ) : ℤ := x^6 - 75*x^4 + 1000*x^2 - 6000

theorem negative_solutions_count :
  ∃! (S : Finset ℤ), (∀ x ∈ S, f x < 0) ∧ (∀ x ∉ S, f x ≥ 0) ∧ Finset.card S = 12 := by
  sorry

end NUMINAMATH_CALUDE_negative_solutions_count_l887_88714


namespace NUMINAMATH_CALUDE_student_sums_correct_l887_88756

theorem student_sums_correct (total_sums : ℕ) (wrong_ratio : ℕ) 
  (h1 : total_sums = 48) 
  (h2 : wrong_ratio = 2) : 
  ∃ (correct_sums : ℕ), 
    correct_sums + wrong_ratio * correct_sums = total_sums ∧ 
    correct_sums = 16 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_correct_l887_88756


namespace NUMINAMATH_CALUDE_complex_number_problem_l887_88705

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 →
  z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l887_88705


namespace NUMINAMATH_CALUDE_sara_lunch_bill_l887_88710

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_bill (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch bill is the sum of the hotdog and salad prices -/
theorem sara_lunch_bill :
  lunch_bill 5.36 5.10 = 10.46 :=
by sorry

end NUMINAMATH_CALUDE_sara_lunch_bill_l887_88710


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_t1_l887_88782

-- Define the motion distance function
def S (t : ℝ) : ℝ := t^3 - 2

-- Define the derivative of S
def S_derivative (t : ℝ) : ℝ := 3 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_t1 :
  S_derivative 1 = 3 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_t1_l887_88782


namespace NUMINAMATH_CALUDE_laptop_installment_calculation_l887_88733

/-- Calculates the monthly installment amount for a laptop purchase --/
theorem laptop_installment_calculation (laptop_cost : ℝ) (down_payment_percentage : ℝ) 
  (additional_down_payment : ℝ) (balance_after_four_months : ℝ) 
  (h1 : laptop_cost = 1000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : additional_down_payment = 20)
  (h4 : balance_after_four_months = 520) : 
  ∃ (monthly_installment : ℝ), monthly_installment = 65 := by
  sorry

#check laptop_installment_calculation

end NUMINAMATH_CALUDE_laptop_installment_calculation_l887_88733


namespace NUMINAMATH_CALUDE_clock_hands_angle_l887_88749

theorem clock_hands_angle (n : ℕ) : 0 < n ∧ n < 720 → (∃ k : ℤ, |11 * n / 2 % 360 - 360 * k| = 1) ↔ n = 262 ∨ n = 458 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_angle_l887_88749


namespace NUMINAMATH_CALUDE_seventh_term_approx_l887_88744

/-- Represents a geometric sequence with 10 terms -/
structure GeometricSequence where
  a₁ : ℝ
  r : ℝ
  len : ℕ
  h_len : len = 10
  h_a₁ : a₁ = 4
  h_a₄ : a₁ * r^3 = 64
  h_a₁₀ : a₁ * r^9 = 39304

/-- The 7th term of the geometric sequence -/
def seventh_term (seq : GeometricSequence) : ℝ :=
  seq.a₁ * seq.r^6

/-- Theorem stating that the 7th term is approximately 976 -/
theorem seventh_term_approx (seq : GeometricSequence) :
  ∃ ε > 0, |seventh_term seq - 976| < ε :=
sorry

end NUMINAMATH_CALUDE_seventh_term_approx_l887_88744


namespace NUMINAMATH_CALUDE_defect_rate_calculation_l887_88780

/-- Calculates the overall defect rate given three suppliers' defect rates and their supply ratios -/
def overall_defect_rate (rate1 rate2 rate3 : ℚ) (ratio1 ratio2 ratio3 : ℕ) : ℚ :=
  (rate1 * ratio1 + rate2 * ratio2 + rate3 * ratio3) / (ratio1 + ratio2 + ratio3)

/-- Theorem stating that the overall defect rate for the given problem is 14/15 -/
theorem defect_rate_calculation :
  overall_defect_rate (92/100) (95/100) (94/100) 3 2 1 = 14/15 := by
  sorry

end NUMINAMATH_CALUDE_defect_rate_calculation_l887_88780


namespace NUMINAMATH_CALUDE_f_derivative_l887_88740

/-- The function f(x) = (5x - 4)^3 -/
def f (x : ℝ) : ℝ := (5 * x - 4) ^ 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 15 * (5 * x - 4) ^ 2

theorem f_derivative :
  ∀ x : ℝ, deriv f x = f' x :=
by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l887_88740


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l887_88791

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (λ x => x ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  {t : ℕ | is_valid_turnip_weight t} = {13, 16} := by
  sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l887_88791
