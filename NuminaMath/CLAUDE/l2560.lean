import Mathlib

namespace profit_per_meter_l2560_256034

/-- Given a trader selling cloth, calculate the profit per meter. -/
theorem profit_per_meter (total_profit : ℝ) (total_meters : ℝ) (h1 : total_profit = 1400) (h2 : total_meters = 40) :
  total_profit / total_meters = 35 := by
sorry

end profit_per_meter_l2560_256034


namespace sum_of_divisors_prime_power_sum_of_divisors_two_prime_powers_sum_of_divisors_three_prime_powers_l2560_256081

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Theorem for p^α
theorem sum_of_divisors_prime_power (p : ℕ) (α : ℕ) (hp : Prime p) :
  sumOfDivisors (p^α) = (p^(α+1) - 1) / (p - 1) := by sorry

-- Theorem for p^α q^β
theorem sum_of_divisors_two_prime_powers (p q : ℕ) (α β : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  sumOfDivisors (p^α * q^β) = ((p^(α+1) - 1) / (p - 1)) * ((q^(β+1) - 1) / (q - 1)) := by sorry

-- Theorem for p^α q^β r^γ
theorem sum_of_divisors_three_prime_powers (p q r : ℕ) (α β γ : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  sumOfDivisors (p^α * q^β * r^γ) = ((p^(α+1) - 1) / (p - 1)) * ((q^(β+1) - 1) / (q - 1)) * ((r^(γ+1) - 1) / (r - 1)) := by sorry

end sum_of_divisors_prime_power_sum_of_divisors_two_prime_powers_sum_of_divisors_three_prime_powers_l2560_256081


namespace car_tractor_distance_theorem_l2560_256030

theorem car_tractor_distance_theorem (total_distance : ℝ) 
  (first_meeting_time : ℝ) (car_wait_time : ℝ) (car_catch_up_time : ℝ) :
  total_distance = 160 ∧ 
  first_meeting_time = 4/3 ∧ 
  car_wait_time = 1 ∧ 
  car_catch_up_time = 1/2 →
  ∃ (car_speed tractor_speed : ℝ),
    car_speed > 0 ∧ tractor_speed > 0 ∧
    car_speed + tractor_speed = total_distance / first_meeting_time ∧
    car_speed * (first_meeting_time + car_catch_up_time) = 165 ∧
    tractor_speed * (first_meeting_time + car_wait_time + car_catch_up_time) = 85 :=
by sorry

end car_tractor_distance_theorem_l2560_256030


namespace inequality_may_not_hold_l2560_256094

theorem inequality_may_not_hold (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, a / c ≤ b / c := by
  sorry

end inequality_may_not_hold_l2560_256094


namespace box_bottle_count_l2560_256048

def dozen : ℕ := 12

def water_bottles : ℕ := 2 * dozen

def apple_bottles : ℕ := water_bottles + (dozen / 2)

def total_bottles : ℕ := water_bottles + apple_bottles

theorem box_bottle_count : total_bottles = 54 := by
  sorry

end box_bottle_count_l2560_256048


namespace circle_area_ratio_l2560_256091

theorem circle_area_ratio (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h_diameter : 2 * r = 0.2 * (2 * s)) : 
  (π * r^2) / (π * s^2) = 0.01 := by
  sorry

end circle_area_ratio_l2560_256091


namespace isosceles_triangle_area_l2560_256099

/-- The area of an isosceles triangle with two sides of length 5 and base of length 6 is 12 -/
theorem isosceles_triangle_area : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 6 →
  (∃ (h : ℝ), h^2 = a^2 - (c/2)^2) →
  (1/2) * c * (a^2 - (c/2)^2).sqrt = 12 := by
sorry

end isosceles_triangle_area_l2560_256099


namespace quadratic_roots_relation_l2560_256089

theorem quadratic_roots_relation (p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) ∧ 
  (x₂^2 + p*x₂ + q = 0) ∧ 
  ((x₁ + 1)^2 + q*(x₁ + 1) + p = 0) ∧ 
  ((x₂ + 1)^2 + q*(x₂ + 1) + p = 0) →
  p = -1 ∧ q = -3 := by
sorry

end quadratic_roots_relation_l2560_256089


namespace visitor_difference_l2560_256073

/-- The number of paintings in Buckingham Palace -/
def paintings : ℕ := 39

/-- The number of visitors on the current day -/
def visitors_current : ℕ := 661

/-- The number of visitors on the previous day -/
def visitors_previous : ℕ := 600

/-- Theorem: The difference in visitors between the current day and the previous day is 61 -/
theorem visitor_difference : visitors_current - visitors_previous = 61 := by
  sorry

end visitor_difference_l2560_256073


namespace unique_a_value_l2560_256028

-- Define the inequality function
def inequality (a x : ℝ) : Prop :=
  (a * x - 20) * Real.log (2 * a / x) ≤ 0

-- State the theorem
theorem unique_a_value : 
  ∃! a : ℝ, ∀ x : ℝ, x > 0 → inequality a x :=
by
  -- The unique value of a is √10
  use Real.sqrt 10
  sorry -- Proof omitted

end unique_a_value_l2560_256028


namespace function_increasing_on_interval_l2560_256071

/-- The function f(x) = 3x - x^3 is monotonically increasing on the interval [-1, 1]. -/
theorem function_increasing_on_interval (x : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → (3 * x₁ - x₁^3) < (3 * x₂ - x₂^3)) :=
by sorry

end function_increasing_on_interval_l2560_256071


namespace monthly_calendar_sum_l2560_256043

theorem monthly_calendar_sum : ∃ (x : ℕ), 
  8 ≤ x ∧ x ≤ 24 ∧ 
  ∃ (k : ℕ), (x - 7) + x + (x + 7) = 3 * k ∧ 
  (x - 7) + x + (x + 7) = 33 := by
  sorry

end monthly_calendar_sum_l2560_256043


namespace point_movement_l2560_256041

def Point := ℝ × ℝ

def moveUp (p : Point) (units : ℝ) : Point :=
  (p.1, p.2 + units)

def moveRight (p : Point) (units : ℝ) : Point :=
  (p.1 + units, p.2)

theorem point_movement :
  let original : Point := (-2, 3)
  (moveUp original 2 = (-2, 5)) ∧
  (moveRight original 2 = (0, 3)) := by
  sorry

end point_movement_l2560_256041


namespace joan_football_games_l2560_256005

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to this year and last year -/
def total_games : ℕ := 9

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 5 := by
  sorry

end joan_football_games_l2560_256005


namespace mosquitoes_to_cause_death_l2560_256027

/-- Represents the number of drops of blood sucked by a mosquito of a given species -/
def drops_per_species : Fin 3 → ℕ
  | 0 => 20  -- Species A
  | 1 => 25  -- Species B
  | 2 => 30  -- Species C
  | _ => 0   -- This case is unreachable, but needed for completeness

/-- The number of drops of blood per liter -/
def drops_per_liter : ℕ := 5000

/-- The number of liters of blood loss that causes death -/
def lethal_blood_loss : ℕ := 3

/-- The total number of drops of blood that cause death -/
def lethal_drops : ℕ := lethal_blood_loss * drops_per_liter

/-- The theorem stating the number of mosquitoes of each species required to cause death -/
theorem mosquitoes_to_cause_death :
  ∃ n : ℕ, n > 0 ∧ 
  (n * drops_per_species 0 + n * drops_per_species 1 + n * drops_per_species 2 = lethal_drops) ∧
  n = 200 := by
  sorry

end mosquitoes_to_cause_death_l2560_256027


namespace kot_ycehyj_inequality_l2560_256079

theorem kot_ycehyj_inequality : 
  ∀ (K O T Y C E H J : ℕ),
    K ∈ Finset.range 9 ∧ 
    O ∈ Finset.range 9 ∧ 
    T ∈ Finset.range 9 ∧ 
    Y ∈ Finset.range 9 ∧ 
    C ∈ Finset.range 9 ∧ 
    E ∈ Finset.range 9 ∧ 
    H ∈ Finset.range 9 ∧ 
    J ∈ Finset.range 9 ∧
    K ≠ O ∧ K ≠ T ∧ K ≠ Y ∧ K ≠ C ∧ K ≠ E ∧ K ≠ H ∧ K ≠ J ∧
    O ≠ T ∧ O ≠ Y ∧ O ≠ C ∧ O ≠ E ∧ O ≠ H ∧ O ≠ J ∧
    T ≠ Y ∧ T ≠ C ∧ T ≠ E ∧ T ≠ H ∧ T ≠ J ∧
    Y ≠ C ∧ Y ≠ E ∧ Y ≠ H ∧ Y ≠ J ∧
    C ≠ E ∧ C ≠ H ∧ C ≠ J ∧
    E ≠ H ∧ E ≠ J ∧
    H ≠ J →
    K * O * T < Y * C * E * H * Y * J :=
by sorry

end kot_ycehyj_inequality_l2560_256079


namespace club_membership_l2560_256040

theorem club_membership (total_members : ℕ) (attendance : ℕ) (men : ℕ) (women : ℕ) : 
  total_members = 30 →
  attendance = 18 →
  total_members = men + women →
  attendance = men + (women / 3) →
  men = 12 := by
sorry

end club_membership_l2560_256040


namespace orange_juice_proportion_l2560_256072

theorem orange_juice_proportion (oranges : ℝ) (quarts : ℝ) :
  oranges / quarts = 36 / 48 →
  quarts = 6 →
  oranges = 4.5 := by
  sorry

end orange_juice_proportion_l2560_256072


namespace bottom_price_is_3350_l2560_256000

/-- The price of a bottom pajama in won -/
def bottom_price : ℕ := sorry

/-- The price of a top pajama in won -/
def top_price : ℕ := sorry

/-- The number of pajama sets bought -/
def num_sets : ℕ := 3

/-- The total amount paid in won -/
def total_paid : ℕ := 21000

/-- The price difference between top and bottom in won -/
def price_difference : ℕ := 300

theorem bottom_price_is_3350 : 
  bottom_price = 3350 ∧ 
  top_price = bottom_price + price_difference ∧
  num_sets * (bottom_price + top_price) = total_paid := by
  sorry

end bottom_price_is_3350_l2560_256000


namespace divided_isosceles_triangle_theorem_l2560_256051

/-- An isosceles triangle with a parallel line dividing it -/
structure DividedIsoscelesTriangle where
  /-- The length of the base of the isosceles triangle -/
  base : ℝ
  /-- The length of the parallel line dividing the triangle -/
  parallel_line : ℝ
  /-- The ratio of the area of the smaller region to the whole triangle -/
  area_ratio : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The parallel line is positive and not longer than the base -/
  parallel_line_bounds : 0 < parallel_line ∧ parallel_line ≤ base
  /-- The area ratio is between 0 and 1 -/
  area_ratio_bounds : 0 < area_ratio ∧ area_ratio < 1
  /-- The parallel line divides the triangle according to the area ratio -/
  division_property : (parallel_line / base) ^ 2 = area_ratio

/-- The theorem stating the properties of the divided isosceles triangle -/
theorem divided_isosceles_triangle_theorem (t : DividedIsoscelesTriangle) 
  (h_base : t.base = 24)
  (h_ratio : t.area_ratio = 1/4) : 
  t.parallel_line = 12 := by
  sorry

end divided_isosceles_triangle_theorem_l2560_256051


namespace product_of_three_consecutive_even_numbers_divisible_by_48_l2560_256097

theorem product_of_three_consecutive_even_numbers_divisible_by_48 (k : ℤ) :
  ∃ (n : ℤ), (2*k) * (2*k + 2) * (2*k + 4) = 48 * n :=
sorry

end product_of_three_consecutive_even_numbers_divisible_by_48_l2560_256097


namespace sequence_sum_formula_l2560_256035

/-- Given a sequence a with a₁ = 1 and Sₙ = n² * aₙ for all positive integers n,
    prove that the sum of the first n terms Sₙ is equal to 2n / (n+1). -/
theorem sequence_sum_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) :=
by sorry

end sequence_sum_formula_l2560_256035


namespace chicken_rabbit_equations_l2560_256022

/-- Represents the "chicken-rabbit in the same cage" problem --/
def chicken_rabbit_problem (x y : ℕ) : Prop :=
  let total_heads : ℕ := 35
  let total_feet : ℕ := 94
  let chicken_feet : ℕ := 2
  let rabbit_feet : ℕ := 4
  (x + y = total_heads) ∧ (chicken_feet * x + rabbit_feet * y = total_feet)

/-- Proves that the system of equations correctly represents the problem --/
theorem chicken_rabbit_equations : 
  ∀ x y : ℕ, chicken_rabbit_problem x y ↔ (x + y = 35 ∧ 2*x + 4*y = 94) := by
  sorry

end chicken_rabbit_equations_l2560_256022


namespace combination_sum_l2560_256098

theorem combination_sum : Nat.choose 99 2 + Nat.choose 99 3 = 161700 := by
  sorry

end combination_sum_l2560_256098


namespace fence_posts_count_l2560_256062

theorem fence_posts_count (length width post_distance : ℕ) 
  (h1 : length = 80)
  (h2 : width = 60)
  (h3 : post_distance = 10) : 
  (2 * (length / post_distance + 1) + 2 * (width / post_distance + 1)) - 4 = 28 := by
  sorry

end fence_posts_count_l2560_256062


namespace tangent_line_and_monotonicity_l2560_256056

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x - 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 9

-- Theorem statement
theorem tangent_line_and_monotonicity 
  (a : ℝ) 
  (h1 : a < 0) 
  (h2 : ∃ x₀, ∀ x, f' a x₀ ≤ f' a x ∧ f' a x₀ = -12) :
  a = -3 ∧ 
  (∀ x₁ x₂, x₁ < x₂ → 
    ((x₂ < -1 → f a x₁ < f a x₂) ∧
     (x₁ > 3 → f a x₁ < f a x₂) ∧
     (-1 < x₁ ∧ x₂ < 3 → f a x₁ > f a x₂))) :=
sorry

end tangent_line_and_monotonicity_l2560_256056


namespace cafeteria_apples_l2560_256086

/-- Calculates the number of apples bought by the cafeteria -/
def apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) : ℕ :=
  final - (initial - used)

/-- Proves that the cafeteria bought 6 apples -/
theorem cafeteria_apples : apples_bought 23 20 9 = 6 := by
  sorry

end cafeteria_apples_l2560_256086


namespace simplify_expression_l2560_256047

variable (x y : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := by
  sorry

end simplify_expression_l2560_256047


namespace orange_stack_sum_l2560_256013

def pyramid_stack (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let layers := min base_width base_length
  List.range layers
    |> List.map (λ i => (base_width - i) * (base_length - i))
    |> List.sum

theorem orange_stack_sum :
  pyramid_stack 6 9 = 155 := by
  sorry

end orange_stack_sum_l2560_256013


namespace intersection_length_l2560_256019

/-- Given a line y = kx - 2 intersecting a parabola y² = 8x at points A and B,
    if the x-coordinate of the midpoint of AB is 2, then the length of AB is 2√15. -/
theorem intersection_length (k : ℝ) (A B : ℝ × ℝ) :
  (∀ x y, y = k * x - 2 → y^2 = 8 * x → (x, y) = A ∨ (x, y) = B) →
  (A.1 + B.1) / 2 = 2 →
  ‖A - B‖ = 2 * Real.sqrt 15 :=
sorry

end intersection_length_l2560_256019


namespace mandy_school_ratio_l2560_256087

theorem mandy_school_ratio : 
  ∀ (researched applied accepted : ℕ),
    researched = 42 →
    accepted = 7 →
    2 * accepted = applied →
    (applied : ℚ) / researched = 1 / 3 :=
by sorry

end mandy_school_ratio_l2560_256087


namespace limit_rational_function_l2560_256001

/-- The limit of (2x³ - 3x² + 5x + 7) / (3x³ + 4x² - x + 2) as x approaches infinity is 2/3 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → 
    |((2 * x^3 - 3 * x^2 + 5 * x + 7) / (3 * x^3 + 4 * x^2 - x + 2)) - 2/3| < ε := by
  sorry

end limit_rational_function_l2560_256001


namespace sum_of_squared_coefficients_is_1080_l2560_256003

/-- The polynomial for which we want to calculate the sum of squared coefficients -/
def p (x : ℝ) : ℝ := 6 * (x^3 + 4*x^2 + 2*x + 3)

/-- The sum of the squares of the coefficients of the polynomial p -/
def sum_of_squared_coefficients : ℝ :=
  let coeffs := [6, 24, 12, 18]
  coeffs.map (λ c => c^2) |>.sum

/-- Theorem stating that the sum of the squares of the coefficients of p is 1080 -/
theorem sum_of_squared_coefficients_is_1080 :
  sum_of_squared_coefficients = 1080 := by
  sorry

end sum_of_squared_coefficients_is_1080_l2560_256003


namespace train_speed_in_km_hr_l2560_256009

-- Define the given parameters
def train_length : ℝ := 50
def platform_length : ℝ := 250
def crossing_time : ℝ := 15

-- Define the conversion factor from m/s to km/hr
def m_s_to_km_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_in_km_hr :
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / crossing_time
  speed_m_s * m_s_to_km_hr = 72 := by
  sorry


end train_speed_in_km_hr_l2560_256009


namespace total_pears_picked_l2560_256069

theorem total_pears_picked (jason_pears keith_pears mike_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12) :
  jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end total_pears_picked_l2560_256069


namespace perfect_square_condition_l2560_256021

theorem perfect_square_condition (a b : ℕ+) (h_b_odd : Odd b.val) 
  (h_int : ∃ k : ℤ, ((a.val + b.val)^2 + 4*a.val : ℤ) = k * (a.val * b.val)) :
  ∃ n : ℕ+, a = n^2 := by
sorry

end perfect_square_condition_l2560_256021


namespace linear_equation_condition_l2560_256042

/-- If (a+1)x + 3y^|a| = 1 is a linear equation in x and y, then a = 1 -/
theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, (a + 1) * x + 3 * y^(|a|) = k * x + m * y + 1) → a = 1 := by
  sorry

end linear_equation_condition_l2560_256042


namespace sum_of_digits_up_to_100000_l2560_256061

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of all numbers from 1 to 100000 -/
theorem sum_of_digits_up_to_100000 : sumOfDigitsUpTo 100000 = 2443446 := by sorry

end sum_of_digits_up_to_100000_l2560_256061


namespace chocolate_chip_calculation_l2560_256082

/-- Represents the number of cups of chocolate chips per batch in the recipe -/
def cups_per_batch : ℝ := 2.0

/-- Represents the number of batches that can be made with the available chocolate chips -/
def number_of_batches : ℝ := 11.5

/-- Calculates the total number of cups of chocolate chips -/
def total_chocolate_chips : ℝ := cups_per_batch * number_of_batches

/-- Proves that the total number of cups of chocolate chips is 23 -/
theorem chocolate_chip_calculation : total_chocolate_chips = 23 := by
  sorry

end chocolate_chip_calculation_l2560_256082


namespace inequality_and_minimum_value_l2560_256016

theorem inequality_and_minimum_value (a b m n : ℝ) (x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0) 
  (hx : 0 < x ∧ x < 1/2) : 
  (m^2 / a + n^2 / b ≥ (m + n)^2 / (a + b)) ∧
  (∃ (min_val : ℝ), min_val = 25 ∧ 
    ∀ y, 0 < y ∧ y < 1/2 → 2/y + 9/(1-2*y) ≥ min_val) ∧
  (2/((1:ℝ)/5) + 9/(1-2*((1:ℝ)/5)) = 25) := by
  sorry

end inequality_and_minimum_value_l2560_256016


namespace max_value_function_l2560_256093

theorem max_value_function (x y : ℝ) :
  (2*x + 3*y + 4) / Real.sqrt (x^2 + 2*y^2 + 1) ≤ Real.sqrt 29 := by
  sorry

end max_value_function_l2560_256093


namespace xy_value_l2560_256060

theorem xy_value (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : (x - y)^2 = 9) : x * y = 3 := by
  sorry

end xy_value_l2560_256060


namespace sum_remainder_mod_9_l2560_256084

theorem sum_remainder_mod_9 : (7150 + 7152 + 7154 + 7156 + 7158) % 9 = 2 := by
  sorry

end sum_remainder_mod_9_l2560_256084


namespace square_of_1023_l2560_256010

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end square_of_1023_l2560_256010


namespace coins_after_five_hours_l2560_256053

/-- The number of coins in Tina's jar after five hours -/
def coins_in_jar (initial_deposit : ℕ) (second_third_deposit : ℕ) (fourth_deposit : ℕ) (withdrawal : ℕ) : ℕ :=
  initial_deposit + 2 * second_third_deposit + fourth_deposit - withdrawal

/-- Theorem stating the number of coins in the jar after five hours -/
theorem coins_after_five_hours :
  coins_in_jar 20 30 40 20 = 100 := by
  sorry

#eval coins_in_jar 20 30 40 20

end coins_after_five_hours_l2560_256053


namespace gaming_chair_price_proof_l2560_256029

/-- The price of a set of toy organizers -/
def toy_organizer_price : ℝ := 78

/-- The number of toy organizer sets ordered -/
def toy_organizer_sets : ℕ := 3

/-- The number of gaming chairs ordered -/
def gaming_chairs : ℕ := 2

/-- The delivery fee percentage -/
def delivery_fee_percent : ℝ := 0.05

/-- The total amount Leon paid -/
def total_paid : ℝ := 420

/-- The price of a gaming chair -/
def gaming_chair_price : ℝ := 83

theorem gaming_chair_price_proof :
  gaming_chair_price * gaming_chairs + toy_organizer_price * toy_organizer_sets +
  (gaming_chair_price * gaming_chairs + toy_organizer_price * toy_organizer_sets) * delivery_fee_percent =
  total_paid := by sorry

end gaming_chair_price_proof_l2560_256029


namespace sum_of_f_values_l2560_256006

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_of_f_values : 
  f 1 + f 2 + f (1/2) + f 3 + f (1/3) + f 4 + f (1/4) = 7/2 := by
  sorry

end sum_of_f_values_l2560_256006


namespace parallelogram_area_theorem_l2560_256067

/-- Given two vectors a and b in a vector space, the area_parallelogram function
    computes the area of the parallelogram generated by these vectors. -/
def area_parallelogram (a b : V) : ℝ := sorry

variable (V : Type) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

/-- The theorem states that if the area of the parallelogram generated by vectors a and b
    is 15, then the area of the parallelogram generated by vectors 3a + 4b and 2a - 6b is 390. -/
theorem parallelogram_area_theorem (h : area_parallelogram a b = 15) :
  area_parallelogram (3 • a + 4 • b) (2 • a - 6 • b) = 390 :=
sorry

end parallelogram_area_theorem_l2560_256067


namespace player_a_winning_strategy_l2560_256045

/-- The game board represented as a 3x3 matrix -/
def GameBoard : Matrix (Fin 3) (Fin 3) ℕ :=
  !![7, 8, 9;
     4, 5, 6;
     1, 2, 3]

/-- Checks if two numbers are in the same row or column on the game board -/
def inSameRowOrCol (a b : ℕ) : Prop :=
  ∃ i j k : Fin 3, (GameBoard i j = a ∧ GameBoard i k = b) ∨
                   (GameBoard j i = a ∧ GameBoard k i = b)

/-- Represents a valid move in the game -/
structure Move where
  number : ℕ
  valid : number ≥ 1 ∧ number ≤ 9

/-- Represents the game state -/
structure GameState where
  moves : List Move
  total : ℕ
  lastMove : Move

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move.number ≠ state.lastMove.number ∧
  inSameRowOrCol move.number state.lastMove.number ∧
  state.total + move.number ≤ 30

/-- Represents a winning strategy for Player A -/
def WinningStrategy :=
  ∃ (firstMove : Move),
    ∀ (b1 : Move),
      ∃ (a2 : Move),
        ∀ (b2 : Move),
          ∃ (a3 : Move),
            (isValidMove ⟨[firstMove], firstMove.number, firstMove⟩ b1 →
             isValidMove ⟨[b1, firstMove], firstMove.number + b1.number, b1⟩ a2 →
             isValidMove ⟨[a2, b1, firstMove], firstMove.number + b1.number + a2.number, a2⟩ b2 →
             isValidMove ⟨[b2, a2, b1, firstMove], firstMove.number + b1.number + a2.number + b2.number, b2⟩ a3) ∧
            (firstMove.number + b1.number + a2.number + b2.number + a3.number = 30)

theorem player_a_winning_strategy : WinningStrategy := sorry

end player_a_winning_strategy_l2560_256045


namespace inequality_system_solution_set_l2560_256004

theorem inequality_system_solution_set :
  ∀ a : ℝ, (2 * a - 3 < 0 ∧ 1 - a < 0) ↔ (1 < a ∧ a < 3/2) := by
  sorry

end inequality_system_solution_set_l2560_256004


namespace drum_fill_time_l2560_256032

/-- The time to fill a cylindrical drum with varying rain rate -/
theorem drum_fill_time (initial_rate : ℝ) (area : ℝ) (depth : ℝ) :
  let rate := fun t : ℝ => initial_rate * t^2
  let volume := area * depth
  let fill_time := (volume * 3 / (5 * initial_rate))^(1/3)
  fill_time^3 = volume * 3 / (5 * initial_rate) :=
by sorry

end drum_fill_time_l2560_256032


namespace average_marathons_rounded_l2560_256076

def marathons : List ℕ := [1, 2, 3, 4, 5]
def members : List ℕ := [6, 5, 3, 2, 3]

def total_marathons : ℕ := (List.zip marathons members).map (λ (m, n) => m * n) |>.sum
def total_members : ℕ := members.sum

def average : ℚ := total_marathons / total_members

theorem average_marathons_rounded :
  (average + 1/2).floor = 3 := by
  sorry

end average_marathons_rounded_l2560_256076


namespace sqrt_product_equality_l2560_256085

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l2560_256085


namespace cos_equality_implies_77_l2560_256002

theorem cos_equality_implies_77 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) 
  (h3 : Real.cos (n * π / 180) = Real.cos (283 * π / 180)) : n = 77 := by
  sorry

end cos_equality_implies_77_l2560_256002


namespace malcolm_primes_l2560_256068

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem malcolm_primes (n : ℕ) :
  n > 0 ∧ is_prime n ∧ is_prime (2 * n - 1) ∧ is_prime (4 * n - 1) → n = 2 ∨ n = 3 := by
  sorry

end malcolm_primes_l2560_256068


namespace function_properties_l2560_256088

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x^2 * x + 2 * b - a^3

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 6, f a b x > 0) →
  (∀ x ∈ Set.Iic (-2 : ℝ) ∪ Set.Ici 6, f a b x < 0) →
  f a b (-2) = 0 →
  f a b 6 = 0 →
  (∃ c d : ℝ, c = -4 ∧ d = -48 ∧ ∀ x, f a b x = c * x^2 + 2 * c * x + d) ∧
  (∀ x ∈ Set.Icc 1 10, f a b x ≤ -20) ∧
  (∀ x ∈ Set.Icc 1 10, f a b x ≥ -192) ∧
  (∃ x ∈ Set.Icc 1 10, f a b x = -20) ∧
  (∃ x ∈ Set.Icc 1 10, f a b x = -192) :=
by
  sorry

end function_properties_l2560_256088


namespace min_values_theorem_l2560_256078

theorem min_values_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : Real.log a + Real.log b = Real.log (a + 9*b)) : 
  (a * b ≥ 36) ∧ ((81 / a^2) + (1 / b^2) ≥ 1/2) ∧ (a + b ≥ 16) := by
  sorry

end min_values_theorem_l2560_256078


namespace mobile_phone_price_mobile_phone_price_is_8000_l2560_256063

theorem mobile_phone_price (refrigerator_price : ℝ) (refrigerator_loss_percent : ℝ) 
  (phone_profit_percent : ℝ) (total_profit : ℝ) : ℝ :=
  let refrigerator_sale_price := refrigerator_price * (1 - refrigerator_loss_percent)
  let phone_price := (total_profit + refrigerator_price - refrigerator_sale_price) / 
    (phone_profit_percent - refrigerator_loss_percent)
  phone_price

-- Proof that the mobile phone price is 8000
theorem mobile_phone_price_is_8000 : 
  mobile_phone_price 15000 0.04 0.11 280 = 8000 := by
  sorry

end mobile_phone_price_mobile_phone_price_is_8000_l2560_256063


namespace all_lines_have_inclination_angle_not_necessarily_slope_l2560_256012

-- Define what a line is (this is a simplified representation)
structure Line where
  -- You might add more properties here in a real implementation
  dummy : Unit

-- Define the concept of an inclination angle
def has_inclination_angle (l : Line) : Prop := sorry

-- Define the concept of a slope
def has_slope (l : Line) : Prop := sorry

-- The theorem to prove
theorem all_lines_have_inclination_angle_not_necessarily_slope :
  (∀ l : Line, has_inclination_angle l) ∧
  (∃ l : Line, ¬ has_slope l) := by sorry

end all_lines_have_inclination_angle_not_necessarily_slope_l2560_256012


namespace bob_journey_distance_l2560_256026

/-- Calculates the total distance traveled given two journey segments -/
def totalDistance (speed1 speed2 time1 time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that Bob's journey results in a total distance of 180 miles -/
theorem bob_journey_distance :
  let speed1 : ℝ := 60
  let speed2 : ℝ := 45
  let time1 : ℝ := 1.5
  let time2 : ℝ := 2
  totalDistance speed1 speed2 time1 time2 = 180 := by
  sorry

#eval totalDistance 60 45 1.5 2

end bob_journey_distance_l2560_256026


namespace points_per_player_l2560_256017

theorem points_per_player (total_points : ℕ) (num_players : ℕ) 
  (h1 : total_points = 18) (h2 : num_players = 9) :
  total_points / num_players = 2 := by
  sorry

end points_per_player_l2560_256017


namespace seeds_per_flower_bed_l2560_256007

theorem seeds_per_flower_bed 
  (total_seeds : ℕ) 
  (num_flower_beds : ℕ) 
  (h1 : total_seeds = 54) 
  (h2 : num_flower_beds = 9) 
  (h3 : num_flower_beds ≠ 0) : 
  total_seeds / num_flower_beds = 6 := by
sorry

end seeds_per_flower_bed_l2560_256007


namespace max_operation_result_l2560_256038

def operation (n : ℕ) : ℚ :=
  2 * (2/3 * (300 - n))

theorem max_operation_result :
  (∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → operation n ≤ 1160/3) ∧
  (∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ operation n = 1160/3) :=
sorry

end max_operation_result_l2560_256038


namespace shifted_quadratic_sum_l2560_256033

/-- Given a quadratic function f(x) = 3x^2 + 2x + 5, shifting it 5 units to the left
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that a + b + c = 125. -/
theorem shifted_quadratic_sum (a b c : ℝ) : 
  (∀ x, 3 * (x + 5)^2 + 2 * (x + 5) + 5 = a * x^2 + b * x + c) →
  a + b + c = 125 := by
sorry

end shifted_quadratic_sum_l2560_256033


namespace xiaoming_mother_retirement_year_l2560_256090

/-- Calculates the retirement year based on the given retirement plan --/
def calculate_retirement_year (birth_year : ℕ) : ℕ :=
  let original_retirement_year := birth_year + 55
  if original_retirement_year ≥ 2018 ∧ original_retirement_year < 2021
  then original_retirement_year + 1
  else original_retirement_year

/-- Theorem stating that Xiaoming's mother's retirement year is 2020 --/
theorem xiaoming_mother_retirement_year :
  calculate_retirement_year 1964 = 2020 :=
by sorry

end xiaoming_mother_retirement_year_l2560_256090


namespace power_function_through_point_l2560_256008

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 32 →
  ∀ x : ℝ, f x = x^5 := by
sorry

end power_function_through_point_l2560_256008


namespace subset_relation_l2560_256015

theorem subset_relation (x y : ℝ) :
  (abs x + abs y < 1) →
  (Real.sqrt ((x - 1/2)^2 + (y + 1/2)^2) + Real.sqrt ((x + 1/2)^2 + (y - 1/2)^2) < 2 * Real.sqrt 2) :=
by sorry

end subset_relation_l2560_256015


namespace sum_g_one_neg_one_l2560_256050

/-- Given two functions f and g defined on real numbers satisfying certain conditions,
    prove that g(1) + g(-1) = -1. -/
theorem sum_g_one_neg_one (f g : ℝ → ℝ) 
    (h1 : ∀ x y, f (x - y) = f x * g y - g x * f y)
    (h2 : f (-2) = f 1)
    (h3 : f 1 ≠ 0) : 
  g 1 + g (-1) = -1 := by sorry

end sum_g_one_neg_one_l2560_256050


namespace distance_to_origin_of_complex_number_l2560_256046

theorem distance_to_origin_of_complex_number : 
  let i : ℂ := Complex.I
  let z : ℂ := i / (i + 1)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end distance_to_origin_of_complex_number_l2560_256046


namespace antonio_winning_strategy_l2560_256066

/-- Represents the game state with two piles of chips -/
structure GameState where
  m : ℕ
  n : ℕ

/-- Defines the possible moves in the game -/
inductive Move
  | TakeOne : Bool → Move  -- True for first pile, False for second
  | TakeBoth : Move
  | Transfer : Bool → Move  -- True for first to second, False for second to first

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeOne first => 
      if first then ⟨state.m - 1, state.n⟩ else ⟨state.m, state.n - 1⟩
  | Move.TakeBoth => ⟨state.m - 1, state.n - 1⟩
  | Move.Transfer first => 
      if first then ⟨state.m - 1, state.n + 1⟩ else ⟨state.m + 1, state.n - 1⟩

/-- Determines if a move is valid for a given state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.TakeOne first => if first then state.m > 0 else state.n > 0
  | Move.TakeBoth => state.m > 0 ∧ state.n > 0
  | Move.Transfer first => if first then state.m > 0 else state.n > 0

/-- Determines if the game is over (no valid moves) -/
def isGameOver (state : GameState) : Bool :=
  state.m = 0 ∧ state.n = 0

/-- Theorem: The first player (Antonio) has a winning strategy if and only if at least one of m or n is odd -/
theorem antonio_winning_strategy (initialState : GameState) :
  (initialState.m % 2 = 1 ∨ initialState.n % 2 = 1) ↔ 
  ∃ (strategy : GameState → Move), 
    (∀ (state : GameState), 
      ¬isGameOver state → 
      isValidMove state (strategy state) ∧ 
      ¬∃ (counterStrategy : GameState → Move), 
        (∀ (state : GameState), 
          ¬isGameOver state → 
          isValidMove state (counterStrategy state) ∧ 
          isGameOver (applyMove (applyMove state (strategy state)) (counterStrategy (applyMove state (strategy state)))))) :=
sorry

end antonio_winning_strategy_l2560_256066


namespace least_possible_value_z_minus_x_l2560_256065

theorem least_possible_value_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y ∧ y < z) 
  (h2 : y - x > 11) 
  (h3 : Even x) 
  (h4 : Odd y ∧ Odd z) :
  ∀ w, w = z - x → w ≥ 15 ∧ ∃ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧ 
    y' - x' > 11 ∧ 
    Even x' ∧ Odd y' ∧ Odd z' ∧ 
    z' - x' = 15 :=
by sorry

end least_possible_value_z_minus_x_l2560_256065


namespace arithmetic_sequence_property_l2560_256083

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  isArithmeticSequence a →
  a 1 + 2 * a 8 + a 15 = 96 →
  2 * a 9 - a 10 = 24 := by
sorry

end arithmetic_sequence_property_l2560_256083


namespace rectangular_plot_length_difference_l2560_256020

theorem rectangular_plot_length_difference (length breadth : ℝ) : 
  length = 62 →
  length > breadth →
  2 * (length + breadth) * 26.5 = 5300 →
  length - breadth = 24 := by
sorry

end rectangular_plot_length_difference_l2560_256020


namespace largest_triangle_perimeter_l2560_256011

theorem largest_triangle_perimeter : 
  ∀ y : ℤ, 
  (y > 0) → 
  (7 + 9 > y) → 
  (7 + y > 9) → 
  (9 + y > 7) → 
  (∀ z : ℤ, (z > 0) → (7 + 9 > z) → (7 + z > 9) → (9 + z > 7) → (7 + 9 + y ≥ 7 + 9 + z)) →
  (7 + 9 + y = 31) :=
by sorry

end largest_triangle_perimeter_l2560_256011


namespace gcd_299_667_l2560_256074

theorem gcd_299_667 : Nat.gcd 299 667 = 23 := by
  sorry

end gcd_299_667_l2560_256074


namespace wall_volume_l2560_256059

/-- Proves that the volume of a rectangular wall with given dimensions is 6804 cubic units -/
theorem wall_volume : 
  ∀ (width height length : ℕ),
  width = 3 →
  height = 6 * width →
  length = 7 * height →
  width * height * length = 6804 := by
  sorry

end wall_volume_l2560_256059


namespace julian_celine_ratio_is_one_l2560_256096

/-- The number of erasers collected by Celine -/
def celine_erasers : ℕ := 10

/-- The number of erasers collected by Julian -/
def julian_erasers : ℕ := celine_erasers

/-- The total number of erasers collected -/
def total_erasers : ℕ := 35

/-- The ratio of erasers collected by Julian to Celine -/
def julian_to_celine_ratio : ℚ := julian_erasers / celine_erasers

theorem julian_celine_ratio_is_one : julian_to_celine_ratio = 1 := by
  sorry

end julian_celine_ratio_is_one_l2560_256096


namespace triangle_sides_from_heights_l2560_256064

theorem triangle_sides_from_heights (h_a h_b h_c A : ℝ) (h_positive : h_a > 0 ∧ h_b > 0 ∧ h_c > 0) (h_area : A > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    A = (1/2) * a * h_a ∧
    A = (1/2) * b * h_b ∧
    A = (1/2) * c * h_c :=
sorry


end triangle_sides_from_heights_l2560_256064


namespace raj_earns_more_by_200_l2560_256031

/-- Represents the dimensions of a rectangular plot of land -/
structure Plot where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular plot -/
def area (p : Plot) : ℕ := p.length * p.width

/-- Calculates the earnings from selling a plot given a price per square foot -/
def earnings (p : Plot) (price_per_sqft : ℕ) : ℕ := area p * price_per_sqft

/-- The difference in earnings between two plots -/
def earnings_difference (p1 p2 : Plot) (price_per_sqft : ℕ) : ℤ :=
  (earnings p1 price_per_sqft : ℤ) - (earnings p2 price_per_sqft : ℤ)

theorem raj_earns_more_by_200 :
  let raj_plot : Plot := ⟨30, 50⟩
  let lena_plot : Plot := ⟨40, 35⟩
  let price_per_sqft : ℕ := 2
  earnings_difference raj_plot lena_plot price_per_sqft = 200 :=
sorry

end raj_earns_more_by_200_l2560_256031


namespace all_lines_pass_through_point_l2560_256023

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Defines a geometric progression for three real numbers -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

/-- The theorem stating that all lines with a, b, c in geometric progression pass through (0, 1) -/
theorem all_lines_pass_through_point :
  ∀ l : Line, isGeometricProgression l.a l.b l.c → l.contains 0 1 :=
sorry

end all_lines_pass_through_point_l2560_256023


namespace sector_area_l2560_256075

theorem sector_area (circle_area : ℝ) (sector_angle : ℝ) : 
  circle_area = 9 * Real.pi →
  sector_angle = 120 →
  (sector_angle / 360) * circle_area = 3 * Real.pi :=
by sorry

end sector_area_l2560_256075


namespace min_dot_product_on_hyperbola_l2560_256044

/-- The minimum dot product of two points on the hyperbola x² - y² = 2 -/
theorem min_dot_product_on_hyperbola :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  x₁ > 0 → x₂ > 0 →
  x₁^2 - y₁^2 = 2 →
  x₂^2 - y₂^2 = 2 →
  x₁ * x₂ + y₁ * y₂ ≥ 2 ∧
  ∃ (x₁' y₁' x₂' y₂' : ℝ),
    x₁' > 0 ∧ x₂' > 0 ∧
    x₁'^2 - y₁'^2 = 2 ∧
    x₂'^2 - y₂'^2 = 2 ∧
    x₁' * x₂' + y₁' * y₂' = 2 :=
by sorry

end min_dot_product_on_hyperbola_l2560_256044


namespace linear_equations_compatibility_l2560_256049

theorem linear_equations_compatibility (a b c d : ℝ) :
  (∃ x : ℝ, a * x + b = 0 ∧ c * x + d = 0) ↔ a * d - b * c = 0 := by
  sorry

end linear_equations_compatibility_l2560_256049


namespace count_eight_digit_integers_l2560_256092

/-- The number of different 8-digit positive integers -/
def eight_digit_integers : ℕ :=
  9 * (10 ^ 7)

/-- Theorem: The number of different 8-digit positive integers is 90,000,000 -/
theorem count_eight_digit_integers :
  eight_digit_integers = 90000000 := by
  sorry

end count_eight_digit_integers_l2560_256092


namespace right_triangle_leg_sum_l2560_256018

theorem right_triangle_leg_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive lengths
  b = a + 2 →              -- one leg is 2 units longer
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  c = 29 →                 -- hypotenuse is 29 units
  a + b = 40 :=            -- sum of legs is 40
by sorry

end right_triangle_leg_sum_l2560_256018


namespace shoe_price_calculation_shoe_price_proof_l2560_256014

theorem shoe_price_calculation (initial_price : ℝ) 
  (increase_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let price_after_increase := initial_price * (1 + increase_percentage)
  let final_price := price_after_increase * (1 - discount_percentage)
  final_price

theorem shoe_price_proof :
  shoe_price_calculation 50 0.2 0.15 = 51 := by
  sorry

end shoe_price_calculation_shoe_price_proof_l2560_256014


namespace max_principals_is_four_l2560_256025

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 15

/-- Represents the duration of each principal's term in years -/
def term_duration : ℕ := 4

/-- Calculates the maximum number of principals that can serve during the given period -/
def max_principals : ℕ := (period_duration - 1) / term_duration + 1

/-- Theorem stating that the maximum number of principals is 4 -/
theorem max_principals_is_four : max_principals = 4 := by
  sorry

end max_principals_is_four_l2560_256025


namespace math_problem_solutions_l2560_256037

theorem math_problem_solutions :
  (∃ (x : ℝ), x = Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 ∧ x = 4 + Real.sqrt 6) ∧
  (∃ (y : ℝ), y = (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 ∧ y = 1) :=
by sorry

end math_problem_solutions_l2560_256037


namespace average_of_abc_l2560_256052

theorem average_of_abc (A B C : ℚ) 
  (eq1 : 2002 * C + 4004 * A = 8008)
  (eq2 : 3003 * B - 5005 * A = 7007) : 
  (A + B + C) / 3 = 22 / 9 := by sorry

end average_of_abc_l2560_256052


namespace kennel_arrangement_l2560_256095

/-- The number of ways to arrange animals in cages -/
def arrange_animals (num_chickens num_dogs num_cats : ℕ) : ℕ :=
  (Nat.factorial 3) * 
  (Nat.factorial num_chickens) * 
  (Nat.factorial num_dogs) * 
  (Nat.factorial num_cats)

/-- Theorem: The number of ways to arrange 3 chickens, 3 dogs, and 4 cats
    in a row of 10 cages, with animals of each type in adjacent cages,
    is 5184 -/
theorem kennel_arrangement : arrange_animals 3 3 4 = 5184 := by
  sorry

end kennel_arrangement_l2560_256095


namespace surface_area_cube_with_corners_removed_l2560_256077

/-- Represents the dimensions of a cube in centimeters -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube with given dimensions -/
def cubeSurfaceArea (d : CubeDimensions) : ℝ :=
  6 * d.length * d.width

/-- Calculates the surface area of a cube with corners removed -/
def cubeWithCornersRemovedSurfaceArea (originalCube : CubeDimensions) (removedCorner : CubeDimensions) : ℝ :=
  cubeSurfaceArea originalCube

/-- Theorem stating that the surface area of a 4x4x4 cube with 1x1x1 corners removed is 96 cm² -/
theorem surface_area_cube_with_corners_removed :
  let originalCube : CubeDimensions := ⟨4, 4, 4⟩
  let removedCorner : CubeDimensions := ⟨1, 1, 1⟩
  cubeWithCornersRemovedSurfaceArea originalCube removedCorner = 96 := by
  sorry

end surface_area_cube_with_corners_removed_l2560_256077


namespace correct_number_of_choices_l2560_256054

/-- Represents a team in the club -/
inductive Team
| A
| B

/-- Represents the gender of a club member -/
inductive Gender
| Boy
| Girl

/-- Represents the composition of a team -/
structure TeamComposition :=
  (boys : ℕ)
  (girls : ℕ)

/-- The total number of members in the club -/
def totalMembers : ℕ := 24

/-- The number of boys in the club -/
def totalBoys : ℕ := 14

/-- The number of girls in the club -/
def totalGirls : ℕ := 10

/-- The composition of Team A -/
def teamA : TeamComposition := ⟨8, 6⟩

/-- The composition of Team B -/
def teamB : TeamComposition := ⟨6, 4⟩

/-- Returns the number of ways to choose a president and vice-president -/
def chooseLeaders : ℕ := sorry

/-- Theorem stating that the number of ways to choose a president and vice-president
    of different genders and from different teams is 136 -/
theorem correct_number_of_choices :
  chooseLeaders = 136 := by sorry

end correct_number_of_choices_l2560_256054


namespace typing_job_solution_l2560_256057

/-- Represents the time taken by two typists to complete a typing job -/
structure TypingJob where
  combined_time : ℝ  -- Time taken when working together
  sequential_time : ℝ  -- Time taken when working sequentially (half each)
  first_typist_time : ℝ  -- Time for first typist to complete job alone
  second_typist_time : ℝ  -- Time for second typist to complete job alone

/-- Theorem stating the solution to the typing job problem -/
theorem typing_job_solution (job : TypingJob) 
  (h1 : job.combined_time = 12)
  (h2 : job.sequential_time = 25)
  (h3 : job.first_typist_time + job.second_typist_time = 50)
  (h4 : job.first_typist_time * job.second_typist_time = 600) :
  job.first_typist_time = 20 ∧ job.second_typist_time = 30 := by
  sorry

#check typing_job_solution

end typing_job_solution_l2560_256057


namespace arithmetic_sequence_2017_l2560_256036

/-- An arithmetic sequence satisfying the given condition -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a n + 2 * a (n + 1) + 3 * a (n + 2) = 6 * n + 22

/-- The 2017th term of the arithmetic sequence is 6058/3 -/
theorem arithmetic_sequence_2017 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  a 2017 = 6058 / 3 := by
  sorry

end arithmetic_sequence_2017_l2560_256036


namespace convex_polygon_30_sides_diagonals_l2560_256024

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end convex_polygon_30_sides_diagonals_l2560_256024


namespace inverse_proportion_ratio_l2560_256055

/-- Given that x is inversely proportional to y, this theorem proves that
    if x₁/x₂ = 3/4, then y₁/y₂ = 4/3, where y₁ and y₂ are the corresponding y values. -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
    (h_prop : ∃ k : ℝ, ∀ x y, x * y = k) (h_ratio : x₁ / x₂ = 3 / 4) :
    y₁ / y₂ = 4 / 3 := by
  sorry

end inverse_proportion_ratio_l2560_256055


namespace coin_equality_l2560_256070

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Theorem stating that if 25 quarters and 15 dimes equal 15 quarters and n nickels, then n = 80 -/
theorem coin_equality (n : ℕ) : 
  25 * quarter_value + 15 * dime_value = 15 * quarter_value + n * nickel_value → n = 80 := by
  sorry


end coin_equality_l2560_256070


namespace midpoint_exists_but_no_centroid_l2560_256039

/-- A triangle in 2D space -/
structure Triangle :=
  (v1 v2 v3 : ℝ × ℝ)

/-- Check if a point is inside a triangle -/
def isInsideTriangle (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is on the perimeter of a triangle -/
def isOnPerimeter (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (a b m : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is the centroid of a triangle -/
def isCentroid (t : Triangle) (c : ℝ × ℝ) : Prop :=
  sorry

theorem midpoint_exists_but_no_centroid (t : Triangle) (p : ℝ × ℝ) 
  (h : isInsideTriangle t p) :
  (∃ a b : ℝ × ℝ, isOnPerimeter t a ∧ isOnPerimeter t b ∧ isMidpoint a b p) ∧
  (¬ ∃ a b c : ℝ × ℝ, isOnPerimeter t a ∧ isOnPerimeter t b ∧ isOnPerimeter t c ∧
                      isCentroid (Triangle.mk a b c) p) :=
by sorry

end midpoint_exists_but_no_centroid_l2560_256039


namespace final_painting_width_l2560_256058

theorem final_painting_width :
  let total_paintings : ℕ := 5
  let total_area : ℝ := 200
  let small_painting_count : ℕ := 3
  let small_painting_side : ℝ := 5
  let large_painting_width : ℝ := 10
  let large_painting_height : ℝ := 8
  let final_painting_height : ℝ := 5

  let small_paintings_area : ℝ := small_painting_count * small_painting_side * small_painting_side
  let large_painting_area : ℝ := large_painting_width * large_painting_height
  let known_paintings_area : ℝ := small_paintings_area + large_painting_area
  let final_painting_area : ℝ := total_area - known_paintings_area
  let final_painting_width : ℝ := final_painting_area / final_painting_height

  final_painting_width = 9 :=
by sorry

end final_painting_width_l2560_256058


namespace triangle_properties_l2560_256080

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C, 
    prove the following properties based on given conditions. -/
theorem triangle_properties (a b c A B C : ℝ) (h1 : a * Real.cos B = 4) 
    (h2 : b * Real.sin A = 3) (h3 : (1/2) * a * c * Real.sin B = 9) :
    Real.tan B = 3/4 ∧ a = 5 ∧ a + b + c = 11 + Real.sqrt 13 := by
  sorry

end triangle_properties_l2560_256080
