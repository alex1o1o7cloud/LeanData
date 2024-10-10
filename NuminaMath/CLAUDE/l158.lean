import Mathlib

namespace kickball_players_l158_15865

theorem kickball_players (wednesday : ℕ) (thursday : ℕ) (difference : ℕ) : 
  wednesday = 37 →
  difference = 9 →
  thursday = wednesday - difference →
  wednesday + thursday = 65 := by
sorry

end kickball_players_l158_15865


namespace square_root_equation_l158_15864

theorem square_root_equation : Real.sqrt 1936 / 11 = 4 := by
  sorry

end square_root_equation_l158_15864


namespace square_land_area_l158_15831

/-- The area of a square land plot with side length 20 units is 400 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 20) : side_length ^ 2 = 400 := by
  sorry

end square_land_area_l158_15831


namespace solution_set_abs_inequality_l158_15890

theorem solution_set_abs_inequality (x : ℝ) :
  (|x - 3| < 2) ↔ (1 < x ∧ x < 5) := by
  sorry

end solution_set_abs_inequality_l158_15890


namespace partnership_profit_calculation_l158_15852

/-- Represents a business partnership --/
structure Partnership where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  investment_D : ℕ
  profit_ratio_A : ℕ
  profit_ratio_B : ℕ
  profit_ratio_C : ℕ
  profit_ratio_D : ℕ
  C_profit_share : ℕ

/-- Calculates the total profit of a partnership --/
def calculate_total_profit (p : Partnership) : ℕ :=
  let x := p.C_profit_share / p.profit_ratio_C
  x * (p.profit_ratio_A + p.profit_ratio_B + p.profit_ratio_C + p.profit_ratio_D)

/-- Theorem stating that for the given partnership, the total profit is 144000 --/
theorem partnership_profit_calculation (p : Partnership)
  (h1 : p.investment_A = 27000)
  (h2 : p.investment_B = 72000)
  (h3 : p.investment_C = 81000)
  (h4 : p.investment_D = 63000)
  (h5 : p.profit_ratio_A = 2)
  (h6 : p.profit_ratio_B = 3)
  (h7 : p.profit_ratio_C = 4)
  (h8 : p.profit_ratio_D = 3)
  (h9 : p.C_profit_share = 48000) :
  calculate_total_profit p = 144000 := by
  sorry

end partnership_profit_calculation_l158_15852


namespace integer_solution_congruence_l158_15805

theorem integer_solution_congruence (x y z : ℤ) 
  (eq1 : x - 3*y + 2*z = 1)
  (eq2 : 2*x + y - 5*z = 7) :
  z ≡ 1 [ZMOD 7] :=
sorry

end integer_solution_congruence_l158_15805


namespace solve_baseball_card_problem_l158_15800

def baseball_card_problem (initial_cards : ℕ) (final_cards : ℕ) : Prop :=
  ∃ (cards_to_peter : ℕ),
    let cards_after_maria := initial_cards - (initial_cards + 1) / 2
    let cards_before_paul := cards_after_maria - cards_to_peter
    3 * cards_before_paul = final_cards ∧
    cards_to_peter = 1

theorem solve_baseball_card_problem :
  baseball_card_problem 15 18 :=
sorry

end solve_baseball_card_problem_l158_15800


namespace ferry_river_crossing_l158_15881

/-- The width of a river crossed by two ferries --/
def river_width : ℝ := 1280

/-- The distance from the nearest shore where the ferries first meet --/
def first_meeting_distance : ℝ := 720

/-- The distance from the other shore where the ferries meet on the return trip --/
def second_meeting_distance : ℝ := 400

/-- Theorem stating that the width of the river is 1280 meters given the conditions --/
theorem ferry_river_crossing :
  let w := river_width
  let d1 := first_meeting_distance
  let d2 := second_meeting_distance
  (d1 + (w - d1) = w) ∧
  (3 * w = 2 * w + 2 * d1) ∧
  (3 * d1 = 2 * w - d2) →
  w = 1280 := by sorry


end ferry_river_crossing_l158_15881


namespace game_probability_l158_15822

theorem game_probability (n : ℕ) (p_alex p_mel p_chelsea : ℝ) : 
  n = 8 →
  p_alex = 1/2 →
  p_mel = 3 * p_chelsea →
  p_alex + p_mel + p_chelsea = 1 →
  (n.choose 4 * n.choose 3 * n.choose 1) * p_alex^4 * p_mel^3 * p_chelsea = 945/8192 :=
by sorry

end game_probability_l158_15822


namespace correct_celsius_to_fahrenheit_conversion_l158_15834

/-- Conversion function from Celsius to Fahrenheit -/
def celsiusToFahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating the correct conversion from Celsius to Fahrenheit -/
theorem correct_celsius_to_fahrenheit_conversion (c : ℝ) : 
  celsiusToFahrenheit c = 1.8 * c + 32 := by
  sorry

end correct_celsius_to_fahrenheit_conversion_l158_15834


namespace root_equation_a_value_l158_15854

theorem root_equation_a_value 
  (x₁ x₂ x₃ a b : ℚ) : 
  x₁ = -3 - 5 * Real.sqrt 3 → 
  x₂ = -3 + 5 * Real.sqrt 3 → 
  x₃ = 15 / 11 → 
  x₁ * x₂ * x₃ = -90 → 
  x₁^3 + a*x₁^2 + b*x₁ + 90 = 0 → 
  a = -15 / 11 := by
sorry

end root_equation_a_value_l158_15854


namespace passing_percentage_is_33_percent_l158_15894

/-- The passing percentage for an exam -/
def passing_percentage (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ) : ℚ :=
  ((marks_obtained + marks_failed_by : ℚ) / max_marks) * 100

/-- Theorem: The passing percentage is 33% given the problem conditions -/
theorem passing_percentage_is_33_percent : 
  passing_percentage 59 40 300 = 33 := by sorry

end passing_percentage_is_33_percent_l158_15894


namespace quadratic_equation_set_l158_15835

theorem quadratic_equation_set (a : ℝ) : 
  (∃! x, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9/8) := by
  sorry

end quadratic_equation_set_l158_15835


namespace product_19_reciprocal_squares_sum_l158_15812

theorem product_19_reciprocal_squares_sum :
  ∀ a b : ℕ+, a * b = 19 → (1 : ℚ) / a^2 + (1 : ℚ) / b^2 = 362 / 361 := by
  sorry

end product_19_reciprocal_squares_sum_l158_15812


namespace max_knights_count_l158_15892

/-- Represents the type of islander: Knight or Liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the statement made by an islander -/
inductive Statement
  | BothNeighborsLiars
  | OneNeighborLiar

/-- Configuration of islanders around the table -/
structure IslanderConfig where
  total : Nat
  half_both_liars : Nat
  half_one_liar : Nat
  knight_count : Nat

/-- Checks if the given configuration is valid -/
def is_valid_config (config : IslanderConfig) : Prop :=
  config.total = 100 ∧
  config.half_both_liars = 50 ∧
  config.half_one_liar = 50 ∧
  config.knight_count ≤ config.total

/-- Theorem stating the maximum number of knights possible -/
theorem max_knights_count (config : IslanderConfig) 
  (h_valid : is_valid_config config) : 
  config.knight_count ≤ 67 :=
sorry

end max_knights_count_l158_15892


namespace parabola_m_values_l158_15832

-- Define the parabola function
def parabola (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

-- State the theorem
theorem parabola_m_values (a h k m : ℝ) :
  (parabola a h k (-1) = 0) →
  (parabola a h k 5 = 0) →
  (a * (4 - h + m)^2 + k = 0) →
  (m = -5 ∨ m = 1) :=
by sorry

end parabola_m_values_l158_15832


namespace remainder_67_power_67_plus_67_mod_68_l158_15861

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end remainder_67_power_67_plus_67_mod_68_l158_15861


namespace extended_quadrilateral_area_l158_15867

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- Original quadrilateral
  area : ℝ
  -- Lengths of sides
  wz : ℝ
  zx : ℝ
  xy : ℝ
  yw : ℝ
  -- Conditions for extended sides
  wz_extended : ℝ
  zx_extended : ℝ
  xy_extended : ℝ
  yw_extended : ℝ
  -- Conditions for double length
  wz_double : wz_extended = 2 * wz
  zx_double : zx_extended = 2 * zx
  xy_double : xy_extended = 2 * xy
  yw_double : yw_extended = 2 * yw

/-- Theorem stating the relationship between areas of original and extended quadrilaterals -/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral) : 
  ∃ (extended_area : ℝ), extended_area = 9 * q.area := by
  sorry

end extended_quadrilateral_area_l158_15867


namespace coffee_machine_discount_l158_15897

def coffee_machine_problem (original_price : ℝ) (home_cost : ℝ) (previous_coffees : ℕ) (previous_price : ℝ) (payoff_days : ℕ) : Prop :=
  let previous_daily_cost := previous_coffees * previous_price
  let daily_savings := previous_daily_cost - home_cost
  let total_savings := daily_savings * payoff_days
  let discount := original_price - total_savings
  original_price = 200 ∧ 
  home_cost = 3 ∧ 
  previous_coffees = 2 ∧ 
  previous_price = 4 ∧ 
  payoff_days = 36 →
  discount = 20

theorem coffee_machine_discount :
  coffee_machine_problem 200 3 2 4 36 :=
by sorry

end coffee_machine_discount_l158_15897


namespace e_pow_pi_gt_pi_pow_e_l158_15878

/-- Prove that e^π > π^e, given that π > e -/
theorem e_pow_pi_gt_pi_pow_e : Real.exp π > π ^ Real.exp 1 := by
  sorry

end e_pow_pi_gt_pi_pow_e_l158_15878


namespace least_n_satisfying_inequality_l158_15891

theorem least_n_satisfying_inequality : ∃ n : ℕ, 
  (∀ k : ℕ, k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧
  n = 4 :=
by
  sorry

end least_n_satisfying_inequality_l158_15891


namespace video_game_lives_l158_15886

theorem video_game_lives (initial_lives won_lives gained_lives : Float) 
  (h1 : initial_lives = 43.0)
  (h2 : won_lives = 14.0)
  (h3 : gained_lives = 27.0) :
  initial_lives + won_lives + gained_lives = 84.0 := by
  sorry

end video_game_lives_l158_15886


namespace anya_pancakes_l158_15829

theorem anya_pancakes (x : ℝ) (x_pos : x > 0) : 
  let flipped := x * (2/3)
  let not_burnt := flipped * 0.6
  let not_dropped := not_burnt * 0.8
  not_dropped / x = 0.32 := by sorry

end anya_pancakes_l158_15829


namespace quadratic_max_value_l158_15801

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, x^2 + 2*x + a ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, x^2 + 2*x + a = 4) →
  a = -4 := by
sorry

end quadratic_max_value_l158_15801


namespace yellow_crayon_count_prove_yellow_crayons_l158_15888

/-- Proves that the number of yellow crayons is 32 given the conditions of the problem. -/
theorem yellow_crayon_count : ℕ → ℕ → ℕ → Prop :=
  fun red blue yellow =>
    (red = 14) →
    (blue = red + 5) →
    (yellow = 2 * blue - 6) →
    (yellow = 32)

/-- The main theorem that proves the number of yellow crayons. -/
theorem prove_yellow_crayons :
  ∃ (red blue yellow : ℕ),
    yellow_crayon_count red blue yellow :=
by
  sorry

end yellow_crayon_count_prove_yellow_crayons_l158_15888


namespace range_of_a_l158_15862

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ici 1 ∧ |x - a| + x - 4 ≤ 0) → a ∈ Set.Icc (-2) 4 := by
  sorry

end range_of_a_l158_15862


namespace base_four_for_64_l158_15856

theorem base_four_for_64 : ∃! b : ℕ, b > 1 ∧ b ^ 3 ≤ 64 ∧ 64 < b ^ 4 := by
  sorry

end base_four_for_64_l158_15856


namespace compare_expressions_l158_15821

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) > (Real.sqrt (a*b) + 1/Real.sqrt (a*b))^2 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) > ((a+b)/2 + 2/(a+b))^2 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (a + 1/a) * (b + 1/b) < ((a+b)/2 + 2/(a+b))^2 :=
by sorry

end compare_expressions_l158_15821


namespace shirts_per_minute_l158_15806

/-- A machine that makes shirts -/
structure ShirtMachine where
  yesterday_production : ℕ
  today_production : ℕ
  total_working_time : ℕ

/-- Theorem: The machine can make 8 shirts per minute -/
theorem shirts_per_minute (m : ShirtMachine)
  (h1 : m.yesterday_production = 13)
  (h2 : m.today_production = 3)
  (h3 : m.total_working_time = 2) :
  (m.yesterday_production + m.today_production) / m.total_working_time = 8 := by
  sorry

end shirts_per_minute_l158_15806


namespace min_value_abc_l158_15883

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/1152 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^4 * b₀^3 * c₀^2 = 1/1152 :=
by sorry

end min_value_abc_l158_15883


namespace largest_divisor_of_expression_l158_15866

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (15 * x + 3) * (15 * x + 9) * (10 * x + 10) = 1920 * k) ∧
  (∀ (n : ℤ), n > 1920 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (m : ℤ), (15 * y + 3) * (15 * y + 9) * (10 * y + 10) = n * m)) :=
sorry

end largest_divisor_of_expression_l158_15866


namespace combined_mpg_l158_15813

/-- Combined rate of miles per gallon for two cars -/
theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℚ) :
  ray_mpg = 50 →
  tom_mpg = 25 →
  ray_miles = 100 →
  tom_miles = 200 →
  (ray_miles + tom_miles) / (ray_miles / ray_mpg + tom_miles / tom_mpg) = 30 := by
  sorry


end combined_mpg_l158_15813


namespace rhombus_perimeter_l158_15850

/-- The perimeter of a rhombus with diagonals of 12 inches and 30 inches is 4√261 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 4 * Real.sqrt 261 := by
  sorry

end rhombus_perimeter_l158_15850


namespace petya_vasya_equal_numbers_possible_l158_15871

theorem petya_vasya_equal_numbers_possible :
  ∃ (n : ℤ) (k : ℕ), n ≠ 0 ∧ 
  (n + 10 * k) * 2014 = (n - 10 * k) / 2014 := by
  sorry

end petya_vasya_equal_numbers_possible_l158_15871


namespace min_value_reciprocal_sum_l158_15898

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 1) :
  1/x + 3/y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 1 ∧ 1/x₀ + 3/y₀ = 16 :=
sorry

end min_value_reciprocal_sum_l158_15898


namespace largest_area_quadrilateral_in_sector_l158_15870

/-- The largest area of a right-angled quadrilateral inscribed in a circular sector -/
theorem largest_area_quadrilateral_in_sector (r : ℝ) (h : r > 0) :
  let max_area (α : ℝ) := 
    (2 * r^2 * Real.sin (α/2) * Real.sin (α/2)) / Real.sin α
  (max_area (2*π/3) = (r^2 * Real.sqrt 3) / 3) ∧ 
  (max_area (4*π/3) = r^2 * Real.sqrt 3) := by
  sorry

end largest_area_quadrilateral_in_sector_l158_15870


namespace min_value_sum_l158_15824

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) :
  x + y ≥ 4 + 2 * Real.sqrt 3 := by
sorry

end min_value_sum_l158_15824


namespace p_necessary_not_sufficient_for_q_l158_15817

def p (x : ℝ) : Prop := |2*x - 3| < 1

def q (x : ℝ) : Prop := x * (x - 3) < 0

theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end p_necessary_not_sufficient_for_q_l158_15817


namespace quadratic_value_l158_15802

/-- A quadratic function with specific properties -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 2)^2 + 7

theorem quadratic_value (a : ℝ) :
  (∀ x, f a x ≤ 7) →  -- Maximum value condition
  (f a 2 = 7) →       -- Maximum occurs at x = 2
  (f a 0 = -7) →      -- Passes through (0, -7)
  (a < 0) →           -- Implied by maximum condition
  (f a 5 = -24.5) :=  -- The value at x = 5
by sorry

end quadratic_value_l158_15802


namespace grocery_cost_is_correct_l158_15833

def grocery_cost (egg_quantity : ℕ) (egg_price : ℚ) (milk_quantity : ℕ) (milk_price : ℚ)
  (bread_quantity : ℕ) (bread_price : ℚ) (egg_milk_tax : ℚ) (bread_tax : ℚ)
  (egg_discount : ℚ) (milk_discount : ℚ) : ℚ :=
  let egg_subtotal := egg_quantity * egg_price
  let milk_subtotal := milk_quantity * milk_price
  let bread_subtotal := bread_quantity * bread_price
  let egg_discounted := egg_subtotal * (1 - egg_discount)
  let milk_discounted := milk_subtotal * (1 - milk_discount)
  let egg_with_tax := egg_discounted * (1 + egg_milk_tax)
  let milk_with_tax := milk_discounted * (1 + egg_milk_tax)
  let bread_with_tax := bread_subtotal * (1 + bread_tax)
  egg_with_tax + milk_with_tax + bread_with_tax

theorem grocery_cost_is_correct :
  grocery_cost 36 0.5 2 3 4 1.25 0.05 0.02 0.1 0.05 = 12.51 := by
  sorry

end grocery_cost_is_correct_l158_15833


namespace mysterious_quadratic_polynomial_value_at_zero_l158_15841

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

/-- A polynomial is mysterious if p(p(x))=0 has exactly four real roots, including multiplicities -/
def IsMysteri6ous (p : ℝ → ℝ) : Prop :=
  ∃ (roots : Finset ℝ), (∀ x, x ∈ roots ↔ p (p x) = 0) ∧ roots.card = 4

/-- The sum of roots of a quadratic polynomial -/
def SumOfRoots (b c : ℝ) : ℝ := -b

theorem mysterious_quadratic_polynomial_value_at_zero
  (b c : ℝ)
  (h_mysterious : IsMysteri6ous (QuadraticPolynomial b c))
  (h_minimal_sum : ∀ b' c', IsMysteri6ous (QuadraticPolynomial b' c') → SumOfRoots b c ≤ SumOfRoots b' c') :
  QuadraticPolynomial b c 0 = 4 := by
  sorry

end mysterious_quadratic_polynomial_value_at_zero_l158_15841


namespace min_value_expression_l158_15880

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  ∃ (min : ℝ), min = 6 ∧ 
  (∀ x, x = m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) → x ≥ min) ∧
  (∃ y, y = m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ∧ y = min) :=
sorry

end min_value_expression_l158_15880


namespace matrix_equation_solution_l158_15803

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 0; 0, 1]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0, 5]

theorem matrix_equation_solution :
  ∀ X : Matrix (Fin 2) (Fin 1) ℝ,
  B⁻¹ * A⁻¹ * X = !![5; 1] →
  X = !![28; 5] := by
sorry

end matrix_equation_solution_l158_15803


namespace grade_assignment_count_l158_15858

theorem grade_assignment_count : 
  (number_of_grades : ℕ) → 
  (number_of_students : ℕ) → 
  number_of_grades = 4 → 
  number_of_students = 12 → 
  number_of_grades ^ number_of_students = 16777216 :=
by
  sorry

end grade_assignment_count_l158_15858


namespace functional_equation_solution_l158_15836

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x, 2 * f x + f (1 - x) = x^2) →
  (∀ x, f x = (1/3) * x^2 + (2/3) * x - 1/3) :=
by sorry

end functional_equation_solution_l158_15836


namespace cylinder_volume_from_rectangle_l158_15816

/-- The volume of a cylinder formed by rotating a rectangle about its longer side. -/
theorem cylinder_volume_from_rectangle (width length : ℝ) (h_width : width = 8) (h_length : length = 20) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  volume = 320 * π := by sorry

end cylinder_volume_from_rectangle_l158_15816


namespace greatest_integer_fraction_inequality_l158_15877

theorem greatest_integer_fraction_inequality :
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 17 ↔ x ≤ 12 :=
by sorry

end greatest_integer_fraction_inequality_l158_15877


namespace system_solution_l158_15884

theorem system_solution :
  let S := {(x, y) : ℝ × ℝ | x + y = 3 ∧ 2*x - 3*y = 1}
  S = {(2, 1)} := by sorry

end system_solution_l158_15884


namespace unique_solution_power_equation_l158_15873

theorem unique_solution_power_equation : 
  ∃! (x : ℝ), x ≠ 0 ∧ (9 * x)^18 = (18 * x)^9 :=
by
  use 2/9
  sorry

end unique_solution_power_equation_l158_15873


namespace no_inscribed_sphere_when_black_exceeds_white_l158_15828

/-- Represents a face of a polyhedron -/
structure Face where
  area : ℝ
  color : Bool  -- True for black, False for white

/-- Represents a convex polyhedron -/
structure Polyhedron where
  faces : List Face
  is_convex : Bool
  no_adjacent_black : Bool

/-- Checks if a sphere can be inscribed in the polyhedron -/
def can_inscribe_sphere (p : Polyhedron) : Prop :=
  sorry

/-- Calculates the total area of faces of a given color -/
def total_area (p : Polyhedron) (color : Bool) : ℝ :=
  sorry

/-- Main theorem -/
theorem no_inscribed_sphere_when_black_exceeds_white (p : Polyhedron) :
  p.is_convex ∧ 
  p.no_adjacent_black ∧ 
  (total_area p true > total_area p false) →
  ¬(can_inscribe_sphere p) :=
by
  sorry

end no_inscribed_sphere_when_black_exceeds_white_l158_15828


namespace min_value_expression_l158_15859

theorem min_value_expression (x y z : ℝ) (hx : x ≥ 3) (hy : y ≥ 3) (hz : z ≥ 3) :
  let A := ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x*y + y*z + z*x)
  A ≥ 1 ∧ (A = 1 ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end min_value_expression_l158_15859


namespace surface_area_between_cylinders_l158_15899

/-- The total surface area of the space between two concentric cylinders -/
theorem surface_area_between_cylinders (h inner_radius : ℝ) 
  (h_pos : h > 0) (inner_radius_pos : inner_radius > 0) :
  let outer_radius := inner_radius + 1
  2 * π * h * (outer_radius - inner_radius) = 16 * π := by
  sorry

end surface_area_between_cylinders_l158_15899


namespace locus_of_Q_l158_15860

/-- The locus of point Q given an ellipse with specific properties -/
theorem locus_of_Q (a b : ℝ) (P : ℝ × ℝ) (E : ℝ × ℝ) (Q : ℝ × ℝ) :
  a > b → b > 0 →
  (P.1^2 / a^2) + (P.2^2 / b^2) = 1 →
  P ≠ (-2, 0) → P ≠ (2, 0) →
  a = 2 →
  (1 : ℝ) / 2 = Real.sqrt (1 - b^2 / a^2) →
  E.1 - (-4) = (3 / 5) * (P.1 - (-4)) →
  E.2 = (3 / 5) * P.2 →
  (Q.2 + 2) / (Q.1 + 2) = P.2 / (P.1 + 2) →
  (Q.2 - 0) / (Q.1 - 2) = E.2 / (E.1 - 2) →
  Q.2 ≠ 0 →
  (Q.1 + 1)^2 + (4 * Q.2^2) / 3 = 1 := by
sorry

end locus_of_Q_l158_15860


namespace exponential_function_fixed_point_l158_15849

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  a^(0 : ℝ) = 1 := by sorry

end exponential_function_fixed_point_l158_15849


namespace quadratic_roots_bounds_l158_15808

theorem quadratic_roots_bounds (a b : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ < x₂ →
  (∀ x, -1 < x ∧ x < 1 → x^2 + a*x + b < 0) →
  -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 :=
by sorry

end quadratic_roots_bounds_l158_15808


namespace cube_edge_product_equality_l158_15843

-- Define a cube as a structure with 12 edges
structure Cube :=
  (edges : Fin 12 → ℕ)

-- Define a predicate to check if the edges contain all numbers from 1 to 12
def validEdges (c : Cube) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → ∃ i : Fin 12, c.edges i = n

-- Define the top and bottom faces of the cube
def topFace (c : Cube) : Fin 4 → ℕ := λ i => c.edges i
def bottomFace (c : Cube) : Fin 4 → ℕ := λ i => c.edges (i + 8)

-- Define the product of numbers on a face
def faceProduct (face : Fin 4 → ℕ) : ℕ := (face 0) * (face 1) * (face 2) * (face 3)

-- Theorem statement
theorem cube_edge_product_equality :
  ∃ c : Cube, validEdges c ∧ faceProduct (topFace c) = faceProduct (bottomFace c) := by
  sorry

end cube_edge_product_equality_l158_15843


namespace four_digit_numbers_count_l158_15846

/-- The number of digits in the given number -/
def total_digits : ℕ := 4

/-- The number of distinct digits in the given number -/
def distinct_digits : ℕ := 2

/-- The number of occurrences of the first digit (3) -/
def count_first_digit : ℕ := 2

/-- The number of occurrences of the second digit (0) -/
def count_second_digit : ℕ := 2

/-- The function to calculate the number of permutations -/
def permutations (n : ℕ) (n1 : ℕ) (n2 : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial n1 * Nat.factorial n2)

/-- The theorem stating the number of different four-digit numbers -/
theorem four_digit_numbers_count : 
  permutations (total_digits - 1) (count_first_digit - 1) count_second_digit = 3 := by
  sorry

end four_digit_numbers_count_l158_15846


namespace prob_end_two_tails_after_second_head_l158_15830

/-- A fair coin flip can result in either heads or tails with equal probability -/
def FairCoin : Type := Bool

/-- The outcome of a sequence of coin flips -/
inductive FlipOutcome
| TwoHeads
| TwoTails
| Incomplete

/-- The state of the coin flipping process -/
structure FlipState :=
  (seenSecondHead : Bool)
  (lastFlip : Option Bool)
  (outcome : FlipOutcome)

/-- Simulates a single coin flip and updates the state -/
def flipCoin (state : FlipState) : FlipState := sorry

/-- Calculates the probability of ending with two tails after seeing the second head -/
def probEndTwoTailsAfterSecondHead : ℝ := sorry

/-- The main theorem to prove -/
theorem prob_end_two_tails_after_second_head :
  probEndTwoTailsAfterSecondHead = 1 / 24 := by sorry

end prob_end_two_tails_after_second_head_l158_15830


namespace chocolate_chip_cookie_recipes_l158_15838

/-- Given a recipe that requires a certain amount of an ingredient and a total amount of that ingredient needed, calculate the number of recipes that can be made. -/
def recipes_to_make (cups_per_recipe : ℚ) (total_cups_needed : ℚ) : ℚ :=
  total_cups_needed / cups_per_recipe

/-- Prove that 23 recipes can be made given the conditions of the chocolate chip cookie problem. -/
theorem chocolate_chip_cookie_recipes : 
  recipes_to_make 2 46 = 23 := by
  sorry

end chocolate_chip_cookie_recipes_l158_15838


namespace complex_fraction_calculation_l158_15823

theorem complex_fraction_calculation : (9 * 9 - 2 * 2) / ((1 / 12) - (1 / 19)) = 2508 := by
  sorry

end complex_fraction_calculation_l158_15823


namespace cos_negative_52_thirds_pi_l158_15825

theorem cos_negative_52_thirds_pi : 
  Real.cos (-52 / 3 * Real.pi) = -1 / 2 := by sorry

end cos_negative_52_thirds_pi_l158_15825


namespace root_product_theorem_l158_15885

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 13/3 := by
sorry

end root_product_theorem_l158_15885


namespace alices_number_l158_15851

def possible_numbers : List ℕ := [1080, 1440, 1800, 2160, 2520, 2880]

theorem alices_number (n : ℕ) :
  (40 ∣ n) →
  (72 ∣ n) →
  1000 < n →
  n < 3000 →
  n ∈ possible_numbers := by
sorry

end alices_number_l158_15851


namespace betty_and_sister_book_ratio_l158_15827

theorem betty_and_sister_book_ratio : 
  ∀ (betty_books sister_books : ℕ),
    betty_books = 20 →
    betty_books + sister_books = 45 →
    (sister_books : ℚ) / betty_books = 5 / 4 :=
by
  sorry

end betty_and_sister_book_ratio_l158_15827


namespace lg2_bounds_l158_15844

theorem lg2_bounds :
  (10 : ℝ)^3 = 1000 ∧ (10 : ℝ)^4 = 10000 ∧
  (2 : ℝ)^10 = 1024 ∧ (2 : ℝ)^11 = 2048 ∧
  (2 : ℝ)^12 = 4096 ∧ (2 : ℝ)^13 = 8192 →
  3/10 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 4/13 := by
  sorry

end lg2_bounds_l158_15844


namespace circle_equation_proof_l158_15893

/-- A circle with center on the x-axis passing through two given points -/
structure CircleOnXAxis where
  center : ℝ  -- x-coordinate of the center
  passesThrough : (ℝ × ℝ) → (ℝ × ℝ) → Prop

/-- The equation of a circle given its center and a point on the circle -/
def circleEquation (h : ℝ) (k : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = ((5 - h)^2 + 2^2)

theorem circle_equation_proof (c : CircleOnXAxis) 
  (h1 : c.passesThrough (5, 2) (-1, 4)) :
  ∀ x y, circleEquation 1 0 x y ↔ (x - 1)^2 + y^2 = 20 := by
  sorry

end circle_equation_proof_l158_15893


namespace cubic_root_sum_cubes_l158_15857

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 505 * a + 1010 = 0) →
  (5 * b^3 + 505 * b + 1010 = 0) →
  (5 * c^3 + 505 * c + 1010 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 606 := by
  sorry

end cubic_root_sum_cubes_l158_15857


namespace age_difference_l158_15815

/-- Proves that A is 10 years older than B given the conditions in the problem -/
theorem age_difference (A B : ℕ) : 
  B = 70 →  -- B's present age is 70 years
  A + 20 = 2 * (B - 20) →  -- In 20 years, A will be twice as old as B was 20 years ago
  A - B = 10  -- A is 10 years older than B
  := by sorry

end age_difference_l158_15815


namespace lcm_hcf_problem_l158_15809

theorem lcm_hcf_problem (n : ℕ) 
  (h1 : Nat.lcm 12 n = 60) 
  (h2 : Nat.gcd 12 n = 3) : 
  n = 15 := by
sorry

end lcm_hcf_problem_l158_15809


namespace range_of_a_l158_15875

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 5*x + 4 ≤ 0) →
  a < 0 →
  -4/3 ≤ a ∧ a ≤ -1 :=
by sorry

end range_of_a_l158_15875


namespace xyz_sum_sqrt_l158_15848

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 18) 
  (h2 : z + x = 19) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 24150.1875 := by
  sorry

end xyz_sum_sqrt_l158_15848


namespace percentage_gain_calculation_l158_15869

def calculate_percentage_gain (total_bought : ℕ) (cost_per_bowl : ℚ) (total_sold : ℕ) (sell_per_bowl : ℚ) : ℚ :=
  let total_cost := total_bought * cost_per_bowl
  let total_revenue := total_sold * sell_per_bowl
  let profit := total_revenue - total_cost
  (profit / total_cost) * 100

theorem percentage_gain_calculation :
  let total_bought : ℕ := 114
  let cost_per_bowl : ℚ := 13
  let total_sold : ℕ := 108
  let sell_per_bowl : ℚ := 17
  abs (calculate_percentage_gain total_bought cost_per_bowl total_sold sell_per_bowl - 23.88) < 0.01 := by
  sorry

end percentage_gain_calculation_l158_15869


namespace lowest_salary_grade_l158_15814

/-- Represents the salary grade of an employee -/
def SalaryGrade := {s : ℝ // 1 ≤ s ∧ s ≤ 5}

/-- Calculates the hourly wage based on the salary grade -/
def hourlyWage (s : SalaryGrade) : ℝ :=
  7.50 + 0.25 * (s.val - 1)

/-- States that the difference in hourly wage between the highest and lowest salary grade is $1.25 -/
axiom wage_difference (s_min s_max : SalaryGrade) :
  s_min.val = 1 ∧ s_max.val = 5 →
  hourlyWage s_max - hourlyWage s_min = 1.25

theorem lowest_salary_grade :
  ∃ (s_min : SalaryGrade), s_min.val = 1 ∧
  ∀ (s : SalaryGrade), s_min.val ≤ s.val :=
by sorry

end lowest_salary_grade_l158_15814


namespace number_of_ways_to_draw_l158_15895

/-- The number of balls in the bin -/
def total_balls : ℕ := 15

/-- The number of balls to be drawn -/
def drawn_balls : ℕ := 4

/-- The sequence of colors to be drawn -/
def color_sequence : List String := ["Red", "Green", "Blue", "Yellow"]

/-- Function to calculate the number of ways to draw the balls -/
def ways_to_draw : ℕ := (total_balls - 0) * (total_balls - 1) * (total_balls - 2) * (total_balls - 3)

/-- Theorem stating the number of ways to draw the balls -/
theorem number_of_ways_to_draw :
  ways_to_draw = 32760 := by sorry

end number_of_ways_to_draw_l158_15895


namespace max_a_value_l158_15845

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2 + |x^3 - 2*x| ≥ a*x) → 
  a ≤ 2 * Real.sqrt 2 := by
sorry

end max_a_value_l158_15845


namespace intersection_point_mod17_l158_15879

theorem intersection_point_mod17 :
  ∃ x : ℕ, x < 17 ∧
  (∀ y : ℕ, (y ≡ 7 * x + 3 [MOD 17]) ↔ (y ≡ 13 * x + 4 [MOD 17])) ∧
  x = 14 :=
by sorry

end intersection_point_mod17_l158_15879


namespace system_solution_l158_15863

theorem system_solution : 
  ∃ (x y : ℝ), (x^4 - y^4 = 3 * Real.sqrt (abs y) - 3 * Real.sqrt (abs x)) ∧ 
                (x^2 - 2*x*y = 27) ↔ 
  ((x = 3 ∧ y = -3) ∨ (x = -3 ∧ y = 3)) :=
by sorry

end system_solution_l158_15863


namespace rug_dimension_l158_15837

theorem rug_dimension (floor_area : ℝ) (rug_width : ℝ) (coverage_fraction : ℝ) :
  floor_area = 64 ∧ 
  rug_width = 2 ∧ 
  coverage_fraction = 0.21875 →
  ∃ (rug_length : ℝ), 
    rug_length * rug_width = floor_area * coverage_fraction ∧
    rug_length = 7 := by
  sorry

end rug_dimension_l158_15837


namespace newspaper_conference_max_overlap_l158_15811

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) :
  total = 100 →
  writers = 45 →
  editors > 36 →
  writers + editors - x + 2 * x = total →
  x ≤ 18 :=
sorry

end newspaper_conference_max_overlap_l158_15811


namespace largest_number_in_sample_l158_15807

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  total_items : ℕ
  first_number : ℕ
  second_number : ℕ
  sample_size : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSample) : ℕ :=
  s.first_number + (s.sample_size - 1) * (s.second_number - s.first_number)

/-- Theorem stating the largest number in the given systematic sample -/
theorem largest_number_in_sample :
  let s : SystematicSample := {
    total_items := 400,
    first_number := 8,
    second_number := 33,
    sample_size := 16
  }
  largest_sample_number s = 383 := by sorry

end largest_number_in_sample_l158_15807


namespace power_sixteen_divided_by_eight_l158_15896

theorem power_sixteen_divided_by_eight (m : ℕ) : m = 16^2023 → m / 8 = 2^8089 := by
  sorry

end power_sixteen_divided_by_eight_l158_15896


namespace wangwa_smallest_growth_rate_l158_15887

structure BreedingBase where
  name : String
  growthRate : Float

def liwa : BreedingBase := { name := "Liwa", growthRate := 3.25 }
def wangwa : BreedingBase := { name := "Wangwa", growthRate := -2.75 }
def jiazhuang : BreedingBase := { name := "Jiazhuang", growthRate := 4.6 }
def wuzhuang : BreedingBase := { name := "Wuzhuang", growthRate := -1.76 }

def breedingBases : List BreedingBase := [liwa, wangwa, jiazhuang, wuzhuang]

theorem wangwa_smallest_growth_rate :
  ∀ b ∈ breedingBases, wangwa.growthRate ≤ b.growthRate :=
by sorry

end wangwa_smallest_growth_rate_l158_15887


namespace unique_number_property_l158_15847

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end unique_number_property_l158_15847


namespace bob_spending_theorem_l158_15872

def monday_spending (initial_amount : ℚ) : ℚ := initial_amount / 2

def tuesday_spending (monday_remainder : ℚ) : ℚ := monday_remainder / 5

def wednesday_spending (tuesday_remainder : ℚ) : ℚ := tuesday_remainder * 3 / 8

def final_amount (initial_amount : ℚ) : ℚ :=
  let monday_remainder := initial_amount - monday_spending initial_amount
  let tuesday_remainder := monday_remainder - tuesday_spending monday_remainder
  tuesday_remainder - wednesday_spending tuesday_remainder

theorem bob_spending_theorem :
  final_amount 80 = 20 := by
  sorry

end bob_spending_theorem_l158_15872


namespace circumradius_ge_twice_inradius_l158_15819

/-- A triangle is represented by its three vertices in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Theorem: The circumradius is greater than or equal to twice the inradius for any triangle,
    with equality if and only if the triangle is equilateral -/
theorem circumradius_ge_twice_inradius (t : Triangle) :
  circumradius t ≥ 2 * inradius t ∧
  (circumradius t = 2 * inradius t ↔ is_equilateral t) := by
  sorry

end circumradius_ge_twice_inradius_l158_15819


namespace circle_radius_determines_c_l158_15842

/-- The equation of a circle with center (h, k) and radius r can be written as
    (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r c : ℝ) : Prop :=
  ∀ x y, x^2 + 6*x + y^2 - 4*y + c = 0 ↔ (x + 3)^2 + (y - 2)^2 = r^2

theorem circle_radius_determines_c : 
  ∀ c : ℝ, (CircleEquation (-3) 2 4 c) → c = -3 := by
  sorry

end circle_radius_determines_c_l158_15842


namespace lower_right_is_four_l158_15868

-- Define the grid type
def Grid := Fin 5 → Fin 5 → Fin 5

-- Define a valid grid
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧ 
  (∀ i j k, i ≠ j → g k i ≠ g k j)

-- Define the initial configuration
def initial_config (g : Grid) : Prop :=
  g 0 0 = 1 ∧ g 0 2 = 2 ∧ g 0 3 = 3 ∧
  g 1 0 = 2 ∧ g 1 1 = 3 ∧ g 1 4 = 1 ∧
  g 2 1 = 1 ∧ g 2 3 = 5 ∧
  g 4 2 = 4

-- Theorem statement
theorem lower_right_is_four :
  ∀ g : Grid, is_valid_grid g → initial_config g → g 4 4 = 4 :=
sorry

end lower_right_is_four_l158_15868


namespace min_sum_exponents_520_l158_15840

/-- Given a natural number n, returns the sum of exponents in its binary representation -/
def sumOfExponents (n : ℕ) : ℕ := sorry

/-- Expresses a natural number as a sum of distinct powers of 2 -/
def expressAsPowersOf2 (n : ℕ) : List ℕ := sorry

theorem min_sum_exponents_520 :
  let powers := expressAsPowersOf2 520
  powers.length ≥ 2 ∧ sumOfExponents 520 = 12 :=
sorry

end min_sum_exponents_520_l158_15840


namespace simplify_expression_l158_15818

theorem simplify_expression (y : ℝ) : 4*y + 8*y^3 + 6 - (3 - 4*y - 8*y^3) = 16*y^3 + 8*y + 3 := by
  sorry

end simplify_expression_l158_15818


namespace geometric_sequence_third_term_l158_15874

/-- A geometric sequence with first term 2 and fifth term 8 has its third term equal to 4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- a is a sequence of real numbers indexed by natural numbers
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1))  -- a is a geometric sequence
  (h_a1 : a 1 = 2)  -- first term is 2
  (h_a5 : a 5 = 8)  -- fifth term is 8
  : a 3 = 4 := by
  sorry

end geometric_sequence_third_term_l158_15874


namespace profit_after_reduction_profit_for_target_l158_15889

/-- Represents the daily sales and profit calculations for a product. -/
structure ProductSales where
  basePrice : ℝ
  baseSales : ℝ
  profitPerItem : ℝ
  salesIncreasePerYuan : ℝ

/-- Calculates the daily profit given a price reduction. -/
def dailyProfit (p : ProductSales) (priceReduction : ℝ) : ℝ :=
  (p.profitPerItem - priceReduction) * (p.baseSales + p.salesIncreasePerYuan * priceReduction)

/-- Theorem stating that a 3 yuan price reduction results in 1692 yuan daily profit. -/
theorem profit_after_reduction (p : ProductSales) 
  (h1 : p.basePrice = 50)
  (h2 : p.baseSales = 30)
  (h3 : p.profitPerItem = 50)
  (h4 : p.salesIncreasePerYuan = 2) :
  dailyProfit p 3 = 1692 := by sorry

/-- Theorem stating that a 25 yuan price reduction results in 2000 yuan daily profit. -/
theorem profit_for_target (p : ProductSales)
  (h1 : p.basePrice = 50)
  (h2 : p.baseSales = 30)
  (h3 : p.profitPerItem = 50)
  (h4 : p.salesIncreasePerYuan = 2) :
  dailyProfit p 25 = 2000 := by sorry

end profit_after_reduction_profit_for_target_l158_15889


namespace min_value_theorem_l158_15839

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (3 / x + 4 / y) ≥ 7 + 4 * Real.sqrt 3 := by
  sorry

end min_value_theorem_l158_15839


namespace cliff_rock_collection_l158_15810

theorem cliff_rock_collection (igneous sedimentary : ℕ) : 
  igneous = sedimentary / 2 →
  igneous / 3 = 30 →
  igneous + sedimentary = 270 :=
by
  sorry

end cliff_rock_collection_l158_15810


namespace largest_eight_digit_with_even_digits_l158_15855

/-- The set of even digits -/
def evenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- A function to check if a natural number contains all even digits -/
def containsAllEvenDigits (n : Nat) : Prop :=
  ∀ d ∈ evenDigits, ∃ k : Nat, n / (10 ^ k) % 10 = d

/-- A function to check if a natural number is an eight-digit number -/
def isEightDigitNumber (n : Nat) : Prop :=
  10000000 ≤ n ∧ n ≤ 99999999

/-- The theorem stating that 99986420 is the largest eight-digit number containing all even digits -/
theorem largest_eight_digit_with_even_digits :
  (∀ n : Nat, isEightDigitNumber n → containsAllEvenDigits n → n ≤ 99986420) ∧
  isEightDigitNumber 99986420 ∧
  containsAllEvenDigits 99986420 :=
sorry

end largest_eight_digit_with_even_digits_l158_15855


namespace ninth_power_sum_l158_15826

/-- Given two real numbers m and n satisfying specific conditions, prove that m⁹ + n⁹ = 76 -/
theorem ninth_power_sum (m n : ℝ) 
  (h1 : m + n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 4)
  (h4 : m^4 + n^4 = 7)
  (h5 : m^5 + n^5 = 11) : 
  m^9 + n^9 = 76 := by
  sorry

#check ninth_power_sum

end ninth_power_sum_l158_15826


namespace not_sufficient_for_geometric_sequence_l158_15876

theorem not_sufficient_for_geometric_sequence 
  (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  ¬ (∀ n : ℕ, ∃ r : ℝ, ∀ k : ℕ, a (n + k) = a n * r ^ k) :=
sorry

end not_sufficient_for_geometric_sequence_l158_15876


namespace min_value_inequality_l158_15853

theorem min_value_inequality (x : ℝ) (h : x ≥ 4) : x + 4 / (x - 1) ≥ 5 := by
  sorry

end min_value_inequality_l158_15853


namespace park_grass_area_calculation_l158_15820

/-- Represents the geometry of a circular park with a path and square plot -/
structure ParkGeometry where
  circle_diameter : ℝ
  path_width : ℝ
  square_side : ℝ

/-- Calculates the remaining grass area in the park -/
def remaining_grass_area (park : ParkGeometry) : ℝ :=
  sorry

/-- Theorem stating the remaining grass area for the given park configuration -/
theorem park_grass_area_calculation (park : ParkGeometry) 
  (h1 : park.circle_diameter = 20)
  (h2 : park.path_width = 4)
  (h3 : park.square_side = 6) :
  remaining_grass_area park = 78.21 * Real.pi + 13 := by
  sorry

end park_grass_area_calculation_l158_15820


namespace vector_collinearity_l158_15882

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_collinearity (m : ℝ) :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (m, 2*m - 3)
  collinear a b → m = -3 := by sorry

end vector_collinearity_l158_15882


namespace smallest_prime_digit_sum_23_l158_15804

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem stating that 599 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 :
  (is_prime 599) ∧ 
  (digit_sum 599 = 23) ∧ 
  (∀ n : ℕ, n < 599 → ¬(is_prime n ∧ digit_sum n = 23)) :=
by sorry

end smallest_prime_digit_sum_23_l158_15804
