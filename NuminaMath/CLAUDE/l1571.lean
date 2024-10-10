import Mathlib

namespace union_of_I_is_odd_integers_l1571_157133

def I : ℕ → Set ℤ
  | 0 => {-1, 1}
  | n + 1 => {x : ℤ | ∃ y ∈ I n, x^2 - 2*x*y + y^2 = 4^(n + 1)}

def OddIntegers : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2*k + 1}

theorem union_of_I_is_odd_integers :
  (⋃ n : ℕ, I n) = OddIntegers :=
sorry

end union_of_I_is_odd_integers_l1571_157133


namespace certain_number_of_seconds_l1571_157151

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℕ) : ℕ := m * 60

/-- The proportion given in the problem -/
def proportion (x : ℕ) : Prop :=
  15 / x = 30 / minutes_to_seconds 10

theorem certain_number_of_seconds : ∃ x : ℕ, proportion x ∧ x = 300 :=
  sorry

end certain_number_of_seconds_l1571_157151


namespace car_speed_from_tire_rotation_l1571_157161

/-- Given a tire rotating at a certain rate with a specific circumference,
    calculate the speed of the car in km/h. -/
theorem car_speed_from_tire_rotation 
  (revolutions_per_minute : ℝ) 
  (tire_circumference : ℝ) 
  (h1 : revolutions_per_minute = 400) 
  (h2 : tire_circumference = 5) : 
  (revolutions_per_minute * tire_circumference * 60) / 1000 = 120 := by
  sorry

#check car_speed_from_tire_rotation

end car_speed_from_tire_rotation_l1571_157161


namespace sum_and_reciprocal_integer_l1571_157109

theorem sum_and_reciprocal_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) :
  (∃ m : ℤ, a^2 + 1/a^2 = m) ∧ (∀ n : ℕ, ∃ l : ℤ, a^n + 1/a^n = l) := by
  sorry


end sum_and_reciprocal_integer_l1571_157109


namespace max_sum_of_factors_l1571_157139

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 3003 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 117 :=
by sorry

end max_sum_of_factors_l1571_157139


namespace digits_of_3_power_20_times_5_power_15_l1571_157181

theorem digits_of_3_power_20_times_5_power_15 : ∃ n : ℕ, 
  (10 ^ (n - 1) ≤ 3^20 * 5^15) ∧ (3^20 * 5^15 < 10^n) ∧ (n = 16) := by sorry

end digits_of_3_power_20_times_5_power_15_l1571_157181


namespace g_of_5_equals_22_l1571_157130

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x + 2

-- Theorem statement
theorem g_of_5_equals_22 : g 5 = 22 := by
  sorry

end g_of_5_equals_22_l1571_157130


namespace exists_bound_for_factorial_digit_sum_l1571_157166

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: Existence of a bound for factorial digit sum -/
theorem exists_bound_for_factorial_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10^100 := by
  sorry

end exists_bound_for_factorial_digit_sum_l1571_157166


namespace natasha_quarters_l1571_157112

theorem natasha_quarters : 
  ∃ (n : ℕ), 8 < n ∧ n < 80 ∧ 
  n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 ∧
  (∀ (m : ℕ), 8 < m ∧ m < 80 ∧ 
   m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 → m = n) ∧
  n = 61 :=
by sorry

end natasha_quarters_l1571_157112


namespace quadratic_roots_and_integer_case_l1571_157118

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 + (2*k + 1) * x + 2

-- Theorem statement
theorem quadratic_roots_and_integer_case :
  (∀ k : ℝ, k ≠ 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) ∧
  (∀ k : ℕ+, (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) → k = 1) :=
by sorry


end quadratic_roots_and_integer_case_l1571_157118


namespace square_sum_zero_iff_both_zero_l1571_157189

theorem square_sum_zero_iff_both_zero (x y : ℝ) : x^2 + y^2 = 0 ↔ x = 0 ∧ y = 0 := by
  sorry

end square_sum_zero_iff_both_zero_l1571_157189


namespace multiply_power_equals_power_sum_problem_solution_l1571_157156

theorem multiply_power_equals_power_sum (a : ℕ) (m n : ℕ) : 
  a * (a^n) = a^(n + 1) := by sorry

theorem problem_solution : 
  3000 * (3000^3000) = 3000^3001 := by sorry

end multiply_power_equals_power_sum_problem_solution_l1571_157156


namespace trig_expression_approx_value_l1571_157127

theorem trig_expression_approx_value : 
  let expr := (2 * Real.sin (30 * π / 180) * Real.cos (10 * π / 180) + 
               3 * Real.cos (150 * π / 180) * Real.cos (110 * π / 180)) /
              (4 * Real.sin (40 * π / 180) * Real.cos (20 * π / 180) + 
               5 * Real.cos (140 * π / 180) * Real.cos (100 * π / 180))
  ∃ ε > 0, abs (expr - 0.465) < ε := by
  sorry

end trig_expression_approx_value_l1571_157127


namespace median_lengths_l1571_157113

/-- Given a triangle with sides a, b, and c, this theorem states the formulas for the lengths of the medians sa, sb, and sc. -/
theorem median_lengths (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (sa sb sc : ℝ),
    sa = Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2 ∧
    sb = Real.sqrt (2 * a^2 + 2 * c^2 - b^2) / 2 ∧
    sc = Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2 :=
by sorry


end median_lengths_l1571_157113


namespace chess_tournament_games_l1571_157178

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 5 → total_games = 10 → (n * (n - 1)) / 2 = total_games → n - 1 = 4 := by
  sorry

end chess_tournament_games_l1571_157178


namespace trains_crossing_time_l1571_157110

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 500 → 
  length2 = 750 → 
  speed1 = 60 → 
  speed2 = 40 → 
  (((length1 + length2) / ((speed1 + speed2) * (5/18))) : ℝ) = 45 := by
  sorry

#check trains_crossing_time

end trains_crossing_time_l1571_157110


namespace roots_sum_product_l1571_157193

theorem roots_sum_product (a b : ℂ) : 
  (a ≠ b) → 
  (a^3 + 3*a^2 + a + 1 = 0) → 
  (b^3 + 3*b^2 + b + 1 = 0) → 
  (a^2 * b + a * b^2 + 3*a*b = 1) := by
sorry

end roots_sum_product_l1571_157193


namespace smallest_consecutive_product_seven_consecutive_product_seven_is_smallest_l1571_157115

theorem smallest_consecutive_product (n : ℕ) : n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 5040 → n ≥ 7 :=
by sorry

theorem seven_consecutive_product : 7 * 8 * 9 * 10 = 5040 :=
by sorry

theorem seven_is_smallest : ∃ (n : ℕ), n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 5040 ∧ n = 7 :=
by sorry

end smallest_consecutive_product_seven_consecutive_product_seven_is_smallest_l1571_157115


namespace sum_reciprocals_bound_l1571_157163

/-- A function that checks if a positive integer contains the consecutive digit sequence 2048 -/
def contains2048 (n : ℕ+) : Prop := sorry

/-- The set of all positive integers that do not contain the consecutive digit sequence 2048 -/
def S : Set ℕ+ := {n : ℕ+ | ¬contains2048 n}

/-- The theorem to be proved -/
theorem sum_reciprocals_bound (T : Set ℕ+) (h : T ⊆ S) :
  ∑' (n : T), (1 : ℝ) / n ≤ 400000 := by sorry

end sum_reciprocals_bound_l1571_157163


namespace fraction_denominator_proof_l1571_157177

theorem fraction_denominator_proof (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (9 * y) / 20 + (3 * y) / x = 0.75 * y) : x = 10 := by
  sorry

end fraction_denominator_proof_l1571_157177


namespace mersenne_coprime_iff_exponents_coprime_l1571_157122

theorem mersenne_coprime_iff_exponents_coprime (p q : ℕ+) :
  Nat.gcd ((2 : ℕ) ^ p.val - 1) ((2 : ℕ) ^ q.val - 1) = 1 ↔ Nat.gcd p.val q.val = 1 := by
  sorry

end mersenne_coprime_iff_exponents_coprime_l1571_157122


namespace school_visit_arrangements_l1571_157128

/-- Represents the number of days in a week -/
def week_days : Nat := 7

/-- Represents the number of consecutive days School A visits -/
def school_a_days : Nat := 2

/-- Represents the number of days School B visits -/
def school_b_days : Nat := 1

/-- Represents the number of days School C visits -/
def school_c_days : Nat := 1

/-- Calculates the number of arrangements for the school visits -/
def calculate_arrangements : Nat :=
  (week_days - school_a_days - school_b_days - school_c_days + 1) *
  (week_days - school_a_days - school_b_days - school_c_days)

/-- Theorem stating that the number of arrangements is 40 -/
theorem school_visit_arrangements :
  calculate_arrangements = 40 := by
  sorry

end school_visit_arrangements_l1571_157128


namespace min_value_xy_l1571_157159

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_seq : (1/4 * Real.log x) * (Real.log y) = 1/16) : 
  x * y ≥ Real.exp 1 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ (1/4 * Real.log x) * (Real.log y) = 1/16 ∧ x * y = Real.exp 1 := by
  sorry

end min_value_xy_l1571_157159


namespace tom_seashells_l1571_157117

/-- The number of seashells Tom found -/
def total_seashells (broken : ℕ) (unbroken : ℕ) : ℕ :=
  broken + unbroken

/-- Theorem stating that Tom found 7 seashells in total -/
theorem tom_seashells : total_seashells 4 3 = 7 := by
  sorry

end tom_seashells_l1571_157117


namespace partner_investment_time_l1571_157134

/-- Given two partners P and Q with investments and profits, prove Q's investment time -/
theorem partner_investment_time 
  (investment_ratio : ℚ) -- Ratio of P's investment to Q's investment
  (profit_ratio : ℚ) -- Ratio of P's profit to Q's profit
  (p_time : ℕ) -- Time P invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 14)
  (h3 : p_time = 5) :
  ∃ (q_time : ℕ), q_time = 14 := by
  sorry


end partner_investment_time_l1571_157134


namespace log_equation_solution_l1571_157148

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) = 1 →
  x = b^((3 + Real.sqrt 5) / 2) ∨ x = b^((3 - Real.sqrt 5) / 2) := by
sorry

end log_equation_solution_l1571_157148


namespace staircase_has_31_steps_l1571_157152

/-- Represents a staircase with a middle step and specific movement rules -/
structure Staircase where
  total_steps : ℕ
  middle_step : ℕ
  (middle_property : middle_step * 2 - 1 = total_steps)
  (movement_property : middle_step + 7 - 15 = 8)

/-- Theorem stating that a staircase satisfying the given conditions has 31 steps -/
theorem staircase_has_31_steps (s : Staircase) : s.total_steps = 31 := by
  sorry

#check staircase_has_31_steps

end staircase_has_31_steps_l1571_157152


namespace unique_prime_solution_l1571_157140

theorem unique_prime_solution : 
  ∀ p q r : ℕ, 
    Prime p ∧ Prime q ∧ Prime r →
    p^2 + 1 = 74 * (q^2 + r^2) →
    p = 31 ∧ q = 2 ∧ r = 3 :=
by
  sorry

end unique_prime_solution_l1571_157140


namespace hyperbola_equation_l1571_157105

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_equation :
  (∀ x y : ℝ, asymptotes x y → (∃ a b : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ b^2 = 3 * a^2)) →
  hyperbola 2 3 →
  ∀ x y : ℝ, hyperbola x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end hyperbola_equation_l1571_157105


namespace cloth_cost_price_l1571_157116

/-- Represents the cost and profit scenario for cloth selling --/
structure ClothSelling where
  total_length : ℕ
  first_half : ℕ
  second_half : ℕ
  total_price : ℚ
  profit_first : ℚ
  profit_second : ℚ

/-- The theorem stating the cost price per meter if it's the same for both halves --/
theorem cloth_cost_price (cs : ClothSelling)
  (h_total : cs.total_length = 120)
  (h_half : cs.first_half = cs.second_half)
  (h_length : cs.first_half + cs.second_half = cs.total_length)
  (h_price : cs.total_price = 15360)
  (h_profit1 : cs.profit_first = 1/10)
  (h_profit2 : cs.profit_second = 1/5)
  (h_equal_cost : ∃ (c : ℚ), 
    cs.first_half * (1 + cs.profit_first) * c + 
    cs.second_half * (1 + cs.profit_second) * c = cs.total_price) :
  ∃ (c : ℚ), c = 11130 / 100 := by
  sorry

end cloth_cost_price_l1571_157116


namespace tangent_circles_distance_l1571_157187

/-- The distance between the centers of two tangent circles with radii 1 and 7 is either 6 or 8 -/
theorem tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 1 → r₂ = 7 → (d = r₁ + r₂ ∨ d = |r₂ - r₁|) → d = 6 ∨ d = 8 := by sorry

end tangent_circles_distance_l1571_157187


namespace quadratic_equation_solution_l1571_157129

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 5*x + 6 = 0 ↔ x = 3 ∨ x = 2 := by
sorry

end quadratic_equation_solution_l1571_157129


namespace total_precious_stones_l1571_157111

theorem total_precious_stones :
  let agate : ℕ := 24
  let olivine : ℕ := agate + 5
  let sapphire : ℕ := 2 * olivine
  let diamond : ℕ := olivine + 11
  let amethyst : ℕ := sapphire + diamond
  let ruby : ℕ := (5 * olivine + 1) / 2  -- Rounded up
  let garnet : ℕ := amethyst - ruby - 5
  let topaz : ℕ := garnet / 2
  agate + olivine + sapphire + diamond + amethyst + ruby + garnet + topaz = 352 := by
sorry

end total_precious_stones_l1571_157111


namespace multiply_by_three_l1571_157154

theorem multiply_by_three (x : ℤ) : x + 14 = 56 → 3 * x = 126 := by
  sorry

end multiply_by_three_l1571_157154


namespace negation_of_universal_statement_l1571_157195

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) := by
  sorry

end negation_of_universal_statement_l1571_157195


namespace ellen_orange_juice_amount_l1571_157142

/-- The amount of orange juice in Ellen's smoothie --/
def orange_juice_amount (strawberries yogurt total : ℚ) : ℚ :=
  total - (strawberries + yogurt)

/-- Theorem: Ellen used 0.2 cups of orange juice in her smoothie --/
theorem ellen_orange_juice_amount :
  orange_juice_amount (2/10) (1/10) (5/10) = 2/10 := by
  sorry

end ellen_orange_juice_amount_l1571_157142


namespace binomial_coefficient_seven_four_l1571_157147

theorem binomial_coefficient_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end binomial_coefficient_seven_four_l1571_157147


namespace sqrt_meaningful_condition_l1571_157172

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x + 6) ↔ x ≥ -3 :=
sorry

end sqrt_meaningful_condition_l1571_157172


namespace equation_equivalence_l1571_157182

theorem equation_equivalence (a b c : ℝ) (h : b > 0) :
  (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1 →
  a * c = 36 * b :=
by sorry

end equation_equivalence_l1571_157182


namespace angle_EFG_value_l1571_157132

/-- A configuration of a square inside a regular octagon sharing one side -/
structure SquareInOctagon where
  /-- The measure of an internal angle of the regular octagon -/
  octagon_angle : ℝ
  /-- The measure of an internal angle of the square -/
  square_angle : ℝ
  /-- The measure of angle EFH -/
  angle_EFH : ℝ
  /-- The measure of angle EFG -/
  angle_EFG : ℝ

/-- Properties of the SquareInOctagon configuration -/
axiom octagon_angle_value (config : SquareInOctagon) : config.octagon_angle = 135
axiom square_angle_value (config : SquareInOctagon) : config.square_angle = 90
axiom angle_EFH_value (config : SquareInOctagon) : config.angle_EFH = config.octagon_angle - config.square_angle
axiom isosceles_triangle (config : SquareInOctagon) : config.angle_EFG = (180 - config.angle_EFH) / 2

/-- The main theorem: angle EFG measures 67.5° -/
theorem angle_EFG_value (config : SquareInOctagon) : config.angle_EFG = 67.5 := by
  sorry

end angle_EFG_value_l1571_157132


namespace unique_function_solution_l1571_157144

/-- A function from non-negative reals to non-negative reals -/
def NonnegFunction := {f : ℝ → ℝ // ∀ x, 0 ≤ x → 0 ≤ f x}

theorem unique_function_solution (f : NonnegFunction) 
  (h : ∀ x : ℝ, 0 ≤ x → f.val (f.val x) + f.val x = 12 * x) :
  ∀ x : ℝ, 0 ≤ x → f.val x = 3 * x := by
  sorry

end unique_function_solution_l1571_157144


namespace river_straight_parts_length_l1571_157168

theorem river_straight_parts_length 
  (total_length : ℝ) 
  (straight_percentage : ℝ) 
  (h1 : total_length = 80) 
  (h2 : straight_percentage = 0.25) : 
  straight_percentage * total_length = 20 := by
  sorry

end river_straight_parts_length_l1571_157168


namespace existence_of_representation_l1571_157153

theorem existence_of_representation (m : ℤ) :
  ∃ (a b k : ℤ), Odd a ∧ Odd b ∧ k ≥ 0 ∧ 2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end existence_of_representation_l1571_157153


namespace fifth_largest_divisor_l1571_157191

def n : ℕ := 1209600000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → (x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n))

theorem fifth_largest_divisor :
  is_fifth_largest_divisor 75600000 :=
sorry

end fifth_largest_divisor_l1571_157191


namespace min_value_theorem_l1571_157136

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (1 / a) + (1 / b) = 1) :
  6 ≤ (4 / (a - 1)) + (9 / (b - 1)) ∧ 
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ (1 / a₀) + (1 / b₀) = 1 ∧ (4 / (a₀ - 1)) + (9 / (b₀ - 1)) = 6 := by
  sorry

end min_value_theorem_l1571_157136


namespace parabola_directrix_l1571_157158

/-- The directrix of the parabola x = -1/4 * y^2 is the line x = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (x = -1/4 * y^2) → (∃ (p : ℝ), p = 1 ∧ 
    (∀ (x₀ y₀ : ℝ), x₀ = -1/4 * y₀^2 → 
      ((x₀ + 1)^2 + y₀^2 = (x₀ - p)^2))) := by
  sorry

end parabola_directrix_l1571_157158


namespace cody_dumplings_l1571_157102

/-- The number of dumplings Cody cooked -/
def dumplings_cooked : ℕ := sorry

/-- The number of dumplings Cody ate -/
def dumplings_eaten : ℕ := 7

/-- The number of dumplings Cody has left -/
def dumplings_left : ℕ := 7

theorem cody_dumplings : dumplings_cooked = dumplings_eaten + dumplings_left := by
  sorry

end cody_dumplings_l1571_157102


namespace inverse_cube_theorem_l1571_157169

-- Define the relationship between z and x
def inverse_cube_relation (z x : ℝ) : Prop :=
  ∃ k : ℝ, 7 * z = k / (x^3)

-- State the theorem
theorem inverse_cube_theorem :
  ∀ z₁ z₂ : ℝ,
  inverse_cube_relation z₁ 2 ∧ z₁ = 4 →
  inverse_cube_relation z₂ 4 →
  z₂ = 0.5 := by
  sorry

end inverse_cube_theorem_l1571_157169


namespace f_at_three_fifths_l1571_157180

def f (x : ℝ) : ℝ := 15 * x^5 + 6 * x^4 + x^3 - x^2 - 2*x - 1

theorem f_at_three_fifths :
  f (3/5) = -2/5 := by sorry

end f_at_three_fifths_l1571_157180


namespace total_paintable_area_l1571_157186

/-- Represents the dimensions of a bedroom -/
structure Bedroom where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the wall area of a bedroom -/
def wallArea (b : Bedroom) : ℝ :=
  2 * (b.length * b.height + b.width * b.height)

/-- Calculates the paintable area of a bedroom -/
def paintableArea (b : Bedroom) (unpaintableArea : ℝ) : ℝ :=
  wallArea b - unpaintableArea

/-- The four bedrooms in Isabella's house -/
def isabellasBedrooms : List Bedroom := [
  { length := 14, width := 12, height := 9 },
  { length := 13, width := 11, height := 9 },
  { length := 15, width := 10, height := 9 },
  { length := 12, width := 12, height := 9 }
]

/-- The area occupied by doorways and windows in each bedroom -/
def unpaintableAreaPerRoom : ℝ := 70

/-- Theorem: The total area of walls to be painted in Isabella's house is 1502 square feet -/
theorem total_paintable_area :
  (isabellasBedrooms.map (fun b => paintableArea b unpaintableAreaPerRoom)).sum = 1502 := by
  sorry

end total_paintable_area_l1571_157186


namespace find_A_l1571_157176

theorem find_A : ∀ A : ℝ, 10 + A = 15 → A = 5 := by
  sorry

end find_A_l1571_157176


namespace reflection_of_circle_center_l1571_157174

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  reflect_about_y_neg_x original_center = (-3, -8) := by
  sorry

end reflection_of_circle_center_l1571_157174


namespace odot_one_four_odot_comm_l1571_157164

-- Define the ⊙ operation for rational numbers
def odot (a b : ℚ) : ℚ := a - a * b + b + 3

-- Theorem: 1 ⊙ 4 = 4
theorem odot_one_four : odot 1 4 = 4 := by sorry

-- Theorem: ⊙ is commutative
theorem odot_comm (a b : ℚ) : odot a b = odot b a := by sorry

end odot_one_four_odot_comm_l1571_157164


namespace hyperbola_vertices_distance_l1571_157165

/-- The distance between the vertices of a hyperbola with equation x^2/64 - y^2/49 = 1 is 16 -/
theorem hyperbola_vertices_distance : 
  ∀ (x y : ℝ), x^2/64 - y^2/49 = 1 → ∃ (v1 v2 : ℝ × ℝ), 
    (v1.1^2/64 - v1.2^2/49 = 1) ∧ 
    (v2.1^2/64 - v2.2^2/49 = 1) ∧ 
    (v1.2 = 0) ∧ (v2.2 = 0) ∧
    (v2.1 = -v1.1) ∧
    (v2.1 - v1.1 = 16) :=
by sorry

end hyperbola_vertices_distance_l1571_157165


namespace max_b_value_l1571_157145

theorem max_b_value (y : ℤ) (b : ℕ+) (h : y^2 + b * y = -21) :
  b ≤ 22 ∧ ∃ y : ℤ, y^2 + 22 * y = -21 :=
sorry

end max_b_value_l1571_157145


namespace lcm_28_72_l1571_157101

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l1571_157101


namespace sqrt_meaningful_iff_geq_two_l1571_157149

theorem sqrt_meaningful_iff_geq_two (a : ℝ) : ∃ x : ℝ, x^2 = a - 2 ↔ a ≥ 2 := by sorry

end sqrt_meaningful_iff_geq_two_l1571_157149


namespace parabola_distance_sum_lower_bound_l1571_157170

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point N
def N : ℝ × ℝ := (2, 2)

-- Statement of the theorem
theorem parabola_distance_sum_lower_bound :
  ∀ (M : ℝ × ℝ), parabola M.1 M.2 →
  dist M focus + dist M N ≥ 3 :=
sorry

end parabola_distance_sum_lower_bound_l1571_157170


namespace fencing_final_probability_l1571_157160

theorem fencing_final_probability (p_a : ℝ) (h1 : p_a = 0.41) :
  let p_b := 1 - p_a
  p_b = 0.59 := by
sorry

end fencing_final_probability_l1571_157160


namespace otimes_neg_two_neg_one_l1571_157192

/-- Custom binary operation ⊗ -/
def otimes (a b : ℝ) : ℝ := a^2 - abs b

/-- Theorem stating that (-2) ⊗ (-1) = 3 -/
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end otimes_neg_two_neg_one_l1571_157192


namespace drums_per_day_l1571_157108

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums are filled per day. -/
theorem drums_per_day : 
  let total_drums : ℕ := 2916
  let total_days : ℕ := 9
  let drums_per_day : ℕ := total_drums / total_days
  drums_per_day = 324 := by sorry

end drums_per_day_l1571_157108


namespace intersection_complement_theorem_l1571_157188

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | x ≥ 2}

theorem intersection_complement_theorem : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end intersection_complement_theorem_l1571_157188


namespace max_discount_rate_l1571_157183

/-- Represents the maximum discount rate problem -/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (original_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4) 
  (h2 : original_price = 5) 
  (h3 : min_profit_margin = 0.1) : 
  ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount → 
      original_price * (1 - discount / 100) - cost_price ≥ min_profit_margin * cost_price :=
sorry

end max_discount_rate_l1571_157183


namespace quadratic_equation_roots_l1571_157141

theorem quadratic_equation_roots (a b c : ℝ) : 
  (∀ x, a * x * (x + 1) + b * x * (x + 2) + c * (x + 1) * (x + 2) = 0 ↔ x = 1 ∨ x = 2) →
  a + b + c = 2 →
  a = 12 := by sorry

end quadratic_equation_roots_l1571_157141


namespace ferris_wheel_capacity_is_56_l1571_157198

/-- The number of people the Ferris wheel can seat -/
def ferris_wheel_capacity (total_waiting : ℕ) (not_riding : ℕ) : ℕ :=
  total_waiting - not_riding

/-- Theorem: The Ferris wheel capacity is 56 people given the problem conditions -/
theorem ferris_wheel_capacity_is_56 :
  ferris_wheel_capacity 92 36 = 56 := by
  sorry

end ferris_wheel_capacity_is_56_l1571_157198


namespace lee_cookies_l1571_157106

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this theorem proves he can make 36 cookies with 4.5 cups of flour. -/
theorem lee_cookies (cookies_per_3_cups : ℕ) (cookies_per_4_5_cups : ℕ) 
  (h1 : cookies_per_3_cups = 24) :
  cookies_per_4_5_cups = 36 :=
by
  sorry

#check lee_cookies

end lee_cookies_l1571_157106


namespace smallest_d_value_l1571_157185

def no_triangle (a b c : ℝ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_d_value (c d : ℝ) 
  (h1 : 2 < c ∧ c < d)
  (h2 : no_triangle 2 c d)
  (h3 : no_triangle (1/d) (1/c) 2) :
  d = 2 + Real.sqrt 2 := by
sorry

end smallest_d_value_l1571_157185


namespace green_yards_calculation_l1571_157184

/-- The number of yards of silk dyed green in a factory order -/
def green_yards (total_yards pink_yards : ℕ) : ℕ :=
  total_yards - pink_yards

/-- Theorem stating that the number of yards dyed green is 61921 -/
theorem green_yards_calculation :
  green_yards 111421 49500 = 61921 := by
  sorry

end green_yards_calculation_l1571_157184


namespace seating_arrangements_l1571_157173

structure Table :=
  (chairs : ℕ)
  (couples : ℕ)

def valid_seating (t : Table) (arrangements : ℕ) : Prop :=
  t.chairs = 12 ∧
  t.couples = 6 ∧
  arrangements = 43200

theorem seating_arrangements (t : Table) :
  valid_seating t 43200 := by
  sorry

end seating_arrangements_l1571_157173


namespace regression_maximum_fitting_l1571_157135

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents the true relationship between x and y --/
def true_relationship : ℝ → ℝ := sorry

/-- Measures the degree of fitting between the regression model and the true relationship --/
def fitting_degree (model : LinearRegression) : ℝ := sorry

/-- The regression equation represents the maximum degree of fitting --/
theorem regression_maximum_fitting (data : List (ℝ × ℝ)) :
  ∃ (model : LinearRegression),
    ∀ (other_model : LinearRegression),
      fitting_degree model ≥ fitting_degree other_model := by
  sorry

end regression_maximum_fitting_l1571_157135


namespace average_marks_abc_l1571_157143

theorem average_marks_abc (M : ℝ) (D : ℝ) :
  -- The average marks of a, b, c is M
  -- When d joins, the average becomes 47
  3 * M + D = 4 * 47 →
  -- The average marks of b, c, d, e is 48
  -- E has 3 more marks than d
  -- The marks of a is 43
  (3 * M - 43) + D + (D + 3) = 4 * 48 →
  -- The average marks of a, b, c is 48
  M = 48 := by
  sorry

end average_marks_abc_l1571_157143


namespace special_line_equation_l1571_157124

/-- A line passing through (6, -2) with x-intercept 1 greater than y-intercept -/
structure SpecialLine where
  /-- The equation of the line in the form ax + by + c = 0 -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- The line passes through (6, -2) -/
  point_condition : a * 6 + b * (-2) + c = 0
  /-- The x-intercept is 1 greater than the y-intercept -/
  intercept_condition : -c/a = -c/b + 1

/-- The equation of the special line is either x + 2y - 2 = 0 or 2x + 3y - 6 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (l.a = 1 ∧ l.b = 2 ∧ l.c = -2) ∨ (l.a = 2 ∧ l.b = 3 ∧ l.c = -6) :=
sorry

end special_line_equation_l1571_157124


namespace consecutive_products_sum_l1571_157104

theorem consecutive_products_sum : ∃ (a b c d e : ℕ), 
  (b = a + 1) ∧ 
  (d = c + 1) ∧ 
  (e = d + 1) ∧ 
  (a * b = 210) ∧ 
  (c * d * e = 210) ∧ 
  (a + b + c + d + e = 47) := by
  sorry

end consecutive_products_sum_l1571_157104


namespace equation_equivalence_l1571_157194

theorem equation_equivalence : ∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end equation_equivalence_l1571_157194


namespace height_comparison_l1571_157196

theorem height_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end height_comparison_l1571_157196


namespace five_dollar_neg_three_l1571_157138

-- Define the $ operation
def dollar_op (a b : Int) : Int := a * (b - 1) + a * b

-- Theorem statement
theorem five_dollar_neg_three : dollar_op 5 (-3) = -35 := by
  sorry

end five_dollar_neg_three_l1571_157138


namespace smartphone_savings_smartphone_savings_proof_l1571_157123

/-- Calculates the weekly savings required to purchase a smartphone --/
theorem smartphone_savings (smartphone_cost current_savings : ℚ) : ℚ :=
  let remaining_amount := smartphone_cost - current_savings
  let weeks_in_two_months := 2 * (52 / 12 : ℚ)
  remaining_amount / weeks_in_two_months

/-- Proves that the weekly savings for the given scenario is approximately $13.86 --/
theorem smartphone_savings_proof :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |smartphone_savings 160 40 - (13.86 : ℚ)| < ε :=
sorry

end smartphone_savings_smartphone_savings_proof_l1571_157123


namespace fan_sales_theorem_l1571_157121

/-- Represents the sales data for a week -/
structure WeeklySales where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the fan models and their properties -/
structure FanModels where
  purchasePriceA : ℕ
  purchasePriceB : ℕ
  sellingPriceA : ℕ
  sellingPriceB : ℕ

def totalUnits : ℕ := 40
def maxBudget : ℕ := 7650

def weekOneSales : WeeklySales := {
  modelA := 3
  modelB := 5
  revenue := 2150
}

def weekTwoSales : WeeklySales := {
  modelA := 4
  modelB := 10
  revenue := 3700
}

def fanModels : FanModels := {
  purchasePriceA := 210
  purchasePriceB := 180
  sellingPriceA := 300
  sellingPriceB := 250
}

theorem fan_sales_theorem (w1 : WeeklySales) (w2 : WeeklySales) (f : FanModels) :
  w1 = weekOneSales ∧ w2 = weekTwoSales ∧ f.purchasePriceA = 210 ∧ f.purchasePriceB = 180 →
  (f.sellingPriceA * w1.modelA + f.sellingPriceB * w1.modelB = w1.revenue ∧
   f.sellingPriceA * w2.modelA + f.sellingPriceB * w2.modelB = w2.revenue) →
  f.sellingPriceA = 300 ∧ f.sellingPriceB = 250 ∧
  (∀ a : ℕ, a ≤ totalUnits →
    f.purchasePriceA * a + f.purchasePriceB * (totalUnits - a) ≤ maxBudget →
    (f.sellingPriceA - f.purchasePriceA) * a + (f.sellingPriceB - f.purchasePriceB) * (totalUnits - a) ≤ 3100) ∧
  ∃ a : ℕ, a ≤ totalUnits ∧
    f.purchasePriceA * a + f.purchasePriceB * (totalUnits - a) ≤ maxBudget ∧
    (f.sellingPriceA - f.purchasePriceA) * a + (f.sellingPriceB - f.purchasePriceB) * (totalUnits - a) = 3100 :=
by sorry

end fan_sales_theorem_l1571_157121


namespace dragon_jewels_l1571_157150

theorem dragon_jewels (D : ℕ) : 
  6 = D / 3 →  -- The new jewels (6) are one-third of the original count
  21 = D - 3 + 6 -- The final count is the original count minus 3 (stolen) plus 6 (taken from king)
  := by sorry

end dragon_jewels_l1571_157150


namespace fishing_competition_result_l1571_157199

/-- The total number of days in the fishing season -/
def season_days : ℕ := 213

/-- The number of fish caught per day by the first fisherman -/
def first_fisherman_rate : ℕ := 3

/-- The number of days the second fisherman catches 1 fish per day -/
def second_fisherman_phase1_days : ℕ := 30

/-- The number of days the second fisherman catches 2 fish per day -/
def second_fisherman_phase2_days : ℕ := 60

/-- The number of fish caught per day by the second fisherman in phase 1 -/
def second_fisherman_phase1_rate : ℕ := 1

/-- The number of fish caught per day by the second fisherman in phase 2 -/
def second_fisherman_phase2_rate : ℕ := 2

/-- The number of fish caught per day by the second fisherman in phase 3 -/
def second_fisherman_phase3_rate : ℕ := 4

/-- The total number of fish caught by the first fisherman -/
def first_fisherman_total : ℕ := first_fisherman_rate * season_days

/-- The total number of fish caught by the second fisherman -/
def second_fisherman_total : ℕ :=
  second_fisherman_phase1_rate * second_fisherman_phase1_days +
  second_fisherman_phase2_rate * second_fisherman_phase2_days +
  second_fisherman_phase3_rate * (season_days - second_fisherman_phase1_days - second_fisherman_phase2_days)

theorem fishing_competition_result :
  second_fisherman_total - first_fisherman_total = 3 := by sorry

end fishing_competition_result_l1571_157199


namespace arithmetic_sum_l1571_157131

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 := by
sorry

end arithmetic_sum_l1571_157131


namespace sum_of_first_60_digits_l1571_157175

/-- The decimal representation of 1/2222 -/
def decimal_rep : ℚ := 1 / 2222

/-- The repeating sequence in the decimal representation -/
def repeating_sequence : List ℕ := [0, 0, 0, 4, 5]

/-- The length of the repeating sequence -/
def sequence_length : ℕ := repeating_sequence.length

/-- The sum of digits in one repetition of the sequence -/
def sequence_sum : ℕ := repeating_sequence.sum

/-- The number of complete repetitions in the first 60 digits -/
def num_repetitions : ℕ := 60 / sequence_length

theorem sum_of_first_60_digits : 
  (num_repetitions * sequence_sum = 108) := by sorry

end sum_of_first_60_digits_l1571_157175


namespace smallest_multiple_l1571_157190

theorem smallest_multiple (n : ℕ) : n = 481 ↔ 
  n > 0 ∧ 
  (∃ k : ℤ, n = 37 * k) ∧ 
  (∃ m : ℤ, n - 7 = 97 * m) ∧ 
  (∀ x : ℕ, x > 0 ∧ (∃ k : ℤ, x = 37 * k) ∧ (∃ m : ℤ, x - 7 = 97 * m) → x ≥ n) :=
sorry

end smallest_multiple_l1571_157190


namespace petra_beads_removal_l1571_157171

/-- Represents the number of blue beads Petra has initially -/
def initial_blue_beads : ℕ := 49

/-- Represents the number of red beads Petra has initially -/
def initial_red_beads : ℕ := 1

/-- Represents the total number of beads Petra has initially -/
def initial_total_beads : ℕ := initial_blue_beads + initial_red_beads

/-- Represents the number of beads Petra needs to remove -/
def beads_to_remove : ℕ := 40

/-- Represents the desired percentage of blue beads after removal -/
def desired_blue_percentage : ℚ := 90 / 100

theorem petra_beads_removal :
  let remaining_beads := initial_total_beads - beads_to_remove
  let remaining_blue_beads := initial_blue_beads - (beads_to_remove - initial_red_beads)
  (remaining_blue_beads : ℚ) / remaining_beads = desired_blue_percentage :=
sorry

end petra_beads_removal_l1571_157171


namespace probability_is_five_twelfths_l1571_157126

/-- Represents a person with 6 differently colored blocks -/
structure Person :=
  (blocks : Fin 6 → Color)

/-- Represents the colors of blocks -/
inductive Color
  | Red | Blue | Yellow | White | Green | Purple

/-- Represents a box with placed blocks -/
structure Box :=
  (blocks : Fin 3 → Color)

/-- The probability of at least one box receiving blocks of the same color from at least two different people -/
def probability_same_color (people : Fin 3 → Person) (boxes : Fin 5 → Box) : ℚ :=
  sorry

/-- The main theorem stating that the probability is 5/12 -/
theorem probability_is_five_twelfths :
  ∃ (people : Fin 3 → Person) (boxes : Fin 5 → Box),
    probability_same_color people boxes = 5 / 12 :=
  sorry

end probability_is_five_twelfths_l1571_157126


namespace danny_wrappers_found_l1571_157119

/-- Represents the number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- Represents the number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- Represents the difference between bottle caps and wrappers found -/
def difference : ℕ := 4

theorem danny_wrappers_found :
  bottle_caps_found = wrappers_found + difference →
  wrappers_found = 46 := by
  sorry

end danny_wrappers_found_l1571_157119


namespace t_shirt_cost_is_20_l1571_157100

/-- The cost of a single t-shirt -/
def t_shirt_cost : ℝ := sorry

/-- The number of t-shirts bought -/
def num_t_shirts : ℕ := 3

/-- The cost of pants -/
def pants_cost : ℝ := 50

/-- The total amount spent -/
def total_spent : ℝ := 110

theorem t_shirt_cost_is_20 : t_shirt_cost = 20 := by
  sorry

end t_shirt_cost_is_20_l1571_157100


namespace f_derivative_at_negative_one_l1571_157114

noncomputable def f (x : ℝ) : ℝ := -x^3 + 1/x

theorem f_derivative_at_negative_one :
  (deriv f) (-1) = -4 :=
sorry

end f_derivative_at_negative_one_l1571_157114


namespace area_of_larger_rectangle_l1571_157167

/-- A rectangle with area 2 and length twice its width -/
structure SmallerRectangle where
  width : ℝ
  length : ℝ
  area_eq_two : width * length = 2
  length_eq_twice_width : length = 2 * width

/-- The larger rectangle formed by three smaller rectangles -/
def LargerRectangle (r : SmallerRectangle) : ℝ × ℝ :=
  (3 * r.length, r.width)

/-- The theorem to be proved -/
theorem area_of_larger_rectangle (r : SmallerRectangle) :
  (LargerRectangle r).1 * (LargerRectangle r).2 = 6 := by
  sorry

#check area_of_larger_rectangle

end area_of_larger_rectangle_l1571_157167


namespace cos_2theta_value_l1571_157162

theorem cos_2theta_value (θ : ℝ) :
  let a : ℝ × ℝ := (1, Real.cos (2 * x))
  let b : ℝ × ℝ := (Real.sin (2 * x), -Real.sqrt 3)
  let f : ℝ → ℝ := λ x => a.1 * b.1 + a.2 * b.2
  f (θ / 2 + 2 * Real.pi / 3) = 6 / 5 →
  Real.cos (2 * θ) = 7 / 25 :=
by sorry

end cos_2theta_value_l1571_157162


namespace seating_arrangements_l1571_157155

/-- Represents the number of people sitting around the table -/
def total_people : ℕ := 8

/-- Represents the number of people in the special block (leader, vice leader, recorder) -/
def special_block : ℕ := 3

/-- Represents the number of units to arrange (treating the special block as one unit) -/
def units_to_arrange : ℕ := total_people - special_block + 1

/-- Represents the number of ways to arrange the people within the special block -/
def internal_arrangements : ℕ := 2

/-- Calculates the number of unique circular arrangements for n elements -/
def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The main theorem stating the number of unique seating arrangements -/
theorem seating_arrangements : 
  circular_arrangements units_to_arrange * internal_arrangements = 240 := by
  sorry

end seating_arrangements_l1571_157155


namespace area_of_region_l1571_157146

/-- The region defined by the inequality |4x-24|+|3y+10| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 24| + |3 * p.2 + 10| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the area of the region is 12 -/
theorem area_of_region : area Region = 12 := by sorry

end area_of_region_l1571_157146


namespace max_intersections_three_polygons_l1571_157125

/-- Represents a convex polygon with a given number of sides -/
structure ConvexPolygon where
  sides : ℕ

/-- Theorem stating the maximum number of intersections among three convex polygons -/
theorem max_intersections_three_polygons
  (P1 P2 P3 : ConvexPolygon)
  (h1 : P1.sides ≤ P2.sides)
  (h2 : P2.sides ≤ P3.sides)
  (h_no_shared_segments : True)  -- Represents the condition that polygons don't share line segments
  : ℕ := by
  sorry

end max_intersections_three_polygons_l1571_157125


namespace total_rubber_bands_l1571_157120

def harper_rubber_bands : ℕ := 100
def brother_difference : ℕ := 56
def sister_difference : ℕ := 47

theorem total_rubber_bands :
  harper_rubber_bands +
  (harper_rubber_bands - brother_difference) +
  (harper_rubber_bands - brother_difference + sister_difference) = 235 := by
  sorry

end total_rubber_bands_l1571_157120


namespace hyperbola_eccentricity_l1571_157157

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  2 * x - y = 0

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y, hyperbola a b x y ∧ asymptote x y) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l1571_157157


namespace solution_set_inequality_l1571_157107

theorem solution_set_inequality (x : ℝ) : 
  (x + 1/2) * (3/2 - x) ≥ 0 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

end solution_set_inequality_l1571_157107


namespace final_worker_bees_count_l1571_157137

/-- Represents the state of the bee hive --/
structure BeeHive where
  workers : ℕ
  drones : ℕ
  queens : ℕ
  guards : ℕ

/-- Applies the series of events to the bee hive --/
def applyEvents (hive : BeeHive) : BeeHive :=
  let hive1 := { hive with 
    workers := hive.workers - 28,
    drones := hive.drones - 12,
    guards := hive.guards - 5 }
  let hive2 := { hive1 with 
    workers := hive1.workers - 30,
    guards := hive1.guards + 30 }
  let hive3 := { hive2 with 
    workers := hive2.workers + 15 }
  { hive3 with 
    workers := 0 }

/-- The theorem to be proved --/
theorem final_worker_bees_count (initialHive : BeeHive) 
  (h1 : initialHive.workers = 400)
  (h2 : initialHive.drones = 75)
  (h3 : initialHive.queens = 1)
  (h4 : initialHive.guards = 50) :
  (applyEvents initialHive).workers = 0 := by
  sorry

#check final_worker_bees_count

end final_worker_bees_count_l1571_157137


namespace car_total_distance_l1571_157197

/-- A car driving through a ring in a tunnel -/
structure CarInRing where
  /-- Number of right-hand turns in the ring -/
  turns : ℕ
  /-- Distance traveled after the 1st turn -/
  dist1 : ℝ
  /-- Distance traveled after the 2nd turn -/
  dist2 : ℝ
  /-- Distance traveled after the 3rd turn -/
  dist3 : ℝ

/-- The total distance driven by the car around the ring -/
def totalDistance (car : CarInRing) : ℝ :=
  car.dist1 + car.dist2 + car.dist3

/-- Theorem stating the total distance driven by the car -/
theorem car_total_distance (car : CarInRing) 
  (h1 : car.turns = 4)
  (h2 : car.dist1 = 5)
  (h3 : car.dist2 = 8)
  (h4 : car.dist3 = 10) : 
  totalDistance car = 23 := by
  sorry

end car_total_distance_l1571_157197


namespace stating_perpendicular_bisector_correct_l1571_157103

/-- The perpendicular bisector of a line segment. -/
def perpendicular_bisector (line_eq : ℝ → ℝ → Prop) (x_range : Set ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * x + y - 3 = 0

/-- The original line segment equation. -/
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- The range of x for the original line segment. -/
def x_range : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

/-- 
Theorem stating that the perpendicular_bisector function correctly defines 
the perpendicular bisector of the line segment given by the original_line 
equation within the specified x_range.
-/
theorem perpendicular_bisector_correct : 
  perpendicular_bisector original_line x_range = 
    fun x y => 2 * x + y - 3 = 0 := by sorry

end stating_perpendicular_bisector_correct_l1571_157103


namespace ramesh_refrigerator_price_l1571_157179

/-- The price Ramesh paid for the refrigerator --/
def price_paid (labelled_price : ℝ) : ℝ :=
  0.8 * labelled_price + 125 + 250

/-- The theorem stating the price Ramesh paid for the refrigerator --/
theorem ramesh_refrigerator_price :
  ∃ (labelled_price : ℝ),
    1.2 * labelled_price = 19200 ∧
    price_paid labelled_price = 13175 := by
  sorry

end ramesh_refrigerator_price_l1571_157179
