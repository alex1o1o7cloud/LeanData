import Mathlib

namespace smallest_valid_l71_7169

/-- A positive integer n is valid if 2n is a perfect square and 3n is a perfect cube. -/
def is_valid (n : ℕ+) : Prop :=
  ∃ k m : ℕ+, 2 * n = k^2 ∧ 3 * n = m^3

/-- 72 is the smallest positive integer that is valid. -/
theorem smallest_valid : (∀ n : ℕ+, n < 72 → ¬ is_valid n) ∧ is_valid 72 := by
  sorry

end smallest_valid_l71_7169


namespace no_prime_roots_sum_64_l71_7119

theorem no_prime_roots_sum_64 : ¬∃ (p q k : ℕ), 
  Prime p ∧ Prime q ∧ 
  p * q = k ∧
  p + q = 64 ∧
  p^2 - 64*p + k = 0 ∧
  q^2 - 64*q + k = 0 :=
sorry

end no_prime_roots_sum_64_l71_7119


namespace power_product_positive_l71_7126

theorem power_product_positive (m n : ℕ) (hm : m > 2) :
  ∃ k : ℕ+, (2^m - 1) * (2^n + 1) = k := by
  sorry

end power_product_positive_l71_7126


namespace ways_A_to_C_via_B_l71_7157

/-- The number of ways to get from point A to point B -/
def ways_AB : ℕ := 3

/-- The number of ways to get from point B to point C -/
def ways_BC : ℕ := 4

/-- The total number of ways to get from point A to point C via point B -/
def total_ways : ℕ := ways_AB * ways_BC

theorem ways_A_to_C_via_B : total_ways = 12 := by
  sorry

end ways_A_to_C_via_B_l71_7157


namespace lee_family_concert_cost_is_86_l71_7161

/-- Represents the cost calculation for the Lee family concert tickets --/
def lee_family_concert_cost : ℝ :=
  let regular_ticket_cost : ℝ := 10
  let booking_fee : ℝ := 1.5
  let youngest_discount : ℝ := 0.4
  let oldest_discount : ℝ := 0.3
  let middle_discount : ℝ := 0.2
  let youngest_count : ℕ := 3
  let oldest_count : ℕ := 3
  let middle_count : ℕ := 4
  let total_tickets : ℕ := youngest_count + oldest_count + middle_count

  let youngest_cost : ℝ := youngest_count * (regular_ticket_cost * (1 - youngest_discount))
  let oldest_cost : ℝ := oldest_count * (regular_ticket_cost * (1 - oldest_discount))
  let middle_cost : ℝ := middle_count * (regular_ticket_cost * (1 - middle_discount))
  
  let total_ticket_cost : ℝ := youngest_cost + oldest_cost + middle_cost
  let total_booking_fees : ℝ := total_tickets * booking_fee

  total_ticket_cost + total_booking_fees

/-- Theorem stating that the total cost for the Lee family concert tickets is $86.00 --/
theorem lee_family_concert_cost_is_86 : lee_family_concert_cost = 86 := by
  sorry

end lee_family_concert_cost_is_86_l71_7161


namespace complex_magnitude_problem_l71_7143

theorem complex_magnitude_problem (z : ℂ) (h : (3 - 4*I) * z = 4 + 3*I) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l71_7143


namespace max_value_of_expression_l71_7127

theorem max_value_of_expression (m : ℝ) : 
  (4 - |2 - m|) ≤ 4 ∧ ∃ m : ℝ, 4 - |2 - m| = 4 :=
by sorry

end max_value_of_expression_l71_7127


namespace horner_rule_v₂_l71_7129

def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

def v₀ : ℝ := 4

def v₁ (x : ℝ) : ℝ := v₀ * x + 3

def v₂ (x : ℝ) : ℝ := v₁ x * x - 6

theorem horner_rule_v₂ : v₂ (-1) = -5 := by sorry

end horner_rule_v₂_l71_7129


namespace collins_total_petals_l71_7123

/-- The number of petals Collin has after receiving flowers from Ingrid -/
theorem collins_total_petals (collins_initial_flowers ingrid_flowers petals_per_flower : ℕ) : 
  collins_initial_flowers = 25 →
  ingrid_flowers = 33 →
  petals_per_flower = 4 →
  (collins_initial_flowers + ingrid_flowers / 3) * petals_per_flower = 144 := by
  sorry

#check collins_total_petals

end collins_total_petals_l71_7123


namespace exists_triangle_from_polygon_with_inscribed_circle_l71_7176

/-- A polygon with an inscribed circle. -/
structure PolygonWithInscribedCircle where
  /-- The number of sides of the polygon. -/
  n : ℕ
  /-- The lengths of the sides of the polygon. -/
  sides : Fin n → ℝ
  /-- The radius of the inscribed circle. -/
  radius : ℝ
  /-- All sides are positive. -/
  sides_positive : ∀ i, sides i > 0
  /-- The inscribed circle is tangent to all sides. -/
  tangent_to_all_sides : ∀ i, ∃ t, 0 < t ∧ t < sides i ∧ t = radius

/-- Theorem: In a polygon with an inscribed circle, there exist three sides that form a triangle. -/
theorem exists_triangle_from_polygon_with_inscribed_circle
  (p : PolygonWithInscribedCircle)
  (h : p.n ≥ 3) :
  ∃ i j k : Fin p.n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    p.sides i + p.sides j > p.sides k ∧
    p.sides j + p.sides k > p.sides i ∧
    p.sides k + p.sides i > p.sides j :=
  sorry

end exists_triangle_from_polygon_with_inscribed_circle_l71_7176


namespace existence_of_m_and_n_l71_7158

theorem existence_of_m_and_n :
  ∃ (m n : ℕ) (a b : ℝ), (-2 * a^n * b^n)^m + (3 * a^m * b^m)^n = a^6 * b^6 := by
  sorry

end existence_of_m_and_n_l71_7158


namespace difference_of_squares_l71_7114

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l71_7114


namespace inscribed_cube_volume_l71_7180

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube. -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  ∃ (inner_cube_volume : ℝ),
    inner_cube_volume = 192 * Real.sqrt 3 ∧
    inner_cube_volume = (outer_cube_edge / Real.sqrt 3) ^ 3 :=
by sorry

end inscribed_cube_volume_l71_7180


namespace least_addition_for_divisibility_l71_7156

theorem least_addition_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x ≤ 3 ∧ (1202 + x) % 4 = 0 ∧ ∀ (y : ℕ), y < x → (1202 + y) % 4 ≠ 0 :=
sorry

end least_addition_for_divisibility_l71_7156


namespace product_of_numbers_l71_7163

theorem product_of_numbers (x y : ℝ) : x + y = 30 → x^2 + y^2 = 840 → x * y = 30 := by
  sorry

end product_of_numbers_l71_7163


namespace system_is_linear_l71_7146

-- Define a linear equation in two variables
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

-- Define the system of equations
def equation1 (x y : ℝ) : ℝ := x + y - 2
def equation2 (x y : ℝ) : ℝ := x - 2 * y

-- Theorem statement
theorem system_is_linear :
  is_linear_equation equation1 ∧ is_linear_equation equation2 :=
sorry

end system_is_linear_l71_7146


namespace syllogism_correctness_l71_7137

theorem syllogism_correctness : 
  (∀ n : ℕ, (n : ℤ) = n) →  -- All natural numbers are integers
  (4 : ℕ) = 4 →             -- 4 is a natural number
  (4 : ℤ) = 4               -- Therefore, 4 is an integer
  := by sorry

end syllogism_correctness_l71_7137


namespace quadratic_root_proof_l71_7103

theorem quadratic_root_proof : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ 
  (a * (2 - Real.sqrt 7)^2 + b * (2 - Real.sqrt 7) + c = 0) ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 - 4*x - 3 = 0) :=
sorry

end quadratic_root_proof_l71_7103


namespace rectangular_container_volume_l71_7171

theorem rectangular_container_volume 
  (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 := by
sorry

end rectangular_container_volume_l71_7171


namespace gardening_project_total_cost_l71_7124

def gardening_project_cost (rose_bushes : ℕ) (rose_bush_cost : ℕ) 
  (gardener_hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ)
  (soil_volume : ℕ) (soil_cost_per_unit : ℕ) : ℕ :=
  rose_bushes * rose_bush_cost + 
  gardener_hourly_rate * hours_per_day * days_worked +
  soil_volume * soil_cost_per_unit

theorem gardening_project_total_cost : 
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end gardening_project_total_cost_l71_7124


namespace expression_evaluation_l71_7199

theorem expression_evaluation (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  ((3*m + n) * (m - n) - (2*m - n)^2 + (m - 2*n) * (m + 2*n)) / (n / 2) = 28 :=
by sorry

end expression_evaluation_l71_7199


namespace divisibility_by_48_l71_7125

theorem divisibility_by_48 (a b c : ℤ) (h1 : a < c) (h2 : a^2 + c^2 = 2*b^2) :
  ∃ k : ℤ, c^2 - a^2 = 48 * k := by
sorry

end divisibility_by_48_l71_7125


namespace not_convex_pentagon_with_diagonals_l71_7112

/-- A list of segment lengths -/
def segment_lengths : List ℝ := [2, 3, 5, 7, 8, 9, 10, 11, 13, 15]

/-- A predicate that checks if three real numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- A predicate that checks if a list of real numbers can form a convex pentagon with sides and diagonals -/
def is_convex_pentagon_with_diagonals (lengths : List ℝ) : Prop :=
  lengths.length = 10 ∧
  ∀ (a b c : ℝ), a ∈ lengths → b ∈ lengths → c ∈ lengths →
    a ≠ b ∧ b ≠ c ∧ a ≠ c → is_triangle a b c

/-- Theorem stating that the given segment lengths cannot form a convex pentagon with diagonals -/
theorem not_convex_pentagon_with_diagonals :
  ¬ is_convex_pentagon_with_diagonals segment_lengths := by
  sorry

end not_convex_pentagon_with_diagonals_l71_7112


namespace complex_fraction_simplification_l71_7107

/-- Given that i² = -1, prove that (3 + 2i) / (4 - 5i) = 2/41 + (23/41)i -/
theorem complex_fraction_simplification :
  (Complex.I : ℂ)^2 = -1 →
  (3 + 2 * Complex.I) / (4 - 5 * Complex.I) = (2 : ℂ) / 41 + (23 : ℂ) / 41 * Complex.I :=
by sorry

end complex_fraction_simplification_l71_7107


namespace probability_all_heads_or_some_tails_l71_7111

def num_coins : ℕ := 5

def coin_outcomes : ℕ := 2

def all_outcomes : ℕ := coin_outcomes ^ num_coins

theorem probability_all_heads_or_some_tails :
  (1 : ℚ) / all_outcomes + ((all_outcomes - 1 : ℕ) : ℚ) / all_outcomes = 1 :=
by sorry

end probability_all_heads_or_some_tails_l71_7111


namespace work_completion_theorem_l71_7145

/-- Represents the work completion scenario -/
structure WorkCompletion where
  initial_men : ℕ
  initial_hours_per_day : ℕ
  initial_days : ℕ
  new_men : ℕ
  new_days : ℕ

/-- Calculates the hours per day for the new workforce -/
def hours_per_day (w : WorkCompletion) : ℚ :=
  (w.initial_men * w.initial_hours_per_day * w.initial_days : ℚ) / (w.new_men * w.new_days)

theorem work_completion_theorem (w : WorkCompletion) 
    (h1 : w.initial_men = 10)
    (h2 : w.initial_hours_per_day = 7)
    (h3 : w.initial_days = 18)
    (h4 : w.new_days = 12)
    (h5 : w.new_men > 10) :
    hours_per_day w = 1260 / (12 * w.new_men) := by
  sorry

end work_completion_theorem_l71_7145


namespace internal_angle_pentadecagon_is_156_l71_7194

/-- The measure of one internal angle of a regular pentadecagon -/
def internal_angle_pentadecagon : ℝ :=
  156

/-- The number of sides in a pentadecagon -/
def pentadecagon_sides : ℕ := 15

theorem internal_angle_pentadecagon_is_156 :
  internal_angle_pentadecagon = 156 :=
by
  sorry

#check internal_angle_pentadecagon_is_156

end internal_angle_pentadecagon_is_156_l71_7194


namespace certain_number_problem_l71_7191

theorem certain_number_problem (x : ℝ) : 
  3.6 * x * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 0.48 := by
  sorry

end certain_number_problem_l71_7191


namespace angle_not_in_second_quadrant_l71_7192

def is_in_second_quadrant (angle : ℝ) : Prop :=
  let normalized_angle := angle % 360
  90 < normalized_angle ∧ normalized_angle ≤ 180

theorem angle_not_in_second_quadrant :
  is_in_second_quadrant 160 ∧
  is_in_second_quadrant 480 ∧
  is_in_second_quadrant (-960) ∧
  ¬ is_in_second_quadrant 1530 :=
by sorry

end angle_not_in_second_quadrant_l71_7192


namespace line_equation_through_point_with_slope_l71_7134

theorem line_equation_through_point_with_slope (x y : ℝ) :
  let point : ℝ × ℝ := (-2, 1)
  let slope : ℝ := Real.tan (135 * π / 180)
  (x - point.1) * slope = y - point.2 →
  x + y + 1 = 0 := by
  sorry

end line_equation_through_point_with_slope_l71_7134


namespace sum_equals_two_thirds_l71_7109

theorem sum_equals_two_thirds : 
  let original_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
  let remaining_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/18
  remaining_sum = 2/3 := by sorry

end sum_equals_two_thirds_l71_7109


namespace product_of_pairs_l71_7177

/-- Given three pairs of real numbers satisfying specific equations, 
    their product in a certain form equals a specific value -/
theorem product_of_pairs (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (eq1₁ : x₁^3 - 3*x₁*y₁^2 = 2007)
  (eq2₁ : y₁^3 - 3*x₁^2*y₁ = 2006)
  (eq1₂ : x₂^3 - 3*x₂*y₂^2 = 2007)
  (eq2₂ : y₂^3 - 3*x₂^2*y₂ = 2006)
  (eq1₃ : x₃^3 - 3*x₃*y₃^2 = 2007)
  (eq2₃ : y₃^3 - 3*x₃^2*y₃ = 2006) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/1003.5 := by
  sorry

end product_of_pairs_l71_7177


namespace rationalize_denominator_l71_7190

theorem rationalize_denominator :
  5 / (2 + Real.sqrt 5) = -10 + 5 * Real.sqrt 5 := by
  sorry

end rationalize_denominator_l71_7190


namespace polygon_sides_l71_7154

theorem polygon_sides (sum_interior_angles : ℕ) : sum_interior_angles = 1440 → ∃ n : ℕ, n = 10 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end polygon_sides_l71_7154


namespace counterfeit_coin_findable_l71_7174

/-- Represents the type of scale: regular or magical -/
inductive ScaleType
| Regular
| Magical

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftHeavier
| RightHeavier
| Equal

/-- Represents a coin -/
structure Coin := (id : Nat)

/-- Represents a weighing action -/
structure Weighing := 
  (left : List Coin)
  (right : List Coin)

/-- Represents the state of the problem -/
structure ProblemState :=
  (coins : List Coin)
  (counterfeitCoin : Coin)
  (scaleType : ScaleType)

/-- Function to perform a weighing -/
def performWeighing (state : ProblemState) (w : Weighing) : WeighingResult :=
  sorry

/-- Function representing a strategy to find the counterfeit coin -/
def findCounterfeitStrategy : ProblemState → List Weighing → Option Coin :=
  sorry

/-- Theorem stating that it's possible to find the counterfeit coin in 3 weighings -/
theorem counterfeit_coin_findable :
  ∀ (coins : List Coin) (counterfeitCoin : Coin) (scaleType : ScaleType),
    coins.length = 12 →
    counterfeitCoin ∈ coins →
    ∃ (strategy : List Weighing),
      strategy.length ≤ 3 ∧
      (findCounterfeitStrategy ⟨coins, counterfeitCoin, scaleType⟩ strategy = some counterfeitCoin) :=
sorry

end counterfeit_coin_findable_l71_7174


namespace problem_1_problem_2_l71_7108

-- Problem 1
theorem problem_1 (t : ℝ) : 
  (∀ x : ℝ, |2*x + t| - t ≤ 8 ↔ -5 ≤ x ∧ x ≤ 4) → t = 1 :=
sorry

-- Problem 2
theorem problem_2 (x y z : ℝ) :
  x^2 + (1/4)*y^2 + (1/9)*z^2 = 2 →
  x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end problem_1_problem_2_l71_7108


namespace units_digit_of_seven_pow_five_cubed_l71_7155

theorem units_digit_of_seven_pow_five_cubed : 7^(5^3) % 10 = 7 := by
  sorry

end units_digit_of_seven_pow_five_cubed_l71_7155


namespace johns_work_days_l71_7178

/-- Proves that John drives to work 5 days a week given his car's efficiency,
    distance to work, leisure travel, and weekly gas usage. -/
theorem johns_work_days (efficiency : ℝ) (distance_to_work : ℝ) (leisure_miles : ℝ) (gas_usage : ℝ)
    (h1 : efficiency = 30)
    (h2 : distance_to_work = 20)
    (h3 : leisure_miles = 40)
    (h4 : gas_usage = 8) :
    (gas_usage * efficiency - leisure_miles) / (2 * distance_to_work) = 5 := by
  sorry

end johns_work_days_l71_7178


namespace algebra_test_male_students_l71_7184

/-- Proves that given the conditions of the algebra test problem, the number of male students is 8 -/
theorem algebra_test_male_students
  (total_average : ℝ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ)
  (h_total_average : total_average = 90)
  (h_male_average : male_average = 83)
  (h_female_average : female_average = 92)
  (h_female_count : female_count = 28) :
  ∃ (male_count : ℕ),
    (male_count : ℝ) * male_average + (female_count : ℝ) * female_average =
      (male_count + female_count : ℝ) * total_average ∧
    male_count = 8 := by
  sorry

end algebra_test_male_students_l71_7184


namespace flag_puzzle_l71_7116

theorem flag_puzzle (x : ℝ) : 
  (8 * 5 : ℝ) + (10 * 7 : ℝ) + (x * 5 : ℝ) = (15 * 9 : ℝ) → x = 5 := by
sorry

end flag_puzzle_l71_7116


namespace number_problem_l71_7101

theorem number_problem (x : ℝ) : 0.3 * x - 70 = 20 → x = 300 := by
  sorry

end number_problem_l71_7101


namespace loss_calculation_l71_7168

/-- Calculates the loss for the investor with larger capital -/
def loss_larger_investor (total_loss : ℚ) : ℚ :=
  (9 / 10) * total_loss

theorem loss_calculation (total_loss : ℚ) (pyarelal_loss : ℚ) 
  (h1 : total_loss = 900) 
  (h2 : pyarelal_loss = loss_larger_investor total_loss) : 
  pyarelal_loss = 810 := by
  sorry

end loss_calculation_l71_7168


namespace cloth_sale_quantity_l71_7120

/-- Proves that the number of metres of cloth sold is 300 given the specified conditions --/
theorem cloth_sale_quantity (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) :
  total_selling_price = 18000 →
  loss_per_metre = 5 →
  cost_price_per_metre = 65 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 300 := by
  sorry

end cloth_sale_quantity_l71_7120


namespace moon_arrangements_count_l71_7197

/-- The number of distinct arrangements of letters in "MOON" -/
def moon_arrangements : ℕ := 12

/-- The total number of letters in "MOON" -/
def total_letters : ℕ := 4

/-- The number of times 'O' appears in "MOON" -/
def o_count : ℕ := 2

/-- Theorem stating that the number of distinct arrangements of letters in "MOON" is 12 -/
theorem moon_arrangements_count : 
  moon_arrangements = (total_letters.factorial) / (o_count.factorial) := by
  sorry

end moon_arrangements_count_l71_7197


namespace boat_upstream_speed_l71_7132

/-- Proves that given a boat with a speed of 7 km/hr in still water and a
    downstream speed of 10 km/hr, the upstream speed of the boat is 4 km/hr. -/
theorem boat_upstream_speed
  (still_water_speed : ℝ)
  (downstream_speed : ℝ)
  (h1 : still_water_speed = 7)
  (h2 : downstream_speed = 10) :
  still_water_speed - (downstream_speed - still_water_speed) = 4 := by
  sorry


end boat_upstream_speed_l71_7132


namespace red_balls_count_l71_7135

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → 
  prob = 1 / 21 →
  ∃ (red : ℕ), red ≤ total ∧ 
    (red : ℚ) / total * (red - 1) / (total - 1) = prob ∧
    red = 5 :=
by sorry

end red_balls_count_l71_7135


namespace polynomial_division_theorem_l71_7189

theorem polynomial_division_theorem (x : ℝ) :
  4 * x^4 - 3 * x^3 + 6 * x^2 - 9 * x + 3 = 
  (x + 2) * (4 * x^3 - 11 * x^2 + 28 * x - 65) + 133 := by
  sorry

end polynomial_division_theorem_l71_7189


namespace infinitely_many_divisors_l71_7164

theorem infinitely_many_divisors (a : ℕ) :
  Set.Infinite {n : ℕ | n ∣ a^(n - a + 1) - 1} :=
sorry

end infinitely_many_divisors_l71_7164


namespace stratified_sampling_car_models_l71_7100

/-- Represents the number of units to sample from a stratum in stratified sampling -/
def stratified_sample_size (stratum_size : ℕ) (total_population : ℕ) (total_sample : ℕ) : ℕ :=
  (stratum_size * total_sample) / total_population

/-- Theorem stating the correct sample sizes for the given problem -/
theorem stratified_sampling_car_models :
  let model1_size : ℕ := 1200
  let model2_size : ℕ := 6000
  let model3_size : ℕ := 2000
  let total_population : ℕ := model1_size + model2_size + model3_size
  let total_sample : ℕ := 46
  stratified_sample_size model1_size total_population total_sample = 6 ∧
  stratified_sample_size model2_size total_population total_sample = 30 ∧
  stratified_sample_size model3_size total_population total_sample = 10 :=
by sorry

end stratified_sampling_car_models_l71_7100


namespace hamburger_combinations_l71_7117

-- Define the number of patty options
def patty_options : Nat := 4

-- Define the number of condiments
def num_condiments : Nat := 9

-- Theorem statement
theorem hamburger_combinations :
  (patty_options * 2^num_condiments) = 2048 := by
  sorry

end hamburger_combinations_l71_7117


namespace percentage_equality_l71_7181

theorem percentage_equality :
  ∃! k : ℚ, (k / 100) * 25 = (20 / 100) * 30 := by sorry

end percentage_equality_l71_7181


namespace subtraction_with_division_l71_7142

theorem subtraction_with_division : 5100 - (102 / 20.4) = 5095 := by
  sorry

end subtraction_with_division_l71_7142


namespace circumscribed_sphere_surface_area_l71_7175

theorem circumscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 1) :
  let sphere_radius := (Real.sqrt 3 / 2) * cube_edge
  4 * Real.pi * sphere_radius ^ 2 = 3 * Real.pi :=
by sorry

end circumscribed_sphere_surface_area_l71_7175


namespace sum_of_roots_l71_7173

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 2026*x = 2023)
  (hy : y^3 + 6*y^2 + 2035*y = -4053) : 
  x + y = -1 := by
sorry

end sum_of_roots_l71_7173


namespace satellite_units_l71_7104

/-- Represents a satellite with modular units and sensors. -/
structure Satellite where
  units : ℕ  -- Number of modular units
  non_upgraded_per_unit : ℕ  -- Number of non-upgraded sensors per unit
  total_upgraded : ℕ  -- Total number of upgraded sensors

/-- The conditions given in the problem. -/
def satellite_conditions (s : Satellite) : Prop :=
  -- Condition 2: Non-upgraded sensors per unit is 1/8 of total upgraded
  s.non_upgraded_per_unit = s.total_upgraded / 8 ∧
  -- Condition 3: 25% of all sensors are upgraded
  s.total_upgraded = (s.units * s.non_upgraded_per_unit + s.total_upgraded) / 4

/-- The theorem stating that a satellite satisfying the given conditions has 24 units. -/
theorem satellite_units (s : Satellite) (h : satellite_conditions s) : s.units = 24 := by
  sorry


end satellite_units_l71_7104


namespace expression_factorization_l71_7149

theorem expression_factorization (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (a*b + a*c + b*c) :=
by sorry

end expression_factorization_l71_7149


namespace tangent_slope_and_sum_inequality_l71_7193

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem tangent_slope_and_sum_inequality
  (a : ℝ)
  (h1 : (deriv (f a)) 0 = -1)
  (x₁ x₂ : ℝ)
  (h2 : x₁ < Real.log 2)
  (h3 : x₂ > Real.log 2)
  (h4 : f a x₁ = f a x₂) :
  x₁ + x₂ < 2 * Real.log 2 :=
by sorry

end tangent_slope_and_sum_inequality_l71_7193


namespace best_fit_model_l71_7151

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Determines if one model has a better fit than another based on R² -/
def better_fit (m1 m2 : RegressionModel) : Prop :=
  m1.r_squared > m2.r_squared

theorem best_fit_model (m1 m2 m3 m4 : RegressionModel)
  (h1 : m1.r_squared = 0.87)
  (h2 : m2.r_squared = 0.97)
  (h3 : m3.r_squared = 0.50)
  (h4 : m4.r_squared = 0.25) :
  better_fit m2 m1 ∧ better_fit m2 m3 ∧ better_fit m2 m4 := by
  sorry

end best_fit_model_l71_7151


namespace triangle_third_side_bounds_l71_7121

theorem triangle_third_side_bounds (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  let c_min := Int.ceil (max (b - a) (a - b))
  let c_max := Int.floor (a + b - 1)
  (c_min = 5 ∧ c_max = 17) := by sorry

end triangle_third_side_bounds_l71_7121


namespace binary_multiplication_example_l71_7159

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinary (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
    toBinary n

theorem binary_multiplication_example :
  let a := [false, true, true, false, true, true]  -- 110110₂
  let b := [true, true, true]  -- 111₂
  let result := [false, true, false, false, true, false, false, true]  -- 10010010₂
  binaryToNat a * binaryToNat b = binaryToNat result := by
  sorry

end binary_multiplication_example_l71_7159


namespace sum_seven_consecutive_integers_l71_7153

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end sum_seven_consecutive_integers_l71_7153


namespace max_product_sum_300_l71_7150

theorem max_product_sum_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500 := by
  sorry

end max_product_sum_300_l71_7150


namespace area_ratio_correct_l71_7105

/-- Represents a rectangle inscribed in a circle with a smaller rectangle inside it. -/
structure InscribedRectangles where
  /-- The ratio of the smaller rectangle's width to the larger rectangle's width -/
  x : ℝ
  /-- The ratio of the smaller rectangle's height to the larger rectangle's height -/
  y : ℝ
  /-- Constraint ensuring the smaller rectangle's vertices lie on the circle -/
  h_circle : 4 * y^2 + 4 * y + x^2 = 1

/-- The area ratio of the smaller rectangle to the larger rectangle -/
def areaRatio (r : InscribedRectangles) : ℝ := r.x * r.y

theorem area_ratio_correct (r : InscribedRectangles) : 
  areaRatio r = r.x * r.y := by sorry

end area_ratio_correct_l71_7105


namespace alex_earnings_l71_7165

/-- Alex's work hours and earnings problem -/
theorem alex_earnings (hours_week3 : ℕ) (hours_difference : ℕ) (earnings_difference : ℕ) :
  hours_week3 = 28 →
  hours_difference = 10 →
  earnings_difference = 80 →
  (hours_week3 - hours_difference) * (earnings_difference / hours_difference) +
  hours_week3 * (earnings_difference / hours_difference) = 368 := by
  sorry

end alex_earnings_l71_7165


namespace family_reunion_count_l71_7115

/-- The number of people at a family reunion -/
def family_reunion_attendance (male_adults female_adults children : ℕ) : ℕ :=
  male_adults + female_adults + children

/-- Theorem stating the total number of people at the family reunion -/
theorem family_reunion_count :
  ∃ (male_adults female_adults children : ℕ),
    male_adults = 100 ∧
    female_adults = male_adults + 50 ∧
    children = 2 * (male_adults + female_adults) ∧
    family_reunion_attendance male_adults female_adults children = 750 :=
by
  sorry


end family_reunion_count_l71_7115


namespace profit_growth_equation_l71_7136

/-- Represents the profit growth of a supermarket over a 2-month period -/
theorem profit_growth_equation (initial_profit : ℝ) (final_profit : ℝ) (growth_rate : ℝ) :
  initial_profit = 5000 →
  final_profit = 7200 →
  initial_profit * (1 + growth_rate)^2 = final_profit :=
by sorry

end profit_growth_equation_l71_7136


namespace kittens_count_l71_7133

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens in terms of puppies
def num_kittens : ℕ := 2 * num_puppies + 14

-- Theorem to prove
theorem kittens_count : num_kittens = 78 := by
  sorry

end kittens_count_l71_7133


namespace range_of_t_l71_7187

/-- Given a set A containing 1 and a real number t, prove that the range of t is all real numbers except 1. -/
theorem range_of_t (t : ℝ) (A : Set ℝ) (h : A = {1, t}) : 
  {x : ℝ | x ≠ 1} = {x : ℝ | ∃ (s : Set ℝ), s = {1, x} ∧ s = A} :=
by sorry

end range_of_t_l71_7187


namespace ellipse_symmetric_points_exist_l71_7113

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1/3

-- Define point Q
def Q : ℝ × ℝ := (0, 3)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for symmetry with respect to the line
def symmetric_points (A B : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), line P.1 P.2 ∧ 
    A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2

-- State the theorem
theorem ellipse_symmetric_points_exist : 
  ∃ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    symmetric_points A B ∧ 
    3 * dot_product (A.1 - Q.1, A.2 - Q.2) (B.1 - Q.1, B.2 - Q.2) = 32 :=
by
  sorry

end ellipse_symmetric_points_exist_l71_7113


namespace sin_15_cos_15_l71_7185

theorem sin_15_cos_15 : 
  (∀ θ : ℝ, Real.sin (2 * θ) = 2 * Real.sin θ * Real.cos θ) →
  Real.sin (30 * π / 180) = 1 / 2 →
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
sorry

end sin_15_cos_15_l71_7185


namespace equation_solutions_l71_7188

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 3 = 3 * (x + 1) ∧ x = -6) ∧
  (∃ x : ℚ, (1/2) * x - (9 * x - 2) / 6 - 2 = 0 ∧ x = -5/3) := by
  sorry

end equation_solutions_l71_7188


namespace swimming_distance_l71_7106

theorem swimming_distance (x : ℝ) 
  (h1 : x > 0)
  (h2 : (4 * x) / (5 * x) = 4 / 5)
  (h3 : (4 * x - 200) / (5 * x + 100) = 5 / 8) :
  4 * x = 1200 ∧ 5 * x = 1500 := by
  sorry

end swimming_distance_l71_7106


namespace inequality_proof_l71_7152

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x / y + y / z + z / x) / 3 ≥ 1) ∧
  (x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ (x / y + y / z + z / x)^2 / 3) ∧
  (x^2 / y^2 + y^2 / z^2 + z^2 / x^2 ≥ x / y + y / z + z / x) :=
by sorry

end inequality_proof_l71_7152


namespace ticket_sales_income_l71_7167

/-- Calculates the total income from ticket sales given the number of student and adult tickets sold and their respective prices. -/
def total_income (student_tickets : ℕ) (adult_tickets : ℕ) (student_price : ℚ) (adult_price : ℚ) : ℚ :=
  student_tickets * student_price + adult_tickets * adult_price

/-- Proves that the total income from selling 20 tickets, where 12 are student tickets at $2.00 each and 8 are adult tickets at $4.50 each, is equal to $60.00. -/
theorem ticket_sales_income :
  let student_tickets : ℕ := 12
  let adult_tickets : ℕ := 8
  let student_price : ℚ := 2
  let adult_price : ℚ := 9/2
  total_income student_tickets adult_tickets student_price adult_price = 60 := by
  sorry


end ticket_sales_income_l71_7167


namespace joans_savings_l71_7118

/-- The number of quarters Joan has saved --/
def num_quarters : ℕ := 6

/-- The value of one quarter in cents --/
def cents_per_quarter : ℕ := 25

/-- Theorem: The total value of Joan's quarters in cents --/
theorem joans_savings : num_quarters * cents_per_quarter = 150 := by
  sorry

end joans_savings_l71_7118


namespace problem_sampling_is_systematic_l71_7130

/-- Represents a sampling method -/
inductive SamplingMethod
| DrawingLots
| RandomNumberTable
| SystematicSampling
| Other

/-- Represents a high school with classes and student numbering -/
structure HighSchool where
  num_classes : Nat
  students_per_class : Nat
  selected_number : Nat

/-- Defines the conditions of the problem -/
def problem_conditions : HighSchool :=
  { num_classes := 12
  , students_per_class := 50
  , selected_number := 20 }

/-- Defines systematic sampling -/
def is_systematic_sampling (school : HighSchool) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.SystematicSampling ∧
  school.num_classes > 0 ∧
  school.students_per_class > 0 ∧
  school.selected_number > 0 ∧
  school.selected_number ≤ school.students_per_class

/-- Theorem stating that the sampling method in the problem is systematic sampling -/
theorem problem_sampling_is_systematic :
  is_systematic_sampling problem_conditions SamplingMethod.SystematicSampling :=
by
  sorry

end problem_sampling_is_systematic_l71_7130


namespace multiple_without_zero_digit_l71_7140

theorem multiple_without_zero_digit (n : ℕ) (hn : n % 10 ≠ 0) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ ∀ d : ℕ, d < 10 → (m / 10^d) % 10 ≠ 0 := by
  sorry

end multiple_without_zero_digit_l71_7140


namespace brent_candy_count_l71_7195

/-- The number of pieces of candy Brent has left after trick-or-treating and giving some away. -/
def candy_left : ℕ :=
  let kit_kat := 5
  let hershey := 3 * kit_kat
  let nerds := 8
  let lollipops := 11
  let baby_ruth := 10
  let reeses := baby_ruth / 2
  let total := kit_kat + hershey + nerds + lollipops + baby_ruth + reeses
  let given_away := 5
  total - given_away

/-- Theorem stating that Brent has 49 pieces of candy left. -/
theorem brent_candy_count : candy_left = 49 := by
  sorry

end brent_candy_count_l71_7195


namespace root_comparison_l71_7147

theorem root_comparison (m n : ℕ) : 
  min ((n : ℝ) ^ (1 / m : ℝ)) ((m : ℝ) ^ (1 / n : ℝ)) ≤ (3 : ℝ) ^ (1 / 3 : ℝ) := by
  sorry

end root_comparison_l71_7147


namespace sum_odd_when_sum_of_squares_odd_l71_7144

theorem sum_odd_when_sum_of_squares_odd (n m : ℤ) (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end sum_odd_when_sum_of_squares_odd_l71_7144


namespace fraction_equals_seven_l71_7148

theorem fraction_equals_seven (x : ℝ) (h : x = 2) : (x^4 + 6*x^2 + 9) / (x^2 + 3) = 7 := by
  sorry

end fraction_equals_seven_l71_7148


namespace min_blocking_tiles_18x8_l71_7138

/-- Represents an L-shaped tile that covers exactly 3 squares --/
structure LTile :=
  (covers : Nat)

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Calculates the total number of squares on the chessboard --/
def totalSquares (board : Chessboard) : Nat :=
  board.rows * board.cols

/-- Defines the minimum number of L-tiles needed to block further placement --/
def minBlockingTiles (board : Chessboard) (tile : LTile) : Nat :=
  11

/-- Main theorem: The minimum number of L-tiles to block further placement on an 18x8 board is 11 --/
theorem min_blocking_tiles_18x8 :
  let board : Chessboard := ⟨18, 8⟩
  let tile : LTile := ⟨3⟩
  minBlockingTiles board tile = 11 := by
  sorry

end min_blocking_tiles_18x8_l71_7138


namespace count_between_multiples_l71_7141

def multiples_of_4 : List Nat := List.filter (fun n => n % 4 = 0) (List.range 100)

def fifth_from_left : Nat := multiples_of_4[4]

def eighth_from_right : Nat := multiples_of_4[multiples_of_4.length - 8]

theorem count_between_multiples :
  (List.filter (fun n => n > fifth_from_left ∧ n < eighth_from_right) multiples_of_4).length = 11 := by
  sorry

end count_between_multiples_l71_7141


namespace tan_G_in_right_triangle_l71_7196

theorem tan_G_in_right_triangle (GH FG : ℝ) (h_right_triangle : GH^2 + FG^2 = 25^2)
  (h_GH : GH = 20) (h_FG : FG = 25) : Real.tan (Real.arcsin (GH / FG)) = 3/4 := by
  sorry

end tan_G_in_right_triangle_l71_7196


namespace friend_reading_time_l71_7131

/-- Proves that given the conditions on reading speeds and time, 
    the friend's reading time for one volume is 0.3 hours -/
theorem friend_reading_time 
  (my_speed : ℝ) 
  (friend_speed : ℝ) 
  (my_time_two_volumes : ℝ) 
  (h1 : my_speed = (1 / 5) * friend_speed) 
  (h2 : my_time_two_volumes = 3) : 
  (my_time_two_volumes / 2) / 5 = 0.3 := by
  sorry

end friend_reading_time_l71_7131


namespace lemonade_sales_difference_l71_7110

/-- 
Given Stanley's and Carl's hourly lemonade sales rates and a fixed time period,
prove the difference in their total sales.
-/
theorem lemonade_sales_difference 
  (stanley_rate : ℕ) 
  (carl_rate : ℕ) 
  (time_period : ℕ) 
  (h1 : stanley_rate = 4)
  (h2 : carl_rate = 7)
  (h3 : time_period = 3) :
  carl_rate * time_period - stanley_rate * time_period = 9 := by
  sorry

#check lemonade_sales_difference

end lemonade_sales_difference_l71_7110


namespace order_of_expressions_l71_7122

theorem order_of_expressions : 
  let a : ℝ := (4 : ℝ) ^ (1/10)
  let b : ℝ := Real.log 0.1 / Real.log 4
  let c : ℝ := (0.4 : ℝ) ^ (1/5)
  a > c ∧ c > b := by sorry

end order_of_expressions_l71_7122


namespace parametric_equation_of_lineL_l71_7182

/-- A line in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (l : Line2D) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧ p.2 = l.point.2 + t * l.direction.2

/-- The line passing through (3, 5) and parallel to (4, 2) -/
def lineL : Line2D :=
  { point := (3, 5)
    direction := (4, 2) }

/-- Theorem: The parametric equation (x - 3)/4 = (y - 5)/2 represents lineL -/
theorem parametric_equation_of_lineL :
  ∀ x y : ℝ, pointOnLine lineL (x, y) ↔ (x - 3) / 4 = (y - 5) / 2 := by
  sorry

end parametric_equation_of_lineL_l71_7182


namespace sector_area_l71_7139

/-- Given a sector with perimeter 16 cm and central angle 2 radians, its area is 16 cm² -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 16 → central_angle = 2 → area = (1/2) * central_angle * ((perimeter / (2 + central_angle))^2) → area = 16 := by
  sorry

end sector_area_l71_7139


namespace sum_areas_circles_6_8_10_triangle_l71_7162

/-- Given a 6-8-10 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 56π. -/
theorem sum_areas_circles_6_8_10_triangle : 
  ∃ (α β γ : ℝ),
    α + β = 6 ∧
    α + γ = 8 ∧
    β + γ = 10 ∧
    α > 0 ∧ β > 0 ∧ γ > 0 →
    π * (α^2 + β^2 + γ^2) = 56 * π := by
  sorry


end sum_areas_circles_6_8_10_triangle_l71_7162


namespace odd_even_function_sum_l71_7160

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_even_function_sum (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h_diff : ∀ x, f x - g x = 2 * x^3 + x^2 + 3) :
  f 2 + g 2 = 9 := by
sorry

end odd_even_function_sum_l71_7160


namespace expansion_properties_l71_7186

theorem expansion_properties (x : ℝ) : 
  let expansion := (x + 1) * (x + 2)^4
  ∃ (a b c d e f : ℝ), 
    expansion = a*x^5 + b*x^4 + c*x^3 + 56*x^2 + d*x + e ∧
    a + b + c + 56 + d + e = 162 :=
by sorry

end expansion_properties_l71_7186


namespace inclined_prism_volume_l71_7179

/-- The volume of an inclined prism with a parallelogram base and inclined lateral edge. -/
theorem inclined_prism_volume 
  (base_side1 base_side2 lateral_edge : ℝ) 
  (base_angle lateral_angle : ℝ) : 
  base_side1 = 3 →
  base_side2 = 6 →
  lateral_edge = 4 →
  base_angle = Real.pi / 4 →
  lateral_angle = Real.pi / 6 →
  (base_side1 * base_side2 * Real.sin base_angle) * (lateral_edge * Real.sin lateral_angle) = 18 * Real.sqrt 2 :=
by sorry

end inclined_prism_volume_l71_7179


namespace investment_difference_l71_7166

def emma_investment : ℝ := 300
def briana_investment : ℝ := 500
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def time_period : ℝ := 2

theorem investment_difference :
  briana_investment * briana_yield_rate * time_period - 
  emma_investment * emma_yield_rate * time_period = 10 := by
  sorry

end investment_difference_l71_7166


namespace complex_equation_solution_l71_7198

theorem complex_equation_solution (a : ℝ) : (Complex.mk 2 a) * (Complex.mk a (-2)) = 8 → a = 2 := by
  sorry

end complex_equation_solution_l71_7198


namespace berry_reading_problem_l71_7128

theorem berry_reading_problem (pages_per_day : ℕ) (days_in_week : ℕ) 
  (pages_sun : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) 
  (pages_fri : ℕ) (pages_sat : ℕ) :
  pages_per_day = 50 →
  days_in_week = 7 →
  pages_sun = 43 →
  pages_mon = 65 →
  pages_tue = 28 →
  pages_wed = 0 →
  pages_fri = 56 →
  pages_sat = 88 →
  ∃ pages_thu : ℕ, 
    pages_thu = pages_per_day * days_in_week - 
      (pages_sun + pages_mon + pages_tue + pages_wed + pages_fri + pages_sat) ∧
    pages_thu = 70 :=
by
  sorry

end berry_reading_problem_l71_7128


namespace araceli_luana_numbers_l71_7172

theorem araceli_luana_numbers : ∃ (a b c : ℕ), 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a) ∧
  a = 1 ∧ b = 9 ∧ c = 8 := by
sorry

end araceli_luana_numbers_l71_7172


namespace perfect_square_trinomial_l71_7102

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-3)*x + 16 = (a*x + b)^2) →
  (m = 7 ∨ m = -1) :=
by sorry

end perfect_square_trinomial_l71_7102


namespace exam_average_l71_7170

theorem exam_average (total_candidates : ℕ) (first_ten_avg : ℚ) (last_eleven_avg : ℚ) (eleventh_candidate_score : ℕ) :
  total_candidates = 22 →
  first_ten_avg = 55 →
  last_eleven_avg = 40 →
  eleventh_candidate_score = 66 →
  (((first_ten_avg * 10) + eleventh_candidate_score + (last_eleven_avg * 11 - eleventh_candidate_score)) / total_candidates : ℚ) = 45 := by
sorry

end exam_average_l71_7170


namespace inequality_implies_upper_bound_l71_7183

theorem inequality_implies_upper_bound (m : ℝ) : 
  (∀ x : ℝ, |x + 4| + |x + 8| ≥ m) → m ≤ 4 := by
  sorry

end inequality_implies_upper_bound_l71_7183
