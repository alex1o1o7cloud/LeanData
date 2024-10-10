import Mathlib

namespace trig_expression_equals_four_l3363_336355

theorem trig_expression_equals_four :
  1 / Real.cos (10 * π / 180) - Real.sqrt 3 / Real.sin (10 * π / 180) = 4 := by
  sorry

end trig_expression_equals_four_l3363_336355


namespace distance_between_vertices_l3363_336341

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) : 
  let vertex1 := (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c)
  let vertex2 := (- e / (2 * d), d * (- e / (2 * d))^2 + e * (- e / (2 * d)) + f)
  let distance := Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2)
  (a = 1 ∧ b = -4 ∧ c = 7 ∧ d = 1 ∧ e = 6 ∧ f = 20) → distance = Real.sqrt 89 :=
by sorry

end distance_between_vertices_l3363_336341


namespace extrema_of_squared_sum_l3363_336321

theorem extrema_of_squared_sum (a b c : ℝ) 
  (h : |a + b| + |b + c| + |c + a| = 8) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 16/3 ∧ 
    |x + y| + |y + z| + |z + x| = 8 ∧
    ∀ (p q r : ℝ), |p + q| + |q + r| + |r + p| = 8 → 
      p^2 + q^2 + r^2 ≥ 16/3) ∧
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 32 ∧ 
    |x + y| + |y + z| + |z + x| = 8 ∧
    ∀ (p q r : ℝ), |p + q| + |q + r| + |r + p| = 8 → 
      p^2 + q^2 + r^2 ≤ 32) :=
by sorry

end extrema_of_squared_sum_l3363_336321


namespace dividend_proof_l3363_336392

theorem dividend_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) : 
  dividend = 10918788 ∧ divisor = 12 ∧ quotient = 909899 → 
  dividend / divisor = quotient := by
  sorry

end dividend_proof_l3363_336392


namespace extremum_implies_slope_l3363_336385

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := (x - 2) * (x^2 + c)

-- State the theorem
theorem extremum_implies_slope (c : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f c x ≤ f c 2 ∨ f c x ≥ f c 2) →
  (deriv (f c)) 1 = -5 :=
sorry

end extremum_implies_slope_l3363_336385


namespace inscribed_circle_radius_l3363_336329

/-- The radius of an inscribed circle in a sector that is one-third of a larger circle -/
theorem inscribed_circle_radius (R : ℝ) (h : R = 6) :
  let sector_angle : ℝ := 2 * Real.pi / 3
  let inscribed_radius : ℝ := R * (Real.sqrt 2 - 1)
  inscribed_radius = R * (Real.sqrt 2 - 1) := by sorry

end inscribed_circle_radius_l3363_336329


namespace evaluate_expression_l3363_336362

theorem evaluate_expression : (27^24) / (81^12) = 3^24 := by
  sorry

end evaluate_expression_l3363_336362


namespace sum_of_repeating_decimals_l3363_336309

def repeating_decimal_to_fraction (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  let a := repeating_decimal_to_fraction 6
  let b := repeating_decimal_to_fraction 2
  let c := repeating_decimal_to_fraction 4
  let d := repeating_decimal_to_fraction 7
  a + b - c - d = -1/3 := by sorry

end sum_of_repeating_decimals_l3363_336309


namespace complete_solution_set_l3363_336386

def S : Set (ℕ × ℕ × ℕ) :=
  {(4, 33, 30), (32, 9, 30), (40, 9, 18), (12, 31, 30), (24, 23, 30), (4, 15, 22), (36, 15, 42)}

def is_solution (t : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t
  a^2 + b^2 + c^2 = 2005 ∧ 0 < a ∧ a ≤ b ∧ b ≤ c

theorem complete_solution_set :
  ∀ (a b c : ℕ), is_solution (a, b, c) ↔ (a, b, c) ∈ S := by
  sorry

end complete_solution_set_l3363_336386


namespace problem_statement_l3363_336338

theorem problem_statement (m n : ℝ) (h : |m - 3| + (n + 2)^2 = 0) : m + 2*n = -1 := by
  sorry

end problem_statement_l3363_336338


namespace convergence_of_beta_series_l3363_336342

theorem convergence_of_beta_series (α : ℕ → ℝ) (β : ℕ → ℝ) :
  (∀ n : ℕ, α n > 0) →
  (∀ n : ℕ, β n = (α n * n) / (n + 1)) →
  Summable α →
  Summable β := by
sorry

end convergence_of_beta_series_l3363_336342


namespace remainder_sum_l3363_336351

theorem remainder_sum (a b : ℤ) : 
  a % 45 = 37 → b % 30 = 9 → (a + b) % 15 = 1 := by
  sorry

end remainder_sum_l3363_336351


namespace table_free_sides_length_l3363_336307

theorem table_free_sides_length (length width : ℝ) : 
  length > 0 → 
  width > 0 → 
  length = 2 * width → 
  length * width = 128 → 
  length + 2 * width = 32 := by
sorry

end table_free_sides_length_l3363_336307


namespace problem_solution_l3363_336312

theorem problem_solution (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 := by
  sorry

end problem_solution_l3363_336312


namespace kindergarten_attendance_l3363_336347

/-- Calculates the total number of students present in two kindergarten sessions -/
def total_students (morning_registered : Nat) (morning_absent : Nat) 
                   (afternoon_registered : Nat) (afternoon_absent : Nat) : Nat :=
  (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)

/-- Theorem: The total number of students present over two kindergarten sessions is 42 -/
theorem kindergarten_attendance : 
  total_students 25 3 24 4 = 42 := by
  sorry

end kindergarten_attendance_l3363_336347


namespace approx_625_to_four_fifths_l3363_336349

-- Define the problem
theorem approx_625_to_four_fifths : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |Real.rpow 625 (4/5) - 238| < ε :=
sorry

end approx_625_to_four_fifths_l3363_336349


namespace elijah_coffee_pints_l3363_336395

-- Define the conversion rate from cups to pints
def cups_to_pints : ℚ := 1 / 2

-- Define the total amount of liquid consumed in cups
def total_liquid_cups : ℚ := 36

-- Define the amount of water Emilio drank in pints
def emilio_water_pints : ℚ := 9.5

-- Theorem statement
theorem elijah_coffee_pints :
  (total_liquid_cups * cups_to_pints) - emilio_water_pints = 8.5 := by
  sorry

end elijah_coffee_pints_l3363_336395


namespace multiply_121_54_l3363_336326

theorem multiply_121_54 : 121 * 54 = 6534 := by sorry

end multiply_121_54_l3363_336326


namespace max_value_of_expression_l3363_336303

theorem max_value_of_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (sum_eq_two : a + b + c = 2) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 2 ∧
    (a' * b' / (a' + b') + a' * c' / (a' + c') + b' * c' / (b' + c')) = 1 :=
by sorry

end max_value_of_expression_l3363_336303


namespace rearranged_rectangles_perimeter_l3363_336315

/-- The perimeter of a figure formed by rearranging two equal rectangles cut from a square --/
theorem rearranged_rectangles_perimeter (square_side : ℝ) : square_side = 100 → 
  let rectangle_width := square_side / 2
  let rectangle_length := square_side
  let perimeter := 3 * rectangle_length + 4 * rectangle_width
  perimeter = 500 := by
sorry


end rearranged_rectangles_perimeter_l3363_336315


namespace repeating_decimal_as_fraction_l3363_336334

-- Define the repeating decimal 0.4555...
def repeating_decimal : ℚ := 0.4555555555555555

-- Theorem statement
theorem repeating_decimal_as_fraction : repeating_decimal = 41 / 90 := by
  sorry

end repeating_decimal_as_fraction_l3363_336334


namespace hiking_team_participants_l3363_336327

theorem hiking_team_participants (total_gloves : ℕ) (gloves_per_participant : ℕ) : 
  total_gloves = 164 → gloves_per_participant = 2 → total_gloves / gloves_per_participant = 82 := by
  sorry

end hiking_team_participants_l3363_336327


namespace part_one_part_two_l3363_336305

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |k * x - 1|

-- Part I
theorem part_one (k : ℝ) :
  (∀ x, f k x ≤ 3 ↔ x ∈ Set.Icc (-2) 1) → k = -2 := by sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f 1 (x + 2) - f 1 (2 * x + 1) ≤ 3 - 2 * m) → m ≤ 1 := by sorry

end part_one_part_two_l3363_336305


namespace largest_four_digit_number_with_conditions_l3363_336370

/-- A function that checks if all digits in a number are different -/
def allDigitsDifferent (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- The theorem statement -/
theorem largest_four_digit_number_with_conditions :
  ∃ (n : ℕ),
    n = 8910 ∧
    1000 ≤ n ∧ n < 10000 ∧
    allDigitsDifferent n ∧
    n % 2 = 0 ∧ n % 5 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0 ∧
    ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ allDigitsDifferent m ∧
      m % 2 = 0 ∧ m % 5 = 0 ∧ m % 9 = 0 ∧ m % 11 = 0 → m ≤ n :=
by
  sorry

end largest_four_digit_number_with_conditions_l3363_336370


namespace min_value_expression_l3363_336359

theorem min_value_expression (x y : ℝ) : 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4 ≥ 3 := by
  sorry

end min_value_expression_l3363_336359


namespace limit_of_sequence_l3363_336310

/-- The limit of the sequence (√(n+1) - ∛(n³+1)) / (⁴√(n+1) - ⁵√(n⁵+1)) as n approaches infinity is 1 -/
theorem limit_of_sequence (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → 
    |((n + 1: ℝ)^(1/2) - (n^3 + 1 : ℝ)^(1/3)) / ((n + 1 : ℝ)^(1/4) - (n^5 + 1 : ℝ)^(1/5)) - 1| < ε :=
by
  sorry

#check limit_of_sequence

end limit_of_sequence_l3363_336310


namespace problem_solution_l3363_336317

theorem problem_solution (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 := by
  sorry

end problem_solution_l3363_336317


namespace gilbert_herb_plants_l3363_336365

/-- The number of herb plants Gilbert had at the end of spring -/
def herb_plants_at_end_of_spring : ℕ :=
  let initial_basil : ℕ := 3
  let initial_parsley : ℕ := 1
  let initial_mint : ℕ := 2
  let new_basil : ℕ := 1
  let eaten_mint : ℕ := 2
  (initial_basil + initial_parsley + initial_mint + new_basil) - eaten_mint

theorem gilbert_herb_plants : herb_plants_at_end_of_spring = 5 := by
  sorry

end gilbert_herb_plants_l3363_336365


namespace yonderland_license_plates_l3363_336369

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of non-zero digits (1-9) -/
def non_zero_digit_count : ℕ := 9

/-- The number of letters in a license plate -/
def letter_count : ℕ := 3

/-- The number of digits in a license plate -/
def digit_position_count : ℕ := 4

/-- The total number of valid license plates in Yonderland -/
def valid_license_plate_count : ℕ :=
  alphabet_size * (alphabet_size - 1) * (alphabet_size - 2) *
  non_zero_digit_count * digit_count^(digit_position_count - 1)

theorem yonderland_license_plates :
  valid_license_plate_count = 702000000 := by
  sorry

end yonderland_license_plates_l3363_336369


namespace decreasing_f_implies_a_leq_neg_five_l3363_336377

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a+1)*x + 2

-- State the theorem
theorem decreasing_f_implies_a_leq_neg_five (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -5 := by
  sorry

end decreasing_f_implies_a_leq_neg_five_l3363_336377


namespace opposite_side_of_five_times_five_l3363_336308

/-- A standard 6-sided die with opposite sides summing to 7 -/
structure StandardDie where
  sides : Fin 6 → Nat
  valid_range : ∀ i, sides i ∈ Finset.range 7 \ {0}
  opposite_sum : ∀ i, sides i + sides (5 - i) = 7

/-- The number of eyes on the opposite side of 5 multiplied by 5 is 10 -/
theorem opposite_side_of_five_times_five (d : StandardDie) :
  5 * d.sides (5 - 5) = 10 := by
  sorry

end opposite_side_of_five_times_five_l3363_336308


namespace max_value_of_even_quadratic_function_l3363_336328

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem max_value_of_even_quadratic_function (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  (∃ x ∈ Set.Icc (a - 1) (2 * a), ∀ y ∈ Set.Icc (a - 1) (2 * a), f a b y ≤ f a b x) →
  (∃ x ∈ Set.Icc (a - 1) (2 * a), f a b x = 31 / 27) :=
by sorry

end max_value_of_even_quadratic_function_l3363_336328


namespace geometric_identity_l3363_336387

theorem geometric_identity 
  (a b c p x : ℝ) 
  (h1 : a + b + c = 2 * p) 
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c)) 
  (h3 : c ≠ 0) : 
  b^2 - x^2 = (4 / c^2) * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end geometric_identity_l3363_336387


namespace expression_value_l3363_336366

theorem expression_value (a b : ℤ) (ha : a = 3) (hb : b = -2) :
  -a^2 - b^3 + a*b = -7 := by
  sorry

end expression_value_l3363_336366


namespace distance_to_circle_center_l3363_336350

/-- The distance from a point in polar coordinates to the center of a circle defined by a polar equation --/
theorem distance_to_circle_center (ρ₀ : ℝ) (θ₀ : ℝ) :
  let circle := fun θ => 2 * Real.cos θ
  let center_x := 1
  let center_y := 0
  let point_x := ρ₀ * Real.cos θ₀
  let point_y := ρ₀ * Real.sin θ₀
  (ρ₀ = 2 ∧ θ₀ = Real.pi / 3) →
  Real.sqrt ((point_x - center_x)^2 + (point_y - center_y)^2) = Real.sqrt 3 :=
by sorry

end distance_to_circle_center_l3363_336350


namespace sin_fourth_powers_sum_l3363_336311

theorem sin_fourth_powers_sum : 
  Real.sin (π / 8) ^ 4 + Real.sin (3 * π / 8) ^ 4 + 
  Real.sin (5 * π / 8) ^ 4 + Real.sin (7 * π / 8) ^ 4 = 3 / 2 := by
  sorry

end sin_fourth_powers_sum_l3363_336311


namespace expression_value_l3363_336332

theorem expression_value (b c a : ℤ) (h1 : b = 10) (h2 : c = 3) (h3 : a = 2 * b) :
  (a - (b - c)) - ((a - b) - c) = 6 := by
  sorry

end expression_value_l3363_336332


namespace cricket_average_score_l3363_336383

theorem cricket_average_score 
  (total_matches : ℕ) 
  (matches_set1 : ℕ) 
  (matches_set2 : ℕ) 
  (avg_score_set1 : ℝ) 
  (avg_score_set2 : ℝ) 
  (h1 : total_matches = matches_set1 + matches_set2)
  (h2 : matches_set1 = 2)
  (h3 : matches_set2 = 3)
  (h4 : avg_score_set1 = 20)
  (h5 : avg_score_set2 = 30) :
  (matches_set1 * avg_score_set1 + matches_set2 * avg_score_set2) / total_matches = 26 := by
  sorry

end cricket_average_score_l3363_336383


namespace peter_five_theorem_l3363_336346

theorem peter_five_theorem (N : ℕ+) :
  ∃ K : ℕ, ∀ k : ℕ, k ≥ K → (∃ d m n : ℕ, N * 5^k = 10^n * (10 * m + 5) + d ∧ d < 10^n) :=
sorry

end peter_five_theorem_l3363_336346


namespace quadratic_inequality_solution_range_l3363_336320

theorem quadratic_inequality_solution_range (m : ℝ) : 
  m > 0 ∧ 
  (∃ a b : ℤ, a ≠ b ∧ 
    (∀ x : ℝ, (2*x^2 - 2*m*x + m < 0) ↔ (a < x ∧ x < b)) ∧
    (∀ c : ℤ, (2*c^2 - 2*m*c + m < 0) → (c = a ∨ c = b)))
  → 
  8/3 < m ∧ m ≤ 18/5 :=
sorry

end quadratic_inequality_solution_range_l3363_336320


namespace min_bricks_for_cube_l3363_336356

/-- The width of a brick in centimeters -/
def brick_width : ℕ := 18

/-- The depth of a brick in centimeters -/
def brick_depth : ℕ := 12

/-- The height of a brick in centimeters -/
def brick_height : ℕ := 9

/-- The volume of a single brick in cubic centimeters -/
def brick_volume : ℕ := brick_width * brick_depth * brick_height

/-- The side length of the smallest cube that can be formed using the bricks -/
def cube_side_length : ℕ := Nat.lcm (Nat.lcm brick_width brick_depth) brick_height

/-- The volume of the smallest cube that can be formed using the bricks -/
def cube_volume : ℕ := cube_side_length ^ 3

/-- The theorem stating the minimum number of bricks required to make a cube -/
theorem min_bricks_for_cube : cube_volume / brick_volume = 24 := by
  sorry

end min_bricks_for_cube_l3363_336356


namespace student_multiplication_problem_l3363_336378

theorem student_multiplication_problem (x y : ℝ) : 
  x = 127 → x * y - 152 = 102 → y = 2 := by
sorry

end student_multiplication_problem_l3363_336378


namespace rational_square_roots_existence_l3363_336323

theorem rational_square_roots_existence : ∃ (x : ℚ), 
  3 < x ∧ x < 4 ∧ 
  ∃ (a b : ℚ), a^2 = x - 3 ∧ b^2 = x + 1 ∧
  x = 481 / 144 := by
  sorry

end rational_square_roots_existence_l3363_336323


namespace line_points_relation_l3363_336384

/-- 
Given a line with equation x = 6y + 5, 
if two points (m, n) and (m + Q, n + p) lie on this line, 
and p = 1/3, then Q = 2.
-/
theorem line_points_relation (m n Q p : ℝ) : 
  (m = 6 * n + 5) →
  (m + Q = 6 * (n + p) + 5) →
  (p = 1/3) →
  Q = 2 := by
  sorry

end line_points_relation_l3363_336384


namespace correct_average_l3363_336371

theorem correct_average (n : ℕ) (initial_avg wrong_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = n * 16 :=
by sorry

end correct_average_l3363_336371


namespace range_of_p_l3363_336354

-- Define the function p(x)
def p (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

-- State the theorem
theorem range_of_p :
  {y : ℝ | ∃ x ≥ 0, p x = y} = {y : ℝ | y ≥ 9} :=
sorry

end range_of_p_l3363_336354


namespace max_value_abc_l3363_336352

theorem max_value_abc (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  10 * a + 3 * b + 15 * c ≤ Real.sqrt (337 / 36) :=
by sorry

end max_value_abc_l3363_336352


namespace max_distance_with_tire_swap_l3363_336360

/-- Represents the maximum distance a tire can travel on the rear wheel before wearing out. -/
def rear_tire_limit : ℝ := 15000

/-- Represents the maximum distance a tire can travel on the front wheel before wearing out. -/
def front_tire_limit : ℝ := 25000

/-- Represents the maximum distance a truck can travel before all four tires are worn out,
    given that tires can be swapped between front and rear positions. -/
def max_truck_distance : ℝ := 18750

/-- Theorem stating that the maximum distance a truck can travel before all four tires
    are worn out is 18750 km, given the conditions on tire wear and the ability to swap tires. -/
theorem max_distance_with_tire_swap :
  max_truck_distance = 18750 :=
by sorry

end max_distance_with_tire_swap_l3363_336360


namespace prime_4k_plus_1_properties_l3363_336382

theorem prime_4k_plus_1_properties (k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_form : p = 4 * k + 1) :
  (∃ x : ℤ, (x^2 + 1) % p = 0) ∧
  (∃ r₁ r₂ s₁ s₂ : ℕ,
    r₁ < Real.sqrt p ∧ r₂ < Real.sqrt p ∧ s₁ < Real.sqrt p ∧ s₂ < Real.sqrt p ∧
    (r₁ ≠ r₂ ∨ s₁ ≠ s₂) ∧
    ∃ x : ℤ, (r₁ * x + s₁) % p = (r₂ * x + s₂) % p) ∧
  (∃ r₁ r₂ s₁ s₂ : ℕ,
    r₁ < Real.sqrt p ∧ r₂ < Real.sqrt p ∧ s₁ < Real.sqrt p ∧ s₂ < Real.sqrt p ∧
    p = (r₁ - r₂)^2 + (s₁ - s₂)^2) :=
by sorry

end prime_4k_plus_1_properties_l3363_336382


namespace fixed_point_on_line_l3363_336398

theorem fixed_point_on_line (m : ℝ) : (m - 1) * (7/2) - (m + 3) * (5/2) - (m - 11) = 0 := by
  sorry

end fixed_point_on_line_l3363_336398


namespace expression_evaluation_l3363_336316

theorem expression_evaluation :
  let x : ℚ := -1/4
  let y : ℚ := -1/2
  4*x*y - ((x^2 + 5*x*y - y^2) - (x^2 + 3*x*y - 2*y^2)) = 0 := by
sorry

end expression_evaluation_l3363_336316


namespace perimeter_ABCDE_l3363_336364

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_8 : dist A E = 8
axiom ED_eq_7 : dist E D = 7
axiom angle_AED_right : (E.1 - A.1) * (E.1 - D.1) + (E.2 - A.2) * (E.2 - D.2) = 0
axiom angle_ABC_right : (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0

-- Define the theorem
theorem perimeter_ABCDE :
  dist A B + dist B C + dist C D + dist D E + dist E A = 28 :=
sorry

end perimeter_ABCDE_l3363_336364


namespace tan_beta_value_l3363_336336

theorem tan_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 4/3) (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2/11 := by
  sorry

end tan_beta_value_l3363_336336


namespace log_20_over_27_not_calculable_l3363_336391

-- Define the given logarithms
def log5 : ℝ := 0.6990
def log3 : ℝ := 0.4771

-- Define a function to represent the ability to calculate a logarithm
def can_calculate (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), x = f log5 log3

-- Theorem statement
theorem log_20_over_27_not_calculable :
  ¬(can_calculate (Real.log (20/27))) ∧
  (can_calculate (Real.log 225)) ∧
  (can_calculate (Real.log 750)) ∧
  (can_calculate (Real.log 0.03)) ∧
  (can_calculate (Real.log 9)) :=
sorry

end log_20_over_27_not_calculable_l3363_336391


namespace number_of_elements_l3363_336343

theorem number_of_elements (incorrect_avg : ℝ) (correct_avg : ℝ) (difference : ℝ) : 
  incorrect_avg = 16 → correct_avg = 17 → difference = 10 →
  ∃ n : ℕ, n * correct_avg = n * incorrect_avg + difference ∧ n = 10 := by
  sorry

end number_of_elements_l3363_336343


namespace congruence_problem_l3363_336318

theorem congruence_problem : 
  ∀ n : ℤ, 10 ≤ n ∧ n ≤ 20 ∧ n % 7 = 12345 % 7 → n = 11 ∨ n = 18 := by
  sorry

end congruence_problem_l3363_336318


namespace cone_height_l3363_336353

/-- A cone with volume 8192π cubic inches and a vertical cross-section vertex angle of 90 degrees has a height equal to the cube root of 24576 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : ℝ) :
  V = 8192 * Real.pi ∧ θ = 90 → h = (24576 : ℝ) ^ (1/3) :=
by sorry

end cone_height_l3363_336353


namespace rectangular_box_volume_l3363_336380

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 40)
  (area2 : w * h = 15)
  (area3 : l * h = 12) :
  l * w * h = 60 := by
sorry

end rectangular_box_volume_l3363_336380


namespace tangent_line_at_one_symmetry_condition_extreme_values_condition_l3363_336339

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem tangent_line_at_one (a : ℝ) :
  a = -1 →
  ∃ m b : ℝ, ∀ x : ℝ, (m * (x - 1) + b = f a x) ∧ (m = -Real.log 2) :=
sorry

theorem symmetry_condition (a b : ℝ) :
  (∀ x : ℝ, f a (1/x) = f a (1/(2*b - x))) ↔ (a = 1/2 ∧ b = -1/2) :=
sorry

theorem extreme_values_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≤ f a x) ↔ (0 < a ∧ a < 1/2) :=
sorry

end tangent_line_at_one_symmetry_condition_extreme_values_condition_l3363_336339


namespace acme_profit_calculation_l3363_336389

def initial_outlay : ℝ := 12450
def manufacturing_cost_per_set : ℝ := 20.75
def selling_price_per_set : ℝ := 50
def marketing_expense_rate : ℝ := 0.05
def shipping_cost_rate : ℝ := 0.03
def number_of_sets : ℕ := 950

def revenue : ℝ := selling_price_per_set * number_of_sets
def total_manufacturing_cost : ℝ := initial_outlay + manufacturing_cost_per_set * number_of_sets
def additional_variable_costs : ℝ := (marketing_expense_rate + shipping_cost_rate) * revenue

def profit : ℝ := revenue - total_manufacturing_cost - additional_variable_costs

theorem acme_profit_calculation : profit = 11537.50 := by
  sorry

end acme_profit_calculation_l3363_336389


namespace ellipse_eccentricity_l3363_336319

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (y : ℝ) : 
  -- The ellipse equation
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1) →
  -- F₁ and F₂ are foci of the ellipse
  (∃ F₁ F₂ : ℝ × ℝ, F₁.1 = -c ∧ F₁.2 = 0 ∧ F₂.1 = c ∧ F₂.2 = 0) →
  -- Point P is on the line x = -a
  (∃ P : ℝ × ℝ, P.1 = -a ∧ P.2 = y) →
  -- |PF₁| = |F₁F₂|
  ((a - c)^2 + y^2 = (2*c)^2) →
  -- ∠PF₁F₂ = 120°
  (y / (a - c) = Real.sqrt 3) →
  -- The eccentricity is 1/2
  c / a = 1 / 2 := by
sorry

end ellipse_eccentricity_l3363_336319


namespace total_holes_dug_l3363_336397

-- Define Pearl's digging rate
def pearl_rate : ℚ := 4 / 7

-- Define Miguel's digging rate
def miguel_rate : ℚ := 2 / 3

-- Define the duration of work
def work_duration : ℕ := 21

-- Theorem to prove
theorem total_holes_dug : 
  ⌊(pearl_rate * work_duration) + (miguel_rate * work_duration)⌋ = 26 := by
  sorry


end total_holes_dug_l3363_336397


namespace problem_G10_1_l3363_336330

theorem problem_G10_1 (a : ℝ) : 
  (6 * Real.sqrt 3) / (3 * Real.sqrt 2 - 2 * Real.sqrt 3) = 3 * Real.sqrt a + 6 → a = 6 := by
  sorry

end problem_G10_1_l3363_336330


namespace M_equals_N_l3363_336302

def M : Set ℝ := {x | ∃ k : ℤ, x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 5 * Real.pi / 6 + 2 * k * Real.pi}

def N : Set ℝ := {x | ∃ k : ℤ, x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = -7 * Real.pi / 6 + 2 * k * Real.pi}

theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l3363_336302


namespace exists_three_integers_with_cube_product_l3363_336300

/-- A set of 9 distinct integers with prime factors at most 3 -/
def SetWithPrimeFactorsUpTo3 : Type :=
  { S : Finset ℕ // S.card = 9 ∧ ∀ n ∈ S, ∀ p : ℕ, Nat.Prime p → p ∣ n → p ≤ 3 }

/-- The theorem stating that there exist three distinct integers in S whose product is a perfect cube -/
theorem exists_three_integers_with_cube_product (S : SetWithPrimeFactorsUpTo3) :
  ∃ a b c : ℕ, a ∈ S.val ∧ b ∈ S.val ∧ c ∈ S.val ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ∃ k : ℕ, a * b * c = k^3 :=
sorry

end exists_three_integers_with_cube_product_l3363_336300


namespace tangent_parallel_points_tangent_equations_l3363_336373

/-- The function f(x) = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- The slope of the line y = 4x - 1 -/
def m : ℝ := 4

/-- The set of points where the tangent line is parallel to y = 4x - 1 -/
def tangent_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f' p.1 = m ∧ p.2 = f p.1}

/-- The equation of the tangent line at a point (a, f(a)) -/
def tangent_line (a : ℝ) (x y : ℝ) : Prop :=
  y - f a = f' a * (x - a)

theorem tangent_parallel_points :
  tangent_points = {(1, 0), (-1, -4)} :=
sorry

theorem tangent_equations (a : ℝ) (h : (a, f a) ∈ tangent_points) :
  (∀ x y, tangent_line a x y ↔ (4 * x - y - 4 = 0 ∨ 4 * x - y = 0)) :=
sorry

end tangent_parallel_points_tangent_equations_l3363_336373


namespace complement_intersection_equals_singleton_l3363_336393

-- Define the universal set I
def I : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 - 3 = p.1 - 2 ∧ p.1 ≠ 2}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_intersection_equals_singleton :
  (Set.compl M ∩ Set.compl N : Set (ℝ × ℝ)) = {(2, 3)} := by
  sorry

end complement_intersection_equals_singleton_l3363_336393


namespace gcd_of_specific_numbers_l3363_336374

theorem gcd_of_specific_numbers : Nat.gcd 33333 666666 = 3 := by
  sorry

end gcd_of_specific_numbers_l3363_336374


namespace diophantine_equation_solutions_l3363_336314

theorem diophantine_equation_solutions :
  ∀ m n : ℕ+,
    (1 : ℚ) / m + (1 : ℚ) / n - (1 : ℚ) / (m * n) = 2 / 5 ↔
    ((m = 3 ∧ n = 10) ∨ (m = 10 ∧ n = 3) ∨ (m = 4 ∧ n = 5) ∨ (m = 5 ∧ n = 4)) := by
  sorry

end diophantine_equation_solutions_l3363_336314


namespace fathers_age_l3363_336313

theorem fathers_age (f d : ℕ) (h1 : f / d = 4) (h2 : f + d + 10 = 50) : f = 32 := by
  sorry

end fathers_age_l3363_336313


namespace smallest_integer_negative_quadratic_l3363_336379

theorem smallest_integer_negative_quadratic :
  ∃ (n : ℤ), (∀ (m : ℤ), m^2 - 11*m + 28 < 0 → n ≤ m) ∧ (n^2 - 11*n + 28 < 0) ∧ n = 5 := by
  sorry

end smallest_integer_negative_quadratic_l3363_336379


namespace tomato_drying_l3363_336324

/-- Given an initial mass of tomatoes with a certain water content,
    calculate the final mass after water content reduction -/
theorem tomato_drying (initial_mass : ℝ) (initial_water_content : ℝ) (water_reduction : ℝ)
  (h1 : initial_mass = 1000)
  (h2 : initial_water_content = 0.99)
  (h3 : water_reduction = 0.04)
  : ∃ (final_mass : ℝ), final_mass = 200 := by
  sorry


end tomato_drying_l3363_336324


namespace smallest_congruent_number_l3363_336399

theorem smallest_congruent_number : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 7 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  (n = 281) := by
sorry

end smallest_congruent_number_l3363_336399


namespace expression_evaluation_l3363_336396

theorem expression_evaluation :
  let m : ℚ := 2
  let expr := (m^2 - 9) / (m^2 - 6*m + 9) / (1 - 2/(m - 3))
  expr = -5/3 := by sorry

end expression_evaluation_l3363_336396


namespace problem_solution_l3363_336358

theorem problem_solution : 45 / (7 - 3/4) = 36/5 := by
  sorry

end problem_solution_l3363_336358


namespace triangle_minimum_area_l3363_336306

theorem triangle_minimum_area :
  ∀ (S : ℝ), 
  (∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →
    a + b > c ∧ b + c > a ∧ c + a > b →
    (∃ (h : ℝ), h ≤ 1 ∧ S = 1/2 * (a * h)) →
    (∀ (w : ℝ), w < 1 → 
      ¬(∃ (h : ℝ), h ≤ w ∧ S = 1/2 * (a * h)))) →
  S ≥ 1 / Real.sqrt 3 :=
by sorry

end triangle_minimum_area_l3363_336306


namespace two_functions_satisfy_equation_l3363_336304

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := λ _ => 0

/-- The square function -/
def SquareFunction : ℝ → ℝ := λ x => x^2

/-- The main theorem stating that there are exactly two functions satisfying the equation -/
theorem two_functions_satisfy_equation :
  ∃! (s : Set (ℝ → ℝ)), 
    (∀ f ∈ s, SatisfiesFunctionalEquation f) ∧ 
    s = {ZeroFunction, SquareFunction} :=
  sorry

end two_functions_satisfy_equation_l3363_336304


namespace five_chicks_per_hen_l3363_336301

/-- Represents the poultry farm scenario --/
structure PoultryFarm where
  num_hens : ℕ
  hen_to_rooster_ratio : ℕ
  total_chickens : ℕ

/-- Calculates the number of chicks per hen --/
def chicks_per_hen (farm : PoultryFarm) : ℕ :=
  let num_roosters := farm.num_hens / farm.hen_to_rooster_ratio
  let num_adult_chickens := farm.num_hens + num_roosters
  let num_chicks := farm.total_chickens - num_adult_chickens
  num_chicks / farm.num_hens

/-- Theorem stating that for the given farm conditions, each hen has 5 chicks --/
theorem five_chicks_per_hen (farm : PoultryFarm) 
    (h1 : farm.num_hens = 12)
    (h2 : farm.hen_to_rooster_ratio = 3)
    (h3 : farm.total_chickens = 76) : 
  chicks_per_hen farm = 5 := by
  sorry

end five_chicks_per_hen_l3363_336301


namespace same_solution_implies_c_value_l3363_336376

theorem same_solution_implies_c_value (x c : ℝ) : 
  (3 * x + 8 = 5) ∧ (c * x - 15 = -3) → c = -12 := by
  sorry

end same_solution_implies_c_value_l3363_336376


namespace conversation_year_1941_l3363_336368

def is_valid_year (y : ℕ) : Prop := 1900 ≤ y ∧ y ≤ 1999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def swap_digits (n : ℕ) : ℕ :=
  ((n % 10) * 10) + (n / 10)

theorem conversation_year_1941 :
  ∃! (conv_year : ℕ) (elder_birth : ℕ) (younger_birth : ℕ),
    is_valid_year conv_year ∧
    is_valid_year elder_birth ∧
    is_valid_year younger_birth ∧
    elder_birth < younger_birth ∧
    conv_year - elder_birth = digit_sum younger_birth ∧
    conv_year - younger_birth = digit_sum elder_birth ∧
    swap_digits (conv_year - elder_birth) = conv_year - younger_birth ∧
    conv_year = 1941 :=
  sorry

end conversation_year_1941_l3363_336368


namespace simplify_and_evaluate_l3363_336390

theorem simplify_and_evaluate (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) :
  (1 - 1/a) / ((a^2 - 2*a + 1) / a) = 1 / (a - 1) ∧
  (1 - 1/2) / ((2^2 - 2*2 + 1) / 2) = 1 :=
by sorry

end simplify_and_evaluate_l3363_336390


namespace sum_of_87th_and_95th_odd_integers_l3363_336322

theorem sum_of_87th_and_95th_odd_integers : 
  (2 * 87 - 1) + (2 * 95 - 1) = 362 := by
  sorry

end sum_of_87th_and_95th_odd_integers_l3363_336322


namespace smallest_b_probability_l3363_336333

/-- The number of cards in the deck -/
def deckSize : ℕ := 40

/-- The probability that Carly and Fiona are on the same team when Carly picks card number b and Fiona picks card number b+7 -/
def q (b : ℕ) : ℚ :=
  let totalCombinations := (deckSize - 2).choose 2
  let lowerTeamCombinations := (deckSize - b - 7).choose 2
  let higherTeamCombinations := (b - 1).choose 2
  (lowerTeamCombinations + higherTeamCombinations : ℚ) / totalCombinations

/-- The smallest value of b for which q(b) ≥ 1/2 -/
def smallestB : ℕ := 18

theorem smallest_b_probability (b : ℕ) :
  b < smallestB → q b < 1/2 ∧
  q smallestB = 318/703 :=
sorry

end smallest_b_probability_l3363_336333


namespace tan_pi_minus_alpha_l3363_336394

theorem tan_pi_minus_alpha (α : Real) 
  (h1 : α > π / 2) 
  (h2 : α < π) 
  (h3 : 3 * Real.cos (2 * α) - Real.sin α = 2) : 
  Real.tan (π - α) = Real.sqrt 2 / 4 := by
sorry

end tan_pi_minus_alpha_l3363_336394


namespace even_number_2018_in_group_27_l3363_336361

/-- The sum of the number of elements in the first n groups --/
def S (n : ℕ) : ℕ := (3 * n^2 - n) / 2

/-- The proposition that 2018 is in the 27th group --/
theorem even_number_2018_in_group_27 :
  S 26 < 1009 ∧ 1009 ≤ S 27 :=
sorry

end even_number_2018_in_group_27_l3363_336361


namespace ellipse_major_axis_length_l3363_336345

/-- An ellipse with foci at (3, 5) and (23, 40) that is tangent to the y-axis has a major axis of length 43.835 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ Y : ℝ × ℝ),
  F₁ = (3, 5) →
  F₂ = (23, 40) →
  (∀ P ∈ E, ∃ k, dist P F₁ + dist P F₂ = k) →
  (∃ t, Y = (0, t) ∧ Y ∈ E) →
  (∀ P : ℝ × ℝ, P.1 = 0 → dist P F₁ + dist P F₂ ≥ dist Y F₁ + dist Y F₂) →
  dist F₁ F₂ = 43.835 := by
sorry

end ellipse_major_axis_length_l3363_336345


namespace highest_water_level_in_narrow_neck_vase_l3363_336375

/-- Represents a vase with a specific shape --/
inductive VaseShape
  | NarrowNeck
  | Symmetrical
  | WideTop

/-- Represents a vase with its properties --/
structure Vase where
  shape : VaseShape
  height : ℝ
  volume : ℝ

/-- Calculates the water level in a vase given the amount of water --/
noncomputable def waterLevel (v : Vase) (waterAmount : ℝ) : ℝ :=
  sorry

theorem highest_water_level_in_narrow_neck_vase 
  (vases : Fin 5 → Vase)
  (h_same_height : ∀ i j, (vases i).height = (vases j).height)
  (h_same_volume : ∀ i, (vases i).volume = 1)
  (h_water_amount : ∀ i, waterLevel (vases i) 0.5 > 0)
  (h_vase_a_narrow : (vases 0).shape = VaseShape.NarrowNeck)
  (h_other_shapes : ∀ i, i ≠ 0 → (vases i).shape ≠ VaseShape.NarrowNeck) :
  ∀ i, i ≠ 0 → waterLevel (vases 0) 0.5 > waterLevel (vases i) 0.5 :=
sorry

end highest_water_level_in_narrow_neck_vase_l3363_336375


namespace distance_walked_l3363_336357

-- Define the walking time in hours
def walking_time : ℝ := 1.25

-- Define the walking rate in miles per hour
def walking_rate : ℝ := 4.8

-- Theorem statement
theorem distance_walked : walking_time * walking_rate = 6 := by
  sorry

end distance_walked_l3363_336357


namespace quadratic_equation_solution_existence_l3363_336331

theorem quadratic_equation_solution_existence 
  (a b c : ℝ) 
  (h_a : a ≠ 0)
  (h_1 : a * (3.24 : ℝ)^2 + b * (3.24 : ℝ) + c = -0.02)
  (h_2 : a * (3.25 : ℝ)^2 + b * (3.25 : ℝ) + c = 0.03) :
  ∃ x : ℝ, x > 3.24 ∧ x < 3.25 ∧ a * x^2 + b * x + c = 0 := by
  sorry

end quadratic_equation_solution_existence_l3363_336331


namespace distance_to_y_axis_l3363_336335

/-- Given a point P with coordinates (x, -4), if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 8. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -4)
  let dist_to_x_axis : ℝ := |P.2|
  let dist_to_y_axis : ℝ := |P.1|
  dist_to_x_axis = (1/2 : ℝ) * dist_to_y_axis →
  dist_to_y_axis = 8 := by
sorry

end distance_to_y_axis_l3363_336335


namespace tan_expression_value_l3363_336367

theorem tan_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) :
  (2 * (Real.cos (x / 2))^2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 := by
  sorry

end tan_expression_value_l3363_336367


namespace expected_other_marbles_l3363_336372

/-- Represents the distribution of marble colors in Percius's collection -/
structure MarbleCollection where
  clear_percent : ℝ
  black_percent : ℝ
  other_percent : ℝ
  sum_to_one : clear_percent + black_percent + other_percent = 1

/-- Percius's marble collection -/
def percius_marbles : MarbleCollection where
  clear_percent := 0.4
  black_percent := 0.2
  other_percent := 0.4
  sum_to_one := by norm_num

/-- The number of marbles selected by the friend -/
def selected_marbles : ℕ := 5

/-- Theorem: The expected number of marbles of other colors when selecting 5 marbles is 2 -/
theorem expected_other_marbles :
  (selected_marbles : ℝ) * percius_marbles.other_percent = 2 := by sorry

end expected_other_marbles_l3363_336372


namespace puppy_feeding_last_two_weeks_l3363_336325

/-- Represents the feeding schedule and amount for a puppy over 4 weeks -/
structure PuppyFeeding where
  total_food : ℚ
  first_day_food : ℚ
  first_two_weeks_daily_feeding : ℚ
  first_two_weeks_feeding_frequency : ℕ
  last_two_weeks_feeding_frequency : ℕ
  days_in_week : ℕ
  total_weeks : ℕ

/-- Calculates the amount of food fed to the puppy twice a day for the last two weeks -/
def calculate_last_two_weeks_feeding (pf : PuppyFeeding) : ℚ :=
  let first_two_weeks_food := pf.first_two_weeks_daily_feeding * pf.first_two_weeks_feeding_frequency * (2 * pf.days_in_week)
  let total_food_minus_first_day := pf.total_food - pf.first_day_food
  let last_two_weeks_food := total_food_minus_first_day - first_two_weeks_food
  let last_two_weeks_feedings := 2 * pf.last_two_weeks_feeding_frequency * pf.days_in_week
  last_two_weeks_food / last_two_weeks_feedings

/-- Theorem stating that the amount of food fed to the puppy twice a day for the last two weeks is 1/2 cup -/
theorem puppy_feeding_last_two_weeks
  (pf : PuppyFeeding)
  (h1 : pf.total_food = 25)
  (h2 : pf.first_day_food = 1/2)
  (h3 : pf.first_two_weeks_daily_feeding = 1/4)
  (h4 : pf.first_two_weeks_feeding_frequency = 3)
  (h5 : pf.last_two_weeks_feeding_frequency = 2)
  (h6 : pf.days_in_week = 7)
  (h7 : pf.total_weeks = 4) :
  calculate_last_two_weeks_feeding pf = 1/2 := by
  sorry

end puppy_feeding_last_two_weeks_l3363_336325


namespace smallest_prime_dividing_sum_l3363_336388

theorem smallest_prime_dividing_sum : ∃ p : Nat, 
  Prime p ∧ p > 7 ∧ p ∣ (2^14 + 7^8) ∧ 
  ∀ q : Nat, Prime q → q ∣ (2^14 + 7^8) → q ≥ p :=
by sorry

end smallest_prime_dividing_sum_l3363_336388


namespace function_shift_l3363_336363

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_shift (x : ℝ) :
  (∀ y : ℝ, f (y + 1) = y^2 + 2*y) →
  f (x - 1) = x^2 - 2*x := by
  sorry

end function_shift_l3363_336363


namespace village_population_proof_l3363_336340

theorem village_population_proof (P : ℕ) : 
  (0.85 : ℝ) * ((0.90 : ℝ) * P) = 6514 → P = 8518 := by sorry

end village_population_proof_l3363_336340


namespace cloth_cost_price_l3363_336381

theorem cloth_cost_price 
  (selling_price : ℕ) 
  (cloth_length : ℕ) 
  (loss_per_meter : ℕ) 
  (h1 : selling_price = 18000) 
  (h2 : cloth_length = 600) 
  (h3 : loss_per_meter = 5) : 
  (selling_price + cloth_length * loss_per_meter) / cloth_length = 35 := by
sorry

end cloth_cost_price_l3363_336381


namespace multiplication_subtraction_equality_l3363_336344

theorem multiplication_subtraction_equality : 210 * 6 - 52 * 5 = 1000 := by
  sorry

end multiplication_subtraction_equality_l3363_336344


namespace annie_extracurricular_hours_l3363_336337

/-- Calculates the total extracurricular hours before midterms -/
def extracurricular_hours_before_midterms (
  chess_hours_per_week : ℕ)
  (drama_hours_per_week : ℕ)
  (glee_hours_per_week : ℕ)
  (weeks_in_semester : ℕ)
  (weeks_off_sick : ℕ) : ℕ :=
  let total_hours_per_week := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week
  let weeks_before_midterms := weeks_in_semester / 2
  let active_weeks := weeks_before_midterms - weeks_off_sick
  total_hours_per_week * active_weeks

theorem annie_extracurricular_hours :
  extracurricular_hours_before_midterms 2 8 3 12 2 = 52 := by
  sorry

end annie_extracurricular_hours_l3363_336337


namespace geometric_sequence_terms_l3363_336348

/-- 
Given a geometric sequence where:
- The first term is 9/8
- The last term is 1/3
- The common ratio is 2/3
This theorem proves that the number of terms in the sequence is 4.
-/
theorem geometric_sequence_terms : 
  ∀ (a : ℚ) (r : ℚ) (last : ℚ) (n : ℕ),
  a = 9/8 → r = 2/3 → last = 1/3 →
  last = a * r^(n-1) →
  n = 4 := by sorry

end geometric_sequence_terms_l3363_336348
