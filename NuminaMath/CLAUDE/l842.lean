import Mathlib

namespace company_y_installation_charge_l842_84274

-- Define the given constants
def company_x_price : ℝ := 575
def company_x_surcharge_rate : ℝ := 0.04
def company_x_installation : ℝ := 82.50
def company_y_price : ℝ := 530
def company_y_surcharge_rate : ℝ := 0.03
def total_charge_difference : ℝ := 41.60

-- Define the function to calculate total cost
def total_cost (price surcharge_rate installation : ℝ) : ℝ :=
  price + price * surcharge_rate + installation

-- State the theorem
theorem company_y_installation_charge :
  ∃ (company_y_installation : ℝ),
    company_y_installation = 93 ∧
    total_cost company_x_price company_x_surcharge_rate company_x_installation -
    total_cost company_y_price company_y_surcharge_rate company_y_installation =
    total_charge_difference :=
by
  sorry

end company_y_installation_charge_l842_84274


namespace fraction_meaningful_l842_84256

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 4 / (m - 1)) ↔ m ≠ 1 :=
by sorry

end fraction_meaningful_l842_84256


namespace invisible_square_exists_l842_84266

theorem invisible_square_exists (n : ℕ) : 
  ∃ (a b : ℤ), ∀ (i j : ℕ), i < n → j < n → Nat.gcd (Int.toNat (a + i)) (Int.toNat (b + j)) > 1 := by
  sorry

end invisible_square_exists_l842_84266


namespace triangle_circles_area_sum_l842_84261

theorem triangle_circles_area_sum : 
  ∀ (u v w : ℝ),
  u > 0 ∧ v > 0 ∧ w > 0 →
  u + v = 6 →
  u + w = 8 →
  v + w = 10 →
  π * (u^2 + v^2 + w^2) = 56 * π :=
by
  sorry

end triangle_circles_area_sum_l842_84261


namespace smallest_divisible_number_proof_l842_84209

/-- The smallest 5-digit number divisible by 15, 32, 45, and a multiple of 9 and 6 -/
def smallest_divisible_number : ℕ := 11520

theorem smallest_divisible_number_proof :
  smallest_divisible_number ≥ 10000 ∧
  smallest_divisible_number < 100000 ∧
  smallest_divisible_number % 15 = 0 ∧
  smallest_divisible_number % 32 = 0 ∧
  smallest_divisible_number % 45 = 0 ∧
  smallest_divisible_number % 9 = 0 ∧
  smallest_divisible_number % 6 = 0 ∧
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧
    n % 15 = 0 ∧ n % 32 = 0 ∧ n % 45 = 0 ∧ n % 9 = 0 ∧ n % 6 = 0 →
    n ≥ smallest_divisible_number :=
by sorry

#eval smallest_divisible_number

end smallest_divisible_number_proof_l842_84209


namespace max_areas_theorem_l842_84252

/-- Represents the number of non-overlapping areas in a circular disk -/
def max_areas (n : ℕ+) : ℕ :=
  3 * n + 3

/-- 
Theorem: The maximum number of non-overlapping areas in a circular disk 
divided by 2n equally spaced radii and two optimally placed secant lines 
is 3n + 3, where n is a positive integer.
-/
theorem max_areas_theorem (n : ℕ+) : 
  max_areas n = 3 * n + 3 := by
  sorry

#check max_areas_theorem

end max_areas_theorem_l842_84252


namespace log_product_equality_l842_84251

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log y^4) * (Real.log y^3 / Real.log x^3) *
  (Real.log x^4 / Real.log y^5) * (Real.log y^4 / Real.log x^2) *
  (Real.log x^3 / Real.log y^3) = (1/5) * (Real.log x / Real.log y) := by
  sorry

end log_product_equality_l842_84251


namespace joy_visits_grandma_l842_84240

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours until Joy sees her grandma -/
def hours_until_visit : ℕ := 48

/-- The number of days until Joy sees her grandma -/
def days_until_visit : ℕ := hours_until_visit / hours_per_day

theorem joy_visits_grandma : days_until_visit = 2 := by
  sorry

end joy_visits_grandma_l842_84240


namespace y_intercept_for_specific_line_l842_84260

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope 3 and x-intercept (7, 0), the y-intercept is (0, -21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := 3, x_intercept := (7, 0) }
  y_intercept l = (0, -21) := by
  sorry

end y_intercept_for_specific_line_l842_84260


namespace volume_of_cube_with_triple_surface_area_l842_84278

/-- The volume of a cube given its side length -/
def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

/-- The surface area of a cube given its side length -/
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length ^ 2

theorem volume_of_cube_with_triple_surface_area :
  ∀ (side_length1 side_length2 : ℝ),
  side_length1 > 0 →
  side_length2 > 0 →
  cube_volume side_length1 = 8 →
  cube_surface_area side_length2 = 3 * cube_surface_area side_length1 →
  cube_volume side_length2 = 24 * Real.sqrt 3 := by
sorry

end volume_of_cube_with_triple_surface_area_l842_84278


namespace repeating_decimal_to_fraction_l842_84221

/-- Expresses the repeating decimal 3.464646... as a rational number -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 46 / 99) ∧ (x = 343 / 99) := by sorry

end repeating_decimal_to_fraction_l842_84221


namespace limit_at_one_equals_five_l842_84254

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem limit_at_one_equals_five :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (1 + Δx) - f 1) / Δx - 5| < ε :=
sorry

end limit_at_one_equals_five_l842_84254


namespace martine_has_sixteen_peaches_l842_84297

/-- Given the number of peaches Gabrielle has -/
def gabrielle_peaches : ℕ := 15

/-- Benjy's peaches in terms of Gabrielle's -/
def benjy_peaches : ℕ := gabrielle_peaches / 3

/-- Martine's peaches in terms of Benjy's -/
def martine_peaches : ℕ := 2 * benjy_peaches + 6

/-- Theorem: Martine has 16 peaches -/
theorem martine_has_sixteen_peaches : martine_peaches = 16 := by
  sorry

end martine_has_sixteen_peaches_l842_84297


namespace total_combinations_eq_40_l842_84211

/-- Represents the number of helper options for each day of the week --/
def helperOptions : Fin 5 → ℕ
  | 0 => 1  -- Monday
  | 1 => 2  -- Tuesday
  | 2 => 4  -- Wednesday
  | 3 => 5  -- Thursday
  | 4 => 1  -- Friday

/-- The total number of different combinations of helpers for the week --/
def totalCombinations : ℕ := (List.range 5).map helperOptions |>.prod

/-- Theorem stating that the total number of combinations is 40 --/
theorem total_combinations_eq_40 : totalCombinations = 40 := by
  sorry

end total_combinations_eq_40_l842_84211


namespace triangle_cosA_value_l842_84249

theorem triangle_cosA_value (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h1 : (a^2 + b^2) * Real.tan C = 8 * S)
  (h2 : Real.sin A * Real.cos B = 2 * Real.cos A * Real.sin B)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π)
  (h8 : S = (1/2) * a * b * Real.sin C) :
  Real.cos A = Real.sqrt 30 / 15 := by
sorry

end triangle_cosA_value_l842_84249


namespace waiter_customers_l842_84284

/-- Calculates the final number of customers for a waiter given the initial number,
    the number who left, and the number of new customers. -/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Proves that for the given scenario, the final number of customers is 41. -/
theorem waiter_customers : final_customers 19 14 36 = 41 := by
  sorry

end waiter_customers_l842_84284


namespace square_side_significant_digits_l842_84299

/-- Given a square with area 0.12321 m², the number of significant digits in its side length is 5 -/
theorem square_side_significant_digits :
  ∀ (s : ℝ), 
  (s^2 ≥ 0.123205 ∧ s^2 < 0.123215) →  -- Area to the nearest ten-thousandth
  (∃ (n : ℕ), n ≥ 10000 ∧ n < 100000 ∧ s = (n : ℝ) / 100000) := by
  sorry

#check square_side_significant_digits

end square_side_significant_digits_l842_84299


namespace line_y_intercept_implies_m_l842_84268

/-- Given a line equation x + my + 3 - 2m = 0 with y-intercept -1, prove that m = 1 -/
theorem line_y_intercept_implies_m (m : ℝ) :
  (∀ x y : ℝ, x + m * y + 3 - 2 * m = 0) →  -- Line equation
  (0 + m * (-1) + 3 - 2 * m = 0) →          -- y-intercept is -1
  m = 1 :=                                  -- Conclusion: m = 1
by
  sorry

end line_y_intercept_implies_m_l842_84268


namespace prob_face_or_ace_two_draws_l842_84294

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (special_cards : ℕ)

/-- Calculates the probability of drawing at least one special card in two draws with replacement -/
def prob_at_least_one_special (d : Deck) : ℚ :=
  1 - (1 - d.special_cards / d.total_cards) ^ 2

/-- Theorem: The probability of drawing at least one face card or ace in two draws with replacement from a standard 52-card deck is 88/169 -/
theorem prob_face_or_ace_two_draws :
  let standard_deck : Deck := ⟨52, 16⟩
  prob_at_least_one_special standard_deck = 88 / 169 := by
  sorry


end prob_face_or_ace_two_draws_l842_84294


namespace hyperbola_properties_l842_84218

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- The focal length and eccentricity of a hyperbola -/
structure HyperbolaProperties where
  focal_length : ℝ
  eccentricity : ℝ

theorem hyperbola_properties (C : Hyperbola) (l : Line) :
  l.m = Real.sqrt 3 ∧ 
  l.c = -4 * Real.sqrt 3 ∧ 
  (∃ (x y : ℝ), x^2 / C.a^2 - y^2 / C.b^2 = 1 ∧ y = l.m * x + l.c) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / C.a^2 - y₁^2 / C.b^2 = 1 ∧ y₁ = l.m * x₁ + l.c ∧ 
    x₂^2 / C.a^2 - y₂^2 / C.b^2 = 1 ∧ y₂ = l.m * x₂ + l.c → 
    x₁ = x₂ ∧ y₁ = y₂) →
  ∃ (props : HyperbolaProperties), 
    props.focal_length = 8 ∧ 
    props.eccentricity = 8/3 := by
  sorry

end hyperbola_properties_l842_84218


namespace partial_fraction_decomposition_l842_84220

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  6 * x / ((x - 4) * (x - 2)^2) = 6 / (x - 4) + (-6) / (x - 2) + (-6) / (x - 2)^2 := by
  sorry

end partial_fraction_decomposition_l842_84220


namespace parabola_intercepts_sum_l842_84216

/-- Parabola equation: x = 3y^2 - 9y + 5 -/
def parabola_eq (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- X-intercept of the parabola -/
def x_intercept (a : ℝ) : Prop := parabola_eq a 0

/-- Y-intercepts of the parabola -/
def y_intercepts (b c : ℝ) : Prop := parabola_eq 0 b ∧ parabola_eq 0 c ∧ b ≠ c

theorem parabola_intercepts_sum :
  ∀ a b c : ℝ, x_intercept a → y_intercepts b c → a + b + c = 8 := by
  sorry

end parabola_intercepts_sum_l842_84216


namespace power_of_three_mod_five_l842_84203

theorem power_of_three_mod_five : 3^2023 % 5 = 2 := by
  sorry

end power_of_three_mod_five_l842_84203


namespace triangle_side_sum_max_l842_84291

theorem triangle_side_sum_max (a b c : ℝ) (A : ℝ) :
  a = 4 → A = π / 3 → b + c ≤ 8 :=
sorry

end triangle_side_sum_max_l842_84291


namespace distance_not_equal_sum_l842_84225

theorem distance_not_equal_sum : ∀ (a b : ℤ), 
  a = -2 ∧ b = 10 → |b - a| ≠ -2 + 10 := by
  sorry

end distance_not_equal_sum_l842_84225


namespace complex_multiplication_l842_84273

theorem complex_multiplication (i : ℂ) : i * i = -1 → 2 * i * (1 - i) = 2 + 2 * i := by
  sorry

end complex_multiplication_l842_84273


namespace dance_studio_dancers_l842_84208

/-- The number of performances -/
def num_performances : ℕ := 40

/-- The number of dancers in each performance -/
def dancers_per_performance : ℕ := 10

/-- The maximum number of times any pair of dancers can perform together -/
def max_pair_performances : ℕ := 1

/-- The minimum number of dancers required -/
def min_dancers : ℕ := 60

theorem dance_studio_dancers :
  ∀ (n : ℕ), n ≥ min_dancers →
  (n.choose 2) ≥ num_performances * (dancers_per_performance.choose 2) :=
by sorry

end dance_studio_dancers_l842_84208


namespace raw_silk_calculation_l842_84215

/-- The amount of raw silk that results in 12 pounds of dried silk -/
def original_raw_silk : ℚ := 96 / 7

/-- The weight loss during drying in pounds -/
def weight_loss : ℚ := 3 + 12 / 16

theorem raw_silk_calculation (initial_raw : ℚ) (dried : ℚ) 
  (h1 : initial_raw = 30)
  (h2 : dried = 12)
  (h3 : initial_raw - weight_loss = dried) :
  original_raw_silk * (initial_raw - weight_loss) = dried * initial_raw :=
sorry

end raw_silk_calculation_l842_84215


namespace methane_moles_needed_l842_84200

/-- Represents the chemical reaction C6H6 + CH4 → C7H8 + H2 -/
structure ChemicalReaction where
  benzene : ℝ
  methane : ℝ
  toluene : ℝ
  hydrogen : ℝ

/-- The molar mass of Benzene in g/mol -/
def benzene_molar_mass : ℝ := 78

/-- The total amount of Benzene required in grams -/
def total_benzene : ℝ := 156

/-- The number of moles of Toluene produced -/
def toluene_moles : ℝ := 2

/-- The number of moles of Hydrogen produced -/
def hydrogen_moles : ℝ := 2

theorem methane_moles_needed (reaction : ChemicalReaction) :
  reaction.benzene = total_benzene / benzene_molar_mass ∧
  reaction.toluene = toluene_moles ∧
  reaction.hydrogen = hydrogen_moles ∧
  reaction.benzene = reaction.methane →
  reaction.methane = 2 := by
  sorry

end methane_moles_needed_l842_84200


namespace third_candidate_votes_l842_84219

theorem third_candidate_votes (total_votes : ℕ) (john_votes : ℕ) (james_percentage : ℚ) : 
  total_votes = 1150 →
  john_votes = 150 →
  james_percentage = 70 / 100 →
  ∃ (third_votes : ℕ), 
    third_votes = total_votes - john_votes - (james_percentage * (total_votes - john_votes)).floor ∧
    third_votes = john_votes + 150 :=
by sorry

end third_candidate_votes_l842_84219


namespace mango_rate_calculation_l842_84210

/-- Given Andrew's purchase of grapes and mangoes, prove the rate per kg for mangoes -/
theorem mango_rate_calculation (grapes_kg : ℕ) (grapes_rate : ℕ) (mangoes_kg : ℕ) (total_paid : ℕ) :
  grapes_kg = 11 →
  grapes_rate = 98 →
  mangoes_kg = 7 →
  total_paid = 1428 →
  (total_paid - grapes_kg * grapes_rate) / mangoes_kg = 50 := by
  sorry

end mango_rate_calculation_l842_84210


namespace lisa_marbles_problem_l842_84285

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (n : ℕ) (initial : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - initial

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marbles_problem :
  min_additional_marbles 12 40 = 38 := by
  sorry

end lisa_marbles_problem_l842_84285


namespace rational_inequality_solution_l842_84243

theorem rational_inequality_solution (x : ℝ) :
  (2 * x - 1) / (x + 1) < 0 ↔ -1 < x ∧ x < 1/2 := by
  sorry

end rational_inequality_solution_l842_84243


namespace remainder_sum_l842_84229

theorem remainder_sum (n : ℤ) (h : n % 18 = 4) : (n % 2 + n % 9 = 4) := by
  sorry

end remainder_sum_l842_84229


namespace bear_laps_in_scenario_l842_84292

/-- Represents the number of laps completed by the bear in one hour -/
def bear_laps (lake_perimeter : ℝ) (salmon1_speed : ℝ) (salmon2_speed : ℝ) (bear_speed : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of laps completed by the bear in the given scenario -/
theorem bear_laps_in_scenario : 
  bear_laps 1000 500 750 200 = 7 := by sorry

end bear_laps_in_scenario_l842_84292


namespace factorial_22_representation_l842_84239

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def base_ten_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem factorial_22_representation (V R C : ℕ) :
  V < 10 ∧ R < 10 ∧ C < 10 →
  base_ten_representation (factorial 22) =
    [1, 1, 2, 4, 0, V, 4, 6, 1, 7, 4, R, C, 8, 8, 0, 0, 0, 0] →
  V + R + C = 8 := by
  sorry

end factorial_22_representation_l842_84239


namespace inequality_equivalence_l842_84222

theorem inequality_equivalence (x : ℝ) : 
  (x / 2 ≤ 5 - x ∧ 5 - x < -3 * (2 + x)) ↔ x < -11 / 2 := by
  sorry

end inequality_equivalence_l842_84222


namespace count_3digit_even_no_repeat_is_360_l842_84288

/-- A function that counts the number of 3-digit even numbers with no repeated digits -/
def count_3digit_even_no_repeat : ℕ :=
  let first_digit_options := 9  -- 1 to 9
  let second_digit_options := 8  -- Any digit except the first
  let last_digit_zero := first_digit_options * second_digit_options
  let last_digit_even_not_zero := first_digit_options * second_digit_options * 4
  last_digit_zero + last_digit_even_not_zero

/-- Theorem stating that the count of 3-digit even numbers with no repeated digits is 360 -/
theorem count_3digit_even_no_repeat_is_360 :
  count_3digit_even_no_repeat = 360 := by
  sorry

end count_3digit_even_no_repeat_is_360_l842_84288


namespace discount_profit_calculation_l842_84279

theorem discount_profit_calculation (discount : ℝ) (no_discount_profit : ℝ) (with_discount_profit : ℝ) :
  discount = 0.04 →
  no_discount_profit = 0.4375 →
  with_discount_profit = (1 + no_discount_profit) * (1 - discount) - 1 →
  with_discount_profit = 0.38 := by
sorry

end discount_profit_calculation_l842_84279


namespace probability_not_snow_l842_84295

theorem probability_not_snow (p : ℚ) (h : p = 2 / 5) : 1 - p = 3 / 5 := by
  sorry

end probability_not_snow_l842_84295


namespace cricket_team_size_l842_84293

/-- A cricket team with the following properties:
  * The team is 25 years old
  * The wicket keeper is 3 years older than the team
  * The average age of the remaining players (excluding team and wicket keeper) is 1 year less than the average age of the whole team
  * The average age of the team is 22 years
-/
structure CricketTeam where
  n : ℕ  -- number of team members
  team_age : ℕ
  wicket_keeper_age : ℕ
  avg_age : ℕ
  team_age_eq : team_age = 25
  wicket_keeper_age_eq : wicket_keeper_age = team_age + 3
  avg_age_eq : avg_age = 22
  remaining_avg_age : (n * avg_age - team_age - wicket_keeper_age) / (n - 2) = avg_age - 1

/-- The number of members in the cricket team is 11 -/
theorem cricket_team_size (team : CricketTeam) : team.n = 11 := by
  sorry

end cricket_team_size_l842_84293


namespace base8_digit_product_l842_84248

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product :
  productOfList (toBase8 8679) = 392 := by
  sorry

end base8_digit_product_l842_84248


namespace min_sum_squares_l842_84236

/-- The polynomial equation we're considering -/
def P (a b x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + a*x + 1

/-- The condition that the polynomial has at least one real root -/
def has_real_root (a b : ℝ) : Prop := ∃ x : ℝ, P a b x = 0

/-- The theorem statement -/
theorem min_sum_squares (a b : ℝ) (h : has_real_root a b) :
  ∃ (a₀ b₀ : ℝ), has_real_root a₀ b₀ ∧ a₀^2 + b₀^2 = 4/5 ∧ 
  ∀ (a' b' : ℝ), has_real_root a' b' → a'^2 + b'^2 ≥ 4/5 :=
sorry

end min_sum_squares_l842_84236


namespace inequality_proof_l842_84265

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end inequality_proof_l842_84265


namespace min_value_a1a3_l842_84202

theorem min_value_a1a3 (a₁ a₂ a₃ : ℝ) (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (h_a₂ : a₂ = 6) 
  (h_arithmetic : ∃ d : ℝ, 1 / (a₃ + 3) - 1 / (a₂ + 2) = 1 / (a₂ + 2) - 1 / (a₁ + 1)) :
  a₁ * a₃ ≥ 16 * Real.sqrt 3 + 3 := by
  sorry

end min_value_a1a3_l842_84202


namespace prob_advance_four_shots_value_l842_84238

/-- The probability of a successful shot -/
def p : ℝ := 0.6

/-- The probability of advancing after exactly four shots in a basketball contest -/
def prob_advance_four_shots : ℝ :=
  (1 : ℝ) * (1 - p) * p * p

/-- Theorem stating the probability of advancing after exactly four shots -/
theorem prob_advance_four_shots_value :
  prob_advance_four_shots = 18 / 125 := by
  sorry

end prob_advance_four_shots_value_l842_84238


namespace gcf_of_180_240_300_l842_84213

theorem gcf_of_180_240_300 : Nat.gcd 180 (Nat.gcd 240 300) = 60 := by
  sorry

end gcf_of_180_240_300_l842_84213


namespace quadratic_inequality_solution_set_l842_84264

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end quadratic_inequality_solution_set_l842_84264


namespace lego_set_cost_lego_set_cost_is_20_l842_84255

/-- The cost of each lego set when Tonya buys Christmas gifts for her sisters -/
theorem lego_set_cost (doll_cost : ℕ) (num_dolls : ℕ) (num_lego_sets : ℕ) : ℕ :=
  let total_doll_cost := doll_cost * num_dolls
  total_doll_cost / num_lego_sets

/-- Proof that each lego set costs $20 -/
theorem lego_set_cost_is_20 :
  lego_set_cost 15 4 3 = 20 := by
  sorry

end lego_set_cost_lego_set_cost_is_20_l842_84255


namespace stadium_entry_count_l842_84206

def basket_capacity : ℕ := 4634
def placards_per_person : ℕ := 2

theorem stadium_entry_count :
  let total_placards : ℕ := basket_capacity
  let people_entered : ℕ := total_placards / placards_per_person
  people_entered = 2317 := by sorry

end stadium_entry_count_l842_84206


namespace geometric_sequence_property_l842_84257

/-- A geometric sequence. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * q ^ (n - 1)

/-- The theorem stating the properties of the specific geometric sequence. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_diff1 : a 5 - a 1 = 15) 
    (h_diff2 : a 4 - a 2 = 6) : 
    a 3 = 4 ∨ a 3 = -4 := by
  sorry

end geometric_sequence_property_l842_84257


namespace unique_integer_solution_l842_84283

theorem unique_integer_solution :
  ∃! (a b c : ℤ), a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c :=
by
  -- The proof goes here
  sorry

end unique_integer_solution_l842_84283


namespace quadratic_minimum_l842_84245

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2*x + 14

theorem quadratic_minimum :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 := by
  sorry

end quadratic_minimum_l842_84245


namespace mold_radius_l842_84247

/-- The radius of a circular mold with diameter 4 inches is 2 inches -/
theorem mold_radius (d : ℝ) (h : d = 4) : d / 2 = 2 := by
  sorry

end mold_radius_l842_84247


namespace jerry_stickers_l842_84224

theorem jerry_stickers (fred_stickers : ℕ) (george_stickers : ℕ) (jerry_stickers : ℕ) : 
  fred_stickers = 18 →
  george_stickers = fred_stickers - 6 →
  jerry_stickers = 3 * george_stickers →
  jerry_stickers = 36 := by
sorry

end jerry_stickers_l842_84224


namespace lee_savings_l842_84231

theorem lee_savings (initial_savings : ℕ) (num_figures : ℕ) (price_per_figure : ℕ) (sneaker_cost : ℕ) : 
  initial_savings = 15 →
  num_figures = 10 →
  price_per_figure = 10 →
  sneaker_cost = 90 →
  initial_savings + num_figures * price_per_figure - sneaker_cost = 25 := by
sorry

end lee_savings_l842_84231


namespace game_correct_answers_l842_84270

theorem game_correct_answers (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3) :
  ∃ (x : ℕ), x * correct_reward = (total_questions - x) * incorrect_penalty ∧ x = 15 := by
sorry

end game_correct_answers_l842_84270


namespace sin_105_degrees_l842_84233

theorem sin_105_degrees : Real.sin (105 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end sin_105_degrees_l842_84233


namespace regression_properties_l842_84235

-- Define the data points
def data_points : List (ℝ × ℝ) := [(5, 17), (6, 20), (8, 25), (9, 28), (12, 35)]

-- Define the empirical regression equation
def regression_equation (x : ℝ) : ℝ := 2.6 * x + 4.2

-- Theorem to prove the three statements
theorem regression_properties :
  -- 1. The point (8, 25) lies on the regression line
  regression_equation 8 = 25 ∧
  -- 2. The y-intercept of the regression line is 4.2
  regression_equation 0 = 4.2 ∧
  -- 3. The residual for x = 5 is -0.2
  17 - regression_equation 5 = -0.2 := by
  sorry

end regression_properties_l842_84235


namespace more_girls_than_boys_l842_84246

theorem more_girls_than_boys (total_students : ℕ) (boys_ratio girls_ratio : ℕ) : 
  total_students = 42 →
  boys_ratio = 3 →
  girls_ratio = 4 →
  (girls_ratio - boys_ratio) * (total_students / (boys_ratio + girls_ratio)) = 6 := by
  sorry

end more_girls_than_boys_l842_84246


namespace complex_modulus_problem_l842_84241

theorem complex_modulus_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := (i / (1 - i))^2
  Complex.abs z = 1/2 := by
  sorry

end complex_modulus_problem_l842_84241


namespace max_x_value_l842_84230

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_sum_eq : x*y + x*z + y*z = 12) :
  x ≤ (13 + Real.sqrt 160) / 6 :=
by sorry

end max_x_value_l842_84230


namespace sum_A_B_equals_twice_cube_l842_84223

/-- The sum of numbers in the nth group of positive integers -/
def A (n : ℕ) : ℕ :=
  (2 * n - 1) * (n^2 - n + 1)

/-- The difference between the latter and former number in the nth group of cubes -/
def B (n : ℕ) : ℕ :=
  3 * n^2 - 3 * n + 1

/-- Theorem stating that A_n + B_n = 2n^3 for all positive integers n -/
theorem sum_A_B_equals_twice_cube (n : ℕ) :
  A n + B n = 2 * n^3 := by
  sorry

end sum_A_B_equals_twice_cube_l842_84223


namespace total_red_pencils_l842_84276

/-- The number of packs of colored pencils Johnny bought -/
def total_packs : ℕ := 35

/-- The number of packs with 3 extra red pencils -/
def packs_with_3_extra_red : ℕ := 7

/-- The number of packs with 2 extra blue pencils and 1 extra red pencil -/
def packs_with_2_extra_blue_1_extra_red : ℕ := 4

/-- The number of packs with 1 extra green pencil and 2 extra red pencils -/
def packs_with_1_extra_green_2_extra_red : ℕ := 10

/-- The number of red pencils in each pack without extra pencils -/
def red_pencils_per_pack : ℕ := 1

/-- Theorem: The total number of red pencils Johnny bought is 59 -/
theorem total_red_pencils : 
  total_packs * red_pencils_per_pack + 
  packs_with_3_extra_red * 3 + 
  packs_with_2_extra_blue_1_extra_red * 1 + 
  packs_with_1_extra_green_2_extra_red * 2 = 59 := by
  sorry

end total_red_pencils_l842_84276


namespace tilly_star_count_l842_84259

theorem tilly_star_count (east_stars : ℕ) : 
  east_stars + 6 * east_stars = 840 → east_stars = 120 := by
  sorry

end tilly_star_count_l842_84259


namespace monotone_increasing_condition_l842_84207

/-- The function f(x) = (ax - 1)e^x is monotonically increasing on [0,1] if and only if a ≥ 1 -/
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, Monotone (fun x => (a * x - 1) * Real.exp x)) ↔ a ≥ 1 := by
  sorry

end monotone_increasing_condition_l842_84207


namespace perimeter_ratio_of_similar_triangles_l842_84277

/-- Two triangles are similar -/
def SimilarTriangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- The similarity ratio between two triangles -/
def SimilarityRatio (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- The perimeter of a triangle -/
def Perimeter (t : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- Theorem: If two triangles are similar with a ratio of 1:2, then their perimeters have the same ratio -/
theorem perimeter_ratio_of_similar_triangles (ABC DEF : Set (Fin 3 → ℝ × ℝ)) :
  SimilarTriangles ABC DEF →
  SimilarityRatio ABC DEF = 1 / 2 →
  Perimeter ABC / Perimeter DEF = 1 / 2 := by
  sorry

end perimeter_ratio_of_similar_triangles_l842_84277


namespace ninth_term_of_arithmetic_sequence_l842_84205

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem ninth_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_third : a 3 = 5/11) 
  (h_fifteenth : a 15 = 7/8) : 
  a 9 = 117/176 := by
sorry

end ninth_term_of_arithmetic_sequence_l842_84205


namespace area_r_is_twelve_point_five_percent_l842_84289

/-- Represents a circular spinner with specific properties -/
structure CircularSpinner where
  /-- Diameter PQ passes through the center -/
  has_diameter_through_center : Bool
  /-- Areas R and S are equal -/
  r_equals_s : Bool
  /-- R and S together form a quadrant -/
  r_plus_s_is_quadrant : Bool

/-- Calculates the percentage of the total area occupied by region R -/
def area_percentage_r (spinner : CircularSpinner) : ℝ :=
  sorry

/-- Theorem stating that the area of region R is 12.5% of the total circle area -/
theorem area_r_is_twelve_point_five_percent (spinner : CircularSpinner) 
  (h1 : spinner.has_diameter_through_center = true)
  (h2 : spinner.r_equals_s = true)
  (h3 : spinner.r_plus_s_is_quadrant = true) : 
  area_percentage_r spinner = 12.5 := by
  sorry

end area_r_is_twelve_point_five_percent_l842_84289


namespace quadratic_rational_roots_l842_84280

theorem quadratic_rational_roots 
  (n p q : ℚ) 
  (h : p = n + q / n) : 
  ∃ (x y : ℚ), x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 := by
  sorry

end quadratic_rational_roots_l842_84280


namespace inequality_proof_l842_84262

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end inequality_proof_l842_84262


namespace polynomial_simplification_l842_84232

theorem polynomial_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^11 + 5*y^9 + y^8) = 
  15*y^13 - y^12 + 6*y^11 + 5*y^10 - 7*y^9 - 2*y^8 := by
  sorry

end polynomial_simplification_l842_84232


namespace ladder_problem_l842_84267

theorem ladder_problem (ladder_length height : Real) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : Real, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end ladder_problem_l842_84267


namespace polynomial_equality_l842_84271

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (3*x + Real.sqrt 7)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 16 := by
  sorry

end polynomial_equality_l842_84271


namespace inequality_proof_l842_84250

theorem inequality_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) : 
  1 / (a + b) < 1 / (a * b) := by
sorry

end inequality_proof_l842_84250


namespace seventeen_factorial_minus_fifteen_factorial_prime_divisors_l842_84204

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of prime divisors of a natural number -/
def numPrimeDivisors (n : ℕ) : ℕ := (Nat.factors n).length

/-- The main theorem -/
theorem seventeen_factorial_minus_fifteen_factorial_prime_divisors :
  numPrimeDivisors (factorial 17 - factorial 15) = 7 := by
  sorry

end seventeen_factorial_minus_fifteen_factorial_prime_divisors_l842_84204


namespace sam_pennies_l842_84281

theorem sam_pennies (initial : ℕ) (spent : ℕ) (remaining : ℕ) : 
  initial = 98 → spent = 93 → remaining = initial - spent → remaining = 5 :=
by sorry

end sam_pennies_l842_84281


namespace quadratic_point_m_value_l842_84201

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end quadratic_point_m_value_l842_84201


namespace remainder_x_plus_2_pow_2022_l842_84214

theorem remainder_x_plus_2_pow_2022 (x : ℤ) :
  (x^3 % (x^2 + x + 1) = 1) →
  ((x + 2)^2022 % (x^2 + x + 1) = 1) :=
by
  sorry

end remainder_x_plus_2_pow_2022_l842_84214


namespace binomial_square_polynomial_l842_84234

theorem binomial_square_polynomial : ∃ (r s : ℝ), (r * X + s) ^ 2 = 4 * X ^ 2 + 12 * X + 9 :=
sorry

end binomial_square_polynomial_l842_84234


namespace simplify_expression_l842_84212

theorem simplify_expression (x y : ℝ) : 3*x + 4*x - 2*x + 5*y - y = 5*x + 4*y := by
  sorry

end simplify_expression_l842_84212


namespace two_numbers_sum_squares_and_product_l842_84269

theorem two_numbers_sum_squares_and_product : ∃ u v : ℝ, 
  u^2 + v^2 = 20 ∧ u * v = 8 ∧ ((u = 2 ∧ v = 4) ∨ (u = 4 ∧ v = 2)) := by
  sorry

end two_numbers_sum_squares_and_product_l842_84269


namespace find_x_l842_84226

def numbers : List ℕ := [201, 202, 204, 205, 206, 209, 209, 210]

theorem find_x (x : ℕ) :
  let all_numbers := numbers ++ [x]
  (all_numbers.sum / all_numbers.length : ℚ) = 207 →
  x = 217 := by sorry

end find_x_l842_84226


namespace range_of_m_solution_sets_l842_84253

-- Define the function y
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1: Range of m
theorem range_of_m :
  {m : ℝ | ∀ x, y m x < 0} = Set.Ioc (-4) 0 :=
sorry

-- Part 2: Solution sets
theorem solution_sets (m : ℝ) :
  {x : ℝ | y m x < (1 - m) * x - 1} =
    if m = 0 then
      {x : ℝ | x > 0}
    else if m > 0 then
      {x : ℝ | 0 < x ∧ x < 1 / m}
    else
      {x : ℝ | x < 1 / m ∨ x > 0} :=
sorry

end range_of_m_solution_sets_l842_84253


namespace determinant_fraction_equality_l842_84282

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem determinant_fraction_equality (θ : ℝ) : 
  det (Real.sin θ) 2 (Real.cos θ) 3 = 0 →
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 :=
by sorry

end determinant_fraction_equality_l842_84282


namespace quadratic_equation_m_value_l842_84275

theorem quadratic_equation_m_value 
  (x₁ x₂ m : ℝ) 
  (h1 : x₁^2 - 5*x₁ + m = 0)
  (h2 : x₂^2 - 5*x₂ + m = 0)
  (h3 : 3*x₁ - 2*x₂ = 5) :
  m = 6 := by
sorry

end quadratic_equation_m_value_l842_84275


namespace simplest_quadratic_radical_l842_84286

/-- A quadratic radical is considered simpler if it cannot be further simplified 
    by extracting perfect square factors or rationalizing the denominator. -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop := sorry

theorem simplest_quadratic_radical : 
  IsSimplestQuadraticRadical (-Real.sqrt 2) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt (3/2)) ∧
  ¬IsSimplestQuadraticRadical (1 / Real.sqrt 5) :=
sorry

end simplest_quadratic_radical_l842_84286


namespace expected_mass_of_disks_l842_84237

/-- The expected mass of 100 metal disks with manufacturing errors -/
theorem expected_mass_of_disks (
  perfect_diameter : ℝ) 
  (perfect_mass : ℝ) 
  (radius_std_dev : ℝ) 
  (num_disks : ℕ) 
  (h1 : perfect_diameter = 1) 
  (h2 : perfect_mass = 100) 
  (h3 : radius_std_dev = 0.01) 
  (h4 : num_disks = 100) : 
  ∃ (expected_mass : ℝ), expected_mass = 10004 := by
  sorry

end expected_mass_of_disks_l842_84237


namespace inequality_solution_l842_84290

theorem inequality_solution (x : ℝ) : 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9 ↔ x > 45 / 26 := by
  sorry

end inequality_solution_l842_84290


namespace coupon1_best_discount_best_prices_l842_84263

/-- Represents the discount offered by Coupon 1 -/
def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

/-- Represents the discount offered by Coupon 2 -/
def coupon2_discount : ℝ := 50

/-- Represents the discount offered by Coupon 3 -/
def coupon3_discount (x : ℝ) : ℝ := 0.25 * (x - 250)

/-- Theorem stating when Coupon 1 offers the best discount -/
theorem coupon1_best_discount (x : ℝ) :
  (x ≥ 200 ∧ x ≥ 250) →
  (coupon1_discount x > coupon2_discount ∧
   coupon1_discount x > coupon3_discount x) ↔
  (333.33 < x ∧ x < 625) :=
by sorry

/-- Checks if a given price satisfies the condition for Coupon 1 being the best -/
def is_coupon1_best (price : ℝ) : Prop :=
  333.33 < price ∧ price < 625

/-- Theorem stating which of the given prices satisfy the condition -/
theorem best_prices :
  is_coupon1_best 349.95 ∧
  is_coupon1_best 399.95 ∧
  is_coupon1_best 449.95 ∧
  is_coupon1_best 499.95 ∧
  ¬is_coupon1_best 299.95 :=
by sorry

end coupon1_best_discount_best_prices_l842_84263


namespace mixing_solutions_theorem_l842_84287

/-- Proves that mixing 300 mL of 10% alcohol solution with 900 mL of 30% alcohol solution 
    results in a 25% alcohol solution -/
theorem mixing_solutions_theorem (x_volume y_volume : ℝ) 
  (x_concentration y_concentration final_concentration : ℝ) :
  x_volume = 300 →
  y_volume = 900 →
  x_concentration = 0.1 →
  y_concentration = 0.3 →
  final_concentration = 0.25 →
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = final_concentration :=
by
  sorry

#check mixing_solutions_theorem

end mixing_solutions_theorem_l842_84287


namespace triangle_side_range_l842_84228

theorem triangle_side_range (x : ℝ) : 
  x > 0 → 
  (4 + 5 > x ∧ 4 + x > 5 ∧ 5 + x > 4) → 
  1 < x ∧ x < 9 :=
by sorry

end triangle_side_range_l842_84228


namespace algebraic_expression_value_l842_84242

theorem algebraic_expression_value : ∀ a b : ℝ,
  (a * 1^3 + b * 1 + 2022 = 2020) →
  (a * (-1)^3 + b * (-1) + 2023 = 2025) :=
by
  sorry

end algebraic_expression_value_l842_84242


namespace complex_square_l842_84244

theorem complex_square (a b : ℝ) (i : ℂ) (h : i * i = -1) (eq : a + i = 2 - b * i) :
  (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end complex_square_l842_84244


namespace modulus_z_l842_84217

theorem modulus_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : Complex.abs w = Real.sqrt 13) :
  Complex.abs z = (25 * Real.sqrt 13) / 13 := by
  sorry

end modulus_z_l842_84217


namespace modulus_of_complex_l842_84258

theorem modulus_of_complex (i : ℂ) (a : ℝ) : 
  i * i = -1 →
  (1 - i) * (1 - a * i) ∈ {z : ℂ | z.re = 0} →
  Complex.abs (1 - a * i) = Real.sqrt 2 := by
  sorry

end modulus_of_complex_l842_84258


namespace power_sum_reciprocal_integer_l842_84272

theorem power_sum_reciprocal_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
sorry

end power_sum_reciprocal_integer_l842_84272


namespace quadratic_intersection_and_root_distance_l842_84227

theorem quadratic_intersection_and_root_distance 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b * x₁ + c = 0 ∧ a * x₂^2 + 2*b * x₂ + c = 0) ∧
  (∀ x₁ x₂ : ℝ, a * x₁^2 + 2*b * x₁ + c = 0 → a * x₂^2 + 2*b * x₂ + c = 0 → 
    Real.sqrt 3 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3) :=
by
  sorry

end quadratic_intersection_and_root_distance_l842_84227


namespace quadratic_inequality_range_l842_84296

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ -16 ≤ a ∧ a ≤ 0 := by sorry

end quadratic_inequality_range_l842_84296


namespace fraction_equality_l842_84298

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) : 
  18 / 7 + ((2 * q - p) / (2 * q + p)) = 3 := by sorry

end fraction_equality_l842_84298
