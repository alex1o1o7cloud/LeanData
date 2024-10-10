import Mathlib

namespace square_difference_identity_l972_97205

theorem square_difference_identity : (25 + 15)^2 - (25^2 + 15^2) = 750 := by
  sorry

end square_difference_identity_l972_97205


namespace smallest_sum_divisible_by_2016_l972_97235

theorem smallest_sum_divisible_by_2016 :
  ∃ (n₁ n₂ n₃ n₄ n₅ n₆ n₇ : ℕ),
    0 < n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ < n₄ ∧ n₄ < n₅ ∧ n₅ < n₆ ∧ n₆ < n₇ ∧
    (n₁ * n₂ * n₃ * n₄ * n₅ * n₆ * n₇) % 2016 = 0 ∧
    n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ = 31 ∧
    ∀ (m₁ m₂ m₃ m₄ m₅ m₆ m₇ : ℕ),
      0 < m₁ ∧ m₁ < m₂ ∧ m₂ < m₃ ∧ m₃ < m₄ ∧ m₄ < m₅ ∧ m₅ < m₆ ∧ m₆ < m₇ →
      (m₁ * m₂ * m₃ * m₄ * m₅ * m₆ * m₇) % 2016 = 0 →
      m₁ + m₂ + m₃ + m₄ + m₅ + m₆ + m₇ ≥ 31 :=
by sorry

end smallest_sum_divisible_by_2016_l972_97235


namespace team_a_win_probability_l972_97265

/-- Probability of Team A winning a non-fifth set -/
def p : ℚ := 2/3

/-- Probability of Team A winning the fifth set -/
def p_fifth : ℚ := 1/2

/-- The probability of Team A winning the volleyball match -/
theorem team_a_win_probability : 
  (p^3) + (3 * p^2 * (1-p) * p) + (6 * p^2 * (1-p)^2 * p_fifth) = 20/27 := by
  sorry

end team_a_win_probability_l972_97265


namespace probability_of_green_ball_l972_97285

theorem probability_of_green_ball (total_balls : ℕ) (green_balls : ℕ) (red_balls : ℕ)
  (h1 : total_balls = 10)
  (h2 : green_balls = 7)
  (h3 : red_balls = 3)
  (h4 : total_balls = green_balls + red_balls) :
  (green_balls : ℚ) / total_balls = 7 / 10 := by
  sorry

end probability_of_green_ball_l972_97285


namespace viewing_time_theorem_l972_97234

/-- Represents the duration of the show in minutes -/
def show_duration : ℕ := 30

/-- Represents the number of days Max watches the show -/
def days_watched : ℕ := 4

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℚ :=
  (minutes : ℚ) / 60

/-- Theorem stating that watching a 30-minute show for 4 days results in 2 hours of viewing time -/
theorem viewing_time_theorem :
  minutes_to_hours (show_duration * days_watched) = 2 := by
  sorry

end viewing_time_theorem_l972_97234


namespace quadratic_real_root_l972_97236

/-- 
For a quadratic equation x^2 + bx + 9, the equation has at least one real root 
if and only if b belongs to the set (-∞, -6] ∪ [6, ∞)
-/
theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := by sorry

end quadratic_real_root_l972_97236


namespace weight_of_single_pencil_l972_97207

/-- The weight of a single pencil given the weight of a dozen pencils -/
theorem weight_of_single_pencil (dozen_weight : ℝ) (h : dozen_weight = 182.88) :
  dozen_weight / 12 = 15.24 := by
  sorry

end weight_of_single_pencil_l972_97207


namespace problem_1_l972_97231

theorem problem_1 : (-1/12) / (-1/2 + 2/3 + 3/4) = -1/11 := by
  sorry

end problem_1_l972_97231


namespace p_or_q_necessary_not_sufficient_l972_97249

theorem p_or_q_necessary_not_sufficient (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end p_or_q_necessary_not_sufficient_l972_97249


namespace point_A_in_fourth_quadrant_l972_97237

def point_A : ℝ × ℝ := (2, -3)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_A_in_fourth_quadrant : in_fourth_quadrant point_A := by
  sorry

end point_A_in_fourth_quadrant_l972_97237


namespace tagged_ratio_is_one_thirtieth_l972_97214

/-- Represents the fish population in a pond -/
structure FishPopulation where
  initialTagged : ℕ
  secondCatchTotal : ℕ
  secondCatchTagged : ℕ
  estimatedTotal : ℕ

/-- Calculates the ratio of tagged fish to total fish in the second catch -/
def taggedRatio (fp : FishPopulation) : ℚ :=
  fp.secondCatchTagged / fp.secondCatchTotal

/-- The specific fish population described in the problem -/
def pondPopulation : FishPopulation :=
  { initialTagged := 60
  , secondCatchTotal := 60
  , secondCatchTagged := 2
  , estimatedTotal := 1800 }

/-- Theorem stating that the ratio of tagged fish to total fish in the second catch is 1/30 -/
theorem tagged_ratio_is_one_thirtieth :
  taggedRatio pondPopulation = 1 / 30 := by
  sorry

end tagged_ratio_is_one_thirtieth_l972_97214


namespace church_cookie_baking_l972_97276

theorem church_cookie_baking (members : ℕ) (cookies_per_sheet : ℕ) (total_cookies : ℕ) 
  (h1 : members = 100)
  (h2 : cookies_per_sheet = 16)
  (h3 : total_cookies = 16000) :
  total_cookies / (members * cookies_per_sheet) = 10 :=
by
  sorry

end church_cookie_baking_l972_97276


namespace sum_of_reciprocals_l972_97289

theorem sum_of_reciprocals (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 3) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) = 3 := by
sorry

end sum_of_reciprocals_l972_97289


namespace molecular_weight_CaCO3_is_100_09_l972_97284

/-- The molecular weight of calcium carbonate (CaCO3) -/
def molecular_weight_CaCO3 : ℝ :=
  let calcium_weight : ℝ := 40.08
  let carbon_weight : ℝ := 12.01
  let oxygen_weight : ℝ := 16.00
  calcium_weight + carbon_weight + 3 * oxygen_weight

/-- Theorem stating that the molecular weight of CaCO3 is approximately 100.09 -/
theorem molecular_weight_CaCO3_is_100_09 :
  ∃ ε > 0, |molecular_weight_CaCO3 - 100.09| < ε :=
sorry

end molecular_weight_CaCO3_is_100_09_l972_97284


namespace smaugs_hoard_l972_97241

theorem smaugs_hoard (gold_coins : ℕ) (silver_coins : ℕ) (copper_coins : ℕ) 
  (silver_to_copper : ℕ) (total_value : ℕ) :
  gold_coins = 100 →
  silver_coins = 60 →
  copper_coins = 33 →
  silver_to_copper = 8 →
  total_value = 2913 →
  total_value = gold_coins * silver_to_copper * (silver_coins / gold_coins) + 
                silver_coins * silver_to_copper + 
                copper_coins →
  silver_coins / gold_coins = 3 := by
  sorry

end smaugs_hoard_l972_97241


namespace min_value_expression_l972_97248

theorem min_value_expression (a b : ℝ) (h : a - b^2 = 4) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (x y : ℝ), x - y^2 = 4 → x^2 - 3*y^2 + x - 14 ≥ m :=
by sorry

end min_value_expression_l972_97248


namespace geometric_sequence_product_l972_97261

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 := by
  sorry


end geometric_sequence_product_l972_97261


namespace apple_pricing_l972_97228

/-- The price function for apples -/
noncomputable def price (l q x : ℝ) (k : ℝ) : ℝ :=
  if k ≤ x then l * k else l * x + q * (k - x)

theorem apple_pricing (l q x : ℝ) : 
  (price l q x 33 = 11.67) →
  (price l q x 36 = 12.48) →
  (price l q x 10 = 3.62) →
  (x = 30) := by
sorry

end apple_pricing_l972_97228


namespace solution_set_part1_range_of_a_part2_l972_97270

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_part2_l972_97270


namespace min_value_reciprocal_sum_l972_97279

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (1/x + 1/y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_reciprocal_sum_l972_97279


namespace ab_plus_cd_value_l972_97297

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -1)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 0) :
  a * b + c * d = -127 / 9 := by
sorry

end ab_plus_cd_value_l972_97297


namespace remainder_mod_12_l972_97222

theorem remainder_mod_12 : (1234^567 + 89^1011) % 12 = 9 := by
  sorry

end remainder_mod_12_l972_97222


namespace buttons_pattern_l972_97258

/-- Represents the number of buttons in the nth box -/
def buttonsInBox (n : ℕ) : ℕ := 3^(n - 1)

/-- Represents the total number of buttons up to the nth box -/
def totalButtons (n : ℕ) : ℕ := (3^n - 1) / 2

theorem buttons_pattern (n : ℕ) (h : n > 0) :
  (buttonsInBox 1 = 1) ∧
  (buttonsInBox 2 = 3) ∧
  (buttonsInBox 3 = 9) ∧
  (buttonsInBox 4 = 27) ∧
  (buttonsInBox 5 = 81) →
  (∀ k : ℕ, k > 0 → buttonsInBox k = 3^(k - 1)) ∧
  (totalButtons n = (3^n - 1) / 2) :=
sorry

end buttons_pattern_l972_97258


namespace cube_side_ratio_l972_97275

theorem cube_side_ratio (w1 w2 s1 s2 : ℝ) (hw1 : w1 = 6) (hw2 : w2 = 48) 
  (hv : w2 / w1 = (s2 / s1)^3) : s2 / s1 = 2 := by
  sorry

end cube_side_ratio_l972_97275


namespace division_by_self_l972_97245

theorem division_by_self (a : ℝ) (h : a ≠ 0) : 3 * a / a = 3 := by
  sorry

end division_by_self_l972_97245


namespace dennis_lives_on_sixth_floor_l972_97252

def frank_floor : ℕ := 16

def charlie_floor (frank : ℕ) : ℕ := frank / 4

def dennis_floor (charlie : ℕ) : ℕ := charlie + 2

theorem dennis_lives_on_sixth_floor :
  dennis_floor (charlie_floor frank_floor) = 6 := by
  sorry

end dennis_lives_on_sixth_floor_l972_97252


namespace rogers_age_multiple_rogers_age_multiple_is_two_l972_97253

/-- Proves that the multiple of Jill's age that relates to Roger's age is 2 -/
theorem rogers_age_multiple : ℕ → Prop := fun m =>
  let jill_age : ℕ := 20
  let finley_age : ℕ := 40
  let years_passed : ℕ := 15
  let roger_age : ℕ := m * jill_age + 5
  let jill_future_age : ℕ := jill_age + years_passed
  let roger_future_age : ℕ := roger_age + years_passed
  let finley_future_age : ℕ := finley_age + years_passed
  let future_age_difference : ℕ := roger_future_age - jill_future_age
  (future_age_difference = finley_future_age - 30) → (m = 2)

/-- The theorem holds for m = 2 -/
theorem rogers_age_multiple_is_two : rogers_age_multiple 2 := by
  sorry

end rogers_age_multiple_rogers_age_multiple_is_two_l972_97253


namespace square_root_of_16_l972_97200

theorem square_root_of_16 : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_16_l972_97200


namespace floor_sum_possible_values_l972_97295

theorem floor_sum_possible_values (x y z : ℝ) 
  (hx : ⌊x⌋ = 5) (hy : ⌊y⌋ = -3) (hz : ⌊z⌋ = -2) : 
  ⌊x - y + z⌋ ∈ ({5, 6, 7} : Set ℤ) := by
  sorry

end floor_sum_possible_values_l972_97295


namespace theater_revenue_l972_97232

theorem theater_revenue (n : ℕ) (C : ℝ) :
  (∃ R : ℝ, R = 1.20 * C) →
  (∃ R_95 : ℝ, R_95 = 0.95 * 1.20 * C ∧ R_95 = 1.14 * C) :=
by sorry

end theater_revenue_l972_97232


namespace hierarchy_combinations_l972_97260

def society_size : ℕ := 12
def num_dukes : ℕ := 3
def knights_per_duke : ℕ := 2

def choose_hierarchy : ℕ := 
  society_size * 
  (society_size - 1) * 
  (society_size - 2) * 
  (society_size - 3) * 
  (Nat.choose (society_size - 4) knights_per_duke) * 
  (Nat.choose (society_size - 4 - knights_per_duke) knights_per_duke) * 
  (Nat.choose (society_size - 4 - 2 * knights_per_duke) knights_per_duke)

theorem hierarchy_combinations : 
  choose_hierarchy = 907200 :=
by sorry

end hierarchy_combinations_l972_97260


namespace ralph_tv_time_l972_97216

/-- Represents Ralph's TV watching schedule for a week -/
structure TVSchedule where
  weekdayHours : ℝ
  weekdayShows : ℕ × ℕ  -- (number of 1-hour shows, number of 30-minute shows)
  videoGameDays : ℕ
  weekendHours : ℝ
  weekendShows : ℕ × ℕ  -- (number of 1-hour shows, number of 45-minute shows)
  weekendBreak : ℝ

/-- Calculates the total TV watching time for a week given a TV schedule -/
def totalTVTime (schedule : TVSchedule) : ℝ :=
  let weekdayTotal := schedule.weekdayHours * 5
  let weekendTotal := (schedule.weekendHours - schedule.weekendBreak) * 2
  weekdayTotal + weekendTotal

/-- Ralph's actual TV schedule -/
def ralphSchedule : TVSchedule :=
  { weekdayHours := 3
  , weekdayShows := (1, 4)
  , videoGameDays := 3
  , weekendHours := 6
  , weekendShows := (3, 4)
  , weekendBreak := 0.5 }

/-- Theorem stating that Ralph's total TV watching time in one week is 26 hours -/
theorem ralph_tv_time : totalTVTime ralphSchedule = 26 := by
  sorry


end ralph_tv_time_l972_97216


namespace range_of_m_l972_97263

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), x^2 + 2*x - m > 0) ↔ 
  (1^2 + 2*1 - m ≤ 0 ∧ 2^2 + 2*2 - m > 0) :=
by sorry

end range_of_m_l972_97263


namespace triangle_perimeter_l972_97220

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 6) (hc : c = 7) :
  a + b + c = 23 := by
  sorry

end triangle_perimeter_l972_97220


namespace a_value_l972_97298

theorem a_value (a : ℚ) (h : a + a/4 = 10/4) : a = 2 := by
  sorry

end a_value_l972_97298


namespace area_remaining_after_iterations_l972_97257

/-- The fraction of area that remains after each iteration -/
def remaining_fraction : ℚ := 3 / 4

/-- The number of iterations -/
def num_iterations : ℕ := 5

/-- The final fraction of the original area remaining -/
def final_fraction : ℚ := 243 / 1024

theorem area_remaining_after_iterations :
  remaining_fraction ^ num_iterations = final_fraction := by
  sorry

end area_remaining_after_iterations_l972_97257


namespace complex_equality_implies_power_l972_97225

/-- Given complex numbers z₁ and z₂, where z₁ = -1 + 3i and z₂ = a + bi³,
    if z₁ = z₂, then b^a = -1/3 -/
theorem complex_equality_implies_power (a b : ℝ) :
  let z₁ : ℂ := -1 + 3 * Complex.I
  let z₂ : ℂ := a + b * Complex.I^3
  z₁ = z₂ → b^a = -1/3 := by sorry

end complex_equality_implies_power_l972_97225


namespace factor_implies_a_value_l972_97280

theorem factor_implies_a_value (a b : ℝ) :
  (∀ x : ℝ, (x^2 + x - 6) ∣ (2*x^4 + x^3 - a*x^2 + b*x + a + b - 1)) →
  a = 16 := by
sorry

end factor_implies_a_value_l972_97280


namespace quadrilateral_complex_point_l972_97208

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  z : ℂ

/-- Represents a quadrilateral with vertices A, B, C, D -/
structure Quadrilateral where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint

/-- Theorem: In quadrilateral ABCD, if A, B, and C correspond to given complex numbers,
    then D corresponds to 1+3i -/
theorem quadrilateral_complex_point (q : Quadrilateral)
    (hA : q.A.z = 2 + I)
    (hB : q.B.z = 4 + 3*I)
    (hC : q.C.z = 3 + 5*I) :
    q.D.z = 1 + 3*I := by
  sorry

end quadrilateral_complex_point_l972_97208


namespace cone_base_diameter_l972_97296

/-- For a cone with surface area 3π and lateral surface that unfolds into a semicircle, 
    the diameter of its base is 2. -/
theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * Real.pi * l^2 + Real.pi * r^2 = 3 * Real.pi) 
  (h2 : Real.pi * l = 2 * Real.pi * r) : 
  2 * r = 2 := by sorry

end cone_base_diameter_l972_97296


namespace quadratic_linear_intersection_l972_97269

theorem quadratic_linear_intersection (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 2 = -3 * x - 2) ↔ a = 25 / 16 := by
  sorry

end quadratic_linear_intersection_l972_97269


namespace inverse_as_linear_combination_l972_97282

def M : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (a b : ℚ), M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ 
  a = -1/12 ∧ b = 7/12 := by
sorry

end inverse_as_linear_combination_l972_97282


namespace subset_with_unique_sum_representation_l972_97281

theorem subset_with_unique_sum_representation :
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n :=
sorry

end subset_with_unique_sum_representation_l972_97281


namespace sufficient_not_necessary_condition_l972_97226

def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x|

theorem sufficient_not_necessary_condition :
  (∀ a < 0, ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) ∧
  (∃ a ≥ 0, ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) :=
sorry

end sufficient_not_necessary_condition_l972_97226


namespace one_ounce_in_gallons_l972_97293

/-- The number of ounces in one gallon of water -/
def ounces_per_gallon : ℚ := 128

/-- The number of ounces Jimmy drinks each time -/
def ounces_per_serving : ℚ := 8

/-- The number of times Jimmy drinks water per day -/
def servings_per_day : ℚ := 8

/-- The number of gallons Jimmy prepares for 5 days -/
def gallons_for_five_days : ℚ := 5/2

/-- The number of days Jimmy prepares water for -/
def days_prepared : ℚ := 5

/-- Theorem stating that 1 ounce of water is equal to 1/128 gallons -/
theorem one_ounce_in_gallons :
  1 / ounces_per_gallon = 
    gallons_for_five_days / (ounces_per_serving * servings_per_day * days_prepared) :=
by sorry

end one_ounce_in_gallons_l972_97293


namespace domino_game_strategy_l972_97211

/-- Represents the players in the game -/
inductive Player
| Alice
| Bob

/-- Represents the outcome of the game -/
inductive Outcome
| Win
| Lose

/-- Represents a grid in the domino game -/
structure Grid :=
  (n : ℕ)
  (m : ℕ)

/-- Determines if a player has a winning strategy on a given grid -/
def has_winning_strategy (player : Player) (grid : Grid) : Prop :=
  match player with
  | Player.Alice => 
      (grid.n % 2 = 0 ∧ grid.m % 2 = 1) ∨
      (grid.n % 2 = 1 ∧ grid.m % 2 = 0)
  | Player.Bob => 
      (grid.n % 2 = 0 ∧ grid.m % 2 = 0)

/-- Theorem stating the winning strategies for the domino game -/
theorem domino_game_strategy (grid : Grid) :
  (grid.n % 2 = 0 ∧ grid.m % 2 = 0 → has_winning_strategy Player.Bob grid) ∧
  (grid.n % 2 = 0 ∧ grid.m % 2 = 1 → has_winning_strategy Player.Alice grid) :=
sorry

end domino_game_strategy_l972_97211


namespace quadratic_form_sum_l972_97221

/-- Given a quadratic function f(x) = x^2 - 26x + 129, 
    prove that when written in the form (x+d)^2 + e, d + e = -53 -/
theorem quadratic_form_sum (x : ℝ) : 
  ∃ (d e : ℝ), (∀ x, x^2 - 26*x + 129 = (x+d)^2 + e) ∧ (d + e = -53) := by
  sorry

end quadratic_form_sum_l972_97221


namespace ellipse_eccentricity_range_l972_97230

/-- The eccentricity of an ellipse with given conditions is between 0 and 2√5/5 -/
theorem ellipse_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt (1 - b^2 / a^2)
  let l := {p : ℝ × ℝ | p.2 = 1/2 * (p.1 + a)}
  let C₁ := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  let C₂ := {p : ℝ × ℝ | p.1^2 + p.2^2 = b^2}
  (∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ l ∩ C₂ ∧ q ∈ l ∩ C₂) →
  0 < e ∧ e < 2 * Real.sqrt 5 / 5 := by
sorry

end ellipse_eccentricity_range_l972_97230


namespace solve_airport_distance_l972_97217

/-- Represents the problem of calculating the distance to the airport --/
def airport_distance_problem (initial_speed : ℝ) (speed_increase : ℝ) (initial_time : ℝ) 
  (late_time : ℝ) (early_time : ℝ) : Prop :=
  ∃ (distance : ℝ) (total_time : ℝ),
    -- Initial part of the journey
    initial_speed * initial_time = initial_speed
    -- Total distance equation
    ∧ distance = initial_speed * (total_time + late_time)
    -- Remaining distance equation with increased speed
    ∧ distance - initial_speed * initial_time = (initial_speed + speed_increase) * (total_time - initial_time - early_time)
    -- The solution
    ∧ distance = 264

/-- The theorem stating the solution to the airport distance problem --/
theorem solve_airport_distance : 
  airport_distance_problem 45 20 1 0.75 0.75 := by
  sorry

end solve_airport_distance_l972_97217


namespace b_completes_in_20_days_l972_97271

/-- The number of days it takes for person A to complete the work alone -/
def days_A : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.3

/-- The number of days it takes for person B to complete the work alone -/
def days_B : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work alone in 20 days -/
theorem b_completes_in_20_days :
  days_together * (1 / days_A + 1 / days_B) = 1 - work_left :=
sorry

end b_completes_in_20_days_l972_97271


namespace number_not_perfect_square_l972_97287

theorem number_not_perfect_square (n : ℕ) 
  (h : ∃ k, n = 6 * (10^600 - 1) / 9 + k * 10^600 ∧ k ≥ 0) : 
  ¬∃ m : ℕ, n = m^2 := by
sorry

end number_not_perfect_square_l972_97287


namespace tournament_512_players_games_l972_97250

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  initial_players : ℕ
  games_played : ℕ

/-- Calculates the number of games required to determine a champion. -/
def games_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

/-- Theorem stating that a tournament with 512 initial players requires 511 games. -/
theorem tournament_512_players_games (tournament : SingleEliminationTournament) 
    (h : tournament.initial_players = 512) : 
    games_required tournament = 511 := by
  sorry

#eval games_required ⟨512, 0⟩

end tournament_512_players_games_l972_97250


namespace circle_equation_l972_97259

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the properties of the circle
def is_tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

def center_on_line (c : Circle) : Prop :=
  c.center.1 = 3 * c.center.2

def cuts_chord_on_line (c : Circle) (chord_length : ℝ) : Prop :=
  ∃ (p q : ℝ × ℝ),
    p.1 - p.2 = 0 ∧ q.1 - q.2 = 0 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2 ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation (c : Circle) :
  is_tangent_to_y_axis c →
  center_on_line c →
  cuts_chord_on_line c (2 * Real.sqrt 7) →
  (∀ x y : ℝ, (x - 3)^2 + (y - 1)^2 = 9 ∨ (x + 3)^2 + (y + 1)^2 = 9 ↔
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
by sorry

end circle_equation_l972_97259


namespace sum_of_two_numbers_l972_97286

theorem sum_of_two_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 := by
  sorry

end sum_of_two_numbers_l972_97286


namespace square_difference_equality_l972_97203

theorem square_difference_equality : (36 + 9)^2 - (9^2 + 36^2) = 648 := by
  sorry

end square_difference_equality_l972_97203


namespace constant_term_binomial_expansion_l972_97247

/-- The constant term in the binomial expansion of (3x^2 - 2/x^3)^5 is 1080 -/
theorem constant_term_binomial_expansion :
  let f := fun x : ℝ => (3 * x^2 - 2 / x^3)^5
  ∃ c : ℝ, c = 1080 ∧ (∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x) :=
by sorry

end constant_term_binomial_expansion_l972_97247


namespace west_movement_representation_l972_97239

/-- Represents direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Converts a movement to its coordinate representation --/
def toCoordinate (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

theorem west_movement_representation :
  let westMovement : Movement := ⟨80, Direction.West⟩
  toCoordinate westMovement = -80 := by sorry

end west_movement_representation_l972_97239


namespace intersection_A_complement_B_l972_97246

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 3, 4}

-- Define set B
def B : Set Nat := {4, 5, 6}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by sorry

end intersection_A_complement_B_l972_97246


namespace circle_tangency_l972_97255

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 - r2)^2

theorem circle_tangency (m : ℝ) :
  let c1 : ℝ × ℝ := (0, 0)
  let r1 : ℝ := Real.sqrt m
  let c2 : ℝ × ℝ := (-3, 4)
  let r2 : ℝ := 6
  internally_tangent c1 c2 r1 r2 → m = 1 ∨ m = 121 := by
sorry

end circle_tangency_l972_97255


namespace one_fourth_of_eight_point_eight_l972_97277

theorem one_fourth_of_eight_point_eight (x : ℚ) : x = 8.8 → (1 / 4 : ℚ) * x = 11 / 5 := by
  sorry

end one_fourth_of_eight_point_eight_l972_97277


namespace no_solutions_inequality_l972_97256

theorem no_solutions_inequality : ¬∃ (n k : ℕ), n ≤ n! - k^n ∧ n! - k^n ≤ k * n := by
  sorry

end no_solutions_inequality_l972_97256


namespace function_domain_implies_k_range_l972_97218

theorem function_domain_implies_k_range
  (a : ℝ) (k : ℝ)
  (h_a_pos : a > 0)
  (h_a_neq_one : a ≠ 1)
  (h_defined : ∀ x : ℝ, x^2 - 2*k*x + 2*k + 3 > 0) :
  -1 < k ∧ k < 3 :=
sorry

end function_domain_implies_k_range_l972_97218


namespace sufficient_but_not_necessary_l972_97244

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem sufficient_but_not_necessary
  (a₁ q : ℝ) :
  (∀ n : ℕ, n > 0 → geometric_sequence a₁ q (n + 1) > geometric_sequence a₁ q n) ↔
  (a₁ > 0 ∧ q > 1 ∨
   ∃ a₁' q', (a₁' ≤ 0 ∨ q' ≤ 1) ∧
   ∀ n : ℕ, n > 0 → geometric_sequence a₁' q' (n + 1) > geometric_sequence a₁' q' n) :=
by sorry

end sufficient_but_not_necessary_l972_97244


namespace pizza_combinations_l972_97213

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end pizza_combinations_l972_97213


namespace shopping_money_l972_97229

/-- Proves that if a person spends $20 and is left with $4 more than half of their original amount, then their original amount was $48. -/
theorem shopping_money (original : ℕ) : 
  (original / 2 + 4 = original - 20) → original = 48 :=
by sorry

end shopping_money_l972_97229


namespace book_purchase_l972_97242

theorem book_purchase (total_volumes : ℕ) (paperback_cost hardcover_cost total_cost : ℕ) 
  (h : total_volumes = 10)
  (h1 : paperback_cost = 18)
  (h2 : hardcover_cost = 28)
  (h3 : total_cost = 240) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_cost + (total_volumes - hardcover_count) * paperback_cost = total_cost ∧ 
    hardcover_count = 6 := by
  sorry

end book_purchase_l972_97242


namespace domain_of_g_l972_97273

-- Define the function f with domain [-3, 1]
def f : Set ℝ → Set ℝ := fun D ↦ {x | x ∈ D ∧ -3 ≤ x ∧ x ≤ 1}

-- Define the function g in terms of f
def g (f : Set ℝ → Set ℝ) : Set ℝ → Set ℝ := fun D ↦ {x | (x + 1) ∈ f D}

-- Theorem statement
theorem domain_of_g (D : Set ℝ) :
  g f D = {x : ℝ | -4 ≤ x ∧ x ≤ 0} :=
sorry

end domain_of_g_l972_97273


namespace movie_deal_savings_l972_97201

theorem movie_deal_savings : 
  let deal_price : ℚ := 20
  let movie_price : ℚ := 8
  let popcorn_price : ℚ := movie_price - 3
  let drink_price : ℚ := popcorn_price + 1
  let candy_price : ℚ := drink_price / 2
  let total_price : ℚ := movie_price + popcorn_price + drink_price + candy_price
  total_price - deal_price = 2 := by sorry

end movie_deal_savings_l972_97201


namespace ellipse_center_correct_l972_97212

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  (3 * y + 6)^2 / 7^2 + (4 * x - 8)^2 / 6^2 = 1

/-- The center of the ellipse -/
def ellipse_center : ℝ × ℝ := (-2, 2)

/-- Theorem stating that ellipse_center is the center of the ellipse defined by ellipse_equation -/
theorem ellipse_center_correct :
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    ((y - ellipse_center.2)^2 / (7/3)^2 + (x - ellipse_center.1)^2 / (3/2)^2 = 1) :=
by sorry

end ellipse_center_correct_l972_97212


namespace gdp_equality_l972_97227

/-- Represents the GDP value in billions of yuan -/
def gdp_billions : ℝ := 4504.5

/-- Represents the same GDP value in scientific notation -/
def gdp_scientific : ℝ := 4.5045 * (10 ^ 12)

/-- Theorem stating that the GDP value in billions is equal to its scientific notation representation -/
theorem gdp_equality : gdp_billions * (10 ^ 9) = gdp_scientific := by sorry

end gdp_equality_l972_97227


namespace minimum_loaves_needed_l972_97202

def slices_per_loaf : ℕ := 20
def regular_sandwich_slices : ℕ := 2
def double_meat_sandwich_slices : ℕ := 3
def triple_decker_sandwich_slices : ℕ := 4
def club_sandwich_slices : ℕ := 5
def regular_sandwiches : ℕ := 25
def double_meat_sandwiches : ℕ := 18
def triple_decker_sandwiches : ℕ := 12
def club_sandwiches : ℕ := 8

theorem minimum_loaves_needed : 
  ∃ (loaves : ℕ), 
    loaves * slices_per_loaf = 
      regular_sandwiches * regular_sandwich_slices +
      double_meat_sandwiches * double_meat_sandwich_slices +
      triple_decker_sandwiches * triple_decker_sandwich_slices +
      club_sandwiches * club_sandwich_slices ∧
    loaves = 10 := by
  sorry

end minimum_loaves_needed_l972_97202


namespace equation_solution_l972_97264

theorem equation_solution (x : ℝ) :
  8.438 * Real.cos (x - π/4) * (1 - 4 * Real.cos (2*x)^2) - 2 * Real.cos (4*x) = 3 →
  ∃ k : ℤ, x = π/4 * (8*k + 1) :=
by sorry

end equation_solution_l972_97264


namespace angle_B_is_60_degrees_l972_97266

-- Define a structure for a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem angle_B_is_60_degrees (t : Triangle) 
  (h1 : t.B = 2 * t.A)
  (h2 : t.C = 3 * t.A)
  (h3 : t.A + t.B + t.C = 180) : 
  t.B = 60 := by
  sorry

end angle_B_is_60_degrees_l972_97266


namespace extended_hexagon_area_l972_97238

structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ
  area : ℝ

def extend_hexagon (h : Hexagon) : Hexagon := sorry

theorem extended_hexagon_area (h : Hexagon) 
  (side_lengths : Fin 6 → ℝ)
  (h_sides : ∀ i, dist (h.vertices i) (h.vertices ((i + 1) % 6)) = side_lengths i)
  (h_area : h.area = 30)
  (h_side_lengths : side_lengths = ![3, 4, 5, 6, 7, 8]) :
  (extend_hexagon h).area = 90 := by
  sorry

end extended_hexagon_area_l972_97238


namespace quadratic_substitution_roots_l972_97283

/-- Given a quadratic equation ax^2 + bx + c = 0, this theorem proves the conditions for equal
    product of roots after substitution and the sum of all roots in those cases. -/
theorem quadratic_substitution_roots (a b c : ℝ) (h : a ≠ 0) :
  ∃ k : ℝ, 
    (k = 0 ∨ k = -b/a) ∧ 
    (∀ y : ℝ, c/a = (a*k^2 + b*k + c)/a) ∧
    ((k = 0 → ((-b/a) + (-b/a) = -2*b/a)) ∧ 
     (k = -b/a → ((-b/a) + (b/a) = 0))) := by
  sorry


end quadratic_substitution_roots_l972_97283


namespace distance_on_line_l972_97240

/-- The distance between two points on a line --/
theorem distance_on_line (n m p q r s : ℝ) :
  q = n * p + m →
  s = n * r + m →
  Real.sqrt ((r - p)^2 + (s - q)^2) = |r - p| * Real.sqrt (1 + n^2) := by
  sorry

end distance_on_line_l972_97240


namespace power_mod_eleven_l972_97219

theorem power_mod_eleven : 3^21 % 11 = 3 := by
  sorry

end power_mod_eleven_l972_97219


namespace problem_statement_l972_97288

theorem problem_statement : (-1)^53 + 2^(4^3 + 5^2 - 7^2) = 1099511627775 := by
  sorry

end problem_statement_l972_97288


namespace ribbon_remaining_length_l972_97262

/-- The length of the original ribbon in meters -/
def original_length : ℝ := 51

/-- The number of pieces cut from the ribbon -/
def num_pieces : ℕ := 100

/-- The length of each piece in centimeters -/
def piece_length_cm : ℝ := 15

/-- Conversion factor from centimeters to meters -/
def cm_to_m : ℝ := 0.01

/-- The remaining length of the ribbon after cutting the pieces -/
def remaining_length : ℝ := original_length - (num_pieces : ℝ) * piece_length_cm * cm_to_m

theorem ribbon_remaining_length :
  remaining_length = 36 := by sorry

end ribbon_remaining_length_l972_97262


namespace ellipse_theorem_l972_97243

-- Define the ellipse E
def ellipse_E (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the vertex condition
def vertex_condition (a b : ℝ) : Prop :=
  ellipse_E 0 1 a b

-- Define the focal length condition
def focal_length_condition (a b : ℝ) : Prop :=
  2 * Real.sqrt 3 = 2 * Real.sqrt (a^2 - b^2)

-- Define the intersection condition
def intersection_condition (k : ℝ) (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_E x₁ y₁ a b ∧
    ellipse_E x₂ y₂ a b ∧
    y₁ - 1 = k * (x₁ + 2) ∧
    y₂ - 1 = k * (x₂ + 2) ∧
    x₁ ≠ x₂

-- Define the x-intercept distance condition
def x_intercept_distance (k : ℝ) (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    intersection_condition k a b ∧
    |x₁ / (1 - y₁) - x₂ / (1 - y₂)| = 2

-- Theorem statement
theorem ellipse_theorem (a b : ℝ) :
  vertex_condition a b ∧ focal_length_condition a b →
  (a = 2 ∧ b = 1) ∧
  (∀ k : ℝ, x_intercept_distance k a b → k = -4) :=
sorry

end ellipse_theorem_l972_97243


namespace albert_joshua_difference_l972_97224

-- Define the number of rocks each person collected
def joshua_rocks : ℕ := 80
def jose_rocks : ℕ := joshua_rocks - 14
def albert_rocks : ℕ := jose_rocks + 20

-- Theorem statement
theorem albert_joshua_difference :
  albert_rocks - joshua_rocks = 6 := by
sorry

end albert_joshua_difference_l972_97224


namespace unique_two_digit_number_exists_l972_97291

/-- A two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Get the tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- Get the units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- Reverse the digits of a two-digit number -/
def reverse_digits (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨10 * (units_digit n) + (tens_digit n), by sorry⟩

theorem unique_two_digit_number_exists :
  ∃! (X : TwoDigitNumber),
    (tens_digit X) * (units_digit X) = 24 ∧
    (reverse_digits X).val = X.val + 18 ∧
    X.val = 46 := by sorry

end unique_two_digit_number_exists_l972_97291


namespace range_of_function_l972_97299

theorem range_of_function :
  ∀ (x : ℝ), -2/3 ≤ (Real.sin x - 1) / (2 - Real.sin x) ∧ 
             (Real.sin x - 1) / (2 - Real.sin x) ≤ 0 ∧
  (∃ (y : ℝ), (Real.sin y - 1) / (2 - Real.sin y) = -2/3) ∧
  (∃ (z : ℝ), (Real.sin z - 1) / (2 - Real.sin z) = 0) :=
by sorry

end range_of_function_l972_97299


namespace woods_length_l972_97210

/-- Given a rectangular area of woods with width 8 miles and total area 24 square miles,
    prove that the length of the woods is 3 miles. -/
theorem woods_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 8 → area = 24 → area = width * length → length = 3 := by sorry

end woods_length_l972_97210


namespace two_players_percentage_of_goals_l972_97267

def total_goals : ℕ := 300
def player_goals : ℕ := 30
def num_players : ℕ := 2

theorem two_players_percentage_of_goals :
  (player_goals * num_players : ℚ) / total_goals * 100 = 20 := by
  sorry

end two_players_percentage_of_goals_l972_97267


namespace add_preserves_inequality_l972_97215

theorem add_preserves_inequality (a b c : ℝ) : a < b → a + c < b + c := by
  sorry

end add_preserves_inequality_l972_97215


namespace perimeter_of_square_C_l972_97294

/-- Given three squares A, B, and C, prove that the perimeter of C is 90 -/
theorem perimeter_of_square_C (a b c : ℝ) : 
  (4 * a = 30) →  -- perimeter of A is 30
  (b = 2 * a) →   -- side of B is twice the side of A
  (c = a + b) →   -- side of C is sum of sides of A and B
  (4 * c = 90) :=  -- perimeter of C is 90
by sorry

end perimeter_of_square_C_l972_97294


namespace quadratic_roots_distinct_real_l972_97272

/-- Given a quadratic equation bx^2 - 3x√5 + d = 0 with real constants b and d,
    and a discriminant of 25, the roots are distinct and real. -/
theorem quadratic_roots_distinct_real (b d : ℝ) : 
  let discriminant := (-3 * Real.sqrt 5) ^ 2 - 4 * b * d
  ∀ x : ℝ, (b * x^2 - 3 * x * Real.sqrt 5 + d = 0 ∧ discriminant = 25) →
    ∃ y : ℝ, x ≠ y ∧ b * y^2 - 3 * y * Real.sqrt 5 + d = 0 := by
  sorry


end quadratic_roots_distinct_real_l972_97272


namespace ratio_sum_problem_l972_97204

theorem ratio_sum_problem (a b : ℕ) : 
  a * 3 = b * 8 →  -- The two numbers are in the ratio 8 to 3
  b = 104 →        -- The bigger number is 104
  a + b = 143      -- The sum of the numbers is 143
  := by sorry

end ratio_sum_problem_l972_97204


namespace parabola_properties_l972_97268

-- Define the parabola and its coefficients
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points that the parabola passes through
def point_A : ℝ × ℝ := (-1, 0)
def point_C : ℝ × ℝ := (0, 3)
def point_B : ℝ × ℝ := (2, -3)

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  ∃ (a b c : ℝ),
    -- The parabola passes through the given points
    (parabola a b c (point_A.1) = point_A.2) ∧
    (parabola a b c (point_C.1) = point_C.2) ∧
    (parabola a b c (point_B.1) = point_B.2) ∧
    -- The parabola equation is y = -2x² + x + 3
    (a = -2 ∧ b = 1 ∧ c = 3) ∧
    -- The axis of symmetry is x = 1/4
    (- b / (2 * a) = 1 / 4) ∧
    -- The vertex coordinates are (1/4, 25/8)
    (parabola a b c (1 / 4) = 25 / 8) := by
  sorry


end parabola_properties_l972_97268


namespace solve_sqrt_equation_l972_97254

theorem solve_sqrt_equation (x : ℝ) :
  Real.sqrt ((3 / x) + 3) = 5/3 → x = -27/2 := by
  sorry

end solve_sqrt_equation_l972_97254


namespace triangle_third_side_length_triangle_third_side_length_proof_l972_97206

/-- Given a triangle with perimeter 160 and two sides of lengths 40 and 50,
    the length of the third side is 70. -/
theorem triangle_third_side_length : ℝ → ℝ → ℝ → Prop :=
  fun (perimeter side1 side2 : ℝ) =>
    perimeter = 160 ∧ side1 = 40 ∧ side2 = 50 →
    ∃ (side3 : ℝ), side3 = 70 ∧ perimeter = side1 + side2 + side3

/-- Proof of the theorem -/
theorem triangle_third_side_length_proof :
  triangle_third_side_length 160 40 50 := by
  sorry

end triangle_third_side_length_triangle_third_side_length_proof_l972_97206


namespace christine_distance_l972_97251

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Christine's distance traveled -/
theorem christine_distance :
  let speed : ℝ := 20
  let time : ℝ := 4
  distance_traveled speed time = 80 := by sorry

end christine_distance_l972_97251


namespace amusement_park_payment_l972_97233

def regular_ticket_cost : ℕ := 109
def child_discount : ℕ := 5
def change_received : ℕ := 74

def family_ticket_cost : ℕ := 
  (regular_ticket_cost - child_discount) * 2 + regular_ticket_cost * 2

theorem amusement_park_payment : 
  family_ticket_cost + change_received = 500 := by
  sorry

end amusement_park_payment_l972_97233


namespace connie_initial_marbles_l972_97274

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 70

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 3

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_initial_marbles : initial_marbles = 73 := by
  sorry

end connie_initial_marbles_l972_97274


namespace total_area_is_36_l972_97292

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- The size of the square grid -/
def gridSize : ℕ := 6

/-- The center point of the grid -/
def gridCenter : Point := { x := 3, y := 3 }

/-- Calculates the area of a triangle given its three points -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Generates all triangles formed by connecting the center to adjacent perimeter points -/
def perimeterTriangles : List Triangle := sorry

/-- Theorem: The total area of triangles formed by connecting the center of a 6x6 square grid
    to each pair of adjacent vertices along the perimeter is equal to 36 -/
theorem total_area_is_36 : 
  (perimeterTriangles.map triangleArea).sum = 36 := by sorry

end total_area_is_36_l972_97292


namespace no_damaged_pool_floats_l972_97290

/-- Prove that the number of damaged pool floats is 0 given the following conditions:
  - Total donations: 300
  - Basketball hoops: 60
  - Half of basketball hoops came with basketballs
  - Pool floats donated: 120
  - Footballs: 50
  - Tennis balls: 40
  - Remaining donations were basketballs
-/
theorem no_damaged_pool_floats (total_donations : ℕ) (basketball_hoops : ℕ) (pool_floats : ℕ)
  (footballs : ℕ) (tennis_balls : ℕ) (h1 : total_donations = 300)
  (h2 : basketball_hoops = 60) (h3 : pool_floats = 120) (h4 : footballs = 50) (h5 : tennis_balls = 40)
  (h6 : 2 * (basketball_hoops / 2) + pool_floats + footballs + tennis_balls +
    (total_donations - (basketball_hoops + pool_floats + footballs + tennis_balls)) = total_donations) :
  total_donations - (basketball_hoops + pool_floats + footballs + tennis_balls) = pool_floats := by
  sorry

#check no_damaged_pool_floats

end no_damaged_pool_floats_l972_97290


namespace smallest_interesting_rectangle_area_l972_97278

/-- A rectangle is interesting if it has integer side lengths and contains
    exactly four lattice points strictly in its interior. -/
def is_interesting (a b : ℕ) : Prop :=
  (a - 1) * (b - 1) = 4

/-- The area of the smallest interesting rectangle is 10. -/
theorem smallest_interesting_rectangle_area : 
  (∃ a b : ℕ, is_interesting a b ∧ a * b = 10) ∧ 
  (∀ a b : ℕ, is_interesting a b → a * b ≥ 10) :=
sorry

end smallest_interesting_rectangle_area_l972_97278


namespace sqrt_15_range_l972_97209

theorem sqrt_15_range : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by
  sorry

end sqrt_15_range_l972_97209


namespace spike_morning_crickets_l972_97223

/-- The number of crickets Spike hunts in the morning. -/
def morning_crickets : ℕ := 5

/-- The number of crickets Spike hunts in the afternoon and evening. -/
def afternoon_evening_crickets : ℕ := 3 * morning_crickets

/-- The total number of crickets Spike hunts per day. -/
def total_crickets : ℕ := 20

/-- Theorem stating that the number of crickets Spike hunts in the morning is 5. -/
theorem spike_morning_crickets :
  morning_crickets = 5 ∧
  afternoon_evening_crickets = 3 * morning_crickets ∧
  total_crickets = morning_crickets + afternoon_evening_crickets ∧
  total_crickets = 20 :=
by sorry

end spike_morning_crickets_l972_97223
