import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3389_338942

/-- The value of m for which the line x + y + m = 0 is tangent to the circle x² + y² = m -/
theorem tangent_line_to_circle (m : ℝ) : 
  (∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 ≠ m) ∧ 
  (∃ x y : ℝ, x + y + m = 0 ∧ x^2 + y^2 = m) → 
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3389_338942


namespace NUMINAMATH_CALUDE_rectangular_field_fencing_l3389_338945

theorem rectangular_field_fencing (area : ℝ) (uncovered_side : ℝ) : 
  area = 210 → uncovered_side = 20 → 
  2 * (area / uncovered_side) + uncovered_side = 41 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fencing_l3389_338945


namespace NUMINAMATH_CALUDE_maci_school_supplies_cost_l3389_338937

/-- Calculates the total cost of school supplies with discounts applied --/
def calculate_total_cost (blue_pen_count : ℕ) (red_pen_count : ℕ) (pencil_count : ℕ) (notebook_count : ℕ) 
  (blue_pen_price : ℚ) (pen_discount_threshold : ℕ) (pen_discount_rate : ℚ) 
  (notebook_discount_threshold : ℕ) (notebook_discount_rate : ℚ) : ℚ :=
  let red_pen_price := 2 * blue_pen_price
  let pencil_price := red_pen_price / 2
  let notebook_price := 10 * blue_pen_price
  
  let total_pen_cost := blue_pen_count * blue_pen_price + red_pen_count * red_pen_price
  let pencil_cost := pencil_count * pencil_price
  let notebook_cost := notebook_count * notebook_price
  
  let pen_discount := if blue_pen_count + red_pen_count > pen_discount_threshold 
                      then pen_discount_rate * total_pen_cost 
                      else 0
  let notebook_discount := if notebook_count > notebook_discount_threshold 
                           then notebook_discount_rate * notebook_cost 
                           else 0
  
  total_pen_cost + pencil_cost + notebook_cost - pen_discount - notebook_discount

/-- Theorem stating that the total cost of Maci's school supplies is $7.10 --/
theorem maci_school_supplies_cost :
  calculate_total_cost 10 15 5 3 (10/100) 12 (10/100) 4 (20/100) = 71/10 := by
  sorry

end NUMINAMATH_CALUDE_maci_school_supplies_cost_l3389_338937


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l3389_338998

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 > 0 → speed2 > 0 → (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km in the first hour and 60 km in the second hour is 75 km/h -/
theorem car_average_speed : 
  let speed1 := 90
  let speed2 := 60
  (speed1 + speed2) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l3389_338998


namespace NUMINAMATH_CALUDE_problem_statement_l3389_338901

theorem problem_statement : (2112 - 2021)^2 / 169 = 49 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3389_338901


namespace NUMINAMATH_CALUDE_roots_sum_problem_l3389_338923

theorem roots_sum_problem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → b^2 - 5*b + 6 = 0 → a^4 + a^5*b^3 + a^3*b^5 + b^4 = 2905 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_problem_l3389_338923


namespace NUMINAMATH_CALUDE_simplify_sum_of_powers_l3389_338931

theorem simplify_sum_of_powers : 2^2 + 2^2 + 2^2 + 2^2 = 2^4 := by sorry

end NUMINAMATH_CALUDE_simplify_sum_of_powers_l3389_338931


namespace NUMINAMATH_CALUDE_fraction_simplification_l3389_338949

theorem fraction_simplification : (5 - 2) / (2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3389_338949


namespace NUMINAMATH_CALUDE_train_length_l3389_338956

/-- The length of a train given its speed and time to cross a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 64 * (5 / 18) → time = 9 → speed * time = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3389_338956


namespace NUMINAMATH_CALUDE_johns_change_l3389_338966

/-- The change John receives when buying barbells -/
def change_received (num_barbells : ℕ) (barbell_cost : ℕ) (money_given : ℕ) : ℕ :=
  money_given - (num_barbells * barbell_cost)

/-- Theorem: John's change when buying 3 barbells at $270 each and giving $850 is $40 -/
theorem johns_change :
  change_received 3 270 850 = 40 := by
  sorry

end NUMINAMATH_CALUDE_johns_change_l3389_338966


namespace NUMINAMATH_CALUDE_candy_distribution_l3389_338954

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 22 → num_bags = 2 → candy_per_bag = total_candy / num_bags → candy_per_bag = 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3389_338954


namespace NUMINAMATH_CALUDE_smallest_three_digit_divisible_l3389_338967

theorem smallest_three_digit_divisible : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (6 ∣ n) ∧ (5 ∣ n) ∧ (8 ∣ n) ∧ (9 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (6 ∣ m) ∧ (5 ∣ m) ∧ (8 ∣ m) ∧ (9 ∣ m) → n ≤ m) ∧
  n = 360 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_divisible_l3389_338967


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l3389_338906

/-- Represents the lottery with MegaBall, WinnerBalls, and BonusBall -/
structure Lottery where
  megaBallCount : ℕ
  winnerBallCount : ℕ
  bonusBallCount : ℕ
  winnerBallsPicked : ℕ

/-- Calculates the probability of winning the lottery -/
def winningProbability (l : Lottery) : ℚ :=
  1 / (l.megaBallCount * (l.winnerBallCount.choose l.winnerBallsPicked) * l.bonusBallCount)

/-- The specific lottery configuration -/
def ourLottery : Lottery :=
  { megaBallCount := 30
    winnerBallCount := 50
    bonusBallCount := 15
    winnerBallsPicked := 5 }

/-- Theorem stating the probability of winning our specific lottery -/
theorem lottery_winning_probability :
    winningProbability ourLottery = 1 / 953658000 := by
  sorry

#eval winningProbability ourLottery

end NUMINAMATH_CALUDE_lottery_winning_probability_l3389_338906


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l3389_338939

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 12*a*b) :
  |((a+b)/(a-b))| = Real.sqrt (7/5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l3389_338939


namespace NUMINAMATH_CALUDE_chord_intercept_l3389_338964

/-- The value of 'a' in the equation of a line that intercepts a chord of length √3 on a circle -/
theorem chord_intercept (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 3 ∧ x + y + a = 0) →  -- Line intersects circle
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + y1^2 = 3 ∧ x2^2 + y2^2 = 3 ∧  -- Two points on circle
    x1 + y1 + a = 0 ∧ x2 + y2 + a = 0 ∧  -- Two points on line
    (x1 - x2)^2 + (y1 - y2)^2 = 3) →  -- Distance between points is √3
  a = 3 * Real.sqrt 2 / 2 ∨ a = -3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_chord_intercept_l3389_338964


namespace NUMINAMATH_CALUDE_tan_identity_implies_cos_squared_l3389_338920

theorem tan_identity_implies_cos_squared (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 4) : 
  Real.cos (θ + π/4)^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_tan_identity_implies_cos_squared_l3389_338920


namespace NUMINAMATH_CALUDE_total_cost_is_1027_2_l3389_338928

/-- The cost relationship between mangos, rice, and flour -/
structure CostRelationship where
  mango_cost : ℝ  -- Cost per kg of mangos
  rice_cost : ℝ   -- Cost per kg of rice
  flour_cost : ℝ  -- Cost per kg of flour
  mango_rice_relation : 10 * mango_cost = 24 * rice_cost
  flour_rice_relation : 6 * flour_cost = 2 * rice_cost
  flour_cost_value : flour_cost = 24

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost (cr : CostRelationship) : ℝ :=
  4 * cr.mango_cost + 3 * cr.rice_cost + 5 * cr.flour_cost

/-- Theorem stating that the total cost is $1027.2 -/
theorem total_cost_is_1027_2 (cr : CostRelationship) :
  total_cost cr = 1027.2 := by
  sorry

#check total_cost_is_1027_2

end NUMINAMATH_CALUDE_total_cost_is_1027_2_l3389_338928


namespace NUMINAMATH_CALUDE_fraction_of_girls_l3389_338907

theorem fraction_of_girls (total_students : ℕ) (boys : ℕ) (h1 : total_students = 160) (h2 : boys = 120) :
  (total_students - boys : ℚ) / total_students = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_girls_l3389_338907


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3389_338919

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 2023)) ↔ x ≠ 2023 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3389_338919


namespace NUMINAMATH_CALUDE_crayon_count_l3389_338978

theorem crayon_count (small_left medium_left large_left : ℕ) 
  (h_small : small_left = 60)
  (h_medium : medium_left = 98)
  (h_large : large_left = 168) :
  ∃ (small_initial medium_initial large_initial : ℕ),
    small_initial = 100 ∧
    medium_initial = 392 ∧
    large_initial = 294 ∧
    small_left = (3 : ℚ) / 5 * small_initial ∧
    medium_left = (1 : ℚ) / 4 * medium_initial ∧
    large_left = (4 : ℚ) / 7 * large_initial ∧
    (2 : ℚ) / 5 * small_initial + 
    (3 : ℚ) / 4 * medium_initial + 
    (3 : ℚ) / 7 * large_initial = 460 := by
  sorry


end NUMINAMATH_CALUDE_crayon_count_l3389_338978


namespace NUMINAMATH_CALUDE_sequence_general_formula_l3389_338908

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 3 + 2^n,
    this theorem proves the general formula for a_n. -/
theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 3 + 2^n) :
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l3389_338908


namespace NUMINAMATH_CALUDE_terminal_side_symmetry_ratio_l3389_338995

theorem terminal_side_symmetry_ratio (θ : Real) (x y : Real) :
  θ ∈ Set.Ioo 0 360 →
  -- Terminal side of θ is symmetric to terminal side of 660° w.r.t. x-axis
  (∃ k : ℤ, θ + 660 = 360 * (2 * k + 1)) →
  x ≠ 0 ∨ y ≠ 0 →  -- P(x, y) is not the origin
  y / x = Real.tan θ →  -- P(x, y) is on the terminal side of θ
  x * y / (x^2 + y^2) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_symmetry_ratio_l3389_338995


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l3389_338969

/-- Represents the number of recommendation spots for each language --/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the gender distribution of candidates --/
structure CandidateGenders :=
  (males : Nat)
  (females : Nat)

/-- Calculates the number of different recommendation plans --/
def count_recommendation_plans (spots : RecommendationSpots) (genders : CandidateGenders) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem recommendation_plans_count :
  let spots := RecommendationSpots.mk 2 2 1
  let genders := CandidateGenders.mk 3 2
  count_recommendation_plans spots genders = 24 := by
  sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l3389_338969


namespace NUMINAMATH_CALUDE_exponential_equation_implication_l3389_338997

theorem exponential_equation_implication (x : ℝ) : 
  4 * (3 : ℝ)^x = 2187 → (x + 2) * (x - 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_implication_l3389_338997


namespace NUMINAMATH_CALUDE_salary_calculation_l3389_338988

theorem salary_calculation (salary : ℝ) 
  (food_expense : salary * (1 / 5) = salary / 5)
  (rent_expense : salary * (1 / 10) = salary / 10)
  (clothes_expense : salary * (3 / 5) = 3 * salary / 5)
  (remaining : salary - (salary / 5 + salary / 10 + 3 * salary / 5) = 14000) :
  salary = 140000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l3389_338988


namespace NUMINAMATH_CALUDE_hotel_elevator_cubic_at_15_l3389_338991

/-- The hotel elevator cubic polynomial -/
def hotel_elevator_cubic (P : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, P x = a*x^3 + b*x^2 + c*x + d) ∧
  P 11 = 11 ∧ P 12 = 12 ∧ P 13 = 14 ∧ P 14 = 15

theorem hotel_elevator_cubic_at_15 (P : ℝ → ℝ) (h : hotel_elevator_cubic P) : P 15 = 13 := by
  sorry

end NUMINAMATH_CALUDE_hotel_elevator_cubic_at_15_l3389_338991


namespace NUMINAMATH_CALUDE_sixth_power_sum_l3389_338927

theorem sixth_power_sum (r : ℝ) (h : (r + 1/r)^4 = 17) : 
  r^6 + 1/r^6 = Real.sqrt 17 - 6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l3389_338927


namespace NUMINAMATH_CALUDE_willies_stickers_l3389_338970

/-- Willie's sticker problem -/
theorem willies_stickers (initial : ℕ) (remaining : ℕ) (given : ℕ) : 
  initial = 36 → remaining = 29 → given = initial - remaining :=
by sorry

end NUMINAMATH_CALUDE_willies_stickers_l3389_338970


namespace NUMINAMATH_CALUDE_bag_probability_l3389_338915

theorem bag_probability (d x : ℕ) : 
  d = x + (x + 1) + (x + 2) →
  (x : ℚ) / d < 1 / 6 →
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_bag_probability_l3389_338915


namespace NUMINAMATH_CALUDE_muirhead_inequality_l3389_338940

open Real

/-- Muirhead's Inequality -/
theorem muirhead_inequality (a₁ a₂ a₃ b₁ b₂ b₃ x y z : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃ ∧ a₃ ≥ 0)
  (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃ ∧ b₃ ≥ 0)
  (hab : a₁ ≥ b₁ ∧ a₁ + a₂ ≥ b₁ + b₂ ∧ a₁ + a₂ + a₃ ≥ b₁ + b₂ + b₃)
  (hxyz : x > 0 ∧ y > 0 ∧ z > 0) : 
  x^a₁ * y^a₂ * z^a₃ + x^a₁ * y^a₃ * z^a₂ + x^a₂ * y^a₁ * z^a₃ + 
  x^a₂ * y^a₃ * z^a₁ + x^a₃ * y^a₁ * z^a₂ + x^a₃ * y^a₂ * z^a₁ ≥ 
  x^b₁ * y^b₂ * z^b₃ + x^b₁ * y^b₃ * z^b₂ + x^b₂ * y^b₁ * z^b₃ + 
  x^b₂ * y^b₃ * z^b₁ + x^b₃ * y^b₁ * z^b₂ + x^b₃ * y^b₂ * z^b₁ :=
sorry

end NUMINAMATH_CALUDE_muirhead_inequality_l3389_338940


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3389_338980

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) :
  ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0),
    (∀ (u v : ℝ) (hu : u > 0) (hv : v > 0),
      Real.sqrt 3 = Real.sqrt (3^u * 3^(2*v)) →
      2/u + 1/v ≥ 2/x + 1/y) ∧
    2/x + 1/y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3389_338980


namespace NUMINAMATH_CALUDE_probability_is_half_l3389_338944

/-- A game where a square is divided into triangular sections and some are shaded -/
structure SquareGame where
  total_sections : ℕ
  shaded_sections : ℕ
  h_total : total_sections = 8
  h_shaded : shaded_sections = 4

/-- The probability of landing on a shaded section -/
def probability_shaded (game : SquareGame) : ℚ :=
  game.shaded_sections / game.total_sections

/-- Theorem: The probability of landing on a shaded section is 1/2 -/
theorem probability_is_half (game : SquareGame) : probability_shaded game = 1/2 := by
  sorry

#eval probability_shaded { total_sections := 8, shaded_sections := 4, h_total := rfl, h_shaded := rfl }

end NUMINAMATH_CALUDE_probability_is_half_l3389_338944


namespace NUMINAMATH_CALUDE_skater_speed_l3389_338952

/-- Given a skater who travels 80 kilometers in 8 hours, prove their speed is 10 kilometers per hour. -/
theorem skater_speed (distance : ℝ) (time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 := by sorry

end NUMINAMATH_CALUDE_skater_speed_l3389_338952


namespace NUMINAMATH_CALUDE_probability_four_even_dice_l3389_338934

theorem probability_four_even_dice (n : ℕ) (p : ℚ) : 
  n = 8 →
  p = 1/2 →
  (n.choose (n/2)) * p^n = 35/128 :=
by sorry

end NUMINAMATH_CALUDE_probability_four_even_dice_l3389_338934


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3389_338943

/-- The quadratic equation (k-2)x^2 + 3x + k^2 - 4 = 0 has one solution as x = 0 -/
def has_zero_solution (k : ℝ) : Prop :=
  k^2 - 4 = 0

/-- The coefficient of x^2 is not zero -/
def is_quadratic (k : ℝ) : Prop :=
  k - 2 ≠ 0

theorem quadratic_equation_solution :
  ∀ k : ℝ, has_zero_solution k → is_quadratic k → k = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3389_338943


namespace NUMINAMATH_CALUDE_inequality_proof_l3389_338994

/-- Given a function f: ℝ → ℝ with derivative f', such that ∀ x ∈ ℝ, f x > f' x,
    prove that 2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) -/
theorem inequality_proof (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : ∀ x : ℝ, HasDerivAt f (f' x) x)
    (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3389_338994


namespace NUMINAMATH_CALUDE_sum_difference_implies_sum_l3389_338986

theorem sum_difference_implies_sum (a b : ℕ+) : 
  (a.val * b.val * (a.val * b.val + 1)) / 2 - 
  (a.val * (a.val + 1) * b.val * (b.val + 1)) / 4 = 1200 →
  a.val + b.val = 21 := by sorry

end NUMINAMATH_CALUDE_sum_difference_implies_sum_l3389_338986


namespace NUMINAMATH_CALUDE_like_terms_exponent_l3389_338921

theorem like_terms_exponent (a : ℝ) : (∃ x : ℝ, x ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ k * x^(2*a) = 5 * x^(a+3)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l3389_338921


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l3389_338957

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.10 * last_year_earnings
  let this_year_earnings := 1.15 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 345 :=
by sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l3389_338957


namespace NUMINAMATH_CALUDE_exactly_two_more_heads_probability_l3389_338904

/-- The number of coins being flipped -/
def num_coins : ℕ := 10

/-- The number of heads required to have exactly two more heads than tails -/
def required_heads : ℕ := (num_coins + 2) / 2

/-- The probability of getting heads on a single fair coin flip -/
def prob_heads : ℚ := 1 / 2

theorem exactly_two_more_heads_probability :
  (Nat.choose num_coins required_heads : ℚ) * prob_heads ^ required_heads * (1 - prob_heads) ^ (num_coins - required_heads) = 210 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_more_heads_probability_l3389_338904


namespace NUMINAMATH_CALUDE_grade_assignment_count_l3389_338965

theorem grade_assignment_count : (4 : ℕ) ^ 15 = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l3389_338965


namespace NUMINAMATH_CALUDE_inequality_proof_l3389_338985

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3389_338985


namespace NUMINAMATH_CALUDE_uncovered_area_of_overlapping_squares_l3389_338979

theorem uncovered_area_of_overlapping_squares :
  ∀ (large_side small_side : ℝ),
    large_side = 10 →
    small_side = 4 →
    large_side > 0 →
    small_side > 0 →
    large_side ≥ small_side →
    (large_side ^ 2 - small_side ^ 2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_of_overlapping_squares_l3389_338979


namespace NUMINAMATH_CALUDE_paris_total_study_hours_l3389_338910

/-- The number of hours Paris studies during the semester -/
def paris_study_hours : ℕ :=
  let weeks_in_semester : ℕ := 15
  let weekday_study_hours : ℕ := 3
  let saturday_study_hours : ℕ := 4
  let sunday_study_hours : ℕ := 5
  let weekdays_per_week : ℕ := 5
  let weekly_study_hours : ℕ := weekday_study_hours * weekdays_per_week + saturday_study_hours + sunday_study_hours
  weekly_study_hours * weeks_in_semester

theorem paris_total_study_hours : paris_study_hours = 360 := by
  sorry

end NUMINAMATH_CALUDE_paris_total_study_hours_l3389_338910


namespace NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l3389_338946

theorem angle_sum_ninety_degrees (A B : Real) (h : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2) :
  A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l3389_338946


namespace NUMINAMATH_CALUDE_square_area_15cm_l3389_338914

/-- The area of a square with side length 15 cm is 225 square centimeters. -/
theorem square_area_15cm (side_length : ℝ) (area : ℝ) : 
  side_length = 15 → area = side_length ^ 2 → area = 225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_15cm_l3389_338914


namespace NUMINAMATH_CALUDE_laptop_gifting_l3389_338971

theorem laptop_gifting (n m : ℕ) (hn : n = 15) (hm : m = 3) :
  (n.factorial / (n - m).factorial) = 2730 := by
  sorry

end NUMINAMATH_CALUDE_laptop_gifting_l3389_338971


namespace NUMINAMATH_CALUDE_parallel_lines_solution_l3389_338948

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line equation: ax + 2y + 6 = 0 -/
def line1 (a x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

/-- The second line equation: x + (a-1)y + (a^2-1) = 0 -/
def line2 (a x y : ℝ) : Prop :=
  x + (a - 1) * y + (a^2 - 1) = 0

/-- The theorem stating that given the two parallel lines, a = -1 -/
theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, parallel_lines a 2 1 (a-1) 1 (a^2-1)) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_solution_l3389_338948


namespace NUMINAMATH_CALUDE_sequence_general_term_l3389_338958

theorem sequence_general_term (n : ℕ+) : 
  let a : ℕ+ → ℝ := fun i => Real.sqrt i
  a n = Real.sqrt n := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3389_338958


namespace NUMINAMATH_CALUDE_circles_intersection_tangent_equality_points_l3389_338913

-- Define the circles and ellipse
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 6*y + 32 = 0
def C2 (a x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*(8-a)*y + 4*a + 12 = 0
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Theorem for part (I)
theorem circles_intersection :
  ∀ a : ℝ, C1 4 2 ∧ C1 6 4 ∧ C2 a 4 2 ∧ C2 a 6 4 := by sorry

-- Theorem for part (II)
theorem tangent_equality_points :
  ∀ x y : ℝ, Ellipse x y →
    (∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧
      (x^2 + y^2 - 10*x - 6*y + 32 = x^2 + y^2 - 2*a₁*x - 2*(8-a₁)*y + 4*a₁ + 12) ∧
      (x^2 + y^2 - 10*x - 6*y + 32 = x^2 + y^2 - 2*a₂*x - 2*(8-a₂)*y + 4*a₂ + 12)) ↔
    ((x = 2 ∧ y = 0) ∨ (x = 6/5 ∧ y = -4/5)) := by sorry

end NUMINAMATH_CALUDE_circles_intersection_tangent_equality_points_l3389_338913


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3389_338922

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.a * l2.b = l2.a * l1.b

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (point : Point) :
  ∃ (result_line : Line),
    parallel result_line given_line ∧
    on_line point result_line ∧
    result_line.a = 1 ∧
    result_line.b = -2 ∧
    result_line.c = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3389_338922


namespace NUMINAMATH_CALUDE_value_of_a_l3389_338955

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}
def A (a : ℝ) : Set ℝ := {2, |a + 1|}

theorem value_of_a (a : ℝ) : U a = A a ∪ {5} → a = -4 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3389_338955


namespace NUMINAMATH_CALUDE_candy_box_total_l3389_338916

theorem candy_box_total (purple orange yellow total : ℕ) : 
  purple + orange + yellow = total →
  2 * orange = 4 * purple →
  5 * purple = 2 * yellow →
  yellow = 40 →
  total = 88 := by
sorry

end NUMINAMATH_CALUDE_candy_box_total_l3389_338916


namespace NUMINAMATH_CALUDE_circle_power_theorem_l3389_338918

structure Circle where
  a : ℝ
  b : ℝ
  R : ℝ

def power (c : Circle) (x₁ y₁ : ℝ) : ℝ :=
  (x₁ - c.a)^2 + (y₁ - c.b)^2 - c.R^2

def distance_squared (x₁ y₁ a b : ℝ) : ℝ :=
  (x₁ - a)^2 + (y₁ - b)^2

theorem circle_power_theorem (c : Circle) (x₁ y₁ : ℝ) :
  -- 1. Power definition
  power c x₁ y₁ = (x₁ - c.a)^2 + (y₁ - c.b)^2 - c.R^2 ∧
  -- 2. Power sign properties
  (distance_squared x₁ y₁ c.a c.b > c.R^2 → power c x₁ y₁ > 0) ∧
  (distance_squared x₁ y₁ c.a c.b < c.R^2 → power c x₁ y₁ < 0) ∧
  (distance_squared x₁ y₁ c.a c.b = c.R^2 → power c x₁ y₁ = 0) ∧
  -- 3. Tangent length property
  (distance_squared x₁ y₁ c.a c.b > c.R^2 → 
    ∃ p, p^2 = power c x₁ y₁ ∧ p ≥ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_power_theorem_l3389_338918


namespace NUMINAMATH_CALUDE_paper_piles_theorem_l3389_338977

theorem paper_piles_theorem (n : ℕ) :
  1000 < n ∧ n < 2000 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) →
  ∃ m : ℕ, m = 41 ∧ m ≠ 1 ∧ m ≠ n ∧ n % m = 0 :=
by sorry

end NUMINAMATH_CALUDE_paper_piles_theorem_l3389_338977


namespace NUMINAMATH_CALUDE_patio_tiles_l3389_338936

theorem patio_tiles (c : ℕ) (h1 : c > 2) : 
  c * 10 = (c - 2) * (10 + 4) → c * 10 = 70 := by
  sorry

#check patio_tiles

end NUMINAMATH_CALUDE_patio_tiles_l3389_338936


namespace NUMINAMATH_CALUDE_jills_age_l3389_338981

/-- Given that the sum of Henry and Jill's present ages is 41, and 7 years ago Henry was twice the age of Jill, prove that Jill's present age is 16 years. -/
theorem jills_age (henry_age jill_age : ℕ) 
  (sum_of_ages : henry_age + jill_age = 41)
  (past_relation : henry_age - 7 = 2 * (jill_age - 7)) : 
  jill_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_jills_age_l3389_338981


namespace NUMINAMATH_CALUDE_min_value_of_u_l3389_338909

/-- Given that x and y are real numbers satisfying 2x + y ≥ 1, 
    the function u = x² + 4x + y² - 2y has a minimum value of -9/5 -/
theorem min_value_of_u (x y : ℝ) (h : 2 * x + y ≥ 1) :
  ∃ (min_u : ℝ), min_u = -9/5 ∧ ∀ (x' y' : ℝ), 2 * x' + y' ≥ 1 → 
    x'^2 + 4*x' + y'^2 - 2*y' ≥ min_u :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_u_l3389_338909


namespace NUMINAMATH_CALUDE_teal_color_perception_l3389_338932

theorem teal_color_perception (total : ℕ) (greenish : ℕ) (both : ℕ) (neither : ℕ) :
  total = 120 →
  greenish = 80 →
  both = 35 →
  neither = 20 →
  ∃ bluish : ℕ, bluish = 55 ∧ bluish = total - (greenish - both) - both - neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_perception_l3389_338932


namespace NUMINAMATH_CALUDE_simplify_fraction_l3389_338903

theorem simplify_fraction : (18 : ℚ) * (8 / 12) * (1 / 9) * 4 = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3389_338903


namespace NUMINAMATH_CALUDE_calculate_running_speed_l3389_338990

/-- Given a swimming speed and an average speed for swimming and running,
    calculate the running speed. -/
theorem calculate_running_speed
  (swimming_speed : ℝ)
  (average_speed : ℝ)
  (h1 : swimming_speed = 1)
  (h2 : average_speed = 4.5)
  : (2 * average_speed - swimming_speed) = 8 := by
  sorry

#check calculate_running_speed

end NUMINAMATH_CALUDE_calculate_running_speed_l3389_338990


namespace NUMINAMATH_CALUDE_billy_score_is_13_l3389_338941

/-- Represents a contestant's performance on the AMC 8 contest -/
structure AMC8Performance where
  total_questions : Nat
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered : Nat
  correct_point_value : Nat
  incorrect_point_value : Nat
  unanswered_point_value : Nat

/-- Calculates the score for an AMC 8 performance -/
def calculate_score (performance : AMC8Performance) : Nat :=
  performance.correct_answers * performance.correct_point_value +
  performance.incorrect_answers * performance.incorrect_point_value +
  performance.unanswered * performance.unanswered_point_value

/-- Billy's performance on the AMC 8 contest -/
def billy_performance : AMC8Performance := {
  total_questions := 25,
  correct_answers := 13,
  incorrect_answers := 7,
  unanswered := 5,
  correct_point_value := 1,
  incorrect_point_value := 0,
  unanswered_point_value := 0
}

theorem billy_score_is_13 : calculate_score billy_performance = 13 := by
  sorry

end NUMINAMATH_CALUDE_billy_score_is_13_l3389_338941


namespace NUMINAMATH_CALUDE_arcade_ticket_difference_l3389_338962

def arcade_tickets : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ := fun initial toys clothes food accessories =>
  let food_discounted := food * 0.85
  let combined := clothes + food_discounted + accessories
  combined - toys

theorem arcade_ticket_difference : arcade_tickets 250 58 85 60 45.5 = 123.5 := by
  sorry

end NUMINAMATH_CALUDE_arcade_ticket_difference_l3389_338962


namespace NUMINAMATH_CALUDE_trash_can_purchase_l3389_338996

/-- Represents the unit price of trash can type A -/
def price_A : ℕ := 500

/-- Represents the unit price of trash can type B -/
def price_B : ℕ := 550

/-- Represents the total number of trash cans to be purchased -/
def total_cans : ℕ := 6

/-- Represents the maximum total cost allowed -/
def max_cost : ℕ := 3100

/-- Theorem stating the correct unit prices and purchase options -/
theorem trash_can_purchase :
  (price_B = price_A + 50) ∧
  (2000 / price_A = 2200 / price_B) ∧
  (∀ a b : ℕ, 
    a + b = total_cans ∧ 
    price_A * a + price_B * b ≤ max_cost ∧
    a ≥ 0 ∧ b ≥ 0 →
    (a = 4 ∧ b = 2) ∨ (a = 5 ∧ b = 1) ∨ (a = 6 ∧ b = 0)) := by
  sorry

end NUMINAMATH_CALUDE_trash_can_purchase_l3389_338996


namespace NUMINAMATH_CALUDE_last_four_digits_5_2011_l3389_338975

-- Define a function to get the last four digits of a number
def lastFourDigits (n : ℕ) : ℕ := n % 10000

-- Define the cycle length of the last four digits of powers of 5
def cycleLengthPowersOf5 : ℕ := 4

-- Theorem statement
theorem last_four_digits_5_2011 :
  lastFourDigits (5^2011) = lastFourDigits (5^7) :=
by
  sorry

#eval lastFourDigits (5^7)  -- This should output 8125

end NUMINAMATH_CALUDE_last_four_digits_5_2011_l3389_338975


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3389_338900

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ (m : ℝ), m = 4 ∧ ∀ x y : ℝ, x + y = 2 → 2^x + 2^y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3389_338900


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l3389_338992

theorem inscribed_rectangle_circle_circumference 
  (width : Real) (height : Real) (circle : Real → Prop) 
  (rectangle : Real → Real → Prop) (circumference : Real) :
  width = 9 →
  height = 12 →
  rectangle width height →
  (∀ x y, rectangle x y → circle (Real.sqrt (x^2 + y^2))) →
  circumference = Real.pi * Real.sqrt (width^2 + height^2) →
  circumference = 15 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l3389_338992


namespace NUMINAMATH_CALUDE_exist_numbers_same_divisors_less_sum_l3389_338976

/-- The number of natural divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The sum of all natural divisors of a natural number -/
def sum_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- There exist two natural numbers with the same number of divisors,
    where one is greater than the other, but has a smaller sum of divisors -/
theorem exist_numbers_same_divisors_less_sum :
  ∃ x y : ℕ, x > y ∧ num_divisors x = num_divisors y ∧ sum_divisors x < sum_divisors y :=
sorry

end NUMINAMATH_CALUDE_exist_numbers_same_divisors_less_sum_l3389_338976


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3389_338935

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3389_338935


namespace NUMINAMATH_CALUDE_g_range_l3389_338974

/-- The function representing the curve C -/
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 3)

/-- The function g(t) representing the product of magnitudes of OP and OQ -/
def g (t : ℝ) : ℝ := |(3 - t) * (1 + t^2)|

/-- Theorem stating that the range of g(t) is [0, ∞) -/
theorem g_range : Set.range g = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_g_range_l3389_338974


namespace NUMINAMATH_CALUDE_fencing_requirement_l3389_338963

/-- A rectangular field with one side of 20 feet and an area of 80 sq. feet requires 28 feet of fencing for the other three sides. -/
theorem fencing_requirement (length width : ℝ) : 
  length = 20 → 
  length * width = 80 → 
  length + 2 * width = 28 := by sorry

end NUMINAMATH_CALUDE_fencing_requirement_l3389_338963


namespace NUMINAMATH_CALUDE_length_of_A_prime_B_prime_l3389_338930

/-- Given points A, B, C, and the conditions for A' and B', prove that |A'B'| = 5√2 -/
theorem length_of_A_prime_B_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (∃ t : ℝ, A + t • (A' - A) = C) →
  (∃ s : ℝ, B + s • (B' - B) = C) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_length_of_A_prime_B_prime_l3389_338930


namespace NUMINAMATH_CALUDE_inequality_system_equivalence_l3389_338938

theorem inequality_system_equivalence :
  ∀ x : ℝ, (x + 1 ≥ 2 ∧ x > 0) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_equivalence_l3389_338938


namespace NUMINAMATH_CALUDE_cit_beaver_difference_l3389_338933

/-- A Beaver-number is a positive 5-digit integer whose digit sum is divisible by 17. -/
def is_beaver_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ (n.digits 10).sum % 17 = 0

/-- A Beaver-pair is a pair of consecutive Beaver-numbers. -/
def is_beaver_pair (m n : ℕ) : Prop :=
  is_beaver_number m ∧ is_beaver_number n ∧ n = m + 1

/-- An MIT Beaver is the smaller number in a Beaver-pair. -/
def is_mit_beaver (m : ℕ) : Prop :=
  ∃ n, is_beaver_pair m n

/-- A CIT Beaver is the larger number in a Beaver-pair. -/
def is_cit_beaver (n : ℕ) : Prop :=
  ∃ m, is_beaver_pair m n

/-- The theorem stating the difference between the maximum and minimum CIT Beaver numbers. -/
theorem cit_beaver_difference : 
  ∃ max min : ℕ, 
    is_cit_beaver max ∧ 
    is_cit_beaver min ∧ 
    (∀ n, is_cit_beaver n → n ≤ max) ∧ 
    (∀ n, is_cit_beaver n → min ≤ n) ∧ 
    max - min = 79200 :=
sorry

end NUMINAMATH_CALUDE_cit_beaver_difference_l3389_338933


namespace NUMINAMATH_CALUDE_intersection_A_B_l3389_338972

-- Define set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + 2*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3389_338972


namespace NUMINAMATH_CALUDE_insurance_covers_80_percent_l3389_338968

def number_of_vaccines : ℕ := 10
def cost_per_vaccine : ℚ := 45
def cost_of_doctors_visit : ℚ := 250
def trip_cost : ℚ := 1200
def toms_payment : ℚ := 1340

def total_medical_cost : ℚ := number_of_vaccines * cost_per_vaccine + cost_of_doctors_visit
def total_trip_cost : ℚ := trip_cost + total_medical_cost
def insurance_coverage : ℚ := total_trip_cost - toms_payment
def insurance_coverage_percentage : ℚ := insurance_coverage / total_medical_cost * 100

theorem insurance_covers_80_percent :
  insurance_coverage_percentage = 80 := by sorry

end NUMINAMATH_CALUDE_insurance_covers_80_percent_l3389_338968


namespace NUMINAMATH_CALUDE_algebraic_operation_equality_l3389_338950

theorem algebraic_operation_equality (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_operation_equality_l3389_338950


namespace NUMINAMATH_CALUDE_max_c_value_l3389_338984

theorem max_c_value (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) :
  c ≤ 47 ∧ ∃ d', 5 * 47 + (d' - 12)^2 = 235 := by
  sorry

end NUMINAMATH_CALUDE_max_c_value_l3389_338984


namespace NUMINAMATH_CALUDE_function_lower_bound_l3389_338924

open Real

/-- A function satisfying the given inequality for all real x -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x, Real.sqrt (2 * f x) - Real.sqrt (2 * f x - f (2 * x)) ≥ 2

/-- The main theorem to be proved -/
theorem function_lower_bound
  (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  ∀ x, f x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l3389_338924


namespace NUMINAMATH_CALUDE_distinct_choices_eq_eight_l3389_338983

/-- Represents the set of marbles Tom has -/
inductive Marble : Type
| Red : Marble
| Green : Marble
| Blue : Marble
| Yellow : Marble

/-- The number of each type of marble Tom has -/
def marbleCounts : Marble → ℕ
| Marble.Red => 1
| Marble.Green => 1
| Marble.Blue => 1
| Marble.Yellow => 4

/-- The total number of marbles Tom has -/
def totalMarbles : ℕ := (marbleCounts Marble.Red) + (marbleCounts Marble.Green) + 
                        (marbleCounts Marble.Blue) + (marbleCounts Marble.Yellow)

/-- A function to calculate the number of distinct ways to choose 3 marbles -/
def distinctChoices : ℕ := sorry

/-- Theorem stating that the number of distinct ways to choose 3 marbles is 8 -/
theorem distinct_choices_eq_eight : distinctChoices = 8 := by sorry

end NUMINAMATH_CALUDE_distinct_choices_eq_eight_l3389_338983


namespace NUMINAMATH_CALUDE_equation_solution_l3389_338960

theorem equation_solution : ∃! n : ℚ, (1 : ℚ) / (n + 1) + (2 : ℚ) / (n + 1) + n / (n + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3389_338960


namespace NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l3389_338959

def remaining_rolls_to_sell (total_required : ℕ) (sales_to_grandmother : ℕ) (sales_to_uncle : ℕ) (sales_to_neighbor : ℕ) : ℕ :=
  total_required - (sales_to_grandmother + sales_to_uncle + sales_to_neighbor)

theorem chandler_wrapping_paper_sales : 
  remaining_rolls_to_sell 12 3 4 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l3389_338959


namespace NUMINAMATH_CALUDE_students_liking_both_sports_and_music_l3389_338917

theorem students_liking_both_sports_and_music
  (total : ℕ)
  (sports : ℕ)
  (music : ℕ)
  (neither : ℕ)
  (h_total : total = 55)
  (h_sports : sports = 43)
  (h_music : music = 34)
  (h_neither : neither = 4) :
  ∃ (both : ℕ), both = sports + music - total + neither ∧ both = 26 :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_and_music_l3389_338917


namespace NUMINAMATH_CALUDE_rhombus_area_l3389_338993

/-- The area of a rhombus with side length 4 cm and an interior angle of 45 degrees is 8√2 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) : 
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3389_338993


namespace NUMINAMATH_CALUDE_signup_ways_eq_64_l3389_338951

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of interest groups --/
def num_groups : ℕ := 4

/-- The number of ways students can sign up for interest groups --/
def num_ways : ℕ := num_groups ^ num_students

/-- Theorem stating that the number of ways to sign up is 64 --/
theorem signup_ways_eq_64 : num_ways = 64 := by
  sorry

end NUMINAMATH_CALUDE_signup_ways_eq_64_l3389_338951


namespace NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l3389_338961

def S : Finset Nat := Finset.range 1999

def f (X : Finset Nat) : Nat :=
  X.sum id

theorem sum_over_subsets_equals_power_of_two :
  (Finset.powerset S).sum (fun E => (f E : ℚ) / (f S : ℚ)) = (2 : ℚ) ^ 1998 :=
sorry

end NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l3389_338961


namespace NUMINAMATH_CALUDE_min_value_theorem_l3389_338953

/-- The line equation ax - 2by = 2 passes through the center of the circle x² + y² - 4x + 2y + 1 = 0 -/
def line_passes_through_center (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 ∧ a*x - 2*b*y = 2

/-- The minimum value of 1/a + 1/b + 1/(ab) given the conditions -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_center : line_passes_through_center a b) : 
  (1/a + 1/b + 1/(a*b)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3389_338953


namespace NUMINAMATH_CALUDE_new_average_age_l3389_338925

theorem new_average_age (n : ℕ) (original_avg : ℝ) (new_person_age : ℝ) :
  n = 9 ∧ original_avg = 15 ∧ new_person_age = 35 →
  (n * original_avg + new_person_age) / (n + 1) = 17 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l3389_338925


namespace NUMINAMATH_CALUDE_museum_ring_display_height_l3389_338911

/-- Calculates the total vertical distance of a sequence of rings -/
def total_vertical_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let n := (top_diameter - bottom_diameter) / 2 + 1
  let sum_inside_diameters := n * (top_diameter - thickness + bottom_diameter - thickness) / 2
  sum_inside_diameters + 2 * thickness

/-- Theorem stating that the total vertical distance for the given ring sequence is 325 cm -/
theorem museum_ring_display_height : total_vertical_distance 36 4 1 = 325 := by
  sorry

#eval total_vertical_distance 36 4 1

end NUMINAMATH_CALUDE_museum_ring_display_height_l3389_338911


namespace NUMINAMATH_CALUDE_quantity_count_l3389_338947

theorem quantity_count (total_sum : ℝ) (total_count : ℕ) 
  (subset1_sum : ℝ) (subset1_count : ℕ) 
  (subset2_sum : ℝ) (subset2_count : ℕ) 
  (h1 : total_sum / total_count = 12)
  (h2 : subset1_sum / subset1_count = 4)
  (h3 : subset2_sum / subset2_count = 24)
  (h4 : subset1_count = 3)
  (h5 : subset2_count = 2)
  (h6 : total_sum = subset1_sum + subset2_sum)
  (h7 : total_count = subset1_count + subset2_count) : 
  total_count = 5 := by
sorry


end NUMINAMATH_CALUDE_quantity_count_l3389_338947


namespace NUMINAMATH_CALUDE_red_light_probability_l3389_338926

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of seeing a red light given the traffic light durations -/
def probability_red_light (d : TrafficLightDuration) : ℚ :=
  d.red / (d.red + d.yellow + d.green)

/-- Theorem: The probability of seeing a red light is 2/5 for the given durations -/
theorem red_light_probability (d : TrafficLightDuration) 
  (h_red : d.red = 30)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 40) : 
  probability_red_light d = 2/5 := by
  sorry

#eval probability_red_light ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_l3389_338926


namespace NUMINAMATH_CALUDE_translation_result_l3389_338982

def translate_point (x y dx dy : Int) : (Int × Int) :=
  (x + dx, y - dy)

theorem translation_result :
  let initial_point := (-2, 3)
  let x_translation := 3
  let y_translation := 2
  translate_point initial_point.1 initial_point.2 x_translation y_translation = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l3389_338982


namespace NUMINAMATH_CALUDE_max_value_polynomial_l3389_338905

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z : ℝ), x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ z) ∧
  (∀ (z : ℝ), x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ z → 6084/17 ≤ z) :=
sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l3389_338905


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l3389_338912

theorem smallest_positive_solution (x : ℕ) : 
  (∃ k : ℤ, 45 * x + 15 = 5 + 28 * k) ∧ 
  (∀ y : ℕ, y < x → ¬(∃ k : ℤ, 45 * y + 15 = 5 + 28 * k)) → 
  x = 18 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l3389_338912


namespace NUMINAMATH_CALUDE_garden_furniture_costs_l3389_338929

theorem garden_furniture_costs (total_cost bench_cost table_cost umbrella_cost : ℝ) :
  total_cost = 765 ∧
  table_cost = 2 * bench_cost ∧
  umbrella_cost = 3 * bench_cost →
  bench_cost = 127.5 ∧
  table_cost = 255 ∧
  umbrella_cost = 382.5 :=
by sorry

end NUMINAMATH_CALUDE_garden_furniture_costs_l3389_338929


namespace NUMINAMATH_CALUDE_equation_solution_l3389_338902

theorem equation_solution :
  ∃ x : ℚ, -8 * (2 - x)^3 = 27 ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3389_338902


namespace NUMINAMATH_CALUDE_ag_replacement_terminates_l3389_338999

/-- Represents a sequence of As and Gs -/
inductive AGSequence
| empty : AGSequence
| cons : Char → AGSequence → AGSequence

/-- Represents the operation of replacing "AG" with "GAAA" -/
def replaceAG (s : AGSequence) : AGSequence :=
  sorry

/-- Predicate to check if a sequence contains "AG" -/
def containsAG (s : AGSequence) : Prop :=
  sorry

/-- The main theorem stating that the process will eventually terminate -/
theorem ag_replacement_terminates (initial : AGSequence) :
  ∃ (n : ℕ) (final : AGSequence), (∀ k, k ≥ n → replaceAG^[k] initial = final) ∧ ¬containsAG final :=
  sorry

end NUMINAMATH_CALUDE_ag_replacement_terminates_l3389_338999


namespace NUMINAMATH_CALUDE_maria_carrots_l3389_338989

def carrot_problem (initial_carrots thrown_out_carrots picked_next_day : ℕ) : Prop :=
  initial_carrots - thrown_out_carrots + picked_next_day = 52

theorem maria_carrots : carrot_problem 48 11 15 := by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_l3389_338989


namespace NUMINAMATH_CALUDE_select_defective_theorem_l3389_338973

/-- The number of ways to select at least 2 defective products -/
def select_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℕ :=
  Nat.choose defective 2 * Nat.choose (total - defective) (selected - 2) +
  Nat.choose defective 3 * Nat.choose (total - defective) (selected - 3)

/-- Theorem stating the number of ways to select at least 2 defective products
    from 5 randomly selected products out of 200 total products with 3 defective products -/
theorem select_defective_theorem :
  select_defective 200 3 5 = Nat.choose 3 2 * Nat.choose 197 3 + Nat.choose 3 3 * Nat.choose 197 2 := by
  sorry

end NUMINAMATH_CALUDE_select_defective_theorem_l3389_338973


namespace NUMINAMATH_CALUDE_complex_sum_value_l3389_338987

theorem complex_sum_value : 
  ∀ (c d : ℂ), c = 3 + 2*I ∧ d = 1 - 2*I → 3*c + 4*d = 13 - 2*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_value_l3389_338987
