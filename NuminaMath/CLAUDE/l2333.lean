import Mathlib

namespace houses_with_neither_feature_l2333_233345

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) :
  total = 70 →
  garage = 50 →
  pool = 40 →
  both = 35 →
  total - (garage + pool - both) = 15 :=
by sorry

end houses_with_neither_feature_l2333_233345


namespace alicia_final_collection_l2333_233308

def egyptian_mask_collection (initial : ℕ) : ℕ :=
  let after_guggenheim := initial - 51
  let after_metropolitan := after_guggenheim - (after_guggenheim / 3)
  let after_louvre := after_metropolitan - (after_metropolitan / 4)
  let after_damage := after_louvre - 30
  let after_british := after_damage - (after_damage * 2 / 5)
  after_british - (after_british / 8)

theorem alicia_final_collection :
  egyptian_mask_collection 600 = 129 := by sorry

end alicia_final_collection_l2333_233308


namespace max_k_value_l2333_233334

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2/y^2 + y^2/x^2) + k * (x/y + y/x)) :
  k ≤ (-1 + Real.sqrt 17) / 2 :=
by sorry

end max_k_value_l2333_233334


namespace tangent_line_y_intercept_l2333_233369

/-- The y-intercept of the tangent line to f(x) = ax - ln x at x = 1 is 1 -/
theorem tangent_line_y_intercept (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x - Real.log x
  let f' : ℝ → ℝ := λ x ↦ a - 1 / x
  let tangent_slope : ℝ := f' 1
  let tangent_point : ℝ × ℝ := (1, f 1)
  let tangent_line : ℝ → ℝ := λ x ↦ tangent_slope * (x - tangent_point.1) + tangent_point.2
  tangent_line 0 = 1 := by
sorry

end tangent_line_y_intercept_l2333_233369


namespace joans_games_l2333_233364

theorem joans_games (football_this_year basketball_this_year total_both_years : ℕ)
  (h1 : football_this_year = 4)
  (h2 : basketball_this_year = 3)
  (h3 : total_both_years = 9) :
  total_both_years - (football_this_year + basketball_this_year) = 2 := by
sorry

end joans_games_l2333_233364


namespace ratio_characterization_l2333_233356

/-- Given points A, B, and M on a line, where M ≠ B, this theorem characterizes the position of M based on the ratio AM:BM -/
theorem ratio_characterization (A B M M1 M2 : ℝ) : 
  (M ≠ B) →
  (A < B) →
  (A < M1) → (M1 < B) →
  (A < M2) → (B < M2) →
  (A - M1 = 2 * (M1 - B)) →
  (M2 - A = 2 * (B - A)) →
  (((M - A) > 2 * (B - M) ↔ (M1 < M ∧ M < M2 ∧ M ≠ B)) ∧
   ((M - A) < 2 * (B - M) ↔ (M < M1 ∨ M2 < M))) :=
by sorry

end ratio_characterization_l2333_233356


namespace probability_three_heads_is_one_eighth_l2333_233348

/-- Represents the possible outcomes of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of five coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (halfDollar : CoinOutcome)

/-- The probability of the penny, dime, and quarter all coming up heads -/
def probabilityThreeHeads : ℚ := 1 / 8

/-- Theorem stating that the probability of the penny, dime, and quarter
    all coming up heads when flipping five coins is 1/8 -/
theorem probability_three_heads_is_one_eighth :
  probabilityThreeHeads = 1 / 8 := by
  sorry


end probability_three_heads_is_one_eighth_l2333_233348


namespace square_area_error_l2333_233347

theorem square_area_error (edge : ℝ) (edge_error : ℝ) (area_error : ℝ) : 
  edge_error = 0.02 → 
  area_error = (((1 + edge_error) * edge)^2 - edge^2) / edge^2 * 100 → 
  area_error = 4.04 := by
  sorry

end square_area_error_l2333_233347


namespace fraction_addition_l2333_233385

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l2333_233385


namespace trigonometric_simplification_logarithmic_simplification_l2333_233350

theorem trigonometric_simplification (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (θ + π/2) * Real.cos (π/2 - θ) - Real.cos (π - θ)^2) / (1 + Real.sin θ^2) = 1/3 := by
  sorry

theorem logarithmic_simplification (x : Real) :
  Real.log (Real.sqrt (x^2 + 1) + x) + Real.log (Real.sqrt (x^2 + 1) - x) +
  (Real.log 2 / Real.log 10)^2 + (1 + Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) -
  2 * Real.sin (30 * π / 180) = 0 := by
  sorry

end trigonometric_simplification_logarithmic_simplification_l2333_233350


namespace composite_s_l2333_233307

theorem composite_s (s : ℕ) (h1 : s ≥ 4) :
  (∃ a b c d : ℕ+, (a:ℕ) + b + c + d = s ∧ 
    (s ∣ a * b * c + a * b * d + a * c * d + b * c * d)) →
  ¬(Nat.Prime s) :=
by sorry

end composite_s_l2333_233307


namespace lineup_combinations_l2333_233399

/-- Represents the number of ways to choose a starting lineup in basketball -/
def choose_lineup (total_players : ℕ) (lineup_size : ℕ) 
  (point_guards : ℕ) (shooting_guards : ℕ) (small_forwards : ℕ) 
  (power_center : ℕ) : ℕ :=
  Nat.choose point_guards 1 * 
  Nat.choose shooting_guards 1 * 
  Nat.choose small_forwards 1 * 
  Nat.choose power_center 1 * 
  Nat.choose (power_center - 1) 1

/-- Theorem stating the number of ways to choose a starting lineup -/
theorem lineup_combinations : 
  choose_lineup 12 5 3 2 4 3 = 144 := by
  sorry

end lineup_combinations_l2333_233399


namespace exact_one_common_point_chord_length_when_m_4_l2333_233393

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (4 * t^2, 4 * t)

-- Define the line l in polar form
def line_l (m : ℝ) (ρ θ : ℝ) : Prop := ρ * (4 * Real.cos θ + 3 * Real.sin θ) - m = 0

-- Theorem 1: Value of m for exactly one common point
theorem exact_one_common_point :
  ∃ (m : ℝ), m = -9/4 ∧
  (∃! (t : ℝ), ∃ (ρ θ : ℝ), curve_C t = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ line_l m ρ θ) :=
sorry

-- Theorem 2: Length of chord when m = 4
theorem chord_length_when_m_4 :
  let m := 4
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  (∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ), 
    curve_C t₁ = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧ 
    curve_C t₂ = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) ∧
    line_l m ρ₁ θ₁ ∧ line_l m ρ₂ θ₂) ∧
  let (x₁, y₁) := curve_C t₁
  let (x₂, y₂) := curve_C t₂
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 25/4 :=
sorry

end exact_one_common_point_chord_length_when_m_4_l2333_233393


namespace dice_probability_l2333_233391

def red_die : Finset Nat := {4, 6}
def yellow_die : Finset Nat := {1, 2, 3, 4, 5, 6}

def total_outcomes : Finset (Nat × Nat) :=
  red_die.product yellow_die

def favorable_outcomes : Finset (Nat × Nat) :=
  total_outcomes.filter (fun p => p.1 * p.2 > 20)

theorem dice_probability :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 1 / 3 := by
  sorry

end dice_probability_l2333_233391


namespace certain_number_proof_l2333_233336

theorem certain_number_proof (x : ℤ) : x - 82 = 17 → x = 99 := by
  sorry

end certain_number_proof_l2333_233336


namespace perpendicular_line_through_point_l2333_233361

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (-2, -3),
    prove that the line L2 with equation y = -2x - 7 is perpendicular to L1
    and passes through P. -/
theorem perpendicular_line_through_point 
  (L1 : Real → Real → Prop) 
  (P : Real × Real) 
  (L2 : Real → Real → Prop) : 
  (∀ x y, L1 x y ↔ 3 * x - 6 * y = 9) →
  P = (-2, -3) →
  (∀ x y, L2 x y ↔ y = -2 * x - 7) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    (x₂ - x₁) * (P.1 - x₁) + (y₂ - y₁) * (P.2 - y₁) = 0) →
  L2 P.1 P.2 := by
  sorry

end perpendicular_line_through_point_l2333_233361


namespace notebook_cost_l2333_233366

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total_cost : notebook_cost + pencil_cost = 3.40)
  (price_difference : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.70 := by
sorry

end notebook_cost_l2333_233366


namespace smallest_sum_of_squares_l2333_233337

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 133 → 
  (∀ a b : ℕ, a^2 - b^2 = 133 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 205 :=
by
  sorry

end smallest_sum_of_squares_l2333_233337


namespace arctan_two_tan_75_minus_three_tan_15_l2333_233341

theorem arctan_two_tan_75_minus_three_tan_15 :
  Real.arctan (2 * Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180)) = 30 * π / 180 := by
  sorry

end arctan_two_tan_75_minus_three_tan_15_l2333_233341


namespace l_shapes_on_8x8_chessboard_l2333_233374

/-- Represents a chessboard --/
structure Chessboard where
  size : ℕ
  size_pos : size > 0

/-- Represents an L-shaped pattern on a chessboard --/
structure LShape where
  board : Chessboard

/-- Count of L-shaped patterns on a given chessboard --/
def count_l_shapes (board : Chessboard) : ℕ :=
  sorry

theorem l_shapes_on_8x8_chessboard :
  ∃ (board : Chessboard), board.size = 8 ∧ count_l_shapes board = 196 :=
sorry

end l_shapes_on_8x8_chessboard_l2333_233374


namespace paco_cookies_eaten_l2333_233326

/-- Given that Paco had 17 cookies, gave 13 to his friend, and ate 1 more than he gave away,
    prove that Paco ate 14 cookies. -/
theorem paco_cookies_eaten (initial : ℕ) (given : ℕ) (eaten : ℕ) 
    (h1 : initial = 17)
    (h2 : given = 13)
    (h3 : eaten = given + 1) : 
  eaten = 14 := by
  sorry

end paco_cookies_eaten_l2333_233326


namespace cubic_roots_sum_l2333_233349

theorem cubic_roots_sum (m : ℤ) (p q r : ℤ) : 
  (∀ x, x^3 - 2023*x + m = (x - p) * (x - q) * (x - r)) →
  |p| + |q| + |r| = 100 := by
sorry

end cubic_roots_sum_l2333_233349


namespace trigonometric_equation_solution_l2333_233314

theorem trigonometric_equation_solution (x : ℝ) :
  (4 * Real.sin (π / 6 + x) * Real.sin (5 * π / 6 + x) / (Real.cos x)^2 + 2 * Real.tan x = 0) ∧ (Real.cos x ≠ 0) →
  (∃ k : ℤ, x = -Real.arctan (1 / 3) + k * π) ∨ (∃ n : ℤ, x = π / 4 + n * π) :=
by sorry

end trigonometric_equation_solution_l2333_233314


namespace cubic_root_fraction_equality_l2333_233389

theorem cubic_root_fraction_equality (x : ℝ) (h : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := by
  sorry

end cubic_root_fraction_equality_l2333_233389


namespace rug_inner_length_is_three_l2333_233305

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the rug with its three regions -/
structure Rug where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_three (r : Rug) :
  r.inner.width = 2 →
  r.middle.length = r.inner.length + 3 →
  r.middle.width = r.inner.width + 3 →
  r.outer.length = r.middle.length + 3 →
  r.outer.width = r.middle.width + 3 →
  isArithmeticProgression (area r.inner) (area r.middle) (area r.outer) →
  r.inner.length = 3 := by
  sorry

end rug_inner_length_is_three_l2333_233305


namespace tip_calculation_correct_l2333_233311

/-- Calculates the tip for a restaurant check with given conditions. -/
def calculate_tip (check_amount : ℚ) (tax_rate : ℚ) (senior_discount : ℚ) (dine_in_surcharge : ℚ) (payment : ℚ) : ℚ :=
  let total_with_tax := check_amount * (1 + tax_rate)
  let discount_amount := check_amount * senior_discount
  let surcharge_amount := check_amount * dine_in_surcharge
  let final_total := total_with_tax - discount_amount + surcharge_amount
  payment - final_total

/-- Theorem stating that the tip calculation for the given conditions results in $2.75. -/
theorem tip_calculation_correct :
  calculate_tip 15 (20/100) (10/100) (5/100) 20 = 275/100 := by
  sorry

end tip_calculation_correct_l2333_233311


namespace divisibility_17_and_289_l2333_233359

theorem divisibility_17_and_289 (n : ℤ) :
  (∃ k : ℤ, n^2 - n - 4 = 17 * k) ↔ (∃ m : ℤ, n = 17 * m - 8) ∧
  ¬(∃ l : ℤ, n^2 - n - 4 = 289 * l) :=
by sorry

end divisibility_17_and_289_l2333_233359


namespace range_of_a_l2333_233340

/-- Given propositions p and q, and the condition that ¬p is a sufficient but not necessary condition for ¬q, prove that the range of real number a is [-1, 2]. -/
theorem range_of_a (a : ℝ) : 
  (∀ x, (x^2 - (2*a+4)*x + a^2 + 4*a < 0) ↔ (a < x ∧ x < a+4)) →
  (∀ x, ((x-2)*(x-3) < 0) ↔ (2 < x ∧ x < 3)) →
  (∀ x, ¬(a < x ∧ x < a+4) → ¬(2 < x ∧ x < 3)) →
  (∃ x, (2 < x ∧ x < 3) ∧ ¬(a < x ∧ x < a+4)) →
  -1 ≤ a ∧ a ≤ 2 :=
by sorry


end range_of_a_l2333_233340


namespace smaller_two_digit_number_l2333_233358

theorem smaller_two_digit_number 
  (x y : ℕ) 
  (h1 : x < y) 
  (h2 : x ≥ 10 ∧ x < 100) 
  (h3 : y ≥ 10 ∧ y < 100) 
  (h4 : x + y = 88) 
  (h5 : (100 * y + x) - (100 * x + y) = 3564) : 
  x = 26 := by
sorry

end smaller_two_digit_number_l2333_233358


namespace geometric_relations_l2333_233328

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Plane → Plane → Line)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem geometric_relations
  (l m : Line) (α β γ : Plane)
  (h1 : intersect β γ = l)
  (h2 : parallel l α)
  (h3 : contains α m)
  (h4 : perpendicular m γ) :
  perpendicularPlanes α γ ∧ perpendicularLines l m :=
by sorry

end geometric_relations_l2333_233328


namespace largest_fraction_of_consecutive_evens_l2333_233392

theorem largest_fraction_of_consecutive_evens (a b c d : ℕ) : 
  2 < a → a < b → b < c → c < d → 
  Even a → Even b → Even c → Even d →
  (b = a + 2) → (c = b + 2) → (d = c + 2) →
  (c + d) / (b + a) > (b + c) / (a + d) ∧
  (c + d) / (b + a) > (a + d) / (c + b) ∧
  (c + d) / (b + a) > (a + c) / (b + d) ∧
  (c + d) / (b + a) > (b + d) / (c + a) := by
  sorry

end largest_fraction_of_consecutive_evens_l2333_233392


namespace rectangular_prism_dimensions_l2333_233323

/-- Proves that a rectangular prism with given conditions has length 9 and width 3 -/
theorem rectangular_prism_dimensions :
  ∀ l w h : ℝ,
  l = 3 * w →
  h = 12 →
  Real.sqrt (l^2 + w^2 + h^2) = 15 →
  l = 9 ∧ w = 3 := by
  sorry

end rectangular_prism_dimensions_l2333_233323


namespace expression_evaluation_l2333_233352

theorem expression_evaluation :
  let x : ℝ := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := by
  sorry

end expression_evaluation_l2333_233352


namespace line_satisfies_conditions_l2333_233363

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 2*y + 4 = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 3*x - 4*y + 7 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := 10*x + 13*y - 26 = 0

-- Theorem statement
theorem line_satisfies_conditions :
  -- The result line passes through the intersection of line1 and line2
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ result_line x y) ∧
  -- The result line passes through the point (3, -2)
  (result_line 3 (-2)) ∧
  -- The result line is perpendicular to line3
  (∃ m1 m2 : ℝ, 
    (∀ x y : ℝ, line3 x y → y = m1 * x + (7 / 4)) ∧
    (∀ x y : ℝ, result_line x y → y = m2 * x + (26 / 10)) ∧
    m1 * m2 = -1) :=
by sorry

end line_satisfies_conditions_l2333_233363


namespace max_value_2x_minus_y_l2333_233342

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ z, 2*x - y ≤ z → z ≤ max :=
by sorry

end max_value_2x_minus_y_l2333_233342


namespace quadratic_equation_roots_l2333_233338

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k^2*x₁^2 - (2*k+1)*x₁ + 1 = 0 ∧ k^2*x₂^2 - (2*k+1)*x₂ + 1 = 0) ↔ 
  (k ≥ -1/4 ∧ k ≠ 0) :=
by sorry

end quadratic_equation_roots_l2333_233338


namespace fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8_l2333_233379

theorem fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8 :
  (16 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 :=
by sorry

end fourth_root_16_times_cube_root_8_times_sqrt_4_eq_8_l2333_233379


namespace largest_four_digit_divisible_by_50_l2333_233310

theorem largest_four_digit_divisible_by_50 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 50 = 0 → n ≤ 9950 :=
by sorry

end largest_four_digit_divisible_by_50_l2333_233310


namespace total_profit_is_36000_l2333_233320

/-- Represents the profit sharing problem of Tom and Jose's shop -/
def ProfitSharing (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : Prop :=
  let tom_total_investment := tom_investment * tom_months
  let jose_total_investment := jose_investment * jose_months
  let total_investment := tom_total_investment + jose_total_investment
  let profit_ratio := tom_total_investment / jose_total_investment
  let tom_profit := (profit_ratio * jose_profit) / (profit_ratio + 1)
  let total_profit := tom_profit + jose_profit
  total_profit = 36000

/-- The main theorem stating that given the investments and Jose's profit, the total profit is 36000 -/
theorem total_profit_is_36000 :
  ProfitSharing 30000 12 45000 10 20000 := by
  sorry

end total_profit_is_36000_l2333_233320


namespace fraction_not_on_time_is_one_eighth_l2333_233321

/-- Represents the fraction of attendees who did not arrive on time at a monthly meeting -/
def fraction_not_on_time (total : ℕ) (male : ℕ) (male_on_time : ℕ) (female_on_time : ℕ) : ℚ :=
  1 - (male_on_time + female_on_time : ℚ) / total

/-- Theorem stating the fraction of attendees who did not arrive on time -/
theorem fraction_not_on_time_is_one_eighth
  (total : ℕ) (male : ℕ) (male_on_time : ℕ) (female_on_time : ℕ)
  (h_total_pos : 0 < total)
  (h_male_ratio : male = (3 * total) / 5)
  (h_male_on_time : male_on_time = (7 * male) / 8)
  (h_female_on_time : female_on_time = (9 * (total - male)) / 10) :
  fraction_not_on_time total male male_on_time female_on_time = 1/8 := by
  sorry

#check fraction_not_on_time_is_one_eighth

end fraction_not_on_time_is_one_eighth_l2333_233321


namespace only_valid_solutions_l2333_233333

/-- A structure representing a solution to the equation AB = B^V --/
structure Solution :=
  (a : Nat) (b : Nat) (v : Nat)
  (h1 : a ≠ b) -- Different letters correspond to different digits
  (h2 : a * 10 + b ≥ 10 ∧ a * 10 + b < 100) -- AB is a two-digit number
  (h3 : a * 10 + b = b ^ v) -- AB = B^V

/-- The set of all valid solutions --/
def validSolutions : Set Solution :=
  { s : Solution | s.a = 3 ∧ s.b = 2 ∧ s.v = 5 ∨
                   s.a = 3 ∧ s.b = 6 ∧ s.v = 2 ∨
                   s.a = 6 ∧ s.b = 4 ∧ s.v = 3 }

/-- Theorem stating that the only solutions are 32 = 2^5, 36 = 6^2, and 64 = 4^3 --/
theorem only_valid_solutions (s : Solution) : s ∈ validSolutions := by
  sorry

end only_valid_solutions_l2333_233333


namespace unique_prime_twice_square_l2333_233387

theorem unique_prime_twice_square : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x y : ℕ), p + 1 = 2 * x^2 ∧ p^2 + 1 = 2 * y^2) ∧ 
    p = 7 := by
  sorry

end unique_prime_twice_square_l2333_233387


namespace spider_has_eight_legs_l2333_233303

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The number of legs a spider has -/
def spider_legs : ℕ := 2 * (2 * human_legs)

/-- Theorem stating that a spider has 8 legs -/
theorem spider_has_eight_legs : spider_legs = 8 := by
  sorry

end spider_has_eight_legs_l2333_233303


namespace matrix_sum_theorem_l2333_233309

def matrix_not_invertible (a b c : ℝ) : Prop :=
  ∀ k : ℝ, Matrix.det
    !![a + k, b + k, c + k;
       b + k, c + k, a + k;
       c + k, a + k, b + k] = 0

theorem matrix_sum_theorem (a b c : ℝ) :
  matrix_not_invertible a b c →
  (a / (b + c) + b / (a + c) + c / (a + b) = -3 ∨
   a / (b + c) + b / (a + c) + c / (a + b) = 3/2) :=
by sorry

end matrix_sum_theorem_l2333_233309


namespace school_time_problem_l2333_233377

/-- Given a boy who reaches school 6 minutes early when walking at 7/6 of his usual rate,
    his usual time to reach the school is 42 minutes. -/
theorem school_time_problem (usual_time : ℝ) (usual_rate : ℝ) : 
  (usual_rate / usual_time = (7/6 * usual_rate) / (usual_time - 6)) → 
  usual_time = 42 := by
  sorry

end school_time_problem_l2333_233377


namespace carrot_usage_l2333_233378

theorem carrot_usage (total_carrots : ℕ) (unused_carrots : ℕ) 
  (h1 : total_carrots = 300)
  (h2 : unused_carrots = 72) : 
  ∃ (x : ℚ), 
    x * total_carrots + (3/5 : ℚ) * (total_carrots - x * total_carrots) = total_carrots - unused_carrots ∧ 
    x = 2/5 := by
  sorry

end carrot_usage_l2333_233378


namespace transform_negative_expression_l2333_233398

theorem transform_negative_expression (a b c : ℝ) :
  -(a - b + c) = -a + b - c := by sorry

end transform_negative_expression_l2333_233398


namespace ada_was_in_seat_two_l2333_233330

/-- Represents the seats in the row --/
inductive Seat
  | one
  | two
  | three
  | four
  | five

/-- Represents the friends --/
inductive Friend
  | ada
  | bea
  | ceci
  | dee
  | edie

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- The initial seating arrangement --/
def initial_arrangement : Arrangement := sorry

/-- The final seating arrangement after all movements --/
def final_arrangement : Arrangement := sorry

/-- Bea moves one seat to the right --/
def bea_moves (arr : Arrangement) : Arrangement := sorry

/-- Ceci moves left and then back --/
def ceci_moves (arr : Arrangement) : Arrangement := sorry

/-- Dee and Edie switch seats, then Edie moves right --/
def dee_edie_move (arr : Arrangement) : Arrangement := sorry

/-- Ada's original seat --/
def ada_original_seat : Seat := sorry

theorem ada_was_in_seat_two :
  ada_original_seat = Seat.two ∧
  final_arrangement = dee_edie_move (ceci_moves (bea_moves initial_arrangement)) ∧
  (final_arrangement Friend.ada = Seat.one ∨ final_arrangement Friend.ada = Seat.five) :=
sorry

end ada_was_in_seat_two_l2333_233330


namespace average_income_P_Q_l2333_233316

/-- Given the monthly incomes of three people P, Q, and R, prove that the average monthly income of P and Q is 5050, given certain conditions. -/
theorem average_income_P_Q (P Q R : ℕ) : 
  (Q + R) / 2 = 6250 →  -- Average income of Q and R
  (P + R) / 2 = 5200 →  -- Average income of P and R
  P = 4000 →            -- Income of P
  (P + Q) / 2 = 5050 :=  -- Average income of P and Q
by sorry

end average_income_P_Q_l2333_233316


namespace max_take_home_pay_l2333_233365

/-- The take-home pay function for income y (in thousand dollars) -/
def P (y : ℝ) : ℝ := -10 * (y - 5)^2 + 1000

/-- The income that yields the greatest take-home pay -/
def max_income : ℝ := 5

theorem max_take_home_pay :
  ∀ y : ℝ, y ≥ 0 → P y ≤ P max_income :=
sorry

end max_take_home_pay_l2333_233365


namespace robins_gum_packages_robins_gum_packages_solution_l2333_233346

theorem robins_gum_packages (candy_packages : ℕ) (pieces_per_package : ℕ) (additional_pieces : ℕ) : ℕ :=
  let total_pieces := candy_packages * pieces_per_package + additional_pieces
  total_pieces / pieces_per_package

theorem robins_gum_packages_solution :
  robins_gum_packages 14 6 7 = 15 := by sorry

end robins_gum_packages_robins_gum_packages_solution_l2333_233346


namespace discount_calculation_l2333_233331

/-- Given a cost price, prove that if the marked price is 150% of the cost price
    and the selling price results in a 1% loss on the cost price, then the discount
    (difference between marked price and selling price) is 51% of the cost price. -/
theorem discount_calculation (CP : ℝ) (CP_pos : CP > 0) : 
  let MP := 1.5 * CP
  let SP := 0.99 * CP
  MP - SP = 0.51 * CP := by sorry

end discount_calculation_l2333_233331


namespace problem_statement_problem_statement_2_l2333_233368

/-- The quadratic function used in the problem -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

/-- Proposition p: x^2 - 2x - 8 ≤ 0 -/
def p (x : ℝ) : Prop := f x ≤ 0

/-- Proposition q: 2 - m ≤ x ≤ 2 + m -/
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

theorem problem_statement (m : ℝ) (h : m > 0) :
  (∀ x, p x → q m x) ∧ (∃ x, q m x ∧ ¬p x) → m ≥ 4 :=
sorry

theorem problem_statement_2 (x : ℝ) :
  let m := 5
  (p x ∨ q m x) ∧ ¬(p x ∧ q m x) →
  (-3 ≤ x ∧ x < -2) ∨ (4 < x ∧ x ≤ 7) :=
sorry

end problem_statement_problem_statement_2_l2333_233368


namespace suzy_jump_ropes_l2333_233300

theorem suzy_jump_ropes (yesterday : ℕ) (additional : ℕ) : 
  yesterday = 247 → additional = 131 → yesterday + (yesterday + additional) = 625 := by
  sorry

end suzy_jump_ropes_l2333_233300


namespace pie_eating_contest_l2333_233397

theorem pie_eating_contest (erik_pie frank_pie : ℝ) 
  (h_erik : erik_pie = 0.67)
  (h_frank : frank_pie = 0.33) :
  erik_pie - frank_pie = 0.34 := by
  sorry

end pie_eating_contest_l2333_233397


namespace epsilon_delta_condition_l2333_233362

def f (x : ℝ) := x^2 + 1

theorem epsilon_delta_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 2 := by
  sorry

end epsilon_delta_condition_l2333_233362


namespace ribbon_division_l2333_233355

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) : 
  total_ribbon = 5 / 12 → num_boxes = 5 → total_ribbon / num_boxes = 1 / 12 := by
  sorry

end ribbon_division_l2333_233355


namespace loggers_required_is_eight_l2333_233370

/-- Represents the number of loggers required to cut down all trees in a forest under specific conditions. -/
def number_of_loggers (forest_length : ℕ) (forest_width : ℕ) (trees_per_square_mile : ℕ) 
  (trees_per_day : ℕ) (days_per_month : ℕ) (months_to_complete : ℕ) : ℕ :=
  (forest_length * forest_width * trees_per_square_mile) / 
  (trees_per_day * days_per_month * months_to_complete)

/-- Theorem stating that the number of loggers required under the given conditions is 8. -/
theorem loggers_required_is_eight :
  number_of_loggers 4 6 600 6 30 10 = 8 := by
  sorry

end loggers_required_is_eight_l2333_233370


namespace probability_four_white_balls_l2333_233344

/-- The probability of drawing 4 white balls from a box containing 7 white balls and 5 black balls -/
theorem probability_four_white_balls (white_balls black_balls drawn : ℕ) : 
  white_balls = 7 →
  black_balls = 5 →
  drawn = 4 →
  (Nat.choose white_balls drawn : ℚ) / (Nat.choose (white_balls + black_balls) drawn) = 7 / 99 := by
  sorry

end probability_four_white_balls_l2333_233344


namespace parabola_point_ordinate_l2333_233357

theorem parabola_point_ordinate (x y : ℝ) : 
  y^2 = 8*x →                  -- Point M(x, y) is on the parabola y^2 = 8x
  (x - 2)^2 + y^2 = 4^2 →      -- Distance from M to focus (2, 0) is 4
  y = 4 ∨ y = -4 :=            -- The ordinate of M is either 4 or -4
by sorry

end parabola_point_ordinate_l2333_233357


namespace club_members_count_l2333_233394

/-- The number of female members in the club -/
def female_members : ℕ := 12

/-- The number of male members in the club -/
def male_members : ℕ := female_members / 2

/-- The total number of members in the club -/
def total_members : ℕ := female_members + male_members

/-- Proof that the total number of members in the club is 18 -/
theorem club_members_count : total_members = 18 := by
  sorry

end club_members_count_l2333_233394


namespace total_fish_l2333_233375

def micah_fish : ℕ := 7

def kenneth_fish (m : ℕ) : ℕ := 3 * m

def matthias_fish (k : ℕ) : ℕ := k - 15

theorem total_fish :
  micah_fish + kenneth_fish micah_fish + matthias_fish (kenneth_fish micah_fish) = 34 := by
  sorry

end total_fish_l2333_233375


namespace bridge_length_l2333_233335

/-- Given a train crossing a bridge, this theorem calculates the length of the bridge. -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 170)
  (h2 : train_speed = 45 * 1000 / 3600)  -- Convert km/hr to m/s
  (h3 : crossing_time = 30) :
  train_speed * crossing_time - train_length = 205 :=
by sorry

end bridge_length_l2333_233335


namespace other_number_proof_l2333_233315

theorem other_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 14) → 
  (Nat.lcm a b = 396) → 
  (a = 36) → 
  (b = 154) := by
sorry

end other_number_proof_l2333_233315


namespace remuneration_problem_l2333_233382

/-- Represents the remuneration problem -/
theorem remuneration_problem (annual_clothing : ℕ) (annual_coins : ℕ) 
  (months_worked : ℕ) (received_clothing : ℕ) (received_coins : ℕ) :
  annual_clothing = 1 →
  annual_coins = 10 →
  months_worked = 7 →
  received_clothing = 1 →
  received_coins = 2 →
  ∃ (clothing_value : ℚ),
    clothing_value = 46 / 5 ∧
    (clothing_value + annual_coins : ℚ) / 12 = (clothing_value + received_coins) / months_worked :=
by sorry

end remuneration_problem_l2333_233382


namespace smallest_four_digit_divisible_by_35_l2333_233383

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1006 :=
by sorry

end smallest_four_digit_divisible_by_35_l2333_233383


namespace vector_magnitude_problem_l2333_233353

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is perpendicular to c, 
    then the magnitude of b is 3√5. -/
theorem vector_magnitude_problem (a b c : ℝ × ℝ) : 
  a = (-2, 1) → 
  b.1 = k ∧ b.2 = -3 → 
  c = (1, 2) → 
  (a.1 - 2 * b.1, a.2 - 2 * b.2) • c = 0 → 
  Real.sqrt (b.1^2 + b.2^2) = 3 * Real.sqrt 5 := by
  sorry

end vector_magnitude_problem_l2333_233353


namespace simple_interest_rate_calculation_l2333_233395

/-- Calculate the simple interest rate given the principal and annual interest -/
theorem simple_interest_rate_calculation
  (principal : ℝ) 
  (annual_interest : ℝ) 
  (h1 : principal = 9000)
  (h2 : annual_interest = 810) :
  (annual_interest / principal) * 100 = 9 := by
sorry

end simple_interest_rate_calculation_l2333_233395


namespace library_books_count_l2333_233329

/-- Given a library with identical bookcases, prove the total number of books -/
theorem library_books_count (num_bookcases : ℕ) (shelves_per_bookcase : ℕ) (books_per_shelf : ℕ) :
  num_bookcases = 28 →
  shelves_per_bookcase = 6 →
  books_per_shelf = 19 →
  num_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end library_books_count_l2333_233329


namespace puzzle_sum_l2333_233339

theorem puzzle_sum (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 48) 
  (h3 : c + a = 59) 
  (h4 : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  a + b + c = 69 := by
  sorry

end puzzle_sum_l2333_233339


namespace flagpole_break_height_l2333_233322

theorem flagpole_break_height (h : ℝ) (b : ℝ) (x : ℝ) 
  (hypotenuse : h = 6)
  (base : b = 2)
  (right_triangle : x^2 + b^2 = h^2) :
  x = Real.sqrt 10 := by
sorry

end flagpole_break_height_l2333_233322


namespace unique_odd_number_with_congruences_l2333_233325

theorem unique_odd_number_with_congruences : ∃! x : ℕ,
  500 < x ∧ x < 1000 ∧
  x % 25 = 6 ∧
  x % 9 = 7 ∧
  Odd x ∧
  x = 781 := by
  sorry

end unique_odd_number_with_congruences_l2333_233325


namespace intersection_point_is_unique_l2333_233396

/-- Represents a 2D point --/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form --/
structure ParametricLine where
  origin : Point
  direction : Point

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 3 },
    direction := { x := 3, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := 6, y := 1 },
    direction := { x := 5, y := -1 } }

/-- The proposed intersection point --/
def intersectionPoint : Point :=
  { x := 20/23, y := 27/23 }

/-- Function to get a point on a parametric line given a parameter --/
def pointOnLine (line : ParametricLine) (t : ℚ) : Point :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- Theorem stating that the given point is the unique intersection of the two lines --/
theorem intersection_point_is_unique :
  ∃! t u, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry

end intersection_point_is_unique_l2333_233396


namespace bisecting_line_exists_unique_l2333_233304

/-- A triangle with sides of length 6, 8, and 10 units. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10

/-- A line that intersects two sides of the triangle. -/
structure BisectingLine (T : Triangle) where
  x : ℝ  -- Intersection point on side b
  y : ℝ  -- Intersection point on side c
  hx : 0 < x ∧ x < T.b
  hy : 0 < y ∧ y < T.c

/-- The bisecting line divides the perimeter in half. -/
def bisects_perimeter (T : Triangle) (L : BisectingLine T) : Prop :=
  L.x + L.y = (T.a + T.b + T.c) / 2

/-- The bisecting line divides the area in half. -/
def bisects_area (T : Triangle) (L : BisectingLine T) : Prop :=
  L.x * L.y = (T.a * T.b) / 4

/-- The main theorem: existence and uniqueness of the bisecting line. -/
theorem bisecting_line_exists_unique (T : Triangle) :
  ∃! L : BisectingLine T, bisects_perimeter T L ∧ bisects_area T L :=
sorry

end bisecting_line_exists_unique_l2333_233304


namespace tangent_line_at_2_f_greater_than_2x_minus_ln_l2333_233327

noncomputable section

def f (x : ℝ) : ℝ := Real.exp x / x

/-- The equation of the tangent line to y = f(x) at x = 2 is e^2x - 4y = 0 -/
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * (x - 2) + f 2 ↔ Real.exp 2 * x - 4 * y = 0 :=
sorry

/-- For all x > 0, f(x) > 2(x - ln x) -/
theorem f_greater_than_2x_minus_ln :
  ∀ x > 0, f x > 2 * (x - Real.log x) :=
sorry

end

end tangent_line_at_2_f_greater_than_2x_minus_ln_l2333_233327


namespace chord_length_squared_l2333_233332

/-- The square of the length of a chord that is a common external tangent to two circles -/
theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : R > 0)
  (h₄ : r₁ + r₂ < R) : 
  let d := R - (r₁ + r₂) + Real.sqrt (r₁ * r₂)
  4 * (R^2 - d^2) = 516 :=
by sorry

end chord_length_squared_l2333_233332


namespace max_value_implies_a_l2333_233354

/-- The function f(x) = -x^2 + 2ax + 1 - a has a maximum value of 2 in the interval [0, 1] -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

/-- The maximum value of f(x) in the interval [0, 1] is 2 -/
def max_value (a : ℝ) : Prop := ∀ x, x ∈ Set.Icc 0 1 → f a x ≤ 2

/-- The theorem stating that if f(x) has a maximum value of 2 in [0, 1], then a = -1 or a = 2 -/
theorem max_value_implies_a (a : ℝ) : max_value a → (a = -1 ∨ a = 2) := by sorry

end max_value_implies_a_l2333_233354


namespace min_socks_for_pair_l2333_233318

theorem min_socks_for_pair (n : ℕ) (h : n = 2019) : ∃ m : ℕ, m = n + 1 ∧ 
  (∀ k : ℕ, k < m → ∃ f : Fin k → Fin n, Function.Injective f) ∧
  (∀ g : Fin m → Fin n, ¬Function.Injective g) :=
by
  sorry

end min_socks_for_pair_l2333_233318


namespace ellipse_to_circle_transformation_l2333_233306

/-- Proves that the given scaling transformation transforms the ellipse into the circle -/
theorem ellipse_to_circle_transformation (x y x' y' : ℝ) :
  (x'^2 / 10 + y'^2 / 8 = 1) →
  (x' = (Real.sqrt 10 / 5) * x ∧ y' = (Real.sqrt 2 / 2) * y) →
  (x^2 + y^2 = 4) :=
by sorry

end ellipse_to_circle_transformation_l2333_233306


namespace checkerboard_sum_l2333_233381

/-- The number of rectangles in a 7x7 checkerboard -/
def r' : ℕ := 784

/-- The number of squares in a 7x7 checkerboard -/
def s' : ℕ := 140

/-- m' and n' are relatively prime positive integers such that s'/r' = m'/n' -/
def m' : ℕ := 5
def n' : ℕ := 28

theorem checkerboard_sum : m' + n' = 33 := by sorry

end checkerboard_sum_l2333_233381


namespace correct_rounding_sum_l2333_233373

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem correct_rounding_sum : round_to_nearest_hundred (125 + 96) = 200 := by
  sorry

end correct_rounding_sum_l2333_233373


namespace tatiana_age_l2333_233367

/-- Calculates the total full years given an age in years, months, weeks, days, and hours -/
def calculate_full_years (years months weeks days hours : ℕ) : ℕ :=
  let months_to_years := months / 12
  let weeks_to_years := weeks / 52
  let days_to_years := days / 365
  let hours_to_years := hours / (24 * 365)
  years + months_to_years + weeks_to_years + days_to_years + hours_to_years

/-- Theorem stating that the age of 72 years, 72 months, 72 weeks, 72 days, and 72 hours is equivalent to 79 full years -/
theorem tatiana_age : calculate_full_years 72 72 72 72 72 = 79 := by
  sorry

end tatiana_age_l2333_233367


namespace sqrt_difference_equality_l2333_233301

theorem sqrt_difference_equality : Real.sqrt (64 + 36) - Real.sqrt (81 - 64) = 10 - Real.sqrt 17 := by
  sorry

end sqrt_difference_equality_l2333_233301


namespace range_of_m_l2333_233343

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - |x| else x^2 - 4*x + 3

theorem range_of_m (m : ℝ) : f (f m) ≥ 0 → m ∈ Set.Icc (-2) (2 + Real.sqrt 2) ∪ Set.Ici 4 := by
  sorry

end range_of_m_l2333_233343


namespace bug_traversal_12_25_l2333_233390

/-- The number of tiles a bug traverses when walking diagonally across a rectangular floor -/
def bugTraversal (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

theorem bug_traversal_12_25 :
  bugTraversal 12 25 = 36 := by
  sorry

end bug_traversal_12_25_l2333_233390


namespace sum_of_ten_consecutive_squares_not_perfect_square_l2333_233312

theorem sum_of_ten_consecutive_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ ∃ m : ℕ, 10 * n^2 + 10 * n + 85 = m^2 := by
  sorry

end sum_of_ten_consecutive_squares_not_perfect_square_l2333_233312


namespace C_younger_than_A_l2333_233313

-- Define variables for ages
variable (A B C : ℕ)

-- Define the condition from the problem
def age_condition (A B C : ℕ) : Prop := A + B = B + C + 12

-- Theorem to prove
theorem C_younger_than_A (h : age_condition A B C) : A = C + 12 := by
  sorry

end C_younger_than_A_l2333_233313


namespace suresh_job_time_l2333_233372

/-- The time it takes Ashutosh to complete the job alone (in hours) -/
def ashutosh_time : ℝ := 15

/-- The time Suresh works on the job (in hours) -/
def suresh_work_time : ℝ := 9

/-- The time Ashutosh works to complete the remaining job (in hours) -/
def ashutosh_remaining_time : ℝ := 6

/-- The time it takes Suresh to complete the job alone (in hours) -/
def suresh_time : ℝ := 15

theorem suresh_job_time :
  suresh_time * (1 / ashutosh_time * ashutosh_remaining_time + 1 / suresh_time * suresh_work_time) = suresh_time := by
  sorry

#check suresh_job_time

end suresh_job_time_l2333_233372


namespace product_of_decimals_l2333_233317

theorem product_of_decimals : (0.7 : ℝ) * 0.8 = 0.56 := by
  sorry

end product_of_decimals_l2333_233317


namespace collinear_vectors_k_l2333_233360

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 1]
def c : Fin 2 → ℝ := ![3, 2]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 2, v i = t * w i

theorem collinear_vectors_k (k : ℝ) :
  collinear c (fun i ↦ k * a i + b i) → k = -1 := by
  sorry

#check collinear_vectors_k

end collinear_vectors_k_l2333_233360


namespace inscribed_circle_area_l2333_233351

/-- The area of the circle inscribed in a right triangle with perimeter 2p and hypotenuse c is π(p - c)². -/
theorem inscribed_circle_area (p c : ℝ) (h1 : 0 < p) (h2 : 0 < c) (h3 : c < 2 * p) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y + c = 2 * p ∧
  ∃ (r : ℝ), r > 0 ∧ r = p - c ∧
  ∃ (S : ℝ), S = π * r^2 ∧ S = π * (p - c)^2 :=
by
  sorry


end inscribed_circle_area_l2333_233351


namespace numbers_not_sum_of_two_elements_l2333_233384

def A : Finset ℕ := {1, 2, 3, 5, 8, 13, 21, 34, 55}

def range_start : ℕ := 3
def range_end : ℕ := 89

def sums_of_two_elements (S : Finset ℕ) : Finset ℕ :=
  (S.product S).image (λ (x : ℕ × ℕ) => x.1 + x.2)

def numbers_in_range : Finset ℕ :=
  Finset.Icc range_start range_end

theorem numbers_not_sum_of_two_elements : 
  (numbers_in_range.card - (numbers_in_range ∩ sums_of_two_elements A).card) = 51 := by
  sorry

end numbers_not_sum_of_two_elements_l2333_233384


namespace steiner_inellipse_center_distance_l2333_233376

/-- Triangle with vertices (0, 0), (3, 0), and (0, 3/2) -/
def T : Set (ℝ × ℝ) := {(0, 0), (3, 0), (0, 3/2)}

/-- The Steiner inellipse of triangle T -/
def E : Set (ℝ × ℝ) := sorry

/-- The center of the Steiner inellipse E -/
def center_E : ℝ × ℝ := sorry

/-- The distance from the center of E to (0, 0) -/
def distance_to_origin : ℝ := sorry

theorem steiner_inellipse_center_distance :
  distance_to_origin = Real.sqrt 5 / 2 := by sorry

end steiner_inellipse_center_distance_l2333_233376


namespace inequality_proof_l2333_233371

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x := by
sorry

end inequality_proof_l2333_233371


namespace bakery_storage_l2333_233386

theorem bakery_storage (sugar flour baking_soda : ℕ) : 
  sugar = flour ∧ 
  flour = 10 * baking_soda ∧ 
  flour = 8 * (baking_soda + 60) → 
  sugar = 2400 := by
sorry

end bakery_storage_l2333_233386


namespace money_division_l2333_233319

theorem money_division (a b c : ℕ) (h1 : a = b / 2) (h2 : b = c / 2) (h3 : c = 400) :
  a + b + c = 700 := by
  sorry

end money_division_l2333_233319


namespace quadratic_two_roots_l2333_233388

theorem quadratic_two_roots (a b c : ℝ) (h1 : 2016 + a^2 + a*c < a*b) (h2 : a ≠ 0) :
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 := by
  sorry

end quadratic_two_roots_l2333_233388


namespace contradiction_proof_l2333_233302

theorem contradiction_proof (a b c : ℝ) 
  (h1 : a + b + c > 0) 
  (h2 : a * b + b * c + a * c > 0) 
  (h3 : a * b * c > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := by
sorry

end contradiction_proof_l2333_233302


namespace product_prs_l2333_233380

theorem product_prs (p r s : ℕ) : 
  4^p + 4^3 = 320 → 
  3^r + 27 = 108 → 
  2^s + 7^4 = 2617 → 
  p * r * s = 112 := by
sorry

end product_prs_l2333_233380


namespace european_scientist_ratio_l2333_233324

theorem european_scientist_ratio (total : ℕ) (usa : ℕ) (canada : ℚ) : 
  total = 70 →
  usa = 21 →
  canada = 1/5 →
  (total - (canada * total).num - usa) / total = 1/2 := by
sorry

end european_scientist_ratio_l2333_233324
