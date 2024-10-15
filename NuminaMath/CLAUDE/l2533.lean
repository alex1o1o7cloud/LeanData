import Mathlib

namespace NUMINAMATH_CALUDE_intersection_when_m_is_2_sufficient_not_necessary_condition_l2533_253315

def p (x : ℝ) := x^2 + 2*x - 8 < 0

def q (x m : ℝ) := (x - 1 + m)*(x - 1 - m) ≤ 0

def A := {x : ℝ | p x}

def B (m : ℝ) := {x : ℝ | q x m}

theorem intersection_when_m_is_2 :
  B 2 ∩ A = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_2_sufficient_not_necessary_condition_l2533_253315


namespace NUMINAMATH_CALUDE_function_and_angle_theorem_l2533_253309

/-- Given a function f and an angle α, proves that f(x) = cos x and 
    (√2 f(2α - π/4) - 1) / (1 - tan α) = 2/5 under certain conditions -/
theorem function_and_angle_theorem (f : ℝ → ℝ) (ω φ α : ℝ) : 
  ω > 0 → 
  0 ≤ φ ∧ φ ≤ π → 
  (∀ x, f x = Real.sin (ω * x + φ)) →
  (∀ x, f x = f (-x)) →
  (∃ x₁ x₂, abs (x₁ - x₂) = Real.sqrt (4 + Real.pi^2) ∧ 
    f x₁ = 1 ∧ f x₂ = -1) →
  Real.tan α + 1 / Real.tan α = 5 →
  (∀ x, f x = Real.cos x) ∧ 
  (Real.sqrt 2 * f (2 * α - Real.pi / 4) - 1) / (1 - Real.tan α) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_function_and_angle_theorem_l2533_253309


namespace NUMINAMATH_CALUDE_a_5_equals_20_l2533_253306

def S (n : ℕ) : ℕ := 2 * n * (n + 1)

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_5_equals_20 : a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_20_l2533_253306


namespace NUMINAMATH_CALUDE_luke_coin_count_l2533_253379

/-- 
Given:
- Luke has 5 piles of quarters and 5 piles of dimes
- Each pile contains 3 coins
Prove that the total number of coins is 30
-/
theorem luke_coin_count (piles_quarters piles_dimes coins_per_pile : ℕ) 
  (h1 : piles_quarters = 5)
  (h2 : piles_dimes = 5)
  (h3 : coins_per_pile = 3) : 
  piles_quarters * coins_per_pile + piles_dimes * coins_per_pile = 30 := by
  sorry

#eval 5 * 3 + 5 * 3  -- Should output 30

end NUMINAMATH_CALUDE_luke_coin_count_l2533_253379


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_ratio_l2533_253333

/-- Given an ellipse and a hyperbola with common foci, prove that the ratio of their minor axes is √3 -/
theorem ellipse_hyperbola_ratio (a₁ b₁ a₂ b₂ c : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a₁ > b₁ ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 →
  P.1^2 / a₁^2 + P.2^2 / b₁^2 = 1 →
  P.1^2 / a₂^2 - P.2^2 / b₂^2 = 1 →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4 * c^2 →
  a₁^2 - b₁^2 = c^2 →
  a₂^2 + b₂^2 = c^2 →
  Real.cos (Real.arccos ((((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt + ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt)^2 - 4*c^2) /
    (2 * ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt)) = 1/2 →
  b₁ / b₂ = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_ratio_l2533_253333


namespace NUMINAMATH_CALUDE_scenic_area_ticket_sales_l2533_253385

/-- Scenic area ticket sales problem -/
theorem scenic_area_ticket_sales 
  (parent_child_price : ℝ) 
  (family_price : ℝ) 
  (parent_child_presale : ℝ) 
  (family_presale : ℝ) 
  (volume_difference : ℕ) 
  (parent_child_planned : ℕ) 
  (family_planned : ℕ) 
  (a : ℝ) :
  family_price = 2 * parent_child_price →
  parent_child_presale = 21000 →
  family_presale = 10500 →
  (parent_child_presale / parent_child_price) - (family_presale / family_price) = volume_difference →
  parent_child_planned = 1600 →
  family_planned = 400 →
  (parent_child_price + 3/4 * a) * (parent_child_planned - 32 * a) + 
    (family_price + a) * family_planned = 
    parent_child_price * parent_child_planned + family_price * family_planned →
  parent_child_price = 35 ∧ a = 20 := by
  sorry

end NUMINAMATH_CALUDE_scenic_area_ticket_sales_l2533_253385


namespace NUMINAMATH_CALUDE_rosa_flower_count_l2533_253349

/-- The number of flowers Rosa has after receiving flowers from Andre -/
def total_flowers (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Rosa's total flowers is the sum of her initial flowers and received flowers -/
theorem rosa_flower_count (initial : ℕ) (received : ℕ) :
  total_flowers initial received = initial + received :=
by sorry

end NUMINAMATH_CALUDE_rosa_flower_count_l2533_253349


namespace NUMINAMATH_CALUDE_maddie_weekend_watch_l2533_253334

def total_episodes : ℕ := 8
def episode_length : ℕ := 44
def monday_watch : ℕ := 138
def thursday_watch : ℕ := 21
def friday_episodes : ℕ := 2

def weekend_watch : ℕ := 105

theorem maddie_weekend_watch :
  let total_watch := total_episodes * episode_length
  let weekday_watch := monday_watch + thursday_watch + (friday_episodes * episode_length)
  total_watch - weekday_watch = weekend_watch := by sorry

end NUMINAMATH_CALUDE_maddie_weekend_watch_l2533_253334


namespace NUMINAMATH_CALUDE_square_of_sum_l2533_253329

theorem square_of_sum (x : ℝ) (h1 : x^2 - 49 ≥ 0) (h2 : x + 7 ≥ 0) :
  (7 - Real.sqrt (x^2 - 49) + Real.sqrt (x + 7))^2 =
  x^2 + x + 7 - 14 * Real.sqrt (x^2 - 49) - 14 * Real.sqrt (x + 7) + 2 * Real.sqrt (x^2 - 49) * Real.sqrt (x + 7) := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2533_253329


namespace NUMINAMATH_CALUDE_limit_equals_six_l2533_253320

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem limit_equals_six 
  (h : deriv f 2 = 3) : 
  ∀ ε > 0, ∃ δ > 0, ∀ x₀ ≠ 0, |x₀| < δ → 
    |((f (2 + 2*x₀) - f 2) / x₀) - 6| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_equals_six_l2533_253320


namespace NUMINAMATH_CALUDE_solve_rent_problem_l2533_253341

def rent_problem (n : ℕ) : Prop :=
  let original_average : ℚ := 800
  let increased_rent : ℚ := 800 * (1 + 1/4)
  let new_average : ℚ := 850
  (n * original_average + (increased_rent - 800)) / n = new_average

theorem solve_rent_problem : 
  ∃ (n : ℕ), n > 0 ∧ rent_problem n ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_rent_problem_l2533_253341


namespace NUMINAMATH_CALUDE_probability_even_sum_l2533_253363

def wheel1 : List ℕ := [1, 1, 2, 3, 3, 4]
def wheel2 : List ℕ := [2, 4, 5, 5, 6]

def is_even (n : ℕ) : Bool := n % 2 = 0

def count_even (l : List ℕ) : ℕ := (l.filter is_even).length

def total_outcomes : ℕ := wheel1.length * wheel2.length

def favorable_outcomes : ℕ := 
  (wheel1.filter is_even).length * (wheel2.filter is_even).length +
  (wheel1.filter (fun x => ¬(is_even x))).length * (wheel2.filter (fun x => ¬(is_even x))).length

theorem probability_even_sum : 
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_even_sum_l2533_253363


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l2533_253312

/-- Represents the profit share calculation for a business partnership. -/
structure ProfitShare where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of A and C. -/
def profit_share_difference (ps : ProfitShare) : ℕ :=
  let total_ratio := ps.a_investment + ps.b_investment + ps.c_investment
  let unit_profit := ps.b_profit * total_ratio / ps.b_investment
  let a_profit := unit_profit * ps.a_investment / total_ratio
  let c_profit := unit_profit * ps.c_investment / total_ratio
  c_profit - a_profit

/-- Theorem stating the difference in profit shares for the given scenario. -/
theorem profit_share_difference_example :
  profit_share_difference ⟨8000, 10000, 12000, 2000⟩ = 800 := by
  sorry


end NUMINAMATH_CALUDE_profit_share_difference_example_l2533_253312


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2533_253368

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ - 1 = 0) ∧ (x₂^2 + x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2533_253368


namespace NUMINAMATH_CALUDE_red_tickets_for_yellow_l2533_253316

/-- The number of yellow tickets needed to win a Bible -/
def yellow_tickets_needed : ℕ := 10

/-- The number of blue tickets needed to obtain one red ticket -/
def blue_per_red : ℕ := 10

/-- Tom's current yellow tickets -/
def tom_yellow : ℕ := 8

/-- Tom's current red tickets -/
def tom_red : ℕ := 3

/-- Tom's current blue tickets -/
def tom_blue : ℕ := 7

/-- Additional blue tickets Tom needs -/
def additional_blue : ℕ := 163

/-- The number of red tickets required to obtain one yellow ticket -/
def red_per_yellow : ℕ := 7

theorem red_tickets_for_yellow : 
  (yellow_tickets_needed - tom_yellow) * red_per_yellow = 
  (additional_blue + tom_blue) / blue_per_red - tom_red := by
  sorry

end NUMINAMATH_CALUDE_red_tickets_for_yellow_l2533_253316


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_four_l2533_253381

/-- Given a function f(x) = 4x + a²/x where x > 0 and x ∈ ℝ, 
    if f attains its minimum value at x = 2, then a = 4. -/
theorem min_value_implies_a_eq_four (a : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ (f : ℝ → ℝ), f x = 4*x + a^2/x) →
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x > 0 → f x ≥ f 2) →
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_a_eq_four_l2533_253381


namespace NUMINAMATH_CALUDE_daisies_given_away_l2533_253331

/-- Proves the number of daisies given away based on initial count, petals per daisy, and remaining petals --/
theorem daisies_given_away 
  (initial_daisies : ℕ) 
  (petals_per_daisy : ℕ) 
  (remaining_petals : ℕ) 
  (h1 : initial_daisies = 5)
  (h2 : petals_per_daisy = 8)
  (h3 : remaining_petals = 24) :
  initial_daisies - (remaining_petals / petals_per_daisy) = 2 :=
by
  sorry

#check daisies_given_away

end NUMINAMATH_CALUDE_daisies_given_away_l2533_253331


namespace NUMINAMATH_CALUDE_problem_solution_l2533_253335

theorem problem_solution (x y : ℝ) (h : |x - Real.sqrt 3 + 1| + Real.sqrt (y - 2) = 0) :
  (x = Real.sqrt 3 - 1 ∧ y = 2) ∧ x^2 + 2*x - 3*y = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2533_253335


namespace NUMINAMATH_CALUDE_rounded_product_less_than_original_l2533_253364

theorem rounded_product_less_than_original
  (x y z : ℝ)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (hz_pos : z > 0)
  (hxy : x > 2*y) :
  (x + z) * (y - z) < x * y :=
by sorry

end NUMINAMATH_CALUDE_rounded_product_less_than_original_l2533_253364


namespace NUMINAMATH_CALUDE_competition_results_l2533_253382

/-- Represents the final scores of competitors -/
structure Scores where
  A : ℚ
  B : ℚ
  C : ℚ

/-- Represents the points awarded for each position -/
structure PointSystem where
  first : ℚ
  second : ℚ
  third : ℚ

/-- Represents the number of times each competitor finished in each position -/
structure CompetitorResults where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The main theorem statement -/
theorem competition_results 
  (scores : Scores)
  (points : PointSystem)
  (A_results B_results C_results : CompetitorResults)
  (h_scores : scores.A = 22 ∧ scores.B = 9 ∧ scores.C = 9)
  (h_B_won_100m : B_results.first ≥ 1)
  (h_no_ties : ∀ event : ℕ, 
    A_results.first + B_results.first + C_results.first = event ∧
    A_results.second + B_results.second + C_results.second = event ∧
    A_results.third + B_results.third + C_results.third = event)
  (h_score_calculation : 
    scores.A = A_results.first * points.first + A_results.second * points.second + A_results.third * points.third ∧
    scores.B = B_results.first * points.first + B_results.second * points.second + B_results.third * points.third ∧
    scores.C = C_results.first * points.first + C_results.second * points.second + C_results.third * points.third)
  : (A_results.first + A_results.second + A_results.third = 4) ∧ 
    (B_results.first + B_results.second + B_results.third = 4) ∧ 
    (C_results.first + C_results.second + C_results.third = 4) ∧
    A_results.second ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l2533_253382


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2533_253365

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := 4 * i / (1 + i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2533_253365


namespace NUMINAMATH_CALUDE_sqrt_n_squared_minus_np_integer_l2533_253387

theorem sqrt_n_squared_minus_np_integer (p : ℕ) (hp : Prime p) (hodd : Odd p) :
  ∃! n : ℕ, n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n^2 - n*p = k^2 ∧ n = ((p + 1)^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_n_squared_minus_np_integer_l2533_253387


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2533_253372

theorem nested_fraction_equality : 
  (2 : ℚ) / (2 + 2 / (3 + 1 / 4)) = 13 / 17 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2533_253372


namespace NUMINAMATH_CALUDE_assistant_productivity_increase_l2533_253399

theorem assistant_productivity_increase 
  (original_bears : ℝ) 
  (original_hours : ℝ) 
  (bear_increase_rate : ℝ) 
  (hour_decrease_rate : ℝ) 
  (h₁ : bear_increase_rate = 0.8) 
  (h₂ : hour_decrease_rate = 0.1) 
  : (((1 + bear_increase_rate) * original_bears) / ((1 - hour_decrease_rate) * original_hours)) / 
    (original_bears / original_hours) - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_assistant_productivity_increase_l2533_253399


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l2533_253383

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPoly (p : ℤ → ℤ) : Prop :=
  ∃ a b c : ℤ, ∀ x, p x = a * x^2 + b * x + c

theorem quadratic_polynomial_value (p : ℤ → ℤ) :
  QuadraticPoly p →
  p 41 = 42 →
  (∃ a b : ℤ, a > 41 ∧ b > 41 ∧ p a = 13 ∧ p b = 73) →
  p 1 = 2842 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l2533_253383


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l2533_253371

/-- The number of sides in a regular hexadecagon -/
def n : ℕ := 16

/-- The number of vertices to choose for each triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular hexadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_triangles_l2533_253371


namespace NUMINAMATH_CALUDE_divisor_product_sum_theorem_l2533_253347

/-- The type of positive divisors of n -/
def Divisor (n : ℕ) := { d : ℕ // d > 0 ∧ n % d = 0 }

/-- The list of all positive divisors of n in ascending order -/
def divisors (n : ℕ) : List (Divisor n) := sorry

/-- The sum of products of consecutive divisors -/
def D (n : ℕ) : ℕ :=
  let ds := divisors n
  (List.zip ds (List.tail ds)).map (fun (d₁, d₂) => d₁.val * d₂.val) |>.sum

/-- The main theorem -/
theorem divisor_product_sum_theorem (n : ℕ) (h : n > 1) :
  D n < n^2 ∧ (D n ∣ n^2 ↔ Nat.Prime n) := by sorry

end NUMINAMATH_CALUDE_divisor_product_sum_theorem_l2533_253347


namespace NUMINAMATH_CALUDE_impossible_to_flip_all_l2533_253398

/-- Represents the color of a button's face -/
inductive ButtonColor
| White
| Black

/-- Represents a configuration of buttons in a circle -/
def ButtonConfiguration := List ButtonColor

/-- Represents a move in the game -/
inductive Move
| FlipAdjacent (i : Nat)  -- Flip two adjacent buttons at position i and i+1
| FlipSeparated (i : Nat) -- Flip two buttons at position i and i+2

/-- The initial configuration of buttons -/
def initial_config : ButtonConfiguration :=
  [ButtonColor.Black] ++ List.replicate 2021 ButtonColor.White

/-- Applies a move to a button configuration -/
def apply_move (config : ButtonConfiguration) (move : Move) : ButtonConfiguration :=
  sorry

/-- Checks if all buttons have been flipped from their initial state -/
def all_flipped (config : ButtonConfiguration) : Prop :=
  sorry

/-- The main theorem stating it's impossible to flip all buttons -/
theorem impossible_to_flip_all (moves : List Move) :
  ¬(all_flipped (moves.foldl apply_move initial_config)) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_flip_all_l2533_253398


namespace NUMINAMATH_CALUDE_vitamin_shop_lcm_l2533_253355

theorem vitamin_shop_lcm : ∃ n : ℕ, n > 0 ∧ n % 7 = 0 ∧ n % 17 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 7 = 0 ∧ m % 17 = 0) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_vitamin_shop_lcm_l2533_253355


namespace NUMINAMATH_CALUDE_unique_solution_unique_solution_l2533_253340

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, 3, k}
def B (a : ℕ) : Set ℕ := {4, 7, a^4, a^2 + 3*a}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- Theorem statement
theorem unique_solution (a k : ℕ) :
  (∀ x ∈ A k, ∃ y ∈ B a, f x = y) ∧ 
  (∀ y ∈ B a, ∃ x ∈ A k, f x = y) →
  a = 2 ∧ k = 5 := by
  sorry

-- Alternative theorem statement if the above doesn't compile
theorem unique_solution' (a k : ℕ) :
  (∀ x, x ∈ A k → ∃ y ∈ B a, f x = y) ∧ 
  (∀ y, y ∈ B a → ∃ x ∈ A k, f x = y) →
  a = 2 ∧ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_unique_solution_l2533_253340


namespace NUMINAMATH_CALUDE_jacksons_grade_calculation_l2533_253302

/-- Calculates Jackson's grade based on his time allocation and point system. -/
def jacksons_grade (video_game_hours : ℝ) (study_ratio : ℝ) (kindness_ratio : ℝ) 
  (study_points_per_hour : ℝ) (kindness_points_per_hour : ℝ) : ℝ :=
  let study_hours := video_game_hours * study_ratio
  let kindness_hours := video_game_hours * kindness_ratio
  study_hours * study_points_per_hour + kindness_hours * kindness_points_per_hour

theorem jacksons_grade_calculation :
  jacksons_grade 12 (1/3) (1/4) 20 40 = 200 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_grade_calculation_l2533_253302


namespace NUMINAMATH_CALUDE_dogs_with_tags_and_collars_l2533_253305

theorem dogs_with_tags_and_collars (total : ℕ) (tags : ℕ) (collars : ℕ) (neither : ℕ) 
  (h_total : total = 80)
  (h_tags : tags = 45)
  (h_collars : collars = 40)
  (h_neither : neither = 1) :
  total = tags + collars - (tags + collars - total + neither) := by
  sorry

#check dogs_with_tags_and_collars

end NUMINAMATH_CALUDE_dogs_with_tags_and_collars_l2533_253305


namespace NUMINAMATH_CALUDE_total_bouncy_balls_l2533_253377

def red_packs : ℕ := 4
def yellow_packs : ℕ := 8
def green_packs : ℕ := 4
def balls_per_pack : ℕ := 10

theorem total_bouncy_balls :
  (red_packs + yellow_packs + green_packs) * balls_per_pack = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_bouncy_balls_l2533_253377


namespace NUMINAMATH_CALUDE_hexagon_side_length_l2533_253344

/-- Given a triangle ABC, prove that a hexagon with sides parallel to the triangle's sides
    and equal length d satisfies the equation: d = (abc) / (ab + bc + ca) -/
theorem hexagon_side_length (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  d = (a * b * c) / (a * b + b * c + c * a) ↔
  (1 / d = 1 / a + 1 / b + 1 / c) :=
sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l2533_253344


namespace NUMINAMATH_CALUDE_heartsuit_not_commutative_l2533_253374

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heartsuit_not_commutative : ¬ ∀ (x y : ℝ), heartsuit x y = heartsuit y x := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_not_commutative_l2533_253374


namespace NUMINAMATH_CALUDE_max_value_abc_fraction_l2533_253380

theorem max_value_abc_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ (1 : ℝ) / 4 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * b * c * (a + b + c)) / ((a + b)^3 * (b + c)^3) = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_fraction_l2533_253380


namespace NUMINAMATH_CALUDE_order_of_four_numbers_l2533_253366

theorem order_of_four_numbers (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by sorry

end NUMINAMATH_CALUDE_order_of_four_numbers_l2533_253366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2533_253330

/-- The sum of the first n terms of an arithmetic sequence with a₁ = 23 and d = -2 -/
def S (n : ℕ+) : ℝ := -n.val^2 + 24 * n.val

/-- The maximum value of S(n) for positive integer n -/
def max_S : ℝ := 144

theorem arithmetic_sequence_max_sum :
  ∃ (n : ℕ+), S n = max_S ∧ ∀ (m : ℕ+), S m ≤ max_S := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2533_253330


namespace NUMINAMATH_CALUDE_male_average_grade_l2533_253319

/-- Proves that the average grade of male students is 87 given the conditions of the problem -/
theorem male_average_grade (total_average : ℝ) (female_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : female_average = 92)
  (h3 : male_count = 8)
  (h4 : female_count = 12) :
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 87 := by
  sorry

end NUMINAMATH_CALUDE_male_average_grade_l2533_253319


namespace NUMINAMATH_CALUDE_teddy_bear_production_solution_l2533_253300

/-- Represents the teddy bear production problem -/
structure TeddyBearProduction where
  /-- The number of days originally planned -/
  days : ℕ
  /-- The number of teddy bears ordered -/
  order : ℕ

/-- The conditions of the teddy bear production problem are satisfied -/
def satisfies_conditions (p : TeddyBearProduction) : Prop :=
  20 * p.days + 100 = p.order ∧ 23 * p.days - 20 = p.order

/-- The theorem stating the solution to the teddy bear production problem -/
theorem teddy_bear_production_solution :
  ∃ (p : TeddyBearProduction), satisfies_conditions p ∧ p.days = 40 ∧ p.order = 900 :=
sorry

end NUMINAMATH_CALUDE_teddy_bear_production_solution_l2533_253300


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2533_253357

/-- A line in the xy-plane can be represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are perpendicular if the product of their slopes is -1. -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_equation (l : Line) (h1 : l.y_intercept = 1) 
    (h2 : perpendicular l (Line.mk (1/2) 0)) : 
  l.slope = -2 ∧ ∀ x y : ℝ, y = l.slope * x + l.y_intercept ↔ y = -2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2533_253357


namespace NUMINAMATH_CALUDE_f_inequality_range_l2533_253395

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1/2 then 2*x + 1
  else if x < 3/2 then 2 - (3-2*x)
  else 2*x + 1 + (2*x-3)

theorem f_inequality_range (a : ℝ) :
  (∃ x, f x < 1) →
  (∀ x, f x ≤ |a|) →
  |a| ≥ 4 := by sorry

end NUMINAMATH_CALUDE_f_inequality_range_l2533_253395


namespace NUMINAMATH_CALUDE_distribute_books_equal_distribute_books_scenario1_distribute_books_scenario2_l2533_253304

/-- The number of ways to distribute 7 different books among 3 people -/
def distribute_books (scenario : Nat) : Nat :=
  match scenario with
  | 1 => 630  -- One person gets 1 book, one gets 2 books, and one gets 4 books
  | 2 => 630  -- One person gets 3 books, and two people each get 2 books
  | _ => 0    -- Invalid scenario

/-- Proof that both distribution scenarios result in 630 ways -/
theorem distribute_books_equal : distribute_books 1 = distribute_books 2 := by
  sorry

/-- Proof that the number of ways to distribute books in scenario 1 is 630 -/
theorem distribute_books_scenario1 : distribute_books 1 = 630 := by
  sorry

/-- Proof that the number of ways to distribute books in scenario 2 is 630 -/
theorem distribute_books_scenario2 : distribute_books 2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_distribute_books_equal_distribute_books_scenario1_distribute_books_scenario2_l2533_253304


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2533_253337

theorem fraction_to_decimal : (3 : ℚ) / 50 = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2533_253337


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l2533_253307

/-- Systematic sampling function -/
def systematicSample (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + interval * (n - 1)

/-- Theorem for systematic sampling in the given problem -/
theorem systematic_sampling_problem :
  let totalStudents : ℕ := 500
  let selectedStudents : ℕ := 50
  let interval : ℕ := totalStudents / selectedStudents
  let start : ℕ := 6
  ∀ n : ℕ, 
    125 ≤ systematicSample start interval n ∧ 
    systematicSample start interval n ≤ 140 → 
    systematicSample start interval n = 126 ∨ 
    systematicSample start interval n = 136 :=
by
  sorry

#check systematic_sampling_problem

end NUMINAMATH_CALUDE_systematic_sampling_problem_l2533_253307


namespace NUMINAMATH_CALUDE_game_night_group_division_l2533_253393

theorem game_night_group_division (n : ℕ) (h : n = 6) :
  Nat.choose n (n / 2) = 20 :=
by sorry

end NUMINAMATH_CALUDE_game_night_group_division_l2533_253393


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l2533_253336

/-- In a Cartesian coordinate system, the coordinates of a point with respect to the origin are equal to the point's coordinates. -/
theorem coordinates_wrt_origin (x y : ℝ) : (x, y) = (x, y) := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l2533_253336


namespace NUMINAMATH_CALUDE_complex_power_simplification_l2533_253321

theorem complex_power_simplification :
  (3 * (Complex.cos (30 * π / 180)) + 3 * Complex.I * (Complex.sin (30 * π / 180)))^4 =
  Complex.mk (-81/2) ((81 * Real.sqrt 3)/2) := by
sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l2533_253321


namespace NUMINAMATH_CALUDE_max_area_difference_l2533_253397

/-- Definition of the ellipse -/
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- One focus of the ellipse -/
def Focus : ℝ × ℝ := (-1, 0)

/-- S₁ is the area of triangle ABD -/
noncomputable def S₁ (A B D : ℝ × ℝ) : ℝ := sorry

/-- S₂ is the area of triangle ABC -/
noncomputable def S₂ (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum difference between S₁ and S₂ -/
theorem max_area_difference :
  ∃ (A B C D : ℝ × ℝ),
    Ellipse A.1 A.2 ∧ Ellipse B.1 B.2 ∧ Ellipse C.1 C.2 ∧ Ellipse D.1 D.2 ∧
    (∀ (E : ℝ × ℝ), Ellipse E.1 E.2 → |S₁ A B D - S₂ A B C| ≤ Real.sqrt 3) ∧
    (∃ (F G : ℝ × ℝ), Ellipse F.1 F.2 ∧ Ellipse G.1 G.2 ∧ |S₁ A B F - S₂ A B G| = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l2533_253397


namespace NUMINAMATH_CALUDE_M_equals_P_l2533_253353

/-- Set M defined as {x | x = a² + 1, a ∈ ℝ} -/
def M : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}

/-- Set P defined as {y | y = b² - 4b + 5, b ∈ ℝ} -/
def P : Set ℝ := {y | ∃ b : ℝ, y = b^2 - 4*b + 5}

/-- Theorem stating that M = P -/
theorem M_equals_P : M = P := by sorry

end NUMINAMATH_CALUDE_M_equals_P_l2533_253353


namespace NUMINAMATH_CALUDE_face_value_of_shares_l2533_253388

/-- Proves that the face value of shares is 40, given the dividend rate, return on investment, and purchase price. -/
theorem face_value_of_shares (dividend_rate : ℝ) (roi_rate : ℝ) (purchase_price : ℝ) :
  dividend_rate = 0.125 →
  roi_rate = 0.25 →
  purchase_price = 20 →
  dividend_rate * (purchase_price / roi_rate) = 40 := by
  sorry

end NUMINAMATH_CALUDE_face_value_of_shares_l2533_253388


namespace NUMINAMATH_CALUDE_sequence_length_five_l2533_253360

theorem sequence_length_five :
  ∃ (b₁ b₂ b₃ b₄ b₅ : ℕ), 
    b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧
    (2^433 + 1) / (2^49 + 1) = 2^b₁ + 2^b₂ + 2^b₃ + 2^b₄ + 2^b₅ := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_five_l2533_253360


namespace NUMINAMATH_CALUDE_probability_is_one_seventh_l2533_253348

/-- Represents the total number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks drawn -/
def socks_drawn : ℕ := 6

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the probability of drawing exactly two pairs of different colors -/
def probability_two_pairs : ℚ :=
  let total_outcomes := choose total_socks socks_drawn
  let favorable_outcomes := choose num_colors 2 * choose (num_colors - 2) 2
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_is_one_seventh :
  probability_two_pairs = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_seventh_l2533_253348


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2533_253310

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence : arithmetic_sequence 3 4 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2533_253310


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l2533_253301

/-- The average speed of a car given its distances traveled in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : d1 = 80 → d2 = 60 → (d1 + d2) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l2533_253301


namespace NUMINAMATH_CALUDE_betty_order_cost_l2533_253350

/-- The total cost of Betty's order -/
def total_cost (slippers_quantity : ℕ) (slippers_price : ℚ) 
               (lipstick_quantity : ℕ) (lipstick_price : ℚ) 
               (hair_color_quantity : ℕ) (hair_color_price : ℚ) : ℚ :=
  slippers_quantity * slippers_price + 
  lipstick_quantity * lipstick_price + 
  hair_color_quantity * hair_color_price

/-- Theorem stating that Betty's total order cost is $44 -/
theorem betty_order_cost : 
  total_cost 6 (5/2) 4 (5/4) 8 3 = 44 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_cost_l2533_253350


namespace NUMINAMATH_CALUDE_pulley_system_theorem_l2533_253369

/-- Represents the configuration of three pulleys --/
structure PulleySystem where
  r : ℝ  -- radius of pulleys
  d12 : ℝ  -- distance between O₁ and O₂
  d13 : ℝ  -- distance between O₁ and O₃
  h : ℝ  -- height of O₃ above the plane of O₁ and O₂

/-- Calculates the possible belt lengths for the pulley system --/
def beltLengths (p : PulleySystem) : Set ℝ :=
  { 32 + 4 * Real.pi, 22 + 2 * Real.sqrt 97 + 4 * Real.pi }

/-- Checks if a given cord length is always sufficient --/
def isAlwaysSufficient (p : PulleySystem) (cordLength : ℝ) : Prop :=
  ∀ l ∈ beltLengths p, l ≤ cordLength

theorem pulley_system_theorem (p : PulleySystem) 
    (h1 : p.r = 2)
    (h2 : p.d12 = 12)
    (h3 : p.d13 = 10)
    (h4 : p.h = 8) :
    (beltLengths p = { 32 + 4 * Real.pi, 22 + 2 * Real.sqrt 97 + 4 * Real.pi }) ∧
    (¬ isAlwaysSufficient p 54) := by
  sorry

end NUMINAMATH_CALUDE_pulley_system_theorem_l2533_253369


namespace NUMINAMATH_CALUDE_task_assignment_count_l2533_253367

/-- The number of ways to assign 4 students to 3 tasks -/
def task_assignments : ℕ := 12

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of tasks -/
def num_tasks : ℕ := 3

/-- The number of students assigned to clean the podium -/
def podium_cleaners : ℕ := 1

/-- The number of students assigned to sweep the floor -/
def floor_sweepers : ℕ := 1

/-- The number of students assigned to mop the floor -/
def floor_moppers : ℕ := 2

theorem task_assignment_count :
  task_assignments = num_students * (num_students - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_task_assignment_count_l2533_253367


namespace NUMINAMATH_CALUDE_like_terms_power_l2533_253351

theorem like_terms_power (a b : ℕ) : 
  (∀ x y : ℝ, ∃ c : ℝ, x^(a+1) * y^2 = c * x^3 * y^b) → 
  a^b = 4 := by
sorry

end NUMINAMATH_CALUDE_like_terms_power_l2533_253351


namespace NUMINAMATH_CALUDE_lauren_tuesday_earnings_l2533_253343

/-- Represents Lauren's earnings from her social media channel on Tuesday -/
def LaurenEarnings : ℝ → ℝ → ℕ → ℕ → ℝ :=
  λ commercial_rate subscription_rate commercial_views subscriptions =>
    commercial_rate * (commercial_views : ℝ) + subscription_rate * (subscriptions : ℝ)

/-- Theorem stating Lauren's earnings on Tuesday -/
theorem lauren_tuesday_earnings :
  LaurenEarnings 0.5 1 100 27 = 77 := by
  sorry

end NUMINAMATH_CALUDE_lauren_tuesday_earnings_l2533_253343


namespace NUMINAMATH_CALUDE_albums_theorem_l2533_253352

-- Define the number of albums for each person
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- Define the total number of albums
def total_albums : ℕ := adele_albums + bridget_albums + katrina_albums + miriam_albums

-- Theorem to prove
theorem albums_theorem : total_albums = 585 := by
  sorry

end NUMINAMATH_CALUDE_albums_theorem_l2533_253352


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2533_253308

/-- Given a complex number z satisfying (1 + z) / i = 1 - z, 
    the imaginary part of z is 1 -/
theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2533_253308


namespace NUMINAMATH_CALUDE_smallest_largest_product_l2533_253389

def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_three_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999

def uses_all_digits (a b c : Nat) : Prop :=
  (digits.card = 9) ∧
  (Finset.card (Finset.image (λ d => d % 10) {a, b, c, a / 10, b / 10, c / 10, a / 100, b / 100, c / 100}) = 9)

theorem smallest_largest_product :
  ∀ a b c : Nat,
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c →
  uses_all_digits a b c →
  (∀ x y z : Nat, is_three_digit x ∧ is_three_digit y ∧ is_three_digit z → uses_all_digits x y z → a * b * c ≤ x * y * z) ∧
  (∀ x y z : Nat, is_three_digit x ∧ is_three_digit y ∧ is_three_digit z → uses_all_digits x y z → x * y * z ≤ 941 * 852 * 763) :=
by sorry

end NUMINAMATH_CALUDE_smallest_largest_product_l2533_253389


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2533_253390

theorem trigonometric_identity : 
  (Real.sin (160 * π / 180) + Real.sin (40 * π / 180)) * 
  (Real.sin (140 * π / 180) + Real.sin (20 * π / 180)) + 
  (Real.sin (50 * π / 180) - Real.sin (70 * π / 180)) * 
  (Real.sin (130 * π / 180) - Real.sin (110 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2533_253390


namespace NUMINAMATH_CALUDE_absolute_difference_l2533_253376

theorem absolute_difference (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l2533_253376


namespace NUMINAMATH_CALUDE_inscribing_square_area_l2533_253362

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0

/-- The square inscribing the circle -/
structure InscribingSquare where
  side : ℝ
  center_x : ℝ
  center_y : ℝ
  inscribes_circle : ∀ (x y : ℝ), circle_equation x y →
    (|x - center_x| ≤ side / 2) ∧ (|y - center_y| ≤ side / 2)
  parallel_to_axes : True  -- This condition is implicit in the structure

/-- The theorem stating that the area of the inscribing square is 4 -/
theorem inscribing_square_area :
  ∀ (s : InscribingSquare), s.side^2 = 4 := by sorry

end NUMINAMATH_CALUDE_inscribing_square_area_l2533_253362


namespace NUMINAMATH_CALUDE_high_heels_cost_high_heels_cost_proof_l2533_253324

/-- The cost of one pair of high heels given the following conditions:
  - Fern buys one pair of high heels and five pairs of ballet slippers
  - The price of five pairs of ballet slippers is 2/3 of the price of the high heels
  - The total cost is $260
-/
theorem high_heels_cost : ℝ → Prop :=
  fun high_heels_price =>
    let ballet_slippers_price := (2 / 3) * high_heels_price
    let total_cost := high_heels_price + 5 * ballet_slippers_price
    total_cost = 260 → high_heels_price = 60

/-- Proof of the high heels cost theorem -/
theorem high_heels_cost_proof : ∃ (price : ℝ), high_heels_cost price :=
  sorry

end NUMINAMATH_CALUDE_high_heels_cost_high_heels_cost_proof_l2533_253324


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2533_253323

/-- Trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := y^2 = 8*x

/-- Line l passing through a point (x, y) with slope t -/
def line_l (t m x y : ℝ) : Prop := x = t*y + m

/-- Angle bisector condition for ∠PBQ -/
def angle_bisector (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ / (x₁ + 3) + y₂ / (x₂ + 3) = 0

/-- Main theorem: Line l passes through (3, 0) given the conditions -/
theorem line_passes_through_fixed_point
  (t m x₁ y₁ x₂ y₂ : ℝ)
  (h_traj₁ : trajectory x₁ y₁)
  (h_traj₂ : trajectory x₂ y₂)
  (h_line₁ : line_l t m x₁ y₁)
  (h_line₂ : line_l t m x₂ y₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_not_vertical : t ≠ 0)
  (h_bisector : angle_bisector x₁ y₁ x₂ y₂) :
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2533_253323


namespace NUMINAMATH_CALUDE_count_zeros_up_to_3017_l2533_253356

/-- A function that checks if a positive integer contains the digit 0 in its base-ten representation -/
def containsZero (n : ℕ+) : Bool :=
  sorry

/-- The count of positive integers less than or equal to 3017 that contain the digit 0 -/
def countZeros : ℕ :=
  sorry

/-- Theorem stating that the count of positive integers less than or equal to 3017
    containing the digit 0 is equal to 1011 -/
theorem count_zeros_up_to_3017 : countZeros = 1011 := by
  sorry

end NUMINAMATH_CALUDE_count_zeros_up_to_3017_l2533_253356


namespace NUMINAMATH_CALUDE_fishing_line_sections_l2533_253318

theorem fishing_line_sections (num_reels : ℕ) (reel_length : ℕ) (section_length : ℕ) : 
  num_reels = 3 → reel_length = 100 → section_length = 10 → 
  (num_reels * reel_length) / section_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_fishing_line_sections_l2533_253318


namespace NUMINAMATH_CALUDE_composition_equality_l2533_253339

/-- Given two functions f and g, prove that if f(g(b)) = 4, then b = -1/2 -/
theorem composition_equality (f g : ℝ → ℝ) (b : ℝ) 
    (hf : ∀ x, f x = x / 3 + 2)
    (hg : ∀ x, g x = 5 - 2 * x)
    (h : f (g b) = 4) : 
  b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_composition_equality_l2533_253339


namespace NUMINAMATH_CALUDE_quadratic_degeneracy_l2533_253322

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the roots of an equation -/
inductive Root
  | Finite (x : ℝ)
  | Infinity

/-- 
Given a quadratic equation ax² + bx + c = 0 where a = 0,
prove that it has one finite root -c/b and one root at infinity.
-/
theorem quadratic_degeneracy (eq : QuadraticEquation) (h : eq.a = 0) :
  ∃ (r₁ r₂ : Root), 
    r₁ = Root.Finite (-eq.c / eq.b) ∧ 
    r₂ = Root.Infinity ∧
    eq.b * (-eq.c / eq.b) + eq.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_degeneracy_l2533_253322


namespace NUMINAMATH_CALUDE_inequality_holds_iff_first_quadrant_l2533_253394

theorem inequality_holds_iff_first_quadrant (θ : Real) :
  (∀ x : Real, x ∈ Set.Icc 0 1 →
    x^2 * Real.cos θ - 3 * x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
  θ ∈ Set.Ioo 0 (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_first_quadrant_l2533_253394


namespace NUMINAMATH_CALUDE_suv_max_distance_l2533_253392

/-- Calculates the maximum distance an SUV can travel given its fuel efficiencies and available fuel -/
theorem suv_max_distance (highway_mpg city_mpg mountain_mpg : ℝ) (fuel : ℝ) : 
  highway_mpg = 12.2 →
  city_mpg = 7.6 →
  mountain_mpg = 9.4 →
  fuel = 22 →
  (highway_mpg + city_mpg + mountain_mpg) * fuel = 642.4 := by
  sorry

end NUMINAMATH_CALUDE_suv_max_distance_l2533_253392


namespace NUMINAMATH_CALUDE_james_age_proof_l2533_253338

/-- James' age -/
def james_age : ℝ := 47.5

/-- Mara's age -/
def mara_age : ℝ := 22.5

/-- James' age is 20 years less than three times Mara's age -/
axiom age_relation : james_age = 3 * mara_age - 20

/-- The sum of their ages is 70 -/
axiom age_sum : james_age + mara_age = 70

theorem james_age_proof : james_age = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_james_age_proof_l2533_253338


namespace NUMINAMATH_CALUDE_decimal_division_proof_l2533_253317

theorem decimal_division_proof : (0.045 : ℝ) / (0.005 : ℝ) = 9 := by sorry

end NUMINAMATH_CALUDE_decimal_division_proof_l2533_253317


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2533_253314

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  discriminant 5 (-9) (-7) = 221 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2533_253314


namespace NUMINAMATH_CALUDE_problem_1_l2533_253313

theorem problem_1 : (-16) + 28 + (-128) - (-66) = -50 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2533_253313


namespace NUMINAMATH_CALUDE_model_a_better_fit_l2533_253332

def model_a (x : ℝ) : ℝ := x^2 + 1
def model_b (x : ℝ) : ℝ := 3*x - 1

def data_points : List (ℝ × ℝ) := [(1, 2), (2, 5), (3, 10.2)]

def error (model : ℝ → ℝ) (point : ℝ × ℝ) : ℝ :=
  (model point.1 - point.2)^2

def total_error (model : ℝ → ℝ) (points : List (ℝ × ℝ)) : ℝ :=
  points.foldl (λ acc p => acc + error model p) 0

theorem model_a_better_fit :
  total_error model_a data_points < total_error model_b data_points := by
  sorry

end NUMINAMATH_CALUDE_model_a_better_fit_l2533_253332


namespace NUMINAMATH_CALUDE_reflection_of_M_l2533_253378

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point M -/
def M : ℝ × ℝ := (5, 2)

/-- Theorem: The reflection of M(5, 2) across the x-axis is (5, -2) -/
theorem reflection_of_M : reflect_x M = (5, -2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_l2533_253378


namespace NUMINAMATH_CALUDE_largest_difference_l2533_253345

def P : ℕ := 3 * 1003^1004
def Q : ℕ := 1003^1004
def R : ℕ := 1002 * 1003^1003
def S : ℕ := 3 * 1003^1003
def T : ℕ := 1003^1003
def U : ℕ := 1003^1002 * Nat.factorial 1002

theorem largest_difference (P Q R S T U : ℕ) 
  (hP : P = 3 * 1003^1004)
  (hQ : Q = 1003^1004)
  (hR : R = 1002 * 1003^1003)
  (hS : S = 3 * 1003^1003)
  (hT : T = 1003^1003)
  (hU : U = 1003^1002 * Nat.factorial 1002) :
  P - Q > max (Q - R) (max (R - S) (max (S - T) (T - U))) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l2533_253345


namespace NUMINAMATH_CALUDE_alice_forest_walk_l2533_253375

def morning_walk : ℕ := 10
def days_per_week : ℕ := 5
def total_distance : ℕ := 110

theorem alice_forest_walk :
  let morning_total := morning_walk * days_per_week
  let forest_total := total_distance - morning_total
  forest_total / days_per_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_forest_walk_l2533_253375


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2533_253361

noncomputable section

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the line l
def l (x y m : ℝ) : Prop := y = x + m

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem ellipse_intersection_theorem :
  ∃ (xA yA xB yB m x₀ : ℝ),
    Γ xA yA ∧ Γ xB yB ∧  -- A and B are on the ellipse
    l xA yA m ∧ l xB yB m ∧  -- A and B are on the line l
    distance xA yA xB yB = 3 * Real.sqrt 2 ∧  -- |AB| = 3√2
    distance x₀ 2 xA yA = distance x₀ 2 xB yB ∧  -- |PA| = |PB|
    (x₀ = -3 ∨ x₀ = -1) :=
by sorry

end

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2533_253361


namespace NUMINAMATH_CALUDE_small_boxes_count_l2533_253303

theorem small_boxes_count (total_bars : ℕ) (bars_per_box : ℕ) (h1 : total_bars = 640) (h2 : bars_per_box = 32) :
  total_bars / bars_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_count_l2533_253303


namespace NUMINAMATH_CALUDE_divisible_by_eight_l2533_253354

theorem divisible_by_eight (n : ℕ) : ∃ k : ℤ, 5^n + 2 * 3^(n-1) + 1 = 8*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l2533_253354


namespace NUMINAMATH_CALUDE_green_probability_is_half_l2533_253373

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  yellow_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def green_probability (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a specific colored cube is 1/2 -/
theorem green_probability_is_half :
  let cube : ColoredCube := {
    total_faces := 6,
    green_faces := 3,
    yellow_faces := 2,
    red_faces := 1
  }
  green_probability cube = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_green_probability_is_half_l2533_253373


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2533_253342

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2533_253342


namespace NUMINAMATH_CALUDE_symmetry_y_axis_coordinates_l2533_253327

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem symmetry_y_axis_coordinates :
  let A : Point := { x := -1, y := 2 }
  let B : Point := symmetricYAxis A
  B.x = 1 ∧ B.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_coordinates_l2533_253327


namespace NUMINAMATH_CALUDE_certain_number_problem_l2533_253326

theorem certain_number_problem (C : ℝ) : C - |(-10 + 6)| = 26 → C = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2533_253326


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l2533_253370

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l2533_253370


namespace NUMINAMATH_CALUDE_no_representation_as_sum_of_squares_and_ninth_power_l2533_253328

theorem no_representation_as_sum_of_squares_and_ninth_power (p : ℕ) (m : ℤ) 
  (h_prime : Nat.Prime p) (h_form : p = 4 * m + 1) :
  ¬ ∃ (x y z : ℤ), 216 * (p : ℤ)^3 = x^2 + y^2 + z^9 := by
  sorry

end NUMINAMATH_CALUDE_no_representation_as_sum_of_squares_and_ninth_power_l2533_253328


namespace NUMINAMATH_CALUDE_line_not_through_point_l2533_253346

theorem line_not_through_point (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + (2*m+1)*x₁ + m^2 + 4 = 0 ∧ x₂^2 + (2*m+1)*x₂ + m^2 + 4 = 0) →
  ¬((2*m-3)*(-2) - 4*m + 7 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_not_through_point_l2533_253346


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2533_253325

theorem smallest_positive_integer_congruence :
  ∃! (x : ℕ), x > 0 ∧ (45 * x + 13) % 30 = 5 ∧ ∀ (y : ℕ), y > 0 ∧ (45 * y + 13) % 30 = 5 → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2533_253325


namespace NUMINAMATH_CALUDE_ratio_comparison_correct_l2533_253311

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
def sport_water_ratio : ℚ := 1 / 2

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 4

/-- Amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 60

/-- The ratio of (flavoring to corn syrup in sport formulation) to (flavoring to corn syrup in standard formulation) -/
def ratio_comparison : ℚ := 3

/-- Theorem stating that the ratio comparison is correct given the problem conditions -/
theorem ratio_comparison_correct : 
  let standard_flavoring_to_corn := standard_ratio 0 / standard_ratio 1
  let sport_flavoring := sport_water * (sport_water_ratio * (standard_ratio 0 / standard_ratio 2))
  let sport_flavoring_to_corn := sport_flavoring / sport_corn_syrup
  (sport_flavoring_to_corn / standard_flavoring_to_corn) = ratio_comparison := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_correct_l2533_253311


namespace NUMINAMATH_CALUDE_purple_sequins_count_l2533_253358

/-- The number of purple sequins in each row on Jane's costume. -/
def purple_sequins_per_row : ℕ :=
  let total_sequins : ℕ := 162
  let blue_rows : ℕ := 6
  let blue_per_row : ℕ := 8
  let purple_rows : ℕ := 5
  let green_rows : ℕ := 9
  let green_per_row : ℕ := 6
  let blue_sequins : ℕ := blue_rows * blue_per_row
  let green_sequins : ℕ := green_rows * green_per_row
  let purple_sequins : ℕ := total_sequins - (blue_sequins + green_sequins)
  purple_sequins / purple_rows

theorem purple_sequins_count : purple_sequins_per_row = 12 := by
  sorry

end NUMINAMATH_CALUDE_purple_sequins_count_l2533_253358


namespace NUMINAMATH_CALUDE_hidden_square_exists_l2533_253386

theorem hidden_square_exists (ℓ : ℕ) : ∃ (x y : ℤ) (p : Fin ℓ → Fin ℓ → ℕ), 
  (∀ (i j : Fin ℓ), Nat.Prime (p i j)) ∧ 
  (∀ (i j k m : Fin ℓ), i ≠ k ∨ j ≠ m → p i j ≠ p k m) ∧
  (∀ (i j : Fin ℓ), x ≡ -i.val [ZMOD (p i j)] ∧ y ≡ -j.val [ZMOD (p i j)]) :=
sorry

end NUMINAMATH_CALUDE_hidden_square_exists_l2533_253386


namespace NUMINAMATH_CALUDE_mork_tax_rate_calculation_l2533_253384

-- Define the variables
def mork_income : ℝ := sorry
def mork_tax_rate : ℝ := sorry
def mindy_tax_rate : ℝ := 0.25
def combined_tax_rate : ℝ := 0.28

-- Define the theorem
theorem mork_tax_rate_calculation :
  mork_tax_rate = 0.4 :=
by
  -- Assume the conditions
  have h1 : mindy_tax_rate = 0.25 := rfl
  have h2 : combined_tax_rate = 0.28 := rfl
  have h3 : mork_income > 0 := sorry
  have h4 : mork_tax_rate * mork_income + mindy_tax_rate * (4 * mork_income) = combined_tax_rate * (5 * mork_income) := sorry

  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_mork_tax_rate_calculation_l2533_253384


namespace NUMINAMATH_CALUDE_no_equal_arithmetic_operations_l2533_253359

theorem no_equal_arithmetic_operations (v t : ℝ) (hv : v > 0) (ht : t > 0) : 
  ¬(v + t = v * t ∧ v + t = v / t) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_arithmetic_operations_l2533_253359


namespace NUMINAMATH_CALUDE_next_multiple_year_l2533_253396

theorem next_multiple_year : ∀ n : ℕ, 
  n > 2016 ∧ 
  n % 6 = 0 ∧ 
  n % 8 = 0 ∧ 
  n % 9 = 0 → 
  n ≥ 2088 :=
by
  sorry

end NUMINAMATH_CALUDE_next_multiple_year_l2533_253396


namespace NUMINAMATH_CALUDE_game_probabilities_l2533_253391

/-- Represents the number of balls of each color in the bag -/
def num_balls_per_color : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := 3 * num_balls_per_color

/-- Represents the number of balls drawn in each game -/
def balls_drawn : ℕ := 3

/-- Represents the number of people participating in the game -/
def num_participants : ℕ := 3

/-- Calculates the probability of winning for one person -/
def prob_win : ℚ := 2 / 5

/-- Calculates the probability that exactly 1 person wins out of 3 -/
def prob_one_winner : ℚ := 54 / 125

theorem game_probabilities :
  (prob_win = 2 / 5) ∧
  (prob_one_winner = 54 / 125) := by
  sorry

end NUMINAMATH_CALUDE_game_probabilities_l2533_253391
