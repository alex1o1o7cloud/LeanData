import Mathlib

namespace rates_sum_of_squares_l592_59238

/-- Represents the rates of biking, jogging, and swimming -/
structure Rates where
  bike : ℕ
  jog : ℕ
  swim : ℕ

/-- The problem statement -/
theorem rates_sum_of_squares (r : Rates) : r.bike^2 + r.jog^2 + r.swim^2 = 314 :=
  by
  have h1 : 2 * r.bike + 3 * r.jog + 4 * r.swim = 74 := by sorry
  have h2 : 4 * r.bike + 2 * r.jog + 3 * r.swim = 91 := by sorry
  sorry

#check rates_sum_of_squares

end rates_sum_of_squares_l592_59238


namespace rectangle_area_diagonal_l592_59240

theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diag : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end rectangle_area_diagonal_l592_59240


namespace unique_solution_l592_59279

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for clothing
structure Clothing :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
structure Children :=
  (alyna : Clothing)
  (bohdan : Clothing)
  (vika : Clothing)
  (grysha : Clothing)

-- Define the conditions
def satisfiesConditions (c : Children) : Prop :=
  (c.alyna.tshirt = Color.Red) ∧
  (c.bohdan.tshirt = Color.Red) ∧
  (c.alyna.shorts ≠ c.bohdan.shorts) ∧
  (c.vika.tshirt ≠ c.grysha.tshirt) ∧
  (c.vika.shorts = Color.Blue) ∧
  (c.grysha.shorts = Color.Blue) ∧
  (c.alyna.tshirt ≠ c.vika.tshirt) ∧
  (c.alyna.shorts ≠ c.vika.shorts)

-- Define the correct answer
def correctAnswer : Children :=
  { alyna := { tshirt := Color.Red, shorts := Color.Red },
    bohdan := { tshirt := Color.Red, shorts := Color.Blue },
    vika := { tshirt := Color.Blue, shorts := Color.Blue },
    grysha := { tshirt := Color.Red, shorts := Color.Blue } }

-- Theorem statement
theorem unique_solution :
  ∀ c : Children, satisfiesConditions c → c = correctAnswer :=
sorry

end unique_solution_l592_59279


namespace symmetry_sum_theorem_l592_59298

/-- Properties of a regular 25-gon -/
structure RegularPolygon25 where
  /-- Number of lines of symmetry -/
  L : ℕ
  /-- Smallest positive angle for rotational symmetry in degrees -/
  R : ℝ
  /-- The polygon has 25 sides -/
  sides_eq : L = 25
  /-- The smallest rotational symmetry angle is 360/25 degrees -/
  angle_eq : R = 360 / 25

/-- Theorem about the sum of symmetry lines and half the rotational angle -/
theorem symmetry_sum_theorem (p : RegularPolygon25) :
  p.L + p.R / 2 = 32.2 := by
  sorry

end symmetry_sum_theorem_l592_59298


namespace largest_x_absolute_value_equation_l592_59232

theorem largest_x_absolute_value_equation : 
  (∃ x : ℝ, |5*x - 3| = 28) → 
  (∃ max_x : ℝ, |5*max_x - 3| = 28 ∧ ∀ y : ℝ, |5*y - 3| = 28 → y ≤ max_x) → 
  (∃ x : ℝ, |5*x - 3| = 28 ∧ ∀ y : ℝ, |5*y - 3| = 28 → y ≤ 31/5) :=
by sorry

end largest_x_absolute_value_equation_l592_59232


namespace solve_shoe_price_l592_59259

def shoe_price_problem (rebate_percentage : ℝ) (num_pairs : ℕ) (total_rebate : ℝ) : Prop :=
  let original_price := total_rebate / (rebate_percentage * num_pairs : ℝ)
  original_price = 28

theorem solve_shoe_price :
  shoe_price_problem 0.1 5 14 := by
  sorry

end solve_shoe_price_l592_59259


namespace smaller_part_area_l592_59211

/-- The area of the smaller part of a field satisfying given conditions -/
theorem smaller_part_area (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 1800 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (smaller_area + larger_area) / 6 →
  smaller_area = 750 := by
  sorry

end smaller_part_area_l592_59211


namespace sandwich_cost_is_three_l592_59203

/-- The cost of a sandwich given the total cost and number of items. -/
def sandwich_cost (total_cost : ℚ) (water_cost : ℚ) (num_sandwiches : ℕ) : ℚ :=
  (total_cost - water_cost) / num_sandwiches

/-- Theorem stating that the cost of each sandwich is 3 given the problem conditions. -/
theorem sandwich_cost_is_three :
  sandwich_cost 11 2 3 = 3 := by
  sorry

end sandwich_cost_is_three_l592_59203


namespace number_of_subsets_l592_59222

theorem number_of_subsets (S : Set ℕ) : 
  (∃ (B : Set ℕ), {1, 2} ⊆ B ∧ B ⊆ {1, 2, 3}) ∧ 
  (∀ (B : Set ℕ), {1, 2} ⊆ B ∧ B ⊆ {1, 2, 3} → B = {1, 2} ∨ B = {1, 2, 3}) :=
by sorry

end number_of_subsets_l592_59222


namespace furniture_by_design_salary_l592_59209

/-- The monthly salary from Furniture by Design -/
def S : ℝ := 1800

/-- The base salary for the commission-based option -/
def base_salary : ℝ := 1600

/-- The commission rate for the commission-based option -/
def commission_rate : ℝ := 0.04

/-- The sales amount at which both payment options are equal -/
def equal_sales : ℝ := 5000

theorem furniture_by_design_salary :
  S = base_salary + commission_rate * equal_sales :=
by sorry

end furniture_by_design_salary_l592_59209


namespace smallest_sum_of_two_three_digit_numbers_l592_59218

-- Define a type for 3-digit numbers
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }

-- Define a function to check if a number uses given digits
def usesGivenDigits (n : ℕ) (digits : List ℕ) : Prop := sorry

-- Define a function to check if two numbers use all given digits exactly once
def useAllDigitsOnce (a b : ℕ) (digits : List ℕ) : Prop := sorry

-- Theorem statement
theorem smallest_sum_of_two_three_digit_numbers :
  ∃ (a b : ThreeDigitNumber),
    useAllDigitsOnce a.val b.val [1, 2, 3, 7, 8, 9] ∧
    (∀ (x y : ThreeDigitNumber),
      useAllDigitsOnce x.val y.val [1, 2, 3, 7, 8, 9] →
      a.val + b.val ≤ x.val + y.val) ∧
    a.val + b.val = 912 := by
  sorry

end smallest_sum_of_two_three_digit_numbers_l592_59218


namespace inequality_solution_set_l592_59235

theorem inequality_solution_set :
  {x : ℝ | 3 * x - 4 > 2} = {x : ℝ | x > 2} := by
  sorry

end inequality_solution_set_l592_59235


namespace distance_inequality_l592_59245

-- Define the types for planes, lines, and points
variable (Plane Line Point : Type)

-- Define the distance function
variable (distance : Point → Point → ℝ)
variable (distance_point_line : Point → Line → ℝ)
variable (distance_line_line : Line → Line → ℝ)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the containment relations
variable (line_in_plane : Line → Plane → Prop)
variable (point_on_line : Point → Line → Prop)

-- Define the specific objects in the problem
variable (α β : Plane) (m n : Line) (A B : Point)

-- Define the distances
variable (a b c : ℝ)

theorem distance_inequality 
  (h_parallel : parallel α β)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_β : line_in_plane n β)
  (h_A_on_m : point_on_line A m)
  (h_B_on_n : point_on_line B n)
  (h_a_def : a = distance A B)
  (h_b_def : b = distance_point_line A n)
  (h_c_def : c = distance_line_line m n) :
  c ≤ b ∧ b ≤ a :=
by sorry

end distance_inequality_l592_59245


namespace product_of_solutions_l592_59208

theorem product_of_solutions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016) :
  (2 - x₁/y₁) * (2 - x₂/y₂) * (2 - x₃/y₃) = 26219/2016 := by
sorry

end product_of_solutions_l592_59208


namespace point_on_line_l592_59233

/-- A point is on a line if it satisfies the equation of the line formed by two other points. -/
def is_on_line (x₁ y₁ x₂ y₂ x y : ℚ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The point (5, 56/3) is on the line formed by (8, 16) and (2, 0). -/
theorem point_on_line : is_on_line 8 16 2 0 5 (56/3) := by
  sorry

end point_on_line_l592_59233


namespace zero_in_interval_l592_59256

def f (x : ℝ) := 2*x + 3*x

theorem zero_in_interval : ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 := by
  sorry

end zero_in_interval_l592_59256


namespace trapezoid_segment_length_l592_59283

/-- Given a trapezoid ABCD where the ratio of the area of triangle ABC to the area of triangle ADC
    is 7:3, and AB + CD = 280, prove that AB = 196. -/
theorem trapezoid_segment_length (AB CD : ℝ) (h : ℝ) : 
  (AB * h / 2) / (CD * h / 2) = 7 / 3 →
  AB + CD = 280 →
  AB = 196 := by
sorry

end trapezoid_segment_length_l592_59283


namespace stock_loss_percentage_l592_59264

theorem stock_loss_percentage (total_stock : ℝ) (profit_percentage : ℝ) (profit_portion : ℝ) (overall_loss : ℝ) :
  total_stock = 12500 →
  profit_percentage = 10 →
  profit_portion = 20 →
  overall_loss = 250 →
  ∃ (L : ℝ),
    overall_loss = (1 - profit_portion / 100) * total_stock * (L / 100) - (profit_portion / 100) * total_stock * (profit_percentage / 100) ∧
    L = 5 := by
  sorry

end stock_loss_percentage_l592_59264


namespace crayons_per_child_l592_59258

theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 7) 
  (h2 : total_crayons = 56) : 
  total_crayons / total_children = 8 := by
  sorry

end crayons_per_child_l592_59258


namespace three_person_subcommittees_from_eight_l592_59295

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l592_59295


namespace fraction_inequivalence_l592_59204

theorem fraction_inequivalence :
  ∃ k : ℝ, k ≠ 0 ∧ k ≠ -1 ∧ (3 * k + 9) / (4 * k + 4) ≠ 3 / 4 := by
  sorry

end fraction_inequivalence_l592_59204


namespace video_game_points_calculation_l592_59257

/-- Calculate points earned in a video game level --/
theorem video_game_points_calculation
  (points_per_enemy : ℕ)
  (bonus_points : ℕ)
  (total_enemies : ℕ)
  (defeated_enemies : ℕ)
  (bonuses_earned : ℕ)
  (h1 : points_per_enemy = 15)
  (h2 : bonus_points = 50)
  (h3 : total_enemies = 25)
  (h4 : defeated_enemies = total_enemies - 5)
  (h5 : bonuses_earned = 2)
  : defeated_enemies * points_per_enemy + bonuses_earned * bonus_points = 400 := by
  sorry

#check video_game_points_calculation

end video_game_points_calculation_l592_59257


namespace solution_for_all_polynomials_l592_59269

/-- A polynomial of degree 3 in x and y -/
def q (b₁ b₂ b₄ b₇ b₈ : ℝ) (x y : ℝ) : ℝ :=
  b₁ * x * (1 - x^2) + b₂ * y * (1 - y^2) + b₄ * (x * y - x^2 * y) + b₇ * x^2 * y + b₈ * x * y^2

/-- The theorem stating that (√(3/2), √(3/2)) is a solution for all such polynomials -/
theorem solution_for_all_polynomials (b₁ b₂ b₄ b₇ b₈ : ℝ) :
  let q := q b₁ b₂ b₄ b₇ b₈
  (q 0 0 = 0) →
  (q 1 0 = 0) →
  (q (-1) 0 = 0) →
  (q 0 1 = 0) →
  (q 0 (-1) = 0) →
  (q 1 1 = 0) →
  (q (-1) (-1) = 0) →
  (q 2 2 = 0) →
  (deriv (fun x => q x 1) 1 = 0) →
  (deriv (fun y => q 1 y) 1 = 0) →
  q (Real.sqrt (3/2)) (Real.sqrt (3/2)) = 0 := by
  sorry

end solution_for_all_polynomials_l592_59269


namespace cow_count_l592_59225

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- The total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- Theorem: In a group where the total number of legs is 12 more than twice 
    the number of heads, the number of cows is 6 -/
theorem cow_count (group : AnimalGroup) 
    (h : totalLegs group = 2 * totalHeads group + 12) : 
    group.cows = 6 := by
  sorry


end cow_count_l592_59225


namespace special_square_area_l592_59271

/-- A square with special points and segments -/
structure SpecialSquare where
  -- The side length of the square
  side : ℝ
  -- The length of BR
  br : ℝ
  -- The length of PR
  pr : ℝ
  -- Assumption that BR = 9
  br_eq : br = 9
  -- Assumption that PR = 12
  pr_eq : pr = 12
  -- Assumption that BP and CQ intersect at right angles
  right_angle : True

/-- The theorem stating that the area of the special square is 324 -/
theorem special_square_area (s : SpecialSquare) : s.side ^ 2 = 324 := by
  sorry

end special_square_area_l592_59271


namespace roots_sum_greater_than_twice_zero_l592_59244

noncomputable section

open Real

def f (x : ℝ) := x * log x
def g (x : ℝ) := x / exp x
def F (x : ℝ) := f x - g x
def m (x : ℝ) := min (f x) (g x)

theorem roots_sum_greater_than_twice_zero 
  (x₀ : ℝ) 
  (h₁ : 1 < x₀ ∧ x₀ < 2) 
  (h₂ : F x₀ = 0) 
  (h₃ : ∀ x, 1 < x ∧ x < 2 ∧ F x = 0 → x = x₀)
  (x₁ x₂ : ℝ) 
  (h₄ : 1 < x₁ ∧ x₁ < x₂)
  (h₅ : ∃ n, m x₁ = n ∧ m x₂ = n)
  : x₁ + x₂ > 2 * x₀ := by
  sorry

end roots_sum_greater_than_twice_zero_l592_59244


namespace mary_received_more_l592_59250

/-- Calculates the profit share difference between two partners in a business --/
def profit_share_difference (mary_investment : ℚ) (harry_investment : ℚ) (total_profit : ℚ) : ℚ :=
  let equal_share := (1 / 3) * total_profit / 2
  let investment_based_profit := (2 / 3) * total_profit
  let mary_investment_share := (mary_investment / (mary_investment + harry_investment)) * investment_based_profit
  let harry_investment_share := (harry_investment / (mary_investment + harry_investment)) * investment_based_profit
  let mary_total := equal_share + mary_investment_share
  let harry_total := equal_share + harry_investment_share
  mary_total - harry_total

/-- Theorem stating that Mary received $800 more than Harry --/
theorem mary_received_more (mary_investment harry_investment total_profit : ℚ) :
  mary_investment = 700 →
  harry_investment = 300 →
  total_profit = 3000 →
  profit_share_difference mary_investment harry_investment total_profit = 800 := by
  sorry

#eval profit_share_difference 700 300 3000

end mary_received_more_l592_59250


namespace quadratic_radical_sum_l592_59254

/-- 
Given that √(3b-1) and ∜(7-b) are of the same type of quadratic radical,
where ∜ represents the (a-1)th root, prove that a + b = 5.
-/
theorem quadratic_radical_sum (a b : ℝ) : 
  (a - 1 = 2) → (3*b - 1 = 7 - b) → a + b = 5 := by
  sorry

end quadratic_radical_sum_l592_59254


namespace overtime_probability_l592_59284

theorem overtime_probability (p_chen p_li p_both : ℝ) : 
  p_chen = 1/3 →
  p_li = 1/4 →
  p_both = 1/6 →
  p_both / p_li = 2/3 := by
sorry

end overtime_probability_l592_59284


namespace smallest_n_property_l592_59223

/-- The smallest positive integer N such that N and N^2 end in the same three-digit sequence abc in base 10, where a is not zero -/
def smallest_n : ℕ := 876

theorem smallest_n_property : 
  ∀ n : ℕ, n > 0 → 
  (n % 1000 = smallest_n % 1000 ∧ n^2 % 1000 = smallest_n % 1000 ∧ (smallest_n % 1000) ≥ 100) → 
  n ≥ smallest_n := by
  sorry

#eval smallest_n

end smallest_n_property_l592_59223


namespace least_subtrahend_l592_59276

theorem least_subtrahend (n : ℕ) (h : n = 427398) : 
  ∃! x : ℕ, x ≤ n ∧ 
  (∀ y : ℕ, y < x → ¬((n - y) % 17 = 0 ∧ (n - y) % 19 = 0 ∧ (n - y) % 31 = 0)) ∧
  (n - x) % 17 = 0 ∧ (n - x) % 19 = 0 ∧ (n - x) % 31 = 0 :=
by sorry

end least_subtrahend_l592_59276


namespace experts_win_probability_value_l592_59214

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Audience winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def experts_score : ℕ := 3

/-- The current score of Audience -/
def audience_score : ℕ := 4

/-- The number of wins needed to win the game -/
def wins_needed : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
def experts_win_probability : ℝ := p^4 + 4 * p^3 * q

/-- Theorem stating that the probability of Experts winning is 0.4752 -/
theorem experts_win_probability_value : 
  experts_win_probability = 0.4752 := by sorry

end experts_win_probability_value_l592_59214


namespace class_enrollment_l592_59249

theorem class_enrollment (q1_correct q2_correct both_correct not_taken : ℕ) 
  (h1 : q1_correct = 25)
  (h2 : q2_correct = 22)
  (h3 : not_taken = 5)
  (h4 : both_correct = 22) :
  q1_correct + q2_correct - both_correct + not_taken = 30 :=
by
  sorry

end class_enrollment_l592_59249


namespace tournament_winner_percentage_l592_59224

theorem tournament_winner_percentage (n : ℕ) (total_games : ℕ) 
  (top_player_advantage : ℝ) (least_successful_percentage : ℝ) 
  (remaining_players_percentage : ℝ) :
  n = 8 →
  total_games = 560 →
  top_player_advantage = 0.15 →
  least_successful_percentage = 0.08 →
  remaining_players_percentage = 0.35 →
  ∃ (top_player_percentage : ℝ),
    top_player_percentage = 0.395 ∧
    top_player_percentage = 
      (1 - (2 * least_successful_percentage + remaining_players_percentage)) / 2 + 
      top_player_advantage :=
by sorry

end tournament_winner_percentage_l592_59224


namespace purely_imaginary_complex_number_l592_59294

theorem purely_imaginary_complex_number (a : ℝ) :
  (a^2 - 4 : ℂ) + (a - 2 : ℂ) * Complex.I = (0 : ℂ) + (b : ℂ) * Complex.I ∧ b ≠ 0 → a = -2 := by
  sorry

end purely_imaginary_complex_number_l592_59294


namespace equation_solution_l592_59227

theorem equation_solution :
  let f (x : ℝ) := (6*x + 3) / (3*x^2 + 6*x - 9) - 3*x / (3*x - 3)
  ∀ x : ℝ, x ≠ 1 → (f x = 0 ↔ x = (3 + Real.sqrt 21) / 2 ∨ x = (3 - Real.sqrt 21) / 2) :=
by sorry

end equation_solution_l592_59227


namespace factorial_305_trailing_zeros_l592_59237

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 305! ends with 75 zeros -/
theorem factorial_305_trailing_zeros :
  trailingZeros 305 = 75 := by
  sorry

end factorial_305_trailing_zeros_l592_59237


namespace radar_coverage_theorem_l592_59261

-- Define constants
def num_radars : ℕ := 8
def radar_radius : ℝ := 15
def ring_width : ℝ := 18

-- Define the theorem
theorem radar_coverage_theorem :
  let center_to_radar : ℝ := 12 / Real.sin (22.5 * π / 180)
  let ring_area : ℝ := 432 * π / Real.tan (22.5 * π / 180)
  (∀ (r : ℝ), r = center_to_radar →
    (num_radars : ℝ) * (2 * radar_radius - ring_width) = 2 * π * r * Real.sin (π / num_radars)) ∧
  (∀ (A : ℝ), A = ring_area →
    A = π * ((r + ring_width / 2)^2 - (r - ring_width / 2)^2)) :=
by sorry

end radar_coverage_theorem_l592_59261


namespace ear_muffs_december_l592_59255

theorem ear_muffs_december (before_december : ℕ) (total : ℕ) (during_december : ℕ) : 
  before_december = 1346 →
  total = 7790 →
  during_december = total - before_december →
  during_december = 6444 := by
sorry

end ear_muffs_december_l592_59255


namespace television_regular_price_l592_59239

theorem television_regular_price (sale_price : ℝ) (discount_rate : ℝ) (regular_price : ℝ) :
  sale_price = regular_price * (1 - discount_rate) →
  discount_rate = 0.2 →
  sale_price = 480 →
  regular_price = 600 := by
sorry

end television_regular_price_l592_59239


namespace total_fruits_is_fifteen_l592_59241

-- Define the three types of fruit
inductive FruitType
| A
| B
| C

-- Define a function that returns the quantity of each fruit type
def fruitQuantity (t : FruitType) : Nat :=
  match t with
  | FruitType.A => 5
  | FruitType.B => 6
  | FruitType.C => 4

-- Theorem: The total number of fruits is 15
theorem total_fruits_is_fifteen :
  (fruitQuantity FruitType.A) + (fruitQuantity FruitType.B) + (fruitQuantity FruitType.C) = 15 :=
by
  sorry

end total_fruits_is_fifteen_l592_59241


namespace simplify_2A_minus_3B_value_2A_minus_3B_l592_59299

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Theorem for the simplified form of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7 * x + 7 * y - 11 * x * y :=
sorry

-- Theorem for the value of 2A - 3B under given conditions
theorem value_2A_minus_3B :
  ∃ (x y : ℝ), x + y = -6/7 ∧ x * y = 1 ∧ 2 * A x y - 3 * B x y = -17 :=
sorry

end simplify_2A_minus_3B_value_2A_minus_3B_l592_59299


namespace ratio_of_w_to_y_l592_59229

theorem ratio_of_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 3 / 2)
  (hz_x : z / x = 1 / 3) :
  w / y = 8 / 3 := by
  sorry

end ratio_of_w_to_y_l592_59229


namespace fruit_punch_water_quarts_l592_59289

theorem fruit_punch_water_quarts 
  (water_parts juice_parts : ℕ) 
  (total_gallons : ℚ) 
  (quarts_per_gallon : ℕ) : 
  water_parts = 5 → 
  juice_parts = 2 → 
  total_gallons = 3 → 
  quarts_per_gallon = 4 → 
  (water_parts : ℚ) * total_gallons * quarts_per_gallon / (water_parts + juice_parts) = 60 / 7 := by
  sorry

end fruit_punch_water_quarts_l592_59289


namespace sine_function_monotonicity_l592_59260

theorem sine_function_monotonicity (ω : ℝ) (h1 : ω > 0) : 
  (∀ x ∈ Set.Icc (-π/3) (π/4), 
    ∀ y ∈ Set.Icc (-π/3) (π/4), 
    x < y → 2 * Real.sin (ω * x) < 2 * Real.sin (ω * y)) 
  → 0 < ω ∧ ω ≤ 3/2 := by
sorry

end sine_function_monotonicity_l592_59260


namespace distinct_values_of_triple_exponent_l592_59290

-- Define the base number
def base : ℕ := 3

-- Define the function to calculate the number of distinct values
def distinct_values (n : ℕ) : ℕ :=
  -- The actual implementation is not provided, as we're only writing the statement
  sorry

-- Theorem statement
theorem distinct_values_of_triple_exponent :
  distinct_values base = 2 :=
sorry

end distinct_values_of_triple_exponent_l592_59290


namespace sum_in_Q_l592_59275

-- Define the sets P, Q, and M
def P : Set Int := {x | ∃ k, x = 2 * k}
def Q : Set Int := {x | ∃ k, x = 2 * k - 1}
def M : Set Int := {x | ∃ k, x = 4 * k + 1}

-- Theorem statement
theorem sum_in_Q (a b : Int) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end sum_in_Q_l592_59275


namespace angle_phi_value_l592_59248

-- Define the problem statement
theorem angle_phi_value (φ : Real) (h1 : 0 < φ ∧ φ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin (20 * π / 180) = Real.cos φ - Real.sin φ) : 
  φ = 25 * π / 180 := by
  sorry

#check angle_phi_value

end angle_phi_value_l592_59248


namespace inequality_proof_l592_59212

theorem inequality_proof (n : ℕ) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end inequality_proof_l592_59212


namespace a_travel_time_l592_59252

-- Define the speed ratio of A to B
def speed_ratio : ℚ := 3 / 4

-- Define the time difference between A and B in hours
def time_difference : ℚ := 1 / 2

-- Theorem statement
theorem a_travel_time (t : ℚ) : 
  (t + time_difference) / t = 1 / speed_ratio → t + time_difference = 2 := by
  sorry

end a_travel_time_l592_59252


namespace tangent_circles_a_values_l592_59206

/-- Two circles are tangent if the distance between their centers is equal to
    the sum or difference of their radii -/
def are_tangent (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = (r1 + r2)^2) ∨
  (((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = (r1 - r2)^2)

theorem tangent_circles_a_values :
  ∀ a : ℝ,
  are_tangent (0, 0) (-4, a) 1 5 →
  (a = 0 ∨ a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5) :=
by sorry

end tangent_circles_a_values_l592_59206


namespace b_10_equals_64_l592_59277

/-- Given two sequences {aₙ} and {bₙ} satisfying certain conditions, prove that b₁₀ = 64 -/
theorem b_10_equals_64 (a b : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a n + a (n + 1) = b n)
  (h3 : ∀ n, a n * a (n + 1) = 2^n) :
  b 10 = 64 := by
  sorry

end b_10_equals_64_l592_59277


namespace training_effect_l592_59200

/-- Represents the scores and their frequencies in a test --/
structure TestScores :=
  (scores : List Nat)
  (frequencies : List Nat)

/-- Calculates the median of a list of scores --/
def median (scores : List Nat) : Nat :=
  sorry

/-- Calculates the mode of a list of scores --/
def mode (scores : List Nat) (frequencies : List Nat) : Nat :=
  sorry

/-- Calculates the average score --/
def average (scores : List Nat) (frequencies : List Nat) : Real :=
  sorry

/-- Calculates the number of students with scores greater than or equal to a threshold --/
def countExcellent (scores : List Nat) (frequencies : List Nat) (threshold : Nat) : Nat :=
  sorry

theorem training_effect (baselineScores simExamScores : TestScores)
  (totalStudents sampleSize : Nat)
  (hTotalStudents : totalStudents = 800)
  (hSampleSize : sampleSize = 50)
  (hBaselineScores : baselineScores = ⟨[6, 7, 8, 9, 10], [16, 8, 9, 9, 8]⟩)
  (hSimExamScores : simExamScores = ⟨[6, 7, 8, 9, 10], [5, 8, 6, 12, 19]⟩) :
  (median baselineScores.scores = 8 ∧ 
   mode simExamScores.scores simExamScores.frequencies = 10) ∧
  (average simExamScores.scores simExamScores.frequencies - 
   average baselineScores.scores baselineScores.frequencies = 0.94) ∧
  (totalStudents * (countExcellent simExamScores.scores simExamScores.frequencies 9) / sampleSize = 496) :=
by sorry

end training_effect_l592_59200


namespace units_digit_sum_powers_l592_59272

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_powers : units_digit ((35 ^ 7) + (93 ^ 45)) = 8 := by
  sorry

end units_digit_sum_powers_l592_59272


namespace intersection_P_Q_l592_59285

def P : Set ℝ := {x | x^2 - x = 0}
def Q : Set ℝ := {x | x^2 + x = 0}

theorem intersection_P_Q : P ∩ Q = {0} := by sorry

end intersection_P_Q_l592_59285


namespace baseball_card_value_decrease_l592_59288

theorem baseball_card_value_decrease (initial_value : ℝ) (h : initial_value > 0) :
  let value_after_first_year := initial_value * (1 - 0.2)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 28 := by sorry

end baseball_card_value_decrease_l592_59288


namespace sqrt_meaningful_range_l592_59270

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by sorry

end sqrt_meaningful_range_l592_59270


namespace continued_fraction_solution_l592_59226

theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) → x = (3 + Real.sqrt 69) / 2 :=
by sorry

end continued_fraction_solution_l592_59226


namespace olivias_wallet_problem_l592_59274

/-- Given an initial amount of 78 dollars and a spending of 15 dollars,
    the remaining amount is 63 dollars. -/
theorem olivias_wallet_problem (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 78 ∧ spent_amount = 15 → remaining_amount = initial_amount - spent_amount → remaining_amount = 63 := by
  sorry

end olivias_wallet_problem_l592_59274


namespace ab_value_l592_59273

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end ab_value_l592_59273


namespace line_slope_intercept_sum_l592_59242

/-- Given a line with slope 5 passing through (-2, 3), prove m + b = 18 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 5 → 
  3 = 5 * (-2) + b → 
  m + b = 18 := by
sorry

end line_slope_intercept_sum_l592_59242


namespace angle_420_equivalent_to_60_l592_59228

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem angle_420_equivalent_to_60 :
  same_terminal_side 420 60 :=
sorry

end angle_420_equivalent_to_60_l592_59228


namespace calculate_expression_l592_59278

theorem calculate_expression : (1 / 3 : ℚ) * 9 * 15 - 7 = 38 := by sorry

end calculate_expression_l592_59278


namespace ellipse_properties_l592_59217

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_eccentricity : (a^2 - b^2) / a^2 = 3/4
  h_point_on_ellipse : 2/a^2 + 1/(2*b^2) = 1

/-- The theorem statement -/
theorem ellipse_properties (C : Ellipse) :
  C.a^2 = 4 ∧ C.b^2 = 2 ∧
  (∀ (P Q : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    P ∈ l ∧ Q ∈ l ∧
    P.1^2/4 + P.2^2 = 1 ∧
    Q.1^2/4 + Q.2^2 = 1 ∧
    P.1 * Q.1 + P.2 * Q.2 = 0 →
    1/2 * abs (P.1 * Q.2 - P.2 * Q.1) ≥ 4/5) :=
by sorry

end ellipse_properties_l592_59217


namespace essays_total_pages_l592_59213

def words_per_page : ℕ := 235

def johnny_words : ℕ := 195
def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 50
def samantha_words : ℕ := 3 * madeline_words
def ryan_words : ℕ := johnny_words + 100

def pages_needed (words : ℕ) : ℕ :=
  (words + words_per_page - 1) / words_per_page

def total_pages : ℕ :=
  pages_needed johnny_words +
  pages_needed madeline_words +
  pages_needed timothy_words +
  pages_needed samantha_words +
  pages_needed ryan_words

theorem essays_total_pages : total_pages = 12 := by
  sorry

end essays_total_pages_l592_59213


namespace max_ratio_abcd_l592_59246

theorem max_ratio_abcd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d ≥ 0)
  (h5 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3/8) :
  (∀ x y z w, x ≥ y ∧ y ≥ z ∧ z ≥ w ∧ w ≥ 0 ∧ 
    (x^2 + y^2 + z^2 + w^2) / (x + y + z + w)^2 = 3/8 →
    (x + z) / (y + w) ≤ (a + c) / (b + d)) ∧
  (a + c) / (b + d) ≤ 3 :=
sorry

end max_ratio_abcd_l592_59246


namespace completing_square_equivalence_l592_59287

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
sorry

end completing_square_equivalence_l592_59287


namespace multiple_of_seven_square_gt_200_lt_30_l592_59293

theorem multiple_of_seven_square_gt_200_lt_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 7 * k)
  (h2 : x^2 > 200)
  (h3 : x < 30) :
  x = 21 ∨ x = 28 := by
sorry

end multiple_of_seven_square_gt_200_lt_30_l592_59293


namespace charles_vowel_learning_time_l592_59292

/-- The number of days Charles takes to learn one alphabet -/
def days_per_alphabet : ℕ := 7

/-- The number of vowels in the English alphabet -/
def number_of_vowels : ℕ := 5

/-- The total number of days Charles needs to finish learning all vowels -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem charles_vowel_learning_time : total_days = 35 := by
  sorry

end charles_vowel_learning_time_l592_59292


namespace quadratic_inequality_solution_l592_59286

theorem quadratic_inequality_solution (n : ℤ) : 
  n^2 - 13*n + 36 < 0 ↔ n ∈ ({5, 6, 7, 8} : Set ℤ) := by
  sorry

end quadratic_inequality_solution_l592_59286


namespace a_plus_b_value_l592_59221

theorem a_plus_b_value (a b : ℝ) (h1 : |a| = 3) (h2 : b^2 = 25) (h3 : a*b < 0) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end a_plus_b_value_l592_59221


namespace recipe_salt_amount_l592_59253

def recipe_salt (total_flour sugar flour_added : ℕ) : ℕ :=
  let remaining_flour := total_flour - flour_added
  remaining_flour - 3

theorem recipe_salt_amount :
  recipe_salt 12 14 2 = 7 :=
by
  sorry

end recipe_salt_amount_l592_59253


namespace larger_number_problem_l592_59247

theorem larger_number_problem (x y : ℝ) : 
  5 * y = 6 * x → y - x = 10 → y = 60 := by
  sorry

end larger_number_problem_l592_59247


namespace last_two_digits_of_factorial_sum_l592_59220

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_factorial_sum :
  sum_factorials 2003 % 100 = 13 := by sorry

end last_two_digits_of_factorial_sum_l592_59220


namespace gcd_91_49_l592_59266

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end gcd_91_49_l592_59266


namespace parabola_c_value_l592_59263

/-- A parabola passing through two points with equal y-coordinates -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_through_minus_one : 2 = 1 + (-b) + c
  pass_through_three : 2 = 9 + 3*b + c

/-- The value of c for a parabola passing through (-1, 2) and (3, 2) is -1 -/
theorem parabola_c_value (p : Parabola) : p.c = -1 := by
  sorry

end parabola_c_value_l592_59263


namespace spherical_to_rectangular_conversion_l592_59251

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 6
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (2 * Real.sqrt 6, Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end spherical_to_rectangular_conversion_l592_59251


namespace rectangle_to_equilateral_triangle_l592_59268

/-- Given a rectangle with length L and width W, and an equilateral triangle with side s,
    if both shapes have the same area A, then s = √(4LW/√3) -/
theorem rectangle_to_equilateral_triangle (L W s A : ℝ) (h1 : A = L * W) 
    (h2 : A = (s^2 * Real.sqrt 3) / 4) : s = Real.sqrt ((4 * L * W) / Real.sqrt 3) := by
  sorry

end rectangle_to_equilateral_triangle_l592_59268


namespace inequality_theorem_stronger_inequality_best_constant_l592_59219

theorem inequality_theorem (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) : 
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| ≥ 2 := by
  sorry

theorem stronger_inequality (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) 
  (pa : a ≥ 0) (pb : b ≥ 0) (pc : c ≥ 0) : 
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| > 3 := by
  sorry

theorem best_constant (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
  |(a + b) / (a - b)| + |(b + c) / (b - c)| + |(c + a) / (c - a)| < 3 + ε := by
  sorry

end inequality_theorem_stronger_inequality_best_constant_l592_59219


namespace power_of_power_l592_59262

theorem power_of_power (a : ℝ) : (a^3)^3 = a^9 := by
  sorry

end power_of_power_l592_59262


namespace function_increasing_interval_implies_b_bound_l592_59236

/-- Given a function f(x) = e^x(x^2 - bx) where b is a real number,
    if f(x) has an increasing interval in [1/2, 2],
    then b < 8/3 -/
theorem function_increasing_interval_implies_b_bound 
  (b : ℝ) 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = Real.exp x * (x^2 - b*x)) 
  (h_increasing : ∃ (a c : ℝ), 1/2 ≤ a ∧ c ≤ 2 ∧ StrictMonoOn f (Set.Icc a c)) : 
  b < 8/3 :=
sorry

end function_increasing_interval_implies_b_bound_l592_59236


namespace expression_evaluation_l592_59280

theorem expression_evaluation :
  let f (x : ℤ) := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)
  f (-2) = 6 := by sorry

end expression_evaluation_l592_59280


namespace quadratic_distinct_rational_roots_l592_59230

theorem quadratic_distinct_rational_roots (a b c : ℚ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hsum : a + b + c = 0) : 
  ∃ (x y : ℚ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end quadratic_distinct_rational_roots_l592_59230


namespace eight_person_handshakes_l592_59234

/-- The number of handshakes in a group where each person shakes hands with every other person once -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 8 people, where each person shakes hands exactly once with every other person, the total number of handshakes is 28 -/
theorem eight_person_handshakes : num_handshakes 8 = 28 := by
  sorry

end eight_person_handshakes_l592_59234


namespace letter_150_is_z_l592_59205

def repeating_sequence : ℕ → Char
  | n => if n % 3 = 1 then 'X' else if n % 3 = 2 then 'Y' else 'Z'

theorem letter_150_is_z : repeating_sequence 150 = 'Z' := by
  sorry

end letter_150_is_z_l592_59205


namespace circle_and_max_distance_l592_59291

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the ray 3x - y = 0 (x ≥ 0)
def Ray := {p : ℝ × ℝ | 3 * p.1 - p.2 = 0 ∧ p.1 ≥ 0}

-- Define the line x = 4
def TangentLine := {p : ℝ × ℝ | p.1 = 4}

-- Define the line 3x + 4y + 10 = 0
def ChordLine := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 10 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem circle_and_max_distance :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- Circle C's center is on the ray
    center ∈ Ray ∧
    -- Circle C is tangent to the line x = 4
    (∃ (p : ℝ × ℝ), p ∈ Circle center radius ∧ p ∈ TangentLine) ∧
    -- The chord intercepted by the line has length 4√3
    (∃ (p q : ℝ × ℝ), p ∈ Circle center radius ∧ q ∈ Circle center radius ∧
      p ∈ ChordLine ∧ q ∈ ChordLine ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = 48) ∧
    -- The equation of circle C is x^2 + y^2 = 16
    Circle center radius = {p : ℝ × ℝ | p.1^2 + p.2^2 = 16} ∧
    -- The maximum value of |PA|^2 + |PB|^2 is 38 + 8√2
    (∀ (p : ℝ × ℝ), p ∈ Circle center radius →
      (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 ≤ 38 + 8 * Real.sqrt 2) ∧
    (∃ (p : ℝ × ℝ), p ∈ Circle center radius ∧
      (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 = 38 + 8 * Real.sqrt 2) :=
by sorry

end circle_and_max_distance_l592_59291


namespace train_length_l592_59297

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 12 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 120 := by
  sorry

end train_length_l592_59297


namespace tan_585_degrees_l592_59210

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end tan_585_degrees_l592_59210


namespace quadratic_function_range_l592_59243

def f (x : ℝ) : ℝ := -x^2 + x + 2

theorem quadratic_function_range (a : ℝ) :
  (∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a + 3 → f x₁ < f x₂) ∧
  (∃ x, a ≤ x ∧ x ≤ a + 3 ∧ f x = -4) →
  -5 ≤ a ∧ a ≤ -5/2 :=
by sorry

end quadratic_function_range_l592_59243


namespace train_passing_platform_l592_59207

/-- Calculates the time for a train to pass a platform given its length, speed, and the platform length -/
theorem train_passing_platform 
  (train_length : Real) 
  (time_to_cross_tree : Real) 
  (platform_length : Real) : 
  train_length = 1200 ∧ 
  time_to_cross_tree = 120 ∧ 
  platform_length = 1000 → 
  (train_length + platform_length) / (train_length / time_to_cross_tree) = 220 := by
  sorry

end train_passing_platform_l592_59207


namespace only_odd_divisor_of_3_pow_n_plus_1_l592_59265

theorem only_odd_divisor_of_3_pow_n_plus_1 :
  ∀ n : ℕ, Odd n → (n ∣ 3^n + 1) → n = 1 := by
  sorry

end only_odd_divisor_of_3_pow_n_plus_1_l592_59265


namespace k_range_theorem_l592_59296

def f (x : ℝ) : ℝ := x * abs x

theorem k_range_theorem (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ici 1 ∧ f (x - 2*k) < k) → k ∈ Set.Ioi (1/4) :=
by sorry

end k_range_theorem_l592_59296


namespace product_divisible_by_49_l592_59215

theorem product_divisible_by_49 (a b : ℕ) (h : 7 ∣ (a^2 + b^2)) : 49 ∣ (a * b) := by
  sorry

end product_divisible_by_49_l592_59215


namespace f_properties_l592_59282

def f (x : ℝ) : ℝ := -(x - 2)^2 + 4

theorem f_properties :
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x : ℝ, f x ≤ 4) ∧
  (∃ x : ℝ, f x = 4) :=
by
  sorry

end f_properties_l592_59282


namespace obtuse_triangle_condition_l592_59267

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Add triangle inequality constraints
  hpos_a : 0 < a
  hpos_b : 0 < b
  hpos_c : 0 < c
  hab : a + b > c
  hbc : b + c > a
  hca : c + a > b

-- Define what it means for a triangle to be obtuse
def is_obtuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2

-- State the theorem
theorem obtuse_triangle_condition (t : Triangle) :
  (t.a^2 + t.b^2 < t.c^2 → is_obtuse t) ∧
  ∃ (t' : Triangle), is_obtuse t' ∧ t'.a^2 + t'.b^2 ≥ t'.c^2 :=
sorry

end obtuse_triangle_condition_l592_59267


namespace sequence_general_term_l592_59231

theorem sequence_general_term (p : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2 + p*n) →
  (∃ r, a 2 * r = a 5 ∧ a 5 * r = a 10) →
  ∃ k, ∀ n, a n = 2*n + k :=
by sorry

end sequence_general_term_l592_59231


namespace impossible_grid_2005_l592_59281

theorem impossible_grid_2005 : ¬ ∃ (a b c d e f g h i : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
   g ≠ h ∧ g ≠ i ∧
   h ≠ i) ∧
  (a * b * c = 2005) ∧ (d * e * f = 2005) ∧ (g * h * i = 2005) ∧
  (a * d * g = 2005) ∧ (b * e * h = 2005) ∧ (c * f * i = 2005) ∧
  (a * e * i = 2005) ∧ (c * e * g = 2005) :=
by sorry


end impossible_grid_2005_l592_59281


namespace ball_probability_l592_59216

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 60)
  (h2 : white = 22)
  (h3 : green = 18)
  (h4 : yellow = 2)
  (h5 : red = 15)
  (h6 : purple = 3)
  (h7 : total = white + green + yellow + red + purple) :
  (white + green + yellow : ℚ) / total = 7 / 10 := by
  sorry

end ball_probability_l592_59216


namespace checkerboard_diagonal_squares_l592_59202

theorem checkerboard_diagonal_squares (m n : ℕ) (hm : m = 91) (hn : n = 28) :
  m + n - Nat.gcd m n = 112 := by
  sorry

end checkerboard_diagonal_squares_l592_59202


namespace square_root_of_sixteen_l592_59201

theorem square_root_of_sixteen (x : ℝ) : x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end square_root_of_sixteen_l592_59201
