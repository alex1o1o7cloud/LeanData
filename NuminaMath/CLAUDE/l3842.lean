import Mathlib

namespace symmetry_condition_l3842_384250

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

theorem symmetry_condition (m : ℝ) :
  (∀ x, f m (2 - x) = f m x) ↔ m = -2 := by
  sorry

end symmetry_condition_l3842_384250


namespace stone_game_ratio_l3842_384204

/-- The stone game on a blackboard -/
def StoneGame (n : ℕ) : Prop :=
  n ≥ 3 →
  ∀ (s t : ℕ), s > 0 ∧ t > 0 →
  ∃ (q : ℚ), q ≥ 1 ∧ q < n - 1 ∧ (t : ℚ) / s = q

theorem stone_game_ratio (n : ℕ) (h : n ≥ 3) :
  StoneGame n :=
sorry

end stone_game_ratio_l3842_384204


namespace cubic_equation_value_l3842_384295

theorem cubic_equation_value (a b : ℝ) :
  (a * (-2)^3 + b * (-2) - 7 = 9) →
  (a * 2^3 + b * 2 - 7 = -23) :=
by sorry

end cubic_equation_value_l3842_384295


namespace compare_sqrt_l3842_384240

theorem compare_sqrt : -2 * Real.sqrt 11 > -3 * Real.sqrt 5 := by
  sorry

end compare_sqrt_l3842_384240


namespace quadratic_integer_roots_l3842_384224

theorem quadratic_integer_roots (m : ℝ) :
  (∃ x : ℤ, (m + 1) * x^2 + 2 * x - 5 * m - 13 = 0) ↔
  (m = -1 ∨ m = -11/10 ∨ m = -1/2) :=
sorry

end quadratic_integer_roots_l3842_384224


namespace remaining_pages_to_read_l3842_384287

/-- Given a book where 83 pages represent 1/3 of the total, 
    the number of remaining pages to read is 166. -/
theorem remaining_pages_to_read (total_pages : ℕ) 
  (h1 : 83 = total_pages / 3) : total_pages - 83 = 166 := by
  sorry

end remaining_pages_to_read_l3842_384287


namespace total_population_two_villages_l3842_384296

/-- The total population of two villages given partial information about each village's population -/
theorem total_population_two_villages
  (village1_90_percent : ℝ)
  (village2_80_percent : ℝ)
  (h1 : village1_90_percent = 45000)
  (h2 : village2_80_percent = 64000) :
  (village1_90_percent / 0.9 + village2_80_percent / 0.8) = 130000 :=
by sorry

end total_population_two_villages_l3842_384296


namespace stating_min_weighings_to_determine_faulty_coin_l3842_384234

/-- Represents a pile of coins with one faulty coin. -/
structure CoinPile :=
  (total : ℕ)  -- Total number of coins
  (faulty : ℕ)  -- Index of the faulty coin (1-based)
  (is_lighter : Bool)  -- True if the faulty coin is lighter, False if heavier

/-- Represents a weighing on a balance scale. -/
inductive Weighing
  | Equal : Weighing  -- The scale is balanced
  | Left : Weighing   -- The left side is heavier
  | Right : Weighing  -- The right side is heavier

/-- Function to perform a weighing on a subset of coins. -/
def weigh (pile : CoinPile) (left : List ℕ) (right : List ℕ) : Weighing :=
  sorry  -- Implementation details omitted

/-- 
Theorem stating that the minimum number of weighings required to determine 
whether the faulty coin is lighter or heavier is 2.
-/
theorem min_weighings_to_determine_faulty_coin (pile : CoinPile) : 
  ∃ (strategy : List (List ℕ × List ℕ)), 
    (strategy.length = 2) ∧ 
    (∀ (outcome : List Weighing), 
      outcome.length = 2 → 
      (∃ (result : Bool), result = pile.is_lighter)) :=
sorry

end stating_min_weighings_to_determine_faulty_coin_l3842_384234


namespace set_operations_l3842_384213

-- Define the universal set U
def U : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- Define set A
def A : Set ℤ := {x : ℤ | x^2 - 5*x - 6 = 0}

-- Define set B
def B : Set ℤ := {x : ℤ | x^2 = 1}

-- Theorem statement
theorem set_operations :
  (A ∪ B = {-1, 1, 6}) ∧
  (A ∩ B = {-1}) ∧
  (U \ (A ∩ B) = {0, 1}) := by
  sorry

end set_operations_l3842_384213


namespace mixture_weight_l3842_384217

/-- Given a mixture of substances a and b in the ratio 9:11, where 26.1 kg of a is used,
    prove that the total weight of the mixture is 58 kg. -/
theorem mixture_weight (a b : ℝ) (h1 : a / b = 9 / 11) (h2 : a = 26.1) :
  a + b = 58 := by sorry

end mixture_weight_l3842_384217


namespace imaginary_part_of_z_l3842_384246

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + 2*i) / (i - 1)
  Complex.im z = -3/2 := by
sorry

end imaginary_part_of_z_l3842_384246


namespace divisibility_rule_37_l3842_384260

-- Define a function to compute the sum of three-digit groups
def sumOfGroups (n : ℕ) : ℕ := sorry

-- State the theorem
theorem divisibility_rule_37 (n : ℕ) :
  37 ∣ n ↔ 37 ∣ sumOfGroups n := by sorry

end divisibility_rule_37_l3842_384260


namespace polynomial_divisibility_l3842_384288

theorem polynomial_divisibility (n : ℕ) :
  ∃ q : Polynomial ℤ, (X + 1 : Polynomial ℤ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end polynomial_divisibility_l3842_384288


namespace probability_of_white_ball_l3842_384261

-- Define the set of possible initial compositions
inductive InitialComposition
| NoWhite
| OneWhite
| TwoWhite

-- Define the probability of drawing a white ball given an initial composition
def probWhiteGivenComposition (ic : InitialComposition) : ℚ :=
  match ic with
  | InitialComposition.NoWhite => 1/3
  | InitialComposition.OneWhite => 2/3
  | InitialComposition.TwoWhite => 1

-- Define the theorem
theorem probability_of_white_ball :
  let initialCompositions := [InitialComposition.NoWhite, InitialComposition.OneWhite, InitialComposition.TwoWhite]
  let numCompositions := initialCompositions.length
  let probEachComposition := 1 / numCompositions
  let totalProb := (initialCompositions.map probWhiteGivenComposition).sum * probEachComposition
  totalProb = 2/3 := by
  sorry

end probability_of_white_ball_l3842_384261


namespace crups_are_arogs_and_brafs_l3842_384203

-- Define the types for our sets
variable (U : Type) -- Universe set
variable (Arog Braf Crup Dramp : Set U)

-- Define the given conditions
variable (h1 : Arog ⊆ Braf)
variable (h2 : Crup ⊆ Braf)
variable (h3 : Arog ⊆ Dramp)
variable (h4 : Crup ⊆ Dramp)

-- Theorem to prove
theorem crups_are_arogs_and_brafs : Crup ⊆ Arog ∩ Braf :=
sorry

end crups_are_arogs_and_brafs_l3842_384203


namespace triangle_external_angle_l3842_384239

theorem triangle_external_angle (a b c x : ℝ) : 
  a = 50 → b = 40 → c = 90 → a + b + c = 180 → 
  x + 45 = 180 → x = 135 := by
  sorry

end triangle_external_angle_l3842_384239


namespace corn_yield_ratio_l3842_384268

/-- Represents the corn yield ratio problem --/
theorem corn_yield_ratio :
  let johnson_hectares : ℕ := 1
  let johnson_yield_per_2months : ℕ := 80
  let neighbor_hectares : ℕ := 2
  let total_months : ℕ := 6
  let total_yield : ℕ := 1200
  let neighbor_yield_ratio : ℚ := 
    (total_yield - johnson_yield_per_2months * (total_months / 2) * johnson_hectares) /
    (johnson_yield_per_2months * (total_months / 2) * neighbor_hectares)
  neighbor_yield_ratio = 2
  := by sorry

end corn_yield_ratio_l3842_384268


namespace women_in_second_group_l3842_384208

/-- Represents the work rate of a man -/
def man_rate : ℝ := sorry

/-- Represents the work rate of a woman -/
def woman_rate : ℝ := sorry

/-- The number of women in the second group -/
def x : ℝ := sorry

/-- First condition: 3 men and 8 women complete a task in the same time as 6 men and x women -/
axiom condition1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + x * woman_rate

/-- Second condition: 2 men and 3 women complete half the work in the same time as the first group -/
axiom condition2 : 2 * man_rate + 3 * woman_rate = 0.5 * (3 * man_rate + 8 * woman_rate)

/-- The theorem to be proved -/
theorem women_in_second_group : x = 2 := by sorry

end women_in_second_group_l3842_384208


namespace a2_value_l3842_384256

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b / a = c / b

theorem a2_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 :=
by
  sorry

end a2_value_l3842_384256


namespace simplify_expression_l3842_384299

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := by
  sorry

end simplify_expression_l3842_384299


namespace max_value_sqrt_sum_l3842_384274

theorem max_value_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 9) :
  Real.sqrt (x + 15) + Real.sqrt (9 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 143 := by
  sorry

end max_value_sqrt_sum_l3842_384274


namespace friends_team_assignment_l3842_384223

theorem friends_team_assignment (n : ℕ) (k : ℕ) :
  n = 8 → k = 4 → (k : ℕ) ^ n = 65536 := by
  sorry

end friends_team_assignment_l3842_384223


namespace sugar_solution_percentage_l3842_384210

/-- Proves that a replacing solution must be 40% sugar by weight given the conditions of the problem. -/
theorem sugar_solution_percentage (original_percentage : ℝ) (replaced_fraction : ℝ) (final_percentage : ℝ) :
  original_percentage = 8 →
  replaced_fraction = 1 / 4 →
  final_percentage = 16 →
  (1 - replaced_fraction) * original_percentage + replaced_fraction * (100 : ℝ) * final_percentage / 100 = 40 := by
  sorry

end sugar_solution_percentage_l3842_384210


namespace yellow_not_more_than_green_l3842_384237

/-- Represents the three types of parrots -/
inductive ParrotType
  | Green
  | Yellow
  | Mottled

/-- Represents whether a parrot tells the truth or lies -/
inductive ParrotResponse
  | Truth
  | Lie

/-- The total number of parrots -/
def totalParrots : Nat := 100

/-- The number of parrots that agreed with each statement -/
def agreeingParrots : Nat := 50

/-- Function that determines how a parrot responds based on its type -/
def parrotBehavior (t : ParrotType) (statement : Nat) : ParrotResponse :=
  match t with
  | ParrotType.Green => ParrotResponse.Truth
  | ParrotType.Yellow => ParrotResponse.Lie
  | ParrotType.Mottled => if statement == 1 then ParrotResponse.Truth else ParrotResponse.Lie

/-- Theorem stating that the number of yellow parrots cannot exceed the number of green parrots -/
theorem yellow_not_more_than_green 
  (G Y M : Nat) 
  (h_total : G + Y + M = totalParrots)
  (h_first_statement : G + M / 2 = agreeingParrots)
  (h_second_statement : M / 2 + Y = agreeingParrots) :
  Y ≤ G :=
sorry

end yellow_not_more_than_green_l3842_384237


namespace bonus_allocation_l3842_384265

theorem bonus_allocation (bonus : ℚ) (kitchen_fraction : ℚ) (christmas_fraction : ℚ) (leftover : ℚ) 
  (h1 : bonus = 1496)
  (h2 : kitchen_fraction = 1 / 22)
  (h3 : christmas_fraction = 1 / 8)
  (h4 : leftover = 867)
  (h5 : bonus * kitchen_fraction + bonus * christmas_fraction + bonus * (holiday_fraction : ℚ) + leftover = bonus) :
  holiday_fraction = 187 / 748 := by
  sorry

end bonus_allocation_l3842_384265


namespace total_bowling_balls_l3842_384270

theorem total_bowling_balls (red : ℕ) (green : ℕ) (blue : ℕ) : 
  red = 30 →
  green = red + 6 →
  blue = 2 * green →
  red + green + blue = 138 := by
sorry

end total_bowling_balls_l3842_384270


namespace hyperbola_parabola_intersection_l3842_384225

/-- The value of p for which the left focus of the hyperbola x²/3 - y² = 1
    is on the directrix of the parabola y² = 2px -/
theorem hyperbola_parabola_intersection (p : ℝ) : p = 4 := by
  -- Define the hyperbola equation
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 3 - y^2 = 1
  -- Define the parabola equation
  let parabola := fun (x y : ℝ) ↦ y^2 = 2 * p * x
  -- Define the condition that the left focus of the hyperbola is on the directrix of the parabola
  let focus_on_directrix := ∃ (x y : ℝ), hyperbola x y ∧ parabola x y
  sorry

end hyperbola_parabola_intersection_l3842_384225


namespace rectangle_perimeter_l3842_384272

theorem rectangle_perimeter (a b : ℝ) (h1 : a + b = 7) (h2 : 2 * a + b = 9.5) :
  2 * (a + b) = 14 := by
  sorry

end rectangle_perimeter_l3842_384272


namespace max_pairs_sum_l3842_384281

theorem max_pairs_sum (k : ℕ) 
  (a b : Fin k → ℕ) 
  (h1 : ∀ i : Fin k, a i < b i)
  (h2 : ∀ i : Fin k, a i ≤ 1500 ∧ b i ≤ 1500)
  (h3 : ∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)
  (h4 : ∀ i : Fin k, a i + b i ≤ 1500)
  (h5 : ∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) :
  k ≤ 599 :=
sorry

end max_pairs_sum_l3842_384281


namespace largest_prime_factor_of_1729_l3842_384285

theorem largest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_1729_l3842_384285


namespace pen_pencil_ratio_l3842_384206

theorem pen_pencil_ratio : 
  ∀ (num_pencils num_pens : ℕ),
  num_pencils = 24 →
  num_pencils = num_pens + 4 →
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 :=
by
  sorry

end pen_pencil_ratio_l3842_384206


namespace x_value_and_n_bound_l3842_384251

theorem x_value_and_n_bound (x n : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : 0 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 5)
  (h4 : 0 < x ∧ x < 3)
  (h5 : x + n < 4) : 
  x = 1 ∧ n < 3 := by
  sorry

end x_value_and_n_bound_l3842_384251


namespace gain_percentage_is_twenty_percent_l3842_384214

def selling_price : ℝ := 180
def gain : ℝ := 30

theorem gain_percentage_is_twenty_percent : 
  (gain / (selling_price - gain)) * 100 = 20 := by
  sorry

end gain_percentage_is_twenty_percent_l3842_384214


namespace shooter_probability_l3842_384290

theorem shooter_probability (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) :
  1 - p_10 - p_9 = 0.48 := by
  sorry

end shooter_probability_l3842_384290


namespace min_value_of_expression_l3842_384245

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) passing through the center of the circle x² + y² + 4x - 4y - 1 = 0,
    the minimum value of 2/a + 3/b is 5 + 2√6 -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : ∃ (x y : ℝ), a * x - b * y + 2 = 0)
    (h_circle : ∃ (x y : ℝ), x^2 + y^2 + 4*x - 4*y - 1 = 0)
    (h_center : ∃ (x y : ℝ), (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (a * x - b * y + 2 = 0)) :
    (∀ (a' b' : ℝ), (a' > 0 ∧ b' > 0) → (2/a' + 3/b' ≥ 5 + 2 * Real.sqrt 6)) ∧
    (∃ (a' b' : ℝ), (a' > 0 ∧ b' > 0) ∧ (2/a' + 3/b' = 5 + 2 * Real.sqrt 6)) := by
  sorry

end min_value_of_expression_l3842_384245


namespace fifth_roots_of_unity_l3842_384205

theorem fifth_roots_of_unity (p q r s t m : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  m^5 = 1 :=
sorry

end fifth_roots_of_unity_l3842_384205


namespace line_through_intersection_and_perpendicular_l3842_384294

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0
def l₃ (x y : ℝ) : Prop := 3*x - y + 1 = 0

-- Define the line l (the answer)
def l (x y : ℝ) : Prop := x + 3*y - 7 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_through_intersection_and_perpendicular :
  (l₁ M.1 M.2 ∧ l₂ M.1 M.2) ∧  -- M is the intersection of l₁ and l₂
  (∀ x y : ℝ, l x y → l₃ x y → (x - M.1) * 3 + (y - M.2) * (-1) = 0) ∧  -- l is perpendicular to l₃
  l M.1 M.2  -- l passes through M
  := by sorry

end line_through_intersection_and_perpendicular_l3842_384294


namespace emilys_marbles_l3842_384269

theorem emilys_marbles (jake_marbles : ℕ) (emily_scale : ℕ) : 
  jake_marbles = 216 → 
  emily_scale = 3 → 
  (emily_scale ^ 3) * jake_marbles = 5832 :=
by sorry

end emilys_marbles_l3842_384269


namespace final_marble_count_l3842_384222

def initial_marbles : ℝ := 87.0
def received_marbles : ℝ := 8.0

theorem final_marble_count :
  initial_marbles + received_marbles = 95.0 := by sorry

end final_marble_count_l3842_384222


namespace negative_square_cubed_l3842_384236

theorem negative_square_cubed (m : ℝ) : (-m^2)^3 = -m^6 := by
  sorry

end negative_square_cubed_l3842_384236


namespace chad_age_l3842_384207

theorem chad_age (diana fabian eduardo chad : ℕ) 
  (h1 : diana = fabian - 5)
  (h2 : fabian = eduardo + 2)
  (h3 : chad = eduardo + 3)
  (h4 : diana = 15) : 
  chad = 21 := by
  sorry

end chad_age_l3842_384207


namespace sum_of_odd_divisors_90_l3842_384284

/-- The sum of the positive odd divisors of 90 -/
def sumOfOddDivisors90 : ℕ := sorry

/-- Theorem stating that the sum of the positive odd divisors of 90 is 78 -/
theorem sum_of_odd_divisors_90 : sumOfOddDivisors90 = 78 := by sorry

end sum_of_odd_divisors_90_l3842_384284


namespace remaining_problems_to_grade_l3842_384297

theorem remaining_problems_to_grade
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (problems_per_worksheet : ℕ)
  (h1 : total_worksheets = 17)
  (h2 : graded_worksheets = 8)
  (h3 : problems_per_worksheet = 7)
  : (total_worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end remaining_problems_to_grade_l3842_384297


namespace sum_of_x_solutions_is_zero_l3842_384220

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 169 ∧ x2^2 + y^2 = 169 ∧ x1 + x2 = 0 := by
sorry

end sum_of_x_solutions_is_zero_l3842_384220


namespace square_feet_per_acre_l3842_384262

/-- Represents the area of a rectangle in square feet -/
def rectangle_area (length width : ℝ) : ℝ := length * width

/-- Represents the total number of acres rented -/
def total_acres : ℝ := 10

/-- Represents the monthly rent for the entire plot -/
def total_rent : ℝ := 300

/-- Represents the length of the rectangular plot in feet -/
def plot_length : ℝ := 360

/-- Represents the width of the rectangular plot in feet -/
def plot_width : ℝ := 1210

theorem square_feet_per_acre :
  (rectangle_area plot_length plot_width) / total_acres = 43560 := by
  sorry

#check square_feet_per_acre

end square_feet_per_acre_l3842_384262


namespace smallest_n_proof_l3842_384215

def has_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ d = 7 ∧ ∃ k m : ℕ, n = k * 10 + d + m * 100

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def smallest_n_with_properties : ℕ := 65536

theorem smallest_n_proof :
  (is_terminating_decimal smallest_n_with_properties) ∧
  (has_digit_seven smallest_n_with_properties) ∧
  (∀ m : ℕ, m < smallest_n_with_properties →
    ¬(is_terminating_decimal m ∧ has_digit_seven m)) :=
by sorry

end smallest_n_proof_l3842_384215


namespace kareem_son_age_ratio_l3842_384298

/-- Proves that the ratio of Kareem's age to his son's age is 3:1 --/
theorem kareem_son_age_ratio :
  let kareem_age : ℕ := 42
  let son_age : ℕ := 14
  let future_sum : ℕ := 76
  let future_years : ℕ := 10
  (kareem_age + future_years) + (son_age + future_years) = future_sum →
  (kareem_age : ℚ) / son_age = 3 / 1 := by
sorry

end kareem_son_age_ratio_l3842_384298


namespace parabola_translation_theorem_l3842_384226

/-- Represents a parabola of the form y = ax^2 + bx + 1 -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a parabola passes through a given point -/
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + 1

/-- Represents the parabola after translation along x-axis -/
def translate (p : Parabola) (m : ℝ) (x : ℝ) : ℝ :=
  p.a * (x - m)^2 + p.b * (x - m) + 1

theorem parabola_translation_theorem (p : Parabola) (m : ℝ) :
  passes_through p 1 (-2) ∧ passes_through p (-2) 13 ∧ m > 0 →
  (∀ x, -1 ≤ x ∧ x ≤ 3 → translate p m x ≥ 6) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ translate p m x = 6) ↔
  m = 6 ∨ m = 4 := by
  sorry

end parabola_translation_theorem_l3842_384226


namespace cubic_sum_theorem_l3842_384293

theorem cubic_sum_theorem (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 2)
  (eq2 : b^2 + 5*c = 3)
  (eq3 : c^2 + 7*a = 6) :
  a^3 + b^3 + c^3 = -0.875 := by
  sorry

end cubic_sum_theorem_l3842_384293


namespace rectangle_area_l3842_384200

/-- A rectangle with a diagonal of 17 cm and a perimeter of 46 cm has an area of 120 cm². -/
theorem rectangle_area (l w : ℝ) : 
  l > 0 → w > 0 → l^2 + w^2 = 17^2 → 2*l + 2*w = 46 → l * w = 120 :=
by sorry

end rectangle_area_l3842_384200


namespace hyperbola_asymptote_angle_l3842_384218

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the angle between asymptotes
def angle_between_asymptotes (h : (x y : ℝ) → Prop) : ℝ := sorry

-- Theorem statement
theorem hyperbola_asymptote_angle :
  angle_between_asymptotes hyperbola = 60 * π / 180 := by sorry

end hyperbola_asymptote_angle_l3842_384218


namespace not_necessary_not_sufficient_neither_necessary_nor_sufficient_l3842_384292

/-- Two lines are parallel if they have the same slope and don't intersect. -/
def are_parallel (m : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
  ∀ (x y : ℝ), (m * x + 4 * y - 6 = 0) ↔ (k * (x + m * y - 3) = 0)

/-- m = 2 is not necessary for the lines to be parallel. -/
theorem not_necessary (m : ℝ) : 
  ∃ m', m' ≠ 2 ∧ are_parallel m' :=
sorry

/-- m = 2 is not sufficient for the lines to be parallel. -/
theorem not_sufficient : ¬(are_parallel 2) :=
sorry

/-- m = 2 is neither necessary nor sufficient for the lines to be parallel. -/
theorem neither_necessary_nor_sufficient : 
  (∃ m', m' ≠ 2 ∧ are_parallel m') ∧ ¬(are_parallel 2) :=
sorry

end not_necessary_not_sufficient_neither_necessary_nor_sufficient_l3842_384292


namespace find_number_l3842_384232

theorem find_number : ∃ x : ℝ, 13 * x - 272 = 105 ∧ x = 29 := by
  sorry

end find_number_l3842_384232


namespace opposite_sign_sum_l3842_384212

theorem opposite_sign_sum (x y : ℝ) : (x + 3)^2 + |y - 2| = 0 → (x + y)^y = 1 := by
  sorry

end opposite_sign_sum_l3842_384212


namespace square_sum_xy_l3842_384201

theorem square_sum_xy (x y a c : ℝ) (h1 : x * y = a) (h2 : 1 / x^2 + 1 / y^2 = c) :
  (x + y)^2 = a * c^2 + 2 * a := by
  sorry

end square_sum_xy_l3842_384201


namespace games_to_reach_target_win_rate_l3842_384209

def initial_games : ℕ := 20
def initial_win_rate : ℚ := 95 / 100
def target_win_rate : ℚ := 96 / 100

theorem games_to_reach_target_win_rate :
  let initial_wins := (initial_games : ℚ) * initial_win_rate
  ∃ (additional_games : ℕ),
    (initial_wins + additional_games) / (initial_games + additional_games) = target_win_rate ∧
    additional_games = 5 := by
  sorry

end games_to_reach_target_win_rate_l3842_384209


namespace min_amount_for_house_l3842_384259

/-- Calculates the minimum amount needed to buy a house given the original price,
    full payment discount percentage, and deed tax percentage. -/
def min_house_purchase_amount (original_price : ℕ) (discount_percent : ℚ) (deed_tax_percent : ℚ) : ℕ :=
  let discounted_price := (original_price : ℚ) * discount_percent
  let deed_tax := discounted_price * deed_tax_percent
  (discounted_price + deed_tax).ceil.toNat

/-- Proves that the minimum amount needed to buy the house is 311,808 yuan. -/
theorem min_amount_for_house :
  min_house_purchase_amount 320000 (96 / 100) (3 / 200) = 311808 := by
  sorry

#eval min_house_purchase_amount 320000 (96 / 100) (3 / 200)

end min_amount_for_house_l3842_384259


namespace parallel_lines_iff_a_eq_one_l3842_384231

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ : ℝ) (m₂ n₂ : ℝ) : Prop := m₁ * n₂ = m₂ * n₁

/-- The statement that a = 1 is necessary and sufficient for the lines to be parallel -/
theorem parallel_lines_iff_a_eq_one :
  ∀ a : ℝ, are_parallel a 1 3 (a + 2) ↔ a = 1 := by sorry

end parallel_lines_iff_a_eq_one_l3842_384231


namespace choose_four_from_ten_l3842_384291

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end choose_four_from_ten_l3842_384291


namespace partial_fraction_decomposition_l3842_384252

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ),
    (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 →
      5 * x^2 / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 20 ∧ Q = -15 ∧ R = -10 := by
  sorry

end partial_fraction_decomposition_l3842_384252


namespace cube_edge_length_l3842_384211

theorem cube_edge_length (a : ℝ) :
  (6 * a^2 = a^3 → a = 6) ∧
  (6 * a^2 = (a^3)^2 → a = Real.rpow 6 (1/4)) ∧
  ((6 * a^2)^3 = a^3 → a = 1/36) :=
by sorry

end cube_edge_length_l3842_384211


namespace height_on_side_BC_l3842_384277

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = √3, b = √2, and 1 + 2cos(B+C) = 0, 
    prove that the height h on side BC is equal to (√3 + 1) / 2. -/
theorem height_on_side_BC (A B C : ℝ) (a b c : ℝ) (h : ℝ) : 
  a = Real.sqrt 3 → 
  b = Real.sqrt 2 → 
  1 + 2 * Real.cos (B + C) = 0 → 
  h = (Real.sqrt 3 + 1) / 2 := by
sorry

end height_on_side_BC_l3842_384277


namespace remainder_23_pow_2047_mod_17_l3842_384282

theorem remainder_23_pow_2047_mod_17 : 23^2047 % 17 = 11 := by
  sorry

end remainder_23_pow_2047_mod_17_l3842_384282


namespace tuesday_books_brought_back_l3842_384243

/-- Calculates the number of books brought back on Tuesday given the initial number of books,
    the number of books taken out on Monday, and the final number of books on Tuesday. -/
def books_brought_back (initial : ℕ) (taken_out : ℕ) (final : ℕ) : ℕ :=
  final - (initial - taken_out)

/-- Theorem stating that 22 books were brought back on Tuesday given the specified conditions. -/
theorem tuesday_books_brought_back :
  books_brought_back 336 124 234 = 22 := by
  sorry

end tuesday_books_brought_back_l3842_384243


namespace no_solution_iff_k_eq_six_l3842_384230

theorem no_solution_iff_k_eq_six (k : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 7 → (x - 1) / (x - 2) ≠ (x - k) / (x - 7)) ↔ k = 6 :=
by sorry

end no_solution_iff_k_eq_six_l3842_384230


namespace replaced_person_weight_l3842_384255

/-- Proves that the weight of the replaced person is 35 kg given the conditions -/
theorem replaced_person_weight (initial_count : Nat) (weight_increase : Real) (new_person_weight : Real) :
  initial_count = 8 ∧ 
  weight_increase = 2.5 ∧ 
  new_person_weight = 55 →
  (initial_count * weight_increase = new_person_weight - (initial_count * weight_increase - new_person_weight)) :=
by
  sorry

#check replaced_person_weight

end replaced_person_weight_l3842_384255


namespace dalmatians_with_right_ear_spot_l3842_384238

theorem dalmatians_with_right_ear_spot (total : ℕ) (left_only : ℕ) (right_only : ℕ) (no_spots : ℕ) :
  total = 101 →
  left_only = 29 →
  right_only = 17 →
  no_spots = 22 →
  total - no_spots - left_only = 50 :=
by sorry

end dalmatians_with_right_ear_spot_l3842_384238


namespace rectangle_area_l3842_384221

/-- Given three similar rectangles where ABCD is the largest, prove its area --/
theorem rectangle_area (width height : ℝ) (h1 : width = 15) (h2 : height = width * Real.sqrt 6) :
  width * height = 75 * Real.sqrt 6 := by
  sorry

end rectangle_area_l3842_384221


namespace prop_logic_evaluation_l3842_384219

theorem prop_logic_evaluation (p q : Prop) (hp : p ↔ (2 < 3)) (hq : q ↔ (2 > 3)) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end prop_logic_evaluation_l3842_384219


namespace sheet_area_difference_l3842_384286

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem sheet_area_difference :
  areaDifference 11 17 8.5 11 = 187 := by
  sorry

end sheet_area_difference_l3842_384286


namespace ellipse_and_line_intersection_perpendicular_intersection_l3842_384242

-- Define the line l
def line_l (x : ℝ) : ℝ := -x + 3

-- Define the ellipse C
def ellipse_C (m n x y : ℝ) : Prop := m * x^2 + n * y^2 = 1

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define the line l'
def line_l' (b x : ℝ) : ℝ := -x + b

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_and_line_intersection :
  ∀ m n : ℝ, n > m → m > 0 →
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 1 ∧ line_l p.1 = p.2 ∧ ellipse_C m n p.1 p.2) →
  (∀ x y : ℝ, ellipse_C m n x y ↔ standard_ellipse x y) :=
sorry

theorem perpendicular_intersection :
  ∀ b : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    standard_ellipse A.1 A.2 ∧ standard_ellipse B.1 B.2 ∧
    line_l' b A.1 = A.2 ∧ line_l' b B.1 = B.2 ∧
    perpendicular A.1 A.2 B.1 B.2) →
  b = 2 :=
sorry

end ellipse_and_line_intersection_perpendicular_intersection_l3842_384242


namespace intersection_points_l3842_384228

-- Define the slopes and y-intercepts of the lines
def m₁ : ℚ := 3
def b₁ : ℚ := -4
def b₃ : ℚ := -3

-- Define Point A
def A : ℚ × ℚ := (3, 2)

-- Define the perpendicular slope
def m₂ : ℚ := -1 / m₁

-- Define the equations of the lines
def line1 (x : ℚ) : ℚ := m₁ * x + b₁
def line2 (x : ℚ) : ℚ := m₂ * (x - A.1) + A.2
def line3 (x : ℚ) : ℚ := m₁ * x + b₃

-- State the theorem
theorem intersection_points :
  ∃ (P Q : ℚ × ℚ),
    (P.1 = 21/10 ∧ P.2 = 23/10 ∧ line1 P.1 = line2 P.1) ∧
    (Q.1 = 9/5 ∧ Q.2 = 12/5 ∧ line2 Q.1 = line3 Q.1) :=
by sorry

end intersection_points_l3842_384228


namespace circle_equation_l3842_384264

/-- A circle with center on the line x + y = 0 and intersecting the x-axis at (-3, 0) and (1, 0) -/
structure Circle where
  center : ℝ × ℝ
  center_on_line : center.1 + center.2 = 0
  intersects_x_axis : ∃ (t : ℝ), t^2 = (center.1 + 3)^2 + center.2^2 ∧ t^2 = (center.1 - 1)^2 + center.2^2

/-- The equation of the circle is (x+1)² + (y-1)² = 5 -/
theorem circle_equation (c : Circle) : 
  ∀ (x y : ℝ), (x + 1)^2 + (y - 1)^2 = 5 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = ((c.center.1 + 3)^2 + c.center.2^2) :=
by sorry

end circle_equation_l3842_384264


namespace intersection_of_three_lines_l3842_384276

/-- 
Given three lines that intersect at the same point:
1. y = 2x + 7
2. y = -3x - 6
3. y = 4x + m
Prove that m = 61/5
-/
theorem intersection_of_three_lines (x y m : ℝ) : 
  (y = 2*x + 7) ∧ 
  (y = -3*x - 6) ∧ 
  (y = 4*x + m) → 
  m = 61/5 := by
sorry

end intersection_of_three_lines_l3842_384276


namespace symmetric_points_line_equation_l3842_384257

/-- Given two points A and B that are symmetric with respect to a line l,
    prove that the equation of line l is 3x + y + 4 = 0 --/
theorem symmetric_points_line_equation (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  A = (1, 3) →
  B = (-5, 1) →
  (∀ (P : ℝ × ℝ), P ∈ l ↔ dist P A = dist P B) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3 * x + y + 4 = 0) :=
by sorry


end symmetric_points_line_equation_l3842_384257


namespace bus_distance_problem_l3842_384249

/-- Proves that given a total distance of 250 km, covered partly at 40 kmph and partly at 60 kmph,
    with a total travel time of 6 hours, the distance covered at 40 kmph is 220 km. -/
theorem bus_distance_problem (x : ℝ) 
    (h1 : x ≥ 0) 
    (h2 : x ≤ 250) 
    (h3 : x / 40 + (250 - x) / 60 = 6) : x = 220 := by
  sorry

#check bus_distance_problem

end bus_distance_problem_l3842_384249


namespace cards_given_by_jeff_l3842_384244

theorem cards_given_by_jeff (initial_cards final_cards : ℝ) 
  (h1 : initial_cards = 304.0)
  (h2 : final_cards = 580) :
  final_cards - initial_cards = 276 := by
  sorry

end cards_given_by_jeff_l3842_384244


namespace shaded_area_calculation_l3842_384278

/-- Represents a rectangle with its diagonal divided into 12 equal segments -/
structure DividedRectangle where
  totalSegments : ℕ
  nonShadedArea : ℝ

/-- Calculates the area of shaded parts in a divided rectangle -/
def shadedArea (rect : DividedRectangle) : ℝ :=
  sorry

/-- Theorem stating the relationship between non-shaded and shaded areas -/
theorem shaded_area_calculation (rect : DividedRectangle) 
  (h1 : rect.totalSegments = 12)
  (h2 : rect.nonShadedArea = 10) :
  shadedArea rect = 14 := by
  sorry

end shaded_area_calculation_l3842_384278


namespace fixed_point_of_exponential_function_l3842_384280

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 1
  f 2 = 2 := by
sorry

end fixed_point_of_exponential_function_l3842_384280


namespace smallest_reducible_fraction_l3842_384227

def is_reducible (n d : ℤ) : Prop := ∃ k : ℤ, k > 1 ∧ k ∣ n ∧ k ∣ d

theorem smallest_reducible_fraction :
  ∀ m : ℕ, m > 0 →
    (m < 30 → ¬(is_reducible (m - 17) (7 * m + 11))) ∧
    (is_reducible (30 - 17) (7 * 30 + 11)) :=
by sorry

end smallest_reducible_fraction_l3842_384227


namespace f_second_derivative_at_zero_l3842_384258

-- Define the function f
def f (x : ℝ) (f''_1 : ℝ) : ℝ := x^3 - 2 * x * f''_1

-- State the theorem
theorem f_second_derivative_at_zero (f''_1 : ℝ) : 
  (deriv (deriv (f · f''_1))) 0 = -2 :=
sorry

end f_second_derivative_at_zero_l3842_384258


namespace amount_distribution_l3842_384229

theorem amount_distribution (amount : ℕ) : 
  (∀ (x y : ℕ), x = amount / 14 ∧ y = amount / 18 → x = y + 80) →
  amount = 5040 := by
sorry

end amount_distribution_l3842_384229


namespace women_work_nine_hours_l3842_384267

/-- Represents the work scenario with men and women -/
structure WorkScenario where
  men_count : ℕ
  men_days : ℕ
  men_hours_per_day : ℕ
  women_count : ℕ
  women_days : ℕ
  women_efficiency : Rat

/-- Calculates the number of hours women work per day -/
def women_hours_per_day (ws : WorkScenario) : Rat :=
  (ws.men_count * ws.men_days * ws.men_hours_per_day : Rat) /
  (ws.women_count * ws.women_days * ws.women_efficiency)

/-- The given work scenario -/
def given_scenario : WorkScenario :=
  { men_count := 15
  , men_days := 21
  , men_hours_per_day := 8
  , women_count := 21
  , women_days := 20
  , women_efficiency := 2/3 }

theorem women_work_nine_hours : women_hours_per_day given_scenario = 9 := by
  sorry

end women_work_nine_hours_l3842_384267


namespace binomial_square_expansion_l3842_384271

theorem binomial_square_expansion : 121 + 2*(11*9) + 81 = 400 := by sorry

end binomial_square_expansion_l3842_384271


namespace smallest_a_value_l3842_384202

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  ∀ a' : ℝ, (0 ≤ a' ∧ (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x))) → a ≤ a' → a = 17 :=
sorry

end smallest_a_value_l3842_384202


namespace fraction_problem_l3842_384279

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x > 0) (h2 : x = 1/3) 
  (h3 : f * x = (16/216) * (1/x)) : f = 2/3 := by
  sorry

end fraction_problem_l3842_384279


namespace pamelas_remaining_skittles_l3842_384216

def initial_skittles : ℕ := 50
def skittles_given : ℕ := 7

theorem pamelas_remaining_skittles :
  initial_skittles - skittles_given = 43 := by
  sorry

end pamelas_remaining_skittles_l3842_384216


namespace largest_square_from_rectangle_l3842_384241

/-- Given a rectangular paper of length 54 cm and width 20 cm, 
    the largest side length of three equal squares that can be cut from this paper is 18 cm. -/
theorem largest_square_from_rectangle : ∀ (side_length : ℝ), 
  side_length > 0 ∧ 
  3 * side_length ≤ 54 ∧ 
  side_length ≤ 20 →
  side_length ≤ 18 :=
by sorry

end largest_square_from_rectangle_l3842_384241


namespace lioness_hyena_age_ratio_l3842_384235

/-- The ratio of a lioness's age to a hyena's age in a park -/
theorem lioness_hyena_age_ratio :
  ∀ (hyena_age : ℕ) (k : ℕ+),
  k * hyena_age = 12 →
  (6 + 5) + (hyena_age / 2 + 5) = 19 →
  (12 : ℚ) / hyena_age = 2 := by
  sorry

end lioness_hyena_age_ratio_l3842_384235


namespace vidya_mother_age_l3842_384248

theorem vidya_mother_age (vidya_age : ℕ) (mother_age : ℕ) : 
  vidya_age = 13 → 
  mother_age = 3 * vidya_age + 5 → 
  mother_age = 44 := by
sorry

end vidya_mother_age_l3842_384248


namespace unique_solution_quadratic_l3842_384289

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end unique_solution_quadratic_l3842_384289


namespace system_solution_l3842_384275

theorem system_solution (a : ℝ) (x y z : ℝ) :
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = a^2) →
  (x^3 + y^3 + z^3 = a^3) →
  ((x = 0 ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = 0)) :=
by sorry

end system_solution_l3842_384275


namespace gcd_8369_4087_2159_l3842_384247

theorem gcd_8369_4087_2159 : Nat.gcd 8369 (Nat.gcd 4087 2159) = 1 := by
  sorry

end gcd_8369_4087_2159_l3842_384247


namespace arcsin_of_one_l3842_384283

theorem arcsin_of_one : Real.arcsin 1 = π / 2 := by
  sorry

end arcsin_of_one_l3842_384283


namespace tangent_line_and_inequality_l3842_384253

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem tangent_line_and_inequality :
  (∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ exp 2 * x - 4 * y = 0) ∧
  (∀ x, x > 0 → f x > 2 * (x - log x)) := by
  sorry

end tangent_line_and_inequality_l3842_384253


namespace gcd_lcm_product_l3842_384263

theorem gcd_lcm_product (a b : ℕ) (ha : a = 180) (hb : b = 250) :
  (Nat.gcd a b) * (Nat.lcm a b) = 45000 := by
  sorry

end gcd_lcm_product_l3842_384263


namespace sum_of_roots_equation_l3842_384266

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 18*x₁ + 30 = 2 * Real.sqrt (x₁^2 + 18*x₁ + 45)) ∧ 
                (x₂^2 + 18*x₂ + 30 = 2 * Real.sqrt (x₂^2 + 18*x₂ + 45)) ∧ 
                (∀ y : ℝ, y^2 + 18*y + 30 = 2 * Real.sqrt (y^2 + 18*y + 45) → y = x₁ ∨ y = x₂)) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 18*x₁ + 30 = 2 * Real.sqrt (x₁^2 + 18*x₁ + 45)) ∧ 
                (x₂^2 + 18*x₂ + 30 = 2 * Real.sqrt (x₂^2 + 18*x₂ + 45)) ∧ 
                (x₁ + x₂ = -18)) := by
  sorry

end sum_of_roots_equation_l3842_384266


namespace expression_evaluation_l3842_384233

theorem expression_evaluation : 3^(0^(2^8)) + ((3^0)^2)^8 = 2 := by sorry

end expression_evaluation_l3842_384233


namespace tangent_line_equation_l3842_384273

/-- The curve function f(x) = x³ + 1 -/
def f (x : ℝ) : ℝ := x^3 + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_equation :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3 * x - y + 3 = 0) :=
by sorry

end tangent_line_equation_l3842_384273


namespace prob_two_primes_equals_216_625_l3842_384254

-- Define a 10-sided die
def tenSidedDie : Finset ℕ := Finset.range 10

-- Define the set of prime numbers on a 10-sided die
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define the probability of rolling a prime number on one die
def probPrime : ℚ := (primes.card : ℚ) / (tenSidedDie.card : ℚ)

-- Define the probability of not rolling a prime number on one die
def probNotPrime : ℚ := 1 - probPrime

-- Define the number of ways to choose 2 dice out of 4
def waysToChoose : ℕ := Nat.choose 4 2

-- Define the probability of exactly two dice showing a prime number
def probTwoPrimes : ℚ := (waysToChoose : ℚ) * probPrime^2 * probNotPrime^2

-- Theorem statement
theorem prob_two_primes_equals_216_625 : probTwoPrimes = 216 / 625 := by
  sorry

end prob_two_primes_equals_216_625_l3842_384254
