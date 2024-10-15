import Mathlib

namespace NUMINAMATH_CALUDE_price_reduction_profit_l3693_369313

/-- Represents the daily sales and profit scenario of a product in a shopping mall -/
structure MallSales where
  initialSales : ℕ  -- Initial daily sales in units
  initialProfit : ℕ  -- Initial profit per unit in yuan
  salesIncrease : ℕ  -- Increase in sales units per yuan of price reduction
  priceReduction : ℕ  -- Price reduction per unit in yuan

/-- Calculates the daily profit based on the given sales scenario -/
def dailyProfit (m : MallSales) : ℕ :=
  (m.initialSales + m.salesIncrease * m.priceReduction) * (m.initialProfit - m.priceReduction)

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_profit (m : MallSales) 
  (h1 : m.initialSales = 30)
  (h2 : m.initialProfit = 50)
  (h3 : m.salesIncrease = 2)
  (h4 : m.priceReduction = 20) :
  dailyProfit m = 2100 := by
  sorry

#eval dailyProfit { initialSales := 30, initialProfit := 50, salesIncrease := 2, priceReduction := 20 }

end NUMINAMATH_CALUDE_price_reduction_profit_l3693_369313


namespace NUMINAMATH_CALUDE_probability_all_red_balls_l3693_369323

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def blue_balls : ℕ := 5
def drawn_balls : ℕ := 5

theorem probability_all_red_balls :
  (Nat.choose red_balls drawn_balls) / (Nat.choose total_balls drawn_balls) = 1 / 252 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_balls_l3693_369323


namespace NUMINAMATH_CALUDE_max_x_value_l3693_369391

theorem max_x_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x - 2 * Real.sqrt y = Real.sqrt (2 * x - y)) : 
  (∀ z : ℝ, z > 0 ∧ ∃ w : ℝ, w > 0 ∧ z - 2 * Real.sqrt w = Real.sqrt (2 * z - w) → z ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_max_x_value_l3693_369391


namespace NUMINAMATH_CALUDE_johnson_family_has_four_children_l3693_369312

/-- Represents the Johnson family -/
structure JohnsonFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  children_ages : List ℕ

/-- The conditions of the Johnson family -/
def johnson_family_conditions (family : JohnsonFamily) : Prop :=
  family.father_age = 55 ∧
  family.num_children + 2 = 6 ∧
  (family.father_age + family.mother_age + family.children_ages.sum) / 6 = 25 ∧
  (family.mother_age + family.children_ages.sum) / (family.num_children + 1) = 18

/-- The theorem stating that the Johnson family has 4 children -/
theorem johnson_family_has_four_children (family : JohnsonFamily) 
  (h : johnson_family_conditions family) : family.num_children = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_has_four_children_l3693_369312


namespace NUMINAMATH_CALUDE_obtuse_triangle_area_l3693_369325

theorem obtuse_triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 8) (h2 : b = 12) (h3 : C = 150 * π / 180) :
  let area := (1/2) * a * b * Real.sin C
  area = 24 := by
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_area_l3693_369325


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_parallel_perpendicular_transitivity_l3693_369357

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_implication 
  (m n : Line) (α : Plane) : 
  perpendicular m α → parallel_line_plane n α → perpendicular_lines m n :=
sorry

-- Theorem 2
theorem parallel_perpendicular_transitivity 
  (m : Line) (α β γ : Plane) :
  parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_parallel_perpendicular_transitivity_l3693_369357


namespace NUMINAMATH_CALUDE_dragons_total_games_dragons_total_games_is_90_l3693_369352

theorem dragons_total_games : ℕ → Prop :=
  fun total_games =>
    ∃ (pre_tournament_games : ℕ) (pre_tournament_wins : ℕ),
      -- Condition 1: 60% win rate before tournament
      pre_tournament_wins = (6 * pre_tournament_games) / 10 ∧
      -- Condition 2: 9 wins and 3 losses in tournament
      total_games = pre_tournament_games + 12 ∧
      -- Condition 3: 62% overall win rate after tournament
      (pre_tournament_wins + 9) = (62 * total_games) / 100 ∧
      -- Prove that total games is 90
      total_games = 90

theorem dragons_total_games_is_90 : dragons_total_games 90 := by
  sorry

end NUMINAMATH_CALUDE_dragons_total_games_dragons_total_games_is_90_l3693_369352


namespace NUMINAMATH_CALUDE_prob_second_white_given_first_white_l3693_369381

/-- Represents the total number of balls -/
def total_balls : ℕ := 9

/-- Represents the number of white balls -/
def white_balls : ℕ := 5

/-- Represents the number of black balls -/
def black_balls : ℕ := 4

/-- Represents the probability of drawing a white ball first -/
def prob_first_white : ℚ := white_balls / total_balls

/-- Represents the probability of drawing two white balls consecutively -/
def prob_both_white : ℚ := (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))

/-- Theorem stating the probability of drawing a white ball second, given the first was white -/
theorem prob_second_white_given_first_white :
  prob_both_white / prob_first_white = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_second_white_given_first_white_l3693_369381


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3693_369340

theorem magnitude_of_complex_fraction (z : ℂ) : 
  z = (3 + Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3693_369340


namespace NUMINAMATH_CALUDE_distance_to_circle_center_l3693_369375

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the problem
theorem distance_to_circle_center 
  (ABC : Triangle)
  (circle : Circle)
  (M : ℝ × ℝ)
  (h1 : ABC.C.1 = ABC.A.1 ∧ ABC.C.2 = ABC.B.2) -- Right triangle condition
  (h2 : (ABC.A.1 - ABC.B.1)^2 + (ABC.A.2 - ABC.B.2)^2 = 
        (ABC.C.1 - ABC.B.1)^2 + (ABC.C.2 - ABC.B.2)^2) -- Equal legs condition
  (h3 : circle.radius = (ABC.C.1 - ABC.A.1) / 2) -- Circle diameter is AC
  (h4 : circle.center = ((ABC.A.1 + ABC.C.1) / 2, (ABC.A.2 + ABC.C.2) / 2)) -- Circle center is midpoint of AC
  (h5 : (M.1 - ABC.A.1)^2 + (M.2 - ABC.A.2)^2 = circle.radius^2) -- M is on the circle
  (h6 : (M.1 - ABC.B.1)^2 + (M.2 - ABC.B.2)^2 = 2) -- BM = √2
  : (ABC.B.1 - circle.center.1)^2 + (ABC.B.2 - circle.center.2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_circle_center_l3693_369375


namespace NUMINAMATH_CALUDE_perimeter_semicircular_bounded_rectangle_l3693_369398

/-- The perimeter of a region bounded by semicircular arcs constructed on each side of a rectangle --/
theorem perimeter_semicircular_bounded_rectangle (l w : ℝ) (hl : l = 4 / π) (hw : w = 2 / π) :
  let semicircle_length := π * l / 2
  let semicircle_width := π * w / 2
  semicircle_length + semicircle_length + semicircle_width + semicircle_width = 6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_bounded_rectangle_l3693_369398


namespace NUMINAMATH_CALUDE_cost_price_is_41_l3693_369386

/-- Calculates the cost price per metre of cloth given the total length,
    total selling price, and loss per metre. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Proves that the cost price per metre of cloth is 41 rupees given the specified conditions. -/
theorem cost_price_is_41 :
  cost_price_per_metre 500 18000 5 = 41 := by
  sorry

#eval cost_price_per_metre 500 18000 5

end NUMINAMATH_CALUDE_cost_price_is_41_l3693_369386


namespace NUMINAMATH_CALUDE_largest_power_is_396_l3693_369321

def pow (n : ℕ) : ℕ :=
  sorry

def largest_divisible_power (upper_bound : ℕ) : ℕ :=
  sorry

theorem largest_power_is_396 :
  largest_divisible_power 4000 = 396 :=
sorry

end NUMINAMATH_CALUDE_largest_power_is_396_l3693_369321


namespace NUMINAMATH_CALUDE_expression_evaluation_l3693_369319

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  -2*a - b^2 + 2*a*b = -41 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3693_369319


namespace NUMINAMATH_CALUDE_product_of_roots_plus_two_l3693_369388

theorem product_of_roots_plus_two (u v w : ℝ) : 
  (u^3 - 18*u^2 + 20*u - 8 = 0) →
  (v^3 - 18*v^2 + 20*v - 8 = 0) →
  (w^3 - 18*w^2 + 20*w - 8 = 0) →
  (2+u)*(2+v)*(2+w) = 128 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_two_l3693_369388


namespace NUMINAMATH_CALUDE_equation_solution_l3693_369373

theorem equation_solution : 
  ∃ x : ℚ, (1 / 4 : ℚ) + 5 / x = 12 / x + (1 / 15 : ℚ) → x = 420 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3693_369373


namespace NUMINAMATH_CALUDE_milk_mixture_theorem_l3693_369377

/-- Proves that adding 24 gallons of 10% butterfat milk to 8 gallons of 50% butterfat milk 
    results in a mixture with 20% butterfat. -/
theorem milk_mixture_theorem :
  let initial_volume : ℝ := 8
  let initial_butterfat_percent : ℝ := 50
  let added_volume : ℝ := 24
  let added_butterfat_percent : ℝ := 10
  let desired_butterfat_percent : ℝ := 20
  let total_volume := initial_volume + added_volume
  let total_butterfat := (initial_volume * initial_butterfat_percent / 100) + 
                         (added_volume * added_butterfat_percent / 100)
  (total_butterfat / total_volume) * 100 = desired_butterfat_percent :=
by sorry

end NUMINAMATH_CALUDE_milk_mixture_theorem_l3693_369377


namespace NUMINAMATH_CALUDE_exactly_four_pairs_l3693_369360

/-- The number of ordered pairs (m,n) of positive integers satisfying the given conditions -/
def count_pairs : ℕ := 4

/-- Predicate to check if a pair (m,n) satisfies the required conditions -/
def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ n ∧ m * m - n * n = 144

/-- The theorem stating that there are exactly 4 valid pairs -/
theorem exactly_four_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = count_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ pairs ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_exactly_four_pairs_l3693_369360


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3693_369372

theorem perfect_square_trinomial (m : ℚ) : 
  m > 0 → 
  (∃ a : ℚ, ∀ x : ℚ, x^2 - 2*m*x + 36 = (x - a)^2) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3693_369372


namespace NUMINAMATH_CALUDE_base8_45_equals_base10_37_l3693_369364

/-- Converts a two-digit base-eight number to base-ten -/
def base8_to_base10 (tens : Nat) (units : Nat) : Nat :=
  tens * 8 + units

/-- The base-eight number 45 is equal to the base-ten number 37 -/
theorem base8_45_equals_base10_37 : base8_to_base10 4 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_base8_45_equals_base10_37_l3693_369364


namespace NUMINAMATH_CALUDE_fraction_ordering_l3693_369396

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 11 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3693_369396


namespace NUMINAMATH_CALUDE_range_of_M_l3693_369318

theorem range_of_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b < 1) :
  let M := 1 / (1 + a) + 1 / (1 + b)
  1 < M ∧ M < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_M_l3693_369318


namespace NUMINAMATH_CALUDE_line_intersection_difference_l3693_369320

/-- Given a line y = 2x - 4 intersecting the x-axis at point A(m, 0) and the y-axis at point B(0, n), prove that m - n = 6 -/
theorem line_intersection_difference (m n : ℝ) : 
  (∀ x y, y = 2 * x - 4) →  -- Line equation
  0 = 2 * m - 4 →           -- A(m, 0) satisfies the line equation
  n = -4 →                  -- B(0, n) satisfies the line equation
  m - n = 6 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_difference_l3693_369320


namespace NUMINAMATH_CALUDE_coin_probability_impossibility_l3693_369354

theorem coin_probability_impossibility : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
sorry

end NUMINAMATH_CALUDE_coin_probability_impossibility_l3693_369354


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3693_369322

theorem algebraic_simplification (x y : ℝ) :
  ((-3 * x * y^2)^3 * (-6 * x^2 * y)) / (9 * x^4 * y^5) = 18 * x * y^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3693_369322


namespace NUMINAMATH_CALUDE_blue_jellybean_probability_l3693_369345

/-- The probability of drawing 3 blue jellybeans in succession from a bag of 10 red and 10 blue jellybeans without replacement is 1/9.5. -/
theorem blue_jellybean_probability : 
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  let prob_first : ℚ := blue_jellybeans / total_jellybeans
  let prob_second : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1)
  let prob_third : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2)
  prob_first * prob_second * prob_third = 1 / (19 / 2) :=
by sorry

end NUMINAMATH_CALUDE_blue_jellybean_probability_l3693_369345


namespace NUMINAMATH_CALUDE_correct_password_contains_one_and_seven_l3693_369362

/-- Represents a four-digit password -/
def Password := Fin 4 → Fin 10

/-- Checks if two passwords have exactly two matching digits in different positions -/
def hasTwoMatchingDigits (p1 p2 : Password) : Prop :=
  (∃ i j : Fin 4, i ≠ j ∧ p1 i = p2 i ∧ p1 j = p2 j) ∧
  (∀ i j k : Fin 4, i ≠ j → j ≠ k → k ≠ i → ¬(p1 i = p2 i ∧ p1 j = p2 j ∧ p1 k = p2 k))

/-- The first four incorrect attempts -/
def attempts : Fin 4 → Password
| 0 => λ i => [3, 4, 0, 6].get i
| 1 => λ i => [1, 6, 3, 0].get i
| 2 => λ i => [7, 3, 6, 4].get i
| 3 => λ i => [6, 1, 7, 3].get i

/-- The theorem stating that the correct password must contain 1 and 7 -/
theorem correct_password_contains_one_and_seven 
  (correct : Password)
  (h1 : ∀ i : Fin 4, hasTwoMatchingDigits (attempts i) correct)
  (h2 : correct ≠ attempts 0 ∧ correct ≠ attempts 1 ∧ correct ≠ attempts 2 ∧ correct ≠ attempts 3) :
  (∃ i j : Fin 4, i ≠ j ∧ correct i = 1 ∧ correct j = 7) :=
sorry

end NUMINAMATH_CALUDE_correct_password_contains_one_and_seven_l3693_369362


namespace NUMINAMATH_CALUDE_exist_positive_reals_satisfying_inequalities_l3693_369390

theorem exist_positive_reals_satisfying_inequalities :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 + c^2 > 2 ∧
    a^3 + b^3 + c^3 < 2 ∧
    a^4 + b^4 + c^4 > 2 := by
  sorry

end NUMINAMATH_CALUDE_exist_positive_reals_satisfying_inequalities_l3693_369390


namespace NUMINAMATH_CALUDE_abc_product_magnitude_l3693_369327

theorem abc_product_magnitude (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a - 1/b = b - 1/c ∧ b - 1/c = c - 1/a) :
    |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_magnitude_l3693_369327


namespace NUMINAMATH_CALUDE_smallest_difference_vovochka_sum_l3693_369348

/-- Vovochka's sum method for three-digit numbers -/
def vovochkaSum (a b c d e f : ℕ) : ℕ := 
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- Correct sum method for three-digit numbers -/
def correctSum (a b c d e f : ℕ) : ℕ := 
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f)

/-- Theorem: The smallest positive difference between Vovochka's sum and the correct sum is 1800 -/
theorem smallest_difference_vovochka_sum : 
  ∀ a b c d e f : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
  vovochkaSum a b c d e f - correctSum a b c d e f ≥ 1800 ∧
  ∃ a b c d e f : ℕ, 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    vovochkaSum a b c d e f - correctSum a b c d e f = 1800 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_vovochka_sum_l3693_369348


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_correct_l3693_369389

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The least three-digit 7-heavy number -/
def least_three_digit_7_heavy : ℕ := 103

theorem least_three_digit_7_heavy_correct :
  (least_three_digit_7_heavy ≥ 100) ∧
  (least_three_digit_7_heavy < 1000) ∧
  is_7_heavy least_three_digit_7_heavy ∧
  ∀ n : ℕ, (n ≥ 100) ∧ (n < 1000) ∧ is_7_heavy n → n ≥ least_three_digit_7_heavy :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_correct_l3693_369389


namespace NUMINAMATH_CALUDE_de_length_l3693_369333

/-- Triangle ABC with given side lengths and a line DE parallel to BC containing the incenter --/
structure TriangleWithParallelLine where
  -- Define the triangle
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Ensure AB = 21, AC = 22, BC = 20
  h_AB : AB = 21
  h_AC : AC = 22
  h_BC : BC = 20
  -- Points D and E are on AB and AC respectively
  D : ℝ
  E : ℝ
  h_D : D ≥ 0 ∧ D ≤ AB
  h_E : E ≥ 0 ∧ E ≤ AC
  -- DE is parallel to BC
  h_parallel : True  -- We can't directly express parallelism here, so we assume it's true
  -- DE contains the incenter
  h_incenter : True  -- We can't directly express this, so we assume it's true

/-- The main theorem --/
theorem de_length (t : TriangleWithParallelLine) : ∃ (DE : ℝ), DE = 860 / 63 := by
  sorry

end NUMINAMATH_CALUDE_de_length_l3693_369333


namespace NUMINAMATH_CALUDE_gcd_problems_l3693_369371

theorem gcd_problems : 
  (Nat.gcd 91 49 = 7) ∧ (Nat.gcd (Nat.gcd 319 377) 116 = 29) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l3693_369371


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3693_369326

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m : Line) (α β : Plane) (h_diff : α ≠ β) :
  parallel m α → perpendicular m β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3693_369326


namespace NUMINAMATH_CALUDE_unique_number_property_l3693_369341

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l3693_369341


namespace NUMINAMATH_CALUDE_gcd_of_sum_and_lcm_l3693_369365

theorem gcd_of_sum_and_lcm (a b : ℕ+) (h1 : a + b = 33) (h2 : Nat.lcm a b = 90) : 
  Nat.gcd a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sum_and_lcm_l3693_369365


namespace NUMINAMATH_CALUDE_expression_evaluation_l3693_369394

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 + 1
  ((x + 1) / (x - 1) + 1 / (x^2 - 2*x + 1)) / (x / (x - 1)) = 1 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3693_369394


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l3693_369361

theorem ac_squared_gt_bc_squared_sufficient_not_necessary (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l3693_369361


namespace NUMINAMATH_CALUDE_problem_solution_l3693_369305

theorem problem_solution (x : ℕ) (h : x = 36) : 
  2 * ((((x + 10) * 2) / 2) - 2) = 88 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3693_369305


namespace NUMINAMATH_CALUDE_equation_solution_inequality_solution_l3693_369324

-- Equation problem
theorem equation_solution :
  ∀ x : ℝ, 6 * x - 2 * (x - 3) = 14 ↔ x = 2 := by sorry

-- Inequality problem
theorem inequality_solution :
  ∀ x : ℝ, 3 * (x + 3) < x + 7 ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_solution_l3693_369324


namespace NUMINAMATH_CALUDE_necklace_calculation_l3693_369366

theorem necklace_calculation (spools : Nat) (spool_length : Nat) (feet_per_necklace : Nat) : 
  spools = 3 → spool_length = 20 → feet_per_necklace = 4 → 
  (spools * spool_length) / feet_per_necklace = 15 := by
  sorry

end NUMINAMATH_CALUDE_necklace_calculation_l3693_369366


namespace NUMINAMATH_CALUDE_sufficient_condition_range_exclusive_or_range_l3693_369358

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

/-- Definition of proposition q -/
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

/-- Theorem for part (1) -/
theorem sufficient_condition_range (m : ℝ) :
  (∀ x : ℝ, p x → q m x) → m ∈ Set.Ici 4 :=
sorry

/-- Theorem for part (2) -/
theorem exclusive_or_range (x : ℝ) :
  (p x ∨ q 5 x) ∧ ¬(p x ∧ q 5 x) → x ∈ Set.Icc (-3) (-2) ∪ Set.Ioo 6 7 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_exclusive_or_range_l3693_369358


namespace NUMINAMATH_CALUDE_complex_number_problem_l3693_369337

/-- Given a complex number z = 3 + bi where b is a positive real number,
    and (z - 2)² is a pure imaginary number, prove that:
    1. z = 3 + i
    2. |z / (2 + i)| = √2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) 
    (h1 : b > 0)
    (h2 : z = 3 + b * I)
    (h3 : ∃ (y : ℝ), (z - 2)^2 = y * I) :
  z = 3 + I ∧ Complex.abs (z / (2 + I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3693_369337


namespace NUMINAMATH_CALUDE_sin_cos_difference_75_15_l3693_369311

theorem sin_cos_difference_75_15 :
  Real.sin (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_75_15_l3693_369311


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3693_369301

-- Define the logarithm with base 0.5
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log 0.5

-- State the theorem
theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  2 * (log_half x)^2 + 9 * log_half x + 9 ≤ 0 ↔ 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3693_369301


namespace NUMINAMATH_CALUDE_train_length_calculation_l3693_369355

/-- The length of a train that passes a tree in 12 seconds while traveling at 90 km/hr is 300 meters. -/
theorem train_length_calculation (passing_time : ℝ) (speed_kmh : ℝ) : 
  passing_time = 12 → speed_kmh = 90 → passing_time * (speed_kmh * (1000 / 3600)) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3693_369355


namespace NUMINAMATH_CALUDE_transformation_result_l3693_369304

/-- Rotates a point (x, y) 180° clockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformation_result (c d : ℝ) :
  let (x₁, y₁) := rotate180 c d 2 (-3)
  let (x₂, y₂) := reflectAboutYEqualsX x₁ y₁
  (x₂ = 5 ∧ y₂ = -4) → d - c = -19 := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l3693_369304


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3693_369330

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α + π/4) = 12/13)
  (h2 : π/4 < α) 
  (h3 : α < 3*π/4) : 
  Real.cos α = 7*Real.sqrt 2/26 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3693_369330


namespace NUMINAMATH_CALUDE_shortest_tree_height_l3693_369367

/-- Given three trees in a town square, this theorem proves the height of the shortest tree. -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 →
  middle = 2/3 * tallest →
  shortest = 1/2 * middle →
  shortest = 50 := by
sorry

end NUMINAMATH_CALUDE_shortest_tree_height_l3693_369367


namespace NUMINAMATH_CALUDE_total_passengers_is_420_l3693_369339

/-- Represents the number of carriages in a train -/
def carriages_per_train : ℕ := 4

/-- Represents the original number of seats in each carriage -/
def original_seats_per_carriage : ℕ := 25

/-- Represents the additional number of passengers each carriage can accommodate -/
def additional_passengers_per_carriage : ℕ := 10

/-- Represents the number of trains -/
def number_of_trains : ℕ := 3

/-- Calculates the total number of passengers that can fill up the given number of trains -/
def total_passengers : ℕ :=
  number_of_trains * carriages_per_train * (original_seats_per_carriage + additional_passengers_per_carriage)

theorem total_passengers_is_420 : total_passengers = 420 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_is_420_l3693_369339


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l3693_369303

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem mans_speed_in_still_water
  (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ)
  (h_downstream : downstream_distance = 40)
  (h_upstream : upstream_distance = 56)
  (h_time : time = 8)
  : ∃ (v_m : ℝ), v_m = 6 ∧ 
    downstream_distance / time = v_m + (downstream_distance - upstream_distance) / (2 * time) ∧
    upstream_distance / time = v_m - (downstream_distance - upstream_distance) / (2 * time) :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l3693_369303


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3693_369307

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

#eval weight_of_replaced_person 8 (5/2) 85

end NUMINAMATH_CALUDE_replaced_person_weight_l3693_369307


namespace NUMINAMATH_CALUDE_john_phone_bill_cost_l3693_369350

/-- Calculates the total cost of a phone bill given the monthly fee, per-minute rate, and minutes used. -/
def phoneBillCost (monthlyFee : ℝ) (perMinuteRate : ℝ) (minutesUsed : ℝ) : ℝ :=
  monthlyFee + perMinuteRate * minutesUsed

theorem john_phone_bill_cost :
  phoneBillCost 5 0.25 28.08 = 12.02 := by
  sorry

end NUMINAMATH_CALUDE_john_phone_bill_cost_l3693_369350


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3693_369346

/-- A point in a 2D plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis -/
def symmetricXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The problem statement -/
theorem symmetric_point_coordinates :
  let M : Point := ⟨-2, 1⟩
  let N : Point := ⟨-2, -1⟩
  symmetricXAxis M N := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3693_369346


namespace NUMINAMATH_CALUDE_range_of_a_l3693_369370

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (h_odd : is_odd f) 
  (h_domain : ∀ x ∈ Set.Ioo (-1) 1, f x ≠ 0) 
  (h_ineq : ∀ a : ℝ, f (1 - a) + f (2 * a - 1) < 0) :
  Set.Ioo 0 1 = {a : ℝ | 0 < a ∧ a < 1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3693_369370


namespace NUMINAMATH_CALUDE_at_least_one_seedling_exactly_one_success_l3693_369315

-- Define the probabilities
def prob_A_seedling : ℝ := 0.6
def prob_B_seedling : ℝ := 0.5
def prob_A_survive : ℝ := 0.7
def prob_B_survive : ℝ := 0.9

-- Theorem 1: Probability that at least one type of fruit tree becomes a seedling
theorem at_least_one_seedling :
  1 - (1 - prob_A_seedling) * (1 - prob_B_seedling) = 0.8 := by sorry

-- Theorem 2: Probability that exactly one type of fruit tree is successfully cultivated and survives
theorem exactly_one_success :
  let prob_A_success := prob_A_seedling * prob_A_survive
  let prob_B_success := prob_B_seedling * prob_B_survive
  prob_A_success * (1 - prob_B_success) + (1 - prob_A_success) * prob_B_success = 0.492 := by sorry

end NUMINAMATH_CALUDE_at_least_one_seedling_exactly_one_success_l3693_369315


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3693_369376

theorem arithmetic_evaluation : 1523 + 180 / 60 - 223 = 1303 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3693_369376


namespace NUMINAMATH_CALUDE_max_intersection_area_l3693_369309

/-- A right prism with a square base centered at the origin --/
structure Prism :=
  (side_length : ℝ)
  (center : ℝ × ℝ × ℝ := (0, 0, 0))

/-- A plane in 3D space defined by its equation coefficients --/
structure Plane :=
  (a b c d : ℝ)

/-- The intersection of a prism and a plane --/
def intersection (p : Prism) (plane : Plane) : Set (ℝ × ℝ × ℝ) :=
  {pt : ℝ × ℝ × ℝ | 
    let (x, y, z) := pt
    plane.a * x + plane.b * y + plane.c * z = plane.d ∧
    |x| ≤ p.side_length / 2 ∧
    |y| ≤ p.side_length / 2}

/-- The area of a set in 3D space --/
noncomputable def area (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the maximum area of the intersection is equal to the area of the square base --/
theorem max_intersection_area (p : Prism) (plane : Plane) :
  p.side_length = 12 ∧
  plane = {a := 3, b := -6, c := 2, d := 24} →
  area (intersection p plane) ≤ p.side_length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_area_l3693_369309


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l3693_369300

/-- Given a student's scores in mathematics, physics, and chemistry, prove that the difference between chemistry and physics scores is 20 marks. -/
theorem chemistry_physics_difference
  (M P C : ℕ)  -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 60)  -- Total marks in mathematics and physics is 60
  (h2 : ∃ X : ℕ, C = P + X)  -- Chemistry score is some marks more than physics
  (h3 : (M + C) / 2 = 40)  -- Average marks in mathematics and chemistry is 40
  : ∃ X : ℕ, C = P + X ∧ X = 20 := by
  sorry

#check chemistry_physics_difference

end NUMINAMATH_CALUDE_chemistry_physics_difference_l3693_369300


namespace NUMINAMATH_CALUDE_quadratic_y_values_order_l3693_369374

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2

theorem quadratic_y_values_order :
  ∀ (y₁ y₂ y₃ : ℝ),
  quadratic_function 1 = y₁ →
  quadratic_function 2 = y₂ →
  quadratic_function (-3) = y₃ →
  y₃ < y₁ ∧ y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_y_values_order_l3693_369374


namespace NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l3693_369378

/-- The sum of interior angles of a triangle in degrees -/
def triangle_angle_sum : ℝ := 180

/-- The number of triangles in the diagram -/
def number_of_triangles : ℕ := 2

/-- Theorem: The sum of all interior angles in two triangles is 360° -/
theorem sum_of_angles_two_triangles : 
  (↑number_of_triangles : ℝ) * triangle_angle_sum = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l3693_369378


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3693_369334

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, 4 * x^2 + 8 * x + 16 ≥ 12) ∧ 
  (∃ x, 4 * x^2 + 8 * x + 16 = 12) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3693_369334


namespace NUMINAMATH_CALUDE_geometric_progression_property_l3693_369316

def geometric_progression (b : ℕ → ℝ) := 
  ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1

theorem geometric_progression_property (b : ℕ → ℝ) 
  (h₁ : geometric_progression b) 
  (h₂ : ∀ n : ℕ, b n > 0) : 
  (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) ^ (1/6) = (b 3 * b 4) ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_property_l3693_369316


namespace NUMINAMATH_CALUDE_asaf_age_l3693_369314

theorem asaf_age :
  ∀ (asaf_age alexander_age asaf_pencils : ℕ),
    -- Sum of ages is 140
    asaf_age + alexander_age = 140 →
    -- Age difference is half of Asaf's pencils
    alexander_age - asaf_age = asaf_pencils / 2 →
    -- Total pencils is 220
    asaf_pencils + (asaf_pencils + 60) = 220 →
    -- Asaf's age is 90
    asaf_age = 90 := by
  sorry

end NUMINAMATH_CALUDE_asaf_age_l3693_369314


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3693_369363

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) = a n + d) ∧
  ∃ r : ℝ, r ≠ 0 ∧ a 3 = r * a 1 ∧ a 6 = r * a 3
  →
  ∃ r : ℝ, r = 3/2 ∧ a 3 = r * a 1 ∧ a 6 = r * a 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3693_369363


namespace NUMINAMATH_CALUDE_seven_classes_matches_l3693_369382

/-- 
Given a number of classes, calculates the total number of matches 
when each class plays against every other class exactly once.
-/
def totalMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- 
Theorem: When 7 classes play against each other once, 
the total number of matches is 21.
-/
theorem seven_classes_matches : totalMatches 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_classes_matches_l3693_369382


namespace NUMINAMATH_CALUDE_cuboid_distance_theorem_l3693_369380

/-- Given a cuboid with edges a, b, and c, and a vertex P, 
    the distance m from P to the plane passing through the vertices adjacent to P 
    satisfies the equation: 1/m² = 1/a² + 1/b² + 1/c² -/
theorem cuboid_distance_theorem (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hm : m > 0) :
  (1 / m^2) = (1 / a^2) + (1 / b^2) + (1 / c^2) :=
sorry

end NUMINAMATH_CALUDE_cuboid_distance_theorem_l3693_369380


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l3693_369395

theorem triangle_abc_problem (a b c A B C : ℝ) 
  (h1 : a * Real.sin A = 4 * b * Real.sin B)
  (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2)) :
  Real.cos A = -(Real.sqrt 5) / 5 ∧ 
  Real.sin (2 * B - A) = -2 * (Real.sqrt 5) / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l3693_369395


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l3693_369310

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ (11 / 8)) :=
sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l3693_369310


namespace NUMINAMATH_CALUDE_complementary_sets_imply_a_eq_two_subset_implies_a_range_l3693_369356

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Theorem 1: If A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem complementary_sets_imply_a_eq_two (a : ℝ) :
  A a ∩ B = ∅ ∧ A a ∪ B = Set.univ → a = 2 := by sorry

-- Theorem 2: If A ⊆ B, then a ∈ (-∞, 0] ∪ [4, +∞)
theorem subset_implies_a_range (a : ℝ) :
  A a ⊆ B → a ≤ 0 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_complementary_sets_imply_a_eq_two_subset_implies_a_range_l3693_369356


namespace NUMINAMATH_CALUDE_locus_of_P_l3693_369387

/-- Given two variable points A and B on the x-axis and y-axis respectively,
    such that AB is in the first quadrant and has fixed length 2d,
    and a point P such that P and the origin are on opposite sides of AB,
    and PC is perpendicular to AB with length d (where C is the midpoint of AB),
    prove that P lies on the line y = x and its distance from the origin
    is between d√2 and 2d inclusive. -/
theorem locus_of_P (d : ℝ) (A B P : ℝ × ℝ) (h_d : d > 0) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (A.2 = 0) →
  (B.1 = 0) →
  (A.1 ≥ 0 ∧ B.2 ≥ 0) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*d)^2 →
  ((P.1 - C.1) * (B.1 - A.1) + (P.2 - C.2) * (B.2 - A.2) = 0) →
  ((P.1 - C.1)^2 + (P.2 - C.2)^2 = d^2) →
  (P.1 * B.2 > P.2 * A.1) →
  (P.1 = P.2 ∧ d * Real.sqrt 2 ≤ Real.sqrt (P.1^2 + P.2^2) ∧ Real.sqrt (P.1^2 + P.2^2) ≤ 2*d) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_P_l3693_369387


namespace NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l3693_369336

/-- The number of ways to arrange n distinguishable people around a circular table,
    where rotations are considered the same arrangement -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

theorem senate_committee_seating_arrangements :
  circularArrangements 10 = 362880 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l3693_369336


namespace NUMINAMATH_CALUDE_complex_modulus_l3693_369347

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3693_369347


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3693_369369

/-- The sum of the solutions to the quadratic equation x² - 6x - 8 = 2x + 18 is 8 -/
theorem sum_of_quadratic_solutions : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x - 8 - (2*x + 18)
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3693_369369


namespace NUMINAMATH_CALUDE_ab_difference_l3693_369397

theorem ab_difference (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ab_difference_l3693_369397


namespace NUMINAMATH_CALUDE_total_houses_l3693_369393

theorem total_houses (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : garage = 50) 
  (h2 : pool = 40) 
  (h3 : both = 35) : 
  garage + pool - both = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_houses_l3693_369393


namespace NUMINAMATH_CALUDE_quadratic_equation_c_value_l3693_369384

theorem quadratic_equation_c_value (b c : ℝ) : 
  (∀ x : ℝ, x^2 - b*x + c = 0 → 
    ∃ y : ℝ, y^2 - b*y + c = 0 ∧ x ≠ y ∧ x * y = 20 ∧ x + y = 12) →
  c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_c_value_l3693_369384


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3693_369385

theorem sqrt_difference_equality : Real.sqrt (64 + 81) - Real.sqrt (49 - 36) = Real.sqrt 145 - Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3693_369385


namespace NUMINAMATH_CALUDE_parabola_segment_length_squared_l3693_369338

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y = 3x^2 - 4x + 5 -/
def onParabola (p : Point) : Prop :=
  p.y = 3 * p.x^2 - 4 * p.x + 5

/-- The origin (0, 0) is the midpoint of two points -/
def originIsMidpoint (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- The square of the distance between two points -/
def squareDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The main theorem -/
theorem parabola_segment_length_squared :
  ∀ p q : Point,
  onParabola p → onParabola q → originIsMidpoint p q →
  squareDistance p q = 8900 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_segment_length_squared_l3693_369338


namespace NUMINAMATH_CALUDE_g_stable_point_fixed_points_subset_stable_points_l3693_369335

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define fixed point
def is_fixed_point (f : RealFunction) (x : ℝ) : Prop := f x = x

-- Define stable point
def is_stable_point (f : RealFunction) (x : ℝ) : Prop := f (f x) = x

-- Define the set of fixed points
def fixed_points (f : RealFunction) : Set ℝ := {x | is_fixed_point f x}

-- Define the set of stable points
def stable_points (f : RealFunction) : Set ℝ := {x | is_stable_point f x}

-- Define the function g(x) = 3x - 8
def g : RealFunction := λ x ↦ 3 * x - 8

-- Theorem: The stable point of g(x) = 3x - 8 is x = 4
theorem g_stable_point : is_stable_point g 4 := by sorry

-- Theorem: For any function, the set of fixed points is a subset of the set of stable points
theorem fixed_points_subset_stable_points (f : RealFunction) : 
  fixed_points f ⊆ stable_points f := by sorry

end NUMINAMATH_CALUDE_g_stable_point_fixed_points_subset_stable_points_l3693_369335


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_l3693_369359

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

theorem extreme_points_imply_a (a b : ℝ) :
  (∀ x, x > 0 → (deriv (f a b)) x = 0 ↔ x = 1 ∨ x = 2) →
  a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_l3693_369359


namespace NUMINAMATH_CALUDE_number_property_l3693_369342

theorem number_property (y : ℝ) : y = (1 / y) * (-y) + 3 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_property_l3693_369342


namespace NUMINAMATH_CALUDE_base_conversion_digit_sum_l3693_369368

theorem base_conversion_digit_sum : 
  (∃ (d_min d_max : ℕ), 
    (∀ n : ℕ, 
      (9^3 ≤ n ∧ n < 9^4) → 
      (d_min ≤ Nat.log2 (n + 1) ∧ Nat.log2 (n + 1) ≤ d_max)) ∧
    (d_max - d_min = 2) ∧
    (d_min + (d_min + 1) + d_max = 33)) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_digit_sum_l3693_369368


namespace NUMINAMATH_CALUDE_numeric_methods_students_l3693_369343

/-- The total number of students in the faculty -/
def total_students : ℕ := 653

/-- The number of second-year students studying automatic control -/
def auto_control_students : ℕ := 423

/-- The number of second-year students studying both numeric methods and automatic control -/
def both_subjects_students : ℕ := 134

/-- The approximate percentage of second-year students in the faculty -/
def second_year_percentage : ℚ := 80/100

/-- The number of second-year students (rounded) -/
def second_year_students : ℕ := 522

/-- Theorem stating the number of second-year students studying numeric methods -/
theorem numeric_methods_students : 
  ∃ (n : ℕ), n = second_year_students - (auto_control_students - both_subjects_students) ∧ n = 233 :=
sorry

end NUMINAMATH_CALUDE_numeric_methods_students_l3693_369343


namespace NUMINAMATH_CALUDE_triangle_properties_l3693_369379

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of a specific triangle --/
theorem triangle_properties (t : Triangle) 
  (h1 : t.B = π / 3) :
  (t.a = 2 ∧ t.b = 2 * Real.sqrt 3 → t.c = 4) ∧
  (Real.tan t.A = 2 * Real.sqrt 3 → Real.tan t.C = 3 * Real.sqrt 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3693_369379


namespace NUMINAMATH_CALUDE_carreys_rental_cost_l3693_369344

/-- The cost per kilometer for Carrey's car rental -/
def carreys_cost_per_km : ℝ := 0.25

/-- The initial cost for Carrey's car rental -/
def carreys_initial_cost : ℝ := 20

/-- The initial cost for Samuel's car rental -/
def samuels_initial_cost : ℝ := 24

/-- The cost per kilometer for Samuel's car rental -/
def samuels_cost_per_km : ℝ := 0.16

/-- The distance driven by both Carrey and Samuel -/
def distance_driven : ℝ := 44.44444444444444

theorem carreys_rental_cost (x : ℝ) :
  carreys_initial_cost + x * distance_driven =
  samuels_initial_cost + samuels_cost_per_km * distance_driven →
  x = carreys_cost_per_km :=
by sorry

end NUMINAMATH_CALUDE_carreys_rental_cost_l3693_369344


namespace NUMINAMATH_CALUDE_blue_marble_probability_l3693_369308

/-- Given odds for an event, calculates the probability of the event not occurring -/
def probability_of_not_occurring (favorable : ℕ) (unfavorable : ℕ) : ℚ :=
  unfavorable / (favorable + unfavorable)

/-- Theorem: If the odds for drawing a blue marble are 5:9, 
    the probability of not drawing a blue marble is 9/14 -/
theorem blue_marble_probability :
  probability_of_not_occurring 5 9 = 9 / 14 := by
sorry

end NUMINAMATH_CALUDE_blue_marble_probability_l3693_369308


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3693_369306

def f (x : ℤ) : ℤ := x^3 - 3*x^2 - 13*x + 15

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {-3, 1, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3693_369306


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l3693_369331

/-- The time taken for a power boat to travel downstream from dock A to dock B,
    given the conditions of the river journey problem. -/
theorem power_boat_travel_time
  (r : ℝ) -- speed of the river current
  (p : ℝ) -- relative speed of the power boat with respect to the river
  (h1 : r > 0) -- river speed is positive
  (h2 : p > r) -- power boat speed is greater than river speed
  : ∃ t : ℝ,
    t > 0 ∧
    t = (12 * r) / (6 * p - r) ∧
    (p + r) * t + (p - r) * (12 - t) = 12 * r :=
by sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l3693_369331


namespace NUMINAMATH_CALUDE_expected_turns_to_second_ace_prove_expected_turns_l3693_369328

/-- A deck of cards -/
structure Deck :=
  (n : ℕ)  -- Total number of cards
  (h : n ≥ 3)  -- There are at least 3 cards (for the 3 aces)

/-- The expected number of cards turned up until the second ace appears -/
def expectedTurnsToSecondAce (d : Deck) : ℚ :=
  (d.n + 1) / 2

/-- Theorem stating that the expected number of cards turned up until the second ace appears is (n+1)/2 -/
theorem expected_turns_to_second_ace (d : Deck) :
  expectedTurnsToSecondAce d = (d.n + 1) / 2 := by
  sorry

/-- Main theorem proving the expected number of cards turned up -/
theorem prove_expected_turns (d : Deck) :
  ∃ (e : ℚ), e = expectedTurnsToSecondAce d ∧ e = (d.n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_turns_to_second_ace_prove_expected_turns_l3693_369328


namespace NUMINAMATH_CALUDE_conic_single_point_implies_d_eq_11_l3693_369329

/-- A conic section represented by the equation 2x^2 + y^2 + 4x - 6y + d = 0 -/
def conic (d : ℝ) (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 4 * x - 6 * y + d = 0

/-- The conic degenerates to a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, conic d p.1 p.2

/-- If the conic degenerates to a single point, then d = 11 -/
theorem conic_single_point_implies_d_eq_11 :
  ∀ d : ℝ, is_single_point d → d = 11 := by sorry

end NUMINAMATH_CALUDE_conic_single_point_implies_d_eq_11_l3693_369329


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_min_value_f_l3693_369392

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_min_value_f_l3693_369392


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3693_369332

theorem inequality_system_solution (x : ℝ) : 
  ((3*x - 2) / (x - 6) ≤ 1 ∧ 2*x^2 - x - 1 > 0) ↔ 
  ((-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6)) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3693_369332


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3693_369351

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

def inscribed_square (s : ℝ) : Prop :=
  ellipse s s ∧ ellipse (-s) s ∧ ellipse s (-s) ∧ ellipse (-s) (-s)

theorem inscribed_square_area :
  ∃ s : ℝ, inscribed_square s ∧ (2*s)^2 = 32/3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3693_369351


namespace NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l3693_369302

/-- Given a square and a regular octagon with equal perimeters, 
    if the square's area is 16, then the area of the octagon is 8(1+√2) -/
theorem octagon_area_given_equal_perimeter_and_square_area (a b : ℝ) : 
  a > 0 → b > 0 →
  (4 * a = 8 * b) →  -- Equal perimeters
  (a ^ 2 = 16) →     -- Square's area is 16
  (2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l3693_369302


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l3693_369383

/-- The number of ways to arrange 8 athletes on 8 tracks with 3 specified athletes in consecutive tracks -/
def arrangement_count : ℕ := 4320

/-- The number of tracks in the stadium -/
def num_tracks : ℕ := 8

/-- The total number of athletes -/
def num_athletes : ℕ := 8

/-- The number of specified athletes that must be in consecutive tracks -/
def num_specified : ℕ := 3

/-- The number of ways to arrange the specified athletes in consecutive tracks -/
def consecutive_arrangements : ℕ := num_tracks - num_specified + 1

/-- The number of permutations of the specified athletes -/
def specified_permutations : ℕ := Nat.factorial num_specified

/-- The number of permutations of the remaining athletes -/
def remaining_permutations : ℕ := Nat.factorial (num_athletes - num_specified)

theorem arrangement_count_proof : 
  arrangement_count = consecutive_arrangements * specified_permutations * remaining_permutations :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l3693_369383


namespace NUMINAMATH_CALUDE_not_power_of_two_l3693_369353

theorem not_power_of_two (m n : ℕ+) : ¬∃ k : ℕ, (36 * m.val + n.val) * (m.val + 36 * n.val) = 2^k := by
  sorry

end NUMINAMATH_CALUDE_not_power_of_two_l3693_369353


namespace NUMINAMATH_CALUDE_special_number_unique_l3693_369349

/-- The unique integer between 10000 and 99999 satisfying the given conditions -/
def special_number : ℕ := 11311

/-- Checks if a natural number is between 10000 and 99999 -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Extracts the digits of a five-digit number -/
def digits (n : ℕ) : Fin 5 → ℕ
| 0 => n / 10000
| 1 => (n / 1000) % 10
| 2 => (n / 100) % 10
| 3 => (n / 10) % 10
| 4 => n % 10

theorem special_number_unique :
  ∀ n : ℕ, is_five_digit n →
    (digits n 0 = n % 2) →
    (digits n 1 = n % 3) →
    (digits n 2 = n % 4) →
    (digits n 3 = n % 5) →
    (digits n 4 = n % 6) →
    n = special_number := by sorry

end NUMINAMATH_CALUDE_special_number_unique_l3693_369349


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_prime_factorization_property_l3693_369317

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def prime_factorization_sum_property (n : ℕ) : Prop :=
  ∃ (factors : List ℕ) (exponents : List ℕ),
    n = (factors.zip exponents).foldl (λ acc (p, e) => acc * p^e) 1 ∧
    factors.all Nat.Prime ∧
    factors.sum = exponents.sum

theorem four_digit_numbers_with_prime_factorization_property :
  {n : ℕ | is_four_digit n ∧ prime_factorization_sum_property n} =
  {1792, 2000, 3125, 3840, 5000, 5760, 6272, 8640, 9600} := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_prime_factorization_property_l3693_369317


namespace NUMINAMATH_CALUDE_number_exists_l3693_369399

theorem number_exists : ∃ x : ℝ, (x^2 * 9^2) / 356 = 51.193820224719104 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l3693_369399
