import Mathlib

namespace lily_paint_cans_l4119_411964

/-- Given the initial paint coverage, lost cans, and remaining coverage, 
    calculate the number of cans used for the remaining rooms --/
def paint_cans_used (initial_coverage : ℕ) (lost_cans : ℕ) (remaining_coverage : ℕ) : ℕ :=
  (remaining_coverage * lost_cans) / (initial_coverage - remaining_coverage)

/-- Theorem stating that under the given conditions, 16 cans were used for 32 rooms --/
theorem lily_paint_cans : paint_cans_used 40 4 32 = 16 := by
  sorry

end lily_paint_cans_l4119_411964


namespace dot_product_equality_l4119_411981

/-- Given points in 2D space, prove that OA · OP₃ = OP₁ · OP₂ -/
theorem dot_product_equality (α β : ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let P₁ : ℝ × ℝ := (Real.cos α, Real.sin α)
  let P₂ : ℝ × ℝ := (Real.cos β, -Real.sin β)
  let P₃ : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
  let A : ℝ × ℝ := (1, 0)
  (A.1 - O.1) * (P₃.1 - O.1) + (A.2 - O.2) * (P₃.2 - O.2) =
  (P₁.1 - O.1) * (P₂.1 - O.1) + (P₁.2 - O.2) * (P₂.2 - O.2) :=
by
  sorry

#check dot_product_equality

end dot_product_equality_l4119_411981


namespace exactly_two_out_of_four_probability_l4119_411920

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The binomial probability mass function -/
def binomialPMF (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_out_of_four_probability :
  binomialPMF n k p = 0.3456 := by
  sorry

end exactly_two_out_of_four_probability_l4119_411920


namespace no_real_solution_l4119_411939

theorem no_real_solution : ¬∃ (x : ℝ), 
  (x + 5 > 0) ∧ 
  (x - 3 > 0) ∧ 
  (x^2 - 8*x + 7 > 0) ∧ 
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 7)) := by
sorry

end no_real_solution_l4119_411939


namespace roxy_plants_l4119_411924

def plants_problem (initial_flowering : ℕ) (initial_fruiting_ratio : ℕ) 
  (bought_fruiting : ℕ) (given_away_flowering : ℕ) (given_away_fruiting : ℕ) 
  (final_total : ℕ) : Prop :=
  ∃ (bought_flowering : ℕ),
    let initial_fruiting := initial_flowering * initial_fruiting_ratio
    let initial_total := initial_flowering + initial_fruiting
    let after_buying := initial_total + bought_flowering + bought_fruiting
    let after_giving := after_buying - given_away_flowering - given_away_fruiting
    after_giving = final_total ∧ bought_flowering = 3

theorem roxy_plants : plants_problem 7 2 2 1 4 21 := by
  sorry

end roxy_plants_l4119_411924


namespace symmetric_point_theorem_l4119_411969

/-- A point in a 2D Cartesian coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis. -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

/-- The theorem stating that the symmetric point of (2, -8) with respect to the y-axis is (-2, -8). -/
theorem symmetric_point_theorem :
  let A : Point := ⟨2, -8⟩
  let B : Point := ⟨-2, -8⟩
  symmetricAboutYAxis A B := by
  sorry

end symmetric_point_theorem_l4119_411969


namespace case_cost_is_nine_l4119_411906

/-- The cost of a case of paper towels -/
def case_cost (num_rolls : ℕ) (individual_roll_cost : ℚ) (savings_percent : ℚ) : ℚ :=
  num_rolls * (individual_roll_cost * (1 - savings_percent / 100))

/-- Theorem stating the cost of a case of 12 rolls is $9 -/
theorem case_cost_is_nine :
  case_cost 12 1 25 = 9 := by
  sorry

end case_cost_is_nine_l4119_411906


namespace garlic_cloves_remaining_l4119_411967

theorem garlic_cloves_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 237 → used = 184 → remaining = initial - used → remaining = 53 := by
sorry

end garlic_cloves_remaining_l4119_411967


namespace aarons_playground_area_l4119_411963

/-- Represents a rectangular playground with fence posts. -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Calculates the area of the playground given its specifications. -/
def playground_area (p : Playground) : ℕ :=
  (p.short_side_posts - 1) * p.post_spacing * (p.long_side_posts - 1) * p.post_spacing

/-- Theorem stating the area of Aaron's playground is 400 square yards. -/
theorem aarons_playground_area :
  ∃ (p : Playground),
    p.total_posts = 24 ∧
    p.post_spacing = 5 ∧
    p.long_side_posts = 3 * p.short_side_posts - 2 ∧
    playground_area p = 400 := by
  sorry


end aarons_playground_area_l4119_411963


namespace decimal_arithmetic_l4119_411903

theorem decimal_arithmetic : 0.5 - 0.03 + 0.007 = 0.477 := by
  sorry

end decimal_arithmetic_l4119_411903


namespace functional_equation_unique_solution_l4119_411979

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)

/-- The main theorem stating that the only function satisfying the equation is f(x) = x - 1 -/
theorem functional_equation_unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → ∀ x : ℝ, f x = x - 1 := by
  sorry

end functional_equation_unique_solution_l4119_411979


namespace preimage_of_2_neg4_l4119_411927

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem preimage_of_2_neg4 : f (-1, -3) = (2, -4) := by
  sorry

end preimage_of_2_neg4_l4119_411927


namespace propositions_truth_l4119_411968

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem propositions_truth : 
  (∀ a : ℝ, a^2 ≥ 0) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ ∧ f x₁ > f x₂) :=
by sorry

end propositions_truth_l4119_411968


namespace shop_profit_percentage_l4119_411928

/-- Calculates the total profit percentage for a shop selling two types of items -/
theorem shop_profit_percentage
  (cost_price_ratio_A : ℝ)
  (cost_price_ratio_B : ℝ)
  (quantity_A : ℕ)
  (quantity_B : ℕ)
  (price_A : ℝ)
  (price_B : ℝ)
  (h1 : cost_price_ratio_A = 0.95)
  (h2 : cost_price_ratio_B = 0.90)
  (h3 : quantity_A = 100)
  (h4 : quantity_B = 150)
  (h5 : price_A = 50)
  (h6 : price_B = 60) :
  let profit_A := quantity_A * price_A * (1 - cost_price_ratio_A)
  let profit_B := quantity_B * price_B * (1 - cost_price_ratio_B)
  let total_profit := profit_A + profit_B
  let total_cost := quantity_A * price_A * cost_price_ratio_A + quantity_B * price_B * cost_price_ratio_B
  let profit_percentage := (total_profit / total_cost) * 100
  ∃ ε > 0, |profit_percentage - 8.95| < ε :=
by
  sorry

end shop_profit_percentage_l4119_411928


namespace kindergarten_card_problem_l4119_411956

/-- Represents the distribution of cards among children in a kindergarten. -/
structure CardDistribution where
  ma_three : ℕ  -- Number of children with three "MA" cards
  ma_two : ℕ    -- Number of children with two "MA" cards and one "NY" card
  ny_two : ℕ    -- Number of children with two "NY" cards and one "MA" card
  ny_three : ℕ  -- Number of children with three "NY" cards

/-- The conditions given in the problem. -/
def problem_conditions (d : CardDistribution) : Prop :=
  d.ma_three + d.ma_two = 20 ∧
  d.ny_two + d.ny_three = 30 ∧
  d.ma_two + d.ny_two = 40

/-- The theorem stating that given the problem conditions, 
    the number of children with all three cards the same is 10. -/
theorem kindergarten_card_problem (d : CardDistribution) :
  problem_conditions d → d.ma_three + d.ny_three = 10 := by
  sorry


end kindergarten_card_problem_l4119_411956


namespace side_significant_digits_l4119_411989

-- Define the area of the square
def area : ℝ := 0.6400

-- Define the precision of the area measurement
def area_precision : ℝ := 0.0001

-- Define the function to calculate the number of significant digits
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem side_significant_digits :
  let side := Real.sqrt area
  count_significant_digits side = 4 := by sorry

end side_significant_digits_l4119_411989


namespace gcf_360_180_l4119_411917

theorem gcf_360_180 : Nat.gcd 360 180 = 180 := by
  sorry

end gcf_360_180_l4119_411917


namespace find_a_l4119_411993

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 < a^2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B (a : ℝ) : Set ℝ := A a ∩ B

-- State the theorem
theorem find_a : ∀ a : ℝ, A_intersect_B a = {x : ℝ | 1 < x ∧ x < 2} → a = 2 ∨ a = -2 := by
  sorry

end find_a_l4119_411993


namespace equation_solution_l4119_411972

theorem equation_solution : ∃ x : ℝ, 2 * (x + 3) = 5 * x ∧ x = 2 := by
  sorry

end equation_solution_l4119_411972


namespace skating_minutes_proof_l4119_411929

/-- The number of minutes Gage skated per day for the first 4 days -/
def minutes_per_day_first_4 : ℕ := 70

/-- The number of minutes Gage skated per day for the next 4 days -/
def minutes_per_day_next_4 : ℕ := 100

/-- The total number of days Gage skated -/
def total_days : ℕ := 9

/-- The desired average number of minutes skated per day -/
def desired_average : ℕ := 100

/-- The number of minutes Gage must skate on the ninth day to achieve the desired average -/
def minutes_on_ninth_day : ℕ := 220

theorem skating_minutes_proof :
  minutes_on_ninth_day = 
    total_days * desired_average - 
    (4 * minutes_per_day_first_4 + 4 * minutes_per_day_next_4) := by
  sorry

end skating_minutes_proof_l4119_411929


namespace inequality_solution_set_l4119_411915

-- Define the inequality
def inequality (x m : ℝ) : Prop :=
  |x + 1| + |x - 2| + m - 7 > 0

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  {x : ℝ | inequality x m}

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  solution_set m = Set.univ → m > 4 :=
by sorry

end inequality_solution_set_l4119_411915


namespace trigonometric_identity_l4119_411930

theorem trigonometric_identity (α β γ : Real) 
  (h : (Real.sin (β + γ) * Real.sin (γ + α)) / (Real.cos α * Real.cos γ) = 4/9) :
  (Real.sin (β + γ) * Real.sin (γ + α)) / (Real.cos (α + β + γ) * Real.cos γ) = 4/5 := by
  sorry

end trigonometric_identity_l4119_411930


namespace complex_sum_problem_l4119_411904

theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 5 → 
  t = -p - r → 
  (p + q * I) + (r + s * I) + (t + u * I) = 4 * I → 
  s + u = -1 := by
sorry

end complex_sum_problem_l4119_411904


namespace binary_remainder_by_eight_l4119_411932

/-- The remainder when 110111100101₂ is divided by 8 is 5 -/
theorem binary_remainder_by_eight : Nat.mod 0b110111100101 8 = 5 := by
  sorry

end binary_remainder_by_eight_l4119_411932


namespace max_distance_sin_cosin_l4119_411982

/-- The maximum distance between sin x and sin(π/2 - x) for any real x is √2 -/
theorem max_distance_sin_cosin (x : ℝ) : 
  ∃ (m : ℝ), ∀ (x : ℝ), |Real.sin x - Real.sin (π/2 - x)| ≤ m ∧ 
  ∃ (y : ℝ), |Real.sin y - Real.sin (π/2 - y)| = m ∧ 
  m = Real.sqrt 2 := by
sorry

end max_distance_sin_cosin_l4119_411982


namespace soldier_rearrangement_l4119_411948

theorem soldier_rearrangement (n : Nat) (h : n = 20 ∨ n = 21) :
  ∃ (d : ℝ), d = 10 * Real.sqrt 2 ∧
  (∀ (rearrangement : Fin n × Fin n → Fin n × Fin n),
    (∀ (i j : Fin n), (rearrangement (i, j) ≠ (i, j)) →
      Real.sqrt ((i.val - (rearrangement (i, j)).1.val)^2 +
                 (j.val - (rearrangement (i, j)).2.val)^2) ≥ d) →
    (∀ (i j : Fin n), ∃ (k l : Fin n), rearrangement (k, l) = (i, j))) ∧
  (∀ (d' : ℝ), d' > d →
    ¬∃ (rearrangement : Fin n × Fin n → Fin n × Fin n),
      (∀ (i j : Fin n), (rearrangement (i, j) ≠ (i, j)) →
        Real.sqrt ((i.val - (rearrangement (i, j)).1.val)^2 +
                   (j.val - (rearrangement (i, j)).2.val)^2) ≥ d') ∧
      (∀ (i j : Fin n), ∃ (k l : Fin n), rearrangement (k, l) = (i, j))) :=
by sorry

end soldier_rearrangement_l4119_411948


namespace complex_magnitude_theorem_l4119_411999

theorem complex_magnitude_theorem (z₁ z₂ z₃ : ℂ) (a b c : ℝ) 
  (h₁ : (z₁ / z₂ + z₂ / z₃ + z₃ / z₁).im = 0)
  (h₂ : Complex.abs z₁ = 1)
  (h₃ : Complex.abs z₂ = 1)
  (h₄ : Complex.abs z₃ = 1) :
  ∃ (x : ℝ), x = Complex.abs (a * z₁ + b * z₂ + c * z₃) ∧
    (x = Real.sqrt ((a + b)^2 + c^2) ∨
     x = Real.sqrt ((a + c)^2 + b^2) ∨
     x = Real.sqrt ((b + c)^2 + a^2)) :=
by sorry

end complex_magnitude_theorem_l4119_411999


namespace probability_problem_l4119_411958

theorem probability_problem (p_biology : ℚ) (p_no_chemistry : ℚ)
  (h1 : p_biology = 5/8)
  (h2 : p_no_chemistry = 1/2) :
  let p_no_biology := 1 - p_biology
  let p_neither := p_no_biology * p_no_chemistry
  (p_no_biology = 3/8) ∧ (p_neither = 3/16) := by
  sorry

end probability_problem_l4119_411958


namespace complex_division_result_l4119_411959

theorem complex_division_result : ∃ (i : ℂ), i * i = -1 ∧ (2 : ℂ) / (1 - i) = 1 + i := by
  sorry

end complex_division_result_l4119_411959


namespace tank_filling_solution_l4119_411978

/-- Represents the tank filling problem -/
def TankFillingProblem (tankCapacity : Real) (initialFillRatio : Real) 
  (fillingRate : Real) (drain1Rate : Real) (drain2Rate : Real) : Prop :=
  let remainingVolume := tankCapacity * (1 - initialFillRatio)
  let netFlowRate := fillingRate - drain1Rate - drain2Rate
  let timeToFill := remainingVolume / netFlowRate
  timeToFill = 6

/-- The theorem stating the solution to the tank filling problem -/
theorem tank_filling_solution :
  TankFillingProblem 1000 0.5 (1/2) (1/4) (1/6) := by
  sorry

#check tank_filling_solution

end tank_filling_solution_l4119_411978


namespace units_digit_of_sum_is_seven_l4119_411907

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_less_than_10 : hundreds < 10
  tens_less_than_10 : tens < 10
  units_less_than_10 : units < 10
  hundreds_not_zero : hundreds ≠ 0

/-- The condition that the hundreds digit is 3 less than twice the units digit -/
def hundreds_units_relation (n : ThreeDigitNumber) : Prop :=
  n.hundreds = 2 * n.units - 3

/-- The value of the three-digit number -/
def number_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed number -/
def reversed_number (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The theorem to be proved -/
theorem units_digit_of_sum_is_seven (n : ThreeDigitNumber) 
  (h : hundreds_units_relation n) : 
  (number_value n + reversed_number n) % 10 = 7 := by
  sorry

end units_digit_of_sum_is_seven_l4119_411907


namespace population_change_l4119_411952

theorem population_change (initial_population : ℕ) 
  (increase_rate : ℚ) (decrease_rate : ℚ) : 
  initial_population = 10000 →
  increase_rate = 20 / 100 →
  decrease_rate = 20 / 100 →
  (initial_population * (1 + increase_rate) * (1 - decrease_rate)).floor = 9600 := by
sorry

end population_change_l4119_411952


namespace cos_four_arccos_two_fifths_l4119_411916

theorem cos_four_arccos_two_fifths : 
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end cos_four_arccos_two_fifths_l4119_411916


namespace inverse_square_direct_cube_relation_l4119_411960

/-- Given that x varies inversely as the square of y and directly as the cube of z,
    prove that x = 64/243 when y = 6 and z = 4, given the initial conditions x = 1, y = 2, and z = 3. -/
theorem inverse_square_direct_cube_relation
  (k : ℚ)
  (h : ∀ (x y z : ℚ), x = k * z^3 / y^2)
  (h_init : 1 = k * 3^3 / 2^2) :
  k * 4^3 / 6^2 = 64/243 := by
  sorry

end inverse_square_direct_cube_relation_l4119_411960


namespace colored_graph_color_bound_l4119_411931

/-- A graph with colored edges satisfying certain properties -/
structure ColoredGraph where
  n : ℕ  -- number of vertices
  c : ℕ  -- number of colors
  edge_count : ℕ  -- number of edges
  edge_count_lower_bound : edge_count ≥ n^2 / 10
  no_incident_same_color : Bool  -- property that no two incident edges have the same color
  no_same_color_10_cycle : Bool  -- property that no cycles of size 10 have the same set of colors

/-- Main theorem: There exists a constant k such that c ≥ k * n^(8/5) for any colored graph satisfying the given properties -/
theorem colored_graph_color_bound (G : ColoredGraph) :
  ∃ (k : ℝ), G.c ≥ k * G.n^(8/5) := by
  sorry

end colored_graph_color_bound_l4119_411931


namespace white_area_of_sign_l4119_411940

/-- Represents a block letter in the sign --/
structure BlockLetter where
  width : ℕ
  height : ℕ
  stroke_width : ℕ
  covered_area : ℕ

/-- Represents the sign --/
structure Sign where
  width : ℕ
  height : ℕ
  letters : List BlockLetter

def m_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def a_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def t_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 24
}

def h_letter : BlockLetter := {
  width := 7,
  height := 8,
  stroke_width := 2,
  covered_area := 40
}

def math_sign : Sign := {
  width := 28,
  height := 8,
  letters := [m_letter, a_letter, t_letter, h_letter]
}

theorem white_area_of_sign (s : Sign) : 
  s.width * s.height - (s.letters.map BlockLetter.covered_area).sum = 80 :=
by sorry

end white_area_of_sign_l4119_411940


namespace cosine_graph_minimum_l4119_411922

theorem cosine_graph_minimum (c : ℝ) (h1 : c > 0) : 
  (∀ x : ℝ, 3 * Real.cos (5 * x + c) ≥ 3 * Real.cos c) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo 0 ε, 3 * Real.cos (5 * x + c) > 3 * Real.cos c) → 
  c = Real.pi := by
sorry

end cosine_graph_minimum_l4119_411922


namespace share_calculation_l4119_411914

theorem share_calculation (total_amount : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) : 
  total_amount = 15800 → 
  ratio_a = 5 →
  ratio_b = 9 →
  ratio_c = 6 →
  ratio_d = 5 →
  (ratio_a * total_amount / (ratio_a + ratio_b + ratio_c + ratio_d) + 
   ratio_c * total_amount / (ratio_a + ratio_b + ratio_c + ratio_d)) = 6952 := by
sorry

end share_calculation_l4119_411914


namespace ratio_to_thirteen_l4119_411902

theorem ratio_to_thirteen : ∃ x : ℚ, (5 : ℚ) / 1 = x / 13 ∧ x = 65 := by
  sorry

end ratio_to_thirteen_l4119_411902


namespace negative_inequality_l4119_411900

theorem negative_inequality (x y : ℝ) (h : x < y) : -x > -y := by
  sorry

end negative_inequality_l4119_411900


namespace vertex_determines_parameters_l4119_411991

def quadratic_function (h k : ℝ) (x : ℝ) : ℝ := -3 * (x - h)^2 + k

theorem vertex_determines_parameters (h k : ℝ) :
  (∀ x, quadratic_function h k x = quadratic_function 1 (-2) x) →
  h = 1 ∧ k = -2 := by
  sorry

end vertex_determines_parameters_l4119_411991


namespace moon_speed_conversion_l4119_411962

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 0.2

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 720 := by
  sorry

end moon_speed_conversion_l4119_411962


namespace stock_price_increase_l4119_411941

theorem stock_price_increase (initial_price : ℝ) (first_year_increase : ℝ) : 
  initial_price > 0 →
  first_year_increase > 0 →
  initial_price * (1 + first_year_increase / 100) * 0.75 * 1.2 = initial_price * 1.08 →
  first_year_increase = 20 := by
sorry

end stock_price_increase_l4119_411941


namespace exist_three_integers_sum_zero_thirteenth_powers_square_l4119_411945

theorem exist_three_integers_sum_zero_thirteenth_powers_square :
  ∃ (a b c : ℤ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧  -- nonzero
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- pairwise distinct
    a + b + c = 0 ∧          -- sum is zero
    ∃ (n : ℕ), a^13 + b^13 + c^13 = n^2  -- sum of 13th powers is a perfect square
  := by sorry

end exist_three_integers_sum_zero_thirteenth_powers_square_l4119_411945


namespace jerrys_average_score_l4119_411986

theorem jerrys_average_score (current_total : ℝ) (desired_average : ℝ) (fourth_test_score : ℝ) :
  (current_total / 3 + 2 = desired_average) →
  (current_total + fourth_test_score) / 4 = desired_average →
  fourth_test_score = 98 →
  current_total / 3 = 90 :=
by sorry

end jerrys_average_score_l4119_411986


namespace diophantine_equation_solutions_l4119_411926

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ × ℤ) := {(x, y, z) | 5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30}
  S = {(1, 5, 0), (1, -5, 0), (-1, 5, 0), (-1, -5, 0)} := by
  sorry

end diophantine_equation_solutions_l4119_411926


namespace smallest_angle_in_triangle_l4119_411974

theorem smallest_angle_in_triangle (x y z : ℝ) (hx : x = 60) (hy : y = 70) 
  (hsum : x + y + z = 180) : min x (min y z) = 50 := by
  sorry

end smallest_angle_in_triangle_l4119_411974


namespace weekly_running_distance_l4119_411970

/-- Calculates the total distance run in a week given the number of days, hours per day, and speed. -/
def total_distance_run (days_per_week : ℕ) (hours_per_day : ℝ) (speed_mph : ℝ) : ℝ :=
  days_per_week * hours_per_day * speed_mph

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem weekly_running_distance :
  total_distance_run 5 1.5 8 = 60 := by
  sorry

end weekly_running_distance_l4119_411970


namespace square_difference_49_16_l4119_411910

theorem square_difference_49_16 : 49^2 - 16^2 = 2145 := by
  sorry

end square_difference_49_16_l4119_411910


namespace x_value_proof_l4119_411911

theorem x_value_proof (x : ℝ) : -(-(-(-x))) = -4 → x = -4 := by
  sorry

end x_value_proof_l4119_411911


namespace john_ate_three_cookies_l4119_411937

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John has left -/
def cookies_left : ℕ := 21

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := dozens_bought * dozen - cookies_left

theorem john_ate_three_cookies : cookies_eaten = 3 := by
  sorry

end john_ate_three_cookies_l4119_411937


namespace debt_installments_l4119_411901

theorem debt_installments (first_payment : ℕ) (additional_amount : ℕ) (average_payment : ℕ) : 
  let n := (12 * first_payment + 780) / 15
  let remaining_payment := first_payment + additional_amount
  12 * first_payment + (n - 12) * remaining_payment = n * average_payment →
  n = 52 :=
by
  sorry

#check debt_installments 410 65 460

end debt_installments_l4119_411901


namespace geometric_sequence_common_ratio_l4119_411998

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1/4) :
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end geometric_sequence_common_ratio_l4119_411998


namespace mean_proportional_problem_l4119_411987

theorem mean_proportional_problem (n : ℝ) : (156 : ℝ) ^ 2 = n * 104 → n = 234 := by
  sorry

end mean_proportional_problem_l4119_411987


namespace triangle_area_tripled_sides_l4119_411935

/-- Given a triangle with sides a and b and included angle θ,
    if we triple the sides to 3a and 3b while keeping θ unchanged,
    then the new area A' is 9 times the original area A. -/
theorem triangle_area_tripled_sides (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let A := (a * b * Real.sin θ) / 2
  let A' := (3 * a * 3 * b * Real.sin θ) / 2
  A' = 9 * A := by sorry

end triangle_area_tripled_sides_l4119_411935


namespace equation_solutions_l4119_411921

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 25 = 0 ↔ x = 5/2 ∨ x = -5/2) ∧
  (∀ x : ℝ, (x + 1)^3 = -27 ↔ x = -4) := by
  sorry

end equation_solutions_l4119_411921


namespace frisbee_sales_minimum_receipts_l4119_411947

theorem frisbee_sales_minimum_receipts :
  ∀ (x y : ℕ),
  x + y = 64 →
  y ≥ 8 →
  3 * x + 4 * y ≥ 200 :=
by
  sorry

end frisbee_sales_minimum_receipts_l4119_411947


namespace pizza_slices_with_both_toppings_l4119_411996

-- Define the total number of slices
def total_slices : ℕ := 24

-- Define the number of slices with pepperoni
def pepperoni_slices : ℕ := 12

-- Define the number of slices with mushrooms
def mushroom_slices : ℕ := 14

-- Define the number of vegetarian slices
def vegetarian_slices : ℕ := 4

-- Theorem to prove
theorem pizza_slices_with_both_toppings :
  ∃ n : ℕ, 
    -- Every slice has at least one condition met
    (n + (pepperoni_slices - n) + (mushroom_slices - n) + vegetarian_slices = total_slices) ∧
    -- n is the number of slices with both pepperoni and mushrooms
    n = 6 := by
  sorry

end pizza_slices_with_both_toppings_l4119_411996


namespace same_terminal_side_angle_angle_in_range_same_terminal_side_750_l4119_411923

theorem same_terminal_side_angle : ℤ → ℝ → ℝ
  | k, α => k * 360 + α

theorem angle_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

theorem same_terminal_side_750 :
  ∃ (θ : ℝ), angle_in_range θ ∧ ∃ (k : ℤ), same_terminal_side_angle k θ = 750 ∧ θ = 30 :=
sorry

end same_terminal_side_angle_angle_in_range_same_terminal_side_750_l4119_411923


namespace encoded_CDE_is_174_l4119_411957

/-- Represents the encoding of a base-6 digit --/
inductive Digit
| A | B | C | D | E | F

/-- Converts a Digit to its corresponding base-6 value --/
def digit_to_base6 : Digit → Nat
| Digit.A => 5
| Digit.B => 0
| Digit.C => 4
| Digit.D => 5
| Digit.E => 0
| Digit.F => 1

/-- Converts a base-6 number represented as a list of Digits to base-10 --/
def base6_to_base10 (digits : List Digit) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + (digit_to_base6 d) * (6^i)) 0

/-- The main theorem to prove --/
theorem encoded_CDE_is_174 :
  base6_to_base10 [Digit.C, Digit.D, Digit.E] = 174 :=
by sorry

end encoded_CDE_is_174_l4119_411957


namespace intersection_polyhedron_volume_l4119_411988

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The polyhedron formed by the intersection of a regular tetrahedron with its image under symmetry relative to the midpoint of its height -/
def IntersectionPolyhedron (t : RegularTetrahedron) : Set (Fin 3 → ℝ) :=
  sorry

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Theorem: The volume of the intersection polyhedron is (a^3 * √2) / 54 -/
theorem intersection_polyhedron_volume (t : RegularTetrahedron) :
    volume (IntersectionPolyhedron t) = (t.edge_length^3 * Real.sqrt 2) / 54 :=
  sorry

end intersection_polyhedron_volume_l4119_411988


namespace jerry_medical_bills_l4119_411933

/-- The amount Jerry is claiming for medical bills -/
def medical_bills : ℝ := sorry

/-- Jerry's annual salary -/
def annual_salary : ℝ := 50000

/-- Number of years of lost salary -/
def years_of_lost_salary : ℕ := 30

/-- Total lost salary -/
def total_lost_salary : ℝ := annual_salary * years_of_lost_salary

/-- Punitive damages multiplier -/
def punitive_multiplier : ℕ := 3

/-- Percentage of claim Jerry receives -/
def claim_percentage : ℝ := 0.8

/-- Total amount Jerry receives -/
def total_received : ℝ := 5440000

/-- Theorem stating the amount of medical bills Jerry is claiming -/
theorem jerry_medical_bills :
  claim_percentage * (total_lost_salary + medical_bills + 
    punitive_multiplier * (total_lost_salary + medical_bills)) = total_received ∧
  medical_bills = 200000 := by sorry

end jerry_medical_bills_l4119_411933


namespace smallest_number_divisible_by_all_l4119_411971

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 6297 ∧ ∀ m : ℕ, m < 6297 → ¬is_divisible_by_all m :=
by sorry

end smallest_number_divisible_by_all_l4119_411971


namespace average_of_fifths_and_tenths_l4119_411942

/-- The average of two rational numbers -/
def average (a b : ℚ) : ℚ := (a + b) / 2

/-- Theorem: If the average of 1/5 and 1/10 is 1/x, then x = 20/3 -/
theorem average_of_fifths_and_tenths (x : ℚ) :
  average (1/5) (1/10) = 1/x → x = 20/3 := by
  sorry

end average_of_fifths_and_tenths_l4119_411942


namespace curve_point_when_a_is_one_curve_passes_through_fixed_point_l4119_411912

-- Define the curve equation
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0

-- Theorem for case a = 1
theorem curve_point_when_a_is_one :
  ∀ x y : ℝ, curve x y 1 ↔ x = 1 ∧ y = 1 :=
sorry

-- Theorem for case a ≠ 1
theorem curve_passes_through_fixed_point :
  ∀ a : ℝ, a ≠ 1 → curve 1 1 a :=
sorry

end curve_point_when_a_is_one_curve_passes_through_fixed_point_l4119_411912


namespace shell_collection_division_l4119_411985

theorem shell_collection_division (lino_morning : ℝ) (maria_morning : ℝ) 
  (lino_afternoon : ℝ) (maria_afternoon : ℝ) 
  (h1 : lino_morning = 292.5) 
  (h2 : maria_morning = 375.25)
  (h3 : lino_afternoon = 324.75)
  (h4 : maria_afternoon = 419.3) : 
  (lino_morning + lino_afternoon + maria_morning + maria_afternoon) / 2 = 705.9 := by
  sorry

end shell_collection_division_l4119_411985


namespace min_value_a_plus_2b_min_value_equals_7_plus_2sqrt6_l4119_411980

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 3/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_equals_7_plus_2sqrt6 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  a + 2*b = 7 + 2*Real.sqrt 6 :=
by sorry

end min_value_a_plus_2b_min_value_equals_7_plus_2sqrt6_l4119_411980


namespace fifth_page_stickers_l4119_411905

def sticker_sequence (n : ℕ) : ℕ := 8 * n

theorem fifth_page_stickers : sticker_sequence 5 = 40 := by
  sorry

end fifth_page_stickers_l4119_411905


namespace total_shaded_area_is_71_l4119_411990

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  r.width * r.height

theorem total_shaded_area_is_71 (rect1 rect2 overlap : Rectangle)
    (h1 : rect1.width = 4 ∧ rect1.height = 12)
    (h2 : rect2.width = 5 ∧ rect2.height = 7)
    (h3 : overlap.width = 3 ∧ overlap.height = 4) :
    area rect1 + area rect2 - area overlap = 71 := by
  sorry

#check total_shaded_area_is_71

end total_shaded_area_is_71_l4119_411990


namespace starters_with_twin_l4119_411983

def total_players : ℕ := 16
def starters : ℕ := 6
def twins : ℕ := 2

theorem starters_with_twin (total_players starters twins : ℕ) :
  total_players = 16 →
  starters = 6 →
  twins = 2 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - twins) starters) = 5005 := by
  sorry

end starters_with_twin_l4119_411983


namespace absolute_value_equation_solution_l4119_411973

theorem absolute_value_equation_solution (x : ℝ) : 
  |3*x - 2| + |3*x + 1| = 3 ↔ x = -2/3 ∨ (-1/3 < x ∧ x ≤ 2/3) :=
sorry

end absolute_value_equation_solution_l4119_411973


namespace min_omega_value_l4119_411934

theorem min_omega_value (f : ℝ → ℝ) (ω φ : ℝ) : 
  (∀ x, f x = Real.sin (ω * x + φ)) →
  ω > 0 →
  abs φ < π / 2 →
  f 0 = 1 / 2 →
  (∀ x, f x ≤ f (π / 12)) →
  ω ≥ 4 ∧ (∀ ω', ω' > 0 ∧ ω' < 4 → 
    ∃ φ', abs φ' < π / 2 ∧ 
    Real.sin φ' = 1 / 2 ∧ 
    ∃ x, Real.sin (ω' * x + φ') > Real.sin (ω' * π / 12 + φ')) :=
by sorry

end min_omega_value_l4119_411934


namespace not_always_same_digit_sum_l4119_411909

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), ∃ (k : ℕ), sum_of_digits (N + k * M) ≠ sum_of_digits N :=
sorry

end not_always_same_digit_sum_l4119_411909


namespace inequality_relation_l4119_411951

theorem inequality_relation : 
  ∃ (x : ℝ), (x^2 - x - 6 > 0 ∧ x ≥ -5) ∧ 
  ∀ (y : ℝ), y < -5 → y^2 - y - 6 > 0 :=
by sorry

end inequality_relation_l4119_411951


namespace find_a_range_of_m_l4119_411955

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1: Prove that a = 2
theorem find_a (a : ℝ) : 
  (∀ x, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 := by sorry

-- Theorem 2: Prove the range of m
theorem range_of_m : 
  ∀ x, f 2 x + f 2 (x + 5) ≥ 5 ∧ 
  ∀ ε > 0, ∃ x, f 2 x + f 2 (x + 5) < 5 + ε := by sorry

end find_a_range_of_m_l4119_411955


namespace camping_trip_purchases_l4119_411946

/-- Given Rebecca's camping trip purchases, prove the difference between water bottles and tent stakes --/
theorem camping_trip_purchases (total_items tent_stakes drink_mix water_bottles : ℕ) : 
  total_items = 22 →
  tent_stakes = 4 →
  drink_mix = 3 * tent_stakes →
  total_items = tent_stakes + drink_mix + water_bottles →
  water_bottles - tent_stakes = 2 := by
  sorry

end camping_trip_purchases_l4119_411946


namespace car_average_speed_l4119_411919

/-- Given a car that travels 80 km in the first hour and 40 km in the second hour,
    prove that its average speed is 60 km/h. -/
theorem car_average_speed (distance_first_hour : ℝ) (distance_second_hour : ℝ)
    (h1 : distance_first_hour = 80)
    (h2 : distance_second_hour = 40) :
    (distance_first_hour + distance_second_hour) / 2 = 60 := by
  sorry

end car_average_speed_l4119_411919


namespace regression_estimate_l4119_411944

/-- Represents a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Represents a data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Parameters of the regression problem -/
structure RegressionProblem where
  original_regression : LinearRegression
  original_mean_x : ℝ
  removed_points : List DataPoint
  new_slope : ℝ

theorem regression_estimate (problem : RegressionProblem) :
  let new_intercept := problem.original_regression.intercept +
    problem.original_regression.slope * problem.original_mean_x -
    problem.new_slope * problem.original_mean_x
  let new_regression := LinearRegression.mk problem.new_slope new_intercept
  let estimate_at_6 := new_regression.slope * 6 + new_regression.intercept
  problem.original_regression = LinearRegression.mk 1.5 1 →
  problem.original_mean_x = 2 →
  problem.removed_points = [DataPoint.mk 2.6 2.8, DataPoint.mk 1.4 5.2] →
  problem.new_slope = 1.4 →
  estimate_at_6 = 9.6 := by
  sorry

end regression_estimate_l4119_411944


namespace business_profit_calculation_l4119_411918

def business_profit (a_investment b_investment total_profit : ℚ) : ℚ :=
  let total_investment := a_investment + b_investment
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let a_share_ratio := a_investment / total_investment
  let a_share := a_share_ratio * remaining_profit
  management_fee + a_share

theorem business_profit_calculation :
  business_profit 3500 1500 9600 = 7008 :=
by sorry

end business_profit_calculation_l4119_411918


namespace correct_operation_l4119_411966

theorem correct_operation (x : ℝ) : 4 * x^2 * (3 * x) = 12 * x^3 := by
  sorry

end correct_operation_l4119_411966


namespace mira_total_distance_l4119_411984

/-- Mira's jogging schedule for five days -/
structure JoggingSchedule where
  monday_speed : ℝ
  monday_time : ℝ
  tuesday_speed : ℝ
  tuesday_time : ℝ
  wednesday_speed : ℝ
  wednesday_time : ℝ
  thursday_speed : ℝ
  thursday_time : ℝ
  friday_speed : ℝ
  friday_time : ℝ

/-- Calculate the total distance jogged given a schedule -/
def total_distance (schedule : JoggingSchedule) : ℝ :=
  schedule.monday_speed * schedule.monday_time +
  schedule.tuesday_speed * schedule.tuesday_time +
  schedule.wednesday_speed * schedule.wednesday_time +
  schedule.thursday_speed * schedule.thursday_time +
  schedule.friday_speed * schedule.friday_time

/-- Mira's actual jogging schedule -/
def mira_schedule : JoggingSchedule := {
  monday_speed := 4
  monday_time := 2
  tuesday_speed := 5
  tuesday_time := 1.5
  wednesday_speed := 6
  wednesday_time := 2
  thursday_speed := 5
  thursday_time := 2.5
  friday_speed := 3
  friday_time := 1
}

/-- Theorem stating that Mira jogs a total of 43 miles in five days -/
theorem mira_total_distance : total_distance mira_schedule = 43 := by
  sorry

end mira_total_distance_l4119_411984


namespace sequence_max_value_l4119_411994

theorem sequence_max_value (n : ℤ) : -2 * n^2 + 29 * n + 3 ≤ 108 := by
  sorry

end sequence_max_value_l4119_411994


namespace cos_zeros_range_l4119_411908

theorem cos_zeros_range (ω : ℝ) (h_pos : ω > 0) : 
  (∃ (z₁ z₂ z₃ : ℝ), z₁ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₂ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₃ ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi) ∧ 
                      z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
                      Real.cos (ω * z₁ - Real.pi / 6) = 0 ∧
                      Real.cos (ω * z₂ - Real.pi / 6) = 0 ∧
                      Real.cos (ω * z₃ - Real.pi / 6) = 0 ∧
                      (∀ z ∈ Set.Ioo (7 * Real.pi / (6 * ω)) (2 * Real.pi), 
                        Real.cos (ω * z - Real.pi / 6) = 0 → z = z₁ ∨ z = z₂ ∨ z = z₃)) →
  11 / 6 ≤ ω ∧ ω < 7 / 3 :=
by sorry

end cos_zeros_range_l4119_411908


namespace custom_mul_equality_l4119_411950

/-- Custom multiplication operation for real numbers -/
def custom_mul (a b : ℝ) : ℝ := (a - b^3)^2

/-- Theorem stating the equality for the given expression -/
theorem custom_mul_equality (x y : ℝ) :
  custom_mul ((x - y)^2) ((y^2 - x^2)^2) = ((x - y)^2 - (y^4 - 2*x^2*y^2 + x^4)^3)^2 := by
  sorry

end custom_mul_equality_l4119_411950


namespace power_function_increasing_condition_l4119_411961

theorem power_function_increasing_condition (m : ℝ) : 
  (m^2 - m - 1 = 1) ∧ (m^2 + m - 3 > 0) ↔ m = 2 :=
by sorry

end power_function_increasing_condition_l4119_411961


namespace sum_of_repeating_decimals_l4119_411943

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem sum_of_repeating_decimals :
  SingleDigitRepeatingDecimal 0 1 + TwoDigitRepeatingDecimal 0 1 = 4 / 33 :=
by sorry

end sum_of_repeating_decimals_l4119_411943


namespace restaurant_group_size_l4119_411965

theorem restaurant_group_size (adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) (children : ℕ) : 
  adults = 2 →
  meal_cost = 3 →
  total_bill = 21 →
  children * meal_cost + adults * meal_cost = total_bill →
  children = 5 := by
sorry

end restaurant_group_size_l4119_411965


namespace intersection_of_M_and_N_l4119_411949

def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 5} := by
  sorry

end intersection_of_M_and_N_l4119_411949


namespace tims_balloons_l4119_411976

def dans_balloons : ℝ := 29.0
def dans_multiple : ℝ := 7.0

theorem tims_balloons : ⌊dans_balloons / dans_multiple⌋ = 4 := by
  sorry

end tims_balloons_l4119_411976


namespace quadratic_roots_distinct_and_sum_constant_l4119_411975

/-- Represents the quadratic equation -3(x-1)^2 + m = 0 --/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  -3 * (x - 1)^2 + m = 0

/-- The discriminant of the quadratic equation --/
def discriminant (m : ℝ) : ℝ :=
  12 * m

theorem quadratic_roots_distinct_and_sum_constant (m : ℝ) (h : m > 0) :
  ∃ (x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    quadratic_equation m x₁ ∧
    quadratic_equation m x₂ ∧
    x₁ + x₂ = 2 :=
sorry

end quadratic_roots_distinct_and_sum_constant_l4119_411975


namespace divisibility_by_1897_l4119_411992

theorem divisibility_by_1897 (n : ℕ) : 
  (1897 : ℤ) ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end divisibility_by_1897_l4119_411992


namespace soccer_team_probability_l4119_411925

theorem soccer_team_probability (total_players defenders : ℕ) 
  (h1 : total_players = 12)
  (h2 : defenders = 6) :
  (Nat.choose defenders 2 : ℚ) / (Nat.choose total_players 2) = 5 / 22 := by
  sorry

end soccer_team_probability_l4119_411925


namespace rectangular_plot_breadth_l4119_411913

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 2700 →
  width = 30 := by
sorry

end rectangular_plot_breadth_l4119_411913


namespace exists_question_with_different_answers_l4119_411954

/-- Represents a person who always tells the truth -/
structure TruthTeller where
  name : String
  always_truthful : Bool

/-- Represents a question that can be asked -/
inductive Question where
  | count_questions : Question

/-- Represents the state of the conversation -/
structure ConversationState where
  questions_asked : Nat

/-- The answer given by a TruthTeller to a Question in a given ConversationState -/
def answer (person : TruthTeller) (q : Question) (state : ConversationState) : Nat :=
  match q with
  | Question.count_questions => state.questions_asked

/-- Theorem stating that there exists a question that can have different truthful answers when asked twice -/
theorem exists_question_with_different_answers (ilya : TruthTeller) 
    (h_truthful : ilya.always_truthful = true) :
    ∃ (q : Question) (s1 s2 : ConversationState), 
      s1 ≠ s2 ∧ answer ilya q s1 ≠ answer ilya q s2 := by
  sorry


end exists_question_with_different_answers_l4119_411954


namespace book_pages_book_pages_proof_l4119_411997

theorem book_pages : ℝ → Prop :=
  fun x => 
    let day1_read := x / 4 + 17
    let day1_remain := x - day1_read
    let day2_read := day1_remain / 3 + 20
    let day2_remain := day1_remain - day2_read
    let day3_read := day2_remain / 2 + 23
    let day3_remain := day2_remain - day3_read
    day3_remain = 70 → x = 394

-- The proof goes here
theorem book_pages_proof : ∃ x : ℝ, book_pages x := by
  sorry

end book_pages_book_pages_proof_l4119_411997


namespace christen_peeled_24_l4119_411936

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initialPile : ℕ
  homerRate : ℕ
  christenJoinTime : ℕ
  christenRate : ℕ
  alexExtra : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledCount (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 24 potatoes in the given scenario -/
theorem christen_peeled_24 (scenario : PotatoPeeling) 
  (h1 : scenario.initialPile = 60)
  (h2 : scenario.homerRate = 4)
  (h3 : scenario.christenJoinTime = 6)
  (h4 : scenario.christenRate = 6)
  (h5 : scenario.alexExtra = 2) :
  christenPeeledCount scenario = 24 := by
  sorry

end christen_peeled_24_l4119_411936


namespace automobile_distance_l4119_411953

/-- Proves that an automobile traveling a/4 feet in r seconds will cover 20a/r yards in 4 minutes if it maintains the same rate. -/
theorem automobile_distance (a r : ℝ) (ha : a > 0) (hr : r > 0) : 
  let rate_feet_per_second : ℝ := a / (4 * r)
  let rate_yards_per_second : ℝ := rate_feet_per_second / 3
  let time_in_seconds : ℝ := 4 * 60
  rate_yards_per_second * time_in_seconds = 20 * a / r :=
by sorry

end automobile_distance_l4119_411953


namespace factorial_ratio_l4119_411995

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end factorial_ratio_l4119_411995


namespace school_store_pricing_l4119_411977

/-- Given the cost of pencils and notebooks in a school store, 
    calculate the cost of a specific combination. -/
theorem school_store_pricing 
  (pencil_cost notebook_cost : ℚ) 
  (h1 : 6 * pencil_cost + 6 * notebook_cost = 390/100)
  (h2 : 8 * pencil_cost + 4 * notebook_cost = 328/100) : 
  20 * pencil_cost + 14 * notebook_cost = 1012/100 := by
  sorry

end school_store_pricing_l4119_411977


namespace company_production_l4119_411938

/-- Calculates the total number of parts produced by a company given specific production conditions. -/
def totalPartsProduced (initialPartsPerDay : ℕ) (initialDays : ℕ) (increasedPartsPerDay : ℕ) (extraParts : ℕ) : ℕ :=
  let totalInitialParts := initialPartsPerDay * initialDays
  let increasedProduction := initialPartsPerDay + increasedPartsPerDay
  let additionalDays := extraParts / increasedPartsPerDay
  let totalIncreasedParts := increasedProduction * additionalDays
  totalInitialParts + totalIncreasedParts

/-- Theorem stating that under given conditions, the company produces 1107 parts. -/
theorem company_production : 
  totalPartsProduced 40 3 7 150 = 1107 := by
  sorry

#eval totalPartsProduced 40 3 7 150

end company_production_l4119_411938
