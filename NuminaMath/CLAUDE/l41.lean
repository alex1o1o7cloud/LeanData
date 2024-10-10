import Mathlib

namespace infinite_binary_sequences_and_powerset_cardinality_l41_4184

/-- The type of infinite binary sequences -/
def InfiniteBinarySequence := ℕ → Fin 2

/-- The cardinality of the continuum -/
def ContinuumCardinality := Cardinal.mk (Set ℝ)

theorem infinite_binary_sequences_and_powerset_cardinality :
  (Cardinal.mk (Set InfiniteBinarySequence) = ContinuumCardinality) ∧
  (Cardinal.mk (Set (Set ℕ)) = ContinuumCardinality) := by
  sorry

end infinite_binary_sequences_and_powerset_cardinality_l41_4184


namespace team_cost_comparison_l41_4181

/-- The cost calculation for Team A and Team B based on the number of people and ticket price --/
def cost_comparison (n : ℕ+) (x : ℝ) : Prop :=
  let cost_A := x + (3/4) * x * (n - 1)
  let cost_B := (4/5) * x * n
  (n = 5 → cost_A = cost_B) ∧
  (n > 5 → cost_A < cost_B) ∧
  (n < 5 → cost_A > cost_B)

/-- Theorem stating the cost comparison between Team A and Team B --/
theorem team_cost_comparison (n : ℕ+) (x : ℝ) (hx : x > 0) :
  cost_comparison n x := by
  sorry

end team_cost_comparison_l41_4181


namespace triangle_properties_l41_4127

/-- Given a triangle ABC with the following properties:
    1. f(x) = sin(2x + B) + √3 cos(2x + B) is an even function
    2. b = f(π/12)
    3. a = 3
    Prove that b = √3 and the area S of triangle ABC is either (3√3)/2 or (3√3)/4 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (∀ x, Real.sin (2 * x + B) + Real.sqrt 3 * Real.cos (2 * x + B) =
        Real.sin (2 * -x + B) + Real.sqrt 3 * Real.cos (2 * -x + B)) →
  b = Real.sin (2 * (π / 12) + B) + Real.sqrt 3 * Real.cos (2 * (π / 12) + B) →
  a = 3 →
  b = Real.sqrt 3 ∧ (
    (1/2 * a * b = (3 * Real.sqrt 3) / 2) ∨
    (1/2 * a * b = (3 * Real.sqrt 3) / 4)
  ) := by sorry

end triangle_properties_l41_4127


namespace purely_imaginary_complex_number_l41_4119

theorem purely_imaginary_complex_number (a : ℝ) :
  let z := (a + 2 * Complex.I) / (3 - 4 * Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 8/3 := by
  sorry

end purely_imaginary_complex_number_l41_4119


namespace new_years_appetizer_l41_4100

/-- The number of bags of chips Alex bought for his New Year's Eve appetizer -/
def num_bags : ℕ := 3

/-- The cost of each bag of chips in dollars -/
def cost_per_bag : ℚ := 1

/-- The cost of creme fraiche in dollars -/
def cost_creme_fraiche : ℚ := 5

/-- The cost of caviar in dollars -/
def cost_caviar : ℚ := 73

/-- The total cost per person in dollars -/
def cost_per_person : ℚ := 27

theorem new_years_appetizer :
  (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_bags = cost_per_person :=
by
  sorry

end new_years_appetizer_l41_4100


namespace intersection_perpendicular_points_l41_4131

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def l (x y m : ℝ) : Prop := y = 2 * x + m

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem intersection_perpendicular_points (m : ℝ) : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    l x₁ y₁ m ∧ l x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    perpendicular x₁ y₁ x₂ y₂ ↔ 
    m = 2 ∨ m = -2 := by sorry

end intersection_perpendicular_points_l41_4131


namespace nested_fraction_simplification_l41_4177

theorem nested_fraction_simplification :
  2 + 3 / (4 + 5 / 6) = 76 / 29 := by sorry

end nested_fraction_simplification_l41_4177


namespace sufficient_but_not_necessary_l41_4141

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 := by
  sorry

end sufficient_but_not_necessary_l41_4141


namespace solve_for_y_l41_4179

theorem solve_for_y (x y : ℝ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 := by
  sorry

end solve_for_y_l41_4179


namespace stewart_farm_horse_food_l41_4142

/-- The Stewart farm problem -/
theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (sheep_to_horse_ratio : ℚ) : 
  sheep_count = 8 →
  total_horse_food = 12880 →
  sheep_to_horse_ratio = 1 / 7 →
  (total_horse_food : ℚ) / ((sheep_count : ℚ) / sheep_to_horse_ratio) = 230 := by
  sorry

end stewart_farm_horse_food_l41_4142


namespace min_sum_of_equal_powers_l41_4114

theorem min_sum_of_equal_powers (x y z : ℕ+) (h : 2^(x:ℕ) = 5^(y:ℕ) ∧ 5^(y:ℕ) = 6^(z:ℕ)) :
  (x:ℕ) + (y:ℕ) + (z:ℕ) ≥ 26 :=
by sorry

end min_sum_of_equal_powers_l41_4114


namespace circle_center_and_sum_l41_4160

/-- The equation of a circle in the form x² + y² = ax + by + c -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

def circle_equation : CircleEquation :=
  { a := 6, b := -10, c := 9 }

theorem circle_center_and_sum (eq : CircleEquation) :
  ∃ (center : CircleCenter), 
    center.x = eq.a / 2 ∧
    center.y = -eq.b / 2 ∧
    center.x + center.y = -2 := by
  sorry

end circle_center_and_sum_l41_4160


namespace cylinder_volume_scale_l41_4153

/-- Given a cylinder with volume V, radius r, and height h, 
    if the radius is tripled and the height is quadrupled, 
    then the new volume V' is 36 times the original volume V. -/
theorem cylinder_volume_scale (V r h : ℝ) (h1 : V = π * r^2 * h) : 
  let V' := π * (3*r)^2 * (4*h)
  V' = 36 * V := by
sorry

end cylinder_volume_scale_l41_4153


namespace action_figure_cost_l41_4189

theorem action_figure_cost 
  (current : ℕ) 
  (total : ℕ) 
  (cost : ℚ) : 
  current = 3 → 
  total = 8 → 
  cost = 30 → 
  (cost / (total - current) : ℚ) = 6 := by
sorry

end action_figure_cost_l41_4189


namespace proposition_b_proposition_c_proposition_d_l41_4146

-- Define the types for planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (m n : Set (ℝ × ℝ × ℝ))

-- Define the perpendicular and parallel relations
def perpendicular (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def parallel (a b : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the subset relation
def subset (a b : Set (ℝ × ℝ × ℝ)) : Prop := ∀ x, x ∈ a → x ∈ b

-- Define the angle between a line and a plane
def angle (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem B
theorem proposition_b (h1 : perpendicular m α) (h2 : parallel n α) :
  perpendicular m n := sorry

-- Theorem C
theorem proposition_c (h1 : parallel α β) (h2 : subset m α) :
  parallel m β := sorry

-- Theorem D
theorem proposition_d (h1 : parallel m n) (h2 : parallel α β) :
  angle m α = angle n β := sorry

end proposition_b_proposition_c_proposition_d_l41_4146


namespace min_value_7x_5y_min_value_achieved_min_value_is_7_plus_2sqrt6_l41_4196

theorem min_value_7x_5y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (2 * x + y) + 4 / (x + y) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / (2 * a + b) + 4 / (a + b) = 2 → 7 * x + 5 * y ≤ 7 * a + 5 * b :=
by sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (2 * x + y) + 4 / (x + y) = 2) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (2 * a + b) + 4 / (a + b) = 2 ∧ 7 * a + 5 * b = 7 + 2 * Real.sqrt 6 :=
by sorry

theorem min_value_is_7_plus_2sqrt6 :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / (2 * x + y) + 4 / (x + y) = 2 ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (2 * a + b) + 4 / (a + b) = 2 → 7 * x + 5 * y ≤ 7 * a + 5 * b) ∧
  7 * x + 5 * y = 7 + 2 * Real.sqrt 6 :=
by sorry

end min_value_7x_5y_min_value_achieved_min_value_is_7_plus_2sqrt6_l41_4196


namespace soccer_league_games_l41_4199

theorem soccer_league_games (C D : ℕ) : 
  (3 * C = 4 * (C - (C / 4))) →  -- Team C has won 3/4 of its games
  (2 * (C + 6) = 3 * ((C + 6) - ((C + 6) / 3))) →  -- Team D has won 2/3 of its games
  (C + 6 = D) →  -- Team D has played 6 more games than team C
  (C = 12) :=  -- Prove that team C has played 12 games
by sorry

end soccer_league_games_l41_4199


namespace car_speed_proof_l41_4170

/-- Proves that a car traveling at 400 km/h takes 9 seconds to travel 1 kilometer,
    given that it takes 5 seconds longer than traveling 1 kilometer at 900 km/h. -/
theorem car_speed_proof (v : ℝ) (h1 : v > 0) :
  (1 / v) * 3600 = 9 ↔ v = 400 ∧ (1 / 900) * 3600 + 5 = 9 := by
  sorry

end car_speed_proof_l41_4170


namespace expression_evaluation_l41_4158

theorem expression_evaluation (x : ℝ) (h1 : x^2 - 3*x + 2 = 0) (h2 : x ≠ 2) :
  (x^2 / (x - 2) - x - 2) / (4*x / (x^2 - 4)) = 3 := by
  sorry

end expression_evaluation_l41_4158


namespace base9_85_to_decimal_l41_4105

/-- Converts a two-digit number in base 9 to its decimal representation -/
def base9_to_decimal (tens : Nat) (ones : Nat) : Nat :=
  tens * 9 + ones

/-- States that 85 in base 9 is equal to 77 in decimal -/
theorem base9_85_to_decimal : base9_to_decimal 8 5 = 77 := by
  sorry

end base9_85_to_decimal_l41_4105


namespace cookie_radius_l41_4143

-- Define the cookie equation
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6*x + 24*y

-- Theorem statement
theorem cookie_radius :
  ∃ (h k r : ℝ), r = Real.sqrt 117 ∧
  ∀ (x y : ℝ), cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end cookie_radius_l41_4143


namespace total_energy_calculation_l41_4130

def light_energy (base_watts : ℕ) (multiplier : ℕ) (hours : ℕ) : ℕ :=
  base_watts * multiplier * hours

theorem total_energy_calculation (base_watts : ℕ) (hours : ℕ) 
  (h1 : base_watts = 6)
  (h2 : hours = 2) :
  light_energy base_watts 1 hours + 
  light_energy base_watts 3 hours + 
  light_energy base_watts 4 hours = 96 :=
by sorry

end total_energy_calculation_l41_4130


namespace cos_equality_theorem_l41_4148

theorem cos_equality_theorem :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (942 * π / 180) ∧ n = 138 := by
  sorry

end cos_equality_theorem_l41_4148


namespace trig_identities_l41_4178

theorem trig_identities (α : Real) (h : Real.tan α = 2) : 
  ((Real.sin (Real.pi - α) + Real.cos (α - Real.pi/2) - Real.cos (3*Real.pi + α)) / 
   (Real.cos (Real.pi/2 + α) - Real.sin (2*Real.pi + α) + 2*Real.sin (α - Real.pi/2)) = -5/6) ∧ 
  (Real.cos (2*α) + Real.sin α * Real.cos α = -1/5) := by
  sorry

end trig_identities_l41_4178


namespace range_of_a_l41_4168

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
sorry

end range_of_a_l41_4168


namespace total_spent_is_correct_l41_4144

def clothes_price : ℝ := 250
def clothes_discount : ℝ := 0.15
def movie_ticket_price : ℝ := 24
def movie_tickets : ℕ := 3
def movie_discount : ℝ := 0.10
def beans_price : ℝ := 1.25
def beans_quantity : ℕ := 20
def cucumber_price : ℝ := 2.50
def cucumber_quantity : ℕ := 5
def tomato_price : ℝ := 5.00
def tomato_quantity : ℕ := 3
def pineapple_price : ℝ := 6.50
def pineapple_quantity : ℕ := 2

def total_spent : ℝ := 
  clothes_price * (1 - clothes_discount) +
  (movie_ticket_price * movie_tickets) * (1 - movie_discount) +
  (beans_price * beans_quantity) +
  (cucumber_price * cucumber_quantity) +
  (tomato_price * tomato_quantity) +
  (pineapple_price * pineapple_quantity)

theorem total_spent_is_correct : total_spent = 342.80 := by
  sorry

end total_spent_is_correct_l41_4144


namespace quadratic_inequality_roots_l41_4169

/-- Given that -x^2 + bx - 4 < 0 only when x ∈ (-∞, 0) ∪ (4, ∞), prove that b = 4 -/
theorem quadratic_inequality_roots (b : ℝ) 
  (h : ∀ x : ℝ, (-x^2 + b*x - 4 < 0) ↔ (x < 0 ∨ x > 4)) : 
  b = 4 := by sorry

end quadratic_inequality_roots_l41_4169


namespace sin_210_degrees_l41_4139

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l41_4139


namespace vector_on_line_and_parallel_l41_4149

/-- A line parameterized by x = 5t + 3 and y = 2t + 3 -/
def parameterized_line (t : ℝ) : ℝ × ℝ := (5 * t + 3, 2 * t + 3)

/-- The vector we want to prove is on the line and parallel to (5, 2) -/
def vector : ℝ × ℝ := (-1.5, -0.6)

/-- The direction vector we want our vector to be parallel to -/
def direction : ℝ × ℝ := (5, 2)

theorem vector_on_line_and_parallel :
  ∃ (t : ℝ), parameterized_line t = vector ∧
  ∃ (k : ℝ), vector.1 = k * direction.1 ∧ vector.2 = k * direction.2 := by
  sorry

end vector_on_line_and_parallel_l41_4149


namespace negative_half_to_fourth_power_l41_4194

theorem negative_half_to_fourth_power :
  (-1/2 : ℚ)^4 = 1/16 := by
  sorry

end negative_half_to_fourth_power_l41_4194


namespace sum_bounds_l41_4125

theorem sum_bounds (r s t u : ℝ) 
  (eq : 5*r + 4*s + 3*t + 6*u = 100)
  (h1 : r ≥ s) (h2 : s ≥ t) (h3 : t ≥ u) (h4 : u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := by
  sorry

end sum_bounds_l41_4125


namespace existence_of_equal_elements_l41_4135

theorem existence_of_equal_elements
  (p q n : ℕ+)
  (h_sum : p + q < n)
  (x : Fin (n + 1) → ℤ)
  (h_boundary : x 0 = 0 ∧ x n = 0)
  (h_diff : ∀ i : Fin n, x (i + 1) - x i = p ∨ x (i + 1) - x i = -q) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
by sorry

end existence_of_equal_elements_l41_4135


namespace factor_t_squared_minus_64_l41_4188

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_t_squared_minus_64_l41_4188


namespace max_equilateral_triangles_l41_4126

-- Define the number of line segments
def num_segments : ℕ := 6

-- Define the length of each segment
def segment_length : ℝ := 2

-- Define the side length of the equilateral triangles
def triangle_side_length : ℝ := 2

-- State the theorem
theorem max_equilateral_triangles :
  ∃ (n : ℕ), n ≤ 4 ∧
  (∀ (m : ℕ), (∃ (arrangement : List (List ℕ)),
    (∀ triangle ∈ arrangement, triangle.length = 3 ∧
     (∀ side ∈ triangle, side ≤ num_segments) ∧
     arrangement.length = m) →
    m ≤ n)) ∧
  (∃ (arrangement : List (List ℕ)),
    (∀ triangle ∈ arrangement, triangle.length = 3 ∧
     (∀ side ∈ triangle, side ≤ num_segments) ∧
     arrangement.length = 4)) :=
by sorry

end max_equilateral_triangles_l41_4126


namespace simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l41_4101

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := by sorry

-- Problem 3
theorem simplify_expression_3 (a b : ℝ) :
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := by sorry

-- Problem 4
theorem simplify_expression_4 (x y : ℝ) :
  6 * x * y^2 - (2 * x - (1/2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := by sorry

end simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l41_4101


namespace real_part_of_z_l41_4113

/-- Given that z = (1+i)(1-2i)(i) where i is the imaginary unit, prove that the real part of z is 3 -/
theorem real_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 + i) * (1 - 2*i) * i
  (z.re : ℝ) = 3 := by sorry

end real_part_of_z_l41_4113


namespace max_quotient_value_l41_4185

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 500 ≤ y ∧ y ≤ 1500 → y / x ≤ b / a) → b / a = 15 :=
by sorry

end max_quotient_value_l41_4185


namespace domain_intersection_subset_l41_4192

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def C (m : ℝ) : Set ℝ := {x | 3*x < 2*m - 1}

-- State the theorem
theorem domain_intersection_subset (m : ℝ) : 
  (A ∩ B) ⊆ C m → m > 5 := by
  sorry

end domain_intersection_subset_l41_4192


namespace lilly_and_rosy_fish_l41_4155

/-- The number of fish Lilly and Rosy have together -/
def total_fish (lilly_fish rosy_fish : ℕ) : ℕ := lilly_fish + rosy_fish

/-- Theorem: Lilly and Rosy have 22 fish in total -/
theorem lilly_and_rosy_fish : total_fish 10 12 = 22 := by
  sorry

end lilly_and_rosy_fish_l41_4155


namespace marias_painting_earnings_l41_4140

/-- Calculates Maria's earnings from selling a painting --/
theorem marias_painting_earnings
  (brush_cost : ℕ)
  (canvas_cost_multiplier : ℕ)
  (paint_cost_per_liter : ℕ)
  (paint_liters : ℕ)
  (selling_price : ℕ)
  (h1 : brush_cost = 20)
  (h2 : canvas_cost_multiplier = 3)
  (h3 : paint_cost_per_liter = 8)
  (h4 : paint_liters ≥ 5)
  (h5 : selling_price = 200) :
  selling_price - (brush_cost + canvas_cost_multiplier * brush_cost + paint_cost_per_liter * paint_liters) = 80 :=
by sorry

end marias_painting_earnings_l41_4140


namespace floor_product_l41_4162

theorem floor_product : ⌊(21.7 : ℝ)⌋ * ⌊(-21.7 : ℝ)⌋ = -462 := by
  sorry

end floor_product_l41_4162


namespace rice_containers_l41_4197

theorem rice_containers (total_weight : ℚ) (container_capacity : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 25 / 4 →
  container_capacity = 25 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce / container_capacity : ℚ) = 4 := by
  sorry

end rice_containers_l41_4197


namespace initial_gifts_count_l41_4166

/-- The number of gifts sent to the orphanage -/
def gifts_sent : ℕ := 66

/-- The number of gifts left under the tree -/
def gifts_left : ℕ := 11

/-- The initial number of gifts -/
def initial_gifts : ℕ := gifts_sent + gifts_left

theorem initial_gifts_count : initial_gifts = 77 := by
  sorry

end initial_gifts_count_l41_4166


namespace fifth_root_inequality_l41_4175

theorem fifth_root_inequality (x y : ℝ) : x < y → x^(1/5) > y^(1/5) := by
  sorry

end fifth_root_inequality_l41_4175


namespace f_min_at_neg_three_p_half_l41_4173

/-- The function f(x) = x^2 + 3px + 2p^2 -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + 3*p*x + 2*p^2

/-- Theorem: The minimum of f(x) occurs at x = -3p/2 when p > 0 -/
theorem f_min_at_neg_three_p_half (p : ℝ) (h : p > 0) :
  ∀ x : ℝ, f p (-3*p/2) ≤ f p x :=
sorry

end f_min_at_neg_three_p_half_l41_4173


namespace factorial_ratio_evaluation_l41_4151

theorem factorial_ratio_evaluation : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 5 / 21 := by
  sorry

end factorial_ratio_evaluation_l41_4151


namespace distance_to_line_implies_ab_bound_l41_4102

theorem distance_to_line_implies_ab_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let P : ℝ × ℝ := (1, 1)
  let line (x y : ℝ) := (a + 1) * x + (b + 1) * y - 2 = 0
  let distance_to_line := |((a + 1) * P.1 + (b + 1) * P.2 - 2)| / Real.sqrt ((a + 1)^2 + (b + 1)^2)
  distance_to_line = 1 → a * b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end distance_to_line_implies_ab_bound_l41_4102


namespace rectangle_area_change_l41_4103

/-- Given a rectangle with dimensions 4 and 6, if shortening one side by 1
    results in an area of 18, then shortening the other side by 1
    results in an area of 20. -/
theorem rectangle_area_change (l w : ℝ) : 
  l = 4 ∧ w = 6 ∧ 
  ((l - 1) * w = 18 ∨ l * (w - 1) = 18) →
  (l * (w - 1) = 20 ∨ (l - 1) * w = 20) :=
by sorry

end rectangle_area_change_l41_4103


namespace symmetry_xoz_of_point_l41_4122

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Performs symmetry about the xOz plane -/
def symmetryXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_xoz_of_point :
  let A : Point3D := { x := 9, y := 8, z := 5 }
  symmetryXOZ A = { x := 9, y := -8, z := 5 } := by
  sorry

end symmetry_xoz_of_point_l41_4122


namespace even_function_extension_l41_4108

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_nonpos : ∀ x ≤ 0, f x = x^3 - x^2) :
  ∀ x > 0, f x = -x^3 - x^2 :=
by sorry

end even_function_extension_l41_4108


namespace inverse_proportion_ratio_l41_4198

theorem inverse_proportion_ratio (c₁ c₂ d₁ d₂ : ℝ) : 
  c₁ ≠ 0 → c₂ ≠ 0 → d₁ ≠ 0 → d₂ ≠ 0 →
  (∃ k : ℝ, ∀ c d, c * d = k) →
  c₁ * d₁ = c₂ * d₂ →
  c₁ / c₂ = 3 / 4 →
  d₁ / d₂ = 4 / 3 := by
sorry

end inverse_proportion_ratio_l41_4198


namespace brown_eyed_brunettes_l41_4115

theorem brown_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) 
  (h1 : total = 60)
  (h2 : blue_eyed_blondes = 20)
  (h3 : brunettes = 36)
  (h4 : brown_eyed = 25) :
  total - brunettes - blue_eyed_blondes + brown_eyed = 21 :=
by sorry

end brown_eyed_brunettes_l41_4115


namespace completing_square_result_l41_4176

theorem completing_square_result (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → ((x - 3)^2 = 1) :=
by
  sorry

end completing_square_result_l41_4176


namespace bucket_size_calculation_l41_4193

/-- Given a leak rate and maximum time away, calculate the required bucket size -/
theorem bucket_size_calculation (leak_rate : ℝ) (max_time : ℝ) 
  (h1 : leak_rate = 1.5)
  (h2 : max_time = 12)
  (h3 : leak_rate > 0)
  (h4 : max_time > 0) :
  2 * (leak_rate * max_time) = 36 :=
by sorry

end bucket_size_calculation_l41_4193


namespace hyperbola_equation_l41_4186

/-- The standard equation of a hyperbola with the same foci as a given ellipse and passing through a specific point -/
theorem hyperbola_equation (e : Real → Real → Prop) (p : Real × Real) :
  (∀ x y, e x y ↔ x^2 / 9 + y^2 / 5 = 1) →
  p = (Real.sqrt 2, Real.sqrt 3) →
  ∃ h : Real → Real → Prop,
    (∀ x y, h x y ↔ x^2 - y^2 / 3 = 1) ∧
    (∀ c : Real, (∃ x, e x 0 ∧ x^2 = c^2) ↔ (∃ x, h x 0 ∧ x^2 = c^2)) ∧
    h p.1 p.2 :=
by sorry

end hyperbola_equation_l41_4186


namespace sequence_existence_and_extension_l41_4106

theorem sequence_existence_and_extension (m : ℕ) (h : m ≥ 2) :
  (∃ x : ℕ → ℤ, ∀ i ∈ Finset.range m, x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) ∧
  (∀ x : ℕ → ℤ, (∀ i ∈ Finset.range m, x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
    ∃ y : ℤ → ℤ, (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
               (∀ i ∈ Finset.range (2 * m), y i = x i)) := by
  sorry

end sequence_existence_and_extension_l41_4106


namespace lunks_for_dozen_apples_l41_4134

-- Define the exchange rates
def lunks_per_kunk : ℚ := 7 / 4
def apples_per_kunk : ℚ := 5 / 3

-- Define a dozen
def dozen : ℕ := 12

-- Theorem statement
theorem lunks_for_dozen_apples : 
  ∃ (l : ℚ), l = dozen * (lunks_per_kunk / apples_per_kunk) ∧ l = 12.6 := by
sorry

end lunks_for_dozen_apples_l41_4134


namespace large_rectangle_perimeter_l41_4137

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.side

/-- Theorem stating the perimeter of the large rectangle -/
theorem large_rectangle_perimeter 
  (square : Square) 
  (small_rect : Rectangle) 
  (h1 : square.perimeter = 24) 
  (h2 : small_rect.perimeter = 16) 
  (h3 : small_rect.length = square.side) :
  let large_rect := Rectangle.mk (3 * square.side + small_rect.length) (square.side + small_rect.width)
  large_rect.perimeter = 52 := by
  sorry


end large_rectangle_perimeter_l41_4137


namespace majorization_iff_transformable_l41_4182

/-- Represents a triplet of real numbers -/
structure Triplet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines majorization relation between two triplets -/
def majorizes (α β : Triplet) : Prop :=
  α.a ≥ β.a ∧ α.a + α.b ≥ β.a + β.b ∧ α.a + α.b + α.c = β.a + β.b + β.c

/-- Represents the allowed operations on triplets -/
inductive Operation
  | op1 : Operation  -- (k, j, i) ↔ (k-1, j+1, i)
  | op2 : Operation  -- (k, j, i) ↔ (k-1, j, i+1)
  | op3 : Operation  -- (k, j, i) ↔ (k, j-1, i+1)

/-- Applies an operation to a triplet -/
def applyOperation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.op1 => ⟨t.a - 1, t.b + 1, t.c⟩
  | Operation.op2 => ⟨t.a - 1, t.b, t.c + 1⟩
  | Operation.op3 => ⟨t.a, t.b - 1, t.c + 1⟩

/-- Checks if one triplet can be obtained from another using allowed operations -/
def canObtain (α β : Triplet) : Prop :=
  ∃ (ops : List Operation), β = ops.foldl applyOperation α

/-- Main theorem: Majorization is equivalent to ability to transform using allowed operations -/
theorem majorization_iff_transformable (α β : Triplet) :
  majorizes α β ↔ canObtain α β := by sorry

end majorization_iff_transformable_l41_4182


namespace negation_equivalence_l41_4120

-- Define a triangle
structure Triangle where
  -- Add necessary fields for a triangle

-- Define an obtuse angle
def isObtuseAngle (angle : Real) : Prop := angle > Real.pi / 2

-- Define the property of having at most one obtuse angle
def atMostOneObtuseAngle (t : Triangle) : Prop :=
  ∃ (a b c : Real), isObtuseAngle a → ¬(isObtuseAngle b ∨ isObtuseAngle c)

-- Define the property of having at least two obtuse angles
def atLeastTwoObtuseAngles (t : Triangle) : Prop :=
  ∃ (a b : Real), isObtuseAngle a ∧ isObtuseAngle b

-- Theorem stating that the negation of "at most one obtuse angle" 
-- is equivalent to "at least two obtuse angles"
theorem negation_equivalence (t : Triangle) : 
  ¬(atMostOneObtuseAngle t) ↔ atLeastTwoObtuseAngles t := by
  sorry

end negation_equivalence_l41_4120


namespace sum_of_xyz_l41_4132

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 24) (hxz : x * z = 48) (hyz : y * z = 72) :
  x + y + z = 22 := by
  sorry

end sum_of_xyz_l41_4132


namespace common_measure_proof_l41_4150

theorem common_measure_proof (a b : ℚ) (ha : a = 4/15) (hb : b = 8/21) :
  ∃ (m : ℚ), m > 0 ∧ ∃ (k₁ k₂ : ℕ), a = k₁ * m ∧ b = k₂ * m :=
by
  -- The proof would go here
  sorry

end common_measure_proof_l41_4150


namespace waiter_tables_l41_4165

/-- Proves that a waiter with 40 customers and tables of 5 women and 3 men each has 5 tables. -/
theorem waiter_tables (total_customers : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  total_customers = 40 →
  women_per_table = 5 →
  men_per_table = 3 →
  total_customers = (women_per_table + men_per_table) * 5 :=
by sorry

end waiter_tables_l41_4165


namespace total_bags_l41_4104

theorem total_bags (points_per_bag : ℕ) (total_points : ℕ) (unrecycled_bags : ℕ) : 
  points_per_bag = 5 → total_points = 45 → unrecycled_bags = 8 →
  (total_points / points_per_bag + unrecycled_bags : ℕ) = 17 := by
sorry

end total_bags_l41_4104


namespace min_dot_product_hyperbola_l41_4109

/-- The minimum dot product of two vectors from the origin to points on the right branch of x² - y² = 1 is 1 -/
theorem min_dot_product_hyperbola (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁ > 0 → x₂ > 0 → x₁^2 - y₁^2 = 1 → x₂^2 - y₂^2 = 1 → x₁*x₂ + y₁*y₂ ≥ 1 := by
  sorry

#check min_dot_product_hyperbola

end min_dot_product_hyperbola_l41_4109


namespace difference_of_squares_65_35_l41_4133

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l41_4133


namespace ryan_english_time_l41_4163

/-- The time Ryan spends on learning English, given the total time spent on learning
    English and Chinese, and the time spent on learning Chinese. -/
def time_learning_english (total_time : ℝ) (chinese_time : ℝ) : ℝ :=
  total_time - chinese_time

/-- Theorem stating that Ryan spends 2 hours learning English -/
theorem ryan_english_time :
  time_learning_english 3 1 = 2 := by
  sorry

end ryan_english_time_l41_4163


namespace hexagons_in_50th_ring_l41_4123

/-- Represents the number of hexagons in a ring of a hexagonal arrangement -/
def hexagonsInRing (n : ℕ) : ℕ := 6 * n

/-- The hexagonal arrangement has the following properties:
    1. The center is a regular hexagon of unit side length
    2. Surrounded by rings of unit hexagons
    3. The first ring consists of 6 unit hexagons
    4. The second ring contains 12 unit hexagons -/
axiom hexagonal_arrangement_properties : True

theorem hexagons_in_50th_ring : 
  hexagonsInRing 50 = 300 := by sorry

end hexagons_in_50th_ring_l41_4123


namespace marble_problem_solution_l41_4145

/-- Represents a jar of marbles -/
structure Jar where
  blue : ℕ
  green : ℕ

/-- The problem setup -/
def marble_problem : Prop :=
  ∃ (jar1 jar2 : Jar),
    -- Both jars have the same total number of marbles
    jar1.blue + jar1.green = jar2.blue + jar2.green
    -- Ratio of blue to green in Jar 1 is 9:1
    ∧ 9 * jar1.green = jar1.blue
    -- Ratio of blue to green in Jar 2 is 7:2
    ∧ 7 * jar2.green = 2 * jar2.blue
    -- Total number of green marbles is 108
    ∧ jar1.green + jar2.green = 108
    -- The difference in blue marbles between Jar 1 and Jar 2 is 38
    ∧ jar1.blue - jar2.blue = 38

/-- The theorem to prove -/
theorem marble_problem_solution : marble_problem := by
  sorry

end marble_problem_solution_l41_4145


namespace system_solution_l41_4174

theorem system_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0 ∧
   3 * x^2 * y^2 + y^4 = 84) ↔
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
sorry

end system_solution_l41_4174


namespace point_Q_in_third_quadrant_l41_4136

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determine if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The theorem statement -/
theorem point_Q_in_third_quadrant (m : ℝ) :
  let P : Point := ⟨m + 3, 2 * m + 4⟩
  let Q : Point := ⟨m - 3, m⟩
  (P.y = 0) → isInThirdQuadrant Q :=
by sorry

end point_Q_in_third_quadrant_l41_4136


namespace houses_with_one_pet_l41_4124

/-- Represents the number of houses with different pet combinations in a neighborhood --/
structure PetHouses where
  total : ℕ
  dogs : ℕ
  cats : ℕ
  birds : ℕ
  dogsCats : ℕ
  catsBirds : ℕ
  dogsBirds : ℕ

/-- Theorem stating the number of houses with only one type of pet --/
theorem houses_with_one_pet (h : PetHouses) 
  (h_total : h.total = 75)
  (h_dogs : h.dogs = 40)
  (h_cats : h.cats = 30)
  (h_birds : h.birds = 8)
  (h_dogs_cats : h.dogsCats = 10)
  (h_cats_birds : h.catsBirds = 5)
  (h_dogs_birds : h.dogsBirds = 0) :
  h.dogs + h.cats + h.birds - h.dogsCats - h.catsBirds - h.dogsBirds = 48 := by
  sorry


end houses_with_one_pet_l41_4124


namespace equal_pair_proof_l41_4111

theorem equal_pair_proof : 
  ((-3 : ℤ)^2 = Int.sqrt 81) ∧ 
  (|(-3 : ℤ)| ≠ -3) ∧ 
  (-|(-4 : ℤ)| ≠ (-2 : ℤ)^2) ∧ 
  (Int.sqrt ((-4 : ℤ)^2) ≠ -4) :=
by sorry

end equal_pair_proof_l41_4111


namespace shorter_side_is_ten_l41_4171

/-- A rectangular room with given perimeter and area -/
structure Room where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 30
  area_eq : length * width = 200

/-- The shorter side of the room is 10 feet -/
theorem shorter_side_is_ten (room : Room) : min room.length room.width = 10 := by
  sorry

end shorter_side_is_ten_l41_4171


namespace expression_simplification_l41_4112

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (3 * x + 3)) = Real.sqrt 3 := by
  sorry

end expression_simplification_l41_4112


namespace remainder_divisibility_l41_4183

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 20) → (∃ m : ℤ, N = 13 * m + 7) :=
by sorry

end remainder_divisibility_l41_4183


namespace tenth_term_is_18_l41_4191

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 = 2 ∧ 
  a 3 = 4

/-- The 10th term of the arithmetic sequence is 18 -/
theorem tenth_term_is_18 (a : ℕ → ℝ) (h : arithmetic_sequence a) : 
  a 10 = 18 := by
  sorry

end tenth_term_is_18_l41_4191


namespace regression_line_estimate_estimated_y_value_l41_4138

/-- Given a regression line equation and an x-value, calculate the estimated y-value -/
theorem regression_line_estimate (slope intercept x : ℝ) :
  let regression_line := fun (x : ℝ) => slope * x + intercept
  regression_line x = slope * x + intercept := by sorry

/-- The estimated y-value for the given regression line when x = 28 is 390 -/
theorem estimated_y_value :
  let slope : ℝ := 4.75
  let intercept : ℝ := 257
  let x : ℝ := 28
  let regression_line := fun (x : ℝ) => slope * x + intercept
  regression_line x = 390 := by sorry

end regression_line_estimate_estimated_y_value_l41_4138


namespace goldfish_pond_problem_l41_4167

theorem goldfish_pond_problem :
  ∀ (x : ℕ),
  (x > 0) →
  (3 * x / 7 : ℚ) + (4 * x / 7 : ℚ) = x →
  (5 * x / 8 : ℚ) + (3 * x / 8 : ℚ) = x →
  (5 * x / 8 : ℚ) - (3 * x / 7 : ℚ) = 33 →
  x = 168 := by
sorry

end goldfish_pond_problem_l41_4167


namespace trick_decks_total_spend_l41_4118

/-- The total amount spent by Victor and his friend on trick decks -/
def totalSpent (deckCost : ℕ) (victorDecks : ℕ) (friendDecks : ℕ) : ℕ :=
  deckCost * (victorDecks + friendDecks)

/-- Theorem stating the total amount spent by Victor and his friend -/
theorem trick_decks_total_spend :
  totalSpent 8 6 2 = 64 := by
  sorry

end trick_decks_total_spend_l41_4118


namespace max_fibonacci_match_l41_4161

/-- A sequence that matches the Fibonacci sequence for a given number of terms -/
def MatchesFibonacci (t : ℕ → ℝ) (start : ℕ) (count : ℕ) : Prop :=
  ∀ k, k < count → t (start + k + 2) = t (start + k + 1) + t (start + k)

/-- The quadratic sequence defined by A, B, and C -/
def QuadraticSequence (A B C : ℝ) (n : ℕ) : ℝ :=
  A * (n : ℝ)^2 + B * (n : ℝ) + C

/-- The theorem stating the maximum number of consecutive Fibonacci terms -/
theorem max_fibonacci_match (A B C : ℝ) (h : A ≠ 0) :
  (∃ start, MatchesFibonacci (QuadraticSequence A B C) start 4) ∧
  (∀ start count, count > 4 → ¬MatchesFibonacci (QuadraticSequence A B C) start count) ∧
  ((A = 1/2 ∧ B = -1/2 ∧ C = 2) ∨ (A = 1/2 ∧ B = 1/2 ∧ C = 2)) :=
sorry

end max_fibonacci_match_l41_4161


namespace inverse_f_at_46_l41_4110

def f (x : ℝ) : ℝ := 5 * x^3 + 6

theorem inverse_f_at_46 : f⁻¹ 46 = 2 := by sorry

end inverse_f_at_46_l41_4110


namespace mirror_number_max_k_value_l41_4116

/-- Definition of a mirror number -/
def is_mirror_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 10 ≠ 0) ∧ (n / 10 % 10 ≠ 0) ∧ (n / 100 % 10 ≠ 0) ∧ (n / 1000 ≠ 0) ∧
  (n % 10 ≠ n / 10 % 10) ∧ (n % 10 ≠ n / 100 % 10) ∧ (n % 10 ≠ n / 1000) ∧
  (n / 10 % 10 ≠ n / 100 % 10) ∧ (n / 10 % 10 ≠ n / 1000) ∧ (n / 100 % 10 ≠ n / 1000) ∧
  (n % 10 + n / 1000 = n / 10 % 10 + n / 100 % 10)

/-- Definition of F(m) -/
def F (m : ℕ) : ℚ :=
  let m₁ := (m % 10) * 1000 + (m / 10 % 10) * 100 + (m / 100 % 10) * 10 + (m / 1000)
  let m₂ := (m / 1000) * 1000 + (m / 100 % 10) * 100 + (m / 10 % 10) * 10 + (m % 10)
  (m₁ + m₂ : ℚ) / 1111

/-- Main theorem -/
theorem mirror_number_max_k_value 
  (s t : ℕ) 
  (x y e f : ℕ)
  (hs : is_mirror_number s)
  (ht : is_mirror_number t)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (he : 1 ≤ e ∧ e ≤ 9)
  (hf : 1 ≤ f ∧ f ≤ 9)
  (hs_def : s = 1000 * x + 100 * y + 32)
  (ht_def : t = 1500 + 10 * e + f)
  (h_sum : F s + F t = 19)
  : (F s / F t) ≤ 11 / 8 :=
sorry

end mirror_number_max_k_value_l41_4116


namespace average_of_a_and_b_l41_4129

theorem average_of_a_and_b (a b c : ℝ) : 
  (a + b) / 2 = 45 ∧ (b + c) / 2 = 60 ∧ c - a = 30 → (a + b) / 2 = 45 := by
  sorry

end average_of_a_and_b_l41_4129


namespace tom_payment_l41_4128

/-- The amount Tom paid to the shopkeeper -/
def total_amount (apple_quantity apple_rate mango_quantity mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Proof that Tom paid 1055 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end tom_payment_l41_4128


namespace f_of_two_equals_negative_twenty_six_l41_4187

/-- Given a function f(x) = ax^5 + bx^3 + sin(x) - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_of_two_equals_negative_twenty_six 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + Real.sin x - 8) 
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
sorry

end f_of_two_equals_negative_twenty_six_l41_4187


namespace square_difference_divided_problem_solution_l41_4159

theorem square_difference_divided (a b : ℕ) (h : a > b) : 
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (725^2 - 675^2) / 25 = 2800 :=
by 
  have h : 725 > 675 := by sorry
  have key := square_difference_divided 725 675 h
  sorry

end square_difference_divided_problem_solution_l41_4159


namespace leo_mira_sum_difference_l41_4157

def leo_sum : ℕ := (List.range 50).map (· + 1) |>.sum

def digit_replace (n : ℕ) : ℕ :=
  let s := toString n
  let s' := s.replace "2" "1" |>.replace "3" "0"
  s'.toNat!

def mira_sum : ℕ := (List.range 50).map (· + 1 |> digit_replace) |>.sum

theorem leo_mira_sum_difference : leo_sum - mira_sum = 420 := by
  sorry

end leo_mira_sum_difference_l41_4157


namespace score_ordering_l41_4190

-- Define the set of people
inductive Person : Type
| K : Person  -- Kaleana
| Q : Person  -- Quay
| M : Person  -- Marty
| S : Person  -- Shana

-- Define a function to represent the score of each person
variable (score : Person → ℕ)

-- Define the conditions
axiom quay_thought : score Person.Q = score Person.K
axiom marty_thought : score Person.M > score Person.K
axiom shana_thought : score Person.S < score Person.K

-- Define the theorem to prove
theorem score_ordering :
  score Person.S < score Person.Q ∧ score Person.Q < score Person.M :=
sorry

end score_ordering_l41_4190


namespace g_difference_l41_4156

-- Define the function g
noncomputable def g (n : ℤ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((2 + Real.sqrt 7) / 3) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((2 - Real.sqrt 7) / 3) ^ n +
  3

-- Theorem statement
theorem g_difference (n : ℤ) : g (n + 1) - g (n - 1) = g n := by
  sorry

end g_difference_l41_4156


namespace students_in_both_sports_l41_4172

theorem students_in_both_sports (total : ℕ) (baseball : ℕ) (hockey : ℕ) 
  (h1 : total = 36) (h2 : baseball = 25) (h3 : hockey = 19) :
  baseball + hockey - total = 8 := by
  sorry

end students_in_both_sports_l41_4172


namespace parabola_symmetry_axis_l41_4107

/-- A parabola defined by y = 2x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The theorem stating that for a parabola y = 2x^2 + bx + c passing through
    points A(2,5) and B(4,5), the axis of symmetry is x = 3 -/
theorem parabola_symmetry_axis (p : Parabola) (A B : Point)
    (hA : A.y = 2 * A.x^2 + p.b * A.x + p.c)
    (hB : B.y = 2 * B.x^2 + p.b * B.x + p.c)
    (hAx : A.x = 2) (hAy : A.y = 5)
    (hBx : B.x = 4) (hBy : B.y = 5) :
    (A.x + B.x) / 2 = 3 := by sorry

end parabola_symmetry_axis_l41_4107


namespace is_valid_factorization_l41_4195

/-- Proves that x^2 - 2x + 1 = (x - 1)^2 is a valid factorization -/
theorem is_valid_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end is_valid_factorization_l41_4195


namespace min_value_reciprocal_sum_l41_4152

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 :=
sorry

end min_value_reciprocal_sum_l41_4152


namespace max_volume_at_eight_l41_4121

/-- The volume of the box as a function of the side length of the removed square -/
def boxVolume (x : ℝ) : ℝ := (48 - 2*x)^2 * x

/-- The derivative of the box volume with respect to x -/
def boxVolumeDerivative (x : ℝ) : ℝ := (48 - 2*x) * (48 - 6*x)

theorem max_volume_at_eight :
  ∃ (x : ℝ), 0 < x ∧ x < 24 ∧
  (∀ (y : ℝ), 0 < y ∧ y < 24 → boxVolume y ≤ boxVolume x) ∧
  x = 8 := by
sorry

end max_volume_at_eight_l41_4121


namespace cos_2alpha_plus_cos_2beta_l41_4117

theorem cos_2alpha_plus_cos_2beta (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1)
  (h2 : Real.cos α + Real.cos β = 0) :
  Real.cos (2 * α) + Real.cos (2 * β) = 1 := by
  sorry

end cos_2alpha_plus_cos_2beta_l41_4117


namespace rect_to_polar_conversion_l41_4180

/-- Conversion of rectangular coordinates (8, 2√6) to polar coordinates (r, θ) -/
theorem rect_to_polar_conversion :
  ∃ (r θ : ℝ), 
    r = 2 * Real.sqrt 22 ∧ 
    Real.tan θ = Real.sqrt 6 / 4 ∧ 
    r > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry

end rect_to_polar_conversion_l41_4180


namespace algebraic_fraction_simplification_l41_4154

theorem algebraic_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 3*x + 2) / ((x^2 - 6*x + 9) / (x^2 - 7*x + 10)) = (x - 5) / (x - 3) := by
  sorry

end algebraic_fraction_simplification_l41_4154


namespace remainder_theorem_l41_4147

/-- The polynomial p(x) = 3x^5 + 2x^3 - 5x + 8 -/
def p (x : ℝ) : ℝ := 3 * x^5 + 2 * x^3 - 5 * x + 8

/-- The divisor polynomial d(x) = x^2 - 2x + 1 -/
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

/-- The remainder polynomial r(x) = 16x - 8 -/
def r (x : ℝ) : ℝ := 16 * x - 8

/-- The quotient polynomial q(x) -/
noncomputable def q (x : ℝ) : ℝ := (p x - r x) / (d x)

theorem remainder_theorem : ∀ x : ℝ, p x = d x * q x + r x := by
  sorry

end remainder_theorem_l41_4147


namespace fraction_product_l41_4164

theorem fraction_product : (2 : ℚ) / 9 * (5 : ℚ) / 11 = 10 / 99 := by
  sorry

end fraction_product_l41_4164
