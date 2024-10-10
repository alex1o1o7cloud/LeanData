import Mathlib

namespace two_decimals_sum_and_difference_l4059_405907

theorem two_decimals_sum_and_difference (x y : ℝ) : 
  (0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10) → -- x and y are single-digit decimals
  (x + y = 10) →                     -- their sum is 10
  (|x - y| = 0.4) →                  -- their difference is 0.4
  ((x = 4.8 ∧ y = 5.2) ∨ (x = 5.2 ∧ y = 4.8)) := by
sorry

end two_decimals_sum_and_difference_l4059_405907


namespace inequality_solution_l4059_405978

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the inequality function
def f (x m : ℝ) : Prop := lg x ^ 2 - (2 + m) * lg x + m - 1 > 0

-- State the theorem
theorem inequality_solution :
  ∀ m : ℝ, |m| ≤ 1 →
    {x : ℝ | f x m} = {x : ℝ | 0 < x ∧ x < (1/10) ∨ x > 1000} := by sorry

end inequality_solution_l4059_405978


namespace sqrt_256_equals_2_to_n_l4059_405957

theorem sqrt_256_equals_2_to_n (n : ℕ) : (256 : ℝ)^(1/2) = 2^n → n = 4 := by
  sorry

end sqrt_256_equals_2_to_n_l4059_405957


namespace interest_rate_difference_l4059_405903

/-- Given a principal amount, time period, and difference in interest earned,
    calculate the difference between two simple interest rates. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h_principal : principal = 2300)
  (h_time : time = 3)
  (h_interest_diff : interest_diff = 69) :
  let rate_diff := interest_diff / (principal * time / 100)
  rate_diff = 1 := by sorry

end interest_rate_difference_l4059_405903


namespace monic_quartic_polynomial_value_l4059_405968

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) 
  (h_monic : MonicQuarticPolynomial p)
  (h_neg_two : p (-2) = -4)
  (h_one : p 1 = -1)
  (h_three : p 3 = -9)
  (h_five : p 5 = -25) :
  p 0 = -30 := by
  sorry

end monic_quartic_polynomial_value_l4059_405968


namespace tennis_tournament_balls_l4059_405952

theorem tennis_tournament_balls (total_balls : ℕ) (balls_per_can : ℕ) : 
  total_balls = 225 →
  balls_per_can = 3 →
  (8 + 4 + 2 + 1 : ℕ) * (total_balls / balls_per_can / (8 + 4 + 2 + 1)) = 5 := by
  sorry

end tennis_tournament_balls_l4059_405952


namespace subset_X_l4059_405900

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_X : {0} ⊆ X := by
  sorry

end subset_X_l4059_405900


namespace profit_loss_percentage_l4059_405908

/-- 
Given an article with a cost price and two selling prices:
1. An original selling price that yields a 27.5% profit
2. A new selling price that is 2/3 of the original price

This theorem proves that the loss percentage at the new selling price is 15%.
-/
theorem profit_loss_percentage (cost_price : ℝ) (original_price : ℝ) (new_price : ℝ) : 
  original_price = cost_price * (1 + 0.275) →
  new_price = (2/3) * original_price →
  (cost_price - new_price) / cost_price * 100 = 15 := by
sorry

end profit_loss_percentage_l4059_405908


namespace specific_triangle_l4059_405973

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- The main theorem about the specific acute triangle -/
theorem specific_triangle (t : AcuteTriangle) 
  (h1 : t.a = 2 * t.b * Real.sin t.A)
  (h2 : t.a = 3 * Real.sqrt 3)
  (h3 : t.c = 5) :
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry


end specific_triangle_l4059_405973


namespace total_cost_of_eggs_l4059_405944

def dozen : ℕ := 12
def egg_cost : ℚ := 0.50
def num_dozens : ℕ := 3

theorem total_cost_of_eggs :
  (↑num_dozens * ↑dozen) * egg_cost = 18 := by sorry

end total_cost_of_eggs_l4059_405944


namespace complex_fraction_simplification_l4059_405959

theorem complex_fraction_simplification :
  ((3 + 2*Complex.I) / (2 - 3*Complex.I)) - ((3 - 2*Complex.I) / (2 + 3*Complex.I)) = 2*Complex.I :=
by sorry

end complex_fraction_simplification_l4059_405959


namespace final_milk_water_ratio_l4059_405998

/- Given conditions -/
def initial_ratio : Rat := 1 / 5
def can_capacity : ℝ := 8
def additional_milk : ℝ := 2

/- Theorem to prove -/
theorem final_milk_water_ratio :
  let initial_mixture := can_capacity - additional_milk
  let initial_milk := initial_mixture * (initial_ratio / (1 + initial_ratio))
  let initial_water := initial_mixture * (1 / (1 + initial_ratio))
  let final_milk := initial_milk + additional_milk
  let final_water := initial_water
  (final_milk / final_water) = 3 / 5 := by
  sorry

end final_milk_water_ratio_l4059_405998


namespace tail_cut_divisibility_by_7_l4059_405986

def tail_cut (n : ℕ) : ℕ :=
  (n / 10) - 2 * (n % 10)

def is_divisible_by_7 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem tail_cut_divisibility_by_7 (A : ℕ) :
  (A > 0) →
  (is_divisible_by_7 A ↔ 
    ∃ (k : ℕ), is_divisible_by_7 (Nat.iterate tail_cut k A)) :=
by sorry

end tail_cut_divisibility_by_7_l4059_405986


namespace snow_probability_first_week_l4059_405921

theorem snow_probability_first_week (p1 p2 : ℝ) : 
  p1 = 1/3 → p2 = 1/4 → 
  (1 - (1 - p1)^4 * (1 - p2)^3) = 11/12 :=
by sorry

end snow_probability_first_week_l4059_405921


namespace final_balance_percentage_l4059_405997

def starting_balance : ℝ := 125
def initial_increase : ℝ := 0.25
def first_usd_to_eur : ℝ := 0.85
def decrease_in_eur : ℝ := 0.20
def eur_to_usd : ℝ := 1.15
def increase_in_usd : ℝ := 0.15
def decrease_in_usd : ℝ := 0.10
def final_usd_to_eur : ℝ := 0.88

theorem final_balance_percentage (starting_balance initial_increase first_usd_to_eur
  decrease_in_eur eur_to_usd increase_in_usd decrease_in_usd final_usd_to_eur : ℝ) :
  let initial_eur := starting_balance * (1 + initial_increase) * first_usd_to_eur
  let after_decrease_eur := initial_eur * (1 - decrease_in_eur)
  let back_to_usd := after_decrease_eur * eur_to_usd
  let after_increase_usd := back_to_usd * (1 + increase_in_usd)
  let after_decrease_usd := after_increase_usd * (1 - decrease_in_usd)
  let final_eur := after_decrease_usd * final_usd_to_eur
  let starting_eur := starting_balance * first_usd_to_eur
  (final_eur / starting_eur) * 100 = 104.75 :=
by sorry

end final_balance_percentage_l4059_405997


namespace negation_of_universal_proposition_l4059_405904

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ∈ Set.Ici (-2) → x + 3 ≥ 1)) ↔ 
  (∃ x : ℝ, x ∈ Set.Ici (-2) ∧ x + 3 < 1) :=
by sorry

end negation_of_universal_proposition_l4059_405904


namespace triangle_isosceles_l4059_405989

/-- Given a triangle ABC with angles α, β, γ and sides a, b,
    if a + b = tan(γ/2) * (a * tan(α) + b * tan(β)),
    then the triangle ABC is isosceles. -/
theorem triangle_isosceles (α β γ a b : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ ∧ 
  α + β + γ = Real.pi ∧
  0 < a ∧ 0 < b ∧
  a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β) →
  a = b ∨ α = β := by
  sorry

end triangle_isosceles_l4059_405989


namespace kolya_speed_increase_l4059_405936

theorem kolya_speed_increase (N : ℕ) (x : ℕ) : 
  -- Total problems for each student
  N = (3 * x) / 2 →
  -- Kolya has solved 1/3 of what Seryozha has left
  x / 6 = (x / 2) / 3 →
  -- Seryozha has solved half of his problems
  x = N / 2 →
  -- The factor by which Kolya needs to increase his speed
  (((3 * x) / 2 - x / 6) / (x / 2)) / (x / 6 / x) = 16 :=
by
  sorry


end kolya_speed_increase_l4059_405936


namespace cos_four_thirds_pi_plus_alpha_l4059_405915

theorem cos_four_thirds_pi_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos ((4 / 3) * π + α) = -(1 / 3) := by
  sorry

end cos_four_thirds_pi_plus_alpha_l4059_405915


namespace hyperbola_eccentricity_l4059_405933

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
  (x + c)^2 / 4 + y^2 / 4 = b^2) → 
  c^2 / a^2 = 5 := by sorry

end hyperbola_eccentricity_l4059_405933


namespace cherry_strawberry_cost_ratio_l4059_405970

/-- The cost of a pound of strawberries in dollars -/
def strawberry_cost : ℚ := 2.20

/-- The cost of 5 pounds of strawberries and 5 pounds of cherries in dollars -/
def total_cost : ℚ := 77

/-- The ratio of the cost of cherries to strawberries -/
def cherry_strawberry_ratio : ℚ := 6

theorem cherry_strawberry_cost_ratio :
  ∃ (cherry_cost : ℚ),
    cherry_cost > 0 ∧
    5 * strawberry_cost + 5 * cherry_cost = total_cost ∧
    cherry_cost / strawberry_cost = cherry_strawberry_ratio :=
by sorry

end cherry_strawberry_cost_ratio_l4059_405970


namespace lara_swimming_theorem_l4059_405940

/-- The number of minutes Lara must swim on the ninth day to average 100 minutes per day over 9 days -/
def minutes_to_swim_on_ninth_day (
  days_at_80_min : ℕ)  -- Number of days Lara swam 80 minutes
  (days_at_105_min : ℕ) -- Number of days Lara swam 105 minutes
  (target_average : ℕ)  -- Target average minutes per day
  (total_days : ℕ)      -- Total number of days
  : ℕ :=
  target_average * total_days - (days_at_80_min * 80 + days_at_105_min * 105)

/-- Theorem stating the correct number of minutes Lara must swim on the ninth day -/
theorem lara_swimming_theorem :
  minutes_to_swim_on_ninth_day 6 2 100 9 = 210 := by
  sorry

end lara_swimming_theorem_l4059_405940


namespace boat_accident_proof_l4059_405932

/-- The number of sheep that drowned in a boat accident -/
def drowned_sheep : ℕ := 3

theorem boat_accident_proof :
  let initial_sheep : ℕ := 20
  let initial_cows : ℕ := 10
  let initial_dogs : ℕ := 14
  let drowned_cows : ℕ := 2 * drowned_sheep
  let survived_dogs : ℕ := initial_dogs
  let total_survived : ℕ := 35
  total_survived = (initial_sheep - drowned_sheep) + (initial_cows - drowned_cows) + survived_dogs :=
by sorry

end boat_accident_proof_l4059_405932


namespace single_burger_cost_is_one_l4059_405950

/-- Calculates the cost of a single burger given the total spent, total number of hamburgers,
    number of double burgers, and cost of a double burger. -/
def single_burger_cost (total_spent : ℚ) (total_burgers : ℕ) (double_burgers : ℕ) (double_cost : ℚ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let double_total := double_burgers * double_cost
  let single_total := total_spent - double_total
  single_total / single_burgers

/-- Proves that the cost of a single burger is $1.00 given the specified conditions. -/
theorem single_burger_cost_is_one :
  single_burger_cost 70.50 50 41 1.50 = 1 := by
  sorry

end single_burger_cost_is_one_l4059_405950


namespace bulk_warehouse_case_price_l4059_405966

/-- The price of a case at the bulk warehouse -/
def bulk_case_price (cans_per_case : ℕ) (grocery_cans : ℕ) (grocery_price : ℚ) (price_difference : ℚ) : ℚ :=
  let grocery_price_per_can : ℚ := grocery_price / grocery_cans
  let bulk_price_per_can : ℚ := grocery_price_per_can - price_difference
  cans_per_case * bulk_price_per_can

/-- Theorem stating that the price of a case at the bulk warehouse is $12.00 -/
theorem bulk_warehouse_case_price :
  bulk_case_price 48 12 6 (25/100) = 12 := by
  sorry

end bulk_warehouse_case_price_l4059_405966


namespace positive_integer_pairs_satisfying_equation_l4059_405938

theorem positive_integer_pairs_satisfying_equation :
  ∀ x y : ℕ+, 
    (x : ℤ)^2 + (y : ℤ)^2 - 5*(x : ℤ)*(y : ℤ) + 5 = 0 ↔ 
    ((x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2)) :=
by sorry

#check positive_integer_pairs_satisfying_equation

end positive_integer_pairs_satisfying_equation_l4059_405938


namespace parallelogram_base_l4059_405935

/-- Given a parallelogram with area 462 square centimeters and height 21 cm, its base is 22 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 462 → height = 21 → area = base * height → base = 22 := by sorry

end parallelogram_base_l4059_405935


namespace unique_prime_base_1021_l4059_405980

theorem unique_prime_base_1021 : ∃! (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^3 + 2*n + 1) :=
sorry

end unique_prime_base_1021_l4059_405980


namespace correct_line_representation_incorrect_representation_A_incorrect_representation_B_incorrect_representation_C_l4059_405934

-- Define a line in 2D space
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Theorem for the correct representation (Option D)
theorem correct_line_representation (n : ℝ) (k : ℝ) (h : k ≠ 0) :
  ∃ (l : Line), l.slope = k ∧ pointOnLine l ⟨n, 0⟩ ∧
    ∀ (x y : ℝ), pointOnLine l ⟨x, y⟩ ↔ x = k * y + n :=
sorry

-- Theorem for the incorrectness of Option A
theorem incorrect_representation_A (x₀ y₀ : ℝ) :
  ¬ (∀ (l : Line), ∃ (k : ℝ), ∀ (x y : ℝ),
    pointOnLine l ⟨x, y⟩ ↔ y - y₀ = k * (x - x₀)) :=
sorry

-- Theorem for the incorrectness of Option B
theorem incorrect_representation_B (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂ ∨ y₁ ≠ y₂) :
  ¬ (∀ (l : Line), ∀ (x y : ℝ),
    pointOnLine l ⟨x, y⟩ ↔ (y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁)) :=
sorry

-- Theorem for the incorrectness of Option C
theorem incorrect_representation_C :
  ¬ (∀ (l : Line) (a b : ℝ), (¬ pointOnLine l ⟨0, 0⟩) →
    (∀ (x y : ℝ), pointOnLine l ⟨x, y⟩ ↔ x / a + y / b = 1)) :=
sorry

end correct_line_representation_incorrect_representation_A_incorrect_representation_B_incorrect_representation_C_l4059_405934


namespace surface_area_of_drilled_cube_l4059_405993

-- Define the cube
def cube_side_length : ℝ := 10

-- Define points on the cube
def point_A : ℝ × ℝ × ℝ := (0, 0, 0)
def point_G : ℝ × ℝ × ℝ := (cube_side_length, cube_side_length, cube_side_length)

-- Define the distance of H, I, J from A
def distance_from_A : ℝ := 3

-- Define the solid T
def solid_T (cube_side_length : ℝ) (distance_from_A : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

-- Calculate the surface area of the solid T
def surface_area_T (t : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem surface_area_of_drilled_cube :
  surface_area_T (solid_T cube_side_length distance_from_A) = 526.5 := by sorry

end surface_area_of_drilled_cube_l4059_405993


namespace hyperbola_imaginary_axis_length_l4059_405927

/-- Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
    if the distance from its foci to the asymptote is 3,
    then the length of its imaginary axis is 6. -/
theorem hyperbola_imaginary_axis_length
  (b : ℝ)
  (h_b_pos : b > 0)
  (h_distance : b * Real.sqrt (4 + b^2) / Real.sqrt (4 + b^2) = 3) :
  2 * b = 6 :=
sorry

end hyperbola_imaginary_axis_length_l4059_405927


namespace sin_cos_sum_squared_l4059_405954

theorem sin_cos_sum_squared (x : Real) : 
  (Real.sin x + Real.cos x = Real.sqrt 2 / 2) → 
  (Real.sin x)^4 + (Real.cos x)^4 = 7/8 := by
sorry

end sin_cos_sum_squared_l4059_405954


namespace triangle_inequality_from_seven_numbers_l4059_405972

theorem triangle_inequality_from_seven_numbers
  (a : Fin 7 → ℝ)
  (h : ∀ i, 1 < a i ∧ a i < 13) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    a i + a j > a k ∧
    a j + a k > a i ∧
    a k + a i > a j :=
by sorry

end triangle_inequality_from_seven_numbers_l4059_405972


namespace solve_for_r_l4059_405930

theorem solve_for_r : 
  let r := Real.sqrt (8^2 + 15^2) / Real.sqrt 25
  r = 17 / 5 := by sorry

end solve_for_r_l4059_405930


namespace first_chapter_pages_l4059_405971

theorem first_chapter_pages (total_chapters : Nat) (second_chapter_pages : Nat) (third_chapter_pages : Nat) (total_pages : Nat)
  (h1 : total_chapters = 3)
  (h2 : second_chapter_pages = 35)
  (h3 : third_chapter_pages = 24)
  (h4 : total_pages = 125) :
  total_pages - (second_chapter_pages + third_chapter_pages) = 66 := by
  sorry

end first_chapter_pages_l4059_405971


namespace geometric_sequence_minimum_l4059_405988

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive sequence
  a 1 = 1 →  -- a_1 = 1
  a 7 = a 6 + 2 * a 5 →  -- a_7 = a_6 + 2a_5
  (∃ q : ℝ, q > 0 ∧ ∀ k, a (k + 1) = q * a k) →  -- geometric sequence
  a m * a n = 16 →  -- a_m * a_n = 16
  m > 0 ∧ n > 0 →  -- m and n are positive
  1 / m + 4 / n ≥ 3 / 2 :=
by sorry

end geometric_sequence_minimum_l4059_405988


namespace gilled_mushroom_count_l4059_405909

/-- Represents the types of mushrooms --/
inductive MushroomType
  | Spotted
  | Gilled

/-- Represents a mushroom --/
structure Mushroom where
  type : MushroomType

/-- Represents a collection of mushrooms on a log --/
structure MushroomLog where
  mushrooms : Finset Mushroom
  total_count : Nat
  spotted_count : Nat
  gilled_count : Nat
  h_total : total_count = mushrooms.card
  h_partition : total_count = spotted_count + gilled_count
  h_types : ∀ m ∈ mushrooms, m.type = MushroomType.Spotted ∨ m.type = MushroomType.Gilled
  h_ratio : spotted_count = 9 * gilled_count

theorem gilled_mushroom_count (log : MushroomLog) (h : log.total_count = 30) :
  log.gilled_count = 3 := by
  sorry

end gilled_mushroom_count_l4059_405909


namespace rectangle_max_area_l4059_405965

/-- Given a fixed perimeter, the area of a rectangle is maximized when it is a square -/
theorem rectangle_max_area (P : ℝ) (h : P > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = P / 2 →
  x * y ≤ (P / 4) ^ 2 ∧ 
  (x * y = (P / 4) ^ 2 ↔ x = y) := by
sorry


end rectangle_max_area_l4059_405965


namespace quadrilateral_sum_of_squares_l4059_405919

/-- A quadrilateral with sides a, b, c, d, diagonals m, n, and distance t between midpoints of diagonals -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  t : ℝ

/-- The sum of squares of sides equals the sum of squares of diagonals plus four times the square of the distance between midpoints of diagonals -/
theorem quadrilateral_sum_of_squares (q : Quadrilateral) :
  q.a^2 + q.b^2 + q.c^2 + q.d^2 = q.m^2 + q.n^2 + 4 * q.t^2 := by
  sorry

end quadrilateral_sum_of_squares_l4059_405919


namespace haley_cider_production_l4059_405914

/-- Represents the number of pints of cider Haley can make -/
def cider_pints (golden_per_pint pink_per_pint farmhands apples_per_hour work_hours golden_ratio pink_ratio : ℕ) : ℕ :=
  let total_apples := farmhands * apples_per_hour * work_hours
  let apples_per_pint := golden_per_pint + pink_per_pint
  total_apples / apples_per_pint

/-- Theorem stating that Haley can make 120 pints of cider given the conditions -/
theorem haley_cider_production :
  cider_pints 20 40 6 240 5 1 2 = 120 := by
  sorry

#eval cider_pints 20 40 6 240 5 1 2

end haley_cider_production_l4059_405914


namespace select_two_from_seven_l4059_405925

theorem select_two_from_seven : Nat.choose 7 2 = 21 := by
  sorry

end select_two_from_seven_l4059_405925


namespace feed_animals_theorem_l4059_405979

/-- The number of ways to feed animals in a conservatory -/
def feed_animals (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * n * feed_animals (n - 1)

/-- Theorem: Given 5 pairs of different animals, alternating between male and female,
    and starting with a female hippopotamus, there are 2880 ways to complete feeding all animals -/
theorem feed_animals_theorem : feed_animals 5 = 2880 := by
  sorry

end feed_animals_theorem_l4059_405979


namespace inverse_function_inequality_l4059_405901

/-- A function satisfying f(x₁x₂) = f(x₁) + f(x₂) for positive x₁ and x₂ -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

/-- Theorem statement for the given problem -/
theorem inverse_function_inequality (f : ℝ → ℝ) (hf : FunctionalEquation f)
    (hfinv : ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f) :
    ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
      f⁻¹ x₁ + f⁻¹ x₂ ≥ 2 * (f⁻¹ (x₁ / 2) * f⁻¹ (x₂ / 2)) :=
  sorry

end inverse_function_inequality_l4059_405901


namespace cindy_jump_rope_time_l4059_405987

/-- Cindy's jump rope time in minutes -/
def cindy_time : ℕ := 12

/-- Betsy's jump rope time in minutes -/
def betsy_time : ℕ := cindy_time / 2

/-- Tina's jump rope time in minutes -/
def tina_time : ℕ := 3 * betsy_time

theorem cindy_jump_rope_time :
  cindy_time = 12 ∧
  betsy_time = cindy_time / 2 ∧
  tina_time = 3 * betsy_time ∧
  tina_time = cindy_time + 6 :=
by sorry

end cindy_jump_rope_time_l4059_405987


namespace sally_pokemon_cards_l4059_405949

theorem sally_pokemon_cards (initial cards_from_dan cards_bought cards_traded cards_lost : ℕ) 
  (h1 : initial = 27)
  (h2 : cards_from_dan = 41)
  (h3 : cards_bought = 20)
  (h4 : cards_traded = 15)
  (h5 : cards_lost = 7) :
  initial + cards_from_dan + cards_bought - cards_traded - cards_lost = 66 := by
  sorry

end sally_pokemon_cards_l4059_405949


namespace apples_remaining_l4059_405913

def initial_apples : ℕ := 128
def sale_percentage : ℚ := 25 / 100

theorem apples_remaining (initial : ℕ) (sale_percent : ℚ) : 
  initial - ⌊initial * sale_percent⌋ - ⌊(initial - ⌊initial * sale_percent⌋) * sale_percent⌋ - 1 = 71 :=
by sorry

end apples_remaining_l4059_405913


namespace tomatoes_on_tuesday_eq_2500_l4059_405969

/-- Calculates the amount of tomatoes ready for sale on Tuesday given the initial shipment,
    sales, rotting, and new shipment. -/
def tomatoesOnTuesday (initialShipment sales rotted : ℕ) : ℕ :=
  let remainingAfterSales := initialShipment - sales
  let remainingAfterRotting := remainingAfterSales - rotted
  let newShipment := 2 * initialShipment
  remainingAfterRotting + newShipment

/-- Theorem stating that given the specific conditions, the amount of tomatoes
    ready for sale on Tuesday is 2500 kg. -/
theorem tomatoes_on_tuesday_eq_2500 :
  tomatoesOnTuesday 1000 300 200 = 2500 := by
  sorry

end tomatoes_on_tuesday_eq_2500_l4059_405969


namespace prob_at_least_one_event_l4059_405946

/-- The probability that at least one of two independent events occurs -/
theorem prob_at_least_one_event (A B : ℝ) (hA : 0 ≤ A ∧ A ≤ 1) (hB : 0 ≤ B ∧ B ≤ 1) 
  (hAval : A = 0.9) (hBval : B = 0.8) :
  1 - (1 - A) * (1 - B) = 0.98 := by
sorry

end prob_at_least_one_event_l4059_405946


namespace f_decreasing_interval_l4059_405955

-- Define the function f'(x)
def f' (x : ℝ) : ℝ := x^2 - 2*x - 3

-- State the theorem
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, f' x < 0 :=
sorry

end f_decreasing_interval_l4059_405955


namespace cafe_tables_l4059_405975

def base5_to_decimal (n : Nat) : Nat :=
  3 * 5^2 + 1 * 5^1 + 0 * 5^0

theorem cafe_tables :
  let total_chairs := base5_to_decimal 310
  let people_per_table := 3
  (total_chairs / people_per_table : Nat) = 26 := by
  sorry

end cafe_tables_l4059_405975


namespace sad_children_count_l4059_405983

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) : ℕ :=
  by
  -- Assume the given conditions
  have h1 : total = 60 := by sorry
  have h2 : happy = 30 := by sorry
  have h3 : neither = 20 := by sorry
  have h4 : boys = 16 := by sorry
  have h5 : girls = 44 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 4 := by sorry

  -- Prove that the number of sad children is 10
  exact total - happy - neither

end sad_children_count_l4059_405983


namespace teal_survey_l4059_405902

theorem teal_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : more_green = 90)
  (h3 : both = 40)
  (h4 : neither = 20) :
  ∃ more_blue : ℕ, more_blue = 80 ∧ 
    total = more_green + more_blue - both + neither :=
by sorry

end teal_survey_l4059_405902


namespace hyperbola_equation_l4059_405910

/-- Given a hyperbola with one focus at (2√5, 0) and asymptotes y = ±(1/2)x, 
    its standard equation is x²/16 - y²/4 = 1 -/
theorem hyperbola_equation (f : ℝ × ℝ) (m : ℝ) :
  f = (2 * Real.sqrt 5, 0) →
  m = 1/2 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔
      (y = m*x ∨ y = -m*x) ∧ 
      (x - f.1)^2 / a^2 - (y - f.2)^2 / b^2 = 1) ∧
    a^2 = 16 ∧ b^2 = 4 :=
by sorry

end hyperbola_equation_l4059_405910


namespace complex_sum_roots_of_unity_l4059_405960

theorem complex_sum_roots_of_unity (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (Finset.range 16).sum (λ k => ω^(20 + 4*k)) = -ω^2 := by sorry

end complex_sum_roots_of_unity_l4059_405960


namespace perpendicular_line_equation_l4059_405961

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation : 
  let l1 : Line := { a := 4, b := -5, c := 9 }
  let p : Point := { x := 4, y := 1 }
  let l2 : Line := { a := 5, b := 4, c := -24 }
  (l2.contains p ∧ Line.perpendicular l1 l2) → 
  ∀ (x y : ℝ), 5 * x + 4 * y - 24 = 0 ↔ l2.contains { x := x, y := y } :=
by
  sorry

end perpendicular_line_equation_l4059_405961


namespace train_speed_theorem_l4059_405928

-- Define the length of the train in meters
def train_length : ℝ := 300

-- Define the time taken to cross the platform in seconds
def crossing_time : ℝ := 60

-- Define the total distance covered (train length + platform length)
def total_distance : ℝ := 2 * train_length

-- Define the speed conversion factor from m/s to km/h
def speed_conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_theorem :
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * speed_conversion_factor
  speed_kmh = 36 := by
  sorry

end train_speed_theorem_l4059_405928


namespace sheridan_fish_problem_l4059_405967

/-- The number of fish Mrs. Sheridan gave to her sister -/
def fish_given (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem sheridan_fish_problem :
  fish_given 47.0 25 = 22 :=
by sorry

end sheridan_fish_problem_l4059_405967


namespace number_relations_with_180_l4059_405924

theorem number_relations_with_180 :
  (∃ n : ℤ, n = 180 + 15 ∧ n = 195) ∧
  (∃ m : ℤ, m = 180 - 15 ∧ m = 165) := by
  sorry

end number_relations_with_180_l4059_405924


namespace final_quantities_correct_l4059_405984

/-- Represents the inventory and transactions of a stationery shop -/
structure StationeryShop where
  x : ℝ  -- initial number of pencils
  y : ℝ  -- initial number of pens
  z : ℝ  -- initial number of rulers

/-- Calculates the final quantities after transactions -/
def finalQuantities (shop : StationeryShop) : ℝ × ℝ × ℝ :=
  let remainingPencils := shop.x * 0.75
  let remainingPens := shop.y * 0.60
  let remainingRulers := shop.z * 0.80
  let finalPencils := remainingPencils + remainingPencils * 2.50
  let finalPens := remainingPens + 100
  let finalRulers := remainingRulers + remainingRulers * 5
  (finalPencils, finalPens, finalRulers)

/-- Theorem stating the correctness of the final quantities calculation -/
theorem final_quantities_correct (shop : StationeryShop) :
  finalQuantities shop = (2.625 * shop.x, 0.60 * shop.y + 100, 4.80 * shop.z) := by
  sorry

end final_quantities_correct_l4059_405984


namespace water_in_sport_is_105_l4059_405916

/-- Represents the ratios of ingredients in a flavored drink formulation -/
structure DrinkFormulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the flavored drink -/
def standard : DrinkFormulation :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation of the flavored drink -/
def sport : DrinkFormulation :=
  { flavoring := standard.flavoring,
    corn_syrup := standard.corn_syrup / 3,
    water := standard.water * 2 }

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- Calculates the amount of water in the sport formulation -/
def water_in_sport : ℚ :=
  (sport_corn_syrup * sport.water) / sport.corn_syrup

/-- Theorem stating that the amount of water in the sport formulation is 105 ounces -/
theorem water_in_sport_is_105 : water_in_sport = 105 := by
  sorry


end water_in_sport_is_105_l4059_405916


namespace square_area_from_perimeter_l4059_405917

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4)^2 → area = 100 := by sorry

end square_area_from_perimeter_l4059_405917


namespace triangle_area_l4059_405995

/-- Given a triangle ABC with sides AC = 8 and BC = 10, and the condition that 32 cos(A - B) = 31,
    prove that the area of the triangle is 15√7. -/
theorem triangle_area (A B C : ℝ) (AC BC : ℝ) (h1 : AC = 8) (h2 : BC = 10) 
    (h3 : 32 * Real.cos (A - B) = 31) : 
    (1/2 : ℝ) * AC * BC * Real.sin (A + B - π) = 15 * Real.sqrt 7 := by
  sorry

end triangle_area_l4059_405995


namespace resort_tips_fraction_l4059_405962

theorem resort_tips_fraction (total_months : ℕ) (special_month_factor : ℕ) 
  (h1 : total_months = 7) 
  (h2 : special_month_factor = 4) : 
  (special_month_factor : ℚ) / ((total_months - 1 : ℕ) + special_month_factor : ℚ) = 2 / 5 := by
  sorry

end resort_tips_fraction_l4059_405962


namespace second_outlet_pipe_rate_l4059_405926

-- Define the volume of the tank in cubic inches
def tank_volume : ℝ := 30 * 1728

-- Define the inlet pipe rate in cubic inches per minute
def inlet_rate : ℝ := 3

-- Define the first outlet pipe rate in cubic inches per minute
def outlet_rate_1 : ℝ := 6

-- Define the time to empty the tank in minutes
def emptying_time : ℝ := 3456

-- Define the unknown rate of the second outlet pipe
def outlet_rate_2 : ℝ := 12

-- Theorem statement
theorem second_outlet_pipe_rate : 
  tank_volume / (outlet_rate_1 + outlet_rate_2 - inlet_rate) = emptying_time :=
sorry

end second_outlet_pipe_rate_l4059_405926


namespace max_gcd_sum_1980_l4059_405999

theorem max_gcd_sum_1980 :
  ∃ (a b : ℕ+), a + b = 1980 ∧
  ∀ (c d : ℕ+), c + d = 1980 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 990 :=
sorry

end max_gcd_sum_1980_l4059_405999


namespace zero_discriminant_implies_geometric_progression_l4059_405905

/-- Given a quadratic equation ax^2 + 3bx + c = 0 with zero discriminant,
    prove that a, b, and c form a geometric progression. -/
theorem zero_discriminant_implies_geometric_progression
  (a b c : ℝ) (h : 9 * b^2 - 4 * a * c = 0) :
  ∃ r : ℝ, b = a * r ∧ c = b * r :=
sorry

end zero_discriminant_implies_geometric_progression_l4059_405905


namespace square_area_from_perimeter_l4059_405964

/-- A square with perimeter 36 cm has an area of 81 cm² -/
theorem square_area_from_perimeter : 
  ∀ (s : ℝ), s > 0 → 4 * s = 36 → s^2 = 81 :=
by
  sorry


end square_area_from_perimeter_l4059_405964


namespace perpendicular_slope_l4059_405948

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let original_slope := a / b
  let perpendicular_slope := -1 / original_slope
  (5 : ℝ) * x - (4 : ℝ) * y = (20 : ℝ) → perpendicular_slope = -(4 : ℝ) / (5 : ℝ) :=
by
  sorry

end perpendicular_slope_l4059_405948


namespace baseball_price_proof_l4059_405937

/-- The price of a basketball in dollars -/
def basketball_price : ℝ := 29

/-- The number of basketballs bought by Coach A -/
def num_basketballs : ℕ := 10

/-- The number of baseballs bought by Coach B -/
def num_baseballs : ℕ := 14

/-- The price of the baseball bat in dollars -/
def bat_price : ℝ := 18

/-- The difference in spending between Coach A and Coach B in dollars -/
def spending_difference : ℝ := 237

/-- The price of a baseball in dollars -/
def baseball_price : ℝ := 2.5

theorem baseball_price_proof :
  num_basketballs * basketball_price = 
  num_baseballs * baseball_price + bat_price + spending_difference :=
by
  sorry

end baseball_price_proof_l4059_405937


namespace binary_division_remainder_l4059_405981

theorem binary_division_remainder (n : ℕ) (h : n = 0b111001011110) : n % 4 = 2 := by
  sorry

end binary_division_remainder_l4059_405981


namespace travel_time_to_madison_l4059_405911

/-- Represents the travel time problem from Gardensquare to Madison -/
theorem travel_time_to_madison 
  (map_distance : ℝ) 
  (map_scale : ℝ) 
  (average_speed : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : map_scale = 0.016666666666666666) 
  (h3 : average_speed = 60) : 
  map_distance / (map_scale * average_speed) = 5 := by
  sorry

end travel_time_to_madison_l4059_405911


namespace selling_price_l4059_405912

/-- Represents the labelled price of a refrigerator -/
def R : ℝ := sorry

/-- Represents the labelled price of a washing machine -/
def W : ℝ := sorry

/-- The condition that the total discounted price is 35000 -/
axiom purchase_price : 0.80 * R + 0.85 * W = 35000

/-- The theorem stating the selling price formula -/
theorem selling_price : 
  0.80 * R + 0.85 * W = 35000 → 
  (1.10 * R + 1.12 * W) = (1.10 * R + 1.12 * W) :=
by sorry

end selling_price_l4059_405912


namespace unique_solution_3644_l4059_405941

def repeating_decimal_ab (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

def repeating_decimal_abcd (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d : ℚ) / 9999

theorem unique_solution_3644 (a b c d : ℕ) :
  a ∈ Finset.range 10 →
  b ∈ Finset.range 10 →
  c ∈ Finset.range 10 →
  d ∈ Finset.range 10 →
  repeating_decimal_ab a b + repeating_decimal_abcd a b c d = 27 / 37 →
  a = 3 ∧ b = 6 ∧ c = 4 ∧ d = 4 :=
by sorry

end unique_solution_3644_l4059_405941


namespace chord_cosine_l4059_405974

theorem chord_cosine (r : ℝ) (γ δ : ℝ) : 
  0 < r →
  0 < γ →
  0 < δ →
  γ + δ < π →
  5^2 = 2 * r^2 * (1 - Real.cos γ) →
  12^2 = 2 * r^2 * (1 - Real.cos δ) →
  13^2 = 2 * r^2 * (1 - Real.cos (γ + δ)) →
  Real.cos γ = 7 / 25 := by
  sorry

end chord_cosine_l4059_405974


namespace inequality_proof_l4059_405991

theorem inequality_proof (a b c : ℝ) (M N P : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < 1)
  (hM : M = 2^a) (hN : N = 5^(-b)) (hP : P = Real.log c) :
  P < N ∧ N < M := by
  sorry

end inequality_proof_l4059_405991


namespace f_properties_l4059_405982

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) := by
  sorry

end f_properties_l4059_405982


namespace solve_system_l4059_405996

theorem solve_system (B C : ℝ) (eq1 : 5 * B - 3 = 32) (eq2 : 2 * B + 2 * C = 18) :
  B = 7 ∧ C = 2 := by
  sorry

end solve_system_l4059_405996


namespace complex_number_in_fourth_quadrant_l4059_405906

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 - Complex.I) / (2 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l4059_405906


namespace grasshopper_jumps_l4059_405976

/-- Represents the state of three objects in a line -/
inductive Position
| ABC
| ACB
| BAC
| BCA
| CAB
| CBA

/-- Represents a single jump of one object over another -/
def jump (p : Position) : Position :=
  match p with
  | Position.ABC => Position.BAC
  | Position.ACB => Position.CAB
  | Position.BAC => Position.BCA
  | Position.BCA => Position.CBA
  | Position.CAB => Position.ACB
  | Position.CBA => Position.ABC

/-- Applies n jumps to a given position -/
def jumpN (p : Position) (n : Nat) : Position :=
  match n with
  | 0 => p
  | n + 1 => jump (jumpN p n)

theorem grasshopper_jumps (n : Nat) (h : Odd n) :
  ∀ p : Position, jumpN p n ≠ p :=
sorry

end grasshopper_jumps_l4059_405976


namespace rhombus_dot_product_l4059_405990

/-- A rhombus OABC in a Cartesian coordinate system -/
structure Rhombus where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Vector representation -/
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem rhombus_dot_product (r : Rhombus) : 
  r.O = (0, 0) → 
  r.A = (1, 1) → 
  dot_product (vec r.O r.A) (vec r.O r.C) = 1 → 
  dot_product (vec r.A r.B) (vec r.A r.C) = 1 := by
  sorry

#check rhombus_dot_product

end rhombus_dot_product_l4059_405990


namespace parabola_a_range_l4059_405943

/-- Parabola defined by y = ax^2 - 2a^2x + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a : a ≠ 0
  h_c : c > 0

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = p.a * x^2 - 2 * p.a^2 * x + p.c

theorem parabola_a_range (p : Parabola) 
  (point1 : PointOnParabola p) 
  (point2 : PointOnParabola p)
  (h_x1 : point1.x = 2 * p.a + 1)
  (h_x2 : 2 ≤ point2.x ∧ point2.x ≤ 4)
  (h_y : point1.y > p.c ∧ p.c > point2.y) :
  p.a > 2 ∨ p.a < -1/2 := by
  sorry

end parabola_a_range_l4059_405943


namespace four_greater_than_sqrt_fifteen_l4059_405953

theorem four_greater_than_sqrt_fifteen : 4 > Real.sqrt 15 := by
  sorry

end four_greater_than_sqrt_fifteen_l4059_405953


namespace kevins_tshirts_l4059_405945

/-- Calculates the number of T-shirts Kevin can buy given the following conditions:
  * T-shirt price is $8
  * Sweater price is $18
  * Jacket original price is $80
  * Jacket discount is 10%
  * Sales tax is 5%
  * Kevin buys 4 sweaters and 5 jackets
  * Total payment including tax is $504
-/
theorem kevins_tshirts :
  let tshirt_price : ℚ := 8
  let sweater_price : ℚ := 18
  let jacket_original_price : ℚ := 80
  let jacket_discount : ℚ := 0.1
  let sales_tax : ℚ := 0.05
  let num_sweaters : ℕ := 4
  let num_jackets : ℕ := 5
  let total_payment : ℚ := 504

  let jacket_discounted_price := jacket_original_price * (1 - jacket_discount)
  let sweaters_cost := sweater_price * num_sweaters
  let jackets_cost := jacket_discounted_price * num_jackets
  let subtotal := sweaters_cost + jackets_cost
  let tax_amount := subtotal * sales_tax
  let total_without_tshirts := subtotal + tax_amount
  let amount_for_tshirts := total_payment - total_without_tshirts
  let num_tshirts := ⌊amount_for_tshirts / tshirt_price⌋

  num_tshirts = 6 := by sorry

end kevins_tshirts_l4059_405945


namespace monkeys_for_48_bananas_l4059_405958

/-- Given that 8 monkeys can eat 8 bananas in some time, 
    this function calculates the number of monkeys needed to eat 48 bananas in 48 minutes -/
def monkeys_needed (initial_monkeys : ℕ) (initial_bananas : ℕ) (target_bananas : ℕ) : ℕ :=
  initial_monkeys * (target_bananas / initial_bananas)

/-- Theorem stating that 48 monkeys are needed to eat 48 bananas in 48 minutes -/
theorem monkeys_for_48_bananas : monkeys_needed 8 8 48 = 48 := by
  sorry

end monkeys_for_48_bananas_l4059_405958


namespace matrix_not_invertible_sum_l4059_405918

/-- Given a 3x3 matrix with real entries x, y, z in the form
    [[x, y, z], [y, z, x], [z, x, y]],
    if the matrix is not invertible, then the sum
    x/(y+z) + y/(z+x) + z/(x+y) is equal to either -3 or 3/2 -/
theorem matrix_not_invertible_sum (x y z : ℝ) :
  let M := ![![x, y, z], ![y, z, x], ![z, x, y]]
  ¬ IsUnit (Matrix.det M) →
  (x / (y + z) + y / (z + x) + z / (x + y) = -3) ∨
  (x / (y + z) + y / (z + x) + z / (x + y) = 3/2) :=
by sorry

end matrix_not_invertible_sum_l4059_405918


namespace isosceles_triangle_proof_l4059_405947

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∨ t.angle1 = t.angle3 ∨ t.angle2 = t.angle3

-- Define the sum of angles in a triangle
def angleSum (t : Triangle) : ℝ :=
  t.angle1 + t.angle2 + t.angle3

-- Theorem statement
theorem isosceles_triangle_proof (t : Triangle) 
  (h1 : t.angle1 = 40)
  (h2 : t.angle2 = 70)
  (h3 : angleSum t = 180) :
  isIsosceles t :=
sorry

end isosceles_triangle_proof_l4059_405947


namespace gumball_distribution_l4059_405929

/-- Theorem: Gumball Distribution
Given:
- Joanna initially had 40 gumballs
- Jacques initially had 60 gumballs
- They each purchased 4 times their initial amount
- They put all gumballs together and shared equally

Prove: Each person gets 250 gumballs -/
theorem gumball_distribution (joanna_initial : Nat) (jacques_initial : Nat)
  (h1 : joanna_initial = 40)
  (h2 : jacques_initial = 60)
  (purchase_multiplier : Nat)
  (h3 : purchase_multiplier = 4) :
  let joanna_total := joanna_initial + joanna_initial * purchase_multiplier
  let jacques_total := jacques_initial + jacques_initial * purchase_multiplier
  let total_gumballs := joanna_total + jacques_total
  (total_gumballs / 2 : Nat) = 250 := by
  sorry

end gumball_distribution_l4059_405929


namespace manager_average_salary_l4059_405939

/-- Proves that the average salary of managers is $90,000 given the conditions of the company. -/
theorem manager_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (associate_avg_salary : ℚ) 
  (company_avg_salary : ℚ) : 
  num_managers = 15 → 
  num_associates = 75 → 
  associate_avg_salary = 30000 → 
  company_avg_salary = 40000 → 
  (num_managers * (num_managers * company_avg_salary - num_associates * associate_avg_salary)) / 
   (num_managers * (num_managers + num_associates)) = 90000 := by
  sorry

end manager_average_salary_l4059_405939


namespace a_max_at_6_l4059_405931

-- Define the sequence a_n
def a (n : ℤ) : ℚ := (10 / 11) ^ n * (3 * n + 13)

-- Theorem stating that a_n is maximized when n = 6
theorem a_max_at_6 : ∀ (k : ℤ), a 6 ≥ a k := by sorry

end a_max_at_6_l4059_405931


namespace R3_sequence_arithmetic_l4059_405994

def is_R3_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 3 → a (n - 3) + a (n + 3) = 2 * a n) ∧
  (∀ n : ℕ, a (n + 1) ≥ a n)

def is_arithmetic_subsequence (b : ℕ → ℝ) (start : ℕ) (step : ℕ) (count : ℕ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, i < count → b (start + i * step) - b (start + (i + 1) * step) = d

theorem R3_sequence_arithmetic (a : ℕ → ℝ) (h1 : is_R3_sequence a) 
  (h2 : ∃ p : ℕ, p > 1 ∧ is_arithmetic_subsequence a (3 * p - 3) 2 4) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d := by
  sorry

end R3_sequence_arithmetic_l4059_405994


namespace continuous_stripe_probability_l4059_405963

/-- A regular tetrahedron with painted stripes on each face -/
structure StripedTetrahedron :=
  (faces : Fin 4 → Fin 2)

/-- The probability of a specific stripe configuration -/
def stripe_probability : ℚ := 1 / 16

/-- A continuous stripe encircles the tetrahedron -/
def has_continuous_stripe (t : StripedTetrahedron) : Prop := sorry

/-- The number of stripe configurations that result in a continuous stripe -/
def continuous_stripe_count : ℕ := 2

theorem continuous_stripe_probability :
  (continuous_stripe_count : ℚ) * stripe_probability = 1 / 8 := by sorry

end continuous_stripe_probability_l4059_405963


namespace geometric_series_relation_l4059_405992

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 5/7 -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (∑' n, c / d^n) = 5) :
    (∑' n, c / (c + 2*d)^n) = 5/7 := by
  sorry

end geometric_series_relation_l4059_405992


namespace calculate_divisor_l4059_405920

/-- Given a dividend, quotient, and remainder, calculate the divisor -/
theorem calculate_divisor (dividend : ℝ) (quotient : ℝ) (remainder : ℝ) :
  dividend = 63584 ∧ quotient = 127.8 ∧ remainder = 45.5 →
  ∃ divisor : ℝ, divisor = 497.1 ∧ dividend = divisor * quotient + remainder :=
by sorry

end calculate_divisor_l4059_405920


namespace equal_roots_quadratic_equation_l4059_405942

theorem equal_roots_quadratic_equation :
  ∃! r : ℝ, ∀ x : ℝ, x^2 - r*x - r^2 = 0 → (∃! y : ℝ, y^2 - r*y - r^2 = 0) := by
  sorry

end equal_roots_quadratic_equation_l4059_405942


namespace transform_second_to_third_l4059_405977

/-- A point in the 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant. -/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Transforms a point according to the given rule. -/
def transformPoint (p : Point2D) : Point2D :=
  ⟨3 * p.x - 2, -p.y⟩

/-- Determines if a point is in the third quadrant. -/
def isInThirdQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- 
Theorem: If a point is in the second quadrant, 
then its transformed point is in the third quadrant.
-/
theorem transform_second_to_third (p : Point2D) :
  isInSecondQuadrant p → isInThirdQuadrant (transformPoint p) := by
  sorry

end transform_second_to_third_l4059_405977


namespace courier_cost_formula_l4059_405956

/-- The cost function for a courier service --/
def courier_cost (P : ℕ) : ℕ :=
  5 + 12 + 5 * (P - 1)

/-- Theorem: The courier cost is equal to 5P + 12 --/
theorem courier_cost_formula (P : ℕ) (h : P ≥ 1) : courier_cost P = 5 * P + 12 := by
  sorry

#check courier_cost_formula

end courier_cost_formula_l4059_405956


namespace square_difference_solutions_l4059_405922

theorem square_difference_solutions :
  (∀ x y : ℕ, x^2 - y^2 = 31 ↔ (x = 16 ∧ y = 15)) ∧
  (∀ x y : ℕ, x^2 - y^2 = 303 ↔ (x = 152 ∧ y = 151) ∨ (x = 52 ∧ y = 49)) := by
  sorry

end square_difference_solutions_l4059_405922


namespace f_properties_l4059_405985

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x - a * x + 1

theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x ∈ Set.Icc (2 - Real.exp 1) 1) ∧
  (∀ a ≤ 0, ∃! x, f a x = 0) ∧
  (∀ a > 0, ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z, f a z = 0 → z = x ∨ z = y) :=
by sorry

end f_properties_l4059_405985


namespace complex_magnitude_equality_l4059_405951

theorem complex_magnitude_equality (t : ℝ) (h : t > 0) :
  Complex.abs (-4 + 2 * t * Complex.I) = 3 * Real.sqrt 5 ↔ t = Real.sqrt 29 / 2 := by
  sorry

end complex_magnitude_equality_l4059_405951


namespace smallest_addend_for_divisibility_problem_solution_l4059_405923

theorem smallest_addend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n + k) % d = 0 ∧ ∀ (j : ℕ), j < k → (n + j) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 913475821
  let d := 13
  ∃ (k : ℕ), k = 2 ∧ k < d ∧ (n + k) % d = 0 ∧ ∀ (j : ℕ), j < k → (n + j) % d ≠ 0 :=
by
  sorry

end smallest_addend_for_divisibility_problem_solution_l4059_405923
