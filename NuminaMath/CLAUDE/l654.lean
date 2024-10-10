import Mathlib

namespace jacket_cost_calculation_l654_65458

def initial_amount : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def found_money : ℚ := 7.43

theorem jacket_cost_calculation : 
  let remaining_after_shirt := initial_amount - shirt_cost
  let total_remaining := remaining_after_shirt + found_money
  total_remaining = 9.28 := by sorry

end jacket_cost_calculation_l654_65458


namespace product_of_integers_l654_65451

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 
  2 * (a * b) + 14 * a = 5 * b + 290 → 
  a * b = 104 := by
sorry

end product_of_integers_l654_65451


namespace reciprocal_equal_reciprocal_equal_opposite_sign_l654_65463

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem for numbers equal to their own reciprocal
theorem reciprocal_equal (x : ℝ) : x = 1 / x ↔ x = 1 ∨ x = -1 := by sorry

-- Theorem for numbers equal to their own reciprocal with opposite sign
theorem reciprocal_equal_opposite_sign (y : ℂ) : y = -1 / y ↔ y = i ∨ y = -i := by sorry

end reciprocal_equal_reciprocal_equal_opposite_sign_l654_65463


namespace product_104_96_l654_65420

theorem product_104_96 : 104 * 96 = 9984 := by
  sorry

end product_104_96_l654_65420


namespace inequality_max_a_l654_65496

theorem inequality_max_a : 
  (∀ x : ℝ, x ∈ Set.Icc 1 12 → x^2 + 25 + |x^3 - 5*x^2| ≥ (5/2)*x) ∧ 
  (∀ ε > 0, ∃ x : ℝ, x ∈ Set.Icc 1 12 ∧ x^2 + 25 + |x^3 - 5*x^2| < (5/2 + ε)*x) :=
by sorry

end inequality_max_a_l654_65496


namespace k_is_even_if_adjacent_to_odds_l654_65445

/-- A circular arrangement of numbers from 1 to 1000 -/
def CircularArrangement := Fin 1000 → ℕ

/-- Property that each number is a divisor of the sum of its neighbors -/
def IsDivisorOfNeighborsSum (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 1000, arr i ∣ (arr (i - 1) + arr (i + 1))

/-- Theorem: If k is adjacent to two odd numbers in a valid circular arrangement, then k is even -/
theorem k_is_even_if_adjacent_to_odds
  (arr : CircularArrangement)
  (h_valid : IsDivisorOfNeighborsSum arr)
  (k : Fin 1000)
  (h_k_adj_odd : Odd (arr (k - 1)) ∧ Odd (arr (k + 1))) :
  Even (arr k) := by
  sorry

end k_is_even_if_adjacent_to_odds_l654_65445


namespace cookies_per_pack_l654_65449

theorem cookies_per_pack (trays : ℕ) (cookies_per_tray : ℕ) (packs : ℕ) 
  (h1 : trays = 8) 
  (h2 : cookies_per_tray = 36) 
  (h3 : packs = 12) :
  (trays * cookies_per_tray) / packs = 24 := by
  sorry

#check cookies_per_pack

end cookies_per_pack_l654_65449


namespace arithmetic_sequence_length_l654_65412

theorem arithmetic_sequence_length :
  ∀ (a₁ aₙ d n : ℤ),
    a₁ = -38 →
    aₙ = 69 →
    d = 6 →
    aₙ = a₁ + (n - 1) * d →
    n = 18 :=
by
  sorry

end arithmetic_sequence_length_l654_65412


namespace dessert_preference_theorem_l654_65475

/-- Represents the dessert preferences of a group of students -/
structure DessertPreferences where
  total : ℕ
  apple : ℕ
  chocolate : ℕ
  carrot : ℕ
  none : ℕ
  apple_chocolate_not_carrot : ℕ

/-- The theorem stating the number of students who like both apple pie and chocolate cake but not carrot cake -/
theorem dessert_preference_theorem (prefs : DessertPreferences) : 
  prefs.total = 50 ∧ 
  prefs.apple = 23 ∧ 
  prefs.chocolate = 20 ∧ 
  prefs.carrot = 10 ∧ 
  prefs.none = 15 → 
  prefs.apple_chocolate_not_carrot = 7 := by
  sorry

end dessert_preference_theorem_l654_65475


namespace cosine_of_point_on_terminal_side_l654_65483

def point_on_terminal_side (α : Real) (x y : Real) : Prop :=
  ∃ t : Real, t > 0 ∧ x = t * Real.cos α ∧ y = t * Real.sin α

theorem cosine_of_point_on_terminal_side (α : Real) :
  point_on_terminal_side α (-3) 4 → Real.cos α = -3/5 := by
  sorry

end cosine_of_point_on_terminal_side_l654_65483


namespace fraction_sum_equals_seven_l654_65471

theorem fraction_sum_equals_seven : 
  let U := (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - Real.sqrt 14)) + 
           (1 / (Real.sqrt 14 - Real.sqrt 13)) - (1 / (Real.sqrt 13 - Real.sqrt 12)) + 
           (1 / (Real.sqrt 12 - 3))
  U = 7 := by sorry

end fraction_sum_equals_seven_l654_65471


namespace power_of_four_l654_65493

theorem power_of_four (x : ℕ) 
  (h1 : 2 * x + 5 + 2 = 29) : x = 11 := by
  sorry

#check power_of_four

end power_of_four_l654_65493


namespace right_triangle_is_stable_l654_65462

-- Define the concept of a shape
structure Shape :=
  (name : String)

-- Define the property of stability
def is_stable (s : Shape) : Prop := sorry

-- Define a right triangle
def right_triangle : Shape :=
  { name := "Right Triangle" }

-- Define structural rigidity
def has_structural_rigidity (s : Shape) : Prop := sorry

-- Define resistance to deformation
def resists_deformation (s : Shape) : Prop := sorry

-- Theorem: A right triangle is stable
theorem right_triangle_is_stable :
  has_structural_rigidity right_triangle →
  resists_deformation right_triangle →
  is_stable right_triangle :=
by sorry

end right_triangle_is_stable_l654_65462


namespace geometric_series_sum_l654_65479

theorem geometric_series_sum : 
  let a : ℚ := 3 / 4
  let r : ℚ := 3 / 4
  let n : ℕ := 15
  let series_sum := (a * (1 - r^n)) / (1 - r)
  series_sum = 3216929751 / 1073741824 := by
  sorry

end geometric_series_sum_l654_65479


namespace sqrt_expression_equals_six_l654_65486

theorem sqrt_expression_equals_six :
  (Real.sqrt 27 - 3 * Real.sqrt (1/3)) / (1 / Real.sqrt 3) = 6 := by
  sorry

end sqrt_expression_equals_six_l654_65486


namespace sugar_for_recipe_l654_65481

/-- The amount of sugar needed for a cake recipe, given the amounts for frosting and cake. -/
theorem sugar_for_recipe (frosting_sugar cake_sugar : ℚ) 
  (h1 : frosting_sugar = 0.6)
  (h2 : cake_sugar = 0.2) : 
  frosting_sugar + cake_sugar = 0.8 := by
  sorry

end sugar_for_recipe_l654_65481


namespace remainder_theorem_l654_65446

theorem remainder_theorem (x : ℤ) : 
  (2*x + 3)^504 ≡ 16*x + 5 [ZMOD (x^2 - x + 1)] :=
sorry

end remainder_theorem_l654_65446


namespace expression_equals_8_175_l654_65473

-- Define the expression
def expression : ℝ := (4.5 - 1.23) * 2.5

-- State the theorem
theorem expression_equals_8_175 : expression = 8.175 := by
  sorry

end expression_equals_8_175_l654_65473


namespace max_stamps_with_50_dollars_l654_65427

theorem max_stamps_with_50_dollars (stamp_price : ℚ) (total_money : ℚ) :
  stamp_price = 25 / 100 →
  total_money = 50 →
  ⌊total_money / stamp_price⌋ = 200 := by
  sorry

end max_stamps_with_50_dollars_l654_65427


namespace vector_operations_l654_65453

/-- Given vectors a and b, prove their sum and dot product -/
theorem vector_operations (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, 2, 2)) (hb : b = (6, -3, 2)) : 
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2) = (7, -1, 4) ∧ 
  (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2) = 4 := by
  sorry

end vector_operations_l654_65453


namespace farm_animals_feet_count_l654_65405

theorem farm_animals_feet_count (total_heads : Nat) (hen_count : Nat) : 
  total_heads = 60 → hen_count = 20 → (total_heads - hen_count) * 4 + hen_count * 2 = 200 := by
  sorry

end farm_animals_feet_count_l654_65405


namespace area_of_four_squares_l654_65440

/-- The area of a shape composed of four identical squares with side length 3 cm -/
theorem area_of_four_squares : 
  ∀ (side_length : ℝ) (num_squares : ℕ),
    side_length = 3 →
    num_squares = 4 →
    (num_squares : ℝ) * (side_length^2) = 36 := by
  sorry

end area_of_four_squares_l654_65440


namespace shelf_filling_l654_65438

/-- Given a shelf that can be filled with books, this theorem relates the number of
    physics and chemistry books needed to fill it. -/
theorem shelf_filling (P C R B G : ℕ) : 
  (P > 0) → (C > 0) → (R > 0) → (B > 0) → (G > 0) →  -- Positive integers
  (P ≠ C) → (P ≠ R) → (P ≠ B) → (P ≠ G) →  -- Distinct values
  (C ≠ R) → (C ≠ B) → (C ≠ G) →
  (R ≠ B) → (R ≠ G) →
  (B ≠ G) →
  (∃ (x : ℚ), x > 0 ∧ P * x + 2 * C * x = G * x) →  -- Shelf filling condition
  (∃ (x : ℚ), x > 0 ∧ R * x + 2 * B * x = G * x) →  -- Alternative filling
  G = P + 2 * C :=
by sorry

end shelf_filling_l654_65438


namespace max_salad_servings_is_56_l654_65465

/-- Represents the ingredients required for one serving of salad -/
structure SaladServing where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the restaurant's warehouse -/
structure WarehouseStock where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of salad servings that can be made -/
def maxSaladServings (serving : SaladServing) (stock : WarehouseStock) : ℕ :=
  min
    (stock.cucumbers / serving.cucumbers)
    (min
      (stock.tomatoes / serving.tomatoes)
      (min
        (stock.brynza / serving.brynza)
        (stock.peppers / serving.peppers)))

/-- Theorem stating that the maximum number of salad servings is 56 -/
theorem max_salad_servings_is_56 :
  let serving := SaladServing.mk 2 2 75 1
  let stock := WarehouseStock.mk 117 116 4200 60
  maxSaladServings serving stock = 56 := by
  sorry

#eval maxSaladServings (SaladServing.mk 2 2 75 1) (WarehouseStock.mk 117 116 4200 60)

end max_salad_servings_is_56_l654_65465


namespace quadratic_term_coefficient_and_constant_term_l654_65460

/-- Represents a quadratic equation in the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation -3x² - 2x = 0 -/
def givenEquation : QuadraticEquation :=
  { a := -3, b := -2, c := 0 }

theorem quadratic_term_coefficient_and_constant_term :
  (givenEquation.a = -3) ∧ (givenEquation.c = 0) := by
  sorry

end quadratic_term_coefficient_and_constant_term_l654_65460


namespace max_value_quadratic_l654_65467

theorem max_value_quadratic (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, x^2 - a*x - a ≤ 1) ∧ 
  (∃ x ∈ Set.Icc 0 2, x^2 - a*x - a = 1) → 
  a = 1 := by
sorry

end max_value_quadratic_l654_65467


namespace gcd_78_36_l654_65409

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end gcd_78_36_l654_65409


namespace complex_modulus_problem_l654_65418

theorem complex_modulus_problem (z : ℂ) (h : (3 - I) / z = 1 + I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l654_65418


namespace degree_to_radian_conversion_l654_65404

theorem degree_to_radian_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    -885 * (π / 180) = 2 * k * π + α ∧
    0 ≤ α ∧ α ≤ 2 * π ∧
    2 * k * π + α = -6 * π + 13 * π / 12 :=
by sorry

end degree_to_radian_conversion_l654_65404


namespace transversal_exists_l654_65488

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- Two lines are in general position (not parallel or coincident) -/
def general_position (l1 l2 : Line3D) : Prop := 
  ¬ parallel l1 l2 ∧ ¬ (l1 = l2)

/-- Theorem: Existence of a transversal parallel to a given line -/
theorem transversal_exists (l1 l2 lp : Line3D) 
  (h : general_position l1 l2) : 
  ∃ lt : Line3D, intersect lt l1 ∧ intersect lt l2 ∧ parallel lt lp := by
  sorry

end transversal_exists_l654_65488


namespace time_difference_l654_65491

theorem time_difference (brian_time todd_time : ℕ) 
  (h1 : brian_time = 96) 
  (h2 : todd_time = 88) : 
  brian_time - todd_time = 8 := by
sorry

end time_difference_l654_65491


namespace gerald_toy_cars_l654_65477

theorem gerald_toy_cars (initial_cars : ℕ) (donation_fraction : ℚ) (remaining_cars : ℕ) :
  initial_cars = 20 →
  donation_fraction = 1 / 4 →
  remaining_cars = initial_cars - (initial_cars * donation_fraction).floor →
  remaining_cars = 15 := by
  sorry

end gerald_toy_cars_l654_65477


namespace squared_inequality_condition_l654_65413

theorem squared_inequality_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end squared_inequality_condition_l654_65413


namespace cube_volume_problem_l654_65406

theorem cube_volume_problem (x : ℝ) (h : x > 0) (eq : x^3 + 6*x^2 = 16*x) :
  27 * x^3 = 216 := by sorry

end cube_volume_problem_l654_65406


namespace parallelogram_area_l654_65403

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area : 
  ∀ (base height : ℝ), 
  base = 20 → 
  height = 4 → 
  base * height = 80 := by
sorry

end parallelogram_area_l654_65403


namespace sum_quadratic_distinct_roots_l654_65454

/-- A quadratic function f(x) = x^2 + ax + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- The discriminant of a quadratic function -/
def discriminant (f : QuadraticFunction) : ℝ := f.a^2 - 4*f.b

/-- The sum of two quadratic functions -/
def sum_quadratic (f g : QuadraticFunction) : QuadraticFunction :=
  ⟨f.a + g.a, f.b + g.b⟩

/-- The sum of a list of quadratic functions -/
def sum_quadratic_list (fs : List QuadraticFunction) : QuadraticFunction :=
  fs.foldl sum_quadratic ⟨0, 0⟩

/-- Theorem: Given conditions on quadratic functions, their sum has distinct real roots -/
theorem sum_quadratic_distinct_roots
  (n : ℕ)
  (hn : n ≥ 2)
  (fs : List QuadraticFunction)
  (hfs : fs.length = n)
  (h_same_discriminant : ∀ (f g : QuadraticFunction), f ∈ fs → g ∈ fs → discriminant f = discriminant g)
  (h_distinct_roots : ∀ (f g : QuadraticFunction), f ∈ fs → g ∈ fs → f ≠ g →
    (discriminant (sum_quadratic f g) > 0)) :
  discriminant (sum_quadratic_list fs) > 0 := by
  sorry

end sum_quadratic_distinct_roots_l654_65454


namespace son_age_theorem_l654_65437

/-- Represents the ages of three generations in a family -/
structure FamilyAges where
  grandson_days : ℕ
  son_months : ℕ
  grandfather_years : ℕ

/-- Calculates the son's age in weeks given the family ages -/
def son_age_weeks (ages : FamilyAges) : ℕ :=
  ages.son_months * 4 -- Approximate weeks in a month

/-- The main theorem stating the son's age in weeks -/
theorem son_age_theorem (ages : FamilyAges) : 
  ages.grandson_days = ages.son_months ∧ 
  ages.grandson_days / 30 = ages.grandfather_years ∧ 
  ages.grandson_days / 360 + ages.son_months / 12 + ages.grandfather_years = 140 ∧ 
  ages.grandfather_years = 84 →
  son_age_weeks ages = 2548 := by
  sorry

#eval son_age_weeks { grandson_days := 2520, son_months := 588, grandfather_years := 84 }

end son_age_theorem_l654_65437


namespace adjusted_equilateral_triangle_l654_65492

/-- Given a triangle XYZ that was originally equilateral, prove that if angle X is decreased by 5 degrees, 
    then angles Y and Z will each measure 62.5 degrees. -/
theorem adjusted_equilateral_triangle (X Y Z : ℝ) : 
  X + Y + Z = 180 →  -- Sum of angles in a triangle is 180°
  X = 55 →           -- Angle X after decrease
  Y = Z →            -- Angles Y and Z remain equal
  Y = 62.5 ∧ Z = 62.5 := by
sorry

end adjusted_equilateral_triangle_l654_65492


namespace regular_tetradecagon_side_length_l654_65476

/-- A regular tetradecagon with perimeter 154 cm has sides of length 11 cm. -/
theorem regular_tetradecagon_side_length :
  ∀ (perimeter : ℝ) (num_sides : ℕ) (side_length : ℝ),
    perimeter = 154 →
    num_sides = 14 →
    side_length * num_sides = perimeter →
    side_length = 11 :=
by sorry

end regular_tetradecagon_side_length_l654_65476


namespace math_team_combinations_l654_65470

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of girls in the math club. -/
def num_girls : ℕ := 4

/-- The number of boys in the math club. -/
def num_boys : ℕ := 6

/-- The number of girls required in the team. -/
def girls_in_team : ℕ := 3

/-- The number of boys required in the team. -/
def boys_in_team : ℕ := 4

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 60 := by
  sorry

end math_team_combinations_l654_65470


namespace pizza_theorem_l654_65421

/-- Calculates the number of pizza slices remaining after a series of consumption events. -/
def remainingSlices (initialSlices : ℕ) : ℕ :=
  let afterLunch := initialSlices / 2
  let afterDinner := afterLunch - (afterLunch / 3)
  let afterSharing := afterDinner - (afterDinner / 4)
  afterSharing - (afterSharing / 5)

/-- Theorem stating that given 12 initial slices, 3 slices remain after the described events. -/
theorem pizza_theorem : remainingSlices 12 = 3 := by
  sorry

end pizza_theorem_l654_65421


namespace cookie_problem_l654_65452

theorem cookie_problem (total_cookies : ℕ) (nuts_per_cookie : ℕ) (fraction_with_nuts : ℚ) (total_nuts_used : ℕ) :
  nuts_per_cookie = 2 →
  fraction_with_nuts = 1/4 →
  total_nuts_used = 72 →
  (fraction_with_nuts * total_cookies : ℚ).num * nuts_per_cookie = total_nuts_used →
  total_cookies = 144 :=
by sorry

end cookie_problem_l654_65452


namespace parallel_resistor_calculation_l654_65410

/-- Calculates the resistance of the second resistor in a parallel circuit -/
theorem parallel_resistor_calculation (R1 R_total : ℝ) (h1 : R1 = 9) (h2 : R_total = 4.235294117647059) :
  ∃ R2 : ℝ, R2 = 8 ∧ 1 / R_total = 1 / R1 + 1 / R2 := by
sorry

end parallel_resistor_calculation_l654_65410


namespace power_of_power_l654_65431

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end power_of_power_l654_65431


namespace sin_cos_pi_twelve_eq_one_fourth_l654_65407

theorem sin_cos_pi_twelve_eq_one_fourth : 
  Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by sorry

end sin_cos_pi_twelve_eq_one_fourth_l654_65407


namespace house_wall_planks_l654_65436

/-- The number of large planks needed for the house wall. -/
def large_planks : ℕ := 12

/-- The number of small planks needed for the house wall. -/
def small_planks : ℕ := 17

/-- The total number of planks needed for the house wall. -/
def total_planks : ℕ := large_planks + small_planks

theorem house_wall_planks : total_planks = 29 := by
  sorry

end house_wall_planks_l654_65436


namespace spencer_jumps_l654_65474

/-- Calculates the total number of jumps Spencer will do in 5 days -/
def total_jumps (jumps_per_minute : ℕ) (minutes_per_session : ℕ) (sessions_per_day : ℕ) (days : ℕ) : ℕ :=
  jumps_per_minute * minutes_per_session * sessions_per_day * days

/-- Theorem stating that Spencer will do 400 jumps in 5 days -/
theorem spencer_jumps :
  total_jumps 4 10 2 5 = 400 := by
  sorry

end spencer_jumps_l654_65474


namespace solution_set_f_min_value_sum_equality_condition_l654_65478

-- Define the function f
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

-- Theorem 1: Solution set of f(x + 3/2) ≥ 0
theorem solution_set_f (x : ℝ) : 
  f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) := by sorry

-- Theorem 2: Minimum value of 3p + 2q + r
theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 := by sorry

-- Theorem 3: Condition for equality in Theorem 2
theorem equality_condition (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r = 9/4 ↔ p = 1/4 ∧ q = 3/8 ∧ r = 3/4 := by sorry

end solution_set_f_min_value_sum_equality_condition_l654_65478


namespace range_of_a_l654_65443

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2 * x * (3 * x + a) < 1) ↔ a < 1 := by
  sorry

end range_of_a_l654_65443


namespace pet_store_combinations_l654_65459

def num_puppies : ℕ := 15
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 8
def num_people : ℕ := 3

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * Nat.factorial num_people = 7200 := by
  sorry

end pet_store_combinations_l654_65459


namespace spiral_grid_sum_third_row_l654_65485

/-- Represents a square grid with side length n -/
def Grid (n : ℕ) := Fin n → Fin n → ℕ

/-- Fills the grid in a clockwise spiral starting from the center -/
def fillSpiral (n : ℕ) : Grid n :=
  sorry

/-- Returns the largest number in a given row of the grid -/
def largestInRow (g : Grid 17) (row : Fin 17) : ℕ :=
  sorry

/-- Returns the smallest number in a given row of the grid -/
def smallestInRow (g : Grid 17) (row : Fin 17) : ℕ :=
  sorry

theorem spiral_grid_sum_third_row :
  let g := fillSpiral 17
  let thirdRow : Fin 17 := 2
  (largestInRow g thirdRow) + (smallestInRow g thirdRow) = 526 :=
by sorry

end spiral_grid_sum_third_row_l654_65485


namespace hot_dogs_remainder_l654_65464

theorem hot_dogs_remainder (total : Nat) (package_size : Nat) : 
  total = 25197624 → package_size = 4 → total % package_size = 0 := by
  sorry

end hot_dogs_remainder_l654_65464


namespace larger_number_problem_l654_65416

theorem larger_number_problem (S L : ℕ) 
  (h1 : L - S = 50000)
  (h2 : L = 13 * S + 317) :
  L = 54140 := by
  sorry

end larger_number_problem_l654_65416


namespace classroom_children_l654_65411

theorem classroom_children (total_pencils : ℕ) (pencils_per_student : ℕ) (h1 : total_pencils = 8) (h2 : pencils_per_student = 2) :
  total_pencils / pencils_per_student = 4 :=
by sorry

end classroom_children_l654_65411


namespace quadratic_minimum_l654_65482

/-- The function f(x) = 5x^2 - 20x + 7 has a minimum value when x = 2 -/
theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ (y : ℝ), 5 * x^2 - 20 * x + 7 ≥ 5 * min_x^2 - 20 * min_x + 7 ∧ min_x = 2 := by
  sorry

end quadratic_minimum_l654_65482


namespace larger_number_problem_l654_65489

theorem larger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 55) 
  (diff_eq : x - y = 15) : 
  x = 35 := by
sorry

end larger_number_problem_l654_65489


namespace min_value_theorem_l654_65484

theorem min_value_theorem (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w))) + 
  (1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) ≥ 2 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0) * (1 - 0))) + 
  (1 / ((1 + 0) * (1 + 0) * (1 + 0) * (1 + 0))) = 2 := by
  sorry

end min_value_theorem_l654_65484


namespace base_10_to_base_8_l654_65419

theorem base_10_to_base_8 : 
  ∃ (a b c d : ℕ), 
    947 = a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0 ∧ 
    a = 1 ∧ b = 6 ∧ c = 6 ∧ d = 3 := by
  sorry

end base_10_to_base_8_l654_65419


namespace relationship_holds_l654_65435

/-- The function describing the relationship between x and y -/
def f (x : ℕ) : ℕ := x^2 - 3*x + 2

/-- The set of x values given in the table -/
def X : Set ℕ := {2, 3, 4, 5, 6}

/-- The proposition that the function f correctly describes the relationship for all x in X -/
theorem relationship_holds (x : ℕ) (h : x ∈ X) : 
  (x = 2 → f x = 0) ∧ 
  (x = 3 → f x = 2) ∧ 
  (x = 4 → f x = 6) ∧ 
  (x = 5 → f x = 12) ∧ 
  (x = 6 → f x = 20) :=
by sorry

end relationship_holds_l654_65435


namespace polyhedron_volume_l654_65495

/-- Represents a polygon in the figure -/
inductive Polygon
| ScaleneRightTriangle : Polygon
| Rectangle : Polygon
| EquilateralTriangle : Polygon

/-- The figure consisting of multiple polygons -/
structure Figure where
  scaleneTriangles : Fin 3 → Polygon
  rectangles : Fin 3 → Polygon
  equilateralTriangle : Polygon
  scaleneTriangleLegs : ℝ × ℝ
  rectangleDimensions : ℝ × ℝ

/-- The polyhedron formed by folding the figure -/
def Polyhedron (f : Figure) : Type := Unit

/-- The volume of the polyhedron -/
noncomputable def volume (p : Polyhedron f) : ℝ := sorry

/-- The main theorem stating the volume of the polyhedron is 4 -/
theorem polyhedron_volume (f : Figure)
  (h1 : ∀ i, f.scaleneTriangles i = Polygon.ScaleneRightTriangle)
  (h2 : ∀ i, f.rectangles i = Polygon.Rectangle)
  (h3 : f.equilateralTriangle = Polygon.EquilateralTriangle)
  (h4 : f.scaleneTriangleLegs = (1, 2))
  (h5 : f.rectangleDimensions = (1, 2))
  (p : Polyhedron f) :
  volume p = 4 := by sorry

end polyhedron_volume_l654_65495


namespace polygon_construction_possible_l654_65469

/-- Represents a line segment with a fixed length -/
structure Segment where
  length : ℝ

/-- Represents a polygon constructed from line segments -/
structure Polygon where
  segments : List Segment

/-- Calculates the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Checks if all segments in a polygon are used -/
def allSegmentsUsed (p : Polygon) (segments : List Segment) : Prop := sorry

theorem polygon_construction_possible (segments : List Segment) :
  segments.length = 12 ∧ 
  ∀ s ∈ segments, s.length = 2 →
  ∃ p : Polygon, calculateArea p = 16 ∧ allSegmentsUsed p segments :=
sorry

end polygon_construction_possible_l654_65469


namespace inequality_solution_set_l654_65432

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 - x - 3 > 0) ↔ (x > 3/2 ∨ x < -1) := by
sorry

end inequality_solution_set_l654_65432


namespace parabola_complementary_lines_l654_65498

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

theorem parabola_complementary_lines (para : Parabola) (P A B : Point)
  (h_P_on_para : P.y^2 = 2 * para.p * P.x)
  (h_P_y_pos : P.y > 0)
  (h_A_on_para : A.y^2 = 2 * para.p * A.x)
  (h_B_on_para : B.y^2 = 2 * para.p * B.x)
  (h_PA_slope_exists : A.x ≠ P.x)
  (h_PB_slope_exists : B.x ≠ P.x)
  (h_complementary : 
    (A.y - P.y) / (A.x - P.x) * (B.y - P.y) / (B.x - P.x) = -1) :
  (A.y + B.y) / P.y = -2 := by
  sorry

end parabola_complementary_lines_l654_65498


namespace dropped_student_score_l654_65497

theorem dropped_student_score 
  (initial_students : ℕ) 
  (remaining_students : ℕ) 
  (initial_average : ℚ) 
  (new_average : ℚ) 
  (h1 : initial_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 61.5) 
  (h4 : new_average = 64) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 24 := by
  sorry

#check dropped_student_score

end dropped_student_score_l654_65497


namespace expression_evaluation_l654_65415

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := by
  sorry

end expression_evaluation_l654_65415


namespace system_solution_l654_65402

theorem system_solution (a b c d : ℚ) 
  (eq1 : 4 * a + 2 * b + 6 * c + 8 * d = 48)
  (eq2 : 4 * d + 2 * c = 2 * b)
  (eq3 : 4 * b + 2 * c = 2 * a)
  (eq4 : c + 2 = d) :
  a * b * c * d = -11033 / 1296 := by
sorry

end system_solution_l654_65402


namespace function_inequality_range_l654_65430

theorem function_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
by sorry

end function_inequality_range_l654_65430


namespace ring_arrangements_count_l654_65408

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of possible four-ring arrangements on four fingers of one hand,
    given seven distinguishable rings, where the order matters and not all
    fingers need to have a ring -/
def ring_arrangements : ℕ :=
  choose 7 4 * factorial 4 * choose 7 3

theorem ring_arrangements_count :
  ring_arrangements = 29400 := by sorry

end ring_arrangements_count_l654_65408


namespace equality_of_expressions_l654_65466

theorem equality_of_expressions (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end equality_of_expressions_l654_65466


namespace representation_of_2015_l654_65422

theorem representation_of_2015 : ∃ (a b c : ℤ),
  a + b + c = 2015 ∧
  Nat.Prime a.natAbs ∧
  ∃ (k : ℤ), b = 3 * k ∧
  400 < c ∧ c < 500 ∧
  ¬∃ (m : ℤ), c = 3 * m :=
by sorry

end representation_of_2015_l654_65422


namespace total_trees_formula_l654_65450

/-- The total number of trees planted by three teams under specific conditions -/
def total_trees (a : ℕ) : ℕ :=
  let team1 := a
  let team2 := 2 * a + 8
  let team3 := (team2 / 2) - 6
  team1 + team2 + team3

/-- Theorem stating the total number of trees planted by the three teams -/
theorem total_trees_formula (a : ℕ) : total_trees a = 4 * a + 6 := by
  sorry

#eval total_trees 100  -- Should output 406

end total_trees_formula_l654_65450


namespace base_k_addition_l654_65401

/-- Represents a digit in base k -/
def Digit (k : ℕ) := Fin k

/-- Converts a natural number to its representation in base k -/
def toBaseK (n : ℕ) (k : ℕ) : List (Digit k) :=
  sorry

/-- Adds two numbers represented in base k -/
def addBaseK (a b : List (Digit k)) : List (Digit k) :=
  sorry

/-- Checks if two lists of digits are equal -/
def digitListEq (a b : List (Digit k)) : Prop :=
  sorry

theorem base_k_addition :
  ∃ k : ℕ, k > 1 ∧
    digitListEq
      (addBaseK (toBaseK 8374 k) (toBaseK 9423 k))
      (toBaseK 20397 k) ∧
    k = 18 :=
  sorry

end base_k_addition_l654_65401


namespace geometric_sequence_product_bound_l654_65461

theorem geometric_sequence_product_bound (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_seq : (4 * a^2 + b^2)^2 = a * b) : a * b ≥ 2 := by
  sorry

end geometric_sequence_product_bound_l654_65461


namespace hannah_age_proof_l654_65487

def siblings_ages : List ℕ := [103, 124, 146, 81, 114, 195, 183]

def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

theorem hannah_age_proof :
  let avg_sibling_age := average_age siblings_ages
  let hannah_age := 3.2 * avg_sibling_age
  hannah_age = 432 := by sorry

end hannah_age_proof_l654_65487


namespace equation_solutions_l654_65441

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 + 2*x*(x - 3)
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = 1 := by
sorry

end equation_solutions_l654_65441


namespace parentheses_removal_equality_l654_65456

theorem parentheses_removal_equality (x : ℝ) : -(x - 2) - 2 * (x^2 + 2) = -x + 2 - 2*x^2 - 4 := by
  sorry

end parentheses_removal_equality_l654_65456


namespace wool_production_l654_65448

variables (x y z w v : ℝ)
variable (breed_A_production : ℝ → ℝ → ℝ → ℝ)
variable (breed_B_production : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ)

-- Breed A production rate
axiom breed_A_rate : breed_A_production x y z = y / (x * z)

-- Breed B produces twice as much as Breed A
axiom breed_B_rate : breed_B_production x y z w v = 2 * breed_A_production x y z * w * v

-- Theorem to prove
theorem wool_production : breed_B_production x y z w v = (2 * y * w * v) / (x * z) := by
  sorry

end wool_production_l654_65448


namespace smallest_m_value_l654_65414

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_m_value :
  ∃ (m x y : ℕ),
    m = x * y * (10 * x + y) ∧
    100 ≤ m ∧ m < 1000 ∧
    x < 10 ∧ y < 10 ∧
    x ≠ y ∧
    is_prime (10 * x + y) ∧
    is_prime (x + y) ∧
    (∀ (m' x' y' : ℕ),
      m' = x' * y' * (10 * x' + y') →
      100 ≤ m' ∧ m' < 1000 →
      x' < 10 ∧ y' < 10 →
      x' ≠ y' →
      is_prime (10 * x' + y') →
      is_prime (x' + y') →
      m ≤ m') ∧
    m = 138 :=
by sorry

end smallest_m_value_l654_65414


namespace magnitude_relationship_l654_65444

noncomputable def a : ℝ := Real.sqrt 5 + 2
noncomputable def b : ℝ := 2 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 5 - 2

theorem magnitude_relationship : a > c ∧ c > b :=
by sorry

end magnitude_relationship_l654_65444


namespace even_function_m_value_l654_65439

-- Define a function f
def f (m : ℝ) (x : ℝ) : ℝ := (x - 2) * (x - m)

-- State the theorem
theorem even_function_m_value :
  (∀ x : ℝ, f m x = f m (-x)) → m = -2 := by
  sorry

end even_function_m_value_l654_65439


namespace pascal_triangle_cube_sum_l654_65424

/-- Pascal's Triangle interior numbers sum for row n -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- Sum of cubes of interior numbers in the fifth row -/
def fifth_row_cube_sum : ℕ := 468

/-- Sum of cubes of interior numbers in the sixth row -/
def sixth_row_cube_sum : ℕ := 14750

/-- Theorem: If the sum of cubes of interior numbers in the fifth row is 468,
    then the sum of cubes of interior numbers in the sixth row is 14750 -/
theorem pascal_triangle_cube_sum :
  fifth_row_cube_sum = 468 → sixth_row_cube_sum = 14750 := by
  sorry

end pascal_triangle_cube_sum_l654_65424


namespace special_gp_ratio_equation_ratio_approx_value_l654_65472

/-- A geometric progression with positive terms where any term is equal to the sum of the next three following terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a special geometric progression satisfies a specific equation -/
theorem special_gp_ratio_equation (gp : SpecialGeometricProgression) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 :=
sorry

/-- The positive real solution to the equation x^3 + x^2 + x - 1 = 0 is approximately 0.543689 -/
theorem ratio_approx_value :
  ∃ x : ℝ, x > 0 ∧ x^3 + x^2 + x - 1 = 0 ∧ abs (x - 0.543689) < 0.000001 :=
sorry

end special_gp_ratio_equation_ratio_approx_value_l654_65472


namespace snackles_leftover_candies_l654_65442

theorem snackles_leftover_candies (m : ℕ) (h : m % 9 = 8) : (2 * m) % 9 = 7 := by
  sorry

end snackles_leftover_candies_l654_65442


namespace digit_1983_is_7_l654_65494

/-- Represents the decimal number x as described in the problem -/
def x : ℝ :=
  sorry

/-- Returns the nth digit after the decimal point in x -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the 1983rd digit of x is 7 -/
theorem digit_1983_is_7 : nthDigit 1983 = 7 := by
  sorry

end digit_1983_is_7_l654_65494


namespace parabola_symmetry_l654_65468

-- Define the parabolas
def C₁ (x : ℝ) : ℝ := x^2 - 2*x + 3
def C₂ (x : ℝ) : ℝ := C₁ (x + 1)
def C₃ (x : ℝ) : ℝ := C₂ (-x)

-- State the theorem
theorem parabola_symmetry :
  ∀ x : ℝ, C₃ x = x^2 + 2 :=
by sorry

end parabola_symmetry_l654_65468


namespace fifth_term_of_geometric_sequence_l654_65433

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 = 16 →
  a 7 = 2 →
  a 5 = 8 := by
sorry

end fifth_term_of_geometric_sequence_l654_65433


namespace ceiling_of_e_l654_65429

theorem ceiling_of_e : ⌈Real.exp 1⌉ = 3 := by sorry

end ceiling_of_e_l654_65429


namespace expression_simplification_l654_65447

theorem expression_simplification 
  (a b c k x : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_k_nonzero : k ≠ 0) :
  k * ((x + a)^2 / ((a - b)*(a - c)) + 
       (x + b)^2 / ((b - a)*(b - c)) + 
       (x + c)^2 / ((c - a)*(c - b))) = k :=
by sorry

end expression_simplification_l654_65447


namespace unique_numbers_satisfying_condition_l654_65417

theorem unique_numbers_satisfying_condition : ∃! (a b : ℕ), 
  100 ≤ a ∧ a < 1000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  10000 * a + b = 7 * a * b ∧ 
  a + b = 1458 := by sorry

end unique_numbers_satisfying_condition_l654_65417


namespace minutes_to_seconds_l654_65423

theorem minutes_to_seconds (minutes : ℝ) : minutes * 60 = 750 → minutes = 12.5 := by
  sorry

end minutes_to_seconds_l654_65423


namespace roots_opposite_signs_n_value_l654_65425

/-- 
Given an equation of the form (x^2 - (a+1)x) / ((b+1)x - d) = (n-2) / (n+2),
if the roots of this equation are numerically equal but of opposite signs,
then n = 2(b-a) / (a+b+2).
-/
theorem roots_opposite_signs_n_value 
  (a b d n : ℝ) 
  (eq : ∀ x, (x^2 - (a+1)*x) / ((b+1)*x - d) = (n-2) / (n+2)) 
  (roots_opposite : ∃ r : ℝ, (r^2 - (a+1)*r) / ((b+1)*r - d) = (n-2) / (n+2) ∧ 
                              ((-r)^2 - (a+1)*(-r)) / ((b+1)*(-r) - d) = (n-2) / (n+2)) :
  n = 2*(b-a) / (a+b+2) := by
sorry

end roots_opposite_signs_n_value_l654_65425


namespace sufficient_not_necessary_condition_l654_65499

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, a ≥ 1 ∧ b ≥ 1 → a + b ≥ 2) ∧ 
  (∃ a b : ℝ, a + b ≥ 2 ∧ ¬(a ≥ 1 ∧ b ≥ 1)) := by
  sorry

end sufficient_not_necessary_condition_l654_65499


namespace tshirt_price_l654_65457

/-- The original price of a t-shirt satisfies the given conditions -/
theorem tshirt_price (discount : ℚ) (quantity : ℕ) (revenue : ℚ) 
  (h1 : discount = 8)
  (h2 : quantity = 130)
  (h3 : revenue = 5590) :
  ∃ (original_price : ℚ), 
    quantity * (original_price - discount) = revenue ∧ 
    original_price = 51 := by
  sorry

end tshirt_price_l654_65457


namespace trig_simplification_l654_65434

/-- Proves that 1/cos(80°) - √3/sin(80°) = 4 --/
theorem trig_simplification : 
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end trig_simplification_l654_65434


namespace grant_baseball_gear_sale_total_l654_65400

theorem grant_baseball_gear_sale_total (cards_price bat_price glove_original_price glove_discount cleats_price cleats_count : ℝ) :
  cards_price = 25 →
  bat_price = 10 →
  glove_original_price = 30 →
  glove_discount = 0.2 →
  cleats_price = 10 →
  cleats_count = 2 →
  cards_price + bat_price + (glove_original_price * (1 - glove_discount)) + (cleats_price * cleats_count) = 79 := by
  sorry

end grant_baseball_gear_sale_total_l654_65400


namespace intersection_of_A_and_B_l654_65480

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l654_65480


namespace class_size_l654_65455

/-- Given a class of children where:
  * 19 play tennis
  * 21 play squash
  * 10 play neither sport
  * 12 play both sports
  This theorem proves that there are 38 children in the class. -/
theorem class_size (T S N B : ℕ) (h1 : T = 19) (h2 : S = 21) (h3 : N = 10) (h4 : B = 12) :
  T + S - B + N = 38 := by
  sorry

end class_size_l654_65455


namespace billy_watches_95_videos_billy_within_time_constraint_l654_65490

/-- The number of videos Billy watches in total -/
def total_videos (suggestions_per_trial : ℕ) (num_trials : ℕ) (suggestions_per_category : ℕ) (num_categories : ℕ) : ℕ :=
  suggestions_per_trial * num_trials + suggestions_per_category * num_categories

/-- Theorem stating that Billy watches 95 videos in total -/
theorem billy_watches_95_videos :
  total_videos 15 5 10 2 = 95 := by
  sorry

/-- Billy's time constraint in minutes -/
def time_constraint : ℕ := 60

/-- Time taken to watch each video in minutes -/
def time_per_video : ℕ := 4

/-- Theorem stating that Billy's total watching time does not exceed the time constraint -/
theorem billy_within_time_constraint :
  total_videos 15 5 10 2 * time_per_video ≤ time_constraint := by
  sorry

end billy_watches_95_videos_billy_within_time_constraint_l654_65490


namespace decimal_77_to_octal_l654_65426

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem decimal_77_to_octal :
  decimal_to_octal 77 = [5, 1, 1] :=
sorry

end decimal_77_to_octal_l654_65426


namespace stratified_sampling_proportion_l654_65428

theorem stratified_sampling_proportion (total_male : ℕ) (total_female : ℕ) (selected_male : ℕ) :
  total_male = 56 →
  total_female = 42 →
  selected_male = 8 →
  ∃ (selected_female : ℕ),
    selected_female = 6 ∧
    (selected_male : ℚ) / total_male = (selected_female : ℚ) / total_female :=
by
  sorry

end stratified_sampling_proportion_l654_65428
