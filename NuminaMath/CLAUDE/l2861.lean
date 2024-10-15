import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l2861_286112

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)
  let g (x : ℝ) := (x - 2) * (x - 4) * (x - 2) * (x - 5)
  ∀ x : ℝ, (g x ≠ 0 ∧ f x / g x = 1) ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2861_286112


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2861_286121

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 8)*(x - 6) = -54 + k*x) ↔ 
  (k = 6*Real.sqrt 2 - 10 ∨ k = -6*Real.sqrt 2 - 10) := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2861_286121


namespace NUMINAMATH_CALUDE_xiaoxi_has_largest_result_l2861_286131

def start_number : ℕ := 8

def laura_result (n : ℕ) : ℕ := ((n - 2) * 3) + 3

def navin_result (n : ℕ) : ℕ := (n * 3 - 2) + 3

def xiaoxi_result (n : ℕ) : ℕ := ((n - 2) + 3) * 3

theorem xiaoxi_has_largest_result :
  xiaoxi_result start_number > laura_result start_number ∧
  xiaoxi_result start_number > navin_result start_number :=
by sorry

end NUMINAMATH_CALUDE_xiaoxi_has_largest_result_l2861_286131


namespace NUMINAMATH_CALUDE_inequality_proof_l2861_286129

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1/2) :
  (Real.sqrt x) / (4 * x + 1) + (Real.sqrt y) / (4 * y + 1) + (Real.sqrt z) / (4 * z + 1) ≤ 3 * Real.sqrt 6 / 10 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2861_286129


namespace NUMINAMATH_CALUDE_team_a_more_uniform_heights_l2861_286146

/-- Represents a team with its height statistics -/
structure Team where
  averageHeight : ℝ
  variance : ℝ

/-- Defines when a team has more uniform heights than another -/
def hasMoreUniformHeights (t1 t2 : Team) : Prop :=
  t1.variance < t2.variance

/-- Theorem stating that Team A has more uniform heights than Team B -/
theorem team_a_more_uniform_heights :
  let teamA : Team := { averageHeight := 1.82, variance := 0.56 }
  let teamB : Team := { averageHeight := 1.82, variance := 2.1 }
  hasMoreUniformHeights teamA teamB := by
  sorry

end NUMINAMATH_CALUDE_team_a_more_uniform_heights_l2861_286146


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l2861_286104

theorem fruit_basket_problem (total_fruit : ℕ) 
  (jacques_apples jacques_pears gillian_apples gillian_pears : ℕ) : 
  total_fruit = 25 →
  jacques_apples = 1 →
  jacques_pears = 3 →
  gillian_apples = 3 →
  gillian_pears = 2 →
  ∃ (initial_apples initial_pears : ℕ),
    initial_apples + initial_pears = total_fruit ∧
    initial_apples - jacques_apples - gillian_apples = 
      initial_pears - jacques_pears - gillian_pears →
  initial_pears = 13 := by
sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l2861_286104


namespace NUMINAMATH_CALUDE_class_representatives_count_l2861_286156

/-- The number of ways to select and arrange class representatives -/
def class_representatives (num_male num_female : ℕ) (male_to_select female_to_select : ℕ) : ℕ :=
  Nat.choose num_male male_to_select *
  Nat.choose num_female female_to_select *
  Nat.factorial (male_to_select + female_to_select)

/-- Theorem stating that the number of ways to select and arrange class representatives
    from 3 male and 3 female students, selecting 1 male and 2 females, is 54 -/
theorem class_representatives_count :
  class_representatives 3 3 1 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_representatives_count_l2861_286156


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2861_286159

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 2*x*y + 3*y^2 - z^2 = 17) ∧ 
    (-x^2 + 4*y*z + z^2 = 28) ∧ 
    (x^2 + 2*x*y + 5*z^2 = 42) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2861_286159


namespace NUMINAMATH_CALUDE_milk_tea_price_proof_l2861_286124

/-- The cost of a cup of milk tea in dollars -/
def milk_tea_cost : ℝ := 2.4

/-- The cost of a slice of cake in dollars -/
def cake_slice_cost : ℝ := 0.75 * milk_tea_cost

theorem milk_tea_price_proof :
  (cake_slice_cost = 0.75 * milk_tea_cost) ∧
  (2 * cake_slice_cost + milk_tea_cost = 6) →
  milk_tea_cost = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_milk_tea_price_proof_l2861_286124


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l2861_286120

/-- Sum of an arithmetic series -/
def arithmetic_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- The sum of the arithmetic series with first term 22, last term 73, and common difference 3/7 is 5700 -/
theorem arithmetic_series_sum :
  arithmetic_sum 22 73 (3/7) = 5700 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l2861_286120


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2861_286179

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 36 → b = 48 → c^2 = a^2 + b^2 → c = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2861_286179


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2861_286165

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (∀ θ : ℝ, θ = 160 ∧ θ = (n - 2 : ℝ) * 180 / n) → n = 18 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2861_286165


namespace NUMINAMATH_CALUDE_car_rental_maximum_profit_l2861_286130

/-- Represents the car rental company's profit function --/
def profit_function (n : ℝ) : ℝ :=
  -50 * (n^2 - 100*n + 630000)

/-- Represents the rental fee calculation --/
def rental_fee (n : ℝ) : ℝ :=
  3000 + 50 * n

theorem car_rental_maximum_profit :
  ∃ (n : ℝ),
    profit_function n = 307050 ∧
    rental_fee n = 4050 ∧
    ∀ (m : ℝ), profit_function m ≤ profit_function n :=
by
  sorry

end NUMINAMATH_CALUDE_car_rental_maximum_profit_l2861_286130


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l2861_286123

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2*a| + |x - 3*a|

-- Theorem 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 2 ∨ a = -2 :=
sorry

-- Theorem 2
theorem inequality_implies_m_range (m : ℝ) :
  (∀ x, ∃ a ∈ Set.Icc (-2) 2, m^2 - |m| - f x a < 0) →
  -1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l2861_286123


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l2861_286108

theorem kevin_kangaroo_hops (n : ℕ) (a : ℚ) (r : ℚ) : 
  n = 7 ∧ a = 1 ∧ r = 3/4 → 
  4 * (a * (1 - r^n) / (1 - r)) = 7086/2048 := by
sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l2861_286108


namespace NUMINAMATH_CALUDE_sine_equality_equivalence_l2861_286153

theorem sine_equality_equivalence (α β : ℝ) : 
  (∃ k : ℤ, α = k * Real.pi + (-1)^k * β) ↔ Real.sin α = Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sine_equality_equivalence_l2861_286153


namespace NUMINAMATH_CALUDE_carina_coffee_amount_l2861_286145

/-- The number of 10-ounce packages of coffee -/
def num_10oz_packages : ℕ := 4

/-- The number of 5-ounce packages of coffee -/
def num_5oz_packages : ℕ := num_10oz_packages + 2

/-- The total amount of coffee in ounces -/
def total_coffee : ℕ := num_10oz_packages * 10 + num_5oz_packages * 5

theorem carina_coffee_amount : total_coffee = 70 := by
  sorry

end NUMINAMATH_CALUDE_carina_coffee_amount_l2861_286145


namespace NUMINAMATH_CALUDE_tangent_two_implications_l2861_286117

open Real

theorem tangent_two_implications (α : ℝ) (h : tan α = 2) :
  (sin α + 2 * cos α) / (4 * cos α - sin α) = 2 ∧
  Real.sqrt 2 * sin (2 * α + π / 4) + 1 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_two_implications_l2861_286117


namespace NUMINAMATH_CALUDE_centroid_incenter_ratio_l2861_286197

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumcenter, incenter, and centroid
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def incenter (t : Triangle) : ℝ × ℝ := sorry
def centroid_of_arc_midpoints (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem centroid_incenter_ratio (t : Triangle) :
  let O := circumcenter t
  let I := incenter t
  let G := centroid_of_arc_midpoints t
  distance A B = 13 →
  distance B C = 14 →
  distance C A = 15 →
  (distance G O) / (distance G I) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_centroid_incenter_ratio_l2861_286197


namespace NUMINAMATH_CALUDE_car_average_speed_l2861_286163

/-- Given a car's speed for two consecutive hours, calculate its average speed. -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 100) (h2 : speed2 = 30) :
  (speed1 + speed2) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l2861_286163


namespace NUMINAMATH_CALUDE_computers_produced_per_month_l2861_286107

/-- The number of days in a month -/
def days_per_month : ℕ := 28

/-- The number of computers produced in 30 minutes -/
def computers_per_interval : ℕ := 3

/-- The number of 30-minute intervals in a day -/
def intervals_per_day : ℕ := 24 * 2

/-- Calculates the number of computers produced in a month -/
def computers_per_month : ℕ :=
  days_per_month * intervals_per_day * computers_per_interval

/-- Theorem stating that the number of computers produced per month is 4032 -/
theorem computers_produced_per_month :
  computers_per_month = 4032 := by
  sorry


end NUMINAMATH_CALUDE_computers_produced_per_month_l2861_286107


namespace NUMINAMATH_CALUDE_restaurant_order_combinations_l2861_286188

def menu_size : ℕ := 15
def num_people : ℕ := 3

theorem restaurant_order_combinations :
  menu_size ^ num_people = 3375 := by sorry

end NUMINAMATH_CALUDE_restaurant_order_combinations_l2861_286188


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2861_286166

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2861_286166


namespace NUMINAMATH_CALUDE_village_population_equality_second_village_initial_population_l2861_286162

/-- The initial population of Village X -/
def initial_pop_X : ℕ := 68000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The yearly increase in population of the second village -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 13

/-- The initial population of the second village -/
def initial_pop_Y : ℕ := 42000

theorem village_population_equality :
  initial_pop_X - years * decrease_rate_X = initial_pop_Y + years * increase_rate_Y :=
by sorry

theorem second_village_initial_population :
  initial_pop_Y = 42000 :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_second_village_initial_population_l2861_286162


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l2861_286177

theorem integer_ratio_problem (s l : ℕ) : 
  s = 32 →
  ∃ k : ℕ, l = k * s →
  s + l = 96 →
  l / s = 2 :=
by sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l2861_286177


namespace NUMINAMATH_CALUDE_correct_propositions_l2861_286122

theorem correct_propositions : 
  -- Proposition 2
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ (a * b) / (a + b)) ∧
  -- Proposition 3
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  -- Proposition 4
  (Real.log 9 * Real.log 11 < 1) ∧
  -- Proposition 5
  (∀ a b : ℝ, a > b ∧ 1/a > 1/b → a > 0 ∧ b < 0) ∧
  -- Proposition 1 (incorrect)
  ¬(∀ a b : ℝ, a < b ∧ b < 0 → 1/a < 1/b) ∧
  -- Proposition 6 (incorrect)
  ¬(∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → (x + 2*y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1 ∧ x₀ + 2*y₀ = 6)) :=
by sorry


end NUMINAMATH_CALUDE_correct_propositions_l2861_286122


namespace NUMINAMATH_CALUDE_diorama_building_time_l2861_286151

/-- Given the total time spent on a diorama and the relationship between building and planning time,
    prove the time spent building the diorama. -/
theorem diorama_building_time (total_time planning_time building_time : ℕ) 
    (h1 : total_time = 67)
    (h2 : building_time = 3 * planning_time - 5)
    (h3 : total_time = planning_time + building_time) : 
    building_time = 49 := by
  sorry

end NUMINAMATH_CALUDE_diorama_building_time_l2861_286151


namespace NUMINAMATH_CALUDE_product_ABCD_eq_one_l2861_286133

/-- Given A, B, C, and D as defined, prove that their product is 1 -/
theorem product_ABCD_eq_one (A B C D : ℝ) 
  (hA : A = Real.sqrt 2008 + Real.sqrt 2009)
  (hB : B = -(Real.sqrt 2008) - Real.sqrt 2009)
  (hC : C = Real.sqrt 2008 - Real.sqrt 2009)
  (hD : D = Real.sqrt 2009 - Real.sqrt 2008) : 
  A * B * C * D = 1 := by
  sorry


end NUMINAMATH_CALUDE_product_ABCD_eq_one_l2861_286133


namespace NUMINAMATH_CALUDE_typhoon_tree_problem_l2861_286170

theorem typhoon_tree_problem :
  ∀ (total survived died : ℕ),
    total = 14 →
    died = survived + 4 →
    survived + died = total →
    died = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_typhoon_tree_problem_l2861_286170


namespace NUMINAMATH_CALUDE_target_hit_probability_l2861_286152

theorem target_hit_probability 
  (prob_A prob_B prob_C : ℚ)
  (h_A : prob_A = 1/2)
  (h_B : prob_B = 1/3)
  (h_C : prob_C = 1/4) :
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2861_286152


namespace NUMINAMATH_CALUDE_sally_forgot_seven_poems_l2861_286143

/-- Represents the number of poems in different categories --/
structure PoemCounts where
  initial : ℕ
  correct : ℕ
  mixed : ℕ

/-- Calculates the number of completely forgotten poems --/
def forgotten_poems (counts : PoemCounts) : ℕ :=
  counts.initial - (counts.correct + counts.mixed)

/-- Theorem stating that Sally forgot 7 poems --/
theorem sally_forgot_seven_poems : 
  let sally_counts : PoemCounts := ⟨15, 5, 3⟩
  forgotten_poems sally_counts = 7 := by
  sorry

end NUMINAMATH_CALUDE_sally_forgot_seven_poems_l2861_286143


namespace NUMINAMATH_CALUDE_power_sum_equals_eighteen_l2861_286113

theorem power_sum_equals_eighteen :
  (-3)^4 + (-3)^2 + (-3)^1 + 3^1 - 3^4 + 3^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_eighteen_l2861_286113


namespace NUMINAMATH_CALUDE_slope_values_theorem_l2861_286139

def valid_slopes : List ℕ := [81, 192, 399, 501, 1008, 2019]

def intersects_parabola (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁^2 = k * x₁ + 2020 ∧ x₂^2 = k * x₂ + 2020

theorem slope_values_theorem :
  ∀ k : ℝ, k > 0 → intersects_parabola k → k ∈ valid_slopes.map (λ x => x : ℕ → ℝ) := by
  sorry

end NUMINAMATH_CALUDE_slope_values_theorem_l2861_286139


namespace NUMINAMATH_CALUDE_smallest_k_multiple_of_144_l2861_286118

def sum_of_squares (k : ℕ+) : ℕ := k.val * (k.val + 1) * (2 * k.val + 1) / 6

theorem smallest_k_multiple_of_144 :
  ∀ k : ℕ+, k.val < 26 → ¬(144 ∣ sum_of_squares k) ∧
  144 ∣ sum_of_squares 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_multiple_of_144_l2861_286118


namespace NUMINAMATH_CALUDE_occupation_combinations_eq_636_l2861_286191

/-- The number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- The number of Mars-like planets -/
def mars_like_planets : ℕ := 9

/-- The number of colonization units required for an Earth-like planet -/
def earth_units : ℕ := 2

/-- The number of colonization units required for a Mars-like planet -/
def mars_units : ℕ := 1

/-- The total number of available colonization units -/
def total_units : ℕ := 14

/-- Function to calculate the number of combinations of occupying planets -/
def occupation_combinations : ℕ := sorry

theorem occupation_combinations_eq_636 : occupation_combinations = 636 := by sorry

end NUMINAMATH_CALUDE_occupation_combinations_eq_636_l2861_286191


namespace NUMINAMATH_CALUDE_hybrid_car_trip_length_l2861_286183

theorem hybrid_car_trip_length : 
  ∀ (trip_length : ℝ),
  (trip_length / (0.02 * (trip_length - 40)) = 55) →
  trip_length = 440 :=
by sorry

end NUMINAMATH_CALUDE_hybrid_car_trip_length_l2861_286183


namespace NUMINAMATH_CALUDE_geometric_progression_sum_180_l2861_286103

theorem geometric_progression_sum_180 :
  ∃ (a b c d : ℝ) (e f g h : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧
    a + b + c + d = 180 ∧
    b / a = c / b ∧ c / b = d / c ∧
    c = a + 36 ∧
    e + f + g + h = 180 ∧
    f / e = g / f ∧ g / f = h / g ∧
    g = e + 36 ∧
    ((a = 9/2 ∧ b = 27/2 ∧ c = 81/2 ∧ d = 243/2) ∨
     (a = 12 ∧ b = 24 ∧ c = 48 ∧ d = 96)) ∧
    ((e = 9/2 ∧ f = 27/2 ∧ g = 81/2 ∧ h = 243/2) ∨
     (e = 12 ∧ f = 24 ∧ g = 48 ∧ h = 96)) ∧
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_180_l2861_286103


namespace NUMINAMATH_CALUDE_only_blue_possible_all_blue_possible_l2861_286132

/-- Represents the number of sheep of each color -/
structure SheepCounts where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- Represents a valid transformation of sheep colors -/
inductive SheepTransform : SheepCounts → SheepCounts → Prop where
  | blue_red_to_green : ∀ b r g, SheepTransform ⟨b+1, r+1, g-2⟩ ⟨b, r, g⟩
  | blue_green_to_red : ∀ b r g, SheepTransform ⟨b+1, r-2, g+1⟩ ⟨b, r, g⟩
  | red_green_to_blue : ∀ b r g, SheepTransform ⟨b-2, r+1, g+1⟩ ⟨b, r, g⟩

/-- Represents a sequence of transformations -/
def TransformSequence : SheepCounts → SheepCounts → Prop :=
  Relation.ReflTransGen SheepTransform

/-- The theorem stating that only blue is possible as the final uniform color -/
theorem only_blue_possible (initial : SheepCounts) (final : SheepCounts) :
  initial = ⟨22, 18, 15⟩ →
  TransformSequence initial final →
  (final.blue = 0 ∧ final.red = 55) ∨ (final.blue = 0 ∧ final.green = 55) →
  False :=
sorry

/-- The theorem stating that all blue is possible -/
theorem all_blue_possible (initial : SheepCounts) (final : SheepCounts) :
  initial = ⟨22, 18, 15⟩ →
  ∃ final, TransformSequence initial final ∧ final = ⟨55, 0, 0⟩ :=
sorry

end NUMINAMATH_CALUDE_only_blue_possible_all_blue_possible_l2861_286132


namespace NUMINAMATH_CALUDE_g_difference_l2861_286190

/-- The function g(x) = 2x^3 + 5x^2 - 2x - 1 -/
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

/-- Theorem stating that g(x+h) - g(x) = h(6x^2 + 6xh + 2h^2 + 10x + 5h - 2) for all x and h -/
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l2861_286190


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2861_286176

theorem complex_number_in_second_quadrant :
  let z : ℂ := (-2 + 3 * Complex.I) / (3 - 4 * Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2861_286176


namespace NUMINAMATH_CALUDE_expression_evaluation_l2861_286111

theorem expression_evaluation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2861_286111


namespace NUMINAMATH_CALUDE_decagon_perimeter_l2861_286195

/-- The perimeter of a regular polygon with n sides of length s -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- The side length of our specific decagon -/
def side_length : ℝ := 3

theorem decagon_perimeter : perimeter decagon_sides side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_l2861_286195


namespace NUMINAMATH_CALUDE_boat_drift_l2861_286184

/-- Calculate the drift of a boat crossing a river -/
theorem boat_drift (river_width : ℝ) (boat_speed : ℝ) (crossing_time : ℝ) :
  river_width = 400 ∧ boat_speed = 10 ∧ crossing_time = 50 →
  boat_speed * crossing_time - river_width = 100 := by
  sorry

end NUMINAMATH_CALUDE_boat_drift_l2861_286184


namespace NUMINAMATH_CALUDE_tangent_points_concyclic_l2861_286169

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define a predicate for a point being outside a circle
variable (is_outside : Point → Circle → Prop)

-- Define a predicate for two circles being concentric
variable (concentric : Circle → Circle → Prop)

-- Define a predicate for a line being tangent to a circle at a point
variable (is_tangent : Point → Point → Circle → Prop)

-- Define a predicate for points being concyclic
variable (concyclic : List Point → Prop)

-- State the theorem
theorem tangent_points_concyclic 
  (O : Point) (c1 c2 : Circle) (M A B C D : Point) :
  center c1 = O →
  center c2 = O →
  concentric c1 c2 →
  is_outside M c1 →
  is_outside M c2 →
  is_tangent M A c1 →
  is_tangent M B c1 →
  is_tangent M C c2 →
  is_tangent M D c2 →
  concyclic [M, A, B, C, D] :=
sorry

end NUMINAMATH_CALUDE_tangent_points_concyclic_l2861_286169


namespace NUMINAMATH_CALUDE_problem_solution_l2861_286127

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem problem_solution (n : ℕ) (h : n * factorial n + 2 * factorial n = 5040) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2861_286127


namespace NUMINAMATH_CALUDE_project_payment_l2861_286150

/-- Represents the hourly wage of candidate q -/
def q_wage : ℝ := 14

/-- Represents the hourly wage of candidate p -/
def p_wage : ℝ := 21

/-- Represents the number of hours candidate p needs to complete the job -/
def p_hours : ℝ := 20

/-- Represents the number of hours candidate q needs to complete the job -/
def q_hours : ℝ := p_hours + 10

/-- The total payment for the project -/
def total_payment : ℝ := 420

theorem project_payment :
  (p_wage = q_wage * 1.5) ∧
  (p_wage = q_wage + 7) ∧
  (q_hours = p_hours + 10) ∧
  (p_wage * p_hours = q_wage * q_hours) →
  total_payment = 420 := by sorry

end NUMINAMATH_CALUDE_project_payment_l2861_286150


namespace NUMINAMATH_CALUDE_min_a_value_l2861_286168

theorem min_a_value (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + Real.sqrt (x * y) ≤ a * (x + y)) → 
  a ≥ (Real.sqrt 2 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l2861_286168


namespace NUMINAMATH_CALUDE_last_digit_98_base5_l2861_286173

def last_digit_base5 (n : ℕ) : ℕ :=
  n % 5

theorem last_digit_98_base5 :
  last_digit_base5 98 = 3 := by
sorry

end NUMINAMATH_CALUDE_last_digit_98_base5_l2861_286173


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l2861_286147

/-- Calculates the total grocery bill for Ray's purchase with a store rewards discount --/
theorem rays_grocery_bill :
  let meat_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let subtotal : ℚ := meat_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  let discount : ℚ := subtotal * discount_rate
  let total : ℚ := subtotal - discount

  total = 18 := by sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l2861_286147


namespace NUMINAMATH_CALUDE_log_simplification_l2861_286125

-- Define the variables as positive real numbers
variable (a b d e y z : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hd : 0 < d) (he : 0 < e) (hy : 0 < y) (hz : 0 < z)

-- State the theorem
theorem log_simplification :
  Real.log (a / b) + Real.log (b / e) + Real.log (e / d) - Real.log (a * z / (d * y)) = Real.log (d * y / z) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l2861_286125


namespace NUMINAMATH_CALUDE_circle_symmetry_minimum_l2861_286137

/-- Given a circle x^2 + y^2 + 2x - 4y + 1 = 0 symmetric with respect to the line 2ax - by + 2 = 0,
    where a > 0 and b > 0, the minimum value of 4/a + 1/b is 9. -/
theorem circle_symmetry_minimum (a b : ℝ) : a > 0 → b > 0 →
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 →
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧
      2*a*x - b*y + 2 = 2*a*x' - b*y' + 2) →
  (∀ t : ℝ, 4/a + 1/b ≥ t) →
  t = 9 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_minimum_l2861_286137


namespace NUMINAMATH_CALUDE_movie_children_count_prove_movie_children_count_l2861_286148

theorem movie_children_count : ℕ → Prop :=
  fun num_children =>
    let total_cost : ℕ := 76
    let num_adults : ℕ := 5
    let adult_ticket_cost : ℕ := 10
    let child_ticket_cost : ℕ := 7
    let concession_cost : ℕ := 12
    
    (num_adults * adult_ticket_cost + num_children * child_ticket_cost + concession_cost = total_cost) →
    num_children = 2

theorem prove_movie_children_count : ∃ (n : ℕ), movie_children_count n :=
  sorry

end NUMINAMATH_CALUDE_movie_children_count_prove_movie_children_count_l2861_286148


namespace NUMINAMATH_CALUDE_tenth_finger_is_two_l2861_286187

-- Define the functions f and g
def f : ℕ → ℕ
| 4 => 3
| 1 => 8
| 7 => 2
| _ => 0  -- Default case

def g : ℕ → ℕ
| 3 => 1
| 8 => 7
| 2 => 1
| _ => 0  -- Default case

-- Define a function that applies f and g alternately n times
def applyAlternately (n : ℕ) (start : ℕ) : ℕ :=
  match n with
  | 0 => start
  | n + 1 => if n % 2 = 0 then g (applyAlternately n start) else f (applyAlternately n start)

-- Theorem statement
theorem tenth_finger_is_two : applyAlternately 9 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_finger_is_two_l2861_286187


namespace NUMINAMATH_CALUDE_anna_coins_value_l2861_286116

/-- Represents the number and value of coins Anna has. -/
structure Coins where
  pennies : ℕ
  nickels : ℕ
  total : ℕ
  penny_nickel_relation : pennies = 2 * (nickels + 1) + 1
  total_coins : pennies + nickels = total

/-- The value of Anna's coins in cents -/
def coin_value (c : Coins) : ℕ := c.pennies + 5 * c.nickels

/-- Theorem stating that Anna's coins are worth 31 cents -/
theorem anna_coins_value :
  ∃ c : Coins, c.total = 15 ∧ coin_value c = 31 := by
  sorry


end NUMINAMATH_CALUDE_anna_coins_value_l2861_286116


namespace NUMINAMATH_CALUDE_constant_term_of_binomial_expansion_l2861_286141

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the constant term in the expansion of (x - 1/x)^8
def constantTerm : ℕ := binomial 8 4

-- Theorem statement
theorem constant_term_of_binomial_expansion :
  constantTerm = 70 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_binomial_expansion_l2861_286141


namespace NUMINAMATH_CALUDE_probability_is_correct_l2861_286178

/-- Represents a unit cube within the larger cube -/
structure UnitCube where
  painted_faces : Nat
  deriving Repr

/-- Represents the larger 5x5x5 cube -/
def LargeCube : Type := Array UnitCube

/-- Creates a large cube with the specified painting configuration -/
def create_large_cube : LargeCube :=
  sorry

/-- Calculates the number of ways to choose 2 items from n items -/
def choose_two (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the probability of selecting one cube with two painted faces
    and one cube with no painted faces -/
def probability_two_and_none (cube : LargeCube) : Rat :=
  sorry

theorem probability_is_correct (cube : LargeCube) :
  probability_two_and_none (create_large_cube) = 187 / 3875 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l2861_286178


namespace NUMINAMATH_CALUDE_books_for_girls_l2861_286140

theorem books_for_girls (num_girls num_boys total_books : ℕ) 
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375)
  (h_equal_division : ∃ (books_per_student : ℕ), 
    total_books = books_per_student * (num_girls + num_boys)) :
  ∃ (books_for_girls : ℕ), books_for_girls = 225 ∧ 
    books_for_girls = num_girls * (total_books / (num_girls + num_boys)) := by
  sorry

end NUMINAMATH_CALUDE_books_for_girls_l2861_286140


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_3_l2861_286105

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ year < 3000 ∧ sum_of_digits year = 3

theorem first_year_after_2010_with_digit_sum_3 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2100 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_3_l2861_286105


namespace NUMINAMATH_CALUDE_cubic_root_problem_l2861_286128

theorem cubic_root_problem (a b r s : ℤ) : 
  a ≠ 0 → b ≠ 0 → 
  (∀ x : ℤ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) →
  (r = s ∨ r = -2 ∨ s = -2) →
  (|a*b| = 272) :=
sorry

end NUMINAMATH_CALUDE_cubic_root_problem_l2861_286128


namespace NUMINAMATH_CALUDE_triangle_side_length_l2861_286102

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 3 → c = 4 → Real.cos C = -(1/4 : ℝ) →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  b = 7/2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2861_286102


namespace NUMINAMATH_CALUDE_tens_digit_of_smallest_divisible_l2861_286144

-- Define the smallest positive integer divisible by 20, 16, and 2016
def smallest_divisible : ℕ := 10080

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Theorem statement
theorem tens_digit_of_smallest_divisible :
  tens_digit smallest_divisible = 8 ∧
  ∀ m : ℕ, m > 0 ∧ 20 ∣ m ∧ 16 ∣ m ∧ 2016 ∣ m → m ≥ smallest_divisible :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_smallest_divisible_l2861_286144


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l2861_286110

def inequality (x : ℝ) : Prop :=
  (3 / (x + 2)) + (4 / (x + 6)) > 1

def solution_set (x : ℝ) : Prop :=
  x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l2861_286110


namespace NUMINAMATH_CALUDE_q_transformation_l2861_286199

theorem q_transformation (w d z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
  (h1 : ∀ w d z, q w d z = 5 * w / (4 * d * z^2))
  (h2 : ∃ k, q (k * w) (2 * d) (3 * z) = 0.2222222222222222 * q w d z) :
  ∃ k, k = 4 ∧ q (k * w) (2 * d) (3 * z) = 0.2222222222222222 * q w d z :=
sorry

end NUMINAMATH_CALUDE_q_transformation_l2861_286199


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2861_286138

theorem polygon_interior_angles_sum (n : ℕ) : 
  (n ≥ 3) → ((n - 2) * 180 = 900) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l2861_286138


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2861_286185

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 2 →
  B = π / 3 ∧
  (∃ (S : ℝ), S = Real.sqrt 3 ∧ ∀ (S' : ℝ), S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2861_286185


namespace NUMINAMATH_CALUDE_max_a_value_l2861_286196

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → Real.exp x + Real.sin x - 2*x ≥ a*x^2 + 1) → 
  a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2861_286196


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2861_286174

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I + z = 2) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2861_286174


namespace NUMINAMATH_CALUDE_chocolate_bar_squares_l2861_286100

theorem chocolate_bar_squares (gerald_bars : ℕ) (students : ℕ) (squares_per_student : ℕ) :
  gerald_bars = 7 →
  students = 24 →
  squares_per_student = 7 →
  (gerald_bars + 2 * gerald_bars) * (squares_in_each_bar : ℕ) = students * squares_per_student →
  squares_in_each_bar = 8 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bar_squares_l2861_286100


namespace NUMINAMATH_CALUDE_betty_age_l2861_286136

/-- Given the ages of Alice, Betty, and Carol satisfying certain conditions,
    prove that Betty's age is 7.5 years. -/
theorem betty_age (alice carol betty : ℝ) 
    (h1 : carol = 5 * alice)
    (h2 : carol = 2 * betty)
    (h3 : alice = carol - 12) : 
  betty = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l2861_286136


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l2861_286161

/-- Given two people moving in opposite directions, this theorem proves
    the speed of one person given the conditions of the problem. -/
theorem opposite_direction_speed
  (time : ℝ)
  (total_distance : ℝ)
  (speed_person1 : ℝ)
  (h1 : time = 4)
  (h2 : total_distance = 28)
  (h3 : speed_person1 = 3)
  : ∃ speed_person2 : ℝ,
    speed_person2 = 4 ∧ 
    total_distance = time * (speed_person1 + speed_person2) :=
by
  sorry

#check opposite_direction_speed

end NUMINAMATH_CALUDE_opposite_direction_speed_l2861_286161


namespace NUMINAMATH_CALUDE_x_value_l2861_286109

theorem x_value : ∃ x : ℝ, x ≠ 0 ∧ x = 3 * (1 / x * (-x)) + 3 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2861_286109


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2861_286157

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 16 * x + 12 = 0) : 
  ∃ x, b * x^2 + 16 * x + 12 = 0 ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2861_286157


namespace NUMINAMATH_CALUDE_sin_30_tan_45_calculation_l2861_286154

theorem sin_30_tan_45_calculation : 2 * Real.sin (30 * π / 180) - Real.tan (45 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_tan_45_calculation_l2861_286154


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2861_286189

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3}
def N : Set ℕ := {1, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2861_286189


namespace NUMINAMATH_CALUDE_sum_of_products_l2861_286134

theorem sum_of_products : 5 * 7 + 6 * 12 + 15 * 4 + 4 * 9 = 203 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l2861_286134


namespace NUMINAMATH_CALUDE_turtleneck_profit_l2861_286164

/-- Represents the pricing strategy and profit calculation for turtleneck sweaters -/
theorem turtleneck_profit (C : ℝ) : 
  let initial_price := C * 1.20
  let new_year_price := initial_price * 1.25
  let february_price := new_year_price * 0.94
  let profit := february_price - C
  profit = C * 0.41 := by sorry

end NUMINAMATH_CALUDE_turtleneck_profit_l2861_286164


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_proof_l2861_286142

/-- The smallest four-digit number divisible by 35 -/
def smallest_four_digit_divisible_by_35 : Nat := 1170

/-- A number is four digits if it's between 1000 and 9999 -/
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_35_proof :
  (is_four_digit smallest_four_digit_divisible_by_35) ∧ 
  (smallest_four_digit_divisible_by_35 % 35 = 0) ∧
  (∀ n : Nat, is_four_digit n → n % 35 = 0 → n ≥ smallest_four_digit_divisible_by_35) := by
  sorry

#eval smallest_four_digit_divisible_by_35

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_proof_l2861_286142


namespace NUMINAMATH_CALUDE_calculation_proof_l2861_286158

theorem calculation_proof : (8.036 / 0.04) * (1.5 / 0.03) = 10045 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2861_286158


namespace NUMINAMATH_CALUDE_intra_division_games_is_56_l2861_286101

/-- Represents a basketball league with specific conditions -/
structure BasketballLeague where
  N : ℕ  -- Number of times teams within the same division play each other
  M : ℕ  -- Number of times teams from different divisions play each other
  division_size : ℕ  -- Number of teams in each division
  total_games : ℕ  -- Total number of games each team plays in the season
  h1 : 3 * N = 5 * M + 8
  h2 : M > 6
  h3 : division_size = 5
  h4 : total_games = 82
  h5 : (division_size - 1) * N + division_size * M = total_games

/-- The number of games a team plays within its own division -/
def intra_division_games (league : BasketballLeague) : ℕ :=
  (league.division_size - 1) * league.N

/-- Theorem stating that each team plays 56 games within its own division -/
theorem intra_division_games_is_56 (league : BasketballLeague) :
  intra_division_games league = 56 := by
  sorry

end NUMINAMATH_CALUDE_intra_division_games_is_56_l2861_286101


namespace NUMINAMATH_CALUDE_largest_y_value_l2861_286135

theorem largest_y_value (y : ℝ) : 
  5 * (4 * y^2 + 12 * y + 15) = y * (4 * y - 25) →
  y ≤ (-85 + 5 * Real.sqrt 97) / 32 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l2861_286135


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2861_286172

/-- Given a man's speed with the current and the speed of the current,
    calculates the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem stating that given the specific conditions of the problem,
    the man's speed against the current is 18 kmph. -/
theorem mans_speed_against_current :
  speedAgainstCurrent 20 1 = 18 := by
  sorry

#eval speedAgainstCurrent 20 1

end NUMINAMATH_CALUDE_mans_speed_against_current_l2861_286172


namespace NUMINAMATH_CALUDE_raise_upper_bound_l2861_286186

/-- Represents a percentage as a real number between 0 and 1 -/
def Percentage := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The lower bound of the raise -/
def lower_bound : Percentage := ⟨0.05, by sorry⟩

/-- A possible raise value within the range -/
def possible_raise : Percentage := ⟨0.08, by sorry⟩

/-- The upper bound of the raise -/
def upper_bound : Percentage := ⟨0.09, by sorry⟩

theorem raise_upper_bound :
  lower_bound.val < possible_raise.val ∧
  possible_raise.val < upper_bound.val ∧
  ∀ (p : Percentage), lower_bound.val < p.val → p.val < upper_bound.val →
    p.val ≤ possible_raise.val ∨ possible_raise.val < p.val :=
by sorry

end NUMINAMATH_CALUDE_raise_upper_bound_l2861_286186


namespace NUMINAMATH_CALUDE_janice_purchase_problem_l2861_286114

theorem janice_purchase_problem :
  ∃ (a d b c : ℕ), 
    a + d + b + c = 50 ∧ 
    30 * a + 150 * d + 200 * b + 300 * c = 6000 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_problem_l2861_286114


namespace NUMINAMATH_CALUDE_equation_solutions_l2861_286194

theorem equation_solutions :
  (∃ x : ℝ, 3 * x + 7 = 32 - 2 * x ∧ x = 5) ∧
  (∃ x : ℝ, (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 ∧ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2861_286194


namespace NUMINAMATH_CALUDE_cindy_wins_prob_l2861_286160

-- Define the probability of tossing a five
def prob_five : ℚ := 1/6

-- Define the probability of not tossing a five
def prob_not_five : ℚ := 1 - prob_five

-- Define the probability of Cindy winning in the first cycle
def prob_cindy_first_cycle : ℚ := prob_not_five * prob_five

-- Define the probability of the game continuing after one full cycle
def prob_continue : ℚ := prob_not_five^3

-- Theorem statement
theorem cindy_wins_prob : 
  let a : ℚ := prob_cindy_first_cycle
  let r : ℚ := prob_continue
  (a / (1 - r)) = 30/91 := by
  sorry

end NUMINAMATH_CALUDE_cindy_wins_prob_l2861_286160


namespace NUMINAMATH_CALUDE_function_inequality_solution_l2861_286193

/-- Given a real number q with |q| < 1 and q ≠ 0, there exists a function f and a non-negative function g
    satisfying the given conditions. -/
theorem function_inequality_solution (q : ℝ) (hq1 : |q| < 1) (hq2 : q ≠ 0) :
  ∃ (f g : ℝ → ℝ) (a : ℕ → ℝ),
    (∀ x, g x ≥ 0) ∧
    (∀ x, f x = (1 - q * x) * f (q * x) + g x) ∧
    (∀ x, f x = ∑' i, a i * x^i) ∧
    (∀ k, k > 0 → a k = (a (k-1) * q^k - (1 / k.factorial) * (deriv^[k] g) 0) / (q^k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l2861_286193


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2861_286182

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (r s : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, r x * s x = k

theorem inverse_variation_problem (r s : ℝ → ℝ) 
  (h1 : VaryInversely r s)
  (h2 : r 1 = 1500)
  (h3 : s 1 = 0.4)
  (h4 : r 2 = 3000) :
  s 2 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2861_286182


namespace NUMINAMATH_CALUDE_right_angled_triangle_unique_k_l2861_286119

/-- A triangle with side lengths 13, 17, and k is right-angled if and only if k = 21 -/
theorem right_angled_triangle_unique_k : ∃! (k : ℕ), k > 0 ∧ 
  (13^2 + 17^2 = k^2 ∨ 13^2 + k^2 = 17^2 ∨ 17^2 + k^2 = 13^2) := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_unique_k_l2861_286119


namespace NUMINAMATH_CALUDE_office_staff_composition_l2861_286155

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := sorry

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Represents the average salary of all employees in rupees -/
def avg_salary_all : ℚ := 120

/-- Represents the average salary of officers in rupees -/
def avg_salary_officers : ℚ := 440

/-- Represents the average salary of non-officers in rupees -/
def avg_salary_non_officers : ℚ := 110

theorem office_staff_composition :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / (num_officers + num_non_officers) = avg_salary_all ∧
  num_non_officers = 480 := by sorry

end NUMINAMATH_CALUDE_office_staff_composition_l2861_286155


namespace NUMINAMATH_CALUDE_stock_price_uniqueness_l2861_286115

theorem stock_price_uniqueness : ¬∃ (k l : ℕ), (117/100)^k * (83/100)^l = 1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_uniqueness_l2861_286115


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2861_286167

theorem tan_105_degrees : Real.tan (105 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2861_286167


namespace NUMINAMATH_CALUDE_triangle_theorem_l2861_286198

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.a = 2 * Real.sqrt 3) 
  (h3 : t.b = 2) : 
  t.A = Real.pi / 3 ∧ Real.cos t.C = 0 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2861_286198


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2861_286180

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2861_286180


namespace NUMINAMATH_CALUDE_grid_broken_lines_theorem_l2861_286149

/-- Represents a grid of cells -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a broken line in the grid -/
structure BrokenLine :=
  (length : ℕ)

/-- Checks if it's possible to construct a set of broken lines in a grid -/
def canConstructBrokenLines (g : Grid) (lines : List BrokenLine) : Prop :=
  -- The actual implementation would be complex and is omitted
  sorry

theorem grid_broken_lines_theorem (g : Grid) :
  g.width = 11 ∧ g.height = 1 →
  (canConstructBrokenLines g (List.replicate 8 ⟨5⟩)) ∧
  ¬(canConstructBrokenLines g (List.replicate 5 ⟨8⟩)) :=
by sorry

end NUMINAMATH_CALUDE_grid_broken_lines_theorem_l2861_286149


namespace NUMINAMATH_CALUDE_solution_implies_m_range_l2861_286106

/-- A function representing the quadratic equation x^2 - mx + 2 = 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- The theorem stating that if the equation x^2 - mx + 2 = 0 has a solution 
    in the interval [1, 2], then m is in the range [2√2, 3] -/
theorem solution_implies_m_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f m x = 0) → 
  m ∈ Set.Icc (2 * Real.sqrt 2) 3 := by
  sorry


end NUMINAMATH_CALUDE_solution_implies_m_range_l2861_286106


namespace NUMINAMATH_CALUDE_rachel_plant_arrangement_l2861_286181

def num_arrangements (n : ℕ) : ℕ :=
  let all_under_one := 2  -- All plants under one white lamp or one red lamp
  let all_same_color := 2 * (n.choose 2)  -- All plants under lamps of the same color
  let diff_colors := (n.choose 2) + 2 * (n.choose 1)  -- Plants under lamps of different colors
  all_under_one + all_same_color + diff_colors

theorem rachel_plant_arrangement :
  num_arrangements 4 = 28 :=
by sorry

end NUMINAMATH_CALUDE_rachel_plant_arrangement_l2861_286181


namespace NUMINAMATH_CALUDE_dark_light_difference_l2861_286126

/-- Represents a square on the grid -/
inductive Square
| Dark
| Light

/-- Represents a row in the grid -/
def Row := Vector Square 9

/-- Represents the entire 9x9 grid -/
def Grid := Vector Row 9

/-- Creates an alternating row starting with the given square color -/
def alternatingRow (start : Square) : Row := sorry

/-- Creates the 9x9 grid with alternating pattern -/
def createGrid : Grid :=
  Vector.ofFn (λ i => alternatingRow (if i % 2 = 0 then Square.Dark else Square.Light))

/-- Counts the number of dark squares in the grid -/
def countDarkSquares (grid : Grid) : Nat := sorry

/-- Counts the number of light squares in the grid -/
def countLightSquares (grid : Grid) : Nat := sorry

/-- The main theorem stating the difference between dark and light squares -/
theorem dark_light_difference :
  let grid := createGrid
  countDarkSquares grid = countLightSquares grid + 1 := by sorry

end NUMINAMATH_CALUDE_dark_light_difference_l2861_286126


namespace NUMINAMATH_CALUDE_formula_always_zero_l2861_286192

theorem formula_always_zero :
  ∃ (F : (ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ) → ℝ → ℝ), 
    ∀ (sub mul : ℝ → ℝ → ℝ) (a : ℝ), 
      (∀ x y, sub x y = x - y ∨ sub x y = x * y) →
      (∀ x y, mul x y = x * y ∨ mul x y = x - y) →
      F sub mul a = 0 :=
by sorry

end NUMINAMATH_CALUDE_formula_always_zero_l2861_286192


namespace NUMINAMATH_CALUDE_log_product_equality_l2861_286171

theorem log_product_equality : (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) *
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equality_l2861_286171


namespace NUMINAMATH_CALUDE_lizette_quiz_average_l2861_286175

theorem lizette_quiz_average (q1 q2 : ℝ) : 
  (q1 + q2 + 92) / 3 = 94 → (q1 + q2) / 2 = 95 := by
  sorry

end NUMINAMATH_CALUDE_lizette_quiz_average_l2861_286175
