import Mathlib

namespace NUMINAMATH_CALUDE_correct_delivery_probability_l1756_175697

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- Probability of exactly k out of n packages being delivered to their correct houses -/
def probability_correct_delivery (n k : ℕ) : ℚ :=
  (n.choose k * (k.factorial : ℚ) * ((n - k).factorial : ℚ)) / (n.factorial : ℚ)

/-- Theorem stating the probability of exactly 3 out of 5 packages being delivered correctly -/
theorem correct_delivery_probability :
  probability_correct_delivery n k = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_correct_delivery_probability_l1756_175697


namespace NUMINAMATH_CALUDE_zero_in_interval_l1756_175608

-- Define the function f(x) = 2x - 5
def f (x : ℝ) : ℝ := 2 * x - 5

-- State the theorem
theorem zero_in_interval :
  (∀ x y, x < y → f x < f y) →  -- f is monotonically increasing
  Continuous f →                -- f is continuous
  ∃ c ∈ Set.Ioo 2 3, f c = 0    -- there exists a c in (2, 3) such that f(c) = 0
:= by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1756_175608


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1756_175603

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^2 = 18) -- third term is 18
  (h2 : a * r^4 = 72) -- fifth term is 72
  : a = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1756_175603


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l1756_175605

theorem chocolate_milk_probability (n : ℕ) (k : ℕ) (p : ℚ) :
  n = 6 →
  k = 5 →
  p = 2/3 →
  Nat.choose n k * p^k * (1 - p)^(n - k) = 64/243 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l1756_175605


namespace NUMINAMATH_CALUDE_fib_inequality_fib_upper_bound_l1756_175698

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Statement 1
theorem fib_inequality (n : ℕ) (h : n ≥ 2) : fib (n + 5) > 10 * fib n := by
  sorry

-- Statement 2
theorem fib_upper_bound (n k : ℕ) (h : fib (n + 1) < 10^k) : n ≤ 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fib_inequality_fib_upper_bound_l1756_175698


namespace NUMINAMATH_CALUDE_initial_solution_amount_l1756_175613

theorem initial_solution_amount 
  (x : ℝ) -- initial amount of solution in ml
  (h1 : x - 200 + 1000 = 2000) -- equation representing the process
  : x = 1200 :=
by sorry

end NUMINAMATH_CALUDE_initial_solution_amount_l1756_175613


namespace NUMINAMATH_CALUDE_equilateral_triangle_and_regular_pentagon_not_similar_l1756_175624

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a regular pentagon
structure RegularPentagon where
  side : ℝ
  side_positive : side > 0

-- Define similarity between shapes
def similar (shape1 shape2 : Type) : Prop := sorry

-- Theorem statement
theorem equilateral_triangle_and_regular_pentagon_not_similar :
  ∀ (t : EquilateralTriangle) (p : RegularPentagon), ¬(similar EquilateralTriangle RegularPentagon) :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_and_regular_pentagon_not_similar_l1756_175624


namespace NUMINAMATH_CALUDE_root_relationship_l1756_175674

theorem root_relationship (a b k x₁ x₂ x₃ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hk : k ≠ 0)
  (hx₁ : a * x₁^2 = k * x₁ + b)
  (hx₂ : a * x₂^2 = k * x₂ + b)
  (hx₃ : k * x₃ + b = 0) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
sorry

end NUMINAMATH_CALUDE_root_relationship_l1756_175674


namespace NUMINAMATH_CALUDE_count_ordered_pairs_3255_l1756_175628

theorem count_ordered_pairs_3255 : 
  let n : ℕ := 3255
  let prime_factorization : List ℕ := [5, 13, 17]
  ∀ (x y : ℕ), x * y = n → x > 0 ∧ y > 0 →
  (∃! (pairs : List (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ pairs ↔ p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) ∧
    pairs.length = 8) :=
by sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_3255_l1756_175628


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1756_175657

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := -Real.log x

-- Define the second derivative of g
def g'' (x : ℝ) : ℝ := 1 / x^2

-- Theorem statement
theorem tangent_line_slope :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
  (∀ x, f x - f x₁ = (x - x₁) * (2 * x₁)) ∧
  (∀ x, g'' x - g'' x₂ = (x - x₂) * (1 / x₂^2)) ∧
  2 * x₁ = 1 / x₂^2 →
  2 * x₁ = 4 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_slope_l1756_175657


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1756_175678

theorem cube_volume_surface_area (x : ℝ) : x > 0 → 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 7*x ∧ 6*s^2 = 2*x) → x = 1323 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1756_175678


namespace NUMINAMATH_CALUDE_zero_in_A_l1756_175676

def A : Set ℝ := {x : ℝ | x * (x - 2) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by sorry

end NUMINAMATH_CALUDE_zero_in_A_l1756_175676


namespace NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l1756_175641

/-- Given that Joe has 50 toy cars initially and will have 62 cars after getting more,
    prove that he needs to get 12 more toy cars. -/
theorem joe_needs_twelve_more_cars 
  (initial_cars : ℕ) 
  (final_cars : ℕ) 
  (h1 : initial_cars = 50) 
  (h2 : final_cars = 62) : 
  final_cars - initial_cars = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l1756_175641


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l1756_175602

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) :
  a > 0 ∧ b > 0 ∧
  a / b = 4 / 5 ∧
  x = a * 1.25 ∧
  m = b * (1 - p / 100) ∧
  m / x = 0.2
  → p = 80 := by sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l1756_175602


namespace NUMINAMATH_CALUDE_rational_division_equality_l1756_175693

theorem rational_division_equality : 
  (-2 / 21) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_rational_division_equality_l1756_175693


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l1756_175677

theorem water_mixture_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 150 ∧
  added_water = 10 ∧
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l1756_175677


namespace NUMINAMATH_CALUDE_total_players_on_ground_l1756_175616

theorem total_players_on_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 12)
  (h2 : hockey_players = 17)
  (h3 : football_players = 11)
  (h4 : softball_players = 10) :
  cricket_players + hockey_players + football_players + softball_players = 50 := by
sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l1756_175616


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1756_175619

/-- Jessie's weight loss calculation -/
theorem jessie_weight_loss (initial_weight current_weight : ℕ) :
  initial_weight = 69 →
  current_weight = 34 →
  initial_weight - current_weight = 35 := by
sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1756_175619


namespace NUMINAMATH_CALUDE_find_y_value_l1756_175611

-- Define the operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- State the theorem
theorem find_y_value : ∃ y : ℤ, customOp y 12 = 110 ∧ y = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1756_175611


namespace NUMINAMATH_CALUDE_new_students_calculation_l1756_175649

/-- The number of elementary schools in Lansing -/
def num_schools : ℝ := 25.0

/-- The average number of new students per school -/
def avg_students_per_school : ℝ := 9.88

/-- The total number of new elementary students in Lansing -/
def total_new_students : ℕ := 247

/-- Theorem stating that the total number of new elementary students in Lansing
    is equal to the product of the number of schools and the average number of
    new students per school, rounded to the nearest integer. -/
theorem new_students_calculation :
  total_new_students = round (num_schools * avg_students_per_school) :=
by sorry

end NUMINAMATH_CALUDE_new_students_calculation_l1756_175649


namespace NUMINAMATH_CALUDE_system_solution_l1756_175662

theorem system_solution : 
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    (2 * x₁^2 - 5 * x₁ + 3 = 0) ∧ 
    (y₁ = 3 * x₁ + 1) ∧
    (2 * x₂^2 - 5 * x₂ + 3 = 0) ∧ 
    (y₂ = 3 * x₂ + 1) ∧
    (x₁ = 1.5 ∧ y₁ = 5.5) ∧ 
    (x₂ = 1 ∧ y₂ = 4) ∧
    (∀ (x y : ℝ), (2 * x^2 - 5 * x + 3 = 0) ∧ (y = 3 * x + 1) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1756_175662


namespace NUMINAMATH_CALUDE_cinnamon_swirl_division_l1756_175658

theorem cinnamon_swirl_division (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → 
  num_people = 3 → 
  total_pieces = num_people * pieces_per_person → 
  pieces_per_person = 4 := by
sorry

end NUMINAMATH_CALUDE_cinnamon_swirl_division_l1756_175658


namespace NUMINAMATH_CALUDE_number_operation_result_l1756_175636

theorem number_operation_result : 
  let x : ℕ := 265
  (x / 5 + 8 : ℚ) = 61 := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l1756_175636


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1756_175631

theorem regular_polygon_sides : ∀ n : ℕ, 
  n > 2 → (3 * (n * (n - 3) / 2) - n = 21 ↔ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1756_175631


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l1756_175675

/-- Represents the price reduction and restoration process of a jacket --/
theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.10)
  let required_increase_percentage := (initial_price / price_after_second_reduction - 1) * 100
  ∃ ε > 0, abs (required_increase_percentage - 48.15) < ε :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l1756_175675


namespace NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l1756_175642

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l1756_175642


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1756_175667

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ y : ℤ, 4 * |y| - 6 < 34 → y ≤ x) ∧ (4 * |x| - 6 < 34) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1756_175667


namespace NUMINAMATH_CALUDE_min_value_theorem_l1756_175633

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 1) + 9 / y = 1) : 
  4 * x + y ≥ 21 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1 / (x₀ + 1) + 9 / y₀ = 1 ∧ 4 * x₀ + y₀ = 21 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1756_175633


namespace NUMINAMATH_CALUDE_constant_term_in_expansion_l1756_175690

theorem constant_term_in_expansion :
  ∃ (k : ℕ), k > 0 ∧ k < 5 ∧ (2 * 5 = 5 * k) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_in_expansion_l1756_175690


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1756_175655

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (g h : Polynomial ℤ), (Polynomial.degree g ≥ 1) ∧ (Polynomial.degree h ≥ 1) ∧
    (Polynomial.X : Polynomial ℤ)^n + a * (Polynomial.X : Polynomial ℤ)^(n-1) + (p * q : ℤ) = g * h) ↔
  (a = (-1)^n * (p * q : ℤ) + 1 ∨ a = -(p * q : ℤ) - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1756_175655


namespace NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1756_175617

/-- Probability of selecting two non-defective pens from a box -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 16) (h2 : defective_pens = 3) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_non_defective_pens_l1756_175617


namespace NUMINAMATH_CALUDE_mountain_loop_trail_length_l1756_175671

/-- Represents the hiking trip on Mountain Loop Trail -/
structure HikingTrip where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hiking trip -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.day1 + trip.day2 + trip.day3 = 45 ∧
  (trip.day2 + trip.day4) / 2 = 18 ∧
  trip.day3 + trip.day4 + trip.day5 = 60 ∧
  trip.day1 + trip.day4 = 32

/-- The theorem stating the total length of the trail -/
theorem mountain_loop_trail_length (trip : HikingTrip) 
  (h : validHikingTrip trip) : 
  trip.day1 + trip.day2 + trip.day3 + trip.day4 + trip.day5 = 69 := by
  sorry


end NUMINAMATH_CALUDE_mountain_loop_trail_length_l1756_175671


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1756_175666

/-- 
A geometric progression with 2 terms, where:
- The last term is 1/3
- The common ratio is 1/3
- The sum of terms is 40/3
Then, the first term is 10.
-/
theorem geometric_progression_first_term 
  (n : ℕ) 
  (last_term : ℚ) 
  (common_ratio : ℚ) 
  (sum : ℚ) : 
  n = 2 ∧ 
  last_term = 1/3 ∧ 
  common_ratio = 1/3 ∧ 
  sum = 40/3 → 
  ∃ (a : ℚ), a = 10 ∧ sum = a * (1 - common_ratio^n) / (1 - common_ratio) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1756_175666


namespace NUMINAMATH_CALUDE_matrix_product_equality_l1756_175696

theorem matrix_product_equality : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 3, -1; 1, 5, -2; 0, 6, 2]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![3, -1, 0; 2, 1, -4; 5, 0, 1]
  A * B = !![7, 1, -13; 3, 4, -22; 22, 6, -22] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l1756_175696


namespace NUMINAMATH_CALUDE_integer_fraction_count_l1756_175692

theorem integer_fraction_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 0 < n ∧ n < 50 ∧ ∃ k : ℕ, n = k * (50 - n)) ∧ 
    Finset.card S = 2 := by sorry

end NUMINAMATH_CALUDE_integer_fraction_count_l1756_175692


namespace NUMINAMATH_CALUDE_triangular_prism_theorem_l1756_175645

theorem triangular_prism_theorem (V k : ℝ) (S H : Fin 4 → ℝ) : 
  (∀ i : Fin 4, S i = (i.val + 1 : ℕ) * k) →
  (∀ i : Fin 4, V = (1/3) * S i * H i) →
  H 0 + 2 * H 1 + 3 * H 2 + 4 * H 3 = 3 * V / k :=
by sorry

end NUMINAMATH_CALUDE_triangular_prism_theorem_l1756_175645


namespace NUMINAMATH_CALUDE_rational_function_property_l1756_175680

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ  -- Linear function
  q : ℝ → ℝ  -- Quadratic function
  linear_p : ∃ a b : ℝ, ∀ x, p x = a * x + b
  quadratic_q : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_minus_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_through_point : p 1 / q 1 = 2

theorem rational_function_property (f : RationalFunction) : f.p 0 / f.q 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_property_l1756_175680


namespace NUMINAMATH_CALUDE_kerosene_cost_calculation_l1756_175630

/-- The cost of a certain number of eggs in cents -/
def egg_cost : ℝ := sorry

/-- The cost of a pound of rice in cents -/
def rice_cost : ℝ := 24

/-- The cost of a half-liter of kerosene in cents -/
def half_liter_kerosene_cost : ℝ := sorry

/-- The cost of a liter of kerosene in cents -/
def liter_kerosene_cost : ℝ := sorry

theorem kerosene_cost_calculation :
  (egg_cost = rice_cost) →
  (half_liter_kerosene_cost = 6 * egg_cost) →
  (liter_kerosene_cost = 2 * half_liter_kerosene_cost) →
  liter_kerosene_cost = 288 := by
  sorry

end NUMINAMATH_CALUDE_kerosene_cost_calculation_l1756_175630


namespace NUMINAMATH_CALUDE_recipe_calculation_l1756_175663

/-- Represents the relationship between flour, cookies, and sugar -/
structure RecipeRelation where
  flour_to_cookies : ℝ → ℝ  -- Function from flour to cookies
  flour_to_sugar : ℝ → ℝ    -- Function from flour to sugar

/-- Given the recipe relationships, prove the number of cookies and amount of sugar for 4 cups of flour -/
theorem recipe_calculation (r : RecipeRelation) 
  (h1 : r.flour_to_cookies 3 = 24)  -- 24 cookies from 3 cups of flour
  (h2 : r.flour_to_sugar 3 = 1.5)   -- 1.5 cups of sugar for 3 cups of flour
  (h3 : ∀ x y, r.flour_to_cookies (x * y) = r.flour_to_cookies x * y)  -- Linear relationship for cookies
  (h4 : ∀ x y, r.flour_to_sugar (x * y) = r.flour_to_sugar x * y)      -- Linear relationship for sugar
  : r.flour_to_cookies 4 = 32 ∧ r.flour_to_sugar 4 = 2 := by
  sorry

#check recipe_calculation

end NUMINAMATH_CALUDE_recipe_calculation_l1756_175663


namespace NUMINAMATH_CALUDE_smallest_base_for_125_l1756_175648

theorem smallest_base_for_125 : 
  ∃ (b : ℕ), b = 6 ∧ 
  (∀ (x : ℕ), x < b → (x^3 ≤ 125 → x^2 < 125)) ∧
  (b^2 ≤ 125 ∧ 125 < b^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_125_l1756_175648


namespace NUMINAMATH_CALUDE_floor_negative_seven_thirds_l1756_175600

theorem floor_negative_seven_thirds : ⌊(-7 : ℚ) / 3⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_thirds_l1756_175600


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l1756_175635

/-- Represents the number of varieties in a category -/
structure Category where
  varieties : ℕ

/-- Represents the total population of varieties -/
def total_population (categories : List Category) : ℕ :=
  categories.map (·.varieties) |> List.sum

/-- Calculates the number of items in a stratified sample for a given category -/
def stratified_sample_size (category : Category) (total_pop : ℕ) (sample_size : ℕ) : ℕ :=
  (category.varieties * sample_size) / total_pop

/-- Theorem: The sum of vegetable oils and fruits/vegetables in a stratified sample is 6 -/
theorem stratified_sample_sum (vegetable_oils fruits_vegetables : Category)
    (h1 : vegetable_oils.varieties = 10)
    (h2 : fruits_vegetables.varieties = 20)
    (h3 : total_population [vegetable_oils, fruits_vegetables] = 30)
    (h4 : total_population [Category.mk 40, vegetable_oils, Category.mk 30, fruits_vegetables] = 100) :
    stratified_sample_size vegetable_oils 100 20 + stratified_sample_size fruits_vegetables 100 20 = 6 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_sum_l1756_175635


namespace NUMINAMATH_CALUDE_books_ratio_l1756_175638

theorem books_ratio (harry_books : ℕ) (total_books : ℕ) : 
  harry_books = 50 →
  total_books = 175 →
  ∃ (flora_books : ℕ),
    flora_books = 2 * harry_books ∧
    harry_books + flora_books + (harry_books / 2) = total_books :=
by sorry

end NUMINAMATH_CALUDE_books_ratio_l1756_175638


namespace NUMINAMATH_CALUDE_sprinting_competition_races_verify_sprinting_competition_races_l1756_175647

/-- Calculates the number of races needed to determine a champion in a sprinting competition. -/
def races_needed (total_sprinters : ℕ) (sprinters_per_race : ℕ) (eliminations_per_race : ℕ) : ℕ :=
  (total_sprinters - 1) / eliminations_per_race

/-- Theorem stating that 43 races are needed for the given competition setup. -/
theorem sprinting_competition_races : 
  races_needed 216 6 5 = 43 := by
  sorry

/-- Verifies the result by simulating rounds of the competition. -/
def verify_races (total_sprinters : ℕ) (sprinters_per_race : ℕ) : ℕ :=
  let first_round := total_sprinters / sprinters_per_race
  let second_round := first_round / sprinters_per_race
  let third_round := if second_round ≥ sprinters_per_race then 1 else 0
  first_round + second_round + third_round

/-- Theorem stating that the verification method also yields 43 races. -/
theorem verify_sprinting_competition_races :
  verify_races 216 6 = 43 := by
  sorry

end NUMINAMATH_CALUDE_sprinting_competition_races_verify_sprinting_competition_races_l1756_175647


namespace NUMINAMATH_CALUDE_intersection_passes_through_center_l1756_175651

-- Define the cube
def Cube : Type := Unit

-- Define a point in 3D space
def Point : Type := Unit

-- Define a plane
def Plane : Type := Unit

-- Define a hexagon
structure Hexagon :=
  (A B C D E F : Point)

-- Define the intersection of a cube and a plane
def intersection (c : Cube) (p : Plane) : Hexagon := sorry

-- Define the center of a cube
def center (c : Cube) : Point := sorry

-- Define a function to check if three lines intersect at a point
def intersect_at (p1 p2 p3 p4 p5 p6 : Point) (O : Point) : Prop := sorry

-- Theorem statement
theorem intersection_passes_through_center (c : Cube) (p : Plane) :
  let h := intersection c p
  intersect_at h.A h.D h.B h.E h.C h.F (center c) := by sorry

end NUMINAMATH_CALUDE_intersection_passes_through_center_l1756_175651


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l1756_175607

theorem sqrt_sum_squares_equals_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + b*c + c*a = 0 ∧ a + b + c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l1756_175607


namespace NUMINAMATH_CALUDE_gcd_2048_2101_l1756_175684

theorem gcd_2048_2101 : Nat.gcd 2048 2101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2048_2101_l1756_175684


namespace NUMINAMATH_CALUDE_square_root_of_64_l1756_175686

theorem square_root_of_64 : {x : ℝ | x^2 = 64} = {-8, 8} := by sorry

end NUMINAMATH_CALUDE_square_root_of_64_l1756_175686


namespace NUMINAMATH_CALUDE_exists_multiple_factorizations_l1756_175669

-- Define the set V
def V (p : Nat) : Set Nat :=
  {n : Nat | ∃ k : Nat, (n = k * p + 1 ∨ n = k * p - 1) ∧ k > 0}

-- Define indecomposability in V
def isIndecomposable (p : Nat) (n : Nat) : Prop :=
  n ∈ V p ∧ ∀ k l : Nat, k ∈ V p → l ∈ V p → n ≠ k * l

-- Theorem statement
theorem exists_multiple_factorizations (p : Nat) (h : p > 5) :
  ∃ N : Nat, N ∈ V p ∧
    ∃ (factors1 factors2 : List Nat),
      factors1 ≠ factors2 ∧
      (∀ f ∈ factors1, isIndecomposable p f) ∧
      (∀ f ∈ factors2, isIndecomposable p f) ∧
      N = factors1.prod ∧
      N = factors2.prod :=
by sorry

end NUMINAMATH_CALUDE_exists_multiple_factorizations_l1756_175669


namespace NUMINAMATH_CALUDE_max_leap_years_in_200_years_l1756_175661

/-- A calendar system where leap years occur every 5 years -/
structure Calendar :=
  (leap_year_frequency : ℕ)
  (total_years : ℕ)
  (h_leap_frequency : leap_year_frequency = 5)

/-- The number of leap years in the calendar system -/
def leap_years (c : Calendar) : ℕ := c.total_years / c.leap_year_frequency

/-- Theorem: The maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_200_years (c : Calendar) 
  (h_total_years : c.total_years = 200) : 
  leap_years c = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_leap_years_in_200_years_l1756_175661


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1756_175612

theorem rational_equation_solution (x : ℝ) : 
  (1 / (x^2 + 8*x - 6) + 1 / (x^2 + 5*x - 6) + 1 / (x^2 - 14*x - 6) = 0) ↔ 
  (x = 3 ∨ x = -2 ∨ x = -6 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1756_175612


namespace NUMINAMATH_CALUDE_coffee_lasts_12_days_l1756_175681

-- Define the constants
def coffee_lbs : ℕ := 3
def cups_per_lb : ℕ := 40
def weekday_consumption : ℕ := 3 + 2 + 4
def weekend_consumption : ℕ := 2 + 3 + 5
def days_in_week : ℕ := 7
def weekdays_per_week : ℕ := 5
def weekend_days_per_week : ℕ := 2

-- Define the theorem
theorem coffee_lasts_12_days :
  let total_cups := coffee_lbs * cups_per_lb
  let weekly_consumption := weekday_consumption * weekdays_per_week + weekend_consumption * weekend_days_per_week
  let days_coffee_lasts := (total_cups * days_in_week) / weekly_consumption
  days_coffee_lasts = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_lasts_12_days_l1756_175681


namespace NUMINAMATH_CALUDE_xy_length_l1756_175639

/-- Triangle similarity and side length properties -/
structure TriangleSimilarity where
  PQ : ℝ
  QR : ℝ
  YZ : ℝ
  perimeter_XYZ : ℝ
  similar : Bool  -- Represents that PQR is similar to XYZ

/-- Main theorem: XY length in similar triangles -/
theorem xy_length (t : TriangleSimilarity) 
  (h1 : t.PQ = 8)
  (h2 : t.QR = 16)
  (h3 : t.YZ = 24)
  (h4 : t.perimeter_XYZ = 60)
  (h5 : t.similar = true) : 
  ∃ XY : ℝ, XY = 12 ∧ XY + t.YZ + (t.perimeter_XYZ - XY - t.YZ) = t.perimeter_XYZ :=
sorry

end NUMINAMATH_CALUDE_xy_length_l1756_175639


namespace NUMINAMATH_CALUDE_total_cakes_served_l1756_175665

def cakes_served_lunch_today : ℕ := 5
def cakes_served_dinner_today : ℕ := 6
def cakes_served_yesterday : ℕ := 3

theorem total_cakes_served :
  cakes_served_lunch_today + cakes_served_dinner_today + cakes_served_yesterday = 14 :=
by sorry

end NUMINAMATH_CALUDE_total_cakes_served_l1756_175665


namespace NUMINAMATH_CALUDE_smallest_number_l1756_175618

theorem smallest_number (π : Real) : 
  -π < -3 ∧ -π < -Real.sqrt 2 ∧ -π < -(5/2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1756_175618


namespace NUMINAMATH_CALUDE_arctan_sum_special_l1756_175699

theorem arctan_sum_special : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_l1756_175699


namespace NUMINAMATH_CALUDE_B_coordinates_l1756_175694

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Moves a point up by a given number of units -/
def moveUp (p : Point) (units : ℤ) : Point :=
  { x := p.x, y := p.y + units }

/-- Moves a point left by a given number of units -/
def moveLeft (p : Point) (units : ℤ) : Point :=
  { x := p.x - units, y := p.y }

/-- The initial point A -/
def A : Point := { x := -3, y := -5 }

/-- The final point B after moving A -/
def B : Point := moveLeft (moveUp A 4) 3

/-- Theorem stating that B has the correct coordinates -/
theorem B_coordinates : B.x = -6 ∧ B.y = -1 := by sorry

end NUMINAMATH_CALUDE_B_coordinates_l1756_175694


namespace NUMINAMATH_CALUDE_harrys_creation_weight_is_25_l1756_175653

/-- The weight of Harry's custom creation at the gym -/
def harrys_creation_weight (blue_weight green_weight : ℕ) (blue_count green_count bar_weight : ℕ) : ℕ :=
  blue_weight * blue_count + green_weight * green_count + bar_weight

/-- Theorem stating that Harry's creation weighs 25 pounds -/
theorem harrys_creation_weight_is_25 :
  harrys_creation_weight 2 3 4 5 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_harrys_creation_weight_is_25_l1756_175653


namespace NUMINAMATH_CALUDE_complex_real_condition_l1756_175640

theorem complex_real_condition (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑(1 : ℝ) - Complex.I) * (↑a + Complex.I) ∈ Set.range (Complex.ofReal) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1756_175640


namespace NUMINAMATH_CALUDE_cost_increase_doubles_b_l1756_175625

/-- The cost function for a given parameter b and coefficient t -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem stating that if the new cost is 1600% of the original cost,
    then the new value of b is 2 times the original value -/
theorem cost_increase_doubles_b (t : ℝ) (b₁ b₂ : ℝ) (h : t > 0) :
  cost t b₂ = 16 * cost t b₁ → b₂ = 2 * b₁ := by
  sorry


end NUMINAMATH_CALUDE_cost_increase_doubles_b_l1756_175625


namespace NUMINAMATH_CALUDE_sum_of_C_and_D_l1756_175660

/-- The number of four-digit numbers that are both odd and divisible by 5 -/
def C : ℕ := 900

/-- The number of four-digit numbers that are divisible by 25 -/
def D : ℕ := 360

/-- Theorem: The sum of C and D is 1260 -/
theorem sum_of_C_and_D : C + D = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_C_and_D_l1756_175660


namespace NUMINAMATH_CALUDE_max_value_is_b_l1756_175637

theorem max_value_is_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b = max (1/2) (max b (max (2*a*b) (a^2 + b^2))) := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_b_l1756_175637


namespace NUMINAMATH_CALUDE_sum_of_scores_l1756_175629

/-- The sum of scores in a guessing game -/
theorem sum_of_scores (hajar_score : ℕ) (score_difference : ℕ) : 
  hajar_score = 24 →
  score_difference = 21 →
  hajar_score + (hajar_score + score_difference) = 69 := by
  sorry

#check sum_of_scores

end NUMINAMATH_CALUDE_sum_of_scores_l1756_175629


namespace NUMINAMATH_CALUDE_profit_per_meter_cloth_l1756_175614

theorem profit_per_meter_cloth (meters_sold : ℕ) (selling_price : ℕ) (cost_price_per_meter : ℕ) 
  (h1 : meters_sold = 66)
  (h2 : selling_price = 660)
  (h3 : cost_price_per_meter = 5) :
  (selling_price - meters_sold * cost_price_per_meter) / meters_sold = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_cloth_l1756_175614


namespace NUMINAMATH_CALUDE_multiple_of_twelve_l1756_175601

theorem multiple_of_twelve (x : ℤ) : 
  (∃ k : ℤ, 7 * x - 3 = 12 * k) ↔ 
  (∃ t : ℤ, x = 12 * t + 9 ∨ x = 12 * t + 1029) :=
sorry

end NUMINAMATH_CALUDE_multiple_of_twelve_l1756_175601


namespace NUMINAMATH_CALUDE_least_k_for_error_bound_l1756_175609

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * u n - 3 * (u n)^2

def L : ℚ := 1/3

theorem least_k_for_error_bound :
  (∀ k < 9, |u k - L| > 1/2^500) ∧
  |u 9 - L| ≤ 1/2^500 := by
  sorry

end NUMINAMATH_CALUDE_least_k_for_error_bound_l1756_175609


namespace NUMINAMATH_CALUDE_louise_boxes_l1756_175654

/-- The number of pencils each box can hold -/
def pencils_per_box : ℕ := 20

/-- The number of red pencils Louise has -/
def red_pencils : ℕ := 20

/-- The number of blue pencils Louise has -/
def blue_pencils : ℕ := 2 * red_pencils

/-- The number of yellow pencils Louise has -/
def yellow_pencils : ℕ := 40

/-- The number of green pencils Louise has -/
def green_pencils : ℕ := red_pencils + blue_pencils

/-- The total number of pencils Louise has -/
def total_pencils : ℕ := red_pencils + blue_pencils + yellow_pencils + green_pencils

/-- The number of boxes Louise needs -/
def boxes_needed : ℕ := total_pencils / pencils_per_box

theorem louise_boxes : boxes_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_louise_boxes_l1756_175654


namespace NUMINAMATH_CALUDE_toms_next_birthday_l1756_175652

theorem toms_next_birthday (sally tom jenny : ℝ) 
  (h1 : sally = 1.25 * tom)  -- Sally is 25% older than Tom
  (h2 : tom = 0.7 * jenny)   -- Tom is 30% younger than Jenny
  (h3 : sally + tom + jenny = 30)  -- Sum of ages is 30
  : ⌊tom⌋ + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_toms_next_birthday_l1756_175652


namespace NUMINAMATH_CALUDE_solve_system_for_w_l1756_175682

theorem solve_system_for_w (x y z w : ℝ) 
  (eq1 : 2*x + y + z + w = 1)
  (eq2 : x + 3*y + z + w = 2)
  (eq3 : x + y + 4*z + w = 3)
  (eq4 : x + y + z + 5*w = 25) : 
  w = 11/2 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_w_l1756_175682


namespace NUMINAMATH_CALUDE_pool_filling_time_l1756_175604

/-- Proves that filling a pool of given capacity with a specific number of hoses 
    and flow rate takes the calculated number of hours -/
theorem pool_filling_time 
  (pool_capacity : ℕ) 
  (num_hoses : ℕ) 
  (flow_rate_per_hose : ℕ) 
  (hours_to_fill : ℕ) 
  (h1 : pool_capacity = 32000)
  (h2 : num_hoses = 3)
  (h3 : flow_rate_per_hose = 4)
  (h4 : hours_to_fill = 44) : 
  pool_capacity = num_hoses * flow_rate_per_hose * 60 * hours_to_fill :=
by
  sorry

#check pool_filling_time

end NUMINAMATH_CALUDE_pool_filling_time_l1756_175604


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1756_175621

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1756_175621


namespace NUMINAMATH_CALUDE_probability_ten_red_balls_in_twelve_draws_l1756_175656

theorem probability_ten_red_balls_in_twelve_draws 
  (total_balls : Nat) (white_balls : Nat) (red_balls : Nat)
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 5)
  (h3 : red_balls = 3) :
  let p_red := red_balls / total_balls
  let p_white := white_balls / total_balls
  let n := 11  -- number of draws before the last one
  let k := 9   -- number of red balls in the first 11 draws
  Nat.choose n k * p_red^k * p_white^(n-k) * p_red = 
    Nat.choose 11 9 * (3/8)^9 * (5/8)^2 * (3/8) :=
by sorry

end NUMINAMATH_CALUDE_probability_ten_red_balls_in_twelve_draws_l1756_175656


namespace NUMINAMATH_CALUDE_car_distance_proof_l1756_175695

/-- Proves that the distance a car needs to cover is 630 km, given the original time, 
    new time factor, and new speed. -/
theorem car_distance_proof (original_time : ℝ) (new_time_factor : ℝ) (new_speed : ℝ) : 
  original_time = 6 → 
  new_time_factor = 3 / 2 → 
  new_speed = 70 → 
  original_time * new_time_factor * new_speed = 630 := by
  sorry

#check car_distance_proof

end NUMINAMATH_CALUDE_car_distance_proof_l1756_175695


namespace NUMINAMATH_CALUDE_managers_salary_l1756_175634

/-- Given an organization with employees and their salaries, this theorem proves
    the salary of an additional member that would increase the average by a specific amount. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 20 →
  avg_salary = 1700 →
  avg_increase = 100 →
  (num_employees * avg_salary + 3800) / (num_employees + 1) = avg_salary + avg_increase := by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l1756_175634


namespace NUMINAMATH_CALUDE_sequence_inequality_l1756_175685

theorem sequence_inequality (A B : ℝ) (a : ℕ → ℝ) 
  (hA : A > 1) (hB : B > 1) (ha : ∀ n, 1 ≤ a n ∧ a n ≤ A * B) :
  ∃ b : ℕ → ℝ, (∀ n, 1 ≤ b n ∧ b n ≤ A) ∧
    (∀ m n : ℕ, a m / a n ≤ B * (b m / b n)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1756_175685


namespace NUMINAMATH_CALUDE_classroom_students_l1756_175672

theorem classroom_students (total_notebooks : ℕ) (notebooks_per_half1 : ℕ) (notebooks_per_half2 : ℕ) :
  total_notebooks = 112 →
  notebooks_per_half1 = 5 →
  notebooks_per_half2 = 3 →
  ∃ (num_students : ℕ),
    num_students % 2 = 0 ∧
    (num_students / 2) * notebooks_per_half1 + (num_students / 2) * notebooks_per_half2 = total_notebooks ∧
    num_students = 28 := by
  sorry

end NUMINAMATH_CALUDE_classroom_students_l1756_175672


namespace NUMINAMATH_CALUDE_matthews_friends_l1756_175644

theorem matthews_friends (total_crackers : ℕ) (crackers_per_friend : ℕ) 
  (h1 : total_crackers = 22)
  (h2 : crackers_per_friend = 2) :
  total_crackers / crackers_per_friend = 11 := by
  sorry

end NUMINAMATH_CALUDE_matthews_friends_l1756_175644


namespace NUMINAMATH_CALUDE_school_books_count_l1756_175623

def total_books : ℕ := 58
def sports_books : ℕ := 39

theorem school_books_count : total_books - sports_books = 19 := by
  sorry

end NUMINAMATH_CALUDE_school_books_count_l1756_175623


namespace NUMINAMATH_CALUDE_arcade_candy_cost_l1756_175664

theorem arcade_candy_cost (tickets_game1 tickets_game2 candies : ℕ) 
  (h1 : tickets_game1 = 33)
  (h2 : tickets_game2 = 9)
  (h3 : candies = 7) :
  (tickets_game1 + tickets_game2) / candies = 6 :=
by sorry

end NUMINAMATH_CALUDE_arcade_candy_cost_l1756_175664


namespace NUMINAMATH_CALUDE_subtract_p_q_equals_five_twentyfourths_l1756_175691

theorem subtract_p_q_equals_five_twentyfourths 
  (p q : ℚ) 
  (hp : 3 / p = 8) 
  (hq : 3 / q = 18) : 
  p - q = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_subtract_p_q_equals_five_twentyfourths_l1756_175691


namespace NUMINAMATH_CALUDE_middle_term_is_average_l1756_175687

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term (middle term)
  d : ℝ  -- fourth term
  e : ℝ  -- fifth term
  is_arithmetic : ∃ (r : ℝ), b - a = r ∧ c - b = r ∧ d - c = r ∧ e - d = r

/-- The theorem stating that the middle term of a 5-term arithmetic sequence
    is the average of the first and last terms -/
theorem middle_term_is_average (seq : ArithmeticSequence5) (h1 : seq.a = 23) (h2 : seq.e = 47) :
  seq.c = 35 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_is_average_l1756_175687


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1756_175673

/-- Given a positive term geometric sequence {a_n} with common ratio q,
    prove that if 2a_5 - 3a_4 = 2a_3, then q = 2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- Sequence of real numbers indexed by natural numbers
  (q : ℝ)      -- Common ratio
  (h_pos : ∀ n, a n > 0)  -- Positive term sequence
  (h_geom : ∀ n, a (n + 1) = q * a n)  -- Geometric sequence property
  (h_eq : 2 * a 5 - 3 * a 4 = 2 * a 3)  -- Given equation
  : q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1756_175673


namespace NUMINAMATH_CALUDE_mutuallyExclusive_but_not_complementary_l1756_175679

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first : Color)
  (second : Color)

/-- The set of all possible outcomes when drawing two balls -/
def sampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite : Set DrawOutcome := sorry

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite : Set DrawOutcome := sorry

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = sampleSpace

theorem mutuallyExclusive_but_not_complementary :
  mutuallyExclusive exactlyOneWhite exactlyTwoWhite ∧
  ¬complementary exactlyOneWhite exactlyTwoWhite :=
sorry

end NUMINAMATH_CALUDE_mutuallyExclusive_but_not_complementary_l1756_175679


namespace NUMINAMATH_CALUDE_study_group_probability_l1756_175646

theorem study_group_probability (total_members : ℕ) (h1 : total_members > 0) : 
  let women_percentage : ℝ := 0.9
  let lawyer_percentage : ℝ := 0.6
  let women_count : ℝ := women_percentage * total_members
  let women_lawyers_count : ℝ := lawyer_percentage * women_count
  women_lawyers_count / total_members = 0.54 := by sorry

end NUMINAMATH_CALUDE_study_group_probability_l1756_175646


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1756_175627

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x | x < -5 ∨ x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1756_175627


namespace NUMINAMATH_CALUDE_age_difference_l1756_175643

/-- Given Sierra's current age and Diaz's age 20 years from now, 
    prove that the difference between (10 times Diaz's current age minus 40) 
    and (10 times Sierra's current age) is 20. -/
theorem age_difference (sierra_age : ℕ) (diaz_future_age : ℕ) : 
  sierra_age = 30 → 
  diaz_future_age = 56 → 
  (10 * (diaz_future_age - 20) - 40) - (10 * sierra_age) = 20 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l1756_175643


namespace NUMINAMATH_CALUDE_problem_statement_l1756_175626

/-- Given two expressions A and B in terms of a and b, prove that A + 2B has a specific form
    and that when it's independent of b, a has a specific value. -/
theorem problem_statement (a b : ℝ) : 
  let A := 2*a^2 + 3*a*b - 2*b - 1
  let B := -a^2 - a*b + 1
  (A + 2*B = a*b - 2*b + 1) ∧ 
  (∀ b, A + 2*B = a*b - 2*b + 1 → a = 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1756_175626


namespace NUMINAMATH_CALUDE_crepe_myrtle_count_l1756_175610

theorem crepe_myrtle_count (total : ℕ) (pink : ℕ) (red : ℕ) (white : ℕ) : 
  total = 42 →
  pink = total / 3 →
  red = 2 →
  white > pink →
  white > red →
  total = pink + red + white →
  white = 26 := by
sorry

end NUMINAMATH_CALUDE_crepe_myrtle_count_l1756_175610


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1756_175620

theorem abs_sum_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1756_175620


namespace NUMINAMATH_CALUDE_cube_increase_correct_l1756_175668

/-- Represents the percentage increase in a cube's dimensions and properties -/
structure CubeIncrease where
  edge : ℝ
  surface_area : ℝ
  volume : ℝ

/-- The percentage increases when a cube's edge is increased by 60% -/
def cube_increase : CubeIncrease :=
  { edge := 60
  , surface_area := 156
  , volume := 309.6 }

theorem cube_increase_correct :
  let original_edge := 1
  let new_edge := original_edge * (1 + cube_increase.edge / 100)
  let original_surface_area := 6 * original_edge^2
  let new_surface_area := 6 * new_edge^2
  let original_volume := original_edge^3
  let new_volume := new_edge^3
  (new_surface_area / original_surface_area - 1) * 100 = cube_increase.surface_area ∧
  (new_volume / original_volume - 1) * 100 = cube_increase.volume :=
by sorry

end NUMINAMATH_CALUDE_cube_increase_correct_l1756_175668


namespace NUMINAMATH_CALUDE_order_of_magnitude_l1756_175650

theorem order_of_magnitude : Real.log 0.65 < 0.65 ∧ 0.65 < 50.6 := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l1756_175650


namespace NUMINAMATH_CALUDE_min_teachers_for_given_problem_l1756_175606

/-- Represents the number of teachers for each subject -/
structure SubjectTeachers where
  english : Nat
  history : Nat
  geography : Nat

/-- The minimum number of teachers required given the subject teachers -/
def minTeachersRequired (s : SubjectTeachers) : Nat :=
  sorry

/-- Theorem stating the minimum number of teachers required for the given problem -/
theorem min_teachers_for_given_problem :
  let s : SubjectTeachers := ⟨9, 7, 6⟩
  minTeachersRequired s = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_teachers_for_given_problem_l1756_175606


namespace NUMINAMATH_CALUDE_equation_solution_range_l1756_175659

theorem equation_solution_range (x m : ℝ) : 
  (x / (x - 3) - 2 = m / (x - 3) ∧ x > 0) → (m < 6 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1756_175659


namespace NUMINAMATH_CALUDE_bicycle_profit_calculation_l1756_175683

theorem bicycle_profit_calculation (profit_A profit_B final_price : ℝ) :
  profit_A = 0.60 ∧ profit_B = 0.25 ∧ final_price = 225 →
  ∃ cost_price_A : ℝ,
    cost_price_A * (1 + profit_A) * (1 + profit_B) = final_price ∧
    cost_price_A = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_profit_calculation_l1756_175683


namespace NUMINAMATH_CALUDE_papers_per_envelope_l1756_175632

theorem papers_per_envelope 
  (total_papers : ℕ) 
  (num_envelopes : ℕ) 
  (h1 : total_papers = 120) 
  (h2 : num_envelopes = 12) : 
  total_papers / num_envelopes = 10 := by
sorry

end NUMINAMATH_CALUDE_papers_per_envelope_l1756_175632


namespace NUMINAMATH_CALUDE_no_m_exists_for_equal_sets_l1756_175670

theorem no_m_exists_for_equal_sets : ¬∃ m : ℝ, 
  {x : ℝ | x^2 - 8*x - 20 ≤ 0} = {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m} := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equal_sets_l1756_175670


namespace NUMINAMATH_CALUDE_time_after_elapsed_hours_l1756_175615

def hours_elapsed : ℕ := 2023
def starting_time : ℕ := 3
def clock_hours : ℕ := 12

theorem time_after_elapsed_hours :
  (starting_time + hours_elapsed) % clock_hours = 10 :=
by sorry

end NUMINAMATH_CALUDE_time_after_elapsed_hours_l1756_175615


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1756_175622

theorem perfect_square_condition (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, x^2 + x + 2*m = y^2) → m = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1756_175622


namespace NUMINAMATH_CALUDE_range_of_t_l1756_175689

/-- The solution set of the inequality x^2 - 3x + t ≤ 0 -/
def A (t : ℝ) : Set ℝ := {x : ℝ | x^2 - 3*x + t ≤ 0}

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t :
  ∀ t : ℝ, (∃ x : ℝ, x ∈ A t ∧ x ≤ t) ↔ t ∈ Set.Icc 0 (9/4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l1756_175689


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_A_subset_complement_B_l1756_175688

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | x ≥ 2}

-- Part 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

-- Part 2
theorem range_of_a_when_A_subset_complement_B :
  ∀ a : ℝ, A a ⊆ (Set.univ \ B) → a < 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_A_subset_complement_B_l1756_175688
