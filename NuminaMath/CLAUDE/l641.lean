import Mathlib

namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_25_l641_64102

theorem smallest_multiple_of_45_and_60_not_25 :
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 60 ∣ n ∧ ¬(25 ∣ n) ∧
  ∀ m : ℕ, m > 0 ∧ 45 ∣ m ∧ 60 ∣ m ∧ ¬(25 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_60_not_25_l641_64102


namespace NUMINAMATH_CALUDE_tangent_line_slope_l641_64186

/-- The curve function f(x) = x³ - 3x² + 2x --/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, (k * x₀ = f x₀) ∧ (∀ x : ℝ, k * x ≤ f x) ∧ (k = f' x₀)) →
  (k = 2 ∨ k = -1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l641_64186


namespace NUMINAMATH_CALUDE_min_sides_convex_polygon_l641_64126

/-- A convex polygon is a closed planar figure with straight sides. -/
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool

/-- Theorem: The minimum number of sides for a convex polygon is 3. -/
theorem min_sides_convex_polygon :
  ∀ p : ConvexPolygon, p.is_convex → p.sides ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_sides_convex_polygon_l641_64126


namespace NUMINAMATH_CALUDE_shifted_function_eq_l641_64108

def original_function (x : ℝ) : ℝ := 2 * x

def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f x - shift

def shifted_function : ℝ → ℝ := vertical_shift original_function 2

theorem shifted_function_eq : shifted_function = fun x => 2 * x - 2 := by sorry

end NUMINAMATH_CALUDE_shifted_function_eq_l641_64108


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l641_64140

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := -Real.log x

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂) :=
by sorry

end

end NUMINAMATH_CALUDE_f_satisfies_conditions_l641_64140


namespace NUMINAMATH_CALUDE_cylinder_volume_problem_l641_64173

theorem cylinder_volume_problem (h₁ : ℝ) (h₂ : ℝ) (r₁ r₂ : ℝ) :
  r₁ = 7 →
  r₂ = 1.2 * r₁ →
  h₂ = 0.85 * h₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  π * r₁^2 * h₁ = 49 * π * h₁ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_problem_l641_64173


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l641_64153

theorem product_of_three_numbers 
  (x y z : ℝ) 
  (sum_eq : x + y = 18) 
  (sum_squares_eq : x^2 + y^2 = 220) 
  (diff_eq : z = x - y) : 
  x * y * z = 104 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l641_64153


namespace NUMINAMATH_CALUDE_car_distance_problem_car_distance_solution_l641_64146

/-- The distance a car needs to cover, given specific time and speed conditions. -/
theorem car_distance_problem (initial_time : ℝ) (speed_factor : ℝ) (new_speed : ℝ) : ℝ :=
  let new_time := initial_time * speed_factor
  let distance := new_speed * new_time
  distance

/-- Proof of the specific car distance problem -/
theorem car_distance_solution :
  car_distance_problem 6 (3/2) 40 = 360 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_problem_car_distance_solution_l641_64146


namespace NUMINAMATH_CALUDE_marble_ratio_l641_64114

def marble_problem (pink : ℕ) (orange_diff : ℕ) (total : ℕ) : Prop :=
  let orange := pink - orange_diff
  let purple := total - pink - orange
  purple = 4 * orange

theorem marble_ratio :
  marble_problem 13 9 33 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l641_64114


namespace NUMINAMATH_CALUDE_largest_gold_coin_distribution_l641_64177

theorem largest_gold_coin_distribution (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ k : ℕ, n = 15 * k + 3) : n ≤ 138 := by
  sorry

end NUMINAMATH_CALUDE_largest_gold_coin_distribution_l641_64177


namespace NUMINAMATH_CALUDE_brown_eyes_ratio_l641_64104

/-- Represents the number of people with different eye colors in a theater. -/
structure TheaterEyeColors where
  total : ℕ
  blue : ℕ
  black : ℕ
  green : ℕ
  brown : ℕ

/-- Theorem stating the ratio of people with brown eyes to total people in the theater. -/
theorem brown_eyes_ratio (t : TheaterEyeColors) :
  t.total = 100 ∧ 
  t.blue = 19 ∧ 
  t.black = t.total / 4 ∧ 
  t.green = 6 ∧ 
  t.brown = t.total - (t.blue + t.black + t.green) →
  2 * t.brown = t.total := by
  sorry

#check brown_eyes_ratio

end NUMINAMATH_CALUDE_brown_eyes_ratio_l641_64104


namespace NUMINAMATH_CALUDE_temperature_difference_problem_l641_64106

theorem temperature_difference_problem (M L N : ℝ) : 
  M = L + N →                           -- Minneapolis is N degrees warmer at noon
  (M - 8) - (L + 6) = 4 ∨ (M - 8) - (L + 6) = -4 →  -- Temperature difference at 6:00 PM
  (N = 18 ∨ N = 10) ∧ N * N = 180 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_problem_l641_64106


namespace NUMINAMATH_CALUDE_power_division_equality_l641_64122

theorem power_division_equality (m : ℕ) (h : m = 32^500) : m / 8 = 2^2497 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l641_64122


namespace NUMINAMATH_CALUDE_trouser_original_price_l641_64113

theorem trouser_original_price (sale_price : ℝ) (discount_percent : ℝ) : 
  sale_price = 55 → discount_percent = 45 → 
  ∃ (original_price : ℝ), original_price = 100 ∧ sale_price = original_price * (1 - discount_percent / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_trouser_original_price_l641_64113


namespace NUMINAMATH_CALUDE_bicycle_selling_price_l641_64190

/-- The final selling price of a bicycle given initial cost and profit percentages -/
theorem bicycle_selling_price 
  (initial_cost : ℝ) 
  (profit_a_percent : ℝ) 
  (profit_b_percent : ℝ) 
  (h1 : initial_cost = 150)
  (h2 : profit_a_percent = 20)
  (h3 : profit_b_percent = 25) : 
  initial_cost * (1 + profit_a_percent / 100) * (1 + profit_b_percent / 100) = 225 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_selling_price_l641_64190


namespace NUMINAMATH_CALUDE_inequality_proof_l641_64110

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l641_64110


namespace NUMINAMATH_CALUDE_milkman_profit_is_90_l641_64123

/-- Calculates the profit of a milkman given the following conditions:
  * The milkman has 30 liters of milk
  * 5 liters of water is mixed with 20 liters of pure milk
  * Water is freely available
  * Cost of pure milk is Rs. 18 per liter
  * Milkman sells all the mixture at cost price
-/
def milkman_profit (total_milk : ℕ) (mixed_milk : ℕ) (water : ℕ) (cost_per_liter : ℕ) : ℕ :=
  let mixture_volume := mixed_milk + water
  let mixture_revenue := mixture_volume * cost_per_liter
  let mixed_milk_cost := mixed_milk * cost_per_liter
  mixture_revenue - mixed_milk_cost

/-- The profit of the milkman is Rs. 90 given the specified conditions. -/
theorem milkman_profit_is_90 :
  milkman_profit 30 20 5 18 = 90 := by
  sorry

end NUMINAMATH_CALUDE_milkman_profit_is_90_l641_64123


namespace NUMINAMATH_CALUDE_triangle_inequality_cosine_law_l641_64139

theorem triangle_inequality_cosine_law (x y z α β γ : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_angle_range : 0 ≤ α ∧ α < π ∧ 0 ≤ β ∧ β < π ∧ 0 ≤ γ ∧ γ < π)
  (h_angle_sum : α + β > γ ∧ β + γ > α ∧ γ + α > β) :
  Real.sqrt (x^2 + y^2 - 2*x*y*(Real.cos α)) + Real.sqrt (y^2 + z^2 - 2*y*z*(Real.cos β)) 
  ≥ Real.sqrt (z^2 + x^2 - 2*z*x*(Real.cos γ)) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_cosine_law_l641_64139


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l641_64137

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l641_64137


namespace NUMINAMATH_CALUDE_correct_recommendation_count_l641_64168

/-- Represents the number of recommendation spots for each language -/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the number of male and female candidates -/
structure Candidates :=
  (males : Nat)
  (females : Nat)

/-- Calculate the number of different recommendation plans -/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : Candidates) : Nat :=
  sorry

theorem correct_recommendation_count :
  let spots := RecommendationSpots.mk 2 2 1
  let candidates := Candidates.mk 3 2
  countRecommendationPlans spots candidates = 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_recommendation_count_l641_64168


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l641_64178

theorem simplify_radical_expression :
  (Real.sqrt 6 + 4 * Real.sqrt 3 + 3 * Real.sqrt 2) / 
  ((Real.sqrt 6 + Real.sqrt 3) * (Real.sqrt 3 + Real.sqrt 2)) = 
  Real.sqrt 6 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l641_64178


namespace NUMINAMATH_CALUDE_exists_a_divides_a_squared_minus_a_l641_64112

theorem exists_a_divides_a_squared_minus_a (n k : ℕ) 
  (h1 : n > 1) 
  (h2 : k = (Nat.factors n).card) : 
  ∃ a : ℕ, 1 < a ∧ a < n / k + 1 ∧ n ∣ (a^2 - a) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_divides_a_squared_minus_a_l641_64112


namespace NUMINAMATH_CALUDE_quadratic_not_equal_linear_l641_64172

theorem quadratic_not_equal_linear : ¬∃ (a b c A B : ℝ), a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = A * x + B := by
  sorry

end NUMINAMATH_CALUDE_quadratic_not_equal_linear_l641_64172


namespace NUMINAMATH_CALUDE_mame_on_top_probability_l641_64135

/-- Represents a piece of paper with 8 quadrants -/
structure Paper :=
  (quadrants : Fin 8)

/-- The probability of a specific quadrant being on top -/
def probability_on_top (p : Paper) : ℚ :=
  1 / 8

/-- The quadrant where "MAME" is written -/
def mame_quadrant : Fin 8 := 0

/-- Theorem: The probability of "MAME" being on top is 1/8 -/
theorem mame_on_top_probability :
  probability_on_top {quadrants := mame_quadrant} = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_mame_on_top_probability_l641_64135


namespace NUMINAMATH_CALUDE_angie_pretzels_l641_64156

theorem angie_pretzels (barry_pretzels : ℕ) (shelly_pretzels : ℕ) (angie_pretzels : ℕ) :
  barry_pretzels = 12 →
  shelly_pretzels = barry_pretzels / 2 →
  angie_pretzels = 3 * shelly_pretzels →
  angie_pretzels = 18 := by
sorry

end NUMINAMATH_CALUDE_angie_pretzels_l641_64156


namespace NUMINAMATH_CALUDE_fraction_equivalences_l641_64175

theorem fraction_equivalences : 
  ∃ (n : ℕ) (p : ℕ) (d : ℚ),
    (n : ℚ) / 15 = 4 / 5 ∧
    (4 : ℚ) / 5 = p / 100 ∧
    (4 : ℚ) / 5 = d ∧
    d = 0.8 ∧
    p = 80 :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalences_l641_64175


namespace NUMINAMATH_CALUDE_cloth_cost_price_per_meter_l641_64103

/-- Given a cloth sale scenario, prove the cost price per meter. -/
theorem cloth_cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 66)
  (h2 : total_selling_price = 660)
  (h3 : profit_per_meter = 5) :
  (total_selling_price - total_length * profit_per_meter) / total_length = 5 :=
by sorry

end NUMINAMATH_CALUDE_cloth_cost_price_per_meter_l641_64103


namespace NUMINAMATH_CALUDE_continuous_function_property_l641_64162

theorem continuous_function_property (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_prop : ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_property_l641_64162


namespace NUMINAMATH_CALUDE_seventh_term_is_4374_l641_64188

/-- A geometric sequence of positive integers with first term 6 and fifth term 486 -/
def GeometricSequence : ℕ → ℕ :=
  fun n => 6 * (486 / 6) ^ ((n - 1) / 4)

/-- The seventh term of the geometric sequence is 4374 -/
theorem seventh_term_is_4374 : GeometricSequence 7 = 4374 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_4374_l641_64188


namespace NUMINAMATH_CALUDE_irrational_shift_exists_rational_shift_not_exists_l641_64150

variable {n : ℕ}
variable (a : Fin n → ℝ)

theorem irrational_shift_exists :
  ∃ (α : ℝ), ∀ (i : Fin n), ¬(∃ (p q : ℤ), a i + α = p / q ∧ q ≠ 0) :=
sorry

theorem rational_shift_not_exists :
  ¬(∃ (α : ℝ), ∀ (i : Fin n), ∃ (p q : ℤ), a i + α = p / q ∧ q ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_irrational_shift_exists_rational_shift_not_exists_l641_64150


namespace NUMINAMATH_CALUDE_curve_C_cartesian_equation_l641_64161

/-- Given a curve C in polar coordinates, prove its Cartesian equation --/
theorem curve_C_cartesian_equation (ρ θ : ℝ) (h : ρ = ρ * Real.cos θ + 2) :
  ∃ x y : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ y^2 = 4*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_curve_C_cartesian_equation_l641_64161


namespace NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l641_64160

/-- Any quadratic trinomial can be represented as the sum of two quadratic trinomials with zero discriminants -/
theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ (p q r s t u : ℝ), 
    (∀ x, a * x^2 + b * x + c = (p * x^2 + q * x + r) + (s * x^2 + t * x + u)) ∧
    (q^2 - 4 * p * r = 0) ∧
    (t^2 - 4 * s * u = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l641_64160


namespace NUMINAMATH_CALUDE_even_function_m_value_l641_64195

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^2 + (m + 2)x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l641_64195


namespace NUMINAMATH_CALUDE_M_subset_N_l641_64199

/-- Set M definition -/
def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}

/-- Set N definition -/
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

/-- Theorem stating that M is a subset of N -/
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l641_64199


namespace NUMINAMATH_CALUDE_mundane_goblet_points_difference_l641_64158

def round_robin_tournament (n : ℕ) := n * (n - 1) / 2

theorem mundane_goblet_points_difference :
  let num_teams : ℕ := 6
  let num_matches := round_robin_tournament num_teams
  let max_points := num_matches * 3
  let min_points := num_matches * 2
  max_points - min_points = 15 := by
  sorry

end NUMINAMATH_CALUDE_mundane_goblet_points_difference_l641_64158


namespace NUMINAMATH_CALUDE_remaining_distance_to_grandma_l641_64159

theorem remaining_distance_to_grandma (total_distance : ℕ) 
  (distance1 distance2 distance3 distance4 distance5 distance6 : ℕ) : 
  total_distance = 78 ∧ 
  distance1 = 35 ∧ 
  distance2 = 7 ∧ 
  distance3 = 18 ∧ 
  distance4 = 3 ∧ 
  distance5 = 12 ∧ 
  distance6 = 2 → 
  total_distance - (distance1 + distance2 + distance3 + distance4 + distance5 + distance6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_grandma_l641_64159


namespace NUMINAMATH_CALUDE_age_cube_sum_l641_64185

theorem age_cube_sum (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r) 
  (h2 : j^3 + a^3 = (1/2) * r^3) (h3 : j + a + r = 50) : 
  j^3 + a^3 + r^3 = 24680 := by
  sorry

end NUMINAMATH_CALUDE_age_cube_sum_l641_64185


namespace NUMINAMATH_CALUDE_quadratic_roots_l641_64125

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l641_64125


namespace NUMINAMATH_CALUDE_area_of_second_square_l641_64121

-- Define the circle
def Circle : Type := Unit

-- Define squares
structure Square where
  area : ℝ

-- Define the inscribed square
def inscribed_square (c : Circle) (s : Square) : Prop :=
  s.area = 16

-- Define the second square
def second_square (c : Circle) (s1 s2 : Square) : Prop :=
  -- Vertices E and F are on sides of s1, G and H are on the circle
  True

-- Theorem statement
theorem area_of_second_square 
  (c : Circle) 
  (s1 s2 : Square) 
  (h1 : inscribed_square c s1) 
  (h2 : second_square c s1 s2) : 
  s2.area = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_second_square_l641_64121


namespace NUMINAMATH_CALUDE_not_even_if_unequal_l641_64182

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem not_even_if_unequal :
  f (-2) ≠ f 2 → ¬(IsEven f) := by
  sorry

end NUMINAMATH_CALUDE_not_even_if_unequal_l641_64182


namespace NUMINAMATH_CALUDE_flock_size_l641_64148

/-- Represents the number of sheep in a flock -/
structure Flock :=
  (rams : ℕ)
  (ewes : ℕ)

/-- The ratio of rams to ewes after one ram runs away -/
def ratio_after_ram_leaves (f : Flock) : ℚ :=
  (f.rams - 1 : ℚ) / f.ewes

/-- The ratio of rams to ewes after the ram returns and one ewe runs away -/
def ratio_after_ewe_leaves (f : Flock) : ℚ :=
  (f.rams : ℚ) / (f.ewes - 1)

/-- The theorem stating the total number of sheep in the flock -/
theorem flock_size (f : Flock) :
  (ratio_after_ram_leaves f = 7/5) →
  (ratio_after_ewe_leaves f = 5/3) →
  f.rams + f.ewes = 25 := by
  sorry

end NUMINAMATH_CALUDE_flock_size_l641_64148


namespace NUMINAMATH_CALUDE_scavenger_hunt_items_l641_64138

theorem scavenger_hunt_items (tanya samantha lewis james : ℕ) : 
  tanya = 4 ∧ 
  samantha = 4 * tanya ∧ 
  lewis = samantha + 4 ∧ 
  james = 2 * lewis →
  lewis = 20 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_items_l641_64138


namespace NUMINAMATH_CALUDE_ryan_reads_more_pages_l641_64111

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pages Ryan read -/
def ryan_total_pages : ℕ := 2100

/-- The number of pages Ryan's brother read per day -/
def brother_pages_per_day : ℕ := 200

/-- The difference in average pages read per day between Ryan and his brother -/
def page_difference : ℕ := ryan_total_pages / days_in_week - brother_pages_per_day

theorem ryan_reads_more_pages :
  page_difference = 100 := by sorry

end NUMINAMATH_CALUDE_ryan_reads_more_pages_l641_64111


namespace NUMINAMATH_CALUDE_total_oranges_l641_64124

theorem total_oranges (oranges_per_child : ℕ) (num_children : ℕ) 
  (h1 : oranges_per_child = 3) 
  (h2 : num_children = 4) : 
  oranges_per_child * num_children = 12 :=
by sorry

end NUMINAMATH_CALUDE_total_oranges_l641_64124


namespace NUMINAMATH_CALUDE_cupcakes_per_tray_l641_64179

/-- Proves that the number of cupcakes on each tray is 20 given the problem conditions -/
theorem cupcakes_per_tray (
  num_trays : ℕ)
  (price_per_cupcake : ℚ)
  (sold_fraction : ℚ)
  (total_earnings : ℚ)
  (h1 : num_trays = 4)
  (h2 : price_per_cupcake = 2)
  (h3 : sold_fraction = 3 / 5)
  (h4 : total_earnings = 96) :
  ∃ (cupcakes_per_tray : ℕ), 
    cupcakes_per_tray = 20 ∧
    (↑num_trays * ↑cupcakes_per_tray : ℚ) * sold_fraction * price_per_cupcake = total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_tray_l641_64179


namespace NUMINAMATH_CALUDE_twenty_fifth_digit_sum_l641_64198

/-- The decimal representation of 1/9 -/
def decimal_1_9 : ℚ := 1/9

/-- The decimal representation of 1/11 -/
def decimal_1_11 : ℚ := 1/11

/-- The sum of the decimal representations of 1/9 and 1/11 -/
def sum_decimals : ℚ := decimal_1_9 + decimal_1_11

/-- The 25th digit after the decimal point in a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem twenty_fifth_digit_sum :
  nth_digit_after_decimal sum_decimals 25 = 2 := by sorry

end NUMINAMATH_CALUDE_twenty_fifth_digit_sum_l641_64198


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l641_64142

theorem largest_three_digit_congruence :
  ∀ n : ℕ,
  n ≤ 998 →
  100 ≤ n →
  n ≤ 999 →
  70 * n ≡ 210 [MOD 350] →
  ∃ m : ℕ,
  m = 998 ∧
  70 * m ≡ 210 [MOD 350] ∧
  ∀ k : ℕ,
  100 ≤ k →
  k ≤ 999 →
  70 * k ≡ 210 [MOD 350] →
  k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l641_64142


namespace NUMINAMATH_CALUDE_tensor_product_of_A_and_B_l641_64174

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {y : ℝ | y ≥ 0}

-- Define the ⊗ operation
def tensorProduct (X Y : Set ℝ) : Set ℝ := (X ∪ Y) \ (X ∩ Y)

-- Theorem statement
theorem tensor_product_of_A_and_B :
  tensorProduct A B = {x : ℝ | x = 0 ∨ x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_tensor_product_of_A_and_B_l641_64174


namespace NUMINAMATH_CALUDE_pentagon_angle_sequences_l641_64118

def is_valid_sequence (x d : ℕ) : Prop :=
  x > 0 ∧ d > 0 ∧
  x + (x+d) + (x+2*d) + (x+3*d) + (x+4*d) = 540 ∧
  x + 4*d < 120

theorem pentagon_angle_sequences :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ s : Finset (ℕ × ℕ), s.card = n ∧ 
    (∀ p ∈ s, is_valid_sequence p.1 p.2) ∧
    (∀ x d : ℕ, is_valid_sequence x d → (x, d) ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_pentagon_angle_sequences_l641_64118


namespace NUMINAMATH_CALUDE_person_a_work_time_l641_64187

theorem person_a_work_time (b : ℝ) (combined_rate : ℝ) (combined_time : ℝ) 
  (hb : b = 45)
  (hcombined : combined_rate * combined_time = 1 / 9)
  (htime : combined_time = 2) :
  ∃ a : ℝ, a = 30 ∧ combined_rate = 1 / a + 1 / b := by
  sorry

end NUMINAMATH_CALUDE_person_a_work_time_l641_64187


namespace NUMINAMATH_CALUDE_hulk_jump_l641_64131

theorem hulk_jump (n : ℕ) (a : ℕ → ℝ) :
  (∀ k, a k = 3 * (3 ^ (k - 1))) →
  (∀ k < 6, a k ≤ 500) ∧ (a 6 > 500) :=
by sorry

end NUMINAMATH_CALUDE_hulk_jump_l641_64131


namespace NUMINAMATH_CALUDE_intersection_right_isosceles_l641_64101

-- Define the universe set of all triangles
def Triangle : Type := sorry

-- Define the property of being a right triangle
def IsRight (t : Triangle) : Prop := sorry

-- Define the property of being an isosceles triangle
def IsIsosceles (t : Triangle) : Prop := sorry

-- Define the set of right triangles
def RightTriangles : Set Triangle := {t : Triangle | IsRight t}

-- Define the set of isosceles triangles
def IsoscelesTriangles : Set Triangle := {t : Triangle | IsIsosceles t}

-- Define the property of being both right and isosceles
def IsRightAndIsosceles (t : Triangle) : Prop := IsRight t ∧ IsIsosceles t

-- Define the set of isosceles right triangles
def IsoscelesRightTriangles : Set Triangle := {t : Triangle | IsRightAndIsosceles t}

-- Theorem statement
theorem intersection_right_isosceles :
  RightTriangles ∩ IsoscelesTriangles = IsoscelesRightTriangles := by sorry

end NUMINAMATH_CALUDE_intersection_right_isosceles_l641_64101


namespace NUMINAMATH_CALUDE_mitya_visits_l641_64147

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
inductive Month
  | February
  | March

/-- Returns the number of days in a given month of a non-leap year -/
def daysInMonth (m : Month) : Nat :=
  match m with
  | Month.February => 28
  | Month.March => 31

/-- Returns the day of the week for a given date in a month -/
def dayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Returns the date of the first Tuesday in a month -/
def firstTuesday (m : Month) : Nat :=
  sorry

/-- Returns the date of the first Monday in a month -/
def firstMonday (m : Month) : Nat :=
  sorry

/-- Main theorem to prove -/
theorem mitya_visits (febFirstTues : firstTuesday Month.February = 1)
                     (febFirstTuesAfterMon : firstTuesday Month.February + 7 = 8) :
  firstTuesday Month.March = 1 ∧ firstTuesday Month.March + 7 = 8 :=
sorry

end NUMINAMATH_CALUDE_mitya_visits_l641_64147


namespace NUMINAMATH_CALUDE_min_m_value_l641_64151

/-- Given a function f(x) = 2^(|x-a|) where a ∈ ℝ, if f(1+x) = f(1-x) for all x ∈ ℝ 
    and f(x) is monotonically increasing on [m,+∞), then the minimum value of m is 1. -/
theorem min_m_value (a : ℝ) (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x, f x = 2^(|x - a|))
    (h2 : ∀ x, f (1 + x) = f (1 - x))
    (h3 : MonotoneOn f (Set.Ici m)) :
  ∃ m₀ : ℝ, m₀ = 1 ∧ ∀ m' : ℝ, (∀ x ≥ m', MonotoneOn f (Set.Ici x)) → m' ≥ m₀ :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l641_64151


namespace NUMINAMATH_CALUDE_painting_time_equation_l641_64171

theorem painting_time_equation (doug_time dave_time t : ℝ) :
  doug_time = 6 →
  dave_time = 8 →
  (1 / doug_time + 1 / dave_time) * (t - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l641_64171


namespace NUMINAMATH_CALUDE_intersection_implies_equality_l641_64129

theorem intersection_implies_equality (k b a c : ℝ) : 
  k ≠ b → 
  (∃! p : ℝ × ℝ, (p.2 = k * p.1 + k) ∧ (p.2 = b * p.1 + b) ∧ (p.2 = a * p.1 + c)) →
  a = c := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_equality_l641_64129


namespace NUMINAMATH_CALUDE_unique_integer_solution_l641_64116

theorem unique_integer_solution :
  ∃! (x : ℤ), x - 8 / (x - 2) = 5 - 8 / (x - 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l641_64116


namespace NUMINAMATH_CALUDE_f_derivative_condition_implies_a_range_g_minimum_value_l641_64164

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + ((a-3)/2) * x^2 + (a^2-3*a) * x - 2*a

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-3)*x + a^2 - 3*a

-- Define the function g
def g (a x₁ x₂ : ℝ) : ℝ := x₁^3 + x₂^3 + a^3

theorem f_derivative_condition_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f_derivative a x > a^2) → a ∈ Set.Ioi (-2) :=
sorry

theorem g_minimum_value (a x₁ x₂ : ℝ) :
  a ∈ Set.Ioo (-1) 3 →
  x₁ ≠ x₂ →
  f_derivative a x₁ = 0 →
  f_derivative a x₂ = 0 →
  g a x₁ x₂ ≥ 15 :=
sorry

end

end NUMINAMATH_CALUDE_f_derivative_condition_implies_a_range_g_minimum_value_l641_64164


namespace NUMINAMATH_CALUDE_simplify_expression_l641_64176

theorem simplify_expression : 0.2 * 0.4 + 0.6 * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l641_64176


namespace NUMINAMATH_CALUDE_cody_lost_tickets_l641_64169

theorem cody_lost_tickets (initial : Real) (spent : Real) (left : Real) : 
  initial = 49.0 → spent = 25.0 → left = 18 → initial - spent - left = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_cody_lost_tickets_l641_64169


namespace NUMINAMATH_CALUDE_max_pieces_of_pie_l641_64107

def is_valid_assignment (p k u s o i r g : ℕ) : Prop :=
  p ≠ k ∧ p ≠ u ∧ p ≠ s ∧ p ≠ o ∧ p ≠ i ∧ p ≠ r ∧ p ≠ g ∧
  k ≠ u ∧ k ≠ s ∧ k ≠ o ∧ k ≠ i ∧ k ≠ r ∧ k ≠ g ∧
  u ≠ s ∧ u ≠ o ∧ u ≠ i ∧ u ≠ r ∧ u ≠ g ∧
  s ≠ o ∧ s ≠ i ∧ s ≠ r ∧ s ≠ g ∧
  o ≠ i ∧ o ≠ r ∧ o ≠ g ∧
  i ≠ r ∧ i ≠ g ∧
  r ≠ g ∧
  p ≠ 0 ∧ k ≠ 0

def pirog (p i r o g : ℕ) : ℕ := p * 10000 + i * 1000 + r * 100 + o * 10 + g

def kusok (k u s o k : ℕ) : ℕ := k * 10000 + u * 1000 + s * 100 + o * 10 + k

theorem max_pieces_of_pie :
  ∀ p i r o g k u s n,
    is_valid_assignment p k u s o i r g →
    pirog p i r o g = n * kusok k u s o k →
    n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_of_pie_l641_64107


namespace NUMINAMATH_CALUDE_oil_cylinder_capacity_l641_64141

theorem oil_cylinder_capacity (C : ℚ) 
  (h1 : (4/5 : ℚ) * C - (3/4 : ℚ) * C = 4) : C = 80 := by
  sorry

end NUMINAMATH_CALUDE_oil_cylinder_capacity_l641_64141


namespace NUMINAMATH_CALUDE_plant_supplier_remaining_money_l641_64149

/-- Calculates the remaining money for a plant supplier after sales and expenses. -/
theorem plant_supplier_remaining_money
  (orchid_price : ℕ)
  (orchid_quantity : ℕ)
  (money_plant_price : ℕ)
  (money_plant_quantity : ℕ)
  (worker_pay : ℕ)
  (worker_count : ℕ)
  (pot_cost : ℕ)
  (h1 : orchid_price = 50)
  (h2 : orchid_quantity = 20)
  (h3 : money_plant_price = 25)
  (h4 : money_plant_quantity = 15)
  (h5 : worker_pay = 40)
  (h6 : worker_count = 2)
  (h7 : pot_cost = 150) :
  (orchid_price * orchid_quantity + money_plant_price * money_plant_quantity) -
  (worker_pay * worker_count + pot_cost) = 1145 := by
  sorry

end NUMINAMATH_CALUDE_plant_supplier_remaining_money_l641_64149


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l641_64197

theorem walnut_trees_planted (initial_trees final_trees : ℕ) 
  (h1 : initial_trees = 22)
  (h2 : final_trees = 55) :
  final_trees - initial_trees = 33 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l641_64197


namespace NUMINAMATH_CALUDE_new_year_markup_l641_64119

theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.20 →
  february_discount = 0.12 →
  final_profit = 0.32 →
  ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - february_discount) = 1 + final_profit ∧
    new_year_markup = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_new_year_markup_l641_64119


namespace NUMINAMATH_CALUDE_part_one_part_two_l641_64128

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Part I
theorem part_one (a : ℝ) :
  (∀ x : ℝ, f a x + f a (x - 2) ≥ 1) → (a ≥ 1/2 ∨ a ≤ -1/2) :=
sorry

-- Part II
theorem part_two (a b c : ℝ) :
  f a ((a - 1) / a) + f a ((b - 1) / a) + f a ((c - 1) / a) = 4 →
  (f a ((a^2 - 1) / a) + f a ((b^2 - 1) / a) + f a ((c^2 - 1) / a) ≥ 16/3 ∧
   ∃ x y z : ℝ, f a ((x^2 - 1) / a) + f a ((y^2 - 1) / a) + f a ((z^2 - 1) / a) = 16/3) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l641_64128


namespace NUMINAMATH_CALUDE_walking_time_l641_64167

/-- Given a walking speed of 10 km/hr and a distance of 4 km, the time taken is 24 minutes. -/
theorem walking_time (speed : ℝ) (distance : ℝ) : 
  speed = 10 → distance = 4 → (distance / speed) * 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_l641_64167


namespace NUMINAMATH_CALUDE_total_beads_l641_64132

/-- Given the number of necklaces and beads per necklace, calculate the total number of beads --/
theorem total_beads (necklaces : ℕ) (beads_per_necklace : ℕ) :
  necklaces * beads_per_necklace = necklaces * beads_per_necklace := by
  sorry

end NUMINAMATH_CALUDE_total_beads_l641_64132


namespace NUMINAMATH_CALUDE_cone_volume_l641_64194

/-- Given a cone with slant height 15 cm and height 9 cm, its volume is 432π cubic centimeters -/
theorem cone_volume (s h r : ℝ) (hs : s = 15) (hh : h = 9) 
  (hr : r^2 = s^2 - h^2) : (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l641_64194


namespace NUMINAMATH_CALUDE_x_range_theorem_l641_64127

theorem x_range_theorem (x : ℝ) :
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) →
  x < 1 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l641_64127


namespace NUMINAMATH_CALUDE_coefficients_of_equation_l641_64189

/-- Given a quadratic equation ax² + bx + c = 0, this function returns its coefficients (a, b, c) -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The coefficients of the quadratic equation 4x² - 6x + 1 = 0 are (4, -6, 1) -/
theorem coefficients_of_equation : quadratic_coefficients 4 (-6) 1 = (4, -6, 1) := by sorry

end NUMINAMATH_CALUDE_coefficients_of_equation_l641_64189


namespace NUMINAMATH_CALUDE_probability_4_club_2_is_1_663_l641_64170

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of 4s in a standard deck -/
def NumberOf4s : ℕ := 4

/-- Number of clubs in a standard deck -/
def NumberOfClubs : ℕ := 13

/-- Number of 2s in a standard deck -/
def NumberOf2s : ℕ := 4

/-- Probability of drawing a 4 as the first card, a club as the second card, 
    and a 2 as the third card from a standard 52-card deck -/
def probability_4_club_2 : ℚ :=
  (NumberOf4s : ℚ) / StandardDeck *
  NumberOfClubs / (StandardDeck - 1) *
  NumberOf2s / (StandardDeck - 2)

theorem probability_4_club_2_is_1_663 : 
  probability_4_club_2 = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_4_club_2_is_1_663_l641_64170


namespace NUMINAMATH_CALUDE_berry_exchange_theorem_l641_64117

/-- The number of blueberries in each blue box -/
def B : ℕ := 35

/-- The number of strawberries in each red box -/
def S : ℕ := 100 + B

/-- The change in total berries when exchanging one blue box for one red box -/
def ΔT : ℤ := S - B

theorem berry_exchange_theorem : ΔT = 65 := by
  sorry

end NUMINAMATH_CALUDE_berry_exchange_theorem_l641_64117


namespace NUMINAMATH_CALUDE_chris_money_left_l641_64109

/-- Calculates the money left over after purchases given the following conditions:
  * Video game cost: $60
  * Candy cost: $5
  * Babysitting pay rate: $8 per hour
  * Hours worked: 9
-/
def money_left_over (video_game_cost : ℕ) (candy_cost : ℕ) (pay_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  pay_rate * hours_worked - (video_game_cost + candy_cost)

theorem chris_money_left : money_left_over 60 5 8 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_left_l641_64109


namespace NUMINAMATH_CALUDE_no_three_distinct_rational_roots_l641_64144

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ (∃ (u v w : ℚ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
    (u^3 + (2*a+1)*u^2 + (2*a^2+2*a-3)*u + b = 0) ∧
    (v^3 + (2*a+1)*v^2 + (2*a^2+2*a-3)*v + b = 0) ∧
    (w^3 + (2*a+1)*w^2 + (2*a^2+2*a-3)*w + b = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_three_distinct_rational_roots_l641_64144


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l641_64192

/-- Given two adjacent vertices of a rectangle at (-3, 2) and (1, -6),
    with the third vertex forming a right angle at (-3, 2) and
    the fourth vertex aligning vertically with (-3, 2),
    prove that the area of the rectangle is 32√5. -/
theorem rectangle_area : ℝ → Prop :=
  fun area =>
    let v1 : ℝ × ℝ := (-3, 2)
    let v2 : ℝ × ℝ := (1, -6)
    let v3 : ℝ × ℝ := (-3, 2 - Real.sqrt 80)
    let v4 : ℝ × ℝ := (-3, -6)
    (v1.1 = v3.1 ∧ v1.1 = v4.1) →  -- fourth vertex aligns vertically with (-3, 2)
    (v1.2 - v3.2)^2 + (v1.1 - v2.1)^2 = (v1.2 - v2.2)^2 →  -- right angle at (-3, 2)
    area = 32 * Real.sqrt 5

-- The proof of this theorem
theorem rectangle_area_proof : rectangle_area (32 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l641_64192


namespace NUMINAMATH_CALUDE_max_digit_sum_three_digit_number_l641_64191

theorem max_digit_sum_three_digit_number (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * b + c) + (100 * a + 10 * c + b) = 1732 →
  a + b + c ≤ 20 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_three_digit_number_l641_64191


namespace NUMINAMATH_CALUDE_normal_distribution_problem_l641_64183

theorem normal_distribution_problem (σ μ : ℝ) (h1 : σ = 2) (h2 : μ = 55) :
  ∃ k : ℕ, k = 3 ∧ μ - k * σ > 48 ∧ ∀ m : ℕ, m > k → μ - m * σ ≤ 48 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_problem_l641_64183


namespace NUMINAMATH_CALUDE_no_solution_floor_equation_l641_64145

theorem no_solution_floor_equation :
  ¬ ∃ (x : ℤ), (⌊x⌋ : ℤ) + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ = 12345 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_floor_equation_l641_64145


namespace NUMINAMATH_CALUDE_meter_to_skips_l641_64180

theorem meter_to_skips 
  (b c d e f g : ℝ) 
  (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0)
  (hop_skip : b * 1 = c * 1)  -- b hops = c skips
  (jump_hop : d * 1 = e * 1)  -- d jumps = e hops
  (jump_meter : f * 1 = g * 1)  -- f jumps = g meters
  : 1 = (c * e * f) / (b * d * g) := by
  sorry

end NUMINAMATH_CALUDE_meter_to_skips_l641_64180


namespace NUMINAMATH_CALUDE_transistor_count_2002_l641_64134

def moores_law (initial_year final_year : ℕ) (initial_transistors : ℕ) : ℕ :=
  initial_transistors * 2^((final_year - initial_year) / 2)

theorem transistor_count_2002 :
  moores_law 1988 2002 500000 = 64000000 := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_2002_l641_64134


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l641_64166

theorem min_value_theorem (x : ℝ) (h : x > 0) : 4 * x + 1 / x^4 ≥ 5 := by
  sorry

theorem equality_condition : 4 * 1 + 1 / 1^4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l641_64166


namespace NUMINAMATH_CALUDE_tangent_line_parallelism_l641_64120

theorem tangent_line_parallelism (a : ℝ) :
  a > -2 * Real.sqrt 2 →
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ 2 * x₁ + 1 / x₁ = Real.exp x₂ - a :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallelism_l641_64120


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l641_64165

-- Define p and q
def p (x : ℝ) : Prop := x^2 > 4
def q (x : ℝ) : Prop := x ≤ 2

-- Define the negation of p
def not_p (x : ℝ) : Prop := ¬(p x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, not_p x → q x) ∧ 
  (∃ x : ℝ, q x ∧ ¬(not_p x)) :=
by sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l641_64165


namespace NUMINAMATH_CALUDE_plane_hit_probability_l641_64136

theorem plane_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.5) :
  1 - (1 - p_A) * (1 - p_B) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_plane_hit_probability_l641_64136


namespace NUMINAMATH_CALUDE_basketball_score_calculation_l641_64181

/-- Given a basketball player who made 7 two-point shots and 3 three-point shots,
    the total points scored is equal to 23. -/
theorem basketball_score_calculation (two_point_shots three_point_shots : ℕ) : 
  two_point_shots = 7 →
  three_point_shots = 3 →
  2 * two_point_shots + 3 * three_point_shots = 23 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_calculation_l641_64181


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l641_64196

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_bottles (total_bottles regular_soda_bottles : ℕ) 
  (h1 : total_bottles = 38) 
  (h2 : regular_soda_bottles = 30) : 
  total_bottles - regular_soda_bottles = 8 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l641_64196


namespace NUMINAMATH_CALUDE_eagles_falcons_games_l641_64100

theorem eagles_falcons_games (N : ℕ) : 
  (∀ n : ℕ, n < N → (3 + n : ℚ) / (7 + n) < 9/10) ∧ 
  (3 + N : ℚ) / (7 + N) ≥ 9/10 → 
  N = 33 :=
sorry

end NUMINAMATH_CALUDE_eagles_falcons_games_l641_64100


namespace NUMINAMATH_CALUDE_max_container_weight_for_transport_l641_64152

/-- Represents a container with a weight in tons -/
structure Container where
  weight : ℕ

/-- Represents a platform with a maximum load capacity -/
structure Platform where
  capacity : ℕ

/-- Represents a train with a number of platforms -/
structure Train where
  platforms : List Platform

/-- Checks if a given configuration of containers can be loaded onto a train -/
def canLoad (containers : List Container) (train : Train) : Prop :=
  sorry

/-- The main theorem stating that 26 is the maximum container weight that guarantees
    1500 tons can be transported -/
theorem max_container_weight_for_transport
  (total_weight : ℕ)
  (num_platforms : ℕ)
  (platform_capacity : ℕ)
  (h_total_weight : total_weight = 1500)
  (h_num_platforms : num_platforms = 25)
  (h_platform_capacity : platform_capacity = 80)
  : (∃ k : ℕ, k = 26 ∧
    (∀ containers : List Container,
      (∀ c ∈ containers, c.weight ≤ k ∧ c.weight > 0) →
      (containers.map (λ c => c.weight)).sum = total_weight →
      canLoad containers (Train.mk (List.replicate num_platforms (Platform.mk platform_capacity)))) ∧
    (∀ k' > k, ∃ containers : List Container,
      (∀ c ∈ containers, c.weight ≤ k' ∧ c.weight > 0) ∧
      (containers.map (λ c => c.weight)).sum = total_weight ∧
      ¬canLoad containers (Train.mk (List.replicate num_platforms (Platform.mk platform_capacity))))) :=
  sorry

end NUMINAMATH_CALUDE_max_container_weight_for_transport_l641_64152


namespace NUMINAMATH_CALUDE_rectangle_area_l641_64154

/-- Given a rectangle ABCD divided into four identical squares with side length s,
    prove that its area is 2500 square centimeters when three of its sides total 100 cm. -/
theorem rectangle_area (s : ℝ) : 
  s > 0 →                            -- s is positive (implied by the context)
  4 * s = 100 →                      -- three sides total 100 cm
  (2 * s) * (2 * s) = 2500 :=        -- area of ABCD is 2500 sq cm
by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l641_64154


namespace NUMINAMATH_CALUDE_isosceles_triangle_on_cube_l641_64184

-- Define a cube
def Cube : Type := Unit

-- Define a function to count the number of ways to choose 3 vertices from 8
def choose_3_from_8 : ℕ := 56

-- Define the number of isosceles triangles that can be formed on the cube
def isosceles_triangles_count : ℕ := 32

-- Define the probability of forming an isosceles triangle
def isosceles_triangle_probability : ℚ := 4/7

-- Theorem statement
theorem isosceles_triangle_on_cube :
  (isosceles_triangles_count : ℚ) / choose_3_from_8 = isosceles_triangle_probability :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_on_cube_l641_64184


namespace NUMINAMATH_CALUDE_cosine_product_bounds_l641_64163

theorem cosine_product_bounds : 
  1/8 < Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (70 * π / 180) ∧ 
  Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (70 * π / 180) < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_product_bounds_l641_64163


namespace NUMINAMATH_CALUDE_west_is_negative_of_east_l641_64115

/-- Represents distance and direction, where positive values indicate east and negative values indicate west. -/
def Distance := ℤ

/-- Converts a distance in kilometers to the corresponding Distance representation. -/
def km_to_distance (x : ℤ) : Distance := x

/-- The distance representation for 2km east. -/
def two_km_east : Distance := km_to_distance 2

/-- The distance representation for 1km west. -/
def one_km_west : Distance := km_to_distance (-1)

theorem west_is_negative_of_east (h : two_km_east = km_to_distance 2) :
  one_km_west = km_to_distance (-1) := by sorry

end NUMINAMATH_CALUDE_west_is_negative_of_east_l641_64115


namespace NUMINAMATH_CALUDE_independent_events_probability_l641_64155

theorem independent_events_probability (a b : Set ℝ) (p : Set ℝ → ℝ) 
  (h1 : p a = 4/5)
  (h2 : p b = 2/5)
  (h3 : p (a ∩ b) = 0.32)
  (h4 : p (a ∩ b) = p a * p b) : 
  p b = 2/5 := by
sorry

end NUMINAMATH_CALUDE_independent_events_probability_l641_64155


namespace NUMINAMATH_CALUDE_complex_product_equals_five_l641_64143

theorem complex_product_equals_five (a : ℝ) : 
  let z₁ : ℂ := -1 + 2*I
  let z₂ : ℂ := a - 2*I
  z₁ * z₂ = 5 → a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_equals_five_l641_64143


namespace NUMINAMATH_CALUDE_expression_equality_l641_64157

theorem expression_equality (x y z u a b c d : ℝ) :
  (a*x + b*y + c*z + d*u)^2 + (b*x + c*y + d*z + a*u)^2 + 
  (c*x + d*y + a*z + b*u)^2 + (d*x + a*y + b*z + c*u)^2 =
  (d*x + c*y + b*z + a*u)^2 + (c*x + b*y + a*z + d*u)^2 + 
  (b*x + a*y + d*z + c*u)^2 + (a*x + d*y + c*z + b*u)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l641_64157


namespace NUMINAMATH_CALUDE_candy_probability_difference_l641_64130

theorem candy_probability_difference : 
  let total_candies : ℕ := 2004
  let banana_candies : ℕ := 1002
  let apple_candies : ℕ := 1002
  let different_flavor_prob : ℚ := banana_candies * apple_candies / (total_candies * (total_candies - 1))
  let same_flavor_prob : ℚ := (banana_candies * (banana_candies - 1) + apple_candies * (apple_candies - 1)) / (total_candies * (total_candies - 1))
  different_flavor_prob - same_flavor_prob = 1 / 2003 := by
sorry

end NUMINAMATH_CALUDE_candy_probability_difference_l641_64130


namespace NUMINAMATH_CALUDE_initial_students_count_l641_64133

/-- The number of students initially on the bus -/
def initial_students : ℕ := sorry

/-- The number of students who got on at the first stop -/
def students_who_got_on : ℕ := 3

/-- The total number of students on the bus after the first stop -/
def total_students : ℕ := 13

/-- Theorem stating that the initial number of students was 10 -/
theorem initial_students_count : initial_students = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_students_count_l641_64133


namespace NUMINAMATH_CALUDE_reading_time_calculation_l641_64105

/-- Calculates the time needed to read a book given the reading speed and book properties -/
theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : 
  reading_speed = 200 →
  paragraphs_per_page = 20 →
  sentences_per_paragraph = 10 →
  total_pages = 50 →
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed = 50 := by
  sorry

#check reading_time_calculation

end NUMINAMATH_CALUDE_reading_time_calculation_l641_64105


namespace NUMINAMATH_CALUDE_hyperbola_equation_l641_64193

-- Define the hyperbola
def Hyperbola (x y : ℝ) := y^2 - x^2/2 = 1

-- Define the asymptotes
def Asymptotes (x y : ℝ) := (x + Real.sqrt 2 * y = 0) ∨ (x - Real.sqrt 2 * y = 0)

theorem hyperbola_equation :
  ∀ (x y : ℝ),
  Asymptotes x y →
  Hyperbola (-2) (Real.sqrt 3) →
  Hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l641_64193
