import Mathlib

namespace floor_of_e_l847_84748

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_of_e_l847_84748


namespace boat_distance_theorem_l847_84745

/-- Proves that a boat traveling downstream in 2 hours and upstream in 3 hours,
    with a speed of 5 km/h in still water, covers a distance of 12 km. -/
theorem boat_distance_theorem (boat_speed : ℝ) (downstream_time upstream_time : ℝ) :
  boat_speed = 5 ∧ downstream_time = 2 ∧ upstream_time = 3 →
  ∃ (stream_speed : ℝ),
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time ∧
    (boat_speed + stream_speed) * downstream_time = 12 := by
  sorry

end boat_distance_theorem_l847_84745


namespace intersection_integer_points_l847_84708

theorem intersection_integer_points (k : ℝ) : 
  (∃! (n : ℕ), n = 3 ∧ 
    (∀ (x y : ℤ), 
      (y = 4*k*x - 1/k ∧ y = (1/k)*x + 2) → 
      (∃ (k₁ k₂ k₃ : ℝ), k = k₁ ∨ k = k₂ ∨ k = k₃))) :=
by sorry

end intersection_integer_points_l847_84708


namespace exists_equal_face_products_l847_84773

/-- A cube arrangement is a function from the set of 12 edges to the set of numbers 1 to 12 -/
def CubeArrangement := Fin 12 → Fin 12

/-- The set of edges on the top face of the cube -/
def topFace : Finset (Fin 12) := {0, 1, 2, 3}

/-- The set of edges on the bottom face of the cube -/
def bottomFace : Finset (Fin 12) := {4, 5, 6, 7}

/-- The product of numbers on a given face for a given arrangement -/
def faceProduct (arrangement : CubeArrangement) (face : Finset (Fin 12)) : ℕ :=
  face.prod (fun edge => (arrangement edge).val + 1)

/-- Theorem stating that there exists a cube arrangement where the product of
    numbers on the top face equals the product of numbers on the bottom face -/
theorem exists_equal_face_products : ∃ (arrangement : CubeArrangement),
  faceProduct arrangement topFace = faceProduct arrangement bottomFace := by
  sorry

end exists_equal_face_products_l847_84773


namespace g_inverse_composition_l847_84704

def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 1
| 4 => 5
| 5 => 2

theorem g_inverse_composition (h : Function.Bijective g) :
  (Function.invFun g (Function.invFun g (Function.invFun g 3))) = 4 := by
  sorry

end g_inverse_composition_l847_84704


namespace joshua_bottle_caps_l847_84759

theorem joshua_bottle_caps (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 40 → bought = 7 → total = initial + bought → total = 47 := by
  sorry

end joshua_bottle_caps_l847_84759


namespace season_win_percentage_l847_84791

/-- 
Given a team that:
- Won 70 percent of its first 100 games
- Played a total of 100 games

Prove that the percentage of games won for the entire season is 70%.
-/
theorem season_win_percentage 
  (total_games : ℕ) 
  (first_100_win_percentage : ℚ) 
  (h1 : total_games = 100)
  (h2 : first_100_win_percentage = 70/100) : 
  first_100_win_percentage = 70/100 := by
sorry

end season_win_percentage_l847_84791


namespace nero_speed_l847_84717

/-- Given a trail that takes Jerome 6 hours to run at 4 MPH, and Nero 3 hours to run,
    prove that Nero's speed is 8 MPH. -/
theorem nero_speed (jerome_time : ℝ) (nero_time : ℝ) (jerome_speed : ℝ) :
  jerome_time = 6 →
  nero_time = 3 →
  jerome_speed = 4 →
  jerome_time * jerome_speed = nero_time * (jerome_time * jerome_speed / nero_time) :=
by sorry

end nero_speed_l847_84717


namespace tangent_line_and_monotonicity_l847_84770

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*a*x^2

theorem tangent_line_and_monotonicity :
  -- Part I: Tangent line equation
  (∀ x y : ℝ, y = f (-1) x → (x = 1 ∧ y = 7) →
    ∃ k m : ℝ, k = 15 ∧ m = -8 ∧ k*x + (-1)*y + m = 0) ∧
  -- Part II: Monotonicity
  (∀ a : ℝ,
    -- Case a = 0
    (a = 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
    -- Case a < 0
    (a < 0 → ∀ x₁ x₂ : ℝ,
      ((x₁ < x₂ ∧ x₂ < 4*a) ∨ (x₁ < x₂ ∧ 0 < x₁)) → f a x₁ < f a x₂) ∧
    (a < 0 → ∀ x₁ x₂ : ℝ, (4*a < x₁ ∧ x₁ < x₂ ∧ x₂ < 0) → f a x₁ > f a x₂) ∧
    -- Case a > 0
    (a > 0 → ∀ x₁ x₂ : ℝ,
      ((x₁ < x₂ ∧ x₂ < 0) ∨ (4*a < x₁ ∧ x₁ < x₂)) → f a x₁ < f a x₂) ∧
    (a > 0 → ∀ x₁ x₂ : ℝ, (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 4*a) → f a x₁ > f a x₂)) :=
by sorry

end tangent_line_and_monotonicity_l847_84770


namespace jessica_age_when_justin_born_jessica_age_proof_l847_84732

theorem jessica_age_when_justin_born (justin_current_age : ℕ) (james_jessica_age_diff : ℕ) (james_future_age : ℕ) (years_to_future : ℕ) : ℕ :=
  let james_current_age := james_future_age - years_to_future
  let jessica_current_age := james_current_age - james_jessica_age_diff
  jessica_current_age - justin_current_age

/- Proof that Jessica was 6 years old when Justin was born -/
theorem jessica_age_proof :
  jessica_age_when_justin_born 26 7 44 5 = 6 := by
sorry

end jessica_age_when_justin_born_jessica_age_proof_l847_84732


namespace sequence_sum_l847_84783

/-- Given a sequence defined by a₁ + b₁ = 1, a² + b² = 3, a³ + b³ = 4, a⁴ + b⁴ = 7, a⁵ + b⁵ = 11,
    and for n ≥ 3, aⁿ + bⁿ = (aⁿ⁻¹ + bⁿ⁻¹) + (aⁿ⁻² + bⁿ⁻²),
    prove that a¹¹ + b¹¹ = 199 -/
theorem sequence_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^11 + b^11 = 199 := by
  sorry


end sequence_sum_l847_84783


namespace math_competition_prizes_l847_84713

theorem math_competition_prizes (x y s : ℝ) 
  (h1 : 100 * (x + 3 * y) = s)
  (h2 : 80 * (x + 5 * y) = s) :
  x = 5 * y ∧ s = 160 * x ∧ s = 800 * y := by
  sorry

end math_competition_prizes_l847_84713


namespace subset_implies_a_values_l847_84761

def M : Set ℝ := {x | x^2 + 6*x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x | x*a - 3 = 0}

theorem subset_implies_a_values (h : N a ⊆ M) : a = -3/8 ∨ a = 0 ∨ a = 3/2 := by
  sorry

end subset_implies_a_values_l847_84761


namespace tangent_sum_simplification_l847_84765

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.sin (40 * π / 180) =
  (Real.sin (50 * π / 180) * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.cos (20 * π / 180) * Real.cos (30 * π / 180))) /
  (Real.sin (40 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end tangent_sum_simplification_l847_84765


namespace coat_price_calculation_l847_84778

/-- Calculates the final price of a coat after discounts, coupons, rebates, and tax -/
def finalPrice (initialPrice : ℝ) (discountRate : ℝ) (couponValue : ℝ) (rebateValue : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := initialPrice * (1 - discountRate)
  let afterCoupon := discountedPrice - couponValue
  let afterRebate := afterCoupon - rebateValue
  afterRebate * (1 + taxRate)

/-- Theorem stating that the final price of the coat is $72.45 -/
theorem coat_price_calculation :
  finalPrice 120 0.30 10 5 0.05 = 72.45 := by
  sorry

#eval finalPrice 120 0.30 10 5 0.05

end coat_price_calculation_l847_84778


namespace association_properties_l847_84775

/-- A function f is associated with a set S if for any x₂-x₁ ∈ S, f(x₂)-f(x₁) ∈ S -/
def associated (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₂ - x₁ ∈ S → f x₂ - f x₁ ∈ S

theorem association_properties :
  let f₁ : ℝ → ℝ := λ x ↦ 2*x - 1
  let f₂ : ℝ → ℝ := λ x ↦ if x < 3 then x^2 - 2*x else x^2 - 2*x + 3
  (associated f₁ (Set.Ici 0) ∧ ¬ associated f₁ (Set.Icc 0 1)) ∧
  (associated f₂ {3} → Set.Icc (Real.sqrt 3 + 1) 5 = {x | 2 ≤ f₂ x ∧ f₂ x ≤ 3}) ∧
  (∀ f : ℝ → ℝ, (associated f {1} ∧ associated f (Set.Ici 0)) ↔ associated f (Set.Icc 1 2)) :=
by sorry

#check association_properties

end association_properties_l847_84775


namespace instantaneous_velocity_at_10_l847_84798

/-- The displacement function of a moving object -/
def s (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

/-- The velocity function derived from the displacement function -/
def v (t : ℝ) : ℝ := 6 * t - 2

theorem instantaneous_velocity_at_10 : v 10 = 58 := by
  sorry

end instantaneous_velocity_at_10_l847_84798


namespace water_transfer_problem_l847_84793

theorem water_transfer_problem (left_initial right_initial : ℕ) 
  (difference_after_transfer : ℕ) (h1 : left_initial = 2800) 
  (h2 : right_initial = 1500) (h3 : difference_after_transfer = 360) :
  ∃ (x : ℕ), x = 470 ∧ 
  left_initial - x = right_initial + x + difference_after_transfer :=
sorry

end water_transfer_problem_l847_84793


namespace shaded_rectangle_probability_l847_84722

theorem shaded_rectangle_probability : 
  let width : ℕ := 2004
  let height : ℕ := 2
  let shaded_pos1 : ℕ := 501
  let shaded_pos2 : ℕ := 1504
  let total_rectangles : ℕ := height * (width.choose 2)
  let shaded_rectangles_per_row : ℕ := 
    shaded_pos1 * (width - shaded_pos1 + 1) + 
    (shaded_pos2 - shaded_pos1) * (width - shaded_pos2 + 1)
  let total_shaded_rectangles : ℕ := height * shaded_rectangles_per_row
  (total_rectangles - total_shaded_rectangles : ℚ) / total_rectangles = 1501 / 4008 :=
by
  sorry

end shaded_rectangle_probability_l847_84722


namespace solution_set_min_value_m_plus_n_l847_84715

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x - 2|

-- Part 1: Solution set of f(x) ≥ 3
theorem solution_set (x : ℝ) : f x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 := by sorry

-- Part 2: Minimum value of m+n
theorem min_value_m_plus_n (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x, f x ≥ 1/m + 1/n) : 
  m + n ≥ 8/3 ∧ (m + n = 8/3 ↔ m = n) := by sorry

end solution_set_min_value_m_plus_n_l847_84715


namespace stock_yield_percentage_l847_84776

/-- Calculates the yield percentage of a stock given its dividend rate, par value, and market value. -/
def yield_percentage (dividend_rate : ℚ) (par_value : ℚ) (market_value : ℚ) : ℚ :=
  (dividend_rate * par_value) / market_value

/-- Proves that a 6% stock with a market value of $75 and an assumed par value of $100 has a yield percentage of 8%. -/
theorem stock_yield_percentage :
  let dividend_rate : ℚ := 6 / 100
  let par_value : ℚ := 100
  let market_value : ℚ := 75
  yield_percentage dividend_rate par_value market_value = 8 / 100 := by
sorry

#eval yield_percentage (6/100) 100 75

end stock_yield_percentage_l847_84776


namespace problem_solution_l847_84733

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem problem_solution :
  (∀ m : ℝ, (∀ x : ℝ, f x ≤ -m^2 + 6*m) ↔ (1 ≤ m ∧ m ≤ 5)) ∧
  (∃ m₀ : ℝ, m₀ = 1 ∧ ∀ m : ℝ, (∀ x : ℝ, f x ≤ -m^2 + 6*m) → m₀ ≤ m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 1 →
    a^2 + b^2 + c^2 ≥ 1/50 ∧ ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
      3*a₀ + 4*b₀ + 5*c₀ = 1 ∧ a₀^2 + b₀^2 + c₀^2 = 1/50) := by
  sorry

end problem_solution_l847_84733


namespace bookstore_new_releases_l847_84774

theorem bookstore_new_releases (total_books : ℕ) (total_books_pos : total_books > 0) :
  let historical_fiction := (2 : ℚ) / 5 * total_books
  let other_books := total_books - historical_fiction
  let historical_fiction_new := (2 : ℚ) / 5 * historical_fiction
  let other_new := (1 : ℚ) / 5 * other_books
  let total_new := historical_fiction_new + other_new
  historical_fiction_new / total_new = 4 / 7 :=
by sorry

end bookstore_new_releases_l847_84774


namespace simplify_and_evaluate_l847_84772

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -3) (hy : y = 2) :
  (x + y)^2 - y * (2 * x - y) = 17 := by
  sorry

end simplify_and_evaluate_l847_84772


namespace tens_digit_of_2023_power_2024_plus_2025_l847_84799

theorem tens_digit_of_2023_power_2024_plus_2025 :
  (2023^2024 + 2025) % 100 / 10 = 5 := by
  sorry

end tens_digit_of_2023_power_2024_plus_2025_l847_84799


namespace abs_two_minus_sqrt_three_l847_84781

theorem abs_two_minus_sqrt_three : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 := by
  sorry

end abs_two_minus_sqrt_three_l847_84781


namespace simplify_fraction_l847_84768

theorem simplify_fraction : (160 : ℚ) / 2880 * 40 = 20 / 9 := by
  sorry

end simplify_fraction_l847_84768


namespace tangent_line_condition_no_positive_max_for_negative_integer_a_l847_84750

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * Real.exp x) / x + x

theorem tangent_line_condition (a : ℝ) :
  (∃ (m b : ℝ), m * 1 + b = f a 1 ∧ m * (1 - 0) = f a 1 - (-1) ∧ 0 * m + b = -1) →
  a = -1 / Real.exp 1 := by
  sorry

theorem no_positive_max_for_negative_integer_a :
  ∀ a : ℤ, a < 0 →
  ¬∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a x ≥ f a y ∧ f a x > 0 := by
  sorry

end tangent_line_condition_no_positive_max_for_negative_integer_a_l847_84750


namespace complex_multiplication_l847_84737

theorem complex_multiplication (z : ℂ) (h : z + 1 = 2 + I) : z * (1 - I) = 2 := by
  sorry

end complex_multiplication_l847_84737


namespace at_least_one_greater_than_one_l847_84700

theorem at_least_one_greater_than_one (a b : ℝ) :
  a + b > 2 → max a b > 1 := by
  sorry

end at_least_one_greater_than_one_l847_84700


namespace smallest_configuration_l847_84739

/-- A configuration of points on a plane where each point is 1 unit away from exactly four others -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  distinct : ∀ i j, i ≠ j → points i ≠ points j
  distance_condition : ∀ i, ∃ s : Finset (Fin n), s.card = 4 ∧ 
    ∀ j ∈ s, (i ≠ j) ∧ Real.sqrt (((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2) = 1

/-- The smallest possible number of points in a valid configuration is 9 -/
theorem smallest_configuration : 
  (∃ c : PointConfiguration, c.n = 9) ∧ 
  (∀ c : PointConfiguration, c.n ≥ 9) :=
sorry

end smallest_configuration_l847_84739


namespace new_jasmine_concentration_l847_84735

/-- Calculates the new jasmine concentration after adding pure jasmine and water to a solution -/
theorem new_jasmine_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 80)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 15) :
  let initial_jasmine := initial_volume * initial_concentration
  let new_jasmine := initial_jasmine + added_jasmine
  let new_volume := initial_volume + added_jasmine + added_water
  new_jasmine / new_volume = 0.13 :=
by sorry

end new_jasmine_concentration_l847_84735


namespace circle_intersection_theorem_l847_84738

-- Define a circle with a center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being non-intersecting
def non_intersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≥ (c1.radius + c2.radius)^2

-- Define the property of a circle intersecting another circle
def intersects (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≤ (c1.radius + c2.radius)^2

-- Theorem statement
theorem circle_intersection_theorem 
  (c1 c2 c3 c4 c5 c6 : Circle) 
  (h1 : c1.radius ≥ 1) 
  (h2 : c2.radius ≥ 1) 
  (h3 : c3.radius ≥ 1) 
  (h4 : c4.radius ≥ 1) 
  (h5 : c5.radius ≥ 1) 
  (h6 : c6.radius ≥ 1) 
  (h_non_intersect : 
    non_intersecting c1 c2 ∧ non_intersecting c1 c3 ∧ non_intersecting c1 c4 ∧ 
    non_intersecting c1 c5 ∧ non_intersecting c1 c6 ∧ non_intersecting c2 c3 ∧ 
    non_intersecting c2 c4 ∧ non_intersecting c2 c5 ∧ non_intersecting c2 c6 ∧ 
    non_intersecting c3 c4 ∧ non_intersecting c3 c5 ∧ non_intersecting c3 c6 ∧ 
    non_intersecting c4 c5 ∧ non_intersecting c4 c6 ∧ non_intersecting c5 c6)
  (c : Circle)
  (h_intersect : 
    intersects c c1 ∧ intersects c c2 ∧ intersects c c3 ∧ 
    intersects c c4 ∧ intersects c c5 ∧ intersects c c6) :
  c.radius ≥ 1 := by
  sorry


end circle_intersection_theorem_l847_84738


namespace towel_shrinkage_l847_84706

/-- If a rectangle's breadth decreases by 10% and its area decreases by 28%, then its length decreases by 20%. -/
theorem towel_shrinkage (L B : ℝ) (L' B' : ℝ) (h1 : B' = 0.9 * B) (h2 : L' * B' = 0.72 * L * B) :
  L' = 0.8 * L := by
  sorry

end towel_shrinkage_l847_84706


namespace product_digit_sum_l847_84792

def first_number : ℕ := 141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141
def second_number : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem product_digit_sum :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 := by
  sorry

end product_digit_sum_l847_84792


namespace smallest_divisible_by_12_13_14_l847_84771

theorem smallest_divisible_by_12_13_14 : ∃ n : ℕ, n > 0 ∧ 12 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ ∀ m : ℕ, m > 0 → 12 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by sorry

end smallest_divisible_by_12_13_14_l847_84771


namespace greatest_sum_consecutive_integers_l847_84734

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (∀ k : ℕ, k * (k + 1) < 500 → k ≤ n) → 
  n * (n + 1) < 500 → 
  n + (n + 1) = 43 :=
by sorry

end greatest_sum_consecutive_integers_l847_84734


namespace divisors_of_90_l847_84725

def n : ℕ := 90

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The sum of all positive divisors of n -/
def sum_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem divisors_of_90 :
  num_divisors n = 12 ∧ sum_divisors n = 234 := by sorry

end divisors_of_90_l847_84725


namespace A_investment_is_4410_l847_84786

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  investment_B : ℕ
  investment_C : ℕ
  total_profit : ℕ
  A_profit_share : ℕ

/-- Calculates A's investment given the partnership details --/
def calculate_A_investment (p : Partnership) : ℕ :=
  (p.A_profit_share * (p.investment_B + p.investment_C)) / (p.total_profit - p.A_profit_share)

/-- Theorem stating that A's investment is 4410 given the specified conditions --/
theorem A_investment_is_4410 (p : Partnership) 
  (hB : p.investment_B = 4200)
  (hC : p.investment_C = 10500)
  (hProfit : p.total_profit = 13600)
  (hAShare : p.A_profit_share = 4080) :
  calculate_A_investment p = 4410 := by
  sorry

#eval calculate_A_investment ⟨4200, 10500, 13600, 4080⟩

end A_investment_is_4410_l847_84786


namespace equation_solution_l847_84719

theorem equation_solution (x : ℝ) : (x + 1) ^ (x + 3) = 1 ↔ x = -3 ∨ x = 0 := by
  sorry

end equation_solution_l847_84719


namespace maria_gum_count_l847_84755

/-- The number of gum pieces Maria has after receiving gum from Tommy and Luis -/
def total_gum (initial : ℕ) (from_tommy : ℕ) (from_luis : ℕ) : ℕ :=
  initial + from_tommy + from_luis

/-- Theorem stating that Maria has 61 pieces of gum after receiving gum from Tommy and Luis -/
theorem maria_gum_count : total_gum 25 16 20 = 61 := by
  sorry

end maria_gum_count_l847_84755


namespace fraction_simplification_l847_84766

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end fraction_simplification_l847_84766


namespace circle_equation_l847_84709

theorem circle_equation (x y : ℝ) :
  let center := (3, 4)
  let point := (0, 0)
  let equation := (x - 3)^2 + (y - 4)^2 = 25
  (∀ p, p.1^2 + p.2^2 = (p.1 - center.1)^2 + (p.2 - center.2)^2 → p = point) →
  equation :=
by sorry

end circle_equation_l847_84709


namespace min_value_of_f_l847_84760

def f (x : ℝ) : ℝ := x^4 + 2*x^2 - 1

theorem min_value_of_f :
  ∃ (m : ℝ), m = -1 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ≥ m :=
by sorry

end min_value_of_f_l847_84760


namespace smallest_dual_base_representation_l847_84754

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧
    n = 1 * a + 4 ∧
    n = 2 * b + 3

theorem smallest_dual_base_representation : 
  is_valid_representation 11 ∧ 
  ∀ m : ℕ, m < 11 → ¬(is_valid_representation m) :=
sorry

end smallest_dual_base_representation_l847_84754


namespace interesting_iff_prime_power_l847_84764

def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧
  ∀ x y : ℕ, (Nat.gcd x n ≠ 1 ∧ Nat.gcd y n ≠ 1) → Nat.gcd (x + y) n ≠ 1

theorem interesting_iff_prime_power (n : ℕ) :
  is_interesting n ↔ ∃ (p : ℕ) (s : ℕ), Nat.Prime p ∧ s > 0 ∧ n = p^s :=
sorry

end interesting_iff_prime_power_l847_84764


namespace olivia_supermarket_spending_l847_84782

/-- The amount of money Olivia spent at the supermarket -/
def money_spent (initial_amount : ℕ) (amount_left : ℕ) : ℕ :=
  initial_amount - amount_left

theorem olivia_supermarket_spending :
  money_spent 128 90 = 38 := by
  sorry

end olivia_supermarket_spending_l847_84782


namespace equation_solution_l847_84743

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + x + 1) / (x + 2) = x + 1 ∧ x = -1/2 := by
  sorry

end equation_solution_l847_84743


namespace wire_length_around_square_field_l847_84763

theorem wire_length_around_square_field (field_area : Real) (num_rounds : Nat) : 
  field_area = 24336 ∧ num_rounds = 13 → 
  13 * 4 * Real.sqrt field_area = 8112 := by
  sorry

end wire_length_around_square_field_l847_84763


namespace license_plate_count_l847_84767

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 3

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := 4

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_in_plate) * (num_letters ^ letters_in_plate)

theorem license_plate_count : total_license_plates = 70304000 := by
  sorry

end license_plate_count_l847_84767


namespace product_of_largest_and_smallest_three_digit_l847_84757

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * (if a > b ∧ a > c then max b c else if b > a ∧ b > c then max a c else max a b) +
  min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  100 * (if a > 0 then a else if b > 0 then b else c) +
  10 * (if a > 0 ∧ b > 0 then min a b else if a > 0 ∧ c > 0 then min a c else min b c) +
  (if a = 0 then 0 else if b = 0 then 0 else c)

theorem product_of_largest_and_smallest_three_digit :
  largest_three_digit 6 0 2 * smallest_three_digit 6 0 2 = 127720 := by
  sorry

end product_of_largest_and_smallest_three_digit_l847_84757


namespace total_stickers_l847_84729

def folders : Nat := 3
def sheets_per_folder : Nat := 10
def stickers_red : Nat := 3
def stickers_green : Nat := 2
def stickers_blue : Nat := 1

theorem total_stickers :
  folders * sheets_per_folder * stickers_red +
  folders * sheets_per_folder * stickers_green +
  folders * sheets_per_folder * stickers_blue = 60 := by
  sorry

end total_stickers_l847_84729


namespace book_distribution_l847_84752

theorem book_distribution (x : ℕ) : 
  (∀ (total_books : ℕ), total_books = 9 * x + 7 → 
    (∀ (student : ℕ), student < x → ∃ (books : ℕ), books = 9)) ∧ 
  (∀ (total_books : ℕ), total_books ≤ 11 * x - 1 → 
    (∃ (student : ℕ), student < x ∧ ∀ (books : ℕ), books < 11)) →
  9 * x + 7 < 11 * x := by
sorry

end book_distribution_l847_84752


namespace extended_segment_endpoint_l847_84746

/-- Given a segment AB with endpoints A(2, -2) and B(14, 4), extended through B to point C
    such that BC = 1/3 * AB, prove that the coordinates of point C are (18, 6). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (2, -2) →
  B = (14, 4) →
  C.1 - B.1 = (1/3) * (B.1 - A.1) →
  C.2 - B.2 = (1/3) * (B.2 - A.2) →
  C = (18, 6) := by
  sorry

end extended_segment_endpoint_l847_84746


namespace club_enrollment_l847_84705

/-- Given a club with the following properties:
  * Total members: 85
  * Members enrolled in coding course: 45
  * Members enrolled in design course: 32
  * Members enrolled in both courses: 18
  Prove that the number of members not enrolled in either course is 26. -/
theorem club_enrollment (total : ℕ) (coding : ℕ) (design : ℕ) (both : ℕ)
  (h_total : total = 85)
  (h_coding : coding = 45)
  (h_design : design = 32)
  (h_both : both = 18) :
  total - (coding + design - both) = 26 := by
  sorry

end club_enrollment_l847_84705


namespace digital_earth_data_source_is_high_speed_network_databases_l847_84779

/-- Represents the possible sources of spatial data for the digital Earth -/
inductive SpatialDataSource
  | SatelliteRemoteSensing
  | HighSpeedNetworkDatabases
  | InformationHighway
  | GISExchangeData

/-- Represents the digital Earth -/
structure DigitalEarth where
  mainDataSource : SpatialDataSource

/-- Axiom: The main source of basic spatial data for the digital Earth is from high-speed network databases -/
axiom digital_earth_main_data_source :
  ∀ (de : DigitalEarth), de.mainDataSource = SpatialDataSource.HighSpeedNetworkDatabases

/-- Theorem: The main source of basic spatial data for the digital Earth is from high-speed network databases -/
theorem digital_earth_data_source_is_high_speed_network_databases (de : DigitalEarth) :
  de.mainDataSource = SpatialDataSource.HighSpeedNetworkDatabases :=
by sorry

end digital_earth_data_source_is_high_speed_network_databases_l847_84779


namespace meeting_time_l847_84747

/-- The speed of l in km/hr -/
def speed_l : ℝ := 50

/-- The speed of k in km/hr -/
def speed_k : ℝ := speed_l * 1.5

/-- The time difference between k's and l's start times in hours -/
def time_difference : ℝ := 1

/-- The total distance between k and l in km -/
def total_distance : ℝ := 300

/-- The time when l starts -/
def start_time_l : ℕ := 9

/-- The time when k starts -/
def start_time_k : ℕ := 10

theorem meeting_time :
  let distance_traveled_by_l := speed_l * time_difference
  let remaining_distance := total_distance - distance_traveled_by_l
  let relative_speed := speed_l + speed_k
  let time_to_meet := remaining_distance / relative_speed
  start_time_k + ⌊time_to_meet⌋ = 12 := by sorry

end meeting_time_l847_84747


namespace average_age_combined_l847_84797

/-- The average age of a group of fifth-graders, parents, and teachers -/
theorem average_age_combined (num_fifth_graders : ℕ) (num_parents : ℕ) (num_teachers : ℕ)
  (avg_age_fifth_graders : ℚ) (avg_age_parents : ℚ) (avg_age_teachers : ℚ)
  (h1 : num_fifth_graders = 40)
  (h2 : num_parents = 60)
  (h3 : num_teachers = 10)
  (h4 : avg_age_fifth_graders = 10)
  (h5 : avg_age_parents = 35)
  (h6 : avg_age_teachers = 45) :
  (num_fifth_graders * avg_age_fifth_graders +
   num_parents * avg_age_parents +
   num_teachers * avg_age_teachers) /
  (num_fifth_graders + num_parents + num_teachers : ℚ) = 295 / 11 := by
  sorry

end average_age_combined_l847_84797


namespace contractor_fine_calculation_l847_84731

/-- Calculates the daily fine for absence given the contract parameters -/
def calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let earned_amount := daily_pay * worked_days
  (earned_amount - total_payment) / absent_days

/-- Proves that the daily fine is 7.5 given the contract parameters -/
theorem contractor_fine_calculation :
  calculate_daily_fine 30 25 490 8 = 7.5 := by
  sorry

end contractor_fine_calculation_l847_84731


namespace simplify_and_evaluate_l847_84703

theorem simplify_and_evaluate :
  let x : ℚ := -1/3
  let a : ℤ := -2
  let b : ℤ := -1
  (6*x^2 + 5*x^2 - 2*(3*x - 2*x^2) = 11/3) ∧
  (5*a^2 - a*b - 2*(3*a*b - (a*b - 2*a^2)) = -6) := by
  sorry

end simplify_and_evaluate_l847_84703


namespace absolute_value_equation_solution_l847_84787

theorem absolute_value_equation_solution :
  ∀ x : ℚ, (|x - 3| = 2*x + 4) ↔ (x = -1/3) :=
sorry

end absolute_value_equation_solution_l847_84787


namespace solution_count_equals_divisors_of_square_l847_84710

/-- 
Given a positive integer n, count_solutions n returns the number of
ordered pairs (x, y) of positive integers satisfying xy/(x+y) = n
-/
def count_solutions (n : ℕ+) : ℕ :=
  sorry

/--
Given a positive integer n, num_divisors_square n returns the number of
positive divisors of n²
-/
def num_divisors_square (n : ℕ+) : ℕ :=
  sorry

/--
For any positive integer n, the number of ordered pairs (x, y) of
positive integers satisfying xy/(x+y) = n is equal to the number of
positive divisors of n²
-/
theorem solution_count_equals_divisors_of_square (n : ℕ+) :
  count_solutions n = num_divisors_square n :=
by sorry

end solution_count_equals_divisors_of_square_l847_84710


namespace december_savings_l847_84711

def savings_plan (initial_amount : ℕ) (months : ℕ) : ℕ :=
  (initial_amount : ℕ) * (3 ^ (months - 1))

theorem december_savings :
  savings_plan 10 12 = 1771470 := by
  sorry

end december_savings_l847_84711


namespace extraneous_roots_imply_k_equals_one_l847_84720

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x - 6) / (x - 5) = k / (5 - x)

-- Define the condition for extraneous roots
def has_extraneous_roots (k : ℝ) : Prop :=
  ∃ x, equation x k ∧ x = 5

-- Theorem statement
theorem extraneous_roots_imply_k_equals_one :
  ∀ k : ℝ, has_extraneous_roots k → k = 1 := by
  sorry

end extraneous_roots_imply_k_equals_one_l847_84720


namespace bicycle_trip_time_l847_84736

theorem bicycle_trip_time (mary_speed john_speed : ℝ) (distance : ℝ) : 
  mary_speed = 12 → 
  john_speed = 9 → 
  distance = 90 → 
  ∃ t : ℝ, t = 6 ∧ mary_speed * t ^ 2 + john_speed * t ^ 2 = distance ^ 2 :=
by
  sorry

end bicycle_trip_time_l847_84736


namespace min_value_quadratic_l847_84785

theorem min_value_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*k*x + k^2 + k + 3 = 0 ∧ y^2 + 2*k*y + k^2 + k + 3 = 0) →
  (∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*m*x + m^2 + m + 3 = 0 ∧ y^2 + 2*m*y + m^2 + m + 3 = 0) → 
  m^2 + m + 3 ≥ 9) :=
sorry

end min_value_quadratic_l847_84785


namespace solitaire_game_solvable_l847_84784

/-- Represents the state of a marker on the solitaire board -/
inductive MarkerState
| White
| Black

/-- Represents the solitaire game board -/
def Board (m n : ℕ) := Fin m → Fin n → MarkerState

/-- Initializes the board with all white markers except one black corner -/
def initBoard (m n : ℕ) : Board m n := sorry

/-- Represents a valid move in the game -/
def validMove (b : Board m n) (i : Fin m) (j : Fin n) : Prop := sorry

/-- The state of the board after making a move -/
def makeMove (b : Board m n) (i : Fin m) (j : Fin n) : Board m n := sorry

/-- Predicate to check if all markers have been removed from the board -/
def allMarkersRemoved (b : Board m n) : Prop := sorry

/-- Predicate to check if it's possible to remove all markers from the board -/
def canRemoveAllMarkers (m n : ℕ) : Prop := 
  ∃ (moves : List (Fin m × Fin n)), 
    let finalBoard := moves.foldl (λ b move => makeMove b move.1 move.2) (initBoard m n)
    allMarkersRemoved finalBoard

/-- The main theorem stating the condition for removing all markers -/
theorem solitaire_game_solvable (m n : ℕ) : 
  canRemoveAllMarkers m n ↔ m % 2 = 1 ∨ n % 2 = 1 := by sorry

end solitaire_game_solvable_l847_84784


namespace article_price_l847_84796

theorem article_price (P : ℝ) : 
  P > 0 →                            -- Initial price is positive
  0.9 * (0.8 * P) = 36 →             -- Final price after discounts is $36
  P = 50 :=                          -- Initial price is $50
by sorry

end article_price_l847_84796


namespace systematic_sampling_interval_example_l847_84769

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (totalPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  totalPopulation / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 50 is 20 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 1000 50 = 20 := by
  sorry

end systematic_sampling_interval_example_l847_84769


namespace total_carriages_eq_460_l847_84721

/-- The number of carriages in Euston -/
def euston : ℕ := 130

/-- The number of carriages in Norfolk -/
def norfolk : ℕ := euston - 20

/-- The number of carriages in Norwich -/
def norwich : ℕ := 100

/-- The number of carriages in Flying Scotsman -/
def flying_scotsman : ℕ := norwich + 20

/-- The total number of carriages -/
def total_carriages : ℕ := euston + norfolk + norwich + flying_scotsman

theorem total_carriages_eq_460 : total_carriages = 460 := by
  sorry

end total_carriages_eq_460_l847_84721


namespace largest_of_three_numbers_l847_84744

theorem largest_of_three_numbers (d e f : ℝ) 
  (sum_eq : d + e + f = 3)
  (sum_prod_eq : d * e + d * f + e * f = -14)
  (prod_eq : d * e * f = 21) :
  max d (max e f) = Real.sqrt 7 := by
sorry

end largest_of_three_numbers_l847_84744


namespace circle_placement_theorem_l847_84716

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  sideLength : ℝ

/-- Represents a circle with given diameter -/
structure Circle where
  diameter : ℝ

/-- Theorem: In a 20x25 rectangle with 120 unit squares, there exists a point for a circle with diameter 1 -/
theorem circle_placement_theorem (rect : Rectangle) (squares : Finset Square) (circ : Circle) :
  rect.width = 20 ∧ rect.height = 25 ∧
  squares.card = 120 ∧ (∀ s ∈ squares, s.sideLength = 1) ∧
  circ.diameter = 1 →
  ∃ (center : ℝ × ℝ),
    (center.1 ≥ 0 ∧ center.1 ≤ rect.width ∧ center.2 ≥ 0 ∧ center.2 ≤ rect.height) ∧
    (∀ s ∈ squares, ∀ (point : ℝ × ℝ),
      (point.1 - center.1)^2 + (point.2 - center.2)^2 ≤ (circ.diameter / 2)^2 →
      ¬(point.1 ≥ s.sideLength ∧ point.1 ≤ s.sideLength + 1 ∧
        point.2 ≥ s.sideLength ∧ point.2 ≤ s.sideLength + 1)) :=
by
  sorry

end circle_placement_theorem_l847_84716


namespace largest_prime_factor_of_sum_of_divisors_450_l847_84724

-- Define the sum of divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Define the largest prime factor function
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_sum_of_divisors_450 :
  largest_prime_factor (sum_of_divisors 450) = 31 := by sorry

end largest_prime_factor_of_sum_of_divisors_450_l847_84724


namespace line_growth_limit_l847_84702

theorem line_growth_limit :
  let initial_length : ℝ := 2
  let growth_series (n : ℕ) : ℝ := (1 / 3^n) * (1 + Real.sqrt 3)
  (initial_length + ∑' n, growth_series n) = (6 + Real.sqrt 3) / 2 :=
by
  sorry

end line_growth_limit_l847_84702


namespace negation_of_universal_proposition_l847_84758

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l847_84758


namespace rectangle_perimeter_in_square_l847_84726

/-- Given a square of side length y containing a smaller square of side length x,
    the perimeter of one of the four congruent rectangles formed in the remaining area
    is equal to 2y. -/
theorem rectangle_perimeter_in_square (y x : ℝ) (h1 : 0 < y) (h2 : 0 < x) (h3 : x < y) :
  2 * (y - x) + 2 * x = 2 * y :=
sorry

end rectangle_perimeter_in_square_l847_84726


namespace triangle_lines_theorem_l847_84751

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  medianCM : ℝ → ℝ → ℝ
  altitudeBH : ℝ → ℝ → ℝ

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  medianCM := λ x y => 2*x - y - 5
  altitudeBH := λ x y => x - 2*y - 5

/-- The equation of line BC -/
def line_BC (t : Triangle) : ℝ → ℝ → ℝ :=
  λ x y => 6*x - 5*y - 9

/-- The equation of the line symmetric to BC with respect to CM -/
def symmetric_line_BC (t : Triangle) : ℝ → ℝ → ℝ :=
  λ x y => 38*x - 9*y - 125

/-- Main theorem proving the equations of lines -/
theorem triangle_lines_theorem (t : Triangle) (h : t = given_triangle) :
  (line_BC t = λ x y => 6*x - 5*y - 9) ∧
  (symmetric_line_BC t = λ x y => 38*x - 9*y - 125) := by
  sorry

end triangle_lines_theorem_l847_84751


namespace basket_apples_theorem_l847_84740

/-- The total number of apples in the basket -/
def total_apples : ℕ := 5

/-- The probability of selecting at least one spoiled apple when picking 2 apples randomly -/
def prob_spoiled : ℚ := 2/5

/-- The number of spoiled apples in the basket -/
def spoiled_apples : ℕ := 1

/-- The number of good apples in the basket -/
def good_apples : ℕ := total_apples - spoiled_apples

theorem basket_apples_theorem :
  (total_apples = spoiled_apples + good_apples) ∧
  (prob_spoiled = 1 - (good_apples / total_apples) * ((good_apples - 1) / (total_apples - 1))) :=
by sorry

end basket_apples_theorem_l847_84740


namespace sum_of_ages_l847_84789

/-- Given the ages of Masc and Sam, prove that the sum of their ages is 27. -/
theorem sum_of_ages (Masc_age Sam_age : ℕ) 
  (h1 : Masc_age = Sam_age + 7)
  (h2 : Masc_age = 17)
  (h3 : Sam_age = 10) : 
  Masc_age + Sam_age = 27 := by
  sorry

end sum_of_ages_l847_84789


namespace min_double_rooms_part1_min_triple_rooms_part2_l847_84728

/-- Represents a hotel room configuration --/
structure RoomConfig where
  double_rooms : ℕ
  triple_rooms : ℕ

/-- Calculates the total number of students that can be accommodated --/
def total_students (config : RoomConfig) : ℕ :=
  2 * config.double_rooms + 3 * config.triple_rooms

/-- Calculates the total cost of the room configuration --/
def total_cost (config : RoomConfig) (double_price : ℕ) (triple_price : ℕ) : ℕ :=
  config.double_rooms * double_price + config.triple_rooms * triple_price

/-- Theorem for part (1) --/
theorem min_double_rooms_part1 (male_students female_students : ℕ) 
  (h_total : male_students + female_students = 50)
  (h_male : male_students = 27)
  (h_female : female_students = 23)
  (double_price triple_price : ℕ)
  (h_double_price : double_price = 200)
  (h_triple_price : triple_price = 250) :
  ∃ (config : RoomConfig), 
    total_students config ≥ 50 ∧ 
    config.double_rooms = 1 ∧
    (∀ (other_config : RoomConfig), 
      total_students other_config ≥ 50 → 
      total_cost config double_price triple_price ≤ total_cost other_config double_price triple_price) :=
sorry

/-- Theorem for part (2) --/
theorem min_triple_rooms_part2 (male_students female_students : ℕ) 
  (h_total : male_students + female_students = 50)
  (h_male : male_students = 27)
  (h_female : female_students = 23)
  (double_price triple_price : ℕ)
  (h_double_price : double_price = 160)  -- 20% discount applied
  (h_triple_price : triple_price = 250)
  (max_double_rooms : ℕ)
  (h_max_double : max_double_rooms = 15) :
  ∃ (config : RoomConfig), 
    total_students config ≥ 50 ∧ 
    config.double_rooms ≤ max_double_rooms ∧
    config.triple_rooms = 8 ∧
    (∀ (other_config : RoomConfig), 
      total_students other_config ≥ 50 ∧ 
      other_config.double_rooms ≤ max_double_rooms → 
      total_cost config double_price triple_price ≤ total_cost other_config double_price triple_price) :=
sorry

end min_double_rooms_part1_min_triple_rooms_part2_l847_84728


namespace division_problem_l847_84714

theorem division_problem : (100 : ℚ) / ((6 : ℚ) / 2) = 100 / 3 := by sorry

end division_problem_l847_84714


namespace original_weight_correct_l847_84756

/-- Represents the original weight Tom could lift per hand in kg -/
def original_weight : ℝ := 80

/-- Represents the total weight Tom can hold with both hands after training in kg -/
def total_weight : ℝ := 352

/-- Theorem stating that the original weight satisfies the given conditions -/
theorem original_weight_correct : 
  2 * (2 * original_weight * 1.1) = total_weight := by sorry

end original_weight_correct_l847_84756


namespace greatest_non_sum_of_complex_l847_84727

/-- A natural number is complex if it has at least two different prime divisors. -/
def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n

/-- A natural number is representable as the sum of two complex numbers. -/
def is_sum_of_complex (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ n = a + b

/-- 23 is the greatest natural number that cannot be represented as the sum of two complex numbers. -/
theorem greatest_non_sum_of_complex : ∀ n : ℕ, n > 23 → is_sum_of_complex n ∧ ¬is_sum_of_complex 23 :=
sorry

end greatest_non_sum_of_complex_l847_84727


namespace fraction_simplification_l847_84794

theorem fraction_simplification (x : ℝ) : (x - 1) / 3 + (-2 - 3 * x) / 2 = (-7 * x - 8) / 6 := by
  sorry

end fraction_simplification_l847_84794


namespace board_structure_count_l847_84795

/-- The number of ways to structure a corporate board -/
def board_structures (n : ℕ) : ℕ :=
  let president_choices := n
  let vp_choices := n - 1
  let remaining_after_vps := n - 3
  let dh_choices_vp1 := remaining_after_vps.choose 3
  let dh_choices_vp2 := (remaining_after_vps - 3).choose 3
  president_choices * (vp_choices * (vp_choices - 1)) * dh_choices_vp1 * dh_choices_vp2

/-- Theorem stating the number of ways to structure a 13-member board -/
theorem board_structure_count :
  board_structures 13 = 655920 := by
  sorry

end board_structure_count_l847_84795


namespace sufficient_condition_for_inequality_l847_84777

theorem sufficient_condition_for_inequality (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0 := by
sorry

end sufficient_condition_for_inequality_l847_84777


namespace no_natural_square_difference_2014_l847_84712

theorem no_natural_square_difference_2014 :
  ∀ m n : ℕ, m^2 ≠ n^2 + 2014 := by
sorry

end no_natural_square_difference_2014_l847_84712


namespace max_true_statements_l847_84788

theorem max_true_statements (a b : ℝ) : 
  ¬(∃ (p1 p2 p3 p4 : Prop), 
    (p1 ∧ p2 ∧ p3 ∧ p4) ∧
    (p1 → a < b) ∧
    (p2 → b < 0) ∧
    (p3 → a < 0) ∧
    (p4 → 1 / a < 1 / b) ∧
    (p1 ∨ p2 ∨ p3 ∨ p4 → a^2 < b^2)) :=
by sorry

end max_true_statements_l847_84788


namespace dog_reachable_area_is_8pi_l847_84780

/-- The area a dog can reach when tethered to a vertex of a regular hexagonal doghouse -/
def dogReachableArea (side_length : ℝ) (rope_length : ℝ) : ℝ :=
  -- Define the area calculation here
  sorry

/-- Theorem stating the area a dog can reach for the given conditions -/
theorem dog_reachable_area_is_8pi :
  dogReachableArea 2 3 = 8 * Real.pi := by
  sorry

end dog_reachable_area_is_8pi_l847_84780


namespace ellipse_focal_length_l847_84753

/-- An ellipse with equation x²/m + y²/5 = 1 and focal length 2 has m = 4 -/
theorem ellipse_focal_length (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 5 = 1) →  -- Ellipse equation
  2 = 2 * (Real.sqrt (5 - m)) →         -- Focal length is 2
  m = 4 := by
sorry

end ellipse_focal_length_l847_84753


namespace smallest_equal_partition_is_seven_l847_84718

/-- The sum of squares from 1 to n -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Checks if there exists a subset of squares that sum to half the total sum -/
def existsEqualPartition (n : ℕ) : Prop :=
  ∃ (subset : Finset ℕ), subset ⊆ Finset.range n ∧ 
    subset.sum (λ i => (i + 1)^2) = sumOfSquares n / 2

/-- The smallest n for which an equal partition exists -/
def smallestEqualPartition : ℕ := 7

theorem smallest_equal_partition_is_seven :
  (smallestEqualPartition = 7) ∧ 
  (existsEqualPartition 7) ∧ 
  (∀ k < 7, ¬ existsEqualPartition k) :=
sorry

end smallest_equal_partition_is_seven_l847_84718


namespace smallest_n_for_3003_terms_l847_84742

theorem smallest_n_for_3003_terms : ∃ (N : ℕ), 
  (N = 19) ∧ 
  (∀ k < N, (Nat.choose (k + 1) 5) < 3003) ∧
  (Nat.choose (N + 1) 5 = 3003) :=
sorry

end smallest_n_for_3003_terms_l847_84742


namespace rubber_band_length_l847_84723

theorem rubber_band_length (r₁ r₂ d : ℝ) (hr₁ : r₁ = 3) (hr₂ : r₂ = 9) (hd : d = 12) :
  ∃ (L : ℝ), L = 4 * Real.pi + 12 * Real.sqrt 3 ∧
  L = 2 * (r₁ * Real.arctan ((Real.sqrt (d^2 - (r₂ - r₁)^2)) / (r₂ - r₁)) +
           r₂ * Real.arctan ((Real.sqrt (d^2 - (r₂ - r₁)^2)) / (r₂ - r₁)) +
           Real.sqrt (d^2 - (r₂ - r₁)^2)) :=
by sorry

end rubber_band_length_l847_84723


namespace rope_cutting_problem_l847_84741

theorem rope_cutting_problem :
  Nat.gcd 48 (Nat.gcd 72 (Nat.gcd 96 120)) = 24 := by sorry

end rope_cutting_problem_l847_84741


namespace hyperbola_eccentricity_l847_84707

/-- The eccentricity of a hyperbola with equation x²/2 - y² = -1 is √3 -/
theorem hyperbola_eccentricity : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2/2 - y^2 = -1
  ∃ e : ℝ, e = Real.sqrt 3 ∧ 
    ∀ x y : ℝ, h x y → 
      e = Real.sqrt (1 + (x^2/2)/(y^2)) := by sorry

end hyperbola_eccentricity_l847_84707


namespace proposal_i_percentage_l847_84749

def survey_results (P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all : ℝ) : Prop :=
  P_i + P_ii + P_iii - P_i_and_ii - P_i_and_iii - P_ii_and_iii + P_all = 78 ∧
  P_ii = 30 ∧
  P_iii = 20 ∧
  P_all = 5 ∧
  P_i_and_ii + P_i_and_iii + P_ii_and_iii = 32

theorem proposal_i_percentage :
  ∀ P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all : ℝ,
  survey_results P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all →
  P_i = 55 :=
by sorry

end proposal_i_percentage_l847_84749


namespace interval_constraint_l847_84701

theorem interval_constraint (x : ℝ) : (1 < 2*x ∧ 2*x < 2) ∧ (1 < 3*x ∧ 3*x < 2) ↔ 1/2 < x ∧ x < 2/3 := by
  sorry

end interval_constraint_l847_84701


namespace divisor_problem_l847_84730

theorem divisor_problem (original : Nat) (subtracted : Nat) (divisor : Nat) : 
  original = 427398 →
  subtracted = 8 →
  divisor = 10 →
  (original - subtracted) % divisor = 0 :=
by
  sorry

end divisor_problem_l847_84730


namespace good_number_iff_divisible_by_8_l847_84790

def is_good_number (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (2*k - 3) + (2*k - 1) + (2*k + 1) + (2*k + 3)

theorem good_number_iff_divisible_by_8 (n : ℕ) :
  is_good_number n ↔ n % 8 = 0 := by sorry

end good_number_iff_divisible_by_8_l847_84790


namespace chopstick_length_l847_84762

/-- The length of a chopstick given specific wetness conditions -/
theorem chopstick_length (wetted_length : ℝ) (h1 : wetted_length = 8) 
  (h2 : wetted_length + wetted_length / 2 + wetted_length = 24) : ℝ :=
by
  sorry

#check chopstick_length

end chopstick_length_l847_84762
