import Mathlib

namespace NUMINAMATH_CALUDE_max_value_quadratic_l3609_360944

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) :
  (6 - x) * x ≤ 9 ∧ ∃ y, 0 < y ∧ y < 6 ∧ (6 - y) * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3609_360944


namespace NUMINAMATH_CALUDE_accounting_equation_l3609_360946

def p : ℂ := 7
def z : ℂ := 7 + 175 * Complex.I

theorem accounting_equation (h : 3 * p - z = 15000) : 
  p = 5002 + (175 / 3) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_accounting_equation_l3609_360946


namespace NUMINAMATH_CALUDE_cube_spheres_surface_area_ratio_l3609_360966

/-- The ratio of the surface area of a cube's inscribed sphere to its circumscribed sphere -/
theorem cube_spheres_surface_area_ratio (a : ℝ) (h : a > 0) : 
  (4 * Real.pi * (a / 2)^2) / (4 * Real.pi * (a * Real.sqrt 3 / 2)^2) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cube_spheres_surface_area_ratio_l3609_360966


namespace NUMINAMATH_CALUDE_students_not_enrolled_in_french_or_german_l3609_360928

theorem students_not_enrolled_in_french_or_german 
  (total_students : ℕ) 
  (french_students : ℕ) 
  (german_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 78)
  (h2 : french_students = 41)
  (h3 : german_students = 22)
  (h4 : both_students = 9) :
  total_students - (french_students + german_students - both_students) = 24 :=
by sorry


end NUMINAMATH_CALUDE_students_not_enrolled_in_french_or_german_l3609_360928


namespace NUMINAMATH_CALUDE_arithmetic_sequence_line_passes_through_point_l3609_360930

/-- Given that A, B, and C form an arithmetic sequence,
    prove that the line Ax + By + C = 0 passes through the point (1, -2) -/
theorem arithmetic_sequence_line_passes_through_point
  (A B C : ℝ) (h : 2 * B = A + C) :
  A * 1 + B * (-2) + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_line_passes_through_point_l3609_360930


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3609_360996

theorem boys_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (neither_children : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 22)
  (h6 : total_girls = 38)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3609_360996


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l3609_360984

def binary_to_decimal (b₅ b₄ b₃ b₂ b₁ b₀ : ℕ) : ℕ :=
  b₀ + 2 * b₁ + 2^2 * b₂ + 2^3 * b₃ + 2^4 * b₄ + 2^5 * b₅

theorem binary_110011_equals_51 : binary_to_decimal 1 1 0 0 1 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l3609_360984


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3609_360940

theorem binomial_coefficient_sum (x a : ℝ) (x_nonzero : x ≠ 0) (a_nonzero : a ≠ 0) :
  (Finset.range 7).sum (λ k => Nat.choose 6 k) = 64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3609_360940


namespace NUMINAMATH_CALUDE_cos_sin_inequalities_l3609_360914

theorem cos_sin_inequalities (x : ℝ) (h : 0 < x ∧ x < π/4) :
  (Real.cos x) ^ ((Real.cos x)^2) > (Real.sin x) ^ ((Real.sin x)^2) ∧
  (Real.cos x) ^ ((Real.cos x)^4) < (Real.sin x) ^ ((Real.sin x)^4) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_inequalities_l3609_360914


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l3609_360925

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℚ := 75 + 12 * x

/-- Gamma T-shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℚ := 16 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme : ℕ := 19

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme < gamma_cost min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → acme_cost n ≥ gamma_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l3609_360925


namespace NUMINAMATH_CALUDE_vector_properties_l3609_360967

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_properties :
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let proj_b_on_a := (Real.sqrt (b.1^2 + b.2^2) * cos_theta)
  cos_theta = (4 * Real.sqrt 65) / 65 ∧
  proj_b_on_a = (8 * Real.sqrt 13) / 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3609_360967


namespace NUMINAMATH_CALUDE_degree_of_g_l3609_360904

def f (x : ℝ) : ℝ := -7 * x^4 + 3 * x^3 + x - 5

theorem degree_of_g (g : ℝ → ℝ) :
  (∃ (a b : ℝ), ∀ x, f x + g x = a * x + b) →
  (∃ (a b c d e : ℝ), a ≠ 0 ∧ ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e) :=
by sorry

end NUMINAMATH_CALUDE_degree_of_g_l3609_360904


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3609_360986

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3609_360986


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3609_360991

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3609_360991


namespace NUMINAMATH_CALUDE_certain_value_multiplication_l3609_360938

theorem certain_value_multiplication (x : ℝ) : x * (1/7)^2 = 7^3 → x = 16807 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_multiplication_l3609_360938


namespace NUMINAMATH_CALUDE_candy_distribution_l3609_360970

/-- 
Given the initial number of candies, the number of friends, and the additional candies bought,
prove that the number of candies each friend will receive is equal to the total number of candies
divided by the number of friends.
-/
theorem candy_distribution (initial_candies : ℕ) (friends : ℕ) (additional_candies : ℕ)
  (h1 : initial_candies = 35)
  (h2 : friends = 10)
  (h3 : additional_candies = 15)
  (h4 : friends > 0) :
  (initial_candies + additional_candies) / friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3609_360970


namespace NUMINAMATH_CALUDE_current_speed_l3609_360902

/-- Calculates the speed of the current given boat travel information -/
theorem current_speed (boat_speed : ℝ) (distance : ℝ) (time_against : ℝ) (time_with : ℝ) :
  boat_speed = 15.6 →
  distance = 96 →
  time_against = 8 →
  time_with = 5 →
  ∃ (current_speed : ℝ),
    distance = time_against * (boat_speed - current_speed) ∧
    distance = time_with * (boat_speed + current_speed) ∧
    current_speed = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l3609_360902


namespace NUMINAMATH_CALUDE_quadratic_b_value_l3609_360941

/-- A quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x-coordinate on a quadratic function -/
def QuadraticFunction.y (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_b_value (f : QuadraticFunction) (y₁ y₂ : ℝ) 
  (h₁ : f.y 2 = y₁)
  (h₂ : f.y (-2) = y₂)
  (h₃ : y₁ - y₂ = -12) :
  f.b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l3609_360941


namespace NUMINAMATH_CALUDE_smallest_factor_for_square_l3609_360988

theorem smallest_factor_for_square (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 10 → ¬∃ k : ℕ, 4410 * m = k * k) ∧ 
  (∃ k : ℕ, 4410 * 10 = k * k) := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_square_l3609_360988


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l3609_360923

/-- Sum of the first n even integers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of the first n odd integers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- Count of odd integers divisible by 5 up to 2n-1 -/
def count_odd_div_5 (n : ℕ) : ℕ := (2*n - 1) / 10 + 1

/-- Sum of odd integers divisible by 5 up to 2n-1 -/
def sum_odd_div_5 (n : ℕ) : ℕ := 5 * (count_odd_div_5 n) * (count_odd_div_5 n)

/-- Sum of odd integers not divisible by 5 up to 2n-1 -/
def sum_odd_not_div_5 (n : ℕ) : ℕ := sum_odd n - sum_odd_div_5 n

theorem even_odd_sum_difference (n : ℕ) : 
  sum_even n - sum_odd_not_div_5 n = 51000 := by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l3609_360923


namespace NUMINAMATH_CALUDE_sum_abc_equals_13_l3609_360915

theorem sum_abc_equals_13 (a b c : ℕ+) 
  (h : (a : ℝ)^2 + (b : ℝ)^2 + (c : ℝ)^2 + 43 ≤ (a : ℝ) * (b : ℝ) + 9 * (b : ℝ) + 8 * (c : ℝ)) :
  (a : ℕ) + b + c = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_13_l3609_360915


namespace NUMINAMATH_CALUDE_beach_volleyball_max_players_l3609_360955

theorem beach_volleyball_max_players : ∃ (n : ℕ), n > 0 ∧ n ≤ 13 ∧ 
  (∀ (m : ℕ), m > 13 → ¬(
    (∃ (games : Finset (Finset ℕ)), 
      games.card = m ∧ 
      (∀ g ∈ games, g.card = 4) ∧
      (∀ i j, i < m ∧ j < m ∧ i ≠ j → ∃ g ∈ games, i ∈ g ∧ j ∈ g)
    )
  )) := by sorry

end NUMINAMATH_CALUDE_beach_volleyball_max_players_l3609_360955


namespace NUMINAMATH_CALUDE_relay_schemes_count_l3609_360971

/-- The number of segments in the Olympic torch relay route -/
def num_segments : ℕ := 6

/-- The number of torchbearers -/
def num_torchbearers : ℕ := 6

/-- The set of possible first runners -/
inductive FirstRunner
| A
| B
| C

/-- The set of possible last runners -/
inductive LastRunner
| A
| B

/-- A function to calculate the number of relay schemes -/
def count_relay_schemes : ℕ := sorry

/-- Theorem stating that the number of relay schemes is 96 -/
theorem relay_schemes_count :
  count_relay_schemes = 96 := by sorry

end NUMINAMATH_CALUDE_relay_schemes_count_l3609_360971


namespace NUMINAMATH_CALUDE_floor_of_2_99_l3609_360993

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the properties of the floor function
axiom floor_le (x : ℝ) : (floor x : ℝ) ≤ x
axiom floor_lt (x : ℝ) : x < (floor x : ℝ) + 1

-- Theorem statement
theorem floor_of_2_99 : floor 2.99 = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_2_99_l3609_360993


namespace NUMINAMATH_CALUDE_intersection_A_B_l3609_360980

open Set

def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_A_B : A ∩ B = Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3609_360980


namespace NUMINAMATH_CALUDE_inequality_solution_l3609_360998

theorem inequality_solution (x : ℝ) : 2 - 1 / (3 * x + 4) < 5 ↔ x > -4/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3609_360998


namespace NUMINAMATH_CALUDE_average_weight_increase_l3609_360903

theorem average_weight_increase (group_size : ℕ) (original_weight new_weight : ℝ) :
  group_size = 6 →
  original_weight = 65 →
  new_weight = 74 →
  (new_weight - original_weight) / group_size = 1.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3609_360903


namespace NUMINAMATH_CALUDE_parking_lot_search_time_l3609_360921

/-- Calculates the time spent searching a parking lot given the layout and walking speed. -/
theorem parking_lot_search_time
  (section_g_rows : ℕ)
  (section_g_cars_per_row : ℕ)
  (section_h_rows : ℕ)
  (section_h_cars_per_row : ℕ)
  (cars_passed_per_minute : ℕ)
  (h_section_g_rows : section_g_rows = 15)
  (h_section_g_cars : section_g_cars_per_row = 10)
  (h_section_h_rows : section_h_rows = 20)
  (h_section_h_cars : section_h_cars_per_row = 9)
  (h_cars_passed : cars_passed_per_minute = 11)
  : (section_g_rows * section_g_cars_per_row + section_h_rows * section_h_cars_per_row) / cars_passed_per_minute = 30 := by
  sorry


end NUMINAMATH_CALUDE_parking_lot_search_time_l3609_360921


namespace NUMINAMATH_CALUDE_strawberry_division_l3609_360962

theorem strawberry_division (brother_baskets : ℕ) (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) (parents_difference : ℕ) (family_members : ℕ) :
  brother_baskets = 3 →
  strawberries_per_basket = 15 →
  kimberly_multiplier = 8 →
  parents_difference = 93 →
  family_members = 4 →
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let kimberly_strawberries := kimberly_multiplier * brother_strawberries
  let parents_strawberries := kimberly_strawberries - parents_difference
  let total_strawberries := brother_strawberries + kimberly_strawberries + parents_strawberries
  (total_strawberries / family_members : ℕ) = 168 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_division_l3609_360962


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3609_360981

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = (2 + Real.sqrt 2) / 2 ∧ 2 * x₁^2 = 4 * x₁ - 1) ∧
  (x₂ = (2 - Real.sqrt 2) / 2 ∧ 2 * x₂^2 = 4 * x₂ - 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3609_360981


namespace NUMINAMATH_CALUDE_courtyard_length_l3609_360992

/-- Proves that a courtyard with given width and number of bricks of specific dimensions has a certain length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) :
  width = 14 →
  brick_length = 0.25 →
  brick_width = 0.15 →
  num_bricks = 8960 →
  (width * (num_bricks * brick_length * brick_width / width)) = 24 := by
  sorry

#check courtyard_length

end NUMINAMATH_CALUDE_courtyard_length_l3609_360992


namespace NUMINAMATH_CALUDE_ellipse_k_range_l3609_360959

/-- An ellipse represented by the equation x^2 + ky^2 = 2 with foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ x y : ℝ, x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the condition that foci are on y-axis

/-- The range of k for the given ellipse -/
def k_range (e : Ellipse) : Set ℝ :=
  {k : ℝ | 0 < k ∧ k < 1}

/-- Theorem stating that for the given ellipse, k is in the range (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : e.k ∈ k_range e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l3609_360959


namespace NUMINAMATH_CALUDE_box_triples_count_l3609_360960

/-- The number of ordered triples (a, b, c) satisfying the box conditions -/
def box_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    a ≤ b ∧ b ≤ c ∧ 2 * a * b * c = 2 * a * b + 2 * b * c + 2 * a * c)
    (Finset.product (Finset.range 100) (Finset.product (Finset.range 100) (Finset.range 100)))).card

/-- The main theorem stating that there are exactly 10 ordered triples satisfying the conditions -/
theorem box_triples_count : box_triples = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_triples_count_l3609_360960


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l3609_360901

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 600) 
  (h_product : a * b = 18000) : 
  Nat.gcd a b = 30 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l3609_360901


namespace NUMINAMATH_CALUDE_new_weighted_average_age_l3609_360912

/-- The new weighted average age of a class after new students join -/
theorem new_weighted_average_age
  (n₁ : ℕ) (a₁ : ℝ)
  (n₂ : ℕ) (a₂ : ℝ)
  (n₃ : ℕ) (a₃ : ℝ)
  (n₄ : ℕ) (a₄ : ℝ)
  (n₅ : ℕ) (a₅ : ℝ)
  (h₁ : n₁ = 15) (h₂ : a₁ = 42)
  (h₃ : n₂ = 20) (h₄ : a₂ = 35)
  (h₅ : n₃ = 10) (h₆ : a₃ = 50)
  (h₇ : n₄ = 7)  (h₈ : a₄ = 30)
  (h₉ : n₅ = 11) (h₁₀ : a₅ = 45) :
  (n₁ * a₁ + n₂ * a₂ + n₃ * a₃ + n₄ * a₄ + n₅ * a₅) / (n₁ + n₂ + n₃ + n₄ + n₅) = 2535 / 63 := by
  sorry

#eval (2535 : Float) / 63

end NUMINAMATH_CALUDE_new_weighted_average_age_l3609_360912


namespace NUMINAMATH_CALUDE_f_neither_even_nor_odd_l3609_360978

-- Define the function f on the given domain
def f : {x : ℝ | -1 < x ∧ x ≤ 1} → ℝ := fun x => x.val ^ 2

-- State the theorem
theorem f_neither_even_nor_odd :
  ¬(∀ x : {x : ℝ | -1 < x ∧ x ≤ 1}, f ⟨-x.val, by sorry⟩ = f x) ∧
  ¬(∀ x : {x : ℝ | -1 < x ∧ x ≤ 1}, f ⟨-x.val, by sorry⟩ = -f x) :=
by sorry

end NUMINAMATH_CALUDE_f_neither_even_nor_odd_l3609_360978


namespace NUMINAMATH_CALUDE_age_problem_solution_l3609_360919

/-- Given three people a, b, and c, with their ages represented as natural numbers. -/
def age_problem (a b c : ℕ) : Prop :=
  -- The average age of a, b, and c is 27 years
  (a + b + c) / 3 = 27 ∧
  -- The average age of a and c is 29 years
  (a + c) / 2 = 29 →
  -- The age of b is 23 years
  b = 23

/-- Theorem stating that under the given conditions, b's age is 23 years -/
theorem age_problem_solution :
  ∀ a b c : ℕ, age_problem a b c :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_solution_l3609_360919


namespace NUMINAMATH_CALUDE_direction_vector_of_line_l3609_360927

/-- Given a line 2x - 3y + 1 = 0, prove that (3, 2) is a direction vector --/
theorem direction_vector_of_line (x y : ℝ) : 
  (2 * x - 3 * y + 1 = 0) → (∃ (t : ℝ), x = 3 * t ∧ y = 2 * t) := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_of_line_l3609_360927


namespace NUMINAMATH_CALUDE_population_growth_l3609_360953

theorem population_growth (p : ℕ) : 
  p > 0 →                           -- p is positive
  (p^2 + 121 = q^2 + 16) →          -- 2005 population condition
  (p^2 + 346 = r^2) →               -- 2015 population condition
  ∃ (growth : ℝ), 
    growth = ((p^2 + 346 - p^2) / p^2) * 100 ∧ 
    abs (growth - 111) < abs (growth - 100) ∧ 
    abs (growth - 111) < abs (growth - 105) ∧ 
    abs (growth - 111) < abs (growth - 110) ∧ 
    abs (growth - 111) < abs (growth - 115) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_l3609_360953


namespace NUMINAMATH_CALUDE_product_of_radicals_l3609_360975

theorem product_of_radicals (p : ℝ) (hp : p > 0) :
  Real.sqrt (42 * p) * Real.sqrt (14 * p) * Real.sqrt (7 * p) = 14 * p * Real.sqrt (21 * p) := by
  sorry

end NUMINAMATH_CALUDE_product_of_radicals_l3609_360975


namespace NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l3609_360907

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem for intervals of monotonicity and minimum value
theorem f_monotonicity_and_minimum :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x < f y) ∧
  (∀ x y, x < y ∧ x > 3 ∧ y > 3 → f x < f y) ∧
  (∀ x y, x < y ∧ x > -1 ∧ y < 3 → f x > f y) ∧
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≥ -20) ∧
  (f 2 = -20) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l3609_360907


namespace NUMINAMATH_CALUDE_stream_top_width_l3609_360994

/-- 
Theorem: Given a trapezoidal cross-section of a stream with specified dimensions,
prove that the width at the top of the stream is 10 meters.
-/
theorem stream_top_width 
  (area : ℝ) 
  (depth : ℝ) 
  (bottom_width : ℝ) 
  (h_area : area = 640) 
  (h_depth : depth = 80) 
  (h_bottom : bottom_width = 6) :
  let top_width := (2 * area / depth) - bottom_width
  top_width = 10 := by
sorry

end NUMINAMATH_CALUDE_stream_top_width_l3609_360994


namespace NUMINAMATH_CALUDE_m_range_l3609_360982

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (2*m - 2)*x + 3

-- Define the proposition p
def p (m : ℝ) : Prop := ∀ x < 0, ∀ y < x, f m x > f m y

-- Define the proposition q
def q (m : ℝ) : Prop := ∀ x, x^2 - 4*x + 1 - m > 0

-- State the theorem
theorem m_range :
  (∀ m : ℝ, p m → m ≤ 1) →
  (∀ m : ℝ, q m → m < -3) →
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  ∃ a b : ℝ, a = -3 ∧ b = 1 ∧ ∀ m : ℝ, a ≤ m ∧ m ≤ b :=
sorry

end NUMINAMATH_CALUDE_m_range_l3609_360982


namespace NUMINAMATH_CALUDE_symmetry_xoy_plane_l3609_360987

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the xOy plane -/
def symmetricXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_xoy_plane :
  let A : Point3D := { x := 1, y := 2, z := 3 }
  let B : Point3D := symmetricXOY A
  B = { x := 1, y := 2, z := -3 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_xoy_plane_l3609_360987


namespace NUMINAMATH_CALUDE_num_factors_of_m_l3609_360952

/-- The number of natural-number factors of m = 2^3 * 3^3 * 5^4 * 6^5 -/
def num_factors (m : ℕ) : ℕ := sorry

/-- m is defined as 2^3 * 3^3 * 5^4 * 6^5 -/
def m : ℕ := 2^3 * 3^3 * 5^4 * 6^5

theorem num_factors_of_m :
  num_factors m = 405 := by sorry

end NUMINAMATH_CALUDE_num_factors_of_m_l3609_360952


namespace NUMINAMATH_CALUDE_no_real_roots_l3609_360911

theorem no_real_roots : ¬∃ (x : ℝ), Real.sqrt (x + 9) - Real.sqrt (x - 6) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3609_360911


namespace NUMINAMATH_CALUDE_expression_equality_l3609_360949

theorem expression_equality (x y z : ℝ) : 
  (2 * x - (3 * y - 4 * z)) - ((2 * x - 3 * y) - 5 * z) = 9 * z := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3609_360949


namespace NUMINAMATH_CALUDE_pythagorean_triple_odd_l3609_360947

theorem pythagorean_triple_odd (a : ℕ) (h1 : a ≥ 3) (h2 : Odd a) :
  a^2 + ((a^2 - 1) / 2)^2 = ((a^2 + 1) / 2)^2 := by
  sorry

#check pythagorean_triple_odd

end NUMINAMATH_CALUDE_pythagorean_triple_odd_l3609_360947


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l3609_360948

-- Define the sets A and B as functions of a
def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a+3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

-- State the theorem
theorem set_intersection_and_union :
  ∃ (a : ℝ), (A a ∩ B a = {2, 5}) ∧ (a = 2) ∧ (A a ∪ B a = {-4, 2, 4, 5, 25}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l3609_360948


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3609_360969

theorem sqrt_mixed_number_simplification :
  Real.sqrt (12 + 9/16) = Real.sqrt 201 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l3609_360969


namespace NUMINAMATH_CALUDE_min_sum_first_two_terms_l3609_360913

def is_valid_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) * (b (n + 1) + 1) = b n + 2210

theorem min_sum_first_two_terms (b : ℕ → ℕ) (h : is_valid_sequence b) :
  ∃ b₁ b₂ : ℕ, b 1 = b₁ ∧ b 2 = b₂ ∧ b₁ + b₂ = 147 ∧
  ∀ b₁' b₂' : ℕ, b 1 = b₁' ∧ b 2 = b₂' → b₁' + b₂' ≥ 147 :=
sorry

end NUMINAMATH_CALUDE_min_sum_first_two_terms_l3609_360913


namespace NUMINAMATH_CALUDE_weight_difference_theorem_l3609_360935

def weight_difference (robbie_weight patty_multiplier jim_multiplier mary_multiplier patty_loss jim_loss mary_gain : ℝ) : ℝ :=
  let patty_weight := patty_multiplier * robbie_weight - patty_loss
  let jim_weight := jim_multiplier * robbie_weight - jim_loss
  let mary_weight := mary_multiplier * robbie_weight + mary_gain
  patty_weight + jim_weight + mary_weight - robbie_weight

theorem weight_difference_theorem :
  weight_difference 100 4.5 3 2 235 180 45 = 480 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_theorem_l3609_360935


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3609_360933

theorem fraction_irreducible (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  ¬∃ (f g : ℝ → ℝ → ℝ → ℝ), (∀ x y z, f x y z ≠ 0 ∧ g x y z ≠ 0) ∧ 
    (∀ x y z, (x^2 + y^2 - z^2 + x*y) / (x^2 + z^2 - y^2 + y*z) = f x y z / g x y z) ∧
    (f a b c / g a b c ≠ (a^2 + b^2 - c^2 + a*b) / (a^2 + c^2 - b^2 + b*c)) :=
sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3609_360933


namespace NUMINAMATH_CALUDE_not_square_for_any_base_l3609_360977

-- Define the representation of a number in base b
def base_b_representation (b : ℕ) : ℕ := b^2 + 3*b + 3

-- Theorem statement
theorem not_square_for_any_base :
  ∀ b : ℕ, b ≥ 2 → ¬ ∃ n : ℕ, base_b_representation b = n^2 :=
by sorry

end NUMINAMATH_CALUDE_not_square_for_any_base_l3609_360977


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3609_360964

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : GeometricSequence a)
  (h_third_term : a 3 = 36)
  (h_fourth_term : a 4 = 54) :
  a 1 = 16 := by
  sorry

#check geometric_sequence_first_term

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3609_360964


namespace NUMINAMATH_CALUDE_nested_fourth_root_l3609_360956

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M^(1/4) * M^(1/16)))^(1/4) = M^(21/64) :=
sorry

end NUMINAMATH_CALUDE_nested_fourth_root_l3609_360956


namespace NUMINAMATH_CALUDE_sum_of_cubic_and_quartic_terms_l3609_360957

theorem sum_of_cubic_and_quartic_terms (π : ℝ) : 3 * (3 - π)^3 + 4 * (2 - π)^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubic_and_quartic_terms_l3609_360957


namespace NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l3609_360961

/-- The number of sides of a polygon where the sum of interior angles equals the sum of exterior angles -/
theorem polygon_with_equal_angle_sums (n : ℕ) : n > 2 →
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l3609_360961


namespace NUMINAMATH_CALUDE_correct_seating_arrangement_l3609_360973

/-- Represents whether a person is sitting or not -/
inductive Sitting : Type
| yes : Sitting
| no : Sitting

/-- The seating arrangement of individuals M, I, P, and A -/
structure SeatingArrangement :=
  (M : Sitting)
  (I : Sitting)
  (P : Sitting)
  (A : Sitting)

/-- The theorem stating the correct seating arrangement based on the given conditions -/
theorem correct_seating_arrangement :
  ∀ (arrangement : SeatingArrangement),
    arrangement.M = Sitting.no →
    (arrangement.M = Sitting.no → arrangement.I = Sitting.yes) →
    (arrangement.I = Sitting.yes → arrangement.P = Sitting.yes) →
    arrangement.A = Sitting.no →
    (arrangement.P = Sitting.yes ∧ 
     arrangement.I = Sitting.yes ∧ 
     arrangement.M = Sitting.no ∧ 
     arrangement.A = Sitting.no) :=
by sorry


end NUMINAMATH_CALUDE_correct_seating_arrangement_l3609_360973


namespace NUMINAMATH_CALUDE_sqrt_real_range_l3609_360943

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 3 + x) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l3609_360943


namespace NUMINAMATH_CALUDE_removed_terms_product_l3609_360924

theorem removed_terms_product (s : Finset ℚ) : 
  s ⊆ {1/2, 1/4, 1/6, 1/8, 1/10, 1/12} →
  s.sum id = 1 →
  (({1/2, 1/4, 1/6, 1/8, 1/10, 1/12} \ s).prod id) = 1/80 := by
  sorry

end NUMINAMATH_CALUDE_removed_terms_product_l3609_360924


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l3609_360916

def num_flavors : ℕ := 4
def num_scoops : ℕ := 5

def total_distributions : ℕ := (num_scoops + num_flavors - 1).choose (num_flavors - 1)
def non_mint_distributions : ℕ := (num_scoops + (num_flavors - 1) - 1).choose ((num_flavors - 1) - 1)

theorem ice_cream_flavors :
  total_distributions - non_mint_distributions = 35 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l3609_360916


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3609_360931

/-- A quadratic function with vertex (h, k) has the form f(x) = a(x-h)^2 + k -/
def quadratic_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = quadratic_vertex_form a 2 5 x) →  -- Condition 2: vertex at (2, 5)
  f 3 = 7 →  -- Condition 3: point (3, 7) lies on the graph
  a = 2 := by  -- Question: Find the value of a
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3609_360931


namespace NUMINAMATH_CALUDE_fourth_grade_students_l3609_360918

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 8 → left = 5 → new = 8 → final = initial - left + new → final = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l3609_360918


namespace NUMINAMATH_CALUDE_root_squared_plus_double_eq_three_l3609_360954

theorem root_squared_plus_double_eq_three (m : ℝ) : 
  m^2 + 2*m - 3 = 0 → m^2 + 2*m = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_squared_plus_double_eq_three_l3609_360954


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3609_360958

theorem triangle_side_ratio (a b c : ℝ) (A : ℝ) (h1 : A = 2 * Real.pi / 3) 
  (h2 : a^2 = 2*b*c + 3*c^2) : c/b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3609_360958


namespace NUMINAMATH_CALUDE_quadratic_solution_l3609_360965

/-- A quadratic function passing through specific points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the solutions of the quadratic equation -/
theorem quadratic_solution (a b c : ℝ) :
  (f a b c (-1) = 8) →
  (f a b c 0 = 3) →
  (f a b c 1 = 0) →
  (f a b c 2 = -1) →
  (f a b c 3 = 0) →
  (∀ x : ℝ, f a b c x = 0 ↔ x = 1 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3609_360965


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l3609_360937

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_two_digit_prime_factor_of_binomial_200_100 :
  ∃ (p : ℕ), Prime p ∧ p < 100 ∧ p ∣ binomial 200 100 ∧
  ∀ (q : ℕ), Prime q → q < 100 → q ∣ binomial 200 100 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l3609_360937


namespace NUMINAMATH_CALUDE_time_period_is_seven_days_l3609_360909

/-- The number of horses Minnie mounts per day -/
def minnie_daily_mounts : ℕ := sorry

/-- The number of days in the time period -/
def time_period : ℕ := sorry

/-- The number of horses Mickey mounts per day -/
def mickey_daily_mounts : ℕ := sorry

/-- Mickey mounts six less than twice as many horses per day as Minnie -/
axiom mickey_minnie_relation : mickey_daily_mounts = 2 * minnie_daily_mounts - 6

/-- Minnie mounts three more horses per day than there are days in the time period -/
axiom minnie_time_relation : minnie_daily_mounts = time_period + 3

/-- Mickey mounts 98 horses per week -/
axiom mickey_weekly_mounts : mickey_daily_mounts * 7 = 98

/-- The main theorem: The time period is 7 days -/
theorem time_period_is_seven_days : time_period = 7 := by sorry

end NUMINAMATH_CALUDE_time_period_is_seven_days_l3609_360909


namespace NUMINAMATH_CALUDE_solve_equation_l3609_360990

theorem solve_equation : ∃ x : ℝ, (3 * x) / 4 = 15 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3609_360990


namespace NUMINAMATH_CALUDE_lewis_weekly_rent_l3609_360972

/-- Calculates the weekly rent given the total rent and number of weeks -/
def weekly_rent (total_rent : ℕ) (num_weeks : ℕ) : ℚ :=
  (total_rent : ℚ) / (num_weeks : ℚ)

/-- Theorem: The weekly rent for Lewis during harvest season -/
theorem lewis_weekly_rent :
  weekly_rent 527292 1359 = 388 := by
  sorry

end NUMINAMATH_CALUDE_lewis_weekly_rent_l3609_360972


namespace NUMINAMATH_CALUDE_pentagon_interior_angle_mean_l3609_360929

/-- The mean value of the measures of the interior angles of a pentagon is 108 degrees. -/
theorem pentagon_interior_angle_mean :
  let n : ℕ := 5  -- number of sides in a pentagon
  let sum_of_angles : ℝ := (n - 2) * 180  -- sum of interior angles
  let mean_angle : ℝ := sum_of_angles / n  -- mean value of interior angles
  mean_angle = 108 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_interior_angle_mean_l3609_360929


namespace NUMINAMATH_CALUDE_polynomial_degree_l3609_360936

/-- The degree of the polynomial (3x^5 + 2x^4 - x + 5)(4x^11 - 2x^8 + 5x^5 - 9) - (x^2 - 3)^9 is 18 -/
theorem polynomial_degree : ℕ := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l3609_360936


namespace NUMINAMATH_CALUDE_probability_six_consecutive_heads_l3609_360950

/-- A sequence of coin flips represented as a list of booleans, where true represents heads and false represents tails. -/
def CoinFlips := List Bool

/-- The number of coin flips. -/
def numFlips : Nat := 10

/-- A function that checks if a list of coin flips contains at least 6 consecutive heads. -/
def hasAtLeastSixConsecutiveHeads (flips : CoinFlips) : Bool :=
  sorry

/-- The total number of possible outcomes for 10 coin flips. -/
def totalOutcomes : Nat := 2^numFlips

/-- The number of favorable outcomes (sequences with at least 6 consecutive heads). -/
def favorableOutcomes : Nat := 129

/-- The probability of getting at least 6 consecutive heads in 10 flips of a fair coin. -/
theorem probability_six_consecutive_heads :
  (favorableOutcomes : ℚ) / totalOutcomes = 129 / 1024 :=
sorry

end NUMINAMATH_CALUDE_probability_six_consecutive_heads_l3609_360950


namespace NUMINAMATH_CALUDE_min_longest_palindrome_length_l3609_360934

/-- A string consisting only of characters 'A' and 'B' -/
def ABString : Type := List Char

/-- Check if a string is a palindrome -/
def isPalindrome (s : ABString) : Prop :=
  s = s.reverse

/-- The length of the longest palindromic substring in an ABString -/
def longestPalindromeLength (s : ABString) : ℕ :=
  sorry

theorem min_longest_palindrome_length :
  (∀ s : ABString, s.length = 2021 → longestPalindromeLength s ≥ 4) ∧
  (∃ s : ABString, s.length = 2021 ∧ longestPalindromeLength s = 4) :=
sorry

end NUMINAMATH_CALUDE_min_longest_palindrome_length_l3609_360934


namespace NUMINAMATH_CALUDE_area_of_specific_isosceles_triangle_l3609_360999

/-- An isosceles triangle with given heights -/
structure IsoscelesTriangle where
  /-- Height dropped to the base -/
  baseHeight : ℝ
  /-- Height dropped to the lateral side -/
  lateralHeight : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True

/-- Calculate the area of an isosceles triangle given its heights -/
def areaOfIsoscelesTriangle (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles triangle is 75 -/
theorem area_of_specific_isosceles_triangle :
  let triangle : IsoscelesTriangle := {
    baseHeight := 10,
    lateralHeight := 12,
    isIsosceles := True.intro
  }
  areaOfIsoscelesTriangle triangle = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_isosceles_triangle_l3609_360999


namespace NUMINAMATH_CALUDE_greatest_y_l3609_360985

theorem greatest_y (y : ℕ) (h1 : y > 0) (h2 : ∃ k : ℕ, y = 4 * k) (h3 : y^3 < 8000) :
  y ≤ 16 ∧ ∃ y' : ℕ, y' > 0 ∧ (∃ k : ℕ, y' = 4 * k) ∧ y'^3 < 8000 ∧ y' = 16 :=
sorry

end NUMINAMATH_CALUDE_greatest_y_l3609_360985


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_equality_l3609_360976

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Add conditions for a valid triangle
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π
  -- Add triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_sine_cosine_equality (t : Triangle) :
  t.b * Real.sin t.β + t.a * Real.cos t.β * Real.sin t.γ =
  t.c * Real.sin t.γ + t.a * Real.cos t.γ * Real.sin t.β := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_equality_l3609_360976


namespace NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l3609_360908

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_inscribed_sphere_volume (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (a * b / (a + b + Real.sqrt (a^2 + b^2))) (h / 2)
  (4 / 3) * Real.pi * r^3 = (9 * Real.pi) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l3609_360908


namespace NUMINAMATH_CALUDE_campers_in_two_classes_l3609_360951

/-- Represents the number of campers in a single class -/
def class_size : ℕ := 20

/-- Represents the number of campers in all three classes -/
def in_all_classes : ℕ := 4

/-- Represents the number of campers in exactly one class -/
def in_one_class : ℕ := 24

/-- Represents the total number of campers -/
def total_campers : ℕ := class_size * 3 - 2 * in_all_classes

theorem campers_in_two_classes : 
  ∃ (x : ℕ), x = total_campers - in_one_class - in_all_classes ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_campers_in_two_classes_l3609_360951


namespace NUMINAMATH_CALUDE_problem_statement_l3609_360963

theorem problem_statement (M N : ℝ) 
  (h1 : (4 : ℝ) / 7 = M / 77)
  (h2 : (4 : ℝ) / 7 = 98 / (N^2)) : 
  M + N = 57.1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3609_360963


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_5_8_9_l3609_360926

theorem largest_three_digit_divisible_by_6_5_8_9 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≤ 720 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_6_5_8_9_l3609_360926


namespace NUMINAMATH_CALUDE_reciprocal_expression_l3609_360920

theorem reciprocal_expression (m n : ℝ) (h : m * n = 1) :
  (2 * m - 2 / n) * (1 / m + n) = 0 := by sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l3609_360920


namespace NUMINAMATH_CALUDE_change_in_quadratic_expression_l3609_360974

theorem change_in_quadratic_expression (x b : ℝ) (h : b > 0) :
  let f := fun x => 2 * x^2 + 5
  let change_plus := f (x + b) - f x
  let change_minus := f (x - b) - f x
  change_plus = 4 * x * b + 2 * b^2 ∧ change_minus = -4 * x * b + 2 * b^2 :=
by sorry

end NUMINAMATH_CALUDE_change_in_quadratic_expression_l3609_360974


namespace NUMINAMATH_CALUDE_exists_t_shape_l3609_360910

/-- Represents a grid of squares -/
structure Grid :=
  (size : ℕ)
  (removed : ℕ)

/-- Function that measures the connectivity of the grid -/
def f (g : Grid) : ℤ :=
  2 * g.size^2 - 4 * g.size - 10 * g.removed

/-- Theorem stating that after removing 1950 rectangles, 
    there always exists a square with at least three adjacent squares -/
theorem exists_t_shape (g : Grid) 
  (h1 : g.size = 100) 
  (h2 : g.removed = 1950) : 
  ∃ (square : Unit), f g > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_t_shape_l3609_360910


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l3609_360900

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l3609_360900


namespace NUMINAMATH_CALUDE_halloween_costume_payment_l3609_360932

theorem halloween_costume_payment (last_year_cost : ℝ) (price_increase_percent : ℝ) (deposit_percent : ℝ) : 
  last_year_cost = 250 →
  price_increase_percent = 40 →
  deposit_percent = 10 →
  let this_year_cost := last_year_cost * (1 + price_increase_percent / 100)
  let deposit := this_year_cost * (deposit_percent / 100)
  let remaining_payment := this_year_cost - deposit
  remaining_payment = 315 :=
by
  sorry

end NUMINAMATH_CALUDE_halloween_costume_payment_l3609_360932


namespace NUMINAMATH_CALUDE_bookstore_editions_l3609_360997

-- Define the universe of books in the bookstore
variable (Book : Type)

-- Define a predicate for new editions
variable (is_new_edition : Book → Prop)

-- Theorem statement
theorem bookstore_editions (h : ¬∀ (b : Book), is_new_edition b) :
  (∃ (b : Book), ¬is_new_edition b) ∧ (¬∀ (b : Book), is_new_edition b) := by
  sorry

end NUMINAMATH_CALUDE_bookstore_editions_l3609_360997


namespace NUMINAMATH_CALUDE_no_decreasing_h_for_increasing_f_l3609_360979

-- Define the function f in terms of h
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - x + 1) * h x

-- State the theorem
theorem no_decreasing_h_for_increasing_f :
  ¬ ∃ h : ℝ → ℝ,
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x ≤ y → h y ≤ h x) ∧
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x ≤ y → f h x ≤ f h y) :=
by sorry

end NUMINAMATH_CALUDE_no_decreasing_h_for_increasing_f_l3609_360979


namespace NUMINAMATH_CALUDE_tangent_line_t_range_l3609_360942

/-- A line tangent to a circle and intersecting a parabola at two points -/
structure TangentLineIntersectingParabola where
  k : ℝ
  t : ℝ
  tangent_condition : k^2 = t^2 + 2*t
  distinct_intersections : 16*(t^2 + 2*t) + 16*t > 0

/-- The range of t values for a tangent line intersecting a parabola at two points -/
theorem tangent_line_t_range (l : TangentLineIntersectingParabola) :
  l.t > 0 ∨ l.t < -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_t_range_l3609_360942


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_ratio_l3609_360983

theorem arithmetic_sequence_max_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * (a 1 + a n) / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 17 > 0 →
  S 18 < 0 →
  (∀ k ∈ Finset.range 15, S (k + 1) / a (k + 1) ≤ S 9 / a 9) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_ratio_l3609_360983


namespace NUMINAMATH_CALUDE_order_of_numbers_l3609_360945

def Ψ : ℤ := (1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + (-2012)) / 2

def Ω : ℤ := 1 - 2 + 3 - 4 + 2014

def Θ : ℤ := 1 - 3 + 5 - 7 + 2015

theorem order_of_numbers : Θ < Ω ∧ Ω < Ψ :=
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3609_360945


namespace NUMINAMATH_CALUDE_union_of_sets_l3609_360989

def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {a + 2, 5}

theorem union_of_sets (a : ℕ) (h : A ∩ B a = {3}) : A ∪ B a = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3609_360989


namespace NUMINAMATH_CALUDE_p_amount_l3609_360922

theorem p_amount (p q r : ℝ) : 
  p = (1/8 * p) + (1/8 * p) + 42 → p = 56 := by sorry

end NUMINAMATH_CALUDE_p_amount_l3609_360922


namespace NUMINAMATH_CALUDE_dodecagon_triangle_count_l3609_360939

/-- A regular dodecagon -/
structure RegularDodecagon where
  vertices : Finset ℕ
  regular : vertices.card = 12

/-- Count of triangles with specific properties in a regular dodecagon -/
def triangle_count (d : RegularDodecagon) : ℕ × ℕ :=
  let equilateral := 4  -- Number of equilateral triangles
  let scalene := 168    -- Number of scalene triangles
  (equilateral, scalene)

/-- Theorem stating the correct count of equilateral and scalene triangles in a regular dodecagon -/
theorem dodecagon_triangle_count (d : RegularDodecagon) :
  triangle_count d = (4, 168) := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_triangle_count_l3609_360939


namespace NUMINAMATH_CALUDE_xyz_sum_l3609_360905

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 12)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 28) :
  x*y + y*z + x*z = 16 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l3609_360905


namespace NUMINAMATH_CALUDE_square_area_is_16_l3609_360995

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the horizontal line
def horizontal_line : ℝ := 3

-- Theorem statement
theorem square_area_is_16 : ∃ (x₁ x₂ : ℝ),
  x₁ ≠ x₂ ∧
  parabola x₁ = horizontal_line ∧
  parabola x₂ = horizontal_line ∧
  (x₂ - x₁)^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_square_area_is_16_l3609_360995


namespace NUMINAMATH_CALUDE_carbon_count_in_compound_l3609_360968

/-- Represents the atomic weights of elements in atomic mass units (amu) -/
structure AtomicWeights where
  copper : ℝ
  carbon : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound given its composition -/
def molecularWeight (weights : AtomicWeights) (copperCount : ℕ) (carbonCount : ℕ) (oxygenCount : ℕ) : ℝ :=
  weights.copper * copperCount + weights.carbon * carbonCount + weights.oxygen * oxygenCount

/-- Theorem stating that a compound with 1 Copper, n Carbon, and 3 Oxygen atoms
    with a molecular weight of 124 amu has 1 Carbon atom -/
theorem carbon_count_in_compound (weights : AtomicWeights) 
    (h1 : weights.copper = 63.55)
    (h2 : weights.carbon = 12.01)
    (h3 : weights.oxygen = 16.00) :
  ∃ (n : ℕ), molecularWeight weights 1 n 3 = 124 ∧ n = 1 := by
  sorry


end NUMINAMATH_CALUDE_carbon_count_in_compound_l3609_360968


namespace NUMINAMATH_CALUDE_percentage_to_pass_l3609_360917

/-- The percentage needed to pass an exam, given the achieved score, shortfall, and maximum possible marks. -/
theorem percentage_to_pass 
  (achieved_score : ℕ) 
  (shortfall : ℕ) 
  (max_marks : ℕ) 
  (h1 : achieved_score = 212)
  (h2 : shortfall = 28)
  (h3 : max_marks = 800) : 
  (achieved_score + shortfall) / max_marks * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l3609_360917


namespace NUMINAMATH_CALUDE_diamond_example_l3609_360906

/-- The diamond operation -/
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

/-- Theorem stating the result of (3 ◇ 4) ◇ 2 -/
theorem diamond_example : diamond (diamond 3 4) 2 = 179 := by
  sorry

end NUMINAMATH_CALUDE_diamond_example_l3609_360906
