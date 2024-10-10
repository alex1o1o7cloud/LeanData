import Mathlib

namespace triangle_properties_l883_88375

/-- Given a triangle ABC with specific properties, prove its angle B and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b * Real.cos C = (2 * a - c) * Real.cos B →
  b = Real.sqrt 7 →
  a + c = 4 →
  B = π / 3 ∧ 
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 :=
by sorry

end triangle_properties_l883_88375


namespace book_ratio_l883_88328

/-- The number of books Pete read last year -/
def P : ℕ := sorry

/-- The number of books Matt read last year -/
def M : ℕ := sorry

/-- Pete doubles his reading this year -/
axiom pete_doubles : P * 2 = 300 - P

/-- Matt reads 50% more this year -/
axiom matt_increases : M * 3/2 = 75

/-- Pete read 300 books across both years -/
axiom pete_total : P + P * 2 = 300

/-- Matt read 75 books in his second year -/
axiom matt_second_year : M * 3/2 = 75

/-- The ratio of books Pete read last year to books Matt read last year is 2:1 -/
theorem book_ratio : P / M = 2 := by sorry

end book_ratio_l883_88328


namespace fraction_equality_l883_88318

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 1 / 3) :
  t / q = 2 := by sorry

end fraction_equality_l883_88318


namespace matrix_equality_l883_88337

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![20/3, 4/3], ![-8/3, 8/3]]) : 
  B * A = ![![20/3, 4/3], ![-8/3, 8/3]] := by
  sorry

end matrix_equality_l883_88337


namespace students_with_puppies_and_parrots_l883_88322

theorem students_with_puppies_and_parrots 
  (total_students : ℕ) 
  (puppy_percentage : ℚ) 
  (parrot_percentage : ℚ) 
  (h1 : total_students = 40)
  (h2 : puppy_percentage = 80 / 100)
  (h3 : parrot_percentage = 25 / 100) :
  ⌊(total_students : ℚ) * puppy_percentage * parrot_percentage⌋ = 8 := by
  sorry

end students_with_puppies_and_parrots_l883_88322


namespace geometric_progression_first_term_l883_88373

/-- A geometric progression with sum to infinity 8 and sum of first three terms 7 has first term 4 -/
theorem geometric_progression_first_term :
  ∀ (a r : ℝ),
  (a / (1 - r) = 8) →  -- sum to infinity
  (a + a*r + a*r^2 = 7) →  -- sum of first three terms
  a = 4 := by
sorry

end geometric_progression_first_term_l883_88373


namespace symmetric_point_about_x_axis_l883_88371

/-- Given a point M with coordinates (3,-4), its symmetric point M' about the x-axis has coordinates (3,4). -/
theorem symmetric_point_about_x_axis :
  let M : ℝ × ℝ := (3, -4)
  let M' : ℝ × ℝ := (M.1, -M.2)
  M' = (3, 4) := by sorry

end symmetric_point_about_x_axis_l883_88371


namespace smallest_c_inequality_l883_88340

theorem smallest_c_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (∀ c : ℝ, c > 0 → c * |x^(2/3) - y^(2/3)| + (x*y)^(1/3) ≥ (x^(2/3) + y^(2/3))/2) ∧
  (∀ ε > 0, ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ ((1/2 - ε) * |x^(2/3) - y^(2/3)| + (x*y)^(1/3) < (x^(2/3) + y^(2/3))/2)) :=
by sorry

end smallest_c_inequality_l883_88340


namespace video_votes_l883_88386

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 120 ∧ 
  like_percentage = 3/4 ∧ 
  (∀ (total_votes : ℕ), 
    (↑total_votes : ℚ) * like_percentage - (↑total_votes : ℚ) * (1 - like_percentage) = score) →
  ∃ (total_votes : ℕ), total_votes = 240 := by
sorry

end video_votes_l883_88386


namespace stone_blocks_per_step_l883_88380

theorem stone_blocks_per_step 
  (levels : ℕ) 
  (steps_per_level : ℕ) 
  (total_blocks : ℕ) 
  (h1 : levels = 4) 
  (h2 : steps_per_level = 8) 
  (h3 : total_blocks = 96) : 
  total_blocks / (levels * steps_per_level) = 3 := by
sorry

end stone_blocks_per_step_l883_88380


namespace curve_satisfies_conditions_l883_88319

/-- The curve that satisfies the given conditions -/
def curve (x y : ℝ) : Prop := x * y = 4

/-- The tangent line to the curve at point (x,y) -/
def tangent_line (x y : ℝ) : Set (ℝ × ℝ) :=
  {(t, s) | s - y = -(y / x) * (t - x)}

theorem curve_satisfies_conditions :
  -- The curve passes through (1,4)
  curve 1 4 ∧
  -- For any point (x,y) on the curve, the tangent line intersects
  -- the x-axis at (2x,0) and the y-axis at (0,2y)
  ∀ x y : ℝ, x > 0 → y > 0 → curve x y →
    (2*x, 0) ∈ tangent_line x y ∧ (0, 2*y) ∈ tangent_line x y :=
by sorry


end curve_satisfies_conditions_l883_88319


namespace least_x_for_even_prime_fraction_l883_88346

theorem least_x_for_even_prime_fraction (x p : ℕ) : 
  x > 0 → 
  Prime p → 
  Prime (x / (12 * p)) → 
  Even (x / (12 * p)) → 
  (∀ y : ℕ, y > 0 ∧ (∃ q : ℕ, Prime q ∧ Prime (y / (12 * q)) ∧ Even (y / (12 * q))) → x ≤ y) → 
  x = 48 := by
sorry

end least_x_for_even_prime_fraction_l883_88346


namespace no_squarish_numbers_l883_88366

/-- A number is squarish if it satisfies all the given conditions -/
def is_squarish (n : ℕ) : Prop :=
  -- Six-digit number
  100000 ≤ n ∧ n < 1000000 ∧
  -- Each digit between 1 and 8
  (∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 8) ∧
  -- Perfect square
  ∃ x, n = x^2 ∧
  -- First two digits are a perfect square
  ∃ y, (n / 10000) = y^2 ∧
  -- Middle two digits are a perfect square and divisible by 2
  ∃ z, ((n / 100) % 100) = z^2 ∧ ((n / 100) % 100) % 2 = 0 ∧
  -- Last two digits are a perfect square
  ∃ w, (n % 100) = w^2

theorem no_squarish_numbers : ¬∃ n, is_squarish n := by
  sorry

end no_squarish_numbers_l883_88366


namespace ellipse_standard_equation_l883_88381

/-- An ellipse passing through two given points with a focus on a coordinate axis. -/
structure Ellipse where
  -- The coefficients of the ellipse equation x²/a² + y²/b² = 1
  a : ℝ
  b : ℝ
  -- Condition: a > 0 and b > 0
  ha : a > 0
  hb : b > 0
  -- Condition: Passes through P1(-√6, 1)
  passes_p1 : 6 / a^2 + 1 / b^2 = 1
  -- Condition: Passes through P2(√3, -√2)
  passes_p2 : 3 / a^2 + 2 / b^2 = 1
  -- Condition: One focus on coordinate axis, perpendicular to minor axis vertices, passes through (-3, 3√2/2)
  focus_condition : 9 / a^2 + (9/2) / b^2 = 1

/-- The standard equation of the ellipse satisfies one of the given forms. -/
theorem ellipse_standard_equation (e : Ellipse) : 
  (e.a^2 = 9 ∧ e.b^2 = 3) ∨ 
  (e.a^2 = 18 ∧ e.b^2 = 9) ∨ 
  (e.a^2 = 45/4 ∧ e.b^2 = 45/2) :=
sorry

end ellipse_standard_equation_l883_88381


namespace joe_egg_count_l883_88313

/-- The number of eggs Joe found around the club house -/
def club_house_eggs : ℕ := 12

/-- The number of eggs Joe found around the park -/
def park_eggs : ℕ := 5

/-- The number of eggs Joe found in the town hall garden -/
def town_hall_eggs : ℕ := 3

/-- The total number of eggs Joe found -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs

theorem joe_egg_count : total_eggs = 20 := by
  sorry

end joe_egg_count_l883_88313


namespace truck_filling_problem_l883_88310

/-- A problem about filling a truck with stone blocks -/
theorem truck_filling_problem 
  (truck_capacity : ℕ) 
  (initial_workers : ℕ) 
  (work_rate : ℕ) 
  (initial_work_time : ℕ) 
  (total_time : ℕ)
  (h1 : truck_capacity = 6000)
  (h2 : initial_workers = 2)
  (h3 : work_rate = 250)
  (h4 : initial_work_time = 4)
  (h5 : total_time = 6)
  : ∃ (joined_workers : ℕ),
    (initial_workers * work_rate * initial_work_time) + 
    ((initial_workers + joined_workers) * work_rate * (total_time - initial_work_time)) = 
    truck_capacity ∧ joined_workers = 6 := by
  sorry


end truck_filling_problem_l883_88310


namespace complete_square_sum_l883_88370

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (49 * x^2 + 70 * x - 81 = 0 ↔ (a * x + b)^2 = c) ∧ 
  a > 0 ∧ 
  a + b + c = -44 := by
  sorry

end complete_square_sum_l883_88370


namespace hyperbola_asymptotic_lines_l883_88331

/-- Given a hyperbola with equation 9x^2 - 16y^2 = 144, 
    its asymptotic lines are y = ± 3/4 x -/
theorem hyperbola_asymptotic_lines :
  let hyperbola := {(x, y) : ℝ × ℝ | 9 * x^2 - 16 * y^2 = 144}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = 3/4 * x ∨ y = -3/4 * x}
  asymptotic_lines = {(x, y) : ℝ × ℝ | ∃ (t : ℝ), t ≠ 0 ∧ (t*x, t*y) ∈ hyperbola} :=
by sorry

end hyperbola_asymptotic_lines_l883_88331


namespace book_sale_profit_l883_88316

/-- Calculates the percent profit for a book sale given the cost, markup percentage, and discount percentage. -/
theorem book_sale_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  cost = 50 ∧ markup_percent = 30 ∧ discount_percent = 10 →
  (((cost * (1 + markup_percent / 100)) * (1 - discount_percent / 100) - cost) / cost) * 100 = 17 := by
sorry

end book_sale_profit_l883_88316


namespace quadratic_min_max_l883_88350

theorem quadratic_min_max (x : ℝ) (n : ℝ) :
  (∀ x, x^2 - 4*x - 3 ≥ -7) ∧
  (n = 6 - x → ∀ x, Real.sqrt (x^2 - 2*n^2) ≤ 6 * Real.sqrt 2) := by
  sorry

end quadratic_min_max_l883_88350


namespace simplify_expression_l883_88394

theorem simplify_expression (a b : ℝ) : (2*a - b) - 2*(a - 2*b) = 3*b := by
  sorry

end simplify_expression_l883_88394


namespace subcommittee_count_l883_88377

def total_members : ℕ := 12
def num_teachers : ℕ := 5
def subcommittee_size : ℕ := 5

def valid_subcommittees : ℕ := Nat.choose total_members subcommittee_size - Nat.choose (total_members - num_teachers) subcommittee_size

theorem subcommittee_count :
  valid_subcommittees = 771 :=
sorry

end subcommittee_count_l883_88377


namespace no_real_roots_iff_b_positive_l883_88304

/-- The polynomial has no real roots if and only if b is positive -/
theorem no_real_roots_iff_b_positive (b : ℝ) : 
  (∀ x : ℝ, x^4 + b*x^3 - 2*x^2 + b*x + 2 ≠ 0) ↔ b > 0 := by
  sorry

end no_real_roots_iff_b_positive_l883_88304


namespace triple_application_of_f_l883_88330

def f (p : ℝ) : ℝ := 2 * p + 20

theorem triple_application_of_f :
  ∃ p : ℝ, f (f (f p)) = -4 ∧ p = -18 := by
  sorry

end triple_application_of_f_l883_88330


namespace caterpillar_insane_bill_sane_l883_88368

-- Define the mental state of a character
inductive MentalState
| Sane
| Insane

-- Define the characters
structure Character where
  name : String
  state : MentalState

-- Define the Caterpillar's belief
def caterpillarBelief (caterpillar : Character) (bill : Character) : Prop :=
  caterpillar.state = MentalState.Insane ∧ bill.state = MentalState.Insane

-- Theorem statement
theorem caterpillar_insane_bill_sane 
  (caterpillar : Character) 
  (bill : Character) 
  (h : caterpillarBelief caterpillar bill) : 
  caterpillar.state = MentalState.Insane ∧ bill.state = MentalState.Sane :=
sorry

end caterpillar_insane_bill_sane_l883_88368


namespace min_cuboid_height_l883_88392

/-- Represents a cuboid with a square base -/
structure Cuboid where
  base_side : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- The minimum height of a cuboid that can contain given spheres -/
def min_height (base_side : ℝ) (small_spheres : List Sphere) (large_sphere : Sphere) : ℝ :=
  sorry

theorem min_cuboid_height :
  let cuboid : Cuboid := { base_side := 4, height := min_height 4 (List.replicate 8 { radius := 1 }) { radius := 2 } }
  let small_spheres : List Sphere := List.replicate 8 { radius := 1 }
  let large_sphere : Sphere := { radius := 2 }
  cuboid.height = 2 + 2 * Real.sqrt 7 :=
by sorry

end min_cuboid_height_l883_88392


namespace corner_sum_is_164_l883_88300

/-- Represents a 9x9 checkerboard with numbers from 1 to 81 -/
def Checkerboard : Type := Fin 9 → Fin 9 → Nat

/-- The number at position (i, j) on the checkerboard -/
def number_at (board : Checkerboard) (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The sum of the numbers in the four corners of the checkerboard -/
def corner_sum (board : Checkerboard) : Nat :=
  number_at board 0 0 +
  number_at board 0 8 +
  number_at board 8 0 +
  number_at board 8 8

/-- The theorem stating that the sum of the numbers in the four corners is 164 -/
theorem corner_sum_is_164 (board : Checkerboard) : corner_sum board = 164 := by
  sorry

end corner_sum_is_164_l883_88300


namespace solution_of_equation_l883_88367

theorem solution_of_equation (x : ℝ) : 
  (3 / (x + 2) - 1 / x = 0) ↔ (x = 1) :=
by sorry

end solution_of_equation_l883_88367


namespace geometric_sequence_condition_iff_strictly_increasing_l883_88327

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The condition that a_{n+2} > a_n for all positive integers n -/
def Condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) > a n

/-- The sequence is strictly increasing -/
def StrictlyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_condition_iff_strictly_increasing
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  Condition a ↔ StrictlyIncreasing a :=
sorry

end geometric_sequence_condition_iff_strictly_increasing_l883_88327


namespace investment_ratio_is_two_to_three_l883_88393

/-- A partnership problem with three investors A, B, and C. -/
structure Partnership where
  /-- B's investment amount -/
  b_investment : ℝ
  /-- Total profit earned -/
  total_profit : ℝ
  /-- B's share of the profit -/
  b_share : ℝ
  /-- A's investment is 3 times B's investment -/
  a_investment_prop : ℝ := 3 * b_investment
  /-- Assumption that total_profit and b_share are positive -/
  h_positive : 0 < total_profit ∧ 0 < b_share

/-- The ratio of B's investment to C's investment in the partnership -/
def investment_ratio (p : Partnership) : ℚ × ℚ :=
  (2, 3)

/-- Theorem stating that the investment ratio is 2:3 given the partnership conditions -/
theorem investment_ratio_is_two_to_three (p : Partnership)
  (h1 : p.total_profit = 3300)
  (h2 : p.b_share = 600) :
  investment_ratio p = (2, 3) := by
  sorry

#check investment_ratio_is_two_to_three

end investment_ratio_is_two_to_three_l883_88393


namespace strawberry_cost_l883_88306

theorem strawberry_cost (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 301 → 
  7 * J * N / 100 = 196 / 100 :=
by sorry

end strawberry_cost_l883_88306


namespace red_squares_less_than_half_l883_88342

/-- Represents a cube with side length 3, composed of 27 unit cubes -/
structure LargeCube where
  total_units : Nat
  red_units : Nat
  blue_units : Nat
  side_length : Nat

/-- Calculates the total number of visible unit squares on the surface of the large cube -/
def total_surface_squares (cube : LargeCube) : Nat :=
  6 * (cube.side_length * cube.side_length)

/-- Calculates the maximum number of red unit squares that can be visible on the surface -/
def max_red_surface_squares (cube : LargeCube) : Nat :=
  (cube.side_length - 1) * (cube.side_length - 1) * 3 + (cube.side_length - 1) * 3 * 2 + 8 * 3

/-- Theorem stating that the maximum number of red squares on the surface is less than half the total -/
theorem red_squares_less_than_half (cube : LargeCube) 
  (h1 : cube.total_units = 27)
  (h2 : cube.red_units = 9)
  (h3 : cube.blue_units = 18)
  (h4 : cube.side_length = 3)
  : max_red_surface_squares cube < (total_surface_squares cube) / 2 := by
  sorry

end red_squares_less_than_half_l883_88342


namespace density_difference_of_cubes_l883_88341

theorem density_difference_of_cubes (m₁ : ℝ) (a₁ : ℝ) (m₁_pos : m₁ > 0) (a₁_pos : a₁ > 0) :
  let m₂ := 0.75 * m₁
  let a₂ := 1.25 * a₁
  let ρ₁ := m₁ / (a₁^3)
  let ρ₂ := m₂ / (a₂^3)
  (ρ₁ - ρ₂) / ρ₁ = 0.616 := by
sorry

end density_difference_of_cubes_l883_88341


namespace book_collection_ratio_l883_88324

theorem book_collection_ratio : ∀ (L S : ℕ), 
  L + S = 3000 →  -- Total books
  S = 600 →       -- Susan's books
  L / S = 4       -- Ratio of Lidia's to Susan's books
  := by sorry

end book_collection_ratio_l883_88324


namespace total_spent_usd_value_l883_88359

/-- The total amount spent on souvenirs in US dollars -/
def total_spent_usd (key_chain_bracelet_cost : ℝ) (tshirt_cost_diff : ℝ) 
  (tshirt_discount : ℝ) (key_chain_tax : ℝ) (bracelet_tax : ℝ) 
  (conversion_rate : ℝ) : ℝ :=
  let tshirt_cost := key_chain_bracelet_cost - tshirt_cost_diff
  let tshirt_actual := tshirt_cost * (1 - tshirt_discount)
  let key_chain_bracelet_actual := key_chain_bracelet_cost * (1 + key_chain_tax + bracelet_tax)
  (tshirt_actual + key_chain_bracelet_actual) * conversion_rate

/-- Theorem stating the total amount spent on souvenirs in US dollars -/
theorem total_spent_usd_value :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_spent_usd 347 146 0.1 0.12 0.08 0.75 - 447.98| < ε :=
sorry

end total_spent_usd_value_l883_88359


namespace product_mod_nine_l883_88372

theorem product_mod_nine : (98 * 102) % 9 = 6 := by
  sorry

end product_mod_nine_l883_88372


namespace f_derivative_at_one_l883_88363

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_one : 
  (deriv f) 1 = 24 := by sorry

end f_derivative_at_one_l883_88363


namespace pizza_pooling_benefit_l883_88339

/-- Represents a square pizza with side length and price --/
structure Pizza where
  side : ℕ
  price : ℕ

/-- Calculates the area of a square pizza --/
def pizzaArea (p : Pizza) : ℕ := p.side * p.side

/-- Calculates the number of pizzas that can be bought with a given amount of money --/
def pizzaCount (p : Pizza) (money : ℕ) : ℕ := money / p.price

/-- The small pizza option --/
def smallPizza : Pizza := { side := 6, price := 10 }

/-- The large pizza option --/
def largePizza : Pizza := { side := 9, price := 20 }

/-- The amount of money each friend has --/
def individualMoney : ℕ := 30

/-- The total amount of money when pooled --/
def pooledMoney : ℕ := 2 * individualMoney

theorem pizza_pooling_benefit :
  pizzaArea largePizza * pizzaCount largePizza pooledMoney -
  2 * (pizzaArea smallPizza * pizzaCount smallPizza individualMoney) = 135 := by
  sorry

end pizza_pooling_benefit_l883_88339


namespace x_squared_coefficient_l883_88302

/-- The coefficient of x² in the expansion of (3x² + 4x + 5)(6x² + 7x + 8) is 82 -/
theorem x_squared_coefficient (x : ℝ) : 
  (3*x^2 + 4*x + 5) * (6*x^2 + 7*x + 8) = 18*x^4 + 39*x^3 + 82*x^2 + 67*x + 40 := by
  sorry

end x_squared_coefficient_l883_88302


namespace valid_pairs_l883_88326

def is_valid_pair (m n : ℕ+) : Prop :=
  (3^m.val + 1) % (m.val * n.val) = 0 ∧ (3^n.val + 1) % (m.val * n.val) = 0

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ 
    ((m = 1 ∧ n = 1) ∨ 
     (m = 1 ∧ n = 2) ∨ 
     (m = 1 ∧ n = 4) ∨ 
     (m = 2 ∧ n = 1) ∨ 
     (m = 4 ∧ n = 1)) :=
by sorry

end valid_pairs_l883_88326


namespace animal_jumping_distances_l883_88347

-- Define the jumping distances for each animal
def grasshopper_jump : ℕ := 36

def frog_jump : ℕ := grasshopper_jump + 17

def mouse_jump : ℕ := frog_jump + 15

def kangaroo_jump : ℕ := 2 * mouse_jump

def rabbit_jump : ℕ := kangaroo_jump / 2 - 12

-- Theorem to prove the jumping distances
theorem animal_jumping_distances :
  grasshopper_jump = 36 ∧
  frog_jump = 53 ∧
  mouse_jump = 68 ∧
  kangaroo_jump = 136 ∧
  rabbit_jump = 56 := by
  sorry


end animal_jumping_distances_l883_88347


namespace speed_conversion_l883_88395

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The speed in meters per second -/
def speed_mps : ℝ := 50

/-- Theorem: Converting 50 mps to kmph equals 180 kmph -/
theorem speed_conversion : speed_mps * mps_to_kmph = 180 := by sorry

end speed_conversion_l883_88395


namespace solution_difference_l883_88321

theorem solution_difference (a b : ℝ) : 
  a ≠ b ∧ 
  (6 * a - 18) / (a^2 + 3 * a - 18) = a + 3 ∧
  (6 * b - 18) / (b^2 + 3 * b - 18) = b + 3 ∧
  a > b →
  a - b = 3 := by sorry

end solution_difference_l883_88321


namespace maximum_marks_l883_88397

/-- 
Given:
1. The passing mark is 36% of the maximum marks.
2. A student gets 130 marks and fails by 14 marks.
Prove that the maximum number of marks is 400.
-/
theorem maximum_marks (passing_percentage : ℚ) (student_marks : ℕ) (failing_margin : ℕ) :
  passing_percentage = 36 / 100 →
  student_marks = 130 →
  failing_margin = 14 →
  ∃ (max_marks : ℕ), max_marks = 400 ∧ 
    (student_marks + failing_margin : ℚ) = passing_percentage * max_marks :=
by sorry

end maximum_marks_l883_88397


namespace arithmetic_sequence_sum_property_l883_88355

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n, S n = n * (a 0 + a (n - 1)) / 2

/-- Theorem stating that if S_2 = 3 and S_4 = 15, then S_6 = 63 for an arithmetic sequence -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
    (h1 : seq.S 2 = 3) (h2 : seq.S 4 = 15) : seq.S 6 = 63 := by
  sorry

end arithmetic_sequence_sum_property_l883_88355


namespace defective_units_percentage_l883_88325

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.05)
  (h2 : total_shipped_defective_ratio = 0.0035) :
  ∃ (defective_ratio : Real),
    defective_ratio = 0.07 ∧
    shipped_defective_ratio * defective_ratio = total_shipped_defective_ratio :=
by sorry

end defective_units_percentage_l883_88325


namespace lizette_stamps_l883_88303

/-- Given that Lizette has 125 more stamps than Minerva and Minerva has 688 stamps,
    prove that Lizette has 813 stamps. -/
theorem lizette_stamps (minerva_stamps : ℕ) (lizette_extra : ℕ) 
  (h1 : minerva_stamps = 688)
  (h2 : lizette_extra = 125) : 
  minerva_stamps + lizette_extra = 813 := by
  sorry

end lizette_stamps_l883_88303


namespace unique_solution_l883_88323

def equation1 (x y : ℝ) : Prop := 3 * x + 4 * y = 26

def equation2 (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 1)^2) + Real.sqrt ((x - 10)^2 + (y - 5)^2) = 10

theorem unique_solution :
  ∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2 ∧ p = (6, 2) := by
  sorry

end unique_solution_l883_88323


namespace mary_max_earnings_l883_88312

/-- Calculates Mary's weekly earnings based on her work hours and pay rates. -/
def maryEarnings (maxHours : Nat) (regularRate : ℚ) (overtimeRate : ℚ) (additionalRate : ℚ) : ℚ :=
  let regularHours := min maxHours 40
  let overtimeHours := min (maxHours - regularHours) 20
  let additionalHours := maxHours - regularHours - overtimeHours
  regularHours * regularRate + overtimeHours * overtimeRate + additionalHours * additionalRate

/-- Theorem stating Mary's earnings for working the maximum hours in a week. -/
theorem mary_max_earnings :
  let maxHours : Nat := 70
  let regularRate : ℚ := 10
  let overtimeRate : ℚ := regularRate * (1 + 30/100)
  let additionalRate : ℚ := regularRate * (1 + 60/100)
  maryEarnings maxHours regularRate overtimeRate additionalRate = 820 := by
  sorry


end mary_max_earnings_l883_88312


namespace sin_symmetry_l883_88314

theorem sin_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x + π / 3)
  let g (x : ℝ) := f (x - π / 12)
  ∀ t, g ((-π / 12) + t) = g ((-π / 12) - t) :=
by sorry

end sin_symmetry_l883_88314


namespace largest_valid_number_l883_88389

def is_valid (n : ℕ) : Prop :=
  n < 10000 ∧
  (∃ a : ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b : ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c : ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1))

theorem largest_valid_number : 
  is_valid 4369 ∧ ∀ m : ℕ, m > 4369 → ¬(is_valid m) :=
by sorry

end largest_valid_number_l883_88389


namespace impossibleToKnowDreamIfDiedAsleep_l883_88396

/-- Represents a person's state --/
inductive PersonState
  | Awake
  | Asleep
  | Dead

/-- Represents a dream --/
structure Dream where
  content : String

/-- Represents a person --/
structure Person where
  state : PersonState
  currentDream : Option Dream

/-- Represents the ability to share dream content --/
def canShareDream (p : Person) : Prop :=
  p.state = PersonState.Awake ∧ p.currentDream.isSome

/-- Represents the event of a person dying while asleep --/
def diedWhileAsleep (p : Person) : Prop :=
  p.state = PersonState.Dead ∧ p.currentDream.isSome

/-- Theorem: If a person died while asleep, it's impossible for others to know their exact dream --/
theorem impossibleToKnowDreamIfDiedAsleep (p : Person) :
  diedWhileAsleep p → ¬(canShareDream p) :=
by
  sorry

end impossibleToKnowDreamIfDiedAsleep_l883_88396


namespace janes_mean_score_l883_88315

def janes_scores : List ℝ := [98, 97, 92, 85, 93]

theorem janes_mean_score :
  (janes_scores.sum / janes_scores.length : ℝ) = 93 := by
  sorry

end janes_mean_score_l883_88315


namespace stratified_sampling_red_balls_l883_88369

/-- Given a set of 100 balls with 20 red balls, prove that a stratified sample of 10 balls should contain 2 red balls. -/
theorem stratified_sampling_red_balls 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (sample_size : ℕ) 
  (h_total : total_balls = 100) 
  (h_red : red_balls = 20) 
  (h_sample : sample_size = 10) : 
  (red_balls : ℚ) / total_balls * sample_size = 2 := by
  sorry

end stratified_sampling_red_balls_l883_88369


namespace tommy_wheel_count_l883_88329

/-- The number of wheels on each truck -/
def truck_wheels : ℕ := 4

/-- The number of wheels on each car -/
def car_wheels : ℕ := 4

/-- The number of trucks Tommy saw -/
def trucks_seen : ℕ := 12

/-- The number of cars Tommy saw -/
def cars_seen : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := truck_wheels * trucks_seen + car_wheels * cars_seen

theorem tommy_wheel_count : total_wheels = 100 := by
  sorry

end tommy_wheel_count_l883_88329


namespace arithmetic_sequence_average_l883_88376

theorem arithmetic_sequence_average (a₁ aₙ d : ℚ) (n : ℕ) (h₁ : a₁ = 15) (h₂ : aₙ = 35) (h₃ : d = 1/4) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  (n * (a₁ + aₙ)) / (2 * n) = 25 :=
sorry

end arithmetic_sequence_average_l883_88376


namespace trigonometric_expression_equals_one_l883_88345

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1 := by
  sorry

end trigonometric_expression_equals_one_l883_88345


namespace three_part_division_l883_88385

theorem three_part_division (total : ℚ) (p1 p2 p3 : ℚ) (h1 : total = 78)
  (h2 : p1 + p2 + p3 = total) (h3 : p2 = (1/3) * p1) (h4 : p3 = (1/6) * p1) :
  p2 = 17 + (1/3) :=
by sorry

end three_part_division_l883_88385


namespace tangent_line_condition_l883_88362

/-- Given a function f(x) = x³ + ax², prove that if the tangent line
    at point (x₀, f(x₀)) has equation x + y = 0, then x₀ = ±1 and f(x₀) = -x₀ -/
theorem tangent_line_condition (a : ℝ) :
  ∃ x₀ : ℝ, (x₀ = 1 ∨ x₀ = -1) ∧
  let f := λ x : ℝ => x^3 + a*x^2
  let f' := λ x : ℝ => 3*x^2 + 2*a*x
  f' x₀ = -1 ∧ x₀ + f x₀ = 0 := by
sorry


end tangent_line_condition_l883_88362


namespace first_group_size_l883_88348

/-- The number of beavers in the first group -/
def first_group : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def time_first_group : ℕ := 3

/-- The number of beavers in the second group -/
def second_group : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def time_second_group : ℕ := 5

/-- Theorem stating that the first group consists of 20 beavers -/
theorem first_group_size :
  first_group * time_first_group = second_group * time_second_group :=
by sorry

end first_group_size_l883_88348


namespace inscribed_rectangle_area_l883_88309

/-- A triangle with an inscribed rectangle -/
structure InscribedRectangle where
  /-- The height of the triangle -/
  triangle_height : ℝ
  /-- The base of the triangle -/
  triangle_base : ℝ
  /-- The width of the inscribed rectangle -/
  rectangle_width : ℝ
  /-- The length of the inscribed rectangle -/
  rectangle_length : ℝ
  /-- The width of the rectangle is one-third of its length -/
  width_is_third_of_length : rectangle_width = rectangle_length / 3
  /-- The rectangle is inscribed in the triangle -/
  rectangle_inscribed : rectangle_length ≤ triangle_base

/-- The area of the inscribed rectangle given the triangle's dimensions -/
def rectangle_area (r : InscribedRectangle) : ℝ :=
  r.rectangle_width * r.rectangle_length

/-- Theorem: The area of the inscribed rectangle is 675/64 square inches -/
theorem inscribed_rectangle_area (r : InscribedRectangle)
    (h1 : r.triangle_height = 9)
    (h2 : r.triangle_base = 15) :
    rectangle_area r = 675 / 64 := by
  sorry

end inscribed_rectangle_area_l883_88309


namespace quadratic_equation_coefficients_l883_88308

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns the triple (a, b, c) -/
def quadraticCoefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem quadratic_equation_coefficients :
  quadraticCoefficients (fun x => x^2 - x) = (1, -1, 0) := by
  sorry

end quadratic_equation_coefficients_l883_88308


namespace f_of_f_of_2_equals_394_l883_88344

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 6

-- State the theorem
theorem f_of_f_of_2_equals_394 : f (f 2) = 394 := by
  sorry

end f_of_f_of_2_equals_394_l883_88344


namespace absolute_value_of_c_l883_88360

theorem absolute_value_of_c (a b c : ℤ) : 
  a * (3 + I : ℂ)^4 + b * (3 + I : ℂ)^3 + c * (3 + I : ℂ)^2 + b * (3 + I : ℂ) + a = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 116 :=
by sorry

end absolute_value_of_c_l883_88360


namespace smallest_q_value_l883_88335

theorem smallest_q_value (p q : ℕ+) 
  (h1 : (72 : ℚ) / 487 < p.val / q.val)
  (h2 : p.val / q.val < (18 : ℚ) / 121) :
  ∀ (q' : ℕ+), ((72 : ℚ) / 487 < p.val / q'.val ∧ p.val / q'.val < (18 : ℚ) / 121) → q.val ≤ q'.val →
  q.val = 27 :=
sorry

end smallest_q_value_l883_88335


namespace parity_of_sum_of_powers_l883_88378

theorem parity_of_sum_of_powers : Even (1^1994 + 9^1994 + 8^1994 + 6^1994) := by
  sorry

end parity_of_sum_of_powers_l883_88378


namespace boat_distance_proof_l883_88338

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

theorem boat_distance_proof (boat_speed stream_speed time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 5)
  (h3 : time = 3) :
  distance_downstream boat_speed stream_speed time = 63 := by
  sorry

end boat_distance_proof_l883_88338


namespace min_coin_tosses_l883_88384

theorem min_coin_tosses (n : ℕ) : (1 - (1/2)^n ≥ 15/16) ↔ n ≥ 4 := by sorry

end min_coin_tosses_l883_88384


namespace sarah_wallet_ones_l883_88301

/-- Represents the contents of Sarah's wallet -/
structure Wallet where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The wallet satisfies the given conditions -/
def valid_wallet (w : Wallet) : Prop :=
  w.ones + w.twos + w.fives = 50 ∧
  w.ones + 2 * w.twos + 5 * w.fives = 146

theorem sarah_wallet_ones :
  ∃ w : Wallet, valid_wallet w ∧ w.ones = 14 := by
  sorry

end sarah_wallet_ones_l883_88301


namespace quadratic_equation_special_roots_l883_88387

/-- 
Given a quadratic equation x^2 + px + q = 0 with roots D and 1-D, 
where D is the discriminant of the equation, 
prove that the only possible values for (p, q) are (-1, 0) and (-1, 3/16).
-/
theorem quadratic_equation_special_roots (p q D : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = D ∨ x = 1 - D) ∧ 
  D^2 = p^2 - 4*q →
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3/16) := by
sorry

end quadratic_equation_special_roots_l883_88387


namespace die_roll_counts_l883_88358

/-- Represents the number of sides on a standard die -/
def dieSides : ℕ := 6

/-- Calculates the number of three-digit numbers with all distinct digits -/
def distinctDigits : ℕ := dieSides * (dieSides - 1) * (dieSides - 2)

/-- Calculates the total number of different three-digit numbers -/
def totalNumbers : ℕ := dieSides ^ 3

/-- Calculates the number of three-digit numbers with exactly two digits the same -/
def twoSameDigits : ℕ := 3 * dieSides * (dieSides - 1)

theorem die_roll_counts :
  distinctDigits = 120 ∧ totalNumbers = 216 ∧ twoSameDigits = 90 := by
  sorry

end die_roll_counts_l883_88358


namespace physics_class_size_l883_88332

/-- Proves that the number of students in the physics class is 42 --/
theorem physics_class_size :
  ∀ (total_students : ℕ) 
    (math_only : ℕ) 
    (physics_only : ℕ) 
    (both : ℕ),
  total_students = 53 →
  math_only + physics_only + both = total_students →
  physics_only + both = 2 * (math_only + both) →
  both = 10 →
  physics_only + both = 42 :=
by
  sorry

#check physics_class_size

end physics_class_size_l883_88332


namespace no_valid_numbers_l883_88352

theorem no_valid_numbers :
  ¬∃ (a b c : ℕ), 
    (100 ≤ 100 * a + 10 * b + c) ∧ 
    (100 * a + 10 * b + c < 1000) ∧ 
    (100 * a + 10 * b + c) % 15 = 0 ∧ 
    (10 * b + c) % 4 = 0 ∧ 
    a > b ∧ b > c :=
by sorry

end no_valid_numbers_l883_88352


namespace age_difference_l883_88391

/-- Given that the total age of a and b is 13 years more than the total age of b and c,
    prove that c is 13 years younger than a. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 13) : a = c + 13 := by
  sorry

end age_difference_l883_88391


namespace max_value_f_l883_88361

/-- The function f(x) = x(1 - x^2) -/
def f (x : ℝ) : ℝ := x * (1 - x^2)

/-- The maximum value of f(x) on [0, 1] is 2√3/9 -/
theorem max_value_f : ∃ (c : ℝ), c = (2 * Real.sqrt 3) / 9 ∧ 
  (∀ x ∈ Set.Icc 0 1, f x ≤ c) ∧ 
  (∃ x ∈ Set.Icc 0 1, f x = c) := by
  sorry

end max_value_f_l883_88361


namespace problem_solution_l883_88349

theorem problem_solution : 
  ((-1)^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9) ∧ 
  (2 * ((3 : ℝ)^(1/2) - (2 : ℝ)^(1/2)) - ((2 : ℝ)^(1/2) + (3 : ℝ)^(1/2)) = (3 : ℝ)^(1/2) - 3 * (2 : ℝ)^(1/2)) := by
  sorry

#check problem_solution

end problem_solution_l883_88349


namespace coin_flip_experiment_l883_88379

theorem coin_flip_experiment (total_flips : ℕ) (heads_count : ℕ) (is_fair : Bool) :
  total_flips = 800 →
  heads_count = 440 →
  is_fair = true →
  (heads_count : ℚ) / (total_flips : ℚ) = 11/20 ∧ 
  (1 : ℚ) / 2 = 1/2 :=
by sorry

end coin_flip_experiment_l883_88379


namespace quadratic_factorization_sum_l883_88388

theorem quadratic_factorization_sum (a b c d : ℤ) : 
  (∀ x, x^2 + 13*x + 40 = (x + a) * (x + b)) →
  (∀ x, x^2 - 19*x + 88 = (x - c) * (x - d)) →
  a + b + c + d = 32 := by
sorry

end quadratic_factorization_sum_l883_88388


namespace bee_count_l883_88353

theorem bee_count (initial_bees additional_bees : ℕ) : 
  initial_bees = 16 → additional_bees = 10 → initial_bees + additional_bees = 26 := by
  sorry

end bee_count_l883_88353


namespace sphere_volume_l883_88383

theorem sphere_volume (r : ℝ) (d V : ℝ) (h1 : r = 1/3) (h2 : d = 2*r) (h3 : d = (16/9 * V)^(1/3)) : V = 1/6 := by
  sorry

end sphere_volume_l883_88383


namespace no_geometric_subsequence_of_three_l883_88333

theorem no_geometric_subsequence_of_three (a : ℕ → ℤ) :
  (∀ n, a n = 3^n - 2^n) →
  ¬ ∃ r s t : ℕ, r < s ∧ s < t ∧ ∃ b : ℚ, b ≠ 0 ∧
    (a s : ℚ) / (a r : ℚ) = b ∧ (a t : ℚ) / (a s : ℚ) = b :=
by sorry

end no_geometric_subsequence_of_three_l883_88333


namespace reflection_of_P_l883_88311

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -5)

theorem reflection_of_P :
  reflect_y P = (-3, -5) := by sorry

end reflection_of_P_l883_88311


namespace solution_set_when_a_is_one_range_of_a_for_inequality_l883_88398

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end solution_set_when_a_is_one_range_of_a_for_inequality_l883_88398


namespace shaded_area_of_partitioned_triangle_l883_88356

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_pos : leg_length > 0

/-- Represents a partition of a triangle -/
structure TrianglePartition where
  num_parts : ℕ
  num_parts_pos : num_parts > 0

theorem shaded_area_of_partitioned_triangle
  (t : IsoscelesRightTriangle)
  (p : TrianglePartition)
  (h1 : t.leg_length = 10)
  (h2 : p.num_parts = 25)
  (num_shaded : ℕ)
  (h3 : num_shaded = 15) :
  num_shaded * (t.leg_length^2 / 2) / p.num_parts = 30 := by
  sorry

end shaded_area_of_partitioned_triangle_l883_88356


namespace percentage_equality_l883_88343

theorem percentage_equality (x : ℝ) : (90 / 100 * 600 = 50 / 100 * x) → x = 1080 := by
  sorry

end percentage_equality_l883_88343


namespace equation_solution_l883_88399

theorem equation_solution : 
  ∃ x : ℝ, (4 : ℝ) ^ x = 2 ^ (x + 1) - 1 → x = 0 := by
  sorry

end equation_solution_l883_88399


namespace problem_solution_l883_88317

theorem problem_solution (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) :
  (a + b)^9 + a^6 = 2 := by
  sorry

end problem_solution_l883_88317


namespace circles_intersection_common_chord_l883_88320

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

theorem circles_intersection_common_chord :
  (∃ x y : ℝ, C₁ x y ∧ C₂ x y) →
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → common_chord x y) :=
by sorry

end circles_intersection_common_chord_l883_88320


namespace largest_B_term_l883_88334

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The term B_k in the expansion of (1+0.1)^500 -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.1 ^ k)

/-- Theorem stating that B_k is largest when k = 45 -/
theorem largest_B_term : 
  ∀ k : ℕ, k ≤ 500 → k ≠ 45 → B 45 > B k := by sorry

end largest_B_term_l883_88334


namespace expression_equals_59_l883_88305

theorem expression_equals_59 (a b c : ℝ) (ha : a = 17) (hb : b = 19) (hc : c = 23) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b + 1/c) + b * (1/c + 1/a) + c * (1/a + 1/b)) = 59 := by
  sorry

end expression_equals_59_l883_88305


namespace dice_cube_volume_l883_88374

/-- The volume of a cube formed by stacking dice --/
theorem dice_cube_volume 
  (num_dice : ℕ) 
  (die_edge : ℝ) 
  (h1 : num_dice = 125) 
  (h2 : die_edge = 2) 
  (h3 : ∃ n : ℕ, n ^ 3 = num_dice) : 
  (die_edge * (num_dice : ℝ) ^ (1/3 : ℝ)) ^ 3 = 1000 := by
  sorry

end dice_cube_volume_l883_88374


namespace one_real_zero_l883_88351

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- State the theorem
theorem one_real_zero : ∃! x : ℝ, f x = 0 := by
  sorry

end one_real_zero_l883_88351


namespace hyuksu_meat_consumption_l883_88354

/-- The amount of meat Hyuksu ate yesterday in kilograms -/
def meat_yesterday : ℝ := 2.6

/-- The amount of meat Hyuksu ate today in kilograms -/
def meat_today : ℝ := 5.98

/-- The total amount of meat Hyuksu ate in two days in kilograms -/
def total_meat : ℝ := meat_yesterday + meat_today

theorem hyuksu_meat_consumption : total_meat = 8.58 := by
  sorry

end hyuksu_meat_consumption_l883_88354


namespace no_equidistant_points_l883_88364

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the parallel tangents
def ParallelTangents (O : ℝ × ℝ) (d : ℝ) : Set (ℝ × ℝ) :=
  {P | |P.2 - O.2| = d}

-- Define a point equidistant from circle and tangents
def IsEquidistant (P : ℝ × ℝ) (O : ℝ × ℝ) (r d : ℝ) : Prop :=
  abs (((P.1 - O.1)^2 + (P.2 - O.2)^2).sqrt - r) = abs (|P.2 - O.2| - d)

theorem no_equidistant_points (O : ℝ × ℝ) (r d : ℝ) (h : d > r) :
  ¬∃P, IsEquidistant P O r d :=
by sorry

end no_equidistant_points_l883_88364


namespace only_2015_could_be_hexadecimal_l883_88336

def is_hexadecimal_digit (d : Char) : Bool :=
  ('0' <= d && d <= '9') || ('A' <= d && d <= 'F')

def could_be_hexadecimal (n : Nat) : Bool :=
  n.repr.all is_hexadecimal_digit

theorem only_2015_could_be_hexadecimal :
  (could_be_hexadecimal 66 = false) ∧
  (could_be_hexadecimal 108 = false) ∧
  (could_be_hexadecimal 732 = false) ∧
  (could_be_hexadecimal 2015 = true) :=
by sorry

end only_2015_could_be_hexadecimal_l883_88336


namespace store_revenue_calculation_l883_88382

/-- Represents the revenue calculation for Linda's store --/
def store_revenue (jean_price tee_price_low tee_price_high jacket_price jacket_discount tee_count_low tee_count_high jean_count jacket_count_regular jacket_count_discount sales_tax : ℚ) : ℚ :=
  let tee_revenue := tee_price_low * tee_count_low + tee_price_high * tee_count_high
  let jean_revenue := jean_price * jean_count
  let jacket_revenue_regular := jacket_price * jacket_count_regular
  let jacket_revenue_discount := jacket_price * (1 - jacket_discount) * jacket_count_discount
  let total_revenue := tee_revenue + jean_revenue + jacket_revenue_regular + jacket_revenue_discount
  let total_with_tax := total_revenue * (1 + sales_tax)
  total_with_tax

/-- Theorem stating that the store revenue matches the calculated amount --/
theorem store_revenue_calculation :
  store_revenue 22 15 20 37 0.1 4 3 4 2 3 0.07 = 408.63 :=
by sorry

end store_revenue_calculation_l883_88382


namespace min_candy_removal_l883_88365

def candy_distribution (total : ℕ) (sisters : ℕ) : ℕ :=
  total - sisters * (total / sisters)

theorem min_candy_removal (total : ℕ) (sisters : ℕ) 
  (h1 : total = 24) (h2 : sisters = 5) : 
  candy_distribution total sisters = 4 := by
  sorry

end min_candy_removal_l883_88365


namespace trajectory_of_shared_focus_l883_88307

/-- Given a parabola and a hyperbola sharing a focus, prove the trajectory of (m,n) -/
theorem trajectory_of_shared_focus (n m : ℝ) : 
  n < 0 → 
  (∃ (x y : ℝ), y^2 = 2*n*x) → 
  (∃ (x y : ℝ), x^2/4 - y^2/m^2 = 1) → 
  (∃ (f : ℝ × ℝ), f ∈ {p : ℝ × ℝ | p.1^2/(2*n) = p.2^2/m^2}) →
  n^2/16 - m^2/4 = 1 ∧ n < 0 :=
by sorry

end trajectory_of_shared_focus_l883_88307


namespace max_sum_of_three_numbers_l883_88390

theorem max_sum_of_three_numbers (a b c : ℕ) : 
  a + b = 1014 → c - b = 497 → a > b → (∀ S : ℕ, S = a + b + c → S ≤ 2017) ∧ (∃ S : ℕ, S = a + b + c ∧ S = 2017) :=
by sorry

end max_sum_of_three_numbers_l883_88390


namespace monotonicity_of_f_l883_88357

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

theorem monotonicity_of_f (a : ℝ) :
  (∀ x y, x < 1 → y < 1 → x < y → f a x > f a y) ∧ 
  (∀ x y, x > 1 → y > 1 → x < y → f a x < f a y) ∨
  (a = -Real.exp 1 / 2 ∧ ∀ x y, x < y → f a x < f a y) ∨
  (a < -Real.exp 1 / 2 ∧ 
    (∀ x y, x < 1 → y < 1 → x < y → f a x < f a y) ∧
    (∀ x y, 1 < x → x < Real.log (-2*a) → 1 < y → y < Real.log (-2*a) → x < y → f a x > f a y) ∧
    (∀ x y, x > Real.log (-2*a) → y > Real.log (-2*a) → x < y → f a x < f a y)) ∨
  (-Real.exp 1 / 2 < a ∧ a < 0 ∧
    (∀ x y, x < Real.log (-2*a) → y < Real.log (-2*a) → x < y → f a x < f a y) ∧
    (∀ x y, Real.log (-2*a) < x → x < 1 → Real.log (-2*a) < y → y < 1 → x < y → f a x > f a y) ∧
    (∀ x y, x > 1 → y > 1 → x < y → f a x < f a y)) :=
sorry

end

end monotonicity_of_f_l883_88357
