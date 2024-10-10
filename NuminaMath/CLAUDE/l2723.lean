import Mathlib

namespace max_a_value_l2723_272365

-- Define the events A and B
def event_A (x y a : ℝ) : Prop := x^2 + y^2 ≤ a ∧ a > 0

def event_B (x y : ℝ) : Prop :=
  x - y + 1 ≥ 0 ∧ 5*x - 2*y - 4 ≤ 0 ∧ 2*x + y + 2 ≥ 0

-- Define the conditional probability P(B|A) = 1
def conditional_probability_is_one (a : ℝ) : Prop :=
  ∀ x y, event_A x y a → event_B x y

-- Theorem statement
theorem max_a_value :
  ∃ a_max : ℝ, a_max = 1/2 ∧
  (∀ a : ℝ, conditional_probability_is_one a → a ≤ a_max) ∧
  conditional_probability_is_one a_max :=
sorry

end max_a_value_l2723_272365


namespace books_from_first_shop_l2723_272392

/-- 
Proves that the number of books bought from the first shop is 65, given:
- Total cost of books from first shop is 1150
- 50 books were bought from the second shop for 920
- The average price per book is 18
-/
theorem books_from_first_shop : 
  ∀ (x : ℕ), 
  (1150 + 920 : ℚ) / (x + 50 : ℚ) = 18 → x = 65 := by
sorry

end books_from_first_shop_l2723_272392


namespace pole_length_l2723_272358

theorem pole_length (pole_length : ℝ) (gate_height : ℝ) (gate_width : ℝ) : 
  gate_width = 3 →
  pole_length = gate_height + 1 →
  pole_length^2 = gate_height^2 + gate_width^2 →
  pole_length = 5 := by
sorry

end pole_length_l2723_272358


namespace quadratic_rational_solutions_l2723_272312

/-- A function that checks if a quadratic equation with given coefficients has rational solutions -/
def has_rational_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℚ, a * x^2 + b * x + c = 0

/-- The set of positive integers k for which 3x^2 + 17x + k = 0 has rational solutions -/
def valid_k_set : Set ℤ :=
  {k : ℤ | k > 0 ∧ has_rational_solutions 3 17 k}

theorem quadratic_rational_solutions :
  ∃ k₁ k₂ : ℕ,
    k₁ ≠ k₂ ∧
    (↑k₁ : ℤ) ∈ valid_k_set ∧
    (↑k₂ : ℤ) ∈ valid_k_set ∧
    valid_k_set = {↑k₁, ↑k₂} ∧
    k₁ * k₂ = 240 :=
by sorry

end quadratic_rational_solutions_l2723_272312


namespace find_b_value_l2723_272396

/-- Given two functions p and q, where p(x) = 2x - 11 and q(x) = 5x - b,
    prove that b = 8 when p(q(3)) = 3. -/
theorem find_b_value (b : ℝ) : 
  let p : ℝ → ℝ := λ x ↦ 2 * x - 11
  let q : ℝ → ℝ := λ x ↦ 5 * x - b
  p (q 3) = 3 → b = 8 := by
sorry

end find_b_value_l2723_272396


namespace acute_angle_condition_x_plus_y_value_l2723_272382

-- Define the vectors
def a : Fin 2 → ℝ := ![2, -1]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Theorem 1: Acute angle condition
theorem acute_angle_condition (x : ℝ) :
  (dot_product a (b x) > 0) ↔ (x > 1/2) := by sorry

-- Theorem 2: Value of x + y
theorem x_plus_y_value (x y : ℝ) :
  (3 • a - 2 • (b x) = ![4, y]) → x + y = -4 := by sorry

end acute_angle_condition_x_plus_y_value_l2723_272382


namespace cube_side_length_when_volume_equals_surface_area_l2723_272328

/-- For a cube where the numerical value of its volume equals the numerical value of its surface area, the side length is 6 units. -/
theorem cube_side_length_when_volume_equals_surface_area :
  ∀ s : ℝ, s > 0 → s^3 = 6 * s^2 → s = 6 := by
  sorry

end cube_side_length_when_volume_equals_surface_area_l2723_272328


namespace fence_savings_weeks_l2723_272366

theorem fence_savings_weeks (fence_cost : ℕ) (grandparents_gift : ℕ) (aunt_gift : ℕ) (cousin_gift : ℕ) (weekly_earnings : ℕ) :
  fence_cost = 800 →
  grandparents_gift = 120 →
  aunt_gift = 80 →
  cousin_gift = 20 →
  weekly_earnings = 20 →
  ∃ (weeks : ℕ), weeks = 29 ∧ fence_cost = grandparents_gift + aunt_gift + cousin_gift + weeks * weekly_earnings :=
by sorry

end fence_savings_weeks_l2723_272366


namespace rectangle_equal_diagonals_l2723_272379

-- Define a rectangle
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define equal diagonals
def equal_diagonals (A B C D : Point) : Prop := sorry

-- Theorem statement
theorem rectangle_equal_diagonals (A B C D : Point) :
  is_rectangle A B C D → equal_diagonals A B C D := by sorry

end rectangle_equal_diagonals_l2723_272379


namespace intersection_with_complement_l2723_272333

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem intersection_with_complement : A ∩ (U \ B) = {4, 5} := by
  sorry

end intersection_with_complement_l2723_272333


namespace omitted_angle_measure_l2723_272314

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180° --/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The property that the sum of interior angles is divisible by 180° --/
def is_valid_sum (s : ℕ) : Prop := ∃ k : ℕ, s = k * 180

/-- The sum calculated by Angela --/
def angela_sum : ℕ := 2583

/-- The theorem to prove --/
theorem omitted_angle_measure :
  ∃ (n : ℕ), 
    n > 2 ∧ 
    is_valid_sum (sum_interior_angles n) ∧ 
    sum_interior_angles n = angela_sum + 117 := by
  sorry

end omitted_angle_measure_l2723_272314


namespace divisible_by_eleven_l2723_272370

theorem divisible_by_eleven (n : ℕ) : 
  11 ∣ (6^(2*n) + 3^(n+2) + 3^n) := by
sorry

end divisible_by_eleven_l2723_272370


namespace square_and_arithmetic_computation_l2723_272388

theorem square_and_arithmetic_computation : 7^2 - (4 * 6) / 2 + 6^2 = 73 := by
  sorry

end square_and_arithmetic_computation_l2723_272388


namespace sphere_surface_area_circumscribing_cube_l2723_272376

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (surface_area : ℝ) :
  edge_length = 2 →
  surface_area = 4 * Real.pi * (((edge_length ^ 2 + edge_length ^ 2 + edge_length ^ 2) / 4) : ℝ) →
  surface_area = 12 * Real.pi :=
by sorry

end sphere_surface_area_circumscribing_cube_l2723_272376


namespace certain_number_proof_l2723_272303

theorem certain_number_proof : ∃ x : ℕ, (2994 : ℚ) / x = 177 ∧ x = 17 := by
  sorry

end certain_number_proof_l2723_272303


namespace mike_owes_laura_l2723_272304

theorem mike_owes_laura (rate : ℚ) (rooms : ℚ) (total : ℚ) : 
  rate = 13 / 3 → rooms = 8 / 5 → total = rate * rooms → total = 104 / 15 := by
  sorry

end mike_owes_laura_l2723_272304


namespace soda_price_calculation_l2723_272357

/-- Proves that the original price of each soda is $20/9 given the conditions of the problem -/
theorem soda_price_calculation (num_sodas : ℕ) (discount_rate : ℚ) (total_paid : ℚ) :
  num_sodas = 3 →
  discount_rate = 1/10 →
  total_paid = 6 →
  ∃ (original_price : ℚ), 
    original_price = 20/9 ∧ 
    num_sodas * (original_price * (1 - discount_rate)) = total_paid :=
by sorry

end soda_price_calculation_l2723_272357


namespace sufficient_not_necessary_condition_l2723_272337

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (((a > 0 ∧ b > 0) → (a * b > 0)) ∧
   (∃ a b : ℝ, (a * b > 0) ∧ ¬(a > 0 ∧ b > 0))) :=
by sorry

end sufficient_not_necessary_condition_l2723_272337


namespace point_symmetry_l2723_272387

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem point_symmetry (a b : ℝ) :
  symmetric_wrt_origin (3, a - 2) (b, a) → a + b = -2 := by
  sorry

end point_symmetry_l2723_272387


namespace arithmetic_sequence_sum_mod_15_l2723_272339

theorem arithmetic_sequence_sum_mod_15 (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 2 →
  d = 5 →
  aₙ = 102 →
  n * (a₁ + aₙ) / 2 ≡ 12 [MOD 15] :=
by sorry

end arithmetic_sequence_sum_mod_15_l2723_272339


namespace trigonometric_equation_solution_l2723_272346

theorem trigonometric_equation_solution (x : ℝ) : 
  2 * (Real.sin x ^ 6 + Real.cos x ^ 6) - 3 * (Real.sin x ^ 4 + Real.cos x ^ 4) = Real.cos (2 * x) →
  ∃ k : ℤ, x = π / 2 * (2 * ↑k + 1) :=
by sorry

end trigonometric_equation_solution_l2723_272346


namespace cold_drink_recipe_l2723_272355

theorem cold_drink_recipe (tea_per_drink : ℚ) (total_mixture : ℚ) (total_lemonade : ℚ)
  (h1 : tea_per_drink = 1/4)
  (h2 : total_mixture = 18)
  (h3 : total_lemonade = 15) :
  (total_lemonade / ((total_mixture - total_lemonade) / tea_per_drink)) = 5/4 := by
  sorry

end cold_drink_recipe_l2723_272355


namespace softball_team_savings_l2723_272391

/-- Calculates the savings for a softball team when buying uniforms with a group discount -/
theorem softball_team_savings 
  (regular_shirt_price regular_pants_price regular_socks_price : ℚ)
  (discounted_shirt_price discounted_pants_price discounted_socks_price : ℚ)
  (team_size : ℕ)
  (h_regular_shirt : regular_shirt_price = 7.5)
  (h_regular_pants : regular_pants_price = 15)
  (h_regular_socks : regular_socks_price = 4.5)
  (h_discounted_shirt : discounted_shirt_price = 6.75)
  (h_discounted_pants : discounted_pants_price = 13.5)
  (h_discounted_socks : discounted_socks_price = 3.75)
  (h_team_size : team_size = 12) :
  let regular_uniform_cost := regular_shirt_price + regular_pants_price + regular_socks_price
  let discounted_uniform_cost := discounted_shirt_price + discounted_pants_price + discounted_socks_price
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by sorry


end softball_team_savings_l2723_272391


namespace number_equation_solution_l2723_272327

theorem number_equation_solution : 
  ∃ x : ℝ, x - (1002 / 20.04) = 2984 ∧ x = 3034 := by
  sorry

end number_equation_solution_l2723_272327


namespace marilyn_bottle_caps_l2723_272374

/-- The number of bottle caps Marilyn shared -/
def shared_caps : ℕ := 36

/-- The number of bottle caps Marilyn ended up with -/
def remaining_caps : ℕ := 15

/-- The initial number of bottle caps Marilyn had -/
def initial_caps : ℕ := shared_caps + remaining_caps

theorem marilyn_bottle_caps : initial_caps = 51 := by
  sorry

end marilyn_bottle_caps_l2723_272374


namespace roundness_of_eight_million_l2723_272394

/-- Roundness of a positive integer is the sum of exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million :
  roundness 8000000 = 15 := by sorry

end roundness_of_eight_million_l2723_272394


namespace multitive_function_thirtysix_l2723_272385

/-- A function satisfying f(a · b) = f(a) + f(b) -/
def MultitiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ a b, f (a * b) = f a + f b

/-- Theorem: Given a multitive function f with f(2) = p and f(3) = q, prove f(36) = 2(p + q) -/
theorem multitive_function_thirtysix
  (f : ℝ → ℝ) (p q : ℝ)
  (hf : MultitiveFunction f)
  (h2 : f 2 = p)
  (h3 : f 3 = q) :
  f 36 = 2 * (p + q) := by
  sorry

end multitive_function_thirtysix_l2723_272385


namespace sum_of_squares_l2723_272313

theorem sum_of_squares (x y z p q r : ℝ) 
  (h1 : x + y = p) 
  (h2 : y + z = q) 
  (h3 : z + x = r) 
  (hp : p ≠ 0) 
  (hq : q ≠ 0) 
  (hr : r ≠ 0) : 
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p*q - q*r - r*p) / 2 := by
  sorry

end sum_of_squares_l2723_272313


namespace arithmetic_geometric_mean_inequality_l2723_272393

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end arithmetic_geometric_mean_inequality_l2723_272393


namespace elimination_tournament_sequences_l2723_272349

def team_size : ℕ := 7

/-- The number of possible sequences in the elimination tournament -/
def elimination_sequences (n : ℕ) : ℕ :=
  2 * (Nat.choose (2 * n - 1) (n - 1))

/-- The theorem stating the number of possible sequences for the given problem -/
theorem elimination_tournament_sequences :
  elimination_sequences team_size = 3432 := by
  sorry

end elimination_tournament_sequences_l2723_272349


namespace ellipse_properties_l2723_272324

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + Real.sqrt 2 = 0

-- Define the theorem
theorem ellipse_properties
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_ecc : eccentricity a b = Real.sqrt 3 / 2)
  (h_tangent : ∃ (x y : ℝ), x^2 + y^2 = b^2 ∧ tangent_line x y) :
  -- 1. Equation of C
  (∀ x y, ellipse a b x y ↔ x^2/4 + y^2 = 1) ∧
  -- 2. Range of slope k
  (∀ M N E : ℝ × ℝ,
    ellipse a b M.1 M.2 →
    ellipse a b N.1 N.2 →
    M.1 = N.1 ∧ M.2 = -N.2 →
    M ≠ N →
    ellipse a b E.1 E.2 →
    (∃ k : ℝ, k ≠ 0 ∧ 
      N.2 - 0 = k * (N.1 - 4) ∧
      E.2 - 0 = k * (E.1 - 4)) →
    -Real.sqrt 3 / 6 < k ∧ k < Real.sqrt 3 / 6) ∧
  -- 3. Fixed intersection point
  (∀ M N E : ℝ × ℝ,
    ellipse a b M.1 M.2 →
    ellipse a b N.1 N.2 →
    M.1 = N.1 ∧ M.2 = -N.2 →
    M ≠ N →
    ellipse a b E.1 E.2 →
    (∃ k : ℝ, k ≠ 0 ∧ 
      N.2 - 0 = k * (N.1 - 4) ∧
      E.2 - 0 = k * (E.1 - 4)) →
    ∃ t : ℝ, M.2 - E.2 = ((E.2 + M.2) / (E.1 - M.1)) * (t - E.1) ∧ t = 1) :=
by sorry

end

end ellipse_properties_l2723_272324


namespace inequality_proof_l2723_272320

theorem inequality_proof (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by sorry

end inequality_proof_l2723_272320


namespace linear_inequality_solution_l2723_272360

theorem linear_inequality_solution (x : ℝ) : 3 * (x + 1) > 9 ↔ x > 2 := by
  sorry

end linear_inequality_solution_l2723_272360


namespace joe_money_left_l2723_272315

/-- The amount of money Joe has left after shopping and donating to charity -/
def money_left (initial_amount notebooks books pens stickers notebook_price book_price pen_price sticker_price charity : ℕ) : ℕ :=
  initial_amount - (notebooks * notebook_price + books * book_price + pens * pen_price + stickers * sticker_price + charity)

/-- Theorem stating that Joe has $60 left after his shopping trip and charity donation -/
theorem joe_money_left :
  money_left 150 7 2 5 3 4 12 2 6 10 = 60 := by
  sorry

#eval money_left 150 7 2 5 3 4 12 2 6 10

end joe_money_left_l2723_272315


namespace point_movement_to_origin_l2723_272395

theorem point_movement_to_origin (a b : ℝ) :
  (2 * a - 2 = 0) ∧ (-3 * b - 3 = 0) →
  (2 * a = 2) ∧ (-3 * b = 3) :=
by sorry

end point_movement_to_origin_l2723_272395


namespace infinitely_many_consecutive_epsilon_squarish_l2723_272342

/-- A positive integer is ε-squarish if it's the product of two integers a and b
    where 1 < a < b < (1 + ε)a -/
def IsEpsilonSquarish (ε : ℝ) (k : ℕ) : Prop :=
  ∃ (a b : ℕ), k = a * b ∧ 1 < a ∧ a < b ∧ b < (1 + ε) * a

/-- There exist infinitely many positive integers n such that
    n², n² - 1, n² - 2, n² - 3, n² - 4, and n² - 5 are all ε-squarish -/
theorem infinitely_many_consecutive_epsilon_squarish (ε : ℝ) (hε : ε > 0) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧
    IsEpsilonSquarish ε (n^2) ∧
    IsEpsilonSquarish ε (n^2 - 1) ∧
    IsEpsilonSquarish ε (n^2 - 2) ∧
    IsEpsilonSquarish ε (n^2 - 3) ∧
    IsEpsilonSquarish ε (n^2 - 4) ∧
    IsEpsilonSquarish ε (n^2 - 5) :=
by
  sorry

end infinitely_many_consecutive_epsilon_squarish_l2723_272342


namespace sqrt_inequality_l2723_272318

theorem sqrt_inequality (x : ℝ) :
  x > 0 → (Real.sqrt x > 3 * x - 2 ↔ 4/9 < x ∧ x < 1) := by sorry

end sqrt_inequality_l2723_272318


namespace angle_ratio_theorem_l2723_272334

theorem angle_ratio_theorem (α : Real) (m : Real) :
  m < 0 →
  let P : Real × Real := (4 * m, -3 * m)
  (P.1 / (Real.sqrt (P.1^2 + P.2^2)) = -4/5) →
  (P.2 / (Real.sqrt (P.1^2 + P.2^2)) = 3/5) →
  (2 * (P.2 / (Real.sqrt (P.1^2 + P.2^2))) + (P.1 / (Real.sqrt (P.1^2 + P.2^2)))) /
  ((P.2 / (Real.sqrt (P.1^2 + P.2^2))) - (P.1 / (Real.sqrt (P.1^2 + P.2^2)))) = 2/7 := by
sorry

end angle_ratio_theorem_l2723_272334


namespace certain_amount_proof_l2723_272359

/-- The interest rate per annum -/
def interest_rate : ℚ := 8 / 100

/-- The time period for the first amount in years -/
def time1 : ℚ := 25 / 2

/-- The time period for the second amount in years -/
def time2 : ℚ := 4

/-- The first principal amount in Rs -/
def principal1 : ℚ := 160

/-- The second principal amount (the certain amount) in Rs -/
def principal2 : ℚ := 500

/-- Simple interest formula -/
def simple_interest (p r t : ℚ) : ℚ := p * r * t

theorem certain_amount_proof :
  simple_interest principal1 interest_rate time1 = simple_interest principal2 interest_rate time2 :=
sorry

end certain_amount_proof_l2723_272359


namespace expression_never_equals_negative_one_l2723_272307

theorem expression_never_equals_negative_one (a y : ℝ) (ha : a ≠ 0) (hy1 : y ≠ -a) (hy2 : y ≠ 2*a) :
  (2*a^2 + y^2) / (a*y - y^2 - a^2) ≠ -1 := by
  sorry

end expression_never_equals_negative_one_l2723_272307


namespace line_perp_parallel_planes_l2723_272309

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : perpendicular l α) 
  (h2 : parallel α β) : 
  perpendicular l β :=
sorry

end line_perp_parallel_planes_l2723_272309


namespace peter_marbles_l2723_272317

theorem peter_marbles (initial_marbles lost_marbles : ℕ) 
  (h1 : initial_marbles = 33)
  (h2 : lost_marbles = 15) :
  initial_marbles - lost_marbles = 18 := by
  sorry

end peter_marbles_l2723_272317


namespace base_conversion_subtraction_l2723_272367

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 2, 4]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 5]
def base2 : Nat := 6

-- State the theorem
theorem base_conversion_subtraction :
  (to_base_10 num1 base1) - (to_base_10 num2 base2) = 182 := by
  sorry

end base_conversion_subtraction_l2723_272367


namespace intersection_of_A_and_B_l2723_272372

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l2723_272372


namespace fashion_show_models_l2723_272378

/-- The number of bathing suit sets each model wears -/
def bathing_suit_sets : ℕ := 2

/-- The number of evening wear sets each model wears -/
def evening_wear_sets : ℕ := 3

/-- The time in minutes for one runway walk -/
def runway_walk_time : ℕ := 2

/-- The total runway time for the show in minutes -/
def total_runway_time : ℕ := 60

/-- The number of models in the fashion show -/
def number_of_models : ℕ := 6

theorem fashion_show_models :
  (bathing_suit_sets + evening_wear_sets) * runway_walk_time * number_of_models = total_runway_time :=
by sorry

end fashion_show_models_l2723_272378


namespace opposite_unit_vector_l2723_272335

def vec_a : ℝ × ℝ := (4, 2)

theorem opposite_unit_vector :
  let opposite_unit := (-vec_a.1 / Real.sqrt (vec_a.1^2 + vec_a.2^2),
                        -vec_a.2 / Real.sqrt (vec_a.1^2 + vec_a.2^2))
  opposite_unit = (-2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5) :=
by sorry

end opposite_unit_vector_l2723_272335


namespace max_profit_at_60_l2723_272330

/-- The profit function for a travel agency chartering a plane -/
def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then
    900 * x - 15000
  else if x ≤ 75 then
    (-10 * x + 1200) * x - 15000
  else
    0

/-- The maximum number of people allowed in the tour group -/
def max_people : ℕ := 75

/-- The charter fee for the travel agency -/
def charter_fee : ℝ := 15000

theorem max_profit_at_60 :
  ∀ x : ℕ, x ≤ max_people → profit x ≤ profit 60 ∧ profit 60 = 21000 :=
sorry

end max_profit_at_60_l2723_272330


namespace first_player_can_avoid_losing_l2723_272364

/-- A strategy for selecting vectors -/
def Strategy := List (ℝ × ℝ) → ℝ × ℝ

/-- The game state, including all vectors and the current player's turn -/
structure GameState where
  vectors : List (ℝ × ℝ)
  player_turn : ℕ

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- Play the game with given strategies -/
def play_game (initial_vectors : List (ℝ × ℝ)) (strategy1 strategy2 : Strategy) : GameResult :=
  sorry

/-- Theorem stating that the first player can always avoid losing -/
theorem first_player_can_avoid_losing (vectors : List (ℝ × ℝ)) 
  (h : vectors.length = 1992) : 
  ∃ (strategy1 : Strategy), ∀ (strategy2 : Strategy),
    play_game vectors strategy1 strategy2 ≠ GameResult.SecondPlayerWins :=
  sorry

end first_player_can_avoid_losing_l2723_272364


namespace projection_matrix_condition_l2723_272332

/-- A projection matrix is idempotent (P^2 = P) -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix form given in the problem -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 20/49], ![c, 29/49]]

/-- The theorem stating the conditions for the given matrix to be a projection matrix -/
theorem projection_matrix_condition (a c : ℚ) :
  is_projection_matrix (P a c) ↔ a = 1 ∧ c = 0 := by
  sorry

#check projection_matrix_condition

end projection_matrix_condition_l2723_272332


namespace composite_divisor_bound_l2723_272384

/-- A number is composite if it's a natural number greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- Theorem: Every composite number has a divisor greater than 1 but not greater than its square root -/
theorem composite_divisor_bound {n : ℕ} (h : IsComposite n) :
  ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d ≤ Real.sqrt (n : ℝ) :=
sorry

end composite_divisor_bound_l2723_272384


namespace staples_left_l2723_272329

def initial_staples : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

theorem staples_left : initial_staples - reports_stapled = 14 := by
  sorry

end staples_left_l2723_272329


namespace election_winner_votes_l2723_272397

theorem election_winner_votes (total_votes : ℕ) (candidates : ℕ) 
  (difference1 : ℕ) (difference2 : ℕ) (difference3 : ℕ) :
  total_votes = 963 →
  candidates = 4 →
  difference1 = 53 →
  difference2 = 79 →
  difference3 = 105 →
  ∃ (winner_votes : ℕ),
    winner_votes + (winner_votes - difference1) + 
    (winner_votes - difference2) + (winner_votes - difference3) = total_votes ∧
    winner_votes = 300 :=
by sorry

end election_winner_votes_l2723_272397


namespace josie_remaining_money_l2723_272371

/-- Calculates the remaining money after Josie's grocery shopping --/
def remaining_money (initial_amount : ℚ) 
  (milk_price : ℚ) (milk_discount : ℚ) 
  (bread_price : ℚ) 
  (detergent_price : ℚ) (detergent_coupon : ℚ) 
  (banana_price_per_pound : ℚ) (banana_pounds : ℚ) : ℚ :=
  let milk_cost := milk_price * (1 - milk_discount)
  let detergent_cost := detergent_price - detergent_coupon
  let banana_cost := banana_price_per_pound * banana_pounds
  let total_cost := milk_cost + bread_price + detergent_cost + banana_cost
  initial_amount - total_cost

/-- Theorem stating that Josie has $4.00 left after shopping --/
theorem josie_remaining_money :
  remaining_money 20 4 (1/2) 3.5 10.25 1.25 0.75 2 = 4 := by
  sorry

end josie_remaining_money_l2723_272371


namespace remainder_16_pow_2048_mod_11_l2723_272300

theorem remainder_16_pow_2048_mod_11 : 16^2048 % 11 = 4 := by
  sorry

end remainder_16_pow_2048_mod_11_l2723_272300


namespace car_discount_proof_l2723_272383

/-- Proves that the initial discount on a car's original price was 30%, given specific selling conditions --/
theorem car_discount_proof (P : ℝ) (D : ℝ) : 
  P > 0 →  -- Original price is positive
  0 ≤ D ∧ D < 1 →  -- Discount is between 0 and 1 (exclusive)
  P * (1 - D) * 1.7 = P * 1.18999999999999993 →  -- Selling price equation
  D = 0.3 := by
sorry

end car_discount_proof_l2723_272383


namespace function_inequality_l2723_272325

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x^2 - 3*x + 2) * deriv f x ≤ 0) :
  ∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 := by
  sorry

end function_inequality_l2723_272325


namespace least_subtraction_for_divisibility_l2723_272306

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(29 ∣ (87654321 - y))) ∧ 
  (29 ∣ (87654321 - x)) :=
sorry

end least_subtraction_for_divisibility_l2723_272306


namespace correct_prediction_probability_l2723_272331

theorem correct_prediction_probability :
  let n_monday : ℕ := 5
  let n_tuesday : ℕ := 6
  let n_total : ℕ := n_monday + n_tuesday
  let n_correct : ℕ := 7
  let n_correct_monday : ℕ := 3
  let n_correct_tuesday : ℕ := n_correct - n_correct_monday
  let p : ℝ := 1 / 2

  (Nat.choose n_monday n_correct_monday * p^n_monday * (1-p)^(n_monday - n_correct_monday)) *
  (Nat.choose n_tuesday n_correct_tuesday * p^n_tuesday * (1-p)^(n_tuesday - n_correct_tuesday)) /
  (Nat.choose n_total n_correct * p^n_correct * (1-p)^(n_total - n_correct)) = 5 / 11 :=
by
  sorry

end correct_prediction_probability_l2723_272331


namespace least_five_digit_divisible_by_18_12_15_l2723_272301

theorem least_five_digit_divisible_by_18_12_15 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n → n ≥ 10080 :=
by sorry

end least_five_digit_divisible_by_18_12_15_l2723_272301


namespace adjacent_xue_rong_rong_arrangements_l2723_272363

def num_bing_dung_dung : ℕ := 4
def num_xue_rong_rong : ℕ := 3

def adjacent_arrangements (n_bdd : ℕ) (n_xrr : ℕ) : ℕ :=
  2 * (n_bdd + 2).factorial * (n_bdd + 1)

theorem adjacent_xue_rong_rong_arrangements :
  adjacent_arrangements num_bing_dung_dung num_xue_rong_rong = 960 := by
  sorry

end adjacent_xue_rong_rong_arrangements_l2723_272363


namespace finleys_age_l2723_272386

/-- Proves Finley's age given the conditions in the problem -/
theorem finleys_age (jill_age : ℕ) (roger_age : ℕ) (finley_age : ℕ) : 
  jill_age = 20 →
  roger_age = 2 * jill_age + 5 →
  (roger_age + 15) - (jill_age + 15) = finley_age - 30 →
  finley_age = 40 :=
by
  sorry

end finleys_age_l2723_272386


namespace existence_of_divisible_m_l2723_272343

theorem existence_of_divisible_m : ∃ m : ℕ+, (3^100 * m.val + 3^100 - 1) % 1988 = 0 := by
  sorry

end existence_of_divisible_m_l2723_272343


namespace second_player_wins_l2723_272361

/-- A game played on a circle with 2n + 1 equally spaced points. -/
structure CircleGame where
  n : ℕ
  h : n ≥ 2

/-- A player in the game. -/
inductive Player
  | First
  | Second

/-- A strategy for a player. -/
def Strategy (g : CircleGame) := List (Fin (2 * g.n + 1)) → Fin (2 * g.n + 1)

/-- Predicate to check if all remaining triangles are obtuse. -/
def AllTrianglesObtuse (g : CircleGame) (remaining : List (Fin (2 * g.n + 1))) : Prop :=
  sorry

/-- Predicate to check if a strategy is winning for a player. -/
def IsWinningStrategy (g : CircleGame) (p : Player) (s : Strategy g) : Prop :=
  sorry

/-- Theorem stating that the second player has a winning strategy. -/
theorem second_player_wins (g : CircleGame) :
  ∃ (s : Strategy g), IsWinningStrategy g Player.Second s :=
sorry

end second_player_wins_l2723_272361


namespace extra_bananas_l2723_272377

/-- Given the total number of children, the number of absent children, and the planned distribution,
    prove that each present child received 2 extra bananas. -/
theorem extra_bananas (total_children absent_children planned_per_child : ℕ) 
  (h1 : total_children = 660)
  (h2 : absent_children = 330)
  (h3 : planned_per_child = 2) :
  let present_children := total_children - absent_children
  let total_bananas := total_children * planned_per_child
  let actual_per_child := total_bananas / present_children
  actual_per_child - planned_per_child = 2 := by
  sorry

end extra_bananas_l2723_272377


namespace simplify_polynomial_l2723_272305

theorem simplify_polynomial (w : ℝ) : 
  3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end simplify_polynomial_l2723_272305


namespace quadratic_equation_roots_l2723_272368

theorem quadratic_equation_roots (m : ℚ) :
  (∃ x : ℚ, x^2 + 2*x + 3*m - 4 = 0) ∧ 
  (2^2 + 2*2 + 3*m - 4 = 0) →
  ((-4)^2 + 2*(-4) + 3*m - 4 = 0) ∧ 
  m = -4/3 := by
sorry

end quadratic_equation_roots_l2723_272368


namespace fibonacci_identity_l2723_272348

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fibonacci_identity (θ : ℝ) (x : ℝ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π)
  (h2 : x + 1/x = 2 * Real.cos (2 * θ)) :
  x^(fib n) + 1/(x^(fib n)) = 2 * Real.cos (2 * (fib n) * θ) := by
  sorry

end fibonacci_identity_l2723_272348


namespace triangle_intersection_invariance_l2723_272338

/-- Represents a right triangle in a plane -/
structure RightTriangle where
  leg1 : Real
  leg2 : Real

/-- Represents a line in a plane -/
structure Line where
  slope : Real
  intercept : Real

/-- Represents the configuration of three right triangles relative to a line -/
structure TriangleConfiguration where
  triangles : Fin 3 → RightTriangle
  base_line : Line
  intersecting_line : Line

/-- Checks if a line intersects three triangles into equal segments -/
def intersects_equally (config : TriangleConfiguration) : Prop :=
  sorry

/-- The main theorem -/
theorem triangle_intersection_invariance 
  (initial_config : TriangleConfiguration)
  (rotated_config : TriangleConfiguration)
  (h1 : intersects_equally initial_config)
  (h2 : ∀ i : Fin 3, 
    (initial_config.triangles i).leg1 = (rotated_config.triangles i).leg2 ∧
    (initial_config.triangles i).leg2 = (rotated_config.triangles i).leg1)
  (h3 : initial_config.base_line = rotated_config.base_line) :
  ∃ new_line : Line, 
    new_line.slope = initial_config.intersecting_line.slope ∧
    intersects_equally { triangles := rotated_config.triangles,
                         base_line := rotated_config.base_line,
                         intersecting_line := new_line } :=
sorry

end triangle_intersection_invariance_l2723_272338


namespace arithmetic_sequence_problem_l2723_272340

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 40 := by
  sorry

end arithmetic_sequence_problem_l2723_272340


namespace cost_of_20_pencils_12_notebooks_l2723_272356

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℚ := sorry

/-- The first condition: 8 pencils and 10 notebooks cost $5.20 -/
axiom condition1 : 8 * pencil_cost + 10 * notebook_cost = 5.20

/-- The second condition: 6 pencils and 4 notebooks cost $2.24 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.24

/-- The theorem to prove -/
theorem cost_of_20_pencils_12_notebooks : 
  20 * pencil_cost + 12 * notebook_cost = 6.84 := by sorry

end cost_of_20_pencils_12_notebooks_l2723_272356


namespace line_equation_through_midpoint_l2723_272350

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is on the x-axis -/
def Point.onXAxis (p : Point) : Prop :=
  p.y = 0

/-- Check if a point is on the y-axis -/
def Point.onYAxis (p : Point) : Prop :=
  p.x = 0

/-- Check if a point is the midpoint of two other points -/
def Point.isMidpointOf (m p q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

theorem line_equation_through_midpoint (m p q : Point) (l : Line) :
  m = Point.mk 1 (-2) →
  p.onXAxis →
  q.onYAxis →
  m.isMidpointOf p q →
  p.onLine l →
  q.onLine l →
  m.onLine l →
  l = Line.mk 2 (-1) (-4) := by
  sorry

end line_equation_through_midpoint_l2723_272350


namespace inequality_not_always_true_l2723_272310

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, ¬(a * c^2 > b * c^2) :=
by
  -- Proof goes here
  sorry

end inequality_not_always_true_l2723_272310


namespace point_translation_l2723_272398

/-- Given a point A(-5, 6) in a Cartesian coordinate system, 
    moving it 5 units right and 6 units up results in point A₁(0, 12) -/
theorem point_translation :
  let A : ℝ × ℝ := (-5, 6)
  let right_shift : ℝ := 5
  let up_shift : ℝ := 6
  let A₁ : ℝ × ℝ := (A.1 + right_shift, A.2 + up_shift)
  A₁ = (0, 12) := by
sorry

end point_translation_l2723_272398


namespace sum_of_roots_equation_l2723_272373

theorem sum_of_roots_equation (x : ℝ) :
  (x ≠ 3 ∧ x ≠ -3) →
  ((-6 * x) / (x^2 - 9) = (3 * x) / (x + 3) - 2 / (x - 3) + 1) →
  ∃ (y : ℝ), (y ≠ 3 ∧ y ≠ -3) ∧
             ((-6 * y) / (y^2 - 9) = (3 * y) / (y + 3) - 2 / (y - 3) + 1) ∧
             x + y = 5/4 :=
by sorry

end sum_of_roots_equation_l2723_272373


namespace negative_two_a_cubed_l2723_272369

theorem negative_two_a_cubed (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end negative_two_a_cubed_l2723_272369


namespace line_through_parabola_vertex_l2723_272344

theorem line_through_parabola_vertex :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  ∀ b : ℝ, b ∈ s ↔ 
    (∃ x y : ℝ, y = 2*x + b ∧ 
               y = x^2 + b^2 - 1 ∧ 
               ∀ x' : ℝ, x'^2 + b^2 - 1 ≤ y) :=
by sorry

end line_through_parabola_vertex_l2723_272344


namespace solve_cassette_problem_l2723_272341

structure AudioVideoCassettes where
  audioCost : ℝ
  videoCost : ℝ
  firstSetAudioCount : ℝ
  secondSetAudioCount : ℝ

def cassetteProblem (c : AudioVideoCassettes) : Prop :=
  c.videoCost = 300 ∧
  c.firstSetAudioCount * c.audioCost + 4 * c.videoCost = 1350 ∧
  7 * c.audioCost + 3 * c.videoCost = 1110 ∧
  c.secondSetAudioCount = 7

theorem solve_cassette_problem :
  ∃ c : AudioVideoCassettes, cassetteProblem c :=
by
  sorry

end solve_cassette_problem_l2723_272341


namespace flag_distribution_theorem_l2723_272347

/-- Represents the box of flags -/
structure FlagBox where
  total : ℕ
  blue : ℕ
  red : ℕ

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  total : ℕ
  blue : ℕ
  red : ℕ
  both : ℕ

def is_valid_box (box : FlagBox) : Prop :=
  box.total = box.blue + box.red ∧ box.total % 2 = 0

def is_valid_distribution (box : FlagBox) (dist : FlagDistribution) : Prop :=
  dist.total = box.total / 2 ∧
  dist.blue = (6 * dist.total) / 10 ∧
  dist.red = (6 * dist.total) / 10 ∧
  dist.total = dist.blue + dist.red - dist.both

theorem flag_distribution_theorem (box : FlagBox) (dist : FlagDistribution) :
  is_valid_box box → is_valid_distribution box dist →
  dist.both = dist.total / 5 :=
sorry

end flag_distribution_theorem_l2723_272347


namespace multiply_123_32_125_l2723_272351

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end multiply_123_32_125_l2723_272351


namespace no_integer_solutions_l2723_272321

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 + 24 := by
  sorry

end no_integer_solutions_l2723_272321


namespace train_length_l2723_272322

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 32 → (speed * (5/18) * time) = 373.33 := by
  sorry

end train_length_l2723_272322


namespace inequality_proof_l2723_272323

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3 := by
  sorry

end inequality_proof_l2723_272323


namespace absolute_value_inequality_l2723_272375

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 := by
  sorry

end absolute_value_inequality_l2723_272375


namespace number_of_piglets_born_l2723_272336

def sellPrice : ℕ := 300
def feedCost : ℕ := 10
def profitEarned : ℕ := 960

def pigsSoldAt12Months : ℕ := 3
def pigsSoldAt16Months : ℕ := 3

def totalPigsSold : ℕ := pigsSoldAt12Months + pigsSoldAt16Months

theorem number_of_piglets_born (sellPrice feedCost profitEarned 
  pigsSoldAt12Months pigsSoldAt16Months totalPigsSold : ℕ) :
  sellPrice = 300 →
  feedCost = 10 →
  profitEarned = 960 →
  pigsSoldAt12Months = 3 →
  pigsSoldAt16Months = 3 →
  totalPigsSold = pigsSoldAt12Months + pigsSoldAt16Months →
  totalPigsSold = 6 :=
by sorry

end number_of_piglets_born_l2723_272336


namespace f_is_quadratic_l2723_272345

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := 11 * x^2 + 29 * x

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l2723_272345


namespace hyperbola_vertices_distance_l2723_272390

/-- The distance between the vertices of a hyperbola -/
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 16 = 1

theorem hyperbola_vertices_distance :
  ∃ (a : ℝ), a^2 = 36 ∧ distance_between_vertices a = 12 :=
by sorry

end hyperbola_vertices_distance_l2723_272390


namespace two_distinct_roots_root_three_implies_sum_l2723_272380

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 + 2*m*x + m^2 - 2 = 0

-- Part 1: The equation always has two distinct real roots
theorem two_distinct_roots :
  ∀ m : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

-- Part 2: If 3 is a root, then 2m^2 + 12m + 2043 = 2029
theorem root_three_implies_sum (m : ℝ) :
  quadratic_equation m 3 → 2*m^2 + 12*m + 2043 = 2029 :=
sorry

end two_distinct_roots_root_three_implies_sum_l2723_272380


namespace bridge_length_l2723_272326

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 ∧ 
  train_speed_kmh = 54 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 320 := by
  sorry

end bridge_length_l2723_272326


namespace circle_properties_l2723_272319

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a diameter
def diameter (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = (2 * c.radius)^2

-- Define a point on the circle
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_properties (c : Circle) :
  (∀ p q : ℝ × ℝ, diameter c p q → ∀ r s : ℝ × ℝ, diameter c r s → 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = (r.1 - s.1)^2 + (r.2 - s.2)^2) ∧
  (∀ p q : ℝ × ℝ, onCircle c p → onCircle c q → 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2) :=
sorry

end circle_properties_l2723_272319


namespace binomial_expansion_coefficient_l2723_272381

theorem binomial_expansion_coefficient (n : ℕ) : 
  (8 * (Nat.choose n 3) * 2^3 = 16 * n) → n = 5 := by
  sorry

end binomial_expansion_coefficient_l2723_272381


namespace smallest_divisible_by_prime_main_result_l2723_272353

def consecutive_even_product (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n/2 + 1)) (λ i => 2 * i)

theorem smallest_divisible_by_prime (p : ℕ) (hp : Nat.Prime p) :
  (∀ m : ℕ, m < 2 * p → ¬(p ∣ consecutive_even_product m)) ∧
  (p ∣ consecutive_even_product (2 * p)) :=
sorry

theorem main_result : 
  ∀ n : ℕ, n < 63994 → ¬(31997 ∣ consecutive_even_product n) ∧
  31997 ∣ consecutive_even_product 63994 :=
sorry

end smallest_divisible_by_prime_main_result_l2723_272353


namespace arithmetic_mean_relation_l2723_272352

theorem arithmetic_mean_relation (n : ℕ) (d : ℕ) (h1 : d > 0) :
  let seq := List.range d
  let arithmetic_mean := (n * d + (d * (d - 1)) / 2) / d
  let largest := n + d - 1
  arithmetic_mean = 5 * n →
  largest / arithmetic_mean = 9 / 5 := by
sorry

end arithmetic_mean_relation_l2723_272352


namespace nine_multiple_plus_k_equals_ones_l2723_272302

/-- Given a natural number N and a positive integer k, there exists a number M
    consisting of k ones such that N · 9 + k = M. -/
theorem nine_multiple_plus_k_equals_ones (N : ℕ) (k : ℕ+) :
  ∃ M : ℕ, (∀ d : ℕ, d < k → (M / 10^d) % 10 = 1) ∧ N * 9 + k = M :=
sorry

end nine_multiple_plus_k_equals_ones_l2723_272302


namespace gcd_2210_145_l2723_272316

theorem gcd_2210_145 : Int.gcd 2210 145 = 5 := by
  sorry

end gcd_2210_145_l2723_272316


namespace quadratic_completing_square_l2723_272308

theorem quadratic_completing_square : ∀ x : ℝ, x^2 - 4*x + 5 = (x - 2)^2 + 1 := by
  sorry

end quadratic_completing_square_l2723_272308


namespace negation_of_p_l2723_272362

-- Define the proposition p
def p : Prop := ∀ a : ℝ, a ≥ 0 → ∃ x : ℝ, x^2 + a*x + 1 = 0

-- State the theorem
theorem negation_of_p : 
  ¬p ↔ ∃ a : ℝ, a ≥ 0 ∧ ¬∃ x : ℝ, x^2 + a*x + 1 = 0 :=
by sorry

end negation_of_p_l2723_272362


namespace intersection_A_complement_B_range_of_a_for_not_p_sufficient_not_necessary_for_q_l2723_272354

-- Define the sets A and B
def A : Set ℝ := {x | 6 + 5*x - x^2 > 0}
def B (a : ℝ) : Set ℝ := {x | (x - (1-a)) * (x - (1+a)) > 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- Statement 1: A ∩ (ℝ\B) when a = 2
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

-- Statement 2: Range of a where ¬p is sufficient but not necessary for q
theorem range_of_a_for_not_p_sufficient_not_necessary_for_q :
  {a : ℝ | 0 < a ∧ a < 2} =
  {a : ℝ | ∀ x, ¬(p x) → q a x ∧ ∃ y, q a y ∧ p y} :=
sorry

end intersection_A_complement_B_range_of_a_for_not_p_sufficient_not_necessary_for_q_l2723_272354


namespace x_not_greater_than_one_l2723_272399

theorem x_not_greater_than_one (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 := by
  sorry

end x_not_greater_than_one_l2723_272399


namespace city_population_change_l2723_272389

theorem city_population_change (n : ℕ) : 
  (0.85 * (n + 1500) : ℚ).floor = n - 50 → n = 8833 := by
  sorry

end city_population_change_l2723_272389


namespace distribute_five_to_three_l2723_272311

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  let c311 := (n.choose 3) * ((n - 3).choose 1) * ((n - 4).choose 1) / 2
  let c221 := (n.choose 2) * ((n - 2).choose 2) * ((n - 4).choose 1) / 2
  (c311 + c221) * 6

theorem distribute_five_to_three :
  distribute_objects 5 3 = 150 := by sorry

end distribute_five_to_three_l2723_272311
