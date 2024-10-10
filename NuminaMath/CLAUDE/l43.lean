import Mathlib

namespace parabola_intersects_x_axis_twice_l43_4330

theorem parabola_intersects_x_axis_twice (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ m * x₁^2 + (m - 3) * x₁ - 1 = 0 ∧ m * x₂^2 + (m - 3) * x₂ - 1 = 0 :=
by sorry

end parabola_intersects_x_axis_twice_l43_4330


namespace mrs_heine_biscuits_l43_4382

/-- Given a number of dogs and biscuits per dog, calculates the total number of biscuits needed -/
def total_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog

/-- Theorem: Mrs. Heine needs to buy 6 biscuits for her 2 dogs, given 3 biscuits per dog -/
theorem mrs_heine_biscuits : total_biscuits 2 3 = 6 := by
  sorry

end mrs_heine_biscuits_l43_4382


namespace small_triangle_area_ratio_l43_4393

/-- Represents a right triangle divided into a square and two smaller right triangles -/
structure DividedRightTriangle where
  /-- Area of the square -/
  square_area : ℝ
  /-- Area of the first small right triangle -/
  small_triangle1_area : ℝ
  /-- Area of the second small right triangle -/
  small_triangle2_area : ℝ
  /-- The first small triangle's area is n times the square's area -/
  small_triangle1_prop : small_triangle1_area = square_area * n
  /-- The square and two small triangles form a right triangle -/
  forms_right_triangle : square_area + small_triangle1_area + small_triangle2_area > 0

/-- 
If one small right triangle has an area n times the square's area, 
then the other small right triangle has an area 1/(4n) times the square's area 
-/
theorem small_triangle_area_ratio 
  (t : DividedRightTriangle) (n : ℝ) (hn : n > 0) :
  t.small_triangle2_area / t.square_area = 1 / (4 * n) := by
  sorry

end small_triangle_area_ratio_l43_4393


namespace practice_multiple_days_l43_4362

/-- Given a person who practices a constant amount each day, and 20 days ago had half as much
    practice as they have currently, prove that it takes 40(M - 1) days to reach M times
    their current practice. -/
theorem practice_multiple_days (d : ℝ) (P : ℝ) (M : ℝ) :
  (P / 2 + 20 * d = P) →  -- 20 days ago, had half as much practice
  (P = 40 * d) →          -- Current practice
  (∃ D : ℝ, D * d = M * P - P ∧ D = 40 * (M - 1)) :=
by sorry

end practice_multiple_days_l43_4362


namespace incorrect_operation_l43_4312

theorem incorrect_operation : 
  (5 - (-2) = 7) ∧ 
  (-9 / (-3) = 3) ∧ 
  (-4 * (-5) = 20) ∧ 
  (-5 + 3 ≠ 8) := by
sorry

end incorrect_operation_l43_4312


namespace unique_base_for_315_l43_4355

theorem unique_base_for_315 :
  ∃! b : ℕ, b ≥ 2 ∧ b^4 ≤ 315 ∧ 315 < b^5 :=
by sorry

end unique_base_for_315_l43_4355


namespace sum_and_product_reciprocal_sum_cube_surface_area_probability_white_ball_equilateral_triangle_area_l43_4338

-- Problem 1
theorem sum_and_product_reciprocal_sum (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem cube_surface_area (a : ℝ) :
  6 * (a + 1)^2 = 54 := by sorry

-- Problem 3
theorem probability_white_ball (b : ℝ) (c : ℝ) :
  (b - 4) / (2 * b + 42) = c / 6 := by sorry

-- Problem 4
theorem equilateral_triangle_area (c d : ℝ) :
  d * Real.sqrt 3 = (Real.sqrt 3 / 4) * c^2 := by sorry

end sum_and_product_reciprocal_sum_cube_surface_area_probability_white_ball_equilateral_triangle_area_l43_4338


namespace circle_inequality_l43_4336

/-- Given a circle with diameter AC = 1, AB tangent to the circle, and BC intersecting the circle again at D,
    prove that if AB = a and CD = b, then 1/(a^2 + 1/2) < b/a < 1/a^2 -/
theorem circle_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  1 / (a^2 + 1/2) < b / a ∧ b / a < 1 / a^2 := by
  sorry

#check circle_inequality

end circle_inequality_l43_4336


namespace coplanar_condition_l43_4364

open Vector

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (O P Q R S : V)

-- Define the coplanarity condition
def coplanar (P Q R S : V) : Prop :=
  ∃ (a b c d : ℝ), a • (P - O) + b • (Q - O) + c • (R - O) + d • (S - O) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)

-- State the theorem
theorem coplanar_condition (O P Q R S : V) :
  4 • (P - O) - 3 • (Q - O) + 6 • (R - O) + (-7) • (S - O) = 0 →
  coplanar O P Q R S :=
by sorry

end coplanar_condition_l43_4364


namespace minimum_k_value_l43_4377

theorem minimum_k_value : ∃ (k : ℝ), 
  (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → 
    (∃ (a b : ℝ), (a = x ∧ b = y) ∨ (a = x ∧ b = z) ∨ (a = y ∧ b = z) ∧ 
      (|a - b| ≤ k ∨ |1/a - 1/b| ≤ k))) ∧
  (∀ (k' : ℝ), 
    (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → 
      (∃ (a b : ℝ), (a = x ∧ b = y) ∨ (a = x ∧ b = z) ∨ (a = y ∧ b = z) ∧ 
        (|a - b| ≤ k' ∨ |1/a - 1/b| ≤ k'))) → 
    k ≤ k') ∧
  k = 3/2 := by
sorry

end minimum_k_value_l43_4377


namespace determine_h_of_x_l43_4381

theorem determine_h_of_x (x : ℝ) (h : ℝ → ℝ) : 
  (4 * x^4 + 5 * x^2 - 2 * x + 1 + h x = 6 * x^3 - 4 * x^2 + 7 * x - 5) → 
  (h x = -4 * x^4 + 6 * x^3 - 9 * x^2 + 9 * x - 6) := by
  sorry

end determine_h_of_x_l43_4381


namespace all_descendants_have_no_daughters_l43_4317

/-- Represents Bertha's family tree -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  great_granddaughters : ℕ

/-- The number of Bertha's daughters who have daughters -/
def daughters_with_daughters (f : BerthaFamily) : ℕ := f.granddaughters / 5

/-- The number of Bertha's descendants who have no daughters -/
def descendants_without_daughters (f : BerthaFamily) : ℕ :=
  f.daughters + f.granddaughters

theorem all_descendants_have_no_daughters (f : BerthaFamily) :
  f.daughters = 8 →
  f.daughters + f.granddaughters + f.great_granddaughters = 48 →
  f.great_granddaughters = 0 →
  daughters_with_daughters f * 5 = f.granddaughters →
  descendants_without_daughters f = f.daughters + f.granddaughters + f.great_granddaughters :=
by sorry

end all_descendants_have_no_daughters_l43_4317


namespace oil_distribution_l43_4310

theorem oil_distribution (total oil_A oil_B oil_C : ℕ) : 
  total = 3000 →
  oil_A = oil_B + 200 →
  oil_B = oil_C + 200 →
  total = oil_A + oil_B + oil_C →
  oil_B = 1000 := by
  sorry

end oil_distribution_l43_4310


namespace base5_digits_of_1234_l43_4389

/-- The number of digits in the base-5 representation of a positive integer n -/
def base5Digits (n : ℕ+) : ℕ :=
  Nat.log 5 n + 1

/-- Theorem: The number of digits in the base-5 representation of 1234 is 5 -/
theorem base5_digits_of_1234 : base5Digits 1234 = 5 := by
  sorry

end base5_digits_of_1234_l43_4389


namespace linear_function_preserves_arithmetic_progression_l43_4370

/-- A sequence (xₙ) is an arithmetic progression if there exists a constant d
    such that xₙ₊₁ = xₙ + d for all n. -/
def is_arithmetic_progression (x : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, x (n + 1) = x n + d

/-- A function f is linear if there exist constants k and b such that
    f(x) = kx + b for all x. -/
def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x : ℝ, f x = k * x + b

theorem linear_function_preserves_arithmetic_progression
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (hf : is_linear_function f)
  (hx : is_arithmetic_progression x) :
  is_arithmetic_progression (fun n ↦ f (x n)) :=
sorry

end linear_function_preserves_arithmetic_progression_l43_4370


namespace mobius_rest_stop_time_l43_4342

/-- Proves that the rest stop time for each half of the trip is 1 hour given the conditions of Mobius's journey --/
theorem mobius_rest_stop_time 
  (distance : ℝ) 
  (speed_with_load : ℝ) 
  (speed_without_load : ℝ) 
  (total_trip_time : ℝ) 
  (h1 : distance = 143) 
  (h2 : speed_with_load = 11) 
  (h3 : speed_without_load = 13) 
  (h4 : total_trip_time = 26) : 
  (total_trip_time - (distance / speed_with_load + distance / speed_without_load)) / 2 = 1 := by
sorry

end mobius_rest_stop_time_l43_4342


namespace group_size_is_nine_l43_4390

/-- The number of people in the original group -/
def n : ℕ := sorry

/-- The age of the person joining the group -/
def joining_age : ℕ := 34

/-- The original average age of the group -/
def original_average : ℕ := 14

/-- The new average age after the person joins -/
def new_average : ℕ := 16

/-- The minimum age in the group -/
def min_age : ℕ := 10

/-- There are two sets of twins in the group -/
axiom twin_sets : ∃ (a b : ℕ), (2 * a + 2 * b ≤ n)

/-- All individuals in the group are at least 10 years old -/
axiom all_above_min_age : ∀ (age : ℕ), age ≥ min_age

/-- The sum of ages in the original group -/
def original_sum : ℕ := n * original_average

/-- The sum of ages after the new person joins -/
def new_sum : ℕ := original_sum + joining_age

theorem group_size_is_nine :
  n * original_average + joining_age = new_average * (n + 1) →
  n = 9 := by sorry

end group_size_is_nine_l43_4390


namespace pizza_consumption_order_l43_4321

-- Define the fractions of pizza eaten by each friend
def samuel_fraction : ℚ := 1/6
def teresa_fraction : ℚ := 2/5
def uma_fraction : ℚ := 1/4

-- Define the amount of pizza eaten by Victor
def victor_fraction : ℚ := 1 - (samuel_fraction + teresa_fraction + uma_fraction)

-- Define a function to compare two fractions
def eats_more (a b : ℚ) : Prop := a > b

-- Theorem stating the order of pizza consumption
theorem pizza_consumption_order :
  eats_more teresa_fraction uma_fraction ∧
  eats_more uma_fraction victor_fraction ∧
  eats_more victor_fraction samuel_fraction :=
sorry

end pizza_consumption_order_l43_4321


namespace one_pair_probability_l43_4359

-- Define the total number of socks
def total_socks : ℕ := 12

-- Define the number of colors
def num_colors : ℕ := 4

-- Define the number of socks per color
def socks_per_color : ℕ := 3

-- Define the number of socks drawn
def socks_drawn : ℕ := 5

-- Define the probability of drawing exactly one pair of socks with the same color
def prob_one_pair : ℚ := 9/22

-- Theorem statement
theorem one_pair_probability :
  (total_socks = num_colors * socks_per_color) →
  (socks_drawn = 5) →
  (prob_one_pair = 9/22) := by
  sorry

end one_pair_probability_l43_4359


namespace initial_pens_l43_4360

theorem initial_pens (initial : ℕ) (mike_gives : ℕ) (cindy_doubles : ℕ → ℕ) (sharon_takes : ℕ) (final : ℕ) : 
  mike_gives = 22 →
  cindy_doubles = (· * 2) →
  sharon_takes = 19 →
  final = 75 →
  cindy_doubles (initial + mike_gives) - sharon_takes = final →
  initial = 25 :=
by sorry

end initial_pens_l43_4360


namespace circle_equation_solution_l43_4391

theorem circle_equation_solution (a b : ℝ) (h : a^2 + b^2 = 12*a - 4*b + 20) : a - b = 8 := by
  sorry

end circle_equation_solution_l43_4391


namespace sin_50_plus_sqrt3_tan_10_equals_1_l43_4366

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end sin_50_plus_sqrt3_tan_10_equals_1_l43_4366


namespace solution_mixture_l43_4376

/-- Proves that 112 ounces of Solution B is needed to create a 140-ounce mixture
    that is 80% salt when combined with Solution A (40% salt) --/
theorem solution_mixture (solution_a_salt_percentage : ℝ) (solution_b_salt_percentage : ℝ)
  (total_mixture_ounces : ℝ) (target_salt_percentage : ℝ) :
  solution_a_salt_percentage = 0.4 →
  solution_b_salt_percentage = 0.9 →
  total_mixture_ounces = 140 →
  target_salt_percentage = 0.8 →
  ∃ (solution_b_ounces : ℝ),
    solution_b_ounces = 112 ∧
    solution_b_ounces + (total_mixture_ounces - solution_b_ounces) = total_mixture_ounces ∧
    solution_a_salt_percentage * (total_mixture_ounces - solution_b_ounces) +
      solution_b_salt_percentage * solution_b_ounces =
      target_salt_percentage * total_mixture_ounces :=
by sorry


end solution_mixture_l43_4376


namespace parabola_shift_sum_l43_4314

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_sum (p : Parabola) :
  (shift (shift p 1 0) 0 2) = { a := 1, b := -4, c := 5 } →
  p.a + p.b + p.c = 1 := by
  sorry

end parabola_shift_sum_l43_4314


namespace least_m_for_x_bound_l43_4357

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 5) / (x n + 7)

theorem least_m_for_x_bound : 
  ∃ m : ℕ, m = 89 ∧ x m ≤ 5 + 1 / (2^10) ∧ ∀ k < m, x k > 5 + 1 / (2^10) :=
sorry

end least_m_for_x_bound_l43_4357


namespace parabola_vertex_position_l43_4373

/-- A parabola with points P, Q, and M satisfying specific conditions -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  m : ℝ
  h1 : y₁ = a * (-2)^2 + b * (-2) + c
  h2 : y₂ = a * 4^2 + b * 4 + c
  h3 : y₃ = a * m^2 + b * m + c
  h4 : 2 * a * m + b = 0
  h5 : y₃ ≥ y₂
  h6 : y₂ > y₁

theorem parabola_vertex_position (p : Parabola) : p.m > 1 := by
  sorry

end parabola_vertex_position_l43_4373


namespace correct_answer_is_105_l43_4304

theorem correct_answer_is_105 (x : ℕ) : 
  (x - 5 = 95) → (x + 5 = 105) :=
by
  sorry

end correct_answer_is_105_l43_4304


namespace fraction_equals_zero_l43_4306

theorem fraction_equals_zero (x : ℝ) : 
  (x^2 - 4) / (x - 2) = 0 ∧ x ≠ 2 → x = -2 :=
by sorry

end fraction_equals_zero_l43_4306


namespace max_value_theorem_l43_4361

theorem max_value_theorem (a b : ℝ) 
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ M : ℝ, M = 80 ∧ 
    (∀ a' b' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a' * x + b'| ≤ 1) → 
      |20 * a' + 14 * b'| + |20 * a' - 14 * b'| ≤ M) ∧
    (∃ a' b' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a' * x + b'| ≤ 1) ∧ 
      |20 * a' + 14 * b'| + |20 * a' - 14 * b'| = M) :=
by
  sorry

end max_value_theorem_l43_4361


namespace fiona_earnings_l43_4340

-- Define the time worked each day in hours
def monday_hours : ℝ := 1.5
def tuesday_hours : ℝ := 1.25
def wednesday_hours : ℝ := 3.1667
def thursday_hours : ℝ := 0.75

-- Define the hourly rate
def hourly_rate : ℝ := 4

-- Define the total hours worked
def total_hours : ℝ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours

-- Define the weekly earnings
def weekly_earnings : ℝ := total_hours * hourly_rate

-- Theorem statement
theorem fiona_earnings : 
  ∃ ε > 0, |weekly_earnings - 26.67| < ε :=
sorry

end fiona_earnings_l43_4340


namespace z_relation_to_x_minus_2y_l43_4372

theorem z_relation_to_x_minus_2y (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : z = (x + 3) - 2 * (y - 5)) :
  z = x - 2 * y + 13 := by
  sorry

end z_relation_to_x_minus_2y_l43_4372


namespace square_area_unchanged_l43_4329

theorem square_area_unchanged (k : ℝ) : k > 0 → k^2 = 1 → k = 1 := by sorry

end square_area_unchanged_l43_4329


namespace crayons_per_row_l43_4320

theorem crayons_per_row (rows : ℕ) (pencils_per_row : ℕ) (total_items : ℕ) 
  (h1 : rows = 11)
  (h2 : pencils_per_row = 31)
  (h3 : total_items = 638) :
  (total_items - rows * pencils_per_row) / rows = 27 := by
  sorry

end crayons_per_row_l43_4320


namespace scalene_triangle_gp_ratio_bounds_l43_4328

/-- A scalene triangle with sides in geometric progression -/
structure ScaleneTriangleGP where
  -- The first side of the triangle
  a : ℝ
  -- The common ratio of the geometric progression
  q : ℝ
  -- Ensure the triangle is scalene and sides are positive
  h_scalene : a ≠ a * q ∧ a * q ≠ a * q^2 ∧ a ≠ a * q^2 ∧ a > 0 ∧ q > 0

/-- The common ratio of a scalene triangle with sides in geometric progression
    must be between (1 - √5)/2 and (1 + √5)/2 -/
theorem scalene_triangle_gp_ratio_bounds (t : ScaleneTriangleGP) :
  (1 - Real.sqrt 5) / 2 < t.q ∧ t.q < (1 + Real.sqrt 5) / 2 :=
by sorry

end scalene_triangle_gp_ratio_bounds_l43_4328


namespace central_cell_value_l43_4394

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end central_cell_value_l43_4394


namespace rectangular_solid_volume_l43_4387

theorem rectangular_solid_volume (x y z : ℝ) 
  (h1 : x * y = 15)  -- Area of side face
  (h2 : y * z = 10)  -- Area of front face
  (h3 : x * z = 6)   -- Area of bottom face
  : x * y * z = 30 := by
  sorry

end rectangular_solid_volume_l43_4387


namespace animals_per_aquarium_l43_4369

/-- Given that Tyler has 56 saltwater aquariums and 2184 saltwater animals,
    prove that there are 39 animals in each saltwater aquarium. -/
theorem animals_per_aquarium (saltwater_aquariums : ℕ) (saltwater_animals : ℕ)
    (h1 : saltwater_aquariums = 56)
    (h2 : saltwater_animals = 2184) :
    saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end animals_per_aquarium_l43_4369


namespace calculate_expression_l43_4347

theorem calculate_expression : ((15^10 / 15^9)^3 * 5^3) / 3^3 = 15625 := by
  sorry

end calculate_expression_l43_4347


namespace markup_calculation_l43_4399

/-- The markup required for an article with given purchase price, overhead percentage, and desired net profit. -/
def required_markup (purchase_price : ℝ) (overhead_percent : ℝ) (net_profit : ℝ) : ℝ :=
  purchase_price * overhead_percent + net_profit

/-- Theorem stating that the required markup for the given conditions is $34.80 -/
theorem markup_calculation :
  required_markup 48 0.35 18 = 34.80 := by
  sorry

end markup_calculation_l43_4399


namespace inscribed_triangle_area_l43_4344

/-- The area of a triangle inscribed in a circle, where the triangle's vertices
    divide the circle into three arcs of lengths 4, 5, and 7. -/
theorem inscribed_triangle_area : ∃ (A : ℝ), 
  (∀ (r : ℝ), r > 0 → r = 8 / Real.pi → 
    ∃ (θ₁ θ₂ θ₃ : ℝ), 
      θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧
      4 * r = 4 * θ₁ ∧
      5 * r = 5 * θ₂ ∧
      7 * r = 7 * θ₃ ∧
      θ₁ + θ₂ + θ₃ = 2 * Real.pi ∧
      A = (1/2) * r^2 * (Real.sin (2*θ₁) + Real.sin (2*(θ₁+θ₂)) + Real.sin (2*(θ₁+θ₂+θ₃)))) ∧
  A = 147.6144 / Real.pi^2 := by
sorry

end inscribed_triangle_area_l43_4344


namespace class_size_l43_4396

theorem class_size (total : ℝ) 
  (h1 : 0.25 * total = total - (0.75 * total))
  (h2 : 0.1875 * total = 0.25 * (0.75 * total))
  (h3 : 18 = 0.75 * total - 0.1875 * total) : 
  total = 32 := by
  sorry

end class_size_l43_4396


namespace complex_square_sum_l43_4368

theorem complex_square_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (2 - i)^2 = a + b * i^3 → a + b = 7 := by
  sorry

end complex_square_sum_l43_4368


namespace ellipse_region_area_l43_4335

/-- The area of the region formed by all points on ellipses passing through (√3, 1) where y ≥ 1 -/
theorem ellipse_region_area :
  ∀ a b : ℝ,
  a ≥ b ∧ b > 0 →
  (3 / a^2) + (1 / b^2) = 1 →
  (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ y ≥ 1) →
  (∃ area : ℝ, area = 4 * Real.pi / 3 - Real.sqrt 3) :=
by sorry

end ellipse_region_area_l43_4335


namespace empty_solution_set_implies_a_leq_neg_two_l43_4392

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 1

-- State the theorem
theorem empty_solution_set_implies_a_leq_neg_two (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) → a ≤ -2 := by
  sorry

end empty_solution_set_implies_a_leq_neg_two_l43_4392


namespace lcm_48_90_l43_4375

theorem lcm_48_90 : Nat.lcm 48 90 = 720 := by
  sorry

end lcm_48_90_l43_4375


namespace abc_product_l43_4388

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 160) (h2 : b * (c + a) = 168) (h3 : c * (a + b) = 180) :
  a * b * c = 772 := by
  sorry

end abc_product_l43_4388


namespace juice_bar_group_size_l43_4309

theorem juice_bar_group_size :
  let total_spent : ℕ := 94
  let mango_price : ℕ := 5
  let pineapple_price : ℕ := 6
  let pineapple_spent : ℕ := 54
  let mango_spent : ℕ := total_spent - pineapple_spent
  let mango_people : ℕ := mango_spent / mango_price
  let pineapple_people : ℕ := pineapple_spent / pineapple_price
  mango_people + pineapple_people = 17 :=
by sorry

end juice_bar_group_size_l43_4309


namespace cos_thirty_degrees_l43_4383

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_thirty_degrees_l43_4383


namespace sqrt_equation_solution_l43_4384

theorem sqrt_equation_solution : 
  {x : ℝ | Real.sqrt (2*x - 4) - Real.sqrt (x + 5) = 1} = {4, 20} := by
  sorry

end sqrt_equation_solution_l43_4384


namespace largest_of_six_consecutive_odds_l43_4349

theorem largest_of_six_consecutive_odds (a : ℕ) (h1 : a > 0) 
  (h2 : a % 2 = 1) 
  (h3 : (a * (a + 2) * (a + 4) * (a + 6) * (a + 8) * (a + 10) = 135135)) : 
  a + 10 = 13 := by
  sorry

end largest_of_six_consecutive_odds_l43_4349


namespace unique_natural_pair_l43_4363

theorem unique_natural_pair : ∃! (a b : ℕ), 
  a ≠ b ∧ 
  (∃ (k : ℕ), ∃ (p : ℕ), Prime p ∧ b^2 + a = p^k) ∧
  (∃ (m : ℕ), (a^2 + b) * m = b^2 + a) ∧
  a = 2 ∧ 
  b = 5 := by
sorry

end unique_natural_pair_l43_4363


namespace subtracted_number_l43_4327

theorem subtracted_number (x y : ℤ) : x = 125 ∧ 2 * x - y = 112 → y = 138 := by
  sorry

end subtracted_number_l43_4327


namespace cone_volume_l43_4398

/-- A cone with lateral area √5π and whose unfolded lateral area forms a sector with central angle 2√5π/5 has volume 2π/3 -/
theorem cone_volume (lateral_area : ℝ) (central_angle : ℝ) :
  lateral_area = Real.sqrt 5 * Real.pi →
  central_angle = 2 * Real.sqrt 5 * Real.pi / 5 →
  ∃ (r h : ℝ), 
    r > 0 ∧ h > 0 ∧
    lateral_area = Real.pi * r * Real.sqrt (r^2 + h^2) ∧
    (1/3) * Real.pi * r^2 * h = (2/3) * Real.pi :=
by sorry

end cone_volume_l43_4398


namespace noodle_portions_l43_4319

-- Define the variables
def total_spent : ℕ := 3000
def total_portions : ℕ := 170
def price_mixed : ℕ := 15
def price_beef : ℕ := 20

-- Define the theorem
theorem noodle_portions :
  ∃ (mixed beef : ℕ),
    mixed + beef = total_portions ∧
    price_mixed * mixed + price_beef * beef = total_spent ∧
    mixed = 80 ∧
    beef = 90 := by
  sorry

end noodle_portions_l43_4319


namespace shaded_area_of_square_shaded_percentage_l43_4323

/-- The shaded area of a square with side length 6 units -/
theorem shaded_area_of_square (side_length : ℝ) (shaded_square : ℝ) (shaded_region : ℝ) (shaded_strip : ℝ) : 
  side_length = 6 →
  shaded_square = 2^2 →
  shaded_region = 5^2 - 3^2 →
  shaded_strip = 6 * 1 →
  shaded_square + shaded_region + shaded_strip = 26 := by
sorry

/-- The percentage of the square that is shaded -/
theorem shaded_percentage (total_area : ℝ) (shaded_area : ℝ) :
  total_area = 6^2 →
  shaded_area = 26 →
  (shaded_area / total_area) * 100 = 72.22 := by
sorry

end shaded_area_of_square_shaded_percentage_l43_4323


namespace business_school_majors_l43_4348

theorem business_school_majors (p q r s : ℕ) (h1 : p * q * r * s = 1365) (h2 : p = 3) :
  q * r * s = 455 := by
  sorry

end business_school_majors_l43_4348


namespace max_books_buyable_l43_4332

def total_money : ℚ := 24.41
def book_price : ℚ := 2.75

theorem max_books_buyable : 
  ∀ n : ℕ, n * book_price ≤ total_money ∧ 
  (n + 1) * book_price > total_money → n = 8 := by
sorry

end max_books_buyable_l43_4332


namespace f_composition_value_l43_4386

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x + 1

theorem f_composition_value : f (f 2) = 78652 := by
  sorry

end f_composition_value_l43_4386


namespace lines_perpendicular_to_plane_are_parallel_l43_4322

-- Define the plane and lines
variable (α : Plane)
variable (m n : Line)

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel :
  perpendicular m α → perpendicular n α → parallel m n := by sorry

end lines_perpendicular_to_plane_are_parallel_l43_4322


namespace journey_time_proof_l43_4326

theorem journey_time_proof (highway_distance : ℝ) (mountain_distance : ℝ) 
  (speed_ratio : ℝ) (mountain_time : ℝ) :
  highway_distance = 60 →
  mountain_distance = 20 →
  speed_ratio = 4 →
  mountain_time = 40 →
  highway_distance / (speed_ratio * (mountain_distance / mountain_time)) + mountain_time = 70 :=
by sorry

end journey_time_proof_l43_4326


namespace solve_for_x_l43_4307

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end solve_for_x_l43_4307


namespace triangle_side_lengths_l43_4325

/-- Given a function f and a triangle ABC, prove the lengths of sides b and c. -/
theorem triangle_side_lengths 
  (f : ℝ → ℝ) 
  (vec_a vec_b : ℝ → ℝ × ℝ)
  (A B C : ℝ) 
  (a b c : ℝ) :
  (∀ x, f x = (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2) →
  (∀ x, vec_a x = (2 * Real.cos x, -Real.sqrt 3 * Real.sin (2 * x))) →
  (∀ x, vec_b x = (Real.cos x, 1)) →
  f A = -1 →
  a = Real.sqrt 7 / 2 →
  ∃ (k : ℝ), 3 * Real.sin C = 2 * Real.sin B →
  b = 3/2 ∧ c = 1 := by
  sorry

end triangle_side_lengths_l43_4325


namespace chess_competition_probabilities_l43_4352

/-- Scoring system for the chess competition -/
structure ScoringSystem where
  win : Nat
  lose : Nat
  draw : Nat

/-- Probabilities for player A in a single game -/
structure PlayerProbabilities where
  win : Real
  lose : Real
  draw : Real

/-- Function to calculate the probability of player A scoring exactly 2 points in two games -/
def prob_A_scores_2 (s : ScoringSystem) (p : PlayerProbabilities) : Real :=
  sorry

/-- Function to calculate the probability of player B scoring at least 2 points in two games -/
def prob_B_scores_at_least_2 (s : ScoringSystem) (p : PlayerProbabilities) : Real :=
  sorry

theorem chess_competition_probabilities 
  (s : ScoringSystem) 
  (p : PlayerProbabilities) 
  (h1 : s.win = 2 ∧ s.lose = 0 ∧ s.draw = 1)
  (h2 : p.win = 0.5 ∧ p.lose = 0.3 ∧ p.draw = 0.2)
  (h3 : p.win + p.lose + p.draw = 1) :
  prob_A_scores_2 s p = 0.34 ∧ prob_B_scores_at_least_2 s p = 0.55 := by
  sorry

end chess_competition_probabilities_l43_4352


namespace min_value_of_function_l43_4331

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∃ y : ℝ, y = 9 ∧ ∀ z : ℝ, (4 / x + 1 / (1 - x)) ≥ y := by
  sorry

end min_value_of_function_l43_4331


namespace quadratic_discriminant_l43_4316

theorem quadratic_discriminant : 
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 4
  (b^2 - 4*a*c) = 33 := by
sorry

end quadratic_discriminant_l43_4316


namespace widget_purchase_problem_l43_4311

theorem widget_purchase_problem (C W : ℚ) 
  (h1 : 8 * (C - 1.25) = 16.67)
  (h2 : 16.67 / C = W) : 
  W = 5 := by
sorry

end widget_purchase_problem_l43_4311


namespace largest_possible_median_l43_4324

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 6, 4, 7}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ i => i ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ i => i ≥ m)).card ≥ s.card

theorem largest_possible_median :
  ∃ (x : ℤ), is_median 7 (number_set x) ∧
  ∀ (y : ℤ) (m : ℤ), is_median m (number_set y) → m ≤ 7 :=
by sorry

end largest_possible_median_l43_4324


namespace red_packet_probability_l43_4374

def red_packet_amounts : List ℝ := [1.49, 1.31, 2.19, 3.40, 0.61]

def total_amount : ℝ := 9

def num_people : ℕ := 5

def threshold : ℝ := 4

def probability_ab_sum_ge_threshold (amounts : List ℝ) (total : ℝ) (n : ℕ) (t : ℝ) : ℚ :=
  sorry

theorem red_packet_probability :
  probability_ab_sum_ge_threshold red_packet_amounts total_amount num_people threshold = 2/5 :=
sorry

end red_packet_probability_l43_4374


namespace final_student_count_l43_4385

/-- The number of students in Beth's class at different stages --/
def students_in_class (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the final number of students in Beth's class --/
theorem final_student_count :
  students_in_class 150 30 15 = 165 := by
  sorry

end final_student_count_l43_4385


namespace wendys_washing_machine_capacity_l43_4354

-- Define the number of shirts
def shirts : ℕ := 39

-- Define the number of sweaters
def sweaters : ℕ := 33

-- Define the number of loads
def loads : ℕ := 9

-- Define the function to calculate the washing machine capacity
def washing_machine_capacity (s : ℕ) (w : ℕ) (l : ℕ) : ℕ :=
  (s + w) / l

-- Theorem statement
theorem wendys_washing_machine_capacity :
  washing_machine_capacity shirts sweaters loads = 8 := by
  sorry

end wendys_washing_machine_capacity_l43_4354


namespace soccer_enjoyment_fraction_l43_4343

theorem soccer_enjoyment_fraction (total : ℝ) (h_total : total > 0) :
  let enjoy_soccer := 0.7 * total
  let dont_enjoy_soccer := 0.3 * total
  let say_enjoy := 0.75 * enjoy_soccer
  let enjoy_but_say_dont := 0.25 * enjoy_soccer
  let say_dont_enjoy := 0.85 * dont_enjoy_soccer
  let total_say_dont := say_dont_enjoy + enjoy_but_say_dont
  enjoy_but_say_dont / total_say_dont = 35 / 86 := by
sorry

end soccer_enjoyment_fraction_l43_4343


namespace smallest_number_l43_4334

theorem smallest_number (S : Set ℕ) (h : S = {10, 11, 12, 13, 14}) : 
  ∃ n ∈ S, ∀ m ∈ S, n ≤ m ∧ n = 10 := by
  sorry

end smallest_number_l43_4334


namespace negation_of_existence_square_leq_power_of_two_negation_l43_4303

theorem negation_of_existence (p : Nat → Prop) :
  (¬ ∃ n : Nat, p n) ↔ (∀ n : Nat, ¬ p n) := by sorry

theorem square_leq_power_of_two_negation :
  (¬ ∃ n : Nat, n^2 > 2^n) ↔ (∀ n : Nat, n^2 ≤ 2^n) := by sorry

end negation_of_existence_square_leq_power_of_two_negation_l43_4303


namespace clive_olive_money_l43_4337

/-- Proves that Clive has $10.00 to spend on olives given the problem conditions -/
theorem clive_olive_money : 
  -- Define the given conditions
  let olives_needed : ℕ := 80
  let olives_per_jar : ℕ := 20
  let cost_per_jar : ℚ := 3/2  -- $1.50 represented as a rational number
  let change : ℚ := 4  -- $4.00 change

  -- Calculate the number of jars needed
  let jars_needed : ℕ := olives_needed / olives_per_jar

  -- Calculate the total cost of olives
  let total_cost : ℚ := jars_needed * cost_per_jar

  -- Define Clive's total money as the sum of total cost and change
  let clive_money : ℚ := total_cost + change

  -- Prove that Clive's total money is $10.00
  clive_money = 10 := by sorry

end clive_olive_money_l43_4337


namespace B_power_66_l43_4318

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; -1, 0, 0; 0, 0, 1]

theorem B_power_66 : B ^ 66 = !![(-1 : ℝ), 0, 0; 0, -1, 0; 0, 0, 1] := by
  sorry

end B_power_66_l43_4318


namespace participation_and_optimality_l43_4339

/-- Represents a company in country A --/
structure Company where
  investmentCost : ℝ
  successProbability : ℝ
  potentialRevenue : ℝ

/-- Conditions for the problem --/
axiom probability_bounds {α : ℝ} : 0 < α ∧ α < 1

/-- Expected income when both companies participate --/
def expectedIncomeBoth (c : Company) : ℝ :=
  c.successProbability * (1 - c.successProbability) * c.potentialRevenue +
  0.5 * c.successProbability^2 * c.potentialRevenue

/-- Expected income when only one company participates --/
def expectedIncomeOne (c : Company) : ℝ :=
  c.successProbability * c.potentialRevenue

/-- Condition for a company to participate --/
def willParticipate (c : Company) : Prop :=
  expectedIncomeBoth c - c.investmentCost ≥ 0

/-- Social welfare as total profit of both companies --/
def socialWelfare (c1 c2 : Company) : ℝ :=
  2 * (expectedIncomeBoth c1 - c1.investmentCost)

/-- Theorem stating both companies will participate and it's not socially optimal --/
theorem participation_and_optimality (c1 c2 : Company)
  (h1 : c1.potentialRevenue = 24 ∧ c1.successProbability = 0.5 ∧ c1.investmentCost = 7)
  (h2 : c2.potentialRevenue = 24 ∧ c2.successProbability = 0.5 ∧ c2.investmentCost = 7) :
  willParticipate c1 ∧ willParticipate c2 ∧
  socialWelfare c1 c2 < expectedIncomeOne c1 - c1.investmentCost := by
  sorry

end participation_and_optimality_l43_4339


namespace place_value_ratio_l43_4358

def number : ℚ := 86304.2957

theorem place_value_ratio :
  let thousands_place_value : ℚ := 1000
  let tenths_place_value : ℚ := 0.1
  (thousands_place_value / tenths_place_value : ℚ) = 10000 := by
  sorry

end place_value_ratio_l43_4358


namespace area_of_trapezoid_psrt_l43_4341

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the trapezoid PSRT -/
structure Trapezoid where
  area : ℝ

/-- Represents the diagram configuration -/
structure DiagramConfig where
  pqr : Triangle
  smallestTriangles : Finset Triangle
  psrt : Trapezoid

/-- The main theorem statement -/
theorem area_of_trapezoid_psrt (config : DiagramConfig) : config.psrt.area = 53.5 :=
  by
  have h1 : config.pqr.area = 72 := by sorry
  have h2 : config.smallestTriangles.card = 9 := by sorry
  have h3 : ∀ t ∈ config.smallestTriangles, t.area = 2 := by sorry
  have h4 : ∀ t : Triangle, t ∈ config.smallestTriangles → t.area ≤ config.pqr.area := by sorry
  sorry

/-- Auxiliary definition for isosceles triangle -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Auxiliary definition for triangle similarity -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Additional properties of the configuration -/
axiom pqr_is_isosceles (config : DiagramConfig) : isIsosceles config.pqr
axiom all_triangles_similar (config : DiagramConfig) (t : Triangle) : 
  t ∈ config.smallestTriangles → areSimilar t config.pqr

end area_of_trapezoid_psrt_l43_4341


namespace ellipse_intersection_sum_of_squares_l43_4302

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  a : ℝ
  b : ℝ

def Ellipse.standard : Ellipse := { a := 2, b := 1 }

/-- Check if a point is on the ellipse -/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a line intersects the ellipse -/
def Line.intersectsEllipse (l : Line) (e : Ellipse) : Prop :=
  ∃ p : Point, e.contains p ∧ p.y = l.slope * p.x + l.intercept

/-- Calculate the distance squared from origin to a point -/
def Point.distanceSquared (p : Point) : ℝ :=
  p.x^2 + p.y^2

/-- Theorem statement -/
theorem ellipse_intersection_sum_of_squares :
  ∀ (l : Line),
    l.slope = 1/2 ∨ l.slope = -1/2 →
    l.intercept ≠ 0 →
    l.intersectsEllipse Ellipse.standard →
    ∃ (p1 p2 : Point),
      Ellipse.standard.contains p1 ∧
      Ellipse.standard.contains p2 ∧
      p1 ≠ p2 ∧
      p1.y = l.slope * p1.x + l.intercept ∧
      p2.y = l.slope * p2.x + l.intercept ∧
      p1.distanceSquared + p2.distanceSquared = 5 :=
sorry

end ellipse_intersection_sum_of_squares_l43_4302


namespace spongebob_earnings_l43_4397

/-- Represents the earnings from selling burgers -/
def burger_earnings (num_burgers : ℕ) (price_per_burger : ℚ) : ℚ :=
  num_burgers * price_per_burger

/-- Represents the earnings from selling large fries -/
def fries_earnings (num_fries : ℕ) (price_per_fries : ℚ) : ℚ :=
  num_fries * price_per_fries

/-- Represents the total earnings for the day -/
def total_earnings (burger_earn : ℚ) (fries_earn : ℚ) : ℚ :=
  burger_earn + fries_earn

theorem spongebob_earnings :
  let num_burgers : ℕ := 30
  let price_per_burger : ℚ := 2
  let num_fries : ℕ := 12
  let price_per_fries : ℚ := 3/2
  let burger_earn := burger_earnings num_burgers price_per_burger
  let fries_earn := fries_earnings num_fries price_per_fries
  total_earnings burger_earn fries_earn = 78 := by
sorry

end spongebob_earnings_l43_4397


namespace min_value_expression_min_value_attainable_l43_4395

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  (5 * (a + b)^2 + 4 * (b - c)^2 + 3 * (c - a)^2) / (2 * b^2) ≥ 24 :=
by sorry

theorem min_value_attainable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), b > c ∧ c > a ∧ b ≠ 0 ∧
  (5 * (a + b)^2 + 4 * (b - c)^2 + 3 * (c - a)^2) / (2 * b^2) < 24 + ε :=
by sorry

end min_value_expression_min_value_attainable_l43_4395


namespace probability_both_truth_l43_4346

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h1 : prob_A = 0.8) (h2 : prob_B = 0.6) :
  prob_A * prob_B = 0.48 := by
sorry

end probability_both_truth_l43_4346


namespace sandcastle_height_difference_l43_4305

/-- The height difference between Janet's sandcastle and her sister's sandcastle -/
def height_difference : ℝ := 1.333333333333333

/-- Janet's sandcastle height in feet -/
def janet_height : ℝ := 3.6666666666666665

/-- Janet's sister's sandcastle height in feet -/
def sister_height : ℝ := 2.3333333333333335

/-- Theorem stating that the height difference between Janet's sandcastle and her sister's sandcastle
    is equal to Janet's sandcastle height minus her sister's sandcastle height -/
theorem sandcastle_height_difference :
  height_difference = janet_height - sister_height := by
  sorry

end sandcastle_height_difference_l43_4305


namespace complex_abs_calculation_l43_4308

def z : ℂ := 7 + 3 * Complex.I

theorem complex_abs_calculation : Complex.abs (z^2 + 4*z + 40) = 54 * Real.sqrt 5 := by
  sorry

end complex_abs_calculation_l43_4308


namespace smallest_side_difference_l43_4315

def is_valid_triangle (pq qr pr : ℕ) : Prop :=
  pq + qr > pr ∧ pq + pr > qr ∧ qr + pr > pq

theorem smallest_side_difference (pq qr pr : ℕ) :
  pq + qr + pr = 3030 →
  pq < qr →
  qr ≤ pr →
  is_valid_triangle pq qr pr →
  (∀ pq' qr' pr' : ℕ, 
    pq' + qr' + pr' = 3030 →
    pq' < qr' →
    qr' ≤ pr' →
    is_valid_triangle pq' qr' pr' →
    qr - pq ≤ qr' - pq') →
  qr - pq = 15 :=
by sorry

end smallest_side_difference_l43_4315


namespace not_necessary_not_sufficient_l43_4365

theorem not_necessary_not_sufficient (a : ℝ) : 
  ¬(∀ a, a < 2 → a^2 < 2*a) ∧ ¬(∀ a, a^2 < 2*a → a < 2) := by
  sorry

end not_necessary_not_sufficient_l43_4365


namespace solution_set_part1_range_of_a_part2_l43_4351

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Solution set of f(x) ≥ 6 when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a for which f(x) > -a for all x
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_part2_l43_4351


namespace min_envelopes_correct_l43_4371

/-- The number of different flags -/
def num_flags : ℕ := 12

/-- The number of flags in each envelope -/
def flags_per_envelope : ℕ := 2

/-- The probability threshold for having a repeated flag -/
def probability_threshold : ℚ := 1/2

/-- Calculates the probability of all flags being different when opening k envelopes -/
def prob_all_different (k : ℕ) : ℚ :=
  (num_flags.descFactorial (k * flags_per_envelope)) / (num_flags ^ (k * flags_per_envelope))

/-- The minimum number of envelopes to open -/
def min_envelopes : ℕ := 3

theorem min_envelopes_correct :
  (∀ k < min_envelopes, prob_all_different k > probability_threshold) ∧
  (prob_all_different min_envelopes ≤ probability_threshold) :=
sorry

end min_envelopes_correct_l43_4371


namespace smallest_n_for_g_nine_l43_4379

/-- Sum of digits in base 5 representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 9 representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- The smallest positive integer n such that g(n) = 9 -/
theorem smallest_n_for_g_nine : 
  (∀ m : ℕ, m > 0 ∧ m < 344 → g m ≠ 9) ∧ g 344 = 9 := by sorry

end smallest_n_for_g_nine_l43_4379


namespace cafeteria_pies_l43_4333

theorem cafeteria_pies (total_apples : Real) (handed_out : Real) (apples_per_pie : Real) 
  (h1 : total_apples = 135.5)
  (h2 : handed_out = 89.75)
  (h3 : apples_per_pie = 5.25) :
  ⌊(total_apples - handed_out) / apples_per_pie⌋ = 8 := by
  sorry

end cafeteria_pies_l43_4333


namespace max_sum_of_coefficients_l43_4378

/-- Given a temperature function T(t) = a * sin(t) + b * cos(t) where a and b are positive real
    numbers, and the maximum temperature difference is 10 degrees Celsius, 
    the maximum value of a + b is 5√2. -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ t : ℝ, t > 0 → ∃ T : ℝ, T = a * Real.sin t + b * Real.cos t) →
  (∃ t₁ t₂ : ℝ, t₁ > 0 ∧ t₂ > 0 ∧ 
    (a * Real.sin t₁ + b * Real.cos t₁) - (a * Real.sin t₂ + b * Real.cos t₂) = 10) →
  a + b ≤ 5 * Real.sqrt 2 :=
by sorry

end max_sum_of_coefficients_l43_4378


namespace smallest_sticker_collection_l43_4301

theorem smallest_sticker_collection : ∃ (S : ℕ), 
  S > 2 ∧
  S % 4 = 2 ∧
  S % 6 = 2 ∧
  S % 9 = 2 ∧
  S % 10 = 2 ∧
  (∀ (T : ℕ), T > 2 → T % 4 = 2 → T % 6 = 2 → T % 9 = 2 → T % 10 = 2 → S ≤ T) ∧
  S = 182 := by
  sorry

end smallest_sticker_collection_l43_4301


namespace triangle_area_is_24_l43_4300

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (0, 8)

-- Define the equation
def satisfies_equation (p : ℝ × ℝ) : Prop :=
  |4 * p.1| + |3 * p.2| + |24 - 4 * p.1 - 3 * p.2| = 24

-- Theorem statement
theorem triangle_area_is_24 :
  satisfies_equation A ∧ satisfies_equation B ∧ satisfies_equation C →
  (1/2 : ℝ) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)| = 24 :=
by sorry

end triangle_area_is_24_l43_4300


namespace geometric_sequence_sum_inequality_l43_4313

def a (n : ℕ+) : ℝ := 3 * 2^(n.val - 1)

def S (n : ℕ+) : ℝ := 3 * (2^n.val - 1)

theorem geometric_sequence_sum_inequality {k : ℝ} :
  (∀ n : ℕ+, a (n + 1) + a n = 9 * 2^(n.val - 1)) →
  (∀ n : ℕ+, S n > k * a n - 2) →
  k < 5/3 :=
by sorry

end geometric_sequence_sum_inequality_l43_4313


namespace phd_time_calculation_l43_4353

/-- Calculates the total time John spent on his PhD --/
def total_phd_time (acclimation_time : ℝ) (basics_time : ℝ) (research_multiplier : ℝ) 
  (sabbatical_time : ℝ) (dissertation_fraction : ℝ) (conference_time : ℝ) : ℝ :=
  let research_time := basics_time * (1 + research_multiplier) + sabbatical_time
  let dissertation_time := acclimation_time * dissertation_fraction + conference_time
  acclimation_time + basics_time + research_time + dissertation_time

theorem phd_time_calculation :
  total_phd_time 1 2 0.75 0.5 0.5 0.25 = 7.75 := by
  sorry

end phd_time_calculation_l43_4353


namespace unique_solution_exists_l43_4350

/-- A function satisfying the given functional equation for a constant k -/
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = x + y + k

/-- The theorem stating the uniqueness and form of the solution -/
theorem unique_solution_exists (k : ℝ) :
  ∃! f : ℝ → ℝ, SatisfiesEquation f k ∧ ∀ x : ℝ, f x = x - k :=
sorry

end unique_solution_exists_l43_4350


namespace exists_committees_with_common_members_l43_4380

/-- Represents a committee system with members and committees. -/
structure CommitteeSystem where
  members : Finset ℕ
  committees : Finset (Finset ℕ)
  member_count : members.card = 1600
  committee_count : committees.card = 16000
  committee_size : ∀ c ∈ committees, c.card = 80

/-- Theorem stating that in a committee system satisfying the given conditions,
    there exist at least two committees sharing at least 4 members. -/
theorem exists_committees_with_common_members (cs : CommitteeSystem) :
  ∃ (c1 c2 : Finset ℕ), c1 ∈ cs.committees ∧ c2 ∈ cs.committees ∧ c1 ≠ c2 ∧
  (c1 ∩ c2).card ≥ 4 := by
  sorry

end exists_committees_with_common_members_l43_4380


namespace hayden_ride_payment_l43_4356

def hayden_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (gas_gallons : ℝ) (gas_price : ℝ) 
  (num_reviews : ℕ) (review_bonus : ℝ) (num_rides : ℕ) (total_owed : ℝ) : Prop :=
  let base_earnings := hourly_wage * hours_worked + gas_gallons * gas_price + num_reviews * review_bonus
  let ride_earnings := total_owed - base_earnings
  ride_earnings / num_rides = 5

theorem hayden_ride_payment : 
  hayden_earnings 15 8 17 3 2 20 3 226 := by sorry

end hayden_ride_payment_l43_4356


namespace characterization_of_M_inequality_for_M_elements_l43_4345

-- Define the set M
def M : Set ℝ := {x : ℝ | |2*x - 1| < 1}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by sorry

end characterization_of_M_inequality_for_M_elements_l43_4345


namespace point_A_coordinates_l43_4367

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation to the left -/
def translateLeft (p : Point) (dx : ℝ) : Point :=
  ⟨p.x - dx, p.y⟩

/-- Translation upwards -/
def translateUp (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

theorem point_A_coordinates :
  ∃ (A : Point),
    ∃ (dx dy : ℝ),
      translateLeft A dx = Point.mk 1 2 ∧
      translateUp A dy = Point.mk 3 4 ∧
      A = Point.mk 3 2 := by
  sorry

end point_A_coordinates_l43_4367
