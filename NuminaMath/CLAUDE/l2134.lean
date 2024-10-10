import Mathlib

namespace negation_of_proposition_l2134_213414

theorem negation_of_proposition :
  (¬∃ x : ℝ, x > 0 ∧ Real.sin x > 2^x - 1) ↔ (∀ x : ℝ, x > 0 → Real.sin x ≤ 2^x - 1) :=
by sorry

end negation_of_proposition_l2134_213414


namespace arithmetic_mean_of_a_and_b_l2134_213432

theorem arithmetic_mean_of_a_and_b (a b : ℝ) : 
  a = Real.sqrt 3 + Real.sqrt 2 → 
  b = Real.sqrt 3 - Real.sqrt 2 → 
  (a + b) / 2 = Real.sqrt 3 := by
sorry

end arithmetic_mean_of_a_and_b_l2134_213432


namespace average_of_five_l2134_213409

/-- Given five real numbers x₁, x₂, x₃, x₄, x₅, if the average of x₁ and x₂ is 2
    and the average of x₃, x₄, and x₅ is 4, then the average of all five numbers is 3.2. -/
theorem average_of_five (x₁ x₂ x₃ x₄ x₅ : ℝ) 
    (h₁ : (x₁ + x₂) / 2 = 2)
    (h₂ : (x₃ + x₄ + x₅) / 3 = 4) :
    (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 3.2 := by
  sorry

end average_of_five_l2134_213409


namespace system_solution_l2134_213495

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x - 1 / ((x - y)^2) + y = -10
def equation2 (x y : ℝ) : Prop := x * y = 20

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(-4, -5), (-5, -4), (-2.7972, -7.15), (-7.15, -2.7972), (4.5884, 4.3588), (4.3588, 4.5884)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end system_solution_l2134_213495


namespace smaller_number_problem_l2134_213418

theorem smaller_number_problem (a b : ℤ) : 
  a + b = 18 → a - b = 24 → min a b = -3 := by
  sorry

end smaller_number_problem_l2134_213418


namespace first_triangle_isosceles_l2134_213488

theorem first_triangle_isosceles (α β γ : Real) (θ₁ θ₂ : Real) : 
  α + β + γ = π → 
  α + β = θ₁ → 
  α + γ = θ₂ → 
  θ₁ + θ₂ < π →
  β = γ := by
sorry

end first_triangle_isosceles_l2134_213488


namespace ronald_banana_count_l2134_213496

/-- The number of times Ronald went to the store last month -/
def store_visits : ℕ := 2

/-- The number of bananas Ronald buys each time he goes to the store -/
def bananas_per_visit : ℕ := 10

/-- The total number of bananas Ronald bought last month -/
def total_bananas : ℕ := store_visits * bananas_per_visit

theorem ronald_banana_count : total_bananas = 20 := by
  sorry

end ronald_banana_count_l2134_213496


namespace quadratic_coefficient_l2134_213445

/-- A quadratic function with vertex at (-3, 0) passing through (2, -64) has a = -64/25 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c) → -- quadratic function
  (0 = a * (-3)^2 + b * (-3) + c) → -- vertex at (-3, 0)
  (-64 = a * 2^2 + b * 2 + c) → -- passes through (2, -64)
  a = -64/25 := by
sorry

end quadratic_coefficient_l2134_213445


namespace min_value_theorem_l2134_213404

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 80 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    p' * q' * r' * s' = 16 ∧
    t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 80 :=
by sorry

end min_value_theorem_l2134_213404


namespace john_juice_bottles_l2134_213435

/-- The number of fluid ounces John needs -/
def required_oz : ℝ := 60

/-- The size of each bottle in milliliters -/
def bottle_size_ml : ℝ := 150

/-- The number of fluid ounces in 1 liter -/
def oz_per_liter : ℝ := 34

/-- The number of milliliters in 1 liter -/
def ml_per_liter : ℝ := 1000

/-- The smallest number of bottles John should buy -/
def min_bottles : ℕ := 12

theorem john_juice_bottles : 
  ∃ (n : ℕ), n = min_bottles ∧ 
  (n : ℝ) * bottle_size_ml / ml_per_liter * oz_per_liter ≥ required_oz ∧
  ∀ (m : ℕ), m < n → (m : ℝ) * bottle_size_ml / ml_per_liter * oz_per_liter < required_oz :=
by sorry

end john_juice_bottles_l2134_213435


namespace a_13_value_l2134_213442

/-- An arithmetic sequence with specific terms -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_13_value (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) 
  (h_a5 : a 5 = 6)
  (h_a8 : a 8 = 15) : 
  a 13 = 30 := by
sorry

end a_13_value_l2134_213442


namespace vegetarian_eaters_l2134_213477

theorem vegetarian_eaters (only_veg : ℕ) (only_non_veg : ℕ) (both : ℕ) :
  only_veg = 13 →
  only_non_veg = 8 →
  both = 6 →
  only_veg + both = 19 :=
by
  sorry

end vegetarian_eaters_l2134_213477


namespace aubrey_gum_count_l2134_213484

theorem aubrey_gum_count (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : john_gum + cole_gum + aubrey_gum = 33 * 3) :
  aubrey_gum = 0 := by
sorry

end aubrey_gum_count_l2134_213484


namespace quadratic_distinct_roots_l2134_213417

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots if and only if k > 1/2 and k ≠ 1 -/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 + 2 * x - 2 = 0 ∧ (k - 1) * y^2 + 2 * y - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end quadratic_distinct_roots_l2134_213417


namespace sum_of_squares_constant_l2134_213472

/-- A triangle with side lengths a, b, c and median length m from vertex A to the midpoint of side BC. -/
structure Triangle :=
  (a b c m : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (positive_m : 0 < m)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

/-- The sum of squares of two sides in a triangle given the length of the third side and the median to its midpoint. -/
def sumOfSquares (t : Triangle) : ℝ := t.b^2 + t.c^2

/-- The theorem stating that for a triangle with side length 10 and median length 7,
    the difference between the maximum and minimum possible values of the sum of squares
    of the other two sides is 0. -/
theorem sum_of_squares_constant (t : Triangle) 
  (side_length : t.a = 10)
  (median_length : t.m = 7) :
  ∀ (t' : Triangle), 
    t'.a = t.a → 
    t'.m = t.m → 
    sumOfSquares t = sumOfSquares t' :=
sorry

end sum_of_squares_constant_l2134_213472


namespace sand_heap_radius_l2134_213452

/-- The radius of a conical heap of sand formed from a cylindrical bucket -/
theorem sand_heap_radius (h_cylinder h_cone r_cylinder : ℝ) 
  (h_cylinder_pos : h_cylinder > 0)
  (h_cone_pos : h_cone > 0)
  (r_cylinder_pos : r_cylinder > 0)
  (h_cylinder_val : h_cylinder = 36)
  (h_cone_val : h_cone = 12)
  (r_cylinder_val : r_cylinder = 21) :
  ∃ r_cone : ℝ, r_cone > 0 ∧ r_cone^2 = 3 * r_cylinder^2 :=
by sorry

end sand_heap_radius_l2134_213452


namespace max_nickels_in_jar_l2134_213426

theorem max_nickels_in_jar (total_nickels : ℕ) (jar_score : ℕ) (ground_score : ℕ) (final_score : ℕ) :
  total_nickels = 40 →
  jar_score = 5 →
  ground_score = 2 →
  final_score = 88 →
  ∃ (jar_nickels ground_nickels : ℕ),
    jar_nickels + ground_nickels = total_nickels ∧
    jar_score * jar_nickels - ground_score * ground_nickels = final_score ∧
    jar_nickels ≤ 24 ∧
    (∀ (x : ℕ), x > 24 →
      ¬(∃ (y : ℕ), x + y = total_nickels ∧
        jar_score * x - ground_score * y = final_score)) :=
by sorry

end max_nickels_in_jar_l2134_213426


namespace product_positive_l2134_213408

theorem product_positive (x y z t : ℝ) 
  (h1 : x > y^3) 
  (h2 : y > z^3) 
  (h3 : z > t^3) 
  (h4 : t > x^3) : 
  x * y * z * t > 0 := by
sorry

end product_positive_l2134_213408


namespace concert_theorem_l2134_213405

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  elsa : ℕ

/-- The conditions of the problem -/
def concert_conditions (s : SongCounts) : Prop :=
  s.hanna = 9 ∧ 
  s.mary = 3 ∧ 
  s.alina + s.tina = 16 ∧
  s.hanna > s.alina ∧ s.hanna > s.tina ∧ s.hanna > s.elsa ∧
  s.alina > s.mary ∧ s.tina > s.mary ∧ s.elsa > s.mary

/-- The total number of songs sung -/
def total_songs (s : SongCounts) : ℕ :=
  (s.mary + s.alina + s.tina + s.hanna + s.elsa) / 4

/-- The main theorem: given the conditions, the total number of songs is 8 -/
theorem concert_theorem (s : SongCounts) : 
  concert_conditions s → total_songs s = 8 := by
  sorry

end concert_theorem_l2134_213405


namespace f_derivative_sum_l2134_213439

def f (x : ℝ) := x^4 + x - 1

theorem f_derivative_sum : (deriv f 1) + (deriv f (-1)) = 2 := by sorry

end f_derivative_sum_l2134_213439


namespace both_reunions_count_l2134_213457

/-- The number of people attending both the Oates and Yellow reunions -/
def both_reunions (total_guests oates_guests yellow_guests : ℕ) : ℕ :=
  oates_guests + yellow_guests - total_guests

theorem both_reunions_count :
  both_reunions 100 42 65 = 7 := by
  sorry

end both_reunions_count_l2134_213457


namespace ellipse_equation_l2134_213412

/-- Represents an ellipse with focus on the x-axis -/
structure Ellipse where
  /-- Distance from the right focus to the short axis endpoint -/
  short_axis_dist : ℝ
  /-- Distance from the right focus to the left vertex -/
  left_vertex_dist : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem ellipse_equation (e : Ellipse) (h1 : e.short_axis_dist = 2) (h2 : e.left_vertex_dist = 3) :
  ∀ x y : ℝ, standard_equation e x y ↔ x^2 / 4 + y^2 / 3 = 1 :=
by sorry

end ellipse_equation_l2134_213412


namespace cube_root_cube_equality_l2134_213440

theorem cube_root_cube_equality (x : ℝ) : x = (x^3)^(1/3) := by sorry

end cube_root_cube_equality_l2134_213440


namespace sweetsies_leftover_l2134_213470

theorem sweetsies_leftover (m : ℕ) : 
  (∃ k : ℕ, m = 8 * k + 5) →  -- One bag leaves 5 when divided by 8
  (∃ l : ℕ, 4 * m = 8 * l + 4) -- Four bags leave 4 when divided by 8
  := by sorry

end sweetsies_leftover_l2134_213470


namespace solution_is_two_l2134_213497

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  log10 (x^2 - 3) = log10 (3*x - 5) ∧ x^2 - 3 > 0 ∧ 3*x - 5 > 0

-- Theorem stating that 2 is the solution to the equation
theorem solution_is_two : equation 2 := by sorry

end solution_is_two_l2134_213497


namespace max_perimeter_special_triangle_l2134_213455

/-- A triangle with integer side lengths, where one side is twice another, and the third side is 10 -/
structure SpecialTriangle where
  x : ℕ
  side1 : ℕ := x
  side2 : ℕ := 2 * x
  side3 : ℕ := 10

/-- The perimeter of a SpecialTriangle -/
def perimeter (t : SpecialTriangle) : ℕ := t.side1 + t.side2 + t.side3

/-- The triangle inequality for SpecialTriangle -/
def is_valid (t : SpecialTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The theorem stating the maximum perimeter of a valid SpecialTriangle -/
theorem max_perimeter_special_triangle :
  ∃ (t : SpecialTriangle), is_valid t ∧
  ∀ (t' : SpecialTriangle), is_valid t' → perimeter t' ≤ perimeter t ∧
  perimeter t = 37 := by sorry

end max_perimeter_special_triangle_l2134_213455


namespace five_balls_three_boxes_l2134_213482

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 41 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 41 := by
  sorry

end five_balls_three_boxes_l2134_213482


namespace mod_eight_equivalence_l2134_213476

theorem mod_eight_equivalence : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -3737 [ZMOD 8] ∧ n = 7 := by
  sorry

end mod_eight_equivalence_l2134_213476


namespace smallest_sum_square_config_l2134_213447

/-- A configuration of four positive integers on a square's vertices. -/
structure SquareConfig where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+

/-- Predicate to check if one number is a multiple of another. -/
def isMultiple (x y : ℕ+) : Prop := ∃ k : ℕ+, x = k * y

/-- Predicate to check if the configuration satisfies the edge multiple condition. -/
def satisfiesEdgeCondition (config : SquareConfig) : Prop :=
  (isMultiple config.a config.b ∨ isMultiple config.b config.a) ∧
  (isMultiple config.b config.c ∨ isMultiple config.c config.b) ∧
  (isMultiple config.c config.d ∨ isMultiple config.d config.c) ∧
  (isMultiple config.d config.a ∨ isMultiple config.a config.d)

/-- Predicate to check if the configuration satisfies the diagonal non-multiple condition. -/
def satisfiesDiagonalCondition (config : SquareConfig) : Prop :=
  ¬(isMultiple config.a config.c ∨ isMultiple config.c config.a) ∧
  ¬(isMultiple config.b config.d ∨ isMultiple config.d config.b)

/-- Theorem stating the smallest possible sum of the four integers. -/
theorem smallest_sum_square_config :
  ∀ config : SquareConfig,
    satisfiesEdgeCondition config →
    satisfiesDiagonalCondition config →
    (config.a + config.b + config.c + config.d : ℕ) ≥ 35 :=
by sorry

end smallest_sum_square_config_l2134_213447


namespace solution_set_f_leq_5_range_of_m_for_f_geq_x_minus_m_l2134_213423

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for part I
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -2 ≤ x ∧ x ≤ 4/3} :=
sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_x_minus_m :
  {m : ℝ | ∀ x, f x ≥ x - m} = {m : ℝ | m ≥ -3} :=
sorry

end solution_set_f_leq_5_range_of_m_for_f_geq_x_minus_m_l2134_213423


namespace initial_machines_count_l2134_213403

/-- The number of shirts produced by a group of machines -/
def shirts_produced (num_machines : ℕ) (time : ℕ) : ℕ := sorry

/-- The production rate of a single machine in shirts per minute -/
def machine_rate : ℚ := sorry

/-- The total production rate of all machines in shirts per minute -/
def total_rate : ℕ := 32

theorem initial_machines_count :
  ∃ (n : ℕ), 
    shirts_produced 8 10 = 160 ∧
    (n : ℚ) * machine_rate = total_rate ∧
    n = 16 :=
sorry

end initial_machines_count_l2134_213403


namespace product_of_numbers_l2134_213499

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 218) : x * y = 13 := by
  sorry

end product_of_numbers_l2134_213499


namespace second_box_weight_l2134_213443

/-- The weight of the second box in a set of three boxes -/
def weight_of_second_box (weight_first weight_last total_weight : ℕ) : ℕ :=
  total_weight - weight_first - weight_last

/-- Theorem: The weight of the second box is 11 pounds -/
theorem second_box_weight :
  weight_of_second_box 2 5 18 = 11 := by
  sorry

end second_box_weight_l2134_213443


namespace work_completion_time_l2134_213487

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 20

/-- The fraction of work left after A and B work together for 3 days -/
def work_left : ℝ := 0.65

/-- The number of days A and B work together -/
def days_together : ℝ := 3

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 15

theorem work_completion_time :
  ∃ (x : ℝ), x > 0 ∧ 
  days_together * (1 / x + 1 / b_days) = 1 - work_left ∧
  x = a_days := by sorry

end work_completion_time_l2134_213487


namespace cone_volume_from_cylinder_l2134_213481

/-- Given a cylinder with volume 72π cm³ and a cone with the same height
    as the cylinder and half its radius, prove that the volume of the cone is 6π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) →
  (1/3 * π * (r/2)^2 * h = 6 * π) :=
by sorry

end cone_volume_from_cylinder_l2134_213481


namespace functional_equation_solution_l2134_213461

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f x * f y = f x + f y - f (x * y))
  (h2 : ∀ x y : ℚ, 1 + f (x + y) = f (x * y) + f x * f y) :
  (∀ x : ℚ, f x = 1) ∨ (∀ x : ℚ, f x = 1 - x) := by sorry

end functional_equation_solution_l2134_213461


namespace system_solution_l2134_213492

theorem system_solution (x y : ℝ) : 
  x > 0 → y > 0 → 
  Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1 + Real.log 9 / Real.log 4 →
  x + y = 20 →
  ((x = 2 ∧ y = 18) ∨ (x = 18 ∧ y = 2)) :=
by sorry

end system_solution_l2134_213492


namespace initial_average_mark_l2134_213468

/-- Proves that the initial average mark of a class is 80, given specific conditions --/
theorem initial_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 10 →
  excluded_students = 5 →
  excluded_avg = 70 →
  remaining_avg = 90 →
  (total_students * (total_students * remaining_avg - excluded_students * excluded_avg)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end initial_average_mark_l2134_213468


namespace weight_of_bart_and_cindy_l2134_213407

/-- Given the weights of pairs of people, prove the weight of a specific pair -/
theorem weight_of_bart_and_cindy 
  (abby bart cindy damon : ℝ) 
  (h1 : abby + bart = 280) 
  (h2 : cindy + damon = 290) 
  (h3 : abby + damon = 300) : 
  bart + cindy = 270 := by
  sorry

end weight_of_bart_and_cindy_l2134_213407


namespace soda_cost_l2134_213421

theorem soda_cost (bill : ℕ) (change : ℕ) (num_sodas : ℕ) (h1 : bill = 20) (h2 : change = 14) (h3 : num_sodas = 3) :
  (bill - change) / num_sodas = 2 := by
sorry

end soda_cost_l2134_213421


namespace camel_cost_l2134_213498

/-- The cost of animals in rupees -/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 140000

/-- The theorem stating that under the given conditions, a camel costs 5600 rupees -/
theorem camel_cost (costs : AnimalCosts) : 
  problem_conditions costs → costs.camel = 5600 := by
  sorry

end camel_cost_l2134_213498


namespace gcd_117_182_l2134_213429

theorem gcd_117_182 : Nat.gcd 117 182 = 13 := by sorry

end gcd_117_182_l2134_213429


namespace strawberry_remainder_l2134_213493

/-- Given 3 kg and 300 g of strawberries, prove that after giving away 1 kg and 900 g, 
    the remaining amount is 1400 g. -/
theorem strawberry_remainder : 
  let total_kg : ℕ := 3
  let total_g : ℕ := 300
  let given_kg : ℕ := 1
  let given_g : ℕ := 900
  let g_per_kg : ℕ := 1000
  (total_kg * g_per_kg + total_g) - (given_kg * g_per_kg + given_g) = 1400 := by
  sorry

end strawberry_remainder_l2134_213493


namespace parabola_focus_l2134_213485

/-- The parabola with equation y² = -8x has its focus at (-2, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -8*x → (x + 2)^2 + y^2 = 4 := by sorry

end parabola_focus_l2134_213485


namespace josh_and_anna_marriage_problem_l2134_213490

/-- Josh and Anna's marriage problem -/
theorem josh_and_anna_marriage_problem 
  (josh_marriage_age : ℕ) 
  (marriage_duration : ℕ) 
  (combined_age_factor : ℕ) 
  (h1 : josh_marriage_age = 22)
  (h2 : marriage_duration = 30)
  (h3 : combined_age_factor = 5)
  (h4 : josh_marriage_age + marriage_duration + (josh_marriage_age + marriage_duration + anna_marriage_age) = combined_age_factor * josh_marriage_age) :
  anna_marriage_age = 28 :=
by sorry

end josh_and_anna_marriage_problem_l2134_213490


namespace min_a_value_for_common_points_l2134_213419

/-- Given two curves C₁ and C₂, where C₁ is y = ax² (a > 0) and C₂ is y = eˣ, 
    if they have common points in (0, +∞), then the minimum value of a is e²/4 -/
theorem min_a_value_for_common_points (a : ℝ) (h1 : a > 0) :
  (∃ x : ℝ, x > 0 ∧ a * x^2 = Real.exp x) → a ≥ Real.exp 2 / 4 := by
  sorry

end min_a_value_for_common_points_l2134_213419


namespace diagonal_contains_all_numbers_l2134_213453

theorem diagonal_contains_all_numbers (n : ℕ) (h_odd : Odd n) 
  (grid : Fin n → Fin n → Fin n)
  (h_row : ∀ i j k, i ≠ k → grid i j ≠ grid k j)
  (h_col : ∀ i j k, j ≠ k → grid i j ≠ grid i k)
  (h_sym : ∀ i j, grid i j = grid j i) :
  ∀ k : Fin n, ∃ i : Fin n, grid i i = k := by
sorry

end diagonal_contains_all_numbers_l2134_213453


namespace souvenir_sales_theorem_l2134_213454

/-- Represents the souvenir sales scenario -/
structure SouvenirSales where
  purchase_price : ℝ
  base_selling_price : ℝ
  base_daily_sales : ℝ
  price_sales_ratio : ℝ

/-- Calculates the daily sales quantity for a given selling price -/
def daily_sales (s : SouvenirSales) (selling_price : ℝ) : ℝ :=
  s.base_daily_sales - s.price_sales_ratio * (selling_price - s.base_selling_price)

/-- Calculates the daily profit for a given selling price -/
def daily_profit (s : SouvenirSales) (selling_price : ℝ) : ℝ :=
  (selling_price - s.purchase_price) * (daily_sales s selling_price)

/-- The main theorem about the souvenir sales scenario -/
theorem souvenir_sales_theorem (s : SouvenirSales) 
  (h1 : s.purchase_price = 40)
  (h2 : s.base_selling_price = 50)
  (h3 : s.base_daily_sales = 200)
  (h4 : s.price_sales_ratio = 10) :
  (daily_sales s 52 = 180) ∧ 
  (∃ x : ℝ, ∀ y : ℝ, daily_profit s x ≥ daily_profit s y) ∧
  (daily_profit s 55 = 2250) := by
  sorry

#check souvenir_sales_theorem

end souvenir_sales_theorem_l2134_213454


namespace largest_valid_sequence_length_l2134_213494

def isPrimePower (n : ℕ) : Prop :=
  ∃ p k, Prime p ∧ n = p ^ k

def validSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i, i ≤ n → isPrimePower (a i)) ∧
  (∀ i, 3 ≤ i ∧ i ≤ n → a i = a (i - 1) + a (i - 2))

theorem largest_valid_sequence_length :
  (∃ a : ℕ → ℕ, validSequence a 7) ∧
  (∀ n : ℕ, n > 7 → ¬∃ a : ℕ → ℕ, validSequence a n) :=
sorry

end largest_valid_sequence_length_l2134_213494


namespace linear_implies_constant_derivative_constant_derivative_not_sufficient_for_linear_l2134_213431

-- Define a linear function
def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define a constant derivative
def has_constant_derivative (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x, deriv f x = c

theorem linear_implies_constant_derivative :
  ∀ f : ℝ → ℝ, is_linear f → has_constant_derivative f :=
sorry

theorem constant_derivative_not_sufficient_for_linear :
  ∃ f : ℝ → ℝ, has_constant_derivative f ∧ ¬is_linear f :=
sorry

end linear_implies_constant_derivative_constant_derivative_not_sufficient_for_linear_l2134_213431


namespace divisibility_problem_l2134_213456

theorem divisibility_problem (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 45)
  (h2 : Nat.gcd q r = 75)
  (h3 : Nat.gcd r s = 90)
  (h4 : 150 < Nat.gcd s p ∧ Nat.gcd s p < 200) :
  10 ∣ p.val := by
  sorry

end divisibility_problem_l2134_213456


namespace simplify_fraction_l2134_213448

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 := by
  sorry

end simplify_fraction_l2134_213448


namespace four_valid_orders_l2134_213459

/-- Represents a runner in the relay team -/
inductive Runner : Type
| Jordan : Runner
| Friend1 : Runner  -- The fastest friend
| Friend2 : Runner
| Friend3 : Runner

/-- Represents a lap in the relay race -/
inductive Lap : Type
| First : Lap
| Second : Lap
| Third : Lap
| Fourth : Lap

/-- A valid running order for the relay team -/
def RunningOrder : Type := Lap → Runner

/-- Checks if a running order is valid according to the given conditions -/
def isValidOrder (order : RunningOrder) : Prop :=
  (order Lap.First = Runner.Friend1) ∧  -- Fastest friend starts
  ((order Lap.Third = Runner.Jordan) ∨ (order Lap.Fourth = Runner.Jordan)) ∧  -- Jordan runs 3rd or 4th
  (∀ l : Lap, ∃! r : Runner, order l = r)  -- Each lap has exactly one runner

/-- The main theorem: there are exactly 4 valid running orders -/
theorem four_valid_orders :
  ∃ (orders : Finset RunningOrder),
    (∀ o ∈ orders, isValidOrder o) ∧
    (∀ o : RunningOrder, isValidOrder o → o ∈ orders) ∧
    (Finset.card orders = 4) :=
sorry

end four_valid_orders_l2134_213459


namespace min_value_C_over_D_l2134_213441

theorem min_value_C_over_D (C D y : ℝ) (hC : C > 0) (hD : D > 0) (hy : y > 0)
  (hCy : y^3 + 1/y^3 = C) (hDy : y - 1/y = D) :
  C / D ≥ 6 ∧ ∃ y > 0, y^3 + 1/y^3 = C ∧ y - 1/y = D ∧ C / D = 6 :=
by sorry

end min_value_C_over_D_l2134_213441


namespace arrangement_count_l2134_213427

/-- The number of ways to arrange 4 boys and 3 girls in a row -/
def total_arrangements : ℕ := Nat.factorial 7

/-- The number of ways to arrange 4 boys and 3 girls where all 3 girls are adjacent -/
def three_girls_adjacent : ℕ := Nat.factorial 5 * Nat.factorial 3

/-- The number of ways to arrange 4 boys and 3 girls where exactly 2 girls are adjacent -/
def two_girls_adjacent : ℕ := Nat.factorial 6 * Nat.factorial 2 * 3

/-- The number of valid arrangements -/
def valid_arrangements : ℕ := two_girls_adjacent - three_girls_adjacent

theorem arrangement_count : valid_arrangements = 3600 := by sorry

end arrangement_count_l2134_213427


namespace seven_digit_numbers_existence_l2134_213446

theorem seven_digit_numbers_existence :
  ∃ (x y : ℕ),
    (10^6 ≤ x ∧ x < 10^7) ∧
    (10^6 ≤ y ∧ y < 10^7) ∧
    (3 * x * y = 10^7 * x + y) ∧
    (x = 166667 ∧ y = 333334) := by
  sorry

end seven_digit_numbers_existence_l2134_213446


namespace algebraic_expression_value_l2134_213450

theorem algebraic_expression_value (a x : ℝ) : 
  (3 * a - x = x + 2) → (x = 2) → (a^2 - 2*a + 1 = 1) := by
  sorry

end algebraic_expression_value_l2134_213450


namespace prime_sum_product_l2134_213460

theorem prime_sum_product : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p + q = 95 ∧ p * q = 178 := by
  sorry

end prime_sum_product_l2134_213460


namespace fruit_basket_total_cost_l2134_213433

/-- Represents the cost of a fruit basket -/
def fruit_basket_cost (banana_price : ℚ) (apple_price : ℚ) (strawberry_price : ℚ) 
  (avocado_price : ℚ) (grape_price : ℚ) : ℚ :=
  4 * banana_price + 3 * apple_price + 24 * strawberry_price / 12 + 
  2 * avocado_price + 2 * grape_price

/-- Theorem stating the total cost of the fruit basket -/
theorem fruit_basket_total_cost : 
  fruit_basket_cost 1 2 (4/12) 3 2 = 28 := by
  sorry

end fruit_basket_total_cost_l2134_213433


namespace power_six_mod_fifty_l2134_213465

theorem power_six_mod_fifty : 6^2040 ≡ 26 [ZMOD 50] := by sorry

end power_six_mod_fifty_l2134_213465


namespace expression_evaluation_l2134_213425

theorem expression_evaluation (a x : ℝ) (h1 : a = x^2) (h2 : a = Real.sqrt 2) :
  4 * a^3 / (x^4 + a^4) + 1 / (a + x) + 2 * a / (x^2 + a^2) + 1 / (a - x) = 16 * Real.sqrt 2 / 3 := by
  sorry

end expression_evaluation_l2134_213425


namespace min_sum_squares_l2134_213471

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 120) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 6100 / 9 ∧ 
  ∃ (y₁' y₂' y₃' : ℝ), y₁'^2 + y₂'^2 + y₃'^2 = 6100 / 9 ∧ 
    y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 2 * y₁' + 3 * y₂' + 4 * y₃' = 120 :=
by sorry

end min_sum_squares_l2134_213471


namespace equation_solution_l2134_213401

theorem equation_solution : 
  ∃! x : ℝ, 12 * (x - 3) - 1 = 2 * x + 3 :=
by
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end equation_solution_l2134_213401


namespace rational_square_difference_l2134_213420

theorem rational_square_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by sorry

end rational_square_difference_l2134_213420


namespace area_of_triangle_AGE_l2134_213473

/-- Square ABCD with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 5) ∧ B = (0, 0) ∧ C = (5, 0) ∧ D = (5, 5))

/-- Point E on side BC -/
def E : ℝ × ℝ := (2, 0)

/-- Point G on diagonal BD -/
def G : ℝ × ℝ := sorry

/-- Circumscribed circle of triangle ABE -/
def circle_ABE (sq : Square) : Set (ℝ × ℝ) := sorry

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem area_of_triangle_AGE (sq : Square) :
  G ∈ circle_ABE sq →
  G.1 + G.2 = 5 →
  triangle_area sq.A G E = 54.5 := by sorry

end area_of_triangle_AGE_l2134_213473


namespace coordinates_wrt_origin_l2134_213463

/-- Given a point A with coordinates (-1, 2) in the plane rectangular coordinate system xOy,
    prove that its coordinates with respect to the origin are (-1, 2). -/
theorem coordinates_wrt_origin (A : ℝ × ℝ) (h : A = (-1, 2)) :
  A = (-1, 2) := by sorry

end coordinates_wrt_origin_l2134_213463


namespace tangent_line_equation_l2134_213475

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + 2

theorem tangent_line_equation :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := (Real.cos x₀ + Real.exp x₀)
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = 2 * x + 3 :=
by sorry

end tangent_line_equation_l2134_213475


namespace greatest_gcd_with_linear_combination_l2134_213415

theorem greatest_gcd_with_linear_combination (m n : ℕ) : 
  Nat.gcd m n = 1 → 
  (∃ (a b : ℕ), Nat.gcd (m + 2000 * n) (n + 2000 * m) = a ∧ 
                a ≤ b ∧ 
                ∀ (c : ℕ), Nat.gcd (m + 2000 * n) (n + 2000 * m) ≤ c → c ≤ b) ∧
  3999999 = Nat.gcd (m + 2000 * n) (n + 2000 * m) := by
  sorry

end greatest_gcd_with_linear_combination_l2134_213415


namespace product_equality_l2134_213464

theorem product_equality : 375680169467 * 4565579427629 = 1715110767607750737263 := by
  sorry

end product_equality_l2134_213464


namespace train_length_l2134_213479

/-- The length of a train given specific crossing times and platform length -/
theorem train_length (platform_cross_time signal_cross_time : ℝ) (platform_length : ℝ) : 
  platform_cross_time = 54 →
  signal_cross_time = 18 →
  platform_length = 600.0000000000001 →
  ∃ (train_length : ℝ), train_length = 300.00000000000005 :=
by
  sorry

end train_length_l2134_213479


namespace smallest_b_for_scaled_property_l2134_213462

/-- A function with period 30 -/
def IsPeriodic30 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

/-- The property we want to prove for the scaled function -/
def HasScaledProperty (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 10) = g (x / 10)

/-- The main theorem -/
theorem smallest_b_for_scaled_property (g : ℝ → ℝ) (h : IsPeriodic30 g) :
    (∃ b > 0, HasScaledProperty g b) →
    (∃ b > 0, HasScaledProperty g b ∧ ∀ b' > 0, HasScaledProperty g b' → b ≤ b') →
    (∃ b > 0, HasScaledProperty g b ∧ b = 300) :=
  sorry

end smallest_b_for_scaled_property_l2134_213462


namespace sprinkler_water_usage_5_days_l2134_213406

/-- A sprinkler system for a desert garden -/
structure SprinklerSystem where
  morning_usage : ℕ  -- Water usage in the morning in liters
  evening_usage : ℕ  -- Water usage in the evening in liters

/-- Calculates the total water usage for a given number of days -/
def total_water_usage (s : SprinklerSystem) (days : ℕ) : ℕ :=
  (s.morning_usage + s.evening_usage) * days

/-- Theorem: The sprinkler system uses 50 liters of water in 5 days -/
theorem sprinkler_water_usage_5_days :
  ∃ (s : SprinklerSystem), s.morning_usage = 4 ∧ s.evening_usage = 6 ∧ total_water_usage s 5 = 50 := by
  sorry

end sprinkler_water_usage_5_days_l2134_213406


namespace vector_parallel_implies_x_eq_half_l2134_213491

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define the parallel condition
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v 0 * w 1 = k * v 1 * w 0

-- State the theorem
theorem vector_parallel_implies_x_eq_half :
  ∀ x : ℝ, are_parallel (a + 2 • b x) (2 • a - 2 • b x) → x = 1/2 := by
  sorry

end vector_parallel_implies_x_eq_half_l2134_213491


namespace fraction_simplification_l2134_213466

theorem fraction_simplification :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 := by
  sorry

end fraction_simplification_l2134_213466


namespace factor_condition_l2134_213424

theorem factor_condition (n : ℕ) (hn : n ≥ 2) 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (k : ℕ), n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧
    a = (-2 * Real.cos ((2 * k + 1 : ℝ) * π / (2 * n : ℝ))) ^ (2 * n / (2 * n - 1 : ℝ)) ∧
    b = (2 * Real.cos ((2 * k + 1 : ℝ) * π / (2 * n : ℝ))) ^ (2 / (2 * n - 1 : ℝ))) ↔
  (∀ x : ℂ, (x ^ 2 + a * x + b = 0) → (a * x ^ (2 * n) + (a * x + b) ^ (2 * n) = 0)) :=
by sorry

end factor_condition_l2134_213424


namespace four_common_tangents_min_area_PAOB_l2134_213410

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the moving circle C
def circle_C (x y k : ℝ) : Prop := (x - k)^2 + (y - Real.sqrt 3 * k)^2 = 4

-- Define the line
def line (x y : ℝ) : Prop := x + y = 4

-- Theorem 1: Four common tangents condition
theorem four_common_tangents (k : ℝ) :
  (∀ x y, circle_O x y → ∀ x' y', circle_C x' y' k → 
    ∃! t1 t2 t3 t4 : ℝ × ℝ, 
      (circle_O t1.1 t1.2 ∧ circle_C t1.1 t1.2 k) ∧
      (circle_O t2.1 t2.2 ∧ circle_C t2.1 t2.2 k) ∧
      (circle_O t3.1 t3.2 ∧ circle_C t3.1 t3.2 k) ∧
      (circle_O t4.1 t4.2 ∧ circle_C t4.1 t4.2 k)) ↔
  abs k > 2 := by sorry

-- Theorem 2: Minimum area of quadrilateral PAOB
theorem min_area_PAOB :
  ∃ min_area : ℝ, 
    min_area = 4 ∧
    ∀ P A B O : ℝ × ℝ,
      line P.1 P.2 →
      circle_O A.1 A.2 →
      circle_O B.1 B.2 →
      O = (0, 0) →
      (∀ x y, (x - P.1) * (A.1 - P.1) + (y - P.2) * (A.2 - P.2) = 0 → ¬ circle_O x y) →
      (∀ x y, (x - P.1) * (B.1 - P.1) + (y - P.2) * (B.2 - P.2) = 0 → ¬ circle_O x y) →
      let area := abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2))
      area ≥ min_area := by sorry

end four_common_tangents_min_area_PAOB_l2134_213410


namespace unique_solution_for_complex_equation_l2134_213458

theorem unique_solution_for_complex_equation (x : ℝ) :
  x - 8 ≥ 0 →
  (7 / (Real.sqrt (x - 8) - 10) + 2 / (Real.sqrt (x - 8) - 4) +
   9 / (Real.sqrt (x - 8) + 4) + 14 / (Real.sqrt (x - 8) + 10) = 0) ↔
  x = 55 := by
  sorry

end unique_solution_for_complex_equation_l2134_213458


namespace flea_initial_position_l2134_213449

def electronic_flea (K : ℕ → ℤ) : Prop :=
  K 100 = 20 ∧ ∀ n : ℕ, K (n + 1) = K n + (-1)^n * (n + 1)

theorem flea_initial_position (K : ℕ → ℤ) (h : electronic_flea K) : K 0 = -30 := by
  sorry

end flea_initial_position_l2134_213449


namespace orange_purchase_price_l2134_213474

/-- The price of oranges per 3 pounds -/
def price_per_3_pounds : ℝ := 3

/-- The weight of oranges purchased in pounds -/
def weight_purchased : ℝ := 18

/-- The discount rate applied for purchases over 15 pounds -/
def discount_rate : ℝ := 0.05

/-- The minimum weight for discount eligibility in pounds -/
def discount_threshold : ℝ := 15

/-- The final price paid by the customer for the oranges -/
def final_price : ℝ := 17.10

theorem orange_purchase_price :
  weight_purchased > discount_threshold →
  final_price = (weight_purchased / 3 * price_per_3_pounds) * (1 - discount_rate) :=
by sorry

end orange_purchase_price_l2134_213474


namespace bus_trip_distance_l2134_213436

/-- The distance of a bus trip in miles. -/
def trip_distance : ℝ := 280

/-- The actual average speed of the bus in miles per hour. -/
def actual_speed : ℝ := 35

/-- The increased speed of the bus in miles per hour. -/
def increased_speed : ℝ := 40

/-- Theorem stating that the trip distance is 280 miles given the conditions. -/
theorem bus_trip_distance :
  (trip_distance / actual_speed - trip_distance / increased_speed = 1) →
  trip_distance = 280 := by
  sorry


end bus_trip_distance_l2134_213436


namespace basketball_spectators_l2134_213416

theorem basketball_spectators (total : Nat) (men : Nat) (women : Nat) (children : Nat) :
  total = 10000 →
  men = 7000 →
  total = men + women + children →
  children = 5 * women →
  children = 2500 := by
sorry

end basketball_spectators_l2134_213416


namespace decimal_point_shift_l2134_213469

theorem decimal_point_shift (x : ℝ) : 10 * x = x + 37.89 → 100 * x = 421 := by
  sorry

end decimal_point_shift_l2134_213469


namespace stream_speed_calculation_l2134_213413

-- Define the swimming speed in still water
def still_water_speed : ℝ := 6

-- Define the function for downstream speed
def downstream_speed (stream_speed : ℝ) : ℝ := still_water_speed + stream_speed

-- Define the function for upstream speed
def upstream_speed (stream_speed : ℝ) : ℝ := still_water_speed - stream_speed

-- Theorem statement
theorem stream_speed_calculation :
  ∃ (stream_speed : ℝ),
    stream_speed > 0 ∧
    downstream_speed stream_speed / upstream_speed stream_speed = 2 ∧
    stream_speed = 2 := by
  sorry

end stream_speed_calculation_l2134_213413


namespace power_multiplication_addition_l2134_213437

theorem power_multiplication_addition : 2^4 * 3^2 * 5^2 + 7^3 = 3943 := by
  sorry

end power_multiplication_addition_l2134_213437


namespace sphere_minus_cylinder_volume_l2134_213428

/-- The volume of space inside a sphere but outside an inscribed right cylinder -/
theorem sphere_minus_cylinder_volume (r_sphere : ℝ) (r_cylinder : ℝ) : 
  r_sphere = 6 → r_cylinder = 4 → 
  (4/3 * Real.pi * r_sphere^3) - (Real.pi * r_cylinder^2 * Real.sqrt (r_sphere^2 - r_cylinder^2)) = 
  (288 - 64 * Real.sqrt 5) * Real.pi :=
by sorry

end sphere_minus_cylinder_volume_l2134_213428


namespace supplement_of_supplement_58_l2134_213486

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_supplement_58 :
  supplement (supplement 58) = 58 := by
  sorry

end supplement_of_supplement_58_l2134_213486


namespace largest_solution_quadratic_l2134_213400

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := λ x ↦ 9*x^2 - 45*x + 50
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x ∧ x = 10/3 :=
by sorry

end largest_solution_quadratic_l2134_213400


namespace trivia_team_selection_l2134_213467

theorem trivia_team_selection (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) :
  total_students = 17 →
  num_groups = 3 →
  students_per_group = 4 →
  total_students - (num_groups * students_per_group) = 5 := by
  sorry

end trivia_team_selection_l2134_213467


namespace optimal_purchase_solution_max_basketballs_part2_l2134_213434

def basketball_price : ℕ := 100
def soccer_ball_price : ℕ := 80
def total_budget : ℕ := 5600
def total_items : ℕ := 60

theorem optimal_purchase_solution :
  ∃! (basketballs soccer_balls : ℕ),
    basketballs + soccer_balls = total_items ∧
    basketball_price * basketballs + soccer_ball_price * soccer_balls = total_budget ∧
    basketballs = 40 ∧
    soccer_balls = 20 :=
by sorry

theorem max_basketballs_part2 (new_budget : ℕ) (new_total_items : ℕ)
  (h1 : new_budget = 6890) (h2 : new_total_items = 80) :
  ∃ (max_basketballs : ℕ),
    max_basketballs ≤ new_total_items ∧
    basketball_price * max_basketballs + soccer_ball_price * (new_total_items - max_basketballs) ≤ new_budget ∧
    ∀ (basketballs : ℕ),
      basketballs ≤ new_total_items →
      basketball_price * basketballs + soccer_ball_price * (new_total_items - basketballs) ≤ new_budget →
      basketballs ≤ max_basketballs ∧
    max_basketballs = 24 :=
by sorry

end optimal_purchase_solution_max_basketballs_part2_l2134_213434


namespace parallelogram_fourth_vertex_x_sum_l2134_213402

/-- The sum of all possible x coordinates of the 4th vertex of a parallelogram 
    with three vertices at (1,2), (3,8), and (4,1) is equal to 8. -/
theorem parallelogram_fourth_vertex_x_sum : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (3, 8)
  let C : ℝ × ℝ := (4, 1)
  let D₁ : ℝ × ℝ := B + C - A
  let D₂ : ℝ × ℝ := A + C - B
  let D₃ : ℝ × ℝ := A + B - C
  (D₁.1 + D₂.1 + D₃.1 : ℝ) = 8 := by
  sorry

end parallelogram_fourth_vertex_x_sum_l2134_213402


namespace square_root_sum_l2134_213422

theorem square_root_sum (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
sorry

end square_root_sum_l2134_213422


namespace inequality_proof_l2134_213489

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 1) :
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end inequality_proof_l2134_213489


namespace compound_interest_problem_l2134_213451

/-- Calculates compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Theorem: Compound interest calculation for given problem -/
theorem compound_interest_problem :
  let principal := 1200
  let rate := 0.20
  let time := 4
  abs (compound_interest principal rate time - 1288.32) < 0.01 := by
sorry

end compound_interest_problem_l2134_213451


namespace sum_of_reciprocal_roots_l2134_213444

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  (6 * a^2 + 5 * a + 7 = 0) →
  (6 * b^2 + 5 * b + 7 = 0) →
  a ≠ b →
  a ≠ 0 →
  b ≠ 0 →
  (1 / a) + (1 / b) = -5 / 7 := by
sorry

end sum_of_reciprocal_roots_l2134_213444


namespace trapezoid_diagonal_inequality_l2134_213430

/-- A trapezoid with non-parallel sides b and d, and diagonals e and f -/
structure Trapezoid where
  b : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  b_pos : 0 < b
  d_pos : 0 < d
  e_pos : 0 < e
  f_pos : 0 < f

/-- The inequality |e - f| > |b - d| holds for a trapezoid -/
theorem trapezoid_diagonal_inequality (t : Trapezoid) : |t.e - t.f| > |t.b - t.d| := by
  sorry

end trapezoid_diagonal_inequality_l2134_213430


namespace smallest_sum_is_four_ninths_l2134_213483

theorem smallest_sum_is_four_ninths :
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/9]
  (∀ s ∈ sums, 1/3 + 1/9 ≤ s) ∧ (1/3 + 1/9 = 4/9) := by
  sorry

end smallest_sum_is_four_ninths_l2134_213483


namespace amanda_pay_calculation_l2134_213411

/-- Calculates the amount Amanda receives if she doesn't finish her sales report --/
theorem amanda_pay_calculation (hourly_rate : ℝ) (hours_worked : ℝ) (withholding_percentage : ℝ) : 
  hourly_rate = 50 →
  hours_worked = 10 →
  withholding_percentage = 0.2 →
  hourly_rate * hours_worked * (1 - withholding_percentage) = 400 := by
sorry

end amanda_pay_calculation_l2134_213411


namespace total_books_l2134_213478

/-- The number of books each person has -/
structure Books where
  beatrix : ℕ
  alannah : ℕ
  queen : ℕ

/-- The conditions of the problem -/
def book_conditions (b : Books) : Prop :=
  b.beatrix = 30 ∧
  b.alannah = b.beatrix + 20 ∧
  b.queen = b.alannah + (b.alannah / 5)

/-- The theorem to prove -/
theorem total_books (b : Books) :
  book_conditions b → b.beatrix + b.alannah + b.queen = 140 := by
  sorry

end total_books_l2134_213478


namespace unique_representation_of_nonnegative_integers_l2134_213480

theorem unique_representation_of_nonnegative_integers (n : ℕ) :
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
by sorry

end unique_representation_of_nonnegative_integers_l2134_213480


namespace bob_painting_fraction_l2134_213438

-- Define the time it takes Bob to paint a whole house
def full_painting_time : ℕ := 60

-- Define the time we want to calculate the fraction for
def partial_painting_time : ℕ := 15

-- Theorem statement
theorem bob_painting_fraction :
  (partial_painting_time : ℚ) / full_painting_time = 1 / 4 := by
  sorry

end bob_painting_fraction_l2134_213438
