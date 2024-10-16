import Mathlib

namespace NUMINAMATH_CALUDE_birth_year_problem_l3779_377938

theorem birth_year_problem (x : ℕ) : 
  (1800 ≤ x^2 - x) ∧ (x^2 - x < 1850) ∧ (x^2 = x + 1806) → x^2 - x = 1806 :=
by sorry

end NUMINAMATH_CALUDE_birth_year_problem_l3779_377938


namespace NUMINAMATH_CALUDE_two_times_larger_by_one_l3779_377973

theorem two_times_larger_by_one (a : ℝ) : 
  (2 * a + 1) = (2 * a) + 1 := by sorry

end NUMINAMATH_CALUDE_two_times_larger_by_one_l3779_377973


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_from_unit_square_l3779_377944

/-- Represents a triangle formed from a unit square --/
structure TriangleFromUnitSquare where
  /-- The base of the isosceles triangle --/
  base : ℝ
  /-- One leg of the isosceles triangle --/
  leg : ℝ
  /-- The triangle is isosceles --/
  isIsosceles : base = 2 * leg
  /-- The base is formed by two sides of the unit square --/
  baseFromSquare : base = Real.sqrt 2
  /-- Each leg is formed by one side of the unit square --/
  legFromSquare : leg = Real.sqrt 2 / 2

/-- The perimeter of the triangle formed from a unit square is 2√2 --/
theorem perimeter_of_triangle_from_unit_square (t : TriangleFromUnitSquare) :
  t.base + 2 * t.leg = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_from_unit_square_l3779_377944


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_is_zero_l3779_377911

/-- A piecewise linear function composed of six line segments -/
def PiecewiseLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
    (∀ x, (x ≤ x₁ ∨ x₁ < x ∧ x ≤ x₂ ∨ x₂ < x ∧ x ≤ x₃ ∨
           x₃ < x ∧ x ≤ x₄ ∨ x₄ < x ∧ x ≤ x₅ ∨ x₅ < x) →
      ∃ (a b : ℝ), ∀ y ∈ Set.Icc x₁ x, f y = a * y + b)

/-- The graph of g intersects with y = x - 1 at exactly three points -/
def ThreeIntersections (g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x, g x = x - 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem sum_of_x_coordinates_is_zero
  (g : ℝ → ℝ)
  (h₁ : PiecewiseLinearFunction g)
  (h₂ : ThreeIntersections g) :
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x, g x = x - 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
                    x₁ + x₂ + x₃ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_is_zero_l3779_377911


namespace NUMINAMATH_CALUDE_pentagon_area_l3779_377903

/-- Represents a pentagon formed by removing a triangular section from a rectangle --/
structure Pentagon where
  sides : Finset ℕ
  area : ℕ

/-- The theorem stating the area of the specific pentagon --/
theorem pentagon_area : ∃ (p : Pentagon), 
  p.sides = {17, 23, 26, 28, 34} ∧ p.area = 832 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l3779_377903


namespace NUMINAMATH_CALUDE_tom_initial_investment_l3779_377914

/-- Represents the initial investment of Tom in rupees -/
def tom_investment : ℝ := 3000

/-- Represents Jose's investment in rupees -/
def jose_investment : ℝ := 4500

/-- Represents the total duration of the business in months -/
def total_duration : ℝ := 12

/-- Represents the time after which Jose joined in months -/
def jose_join_time : ℝ := 2

/-- Represents the total profit in rupees -/
def total_profit : ℝ := 6300

/-- Represents Jose's share of the profit in rupees -/
def jose_profit : ℝ := 3500

theorem tom_initial_investment :
  tom_investment * total_duration / (jose_investment * (total_duration - jose_join_time)) =
  (total_profit - jose_profit) / jose_profit :=
sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l3779_377914


namespace NUMINAMATH_CALUDE_campaign_funds_proof_l3779_377952

/-- The total campaign funds raised by the 40th president -/
def total_funds : ℝ := 10000

/-- The amount raised by friends -/
def friends_contribution (total : ℝ) : ℝ := 0.4 * total

/-- The amount raised by family -/
def family_contribution (total : ℝ) : ℝ := 0.3 * (total - friends_contribution total)

/-- The amount contributed by the president himself -/
def president_contribution : ℝ := 4200

theorem campaign_funds_proof :
  friends_contribution total_funds +
  family_contribution total_funds +
  president_contribution = total_funds :=
by sorry

end NUMINAMATH_CALUDE_campaign_funds_proof_l3779_377952


namespace NUMINAMATH_CALUDE_inequality_proof_l3779_377984

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 3) :
  1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3779_377984


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l3779_377994

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 := by sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l3779_377994


namespace NUMINAMATH_CALUDE_max_shape_pairs_l3779_377996

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a pair of shapes: a corner and a 2x2 square -/
structure ShapePair where
  corner : Unit
  square : Unit

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area occupied by a single ShapePair -/
def ShapePair.area : ℕ := 7  -- 3 for corner + 4 for 2x2 square

/-- The main theorem to prove -/
theorem max_shape_pairs (r : Rectangle) (h1 : r.width = 3) (h2 : r.height = 100) :
  ∃ (n : ℕ), n = 33 ∧ 
  n * ShapePair.area ≤ r.area ∧
  ∀ (m : ℕ), m * ShapePair.area ≤ r.area → m ≤ n :=
by sorry


end NUMINAMATH_CALUDE_max_shape_pairs_l3779_377996


namespace NUMINAMATH_CALUDE_flour_yield_l3779_377988

theorem flour_yield (total : ℚ) : 
  (total - (1 / 10) * total = 1) → total = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_flour_yield_l3779_377988


namespace NUMINAMATH_CALUDE_m_range_theorem_l3779_377906

/-- The function f(x) = x^2 + mx + 1 has two distinct roots -/
def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- There exists an x such that 4x^2 + 4(m-2)x + 1 ≤ 0 -/
def exists_nonpositive (m : ℝ) : Prop :=
  ∃ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≤ 0

/-- The range of m is (-∞, -2) ∪ [3, +∞) -/
def m_range (m : ℝ) : Prop :=
  m < -2 ∨ m ≥ 3

theorem m_range_theorem (m : ℝ) :
  has_two_distinct_roots m ∧ exists_nonpositive m → m_range m := by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3779_377906


namespace NUMINAMATH_CALUDE_stock_price_return_l3779_377950

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_increase := initial_price * 1.25
  let price_after_decrease := price_after_increase * 0.8
  price_after_decrease = initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_return_l3779_377950


namespace NUMINAMATH_CALUDE_pretzel_problem_l3779_377940

theorem pretzel_problem (john_pretzels alan_pretzels marcus_pretzels initial_pretzels : ℕ) :
  john_pretzels = 28 →
  alan_pretzels = john_pretzels - 9 →
  marcus_pretzels = john_pretzels + 12 →
  marcus_pretzels = 40 →
  initial_pretzels = john_pretzels + alan_pretzels + marcus_pretzels →
  initial_pretzels = 87 := by
  sorry

end NUMINAMATH_CALUDE_pretzel_problem_l3779_377940


namespace NUMINAMATH_CALUDE_max_edge_product_sum_is_9420_l3779_377949

/-- Represents a cube with labeled vertices -/
structure LabeledCube where
  vertices : Fin 8 → ℕ
  is_square_label : ∀ i, ∃ j, vertices i = j^2
  is_permutation : Function.Bijective vertices

/-- Calculates the sum of products of numbers at the ends of each edge -/
def edge_product_sum (cube : LabeledCube) : ℕ := sorry

/-- The maximum possible sum of edge products -/
def max_edge_product_sum : ℕ := 9420

/-- Theorem stating that the maximum sum of edge products is 9420 -/
theorem max_edge_product_sum_is_9420 :
  ∀ cube : LabeledCube, edge_product_sum cube ≤ max_edge_product_sum :=
sorry

end NUMINAMATH_CALUDE_max_edge_product_sum_is_9420_l3779_377949


namespace NUMINAMATH_CALUDE_average_growth_rate_equation_l3779_377919

/-- Represents the average monthly growth rate as a real number between 0 and 1 -/
def average_growth_rate : ℝ := sorry

/-- The initial output value in January in billions of yuan -/
def initial_output : ℝ := 50

/-- The final output value in March in billions of yuan -/
def final_output : ℝ := 60

/-- The number of months between January and March -/
def months : ℕ := 2

theorem average_growth_rate_equation :
  initial_output * (1 + average_growth_rate) ^ months = final_output :=
sorry

end NUMINAMATH_CALUDE_average_growth_rate_equation_l3779_377919


namespace NUMINAMATH_CALUDE_pet_store_dogs_l3779_377982

/-- The number of dogs in a pet store after receiving additional dogs over two days -/
def total_dogs (initial : ℕ) (sunday_addition : ℕ) (monday_addition : ℕ) : ℕ :=
  initial + sunday_addition + monday_addition

/-- Theorem stating that starting with 2 dogs, adding 5 on Sunday and 3 on Monday results in 10 dogs -/
theorem pet_store_dogs : total_dogs 2 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l3779_377982


namespace NUMINAMATH_CALUDE_no_solution_for_system_l3779_377955

theorem no_solution_for_system :
  ¬ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^(1/3) - y^(1/3) - z^(1/3) = 64 ∧
    x^(1/4) - y^(1/4) - z^(1/4) = 32 ∧
    x^(1/6) - y^(1/6) - z^(1/6) = 8 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l3779_377955


namespace NUMINAMATH_CALUDE_tangent_circle_center_l3779_377971

/-- A circle tangent to two parallel lines with its center on a third line -/
structure TangentCircle where
  -- First tangent line: 3x - 4y = 20
  tangent_line1 : (ℝ × ℝ) → Prop := fun (x, y) ↦ 3 * x - 4 * y = 20
  -- Second tangent line: 3x - 4y = -40
  tangent_line2 : (ℝ × ℝ) → Prop := fun (x, y) ↦ 3 * x - 4 * y = -40
  -- Line containing the center: x - 3y = 0
  center_line : (ℝ × ℝ) → Prop := fun (x, y) ↦ x - 3 * y = 0

/-- The center of the tangent circle is at (-6, -2) -/
theorem tangent_circle_center (c : TangentCircle) : 
  ∃ (x y : ℝ), c.center_line (x, y) ∧ x = -6 ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_center_l3779_377971


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l3779_377918

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem: The fraction of upgraded sensors on the satellite is 1/4 -/
theorem upgraded_fraction_is_one_fourth (s : Satellite) 
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.total_upgraded / 8) :
  upgraded_fraction s = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_upgraded_fraction_is_one_fourth_l3779_377918


namespace NUMINAMATH_CALUDE_garrison_size_l3779_377924

/-- Represents the initial number of men in the garrison -/
def initial_men : ℕ := 150

/-- Represents the number of days the provisions were initially meant to last -/
def initial_days : ℕ := 31

/-- Represents the number of days that passed before reinforcements arrived -/
def days_before_reinforcement : ℕ := 16

/-- Represents the number of reinforcement men that arrived -/
def reinforcement_men : ℕ := 300

/-- Represents the number of days the provisions lasted after reinforcements arrived -/
def remaining_days : ℕ := 5

theorem garrison_size :
  initial_men * initial_days = 
  initial_men * (initial_days - days_before_reinforcement) ∧
  initial_men * (initial_days - days_before_reinforcement) = 
  (initial_men + reinforcement_men) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_garrison_size_l3779_377924


namespace NUMINAMATH_CALUDE_inequality_proof_l3779_377930

theorem inequality_proof (a b : ℝ) (θ : ℝ) : 
  abs a + abs b ≤ 
  Real.sqrt (a^2 * Real.cos θ^2 + b^2 * Real.sin θ^2) + 
  Real.sqrt (a^2 * Real.sin θ^2 + b^2 * Real.cos θ^2) ∧
  Real.sqrt (a^2 * Real.cos θ^2 + b^2 * Real.sin θ^2) + 
  Real.sqrt (a^2 * Real.sin θ^2 + b^2 * Real.cos θ^2) ≤ 
  Real.sqrt (2 * (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3779_377930


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l3779_377959

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) : 
  total_students = 500 → cat_owners = 75 → 
  (cat_owners : ℚ) / (total_students : ℚ) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l3779_377959


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l3779_377961

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l3779_377961


namespace NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_in_sphere_l3779_377905

theorem max_surface_area_rectangular_solid_in_sphere :
  ∀ (a b c : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a^2 + b^2 + c^2 = 4) →
  2 * (a * b + a * c + b * c) ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_in_sphere_l3779_377905


namespace NUMINAMATH_CALUDE_sophie_gave_one_box_to_mom_l3779_377923

/-- Represents the number of donuts in a box --/
def donuts_per_box : ℕ := 12

/-- Represents the number of boxes Sophie bought --/
def boxes_bought : ℕ := 4

/-- Represents the number of donuts Sophie gave to her sister --/
def donuts_to_sister : ℕ := 6

/-- Represents the number of donuts Sophie had left for herself --/
def donuts_left_for_sophie : ℕ := 30

/-- Calculates the number of boxes Sophie gave to her mom --/
def boxes_to_mom : ℕ :=
  (boxes_bought * donuts_per_box - donuts_to_sister - donuts_left_for_sophie) / donuts_per_box

theorem sophie_gave_one_box_to_mom :
  boxes_to_mom = 1 :=
sorry

end NUMINAMATH_CALUDE_sophie_gave_one_box_to_mom_l3779_377923


namespace NUMINAMATH_CALUDE_function_value_at_three_l3779_377991

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem function_value_at_three
    (f : ℝ → ℝ)
    (h1 : FunctionalEquation f)
    (h2 : f 1 = 2) :
    f 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l3779_377991


namespace NUMINAMATH_CALUDE_football_team_yardage_l3779_377908

theorem football_team_yardage (initial_loss : ℤ) : 
  (initial_loss < 0) →  -- The team lost some yards initially
  (-initial_loss + 11 = 6) →  -- The team gained 11 yards and ended up with 6 yards progress
  initial_loss = -5 :=  -- The initial loss was 5 yards
by
  sorry

end NUMINAMATH_CALUDE_football_team_yardage_l3779_377908


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3779_377920

-- Problem 1
theorem problem_1 : -3 - (-10) + (-9) - 10 = -12 := by sorry

-- Problem 2
theorem problem_2 : (1/4 : ℚ) + (-1/8) + (-7/8) - (3/4) = -3/2 := by sorry

-- Problem 3
theorem problem_3 : -25 * (-18) + (-25) * 12 + 25 * (-10) = -100 := by sorry

-- Problem 4
theorem problem_4 : -48 * (-1/6 + 3/4 - 1/24) = -26 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3779_377920


namespace NUMINAMATH_CALUDE_smaller_of_two_numbers_l3779_377927

theorem smaller_of_two_numbers (x y a b c : ℝ) : 
  x > 0 → y > 0 → x * y = c → x^2 - b*x + a*y = 0 → 0 < a → a < b → 
  min x y = c / a :=
by sorry

end NUMINAMATH_CALUDE_smaller_of_two_numbers_l3779_377927


namespace NUMINAMATH_CALUDE_sin_three_pi_fourth_minus_alpha_l3779_377986

theorem sin_three_pi_fourth_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 + α) = Real.sqrt 3 / 2) : 
  Real.sin (3 * π / 4 - α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_fourth_minus_alpha_l3779_377986


namespace NUMINAMATH_CALUDE_larger_number_l3779_377904

theorem larger_number (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l3779_377904


namespace NUMINAMATH_CALUDE_power_of_power_l3779_377962

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3779_377962


namespace NUMINAMATH_CALUDE_games_in_division_is_sixty_l3779_377917

/-- Represents a baseball league with specified conditions -/
structure BaseballLeague where
  n : ℕ  -- Number of games against each team in the same division
  m : ℕ  -- Number of games against each team in the other division
  h1 : n > 2 * m
  h2 : m > 5
  h3 : 4 * n + 5 * m = 100

/-- The number of games a team plays within its own division -/
def gamesInDivision (league : BaseballLeague) : ℕ := 4 * league.n

theorem games_in_division_is_sixty (league : BaseballLeague) :
  gamesInDivision league = 60 := by
  sorry

#check games_in_division_is_sixty

end NUMINAMATH_CALUDE_games_in_division_is_sixty_l3779_377917


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3779_377902

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  a 2 * a 5 = 32 → a 4 * a 7 = 512 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3779_377902


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3779_377976

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 100 ∣ n^3) :
  100 = Nat.gcd n (Nat.factorial 100) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3779_377976


namespace NUMINAMATH_CALUDE_walking_probability_is_four_sevenths_l3779_377941

/-- The number of bus stops -/
def num_stops : ℕ := 15

/-- The distance between adjacent stops in feet -/
def distance_between_stops : ℕ := 100

/-- The maximum walking distance in feet -/
def max_walking_distance : ℕ := 500

/-- The probability of walking 500 feet or less between two randomly chosen stops -/
def walking_probability : ℚ :=
  let total_possibilities := num_stops * (num_stops - 1)
  let favorable_outcomes := 120  -- This is derived from the problem, not the solution
  favorable_outcomes / total_possibilities

theorem walking_probability_is_four_sevenths :
  walking_probability = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_walking_probability_is_four_sevenths_l3779_377941


namespace NUMINAMATH_CALUDE_certain_number_problem_l3779_377909

theorem certain_number_problem : ∃ x : ℝ, 11*x + 12*x + 15*x + 11 = 125 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3779_377909


namespace NUMINAMATH_CALUDE_solve_matrix_vector_problem_l3779_377977

def matrix_vector_problem (M : Matrix (Fin 2) (Fin 2) ℝ) 
                          (u v w : Fin 2 → ℝ) : Prop :=
  (M.mulVec u = ![1, 2]) ∧ 
  (M.mulVec v = ![3, 4]) ∧ 
  (M.mulVec w = ![5, 6]) →
  M.mulVec (2 • u + v - 2 • w) = ![-5, -4]

theorem solve_matrix_vector_problem :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ) (u v w : Fin 2 → ℝ),
  matrix_vector_problem M u v w :=
by
  sorry

end NUMINAMATH_CALUDE_solve_matrix_vector_problem_l3779_377977


namespace NUMINAMATH_CALUDE_latest_start_time_is_10am_l3779_377990

-- Define the number of turkeys
def num_turkeys : ℕ := 2

-- Define the weight of each turkey in pounds
def turkey_weight : ℕ := 16

-- Define the roasting time per pound in minutes
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time (18:00 in 24-hour format)
def dinner_time : ℕ := 18 * 60

-- Define the function to calculate the total roasting time in minutes
def total_roasting_time : ℕ := num_turkeys * turkey_weight * roasting_time_per_pound

-- Define the function to calculate the latest start time in minutes after midnight
def latest_start_time : ℕ := dinner_time - total_roasting_time

-- Theorem stating that the latest start time is 10:00 am (600 minutes after midnight)
theorem latest_start_time_is_10am : latest_start_time = 600 := by
  sorry

end NUMINAMATH_CALUDE_latest_start_time_is_10am_l3779_377990


namespace NUMINAMATH_CALUDE_double_percentage_increase_l3779_377956

theorem double_percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1 + 44 / 100 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_double_percentage_increase_l3779_377956


namespace NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3779_377951

theorem cyclic_ratio_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3779_377951


namespace NUMINAMATH_CALUDE_president_and_committee_from_ten_l3779_377979

/-- The number of ways to choose a president and a committee from a group --/
def choose_president_and_committee (group_size : ℕ) (committee_size : ℕ) : ℕ :=
  group_size * Nat.choose (group_size - 1) committee_size

/-- Theorem stating the number of ways to choose a president and a 3-person committee from 10 people --/
theorem president_and_committee_from_ten :
  choose_president_and_committee 10 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_from_ten_l3779_377979


namespace NUMINAMATH_CALUDE_division_problem_l3779_377946

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 122 →
  quotient = 6 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  divisor = 20 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3779_377946


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3779_377981

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 6*x - 1 = 0) ↔ ((x - 3)^2 = 10) := by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3779_377981


namespace NUMINAMATH_CALUDE_sequence_properties_l3779_377931

theorem sequence_properties :
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n, a (n + 1) = a n + n + 1) ∧ a 20 = 211) ∧
  (∃ b : ℕ → ℕ, b 1 = 1 ∧ (∀ n, b (n + 1) = 3 * b n + 2) ∧ b 4 = 53) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3779_377931


namespace NUMINAMATH_CALUDE_minimum_walnuts_l3779_377993

/-- Represents the process of a child dividing and taking walnuts -/
def childProcess (n : ℕ) : ℕ := (n - 1) * 4 / 5

/-- Represents the final division process -/
def finalDivision (n : ℕ) : ℕ := n - 1

/-- Represents the entire walnut distribution process -/
def walnutDistribution (initial : ℕ) : ℕ :=
  finalDivision (childProcess (childProcess (childProcess (childProcess (childProcess initial)))))

theorem minimum_walnuts :
  ∃ (n : ℕ), n > 0 ∧ walnutDistribution n = 0 ∧ ∀ m : ℕ, m > 0 ∧ m < n → walnutDistribution m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_minimum_walnuts_l3779_377993


namespace NUMINAMATH_CALUDE_power_of_point_outside_circle_l3779_377910

/-- Given a circle with radius R and a point M outside the circle at distance d from the center,
    prove that for any line through M intersecting the circle at A and B, MA * MB = d² - R² -/
theorem power_of_point_outside_circle (R d : ℝ) (h : 0 < R) (h' : R < d) :
  ∀ (M A B : ℝ × ℝ),
    ‖M - (0, 0)‖ = d →
    ‖A - (0, 0)‖ = R →
    ‖B - (0, 0)‖ = R →
    (∃ t : ℝ, A = M + t • (B - M)) →
    ‖M - A‖ * ‖M - B‖ = d^2 - R^2 :=
by sorry

end NUMINAMATH_CALUDE_power_of_point_outside_circle_l3779_377910


namespace NUMINAMATH_CALUDE_rectangular_program_box_indicates_input_output_l3779_377997

/-- Represents the function of a program box in an algorithm -/
inductive ProgramBoxFunction
  | StartEnd
  | InputOutput
  | AssignmentCalculation
  | ConnectBoxes

/-- The function of a rectangular program box in an algorithm -/
def rectangularProgramBoxFunction : ProgramBoxFunction := ProgramBoxFunction.InputOutput

/-- Theorem stating that a rectangular program box indicates input and output information -/
theorem rectangular_program_box_indicates_input_output :
  rectangularProgramBoxFunction = ProgramBoxFunction.InputOutput := by
  sorry

end NUMINAMATH_CALUDE_rectangular_program_box_indicates_input_output_l3779_377997


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_x_l3779_377939

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_x :
  (∃ x : ℝ, x > 1 ∧ x^2 > x) ∧
  (∃ x : ℝ, x^2 > x ∧ ¬(x > 1)) ∧
  (∀ x : ℝ, x > 1 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_x_l3779_377939


namespace NUMINAMATH_CALUDE_z_sixth_power_l3779_377987

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 3 + Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_sixth_power_l3779_377987


namespace NUMINAMATH_CALUDE_fred_has_eighteen_balloons_l3779_377995

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 6

/-- The factor by which Fred has more balloons than Sally -/
def fred_factor : ℕ := 3

/-- The number of blue balloons Fred has -/
def fred_balloons : ℕ := sally_balloons * fred_factor

/-- Theorem stating that Fred has 18 blue balloons -/
theorem fred_has_eighteen_balloons : fred_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_eighteen_balloons_l3779_377995


namespace NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l3779_377921

theorem no_geometric_progression_of_2n_plus_1 :
  ¬ ∃ (k m n : ℕ), k ≠ m ∧ m ≠ n ∧ k ≠ n ∧
    (2^m + 1)^2 = (2^k + 1) * (2^n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l3779_377921


namespace NUMINAMATH_CALUDE_oranges_remaining_proof_l3779_377983

/-- The number of oranges Michaela needs to eat until she gets full -/
def michaela_oranges : ℕ := 30

/-- The number of oranges Cassandra needs to eat until she gets full -/
def cassandra_oranges : ℕ := 3 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 200

/-- The number of oranges remaining after Michaela and Cassandra eat until they're full -/
def remaining_oranges : ℕ := total_oranges - (michaela_oranges + cassandra_oranges)

theorem oranges_remaining_proof : remaining_oranges = 80 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_proof_l3779_377983


namespace NUMINAMATH_CALUDE_sine_inequality_solution_l3779_377958

theorem sine_inequality_solution (x y : Real) :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), 
   ∀ y ∈ Set.Icc 0 (2 * Real.pi), 
   Real.sin (x + y) ≤ Real.sin x + Real.sin y) ↔ 
  y ∈ Set.Icc 0 Real.pi :=
sorry

end NUMINAMATH_CALUDE_sine_inequality_solution_l3779_377958


namespace NUMINAMATH_CALUDE_trapezium_side_length_l3779_377943

/-- Given a trapezium with area 342 cm², one parallel side of 14 cm, and height 18 cm,
    prove that the length of the other parallel side is 24 cm. -/
theorem trapezium_side_length (area : ℝ) (side1 : ℝ) (height : ℝ) (side2 : ℝ) :
  area = 342 →
  side1 = 14 →
  height = 18 →
  area = (1 / 2) * (side1 + side2) * height →
  side2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l3779_377943


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_l3779_377968

/-- Given a rectangle ACDE, a point B on AC, a point F on AE, and an equilateral triangle CEF,
    prove that the area of ACDE + CEF - ABF is 1100 + (225 * Real.sqrt 3) / 4 -/
theorem rectangle_triangle_area (A B C D E F : ℝ × ℝ) : 
  let AC : ℝ := 40
  let AE : ℝ := 30
  let AB : ℝ := AC / 3
  let AF : ℝ := AE / 2
  let area_ACDE : ℝ := AC * AE
  let area_CEF : ℝ := (Real.sqrt 3 / 4) * AF^2
  let area_ABF : ℝ := (1 / 2) * AB * AF
  area_ACDE + area_CEF - area_ABF = 1100 + (225 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_l3779_377968


namespace NUMINAMATH_CALUDE_equation_solution_l3779_377998

theorem equation_solution (x : ℝ) (some_number : ℝ) 
  (h1 : x + 1 = some_number) (h2 : x = 4) : some_number = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3779_377998


namespace NUMINAMATH_CALUDE_correct_alarm_time_l3779_377913

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60 := by sorry

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ := by sorry

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time := by sorry

theorem correct_alarm_time :
  let alarmSetTime : Time := ⟨7, 0, by sorry⟩
  let museumArrivalTime : Time := ⟨8, 50, by sorry⟩
  let museumVisitDuration : ℕ := 90 -- in minutes
  let returnHomeTime : Time := ⟨11, 50, by sorry⟩
  
  let totalTripTime := timeDifference alarmSetTime returnHomeTime
  let walkingTime := totalTripTime - museumVisitDuration
  let oneWayWalkingTime := walkingTime / 2
  
  let museumDepartureTime := addMinutes museumArrivalTime museumVisitDuration
  let actualReturnTime := addMinutes museumDepartureTime oneWayWalkingTime
  
  let correctTime := addMinutes actualReturnTime 30

  correctTime = ⟨12, 0, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_correct_alarm_time_l3779_377913


namespace NUMINAMATH_CALUDE_larger_number_proof_l3779_377907

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 7 * S + 15) : 
  L = 1590 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3779_377907


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3779_377978

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3779_377978


namespace NUMINAMATH_CALUDE_vanessa_large_orders_l3779_377925

/-- The number of grams of packing peanuts needed for a large order -/
def large_order_peanuts : ℕ := 200

/-- The number of grams of packing peanuts needed for a small order -/
def small_order_peanuts : ℕ := 50

/-- The total number of grams of packing peanuts used -/
def total_peanuts_used : ℕ := 800

/-- The number of small orders sent -/
def num_small_orders : ℕ := 4

/-- The number of large orders sent -/
def num_large_orders : ℕ := 3

theorem vanessa_large_orders :
  num_large_orders * large_order_peanuts + num_small_orders * small_order_peanuts = total_peanuts_used :=
by sorry

end NUMINAMATH_CALUDE_vanessa_large_orders_l3779_377925


namespace NUMINAMATH_CALUDE_min_intersection_cardinality_l3779_377900

-- Define the cardinality of a set
def card (S : Set α) : ℕ := sorry

-- Define the number of subsets of a set
def n (S : Set α) : ℕ := 2^(card S)

-- Define the theorem
theorem min_intersection_cardinality 
  (A B C : Set α) 
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : card A = 100)
  (h3 : card B = 101)
  (h4 : card (A ∩ B) ≥ 95) :
  96 ≤ card (A ∩ B ∩ C) := by
  sorry

end NUMINAMATH_CALUDE_min_intersection_cardinality_l3779_377900


namespace NUMINAMATH_CALUDE_number_of_divisors_of_n_l3779_377945

def n : ℕ := 293601000

theorem number_of_divisors_of_n : Nat.card {d : ℕ | d ∣ n} = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_n_l3779_377945


namespace NUMINAMATH_CALUDE_hyperbola_specific_equation_l3779_377901

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The general equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The focus of a hyperbola -/
def focus (x y : ℝ) : Prop := x = 2 ∧ y = 0

/-- The asymptotes of a hyperbola -/
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The theorem stating the specific equation of the hyperbola given the conditions -/
theorem hyperbola_specific_equation (a b : ℝ) (h : Hyperbola a b) 
  (focus_cond : focus 2 0)
  (asymp_cond : ∀ x y, asymptotes x y ↔ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) :
  ∀ x y, hyperbola_equation h x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_specific_equation_l3779_377901


namespace NUMINAMATH_CALUDE_updated_mean_example_l3779_377934

/-- The updated mean of a dataset after corrections -/
def updated_mean (original_mean original_count : ℕ) 
                 (decrement : ℕ) 
                 (missing_obs : List ℕ) 
                 (extra_obs : ℕ) : ℚ :=
  let original_sum := original_mean * original_count
  let corrected_sum := original_sum - decrement * original_count + missing_obs.sum - extra_obs
  let corrected_count := original_count - 1 + missing_obs.length
  (corrected_sum : ℚ) / corrected_count

/-- Theorem stating the updated mean after corrections -/
theorem updated_mean_example : 
  updated_mean 200 50 34 [150, 190, 210] 250 = 8600 / 52 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_example_l3779_377934


namespace NUMINAMATH_CALUDE_plate_arrangement_circular_table_l3779_377929

def plate_arrangement (b r g o y : ℕ) : ℕ :=
  let total := b + r + g + o + y
  let all_arrangements := Nat.factorial (total - 1) / (Nat.factorial b * Nat.factorial r * Nat.factorial g * Nat.factorial o * Nat.factorial y)
  let adjacent_green := Nat.factorial (total - g + 1) / (Nat.factorial b * Nat.factorial r * Nat.factorial o * Nat.factorial y) * Nat.factorial g
  all_arrangements - adjacent_green

theorem plate_arrangement_circular_table :
  plate_arrangement 6 3 3 2 2 = 
    Nat.factorial 15 / (Nat.factorial 6 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2) - 
    (Nat.factorial 14 / (Nat.factorial 6 * Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1) * Nat.factorial 3) :=
by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_circular_table_l3779_377929


namespace NUMINAMATH_CALUDE_expected_distinct_faces_formula_l3779_377928

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of times the die is rolled -/
def numRolls : ℕ := 6

/-- The probability that a specific face does not appear in any of the rolls -/
def probNoAppearance : ℚ := (numFaces - 1 : ℚ) / numFaces ^ numRolls

/-- The expected number of distinct faces that appear when a die is rolled multiple times -/
def expectedDistinctFaces : ℚ := numFaces * (1 - probNoAppearance)

/-- Theorem: The expected number of distinct faces that appear when a die is rolled six times 
    is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_distinct_faces_formula : 
  expectedDistinctFaces = (numFaces ^ numRolls - (numFaces - 1) ^ numRolls : ℚ) / numFaces ^ (numRolls - 1) := by
  sorry

end NUMINAMATH_CALUDE_expected_distinct_faces_formula_l3779_377928


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l3779_377965

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 225) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 28 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l3779_377965


namespace NUMINAMATH_CALUDE_candidate_count_l3779_377937

theorem candidate_count (total_selections : ℕ) (h : total_selections = 90) : 
  ∃ n : ℕ, n * (n - 1) = total_selections ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_candidate_count_l3779_377937


namespace NUMINAMATH_CALUDE_annual_insurance_payment_l3779_377974

/-- The number of quarters in a year -/
def quarters_per_year : ℕ := 4

/-- The quarterly insurance payment in dollars -/
def quarterly_payment : ℕ := 378

/-- The annual insurance payment in dollars -/
def annual_payment : ℕ := quarterly_payment * quarters_per_year

theorem annual_insurance_payment :
  annual_payment = 1512 :=
by sorry

end NUMINAMATH_CALUDE_annual_insurance_payment_l3779_377974


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_denominator_factorization_l3779_377964

theorem partial_fraction_decomposition (x : ℝ) : 
  let A : ℝ := 1/2
  let B : ℝ := 9/2
  (6*x - 7) / (3*x^2 + 2*x - 8) = A / (x - 2) + B / (3*x + 4) :=
by
  sorry

-- Auxiliary theorem to establish the factorization of the denominator
theorem denominator_factorization (x : ℝ) :
  3*x^2 + 2*x - 8 = (3*x + 4)*(x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_denominator_factorization_l3779_377964


namespace NUMINAMATH_CALUDE_garden_perimeter_l3779_377966

theorem garden_perimeter (garden_width playground_length : ℝ) 
  (h1 : garden_width = 4)
  (h2 : playground_length = 16)
  (h3 : ∃ (garden_length playground_width : ℝ), 
    garden_width * garden_length = playground_length * playground_width)
  (h4 : ∃ (garden_length : ℝ), 2 * (garden_width + garden_length) = 104) :
  ∃ (garden_length : ℝ), 2 * (garden_width + garden_length) = 104 :=
by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_garden_perimeter_l3779_377966


namespace NUMINAMATH_CALUDE_function_continuity_l3779_377915

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f(x) + f(ax) is continuous for any a > 1
def condition (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 1 → Continuous (fun x ↦ f x + f (a * x))

-- State the theorem
theorem function_continuity (hf : condition f) : Continuous f := by
  sorry

end NUMINAMATH_CALUDE_function_continuity_l3779_377915


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l3779_377947

theorem unique_congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ -200 ≡ n [ZMOD 19] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l3779_377947


namespace NUMINAMATH_CALUDE_chives_count_l3779_377953

/-- Given a garden with the following properties:
  - The garden has 20 rows with 10 plants in each row.
  - Parsley is planted in the first 3 rows.
  - Rosemary is planted in the last 2 rows.
  - The remaining rows are planted with chives.
  This theorem proves that the number of chives planted is 150. -/
theorem chives_count (total_rows : ℕ) (plants_per_row : ℕ) 
  (parsley_rows : ℕ) (rosemary_rows : ℕ) : 
  total_rows = 20 → 
  plants_per_row = 10 → 
  parsley_rows = 3 → 
  rosemary_rows = 2 → 
  (total_rows - (parsley_rows + rosemary_rows)) * plants_per_row = 150 := by
  sorry

end NUMINAMATH_CALUDE_chives_count_l3779_377953


namespace NUMINAMATH_CALUDE_six_digit_palindrome_divisible_by_11_l3779_377948

theorem six_digit_palindrome_divisible_by_11 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  let W := 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a
  11 ∣ W :=
by sorry

end NUMINAMATH_CALUDE_six_digit_palindrome_divisible_by_11_l3779_377948


namespace NUMINAMATH_CALUDE_original_number_proof_l3779_377957

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 10^4 * x = 4 * (1/x)) : x = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3779_377957


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3779_377992

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_fourth : a 4 = 512) 
  (h_ninth : a 9 = 8) : 
  a 6 = 128 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3779_377992


namespace NUMINAMATH_CALUDE_melanie_initial_dimes_l3779_377932

theorem melanie_initial_dimes :
  ∀ (initial : ℕ), 
    (initial + 8 + 4 = 19) → initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_melanie_initial_dimes_l3779_377932


namespace NUMINAMATH_CALUDE_sqrt_square_equality_implies_geq_l3779_377999

theorem sqrt_square_equality_implies_geq (a : ℝ) : 
  Real.sqrt ((a - 2)^2) = a - 2 → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_equality_implies_geq_l3779_377999


namespace NUMINAMATH_CALUDE_list_number_relation_l3779_377985

theorem list_number_relation (l : List ℝ) (n : ℝ) : 
  l.length = 21 ∧ 
  n ∈ l ∧ 
  n = (l.sum / 6) →
  n = 4 * ((l.sum - n) / 20) := by
sorry

end NUMINAMATH_CALUDE_list_number_relation_l3779_377985


namespace NUMINAMATH_CALUDE_same_color_probability_eight_nine_l3779_377926

/-- The probability of drawing two balls of the same color from a box containing 
    8 white balls and 9 black balls. -/
def same_color_probability (white : ℕ) (black : ℕ) : ℚ :=
  let total := white + black
  let same_color_ways := (white.choose 2) + (black.choose 2)
  let total_ways := total.choose 2
  same_color_ways / total_ways

/-- Theorem stating that the probability of drawing two balls of the same color 
    from a box with 8 white balls and 9 black balls is 8/17. -/
theorem same_color_probability_eight_nine : 
  same_color_probability 8 9 = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_eight_nine_l3779_377926


namespace NUMINAMATH_CALUDE_base_subtraction_equals_160_l3779_377916

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem base_subtraction_equals_160 :
  let base9_to_decimal := base_to_decimal [3, 2, 5] 9
  let base6_to_decimal := base_to_decimal [2, 5, 4] 6
  base9_to_decimal - base6_to_decimal = 160 := by
sorry

end NUMINAMATH_CALUDE_base_subtraction_equals_160_l3779_377916


namespace NUMINAMATH_CALUDE_three_objects_five_containers_l3779_377942

/-- The number of ways to place n distinct objects into m distinct containers -/
def placement_count (n m : ℕ) : ℕ := m^n

/-- Theorem: Placing 3 distinct objects into 5 distinct containers results in 125 different arrangements -/
theorem three_objects_five_containers : placement_count 3 5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_three_objects_five_containers_l3779_377942


namespace NUMINAMATH_CALUDE_maria_papers_left_l3779_377967

/-- The number of papers Maria has left after giving away some papers -/
def papers_left (desk : ℕ) (backpack : ℕ) (given_away : ℕ) : ℕ :=
  desk + backpack - given_away

/-- Theorem stating that Maria has 91 - x papers left after giving away x papers -/
theorem maria_papers_left (x : ℕ) :
  papers_left 50 41 x = 91 - x :=
by sorry

end NUMINAMATH_CALUDE_maria_papers_left_l3779_377967


namespace NUMINAMATH_CALUDE_antenna_tower_height_l3779_377960

/-- Given an antenna tower on flat terrain, if the sum of the angles of elevation
    measured at distances of 100 m, 200 m, and 300 m from its base is 90°,
    then the height of the tower is 100 m. -/
theorem antenna_tower_height (α β γ : Real) (h : Real) :
  (α + β + γ = Real.pi / 2) →
  (h / 100 = Real.tan α) →
  (h / 200 = Real.tan β) →
  (h / 300 = Real.tan γ) →
  h = 100 := by
  sorry

#check antenna_tower_height

end NUMINAMATH_CALUDE_antenna_tower_height_l3779_377960


namespace NUMINAMATH_CALUDE_cube_edge_length_l3779_377963

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 24) :
  ∃ edge_length : ℝ, edge_length > 0 ∧ 6 * edge_length^2 = surface_area ∧ edge_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l3779_377963


namespace NUMINAMATH_CALUDE_probability_x_plus_2y_leq_6_l3779_377954

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5}

-- Define the condition x + 2y ≤ 6
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + 2 * p.2 ≤ 6

-- Define the probability measure on the region
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) :=
  sorry

-- State the theorem
theorem probability_x_plus_2y_leq_6 :
  prob {p ∈ region | condition p} / prob region = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_2y_leq_6_l3779_377954


namespace NUMINAMATH_CALUDE_triangle_properties_l3779_377970

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 * b ∧
  Real.sin A + Real.sin B = 2 * Real.sin C ∧
  (1 / 2) * b * c * Real.sin A = (8 * Real.sqrt 15) / 3 →
  Real.cos A = -1 / 4 ∧ c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3779_377970


namespace NUMINAMATH_CALUDE_reciprocal_of_opposite_l3779_377922

theorem reciprocal_of_opposite (x : ℝ) (h : x ≠ 0) : 
  (-(1 / x)) = 1 / (-x) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_of_opposite_l3779_377922


namespace NUMINAMATH_CALUDE_system_solution_l3779_377969

theorem system_solution :
  ∀ x y : ℕ+,
  (x.val * y.val + x.val + y.val = 71 ∧
   x.val^2 * y.val + x.val * y.val^2 = 880) →
  ((x.val = 11 ∧ y.val = 5) ∨ (x.val = 5 ∧ y.val = 11)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3779_377969


namespace NUMINAMATH_CALUDE_donut_theft_ratio_l3779_377933

theorem donut_theft_ratio (initial_donuts : ℕ) (bill_eaten : ℕ) (secretary_taken : ℕ) (final_donuts : ℕ)
  (h1 : initial_donuts = 50)
  (h2 : bill_eaten = 2)
  (h3 : secretary_taken = 4)
  (h4 : final_donuts = 22) :
  (initial_donuts - bill_eaten - secretary_taken - final_donuts) / (initial_donuts - bill_eaten - secretary_taken) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_donut_theft_ratio_l3779_377933


namespace NUMINAMATH_CALUDE_limit_ratio_recursive_sequences_l3779_377912

/-- Two sequences satisfying given recursive relations -/
def RecursiveSequences (a b : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ b 1 = 7 ∧
  ∀ n, a (n + 1) = b n - 2 * a n ∧ b (n + 1) = 3 * b n - 4 * a n

/-- The limit of the ratio of two sequences -/
def LimitRatio (a b : ℕ → ℝ) (l : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n / b n - l| < ε

/-- The main theorem stating the limit of the ratio of the sequences -/
theorem limit_ratio_recursive_sequences (a b : ℕ → ℝ) (h : RecursiveSequences a b) :
  LimitRatio a b (1/4) := by
  sorry

end NUMINAMATH_CALUDE_limit_ratio_recursive_sequences_l3779_377912


namespace NUMINAMATH_CALUDE_cafe_staff_remaining_l3779_377989

/-- Calculates the total number of remaining staff given the initial numbers and dropouts. -/
def remaining_staff (initial_chefs initial_waiters chefs_dropout waiters_dropout : ℕ) : ℕ :=
  (initial_chefs - chefs_dropout) + (initial_waiters - waiters_dropout)

/-- Theorem stating that given the specific numbers in the problem, the total remaining staff is 23. -/
theorem cafe_staff_remaining :
  remaining_staff 16 16 6 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_cafe_staff_remaining_l3779_377989


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3779_377975

theorem line_tangent_to_parabola :
  ∃ (m : ℝ), m = 49 ∧
  ∀ (x y : ℝ),
    (4 * x + 7 * y + m = 0) →
    (y^2 = 16 * x) →
    ∃! (x₀ y₀ : ℝ), 4 * x₀ + 7 * y₀ + m = 0 ∧ y₀^2 = 16 * x₀ :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3779_377975


namespace NUMINAMATH_CALUDE_floor_counterexamples_l3779_377935

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Statement of the theorem
theorem floor_counterexamples : ∃ (x y : ℝ),
  (floor (2^x) ≠ floor (2^(floor x))) ∧
  (floor (y^2) ≠ (floor y)^2) := by
  sorry

end NUMINAMATH_CALUDE_floor_counterexamples_l3779_377935


namespace NUMINAMATH_CALUDE_solution_characterization_l3779_377936

def f (p : ℕ × ℕ) : ℕ × ℕ :=
  (p.2, 5 * p.2 - p.1)

def h (p : ℕ × ℕ) : ℕ × ℕ :=
  (p.2, p.1)

def solution_set : Set (ℕ × ℕ) :=
  {(1, 2), (1, 3), (2, 1), (3, 1)} ∪
  {p | ∃ n : ℕ, p = Nat.iterate f n (1, 2) ∨ p = Nat.iterate f n (1, 3)} ∪
  {p | ∃ n : ℕ, p = h (Nat.iterate f n (1, 2)) ∨ p = h (Nat.iterate f n (1, 3))}

theorem solution_characterization :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (x^2 + y^2 - 5*x*y + 5 = 0 ↔ (x, y) ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l3779_377936


namespace NUMINAMATH_CALUDE_relationship_abc_l3779_377972

theorem relationship_abc : 
  let a : ℝ := 1.1 * Real.log 1.1
  let b : ℝ := 0.1 * Real.exp 0.1
  let c : ℝ := 1 / 9
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3779_377972


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3779_377980

/-- The interest rate (as a percentage) at which A lent money to B -/
def interest_rate_A_to_B : ℝ := 10

/-- The principal amount lent -/
def principal : ℝ := 3500

/-- The interest rate (as a percentage) at which B lent money to C -/
def interest_rate_B_to_C : ℝ := 15

/-- The time period in years -/
def time : ℝ := 3

/-- B's gain over the time period -/
def B_gain : ℝ := 525

theorem interest_rate_calculation :
  let interest_C := principal * interest_rate_B_to_C / 100 * time
  let interest_A := interest_C - B_gain
  interest_A = principal * interest_rate_A_to_B / 100 * time := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3779_377980
