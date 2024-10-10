import Mathlib

namespace not_p_or_q_l1292_129222

-- Define proposition p
def p : Prop := ∀ (A B C : ℝ) (sinA sinB : ℝ),
  (sinA = Real.sin A ∧ sinB = Real.sin B) →
  (A > B → sinA > sinB) ∧ ¬(sinA > sinB → A > B)

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + 2*x + 2 ≤ 0

-- Theorem to prove
theorem not_p_or_q : ¬p ∨ q := by sorry

end not_p_or_q_l1292_129222


namespace equal_incircle_radii_of_original_triangles_l1292_129268

/-- A structure representing a triangle with an inscribed circle -/
structure TriangleWithIncircle where
  vertices : Fin 3 → ℝ × ℝ
  incircle_center : ℝ × ℝ
  incircle_radius : ℝ

/-- A structure representing the configuration of two intersecting triangles -/
structure IntersectingTriangles where
  triangle1 : TriangleWithIncircle
  triangle2 : TriangleWithIncircle
  hexagon_vertices : Fin 6 → ℝ × ℝ
  small_triangles : Fin 6 → TriangleWithIncircle

/-- The theorem statement -/
theorem equal_incircle_radii_of_original_triangles 
  (config : IntersectingTriangles)
  (h_equal_small_radii : ∀ i j : Fin 6, (config.small_triangles i).incircle_radius = (config.small_triangles j).incircle_radius) :
  config.triangle1.incircle_radius = config.triangle2.incircle_radius :=
sorry

end equal_incircle_radii_of_original_triangles_l1292_129268


namespace limit_of_a_l1292_129229

def a (n : ℕ) : ℚ := (2 * n + 3) / (n + 5)

theorem limit_of_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
by
  sorry

end limit_of_a_l1292_129229


namespace largest_b_value_l1292_129255

theorem largest_b_value (b : ℚ) (h : (3 * b + 7) * (b - 2) = 8 * b) : 
  b ≤ 7 / 2 ∧ ∃ (b₀ : ℚ), (3 * b₀ + 7) * (b₀ - 2) = 8 * b₀ ∧ b₀ = 7 / 2 := by
  sorry

end largest_b_value_l1292_129255


namespace proportionality_statements_l1292_129234

-- Define the basic concepts
def is_direct_proportion (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * g x

def is_inverse_proportion (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x * g x = k

def is_not_proportional (f g : ℝ → ℝ) : Prop :=
  ¬(is_direct_proportion f g) ∧ ¬(is_inverse_proportion f g)

-- Define the specific relationships
def brick_area (n : ℝ) : ℝ := sorry
def brick_count (n : ℝ) : ℝ := sorry

def walk_speed (t : ℝ) : ℝ := sorry
def walk_time (t : ℝ) : ℝ := sorry

def circle_area (r : ℝ) : ℝ := sorry
def circle_radius (r : ℝ) : ℝ := sorry

-- State the theorem
theorem proportionality_statements :
  (is_direct_proportion brick_area brick_count) ∧
  (is_inverse_proportion walk_speed walk_time) ∧
  (is_not_proportional circle_area circle_radius) :=
by sorry

end proportionality_statements_l1292_129234


namespace william_car_wash_body_time_l1292_129264

/-- Represents the time William spends washing vehicles -/
def WilliamCarWash :=
  {time_body : ℕ //
    ∃ (time_normal time_suv : ℕ),
      time_normal = time_body + 17 ∧
      time_suv = 2 * time_normal ∧
      2 * time_normal + time_suv = 96}

/-- Theorem stating that William spends 7 minutes washing the car body -/
theorem william_car_wash_body_time :
  ∀ w : WilliamCarWash, w.val = 7 := by
  sorry

end william_car_wash_body_time_l1292_129264


namespace cs_majors_consecutive_probability_l1292_129267

def total_people : ℕ := 12
def cs_majors : ℕ := 5
def chem_majors : ℕ := 4
def lit_majors : ℕ := 3

theorem cs_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let consecutive_arrangements := Nat.factorial (total_people - cs_majors) * Nat.factorial cs_majors
  (consecutive_arrangements : ℚ) / total_arrangements = 1 / 66 := by
  sorry

end cs_majors_consecutive_probability_l1292_129267


namespace waiter_customer_count_l1292_129258

theorem waiter_customer_count :
  let num_tables : ℕ := 9
  let women_per_table : ℕ := 7
  let men_per_table : ℕ := 3
  let total_customers := num_tables * (women_per_table + men_per_table)
  total_customers = 90 := by
  sorry

end waiter_customer_count_l1292_129258


namespace four_distinct_roots_l1292_129269

theorem four_distinct_roots (m : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ (x : ℝ), x^2 - 4*|x| + 5 - m = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)))
  ↔ (1 < m ∧ m < 5) :=
sorry

end four_distinct_roots_l1292_129269


namespace refrigerator_savings_l1292_129295

/- Define the parameters of the problem -/
def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def num_installments : ℕ := 30
def installment_amount : ℕ := 300

/- Define the total amount paid in installments -/
def total_installments : ℕ := num_installments * installment_amount + deposit

/- Theorem statement -/
theorem refrigerator_savings : total_installments - cash_price = 4000 := by
  sorry

end refrigerator_savings_l1292_129295


namespace line_inclination_45_degrees_l1292_129225

/-- Proves that for a line passing through points (1, 2) and (3, m) with an inclination angle of 45°, m = 4 -/
theorem line_inclination_45_degrees (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (1, 2) ∈ line ∧ 
    (3, m) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - 2) = (x - 1))) → 
  m = 4 := by
sorry

end line_inclination_45_degrees_l1292_129225


namespace train_bridge_crossing_time_l1292_129254

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 80)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 295) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l1292_129254


namespace polynomial_problem_l1292_129217

theorem polynomial_problem (f : ℝ → ℝ) :
  (∃ (a b c d e : ℤ), ∀ x, f x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  f (1 + Real.rpow 3 (1/3)) = 1 + Real.rpow 3 (1/3) →
  f (1 + Real.sqrt 3) = 7 + Real.sqrt 3 →
  ∀ x, f x = x^4 - 3*x^3 + 3*x^2 - 3*x :=
by sorry

end polynomial_problem_l1292_129217


namespace quadrilateral_angle_sum_l1292_129227

structure Quadrilateral where
  diagonals_intersect : Bool
  intersection_not_on_side : Bool

def sum_of_angles (q : Quadrilateral) : ℝ :=
  if q.diagonals_intersect ∧ q.intersection_not_on_side then 720 else 0

theorem quadrilateral_angle_sum (q : Quadrilateral) 
  (h1 : q.diagonals_intersect = true) 
  (h2 : q.intersection_not_on_side = true) : 
  sum_of_angles q = 720 := by
  sorry

#check quadrilateral_angle_sum

end quadrilateral_angle_sum_l1292_129227


namespace university_applications_l1292_129296

theorem university_applications (n : ℕ) (s : Fin 5 → ℕ) : 
  (∀ i, s i ≥ n / 2) → 
  ∃ i j, i ≠ j ∧ (s i).min (s j) ≥ n / 5 := by
  sorry


end university_applications_l1292_129296


namespace max_sum_four_numbers_l1292_129213

theorem max_sum_four_numbers (a b c d : ℕ) :
  a < b → b < c → c < d →
  (b + d) + (c + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 806 := by
  sorry

end max_sum_four_numbers_l1292_129213


namespace frank_pizza_slices_l1292_129259

/-- Proves that Frank ate 3 slices of Hawaiian pizza given the conditions of the problem -/
theorem frank_pizza_slices (total_slices dean_slices sammy_slices leftover_slices : ℕ) :
  total_slices = 2 * 12 →
  dean_slices = 12 / 2 →
  sammy_slices = 12 / 3 →
  leftover_slices = 11 →
  total_slices - leftover_slices - dean_slices - sammy_slices = 3 := by
  sorry

#check frank_pizza_slices

end frank_pizza_slices_l1292_129259


namespace triangle_equation_no_real_roots_l1292_129216

theorem triangle_equation_no_real_roots 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∀ x : ℝ, a^2 * x^2 - (c^2 - a^2 - b^2) * x + b^2 ≠ 0 := by
sorry

end triangle_equation_no_real_roots_l1292_129216


namespace combined_average_score_l1292_129263

theorem combined_average_score (score1 score2 : ℝ) (ratio1 ratio2 : ℕ) : 
  score1 = 88 →
  score2 = 75 →
  ratio1 = 2 →
  ratio2 = 3 →
  (ratio1 * score1 + ratio2 * score2) / (ratio1 + ratio2) = 80 :=
by sorry

end combined_average_score_l1292_129263


namespace average_fuel_efficiency_l1292_129288

/-- Calculate the average fuel efficiency for a round trip with two different vehicles -/
theorem average_fuel_efficiency
  (total_distance : ℝ)
  (distance_first_leg : ℝ)
  (efficiency_first_vehicle : ℝ)
  (efficiency_second_vehicle : ℝ)
  (h1 : total_distance = 300)
  (h2 : distance_first_leg = total_distance / 2)
  (h3 : efficiency_first_vehicle = 50)
  (h4 : efficiency_second_vehicle = 25) :
  (total_distance) / ((distance_first_leg / efficiency_first_vehicle) + 
  (distance_first_leg / efficiency_second_vehicle)) = 33 := by
  sorry

#check average_fuel_efficiency

end average_fuel_efficiency_l1292_129288


namespace probability_of_double_l1292_129299

/-- The number of integers on the dominoes (from 0 to 12, inclusive) -/
def n : ℕ := 12

/-- The total number of dominoes in the set -/
def total_dominoes : ℕ := (n + 1) * (n + 2) / 2

/-- The number of doubles in the set -/
def num_doubles : ℕ := n + 1

/-- The probability of selecting a double -/
def prob_double : ℚ := num_doubles / total_dominoes

/-- Theorem stating the probability of selecting a double -/
theorem probability_of_double : prob_double = 13 / 91 := by sorry

end probability_of_double_l1292_129299


namespace seymours_venus_flytraps_l1292_129289

/-- Represents the plant shop inventory and fertilizer requirements --/
structure PlantShop where
  petunia_flats : ℕ
  petunias_per_flat : ℕ
  rose_flats : ℕ
  roses_per_flat : ℕ
  fertilizer_per_petunia : ℕ
  fertilizer_per_rose : ℕ
  fertilizer_per_venus_flytrap : ℕ
  total_fertilizer : ℕ

/-- Calculates the number of Venus flytraps in the shop --/
def venus_flytraps (shop : PlantShop) : ℕ :=
  let petunia_fertilizer := shop.petunia_flats * shop.petunias_per_flat * shop.fertilizer_per_petunia
  let rose_fertilizer := shop.rose_flats * shop.roses_per_flat * shop.fertilizer_per_rose
  let remaining_fertilizer := shop.total_fertilizer - petunia_fertilizer - rose_fertilizer
  remaining_fertilizer / shop.fertilizer_per_venus_flytrap

/-- Theorem stating that Seymour's shop has 2 Venus flytraps --/
theorem seymours_venus_flytraps :
  let shop : PlantShop := {
    petunia_flats := 4,
    petunias_per_flat := 8,
    rose_flats := 3,
    roses_per_flat := 6,
    fertilizer_per_petunia := 8,
    fertilizer_per_rose := 3,
    fertilizer_per_venus_flytrap := 2,
    total_fertilizer := 314
  }
  venus_flytraps shop = 2 := by
  sorry

end seymours_venus_flytraps_l1292_129289


namespace triangle_rotation_l1292_129292

/-- Triangle OAB with given properties and rotation of OA --/
theorem triangle_rotation (A : ℝ × ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (5, 0)
  let angle_ABO : ℝ := π / 2  -- 90°
  let angle_AOB : ℝ := π / 6  -- 30°
  let rotation_angle : ℝ := 2 * π / 3  -- 120°
  A.1 > 0 ∧ A.2 > 0 →  -- A is in the first quadrant
  (A.1 - O.1) * (B.2 - O.2) = (B.1 - O.1) * (A.2 - O.2) →  -- ABO is a right angle
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 
    Real.cos angle_AOB * Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) * Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2) →
  let rotated_A : ℝ × ℝ := (
    A.1 * Real.cos rotation_angle - A.2 * Real.sin rotation_angle,
    A.1 * Real.sin rotation_angle + A.2 * Real.cos rotation_angle
  )
  rotated_A = (-5/2 * (1 + Real.sqrt 3), 5/2 * (Real.sqrt 3 - 1)) :=
by sorry

end triangle_rotation_l1292_129292


namespace distinct_z_values_l1292_129205

/-- Given two integers x and y where:
    1. 200 ≤ x ≤ 999
    2. 100 ≤ y ≤ 999
    3. y is the number formed by reversing the digits of x
    4. z = x + y
    This theorem states that there are exactly 1878 distinct possible values for z. -/
theorem distinct_z_values (x y z : ℕ) 
  (hx : 200 ≤ x ∧ x ≤ 999)
  (hy : 100 ≤ y ∧ y ≤ 999)
  (hrev : y = 100 * (x % 10) + 10 * ((x / 10) % 10) + (x / 100))
  (hz : z = x + y) :
  ∃! (s : Finset ℕ), s = {z | ∃ (x y : ℕ), 
    200 ≤ x ∧ x ≤ 999 ∧
    100 ≤ y ∧ y ≤ 999 ∧
    y = 100 * (x % 10) + 10 * ((x / 10) % 10) + (x / 100) ∧
    z = x + y} ∧ 
  Finset.card s = 1878 :=
by sorry

end distinct_z_values_l1292_129205


namespace geometric_sequence_common_ratio_l1292_129265

/-- Given a geometric sequence {a_n} where a_2 = 8 and a_5 = 64, the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 2 = 8 →                     -- Given condition
  a 5 = 64 →                    -- Given condition
  q = 2 :=                      -- Conclusion to prove
by
  sorry


end geometric_sequence_common_ratio_l1292_129265


namespace burger_slices_l1292_129298

theorem burger_slices (total_burgers : ℕ) (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) (era_slices : ℕ) :
  total_burgers = 5 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  era_slices = 1 →
  (friend1_slices + friend2_slices + friend3_slices + friend4_slices + era_slices) / total_burgers = 2 :=
by sorry

end burger_slices_l1292_129298


namespace prove_weekly_savings_l1292_129248

def employee1_rate : ℝ := 20
def employee2_rate : ℝ := 22
def subsidy_rate : ℝ := 6
def hours_per_week : ℝ := 40

def weekly_savings : ℝ := (employee1_rate * hours_per_week) - ((employee2_rate - subsidy_rate) * hours_per_week)

theorem prove_weekly_savings : weekly_savings = 160 := by
  sorry

end prove_weekly_savings_l1292_129248


namespace haley_marble_distribution_l1292_129274

/-- Represents the number of marbles each boy receives when Haley's marbles are distributed equally -/
def marbles_per_boy (total_marbles : ℕ) (num_boys : ℕ) : ℕ :=
  total_marbles / num_boys

/-- Theorem stating that when 20 marbles are distributed equally between 2 boys, each boy receives 10 marbles -/
theorem haley_marble_distribution :
  marbles_per_boy 20 2 = 10 := by
  sorry

end haley_marble_distribution_l1292_129274


namespace unique_odd_natural_from_primes_l1292_129294

theorem unique_odd_natural_from_primes :
  ∃! (n : ℕ), 
    n % 2 = 1 ∧ 
    ∃ (p q : ℕ), 
      Prime p ∧ Prime q ∧ p > q ∧ 
      n = (p + q) / (p - q) :=
by
  -- The proof goes here
  sorry

end unique_odd_natural_from_primes_l1292_129294


namespace percentage_problem_l1292_129249

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * y = 16) : y = 40 := by
  sorry

end percentage_problem_l1292_129249


namespace total_pizza_cost_l1292_129211

def pizza_cost : ℕ := 8
def number_of_pizzas : ℕ := 3

theorem total_pizza_cost : pizza_cost * number_of_pizzas = 24 := by
  sorry

end total_pizza_cost_l1292_129211


namespace profit_percentage_is_25_percent_l1292_129232

/-- Calculates the profit percentage given cost price, marked price, and discount percentage. -/
def profit_percentage (cost_price marked_price discount_percent : ℚ) : ℚ :=
  let discount := (discount_percent / 100) * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that for the given conditions, the profit percentage is 25%. -/
theorem profit_percentage_is_25_percent :
  profit_percentage 95 125 5 = 25 := by
  sorry

end profit_percentage_is_25_percent_l1292_129232


namespace hyperbola_eccentricity_l1292_129256

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 where a > 0 and eccentricity is 2,
    prove that a = √6/3 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 2 = 1) →
  (∃ c : ℝ, c / a = 2) →
  a = Real.sqrt 6 / 3 := by
  sorry

end hyperbola_eccentricity_l1292_129256


namespace raven_age_is_55_l1292_129228

-- Define the current ages
def phoebe_age : ℕ := 10
def raven_age : ℕ := 55

-- Define the conditions
def condition1 : Prop := raven_age + 5 = 4 * (phoebe_age + 5)
def condition2 : Prop := phoebe_age = 10
def condition3 : Prop := ∃ sam_age : ℕ, sam_age = 2 * ((raven_age + 3) - (phoebe_age + 3))

-- Theorem statement
theorem raven_age_is_55 : 
  condition1 ∧ condition2 ∧ condition3 → raven_age = 55 :=
by sorry

end raven_age_is_55_l1292_129228


namespace min_three_colors_proof_l1292_129231

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : ℕ
  green : ℕ
  blue : ℕ
  white : ℕ

/-- The total number of balls in the box -/
def total_balls : ℕ := 111

/-- The number of balls that, when drawn, ensures getting all four colors -/
def all_colors_draw : ℕ := 100

/-- Predicate to check if a BallCounts configuration is valid -/
def valid_configuration (counts : BallCounts) : Prop :=
  counts.red + counts.green + counts.blue + counts.white = total_balls ∧
  ∀ (n : ℕ), n ≥ all_colors_draw →
    n - counts.red < all_colors_draw ∧
    n - counts.green < all_colors_draw ∧
    n - counts.blue < all_colors_draw ∧
    n - counts.white < all_colors_draw

/-- The smallest number of balls to draw to ensure at least three colors -/
def min_three_colors_draw : ℕ := 88

theorem min_three_colors_proof :
  ∀ (counts : BallCounts),
    valid_configuration counts →
    (∀ (n : ℕ), n ≥ min_three_colors_draw →
      ∃ (colors : Finset (Fin 4)),
        colors.card ≥ 3 ∧
        (∀ (i : Fin 4),
          i ∈ colors ↔
            (i = 0 ∧ n > total_balls - counts.red) ∨
            (i = 1 ∧ n > total_balls - counts.green) ∨
            (i = 2 ∧ n > total_balls - counts.blue) ∨
            (i = 3 ∧ n > total_balls - counts.white))) ∧
    (∀ (m : ℕ), m < min_three_colors_draw →
      ∃ (counts' : BallCounts),
        valid_configuration counts' ∧
        ∃ (colors : Finset (Fin 4)),
          colors.card < 3 ∧
          (∀ (i : Fin 4),
            i ∈ colors ↔
              (i = 0 ∧ m > total_balls - counts'.red) ∨
              (i = 1 ∧ m > total_balls - counts'.green) ∨
              (i = 2 ∧ m > total_balls - counts'.blue) ∨
              (i = 3 ∧ m > total_balls - counts'.white))) :=
by sorry

end min_three_colors_proof_l1292_129231


namespace vector_expression_not_equal_PQ_l1292_129287

variable {V : Type*} [AddCommGroup V]
variable (A B P Q : V)

theorem vector_expression_not_equal_PQ :
  A - B + B - P - (A - Q) ≠ P - Q :=
sorry

end vector_expression_not_equal_PQ_l1292_129287


namespace simplification_proofs_l1292_129226

theorem simplification_proofs :
  (3.5 * 101 = 353.5) ∧
  (11 * 5.9 - 5.9 = 59) ∧
  (88 - 17.5 - 12.5 = 58) := by
  sorry

end simplification_proofs_l1292_129226


namespace bret_dinner_coworkers_l1292_129241

theorem bret_dinner_coworkers :
  let main_meal_cost : ℚ := 12
  let appetizer_cost : ℚ := 6
  let num_appetizers : ℕ := 2
  let tip_percentage : ℚ := 0.2
  let rush_order_fee : ℚ := 5
  let total_spent : ℚ := 77

  let total_people (coworkers : ℕ) : ℕ := coworkers + 1
  let main_meals_cost (coworkers : ℕ) : ℚ := main_meal_cost * (total_people coworkers : ℚ)
  let appetizers_total : ℚ := appetizer_cost * (num_appetizers : ℚ)
  let subtotal (coworkers : ℕ) : ℚ := main_meals_cost coworkers + appetizers_total
  let tip (coworkers : ℕ) : ℚ := tip_percentage * subtotal coworkers
  let total_cost (coworkers : ℕ) : ℚ := subtotal coworkers + tip coworkers + rush_order_fee

  ∃ (coworkers : ℕ), total_cost coworkers = total_spent ∧ coworkers = 3 :=
by sorry

end bret_dinner_coworkers_l1292_129241


namespace average_price_per_book_l1292_129243

def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1150
def books_shop2 : ℕ := 50
def price_shop2 : ℕ := 920

theorem average_price_per_book :
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 18 := by
  sorry

end average_price_per_book_l1292_129243


namespace opposite_of_negative_five_l1292_129285

theorem opposite_of_negative_five : -((-5) : ℝ) = (5 : ℝ) := by
  sorry

end opposite_of_negative_five_l1292_129285


namespace perimeter_of_square_III_l1292_129207

/-- Given three squares I, II, and III, prove that the perimeter of III is 36. -/
theorem perimeter_of_square_III (I II III : Real) : 
  (I > 0) →  -- I is positive (side length of a square)
  (II > 0) → -- II is positive (side length of a square)
  (4 * I = 12) → -- Perimeter of I is 12
  (4 * II = 24) → -- Perimeter of II is 24
  (III = I + II) → -- Side length of III is sum of side lengths of I and II
  (4 * III = 36) := by -- Perimeter of III is 36
sorry

end perimeter_of_square_III_l1292_129207


namespace square_sum_implies_product_l1292_129202

theorem square_sum_implies_product (m : ℝ) 
  (h : (m - 2023)^2 + (2024 - m)^2 = 2025) : 
  (m - 2023) * (2024 - m) = -1012 := by
  sorry

end square_sum_implies_product_l1292_129202


namespace intersecting_circles_sum_l1292_129291

/-- Given two circles intersecting at points A and B, with their centers on a line,
    prove that m+2c equals 26 -/
theorem intersecting_circles_sum (m c : ℝ) : 
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (-6, m)
  let center_line (x y : ℝ) := x - y + c = 0
  -- Assume the centers of both circles lie on the line x - y + c = 0
  -- Assume A and B are intersection points of the two circles
  m + 2*c = 26 := by
  sorry

end intersecting_circles_sum_l1292_129291


namespace odd_function_properties_l1292_129270

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * x + m) / (x^2 + 1)

theorem odd_function_properties :
  ∃ (m : ℝ),
    (∀ x, f m x = -f m (-x)) ∧
    (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f m x < f m y) ∧
    (∀ x y, 1 ≤ x ∧ x < y → f m x > f m y) ∧
    (∀ x y, 0 ≤ x ∧ 0 ≤ y → f m x - f m y ≤ 3/2) :=
by sorry

end odd_function_properties_l1292_129270


namespace valid_numbers_l1292_129261

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  let tens := n / 10
  let ones := n % 10
  2 * (tens + ones) = tens * ones

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {63, 44, 36} := by sorry

end valid_numbers_l1292_129261


namespace solutions_based_on_discriminant_l1292_129247

/-- Represents the system of equations -/
def SystemOfEquations (a b c : ℝ) (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (a ≠ 0) ∧ 
  (∀ i ∈ Finset.range n, a * (x i)^2 + b * (x i) + c = x ((i + 1) % n))

/-- Theorem stating the number of solutions based on the discriminant -/
theorem solutions_based_on_discriminant (a b c : ℝ) (n : ℕ) :
  (a ≠ 0) ∧ (n > 0) →
  (((b - 1)^2 - 4*a*c < 0 → ¬∃ x, SystemOfEquations a b c n x) ∧
   ((b - 1)^2 - 4*a*c = 0 → ∃! x, SystemOfEquations a b c n x) ∧
   ((b - 1)^2 - 4*a*c > 0 → ∃ x y, x ≠ y ∧ SystemOfEquations a b c n x ∧ SystemOfEquations a b c n y)) :=
by sorry

end solutions_based_on_discriminant_l1292_129247


namespace quadratic_roots_l1292_129282

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end quadratic_roots_l1292_129282


namespace simplify_expression_l1292_129230

theorem simplify_expression (z : ℝ) : (5 - 2 * z^2) - (4 * z^2 - 7) = 12 - 6 * z^2 := by
  sorry

end simplify_expression_l1292_129230


namespace calorie_difference_l1292_129237

-- Define the number of squirrels and rabbits caught per hour
def squirrels_per_hour : ℕ := 6
def rabbits_per_hour : ℕ := 2

-- Define the calorie content of each animal
def calories_per_squirrel : ℕ := 300
def calories_per_rabbit : ℕ := 800

-- Define the total calories from squirrels and rabbits
def total_calories_squirrels : ℕ := squirrels_per_hour * calories_per_squirrel
def total_calories_rabbits : ℕ := rabbits_per_hour * calories_per_rabbit

-- Theorem to prove
theorem calorie_difference : total_calories_squirrels - total_calories_rabbits = 200 := by
  sorry

end calorie_difference_l1292_129237


namespace triangle_side_length_l1292_129260

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 5 →
  c = 8 →
  B = Real.pi / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  b = 7 := by sorry

end triangle_side_length_l1292_129260


namespace factorization_identity_l1292_129250

theorem factorization_identity (a b : ℝ) : a^2 - 2*a*b = a*(a - 2*b) := by
  sorry

end factorization_identity_l1292_129250


namespace best_player_hits_l1292_129293

/-- Represents a baseball team -/
structure BaseballTeam where
  totalPlayers : ℕ
  averageHitsPerGame : ℕ
  gamesPlayed : ℕ
  otherPlayersAverageHits : ℕ
  otherPlayersGames : ℕ

/-- Calculates the total hits of the best player -/
def bestPlayerTotalHits (team : BaseballTeam) : ℕ :=
  team.averageHitsPerGame * team.gamesPlayed - 
  (team.totalPlayers - 1) * team.otherPlayersAverageHits

/-- Theorem stating the best player's total hits -/
theorem best_player_hits (team : BaseballTeam) 
  (h1 : team.totalPlayers = 11)
  (h2 : team.averageHitsPerGame = 15)
  (h3 : team.gamesPlayed = 5)
  (h4 : team.otherPlayersAverageHits = 6)
  (h5 : team.otherPlayersGames = 6) :
  bestPlayerTotalHits team = 25 := by
  sorry

#eval bestPlayerTotalHits { 
  totalPlayers := 11, 
  averageHitsPerGame := 15, 
  gamesPlayed := 5, 
  otherPlayersAverageHits := 6, 
  otherPlayersGames := 6
}

end best_player_hits_l1292_129293


namespace earth_circumference_scientific_notation_l1292_129244

theorem earth_circumference_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ |a| ∧ |a| < 10 ∧
    n = 6 ∧
    4010000 = a * (10 : ℝ) ^ n :=
by sorry

end earth_circumference_scientific_notation_l1292_129244


namespace complex_sum_simplification_l1292_129275

theorem complex_sum_simplification : 
  ((-2 + Complex.I * Real.sqrt 7) / 3) ^ 4 + ((-2 - Complex.I * Real.sqrt 7) / 3) ^ 4 = 242 / 81 := by
  sorry

end complex_sum_simplification_l1292_129275


namespace largest_divisor_of_four_consecutive_integers_l1292_129253

theorem largest_divisor_of_four_consecutive_integers (n : ℕ) :
  (∀ m : ℕ, (m * (m + 1) * (m + 2) * (m + 3)) % 24 = 0) ∧
  (∃ k : ℕ, (k * (k + 1) * (k + 2) * (k + 3)) % 25 ≠ 0) :=
by sorry

end largest_divisor_of_four_consecutive_integers_l1292_129253


namespace simplify_radicals_l1292_129257

theorem simplify_radicals : 
  Real.sqrt 18 * Real.sqrt 72 - Real.sqrt 32 = 36 - 4 * Real.sqrt 2 := by
  sorry

end simplify_radicals_l1292_129257


namespace geometric_sequence_problem_l1292_129272

/-- Given a geometric sequence {aₙ}, prove that if a₇ × a₉ = 4 and a₄ = 1, then a₁₂ = 16 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∃ (r : ℝ), ∀ n, a (n + 1) = a n * r) →  -- {aₙ} is a geometric sequence
  a 7 * a 9 = 4 →                         -- a₇ × a₉ = 4
  a 4 = 1 →                               -- a₄ = 1
  a 12 = 16 :=                            -- a₁₂ = 16
by
  sorry


end geometric_sequence_problem_l1292_129272


namespace dogwood_trees_in_park_l1292_129218

theorem dogwood_trees_in_park (current_trees : ℕ) : current_trees = 34 :=
  by
  have h1 : current_trees + 49 = 83 := by sorry
  sorry

end dogwood_trees_in_park_l1292_129218


namespace cannot_reach_target_l1292_129284

/-- Represents a positive integer as a list of digits (most significant digit first) -/
def Digits := List Nat

/-- The starting number -/
def startNum : Digits := [1]

/-- The target 100-digit number -/
def targetNum : Digits := List.replicate 98 2 ++ [5, 2, 2, 2, 1]

/-- Checks if a number is valid (non-zero first digit) -/
def isValidNumber (d : Digits) : Prop := d.head? ≠ some 0

/-- Represents the operation of multiplying by 5 -/
def multiplyBy5 (d : Digits) : Digits := sorry

/-- Represents the operation of rearranging digits -/
def rearrangeDigits (d : Digits) : Digits := sorry

/-- Represents a sequence of operations -/
inductive Operation
| Multiply
| Rearrange

def applyOperation (op : Operation) (d : Digits) : Digits :=
  match op with
  | Operation.Multiply => multiplyBy5 d
  | Operation.Rearrange => rearrangeDigits d

/-- Theorem stating the impossibility of reaching the target number -/
theorem cannot_reach_target : 
  ∀ (ops : List Operation), 
    let finalNum := ops.foldl (λ acc op => applyOperation op acc) startNum
    isValidNumber finalNum → finalNum ≠ targetNum :=
by sorry

end cannot_reach_target_l1292_129284


namespace difference_of_squares_special_case_l1292_129239

theorem difference_of_squares_special_case : (2 + Real.sqrt 2) * (2 - Real.sqrt 2) = 2 := by
  sorry

end difference_of_squares_special_case_l1292_129239


namespace members_not_playing_specific_club_l1292_129219

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  both : ℕ

/-- The number of members who don't play either badminton or tennis -/
def members_not_playing (club : SportsClub) : ℕ :=
  club.total - (club.badminton + club.tennis - club.both)

/-- Theorem stating the number of members not playing either sport in the given scenario -/
theorem members_not_playing_specific_club :
  let club : SportsClub := {
    total := 30,
    badminton := 17,
    tennis := 19,
    both := 8
  }
  members_not_playing club = 2 := by
  sorry

end members_not_playing_specific_club_l1292_129219


namespace square_sum_nonzero_iff_exists_nonzero_l1292_129246

theorem square_sum_nonzero_iff_exists_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by sorry

end square_sum_nonzero_iff_exists_nonzero_l1292_129246


namespace sam_football_games_l1292_129252

theorem sam_football_games (games_this_year games_last_year : ℕ) 
  (h1 : games_this_year = 14)
  (h2 : games_last_year = 29) :
  games_this_year + games_last_year = 43 := by
  sorry

end sam_football_games_l1292_129252


namespace mushroom_collectors_l1292_129204

theorem mushroom_collectors (n : ℕ) : 
  (n^2 + 9*n - 2) % (n + 11) = 0 → n < 11 := by
  sorry

end mushroom_collectors_l1292_129204


namespace hydrogen_mass_percentage_l1292_129209

/-- Molecular weight of C3H6O in g/mol -/
def mw_C3H6O : ℝ := 58.09

/-- Molecular weight of NH3 in g/mol -/
def mw_NH3 : ℝ := 17.04

/-- Molecular weight of H2SO4 in g/mol -/
def mw_H2SO4 : ℝ := 98.09

/-- Mass of hydrogen in one mole of C3H6O in g -/
def mass_H_in_C3H6O : ℝ := 6.06

/-- Mass of hydrogen in one mole of NH3 in g -/
def mass_H_in_NH3 : ℝ := 3.03

/-- Mass of hydrogen in one mole of H2SO4 in g -/
def mass_H_in_H2SO4 : ℝ := 2.02

/-- Number of moles of C3H6O in the mixture -/
def moles_C3H6O : ℝ := 3

/-- Number of moles of NH3 in the mixture -/
def moles_NH3 : ℝ := 2

/-- Number of moles of H2SO4 in the mixture -/
def moles_H2SO4 : ℝ := 1

/-- Theorem stating that the mass percentage of hydrogen in the given mixture is approximately 8.57% -/
theorem hydrogen_mass_percentage :
  let total_mass_H := moles_C3H6O * mass_H_in_C3H6O + moles_NH3 * mass_H_in_NH3 + moles_H2SO4 * mass_H_in_H2SO4
  let total_mass_mixture := moles_C3H6O * mw_C3H6O + moles_NH3 * mw_NH3 + moles_H2SO4 * mw_H2SO4
  let mass_percentage_H := (total_mass_H / total_mass_mixture) * 100
  ∃ ε > 0, |mass_percentage_H - 8.57| < ε :=
by
  sorry

end hydrogen_mass_percentage_l1292_129209


namespace power_difference_l1292_129208

theorem power_difference (a m n : ℝ) (hm : a^m = 6) (hn : a^n = 2) : a^(m-n) = 3 := by
  sorry

end power_difference_l1292_129208


namespace hansel_raise_percentage_l1292_129221

/-- Proves that Hansel's raise percentage is 10% given the problem conditions --/
theorem hansel_raise_percentage : 
  ∀ (hansel_initial gretel_initial hansel_final gretel_final gretel_raise hansel_raise : ℝ),
  hansel_initial = 30000 →
  gretel_initial = 30000 →
  gretel_raise = 0.15 →
  gretel_final = gretel_initial * (1 + gretel_raise) →
  hansel_final = gretel_final - 1500 →
  hansel_raise = (hansel_final - hansel_initial) / hansel_initial →
  hansel_raise = 0.1 := by
sorry


end hansel_raise_percentage_l1292_129221


namespace percentage_equality_theorem_l1292_129233

theorem percentage_equality_theorem (x : ℚ) : 
  (30 : ℚ) / 100 * x = (25 : ℚ) / 100 * 40 → x = 100 / 3 := by
  sorry

end percentage_equality_theorem_l1292_129233


namespace problem_solution_l1292_129210

-- Define the function f(x) = |x-a| + 3x
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

theorem problem_solution :
  (∀ x : ℝ, f 1 x > 3 * x + 2 ↔ (x > 3 ∨ x < -1)) ∧
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
by sorry

end problem_solution_l1292_129210


namespace school_dance_attendance_l1292_129201

theorem school_dance_attendance (P : ℕ) : 
  (P * 10 / 100 = P / 10) →  -- 10% of P are faculty and staff
  (P * 90 / 100 = P * 9 / 10) →  -- 90% of P are students
  ((P * 9 / 10) * 2 / 3 = (P * 9 / 10) - 30) →  -- Two-thirds of students are girls
  ((P * 9 / 10) * 1 / 3 = 30) →  -- One-third of students are boys
  P = 100 := by sorry

end school_dance_attendance_l1292_129201


namespace f_pi_third_is_nonnegative_reals_l1292_129245

-- Define the set f(x)
def f (φ : Real) : Set Real :=
  {x : Real | x ≥ 0}

-- Theorem statement
theorem f_pi_third_is_nonnegative_reals :
  f (π / 3) = {x : Real | x ≥ 0} := by
  sorry

end f_pi_third_is_nonnegative_reals_l1292_129245


namespace scavenger_hunt_theorem_l1292_129251

/-- Represents the number of choices for each day of the scavenger hunt --/
def scavenger_hunt_choices : List Nat := [1, 2, 4, 3, 1]

/-- The total number of combinations for the scavenger hunt --/
def total_combinations : Nat := scavenger_hunt_choices.prod

theorem scavenger_hunt_theorem :
  total_combinations = 24 := by
  sorry

end scavenger_hunt_theorem_l1292_129251


namespace probability_under_20_is_7_16_l1292_129262

/-- Represents a group of people with age categories --/
structure AgeGroup where
  total : ℕ
  over30 : ℕ
  under20 : ℕ
  h1 : over30 + under20 = total

/-- The probability of selecting a person under 20 years old --/
def probabilityUnder20 (group : AgeGroup) : ℚ :=
  group.under20 / group.total

theorem probability_under_20_is_7_16 (group : AgeGroup) 
  (h2 : group.total = 160) 
  (h3 : group.over30 = 90) : 
  probabilityUnder20 group = 7 / 16 := by
  sorry

#check probability_under_20_is_7_16

end probability_under_20_is_7_16_l1292_129262


namespace probability_monotonic_increasing_l1292_129206

def cube_faces : Finset ℤ := {-2, -1, 0, 1, 2, 3}

def is_monotonic_increasing (a b : ℤ) : Prop :=
  a ≥ 0 ∧ b ≥ 0

def favorable_outcomes : Finset (ℤ × ℤ) :=
  (cube_faces.filter (λ x => x ≥ 0)).product (cube_faces.filter (λ x => x ≥ 0))

def total_outcomes : Finset (ℤ × ℤ) :=
  cube_faces.product cube_faces

theorem probability_monotonic_increasing :
  (favorable_outcomes.card : ℚ) / (total_outcomes.card : ℚ) = 4 / 9 := by
  sorry

end probability_monotonic_increasing_l1292_129206


namespace missing_number_proof_l1292_129297

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + 42 + 78 + y) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  y = 104 := by
  sorry

end missing_number_proof_l1292_129297


namespace not_third_PSU_l1292_129266

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the ordering relation for runners
def beats (a b : Runner) : Prop := sorry

-- Define the conditions
axiom P_beats_Q : beats Runner.P Runner.Q
axiom Q_beats_R : beats Runner.Q Runner.R
axiom T_beats_S : beats Runner.T Runner.S
axiom T_beats_U : beats Runner.T Runner.U
axiom U_after_P_before_Q : beats Runner.P Runner.U ∧ beats Runner.U Runner.Q

-- Define what it means to finish third
def finishes_third (r : Runner) : Prop := sorry

-- Theorem statement
theorem not_third_PSU : 
  ¬(finishes_third Runner.P) ∧ 
  ¬(finishes_third Runner.S) ∧ 
  ¬(finishes_third Runner.U) := by sorry

end not_third_PSU_l1292_129266


namespace tech_club_enrollment_l1292_129281

theorem tech_club_enrollment (total : ℕ) (cs : ℕ) (robotics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 90)
  (h3 : robotics = 70)
  (h4 : both = 20) :
  total - (cs + robotics - both) = 10 := by
  sorry

end tech_club_enrollment_l1292_129281


namespace solve_for_a_l1292_129236

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^2

-- Theorem statement
theorem solve_for_a : ∃ a : ℝ, star a 3 = 15 ∧ a = 12 := by
  sorry

end solve_for_a_l1292_129236


namespace carls_garden_area_l1292_129286

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_separation : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the garden --/
def garden_area (g : Garden) : ℕ :=
  let longer_side_posts := 2 * g.shorter_side_posts
  let shorter_side_length := (g.shorter_side_posts - 1) * g.post_separation
  let longer_side_length := (longer_side_posts - 1) * g.post_separation
  shorter_side_length * longer_side_length

/-- Theorem stating that Carl's garden has an area of 900 square yards --/
theorem carls_garden_area :
  ∀ (g : Garden),
    g.total_posts = 26 ∧
    g.post_separation = 5 ∧
    g.shorter_side_posts = 5 →
    garden_area g = 900 := by
  sorry

end carls_garden_area_l1292_129286


namespace positive_solution_x_l1292_129200

theorem positive_solution_x (x y z : ℝ) : 
  x * y = 8 - 2 * x - 3 * y →
  y * z = 8 - 4 * y - 2 * z →
  x * z = 40 - 5 * x - 3 * z →
  x > 0 →
  x = 10 := by
sorry

end positive_solution_x_l1292_129200


namespace james_total_points_l1292_129220

/-- Quiz Bowl Scoring System -/
structure QuizBowl where
  correct_points : ℕ := 2
  incorrect_penalty : ℕ := 1
  quick_answer_bonus : ℕ := 1
  rounds : ℕ := 5
  questions_per_round : ℕ := 5

/-- James' Performance -/
structure Performance where
  correct_answers : ℕ
  missed_questions : ℕ
  quick_answers : ℕ

/-- Calculate total points for a given performance in the quiz bowl -/
def calculate_points (qb : QuizBowl) (perf : Performance) : ℕ :=
  qb.correct_points * perf.correct_answers + qb.quick_answer_bonus * perf.quick_answers

/-- Theorem: James' total points in the quiz bowl -/
theorem james_total_points (qb : QuizBowl) (james : Performance) 
  (h1 : james.correct_answers = qb.rounds * qb.questions_per_round - james.missed_questions)
  (h2 : james.missed_questions = 1)
  (h3 : james.quick_answers = 4) :
  calculate_points qb james = 52 := by
  sorry

end james_total_points_l1292_129220


namespace line_parallel_to_skew_line_l1292_129279

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relations between lines
variable (parallel skew intersecting : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_to_skew_line
  (l1 l2 l3 : Line)
  (h1 : skew l1 l2)
  (h2 : parallel l3 l1) :
  intersecting l3 l2 ∨ skew l3 l2 :=
sorry

end line_parallel_to_skew_line_l1292_129279


namespace triangle_area_l1292_129214

theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5 * k, 12 * k, 13 * k) → k > 0) 
  (h_perimeter : a + b + c = 60) : (a * b : ℝ) / 2 = 120 := by
  sorry

end triangle_area_l1292_129214


namespace duplicated_chromosome_configuration_l1292_129283

/-- Represents a duplicated chromosome -/
structure DuplicatedChromosome where
  centromeres : ℕ
  chromatids : ℕ
  dna_molecules : ℕ

/-- The correct configuration of a duplicated chromosome -/
def correct_configuration : DuplicatedChromosome :=
  { centromeres := 1
  , chromatids := 2
  , dna_molecules := 2 }

/-- Theorem stating that a duplicated chromosome has the correct configuration -/
theorem duplicated_chromosome_configuration :
  ∀ (dc : DuplicatedChromosome), dc = correct_configuration :=
by sorry

end duplicated_chromosome_configuration_l1292_129283


namespace kelly_has_8_students_l1292_129280

/-- Represents the number of students in Kelly's class -/
def num_students : ℕ := sorry

/-- Represents the number of construction paper pieces needed per student -/
def paper_per_student : ℕ := 3

/-- Represents the number of glue bottles Kelly bought -/
def glue_bottles : ℕ := 6

/-- Represents the number of additional construction paper pieces Kelly bought -/
def additional_paper : ℕ := 5

/-- Represents the total number of supplies Kelly has left -/
def total_supplies : ℕ := 20

/-- Theorem stating that Kelly has 8 students given the conditions -/
theorem kelly_has_8_students :
  (((num_students * paper_per_student + glue_bottles) / 2 + additional_paper) = total_supplies) →
  num_students = 8 := by
  sorry

end kelly_has_8_students_l1292_129280


namespace new_person_weight_l1292_129278

/-- Given a group of 8 people where one person weighing 66 kg is replaced by a new person,
    if the average weight of the group increases by 2.5 kg,
    then the weight of the new person is 86 kg. -/
theorem new_person_weight (initial_group_size : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_group_size = 8 →
  weight_increase = 2.5 →
  replaced_weight = 66 →
  (initial_group_size : ℝ) * weight_increase + replaced_weight = 86 :=
by sorry

end new_person_weight_l1292_129278


namespace triangle_numbers_exist_l1292_129212

theorem triangle_numbers_exist : 
  ∃ (a b c d e f g : ℕ), 
    (b = c * d) ∧ 
    (e - f = a + c * d - a * c) ∧ 
    (e - f = g) ∧ 
    (g = a + d) ∧ 
    (c > 0) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) := by
  sorry

end triangle_numbers_exist_l1292_129212


namespace extremum_implies_f2_value_l1292_129240

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f2_value (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 11 ∨ f a b 2 = 18 := by
  sorry

end extremum_implies_f2_value_l1292_129240


namespace punch_machine_settings_l1292_129203

/-- Represents a punching pattern for a 9-field ticket -/
def PunchingPattern := Fin 9 → Bool

/-- Checks if a punching pattern is symmetric when reversed -/
def is_symmetric (p : PunchingPattern) : Prop :=
  ∀ i : Fin 9, p i = p (8 - i)

/-- The total number of possible punching patterns -/
def total_patterns : ℕ := 2^9

/-- The number of symmetric punching patterns -/
def symmetric_patterns : ℕ := 2^6

/-- The number of valid punching patterns (different when reversed) -/
def valid_patterns : ℕ := total_patterns - symmetric_patterns

theorem punch_machine_settings :
  valid_patterns = 448 :=
sorry

end punch_machine_settings_l1292_129203


namespace not_divisible_by_6_and_11_l1292_129215

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  (n - 1) - (n - 1) / a - (n - 1) / b + (n - 1) / (a * b)

theorem not_divisible_by_6_and_11 :
  count_not_divisible 1500 6 11 = 1136 := by
sorry

end not_divisible_by_6_and_11_l1292_129215


namespace tangent_line_sum_range_l1292_129224

theorem tangent_line_sum_range (m n : ℝ) :
  (∀ x y : ℝ, m * x + n * y - 2 = 0 → x^2 + y^2 ≠ 1) ∧
  (∃ x y : ℝ, m * x + n * y - 2 = 0 ∧ x^2 + y^2 = 1) →
  -2 * Real.sqrt 2 ≤ m + n ∧ m + n ≤ 2 * Real.sqrt 2 :=
sorry

end tangent_line_sum_range_l1292_129224


namespace penalty_kick_probability_l1292_129290

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem penalty_kick_probability :
  let n : ℕ := 5
  let k : ℕ := 3
  let p : ℝ := 0.05
  abs (binomial_probability n k p - 0.00113) < 0.000001 := by
  sorry

end penalty_kick_probability_l1292_129290


namespace greater_l_conference_teams_l1292_129271

/-- The number of teams in the GREATER L conference -/
def n : ℕ := sorry

/-- The total number of games played in the season -/
def total_games : ℕ := 90

/-- The formula for the total number of games when each team plays every other team twice -/
def games_formula (x : ℕ) : ℕ := x * (x - 1)

theorem greater_l_conference_teams :
  n = 10 ∧ games_formula n = total_games := by sorry

end greater_l_conference_teams_l1292_129271


namespace intersection_coordinate_sum_l1292_129235

/-- Given a triangle ABC with vertices A(2,8), B(2,2), C(10,2), 
    D is the midpoint of AB, E is the midpoint of BC, 
    and F is the intersection point of AE and CD. -/
theorem intersection_coordinate_sum (A B C D E F : ℝ × ℝ) : 
  A = (2, 8) → B = (2, 2) → C = (10, 2) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (F.2 - A.2) / (F.1 - A.1) = (E.2 - A.2) / (E.1 - A.1) →
  (F.2 - C.2) / (F.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  F.1 + F.2 = 13 := by
sorry

end intersection_coordinate_sum_l1292_129235


namespace product_difference_theorem_l1292_129223

theorem product_difference_theorem (number value : ℕ) (h1 : number = 15) (h2 : value = 13) :
  number * value - number = 180 := by
  sorry

end product_difference_theorem_l1292_129223


namespace inequality_proof_l1292_129277

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h : (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2*a + b + c) * (2*b + c + d) * (2*c + d + a) * (2*d + a + b) * (a*b*c*d)^2 ≤ 1/16 := by
  sorry

end inequality_proof_l1292_129277


namespace num_boys_is_three_l1292_129242

/-- The number of boys sitting at the table -/
def num_boys : ℕ := sorry

/-- The number of girls sitting at the table -/
def num_girls : ℕ := 5

/-- The total number of buns on the plate -/
def total_buns : ℕ := 30

/-- The number of buns given by girls to boys they know -/
def buns_girls_to_boys : ℕ := num_girls * num_boys

/-- The number of buns given by boys to girls they don't know -/
def buns_boys_to_girls : ℕ := num_boys * num_girls

/-- Theorem stating that the number of boys is 3 -/
theorem num_boys_is_three : num_boys = 3 :=
  by
    have h1 : buns_girls_to_boys + buns_boys_to_girls = total_buns := sorry
    have h2 : num_girls * num_boys + num_boys * num_girls = total_buns := sorry
    have h3 : 2 * (num_girls * num_boys) = total_buns := sorry
    have h4 : 2 * (5 * num_boys) = 30 := sorry
    have h5 : 10 * num_boys = 30 := sorry
    sorry

end num_boys_is_three_l1292_129242


namespace weekday_hourly_brew_l1292_129276

/-- Represents a coffee shop's brewing schedule and output -/
structure CoffeeShop where
  weekdayHourlyBrew : ℕ
  weekendTotalBrew : ℕ
  dailyHours : ℕ
  weeklyTotalBrew : ℕ

/-- Theorem stating the number of coffee cups brewed per hour on a weekday -/
theorem weekday_hourly_brew (shop : CoffeeShop) 
  (h1 : shop.dailyHours = 5)
  (h2 : shop.weekendTotalBrew = 120)
  (h3 : shop.weeklyTotalBrew = 370) :
  shop.weekdayHourlyBrew = 10 := by
  sorry

end weekday_hourly_brew_l1292_129276


namespace quadratic_equation_solution_l1292_129238

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

end quadratic_equation_solution_l1292_129238


namespace prism_volume_l1292_129273

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (x y z : ℝ) 
  (h₁ : x * y = 20)  -- side face area
  (h₂ : y * z = 12)  -- front face area
  (h₃ : x * z = 8)   -- bottom face area
  : x * y * z = 8 * Real.sqrt 15 := by
  sorry

end prism_volume_l1292_129273
