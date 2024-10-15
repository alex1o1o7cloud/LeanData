import Mathlib

namespace NUMINAMATH_CALUDE_sarah_initial_cupcakes_l2319_231985

theorem sarah_initial_cupcakes :
  ∀ (initial_cupcakes : ℕ),
    (initial_cupcakes / 3 : ℚ) + 5 = 11 - (2 * initial_cupcakes / 3 : ℚ) →
    initial_cupcakes = 9 := by
  sorry

end NUMINAMATH_CALUDE_sarah_initial_cupcakes_l2319_231985


namespace NUMINAMATH_CALUDE_problem_solution_l2319_231991

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -15)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 6) :
  b / (a + b) + c / (b + c) + a / (c + a) = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2319_231991


namespace NUMINAMATH_CALUDE_only_cri_du_chat_chromosomal_variation_l2319_231955

-- Define the types of genetic causes
inductive GeneticCause
| GeneMutation
| ChromosomalVariation

-- Define the genetic diseases
inductive GeneticDisease
| Albinism
| Hemophilia
| CriDuChatSyndrome
| SickleCellAnemia

-- Define a function that assigns a cause to each disease
def diseaseCause : GeneticDisease → GeneticCause
| GeneticDisease.Albinism => GeneticCause.GeneMutation
| GeneticDisease.Hemophilia => GeneticCause.GeneMutation
| GeneticDisease.CriDuChatSyndrome => GeneticCause.ChromosomalVariation
| GeneticDisease.SickleCellAnemia => GeneticCause.GeneMutation

-- Theorem stating that only Cri-du-chat syndrome is caused by chromosomal variation
theorem only_cri_du_chat_chromosomal_variation :
  ∀ (d : GeneticDisease), diseaseCause d = GeneticCause.ChromosomalVariation ↔ d = GeneticDisease.CriDuChatSyndrome :=
by sorry


end NUMINAMATH_CALUDE_only_cri_du_chat_chromosomal_variation_l2319_231955


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_two_l2319_231927

-- Define the hyperbola equation
def hyperbola_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote equations
def asymptote_equations (x y : ℝ) : Prop :=
  (3*x + 2*y = 0) ∧ (3*x - 2*y = 0)

theorem hyperbola_asymptote_implies_a_equals_two :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, hyperbola_equation x y a → asymptote_equations x y) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_equals_two_l2319_231927


namespace NUMINAMATH_CALUDE_comparison_inequality_l2319_231962

theorem comparison_inequality (h1 : 0.83 > 0.73) 
  (h2 : Real.log 0.4 / Real.log 0.5 > Real.log 0.6 / Real.log 0.5)
  (h3 : Real.log 1.6 > Real.log 1.4) : 
  0.75 - 0.1 > 0.75 * 0.1 := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequality_l2319_231962


namespace NUMINAMATH_CALUDE_solve_equation_l2319_231975

theorem solve_equation : ∃ y : ℝ, (y - 3)^4 = (1/16)⁻¹ ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2319_231975


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2319_231993

/-- Given a function f: ℝ → ℝ with a tangent line y=-x+8 at x=5, prove f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x, f 5 + (deriv f 5) * (x - 5) = -x + 8) : 
  f 5 + deriv f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2319_231993


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2319_231994

theorem quadratic_root_condition (a b : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x - 1 = 0 ∧ a * y^2 + b * y - 1 = 0) →
  (∃ r : ℝ, 1 < r ∧ r < 2 ∧ a * r^2 + b * r - 1 = 0) →
  ∀ z : ℝ, z > -1 → ∃ a' b' : ℝ, a' - b' = z ∧ 
    a' > 0 ∧
    (∃ x y : ℝ, x ≠ y ∧ a' * x^2 + b' * x - 1 = 0 ∧ a' * y^2 + b' * y - 1 = 0) ∧
    (∃ r : ℝ, 1 < r ∧ r < 2 ∧ a' * r^2 + b' * r - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2319_231994


namespace NUMINAMATH_CALUDE_number_of_men_is_correct_l2319_231903

/-- The number of men in a group where:
  1. The average age of the group increases by 2 years when two men are replaced.
  2. The ages of the two men being replaced are 21 and 23 years.
  3. The average age of the two new men is 37 years. -/
def number_of_men : ℕ :=
  let age_increase : ℕ := 2
  let replaced_men_ages : Fin 2 → ℕ := ![21, 23]
  let new_men_average_age : ℕ := 37
  15

theorem number_of_men_is_correct :
  let age_increase : ℕ := 2
  let replaced_men_ages : Fin 2 → ℕ := ![21, 23]
  let new_men_average_age : ℕ := 37
  number_of_men = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_is_correct_l2319_231903


namespace NUMINAMATH_CALUDE_abs_a_b_sum_l2319_231947

theorem abs_a_b_sum (a b : ℝ) (ha : |a| = 7) (hb : |b| = 3) (hab : a * b > 0) :
  a + b = 10 ∨ a + b = -10 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_b_sum_l2319_231947


namespace NUMINAMATH_CALUDE_tangent_line_equation_f_decreasing_intervals_l2319_231900

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for the equation of the tangent line
theorem tangent_line_equation :
  ∀ x y : ℝ, y = f x → (x = 0 → 9*x - y - 2 = 0) :=
sorry

-- Theorem for the intervals where f is decreasing
theorem f_decreasing_intervals :
  ∀ x : ℝ, (x < -1 ∨ x > 3) → (f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_f_decreasing_intervals_l2319_231900


namespace NUMINAMATH_CALUDE_largest_reciprocal_l2319_231930

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/7 → c = 2 → d = 7 → e = 1000 →
  (1/a > 1/b) ∧ (1/a > 1/c) ∧ (1/a > 1/d) ∧ (1/a > 1/e) :=
by
  sorry

#check largest_reciprocal

end NUMINAMATH_CALUDE_largest_reciprocal_l2319_231930


namespace NUMINAMATH_CALUDE_fourth_power_product_l2319_231982

theorem fourth_power_product : 
  (((2^4 - 1) / (2^4 + 1)) * ((3^4 - 1) / (3^4 + 1)) * 
   ((4^4 - 1) / (4^4 + 1)) * ((5^4 - 1) / (5^4 + 1))) = 432 / 1105 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_product_l2319_231982


namespace NUMINAMATH_CALUDE_cost_of_beads_per_bracelet_l2319_231992

/-- Proves the cost of beads per bracelet given the selling price, string cost, number of bracelets sold, and total profit -/
theorem cost_of_beads_per_bracelet 
  (selling_price : ℝ)
  (string_cost : ℝ)
  (bracelets_sold : ℕ)
  (total_profit : ℝ)
  (h1 : selling_price = 6)
  (h2 : string_cost = 1)
  (h3 : bracelets_sold = 25)
  (h4 : total_profit = 50) :
  let bead_cost := (bracelets_sold : ℝ) * selling_price - total_profit - bracelets_sold * string_cost
  bead_cost / (bracelets_sold : ℝ) = 3 := by
sorry

end NUMINAMATH_CALUDE_cost_of_beads_per_bracelet_l2319_231992


namespace NUMINAMATH_CALUDE_cube_root_difference_l2319_231983

theorem cube_root_difference (a b : ℝ) 
  (h1 : (a ^ (1/3) : ℝ) - (b ^ (1/3) : ℝ) = 12)
  (h2 : a * b = ((a + b + 8) / 6) ^ 3) :
  a - b = 468 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_difference_l2319_231983


namespace NUMINAMATH_CALUDE_determinant_value_l2319_231997

-- Define the operation
def determinant (a b c d : ℚ) : ℚ := a * d - b * c

-- Define the theorem
theorem determinant_value :
  let a : ℚ := -(1^2)
  let b : ℚ := (-2)^2 - 1
  let c : ℚ := -(3^2) + 5
  let d : ℚ := (3/4) / (-1/4)
  determinant a b c d = 15 := by
  sorry

end NUMINAMATH_CALUDE_determinant_value_l2319_231997


namespace NUMINAMATH_CALUDE_product_ratio_equals_one_l2319_231972

theorem product_ratio_equals_one
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_equals_one_l2319_231972


namespace NUMINAMATH_CALUDE_kevin_run_distance_l2319_231934

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Kevin's run with three segments -/
structure KevinRun where
  flat_speed : ℝ
  flat_time : ℝ
  uphill_speed : ℝ
  uphill_time : ℝ
  downhill_speed : ℝ
  downhill_time : ℝ

/-- Calculates the total distance of Kevin's run -/
def total_distance (run : KevinRun) : ℝ :=
  distance run.flat_speed run.flat_time +
  distance run.uphill_speed run.uphill_time +
  distance run.downhill_speed run.downhill_time

/-- Theorem stating that Kevin's total run distance is 17 miles -/
theorem kevin_run_distance :
  let run : KevinRun := {
    flat_speed := 10,
    flat_time := 0.5,
    uphill_speed := 20,
    uphill_time := 0.5,
    downhill_speed := 8,
    downhill_time := 0.25
  }
  total_distance run = 17 := by sorry

end NUMINAMATH_CALUDE_kevin_run_distance_l2319_231934


namespace NUMINAMATH_CALUDE_sasha_remainder_l2319_231976

theorem sasha_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  b < 102 ∧ 
  d < 103 ∧ 
  a + d = 20 → 
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_sasha_remainder_l2319_231976


namespace NUMINAMATH_CALUDE_binomial_expected_value_theorem_l2319_231966

/-- A random variable following a Binomial distribution -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  prob : ℝ
  property : prob = p

/-- The expected value of a Binomial distribution -/
def expectedValue (ξ : BinomialDistribution n p) : ℝ := n * p

theorem binomial_expected_value_theorem (ξ : BinomialDistribution 18 p) 
  (h : expectedValue ξ = 9) : p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expected_value_theorem_l2319_231966


namespace NUMINAMATH_CALUDE_max_ratio_squared_l2319_231924

theorem max_ratio_squared (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 - 2*b^2 = 0)
  (h2 : a^2 + y^2 = b^2 + x^2)
  (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
  (h4 : 0 ≤ x ∧ x < a)
  (h5 : 0 ≤ y ∧ y < b)
  (h6 : x^2 + y^2 = b^2) :
  ∃ (ρ : ℝ), ρ^2 = 2 ∧ ∀ (c : ℝ), (c = a / b → c^2 ≤ ρ^2) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l2319_231924


namespace NUMINAMATH_CALUDE_numeral_system_base_proof_l2319_231909

theorem numeral_system_base_proof (x : ℕ) : 
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2 → x = 7 := by
sorry

end NUMINAMATH_CALUDE_numeral_system_base_proof_l2319_231909


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2319_231939

/-- Given two 2D vectors a and b, where a = (1,2) and b = (-1,x), 
    if a is parallel to b, then the magnitude of b is √5. -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-1, x]
  (∃ (k : ℝ), ∀ i, b i = k * a i) →
  Real.sqrt ((b 0)^2 + (b 1)^2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2319_231939


namespace NUMINAMATH_CALUDE_fruit_store_inventory_l2319_231956

theorem fruit_store_inventory (initial_amount : ℚ) : 
  initial_amount - 3/10 + 2/5 = 19/20 → initial_amount = 17/20 := by
  sorry

end NUMINAMATH_CALUDE_fruit_store_inventory_l2319_231956


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2319_231961

theorem cos_120_degrees : Real.cos (120 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2319_231961


namespace NUMINAMATH_CALUDE_consecutive_non_primes_under_50_l2319_231942

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem consecutive_non_primes_under_50 :
  ∃ (a b c d e : ℕ),
    a < 50 ∧ b < 50 ∧ c < 50 ∧ d < 50 ∧ e < 50 ∧
    ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ ¬(is_prime d) ∧ ¬(is_prime e) ∧
    b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
    e = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_under_50_l2319_231942


namespace NUMINAMATH_CALUDE_dinos_money_theorem_l2319_231969

/-- Calculates Dino's remaining money after expenses given his work hours and rates. -/
def dinos_remaining_money (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem stating that Dino's remaining money at the end of the month is $500. -/
theorem dinos_money_theorem : 
  dinos_remaining_money 20 30 5 10 20 40 500 = 500 := by
  sorry

#eval dinos_remaining_money 20 30 5 10 20 40 500

end NUMINAMATH_CALUDE_dinos_money_theorem_l2319_231969


namespace NUMINAMATH_CALUDE_min_value_theorem_l2319_231958

theorem min_value_theorem (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : a * b = 1/4) :
  ∃ (min_val : ℝ), min_val = 4 + 4 * Real.sqrt 2 / 3 ∧
    ∀ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 →
      1 / (1 - x) + 2 / (1 - y) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2319_231958


namespace NUMINAMATH_CALUDE_impossibleTransformation_l2319_231922

-- Define the type for a card
def Card := ℤ × ℤ

-- Define the operations of the machines
def machine1 (c : Card) : Card :=
  (c.1 + 1, c.2 + 1)

def machine2 (c : Card) : Option Card :=
  if c.1 % 2 = 0 ∧ c.2 % 2 = 0 then some (c.1 / 2, c.2 / 2) else none

def machine3 (c1 c2 : Card) : Option Card :=
  if c1.2 = c2.1 then some (c1.1, c2.2) else none

-- Define the property that the difference is divisible by 7
def diffDivisibleBy7 (c : Card) : Prop :=
  (c.1 - c.2) % 7 = 0

-- Theorem stating the impossibility of the transformation
theorem impossibleTransformation :
  ∀ (c : Card), diffDivisibleBy7 c →
  ¬∃ (sequence : List (Card → Card)), 
    (List.foldl (λ acc f => f acc) c sequence = (1, 1988)) :=
by sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l2319_231922


namespace NUMINAMATH_CALUDE_number_line_steps_l2319_231919

/-- Given a number line with equally spaced markings, where 35 is reached in 7 steps from 0,
    prove that after 5 steps, the number reached is 25. -/
theorem number_line_steps (total_distance : ℝ) (total_steps : ℕ) (target_steps : ℕ) : 
  total_distance = 35 ∧ total_steps = 7 ∧ target_steps = 5 → 
  (total_distance / total_steps) * target_steps = 25 := by
  sorry

#check number_line_steps

end NUMINAMATH_CALUDE_number_line_steps_l2319_231919


namespace NUMINAMATH_CALUDE_volume_ratio_is_19_89_l2319_231907

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  origin : Point3D
  edge_length : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  point1 : Point3D
  point2 : Point3D
  point3 : Point3D

def cube : Cube := {
  origin := { x := 0, y := 0, z := 0 }
  edge_length := 6
}

def point_A : Point3D := { x := 0, y := 0, z := 0 }
def point_H : Point3D := { x := 6, y := 6, z := 2 }
def point_F : Point3D := { x := 6, y := 6, z := 3 }

def cutting_plane : Plane := {
  point1 := point_A
  point2 := point_H
  point3 := point_F
}

/-- Calculates the volume of a part of the cube cut by the plane -/
def volume_of_part (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem: The volume ratio of the two parts is 19:89 -/
theorem volume_ratio_is_19_89 (c : Cube) (p : Plane) : 
  let v1 := volume_of_part c p
  let v2 := c.edge_length ^ 3 - v1
  v1 / v2 = 19 / 89 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_is_19_89_l2319_231907


namespace NUMINAMATH_CALUDE_sum_a_n_1_to_1499_l2319_231906

def a_n (n : ℕ) : ℕ :=
  if n < 1500 then
    if n % 10 = 0 ∧ n % 15 = 0 then 15
    else if n % 15 = 0 ∧ n % 12 = 0 then 10
    else if n % 12 = 0 ∧ n % 10 = 0 then 12
    else 0
  else 0

theorem sum_a_n_1_to_1499 :
  (Finset.range 1499).sum a_n = 1263 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_n_1_to_1499_l2319_231906


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l2319_231979

theorem least_seven_digit_binary : ∀ n : ℕ, n > 0 → (
  (64 ≤ n ∧ (Nat.log 2 n).succ = 7) ↔ 
  (∀ m : ℕ, m > 0 ∧ m < n → (Nat.log 2 m).succ < 7)
) := by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l2319_231979


namespace NUMINAMATH_CALUDE_everton_college_order_l2319_231978

/-- The number of scientific calculators ordered by Everton college -/
def scientific_calculators : ℕ := 20

/-- The number of graphing calculators ordered by Everton college -/
def graphing_calculators : ℕ := 45 - scientific_calculators

/-- The cost of a single scientific calculator -/
def scientific_calculator_cost : ℕ := 10

/-- The cost of a single graphing calculator -/
def graphing_calculator_cost : ℕ := 57

/-- The total cost of the order -/
def total_cost : ℕ := 1625

/-- The total number of calculators ordered -/
def total_calculators : ℕ := 45

theorem everton_college_order :
  scientific_calculators * scientific_calculator_cost +
  graphing_calculators * graphing_calculator_cost = total_cost ∧
  scientific_calculators + graphing_calculators = total_calculators :=
sorry

end NUMINAMATH_CALUDE_everton_college_order_l2319_231978


namespace NUMINAMATH_CALUDE_rectangle_garden_length_l2319_231987

/-- Represents the perimeter of a rectangle. -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Proves that a rectangular garden with perimeter 950 m and breadth 100 m has a length of 375 m. -/
theorem rectangle_garden_length :
  ∀ (length : ℝ),
  perimeter length 100 = 950 →
  length = 375 := by
sorry

end NUMINAMATH_CALUDE_rectangle_garden_length_l2319_231987


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l2319_231912

def total_balls : ℕ := 25
def white_balls : ℕ := 10
def black_balls : ℕ := 15
def drawn_balls : ℕ := 4

theorem probability_four_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 3 / 181 :=
sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l2319_231912


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2319_231921

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ+), Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (n.val + 1) / 2 :=
by
  -- The unique solution is n = 7
  use 7
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2319_231921


namespace NUMINAMATH_CALUDE_circleplus_properties_l2319_231953

-- Define the ⊕ operation
def circleplus (x y : ℚ) : ℚ := x * y + 1

-- Theorem statement
theorem circleplus_properties :
  (circleplus 2 4 = 9) ∧
  (∀ x : ℚ, circleplus 3 (2*x - 1) = 4 → x = 1) := by
sorry

end NUMINAMATH_CALUDE_circleplus_properties_l2319_231953


namespace NUMINAMATH_CALUDE_mike_ride_length_l2319_231911

/-- Represents the taxi fare structure and trip details for Mike and Annie -/
structure TaxiTrip where
  initialCharge : ℝ
  costPerMile : ℝ
  surcharge : ℝ
  tollFees : ℝ
  miles : ℝ

/-- Calculates the total cost of a taxi trip -/
def tripCost (trip : TaxiTrip) : ℝ :=
  trip.initialCharge + trip.costPerMile * trip.miles + trip.surcharge + trip.tollFees

/-- Theorem stating that Mike's ride was 30 miles long -/
theorem mike_ride_length 
  (mike : TaxiTrip)
  (annie : TaxiTrip)
  (h1 : mike.initialCharge = 2.5)
  (h2 : mike.costPerMile = 0.25)
  (h3 : mike.surcharge = 3)
  (h4 : mike.tollFees = 0)
  (h5 : annie.initialCharge = 2.5)
  (h6 : annie.costPerMile = 0.25)
  (h7 : annie.surcharge = 0)
  (h8 : annie.tollFees = 5)
  (h9 : annie.miles = 22)
  (h10 : tripCost mike = tripCost annie) :
  mike.miles = 30 := by
  sorry

end NUMINAMATH_CALUDE_mike_ride_length_l2319_231911


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2319_231937

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ k = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2319_231937


namespace NUMINAMATH_CALUDE_panda_weekly_consumption_l2319_231949

/-- The amount of bamboo an adult panda eats per day in pounds -/
def adult_panda_daily_consumption : ℕ := 138

/-- The amount of bamboo a baby panda eats per day in pounds -/
def baby_panda_daily_consumption : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total bamboo consumption for an adult panda and a baby panda in a week is 1316 pounds -/
theorem panda_weekly_consumption :
  (adult_panda_daily_consumption * days_in_week) + (baby_panda_daily_consumption * days_in_week) = 1316 :=
by sorry

end NUMINAMATH_CALUDE_panda_weekly_consumption_l2319_231949


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l2319_231908

/-- The smallest positive integer divisible by all positive integers less than 8 -/
def M : ℕ := Nat.lcm 7 (Nat.lcm 6 (Nat.lcm 5 (Nat.lcm 4 (Nat.lcm 3 (Nat.lcm 2 1)))))

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_M :
  sum_of_digits M = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l2319_231908


namespace NUMINAMATH_CALUDE_first_super_monday_l2319_231977

/-- Represents a date with a month and day -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 3 => 31  -- March
  | 4 => 30  -- April
  | 5 => 31  -- May
  | _ => 30  -- Default (not used in this problem)

/-- Checks if a given date is a Monday -/
def isMonday (date : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Counts the number of Mondays in a given month -/
def countMondaysInMonth (month : Nat) (startDate : Date) (startDay : DayOfWeek) : Nat :=
  sorry

/-- Finds the date of the fifth Monday in a given month -/
def fifthMondayInMonth (month : Nat) (startDate : Date) (startDay : DayOfWeek) : Option Date :=
  sorry

/-- Theorem: The first Super Monday after school starts on Tuesday, March 1 is May 30 -/
theorem first_super_monday :
  let schoolStart : Date := ⟨3, 1⟩
  let firstSuperMonday := fifthMondayInMonth 5 schoolStart DayOfWeek.Tuesday
  firstSuperMonday = some ⟨5, 30⟩ :=
by
  sorry

#check first_super_monday

end NUMINAMATH_CALUDE_first_super_monday_l2319_231977


namespace NUMINAMATH_CALUDE_watermelon_pineapple_weight_difference_l2319_231910

/-- Given that 4 watermelons weigh 5200 grams and 3 watermelons plus 4 pineapples
    weigh 5700 grams, prove that a watermelon is 850 grams heavier than a pineapple. -/
theorem watermelon_pineapple_weight_difference :
  let watermelon_weight : ℕ := 5200 / 4
  let pineapple_weight : ℕ := (5700 - 3 * watermelon_weight) / 4
  watermelon_weight - pineapple_weight = 850 := by
sorry

end NUMINAMATH_CALUDE_watermelon_pineapple_weight_difference_l2319_231910


namespace NUMINAMATH_CALUDE_grid_toothpicks_l2319_231933

/-- The number of toothpicks needed to construct a grid of given length and width -/
def toothpicks_needed (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem stating that a grid of 50 toothpicks long and 40 toothpicks wide requires 4090 toothpicks -/
theorem grid_toothpicks : toothpicks_needed 50 40 = 4090 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l2319_231933


namespace NUMINAMATH_CALUDE_payment_equality_l2319_231920

/-- Represents the payment structure and hours worked for Harry and James -/
structure PaymentSystem where
  x : ℝ
  y : ℝ
  james_hours : ℝ
  harry_hours : ℝ

/-- Calculates James' earnings based on the given payment structure -/
def james_earnings (ps : PaymentSystem) : ℝ :=
  40 * ps.x + (ps.james_hours - 40) * 2 * ps.x

/-- Calculates Harry's earnings based on the given payment structure -/
def harry_earnings (ps : PaymentSystem) : ℝ :=
  12 * ps.x + (ps.harry_hours - 12) * ps.y * ps.x

/-- Theorem stating the conditions and the result to be proved -/
theorem payment_equality (ps : PaymentSystem) :
  ps.x > 0 ∧ 
  ps.y > 1 ∧ 
  ps.james_hours = 48 ∧ 
  james_earnings ps = harry_earnings ps →
  ps.harry_hours = 23 ∧ ps.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_payment_equality_l2319_231920


namespace NUMINAMATH_CALUDE_only_negative_three_halves_and_one_half_satisfy_l2319_231960

def numbers : List ℚ := [-3/2, -1, 1/2, 1, 3]

def satisfies_conditions (x : ℚ) : Prop :=
  x < x⁻¹ ∧ x > -3

theorem only_negative_three_halves_and_one_half_satisfy :
  ∀ x ∈ numbers, satisfies_conditions x ↔ (x = -3/2 ∨ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_three_halves_and_one_half_satisfy_l2319_231960


namespace NUMINAMATH_CALUDE_tangent_product_equality_l2319_231999

theorem tangent_product_equality : 
  (1 + Real.tan (20 * π / 180)) * (1 + Real.tan (25 * π / 180)) = 2 :=
by
  sorry

/- Proof hints:
   1. Use the fact that 45° = 20° + 25°
   2. Recall that tan(45°) = 1
   3. Apply the tangent sum formula: tan(A+B) = (tan A + tan B) / (1 - tan A * tan B)
   4. Algebraically manipulate the expressions
-/

end NUMINAMATH_CALUDE_tangent_product_equality_l2319_231999


namespace NUMINAMATH_CALUDE_peter_class_size_l2319_231996

/-- The number of hands in Peter's class, excluding Peter's hands -/
def hands_excluding_peter : ℕ := 20

/-- The number of hands each student has -/
def hands_per_student : ℕ := 2

/-- The total number of hands in the class, including Peter's -/
def total_hands : ℕ := hands_excluding_peter + hands_per_student

/-- The number of students in Peter's class, including Peter -/
def students_in_class : ℕ := total_hands / hands_per_student

theorem peter_class_size :
  students_in_class = 11 :=
sorry

end NUMINAMATH_CALUDE_peter_class_size_l2319_231996


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_120_l2319_231946

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_120 :
  rectangle_area 900 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_120_l2319_231946


namespace NUMINAMATH_CALUDE_complement_of_union_l2319_231989

def I : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {0, 3, 5, 7}

theorem complement_of_union : (I \ (M ∪ N)) = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2319_231989


namespace NUMINAMATH_CALUDE_vector_values_l2319_231995

-- Define the vectors
def OA (m : ℝ) : Fin 2 → ℝ := ![(-2 : ℝ), m]
def OB (n : ℝ) : Fin 2 → ℝ := ![n, (1 : ℝ)]
def OC : Fin 2 → ℝ := ![(5 : ℝ), (-1 : ℝ)]

-- Define collinearity
def collinear (A B C : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), B - A = t • (C - A)

-- Define perpendicularity
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

theorem vector_values (m n : ℝ) :
  collinear (OA m) (OB n) OC ∧
  perpendicular (OA m) (OB n) →
  m = 3 ∧ n = 3/2 := by sorry

end NUMINAMATH_CALUDE_vector_values_l2319_231995


namespace NUMINAMATH_CALUDE_expression_simplification_l2319_231904

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2319_231904


namespace NUMINAMATH_CALUDE_gas_tank_capacity_l2319_231915

/-- Represents the gas prices at each station -/
def gas_prices : List ℝ := [3, 3.5, 4, 4.5]

/-- Represents the total amount spent on gas -/
def total_spent : ℝ := 180

/-- Theorem: If a car owner fills up their tank 4 times at the given prices and spends $180 in total,
    then the gas tank capacity is 12 gallons -/
theorem gas_tank_capacity :
  ∀ (C : ℝ),
  (C > 0) →
  (List.sum (List.map (λ price => price * C) gas_prices) = total_spent) →
  C = 12 := by
  sorry

end NUMINAMATH_CALUDE_gas_tank_capacity_l2319_231915


namespace NUMINAMATH_CALUDE_max_sum_of_four_numbers_l2319_231925

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d → (c + d) + (a + b + c) = 2017 → 
  a + b + c + d ≤ 806 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_numbers_l2319_231925


namespace NUMINAMATH_CALUDE_printer_time_345_pages_l2319_231973

/-- The time (in minutes) it takes to print a given number of pages at a given rate -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  pages / rate

theorem printer_time_345_pages : 
  let pages := 345
  let rate := 23
  Int.floor (print_time pages rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_printer_time_345_pages_l2319_231973


namespace NUMINAMATH_CALUDE_even_function_implies_m_zero_l2319_231984

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Theorem statement
theorem even_function_implies_m_zero (m : ℝ) :
  is_even (f m) → m = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_m_zero_l2319_231984


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l2319_231932

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l2319_231932


namespace NUMINAMATH_CALUDE_prob_two_cards_sum_fifteen_l2319_231971

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face
| Ace

/-- A standard 52-card deck --/
def Deck : Finset Card := sorry

/-- The set of number cards (2 through 10) in the deck --/
def NumberCards : Finset Card := sorry

/-- The probability of drawing two specific cards from the deck --/
def drawTwoCardsProbability (card1 card2 : Card) : ℚ := sorry

/-- The sum of two cards --/
def cardSum (card1 card2 : Card) : ℕ := sorry

/-- The probability of drawing two number cards that sum to 15 --/
def probSumFifteen : ℚ := sorry

theorem prob_two_cards_sum_fifteen :
  probSumFifteen = 16 / 884 := by sorry

end NUMINAMATH_CALUDE_prob_two_cards_sum_fifteen_l2319_231971


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l2319_231945

/-- Given a piece of wood of length x and a rope of length y, 
    if there are 4.5 feet of rope left when measuring the wood 
    and 1 foot left when measuring with half the rope, 
    then the system of equations y - x = 4.5 and x - y/2 = 1 holds. -/
theorem sunzi_wood_measurement (x y : ℝ) 
  (h1 : y - x = 4.5) 
  (h2 : x - y / 2 = 1) : 
  y - x = 4.5 ∧ x - y / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l2319_231945


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2319_231990

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 4 + a 5 = 3) 
  (h_a8 : a 8 = 8) : 
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2319_231990


namespace NUMINAMATH_CALUDE_locus_of_R_l2319_231918

/-- The ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- Point A -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B -/
def B : ℝ × ℝ := (1, 0)

/-- Point C -/
def C : ℝ × ℝ := (3, 0)

/-- The locus equation -/
def locus_equation (x y : ℝ) : Prop := 45 * x^2 - 108 * y^2 = 20

/-- The theorem stating the locus of point R -/
theorem locus_of_R :
  ∀ (R : ℝ × ℝ),
  (∃ (P Q : ℝ × ℝ),
    ellipse P.1 P.2 ∧
    ellipse Q.1 Q.2 ∧
    Q.1 < P.1 ∧
    (∃ (t : ℝ), t > 0 ∧ Q.1 = C.1 - t * (C.1 - P.1) ∧ Q.2 = C.2 - t * (C.2 - P.2)) ∧
    (∃ (s : ℝ), A.1 + s * (P.1 - A.1) = R.1 ∧ A.2 + s * (P.2 - A.2) = R.2) ∧
    (∃ (u : ℝ), B.1 + u * (Q.1 - B.1) = R.1 ∧ B.2 + u * (Q.2 - B.2) = R.2)) →
  locus_equation R.1 R.2 ∧ 2/3 < R.1 ∧ R.1 < 4/3 :=
sorry

end NUMINAMATH_CALUDE_locus_of_R_l2319_231918


namespace NUMINAMATH_CALUDE_probability_five_white_two_red_l2319_231928

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def red_balls : ℕ := 3
def total_balls : ℕ := white_balls + black_balls + red_balls
def drawn_balls : ℕ := 7

theorem probability_five_white_two_red :
  (Nat.choose white_balls 5 * Nat.choose red_balls 2) / Nat.choose total_balls drawn_balls = 63 / 31824 :=
sorry

end NUMINAMATH_CALUDE_probability_five_white_two_red_l2319_231928


namespace NUMINAMATH_CALUDE_audrey_heracles_age_ratio_l2319_231959

/-- Proves that the ratio of Audrey's age in 3 years to Heracles' current age is 2:1 -/
theorem audrey_heracles_age_ratio :
  let heracles_age : ℕ := 10
  let audrey_age : ℕ := heracles_age + 7
  let audrey_age_in_3_years : ℕ := audrey_age + 3
  (audrey_age_in_3_years : ℚ) / heracles_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_audrey_heracles_age_ratio_l2319_231959


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l2319_231938

theorem min_value_reciprocal_sum (x : ℝ) (h : 0 < x ∧ x < 1) :
  1/x + 2/(1-x) ≥ 3 + 2*Real.sqrt 2 :=
sorry

theorem equality_condition (x : ℝ) (h : 0 < x ∧ x < 1) :
  1/x + 2/(1-x) = 3 + 2*Real.sqrt 2 ↔ x = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l2319_231938


namespace NUMINAMATH_CALUDE_max_xy_constraint_min_x_plus_y_constraint_l2319_231965

-- Part 1
theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y + x*y = 12) :
  x*y ≤ 4 ∧ (x*y = 4 → x = 4 ∧ y = 1) :=
sorry

-- Part 2
theorem min_x_plus_y_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = x*y) :
  x + y ≥ 9 ∧ (x + y = 9 → x = 6 ∧ y = 3) :=
sorry

end NUMINAMATH_CALUDE_max_xy_constraint_min_x_plus_y_constraint_l2319_231965


namespace NUMINAMATH_CALUDE_circular_pool_area_l2319_231926

theorem circular_pool_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : ∃ (r : ℝ), r^2 = 244 ∧ π * r^2 = 244 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_pool_area_l2319_231926


namespace NUMINAMATH_CALUDE_least_number_remainder_l2319_231901

theorem least_number_remainder (n : ℕ) : 
  (n % 20 = 1929 % 20 ∧ n % 2535 = 1929 ∧ n % 40 = 34) →
  (∀ m : ℕ, m < n → ¬(m % 20 = 1929 % 20 ∧ m % 2535 = 1929 ∧ m % 40 = 34)) →
  n = 1394 →
  n % 20 = 14 := by
sorry

end NUMINAMATH_CALUDE_least_number_remainder_l2319_231901


namespace NUMINAMATH_CALUDE_olivia_change_olivia_change_proof_l2319_231988

/-- Calculates the change Olivia received after buying basketball and baseball cards -/
theorem olivia_change (basketball_packs : ℕ) (basketball_price : ℕ) 
  (baseball_decks : ℕ) (baseball_price : ℕ) (bill : ℕ) : ℕ :=
  let total_cost := basketball_packs * basketball_price + baseball_decks * baseball_price
  bill - total_cost

/-- Proves that Olivia received $24 in change -/
theorem olivia_change_proof :
  olivia_change 2 3 5 4 50 = 24 := by
  sorry

end NUMINAMATH_CALUDE_olivia_change_olivia_change_proof_l2319_231988


namespace NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l2319_231914

theorem cooking_cleaning_arrangements (n : ℕ) (h : n = 8) : 
  Nat.choose n (n / 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l2319_231914


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_general_form_with_remainder_one_l2319_231923

theorem smallest_integer_with_remainder_one : ∃ (a : ℕ), a > 0 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → a % k = 1) ∧
  (∀ b : ℕ, b > 0 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → b % k = 1) → a ≤ b) ∧
  a = 2521 :=
sorry

theorem general_form_with_remainder_one :
  ∀ (a : ℕ), a > 0 →
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → a % k = 1) →
  ∃ (n : ℕ), a = 2520 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_general_form_with_remainder_one_l2319_231923


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2319_231957

theorem rationalize_denominator : 15 / Real.sqrt 45 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2319_231957


namespace NUMINAMATH_CALUDE_binomial_problem_l2319_231981

def binomial_expansion (m n : ℕ) (x : ℝ) : ℝ :=
  (1 + m * x) ^ n

theorem binomial_problem (m n : ℕ) (h1 : m ≠ 0) (h2 : n ≥ 2) :
  (∃ k, k = 5 ∧ ∀ j, j ≠ k → Nat.choose n j ≤ Nat.choose n k) →
  (Nat.choose n 2 * m^2 = 9 * Nat.choose n 1 * m) →
  (m = 2 ∧ n = 10 ∧ (binomial_expansion m n (-9)) % 6 = 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_problem_l2319_231981


namespace NUMINAMATH_CALUDE_triangle_area_PQR_l2319_231950

/-- The area of a triangle PQR with given coordinates -/
theorem triangle_area_PQR :
  let P : ℝ × ℝ := (-4, 2)
  let Q : ℝ × ℝ := (6, 2)
  let R : ℝ × ℝ := (2, -5)
  let base : ℝ := Q.1 - P.1
  let height : ℝ := P.2 - R.2
  (1 / 2 : ℝ) * base * height = 35 := by sorry

end NUMINAMATH_CALUDE_triangle_area_PQR_l2319_231950


namespace NUMINAMATH_CALUDE_rowing_speed_problem_l2319_231940

/-- Given a man who can row upstream at 26 kmph and downstream at 40 kmph,
    prove that his speed in still water is 33 kmph and the speed of the river current is 7 kmph. -/
theorem rowing_speed_problem (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 26)
  (h_downstream : downstream_speed = 40) :
  ∃ (still_water_speed river_current_speed : ℝ),
    still_water_speed = 33 ∧
    river_current_speed = 7 ∧
    upstream_speed = still_water_speed - river_current_speed ∧
    downstream_speed = still_water_speed + river_current_speed :=
by sorry

end NUMINAMATH_CALUDE_rowing_speed_problem_l2319_231940


namespace NUMINAMATH_CALUDE_complement_probability_l2319_231905

theorem complement_probability (event_prob : ℚ) (h : event_prob = 1/4) :
  1 - event_prob = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_complement_probability_l2319_231905


namespace NUMINAMATH_CALUDE_dog_distance_total_dog_distance_l2319_231964

/-- Proves that a dog running back and forth between two points covers 4000 meters
    by the time a person walks the distance between the points. -/
theorem dog_distance (distance : ℝ) (walking_speed : ℝ) (dog_speed : ℝ) : ℝ :=
  let time := distance * 1000 / walking_speed
  dog_speed * time

/-- The main theorem that calculates the total distance run by the dog. -/
theorem total_dog_distance : dog_distance 1 50 200 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_dog_distance_total_dog_distance_l2319_231964


namespace NUMINAMATH_CALUDE_cucumber_problem_l2319_231970

theorem cucumber_problem (boxes : ℕ) (cucumbers_per_box : ℕ) (rotten : ℕ) (bags : ℕ) :
  boxes = 7 →
  cucumbers_per_box = 16 →
  rotten = 13 →
  bags = 8 →
  (boxes * cucumbers_per_box - rotten) % bags = 3 := by
sorry

end NUMINAMATH_CALUDE_cucumber_problem_l2319_231970


namespace NUMINAMATH_CALUDE_library_books_sold_l2319_231929

theorem library_books_sold (total_books : ℕ) (remaining_fraction : ℚ) (books_sold : ℕ) : 
  total_books = 9900 ∧ remaining_fraction = 4/6 → books_sold = 3300 :=
by sorry

end NUMINAMATH_CALUDE_library_books_sold_l2319_231929


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2319_231936

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem purely_imaginary_condition (m : ℝ) :
  is_purely_imaginary ((m - i) * (1 + i)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2319_231936


namespace NUMINAMATH_CALUDE_trishas_total_distance_is_correct_l2319_231967

/-- The total distance Trisha walked during her vacation in New York City -/
def trishas_total_distance : ℝ :=
  let hotel_to_postcard := 0.11
  let postcard_to_hotel := 0.11
  let hotel_to_tshirt := 1.52
  let tshirt_to_hat := 0.45
  let hat_to_purse := 0.87
  let purse_to_hotel := 2.32
  hotel_to_postcard + postcard_to_hotel + hotel_to_tshirt + tshirt_to_hat + hat_to_purse + purse_to_hotel

/-- Theorem stating that the total distance Trisha walked is 5.38 miles -/
theorem trishas_total_distance_is_correct : trishas_total_distance = 5.38 := by
  sorry

end NUMINAMATH_CALUDE_trishas_total_distance_is_correct_l2319_231967


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2319_231952

/-- The number of apples initially in the cafeteria. -/
def initial_apples : ℕ := sorry

/-- The number of apples handed out to students. -/
def apples_handed_out : ℕ := 19

/-- The number of pies that can be made. -/
def pies_made : ℕ := 7

/-- The number of apples required for each pie. -/
def apples_per_pie : ℕ := 8

/-- The number of apples used for making pies. -/
def apples_for_pies : ℕ := pies_made * apples_per_pie

theorem cafeteria_apples : initial_apples = apples_handed_out + apples_for_pies := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2319_231952


namespace NUMINAMATH_CALUDE_second_number_value_l2319_231948

theorem second_number_value (x y : ℝ) 
  (h1 : x - y = 88)
  (h2 : y = 0.2 * x) : 
  y = 22 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l2319_231948


namespace NUMINAMATH_CALUDE_no_integer_solution_for_equation_l2319_231916

theorem no_integer_solution_for_equation : ¬ ∃ (x y : ℤ), x^2 - y^2 = 1998 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_equation_l2319_231916


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2319_231931

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  2 * x < 3 * (x - 3) + 1 ∧ (3 * x + 2) / 4 > x + a

-- Define the condition for having exactly four integer solutions
def has_four_integer_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℤ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    (∀ x : ℤ, inequality_system x a ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- The theorem to be proved
theorem inequality_system_solution_range :
  ∀ a : ℝ, has_four_integer_solutions a → -11/4 ≤ a ∧ a < -5/2 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2319_231931


namespace NUMINAMATH_CALUDE_basketball_game_score_l2319_231913

/-- Represents the score of a team in a quarter -/
structure QuarterScore where
  score : ℕ
  valid : score ≤ 25

/-- Represents the scores of a team for all four quarters -/
structure GameScore where
  q1 : QuarterScore
  q2 : QuarterScore
  q3 : QuarterScore
  q4 : QuarterScore
  increasing : q1.score < q2.score ∧ q2.score < q3.score ∧ q3.score < q4.score
  arithmetic : ∃ d : ℕ, q2.score = q1.score + d ∧ q3.score = q2.score + d ∧ q4.score = q3.score + d

def total_score (g : GameScore) : ℕ :=
  g.q1.score + g.q2.score + g.q3.score + g.q4.score

def first_half_score (g : GameScore) : ℕ :=
  g.q1.score + g.q2.score

theorem basketball_game_score :
  ∀ raiders wildcats : GameScore,
    raiders.q1 = wildcats.q1 →  -- First quarter tie
    total_score raiders = total_score wildcats + 2 →  -- Raiders win by 2
    first_half_score raiders + first_half_score wildcats = 38 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_score_l2319_231913


namespace NUMINAMATH_CALUDE_salary_increase_with_manager_l2319_231998

/-- Calculates the increase in average salary when a manager's salary is added to a group of employees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 18 → 
  avg_salary = 2000 → 
  manager_salary = 5800 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 200 :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_with_manager_l2319_231998


namespace NUMINAMATH_CALUDE_water_distribution_l2319_231917

/-- Proves that given 122 ounces of water, after filling six 5-ounce glasses and four 8-ounce glasses,
    the remaining water can fill exactly 15 four-ounce glasses. -/
theorem water_distribution (total_water : ℕ) (five_oz_glasses : ℕ) (eight_oz_glasses : ℕ) 
  (four_oz_glasses : ℕ) : 
  total_water = 122 ∧ 
  five_oz_glasses = 6 ∧ 
  eight_oz_glasses = 4 ∧ 
  four_oz_glasses * 4 = total_water - (five_oz_glasses * 5 + eight_oz_glasses * 8) → 
  four_oz_glasses = 15 :=
by sorry

end NUMINAMATH_CALUDE_water_distribution_l2319_231917


namespace NUMINAMATH_CALUDE_student_distribution_l2319_231963

/-- The number of students standing next to exactly one from club A and one from club B -/
def p : ℕ := 16

/-- The number of students standing between two from club A -/
def q : ℕ := 46

/-- The number of students standing between two from club B -/
def r : ℕ := 38

/-- The total number of students -/
def total : ℕ := 100

/-- The number of students standing next to at least one from club A -/
def next_to_A : ℕ := 62

/-- The number of students standing next to at least one from club B -/
def next_to_B : ℕ := 54

theorem student_distribution :
  p + q + r = total ∧
  p + q = next_to_A ∧
  p + r = next_to_B :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_l2319_231963


namespace NUMINAMATH_CALUDE_alice_pairs_l2319_231941

theorem alice_pairs (total_students : ℕ) (h1 : total_students = 12) : 
  (total_students - 2) = 11 := by
  sorry

#check alice_pairs

end NUMINAMATH_CALUDE_alice_pairs_l2319_231941


namespace NUMINAMATH_CALUDE_polygon_diagonals_twice_sides_l2319_231968

theorem polygon_diagonals_twice_sides (n : ℕ) : 
  n ≥ 3 → (n * (n - 3) / 2 = 2 * n ↔ n = 7) := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_twice_sides_l2319_231968


namespace NUMINAMATH_CALUDE_inequality_on_unit_circle_l2319_231944

/-- The complex unit circle -/
def unit_circle : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- The inequality holds for all points on the unit circle -/
theorem inequality_on_unit_circle :
  ∀ z ∈ unit_circle, (Complex.abs (z + 1) - Real.sqrt 2) * (Complex.abs (z - 1) - Real.sqrt 2) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_on_unit_circle_l2319_231944


namespace NUMINAMATH_CALUDE_angle_equality_l2319_231986

theorem angle_equality (a b c t : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < t) :
  let S := Real.sqrt (a^2 + b^2 + c^2)
  let ω1 := Real.arctan ((4*t) / (a^2 + b^2 + c^2))
  let ω2 := Real.arccos ((a^2 + b^2 + c^2) / S)
  ω1 = ω2 := by sorry

end NUMINAMATH_CALUDE_angle_equality_l2319_231986


namespace NUMINAMATH_CALUDE_min_cars_with_stripes_is_two_l2319_231951

/-- Represents the properties of a car group -/
structure CarGroup where
  total : ℕ
  no_ac : ℕ
  max_ac_no_stripes : ℕ

/-- The minimum number of cars with racing stripes -/
def min_cars_with_stripes (group : CarGroup) : ℕ :=
  group.total - group.no_ac - group.max_ac_no_stripes

/-- Theorem stating the minimum number of cars with racing stripes -/
theorem min_cars_with_stripes_is_two (group : CarGroup) 
  (h1 : group.total = 100)
  (h2 : group.no_ac = 49)
  (h3 : group.max_ac_no_stripes = 49) : 
  min_cars_with_stripes group = 2 := by
  sorry

#eval min_cars_with_stripes ⟨100, 49, 49⟩

end NUMINAMATH_CALUDE_min_cars_with_stripes_is_two_l2319_231951


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l2319_231954

theorem polynomial_product_expansion :
  let p₁ : Polynomial ℝ := 3 * X^2 + 4 * X - 5
  let p₂ : Polynomial ℝ := 4 * X^3 - 3 * X^2 + 2 * X - 1
  p₁ * p₂ = 12 * X^5 + 25 * X^4 - 41 * X^3 - 14 * X^2 + 28 * X - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l2319_231954


namespace NUMINAMATH_CALUDE_resulting_solution_percentage_l2319_231980

/-- Calculates the percentage of chemicals in the resulting solution when a portion of a 90% solution is replaced with an equal amount of 20% solution. -/
theorem resulting_solution_percentage 
  (original_concentration : Real) 
  (replacement_concentration : Real)
  (replaced_portion : Real) :
  original_concentration = 0.9 →
  replacement_concentration = 0.2 →
  replaced_portion = 0.7142857142857143 →
  let remaining_portion := 1 - replaced_portion
  let chemicals_in_remaining := remaining_portion * original_concentration
  let chemicals_in_added := replaced_portion * replacement_concentration
  let total_chemicals := chemicals_in_remaining + chemicals_in_added
  let resulting_concentration := total_chemicals / 1
  resulting_concentration = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_resulting_solution_percentage_l2319_231980


namespace NUMINAMATH_CALUDE_sparrow_population_decrease_l2319_231935

def initial_population : ℕ := 1200
def decrease_rate : ℚ := 0.7
def target_percentage : ℚ := 0.15

def population (year : ℕ) : ℚ :=
  (initial_population : ℚ) * decrease_rate ^ (year - 2010)

theorem sparrow_population_decrease (year : ℕ) :
  year = 2016 ↔ 
    (population year < (initial_population : ℚ) * target_percentage ∧
     ∀ y, 2010 ≤ y ∧ y < 2016 → population y ≥ (initial_population : ℚ) * target_percentage) :=
by sorry

end NUMINAMATH_CALUDE_sparrow_population_decrease_l2319_231935


namespace NUMINAMATH_CALUDE_area_of_ADE_l2319_231974

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def Triangle.area (t : Triangle) : ℝ := sorry

def Triangle.isRightAngle (t : Triangle) (vertex : ℝ × ℝ) : Prop := sorry

def angle_bisector (A B C D : ℝ × ℝ) (E : ℝ × ℝ) : Prop := sorry

theorem area_of_ADE (A B C D E : ℝ × ℝ) : 
  let abc := Triangle.mk A B C
  let abd := Triangle.mk A B D
  (Triangle.area abc = 24) →
  (Triangle.isRightAngle abc B) →
  (Triangle.isRightAngle abd B) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64) →
  ((B.1 - C.1)^2 + (B.2 - C.2)^2 = 36) →
  ((A.1 - D.1)^2 + (A.2 - D.2)^2 = 64) →
  (angle_bisector A C A D E) →
  Triangle.area (Triangle.mk A D E) = 20 := by sorry

end NUMINAMATH_CALUDE_area_of_ADE_l2319_231974


namespace NUMINAMATH_CALUDE_new_class_mean_approx_67_percent_l2319_231943

/-- Represents the class statistics for Mr. Thompson's chemistry class -/
structure ChemistryClass where
  total_students : ℕ
  group1_students : ℕ
  group1_average : ℚ
  group2_students : ℕ
  group2_average : ℚ
  group3_students : ℕ
  group3_average : ℚ

/-- Calculates the new class mean for Mr. Thompson's chemistry class -/
def new_class_mean (c : ChemistryClass) : ℚ :=
  (c.group1_students * c.group1_average + 
   c.group2_students * c.group2_average + 
   c.group3_students * c.group3_average) / c.total_students

/-- Theorem stating that the new class mean is approximately 67% -/
theorem new_class_mean_approx_67_percent (c : ChemistryClass) 
  (h1 : c.total_students = 60)
  (h2 : c.group1_students = 50)
  (h3 : c.group1_average = 65/100)
  (h4 : c.group2_students = 8)
  (h5 : c.group2_average = 85/100)
  (h6 : c.group3_students = 2)
  (h7 : c.group3_average = 55/100) :
  ∃ ε > 0, |new_class_mean c - 67/100| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_new_class_mean_approx_67_percent_l2319_231943


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2319_231902

theorem sum_of_coefficients : ∃ (a b c : ℕ+), 
  (a.val : ℝ) * Real.sqrt 6 + (b.val : ℝ) * Real.sqrt 8 = c.val * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) ∧ 
  (∀ (a' b' c' : ℕ+), (a'.val : ℝ) * Real.sqrt 6 + (b'.val : ℝ) * Real.sqrt 8 = c'.val * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) → c'.val ≥ c.val) →
  a.val + b.val + c.val = 67 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2319_231902
