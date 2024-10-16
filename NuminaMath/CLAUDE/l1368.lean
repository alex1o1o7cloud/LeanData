import Mathlib

namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1368_136888

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1368_136888


namespace NUMINAMATH_CALUDE_base_k_subtraction_l1368_136846

/-- Represents a digit in base k -/
def Digit (k : ℕ) := {d : ℕ // d < k}

/-- Converts a two-digit number in base k to its decimal representation -/
def toDecimal (k : ℕ) (x y : Digit k) : ℕ := k * x.val + y.val

theorem base_k_subtraction (k : ℕ) (X Y : Digit k) 
  (h_k : k > 8)
  (h_eq : toDecimal k X Y + toDecimal k X X = 2 * k + 1) :
  X.val - Y.val = k - 4 := by sorry

end NUMINAMATH_CALUDE_base_k_subtraction_l1368_136846


namespace NUMINAMATH_CALUDE_half_marathon_total_yards_l1368_136840

/-- Represents the length of a race in miles and yards -/
structure RaceLength where
  miles : ℕ
  yards : ℚ

def half_marathon : RaceLength := { miles := 13, yards := 192.5 }

def yards_per_mile : ℕ := 1760

def num_races : ℕ := 6

theorem half_marathon_total_yards (m : ℕ) (y : ℚ) 
  (h1 : 0 ≤ y) (h2 : y < yards_per_mile) :
  m * yards_per_mile + y = 
    num_races * (half_marathon.miles * yards_per_mile + half_marathon.yards) → 
  y = 1155 := by
  sorry

end NUMINAMATH_CALUDE_half_marathon_total_yards_l1368_136840


namespace NUMINAMATH_CALUDE_work_rate_problem_l1368_136837

theorem work_rate_problem (A B C D : ℚ) :
  A = 1 / 4 →
  A + C = 1 / 2 →
  B + C = 1 / 3 →
  D = 1 / 5 →
  A + B + C + D = 1 →
  B = 13 / 60 :=
by sorry

end NUMINAMATH_CALUDE_work_rate_problem_l1368_136837


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1368_136890

theorem min_value_of_fraction (x : ℝ) (h : x ≥ 3/2) :
  (2*x^2 - 2*x + 1) / (x - 1) ≥ 2*Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1368_136890


namespace NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l1368_136804

/-- The diameter of a circular field given the cost of fencing. -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- The diameter of the specific circular field is approximately 34 meters. -/
theorem specific_field_diameter : 
  abs (circular_field_diameter 2 213.63 - 34) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_circular_field_diameter_specific_field_diameter_l1368_136804


namespace NUMINAMATH_CALUDE_expression_evaluation_l1368_136857

theorem expression_evaluation (x y z : ℝ) : (x + (y + z)) - ((-x + y) + z) = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1368_136857


namespace NUMINAMATH_CALUDE_nine_to_power_2023_div_3_l1368_136828

theorem nine_to_power_2023_div_3 (n : ℕ) : n = 9^2023 → n / 3 = 3^4045 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_to_power_2023_div_3_l1368_136828


namespace NUMINAMATH_CALUDE_product_equals_eight_l1368_136848

theorem product_equals_eight : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l1368_136848


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l1368_136849

theorem inverse_proposition_false : ¬ (∀ a b : ℝ, a^2 = b^2 → a = b) := by sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l1368_136849


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_five_primes_l1368_136851

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def reciprocals (lst : List Nat) : List Rat :=
  lst.map (λ x => 1 / x)

def arithmetic_mean (lst : List Rat) : Rat :=
  lst.sum / lst.length

theorem arithmetic_mean_of_reciprocals_first_five_primes :
  arithmetic_mean (reciprocals first_five_primes) = 2927 / 11550 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_first_five_primes_l1368_136851


namespace NUMINAMATH_CALUDE_peach_difference_l1368_136887

def steven_peaches : ℕ := 13
def jake_peaches : ℕ := 7

theorem peach_difference : steven_peaches - jake_peaches = 6 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l1368_136887


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_12_pow_3_l1368_136881

theorem greatest_prime_factor_of_5_pow_5_plus_12_pow_3 :
  (Nat.factors (5^5 + 12^3)).maximum? = some 19 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_5_plus_12_pow_3_l1368_136881


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_products_l1368_136876

theorem largest_divisor_of_consecutive_even_products (n : ℕ+) : 
  let Q := (2 * n) * (2 * n + 2) * (2 * n + 4)
  ∃ k : ℕ, Q = 12 * k ∧ ∀ m : ℕ, m > 12 → ¬(∀ n : ℕ+, m ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_products_l1368_136876


namespace NUMINAMATH_CALUDE_boats_in_lake_l1368_136844

theorem boats_in_lake (people_per_boat : ℕ) (total_people : ℕ) (number_of_boats : ℕ) : 
  people_per_boat = 3 → total_people = 15 → number_of_boats * people_per_boat = total_people → 
  number_of_boats = 5 := by
  sorry

end NUMINAMATH_CALUDE_boats_in_lake_l1368_136844


namespace NUMINAMATH_CALUDE_prob_red_ball_specific_l1368_136805

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ
  sum_colors : total = red + yellow + green

/-- The probability of drawing a red ball from a bag of colored balls -/
def prob_red_ball (bag : ColoredBalls) : ℚ :=
  bag.red / bag.total

/-- Theorem: The probability of drawing a red ball from a bag with 15 balls, 
    of which 8 are red, is 8/15 -/
theorem prob_red_ball_specific : 
  ∃ (bag : ColoredBalls), bag.total = 15 ∧ bag.red = 8 ∧ prob_red_ball bag = 8/15 := by
  sorry


end NUMINAMATH_CALUDE_prob_red_ball_specific_l1368_136805


namespace NUMINAMATH_CALUDE_right_triangle_expansion_l1368_136869

theorem right_triangle_expansion : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  a^2 + b^2 = c^2 ∧
  (a + 100)^2 + (b + 100)^2 = (c + 140)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_expansion_l1368_136869


namespace NUMINAMATH_CALUDE_optimal_distribution_l1368_136833

/-- Represents the profit distribution problem for a fruit distributor. -/
structure FruitDistribution where
  /-- Total number of boxes of each fruit type -/
  total_boxes : ℕ
  /-- Profit per box of A fruit at Store A -/
  profit_a_store_a : ℕ
  /-- Profit per box of B fruit at Store A -/
  profit_b_store_a : ℕ
  /-- Profit per box of A fruit at Store B -/
  profit_a_store_b : ℕ
  /-- Profit per box of B fruit at Store B -/
  profit_b_store_b : ℕ
  /-- Minimum profit required for Store B -/
  min_profit_store_b : ℕ

/-- Theorem stating the optimal distribution and maximum profit -/
theorem optimal_distribution (fd : FruitDistribution)
  (h1 : fd.total_boxes = 10)
  (h2 : fd.profit_a_store_a = 11)
  (h3 : fd.profit_b_store_a = 17)
  (h4 : fd.profit_a_store_b = 9)
  (h5 : fd.profit_b_store_b = 13)
  (h6 : fd.min_profit_store_b = 115) :
  ∃ (a_store_a b_store_a a_store_b b_store_b : ℕ),
    a_store_a + b_store_a = fd.total_boxes ∧
    a_store_b + b_store_b = fd.total_boxes ∧
    a_store_a + a_store_b = fd.total_boxes ∧
    b_store_a + b_store_b = fd.total_boxes ∧
    fd.profit_a_store_b * a_store_b + fd.profit_b_store_b * b_store_b ≥ fd.min_profit_store_b ∧
    fd.profit_a_store_a * a_store_a + fd.profit_b_store_a * b_store_a +
    fd.profit_a_store_b * a_store_b + fd.profit_b_store_b * b_store_b = 246 ∧
    a_store_a = 7 ∧ b_store_a = 3 ∧ a_store_b = 3 ∧ b_store_b = 7 ∧
    ∀ (x y z w : ℕ),
      x + y = fd.total_boxes →
      z + w = fd.total_boxes →
      x + z = fd.total_boxes →
      y + w = fd.total_boxes →
      fd.profit_a_store_b * z + fd.profit_b_store_b * w ≥ fd.min_profit_store_b →
      fd.profit_a_store_a * x + fd.profit_b_store_a * y +
      fd.profit_a_store_b * z + fd.profit_b_store_b * w ≤ 246 :=
by sorry


end NUMINAMATH_CALUDE_optimal_distribution_l1368_136833


namespace NUMINAMATH_CALUDE_largest_c_value_l1368_136870

theorem largest_c_value (c : ℝ) : 
  (∀ x : ℝ, -2*x^2 + 8*x - 6 ≥ 0 → x ≤ c) ↔ c = 3 := by sorry

end NUMINAMATH_CALUDE_largest_c_value_l1368_136870


namespace NUMINAMATH_CALUDE_solution_set_for_a_2_a_value_for_even_function_l1368_136875

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_set_for_a_2 :
  {x : ℝ | f 2 x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} := by sorry

-- Part 2
theorem a_value_for_even_function :
  (∀ x, f a x = f a (-x)) → a = -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_2_a_value_for_even_function_l1368_136875


namespace NUMINAMATH_CALUDE_min_value_theorem_l1368_136854

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 2 * m - n * (-2) - 2 = 0) :
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 2 * m' - n' * (-2) - 2 = 0 → 
    1 / m + 2 / n ≤ 1 / m' + 2 / n') ∧ 
  (1 / m + 2 / n = 3 + 2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1368_136854


namespace NUMINAMATH_CALUDE_zoo_field_trip_l1368_136821

theorem zoo_field_trip (students_class1 students_class2 parent_chaperones : ℕ)
  (students_left chaperones_left remaining : ℕ) :
  students_class1 = 10 →
  students_class2 = 10 →
  parent_chaperones = 5 →
  students_left = 10 →
  chaperones_left = 2 →
  remaining = 15 →
  ∃ (teachers : ℕ),
    teachers = 2 ∧
    (students_class1 + students_class2 + parent_chaperones + teachers) -
    (students_left + chaperones_left) = remaining :=
by sorry

end NUMINAMATH_CALUDE_zoo_field_trip_l1368_136821


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1368_136818

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = -3 ∧ x₂ = 4) ∧ 
  (x₁^2 - x₁ - 12 = 0) ∧ 
  (x₂^2 - x₂ - 12 = 0) ∧
  (∀ x : ℝ, x^2 - x - 12 = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1368_136818


namespace NUMINAMATH_CALUDE_geometric_sequence_nth_term_l1368_136885

theorem geometric_sequence_nth_term (a₁ q : ℚ) (n : ℕ) (h1 : a₁ = 1/2) (h2 : q = 1/2) :
  a₁ * q^(n - 1) = 1/32 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_nth_term_l1368_136885


namespace NUMINAMATH_CALUDE_firecracker_explosion_speed_firecracker_explosion_speed_proof_l1368_136895

/-- The speed of a fragment after an explosion, given initial conditions of a firecracker --/
theorem firecracker_explosion_speed 
  (v₀ : ℝ)           -- Initial upward speed of firecracker
  (t : ℝ)            -- Time of explosion after launch
  (g : ℝ)            -- Acceleration due to gravity
  (v_horiz : ℝ)      -- Horizontal speed of first fragment after explosion
  (h : v₀ = 20)      -- Initial speed is 20 m/s
  (ht : t = 1)       -- Explosion occurs at 1 second
  (hg : g = 10)      -- Gravity is 10 m/s²
  (hv : v_horiz = 48) -- First fragment's horizontal speed is 48 m/s
  : ℝ :=
-- The speed of the second fragment after explosion
52

/-- Proof of the firecracker explosion speed theorem --/
theorem firecracker_explosion_speed_proof 
  (v₀ t g v_horiz : ℝ)
  (h : v₀ = 20)
  (ht : t = 1)
  (hg : g = 10)
  (hv : v_horiz = 48)
  : firecracker_explosion_speed v₀ t g v_horiz h ht hg hv = 52 := by
  sorry

end NUMINAMATH_CALUDE_firecracker_explosion_speed_firecracker_explosion_speed_proof_l1368_136895


namespace NUMINAMATH_CALUDE_custom_operation_equality_l1368_136893

/-- Custom operation ⊕ for real numbers -/
def circle_plus (a b : ℝ) : ℝ := (a + b) ^ 2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y : ℝ) : 
  circle_plus (circle_plus ((x + y) ^ 2) ((y + x) ^ 2)) 2 = 4 * ((x + y) ^ 2 + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l1368_136893


namespace NUMINAMATH_CALUDE_complex_root_product_l1368_136809

theorem complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_product_l1368_136809


namespace NUMINAMATH_CALUDE_square_point_selection_probability_square_point_selection_probability_is_three_fifths_l1368_136817

/-- The probability of selecting two points from the vertices and center of a square,
    such that their distance is not less than the side length of the square. -/
theorem square_point_selection_probability : ℚ :=
  let total_selections := (5 : ℕ).choose 2
  let favorable_selections := (4 : ℕ).choose 2
  (favorable_selections : ℚ) / total_selections

/-- The probability of selecting two points from the vertices and center of a square,
    such that their distance is not less than the side length of the square, is 3/5. -/
theorem square_point_selection_probability_is_three_fifths :
  square_point_selection_probability = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_point_selection_probability_square_point_selection_probability_is_three_fifths_l1368_136817


namespace NUMINAMATH_CALUDE_plant_branches_problem_l1368_136835

theorem plant_branches_problem :
  ∃ (x : ℕ),
    (1 : ℕ) + x + x * x = 91 ∧
    (∀ y : ℕ, (1 : ℕ) + y + y * y = 91 → y ≤ x) ∧
    x = 9 :=
by sorry

end NUMINAMATH_CALUDE_plant_branches_problem_l1368_136835


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1368_136873

theorem binomial_coefficient_equation_solution (x : ℕ) : 
  (Nat.choose 12 (x + 1) = Nat.choose 12 (2 * x - 1)) ↔ (x = 2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l1368_136873


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1368_136845

theorem tan_double_angle_special_case (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1368_136845


namespace NUMINAMATH_CALUDE_green_eyed_students_l1368_136841

theorem green_eyed_students (total : ℕ) (brown_green : ℕ) (neither : ℕ) 
  (h1 : total = 45)
  (h2 : brown_green = 9)
  (h3 : neither = 5) :
  ∃ (green : ℕ), 
    green = 10 ∧ 
    total = green + 3 * green - brown_green + neither :=
by
  sorry

end NUMINAMATH_CALUDE_green_eyed_students_l1368_136841


namespace NUMINAMATH_CALUDE_commission_calculation_l1368_136847

/-- Calculates the commission earned from selling a coupe and an SUV --/
theorem commission_calculation (coupe_price : ℝ) (suv_price_multiplier : ℝ) (commission_rate : ℝ) :
  coupe_price = 30000 →
  suv_price_multiplier = 2 →
  commission_rate = 0.02 →
  coupe_price * suv_price_multiplier * commission_rate + coupe_price * commission_rate = 1800 := by
  sorry

end NUMINAMATH_CALUDE_commission_calculation_l1368_136847


namespace NUMINAMATH_CALUDE_largest_vertex_sum_l1368_136839

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h : T ≠ 0

/-- The sum of coordinates of the vertex of the parabola -/
def vertexSum (p : Parabola) : ℤ := p.T - p.a * p.T^2

/-- The parabola passes through the point (2T+1, 28) -/
def passesThroughC (p : Parabola) : Prop :=
  p.a * (2 * p.T + 1) = 28

theorem largest_vertex_sum :
  ∀ p : Parabola, passesThroughC p → vertexSum p ≤ 60 :=
sorry

end NUMINAMATH_CALUDE_largest_vertex_sum_l1368_136839


namespace NUMINAMATH_CALUDE_min_top_supervisors_bound_l1368_136868

/-- Represents the structure of a company --/
structure Company where
  total_employees : ℕ
  supervisor_subordinate_sum : ℕ
  propagation_days : ℕ

/-- Calculates the minimum number of top-level supervisors --/
def min_top_supervisors (c : Company) : ℕ :=
  ((c.total_employees - 1) / (1 + c.supervisor_subordinate_sum + c.supervisor_subordinate_sum ^ 2 + c.supervisor_subordinate_sum ^ 3 + c.supervisor_subordinate_sum ^ 4)) + 1

/-- The theorem to be proved --/
theorem min_top_supervisors_bound (c : Company) 
  (h1 : c.total_employees = 50000)
  (h2 : c.supervisor_subordinate_sum = 7)
  (h3 : c.propagation_days = 4) :
  min_top_supervisors c ≥ 97 := by
  sorry

#eval min_top_supervisors ⟨50000, 7, 4⟩

end NUMINAMATH_CALUDE_min_top_supervisors_bound_l1368_136868


namespace NUMINAMATH_CALUDE_yellow_ball_estimate_l1368_136823

/-- Represents the contents of a bag with red and yellow balls -/
structure BagContents where
  red_balls : ℕ
  yellow_balls : ℕ

/-- Represents the result of multiple trials of drawing balls -/
structure TrialResults where
  num_trials : ℕ
  avg_red_ratio : ℝ

/-- Estimates the number of yellow balls in the bag based on trial results -/
def estimate_yellow_balls (bag : BagContents) (trials : TrialResults) : ℕ :=
  sorry

theorem yellow_ball_estimate (bag : BagContents) (trials : TrialResults) :
  bag.red_balls = 10 ∧ 
  trials.num_trials = 20 ∧ 
  trials.avg_red_ratio = 0.4 →
  estimate_yellow_balls bag trials = 15 :=
sorry

end NUMINAMATH_CALUDE_yellow_ball_estimate_l1368_136823


namespace NUMINAMATH_CALUDE_buses_needed_l1368_136896

theorem buses_needed (total_students : ℕ) (seats_per_bus : ℕ) (h1 : total_students = 111) (h2 : seats_per_bus = 3) :
  (total_students + seats_per_bus - 1) / seats_per_bus = 37 :=
by sorry

end NUMINAMATH_CALUDE_buses_needed_l1368_136896


namespace NUMINAMATH_CALUDE_max_squared_ratio_is_one_l1368_136863

/-- The maximum squared ratio of a to b satisfying the given conditions -/
def max_squared_ratio (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧
  ∃ ρ : ℝ, ρ > 0 ∧
    ∀ x y : ℝ,
      0 ≤ x ∧ x < a ∧
      0 ≤ y ∧ y < b ∧
      a^2 + y^2 = b^2 + x^2 ∧
      b^2 + x^2 = (a - x)^2 + (b - y)^2 ∧
      (a - x) * (b - y) = 0 →
      (a / b)^2 ≤ ρ^2 ∧
      ρ^2 = 1

theorem max_squared_ratio_is_one (a b : ℝ) (h : max_squared_ratio a b) :
  ∃ ρ : ℝ, ρ > 0 ∧ ρ^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_squared_ratio_is_one_l1368_136863


namespace NUMINAMATH_CALUDE_total_pay_per_episode_l1368_136802

def tv_show_pay (main_characters minor_characters minor_pay major_pay_ratio : ℕ) : ℕ :=
  let minor_total := minor_characters * minor_pay
  let major_total := main_characters * (major_pay_ratio * minor_pay)
  minor_total + major_total

theorem total_pay_per_episode :
  tv_show_pay 5 4 15000 3 = 285000 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pay_per_episode_l1368_136802


namespace NUMINAMATH_CALUDE_square_hole_reassembly_l1368_136878

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Represents a square with a square hole -/
structure SquareWithHole :=
  (outer_side : ℝ)
  (hole_side : ℝ)
  (hole_position : ℝ × ℝ)

/-- Function to divide a square with a hole into four quadrilaterals -/
def divide_square (s : SquareWithHole) : Fin 4 → Quadrilateral :=
  sorry

/-- Function to check if a set of quadrilaterals can form a square with a hole -/
def can_form_square_with_hole (quads : Fin 4 → Quadrilateral) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem square_hole_reassembly 
  (s : SquareWithHole) : 
  can_form_square_with_hole (divide_square s) :=
sorry

end NUMINAMATH_CALUDE_square_hole_reassembly_l1368_136878


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1368_136886

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def common_difference (a : ℕ → ℚ) : ℚ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_condition1 : a 7 - 2 * a 4 = -1)
  (h_condition2 : a 3 = 0) :
  common_difference a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1368_136886


namespace NUMINAMATH_CALUDE_triangle_properties_l1368_136808

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Define triangle ABC
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) →
  b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition
  (Real.sqrt 3 / 3) * b * Real.sin C + c * Real.cos B = a →
  -- Part 1
  (a = 2 ∧ b = 1 → (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2) ∧
  -- Part 2
  (c = 2 → 2 * Real.sqrt 3 + 2 < a + b + c ∧ a + b + c ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1368_136808


namespace NUMINAMATH_CALUDE_square_between_squares_l1368_136827

theorem square_between_squares (n k l m : ℕ) :
  m^2 < n ∧ n < (m+1)^2 ∧ n - k = m^2 ∧ n + l = (m+1)^2 →
  ∃ p : ℕ, n - k * l = p^2 := by
sorry

end NUMINAMATH_CALUDE_square_between_squares_l1368_136827


namespace NUMINAMATH_CALUDE_f_properties_l1368_136820

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)

theorem f_properties (f : ℝ → ℝ) (h : f_property f) :
  (∀ x : ℝ, f (f x) ≤ 0) ∧
  (f 0 ≥ 0 → ∀ x : ℝ, f x = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1368_136820


namespace NUMINAMATH_CALUDE_child_ticket_cost_l1368_136853

theorem child_ticket_cost (num_children num_adults : ℕ) (adult_ticket_cost total_cost : ℚ) :
  num_children = 6 →
  num_adults = 10 →
  adult_ticket_cost = 16 →
  total_cost = 220 →
  (total_cost - num_adults * adult_ticket_cost) / num_children = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l1368_136853


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1368_136807

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) :
  |x - y| = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1368_136807


namespace NUMINAMATH_CALUDE_no_solutions_to_equation_l1368_136811

theorem no_solutions_to_equation : 
  ¬∃ (x : ℝ), |x - 1| = |2*x - 4| + |x - 5| := by
sorry

end NUMINAMATH_CALUDE_no_solutions_to_equation_l1368_136811


namespace NUMINAMATH_CALUDE_rs_value_l1368_136872

theorem rs_value (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 5/8) : r * s = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rs_value_l1368_136872


namespace NUMINAMATH_CALUDE_total_cost_new_puppy_l1368_136860

def adoption_fee : ℝ := 20
def dog_food : ℝ := 20
def treat_bag_price : ℝ := 2.5
def num_treat_bags : ℕ := 2
def toys : ℝ := 15
def crate : ℝ := 20
def bed : ℝ := 20
def collar_leash : ℝ := 15
def discount_rate : ℝ := 0.2

theorem total_cost_new_puppy :
  let supplies_cost := dog_food + treat_bag_price * num_treat_bags + toys + crate + bed + collar_leash
  let discounted_supplies_cost := supplies_cost * (1 - discount_rate)
  adoption_fee + discounted_supplies_cost = 96 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_new_puppy_l1368_136860


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1368_136852

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ioo 0 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1368_136852


namespace NUMINAMATH_CALUDE_parallel_no_common_points_relation_l1368_136803

-- Define the concept of lines in a space
axiom Line : Type

-- Define the parallel relation between lines
axiom parallel : Line → Line → Prop

-- Define the property of having no common points
axiom no_common_points : Line → Line → Prop

-- Define the theorem
theorem parallel_no_common_points_relation (a b : Line) :
  (parallel a b → no_common_points a b) ∧
  ¬(no_common_points a b → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_no_common_points_relation_l1368_136803


namespace NUMINAMATH_CALUDE_equilateral_roots_ratio_l1368_136894

/-- Given complex numbers z₁ and z₂ that are roots of z² + pz + q = 0,
    where p and q are complex numbers, and 0, z₁, and z₂ form an
    equilateral triangle in the complex plane, then p²/q = 1. -/
theorem equilateral_roots_ratio (p q z₁ z₂ : ℂ) :
  z₁^2 + p*z₁ + q = 0 →
  z₂^2 + p*z₂ + q = 0 →
  ∃ (ω : ℂ), ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁ →
  p^2 / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_roots_ratio_l1368_136894


namespace NUMINAMATH_CALUDE_willie_cream_total_l1368_136871

/-- The amount of whipped cream Willie needs in total -/
def total_cream (farm_cream : ℕ) (bought_cream : ℕ) : ℕ :=
  farm_cream + bought_cream

/-- Theorem stating that Willie needs 300 lbs. of whipped cream in total -/
theorem willie_cream_total :
  total_cream 149 151 = 300 := by
  sorry

end NUMINAMATH_CALUDE_willie_cream_total_l1368_136871


namespace NUMINAMATH_CALUDE_sqrt_sin_sum_equals_neg_two_cos_three_l1368_136806

theorem sqrt_sin_sum_equals_neg_two_cos_three :
  Real.sqrt (1 + Real.sin 6) + Real.sqrt (1 - Real.sin 6) = -2 * Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sin_sum_equals_neg_two_cos_three_l1368_136806


namespace NUMINAMATH_CALUDE_min_value_of_function_l1368_136859

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  x + 9 / (x - 2) ≥ 8 ∧ ∃ y > 2, y + 9 / (y - 2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1368_136859


namespace NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l1368_136814

theorem cube_sum_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 4)
  (sum_products_eq : x*y + x*z + y*z = 3)
  (product_eq : x*y*z = -10) :
  x^3 + y^3 + z^3 = 10 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_three_numbers_l1368_136814


namespace NUMINAMATH_CALUDE_second_group_size_l1368_136830

/-- Given a gym class with two groups of students, prove that the second group has 37 students. -/
theorem second_group_size (total : ℕ) (group1 : ℕ) (group2 : ℕ) : 
  total = 71 → group1 = 34 → total = group1 + group2 → group2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l1368_136830


namespace NUMINAMATH_CALUDE_two_common_points_l1368_136898

/-- Two curves in the xy-plane -/
structure Curves where
  curve1 : ℝ → ℝ → Prop
  curve2 : ℝ → ℝ → Prop

/-- The specific curves from the problem -/
def problem_curves : Curves where
  curve1 := λ x y => x^2 + 9*y^2 = 9
  curve2 := λ x y => 9*x^2 + y^2 = 1

/-- A point that satisfies both curves -/
def is_common_point (c : Curves) (x y : ℝ) : Prop :=
  c.curve1 x y ∧ c.curve2 x y

/-- The set of all common points -/
def common_points (c : Curves) : Set (ℝ × ℝ) :=
  {p | is_common_point c p.1 p.2}

/-- The theorem stating that there are exactly two common points -/
theorem two_common_points :
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧
  common_points problem_curves = {p1, p2} :=
sorry

end NUMINAMATH_CALUDE_two_common_points_l1368_136898


namespace NUMINAMATH_CALUDE_sum_of_roots_l1368_136882

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1368_136882


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1368_136865

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x - 4*a > 0) →
  (-16 ≤ a ∧ a ≤ 0) →
  ∃ b : ℝ, (∀ x : ℝ, x^2 + b*x - 4*b > 0) ∧ (b < -16 ∨ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1368_136865


namespace NUMINAMATH_CALUDE_cone_volume_l1368_136855

theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 9) :
  let radius := Real.sqrt (slant_height^2 - height^2)
  (1 / 3 : ℝ) * π * radius^2 * height = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1368_136855


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1368_136831

theorem smallest_integer_with_remainders : 
  ∃ n : ℕ, 
    n > 1 ∧
    n % 3 = 2 ∧ 
    n % 7 = 2 ∧ 
    n % 5 = 1 ∧
    (∀ m : ℕ, m > 1 ∧ m % 3 = 2 ∧ m % 7 = 2 ∧ m % 5 = 1 → n ≤ m) ∧
    n = 86 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1368_136831


namespace NUMINAMATH_CALUDE_triangle_theorem_l1368_136861

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (Real.cos t.A * Real.cos t.B - Real.sin t.A * Real.sin t.B) = Real.cos (2 * t.C))
  (h2 : 2 * t.c = t.a + t.b)
  (h3 : t.a * t.b * Real.cos t.C = 18) :
  t.C = Real.pi / 3 ∧ t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1368_136861


namespace NUMINAMATH_CALUDE_andrew_sandwiches_l1368_136838

/-- The number of friends Andrew has coming over -/
def num_friends : ℕ := 4

/-- The number of sandwiches Andrew makes for each friend -/
def sandwiches_per_friend : ℕ := 3

/-- The total number of sandwiches Andrew made -/
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_sandwiches : total_sandwiches = 12 := by
  sorry

end NUMINAMATH_CALUDE_andrew_sandwiches_l1368_136838


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1368_136801

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1368_136801


namespace NUMINAMATH_CALUDE_total_legs_in_collection_l1368_136892

theorem total_legs_in_collection (num_ants num_spiders : ℕ) 
  (ant_legs spider_legs : ℕ) (h1 : num_ants = 12) (h2 : num_spiders = 8) 
  (h3 : ant_legs = 6) (h4 : spider_legs = 8) : 
  num_ants * ant_legs + num_spiders * spider_legs = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_collection_l1368_136892


namespace NUMINAMATH_CALUDE_max_value_of_f_l1368_136824

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-3/2) 2

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ interval, f a x ≤ 1) ∧
  (∃ x ∈ interval, f a x = 1) ↔
  (a = 3/4 ∨ a = (-3-2*Real.sqrt 2)/2) := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1368_136824


namespace NUMINAMATH_CALUDE_prob_even_product_is_four_fifths_l1368_136829

/-- Represents a spinner with a given set of numbers -/
structure Spinner where
  numbers : Finset ℕ

/-- The probability of selecting an even number from a spinner -/
def prob_even (s : Spinner) : ℚ :=
  (s.numbers.filter Even).card / s.numbers.card

/-- The probability of selecting an odd number from a spinner -/
def prob_odd (s : Spinner) : ℚ :=
  1 - prob_even s

/-- Spinner A with numbers 1 to 5 -/
def spinner_A : Spinner :=
  ⟨Finset.range 5 ∪ {5}⟩

/-- Spinner B with numbers 1, 2, 4 -/
def spinner_B : Spinner :=
  ⟨{1, 2, 4}⟩

theorem prob_even_product_is_four_fifths :
  1 - (prob_odd spinner_A * prob_odd spinner_B) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_product_is_four_fifths_l1368_136829


namespace NUMINAMATH_CALUDE_photo_difference_l1368_136889

theorem photo_difference (claire_photos : ℕ) (lisa_photos : ℕ) (robert_photos : ℕ) : 
  claire_photos = 12 →
  lisa_photos = 3 * claire_photos →
  robert_photos = lisa_photos →
  robert_photos - claire_photos = 24 := by sorry

end NUMINAMATH_CALUDE_photo_difference_l1368_136889


namespace NUMINAMATH_CALUDE_expression_simplification_l1368_136816

theorem expression_simplification (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 6) :
  (a - b)^2 + b*(3*a - b) - a^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1368_136816


namespace NUMINAMATH_CALUDE_exactly_two_consecutive_sets_sum_18_l1368_136858

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a ConsecutiveSet that sums to 18 -/
def sums_to_18 (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 18

theorem exactly_two_consecutive_sets_sum_18 :
  ∃! (sets : Finset ConsecutiveSet), (∀ s ∈ sets, sums_to_18 s) ∧ sets.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_consecutive_sets_sum_18_l1368_136858


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l1368_136884

def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_check : 
  (¬ canFormTriangle 3 5 10) ∧ 
  (canFormTriangle 5 4 8) ∧ 
  (¬ canFormTriangle 2 4 6) ∧ 
  (¬ canFormTriangle 3 3 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l1368_136884


namespace NUMINAMATH_CALUDE_complex_square_plus_four_l1368_136867

theorem complex_square_plus_four : 
  let i : ℂ := Complex.I
  (2 - 3*i)^2 + 4 = -1 - 12*i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_plus_four_l1368_136867


namespace NUMINAMATH_CALUDE_debby_hat_tickets_l1368_136879

/-- The number of tickets Debby spent on various items at the arcade -/
structure ArcadeTickets where
  total : ℕ
  stuffedAnimal : ℕ
  yoyo : ℕ
  hat : ℕ

/-- Theorem stating that given the conditions, Debby spent 2 tickets on the hat -/
theorem debby_hat_tickets (tickets : ArcadeTickets) 
    (h1 : tickets.total = 14)
    (h2 : tickets.stuffedAnimal = 10)
    (h3 : tickets.yoyo = 2)
    (h4 : tickets.total = tickets.stuffedAnimal + tickets.yoyo + tickets.hat) : 
  tickets.hat = 2 := by
  sorry

end NUMINAMATH_CALUDE_debby_hat_tickets_l1368_136879


namespace NUMINAMATH_CALUDE_sparrow_population_decrease_l1368_136880

/-- The annual decrease rate of the sparrow population -/
def decrease_rate : ℝ := 0.3

/-- The threshold percentage of the initial population -/
def threshold : ℝ := 0.2

/-- The remaining population fraction after one year -/
def remaining_fraction : ℝ := 1 - decrease_rate

/-- The number of years it takes for the population to fall below the threshold -/
def years_to_threshold : ℕ := 5

theorem sparrow_population_decrease :
  (remaining_fraction ^ years_to_threshold) < threshold ∧
  ∀ n : ℕ, n < years_to_threshold → (remaining_fraction ^ n) ≥ threshold :=
by sorry

end NUMINAMATH_CALUDE_sparrow_population_decrease_l1368_136880


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l1368_136850

/-- Represents the seating position of the k-th person -/
def seat_position (n k : ℕ) : ℕ := (k * (k - 1) / 2) % n

/-- Checks if all seating positions are distinct -/
def all_distinct_positions (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → seat_position n i ≠ seat_position n j

/-- Checks if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2^m

theorem seating_arrangement_theorem (n : ℕ) :
  n > 0 → (all_distinct_positions n ↔ is_power_of_two n) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l1368_136850


namespace NUMINAMATH_CALUDE_cos_pi_eighth_times_cos_five_pi_eighth_l1368_136836

theorem cos_pi_eighth_times_cos_five_pi_eighth :
  Real.cos (π / 8) * Real.cos (5 * π / 8) = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_eighth_times_cos_five_pi_eighth_l1368_136836


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l1368_136819

theorem fractional_exponent_simplification (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l1368_136819


namespace NUMINAMATH_CALUDE_mikes_land_profit_l1368_136864

/-- Calculates the profit from a land development project -/
def calculate_profit (total_acres : ℕ) (purchase_price_per_acre : ℕ) (sell_price_per_acre : ℕ) : ℕ :=
  let total_cost := total_acres * purchase_price_per_acre
  let acres_sold := total_acres / 2
  let total_revenue := acres_sold * sell_price_per_acre
  total_revenue - total_cost

/-- Proves that the profit from Mike's land development project is $6,000 -/
theorem mikes_land_profit :
  calculate_profit 200 70 200 = 6000 := by
  sorry

#eval calculate_profit 200 70 200

end NUMINAMATH_CALUDE_mikes_land_profit_l1368_136864


namespace NUMINAMATH_CALUDE_f_max_value_l1368_136862

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (x + Real.pi / 2) + Real.cos (Real.pi / 6 - x)

theorem f_max_value : ∀ x : ℝ, f x ≤ Real.sqrt 13 / 2 ∧ ∃ y : ℝ, f y = Real.sqrt 13 / 2 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1368_136862


namespace NUMINAMATH_CALUDE_star_difference_l1368_136800

def star (x y : ℝ) : ℝ := x * y - 3 * x + y

theorem star_difference : (star 6 5) - (star 5 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l1368_136800


namespace NUMINAMATH_CALUDE_three_pencils_two_pens_cost_l1368_136883

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The first condition: eight pencils and three pens cost $5.20 -/
axiom condition1 : 8 * pencil_cost + 3 * pen_cost = 5.20

/-- The second condition: two pencils and five pens cost $4.40 -/
axiom condition2 : 2 * pencil_cost + 5 * pen_cost = 4.40

/-- Theorem: The cost of three pencils and two pens is $2.5881 -/
theorem three_pencils_two_pens_cost : 
  3 * pencil_cost + 2 * pen_cost = 2.5881 := by sorry

end NUMINAMATH_CALUDE_three_pencils_two_pens_cost_l1368_136883


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l1368_136891

/-- Sum of digits function in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 consecutive natural numbers, there is always one whose sum of digits (in base 10) is divisible by 11 -/
theorem exists_sum_of_digits_div_11 (n : ℕ) : 
  ∃ k ∈ Finset.range 39, (sumOfDigits (n + k)) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l1368_136891


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1368_136877

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 7*x + 10 = 0

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_isosceles : base ≠ leg

-- Define the triangle with sides from the quadratic equation
def triangle_from_equation : IsoscelesTriangle :=
  { base := 2,
    leg := 5,
    is_isosceles := by norm_num }

-- State the theorem
theorem isosceles_triangle_perimeter :
  quadratic_equation triangle_from_equation.base ∧
  quadratic_equation triangle_from_equation.leg →
  triangle_from_equation.base + 2 * triangle_from_equation.leg = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1368_136877


namespace NUMINAMATH_CALUDE_perimeter_semicircular_arcs_square_l1368_136842

/-- The perimeter of a region bounded by four semicircular arcs, each constructed on the sides of a square with side length √2, is equal to 2π√2. -/
theorem perimeter_semicircular_arcs_square (side_length : ℝ) : 
  side_length = Real.sqrt 2 → 
  (4 : ℝ) * (π / 2 * side_length) = 2 * π * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_arcs_square_l1368_136842


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1368_136897

/-- Given a cube with face perimeter 20 cm, its volume is 125 cubic centimeters. -/
theorem cube_volume_from_face_perimeter :
  ∀ (cube : ℝ → ℝ), 
  (∃ (side : ℝ), side > 0 ∧ 4 * side = 20) →
  cube (20 / 4) = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1368_136897


namespace NUMINAMATH_CALUDE_total_ninja_stars_l1368_136826

-- Define the number of ninja throwing stars for each person
def eric_stars : ℕ := 4
def chad_stars_initial : ℕ := 2 * eric_stars
def jeff_bought : ℕ := 2
def jeff_stars_final : ℕ := 6

-- Define Chad's final number of stars
def chad_stars_final : ℕ := chad_stars_initial - jeff_bought

-- Theorem to prove
theorem total_ninja_stars :
  eric_stars + chad_stars_final + jeff_stars_final = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_ninja_stars_l1368_136826


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l1368_136822

theorem sum_of_squares_problem (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (eq1 : a^2 + a*b + b^2 = 1)
  (eq2 : b^2 + b*c + c^2 = 3)
  (eq3 : c^2 + c*a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l1368_136822


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1368_136832

theorem x_plus_2y_equals_10 (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : x + 2*y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1368_136832


namespace NUMINAMATH_CALUDE_image_of_two_zero_l1368_136825

/-- A mapping that transforms a point (x, y) into (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- The image of the point (2, 0) under the mapping f is (2, 2) -/
theorem image_of_two_zero :
  f (2, 0) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_zero_l1368_136825


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1368_136834

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 140 → divisor = 15 → quotient = 9 → 
  dividend = divisor * quotient + remainder → remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1368_136834


namespace NUMINAMATH_CALUDE_remainder_of_n_l1368_136843

theorem remainder_of_n (n : ℕ) (h1 : n^2 % 11 = 9) (h2 : n^3 % 11 = 5) : n % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_l1368_136843


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l1368_136856

-- Define the days of the week
inductive Day :=
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def add_days (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (add_days d m)

theorem tomorrow_is_saturday 
  (h : add_days (next_day (next_day Day.Wednesday)) 5 = Day.Monday) : 
  next_day Day.Friday = Day.Saturday :=
by
  sorry

#check tomorrow_is_saturday

end NUMINAMATH_CALUDE_tomorrow_is_saturday_l1368_136856


namespace NUMINAMATH_CALUDE_exists_counterexample_l1368_136874

-- Define the types for cards
inductive Letter : Type
| S | T | U

inductive Number : Type
| Two | Five | Seven | Eleven

-- Define a card as a pair of a Letter and a Number
def Card : Type := Letter × Number

-- Define the set of cards
def cards : List Card := [
  (Letter.S, Number.Two),
  (Letter.S, Number.Five),
  (Letter.S, Number.Seven),
  (Letter.S, Number.Eleven),
  (Letter.T, Number.Two),
  (Letter.T, Number.Five),
  (Letter.T, Number.Seven),
  (Letter.T, Number.Eleven),
  (Letter.U, Number.Two),
  (Letter.U, Number.Five),
  (Letter.U, Number.Seven),
  (Letter.U, Number.Eleven)
]

-- Define what a consonant is
def isConsonant (l : Letter) : Bool :=
  match l with
  | Letter.S => true
  | Letter.T => true
  | Letter.U => false

-- Define what a prime number is
def isPrime (n : Number) : Bool :=
  match n with
  | Number.Two => true
  | Number.Five => true
  | Number.Seven => true
  | Number.Eleven => true

-- Sam's statement
def samsStatement (c : Card) : Bool :=
  ¬(isConsonant c.1) ∨ isPrime c.2

-- Theorem to prove
theorem exists_counterexample :
  ∃ c ∈ cards, ¬(samsStatement c) :=
sorry

end NUMINAMATH_CALUDE_exists_counterexample_l1368_136874


namespace NUMINAMATH_CALUDE_mika_stickers_bought_l1368_136866

/-- The number of stickers Mika bought from the store -/
def stickers_bought : ℕ := 26

/-- The number of stickers Mika started with -/
def initial_stickers : ℕ := 20

/-- The number of stickers Mika got for her birthday -/
def birthday_stickers : ℕ := 20

/-- The number of stickers Mika gave to her sister -/
def stickers_given : ℕ := 6

/-- The number of stickers Mika used to decorate a greeting card -/
def stickers_used : ℕ := 58

/-- The number of stickers Mika is left with -/
def remaining_stickers : ℕ := 2

theorem mika_stickers_bought : 
  initial_stickers + birthday_stickers + stickers_bought = 
  stickers_given + stickers_used + remaining_stickers :=
sorry

end NUMINAMATH_CALUDE_mika_stickers_bought_l1368_136866


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1368_136899

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (2 * 360 : ℝ) = ((n - 2) * 180 : ℝ) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1368_136899


namespace NUMINAMATH_CALUDE_book_gain_percent_l1368_136812

theorem book_gain_percent (MP : ℝ) (CP : ℝ) (SP : ℝ) : 
  CP = 0.64 * MP →
  SP = 0.84 * MP →
  (SP - CP) / CP * 100 = 31.25 :=
by sorry

end NUMINAMATH_CALUDE_book_gain_percent_l1368_136812


namespace NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l1368_136815

/-- Two lines in a plane are parallel if they do not intersect. -/
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := ∀ p, p ∈ l₁ → p ∉ l₂

/-- A line segment is perpendicular to a line if it forms a right angle with the line. -/
def perpendicular (seg : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- The length of a line segment. -/
def length (seg : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: All perpendicular line segments between two parallel lines are equal in length. -/
theorem perpendicular_segments_equal_length 
  (l₁ l₂ : Set (ℝ × ℝ)) 
  (h_parallel : parallel l₁ l₂) 
  (seg₁ seg₂ : Set (ℝ × ℝ)) 
  (h_perp₁ : perpendicular seg₁ l₁ ∧ perpendicular seg₁ l₂)
  (h_perp₂ : perpendicular seg₂ l₁ ∧ perpendicular seg₂ l₂) :
  length seg₁ = length seg₂ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l1368_136815


namespace NUMINAMATH_CALUDE_adam_final_score_l1368_136813

def trivia_game (first_half_correct : ℕ) (second_half_correct : ℕ) 
                (first_half_points : ℕ) (second_half_points : ℕ) 
                (bonus_points : ℕ) (penalty : ℕ) (total_questions : ℕ) : ℕ :=
  let correct_points := first_half_correct * first_half_points + second_half_correct * second_half_points
  let total_correct := first_half_correct + second_half_correct
  let bonus := (total_correct / 3) * bonus_points
  let incorrect := total_questions - total_correct
  let penalty_points := incorrect * penalty
  correct_points + bonus - penalty_points

theorem adam_final_score : 
  trivia_game 15 12 3 5 2 1 35 = 115 := by sorry

end NUMINAMATH_CALUDE_adam_final_score_l1368_136813


namespace NUMINAMATH_CALUDE_number_of_stoplights_l1368_136810

-- Define the number of stoplights
variable (n : ℕ)

-- Define the time for the first route with all green lights
def green_time : ℕ := 10

-- Define the additional time for each red light
def red_light_delay : ℕ := 3

-- Define the time for the second route
def second_route_time : ℕ := 14

-- Define the additional time when all lights are red compared to the second route
def all_red_additional_time : ℕ := 5

-- Theorem statement
theorem number_of_stoplights :
  (green_time + n * red_light_delay = second_route_time + all_red_additional_time) →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_stoplights_l1368_136810
