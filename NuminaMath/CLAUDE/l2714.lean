import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2714_271490

theorem equation_solution (x : ℝ) : 
  x^6 - 22*x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5)/2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5)/2) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2714_271490


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2714_271428

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m - 3) * x^(m^2 - 7) - 4*x - 8 = a*x^2 + b*x + c) →
  (m - 3 ≠ 0) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2714_271428


namespace NUMINAMATH_CALUDE_height_less_than_sum_of_distances_l2714_271427

/-- Represents a triangle with three unequal sides -/
structure UnequalTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≠ b
  hbc : b ≠ c
  hac : a ≠ c
  longest_side : c > max a b

/-- The height to the longest side of the triangle -/
def height_to_longest_side (t : UnequalTriangle) : ℝ := sorry

/-- Distances from a point on the longest side to the other two sides -/
def distances_to_sides (t : UnequalTriangle) : ℝ × ℝ := sorry

theorem height_less_than_sum_of_distances (t : UnequalTriangle) :
  let x := height_to_longest_side t
  let (y, z) := distances_to_sides t
  x < y + z := by sorry

end NUMINAMATH_CALUDE_height_less_than_sum_of_distances_l2714_271427


namespace NUMINAMATH_CALUDE_dice_roll_probability_l2714_271432

def roll_probability : ℚ := 1 / 12

theorem dice_roll_probability :
  (probability_first_die_three * probability_second_die_odd = roll_probability) :=
by
  sorry

where
  probability_first_die_three : ℚ := 1 / 6
  probability_second_die_odd : ℚ := 1 / 2

end NUMINAMATH_CALUDE_dice_roll_probability_l2714_271432


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2714_271478

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (hc : c ≠ 0) :
  a / c = b / c → a = b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2714_271478


namespace NUMINAMATH_CALUDE_system_solution_l2714_271485

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = -1) ∧ (x + z = 0) ∧ (y + z = 1) ∧ (x = -1) ∧ (y = 0) ∧ (z = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2714_271485


namespace NUMINAMATH_CALUDE_product_equals_specific_number_l2714_271442

theorem product_equals_specific_number : 333333 * (333333 + 1) = 111111222222 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_specific_number_l2714_271442


namespace NUMINAMATH_CALUDE_average_weight_increase_l2714_271407

theorem average_weight_increase (initial_count : ℕ) (replaced_weight original_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 65 →
  original_weight = 85 →
  (original_weight - replaced_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2714_271407


namespace NUMINAMATH_CALUDE_min_operations_cube_l2714_271489

/-- Represents a rhombus configuration --/
structure RhombusConfig :=
  (n : ℕ)
  (rhombuses : ℕ)

/-- Represents a rearrangement operation --/
inductive RearrangementOp
  | insert
  | remove

/-- The minimum number of operations to transform the configuration --/
def min_operations (config : RhombusConfig) : ℕ :=
  config.n^3

/-- Theorem stating that the minimum number of operations is n³ --/
theorem min_operations_cube (config : RhombusConfig) 
  (h1 : config.rhombuses = 3 * config.n^2) :
  min_operations config = config.n^3 := by
  sorry

#check min_operations_cube

end NUMINAMATH_CALUDE_min_operations_cube_l2714_271489


namespace NUMINAMATH_CALUDE_brownies_remaining_l2714_271476

/-- Calculates the number of brownies left after consumption -/
def brownies_left (total : ℕ) (tina_daily : ℕ) (tina_days : ℕ) (husband_daily : ℕ) (husband_days : ℕ) (shared : ℕ) : ℕ :=
  total - (tina_daily * tina_days + husband_daily * husband_days + shared)

/-- Proves that given the specific consumption pattern, 5 brownies are left -/
theorem brownies_remaining :
  brownies_left 24 2 5 1 5 4 = 5 := by
  sorry

#eval brownies_left 24 2 5 1 5 4

end NUMINAMATH_CALUDE_brownies_remaining_l2714_271476


namespace NUMINAMATH_CALUDE_no_solutions_diophantine_equation_l2714_271499

theorem no_solutions_diophantine_equation :
  ¬∃ (n x y k : ℕ), n ≥ 1 ∧ x > 0 ∧ y > 0 ∧ k > 1 ∧ 
  Nat.gcd x y = 1 ∧ 3^n = x^k + y^k :=
sorry

end NUMINAMATH_CALUDE_no_solutions_diophantine_equation_l2714_271499


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2714_271406

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) →
  (c = b + 1) →
  (a + c = 140) →
  (b - a = 2) →
  (a + b + c = 210) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2714_271406


namespace NUMINAMATH_CALUDE_puppies_adopted_l2714_271446

/-- The cost to get a cat ready for adoption -/
def cat_cost : ℕ := 50

/-- The cost to get an adult dog ready for adoption -/
def adult_dog_cost : ℕ := 100

/-- The cost to get a puppy ready for adoption -/
def puppy_cost : ℕ := 150

/-- The number of cats adopted -/
def cats_adopted : ℕ := 2

/-- The number of adult dogs adopted -/
def adult_dogs_adopted : ℕ := 3

/-- The total cost for all adopted animals -/
def total_cost : ℕ := 700

/-- Theorem stating that the number of puppies adopted is 2 -/
theorem puppies_adopted : 
  ∃ (p : ℕ), p = 2 ∧ 
  cat_cost * cats_adopted + adult_dog_cost * adult_dogs_adopted + puppy_cost * p = total_cost :=
sorry

end NUMINAMATH_CALUDE_puppies_adopted_l2714_271446


namespace NUMINAMATH_CALUDE_temperature_conversion_l2714_271444

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 221 → t = 105 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2714_271444


namespace NUMINAMATH_CALUDE_find_k_l2714_271455

theorem find_k : ∃ k : ℕ, 3 * 10 * 4 * k = Nat.factorial 9 ∧ k = 15120 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2714_271455


namespace NUMINAMATH_CALUDE_exception_pair_of_equations_other_pairs_valid_l2714_271424

theorem exception_pair_of_equations (x : ℝ) : 
  (∃ y, y = x ∧ y = x - 2 ∧ x^2 - 2*x = 0) ↔ False :=
by sorry

theorem other_pairs_valid (x : ℝ) :
  ((∃ y, y = x^2 ∧ y = 2*x ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 2*x ∧ y = 0 ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 2*x + 1 ∧ y = 1 ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 1 ∧ y = 2*x - 1 ∧ x^2 - 2*x = 0)) ↔ True :=
by sorry

end NUMINAMATH_CALUDE_exception_pair_of_equations_other_pairs_valid_l2714_271424


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2714_271438

theorem sum_of_numbers_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (a : ℚ) / b = 5 →
  (c : ℚ) / b = 4 →
  c = 400 →
  a + b + c = 1000 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2714_271438


namespace NUMINAMATH_CALUDE_ninth_power_five_and_eleventh_power_five_l2714_271496

theorem ninth_power_five_and_eleventh_power_five :
  9^5 = 59149 ∧ 11^5 = 161051 := by
  sorry

#check ninth_power_five_and_eleventh_power_five

end NUMINAMATH_CALUDE_ninth_power_five_and_eleventh_power_five_l2714_271496


namespace NUMINAMATH_CALUDE_sexual_reproduction_genetic_diversity_l2714_271484

/-- Represents a set of genes -/
def GeneticMaterial : Type := Set Nat

/-- Represents an organism with genetic material -/
structure Organism :=
  (genes : GeneticMaterial)

/-- Represents the process of meiosis -/
def meiosis (parent : Organism) : GeneticMaterial :=
  sorry

/-- Represents the process of fertilization -/
def fertilization (gamete1 gamete2 : GeneticMaterial) : Organism :=
  sorry

/-- Theorem stating that sexual reproduction produces offspring with different genetic combinations -/
theorem sexual_reproduction_genetic_diversity 
  (parent1 parent2 : Organism) : 
  ∃ (offspring : Organism), 
    offspring = fertilization (meiosis parent1) (meiosis parent2) ∧
    offspring.genes ≠ parent1.genes ∧
    offspring.genes ≠ parent2.genes :=
  sorry

end NUMINAMATH_CALUDE_sexual_reproduction_genetic_diversity_l2714_271484


namespace NUMINAMATH_CALUDE_malou_average_score_l2714_271494

def malou_quiz_scores : List ℝ := [91, 90, 92]

theorem malou_average_score : 
  (malou_quiz_scores.sum / malou_quiz_scores.length : ℝ) = 91 := by
  sorry

end NUMINAMATH_CALUDE_malou_average_score_l2714_271494


namespace NUMINAMATH_CALUDE_bunny_burrow_exits_l2714_271415

/-- The number of times a bunny comes out of its burrow per minute -/
def bunny_rate : ℕ := 3

/-- The number of bunnies -/
def num_bunnies : ℕ := 20

/-- The time period in hours -/
def time_period : ℕ := 10

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem bunny_burrow_exits :
  bunny_rate * minutes_per_hour * time_period * num_bunnies = 36000 := by
  sorry

end NUMINAMATH_CALUDE_bunny_burrow_exits_l2714_271415


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l2714_271417

theorem circle_equation_k_value (x y k : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 14*y - k = 0 ↔ (x + 4)^2 + (y + 7)^2 = 25) → 
  k = -40 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l2714_271417


namespace NUMINAMATH_CALUDE_eccentricity_equation_roots_l2714_271483

/-- The cubic equation whose roots are the eccentricities of a hyperbola, an ellipse, and a parabola -/
def eccentricity_equation (x : ℝ) : Prop :=
  2 * x^3 - 7 * x^2 + 7 * x - 2 = 0

/-- Definition of eccentricity for an ellipse -/
def is_ellipse_eccentricity (e : ℝ) : Prop :=
  0 ≤ e ∧ e < 1

/-- Definition of eccentricity for a parabola -/
def is_parabola_eccentricity (e : ℝ) : Prop :=
  e = 1

/-- Definition of eccentricity for a hyperbola -/
def is_hyperbola_eccentricity (e : ℝ) : Prop :=
  e > 1

/-- The theorem stating that the roots of the equation correspond to the eccentricities of the three conic sections -/
theorem eccentricity_equation_roots :
  ∃ (e₁ e₂ e₃ : ℝ),
    eccentricity_equation e₁ ∧
    eccentricity_equation e₂ ∧
    eccentricity_equation e₃ ∧
    is_ellipse_eccentricity e₁ ∧
    is_parabola_eccentricity e₂ ∧
    is_hyperbola_eccentricity e₃ :=
  sorry

end NUMINAMATH_CALUDE_eccentricity_equation_roots_l2714_271483


namespace NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l2714_271412

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (planeParallelPlane : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_necessary_not_sufficient
  (α β : Plane) (l : Line) (h : subset l α) :
  (lineParallelPlane l β → planeParallelPlane α β) ∧
  ¬(planeParallelPlane α β → lineParallelPlane l β) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l2714_271412


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_divisors_greater_than_sqrt_l2714_271486

theorem arithmetic_mean_of_divisors_greater_than_sqrt (n : ℕ) (hn : n > 1) :
  let divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList
  (divisors.sum / divisors.length : ℝ) > Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_divisors_greater_than_sqrt_l2714_271486


namespace NUMINAMATH_CALUDE_alternating_subtraction_theorem_l2714_271461

def alternating_subtraction (n : ℕ) : ℤ :=
  if n % 2 = 0 then 0 else -1

theorem alternating_subtraction_theorem (n : ℕ) :
  alternating_subtraction n = if n % 2 = 0 then 0 else -1 :=
by sorry

-- Examples for the given cases
example : alternating_subtraction 1989 = -1 :=
by sorry

example : alternating_subtraction 1990 = 0 :=
by sorry

end NUMINAMATH_CALUDE_alternating_subtraction_theorem_l2714_271461


namespace NUMINAMATH_CALUDE_shirt_tie_outfits_l2714_271448

theorem shirt_tie_outfits (num_shirts : ℕ) (num_ties : ℕ) 
  (h1 : num_shirts = 6) (h2 : num_ties = 5) : 
  num_shirts * num_ties = 30 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_outfits_l2714_271448


namespace NUMINAMATH_CALUDE_savings_calculation_l2714_271487

theorem savings_calculation (savings : ℚ) : 
  (1 / 2 : ℚ) * savings = 300 → savings = 600 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l2714_271487


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2714_271423

theorem min_value_sum_of_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_3 : a + b + c = 3) :
  1 / (a + b)^2 + 1 / (a + c)^2 + 1 / (b + c)^2 ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2714_271423


namespace NUMINAMATH_CALUDE_race_track_width_l2714_271430

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 440 →
  outer_radius = 84.02817496043394 →
  ∃ width : ℝ, abs (width - 14.02056077700854) < 1e-10 ∧
    width = outer_radius - inner_circumference / (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_race_track_width_l2714_271430


namespace NUMINAMATH_CALUDE_trig_expression_equality_l2714_271471

theorem trig_expression_equality (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π)  -- α is in the second quadrant
  (h2 : Real.sin (π/2 + α) = -Real.sqrt 5 / 5) :
  (Real.cos α ^ 3 + Real.sin α) / Real.cos (α - π/4) = 9 * Real.sqrt 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l2714_271471


namespace NUMINAMATH_CALUDE_yz_circle_radius_l2714_271418

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the circle in xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xy-plane -/
  xy_radius : ℝ
  /-- Center of the circle in yz-plane -/
  yz_center : ℝ × ℝ × ℝ

/-- Theorem: The radius of the circle formed by the intersection of the sphere and the yz-plane -/
theorem yz_circle_radius (s : IntersectingSphere) 
  (h_xy : s.xy_center = (3, 5, -2) ∧ s.xy_radius = 3)
  (h_yz : s.yz_center = (-2, 5, 3)) :
  ∃ r : ℝ, r = Real.sqrt 46 ∧ 
  r = Real.sqrt ((Real.sqrt 50 : ℝ) ^ 2 - 2 ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_yz_circle_radius_l2714_271418


namespace NUMINAMATH_CALUDE_electronic_items_loss_percentage_l2714_271436

/-- Calculate the overall loss percentage for three electronic items -/
theorem electronic_items_loss_percentage :
  let cost_prices : List ℚ := [1500, 2500, 800]
  let sale_prices : List ℚ := [1275, 2300, 700]
  let total_cost := cost_prices.sum
  let total_sale := sale_prices.sum
  let loss := total_cost - total_sale
  let loss_percentage := (loss / total_cost) * 100
  loss_percentage = 10.9375 := by
  sorry

end NUMINAMATH_CALUDE_electronic_items_loss_percentage_l2714_271436


namespace NUMINAMATH_CALUDE_correct_number_proof_l2714_271410

theorem correct_number_proof (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_error : ℚ) (correct_second : ℚ) : 
  n = 10 → 
  initial_avg = 40.2 → 
  correct_avg = 40.3 → 
  first_error = 19 → 
  second_error = 13 → 
  (n : ℚ) * initial_avg - first_error - second_error + correct_second = (n : ℚ) * correct_avg → 
  correct_second = 33 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_proof_l2714_271410


namespace NUMINAMATH_CALUDE_investment_ratio_problem_l2714_271454

theorem investment_ratio_problem (profit_ratio_p profit_ratio_q : ℚ) 
  (investment_time_p investment_time_q : ℚ) 
  (investment_ratio_p investment_ratio_q : ℚ) : 
  profit_ratio_p / profit_ratio_q = 7 / 11 →
  investment_time_p = 5 →
  investment_time_q = 10.999999999999998 →
  (investment_ratio_p * investment_time_p) / (investment_ratio_q * investment_time_q) = profit_ratio_p / profit_ratio_q →
  investment_ratio_p / investment_ratio_q = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_problem_l2714_271454


namespace NUMINAMATH_CALUDE_brokerage_percentage_calculation_l2714_271429

/-- The brokerage percentage calculation problem -/
theorem brokerage_percentage_calculation
  (cash_realized : ℝ)
  (total_amount : ℝ)
  (h1 : cash_realized = 106.25)
  (h2 : total_amount = 106) :
  let brokerage_amount := cash_realized - total_amount
  let brokerage_percentage := (brokerage_amount / total_amount) * 100
  ∃ ε > 0, abs (brokerage_percentage - 0.236) < ε :=
by sorry

end NUMINAMATH_CALUDE_brokerage_percentage_calculation_l2714_271429


namespace NUMINAMATH_CALUDE_money_conditions_l2714_271413

theorem money_conditions (a b : ℝ) 
  (h1 : 6 * a - b = 45)
  (h2 : 4 * a + b > 60) : 
  a > 10.5 ∧ b > 18 := by
sorry

end NUMINAMATH_CALUDE_money_conditions_l2714_271413


namespace NUMINAMATH_CALUDE_divisibility_condition_l2714_271419

theorem divisibility_condition (a b : ℤ) (ha : a ≥ 3) (hb : b ≥ 3) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2714_271419


namespace NUMINAMATH_CALUDE_associated_functions_range_l2714_271408

/-- Two functions are associated on an interval if their difference has two distinct zeros in that interval. -/
def associated_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b ∧ f x₁ = g x₁ ∧ f x₂ = g x₂

/-- The statement of the problem. -/
theorem associated_functions_range (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 4
  let g : ℝ → ℝ := λ x ↦ 2*x + m
  associated_functions f g 0 3 → -9/4 < m ∧ m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_associated_functions_range_l2714_271408


namespace NUMINAMATH_CALUDE_biff_break_even_hours_l2714_271403

/-- Calculates the number of hours required to break even on a bus trip. -/
def hours_to_break_even (ticket_cost snacks_cost headphones_cost hourly_rate wifi_cost : ℚ) : ℚ :=
  let total_expenses := ticket_cost + snacks_cost + headphones_cost
  let net_hourly_rate := hourly_rate - wifi_cost
  total_expenses / net_hourly_rate

/-- Proves that given Biff's expenses and earnings, the number of hours required to break even on a bus trip is 3 hours. -/
theorem biff_break_even_hours :
  hours_to_break_even 11 3 16 12 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_biff_break_even_hours_l2714_271403


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2714_271416

-- Problem 1
theorem problem_1 : -105 - (-112) + 20 + 18 = 45 := by
  sorry

-- Problem 2
theorem problem_2 : 13 + (-22) - 25 - (-18) = -16 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2714_271416


namespace NUMINAMATH_CALUDE_insurance_payment_percentage_l2714_271435

theorem insurance_payment_percentage
  (total_cost : ℝ)
  (individual_payment_percentage : ℝ)
  (individual_payment : ℝ)
  (h1 : total_cost = 110000)
  (h2 : individual_payment_percentage = 20)
  (h3 : individual_payment = 22000)
  (h4 : individual_payment = (individual_payment_percentage / 100) * total_cost) :
  100 - individual_payment_percentage = 80 := by
  sorry

end NUMINAMATH_CALUDE_insurance_payment_percentage_l2714_271435


namespace NUMINAMATH_CALUDE_z_share_per_x_rupee_l2714_271451

/-- Given a total amount divided among three parties x, y, and z, 
    this theorem proves the ratio of z's share to x's share. -/
theorem z_share_per_x_rupee 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 78) 
  (h2 : y_share = 18) 
  (h3 : y_share = 0.45 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  z_share / x_share = 0.5 := by
sorry

end NUMINAMATH_CALUDE_z_share_per_x_rupee_l2714_271451


namespace NUMINAMATH_CALUDE_train_platform_problem_l2714_271469

/-- The length of a train in meters. -/
def train_length : ℝ := 110

/-- The time taken to cross the first platform in seconds. -/
def time_first : ℝ := 15

/-- The time taken to cross the second platform in seconds. -/
def time_second : ℝ := 20

/-- The length of the second platform in meters. -/
def second_platform_length : ℝ := 250

/-- The length of the first platform in meters. -/
def first_platform_length : ℝ := 160

theorem train_platform_problem :
  (train_length + first_platform_length) / time_first =
  (train_length + second_platform_length) / time_second :=
sorry

end NUMINAMATH_CALUDE_train_platform_problem_l2714_271469


namespace NUMINAMATH_CALUDE_rectangle_configuration_exists_l2714_271420

/-- Represents a rectangle with vertical and horizontal sides -/
structure Rectangle where
  x : ℝ × ℝ  -- x-coordinates of left and right sides
  y : ℝ × ℝ  -- y-coordinates of bottom and top sides

/-- Checks if two rectangles meet (have at least one point in common) -/
def rectangles_meet (r1 r2 : Rectangle) : Prop :=
  (r1.x.1 ≤ r2.x.2 ∧ r2.x.1 ≤ r1.x.2) ∧ (r1.y.1 ≤ r2.y.2 ∧ r2.y.1 ≤ r1.y.2)

/-- Checks if two rectangles follow each other based on their indices -/
def rectangles_follow (i j n : ℕ) : Prop :=
  i % n = (j + 1) % n ∨ j % n = (i + 1) % n

/-- Represents a valid configuration of n rectangles -/
def valid_configuration (n : ℕ) (rectangles : Fin n → Rectangle) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    rectangles_meet (rectangles i) (rectangles j) ↔ ¬rectangles_follow i.val j.val n

/-- The main theorem stating that a valid configuration exists if and only if n ≤ 5 -/
theorem rectangle_configuration_exists (n : ℕ) (h : n ≥ 1) :
  (∃ rectangles : Fin n → Rectangle, valid_configuration n rectangles) ↔ n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_rectangle_configuration_exists_l2714_271420


namespace NUMINAMATH_CALUDE_exemplary_sequences_count_l2714_271439

/-- The number of distinct 6-letter sequences from "EXEMPLARY" with given conditions -/
def exemplary_sequences : ℕ :=
  let available_letters := 6  -- X, A, M, P, L, R
  let positions_to_fill := 4  -- positions 2, 3, 4, 5
  Nat.factorial available_letters / Nat.factorial (available_letters - positions_to_fill)

/-- Theorem stating the number of distinct sequences is 360 -/
theorem exemplary_sequences_count :
  exemplary_sequences = 360 := by
  sorry

#eval exemplary_sequences  -- Should output 360

end NUMINAMATH_CALUDE_exemplary_sequences_count_l2714_271439


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2714_271466

/-- The sine of 45 degrees is equal to √2/2 -/
theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2714_271466


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l2714_271457

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l2714_271457


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_product_l2714_271477

/-- Represents a conic section (ellipse or hyperbola) -/
structure Conic where
  center : ℝ × ℝ
  foci : ℝ × ℝ
  eccentricity : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem ellipse_hyperbola_eccentricity_product (C₁ C₂ : Conic) (P : Point) :
  C₁.center = (0, 0) →
  C₂.center = (0, 0) →
  C₁.foci.1 < 0 →
  C₁.foci.2 > 0 →
  C₂.foci = C₁.foci →
  P.x > 0 →
  P.y > 0 →
  (P.x - C₁.foci.1)^2 + P.y^2 = (P.x - C₁.foci.2)^2 + P.y^2 →
  C₁.eccentricity * C₂.eccentricity > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_product_l2714_271477


namespace NUMINAMATH_CALUDE_s_99_digits_l2714_271434

/-- s(n) is the n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- count_digits n returns the number of digits in the natural number n -/
def count_digits (n : ℕ) : ℕ := sorry

/-- The theorem states that s(99) has 189 digits -/
theorem s_99_digits : count_digits (s 99) = 189 := by sorry

end NUMINAMATH_CALUDE_s_99_digits_l2714_271434


namespace NUMINAMATH_CALUDE_smallest_angle_cosine_equality_l2714_271440

theorem smallest_angle_cosine_equality (θ : Real) : 
  (θ > 0) →
  (Real.cos θ = Real.sin (π/4) + Real.cos (π/3) - Real.sin (π/6) - Real.cos (π/12)) →
  (θ = π/6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_cosine_equality_l2714_271440


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2714_271411

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define what it means for two circles to be externally tangent
def externally_tangent (C₁ C₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧
  ∀ (x' y' : ℝ), (C₁ x' y' ∧ C₂ x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_externally_tangent : externally_tangent C₁ C₂ :=
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l2714_271411


namespace NUMINAMATH_CALUDE_age_sum_is_21_l2714_271437

/-- Given two people p and q, where 6 years ago p was half the age of q,
    and the ratio of their present ages is 3:4, prove that the sum of
    their present ages is 21 years. -/
theorem age_sum_is_21 (p q : ℕ) : 
  (p - 6 = (q - 6) / 2) →  -- 6 years ago, p was half of q in age
  (p : ℚ) / q = 3 / 4 →    -- The ratio of their present ages is 3:4
  p + q = 21 :=            -- The sum of their present ages is 21
by sorry

end NUMINAMATH_CALUDE_age_sum_is_21_l2714_271437


namespace NUMINAMATH_CALUDE_division_value_problem_l2714_271400

theorem division_value_problem (x : ℝ) : 
  (1152 / x) - 189 = 3 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_division_value_problem_l2714_271400


namespace NUMINAMATH_CALUDE_melissa_games_played_l2714_271456

/-- Given that Melissa scored 12 points in each game and a total of 36 points,
    prove that she played 3 games. -/
theorem melissa_games_played (points_per_game : ℕ) (total_points : ℕ) 
  (h1 : points_per_game = 12) 
  (h2 : total_points = 36) : 
  total_points / points_per_game = 3 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l2714_271456


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2714_271470

/-- The sum of the infinite series ∑(n=1 to ∞) (5n-2)/(3^n) is equal to 11/4 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (5 * n - 2 : ℝ) / (3 ^ n)) = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2714_271470


namespace NUMINAMATH_CALUDE_quadratic_roots_and_exponential_inequality_l2714_271481

theorem quadratic_roots_and_exponential_inequality (a : ℝ) :
  (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ∧ 
  (∀ x : ℝ, Real.exp x + 1 / Real.exp x > a) →
  -4 ≤ a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_exponential_inequality_l2714_271481


namespace NUMINAMATH_CALUDE_converse_and_inverse_true_l2714_271462

-- Define the properties
def is_circle (shape : Type) : Prop := sorry
def has_constant_curvature (shape : Type) : Prop := sorry

-- Given statement
axiom circle_implies_constant_curvature : 
  ∀ (shape : Type), is_circle shape → has_constant_curvature shape

-- Theorem to prove
theorem converse_and_inverse_true : 
  (∀ (shape : Type), has_constant_curvature shape → is_circle shape) ∧ 
  (∀ (shape : Type), ¬is_circle shape → ¬has_constant_curvature shape) := by
  sorry

end NUMINAMATH_CALUDE_converse_and_inverse_true_l2714_271462


namespace NUMINAMATH_CALUDE_class_average_l2714_271402

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (high_score : ℕ) 
  (zero_scorers : ℕ) (rest_average : ℕ) : 
  total_students = 27 →
  high_scorers = 5 →
  high_score = 95 →
  zero_scorers = 3 →
  rest_average = 45 →
  (total_students - high_scorers - zero_scorers) * rest_average + 
    high_scorers * high_score = 1330 →
  (1330 : ℚ) / total_students = 1330 / 27 := by
sorry

end NUMINAMATH_CALUDE_class_average_l2714_271402


namespace NUMINAMATH_CALUDE_manager_selection_problem_l2714_271479

theorem manager_selection_problem (n m k : ℕ) (h1 : n = 7) (h2 : m = 4) (h3 : k = 2) :
  (Nat.choose n m) - (Nat.choose (n - k) (m - k)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_manager_selection_problem_l2714_271479


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l2714_271465

theorem right_angled_triangle_set : ∃ (a b c : ℝ), 
  (a = Real.sqrt 2 ∧ b = Real.sqrt 3 ∧ c = Real.sqrt 5) ∧ 
  a^2 + b^2 = c^2 ∧ 
  (∀ (x y z : ℝ), 
    ((x = Real.sqrt 3 ∧ y = 2 ∧ z = Real.sqrt 5) ∨ 
     (x = 3 ∧ y = 4 ∧ z = 5) ∨ 
     (x = 1 ∧ y = 2 ∧ z = 3)) → 
    x^2 + y^2 ≠ z^2) :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l2714_271465


namespace NUMINAMATH_CALUDE_sin_equality_implies_zero_l2714_271425

theorem sin_equality_implies_zero (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) 
  (h3 : Real.sin (n * π / 180) = Real.sin (720 * π / 180)) : n = 0 :=
by sorry

end NUMINAMATH_CALUDE_sin_equality_implies_zero_l2714_271425


namespace NUMINAMATH_CALUDE_number_multiplication_problem_l2714_271472

theorem number_multiplication_problem (x : ℝ) : 15 * x = x + 196 → 15 * x = 210 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_problem_l2714_271472


namespace NUMINAMATH_CALUDE_fraction_modification_l2714_271473

theorem fraction_modification (a b c d x : ℚ) : 
  a ≠ b →
  b ≠ 0 →
  (2 * a + x) / (3 * b + x) = c / d →
  ∃ (k₁ k₂ : ℚ), c = k₁ * x ∧ d = k₂ * x →
  x = (3 * b * c - 2 * a * d) / (d - c) := by
sorry

end NUMINAMATH_CALUDE_fraction_modification_l2714_271473


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2714_271409

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 60,
    prove that the second term is 8. -/
theorem geometric_sequence_second_term :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio q = 2
  (a 1 + a 2 + a 3 + a 4 = 60) →  -- Sum of first 4 terms S_4 = 60
  a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2714_271409


namespace NUMINAMATH_CALUDE_gcd_153_119_l2714_271491

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l2714_271491


namespace NUMINAMATH_CALUDE_ourDie_expected_value_l2714_271464

/-- Represents the four-sided die with its probabilities and winnings --/
structure UnusualDie where
  side1_prob : ℚ
  side1_win : ℚ
  side2_prob : ℚ
  side2_win : ℚ
  side3_prob : ℚ
  side3_win : ℚ
  side4_prob : ℚ
  side4_win : ℚ

/-- The specific unusual die described in the problem --/
def ourDie : UnusualDie :=
  { side1_prob := 1/4
  , side1_win := 2
  , side2_prob := 1/4
  , side2_win := 4
  , side3_prob := 1/3
  , side3_win := -6
  , side4_prob := 1/6
  , side4_win := 0 }

/-- Calculates the expected value of rolling the die --/
def expectedValue (d : UnusualDie) : ℚ :=
  d.side1_prob * d.side1_win +
  d.side2_prob * d.side2_win +
  d.side3_prob * d.side3_win +
  d.side4_prob * d.side4_win

/-- Theorem stating that the expected value of rolling ourDie is -1/2 --/
theorem ourDie_expected_value :
  expectedValue ourDie = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ourDie_expected_value_l2714_271464


namespace NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l2714_271480

/-- Chandler's wrapping paper sales problem -/
theorem chandler_wrapping_paper_sales 
  (total_goal : ℕ) 
  (sold_to_grandmother : ℕ) 
  (sold_to_uncle : ℕ) 
  (sold_to_neighbor : ℕ) 
  (h1 : total_goal = 12)
  (h2 : sold_to_grandmother = 3)
  (h3 : sold_to_uncle = 4)
  (h4 : sold_to_neighbor = 3) :
  total_goal - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 2 :=
by sorry

end NUMINAMATH_CALUDE_chandler_wrapping_paper_sales_l2714_271480


namespace NUMINAMATH_CALUDE_largest_non_sum_42multiple_composite_l2714_271474

def is_composite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_sum_of_42multiple_and_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 42 * a + b ∧ a > 0 ∧ is_composite b

theorem largest_non_sum_42multiple_composite :
  (∀ n : ℕ, n > 215 → is_sum_of_42multiple_and_composite n) ∧
  ¬is_sum_of_42multiple_and_composite 215 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_42multiple_composite_l2714_271474


namespace NUMINAMATH_CALUDE_scientific_notation_of_1206_million_l2714_271498

theorem scientific_notation_of_1206_million : 
  ∃ (a : ℝ) (n : ℤ), 1206000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.206 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1206_million_l2714_271498


namespace NUMINAMATH_CALUDE_triangle_ABC_theorem_l2714_271426

open Real

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π

theorem triangle_ABC_theorem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_eq : a * sin B - Real.sqrt 3 * b * cos A = 0) :
  A = π / 3 ∧ 
  (a = 3 → 
    (∃ (max_area : ℝ), max_area = 9 * Real.sqrt 3 / 4 ∧
      ∀ (b' c' : ℝ), triangle_ABC 3 b' c' A B C → 
        1/2 * 3 * b' * sin A ≤ max_area ∧
        (1/2 * 3 * b' * sin A = max_area → b' = 3 ∧ c' = 3))) :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_theorem_l2714_271426


namespace NUMINAMATH_CALUDE_prob_three_red_cards_standard_deck_l2714_271460

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- A standard deck of cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing three red cards in succession from a standard deck -/
def prob_three_red_cards (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit : ℚ) / d.total_cards *
  ((d.red_suits * d.cards_per_suit - 1) : ℚ) / (d.total_cards - 1) *
  ((d.red_suits * d.cards_per_suit - 2) : ℚ) / (d.total_cards - 2)

/-- Theorem: The probability of drawing three red cards in succession from a standard deck is 2/17 -/
theorem prob_three_red_cards_standard_deck :
  prob_three_red_cards standard_deck = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_cards_standard_deck_l2714_271460


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l2714_271414

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_exponential_equation :
  (¬ ∃ x : ℝ, Real.exp x = x - 1) ↔ (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l2714_271414


namespace NUMINAMATH_CALUDE_largest_digit_sum_quotient_l2714_271433

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

theorem largest_digit_sum_quotient :
  (∀ n : ThreeDigitNumber, (value n : ℚ) / (digitSum n : ℚ) ≤ 100) ∧
  (∃ n : ThreeDigitNumber, (value n : ℚ) / (digitSum n : ℚ) = 100) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_sum_quotient_l2714_271433


namespace NUMINAMATH_CALUDE_second_mechanic_rate_calculation_l2714_271452

/-- Represents the hourly rate of the second mechanic -/
def second_mechanic_rate : ℝ := sorry

/-- The first mechanic's hourly rate -/
def first_mechanic_rate : ℝ := 45

/-- Total combined work hours -/
def total_hours : ℝ := 20

/-- Total charge for both mechanics -/
def total_charge : ℝ := 1100

/-- Hours worked by the second mechanic -/
def second_mechanic_hours : ℝ := 5

theorem second_mechanic_rate_calculation : 
  second_mechanic_rate = 85 :=
by
  sorry

#check second_mechanic_rate_calculation

end NUMINAMATH_CALUDE_second_mechanic_rate_calculation_l2714_271452


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l2714_271443

theorem greatest_power_of_two (n : ℕ) : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (10^1002 - 4^501) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (10^1002 - 4^501) → m ≤ k :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l2714_271443


namespace NUMINAMATH_CALUDE_rosencrantz_win_probability_value_l2714_271404

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the state of the game -/
inductive GameState
| InProgress
| RosencrantzWins
| GuildensternWins

/-- Represents the game rules -/
def game_rules : List CoinFlip → GameState :=
  sorry

/-- The probability of Rosencrantz winning the game -/
def rosencrantz_win_probability : ℚ :=
  sorry

/-- Theorem stating the probability of Rosencrantz winning -/
theorem rosencrantz_win_probability_value :
  rosencrantz_win_probability = (2^2009 - 1) / (3 * 2^2008 - 1) :=
sorry

end NUMINAMATH_CALUDE_rosencrantz_win_probability_value_l2714_271404


namespace NUMINAMATH_CALUDE_subtract_negative_add_l2714_271459

theorem subtract_negative_add : 3 - (-5) + 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_add_l2714_271459


namespace NUMINAMATH_CALUDE_f_properties_l2714_271482

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^x / (a^x + 1)

-- Main theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- 1. The range of f(x) is (0, 1)
  (∀ x, 0 < f a x ∧ f a x < 1) ∧
  -- 2. If the maximum value of f(x) on [-1, 2] is 3/4, then a = √3 or a = 1/3
  (Set.Icc (-1) 2 ⊆ f a ⁻¹' Set.Iio (3/4) → a = Real.sqrt 3 ∨ a = 1/3) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2714_271482


namespace NUMINAMATH_CALUDE_book_length_proof_l2714_271422

theorem book_length_proof (pages_read : ℕ) (pages_difference : ℕ) : 
  pages_read = 2323 → pages_difference = 90 → 
  pages_read = (pages_read - pages_difference) + pages_difference → 
  pages_read + (pages_read - pages_difference) = 4556 :=
by
  sorry

end NUMINAMATH_CALUDE_book_length_proof_l2714_271422


namespace NUMINAMATH_CALUDE_function_properties_l2714_271492

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = a * x^2 - (a + 2) * x + Real.log x) →
  (a = 1 → ∀ x : ℝ, x > 0 → (f 1 x - f 1 1) = 0 * (x - 1)) ∧
  (a > 0 → (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) → (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x = -2) → a ≥ 1) ∧
  ((∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a x₁ + 2 * x₁ - (f a x₂ + 2 * x₂)) / (x₁ - x₂) > 0) → 0 ≤ a ∧ a ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2714_271492


namespace NUMINAMATH_CALUDE_semicircle_tangent_circle_and_triangle_l2714_271447

/-- Given a semicircle with diameter AB and center O, where AO = OB = R,
    and two semicircles drawn over AO and BO, this theorem proves:
    1. The radius of the circle tangent to all three semicircles is R/3
    2. The sides of the triangle formed by the tangency points are 2R/5 and (R/5)√10 -/
theorem semicircle_tangent_circle_and_triangle (R : ℝ) (R_pos : R > 0) :
  ∃ (r a b : ℝ),
    r = R / 3 ∧
    2 * a = 2 * R / 5 ∧
    b = (R / 5) * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_tangent_circle_and_triangle_l2714_271447


namespace NUMINAMATH_CALUDE_gift_wrapping_l2714_271441

theorem gift_wrapping (total_gifts : ℕ) (total_rolls : ℕ) (first_roll_gifts : ℕ) (third_roll_gifts : ℕ) :
  total_gifts = 12 →
  total_rolls = 3 →
  first_roll_gifts = 3 →
  third_roll_gifts = 4 →
  ∃ (second_roll_gifts : ℕ),
    first_roll_gifts + second_roll_gifts + third_roll_gifts = total_gifts ∧
    second_roll_gifts = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_l2714_271441


namespace NUMINAMATH_CALUDE_painters_rooms_theorem_l2714_271488

/-- Given that 3 painters can complete 3 rooms in 3 hours, 
    prove that 9 painters can complete 27 rooms in 9 hours. -/
theorem painters_rooms_theorem (painters_rate : ℕ → ℕ → ℕ → ℕ) 
  (h : painters_rate 3 3 3 = 3) : painters_rate 9 9 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_painters_rooms_theorem_l2714_271488


namespace NUMINAMATH_CALUDE_not_complete_residue_sum_l2714_271449

theorem not_complete_residue_sum (n : ℕ) (a b : Fin n → ℕ) : 
  Even n →
  (∀ k : Fin n, ∃ i : Fin n, a i ≡ k [ZMOD n]) →
  (∀ k : Fin n, ∃ i : Fin n, b i ≡ k [ZMOD n]) →
  ¬(∀ k : Fin n, ∃ i : Fin n, (a i + b i) ≡ k [ZMOD n]) :=
by sorry

end NUMINAMATH_CALUDE_not_complete_residue_sum_l2714_271449


namespace NUMINAMATH_CALUDE_equal_sets_imply_a_eq_5_intersection_conditions_imply_a_eq_neg_2_l2714_271475

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem 1
theorem equal_sets_imply_a_eq_5 :
  ∀ a : ℝ, A a = B → a = 5 := by sorry

-- Theorem 2
theorem intersection_conditions_imply_a_eq_neg_2 :
  ∀ a : ℝ, (B ∩ A a ≠ ∅) ∧ (C ∩ A a = ∅) → a = -2 := by sorry

end NUMINAMATH_CALUDE_equal_sets_imply_a_eq_5_intersection_conditions_imply_a_eq_neg_2_l2714_271475


namespace NUMINAMATH_CALUDE_cos_36_degrees_l2714_271401

theorem cos_36_degrees : Real.cos (36 * Real.pi / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_degrees_l2714_271401


namespace NUMINAMATH_CALUDE_cube_regions_tetrahedron_regions_l2714_271450

/-- Represents a set of planes in 3D space -/
structure PlaneSet where
  num_planes : ℕ

/-- Calculates the number of regions created by a set of planes -/
def num_regions (planes : PlaneSet) : ℕ := sorry

/-- A cube's faces represented as 6 planes -/
def cube_faces : PlaneSet := { num_planes := 6 }

/-- A tetrahedron's faces represented as 4 planes -/
def tetrahedron_faces : PlaneSet := { num_planes := 4 }

/-- Theorem: The number of regions created by a cube's faces is 27 -/
theorem cube_regions :
  num_regions cube_faces = 27 := by sorry

/-- Theorem: The number of regions created by a tetrahedron's faces is 15 -/
theorem tetrahedron_regions :
  num_regions tetrahedron_faces = 15 := by sorry

end NUMINAMATH_CALUDE_cube_regions_tetrahedron_regions_l2714_271450


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2714_271431

theorem fixed_point_theorem (f : ℝ → ℝ) :
  Continuous f →
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) →
  ∃ x ∈ Set.Icc 0 1, f x = x := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2714_271431


namespace NUMINAMATH_CALUDE_ratio_x_sqrt_w_l2714_271497

theorem ratio_x_sqrt_w (x y z w v : ℝ) 
  (hx : x = 1.20 * y)
  (hy : y = 0.30 * z)
  (hz : z = 1.35 * w)
  (hw : w = v^2)
  (hv : v = 0.50 * x) :
  x / Real.sqrt w = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_sqrt_w_l2714_271497


namespace NUMINAMATH_CALUDE_student_age_problem_l2714_271453

theorem student_age_problem (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ) 
  (total_avg_age : ℕ) (group1_avg_age : ℕ) (group2_avg_age : ℕ) :
  total_students = 20 →
  group1_students = 9 →
  group2_students = 10 →
  total_avg_age = 20 →
  group1_avg_age = 11 →
  group2_avg_age = 24 →
  (total_students * total_avg_age) - (group1_students * group1_avg_age + group2_students * group2_avg_age) = 61 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l2714_271453


namespace NUMINAMATH_CALUDE_jenna_blouses_count_l2714_271445

/-- The number of blouses Jenna needs to dye -/
def num_blouses : ℕ := 100

/-- The number of dots per blouse -/
def dots_per_blouse : ℕ := 20

/-- The amount of dye (in ml) needed per dot -/
def dye_per_dot : ℕ := 10

/-- The number of bottles of dye Jenna needs to buy -/
def num_bottles : ℕ := 50

/-- The volume (in ml) of each bottle of dye -/
def bottle_volume : ℕ := 400

/-- Theorem stating that the number of blouses Jenna needs to dye is correct -/
theorem jenna_blouses_count : 
  num_blouses * (dots_per_blouse * dye_per_dot) = num_bottles * bottle_volume :=
sorry

end NUMINAMATH_CALUDE_jenna_blouses_count_l2714_271445


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l2714_271421

/-- The number of distinguishable arrangements of coins -/
def coin_arrangements (gold : Nat) (silver : Nat) : Nat :=
  Nat.choose (gold + silver) gold * (gold + silver + 1)

/-- Theorem stating the number of distinguishable arrangements for the given problem -/
theorem coin_stack_arrangements :
  coin_arrangements 5 3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l2714_271421


namespace NUMINAMATH_CALUDE_first_house_receives_90_bottles_l2714_271493

/-- Calculates the number of bottles the first house receives given the total number of bottles and the number of bottles containing only cider and only beer. -/
def bottlesForFirstHouse (total : ℕ) (ciderOnly : ℕ) (beerOnly : ℕ) : ℕ :=
  let mixedBottles := total - (ciderOnly + beerOnly)
  (ciderOnly / 2) + (beerOnly / 2) + (mixedBottles / 2)

/-- Theorem stating that given 180 total bottles, with 40 cider-only and 80 beer-only,
    the first house receives 90 bottles when given half of each type. -/
theorem first_house_receives_90_bottles :
  bottlesForFirstHouse 180 40 80 = 90 := by
  sorry

end NUMINAMATH_CALUDE_first_house_receives_90_bottles_l2714_271493


namespace NUMINAMATH_CALUDE_triangle_properties_l2714_271468

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B ∧
  t.b = 3 ∧
  Real.sin t.C = 2 * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2714_271468


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2714_271467

/-- Represents the speed of a train in various conditions -/
structure TrainSpeed where
  /-- Speed of the train including stoppages (in kmph) -/
  average_speed : ℝ
  /-- Time the train stops per hour (in minutes) -/
  stop_time : ℝ
  /-- Speed of the train when not stopping (in kmph) -/
  actual_speed : ℝ

/-- Theorem stating the relationship between average speed, stop time, and actual speed -/
theorem train_speed_calculation (t : TrainSpeed) (h1 : t.average_speed = 36) 
    (h2 : t.stop_time = 24) : t.actual_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2714_271467


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2714_271463

theorem largest_n_satisfying_inequality :
  ∃ (n : ℕ), n^300 < 3^500 ∧ ∀ (m : ℕ), m^300 < 3^500 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2714_271463


namespace NUMINAMATH_CALUDE_camila_garden_walkway_area_camila_garden_walkway_area_proof_l2714_271495

/-- The total area of walkways in Camila's garden -/
theorem camila_garden_walkway_area : ℕ :=
  let num_rows : ℕ := 4
  let num_cols : ℕ := 3
  let bed_width : ℕ := 8
  let bed_height : ℕ := 3
  let walkway_width : ℕ := 2
  let total_width : ℕ := num_cols * bed_width + (num_cols + 1) * walkway_width
  let total_height : ℕ := num_rows * bed_height + (num_rows + 1) * walkway_width
  let total_area : ℕ := total_width * total_height
  let total_bed_area : ℕ := num_rows * num_cols * bed_width * bed_height
  let walkway_area : ℕ := total_area - total_bed_area
  416

theorem camila_garden_walkway_area_proof : camila_garden_walkway_area = 416 := by
  sorry

end NUMINAMATH_CALUDE_camila_garden_walkway_area_camila_garden_walkway_area_proof_l2714_271495


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2714_271458

/-- The line passing through points (-1, 0) and (0, 1) is represented by the equation x - y + 1 = 0 -/
theorem line_equation_through_points : 
  ∀ (x y : ℝ), (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) → x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2714_271458


namespace NUMINAMATH_CALUDE_ant_final_position_l2714_271405

/-- Represents the position of the ant on a 2D plane -/
structure Position where
  x : Int
  y : Int

/-- Represents the direction the ant is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of the ant at any given moment -/
structure AntState where
  pos : Position
  dir : Direction
  moveCount : Nat

/-- The movement function for the ant -/
def move (state : AntState) : AntState :=
  sorry

/-- The main theorem stating the final position of the ant -/
theorem ant_final_position :
  let initial_state : AntState :=
    { pos := { x := -25, y := 25 }
    , dir := Direction.North
    , moveCount := 0
    }
  let final_state := (move^[1010]) initial_state
  final_state.pos = { x := 1491, y := -481 } :=
sorry

end NUMINAMATH_CALUDE_ant_final_position_l2714_271405
