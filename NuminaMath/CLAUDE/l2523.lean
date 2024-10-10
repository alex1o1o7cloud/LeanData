import Mathlib

namespace permutations_count_l2523_252312

/-- The total number of permutations of the string "HMMTHMMT" -/
def total_permutations : ℕ := 420

/-- The number of permutations containing the substring "HMMT" -/
def permutations_with_substring : ℕ := 60

/-- The number of cases over-counted -/
def over_counted_cases : ℕ := 1

/-- The number of permutations without the consecutive substring "HMMT" -/
def permutations_without_substring : ℕ := total_permutations - permutations_with_substring + over_counted_cases

theorem permutations_count : permutations_without_substring = 361 := by
  sorry

end permutations_count_l2523_252312


namespace real_condition_pure_imaginary_condition_fourth_quadrant_condition_l2523_252391

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a - 3) (a^2 + a - 12)

-- (I) z is a real number iff a = -4 or a = 3
theorem real_condition (a : ℝ) : z a = Complex.mk (z a).re 0 ↔ a = -4 ∨ a = 3 := by sorry

-- (II) z is a pure imaginary number iff a = -1
theorem pure_imaginary_condition (a : ℝ) : z a = Complex.mk 0 (z a).im ∧ (z a).im ≠ 0 ↔ a = -1 := by sorry

-- (III) z is in the fourth quadrant iff -4 < a < -1
theorem fourth_quadrant_condition (a : ℝ) : (z a).re > 0 ∧ (z a).im < 0 ↔ -4 < a ∧ a < -1 := by sorry

end real_condition_pure_imaginary_condition_fourth_quadrant_condition_l2523_252391


namespace seventh_grade_class_size_l2523_252363

theorem seventh_grade_class_size (girls boys : ℕ) : 
  girls * 3 + boys = 24 → 
  boys / 3 = 6 → 
  girls + boys = 24 :=
by
  sorry

end seventh_grade_class_size_l2523_252363


namespace unique_solution_absolute_value_equation_l2523_252311

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 1| + |x - 4| :=
by
  -- The unique solution is x = 4
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove that any solution must equal 4
    sorry

#check unique_solution_absolute_value_equation

end unique_solution_absolute_value_equation_l2523_252311


namespace c_rent_share_l2523_252338

/-- Represents the rent share calculation for a pasture -/
def RentShare (total_rent : ℕ) (a_oxen b_oxen c_oxen : ℕ) (a_months b_months c_months : ℕ) : ℕ :=
  let total_ox_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months
  let c_ox_months := c_oxen * c_months
  (total_rent * c_ox_months) / total_ox_months

/-- Theorem stating that c's share of the rent is 45 Rs -/
theorem c_rent_share :
  RentShare 175 10 12 15 7 5 3 = 45 := by
  sorry

end c_rent_share_l2523_252338


namespace no_solution_for_equation_expression_simplifies_to_half_l2523_252337

-- Define the domain for x
def X := {x : ℤ | -3 < x ∧ x ≤ 0}

-- Problem 1
theorem no_solution_for_equation :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → (2 / (x - 2) - 4 / (x^2 - 4) ≠ 1 / (x + 2)) :=
sorry

-- Problem 2
theorem expression_simplifies_to_half :
  ∀ x ∈ X, x = 0 →
  (x^2 / (x + 1) - x + 1) / ((x + 2) / (x^2 + 2*x + 1)) = 1/2 :=
sorry

end no_solution_for_equation_expression_simplifies_to_half_l2523_252337


namespace circle_tangent_properties_l2523_252381

-- Define the circle C
def circle_C (a r : ℝ) := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - a)^2 = r^2}

-- Define the tangent line
def tangent_line := {(x, y) : ℝ × ℝ | x + 2*y - 7 = 0}

-- Define the condition that the line is tangent to the circle at (3, 2)
def is_tangent (a r : ℝ) : Prop :=
  (3, 2) ∈ circle_C a r ∧ (3, 2) ∈ tangent_line ∧
  ∀ (x y : ℝ), (x, y) ∈ circle_C a r ∩ tangent_line → (x, y) = (3, 2)

-- Theorem statement
theorem circle_tangent_properties (a r : ℝ) (h : is_tangent a r) :
  a = 0 ∧ (-1, -1) ∉ circle_C a r :=
sorry

end circle_tangent_properties_l2523_252381


namespace complex_simplification_l2523_252372

theorem complex_simplification :
  (-5 + 3*I : ℂ) - (2 - 7*I) + (1 + 2*I) * (4 - 3*I) = 3 + 15*I :=
by sorry

end complex_simplification_l2523_252372


namespace rectangular_field_area_l2523_252353

theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 90 →
  area = width * length →
  area = 379.6875 := by
  sorry

end rectangular_field_area_l2523_252353


namespace recipe_scaling_l2523_252396

def original_flour : ℚ := 20/3

theorem recipe_scaling :
  let scaled_flour : ℚ := (1/3) * original_flour
  let scaled_sugar : ℚ := (1/2) * scaled_flour
  scaled_flour = 20/9 ∧ scaled_sugar = 10/9 := by sorry

end recipe_scaling_l2523_252396


namespace quadratic_inequality_solution_set_l2523_252357

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} := by sorry

end quadratic_inequality_solution_set_l2523_252357


namespace total_hired_is_35_l2523_252379

/-- Represents the daily pay for heavy equipment operators -/
def heavy_equipment_pay : ℕ := 140

/-- Represents the daily pay for general laborers -/
def general_laborer_pay : ℕ := 90

/-- Represents the total payroll -/
def total_payroll : ℕ := 3950

/-- Represents the number of general laborers employed -/
def num_laborers : ℕ := 19

/-- Calculates the total number of people hired given the conditions -/
def total_hired : ℕ := 
  let num_operators := (total_payroll - general_laborer_pay * num_laborers) / heavy_equipment_pay
  num_operators + num_laborers

/-- Proves that the total number of people hired is 35 -/
theorem total_hired_is_35 : total_hired = 35 := by
  sorry

end total_hired_is_35_l2523_252379


namespace inverse_proportion_ordering_l2523_252360

theorem inverse_proportion_ordering (y₁ y₂ y₃ : ℝ) :
  y₁ = 7 / (-3) ∧ y₂ = 7 / (-1) ∧ y₃ = 7 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end inverse_proportion_ordering_l2523_252360


namespace squirrel_mushroom_theorem_l2523_252395

theorem squirrel_mushroom_theorem (N : ℝ) (h : N > 0) :
  let initial_porcini := 0.85 * N
  let initial_saffron := 0.15 * N
  let eaten (x : ℝ) := x
  let remaining_porcini (x : ℝ) := initial_porcini - eaten x
  let remaining_total (x : ℝ) := N - eaten x
  let final_saffron_ratio (x : ℝ) := initial_saffron / remaining_total x
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ initial_porcini ∧ final_saffron_ratio x = 0.3 ∧ eaten x / N = 1/2 :=
by
  sorry

end squirrel_mushroom_theorem_l2523_252395


namespace square_perimeter_ratio_l2523_252350

theorem square_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0 ∧ s₂ > 0) :
  s₂ = 2.5 * s₁ * Real.sqrt 2 / Real.sqrt 2 →
  (4 * s₂) / (4 * s₁) = 5 / 2 := by
  sorry

end square_perimeter_ratio_l2523_252350


namespace line_not_parallel_when_planes_not_perpendicular_l2523_252347

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_not_parallel_when_planes_not_perpendicular
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : ¬ plane_perpendicular α β) :
  ¬ parallel l m :=
sorry

end line_not_parallel_when_planes_not_perpendicular_l2523_252347


namespace water_one_tenth_after_pourings_l2523_252313

/-- The fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  3 / (n + 3)

/-- The number of pourings required to reach one-tenth of the original volume -/
def pouringsToOneTenth : ℕ := 27

theorem water_one_tenth_after_pourings :
  waterRemaining pouringsToOneTenth = 1 / 10 := by
  sorry

#eval waterRemaining pouringsToOneTenth

end water_one_tenth_after_pourings_l2523_252313


namespace determinant_evaluation_l2523_252344

-- Define the matrix
def matrix (x y z : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
λ i j => match i, j with
  | 0, 0 => 1
  | 0, 1 => x
  | 0, 2 => y
  | 0, 3 => z
  | 1, 0 => 1
  | 1, 1 => x + y
  | 1, 2 => y
  | 1, 3 => z
  | 2, 0 => 1
  | 2, 1 => x
  | 2, 2 => x + y
  | 2, 3 => z
  | 3, 0 => 1
  | 3, 1 => x
  | 3, 2 => y
  | 3, 3 => x + y + z

theorem determinant_evaluation (x y z : ℝ) :
  Matrix.det (matrix x y z) = y * x^2 + y^2 * x := by
  sorry

end determinant_evaluation_l2523_252344


namespace james_friends_count_l2523_252321

/-- The number of pages James writes per letter -/
def pages_per_letter : ℕ := 3

/-- The number of times James writes letters per week -/
def times_per_week : ℕ := 2

/-- The total number of pages James writes in a year -/
def total_pages_per_year : ℕ := 624

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

theorem james_friends_count :
  (total_pages_per_year / weeks_per_year / times_per_week) / pages_per_letter = 2 := by
  sorry

end james_friends_count_l2523_252321


namespace cab_driver_income_day2_l2523_252339

def cab_driver_problem (day1 day2 day3 day4 day5 : ℕ) (average : ℚ) : Prop :=
  day1 = 250 ∧
  day3 = 750 ∧
  day4 = 400 ∧
  day5 = 500 ∧
  average = 460 ∧
  (day1 + day2 + day3 + day4 + day5) / 5 = average

theorem cab_driver_income_day2 :
  ∀ (day1 day2 day3 day4 day5 : ℕ) (average : ℚ),
    cab_driver_problem day1 day2 day3 day4 day5 average →
    day2 = 400 := by
  sorry

end cab_driver_income_day2_l2523_252339


namespace cost_of_flour_for_cakes_claire_cake_flour_cost_l2523_252364

/-- The cost of flour for making cakes -/
theorem cost_of_flour_for_cakes (num_cakes : ℕ) (packages_per_cake : ℕ) (cost_per_package : ℕ) : 
  num_cakes * packages_per_cake * cost_per_package = num_cakes * (packages_per_cake * cost_per_package) :=
by sorry

/-- Claire's cake flour cost calculation -/
theorem claire_cake_flour_cost : 2 * (2 * 3) = 12 :=
by sorry

end cost_of_flour_for_cakes_claire_cake_flour_cost_l2523_252364


namespace dividend_calculation_l2523_252307

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 12)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 176 := by
sorry

end dividend_calculation_l2523_252307


namespace ana_win_probability_l2523_252325

/-- Represents a player in the coin flipping game -/
inductive Player
| Juan
| Carlos
| Manu
| Ana

/-- The coin flipping game with four players -/
def CoinFlipGame :=
  {players : List Player // players = [Player.Juan, Player.Carlos, Player.Manu, Player.Ana]}

/-- The probability of flipping heads on a single flip -/
def headsProbability : ℚ := 1/2

/-- The probability of Ana winning the game -/
def anaProbability (game : CoinFlipGame) : ℚ := 1/31

/-- Theorem stating that the probability of Ana winning is 1/31 -/
theorem ana_win_probability (game : CoinFlipGame) :
  anaProbability game = 1/31 := by
  sorry

end ana_win_probability_l2523_252325


namespace triangle_abc_theorem_l2523_252388

noncomputable section

variables {a b c : ℝ} {A B C : ℝ} {O P : ℝ × ℝ}

def triangle_abc (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def angle_condition (a b c A B C : ℝ) : Prop :=
  a * Real.sin A + a * Real.sin C * Real.cos B + b * Real.sin C * Real.cos A = 
  b * Real.sin B + c * Real.sin A

def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)))

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem triangle_abc_theorem (a b c A B C : ℝ) (O P : ℝ × ℝ) :
  triangle_abc a b c →
  angle_condition a b c A B C →
  (a = 2 → acute_triangle A B C → 
    3 + Real.sqrt 3 < a + b + c ∧ a + b + c < 6 + 2 * Real.sqrt 3) →
  (b^2 = a*c → circumradius a b c = 2 → 
    -2 ≤ dot_product (P.1 - O.1, P.2 - O.2) (P.1 - O.1 - a, P.2 - O.2) ∧
    dot_product (P.1 - O.1, P.2 - O.2) (P.1 - O.1 - a, P.2 - O.2) ≤ 6) →
  B = Real.pi / 3 := by sorry

end

end triangle_abc_theorem_l2523_252388


namespace bridge_length_l2523_252301

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 :=
by
  sorry

end bridge_length_l2523_252301


namespace consecutive_product_plus_one_is_square_l2523_252343

theorem consecutive_product_plus_one_is_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end consecutive_product_plus_one_is_square_l2523_252343


namespace floor_length_percentage_l2523_252382

/-- Proves that for a rectangular floor with given length and area, 
    the percentage by which the length is more than the breadth is 200% -/
theorem floor_length_percentage (length : ℝ) (area : ℝ) :
  length = 19.595917942265423 →
  area = 128 →
  let breadth := area / length
  ((length - breadth) / breadth) * 100 = 200 := by sorry

end floor_length_percentage_l2523_252382


namespace edward_spent_five_on_supplies_l2523_252348

/-- Edward's lawn mowing business finances -/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℤ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = 5

/-- Theorem: Edward spent $5 on supplies -/
theorem edward_spent_five_on_supplies :
  lawn_mowing_problem 2 27 24 := by sorry

end edward_spent_five_on_supplies_l2523_252348


namespace special_function_at_one_l2523_252324

/-- A monotonic function on positive real numbers satisfying f(f(x) - ln x) = 1 + e -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, ∀ y > 0, x < y → f x < f y) ∧
  (∀ x > 0, f (f x - Real.log x) = 1 + Real.exp 1)

/-- The value of f(1) for a special function f is e -/
theorem special_function_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = Real.exp 1 := by
  sorry

end special_function_at_one_l2523_252324


namespace circle_properties_l2523_252309

/-- Given a circle with area 81π cm², prove its radius is 9 cm and circumference is 18π cm. -/
theorem circle_properties (A : ℝ) (h : A = 81 * Real.pi) :
  ∃ (r C : ℝ), r = 9 ∧ C = 18 * Real.pi ∧ A = Real.pi * r^2 ∧ C = 2 * Real.pi * r := by
  sorry

end circle_properties_l2523_252309


namespace power_of_power_l2523_252333

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2523_252333


namespace equation_solution_l2523_252362

theorem equation_solution : 
  ∃ x : ℚ, (3 + 2*x) / (1 + 2*x) - (5 + 2*x) / (7 + 2*x) = 1 - (4*x^2 - 2) / (7 + 16*x + 4*x^2) ∧ x = 7/8 := by
  sorry

end equation_solution_l2523_252362


namespace axis_of_symmetry_parabola_l2523_252366

/-- The axis of symmetry for the parabola y² = -8x is the line x = 2 -/
theorem axis_of_symmetry_parabola (x y : ℝ) : 
  y^2 = -8*x → (x = 2 ↔ ∀ y', y'^2 = -8*x → y'^2 = y^2) :=
by sorry

end axis_of_symmetry_parabola_l2523_252366


namespace walking_problem_l2523_252367

theorem walking_problem (distance : ℝ) (initial_meeting_time : ℝ) 
  (speed_ratio : ℝ) (h1 : distance = 100) (h2 : initial_meeting_time = 3) 
  (h3 : speed_ratio = 4) : 
  ∃ (speed_A speed_B : ℝ) (meeting_times : List ℝ),
    speed_A = 80 / 3 ∧ 
    speed_B = 20 / 3 ∧
    speed_A = speed_ratio * speed_B ∧
    initial_meeting_time * (speed_A + speed_B) = distance ∧
    meeting_times = [3, 5, 9, 15] ∧
    (∀ t ∈ meeting_times, 
      (t ≤ distance / speed_B) ∧ 
      (∃ n : ℕ, t * speed_B = 2 * n * distance - t * speed_A ∨ 
               t * speed_B = (2 * n + 1) * distance - (distance - t * speed_A))) :=
by sorry

end walking_problem_l2523_252367


namespace tangent_circles_radii_l2523_252351

/-- Given a sequence of six circles tangent to each other and two parallel lines,
    where the radii form a geometric sequence, prove that if the smallest radius
    is 5 and the largest is 20, then the radius of the third circle is 5 * 2^(2/5). -/
theorem tangent_circles_radii (r : Fin 6 → ℝ) : 
  (∀ i : Fin 5, r i > 0) →  -- All radii are positive
  (∀ i : Fin 5, r i < r i.succ) →  -- Radii are in increasing order
  (∀ i j : Fin 5, i < j → r j / r i = r (j+1) / r (j : Fin 6)) →  -- Geometric sequence
  r 0 = 5 →  -- Smallest radius
  r 5 = 20 →  -- Largest radius
  r 2 = 5 * 2^(2/5) := by
sorry

end tangent_circles_radii_l2523_252351


namespace not_balanced_numbers_l2523_252385

/-- Definition of balanced numbers with respect to l -/
def balanced (a b : ℝ) : Prop := a + b = 2

/-- Given equation -/
axiom equation : ∃ m : ℝ, (Real.sqrt 3 + m) * (Real.sqrt 3 - 1) = 2

/-- Theorem to prove -/
theorem not_balanced_numbers : ¬∃ m : ℝ, 
  (Real.sqrt 3 + m) * (Real.sqrt 3 - 1) = 2 ∧ 
  balanced (m + Real.sqrt 3) (2 - Real.sqrt 3) := by
  sorry

end not_balanced_numbers_l2523_252385


namespace square_sum_geq_linear_l2523_252356

theorem square_sum_geq_linear (a b : ℝ) : a^2 + b^2 ≥ 2*a - 2*b - 2 := by
  sorry

end square_sum_geq_linear_l2523_252356


namespace pamphlets_total_l2523_252384

/-- Calculates the total number of pamphlets printed by Mike and Leo -/
def total_pamphlets (mike_initial_speed : ℕ) (mike_initial_hours : ℕ) (mike_final_hours : ℕ) : ℕ :=
  let mike_initial_pamphlets := mike_initial_speed * mike_initial_hours
  let mike_final_speed := mike_initial_speed / 3
  let mike_final_pamphlets := mike_final_speed * mike_final_hours
  let leo_hours := mike_initial_hours / 3
  let leo_speed := mike_initial_speed * 2
  let leo_pamphlets := leo_speed * leo_hours
  mike_initial_pamphlets + mike_final_pamphlets + leo_pamphlets

/-- Theorem stating that the total number of pamphlets printed is 9400 -/
theorem pamphlets_total : total_pamphlets 600 9 2 = 9400 := by
  sorry

end pamphlets_total_l2523_252384


namespace square_roots_sum_product_l2523_252304

theorem square_roots_sum_product (m n : ℂ) : 
  m ^ 2 = 2023 → n ^ 2 = 2023 → m + 2 * m * n + n = -4046 := by
  sorry

end square_roots_sum_product_l2523_252304


namespace total_cost_usd_l2523_252315

/-- The cost of items in British pounds and US dollars -/
def cost_in_usd (tea_gbp : ℝ) (scone_gbp : ℝ) (exchange_rate : ℝ) : ℝ :=
  (tea_gbp + scone_gbp) * exchange_rate

/-- Theorem: The total cost in USD for a tea and a scone is $10.80 -/
theorem total_cost_usd :
  cost_in_usd 5 3 1.35 = 10.80 := by
  sorry

end total_cost_usd_l2523_252315


namespace ratio_w_to_y_l2523_252316

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 9) :
  w / y = 8 / 1 := by
sorry

end ratio_w_to_y_l2523_252316


namespace sams_sandwich_count_l2523_252378

/-- Represents the number of different types for each sandwich component -/
structure SandwichOptions where
  bread : Nat
  meat : Nat
  cheese : Nat

/-- Calculates the number of sandwiches Sam can order given the options and restrictions -/
def samsSandwichOptions (options : SandwichOptions) : Nat :=
  options.bread * options.meat * options.cheese - 
  options.bread - 
  options.cheese - 
  options.bread

/-- The theorem stating the number of sandwich options for Sam -/
theorem sams_sandwich_count :
  samsSandwichOptions ⟨5, 7, 6⟩ = 194 := by
  sorry

#eval samsSandwichOptions ⟨5, 7, 6⟩

end sams_sandwich_count_l2523_252378


namespace holly_initial_milk_l2523_252345

/-- Represents the amount of chocolate milk Holly has throughout the day -/
structure ChocolateMilk where
  initial : ℕ
  breakfast : ℕ
  lunch_purchased : ℕ
  lunch : ℕ
  dinner : ℕ
  final : ℕ

/-- The conditions of Holly's chocolate milk consumption -/
def holly_milk : ChocolateMilk where
  breakfast := 8
  lunch_purchased := 64
  lunch := 8
  dinner := 8
  final := 56
  initial := 0  -- This will be proven

/-- Theorem stating that Holly's initial amount of chocolate milk was 80 ounces -/
theorem holly_initial_milk :
  holly_milk.initial = 80 :=
by sorry

end holly_initial_milk_l2523_252345


namespace vacant_seats_l2523_252327

/-- Given a hall with 600 seats where 62% are filled, prove that 228 seats are vacant. -/
theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 62/100) : 
  (total_seats : ℚ) * (1 - filled_percentage) = 228 := by
  sorry

end vacant_seats_l2523_252327


namespace fractional_sum_zero_l2523_252383

theorem fractional_sum_zero (a b c k : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : k ≠ 0) 
  (h5 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 := by
  sorry

end fractional_sum_zero_l2523_252383


namespace limit_at_one_l2523_252342

def f (x : ℝ) : ℝ := x^2

theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 2| < ε :=
  sorry

end limit_at_one_l2523_252342


namespace third_concert_highest_attendance_l2523_252389

/-- Represents a concert with its attendance and early departure numbers -/
structure Concert where
  attendance : ℕ
  early_departure : ℕ

/-- Calculates the number of people who remained until the end of the concert -/
def remaining_attendance (c : Concert) : ℕ :=
  c.attendance - c.early_departure

/-- The three concerts attended -/
def concert1 : Concert := { attendance := 65899, early_departure := 375 }
def concert2 : Concert := { attendance := 65899 + 119, early_departure := 498 }
def concert3 : Concert := { attendance := 80453, early_departure := 612 }

theorem third_concert_highest_attendance :
  remaining_attendance concert3 > remaining_attendance concert1 ∧
  remaining_attendance concert3 > remaining_attendance concert2 :=
by sorry

end third_concert_highest_attendance_l2523_252389


namespace seven_non_similar_triangles_l2523_252397

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base : Point) (top : Point)

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop :=
  sorry

/-- Checks if all sides of a triangle are unequal -/
def hasUnequalSides (t : Triangle) : Prop :=
  sorry

/-- Checks if three lines intersect at a single point -/
def intersectAtPoint (a b c : Altitude) (H : Point) : Prop :=
  sorry

/-- Counts the number of non-similar triangle types in the figure -/
def countNonSimilarTriangles (t : Triangle) (AD BE CF : Altitude) (H : Point) : ℕ :=
  sorry

/-- The main theorem -/
theorem seven_non_similar_triangles 
  (ABC : Triangle) 
  (AD BE CF : Altitude) 
  (H : Point) 
  (h1 : isAcuteAngled ABC) 
  (h2 : hasUnequalSides ABC)
  (h3 : intersectAtPoint AD BE CF H) :
  countNonSimilarTriangles ABC AD BE CF H = 7 :=
sorry

end seven_non_similar_triangles_l2523_252397


namespace find_number_l2523_252387

theorem find_number : ∃ x : ℝ, ((x * 0.5 + 26.1) / 0.4) - 35 = 35 := by
  use 3.8
  sorry

end find_number_l2523_252387


namespace gwen_gave_away_seven_games_l2523_252354

/-- The number of games Gwen gave away -/
def games_given_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Gwen gave away 7 games -/
theorem gwen_gave_away_seven_games :
  let initial := 98
  let remaining := 91
  games_given_away initial remaining = 7 := by
  sorry

end gwen_gave_away_seven_games_l2523_252354


namespace polygon_has_12_sides_l2523_252308

/-- A polygon has n sides. -/
structure Polygon where
  n : ℕ

/-- The sum of interior angles of a polygon with n sides. -/
def sumInteriorAngles (p : Polygon) : ℝ :=
  (p.n - 2) * 180

/-- The sum of exterior angles of any polygon. -/
def sumExteriorAngles : ℝ := 360

/-- Theorem: A polygon has 12 sides if the sum of its interior angles
    is equal to five times the sum of its exterior angles. -/
theorem polygon_has_12_sides (p : Polygon) : 
  sumInteriorAngles p = 5 * sumExteriorAngles → p.n = 12 := by
  sorry

end polygon_has_12_sides_l2523_252308


namespace college_students_count_l2523_252369

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 175) :
  boys + girls = 455 := by
  sorry

end college_students_count_l2523_252369


namespace combined_mixture_indeterminate_l2523_252390

structure TrailMix where
  nuts : ℝ
  dried_fruit : ℝ
  chocolate_chips : ℝ
  pretzels : ℝ
  granola : ℝ
  sum_to_one : nuts + dried_fruit + chocolate_chips + pretzels + granola = 1

def sue_mix : TrailMix := {
  nuts := 0.3,
  dried_fruit := 0.7,
  chocolate_chips := 0,
  pretzels := 0,
  granola := 0,
  sum_to_one := by norm_num
}

def jane_mix : TrailMix := {
  nuts := 0.6,
  dried_fruit := 0,
  chocolate_chips := 0.3,
  pretzels := 0.1,
  granola := 0,
  sum_to_one := by norm_num
}

def tom_mix : TrailMix := {
  nuts := 0.4,
  dried_fruit := 0.5,
  chocolate_chips := 0,
  pretzels := 0,
  granola := 0.1,
  sum_to_one := by norm_num
}

theorem combined_mixture_indeterminate 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c = 1) 
  (h_nuts : a * sue_mix.nuts + b * jane_mix.nuts + c * tom_mix.nuts = 0.45) :
  ∃ (x y : ℝ), 
    x ≠ y ∧ 
    (a * sue_mix.dried_fruit + b * jane_mix.dried_fruit + c * tom_mix.dried_fruit = x) ∧
    (a * sue_mix.dried_fruit + b * jane_mix.dried_fruit + c * tom_mix.dried_fruit = y) :=
sorry

end combined_mixture_indeterminate_l2523_252390


namespace five_fridays_in_july_l2523_252346

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- June of year N -/
def june : Month := {
  days := 30,
  firstDay := DayOfWeek.Tuesday  -- Assuming the first Tuesday is on the 2nd
}

/-- July of year N -/
def july : Month := {
  days := 31,
  firstDay := DayOfWeek.Wednesday  -- Based on June's last day being Tuesday
}

/-- Count occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Main theorem -/
theorem five_fridays_in_july (h : countDayOccurrences june DayOfWeek.Tuesday = 5) :
  countDayOccurrences july DayOfWeek.Friday = 5 := by
  sorry

end five_fridays_in_july_l2523_252346


namespace intersection_nonempty_implies_b_range_l2523_252334

def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
def B (a b : ℝ) : Set ℝ := {x | |x - b| < a}

theorem intersection_nonempty_implies_b_range :
  (∀ b : ℝ, (A ∩ B 1 b).Nonempty) →
  ∀ b : ℝ, -2 < b ∧ b < 2 :=
sorry

end intersection_nonempty_implies_b_range_l2523_252334


namespace binary_multiplication_addition_l2523_252365

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits (least significant bit first) -/
def Binary := List Bool

def binary_11011 : Binary := [true, true, false, true, true]
def binary_111 : Binary := [true, true, true]
def binary_1010 : Binary := [false, true, false, true]
def binary_11000111 : Binary := [true, true, true, false, false, false, true, true]

theorem binary_multiplication_addition :
  (binaryToDecimal binary_11011 * binaryToDecimal binary_111 + binaryToDecimal binary_1010) =
  binaryToDecimal binary_11000111 := by
  sorry

end binary_multiplication_addition_l2523_252365


namespace l_shape_tiling_l2523_252300

/-- Number of ways to tile an L-shaped region with dominos -/
def tiling_count (m n : ℕ) : ℕ :=
  sorry

/-- The L-shaped region is formed by attaching two 2 by 5 rectangles to adjacent sides of a 2 by 2 square -/
theorem l_shape_tiling :
  tiling_count 5 5 = 208 :=
sorry

end l_shape_tiling_l2523_252300


namespace largest_tile_size_378_525_l2523_252314

/-- The largest square tile size that can exactly pave a rectangular courtyard -/
def largest_tile_size (length width : ℕ) : ℕ :=
  Nat.gcd length width

/-- Theorem: The largest square tile size for a 378 cm by 525 cm courtyard is 21 cm -/
theorem largest_tile_size_378_525 :
  largest_tile_size 378 525 = 21 := by
  sorry

#eval largest_tile_size 378 525

end largest_tile_size_378_525_l2523_252314


namespace system_solution_l2523_252370

def solution_set : Set (ℝ × ℝ) :=
  {(-1/Real.sqrt 10, 3/Real.sqrt 10), (-1/Real.sqrt 10, -3/Real.sqrt 10),
   (1/Real.sqrt 10, 3/Real.sqrt 10), (1/Real.sqrt 10, -3/Real.sqrt 10)}

def satisfies_system (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 ≤ 1 ∧
  x^4 - 18*x^2*y^2 + 81*y^4 - 20*x^2 - 180*y^2 + 100 = 0

theorem system_solution :
  {p : ℝ × ℝ | satisfies_system p} = solution_set :=
by sorry

end system_solution_l2523_252370


namespace value_of_expression_l2523_252306

theorem value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 := by
  sorry

end value_of_expression_l2523_252306


namespace average_after_addition_l2523_252394

theorem average_after_addition (numbers : List ℝ) (target_avg : ℝ) : 
  numbers = [6, 16, 8, 12, 21] → target_avg = 17 →
  ∃ x : ℝ, (numbers.sum + x) / (numbers.length + 1 : ℝ) = target_avg ∧ x = 39 := by
sorry

end average_after_addition_l2523_252394


namespace range_of_a_l2523_252386

-- Define propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x ≤ a

-- Define the relationship between p and q
def sufficient_not_necessary (p q : Prop) : Prop :=
  (¬p → ¬q) ∧ ¬(q → p)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, sufficient_not_necessary (p x) (q x a)) → a < -3 :=
sorry

end range_of_a_l2523_252386


namespace rectangular_box_dimensions_l2523_252349

theorem rectangular_box_dimensions :
  ∃! (a b c : ℕ),
    2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
    Even a ∧ Even b ∧ Even c ∧
    2 * (a * b + a * c + b * c) = 4 * (a + b + c) ∧
    a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end rectangular_box_dimensions_l2523_252349


namespace students_taking_no_subjects_l2523_252399

/-- Represents the number of students in various subject combinations --/
structure ScienceClub where
  total : ℕ
  math : ℕ
  physics : ℕ
  chemistry : ℕ
  math_physics : ℕ
  physics_chemistry : ℕ
  math_chemistry : ℕ
  all_three : ℕ

/-- Theorem stating the number of students taking no subjects --/
theorem students_taking_no_subjects (club : ScienceClub)
  (h_total : club.total = 150)
  (h_math : club.math = 85)
  (h_physics : club.physics = 63)
  (h_chemistry : club.chemistry = 40)
  (h_math_physics : club.math_physics = 20)
  (h_physics_chemistry : club.physics_chemistry = 15)
  (h_math_chemistry : club.math_chemistry = 10)
  (h_all_three : club.all_three = 5) :
  club.total - (club.math + club.physics + club.chemistry
    - club.math_physics - club.physics_chemistry - club.math_chemistry
    + club.all_three) = 2 := by
  sorry

#check students_taking_no_subjects

end students_taking_no_subjects_l2523_252399


namespace fish_tank_leak_bucket_size_l2523_252341

/-- 
Given a fish tank leaking at a rate of 1.5 ounces per hour and a maximum time away of 12 hours,
prove that a bucket with twice the capacity of the total leakage will hold 36 ounces.
-/
theorem fish_tank_leak_bucket_size 
  (leak_rate : ℝ) 
  (max_time : ℝ) 
  (h1 : leak_rate = 1.5)
  (h2 : max_time = 12) : 
  2 * (leak_rate * max_time) = 36 := by
  sorry

#check fish_tank_leak_bucket_size

end fish_tank_leak_bucket_size_l2523_252341


namespace specific_arrangement_probability_l2523_252329

/-- The number of X tiles -/
def num_x : ℕ := 5

/-- The number of O tiles -/
def num_o : ℕ := 4

/-- The total number of tiles -/
def total_tiles : ℕ := num_x + num_o

/-- The probability of the specific arrangement -/
def prob_specific_arrangement : ℚ := 1 / (total_tiles.choose num_x)

theorem specific_arrangement_probability :
  prob_specific_arrangement = 1 / 126 := by sorry

end specific_arrangement_probability_l2523_252329


namespace ellipse_y_axis_l2523_252310

theorem ellipse_y_axis (k : ℝ) (h : k < -1) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (1 - k) * x^2 + y^2 = k^2 - 1 ↔ (x^2 / b^2) + (y^2 / a^2) = 1 :=
by
  sorry

end ellipse_y_axis_l2523_252310


namespace levi_basketball_score_l2523_252374

theorem levi_basketball_score (levi_initial : ℕ) (brother_initial : ℕ) (brother_additional : ℕ) (goal_difference : ℕ) :
  levi_initial = 8 →
  brother_initial = 12 →
  brother_additional = 3 →
  goal_difference = 5 →
  (brother_initial + brother_additional + goal_difference) - levi_initial = 12 :=
by sorry

end levi_basketball_score_l2523_252374


namespace unique_four_digit_reverse_l2523_252361

/-- Reverses the digits of a four-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

/-- Checks if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_reverse : ∃! n : ℕ, is_four_digit n ∧ 4 * n = reverse_digits n :=
  sorry

end unique_four_digit_reverse_l2523_252361


namespace ellipse_problem_l2523_252303

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The problem statement -/
theorem ellipse_problem (E : Ellipse) 
  (h_major_axis : E.a = 2 * Real.sqrt 2)
  (A B C : PointOnEllipse E)
  (h_A_vertex : A.x = E.a ∧ A.y = 0)
  (h_BC_origin : ∃ t : ℝ, B.x * t = C.x ∧ B.y * t = C.y)
  (h_B_first_quad : B.x > 0 ∧ B.y > 0)
  (h_BC_AB : Real.sqrt ((B.x - C.x)^2 + (B.y - C.y)^2) = 2 * Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2))
  (h_cos_ABC : (A.x - B.x) / Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 1/5) :
  (E.a^2 = 8 ∧ E.b^2 = 4) ∧
  ∃ (lower upper : ℝ), lower = Real.sqrt 14 / 2 ∧ upper = Real.sqrt 6 ∧
    ∀ (M N : PointOnEllipse E) (l : ℝ → ℝ),
      (∀ x y : ℝ, x^2 + y^2 = 1 → (y - l x) * (1 + l x * l x) = 0) →
      M ≠ N →
      (∃ t : ℝ, M.y = l M.x + t ∧ N.y = l N.x + t) →
      lower < (1/2 * Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)) ∧
      (1/2 * Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)) ≤ upper := by
  sorry

end ellipse_problem_l2523_252303


namespace tim_younger_than_jenny_l2523_252375

-- Define the ages
def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2

-- Theorem statement
theorem tim_younger_than_jenny : jenny_age - tim_age = 12 := by
  sorry

end tim_younger_than_jenny_l2523_252375


namespace sphere_surface_area_with_inscribed_cube_l2523_252380

theorem sphere_surface_area_with_inscribed_cube (cube_surface_area : ℝ) 
  (h : cube_surface_area = 54) : 
  ∃ (sphere_surface_area : ℝ), sphere_surface_area = 27 * Real.pi := by
  sorry

end sphere_surface_area_with_inscribed_cube_l2523_252380


namespace equation_solution_l2523_252332

theorem equation_solution : ∃ x : ℝ, 3 * x + 6 = |(-5 * 4 + 2)| ∧ x = 4 := by
  sorry

end equation_solution_l2523_252332


namespace sum_of_variables_l2523_252393

theorem sum_of_variables (a b c : ℚ) 
  (eq1 : b + c = 15 - 4*a)
  (eq2 : a + c = -18 - 4*b)
  (eq3 : a + b = 10 - 4*c) : 
  2*a + 2*b + 2*c = 7/3 := by
  sorry

end sum_of_variables_l2523_252393


namespace arithmetic_calculations_l2523_252376

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) - 15 = 8) ∧ 
  (-1^4 + (-2)^3 * (-1/2) - |(-1-5)| = -3) := by sorry

end arithmetic_calculations_l2523_252376


namespace nina_widget_purchase_l2523_252373

/-- The number of widgets Nina can purchase at the original price -/
def widgets_purchased (total_money : ℕ) (original_price : ℕ) : ℕ :=
  total_money / original_price

/-- The condition that if the price is reduced by 1, Nina can buy exactly 8 widgets -/
def price_reduction_condition (original_price : ℕ) (total_money : ℕ) : Prop :=
  8 * (original_price - 1) = total_money

theorem nina_widget_purchase :
  ∀ (original_price : ℕ),
    original_price > 0 →
    price_reduction_condition original_price 24 →
    widgets_purchased 24 original_price = 6 := by
  sorry

end nina_widget_purchase_l2523_252373


namespace optimal_prevention_plan_l2523_252330

/-- Represents the cost and effectiveness of a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given preventive measures and event parameters -/
def totalCost (measures : List PreventiveMeasure) (eventProbability : ℝ) (eventLoss : ℝ) : ℝ :=
  (measures.map (·.cost)).sum + eventLoss * (1 - (measures.map (·.effectiveness)).prod)

theorem optimal_prevention_plan (eventProbability : ℝ) (eventLoss : ℝ)
  (measureA : PreventiveMeasure) (measureB : PreventiveMeasure) :
  eventProbability = 0.3 →
  eventLoss = 4 →
  measureA.cost = 0.45 →
  measureB.cost = 0.3 →
  measureA.effectiveness = 0.9 →
  measureB.effectiveness = 0.85 →
  totalCost [measureA, measureB] eventProbability eventLoss <
    min (totalCost [] eventProbability eventLoss)
      (min (totalCost [measureA] eventProbability eventLoss)
        (totalCost [measureB] eventProbability eventLoss)) :=
by
  sorry

end optimal_prevention_plan_l2523_252330


namespace problem_statement_l2523_252317

theorem problem_statement (a b : ℝ) : 
  (Real.sqrt (a + 2) + |b - 1| = 0) → ((a + b)^2007 = -1) := by
  sorry

end problem_statement_l2523_252317


namespace student_selection_l2523_252398

theorem student_selection (n : ℕ) (h : n = 30) : 
  (Nat.choose n 2 = 435) ∧ (Nat.choose n 3 = 4060) := by
  sorry

#check student_selection

end student_selection_l2523_252398


namespace odot_problem_l2523_252305

/-- Definition of the ⊙ operation -/
def odot (x y : ℝ) : ℝ := 2 * x + y

/-- Theorem statement -/
theorem odot_problem (a b : ℝ) (h : odot a (-6 * b) = 4) :
  odot (a - 5 * b) (a + b) = 6 := by
  sorry

end odot_problem_l2523_252305


namespace purely_imaginary_complex_number_l2523_252371

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 2*a - 3) (a + 1)).im ≠ 0 ∧ 
  (Complex.mk (a^2 - 2*a - 3) (a + 1)).re = 0 → 
  a = 3 := by
  sorry

end purely_imaginary_complex_number_l2523_252371


namespace alkaline_probability_l2523_252336

/-- Represents the total number of solutions -/
def total_solutions : ℕ := 5

/-- Represents the number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- Represents the probability of selecting an alkaline solution -/
def probability : ℚ := alkaline_solutions / total_solutions

/-- Theorem stating that the probability of selecting an alkaline solution is 2/5 -/
theorem alkaline_probability : probability = 2 / 5 := by sorry

end alkaline_probability_l2523_252336


namespace chef_cooks_25_wings_l2523_252319

/-- The number of additional chicken wings cooked by the chef for a group of friends -/
def additional_chicken_wings (num_friends : ℕ) (pre_cooked : ℕ) (wings_per_person : ℕ) : ℕ :=
  num_friends * wings_per_person - pre_cooked

/-- Theorem stating that for 9 friends, 2 pre-cooked wings, and 3 wings per person, 
    the chef needs to cook 25 additional wings -/
theorem chef_cooks_25_wings : additional_chicken_wings 9 2 3 = 25 := by
  sorry

end chef_cooks_25_wings_l2523_252319


namespace dragon_eye_centering_l2523_252320

-- Define a circle with a figure drawn on it
structure FiguredCircle where
  center : ℝ × ℝ
  radius : ℝ
  figure : Set (ℝ × ℝ)

-- Define a point that represents the dragon's eye
def dragonEye (fc : FiguredCircle) : ℝ × ℝ := 
  sorry

-- State the theorem
theorem dragon_eye_centering 
  (c1 c2 : FiguredCircle) 
  (h_congruent : c1.radius = c2.radius) 
  (h_identical_figures : c1.figure = c2.figure) 
  (h_c1_centered : dragonEye c1 = c1.center) 
  (h_c2_not_centered : dragonEye c2 ≠ c2.center) : 
  ∃ (part1 part2 : Set (ℝ × ℝ)), 
    (∃ (c3 : FiguredCircle), 
      c3.radius = c1.radius ∧ 
      c3.figure = c1.figure ∧ 
      dragonEye c3 = c3.center ∧ 
      c3.figure = part1 ∪ part2 ∧ 
      part1 ∩ part2 = ∅ ∧ 
      part1 ∪ part2 = c2.figure) :=
sorry

end dragon_eye_centering_l2523_252320


namespace trace_bag_weight_proof_l2523_252318

/-- The weight of one of Trace's shopping bags -/
def trace_bag_weight (
  trace_bags : ℕ)
  (gordon_bags : ℕ)
  (gordon_bag1_weight : ℕ)
  (gordon_bag2_weight : ℕ)
  (lola_bags : ℕ)
  (lola_total_weight : ℕ)
  : ℕ :=
2

theorem trace_bag_weight_proof 
  (trace_bags : ℕ)
  (gordon_bags : ℕ)
  (gordon_bag1_weight : ℕ)
  (gordon_bag2_weight : ℕ)
  (lola_bags : ℕ)
  (lola_total_weight : ℕ)
  (h1 : trace_bags = 5)
  (h2 : gordon_bags = 2)
  (h3 : gordon_bag1_weight = 3)
  (h4 : gordon_bag2_weight = 7)
  (h5 : lola_bags = 4)
  (h6 : trace_bags * trace_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags lola_total_weight = gordon_bag1_weight + gordon_bag2_weight)
  (h7 : lola_total_weight = gordon_bag1_weight + gordon_bag2_weight - 2)
  : trace_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags lola_total_weight = 2 := by
  sorry

#check trace_bag_weight_proof

end trace_bag_weight_proof_l2523_252318


namespace car_rental_cost_per_mile_l2523_252326

theorem car_rental_cost_per_mile 
  (rental_cost : ℝ) 
  (gas_price : ℝ) 
  (gas_amount : ℝ) 
  (miles_driven : ℝ) 
  (total_expense : ℝ) 
  (h1 : rental_cost = 150) 
  (h2 : gas_price = 3.5) 
  (h3 : gas_amount = 8) 
  (h4 : miles_driven = 320) 
  (h5 : total_expense = 338) :
  (total_expense - (rental_cost + gas_price * gas_amount)) / miles_driven = 0.5 := by
sorry


end car_rental_cost_per_mile_l2523_252326


namespace handshakes_in_specific_event_l2523_252355

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group1_size : ℕ  -- people who know each other
  group2_size : ℕ  -- people who know no one
  h_total : total_people = group1_size + group2_size

/-- Calculates the number of handshakes in a social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  (event.group2_size * (event.total_people - 1)) / 2

/-- Theorem stating the number of handshakes in the specific social event -/
theorem handshakes_in_specific_event :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group1_size = 25 ∧
    event.group2_size = 15 ∧
    count_handshakes event = 292 := by
  sorry

end handshakes_in_specific_event_l2523_252355


namespace tv_sales_decrease_l2523_252358

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (h_price_increase : ℝ) (h_revenue_increase : ℝ) :
  original_price > 0 →
  original_quantity > 0 →
  h_price_increase = 0.6 →
  h_revenue_increase = 0.28 →
  let new_price := original_price * (1 + h_price_increase)
  let new_revenue := (1 + h_revenue_increase) * (original_price * original_quantity)
  let sales_decrease := 1 - (new_revenue / (new_price * original_quantity))
  sales_decrease = 0.2 := by
sorry

end tv_sales_decrease_l2523_252358


namespace det_of_matrix_l2523_252328

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![8, 5; -2, 3]

theorem det_of_matrix : Matrix.det matrix = 34 := by sorry

end det_of_matrix_l2523_252328


namespace group_message_problem_l2523_252322

theorem group_message_problem (n : ℕ) (k : ℕ) : 
  n > 1 → 
  k > 0 → 
  k * n * (n - 1) = 440 → 
  n = 2 ∨ n = 5 ∨ n = 11 :=
by sorry

end group_message_problem_l2523_252322


namespace exam_comparison_l2523_252323

theorem exam_comparison (total_items : ℕ) (liza_percentage : ℚ) (rose_incorrect : ℕ) : 
  total_items = 60 →
  liza_percentage = 90 / 100 →
  rose_incorrect = 4 →
  (rose_incorrect : ℚ) < total_items →
  ∃ (liza_correct rose_correct : ℕ),
    (liza_correct : ℚ) = liza_percentage * total_items ∧
    rose_correct = total_items - rose_incorrect ∧
    rose_correct - liza_correct = 2 := by
sorry

end exam_comparison_l2523_252323


namespace ratio_equality_l2523_252335

theorem ratio_equality (a b : ℝ) (h : a / b = 4 / 7) : 7 * a = 4 * b := by
  sorry

end ratio_equality_l2523_252335


namespace existence_of_solution_l2523_252340

theorem existence_of_solution (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3 * n^2 + 4 * n) 
  (hb : b ≤ 3 * n^2 + 4 * n) 
  (hc : c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ 
    (abs x ≤ 2 * n) ∧ 
    (abs y ≤ 2 * n) ∧ 
    (abs z ≤ 2 * n) ∧ 
    (a * x + b * y + c * z = 0) :=
sorry

end existence_of_solution_l2523_252340


namespace spaceship_total_distance_l2523_252359

/-- The total distance traveled by a spaceship between Earth and various planets -/
theorem spaceship_total_distance (d_earth_x d_x_y d_y_z d_z_w d_w_earth : ℝ) 
  (h1 : d_earth_x = 3.37)
  (h2 : d_x_y = 1.57)
  (h3 : d_y_z = 2.19)
  (h4 : d_z_w = 4.27)
  (h5 : d_w_earth = 1.89) :
  d_earth_x + d_x_y + d_y_z + d_z_w + d_w_earth = 13.29 := by
  sorry

end spaceship_total_distance_l2523_252359


namespace ages_sum_l2523_252377

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
sorry

end ages_sum_l2523_252377


namespace cos_210_degrees_l2523_252368

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l2523_252368


namespace james_record_beat_l2523_252331

/-- James' football scoring record --/
theorem james_record_beat (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (old_record : ℕ) : 
  touchdowns_per_game = 4 →
  points_per_touchdown = 6 →
  games_in_season = 15 →
  two_point_conversions = 6 →
  old_record = 300 →
  (touchdowns_per_game * points_per_touchdown * games_in_season + 
   two_point_conversions * 2) - old_record = 72 := by
  sorry

#check james_record_beat

end james_record_beat_l2523_252331


namespace hyperbola_asymptotes_l2523_252302

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = (3/4) * x ∨ y = -(3/4) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l2523_252302


namespace people_in_line_l2523_252392

theorem people_in_line (initial_people : ℕ) (additional_people : ℕ) : 
  initial_people = 61 → additional_people = 22 → initial_people + additional_people = 83 := by
  sorry

end people_in_line_l2523_252392


namespace solve_for_y_l2523_252352

theorem solve_for_y (x y : ℚ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1/10 := by
  sorry

end solve_for_y_l2523_252352
