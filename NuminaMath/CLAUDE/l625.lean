import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_symmetric_roots_l625_62576

theorem sum_of_symmetric_roots (f : ℝ → ℝ) 
  (h_sym : ∀ x, f (3 + x) = f (3 - x)) 
  (h_roots : ∃! (roots : Finset ℝ), roots.card = 6 ∧ ∀ x ∈ roots, f x = 0) : 
  ∃ (roots : Finset ℝ), roots.card = 6 ∧ (∀ x ∈ roots, f x = 0) ∧ (roots.sum id = 18) := by
sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_roots_l625_62576


namespace NUMINAMATH_CALUDE_brocard_and_interior_angle_bound_l625_62579

/-- The Brocard angle of a triangle -/
def brocardAngle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def isInsideOrOnBoundary (M A B C : ℝ × ℝ) : Prop := sorry

theorem brocard_and_interior_angle_bound (A B C M : ℝ × ℝ) :
  isInsideOrOnBoundary M A B C →
  min (brocardAngle A B C) (min (angle A B M) (min (angle B C M) (angle C A M))) ≤ Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_brocard_and_interior_angle_bound_l625_62579


namespace NUMINAMATH_CALUDE_john_shopping_cost_l625_62594

/-- The total cost of buying shirts and ties -/
def total_cost (num_shirts : ℕ) (shirt_price : ℚ) (num_ties : ℕ) (tie_price : ℚ) : ℚ :=
  num_shirts * shirt_price + num_ties * tie_price

/-- Theorem: The total cost of 3 shirts at $15.75 each and 2 ties at $9.40 each is $66.05 -/
theorem john_shopping_cost : 
  total_cost 3 (15.75 : ℚ) 2 (9.40 : ℚ) = (66.05 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_john_shopping_cost_l625_62594


namespace NUMINAMATH_CALUDE_solve_equation_l625_62596

theorem solve_equation (y : ℝ) : (45 / 75 = Real.sqrt (3 * y / 75)) → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l625_62596


namespace NUMINAMATH_CALUDE_fraction_value_at_sqrt_two_l625_62565

theorem fraction_value_at_sqrt_two :
  let x := Real.sqrt 2
  (x^2 - 1) / (x^2 - x) - 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_sqrt_two_l625_62565


namespace NUMINAMATH_CALUDE_university_theater_tickets_l625_62511

/-- The total number of tickets sold at University Theater -/
def total_tickets (adult_price senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) : ℕ :=
  senior_tickets + ((total_receipts - senior_price * senior_tickets) / adult_price)

/-- Theorem stating that the total number of tickets sold is 509 -/
theorem university_theater_tickets :
  total_tickets 21 15 8748 327 = 509 := by
  sorry

end NUMINAMATH_CALUDE_university_theater_tickets_l625_62511


namespace NUMINAMATH_CALUDE_co_molecular_weight_l625_62566

-- Define the atomic weights
def atomic_weight_carbon : ℝ := 12.01
def atomic_weight_oxygen : ℝ := 16.00

-- Define the molecular weight calculation function
def molecular_weight (carbon_atoms : ℕ) (oxygen_atoms : ℕ) : ℝ :=
  carbon_atoms * atomic_weight_carbon + oxygen_atoms * atomic_weight_oxygen

-- Theorem statement
theorem co_molecular_weight :
  molecular_weight 1 1 = 28.01 := by sorry

end NUMINAMATH_CALUDE_co_molecular_weight_l625_62566


namespace NUMINAMATH_CALUDE_all_graphs_different_l625_62533

-- Define the equations
def eq1 (x y : ℝ) : Prop := y = 2 * x - 1
def eq2 (x y : ℝ) : Prop := y = (4 * x^2 - 1) / (2 * x + 1)
def eq3 (x y : ℝ) : Prop := (2 * x + 1) * y = 4 * x^2 - 1

-- Define the graph of an equation as the set of points (x, y) that satisfy it
def graph (eq : ℝ → ℝ → Prop) : Set (ℝ × ℝ) := {p : ℝ × ℝ | eq p.1 p.2}

-- Theorem stating that all graphs are different
theorem all_graphs_different :
  graph eq1 ≠ graph eq2 ∧ graph eq1 ≠ graph eq3 ∧ graph eq2 ≠ graph eq3 :=
sorry

end NUMINAMATH_CALUDE_all_graphs_different_l625_62533


namespace NUMINAMATH_CALUDE_both_questions_correct_l625_62541

/-- Represents a class of students and their test results. -/
structure ClassTestResults where
  total_students : ℕ
  correct_q1 : ℕ
  correct_q2 : ℕ
  absent : ℕ

/-- Calculates the number of students who answered both questions correctly. -/
def both_correct (c : ClassTestResults) : ℕ :=
  c.correct_q1 + c.correct_q2 - (c.total_students - c.absent)

/-- Theorem stating that given the specific class conditions, 
    22 students answered both questions correctly. -/
theorem both_questions_correct 
  (c : ClassTestResults) 
  (h1 : c.total_students = 30)
  (h2 : c.correct_q1 = 25)
  (h3 : c.correct_q2 = 22)
  (h4 : c.absent = 5) :
  both_correct c = 22 := by
  sorry

#eval both_correct ⟨30, 25, 22, 5⟩

end NUMINAMATH_CALUDE_both_questions_correct_l625_62541


namespace NUMINAMATH_CALUDE_new_person_weight_l625_62597

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 70 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l625_62597


namespace NUMINAMATH_CALUDE_squares_sum_difference_l625_62585

theorem squares_sum_difference : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 241 := by
  sorry

end NUMINAMATH_CALUDE_squares_sum_difference_l625_62585


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l625_62515

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8)*x + 20 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l625_62515


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l625_62589

/-- A partnership business between A and B -/
structure Partnership where
  /-- A's investment as a multiple of B's investment -/
  a_investment_multiple : ℝ
  /-- B's profit -/
  b_profit : ℝ
  /-- Total profit -/
  total_profit : ℝ

/-- The ratio of A's investment to B's investment in the partnership -/
def investment_ratio (p : Partnership) : ℝ := p.a_investment_multiple

/-- Theorem stating the investment ratio in the given partnership scenario -/
theorem partnership_investment_ratio (p : Partnership) 
  (h1 : p.b_profit = 4000)
  (h2 : p.total_profit = 28000) : 
  investment_ratio p = 3 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l625_62589


namespace NUMINAMATH_CALUDE_count_even_perfect_square_factors_l625_62500

/-- The number of even perfect square factors of 2^6 * 7^3 * 3^4 -/
def evenPerfectSquareFactors : ℕ := 18

/-- The exponent of 2 in the given number -/
def exponent2 : ℕ := 6

/-- The exponent of 7 in the given number -/
def exponent7 : ℕ := 3

/-- The exponent of 3 in the given number -/
def exponent3 : ℕ := 4

theorem count_even_perfect_square_factors :
  evenPerfectSquareFactors = (exponent2 / 2 + 1) * ((exponent7 / 2) + 1) * ((exponent3 / 2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_count_even_perfect_square_factors_l625_62500


namespace NUMINAMATH_CALUDE_otimes_composition_l625_62572

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^3 - y

-- Theorem statement
theorem otimes_composition (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l625_62572


namespace NUMINAMATH_CALUDE_square_area_7m_l625_62518

theorem square_area_7m (side_length : ℝ) (area : ℝ) : 
  side_length = 7 → area = side_length ^ 2 → area = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_7m_l625_62518


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_equation_l625_62514

theorem largest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 5 * (9 * x^2 + 9 * x + 11) - x * (10 * x - 50)
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x
  ↔ 
  ∃ x : ℝ, x = (-19 + Real.sqrt 53) / 14 ∧ f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_equation_l625_62514


namespace NUMINAMATH_CALUDE_salesman_pears_sold_l625_62553

/-- The amount of pears sold by a salesman in a day -/
theorem salesman_pears_sold (morning_sales afternoon_sales : ℕ) 
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : morning_sales = 120)
  (h3 : afternoon_sales = 240) : 
  morning_sales + afternoon_sales = 360 := by
  sorry

end NUMINAMATH_CALUDE_salesman_pears_sold_l625_62553


namespace NUMINAMATH_CALUDE_exists_islands_with_inverse_area_relation_l625_62534

/-- Represents a rectangular island with length and width in kilometers. -/
structure Island where
  length : ℝ
  width : ℝ

/-- Calculates the area of an island in square kilometers. -/
def islandArea (i : Island) : ℝ :=
  i.length * i.width

/-- Calculates the coastal water area of an island in square kilometers. 
    Coastal water is defined as the area within 50 km of the shore. -/
def coastalWaterArea (i : Island) : ℝ :=
  (i.length + 100) * (i.width + 100) - islandArea i

/-- Theorem stating that there exist two islands where the first has smaller area
    but larger coastal water area compared to the second. -/
theorem exists_islands_with_inverse_area_relation : 
  ∃ (i1 i2 : Island), 
    islandArea i1 < islandArea i2 ∧ 
    coastalWaterArea i1 > coastalWaterArea i2 :=
sorry

end NUMINAMATH_CALUDE_exists_islands_with_inverse_area_relation_l625_62534


namespace NUMINAMATH_CALUDE_independence_day_absences_l625_62586

theorem independence_day_absences (total_children : ℕ) 
  (h1 : total_children = 780)
  (present_children : ℕ)
  (absent_children : ℕ)
  (h2 : total_children = present_children + absent_children)
  (bananas_distributed : ℕ)
  (h3 : bananas_distributed = 4 * present_children)
  (h4 : bananas_distributed = 2 * total_children) :
  absent_children = 390 := by
sorry

end NUMINAMATH_CALUDE_independence_day_absences_l625_62586


namespace NUMINAMATH_CALUDE_tom_bus_time_l625_62555

def minutes_in_hour : ℕ := 60

def tom_schedule : Prop :=
  let wake_up_time : ℕ := 6 * 60 + 45
  let leave_for_school_time : ℕ := 7 * 60 + 15
  let class_duration : ℕ := 55
  let num_classes : ℕ := 7
  let lunch_duration : ℕ := 40
  let additional_activities_duration : ℕ := (5 / 2) * minutes_in_hour
  let return_home_time : ℕ := 17 * 60
  let total_time_away : ℕ := return_home_time - leave_for_school_time
  let total_school_activities : ℕ := num_classes * class_duration + lunch_duration + additional_activities_duration
  let bus_time : ℕ := total_time_away - total_school_activities
  bus_time = 10

theorem tom_bus_time : tom_schedule := by
  sorry

end NUMINAMATH_CALUDE_tom_bus_time_l625_62555


namespace NUMINAMATH_CALUDE_non_swimmers_playing_soccer_l625_62529

/-- Represents the percentage of children who play soccer at Lakeview Summer Camp -/
def soccer_players : ℝ := 0.7

/-- Represents the percentage of children who swim at Lakeview Summer Camp -/
def swimmers : ℝ := 0.5

/-- Represents the percentage of soccer players who also swim -/
def soccer_swimmers : ℝ := 0.3

/-- Theorem stating that the percentage of non-swimmers who play soccer is 98% -/
theorem non_swimmers_playing_soccer :
  (soccer_players - soccer_players * soccer_swimmers) / (1 - swimmers) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_non_swimmers_playing_soccer_l625_62529


namespace NUMINAMATH_CALUDE_percent_asian_in_west_1990_l625_62571

/-- Represents the population of Asians in millions for each region in the U.S. in 1990 -/
structure AsianPopulation where
  northeast : ℕ
  midwest : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the percentage of Asians living in the West to the nearest percent -/
def percentInWest (pop : AsianPopulation) : ℕ :=
  let total := pop.northeast + pop.midwest + pop.south + pop.west
  let westPercentage := (pop.west * 100) / total
  -- Round to nearest percent
  if westPercentage % 10 ≥ 5 then
    (westPercentage / 10 + 1) * 10
  else
    (westPercentage / 10) * 10

/-- The given population data for Asians in 1990 -/
def population1990 : AsianPopulation :=
  { northeast := 2
  , midwest := 2
  , south := 2
  , west := 6 }

/-- Theorem stating that the percentage of Asians living in the West in 1990 is 50% -/
theorem percent_asian_in_west_1990 : percentInWest population1990 = 50 := by
  sorry


end NUMINAMATH_CALUDE_percent_asian_in_west_1990_l625_62571


namespace NUMINAMATH_CALUDE_parallel_vectors_l625_62564

theorem parallel_vectors (a b : ℝ × ℝ) :
  a = (-1, 3) →
  b.1 = 2 →
  (a.1 * b.2 = a.2 * b.1) →
  b.2 = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l625_62564


namespace NUMINAMATH_CALUDE_maxim_birth_probability_l625_62598

/-- The year Maxim starts school -/
def school_start_year : ℕ := 2014

/-- The month Maxim starts school (1-based) -/
def school_start_month : ℕ := 9

/-- The day Maxim starts school -/
def school_start_day : ℕ := 1

/-- Maxim's age when starting school -/
def school_start_age : ℕ := 6

/-- Whether the school start date is Maxim's birthday -/
def is_birthday : Prop := False

/-- The number of days from Jan 1, 2008 to Aug 31, 2008 inclusive -/
def days_in_2008 : ℕ := 244

/-- The total number of possible birth dates -/
def total_possible_days : ℕ := 365

/-- The probability that Maxim was born in 2008 -/
def prob_born_2008 : ℚ := days_in_2008 / total_possible_days

theorem maxim_birth_probability : 
  prob_born_2008 = 244 / 365 := by sorry

end NUMINAMATH_CALUDE_maxim_birth_probability_l625_62598


namespace NUMINAMATH_CALUDE_rectangle_area_l625_62581

/-- Theorem: Area of a rectangle with length 15 cm and width 0.9 times its length -/
theorem rectangle_area (length : ℝ) (width : ℝ) : 
  length = 15 →
  width = 0.9 * length →
  length * width = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l625_62581


namespace NUMINAMATH_CALUDE_system_equations_properties_l625_62567

theorem system_equations_properties (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  (b = 7 - 2 * a) ∧ 
  (a = b + 2) ∧ 
  (3 * a = 9) ∧ 
  (3 * b = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_equations_properties_l625_62567


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l625_62523

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the fixed point A
def point_A : ℝ × ℝ := (4, 3)

-- Define a point P outside circle O
def point_P (a b : ℝ) : Prop := a^2 + b^2 > 5

-- Define the tangent line condition
def is_tangent (a b : ℝ) : Prop := ∃ (t : ℝ), circle_O (a + t) (b + t) ∧ ∀ (s : ℝ), s ≠ t → ¬ circle_O (a + s) (b + s)

-- Define the equality of lengths PQ and PA
def length_equality (a b : ℝ) : Prop := (a - 4)^2 + (b - 3)^2 = a^2 + b^2 - 5

theorem circle_tangent_properties (a b : ℝ) 
  (h1 : point_P a b) 
  (h2 : is_tangent a b) 
  (h3 : length_equality a b) :
  -- 1. Relationship between a and b
  (4 * a + 3 * b - 15 = 0) ∧
  -- 2. Minimum length of PQ
  (∀ (x y : ℝ), point_P x y → is_tangent x y → length_equality x y → 
    (x - 4)^2 + (y - 3)^2 ≥ 16) ∧
  -- 3. Equation of circle P with minimum radius
  (∃ (r : ℝ), r = 3 - Real.sqrt 5 ∧
    ∀ (x y : ℝ), (x - 12/5)^2 + (y - 9/5)^2 = r^2 →
      ∃ (t : ℝ), circle_O (x + t) (y + t)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l625_62523


namespace NUMINAMATH_CALUDE_complex_multiplication_l625_62524

theorem complex_multiplication (Q E D : ℂ) : 
  Q = 7 + 3*I ∧ E = 2 + I ∧ D = 7 - 3*I → Q * E * D = 116 + 58*I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l625_62524


namespace NUMINAMATH_CALUDE_shirt_price_l625_62546

/-- The cost of a pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of a shirt -/
def shirt_cost : ℝ := sorry

/-- The total cost of 3 pairs of jeans and 2 shirts is $69 -/
axiom first_purchase : 3 * jeans_cost + 2 * shirt_cost = 69

/-- The total cost of 2 pairs of jeans and 3 shirts is $66 -/
axiom second_purchase : 2 * jeans_cost + 3 * shirt_cost = 66

/-- The cost of one shirt is $12 -/
theorem shirt_price : shirt_cost = 12 := by sorry

end NUMINAMATH_CALUDE_shirt_price_l625_62546


namespace NUMINAMATH_CALUDE_quadratic_function_product_sign_l625_62542

theorem quadratic_function_product_sign
  (a b c m n p x₁ x₂ : ℝ)
  (h_a_pos : a > 0)
  (h_roots : x₁ < x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0)
  (h_order : m < x₁ ∧ x₁ < n ∧ n < x₂ ∧ x₂ < p) :
  let f := fun x => a * x^2 + b * x + c
  f m * f n * f p < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_product_sign_l625_62542


namespace NUMINAMATH_CALUDE_susan_took_35_oranges_l625_62599

/-- The number of oranges Susan took from the box -/
def oranges_taken (initial final : ℕ) : ℕ := initial - final

/-- Proof that Susan took 35 oranges from the box -/
theorem susan_took_35_oranges (initial final taken : ℕ) 
  (h_initial : initial = 55)
  (h_final : final = 20)
  (h_taken : taken = oranges_taken initial final) : 
  taken = 35 := by
  sorry

end NUMINAMATH_CALUDE_susan_took_35_oranges_l625_62599


namespace NUMINAMATH_CALUDE_min_value_theorem_l625_62570

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = 2 + 2 * Real.sqrt 2 ∧ ∀ z, z = (2 / x) + (x / y) → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l625_62570


namespace NUMINAMATH_CALUDE_total_amount_is_454_5_l625_62506

/-- Represents the share distribution problem -/
def ShareDistribution (w x y z p : ℚ) : Prop :=
  x = (3/2) * w ∧
  y = (1/3) * w ∧
  z = (3/4) * w ∧
  p = (5/8) * w ∧
  y = 36

/-- Theorem stating that the total amount is 454.5 rupees -/
theorem total_amount_is_454_5 (w x y z p : ℚ) 
  (h : ShareDistribution w x y z p) : 
  w + x + y + z + p = 454.5 := by
  sorry

#eval (454.5 : ℚ)

end NUMINAMATH_CALUDE_total_amount_is_454_5_l625_62506


namespace NUMINAMATH_CALUDE_factory_production_rate_l625_62577

/-- Represents the production setup of a factory --/
structure Factory where
  original_machines : ℕ
  original_hours : ℕ
  new_machine_hours : ℕ
  price_per_kg : ℕ
  daily_earnings : ℕ

/-- Calculates the hourly production rate of a single machine --/
def hourly_production_rate (f : Factory) : ℚ :=
  let total_machine_hours := f.original_machines * f.original_hours + f.new_machine_hours
  let daily_production := f.daily_earnings / f.price_per_kg
  daily_production / total_machine_hours

/-- Theorem stating the hourly production rate of a single machine --/
theorem factory_production_rate (f : Factory) 
  (h1 : f.original_machines = 3)
  (h2 : f.original_hours = 23)
  (h3 : f.new_machine_hours = 12)
  (h4 : f.price_per_kg = 50)
  (h5 : f.daily_earnings = 8100) :
  hourly_production_rate f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_rate_l625_62577


namespace NUMINAMATH_CALUDE_positive_x_squared_1024_l625_62522

theorem positive_x_squared_1024 (x : ℝ) (h1 : x > 0) (h2 : 4 * x^2 = 1024) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_squared_1024_l625_62522


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l625_62592

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

-- State the theorem
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x : ℝ | x < 4 ∨ x ≥ 10} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l625_62592


namespace NUMINAMATH_CALUDE_fraction_equality_l625_62558

theorem fraction_equality (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x - y) / (x + y) = -1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l625_62558


namespace NUMINAMATH_CALUDE_area_inner_octagon_l625_62517

/-- The area of a regular octagon formed by connecting the midpoints of four alternate sides of a regular octagon with side length 12 cm. -/
theorem area_inner_octagon (side_length : ℝ) (h_side : side_length = 12) : 
  ∃ area : ℝ, area = 576 + 288 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_area_inner_octagon_l625_62517


namespace NUMINAMATH_CALUDE_photo_arrangements_l625_62520

/-- The number of ways to arrange 1 teacher and 4 students in a row with the teacher in the middle -/
def arrangements_count : ℕ := 24

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of ways to arrange the students -/
def student_arrangements : ℕ := Nat.factorial num_students

theorem photo_arrangements :
  arrangements_count = student_arrangements := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l625_62520


namespace NUMINAMATH_CALUDE_triangle_perimeter_l625_62554

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  c^2 = a * Real.cos B + b * Real.cos A →
  a = 3 →
  b = 3 →
  a + b + c = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l625_62554


namespace NUMINAMATH_CALUDE_sum_of_fractions_l625_62547

theorem sum_of_fractions : 
  let fractions : List ℚ := [2/8, 4/8, 6/8, 8/8, 10/8, 12/8, 14/8, 16/8, 18/8, 20/8]
  fractions.sum = 13.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l625_62547


namespace NUMINAMATH_CALUDE_quadratic_minimum_l625_62532

/-- Given a quadratic function f(x) = x^2 - 2x + m with a minimum value of 1 
    on the interval [3, +∞), prove that m = -2. -/
theorem quadratic_minimum (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x : ℝ, x ≥ 3 → f x = x^2 - 2*x + m) →
  (∀ x : ℝ, x ≥ 3 → f x ≥ 1) →
  (∃ x : ℝ, x ≥ 3 ∧ f x = 1) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l625_62532


namespace NUMINAMATH_CALUDE_sequence_ratio_l625_62584

-- Define arithmetic sequence
def is_arithmetic_sequence (s : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 3, s (i + 1) - s i = d

-- Define geometric sequence
def is_geometric_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, s (i + 1) / s i = r

theorem sequence_ratio :
  ∀ a₁ a₂ b₁ b₂ b₃ : ℝ,
  let s₁ : Fin 4 → ℝ := ![1, a₁, a₂, 9]
  let s₂ : Fin 5 → ℝ := ![1, b₁, b₂, b₃, 9]
  is_arithmetic_sequence s₁ →
  is_geometric_sequence s₂ →
  b₂ / (a₁ + a₂) = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l625_62584


namespace NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l625_62593

/-- Represents a repeating decimal with a given numerator and denominator -/
def repeating_decimal (n : ℕ) (d : ℕ) : ℚ := n / d

/-- The sum of three specific repeating decimals -/
def decimal_sum : ℚ :=
  repeating_decimal 1 3 + repeating_decimal 2 99 + repeating_decimal 4 9999

theorem decimal_sum_equals_fraction : decimal_sum = 10581 / 29889 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l625_62593


namespace NUMINAMATH_CALUDE_cost_per_box_l625_62539

/-- The cost per box for packaging a fine arts collection -/
theorem cost_per_box (box_volume : ℝ) (total_volume : ℝ) (total_cost : ℝ) : 
  box_volume = 20 * 20 * 15 →
  total_volume = 3060000 →
  total_cost = 663 →
  total_cost / (total_volume / box_volume) = 1.30 := by
  sorry

#eval (663 : ℚ) / ((3060000 : ℚ) / (20 * 20 * 15 : ℚ))

end NUMINAMATH_CALUDE_cost_per_box_l625_62539


namespace NUMINAMATH_CALUDE_seedling_probability_l625_62557

def total_seedlings : ℕ := 14
def selechenskaya_seedlings : ℕ := 6
def vologda_seedlings : ℕ := 8
def selected_seedlings : ℕ := 3

theorem seedling_probability :
  (Nat.choose selechenskaya_seedlings selected_seedlings : ℚ) / 
  (Nat.choose total_seedlings selected_seedlings : ℚ) = 5 / 91 := by
  sorry

end NUMINAMATH_CALUDE_seedling_probability_l625_62557


namespace NUMINAMATH_CALUDE_jiale_pricing_correct_l625_62502

/-- Represents the pricing and discount options for teapots and teacups -/
structure TeaSetPricing where
  teapot_price : ℝ
  teacup_price : ℝ
  option1 : ℝ → ℝ  -- Cost function for Option 1
  option2 : ℝ → ℝ  -- Cost function for Option 2

/-- The specific pricing structure for Jiale Supermarket -/
def jiale_pricing : TeaSetPricing :=
  { teapot_price := 90
    teacup_price := 25
    option1 := λ x => 25 * x + 325
    option2 := λ x => 22.5 * x + 405 }

/-- Theorem stating the correctness of the cost calculations -/
theorem jiale_pricing_correct (x : ℝ) (h : x > 5) :
  let p := jiale_pricing
  p.option1 x = 25 * x + 325 ∧ p.option2 x = 22.5 * x + 405 := by
  sorry

#check jiale_pricing_correct

end NUMINAMATH_CALUDE_jiale_pricing_correct_l625_62502


namespace NUMINAMATH_CALUDE_initial_average_marks_l625_62562

theorem initial_average_marks (n : ℕ) (incorrect_mark correct_mark : ℝ) (correct_average : ℝ) :
  n = 10 ∧ incorrect_mark = 90 ∧ correct_mark = 10 ∧ correct_average = 92 →
  (n * correct_average + (incorrect_mark - correct_mark)) / n = 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_marks_l625_62562


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_variables_l625_62530

theorem arithmetic_geometric_inequality_two_variables 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ 
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_variables_l625_62530


namespace NUMINAMATH_CALUDE_solve_for_y_l625_62550

theorem solve_for_y (x y : ℤ) (h1 : x^2 - 5*x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l625_62550


namespace NUMINAMATH_CALUDE_basketball_team_selection_l625_62560

def total_players : Nat := 18
def quadruplets : Nat := 4
def starters : Nat := 6
def quadruplets_in_lineup : Nat := 2

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 6006 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l625_62560


namespace NUMINAMATH_CALUDE_candy_boxes_minimum_l625_62535

theorem candy_boxes_minimum (x y m : ℕ) : 
  x + y = 176 → 
  m > 1 → 
  x + 16 = m * (y - 16) + 31 → 
  x ≥ 131 :=
by sorry

end NUMINAMATH_CALUDE_candy_boxes_minimum_l625_62535


namespace NUMINAMATH_CALUDE_stop_to_qons_l625_62574

/-- Represents a letter in a 2D coordinate system -/
structure Letter where
  char : Char
  x : ℝ
  y : ℝ

/-- Represents a word as a list of letters -/
def Word := List Letter

/-- Rotates a letter 180° clockwise about the origin -/
def rotate180 (l : Letter) : Letter :=
  { l with x := -l.x, y := -l.y }

/-- Reflects a letter in the x-axis -/
def reflectX (l : Letter) : Letter :=
  { l with y := -l.y }

/-- Applies both transformations to a letter -/
def transform (l : Letter) : Letter :=
  reflectX (rotate180 l)

/-- Applies the transformation to a word -/
def transformWord (w : Word) : Word :=
  w.map transform

/-- The initial word "stop" -/
def initialWord : Word := sorry

/-- The expected final word "qons" -/
def finalWord : Word := sorry

theorem stop_to_qons :
  transformWord initialWord = finalWord := by sorry

end NUMINAMATH_CALUDE_stop_to_qons_l625_62574


namespace NUMINAMATH_CALUDE_quadratic_value_l625_62501

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_value (a b c : ℝ) :
  (∃ (x : ℝ), f a b c x = -6 ∧ ∀ (y : ℝ), f a b c y ≥ -6) ∧  -- Minimum value is -6
  (∀ (x : ℝ), f a b c x ≥ f a b c (-2)) ∧                   -- Minimum occurs at x = -2
  f a b c 0 = 20 →                                          -- Passes through (0, 20)
  f a b c (-3) = 0.5 :=                                     -- Value at x = -3 is 0.5
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l625_62501


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l625_62543

theorem pure_imaginary_condition (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 2*a - 3 : ℝ) + (a + 1 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → 
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l625_62543


namespace NUMINAMATH_CALUDE_addition_proof_l625_62538

theorem addition_proof : 9873 + 3927 = 13800 := by
  sorry

end NUMINAMATH_CALUDE_addition_proof_l625_62538


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l625_62509

-- Part (1)
theorem inequality_one (x : ℝ) :
  (2 < |2*x - 5| ∧ |2*x - 5| ≤ 7) ↔ ((-1 ≤ x ∧ x < 3/2) ∨ (7/2 < x ∧ x ≤ 6)) :=
sorry

-- Part (2)
theorem inequality_two (x : ℝ) :
  (1 / (x - 1) > x + 1) ↔ (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l625_62509


namespace NUMINAMATH_CALUDE_trio_songs_count_l625_62595

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  hanna : Nat
  mary : Nat
  alina : Nat
  tina : Nat

/-- Calculates the total number of songs sung by the trios -/
def totalSongs (counts : SongCounts) : Nat :=
  (counts.hanna + counts.mary + counts.alina + counts.tina) / 3

/-- Theorem stating the conditions and the result to be proved -/
theorem trio_songs_count (counts : SongCounts) 
  (hanna_most : counts.hanna = 7 ∧ counts.hanna > counts.alina ∧ counts.hanna > counts.tina)
  (mary_least : counts.mary = 4 ∧ counts.mary < counts.alina ∧ counts.mary < counts.tina)
  (alina_tina_between : counts.alina > 4 ∧ counts.alina < 7 ∧ counts.tina > 4 ∧ counts.tina < 7)
  : totalSongs counts = 7 := by
  sorry

end NUMINAMATH_CALUDE_trio_songs_count_l625_62595


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l625_62510

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem intersection_nonempty_implies_a_greater_than_neg_one (a : ℝ) :
  (A ∩ B a).Nonempty → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l625_62510


namespace NUMINAMATH_CALUDE_intersection_of_lines_l625_62573

/-- Two lines intersect if and only if their slopes are not equal -/
def lines_intersect (m₁ m₂ : ℝ) : Prop := m₁ ≠ m₂

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line (m : ℝ) : ℝ := m

theorem intersection_of_lines :
  let line1_slope : ℝ := -1  -- slope of x + y - 1 = 0
  let line2_slope : ℝ := 1   -- slope of y = x - 1
  lines_intersect (slope_of_line line1_slope) (slope_of_line line2_slope) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l625_62573


namespace NUMINAMATH_CALUDE_election_ratio_l625_62528

theorem election_ratio (Vx Vy : ℝ) 
  (h1 : 0.64 * (Vx + Vy) = 0.76 * Vx + 0.4000000000000002 * Vy)
  (h2 : Vx > 0)
  (h3 : Vy > 0) :
  Vx / Vy = 2 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l625_62528


namespace NUMINAMATH_CALUDE_nell_cards_given_to_jeff_l625_62519

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff : ℕ := 304 - 276

theorem nell_cards_given_to_jeff :
  cards_given_to_jeff = 28 :=
by sorry

end NUMINAMATH_CALUDE_nell_cards_given_to_jeff_l625_62519


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_planes_line_in_plane_implies_parallel_parallel_lines_planes_implies_equal_angles_l625_62559

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def angle_with_plane (l : Line) (p : Plane) : ℝ := sorry

-- Theorem statements
theorem perpendicular_parallel_implies_perpendicular 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel_lines n α → perpendicular m n := by sorry

theorem parallel_planes_line_in_plane_implies_parallel 
  (m : Line) (α β : Plane) :
  parallel_planes α β → line_in_plane m α → parallel_lines m β := by sorry

theorem parallel_lines_planes_implies_equal_angles 
  (m n : Line) (α β : Plane) :
  parallel_lines m n → parallel_planes α β → 
  angle_with_plane m α = angle_with_plane n β := by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_planes_line_in_plane_implies_parallel_parallel_lines_planes_implies_equal_angles_l625_62559


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l625_62568

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 1) / (x - 3) < 0 ↔ -1 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l625_62568


namespace NUMINAMATH_CALUDE_cyclist_distance_l625_62536

/-- The distance between two points A and B for two cyclists with given conditions -/
theorem cyclist_distance (a k : ℝ) (ha : a > 0) (hk : k > 0) : ∃ (z x y : ℝ),
  z > 0 ∧ x > y ∧ y > 0 ∧
  (z + a) / (z - a) = x / y ∧
  (2 * k + 1) * z / ((2 * k - 1) * z) = x / y ∧
  z = 2 * a * k :=
by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l625_62536


namespace NUMINAMATH_CALUDE_y_share_is_27_l625_62531

/-- Given a sum divided among x, y, and z, where y gets 45 paisa and z gets 50 paisa for each rupee x gets, 
    and the total amount is Rs. 117, prove that y's share is Rs. 27. -/
theorem y_share_is_27 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 117) 
  (h2 : y_share = 0.45 * x_share) 
  (h3 : z_share = 0.50 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  y_share = 27 := by
sorry


end NUMINAMATH_CALUDE_y_share_is_27_l625_62531


namespace NUMINAMATH_CALUDE_two_correct_statements_l625_62526

theorem two_correct_statements :
  let statement1 := ∀ a b : ℝ, (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) → a + b = 0
  let statement2 := ∀ a : ℝ, -a < 0
  let statement3 := ∀ n : ℤ, n ≠ 0
  let statement4 := ∀ a b : ℝ, |a| > |b| → |a| > |b - 0|
  let statement5 := ∀ a : ℝ, a ≠ 0 → |a| > 0
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4 ∧ statement5) :=
by
  sorry

end NUMINAMATH_CALUDE_two_correct_statements_l625_62526


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l625_62513

theorem arithmetic_mean_of_fractions : 
  (3/4 : ℚ) + (5/8 : ℚ) / 2 = 11/16 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l625_62513


namespace NUMINAMATH_CALUDE_difference_of_squares_23_15_l625_62591

theorem difference_of_squares_23_15 : (23 + 15)^2 - (23 - 15)^2 = 304 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_23_15_l625_62591


namespace NUMINAMATH_CALUDE_lara_age_proof_l625_62521

/-- Lara's age 7 years ago -/
def lara_age_7_years_ago : ℕ := 9

/-- Years since Lara was 9 -/
def years_since_9 : ℕ := 7

/-- Years until future age -/
def years_to_future : ℕ := 10

/-- Lara's future age -/
def lara_future_age : ℕ := lara_age_7_years_ago + years_since_9 + years_to_future

theorem lara_age_proof : lara_future_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_lara_age_proof_l625_62521


namespace NUMINAMATH_CALUDE_trigonometric_identity_proof_l625_62507

theorem trigonometric_identity_proof :
  Real.cos (13 * π / 180) * Real.sin (58 * π / 180) - 
  Real.sin (13 * π / 180) * Real.sin (32 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_proof_l625_62507


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l625_62527

/-- Given vectors a and b where a is parallel to b, prove that |3a + 2b| = √5 -/
theorem parallel_vectors_magnitude (y : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![-2, y]
  (∃ (k : ℝ), a = k • b) →
  ‖(3 : ℝ) • a + (2 : ℝ) • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l625_62527


namespace NUMINAMATH_CALUDE_sum_of_seven_smallest_multiples_of_12_l625_62582

theorem sum_of_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (λ n => 12 * (n + 1)) = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_smallest_multiples_of_12_l625_62582


namespace NUMINAMATH_CALUDE_max_value_of_g_l625_62583

-- Define the function g(x)
def g (x : ℝ) : ℝ := 9*x - 2*x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l625_62583


namespace NUMINAMATH_CALUDE_unique_solution_base_6_l625_62504

def base_6_to_decimal (n : ℕ) : ℕ := 
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

def decimal_to_base_6 (n : ℕ) : ℕ := 
  (n / 36) * 100 + ((n / 6) % 6) * 10 + (n % 6)

theorem unique_solution_base_6 :
  ∃! (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A < 6 ∧ B < 6 ∧ C < 6 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    base_6_to_decimal (100 * A + 10 * B + C) + base_6_to_decimal (10 * B + C) = 
      base_6_to_decimal (100 * A + 10 * C + A) ∧
    A = 3 ∧ B = 1 ∧ C = 2 ∧
    decimal_to_base_6 (A + B + C) = 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_base_6_l625_62504


namespace NUMINAMATH_CALUDE_remainder_problem_l625_62575

theorem remainder_problem (x y : ℕ) (hx : x > 0) (hy : y ≥ 0)
  (h1 : ∃ r, x ≡ r [MOD 11] ∧ 0 ≤ r ∧ r < 11)
  (h2 : 2 * x ≡ 1 [MOD 6])
  (h3 : 3 * y = (2 * x) / 6)
  (h4 : 7 * y - x = 3) :
  ∃ r, x ≡ r [MOD 11] ∧ r = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l625_62575


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l625_62549

theorem unique_solution_for_equation (m n p : ℕ+) (h_prime : Nat.Prime p) :
  2^(m : ℕ) * p^2 + 1 = n^5 ↔ m = 1 ∧ n = 3 ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l625_62549


namespace NUMINAMATH_CALUDE_rationalize_denominator_l625_62537

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧
    B = 7 ∧
    C = 9 ∧
    D = 13 ∧
    E = 5 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l625_62537


namespace NUMINAMATH_CALUDE_exists_m_n_satisfying_equation_l625_62580

-- Define the "*" operation
def star_op (a b : ℤ) : ℤ :=
  if a = 0 ∨ b = 0 then
    max (a^2) (b^2)
  else
    (if a * b > 0 then 1 else -1) * (a^2 + b^2)

-- Theorem statement
theorem exists_m_n_satisfying_equation :
  ∃ (m n : ℤ), star_op (m - 1) (n + 2) = -2 :=
sorry

end NUMINAMATH_CALUDE_exists_m_n_satisfying_equation_l625_62580


namespace NUMINAMATH_CALUDE_total_sum_calculation_l625_62505

theorem total_sum_calculation (share_a share_b share_c : ℝ) : 
  3 * share_a = 4 * share_b ∧ 
  3 * share_a = 7 * share_c ∧ 
  share_c = 83.99999999999999 → 
  share_a + share_b + share_c = 426.9999999999999 := by
sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l625_62505


namespace NUMINAMATH_CALUDE_elysses_carrying_capacity_l625_62556

/-- The number of bags Elysse can carry in one trip -/
def elysses_bags : ℕ := 3

/-- The number of trips taken to carry all groceries -/
def total_trips : ℕ := 5

/-- The total number of bags carried -/
def total_bags : ℕ := 30

theorem elysses_carrying_capacity :
  elysses_bags = total_bags / (2 * total_trips) :=
sorry

end NUMINAMATH_CALUDE_elysses_carrying_capacity_l625_62556


namespace NUMINAMATH_CALUDE_exists_abc_for_all_n_l625_62516

def interval (k : ℕ) := Set.Ioo (k^2 : ℝ) (k^2 + k + 3 * Real.sqrt 3)

theorem exists_abc_for_all_n :
  ∀ (n : ℕ), ∃ (a b c : ℝ),
    (∃ (k₁ : ℕ), a ∈ interval k₁) ∧
    (∃ (k₂ : ℕ), b ∈ interval k₂) ∧
    (∃ (k₃ : ℕ), c ∈ interval k₃) ∧
    (n : ℝ) = a * b / c :=
by
  sorry


end NUMINAMATH_CALUDE_exists_abc_for_all_n_l625_62516


namespace NUMINAMATH_CALUDE_simplify_expressions_l625_62512

theorem simplify_expressions (x y : ℝ) :
  (3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4) ∧
  ((2 * x^4 - 5 * x^2 - 4 * x + 3) - (3 * x^3 - 5 * x^2 - 4 * x) = 2 * x^4 - 3 * x^3 + 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l625_62512


namespace NUMINAMATH_CALUDE_units_digit_sum_base7_l625_62540

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a natural number to its representation in base 7 --/
def toBase7 (n : ℕ) : Base7 := sorry

/-- Adds two numbers in base 7 --/
def addBase7 (a b : Base7) : Base7 := sorry

/-- Gets the units digit of a number in base 7 --/
def unitsDigitBase7 (n : Base7) : ℕ := sorry

theorem units_digit_sum_base7 :
  unitsDigitBase7 (addBase7 (toBase7 65) (toBase7 34)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base7_l625_62540


namespace NUMINAMATH_CALUDE_imaginary_unit_cube_l625_62548

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_cube_l625_62548


namespace NUMINAMATH_CALUDE_expression_equals_three_l625_62561

theorem expression_equals_three : 
  3⁻¹ + (Real.sqrt 2 - 1)^0 + 2 * Real.sin (30 * π / 180) - (-2/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l625_62561


namespace NUMINAMATH_CALUDE_m_eq_one_necessary_not_sufficient_l625_62587

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem m_eq_one_necessary_not_sufficient :
  ∃ m : ℝ, isPureImaginary (m * (m - 1) + Complex.I) ∧ m ≠ 1 ∧
  ∀ m : ℝ, m = 1 → isPureImaginary (m * (m - 1) + Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_one_necessary_not_sufficient_l625_62587


namespace NUMINAMATH_CALUDE_rain_probability_l625_62525

-- Define the probability of rain on any given day
def p_rain : ℝ := 0.5

-- Define the number of days
def n : ℕ := 6

-- Define the number of rainy days we're interested in
def k : ℕ := 4

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem rain_probability :
  (binomial_coefficient n k : ℝ) * p_rain ^ k * (1 - p_rain) ^ (n - k) = 0.234375 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l625_62525


namespace NUMINAMATH_CALUDE_product_difference_sum_l625_62503

theorem product_difference_sum : 
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    a * b = 50 ∧ 
    (max a b - min a b) = 5 → 
    a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_sum_l625_62503


namespace NUMINAMATH_CALUDE_equation_solutions_l625_62588

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log (x^2 - 5*x + 10) = 2

-- State the theorem
theorem equation_solutions :
  ∃ (x₁ x₂ : ℝ), equation x₁ ∧ equation x₂ ∧ 
  (abs (x₁ - 4.4) < 0.01) ∧ (abs (x₂ - 0.6) < 0.01) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l625_62588


namespace NUMINAMATH_CALUDE_correct_sum_calculation_l625_62569

theorem correct_sum_calculation (n : ℕ) (h1 : n ≥ 1000 ∧ n < 10000) 
  (h2 : n % 10 = 9) (h3 : (n - 3 + 57) = 1823) : n + 57 = 1826 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_sum_calculation_l625_62569


namespace NUMINAMATH_CALUDE_basketball_probability_l625_62563

-- Define the success rate
def success_rate : ℚ := 1/2

-- Define the total number of shots
def total_shots : ℕ := 10

-- Define the number of successful shots we're interested in
def successful_shots : ℕ := 3

-- Define the probability function
def probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Theorem statement
theorem basketball_probability :
  probability total_shots successful_shots success_rate = 15/128 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l625_62563


namespace NUMINAMATH_CALUDE_common_solution_y_values_l625_62578

theorem common_solution_y_values (x y : ℝ) : 
  (x^2 + y^2 - 3 = 0 ∧ x^2 - 4*y + 6 = 0) →
  (y = -2 + Real.sqrt 13 ∨ y = -2 - Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_values_l625_62578


namespace NUMINAMATH_CALUDE_relative_prime_theorem_l625_62552

theorem relative_prime_theorem (u v w : ℤ) :
  (Nat.gcd u.natAbs v.natAbs = 1 ∧ Nat.gcd v.natAbs w.natAbs = 1 ∧ Nat.gcd u.natAbs w.natAbs = 1) ↔
  Nat.gcd (u * v + v * w + w * u).natAbs (u * v * w).natAbs = 1 := by
  sorry

#check relative_prime_theorem

end NUMINAMATH_CALUDE_relative_prime_theorem_l625_62552


namespace NUMINAMATH_CALUDE_range_of_f_l625_62551

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

-- Theorem statement
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l625_62551


namespace NUMINAMATH_CALUDE_fun_run_no_shows_fun_run_no_shows_solution_l625_62508

/-- Fun Run Attendance Problem -/
theorem fun_run_no_shows (signed_up_last_year : ℕ) (runners_this_year : ℕ) : ℕ :=
  let runners_last_year := runners_this_year / 2
  signed_up_last_year - runners_last_year

/-- The number of people who did not show up to run last year is 40 -/
theorem fun_run_no_shows_solution : fun_run_no_shows 200 320 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fun_run_no_shows_fun_run_no_shows_solution_l625_62508


namespace NUMINAMATH_CALUDE_sphere_volume_equals_area_l625_62590

theorem sphere_volume_equals_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_area_l625_62590


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l625_62545

/-- The number of ways to arrange 7 distinguishable frogs with color restrictions -/
def frog_arrangements : ℕ :=
  let total_frogs : ℕ := 7
  let green_frogs : ℕ := 2
  let red_frogs : ℕ := 3
  let blue_frogs : ℕ := 2
  96

/-- Theorem stating that the number of frog arrangements is 96 -/
theorem frog_arrangement_count :
  frog_arrangements = 96 := by sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l625_62545


namespace NUMINAMATH_CALUDE_min_value_expression_l625_62544

theorem min_value_expression (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 10)
  (hb : 1 ≤ b ∧ b ≤ 10)
  (hc : 1 ≤ c ∧ c ≤ 10)
  (hbc : b < c) :
  4 ≤ 3*a - a*b + a*c ∧ ∃ (a' b' c' : ℕ), 
    1 ≤ a' ∧ a' ≤ 10 ∧
    1 ≤ b' ∧ b' ≤ 10 ∧
    1 ≤ c' ∧ c' ≤ 10 ∧
    b' < c' ∧
    3*a' - a'*b' + a'*c' = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l625_62544
