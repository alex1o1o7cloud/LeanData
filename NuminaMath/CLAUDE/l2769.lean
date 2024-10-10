import Mathlib

namespace murtha_pebble_collection_l2769_276988

/-- The sum of an arithmetic sequence with n terms, first term a, and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebble_collection :
  arithmetic_sum 12 1 1 = 78 := by
  sorry

end murtha_pebble_collection_l2769_276988


namespace emily_subtraction_l2769_276930

theorem emily_subtraction : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end emily_subtraction_l2769_276930


namespace rationalize_denominator_l2769_276980

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end rationalize_denominator_l2769_276980


namespace cost_price_per_metre_l2769_276941

/-- Proves that given a cloth length of 200 metres sold for Rs. 12000 with a loss of Rs. 12 per metre, the cost price for one metre of cloth is Rs. 72. -/
theorem cost_price_per_metre (total_length : ℕ) (selling_price : ℕ) (loss_per_metre : ℕ) :
  total_length = 200 →
  selling_price = 12000 →
  loss_per_metre = 12 →
  (selling_price + total_length * loss_per_metre) / total_length = 72 := by
sorry

end cost_price_per_metre_l2769_276941


namespace parallel_lines_m_value_l2769_276917

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 / b1 = a2 / b2

/-- Line l₁: 2x + my - 7 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  2 * x + m * y - 7 = 0

/-- Line l₂: mx + 8y - 14 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  m * x + 8 * y - 14 = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, (parallel_lines 2 m (-7) m 8 (-14)) → m = -4 :=
by sorry

end parallel_lines_m_value_l2769_276917


namespace expression_simplification_l2769_276968

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) (h4 : x ≠ 5) : 
  ((x^2 - 4*x + 3) / (x^2 - 6*x + 9)) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  (x - 1)*(x - 5) / ((x - 2)*(x - 4)) := by sorry

end expression_simplification_l2769_276968


namespace cyclist_motorcyclist_speed_l2769_276958

theorem cyclist_motorcyclist_speed : ∀ (motorcyclist_speed : ℝ) (cyclist_speed : ℝ),
  motorcyclist_speed > 0 ∧
  cyclist_speed > 0 ∧
  cyclist_speed = motorcyclist_speed - 30 ∧
  120 / motorcyclist_speed + 2 = 120 / cyclist_speed →
  motorcyclist_speed = 60 ∧ cyclist_speed = 30 := by
  sorry

end cyclist_motorcyclist_speed_l2769_276958


namespace complex_fourth_power_l2769_276921

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end complex_fourth_power_l2769_276921


namespace probability_eight_distinct_rolls_l2769_276913

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of rolling eight standard, eight-sided dice and getting eight distinct numbers -/
def probability_distinct_rolls : ℚ :=
  (Nat.factorial num_dice) / (num_sides ^ num_dice)

theorem probability_eight_distinct_rolls :
  probability_distinct_rolls = 5 / 1296 := by
  sorry

end probability_eight_distinct_rolls_l2769_276913


namespace equal_sum_sequence_definition_l2769_276942

def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k

theorem equal_sum_sequence_definition (a : ℕ → ℝ) :
  is_equal_sum_sequence a ↔
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k :=
by sorry

end equal_sum_sequence_definition_l2769_276942


namespace height_difference_l2769_276993

/-- Given the heights of Anne, her sister, and Bella, prove the height difference between Bella and Anne's sister. -/
theorem height_difference (anne_height : ℝ) (sister_ratio : ℝ) (bella_ratio : ℝ)
  (h1 : anne_height = 80)
  (h2 : sister_ratio = 2)
  (h3 : bella_ratio = 3) :
  bella_ratio * anne_height - anne_height / sister_ratio = 200 := by
  sorry

end height_difference_l2769_276993


namespace quadratic_polynomial_sequence_bound_l2769_276944

/-- A real quadratic polynomial with positive leading coefficient and no fixed point -/
structure QuadraticPolynomial where
  f : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
  positive_leading : ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ a > 0
  no_fixed_point : ∀ α : ℝ, f α ≠ α

/-- The theorem statement -/
theorem quadratic_polynomial_sequence_bound (f : QuadraticPolynomial) :
  ∃ n : ℕ+, ∀ (a : ℕ → ℝ),
    (∀ i : ℕ, i ≥ 1 → i ≤ n → a i = f.f (a (i-1))) →
    a n > 2021 := by
  sorry

end quadratic_polynomial_sequence_bound_l2769_276944


namespace washing_machine_loads_l2769_276972

theorem washing_machine_loads (machine_capacity : ℕ) (total_clothes : ℕ) : 
  machine_capacity = 5 → total_clothes = 53 → 
  (total_clothes + machine_capacity - 1) / machine_capacity = 11 := by
sorry

end washing_machine_loads_l2769_276972


namespace division_with_remainder_l2769_276955

theorem division_with_remainder (A : ℕ) (h : 14 = A * 3 + 2) : A = 4 := by
  sorry

end division_with_remainder_l2769_276955


namespace min_perimeter_isosceles_triangles_l2769_276943

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.side : ℝ)^2 - ((t.base : ℝ) / 2)^2) / 2

theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    (t1.base : ℚ) / (t2.base : ℚ) = 5 / 4 ∧
    area t1 = area t2 ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 192 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      (s1.base : ℚ) / (s2.base : ℚ) = 5 / 4 →
      area s1 = area s2 →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 192) :=
by
  sorry

end min_perimeter_isosceles_triangles_l2769_276943


namespace martin_financial_calculation_l2769_276929

theorem martin_financial_calculation (g u q : ℂ) (h1 : g * q - u = 15000) (h2 : g = 10) (h3 : u = 10 + 200 * Complex.I) : q = 1501 + 20 * Complex.I := by
  sorry

end martin_financial_calculation_l2769_276929


namespace circle_coverage_theorem_l2769_276953

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of circles used to cover the main circle -/
structure CoverConfiguration where
  main_circle : Circle
  covering_circles : List Circle

/-- Checks if a point is covered by a circle -/
def is_point_covered (point : ℝ × ℝ) (circle : Circle) : Prop :=
  let (x, y) := point
  let (cx, cy) := circle.center
  (x - cx)^2 + (y - cy)^2 ≤ circle.radius^2

/-- Checks if all points in the main circle are covered by at least one of the covering circles -/
def is_circle_covered (config : CoverConfiguration) : Prop :=
  ∀ point, is_point_covered point config.main_circle →
    ∃ cover_circle ∈ config.covering_circles, is_point_covered point cover_circle

/-- The main theorem stating that a circle with diameter 81.9 can be covered by 5 circles of diameter 50 -/
theorem circle_coverage_theorem :
  ∃ config : CoverConfiguration,
    config.main_circle.radius = 81.9 / 2 ∧
    config.covering_circles.length = 5 ∧
    (∀ circle ∈ config.covering_circles, circle.radius = 50 / 2) ∧
    is_circle_covered config :=
  sorry


end circle_coverage_theorem_l2769_276953


namespace herman_bird_feeding_l2769_276963

/-- The number of days Herman feeds the birds -/
def feeding_days : ℕ := 90

/-- The amount of food Herman gives per feeding in cups -/
def food_per_feeding : ℚ := 1/2

/-- The number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- Calculates the total amount of food needed for the feeding period -/
def total_food_needed (days : ℕ) (food_per_feeding : ℚ) (feedings_per_day : ℕ) : ℚ :=
  (days : ℚ) * food_per_feeding * (feedings_per_day : ℚ)

theorem herman_bird_feeding :
  total_food_needed feeding_days food_per_feeding feedings_per_day = 90 := by
  sorry

end herman_bird_feeding_l2769_276963


namespace favorite_numbers_product_l2769_276940

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- Definition of a favorite number -/
def is_favorite (n : ℕ+) : Prop :=
  n * sum_of_digits n = 10 * n

/-- Theorem statement -/
theorem favorite_numbers_product :
  ∃ (a b c : ℕ+),
    a * b * c = 71668 ∧
    is_favorite a ∧
    is_favorite b ∧
    is_favorite c := by sorry

end favorite_numbers_product_l2769_276940


namespace gcd_lcm_sum_l2769_276908

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 15 45 = 59 := by
  sorry

end gcd_lcm_sum_l2769_276908


namespace m_geq_n_l2769_276900

theorem m_geq_n (a b : ℝ) : 
  let M := a^2 + 12*a - 4*b
  let N := 4*a - 20 - b^2
  M ≥ N := by
sorry

end m_geq_n_l2769_276900


namespace parallel_line_through_point_l2769_276985

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The given line 2x - 3y + 4 = 0 -/
def givenLine : Line := { a := 2, b := -3, c := 4 }

/-- The point (-1, 2) -/
def givenPoint : Point := { x := -1, y := 2 }

/-- The line we want to prove -/
def targetLine : Line := { a := 2, b := -3, c := 8 }

theorem parallel_line_through_point :
  (targetLine.isParallelTo givenLine) ∧
  (givenPoint.liesOn targetLine) := by
  sorry

#check parallel_line_through_point

end parallel_line_through_point_l2769_276985


namespace greatest_x_value_l2769_276914

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 21000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 21000 :=
by sorry

end greatest_x_value_l2769_276914


namespace cube_volume_from_surface_area_l2769_276974

/-- The volume of a cube with surface area 150 cm² is 125 cm³. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 150 → s^3 = 125 :=
by
  sorry

end cube_volume_from_surface_area_l2769_276974


namespace complement_of_union_equals_set_l2769_276939

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5}

-- Define set A
def A : Finset Nat := {1,2}

-- Define set B
def B : Finset Nat := {2,3}

-- Theorem statement
theorem complement_of_union_equals_set (h : U = {1,2,3,4,5} ∧ A = {1,2} ∧ B = {2,3}) :
  (U \ (A ∪ B)) = {4,5} := by
  sorry

end complement_of_union_equals_set_l2769_276939


namespace blue_sky_project_exhibition_l2769_276916

theorem blue_sky_project_exhibition (n : ℕ) (m : ℕ) :
  n = 6 →
  m = 6 →
  (Nat.choose n 2) * (5^(n - 2)) = (Nat.choose 6 2) * 5^4 :=
by sorry

end blue_sky_project_exhibition_l2769_276916


namespace negation_of_all_squares_positive_l2769_276927

theorem negation_of_all_squares_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, ¬(n^2 > 0) := by sorry

end negation_of_all_squares_positive_l2769_276927


namespace optimal_deposit_rate_l2769_276987

/-- The bank's profit function -/
def profit (k : ℝ) (x : ℝ) : ℝ := 0.048 * k * x^2 - k * x^3

/-- The derivative of the profit function -/
def profit_derivative (k : ℝ) (x : ℝ) : ℝ := 0.096 * k * x - 3 * k * x^2

theorem optimal_deposit_rate (k : ℝ) (h : k > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 0.048 ∧ 
  (∀ (y : ℝ), y ∈ Set.Ioo 0 0.048 → profit k x ≥ profit k y) ∧
  x = 0.032 := by
  sorry

#eval (0.032 : ℝ) * 100  -- Should output 3.2

end optimal_deposit_rate_l2769_276987


namespace sufficient_not_necessary_condition_l2769_276903

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a + a^2 > b + b^2) ∧
  (∃ a b, a + a^2 > b + b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end sufficient_not_necessary_condition_l2769_276903


namespace triangle_area_l2769_276911

/-- The area of a triangle ABC with given side lengths and angle -/
theorem triangle_area (a b c : ℝ) (θ : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : θ = 2 * Real.pi / 3) :
  let area := (1/2) * a * b * Real.sin θ
  area = (3 * Real.sqrt 3) / 14 := by
sorry

end triangle_area_l2769_276911


namespace consecutive_even_numbers_l2769_276997

theorem consecutive_even_numbers (x y z : ℕ) : 
  (∃ n : ℕ, x = 2*n ∧ y = 2*n + 2 ∧ z = 2*n + 4) →  -- x, y, z are consecutive even numbers
  (x + y + z = (x + y + z) / 3 + 44) →               -- sum is 44 more than average
  z = 24                                             -- largest number is 24
:= by sorry

end consecutive_even_numbers_l2769_276997


namespace probability_positive_sum_is_one_third_l2769_276995

/-- The set of card values in the bag -/
def card_values : Finset Int := {-2, -1, 2}

/-- The sample space of all possible outcomes when drawing two cards with replacement -/
def sample_space : Finset (Int × Int) :=
  card_values.product card_values

/-- The set of favorable outcomes (sum of drawn cards is positive) -/
def favorable_outcomes : Finset (Int × Int) :=
  sample_space.filter (fun p => p.1 + p.2 > 0)

/-- The probability of drawing two cards with a positive sum -/
def probability_positive_sum : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem probability_positive_sum_is_one_third :
  probability_positive_sum = 1/3 := by sorry

end probability_positive_sum_is_one_third_l2769_276995


namespace pqr_plus_xyz_eq_zero_l2769_276925

theorem pqr_plus_xyz_eq_zero 
  (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := by sorry

end pqr_plus_xyz_eq_zero_l2769_276925


namespace coeff_x3_in_expansion_l2769_276977

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in (x-4)^5
def coeff_x3 (x : ℝ) : ℝ := binomial 5 2 * (-4)^2

-- Theorem statement
theorem coeff_x3_in_expansion :
  coeff_x3 x = 160 := by sorry

end coeff_x3_in_expansion_l2769_276977


namespace managers_salary_l2769_276932

def employee_count : ℕ := 24
def initial_average_salary : ℕ := 1500
def average_increase : ℕ := 400

theorem managers_salary (total_salary : ℕ) (managers_salary : ℕ) :
  total_salary = employee_count * initial_average_salary ∧
  (total_salary + managers_salary) / (employee_count + 1) = initial_average_salary + average_increase →
  managers_salary = 11500 := by
  sorry

end managers_salary_l2769_276932


namespace statement_truth_condition_l2769_276948

theorem statement_truth_condition (g : ℝ → ℝ) (c d : ℝ) :
  (∀ x, g x = 4 * x + 5) →
  c > 0 →
  d > 0 →
  (∀ x, |x + 3| < d → |g x + 7| < c) ↔
  d ≤ c / 4 :=
by sorry

end statement_truth_condition_l2769_276948


namespace reaction_stoichiometry_l2769_276902

-- Define the chemical species
def CaO : Type := Unit
def H2O : Type := Unit
def Ca_OH_2 : Type := Unit

-- Define the reaction
def reaction (cao : CaO) (h2o : H2O) : Ca_OH_2 := sorry

-- Define the number of moles
def moles : Type → ℝ := sorry

-- Theorem statement
theorem reaction_stoichiometry :
  ∀ (cao : CaO) (h2o : H2O),
    moles CaO = 1 →
    moles Ca_OH_2 = 1 →
    moles H2O = 1 :=
by sorry

end reaction_stoichiometry_l2769_276902


namespace video_recorder_wholesale_cost_l2769_276976

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost : ℝ),
    let retail_price := wholesale_cost * 1.2
    let employee_price := retail_price * 0.8
    employee_price = 192 →
    wholesale_cost = 200 :=
by
  sorry

end video_recorder_wholesale_cost_l2769_276976


namespace average_weight_decrease_l2769_276904

theorem average_weight_decrease (initial_average : ℝ) : 
  let initial_total : ℝ := 8 * initial_average
  let new_total : ℝ := initial_total - 86 + 46
  let new_average : ℝ := new_total / 8
  initial_average - new_average = 5 := by sorry

end average_weight_decrease_l2769_276904


namespace salad_dressing_weight_is_700_l2769_276979

/-- Calculates the weight of salad dressing given bowl capacity, oil and vinegar proportions, and their densities. -/
def salad_dressing_weight (bowl_capacity : ℝ) (oil_proportion : ℝ) (vinegar_proportion : ℝ) (oil_density : ℝ) (vinegar_density : ℝ) : ℝ :=
  (bowl_capacity * oil_proportion * oil_density) + (bowl_capacity * vinegar_proportion * vinegar_density)

/-- Theorem stating that the weight of the salad dressing is 700 grams given the specified conditions. -/
theorem salad_dressing_weight_is_700 :
  salad_dressing_weight 150 (2/3) (1/3) 5 4 = 700 := by
  sorry

end salad_dressing_weight_is_700_l2769_276979


namespace binomial_expansion_equality_l2769_276935

theorem binomial_expansion_equality (x : ℝ) : 
  (x - 1)^4 - 4*x*(x - 1)^3 + 6*x^2*(x - 1)^2 - 4*x^3*(x - 1) * x^4 = 1 := by
  sorry

end binomial_expansion_equality_l2769_276935


namespace set_swept_equals_parabola_l2769_276989

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line A_t B_t for a given t -/
def line_A_t_B_t (t : ℝ) (p : Point) : Prop :=
  p.y = t * p.x - t^2 + 1

/-- The set of all points on or below any line A_t B_t -/
def set_swept_by_lines (p : Point) : Prop :=
  ∃ t : ℝ, line_A_t_B_t t p

/-- The parabola y = x^2/4 + 1 -/
def parabola (p : Point) : Prop :=
  p.y ≤ p.x^2 / 4 + 1

theorem set_swept_equals_parabola :
  ∀ p : Point, set_swept_by_lines p ↔ parabola p := by
  sorry


end set_swept_equals_parabola_l2769_276989


namespace inequality_addition_l2769_276936

theorem inequality_addition (m n c : ℝ) : m > n → m + c > n + c := by
  sorry

end inequality_addition_l2769_276936


namespace no_solution_implies_a_equals_one_l2769_276905

theorem no_solution_implies_a_equals_one (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x) / (x - 2) ≠ 4 / (x - 2) + 1) → a = 1 := by
  sorry

end no_solution_implies_a_equals_one_l2769_276905


namespace lacy_correct_percentage_l2769_276991

theorem lacy_correct_percentage (x : ℕ) (x_pos : x > 0) :
  let total_problems := 6 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) = 2 / 3 := by
  sorry

end lacy_correct_percentage_l2769_276991


namespace tangent_line_y_intercept_l2769_276951

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle1 : Circle := ⟨(3, 0), 3⟩
def circle2 : Circle := ⟨(7, 0), 2⟩
def circle3 : Circle := ⟨(11, 0), 1⟩

-- Define the tangent line
structure TangentLine where
  slope : ℝ
  yIntercept : ℝ

-- Function to check if a line is tangent to a circle
def isTangent (l : TangentLine) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  let r := c.radius
  let m := l.slope
  let b := l.yIntercept
  (y₀ - m * x₀ - b)^2 = (m^2 + 1) * r^2

-- Theorem statement
theorem tangent_line_y_intercept :
  ∃ l : TangentLine,
    isTangent l circle1 ∧
    isTangent l circle2 ∧
    isTangent l circle3 ∧
    l.yIntercept = 36 :=
sorry

end tangent_line_y_intercept_l2769_276951


namespace expected_digits_is_correct_l2769_276946

/-- A fair 20-sided die with numbers 1 to 20 -/
def icosahedral_die : Finset ℕ := Finset.range 20

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expected_digits : ℚ :=
  (icosahedral_die.sum (λ i => num_digits (i + 1))) / icosahedral_die.card

/-- Theorem: The expected number of digits is 1.55 -/
theorem expected_digits_is_correct :
  expected_digits = 31 / 20 := by sorry

end expected_digits_is_correct_l2769_276946


namespace value_of_x_l2769_276931

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end value_of_x_l2769_276931


namespace negation_equivalence_l2769_276952

theorem negation_equivalence :
  (¬ ∀ (n : ℕ), ∃ (x : ℝ), n^2 < x) ↔ (∃ (n : ℕ), ∀ (x : ℝ), n^2 ≥ x) := by
  sorry

end negation_equivalence_l2769_276952


namespace square_root_of_two_l2769_276901

theorem square_root_of_two :
  ∀ x : ℝ, x^2 = 2 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by sorry

end square_root_of_two_l2769_276901


namespace equilateral_triangle_figure_divisible_l2769_276906

/-- A figure composed of equilateral triangles -/
structure EquilateralTriangleFigure where
  /-- The set of points in the figure -/
  points : Set ℝ × ℝ
  /-- Predicate asserting that the figure is composed of equal equilateral triangles -/
  is_composed_of_equilateral_triangles : Prop

/-- A straight line in 2D space -/
structure Line where
  /-- Slope of the line -/
  slope : ℝ
  /-- Y-intercept of the line -/
  intercept : ℝ

/-- Predicate asserting that a line divides a figure into two congruent parts -/
def divides_into_congruent_parts (f : EquilateralTriangleFigure) (l : Line) : Prop :=
  sorry

/-- Theorem stating that any figure composed of equal equilateral triangles
    can be divided into two congruent parts by a straight line -/
theorem equilateral_triangle_figure_divisible (f : EquilateralTriangleFigure) :
  ∃ l : Line, divides_into_congruent_parts f l :=
sorry

end equilateral_triangle_figure_divisible_l2769_276906


namespace sum_reciprocals_l2769_276949

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 := by
sorry

end sum_reciprocals_l2769_276949


namespace equation_solutions_l2769_276983

theorem equation_solutions :
  (∃ x : ℚ, 2*x + 1 = -2 - 3*x ∧ x = -3/5) ∧
  (∃ x : ℚ, x + (1-2*x)/3 = 2 - (x+2)/2 ∧ x = 4/5) :=
by sorry

end equation_solutions_l2769_276983


namespace trig_identities_for_point_l2769_276938

/-- Given a point P(1, -3) on the terminal side of angle α, prove trigonometric identities. -/
theorem trig_identities_for_point (α : Real) :
  let P : ℝ × ℝ := (1, -3)
  (P.1 = Real.cos α * Real.sqrt (P.1^2 + P.2^2) ∧ 
   P.2 = Real.sin α * Real.sqrt (P.1^2 + P.2^2)) →
  Real.sin α = -3 * Real.sqrt 10 / 10 ∧ 
  Real.sqrt 10 * Real.cos α + Real.tan α = -2 := by
sorry

end trig_identities_for_point_l2769_276938


namespace parity_condition_l2769_276996

theorem parity_condition (n : ℕ) : n ≥ 2 →
  (∀ i j : ℕ, 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n →
    Even (i + j) ↔ Even (Nat.choose n i + Nat.choose n j)) ↔
  ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by sorry

end parity_condition_l2769_276996


namespace vaccine_cost_reduction_l2769_276971

/-- The cost reduction for vaccine production over one year -/
def costReduction (initialCost : ℝ) (decreaseRate : ℝ) : ℝ :=
  initialCost * decreaseRate - initialCost * decreaseRate^2

/-- Theorem: The cost reduction for producing 1 set of vaccines this year
    compared to last year, given an initial cost of 5000 yuan two years ago
    and an annual average decrease rate of x, is 5000x - 5000x^2 yuan. -/
theorem vaccine_cost_reduction (x : ℝ) :
  costReduction 5000 x = 5000 * x - 5000 * x^2 := by
  sorry

end vaccine_cost_reduction_l2769_276971


namespace factors_of_N_l2769_276945

/-- The number of natural-number factors of N, where N = 2^4 * 3^2 * 5^1 * 7^2 -/
def number_of_factors (N : ℕ) : ℕ :=
  (5 : ℕ) * (3 : ℕ) * (2 : ℕ) * (3 : ℕ)

/-- N is defined as 2^4 * 3^2 * 5^1 * 7^2 -/
def N : ℕ := 2^4 * 3^2 * 5^1 * 7^2

theorem factors_of_N : number_of_factors N = 90 := by
  sorry

end factors_of_N_l2769_276945


namespace smallest_better_discount_l2769_276966

theorem smallest_better_discount (x : ℝ) (h : x > 0) : ∃ (n : ℕ), n = 38 ∧ 
  (∀ (m : ℕ), m < n → 
    ((1 - m / 100) * x < (1 - 0.2) * (1 - 0.2) * x ∨
     (1 - m / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x ∨
     (1 - m / 100) * x < (1 - 0.3) * (1 - 0.1) * x)) ∧
  (1 - n / 100) * x > (1 - 0.2) * (1 - 0.2) * x ∧
  (1 - n / 100) * x > (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x ∧
  (1 - n / 100) * x > (1 - 0.3) * (1 - 0.1) * x :=
sorry

end smallest_better_discount_l2769_276966


namespace largest_divisor_of_difference_of_squares_l2769_276923

theorem largest_divisor_of_difference_of_squares (a b : ℤ) :
  let m : ℤ := 2*a + 3
  let n : ℤ := 2*b + 1
  (n < m) →
  (∃ k : ℤ, m^2 - n^2 = 4*k) ∧
  (∀ d : ℤ, d > 4 → ∃ a' b' : ℤ, 
    let m' : ℤ := 2*a' + 3
    let n' : ℤ := 2*b' + 1
    (n' < m') ∧ (m'^2 - n'^2) % d ≠ 0) :=
by sorry

end largest_divisor_of_difference_of_squares_l2769_276923


namespace arithmetic_sequence_fifth_term_l2769_276928

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 7) :
  a 5 = 15 :=
sorry

end arithmetic_sequence_fifth_term_l2769_276928


namespace p_plus_q_value_l2769_276918

theorem p_plus_q_value (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 25*p - 75 = 0) 
  (hq : 10*q^3 - 75*q^2 - 365*q + 3375 = 0) : 
  p + q = 39/4 := by
sorry

end p_plus_q_value_l2769_276918


namespace inequality_solution_l2769_276973

theorem inequality_solution (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by
  sorry

end inequality_solution_l2769_276973


namespace meaningful_sqrt_range_l2769_276961

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
sorry

end meaningful_sqrt_range_l2769_276961


namespace age_proof_l2769_276994

/-- The age of a person whose current age is three times what it was six years ago. -/
def age : ℕ := 9

theorem age_proof : age = 9 := by
  have h : age = 3 * (age - 6) := by sorry
  sorry

end age_proof_l2769_276994


namespace pencil_distribution_l2769_276992

/-- Given 1204 pens and an unknown number of pencils distributed equally among 28 students,
    prove that the total number of pencils must be a multiple of 28. -/
theorem pencil_distribution (total_pencils : ℕ) : 
  (1204 % 28 = 0) → 
  (∃ (pencils_per_student : ℕ), total_pencils = 28 * pencils_per_student) :=
by sorry

end pencil_distribution_l2769_276992


namespace weight_of_raisins_l2769_276975

/-- Given that Kelly bought peanuts and raisins, prove the weight of raisins. -/
theorem weight_of_raisins 
  (total_weight : ℝ) 
  (peanut_weight : ℝ) 
  (h1 : total_weight = 0.5) 
  (h2 : peanut_weight = 0.1) : 
  total_weight - peanut_weight = 0.4 := by
  sorry

end weight_of_raisins_l2769_276975


namespace casey_savings_l2769_276950

/-- Represents the weekly savings when hiring the cheaper employee --/
def weeklySavings (hourlyRate1 hourlyRate2 subsidy hoursPerWeek : ℝ) : ℝ :=
  (hourlyRate1 * hoursPerWeek) - ((hourlyRate2 - subsidy) * hoursPerWeek)

/-- Proves that Casey saves $160 per week by hiring the cheaper employee --/
theorem casey_savings :
  let hourlyRate1 : ℝ := 20
  let hourlyRate2 : ℝ := 22
  let subsidy : ℝ := 6
  let hoursPerWeek : ℝ := 40
  weeklySavings hourlyRate1 hourlyRate2 subsidy hoursPerWeek = 160 := by
  sorry

end casey_savings_l2769_276950


namespace two_solutions_equation_l2769_276907

/-- The value of a 2x2 matrix [[a, c], [d, b]] is defined as ab - cd -/
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

/-- The equation 2x^2 - x = 3 has exactly two real solutions -/
theorem two_solutions_equation :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ matrix_value (2*x) x 1 x = 3 :=
sorry

end two_solutions_equation_l2769_276907


namespace min_value_is_884_l2769_276959

/-- A type representing a permutation of the numbers 1 to 9 -/
def Perm9 := { f : Fin 9 → Fin 9 // Function.Bijective f }

/-- The expression we want to minimize -/
def expr (p : Perm9) : ℕ :=
  let x₁ := (p.val 0).val + 1
  let x₂ := (p.val 1).val + 1
  let x₃ := (p.val 2).val + 1
  let y₁ := (p.val 3).val + 1
  let y₂ := (p.val 4).val + 1
  let y₃ := (p.val 5).val + 1
  let z₁ := (p.val 6).val + 1
  let z₂ := (p.val 7).val + 1
  let z₃ := (p.val 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃ + x₁ * y₁ * z₁

/-- The theorem stating that the minimum value of the expression is 884 -/
theorem min_value_is_884 : ∀ p : Perm9, expr p ≥ 884 := by
  sorry

end min_value_is_884_l2769_276959


namespace work_completion_time_l2769_276926

theorem work_completion_time 
  (a_time b_time c_time : ℝ) 
  (ha : a_time = 8) 
  (hb : b_time = 12) 
  (hc : c_time = 24) : 
  1 / (1 / a_time + 1 / b_time + 1 / c_time) = 4 := by
  sorry

end work_completion_time_l2769_276926


namespace max_revenue_at_50_10_l2769_276922

/-- Represents the parking lot problem -/
structure ParkingLot where
  carSpace : ℝ
  busSpace : ℝ
  carFee : ℝ
  busFee : ℝ
  totalArea : ℝ
  maxVehicles : ℕ

/-- Revenue function for the parking lot -/
def revenue (p : ParkingLot) (x y : ℝ) : ℝ :=
  p.carFee * x + p.busFee * y

/-- Theorem stating that (50, 10) maximizes revenue for the given parking lot problem -/
theorem max_revenue_at_50_10 (p : ParkingLot)
  (h1 : p.carSpace = 6)
  (h2 : p.busSpace = 30)
  (h3 : p.carFee = 2.5)
  (h4 : p.busFee = 7.5)
  (h5 : p.totalArea = 600)
  (h6 : p.maxVehicles = 60) :
  ∀ x y : ℝ,
  x ≥ 0 → y ≥ 0 →
  x + y ≤ p.maxVehicles →
  p.carSpace * x + p.busSpace * y ≤ p.totalArea →
  revenue p x y ≤ revenue p 50 10 := by
  sorry


end max_revenue_at_50_10_l2769_276922


namespace triangle_problem_l2769_276999

theorem triangle_problem (A B C : ℝ) (AC BC : ℝ) (h1 : AC = 2) (h2 : BC = 3) (h3 : Real.cos A = -4/5) :
  Real.sin B = 2/5 ∧ Real.sin (2*B + π/6) = (12*Real.sqrt 7 + 17) / 50 := by
  sorry

end triangle_problem_l2769_276999


namespace total_fish_count_l2769_276962

/-- Represents the number of fish in Jonah's aquariums -/
def total_fish (x y : ℕ) : ℤ :=
  let first_aquarium := 14 + 2 - 2 * x + 3
  let second_aquarium := 18 + 4 - 4 * y + 5
  first_aquarium + second_aquarium

/-- The theorem stating the total number of fish in both aquariums -/
theorem total_fish_count (x y : ℕ) : total_fish x y = 46 - 2 * x - 4 * y := by
  sorry

end total_fish_count_l2769_276962


namespace typist_salary_problem_l2769_276984

theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 3135) → original_salary = 3000 := by
  sorry

end typist_salary_problem_l2769_276984


namespace problem_l2769_276957

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem problem (a b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  (∀ n, a n ≠ 0) →
  2 * a 3 - a 1 ^ 2 = 0 →
  a 1 = d →
  b 13 = a 2 →
  b 1 = a 1 →
  b 6 * b 8 = 72 := by
sorry

end problem_l2769_276957


namespace equal_coin_count_l2769_276998

def coin_count (t : ℕ) : ℕ := t / 3

theorem equal_coin_count (t : ℕ) (h : t % 3 = 0) :
  let one_dollar_count := coin_count t
  let two_dollar_count := coin_count t
  one_dollar_count * 1 + two_dollar_count * 2 = t ∧
  one_dollar_count = two_dollar_count :=
by sorry

end equal_coin_count_l2769_276998


namespace max_prime_factors_l2769_276924

theorem max_prime_factors (x y : ℕ+) 
  (h_gcd : (Nat.gcd x y).factors.length = 5)
  (h_lcm : (Nat.lcm x y).factors.length = 20)
  (h_fewer : (x : ℕ).factors.length < (y : ℕ).factors.length) :
  (x : ℕ).factors.length ≤ 12 := by
  sorry

end max_prime_factors_l2769_276924


namespace quadratic_root_ratio_sum_l2769_276937

theorem quadratic_root_ratio_sum (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 1 = 0 → 
  x₂^2 - x₂ - 1 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  (x₂ / x₁) + (x₁ / x₂) = -3 := by
sorry

end quadratic_root_ratio_sum_l2769_276937


namespace joan_payment_l2769_276920

/-- Represents the purchase amounts for Joan, Karl, and Lea --/
structure Purchases where
  joan : ℝ
  karl : ℝ
  lea : ℝ

/-- Defines the conditions of the telescope purchase problem --/
def validPurchases (p : Purchases) : Prop :=
  p.joan + p.karl + p.lea = 600 ∧
  2 * p.joan = p.karl + 74 ∧
  p.lea - p.karl = 52

/-- Theorem stating that if the purchases satisfy the given conditions, 
    then Joan's payment is $139.20 --/
theorem joan_payment (p : Purchases) (h : validPurchases p) : 
  p.joan = 139.20 := by
  sorry

end joan_payment_l2769_276920


namespace square_sum_equals_eighteen_l2769_276970

theorem square_sum_equals_eighteen (a b : ℝ) (h1 : a - b = Real.sqrt 2) (h2 : a * b = 4) :
  (a + b)^2 = 18 := by
  sorry

end square_sum_equals_eighteen_l2769_276970


namespace gcd_221_195_l2769_276909

theorem gcd_221_195 : Nat.gcd 221 195 = 13 := by
  sorry

end gcd_221_195_l2769_276909


namespace probability_of_one_in_20_rows_l2769_276912

/-- The number of rows in Pascal's Triangle we're considering -/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def ones_count (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of choosing a 1 from the first n rows of Pascal's Triangle -/
def probability_of_one (n : ℕ) : ℚ := ones_count n / total_elements n

theorem probability_of_one_in_20_rows : 
  probability_of_one n = 13 / 70 := by sorry

end probability_of_one_in_20_rows_l2769_276912


namespace coefficient_x4_is_10_l2769_276982

/-- The coefficient of x^4 in the expansion of (x^2 + 1/x)^5 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 5 2)

/-- Theorem: The coefficient of x^4 in the expansion of (x^2 + 1/x)^5 is 10 -/
theorem coefficient_x4_is_10 : coefficient_x4 = 10 := by
  sorry

end coefficient_x4_is_10_l2769_276982


namespace open_box_volume_l2769_276915

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume (sheet_length sheet_width cut_size : ℝ) 
  (h1 : sheet_length = 100)
  (h2 : sheet_width = 50)
  (h3 : cut_size = 10) : 
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 24000 := by
  sorry

#check open_box_volume

end open_box_volume_l2769_276915


namespace ellipse_eccentricity_l2769_276990

/-- The eccentricity of an ellipse with specific geometric properties -/
theorem ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  let AF := a - (e * a)
  let AB := Real.sqrt (a^2 + b^2)
  let BF := a
  (∃ (r : ℝ), AF * r = AB ∧ AB * r = 3 * BF) →
  e = (Real.sqrt 5 - 1) / 2 := by
sorry

end ellipse_eccentricity_l2769_276990


namespace dany_farm_bushels_l2769_276947

/-- The number of bushels needed for Dany's farm animals for one day -/
def bushels_needed (cows sheep : ℕ) (cow_sheep_bushels : ℕ) (chickens : ℕ) (chicken_bushels : ℕ) : ℕ :=
  (cows + sheep) * cow_sheep_bushels + chicken_bushels

/-- Theorem: Dany needs 17 bushels for his animals for one day -/
theorem dany_farm_bushels :
  bushels_needed 4 3 2 7 3 = 17 := by
  sorry

end dany_farm_bushels_l2769_276947


namespace certain_number_problem_l2769_276919

theorem certain_number_problem (x : ℝ) : 
  (0.3 * x) - (1/3) * (0.3 * x) = 36 → x = 180 := by
  sorry

end certain_number_problem_l2769_276919


namespace min_value_and_solution_set_l2769_276964

def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |x - a|

theorem min_value_and_solution_set (a : ℝ) (h1 : a > 0) :
  (∃ (m : ℝ), m = -3 ∧ ∀ x, f a x ≥ m) →
  (a = 1 ∧ 
   ∀ x, |f a x| ≤ 2 ↔ a / 2 - 2 ≤ x ∧ x < a / 2) :=
by sorry

end min_value_and_solution_set_l2769_276964


namespace two_truthful_students_l2769_276978

/-- Represents the performance of a student in the exam -/
inductive Performance
| Good
| NotGood

/-- Represents a student -/
inductive Student
| A
| B
| C
| D

/-- The statement made by each student -/
def statement (s : Student) (performances : Student → Performance) : Prop :=
  match s with
  | Student.A => ∀ s, performances s = Performance.NotGood
  | Student.B => ∃ s, performances s = Performance.Good
  | Student.C => performances Student.B = Performance.NotGood ∨ performances Student.D = Performance.NotGood
  | Student.D => performances Student.D = Performance.NotGood

/-- Checks if a student's statement is true -/
def isTruthful (s : Student) (performances : Student → Performance) : Prop :=
  statement s performances

theorem two_truthful_students :
  ∃ (performances : Student → Performance),
    (isTruthful Student.B performances ∧ isTruthful Student.C performances) ∧
    (¬isTruthful Student.A performances ∧ ¬isTruthful Student.D performances) ∧
    (∀ (s1 s2 : Student), isTruthful s1 performances ∧ isTruthful s2 performances ∧ s1 ≠ s2 →
      ∀ (s : Student), s ≠ s1 ∧ s ≠ s2 → ¬isTruthful s performances) :=
  sorry

end two_truthful_students_l2769_276978


namespace original_total_is_390_l2769_276910

/-- Represents the number of movies in each format --/
structure MovieCollection where
  dvd : ℕ
  bluray : ℕ
  digital : ℕ

/-- The original collection of movies --/
def original : MovieCollection := sorry

/-- The updated collection after purchasing new movies --/
def updated : MovieCollection := sorry

/-- The ratio of the original collection --/
def original_ratio : MovieCollection := ⟨7, 2, 1⟩

/-- The ratio of the updated collection --/
def updated_ratio : MovieCollection := ⟨13, 4, 2⟩

/-- The number of new Blu-ray movies purchased --/
def new_bluray : ℕ := 5

/-- The number of new digital movies purchased --/
def new_digital : ℕ := 3

theorem original_total_is_390 :
  ∃ (x : ℕ),
    original.dvd = 7 * x ∧
    original.bluray = 2 * x ∧
    original.digital = x ∧
    updated.dvd = original.dvd ∧
    updated.bluray = original.bluray + new_bluray ∧
    updated.digital = original.digital + new_digital ∧
    (updated.dvd : ℚ) / updated_ratio.dvd = (updated.bluray : ℚ) / updated_ratio.bluray ∧
    (updated.dvd : ℚ) / updated_ratio.dvd = (updated.digital : ℚ) / updated_ratio.digital ∧
    original.dvd + original.bluray + original.digital = 390 :=
by
  sorry

end original_total_is_390_l2769_276910


namespace minimum_area_of_rectangle_l2769_276967

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Checks if the actual dimensions are within the reported range --/
def withinReportedRange (reported : Rectangle) (actual : Rectangle) : Prop :=
  (actual.length ≥ reported.length - 0.5) ∧
  (actual.length ≤ reported.length + 0.5) ∧
  (actual.width ≥ reported.width - 0.5) ∧
  (actual.width ≤ reported.width + 0.5)

/-- Checks if the length is at least twice the width --/
def lengthAtLeastTwiceWidth (r : Rectangle) : Prop :=
  r.length ≥ 2 * r.width

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- The reported dimensions of the tile --/
def reportedDimensions : Rectangle :=
  { length := 4, width := 6 }

theorem minimum_area_of_rectangle :
  ∃ (minRect : Rectangle),
    withinReportedRange reportedDimensions minRect ∧
    lengthAtLeastTwiceWidth minRect ∧
    area minRect = 60.5 ∧
    ∀ (r : Rectangle),
      withinReportedRange reportedDimensions r →
      lengthAtLeastTwiceWidth r →
      area r ≥ 60.5 := by
  sorry

end minimum_area_of_rectangle_l2769_276967


namespace return_trip_duration_l2769_276986

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of the plane in still air
  w₁ : ℝ  -- wind speed against the plane
  w₂ : ℝ  -- wind speed with the plane

/-- The conditions of the flight scenario -/
def flight_conditions (f : FlightScenario) : Prop :=
  f.d / (f.p - f.w₁) = 120 ∧  -- outbound trip takes 120 minutes
  f.d / (f.p + f.w₂) = f.d / f.p - 10  -- return trip is 10 minutes faster than in still air

/-- The theorem to prove -/
theorem return_trip_duration (f : FlightScenario) 
  (h : flight_conditions f) : f.d / (f.p + f.w₂) = 72 := by
  sorry


end return_trip_duration_l2769_276986


namespace faye_age_l2769_276956

/-- Represents the ages of Chad, Diana, Eduardo, and Faye --/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 5 ∧
  ages.eduardo = ages.chad + 3 ∧
  ages.faye = ages.chad + 2 ∧
  ages.diana = 18

theorem faye_age (ages : Ages) :
  satisfiesConditions ages → ages.faye = 22 :=
by
  sorry

end faye_age_l2769_276956


namespace intersection_of_M_and_N_l2769_276954

open Set

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x < 0}
def N : Set ℝ := {x | x - 3 ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Ioo 2 3 ∪ Iic 3 := by sorry

end intersection_of_M_and_N_l2769_276954


namespace f_composition_at_two_l2769_276934

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_composition_at_two : f (f (f 2)) = 2 := by
  sorry

end f_composition_at_two_l2769_276934


namespace x_range_for_inequality_l2769_276969

theorem x_range_for_inequality (x : ℝ) :
  (∀ m : ℝ, |m| ≤ 1 → x^2 - 2 > m*x) →
  x < -2 ∨ x > 2 :=
by sorry

end x_range_for_inequality_l2769_276969


namespace exists_m_n_for_any_d_l2769_276981

theorem exists_m_n_for_any_d (d : ℤ) : ∃ (m n : ℤ), d = (n - 2*m + 1) / (m^2 - n) := by
  sorry

end exists_m_n_for_any_d_l2769_276981


namespace rect_to_cylindrical_l2769_276933

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) (h_x : x = 4) (h_y : y = 4 * Real.sqrt 3) (h_z : z = 5) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 8 ∧ θ = Real.pi / 3 ∧ z = 5 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end rect_to_cylindrical_l2769_276933


namespace investment_strategy_l2769_276960

/-- Represents the investment and profit parameters for a manufacturing company's production lines. -/
structure ProductionParameters where
  initialInvestmentA : ℝ  -- Initial investment in production line A (in million yuan)
  profitRateA : ℝ         -- Profit rate for production line A (profit per 10,000 yuan invested)
  investmentReduction : ℝ -- Reduction in investment for A (in million yuan)
  profitIncreaseRate : ℝ  -- Rate of profit increase for A (as a percentage)
  profitRateB : ℝ → ℝ     -- Profit rate function for production line B
  a : ℝ                   -- Parameter a for production line B's profit rate

/-- The main theorem about the manufacturing company's investment strategy. -/
theorem investment_strategy 
  (params : ProductionParameters) 
  (h_initialInvestmentA : params.initialInvestmentA = 50)
  (h_profitRateA : params.profitRateA = 1.5)
  (h_profitIncreaseRate : params.profitIncreaseRate = 0.005)
  (h_profitRateB : params.profitRateB = fun x => 1.5 * (params.a - 0.013 * x))
  (h_a_positive : params.a > 0) :
  (∃ x_range : Set ℝ, x_range = {x | 0 < x ∧ x ≤ 300} ∧ 
    ∀ x ∈ x_range, 
      (params.initialInvestmentA - x) * params.profitRateA * (1 + x * params.profitIncreaseRate) ≥ 
      params.initialInvestmentA * params.profitRateA) ∧
  (∃ a_max : ℝ, a_max = 5.5 ∧
    ∀ x > 0, x * params.profitRateB x ≤ 
      (params.initialInvestmentA - x) * params.profitRateA * (1 + x * params.profitIncreaseRate) ∧
    params.a ≤ a_max) := by
  sorry

end investment_strategy_l2769_276960


namespace average_mpg_calculation_l2769_276965

theorem average_mpg_calculation (initial_reading final_reading : ℕ) (fuel_used : ℕ) :
  initial_reading = 56200 →
  final_reading = 57150 →
  fuel_used = 50 →
  (final_reading - initial_reading : ℚ) / fuel_used = 19 := by
  sorry

end average_mpg_calculation_l2769_276965
