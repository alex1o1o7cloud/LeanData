import Mathlib

namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2447_244719

/-- The y-intercept of a line with slope 1 passing through the midpoint of a line segment --/
theorem y_intercept_of_line (x₁ y₁ x₂ y₂ : ℝ) :
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  let slope := 1
  let y_intercept := midpoint_y - slope * midpoint_x
  x₁ = 2 ∧ y₁ = 8 ∧ x₂ = 14 ∧ y₂ = 4 →
  y_intercept = -2 := by
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2447_244719


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l2447_244783

theorem largest_prime_factors_difference (n : Nat) (h : n = 171689) : 
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧ 
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
    p ∣ n ∧ 
    q ∣ n ∧ 
    p - q = 282 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l2447_244783


namespace NUMINAMATH_CALUDE_money_distribution_l2447_244784

theorem money_distribution (x y z : ℝ) : 
  x + (y/2 + z/2) = 90 →
  y + (x/2 + z/2) = 70 →
  z + (x/2 + y/2) = 56 →
  y = 32 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2447_244784


namespace NUMINAMATH_CALUDE_f_of_g_of_three_l2447_244777

def f (x : ℝ) : ℝ := 2 * x - 5

def g (x : ℝ) : ℝ := x + 2

theorem f_of_g_of_three : f (1 + g 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_three_l2447_244777


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l2447_244786

/-- Given a cubic function f(x) with a maximum at x=1, prove that a/b = -2/3 --/
theorem max_value_implies_ratio (a b : ℝ) :
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∀ x, f x ≤ f 1) ∧ (f 1 = 10) →
  a / b = -2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l2447_244786


namespace NUMINAMATH_CALUDE_sum_in_base6_l2447_244791

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

theorem sum_in_base6 :
  let a := base6ToDecimal [5, 5, 5, 1]
  let b := base6ToDecimal [5, 5, 1]
  let c := base6ToDecimal [5, 1]
  decimalToBase6 (a + b + c) = [2, 2, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base6_l2447_244791


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2447_244774

theorem min_value_squared_sum (x y z a : ℝ) (h : x + 2*y + 3*z = a) :
  x^2 + y^2 + z^2 ≥ a^2 / 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2447_244774


namespace NUMINAMATH_CALUDE_min_trees_for_three_types_l2447_244730

/-- Represents the four types of trees in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset ℕ)
  (type : ℕ → TreeType)
  (total_trees : trees.card = 100)
  (four_types_in_85 : ∀ s : Finset ℕ, s ⊆ trees → s.card = 85 → 
    (∃ i ∈ s, type i = TreeType.Birch) ∧
    (∃ i ∈ s, type i = TreeType.Spruce) ∧
    (∃ i ∈ s, type i = TreeType.Pine) ∧
    (∃ i ∈ s, type i = TreeType.Aspen))

/-- The main theorem stating the minimum number of trees to guarantee at least three types -/
theorem min_trees_for_three_types (g : Grove) :
  ∀ s : Finset ℕ, s ⊆ g.trees → s.card ≥ 69 →
    (∃ t1 t2 t3 : TreeType, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
      (∃ i ∈ s, g.type i = t1) ∧
      (∃ i ∈ s, g.type i = t2) ∧
      (∃ i ∈ s, g.type i = t3)) :=
by sorry

end NUMINAMATH_CALUDE_min_trees_for_three_types_l2447_244730


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l2447_244740

theorem average_of_remaining_numbers
  (total : ℝ) (group1 : ℝ) (group2 : ℝ) (group3 : ℝ)
  (h1 : total = 6 * 3.95)
  (h2 : group1 = 2 * 4.4)
  (h3 : group2 = 2 * 3.85)
  (h4 : group3 = total - (group1 + group2)) :
  group3 / 2 = 3.6 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l2447_244740


namespace NUMINAMATH_CALUDE_car_speed_problem_l2447_244700

/-- Given two cars traveling on a 500-mile highway from opposite ends, 
    one at speed v and the other at 60 mph, meeting after 5 hours, 
    prove that the speed v of the first car is 40 mph. -/
theorem car_speed_problem (v : ℝ) 
  (h1 : v > 0) -- Assuming speed is positive
  (h2 : 5 * v + 5 * 60 = 500) : v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2447_244700


namespace NUMINAMATH_CALUDE_star_example_l2447_244768

-- Define the star operation
def star (a b c d : ℚ) : ℚ := a * c * (d / b)

-- Theorem statement
theorem star_example : star (5/9) (4/6) = 40/3 := by sorry

end NUMINAMATH_CALUDE_star_example_l2447_244768


namespace NUMINAMATH_CALUDE_shop_owner_gain_percentage_l2447_244790

/-- Calculates the shop owner's total gain percentage --/
theorem shop_owner_gain_percentage
  (cost_A cost_B cost_C : ℝ)
  (markup_A markup_B markup_C : ℝ)
  (quantity_A quantity_B quantity_C : ℝ)
  (discount tax : ℝ)
  (h1 : cost_A = 4)
  (h2 : cost_B = 6)
  (h3 : cost_C = 8)
  (h4 : markup_A = 0.25)
  (h5 : markup_B = 0.30)
  (h6 : markup_C = 0.20)
  (h7 : quantity_A = 25)
  (h8 : quantity_B = 15)
  (h9 : quantity_C = 10)
  (h10 : discount = 0.05)
  (h11 : tax = 0.05) :
  ∃ (gain_percentage : ℝ), abs (gain_percentage - 0.2487) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_shop_owner_gain_percentage_l2447_244790


namespace NUMINAMATH_CALUDE_factory_output_equation_l2447_244733

/-- Represents the factory's output model -/
def factory_output (initial_output : ℝ) (growth_rate : ℝ) (months : ℕ) : ℝ :=
  initial_output * (1 + growth_rate) ^ months

/-- Theorem stating that the equation 500(1+x)^2 = 720 correctly represents the factory's output in March -/
theorem factory_output_equation (x : ℝ) : 
  factory_output 500 x 2 = 720 ↔ 500 * (1 + x)^2 = 720 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_equation_l2447_244733


namespace NUMINAMATH_CALUDE_pie_chart_statement_is_false_l2447_244707

-- Define the characteristics of different chart types
def BarChart : Type := Unit
def LineChart : Type := Unit
def PieChart : Type := Unit

-- Define what each chart type can represent
def represents_amount (chart : Type) : Prop := sorry
def represents_changes (chart : Type) : Prop := sorry
def represents_part_whole (chart : Type) : Prop := sorry

-- State the known characteristics of each chart type
axiom bar_chart_amount : represents_amount BarChart
axiom line_chart_amount_and_changes : represents_amount LineChart ∧ represents_changes LineChart
axiom pie_chart_part_whole : represents_part_whole PieChart

-- The statement we want to prove false
def pie_chart_statement : Prop :=
  represents_amount PieChart ∧ represents_changes PieChart

-- The theorem to prove
theorem pie_chart_statement_is_false : ¬pie_chart_statement := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_statement_is_false_l2447_244707


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l2447_244716

/-- Given a hyperbola and a circle with specific intersection points, 
    prove that the fourth intersection point has specific coordinates. -/
theorem fourth_intersection_point 
  (C : Set (ℝ × ℝ)) -- The circle
  (h : C.Nonempty) -- The circle is not empty
  (hyp : Set (ℝ × ℝ)) -- The hyperbola
  (hyp_eq : ∀ p ∈ hyp, p.1 * p.2 = 2) -- Equation of the hyperbola
  (intersect : C ∩ hyp = {(4, 1/2), (-2, -1), (2/3, 3), (-1/2, -4)}) -- Intersection points
  : (-1/2, -4) ∈ C ∩ hyp := by
  sorry


end NUMINAMATH_CALUDE_fourth_intersection_point_l2447_244716


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_and_sum_l2447_244770

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula_and_sum 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 = 11 →
  a 2 + a 6 = 18 →
  (∀ n : ℕ, b n = a n + 3^n) →
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, S n = n^2 + 2*n - 3/2 + 3^(n+1)/2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_and_sum_l2447_244770


namespace NUMINAMATH_CALUDE_cylinder_j_value_l2447_244754

/-- The value of J for a cylinder with specific properties -/
theorem cylinder_j_value (h d r : ℝ) (j : ℝ) : 
  h > 0 → d > 0 → r > 0 →
  h = d →  -- Cylinder height equals diameter
  r = d / 2 →  -- Radius is half the diameter
  6 * 3^2 = 2 * π * r^2 + π * d * h →  -- Surface area of cylinder equals surface area of cube
  j * π / 6 = π * r^2 * h →  -- Volume of cylinder
  j = 324 * Real.sqrt π := by
sorry

end NUMINAMATH_CALUDE_cylinder_j_value_l2447_244754


namespace NUMINAMATH_CALUDE_smallest_n_for_reducible_fraction_l2447_244759

theorem smallest_n_for_reducible_fraction : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 13) ∧ k ∣ (5*m + 6))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 13) ∧ k ∣ (5*n + 6)) ∧
  n = 84 := by
  sorry

#check smallest_n_for_reducible_fraction

end NUMINAMATH_CALUDE_smallest_n_for_reducible_fraction_l2447_244759


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2447_244762

theorem quadratic_inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2447_244762


namespace NUMINAMATH_CALUDE_simplify_expression_l2447_244718

theorem simplify_expression (x : ℝ) (h : x ≥ 2) :
  |2 - x| + (Real.sqrt (x - 2))^2 - Real.sqrt (4 * x^2 - 4 * x + 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2447_244718


namespace NUMINAMATH_CALUDE_investment_period_proof_l2447_244755

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_period_proof (principal : ℝ) (rate1 rate2 : ℝ) (time : ℝ) 
    (h1 : principal = 900)
    (h2 : rate1 = 0.04)
    (h3 : rate2 = 0.045)
    (h4 : simpleInterest principal rate2 time - simpleInterest principal rate1 time = 31.5) :
  time = 7 := by
  sorry

end NUMINAMATH_CALUDE_investment_period_proof_l2447_244755


namespace NUMINAMATH_CALUDE_philip_banana_count_l2447_244763

/-- The number of banana groups in Philip's collection -/
def banana_groups : ℕ := 2

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 145

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := banana_groups * bananas_per_group

theorem philip_banana_count : total_bananas = 290 := by
  sorry

end NUMINAMATH_CALUDE_philip_banana_count_l2447_244763


namespace NUMINAMATH_CALUDE_uncle_zhang_age_uncle_zhang_age_proof_l2447_244738

theorem uncle_zhang_age : Nat → Nat → Prop :=
  fun zhang_age li_age =>
    zhang_age + li_age = 56 ∧
    2 * (li_age - (li_age - zhang_age)) = li_age ∧
    zhang_age = 24

-- The proof is omitted
theorem uncle_zhang_age_proof : ∃ (zhang_age li_age : Nat), uncle_zhang_age zhang_age li_age :=
  sorry

end NUMINAMATH_CALUDE_uncle_zhang_age_uncle_zhang_age_proof_l2447_244738


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l2447_244796

-- Part 1
theorem range_of_x (x : ℝ) :
  (x^2 - 4*x + 3 < 0) → ((x - 3)^2 < 1) → (2 < x ∧ x < 3) := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 ≥ 0 → (x - 3)^2 ≥ 1)) →
  (∃ x : ℝ, (x - 3)^2 ≥ 1 ∧ x^2 - 4*a*x + 3*a^2 < 0) →
  (4/3 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l2447_244796


namespace NUMINAMATH_CALUDE_average_girls_per_grade_l2447_244757

/-- Represents a grade with its student composition -/
structure Grade where
  girls : ℕ
  boys : ℕ
  clubGirls : ℕ
  clubBoys : ℕ

/-- The total number of grades -/
def totalGrades : ℕ := 3

/-- List of grades with their student composition -/
def grades : List Grade := [
  { girls := 28, boys := 35, clubGirls := 6, clubBoys := 6 },
  { girls := 45, boys := 42, clubGirls := 7, clubBoys := 8 },
  { girls := 38, boys := 51, clubGirls := 3, clubBoys := 7 }
]

/-- Calculate the total number of girls across all grades -/
def totalGirls : ℕ := (grades.map (·.girls)).sum

/-- Theorem: The average number of girls per grade is 37 -/
theorem average_girls_per_grade :
  totalGirls / totalGrades = 37 := by sorry

end NUMINAMATH_CALUDE_average_girls_per_grade_l2447_244757


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2447_244773

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y, 2*x - 3*y + 6 = 0 → bx - 3*y - 4 = 0 → 
    (2/3) * (b/3) = -1) → 
  b = -9/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2447_244773


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l2447_244710

theorem orange_juice_percentage (total : ℝ) (watermelon_percent : ℝ) (grape_ounces : ℝ) :
  total = 140 →
  watermelon_percent = 60 →
  grape_ounces = 35 →
  (15 : ℝ) / 100 * total = total - watermelon_percent / 100 * total - grape_ounces :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l2447_244710


namespace NUMINAMATH_CALUDE_exist_six_consecutive_naturals_lcm_property_l2447_244701

theorem exist_six_consecutive_naturals_lcm_property :
  ∃ n : ℕ, lcm (lcm n (n + 1)) (n + 2) > lcm (lcm (n + 3) (n + 4)) (n + 5) := by
  sorry

end NUMINAMATH_CALUDE_exist_six_consecutive_naturals_lcm_property_l2447_244701


namespace NUMINAMATH_CALUDE_line_slope_is_one_l2447_244756

/-- Given a line in the xy-plane with y-intercept -2 and passing through the midpoint
    of the line segment with endpoints (2, 8) and (14, 4), its slope is 1. -/
theorem line_slope_is_one (m : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ m ↔ y = x - 2) →  -- y-intercept is -2
  ((8 : ℝ), 6) ∈ m →  -- passes through midpoint ((2+14)/2, (8+4)/2) = (8, 6)
  (∃ (k b : ℝ), ∀ (x y : ℝ), (x, y) ∈ m ↔ y = k * x + b) →  -- m is a line
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ m ↔ y = k * x - 2) →  -- combine line equation with y-intercept
  ∀ (x y : ℝ), (x, y) ∈ m ↔ y = x - 2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l2447_244756


namespace NUMINAMATH_CALUDE_eighth_group_sample_l2447_244793

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (m : Nat) (k : Nat) : Nat :=
  (k - 1) * 10 + (m + k) % 10

/-- The problem statement as a theorem -/
theorem eighth_group_sample :
  ∀ m : Nat,
  m = 8 →
  systematicSample m 8 = 76 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_sample_l2447_244793


namespace NUMINAMATH_CALUDE_parabola_point_distance_to_x_axis_l2447_244799

/-- Prove that for a point on a specific parabola with a given distance to the focus,
    its distance to the x-axis is 15/16 -/
theorem parabola_point_distance_to_x_axis 
  (x₀ y₀ : ℝ) -- Coordinates of point M
  (h_parabola : x₀^2 = (1/4) * y₀) -- M is on the parabola
  (h_focus_dist : (x₀^2 + (y₀ - 1/16)^2) = 1) -- Distance from M to focus is 1
  : |y₀| = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_to_x_axis_l2447_244799


namespace NUMINAMATH_CALUDE_distance_before_gas_is_32_l2447_244745

/-- The distance driven before stopping for gas -/
def distance_before_gas (total_distance remaining_distance : ℕ) : ℕ :=
  total_distance - remaining_distance

/-- Theorem: The distance driven before stopping for gas is 32 miles -/
theorem distance_before_gas_is_32 :
  distance_before_gas 78 46 = 32 := by
  sorry

end NUMINAMATH_CALUDE_distance_before_gas_is_32_l2447_244745


namespace NUMINAMATH_CALUDE_limit_evaluation_l2447_244717

theorem limit_evaluation : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((1 + 2*n : ℝ)^3 - 8*n^5) / ((1 + 2*n)^2 + 4*n^2) + 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_evaluation_l2447_244717


namespace NUMINAMATH_CALUDE_duck_count_l2447_244736

theorem duck_count (total_animals : ℕ) (total_legs : ℕ) (duck_legs : ℕ) (horse_legs : ℕ) 
  (h1 : total_animals = 11)
  (h2 : total_legs = 30)
  (h3 : duck_legs = 2)
  (h4 : horse_legs = 4) :
  ∃ (ducks horses : ℕ),
    ducks + horses = total_animals ∧
    ducks * duck_legs + horses * horse_legs = total_legs ∧
    ducks = 7 :=
by sorry

end NUMINAMATH_CALUDE_duck_count_l2447_244736


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l2447_244775

theorem triangle_existence_condition 
  (k : ℝ) (α : ℝ) (m_a : ℝ) 
  (h_k : k > 0) 
  (h_α : 0 < α ∧ α < π) 
  (h_m_a : m_a > 0) : 
  (∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b + c = k ∧
    ∃ (β γ : ℝ), 
      0 < β ∧ 0 < γ ∧
      α + β + γ = π ∧
      m_a = (b * c * Real.sin α) / (b + c)) ↔ 
  m_a ≤ (k / 2) * ((1 - Real.sin (α / 2)) / Real.cos (α / 2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l2447_244775


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l2447_244713

def M (a : ℝ) : Set ℝ := {1, a}
def N : Set ℝ := {-1, 0, 1}

theorem a_zero_sufficient_not_necessary (a : ℝ) :
  (a = 0 → M a ⊆ N) ∧ ¬(M a ⊆ N → a = 0) :=
sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_l2447_244713


namespace NUMINAMATH_CALUDE_unique_solution_l2447_244708

/-- The number of positive integer solutions to the equation 2x + 3y = 8 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 2 * p.1 + 3 * p.2 = 8) (Finset.product (Finset.range 9) (Finset.range 9))).card

/-- Theorem stating that there is exactly one positive integer solution to 2x + 3y = 8 -/
theorem unique_solution : solution_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2447_244708


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2447_244735

theorem sphere_surface_area (diameter : ℝ) (h : diameter = 10) :
  4 * Real.pi * (diameter / 2)^2 = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2447_244735


namespace NUMINAMATH_CALUDE_xy_value_l2447_244753

theorem xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x / y = 81) (h4 : y = 0.2222222222222222) :
  x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2447_244753


namespace NUMINAMATH_CALUDE_dad_steps_l2447_244741

/-- Represents the number of steps taken by each person --/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad, Masha, and Yasha --/
def step_relation (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha ∧ 5 * s.masha = 3 * s.yasha

/-- The total number of steps taken by Masha and Yasha --/
def total_masha_yasha (s : Steps) : ℕ := s.masha + s.yasha

/-- Theorem stating that given the conditions, Dad took 90 steps --/
theorem dad_steps :
  ∀ s : Steps,
  step_relation s →
  total_masha_yasha s = 400 →
  s.dad = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l2447_244741


namespace NUMINAMATH_CALUDE_trig_identity_l2447_244747

theorem trig_identity (a : ℝ) (h : Real.sin (π / 3 - a) = 1 / 3) :
  Real.cos (5 * π / 6 - a) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2447_244747


namespace NUMINAMATH_CALUDE_jamie_rice_purchase_l2447_244724

/-- The price of rice in cents per pound -/
def rice_price : ℚ := 60

/-- The price of flour in cents per pound -/
def flour_price : ℚ := 30

/-- The total amount of rice and flour bought in pounds -/
def total_amount : ℚ := 30

/-- The total amount spent in cents -/
def total_spent : ℚ := 1500

/-- The amount of rice bought in pounds -/
def rice_amount : ℚ := 20

theorem jamie_rice_purchase :
  ∃ (flour_amount : ℚ),
    rice_amount + flour_amount = total_amount ∧
    rice_price * rice_amount + flour_price * flour_amount = total_spent :=
by sorry

end NUMINAMATH_CALUDE_jamie_rice_purchase_l2447_244724


namespace NUMINAMATH_CALUDE_three_zeros_condition_l2447_244787

-- Define the function f(x) = x^3 + ax + 2
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

-- Theorem statement
theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l2447_244787


namespace NUMINAMATH_CALUDE_tan_beta_value_l2447_244789

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2447_244789


namespace NUMINAMATH_CALUDE_matrix_power_difference_l2447_244779

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]

theorem matrix_power_difference :
  A^5 - 3 * A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l2447_244779


namespace NUMINAMATH_CALUDE_inequality_proof_l2447_244760

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ≥ 6 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2447_244760


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l2447_244765

/-- Rotation of 180° clockwise about the origin in 2D plane -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (-3, 2)
  let D : ℝ × ℝ := (-2, 5)
  let C' : ℝ × ℝ := (3, -2)
  let D' : ℝ × ℝ := (2, -5)
  rotate180 C = C' ∧ rotate180 D = D' :=
by sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l2447_244765


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_six_l2447_244798

theorem no_linear_term_implies_a_equals_six (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (2*x + a) * (3 - x) = b * x^2 + c) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_equals_six_l2447_244798


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2447_244709

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2447_244709


namespace NUMINAMATH_CALUDE_ln_inequality_l2447_244704

theorem ln_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < Real.exp 1) :
  a * Real.log b > b * Real.log a := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_l2447_244704


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2447_244781

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 3 - 4 * Complex.I) → z = -4 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2447_244781


namespace NUMINAMATH_CALUDE_sand_bag_cost_l2447_244712

/-- The cost of a bag of sand given the dimensions of a square sandbox,
    the area covered by one bag, and the total cost to fill the sandbox. -/
theorem sand_bag_cost
  (sandbox_side : ℝ)
  (bag_area : ℝ)
  (total_cost : ℝ)
  (h_square : sandbox_side = 3)
  (h_bag : bag_area = 3)
  (h_cost : total_cost = 12) :
  total_cost / (sandbox_side ^ 2 / bag_area) = 4 := by
sorry

end NUMINAMATH_CALUDE_sand_bag_cost_l2447_244712


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l2447_244728

/-- Given a sequence {aₙ} satisfying 4aₙ₊₁ - 4aₙ - 9 = 0 for all n,
    prove that {aₙ} is an arithmetic sequence with a common difference of 9/4. -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) 
    (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
    ∃ d, d = 9/4 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l2447_244728


namespace NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l2447_244764

/-- Given a set of boxes with pens and pencils, this theorem proves
    the number of boxes containing neither pens nor pencils. -/
theorem boxes_with_neither_pens_nor_pencils
  (total_boxes : ℕ)
  (pencil_boxes : ℕ)
  (pen_boxes : ℕ)
  (both_boxes : ℕ)
  (h1 : total_boxes = 10)
  (h2 : pencil_boxes = 6)
  (h3 : pen_boxes = 3)
  (h4 : both_boxes = 2)
  : total_boxes - (pencil_boxes + pen_boxes - both_boxes) = 3 := by
  sorry

#check boxes_with_neither_pens_nor_pencils

end NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l2447_244764


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l2447_244722

theorem missing_digit_divisible_by_three (x : Nat) :
  (x < 10) →
  (246 * 100 + x * 10 + 9) % 3 = 0 →
  x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l2447_244722


namespace NUMINAMATH_CALUDE_ninth_power_negative_fourth_l2447_244739

theorem ninth_power_negative_fourth : (1 / 9)^(-1/4 : ℝ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ninth_power_negative_fourth_l2447_244739


namespace NUMINAMATH_CALUDE_complex_quadrant_implies_m_range_l2447_244731

def z (m : ℝ) : ℂ := Complex.mk (m + 1) (3 - m)

def in_second_or_fourth_quadrant (z : ℂ) : Prop :=
  z.re * z.im > 0

theorem complex_quadrant_implies_m_range (m : ℝ) :
  in_second_or_fourth_quadrant (z m) → m ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_implies_m_range_l2447_244731


namespace NUMINAMATH_CALUDE_markup_percentage_is_45_l2447_244703

/-- Given a cost price, discount, and profit percentage, calculate the markup percentage. -/
def calculate_markup_percentage (cost_price discount : ℚ) (profit_percentage : ℚ) : ℚ :=
  let selling_price := cost_price * (1 + profit_percentage)
  let marked_price := selling_price + discount
  let markup := marked_price - cost_price
  (markup / cost_price) * 100

/-- Theorem: Given the specific values in the problem, the markup percentage is 45%. -/
theorem markup_percentage_is_45 :
  let cost_price : ℚ := 180
  let discount : ℚ := 45
  let profit_percentage : ℚ := 0.20
  calculate_markup_percentage cost_price discount profit_percentage = 45 := by
  sorry

#eval calculate_markup_percentage 180 45 0.20

end NUMINAMATH_CALUDE_markup_percentage_is_45_l2447_244703


namespace NUMINAMATH_CALUDE_simplify_expression_l2447_244706

theorem simplify_expression (m : ℝ) : (3*m + 2) - 3*(m^2 - m + 1) + (3 - 6*m) = -3*m^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2447_244706


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2447_244792

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2447_244792


namespace NUMINAMATH_CALUDE_factorization_equality_l2447_244751

theorem factorization_equality (x y z : ℝ) : x^2 + x*y - x*z - y*z = (x + y)*(x - z) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2447_244751


namespace NUMINAMATH_CALUDE_equal_angles_with_perpendicular_circle_l2447_244737

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the properties and relations
variable (passes_through : Circle → Point → Prop)
variable (tangent_to : Circle → Circle → Prop)
variable (perpendicular_to : Circle → Circle → Prop)
variable (angle_between : Circle → Circle → ℝ)

-- State the theorem
theorem equal_angles_with_perpendicular_circle
  (A B : Point) (S S₁ S₂ S₃ : Circle)
  (h1 : passes_through S₁ A ∧ passes_through S₁ B)
  (h2 : passes_through S₂ A ∧ passes_through S₂ B)
  (h3 : tangent_to S₁ S)
  (h4 : tangent_to S₂ S)
  (h5 : perpendicular_to S₃ S) :
  angle_between S₃ S₁ = angle_between S₃ S₂ :=
by sorry

end NUMINAMATH_CALUDE_equal_angles_with_perpendicular_circle_l2447_244737


namespace NUMINAMATH_CALUDE_a_greater_than_b_l2447_244711

theorem a_greater_than_b : ∀ x : ℝ, (x - 3)^2 > (x - 2) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l2447_244711


namespace NUMINAMATH_CALUDE_mirror_side_length_l2447_244726

/-- Proves that the length of each side of a square mirror is 18 inches, given the specified conditions --/
theorem mirror_side_length :
  ∀ (wall_width wall_length mirror_area : ℝ),
    wall_width = 32 →
    wall_length = 20.25 →
    mirror_area = (wall_width * wall_length) / 2 →
    ∃ (mirror_side : ℝ),
      mirror_side * mirror_side = mirror_area ∧
      mirror_side = 18 :=
by sorry

end NUMINAMATH_CALUDE_mirror_side_length_l2447_244726


namespace NUMINAMATH_CALUDE_decagon_perimeter_30_l2447_244767

/-- A regular decagon is a polygon with 10 sides of equal length. -/
structure RegularDecagon where
  side_length : ℝ
  sides : Nat
  sides_eq : sides = 10

/-- The perimeter of a polygon is the sum of the lengths of its sides. -/
def perimeter (d : RegularDecagon) : ℝ := d.side_length * d.sides

theorem decagon_perimeter_30 (d : RegularDecagon) (h : d.side_length = 3) : perimeter d = 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_30_l2447_244767


namespace NUMINAMATH_CALUDE_not_p_sufficient_for_not_q_l2447_244782

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := |3*x - 4| > 2

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0

/-- Theorem stating that not p implies not q, but not q does not necessarily imply not p -/
theorem not_p_sufficient_for_not_q :
  (∃ x : ℝ, ¬(p x) ∧ ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_for_not_q_l2447_244782


namespace NUMINAMATH_CALUDE_sphere_only_circular_views_l2447_244758

-- Define the geometric shapes
inductive Shape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the views
inductive View
  | Front
  | Left
  | Top

-- Define a function to check if a view is circular for a given shape
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | Shape.Cone, View.Top => True
  | _, _ => False

-- Define a function to check if all views are circular for a given shape
def allViewsCircular (s : Shape) : Prop :=
  isCircularView s View.Front ∧ isCircularView s View.Left ∧ isCircularView s View.Top

-- Theorem statement
theorem sphere_only_circular_views :
  ∀ s : Shape, allViewsCircular s ↔ s = Shape.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_circular_views_l2447_244758


namespace NUMINAMATH_CALUDE_anna_win_probability_l2447_244723

-- Define the game state as the sum modulo 4
inductive GameState
| Zero
| One
| Two
| Three

-- Define the die roll
def DieRoll : Type := Fin 6

-- Define the probability of winning for each game state
def winProbability : GameState → ℚ
| GameState.Zero => 0
| GameState.One => 50/99
| GameState.Two => 60/99
| GameState.Three => 62/99

-- Define the transition probability function
def transitionProbability (s : GameState) (r : DieRoll) : GameState :=
  match s, r.val + 1 with
  | GameState.Zero, n => match n % 4 with
    | 0 => GameState.Zero
    | 1 => GameState.One
    | 2 => GameState.Two
    | 3 => GameState.Three
    | _ => GameState.Zero  -- This case should never occur
  | GameState.One, n => match n % 4 with
    | 0 => GameState.One
    | 1 => GameState.Two
    | 2 => GameState.Three
    | 3 => GameState.Zero
    | _ => GameState.One  -- This case should never occur
  | GameState.Two, n => match n % 4 with
    | 0 => GameState.Two
    | 1 => GameState.Three
    | 2 => GameState.Zero
    | 3 => GameState.One
    | _ => GameState.Two  -- This case should never occur
  | GameState.Three, n => match n % 4 with
    | 0 => GameState.Three
    | 1 => GameState.Zero
    | 2 => GameState.One
    | 3 => GameState.Two
    | _ => GameState.Three  -- This case should never occur

-- Theorem statement
theorem anna_win_probability :
  (1 : ℚ) / 6 * (1 - winProbability GameState.Zero) +
  1 / 3 * (1 - winProbability GameState.One) +
  1 / 3 * (1 - winProbability GameState.Two) +
  1 / 6 * (1 - winProbability GameState.Three) = 52 / 99 :=
by sorry


end NUMINAMATH_CALUDE_anna_win_probability_l2447_244723


namespace NUMINAMATH_CALUDE_f_three_zeros_c_range_l2447_244788

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + c

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 12

-- Theorem statement
theorem f_three_zeros_c_range (c : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧ f c x₃ = 0) →
  -16 < c ∧ c < 16 := by sorry

end NUMINAMATH_CALUDE_f_three_zeros_c_range_l2447_244788


namespace NUMINAMATH_CALUDE_max_candy_pieces_l2447_244732

theorem max_candy_pieces (n : ℕ) (μ : ℚ) (min_pieces : ℕ) : 
  n = 35 → 
  μ = 6 → 
  min_pieces = 2 →
  ∃ (max_pieces : ℕ), 
    max_pieces = 142 ∧ 
    (∀ (student_pieces : List ℕ), 
      student_pieces.length = n ∧ 
      (∀ x ∈ student_pieces, x ≥ min_pieces) ∧ 
      (student_pieces.sum : ℚ) / n = μ →
      ∀ x ∈ student_pieces, x ≤ max_pieces) :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l2447_244732


namespace NUMINAMATH_CALUDE_garage_motorcycles_l2447_244727

theorem garage_motorcycles (total_wheels : ℕ) (bicycles : ℕ) (cars : ℕ) 
  (bicycle_wheels : ℕ) (car_wheels : ℕ) (motorcycle_wheels : ℕ) :
  total_wheels = 90 ∧ 
  bicycles = 20 ∧ 
  cars = 10 ∧ 
  bicycle_wheels = 2 ∧ 
  car_wheels = 4 ∧ 
  motorcycle_wheels = 2 → 
  (total_wheels - (bicycles * bicycle_wheels + cars * car_wheels)) / motorcycle_wheels = 5 :=
by sorry

end NUMINAMATH_CALUDE_garage_motorcycles_l2447_244727


namespace NUMINAMATH_CALUDE_regular_hexagon_angles_l2447_244744

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 angles of equal measure. -/
structure RegularHexagon where
  -- We don't need to define any specific fields for this problem

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_measure (h : RegularHexagon) : ℝ := 120

/-- The sum of all exterior angles of a regular hexagon -/
def sum_exterior_angles (h : RegularHexagon) : ℝ := 360

theorem regular_hexagon_angles (h : RegularHexagon) : 
  (interior_angle_measure h = 120) ∧ (sum_exterior_angles h = 360) := by
  sorry

#check regular_hexagon_angles

end NUMINAMATH_CALUDE_regular_hexagon_angles_l2447_244744


namespace NUMINAMATH_CALUDE_cubic_inequality_l2447_244705

theorem cubic_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2447_244705


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2447_244761

theorem inequality_system_solution_set :
  let S := {x : ℝ | -3 * (x - 2) ≥ 4 - x ∧ (1 + 2 * x) / 3 > x - 1}
  S = {x : ℝ | x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2447_244761


namespace NUMINAMATH_CALUDE_xy_2yz_3zx_value_l2447_244772

theorem xy_2yz_3zx_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2/3 = 25)
  (eq2 : y^2/3 + z^2 = 9)
  (eq3 : z^2 + z*x + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xy_2yz_3zx_value_l2447_244772


namespace NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l2447_244748

def pencil_cost : ℝ := 2
def pen_cost : ℝ := pencil_cost + 9

theorem total_cost_of_pen_and_pencil :
  pencil_cost + pen_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l2447_244748


namespace NUMINAMATH_CALUDE_sin_600_degrees_l2447_244721

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l2447_244721


namespace NUMINAMATH_CALUDE_parallelogram_base_l2447_244725

theorem parallelogram_base (height area : ℝ) (h1 : height = 32) (h2 : area = 896) : 
  area / height = 28 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2447_244725


namespace NUMINAMATH_CALUDE_emily_commute_time_l2447_244714

/-- Calculates the total commute time for Emily given her travel distances and local road time --/
theorem emily_commute_time 
  (freeway_distance : ℝ) 
  (local_distance : ℝ) 
  (local_time : ℝ) 
  (h1 : freeway_distance = 100) 
  (h2 : local_distance = 25) 
  (h3 : local_time = 50) 
  (h4 : freeway_distance / local_distance = 4) : 
  local_time + freeway_distance / (2 * local_distance / local_time) = 150 := by
  sorry

#check emily_commute_time

end NUMINAMATH_CALUDE_emily_commute_time_l2447_244714


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2447_244734

theorem arithmetic_expression_equality : 6 + 18 / 3 - 4 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2447_244734


namespace NUMINAMATH_CALUDE_opposite_sides_parameter_set_is_correct_l2447_244715

/-- The set of parameter values for which points A and B lie on opposite sides of a line -/
def opposite_sides_parameter_set : Set ℝ :=
  {a | a < -2 ∨ (0 < a ∧ a < 2/3) ∨ a > 8/7}

/-- Equation of point A -/
def point_A_eq (a x y : ℝ) : Prop :=
  5 * a^2 + 12 * a * x + 4 * a * y + 8 * x^2 + 8 * x * y + 4 * y^2 = 0

/-- Equation of parabola with vertex at point B -/
def parabola_B_eq (a x y : ℝ) : Prop :=
  a * x^2 - 2 * a^2 * x - a * y + a^3 + 4 = 0

/-- Equation of the line -/
def line_eq (x y : ℝ) : Prop :=
  y - 3 * x = 4

/-- Theorem stating that the set of parameter values is correct -/
theorem opposite_sides_parameter_set_is_correct :
  ∀ a : ℝ, a ∈ opposite_sides_parameter_set ↔
    ∃ (x_A y_A x_B y_B : ℝ),
      point_A_eq a x_A y_A ∧
      parabola_B_eq a x_B y_B ∧
      ¬line_eq x_A y_A ∧
      ¬line_eq x_B y_B ∧
      (line_eq x_A y_A ↔ ¬line_eq x_B y_B) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_parameter_set_is_correct_l2447_244715


namespace NUMINAMATH_CALUDE_remainder_2007_div_81_l2447_244778

theorem remainder_2007_div_81 : 2007 % 81 = 63 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2007_div_81_l2447_244778


namespace NUMINAMATH_CALUDE_camilla_original_strawberry_l2447_244794

/-- Represents the number of strawberry jelly beans Camilla originally had. -/
def original_strawberry : ℕ := sorry

/-- Represents the number of grape jelly beans Camilla originally had. -/
def original_grape : ℕ := sorry

/-- States that Camilla originally had three times as many strawberry jelly beans as grape jelly beans. -/
axiom initial_ratio : original_strawberry = 3 * original_grape

/-- States that after eating 12 strawberry jelly beans and 8 grape jelly beans, 
    Camilla now has four times as many strawberry jelly beans as grape jelly beans. -/
axiom final_ratio : original_strawberry - 12 = 4 * (original_grape - 8)

/-- Theorem stating that Camilla originally had 60 strawberry jelly beans. -/
theorem camilla_original_strawberry : original_strawberry = 60 := by sorry

end NUMINAMATH_CALUDE_camilla_original_strawberry_l2447_244794


namespace NUMINAMATH_CALUDE_paper_folding_holes_l2447_244702

/-- The number of small squares along each side after folding a square paper n times -/
def squares_per_side (n : ℕ) : ℕ := 2^n

/-- The number of internal edges along each side after folding -/
def internal_edges (n : ℕ) : ℕ := squares_per_side n - 1

/-- The total number of holes in the middle of the paper after folding n times -/
def total_holes (n : ℕ) : ℕ := internal_edges n * squares_per_side n

/-- Theorem: When a square piece of paper is folded in half 6 times and a notch is cut along
    each edge of the resulting small square, the number of small holes in the middle
    when unfolded is 4032. -/
theorem paper_folding_holes :
  total_holes 6 = 4032 := by sorry

end NUMINAMATH_CALUDE_paper_folding_holes_l2447_244702


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2447_244750

theorem quadratic_root_sum (a b : ℝ) : 
  (1 : ℝ) ^ 2 * a + 1 * b - 3 = 0 → a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2447_244750


namespace NUMINAMATH_CALUDE_max_profit_l2447_244752

noncomputable def fixed_cost : ℝ := 14000
noncomputable def variable_cost : ℝ := 210

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then (1 / 625) * x^2
  else 256

noncomputable def g (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then -(5 / 8) * x + 750
  else 500

noncomputable def c (x : ℝ) : ℝ := fixed_cost + variable_cost * x

noncomputable def Q (x : ℝ) : ℝ := f x * g x - c x

theorem max_profit (x : ℝ) : Q x ≤ 30000 ∧ Q 400 = 30000 := by sorry

end NUMINAMATH_CALUDE_max_profit_l2447_244752


namespace NUMINAMATH_CALUDE_base_h_equation_l2447_244769

/-- Converts a base-h number to decimal --/
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- Checks if a list of digits is valid in base h --/
def valid_digits (digits : List Nat) (h : Nat) : Prop :=
  ∀ d ∈ digits, d < h

theorem base_h_equation (h : Nat) : 
  h > 8 → 
  valid_digits [8, 6, 7, 4] h → 
  valid_digits [4, 3, 2, 9] h → 
  valid_digits [1, 3, 0, 0, 3] h → 
  to_decimal [8, 6, 7, 4] h + to_decimal [4, 3, 2, 9] h = to_decimal [1, 3, 0, 0, 3] h → 
  h = 10 :=
sorry

end NUMINAMATH_CALUDE_base_h_equation_l2447_244769


namespace NUMINAMATH_CALUDE_profit_per_meter_l2447_244766

/-- Given the selling price and cost price of cloth, calculate the profit per meter -/
theorem profit_per_meter (total_meters : ℕ) (selling_price cost_per_meter : ℚ) :
  total_meters = 85 →
  selling_price = 8925 →
  cost_per_meter = 85 →
  (selling_price - total_meters * cost_per_meter) / total_meters = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_per_meter_l2447_244766


namespace NUMINAMATH_CALUDE_cosine_double_angle_equation_cosine_double_angle_special_case_l2447_244746

theorem cosine_double_angle_equation (a b c : ℝ) (x : ℝ) 
  (h : a * (Real.cos x)^2 + b * Real.cos x + c = 0) :
  (1/4) * a^2 * (Real.cos (2*x))^2 + 
  (1/2) * (a^2 - b^2 + 2*a*c) * Real.cos (2*x) + 
  (1/4) * (a^2 + 4*a*c + 4*c^2 - 2*b^2) = 0 := by
  sorry

-- Special case
theorem cosine_double_angle_special_case (x : ℝ) 
  (h : 4 * (Real.cos x)^2 + 2 * Real.cos x - 1 = 0) :
  4 * (Real.cos (2*x))^2 + 2 * Real.cos (2*x) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_double_angle_equation_cosine_double_angle_special_case_l2447_244746


namespace NUMINAMATH_CALUDE_contrapositive_theorem_l2447_244720

theorem contrapositive_theorem (x : ℝ) :
  (x = 1 ∨ x = 2 → x^2 - 3*x + 2 ≤ 0) ↔ (x^2 - 3*x + 2 > 0 → x ≠ 1 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_l2447_244720


namespace NUMINAMATH_CALUDE_license_plate_difference_l2447_244749

/-- The number of letters in the alphabet -/
def numLetters : Nat := 26

/-- The number of digits available -/
def numDigits : Nat := 10

/-- The number of license plates Sunland can issue -/
def sunlandPlates : Nat := numLetters^5 * numDigits^2

/-- The number of license plates Moonland can issue -/
def moonlandPlates : Nat := numLetters^3 * numDigits^3

/-- The difference in the number of license plates between Sunland and Moonland -/
def plateDifference : Nat := sunlandPlates - moonlandPlates

theorem license_plate_difference : plateDifference = 1170561600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2447_244749


namespace NUMINAMATH_CALUDE_all_permissible_triangles_in_final_set_l2447_244743

/-- A permissible triangle for a prime p is represented by its angles as multiples of (180/p) degrees -/
structure PermissibleTriangle (p : ℕ) :=
  (a b c : ℕ)
  (sum_eq_p : a + b + c = p)
  (p_prime : Nat.Prime p)

/-- The set of all permissible triangles for a given prime p -/
def allPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | true}

/-- A function that represents cutting a triangle into two different permissible triangles -/
def cutTriangle (p : ℕ) (t : PermissibleTriangle p) : Option (PermissibleTriangle p × PermissibleTriangle p) :=
  sorry

/-- The set of triangles resulting from repeated cutting until no more cuts are possible -/
def finalTriangleSet (p : ℕ) (initial : PermissibleTriangle p) : Set (PermissibleTriangle p) :=
  sorry

/-- The main theorem: the final set of triangles includes all possible permissible triangles -/
theorem all_permissible_triangles_in_final_set (p : ℕ) (hp : Nat.Prime p) (initial : PermissibleTriangle p) :
  finalTriangleSet p initial = allPermissibleTriangles p :=
sorry

end NUMINAMATH_CALUDE_all_permissible_triangles_in_final_set_l2447_244743


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2447_244785

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon -/
def intersectionProbability (n : RegularNonagon) : ℚ :=
  14 / 39

/-- Theorem: The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon is 14/39 -/
theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersectionProbability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2447_244785


namespace NUMINAMATH_CALUDE_function_inequality_l2447_244780

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) : 3 * f (log 2) < 2 * f (log 3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2447_244780


namespace NUMINAMATH_CALUDE_sarah_final_toads_l2447_244797

-- Define the number of toads each person has
def tim_toads : ℕ := 30
def jim_toads : ℕ := tim_toads + 20
def sarah_initial_toads : ℕ := 2 * jim_toads

-- Define the number of toads Sarah gives away
def sarah_gives_away : ℕ := sarah_initial_toads / 4

-- Define the number of toads Sarah buys
def sarah_buys : ℕ := 15

-- Theorem to prove
theorem sarah_final_toads :
  sarah_initial_toads - sarah_gives_away + sarah_buys = 90 := by
  sorry

end NUMINAMATH_CALUDE_sarah_final_toads_l2447_244797


namespace NUMINAMATH_CALUDE_number_of_cat_only_owners_cat_only_owners_count_l2447_244729

theorem number_of_cat_only_owners (total_pet_owners : ℕ) (only_dog_owners : ℕ) 
  (cat_and_dog_owners : ℕ) (cat_dog_snake_owners : ℕ) (total_snakes : ℕ) : ℕ :=
  let snake_only_owners := total_snakes - cat_dog_snake_owners
  let cat_only_owners := total_pet_owners - only_dog_owners - cat_and_dog_owners - 
                         cat_dog_snake_owners - snake_only_owners
  cat_only_owners

theorem cat_only_owners_count : 
  number_of_cat_only_owners 69 15 5 3 39 = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cat_only_owners_cat_only_owners_count_l2447_244729


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2447_244795

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x > 12 ↔ x ∈ Set.Iio (-2) ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2447_244795


namespace NUMINAMATH_CALUDE_work_completion_time_l2447_244742

theorem work_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a = 4 → b = 12 → 1 / (1 / a + 1 / b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2447_244742


namespace NUMINAMATH_CALUDE_quadratic_trinomial_condition_l2447_244771

theorem quadratic_trinomial_condition (m : ℤ) : 
  (|m| = 2 ∧ m ≠ 2) ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_condition_l2447_244771


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2447_244776

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ + ρ - 3 * ρ * Real.cos θ - 3 = 0

-- Define the Cartesian equations
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

def line_equation (x : ℝ) : Prop :=
  x = -1

-- Theorem statement
theorem polar_to_cartesian :
  ∀ ρ θ x y : ℝ, 
    polar_equation ρ θ ↔ 
    (circle_equation x y ∨ line_equation x) ∧
    x = ρ * Real.cos θ ∧
    y = ρ * Real.sin θ :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2447_244776
