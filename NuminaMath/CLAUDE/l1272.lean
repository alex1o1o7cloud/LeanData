import Mathlib

namespace NUMINAMATH_CALUDE_exam_marks_category_c_l1272_127274

theorem exam_marks_category_c (total_candidates : ℕ) 
                               (category_a_count : ℕ) 
                               (category_b_count : ℕ) 
                               (category_c_count : ℕ) 
                               (category_a_avg : ℕ) 
                               (category_b_avg : ℕ) 
                               (category_c_avg : ℕ) : 
  total_candidates = 80 →
  category_a_count = 30 →
  category_b_count = 25 →
  category_c_count = 25 →
  category_a_avg = 35 →
  category_b_avg = 42 →
  category_c_avg = 46 →
  category_c_count * category_c_avg = 1150 :=
by sorry

end NUMINAMATH_CALUDE_exam_marks_category_c_l1272_127274


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1272_127279

theorem complex_fraction_simplification :
  (3 + 3 * Complex.I) / (-4 + 5 * Complex.I) = 3 / 41 - 27 / 41 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1272_127279


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1021_l1272_127218

theorem modular_inverse_11_mod_1021 : ∃ x : ℕ, x ∈ Finset.range 1021 ∧ (11 * x) % 1021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1021_l1272_127218


namespace NUMINAMATH_CALUDE_problem_solution_l1272_127270

theorem problem_solution :
  ∀ (a b m n : ℝ),
  (m = (a + 4) ^ (1 / (b - 1))) →
  (n = (3 * b - 1) ^ (1 / (a - 2))) →
  ((b - 1) = 2) →
  ((a - 2) = 3) →
  ((m - 2 * n) ^ (1 / 3) = -1) ∧
  (∀ (m' n' : ℝ),
    (m' = Real.sqrt (1 - a) + Real.sqrt (a - 1) + 1) →
    (n' = 25) →
    (Real.sqrt (3 * n' + 6 * m') = 9 ∨ Real.sqrt (3 * n' + 6 * m') = -9)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1272_127270


namespace NUMINAMATH_CALUDE_sum_of_fractions_in_base_10_l1272_127273

/-- Convert a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Express a fraction in base 10 given numerator and denominator in different bases -/
def fractionToBase10 (num : ℕ) (num_base : ℕ) (den : ℕ) (den_base : ℕ) : ℚ := sorry

/-- Main theorem: The integer part of the sum of the given fractions in base 10 is 29 -/
theorem sum_of_fractions_in_base_10 : 
  ⌊(fractionToBase10 254 8 13 4 + fractionToBase10 132 5 22 3)⌋ = 29 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_in_base_10_l1272_127273


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l1272_127285

/-- In a triangle ABC, if sin(A)/a = cos(B)/b, then B = 45° --/
theorem triangle_angle_equality (A B : ℝ) (a b : ℝ) :
  (0 < a) → (0 < b) → (0 < A) → (A < π) → (0 < B) → (B < π) →
  (Real.sin A / a = Real.cos B / b) →
  B = π/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l1272_127285


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1272_127249

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  contains β n → 
  planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1272_127249


namespace NUMINAMATH_CALUDE_expand_product_l1272_127203

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1272_127203


namespace NUMINAMATH_CALUDE_track_event_races_l1272_127244

/-- The number of races needed to determine a champion in a track event -/
def races_needed (total_athletes : ℕ) (lanes_per_race : ℕ) : ℕ :=
  let first_round := total_athletes / lanes_per_race
  let second_round := first_round / lanes_per_race
  let final_round := 1
  first_round + second_round + final_round

/-- Theorem stating that 43 races are needed for 216 athletes with 6 lanes per race -/
theorem track_event_races : races_needed 216 6 = 43 := by
  sorry

#eval races_needed 216 6

end NUMINAMATH_CALUDE_track_event_races_l1272_127244


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1272_127297

theorem set_intersection_problem (M N : Set ℕ) : 
  M = {1, 2, 3} → N = {2, 3, 4} → M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1272_127297


namespace NUMINAMATH_CALUDE_coefficient_x4_in_product_l1272_127238

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - 4*x^2 + x - 1
def q (x : ℝ) : ℝ := 3*x^4 - 4*x^3 + 5*x^2 - 2*x + 6

theorem coefficient_x4_in_product :
  ∃ (a b c d e f g h i j : ℝ),
    p x * q x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + (-38)*x^4 + f*x^3 + g*x^2 + h*x + i :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_product_l1272_127238


namespace NUMINAMATH_CALUDE_rectangular_plot_longer_side_l1272_127241

theorem rectangular_plot_longer_side 
  (width : ℝ) 
  (pole_distance : ℝ) 
  (num_poles : ℕ) 
  (h1 : width = 30)
  (h2 : pole_distance = 5)
  (h3 : num_poles = 32) :
  let perimeter := pole_distance * (num_poles - 1 : ℝ)
  let length := (perimeter / 2) - width
  length = 47.5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_longer_side_l1272_127241


namespace NUMINAMATH_CALUDE_inequality_system_no_solution_l1272_127269

theorem inequality_system_no_solution : 
  ∀ x : ℝ, ¬(2 * x^2 - 5 * x + 3 < 0 ∧ (x - 1) / (2 - x) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_no_solution_l1272_127269


namespace NUMINAMATH_CALUDE_ratio_change_l1272_127213

theorem ratio_change (x y : ℕ) (n : ℕ) (h1 : y = 24) (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_l1272_127213


namespace NUMINAMATH_CALUDE_max_value_abc_l1272_127266

theorem max_value_abc (a b c : ℝ) (h : 2 * a + 3 * b + c = 6) :
  ∃ (max : ℝ), max = 9/2 ∧ ∀ (x y z : ℝ), 2 * x + 3 * y + z = 6 → x * y + x * z + y * z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l1272_127266


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1272_127254

theorem min_value_of_expression (a : ℝ) (h : a > 0) :
  a + 4 / a ≥ 4 ∧ (a + 4 / a = 4 ↔ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1272_127254


namespace NUMINAMATH_CALUDE_percentage_fraction_difference_l1272_127262

theorem percentage_fraction_difference : 
  (85 / 100 * 40) - (4 / 5 * 25) = 14 := by sorry

end NUMINAMATH_CALUDE_percentage_fraction_difference_l1272_127262


namespace NUMINAMATH_CALUDE_sum_of_max_min_a_l1272_127234

theorem sum_of_max_min_a (a : ℝ) : 
  (∀ x y : ℝ, x^2 - a*x - 20*a^2 < 0 ∧ y^2 - a*y - 20*a^2 < 0 → |x - y| ≤ 9) →
  ∃ a_min a_max : ℝ, 
    (∀ a' : ℝ, (∃ x : ℝ, x^2 - a'*x - 20*a'^2 < 0) → a_min ≤ a' ∧ a' ≤ a_max) ∧
    a_min + a_max = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_a_l1272_127234


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1272_127282

/-- Given a rectangle A composed of 3 equal squares with a perimeter of 112 cm,
    prove that a rectangle B composed of 4 of the same squares will have a perimeter of 140 cm. -/
theorem rectangle_perimeter (side_length : ℝ) : 
  (3 * side_length * 2 + 2 * side_length) = 112 → 
  (4 * side_length * 2 + 2 * side_length) = 140 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1272_127282


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l1272_127255

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ := m) : (m^2 + 2^m) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l1272_127255


namespace NUMINAMATH_CALUDE_range_of_a_l1272_127232

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + Real.exp (-x) - a

def range_subset (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x a ≥ 0

theorem range_of_a (a : ℝ) :
  (range_subset f a) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1272_127232


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_sum_l1272_127214

theorem sqrt_plus_square_zero_implies_sum (x y : ℝ) :
  Real.sqrt (x - 1) + (y + 2)^2 = 0 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_sum_l1272_127214


namespace NUMINAMATH_CALUDE_hall_dimension_l1272_127263

/-- Represents a square rug with a given side length. -/
structure Rug where
  side : ℝ
  square : side > 0

/-- Represents a square hall containing two rugs. -/
structure Hall where
  small_rug : Rug
  large_rug : Rug
  opposite_overlap : ℝ
  adjacent_overlap : ℝ
  hall_side : ℝ

/-- The theorem stating the conditions and the conclusion about the hall's dimensions. -/
theorem hall_dimension (h : Hall) : 
  h.large_rug.side = 2 * h.small_rug.side ∧ 
  h.opposite_overlap = 4 ∧ 
  h.adjacent_overlap = 14 → 
  h.hall_side = 19 := by
  sorry


end NUMINAMATH_CALUDE_hall_dimension_l1272_127263


namespace NUMINAMATH_CALUDE_smallest_n_properties_count_non_14_divisors_l1272_127275

def is_perfect_power (x : ℕ) (k : ℕ) : Prop :=
  ∃ y : ℕ, x = y^k

def smallest_n : ℕ :=
  sorry

theorem smallest_n_properties (n : ℕ) (hn : n = smallest_n) :
  is_perfect_power (n / 2) 2 ∧
  is_perfect_power (n / 3) 3 ∧
  is_perfect_power (n / 5) 5 ∧
  is_perfect_power (n / 7) 7 :=
  sorry

theorem count_non_14_divisors (n : ℕ) (hn : n = smallest_n) :
  (Finset.filter (fun d => ¬(14 ∣ d)) (Nat.divisors n)).card = 240 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_properties_count_non_14_divisors_l1272_127275


namespace NUMINAMATH_CALUDE_carla_fish_count_l1272_127267

/-- Given that Carla, Kyle, and Tasha caught a total of 36 fish, Kyle caught 14 fish,
    and Kyle and Tasha caught the same number of fish, prove that Carla caught 8 fish. -/
theorem carla_fish_count (total : ℕ) (kyle_fish : ℕ) (h1 : total = 36) (h2 : kyle_fish = 14)
    (h3 : ∃ (tasha_fish : ℕ), tasha_fish = kyle_fish ∧ total = kyle_fish + tasha_fish + (total - kyle_fish - tasha_fish)) :
  total - kyle_fish - kyle_fish = 8 := by
  sorry

end NUMINAMATH_CALUDE_carla_fish_count_l1272_127267


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_properties_l1272_127209

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral where
  R : ℝ  -- circumradius
  a : ℝ  -- side length
  b : ℝ  -- side length
  c : ℝ  -- side length
  d : ℝ  -- side length
  S : ℝ  -- area
  positive_R : R > 0
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  positive_S : S > 0

-- Define what it means for a cyclic quadrilateral to be a square
def is_square (q : CyclicQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

theorem cyclic_quadrilateral_properties (q : CyclicQuadrilateral) :
  (16 * q.R^2 * q.S^2 = (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)) ∧
  (q.R * q.S * Real.sqrt 2 ≥ (q.a * q.b * q.c * q.d)^(3/4)) ∧
  (q.R * q.S * Real.sqrt 2 = (q.a * q.b * q.c * q.d)^(3/4) ↔ is_square q) :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_properties_l1272_127209


namespace NUMINAMATH_CALUDE_a_greater_equal_two_l1272_127276

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem a_greater_equal_two (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a (-1) ≤ f a x) ∧
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ f a 1) →
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_a_greater_equal_two_l1272_127276


namespace NUMINAMATH_CALUDE_factor_expression_l1272_127207

theorem factor_expression (x : ℝ) : 63 * x + 42 = 21 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1272_127207


namespace NUMINAMATH_CALUDE_pen_notebook_ratio_l1272_127231

theorem pen_notebook_ratio (num_notebooks : ℕ) (num_pens : ℕ) : 
  (num_notebooks = 40) → 
  (num_pens : ℚ) / (num_notebooks : ℚ) = 5 / 4 → 
  num_pens = 50 := by
  sorry

end NUMINAMATH_CALUDE_pen_notebook_ratio_l1272_127231


namespace NUMINAMATH_CALUDE_nested_sqrt_is_five_nested_sqrt_unique_positive_solution_nested_sqrt_value_l1272_127290

-- Define the nested square root expression
def nestedSqrt (x : ℝ) : Prop := x = Real.sqrt (20 + x)

-- Theorem stating that 5 satisfies the nested square root equation
theorem nested_sqrt_is_five : nestedSqrt 5 := by sorry

-- Theorem stating that 5 is the unique positive solution
theorem nested_sqrt_unique_positive_solution :
  ∀ x : ℝ, x > 0 → nestedSqrt x → x = 5 := by sorry

-- Main theorem: The value of the nested square root is 5
theorem nested_sqrt_value : 
  ∃! x : ℝ, x > 0 ∧ nestedSqrt x ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_is_five_nested_sqrt_unique_positive_solution_nested_sqrt_value_l1272_127290


namespace NUMINAMATH_CALUDE_selling_price_example_l1272_127292

/-- Calculates the selling price of an article given the gain and gain percentage. -/
def selling_price (gain : ℚ) (gain_percentage : ℚ) : ℚ :=
  let cost_price := gain / (gain_percentage / 100)
  cost_price + gain

/-- Theorem stating that given a gain of $75 and a gain percentage of 50%, 
    the selling price of an article is $225. -/
theorem selling_price_example : selling_price 75 50 = 225 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_example_l1272_127292


namespace NUMINAMATH_CALUDE_average_difference_l1272_127242

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 80) 
  (hbc : (b + c) / 2 = 180) : 
  a - c = -200 := by sorry

end NUMINAMATH_CALUDE_average_difference_l1272_127242


namespace NUMINAMATH_CALUDE_abcd_efgh_ratio_l1272_127260

theorem abcd_efgh_ratio 
  (a b c d e f g h : ℝ) 
  (hab : a / b = 1 / 3)
  (hbc : b / c = 2)
  (hcd : c / d = 1 / 2)
  (hde : d / e = 3)
  (hef : e / f = 1 / 2)
  (hfg : f / g = 5 / 3)
  (hgh : g / h = 4 / 9)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  a * b * c * d / (e * f * g * h) = 1 / 97 := by
  sorry

end NUMINAMATH_CALUDE_abcd_efgh_ratio_l1272_127260


namespace NUMINAMATH_CALUDE_value_of_x_l1272_127224

theorem value_of_x (x : ℚ) : (1/4 : ℚ) - (1/6 : ℚ) = 4/x → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1272_127224


namespace NUMINAMATH_CALUDE_town_population_problem_l1272_127208

theorem town_population_problem (original_population : ℕ) : 
  (original_population + 1200 : ℕ) * 89 / 100 = original_population - 32 →
  original_population = 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l1272_127208


namespace NUMINAMATH_CALUDE_spatial_relationships_l1272_127229

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallelLines : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem spatial_relationships 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : contains β m) :
  (parallel α β → perpendicularLines l m) ∧ 
  (parallelLines l m → perpendicularPlanes α β) := by
  sorry

end NUMINAMATH_CALUDE_spatial_relationships_l1272_127229


namespace NUMINAMATH_CALUDE_james_weekday_coffees_l1272_127226

/-- Represents the number of weekdays in a week -/
def weekdays : Nat := 5

/-- Represents the number of weekend days in a week -/
def weekend_days : Nat := 2

/-- Cost of a donut in cents -/
def donut_cost : Nat := 60

/-- Cost of a coffee in cents -/
def coffee_cost : Nat := 90

/-- Calculates the total cost for the week in cents -/
def total_cost (weekday_coffees : Nat) : Nat :=
  let weekday_donuts := weekdays - weekday_coffees
  let weekday_cost := weekday_coffees * coffee_cost + weekday_donuts * donut_cost
  let weekend_cost := weekend_days * (coffee_cost + donut_cost)
  weekday_cost + weekend_cost

theorem james_weekday_coffees :
  ∃ (weekday_coffees : Nat),
    weekday_coffees ≤ weekdays ∧
    (∃ (k : Nat), total_cost weekday_coffees = k * 100) ∧
    weekday_coffees = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_weekday_coffees_l1272_127226


namespace NUMINAMATH_CALUDE_landscape_breadth_l1272_127258

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ

/-- The breadth is 6 times the length -/
def breadth_length_relation (l : Landscape) : Prop :=
  l.breadth = 6 * l.length

/-- The playground occupies 1/7th of the total landscape area -/
def playground_proportion (l : Landscape) : Prop :=
  l.playground_area = (1 / 7) * l.length * l.breadth

/-- The playground area is 4200 square meters -/
def playground_area_value (l : Landscape) : Prop :=
  l.playground_area = 4200

/-- Theorem: The breadth of the landscape is 420 meters -/
theorem landscape_breadth (l : Landscape) 
  (h1 : breadth_length_relation l)
  (h2 : playground_proportion l)
  (h3 : playground_area_value l) : 
  l.breadth = 420 := by sorry

end NUMINAMATH_CALUDE_landscape_breadth_l1272_127258


namespace NUMINAMATH_CALUDE_pet_store_count_l1272_127233

/-- Given the ratios of cats to dogs and dogs to birds, and the number of cats,
    prove the number of dogs and birds -/
theorem pet_store_count (cats : ℕ) (dogs : ℕ) (birds : ℕ) : 
  cats = 20 →                   -- There are 20 cats
  5 * cats = 4 * dogs →         -- Ratio of cats to dogs is 4:5
  7 * dogs = 3 * birds →        -- Ratio of dogs to birds is 3:7
  dogs = 25 ∧ birds = 56 :=     -- Prove dogs = 25 and birds = 56
by sorry

end NUMINAMATH_CALUDE_pet_store_count_l1272_127233


namespace NUMINAMATH_CALUDE_sum_of_qp_at_points_l1272_127280

def p (x : ℝ) : ℝ := |x^2 - 4|

def q (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_at_points :
  (evaluation_points.map (λ x => q (p x))).sum = -20 := by sorry

end NUMINAMATH_CALUDE_sum_of_qp_at_points_l1272_127280


namespace NUMINAMATH_CALUDE_penny_collection_difference_l1272_127294

theorem penny_collection_difference (cassandra_pennies james_pennies : ℕ) : 
  cassandra_pennies = 5000 →
  james_pennies < cassandra_pennies →
  cassandra_pennies + james_pennies = 9724 →
  cassandra_pennies - james_pennies = 276 := by
sorry

end NUMINAMATH_CALUDE_penny_collection_difference_l1272_127294


namespace NUMINAMATH_CALUDE_range_of_m_l1272_127221

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3)
  (hineq : ∀ m : ℝ, (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 11) :
  ∀ m : ℝ, (1 < m ∧ m < 2) ↔ (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 11 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1272_127221


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l1272_127289

theorem reciprocal_sum_of_quadratic_roots :
  ∀ (α β : ℝ),
  (∃ (a b : ℝ), 7 * a^2 + 2 * a + 6 = 0 ∧ 
                 7 * b^2 + 2 * b + 6 = 0 ∧ 
                 α = 1 / a ∧ 
                 β = 1 / b) →
  α + β = -1/3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l1272_127289


namespace NUMINAMATH_CALUDE_minimize_sum_with_constraint_l1272_127215

theorem minimize_sum_with_constraint :
  ∀ a b : ℕ+,
  (4 * a.val + b.val = 30) →
  (∀ x y : ℕ+, (4 * x.val + y.val = 30) → (a.val + b.val ≤ x.val + y.val)) →
  (a.val = 7 ∧ b.val = 2) :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_with_constraint_l1272_127215


namespace NUMINAMATH_CALUDE_bus_driver_worked_69_hours_l1272_127236

/-- Represents the payment structure and total compensation for a bus driver --/
structure BusDriverPayment where
  regular_rate : ℝ
  overtime_rate : ℝ
  double_overtime_rate : ℝ
  total_compensation : ℝ

/-- Calculates the total hours worked by a bus driver given their payment structure and total compensation --/
def calculate_total_hours (payment : BusDriverPayment) : ℕ :=
  sorry

/-- Theorem stating that given the specific payment structure and total compensation, the bus driver worked 69 hours --/
theorem bus_driver_worked_69_hours : 
  let payment := BusDriverPayment.mk 14 18.90 24.50 1230
  calculate_total_hours payment = 69 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_worked_69_hours_l1272_127236


namespace NUMINAMATH_CALUDE_units_digit_of_product_units_digit_27_times_46_l1272_127202

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of a product depends only on the units digits of its factors -/
theorem units_digit_of_product (a b : ℕ) :
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by
  sorry

/-- The main theorem: the units digit of 27 * 46 is equal to the units digit of 7 * 6 -/
theorem units_digit_27_times_46 :
  unitsDigit (27 * 46) = unitsDigit (7 * 6) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_units_digit_27_times_46_l1272_127202


namespace NUMINAMATH_CALUDE_exists_function_double_composition_l1272_127200

theorem exists_function_double_composition :
  ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_exists_function_double_composition_l1272_127200


namespace NUMINAMATH_CALUDE_real_estate_commission_l1272_127246

/-- Calculate the commission for a real estate agent given the selling price and commission rate -/
def calculate_commission (selling_price : ℝ) (commission_rate : ℝ) : ℝ :=
  selling_price * commission_rate

/-- Theorem stating that the commission for a house sold at $148,000 with a 6% commission rate is $8,880 -/
theorem real_estate_commission :
  let selling_price : ℝ := 148000
  let commission_rate : ℝ := 0.06
  calculate_commission selling_price commission_rate = 8880 := by
  sorry

end NUMINAMATH_CALUDE_real_estate_commission_l1272_127246


namespace NUMINAMATH_CALUDE_point_coordinates_l1272_127223

def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

theorem point_coordinates :
  ∀ (p : ℝ × ℝ),
    second_quadrant p →
    distance_to_x_axis p = 3 →
    distance_to_y_axis p = 1 →
    p = (-1, 3) :=
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1272_127223


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l1272_127299

/-- Coconut grove problem -/
theorem coconut_grove_problem (x : ℝ) : 
  (60 * (x + 4) + 120 * x + 180 * (x - 4)) / (3 * x) = 100 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l1272_127299


namespace NUMINAMATH_CALUDE_sin_675_degrees_l1272_127245

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l1272_127245


namespace NUMINAMATH_CALUDE_number_division_problem_l1272_127268

theorem number_division_problem :
  ∃ x : ℚ, (x / 5 = 80 + x / 6) → x = 2400 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l1272_127268


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l1272_127296

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Theorem statement
theorem circle_tangent_to_parabola_directrix :
  ∀ x y : ℝ,
  parabola x y →
  (∃ t : ℝ, directrix t ∧ 
    ((x - focus.1)^2 + (y - focus.2)^2 = (t - focus.1)^2)) →
  circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l1272_127296


namespace NUMINAMATH_CALUDE_inserted_numbers_in_arithmetic_sequence_l1272_127205

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : Fin n → ℝ :=
  λ i => a₁ + d * i.val

theorem inserted_numbers_in_arithmetic_sequence :
  let n : ℕ := 8
  let a₁ : ℝ := 8
  let aₙ : ℝ := 36
  let d : ℝ := (aₙ - a₁) / (n - 1)
  let seq := arithmetic_sequence a₁ d n
  (seq 1 = 12) ∧
  (seq 2 = 16) ∧
  (seq 3 = 20) ∧
  (seq 4 = 24) ∧
  (seq 5 = 28) ∧
  (seq 6 = 32) :=
by sorry

end NUMINAMATH_CALUDE_inserted_numbers_in_arithmetic_sequence_l1272_127205


namespace NUMINAMATH_CALUDE_simplified_fraction_value_l1272_127206

theorem simplified_fraction_value (k : ℝ) : 
  ∃ (a b : ℤ), (10 * k + 15) / 5 = a * k + b → a / b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_value_l1272_127206


namespace NUMINAMATH_CALUDE_sanda_minutes_per_day_l1272_127251

/-- The number of minutes Javier exercised per day -/
def javier_minutes_per_day : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of minutes Javier and Sanda exercised -/
def total_minutes : ℕ := 620

/-- The number of days Sanda exercised -/
def sanda_exercise_days : ℕ := 3

/-- Theorem stating that Sanda exercised 90 minutes each day -/
theorem sanda_minutes_per_day :
  (total_minutes - javier_minutes_per_day * days_in_week) / sanda_exercise_days = 90 := by
  sorry

end NUMINAMATH_CALUDE_sanda_minutes_per_day_l1272_127251


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l1272_127298

-- Define the two linear functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x - b

-- Define the symmetry condition about y = x
def symmetric (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_values :
  ∃ a b : ℝ, symmetric (f a) (g b) → a = 1/3 ∧ b = 6 :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l1272_127298


namespace NUMINAMATH_CALUDE_product_of_specific_sum_and_cube_difference_l1272_127253

theorem product_of_specific_sum_and_cube_difference (x y : ℝ) 
  (sum_eq : x + y = 4) 
  (cube_diff_eq : x^3 - y^3 = 64) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_sum_and_cube_difference_l1272_127253


namespace NUMINAMATH_CALUDE_jakes_weight_l1272_127257

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 33 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 153) : 
  jake_weight = 113 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l1272_127257


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l1272_127227

theorem triangle_angle_inequalities (α β γ : ℝ) 
  (h_triangle : α + β + γ = Real.pi) : 
  ((1 - Real.cos α) * (1 - Real.cos β) * (1 - Real.cos γ) ≥ Real.cos α * Real.cos β * Real.cos γ) ∧
  (12 * Real.cos α * Real.cos β * Real.cos γ ≤ 
   2 * Real.cos α * Real.cos β + 2 * Real.cos α * Real.cos γ + 2 * Real.cos β * Real.cos γ) ∧
  (2 * Real.cos α * Real.cos β + 2 * Real.cos α * Real.cos γ + 2 * Real.cos β * Real.cos γ ≤ 
   Real.cos α + Real.cos β + Real.cos γ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l1272_127227


namespace NUMINAMATH_CALUDE_angle_C_is_pi_over_three_sum_a_b_range_l1272_127239

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.c) * (Real.sin t.A - Real.sin t.C) = Real.sin t.B * (t.a - t.b)

-- Theorem for part I
theorem angle_C_is_pi_over_three (t : Triangle) 
  (h : satisfiesCondition t) : t.C = π / 3 := by sorry

-- Theorem for part II
theorem sum_a_b_range (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.c = 2) : 
  2 < t.a + t.b ∧ t.a + t.b ≤ 4 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_over_three_sum_a_b_range_l1272_127239


namespace NUMINAMATH_CALUDE_x_range_l1272_127237

-- Define the inequality condition
def inequality_condition (x m : ℝ) : Prop :=
  2 * x - 1 > m * (x^2 - 1)

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  |m| ≤ 2

-- Theorem statement
theorem x_range :
  (∀ x m : ℝ, m_range m → inequality_condition x m) →
  ∃ a b : ℝ, a = (Real.sqrt 7 - 1) / 2 ∧ b = (Real.sqrt 3 + 1) / 2 ∧
    ∀ x : ℝ, (∀ m : ℝ, m_range m → inequality_condition x m) → a < x ∧ x < b :=
sorry

end NUMINAMATH_CALUDE_x_range_l1272_127237


namespace NUMINAMATH_CALUDE_afforestation_growth_rate_l1272_127291

theorem afforestation_growth_rate :
  ∃ x : ℝ, 2000 * (1 + x)^2 = 2880 ∧ x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_afforestation_growth_rate_l1272_127291


namespace NUMINAMATH_CALUDE_news_watching_probability_l1272_127211

/-- Represents a survey conducted in a town -/
structure TownSurvey where
  total_population : ℕ
  sample_size : ℕ
  news_watchers : ℕ

/-- Calculates the probability of a random person watching the news based on survey results -/
def probability_watch_news (survey : TownSurvey) : ℚ :=
  survey.news_watchers / survey.sample_size

/-- Theorem stating the probability of watching news for the given survey -/
theorem news_watching_probability (survey : TownSurvey) 
  (h1 : survey.total_population = 100000)
  (h2 : survey.sample_size = 2000)
  (h3 : survey.news_watchers = 250) :
  probability_watch_news survey = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_news_watching_probability_l1272_127211


namespace NUMINAMATH_CALUDE_sum_inequality_l1272_127228

theorem sum_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  n * (n + 1) / 2 ≠ m * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1272_127228


namespace NUMINAMATH_CALUDE_second_discount_percentage_second_discount_percentage_proof_l1272_127287

theorem second_discount_percentage (original_price first_discount final_price : ℝ) 
  (h1 : original_price = 528)
  (h2 : first_discount = 0.2)
  (h3 : final_price = 380.16) : ℝ :=
let price_after_first_discount := original_price * (1 - first_discount)
let second_discount := (price_after_first_discount - final_price) / price_after_first_discount
0.1

theorem second_discount_percentage_proof 
  (original_price first_discount final_price : ℝ) 
  (h1 : original_price = 528)
  (h2 : first_discount = 0.2)
  (h3 : final_price = 380.16) : 
  second_discount_percentage original_price first_discount final_price h1 h2 h3 = 0.1 := by
sorry

end NUMINAMATH_CALUDE_second_discount_percentage_second_discount_percentage_proof_l1272_127287


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1272_127271

theorem rectangle_diagonal (l w : ℝ) (h1 : l + w = 15) (h2 : l = 2 * w) :
  Real.sqrt (l^2 + w^2) = Real.sqrt 125 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1272_127271


namespace NUMINAMATH_CALUDE_sum_odd_numbers_less_than_20_l1272_127272

theorem sum_odd_numbers_less_than_20 : 
  (Finset.range 10).sum (fun n => 2 * n + 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_less_than_20_l1272_127272


namespace NUMINAMATH_CALUDE_urn_theorem_l1272_127204

/-- Represents the state of the urn -/
structure UrnState where
  black : ℕ
  white : ℕ

/-- Represents the four possible operations -/
inductive Operation
  | RemoveBlack
  | RemoveBlackWhite
  | RemoveBlackAddWhite
  | RemoveWhiteAddBlack

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.RemoveBlack => ⟨state.black - 1, state.white⟩
  | Operation.RemoveBlackWhite => ⟨state.black, state.white - 1⟩
  | Operation.RemoveBlackAddWhite => ⟨state.black - 1, state.white⟩
  | Operation.RemoveWhiteAddBlack => ⟨state.black + 1, state.white - 1⟩

/-- Checks if the given state is reachable from the initial state -/
def isReachable (initialState : UrnState) (targetState : UrnState) : Prop :=
  ∃ (n : ℕ) (ops : Fin n → Operation),
    (List.foldl applyOperation initialState (List.ofFn ops)) = targetState

/-- The theorem to be proven -/
theorem urn_theorem :
  let initialState : UrnState := ⟨150, 150⟩
  let targetState : UrnState := ⟨50, 50⟩
  isReachable initialState targetState :=
sorry

end NUMINAMATH_CALUDE_urn_theorem_l1272_127204


namespace NUMINAMATH_CALUDE_unique_solution_system_l1272_127264

theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = z + w + z * w * z ∧
    y = w + x + w * z * x ∧
    z = x + y + x * y * x ∧
    w = y + z + z * y * z :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1272_127264


namespace NUMINAMATH_CALUDE_log_inequality_l1272_127220

theorem log_inequality (h1 : 5^5 < 8^4) (h2 : 13^4 < 8^5) :
  Real.log 3 / Real.log 5 < Real.log 5 / Real.log 8 ∧
  Real.log 5 / Real.log 8 < Real.log 8 / Real.log 13 := by
sorry

end NUMINAMATH_CALUDE_log_inequality_l1272_127220


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l1272_127284

/-- Proves that given two cyclists on a 45-mile course, one traveling at 16 mph,
    meeting after 1.5 hours, the speed of the other cyclist must be 14 mph. -/
theorem cyclist_speed_problem (course_length : ℝ) (second_cyclist_speed : ℝ) (meeting_time : ℝ)
  (h1 : course_length = 45)
  (h2 : second_cyclist_speed = 16)
  (h3 : meeting_time = 1.5) :
  ∃ (first_cyclist_speed : ℝ),
    first_cyclist_speed * meeting_time + second_cyclist_speed * meeting_time = course_length ∧
    first_cyclist_speed = 14 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l1272_127284


namespace NUMINAMATH_CALUDE_cost_per_bag_of_chips_l1272_127286

/-- Given three friends buying chips, prove the cost per bag --/
theorem cost_per_bag_of_chips (num_friends : ℕ) (num_bags : ℕ) (payment_per_friend : ℚ) : 
  num_friends = 3 → num_bags = 5 → payment_per_friend = 5 →
  (num_friends * payment_per_friend) / num_bags = 3 := by
sorry

end NUMINAMATH_CALUDE_cost_per_bag_of_chips_l1272_127286


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1272_127281

/-- Theorem: When the length of a rectangle is halved and its breadth is tripled, 
    the percentage change in area is a 50% increase. -/
theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let new_area := (L / 2) * (3 * B)
  let percent_change := (new_area - original_area) / original_area * 100
  percent_change = 50 := by
sorry


end NUMINAMATH_CALUDE_rectangle_area_change_l1272_127281


namespace NUMINAMATH_CALUDE_wrench_hammer_weight_ratio_l1272_127250

/-- Given that hammers and wrenches have uniform weights, prove that if the total weight of 2 hammers
    and 2 wrenches is 1/3 of the weight of 8 hammers and 5 wrenches, then the weight of one wrench
    is 2 times the weight of one hammer. -/
theorem wrench_hammer_weight_ratio 
  (h : ℝ) -- weight of one hammer
  (w : ℝ) -- weight of one wrench
  (h_pos : h > 0) -- hammer weight is positive
  (w_pos : w > 0) -- wrench weight is positive
  (weight_ratio : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) -- given condition
  : w = 2 * h := by
  sorry

end NUMINAMATH_CALUDE_wrench_hammer_weight_ratio_l1272_127250


namespace NUMINAMATH_CALUDE_tv_show_episodes_l1272_127261

/-- Proves that a TV show with given conditions has 20 episodes per season in its first half -/
theorem tv_show_episodes (total_seasons : ℕ) (second_half_episodes : ℕ) (total_episodes : ℕ) :
  total_seasons = 10 →
  second_half_episodes = 25 →
  total_episodes = 225 →
  (total_seasons / 2 : ℕ) * second_half_episodes + (total_seasons / 2 : ℕ) * (total_episodes - (total_seasons / 2 : ℕ) * second_half_episodes) / (total_seasons / 2 : ℕ) = total_episodes →
  (total_episodes - (total_seasons / 2 : ℕ) * second_half_episodes) / (total_seasons / 2 : ℕ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_tv_show_episodes_l1272_127261


namespace NUMINAMATH_CALUDE_printer_price_ratio_printer_price_ratio_proof_l1272_127210

/-- The ratio of the printer price to the total price of enhanced computer and printer -/
theorem printer_price_ratio : ℚ :=
let basic_computer_price : ℕ := 2000
let basic_total_price : ℕ := 2500
let price_difference : ℕ := 500
let printer_price : ℕ := basic_total_price - basic_computer_price
let enhanced_computer_price : ℕ := basic_computer_price + price_difference
let enhanced_total_price : ℕ := enhanced_computer_price + printer_price
1 / 6

theorem printer_price_ratio_proof :
  let basic_computer_price : ℕ := 2000
  let basic_total_price : ℕ := 2500
  let price_difference : ℕ := 500
  let printer_price : ℕ := basic_total_price - basic_computer_price
  let enhanced_computer_price : ℕ := basic_computer_price + price_difference
  let enhanced_total_price : ℕ := enhanced_computer_price + printer_price
  (printer_price : ℚ) / enhanced_total_price = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_printer_price_ratio_printer_price_ratio_proof_l1272_127210


namespace NUMINAMATH_CALUDE_linear_system_solution_l1272_127212

theorem linear_system_solution (x y m : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = m - 3) → 
  (x - y = 2) → 
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1272_127212


namespace NUMINAMATH_CALUDE_divisor_property_l1272_127222

theorem divisor_property (k : ℕ) : 18^k ∣ 624938 → 6^k - k^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l1272_127222


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l1272_127293

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes with exactly 2 painted faces -/
  two_face_cubes : ℕ
  /-- The total number of smaller cubes created -/
  total_cubes : ℕ

/-- Theorem stating that if a cube is cut such that there are 12 smaller cubes
    with 2 painted faces, then the total number of smaller cubes is 8 -/
theorem cut_cube_theorem (c : CutCube) :
  c.two_face_cubes = 12 → c.total_cubes = 8 := by
  sorry

end NUMINAMATH_CALUDE_cut_cube_theorem_l1272_127293


namespace NUMINAMATH_CALUDE_wendy_miles_walked_l1272_127243

def pedometer_max : ℕ := 49999
def flips : ℕ := 60
def final_reading : ℕ := 25000
def steps_per_mile : ℕ := 1500

def total_steps : ℕ := (pedometer_max + 1) * flips + final_reading

def miles_walked : ℚ := total_steps / steps_per_mile

theorem wendy_miles_walked :
  ⌊(miles_walked + 50) / 100⌋ * 100 = 2000 :=
sorry

end NUMINAMATH_CALUDE_wendy_miles_walked_l1272_127243


namespace NUMINAMATH_CALUDE_police_departments_female_officers_l1272_127230

/-- Represents a police department with female officers -/
structure Department where
  totalOfficers : ℕ
  femaleOfficersOnDuty : ℕ
  femaleOfficerPercentage : ℚ

/-- Calculates the total number of female officers in a department -/
def totalFemaleOfficers (d : Department) : ℕ :=
  (d.femaleOfficersOnDuty : ℚ) / d.femaleOfficerPercentage |>.ceil.toNat

theorem police_departments_female_officers 
  (deptA : Department)
  (deptB : Department)
  (deptC : Department)
  (hA : deptA = { totalOfficers := 180, femaleOfficersOnDuty := 90, femaleOfficerPercentage := 18/100 })
  (hB : deptB = { totalOfficers := 200, femaleOfficersOnDuty := 60, femaleOfficerPercentage := 25/100 })
  (hC : deptC = { totalOfficers := 150, femaleOfficersOnDuty := 40, femaleOfficerPercentage := 30/100 }) :
  totalFemaleOfficers deptA = 500 ∧
  totalFemaleOfficers deptB = 240 ∧
  totalFemaleOfficers deptC = 133 ∧
  totalFemaleOfficers deptA + totalFemaleOfficers deptB + totalFemaleOfficers deptC = 873 := by
  sorry

end NUMINAMATH_CALUDE_police_departments_female_officers_l1272_127230


namespace NUMINAMATH_CALUDE_arrangements_equal_72_l1272_127225

-- Define the number of men and women
def num_men : ℕ := 4
def num_women : ℕ := 3

-- Define the number of groups and their sizes
def num_groups : ℕ := 3
def group_sizes : List ℕ := [3, 3, 2]

-- Define the minimum number of men and women in each group
def min_men_per_group : ℕ := 1
def min_women_per_group : ℕ := 1

-- Define a function to calculate the number of arrangements
def num_arrangements (m : ℕ) (w : ℕ) (gs : List ℕ) (min_m : ℕ) (min_w : ℕ) : ℕ := sorry

-- Theorem statement
theorem arrangements_equal_72 :
  num_arrangements num_men num_women group_sizes min_men_per_group min_women_per_group = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_equal_72_l1272_127225


namespace NUMINAMATH_CALUDE_mass_of_man_equals_240kg_l1272_127248

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating that the mass of the man is 240 kg under given conditions. -/
theorem mass_of_man_equals_240kg 
  (boat_length : ℝ) 
  (boat_breadth : ℝ) 
  (boat_sinking : ℝ) 
  (water_density : ℝ) 
  (h1 : boat_length = 8) 
  (h2 : boat_breadth = 3) 
  (h3 : boat_sinking = 0.01) 
  (h4 : water_density = 1000) :
  mass_of_man boat_length boat_breadth boat_sinking water_density = 240 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_equals_240kg_l1272_127248


namespace NUMINAMATH_CALUDE_tiffany_treasures_l1272_127252

theorem tiffany_treasures (points_per_treasure : ℕ) (second_level_treasures : ℕ) (total_score : ℕ) :
  points_per_treasure = 6 →
  second_level_treasures = 5 →
  total_score = 48 →
  ∃ first_level_treasures : ℕ,
    first_level_treasures * points_per_treasure +
    second_level_treasures * points_per_treasure = total_score ∧
    first_level_treasures = 3 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_treasures_l1272_127252


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l1272_127283

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 64) : a / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l1272_127283


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1272_127278

theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_arith_mean : (a + b) / 2 = 5) 
  (h_geom_mean : Real.sqrt (a * b) = 4) 
  (h_a_gt_b : a > b) : 
  ∃ (k : ℝ), k = 1/2 ∧ 
  (∀ (x y : ℝ), (x^2 / a - y^2 / b = 1) → (y = k*x ∨ y = -k*x)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1272_127278


namespace NUMINAMATH_CALUDE_triangle_max_area_l1272_127277

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² + b² = c² + (2/3)ab and the circumradius is (3√2)/2,
    then the maximum possible area of the triangle is 4√2. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 = c^2 + (2/3) * a * b →
  R = (3 * Real.sqrt 2) / 2 →
  R = a / (2 * Real.sin A) →
  R = b / (2 * Real.sin B) →
  R = c / (2 * Real.sin C) →
  (∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧
               ∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ 4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1272_127277


namespace NUMINAMATH_CALUDE_rich_walk_distance_l1272_127265

/-- Calculates the total distance Rich walks based on the given conditions -/
def total_distance : ℝ :=
  let initial_distance := 20 + 200
  let left_turn_distance := 2 * initial_distance
  let halfway_distance := initial_distance + left_turn_distance
  let final_distance := halfway_distance + 0.5 * halfway_distance
  2 * final_distance

/-- Theorem stating that the total distance Rich walks is 1980 feet -/
theorem rich_walk_distance : total_distance = 1980 := by
  sorry

end NUMINAMATH_CALUDE_rich_walk_distance_l1272_127265


namespace NUMINAMATH_CALUDE_largest_difference_in_S_l1272_127217

def S : Set ℤ := {-20, -8, 0, 6, 10, 15, 25}

theorem largest_difference_in_S : 
  ∀ (a b : ℤ), a ∈ S → b ∈ S → (a - b) ≤ 45 ∧ ∃ (x y : ℤ), x ∈ S ∧ y ∈ S ∧ x - y = 45 :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_in_S_l1272_127217


namespace NUMINAMATH_CALUDE_oak_trees_planted_l1272_127295

/-- The number of oak trees planted today in the park -/
def trees_planted (current : ℕ) (final : ℕ) : ℕ := final - current

/-- Theorem stating that the number of oak trees planted today is 4 -/
theorem oak_trees_planted : trees_planted 5 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_planted_l1272_127295


namespace NUMINAMATH_CALUDE_football_game_attendance_l1272_127219

/-- Proves that the number of children attending a football game is 80, given the ticket prices, total attendance, and total money collected. -/
theorem football_game_attendance
  (adult_price : ℕ) -- Price of adult ticket in cents
  (child_price : ℕ) -- Price of child ticket in cents
  (total_attendance : ℕ) -- Total number of attendees
  (total_revenue : ℕ) -- Total revenue in cents
  (h1 : adult_price = 60)
  (h2 : child_price = 25)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 14000) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adults * adult_price + children * child_price = total_revenue ∧
    children = 80 :=
by sorry

end NUMINAMATH_CALUDE_football_game_attendance_l1272_127219


namespace NUMINAMATH_CALUDE_largest_product_of_three_exists_product_72_largest_product_is_72_l1272_127201

def S : Finset Int := {-4, -3, -1, 5, 6}

theorem largest_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (a * b * c : Int) ≤ 72 :=
sorry

theorem exists_product_72 : 
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 72 :=
sorry

theorem largest_product_is_72 : 
  (∀ (a b c : Int), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a * b * c : Int) ≤ 72) ∧
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 72) :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_exists_product_72_largest_product_is_72_l1272_127201


namespace NUMINAMATH_CALUDE_correct_assignment_is_correct_l1272_127216

-- Define the color type
inductive Color
| Red
| Blue
| Green

-- Define the assignment type
structure Assignment where
  one : Color
  two : Color
  three : Color

-- Define the correct assignment
def correct_assignment : Assignment :=
  { one := Color.Green
  , two := Color.Blue
  , three := Color.Red }

-- Theorem stating that the correct_assignment is indeed correct
theorem correct_assignment_is_correct : 
  correct_assignment.one = Color.Green ∧ 
  correct_assignment.two = Color.Blue ∧ 
  correct_assignment.three = Color.Red :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_is_correct_l1272_127216


namespace NUMINAMATH_CALUDE_sum_fractions_equals_11111_l1272_127256

theorem sum_fractions_equals_11111 : 
  4/5 + 9 * (4/5) + 99 * (4/5) + 999 * (4/5) + 9999 * (4/5) + 1 = 11111 := by
  sorry

end NUMINAMATH_CALUDE_sum_fractions_equals_11111_l1272_127256


namespace NUMINAMATH_CALUDE_first_negative_term_position_l1272_127240

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_negative_term_position
  (a₁ : ℤ)
  (d : ℤ)
  (h₁ : a₁ = 1031)
  (h₂ : d = -3) :
  (∀ k < 345, arithmeticSequence a₁ d k ≥ 0) ∧
  arithmeticSequence a₁ d 345 < 0 :=
sorry

end NUMINAMATH_CALUDE_first_negative_term_position_l1272_127240


namespace NUMINAMATH_CALUDE_expected_winnings_l1272_127259

-- Define the spinner outcomes
inductive Outcome
| Green
| Red
| Blue

-- Define the probability function
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Green => 1/4
  | Outcome.Red => 1/2
  | Outcome.Blue => 1/4

-- Define the winnings function
def winnings (o : Outcome) : ℤ :=
  match o with
  | Outcome.Green => 2
  | Outcome.Red => 4
  | Outcome.Blue => -6

-- Define the expected value function
def expectedValue : ℚ :=
  (probability Outcome.Green * winnings Outcome.Green) +
  (probability Outcome.Red * winnings Outcome.Red) +
  (probability Outcome.Blue * winnings Outcome.Blue)

-- Theorem stating the expected winnings
theorem expected_winnings : expectedValue = 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_l1272_127259


namespace NUMINAMATH_CALUDE_bouquet_cost_is_45_l1272_127288

/-- The cost of a bouquet consisting of two dozens of red roses and 3 sunflowers -/
def bouquet_cost (rose_price sunflower_price : ℚ) : ℚ :=
  (24 * rose_price) + (3 * sunflower_price)

/-- Theorem stating that the cost of the bouquet with given prices is $45 -/
theorem bouquet_cost_is_45 :
  bouquet_cost (3/2) 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_is_45_l1272_127288


namespace NUMINAMATH_CALUDE_tan_plus_cot_l1272_127235

theorem tan_plus_cot (α : Real) : 
  sinα - cosα = -Real.sqrt 5 / 2 → tanα + 1 / tanα = -8 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_cot_l1272_127235


namespace NUMINAMATH_CALUDE_min_socks_for_15_pairs_l1272_127247

/-- Represents the number of socks of each color in the box -/
structure SockBox where
  white : Nat
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Calculates the minimum number of socks needed to guarantee at least n pairs -/
def minSocksForPairs (box : SockBox) (n : Nat) : Nat :=
  (box.white.min n + box.red.min n + box.blue.min n + box.green.min n + box.yellow.min n) * 2 - 1

/-- The theorem stating the minimum number of socks needed for 15 pairs -/
theorem min_socks_for_15_pairs (box : SockBox) 
    (h_white : box.white = 150)
    (h_red : box.red = 120)
    (h_blue : box.blue = 90)
    (h_green : box.green = 60)
    (h_yellow : box.yellow = 30) :
    minSocksForPairs box 15 = 146 := by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_15_pairs_l1272_127247
