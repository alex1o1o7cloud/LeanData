import Mathlib

namespace NUMINAMATH_CALUDE_pizza_problem_l1728_172894

theorem pizza_problem (slices_per_pizza : ℕ) (games_played : ℕ) (avg_goals_per_game : ℕ) :
  slices_per_pizza = 12 →
  games_played = 8 →
  avg_goals_per_game = 9 →
  (games_played * avg_goals_per_game) / slices_per_pizza = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l1728_172894


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l1728_172837

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four sequentially meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  meshed : A.teeth * A.speed = B.teeth * B.speed ∧
           B.teeth * B.speed = C.teeth * C.speed ∧
           C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the proportion of angular speeds for the gear system -/
theorem gear_speed_proportion (sys : GearSystem) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.A.speed = k * (sys.B.teeth * sys.C.teeth * sys.D.teeth) ∧
    sys.B.speed = k * (sys.A.teeth * sys.C.teeth * sys.D.teeth) ∧
    sys.C.speed = k * (sys.A.teeth * sys.B.teeth * sys.D.teeth) ∧
    sys.D.speed = k * (sys.A.teeth * sys.B.teeth * sys.C.teeth) :=
  sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l1728_172837


namespace NUMINAMATH_CALUDE_rainfall_problem_l1728_172874

theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 30 →
  ratio = 1.5 →
  ∃ (first_week second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_problem_l1728_172874


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l1728_172876

/-- A positive arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = q * a n

theorem min_value_of_sequence (a : ℕ → ℝ) (m n : ℕ) :
  ArithmeticGeometricSequence a →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 4 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l1728_172876


namespace NUMINAMATH_CALUDE_monotonic_iff_m_geq_one_third_l1728_172879

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- State the theorem
theorem monotonic_iff_m_geq_one_third :
  ∀ m : ℝ, (∀ x : ℝ, Monotone (f m)) ↔ m ≥ 1/3 := by sorry

end NUMINAMATH_CALUDE_monotonic_iff_m_geq_one_third_l1728_172879


namespace NUMINAMATH_CALUDE_circle_point_x_value_l1728_172804

/-- Given a circle with diameter endpoints (-8, 0) and (32, 0), 
    if the point (x, 20) lies on this circle, then x = 12. -/
theorem circle_point_x_value 
  (x : ℝ) 
  (h : (x - 12)^2 + 20^2 = ((32 - (-8)) / 2)^2) : 
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_circle_point_x_value_l1728_172804


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1728_172823

/-- Given a line with equation 2x - 4y = 9, prove that any parallel line has slope 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (2 * x - 4 * y = 9) → (slope_of_parallel_line : ℝ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1728_172823


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l1728_172875

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l1728_172875


namespace NUMINAMATH_CALUDE_general_admission_price_general_admission_price_is_20_l1728_172890

/-- Calculates the price of a general admission ticket given the total number of tickets sold,
    total revenue, VIP ticket price, and the difference between general and VIP tickets sold. -/
theorem general_admission_price 
  (total_tickets : ℕ) 
  (total_revenue : ℝ) 
  (vip_price : ℝ) 
  (ticket_difference : ℕ) : ℝ :=
  let general_tickets := (total_tickets + ticket_difference) / 2
  let vip_tickets := total_tickets - general_tickets
  let general_price := (total_revenue - vip_price * vip_tickets) / general_tickets
  general_price

/-- The price of a general admission ticket is $20 given the specific conditions. -/
theorem general_admission_price_is_20 : 
  general_admission_price 320 7500 40 212 = 20 := by
  sorry

end NUMINAMATH_CALUDE_general_admission_price_general_admission_price_is_20_l1728_172890


namespace NUMINAMATH_CALUDE_ninth_grade_test_attendance_l1728_172862

theorem ninth_grade_test_attendance :
  let total_students : ℕ := 180
  let bombed_finals : ℕ := total_students / 4
  let remaining_students : ℕ := total_students - bombed_finals
  let passed_finals : ℕ := 70
  let less_than_d : ℕ := 20
  let took_test : ℕ := passed_finals + less_than_d
  let didnt_show_up : ℕ := remaining_students - took_test
  (didnt_show_up : ℚ) / remaining_students = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_test_attendance_l1728_172862


namespace NUMINAMATH_CALUDE_jesse_mall_trip_l1728_172852

def mall_trip (initial_amount novel_cost : ℕ) : ℕ :=
  let lunch_cost := 2 * novel_cost
  let total_spent := novel_cost + lunch_cost
  initial_amount - total_spent

theorem jesse_mall_trip :
  mall_trip 50 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_jesse_mall_trip_l1728_172852


namespace NUMINAMATH_CALUDE_girth_bound_l1728_172855

/-- The minimum degree of a graph G -/
def min_degree (G : Type*) : ℕ := sorry

/-- The girth of a graph G -/
def girth (G : Type*) : ℕ := sorry

/-- The number of vertices in a graph G -/
def num_vertices (G : Type*) : ℕ := sorry

/-- Theorem: For any graph G with minimum degree ≥ 3, the girth is less than 2 log |G| -/
theorem girth_bound (G : Type*) (h : min_degree G ≥ 3) : 
  girth G < 2 * Real.log (num_vertices G) := by
  sorry

end NUMINAMATH_CALUDE_girth_bound_l1728_172855


namespace NUMINAMATH_CALUDE_truck_fuel_distance_l1728_172868

/-- Given a truck that travels 300 miles on 10 gallons of fuel,
    prove that it will travel 450 miles on 15 gallons of fuel,
    assuming a proportional relationship between fuel consumption and distance. -/
theorem truck_fuel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ)
    (h1 : initial_distance = 300)
    (h2 : initial_fuel = 10)
    (h3 : new_fuel = 15)
    (h4 : initial_fuel > 0) :
  (new_fuel / initial_fuel) * initial_distance = 450 := by
  sorry

end NUMINAMATH_CALUDE_truck_fuel_distance_l1728_172868


namespace NUMINAMATH_CALUDE_bd_length_is_twelve_l1728_172803

-- Define the triangle ABC
def triangle_ABC : Type := Unit

-- Define point D
def point_D : Type := Unit

-- Define that B is a right angle
def B_is_right_angle (t : triangle_ABC) : Prop := sorry

-- Define that a circle with diameter BC intersects AC at D
def circle_intersects_AC (t : triangle_ABC) (d : point_D) : Prop := sorry

-- Define the area of triangle ABC
def area_ABC (t : triangle_ABC) : ℝ := 120

-- Define the length of AC
def length_AC (t : triangle_ABC) : ℝ := 20

-- Define the length of BD
def length_BD (t : triangle_ABC) (d : point_D) : ℝ := sorry

-- Theorem statement
theorem bd_length_is_twelve (t : triangle_ABC) (d : point_D) :
  B_is_right_angle t →
  circle_intersects_AC t d →
  length_BD t d = 12 :=
sorry

end NUMINAMATH_CALUDE_bd_length_is_twelve_l1728_172803


namespace NUMINAMATH_CALUDE_pyramid_volume_from_star_figure_l1728_172806

/-- The volume of a pyramid formed by folding a star figure cut from a square --/
theorem pyramid_volume_from_star_figure (outer_side : ℝ) (inner_side : ℝ) 
  (h_outer : outer_side = 40)
  (h_inner : inner_side = 15) :
  let base_area := inner_side ^ 2
  let midpoint_to_center := outer_side / 2
  let center_to_inner_side := inner_side / 2
  let triangle_height := midpoint_to_center - center_to_inner_side
  let pyramid_height := Real.sqrt (triangle_height ^ 2 - (inner_side / 2) ^ 2)
  let volume := (1 / 3) * base_area * pyramid_height
  volume = 750 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_from_star_figure_l1728_172806


namespace NUMINAMATH_CALUDE_some_number_divisibility_l1728_172856

theorem some_number_divisibility (x : ℕ) : (1425 * x * 1429) % 12 = 3 ↔ x % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_divisibility_l1728_172856


namespace NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l1728_172846

-- Define the function f
def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

-- Theorem 1: Solution set of f(x) ≥ 0
theorem solution_set_f_nonnegative :
  {x : ℝ | f x ≥ 0} = Set.Iic (-3/5) ∪ Set.Ici 1 := by sorry

-- Theorem 2: Range of a given the condition
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) →
  a ∈ Set.Iic (-3) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l1728_172846


namespace NUMINAMATH_CALUDE_min_pizzas_cover_scooter_cost_l1728_172872

def scooter_cost : ℕ := 8000
def earning_per_pizza : ℕ := 12
def cost_per_delivery : ℕ := 4

def min_pizzas : ℕ := 1000

theorem min_pizzas_cover_scooter_cost :
  ∀ p : ℕ, p ≥ min_pizzas →
  p * (earning_per_pizza - cost_per_delivery) ≥ scooter_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_cover_scooter_cost_l1728_172872


namespace NUMINAMATH_CALUDE_symmetric_function_properties_l1728_172869

/-- A function that is symmetric about the line x=1 and the point (2,0) -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f x) ∧ 
  (∀ x, f (2 + x) = -f x)

theorem symmetric_function_properties (f : ℝ → ℝ) (h : SymmetricFunction f) :
  (∀ x, f (2 - x) = f x) ∧
  (∀ x, f (4 - x) = -f x) ∧
  (∀ x, f (4 + x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_properties_l1728_172869


namespace NUMINAMATH_CALUDE_problem_statement_l1728_172836

/-- The problem statement as a theorem -/
theorem problem_statement 
  (ω : ℝ) 
  (hω : ω > 0)
  (a : ℝ → ℝ × ℝ)
  (b : ℝ → ℝ × ℝ)
  (ha : ∀ x, a x = (Real.sin (ω * x) + Real.cos (ω * x), Real.sqrt 3 * Real.cos (ω * x)))
  (hb : ∀ x, b x = (Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sin (ω * x)))
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (a x).1 * (b x).1 + (a x).2 * (b x).2)
  (hsymmetry : ∀ x, f (x + π / (2 * ω)) = f x)
  (A B C : ℝ)
  (hC : f C = 1)
  (c : ℝ)
  (hc : c = 2)
  (hsin : Real.sin C + Real.sin (B - A) = 3 * Real.sin (2 * A))
  : ω = 1 ∧ 
    (Real.sqrt 3 / 3 * c ^ 2 = 2 * Real.sqrt 3 / 3 ∨ 
     Real.sqrt 3 / 3 * c ^ 2 = 3 * Real.sqrt 3 / 7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1728_172836


namespace NUMINAMATH_CALUDE_hyperbola_tangent_slope_range_l1728_172839

/-- The range of slopes for a line passing through the right focus of a hyperbola
    and intersecting its right branch at exactly one point. -/
theorem hyperbola_tangent_slope_range (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = 1) →  -- Equation of the hyperbola
  ∃ (m : ℝ), -- Slope of the line
    (m ≥ -Real.sqrt 3 ∧ m ≤ Real.sqrt 3) ∧ -- Range of slopes
    (∃ (x₀ y₀ : ℝ), -- Point of intersection
      x₀^2 / 4 - y₀^2 / 12 = 1 ∧ -- Point lies on the hyperbola
      y₀ = m * (x₀ - (Real.sqrt 5))) ∧ -- Line passes through right focus (√5, 0)
    (∀ (x₁ y₁ : ℝ), -- Uniqueness of intersection
      x₁ ≠ x₀ →
      x₁^2 / 4 - y₁^2 / 12 = 1 →
      y₁ ≠ m * (x₁ - (Real.sqrt 5))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_slope_range_l1728_172839


namespace NUMINAMATH_CALUDE_florist_roses_count_l1728_172808

theorem florist_roses_count (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 50 → sold = 15 → picked = 21 → initial - sold + picked = 56 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_count_l1728_172808


namespace NUMINAMATH_CALUDE_g_neg_two_l1728_172858

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem g_neg_two : g (-2) = -16 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_l1728_172858


namespace NUMINAMATH_CALUDE_positive_difference_of_numbers_l1728_172897

theorem positive_difference_of_numbers (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_squares_eq : a^2 - b^2 = 40) : 
  |a - b| = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_of_numbers_l1728_172897


namespace NUMINAMATH_CALUDE_largest_stamp_collection_l1728_172845

theorem largest_stamp_collection (n : ℕ) (friends : ℕ) (extra : ℕ) : 
  friends = 15 →
  extra = 5 →
  n < 150 →
  n % friends = extra →
  ∀ m, m < 150 → m % friends = extra → m ≤ n →
  n = 140 :=
sorry

end NUMINAMATH_CALUDE_largest_stamp_collection_l1728_172845


namespace NUMINAMATH_CALUDE_wheel_probability_l1728_172828

theorem wheel_probability :
  let total_ratio : ℕ := 6 + 2 + 1 + 4
  let red_ratio : ℕ := 6
  let blue_ratio : ℕ := 1
  let target_ratio : ℕ := red_ratio + blue_ratio
  (target_ratio : ℚ) / total_ratio = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_wheel_probability_l1728_172828


namespace NUMINAMATH_CALUDE_exponent_of_p_in_product_l1728_172863

theorem exponent_of_p_in_product (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  ∃ (a b : ℕ), (a + 1) * (b + 1) = 32 ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_p_in_product_l1728_172863


namespace NUMINAMATH_CALUDE_uncle_james_height_difference_l1728_172888

theorem uncle_james_height_difference :
  ∀ (james_initial_height uncle_height james_growth : ℝ),
    james_initial_height = (2/3) * uncle_height →
    uncle_height = 72 →
    james_growth = 10 →
    uncle_height - (james_initial_height + james_growth) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_james_height_difference_l1728_172888


namespace NUMINAMATH_CALUDE_flour_weight_range_l1728_172831

/-- Given a bag of flour labeled as 25 ± 0.02kg, prove that its weight m is within the range 24.98kg ≤ m ≤ 25.02kg -/
theorem flour_weight_range (m : ℝ) (h : |m - 25| ≤ 0.02) : 24.98 ≤ m ∧ m ≤ 25.02 := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_range_l1728_172831


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l1728_172883

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) := x^3 - 3*x

-- Define the interval [-2, 0]
def interval := Set.Icc (-2 : ℝ) 0

-- Theorem statement
theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -2 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l1728_172883


namespace NUMINAMATH_CALUDE_work_group_ratio_l1728_172860

/-- Represents the work done by a group of people -/
structure WorkGroup where
  people : ℕ
  days : ℕ
  fraction : ℚ

/-- Given two work groups with specific conditions, prove the ratio of people in the second group to the first is 2:1 -/
theorem work_group_ratio (group1 group2 : WorkGroup) 
  (h1 : group1.days = 20)
  (h2 : group1.fraction = 1)
  (h3 : group2.days = 5)
  (h4 : group2.fraction = 1/2) :
  group2.people / group1.people = 2 := by
  sorry

#check work_group_ratio

end NUMINAMATH_CALUDE_work_group_ratio_l1728_172860


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l1728_172850

theorem inscribed_triangle_area (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ (r : ℝ), r = 4 ∧ ∃ (A B C : ℝ), 
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧ 
    c / Real.sin C = 2 * r) →
  a * b * c = 16 * Real.sqrt 2 →
  (1 / 2) * a * b * Real.sin (Real.arcsin (c / 8)) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l1728_172850


namespace NUMINAMATH_CALUDE_inequality_division_l1728_172841

theorem inequality_division (a b : ℝ) (h : a > b) : a / (-3) < b / (-3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_division_l1728_172841


namespace NUMINAMATH_CALUDE_tracy_popped_half_balloons_l1728_172827

theorem tracy_popped_half_balloons 
  (brooke_balloons : ℕ) 
  (tracy_initial_balloons : ℕ) 
  (total_balloons_after_popping : ℕ) 
  (tracy_popped_fraction : ℚ) :
  brooke_balloons = 20 →
  tracy_initial_balloons = 30 →
  total_balloons_after_popping = 35 →
  brooke_balloons + tracy_initial_balloons * (1 - tracy_popped_fraction) = total_balloons_after_popping →
  tracy_popped_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tracy_popped_half_balloons_l1728_172827


namespace NUMINAMATH_CALUDE_correct_average_l1728_172870

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 16 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 35 → 
  (n * incorrect_avg + (correct_num - incorrect_num)) / n = 17 := by
sorry

end NUMINAMATH_CALUDE_correct_average_l1728_172870


namespace NUMINAMATH_CALUDE_f_property_f_upper_bound_minimum_M_l1728_172825

/-- The function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The derivative of f(x) -/
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x + b

theorem f_property (b c : ℝ) :
  ∀ x, f_derivative b x ≤ f b c x := sorry

theorem f_upper_bound (b c : ℝ) (h : ∀ x, f_derivative b x ≤ f b c x) :
  ∀ x ≥ 0, f b c x ≤ (x + c)^2 := sorry

theorem minimum_M (b c : ℝ) (h : ∀ x, f_derivative b x ≤ f b c x) :
  ∃ M, (∀ b c, f b c c - f b c b ≤ M * (c^2 - b^2)) ∧
       (∀ M', (∀ b c, f b c c - f b c b ≤ M' * (c^2 - b^2)) → M ≤ M') ∧
       M = 3/2 := sorry

end NUMINAMATH_CALUDE_f_property_f_upper_bound_minimum_M_l1728_172825


namespace NUMINAMATH_CALUDE_finite_quadruples_factorial_sum_l1728_172844

theorem finite_quadruples_factorial_sum : 
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    ∀ (a b c n : ℕ), 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < n → 
      (n.factorial = a^(n-1) + b^(n-1) + c^(n-1)) → 
      (a, b, c, n) ∈ S := by
sorry

end NUMINAMATH_CALUDE_finite_quadruples_factorial_sum_l1728_172844


namespace NUMINAMATH_CALUDE_box_decoration_combinations_l1728_172843

/-- The number of paint color options available -/
def num_colors : ℕ := 4

/-- The number of decoration options available -/
def num_decorations : ℕ := 3

/-- The total number of combinations for painting and decorating a box -/
def total_combinations : ℕ := num_colors * num_decorations

theorem box_decoration_combinations :
  total_combinations = 12 :=
by sorry

end NUMINAMATH_CALUDE_box_decoration_combinations_l1728_172843


namespace NUMINAMATH_CALUDE_smallest_area_ellipse_l1728_172847

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- Checks if an ellipse contains a circle with center (h, 0) and radius 2 -/
def Ellipse.contains_circle (e : Ellipse) (h : ℝ) : Prop :=
  ∀ x y : ℝ, (x - h)^2 + y^2 = 4 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- The theorem stating the smallest possible area of the ellipse -/
theorem smallest_area_ellipse (e : Ellipse) 
  (h_contains_circle1 : e.contains_circle 2)
  (h_contains_circle2 : e.contains_circle (-2)) :
  ∃ k : ℝ, k = Real.sqrt 3 ∧ 
    ∀ e' : Ellipse, e'.contains_circle 2 → e'.contains_circle (-2) → 
      π * e'.a * e'.b ≥ k * π :=
sorry

end NUMINAMATH_CALUDE_smallest_area_ellipse_l1728_172847


namespace NUMINAMATH_CALUDE_product_of_numbers_l1728_172818

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 460) : x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1728_172818


namespace NUMINAMATH_CALUDE_train_journey_l1728_172807

theorem train_journey
  (average_speed : ℝ)
  (first_distance : ℝ)
  (first_time : ℝ)
  (second_time : ℝ)
  (h1 : average_speed = 70)
  (h2 : first_distance = 225)
  (h3 : first_time = 3.5)
  (h4 : second_time = 5)
  : (average_speed * (first_time + second_time) - first_distance) = 370 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_l1728_172807


namespace NUMINAMATH_CALUDE_x_three_times_y_l1728_172885

theorem x_three_times_y (q : ℚ) (x y : ℚ) 
  (hx : x = 5 - q) 
  (hy : y = 3 * q - 1) : 
  q = 4/5 ↔ x = 3 * y := by
sorry

end NUMINAMATH_CALUDE_x_three_times_y_l1728_172885


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l1728_172824

/-- A hyperbola with equation mx^2 + y^2 = 1 where the length of its imaginary axis
    is twice the length of its real axis -/
structure Hyperbola (m : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + y^2 = 1
  axis_ratio : (imaginary_axis_length : ℝ) = 2 * (real_axis_length : ℝ)

/-- The value of m for a hyperbola with the given properties is -1/4 -/
theorem hyperbola_m_value (h : Hyperbola m) : m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l1728_172824


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1728_172880

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 5*x > 22) → x ≤ -4 ∧ 7 - 5*(-4) > 22 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1728_172880


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l1728_172866

theorem minimum_value_of_function (x : ℝ) (h : x > 3) :
  (1 / (x - 3) + x) ≥ 5 ∧ ∃ y > 3, 1 / (y - 3) + y = 5 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l1728_172866


namespace NUMINAMATH_CALUDE_a_max_at_6_l1728_172809

-- Define the sequence a_n
def a (n : ℤ) : ℚ := (10 / 11) ^ n * (3 * n + 13)

-- Theorem stating that a_n is maximized when n = 6
theorem a_max_at_6 : ∀ (k : ℤ), a 6 ≥ a k := by sorry

end NUMINAMATH_CALUDE_a_max_at_6_l1728_172809


namespace NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l1728_172859

theorem ceiling_of_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l1728_172859


namespace NUMINAMATH_CALUDE_age_difference_l1728_172801

/-- Given three people A, B, and C, where the total age of A and B is more than
    the total age of B and C, and C is 13 years younger than A, prove that the
    difference between (A + B) and (B + C) is 13 years. -/
theorem age_difference (A B C : ℕ) 
  (h1 : A + B > B + C) 
  (h2 : C = A - 13) : 
  (A + B) - (B + C) = 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1728_172801


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l1728_172815

theorem fraction_cube_equality : (45000 ^ 3) / (15000 ^ 3) = 27 := by sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l1728_172815


namespace NUMINAMATH_CALUDE_hyperbola_auxiliary_lines_l1728_172857

/-- Represents a hyperbola with given equation and asymptotes -/
structure Hyperbola where
  a : ℝ
  eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / 16 = 1
  asymptote : ∀ x : ℝ, ∃ y : ℝ, y = 4/3 * x ∨ y = -4/3 * x
  a_pos : a > 0

/-- The auxiliary lines of a hyperbola -/
def auxiliary_lines (h : Hyperbola) : Set ℝ :=
  {x : ℝ | x = 9/5 ∨ x = -9/5}

/-- Theorem stating that the auxiliary lines of the given hyperbola are x = ±9/5 -/
theorem hyperbola_auxiliary_lines (h : Hyperbola) :
  auxiliary_lines h = {x : ℝ | x = 9/5 ∨ x = -9/5} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_auxiliary_lines_l1728_172857


namespace NUMINAMATH_CALUDE_money_distribution_correctness_l1728_172849

def bag_distribution : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 489]

def sum_subset (l : List Nat) (subset : List Bool) : Nat :=
  (l.zip subset).foldl (λ acc (x, b) => acc + if b then x else 0) 0

theorem money_distribution_correctness :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 1000 →
    ∃ subset : List Bool, subset.length = 10 ∧ sum_subset bag_distribution subset = n :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_correctness_l1728_172849


namespace NUMINAMATH_CALUDE_parabola_hyperbola_problem_l1728_172835

/-- Given a parabola y² = 2px (p > 0) with a point M(1, m) on it, where the distance from M to the focus is 5,
    and a hyperbola x²/a² - y²/b² = 1 with its left vertex at point A,
    if one of the asymptotes of the hyperbola is perpendicular to line AM, then a = 1/4. -/
theorem parabola_hyperbola_problem (p m a b : ℝ) : 
  p > 0 → 
  m^2 = 2*p*1 → 
  (1 - p/2)^2 + m^2 = 5^2 → 
  ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → 
  ∃ (slope_AM slope_asymptote : ℝ), 
    slope_AM * slope_asymptote = -1 → 
    a = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_parabola_hyperbola_problem_l1728_172835


namespace NUMINAMATH_CALUDE_computer_contract_probability_l1728_172873

theorem computer_contract_probability (p_not_software : ℝ) (p_at_least_one : ℝ) (p_both : ℝ) 
  (h1 : p_not_software = 3/5)
  (h2 : p_at_least_one = 5/6)
  (h3 : p_both = 0.31666666666666654) :
  let p_software := 1 - p_not_software
  let p_hardware := p_at_least_one + p_both - p_software
  p_hardware = 0.75 := by
sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l1728_172873


namespace NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l1728_172816

/-- Represents the duration of each light color in seconds -/
structure LightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total cycle time of the traffic light -/
def totalCycleTime (d : LightDuration) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of seeing a red light -/
def redLightProbability (d : LightDuration) : ℚ :=
  d.red / totalCycleTime d

/-- Theorem: The probability of seeing a red light is 2/5 given the specified light durations -/
theorem red_light_probability_is_two_fifths (d : LightDuration) 
    (h1 : d.red = 30)
    (h2 : d.yellow = 5)
    (h3 : d.green = 40) : 
  redLightProbability d = 2/5 := by
  sorry

#eval redLightProbability ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l1728_172816


namespace NUMINAMATH_CALUDE_josiah_cookies_per_day_l1728_172877

/-- Proves that Josiah purchased 2 cookies each day in March given the conditions --/
theorem josiah_cookies_per_day :
  let total_spent : ℕ := 992
  let cookie_price : ℕ := 16
  let days_in_march : ℕ := 31
  (total_spent / cookie_price) / days_in_march = 2 := by
  sorry

end NUMINAMATH_CALUDE_josiah_cookies_per_day_l1728_172877


namespace NUMINAMATH_CALUDE_coefficient_c_positive_l1728_172813

-- Define a quadratic trinomial
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for no roots
def no_roots (a b c : ℝ) : Prop := ∀ x, quadratic_trinomial a b c x ≠ 0

-- Theorem statement
theorem coefficient_c_positive
  (a b c : ℝ)
  (h1 : no_roots a b c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end NUMINAMATH_CALUDE_coefficient_c_positive_l1728_172813


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1728_172861

def is_midpoint (m x y : ℝ × ℝ) : Prop :=
  m.1 = (x.1 + y.1) / 2 ∧ m.2 = (x.2 + y.2) / 2

theorem line_segment_endpoint (endpoint1 midpoint : ℝ × ℝ) 
  (h : is_midpoint midpoint endpoint1 (1, 18)) : 
  endpoint1 = (5, 2) ∧ midpoint = (3, 10) → (1, 18) = (1, 18) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1728_172861


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1728_172811

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1728_172811


namespace NUMINAMATH_CALUDE_equation_positive_root_m_value_l1728_172886

theorem equation_positive_root_m_value (m x : ℝ) : 
  (m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) → 
  (x > 0) → 
  (m = 6 ∨ m = 12) :=
by sorry

end NUMINAMATH_CALUDE_equation_positive_root_m_value_l1728_172886


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l1728_172865

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l1728_172865


namespace NUMINAMATH_CALUDE_point_distance_to_line_l1728_172889

/-- The distance from a point (a, 2) to the line x - y + 3 = 0 is 1, where a > 0 -/
def distance_to_line (a : ℝ) : Prop :=
  a > 0 ∧ |a + 1| / Real.sqrt 2 = 1

/-- Theorem: If the distance from (a, 2) to the line x - y + 3 = 0 is 1, then a = √2 - 1 -/
theorem point_distance_to_line (a : ℝ) (h : distance_to_line a) : a = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_to_line_l1728_172889


namespace NUMINAMATH_CALUDE_boat_accident_proof_l1728_172810

/-- The number of sheep that drowned in a boat accident -/
def drowned_sheep : ℕ := 3

theorem boat_accident_proof :
  let initial_sheep : ℕ := 20
  let initial_cows : ℕ := 10
  let initial_dogs : ℕ := 14
  let drowned_cows : ℕ := 2 * drowned_sheep
  let survived_dogs : ℕ := initial_dogs
  let total_survived : ℕ := 35
  total_survived = (initial_sheep - drowned_sheep) + (initial_cows - drowned_cows) + survived_dogs :=
by sorry

end NUMINAMATH_CALUDE_boat_accident_proof_l1728_172810


namespace NUMINAMATH_CALUDE_julia_watch_collection_l1728_172892

theorem julia_watch_collection (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = (silver_watches + bronze_watches + gold_watches) / 10 →
  silver_watches + bronze_watches + gold_watches = 88 :=
by sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l1728_172892


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_one_l1728_172820

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum : ℕ → ℝ
  sum_def : ∀ n, sum n = (n * (a 1 + a n)) / 2
  sum_condition : sum 2 = sum 6
  a4_condition : a 4 = 1

/-- The fifth term of the special arithmetic sequence is -1 -/
theorem fifth_term_is_negative_one (seq : ArithmeticSequence) : seq.a 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_one_l1728_172820


namespace NUMINAMATH_CALUDE_mask_production_in_july_l1728_172899

def initial_production : ℕ := 3000
def months_passed : ℕ := 4

theorem mask_production_in_july :
  initial_production * (2 ^ months_passed) = 48000 :=
by
  sorry

end NUMINAMATH_CALUDE_mask_production_in_july_l1728_172899


namespace NUMINAMATH_CALUDE_binomial_sum_abs_coefficients_l1728_172821

theorem binomial_sum_abs_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ),
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_abs_coefficients_l1728_172821


namespace NUMINAMATH_CALUDE_internal_curve_convexity_l1728_172805

-- Define a curve as a function from ℝ to ℝ × ℝ
def Curve := ℝ → ℝ × ℝ

-- Define convexity for a curve
def IsConvex (c : Curve) : Prop := sorry

-- Define the r-neighborhood of a curve
def RNeighborhood (c : Curve) (r : ℝ) : Set (ℝ × ℝ) := sorry

-- Define what it means for a curve to bound a set
def Bounds (c : Curve) (s : Set (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem internal_curve_convexity 
  (K : Curve) (r : ℝ) (C : Curve) 
  (h_K_convex : IsConvex K) 
  (h_r_pos : r > 0) 
  (h_C_bounds : Bounds C (RNeighborhood K r)) : 
  IsConvex C := by
  sorry

end NUMINAMATH_CALUDE_internal_curve_convexity_l1728_172805


namespace NUMINAMATH_CALUDE_investment_growth_l1728_172826

/-- Represents the investment growth over a two-year period -/
theorem investment_growth 
  (initial_investment : ℝ) 
  (final_investment : ℝ) 
  (growth_rate : ℝ) 
  (h1 : initial_investment = 1500)
  (h2 : final_investment = 4250)
  (h3 : initial_investment * (1 + growth_rate)^2 = final_investment) :
  1500 * (1 + growth_rate)^2 = 4250 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1728_172826


namespace NUMINAMATH_CALUDE_min_n_for_expansion_terms_min_n_value_l1728_172878

theorem min_n_for_expansion_terms (n : ℕ) : (n + 1) ^ 2 ≥ 2021 ↔ n ≥ 44 := by sorry

theorem min_n_value : ∃ (n : ℕ), n > 0 ∧ (n + 1) ^ 2 ≥ 2021 ∧ ∀ (m : ℕ), m > 0 → (m + 1) ^ 2 ≥ 2021 → m ≥ n := by
  use 44
  sorry

end NUMINAMATH_CALUDE_min_n_for_expansion_terms_min_n_value_l1728_172878


namespace NUMINAMATH_CALUDE_tangent_sum_difference_l1728_172867

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_difference_l1728_172867


namespace NUMINAMATH_CALUDE_no_chord_length_8_in_circle_radius_3_l1728_172842

theorem no_chord_length_8_in_circle_radius_3 (r : ℝ) (chord_length : ℝ) :
  r = 3 → chord_length ≤ 2 * r → chord_length ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_chord_length_8_in_circle_radius_3_l1728_172842


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l1728_172838

-- Define the universal set U
def U : Set ℝ := {x | x ≥ 0}

-- Define the set P
def P : Set ℝ := {1}

-- Theorem statement
theorem complement_of_P_in_U :
  (U \ P) = {x : ℝ | x ≥ 0 ∧ x ≠ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l1728_172838


namespace NUMINAMATH_CALUDE_bridge_length_l1728_172864

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 300 →
  train_speed_kmh = 35 →
  time_to_pass = 42.68571428571429 →
  ∃ (bridge_length : ℝ),
    bridge_length = 115 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * time_to_pass :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1728_172864


namespace NUMINAMATH_CALUDE_problem_solution_l1728_172854

theorem problem_solution : 
  (3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2) ∧ 
  (Real.sqrt 6 * Real.sqrt 3 + Real.sqrt 2 - 6 * Real.sqrt (1/2) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1728_172854


namespace NUMINAMATH_CALUDE_triangle_arctan_sum_l1728_172848

/-- Given a triangle ABC with sides a, b, c and angles α, β, γ in arithmetic progression
    with the smallest angle α = π/6, prove that arctan(a/(c+b)) + arctan(b/(c+a)) = π/4 -/
theorem triangle_arctan_sum (a b c : ℝ) (α β γ : ℝ) :
  α = π/6 →
  β = α + (γ - α)/2 →
  γ = α + 2*(γ - α)/2 →
  α + β + γ = π →
  a^2 + b^2 = c^2 →
  Real.arctan (a/(c+b)) + Real.arctan (b/(c+a)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arctan_sum_l1728_172848


namespace NUMINAMATH_CALUDE_z_ratio_equals_neg_i_l1728_172817

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the condition that z₁ and z₂ are symmetric with respect to the imaginary axis
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

-- Theorem statement
theorem z_ratio_equals_neg_i
  (h_sym : symmetric_wrt_imaginary_axis z₁ z₂)
  (h_z₁ : z₁ = 1 + I) :
  z₁ / z₂ = -I :=
sorry

end NUMINAMATH_CALUDE_z_ratio_equals_neg_i_l1728_172817


namespace NUMINAMATH_CALUDE_missing_number_proof_l1728_172822

def known_numbers : List ℤ := [744, 745, 747, 748, 752, 752, 753, 755, 755]

theorem missing_number_proof (total_count : ℕ) (average : ℤ) (missing_number : ℤ) :
  total_count = 10 →
  average = 750 →
  missing_number = 1549 →
  (List.sum known_numbers + missing_number) / total_count = average :=
by sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1728_172822


namespace NUMINAMATH_CALUDE_payment_combinations_eq_six_l1728_172853

/-- Represents the number of ways to make a payment of 230 yuan using given bills -/
def payment_combinations : ℕ :=
  (Finset.filter (fun (x, y, z) => 
    50 * x + 20 * y + 10 * z = 230 ∧ 
    x ≤ 5 ∧ y ≤ 6 ∧ z ≤ 7)
    (Finset.product (Finset.range 6) (Finset.product (Finset.range 7) (Finset.range 8)))).card

/-- The theorem stating that there are exactly 6 ways to make the payment -/
theorem payment_combinations_eq_six : payment_combinations = 6 := by
  sorry

end NUMINAMATH_CALUDE_payment_combinations_eq_six_l1728_172853


namespace NUMINAMATH_CALUDE_remainder_101_pow_36_mod_100_l1728_172891

theorem remainder_101_pow_36_mod_100 : 101^36 % 100 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_101_pow_36_mod_100_l1728_172891


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l1728_172802

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 8
  let rectangle_length : ℝ := 12
  let circle_radius : ℝ := 12
  let rectangle_area := rectangle_width * rectangle_length
  let circle_area := Real.pi * circle_radius ^ 2
  let overlap_area := (1 / 4 : ℝ) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 108 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l1728_172802


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l1728_172896

/-- Proves that the marked price should be 140% of the list price given the specified conditions --/
theorem merchant_pricing_strategy (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := list_price * 0.7
  let marked_price := list_price * 1.4
  let selling_price := marked_price * 0.75
  let after_tax_price := selling_price * 0.95
  after_tax_price - purchase_price = 0.3 * after_tax_price := by
  sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l1728_172896


namespace NUMINAMATH_CALUDE_square_area_error_l1728_172833

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1728_172833


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l1728_172881

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 14)
  (h3 : Complex.abs (w - 2*z) = 2) :
  Complex.abs (w^3 + z^3) = 38 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l1728_172881


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l1728_172832

/-- Given a circle with polar coordinate equation ρ = 2(cosθ + sinθ), 
    the polar coordinates of its center are (√2, π/4) -/
theorem circle_center_polar_coordinates :
  ∀ ρ θ : ℝ, 
  ρ = 2 * (Real.cos θ + Real.sin θ) →
  ∃ r α : ℝ, 
    r = Real.sqrt 2 ∧ 
    α = π / 4 ∧ 
    (r * Real.cos α - 1)^2 + (r * Real.sin α - 1)^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l1728_172832


namespace NUMINAMATH_CALUDE_shopping_cost_difference_l1728_172800

theorem shopping_cost_difference (shirt wallet food : ℝ) 
  (shirt_cost : shirt = wallet / 3)
  (wallet_more_expensive : wallet > food)
  (food_cost : food = 30)
  (total_spent : shirt + wallet + food = 150) :
  wallet - food = 60 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cost_difference_l1728_172800


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1728_172882

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- The focal distance of the hyperbola -/
  focal_distance : ℝ
  /-- The distance from a focus to an asymptote -/
  focus_to_asymptote : ℝ
  /-- Assumption that the foci are on the x-axis -/
  foci_on_x_axis : Bool

/-- The equation of a hyperbola given its properties -/
def hyperbola_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ 
    h.focal_distance = 4 ∧ 
    h.focus_to_asymptote = Real.sqrt 3 ∧ 
    h.foci_on_x_axis = true

/-- Theorem stating that a hyperbola with the given properties has the specified equation -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  h.focal_distance = 4 →
  h.focus_to_asymptote = Real.sqrt 3 →
  h.foci_on_x_axis = true →
  hyperbola_equation h :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1728_172882


namespace NUMINAMATH_CALUDE_button_sequence_l1728_172812

theorem button_sequence (a : Fin 6 → ℕ) : 
  a 0 = 1 ∧ 
  (∀ i : Fin 5, a (i + 1) = 3 * a i) ∧ 
  a 4 = 81 ∧ 
  a 5 = 243 → 
  a 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_button_sequence_l1728_172812


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l1728_172893

theorem three_digit_number_proof :
  ∀ a b c : ℕ,
  (100 ≤ a * 100 + b * 10 + c) → 
  (a * 100 + b * 10 + c < 1000) →
  (a * (b + c) = 33) →
  (b * (a + c) = 40) →
  (a * 100 + b * 10 + c = 347) :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l1728_172893


namespace NUMINAMATH_CALUDE_quadratic_radicals_same_type_l1728_172884

-- Define the two quadratic expressions
def f (a : ℝ) : ℝ := 3 * a - 8
def g (a : ℝ) : ℝ := 17 - 2 * a

-- Theorem statement
theorem quadratic_radicals_same_type :
  ∃ (a : ℝ), a = 5 ∧ f a = g a :=
sorry

end NUMINAMATH_CALUDE_quadratic_radicals_same_type_l1728_172884


namespace NUMINAMATH_CALUDE_inequality_proof_l1728_172829

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 - b*c) / (2*a^2 + b*c) + (b^2 - c*a) / (2*b^2 + c*a) + (c^2 - a*b) / (2*c^2 + a*b) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1728_172829


namespace NUMINAMATH_CALUDE_cost_of_72_tulips_l1728_172819

/-- Represents the cost of tulips given the number of tulips -/
def tulip_cost (n : ℕ) : ℚ :=
  let base_cost := (36 : ℚ) * n / 18
  if n > 50 then base_cost * (1 - 1/5) else base_cost

/-- Theorem stating the cost of 72 tulips -/
theorem cost_of_72_tulips : tulip_cost 72 = 115.2 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_72_tulips_l1728_172819


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1728_172851

theorem factor_of_polynomial (x : ℝ) : 
  (x - 1/2) ∣ (8*x^3 + 17*x^2 + 2*x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1728_172851


namespace NUMINAMATH_CALUDE_histogram_group_width_l1728_172898

/-- Represents a group in a frequency histogram -/
structure HistogramGroup where
  a : ℝ
  b : ℝ
  m : ℝ  -- frequency
  h : ℝ  -- height
  h_pos : h > 0

/-- Theorem: The width of a histogram group is equal to its frequency divided by its height -/
theorem histogram_group_width (g : HistogramGroup) : |g.a - g.b| = g.m / g.h := by
  sorry

end NUMINAMATH_CALUDE_histogram_group_width_l1728_172898


namespace NUMINAMATH_CALUDE_hostel_expenditure_increase_l1728_172830

theorem hostel_expenditure_increase 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (budget_decrease : ℕ) 
  (new_total_expenditure : ℕ) 
  (h1 : initial_students = 100)
  (h2 : new_students = 132)
  (h3 : budget_decrease = 10)
  (h4 : new_total_expenditure = 5400) :
  ∃ (original_avg_budget : ℕ),
    new_total_expenditure - initial_students * original_avg_budget = 300 := by
  sorry

end NUMINAMATH_CALUDE_hostel_expenditure_increase_l1728_172830


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits_l1728_172895

/-- A 3-digit number -/
def ThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- The sum of squares of digits of a natural number -/
def SumOfSquaresOfDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * hundreds + tens * tens + ones * ones

/-- The main theorem -/
theorem three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits :
  ∃! (s : Finset ℕ), s.card = 2 ∧ 
    ∀ n ∈ s, ThreeDigitNumber n ∧ 
             n % 11 = 0 ∧
             n / 11 = SumOfSquaresOfDigits n :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits_l1728_172895


namespace NUMINAMATH_CALUDE_incorrect_calculation_l1728_172887

theorem incorrect_calculation (h1 : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6)
  (h2 : Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3)
  (h3 : (-Real.sqrt 2)^2 = 2) :
  Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l1728_172887


namespace NUMINAMATH_CALUDE_units_digit_sum_l1728_172834

theorem units_digit_sum (n m : ℕ) : (35^87 + 3^45) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_l1728_172834


namespace NUMINAMATH_CALUDE_malcolm_flat_path_time_l1728_172840

/-- Represents the time in minutes for different parts of Malcolm's routes to school -/
structure RouteTime where
  uphill : ℕ
  path : ℕ
  final : ℕ

/-- Calculates the total time for the first route -/
def first_route_time (r : RouteTime) : ℕ :=
  r.uphill + r.path + r.final

/-- Calculates the total time for the second route -/
def second_route_time (flat_path : ℕ) : ℕ :=
  flat_path + 2 * flat_path

/-- Theorem stating the correct time Malcolm spent on the flat path in the second route -/
theorem malcolm_flat_path_time : ∃ (r : RouteTime) (flat_path : ℕ),
  r.uphill = 6 ∧
  r.path = 2 * r.uphill ∧
  r.final = (r.uphill + r.path) / 3 ∧
  second_route_time flat_path = first_route_time r + 18 ∧
  flat_path = 14 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_flat_path_time_l1728_172840


namespace NUMINAMATH_CALUDE_gcd_lcm_product_8_12_l1728_172814

theorem gcd_lcm_product_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_8_12_l1728_172814


namespace NUMINAMATH_CALUDE_range_of_a_l1728_172871

theorem range_of_a (a : ℝ) : 
  (a + 1)^(-1/2 : ℝ) < (3 - 2*a)^(-1/2 : ℝ) → 
  2/3 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1728_172871
