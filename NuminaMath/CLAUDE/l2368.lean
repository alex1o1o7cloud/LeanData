import Mathlib

namespace jenny_garden_area_l2368_236884

/-- Represents a rectangular garden with fence posts. -/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  shorter_side_posts : ℕ
  longer_side_posts : ℕ

/-- Calculates the area of a rectangular garden given its specifications. -/
def garden_area (g : Garden) : ℕ :=
  (g.shorter_side_posts - 1) * g.post_spacing * ((g.longer_side_posts - 1) * g.post_spacing)

/-- Theorem: The area of Jenny's garden is 144 square yards. -/
theorem jenny_garden_area :
  ∀ g : Garden,
    g.total_posts = 24 →
    g.post_spacing = 3 →
    g.longer_side_posts = 3 * g.shorter_side_posts →
    g.total_posts = 2 * (g.shorter_side_posts + g.longer_side_posts - 2) →
    garden_area g = 144 := by
  sorry

#eval garden_area { total_posts := 24, post_spacing := 3, shorter_side_posts := 3, longer_side_posts := 9 }

end jenny_garden_area_l2368_236884


namespace triangle_with_angle_ratio_is_right_angled_l2368_236855

/-- Given a triangle ABC where the ratio of angles A : B : C is 2 : 3 : 5, 
    prove that one of the angles is 90°. -/
theorem triangle_with_angle_ratio_is_right_angled (A B C : ℝ) 
  (h_triangle : A + B + C = 180) 
  (h_ratio : ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 5*x) : 
  A = 90 ∨ B = 90 ∨ C = 90 := by
  sorry

end triangle_with_angle_ratio_is_right_angled_l2368_236855


namespace towel_area_decrease_l2368_236887

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let original_area := L * B
  let new_length := 0.8 * L
  let new_breadth := 0.8 * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.36 := by
  sorry

end towel_area_decrease_l2368_236887


namespace hyperbola_eccentricity_l2368_236858

/-- Represents a hyperbola with equation x²/m - y²/3 = 1 -/
structure Hyperbola where
  m : ℝ
  focus : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  h.focus = (2, 0) → eccentricity h = 2 :=
sorry

end hyperbola_eccentricity_l2368_236858


namespace sin_value_at_pi_over_four_l2368_236837

theorem sin_value_at_pi_over_four 
  (φ : Real) 
  (ω : Real)
  (h1 : (- 4 : Real) / 5 = Real.cos φ ∧ (3 : Real) / 5 = Real.sin φ)
  (h2 : (2 * Real.pi) / ω = Real.pi)
  (h3 : ω > 0) :
  Real.sin ((2 : Real) * Real.pi / 4 + φ) = - (4 : Real) / 5 := by
sorry

end sin_value_at_pi_over_four_l2368_236837


namespace jia_test_probability_l2368_236827

/-- The probability of passing a test with given parameters -/
def test_pass_probability (total_questions n_correct_known n_selected n_to_pass : ℕ) : ℚ :=
  let favorable_outcomes := Nat.choose n_correct_known 2 * Nat.choose (total_questions - n_correct_known) 1 +
                            Nat.choose n_correct_known 3
  let total_outcomes := Nat.choose total_questions n_selected
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability of Jia passing the test -/
theorem jia_test_probability :
  test_pass_probability 10 5 3 2 = 1/2 := by
  sorry

end jia_test_probability_l2368_236827


namespace jerrys_breakfast_theorem_l2368_236878

/-- Calculates the total calories in Jerry's breakfast -/
def jerrys_breakfast_calories : ℕ :=
  let pancake_calories : ℕ := 120
  let bacon_calories : ℕ := 100
  let cereal_calories : ℕ := 200
  let num_pancakes : ℕ := 6
  let num_bacon_strips : ℕ := 2
  let num_cereal_bowls : ℕ := 1
  (pancake_calories * num_pancakes) + (bacon_calories * num_bacon_strips) + (cereal_calories * num_cereal_bowls)

/-- Proves that Jerry's breakfast contains 1120 calories -/
theorem jerrys_breakfast_theorem : jerrys_breakfast_calories = 1120 := by
  sorry

end jerrys_breakfast_theorem_l2368_236878


namespace units_digit_of_3_pow_2011_l2368_236897

def units_digit (n : ℕ) : ℕ := n % 10

def power_of_3_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem units_digit_of_3_pow_2011 :
  units_digit (3^2011) = power_of_3_units_digit 2011 :=
by sorry

end units_digit_of_3_pow_2011_l2368_236897


namespace tim_placed_three_pencils_l2368_236869

/-- Given that there were initially 2 pencils in a drawer and after Tim placed some pencils
    there are now 5 pencils in total, prove that Tim placed 3 pencils in the drawer. -/
theorem tim_placed_three_pencils (initial_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : initial_pencils = 2) 
  (h2 : total_pencils = 5) :
  total_pencils - initial_pencils = 3 := by
  sorry

end tim_placed_three_pencils_l2368_236869


namespace intersection_point_ratio_l2368_236846

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type with 60° inclination passing through (1, 0) -/
structure Line where
  x : ℝ
  y : ℝ
  eq : y = Real.sqrt 3 * (x - 1)

/-- Intersection point of the parabola and the line -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : Parabola
  on_line : Line

/-- Theorem stating the ratio of distances from intersection points to focus -/
theorem intersection_point_ratio 
  (A B : IntersectionPoint) 
  (h1 : A.x + 1 > B.x + 1) : 
  (A.x + 1) / (B.x + 1) = 3 := by sorry

end intersection_point_ratio_l2368_236846


namespace parallelogram_height_l2368_236816

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 200 → base = 10 → area = base * height → height = 20 := by
  sorry

end parallelogram_height_l2368_236816


namespace carriage_hire_cost_l2368_236899

/-- The cost of hiring a carriage for a journey, given:
  * The distance to the destination
  * The speed of the horse
  * The hourly rate for the carriage
  * A flat fee for the service
-/
theorem carriage_hire_cost 
  (distance : ℝ) 
  (speed : ℝ) 
  (hourly_rate : ℝ) 
  (flat_fee : ℝ) 
  (h1 : distance = 20)
  (h2 : speed = 10)
  (h3 : hourly_rate = 30)
  (h4 : flat_fee = 20)
  : (distance / speed) * hourly_rate + flat_fee = 80 :=
by
  sorry

end carriage_hire_cost_l2368_236899


namespace sphere_containment_l2368_236868

/-- A point in 3-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A sphere in 3-dimensional space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Predicate to check if a point is inside or on a sphere -/
def Point3D.inSphere (p : Point3D) (s : Sphere) : Prop :=
  (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 ≤ s.radius^2

/-- The main theorem -/
theorem sphere_containment (n : ℕ) (points : Fin n → Point3D) 
    (h : n ≥ 5)
    (h_four : ∀ (a b c d : Fin n), ∃ (s : Sphere), 
      s.radius = 1 ∧ 
      (points a).inSphere s ∧ 
      (points b).inSphere s ∧ 
      (points c).inSphere s ∧ 
      (points d).inSphere s) :
    ∃ (s : Sphere), s.radius = 1 ∧ ∀ (i : Fin n), (points i).inSphere s := by
  sorry

end sphere_containment_l2368_236868


namespace log3_of_9_cubed_l2368_236800

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_of_9_cubed : log3 (9^3) = 6 := by sorry

end log3_of_9_cubed_l2368_236800


namespace actual_distance_calculation_l2368_236839

-- Define the map scale
def map_scale : ℚ := 200

-- Define the measured distance on the map
def map_distance : ℚ := 9/2

-- Theorem to prove
theorem actual_distance_calculation :
  map_scale * map_distance / 100 = 9 := by
  sorry

end actual_distance_calculation_l2368_236839


namespace graduates_distribution_l2368_236818

def distribute_graduates (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem graduates_distribution :
  distribute_graduates 5 3 = 150 := by
  sorry

end graduates_distribution_l2368_236818


namespace root_equation_value_l2368_236804

theorem root_equation_value (m : ℝ) : m^2 - m - 1 = 0 → m^2 - m + 5 = 6 := by
  sorry

end root_equation_value_l2368_236804


namespace remove_horizontal_eliminates_triangles_l2368_236873

/-- Represents a triangular grid constructed with toothpicks -/
structure TriangularGrid where
  toothpicks : ℕ
  rows : ℕ
  columns : ℕ
  triangles : ℕ

/-- The specific triangular grid in the problem -/
def problemGrid : TriangularGrid :=
  { toothpicks := 36
  , rows := 3
  , columns := 5
  , triangles := 35 }

/-- The number of horizontal toothpicks in the grid -/
def horizontalToothpicks (grid : TriangularGrid) : ℕ := grid.rows * grid.columns

/-- Theorem stating that removing all horizontal toothpicks eliminates all triangles -/
theorem remove_horizontal_eliminates_triangles (grid : TriangularGrid) :
  horizontalToothpicks grid = 15 ∧
  horizontalToothpicks grid ≤ grid.toothpicks ∧
  grid.triangles > 35 →
  (∀ n : ℕ, n < 15 → ∃ t : ℕ, t > 0) ∧
  (∀ t : ℕ, t = 0) :=
sorry

#check remove_horizontal_eliminates_triangles problemGrid

end remove_horizontal_eliminates_triangles_l2368_236873


namespace cosine_value_on_unit_circle_l2368_236865

theorem cosine_value_on_unit_circle (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = 1/2 ∧ y = Real.sqrt 3 / 2) →
  Real.cos α = 1/2 := by
  sorry

end cosine_value_on_unit_circle_l2368_236865


namespace range_of_a_for_increasing_f_l2368_236805

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (4 - a/2) * x + 2 else a^x

/-- The theorem stating the range of a for which f is increasing on ℝ -/
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4 ≤ a ∧ a < 8) :=
sorry

end range_of_a_for_increasing_f_l2368_236805


namespace complex_sum_to_polar_l2368_236857

theorem complex_sum_to_polar : ∃ (r θ : ℝ), 
  12 * Complex.exp (3 * π * Complex.I / 13) + 12 * Complex.exp (7 * π * Complex.I / 26) = 
  r * Complex.exp (θ * Complex.I) ∧ 
  r = 12 * Real.sqrt (2 + Real.sqrt 2) ∧ 
  θ = 3.25 * π / 13 := by
  sorry

end complex_sum_to_polar_l2368_236857


namespace division_problem_l2368_236842

theorem division_problem (n : ℤ) : 
  (n / 20 = 15) ∧ (n % 20 = 6) → n = 306 := by
  sorry

end division_problem_l2368_236842


namespace median_and_altitude_length_l2368_236890

/-- An isosceles triangle DEF with DE = DF = 10 and EF = 12 -/
structure IsoscelesTriangle where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- DE equals DF -/
  de_eq_df : de = df
  /-- DE equals 10 -/
  de_eq_ten : de = 10
  /-- EF equals 12 -/
  ef_eq_twelve : ef = 12

/-- The median DM from vertex D to side EF in the isosceles triangle -/
def median (t : IsoscelesTriangle) : ℝ := sorry

/-- The altitude DH from vertex D to side EF in the isosceles triangle -/
def altitude (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: Both the median and altitude have length 8 -/
theorem median_and_altitude_length (t : IsoscelesTriangle) : 
  median t = 8 ∧ altitude t = 8 := by sorry

end median_and_altitude_length_l2368_236890


namespace solution_set_quadratic_inequality_l2368_236867

theorem solution_set_quadratic_inequality :
  let S := {x : ℝ | 2 * x^2 - x - 3 ≥ 0}
  S = {x : ℝ | x ≤ -1 ∨ x ≥ 3/2} := by sorry

end solution_set_quadratic_inequality_l2368_236867


namespace eighth_term_value_l2368_236803

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem eighth_term_value 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 3 + 3 * a 8 + a 13 = 120) : 
  a 8 = 24 := by
sorry

end eighth_term_value_l2368_236803


namespace inequality_relationship_l2368_236891

theorem inequality_relationship (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end inequality_relationship_l2368_236891


namespace maia_remaining_requests_l2368_236815

/-- Calculates the number of remaining client requests after a given number of days -/
def remaining_requests (daily_intake : ℕ) (daily_completion : ℕ) (days : ℕ) : ℕ :=
  (daily_intake - daily_completion) * days

/-- Theorem: Given Maia's work pattern, she will have 10 remaining requests after 5 days -/
theorem maia_remaining_requests :
  remaining_requests 6 4 5 = 10 := by
  sorry

end maia_remaining_requests_l2368_236815


namespace vector_magnitude_l2368_236883

theorem vector_magnitude (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![m, 2]
  (∃ (k : ℝ), b = k • a) → 
  ‖a + 2 • b‖ = 3 * Real.sqrt 5 := by
sorry

end vector_magnitude_l2368_236883


namespace polynomial_product_expansion_l2368_236892

theorem polynomial_product_expansion (x : ℝ) :
  (1 + x + x^3) * (1 - x^4) = 1 + x + x^3 - x^4 - x^5 - x^7 := by
  sorry

end polynomial_product_expansion_l2368_236892


namespace student_weight_loss_l2368_236845

theorem student_weight_loss (student_weight sister_weight : ℝ) 
  (h1 : student_weight = 90)
  (h2 : student_weight + sister_weight = 132) : 
  ∃ (weight_loss : ℝ), 
    weight_loss = 6 ∧ 
    student_weight - weight_loss = 2 * sister_weight :=
by sorry

end student_weight_loss_l2368_236845


namespace dodecagon_arrangement_impossible_l2368_236864

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def arrangement := Fin 12 → Fin 12

def valid_arrangement (a : arrangement) : Prop :=
  ∀ i : Fin 12, ∃ j : Fin 12, a j = i

def adjacent_sum_prime (a : arrangement) : Prop :=
  ∀ i : Fin 12, is_prime ((a i).val + 1 + (a ((i + 1) % 12)).val + 1)

def skip_two_sum_prime (a : arrangement) : Prop :=
  ∀ i : Fin 12, is_prime ((a i).val + 1 + (a ((i + 3) % 12)).val + 1)

theorem dodecagon_arrangement_impossible :
  ¬∃ a : arrangement, valid_arrangement a ∧ adjacent_sum_prime a ∧ skip_two_sum_prime a :=
sorry

end dodecagon_arrangement_impossible_l2368_236864


namespace fraction_simplification_l2368_236859

theorem fraction_simplification 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h_sum : a^2 + b^2 + c^2 ≠ 0) :
  (a^2*b^2 + 2*a^2*b*c + a^2*c^2 - b^4) / (a^4 - b^2*c^2 + 2*a*b*c^2 + c^4) = 
  ((a*b+a*c+b^2)*(a*b+a*c-b^2)) / ((a^2 + b^2 - c^2)*(a^2 - b^2 + c^2)) :=
by sorry

end fraction_simplification_l2368_236859


namespace no_valid_p_exists_l2368_236810

theorem no_valid_p_exists (p M : ℝ) (hp : 0 < p) (hM : 0 < M) (hp2 : p < 2) : 
  ¬∃ p, M * (1 + p / 100) * (1 - 50 * p / 100) > M :=
by
  sorry

end no_valid_p_exists_l2368_236810


namespace complex_point_first_quadrant_l2368_236881

theorem complex_point_first_quadrant : 
  let z : ℂ := 1 / Complex.I
  (z^3 + 1).re > 0 ∧ (z^3 + 1).im > 0 := by sorry

end complex_point_first_quadrant_l2368_236881


namespace x_value_l2368_236811

theorem x_value (x : ℝ) : x = 80 * (1 + 13 / 100) → x = 90.4 := by
  sorry

end x_value_l2368_236811


namespace percentage_difference_l2368_236856

theorem percentage_difference : (0.6 * 50) - (0.45 * 30) = 16.5 := by
  sorry

end percentage_difference_l2368_236856


namespace product_inequality_l2368_236895

theorem product_inequality (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hab : a * b > 1) (hn : n ≥ 2) :
  (a + b)^n > a^n + b^n + 2^n - 2 := by
  sorry

end product_inequality_l2368_236895


namespace parabola_shift_l2368_236888

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 7

/-- The reference parabola function -/
def g (x : ℝ) : ℝ := x^2

/-- The shifted reference parabola function -/
def h (x : ℝ) : ℝ := g (x + 3) - 2

theorem parabola_shift :
  ∀ x : ℝ, f x = h x :=
sorry

end parabola_shift_l2368_236888


namespace bakery_roll_combinations_l2368_236814

theorem bakery_roll_combinations :
  let total_rolls : ℕ := 9
  let fixed_rolls : ℕ := 6
  let remaining_rolls : ℕ := total_rolls - fixed_rolls
  let num_types : ℕ := 4
  (Nat.choose (remaining_rolls + num_types - 1) (num_types - 1)) = 20 := by
  sorry

end bakery_roll_combinations_l2368_236814


namespace base_conversion_equality_l2368_236812

def base_8_to_10 (n : ℕ) : ℕ := sorry

def base_10_to_7 (n : ℕ) : ℕ := sorry

theorem base_conversion_equality :
  base_10_to_7 (base_8_to_10 5314) = 11026 := by sorry

end base_conversion_equality_l2368_236812


namespace hypotenuse_sum_of_two_triangles_l2368_236847

theorem hypotenuse_sum_of_two_triangles : 
  let triangle1_leg1 : ℝ := 120
  let triangle1_leg2 : ℝ := 160
  let triangle2_leg1 : ℝ := 30
  let triangle2_leg2 : ℝ := 40
  let hypotenuse1 := Real.sqrt (triangle1_leg1^2 + triangle1_leg2^2)
  let hypotenuse2 := Real.sqrt (triangle2_leg1^2 + triangle2_leg2^2)
  hypotenuse1 + hypotenuse2 = 250 := by
  sorry

end hypotenuse_sum_of_two_triangles_l2368_236847


namespace solve_table_height_l2368_236850

def table_height_problem (initial_measurement : ℝ) (rearranged_measurement : ℝ) 
  (block_width : ℝ) (table_thickness : ℝ) : Prop :=
  ∃ (h l : ℝ),
    l + h - block_width + table_thickness = initial_measurement ∧
    block_width + h - l + table_thickness = rearranged_measurement ∧
    h = 33

theorem solve_table_height :
  table_height_problem 40 34 6 4 := by
  sorry

end solve_table_height_l2368_236850


namespace second_race_lead_l2368_236889

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

theorem second_race_lead (h d : ℝ) (first_race second_race : Race) :
  h > 0 ∧ d > 0 ∧
  first_race.distance = h ∧
  second_race.distance = h ∧
  first_race.runner_a = second_race.runner_a ∧
  first_race.runner_b = second_race.runner_b ∧
  first_race.runner_a.speed * h = first_race.runner_b.speed * (h - 2 * d) →
  let finish_time := (h + 2 * d) / first_race.runner_a.speed
  finish_time * first_race.runner_a.speed - finish_time * first_race.runner_b.speed = 4 * d^2 / h :=
by sorry

end second_race_lead_l2368_236889


namespace prices_and_min_cost_l2368_236825

/-- Represents the price of a thermometer in yuan -/
def thermometer_price : ℝ := sorry

/-- Represents the price of a barrel of disinfectant in yuan -/
def disinfectant_price : ℝ := sorry

/-- The total cost of 4 thermometers and 2 barrels of disinfectant is 400 yuan -/
axiom equation1 : 4 * thermometer_price + 2 * disinfectant_price = 400

/-- The total cost of 2 thermometers and 4 barrels of disinfectant is 320 yuan -/
axiom equation2 : 2 * thermometer_price + 4 * disinfectant_price = 320

/-- The total number of items to be purchased -/
def total_items : ℕ := 80

/-- The constraint that the number of thermometers is no less than 1/4 of the number of disinfectant -/
def constraint (m : ℕ) : Prop := m ≥ (total_items - m) / 4

/-- The cost function for m thermometers and (80 - m) barrels of disinfectant -/
def cost (m : ℕ) : ℝ := thermometer_price * m + disinfectant_price * (total_items - m)

/-- The theorem stating the unit prices and minimum cost -/
theorem prices_and_min_cost :
  thermometer_price = 80 ∧
  disinfectant_price = 40 ∧
  ∃ m : ℕ, constraint m ∧ cost m = 3840 ∧ ∀ n : ℕ, constraint n → cost m ≤ cost n :=
sorry

end prices_and_min_cost_l2368_236825


namespace not_always_swap_cities_l2368_236853

/-- A graph representing cities and their railroad connections. -/
structure CityGraph where
  V : Type
  E : V → V → Prop

/-- A bijective function representing a renaming of cities. -/
def IsRenaming (G : CityGraph) (f : G.V → G.V) : Prop :=
  Function.Bijective f

/-- A renaming that preserves the graph structure (i.e., a graph isomorphism). -/
def IsValidRenaming (G : CityGraph) (f : G.V → G.V) : Prop :=
  IsRenaming G f ∧ ∀ u v : G.V, G.E u v ↔ G.E (f u) (f v)

/-- For any two cities, there exists a valid renaming that maps one to the other. -/
axiom any_city_can_be_renamed (G : CityGraph) :
  ∀ u v : G.V, ∃ f : G.V → G.V, IsValidRenaming G f ∧ f u = v

/-- The theorem to be proved. -/
theorem not_always_swap_cities (G : CityGraph) :
  ¬(∀ x y : G.V, ∃ f : G.V → G.V, IsValidRenaming G f ∧ f x = y ∧ f y = x) :=
sorry

end not_always_swap_cities_l2368_236853


namespace sam_sticker_spending_l2368_236848

/-- Given Sam's initial penny count and his spending on toys and candy, 
    calculate the amount spent on stickers. -/
theorem sam_sticker_spending 
  (total : ℕ) 
  (toy_cost : ℕ) 
  (candy_cost : ℕ) 
  (h1 : total = 2476) 
  (h2 : toy_cost = 1145) 
  (h3 : candy_cost = 781) :
  total - (toy_cost + candy_cost) = 550 := by
  sorry

#check sam_sticker_spending

end sam_sticker_spending_l2368_236848


namespace units_digit_of_7_power_l2368_236819

theorem units_digit_of_7_power : 7^(100^6) % 10 = 1 := by sorry

end units_digit_of_7_power_l2368_236819


namespace largest_difference_l2368_236828

def A : ℕ := 3 * 1003^1004
def B : ℕ := 1003^1004
def C : ℕ := 1002 * 1003^1003
def D : ℕ := 3 * 1003^1003
def E : ℕ := 1003^1003
def F : ℕ := 1003^1002

def P : ℕ := A - B
def Q : ℕ := B - C
def R : ℕ := C - D
def S : ℕ := D - E
def T : ℕ := E - F

theorem largest_difference :
  P > max Q (max R (max S T)) := by sorry

end largest_difference_l2368_236828


namespace ball_cost_price_l2368_236830

theorem ball_cost_price (cost : ℕ → ℝ) (h1 : cost 11 - 720 = cost 5) : cost 1 = 120 := by
  sorry

end ball_cost_price_l2368_236830


namespace boxes_in_case_l2368_236896

/-- Proves the number of boxes in a case given the total boxes, eggs per box, and total eggs -/
theorem boxes_in_case 
  (total_boxes : ℕ) 
  (eggs_per_box : ℕ) 
  (total_eggs : ℕ) 
  (h1 : total_boxes = 5)
  (h2 : eggs_per_box = 3)
  (h3 : total_eggs = 15)
  (h4 : total_eggs = total_boxes * eggs_per_box) :
  total_boxes = 5 := by
  sorry

end boxes_in_case_l2368_236896


namespace violet_buddy_hiking_time_l2368_236852

/-- Represents the hiking scenario of Violet and Buddy -/
structure HikingScenario where
  violet_water_rate : Real  -- ml per hour
  buddy_water_rate : Real   -- ml per hour
  violet_capacity : Real    -- L
  buddy_capacity : Real     -- L
  hiking_speed : Real       -- km/h
  break_interval : Real     -- hours
  break_duration : Real     -- hours

/-- Calculates the total time Violet and Buddy can spend on the trail before running out of water -/
def total_trail_time (scenario : HikingScenario) : Real :=
  sorry

/-- Theorem stating that Violet and Buddy can spend 6.25 hours on the trail before running out of water -/
theorem violet_buddy_hiking_time :
  let scenario : HikingScenario := {
    violet_water_rate := 800,
    buddy_water_rate := 400,
    violet_capacity := 4.8,
    buddy_capacity := 1.5,
    hiking_speed := 4,
    break_interval := 2,
    break_duration := 0.5
  }
  total_trail_time scenario = 6.25 := by
  sorry

end violet_buddy_hiking_time_l2368_236852


namespace hyperbola_asymptotes_l2368_236854

/-- Given a hyperbola and a point P on it forming a unit-area parallelogram with the origin and points on the asymptotes, prove the equations of the asymptotes. -/
theorem hyperbola_asymptotes (a : ℝ) (h_a : a > 0) (P : ℝ × ℝ) :
  (P.1^2 / a^2 - P.2^2 = 1) →  -- P is on the hyperbola
  (∃ (A B : ℝ × ℝ), 
    (A.2 = (A.1 / a)) ∧  -- A is on one asymptote
    (B.2 = -(B.1 / a)) ∧  -- B is on the other asymptote
    (P.2 - A.2 = (P.1 - A.1) / a) ∧  -- PA is parallel to one asymptote
    (P.2 - B.2 = -(P.1 - B.1) / a) ∧  -- PB is parallel to the other asymptote
    (abs ((A.1 - 0) * (P.2 - 0) - (A.2 - 0) * (P.1 - 0)) = 1)  -- Area of OBPA is 1
  ) →
  a = 2 :=
sorry

end hyperbola_asymptotes_l2368_236854


namespace survey_respondents_count_l2368_236879

theorem survey_respondents_count :
  let brand_x_count : ℕ := 360
  let brand_x_to_y_ratio : ℚ := 9 / 1
  let total_respondents : ℕ := brand_x_count + (brand_x_count / brand_x_to_y_ratio.num * brand_x_to_y_ratio.den).toNat
  total_respondents = 400 :=
by sorry

end survey_respondents_count_l2368_236879


namespace fraction_difference_l2368_236862

theorem fraction_difference (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : m^2 - n^2 = m*n) : 
  n/m - m/n = -1 := by
  sorry

end fraction_difference_l2368_236862


namespace discount_percentage_calculation_l2368_236844

theorem discount_percentage_calculation (MP : ℝ) (h1 : MP > 0) : 
  let CP := 0.36 * MP
  let gain_percent := 122.22222222222223
  let SP := CP * (1 + gain_percent / 100)
  (MP - SP) / MP * 100 = 20 := by sorry

end discount_percentage_calculation_l2368_236844


namespace frog_jump_probability_l2368_236831

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the rectangle boundary -/
def Rectangle := {p : Point | p.x = 0 ∨ p.x = 5 ∨ p.y = 0 ∨ p.y = 5}

/-- Represents a vertical side of the rectangle -/
def VerticalSide := {p : Point | p.x = 0 ∨ p.x = 5}

/-- The probability of ending on a vertical side starting from a given point -/
def probabilityVerticalSide (p : Point) : ℚ := sorry

/-- The frog's starting point -/
def startPoint : Point := ⟨2, 3⟩

/-- Theorem stating the probability of ending on a vertical side -/
theorem frog_jump_probability : probabilityVerticalSide startPoint = 2/3 := by sorry

end frog_jump_probability_l2368_236831


namespace equation_roots_l2368_236874

theorem equation_roots : 
  let f (x : ℝ) := 18 / (x^2 - 9) - 3 / (x - 3) - 2
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -4.5 := by
sorry

end equation_roots_l2368_236874


namespace cube_sum_gt_product_sum_l2368_236882

theorem cube_sum_gt_product_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
sorry

end cube_sum_gt_product_sum_l2368_236882


namespace arccos_difference_equals_negative_pi_over_six_l2368_236877

theorem arccos_difference_equals_negative_pi_over_six :
  Real.arccos ((Real.sqrt 6 + 1) / (2 * Real.sqrt 3)) - Real.arccos (Real.sqrt (2/3)) = -π/6 := by
  sorry

end arccos_difference_equals_negative_pi_over_six_l2368_236877


namespace complex_fraction_equality_complex_fraction_equality_proof_l2368_236821

theorem complex_fraction_equality : Complex → Prop :=
  fun i => (3 : ℂ) / (1 - i)^2 = (3 / 2 : ℂ) * i

-- The proof is omitted
theorem complex_fraction_equality_proof : complex_fraction_equality Complex.I := by sorry

end complex_fraction_equality_complex_fraction_equality_proof_l2368_236821


namespace arithmetic_sequence_common_difference_l2368_236860

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  first_sum : a 1 + a 8 = 10
  second_sum : a 2 + a 9 = 18

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  common_difference seq = 4 := by
  sorry


end arithmetic_sequence_common_difference_l2368_236860


namespace outfits_count_l2368_236893

/-- The number of shirts available. -/
def num_shirts : ℕ := 6

/-- The number of ties available. -/
def num_ties : ℕ := 5

/-- The number of pants available. -/
def num_pants : ℕ := 4

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_ties * num_pants

/-- Theorem stating that the total number of outfits is 120. -/
theorem outfits_count : total_outfits = 120 := by
  sorry

end outfits_count_l2368_236893


namespace absolute_value_minus_self_nonnegative_l2368_236843

theorem absolute_value_minus_self_nonnegative (m : ℚ) : 0 ≤ |m| - m := by
  sorry

end absolute_value_minus_self_nonnegative_l2368_236843


namespace additional_money_needed_mrs_smith_shopping_l2368_236806

/-- Calculates the additional money needed for Mrs. Smith's shopping trip --/
theorem additional_money_needed (total_budget : ℚ) (dress_budget : ℚ) (shoe_budget : ℚ) (accessory_budget : ℚ)
  (increase_ratio : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_needed := (dress_budget + shoe_budget + accessory_budget) * (1 + increase_ratio)
  let discounted_total := total_needed * (1 - discount_rate)
  discounted_total - total_budget

/-- Proves that Mrs. Smith needs $95 more --/
theorem mrs_smith_shopping : 
  additional_money_needed 500 300 150 50 (2/5) (15/100) = 95 := by
  sorry

end additional_money_needed_mrs_smith_shopping_l2368_236806


namespace vertex_sum_is_ten_l2368_236872

/-- Two parabolas intersecting at two points -/
structure IntersectingParabolas where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h1 : -(3 - a)^2 + b = 6
  h2 : (3 - c)^2 + d = 6
  h3 : -(7 - a)^2 + b = 2
  h4 : (7 - c)^2 + d = 2

/-- The sum of x-coordinates of the vertices of two intersecting parabolas -/
def vertexSum (p : IntersectingParabolas) : ℝ := p.a + p.c

/-- Theorem: The sum of x-coordinates of the vertices of two intersecting parabolas is 10 -/
theorem vertex_sum_is_ten (p : IntersectingParabolas) : vertexSum p = 10 := by
  sorry

end vertex_sum_is_ten_l2368_236872


namespace circus_tickets_cost_l2368_236886

def adult_ticket_price : ℚ := 44
def child_ticket_price : ℚ := 28
def num_adults : ℕ := 2
def num_children : ℕ := 5
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 6

def total_cost : ℚ :=
  let total_tickets := num_adults + num_children
  let subtotal := num_adults * adult_ticket_price + num_children * child_ticket_price
  if total_tickets > discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

theorem circus_tickets_cost :
  total_cost = 205.2 := by sorry

end circus_tickets_cost_l2368_236886


namespace no_large_lattice_regular_ngon_l2368_236823

/-- A lattice point in the coordinate plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A regular n-gon in the coordinate plane -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → LatticePoint
  is_regular : ∀ (i j : Fin n), 
    (vertices i).x^2 + (vertices i).y^2 = (vertices j).x^2 + (vertices j).y^2

/-- There does not exist a regular n-gon with n ≥ 7 whose vertices are all lattice points -/
theorem no_large_lattice_regular_ngon :
  ∀ (n : ℕ), n ≥ 7 → ¬∃ (ngon : RegularNGon n), True :=
sorry

end no_large_lattice_regular_ngon_l2368_236823


namespace dodecagon_diagonals_from_one_vertex_l2368_236841

/-- A dodecagon is a polygon with 12 sides. -/
def Dodecagon : Nat := 12

/-- The number of diagonals that can be drawn from one vertex of a polygon with n sides. -/
def diagonalsFromOneVertex (n : Nat) : Nat := n - 3

theorem dodecagon_diagonals_from_one_vertex :
  diagonalsFromOneVertex Dodecagon = 9 := by
  sorry

end dodecagon_diagonals_from_one_vertex_l2368_236841


namespace athletes_same_first_digit_know_each_other_l2368_236876

/-- Represents an athlete with an assigned number -/
structure Athlete where
  id : Nat
  number : Nat

/-- Represents the relation of two athletes knowing each other -/
def knows (a b : Athlete) : Prop := sorry

/-- Returns the first digit of a natural number -/
def firstDigit (n : Nat) : Nat := sorry

/-- Theorem: Given 19100 athletes, where among any 12 athletes at least 2 know each other,
    there exist 2 athletes who know each other and whose assigned numbers start with the same digit -/
theorem athletes_same_first_digit_know_each_other 
  (athletes : Finset Athlete) 
  (h1 : athletes.card = 19100) 
  (h2 : ∀ s : Finset Athlete, s ⊆ athletes → s.card = 12 → 
        ∃ a b : Athlete, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ knows a b) :
  ∃ a b : Athlete, a ∈ athletes ∧ b ∈ athletes ∧ a ≠ b ∧ 
    knows a b ∧ firstDigit a.number = firstDigit b.number := by
  sorry

end athletes_same_first_digit_know_each_other_l2368_236876


namespace luncheon_absence_l2368_236835

/-- The number of people who didn't show up to a luncheon --/
def people_absent (invited : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) : ℕ :=
  invited - (table_capacity * tables_needed)

/-- Proof that 50 people didn't show up to the luncheon --/
theorem luncheon_absence : people_absent 68 3 6 = 50 := by
  sorry

end luncheon_absence_l2368_236835


namespace unique_valid_number_l2368_236851

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  n / 1000 = 764 ∧
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 764280 :=
sorry

end unique_valid_number_l2368_236851


namespace cube_face_area_l2368_236838

theorem cube_face_area (V : ℝ) (h : V = 125) : ∃ (A : ℝ), A = 25 ∧ A = (V ^ (1/3)) ^ 2 := by
  sorry

end cube_face_area_l2368_236838


namespace fraction_evaluation_l2368_236829

theorem fraction_evaluation : (((5 * 4) + 6) : ℝ) / 10 = 2.6 := by
  sorry

end fraction_evaluation_l2368_236829


namespace quadratic_equation_k_value_l2368_236866

theorem quadratic_equation_k_value :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := 3
  let k : ℝ := 16/3
  (4 * b^2 - k * a * c = 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 4 * b * x + c = 0 ∧ a * y^2 + 4 * b * y + c = 0) :=
by sorry

end quadratic_equation_k_value_l2368_236866


namespace count_divisible_integers_l2368_236813

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (2310 : ℤ) ∣ (m^2 - 2)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (2310 : ℤ) ∣ (m^2 - 2) → m ∈ S) ∧
    S.card = 3 :=
by sorry

end count_divisible_integers_l2368_236813


namespace snack_pack_suckers_l2368_236832

/-- The number of suckers needed for snack packs --/
def suckers_needed (pretzels : ℕ) (goldfish_multiplier : ℕ) (kids : ℕ) (items_per_baggie : ℕ) : ℕ :=
  kids * items_per_baggie - (pretzels + goldfish_multiplier * pretzels)

theorem snack_pack_suckers :
  suckers_needed 64 4 16 22 = 32 := by
  sorry

end snack_pack_suckers_l2368_236832


namespace hike_attendance_l2368_236880

/-- The number of people who went on the hike --/
def total_hikers (num_cars num_taxis num_vans : ℕ) 
                 (car_capacity taxi_capacity van_capacity : ℕ) : ℕ :=
  num_cars * car_capacity + num_taxis * taxi_capacity + num_vans * van_capacity

/-- Theorem stating that 58 people went on the hike --/
theorem hike_attendance : 
  total_hikers 3 6 2 4 6 5 = 58 := by
  sorry

end hike_attendance_l2368_236880


namespace simplify_and_evaluate_l2368_236863

theorem simplify_and_evaluate (a b : ℝ) (ha : a = 3) (hb : b = 2) :
  (a + b) / (a * b) / ((a / b) - (b / a)) = 1 := by
  sorry

end simplify_and_evaluate_l2368_236863


namespace geometric_sequence_seventh_term_l2368_236898

theorem geometric_sequence_seventh_term 
  (a : ℝ) (a₃ : ℝ) (n : ℕ) (h₁ : a = 3) (h₂ : a₃ = 3/64) (h₃ : n = 7) :
  a * (a₃ / a) ^ ((n - 1) / 2) = 3/262144 := by
  sorry

end geometric_sequence_seventh_term_l2368_236898


namespace quadratic_roots_condition_l2368_236808

theorem quadratic_roots_condition (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ →
  x₁^2 + (a - 1) * x₁ + 2 * a - 5 = 0 →
  x₂^2 + (a - 1) * x₂ + 2 * a - 5 = 0 →
  1 / x₁ + 1 / x₂ < -3 / 5 →
  a > 5 / 2 ∧ a < 10 :=
by sorry

end quadratic_roots_condition_l2368_236808


namespace mars_ticket_cost_after_30_years_l2368_236826

/-- The cost of a ticket to Mars after a given number of decades, 
    given an initial cost and a halving rate every decade. -/
def mars_ticket_cost (initial_cost : ℚ) (decades : ℕ) : ℚ :=
  initial_cost / (2 ^ decades)

/-- Theorem stating that the cost of a ticket to Mars after 3 decades
    is $125,000, given an initial cost of $1,000,000 and halving every decade. -/
theorem mars_ticket_cost_after_30_years 
  (initial_cost : ℚ) (h_initial : initial_cost = 1000000) :
  mars_ticket_cost initial_cost 3 = 125000 := by
  sorry

#eval mars_ticket_cost 1000000 3

end mars_ticket_cost_after_30_years_l2368_236826


namespace seven_lines_intersection_impossibility_l2368_236801

/-- The maximum number of intersections for n lines in a Euclidean plane -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of intersections required for a given number of triple and double intersections -/
def required_intersections (triple_points double_points : ℕ) : ℕ :=
  triple_points * 3 + double_points

theorem seven_lines_intersection_impossibility :
  let n_lines : ℕ := 7
  let min_triple_points : ℕ := 6
  let min_double_points : ℕ := 4
  required_intersections min_triple_points min_double_points > max_intersections n_lines := by
  sorry


end seven_lines_intersection_impossibility_l2368_236801


namespace min_value_a_l2368_236833

theorem min_value_a (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, -1 < x₀ ∧ x₀ ≤ 2 ∧ x₀ - a > 0) → 
  (∀ b : ℝ, (¬ ∃ x₀ : ℝ, -1 < x₀ ∧ x₀ ≤ 2 ∧ x₀ - b > 0) → a ≤ b) → 
  a = 2 :=
by sorry

end min_value_a_l2368_236833


namespace min_value_quadratic_l2368_236824

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 2 * x^2 - 8 * x + 10 → y ≥ min_y ∧ min_y = 2 := by
  sorry

end min_value_quadratic_l2368_236824


namespace total_birds_count_l2368_236820

/-- The number of geese in the marsh -/
def num_geese : ℕ := 58

/-- The number of ducks in the marsh -/
def num_ducks : ℕ := 37

/-- The total number of birds in the marsh -/
def total_birds : ℕ := num_geese + num_ducks

/-- Theorem: The total number of birds in the marsh is 95 -/
theorem total_birds_count : total_birds = 95 := by
  sorry

end total_birds_count_l2368_236820


namespace sqrt_sum_equality_l2368_236885

theorem sqrt_sum_equality : 
  (Real.sqrt 54 - Real.sqrt 27) + Real.sqrt 3 + 8 * Real.sqrt (1/2) = 
  3 * Real.sqrt 6 - 2 * Real.sqrt 3 + 4 * Real.sqrt 2 := by
  sorry

end sqrt_sum_equality_l2368_236885


namespace space_between_apple_trees_is_12_l2368_236870

/-- The space needed between apple trees in Quinton's backyard --/
def space_between_apple_trees : ℝ :=
  let total_space : ℝ := 71
  let apple_tree_width : ℝ := 10
  let peach_tree_width : ℝ := 12
  let space_between_peach_trees : ℝ := 15
  let num_apple_trees : ℕ := 2
  let num_peach_trees : ℕ := 2
  let peach_trees_space : ℝ := num_peach_trees * peach_tree_width + space_between_peach_trees
  let apple_trees_space : ℝ := total_space - peach_trees_space
  apple_trees_space - (num_apple_trees * apple_tree_width)

theorem space_between_apple_trees_is_12 :
  space_between_apple_trees = 12 := by
  sorry

end space_between_apple_trees_is_12_l2368_236870


namespace age_ratio_in_one_year_l2368_236822

/-- Represents the current ages of Jack and Alex -/
structure Ages where
  jack : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  (ages.jack - 3 = 2 * (ages.alex - 3)) ∧ 
  (ages.jack - 5 = 3 * (ages.alex - 5))

/-- The future ratio of their ages will be 3:2 -/
def future_ratio (ages : Ages) (years : ℕ) : Prop :=
  3 * (ages.alex + years) = 2 * (ages.jack + years)

/-- The theorem to be proved -/
theorem age_ratio_in_one_year (ages : Ages) :
  age_conditions ages → ∃ (y : ℕ), y = 1 ∧ future_ratio ages y :=
sorry

end age_ratio_in_one_year_l2368_236822


namespace organization_size_after_five_years_l2368_236875

def organization_growth (initial_members : ℕ) (leaders : ℕ) (recruitment : ℕ) (years : ℕ) : ℕ :=
  let rec growth (k : ℕ) (members : ℕ) : ℕ :=
    if k = 0 then
      members
    else
      growth (k - 1) (4 * members - 18)
  growth years initial_members

theorem organization_size_after_five_years :
  organization_growth 20 6 3 5 = 14382 :=
by sorry

end organization_size_after_five_years_l2368_236875


namespace arctan_sum_three_seven_l2368_236871

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end arctan_sum_three_seven_l2368_236871


namespace garden_area_l2368_236849

/-- Represents a rectangular garden with specific properties. -/
structure Garden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  lengthCondition : length = 3 * width + 10
  perimeterCondition : perimeter = 2 * (length + width)
  perimeterValue : perimeter = 400

/-- The area of a rectangular garden. -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the area of the garden with given conditions. -/
theorem garden_area (g : Garden) : g.area = 7243.75 := by
  sorry

end garden_area_l2368_236849


namespace A_power_98_l2368_236861

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_98 : A ^ 98 = !![0, 0, 0; 0, -1, 0; 0, 0, -1] := by sorry

end A_power_98_l2368_236861


namespace computer_sticker_price_l2368_236802

theorem computer_sticker_price : 
  ∀ (x : ℝ), 
  (0.80 * x - 80 = 0.70 * x - 40 - 30) → 
  x = 700 := by
sorry

end computer_sticker_price_l2368_236802


namespace sparrow_population_decline_l2368_236894

/-- Proves that the smallest integer t satisfying (0.6^t ≤ 0.05) is 6 -/
theorem sparrow_population_decline (t : ℕ) : 
  (∀ k : ℕ, k < t → (0.6 : ℝ)^k > 0.05) ∧ (0.6 : ℝ)^t ≤ 0.05 → t = 6 :=
by sorry

end sparrow_population_decline_l2368_236894


namespace power_of_three_squared_l2368_236840

theorem power_of_three_squared : 3^2 = 9 := by
  sorry

end power_of_three_squared_l2368_236840


namespace sufficient_not_necessary_and_necessary_not_sufficient_l2368_236817

theorem sufficient_not_necessary_and_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → 1 / a < 1) ∧
  (∃ a : ℝ, 1 / a < 1 ∧ a ≤ 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end sufficient_not_necessary_and_necessary_not_sufficient_l2368_236817


namespace profit_360_implies_price_increase_4_price_13_implies_profit_350_l2368_236809

/-- Represents the daily profit function for a company selling goods -/
def profit_function (x : ℕ) : ℤ := 10 * (x + 2) * (10 - x)

/-- Theorem stating that when the daily profit is 360 yuan, the selling price has increased by 4 yuan -/
theorem profit_360_implies_price_increase_4 :
  ∃ (x : ℕ), 0 ≤ x ∧ x ≤ 10 ∧ profit_function x = 360 → x = 4 := by
  sorry

/-- Theorem stating that when the selling price increases by 3 yuan (to 13 yuan), the profit is 350 yuan -/
theorem price_13_implies_profit_350 :
  profit_function 3 = 350 := by
  sorry

end profit_360_implies_price_increase_4_price_13_implies_profit_350_l2368_236809


namespace two_digit_number_difference_l2368_236807

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 81, the difference between its 
two digits is 9.
-/
theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 81 → x - y = 9 := by
  sorry

end two_digit_number_difference_l2368_236807


namespace ellipse_properties_l2368_236836

noncomputable def ellipse_C (x y a b : ℝ) : Prop :=
  (y^2 / a^2) + (x^2 / b^2) = 1

theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0)
  (h3 : (a^2 - b^2) / a^2 = 6/9)
  (h4 : ellipse_C (2*Real.sqrt 2/3) (Real.sqrt 3/3) a b) :
  (∃ (x y : ℝ), ellipse_C x y 1 (Real.sqrt 3)) ∧
  (∃ (S : ℝ → ℝ → ℝ), 
    (∀ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 1 (Real.sqrt 3) → 
      ellipse_C B.1 B.2 1 (Real.sqrt 3) → 
      (∃ m : ℝ, A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2) → 
      S A.1 A.2 ≤ Real.sqrt 3 / 2) ∧
    (∃ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 1 (Real.sqrt 3) ∧ 
      ellipse_C B.1 B.2 1 (Real.sqrt 3) ∧ 
      (∃ m : ℝ, A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2) ∧ 
      S A.1 A.2 = Real.sqrt 3 / 2)) := by
  sorry

end ellipse_properties_l2368_236836


namespace complex_magnitude_theorem_l2368_236834

theorem complex_magnitude_theorem (x y : ℝ) (z : ℂ) :
  z = x + y * Complex.I →
  (2 * x) / (1 - Complex.I) = 1 + y * Complex.I →
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_theorem_l2368_236834
