import Mathlib

namespace equidistant_point_x_coord_l2552_255298

/-- A point in the coordinate plane equally distant from the x-axis, y-axis, and the line x + y = 4 -/
structure EquidistantPoint where
  x : ℝ
  y : ℝ
  dist_x_axis : |y| = |x|
  dist_y_axis : |x| = |y|
  dist_line : |x + y - 4| / Real.sqrt 2 = |x|

/-- The x-coordinate of an equidistant point is 2 -/
theorem equidistant_point_x_coord (p : EquidistantPoint) : p.x = 2 := by
  sorry

#check equidistant_point_x_coord

end equidistant_point_x_coord_l2552_255298


namespace complex_magnitude_problem_l2552_255266

theorem complex_magnitude_problem (z : ℂ) (h : (3 + 4*Complex.I)*z = 2 + Complex.I) :
  Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end complex_magnitude_problem_l2552_255266


namespace sum_of_odd_symmetric_function_l2552_255215

-- Define an odd function with symmetry about x = 1/2
def is_odd_and_symmetric (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (1/2 + x) = f (1/2 - x))

-- Theorem statement
theorem sum_of_odd_symmetric_function (f : ℝ → ℝ) 
  (h : is_odd_and_symmetric f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end sum_of_odd_symmetric_function_l2552_255215


namespace simplify_absolute_value_l2552_255221

theorem simplify_absolute_value : |-4^2 + (6 - 2)| = 12 := by
  sorry

end simplify_absolute_value_l2552_255221


namespace zero_points_property_l2552_255223

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x

theorem zero_points_property (a : ℝ) :
  (∃ x₂ : ℝ, x₂ ≠ sqrt e ∧ f a (sqrt e) = 0 ∧ f a x₂ = 0) →
  a = sqrt e / (2 * e) ∧ ∀ x₂ : ℝ, x₂ ≠ sqrt e ∧ f a x₂ = 0 → x₂ > e^(3/2) :=
by sorry

end zero_points_property_l2552_255223


namespace zeros_of_f_l2552_255230

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 4 else x^2 - 4

-- Theorem statement
theorem zeros_of_f :
  (∃ x : ℝ, f x = 0) ↔ (x = -4 ∨ x = 2) :=
sorry

end zeros_of_f_l2552_255230


namespace grape_juice_concentration_l2552_255212

/-- Given an initial mixture and added grape juice, calculate the final grape juice concentration -/
theorem grape_juice_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_juice : ℝ) 
  (h1 : initial_volume = 40)
  (h2 : initial_concentration = 0.1)
  (h3 : added_juice = 10) : 
  (initial_volume * initial_concentration + added_juice) / (initial_volume + added_juice) = 0.28 := by
sorry

end grape_juice_concentration_l2552_255212


namespace f_and_g_properties_l2552_255252

-- Define the functions
def f (x : ℝ) : ℝ := 1 + x^2
def g (x : ℝ) : ℝ := |x| + 1

-- Define evenness
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

-- Define monotonically decreasing on (-∞, 0)
def is_decreasing_neg (h : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → h x > h y

theorem f_and_g_properties :
  (is_even f ∧ is_decreasing_neg f) ∧
  (is_even g ∧ is_decreasing_neg g) := by sorry

end f_and_g_properties_l2552_255252


namespace other_solution_of_quadratic_l2552_255205

theorem other_solution_of_quadratic (x : ℚ) : 
  (65 * (6/5)^2 + 18 = 104 * (6/5) - 13) →
  (65 * x^2 + 18 = 104 * x - 13) →
  (x ≠ 6/5) →
  x = 5/13 := by
sorry

end other_solution_of_quadratic_l2552_255205


namespace post_office_mail_theorem_l2552_255264

/-- Calculates the total number of pieces of mail handled by a post office in six months -/
def mail_in_six_months (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) : ℕ :=
  let total_per_day := letters_per_day + packages_per_day
  let total_per_month := total_per_day * days_per_month
  total_per_month * 6

/-- Theorem stating that a post office receiving 60 letters and 20 packages per day
    handles 14400 pieces of mail in six months, assuming 30-day months -/
theorem post_office_mail_theorem :
  mail_in_six_months 60 20 30 = 14400 := by
  sorry

#eval mail_in_six_months 60 20 30

end post_office_mail_theorem_l2552_255264


namespace fox_kolobok_meeting_l2552_255267

theorem fox_kolobok_meeting (n : ℕ) (m : ℕ) (h1 : n = 14) (h2 : m = 92) :
  ∃ (i j : ℕ) (f : ℕ → ℕ), i ≠ j ∧ i < n ∧ j < n ∧ f i = f j ∧ (∀ k < n, f k ≤ m) :=
by
  sorry

end fox_kolobok_meeting_l2552_255267


namespace det_scale_l2552_255204

theorem det_scale (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 10 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 90 := by
sorry

end det_scale_l2552_255204


namespace shaded_area_calculation_l2552_255202

-- Define the circle and square
def circle_radius : ℝ := 4
def square_side : ℝ := 2

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨square_side, 0⟩
def B : Point := ⟨square_side, square_side⟩
def C : Point := ⟨0, square_side⟩

-- Define the extended points D and E
def D : Point := sorry
def E : Point := sorry

-- Define the shaded area
def shaded_area : ℝ := sorry

-- Theorem statement
theorem shaded_area_calculation :
  shaded_area = (16 * π / 3) - 6 * Real.sqrt 3 + 4 := by
  sorry

end shaded_area_calculation_l2552_255202


namespace negation_of_existence_negation_of_proposition_l2552_255214

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 1, P x) ↔ (∀ x > 1, ¬ P x) :=
by
  sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 1 > 0) ↔ (∀ x > 1, x^2 - 1 ≤ 0) :=
by
  sorry

end negation_of_existence_negation_of_proposition_l2552_255214


namespace remainder_theorem_l2552_255275

theorem remainder_theorem :
  (7 * 10^20 + 3^20) % 9 = 7 := by sorry

end remainder_theorem_l2552_255275


namespace initial_people_count_l2552_255241

/-- The number of people who left the table -/
def people_left : ℕ := 6

/-- The number of people who remained at the table -/
def people_remained : ℕ := 5

/-- The initial number of people at the table -/
def initial_people : ℕ := people_left + people_remained

theorem initial_people_count : initial_people = 11 := by
  sorry

end initial_people_count_l2552_255241


namespace weeklyRentIs1200_l2552_255256

/-- Calculates the weekly rent for a flower shop given the following conditions:
  * Utilities cost is 20% of rent
  * 2 employees per shift
  * Store open 16 hours a day for 5 days a week
  * Employee pay is $12.50 per hour
  * Total weekly expenses are $3440
-/
def calculateWeeklyRent (totalExpenses : ℚ) (employeePay : ℚ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) (employeesPerShift : ℕ) : ℚ :=
  let totalHours : ℕ := hoursPerDay * daysPerWeek * employeesPerShift
  let weeklyWages : ℚ := employeePay * totalHours
  (totalExpenses - weeklyWages) / 1.2

/-- Proves that the weekly rent for the flower shop is $1200 -/
theorem weeklyRentIs1200 :
  calculateWeeklyRent 3440 12.5 16 5 2 = 1200 := by
  sorry

end weeklyRentIs1200_l2552_255256


namespace peter_completes_work_in_35_days_l2552_255273

/-- The number of days Matt and Peter take to complete the work together -/
def total_days_together : ℚ := 20

/-- The number of days Matt and Peter work together before Matt stops -/
def days_worked_together : ℚ := 12

/-- The number of days Peter takes to complete the remaining work after Matt stops -/
def peter_remaining_days : ℚ := 14

/-- The fraction of work completed when Matt and Peter work together for 12 days -/
def work_completed_together : ℚ := days_worked_together / total_days_together

/-- The fraction of work Peter completes after Matt stops -/
def peter_remaining_work : ℚ := 1 - work_completed_together

/-- Peter's work rate (fraction of work completed per day) -/
def peter_work_rate : ℚ := peter_remaining_work / peter_remaining_days

/-- The number of days Peter takes to complete the work separately -/
def peter_total_days : ℚ := 1 / peter_work_rate

theorem peter_completes_work_in_35_days :
  peter_total_days = 35 := by sorry

end peter_completes_work_in_35_days_l2552_255273


namespace inequality_proof_l2552_255238

theorem inequality_proof (x b : ℝ) (h1 : x < b) (h2 : b < 0) (h3 : b = -2) :
  x^2 > b*x ∧ b*x > b^2 := by
  sorry

end inequality_proof_l2552_255238


namespace product_mod_23_is_zero_l2552_255271

theorem product_mod_23_is_zero :
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 0 := by
  sorry

end product_mod_23_is_zero_l2552_255271


namespace solution_set_when_a_is_2_range_of_a_l2552_255209

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Theorem for part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Theorem for part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end solution_set_when_a_is_2_range_of_a_l2552_255209


namespace inverse_proposition_false_l2552_255282

theorem inverse_proposition_false (a b : ℝ) (h : a < b) :
  ∃ (f : ℝ → ℝ), Continuous f ∧ 
  (∃ x ∈ Set.Ioo a b, f x = 0) ∧ 
  f a * f b ≥ 0 := by
  sorry

end inverse_proposition_false_l2552_255282


namespace walk_a_thon_earnings_l2552_255218

theorem walk_a_thon_earnings (last_year_rate : ℚ) (last_year_total : ℚ) 
  (extra_miles : ℕ) (this_year_rate : ℚ) : 
  last_year_rate = 4 →
  last_year_total = 44 →
  extra_miles = 5 →
  (last_year_total / last_year_rate + extra_miles) * this_year_rate = last_year_total →
  this_year_rate = 11/4 := by
sorry

#eval (11 : ℚ) / 4  -- To show the decimal representation

end walk_a_thon_earnings_l2552_255218


namespace quadrangular_pyramid_edge_sum_l2552_255269

/-- Represents a hexagonal prism -/
structure HexagonalPrism where
  edge_length : ℝ
  total_edge_length : ℝ
  edges_equal : edge_length > 0
  total_length_constraint : total_edge_length = 18 * edge_length

/-- Represents a quadrangular pyramid -/
structure QuadrangularPyramid where
  edge_length : ℝ
  edges_equal : edge_length > 0

/-- Theorem stating the relationship between hexagonal prism and quadrangular pyramid edge lengths -/
theorem quadrangular_pyramid_edge_sum 
  (h : HexagonalPrism) 
  (q : QuadrangularPyramid) 
  (edge_equality : q.edge_length = h.edge_length) :
  8 * q.edge_length = 36 := by
  sorry

end quadrangular_pyramid_edge_sum_l2552_255269


namespace pythagorean_theorem_isosceles_right_l2552_255207

/-- An isosceles right triangle with legs of unit length -/
structure IsoscelesRightTriangle where
  /-- The length of each leg is 1 -/
  leg : ℝ
  leg_eq_one : leg = 1

/-- The Pythagorean theorem for an isosceles right triangle -/
theorem pythagorean_theorem_isosceles_right (t : IsoscelesRightTriangle) :
  t.leg ^ 2 + t.leg ^ 2 = (Real.sqrt 2) ^ 2 := by
  sorry

#check pythagorean_theorem_isosceles_right

end pythagorean_theorem_isosceles_right_l2552_255207


namespace system_solution_l2552_255216

theorem system_solution (a x y : ℝ) : 
  (x / 2 - (2 * x - 3 * y) / 5 = a - 1) →
  (x + 3 = y / 3) →
  (x < 0 ∧ y > 0) ↔ (7/10 < a ∧ a < 64/10) :=
by sorry

end system_solution_l2552_255216


namespace dollar_square_sum_l2552_255288

/-- The dollar operation -/
def dollar (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the result of (x + y)²$(y + x)² -/
theorem dollar_square_sum (x y : ℝ) : 
  dollar ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end dollar_square_sum_l2552_255288


namespace sufficient_not_necessary_l2552_255258

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) := by
  sorry

end sufficient_not_necessary_l2552_255258


namespace positive_integer_inequality_l2552_255263

theorem positive_integer_inequality (a b c : ℕ+) 
  (h : (a : ℤ) + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end positive_integer_inequality_l2552_255263


namespace circle_center_l2552_255208

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_center (x y : ℝ) :
  (3 * x + 4 * y = 24 ∨ 3 * x + 4 * y = -6) →  -- Circle is tangent to these lines
  (3 * x - y = 0) →                           -- Center lies on this line
  (x = 3/5 ∧ y = 9/5) →                       -- Proposed center coordinates
  ∃ (r : ℝ), r > 0 ∧                          -- There exists a positive radius
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →  -- Points on the circle
      (3 * x' + 4 * y' = 24 ∨ 3 * x' + 4 * y' = -6))  -- Touch the given lines
  := by sorry


end circle_center_l2552_255208


namespace parabola_points_x_coordinate_l2552_255255

/-- The x-coordinate of points on the parabola y^2 = 12x with distance 8 from the focus -/
theorem parabola_points_x_coordinate (x y : ℝ) : 
  y^2 = 12*x →                             -- Point (x,y) is on the parabola
  (x - 3)^2 + y^2 = 64 →                   -- Distance from (x,y) to focus (3,0) is 8
  x = 5 := by
sorry

end parabola_points_x_coordinate_l2552_255255


namespace hilt_garden_border_l2552_255228

/-- The number of rocks in Mrs. Hilt's completed garden border -/
def total_rocks : ℕ := 189

/-- The number of additional rocks Mrs. Hilt has yet to place -/
def remaining_rocks : ℕ := 64

/-- The number of rocks Mrs. Hilt has already placed -/
def placed_rocks : ℕ := total_rocks - remaining_rocks

theorem hilt_garden_border : placed_rocks = 125 := by
  sorry

end hilt_garden_border_l2552_255228


namespace city_graph_property_l2552_255287

/-- A graph representing cities and flights -/
structure CityGraph where
  V : Type*
  E : V → V → Prop
  N : Nat
  vertex_count : Fintype V
  city_count : Fintype.card V = N

/-- Path of length at most 2 between two vertices -/
def PathOfLength2 (G : CityGraph) (u v : G.V) : Prop :=
  G.E u v ∨ ∃ w, G.E u w ∧ G.E w v

/-- The main theorem -/
theorem city_graph_property (G : CityGraph) 
  (not_fully_connected : ∀ v : G.V, ∃ u : G.V, ¬G.E v u)
  (unique_path : ∀ u v : G.V, ∃! p : PathOfLength2 G u v, True) :
  ∃ k : Nat, G.N - 1 = k * k :=
sorry

end city_graph_property_l2552_255287


namespace cookie_recipe_l2552_255268

-- Define the conversion rates
def quart_to_pint : ℚ := 2
def pint_to_cup : ℚ := 1/4

-- Define the recipe for 24 cookies
def milk_for_24 : ℚ := 4  -- in quarts
def sugar_for_24 : ℚ := 6  -- in cups

-- Define the number of cookies we want to bake
def cookies_to_bake : ℚ := 6

-- Define the scaling factor
def scaling_factor : ℚ := cookies_to_bake / 24

-- Theorem to prove
theorem cookie_recipe :
  (milk_for_24 * quart_to_pint * scaling_factor = 2) ∧
  (sugar_for_24 * scaling_factor = 1.5) := by
  sorry


end cookie_recipe_l2552_255268


namespace intersection_complement_equal_l2552_255234

def U : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,4,6}
def B : Finset Nat := {2,4,5,6}

theorem intersection_complement_equal : A ∩ (U \ B) = {1,3} := by
  sorry

end intersection_complement_equal_l2552_255234


namespace midpoint_coordinate_product_l2552_255240

/-- Given that M(3,8) is the midpoint of line segment AB and A(5,6) is one endpoint,
    prove that the product of the coordinates of point B is 10. -/
theorem midpoint_coordinate_product (A B M : ℝ × ℝ) : 
  A = (5, 6) → M = (3, 8) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 * B.2 = 10 := by sorry

end midpoint_coordinate_product_l2552_255240


namespace divisibility_problem_l2552_255259

theorem divisibility_problem (x y : ℤ) 
  (hx : x ≠ -1) (hy : y ≠ -1) 
  (h : ∃ k : ℤ, (x^4 - 1) / (y + 1) + (y^4 - 1) / (x + 1) = k) : 
  ∃ m : ℤ, x^4 * y^44 - 1 = m * (x + 1) := by
  sorry

end divisibility_problem_l2552_255259


namespace sum_of_two_numbers_l2552_255224

theorem sum_of_two_numbers (a b : ℝ) : 
  a + b = 25 → a * b = 144 → |a - b| = 7 → a + b = 25 := by
  sorry

end sum_of_two_numbers_l2552_255224


namespace chloe_apples_l2552_255280

theorem chloe_apples (chloe_apples dylan_apples : ℕ) : 
  chloe_apples = dylan_apples + 8 →
  dylan_apples = chloe_apples / 3 →
  chloe_apples = 12 := by
sorry

end chloe_apples_l2552_255280


namespace currency_and_unit_comparisons_l2552_255219

-- Define the conversion rates
def yuan_to_jiao : ℚ → ℚ := (· * 10)
def dm_to_cm : ℚ → ℚ := (· * 10)
def hectare_to_m2 : ℚ → ℚ := (· * 10000)
def km2_to_hectare : ℚ → ℚ := (· * 100)

-- Define the theorem
theorem currency_and_unit_comparisons :
  (7 > 5.70) ∧
  (70 > 7) ∧
  (80000 > 70000) ∧
  (1 = 1) ∧
  (34 * 6 * 2 = 34 * 12) ∧
  (3.9 = 3.9) := by
  sorry

end currency_and_unit_comparisons_l2552_255219


namespace six_couples_handshakes_l2552_255281

/-- Represents the number of handshakes in a gathering of couples -/
def handshakes_in_gathering (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 6 couples, where each person shakes hands with
    everyone except their spouse and one other person, there are 54 handshakes -/
theorem six_couples_handshakes :
  handshakes_in_gathering 6 = 54 := by
  sorry

end six_couples_handshakes_l2552_255281


namespace complex_coordinates_l2552_255260

theorem complex_coordinates (z : ℂ) (h : z = Complex.I * (2 + 4 * Complex.I)) : 
  z.re = -4 ∧ z.im = 2 := by
  sorry

end complex_coordinates_l2552_255260


namespace square_area_adjacent_vertices_l2552_255229

/-- The area of a square with adjacent vertices at (-2,3) and (4,3) is 36. -/
theorem square_area_adjacent_vertices : 
  let p1 : ℝ × ℝ := (-2, 3)
  let p2 : ℝ × ℝ := (4, 3)
  let distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := distance^2
  area = 36 := by
  sorry

end square_area_adjacent_vertices_l2552_255229


namespace intersection_with_complement_l2552_255201

universe u

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {4, 5}

theorem intersection_with_complement :
  A ∩ (U \ B) = {2, 3} := by sorry

end intersection_with_complement_l2552_255201


namespace diane_gingerbreads_l2552_255257

/-- The number of trays with 25 gingerbreads each -/
def trays_25 : ℕ := 4

/-- The number of gingerbreads in each of the 25-gingerbread trays -/
def gingerbreads_per_tray_25 : ℕ := 25

/-- The number of trays with 20 gingerbreads each -/
def trays_20 : ℕ := 3

/-- The number of gingerbreads in each of the 20-gingerbread trays -/
def gingerbreads_per_tray_20 : ℕ := 20

/-- The total number of gingerbreads Diane bakes -/
def total_gingerbreads : ℕ := trays_25 * gingerbreads_per_tray_25 + trays_20 * gingerbreads_per_tray_20

theorem diane_gingerbreads : total_gingerbreads = 160 := by
  sorry

end diane_gingerbreads_l2552_255257


namespace min_value_geometric_sequence_l2552_255244

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) : 
  b₁ = 3 → 
  b₂ = b₁ * s → 
  b₃ = b₂ * s → 
  ∀ x : ℝ, 3 * b₂ + 7 * b₃ ≥ -18/7 :=
by sorry

end min_value_geometric_sequence_l2552_255244


namespace min_value_theorem_l2552_255291

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_f1 : f a b 1 = 2) :
  (1 / a + 4 / b) ≥ 9 := by
sorry

end min_value_theorem_l2552_255291


namespace remaining_cakes_l2552_255294

/-- The number of cakes Baker initially had -/
def initial_cakes : ℕ := 167

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 108

/-- Theorem: The number of remaining cakes is 59 -/
theorem remaining_cakes : initial_cakes - sold_cakes = 59 := by
  sorry

end remaining_cakes_l2552_255294


namespace units_digit_product_l2552_255233

theorem units_digit_product (a b c : ℕ) : 
  a^2010 * b^1004 * c^1002 ≡ 0 [MOD 10] :=
by
  sorry

end units_digit_product_l2552_255233


namespace quadratic_two_real_roots_l2552_255290

theorem quadratic_two_real_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 4*x + 1 = 0 ∧ (m - 1) * y^2 - 4*y + 1 = 0) ↔ 
  (m ≤ 5 ∧ m ≠ 1) :=
sorry

end quadratic_two_real_roots_l2552_255290


namespace cube_difference_l2552_255237

theorem cube_difference (x : ℝ) (h : x - 1/x = 3) : x^3 - 1/x^3 = 36 := by
  sorry

end cube_difference_l2552_255237


namespace expression_simplification_l2552_255226

theorem expression_simplification :
  500 * 997 * 0.4995 * 100 = 997^2 * 25 := by
  sorry

end expression_simplification_l2552_255226


namespace one_meeting_before_return_l2552_255236

/-- Represents a runner on a rectangular track -/
structure Runner where
  speed : ℝ
  direction : Bool -- True for clockwise, False for counterclockwise

/-- Represents the rectangular track -/
def track_perimeter : ℝ := 140

/-- Calculates the number of meetings between two runners -/
def meetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem one_meeting_before_return (runner1 runner2 : Runner) 
  (h1 : runner1.speed = 6)
  (h2 : runner2.speed = 10)
  (h3 : runner1.direction ≠ runner2.direction) : 
  meetings runner1 runner2 = 1 :=
sorry

end one_meeting_before_return_l2552_255236


namespace max_leftover_candy_l2552_255247

theorem max_leftover_candy (y : ℕ) (h : y > 11) : 
  ∃ (q r : ℕ), y = 11 * q + r ∧ r > 0 ∧ r ≤ 10 := by
sorry

end max_leftover_candy_l2552_255247


namespace natalia_crates_count_l2552_255261

/-- The number of novels Natalia has --/
def novels : ℕ := 145

/-- The number of comics Natalia has --/
def comics : ℕ := 271

/-- The number of documentaries Natalia has --/
def documentaries : ℕ := 419

/-- The number of albums Natalia has --/
def albums : ℕ := 209

/-- The number of items each crate can hold --/
def crate_capacity : ℕ := 9

/-- The total number of items Natalia has --/
def total_items : ℕ := novels + comics + documentaries + albums

/-- The number of crates Natalia needs --/
def crates_needed : ℕ := (total_items + crate_capacity - 1) / crate_capacity

theorem natalia_crates_count : crates_needed = 116 := by
  sorry

end natalia_crates_count_l2552_255261


namespace f_negative_a_equals_negative_three_l2552_255265

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) - 2 / (2^x + 1)

theorem f_negative_a_equals_negative_three (a : ℝ) (h : f a = 1) : f (-a) = -3 := by
  sorry

end f_negative_a_equals_negative_three_l2552_255265


namespace total_amount_paid_l2552_255254

-- Define the structure for an item
structure Item where
  originalPrice : ℝ
  saleDiscount : ℝ
  membershipDiscount : Bool
  taxRate : ℝ

-- Define the function to calculate the final price of an item
def calculateFinalPrice (item : Item) : ℝ :=
  let priceAfterSale := item.originalPrice * (1 - item.saleDiscount)
  let priceAfterMembership := if item.membershipDiscount then priceAfterSale * 0.95 else priceAfterSale
  priceAfterMembership * (1 + item.taxRate)

-- Define the items
def vase : Item := { originalPrice := 250, saleDiscount := 0.25, membershipDiscount := true, taxRate := 0.12 }
def teacups : Item := { originalPrice := 350, saleDiscount := 0.30, membershipDiscount := false, taxRate := 0.08 }
def plate : Item := { originalPrice := 450, saleDiscount := 0, membershipDiscount := true, taxRate := 0.10 }
def ornament : Item := { originalPrice := 150, saleDiscount := 0.20, membershipDiscount := false, taxRate := 0.06 }

-- Theorem statement
theorem total_amount_paid : 
  calculateFinalPrice vase + calculateFinalPrice teacups + calculateFinalPrice plate + calculateFinalPrice ornament = 1061.55 := by
  sorry

end total_amount_paid_l2552_255254


namespace min_draws_for_even_product_l2552_255297

theorem min_draws_for_even_product (n : ℕ) (h : n = 14) :
  let S := Finset.range n
  let evens := S.filter (λ x => x % 2 = 0)
  let odds := S.filter (λ x => x % 2 ≠ 0)
  odds.card + 1 = 8 :=
by sorry

end min_draws_for_even_product_l2552_255297


namespace expression_evaluation_l2552_255286

theorem expression_evaluation : 
  (-1/2)⁻¹ + (π - 3)^0 + |1 - Real.sqrt 2| + Real.sin (45 * π / 180) * Real.sin (30 * π / 180) = 
  5 * Real.sqrt 2 / 4 - 2 := by
  sorry

end expression_evaluation_l2552_255286


namespace tangent_circle_center_height_l2552_255239

/-- A parabola with equation y = 2x^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2 = 2 * p.1 ^ 2}

/-- A circle in the interior of the parabola -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  inside_parabola : center.2 < 2 * center.1 ^ 2
  tangent_points : Set (ℝ × ℝ)
  is_tangent : tangent_points ⊆ Parabola
  on_circle : ∀ p ∈ tangent_points, (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2
  symmetry : ∀ p ∈ tangent_points, (-p.1, p.2) ∈ tangent_points

theorem tangent_circle_center_height (c : TangentCircle) :
  ∃ p ∈ c.tangent_points, c.center.2 - p.2 = 2 :=
sorry

end tangent_circle_center_height_l2552_255239


namespace ernie_makes_four_circles_l2552_255284

/-- The number of circles Ernie can make -/
def ernies_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) : ℕ :=
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle

/-- Theorem: Given the conditions from the problem, Ernie can make 4 circles -/
theorem ernie_makes_four_circles :
  ernies_circles 80 8 10 5 = 4 := by
  sorry

end ernie_makes_four_circles_l2552_255284


namespace work_completion_time_l2552_255270

theorem work_completion_time 
  (total_time : ℝ) 
  (joint_work_time : ℝ) 
  (remaining_work_time : ℝ) 
  (h1 : total_time = 24) 
  (h2 : joint_work_time = 16) 
  (h3 : remaining_work_time = 16) : 
  (total_time * remaining_work_time) / (total_time - joint_work_time) = 48 :=
by sorry

end work_completion_time_l2552_255270


namespace harish_paint_time_l2552_255295

/-- The time it takes Harish to paint the wall alone -/
def harish_time : ℝ := 3

/-- The time it takes Ganpat to paint the wall alone -/
def ganpat_time : ℝ := 6

/-- The time it takes Harish and Ganpat to paint the wall together -/
def combined_time : ℝ := 2

theorem harish_paint_time :
  (1 / harish_time + 1 / ganpat_time = 1 / combined_time) →
  harish_time = 3 := by
sorry

end harish_paint_time_l2552_255295


namespace sphere_volume_increase_l2552_255253

theorem sphere_volume_increase (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end sphere_volume_increase_l2552_255253


namespace incorrect_statement_l2552_255292

theorem incorrect_statement : ¬ (∀ a b c : ℝ, a > b → a * c > b * c) := by
  sorry

end incorrect_statement_l2552_255292


namespace problem_statement_l2552_255242

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (m : ℝ) : Set ℝ := {x | 1 - 2*m ≤ x ∧ x ≤ 2 + m}

theorem problem_statement :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ M → x ∈ N m) ↔ m ≥ 3) ∧
  (∀ m : ℝ, (M ⊂ N m ∧ M ≠ N m) ↔ m ≤ 3/2) :=
sorry

end problem_statement_l2552_255242


namespace arithmetic_sequence_sum_l2552_255227

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d → a 3 = 4 → d = -2 → a 2 + a 6 = 4 := by
  sorry

end arithmetic_sequence_sum_l2552_255227


namespace parabola_intersection_theorem_l2552_255235

/-- Parabola y² = x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

/-- Point lies on x-axis -/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- Line passing through two points -/
def line_through (p q : ℝ × ℝ) (r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem parabola_intersection_theorem 
  (O : ℝ × ℝ)  -- Origin
  (P S T : ℝ × ℝ)  -- Points on x-axis
  (A₁ B₁ A₂ B₂ : ℝ × ℝ)  -- Points on parabola
  (h_O : O = (0, 0))
  (h_P : on_x_axis P)
  (h_S : on_x_axis S)
  (h_T : on_x_axis T)
  (h_A₁ : parabola A₁)
  (h_B₁ : parabola B₁)
  (h_A₂ : parabola A₂)
  (h_B₂ : parabola B₂)
  (h_line₁ : line_through A₁ B₁ P)
  (h_line₂ : line_through A₂ B₂ P)
  (h_line₃ : line_through A₁ B₂ S)
  (h_line₄ : line_through A₂ B₁ T) :
  (S.1 - O.1) * (T.1 - O.1) = (P.1 - O.1)^2 :=
sorry

end parabola_intersection_theorem_l2552_255235


namespace grandma_contribution_correct_l2552_255262

/-- Calculates the amount Zoe's grandma gave her for the trip -/
def grandma_contribution (total_cost : ℚ) (candy_bars : ℕ) (profit_per_bar : ℚ) : ℚ :=
  total_cost - (candy_bars : ℚ) * profit_per_bar

/-- Proves that the grandma's contribution is correct -/
theorem grandma_contribution_correct (total_cost : ℚ) (candy_bars : ℕ) (profit_per_bar : ℚ) :
  grandma_contribution total_cost candy_bars profit_per_bar =
  total_cost - (candy_bars : ℚ) * profit_per_bar :=
by
  sorry

#eval grandma_contribution 485 188 (5/4)

end grandma_contribution_correct_l2552_255262


namespace certain_number_proof_l2552_255211

theorem certain_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = k * 1 + 10 ∧ 2037 = k * 1 + 7) → n = 2040 := by
  sorry

end certain_number_proof_l2552_255211


namespace min_value_reciprocal_sum_l2552_255248

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 :=
by sorry

end min_value_reciprocal_sum_l2552_255248


namespace sqrt_equation_solution_l2552_255225

theorem sqrt_equation_solution :
  ∃! x : ℚ, Real.sqrt (3 - 4 * x) = 8 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l2552_255225


namespace intersection_M_N_l2552_255232

def M : Set ℝ := {x : ℝ | |x + 1| < 3}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l2552_255232


namespace root_product_theorem_l2552_255299

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 16/3 := by
sorry

end root_product_theorem_l2552_255299


namespace second_platform_length_l2552_255220

/-- Given a train and two platforms, calculate the length of the second platform -/
theorem second_platform_length 
  (train_length : ℝ) 
  (first_platform_length : ℝ) 
  (first_crossing_time : ℝ) 
  (second_crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : first_platform_length = 200)
  (h3 : first_crossing_time = 15)
  (h4 : second_crossing_time = 20) :
  (second_crossing_time * ((train_length + first_platform_length) / first_crossing_time)) - train_length = 300 :=
by sorry

end second_platform_length_l2552_255220


namespace inscribed_angles_sum_l2552_255250

/-- Given a circle divided into 16 equal arcs, this theorem proves that 
    the sum of an inscribed angle over 3 arcs and an inscribed angle over 5 arcs is 90°. -/
theorem inscribed_angles_sum (circle : Real) (arcs : ℕ) (x y : Real) :
  arcs = 16 →
  x = 3 * (360 / (2 * arcs)) →
  y = 5 * (360 / (2 * arcs)) →
  x + y = 90 := by
  sorry

end inscribed_angles_sum_l2552_255250


namespace smallest_right_triangle_area_l2552_255217

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := a * b / 2
  let area2 := a * Real.sqrt (b^2 - a^2) / 2
  min area1 area2 = 6 * Real.sqrt 7 := by
sorry

end smallest_right_triangle_area_l2552_255217


namespace system_solution_l2552_255274

theorem system_solution : 
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℚ),
    (x₁ = 0 ∧ y₁ = -1 ∧ z₁ = 1) ∧
    (x₂ = 3 ∧ y₂ = 2 ∧ z₂ = 4) ∧
    (x₁ = (y₁ + 1) / (3 * y₁ - 5) ∧ 
     y₁ = (3 * z₁ - 2) / (2 * z₁ - 3) ∧ 
     z₁ = (3 * x₁ - 1) / (x₁ - 1)) ∧
    (x₂ = (y₂ + 1) / (3 * y₂ - 5) ∧ 
     y₂ = (3 * z₂ - 2) / (2 * z₂ - 3) ∧ 
     z₂ = (3 * x₂ - 1) / (x₂ - 1)) := by
  sorry


end system_solution_l2552_255274


namespace discount_calculation_l2552_255285

theorem discount_calculation (marked_price : ℝ) (discount_rate : ℝ) : 
  marked_price = 17.5 →
  discount_rate = 0.3 →
  2 * marked_price * (1 - discount_rate) = 24.5 := by
  sorry

end discount_calculation_l2552_255285


namespace simple_interest_principal_l2552_255206

/-- Simple interest calculation -/
theorem simple_interest_principal 
  (rate : ℝ) (interest : ℝ) (time : ℝ) (principal : ℝ) :
  rate = 12.5 →
  interest = 100 →
  time = 2 →
  principal = (interest * 100) / (rate * time) →
  principal = 400 :=
by sorry

end simple_interest_principal_l2552_255206


namespace vector_magnitude_l2552_255249

/-- Given vectors a and b in ℝ², where a = (1,3) and (a + b) ⟂ (a - b), prove that |b| = √10 -/
theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0 →
  Real.sqrt ((b.1)^2 + (b.2)^2) = Real.sqrt 10 := by
  sorry

end vector_magnitude_l2552_255249


namespace photos_framed_by_jack_or_taken_by_octavia_or_sam_l2552_255231

/-- Represents a photographer in the exhibition -/
inductive Photographer
| Octavia
| Sam
| Alice
| Max

/-- Represents a framer in the exhibition -/
inductive Framer
| Jack
| Jane

/-- The number of photographs framed by each framer for each photographer -/
def framed_photos (f : Framer) (p : Photographer) : ℕ :=
  match f, p with
  | Framer.Jack, Photographer.Octavia => 24
  | Framer.Jack, Photographer.Sam => 12
  | Framer.Jack, Photographer.Alice => 8
  | Framer.Jack, Photographer.Max => 0
  | Framer.Jane, Photographer.Octavia => 0
  | Framer.Jane, Photographer.Sam => 10
  | Framer.Jane, Photographer.Alice => 6
  | Framer.Jane, Photographer.Max => 18

/-- The total number of photographs taken by each photographer -/
def total_photos (p : Photographer) : ℕ :=
  match p with
  | Photographer.Octavia => 36
  | Photographer.Sam => 20
  | Photographer.Alice => 14
  | Photographer.Max => 32

/-- Theorem stating the number of photographs either framed by Jack or taken by Octavia or Sam -/
theorem photos_framed_by_jack_or_taken_by_octavia_or_sam :
  (framed_photos Framer.Jack Photographer.Octavia +
   framed_photos Framer.Jack Photographer.Sam +
   framed_photos Framer.Jack Photographer.Alice) +
  (total_photos Photographer.Octavia +
   total_photos Photographer.Sam) = 100 := by
  sorry

end photos_framed_by_jack_or_taken_by_octavia_or_sam_l2552_255231


namespace petya_sum_theorem_l2552_255210

/-- Represents Petya's operation on the board numbers -/
def petyaOperation (x y z : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, y, z - 1, x * y)

/-- Represents the invariant property throughout Petya's operations -/
def invariant (x y z : ℕ) (sum : ℕ) : Prop :=
  x * y * z = sum + x * y * z

/-- The main theorem stating that the sum of products on the paper
    equals the initial product of board numbers when process terminates -/
theorem petya_sum_theorem (x y z : ℕ) :
  ∃ (n : ℕ) (sum : ℕ),
    (∃ (a b : ℕ), a * b * 0 = n) ∧
    invariant x y z sum ∧
    sum = x * y * z := by
  sorry

end petya_sum_theorem_l2552_255210


namespace custom_mult_four_three_l2552_255200

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2 + (a-b)^2

-- Theorem statement
theorem custom_mult_four_three : custom_mult 4 3 = 19 := by
  sorry

end custom_mult_four_three_l2552_255200


namespace tangent_equality_l2552_255251

-- Define the types for circles and points
variable (Circle Point : Type)

-- Define the predicates and functions
variable (outside : Circle → Circle → Prop)
variable (touches : Circle → Circle → Point → Point → Prop)
variable (passes_through : Circle → Point → Point → Prop)
variable (intersects_at : Circle → Circle → Point → Prop)
variable (tangent_at : Circle → Point → Point → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem tangent_equality 
  (S₁ S₂ S₃ : Circle) 
  (A B C D K : Point) :
  outside S₁ S₂ →
  touches S₁ S₂ A B →
  passes_through S₃ A B →
  intersects_at S₃ S₁ C →
  intersects_at S₃ S₂ D →
  tangent_at S₁ C K →
  tangent_at S₂ D K →
  distance K C = distance K D :=
sorry

end tangent_equality_l2552_255251


namespace fraction_invariance_l2552_255203

theorem fraction_invariance (x y : ℝ) :
  2 * y^2 / (x - y)^2 = 2 * (3*y)^2 / ((3*x) - (3*y))^2 :=
sorry

end fraction_invariance_l2552_255203


namespace fruit_purchase_change_l2552_255277

/-- The change received when purchasing fruit -/
def change (a : ℝ) : ℝ := 100 - 3 * a

/-- Theorem stating the change received when purchasing fruit -/
theorem fruit_purchase_change (a : ℝ) (h : a ≤ 30) :
  change a = 100 - 3 * a := by
  sorry

end fruit_purchase_change_l2552_255277


namespace find_P_l2552_255243

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set M
def M (P : ℝ) : Set ℕ := {x ∈ U | x^2 - 5*x + P = 0}

-- Define the complement of M in U
def complement_M (P : ℝ) : Set ℕ := U \ M P

-- Theorem statement
theorem find_P : ∃ P : ℝ, complement_M P = {2, 3} ∧ P = 4 := by sorry

end find_P_l2552_255243


namespace lateral_surface_area_l2552_255289

-- Define the frustum
structure Frustum where
  r₁ : ℝ  -- upper base radius
  r₂ : ℝ  -- lower base radius
  h : ℝ   -- height
  l : ℝ   -- slant height

-- Define the conditions
def frustum_conditions (f : Frustum) : Prop :=
  f.r₂ = 4 * f.r₁ ∧ f.h = 4 * f.r₁ ∧ f.l = 10

-- Theorem to prove
theorem lateral_surface_area (f : Frustum) 
  (hf : frustum_conditions f) : 
  π * (f.r₁ + f.r₂) * f.l = 100 * π := by
  sorry

end lateral_surface_area_l2552_255289


namespace overlap_angle_is_90_degrees_l2552_255222

/-- A regular octagon -/
structure RegularOctagon where
  sides : Fin 8 → ℝ × ℝ
  is_regular : ∀ (i j : Fin 8), dist (sides i) (sides ((i + 1) % 8)) = dist (sides j) (sides ((j + 1) % 8))

/-- The angle at the intersection point when two non-adjacent sides of a regular octagon overlap -/
def overlap_angle (octagon : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The angle at the intersection point when two non-adjacent sides of a regular octagon overlap is 90° -/
theorem overlap_angle_is_90_degrees (octagon : RegularOctagon) : 
  overlap_angle octagon = 90 :=
sorry

end overlap_angle_is_90_degrees_l2552_255222


namespace intersection_condition_max_area_rhombus_condition_l2552_255246

-- Define the lines and ellipse
def l₁ (k x : ℝ) : ℝ := k * x + 2
def l₂ (k x : ℝ) : ℝ := k * x - 2
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the intersection points
def intersection_points (k : ℝ) : Prop := ∃ A B C D : ℝ × ℝ,
  ellipse A.1 A.2 ∧ (A.2 = l₁ k A.1 ∨ A.2 = l₂ k A.1) ∧
  ellipse B.1 B.2 ∧ (B.2 = l₁ k B.1 ∨ B.2 = l₂ k B.1) ∧
  ellipse C.1 C.2 ∧ (C.2 = l₁ k C.1 ∨ C.2 = l₂ k C.1) ∧
  ellipse D.1 D.2 ∧ (D.2 = l₁ k D.1 ∨ D.2 = l₂ k D.1)

-- Define the area of the quadrilateral
def area (A B C D : ℝ × ℝ) : ℝ := sorry

-- Define the rhombus condition
def is_rhombus (A B C D : ℝ × ℝ) : Prop := sorry

theorem intersection_condition (k : ℝ) :
  intersection_points k ↔ abs k > Real.sqrt 3 / 3 := sorry

theorem max_area {k : ℝ} (h : intersection_points k) :
  ∃ A B C D : ℝ × ℝ, area A B C D ≤ 4 * Real.sqrt 3 := sorry

theorem rhombus_condition {k : ℝ} (h : intersection_points k) :
  ∃ A B C D : ℝ × ℝ, is_rhombus A B C D → k = Real.sqrt 15 / 3 ∨ k = -Real.sqrt 15 / 3 := sorry

end intersection_condition_max_area_rhombus_condition_l2552_255246


namespace jed_gives_away_two_cards_l2552_255283

/-- Represents the number of cards Jed gives away every two weeks -/
def cards_given_away : ℕ := 2

/-- Represents the initial number of cards Jed has -/
def initial_cards : ℕ := 20

/-- Represents the number of cards Jed gets every week -/
def weekly_cards : ℕ := 6

/-- Represents the number of weeks that have passed -/
def weeks_passed : ℕ := 4

/-- Represents the total number of cards Jed has after 4 weeks -/
def final_cards : ℕ := 40

/-- Theorem stating that Jed gives away 2 cards every two weeks -/
theorem jed_gives_away_two_cards : 
  initial_cards + weekly_cards * weeks_passed - cards_given_away * (weeks_passed / 2) = final_cards :=
sorry

end jed_gives_away_two_cards_l2552_255283


namespace log_sum_equals_two_l2552_255293

theorem log_sum_equals_two :
  2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) = 2 := by
  sorry

end log_sum_equals_two_l2552_255293


namespace giraffe_zebra_ratio_is_three_to_one_l2552_255296

/-- Represents the zoo layout and animal distribution --/
structure Zoo where
  tiger_enclosures : ℕ
  zebra_enclosures_per_tiger : ℕ
  tigers_per_enclosure : ℕ
  zebras_per_enclosure : ℕ
  giraffes_per_enclosure : ℕ
  total_animals : ℕ

/-- Calculates the ratio of giraffe enclosures to zebra enclosures --/
def giraffe_zebra_enclosure_ratio (zoo : Zoo) : ℚ :=
  let total_zebra_enclosures := zoo.tiger_enclosures * zoo.zebra_enclosures_per_tiger
  let total_tigers := zoo.tiger_enclosures * zoo.tigers_per_enclosure
  let total_zebras := total_zebra_enclosures * zoo.zebras_per_enclosure
  let total_giraffes := zoo.total_animals - total_tigers - total_zebras
  let giraffe_enclosures := total_giraffes / zoo.giraffes_per_enclosure
  giraffe_enclosures / total_zebra_enclosures

/-- The main theorem stating the ratio of giraffe enclosures to zebra enclosures --/
theorem giraffe_zebra_ratio_is_three_to_one (zoo : Zoo)
  (h1 : zoo.tiger_enclosures = 4)
  (h2 : zoo.zebra_enclosures_per_tiger = 2)
  (h3 : zoo.tigers_per_enclosure = 4)
  (h4 : zoo.zebras_per_enclosure = 10)
  (h5 : zoo.giraffes_per_enclosure = 2)
  (h6 : zoo.total_animals = 144) :
  giraffe_zebra_enclosure_ratio zoo = 3 / 1 := by
  sorry

end giraffe_zebra_ratio_is_three_to_one_l2552_255296


namespace range_of_a_l2552_255278

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 6)^x > (2*a - 6)^y

def q (a : ℝ) : Prop := ∃ x y : ℝ, x > 3 ∧ y > 3 ∧ x ≠ y ∧
  x^2 - 3*a*x + 2*a^2 + 1 = 0 ∧ y^2 - 3*a*y + 2*a^2 + 1 = 0

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a > 7/2 :=
sorry

end range_of_a_l2552_255278


namespace chess_program_ratio_l2552_255272

theorem chess_program_ratio (total_students : ℕ) (chess_students : ℕ) (tournament_students : ℕ) 
  (h1 : total_students = 24)
  (h2 : tournament_students = 4)
  (h3 : chess_students = 2 * tournament_students)
  : (chess_students : ℚ) / total_students = 1 / 3 := by
  sorry

end chess_program_ratio_l2552_255272


namespace eliana_steps_l2552_255245

def day1_steps : ℕ := 200 + 300

def day2_steps (d1 : ℕ) : ℕ := d1 * d1

def day3_steps (d1 d2 : ℕ) : ℕ := d1 + d2 + 100

def total_steps (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem eliana_steps :
  let d1 := day1_steps
  let d2 := day2_steps d1
  let d3 := day3_steps d1 d2
  total_steps d1 d2 d3 = 501100 := by
  sorry

end eliana_steps_l2552_255245


namespace number_problem_l2552_255213

theorem number_problem (N : ℕ) (h1 : ∃ k : ℕ, N = 5 * k) (h2 : N / 5 = 25) :
  (N - 17) / 6 = 18 := by
sorry

end number_problem_l2552_255213


namespace machine_value_after_two_years_l2552_255276

/-- The market value of a machine after two years of depreciation -/
theorem machine_value_after_two_years
  (purchase_price : ℝ)
  (yearly_depreciation_rate : ℝ)
  (h1 : purchase_price = 8000)
  (h2 : yearly_depreciation_rate = 0.1) :
  purchase_price * (1 - yearly_depreciation_rate)^2 = 6480 := by
  sorry

end machine_value_after_two_years_l2552_255276


namespace football_game_attendance_l2552_255279

theorem football_game_attendance (saturday : ℕ) (expected_total : ℕ) : 
  saturday = 80 →
  expected_total = 350 →
  (saturday + (saturday - 20) + (saturday - 20 + 50) + (saturday + (saturday - 20))) - expected_total = 40 :=
by
  sorry

end football_game_attendance_l2552_255279
