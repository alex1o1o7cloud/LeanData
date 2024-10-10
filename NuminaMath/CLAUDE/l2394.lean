import Mathlib

namespace planes_parallel_if_perp_lines_parallel_l2394_239430

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perp_lines_parallel
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : perpendicular m β)
  (h3 : parallel l m) :
  parallel_planes α β :=
sorry

end planes_parallel_if_perp_lines_parallel_l2394_239430


namespace problem_solution_l2394_239491

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ) 
  (h_xavier : p_xavier = 1/4)
  (h_yvonne : p_yvonne = 1/3)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/32 := by
  sorry

end problem_solution_l2394_239491


namespace tree_growth_fraction_l2394_239473

/-- Represents the height of a tree over time -/
def tree_height (initial_height : ℕ) (growth_rate : ℕ) (years : ℕ) : ℕ :=
  initial_height + growth_rate * years

/-- The fraction representing the increase in height from year a to year b -/
def height_increase_fraction (initial_height : ℕ) (growth_rate : ℕ) (a b : ℕ) : ℚ :=
  (tree_height initial_height growth_rate b - tree_height initial_height growth_rate a) /
  tree_height initial_height growth_rate a

theorem tree_growth_fraction :
  height_increase_fraction 4 1 4 6 = 1 / 4 := by
  sorry

end tree_growth_fraction_l2394_239473


namespace distribute_balls_to_bags_correct_l2394_239406

/-- The number of ways to distribute n identical balls into m numbered bags, such that no bag is empty -/
def distribute_balls_to_bags (n m : ℕ) : ℕ :=
  Nat.choose (n - 1) (m - 1)

/-- Theorem: The number of ways to distribute n identical balls into m numbered bags, 
    such that no bag is empty, is equal to (n-1) choose (m-1) -/
theorem distribute_balls_to_bags_correct (n m : ℕ) (h1 : n > m) (h2 : m > 0) : 
  distribute_balls_to_bags n m = Nat.choose (n - 1) (m - 1) := by
  sorry

#check distribute_balls_to_bags_correct

end distribute_balls_to_bags_correct_l2394_239406


namespace tetrahedron_fits_in_box_l2394_239401

theorem tetrahedron_fits_in_box : ∃ (x y z : ℝ),
  (x^2 + y^2 = 100) ∧
  (x^2 + z^2 = 81) ∧
  (y^2 + z^2 = 64) ∧
  (x < 8) ∧ (y < 8) ∧ (z < 5) := by
  sorry

end tetrahedron_fits_in_box_l2394_239401


namespace charity_plates_delivered_l2394_239446

/-- The number of plates delivered by a charity given the cost of ingredients and total spent -/
theorem charity_plates_delivered (rice_cost chicken_cost total_spent : ℚ) : 
  rice_cost = 1/10 →
  chicken_cost = 4/10 →
  total_spent = 50 →
  (total_spent / (rice_cost + chicken_cost) : ℚ) = 100 := by
  sorry

end charity_plates_delivered_l2394_239446


namespace house_c_to_a_ratio_l2394_239448

/-- Represents the real estate problem with Nigella's sales --/
structure RealEstateProblem where
  base_salary : ℝ
  commission_rate : ℝ
  houses_sold : ℕ
  total_earnings : ℝ
  house_a_cost : ℝ
  house_b_cost : ℝ
  house_c_cost : ℝ

/-- Theorem stating the ratio of House C's cost to House A's cost before subtracting $110,000 --/
theorem house_c_to_a_ratio (problem : RealEstateProblem)
  (h1 : problem.base_salary = 3000)
  (h2 : problem.commission_rate = 0.02)
  (h3 : problem.houses_sold = 3)
  (h4 : problem.total_earnings = 8000)
  (h5 : problem.house_b_cost = 3 * problem.house_a_cost)
  (h6 : problem.house_c_cost = problem.house_a_cost * 2 - 110000)
  (h7 : problem.house_a_cost = 60000) :
  (problem.house_c_cost + 110000) / problem.house_a_cost = 2 := by
  sorry


end house_c_to_a_ratio_l2394_239448


namespace tailor_buttons_total_l2394_239400

theorem tailor_buttons_total (green yellow blue total : ℕ) : 
  green = 90 →
  yellow = green + 10 →
  blue = green - 5 →
  total = green + yellow + blue →
  total = 275 := by
sorry

end tailor_buttons_total_l2394_239400


namespace largest_multiple_of_seven_l2394_239489

theorem largest_multiple_of_seven (n : ℤ) : n = 147 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -150) ∧ 
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) ∧ (-m > -150) → m ≤ n) := by
  sorry

end largest_multiple_of_seven_l2394_239489


namespace min_reciprocal_sum_l2394_239443

-- Define the constraint set
def ConstraintSet : Set (ℝ × ℝ) :=
  {(x, y) | 8 * x - y ≤ 4 ∧ x + y ≥ -1 ∧ y ≤ 4 * x}

-- Define the objective function
def ObjectiveFunction (a b : ℝ) (p : ℝ × ℝ) : ℝ :=
  a * p.1 + b * p.2

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (p : ℝ × ℝ), p ∈ ConstraintSet ∧ 
   ∀ (q : ℝ × ℝ), q ∈ ConstraintSet → ObjectiveFunction a b q ≤ ObjectiveFunction a b p) →
  (∀ (p : ℝ × ℝ), p ∈ ConstraintSet → ObjectiveFunction a b p ≤ 2) →
  1/a + 1/b ≥ 9/2 :=
by sorry

end min_reciprocal_sum_l2394_239443


namespace saline_solution_water_calculation_l2394_239468

/-- Given a saline solution mixture, calculate the amount of water needed for a larger volume -/
theorem saline_solution_water_calculation 
  (salt_solution : ℝ) 
  (initial_water : ℝ) 
  (initial_total : ℝ) 
  (final_volume : ℝ) 
  (h1 : salt_solution = 0.05)
  (h2 : initial_water = 0.03)
  (h3 : initial_total = salt_solution + initial_water)
  (h4 : final_volume = 0.64) :
  final_volume * (initial_water / initial_total) = 0.24 := by
sorry

end saline_solution_water_calculation_l2394_239468


namespace hall_length_is_six_l2394_239494

/-- A rectangular hall with given properties --/
structure Hall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ
  area_floor_ceiling : ℝ
  area_walls : ℝ

/-- The theorem stating the conditions and the result to be proved --/
theorem hall_length_is_six (h : Hall) 
  (h_width : h.width = 6)
  (h_volume : h.volume = 108)
  (h_areas : h.area_floor_ceiling = h.area_walls)
  (h_floor_ceiling : h.area_floor_ceiling = 2 * h.length * h.width)
  (h_walls : h.area_walls = 2 * h.length * h.height + 2 * h.width * h.height)
  (h_volume_calc : h.volume = h.length * h.width * h.height) :
  h.length = 6 := by
  sorry

end hall_length_is_six_l2394_239494


namespace nine_candidates_l2394_239404

/- Define the number of ways to select president and vice president -/
def selection_ways : ℕ := 72

/- Define the property that determines the number of candidates -/
def candidate_count (n : ℕ) : Prop :=
  n * (n - 1) = selection_ways

/- Theorem statement -/
theorem nine_candidates : 
  ∃ (n : ℕ), candidate_count n ∧ n = 9 :=
by sorry

end nine_candidates_l2394_239404


namespace pauls_license_plate_earnings_l2394_239436

theorem pauls_license_plate_earnings 
  (total_states : ℕ) 
  (pauls_states : ℕ) 
  (total_earnings : ℚ) :
  total_states = 50 →
  pauls_states = 40 →
  total_earnings = 160 →
  (total_earnings / (pauls_states / total_states * 100 : ℚ)) = 2 :=
by sorry

end pauls_license_plate_earnings_l2394_239436


namespace derivative_at_one_l2394_239463

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 0 := by sorry

end derivative_at_one_l2394_239463


namespace inequality_system_solution_set_l2394_239470

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 2 ≥ -5 ∧ 3*x < x + 2}
  S = {x : ℝ | -3 ≤ x ∧ x < 1} := by
sorry

end inequality_system_solution_set_l2394_239470


namespace expression_value_l2394_239479

theorem expression_value (a b : ℚ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3/2 := by
  sorry

end expression_value_l2394_239479


namespace log_sum_equality_l2394_239456

theorem log_sum_equality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (1 / (1 + Real.log (c/a) / Real.log (a^2 * b))) +
  (1 / (1 + Real.log (a/b) / Real.log (b^2 * c))) +
  (1 / (1 + Real.log (b/c) / Real.log (c^2 * a))) = 3 := by
  sorry

end log_sum_equality_l2394_239456


namespace pig_count_l2394_239403

theorem pig_count (initial_pigs additional_pigs : Float) 
  (h1 : initial_pigs = 64.0)
  (h2 : additional_pigs = 86.0) :
  initial_pigs + additional_pigs = 150.0 := by
sorry

end pig_count_l2394_239403


namespace line_plane_relationship_l2394_239423

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : parallel_lines a b) 
  (h2 : parallel_line_plane a α) : 
  parallel_line_plane b α ∨ line_in_plane b α :=
sorry

end line_plane_relationship_l2394_239423


namespace cube_difference_prime_factor_l2394_239439

theorem cube_difference_prime_factor (a b p : ℕ) : 
  Nat.Prime p → a^3 - b^3 = 633 * p → a = 16 ∧ b = 13 :=
by sorry

end cube_difference_prime_factor_l2394_239439


namespace total_reams_is_five_l2394_239414

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := 3

/-- The total number of reams of paper bought by Haley's mom -/
def total_reams : ℕ := reams_for_haley + reams_for_sister

theorem total_reams_is_five : total_reams = 5 := by
  sorry

end total_reams_is_five_l2394_239414


namespace complex_number_magnitude_squared_l2394_239402

theorem complex_number_magnitude_squared :
  ∀ (z : ℂ), z + Complex.abs z = 4 + 5*I → Complex.abs z^2 = 1681/64 := by
  sorry

end complex_number_magnitude_squared_l2394_239402


namespace total_cost_special_requirement_l2394_239495

/-- The number of ways to choose 3 consecutive numbers from 01 to 10 -/
def consecutive_three_from_ten : Nat := 8

/-- The number of ways to choose 2 consecutive numbers from 11 to 20 -/
def consecutive_two_from_ten : Nat := 9

/-- The number of ways to choose 1 number from 21 to 30 -/
def one_from_ten : Nat := 10

/-- The number of ways to choose 1 number from 31 to 36 -/
def one_from_six : Nat := 6

/-- The cost of a single entry in yuan -/
def entry_cost : Nat := 2

/-- Theorem: The total cost of purchasing all possible entries meeting the special requirement is 8640 yuan -/
theorem total_cost_special_requirement : 
  consecutive_three_from_ten * consecutive_two_from_ten * one_from_ten * one_from_six * entry_cost = 8640 := by
  sorry

end total_cost_special_requirement_l2394_239495


namespace liner_and_water_speed_theorem_l2394_239428

/-- The distance between Chongqing and Shibaozhai in kilometers -/
def distance : ℝ := 270

/-- The time taken to travel downstream in hours -/
def downstream_time : ℝ := 9

/-- The time taken to travel upstream in hours -/
def upstream_time : ℝ := 13.5

/-- The speed of the liner in still water in km/h -/
def liner_speed : ℝ := 25

/-- The speed of the water flow in km/h -/
def water_speed : ℝ := 5

/-- The distance between Chongqing Port and the new dock in km -/
def new_dock_distance : ℝ := 162

theorem liner_and_water_speed_theorem :
  (downstream_time * (liner_speed + water_speed) = distance) ∧
  (upstream_time * (liner_speed - water_speed) = distance) ∧
  (new_dock_distance / (liner_speed + water_speed) = (distance - new_dock_distance) / (liner_speed - water_speed)) := by
  sorry

#check liner_and_water_speed_theorem

end liner_and_water_speed_theorem_l2394_239428


namespace cylinder_surface_area_l2394_239427

/-- The total surface area of a right cylinder with height 10 cm and radius 3 cm is 78π cm². -/
theorem cylinder_surface_area : 
  let h : ℝ := 10  -- height in cm
  let r : ℝ := 3   -- radius in cm
  let lateral_area := 2 * Real.pi * r * h
  let base_area := Real.pi * r^2
  let total_area := lateral_area + 2 * base_area
  total_area = 78 * Real.pi := by sorry

end cylinder_surface_area_l2394_239427


namespace joshua_total_cars_l2394_239409

/-- The total number of toy cars Joshua has -/
def total_cars (box1 box2 box3 : ℕ) : ℕ := box1 + box2 + box3

/-- Theorem: Joshua has 71 toy cars in total -/
theorem joshua_total_cars :
  total_cars 21 31 19 = 71 := by
  sorry

end joshua_total_cars_l2394_239409


namespace distinct_triangles_on_circle_l2394_239481

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of distinct triangles that can be drawn -/
def num_triangles : ℕ := Nat.choose n k

theorem distinct_triangles_on_circle :
  num_triangles = 220 := by sorry

end distinct_triangles_on_circle_l2394_239481


namespace ellipse_line_intersection_l2394_239476

-- Define the ellipse parameters
def a : ℝ := 2
def b : ℝ := 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define a line passing through M
def line_through_M (k : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = k * (x - M.1)

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  M.1 = (x₁ + x₂) / 2 ∧ M.2 = (y₁ + y₂) / 2

-- Theorem statement
theorem ellipse_line_intersection :
  ∃ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    is_on_ellipse x₁ y₁ ∧
    is_on_ellipse x₂ y₂ ∧
    line_through_M k x₁ y₁ ∧
    line_through_M k x₂ y₂ ∧
    is_midpoint x₁ y₁ x₂ y₂ ∧
    k = -1/2 :=
  sorry

end ellipse_line_intersection_l2394_239476


namespace train_speed_calculation_l2394_239477

/-- Given a train of length 150 meters passing an oak tree in 9.99920006399488 seconds,
    prove that its speed is 54.00287976961843 km/hr. -/
theorem train_speed_calculation (train_length : Real) (time_to_pass : Real) :
  train_length = 150 →
  time_to_pass = 9.99920006399488 →
  (train_length / time_to_pass) * 3.6 = 54.00287976961843 := by
  sorry

#eval (150 / 9.99920006399488) * 3.6

end train_speed_calculation_l2394_239477


namespace lisa_candy_consumption_l2394_239417

/-- The number of candies Lisa eats on other days of the week -/
def candies_on_other_days (total_candies : ℕ) (candies_on_mon_wed : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℚ :=
  let total_days := days_per_week * num_weeks
  let mon_wed_days := 2 * num_weeks
  let other_days := total_days - mon_wed_days
  let candies_on_mon_wed_total := candies_on_mon_wed * mon_wed_days
  let remaining_candies := total_candies - candies_on_mon_wed_total
  (remaining_candies : ℚ) / other_days

theorem lisa_candy_consumption :
  candies_on_other_days 36 2 7 4 = 1 := by sorry

end lisa_candy_consumption_l2394_239417


namespace modulus_of_complex_l2394_239424

theorem modulus_of_complex (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_l2394_239424


namespace power_of_one_seventh_l2394_239441

def is_greatest_power_of_2_factor (x : ℕ) : Prop :=
  2^x ∣ 180 ∧ ∀ k > x, ¬(2^k ∣ 180)

def is_greatest_power_of_3_factor (y : ℕ) : Prop :=
  3^y ∣ 180 ∧ ∀ k > y, ¬(3^k ∣ 180)

theorem power_of_one_seventh (x y : ℕ) 
  (h2 : is_greatest_power_of_2_factor x) 
  (h3 : is_greatest_power_of_3_factor y) : 
  (1/7 : ℚ)^(y - x) = 1 := by
  sorry

end power_of_one_seventh_l2394_239441


namespace rectangle_formations_l2394_239458

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4

def choose (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem rectangle_formations :
  (choose horizontal_lines 2) * (choose vertical_lines 2) = 60 := by
  sorry

end rectangle_formations_l2394_239458


namespace sum_of_max_min_f_l2394_239431

noncomputable def f (x : ℝ) : ℝ := 1 + (Real.sin x) / (2 + Real.cos x)

theorem sum_of_max_min_f : 
  (⨆ (x : ℝ), f x) + (⨅ (x : ℝ), f x) = 2 :=
sorry

end sum_of_max_min_f_l2394_239431


namespace floor_of_7_9_l2394_239485

theorem floor_of_7_9 : ⌊(7.9 : ℝ)⌋ = 7 := by sorry

end floor_of_7_9_l2394_239485


namespace arctan_inequality_implies_a_nonnegative_l2394_239478

theorem arctan_inequality_implies_a_nonnegative (a : ℝ) : 
  (∀ x : ℝ, Real.arctan (Real.sqrt (x^2 + x + 13/4)) ≥ π/3 - a) → a ≥ 0 := by
  sorry

end arctan_inequality_implies_a_nonnegative_l2394_239478


namespace imaginary_part_of_complex_fraction_l2394_239420

theorem imaginary_part_of_complex_fraction : Complex.im ((3 * Complex.I + 4) / (1 + 2 * Complex.I)) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l2394_239420


namespace sine_function_midpoint_l2394_239415

/-- Given a sine function y = A sin(Bx + C) + D that oscillates between 6 and 2, prove that D = 4 -/
theorem sine_function_midpoint (A B C D : ℝ) 
  (h_oscillation : ∀ x, 2 ≤ A * Real.sin (B * x + C) + D ∧ A * Real.sin (B * x + C) + D ≤ 6) : 
  D = 4 := by
  sorry

end sine_function_midpoint_l2394_239415


namespace paper_strip_dimensions_l2394_239412

theorem paper_strip_dimensions 
  (a b c : ℕ) 
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) :
  a = 1 ∧ b + c = 22 := by
sorry

end paper_strip_dimensions_l2394_239412


namespace bella_stamps_l2394_239451

/-- The number of stamps Bella bought -/
def total_stamps (snowflake truck rose : ℕ) : ℕ := snowflake + truck + rose

/-- Theorem stating the total number of stamps Bella bought -/
theorem bella_stamps : ∃ (snowflake truck rose : ℕ),
  snowflake = 11 ∧
  truck = snowflake + 9 ∧
  rose = truck - 13 ∧
  total_stamps snowflake truck rose = 38 := by
  sorry

end bella_stamps_l2394_239451


namespace f_properties_g_property_l2394_239407

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) + 6 * Real.cos (ω * x / 2)^2 - 3

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem f_properties (ω : ℝ) (θ : ℝ) (h_ω : ω > 0) (h_θ : 0 < θ ∧ θ < Real.pi / 2) :
  (is_even (fun x ↦ f ω (x + θ)) ∧ 
   has_period (fun x ↦ f ω (x + θ)) Real.pi ∧
   ∀ p, has_period (fun x ↦ f ω (x + θ)) p → p ≥ Real.pi) →
  ω = 2 ∧ θ = Real.pi / 12 :=
sorry

theorem g_property (ω : ℝ) (h_ω : ω > 0) :
  is_increasing_on (fun x ↦ f ω (3 * x)) 0 (Real.pi / 3) →
  ω ≤ 1 / 6 :=
sorry

end f_properties_g_property_l2394_239407


namespace smallest_period_of_given_functions_l2394_239499

open Real

noncomputable def f1 (x : ℝ) := -cos x
noncomputable def f2 (x : ℝ) := abs (sin x)
noncomputable def f3 (x : ℝ) := cos (2 * x)
noncomputable def f4 (x : ℝ) := tan (2 * x - π / 4)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, is_periodic f q ∧ q > 0 → p ≤ q

theorem smallest_period_of_given_functions :
  smallest_positive_period f2 π ∧
  smallest_positive_period f3 π ∧
  (∀ p, smallest_positive_period f1 p → p > π) ∧
  (∀ p, smallest_positive_period f4 p → p > π) :=
sorry

end smallest_period_of_given_functions_l2394_239499


namespace trigonometric_inequality_l2394_239465

theorem trigonometric_inequality : ∀ (a b c : ℝ),
  a = Real.sin (4/5) →
  b = Real.cos (4/5) →
  c = Real.tan (4/5) →
  c > a ∧ a > b :=
by sorry

end trigonometric_inequality_l2394_239465


namespace symmetric_function_periodic_l2394_239459

/-- A function f: ℝ → ℝ satisfying certain symmetry properties -/
def symmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f (2 - x)) ∧ (∀ x, f (x + 7) = f (7 - x))

/-- Theorem stating that a symmetric function is periodic with period 10 -/
theorem symmetric_function_periodic (f : ℝ → ℝ) (h : symmetricFunction f) :
  ∀ x, f (x + 10) = f x := by
  sorry

end symmetric_function_periodic_l2394_239459


namespace ball_volume_ratio_l2394_239484

theorem ball_volume_ratio :
  ∀ (x y z : ℝ),
    x > 0 → y > 0 → z > 0 →
    x = 3 * (y - x) →
    z - y = 3 * x →
    ∃ (k : ℝ), k > 0 ∧ x = 3 * k ∧ y = 4 * k ∧ z = 13 * k :=
by sorry

end ball_volume_ratio_l2394_239484


namespace diagonals_30_sided_polygon_l2394_239490

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by sorry

end diagonals_30_sided_polygon_l2394_239490


namespace dihydrogen_monoxide_weight_is_18_016_l2394_239422

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of hydrogen atoms in a water molecule -/
def hydrogen_count : ℕ := 2

/-- The number of oxygen atoms in a water molecule -/
def oxygen_count : ℕ := 1

/-- The molecular weight of dihydrogen monoxide (H2O) in g/mol -/
def dihydrogen_monoxide_weight : ℝ := 
  hydrogen_count * hydrogen_weight + oxygen_count * oxygen_weight

/-- Theorem: The molecular weight of dihydrogen monoxide (H2O) is 18.016 g/mol -/
theorem dihydrogen_monoxide_weight_is_18_016 : 
  dihydrogen_monoxide_weight = 18.016 := by
  sorry

end dihydrogen_monoxide_weight_is_18_016_l2394_239422


namespace wire_cutting_l2394_239419

theorem wire_cutting (total_length : ℝ) (piece1 piece2 piece3 : ℝ) : 
  total_length = 95 →
  piece2 = 1.5 * piece1 →
  piece3 = 1.5 * piece2 →
  piece1 + piece2 + piece3 = total_length →
  piece3 = 45 := by
sorry

end wire_cutting_l2394_239419


namespace distinct_z_values_l2394_239471

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_digits (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 * c + 10 * b + a

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values : 
  ∃ (S : Finset ℕ), (∀ x, is_valid_number x → z x ∈ S) ∧ S.card = 10 := by
sorry

end distinct_z_values_l2394_239471


namespace interest_rate_proof_l2394_239455

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal interest time : ℚ) 
  (h1 : principal = 800)
  (h2 : interest = 128)
  (h3 : time = 4)
  (h4 : simple_interest principal (4 : ℚ) time = interest) : 
  ∃ (rate : ℚ), rate = 4 ∧ simple_interest principal rate time = interest := by
  sorry

end interest_rate_proof_l2394_239455


namespace solve_for_y_l2394_239416

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 100) (h2 : x = 50) : y = 25 := by
  sorry

end solve_for_y_l2394_239416


namespace one_third_greater_than_decimal_l2394_239425

theorem one_third_greater_than_decimal : 
  ∃ (ε : ℚ), ε > 0 ∧ ε = 1 / (3 * 10^9) ∧ 1/3 = 0.333333333 + ε := by
  sorry

end one_third_greater_than_decimal_l2394_239425


namespace line_L_equation_trajectory_Q_equation_l2394_239405

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line L
def LineL (x y : ℝ) : Prop := x = 1 ∨ 3*x - 4*y + 5 = 0

-- Define the trajectory of Q
def TrajectoryQ (x y : ℝ) : Prop := x^2/4 + y^2/16 = 1

-- Theorem for Part I
theorem line_L_equation : 
  ∃ (A B : ℝ × ℝ), 
  Circle A.1 A.2 ∧ Circle B.1 B.2 ∧
  LineL A.1 A.2 ∧ LineL B.1 B.2 ∧
  LineL 1 2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
sorry

-- Theorem for Part II
theorem trajectory_Q_equation :
  ∀ (M : ℝ × ℝ), Circle M.1 M.2 →
  ∃ (Q : ℝ × ℝ), 
  Q.1 = M.1 ∧ Q.2 = 2 * M.2 ∧
  TrajectoryQ Q.1 Q.2 :=
sorry

end line_L_equation_trajectory_Q_equation_l2394_239405


namespace fat_thin_eating_time_l2394_239432

/-- The time it takes for two people to eat a certain amount of fruit together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Theorem: Mr. Fat and Mr. Thin will take 46.875 minutes to eat 5 pounds of fruit together -/
theorem fat_thin_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let amount : ℚ := 5         -- Amount of fruit in pounds
  eating_time fat_rate thin_rate amount = 46875 / 1000 := by
sorry

end fat_thin_eating_time_l2394_239432


namespace rancher_cows_count_l2394_239460

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →
  cows + horses = 168 →
  cows = 140 := by
sorry

end rancher_cows_count_l2394_239460


namespace intersection_A_complement_B_l2394_239433

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_A_complement_B_l2394_239433


namespace tim_speed_is_45_l2394_239466

/-- Represents the distance between Tim and Élan in miles -/
def initial_distance : ℝ := 150

/-- Represents Élan's initial speed in mph -/
def elan_initial_speed : ℝ := 5

/-- Represents the distance Tim travels until meeting Élan in miles -/
def tim_travel_distance : ℝ := 100

/-- Represents the number of hours until Tim and Élan meet -/
def meeting_time : ℕ := 2

/-- Represents Tim's initial speed in mph -/
def tim_initial_speed : ℝ := 45

/-- Theorem stating that given the conditions, Tim's initial speed is 45 mph -/
theorem tim_speed_is_45 :
  tim_initial_speed * (2^meeting_time - 1) = initial_distance - elan_initial_speed * (2^meeting_time - 1) :=
sorry

end tim_speed_is_45_l2394_239466


namespace exponent_division_l2394_239454

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^2 / x^8 = 1 / x^6 := by
  sorry

end exponent_division_l2394_239454


namespace jack_king_queen_probability_l2394_239474

theorem jack_king_queen_probability : 
  let deck_size : ℕ := 52
  let jack_count : ℕ := 4
  let king_count : ℕ := 4
  let queen_count : ℕ := 4
  let prob_jack : ℚ := jack_count / deck_size
  let prob_king : ℚ := king_count / (deck_size - 1)
  let prob_queen : ℚ := queen_count / (deck_size - 2)
  prob_jack * prob_king * prob_queen = 8 / 16575 :=
by sorry

end jack_king_queen_probability_l2394_239474


namespace eraser_difference_l2394_239469

theorem eraser_difference (andrea_erasers : ℕ) (anya_multiplier : ℕ) : 
  andrea_erasers = 4 →
  anya_multiplier = 4 →
  anya_multiplier * andrea_erasers - andrea_erasers = 12 := by
  sorry

end eraser_difference_l2394_239469


namespace Q_equals_two_three_four_l2394_239496

-- Define the set P
def P : Set ℕ := {1, 2}

-- Define the set Q
def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

-- Theorem statement
theorem Q_equals_two_three_four : Q = {2, 3, 4} := by
  sorry

end Q_equals_two_three_four_l2394_239496


namespace sin_n_equals_cos_810_l2394_239413

theorem sin_n_equals_cos_810 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * Real.pi / 180) = Real.cos (810 * Real.pi / 180) →
  n = -180 ∨ n = 0 ∨ n = 180 := by
  sorry

end sin_n_equals_cos_810_l2394_239413


namespace fraction_simplification_l2394_239444

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by sorry

end fraction_simplification_l2394_239444


namespace triangle_sinB_sinC_l2394_239429

theorem triangle_sinB_sinC (a b c : Real) (A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- 2c + b = 2a * cos B
  (2 * c + b = 2 * a * Real.cos B) →
  -- Area S = 3/2 * sqrt(3)
  (1/2 * b * c * Real.sin A = 3/2 * Real.sqrt 3) →
  -- c = 2
  (c = 2) →
  -- Then sin B * sin C = 9/38
  (Real.sin B * Real.sin C = 9/38) := by
sorry

end triangle_sinB_sinC_l2394_239429


namespace star_value_for_specific_conditions_l2394_239440

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

-- Theorem statement
theorem star_value_for_specific_conditions (a b : ℤ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 35) :
  star a b = 12 / 35 := by
  sorry

end star_value_for_specific_conditions_l2394_239440


namespace fraction_equality_l2394_239450

theorem fraction_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
sorry

end fraction_equality_l2394_239450


namespace largest_after_three_operations_obtainable_1999_l2394_239426

-- Define the expansion operation
def expand (a b : ℕ) : ℕ := a * b + a + b

-- Theorem for the largest number after three operations
theorem largest_after_three_operations :
  let step1 := expand 1 4
  let step2 := expand 4 step1
  let step3 := expand step1 step2
  step3 = 499 := by sorry

-- Theorem for the obtainability of 1999
theorem obtainable_1999 :
  ∃ (m n : ℕ), 2000 = 2^m * 5^n := by sorry

end largest_after_three_operations_obtainable_1999_l2394_239426


namespace sector_area_l2394_239472

/-- The area of a circular sector with a central angle of 60° and a radius of 10 cm is 50π/3 cm². -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 60 * π / 180) (h2 : r = 10) :
  (θ / (2 * π)) * (π * r^2) = 50 * π / 3 := by
  sorry

end sector_area_l2394_239472


namespace nancy_carrots_l2394_239408

def initial_carrots : ℕ → ℕ → ℕ → Prop :=
  fun x thrown_out additional =>
    x - thrown_out + additional = 31

theorem nancy_carrots : initial_carrots 12 2 21 := by sorry

end nancy_carrots_l2394_239408


namespace paper_length_equals_days_until_due_l2394_239449

/-- The number of pages in Stacy's history paper -/
def paper_length : ℕ := sorry

/-- The number of days until the paper is due -/
def days_until_due : ℕ := 12

/-- The number of pages Stacy needs to write per day to finish on time -/
def pages_per_day : ℕ := 1

/-- Theorem stating that the paper length is equal to the number of days until due -/
theorem paper_length_equals_days_until_due : 
  paper_length = days_until_due := by sorry

end paper_length_equals_days_until_due_l2394_239449


namespace count_valid_numbers_l2394_239462

/-- The set of digits that can be used to form the numbers -/
def digits : Finset Nat := {0, 1, 2, 3}

/-- A predicate that checks if a number is a four-digit even number -/
def is_valid_number (n : Nat) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ Even n

/-- A function that returns the digits of a number as a list -/
def number_to_digits (n : Nat) : List Nat :=
  sorry

/-- A predicate that checks if a number uses only the allowed digits without repetition -/
def uses_valid_digits (n : Nat) : Prop :=
  let d := number_to_digits n
  d.toFinset ⊆ digits ∧ d.length = 4 ∧ d.Nodup

/-- The set of all valid numbers according to the problem conditions -/
def valid_numbers : Finset Nat :=
  sorry

theorem count_valid_numbers : valid_numbers.card = 10 := by
  sorry

end count_valid_numbers_l2394_239462


namespace lee_cookies_l2394_239467

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this function calculates how many cookies he can make with any number of cups. -/
def cookies_per_cups (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_per_cups 5 = 40 := by
  sorry

end lee_cookies_l2394_239467


namespace gcd_2505_7350_l2394_239453

theorem gcd_2505_7350 : Nat.gcd 2505 7350 = 15 := by
  sorry

end gcd_2505_7350_l2394_239453


namespace fraction_equality_l2394_239434

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 4 := by
  sorry

end fraction_equality_l2394_239434


namespace weight_estimation_l2394_239498

-- Define the variables and constants
variable (x y : ℝ)
variable (x_sum y_sum : ℝ)
variable (b_hat : ℝ)
variable (n : ℕ)

-- Define the conditions
def conditions (x_sum y_sum b_hat : ℝ) (n : ℕ) : Prop :=
  x_sum = 1600 ∧ y_sum = 460 ∧ b_hat = 0.85 ∧ n = 10

-- Define the regression line equation
def regression_line (x b_hat a_hat : ℝ) : ℝ :=
  b_hat * x + a_hat

-- Theorem statement
theorem weight_estimation 
  (x_sum y_sum b_hat : ℝ) (n : ℕ) 
  (h : conditions x_sum y_sum b_hat n) : 
  ∃ a_hat : ℝ, regression_line 170 b_hat a_hat = 54.5 :=
sorry

end weight_estimation_l2394_239498


namespace martas_textbook_cost_l2394_239457

/-- The total cost of Marta's textbooks --/
def total_cost (sale_price : ℕ) (sale_quantity : ℕ) (online_cost : ℕ) (bookstore_multiplier : ℕ) : ℕ :=
  sale_price * sale_quantity + online_cost + bookstore_multiplier * online_cost

/-- Theorem stating the total cost of Marta's textbooks --/
theorem martas_textbook_cost :
  total_cost 10 5 40 3 = 210 := by
  sorry

end martas_textbook_cost_l2394_239457


namespace mistake_correction_l2394_239483

theorem mistake_correction (a : ℤ) (h : 31 - a = 12) : 31 + a = 50 := by
  sorry

end mistake_correction_l2394_239483


namespace ellipse_and_intersection_properties_l2394_239418

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of the intersection line -/
def intersection_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 2

/-- Theorem stating the properties of the ellipse and the range of k -/
theorem ellipse_and_intersection_properties :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), ellipse_C x y ∧ x = 1 ∧ y = 3/2) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂ ∧
    x₁ ≠ x₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂ ∧
    x₁ ≠ x₂ →
    (1/3 * x₁) * (2/3 * x₂) + (1/3 * y₁) * (2/3 * y₂) < 
    ((1/3 * x₁)^2 + (1/3 * y₁)^2 + (2/3 * x₂)^2 + (2/3 * y₂)^2) / 2) →
  (k > 1/2 ∧ k < 2 * Real.sqrt 3 / 3) ∨ (k < -1/2 ∧ k > -2 * Real.sqrt 3 / 3) :=
sorry

end ellipse_and_intersection_properties_l2394_239418


namespace min_value_expression_l2394_239442

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x^2 + y^2 + 1/x^2 + 2*y/x ≥ m) ∧
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x^2 + y^2 + 1/x^2 + 2*y/x = m) :=
by
  sorry

end min_value_expression_l2394_239442


namespace right_triangle_check_l2394_239464

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

theorem right_triangle_check :
  ¬ is_right_triangle 1 2 3 ∧
  ¬ is_right_triangle 1 2 2 ∧
  ¬ is_right_triangle (Real.sqrt 2) (Real.sqrt 2) (Real.sqrt 2) ∧
  is_right_triangle 6 8 10 :=
sorry

end right_triangle_check_l2394_239464


namespace c_share_is_45_l2394_239410

/-- Represents the rent share calculation for a pasture -/
structure PastureRent where
  total_rent : ℕ
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ

/-- Calculates C's share of the rent -/
def calculate_c_share (p : PastureRent) : ℕ :=
  let total_ox_months := p.a_oxen * p.a_months + p.b_oxen * p.b_months + p.c_oxen * p.c_months
  let c_ox_months := p.c_oxen * p.c_months
  (c_ox_months * p.total_rent) / total_ox_months

/-- Theorem stating that C's share of the rent is 45 -/
theorem c_share_is_45 (p : PastureRent) 
  (h1 : p.total_rent = 175)
  (h2 : p.a_oxen = 10) (h3 : p.a_months = 7)
  (h4 : p.b_oxen = 12) (h5 : p.b_months = 5)
  (h6 : p.c_oxen = 15) (h7 : p.c_months = 3) :
  calculate_c_share p = 45 := by
  sorry

end c_share_is_45_l2394_239410


namespace percentage_prefer_corn_l2394_239435

def kids_prefer_peas : ℕ := 6
def kids_prefer_carrots : ℕ := 9
def kids_prefer_corn : ℕ := 5

def total_kids : ℕ := kids_prefer_peas + kids_prefer_carrots + kids_prefer_corn

theorem percentage_prefer_corn : 
  (kids_prefer_corn : ℚ) / (total_kids : ℚ) * 100 = 25 := by
  sorry

end percentage_prefer_corn_l2394_239435


namespace girls_to_boys_ratio_l2394_239437

/-- Proves that the ratio of girls to boys is 4:5 given the class conditions -/
theorem girls_to_boys_ratio (total_students : ℕ) (boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  total_students = 30 →
  boys = 10 →
  cups_per_boy = 5 →
  total_cups = 90 →
  (total_students - boys, boys) = (20, 10) ∧ 
  (20 : ℚ) / 10 = 4 / 5 := by
  sorry

#check girls_to_boys_ratio

end girls_to_boys_ratio_l2394_239437


namespace complex_fraction_simplification_l2394_239493

theorem complex_fraction_simplification : 
  (((12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500)) / 
   ((6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500))) = -995 := by
  sorry

end complex_fraction_simplification_l2394_239493


namespace matches_needed_for_new_win_rate_l2394_239497

/-- Given a player who has won 19 out of 20 matches, prove that they need to win 5 more matches
    without any losses to achieve a 96% winning rate. -/
theorem matches_needed_for_new_win_rate
  (initial_matches : Nat)
  (initial_wins : Nat)
  (target_win_rate : Rat)
  (h1 : initial_matches = 20)
  (h2 : initial_wins = 19)
  (h3 : target_win_rate = 24/25) :
  ∃ (additional_wins : Nat),
    additional_wins = 5 ∧
    (initial_wins + additional_wins : Rat) / (initial_matches + additional_wins) = target_win_rate :=
by sorry

end matches_needed_for_new_win_rate_l2394_239497


namespace total_cost_calculation_l2394_239482

def cabin_cost : ℕ := 6000
def land_cost_multiplier : ℕ := 4

theorem total_cost_calculation :
  cabin_cost + land_cost_multiplier * cabin_cost = 30000 := by
  sorry

end total_cost_calculation_l2394_239482


namespace abc_inequality_l2394_239445

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end abc_inequality_l2394_239445


namespace race_speed_ratio_l2394_239492

theorem race_speed_ratio (race_distance : ℕ) (head_start : ℕ) (win_margin : ℕ) :
  race_distance = 500 →
  head_start = 140 →
  win_margin = 20 →
  (race_distance - head_start : ℚ) / (race_distance - win_margin : ℚ) = 3 / 4 := by
  sorry

end race_speed_ratio_l2394_239492


namespace horner_method_operations_l2394_239488

-- Define the polynomial
def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ := ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1

-- Theorem statement
theorem horner_method_operations :
  ∃ (mult_ops add_ops : ℕ),
    (∀ x : ℝ, f x = horner_method x) ∧
    mult_ops = 5 ∧
    add_ops = 5 :=
sorry

end horner_method_operations_l2394_239488


namespace chocolate_bar_count_l2394_239486

/-- Represents a box containing chocolate bars -/
structure ChocolateBox where
  bars : ℕ

/-- Represents a large box containing small boxes of chocolates -/
structure LargeBox where
  smallBoxes : ℕ
  smallBoxContents : ChocolateBox

/-- Calculates the total number of chocolate bars in a large box -/
def totalChocolateBars (box : LargeBox) : ℕ :=
  box.smallBoxes * box.smallBoxContents.bars

theorem chocolate_bar_count (largeBox : LargeBox) 
    (h1 : largeBox.smallBoxes = 15)
    (h2 : largeBox.smallBoxContents.bars = 20) : 
    totalChocolateBars largeBox = 300 := by
  sorry

end chocolate_bar_count_l2394_239486


namespace nth_root_inequality_l2394_239480

theorem nth_root_inequality (m n : ℕ) (h1 : m > n) (h2 : n ≥ 2) :
  (m : ℝ) ^ (1 / n : ℝ) - (n : ℝ) ^ (1 / m : ℝ) > 1 / (m * n : ℝ) := by
  sorry

end nth_root_inequality_l2394_239480


namespace sin_2alpha_values_l2394_239447

theorem sin_2alpha_values (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) = Real.sin (π / 4 - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17/18 := by
  sorry

end sin_2alpha_values_l2394_239447


namespace power_of_power_l2394_239475

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2394_239475


namespace quadratic_coefficient_l2394_239438

theorem quadratic_coefficient (b m : ℝ) : 
  b > 0 ∧ 
  (∀ x, x^2 + b*x + 72 = (x + m)^2 + 12) →
  b = 4 * Real.sqrt 15 := by
sorry

end quadratic_coefficient_l2394_239438


namespace tan_65_degrees_l2394_239421

/-- If tan 110° = α, then tan 65° = (α - 1) / (1 + α) -/
theorem tan_65_degrees (α : ℝ) (h : Real.tan (110 * π / 180) = α) :
  Real.tan (65 * π / 180) = (α - 1) / (α + 1) := by
  sorry

end tan_65_degrees_l2394_239421


namespace range_of_a_l2394_239487

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, 4^x - a*2^(x+1) + a^2 - 1 ≥ 0) ↔ 
  a ∈ Set.Iic 1 ∪ Set.Ici 5 := by
sorry

end range_of_a_l2394_239487


namespace fixed_point_on_line_l2394_239411

/-- The line equation mx+(1-m)y+m-2=0 always passes through the point (1,2) for all real m. -/
theorem fixed_point_on_line (m : ℝ) : m * 1 + (1 - m) * 2 + m - 2 = 0 := by
  sorry

end fixed_point_on_line_l2394_239411


namespace eighteen_men_handshakes_l2394_239452

/-- The maximum number of handshakes among n men without cyclic handshakes -/
def maxHandshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The maximum number of handshakes among 18 men without cyclic handshakes is 153 -/
theorem eighteen_men_handshakes :
  maxHandshakes 18 = 153 := by
  sorry

end eighteen_men_handshakes_l2394_239452


namespace B_max_at_50_l2394_239461

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Definition of B_k -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.1 ^ k)

/-- Statement: B_k is largest when k = 50 -/
theorem B_max_at_50 : ∀ k : ℕ, k ≤ 500 → B k ≤ B 50 := by sorry

end B_max_at_50_l2394_239461
