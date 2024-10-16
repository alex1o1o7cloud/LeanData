import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l399_39954

theorem rectangular_prism_sum (A B C : ℝ) 
  (h1 : A * B = 30)
  (h2 : A * C = 40)
  (h3 : B * C = 60) :
  A + B + C = 9 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l399_39954


namespace NUMINAMATH_CALUDE_cube_root_equation_l399_39978

theorem cube_root_equation (x : ℝ) : (9 * x + 8) ^ (1/3 : ℝ) = 4 → x = 56 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_l399_39978


namespace NUMINAMATH_CALUDE_defective_percentage_is_3_6_percent_l399_39915

/-- Represents the percentage of products manufactured by each machine -/
structure MachineProduction where
  m1 : ℝ
  m2 : ℝ
  m3 : ℝ

/-- Represents the percentage of defective products for each machine -/
structure DefectivePercentage where
  m1 : ℝ
  m2 : ℝ
  m3 : ℝ

/-- Calculates the percentage of defective products in the stockpile -/
def calculateDefectivePercentage (prod : MachineProduction) (defect : DefectivePercentage) : ℝ :=
  prod.m1 * defect.m1 + prod.m2 * defect.m2 + prod.m3 * defect.m3

theorem defective_percentage_is_3_6_percent 
  (prod : MachineProduction)
  (defect : DefectivePercentage)
  (h1 : prod.m1 = 0.4)
  (h2 : prod.m2 = 0.3)
  (h3 : prod.m3 = 0.3)
  (h4 : defect.m1 = 0.03)
  (h5 : defect.m2 = 0.01)
  (h6 : defect.m3 = 0.07) :
  calculateDefectivePercentage prod defect = 0.036 := by
  sorry

#eval calculateDefectivePercentage 
  { m1 := 0.4, m2 := 0.3, m3 := 0.3 } 
  { m1 := 0.03, m2 := 0.01, m3 := 0.07 }

end NUMINAMATH_CALUDE_defective_percentage_is_3_6_percent_l399_39915


namespace NUMINAMATH_CALUDE_min_sales_to_break_even_l399_39902

def old_salary : ℕ := 75000
def new_base_salary : ℕ := 45000
def commission_rate : ℚ := 15 / 100
def sale_amount : ℕ := 750

theorem min_sales_to_break_even :
  let difference := old_salary - new_base_salary
  let commission_per_sale := commission_rate * sale_amount
  let min_sales := (difference : ℚ) / commission_per_sale
  ⌈min_sales⌉ = 267 := by sorry

end NUMINAMATH_CALUDE_min_sales_to_break_even_l399_39902


namespace NUMINAMATH_CALUDE_vector_perpendicular_l399_39948

/-- Given two vectors a and b in R², prove that if a + b is perpendicular to a,
    then the second component of b is -7/2. -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.1 = 2) :
  (a + b) • a = 0 → b.2 = -7/2 := by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l399_39948


namespace NUMINAMATH_CALUDE_joans_remaining_practice_time_l399_39960

/-- Given Joan's music practice schedule, calculate the remaining time for finger exercises. -/
theorem joans_remaining_practice_time :
  let total_time : ℕ := 2 * 60  -- 2 hours in minutes
  let piano_time : ℕ := 30
  let writing_time : ℕ := 25
  let reading_time : ℕ := 38
  let used_time : ℕ := piano_time + writing_time + reading_time
  total_time - used_time = 27 := by
  sorry

#check joans_remaining_practice_time

end NUMINAMATH_CALUDE_joans_remaining_practice_time_l399_39960


namespace NUMINAMATH_CALUDE_factor_calculation_l399_39975

theorem factor_calculation (n f : ℝ) : n = 121 ∧ n * f - 140 = 102 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l399_39975


namespace NUMINAMATH_CALUDE_certain_value_problem_l399_39947

theorem certain_value_problem (n x : ℝ) : n = 5 ∧ n = 5 * (n - x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l399_39947


namespace NUMINAMATH_CALUDE_initial_distance_between_trains_l399_39908

def train_length_1 : ℝ := 120
def train_length_2 : ℝ := 210
def speed_1 : ℝ := 69
def speed_2 : ℝ := 82
def meeting_time : ℝ := 1.9071321976361095

theorem initial_distance_between_trains : 
  let relative_speed := (speed_1 + speed_2) * 1000 / 3600
  let distance_covered := relative_speed * (meeting_time * 3600)
  distance_covered - (train_length_1 + train_length_2) = 287670 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_trains_l399_39908


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l399_39905

def f (x : ℝ) : ℝ := 240 - 60 * x

theorem f_satisfies_data_points : 
  (f 0 = 240) ∧ 
  (f 1 = 180) ∧ 
  (f 2 = 120) ∧ 
  (f 3 = 60) ∧ 
  (f 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_data_points_l399_39905


namespace NUMINAMATH_CALUDE_total_spending_is_48_l399_39970

/-- Represents the savings and spending pattern for a week -/
structure SavingsPattern where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  friday : ℝ
  thursday_spend_ratio : ℝ
  saturday_spend_ratio : ℝ

/-- Calculates the total spending on Thursday and Saturday -/
def total_spending (pattern : SavingsPattern) : ℝ :=
  let initial_savings := pattern.monday + pattern.tuesday + pattern.wednesday
  let thursday_spending := initial_savings * pattern.thursday_spend_ratio
  let friday_total := initial_savings - thursday_spending + pattern.friday
  let saturday_spending := friday_total * pattern.saturday_spend_ratio
  thursday_spending + saturday_spending

/-- Theorem stating that the total spending on Thursday and Saturday is $48 -/
theorem total_spending_is_48 (pattern : SavingsPattern) 
  (h1 : pattern.monday = 15)
  (h2 : pattern.tuesday = 28)
  (h3 : pattern.wednesday = 13)
  (h4 : pattern.friday = 22)
  (h5 : pattern.thursday_spend_ratio = 0.5)
  (h6 : pattern.saturday_spend_ratio = 0.4) :
  total_spending pattern = 48 := by
  sorry


end NUMINAMATH_CALUDE_total_spending_is_48_l399_39970


namespace NUMINAMATH_CALUDE_tan_660_degrees_l399_39992

theorem tan_660_degrees : Real.tan (660 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_660_degrees_l399_39992


namespace NUMINAMATH_CALUDE_trihedral_angle_obtuse_angles_l399_39946

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  AOB : ℝ
  BOC : ℝ
  COA : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: If all plane angles of a trihedral angle are obtuse, then all dihedral angles are obtuse -/
theorem trihedral_angle_obtuse_angles (t : TrihedralAngle)
  (h_AOB : t.AOB > π / 2)
  (h_BOC : t.BOC > π / 2)
  (h_COA : t.COA > π / 2) :
  t.α > π / 2 ∧ t.β > π / 2 ∧ t.γ > π / 2 := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_obtuse_angles_l399_39946


namespace NUMINAMATH_CALUDE_boys_count_is_sixty_l399_39930

/-- Represents the number of boys in a group of 3 students -/
inductive GroupComposition
  | ThreeGirls
  | TwoGirlsOneBoy
  | OneGirlTwoBoys
  | ThreeBoys

/-- Represents the distribution of groups -/
structure GroupDistribution where
  total_groups : Nat
  one_boy_groups : Nat
  at_least_two_boys_groups : Nat
  three_boys_groups : Nat
  three_girls_groups : Nat

/-- Calculates the total number of boys given a group distribution -/
def count_boys (gd : GroupDistribution) : Nat :=
  gd.one_boy_groups + 2 * (gd.at_least_two_boys_groups - gd.three_boys_groups) + 3 * gd.three_boys_groups

/-- The main theorem to be proved -/
theorem boys_count_is_sixty (gd : GroupDistribution) 
  (h1 : gd.total_groups = 35)
  (h2 : gd.one_boy_groups = 10)
  (h3 : gd.at_least_two_boys_groups = 19)
  (h4 : gd.three_boys_groups = 2 * gd.three_girls_groups)
  (h5 : gd.three_girls_groups = gd.total_groups - gd.one_boy_groups - gd.at_least_two_boys_groups) :
  count_boys gd = 60 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_is_sixty_l399_39930


namespace NUMINAMATH_CALUDE_students_liking_both_mountains_and_sea_l399_39938

/-- Given a school with the following properties:
  * There are 500 total students
  * 289 students like mountains
  * 337 students like the sea
  * 56 students like neither mountains nor the sea
  Then the number of students who like both mountains and the sea is 182. -/
theorem students_liking_both_mountains_and_sea 
  (total : ℕ) 
  (like_mountains : ℕ) 
  (like_sea : ℕ) 
  (like_neither : ℕ) 
  (h1 : total = 500)
  (h2 : like_mountains = 289)
  (h3 : like_sea = 337)
  (h4 : like_neither = 56) :
  like_mountains + like_sea - (total - like_neither) = 182 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_mountains_and_sea_l399_39938


namespace NUMINAMATH_CALUDE_comparison_of_powers_l399_39976

theorem comparison_of_powers (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) : 
  (a^a * b^b > a^b * b^a) ∧ 
  (a^a * b^b * c^c > (a*b*c)^((a+b+c)/3)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l399_39976


namespace NUMINAMATH_CALUDE_robertson_seymour_grid_minor_theorem_l399_39971

-- Define a graph type
def Graph := Type

-- Define treewidth for a graph
def treewidth (G : Graph) : ℕ := sorry

-- Define the concept of a minor for graphs
def is_minor (H G : Graph) : Prop := sorry

-- Define a grid graph
def grid_graph (r : ℕ) : Graph := sorry

theorem robertson_seymour_grid_minor_theorem :
  ∀ r : ℕ, ∃ k : ℕ, ∀ G : Graph, treewidth G ≥ k → is_minor (grid_graph r) G := by
  sorry

end NUMINAMATH_CALUDE_robertson_seymour_grid_minor_theorem_l399_39971


namespace NUMINAMATH_CALUDE_flower_count_proof_l399_39989

/-- The number of daisy seeds planted -/
def daisy_seeds : ℕ := 25

/-- The number of sunflower seeds planted -/
def sunflower_seeds : ℕ := 25

/-- The percentage of daisy seeds that germinate -/
def daisy_germination_rate : ℚ := 60 / 100

/-- The percentage of sunflower seeds that germinate -/
def sunflower_germination_rate : ℚ := 80 / 100

/-- The percentage of germinated plants that produce flowers -/
def flower_production_rate : ℚ := 80 / 100

/-- The total number of plants that produce flowers -/
def plants_with_flowers : ℕ := 28

theorem flower_count_proof :
  (daisy_seeds * daisy_germination_rate * flower_production_rate +
   sunflower_seeds * sunflower_germination_rate * flower_production_rate).floor = plants_with_flowers := by
  sorry

end NUMINAMATH_CALUDE_flower_count_proof_l399_39989


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l399_39949

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Two lines are parallel if and only if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_lines_slope (k : ℝ) :
  let line1 : Line := { slope := k, yIntercept := -7 }
  let line2 : Line := { slope := -3, yIntercept := 4 }
  parallel line1 line2 → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l399_39949


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l399_39919

/-- The area of a rectangle inscribed in a triangle -/
theorem inscribed_rectangle_area (b h x : ℝ) (hb : b > 0) (hh : h > 0) (hx : x > 0) (hxh : x < h) :
  let triangle_area := (1/2) * b * h
  let rectangle_base := b * (1 - x/h)
  let rectangle_area := x * rectangle_base
  rectangle_area = (b * x / h) * (h - x) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l399_39919


namespace NUMINAMATH_CALUDE_only_zero_has_linear_factors_l399_39910

/-- A polynomial in x and y with a parameter k -/
def poly (k : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + 2*x + k*y - k

/-- Predicate for linear factors with integer coefficients -/
def has_linear_integer_factors (k : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    poly k x y = (a*x + b*y + c) * (d*x + e*y + f)

theorem only_zero_has_linear_factors :
  ∀ k : ℤ, has_linear_integer_factors k ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_only_zero_has_linear_factors_l399_39910


namespace NUMINAMATH_CALUDE_largest_divisor_of_composite_l399_39980

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n + n^4 - n^3 for composite n > 6 -/
def LargestDivisor : ℕ := 6

theorem largest_divisor_of_composite (n : ℕ) (h1 : IsComposite n) (h2 : n > 6) :
  (∀ d : ℕ, d > LargestDivisor → ∃ m : ℕ, IsComposite m ∧ m > 6 ∧ ¬(d ∣ (m + m^4 - m^3))) ∧
  (LargestDivisor ∣ (n + n^4 - n^3)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_composite_l399_39980


namespace NUMINAMATH_CALUDE_self_reciprocal_numbers_form_set_l399_39944

-- Define the characteristics of a set
def isSet (S : Type) : Prop :=
  (∀ x : S, x = x) ∧  -- definiteness
  (∀ x y : S, x = y → y = x) ∧  -- distinctness
  (∀ x y : S, x ≠ y → y ≠ x)  -- unorderedness

-- Define the set of numbers whose reciprocals are equal to themselves
def SelfReciprocalNumbers : Set ℝ := {x : ℝ | x ≠ 0 ∧ 1 / x = x}

-- State the theorem
theorem self_reciprocal_numbers_form_set :
  isSet SelfReciprocalNumbers ∧
  ¬ isSet (Set (String → Bool)) ∧  -- Taller students in a class
  ¬ isSet (Set (String → Bool)) ∧  -- Long-lived people
  ¬ isSet (Set ℝ)  -- Approximate values of √2
  := by sorry

end NUMINAMATH_CALUDE_self_reciprocal_numbers_form_set_l399_39944


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_35_billion_l399_39952

theorem scientific_notation_of_1_35_billion :
  ∃ (a : ℝ) (n : ℤ), 1.35e9 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 :=
by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_35_billion_l399_39952


namespace NUMINAMATH_CALUDE_average_calculation_l399_39987

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of four numbers
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

-- State the theorem
theorem average_calculation :
  avg4 (avg4 2 2 0 2) (avg2 3 1) 0 3 = 13 / 8 := by
  sorry

end NUMINAMATH_CALUDE_average_calculation_l399_39987


namespace NUMINAMATH_CALUDE_specific_committee_selection_l399_39969

/-- The number of ways to choose a committee with a specific person included -/
def committee_selection (n m k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 462 ways to choose a 6-person committee from 12 people with one specific person included -/
theorem specific_committee_selection :
  committee_selection 12 6 1 = 462 := by
  sorry

end NUMINAMATH_CALUDE_specific_committee_selection_l399_39969


namespace NUMINAMATH_CALUDE_ghost_entry_exit_ways_l399_39932

def num_windows : ℕ := 8

theorem ghost_entry_exit_ways :
  (num_windows : ℕ) * (num_windows - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_entry_exit_ways_l399_39932


namespace NUMINAMATH_CALUDE_payment_difference_l399_39909

def original_price : Float := 42.00000000000004
def discount_rate : Float := 0.10
def tip_rate : Float := 0.15

def discounted_price : Float := original_price * (1 - discount_rate)

def john_payment : Float := original_price + (original_price * tip_rate)
def jane_payment : Float := discounted_price + (discounted_price * tip_rate)

theorem payment_difference : john_payment - jane_payment = 4.830000000000005 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l399_39909


namespace NUMINAMATH_CALUDE_cheese_options_correct_l399_39900

/-- Represents the number of cheese options -/
def cheese_options : ℕ := 3

/-- Represents the number of meat options -/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options -/
def vegetable_options : ℕ := 5

/-- Represents the total number of topping combinations -/
def total_combinations : ℕ := 57

/-- Theorem stating that the number of cheese options is correct -/
theorem cheese_options_correct : 
  cheese_options * (meat_options * (vegetable_options - 1) + 
  (meat_options - 1) * vegetable_options) = total_combinations :=
sorry

end NUMINAMATH_CALUDE_cheese_options_correct_l399_39900


namespace NUMINAMATH_CALUDE_total_time_is_34_hours_l399_39977

/-- Calculates the total time spent on drawing and coloring pictures. -/
def total_time (num_pictures : ℕ) (draw_time : ℝ) (color_time_reduction : ℝ) : ℝ :=
  let color_time := draw_time * (1 - color_time_reduction)
  num_pictures * (draw_time + color_time)

/-- Proves that the total time spent on all pictures is 34 hours. -/
theorem total_time_is_34_hours :
  total_time 10 2 0.3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_34_hours_l399_39977


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l399_39914

theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  r > 0 ∧ d > 0 ∧ 
  40 * r^2 = 10 * d^2 + 16000 ∧ 
  36 * r^2 = 81 * d^2 + 11664 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l399_39914


namespace NUMINAMATH_CALUDE_middle_angle_range_l399_39927

theorem middle_angle_range (α β γ : Real) : 
  (0 ≤ α) → (0 ≤ β) → (0 ≤ γ) →  -- angles are non-negative
  (α + β + γ = 180) →             -- sum of angles in a triangle
  (α ≤ β) → (β ≤ γ) →             -- β is the middle angle
  (0 < β) ∧ (β < 90) :=           -- conclusion
by sorry

end NUMINAMATH_CALUDE_middle_angle_range_l399_39927


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l399_39916

theorem imaginary_part_of_z (z : ℂ) (h : z + (3 - 4*I) = 1) : z.im = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l399_39916


namespace NUMINAMATH_CALUDE_mass_of_man_l399_39903

/-- The mass of a man who causes a boat to sink by a certain amount in water. -/
theorem mass_of_man (boat_length boat_breadth boat_sink_depth water_density : ℝ) 
  (h1 : boat_length = 3)
  (h2 : boat_breadth = 2)
  (h3 : boat_sink_depth = 0.02)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * boat_sink_depth * water_density = 120 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_l399_39903


namespace NUMINAMATH_CALUDE_largest_value_l399_39922

theorem largest_value : 
  max (5 * Real.sqrt 2 - 7) (max (7 - 5 * Real.sqrt 2) (max |4/4 - 4/4| 0.1)) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l399_39922


namespace NUMINAMATH_CALUDE_cupcake_price_l399_39990

theorem cupcake_price (cupcake_count : ℕ) (cookie_count : ℕ) (cookie_price : ℚ)
  (basketball_count : ℕ) (basketball_price : ℚ) (drink_count : ℕ) (drink_price : ℚ) :
  cupcake_count = 50 →
  cookie_count = 40 →
  cookie_price = 1/2 →
  basketball_count = 2 →
  basketball_price = 40 →
  drink_count = 20 →
  drink_price = 2 →
  ∃ (cupcake_price : ℚ),
    cupcake_count * cupcake_price + cookie_count * cookie_price =
    basketball_count * basketball_price + drink_count * drink_price ∧
    cupcake_price = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_cupcake_price_l399_39990


namespace NUMINAMATH_CALUDE_probability_three_red_balls_l399_39985

/-- The probability of drawing 3 red balls from a bag containing 7 red, 9 blue, and 5 green balls -/
theorem probability_three_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
  (h_total : total_balls = red_balls + blue_balls + green_balls)
  (h_red : red_balls = 7)
  (h_blue : blue_balls = 9)
  (h_green : green_balls = 5) :
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) * ((red_balls - 2) / (total_balls - 2)) = 1 / 38 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_red_balls_l399_39985


namespace NUMINAMATH_CALUDE_odd_coefficients_in_binomial_expansion_l399_39901

theorem odd_coefficients_in_binomial_expansion :
  let coefficients := List.range 9 |>.map (fun k => Nat.choose 8 k)
  (coefficients.filter (fun c => c % 2 = 1)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_coefficients_in_binomial_expansion_l399_39901


namespace NUMINAMATH_CALUDE_bug_path_tiles_l399_39974

def width : ℕ := 15
def length : ℕ := 25
def total_tiles : ℕ := 375

theorem bug_path_tiles : 
  width + length - Nat.gcd width length = 35 := by sorry

end NUMINAMATH_CALUDE_bug_path_tiles_l399_39974


namespace NUMINAMATH_CALUDE_sum_of_roots_theorem_l399_39968

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 4

-- State the theorem
theorem sum_of_roots_theorem (a b : ℝ) 
  (h1 : f a = 14) 
  (h2 : f b = -14) : 
  a + b = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_theorem_l399_39968


namespace NUMINAMATH_CALUDE_increase_decrease_percentage_l399_39966

theorem increase_decrease_percentage (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) : 
  let increased := initial * (1 + increase_percent / 100)
  let final := increased * (1 - decrease_percent / 100)
  initial = 80 ∧ increase_percent = 150 ∧ decrease_percent = 25 → final = 150 := by
  sorry

end NUMINAMATH_CALUDE_increase_decrease_percentage_l399_39966


namespace NUMINAMATH_CALUDE_green_ball_probability_l399_39920

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers X, Y, and Z -/
def X : Container := ⟨3, 7⟩
def Y : Container := ⟨8, 2⟩
def Z : Container := ⟨5, 5⟩

/-- The list of all containers -/
def containers : List Container := [X, Y, Z]

/-- The probability of selecting a green ball -/
def probabilityGreenBall : ℚ :=
  (List.sum (containers.map greenProbability)) / containers.length

theorem green_ball_probability :
  probabilityGreenBall = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l399_39920


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l399_39965

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 where a > √2, 
    if the angle between its asymptotes is π/3, 
    then its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > Real.sqrt 2) :
  let angle_between_asymptotes := π / 3
  let slope_of_asymptote := Real.sqrt 2 / a
  let eccentricity := Real.sqrt (a^2 + 2) / a
  (angle_between_asymptotes = π / 3 ∧ 
   slope_of_asymptote = Real.tan (π / 6)) →
  eccentricity = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l399_39965


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l399_39945

theorem a_gt_one_sufficient_not_necessary_for_a_squared_gt_one :
  (∀ a : ℝ, a > 1 → a^2 > 1) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l399_39945


namespace NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l399_39991

theorem gcd_of_lcm_and_ratio (X Y : ℕ) : 
  X ≠ 0 → Y ≠ 0 → 
  lcm X Y = 180 → 
  ∃ (k : ℕ), X = 2 * k ∧ Y = 5 * k → 
  gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l399_39991


namespace NUMINAMATH_CALUDE_sara_flowers_l399_39928

theorem sara_flowers (yellow_flowers : ℕ) (num_bouquets : ℕ) (red_flowers : ℕ) :
  yellow_flowers = 24 →
  num_bouquets = 8 →
  yellow_flowers % num_bouquets = 0 →
  red_flowers = yellow_flowers →
  red_flowers = 24 := by
sorry

end NUMINAMATH_CALUDE_sara_flowers_l399_39928


namespace NUMINAMATH_CALUDE_car_discount_proof_l399_39906

/-- Proves that the discount on a car is 20% of the original price given the specified conditions -/
theorem car_discount_proof (P : ℝ) (D : ℝ) : 
  P > 0 →  -- Assuming positive original price
  D > 0 →  -- Assuming positive discount
  D < P →  -- Discount is less than original price
  (P - D + 0.45 * (P - D)) = (P + 0.16 * P) →  -- Selling price equation
  D = 0.2 * P :=  -- Conclusion: discount is 20% of original price
by sorry  -- Proof is omitted

end NUMINAMATH_CALUDE_car_discount_proof_l399_39906


namespace NUMINAMATH_CALUDE_sequence_sum_divisible_by_37_l399_39953

/-- Represents a three-digit integer -/
structure ThreeDigitInt where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Represents a sequence of four three-digit integers satisfying the given conditions -/
structure SpecialSequence where
  term1 : ThreeDigitInt
  term2 : ThreeDigitInt
  term3 : ThreeDigitInt
  term4 : ThreeDigitInt
  valid_cycle : 
    term2.hundreds = term1.tens ∧ term2.tens = term1.units ∧
    term3.hundreds = term2.tens ∧ term3.tens = term2.units ∧
    term4.hundreds = term3.tens ∧ term4.tens = term3.units ∧
    term1.hundreds = term4.tens ∧ term1.tens = term4.units

/-- The sum of all terms in a SpecialSequence -/
def sequenceSum (seq : SpecialSequence) : Nat :=
  (seq.term1.hundreds * 100 + seq.term1.tens * 10 + seq.term1.units) +
  (seq.term2.hundreds * 100 + seq.term2.tens * 10 + seq.term2.units) +
  (seq.term3.hundreds * 100 + seq.term3.tens * 10 + seq.term3.units) +
  (seq.term4.hundreds * 100 + seq.term4.tens * 10 + seq.term4.units)

theorem sequence_sum_divisible_by_37 (seq : SpecialSequence) : 
  37 ∣ sequenceSum seq := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_divisible_by_37_l399_39953


namespace NUMINAMATH_CALUDE_extended_tile_ratio_l399_39967

/-- The ratio of black tiles to white tiles in an extended rectangular pattern -/
theorem extended_tile_ratio (orig_width orig_height : ℕ) 
  (orig_black orig_white : ℕ) : 
  orig_width = 5 → 
  orig_height = 6 → 
  orig_black = 12 → 
  orig_white = 18 → 
  (orig_black : ℚ) / ((orig_white : ℚ) + 2 * (orig_width + orig_height + 2)) = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_extended_tile_ratio_l399_39967


namespace NUMINAMATH_CALUDE_necessary_condition_k_l399_39913

theorem necessary_condition_k (k : ℝ) : 
  (∀ x : ℝ, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) ∧
  (∃ x : ℝ, (x < k ∨ x > k + 2) ∧ ¬(-4 < x ∧ x < 1)) ↔
  k ≤ -6 ∨ k ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_k_l399_39913


namespace NUMINAMATH_CALUDE_candy_distribution_l399_39950

theorem candy_distribution (total_candy : ℕ) (candy_per_bag : ℕ) (num_bags : ℕ) : 
  total_candy = 42 → 
  candy_per_bag = 21 → 
  total_candy = num_bags * candy_per_bag → 
  num_bags = 2 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l399_39950


namespace NUMINAMATH_CALUDE_factor_expression_l399_39997

theorem factor_expression (y : ℝ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l399_39997


namespace NUMINAMATH_CALUDE_factor_tree_value_l399_39926

theorem factor_tree_value : ∀ (A B C D E : ℕ),
  A = B * C →
  B = 3 * D →
  D = 3 * 2 →
  C = 5 * E →
  E = 5 * 2 →
  A = 900 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_value_l399_39926


namespace NUMINAMATH_CALUDE_number_problem_l399_39934

theorem number_problem (x : ℝ) : 19 * (x - 174) = 3439 → x = 355 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l399_39934


namespace NUMINAMATH_CALUDE_discount_is_ten_percent_l399_39943

/-- Calculates the discount percentage on a retail price given wholesale price, retail price, and profit percentage. -/
def discount_percentage (wholesale_price retail_price profit_percentage : ℚ) : ℚ :=
  let profit := wholesale_price * profit_percentage
  let selling_price := wholesale_price + profit
  let discount_amount := retail_price - selling_price
  (discount_amount / retail_price) * 100

/-- Theorem stating that the discount percentage is 10% given the problem conditions. -/
theorem discount_is_ten_percent :
  discount_percentage 108 144 0.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_ten_percent_l399_39943


namespace NUMINAMATH_CALUDE_sum_and_decimal_shift_l399_39951

theorem sum_and_decimal_shift (A B : ℝ) (h1 : A + B = 13.2) (h2 : 10 * A = B) : A = 1.2 ∧ B = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_decimal_shift_l399_39951


namespace NUMINAMATH_CALUDE_cube_surface_area_l399_39955

/-- The surface area of a cube with edge length 7 cm is 294 cm² -/
theorem cube_surface_area :
  let edge_length : ℝ := 7
  let face_area : ℝ := edge_length * edge_length
  let num_faces : ℕ := 6
  edge_length * edge_length * num_faces = 294 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l399_39955


namespace NUMINAMATH_CALUDE_telescope_cost_l399_39925

theorem telescope_cost (joan karl : ℕ) 
  (h1 : joan + karl = 400)
  (h2 : 2 * joan = karl + 74) : 
  joan = 158 := by
sorry

end NUMINAMATH_CALUDE_telescope_cost_l399_39925


namespace NUMINAMATH_CALUDE_area_to_paint_is_132_l399_39981

/-- The area to be painted on a wall, given its dimensions and the dimensions of an area that doesn't need painting. -/
def areaToPaint (wallHeight wallLength paintingWidth paintingHeight : ℕ) : ℕ :=
  wallHeight * wallLength - paintingWidth * paintingHeight

/-- Theorem stating that the area to be painted is 132 square feet for the given dimensions. -/
theorem area_to_paint_is_132 :
  areaToPaint 10 15 3 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_is_132_l399_39981


namespace NUMINAMATH_CALUDE_triangle_angle_C_l399_39986

theorem triangle_angle_C (a b c : ℝ) (A B C : ℝ) : 
  a + b = Real.sqrt 2 →
  (1/2) * a * b * Real.sin C = (1/6) * Real.sin C →
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →
  C = π/3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l399_39986


namespace NUMINAMATH_CALUDE_original_number_proof_l399_39962

theorem original_number_proof (x : ℝ) : 1 + 1/x = 8/3 → x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l399_39962


namespace NUMINAMATH_CALUDE_smallest_angle_tangent_equation_l399_39911

theorem smallest_angle_tangent_equation (x : Real) : 
  (x > 0) →
  (Real.tan (6 * x * Real.pi / 180) = 
    (Real.cos (2 * x * Real.pi / 180) - Real.sin (2 * x * Real.pi / 180)) / 
    (Real.cos (2 * x * Real.pi / 180) + Real.sin (2 * x * Real.pi / 180))) →
  x = 5.625 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_tangent_equation_l399_39911


namespace NUMINAMATH_CALUDE_unique_solution_condition_l399_39907

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+8)*(x-6) = -55 + k*x) ↔ (k = -10 + 2*Real.sqrt 21 ∨ k = -10 - 2*Real.sqrt 21) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l399_39907


namespace NUMINAMATH_CALUDE_max_sum_of_abs_on_unit_sphere_l399_39963

theorem max_sum_of_abs_on_unit_sphere :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |x| + |y| + |z| ≤ M) ∧
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 1 ∧ |x| + |y| + |z| = M) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_abs_on_unit_sphere_l399_39963


namespace NUMINAMATH_CALUDE_cars_between_15k_and_20k_l399_39995

/-- Given a car dealership with the following properties:
  * There are 3000 cars in total
  * 15% of cars cost less than $15000
  * 40% of cars cost more than $20000
  Prove that the number of cars costing between $15000 and $20000 is 1350 -/
theorem cars_between_15k_and_20k (total_cars : ℕ) (percent_less_15k : ℚ) (percent_more_20k : ℚ) 
  (h_total : total_cars = 3000)
  (h_less_15k : percent_less_15k = 15 / 100)
  (h_more_20k : percent_more_20k = 40 / 100) :
  total_cars - (total_cars * percent_less_15k).floor - (total_cars * percent_more_20k).floor = 1350 := by
  sorry

end NUMINAMATH_CALUDE_cars_between_15k_and_20k_l399_39995


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l399_39941

/-- Represents the number of people in each age group -/
structure AgeGroups where
  over40 : ℕ
  between30and40 : ℕ
  under30 : ℕ

/-- Represents the sample sizes for each age group -/
structure SampleSizes where
  over40 : ℕ
  between30and40 : ℕ
  under30 : ℕ

/-- Calculates the stratified sample sizes given the total population, total sample size, and age group sizes -/
def calculateStratifiedSample (totalPopulation : ℕ) (totalSampleSize : ℕ) (ageGroups : AgeGroups) : SampleSizes :=
  let ratio := totalSampleSize / totalPopulation
  { over40 := ageGroups.over40 * ratio,
    between30and40 := ageGroups.between30and40 * ratio,
    under30 := ageGroups.under30 * ratio }

theorem stratified_sample_theorem (totalPopulation : ℕ) (totalSampleSize : ℕ) (ageGroups : AgeGroups) :
  totalPopulation = 300 →
  totalSampleSize = 30 →
  ageGroups.over40 = 50 →
  ageGroups.between30and40 = 150 →
  ageGroups.under30 = 100 →
  let sample := calculateStratifiedSample totalPopulation totalSampleSize ageGroups
  sample.over40 = 5 ∧ sample.between30and40 = 15 ∧ sample.under30 = 10 :=
by sorry


end NUMINAMATH_CALUDE_stratified_sample_theorem_l399_39941


namespace NUMINAMATH_CALUDE_xiao_ming_english_score_l399_39940

/-- Calculates the weighted average score given three component scores and their weights -/
def weighted_average (listening_score language_score written_score : ℚ) 
  (listening_weight language_weight written_weight : ℕ) : ℚ :=
  (listening_score * listening_weight + language_score * language_weight + written_score * written_weight) / 
  (listening_weight + language_weight + written_weight)

/-- Theorem stating that Xiao Ming's English score is 92.6 given his component scores and the weighting ratio -/
theorem xiao_ming_english_score : 
  weighted_average 92 90 95 3 3 4 = 92.6 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_english_score_l399_39940


namespace NUMINAMATH_CALUDE_complex_equality_l399_39929

theorem complex_equality (a b : ℝ) (h : Complex.I * (a + Complex.I) = b - Complex.I) : a - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l399_39929


namespace NUMINAMATH_CALUDE_adjacent_product_geometric_sequence_l399_39958

/-- Given a geometric sequence with common ratio q, prove that the sequence formed by
    the product of adjacent terms is a geometric sequence with common ratio q² -/
theorem adjacent_product_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  ∀ n : ℕ, (a (n + 1) * a (n + 2)) = q^2 * (a n * a (n + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_adjacent_product_geometric_sequence_l399_39958


namespace NUMINAMATH_CALUDE_field_trip_adults_l399_39979

theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 8 → num_students = 22 → num_vans = 3 → 
  (num_vans * van_capacity - num_students : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l399_39979


namespace NUMINAMATH_CALUDE_fib_sum_equals_five_nineteenths_l399_39936

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the Fibonacci series divided by powers of 5 -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 5^n

theorem fib_sum_equals_five_nineteenths : fibSum = 5 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_equals_five_nineteenths_l399_39936


namespace NUMINAMATH_CALUDE_letter_Y_satisfies_condition_l399_39999

def date_C (d : ℕ) : ℕ := d
def date_A (d : ℕ) : ℕ := d + 2
def date_B (d : ℕ) : ℕ := d + 8
def date_Y (d : ℕ) : ℕ := d + 10

theorem letter_Y_satisfies_condition (d : ℕ) :
  date_A d + date_B d = date_C d + date_Y d :=
by sorry

end NUMINAMATH_CALUDE_letter_Y_satisfies_condition_l399_39999


namespace NUMINAMATH_CALUDE_dragon_defeat_certain_l399_39935

/-- Represents the state of the dragon's heads -/
inductive DragonState
| Heads (n : ℕ)

/-- Represents the possible outcomes after chopping off a head -/
inductive ChopResult
| TwoNewHeads
| OneNewHead
| NoNewHeads

/-- The probability distribution of chop results -/
def chop_prob : ChopResult → ℚ
| ChopResult.TwoNewHeads => 1/4
| ChopResult.OneNewHead => 1/3
| ChopResult.NoNewHeads => 5/12

/-- The state transition function after chopping off a head -/
def next_state (s : DragonState) (r : ChopResult) : DragonState :=
  match s with
  | DragonState.Heads n =>
    match r with
    | ChopResult.TwoNewHeads => DragonState.Heads (n + 1)
    | ChopResult.OneNewHead => DragonState.Heads n
    | ChopResult.NoNewHeads => DragonState.Heads (n - 1)

/-- The probability of eventually defeating the dragon -/
noncomputable def defeat_prob (s : DragonState) : ℝ := sorry

/-- The theorem stating that the probability of defeating the dragon is 1 -/
theorem dragon_defeat_certain :
  defeat_prob (DragonState.Heads 3) = 1 := by sorry

end NUMINAMATH_CALUDE_dragon_defeat_certain_l399_39935


namespace NUMINAMATH_CALUDE_perfect_square_sum_l399_39931

theorem perfect_square_sum (x y : ℕ) : 
  (∃ z : ℕ, 3^x + 7^y = z^2) → 
  Even y → 
  x = 1 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l399_39931


namespace NUMINAMATH_CALUDE_triangle_angle_sum_contradiction_l399_39939

theorem triangle_angle_sum_contradiction :
  ∀ (left right top : ℝ),
  right = 60 →
  left = 2 * right →
  top = 70 →
  left + right + top ≠ 180 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_contradiction_l399_39939


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l399_39904

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) : 
  let original_area := 6 * L^2
  let new_edge_length := 1.3 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 69 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l399_39904


namespace NUMINAMATH_CALUDE_two_segment_trip_avg_speed_l399_39983

/-- Calculates the average speed for a two-segment trip -/
theorem two_segment_trip_avg_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 40) 
  (h2 : speed1 = 30) 
  (h3 : distance2 = 40) 
  (h4 : speed2 = 15) : 
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 20 := by
  sorry

#check two_segment_trip_avg_speed

end NUMINAMATH_CALUDE_two_segment_trip_avg_speed_l399_39983


namespace NUMINAMATH_CALUDE_T_is_three_rays_l399_39956

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set T as described in the problem -/
def T : Set Point :=
  {p : Point | (4 = p.x + 1 ∧ p.y - 5 ≤ 4) ∨
               (4 = p.y - 5 ∧ p.x + 1 ≤ 4) ∨
               (p.x + 1 = p.y - 5 ∧ 4 ≤ p.x + 1)}

/-- A ray starting from a point in a given direction -/
structure Ray where
  start : Point
  direction : ℝ × ℝ

/-- The three rays that should describe T -/
def threeRays : List Ray :=
  [{ start := ⟨3, 9⟩, direction := (0, -1) },   -- Vertically downward
   { start := ⟨3, 9⟩, direction := (-1, 0) },   -- Horizontally leftward
   { start := ⟨3, 9⟩, direction := (1, 1) }]    -- Diagonally upward

/-- Theorem stating that T is equivalent to three rays with a common point -/
theorem T_is_three_rays : 
  ∀ p : Point, p ∈ T ↔ ∃ r ∈ threeRays, ∃ t : ℝ, t ≥ 0 ∧ 
    p.x = r.start.x + t * r.direction.1 ∧ 
    p.y = r.start.y + t * r.direction.2 :=
sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l399_39956


namespace NUMINAMATH_CALUDE_total_triangles_is_18_l399_39937

/-- Represents a figure with different types of triangles -/
structure TriangleFigure where
  smallest : Nat
  medium : Nat
  largest : Nat

/-- Calculates the total number of triangles in a TriangleFigure -/
def totalTriangles (figure : TriangleFigure) : Nat :=
  figure.smallest + figure.medium + figure.largest

/-- The given figure with 6 smallest, 7 medium, and 5 largest triangles -/
def givenFigure : TriangleFigure :=
  { smallest := 6, medium := 7, largest := 5 }

/-- Theorem stating that the total number of triangles in the given figure is 18 -/
theorem total_triangles_is_18 : totalTriangles givenFigure = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_18_l399_39937


namespace NUMINAMATH_CALUDE_dime_count_proof_l399_39917

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the number of dimes given the total amount, number of quarters, and number of nickels -/
def calculate_dimes (total_amount : ℕ) (num_quarters : ℕ) (num_nickels : ℕ) : ℕ :=
  (total_amount * cents_per_dollar - (num_quarters * quarter_value + num_nickels * nickel_value)) / dime_value

theorem dime_count_proof (total_amount : ℕ) (num_quarters : ℕ) (num_nickels : ℕ) 
  (h1 : total_amount = 4)
  (h2 : num_quarters = 10)
  (h3 : num_nickels = 6) :
  calculate_dimes total_amount num_quarters num_nickels = 12 := by
  sorry

end NUMINAMATH_CALUDE_dime_count_proof_l399_39917


namespace NUMINAMATH_CALUDE_intersecting_line_theorem_l399_39918

/-- A line passing through (a, 0) intersecting y^2 = 4x at P and Q -/
structure IntersectingLine (a : ℝ) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  line_through_a : P.2 / (P.1 - a) = Q.2 / (Q.1 - a)
  P_on_parabola : P.2^2 = 4 * P.1
  Q_on_parabola : Q.2^2 = 4 * Q.1

/-- The reciprocal sum of squared distances is constant -/
def constant_sum (a : ℝ) :=
  ∃ (k : ℝ), ∀ (l : IntersectingLine a),
    1 / ((l.P.1 - a)^2 + l.P.2^2) + 1 / ((l.Q.1 - a)^2 + l.Q.2^2) = k

/-- If the reciprocal sum of squared distances is constant, then a = 2 -/
theorem intersecting_line_theorem :
  ∀ a : ℝ, constant_sum a → a = 2 := by sorry

end NUMINAMATH_CALUDE_intersecting_line_theorem_l399_39918


namespace NUMINAMATH_CALUDE_carnival_activity_order_l399_39923

/-- Represents an activity at the school carnival -/
inductive Activity
  | Dodgeball
  | MagicShow
  | PettingZoo
  | FacePainting

/-- Returns the popularity of an activity as a fraction -/
def popularity (a : Activity) : Rat :=
  match a with
  | Activity.Dodgeball => 3 / 8
  | Activity.MagicShow => 9 / 24
  | Activity.PettingZoo => 1 / 3
  | Activity.FacePainting => 5 / 12

/-- Checks if one activity is more popular than another -/
def morePopularThan (a b : Activity) : Prop :=
  popularity a > popularity b

theorem carnival_activity_order :
  morePopularThan Activity.FacePainting Activity.Dodgeball ∧
  morePopularThan Activity.Dodgeball Activity.MagicShow ∧
  morePopularThan Activity.MagicShow Activity.PettingZoo :=
by sorry

end NUMINAMATH_CALUDE_carnival_activity_order_l399_39923


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l399_39984

/-- 
Given an ellipse with equation x²/25 + y²/16 = 1, 
if the distance from a point P on the ellipse to one focus is 3, 
then the distance from P to the other focus is 7.
-/
theorem ellipse_foci_distance (x y : ℝ) (P : ℝ × ℝ) :
  x^2 / 25 + y^2 / 16 = 1 →  -- Ellipse equation
  P.1^2 / 25 + P.2^2 / 16 = 1 →  -- Point P is on the ellipse
  ∃ (F₁ F₂ : ℝ × ℝ), -- There exist two foci F₁ and F₂
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 3 ∨
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 3) →
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 7 ∨
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l399_39984


namespace NUMINAMATH_CALUDE_point_p_coordinates_l399_39994

-- Define a 2D point
structure Point2D where
  x : ℚ
  y : ℚ

-- Define a 2D vector
structure Vector2D where
  x : ℚ
  y : ℚ

-- Define vector between two points
def vectorBetween (p1 p2 : Point2D) : Vector2D :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

-- Define scalar multiplication for vectors
def scalarMult (k : ℚ) (v : Vector2D) : Vector2D :=
  { x := k * v.x, y := k * v.y }

theorem point_p_coordinates 
  (m n p : Point2D)
  (h1 : m = { x := 3, y := -2 })
  (h2 : n = { x := -5, y := -1 })
  (h3 : vectorBetween m p = scalarMult (1/3) (vectorBetween m n)) :
  p = { x := 1/3, y := -5/3 } := by
  sorry


end NUMINAMATH_CALUDE_point_p_coordinates_l399_39994


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l399_39912

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l399_39912


namespace NUMINAMATH_CALUDE_sarah_brother_apple_ratio_l399_39942

def sarah_apples : ℕ := 45
def brother_apples : ℕ := 9

theorem sarah_brother_apple_ratio :
  sarah_apples / brother_apples = 5 :=
sorry

end NUMINAMATH_CALUDE_sarah_brother_apple_ratio_l399_39942


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l399_39957

theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), (x - 2) / (2 * x - 1) + 1 = 3 / (2 - 4 * x) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l399_39957


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l399_39982

/-- The average price Sandy paid per book given her purchases from two shops -/
theorem sandy_average_book_price (books1 : ℕ) (price1 : ℚ) (books2 : ℕ) (price2 : ℚ) 
  (h1 : books1 = 65)
  (h2 : price1 = 1380)
  (h3 : books2 = 55)
  (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2 : ℚ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sandy_average_book_price_l399_39982


namespace NUMINAMATH_CALUDE_ball_color_distribution_l399_39924

theorem ball_color_distribution :
  ∀ (blue red green : ℕ),
  blue + red + green = 15 →
  (blue = red + 1 ∧ blue = green + 5) ∨
  (blue = red + 1 ∧ red = green) ∨
  (red = green ∧ blue = green + 5) →
  blue = 7 ∧ red = 6 ∧ green = 2 := by
sorry

end NUMINAMATH_CALUDE_ball_color_distribution_l399_39924


namespace NUMINAMATH_CALUDE_future_cup_defense_l399_39921

/-- Represents the defensive statistics of a class --/
structure DefensiveStats where
  avgGoalsConceded : ℝ
  stdDevGoalsConceded : ℝ

/-- Determines if one class has better average defensive performance than another --/
def betterAverageDefense (a b : DefensiveStats) : Prop :=
  a.avgGoalsConceded > b.avgGoalsConceded

/-- Determines if one class has less stable defensive performance than another --/
def lessStableDefense (a b : DefensiveStats) : Prop :=
  a.stdDevGoalsConceded > b.stdDevGoalsConceded

/-- Determines if a class has relatively consistent defensive performance --/
def consistentDefense (a : DefensiveStats) : Prop :=
  a.stdDevGoalsConceded < 0.5

theorem future_cup_defense 
  (classA classB : DefensiveStats)
  (hA : classA.avgGoalsConceded = 1.9 ∧ classA.stdDevGoalsConceded = 0.3)
  (hB : classB.avgGoalsConceded = 1.3 ∧ classB.stdDevGoalsConceded = 1.2) :
  betterAverageDefense classA classB ∧ 
  lessStableDefense classB classA ∧ 
  consistentDefense classA := by
  sorry

end NUMINAMATH_CALUDE_future_cup_defense_l399_39921


namespace NUMINAMATH_CALUDE_rachel_piggy_bank_l399_39998

/-- The amount of money originally in Rachel's piggy bank -/
def original_amount : ℕ := 5

/-- The amount of money Rachel took from her piggy bank -/
def amount_taken : ℕ := 2

/-- The amount of money left in Rachel's piggy bank -/
def amount_left : ℕ := original_amount - amount_taken

theorem rachel_piggy_bank : amount_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_rachel_piggy_bank_l399_39998


namespace NUMINAMATH_CALUDE_whole_milk_fat_percentage_l399_39972

theorem whole_milk_fat_percentage :
  let reduced_fat_percentage : ℚ := 2
  let reduction_percentage : ℚ := 40
  let whole_milk_fat_percentage : ℚ := reduced_fat_percentage / (1 - reduction_percentage / 100)
  whole_milk_fat_percentage = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_whole_milk_fat_percentage_l399_39972


namespace NUMINAMATH_CALUDE_rectangle_width_l399_39973

/-- Given a rectangle with perimeter 16 cm and width 2 cm longer than length, prove its width is 5 cm -/
theorem rectangle_width (length width : ℝ) : 
  (2 * (length + width) = 16) →  -- Perimeter is 16 cm
  (width = length + 2) →         -- Width is 2 cm longer than length
  width = 5 := by               -- Prove width is 5 cm
sorry

end NUMINAMATH_CALUDE_rectangle_width_l399_39973


namespace NUMINAMATH_CALUDE_partnership_investment_time_l399_39959

/-- A partnership problem with three partners A, B, and C --/
theorem partnership_investment_time (x : ℝ) : 
  let total_investment := x * 12 + 2 * x * 6 + 3 * x * (12 - m)
  let m := 12 - (36 * x - 24 * x) / (3 * x)
  x > 0 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_time_l399_39959


namespace NUMINAMATH_CALUDE_circle_pentagon_visibility_l399_39996

noncomputable def radius_of_circle (side_length : ℝ) (probability : ℝ) : ℝ :=
  (side_length * Real.sqrt ((5 - 2 * Real.sqrt 5) / 5)) / (2 * 0.9511)

theorem circle_pentagon_visibility 
  (r : ℝ) 
  (side_length : ℝ) 
  (probability : ℝ) 
  (h1 : side_length = 3) 
  (h2 : probability = 1/2) :
  r = radius_of_circle side_length probability :=
by sorry

end NUMINAMATH_CALUDE_circle_pentagon_visibility_l399_39996


namespace NUMINAMATH_CALUDE_equal_shaded_areas_condition_l399_39964

/-- Given a circle with radius s and an angle φ, where 0 < φ < π/4,
    this theorem states the necessary and sufficient condition for
    the equality of two specific areas related to the circle. --/
theorem equal_shaded_areas_condition (s : ℝ) (φ : ℝ) 
    (h1 : 0 < φ) (h2 : φ < π/4) (h3 : s > 0) :
  let sector_area := φ * s^2 / 2
  let triangle_area := s^2 * Real.tan φ / 2
  sector_area = triangle_area ↔ Real.tan φ = 3 * φ :=
sorry

end NUMINAMATH_CALUDE_equal_shaded_areas_condition_l399_39964


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l399_39993

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^4 + (m - 1)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^4 + (m - 1) * x + 1

/-- If f(x) = x^4 + (m - 1)x + 1 is an even function, then m = 1 -/
theorem even_function_implies_m_equals_one (m : ℝ) : IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l399_39993


namespace NUMINAMATH_CALUDE_floor_abs_sum_l399_39933

theorem floor_abs_sum : ⌊|(-7.9 : ℝ)|⌋ + |⌊(-7.9 : ℝ)⌋| = 15 := by sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l399_39933


namespace NUMINAMATH_CALUDE_daughter_age_in_three_years_l399_39961

/-- Given that 5 years ago a mother was twice as old as her daughter,
    and the mother is 41 years old now, prove that the daughter
    will be 26 years old in 3 years. -/
theorem daughter_age_in_three_years
  (mother_age_now : ℕ)
  (h1 : mother_age_now = 41)
  (h2 : mother_age_now - 5 = 2 * ((mother_age_now - 5) / 2)) :
  ((mother_age_now - 5) / 2) + 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_daughter_age_in_three_years_l399_39961


namespace NUMINAMATH_CALUDE_x_forty_percent_greater_than_88_l399_39988

theorem x_forty_percent_greater_than_88 :
  ∀ x : ℝ, x = 88 * (1 + 0.4) → x = 123.2 :=
by
  sorry

end NUMINAMATH_CALUDE_x_forty_percent_greater_than_88_l399_39988
