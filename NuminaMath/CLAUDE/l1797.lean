import Mathlib

namespace cosine_equality_l1797_179727

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (942 * π / 180) → n = 138 := by
  sorry

end cosine_equality_l1797_179727


namespace queenie_earnings_l1797_179762

/-- Calculates the total earnings for a worker given their daily rate, overtime rate, 
    number of days worked, and number of overtime hours. -/
def total_earnings (daily_rate : ℕ) (overtime_rate : ℕ) (days_worked : ℕ) (overtime_hours : ℕ) : ℕ :=
  daily_rate * days_worked + overtime_rate * overtime_hours

/-- Proves that Queenie's total earnings for 5 days of work with 4 hours overtime
    are equal to $770, given her daily rate of $150 and overtime rate of $5 per hour. -/
theorem queenie_earnings : total_earnings 150 5 5 4 = 770 := by
  sorry

end queenie_earnings_l1797_179762


namespace sum_of_square_roots_inequality_l1797_179772

theorem sum_of_square_roots_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab_c : a + b + c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end sum_of_square_roots_inequality_l1797_179772


namespace shape_D_symmetric_l1797_179774

-- Define the shape type
inductive Shape
| A
| B
| C
| D
| E

-- Define the property of being symmetric with respect to a horizontal line
def isSymmetric (s1 s2 : Shape) : Prop := sorry

-- Define the given shape
def givenShape : Shape := sorry

-- Theorem statement
theorem shape_D_symmetric : 
  isSymmetric givenShape Shape.D := by sorry

end shape_D_symmetric_l1797_179774


namespace division_problem_l1797_179703

theorem division_problem : ∃ x : ℝ, 550 - (104 / x) = 545 ∧ x = 20.8 := by
  sorry

end division_problem_l1797_179703


namespace segments_between_five_points_segments_between_five_points_proof_l1797_179743

/-- Given 5 points where no three are collinear, the number of segments needed to connect each pair of points is 10. -/
theorem segments_between_five_points : ℕ → Prop :=
  fun n => n = 5 → (∀ p1 p2 p3 : ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬Collinear p1 p2 p3) →
    (Nat.choose n 2 = 10)
  where
    Collinear (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) : Prop :=
      ∃ t : ℝ, p3 = p1 + t • (p2 - p1)

/-- Proof of the theorem -/
theorem segments_between_five_points_proof : segments_between_five_points 5 := by
  sorry

end segments_between_five_points_segments_between_five_points_proof_l1797_179743


namespace fencing_cost_theorem_l1797_179764

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Theorem: The total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_theorem :
  let length : ℝ := 63
  let breadth : ℝ := 37
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 63 37 26.50

end fencing_cost_theorem_l1797_179764


namespace greatest_number_odd_factors_under_200_l1797_179716

theorem greatest_number_odd_factors_under_200 :
  ∃ (n : ℕ), n < 200 ∧ n = 196 ∧ 
  (∀ m : ℕ, m < 200 → (∃ k : ℕ, m = k^2) → m ≤ n) ∧
  (∃ k : ℕ, n = k^2) :=
sorry

end greatest_number_odd_factors_under_200_l1797_179716


namespace sum_of_factors_l1797_179730

theorem sum_of_factors (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0 →
  a + b + c + d + e = 35 := by
sorry

end sum_of_factors_l1797_179730


namespace determine_origin_l1797_179765

/-- Given two points A and B in a 2D coordinate system, we can uniquely determine the origin. -/
theorem determine_origin (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) :
  ∃! O : ℝ × ℝ, O = (0, 0) ∧ 
  (O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2 = (O.1 - B.1) ^ 2 + (O.2 - B.2) ^ 2 ∧
  (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (O.1 - A.1) ^ 2 + (O.2 - A.2) ^ 2 + (O.1 - B.1) ^ 2 + (O.2 - B.2) ^ 2 :=
by sorry


end determine_origin_l1797_179765


namespace cubic_factorization_l1797_179792

theorem cubic_factorization (a : ℝ) : a^3 - 16*a = a*(a+4)*(a-4) := by
  sorry

end cubic_factorization_l1797_179792


namespace bucket_capacity_first_case_l1797_179726

/-- The capacity of a bucket in the first case, given the following conditions:
  - 22 buckets of water fill a tank in the first case
  - 33 buckets of water fill the same tank in the second case
  - In the second case, each bucket has a capacity of 9 litres
-/
theorem bucket_capacity_first_case : 
  ∀ (capacity_first : ℝ) (tank_volume : ℝ),
  22 * capacity_first = tank_volume →
  33 * 9 = tank_volume →
  capacity_first = 13.5 := by
sorry

end bucket_capacity_first_case_l1797_179726


namespace problem_statement_l1797_179785

theorem problem_statement : 65 * 1515 - 25 * 1515 + 1515 = 62115 := by
  sorry

end problem_statement_l1797_179785


namespace candied_yams_ratio_l1797_179754

/-- The ratio of shoppers who buy candied yams to the total number of shoppers -/
theorem candied_yams_ratio 
  (packages_per_box : ℕ) 
  (boxes_ordered : ℕ) 
  (total_shoppers : ℕ) 
  (h1 : packages_per_box = 25)
  (h2 : boxes_ordered = 5)
  (h3 : total_shoppers = 375) : 
  (boxes_ordered * packages_per_box : ℚ) / total_shoppers = 1 / 3 := by
  sorry

end candied_yams_ratio_l1797_179754


namespace intersection_point_is_unique_l1797_179744

/-- Represents a 2D point -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form -/
structure ParametricLine where
  p : Point  -- Point on the line
  v : Point  -- Direction vector

/-- The first line -/
def line1 : ParametricLine :=
  { p := { x := 2, y := 2 },
    v := { x := 3, y := -4 } }

/-- The second line -/
def line2 : ParametricLine :=
  { p := { x := 7, y := -6 },
    v := { x := 5, y := 3 } }

/-- The claimed intersection point -/
def intersectionPoint : Point :=
  { x := 11, y := -886/87 }

/-- Theorem stating that the given point is the unique intersection of the two lines -/
theorem intersection_point_is_unique :
  ∃! t u : ℚ,
    line1.p.x + t * line1.v.x = intersectionPoint.x ∧
    line1.p.y + t * line1.v.y = intersectionPoint.y ∧
    line2.p.x + u * line2.v.x = intersectionPoint.x ∧
    line2.p.y + u * line2.v.y = intersectionPoint.y :=
  sorry

end intersection_point_is_unique_l1797_179744


namespace gcd_lcm_product_8_12_l1797_179715

theorem gcd_lcm_product_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end gcd_lcm_product_8_12_l1797_179715


namespace range_of_a_l1797_179735

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then x + 4 else x^2 - 2*x

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Icc (-5 : ℝ) (4 : ℝ) :=
sorry

end range_of_a_l1797_179735


namespace train_crossing_time_l1797_179723

/-- Proves that a train with given specifications takes 20 seconds to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (passing_time : ℝ) :
  train_length = 180 →
  platform_length = 270 →
  passing_time = 8 →
  let train_speed := train_length / passing_time
  let total_distance := train_length + platform_length
  let crossing_time := total_distance / train_speed
  crossing_time = 20 := by
sorry

end train_crossing_time_l1797_179723


namespace min_pencils_per_box_l1797_179782

/-- Represents a configuration of pencils in boxes -/
structure PencilConfiguration where
  num_boxes : Nat
  num_colors : Nat
  pencils_per_box : Nat

/-- Checks if a configuration satisfies the color requirement -/
def satisfies_color_requirement (config : PencilConfiguration) : Prop :=
  ∀ (subset : Finset (Fin config.num_boxes)), 
    subset.card = 4 → (subset.card * config.pencils_per_box ≥ config.num_colors)

/-- The main theorem stating the minimum number of pencils required -/
theorem min_pencils_per_box : 
  ∀ (config : PencilConfiguration),
    config.num_boxes = 6 ∧ 
    config.num_colors = 26 ∧ 
    satisfies_color_requirement config →
    config.pencils_per_box ≥ 13 := by
  sorry

end min_pencils_per_box_l1797_179782


namespace two_digit_number_sum_l1797_179748

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_nonzero : 1 ≤ tens ∧ tens ≤ 9
  units_bound : units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The reverse of a two-digit number -/
def TwoDigitNumber.reverse (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  tens_nonzero := by sorry
  units_bound := n.tens_nonzero.2

theorem two_digit_number_sum (n : TwoDigitNumber) :
  (n.value - n.reverse.value = 7 * (n.tens + n.units)) →
  (n.value + n.reverse.value = 99) := by
  sorry

end two_digit_number_sum_l1797_179748


namespace triangle_count_is_nine_l1797_179725

/-- Represents the triangular grid structure described in the problem -/
structure TriangularGrid :=
  (top_row : Nat)
  (middle_row : Nat)
  (bottom_row : Nat)
  (has_inverted_triangle : Bool)

/-- Calculates the total number of triangles in the given grid -/
def count_triangles (grid : TriangularGrid) : Nat :=
  let small_triangles := grid.top_row + grid.middle_row + grid.bottom_row
  let medium_triangles := if grid.top_row ≥ 3 then 1 else 0 +
                          if grid.middle_row + grid.bottom_row ≥ 3 then 1 else 0
  let large_triangle := if grid.has_inverted_triangle then 1 else 0
  small_triangles + medium_triangles + large_triangle

/-- The specific grid described in the problem -/
def problem_grid : TriangularGrid :=
  { top_row := 3,
    middle_row := 2,
    bottom_row := 1,
    has_inverted_triangle := true }

theorem triangle_count_is_nine :
  count_triangles problem_grid = 9 :=
sorry

end triangle_count_is_nine_l1797_179725


namespace mans_speed_against_current_l1797_179708

/-- Given a man's speed with the current and the speed of the current, 
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  2 * speed_with_current - 3 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 11.2 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 18 3.4 = 11.2 := by
  sorry

#eval speed_against_current 18 3.4

end mans_speed_against_current_l1797_179708


namespace arithmetic_seq_sum_l1797_179734

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence where S_9 = 72, a_2 + a_4 + a_9 = 24 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 9 = 72) :
  seq.a 2 + seq.a 4 + seq.a 9 = 24 := by
  sorry

end arithmetic_seq_sum_l1797_179734


namespace hospital_current_age_l1797_179760

/-- Represents the current age of Grant -/
def grants_current_age : ℕ := 25

/-- Represents the number of years in the future when the condition is met -/
def years_in_future : ℕ := 5

/-- Represents the fraction of the hospital's age that Grant will be in the future -/
def age_fraction : ℚ := 2/3

/-- Theorem stating that given the conditions, the current age of the hospital is 40 years -/
theorem hospital_current_age : 
  ∃ (hospital_age : ℕ), 
    (grants_current_age + years_in_future : ℚ) = age_fraction * (hospital_age + years_in_future : ℚ) ∧
    hospital_age = 40 := by
  sorry

end hospital_current_age_l1797_179760


namespace max_value_xyz_expression_l1797_179717

theorem max_value_xyz_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * (x + y + z)) / ((x + z)^2 * (y + z)^2) ≤ (1 : ℝ) / 4 := by
sorry

end max_value_xyz_expression_l1797_179717


namespace four_color_plane_partition_l1797_179794

-- Define the plane as ℝ × ℝ
def Plane := ℝ × ℝ

-- Define a partition of the plane into four subsets
def Partition (A B C D : Set Plane) : Prop :=
  (A ∪ B ∪ C ∪ D = Set.univ) ∧
  (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (A ∩ D = ∅) ∧
  (B ∩ C = ∅) ∧ (B ∩ D = ∅) ∧ (C ∩ D = ∅)

-- Define a circle in the plane
def Circle (center : Plane) (radius : ℝ) : Set Plane :=
  {p : Plane | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem four_color_plane_partition :
  ∃ (A B C D : Set Plane), Partition A B C D ∧
    ∀ (center : Plane) (radius : ℝ),
      (Circle center radius ∩ A).Nonempty ∧
      (Circle center radius ∩ B).Nonempty ∧
      (Circle center radius ∩ C).Nonempty ∧
      (Circle center radius ∩ D).Nonempty :=
by sorry


end four_color_plane_partition_l1797_179794


namespace petrol_price_equation_l1797_179798

/-- The original price of petrol per gallon -/
def P : ℝ := sorry

/-- The reduced price is 90% of the original price -/
def reduced_price : ℝ := 0.9 * P

/-- The equation representing the relationship between the original and reduced prices -/
theorem petrol_price_equation : 250 / reduced_price = 250 / P + 5 := by sorry

end petrol_price_equation_l1797_179798


namespace f_equals_g_l1797_179769

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 1
def g (t : ℝ) : ℝ := t^2 + 1

-- Theorem stating that f and g represent the same function
theorem f_equals_g : f = g := by sorry

end f_equals_g_l1797_179769


namespace product_evaluation_l1797_179777

theorem product_evaluation :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21523360 := by
  sorry

end product_evaluation_l1797_179777


namespace exponent_subtraction_l1797_179739

theorem exponent_subtraction : (-2)^3 - (-3)^2 = -17 := by
  sorry

end exponent_subtraction_l1797_179739


namespace brother_contribution_l1797_179796

/-- The number of wood pieces Alvin needs in total -/
def total_needed : ℕ := 376

/-- The number of wood pieces Alvin's friend gave him -/
def friend_gave : ℕ := 123

/-- The number of wood pieces Alvin still needs to gather -/
def still_needed : ℕ := 117

/-- The number of wood pieces Alvin's brother gave him -/
def brother_gave : ℕ := total_needed - friend_gave - still_needed

theorem brother_contribution : brother_gave = 136 := by
  sorry

end brother_contribution_l1797_179796


namespace salary_change_l1797_179787

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.4)
  let final_salary := decreased_salary * (1 + 0.4)
  (initial_salary - final_salary) / initial_salary = 0.16 := by
  sorry

end salary_change_l1797_179787


namespace find_number_l1797_179737

theorem find_number : ∃ x : ℕ,
  x % 18 = 6 ∧
  190 % 18 = 10 ∧
  x < 190 ∧
  (∀ y : ℕ, y % 18 = 6 → y < 190 → y ≤ x) ∧
  x = 186 := by
  sorry

end find_number_l1797_179737


namespace square_area_is_25_l1797_179756

-- Define the points
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (5, 6)

-- Define the square area function
def square_area (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  (dx * dx + dy * dy)

-- Theorem statement
theorem square_area_is_25 : square_area point1 point2 = 25 := by
  sorry

end square_area_is_25_l1797_179756


namespace carls_playground_area_l1797_179763

/-- Represents a rectangular playground with fence posts. -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℝ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the playground given its specifications. -/
def calculate_area (p : Playground) : ℝ :=
  ((p.short_side_posts - 1) * p.post_spacing) * ((p.long_side_posts - 1) * p.post_spacing)

/-- Theorem stating the area of Carl's playground is 324 square yards. -/
theorem carls_playground_area :
  ∃ (p : Playground),
    p.total_posts = 24 ∧
    p.post_spacing = 3 ∧
    p.long_side_posts = 2 * p.short_side_posts ∧
    calculate_area p = 324 := by
  sorry

end carls_playground_area_l1797_179763


namespace product_sum_relation_l1797_179742

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) := by
  sorry

end product_sum_relation_l1797_179742


namespace investment_growth_l1797_179714

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

end investment_growth_l1797_179714


namespace derivative_y_l1797_179750

noncomputable def y (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_y (x : ℝ) :
  deriv y x = Real.sin (2 * x) + 2 * x * Real.cos (2 * x) := by
  sorry

end derivative_y_l1797_179750


namespace florist_roses_l1797_179707

theorem florist_roses (initial_roses : ℕ) : 
  (initial_roses - 16 + 19 = 40) → initial_roses = 37 := by
  sorry

end florist_roses_l1797_179707


namespace correct_num_schools_l1797_179786

/-- The number of schools receiving soccer ball donations -/
def num_schools : ℕ := 2

/-- The number of classes per school -/
def classes_per_school : ℕ := 9

/-- The number of soccer balls per class -/
def balls_per_class : ℕ := 5

/-- The total number of soccer balls donated -/
def total_balls : ℕ := 90

/-- Theorem stating that the number of schools is correct -/
theorem correct_num_schools : 
  num_schools * classes_per_school * balls_per_class = total_balls :=
by sorry

end correct_num_schools_l1797_179786


namespace hire_year_proof_l1797_179791

/-- Rule of 70 provision: An employee can retire when their age plus years of employment total at least 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year an employee was hired -/
def hire_year : ℕ := 1966

/-- The age at which the employee was hired -/
def hire_age : ℕ := 30

/-- The year the employee becomes eligible to retire -/
def retirement_eligibility_year : ℕ := 2006

/-- Theorem stating that an employee hired at age 30, who becomes eligible to retire under the rule of 70 provision in 2006, was hired in 1966 -/
theorem hire_year_proof :
  rule_of_70 (hire_age + (retirement_eligibility_year - hire_year)) (retirement_eligibility_year - hire_year) ∧
  hire_year = 1966 :=
sorry

end hire_year_proof_l1797_179791


namespace ord₂_3n_minus_1_l1797_179720

-- Define ord₂ function
def ord₂ (i : ℤ) : ℕ :=
  if i = 0 then 0 else (i.natAbs.factors.filter (· = 2)).length

-- Main theorem
theorem ord₂_3n_minus_1 (n : ℕ) (h : n > 0) :
  (ord₂ (3^n - 1) = 1 ↔ n % 2 = 1) ∧
  (¬ ∃ n, ord₂ (3^n - 1) = 2) ∧
  (ord₂ (3^n - 1) = 3 ↔ n % 4 = 2) :=
sorry

-- Additional lemma to ensure ord₂(3ⁿ - 1) > 0 for n > 0
lemma ord₂_3n_minus_1_pos (n : ℕ) (h : n > 0) :
  ord₂ (3^n - 1) > 0 :=
sorry

end ord₂_3n_minus_1_l1797_179720


namespace negation_of_existence_negation_of_quadratic_equation_l1797_179704

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by
  sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) ↔ (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end negation_of_existence_negation_of_quadratic_equation_l1797_179704


namespace misplaced_sheets_count_l1797_179709

/-- Represents a booklet of printed notes -/
structure Booklet where
  total_pages : ℕ
  total_sheets : ℕ
  misplaced_sheets : ℕ
  avg_remaining : ℝ

/-- The theorem stating the number of misplaced sheets -/
theorem misplaced_sheets_count (b : Booklet) 
  (h1 : b.total_pages = 60)
  (h2 : b.total_sheets = 30)
  (h3 : b.avg_remaining = 21) :
  b.misplaced_sheets = 15 := by
  sorry

#check misplaced_sheets_count

end misplaced_sheets_count_l1797_179709


namespace direct_proportion_function_m_l1797_179711

theorem direct_proportion_function_m (m : ℝ) : 
  (m^2 - 3 = 1 ∧ m + 2 ≠ 0) ↔ m = 2 := by sorry

end direct_proportion_function_m_l1797_179711


namespace multiply_mixed_number_l1797_179736

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end multiply_mixed_number_l1797_179736


namespace students_above_115_l1797_179790

/-- Represents the score distribution of a math test -/
structure ScoreDistribution where
  mean : ℝ
  variance : ℝ
  normal : Bool

/-- Represents a class of students who took a math test -/
structure MathClass where
  size : ℕ
  scores : ScoreDistribution
  prob_95_to_105 : ℝ

/-- Calculates the number of students who scored above a given threshold -/
def students_above_threshold (c : MathClass) (threshold : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of students who scored above 115 in the given conditions -/
theorem students_above_115 (c : MathClass) 
  (h1 : c.size = 50)
  (h2 : c.scores.mean = 105)
  (h3 : c.scores.variance = 100)
  (h4 : c.scores.normal = true)
  (h5 : c.prob_95_to_105 = 0.32) :
  students_above_threshold c 115 = 9 :=
sorry

end students_above_115_l1797_179790


namespace calculation_proof_l1797_179700

theorem calculation_proof : 
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end calculation_proof_l1797_179700


namespace donovans_test_score_l1797_179740

theorem donovans_test_score (incorrect_answers : ℕ) (correct_percentage : ℚ) 
  (h1 : incorrect_answers = 13)
  (h2 : correct_percentage = 7292 / 10000) : 
  ∃ (correct_answers : ℕ), 
    (correct_answers : ℚ) / ((correct_answers : ℚ) + (incorrect_answers : ℚ)) = correct_percentage ∧ 
    correct_answers = 35 := by
  sorry

end donovans_test_score_l1797_179740


namespace automobile_distance_l1797_179795

/-- 
Given an automobile that travels 2a/5 feet in r seconds, 
this theorem proves that it will travel 40a/r yards in 5 minutes 
if this rate is maintained.
-/
theorem automobile_distance (a r : ℝ) (hr : r > 0) : 
  let rate_feet_per_second := (2 * a / 5) / r
  let rate_yards_per_second := rate_feet_per_second / 3
  let time_in_seconds := 5 * 60
  rate_yards_per_second * time_in_seconds = 40 * a / r := by
  sorry

end automobile_distance_l1797_179795


namespace real_roots_condition_l1797_179799

theorem real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4/3 :=
by sorry

end real_roots_condition_l1797_179799


namespace quadratic_complex_roots_l1797_179788

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = -1 + 2*I ∧ 
  z₂ = -3 - 2*I ∧ 
  (z₁^2 + 2*z₁ = -3 + 4*I) ∧ 
  (z₂^2 + 2*z₂ = -3 + 4*I) := by
sorry

end quadratic_complex_roots_l1797_179788


namespace hemisphere_surface_area_l1797_179731

/-- The total surface area of a hemisphere with base area 225π is 675π. -/
theorem hemisphere_surface_area : 
  ∀ r : ℝ, 
  r > 0 → 
  π * r^2 = 225 * π → 
  2 * π * r^2 + π * r^2 = 675 * π :=
by
  sorry

end hemisphere_surface_area_l1797_179731


namespace triangle_properties_l1797_179728

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Theorem statement
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.c = 2) 
  (h2 : t.A = π/3) : 
  t.a * Real.sin t.C = Real.sqrt 3 ∧ 
  1 + Real.sqrt 3 < t.a + t.b ∧ 
  t.a + t.b < 4 + 2 * Real.sqrt 3 := by
  sorry

end triangle_properties_l1797_179728


namespace complex_arithmetic_equality_l1797_179753

theorem complex_arithmetic_equality : 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001 = 76802 := by
  sorry

end complex_arithmetic_equality_l1797_179753


namespace set_intersection_theorem_l1797_179793

-- Define the sets P and Q
def P : Set ℝ := {x | x - 1 ≤ 0}
def Q : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

-- State the theorem
theorem set_intersection_theorem : (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end set_intersection_theorem_l1797_179793


namespace quadratic_equation_solution_l1797_179705

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * x - 1
  ∃ x1 x2 : ℝ, x1 = (2 + Real.sqrt 6) / 2 ∧ 
              x2 = (2 - Real.sqrt 6) / 2 ∧ 
              f x1 = 0 ∧ f x2 = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 :=
by sorry

end quadratic_equation_solution_l1797_179705


namespace jogging_ninth_day_l1797_179752

def minutes_jogged_6_days : ℕ := 6 * 80
def minutes_jogged_2_days : ℕ := 2 * 105
def total_minutes_8_days : ℕ := minutes_jogged_6_days + minutes_jogged_2_days
def desired_average : ℕ := 100
def total_days : ℕ := 9

theorem jogging_ninth_day :
  desired_average * total_days - total_minutes_8_days = 210 := by
  sorry

end jogging_ninth_day_l1797_179752


namespace least_possible_b_l1797_179706

-- Define a structure for our triangle
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ
  is_prime_a : Nat.Prime a
  is_prime_b : Nat.Prime b
  a_gt_b : a > b
  angle_sum : a + 2 * b = 180

-- Define the theorem
theorem least_possible_b (t : IsoscelesTriangle) : 
  (∀ t' : IsoscelesTriangle, t'.b ≥ t.b) → t.b = 19 :=
by sorry

end least_possible_b_l1797_179706


namespace floor_equation_solutions_l1797_179776

theorem floor_equation_solutions :
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    ⌊(2.018 : ℝ) * p.1⌋ + ⌊(5.13 : ℝ) * p.2⌋ = 24) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 3 :=
by sorry

end floor_equation_solutions_l1797_179776


namespace candy_bar_price_is_correct_l1797_179746

/-- The price of a candy bar in dollars -/
def candy_bar_price : ℝ := 2

/-- The price of a bag of chips in dollars -/
def chips_price : ℝ := 0.5

/-- The number of students -/
def num_students : ℕ := 5

/-- The total amount needed for all students in dollars -/
def total_amount : ℝ := 15

/-- The number of candy bars each student gets -/
def candy_bars_per_student : ℕ := 1

/-- The number of bags of chips each student gets -/
def chips_per_student : ℕ := 2

theorem candy_bar_price_is_correct : 
  candy_bar_price = 2 :=
by sorry

end candy_bar_price_is_correct_l1797_179746


namespace simplify_expression_l1797_179747

theorem simplify_expression (x : ℝ) : (3 * x + 8) + (50 * x + 25) = 53 * x + 33 := by
  sorry

end simplify_expression_l1797_179747


namespace product_remainder_zero_l1797_179729

theorem product_remainder_zero : (4251 * 7396 * 4625) % 10 = 0 := by
  sorry

end product_remainder_zero_l1797_179729


namespace inequalities_proof_l1797_179702

theorem inequalities_proof :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 ≥ a*b + a*c + b*c) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end inequalities_proof_l1797_179702


namespace apple_ratio_l1797_179732

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := 2 * monday_apples

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := 9

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := total_apples - (monday_apples + tuesday_apples + wednesday_apples + friday_apples)

theorem apple_ratio : thursday_apples = 4 * friday_apples := by sorry

end apple_ratio_l1797_179732


namespace collinear_points_a_equals_4_l1797_179780

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Given three points A(a,2), B(5,1), and C(-4,2a) are collinear, prove that a = 4. -/
theorem collinear_points_a_equals_4 (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 4 := by
  sorry


end collinear_points_a_equals_4_l1797_179780


namespace garden_length_l1797_179713

/-- Proves that a rectangular garden with length twice its width and perimeter 300 yards has a length of 100 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- Length is twice the width
  2 * length + 2 * width = 300 →  -- Perimeter is 300 yards
  length = 100 := by  -- Prove that length is 100 yards
sorry

end garden_length_l1797_179713


namespace triangle_k_range_l1797_179767

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three lines form a triangle -/
def form_triangle (l₁ l₂ l₃ : Line) : Prop :=
  ∀ (x y : ℝ), (l₁.a * x + l₁.b * y + l₁.c = 0 ∧ 
                l₂.a * x + l₂.b * y + l₂.c = 0 ∧
                l₃.a * x + l₃.b * y + l₃.c = 0) → False

/-- The theorem stating the range of k for which the given lines form a triangle -/
theorem triangle_k_range :
  ∀ (k : ℝ),
  let l₁ : Line := ⟨1, -1, 0⟩
  let l₂ : Line := ⟨1, 1, -2⟩
  let l₃ : Line := ⟨5, -k, -15⟩
  form_triangle l₁ l₂ l₃ ↔ k ≠ 5 ∧ k ≠ -5 ∧ k ≠ -10 :=
sorry

end triangle_k_range_l1797_179767


namespace tank_capacity_l1797_179771

theorem tank_capacity (x : ℚ) 
  (h1 : 2/3 * x - 15 = 1/3 * x) : x = 45 := by
  sorry

end tank_capacity_l1797_179771


namespace space_shuttle_speed_l1797_179721

-- Define the speed in kilometers per hour
def speed_kmh : ℝ := 14400

-- Define the conversion factor from hours to seconds
def seconds_per_hour : ℝ := 3600

-- The theorem to prove
theorem space_shuttle_speed : speed_kmh / seconds_per_hour = 4 := by
  sorry

end space_shuttle_speed_l1797_179721


namespace largest_unformable_amount_correct_l1797_179745

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ := {3*n - 2, 6*n - 1, 6*n + 2, 6*n + 5}

/-- Predicate to check if an amount can be formed using given coin denominations -/
def is_formable (amount : ℕ) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), amount = a*(3*n - 2) + b*(6*n - 1) + c*(6*n + 2) + d*(6*n + 5)

/-- The largest amount that cannot be formed using the coin denominations -/
def largest_unformable_amount (n : ℕ) : ℕ := 6*n^2 - 4*n - 3

/-- Main theorem: The largest amount that cannot be formed is 6n^2 - 4n - 3 -/
theorem largest_unformable_amount_correct (n : ℕ) :
  (∀ k > largest_unformable_amount n, is_formable k n) ∧
  ¬is_formable (largest_unformable_amount n) n :=
sorry

end largest_unformable_amount_correct_l1797_179745


namespace johns_purchase_cost_l1797_179719

/-- Calculates the total cost of John's purchase given the number of gum packs, candy bars, and the cost of a candy bar. -/
def total_cost (gum_packs : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℚ) : ℚ :=
  let gum_cost := candy_bar_cost / 2
  gum_packs * gum_cost + candy_bars * candy_bar_cost

/-- Proves that John's total cost for 2 packs of gum and 3 candy bars is $6, given that each candy bar costs $1.5 and gum costs half as much. -/
theorem johns_purchase_cost : total_cost 2 3 (3/2) = 6 := by
  sorry

end johns_purchase_cost_l1797_179719


namespace small_boxes_count_l1797_179733

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 525) 
  (h2 : chocolates_per_box = 25) : 
  total_chocolates / chocolates_per_box = 21 := by
  sorry

#check small_boxes_count

end small_boxes_count_l1797_179733


namespace expression_equality_l1797_179749

theorem expression_equality : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_equality_l1797_179749


namespace expression_value_l1797_179773

theorem expression_value (x : ℝ) (h : x^2 - 5*x - 2006 = 0) :
  ((x-2)^3 - (x-1)^2 + 1) / (x-2) = 2010 := by
  sorry

end expression_value_l1797_179773


namespace total_pens_count_l1797_179758

/-- The number of black pens bought by the teacher -/
def black_pens : ℕ := 7

/-- The number of blue pens bought by the teacher -/
def blue_pens : ℕ := 9

/-- The number of red pens bought by the teacher -/
def red_pens : ℕ := 5

/-- The total number of pens bought by the teacher -/
def total_pens : ℕ := black_pens + blue_pens + red_pens

theorem total_pens_count : total_pens = 21 := by
  sorry

end total_pens_count_l1797_179758


namespace calculation_proof_l1797_179775

theorem calculation_proof :
  (2/3 - 1/4 - 1/6) * 24 = 6 ∧
  (-2)^3 + (-9 + (-3)^2 * (1/3)) = -14 :=
by sorry

end calculation_proof_l1797_179775


namespace broken_line_isoperimetric_inequality_l1797_179779

/-- A non-self-intersecting broken line in a half-plane -/
structure BrokenLine where
  length : ℝ
  area : ℝ
  nonSelfIntersecting : Prop
  endsOnBoundary : Prop

/-- The isoperimetric inequality for the broken line -/
theorem broken_line_isoperimetric_inequality (b : BrokenLine) :
  b.area ≤ b.length^2 / (2 * Real.pi) := by
  sorry

end broken_line_isoperimetric_inequality_l1797_179779


namespace largest_multiple_of_15_with_8_and_0_l1797_179759

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 8 ∨ d = 0

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).filter (· = d) |>.length

theorem largest_multiple_of_15_with_8_and_0 :
  ∃ m : ℕ,
    m > 0 ∧
    15 ∣ m ∧
    is_valid_number m ∧
    count_digit m 8 = 6 ∧
    count_digit m 0 = 1 ∧
    m / 15 = 592592 ∧
    ∀ n : ℕ, n > m → ¬(15 ∣ n ∧ is_valid_number n) :=
  sorry

end largest_multiple_of_15_with_8_and_0_l1797_179759


namespace unique_divisor_sums_l1797_179768

def divisor_sums (n : ℕ+) : Finset ℕ :=
  (Finset.powerset (Nat.divisors n.val)).image (λ s => s.sum id)

def target_sums : Finset ℕ := {4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 46, 48, 50, 54, 60}

theorem unique_divisor_sums (n : ℕ+) : divisor_sums n = target_sums → n = 45 := by
  sorry

end unique_divisor_sums_l1797_179768


namespace ohara_triple_49_16_l1797_179783

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49, 16, x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end ohara_triple_49_16_l1797_179783


namespace fraction_equality_l1797_179784

theorem fraction_equality (w x y z : ℝ) (hw : w ≠ 0) 
  (h : (x + 6*y - 3*z) / (-3*x + 4*w) = (-2*y + z) / (x - w) ∧ 
       (-2*y + z) / (x - w) = 2/3) : 
  x / w = 2/3 := by
sorry

end fraction_equality_l1797_179784


namespace not_q_is_false_l1797_179712

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) := by
  sorry

end not_q_is_false_l1797_179712


namespace tv_show_average_episodes_l1797_179755

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

end tv_show_average_episodes_l1797_179755


namespace inequality_representation_l1797_179722

/-- 
Theorem: The inequality 3x - 2 > 0 correctly represents the statement 
"x is three times the difference between 2".
-/
theorem inequality_representation (x : ℝ) : 
  (3 * x - 2 > 0) ↔ (∃ y : ℝ, x = 3 * y ∧ y > 2) :=
sorry

end inequality_representation_l1797_179722


namespace inequality_proof_l1797_179738

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) ≥ 5/2 := by
  sorry

end inequality_proof_l1797_179738


namespace sequence_integer_count_l1797_179766

def sequence_term (n : ℕ) : ℚ :=
  8820 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = z

theorem sequence_integer_count :
  (∃ n : ℕ, n > 0 ∧ 
    (∀ k < n, is_integer (sequence_term k)) ∧
    ¬is_integer (sequence_term n)) ∧
  (∀ m : ℕ, m > 0 →
    (∀ k < m, is_integer (sequence_term k)) →
    ¬is_integer (sequence_term m) →
    m = 3) :=
sorry

end sequence_integer_count_l1797_179766


namespace sqrt_square_abs_two_div_sqrt_two_l1797_179757

-- Theorem 1: For any real number x, sqrt(x^2) = |x|
theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

-- Theorem 2: 2 / sqrt(2) = sqrt(2)
theorem two_div_sqrt_two : 2 / Real.sqrt 2 = Real.sqrt 2 := by sorry

end sqrt_square_abs_two_div_sqrt_two_l1797_179757


namespace average_glasses_per_box_l1797_179797

/-- Proves that given the specified conditions, the average number of glasses per box is 15 -/
theorem average_glasses_per_box : 
  ∀ (small_boxes large_boxes : ℕ),
  small_boxes > 0 →
  large_boxes = small_boxes + 16 →
  12 * small_boxes + 16 * large_boxes = 480 →
  (480 : ℚ) / (small_boxes + large_boxes) = 15 :=
by
  sorry

end average_glasses_per_box_l1797_179797


namespace two_std_dev_below_mean_is_9_6_l1797_179778

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : std_dev > 0

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 12 and standard deviation 1.2,
    the value that is exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_below_mean_is_9_6 :
  let d : NormalDistribution := ⟨12, 1.2, by norm_num⟩
  value_n_std_dev_below_mean d 2 = 9.6 := by
  sorry

end two_std_dev_below_mean_is_9_6_l1797_179778


namespace decimal_addition_l1797_179718

theorem decimal_addition : (4.358 + 3.892 : ℝ) = 8.250 := by sorry

end decimal_addition_l1797_179718


namespace equation_solution_l1797_179701

theorem equation_solution : 
  ∃ x : ℚ, x - (3 : ℚ) / 4 = (5 : ℚ) / 12 - (1 : ℚ) / 3 ∧ x = (5 : ℚ) / 6 := by
  sorry

end equation_solution_l1797_179701


namespace solve_equation_l1797_179751

theorem solve_equation (x : ℚ) (h : (1 / 4 : ℚ) - (1 / 6 : ℚ) = 4 / x) : x = 48 := by
  sorry

end solve_equation_l1797_179751


namespace total_eyes_in_pond_l1797_179770

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The number of eyes each snake has -/
def eyes_per_snake : ℕ := 2

/-- The number of eyes each alligator has -/
def eyes_per_alligator : ℕ := 2

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_snakes * eyes_per_snake + num_alligators * eyes_per_alligator

theorem total_eyes_in_pond : total_eyes = 56 := by
  sorry

end total_eyes_in_pond_l1797_179770


namespace prob_truth_or_lie_classroom_l1797_179761

/-- Represents the characteristics of a student population -/
structure StudentPopulation where
  total : ℕ
  truth_tellers : ℕ
  liars : ℕ
  both : ℕ
  avoiders : ℕ
  serious_liars_ratio : ℚ

/-- Calculates the probability of a student speaking truth or lying in a serious situation -/
def prob_truth_or_lie (pop : StudentPopulation) : ℚ :=
  let serious_liars := (pop.both : ℚ) * pop.serious_liars_ratio
  (pop.truth_tellers + pop.liars + serious_liars) / pop.total

/-- Theorem stating the probability of a student speaking truth or lying in a serious situation -/
theorem prob_truth_or_lie_classroom (pop : StudentPopulation) 
  (h1 : pop.total = 100)
  (h2 : pop.truth_tellers = 40)
  (h3 : pop.liars = 25)
  (h4 : pop.both = 15)
  (h5 : pop.avoiders = 20)
  (h6 : pop.serious_liars_ratio = 70 / 100)
  (h7 : pop.truth_tellers + pop.liars + pop.both + pop.avoiders = pop.total) :
  prob_truth_or_lie pop = 76 / 100 := by
  sorry

end prob_truth_or_lie_classroom_l1797_179761


namespace x_equals_y_plus_m_percent_l1797_179724

-- Define the relationship between x, y, and m
def is_m_percent_more (x y m : ℝ) : Prop :=
  x = y + (m / 100) * y

-- Theorem statement
theorem x_equals_y_plus_m_percent (x y m : ℝ) :
  is_m_percent_more x y m → x = (100 + m) / 100 * y := by
  sorry

end x_equals_y_plus_m_percent_l1797_179724


namespace max_sum_of_sides_l1797_179789

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

theorem max_sum_of_sides (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a = Real.sqrt 3 →
  f A = 1 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b + c ≤ 2 * Real.sqrt 3 :=
by sorry

end max_sum_of_sides_l1797_179789


namespace transmission_time_is_8_67_minutes_l1797_179741

/-- Represents the number of chunks in a regular block -/
def regular_block_chunks : ℕ := 800

/-- Represents the number of chunks in a large block -/
def large_block_chunks : ℕ := 1600

/-- Represents the number of regular blocks -/
def num_regular_blocks : ℕ := 70

/-- Represents the number of large blocks -/
def num_large_blocks : ℕ := 30

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- Calculates the total number of chunks to be transmitted -/
def total_chunks : ℕ := 
  num_regular_blocks * regular_block_chunks + num_large_blocks * large_block_chunks

/-- Calculates the transmission time in seconds -/
def transmission_time_seconds : ℕ := total_chunks / transmission_rate

/-- Theorem stating that the transmission time is 8.67 minutes -/
theorem transmission_time_is_8_67_minutes : 
  (transmission_time_seconds : ℚ) / 60 = 8.67 := by sorry

end transmission_time_is_8_67_minutes_l1797_179741


namespace inscribed_squares_product_l1797_179781

theorem inscribed_squares_product (a b : ℝ) : 
  (9 : ℝ).sqrt ^ 2 = 9 → 
  (16 : ℝ).sqrt ^ 2 = 16 → 
  a + b = (16 : ℝ).sqrt → 
  ((9 : ℝ).sqrt * Real.sqrt 2) ^ 2 = a ^ 2 + b ^ 2 → 
  a * b = -1 := by
sorry

end inscribed_squares_product_l1797_179781


namespace g_difference_l1797_179710

def g (n : ℕ) : ℚ := (1/4) * n * (n+1) * (n+2) * (n+3)

theorem g_difference (s : ℕ) : g s - g (s-1) = s * (s+1) * (s+2) := by
  sorry

end g_difference_l1797_179710
