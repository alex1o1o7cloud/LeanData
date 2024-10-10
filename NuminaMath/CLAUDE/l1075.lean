import Mathlib

namespace line_passes_through_point_line_has_slope_line_properties_l1075_107519

/-- A line in the xy-plane defined by the equation y = k(x+1) for some real k -/
structure Line where
  k : ℝ

/-- The point (-1, 0) in the xy-plane -/
def point : ℝ × ℝ := (-1, 0)

/-- Checks if a given point (x, y) lies on the line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.k * (p.1 + 1)

/-- States that the line passes through the point (-1, 0) -/
theorem line_passes_through_point (l : Line) : l.contains point := by sorry

/-- States that the line has a defined slope -/
theorem line_has_slope (l : Line) : ∃ m : ℝ, ∀ x y : ℝ, y = m * x + l.k := by sorry

/-- Main theorem combining both properties -/
theorem line_properties (l : Line) : l.contains point ∧ ∃ m : ℝ, ∀ x y : ℝ, y = m * x + l.k := by sorry

end line_passes_through_point_line_has_slope_line_properties_l1075_107519


namespace base_sum_theorem_l1075_107535

/-- Given two integer bases R₁ and R₂, if certain fractions have specific representations
    in these bases, then the sum of R₁ and R₂ is 21. -/
theorem base_sum_theorem (R₁ R₂ : ℕ) : 
  (R₁ > 1) → 
  (R₂ > 1) →
  ((4 * R₁ + 8) / (R₁^2 - 1) = (3 * R₂ + 6) / (R₂^2 - 1)) →
  ((8 * R₁ + 4) / (R₁^2 - 1) = (6 * R₂ + 3) / (R₂^2 - 1)) →
  R₁ + R₂ = 21 :=
by sorry

#check base_sum_theorem

end base_sum_theorem_l1075_107535


namespace simplify_complex_fraction_l1075_107561

theorem simplify_complex_fraction : 
  1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 7 - 2))) = 3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10) :=
by sorry

end simplify_complex_fraction_l1075_107561


namespace non_swimmers_playing_soccer_l1075_107538

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

end non_swimmers_playing_soccer_l1075_107538


namespace walnut_trees_planted_park_walnut_trees_l1075_107585

/-- The number of walnut trees planted in a park -/
def trees_planted (initial_trees final_trees : ℕ) : ℕ :=
  final_trees - initial_trees

/-- Theorem: The number of trees planted is the difference between the final and initial number of trees -/
theorem walnut_trees_planted (initial_trees final_trees : ℕ) 
  (h : initial_trees ≤ final_trees) :
  trees_planted initial_trees final_trees = final_trees - initial_trees :=
by sorry

/-- The specific case for the park problem -/
theorem park_walnut_trees : trees_planted 22 77 = 55 :=
by sorry

end walnut_trees_planted_park_walnut_trees_l1075_107585


namespace bucket_weight_bucket_weight_proof_l1075_107533

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight when it's full. -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let three_fourths_weight := p
  let one_third_weight := q
  let full_weight := (8 * p - 11 * q) / 5
  full_weight

/-- Prove that the calculated full weight is correct -/
theorem bucket_weight_proof (p q : ℝ) : 
  bucket_weight p q = (8 * p - 11 * q) / 5 := by
  sorry

end bucket_weight_bucket_weight_proof_l1075_107533


namespace semicircle_to_cone_volume_l1075_107589

theorem semicircle_to_cone_volume (R : ℝ) (h : R > 0) :
  let semicircle_radius := R
  let cone_base_radius := R / 2
  let cone_height := (Real.sqrt 3 / 2) * R
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius^2 * cone_height
  cone_volume = (Real.sqrt 3 / 24) * Real.pi * R^3 :=
by sorry

end semicircle_to_cone_volume_l1075_107589


namespace new_quadratic_equation_l1075_107580

theorem new_quadratic_equation (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let new_eq := fun y => a * y^2 + b * y + (c - a + Real.sqrt (b^2 - 4*a*c))
  (new_eq (x₁ - 1) = 0) ∧ (new_eq (x₂ + 1) = 0) := by
sorry

end new_quadratic_equation_l1075_107580


namespace selection_theorem_l1075_107570

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of male students. -/
def num_males : ℕ := 4

/-- The number of female students. -/
def num_females : ℕ := 3

/-- The total number of students to be selected. -/
def num_selected : ℕ := 3

/-- The number of ways to select students with both genders represented. -/
def num_ways : ℕ := 
  choose num_males 2 * choose num_females 1 + 
  choose num_males 1 * choose num_females 2

theorem selection_theorem : num_ways = 30 := by
  sorry

end selection_theorem_l1075_107570


namespace units_digits_divisible_by_eight_l1075_107539

theorem units_digits_divisible_by_eight :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 10 ∈ S ↔ ∃ m : Nat, m % 8 = 0 ∧ m % 10 = n % 10) ∧ Finset.card S = 5 := by
  sorry

end units_digits_divisible_by_eight_l1075_107539


namespace complex_equation_solution_l1075_107571

theorem complex_equation_solution (a : ℝ) : 
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end complex_equation_solution_l1075_107571


namespace pie_chart_best_for_part_whole_l1075_107597

-- Define the types of statistical graphs
inductive StatisticalGraph
  | BarGraph
  | PieChart
  | LineGraph
  | FrequencyDistributionHistogram

-- Define a property for highlighting part-whole relationships
def highlightsPartWholeRelationship (graph : StatisticalGraph) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => true
  | _ => false

-- Theorem statement
theorem pie_chart_best_for_part_whole : 
  ∀ (graph : StatisticalGraph), 
    highlightsPartWholeRelationship graph → graph = StatisticalGraph.PieChart :=
by
  sorry


end pie_chart_best_for_part_whole_l1075_107597


namespace cube_sum_minus_triple_product_l1075_107529

theorem cube_sum_minus_triple_product (x y z : ℝ) 
  (sum_eq : x + y + z = 13)
  (sum_products_eq : x*y + x*z + y*z = 32) :
  x^3 + y^3 + z^3 - 3*x*y*z = 949 := by
  sorry

end cube_sum_minus_triple_product_l1075_107529


namespace increasing_digits_mod_1000_l1075_107555

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of 8-digit positive integers with digits in increasing order (1-8, repetition allowed) -/
def M : ℕ := starsAndBars 8 8

theorem increasing_digits_mod_1000 : M % 1000 = 435 := by sorry

end increasing_digits_mod_1000_l1075_107555


namespace arithmetic_sequence_10th_term_l1075_107528

/-- Given an arithmetic sequence of 25 terms with first term 5 and last term 77,
    prove that the 10th term is 32. -/
theorem arithmetic_sequence_10th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    (a 1 = 5) →                          -- first term is 5
    (a 25 = 77) →                        -- last term is 77
    (a 10 = 32) :=                       -- 10th term is 32
by sorry

end arithmetic_sequence_10th_term_l1075_107528


namespace min_value_of_sum_of_abs_min_value_is_achievable_l1075_107579

theorem min_value_of_sum_of_abs (x y : ℝ) : 
  |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

theorem min_value_is_achievable : 
  ∃ (x y : ℝ), |x - 1| + |x| + |y - 1| + |y + 1| = 3 :=
by sorry

end min_value_of_sum_of_abs_min_value_is_achievable_l1075_107579


namespace independence_day_absences_l1075_107546

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

end independence_day_absences_l1075_107546


namespace smallest_angle_in_triangle_l1075_107517

theorem smallest_angle_in_triangle (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_ratio : Real.sin A / Real.sin B = 2 / Real.sqrt 6 ∧ 
             Real.sin B / Real.sin C = Real.sqrt 6 / (Real.sqrt 3 + 1)) : 
  min A (min B C) = Real.pi / 4 := by
sorry

end smallest_angle_in_triangle_l1075_107517


namespace fare_660_equals_3_miles_unique_distance_for_660_l1075_107530

/-- Calculates the taxi fare for a given distance -/
def taxi_fare (distance : ℚ) : ℚ :=
  1 + 0.4 * (5 * distance - 1)

/-- Proves that a fare of $6.60 corresponds to a distance of 3 miles -/
theorem fare_660_equals_3_miles :
  taxi_fare 3 = 6.6 :=
sorry

/-- Proves that 3 miles is the unique distance that results in a fare of $6.60 -/
theorem unique_distance_for_660 :
  ∀ d : ℚ, taxi_fare d = 6.6 → d = 3 :=
sorry

end fare_660_equals_3_miles_unique_distance_for_660_l1075_107530


namespace purely_imaginary_iff_one_i_l1075_107522

/-- A complex number z is purely imaginary if and only if it has the form i (i.e., a = 1 and b = 0) -/
theorem purely_imaginary_iff_one_i (z : ℂ) : 
  (∃ (a b : ℝ), z = Complex.I * a + b) → 
  (z.re = 0 ∧ z.im ≠ 0) ↔ z = Complex.I :=
sorry

end purely_imaginary_iff_one_i_l1075_107522


namespace exactly_one_subset_exactly_one_element_l1075_107573

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- Theorem for part 1
theorem exactly_one_subset (a : ℝ) : (∃! (S : Set ℝ), S ⊆ A a) ↔ a > 1 := by sorry

-- Theorem for part 2
theorem exactly_one_element (a : ℝ) : (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 1 := by sorry

end exactly_one_subset_exactly_one_element_l1075_107573


namespace smaller_solution_of_quadratic_l1075_107554

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 20*x - 72 = 0 → (∃ y : ℝ, y^2 + 20*y - 72 = 0 ∧ y ≤ x) → x = -24 :=
by sorry

end smaller_solution_of_quadratic_l1075_107554


namespace select_four_with_one_girl_l1075_107584

/-- The number of ways to select 4 people from two groups with exactly 1 girl -/
def select_with_one_girl (boys_a boys_b girls_a girls_b : ℕ) : ℕ :=
  (girls_a.choose 1 * boys_a.choose 1 * boys_b.choose 2) +
  (boys_a.choose 2 * girls_b.choose 1 * boys_b.choose 1)

/-- Theorem stating the correct number of selections for the given group compositions -/
theorem select_four_with_one_girl :
  select_with_one_girl 5 6 3 2 = 345 := by
  sorry

end select_four_with_one_girl_l1075_107584


namespace marbles_given_proof_l1075_107531

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 776 - 593

/-- Connie's initial number of marbles -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 593

theorem marbles_given_proof : 
  marbles_given = initial_marbles - remaining_marbles :=
by sorry

end marbles_given_proof_l1075_107531


namespace largest_area_chord_construction_l1075_107598

/-- Represents a direction in 2D space -/
structure Direction where
  angle : Real

/-- Represents an ellipse -/
structure Ellipse where
  semi_major_axis : Real
  semi_minor_axis : Real

/-- Represents a chord of an ellipse -/
structure Chord where
  direction : Direction
  length : Real

/-- Represents a triangle -/
structure Triangle where
  base : Real
  height : Real

/-- Calculates the area of a triangle -/
def triangle_area (t : Triangle) : Real :=
  0.5 * t.base * t.height

/-- Finds the chord that forms the triangle with the largest area -/
def largest_area_chord (e : Ellipse) (i : Direction) : Chord :=
  sorry

/-- Theorem: The chord that forms the largest area triangle is constructed by 
    finding an intersection point and creating two mirrored right triangles -/
theorem largest_area_chord_construction (e : Ellipse) (i : Direction) :
  ∃ (c : Chord), c = largest_area_chord e i ∧
  ∃ (t1 t2 : Triangle), 
    triangle_area t1 = triangle_area t2 ∧
    t1.base = t2.base ∧
    t1.height = t2.height ∧
    t1.base * t1.base + t1.height * t1.height = c.length * c.length :=
  sorry

end largest_area_chord_construction_l1075_107598


namespace solution_set_is_closed_unit_interval_l1075_107582

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the set of a that satisfy f(1) ≤ f(a)
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {a | f 1 ≤ f a}

-- State the theorem
theorem solution_set_is_closed_unit_interval
  (f : ℝ → ℝ) (h_even : is_even f) (h_incr : increasing_on_nonpositive f) :
  solution_set f = Set.Icc (-1) 1 := by sorry

end solution_set_is_closed_unit_interval_l1075_107582


namespace age_ratio_is_two_to_one_l1075_107553

/-- The ratio of a man's age to his son's age in two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  let man_age := son_age + age_difference
  (man_age + 2) / (son_age + 2)

/-- Theorem: The ratio of the man's age to his son's age in two years is 2:1 -/
theorem age_ratio_is_two_to_one (son_age : ℕ) (age_difference : ℕ)
  (h1 : son_age = 23)
  (h2 : age_difference = 25) :
  age_ratio son_age age_difference = 2 := by
  sorry

#eval age_ratio 23 25

end age_ratio_is_two_to_one_l1075_107553


namespace root_equation_k_value_l1075_107593

theorem root_equation_k_value :
  ∀ k : ℝ, ((-2)^2 - k*(-2) + 2 = 0) → k = -3 := by
  sorry

end root_equation_k_value_l1075_107593


namespace trigonometric_simplification_l1075_107506

theorem trigonometric_simplification (θ : ℝ) :
  (Real.sin (π - 2*θ) / (1 - Real.sin (π/2 + 2*θ))) * Real.tan (π + θ) = 1 := by
  sorry

end trigonometric_simplification_l1075_107506


namespace sailboat_sails_height_l1075_107537

theorem sailboat_sails_height (rectangular_length rectangular_width first_triangular_base second_triangular_base second_triangular_height total_canvas : ℝ) 
  (h1 : rectangular_length = 8)
  (h2 : rectangular_width = 5)
  (h3 : first_triangular_base = 3)
  (h4 : second_triangular_base = 4)
  (h5 : second_triangular_height = 6)
  (h6 : total_canvas = 58) :
  let rectangular_area := rectangular_length * rectangular_width
  let second_triangular_area := (second_triangular_base * second_triangular_height) / 2
  let first_triangular_area := total_canvas - rectangular_area - second_triangular_area
  first_triangular_area = (first_triangular_base * 4) / 2 := by
  sorry

end sailboat_sails_height_l1075_107537


namespace condition_relations_l1075_107512

theorem condition_relations (A B C : Prop) 
  (h1 : B → A)  -- A is necessary for B
  (h2 : C → B)  -- C is sufficient for B
  (h3 : ¬(B → C))  -- C is not necessary for B
  : (C → A) ∧ ¬(A → C) := by sorry

end condition_relations_l1075_107512


namespace line_through_circle_center_l1075_107503

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop := x + m*y + 1 = 0

-- Define the center of the circle
def center_C : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_through_circle_center (m : ℝ) :
  line_l (center_C.1) (center_C.2) m → m = -1 :=
by sorry

end line_through_circle_center_l1075_107503


namespace fraction_value_at_sqrt_two_l1075_107577

theorem fraction_value_at_sqrt_two :
  let x := Real.sqrt 2
  (x^2 - 1) / (x^2 - x) - 1 = Real.sqrt 2 / 2 := by
  sorry

end fraction_value_at_sqrt_two_l1075_107577


namespace yella_computer_usage_l1075_107559

def days_in_week : ℕ := 7
def hours_per_day_this_week : ℕ := 8
def hours_difference : ℕ := 35

def computer_usage_last_week : ℕ := days_in_week * hours_per_day_this_week + hours_difference

theorem yella_computer_usage :
  computer_usage_last_week = 91 := by
sorry

end yella_computer_usage_l1075_107559


namespace wire_cutting_l1075_107556

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 70 →
  ratio = 2 / 3 →
  shorter_piece + (shorter_piece + ratio * shorter_piece) = total_length →
  shorter_piece = 26.25 := by
  sorry

end wire_cutting_l1075_107556


namespace four_digit_sum_2008_l1075_107536

theorem four_digit_sum_2008 : ∃ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (n + (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) = 2008) ∧
  (∃ m : ℕ, m ≠ n ∧ 
    (1000 ≤ m ∧ m < 10000) ∧ 
    (m + (m / 1000 + (m / 100 % 10) + (m / 10 % 10) + (m % 10)) = 2008)) :=
by sorry

#check four_digit_sum_2008

end four_digit_sum_2008_l1075_107536


namespace unused_sector_angle_l1075_107543

/-- Given a circular piece of cardboard with radius 18 cm, from which a sector is removed
    to form a cone with radius 15 cm and volume 1350π cubic centimeters,
    the measure of the angle of the unused sector is 60°. -/
theorem unused_sector_angle (r_cardboard : ℝ) (r_cone : ℝ) (v_cone : ℝ) :
  r_cardboard = 18 →
  r_cone = 15 →
  v_cone = 1350 * Real.pi →
  ∃ (angle : ℝ),
    angle = 60 ∧
    angle = 360 - (2 * r_cone * Real.pi) / (2 * r_cardboard * Real.pi) * 360 :=
by sorry

end unused_sector_angle_l1075_107543


namespace fraction_saved_is_one_third_l1075_107550

/-- Represents the fraction of take-home pay saved each month -/
def fraction_saved : ℝ := sorry

/-- Represents the monthly take-home pay -/
def monthly_pay : ℝ := sorry

/-- The total amount saved at the end of the year -/
def total_saved : ℝ := 12 * fraction_saved * monthly_pay

/-- The amount not saved in a month -/
def monthly_not_saved : ℝ := (1 - fraction_saved) * monthly_pay

/-- States that the total amount saved is 6 times the monthly amount not saved -/
axiom total_saved_eq_six_times_not_saved : total_saved = 6 * monthly_not_saved

/-- Theorem stating that the fraction saved each month is 1/3 -/
theorem fraction_saved_is_one_third : fraction_saved = 1/3 := by sorry

end fraction_saved_is_one_third_l1075_107550


namespace roof_tiles_needed_l1075_107558

def land_cost_per_sqm : ℕ := 50
def brick_cost_per_thousand : ℕ := 100
def roof_tile_cost : ℕ := 10
def land_area : ℕ := 2000
def brick_count : ℕ := 10000
def total_cost : ℕ := 106000

theorem roof_tiles_needed : ℕ := by
  -- The number of roof tiles needed is 500
  sorry

end roof_tiles_needed_l1075_107558


namespace cubic_sum_implies_linear_sum_l1075_107505

theorem cubic_sum_implies_linear_sum (x : ℝ) (h : x^3 + 1/x^3 = 52) : x + 1/x = 4 := by
  sorry

end cubic_sum_implies_linear_sum_l1075_107505


namespace function_range_l1075_107534

theorem function_range (a : ℝ) (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x, f x = a^(-x)) (h4 : f (-2) > f (-3)) : 0 < a ∧ a < 1 := by
  sorry

end function_range_l1075_107534


namespace r_amount_l1075_107500

def total_amount : ℝ := 9000

theorem r_amount (p q r : ℝ) 
  (h1 : p + q + r = total_amount)
  (h2 : r = (2/3) * (p + q)) :
  r = 3600 := by
  sorry

end r_amount_l1075_107500


namespace min_value_expression_min_value_achievable_l1075_107562

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c) ≥ 343 :=
sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a * b * c) = 343 :=
sorry

end min_value_expression_min_value_achievable_l1075_107562


namespace line_parallel_to_intersection_of_parallel_planes_l1075_107532

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem line_parallel_to_intersection_of_parallel_planes
  (a : Line) (α β : Plane) (b : Line)
  (h1 : parallel_line_plane a α)
  (h2 : parallel_line_plane a β)
  (h3 : intersect α β = b) :
  parallel_line_line a b :=
sorry

end line_parallel_to_intersection_of_parallel_planes_l1075_107532


namespace daves_remaining_apps_l1075_107592

/-- Represents the number of apps and files on Dave's phone -/
structure PhoneContent where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone -/
def initial : PhoneContent := { apps := 11, files := 3 }

/-- The final state of Dave's phone after deletion -/
def final : PhoneContent := { apps := 2, files := 24 }

/-- Theorem stating that the final number of apps on Dave's phone is 2 -/
theorem daves_remaining_apps :
  final.apps = 2 ∧
  final.files = 24 ∧
  final.files = final.apps + 22 :=
by sorry

end daves_remaining_apps_l1075_107592


namespace star_equation_solution_l1075_107544

/-- The star operation defined on real numbers -/
def star (x y : ℝ) : ℝ := 5*x - 2*y + 2*x*y

/-- Theorem stating that 4 star y = 22 if and only if y = 1/3 -/
theorem star_equation_solution :
  ∀ y : ℝ, star 4 y = 22 ↔ y = 1/3 := by sorry

end star_equation_solution_l1075_107544


namespace geometric_sequence_sum_formula_implies_t_equals_5_l1075_107523

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), n ≥ 1 → a (n + 1) = r * a n

-- Define the sum formula for the first n terms
def sum_formula (S : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ (n : ℕ), S n = t * 5^n - 2

-- Theorem statement
theorem geometric_sequence_sum_formula_implies_t_equals_5 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sum_formula S t) : 
  t = 5 := by sorry

end geometric_sequence_sum_formula_implies_t_equals_5_l1075_107523


namespace magnitude_BC_l1075_107595

/-- Given two vectors BA and AC in R², prove that the magnitude of BC is 5. -/
theorem magnitude_BC (BA AC : ℝ × ℝ) (h1 : BA = (3, -2)) (h2 : AC = (0, 6)) : 
  ‖BA + AC‖ = 5 := by sorry

end magnitude_BC_l1075_107595


namespace remaining_tanning_time_l1075_107518

/-- Calculates the remaining tanning time for the last two weeks of a month. -/
theorem remaining_tanning_time 
  (monthly_limit : ℕ) 
  (daily_time : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (h1 : monthly_limit = 200)
  (h2 : daily_time = 30)
  (h3 : days_per_week = 2)
  (h4 : weeks = 2) :
  monthly_limit - (daily_time * days_per_week * weeks) = 80 := by
  sorry

#check remaining_tanning_time

end remaining_tanning_time_l1075_107518


namespace james_louise_ages_l1075_107564

/-- James and Louise's ages problem -/
theorem james_louise_ages (j l : ℝ) : 
  j = l + 9 →                   -- James is nine years older than Louise
  j + 8 = 3 * (l - 4) →         -- Eight years from now, James will be three times as old as Louise was four years ago
  j + l = 38 :=                 -- The sum of their current ages is 38
by
  sorry

end james_louise_ages_l1075_107564


namespace banana_popsicles_count_l1075_107525

theorem banana_popsicles_count (grape_count cherry_count total_count : ℕ) 
  (h1 : grape_count = 2)
  (h2 : cherry_count = 13)
  (h3 : total_count = 17) :
  total_count - (grape_count + cherry_count) = 2 := by
  sorry

end banana_popsicles_count_l1075_107525


namespace die_roll_probability_l1075_107515

def is_valid_roll (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 6

def angle_in_range (m n : ℕ) : Prop :=
  let a : ℝ × ℝ := (m, n)
  let b : ℝ × ℝ := (1, 0)
  let cos_alpha := (m : ℝ) / Real.sqrt ((m^2 : ℝ) + (n^2 : ℝ))
  Real.sqrt 2 / 2 < cos_alpha ∧ cos_alpha < 1

def count_favorable_outcomes : ℕ := 15

def total_outcomes : ℕ := 36

theorem die_roll_probability :
  (count_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 :=
sorry

end die_roll_probability_l1075_107515


namespace expression_evaluation_l1075_107545

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 3 * x - y / 3 ≠ 0) :
  (3 * x - y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹)^2 = (y + 9 * x)^2 / (3 * x^2 * y^2 * (9 * x - y)) :=
by sorry


end expression_evaluation_l1075_107545


namespace solve_linear_equation_l1075_107588

theorem solve_linear_equation (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 := by
  sorry

end solve_linear_equation_l1075_107588


namespace greatest_integer_difference_l1075_107587

theorem greatest_integer_difference (x y : ℤ) 
  (hx : 7 < x ∧ x < 9)
  (hy : 9 < y ∧ y < 15) :
  ∃ (d : ℤ), d = y - x ∧ d ≤ 6 ∧ ∀ (d' : ℤ), d' = y - x → d' ≤ d :=
sorry

end greatest_integer_difference_l1075_107587


namespace average_of_remaining_numbers_l1075_107509

theorem average_of_remaining_numbers
  (total : ℝ)
  (avg_all : ℝ)
  (avg_first_two : ℝ)
  (avg_second_two : ℝ)
  (h1 : total = 6)
  (h2 : avg_all = 2.80)
  (h3 : avg_first_two = 2.4)
  (h4 : avg_second_two = 2.3) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_second_two) / 2 = 3.7 := by
sorry

end average_of_remaining_numbers_l1075_107509


namespace solution_set_and_range_l1075_107521

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 3|

theorem solution_set_and_range :
  (∀ x, x ∈ {y : ℝ | 0 ≤ y ∧ y ≤ 3} ↔ f 3 x ≤ 6) ∧
  (∀ a, (∀ x, f a x + g x ≥ 5) ↔ a ≥ 11/3) := by sorry

end solution_set_and_range_l1075_107521


namespace pyramid_circumscribed_equivalence_l1075_107557

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with n vertices -/
structure Pyramid (n : ℕ) where
  apex : Point3D
  base : Fin n → Point3D

/-- Predicate for the existence of a circumscribed sphere around a pyramid -/
def has_circumscribed_sphere (p : Pyramid n) : Prop := sorry

/-- Predicate for the existence of a circumscribed circle around the base of a pyramid -/
def has_circumscribed_circle_base (p : Pyramid n) : Prop := sorry

/-- Theorem stating the equivalence of circumscribed sphere and circle for a pyramid -/
theorem pyramid_circumscribed_equivalence (n : ℕ) (p : Pyramid n) :
  has_circumscribed_sphere p ↔ has_circumscribed_circle_base p := by sorry

end pyramid_circumscribed_equivalence_l1075_107557


namespace graph_vertical_shift_l1075_107599

-- Define a continuous function f on the real line
variable (f : ℝ → ℝ)
variable (h : Continuous f)

-- Define the vertical shift operation
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

-- Theorem statement
theorem graph_vertical_shift :
  ∀ (x y : ℝ), y = f x + 2 ↔ y = (verticalShift f 2) x :=
by sorry

end graph_vertical_shift_l1075_107599


namespace sin_cos_pi_over_12_l1075_107514

theorem sin_cos_pi_over_12 : 
  Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by sorry

end sin_cos_pi_over_12_l1075_107514


namespace imaginary_part_of_z_l1075_107511

theorem imaginary_part_of_z (z : ℂ) : z = (1 + 2*I) / ((1 - I)^2) → z.im = 1/2 := by
  sorry

end imaginary_part_of_z_l1075_107511


namespace complex_fraction_power_l1075_107504

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_power (h : i * i = -1) : ((1 + i) / (1 - i)) ^ 2013 = i := by
  sorry

end complex_fraction_power_l1075_107504


namespace cube_roots_unity_sum_l1075_107513

theorem cube_roots_unity_sum (x y : ℂ) : 
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
  sorry

end cube_roots_unity_sum_l1075_107513


namespace percentage_change_condition_l1075_107596

theorem percentage_change_condition
  (p q r M : ℝ)
  (hp : p > 0)
  (hq : 0 < q ∧ q < 100)
  (hr : 0 < r ∧ r < 100)
  (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M ↔
  p > (100 * (q + r)) / (100 - q - r) :=
by sorry

end percentage_change_condition_l1075_107596


namespace driver_net_pay_rate_l1075_107563

-- Define the parameters
def travel_time : ℝ := 3
def speed : ℝ := 50
def fuel_efficiency : ℝ := 25
def pay_rate : ℝ := 0.60
def gasoline_cost : ℝ := 2.50

-- Define the theorem
theorem driver_net_pay_rate :
  let total_distance := travel_time * speed
  let gasoline_used := total_distance / fuel_efficiency
  let gross_earnings := pay_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := gross_earnings - gasoline_expense
  net_earnings / travel_time = 25 := by sorry

end driver_net_pay_rate_l1075_107563


namespace problem_solution_l1075_107520

-- Define the function f
def f (x : ℝ) : ℝ := 6 * x^2 + x - 1

-- State the theorem
theorem problem_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : f (Real.sin α) = 0) :
  Real.sin α = 1 / 3 ∧
  (Real.tan (π + α) * Real.cos (-α)) / (Real.cos (π / 2 - α) * Real.sin (π - α)) = 3 ∧
  Real.sin (α + π / 6) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 :=
by sorry

end problem_solution_l1075_107520


namespace parabola_vertex_l1075_107541

/-- The equation of a parabola in the form y^2 - 4y + 2x + 9 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 2*x + 9 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  eq x y ∧ ∀ x' y', eq x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex (-5/2) 2 parabola_equation :=
sorry

end parabola_vertex_l1075_107541


namespace least_number_with_remainder_four_l1075_107590

theorem least_number_with_remainder_four (n : ℕ) : 
  (∀ m : ℕ, m > 0 → n % m = 4) → 
  (n % 12 = 4) → 
  n ≥ 40 :=
by
  sorry

end least_number_with_remainder_four_l1075_107590


namespace equation_solution_l1075_107586

theorem equation_solution : ∃ x : ℝ, (1 / 7 + 4 / x = 12 / x + 1 / 14) ∧ x = 112 := by
  sorry

end equation_solution_l1075_107586


namespace quadratic_inequality_always_negative_l1075_107507

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -15 * x^2 + 4 * x - 6 < 0 := by
sorry

end quadratic_inequality_always_negative_l1075_107507


namespace shopkeeper_decks_l1075_107548

/-- Represents the number of face cards in a standard deck of playing cards. -/
def face_cards_per_deck : ℕ := 12

/-- Represents the total number of face cards the shopkeeper has. -/
def total_face_cards : ℕ := 60

/-- Calculates the number of complete decks given the total number of face cards. -/
def number_of_decks : ℕ := total_face_cards / face_cards_per_deck

theorem shopkeeper_decks : number_of_decks = 5 := by
  sorry

end shopkeeper_decks_l1075_107548


namespace simplify_fraction_l1075_107568

theorem simplify_fraction : (125 : ℚ) / 10000 * 40 = 5 / 2 := by
  sorry

end simplify_fraction_l1075_107568


namespace monet_paintings_consecutive_probability_l1075_107581

/-- The probability of consecutive Monet paintings in a random arrangement -/
theorem monet_paintings_consecutive_probability 
  (total_pieces : ℕ) 
  (monet_paintings : ℕ) 
  (h1 : total_pieces = 12) 
  (h2 : monet_paintings = 4) :
  (monet_paintings.factorial * (total_pieces - monet_paintings + 1)) / total_pieces.factorial = 18 / 95 := by
  sorry

end monet_paintings_consecutive_probability_l1075_107581


namespace definitely_rain_next_tuesday_is_false_l1075_107567

-- Define a proposition representing the statement "It will definitely rain next Tuesday"
def definitely_rain_next_tuesday : Prop := True

-- Define a proposition representing the uncertainty of future events
def future_events_are_uncertain : Prop := True

-- Theorem stating that the original statement is false
theorem definitely_rain_next_tuesday_is_false : 
  future_events_are_uncertain → ¬definitely_rain_next_tuesday := by
  sorry

end definitely_rain_next_tuesday_is_false_l1075_107567


namespace simplify_expressions_l1075_107501

theorem simplify_expressions (x y : ℝ) :
  (3 * x - 2 * y + 1 + 3 * y - 2 * x - 5 = x + y - 4) ∧
  ((2 * x^4 - 5 * x^2 - 4 * x + 3) - (3 * x^3 - 5 * x^2 - 4 * x) = 2 * x^4 - 3 * x^3 + 3) := by
  sorry

end simplify_expressions_l1075_107501


namespace unique_solution_quadratic_l1075_107552

theorem unique_solution_quadratic (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by sorry

end unique_solution_quadratic_l1075_107552


namespace field_length_correct_l1075_107574

/-- The length of a rectangular field -/
def field_length : ℝ := 75

/-- The width of the rectangular field -/
def field_width : ℝ := 15

/-- The number of times the field is circled -/
def laps : ℕ := 3

/-- The total distance covered -/
def total_distance : ℝ := 540

/-- Theorem stating that the field length is correct given the conditions -/
theorem field_length_correct : 
  2 * (field_length + field_width) * laps = total_distance :=
by sorry

end field_length_correct_l1075_107574


namespace min_value_expression_min_value_attained_l1075_107502

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) ≥ 30 :=
by sorry

theorem min_value_attained :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  8 * x^4 + 12 * y^4 + 18 * z^4 + 25 / (x * y * z) = 30 :=
by sorry

end min_value_expression_min_value_attained_l1075_107502


namespace factorial_divisibility_l1075_107575

theorem factorial_divisibility (n : ℕ) (p : ℕ) (h_pos : n > 0) (h_prime : Nat.Prime p) 
  (h_div : p ^ p ∣ Nat.factorial n) : p ^ (p + 1) ∣ Nat.factorial n := by
  sorry

end factorial_divisibility_l1075_107575


namespace museum_visitors_survey_l1075_107551

theorem museum_visitors_survey (V : ℕ) : 
  (∃ E : ℕ, 
    V = E + 140 ∧ 
    3 * V = 4 * E) →
  V = 560 :=
by
  sorry

end museum_visitors_survey_l1075_107551


namespace m_eq_one_necessary_not_sufficient_l1075_107547

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem m_eq_one_necessary_not_sufficient :
  ∃ m : ℝ, isPureImaginary (m * (m - 1) + Complex.I) ∧ m ≠ 1 ∧
  ∀ m : ℝ, m = 1 → isPureImaginary (m * (m - 1) + Complex.I) :=
by sorry

end m_eq_one_necessary_not_sufficient_l1075_107547


namespace quarters_count_l1075_107508

/-- Given a sum of $3.35 consisting of quarters and dimes, with a total of 23 coins, 
    prove that the number of quarters is 7. -/
theorem quarters_count (total_value : ℚ) (total_coins : ℕ) 
  (h1 : total_value = 335/100) 
  (h2 : total_coins = 23) : ∃ (quarters dimes : ℕ),
  quarters + dimes = total_coins ∧ 
  (25 * quarters + 10 * dimes : ℚ) / 100 = total_value ∧
  quarters = 7 := by
  sorry

end quarters_count_l1075_107508


namespace delta_max_success_ratio_l1075_107569

/-- Represents a participant's score in a math competition --/
structure Score where
  points : ℕ
  total : ℕ
  h_positive : points > 0
  h_valid : points ≤ total

/-- Represents a participant's scores over three days --/
structure ThreeDayScore where
  day1 : Score
  day2 : Score
  day3 : Score
  h_total : day1.total + day2.total + day3.total = 600

def success_ratio (s : Score) : ℚ :=
  s.points / s.total

def three_day_success_ratio (s : ThreeDayScore) : ℚ :=
  (s.day1.points + s.day2.points + s.day3.points) / 600

theorem delta_max_success_ratio 
  (charlie : ThreeDayScore)
  (delta : ThreeDayScore)
  (h_charlie_day1 : charlie.day1 = ⟨210, 350, by sorry, by sorry⟩)
  (h_charlie_day2 : charlie.day2 = ⟨170, 250, by sorry, by sorry⟩)
  (h_charlie_day3 : charlie.day3 = ⟨0, 0, by sorry, by sorry⟩)
  (h_delta_day1 : success_ratio delta.day1 < success_ratio charlie.day1)
  (h_delta_day2 : success_ratio delta.day2 < success_ratio charlie.day2)
  (h_delta_day3_positive : delta.day3.points > 0) :
  three_day_success_ratio delta ≤ 378 / 600 :=
sorry

end delta_max_success_ratio_l1075_107569


namespace commute_time_difference_l1075_107516

def commute_times (x y : ℝ) : List ℝ := [x, y, 8, 11, 9]

theorem commute_time_difference (x y : ℝ) :
  (List.sum (commute_times x y)) / 5 = 8 →
  (List.sum (List.map (λ t => (t - 8)^2) (commute_times x y))) / 5 = 4 →
  |x - y| = 2 := by
sorry

end commute_time_difference_l1075_107516


namespace triple_overlap_area_is_six_l1075_107583

/-- Represents a rectangular rug with width and height -/
structure Rug where
  width : ℝ
  height : ℝ

/-- Represents the hall and the rugs placed in it -/
structure HallWithRugs where
  hallSize : ℝ
  rug1 : Rug
  rug2 : Rug
  rug3 : Rug

/-- Calculates the area covered by all three rugs in the hall -/
def tripleOverlapArea (hall : HallWithRugs) : ℝ :=
  2 * 3

/-- Theorem stating that the area covered by all three rugs is 6 square meters -/
theorem triple_overlap_area_is_six (hall : HallWithRugs) 
  (h1 : hall.hallSize = 10)
  (h2 : hall.rug1 = ⟨6, 8⟩)
  (h3 : hall.rug2 = ⟨6, 6⟩)
  (h4 : hall.rug3 = ⟨5, 7⟩) :
  tripleOverlapArea hall = 6 := by
  sorry

end triple_overlap_area_is_six_l1075_107583


namespace scalene_triangle_angle_difference_l1075_107527

/-- A scalene triangle with one angle of 80 degrees can have a difference of 80 degrees between its other two angles. -/
theorem scalene_triangle_angle_difference : ∃ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- angles are positive
  a + b + c = 180 ∧  -- sum of angles in a triangle is 180°
  a = 80 ∧  -- one angle is 80°
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧  -- all angles are different (scalene)
  |b - c| = 80  -- difference between other two angles is 80°
:= by sorry

end scalene_triangle_angle_difference_l1075_107527


namespace min_PQ_distance_l1075_107524

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := x + Real.log x

-- Define the distance between P and Q
def PQ_distance (a x₁ x₂ : ℝ) : ℝ :=
  |x₂ - x₁|

-- State the theorem
theorem min_PQ_distance :
  ∃ (a : ℝ), ∀ (x₁ x₂ : ℝ),
    f x₁ = a → g x₂ = a →
    (∀ (y₁ y₂ : ℝ), f y₁ = a → g y₂ = a → PQ_distance a x₁ x₂ ≤ PQ_distance a y₁ y₂) →
    PQ_distance a x₁ x₂ = 2 :=
sorry

end

end min_PQ_distance_l1075_107524


namespace smallest_square_containing_circle_l1075_107576

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) : 
  (2 * r) ^ 2 = 196 := by
  sorry

end smallest_square_containing_circle_l1075_107576


namespace gcd_840_1764_l1075_107542

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1075_107542


namespace percent_not_working_projects_l1075_107540

/-- Represents the survey results of employees working on projects -/
structure ProjectSurvey where
  total : ℕ
  projectA : ℕ
  projectB : ℕ
  bothProjects : ℕ

/-- Calculates the percentage of employees not working on either project -/
def percentNotWorking (survey : ProjectSurvey) : ℚ :=
  let workingOnEither := survey.projectA + survey.projectB - survey.bothProjects
  let notWorking := survey.total - workingOnEither
  (notWorking : ℚ) / survey.total * 100

/-- The theorem stating the percentage of employees not working on either project -/
theorem percent_not_working_projects (survey : ProjectSurvey) 
    (h1 : survey.total = 150)
    (h2 : survey.projectA = 90)
    (h3 : survey.projectB = 50)
    (h4 : survey.bothProjects = 30) :
    percentNotWorking survey = 26.67 := by
  sorry


end percent_not_working_projects_l1075_107540


namespace examination_attendance_l1075_107594

theorem examination_attendance :
  ∀ (total_students : ℕ) (passed_percentage : ℚ) (failed_count : ℕ),
    passed_percentage = 35 / 100 →
    failed_count = 520 →
    (1 - passed_percentage) * total_students = failed_count →
    total_students = 800 := by
  sorry

end examination_attendance_l1075_107594


namespace smallest_invertible_domain_l1075_107510

def g (x : ℝ) : ℝ := (2*x - 3)^2 - 4

theorem smallest_invertible_domain (c : ℝ) : 
  (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ∧ 
  (∀ c' : ℝ, c' < c → ∃ x y, x ≥ c' ∧ y ≥ c' ∧ x ≠ y ∧ g x = g y) → 
  c = 3/2 :=
sorry

end smallest_invertible_domain_l1075_107510


namespace abs_T_equals_1024_l1075_107572

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^18 - (1 - i)^18

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by
  sorry

end abs_T_equals_1024_l1075_107572


namespace perpendicular_line_x_intercept_l1075_107591

/-- Given a line L1 defined by 3x - 2y = 6, and a line L2 perpendicular to L1 with y-intercept 4,
    the x-intercept of L2 is 6. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 2 * y = 6
  let m1 : ℝ := 3 / 2  -- slope of L1
  let m2 : ℝ := -2 / 3  -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x + 4  -- equation of L2
  let x_intercept : ℝ := 6
  (∀ x y, L2 x y ↔ y = m2 * x + 4) →  -- L2 is defined correctly
  (m1 * m2 = -1) →  -- L1 and L2 are perpendicular
  L2 x_intercept 0  -- (6, 0) satisfies the equation of L2
  := by sorry

end perpendicular_line_x_intercept_l1075_107591


namespace square_side_increase_l1075_107565

theorem square_side_increase (p : ℝ) : 
  ((1 + p / 100) ^ 2 = 1.44) → p = 20 := by
  sorry

end square_side_increase_l1075_107565


namespace co_molecular_weight_l1075_107578

-- Define the atomic weights
def atomic_weight_carbon : ℝ := 12.01
def atomic_weight_oxygen : ℝ := 16.00

-- Define the molecular weight calculation function
def molecular_weight (carbon_atoms : ℕ) (oxygen_atoms : ℕ) : ℝ :=
  carbon_atoms * atomic_weight_carbon + oxygen_atoms * atomic_weight_oxygen

-- Theorem statement
theorem co_molecular_weight :
  molecular_weight 1 1 = 28.01 := by sorry

end co_molecular_weight_l1075_107578


namespace scientific_notation_exponent_l1075_107549

theorem scientific_notation_exponent (n : ℤ) :
  0.0000502 = 5.02 * (10 : ℝ) ^ n → n = -4 := by
  sorry

end scientific_notation_exponent_l1075_107549


namespace event_probability_estimate_l1075_107526

-- Define the frequency function
def frequency : ℕ → ℝ
| 20 => 0.300
| 50 => 0.360
| 100 => 0.350
| 300 => 0.350
| 500 => 0.352
| 1000 => 0.351
| 5000 => 0.351
| _ => 0  -- For other values, we set the frequency to 0

-- Define the set of trial numbers
def trialNumbers : Set ℕ := {20, 50, 100, 300, 500, 1000, 5000}

-- Theorem statement
theorem event_probability_estimate :
  ∀ ε > 0, ∃ N ∈ trialNumbers, ∀ n ∈ trialNumbers, n ≥ N → |frequency n - 0.35| < ε :=
sorry

end event_probability_estimate_l1075_107526


namespace modified_riemann_zeta_sum_l1075_107560

noncomputable def ξ (x : ℝ) : ℝ := ∑' n, (1 : ℝ) / (2 * n) ^ x

theorem modified_riemann_zeta_sum (h : ∀ x > 2, ξ x = ∑' n, (1 : ℝ) / (2 * n) ^ x) :
  ∑' k, ξ (2 * k + 1) = 1 := by sorry

end modified_riemann_zeta_sum_l1075_107560


namespace vikas_questions_l1075_107566

/-- Prove that given a total of 24 questions submitted in the ratio 7 : 3 : 2,
    the number of questions submitted by the person corresponding to the second part of the ratio is 6. -/
theorem vikas_questions (total : ℕ) (r v a : ℕ) : 
  total = 24 →
  r + v + a = total →
  r = 7 * (total / (7 + 3 + 2)) →
  v = 3 * (total / (7 + 3 + 2)) →
  a = 2 * (total / (7 + 3 + 2)) →
  v = 6 :=
by sorry

end vikas_questions_l1075_107566
