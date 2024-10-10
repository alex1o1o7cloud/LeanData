import Mathlib

namespace sabrina_can_finish_series_l1787_178790

theorem sabrina_can_finish_series 
  (total_books : Nat) 
  (pages_per_book : Nat) 
  (books_read_first_month : Nat) 
  (reading_speed : Nat) 
  (total_days : Nat) 
  (h1 : total_books = 14)
  (h2 : pages_per_book = 200)
  (h3 : books_read_first_month = 4)
  (h4 : reading_speed = 40)
  (h5 : total_days = 60) :
  ∃ (pages_read : Nat), pages_read ≥ total_books * pages_per_book := by
  sorry

#check sabrina_can_finish_series

end sabrina_can_finish_series_l1787_178790


namespace angle_triple_supplement_l1787_178704

theorem angle_triple_supplement (x : ℝ) : x = 3 * (180 - x) → x = 135 := by
  sorry

end angle_triple_supplement_l1787_178704


namespace equal_roots_quadratic_l1787_178782

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x : ℝ, x^2 - p*x + p^2 = 0 → (∃! x : ℝ, x^2 - p*x + p^2 = 0)) :=
by sorry

end equal_roots_quadratic_l1787_178782


namespace largest_a_value_l1787_178712

theorem largest_a_value : ∃ (a_max : ℚ), 
  (∀ a : ℚ, (3 * a + 4) * (a - 2) = 7 * a → a ≤ a_max) ∧ 
  ((3 * a_max + 4) * (a_max - 2) = 7 * a_max) ∧
  a_max = 4 := by
sorry

end largest_a_value_l1787_178712


namespace bus_seat_capacity_l1787_178724

theorem bus_seat_capacity (left_seats right_seats back_seat_capacity total_capacity : ℕ) 
  (h1 : left_seats = 15)
  (h2 : right_seats = left_seats - 3)
  (h3 : back_seat_capacity = 8)
  (h4 : total_capacity = 89) :
  ∃ (seat_capacity : ℕ), 
    seat_capacity * (left_seats + right_seats) + back_seat_capacity = total_capacity ∧ 
    seat_capacity = 3 := by
sorry

end bus_seat_capacity_l1787_178724


namespace geometric_sequence_fifth_term_l1787_178786

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 5 = 16)
  (h_fourth : a 4 = 8) :
  a 5 = 16 :=
sorry

end geometric_sequence_fifth_term_l1787_178786


namespace first_month_sale_l1787_178760

/-- Given the sales data for a grocer over 6 months, prove that the first month's sale was 5420 --/
theorem first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) 
  (h1 : sale2 = 5660)
  (h2 : sale3 = 6200)
  (h3 : sale4 = 6350)
  (h4 : sale5 = 6500)
  (h5 : sale6 = 6470)
  (h6 : average = 6100) :
  let total := 6 * average
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  total - known_sales = 5420 := by
sorry

end first_month_sale_l1787_178760


namespace george_total_blocks_l1787_178709

/-- The total number of blocks George has when combining large, small, and medium blocks. -/
def total_blocks (large_boxes small_boxes large_per_box small_per_box case_boxes medium_per_box : ℕ) : ℕ :=
  (large_boxes * large_per_box) + (small_boxes * small_per_box) + (case_boxes * medium_per_box)

/-- Theorem stating that George has 86 blocks in total. -/
theorem george_total_blocks :
  total_blocks 2 3 6 8 5 10 = 86 := by
  sorry

end george_total_blocks_l1787_178709


namespace intersection_distance_l1787_178799

theorem intersection_distance (m b k : ℝ) (h1 : b ≠ 0) (h2 : 1 = 2 * m + b) :
  let f := fun x => x^2 + 6 * x - 4
  let g := fun x => m * x + b
  let d := |f k - g k|
  (m = 4 ∧ b = -7) → d = 9 :=
by sorry

end intersection_distance_l1787_178799


namespace five_students_two_teachers_arrangement_l1787_178721

/-- The number of ways two teachers can join a fixed line of students -/
def teacher_line_arrangements (num_students : ℕ) (num_teachers : ℕ) : ℕ :=
  (num_students + 1) * (num_students + 2)

/-- Theorem: With 5 students in fixed order and 2 teachers, there are 42 ways to arrange the line -/
theorem five_students_two_teachers_arrangement :
  teacher_line_arrangements 5 2 = 42 := by
  sorry

end five_students_two_teachers_arrangement_l1787_178721


namespace abc_maximum_l1787_178781

theorem abc_maximum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : 2*a + 4*b + 8*c = 16) : a*b*c ≤ 64/27 := by
  sorry

end abc_maximum_l1787_178781


namespace factorization_condition_l1787_178761

-- Define the polynomial
def polynomial (x y m : ℤ) : ℤ := x^2 + 5*x*y + x + 2*m*y - 10

-- Define what it means for a polynomial to be factorizable into linear factors with integer coefficients
def is_factorizable (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    polynomial x y m = (a*x + b*y + c) * (d*x + e*y + f)

-- State the theorem
theorem factorization_condition :
  ∀ m : ℤ, is_factorizable m ↔ m = 5 := by sorry

end factorization_condition_l1787_178761


namespace fixed_point_of_line_l1787_178770

/-- The fixed point of the line mx - y + 2m + 1 = 0 for all real m is (-2, 1) -/
theorem fixed_point_of_line (m : ℝ) : 
  (∀ x y : ℝ, m * x - y + 2 * m + 1 = 0 → (x = -2 ∧ y = 1)) ∧ 
  (m * (-2) - 1 + 2 * m + 1 = 0) := by
  sorry

end fixed_point_of_line_l1787_178770


namespace linear_equation_condition_l1787_178780

theorem linear_equation_condition (m : ℤ) : (|m| - 2 = 1 ∧ m - 3 ≠ 0) ↔ m = -3 := by
  sorry

end linear_equation_condition_l1787_178780


namespace partition_has_all_distances_l1787_178718

-- Define a partition of a metric space into three sets
def Partition (X : Type*) [MetricSpace X] (M₁ M₂ M₃ : Set X) : Prop :=
  (M₁ ∪ M₂ ∪ M₃ = Set.univ) ∧ (M₁ ∩ M₂ = ∅) ∧ (M₁ ∩ M₃ = ∅) ∧ (M₂ ∩ M₃ = ∅)

-- Define the property that a set contains two points with any positive distance
def HasAllDistances (X : Type*) [MetricSpace X] (M : Set X) : Prop :=
  ∀ a : ℝ, a > 0 → ∃ x y : X, x ∈ M ∧ y ∈ M ∧ dist x y = a

-- State the theorem
theorem partition_has_all_distances (X : Type*) [MetricSpace X] (M₁ M₂ M₃ : Set X) 
  (h : Partition X M₁ M₂ M₃) : 
  HasAllDistances X M₁ ∨ HasAllDistances X M₂ ∨ HasAllDistances X M₃ := by
  sorry


end partition_has_all_distances_l1787_178718


namespace cos_zero_degrees_l1787_178771

theorem cos_zero_degrees : Real.cos (0 * π / 180) = 1 := by sorry

end cos_zero_degrees_l1787_178771


namespace ellipse_equation_l1787_178752

/-- An ellipse with specific properties -/
structure Ellipse where
  -- Major axis is on the x-axis
  majorAxisOnX : Bool
  -- Length of the major axis
  majorAxisLength : ℝ
  -- Eccentricity
  eccentricity : ℝ
  -- Point on the ellipse
  pointOnEllipse : ℝ × ℝ

/-- The standard equation of an ellipse -/
def standardEquation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.majorAxisOnX = true)
  (h2 : e.majorAxisLength = 12)
  (h3 : e.eccentricity = 2/3)
  (h4 : e.pointOnEllipse = (-2, -4)) :
  standardEquation 36 20 = standardEquation 36 20 :=
by sorry

end ellipse_equation_l1787_178752


namespace triangle_ratio_equation_l1787_178793

/-- In a triangle ABC, given the ratios of sides to heights, prove the equation. -/
theorem triangle_ratio_equation (a b c h_a h_b h_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0) 
  (h_triangle : h_a * b = h_b * a ∧ h_b * c = h_c * b ∧ h_c * a = h_a * c) 
  (x y z : ℝ) (h_x : x = a / h_a) (h_y : y = b / h_b) (h_z : z = c / h_c) : 
  x^2 + y^2 + z^2 - 2*x*y - 2*y*z - 2*z*x + 4 = 0 := by
sorry

end triangle_ratio_equation_l1787_178793


namespace fraction_problem_l1787_178722

theorem fraction_problem (x : ℚ) : (x * 48 + 15 = 27) → x = 1/4 := by
  sorry

end fraction_problem_l1787_178722


namespace triangle_area_angle_l1787_178764

theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / (4 * Real.sqrt 3)
  ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
    S = 1/2 * a * b * Real.sin C ∧
    C = π/6 := by
  sorry

end triangle_area_angle_l1787_178764


namespace unique_intercept_line_l1787_178720

/-- A line passing through a point with equal absolute horizontal and vertical intercepts -/
structure InterceptLine where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  point_condition : 4 = m * 1 + b  -- line passes through (1, 4)
  intercept_condition : |m| = |b|  -- equal absolute intercepts

/-- There exists a unique line passing through (1, 4) with equal absolute horizontal and vertical intercepts -/
theorem unique_intercept_line : ∃! l : InterceptLine, True :=
  sorry

end unique_intercept_line_l1787_178720


namespace ocean_area_scientific_notation_l1787_178789

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem ocean_area_scientific_notation :
  toScientificNotation 361000000 = ScientificNotation.mk 3.61 8 sorry := by
  sorry

end ocean_area_scientific_notation_l1787_178789


namespace opposite_of_negative_three_l1787_178733

theorem opposite_of_negative_three : -((-3) : ℤ) = 3 := by sorry

end opposite_of_negative_three_l1787_178733


namespace negation_of_existential_proposition_l1787_178717

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, 3^x + x < 0) ↔ (∀ x : ℝ, 3^x + x ≥ 0) := by sorry

end negation_of_existential_proposition_l1787_178717


namespace measuring_cup_size_l1787_178751

/-- Given an 8 cup bag of flour, if removing 8 scoops leaves 6 cups,
    then the size of each scoop is 1/4 cup. -/
theorem measuring_cup_size (total_flour : ℚ) (scoops : ℕ) (remaining_flour : ℚ) 
    (scoop_size : ℚ) : 
    total_flour = 8 → 
    scoops = 8 → 
    remaining_flour = 6 → 
    total_flour - scoops * scoop_size = remaining_flour → 
    scoop_size = 1/4 := by
  sorry

#check measuring_cup_size

end measuring_cup_size_l1787_178751


namespace exists_same_color_rectangle_l1787_178798

/-- A color type with exactly three colors -/
inductive Color
| Red
| Green
| Blue

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- A rectangle in the plane -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (r : Rectangle) : Prop := sorry

/-- Predicate to check if all vertices of a rectangle have the same color -/
def SameColorVertices (r : Rectangle) (c : Coloring) : Prop :=
  c r.p1 = c r.p2 ∧ c r.p1 = c r.p3 ∧ c r.p1 = c r.p4

/-- Theorem: In a plane colored with 3 colors, there exists a rectangle whose vertices are all the same color -/
theorem exists_same_color_rectangle (c : Coloring) :
  ∃ (r : Rectangle), IsRectangle r ∧ SameColorVertices r c := by sorry

end exists_same_color_rectangle_l1787_178798


namespace inequality_solution_l1787_178797

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, 0 < x → x < y → f y < f x
axiom f_at_neg_three : f (-3) = 1

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 3}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f x < 1} = solution_set := by sorry

end inequality_solution_l1787_178797


namespace tan_negative_three_pi_fourth_l1787_178794

theorem tan_negative_three_pi_fourth : Real.tan (-3 * π / 4) = 1 := by
  sorry

end tan_negative_three_pi_fourth_l1787_178794


namespace line_points_Q_value_l1787_178725

/-- Given a line x = 8y + 5 passing through points (m, n) and (m + Q, n + p), where p = 0.25,
    prove that Q = 2. -/
theorem line_points_Q_value (m n Q p : ℝ) : 
  p = 0.25 →
  m = 8 * n + 5 →
  m + Q = 8 * (n + p) + 5 →
  Q = 2 := by
sorry

end line_points_Q_value_l1787_178725


namespace complex_modulus_l1787_178749

theorem complex_modulus (z : ℂ) : z = (1 + 2*Complex.I)/Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l1787_178749


namespace stock_value_indeterminate_l1787_178736

theorem stock_value_indeterminate (yield : ℝ) (market_value : ℝ) 
  (h_yield : yield = 0.08) (h_market_value : market_value = 150) :
  ∀ original_value : ℝ, 
  (original_value > 0 ∧ yield * original_value = market_value) ∨
  (original_value > 0 ∧ yield * original_value ≠ market_value) :=
by sorry

end stock_value_indeterminate_l1787_178736


namespace food_lasts_five_more_days_l1787_178746

/-- Calculates the number of additional days food lasts after more men join -/
def additional_days_food_lasts (initial_men : ℕ) (initial_days : ℕ) (days_before_joining : ℕ) (additional_men : ℕ) : ℕ :=
  let total_food := initial_men * initial_days
  let remaining_food := total_food - (initial_men * days_before_joining)
  let total_men := initial_men + additional_men
  remaining_food / total_men

/-- Proves that given the initial conditions, the food lasts for 5 additional days -/
theorem food_lasts_five_more_days :
  additional_days_food_lasts 760 22 2 2280 = 5 := by
  sorry

#eval additional_days_food_lasts 760 22 2 2280

end food_lasts_five_more_days_l1787_178746


namespace laundry_theorem_l1787_178726

/-- Represents the laundry problem --/
structure LaundryProblem where
  machine_capacity : ℕ  -- in pounds
  shirts_per_pound : ℕ
  pants_pairs_per_pound : ℕ
  shirts_to_wash : ℕ
  loads : ℕ

/-- Calculates the number of pants pairs that can be washed --/
def pants_to_wash (p : LaundryProblem) : ℕ :=
  let total_capacity := p.machine_capacity * p.loads
  let shirt_weight := p.shirts_to_wash / p.shirts_per_pound
  let remaining_capacity := total_capacity - shirt_weight
  remaining_capacity * p.pants_pairs_per_pound

/-- States the theorem for the laundry problem --/
theorem laundry_theorem (p : LaundryProblem) 
  (h1 : p.machine_capacity = 5)
  (h2 : p.shirts_per_pound = 4)
  (h3 : p.pants_pairs_per_pound = 2)
  (h4 : p.shirts_to_wash = 20)
  (h5 : p.loads = 3) :
  pants_to_wash p = 20 := by
  sorry

#eval pants_to_wash { 
  machine_capacity := 5,
  shirts_per_pound := 4,
  pants_pairs_per_pound := 2,
  shirts_to_wash := 20,
  loads := 3
}

end laundry_theorem_l1787_178726


namespace sandys_water_goal_l1787_178741

/-- Sandy's water drinking goal problem -/
theorem sandys_water_goal (water_per_interval : ℕ) (hours_per_interval : ℕ) (total_hours : ℕ) : 
  water_per_interval = 500 →
  hours_per_interval = 2 →
  total_hours = 12 →
  (water_per_interval * (total_hours / hours_per_interval)) / 1000 = 3 := by
  sorry

end sandys_water_goal_l1787_178741


namespace student_count_problem_l1787_178729

theorem student_count_problem (A B : ℕ) : 
  A = (5 : ℕ) * B / (7 : ℕ) →
  A + 3 = (4 : ℕ) * (B - 3) / (5 : ℕ) →
  A = 45 :=
by
  sorry

end student_count_problem_l1787_178729


namespace median_siblings_is_two_l1787_178795

/-- Represents the number of students for each sibling count -/
def sibling_distribution : List (Nat × Nat) :=
  [(0, 2), (1, 3), (2, 2), (3, 1), (4, 2), (5, 1)]

/-- Calculates the total number of students -/
def total_students : Nat :=
  sibling_distribution.foldl (fun acc (_, count) => acc + count) 0

/-- Finds the median position -/
def median_position : Nat :=
  (total_students + 1) / 2

/-- Theorem: The median number of siblings in Mrs. Thompson's History class is 2 -/
theorem median_siblings_is_two :
  let cumulative_count := sibling_distribution.foldl
    (fun acc (siblings, count) => 
      match acc with
      | [] => [(siblings, count)]
      | (_, prev_count) :: _ => (siblings, prev_count + count) :: acc
    ) []
  cumulative_count.reverse.find? (fun (_, count) => count ≥ median_position)
    = some (2, 7) := by sorry

end median_siblings_is_two_l1787_178795


namespace pirate_treasure_probability_l1787_178701

def num_islands : ℕ := 8
def prob_treasure_no_traps : ℚ := 1/3
def prob_treasure_and_traps : ℚ := 1/6
def prob_traps_no_treasure : ℚ := 1/6
def prob_neither : ℚ := 1/3

def target_treasure_islands : ℕ := 4
def target_treasure_and_traps_islands : ℕ := 2

theorem pirate_treasure_probability :
  let prob_treasure := prob_treasure_no_traps + prob_treasure_and_traps
  let prob_non_treasure := prob_traps_no_treasure + prob_neither
  (Nat.choose num_islands target_treasure_islands) *
  (Nat.choose target_treasure_islands target_treasure_and_traps_islands) *
  (prob_treasure ^ target_treasure_islands) *
  (prob_treasure_and_traps ^ target_treasure_and_traps_islands) *
  (prob_treasure_no_traps ^ (target_treasure_islands - target_treasure_and_traps_islands)) *
  (prob_non_treasure ^ (num_islands - target_treasure_islands)) =
  105 / 104976 := by
  sorry

end pirate_treasure_probability_l1787_178701


namespace f_minus_one_lt_f_one_l1787_178713

theorem f_minus_one_lt_f_one
  (f : ℝ → ℝ)
  (h_diff : Differentiable ℝ f)
  (h_eq : ∀ x, f x = x^2 + 2 * x * (deriv f 2)) :
  f (-1) < f 1 := by
sorry

end f_minus_one_lt_f_one_l1787_178713


namespace largest_initial_number_l1787_178791

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 + a + b + c + d + e = 200 ∧
    189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        n % x ≠ 0 ∧ n % y ≠ 0 ∧ n % z ≠ 0 ∧ n % w ≠ 0 ∧ n % v ≠ 0 :=
by sorry


end largest_initial_number_l1787_178791


namespace thomas_worked_four_weeks_l1787_178796

/-- The number of whole weeks Thomas worked given his weekly rate and total amount paid -/
def weeks_worked (weekly_rate : ℕ) (total_amount : ℕ) : ℕ :=
  (total_amount / weekly_rate : ℕ)

/-- Theorem stating that Thomas worked for 4 weeks -/
theorem thomas_worked_four_weeks :
  weeks_worked 4550 19500 = 4 := by
  sorry

end thomas_worked_four_weeks_l1787_178796


namespace geometric_sequence_ratio_l1787_178719

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 5 - (1/2) * a 7 = (1/2) * a 7 - a 6 →
  (a 1 + a 2 + a 3) / (a 2 + a 3 + a 4) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_ratio_l1787_178719


namespace max_visible_cubes_is_274_l1787_178711

/-- The dimension of the cube --/
def n : ℕ := 10

/-- The total number of unit cubes in the cube --/
def total_cubes : ℕ := n^3

/-- The number of unit cubes on one face of the cube --/
def face_cubes : ℕ := n^2

/-- The number of visible faces from a corner --/
def visible_faces : ℕ := 3

/-- The number of shared edges between visible faces --/
def shared_edges : ℕ := 3

/-- The length of each edge --/
def edge_length : ℕ := n

/-- The number of unit cubes along a shared edge, excluding the corner --/
def edge_cubes : ℕ := edge_length - 1

/-- The maximum number of visible unit cubes from a single point --/
def max_visible_cubes : ℕ := visible_faces * face_cubes - shared_edges * edge_cubes + 1

theorem max_visible_cubes_is_274 : max_visible_cubes = 274 := by
  sorry

end max_visible_cubes_is_274_l1787_178711


namespace mans_speed_against_current_l1787_178768

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem: Given the specified conditions, the man's speed against the current is 10 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

#eval speed_against_current 15 2.5

end mans_speed_against_current_l1787_178768


namespace quadratic_maximum_quadratic_maximum_achieved_l1787_178708

theorem quadratic_maximum (r : ℝ) : -3 * r^2 + 30 * r + 24 ≤ 99 :=
sorry

theorem quadratic_maximum_achieved : ∃ r : ℝ, -3 * r^2 + 30 * r + 24 = 99 :=
sorry

end quadratic_maximum_quadratic_maximum_achieved_l1787_178708


namespace mikes_net_salary_calculation_l1787_178734

-- Define the initial conditions
def freds_initial_salary : ℝ := 1000
def freds_bonus : ℝ := 500
def freds_investment_return : ℝ := 0.20
def mikes_salary_multiplier : ℝ := 10
def mikes_bonus_percentage : ℝ := 0.10
def mikes_investment_return : ℝ := 0.25
def mikes_salary_increase : ℝ := 0.40
def mikes_tax_rate : ℝ := 0.15

-- Define the theorem
theorem mikes_net_salary_calculation :
  let mikes_initial_salary := freds_initial_salary * mikes_salary_multiplier
  let mikes_initial_total := mikes_initial_salary * (1 + mikes_bonus_percentage)
  let mikes_investment_result := mikes_initial_total * (1 + mikes_investment_return)
  let mikes_new_salary := mikes_initial_salary * (1 + mikes_salary_increase)
  let mikes_tax := mikes_new_salary * mikes_tax_rate
  mikes_new_salary - mikes_tax = 11900 :=
by sorry

end mikes_net_salary_calculation_l1787_178734


namespace line_circle_intersection_l1787_178754

/-- A line y = x - b intersects a circle (x-2)^2 + y^2 = 1 at two distinct points
    if and only if b is in the open interval (2 - √2, 2 + √2) -/
theorem line_circle_intersection (b : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ - b ∧ y₂ = x₂ - b ∧
    (x₁ - 2)^2 + y₁^2 = 1 ∧
    (x₂ - 2)^2 + y₂^2 = 1) ↔ 
  (2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2) :=
sorry

end line_circle_intersection_l1787_178754


namespace arthur_walk_distance_l1787_178756

/-- The total number of blocks Arthur walked -/
def total_blocks : ℕ := 8 + 16

/-- The number of blocks that are one-third of a mile each -/
def first_blocks : ℕ := 10

/-- The length of each of the first blocks in miles -/
def first_block_length : ℚ := 1 / 3

/-- The length of each additional block in miles -/
def additional_block_length : ℚ := 1 / 4

/-- The total distance Arthur walked in miles -/
def total_distance : ℚ :=
  first_blocks * first_block_length + 
  (total_blocks - first_blocks) * additional_block_length

theorem arthur_walk_distance : total_distance = 41 / 6 := by
  sorry

end arthur_walk_distance_l1787_178756


namespace unripe_oranges_eaten_l1787_178700

theorem unripe_oranges_eaten (total : ℕ) (uneaten : ℕ) : 
  total = 96 →
  uneaten = 78 →
  (1 : ℚ) / 8 = (total / 2 - uneaten) / (total / 2) := by
  sorry

end unripe_oranges_eaten_l1787_178700


namespace cubic_root_reciprocal_sum_l1787_178710

theorem cubic_root_reciprocal_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 26*a - 8 = 0 → 
  b^3 - 15*b^2 + 26*b - 8 = 0 → 
  c^3 - 15*c^2 + 26*c - 8 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 109/16 := by
sorry

end cubic_root_reciprocal_sum_l1787_178710


namespace mrs_hilt_pizzas_l1787_178774

theorem mrs_hilt_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 16) :
  total_slices / slices_per_pizza = 2 :=
by sorry

end mrs_hilt_pizzas_l1787_178774


namespace children_after_addition_l1787_178778

-- Define the event parameters
def total_guests : Nat := 80
def num_men : Nat := 40
def num_women : Nat := num_men / 2
def added_children : Nat := 10

-- Theorem statement
theorem children_after_addition : 
  total_guests - (num_men + num_women) + added_children = 30 := by
  sorry

end children_after_addition_l1787_178778


namespace purely_imaginary_and_circle_l1787_178744

-- Define the complex number z
def z (a : ℝ) : ℂ := a * (1 + Complex.I) - 2 * Complex.I

-- State the theorem
theorem purely_imaginary_and_circle (a : ℝ) :
  (∃ b : ℝ, z a = Complex.I * b) →
  (a = 2 ∧ ∀ w : ℂ, Complex.abs w = 3 ↔ w.re ^ 2 + w.im ^ 2 = 3 ^ 2) :=
sorry

end purely_imaginary_and_circle_l1787_178744


namespace max_value_sum_reciprocals_l1787_178787

theorem max_value_sum_reciprocals (a b : ℝ) (h : a + b = 4) :
  (∃ x y : ℝ, x + y = 4 ∧ (1 / (x^2 + 1) + 1 / (y^2 + 1) ≤ 1 / (a^2 + 1) + 1 / (b^2 + 1))) ∧
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≤ (Real.sqrt 5 + 2) / 4 :=
sorry

end max_value_sum_reciprocals_l1787_178787


namespace wendy_recycling_points_l1787_178757

/-- Calculates the total points earned by Wendy for recycling cans and newspapers -/
def total_points (cans_recycled : ℕ) (newspapers_recycled : ℕ) : ℕ :=
  cans_recycled * 5 + newspapers_recycled * 10

/-- Proves that Wendy's total points earned is 75 given the problem conditions -/
theorem wendy_recycling_points :
  let cans_total : ℕ := 11
  let cans_recycled : ℕ := 9
  let newspapers_recycled : ℕ := 3
  total_points cans_recycled newspapers_recycled = 75 := by
  sorry

#eval total_points 9 3

end wendy_recycling_points_l1787_178757


namespace sin_cos_fourth_power_sum_l1787_178777

theorem sin_cos_fourth_power_sum (α : ℝ) (h : Real.sin α - Real.cos α = 1/2) :
  Real.sin α ^ 4 + Real.cos α ^ 4 = 23/32 := by sorry

end sin_cos_fourth_power_sum_l1787_178777


namespace golden_ratio_between_consecutive_integers_l1787_178745

theorem golden_ratio_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < (Real.sqrt 5 + 1) / 2) ∧ ((Real.sqrt 5 + 1) / 2 < b) → a + b = 3 :=
by sorry

end golden_ratio_between_consecutive_integers_l1787_178745


namespace lassis_from_mangoes_l1787_178759

/-- Given that 20 lassis can be made from 4 mangoes, prove that 80 lassis can be made from 16 mangoes. -/
theorem lassis_from_mangoes (make_lassis : ℕ → ℕ) 
  (h1 : make_lassis 4 = 20) 
  (h2 : ∀ x y : ℕ, make_lassis (x + y) = make_lassis x + make_lassis y) : 
  make_lassis 16 = 80 := by
  sorry

end lassis_from_mangoes_l1787_178759


namespace rectangle_to_square_dissection_l1787_178758

theorem rectangle_to_square_dissection :
  ∃ (a b c d : ℝ),
    -- Rectangle dimensions
    16 * 9 = a * b + c * d ∧
    -- Two parts form a square
    12 * 12 = a * b + c * d ∧
    -- Dimensions are positive
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    -- One dimension of each part matches the square
    (a = 12 ∨ b = 12 ∨ c = 12 ∨ d = 12) :=
by
  sorry

end rectangle_to_square_dissection_l1787_178758


namespace pills_per_week_calculation_l1787_178772

/-- Calculates the number of pills taken in a week given the frequency of pill intake -/
def pills_per_week (hours_between_pills : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem stating that taking a pill every 6 hours results in 28 pills per week -/
theorem pills_per_week_calculation :
  pills_per_week 6 24 7 = 28 := by
  sorry

#eval pills_per_week 6 24 7

end pills_per_week_calculation_l1787_178772


namespace tim_change_l1787_178755

/-- The change Tim received after buying a candy bar -/
def change (initial_amount : ℕ) (candy_cost : ℕ) : ℕ :=
  initial_amount - candy_cost

/-- Theorem stating that Tim's change is 5 cents -/
theorem tim_change :
  change 50 45 = 5 := by
  sorry

end tim_change_l1787_178755


namespace m_range_l1787_178707

theorem m_range (x m : ℝ) : 
  (∀ x, x^2 + 3*x - 4 < 0 → (x - m)^2 > 3*(x - m)) ∧ 
  (∃ x, (x - m)^2 > 3*(x - m) ∧ x^2 + 3*x - 4 ≥ 0) → 
  m ≥ 1 ∨ m ≤ -7 :=
by sorry

end m_range_l1787_178707


namespace billy_tickets_l1787_178785

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def tickets_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * tickets_per_ride

theorem billy_tickets : total_tickets = 50 := by
  sorry

end billy_tickets_l1787_178785


namespace erased_number_proof_l1787_178723

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x : ℚ) / (n - 1 : ℚ) = 35 + 7/17 →
  x = 7 := by
sorry

end erased_number_proof_l1787_178723


namespace M_not_finite_union_of_aps_l1787_178705

-- Define the set M
def M : Set ℕ := {n : ℕ | ∀ x y : ℕ, (1 : ℚ) / x + (1 : ℚ) / y ≠ 3 / n}

-- Define what it means for a set to be representable as a finite union of arithmetic progressions
def is_finite_union_of_aps (S : Set ℕ) : Prop :=
  ∃ (n : ℕ) (a d : Fin n → ℕ), S = ⋃ i, {k : ℕ | ∃ j : ℕ, k = a i + j * d i}

-- State the theorem
theorem M_not_finite_union_of_aps :
  (∀ n : ℕ, n ∉ M → ∀ m : ℕ, m * n ∉ M) →
  (∀ k : ℕ, k > 0 → (7 : ℕ) ^ k ∈ M) →
  ¬ is_finite_union_of_aps M :=
sorry

end M_not_finite_union_of_aps_l1787_178705


namespace T_formula_l1787_178728

def T : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * (n + 3) - 8) * T n

theorem T_formula (n : ℕ) : T n = n.factorial + 2^n := by
  sorry

end T_formula_l1787_178728


namespace sqrt_inequality_l1787_178773

-- Define the variables and conditions
theorem sqrt_inequality (C : ℝ) (hC : C > 1) :
  Real.sqrt (C + 1) - Real.sqrt C < Real.sqrt C - Real.sqrt (C - 1) :=
by
  sorry

end sqrt_inequality_l1787_178773


namespace sin_cos_identity_l1787_178775

theorem sin_cos_identity : Real.sin (15 * π / 180) * Real.sin (105 * π / 180) - 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l1787_178775


namespace rent_calculation_l1787_178703

theorem rent_calculation (monthly_earnings : ℝ) 
  (h1 : monthly_earnings * 0.07 + monthly_earnings * 0.5 + 817 = monthly_earnings) : 
  monthly_earnings * 0.07 = 133 := by
  sorry

end rent_calculation_l1787_178703


namespace unique_valid_number_l1787_178748

def is_valid_number (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∈ Finset.range 10 → (n / 10^(9-k)) % k = 0) ∧
  (∀ d : ℕ, d ∈ Finset.range 10 → (∃! i : ℕ, i ∈ Finset.range 9 ∧ (n / 10^i) % 10 = d))

theorem unique_valid_number :
  ∃! n : ℕ, n = 381654729 ∧ is_valid_number n :=
sorry

end unique_valid_number_l1787_178748


namespace necessary_but_not_sufficient_condition_l1787_178702

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |4*x - 3| < a ∧ a > 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 < 0}

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B) → 0 < a ∧ a ≤ 5 :=
by sorry

end necessary_but_not_sufficient_condition_l1787_178702


namespace fair_distribution_correctness_l1787_178776

/-- Represents the amount of bread each person has initially -/
structure BreadDistribution where
  personA : ℚ
  personB : ℚ

/-- Represents the fair distribution of currency -/
structure CurrencyDistribution where
  personA : ℚ
  personB : ℚ

/-- Calculates the fair distribution of currency based on initial bread distribution -/
def calculateFairDistribution (initial : BreadDistribution) (totalCurrency : ℚ) : CurrencyDistribution :=
  sorry

theorem fair_distribution_correctness 
  (initial : BreadDistribution)
  (h1 : initial.personA = 3)
  (h2 : initial.personB = 2)
  (totalCurrency : ℚ)
  (h3 : totalCurrency = 50) :
  let result := calculateFairDistribution initial totalCurrency
  result.personA = 40 ∧ result.personB = 10 := by
  sorry

end fair_distribution_correctness_l1787_178776


namespace same_group_probability_correct_l1787_178732

def card_count : ℕ := 20
def people_count : ℕ := 4
def drawn_card1 : ℕ := 5
def drawn_card2 : ℕ := 14

def same_group_probability : ℚ := 7/51

theorem same_group_probability_correct :
  let remaining_cards := card_count - 2
  let smaller_group_cases := (card_count - drawn_card2) * (card_count - drawn_card2 - 1) / 2
  let larger_group_cases := (drawn_card1 - 1) * (drawn_card1 - 2) / 2
  let favorable_outcomes := smaller_group_cases + larger_group_cases
  let total_outcomes := remaining_cards * (remaining_cards - 1) / 2
  (favorable_outcomes : ℚ) / total_outcomes = same_group_probability := by
  sorry

end same_group_probability_correct_l1787_178732


namespace consecutive_even_integers_sum_l1787_178766

theorem consecutive_even_integers_sum (x : ℕ) (h : x > 0) :
  (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2)) →
  (x - 2) + x + (x + 2) = 24 := by
sorry

end consecutive_even_integers_sum_l1787_178766


namespace triangle_properties_l1787_178739

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b * sin(A) = (√3/2) * a, a = 2c, and b = 2√6,
    then the measure of angle B is π/3 and the area of the triangle is 4√3. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- acute triangle condition
  b * Real.sin A = (Real.sqrt 3 / 2) * a →  -- given condition
  a = 2 * c →  -- given condition
  b = 2 * Real.sqrt 6 →  -- given condition
  B = π / 3 ∧ (1 / 2) * a * c * Real.sin B = 4 * Real.sqrt 3 := by
  sorry

end triangle_properties_l1787_178739


namespace factorial_sum_of_powers_of_two_l1787_178737

theorem factorial_sum_of_powers_of_two (n : ℕ) :
  (∃ a b : ℕ, n.factorial = 2^a + 2^b) ↔ n = 3 ∨ n = 4 := by
  sorry

end factorial_sum_of_powers_of_two_l1787_178737


namespace diagonals_in_nonagon_l1787_178727

/-- The number of diagonals in a regular nine-sided polygon -/
theorem diagonals_in_nonagon : 
  (let n : ℕ := 9
   let total_connections := n.choose 2
   let num_sides := n
   total_connections - num_sides) = 27 := by
sorry

end diagonals_in_nonagon_l1787_178727


namespace triangle_angle_inequality_l1787_178765

theorem triangle_angle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) : 
  π * (1/A + 1/B + 1/C) ≥ (Real.sin A + Real.sin B + Real.sin C) * 
    (1/Real.sin A + 1/Real.sin B + 1/Real.sin C) := by
  sorry

end triangle_angle_inequality_l1787_178765


namespace flower_bed_lilies_l1787_178740

/-- Given a flower bed with roses, tulips, and lilies, prove the number of lilies. -/
theorem flower_bed_lilies (roses tulips lilies : ℕ) : 
  roses = 57 → 
  tulips = 82 → 
  tulips = roses + lilies + 13 → 
  lilies = 12 := by
sorry


end flower_bed_lilies_l1787_178740


namespace weight_of_b_l1787_178769

/-- Given three weights a, b, and c, prove that b = 31 when:
    1. The average of a, b, and c is 45
    2. The average of a and b is 40
    3. The average of b and c is 43 -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 31 := by
  sorry

end weight_of_b_l1787_178769


namespace smallest_multiple_of_seven_factorial_l1787_178767

theorem smallest_multiple_of_seven_factorial : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k < 7 → ¬(m ∣ Nat.factorial k)) ∧ 
  (m ∣ Nat.factorial 7) ∧
  (∀ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k < 7 → ¬(n ∣ Nat.factorial k)) ∧ (n ∣ Nat.factorial 7) → m ≤ n) :=
by
  use 5040
  sorry

end smallest_multiple_of_seven_factorial_l1787_178767


namespace probability_of_selecting_girl_l1787_178743

theorem probability_of_selecting_girl (num_boys num_girls : ℕ) 
  (h_boys : num_boys = 3) 
  (h_girls : num_girls = 2) : 
  (num_girls : ℚ) / ((num_boys + num_girls) : ℚ) = 2 / 5 :=
by sorry

end probability_of_selecting_girl_l1787_178743


namespace sum_ac_l1787_178763

theorem sum_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 5) : 
  a + c = 42 / 5 := by
  sorry

end sum_ac_l1787_178763


namespace negation_of_statement_l1787_178779

theorem negation_of_statement : 
  (¬(∀ a : ℝ, a ≠ 0 → a^2 > 0)) ↔ (∃ a : ℝ, a = 0 ∧ a^2 ≤ 0) :=
by sorry

end negation_of_statement_l1787_178779


namespace average_fish_is_75_l1787_178714

/-- The number of fish in Boast Pool -/
def boast_pool : ℕ := 75

/-- The number of fish in Onum Lake -/
def onum_lake : ℕ := boast_pool + 25

/-- The number of fish in Riddle Pond -/
def riddle_pond : ℕ := onum_lake / 2

/-- The total number of fish in all three bodies of water -/
def total_fish : ℕ := boast_pool + onum_lake + riddle_pond

/-- The number of bodies of water -/
def num_bodies : ℕ := 3

/-- Theorem stating that the average number of fish in all three bodies of water is 75 -/
theorem average_fish_is_75 : total_fish / num_bodies = 75 := by
  sorry

end average_fish_is_75_l1787_178714


namespace routes_between_plains_cities_l1787_178750

theorem routes_between_plains_cities 
  (total_cities : Nat) 
  (mountainous_cities : Nat) 
  (plains_cities : Nat) 
  (total_routes : Nat) 
  (mountainous_routes : Nat) : 
  total_cities = 100 → 
  mountainous_cities = 30 → 
  plains_cities = 70 → 
  total_routes = 150 → 
  mountainous_routes = 21 → 
  ∃ (plains_routes : Nat), plains_routes = 81 ∧ 
    plains_routes + mountainous_routes + (total_routes - plains_routes - mountainous_routes) = total_routes := by
  sorry

end routes_between_plains_cities_l1787_178750


namespace rob_has_three_dimes_l1787_178731

/-- Represents the number of coins of each type Rob has -/
structure RobsCoins where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value of Rob's coins in cents -/
def totalValue (coins : RobsCoins) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem stating that given Rob's coin counts and total value, he must have 3 dimes -/
theorem rob_has_three_dimes :
  ∀ (coins : RobsCoins),
    coins.quarters = 7 →
    coins.nickels = 5 →
    coins.pennies = 12 →
    totalValue coins = 242 →
    coins.dimes = 3 := by
  sorry


end rob_has_three_dimes_l1787_178731


namespace xy_value_l1787_178753

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l1787_178753


namespace fence_area_inequality_l1787_178742

theorem fence_area_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end fence_area_inequality_l1787_178742


namespace artist_paint_usage_l1787_178715

/-- The amount of paint used for all paintings --/
def total_paint_used (large_paint small_paint large_count small_count : ℕ) : ℕ :=
  large_paint * large_count + small_paint * small_count

/-- Proof that the artist used 17 ounces of paint --/
theorem artist_paint_usage : total_paint_used 3 2 3 4 = 17 := by
  sorry

end artist_paint_usage_l1787_178715


namespace tetrahedron_properties_l1787_178747

-- Define the vertices of the tetrahedron
def A₁ : ℝ × ℝ × ℝ := (1, 5, -7)
def A₂ : ℝ × ℝ × ℝ := (-3, 6, 3)
def A₃ : ℝ × ℝ × ℝ := (-2, 7, 3)
def A₄ : ℝ × ℝ × ℝ := (-4, 8, -12)

-- Define a function to calculate the volume of a tetrahedron
def tetrahedronVolume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the height of a tetrahedron
def tetrahedronHeight (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem stating the volume and height of the tetrahedron
theorem tetrahedron_properties :
  tetrahedronVolume A₁ A₂ A₃ A₄ = 17.5 ∧
  tetrahedronHeight A₁ A₂ A₃ A₄ = 7 := by sorry

end tetrahedron_properties_l1787_178747


namespace log_823_bounds_sum_l1787_178738

theorem log_823_bounds_sum : ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 823 / Real.log 10 ∧ Real.log 823 / Real.log 10 < (d : ℝ) ∧ c + d = 5 := by
  sorry

end log_823_bounds_sum_l1787_178738


namespace periodic_sequence_prime_period_l1787_178788

/-- A sequence a is periodic with period m if a(m+n) = a(n) for all n -/
def isPeriodic (a : ℕ → ℂ) (m : ℕ) : Prop :=
  ∀ n, a (m + n) = a n

/-- m is the smallest positive period of sequence a -/
def isSmallestPeriod (a : ℕ → ℂ) (m : ℕ) : Prop :=
  isPeriodic a m ∧ ∀ k, 0 < k → k < m → ¬isPeriodic a k

/-- q is an m-th root of unity -/
def isRootOfUnity (q : ℂ) (m : ℕ) : Prop :=
  q ^ m = 1

theorem periodic_sequence_prime_period
  (q : ℂ) (m : ℕ) 
  (h1 : isSmallestPeriod (fun n => q^n) m)
  (h2 : m ≥ 2)
  (h3 : Nat.Prime m) :
  isRootOfUnity q m ∧ q ≠ 1 := by
  sorry

end periodic_sequence_prime_period_l1787_178788


namespace solution_set_f_less_than_8_range_of_m_for_solvable_inequality_l1787_178762

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) < 8
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -5/2 < x ∧ x < 3/2} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_solvable_inequality :
  {m : ℝ | ∃ x, f x ≤ |3*m + 1|} = 
    {m : ℝ | m ≤ -5/3 ∨ m ≥ 1} := by sorry

end solution_set_f_less_than_8_range_of_m_for_solvable_inequality_l1787_178762


namespace ellipse_satisfies_equation_l1787_178706

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  -- Line passing through f2 perpendicular to x-axis
  line : Set (ℝ × ℝ)
  -- Intersection points of the line with the ellipse
  a : ℝ × ℝ
  b : ℝ × ℝ
  -- Distance between intersection points
  ab_distance : ℝ
  -- Properties
  f1_def : f1 = (-1, 0)
  f2_def : f2 = (1, 0)
  line_def : line = {p : ℝ × ℝ | p.1 = 1}
  ab_on_line : a ∈ line ∧ b ∈ line
  ab_distance_def : ab_distance = 3
  
/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Theorem stating that the given ellipse satisfies the equation -/
theorem ellipse_satisfies_equation (e : Ellipse) :
  ∀ x y, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation e p.1 p.2} ↔ 
    (∃ t, 0 ≤ t ∧ t ≤ 1 ∧ 
      (x - e.f1.1)^2 + (y - e.f1.2)^2 + 
      (x - e.f2.1)^2 + (y - e.f2.2)^2 = 
      (2 * Real.sqrt ((x - e.f1.1)^2 + (y - e.f1.2)^2 + (x - e.f2.1)^2 + (y - e.f2.2)^2))^2) :=
sorry

end ellipse_satisfies_equation_l1787_178706


namespace max_value_of_2xy_l1787_178792

theorem max_value_of_2xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → 2 * x * y ≤ 2 * a * b → 2 * x * y ≤ 8 :=
by sorry

end max_value_of_2xy_l1787_178792


namespace seven_pow_minus_three_times_two_pow_eq_one_l1787_178716

theorem seven_pow_minus_three_times_two_pow_eq_one
  (m n : ℕ+) : 7^(m:ℕ) - 3 * 2^(n:ℕ) = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end seven_pow_minus_three_times_two_pow_eq_one_l1787_178716


namespace max_area_AOB_l1787_178783

-- Define the circles E and F
def circle_E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def circle_F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C (locus of center of P)
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a line l
def line_l (m n x y : ℝ) : Prop := x = m * y + n

-- Define points A and B on curve C and line l
def point_on_C_and_l (x y m n : ℝ) : Prop :=
  curve_C x y ∧ line_l m n x y

-- Define midpoint M of AB
def midpoint_M (xm ym xa ya xb yb : ℝ) : Prop :=
  xm = (xa + xb) / 2 ∧ ym = (ya + yb) / 2

-- Define |OM| = 1
def OM_unit_length (xm ym : ℝ) : Prop :=
  xm^2 + ym^2 = 1

-- Main theorem
theorem max_area_AOB :
  ∀ (xa ya xb yb xm ym m n : ℝ),
  point_on_C_and_l xa ya m n →
  point_on_C_and_l xb yb m n →
  midpoint_M xm ym xa ya xb yb →
  OM_unit_length xm ym →
  ∃ (S : ℝ), S ≤ 1 ∧
  (∀ (S' : ℝ), S' = abs ((xa * yb - xb * ya) / 2) → S' ≤ S) :=
sorry

end max_area_AOB_l1787_178783


namespace correct_quotient_calculation_l1787_178735

theorem correct_quotient_calculation (A B : ℕ) (dividend : ℕ) : 
  A > 0 → 
  A * 100 + B * 10 > 0 →
  dividend / (A * 10 + B) = 210 → 
  dividend / (A * 100 + B * 10) = 21 := by
sorry

end correct_quotient_calculation_l1787_178735


namespace jimin_english_score_l1787_178730

def jimin_scores (science social_studies english : ℕ) : Prop :=
  social_studies = science + 6 ∧
  science = 87 ∧
  (science + social_studies + english) / 3 = 92

theorem jimin_english_score :
  ∀ science social_studies english : ℕ,
  jimin_scores science social_studies english →
  english = 96 := by sorry

end jimin_english_score_l1787_178730


namespace point_on_y_axis_l1787_178784

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the y-axis -/
def on_y_axis (p : CartesianPoint) : Prop := p.x = 0

/-- Theorem: A point with x-coordinate 0 lies on the y-axis -/
theorem point_on_y_axis (p : CartesianPoint) (h : p.x = 0) : on_y_axis p := by
  sorry

#check point_on_y_axis

end point_on_y_axis_l1787_178784
