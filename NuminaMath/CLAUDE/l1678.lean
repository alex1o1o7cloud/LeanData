import Mathlib

namespace NUMINAMATH_CALUDE_james_weekly_beats_l1678_167827

/-- The number of beats James hears per week -/
def beats_per_week : ℕ :=
  let beats_per_minute : ℕ := 200
  let hours_per_day : ℕ := 2
  let minutes_per_hour : ℕ := 60
  let days_per_week : ℕ := 7
  beats_per_minute * hours_per_day * minutes_per_hour * days_per_week

/-- Theorem stating that James hears 168,000 beats per week -/
theorem james_weekly_beats : beats_per_week = 168000 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_beats_l1678_167827


namespace NUMINAMATH_CALUDE_laundry_time_proof_l1678_167844

/-- Calculates the total time for laundry given the number of loads, time per load for washing, and time for drying. -/
def totalLaundryTime (numLoads : ℕ) (washTimePerLoad : ℕ) (dryTime : ℕ) : ℕ :=
  numLoads * washTimePerLoad + dryTime

/-- Proves that given the specified conditions, the total laundry time is 165 minutes. -/
theorem laundry_time_proof :
  totalLaundryTime 2 45 75 = 165 := by
  sorry

end NUMINAMATH_CALUDE_laundry_time_proof_l1678_167844


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l1678_167813

/-- The number of people born every two seconds in a city -/
def birth_rate : ℕ := sorry

/-- The death rate in the city (people per two seconds) -/
def death_rate : ℕ := 1

/-- The net population increase in one day -/
def daily_net_increase : ℕ := 259200

/-- The number of two-second intervals in a day -/
def intervals_per_day : ℕ := 24 * 60 * 60 / 2

theorem birth_rate_calculation : 
  birth_rate = (daily_net_increase / intervals_per_day) + death_rate := by
  sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l1678_167813


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l1678_167890

/-- The minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (rowing_speed : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (h1 : distance_to_shore = 2)
  (h2 : rowing_speed = 3)
  (h3 : water_intake_rate = 6)
  (h4 : boat_capacity = 60) :
  ∃ (min_bailing_rate : ℝ),
    min_bailing_rate = 4.5 ∧
    ∀ (bailing_rate : ℝ),
      bailing_rate ≥ min_bailing_rate →
      (water_intake_rate - bailing_rate) * (distance_to_shore / rowing_speed * 60) ≤ boat_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l1678_167890


namespace NUMINAMATH_CALUDE_bus_boarding_problem_l1678_167898

theorem bus_boarding_problem (total_rows : Nat) (seats_per_row : Nat) 
  (initial_boarding : Nat) (first_stop_exit : Nat) (second_stop_boarding : Nat) 
  (second_stop_exit : Nat) (final_empty_seats : Nat) :
  let total_seats := total_rows * seats_per_row
  let empty_seats_after_start := total_seats - initial_boarding
  let first_stop_boarding := total_seats - empty_seats_after_start + first_stop_exit - 
    (total_seats - (empty_seats_after_start - (second_stop_boarding - second_stop_exit) - final_empty_seats))
  total_rows = 23 →
  seats_per_row = 4 →
  initial_boarding = 16 →
  first_stop_exit = 3 →
  second_stop_boarding = 17 →
  second_stop_exit = 10 →
  final_empty_seats = 57 →
  first_stop_boarding = 15 := by
    sorry

#check bus_boarding_problem

end NUMINAMATH_CALUDE_bus_boarding_problem_l1678_167898


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l1678_167840

/-- A cubic polynomial with coefficients a, b, and c -/
def cubic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The condition for the roots of a cubic polynomial to form an arithmetic progression -/
def arithmetic_progression_condition (a b c : ℝ) : Prop :=
  2 * a^3 / 27 - a * b / 3 + c = 0

/-- Theorem stating that the roots of a cubic polynomial form an arithmetic progression
    if and only if the coefficients satisfy the arithmetic progression condition -/
theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x y z : ℝ, x - y = y - z ∧ 
    (∀ t : ℝ, cubic_polynomial a b c t = 0 ↔ t = x ∨ t = y ∨ t = z)) ↔ 
  arithmetic_progression_condition a b c :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l1678_167840


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l1678_167889

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (outer1 outer2 shaded : Rectangle) 
  (h1 : outer1.width = 12 ∧ outer1.height = 9)
  (h2 : outer2.width = 5 ∧ outer2.height = 3)
  (h3 : shaded.width = 6 ∧ shaded.height = 3)
  (h4 : area outer1 + area outer2 = 117)
  (h5 : area shaded = 108) :
  ∃ (non_shaded : Rectangle), perimeter non_shaded = 12 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l1678_167889


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1678_167854

/-- A function that generates all three-digit numbers using the digits 5, 6, and 7 only once -/
def threeDigitNumbers : List Nat := sorry

/-- The smallest three-digit number formed using 5, 6, and 7 only once -/
def smallestNumber : Nat := sorry

/-- The largest three-digit number formed using 5, 6, and 7 only once -/
def largestNumber : Nat := sorry

/-- Theorem stating that the sum of the largest and smallest three-digit numbers
    formed using 5, 6, and 7 only once is 1332 -/
theorem sum_of_largest_and_smallest : smallestNumber + largestNumber = 1332 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l1678_167854


namespace NUMINAMATH_CALUDE_family_gallery_photos_l1678_167846

/-- Proves that the initial number of photos in the family gallery was 400 --/
theorem family_gallery_photos : 
  ∀ (P : ℕ), 
  (P + (P / 2) + (P / 2 + 120) = 920) → 
  P = 400 := by
sorry

end NUMINAMATH_CALUDE_family_gallery_photos_l1678_167846


namespace NUMINAMATH_CALUDE_triangle_count_is_102_l1678_167803

/-- Represents a rectangle divided into a 6x2 grid with diagonal lines -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  grid_width : ℕ
  grid_height : ℕ
  has_diagonals : Bool

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles in the specific GridRectangle is 102 -/
theorem triangle_count_is_102 :
  ∃ (rect : GridRectangle),
    rect.width = 6 ∧
    rect.height = 2 ∧
    rect.grid_width = 6 ∧
    rect.grid_height = 2 ∧
    rect.has_diagonals = true ∧
    count_triangles rect = 102 :=
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_102_l1678_167803


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_opposite_mutually_exclusive_not_opposite_l1678_167849

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 2

/-- Represents the total number of students in the group -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of students selected -/
def selected_students : ℕ := 2

/-- Represents the event of exactly one boy being selected -/
def one_boy_selected (k : ℕ) : Prop := k = 1

/-- Represents the event of exactly two boys being selected -/
def two_boys_selected (k : ℕ) : Prop := k = 2

/-- States that the events are mutually exclusive -/
theorem events_mutually_exclusive : 
  ∀ k : ℕ, ¬(one_boy_selected k ∧ two_boys_selected k) :=
sorry

/-- States that the events are not opposite -/
theorem events_not_opposite : 
  ∃ k : ℕ, ¬(one_boy_selected k ∨ two_boys_selected k) :=
sorry

/-- Main theorem stating that the events are mutually exclusive but not opposite -/
theorem mutually_exclusive_not_opposite : 
  (∀ k : ℕ, ¬(one_boy_selected k ∧ two_boys_selected k)) ∧
  (∃ k : ℕ, ¬(one_boy_selected k ∨ two_boys_selected k)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_opposite_mutually_exclusive_not_opposite_l1678_167849


namespace NUMINAMATH_CALUDE_cross_sectional_area_of_cone_l1678_167881

-- Define the cone
structure Cone :=
  (baseRadius : ℝ)
  (height : ℝ)

-- Define the cutting plane
structure CuttingPlane :=
  (distanceFromBase : ℝ)
  (isParallelToBase : Bool)

-- Theorem statement
theorem cross_sectional_area_of_cone (c : Cone) (p : CuttingPlane) :
  c.baseRadius = 2 →
  p.distanceFromBase = c.height / 2 →
  p.isParallelToBase = true →
  (π : ℝ) = π := by sorry

end NUMINAMATH_CALUDE_cross_sectional_area_of_cone_l1678_167881


namespace NUMINAMATH_CALUDE_problem_statement_l1678_167891

theorem problem_statement (a b : ℝ) (h : 2 * a^2 - 3 * b + 5 = 0) :
  9 * b - 6 * a^2 + 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1678_167891


namespace NUMINAMATH_CALUDE_course_selection_schemes_l1678_167894

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def min_courses : ℕ := 2
def max_courses : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 +
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_schemes : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l1678_167894


namespace NUMINAMATH_CALUDE_random_placement_probability_l1678_167892

-- Define the number of bins and items
def num_bins : ℕ := 4
def num_items : ℕ := 4

-- Define the probability of correct placement
def correct_placement_probability : ℚ := 1 / (num_bins.factorial)

-- Theorem statement
theorem random_placement_probability :
  correct_placement_probability = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_random_placement_probability_l1678_167892


namespace NUMINAMATH_CALUDE_largest_d_for_two_in_range_l1678_167853

/-- The function g(x) defined as x^2 - 6x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- Theorem stating that the largest value of d for which 2 is in the range of g(x) is 11 -/
theorem largest_d_for_two_in_range :
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), g d x = 2) → (e ≤ d)) ∧
  (∃ (x : ℝ), g 11 x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_two_in_range_l1678_167853


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1678_167821

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 3 + a 10 + a 11 = 40 →
  a 6 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1678_167821


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l1678_167832

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l1678_167832


namespace NUMINAMATH_CALUDE_bicycle_count_l1678_167825

theorem bicycle_count (tricycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  tricycles = 7 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ bicycles : ℕ, bicycles * bicycle_wheels + tricycles * tricycle_wheels = total_wheels ∧ bicycles = 16 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_count_l1678_167825


namespace NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l1678_167820

/-- A symmetric scanning code is a 5x5 grid that remains unchanged when rotated by multiples of 90° or reflected across diagonal or midpoint lines. -/
def SymmetricScanningCode : Type := Unit

/-- The number of distinct symmetry groups in a 5x5 symmetric scanning code -/
def numSymmetryGroups : ℕ := 5

/-- The number of color choices for each symmetry group -/
def numColorChoices : ℕ := 2

/-- The total number of color combinations for all symmetry groups -/
def totalColorCombinations : ℕ := numColorChoices ^ numSymmetryGroups

/-- The number of invalid color combinations (all white or all black) -/
def invalidColorCombinations : ℕ := 2

theorem count_symmetric_scanning_codes :
  (totalColorCombinations - invalidColorCombinations : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l1678_167820


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l1678_167834

/-- Given a line l with polar equation θ = 2π/3, its Cartesian coordinate equation is √3x + y = 0 -/
theorem polar_to_cartesian_line (l : Set (ℝ × ℝ)) :
  (∀ (r θ : ℝ), (r, θ) ∈ l ↔ θ = 2 * Real.pi / 3) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ Real.sqrt 3 * x + y = 0) :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l1678_167834


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1678_167841

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1678_167841


namespace NUMINAMATH_CALUDE_determine_d_value_l1678_167893

theorem determine_d_value : ∃ d : ℝ, 
  (∀ x : ℝ, x * (2 * x + 3) < d ↔ -5/2 < x ∧ x < 3) → d = 15 := by
  sorry

end NUMINAMATH_CALUDE_determine_d_value_l1678_167893


namespace NUMINAMATH_CALUDE_sum_equals_369_l1678_167857

theorem sum_equals_369 : 333 + 33 + 3 = 369 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_369_l1678_167857


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1678_167807

theorem sufficient_not_necessary (a b : ℝ) :
  (a < b ∧ b < 0 → a^2 > b^2) ∧
  ¬(a^2 > b^2 → a < b ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1678_167807


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1678_167899

/-- Given two vectors a and b in R², prove that if they are parallel,
    then the magnitude of a + 2b is 3√5. -/
theorem parallel_vectors_magnitude (t : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, t]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  ‖(a + 2 • b)‖ = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1678_167899


namespace NUMINAMATH_CALUDE_madeline_rent_correct_l1678_167862

/-- Calculate the amount Madeline needs for rent given her expenses, savings, hourly wage, and hours worked -/
def rent_amount (hourly_wage : ℝ) (hours_worked : ℝ) (groceries : ℝ) (medical : ℝ) (utilities : ℝ) (savings : ℝ) : ℝ :=
  hourly_wage * hours_worked - (groceries + medical + utilities + savings)

/-- Theorem stating that Madeline's rent amount is correct -/
theorem madeline_rent_correct (hourly_wage : ℝ) (hours_worked : ℝ) (groceries : ℝ) (medical : ℝ) (utilities : ℝ) (savings : ℝ) :
  rent_amount hourly_wage hours_worked groceries medical utilities savings = 1210 :=
by
  sorry

#eval rent_amount 15 138 400 200 60 200

end NUMINAMATH_CALUDE_madeline_rent_correct_l1678_167862


namespace NUMINAMATH_CALUDE_sci_fi_readers_l1678_167814

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) (sci_fi : ℕ) : 
  total = 400 → literary = 230 → both = 80 → 
  total = sci_fi + literary - both →
  sci_fi = 250 := by
sorry

end NUMINAMATH_CALUDE_sci_fi_readers_l1678_167814


namespace NUMINAMATH_CALUDE_ordering_proof_l1678_167819

theorem ordering_proof (a b c : ℝ) : 
  a = (1/2)^(1/3) → b = (1/3)^(1/2) → c = Real.log (3/Real.pi) → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ordering_proof_l1678_167819


namespace NUMINAMATH_CALUDE_substitution_remainder_l1678_167882

/-- Represents the number of players on the roster. -/
def totalPlayers : ℕ := 15

/-- Represents the number of players in the starting lineup. -/
def startingLineup : ℕ := 10

/-- Represents the number of substitute players. -/
def substitutes : ℕ := 5

/-- Represents the maximum number of substitutions allowed. -/
def maxSubstitutions : ℕ := 2

/-- Calculates the number of ways to make substitutions given the number of substitutions. -/
def substitutionWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => startingLineup * substitutes
  | 2 => startingLineup * substitutes * (startingLineup - 1) * (substitutes - 1)
  | _ => 0

/-- Calculates the total number of possible substitution scenarios. -/
def totalScenarios : ℕ :=
  (List.range (maxSubstitutions + 1)).map substitutionWays |>.sum

/-- The main theorem stating that the remainder of totalScenarios divided by 500 is 351. -/
theorem substitution_remainder :
  totalScenarios % 500 = 351 := by
  sorry

end NUMINAMATH_CALUDE_substitution_remainder_l1678_167882


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1678_167886

theorem solve_linear_equation (x : ℝ) : 5 * x - 3 = 17 ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1678_167886


namespace NUMINAMATH_CALUDE_power_function_decreasing_interval_l1678_167847

/-- A power function passing through (2, 4) is monotonically decreasing on (-∞, 0) -/
theorem power_function_decreasing_interval 
  (f : ℝ → ℝ) 
  (α : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^α) 
  (h2 : f 2 = 4) :
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_power_function_decreasing_interval_l1678_167847


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l1678_167896

/-- The function f(x) defined as a^(x-2) + 3 for some base a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 3

/-- Theorem stating that (2, 4) is a fixed point of f(x) for any base a > 0 -/
theorem fixed_point_of_f (a : ℝ) (h : a > 0) : f a 2 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l1678_167896


namespace NUMINAMATH_CALUDE_total_money_proof_l1678_167829

def sally_money : ℕ := 100
def jolly_money : ℕ := 50

theorem total_money_proof :
  (sally_money - 20 = 80) ∧ (jolly_money + 20 = 70) →
  sally_money + jolly_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_money_proof_l1678_167829


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l1678_167806

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (width : ℝ)
  (h1 : length = 60)
  (h2 : num_poles = 44)
  (h3 : pole_distance = 5)
  (h4 : 2 * (length + width) = pole_distance * num_poles) :
  width = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l1678_167806


namespace NUMINAMATH_CALUDE_pau_total_cost_l1678_167855

/-- Represents the cost of fried chicken orders for three people -/
def fried_chicken_cost 
  (kobe_pieces : ℕ) 
  (kobe_price : ℚ) 
  (pau_multiplier : ℕ) 
  (pau_extra : ℚ) 
  (pau_price : ℚ) 
  (shaq_multiplier : ℚ) 
  (discount : ℚ) : ℚ :=
  let pau_pieces := pau_multiplier * kobe_pieces + pau_extra
  let pau_initial := pau_pieces * pau_price
  pau_initial + pau_initial * (1 - discount)

/-- Theorem stating the total cost of Pau's fried chicken orders -/
theorem pau_total_cost : 
  fried_chicken_cost 5 (175/100) 2 (5/2) (3/2) (3/2) (15/100) = 346875/10000 := by
  sorry

end NUMINAMATH_CALUDE_pau_total_cost_l1678_167855


namespace NUMINAMATH_CALUDE_total_oranges_l1678_167839

theorem total_oranges (oranges_per_child : ℕ) (num_children : ℕ) 
  (h1 : oranges_per_child = 3) 
  (h2 : num_children = 4) : 
  oranges_per_child * num_children = 12 :=
by sorry

end NUMINAMATH_CALUDE_total_oranges_l1678_167839


namespace NUMINAMATH_CALUDE_friends_attended_reception_l1678_167845

/-- The number of friends attending a wedding reception --/
def friends_at_reception (total_guests : ℕ) (family_couples : ℕ) (coworkers : ℕ) (distant_relatives : ℕ) : ℕ :=
  total_guests - (2 * (2 * family_couples + coworkers + distant_relatives))

/-- Theorem: Given the conditions of the wedding reception, 180 friends attended --/
theorem friends_attended_reception :
  friends_at_reception 400 40 10 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_friends_attended_reception_l1678_167845


namespace NUMINAMATH_CALUDE_min_dot_product_in_triangle_l1678_167828

theorem min_dot_product_in_triangle (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angle_A := Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
                  (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))
  BC = 2 → angle_A = 2 * Real.pi / 3 → 
  (∀ A' B' C' : ℝ × ℝ, 
    let BC' := Real.sqrt ((B'.1 - C'.1)^2 + (B'.2 - C'.2)^2)
    let angle_A' := Real.arccos ((B'.1 - A'.1) * (C'.1 - A'.1) + (B'.2 - A'.2) * (C'.2 - A'.2)) / 
                    (Real.sqrt ((B'.1 - A'.1)^2 + (B'.2 - A'.2)^2) * Real.sqrt ((C'.1 - A'.1)^2 + (C'.2 - A'.2)^2))
    BC' = 2 → angle_A' = 2 * Real.pi / 3 → 
    ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) ≤ 
    ((B'.1 - A'.1) * (C'.1 - A'.1) + (B'.2 - A'.2) * (C'.2 - A'.2))) →
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) = -2/3 := by
sorry

end NUMINAMATH_CALUDE_min_dot_product_in_triangle_l1678_167828


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1678_167867

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 83 →
  a * d + b * c = 174 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_squares_l1678_167867


namespace NUMINAMATH_CALUDE_min_value_expression_l1678_167884

theorem min_value_expression (a b : ℝ) (h1 : a * b - 2 * a - b + 1 = 0) (h2 : a > 1) :
  ∀ x y : ℝ, x * y - 2 * x - y + 1 = 0 → x > 1 → (a + 3) * (b + 2) ≤ (x + 3) * (y + 2) ∧
  ∃ a₀ b₀ : ℝ, a₀ * b₀ - 2 * a₀ - b₀ + 1 = 0 ∧ a₀ > 1 ∧ (a₀ + 3) * (b₀ + 2) = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1678_167884


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1678_167863

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x : ℝ, |x| ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1678_167863


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1678_167879

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 12) 
  (h3 : S = a / (1 - r)) : 
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1678_167879


namespace NUMINAMATH_CALUDE_john_ate_three_slices_l1678_167810

/-- Represents the number of slices in a pizza -/
def total_slices : ℕ := 12

/-- Represents the number of slices left -/
def slices_left : ℕ := 3

/-- Represents the number of slices John ate -/
def john_slices : ℕ := 3

/-- Represents the number of slices Sam ate -/
def sam_slices : ℕ := 2 * john_slices

theorem john_ate_three_slices :
  john_slices = 3 ∧
  sam_slices = 2 * john_slices ∧
  total_slices = john_slices + sam_slices + slices_left :=
by sorry

end NUMINAMATH_CALUDE_john_ate_three_slices_l1678_167810


namespace NUMINAMATH_CALUDE_wig_cost_calculation_l1678_167897

-- Define the given conditions
def total_plays : ℕ := 3
def acts_per_play : ℕ := 5
def wigs_per_act : ℕ := 2
def dropped_play_sale : ℚ := 4
def total_spent : ℚ := 110

-- Define the theorem
theorem wig_cost_calculation :
  let wigs_per_play := acts_per_play * wigs_per_act
  let total_wigs := total_plays * wigs_per_play
  let remaining_wigs := total_wigs - wigs_per_play
  let cost_per_wig := total_spent / remaining_wigs
  cost_per_wig = 5.5 := by sorry

end NUMINAMATH_CALUDE_wig_cost_calculation_l1678_167897


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1678_167833

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1678_167833


namespace NUMINAMATH_CALUDE_inequality_proof_l1678_167880

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1678_167880


namespace NUMINAMATH_CALUDE_bulb_longevity_probability_l1678_167883

/-- Probability that a bulb from Factory X works for over 4000 hours -/
def prob_x : ℝ := 0.59

/-- Probability that a bulb from Factory Y works for over 4000 hours -/
def prob_y : ℝ := 0.65

/-- Proportion of bulbs supplied by Factory X -/
def supply_x : ℝ := 0.60

/-- Proportion of bulbs supplied by Factory Y -/
def supply_y : ℝ := 1 - supply_x

/-- Theorem stating the probability that a purchased bulb will work for longer than 4000 hours -/
theorem bulb_longevity_probability :
  prob_x * supply_x + prob_y * supply_y = 0.614 := by
  sorry

end NUMINAMATH_CALUDE_bulb_longevity_probability_l1678_167883


namespace NUMINAMATH_CALUDE_quadratic_equation_standard_form_l1678_167851

theorem quadratic_equation_standard_form :
  ∀ x : ℝ, (2*x - 1)^2 = (x + 1)*(3*x + 4) ↔ x^2 - 11*x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_standard_form_l1678_167851


namespace NUMINAMATH_CALUDE_subset_implies_x_equals_one_l1678_167856

def A : Set ℝ := {0, 1, 2}
def B (x : ℝ) : Set ℝ := {1, 2/x}

theorem subset_implies_x_equals_one (x : ℝ) (h : B x ⊆ A) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_x_equals_one_l1678_167856


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l1678_167837

theorem not_divisible_by_five (a : ℤ) (h : ¬(5 ∣ a)) : ¬(5 ∣ (3 * a^4 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l1678_167837


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1678_167865

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 7 = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1678_167865


namespace NUMINAMATH_CALUDE_area_at_stage_4_l1678_167850

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ := r.width * r.length

/-- Represents the growth of the rectangle at each stage --/
def grow (r : Rectangle) : Rectangle :=
  { width := r.width + 2, length := r.length + 3 }

/-- Calculates the rectangle at a given stage --/
def rectangleAtStage (n : ℕ) : Rectangle :=
  match n with
  | 0 => { width := 2, length := 3 }
  | n + 1 => grow (rectangleAtStage n)

theorem area_at_stage_4 : area (rectangleAtStage 4) = 150 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_4_l1678_167850


namespace NUMINAMATH_CALUDE_water_in_pool_l1678_167861

-- Define the parameters
def initial_bucket : ℝ := 1
def additional_buckets : ℝ := 8.8
def liters_per_bucket : ℝ := 10
def evaporation_rate : ℝ := 0.2
def splashing_rate : ℝ := 0.5
def time_taken : ℝ := 20

-- Define the theorem
theorem water_in_pool : 
  let total_buckets := initial_bucket + additional_buckets
  let total_water := total_buckets * liters_per_bucket
  let evaporation_loss := evaporation_rate * time_taken
  let splashing_loss := splashing_rate * time_taken
  let total_loss := evaporation_loss + splashing_loss
  let net_water := total_water - total_loss
  net_water = 84 := by
  sorry


end NUMINAMATH_CALUDE_water_in_pool_l1678_167861


namespace NUMINAMATH_CALUDE_train_crossing_time_l1678_167872

/-- Time taken for a faster train to cross a slower train moving in the same direction -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 300) 
  (h2 : length2 = 500)
  (h3 : speed1 = 72)
  (h4 : speed2 = 36)
  (h5 : speed1 > speed2) : 
  (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 80 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1678_167872


namespace NUMINAMATH_CALUDE_thinnest_gold_foil_scientific_notation_l1678_167864

theorem thinnest_gold_foil_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000092 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_thinnest_gold_foil_scientific_notation_l1678_167864


namespace NUMINAMATH_CALUDE_quadratic_equation_exponent_l1678_167843

/-- Given that 2x^m + (2-m)x - 5 = 0 is a quadratic equation in terms of x, prove that m = 2 -/
theorem quadratic_equation_exponent (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ 2*x^m + (2-m)*x - 5 = a*x^2 + b*x + c) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_exponent_l1678_167843


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l1678_167852

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- States that 1729 in base 10 is equal to 5020 in base 7 --/
theorem base_conversion_1729_to_base7 :
  1729 = base7ToBase10 5 0 2 0 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_to_base7_l1678_167852


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_for_steve_and_leroy_l1678_167871

/-- Represents the fishing scenario with Steve and LeRoy --/
structure FishingScenario where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  boat_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to reach shore without sinking --/
def minimum_bailing_rate (scenario : FishingScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum bailing rate for the given scenario --/
theorem minimum_bailing_rate_for_steve_and_leroy :
  let scenario : FishingScenario := {
    distance_to_shore := 2,
    water_intake_rate := 12,
    boat_capacity := 40,
    rowing_speed := 3
  }
  minimum_bailing_rate scenario = 11 := by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_for_steve_and_leroy_l1678_167871


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l1678_167866

theorem arithmetic_geometric_progression (a₁ a₂ a₃ a₄ d : ℝ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0)
  (h_d_nonzero : d ≠ 0)
  (h_arithmetic : a₂ = a₁ + d ∧ a₃ = a₁ + 2*d ∧ a₄ = a₁ + 3*d)
  (h_geometric : a₃^2 = a₁ * a₄) :
  d / a₁ = -1/4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l1678_167866


namespace NUMINAMATH_CALUDE_math_club_team_selection_l1678_167848

theorem math_club_team_selection (boys girls team_size : ℕ) 
  (h1 : boys = 7) 
  (h2 : girls = 9) 
  (h3 : team_size = 5) : 
  Nat.choose (boys + girls) team_size = 4368 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l1678_167848


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l1678_167824

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define the concept of increasing
def increasing (f : ℝ → ℝ) : Prop := 
  ∀ a b, a < b → f a < f b

-- Define the concept of linear correlation strength
def linear_correlation_strength (r : ℝ) : ℝ := sorry

-- Define the concept of functional relationship
def functional_relationship (x y : ℝ → ℝ) : Prop := sorry

theorem correlation_coefficient_properties 
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → increasing y) ∧ 
  (∀ s : ℝ, abs s < abs r → linear_correlation_strength s < linear_correlation_strength r) ∧
  ((r = 1 ∨ r = -1) → functional_relationship x y) := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l1678_167824


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1678_167860

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 3 * a 3 - 6 * a 3 + 8 = 0) →
  (a 15 * a 15 - 6 * a 15 + 8 = 0) →
  (a 1 * a 17) / a 9 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1678_167860


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l1678_167801

def total_republicans : ℕ := 12
def total_democrats : ℕ := 10
def subcommittee_republicans : ℕ := 5
def subcommittee_democrats : ℕ := 4

theorem subcommittee_formation_count :
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 166320 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l1678_167801


namespace NUMINAMATH_CALUDE_quotient_problem_l1678_167888

theorem quotient_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1620 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_quotient_problem_l1678_167888


namespace NUMINAMATH_CALUDE_irrational_shift_exists_rational_shift_not_exists_l1678_167869

variable {n : ℕ}
variable (a : Fin n → ℝ)

theorem irrational_shift_exists :
  ∃ (α : ℝ), ∀ (i : Fin n), ¬(∃ (p q : ℤ), a i + α = p / q ∧ q ≠ 0) :=
sorry

theorem rational_shift_not_exists :
  ¬(∃ (α : ℝ), ∀ (i : Fin n), ∃ (p q : ℤ), a i + α = p / q ∧ q ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_irrational_shift_exists_rational_shift_not_exists_l1678_167869


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l1678_167874

/-- The speed of the first cyclist in meters per second -/
def v : ℝ := 7

/-- The speed of the second cyclist in meters per second -/
def second_cyclist_speed : ℝ := 8

/-- The circumference of the circular track in meters -/
def track_circumference : ℝ := 300

/-- The time taken for the cyclists to meet at the starting point in seconds -/
def meeting_time : ℝ := 20

theorem cyclist_speed_proof :
  v * meeting_time + second_cyclist_speed * meeting_time = track_circumference :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l1678_167874


namespace NUMINAMATH_CALUDE_existence_of_alpha_for_tan_l1678_167826

open Real

theorem existence_of_alpha_for_tan : ∃ α : ℝ, 
  (∃ α₀ : ℝ, tan (π / 2 - α₀) = 1) ∧ 
  (¬∀ α₁ : ℝ, tan (π / 2 - α₁) = 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_alpha_for_tan_l1678_167826


namespace NUMINAMATH_CALUDE_closest_to_fraction_l1678_167870

def options : List ℝ := [0.2, 2, 20, 200, 2000]

theorem closest_to_fraction (x : ℝ) (h : x ∈ options) :
  |403 / 0.21 - 2000| ≤ |403 / 0.21 - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l1678_167870


namespace NUMINAMATH_CALUDE_keenan_essay_words_l1678_167800

/-- Represents Keenan's essay writing scenario -/
structure EssayWriting where
  initial_rate : ℕ  -- Words per hour for first two hours
  later_rate : ℕ    -- Words per hour after first two hours
  total_time : ℕ    -- Total time available in hours

/-- Calculates the total number of words Keenan can write -/
def total_words (e : EssayWriting) : ℕ :=
  (e.initial_rate * 2) + (e.later_rate * (e.total_time - 2))

/-- Theorem stating that Keenan can write 1200 words given the conditions -/
theorem keenan_essay_words :
  ∃ (e : EssayWriting), e.initial_rate = 400 ∧ e.later_rate = 200 ∧ e.total_time = 4 ∧ total_words e = 1200 := by
  sorry

end NUMINAMATH_CALUDE_keenan_essay_words_l1678_167800


namespace NUMINAMATH_CALUDE_female_students_count_l1678_167808

-- Define the total number of students
def total_students : ℕ := 8

-- Define the number of combinations
def total_combinations : ℕ := 30

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem female_students_count :
  ∃ f : ℕ, (f = 2 ∨ f = 3) ∧
  (∃ m : ℕ, m + f = total_students ∧
  combinations m 2 * combinations f 1 = total_combinations) :=
sorry

end NUMINAMATH_CALUDE_female_students_count_l1678_167808


namespace NUMINAMATH_CALUDE_pentagon_from_reflections_l1678_167842

/-- Given a set of reflection points, there exists a unique pentagon satisfying the reflection properties. -/
theorem pentagon_from_reflections (B : Fin 5 → ℝ × ℝ) :
  ∃! (A : Fin 5 → ℝ × ℝ), ∀ i : Fin 5, B i = 2 * A (i.succ) - A i :=
by sorry

end NUMINAMATH_CALUDE_pentagon_from_reflections_l1678_167842


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1678_167815

theorem parabola_vertex_y_coordinate (x y : ℝ) :
  y = -6 * x^2 + 24 * x - 7 →
  ∃ h k : ℝ, y = -6 * (x - h)^2 + k ∧ k = 17 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1678_167815


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l1678_167887

/-- The function f(x) = -x^2 - 4x + 16 -/
def f (x : ℝ) : ℝ := -x^2 - 4*x + 16

/-- The statement that f(x) attains its maximum value when x = -2 -/
theorem f_max_at_neg_two :
  ∀ x : ℝ, f x ≤ f (-2) :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l1678_167887


namespace NUMINAMATH_CALUDE_water_bottles_problem_l1678_167809

theorem water_bottles_problem (initial_bottles : ℕ) : 
  (initial_bottles : ℚ) * (2/3) * (1/2) = 8 → initial_bottles = 24 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_problem_l1678_167809


namespace NUMINAMATH_CALUDE_arithmetic_perfect_power_sequence_exists_l1678_167830

/-- An arithmetic sequence of perfect powers -/
def ArithmeticPerfectPowerSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (d : ℕ), ∀ i j : ℕ, i < n ∧ j < n →
    (∃ (base exponent : ℕ), exponent > 1 ∧ a i = base ^ exponent) ∧
    (a j - a i = d * (j - i))

theorem arithmetic_perfect_power_sequence_exists :
  ∃ (a : ℕ → ℕ), ArithmeticPerfectPowerSequence a 2003 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_perfect_power_sequence_exists_l1678_167830


namespace NUMINAMATH_CALUDE_vertical_shift_l1678_167835

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a constant k for the vertical shift
variable (k : ℝ)

-- Define a point (x, y) on the graph of y = f(x)
variable (x y : ℝ)

-- Theorem: If (x, y) is on the graph of y = f(x), then (x, y + k) is on the graph of y = f(x) + k
theorem vertical_shift (h : y = f x) : (y + k) = (f x + k) := by sorry

end NUMINAMATH_CALUDE_vertical_shift_l1678_167835


namespace NUMINAMATH_CALUDE_initial_distance_between_trains_l1678_167802

/-- Proves that the initial distance between two trains is 200 meters. -/
theorem initial_distance_between_trains (length1 length2 : ℝ) (speed1 speed2 : ℝ) (time : ℝ) :
  length1 = 90 →
  length2 = 100 →
  speed1 = 71 * 1000 / 3600 →
  speed2 = 89 * 1000 / 3600 →
  time = 4.499640028797696 →
  speed1 * time + speed2 * time = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_trains_l1678_167802


namespace NUMINAMATH_CALUDE_fixed_point_on_linear_function_l1678_167859

/-- Given a linear function y = kx + b where 3k - b = 2, 
    prove that the point (-3, -2) lies on the graph of the function. -/
theorem fixed_point_on_linear_function (k b : ℝ) 
  (h : 3 * k - b = 2) : 
  k * (-3) + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_linear_function_l1678_167859


namespace NUMINAMATH_CALUDE_hua_method_is_golden_ratio_l1678_167812

/-- Represents the possible methods used in optimal selection -/
inductive OptimalSelectionMethod
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- The optimal selection method popularized by Hua Luogeng -/
def huaMethod : OptimalSelectionMethod := OptimalSelectionMethod.GoldenRatio

/-- Theorem stating that Hua Luogeng's optimal selection method uses the Golden ratio -/
theorem hua_method_is_golden_ratio :
  huaMethod = OptimalSelectionMethod.GoldenRatio :=
by sorry

end NUMINAMATH_CALUDE_hua_method_is_golden_ratio_l1678_167812


namespace NUMINAMATH_CALUDE_zoo_lion_cubs_l1678_167873

theorem zoo_lion_cubs (initial_count final_count : ℕ) 
  (gorillas_sent : ℕ) (hippo_adopted : ℕ) (rhinos_taken : ℕ) : 
  initial_count = 68 →
  gorillas_sent = 6 →
  hippo_adopted = 1 →
  rhinos_taken = 3 →
  final_count = 90 →
  ∃ (lion_cubs : ℕ), 
    final_count = initial_count - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs + 2 * lion_cubs ∧
    lion_cubs = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_lion_cubs_l1678_167873


namespace NUMINAMATH_CALUDE_min_months_for_committee_repetition_l1678_167804

theorem min_months_for_committee_repetition 
  (total_members : Nat) 
  (women : Nat) 
  (men : Nat) 
  (committee_size : Nat) 
  (h1 : total_members = 13)
  (h2 : women = 6)
  (h3 : men = 7)
  (h4 : committee_size = 5)
  (h5 : women + men = total_members) :
  let total_committees := Nat.choose total_members committee_size
  let women_only_committees := Nat.choose women committee_size
  let men_only_committees := Nat.choose men committee_size
  let valid_committees := total_committees - women_only_committees - men_only_committees
  valid_committees + 1 = 1261 := by
  sorry

end NUMINAMATH_CALUDE_min_months_for_committee_repetition_l1678_167804


namespace NUMINAMATH_CALUDE_gcf_of_45_and_75_l1678_167885

theorem gcf_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_45_and_75_l1678_167885


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_for_nonnegative_f_l1678_167831

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem min_value_when_a_is_one :
  ∃ (x : ℝ), ∀ (y : ℝ), f 1 x ≤ f 1 y ∧ f 1 x = 0 :=
sorry

theorem range_of_a_for_nonnegative_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ 0) ↔ a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_for_nonnegative_f_l1678_167831


namespace NUMINAMATH_CALUDE_gold_alloy_composition_l1678_167868

theorem gold_alloy_composition
  (initial_weight : ℝ)
  (initial_gold_percentage : ℝ)
  (target_gold_percentage : ℝ)
  (added_gold : ℝ)
  (h1 : initial_weight = 48)
  (h2 : initial_gold_percentage = 0.25)
  (h3 : target_gold_percentage = 0.40)
  (h4 : added_gold = 12) :
  let initial_gold := initial_weight * initial_gold_percentage
  let final_weight := initial_weight + added_gold
  let final_gold := initial_gold + added_gold
  (final_gold / final_weight) = target_gold_percentage :=
by
  sorry

#check gold_alloy_composition

end NUMINAMATH_CALUDE_gold_alloy_composition_l1678_167868


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l1678_167811

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) 
  (fish_per_nineteen : ℕ) (h1 : total_fishermen = 20) 
  (h2 : total_fish = 10000) (h3 : fish_per_nineteen = 400) : 
  total_fish - (total_fishermen - 1) * fish_per_nineteen = 2400 :=
by
  sorry

#check twentieth_fisherman_catch

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l1678_167811


namespace NUMINAMATH_CALUDE_meals_left_to_distribute_l1678_167818

theorem meals_left_to_distribute (initial_meals additional_meals distributed_meals : ℕ) :
  initial_meals + additional_meals - distributed_meals =
  (initial_meals + additional_meals) - distributed_meals :=
by sorry

end NUMINAMATH_CALUDE_meals_left_to_distribute_l1678_167818


namespace NUMINAMATH_CALUDE_cubic_sum_value_l1678_167858

theorem cubic_sum_value (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 10) :
  Complex.abs (w^3 + z^3) = 26 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_value_l1678_167858


namespace NUMINAMATH_CALUDE_solution_value_l1678_167876

theorem solution_value (a : ℝ) (h : 3 * a^2 + 2 * a - 1 = 0) : 
  3 * a^2 + 2 * a - 2019 = -2018 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l1678_167876


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l1678_167836

/-- An arithmetic sequence of natural numbers -/
def arithmeticSequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

/-- Predicate for perfect squares -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem infinitely_many_perfect_squares
  (a d : ℕ) -- First term and common difference of the sequence
  (h : ∃ n₀ : ℕ, isPerfectSquare (arithmeticSequence a d n₀)) :
  ∀ m : ℕ, ∃ n > m, isPerfectSquare (arithmeticSequence a d n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l1678_167836


namespace NUMINAMATH_CALUDE_sin_increasing_omega_range_l1678_167817

theorem sin_increasing_omega_range (ω : ℝ) (f : ℝ → ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Icc 0 (π / 3), f x = Real.sin (ω * x)) →
  StrictMonoOn f (Set.Icc 0 (π / 3)) →
  ω ∈ Set.Ioo 0 (3 / 2) :=
sorry

end NUMINAMATH_CALUDE_sin_increasing_omega_range_l1678_167817


namespace NUMINAMATH_CALUDE_smallest_input_129_l1678_167877

def f (n : ℕ+) : ℕ := 9 * n.val + 120

theorem smallest_input_129 :
  ∀ m : ℕ+, f m ≥ f 129 → m ≥ 129 :=
sorry

end NUMINAMATH_CALUDE_smallest_input_129_l1678_167877


namespace NUMINAMATH_CALUDE_milkman_profit_is_90_l1678_167838

/-- Calculates the profit of a milkman given the following conditions:
  * The milkman has 30 liters of milk
  * 5 liters of water is mixed with 20 liters of pure milk
  * Water is freely available
  * Cost of pure milk is Rs. 18 per liter
  * Milkman sells all the mixture at cost price
-/
def milkman_profit (total_milk : ℕ) (mixed_milk : ℕ) (water : ℕ) (cost_per_liter : ℕ) : ℕ :=
  let mixture_volume := mixed_milk + water
  let mixture_revenue := mixture_volume * cost_per_liter
  let mixed_milk_cost := mixed_milk * cost_per_liter
  mixture_revenue - mixed_milk_cost

/-- The profit of the milkman is Rs. 90 given the specified conditions. -/
theorem milkman_profit_is_90 :
  milkman_profit 30 20 5 18 = 90 := by
  sorry

end NUMINAMATH_CALUDE_milkman_profit_is_90_l1678_167838


namespace NUMINAMATH_CALUDE_students_present_l1678_167878

theorem students_present (total : ℕ) (absent_percent : ℚ) : 
  total = 50 → absent_percent = 14/100 → 
  (total : ℚ) * (1 - absent_percent) = 43 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l1678_167878


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1678_167805

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_1 : ℝ × ℝ
  point_2 : ℝ × ℝ

/-- The theorem statement -/
theorem quadratic_function_theorem (f : QuadraticFunction) 
  (h1 : f.min_value = -3)
  (h2 : f.min_x = -2)
  (h3 : f.point_1 = (1, 10))
  (h4 : f.point_2.1 = 3) :
  f.point_2.2 = 298 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1678_167805


namespace NUMINAMATH_CALUDE_number_of_late_classmates_l1678_167895

/-- The number of late classmates given Charlize's lateness, classmates' additional lateness, and total late time -/
def late_classmates (charlize_lateness : ℕ) (classmate_additional_lateness : ℕ) (total_late_time : ℕ) : ℕ :=
  (total_late_time - charlize_lateness) / (charlize_lateness + classmate_additional_lateness)

/-- Theorem stating that the number of late classmates is 4 given the specific conditions -/
theorem number_of_late_classmates :
  late_classmates 20 10 140 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_late_classmates_l1678_167895


namespace NUMINAMATH_CALUDE_max_games_purchasable_l1678_167823

def initial_amount : ℕ := 35
def spent_amount : ℕ := 7
def game_cost : ℕ := 4

theorem max_games_purchasable :
  (initial_amount - spent_amount) / game_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_games_purchasable_l1678_167823


namespace NUMINAMATH_CALUDE_gcd_72_120_168_l1678_167816

theorem gcd_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by sorry

end NUMINAMATH_CALUDE_gcd_72_120_168_l1678_167816


namespace NUMINAMATH_CALUDE_cody_grandmother_age_l1678_167822

/-- Given that Cody is 14 years old and his grandmother is 6 times as old as he is,
    prove that Cody's grandmother is 84 years old. -/
theorem cody_grandmother_age (cody_age : ℕ) (grandmother_age_ratio : ℕ) 
  (h1 : cody_age = 14)
  (h2 : grandmother_age_ratio = 6) :
  cody_age * grandmother_age_ratio = 84 := by
  sorry

end NUMINAMATH_CALUDE_cody_grandmother_age_l1678_167822


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_through_point_parallel_to_line_proof_l1678_167875

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  given_point.x = -1 ∧
  given_point.y = 3 ∧
  result_line.a = 1 ∧
  result_line.b = -2 ∧
  result_line.c = 7 →
  given_point.liesOn result_line ∧
  result_line.isParallelTo given_line

-- The proof of the theorem
theorem line_through_point_parallel_to_line_proof 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : 
  line_through_point_parallel_to_line given_line given_point result_line :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_through_point_parallel_to_line_proof_l1678_167875
