import Mathlib

namespace NUMINAMATH_CALUDE_people_per_car_l1789_178908

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 9) :
  total_people / num_cars = 7 := by
sorry

end NUMINAMATH_CALUDE_people_per_car_l1789_178908


namespace NUMINAMATH_CALUDE_closed_path_vector_sum_l1789_178940

/-- The sum of vectors forming a closed path in a plane is equal to the zero vector. -/
theorem closed_path_vector_sum (A B C D E F : ℝ × ℝ) : 
  (B.1 - A.1, B.2 - A.2) + (C.1 - B.1, C.2 - B.2) + (D.1 - C.1, D.2 - C.2) + 
  (E.1 - D.1, E.2 - D.2) + (F.1 - E.1, F.2 - E.2) + (A.1 - F.1, A.2 - F.2) = (0, 0) := by
sorry

end NUMINAMATH_CALUDE_closed_path_vector_sum_l1789_178940


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1789_178972

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 1 ∧ p.k = 1 →
  let shifted := shift (shift p 2 0) 0 2
  shifted.a = 3 ∧ shifted.h = 3 ∧ shifted.k = 3 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1789_178972


namespace NUMINAMATH_CALUDE_sphere_volume_sphere_surface_area_sphere_surface_eq_cylinder_lateral_l1789_178950

/-- A structure representing a sphere contained in a cylinder -/
structure SphereInCylinder where
  r : ℝ  -- radius of the sphere and base of the cylinder
  h : ℝ  -- height of the cylinder
  sphere_diameter_eq_cylinder : h = 2 * r  -- diameter of sphere equals height of cylinder

/-- The volume of the sphere is (4/3)πr³ -/
theorem sphere_volume (s : SphereInCylinder) : 
  (4 / 3) * Real.pi * s.r ^ 3 = (2 / 3) * Real.pi * s.r ^ 2 * s.h := by sorry

/-- The surface area of the sphere is 4πr² -/
theorem sphere_surface_area (s : SphereInCylinder) :
  4 * Real.pi * s.r ^ 2 = (2 / 3) * (2 * Real.pi * s.r * s.h + 2 * Real.pi * s.r ^ 2) := by sorry

/-- The surface area of the sphere equals the lateral surface area of the cylinder -/
theorem sphere_surface_eq_cylinder_lateral (s : SphereInCylinder) :
  4 * Real.pi * s.r ^ 2 = 2 * Real.pi * s.r * s.h := by sorry

end NUMINAMATH_CALUDE_sphere_volume_sphere_surface_area_sphere_surface_eq_cylinder_lateral_l1789_178950


namespace NUMINAMATH_CALUDE_distance_between_places_l1789_178954

/-- The distance between places A and B in kilometers -/
def distance : ℝ := 150

/-- The speed of bicycling in kilometers per hour -/
def bicycle_speed : ℝ := 15

/-- The speed of walking in kilometers per hour -/
def walking_speed : ℝ := 5

/-- The time difference between return trip and going trip in hours -/
def time_difference : ℝ := 2

theorem distance_between_places : 
  ∃ (return_time : ℝ),
    (distance / 2 / bicycle_speed + distance / 2 / walking_speed = return_time - time_difference) ∧
    (distance = return_time / 3 * bicycle_speed + 2 * return_time / 3 * walking_speed) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_places_l1789_178954


namespace NUMINAMATH_CALUDE_water_depth_ratio_l1789_178915

theorem water_depth_ratio (dean_height : ℝ) (water_depth_difference : ℝ) :
  dean_height = 9 →
  water_depth_difference = 81 →
  (dean_height + water_depth_difference) / dean_height = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_water_depth_ratio_l1789_178915


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1789_178900

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {y | ∃ x ∈ U, y = |x|}

theorem complement_of_A_in_U : 
  {x ∈ U | x ∉ A} = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1789_178900


namespace NUMINAMATH_CALUDE_seating_arrangement_l1789_178963

theorem seating_arrangement (total_people : ℕ) (total_rows : ℕ) 
  (h1 : total_people = 97) 
  (h2 : total_rows = 13) : 
  ∃ (rows_with_8 : ℕ), 
    rows_with_8 * 8 + (total_rows - rows_with_8) * 7 = total_people ∧ 
    rows_with_8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l1789_178963


namespace NUMINAMATH_CALUDE_archer_expected_hits_l1789_178921

/-- The expected value of a binomial distribution with n trials and probability p -/
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The number of shots taken by the archer -/
def num_shots : ℕ := 10

/-- The probability of hitting the bullseye -/
def hit_probability : ℝ := 0.9

/-- Theorem: The expected number of bullseye hits for the archer -/
theorem archer_expected_hits : 
  binomial_expectation num_shots hit_probability = 9 := by
  sorry

end NUMINAMATH_CALUDE_archer_expected_hits_l1789_178921


namespace NUMINAMATH_CALUDE_payroll_tax_threshold_l1789_178987

/-- The payroll tax problem -/
theorem payroll_tax_threshold (tax_rate : ℝ) (tax_paid : ℝ) (total_payroll : ℝ) (T : ℝ) : 
  tax_rate = 0.002 →
  tax_paid = 200 →
  total_payroll = 300000 →
  tax_paid = (total_payroll - T) * tax_rate →
  T = 200000 := by
  sorry


end NUMINAMATH_CALUDE_payroll_tax_threshold_l1789_178987


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1789_178904

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4 * x + 1 = 0 ∧ 
   ∀ y : ℝ, a * y^2 - 4 * y + 1 = 0 → y = x) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1789_178904


namespace NUMINAMATH_CALUDE_initial_average_score_l1789_178927

theorem initial_average_score 
  (total_students : Nat) 
  (remaining_students : Nat)
  (dropped_score : Real)
  (new_average : Real) :
  total_students = 16 →
  remaining_students = 15 →
  dropped_score = 24 →
  new_average = 64 →
  (total_students : Real) * (remaining_students * new_average + dropped_score) / total_students = 61.5 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_score_l1789_178927


namespace NUMINAMATH_CALUDE_rectangle_area_unchanged_l1789_178936

/-- Given a rectangle with area 432 square centimeters, prove that decreasing the length by 20%
    and increasing the width by 25% results in the same area. -/
theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) :
  (0.8 * l) * (1.25 * w) = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_unchanged_l1789_178936


namespace NUMINAMATH_CALUDE_unique_sum_value_l1789_178905

theorem unique_sum_value (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_value_l1789_178905


namespace NUMINAMATH_CALUDE_parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l1789_178953

-- Define the vectors
def OA : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def OB : Fin 2 → ℝ := ![3, -1]
def OC (m : ℝ) : Fin 2 → ℝ := ![m, 1]

-- Define vector operations
def vector_sub (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i - w i
def parallel (v w : Fin 2 → ℝ) : Prop := ∃ k : ℝ, ∀ i, v i = k * w i
def perpendicular (v w : Fin 2 → ℝ) : Prop := v 0 * w 0 + v 1 * w 1 = 0

-- Define the theorems
theorem parallel_implies_m_eq_neg_one (m : ℝ) :
  parallel (vector_sub OB OA) (OC m) → m = -1 := by sorry

theorem perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two (m : ℝ) :
  perpendicular (vector_sub (OC m) OA) (vector_sub (OC m) OB) →
  (m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l1789_178953


namespace NUMINAMATH_CALUDE_sienas_initial_bookmarks_l1789_178934

/-- Calculates the number of pages Siena had before March, given her daily bookmarking rate and final page count. -/
theorem sienas_initial_bookmarks (daily_bookmarks : ℕ) (march_days : ℕ) (final_count : ℕ) : 
  daily_bookmarks = 30 → 
  march_days = 31 → 
  final_count = 1330 → 
  final_count - (daily_bookmarks * march_days) = 400 :=
by
  sorry

#check sienas_initial_bookmarks

end NUMINAMATH_CALUDE_sienas_initial_bookmarks_l1789_178934


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l1789_178967

theorem unique_prime_with_remainder : ∃! n : ℕ, 
  40 < n ∧ n < 50 ∧ 
  Nat.Prime n ∧ 
  n % 9 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l1789_178967


namespace NUMINAMATH_CALUDE_alberts_age_to_marys_age_ratio_l1789_178902

-- Define the ages as natural numbers
def Betty : ℕ := 7
def Albert : ℕ := 4 * Betty
def Mary : ℕ := Albert - 14

-- Define the ratio of Albert's age to Mary's age
def age_ratio : ℚ := Albert / Mary

-- Theorem statement
theorem alberts_age_to_marys_age_ratio :
  age_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_alberts_age_to_marys_age_ratio_l1789_178902


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1789_178926

theorem quadratic_inequality_solution_set :
  {x : ℝ | 4*x^2 - 12*x + 5 < 0} = Set.Ioo (1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1789_178926


namespace NUMINAMATH_CALUDE_lcm_problem_l1789_178994

theorem lcm_problem (n : ℕ+) (h1 : Nat.lcm 40 n = 200) (h2 : Nat.lcm n 45 = 180) : n = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1789_178994


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l1789_178993

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l1789_178993


namespace NUMINAMATH_CALUDE_problem_statement_l1789_178995

theorem problem_statement (x y n : ℝ) : 
  x = 3 → y = 0 → n = x - y^(x+y) → n = 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1789_178995


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l1789_178999

theorem square_perimeter_problem (M N : Real) (h1 : M = 100) (h2 : N = 4 * M) :
  4 * Real.sqrt N = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l1789_178999


namespace NUMINAMATH_CALUDE_ellipse_sum_range_l1789_178983

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 144 + y^2 / 25 = 1

-- Theorem statement
theorem ellipse_sum_range :
  ∀ x y : ℝ, is_on_ellipse x y →
  ∃ (a b : ℝ), a = -13 ∧ b = 13 ∧
  a ≤ x + y ∧ x + y ≤ b ∧
  (∃ (x₁ y₁ : ℝ), is_on_ellipse x₁ y₁ ∧ x₁ + y₁ = a) ∧
  (∃ (x₂ y₂ : ℝ), is_on_ellipse x₂ y₂ ∧ x₂ + y₂ = b) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_range_l1789_178983


namespace NUMINAMATH_CALUDE_proposition_range_l1789_178945

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x > a
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem statement
theorem proposition_range (a : ℝ) : 
  (¬(p a) ∨ q a) = false → -2 < a ∧ a < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_range_l1789_178945


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l1789_178947

/-- The line x₀x + y₀y - a² = 0 is separate from the circle x² + y² = a² (a > 0),
    given that point M(x₀, y₀) is inside the circle and different from its center. -/
theorem line_separate_from_circle
  (a : ℝ) (x₀ y₀ : ℝ) 
  (h_a_pos : a > 0)
  (h_inside : x₀^2 + y₀^2 < a^2)
  (h_not_center : x₀ ≠ 0 ∨ y₀ ≠ 0) :
  let d := a^2 / Real.sqrt (x₀^2 + y₀^2)
  d > a :=
by sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l1789_178947


namespace NUMINAMATH_CALUDE_inequality_addition_l1789_178981

theorem inequality_addition (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a + b > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l1789_178981


namespace NUMINAMATH_CALUDE_harrys_journey_l1789_178903

theorem harrys_journey (total_time bus_time_so_far : ℕ) 
  (h1 : total_time = 60)
  (h2 : bus_time_so_far = 15)
  (h3 : ∃ (total_bus_time walking_time : ℕ), 
    total_bus_time + walking_time = total_time ∧
    walking_time = total_bus_time / 2) :
  ∃ (remaining_bus_time : ℕ), 
    remaining_bus_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_harrys_journey_l1789_178903


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l1789_178977

/-- An arithmetic sequence with the given properties has a common difference of -1/5 -/
theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a : ℕ → ℚ) (d : ℚ),
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 1 = 1 →
  (a 2) * (a 5) = (a 4)^2 →
  d = -1/5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l1789_178977


namespace NUMINAMATH_CALUDE_employee_recorder_price_l1789_178957

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the markup percentage
def markup_percentage : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount_percentage : ℝ := 0.05

-- Define the retail price calculation
def retail_price : ℝ := wholesale_cost * (1 + markup_percentage)

-- Define the employee price calculation
def employee_price : ℝ := retail_price * (1 - employee_discount_percentage)

-- Theorem statement
theorem employee_recorder_price : employee_price = 228 := by
  sorry

end NUMINAMATH_CALUDE_employee_recorder_price_l1789_178957


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1789_178946

def polynomial (x a₂ a₁ : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x + 24

def possible_roots : Set ℤ := {-24, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 24}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | polynomial x a₂ a₁ = 0} ⊆ possible_roots :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1789_178946


namespace NUMINAMATH_CALUDE_remainder_2567139_div_6_l1789_178964

theorem remainder_2567139_div_6 : 2567139 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2567139_div_6_l1789_178964


namespace NUMINAMATH_CALUDE_bert_profit_l1789_178955

def selling_price : ℝ := 90
def markup : ℝ := 10
def tax_rate : ℝ := 0.1

theorem bert_profit : 
  let cost_price := selling_price - markup
  let tax := selling_price * tax_rate
  selling_price - cost_price - tax = 1 := by
  sorry

end NUMINAMATH_CALUDE_bert_profit_l1789_178955


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_18_l1789_178930

theorem largest_of_three_consecutive_integers_sum_18 (a b c : ℤ) : 
  (b = a + 1) →  -- b is the next consecutive integer after a
  (c = b + 1) →  -- c is the next consecutive integer after b
  (a + b + c = 18) →  -- sum of the three integers is 18
  (c = 7) -- c (the largest) is 7
:= by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_integers_sum_18_l1789_178930


namespace NUMINAMATH_CALUDE_day_of_week_N_minus_1_l1789_178971

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  dayNumber : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek :=
  sorry

theorem day_of_week_N_minus_1 
  (N : Int)
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Friday)
  (h2 : dayOfWeek ⟨N+1, 150⟩ = DayOfWeek.Friday) :
  dayOfWeek ⟨N-1, 250⟩ = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_N_minus_1_l1789_178971


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l1789_178935

theorem largest_divisor_of_m (m : ℕ+) (h : (m.val ^ 3) % 847 = 0) : 
  ∃ (k : ℕ+), k.val = 77 ∧ k.val ∣ m.val ∧ ∀ (d : ℕ+), d.val ∣ m.val → d.val ≤ k.val :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l1789_178935


namespace NUMINAMATH_CALUDE_connected_paper_area_l1789_178910

/-- The area of connected square papers -/
theorem connected_paper_area 
  (num_papers : ℕ) 
  (side_length : ℝ) 
  (overlap : ℝ) 
  (h_num : num_papers = 6)
  (h_side : side_length = 30)
  (h_overlap : overlap = 7) : 
  (side_length + (num_papers - 1) * (side_length - overlap)) * side_length = 4350 :=
sorry

end NUMINAMATH_CALUDE_connected_paper_area_l1789_178910


namespace NUMINAMATH_CALUDE_michael_quiz_score_l1789_178958

theorem michael_quiz_score (existing_scores : List ℕ) (target_mean : ℕ) (required_score : ℕ) : 
  existing_scores = [84, 78, 95, 88, 91] →
  target_mean = 90 →
  required_score = 104 →
  (existing_scores.sum + required_score) / (existing_scores.length + 1) = target_mean :=
by sorry

end NUMINAMATH_CALUDE_michael_quiz_score_l1789_178958


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1789_178965

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 3)
  (h2 : x⁻¹ - y⁻¹ = -7) : 
  x + y = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1789_178965


namespace NUMINAMATH_CALUDE_product_equivalence_l1789_178996

theorem product_equivalence (h : 213 * 16 = 3408) : 1.6 * 2.13 = 3.408 := by
  sorry

end NUMINAMATH_CALUDE_product_equivalence_l1789_178996


namespace NUMINAMATH_CALUDE_basketball_teams_l1789_178943

theorem basketball_teams (total : ℕ) (bad : ℕ) (rich : ℕ) (both : ℕ) : 
  total = 60 → 
  bad = (3 * total) / 5 →
  rich = (2 * total) / 3 →
  both ≤ bad :=
by sorry

end NUMINAMATH_CALUDE_basketball_teams_l1789_178943


namespace NUMINAMATH_CALUDE_bethany_portraits_l1789_178982

/-- The number of portraits Bethany saw at the museum -/
def num_portraits : ℕ := 16

/-- The number of still lifes Bethany saw at the museum -/
def num_still_lifes : ℕ := 4 * num_portraits

/-- The total number of paintings Bethany saw at the museum -/
def total_paintings : ℕ := 80

theorem bethany_portraits :
  num_portraits + num_still_lifes = total_paintings ∧
  num_still_lifes = 4 * num_portraits →
  num_portraits = 16 := by sorry

end NUMINAMATH_CALUDE_bethany_portraits_l1789_178982


namespace NUMINAMATH_CALUDE_house_application_proof_l1789_178976

/-- The number of houses available -/
def num_houses : ℕ := 3

/-- The number of persons applying for houses -/
def num_persons : ℕ := 3

/-- The probability that all persons apply for the same house -/
def prob_same_house : ℚ := 1 / 9

/-- The number of houses each person applies for -/
def houses_per_person : ℕ := 1

theorem house_application_proof :
  (prob_same_house = (houses_per_person : ℚ)^2 / num_houses^2) →
  houses_per_person = 1 :=
by sorry

end NUMINAMATH_CALUDE_house_application_proof_l1789_178976


namespace NUMINAMATH_CALUDE_d_is_zero_l1789_178918

def d (n m : ℕ) : ℚ :=
  if m = 0 ∨ m = n then 0
  else if 0 < m ∧ m < n then
    (m * d (n-1) m + (2*n - m) * d (n-1) (m-1)) / m
  else 0

theorem d_is_zero (n m : ℕ) (h : m ≤ n) : d n m = 0 := by
  sorry

end NUMINAMATH_CALUDE_d_is_zero_l1789_178918


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1789_178961

theorem polynomial_factorization (m n a b : ℝ) : 
  (|m - 4| + (n^2 - 8*n + 16) = 0) → 
  (a^2 + 4*b^2 - m*a*b - n = (a - 2*b + 2) * (a - 2*b - 2)) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1789_178961


namespace NUMINAMATH_CALUDE_sum_m_n_equals_negative_two_l1789_178914

/-- A polynomial in x and y -/
def polynomial (m n : ℝ) (x y : ℝ) : ℝ := m * x^2 - n * x * y - 2 * x * y + y - 3

/-- The condition that the polynomial has no quadratic terms when simplified -/
def no_quadratic_terms (m n : ℝ) : Prop :=
  ∀ x y, polynomial m n x y = (-n - 2) * x * y + y - 3

theorem sum_m_n_equals_negative_two (m n : ℝ) (h : no_quadratic_terms m n) : m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_negative_two_l1789_178914


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1789_178941

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

structure Point := (coords : ℝ × ℝ)

def on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p.coords
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def diametrically_opposite (p1 p2 : Point) (c : Circle) : Prop :=
  let (x1, y1) := p1.coords
  let (x2, y2) := p2.coords
  let (cx, cy) := c.center
  (x1 - cx)^2 + (y1 - cy)^2 = c.radius^2 ∧
  (x2 - cx)^2 + (y2 - cy)^2 = c.radius^2 ∧
  (x1 - x2)^2 + (y1 - y2)^2 = 4 * c.radius^2

def angle (p1 p2 p3 : Point) : ℝ := sorry

theorem circle_intersection_theorem (c : Circle) (A B C D M N : Point) :
  on_circle A c ∧ on_circle B c ∧ on_circle C c ∧ on_circle D c →
  (∃ t : ℝ, A.coords = B.coords + t • (C.coords - D.coords)) →
  (∃ s : ℝ, A.coords = D.coords + s • (B.coords - C.coords)) →
  angle B M C = angle C N D ↔
  diametrically_opposite A C c ∨ diametrically_opposite B D c :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1789_178941


namespace NUMINAMATH_CALUDE_constant_sequence_l1789_178968

def is_prime (n : ℤ) : Prop := Nat.Prime n.natAbs

theorem constant_sequence
  (a : ℕ → ℤ)  -- Sequence of integers
  (d : ℤ)      -- Integer d
  (h1 : ∀ n, is_prime (a n))  -- |a_n| is prime for all n
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d)  -- Recurrence relation
  : ∀ n, a n = a 0  -- Conclusion: sequence is constant
  := by sorry

end NUMINAMATH_CALUDE_constant_sequence_l1789_178968


namespace NUMINAMATH_CALUDE_g_of_3_l1789_178980

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_of_3 : g 3 = 147 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l1789_178980


namespace NUMINAMATH_CALUDE_apple_ratio_l1789_178998

theorem apple_ratio (total_weight : ℝ) (apples_per_pie : ℝ) (num_pies : ℝ) 
  (h1 : total_weight = 120)
  (h2 : apples_per_pie = 4)
  (h3 : num_pies = 15) :
  (total_weight - apples_per_pie * num_pies) / total_weight = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_l1789_178998


namespace NUMINAMATH_CALUDE_no_less_equal_two_mo_l1789_178907

theorem no_less_equal_two_mo (N O M : ℝ) (h : N * O ≤ 2 * M * O) : N * O ≤ 2 * M * O := by
  sorry

end NUMINAMATH_CALUDE_no_less_equal_two_mo_l1789_178907


namespace NUMINAMATH_CALUDE_wise_men_strategy_l1789_178979

/-- Represents the color of a hat -/
inductive HatColor
| White
| Black

/-- Represents a wise man with a hat -/
structure WiseMan where
  hat : HatColor

/-- Represents the line of wise men -/
def WiseMenLine := List WiseMan

/-- A strategy is a function that takes the visible hats and returns a guess -/
def Strategy := (visible : WiseMenLine) → HatColor

/-- Counts the number of correct guesses given a line of wise men and a strategy -/
def countCorrectGuesses (line : WiseMenLine) (strategy : Strategy) : Nat :=
  sorry

/-- The main theorem: there exists a strategy where at least n-1 wise men guess correctly -/
theorem wise_men_strategy (n : Nat) :
  ∃ (strategy : Strategy), ∀ (line : WiseMenLine),
    line.length = n →
    countCorrectGuesses line strategy ≥ n - 1 :=
  sorry

end NUMINAMATH_CALUDE_wise_men_strategy_l1789_178979


namespace NUMINAMATH_CALUDE_grain_oil_production_growth_l1789_178931

theorem grain_oil_production_growth (x : ℝ) : 
  (450000 * (1 + x)^2 = 500000) ↔ 
  (∃ (y : ℝ), 450000 * (1 + x) = y ∧ y * (1 + x) = 500000) :=
sorry

end NUMINAMATH_CALUDE_grain_oil_production_growth_l1789_178931


namespace NUMINAMATH_CALUDE_twenty_seven_power_divided_by_nine_l1789_178944

theorem twenty_seven_power_divided_by_nine (m : ℕ) :
  m = 27^1001 → m / 9 = 3^3001 := by
  sorry

end NUMINAMATH_CALUDE_twenty_seven_power_divided_by_nine_l1789_178944


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l1789_178966

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The diagonal is perpendicular to the lateral side -/
  diagonalPerpendicular : Bool

/-- Properties of the isosceles trapezoid -/
def trapezoidProperties : IsoscelesTrapezoid :=
  { smallerBase := 3
  , height := 2
  , diagonalPerpendicular := true }

/-- The theorem stating the properties of the isosceles trapezoid -/
theorem isosceles_trapezoid_theorem (t : IsoscelesTrapezoid) 
  (h1 : t = trapezoidProperties) :
  ∃ (largerBase acuteAngle : ℝ),
    largerBase = 5 ∧ 
    acuteAngle = Real.arctan 2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l1789_178966


namespace NUMINAMATH_CALUDE_rachels_homework_l1789_178919

/-- Rachel's homework problem -/
theorem rachels_homework (reading_pages : ℕ) (math_pages : ℕ) : 
  reading_pages = 4 → reading_pages = math_pages + 1 → math_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_rachels_homework_l1789_178919


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1789_178912

theorem units_digit_of_expression (k : ℕ) : k = 2025^2 + 3^2025 → (k^2 + 3^k) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1789_178912


namespace NUMINAMATH_CALUDE_count_squares_on_marked_grid_l1789_178924

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A square grid with marked points -/
structure MarkedGrid where
  size : ℕ
  points : List GridPoint

/-- A square formed by four points -/
structure Square where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- Check if four points form a valid square -/
def isValidSquare (s : Square) : Bool :=
  sorry

/-- Count the number of valid squares that can be formed from a list of points -/
def countValidSquares (points : List GridPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem count_squares_on_marked_grid :
  ∀ (g : MarkedGrid),
    g.size = 4 ∧ 
    g.points.length = 12 ∧ 
    (∀ p ∈ g.points, p.x < 4 ∧ p.y < 4) ∧
    (∀ x y, x = 0 ∨ x = 3 ∨ y = 0 ∨ y = 3 → ¬∃ p ∈ g.points, p.x = x ∧ p.y = y) →
    countValidSquares g.points = 11 :=
  sorry

end NUMINAMATH_CALUDE_count_squares_on_marked_grid_l1789_178924


namespace NUMINAMATH_CALUDE_sin_2a_minus_pi_6_l1789_178988

theorem sin_2a_minus_pi_6 (a : ℝ) (h : Real.sin (π / 3 - a) = 1 / 4) :
  Real.sin (2 * a - π / 6) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2a_minus_pi_6_l1789_178988


namespace NUMINAMATH_CALUDE_exponential_inequality_l1789_178951

theorem exponential_inequality (x : ℝ) : 3^x < (1:ℝ)/27 ↔ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1789_178951


namespace NUMINAMATH_CALUDE_snow_probability_l1789_178952

theorem snow_probability : 
  let p1 : ℚ := 1/5  -- probability of snow for each of the first 5 days
  let p2 : ℚ := 1/3  -- probability of snow for each of the next 5 days
  let days1 : ℕ := 5  -- number of days with probability p1
  let days2 : ℕ := 5  -- number of days with probability p2
  let prob_at_least_one_snow : ℚ := 1 - (1 - p1)^days1 * (1 - p2)^days2
  prob_at_least_one_snow = 726607/759375 := by
sorry

end NUMINAMATH_CALUDE_snow_probability_l1789_178952


namespace NUMINAMATH_CALUDE_kind_wizard_succeeds_for_odd_n_l1789_178911

/-- Represents a friendship between two dwarves -/
structure Friendship :=
  (dwarf1 : ℕ)
  (dwarf2 : ℕ)

/-- Creates a list of friendships based on the wizard's pairing strategy -/
def createFriendships (n : ℕ) : List Friendship := sorry

/-- Breaks n friendships from the list -/
def breakFriendships (friendships : List Friendship) (n : ℕ) : List Friendship := sorry

/-- Checks if the remaining friendships can form a valid circular arrangement -/
def canFormCircularArrangement (friendships : List Friendship) : Prop := sorry

theorem kind_wizard_succeeds_for_odd_n (n : ℕ) (h : Odd n) :
  ∀ (broken : List Friendship),
    broken.length = n →
    canFormCircularArrangement (breakFriendships (createFriendships n) n) :=
sorry

end NUMINAMATH_CALUDE_kind_wizard_succeeds_for_odd_n_l1789_178911


namespace NUMINAMATH_CALUDE_expression_defined_iff_l1789_178997

def is_defined (x : ℝ) : Prop :=
  x > 2 ∧ x < 5

theorem expression_defined_iff (x : ℝ) :
  is_defined x ↔ (∃ y : ℝ, y = (Real.log (5 - x)) / Real.sqrt (x - 2)) :=
by sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l1789_178997


namespace NUMINAMATH_CALUDE_not_all_monotonic_functions_have_extremum_l1789_178978

-- Define a monotonic function
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the existence of an extremum value
def HasExtremum (f : ℝ → ℝ) : Prop :=
  ∃ x, ∀ y, f y ≤ f x ∨ f x ≤ f y

-- Theorem statement
theorem not_all_monotonic_functions_have_extremum :
  ∃ f : ℝ → ℝ, MonotonicFunction f ∧ ¬HasExtremum f := by
  sorry

end NUMINAMATH_CALUDE_not_all_monotonic_functions_have_extremum_l1789_178978


namespace NUMINAMATH_CALUDE_gcd_problems_l1789_178970

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 459 357 = 51) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l1789_178970


namespace NUMINAMATH_CALUDE_increasing_power_function_m_l1789_178969

/-- A power function f(x) = (m^2 - 3)x^(m+1) is increasing on (0, +∞) -/
def is_increasing_power_function (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → Monotone (fun x => (m^2 - 3) * x^(m+1))

/-- The value of m for which the power function is increasing -/
theorem increasing_power_function_m : 
  ∃ m : ℝ, is_increasing_power_function m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_increasing_power_function_m_l1789_178969


namespace NUMINAMATH_CALUDE_train_meeting_point_train_A_distance_l1789_178962

theorem train_meeting_point (total_distance : ℝ) (time_A time_B : ℝ) (h1 : total_distance = 75) 
  (h2 : time_A = 3) (h3 : time_B = 2) : ℝ :=
  let speed_A := total_distance / time_A
  let speed_B := total_distance / time_B
  let relative_speed := speed_A + speed_B
  let meeting_time := total_distance / relative_speed
  speed_A * meeting_time

theorem train_A_distance (total_distance : ℝ) (time_A time_B : ℝ) (h1 : total_distance = 75) 
  (h2 : time_A = 3) (h3 : time_B = 2) : 
  train_meeting_point total_distance time_A time_B h1 h2 h3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_point_train_A_distance_l1789_178962


namespace NUMINAMATH_CALUDE_smallest_among_four_rationals_l1789_178949

theorem smallest_among_four_rationals :
  let S : Set ℚ := {-1, 0, 1, 2}
  ∀ x ∈ S, -1 ≤ x
  ∧ ∃ y ∈ S, y = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_four_rationals_l1789_178949


namespace NUMINAMATH_CALUDE_cyclist_speed_l1789_178939

/-- Proves that given a hiker walking at 4 km/h and a cyclist who stops 5 minutes after passing the hiker,
    if it takes the hiker 17.5 minutes to catch up to the cyclist, then the cyclist's speed is 14 km/h. -/
theorem cyclist_speed (hiker_speed : ℝ) (cyclist_ride_time : ℝ) (hiker_catch_up_time : ℝ) :
  hiker_speed = 4 →
  cyclist_ride_time = 5 / 60 →
  hiker_catch_up_time = 17.5 / 60 →
  ∃ (cyclist_speed : ℝ),
    cyclist_speed * cyclist_ride_time = hiker_speed * (cyclist_ride_time + hiker_catch_up_time) ∧
    cyclist_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l1789_178939


namespace NUMINAMATH_CALUDE_largest_inscribed_rectangle_area_l1789_178916

theorem largest_inscribed_rectangle_area (r : ℝ) (h : r = 6) :
  let d := 2 * r
  let s := d / Real.sqrt 2
  s * s = 72 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_rectangle_area_l1789_178916


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1789_178959

/-- Proves that given a 10% reduction in the price of oil, if a housewife can obtain 6 kgs more 
    for Rs. 900 after the reduction, then the reduced price per kg of oil is Rs. 15. -/
theorem oil_price_reduction (original_price : ℝ) : 
  let reduced_price := original_price * 0.9
  let original_quantity := 900 / original_price
  let new_quantity := 900 / reduced_price
  new_quantity = original_quantity + 6 →
  reduced_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1789_178959


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1789_178920

theorem sum_of_fourth_powers (a b : ℂ) 
  (h1 : (a + 1) * (b + 1) = 2)
  (h2 : (a^2 + 1) * (b^2 + 1) = 32) :
  ∃ x y : ℂ, 
    (x^4 + 1) * (y^4 + 1) + (a^4 + 1) * (b^4 + 1) = 1924 ∧
    ((x + 1) * (y + 1) = 2 ∧ (x^2 + 1) * (y^2 + 1) = 32) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1789_178920


namespace NUMINAMATH_CALUDE_garden_area_l1789_178937

/-- The total area of a garden with a semicircle and an attached square -/
theorem garden_area (diameter : ℝ) (h : diameter = 8) : 
  let radius := diameter / 2
  let semicircle_area := π * radius^2 / 2
  let square_area := radius^2
  semicircle_area + square_area = 8 * π + 16 := by
  sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l1789_178937


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l1789_178932

theorem last_three_digits_of_7_power_10000 (h : 7^250 ≡ 1 [ZMOD 1250]) :
  7^10000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_power_10000_l1789_178932


namespace NUMINAMATH_CALUDE_max_imag_part_is_sin_45_l1789_178986

-- Define the complex polynomial
def f (z : ℂ) : ℂ := z^6 - z^4 + z^2 - 1

-- Define the set of roots
def roots : Set ℂ := {z : ℂ | f z = 0}

-- Theorem statement
theorem max_imag_part_is_sin_45 :
  ∃ (z : ℂ), z ∈ roots ∧ 
    ∀ (w : ℂ), w ∈ roots → Complex.im w ≤ Complex.im z ∧ 
      Complex.im z = Real.sin (π/4) :=
sorry

end NUMINAMATH_CALUDE_max_imag_part_is_sin_45_l1789_178986


namespace NUMINAMATH_CALUDE_line_through_point_l1789_178901

theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 2) * (-5) = b - 1) → b = 11/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1789_178901


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l1789_178922

theorem percentage_increase_proof (initial_earnings new_earnings : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : new_earnings = 110) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 83.33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l1789_178922


namespace NUMINAMATH_CALUDE_elvin_internet_charge_l1789_178913

/-- Represents Elvin's monthly telephone bill structure -/
structure TelephoneBill where
  fixedCharge : ℝ
  callCharge : ℝ

/-- Calculates the total bill for a given month -/
def totalBill (bill : TelephoneBill) : ℝ :=
  bill.fixedCharge + bill.callCharge

theorem elvin_internet_charge : 
  ∀ (jan : TelephoneBill) (feb : TelephoneBill),
    totalBill jan = 50 →
    totalBill feb = 76 →
    feb.callCharge = 2 * jan.callCharge →
    jan.fixedCharge = feb.fixedCharge →
    jan.fixedCharge = 24 := by
  sorry


end NUMINAMATH_CALUDE_elvin_internet_charge_l1789_178913


namespace NUMINAMATH_CALUDE_tensor_self_zero_tensor_dot_product_identity_l1789_178990

/-- Definition of the ⊗ operation for 2D vectors -/
def tensor (a b : ℝ × ℝ) : ℝ := a.1 * a.2 - b.1 * b.2

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem tensor_self_zero (a : ℝ × ℝ) : tensor a a = 0 := by sorry

theorem tensor_dot_product_identity (a b : ℝ × ℝ) :
  (tensor a b)^2 + (dot_product a b)^2 = (a.1^2 + b.2^2) * (a.2^2 + b.1^2) := by sorry

end NUMINAMATH_CALUDE_tensor_self_zero_tensor_dot_product_identity_l1789_178990


namespace NUMINAMATH_CALUDE_unique_square_pattern_l1789_178991

def fits_pattern (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∀ d₁ d₂ d₃, n = 100 * d₁ + 10 * d₂ + d₃ →
    d₁ * d₁ < 10 ∧
    d₁ * d₂ < 10 ∧
    d₁ * d₃ < 10 ∧
    d₂ * d₂ < 10 ∧
    d₂ * d₃ < 10 ∧
    d₃ * d₃ < 10

theorem unique_square_pattern :
  ∃! n : ℕ, fits_pattern n ∧ n = 233 :=
sorry

end NUMINAMATH_CALUDE_unique_square_pattern_l1789_178991


namespace NUMINAMATH_CALUDE_cos15_cos45_minus_sin165_sin45_l1789_178909

theorem cos15_cos45_minus_sin165_sin45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) - 
  Real.sin (165 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos15_cos45_minus_sin165_sin45_l1789_178909


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1789_178925

theorem polynomial_divisibility (k n : ℕ) (P : Polynomial ℤ) : 
  Even k → 
  (∀ i : ℕ, i < k → Odd (P.coeff i)) → 
  P.degree = k → 
  (∃ Q : Polynomial ℤ, (X + 1)^n - 1 = P * Q) → 
  (k + 1) ∣ n :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1789_178925


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1789_178984

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := fun x ↦ 5 * x^2 - 2 * x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2/5 ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1789_178984


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1789_178923

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, 
  13 * n = 104 ∧ 
  104 ≥ 100 ∧
  104 < 1000 ∧
  ∀ m : ℕ, (13 * m ≥ 100 ∧ 13 * m < 1000) → 13 * m ≥ 104 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l1789_178923


namespace NUMINAMATH_CALUDE_tower_surface_area_l1789_178933

/-- Calculates the visible surface area of a cube in the tower -/
def visibleSurfaceArea (sideLength : ℕ) (isTop : Bool) : ℕ :=
  if isTop then 5 * sideLength^2 else 4 * sideLength^2

/-- Represents the tower of cubes -/
def cubesTower : List ℕ := [9, 1, 7, 3, 5, 4, 6, 8]

/-- Calculates the total visible surface area of the tower -/
def totalVisibleSurfaceArea (tower : List ℕ) : ℕ :=
  let n := tower.length
  tower.enum.foldl (fun acc (i, sideLength) =>
    acc + visibleSurfaceArea sideLength (i == n - 1)) 0

theorem tower_surface_area :
  totalVisibleSurfaceArea cubesTower = 1408 := by
  sorry

#eval totalVisibleSurfaceArea cubesTower

end NUMINAMATH_CALUDE_tower_surface_area_l1789_178933


namespace NUMINAMATH_CALUDE_rajdhani_speed_calculation_l1789_178960

/-- The speed of Bombay Express in km/h -/
def bombay_speed : ℝ := 60

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

/-- The distance at which the two trains meet in km -/
def meeting_distance : ℝ := 480

/-- The speed of Rajdhani Express in km/h -/
def rajdhani_speed : ℝ := 80

theorem rajdhani_speed_calculation :
  let distance_covered_by_bombay : ℝ := bombay_speed * time_difference
  let remaining_distance : ℝ := meeting_distance - distance_covered_by_bombay
  let time_to_meet : ℝ := remaining_distance / bombay_speed
  rajdhani_speed = meeting_distance / time_to_meet :=
by sorry

end NUMINAMATH_CALUDE_rajdhani_speed_calculation_l1789_178960


namespace NUMINAMATH_CALUDE_eight_flavors_twentyeight_sundaes_l1789_178928

/-- The number of unique two scoop sundaes with distinct flavors given n flavors of ice cream -/
def uniqueSundaes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that with 8 flavors, there are 28 unique two scoop sundaes -/
theorem eight_flavors_twentyeight_sundaes : uniqueSundaes 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_flavors_twentyeight_sundaes_l1789_178928


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1789_178975

theorem sqrt_equation_solution (x : ℚ) :
  (Real.sqrt (6 * x) / Real.sqrt (5 * (x - 2)) = 3) → x = 30 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1789_178975


namespace NUMINAMATH_CALUDE_max_house_paintable_area_l1789_178973

/-- The total area of walls to be painted in Max's house -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - non_paintable_area)

/-- Theorem stating the total area of walls to be painted in Max's house -/
theorem max_house_paintable_area :
  total_paintable_area 4 15 12 9 80 = 1624 := by
  sorry

end NUMINAMATH_CALUDE_max_house_paintable_area_l1789_178973


namespace NUMINAMATH_CALUDE_bo_words_per_day_l1789_178929

def words_per_day (total_flashcards : ℕ) (known_percentage : ℚ) (days_to_learn : ℕ) : ℚ :=
  (total_flashcards : ℚ) * (1 - known_percentage) / days_to_learn

theorem bo_words_per_day :
  words_per_day 800 (1/5) 40 = 16 := by sorry

end NUMINAMATH_CALUDE_bo_words_per_day_l1789_178929


namespace NUMINAMATH_CALUDE_unique_cube_pair_l1789_178948

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem unique_cube_pair :
  ∃! (a b : ℕ),
    1000 ≤ a ∧ a < 10000 ∧
    100 ≤ b ∧ b < 1000 ∧
    is_perfect_cube a ∧
    is_perfect_cube b ∧
    a / 100 = b / 10 ∧
    has_unique_digits a ∧
    has_unique_digits b ∧
    a = 1728 ∧
    b = 125 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_pair_l1789_178948


namespace NUMINAMATH_CALUDE_nuts_in_masons_car_l1789_178917

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stored by each busy squirrel per day -/
def busy_nuts_per_day : ℕ := 30

/-- The number of days busy squirrels have been storing nuts -/
def busy_days : ℕ := 35

/-- The number of slightly lazy squirrels -/
def lazy_squirrels : ℕ := 3

/-- The number of nuts stored by each slightly lazy squirrel per day -/
def lazy_nuts_per_day : ℕ := 20

/-- The number of days slightly lazy squirrels have been storing nuts -/
def lazy_days : ℕ := 40

/-- The number of extremely sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts stored by the extremely sleepy squirrel per day -/
def sleepy_nuts_per_day : ℕ := 10

/-- The number of days the extremely sleepy squirrel has been storing nuts -/
def sleepy_days : ℕ := 45

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := busy_squirrels * busy_nuts_per_day * busy_days +
                      lazy_squirrels * lazy_nuts_per_day * lazy_days +
                      sleepy_squirrels * sleepy_nuts_per_day * sleepy_days

theorem nuts_in_masons_car : total_nuts = 4950 := by
  sorry

end NUMINAMATH_CALUDE_nuts_in_masons_car_l1789_178917


namespace NUMINAMATH_CALUDE_compound_interest_principal_l1789_178938

/-- Given a sum of 5292 after 2 years with an interest rate of 5% per annum compounded yearly, 
    prove that the principal amount is 4800. -/
theorem compound_interest_principal (sum : ℝ) (years : ℕ) (rate : ℝ) (principal : ℝ) : 
  sum = 5292 →
  years = 2 →
  rate = 0.05 →
  sum = principal * (1 + rate) ^ years →
  principal = 4800 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l1789_178938


namespace NUMINAMATH_CALUDE_division_problem_l1789_178906

theorem division_problem (A : ℕ) (h1 : 26 = A * 8 + 2) : A = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1789_178906


namespace NUMINAMATH_CALUDE_rhombus_field_area_l1789_178985

/-- Represents the length of the long diagonal of a rhombus-shaped field in miles. -/
def long_diagonal : ℝ := 2500

/-- Represents the area of the rhombus-shaped field in square miles. -/
def field_area : ℝ := 1562500

/-- Theorem stating that the area of the rhombus-shaped field is 1562500 square miles. -/
theorem rhombus_field_area : field_area = (1 / 2) * long_diagonal * (long_diagonal / 2) := by
  sorry

#check rhombus_field_area

end NUMINAMATH_CALUDE_rhombus_field_area_l1789_178985


namespace NUMINAMATH_CALUDE_thirty_people_three_groups_l1789_178992

/-- The number of ways to divide n people into k groups of m people each -/
def group_divisions (n m k : ℕ) : ℕ :=
  if n = m * k then
    Nat.factorial n / (Nat.factorial m ^ k)
  else
    0

/-- Theorem: The number of ways to divide 30 people into 3 groups of 10 each
    is equal to 30! / (10!)³ -/
theorem thirty_people_three_groups :
  group_divisions 30 10 3 = Nat.factorial 30 / (Nat.factorial 10 ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_thirty_people_three_groups_l1789_178992


namespace NUMINAMATH_CALUDE_percentage_increase_l1789_178989

theorem percentage_increase (x : ℝ) : 
  x > 98 ∧ x = 117.6 → (x - 98) / 98 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1789_178989


namespace NUMINAMATH_CALUDE_sum_factorials_25_divisible_by_26_l1789_178974

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_25_divisible_by_26 :
  ∃ k : ℕ, sum_factorials 25 = 26 * k :=
sorry

end NUMINAMATH_CALUDE_sum_factorials_25_divisible_by_26_l1789_178974


namespace NUMINAMATH_CALUDE_sandwich_change_l1789_178942

/-- Calculates the change received when buying a number of items at a given price and paying with a certain amount. -/
def calculate_change (num_items : ℕ) (price_per_item : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_items * price_per_item)

/-- Proves that buying 3 items at $5 each, paid with a $20 bill, results in $5 change. -/
theorem sandwich_change : calculate_change 3 5 20 = 5 := by
  sorry

#eval calculate_change 3 5 20

end NUMINAMATH_CALUDE_sandwich_change_l1789_178942


namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l1789_178956

theorem simplify_trig_fraction (x : ℝ) :
  (2 + 2 * Real.sin x - 2 * Real.cos x) / (2 + 2 * Real.sin x + 2 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l1789_178956
