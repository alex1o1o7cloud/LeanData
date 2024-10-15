import Mathlib

namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2120_212048

def C : Set Nat := {70, 72, 75, 76, 78}

theorem smallest_prime_factor_in_C :
  ∃ n ∈ C, (∃ p : Nat, Nat.Prime p ∧ p ∣ n ∧ p = 2) ∧
  ∀ m ∈ C, ∀ q : Nat, Nat.Prime q → q ∣ m → q ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2120_212048


namespace NUMINAMATH_CALUDE_nitin_ranks_from_last_l2120_212060

/-- Calculates the rank from last given the total number of students and rank from first -/
def rankFromLast (totalStudents : ℕ) (rankFromFirst : ℕ) : ℕ :=
  totalStudents - rankFromFirst + 1

theorem nitin_ranks_from_last 
  (totalStudents : ℕ) 
  (mathRank : ℕ) 
  (englishRank : ℕ) 
  (h1 : totalStudents = 75) 
  (h2 : mathRank = 24) 
  (h3 : englishRank = 18) : 
  (rankFromLast totalStudents mathRank = 52) ∧ 
  (rankFromLast totalStudents englishRank = 58) :=
by
  sorry

end NUMINAMATH_CALUDE_nitin_ranks_from_last_l2120_212060


namespace NUMINAMATH_CALUDE_factor_sum_l2120_212064

theorem factor_sum (R S : ℝ) : 
  (∃ d e : ℝ, (X ^ 2 + 3 * X + 7) * (X ^ 2 + d * X + e) = X ^ 4 + R * X ^ 2 + S) →
  R + S = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2120_212064


namespace NUMINAMATH_CALUDE_circle_distance_relation_l2120_212063

/-- Given a circle with center O and radius 2a, prove the relationship between x and y -/
theorem circle_distance_relation (a : ℝ) (x y : ℝ) : 
  x > 0 → a > 0 → y^2 = x^3 / (2*a + x) := by
  sorry


end NUMINAMATH_CALUDE_circle_distance_relation_l2120_212063


namespace NUMINAMATH_CALUDE_linear_system_solution_l2120_212097

/-- Given a system of linear equations and conditions, prove the range of m and its integer values -/
theorem linear_system_solution (m x y : ℝ) : 
  (2 * x + y = 1 + 2 * m) → 
  (x + 2 * y = 2 - m) → 
  (x + y > 0) → 
  (m > -3) ∧ 
  (((2 * m + 1) * x - 2 * m < 1) → 
   (x > 1) → 
   (m = -2 ∨ m = -1)) := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2120_212097


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l2120_212082

/-- The number of boys in a class with given height information -/
theorem number_of_boys_in_class :
  ∀ (n : ℕ) (initial_avg real_avg wrong_height actual_height : ℝ),
  initial_avg = 180 →
  wrong_height = 166 →
  actual_height = 106 →
  real_avg = 178 →
  initial_avg * n - (wrong_height - actual_height) = real_avg * n →
  n = 30 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l2120_212082


namespace NUMINAMATH_CALUDE_charlies_lollipops_l2120_212051

/-- Given the number of lollipops of each flavor and the number of friends,
    prove that the number of lollipops Charlie keeps is the remainder when
    the total number of lollipops is divided by the number of friends. -/
theorem charlies_lollipops
  (cherry wintergreen grape shrimp_cocktail raspberry : ℕ)
  (friends : ℕ) (friends_pos : friends > 0) :
  let total := cherry + wintergreen + grape + shrimp_cocktail + raspberry
  (total % friends) = total - friends * (total / friends) :=
by sorry

end NUMINAMATH_CALUDE_charlies_lollipops_l2120_212051


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2120_212079

theorem sum_of_three_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2120_212079


namespace NUMINAMATH_CALUDE_neighbor_dog_ate_five_chickens_l2120_212006

/-- The number of chickens eaten by the neighbor's dog -/
def chickens_eaten (initial : ℕ) (final : ℕ) : ℕ :=
  2 * initial + 6 - final

theorem neighbor_dog_ate_five_chickens : chickens_eaten 4 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_dog_ate_five_chickens_l2120_212006


namespace NUMINAMATH_CALUDE_range_of_f_l2120_212040

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The range of f is [2, +∞) -/
theorem range_of_f : Set.range f = { y | 2 ≤ y } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2120_212040


namespace NUMINAMATH_CALUDE_min_sum_of_product_36_l2120_212091

theorem min_sum_of_product_36 (c d : ℤ) (h : c * d = 36) :
  ∃ (m : ℤ), m = -37 ∧ c + d ≥ m ∧ ∃ (c' d' : ℤ), c' * d' = 36 ∧ c' + d' = m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_36_l2120_212091


namespace NUMINAMATH_CALUDE_x_cubed_plus_two_x_squared_plus_2007_l2120_212032

theorem x_cubed_plus_two_x_squared_plus_2007 (x : ℝ) (h : x^2 + x - 1 = 0) :
  x^3 + 2*x^2 + 2007 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_plus_two_x_squared_plus_2007_l2120_212032


namespace NUMINAMATH_CALUDE_kindergarten_pet_distribution_l2120_212093

/-- Represents the kindergarten pet distribution problem -/
theorem kindergarten_pet_distribution 
  (total_children : ℕ) 
  (children_with_both : ℕ) 
  (children_with_cats : ℕ) 
  (h1 : total_children = 30)
  (h2 : children_with_both = 6)
  (h3 : children_with_cats = 12)
  : total_children - children_with_cats = 18 :=
by sorry

end NUMINAMATH_CALUDE_kindergarten_pet_distribution_l2120_212093


namespace NUMINAMATH_CALUDE_fraction_difference_numerator_l2120_212076

theorem fraction_difference_numerator : ∃ (p q : ℕ+), 
  (2024 : ℚ) / 2023 - (2023 : ℚ) / 2024 = (p : ℚ) / q ∧ 
  Nat.gcd p q = 1 ∧ 
  p = 4047 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_numerator_l2120_212076


namespace NUMINAMATH_CALUDE_remainder_5_pow_2021_mod_17_l2120_212007

theorem remainder_5_pow_2021_mod_17 : 5^2021 % 17 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5_pow_2021_mod_17_l2120_212007


namespace NUMINAMATH_CALUDE_sampling_most_appropriate_for_qingming_l2120_212054

-- Define the survey methods
inductive SurveyMethod
| Census
| Sampling

-- Define the survey scenarios
inductive SurveyScenario
| MilkHygiene
| SubwaySecurity
| StudentSleep
| QingmingCommemoration

-- Define a function to determine the appropriateness of a survey method for a given scenario
def is_appropriate (scenario : SurveyScenario) (method : SurveyMethod) : Prop :=
  match scenario, method with
  | SurveyScenario.MilkHygiene, SurveyMethod.Sampling => True
  | SurveyScenario.SubwaySecurity, SurveyMethod.Census => True
  | SurveyScenario.StudentSleep, SurveyMethod.Sampling => True
  | SurveyScenario.QingmingCommemoration, SurveyMethod.Sampling => True
  | _, _ => False

-- Theorem stating that sampling is the most appropriate method for the Qingming commemoration scenario
theorem sampling_most_appropriate_for_qingming :
  ∀ (scenario : SurveyScenario) (method : SurveyMethod),
    is_appropriate scenario method →
    (scenario = SurveyScenario.QingmingCommemoration ∧ method = SurveyMethod.Sampling) ∨
    (scenario ≠ SurveyScenario.QingmingCommemoration) :=
by sorry

end NUMINAMATH_CALUDE_sampling_most_appropriate_for_qingming_l2120_212054


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2120_212001

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2120_212001


namespace NUMINAMATH_CALUDE_special_function_at_two_l2120_212011

/-- A function satisfying the given property for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f x - f (f y + f (-x)) + x

/-- Theorem stating that for any function satisfying the special property, f(2) = -2 -/
theorem special_function_at_two (f : ℝ → ℝ) (h : special_function f) : f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_two_l2120_212011


namespace NUMINAMATH_CALUDE_complex_magnitude_l2120_212016

theorem complex_magnitude (a b : ℝ) : 
  (Complex.I + a * Complex.I) * Complex.I = 1 - b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2120_212016


namespace NUMINAMATH_CALUDE_fletcher_well_diggers_l2120_212088

/-- The number of men hired by Mr Fletcher to dig a well -/
def num_men : ℕ :=
  let hours_day1 : ℕ := 10
  let hours_day2 : ℕ := 8
  let hours_day3 : ℕ := 15
  let total_hours : ℕ := hours_day1 + hours_day2 + hours_day3
  let pay_per_hour : ℕ := 10
  let total_payment : ℕ := 660
  total_payment / (total_hours * pay_per_hour)

theorem fletcher_well_diggers :
  num_men = 2 := by sorry

end NUMINAMATH_CALUDE_fletcher_well_diggers_l2120_212088


namespace NUMINAMATH_CALUDE_negative_response_proportion_l2120_212085

/-- Given 88 total teams and 49 teams with negative responses,
    prove that P = ⌊10000 * (49/88)⌋ = 5568 -/
theorem negative_response_proportion (total_teams : Nat) (negative_responses : Nat)
    (h1 : total_teams = 88)
    (h2 : negative_responses = 49) :
    ⌊(10000 : ℝ) * ((negative_responses : ℝ) / (total_teams : ℝ))⌋ = 5568 := by
  sorry

#check negative_response_proportion

end NUMINAMATH_CALUDE_negative_response_proportion_l2120_212085


namespace NUMINAMATH_CALUDE_P_equals_set_l2120_212073

def P : Set ℝ := {x | x^2 = 1}

theorem P_equals_set : P = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_P_equals_set_l2120_212073


namespace NUMINAMATH_CALUDE_interest_problem_l2120_212057

/-- Proves that given the conditions of the interest problem, the sum is 700 --/
theorem interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 7.5) * 12 / 100 - P * R * 12 / 100 = 630) → P = 700 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l2120_212057


namespace NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l2120_212092

/-- The number of days in a leap year -/
def leapYearDays : ℕ := 366

/-- The number of full weeks in a leap year -/
def fullWeeks : ℕ := leapYearDays / 7

/-- The number of extra days in a leap year after full weeks -/
def extraDays : ℕ := leapYearDays % 7

/-- The number of possible combinations for the extra days -/
def possibleCombinations : ℕ := 7

/-- The number of combinations that include a Monday -/
def combinationsWithMonday : ℕ := 2

/-- The probability of a leap year having 53 Mondays -/
def probabilityOf53Mondays : ℚ := combinationsWithMonday / possibleCombinations

theorem leap_year_53_mondays_probability :
  probabilityOf53Mondays = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_leap_year_53_mondays_probability_l2120_212092


namespace NUMINAMATH_CALUDE_s_squared_minus_c_squared_range_l2120_212056

/-- The theorem states that for any point (x, y, z) in 3D space,
    where r is the distance from the origin to the point,
    s = y/r, and c = x/r, the value of s^2 - c^2 is always
    between -1 and 1, inclusive. -/
theorem s_squared_minus_c_squared_range
  (x y z : ℝ) 
  (r : ℝ) 
  (hr : r = Real.sqrt (x^2 + y^2 + z^2)) 
  (s : ℝ) 
  (hs : s = y / r) 
  (c : ℝ) 
  (hc : c = x / r) : 
  -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_s_squared_minus_c_squared_range_l2120_212056


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_circumcentre_l2120_212062

/-- A point in the Euclidean plane -/
structure Point : Type :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line : Type :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of a cyclic quadrilateral -/
def is_cyclic_quadrilateral (A B X C O : Point) : Prop := sorry

/-- Definition of a point lying on a line -/
def point_on_line (P : Point) (L : Line) : Prop := sorry

/-- Definition of equality of distances -/
def distance_eq (A B C D : Point) : Prop := sorry

/-- Definition of a circumcentre of a triangle -/
def is_circumcentre (O : Point) (A B C : Point) : Prop := sorry

/-- Definition of a perpendicular bisector of a line segment -/
def perpendicular_bisector (L : Line) (A B : Point) : Prop := sorry

/-- Definition of a point lying on a perpendicular bisector -/
def point_on_perp_bisector (P : Point) (L : Line) (A B : Point) : Prop := sorry

theorem cyclic_quadrilateral_circumcentre 
  (A B X C O D E : Point) (BX CX : Line) :
  is_cyclic_quadrilateral A B X C O →
  point_on_line D BX →
  point_on_line E CX →
  distance_eq A D B D →
  distance_eq A E C E →
  ∃ (O₁ : Point), is_circumcentre O₁ D E X ∧
    ∃ (L : Line), perpendicular_bisector L O A ∧ point_on_perp_bisector O₁ L O A :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_circumcentre_l2120_212062


namespace NUMINAMATH_CALUDE_visitors_calculation_l2120_212030

/-- The number of visitors to Buckingham Palace on a specific day, given the total visitors over 85 days and the visitors on the previous day. -/
def visitors_on_day (total_visitors : ℕ) (previous_day_visitors : ℕ) : ℕ :=
  total_visitors - previous_day_visitors

/-- Theorem stating that the number of visitors on a specific day is equal to
    the total visitors over 85 days minus the visitors on the previous day. -/
theorem visitors_calculation (total_visitors previous_day_visitors : ℕ) 
    (h1 : total_visitors = 829)
    (h2 : previous_day_visitors = 45) :
  visitors_on_day total_visitors previous_day_visitors = 784 := by
  sorry

#eval visitors_on_day 829 45

end NUMINAMATH_CALUDE_visitors_calculation_l2120_212030


namespace NUMINAMATH_CALUDE_exists_zero_sum_choice_l2120_212058

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of 8 points in the plane -/
def Points : Finset Point := sorry

/-- The area of a triangle formed by three points -/
def triangleArea (p₁ p₂ p₃ : Point) : ℝ := sorry

/-- The list of areas of all possible triangles formed by the points -/
def triangleAreas : List ℝ := sorry

/-- A choice of signs for the areas -/
def SignChoice := List Bool

/-- Apply a sign choice to a list of areas -/
def applySignChoice (areas : List ℝ) (choice : SignChoice) : List ℝ := sorry

theorem exists_zero_sum_choice :
  ∃ (choice : SignChoice), List.sum (applySignChoice triangleAreas choice) = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_zero_sum_choice_l2120_212058


namespace NUMINAMATH_CALUDE_solve_plug_problem_l2120_212069

def plug_problem (mittens_pairs : ℕ) (original_plug_pairs_diff : ℕ) (final_plugs : ℕ) : Prop :=
  let original_plug_pairs : ℕ := mittens_pairs + original_plug_pairs_diff
  let original_plugs : ℕ := original_plug_pairs * 2
  let added_plugs : ℕ := final_plugs - original_plugs
  let added_pairs : ℕ := added_plugs / 2
  added_pairs = 70

theorem solve_plug_problem :
  plug_problem 150 20 400 :=
by sorry

end NUMINAMATH_CALUDE_solve_plug_problem_l2120_212069


namespace NUMINAMATH_CALUDE_gwen_homework_problems_l2120_212098

/-- Represents the number of problems for each subject -/
structure SubjectProblems where
  math : ℕ
  science : ℕ
  history : ℕ
  english : ℕ

/-- Calculates the total number of problems left for homework -/
def problems_left (initial : SubjectProblems) (completed : SubjectProblems) : ℕ :=
  (initial.math - completed.math) +
  (initial.science - completed.science) +
  (initial.history - completed.history) +
  (initial.english - completed.english)

/-- Theorem: Given Gwen's initial problems and completed problems, she has 19 problems left for homework -/
theorem gwen_homework_problems :
  let initial := SubjectProblems.mk 18 11 15 7
  let completed := SubjectProblems.mk 12 6 10 4
  problems_left initial completed = 19 := by
  sorry

end NUMINAMATH_CALUDE_gwen_homework_problems_l2120_212098


namespace NUMINAMATH_CALUDE_relationship_abcd_l2120_212024

theorem relationship_abcd (a b c d : ℝ) 
  (hab : a < b) 
  (hdc : d < c) 
  (hcab : (c - a) * (c - b) < 0) 
  (hdab : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_relationship_abcd_l2120_212024


namespace NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l2120_212045

theorem triangle_area_with_cosine_root (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → b = 5 → (5 * cos_theta^2 - 7 * cos_theta - 6 = 0) → 
  (1/2 : ℝ) * a * b * cos_theta = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l2120_212045


namespace NUMINAMATH_CALUDE_alexandrov_theorem_l2120_212072

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a face of a polyhedron -/
structure Face where
  -- Add necessary fields here
  mk ::

/-- Represents a planar angle in a polyhedron -/
def PlanarAngle : Type := ℝ

/-- Represents a dihedral angle in a polyhedron -/
def DihedralAngle : Type := ℝ

/-- Check if two polyhedra have correspondingly equal faces -/
def has_equal_faces (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Check if two polyhedra have equal corresponding planar angles -/
def has_equal_planar_angles (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Check if two polyhedra have equal corresponding dihedral angles -/
def has_equal_dihedral_angles (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Alexandrov's Theorem -/
theorem alexandrov_theorem (P Q : ConvexPolyhedron) :
  has_equal_faces P Q → has_equal_planar_angles P Q → has_equal_dihedral_angles P Q :=
by
  sorry

end NUMINAMATH_CALUDE_alexandrov_theorem_l2120_212072


namespace NUMINAMATH_CALUDE_socks_cost_theorem_l2120_212074

def flat_rate : ℝ := 5
def shipping_rate : ℝ := 0.2
def shirt_price : ℝ := 12
def shirt_quantity : ℕ := 3
def shorts_price : ℝ := 15
def shorts_quantity : ℕ := 2
def swim_trunks_price : ℝ := 14
def swim_trunks_quantity : ℕ := 1
def total_bill : ℝ := 102

def known_items_cost : ℝ := 
  shirt_price * shirt_quantity + 
  shorts_price * shorts_quantity + 
  swim_trunks_price * swim_trunks_quantity

theorem socks_cost_theorem (socks_price : ℝ) : 
  (known_items_cost + socks_price > 50 → 
    known_items_cost + socks_price + shipping_rate * (known_items_cost + socks_price) = total_bill) →
  (known_items_cost + socks_price ≤ 50 → 
    known_items_cost + socks_price + flat_rate = total_bill) →
  socks_price = 5 := by
sorry

end NUMINAMATH_CALUDE_socks_cost_theorem_l2120_212074


namespace NUMINAMATH_CALUDE_binary_111111_equals_63_l2120_212021

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111111_equals_63 :
  binary_to_decimal [true, true, true, true, true, true] = 63 := by
  sorry

end NUMINAMATH_CALUDE_binary_111111_equals_63_l2120_212021


namespace NUMINAMATH_CALUDE_frogs_on_logs_count_l2120_212089

/-- The number of frogs that climbed onto logs in the pond -/
def frogs_on_logs (total_frogs lily_pad_frogs rock_frogs : ℕ) : ℕ :=
  total_frogs - (lily_pad_frogs + rock_frogs)

/-- Theorem stating that the number of frogs on logs is 3 -/
theorem frogs_on_logs_count :
  frogs_on_logs 32 5 24 = 3 := by
  sorry

end NUMINAMATH_CALUDE_frogs_on_logs_count_l2120_212089


namespace NUMINAMATH_CALUDE_driver_work_days_l2120_212084

/-- Represents the number of days driven from Monday to Wednesday -/
def days_mon_to_wed : ℕ := 3

/-- Represents the number of days driven from Thursday to Friday -/
def days_thu_to_fri : ℕ := 2

/-- Average driving hours per day -/
def avg_hours_per_day : ℕ := 2

/-- Average speed from Monday to Wednesday in km/h -/
def speed_mon_to_wed : ℕ := 12

/-- Average speed from Thursday to Friday in km/h -/
def speed_thu_to_fri : ℕ := 9

/-- Total distance traveled in km -/
def total_distance : ℕ := 108

theorem driver_work_days : 
  days_mon_to_wed * avg_hours_per_day * speed_mon_to_wed + 
  days_thu_to_fri * avg_hours_per_day * speed_thu_to_fri = total_distance ∧
  days_mon_to_wed + days_thu_to_fri = 5 :=
by sorry

end NUMINAMATH_CALUDE_driver_work_days_l2120_212084


namespace NUMINAMATH_CALUDE_zoo_visitors_saturday_l2120_212037

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := 1250

/-- The ratio of Saturday visitors to Friday visitors -/
def saturday_ratio : ℕ := 3

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := friday_visitors * saturday_ratio

theorem zoo_visitors_saturday : saturday_visitors = 3750 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_saturday_l2120_212037


namespace NUMINAMATH_CALUDE_hockey_handshakes_l2120_212095

theorem hockey_handshakes (team_size : Nat) (num_teams : Nat) (num_referees : Nat) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 → 
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 := by
  sorry

end NUMINAMATH_CALUDE_hockey_handshakes_l2120_212095


namespace NUMINAMATH_CALUDE_segment_mapping_l2120_212077

theorem segment_mapping (a : ℝ) : ∃ (x y : ℝ), 
  (∃ (AB A'B' : ℝ), AB = 3 ∧ A'B' = 6 ∧
  (∀ (P D P' D' : ℝ), 
    (P - D = x ∧ P' - D' = 2*x) →
    (x = a → x + y = 3*a))) :=
by sorry

end NUMINAMATH_CALUDE_segment_mapping_l2120_212077


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l2120_212000

theorem probability_closer_to_center (R : ℝ) (h : R = 4) : 
  (π * 1^2) / (π * R^2) = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l2120_212000


namespace NUMINAMATH_CALUDE_prime_sum_91_l2120_212071

theorem prime_sum_91 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h_sum : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_91_l2120_212071


namespace NUMINAMATH_CALUDE_widgets_per_week_l2120_212034

/-- The number of widgets John can make per hour -/
def widgets_per_hour : ℕ := 20

/-- The number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- The number of days John works per week -/
def days_per_week : ℕ := 5

/-- Theorem: John makes 800 widgets in a week -/
theorem widgets_per_week : 
  widgets_per_hour * hours_per_day * days_per_week = 800 := by
  sorry


end NUMINAMATH_CALUDE_widgets_per_week_l2120_212034


namespace NUMINAMATH_CALUDE_simplify_expression_l2120_212035

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2120_212035


namespace NUMINAMATH_CALUDE_scientific_notation_of_190_million_l2120_212046

theorem scientific_notation_of_190_million :
  (190000000 : ℝ) = 1.9 * (10 : ℝ)^8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_190_million_l2120_212046


namespace NUMINAMATH_CALUDE_min_value_on_line_l2120_212086

theorem min_value_on_line (x y : ℝ) (h : x + 2 * y = 3) :
  2^x + 4^y ≥ 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_line_l2120_212086


namespace NUMINAMATH_CALUDE_carolines_socks_l2120_212050

/-- Given Caroline's sock inventory changes, calculate how many pairs she received as a gift. -/
theorem carolines_socks (initial : ℕ) (lost : ℕ) (donation_fraction : ℚ) (purchased : ℕ) (final : ℕ) : 
  initial = 40 →
  lost = 4 →
  donation_fraction = 2/3 →
  purchased = 10 →
  final = 25 →
  final = initial - lost - (initial - lost) * donation_fraction + purchased + (final - (initial - lost - (initial - lost) * donation_fraction + purchased)) :=
by sorry

end NUMINAMATH_CALUDE_carolines_socks_l2120_212050


namespace NUMINAMATH_CALUDE_gcd_preservation_l2120_212068

theorem gcd_preservation (a b c d x y z G : ℤ) 
  (h : G = Int.gcd a (Int.gcd b (Int.gcd c d))) : 
  G = Int.gcd a (Int.gcd b (Int.gcd c (Int.gcd d (Int.gcd (G*x) (Int.gcd (G*y) (G*z)))))) :=
by sorry

end NUMINAMATH_CALUDE_gcd_preservation_l2120_212068


namespace NUMINAMATH_CALUDE_additional_lawn_needed_l2120_212039

/-- The amount LaKeisha charges per square foot of lawn mowed -/
def lawn_rate : ℚ := 1 / 10

/-- The amount LaKeisha charges per linear foot of hedge trimmed -/
def hedge_rate : ℚ := 1 / 20

/-- The amount LaKeisha charges per square foot of leaves raked -/
def rake_rate : ℚ := 1 / 50

/-- The cost of the book set -/
def book_cost : ℚ := 375

/-- The number of lawns LaKeisha has mowed -/
def lawns_mowed : ℕ := 5

/-- The length of each lawn -/
def lawn_length : ℕ := 30

/-- The width of each lawn -/
def lawn_width : ℕ := 20

/-- The number of linear feet of hedges LaKeisha has trimmed -/
def hedges_trimmed : ℕ := 100

/-- The number of square feet of leaves LaKeisha has raked -/
def leaves_raked : ℕ := 500

/-- The additional square feet of lawn LaKeisha needs to mow -/
def additional_lawn : ℕ := 600

theorem additional_lawn_needed :
  (book_cost - (lawn_rate * (lawns_mowed * lawn_length * lawn_width : ℚ) +
    hedge_rate * hedges_trimmed +
    rake_rate * leaves_raked)) / lawn_rate = additional_lawn := by sorry

end NUMINAMATH_CALUDE_additional_lawn_needed_l2120_212039


namespace NUMINAMATH_CALUDE_phoenix_hike_l2120_212005

/-- Phoenix's hiking problem -/
theorem phoenix_hike (a b c d e : ℝ) 
  (h1 : a + b + c = 36)
  (h2 : (b + c + d) / 3 = 16)
  (h3 : c + d + e = 45)
  (h4 : a + d = 31) :
  a + b + c + d + e = 81 := by
  sorry

end NUMINAMATH_CALUDE_phoenix_hike_l2120_212005


namespace NUMINAMATH_CALUDE_rickey_race_time_l2120_212059

/-- 
Given:
- Prejean's speed is three-quarters that of Rickey's
- The total time for both to run the race is 70 minutes
Prove: Rickey took 30 minutes to finish the race
-/
theorem rickey_race_time :
  ∀ (rickey_speed prejean_speed rickey_time prejean_time : ℚ),
  prejean_speed = (3 / 4) * rickey_speed →
  rickey_time + prejean_time = 70 →
  prejean_time = (4 / 3) * rickey_time →
  rickey_time = 30 :=
by sorry

end NUMINAMATH_CALUDE_rickey_race_time_l2120_212059


namespace NUMINAMATH_CALUDE_fruits_per_person_correct_l2120_212078

/-- The number of fruits each person gets when evenly distributed -/
def fruits_per_person (
  kim_strawberry_multiplier : ℕ)
  (strawberries_per_basket : ℕ)
  (kim_blueberry_baskets : ℕ)
  (blueberries_per_kim_basket : ℕ)
  (brother_strawberry_baskets : ℕ)
  (brother_blackberry_baskets : ℕ)
  (blackberries_per_basket : ℕ)
  (parents_blackberry_difference : ℕ)
  (parents_extra_blueberry_baskets : ℕ)
  (parents_extra_blueberries_per_basket : ℕ)
  (family_size : ℕ) : ℕ :=
  let total_strawberries := 
    (kim_strawberry_multiplier * brother_strawberry_baskets + brother_strawberry_baskets) * strawberries_per_basket
  let total_blueberries := 
    kim_blueberry_baskets * blueberries_per_kim_basket + 
    (kim_blueberry_baskets + parents_extra_blueberry_baskets) * (blueberries_per_kim_basket + parents_extra_blueberries_per_basket)
  let total_blackberries := 
    brother_blackberry_baskets * blackberries_per_basket + 
    (brother_blackberry_baskets * blackberries_per_basket - parents_blackberry_difference)
  let total_fruits := total_strawberries + total_blueberries + total_blackberries
  total_fruits / family_size

theorem fruits_per_person_correct : 
  fruits_per_person 8 15 5 40 3 4 30 75 4 15 4 = 316 := by sorry

end NUMINAMATH_CALUDE_fruits_per_person_correct_l2120_212078


namespace NUMINAMATH_CALUDE_shelter_animals_count_l2120_212013

/-- Calculates the total number of animals in the shelter after adoption and new arrivals --/
def total_animals_after_events (initial_cats initial_dogs initial_rabbits : ℕ) : ℕ :=
  let adopted_cats := initial_cats / 4
  let adopted_dogs := initial_dogs / 3
  let new_cats := 3 * adopted_cats
  let new_dogs := 2 * adopted_dogs
  let final_cats := initial_cats - adopted_cats + new_cats
  let final_dogs := initial_dogs - adopted_dogs + new_dogs
  let final_rabbits := 2 * initial_rabbits
  final_cats + final_dogs + final_rabbits

/-- Theorem stating that given the initial conditions, the total number of animals after events is 210 --/
theorem shelter_animals_count : total_animals_after_events 60 45 30 = 210 := by
  sorry

end NUMINAMATH_CALUDE_shelter_animals_count_l2120_212013


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2120_212044

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 3 * y - 2 = 0) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b : ℝ), a + 3 * b - 2 = 0 → 2^a + 8^b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2120_212044


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l2120_212053

-- Inequality 1
theorem inequality_one (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) ↔ x ≤ -3/2 := by sorry

-- Inequality 2
theorem inequality_two (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l2120_212053


namespace NUMINAMATH_CALUDE_perfect_square_sum_existence_l2120_212008

theorem perfect_square_sum_existence : ∃ (x y z u v w t s : ℕ+), 
  x^2 + y + z + u = (x + v)^2 ∧
  y^2 + x + z + u = (y + w)^2 ∧
  z^2 + x + y + u = (z + t)^2 ∧
  u^2 + x + y + z = (u + s)^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_existence_l2120_212008


namespace NUMINAMATH_CALUDE_max_large_chips_l2120_212081

theorem max_large_chips (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 80 → 
  (∃ (small large prime : ℕ), 
    small + large = total ∧ 
    small = large + prime ∧ 
    is_prime prime) →
  (∀ (large : ℕ), 
    (∃ (small prime : ℕ), 
      small + large = total ∧ 
      small = large + prime ∧ 
      is_prime prime) → 
    large ≤ 39) ∧
  (∃ (small prime : ℕ), 
    small + 39 = total ∧ 
    small = 39 + prime ∧ 
    is_prime prime) :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l2120_212081


namespace NUMINAMATH_CALUDE_group_size_proof_l2120_212049

theorem group_size_proof (W : ℝ) (n : ℕ) 
  (h1 : (W + 20) / n = W / n + 4) 
  (h2 : W > 0) 
  (h3 : n > 0) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2120_212049


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2120_212028

theorem chocolate_distribution (total_pieces : ℕ) (num_boxes : ℕ) (pieces_per_box : ℕ) :
  total_pieces = 3000 →
  num_boxes = 6 →
  total_pieces = num_boxes * pieces_per_box →
  pieces_per_box = 500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2120_212028


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2120_212029

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_factorial_eight_ten :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2120_212029


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2120_212061

theorem trigonometric_identity (x : ℝ) :
  Real.sin (x + 2 * Real.pi) * Real.cos (2 * x - 7 * Real.pi / 2) +
  Real.sin (3 * Real.pi / 2 - x) * Real.sin (2 * x - 5 * Real.pi / 2) =
  Real.cos (3 * x) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2120_212061


namespace NUMINAMATH_CALUDE_intersection_points_quadratic_linear_l2120_212067

theorem intersection_points_quadratic_linear 
  (x y : ℝ) : 
  (y = 3 * x^2 - 6 * x + 5 ∧ y = 2 * x + 1) ↔ 
  ((x = 2 ∧ y = 5) ∨ (x = 2/3 ∧ y = 7/3)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_quadratic_linear_l2120_212067


namespace NUMINAMATH_CALUDE_triangle_count_bound_l2120_212047

/-- A structure representing a configuration of points and equilateral triangles on a plane. -/
structure PointTriangleConfig where
  n : ℕ  -- number of points
  k : ℕ  -- number of equilateral triangles
  n_gt_3 : n > 3
  convex_n_gon : Bool  -- represents that the n points form a convex n-gon
  unit_triangles : Bool  -- represents that the k triangles are equilateral with side length 1

/-- Theorem stating that the number of equilateral triangles is less than 2/3 of the number of points. -/
theorem triangle_count_bound (config : PointTriangleConfig) : config.k < 2 * config.n / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_bound_l2120_212047


namespace NUMINAMATH_CALUDE_normal_pdf_max_normal_pdf_max_decreases_normal_spread_increases_l2120_212042

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

/-- The maximum height of the normal distribution occurs at x = μ -/
theorem normal_pdf_max (μ σ : ℝ) (h : σ > 0) :
  ∀ x, normal_pdf μ σ x ≤ normal_pdf μ σ μ :=
sorry

/-- As σ increases, the maximum height of the normal distribution decreases -/
theorem normal_pdf_max_decreases (μ : ℝ) (σ₁ σ₂ : ℝ) (h₁ : σ₁ > 0) (h₂ : σ₂ > 0) (h₃ : σ₁ < σ₂) :
  normal_pdf μ σ₂ μ < normal_pdf μ σ₁ μ :=
sorry

/-- The spread of the normal distribution increases as σ increases -/
theorem normal_spread_increases (μ : ℝ) (σ₁ σ₂ : ℝ) (h₁ : σ₁ > 0) (h₂ : σ₂ > 0) (h₃ : σ₁ < σ₂) (ε : ℝ) (hε : ε > 0) :
  ∃ x, |x - μ| > ε ∧ normal_pdf μ σ₂ x > normal_pdf μ σ₁ x :=
sorry

end NUMINAMATH_CALUDE_normal_pdf_max_normal_pdf_max_decreases_normal_spread_increases_l2120_212042


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2120_212075

theorem triangle_angle_C (A B C : ℝ) (h_triangle : A + B + C = PI) 
  (h_eq1 : 5 * Real.sin A + 3 * Real.cos B = 8)
  (h_eq2 : 3 * Real.sin B + 5 * Real.cos A = 0) :
  C = PI / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2120_212075


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l2120_212065

/-- The percentage of students who passed an examination, given the total number of students and the number of students who failed. -/
def percentage_passed (total : ℕ) (failed : ℕ) : ℚ :=
  (total - failed : ℚ) / total * 100

/-- Theorem stating that the percentage of students who passed is 35% -/
theorem exam_pass_percentage :
  percentage_passed 540 351 = 35 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l2120_212065


namespace NUMINAMATH_CALUDE_complex_multiplication_l2120_212033

theorem complex_multiplication (i : ℂ) :
  i * i = -1 →
  (1 - i) * (2 + i) = 3 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2120_212033


namespace NUMINAMATH_CALUDE_test_subjects_count_l2120_212025

def number_of_colors : ℕ := 8
def colors_per_code : ℕ := 4
def unidentified_subjects : ℕ := 19

theorem test_subjects_count :
  (number_of_colors.choose colors_per_code) + unidentified_subjects = 299 :=
by sorry

end NUMINAMATH_CALUDE_test_subjects_count_l2120_212025


namespace NUMINAMATH_CALUDE_P_smallest_l2120_212010

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3
def is_perfect_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

def H : ℕ := sorry

axiom H_def : H > 0 ∧ 
  is_perfect_cube (H / 2) ∧ 
  is_perfect_fifth_power (H / 3) ∧ 
  is_perfect_square (H / 5)

axiom H_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_cube (n / 2) → 
  is_perfect_fifth_power (n / 3) → 
  is_perfect_square (n / 5) → 
  H ≤ n

def P : ℕ := sorry

axiom P_def : P > 0 ∧ 
  is_perfect_square (P / 2) ∧ 
  is_perfect_cube (P / 3) ∧ 
  is_perfect_fifth_power (P / 5)

axiom P_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_square (n / 2) → 
  is_perfect_cube (n / 3) → 
  is_perfect_fifth_power (n / 5) → 
  P ≤ n

def S : ℕ := sorry

axiom S_def : S > 0 ∧ 
  is_perfect_fifth_power (S / 2) ∧ 
  is_perfect_square (S / 3) ∧ 
  is_perfect_cube (S / 5)

axiom S_minimal : ∀ n : ℕ, n > 0 → 
  is_perfect_fifth_power (n / 2) → 
  is_perfect_square (n / 3) → 
  is_perfect_cube (n / 5) → 
  S ≤ n

theorem P_smallest : P < S ∧ P < H := by sorry

end NUMINAMATH_CALUDE_P_smallest_l2120_212010


namespace NUMINAMATH_CALUDE_parabola_point_order_l2120_212070

/-- A parabola with equation y = -x^2 + 2x + c -/
structure Parabola where
  c : ℝ

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = -p.x^2 + 2*p.x + para.c

theorem parabola_point_order (para : Parabola) (p1 p2 p3 : Point)
  (h1 : p1.x = 0) (h2 : p2.x = 1) (h3 : p3.x = 3)
  (h4 : lies_on p1 para) (h5 : lies_on p2 para) (h6 : lies_on p3 para) :
  p2.y > p1.y ∧ p1.y > p3.y := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l2120_212070


namespace NUMINAMATH_CALUDE_bee_count_l2120_212043

/-- The total number of bees in a hive after additional bees fly in -/
def total_bees (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 16 initial bees and 9 additional bees, the total is 25 -/
theorem bee_count : total_bees 16 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l2120_212043


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2120_212087

def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1}

theorem intersection_of_A_and_B : A ∩ B = {(0, 0), (1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2120_212087


namespace NUMINAMATH_CALUDE_attend_both_reunions_l2120_212036

/-- The number of people attending both reunions at the Taj Hotel -/
def both_reunions (total guests : ℕ) (oates hall : ℕ) : ℕ :=
  oates + hall - total

/-- Theorem stating the number of people attending both reunions -/
theorem attend_both_reunions : 
  both_reunions 150 70 52 = 28 := by sorry

end NUMINAMATH_CALUDE_attend_both_reunions_l2120_212036


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l2120_212080

-- Define the propositions p and q
def p (m : ℝ) : Prop := 1/4 < m ∧ m < 1

def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ m : ℝ, p m → q m) ∧ 
  (∃ m : ℝ, q m ∧ ¬p m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l2120_212080


namespace NUMINAMATH_CALUDE_ahn_max_number_l2120_212026

theorem ahn_max_number : ∃ (m : ℕ), m = 870 ∧ 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ahn_max_number_l2120_212026


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2120_212066

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 4 * z^2 - 5) * (4 * z^4 - 3 * z^2 + 2) =
  12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2120_212066


namespace NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l2120_212023

/-- Set A is defined as {x | x^2 + x - 6 = 0} -/
def A : Set ℝ := {x | x^2 + x - 6 = 0}

/-- Set B is defined as {x | x * m + 1 = 0} -/
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

/-- Theorem stating a sufficient condition for B to be a proper subset of A -/
theorem sufficient_condition_B_proper_subset_A :
  ∀ m : ℝ, m ∈ ({0, 1/3} : Set ℝ) → B m ⊂ A ∧ B m ≠ A :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l2120_212023


namespace NUMINAMATH_CALUDE_games_within_division_l2120_212090

/-- Represents a baseball league with specific game scheduling rules. -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  N : ℕ
  /-- Number of games played against each team in the other division -/
  M : ℕ
  /-- N is greater than twice M -/
  h1 : N > 2 * M
  /-- M is greater than 6 -/
  h2 : M > 6
  /-- Total number of games each team plays is 92 -/
  h3 : 3 * N + 4 * M = 92

/-- The number of games a team plays within its own division in the given baseball league is 60. -/
theorem games_within_division (league : BaseballLeague) : 3 * league.N = 60 := by
  sorry

end NUMINAMATH_CALUDE_games_within_division_l2120_212090


namespace NUMINAMATH_CALUDE_table_area_is_128_l2120_212094

/-- A rectangular table with one side against a wall -/
structure RectangularTable where
  -- Length of the side opposite the wall
  opposite_side : ℝ
  -- Length of each of the other two free sides
  other_side : ℝ
  -- The side opposite the wall is twice the length of each of the other two free sides
  opposite_twice_other : opposite_side = 2 * other_side
  -- The total length of the table's free sides is 32 feet
  total_free_sides : opposite_side + 2 * other_side = 32

/-- The area of the rectangular table is 128 square feet -/
theorem table_area_is_128 (table : RectangularTable) : table.opposite_side * table.other_side = 128 := by
  sorry

end NUMINAMATH_CALUDE_table_area_is_128_l2120_212094


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l2120_212017

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℝ) : 
  num_students = 10 → 
  student_avg_age = 15 → 
  teacher_age = 26 → 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l2120_212017


namespace NUMINAMATH_CALUDE_monthly_production_l2120_212099

/-- Represents the number of computers produced in a given time period -/
structure ComputerProduction where
  rate : ℝ  -- Computers produced per 30-minute interval
  days : ℕ  -- Number of days in the time period

/-- Calculates the total number of computers produced in the given time period -/
def totalComputers (prod : ComputerProduction) : ℝ :=
  (prod.rate * (prod.days * 24 * 2 : ℝ))

/-- Theorem stating that a factory producing 6.25 computers every 30 minutes
    for 28 days will produce 8400 computers -/
theorem monthly_production :
  totalComputers ⟨6.25, 28⟩ = 8400 := by sorry

end NUMINAMATH_CALUDE_monthly_production_l2120_212099


namespace NUMINAMATH_CALUDE_six_digit_numbers_at_least_two_zeros_l2120_212027

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def six_digit_numbers_no_zeros : ℕ := 531441

/-- The number of 6-digit numbers with exactly one zero -/
def six_digit_numbers_one_zero : ℕ := 295245

/-- Theorem: The number of 6-digit numbers with at least two zeros is 73,314 -/
theorem six_digit_numbers_at_least_two_zeros :
  total_six_digit_numbers - (six_digit_numbers_no_zeros + six_digit_numbers_one_zero) = 73314 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_at_least_two_zeros_l2120_212027


namespace NUMINAMATH_CALUDE_zaras_goats_l2120_212096

theorem zaras_goats (cows sheep : ℕ) (groups : ℕ) (animals_per_group : ℕ) (goats : ℕ) : 
  cows = 24 → 
  sheep = 7 → 
  groups = 3 → 
  animals_per_group = 48 → 
  goats = groups * animals_per_group - (cows + sheep) → 
  goats = 113 := by
  sorry

end NUMINAMATH_CALUDE_zaras_goats_l2120_212096


namespace NUMINAMATH_CALUDE_max_distance_and_squared_distance_coincide_l2120_212019

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the sum of distances from a point to the vertices
def sumDistances (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  distance p t.A + distance p t.B + distance p t.C

-- Define a function to calculate the sum of squared distances from a point to the vertices
def sumSquaredDistances (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  (distance p t.A)^2 + (distance p t.B)^2 + (distance p t.C)^2

-- Define a function to find the shortest side of a triangle
def shortestSide (t : Triangle) : ℝ := sorry

-- Define a function to find the vertex opposite the shortest side
def vertexOppositeShortestSide (t : Triangle) : ℝ × ℝ := sorry

theorem max_distance_and_squared_distance_coincide (t : Triangle) :
  ∃ p : ℝ × ℝ,
    (∀ q : ℝ × ℝ, sumDistances t q ≤ sumDistances t p) ∧
    (∀ q : ℝ × ℝ, sumSquaredDistances t q ≤ sumSquaredDistances t p) ∧
    p = vertexOppositeShortestSide t :=
  sorry


end NUMINAMATH_CALUDE_max_distance_and_squared_distance_coincide_l2120_212019


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2120_212009

/-- 
Given a loan with simple interest where:
- The principal amount is $1200
- The number of years equals the rate of interest
- The total interest paid is $432
Prove that the rate of interest is 6%
-/
theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432) :
  ∃ (rate : ℝ), rate = 6 ∧ interest_paid = principal * (rate / 100) * rate :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2120_212009


namespace NUMINAMATH_CALUDE_expression_value_l2120_212020

theorem expression_value (x y : ℝ) 
  (hx : (x - 15)^2 = 169) 
  (hy : (y - 1)^3 = -0.125) : 
  Real.sqrt x - Real.sqrt (2 * x * y) - (2 * y - x)^(1/3) = 3 ∨
  Real.sqrt x - Real.sqrt (2 * x * y) - (2 * y - x)^(1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2120_212020


namespace NUMINAMATH_CALUDE_similar_triangles_corresponding_side_length_l2120_212031

/-- Given two similar right triangles, where the first triangle has a leg of 15 inches and a hypotenuse
    of 17 inches, and the second triangle has a hypotenuse of 34 inches, the length of the side in the
    second triangle corresponding to the 15-inch leg is 30 inches. -/
theorem similar_triangles_corresponding_side_length (a b c d : ℝ) : 
  a = 15 →  -- First leg of the first triangle
  c = 17 →  -- Hypotenuse of the first triangle
  d = 34 →  -- Hypotenuse of the second triangle
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for the first triangle
  ∃ (k : ℝ), k > 0 ∧ d = k * c ∧ k * a = 30  -- The corresponding side in the second triangle is 30 inches
  := by sorry

end NUMINAMATH_CALUDE_similar_triangles_corresponding_side_length_l2120_212031


namespace NUMINAMATH_CALUDE_triangle_problem_l2120_212004

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the sine law
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

-- Define the cosine law
axiom cosine_law (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

theorem triangle_problem (t : Triangle) 
  (ha : t.a = 4)
  (hc : t.c = Real.sqrt 13)
  (hsin : Real.sin t.A = 4 * Real.sin t.B) :
  t.b = 1 ∧ Real.cos t.C = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2120_212004


namespace NUMINAMATH_CALUDE_triangle_vector_sum_l2120_212055

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the triangle ABC and vectors a and b
variable (A B C : V) (a b : V)

-- State the theorem
theorem triangle_vector_sum (h1 : B - C = a) (h2 : C - A = b) : 
  A - B = -a - b := by sorry

end NUMINAMATH_CALUDE_triangle_vector_sum_l2120_212055


namespace NUMINAMATH_CALUDE_circle_bisection_theorem_l2120_212014

/-- A circle in the 2D plane. -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the 2D plane. -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Predicate to check if a line bisects a circle. -/
def bisects (l : Line) (c : Circle) : Prop :=
  ∃ (x y : ℝ), c.equation x y ∧ l.equation x y

/-- The main theorem stating that if a specific circle is bisected by a specific line, then a = 1. -/
theorem circle_bisection_theorem (a : ℝ) : 
  let c : Circle := { equation := fun x y => x^2 + y^2 + 2*x - 4*y = 0 }
  let l : Line := { equation := fun x y => 3*x + y + a = 0 }
  bisects l c → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_bisection_theorem_l2120_212014


namespace NUMINAMATH_CALUDE_vacation_savings_l2120_212041

def total_income : ℝ := 72800
def total_expenses : ℝ := 54200
def deposit_rate : ℝ := 0.1

theorem vacation_savings : 
  (total_income - total_expenses) * (1 - deposit_rate) = 16740 := by
  sorry

end NUMINAMATH_CALUDE_vacation_savings_l2120_212041


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2120_212015

/-- The total bill for 9 friends dining at a restaurant -/
def total_bill : ℕ := 156

/-- The number of friends dining -/
def num_friends : ℕ := 9

/-- The amount Judi paid -/
def judi_payment : ℕ := 5

/-- The extra amount each remaining friend paid -/
def extra_payment : ℕ := 3

theorem restaurant_bill_proof :
  let regular_share := total_bill / num_friends
  let tom_payment := regular_share / 2
  let remaining_friends := num_friends - 2
  total_bill = 
    (remaining_friends * (regular_share + extra_payment)) + 
    judi_payment + 
    tom_payment :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2120_212015


namespace NUMINAMATH_CALUDE_f_properties_l2120_212018

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x * Real.cos x - 1/2

theorem f_properties :
  ∃ (k : ℤ), 
    (∀ x : ℝ, f x = f (π/4 - x)) ∧ 
    (∀ x : ℝ, (∃ k : ℤ, x ∈ Set.Icc (k * π - 3*π/8) (k * π + π/8)) → (deriv f) x > 0) ∧
    (f (π/2) = -1/2 ∧ ∀ x ∈ Set.Icc 0 (π/2), f x ≥ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2120_212018


namespace NUMINAMATH_CALUDE_counterexample_exists_l2120_212083

theorem counterexample_exists : ∃ n : ℕ, ¬ Nat.Prime n ∧ ¬ Nat.Prime (n - 4) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2120_212083


namespace NUMINAMATH_CALUDE_anika_maddie_age_ratio_l2120_212038

def anika_age : ℕ := 30
def future_years : ℕ := 15
def future_average_age : ℕ := 50

theorem anika_maddie_age_ratio :
  ∃ (maddie_age : ℕ),
    (anika_age + future_years + maddie_age + future_years) / 2 = future_average_age ∧
    anika_age * 4 = maddie_age * 3 := by
  sorry

end NUMINAMATH_CALUDE_anika_maddie_age_ratio_l2120_212038


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2120_212022

/-- Given that nine identical bowling balls weigh the same as four identical canoes,
    and one canoe weighs 36 pounds, prove that one bowling ball weighs 16 pounds. -/
theorem bowling_ball_weight (canoe_weight : ℕ) (ball_weight : ℚ)
  (h1 : canoe_weight = 36)
  (h2 : 9 * ball_weight = 4 * canoe_weight) :
  ball_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2120_212022


namespace NUMINAMATH_CALUDE_remainder_problem_l2120_212012

theorem remainder_problem (x y : ℤ) 
  (hx : x % 60 = 53)
  (hy : y % 45 = 28) :
  (3 * x - 2 * y) % 30 = 13 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2120_212012


namespace NUMINAMATH_CALUDE_union_complement_and_quadratic_set_l2120_212003

/-- Given sets S and T, prove that their union is equal to the set of all real numbers less than or equal to 1 -/
theorem union_complement_and_quadratic_set :
  let S : Set ℝ := {x | x > -2}
  let T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}
  (Sᶜ ∪ T) = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_union_complement_and_quadratic_set_l2120_212003


namespace NUMINAMATH_CALUDE_inequalities_proof_l2120_212002

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (a^3 * b > a * b^3) ∧ (a - b/a > b - a/b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2120_212002


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2120_212052

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2120_212052
