import Mathlib

namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l217_21766

/-- The probability of finding treasure without traps on an island -/
def p_treasure_only : ℚ := 1 / 5

/-- The probability of finding neither treasure nor traps on an island -/
def p_neither : ℚ := 3 / 5

/-- The number of islands -/
def n_islands : ℕ := 7

/-- The number of islands with treasure only -/
def n_treasure_only : ℕ := 3

/-- The number of islands with neither treasure nor traps -/
def n_neither : ℕ := 4

theorem pirate_treasure_probability : 
  (Nat.choose n_islands n_treasure_only : ℚ) * 
  (p_treasure_only ^ n_treasure_only) * 
  (p_neither ^ n_neither) = 81 / 2225 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l217_21766


namespace NUMINAMATH_CALUDE_expected_socks_is_2n_l217_21739

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ :=
  2 * n

/-- Theorem stating that the expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_is_2n (n : ℕ) (h : n > 0) :
  expected_socks n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_expected_socks_is_2n_l217_21739


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_difference_1004_1001_l217_21799

/-- Given an arithmetic sequence with first term 3 and common difference 7,
    the positive difference between the 1004th term and the 1001st term is 21. -/
theorem arithmetic_sequence_difference : ℕ → ℕ :=
  fun n => 3 + (n - 1) * 7

#check arithmetic_sequence_difference 1004 - arithmetic_sequence_difference 1001 = 21

theorem difference_1004_1001 :
  arithmetic_sequence_difference 1004 - arithmetic_sequence_difference 1001 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_difference_1004_1001_l217_21799


namespace NUMINAMATH_CALUDE_sqrt_6_over_3_properties_l217_21725

theorem sqrt_6_over_3_properties : ∃ x : ℝ, x = (Real.sqrt 6) / 3 ∧ 0 < x ∧ x < 1 ∧ Irrational x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_over_3_properties_l217_21725


namespace NUMINAMATH_CALUDE_tangent_at_one_tangent_through_point_l217_21751

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 2

-- Theorem for the tangent line at x = 1
theorem tangent_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 
    (x = 1 ∧ y = f 1) ∨ 
    (y - f 1 = (3 * 1^2 - 2 * 1 + 1) * (x - 1)) :=
sorry

-- Theorem for the tangent lines passing through (1,3)
theorem tangent_through_point :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = m₁*x + b₁ → (∃ t, f t = y ∧ 3*t^2 - 2*t + 1 = m₁ ∧ x = t)) ∧
    (∀ x y, y = m₂*x + b₂ → (∃ t, f t = y ∧ 3*t^2 - 2*t + 1 = m₂ ∧ x = t)) ∧
    m₁ = 1 ∧ b₁ = 2 ∧ m₂ = 2 ∧ b₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_at_one_tangent_through_point_l217_21751


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l217_21755

/-- Proves that the daily wage is 25 given the contract conditions --/
theorem contractor_daily_wage
  (total_days : ℕ)
  (absent_days : ℕ)
  (daily_fine : ℚ)
  (total_payment : ℚ)
  (h1 : total_days = 30)
  (h2 : absent_days = 6)
  (h3 : daily_fine = 7.5)
  (h4 : total_payment = 555)
  : ∃ (daily_wage : ℚ),
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_payment ∧
    daily_wage = 25 := by
  sorry


end NUMINAMATH_CALUDE_contractor_daily_wage_l217_21755


namespace NUMINAMATH_CALUDE_truck_distance_l217_21785

theorem truck_distance (distance : ℝ) (time_minutes : ℝ) (travel_time_hours : ℝ) : 
  distance = 2 ∧ time_minutes = 2.5 ∧ travel_time_hours = 3 →
  (distance / time_minutes) * (travel_time_hours * 60) = 144 := by
sorry

end NUMINAMATH_CALUDE_truck_distance_l217_21785


namespace NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l217_21706

theorem polygon_with_900_degree_sum_is_heptagon :
  ∀ n : ℕ, n ≥ 3 → (n - 2) * 180 = 900 → n = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_900_degree_sum_is_heptagon_l217_21706


namespace NUMINAMATH_CALUDE_proposition_is_true_l217_21723

theorem proposition_is_true : ∀ (x y : ℝ), x + 2*y ≠ 5 → x ≠ 1 ∨ y ≠ 2 := by sorry

end NUMINAMATH_CALUDE_proposition_is_true_l217_21723


namespace NUMINAMATH_CALUDE_expand_and_simplify_l217_21727

theorem expand_and_simplify (x : ℝ) : (x + 3) * (x - 4) = x^2 - x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l217_21727


namespace NUMINAMATH_CALUDE_ceiling_distance_to_square_existence_l217_21744

theorem ceiling_distance_to_square_existence : 
  ∃ (A : ℝ), ∀ (n : ℕ), 
    ∃ (m : ℕ), (⌈A^n⌉ : ℝ) - (m^2 : ℝ) = 2 ∧ 
    ∀ (k : ℕ), k > m → (k^2 : ℝ) > ⌈A^n⌉ :=
by sorry

end NUMINAMATH_CALUDE_ceiling_distance_to_square_existence_l217_21744


namespace NUMINAMATH_CALUDE_telescope_payment_difference_l217_21701

theorem telescope_payment_difference (joan_payment karl_payment : ℕ) : 
  joan_payment = 158 →
  joan_payment + karl_payment = 400 →
  2 * joan_payment - karl_payment = 74 := by
sorry

end NUMINAMATH_CALUDE_telescope_payment_difference_l217_21701


namespace NUMINAMATH_CALUDE_modulus_of_z_l217_21738

open Complex

theorem modulus_of_z (z : ℂ) (h : z * (1 + I) = I) : abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l217_21738


namespace NUMINAMATH_CALUDE_inequality_proof_l217_21721

theorem inequality_proof : -2 < (-1)^3 ∧ (-1)^3 < (-0.6)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l217_21721


namespace NUMINAMATH_CALUDE_initial_diaries_count_l217_21747

theorem initial_diaries_count (initial : ℕ) : 
  (2 * initial - (2 * initial) / 4 = 18) → initial = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_diaries_count_l217_21747


namespace NUMINAMATH_CALUDE_smallest_integer_with_two_cube_sum_representations_l217_21788

def is_sum_of_three_cubes (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a^3 + b^3 + c^3

def has_two_representations (n : ℕ) : Prop :=
  ∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℕ,
    a₁ > 0 ∧ b₁ > 0 ∧ c₁ > 0 ∧
    a₂ > 0 ∧ b₂ > 0 ∧ c₂ > 0 ∧
    n = a₁^3 + b₁^3 + c₁^3 ∧
    n = a₂^3 + b₂^3 + c₂^3 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)

theorem smallest_integer_with_two_cube_sum_representations :
  (has_two_representations 251) ∧
  (∀ m : ℕ, m < 251 → ¬(has_two_representations m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_two_cube_sum_representations_l217_21788


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l217_21763

/-- Calculates the sampling interval for systematic sampling -/
def sampling_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

theorem systematic_sampling_interval :
  sampling_interval 630 45 = 14 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l217_21763


namespace NUMINAMATH_CALUDE_discount_difference_l217_21786

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  bill = 15000 ∧ 
  single_discount = 0.3 ∧ 
  first_discount = 0.25 ∧ 
  second_discount = 0.05 →
  bill * (1 - first_discount) * (1 - second_discount) - bill * (1 - single_discount) = 187.5 := by
sorry

end NUMINAMATH_CALUDE_discount_difference_l217_21786


namespace NUMINAMATH_CALUDE_consecutive_base_problem_l217_21750

/-- Given two consecutive positive integers X and Y, 
    if 312 in base X minus 65 in base Y equals 97 in base (X+Y), 
    then X+Y equals 7 -/
theorem consecutive_base_problem (X Y : ℕ) : 
  X > 0 → Y > 0 → Y = X + 1 → 
  (3 * X^2 + X + 2) - (6 * Y + 5) = 9 * (X + Y) + 7 → 
  X + Y = 7 := by sorry

end NUMINAMATH_CALUDE_consecutive_base_problem_l217_21750


namespace NUMINAMATH_CALUDE_candy_division_l217_21732

theorem candy_division (total_candy : ℕ) (num_students : ℕ) (candy_per_student : ℕ) 
  (h1 : total_candy = 344) 
  (h2 : num_students = 43) 
  (h3 : candy_per_student = total_candy / num_students) :
  candy_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l217_21732


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l217_21704

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2*a - 1)*x + a + 1

-- Define monotonicity in an interval
def monotonic_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ ∀ z, a < z ∧ z < b → f z = f x)

-- State the theorem
theorem quadratic_monotonicity (a : ℝ) :
  monotonic_in (f a) 1 2 → (a ≥ 5/2 ∨ a ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l217_21704


namespace NUMINAMATH_CALUDE_seating_arrangements_l217_21741

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

theorem seating_arrangements (n : ℕ) (h : n = 8) : 
  total_arrangements n - adjacent_arrangements n = 30240 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l217_21741


namespace NUMINAMATH_CALUDE_intersection_and_complement_l217_21797

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x)}

-- State the theorem
theorem intersection_and_complement :
  (M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1}) ∧
  (Nᶜ = {x : ℝ | x > 1}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_l217_21797


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_three_l217_21793

theorem negative_one_greater_than_negative_sqrt_three : -1 > -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_three_l217_21793


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l217_21790

/-- The average speed of a round trip, given the outbound speed and the fact that the return journey takes twice as long -/
theorem round_trip_average_speed (outbound_speed : ℝ) 
  (h1 : outbound_speed = 54) 
  (h2 : return_time = 2 * outbound_time) : 
  average_speed = 36 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l217_21790


namespace NUMINAMATH_CALUDE_three_workers_completion_time_l217_21789

/-- The time taken for three workers to complete a task together, given their individual completion times -/
theorem three_workers_completion_time 
  (x_time y_time z_time : ℝ) 
  (hx : x_time = 30) 
  (hy : y_time = 45) 
  (hz : z_time = 60) : 
  (1 / x_time + 1 / y_time + 1 / z_time)⁻¹ = 180 / 13 := by
  sorry

#check three_workers_completion_time

end NUMINAMATH_CALUDE_three_workers_completion_time_l217_21789


namespace NUMINAMATH_CALUDE_max_xy_value_max_xy_achieved_l217_21769

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

theorem max_xy_achieved : ∃ (x y : ℕ+), 7 * x + 4 * y = 140 ∧ x * y = 168 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_max_xy_achieved_l217_21769


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l217_21794

theorem sqrt_product_equals_sqrt_of_product : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l217_21794


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l217_21713

/-- Given a constant distance and two different walking rates, where one rate
    results in a 14-minute journey and the other in a 12-minute journey,
    prove that the ratio of the faster rate to the slower rate is 7/6. -/
theorem walking_rate_ratio (distance : ℝ) (usual_rate new_rate : ℝ) :
  distance > 0 →
  usual_rate > 0 →
  new_rate > 0 →
  distance = usual_rate * 14 →
  distance = new_rate * 12 →
  new_rate / usual_rate = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l217_21713


namespace NUMINAMATH_CALUDE_count_integers_with_three_digits_under_50000_l217_21787

/-- A function that counts the number of positive integers less than n with at most k different digits. -/
def count_integers_with_limited_digits (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the count of positive integers less than 50,000 with at most three different digits is 7862. -/
theorem count_integers_with_three_digits_under_50000 :
  count_integers_with_limited_digits 50000 3 = 7862 :=
sorry

end NUMINAMATH_CALUDE_count_integers_with_three_digits_under_50000_l217_21787


namespace NUMINAMATH_CALUDE_two_integers_sum_l217_21709

theorem two_integers_sum (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ x - y = 4 ∧ x * y = 156 → x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l217_21709


namespace NUMINAMATH_CALUDE_sector_central_angle_l217_21777

/-- Given a sector with radius 1 and arc length 2, its central angle is 2 radians -/
theorem sector_central_angle (radius : ℝ) (arc_length : ℝ) (h1 : radius = 1) (h2 : arc_length = 2) :
  arc_length / radius = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l217_21777


namespace NUMINAMATH_CALUDE_berry_picking_difference_l217_21742

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  sergey_basket_ratio : ℚ
  dima_basket_ratio : ℚ
  sergey_speed_multiplier : ℕ

/-- The main theorem about the berry picking scenario -/
theorem berry_picking_difference (scenario : BerryPicking) 
  (h1 : scenario.total_berries = 900)
  (h2 : scenario.sergey_basket_ratio = 1/2)
  (h3 : scenario.dima_basket_ratio = 2/3)
  (h4 : scenario.sergey_speed_multiplier = 2) :
  ∃ (sergey_basket dima_basket : ℕ), 
    sergey_basket = 300 ∧ 
    dima_basket = 200 ∧ 
    sergey_basket - dima_basket = 100 := by
  sorry

#check berry_picking_difference

end NUMINAMATH_CALUDE_berry_picking_difference_l217_21742


namespace NUMINAMATH_CALUDE_triangle_area_l217_21749

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l217_21749


namespace NUMINAMATH_CALUDE_equation_one_solution_l217_21792

theorem equation_one_solution (m : ℝ) : 
  (∃! x : ℝ, (3*x+4)*(x-8) = -50 + m*x) ↔ 
  (m = -20 + 6*Real.sqrt 6 ∨ m = -20 - 6*Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_l217_21792


namespace NUMINAMATH_CALUDE_texas_integrated_school_student_count_l217_21708

theorem texas_integrated_school_student_count 
  (initial_classes : ℕ) 
  (students_per_class : ℕ) 
  (additional_classes : ℕ) : 
  initial_classes = 15 → 
  students_per_class = 20 → 
  additional_classes = 5 → 
  (initial_classes + additional_classes) * students_per_class = 400 := by
sorry

end NUMINAMATH_CALUDE_texas_integrated_school_student_count_l217_21708


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l217_21733

def first_caterer_cost (n : ℕ) : ℝ := 50 + 18 * n

def second_caterer_cost (n : ℕ) : ℝ :=
  if n ≥ 30 then 150 + 15 * n else 180 + 15 * n

theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n ≥ 34 → second_caterer_cost n < first_caterer_cost n) ∧
  (∀ n : ℕ, n < 34 → second_caterer_cost n ≥ first_caterer_cost n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l217_21733


namespace NUMINAMATH_CALUDE_inequality_solution_set_l217_21717

theorem inequality_solution_set (x : ℝ) :
  (|2*x - 2| + |2*x + 4| < 10) ↔ (x > -4 ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l217_21717


namespace NUMINAMATH_CALUDE_first_valid_year_is_2015_l217_21752

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ sum_of_digits year = 8

theorem first_valid_year_is_2015 : 
  ∀ year : ℕ, is_valid_year year → year ≥ 2015 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2015_l217_21752


namespace NUMINAMATH_CALUDE_highway_vehicles_l217_21715

theorem highway_vehicles (total : ℕ) (trucks : ℕ) (cars : ℕ) 
  (h1 : total = 300)
  (h2 : cars = 2 * trucks)
  (h3 : total = cars + trucks) :
  trucks = 100 := by
  sorry

end NUMINAMATH_CALUDE_highway_vehicles_l217_21715


namespace NUMINAMATH_CALUDE_kantana_chocolates_l217_21710

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def chocolates_for_self : ℕ := sorry

/-- Represents the number of Saturdays in the month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates bought for Charlie's birthday -/
def chocolates_for_charlie : ℕ := 10

/-- Represents the total number of chocolates bought in the month -/
def total_chocolates : ℕ := 22

/-- Theorem stating that Kantana buys 2 chocolates for herself each Saturday -/
theorem kantana_chocolates : 
  chocolates_for_self = 2 ∧ 
  (chocolates_for_self + 1) * saturdays_in_month + chocolates_for_charlie = total_chocolates :=
by sorry

end NUMINAMATH_CALUDE_kantana_chocolates_l217_21710


namespace NUMINAMATH_CALUDE_four_color_theorem_l217_21771

/-- A type representing the four colors used for edge coloring -/
inductive EdgeColor
| one
| two
| three
| four

/-- A graph with edges colored using four colors -/
structure ColoredGraph (α : Type*) where
  edges : α → α → Option EdgeColor
  edge_coloring_property : ∀ (a b c : α), 
    edges a b ≠ none → edges b c ≠ none → 
    ∀ (d : α), edges c d ≠ none → 
    edges a b ≠ edges c d

/-- A type representing the four colors used for vertex coloring -/
inductive VertexColor
| one
| two
| three
| four

/-- A proper vertex coloring of a graph -/
def ProperVertexColoring (G : ColoredGraph α) (f : α → VertexColor) :=
  ∀ (a b : α), G.edges a b ≠ none → f a ≠ f b

theorem four_color_theorem (α : Type*) (G : ColoredGraph α) :
  ∃ (f : α → VertexColor), ProperVertexColoring G f :=
sorry

end NUMINAMATH_CALUDE_four_color_theorem_l217_21771


namespace NUMINAMATH_CALUDE_power_product_equality_l217_21726

theorem power_product_equality : (3^5 * 4^5) = 248832 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l217_21726


namespace NUMINAMATH_CALUDE_arithmetic_computation_l217_21737

theorem arithmetic_computation : -(12 * 2) - (3 * 2) + (-18 / 3 * -4) = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l217_21737


namespace NUMINAMATH_CALUDE_weeks_to_save_l217_21736

def console_cost : ℕ := 282
def initial_savings : ℕ := 42
def weekly_allowance : ℕ := 24

theorem weeks_to_save : 
  (console_cost - initial_savings) / weekly_allowance = 10 :=
sorry

end NUMINAMATH_CALUDE_weeks_to_save_l217_21736


namespace NUMINAMATH_CALUDE_train_length_l217_21722

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 45 * 1000 / 3600 →
  time = 30 →
  bridge_length = 215 →
  speed * time - bridge_length = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l217_21722


namespace NUMINAMATH_CALUDE_function_extension_theorem_l217_21729

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem function_extension_theorem (f : ℝ → ℝ) 
  (h1 : is_even_function (fun x ↦ f (x + 2)))
  (h2 : ∀ x : ℝ, x ≥ 2 → f x = x^2 - 6*x + 4) :
  ∀ x : ℝ, x < 2 → f x = x^2 - 2*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_function_extension_theorem_l217_21729


namespace NUMINAMATH_CALUDE_expansion_without_x3_x2_implies_m_plus_n_eq_neg_4_l217_21707

theorem expansion_without_x3_x2_implies_m_plus_n_eq_neg_4 
  (m n : ℝ) 
  (h1 : (1 + m) = 0)
  (h2 : (-3*m + n) = 0) :
  m + n = -4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_without_x3_x2_implies_m_plus_n_eq_neg_4_l217_21707


namespace NUMINAMATH_CALUDE_gcd_45736_123456_l217_21758

theorem gcd_45736_123456 : Nat.gcd 45736 123456 = 352 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45736_123456_l217_21758


namespace NUMINAMATH_CALUDE_reflection_result_l217_21778

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -(p.2))

/-- The original point F -/
def F : ℝ × ℝ := (3, -3)

theorem reflection_result :
  (reflect_x (reflect_y F)) = (-3, 3) := by sorry

end NUMINAMATH_CALUDE_reflection_result_l217_21778


namespace NUMINAMATH_CALUDE_probability_geometric_progression_ratio_two_l217_21728

/-- A fair die with 6 faces -/
def FairDie : Type := Fin 6

/-- The outcome of rolling four fair dice -/
def FourDiceRoll : Type := FairDie × FairDie × FairDie × FairDie

/-- The total number of possible outcomes when rolling four fair dice -/
def totalOutcomes : ℕ := 6^4

/-- Checks if a list of four numbers forms a geometric progression with a common ratio of two -/
def isGeometricProgressionWithRatioTwo (roll : List ℕ) : Prop :=
  roll.length = 4 ∧ ∃ a : ℕ, roll = [a, 2*a, 4*a, 8*a]

/-- The number of favorable outcomes (rolls that can be arranged to form a geometric progression with ratio two) -/
def favorableOutcomes : ℕ := 36

/-- The probability of rolling four dice such that the numbers can be arranged 
    to form a geometric progression with a common ratio of two -/
theorem probability_geometric_progression_ratio_two :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 36 := by
  sorry


end NUMINAMATH_CALUDE_probability_geometric_progression_ratio_two_l217_21728


namespace NUMINAMATH_CALUDE_oak_trees_remaining_l217_21720

theorem oak_trees_remaining (initial_trees cut_trees : ℕ) : 
  initial_trees = 9 → cut_trees = 2 → initial_trees - cut_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_remaining_l217_21720


namespace NUMINAMATH_CALUDE_parallelogram_bisector_slope_l217_21746

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- Predicate to check if a line cuts a parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem parallelogram_bisector_slope (p : Parallelogram) (l : Line) :
  p.v1 = (5, 20) →
  p.v2 = (5, 50) →
  p.v3 = (20, 100) →
  p.v4 = (20, 70) →
  cuts_into_congruent_polygons p l →
  l.slope = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_slope_l217_21746


namespace NUMINAMATH_CALUDE_problem_solution_l217_21776

theorem problem_solution (a b : ℚ) 
  (eq1 : 3020 * a + 3026 * b = 3030)
  (eq2 : 3024 * a + 3028 * b = 3034) :
  a - 2 * b = -1509 / 1516 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l217_21776


namespace NUMINAMATH_CALUDE_rectangle_area_l217_21768

/-- The area of a rectangle containing two smaller squares and one larger square -/
theorem rectangle_area (small_square_area : ℝ) (small_square_side : ℝ) (large_square_side : ℝ) :
  small_square_area = small_square_side ^ 2 →
  small_square_area = 4 →
  large_square_side = 3 * small_square_side →
  2 * small_square_area + large_square_side ^ 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l217_21768


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_l217_21772

theorem mod_equivalence_unique : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -150 ≡ n [ZMOD 23] ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_l217_21772


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l217_21703

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 9) 
  (h2 : a 5 = 33) : 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d ∧ d = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l217_21703


namespace NUMINAMATH_CALUDE_stack_height_problem_l217_21716

/-- Calculates the total height of a stack of discs with a cylindrical item on top -/
def total_height (top_diameter : ℕ) (bottom_diameter : ℕ) (disc_thickness : ℕ) (cylinder_height : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  let discs_height := num_discs * disc_thickness
  discs_height + cylinder_height

/-- The problem statement -/
theorem stack_height_problem :
  let top_diameter := 15
  let bottom_diameter := 1
  let disc_thickness := 2
  let cylinder_height := 10
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = 26 := by
  sorry

end NUMINAMATH_CALUDE_stack_height_problem_l217_21716


namespace NUMINAMATH_CALUDE_proposition_evaluation_l217_21761

theorem proposition_evaluation (a b : ℝ) : 
  (¬ (∀ a, a < 2 → a^2 < 4)) ∧ 
  (∀ a, a^2 < 4 → a < 2) ∧ 
  (∀ a, a ≥ 2 → a^2 ≥ 4) ∧ 
  (¬ (∀ a, a^2 ≥ 4 → a ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l217_21761


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l217_21782

/-- Given that the solution set of ax^2 + 5x - 2 > 0 is {x | 1/2 < x < 2}, 
    prove the value of a and the solution set of ax^2 - 5x + a^2 - 1 > 0 -/
theorem quadratic_inequality_problem 
  (h : ∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) :
  (a = -2) ∧ 
  (∀ x : ℝ, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l217_21782


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l217_21748

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (2*x + y)⁻¹ + 4*(2*x + 3*y)⁻¹ = 1) :
  x + y ≥ 9/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    (2*x₀ + y₀)⁻¹ + 4*(2*x₀ + 3*y₀)⁻¹ = 1 ∧ x₀ + y₀ = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l217_21748


namespace NUMINAMATH_CALUDE_matrix_equality_l217_21773

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : A * B + !![2, 0; 0, 2] = A + B)
  (h2 : A * B = !![38/3, 4/3; -8/3, 4/3]) :
  B * A = !![44/3, 4/3; -8/3, 10/3] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l217_21773


namespace NUMINAMATH_CALUDE_constant_b_equals_negative_two_l217_21784

/-- Given a polynomial equation, prove that the constant b must equal -2. -/
theorem constant_b_equals_negative_two :
  ∀ (a c : ℝ) (b : ℝ),
  (fun x : ℝ => (4 * x^3 - 2 * x + 5/2) * (a * x^3 + b * x^2 + c)) =
  (fun x : ℝ => 20 * x^6 - 8 * x^4 + 15 * x^3 - 5 * x^2 + 5) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_constant_b_equals_negative_two_l217_21784


namespace NUMINAMATH_CALUDE_container_volume_ratio_l217_21770

theorem container_volume_ratio : 
  ∀ (A B : ℝ), A > 0 → B > 0 → 
  (4/5 * A = 2/3 * B) → 
  (A / B = 5/6) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l217_21770


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l217_21753

/-- A regular polygon with interior angles of 144° has a sum of interior angles equal to 1440°. -/
theorem regular_polygon_interior_angle_sum (n : ℕ) (h : n ≥ 3) :
  let interior_angle : ℝ := 144
  n * interior_angle = (n - 2) * 180 ∧ n * interior_angle = 1440 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l217_21753


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l217_21745

/-- Given a circle with equation (x-1)^2+(y+1)^2=4, its symmetric circle with respect to the origin has the equation (x+1)^2+(y-1)^2=4 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ x y, (x - 1)^2 + (y + 1)^2 = 4) →
  (∀ x y, (x + 1)^2 + (y - 1)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l217_21745


namespace NUMINAMATH_CALUDE_parabola_directrix_l217_21774

/-- The directrix of the parabola y = x^2 -/
theorem parabola_directrix : ∃ (k : ℝ), ∀ (x y : ℝ),
  y = x^2 → (4 * y + 1 = 0 ↔ y = k) := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l217_21774


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l217_21714

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 9 ∧ 
  ∃ m : ℕ, p + 8 = m^2 ∧ 
  m = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l217_21714


namespace NUMINAMATH_CALUDE_population_1988_l217_21700

/-- The population growth factor for a 4-year period -/
def growth_factor : ℝ := 2

/-- The number of 4-year periods between 1988 and 2008 -/
def num_periods : ℕ := 5

/-- The population of Arloe in 2008 -/
def population_2008 : ℕ := 3456

/-- The population growth function -/
def population (initial : ℕ) (periods : ℕ) : ℝ :=
  initial * growth_factor ^ periods

theorem population_1988 :
  ∃ p : ℕ, population p num_periods = population_2008 ∧ p = 108 := by
  sorry

end NUMINAMATH_CALUDE_population_1988_l217_21700


namespace NUMINAMATH_CALUDE_exists_universal_transport_l217_21724

/-- A graph where each pair of vertices is connected by exactly one edge of either type A or type B -/
structure TransportGraph (V : Type) :=
  (edges : V → V → Bool)
  (edge_type : V → V → Bool)
  (connect : ∀ (u v : V), u ≠ v → edges u v = true)
  (unique : ∀ (u v : V), edges u v = edges v u)

/-- A path in the graph with at most two intermediate vertices -/
def ShortPath {V : Type} (g : TransportGraph V) (t : Bool) (u v : V) : Prop :=
  ∃ (w x : V), (g.edges u w ∧ g.edge_type u w = t) ∧ 
               (g.edges w x ∧ g.edge_type w x = t) ∧ 
               (g.edges x v ∧ g.edge_type x v = t)

/-- Main theorem: There exists a transport type that allows short paths between all vertices -/
theorem exists_universal_transport {V : Type} (g : TransportGraph V) :
  ∃ (t : Bool), ∀ (u v : V), u ≠ v → ShortPath g t u v :=
sorry

end NUMINAMATH_CALUDE_exists_universal_transport_l217_21724


namespace NUMINAMATH_CALUDE_point_outside_circle_l217_21702

theorem point_outside_circle (a b : ℝ) (i : ℂ) : 
  i * i = -1 → 
  Complex.I = i →
  a + b * i = (2 + i) / (1 - i) → 
  a^2 + b^2 > 2 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l217_21702


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_le_two_l217_21759

/-- A quadratic function f(x) = -x² - 2(a-1)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 - 2*(a-1)*x + 5

/-- The theorem states that if f(x) is decreasing on [-1, +∞), then a ≤ 2 -/
theorem decreasing_quadratic_implies_a_le_two (a : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ → f a x₂ < f a x₁) →
  a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_le_two_l217_21759


namespace NUMINAMATH_CALUDE_unique_solution_linear_equation_l217_21712

theorem unique_solution_linear_equation (a b : ℝ) (ha : a ≠ 0) :
  ∃! x : ℝ, a * x = b ∧ x = b / a := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_equation_l217_21712


namespace NUMINAMATH_CALUDE_abc_inequality_l217_21767

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a^(1/7) + b^(1/7) + c^(1/7)) :
  a^a * b^b * c^c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l217_21767


namespace NUMINAMATH_CALUDE_square_plus_cube_equals_one_l217_21711

theorem square_plus_cube_equals_one : 3^2 + (-2)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_cube_equals_one_l217_21711


namespace NUMINAMATH_CALUDE_polynomial_composition_factorization_l217_21796

theorem polynomial_composition_factorization :
  ∀ (p : Polynomial ℤ),
  (Polynomial.degree p ≥ 1) →
  ∃ (q f g : Polynomial ℤ),
    (Polynomial.degree f ≥ 1) ∧
    (Polynomial.degree g ≥ 1) ∧
    (p.comp q = f * g) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_composition_factorization_l217_21796


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l217_21765

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum1 : a 1 + a 2 = 4) 
  (h_sum2 : a 3 + a 4 = 16) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n, a (n + 1) = a n + d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l217_21765


namespace NUMINAMATH_CALUDE_forty_fifth_term_is_91_l217_21740

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- The nth term of an arithmetic sequence. -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

/-- Theorem: In an arithmetic sequence where the first term is 3 and the fifteenth term is 31,
    the forty-fifth term is 91. -/
theorem forty_fifth_term_is_91 :
  ∀ seq : ArithmeticSequence,
  seq.first_term = 3 →
  nth_term seq 15 = 31 →
  nth_term seq 45 = 91 := by
sorry

end NUMINAMATH_CALUDE_forty_fifth_term_is_91_l217_21740


namespace NUMINAMATH_CALUDE_n_has_four_digits_l217_21731

def n : ℕ := 9376

theorem n_has_four_digits :
  (∃ k : ℕ, n^2 % 10000 = n) →
  (∃ m : ℕ, 10^3 ≤ n ∧ n < 10^4) :=
by sorry

end NUMINAMATH_CALUDE_n_has_four_digits_l217_21731


namespace NUMINAMATH_CALUDE_interval_preserving_linear_l217_21735

-- Define the property that f maps intervals to intervals of the same length
def IntervalPreserving (f : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → ∃ c d, c < d ∧ f '' Set.Icc a b = Set.Icc c d ∧ d - c = b - a

-- State the theorem
theorem interval_preserving_linear (f : ℝ → ℝ) (h : IntervalPreserving f) :
  ∃ c : ℝ, (∀ x, f x = x + c) ∨ (∀ x, f x = -x + c) :=
sorry

end NUMINAMATH_CALUDE_interval_preserving_linear_l217_21735


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l217_21791

theorem greatest_divisor_with_remainders : 
  ∃ (d : ℕ), d > 0 ∧ 
    (∃ (q₁ q₂ q₃ : ℕ), 2674 = d * q₁ + 5 ∧ 3486 = d * q₂ + 7 ∧ 4328 = d * q₃ + 9) ∧
    ∀ (k : ℕ), k > 0 → 
      (∃ (r₁ r₂ r₃ : ℕ), 2674 = k * r₁ + 5 ∧ 3486 = k * r₂ + 7 ∧ 4328 = k * r₃ + 9) →
      k ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l217_21791


namespace NUMINAMATH_CALUDE_ellipse_left_vertex_l217_21783

-- Define the ellipse
def is_ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle
def is_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8 = 0

-- Theorem statement
theorem ellipse_left_vertex 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), is_circle x y ∧ is_ellipse a b (x - 3) y) 
  (h4 : 2 * b = 8) :
  is_ellipse a b (-5) 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_left_vertex_l217_21783


namespace NUMINAMATH_CALUDE_ladybugs_per_leaf_l217_21756

theorem ladybugs_per_leaf (total_leaves : ℕ) (total_ladybugs : ℕ) (ladybugs_per_leaf : ℕ) : 
  total_leaves = 84 → 
  total_ladybugs = 11676 → 
  total_ladybugs = total_leaves * ladybugs_per_leaf → 
  ladybugs_per_leaf = 139 := by
sorry

end NUMINAMATH_CALUDE_ladybugs_per_leaf_l217_21756


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l217_21734

/-- Given an inverse proportion function y = (k-1)x^(k^2-5) where k is a constant,
    if y decreases as x increases when x > 0, then k = 2 -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y = (k - 1) * x^(k^2 - 5)) →  -- y is a function of x
  (∀ x1 x2 y1 y2 : ℝ, x1 > 0 → x2 > 0 → x1 < x2 → 
    y1 = (k - 1) * x1^(k^2 - 5) → y2 = (k - 1) * x2^(k^2 - 5) → y1 > y2) →  -- y decreases as x increases
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l217_21734


namespace NUMINAMATH_CALUDE_suresh_work_hours_l217_21781

theorem suresh_work_hours (suresh_rate ashutosh_rate : ℚ) 
  (ashutosh_remaining_time : ℚ) : 
  suresh_rate = 1 / 15 →
  ashutosh_rate = 1 / 25 →
  ashutosh_remaining_time = 10 →
  ∃ (suresh_time : ℚ), 
    suresh_time * suresh_rate + ashutosh_remaining_time * ashutosh_rate = 1 ∧
    suresh_time = 9 := by
sorry

end NUMINAMATH_CALUDE_suresh_work_hours_l217_21781


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l217_21779

def g (x : ℝ) : ℝ := (x + 3)^2 - 10

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y : ℝ, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l217_21779


namespace NUMINAMATH_CALUDE_no_real_solutions_l217_21757

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y + 2 * z^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l217_21757


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l217_21780

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  x + y = -b / a :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 2003 * x - 2004
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  x + y = 2003 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l217_21780


namespace NUMINAMATH_CALUDE_candy_problem_l217_21764

theorem candy_problem (C : ℕ) : 
  (C - (C / 3) - (C - (C / 3)) / 4 + C / 2 - 7 = 16) → C = 23 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l217_21764


namespace NUMINAMATH_CALUDE_free_throw_probability_convergence_l217_21798

/-- Represents the number of successful shots for a given total number of shots -/
def makes : ℕ → ℕ
| 50 => 28
| 100 => 49
| 150 => 78
| 200 => 102
| 300 => 153
| 400 => 208
| 500 => 255
| _ => 0  -- For any other number of shots, we don't have data

/-- Represents the total number of shots taken -/
def shots : List ℕ := [50, 100, 150, 200, 300, 400, 500]

/-- Calculate the make frequency for a given number of shots -/
def makeFrequency (n : ℕ) : ℚ :=
  (makes n : ℚ) / n

/-- The statement to be proved -/
theorem free_throw_probability_convergence :
  ∀ ε > 0, ∃ N, ∀ n ∈ shots, n ≥ N → |makeFrequency n - 51/100| < ε :=
sorry

end NUMINAMATH_CALUDE_free_throw_probability_convergence_l217_21798


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l217_21754

theorem fourth_root_equation_solutions :
  let f (x : ℝ) := (Real.sqrt (Real.sqrt (43 - 2*x))) + (Real.sqrt (Real.sqrt (39 + 2*x)))
  ∃ (S : Set ℝ), S = {x | f x = 4} ∧ S = {21, -13.5} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l217_21754


namespace NUMINAMATH_CALUDE_quadratic_functions_property_l217_21775

/-- Two quadratic functions with specific properties -/
theorem quadratic_functions_property (h j k : ℝ) : 
  (∃ (a b c d : ℕ), a ≠ b ∧ c ≠ d ∧ 
    3 * (a - h)^2 + j = 0 ∧ 
    3 * (b - h)^2 + j = 0 ∧
    2 * (c - h)^2 + k = 0 ∧ 
    2 * (d - h)^2 + k = 0) →
  (3 * h^2 + j = 2013 ∧ 2 * h^2 + k = 2014) →
  h = 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_functions_property_l217_21775


namespace NUMINAMATH_CALUDE_alcohol_quantity_l217_21719

-- Define the initial ratio of alcohol to water
def initial_ratio : ℚ := 4 / 3

-- Define the final ratio of alcohol to water after adding water
def final_ratio : ℚ := 4 / 5

-- Define the amount of water added (in liters)
def water_added : ℚ := 7

-- Theorem stating the quantity of alcohol in the mixture
theorem alcohol_quantity (A : ℚ) (W : ℚ) :
  A / W = initial_ratio →
  A / (W + water_added) = final_ratio →
  A = 14 := by sorry

end NUMINAMATH_CALUDE_alcohol_quantity_l217_21719


namespace NUMINAMATH_CALUDE_system_solution_l217_21760

theorem system_solution : 
  ∃ (x y z w : ℝ), 
    (x = 3 ∧ y = 1 ∧ z = 2 ∧ w = 2) ∧
    (x - y + z - w = 2) ∧
    (x^2 - y^2 + z^2 - w^2 = 6) ∧
    (x^3 - y^3 + z^3 - w^3 = 20) ∧
    (x^4 - y^4 + z^4 - w^4 = 66) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l217_21760


namespace NUMINAMATH_CALUDE_bottle_caps_found_at_park_l217_21743

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  bottleCaps : ℕ
  wrappers : ℕ

/-- Represents the items Danny found at the park --/
structure ParkFindings where
  bottleCaps : ℕ
  wrappers : ℕ

/-- Theorem stating the number of bottle caps Danny found at the park --/
theorem bottle_caps_found_at_park 
  (initialCollection : Collection)
  (parkFindings : ParkFindings)
  (finalCollection : Collection)
  (h1 : parkFindings.wrappers = 18)
  (h2 : finalCollection.wrappers = 67)
  (h3 : finalCollection.bottleCaps = 35)
  (h4 : finalCollection.wrappers = finalCollection.bottleCaps + 32)
  (h5 : finalCollection.bottleCaps = initialCollection.bottleCaps + parkFindings.bottleCaps)
  (h6 : finalCollection.wrappers = initialCollection.wrappers + parkFindings.wrappers) :
  parkFindings.bottleCaps = 18 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_found_at_park_l217_21743


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l217_21705

-- Define the main equation
def main_equation (x a : ℝ) : Prop :=
  Real.arctan (x / 2) + Real.arctan (2 - x) = a

-- Part 1
theorem part_one :
  ∀ x : ℝ, main_equation x (π / 4) →
    Real.arccos (x / 2) = 2 * π / 3 ∨ Real.arccos (x / 2) = 0 := by sorry

-- Part 2
theorem part_two :
  ∃ x a : ℝ, main_equation x a →
    a ∈ Set.Icc (Real.arctan (1 / (-2 * Real.sqrt 10 - 6))) (Real.arctan (1 / (2 * Real.sqrt 10 - 6))) := by sorry

-- Part 3
theorem part_three :
  ∀ α β a : ℝ, 
    α ∈ Set.Icc 5 15 → β ∈ Set.Icc 5 15 →
    α ≠ β →
    main_equation α a → main_equation β a →
    α + β ≤ 19 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l217_21705


namespace NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l217_21795

theorem cinnamon_swirls_distribution (total_pieces : Real) (num_people : Real) (jane_pieces : Real) : 
  total_pieces = 12.0 → num_people = 3.0 → jane_pieces = total_pieces / num_people → jane_pieces = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l217_21795


namespace NUMINAMATH_CALUDE_system_solution_unique_l217_21730

/-- Proves that x = 2 and y = 1 is the unique solution to the given system of equations -/
theorem system_solution_unique :
  ∃! (x y : ℝ), (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l217_21730


namespace NUMINAMATH_CALUDE_new_students_count_l217_21762

theorem new_students_count (initial_students : ℕ) (left_students : ℕ) (final_students : ℕ) 
  (h1 : initial_students = 31)
  (h2 : left_students = 5)
  (h3 : final_students = 37) :
  final_students - (initial_students - left_students) = 11 :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l217_21762


namespace NUMINAMATH_CALUDE_age_difference_l217_21718

theorem age_difference (a b c : ℕ) : 
  b = 16 →
  b = 2 * c →
  a + b + c = 42 →
  a = b + 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l217_21718
