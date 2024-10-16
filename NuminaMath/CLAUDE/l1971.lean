import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1971_197145

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), 2 * (f x) - 16 = f (x - 6) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1971_197145


namespace NUMINAMATH_CALUDE_total_sprockets_produced_l1971_197148

-- Define the production rates and time difference
def machine_x_rate : ℝ := 5.999999999999999
def machine_b_rate : ℝ := machine_x_rate * 1.1
def time_difference : ℝ := 10

-- Define the theorem
theorem total_sprockets_produced :
  ∃ (time_b : ℝ),
    time_b > 0 ∧
    (machine_x_rate * (time_b + time_difference) = machine_b_rate * time_b) ∧
    (machine_x_rate * (time_b + time_difference) + machine_b_rate * time_b = 1320) := by
  sorry


end NUMINAMATH_CALUDE_total_sprockets_produced_l1971_197148


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1971_197169

theorem min_value_of_expression (a b : ℕ) (ha : 0 < a ∧ a ≤ 5) (hb : 0 < b ∧ b ≤ 5) :
  ∀ x y : ℕ, (0 < x ∧ x ≤ 5) → (0 < y ∧ y ≤ 5) → 
  a^2 - a*b + 2*b ≤ x^2 - x*y + 2*y ∧ 
  a^2 - a*b + 2*b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1971_197169


namespace NUMINAMATH_CALUDE_not_possible_N_l1971_197121

-- Define the set M
def M : Set ℝ := {x | x^2 - 6*x - 16 < 0}

-- Define the theorem
theorem not_possible_N (N : Set ℝ) (h1 : M ∩ N = N) : N ≠ Set.Icc (-1 : ℝ) 8 := by
  sorry

end NUMINAMATH_CALUDE_not_possible_N_l1971_197121


namespace NUMINAMATH_CALUDE_team_score_l1971_197128

def basketball_game (tobee jay sean remy alex : ℕ) : Prop :=
  tobee = 4 ∧
  jay = 2 * tobee + 6 ∧
  sean = jay / 2 ∧
  remy = tobee + jay - 3 ∧
  alex = sean + remy + 4

theorem team_score (tobee jay sean remy alex : ℕ) :
  basketball_game tobee jay sean remy alex →
  tobee + jay + sean + remy + alex = 66 := by
  sorry

end NUMINAMATH_CALUDE_team_score_l1971_197128


namespace NUMINAMATH_CALUDE_weight_loss_problem_l1971_197164

theorem weight_loss_problem (x : ℝ) : 
  (x - 12 = 2 * (x - 7) - 80) → x = 82 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_problem_l1971_197164


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l1971_197108

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (∃ a b, (a + 1/a) * (b + 1/b) > (Real.sqrt (a*b) + 1/Real.sqrt (a*b))^2) ∧
  (∃ a b, (a + 1/a) * (b + 1/b) > ((a+b)/2 + 2/(a+b))^2) ∧
  (∃ a b, ((a+b)/2 + 2/(a+b))^2 > (a + 1/a) * (b + 1/b)) :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_expressions_l1971_197108


namespace NUMINAMATH_CALUDE_max_stores_visited_is_four_l1971_197106

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  num_shoppers : ℕ
  two_store_visitors : ℕ
  total_visits : ℕ

/-- The maximum number of stores visited by any individual -/
def max_stores_visited (s : ShoppingScenario) : ℕ :=
  let remaining_visits := s.total_visits - 2 * s.two_store_visitors
  let remaining_shoppers := s.num_shoppers - s.two_store_visitors
  let extra_visits := remaining_visits - remaining_shoppers
  1 + extra_visits

/-- Theorem stating the maximum number of stores visited by any individual in the given scenario -/
theorem max_stores_visited_is_four (s : ShoppingScenario) :
  s.num_stores = 8 ∧ 
  s.num_shoppers = 12 ∧ 
  s.two_store_visitors = 8 ∧ 
  s.total_visits = 23 →
  max_stores_visited s = 4 :=
by
  sorry

#eval max_stores_visited {num_stores := 8, num_shoppers := 12, two_store_visitors := 8, total_visits := 23}

end NUMINAMATH_CALUDE_max_stores_visited_is_four_l1971_197106


namespace NUMINAMATH_CALUDE_range_of_a_l1971_197177

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a : 
  ∃ a : ℝ, p a ∧ ¬(q a) ∧ -1 ≤ a ∧ a < 0 ∧
  ∀ b : ℝ, p b ∧ ¬(q b) → -1 ≤ b ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1971_197177


namespace NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l1971_197127

/-- Given a right triangle DEF with hypotenuse DE = 15, DF = 9, and EF = 12,
    the distance from F to the midpoint of DE is 7.5 -/
theorem right_triangle_median_to_hypotenuse (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l1971_197127


namespace NUMINAMATH_CALUDE_milk_powder_cost_july_l1971_197156

/-- The cost of milk powder and coffee in July -/
def july_cost (june_cost : ℝ) : ℝ × ℝ :=
  (0.4 * june_cost, 3 * june_cost)

/-- The total cost of the mixture in July -/
def mixture_cost (june_cost : ℝ) : ℝ :=
  1.5 * (july_cost june_cost).1 + 1.5 * (july_cost june_cost).2

theorem milk_powder_cost_july :
  ∃ (june_cost : ℝ),
    june_cost > 0 ∧
    mixture_cost june_cost = 5.1 ∧
    (july_cost june_cost).1 = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_milk_powder_cost_july_l1971_197156


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_l1971_197167

/-- The number of unit squares in the nth ring around a 2x3 rectangle -/
def ring_squares (n : ℕ) : ℕ := 4 * n + 8

/-- Theorem: The 100th ring contains 408 unit squares -/
theorem hundredth_ring_squares :
  ring_squares 100 = 408 := by sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_l1971_197167


namespace NUMINAMATH_CALUDE_complex_division_result_l1971_197195

theorem complex_division_result : Complex.I / (1 - Complex.I) = -1/2 + Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l1971_197195


namespace NUMINAMATH_CALUDE_solution_to_equation_l1971_197174

theorem solution_to_equation (x : ℝ) (h : (9 : ℝ) / x^2 = x / 81) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1971_197174


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l1971_197188

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 11 → difference = 3 → friend_cost = total / 2 + difference / 2 → friend_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l1971_197188


namespace NUMINAMATH_CALUDE_egg_laying_hens_l1971_197131

theorem egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : roosters = 28)
  (h3 : non_laying_hens = 20) :
  total_chickens - roosters - non_laying_hens = 277 := by
  sorry

#check egg_laying_hens

end NUMINAMATH_CALUDE_egg_laying_hens_l1971_197131


namespace NUMINAMATH_CALUDE_odd_not_even_function_implication_l1971_197114

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |x - a|

theorem odd_not_even_function_implication (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (∃ x, f a x ≠ f a (-x)) →
  (∃ x, f a x ≠ 0) →
  (a + 1)^2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_not_even_function_implication_l1971_197114


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l1971_197173

theorem shaded_area_fraction (n : ℕ) (h : n = 18) :
  let total_rectangles := n
  let shaded_rectangles := n / 2
  (shaded_rectangles : ℚ) / total_rectangles = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l1971_197173


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1971_197136

def A : Set ℝ := {x : ℝ | -4 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -4 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1971_197136


namespace NUMINAMATH_CALUDE_planes_parallel_from_intersecting_lines_l1971_197129

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_in : Line → Plane → Prop)  -- A line lies in a plane
variable (parallel : Line → Plane → Prop)  -- A line is parallel to a plane
variable (intersect_at : Line → Line → Point → Prop)  -- Two lines intersect at a point
variable (plane_parallel : Plane → Plane → Prop)  -- Two planes are parallel

-- State the theorem
theorem planes_parallel_from_intersecting_lines 
  (l m : Line) (α β : Plane) (P : Point) :
  l ≠ m →  -- l and m are distinct lines
  α ≠ β →  -- α and β are different planes
  lies_in l α →
  lies_in m α →
  intersect_at l m P →
  parallel l β →
  parallel m β →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_intersecting_lines_l1971_197129


namespace NUMINAMATH_CALUDE_max_reflections_theorem_l1971_197153

/-- The angle between two lines in degrees -/
def angle_between_lines : ℝ := 6

/-- The maximum number of reflections before perpendicular incidence -/
def max_reflections : ℕ := 15

/-- Theorem: Given the angle between two lines is 6°, the maximum number of reflections
    before perpendicular incidence is 15 -/
theorem max_reflections_theorem (angle : ℝ) (n : ℕ) 
  (h1 : angle = angle_between_lines)
  (h2 : n = max_reflections) :
  n * angle = 90 ∧ ∀ m : ℕ, m > n → m * angle > 90 := by
  sorry

#check max_reflections_theorem

end NUMINAMATH_CALUDE_max_reflections_theorem_l1971_197153


namespace NUMINAMATH_CALUDE_derivative_sqrt_at_one_l1971_197120

theorem derivative_sqrt_at_one :
  let f : ℝ → ℝ := λ x => Real.sqrt x
  HasDerivAt f (1/2) 1 := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_at_one_l1971_197120


namespace NUMINAMATH_CALUDE_red_toys_after_removal_l1971_197194

/-- Theorem: Number of red toys after removal --/
theorem red_toys_after_removal
  (total : ℕ)
  (h_total : total = 134)
  (red white : ℕ)
  (h_initial : red + white = total)
  (h_after_removal : red - 2 = 2 * white) :
  red - 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_red_toys_after_removal_l1971_197194


namespace NUMINAMATH_CALUDE_probability_two_girls_chosen_l1971_197124

def total_members : ℕ := 12
def num_boys : ℕ := 7
def num_girls : ℕ := 5

theorem probability_two_girls_chosen (total_members num_boys num_girls : ℕ) 
  (h1 : total_members = 12)
  (h2 : num_boys = 7)
  (h3 : num_girls = 5)
  (h4 : total_members = num_boys + num_girls) :
  (Nat.choose num_girls 2 : ℚ) / (Nat.choose total_members 2) = 5 / 33 := by
sorry

end NUMINAMATH_CALUDE_probability_two_girls_chosen_l1971_197124


namespace NUMINAMATH_CALUDE_min_value_sum_l1971_197102

theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h : 1/p + 1/q + 1/r = 1) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z = 1 → p + q + r ≤ x + y + z ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 1 ∧ a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1971_197102


namespace NUMINAMATH_CALUDE_password_probability_l1971_197133

/-- The set of possible first characters in the password -/
def first_char : Finset Char := {'M', 'I', 'N'}

/-- The set of possible second characters in the password -/
def second_char : Finset Char := {'1', '2', '3', '4', '5'}

/-- The type representing a two-character password -/
def Password := Char × Char

/-- The set of all possible passwords -/
def all_passwords : Finset Password :=
  first_char.product second_char

theorem password_probability :
  (Finset.card all_passwords : ℚ) = 15 ∧
  (1 : ℚ) / (Finset.card all_passwords : ℚ) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l1971_197133


namespace NUMINAMATH_CALUDE_power_of_square_l1971_197146

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l1971_197146


namespace NUMINAMATH_CALUDE_xyz_value_l1971_197190

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12)
  (h3 : (x + y + z)^2 = x^2 + y^2 + z^2 + 12) :
  x * y * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1971_197190


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l1971_197163

theorem students_passed_both_tests 
  (total : Nat) 
  (passed_long_jump : Nat) 
  (passed_shot_put : Nat) 
  (failed_both : Nat) : 
  total = 50 → 
  passed_long_jump = 40 → 
  passed_shot_put = 31 → 
  failed_both = 4 → 
  ∃ (passed_both : Nat), 
    passed_both = 25 ∧ 
    total = passed_both + (passed_long_jump - passed_both) + (passed_shot_put - passed_both) + failed_both :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l1971_197163


namespace NUMINAMATH_CALUDE_set_union_problem_l1971_197140

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {2} →
  M ∪ N = {0, 1, 2} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1971_197140


namespace NUMINAMATH_CALUDE_max_b_is_maximum_l1971_197138

def is_lattice_point (x y : ℤ) : Prop := true

def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 50 → is_lattice_point x y → line_equation m x ≠ y

def max_b : ℚ := 11/51

theorem max_b_is_maximum :
  (∀ m : ℚ, 2/5 < m → m < max_b → no_lattice_points m) ∧
  ∀ b : ℚ, b > max_b → ∃ m : ℚ, 2/5 < m ∧ m < b ∧ ¬(no_lattice_points m) :=
sorry

end NUMINAMATH_CALUDE_max_b_is_maximum_l1971_197138


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l1971_197116

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > b ∧ b > 0
  ecc : c / a = Real.sqrt 2 / 2
  perimeter : ℝ
  h_perimeter : perimeter = 4

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_m : m ≠ 0
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  h_line : ∀ x y, y = k * x + m
  h_intersect : A.1^2 / (E.b^2) + A.2^2 / (E.a^2) = 1 ∧
                B.1^2 / (E.b^2) + B.2^2 / (E.a^2) = 1
  h_relation : A.1 + 3 * B.1 = 4 * P.1 ∧ A.2 + 3 * B.2 = 4 * P.2

/-- The main theorem -/
theorem ellipse_and_line_properties (E : Ellipse) (L : IntersectingLine E) :
  (E.a = 1 ∧ E.b = Real.sqrt 2 / 2) ∧
  (L.m ∈ Set.Ioo (-1 : ℝ) (-1/2) ∪ Set.Ioo (1/2 : ℝ) 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l1971_197116


namespace NUMINAMATH_CALUDE_prob_eight_rolls_prime_odd_l1971_197172

/-- A function representing the probability of rolling either 3 or 5 on a standard die -/
def prob_prime_odd_roll : ℚ := 1 / 3

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The probability of getting a product of all rolls that is odd and consists only of prime numbers -/
def prob_all_prime_odd : ℚ := (prob_prime_odd_roll) ^ num_rolls

theorem prob_eight_rolls_prime_odd :
  prob_all_prime_odd = 1 / 6561 := by sorry

end NUMINAMATH_CALUDE_prob_eight_rolls_prime_odd_l1971_197172


namespace NUMINAMATH_CALUDE_work_completion_time_l1971_197191

/-- The time taken by A, B, and C to complete a work given their pairwise completion times -/
theorem work_completion_time 
  (time_AB : ℝ) 
  (time_BC : ℝ) 
  (time_AC : ℝ) 
  (h_AB : time_AB = 8) 
  (h_BC : time_BC = 12) 
  (h_AC : time_AC = 8) : 
  (1 / (1 / time_AB + 1 / time_BC + 1 / time_AC)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1971_197191


namespace NUMINAMATH_CALUDE_batsman_average_after_19th_inning_l1971_197111

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRunsBefore : ℕ
  scoreInLastInning : ℕ
  averageIncrease : ℚ

/-- Calculates the new average of a batsman after their latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRunsBefore + b.scoreInLastInning : ℚ) / b.innings

theorem batsman_average_after_19th_inning 
  (b : Batsman) 
  (h1 : b.innings = 19) 
  (h2 : b.scoreInLastInning = 100) 
  (h3 : b.averageIncrease = 2) :
  newAverage b = 64 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_19th_inning_l1971_197111


namespace NUMINAMATH_CALUDE_winter_holiday_activities_l1971_197147

theorem winter_holiday_activities (total : ℕ) (skating : ℕ) (skiing : ℕ) (both : ℕ) :
  total = 30 →
  skating = 20 →
  skiing = 9 →
  both = 5 →
  total - (skating + skiing - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_winter_holiday_activities_l1971_197147


namespace NUMINAMATH_CALUDE_solve_for_k_l1971_197168

theorem solve_for_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (heq : k * x - y = 3) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l1971_197168


namespace NUMINAMATH_CALUDE_initial_students_on_bus_l1971_197139

theorem initial_students_on_bus (students_left_bus : ℕ) (students_remaining : ℕ) 
  (h1 : students_left_bus = 3) 
  (h2 : students_remaining = 7) : 
  students_left_bus + students_remaining = 10 := by
sorry

end NUMINAMATH_CALUDE_initial_students_on_bus_l1971_197139


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1971_197180

theorem sum_of_decimals :
  5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1971_197180


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1971_197199

/-- Proves that sin 42° * cos 18° - cos 138° * cos 72° = √3/2 -/
theorem trig_identity_proof : 
  Real.sin (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (138 * π / 180) * Real.cos (72 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1971_197199


namespace NUMINAMATH_CALUDE_ceiling_tiling_count_l1971_197154

/-- Represents a rectangular region -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a tile -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- Counts the number of ways to tile a rectangle with given tiles -/
def count_tilings (r : Rectangle) (t : Tile) : ℕ :=
  sorry

/-- Counts the number of ways to tile a rectangle with a beam -/
def count_tilings_with_beam (r : Rectangle) (t : Tile) (beam_pos : ℕ) : ℕ :=
  sorry

theorem ceiling_tiling_count :
  let ceiling := Rectangle.mk 6 4
  let tile := Tile.mk 2 1
  let beam_pos := 2
  count_tilings_with_beam ceiling tile beam_pos = 180 :=
sorry

end NUMINAMATH_CALUDE_ceiling_tiling_count_l1971_197154


namespace NUMINAMATH_CALUDE_subset_proportion_bound_l1971_197149

theorem subset_proportion_bound 
  (total : ℕ) 
  (subset : ℕ) 
  (event1_total : ℕ) 
  (event1_subset : ℕ) 
  (event2_total : ℕ) 
  (event2_subset : ℕ) 
  (h1 : event1_subset < 2 * event1_total / 5)
  (h2 : event2_subset < 2 * event2_total / 5)
  (h3 : event1_subset + event2_subset ≥ subset)
  (h4 : event1_total + event2_total ≥ total) :
  subset < 4 * total / 7 := by
sorry

end NUMINAMATH_CALUDE_subset_proportion_bound_l1971_197149


namespace NUMINAMATH_CALUDE_equation_solution_l1971_197103

theorem equation_solution : ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1971_197103


namespace NUMINAMATH_CALUDE_jasons_textbooks_l1971_197115

/-- Represents the problem of determining the number of textbooks Jason has. -/
theorem jasons_textbooks :
  let bookcase_limit : ℕ := 80  -- Maximum weight the bookcase can hold in pounds
  let hardcover_count : ℕ := 70  -- Number of hardcover books
  let hardcover_weight : ℚ := 1/2  -- Weight of each hardcover book in pounds
  let textbook_weight : ℕ := 2  -- Weight of each textbook in pounds
  let knickknack_count : ℕ := 3  -- Number of knick-knacks
  let knickknack_weight : ℕ := 6  -- Weight of each knick-knack in pounds
  let over_limit : ℕ := 33  -- Amount the total collection is over the weight limit in pounds

  let total_weight := bookcase_limit + over_limit
  let hardcover_total_weight := hardcover_count * hardcover_weight
  let knickknack_total_weight := knickknack_count * knickknack_weight
  let textbook_total_weight := total_weight - (hardcover_total_weight + knickknack_total_weight)

  textbook_total_weight / textbook_weight = 30 := by
  sorry

end NUMINAMATH_CALUDE_jasons_textbooks_l1971_197115


namespace NUMINAMATH_CALUDE_bus_ticket_probability_l1971_197179

/-- Represents the lottery game with given parameters -/
structure LotteryGame where
  initialAmount : ℝ
  ticketCost : ℝ
  winProbability : ℝ
  prizeAmount : ℝ
  targetAmount : ℝ

/-- Calculates the probability of winning enough money to reach the target amount -/
noncomputable def winProbability (game : LotteryGame) : ℝ :=
  let p := game.winProbability
  let q := 1 - p
  (p^2 * (1 + 2*q)) / (1 - 2*p*q^2)

/-- Theorem stating the probability of winning the bus ticket -/
theorem bus_ticket_probability (game : LotteryGame) 
  (h1 : game.initialAmount = 20)
  (h2 : game.ticketCost = 10)
  (h3 : game.winProbability = 0.1)
  (h4 : game.prizeAmount = 30)
  (h5 : game.targetAmount = 45) :
  ∃ ε > 0, |winProbability game - 0.033| < ε :=
sorry

end NUMINAMATH_CALUDE_bus_ticket_probability_l1971_197179


namespace NUMINAMATH_CALUDE_square_probability_l1971_197109

/-- The number of squares in a 6x6 grid -/
def grid_size : ℕ := 36

/-- The number of squares to be chosen -/
def chosen_squares : ℕ := 4

/-- The number of ways to choose 4 squares from 36 -/
def total_combinations : ℕ := Nat.choose grid_size chosen_squares

/-- The number of ways to form a square from the centers of 4 squares in a 6x6 grid -/
def favorable_outcomes : ℕ := 105

/-- The probability of randomly selecting 4 squares whose centers form a square in a 6x6 grid -/
theorem square_probability : 
  (favorable_outcomes : ℚ) / total_combinations = 1 / 561 := by sorry

end NUMINAMATH_CALUDE_square_probability_l1971_197109


namespace NUMINAMATH_CALUDE_characteristic_vector_of_g_sin_value_for_associated_function_l1971_197162

def associated_characteristic_vector (f : ℝ → ℝ) : ℝ × ℝ :=
  sorry

def associated_function (v : ℝ × ℝ) : ℝ → ℝ :=
  sorry

theorem characteristic_vector_of_g :
  let g : ℝ → ℝ := λ x => Real.sin (x + 5 * Real.pi / 6) - Real.sin (3 * Real.pi / 2 - x)
  associated_characteristic_vector g = (-Real.sqrt 3 / 2, 3 / 2) :=
sorry

theorem sin_value_for_associated_function :
  let f := associated_function (1, Real.sqrt 3)
  ∀ x, f x = 8 / 5 → x > -Real.pi / 3 → x < Real.pi / 6 →
    Real.sin x = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end NUMINAMATH_CALUDE_characteristic_vector_of_g_sin_value_for_associated_function_l1971_197162


namespace NUMINAMATH_CALUDE_hunter_frog_count_l1971_197137

/-- The number of frogs Hunter saw in the pond -/
def total_frogs (lily_pad_frogs log_frogs baby_frogs : ℕ) : ℕ :=
  lily_pad_frogs + log_frogs + baby_frogs

/-- Two dozen -/
def two_dozen : ℕ := 2 * 12

theorem hunter_frog_count :
  total_frogs 5 3 two_dozen = 32 := by
  sorry

end NUMINAMATH_CALUDE_hunter_frog_count_l1971_197137


namespace NUMINAMATH_CALUDE_expression_simplification_l1971_197117

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (1 / (x - 1) + 1 / (x + 1)) / (x^2 / (3 * x^2 - 3)) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1971_197117


namespace NUMINAMATH_CALUDE_fraction_simplification_l1971_197155

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1971_197155


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l1971_197132

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 18 → (c + d) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l1971_197132


namespace NUMINAMATH_CALUDE_two_number_problem_l1971_197134

theorem two_number_problem (A B n : ℕ) : 
  B > 0 → 
  A > B → 
  A = 10 * B + n → 
  0 ≤ n → 
  n ≤ 9 → 
  A + B = 2022 → 
  A = 1839 ∧ B = 183 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l1971_197134


namespace NUMINAMATH_CALUDE_root_range_l1971_197178

/-- Given that the equation |x-k| = (√2/2)k√x has two unequal real roots in the interval [k-1, k+1], prove that the range of k is 0 < k ≤ 1. -/
theorem root_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    k - 1 ≤ x₁ ∧ x₁ ≤ k + 1 ∧
    k - 1 ≤ x₂ ∧ x₂ ≤ k + 1 ∧
    |x₁ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₁ ∧
    |x₂ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_root_range_l1971_197178


namespace NUMINAMATH_CALUDE_purchase_cost_l1971_197104

/-- The cost of a single pencil in dollars -/
def pencil_cost : ℚ := 2.5

/-- The cost of a single pen in dollars -/
def pen_cost : ℚ := 3.5

/-- The number of pencils bought -/
def num_pencils : ℕ := 38

/-- The number of pens bought -/
def num_pens : ℕ := 56

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := pencil_cost * num_pencils + pen_cost * num_pens

theorem purchase_cost : total_cost = 291 := by sorry

end NUMINAMATH_CALUDE_purchase_cost_l1971_197104


namespace NUMINAMATH_CALUDE_product_of_positive_reals_l1971_197122

theorem product_of_positive_reals (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15/8) : r * s = Real.sqrt 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_positive_reals_l1971_197122


namespace NUMINAMATH_CALUDE_pastry_production_theorem_l1971_197143

/-- Represents a baker's production --/
structure BakerProduction where
  mini_cupcakes : ℕ
  pop_tarts : ℕ
  blueberry_pies : ℕ
  chocolate_eclairs : ℕ
  macarons : ℕ

/-- Calculates the total number of pastries for a baker --/
def total_pastries (bp : BakerProduction) : ℕ :=
  bp.mini_cupcakes + bp.pop_tarts + bp.blueberry_pies + bp.chocolate_eclairs + bp.macarons

/-- Calculates the total cost of pastries for a baker --/
def total_cost (bp : BakerProduction) : ℚ :=
  bp.mini_cupcakes * (1/2) + bp.pop_tarts * 1 + bp.blueberry_pies * 3 + bp.chocolate_eclairs * 2 + bp.macarons * (3/2)

theorem pastry_production_theorem (lola lulu lila luka : BakerProduction) : 
  lola = { mini_cupcakes := 13, pop_tarts := 10, blueberry_pies := 8, chocolate_eclairs := 6, macarons := 0 } →
  lulu = { mini_cupcakes := 16, pop_tarts := 12, blueberry_pies := 14, chocolate_eclairs := 9, macarons := 0 } →
  lila = { mini_cupcakes := 22, pop_tarts := 15, blueberry_pies := 10, chocolate_eclairs := 12, macarons := 0 } →
  luka = { mini_cupcakes := 18, pop_tarts := 20, blueberry_pies := 7, chocolate_eclairs := 14, macarons := 25 } →
  (total_pastries lola + total_pastries lulu + total_pastries lila + total_pastries luka = 231) ∧
  (total_cost lola + total_cost lulu + total_cost lila + total_cost luka = 328) := by
  sorry

end NUMINAMATH_CALUDE_pastry_production_theorem_l1971_197143


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_min_four_dollar_frisbees_proof_l1971_197187

/-- Given 60 frisbees sold at either $3 or $4 each, with total receipts of $204,
    the minimum number of $4 frisbees sold is 24. -/
theorem min_four_dollar_frisbees : ℕ :=
  let total_frisbees : ℕ := 60
  let total_receipts : ℕ := 204
  let price_low : ℕ := 3
  let price_high : ℕ := 4
  24

/-- Proof that the minimum number of $4 frisbees sold is indeed 24. -/
theorem min_four_dollar_frisbees_proof :
  let total_frisbees : ℕ := 60
  let total_receipts : ℕ := 204
  let price_low : ℕ := 3
  let price_high : ℕ := 4
  let min_high_price_frisbees := min_four_dollar_frisbees
  (∃ (low_price_frisbees : ℕ),
    low_price_frisbees + min_high_price_frisbees = total_frisbees ∧
    low_price_frisbees * price_low + min_high_price_frisbees * price_high = total_receipts) ∧
  (∀ (high_price_frisbees : ℕ),
    high_price_frisbees < min_high_price_frisbees →
    ¬∃ (low_price_frisbees : ℕ),
      low_price_frisbees + high_price_frisbees = total_frisbees ∧
      low_price_frisbees * price_low + high_price_frisbees * price_high = total_receipts) :=
by
  sorry

#check min_four_dollar_frisbees
#check min_four_dollar_frisbees_proof

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_min_four_dollar_frisbees_proof_l1971_197187


namespace NUMINAMATH_CALUDE_clock_angle_at_7pm_l1971_197123

/-- The number of hours on a clock face. -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle. -/
def full_circle_degrees : ℕ := 360

/-- The time in hours (7 p.m. is represented as 19). -/
def time : ℕ := 19

/-- The angle between hour marks on a clock face. -/
def angle_per_hour : ℚ := full_circle_degrees / clock_hours

/-- The number of hour marks between the hour hand and 12 o'clock at the given time. -/
def hour_hand_position : ℕ := time % clock_hours

/-- The angle between the hour and minute hands at the given time. -/
def clock_angle : ℚ := angle_per_hour * hour_hand_position

/-- The smaller angle between the hour and minute hands. -/
def smaller_angle : ℚ := min clock_angle (full_circle_degrees - clock_angle)

/-- 
Theorem: The measure of the smaller angle formed by the hour and minute hands 
of a clock at 7 p.m. is 150°.
-/
theorem clock_angle_at_7pm : smaller_angle = 150 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_7pm_l1971_197123


namespace NUMINAMATH_CALUDE_tomato_field_area_l1971_197112

theorem tomato_field_area (length : ℝ) (width : ℝ) (tomato_area : ℝ) : 
  length = 3.6 →
  width = 2.5 * length →
  tomato_area = (length * width) / 2 →
  tomato_area = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_field_area_l1971_197112


namespace NUMINAMATH_CALUDE_jose_land_share_l1971_197198

def total_land_area : ℝ := 20000
def num_siblings : ℕ := 4

theorem jose_land_share :
  let total_people := num_siblings + 1
  let share := total_land_area / total_people
  share = 4000 := by
  sorry

end NUMINAMATH_CALUDE_jose_land_share_l1971_197198


namespace NUMINAMATH_CALUDE_extra_money_is_seven_l1971_197192

/-- The amount of extra money given by an appreciative customer to Hillary at a flea market. -/
def extra_money (price_per_craft : ℕ) (crafts_sold : ℕ) (deposited : ℕ) (remaining : ℕ) : ℕ :=
  (deposited + remaining) - (price_per_craft * crafts_sold)

/-- Theorem stating that the extra money given to Hillary is 7 dollars. -/
theorem extra_money_is_seven :
  extra_money 12 3 18 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_extra_money_is_seven_l1971_197192


namespace NUMINAMATH_CALUDE_distance_on_number_line_l1971_197161

theorem distance_on_number_line (a b : ℝ) (ha : a = 5) (hb : b = -3) :
  |a - b| = 8 := by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l1971_197161


namespace NUMINAMATH_CALUDE_half_inequality_l1971_197165

theorem half_inequality (a b : ℝ) (h : a > b) : (1/2) * a > (1/2) * b := by
  sorry

end NUMINAMATH_CALUDE_half_inequality_l1971_197165


namespace NUMINAMATH_CALUDE_min_value_theorem_l1971_197171

def f (x : ℝ) := 45 * |2*x - 1|

def g (x : ℝ) := f x + f (x - 1)

theorem min_value_theorem (a m n : ℝ) :
  (∀ x, g x ≥ a) →
  m > 0 →
  n > 0 →
  m + n = a →
  (∀ p q, p > 0 → q > 0 → p + q = a → 4/m + 1/n ≤ 4/p + 1/q) →
  4/m + 1/n = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1971_197171


namespace NUMINAMATH_CALUDE_scientific_notation_of_passenger_trips_l1971_197193

theorem scientific_notation_of_passenger_trips :
  let trips : ℝ := 56.99 * 1000000
  trips = 5.699 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_passenger_trips_l1971_197193


namespace NUMINAMATH_CALUDE_problem_solution_l1971_197182

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 3)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = -4) :
  a / (a + c) + b / (b + a) + c / (c + b) = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1971_197182


namespace NUMINAMATH_CALUDE_inequality_proof_l1971_197125

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1971_197125


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1971_197150

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1971_197150


namespace NUMINAMATH_CALUDE_average_difference_is_negative_six_point_fifteen_l1971_197166

/- Define the parameters of the problem -/
def total_students : ℕ := 120
def total_teachers : ℕ := 6
def dual_enrolled_students : ℕ := 10
def class_enrollments : List ℕ := [40, 30, 25, 15, 5, 5]

/- Define the average number of students per teacher -/
def t : ℚ := (total_students : ℚ) / total_teachers

/- Define the average number of students per student, including dual enrollments -/
def s : ℚ :=
  let total_enrollments := total_students + dual_enrolled_students
  (class_enrollments.map (λ x => (x : ℚ) * x / total_enrollments)).sum

/- The theorem to be proved -/
theorem average_difference_is_negative_six_point_fifteen :
  t - s = -315 / 100 := by sorry

end NUMINAMATH_CALUDE_average_difference_is_negative_six_point_fifteen_l1971_197166


namespace NUMINAMATH_CALUDE_mod_sixteen_equivalence_l1971_197158

theorem mod_sixteen_equivalence : ∃! m : ℤ, 0 ≤ m ∧ m ≤ 15 ∧ m ≡ 12345 [ZMOD 16] ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_sixteen_equivalence_l1971_197158


namespace NUMINAMATH_CALUDE_max_profit_is_120_l1971_197119

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := -x^2 + 21*x

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2*x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := L₁ x + L₂ x

/-- Total sales volume constraint -/
def sales_constraint : ℝ := 15

theorem max_profit_is_120 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ sales_constraint ∧
  ∀ y : ℝ, y ≥ 0 ∧ y ≤ sales_constraint →
  total_profit x ≥ total_profit y ∧
  total_profit x = 120 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_is_120_l1971_197119


namespace NUMINAMATH_CALUDE_inverse_g_84_l1971_197135

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_84_l1971_197135


namespace NUMINAMATH_CALUDE_hamburger_combinations_l1971_197142

theorem hamburger_combinations : 
  let num_condiments : ℕ := 10
  let num_bun_types : ℕ := 2
  let num_patty_choices : ℕ := 3
  (2 ^ num_condiments) * num_bun_types * num_patty_choices = 6144 :=
by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l1971_197142


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1971_197175

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1971_197175


namespace NUMINAMATH_CALUDE_expression_value_l1971_197189

theorem expression_value : -2^4 + 3 * (-1)^6 - (-2)^3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1971_197189


namespace NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l1971_197170

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def A : ℕ := digit_sum (4444^4444)

def B : ℕ := digit_sum A

theorem sum_of_digits_B_is_seven :
  digit_sum B = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l1971_197170


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1971_197118

theorem simplify_sqrt_difference : 
  (Real.sqrt 648 / Real.sqrt 72) - (Real.sqrt 294 / Real.sqrt 98) = 3 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1971_197118


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1971_197113

theorem solve_exponential_equation :
  ∃ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1971_197113


namespace NUMINAMATH_CALUDE_marbles_given_correct_l1971_197181

/-- The number of marbles given to the brother -/
def marbles_given : ℕ := 2

/-- The initial number of marbles you have -/
def initial_marbles : ℕ := 16

/-- The total number of marbles among all three people -/
def total_marbles : ℕ := 63

theorem marbles_given_correct :
  -- After giving marbles, you have double your brother's marbles
  2 * ((initial_marbles - marbles_given) / 2) = initial_marbles - marbles_given ∧
  -- Your friend has triple your marbles after giving
  3 * (initial_marbles - marbles_given) = 
    total_marbles - (initial_marbles - marbles_given) - ((initial_marbles - marbles_given) / 2) :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_correct_l1971_197181


namespace NUMINAMATH_CALUDE_term_free_of_x_l1971_197151

theorem term_free_of_x (m n k : ℕ) : 
  (∃ r : ℕ, r ≤ k ∧ m * k - (m + n) * r = 0) ↔ (m * k) % (m + n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_term_free_of_x_l1971_197151


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1971_197105

/-- Given two planes α and β with normal vectors n1 and n2 respectively,
    prove that if the planes are parallel, then k = 4. -/
theorem parallel_planes_normal_vectors (n1 n2 : ℝ × ℝ × ℝ) (k : ℝ) : 
  n1 = (1, 2, -2) → n2 = (-2, -4, k) → (∃ (c : ℝ), n1 = c • n2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l1971_197105


namespace NUMINAMATH_CALUDE_fraction_simplification_l1971_197160

theorem fraction_simplification (c : ℝ) : (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1971_197160


namespace NUMINAMATH_CALUDE_candy_probability_l1971_197101

/-- Represents the number of red candies in the jar -/
def red_candies : ℕ := 15

/-- Represents the number of blue candies in the jar -/
def blue_candies : ℕ := 20

/-- Represents the total number of candies in the jar -/
def total_candies : ℕ := red_candies + blue_candies

/-- Represents the number of candies each person picks -/
def picks : ℕ := 3

/-- The probability of Terry and Mary getting the same color combination -/
def same_color_probability : ℚ := 243 / 6825

theorem candy_probability : 
  let terry_red_prob := (red_candies.choose picks : ℚ) / (total_candies.choose picks)
  let terry_blue_prob := (blue_candies.choose picks : ℚ) / (total_candies.choose picks)
  let mary_red_prob := ((red_candies - picks).choose picks : ℚ) / ((total_candies - picks).choose picks)
  let mary_blue_prob := ((blue_candies - picks).choose picks : ℚ) / ((total_candies - picks).choose picks)
  terry_red_prob * mary_red_prob + terry_blue_prob * mary_blue_prob = same_color_probability :=
sorry

end NUMINAMATH_CALUDE_candy_probability_l1971_197101


namespace NUMINAMATH_CALUDE_c_minus_d_value_l1971_197157

theorem c_minus_d_value (c d : ℝ) 
  (eq1 : 2020 * c + 2024 * d = 2030)
  (eq2 : 2022 * c + 2026 * d = 2032) : 
  c - d = -4 := by
sorry

end NUMINAMATH_CALUDE_c_minus_d_value_l1971_197157


namespace NUMINAMATH_CALUDE_golf_score_difference_l1971_197152

/-- Given Richard's and Bruno's golf scores, prove the difference between their scores. -/
theorem golf_score_difference (richard_score bruno_score : ℕ) 
  (h1 : richard_score = 62) 
  (h2 : bruno_score = 48) : 
  richard_score - bruno_score = 14 := by
  sorry

end NUMINAMATH_CALUDE_golf_score_difference_l1971_197152


namespace NUMINAMATH_CALUDE_frisbee_committee_formations_l1971_197141

def num_teams : Nat := 5
def team_size : Nat := 8
def host_committee_size : Nat := 4
def non_host_committee_size : Nat := 2

theorem frisbee_committee_formations :
  (num_teams * (Nat.choose team_size host_committee_size) *
   (Nat.choose team_size non_host_committee_size) ^ (num_teams - 1)) =
  215134600 := by
  sorry

end NUMINAMATH_CALUDE_frisbee_committee_formations_l1971_197141


namespace NUMINAMATH_CALUDE_set_union_problem_l1971_197126

theorem set_union_problem (a b : ℝ) : 
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {3^a, b}
  A ∪ B = {-1, 0, 1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1971_197126


namespace NUMINAMATH_CALUDE_radical_product_equals_27_l1971_197185

theorem radical_product_equals_27 : Real.sqrt (Real.sqrt (Real.sqrt 27 * 27) * 81) * Real.sqrt 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_27_l1971_197185


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1971_197159

theorem polygon_interior_angles_sum (n : ℕ) (h : n = 9) :
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1971_197159


namespace NUMINAMATH_CALUDE_card_value_decrease_l1971_197130

/-- Represents the percent decrease in the first year -/
def first_year_decrease : ℝ := sorry

/-- Represents the percent decrease in the second year -/
def second_year_decrease : ℝ := 10

/-- Represents the total percent decrease over two years -/
def total_decrease : ℝ := 55

theorem card_value_decrease :
  (1 - first_year_decrease / 100) * (1 - second_year_decrease / 100) = 1 - total_decrease / 100 ∧
  first_year_decrease = 50 := by sorry

end NUMINAMATH_CALUDE_card_value_decrease_l1971_197130


namespace NUMINAMATH_CALUDE_odd_function_extension_l1971_197100

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x * (1 - x)) →  -- f(x) = x(1-x) for x > 0
  (∀ x < 0, f x = x * (1 + x)) :=  -- f(x) = x(1+x) for x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1971_197100


namespace NUMINAMATH_CALUDE_slope_of_solutions_l1971_197110

-- Define the equation
def satisfies_equation (x y : ℝ) : Prop := (4 / x) + (5 / y) = 0

-- Theorem statement
theorem slope_of_solutions (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : satisfies_equation x₁ y₁)
  (h₂ : satisfies_equation x₂ y₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  (y₂ - y₁) / (x₂ - x₁) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_solutions_l1971_197110


namespace NUMINAMATH_CALUDE_min_Q_value_l1971_197107

def is_special_number (m : ℕ) : Prop :=
  10 ≤ m ∧ m < 100 ∧ m % 10 ≠ m / 10 ∧ m % 10 ≠ 0 ∧ m / 10 ≠ 0

def swap_digits (m : ℕ) : ℕ :=
  (m % 10) * 10 + m / 10

def F (m : ℕ) : ℚ :=
  (m * 100 + swap_digits m - (swap_digits m * 100 + m)) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s : ℚ) / s

theorem min_Q_value (a b x y : ℕ) (h1 : 1 ≤ b) (h2 : b < a) (h3 : a ≤ 7)
    (h4 : 1 ≤ x) (h5 : x ≤ 8) (h6 : 1 ≤ y) (h7 : y ≤ 8)
    (hs : is_special_number (10 * a + b)) (ht : is_special_number (10 * x + y))
    (hFs : F (10 * a + b) % 5 = 1)
    (hFt : F (10 * x + y) - F (10 * a + b) + 18 * x = 36) :
    ∃ (s t : ℕ), is_special_number s ∧ is_special_number t ∧
      Q s t = -42 / 73 ∧ ∀ (s' t' : ℕ), is_special_number s' → is_special_number t' →
        Q s' t' ≥ -42 / 73 :=
  sorry

end NUMINAMATH_CALUDE_min_Q_value_l1971_197107


namespace NUMINAMATH_CALUDE_interest_first_year_l1971_197186

def initial_deposit : ℝ := 5000
def balance_after_first_year : ℝ := 5500
def second_year_increase : ℝ := 0.1
def total_increase : ℝ := 0.21

theorem interest_first_year :
  balance_after_first_year - initial_deposit = 500 :=
sorry

end NUMINAMATH_CALUDE_interest_first_year_l1971_197186


namespace NUMINAMATH_CALUDE_pen_arrangement_count_l1971_197184

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def multinomial (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem pen_arrangement_count :
  let total_pens := 15
  let blue_pens := 7
  let red_pens := 3
  let green_pens := 3
  let black_pens := 2
  let total_arrangements := multinomial total_pens [blue_pens, red_pens, green_pens, black_pens]
  let adjacent_green_arrangements := 
    (multinomial (total_pens - green_pens + 1) [blue_pens, red_pens, 1, black_pens]) * (factorial green_pens)
  total_arrangements - adjacent_green_arrangements = 6098400 := by
  sorry

end NUMINAMATH_CALUDE_pen_arrangement_count_l1971_197184


namespace NUMINAMATH_CALUDE_total_distance_rowed_total_distance_is_15_19_l1971_197196

/-- Calculates the total distance traveled by a man rowing upstream and downstream in a river -/
theorem total_distance_rowed (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (total_time * upstream_speed * downstream_speed) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance

/-- Proves that the total distance traveled is approximately 15.19 km -/
theorem total_distance_is_15_19 :
  ∃ ε > 0, |total_distance_rowed 8 1.8 2 - 15.19| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_rowed_total_distance_is_15_19_l1971_197196


namespace NUMINAMATH_CALUDE_derivative_of_composite_l1971_197144

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition that f'(3) = 9
variable (h : deriv f 3 = 9)

-- State the theorem
theorem derivative_of_composite (f : ℝ → ℝ) (h : deriv f 3 = 9) :
  deriv (fun x ↦ f (3 * x^2)) 1 = 54 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_composite_l1971_197144


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l1971_197176

theorem rectangle_area : ℕ → Prop :=
  fun area =>
    ∃ (square_side : ℕ) (length breadth : ℕ),
      square_side * square_side = 2025 ∧
      length = (2 * square_side) / 5 ∧
      breadth = length / 2 + 5 ∧
      (length + breadth) % 3 = 0 ∧
      length * breadth = area

theorem rectangle_area_is_270 : rectangle_area 270 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_270_l1971_197176


namespace NUMINAMATH_CALUDE_meaningful_expression_l1971_197183

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y * y = x ∧ y ≥ 0) ∧ x ≠ 2 ↔ x ≥ 0 ∧ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1971_197183


namespace NUMINAMATH_CALUDE_range_of_s_l1971_197197

-- Define a composite positive integer not divisible by 3
def IsCompositeNotDivisibleBy3 (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∃ k : ℕ, n = k * k) ∧ ¬ (3 ∣ n)

-- Define the function s
def s (n : ℕ) (h : IsCompositeNotDivisibleBy3 n) : ℕ :=
  sorry -- Implementation of s is not required for the statement

-- The main theorem
theorem range_of_s :
  ∀ m : ℤ, m > 3 ↔ ∃ (n : ℕ) (h : IsCompositeNotDivisibleBy3 n), s n h = m :=
sorry

end NUMINAMATH_CALUDE_range_of_s_l1971_197197
