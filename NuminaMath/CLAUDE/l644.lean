import Mathlib

namespace NUMINAMATH_CALUDE_binary_110_eq_6_l644_64445

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110_eq_6 :
  binary_to_decimal [true, true, false] = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_eq_6_l644_64445


namespace NUMINAMATH_CALUDE_customer_b_bought_five_units_l644_64414

/-- Represents the phone inventory and sales of a store -/
structure PhoneStore where
  total_units : ℕ
  defective_units : ℕ
  customer_a_units : ℕ
  customer_c_units : ℕ

/-- Calculates the number of units sold to Customer B -/
def units_sold_to_b (store : PhoneStore) : ℕ :=
  store.total_units - store.defective_units - store.customer_a_units - store.customer_c_units

/-- Theorem stating that Customer B bought 5 units -/
theorem customer_b_bought_five_units (store : PhoneStore) 
  (h1 : store.total_units = 20)
  (h2 : store.defective_units = 5)
  (h3 : store.customer_a_units = 3)
  (h4 : store.customer_c_units = 7) :
  units_sold_to_b store = 5 := by
  sorry

end NUMINAMATH_CALUDE_customer_b_bought_five_units_l644_64414


namespace NUMINAMATH_CALUDE_half_quarter_difference_l644_64428

theorem half_quarter_difference (n : ℝ) (h : n = 8) : 0.5 * n - 0.25 * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_half_quarter_difference_l644_64428


namespace NUMINAMATH_CALUDE_smallest_integer_l644_64427

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 44) :
  b ≥ 165 ∧ ∃ (b' : ℕ), b' = 165 ∧ Nat.lcm a b' / Nat.gcd a b' = 44 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l644_64427


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l644_64464

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to distribute 7 indistinguishable balls into 4 indistinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l644_64464


namespace NUMINAMATH_CALUDE_shop_prices_l644_64452

theorem shop_prices (x y : ℝ) 
  (sum_condition : x + y = 5)
  (retail_condition : 3 * (x + 1) + 2 * (2 * y - 1) = 19) :
  x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_shop_prices_l644_64452


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l644_64465

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℂ, 3 * x^2 + k * x + 18 = 0 ∧ x = 2 - 3*I) → k = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l644_64465


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l644_64458

theorem arithmetic_calculations :
  ((-15) - (-5) + 6 = -4) ∧
  (81 / (-9/5) * (5/9) = -25) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l644_64458


namespace NUMINAMATH_CALUDE_dye_per_dot_l644_64470

/-- The amount of dye per dot given the number of dots per blouse, 
    total amount of dye, and number of blouses -/
theorem dye_per_dot 
  (dots_per_blouse : ℕ) 
  (total_dye : ℕ) 
  (num_blouses : ℕ) 
  (h1 : dots_per_blouse = 20)
  (h2 : total_dye = 50 * 400)
  (h3 : num_blouses = 100) :
  total_dye / (dots_per_blouse * num_blouses) = 10 := by
  sorry

#check dye_per_dot

end NUMINAMATH_CALUDE_dye_per_dot_l644_64470


namespace NUMINAMATH_CALUDE_frank_remaining_money_l644_64478

def calculate_remaining_money (initial_amount : ℕ) 
                              (action_figure_cost : ℕ) (action_figure_count : ℕ)
                              (board_game_cost : ℕ) (board_game_count : ℕ)
                              (puzzle_set_cost : ℕ) (puzzle_set_count : ℕ) : ℕ :=
  initial_amount - 
  (action_figure_cost * action_figure_count + 
   board_game_cost * board_game_count + 
   puzzle_set_cost * puzzle_set_count)

theorem frank_remaining_money :
  calculate_remaining_money 100 12 3 11 2 6 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_frank_remaining_money_l644_64478


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l644_64448

theorem trig_expression_equals_negative_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l644_64448


namespace NUMINAMATH_CALUDE_integer_An_l644_64438

theorem integer_An (a b : ℕ+) (h1 : a > b) (θ : Real) 
  (h2 : 0 < θ) (h3 : θ < Real.pi / 2) 
  (h4 : Real.sin θ = (2 * a * b : ℝ) / ((a * a + b * b) : ℝ)) :
  ∀ n : ℕ, ∃ k : ℤ, (((a * a + b * b) : ℝ) ^ n * Real.sin (n * θ)) = k := by
  sorry

end NUMINAMATH_CALUDE_integer_An_l644_64438


namespace NUMINAMATH_CALUDE_hours_minutes_conversion_tons_kilograms_conversion_seconds_conversion_square_meters_conversion_l644_64423

-- Define conversion rates
def minutes_per_hour : ℕ := 60
def kilograms_per_ton : ℕ := 1000
def seconds_per_minute : ℕ := 60
def square_meters_per_hectare : ℕ := 10000

-- Define the conversion functions
def hours_minutes_to_minutes (hours minutes : ℕ) : ℕ :=
  hours * minutes_per_hour + minutes

def tons_kilograms_to_kilograms (tons kilograms : ℕ) : ℕ :=
  tons * kilograms_per_ton + kilograms

def seconds_to_minutes_seconds (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / seconds_per_minute, total_seconds % seconds_per_minute)

def square_meters_to_hectares (square_meters : ℕ) : ℕ :=
  square_meters / square_meters_per_hectare

-- State the theorems
theorem hours_minutes_conversion :
  hours_minutes_to_minutes 4 35 = 275 := by sorry

theorem tons_kilograms_conversion :
  tons_kilograms_to_kilograms 4 35 = 4035 := by sorry

theorem seconds_conversion :
  seconds_to_minutes_seconds 678 = (11, 18) := by sorry

theorem square_meters_conversion :
  square_meters_to_hectares 120000 = 12 := by sorry

end NUMINAMATH_CALUDE_hours_minutes_conversion_tons_kilograms_conversion_seconds_conversion_square_meters_conversion_l644_64423


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l644_64474

theorem gcd_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 32515 * k) →
  Int.gcd ((3*x+5)*(5*x+3)*(11*x+7)*(x+17)) x = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l644_64474


namespace NUMINAMATH_CALUDE_circle_and_max_z_l644_64437

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Define the function z
def z (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

-- Theorem statement
theorem circle_and_max_z :
  (∀ p : ℝ × ℝ, p ∈ circle_C → (p.1 = 1 ∧ p.2 = 4) ∨ (p.1 = 3 ∧ p.2 = 2)) ∧
  (∃ c : ℝ × ℝ, c ∈ circle_C ∧ center_line c.1 c.2) →
  (∀ p : ℝ × ℝ, p ∈ circle_C → (p.1 - 1)^2 + (p.2 - 2)^2 = 4) ∧
  (∀ p : ℝ × ℝ, p ∈ circle_C → z p ≤ 3 + 2 * Real.sqrt 2) ∧
  (∃ p : ℝ × ℝ, p ∈ circle_C ∧ z p = 3 + 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_max_z_l644_64437


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l644_64459

-- Define the total number of team members
def total_members : ℕ := 18

-- Define the number of players in the starting lineup
def lineup_size : ℕ := 8

-- Define the number of interchangeable positions
def interchangeable_positions : ℕ := 6

-- Theorem statement
theorem volleyball_lineup_combinations :
  (total_members) *
  (total_members - 1) *
  (Nat.choose (total_members - 2) interchangeable_positions) =
  2448272 := by sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l644_64459


namespace NUMINAMATH_CALUDE_beach_shells_problem_l644_64479

theorem beach_shells_problem (jillian_shells savannah_shells clayton_shells : ℕ) 
  (friend_count friend_received : ℕ) :
  jillian_shells = 29 →
  clayton_shells = 8 →
  friend_count = 2 →
  friend_received = 27 →
  jillian_shells + savannah_shells + clayton_shells = friend_count * friend_received →
  savannah_shells = 17 := by
  sorry

end NUMINAMATH_CALUDE_beach_shells_problem_l644_64479


namespace NUMINAMATH_CALUDE_characterize_valid_functions_l644_64451

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x, f x ≤ x^2) ∧
  (∀ x y, x > y → (x - y) ∣ (f x - f y))

theorem characterize_valid_functions :
  ∀ f : ℕ → ℕ, is_valid_function f ↔
    (∀ x, f x = 0) ∨
    (∀ x, f x = x) ∨
    (∀ x, f x = x^2 - x) ∨
    (∀ x, f x = x^2) :=
sorry


end NUMINAMATH_CALUDE_characterize_valid_functions_l644_64451


namespace NUMINAMATH_CALUDE_perfect_square_sum_l644_64426

theorem perfect_square_sum (a b : ℤ) : 
  (∃ x : ℤ, a^4 + (a+b)^4 + b^4 = x^2) ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l644_64426


namespace NUMINAMATH_CALUDE_polynomial_simplification_l644_64435

theorem polynomial_simplification (p q : ℝ) :
  (4 * q^4 + 2 * p^3 - 7 * p + 8) + (3 * q^4 - 2 * p^3 + 3 * p^2 - 5 * p + 6) =
  7 * q^4 + 3 * p^2 - 12 * p + 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l644_64435


namespace NUMINAMATH_CALUDE_simplify_absolute_value_expression_l644_64412

theorem simplify_absolute_value_expression
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y < 0)
  (hz : z < 0)
  (hxy : abs x > abs y)
  (hzx : abs z > abs x) :
  abs (x + z) - abs (y + z) - abs (x + y) = -2 * x :=
by sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_expression_l644_64412


namespace NUMINAMATH_CALUDE_flea_difference_l644_64466

def flea_treatment (initial_fleas : ℕ) (treatments : ℕ) : ℕ :=
  initial_fleas / (2^treatments)

theorem flea_difference (initial_fleas : ℕ) :
  flea_treatment initial_fleas 4 = 14 →
  initial_fleas - flea_treatment initial_fleas 4 = 210 := by
sorry

end NUMINAMATH_CALUDE_flea_difference_l644_64466


namespace NUMINAMATH_CALUDE_perez_class_cans_collected_l644_64480

/-- Calculates the total number of cans collected by a class during a food drive. -/
def totalCansCollected (totalStudents : ℕ) (halfStudentsCans : ℕ) (nonCollectingStudents : ℕ) (remainingStudentsCans : ℕ) : ℕ :=
  let halfStudents := totalStudents / 2
  let remainingStudents := totalStudents - halfStudents - nonCollectingStudents
  halfStudents * halfStudentsCans + remainingStudents * remainingStudentsCans

/-- Proves that Ms. Perez's class collected 232 cans in total. -/
theorem perez_class_cans_collected :
  totalCansCollected 30 12 2 4 = 232 := by
  sorry

#eval totalCansCollected 30 12 2 4

end NUMINAMATH_CALUDE_perez_class_cans_collected_l644_64480


namespace NUMINAMATH_CALUDE_organization_member_count_l644_64436

/-- Represents an organization with committees and members -/
structure Organization where
  num_committees : Nat
  num_members : Nat
  member_committee_count : Nat
  pair_common_member_count : Nat

/-- The specific organization described in the problem -/
def specific_org : Organization :=
  { num_committees := 5
  , num_members := 10
  , member_committee_count := 2
  , pair_common_member_count := 1
  }

/-- Theorem stating that the organization with the given properties has 10 members -/
theorem organization_member_count :
  ∀ (org : Organization),
    org.num_committees = 5 ∧
    org.member_committee_count = 2 ∧
    org.pair_common_member_count = 1 →
    org.num_members = 10 := by
  sorry

#check organization_member_count

end NUMINAMATH_CALUDE_organization_member_count_l644_64436


namespace NUMINAMATH_CALUDE_determine_sanity_with_one_question_l644_64440

-- Define the types
inductive Species : Type
| Human
| Vampire

inductive MentalState : Type
| Sane
| Insane

-- Define the Transylvanian type
structure Transylvanian :=
  (species : Species)
  (mental_state : MentalState)

-- Define the question type
inductive Question : Type
| AreYouAPerson

-- Define the answer type
inductive Answer : Type
| Yes
| No

-- Define the response function
def respond (t : Transylvanian) (q : Question) : Answer :=
  match t.mental_state, q with
  | MentalState.Sane, Question.AreYouAPerson => Answer.Yes
  | MentalState.Insane, Question.AreYouAPerson => Answer.No

-- Theorem statement
theorem determine_sanity_with_one_question :
  ∃ (q : Question), ∀ (t : Transylvanian),
    (respond t q = Answer.Yes ↔ t.mental_state = MentalState.Sane) ∧
    (respond t q = Answer.No ↔ t.mental_state = MentalState.Insane) :=
by sorry

end NUMINAMATH_CALUDE_determine_sanity_with_one_question_l644_64440


namespace NUMINAMATH_CALUDE_divisibility_by_480_l644_64476

theorem divisibility_by_480 (n : ℤ) 
  (h2 : ¬ 2 ∣ n) 
  (h3 : ¬ 3 ∣ n) 
  (h5 : ¬ 5 ∣ n) : 
  480 ∣ (n^8 - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_480_l644_64476


namespace NUMINAMATH_CALUDE_standard_deviation_reflects_fluctuation_amplitude_l644_64447

/-- Standard deviation of a sample -/
def standard_deviation (sample : List ℝ) : ℝ := sorry

/-- Fluctuation amplitude of a population -/
def fluctuation_amplitude (population : List ℝ) : ℝ := sorry

/-- The standard deviation of a sample approximately reflects 
    the fluctuation amplitude of a population -/
theorem standard_deviation_reflects_fluctuation_amplitude 
  (sample : List ℝ) (population : List ℝ) :
  ∃ (ε : ℝ), ε > 0 ∧ |standard_deviation sample - fluctuation_amplitude population| < ε :=
sorry

end NUMINAMATH_CALUDE_standard_deviation_reflects_fluctuation_amplitude_l644_64447


namespace NUMINAMATH_CALUDE_exponent_multiplication_l644_64463

theorem exponent_multiplication (a : ℝ) : a^2 * a^6 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l644_64463


namespace NUMINAMATH_CALUDE_surface_area_of_specific_solid_l644_64408

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge : String

/-- The solid formed by slicing off the top part of the prism -/
structure SlicedSolid where
  prism : RightPrism
  x : Midpoint
  y : Midpoint
  z : Midpoint

/-- Calculate the surface area of the sliced solid -/
noncomputable def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the specific sliced solid -/
theorem surface_area_of_specific_solid :
  let prism := RightPrism.mk 20 10
  let x := Midpoint.mk "AC"
  let y := Midpoint.mk "BC"
  let z := Midpoint.mk "DF"
  let solid := SlicedSolid.mk prism x y z
  surface_area solid = 100 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 418.75) / 2 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_solid_l644_64408


namespace NUMINAMATH_CALUDE_remainder_of_2_pow_1999_plus_1_mod_17_l644_64420

theorem remainder_of_2_pow_1999_plus_1_mod_17 :
  (2^1999 + 1) % 17 = 10 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2_pow_1999_plus_1_mod_17_l644_64420


namespace NUMINAMATH_CALUDE_ice_cube_calculation_l644_64449

theorem ice_cube_calculation (cubes_per_tray : ℕ) (num_trays : ℕ) 
  (h1 : cubes_per_tray = 9) 
  (h2 : num_trays = 8) : 
  cubes_per_tray * num_trays = 72 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_calculation_l644_64449


namespace NUMINAMATH_CALUDE_ten_circles_l644_64434

/-- The maximum number of intersection points for n circles -/
def max_intersection_points (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * n

/-- Given conditions -/
axiom two_circles : max_intersection_points 2 = 2
axiom three_circles : max_intersection_points 3 = 6

/-- Theorem to prove -/
theorem ten_circles : max_intersection_points 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ten_circles_l644_64434


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l644_64425

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_function_sum (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g) 
  (h_sum : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := by
sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l644_64425


namespace NUMINAMATH_CALUDE_second_wave_infections_l644_64411

/-- Calculates the total number of infections over a given period, given an initial daily rate and an increase factor. -/
def totalInfections (initialRate : ℕ) (increaseFactor : ℕ) (days : ℕ) : ℕ :=
  (initialRate + initialRate * increaseFactor) * days

/-- Theorem stating that given an initial infection rate of 300 per day, 
    with a 4-fold increase in daily infections, the total number of 
    infections over a 14-day period is 21000. -/
theorem second_wave_infections : 
  totalInfections 300 4 14 = 21000 := by
  sorry

end NUMINAMATH_CALUDE_second_wave_infections_l644_64411


namespace NUMINAMATH_CALUDE_necklace_count_l644_64497

/-- The number of unique necklaces made from 5 red and 2 blue beads -/
def unique_necklaces : ℕ := 3

/-- The number of red beads in each necklace -/
def red_beads : ℕ := 5

/-- The number of blue beads in each necklace -/
def blue_beads : ℕ := 2

/-- The total number of beads in each necklace -/
def total_beads : ℕ := red_beads + blue_beads

/-- Theorem stating that the number of unique necklaces is 3 -/
theorem necklace_count : unique_necklaces = 3 := by sorry

end NUMINAMATH_CALUDE_necklace_count_l644_64497


namespace NUMINAMATH_CALUDE_find_m_l644_64400

theorem find_m : ∃ m : ℚ, m * 9999 = 624877405 ∧ m = 62493.5 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l644_64400


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l644_64461

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 675 →
  breadth = 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l644_64461


namespace NUMINAMATH_CALUDE_min_value_of_f_l644_64416

noncomputable def f (x : ℝ) := 12 * x - x^3

theorem min_value_of_f :
  ∃ (m : ℝ), m = -16 ∧ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l644_64416


namespace NUMINAMATH_CALUDE_unique_number_satisfying_condition_l644_64410

theorem unique_number_satisfying_condition : ∃! x : ℕ, 143 - 10 * x = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_condition_l644_64410


namespace NUMINAMATH_CALUDE_least_number_divisibility_l644_64486

theorem least_number_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 9 * k) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m = 9 * k)) ∧
  (∃ r : ℕ, r < 5 ∧ r < 6 ∧ r < 7 ∧ r < 8 ∧
    n % 5 = r ∧ n % 6 = r ∧ n % 7 = r ∧ n % 8 = r) ∧
  n = 1680 →
  n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l644_64486


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l644_64457

theorem circle_diameter_ratio (D C : Real) (shaded_ratio : Real) :
  D = 24 →  -- Diameter of circle D
  C < D →   -- Circle C is inside circle D
  shaded_ratio = 7 →  -- Ratio of shaded area to area of circle C
  C = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l644_64457


namespace NUMINAMATH_CALUDE_value_of_a_l644_64424

theorem value_of_a (a b c : ℚ) 
  (eq1 : a + b = c)
  (eq2 : b + c + 2 * b = 11)
  (eq3 : c = 7) :
  a = 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l644_64424


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_seven_largest_n_is_99996_l644_64494

theorem largest_n_multiple_of_seven (n : ℕ) : n < 100000 →
  (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0 →
  n ≤ 99996 :=
by sorry

theorem largest_n_is_99996 :
  (9 * (99996 - 3)^6 - 99996^3 + 16 * 99996 - 27) % 7 = 0 ∧
  99996 < 100000 ∧
  ∀ m : ℕ, m < 100000 →
    (9 * (m - 3)^6 - m^3 + 16 * m - 27) % 7 = 0 →
    m ≤ 99996 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_seven_largest_n_is_99996_l644_64494


namespace NUMINAMATH_CALUDE_jack_cookies_needed_l644_64430

/-- Represents the sales data and goals for Jack's bake sale -/
structure BakeSale where
  brownies_sold : Nat
  brownies_price : Nat
  lemon_squares_sold : Nat
  lemon_squares_price : Nat
  cookie_price : Nat
  bulk_pack_size : Nat
  bulk_pack_price : Nat
  sales_goal : Nat

/-- Calculates the minimum number of cookies needed to reach the sales goal -/
def min_cookies_needed (sale : BakeSale) : Nat :=
  sorry

/-- Theorem stating that Jack needs to sell 8 cookies to reach his goal -/
theorem jack_cookies_needed (sale : BakeSale) 
  (h1 : sale.brownies_sold = 4)
  (h2 : sale.brownies_price = 3)
  (h3 : sale.lemon_squares_sold = 5)
  (h4 : sale.lemon_squares_price = 2)
  (h5 : sale.cookie_price = 4)
  (h6 : sale.bulk_pack_size = 5)
  (h7 : sale.bulk_pack_price = 17)
  (h8 : sale.sales_goal = 50) :
  min_cookies_needed sale = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_cookies_needed_l644_64430


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l644_64419

theorem six_digit_divisibility (a b c d e f : ℕ) 
  (h_six_digit : 100000 ≤ a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧ 
                 a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f < 1000000)
  (h_sum_equal : a + d = b + e ∧ b + e = c + f) : 
  ∃ k : ℕ, a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f = 37 * k :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l644_64419


namespace NUMINAMATH_CALUDE_school_sections_l644_64481

theorem school_sections (num_boys num_girls : ℕ) (h1 : num_boys = 408) (h2 : num_girls = 240) :
  let section_size := Nat.gcd num_boys num_girls
  let boys_sections := num_boys / section_size
  let girls_sections := num_girls / section_size
  boys_sections + girls_sections = 27 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l644_64481


namespace NUMINAMATH_CALUDE_train_length_is_400_l644_64482

/-- Calculates the length of a train given its speed, the speed and length of a platform
    moving in the opposite direction, and the time taken to cross the platform. -/
def trainLength (trainSpeed : ℝ) (platformSpeed : ℝ) (platformLength : ℝ) (crossingTime : ℝ) : ℝ :=
  (trainSpeed + platformSpeed) * crossingTime - platformLength

/-- Theorem stating that under the given conditions, the train length is 400 meters. -/
theorem train_length_is_400 :
  let trainSpeed : ℝ := 20
  let platformSpeed : ℝ := 5
  let platformLength : ℝ := 250
  let crossingTime : ℝ := 26
  trainLength trainSpeed platformSpeed platformLength crossingTime = 400 := by
sorry

end NUMINAMATH_CALUDE_train_length_is_400_l644_64482


namespace NUMINAMATH_CALUDE_chalkboard_width_l644_64404

theorem chalkboard_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 2 * width →
  area = width * length →
  area = 18 →
  width = 3 :=
by sorry

end NUMINAMATH_CALUDE_chalkboard_width_l644_64404


namespace NUMINAMATH_CALUDE_total_chicken_pieces_l644_64472

-- Define the number of chicken pieces per order type
def chicken_pasta_pieces : ℕ := 2
def barbecue_chicken_pieces : ℕ := 3
def fried_chicken_dinner_pieces : ℕ := 8

-- Define the number of orders for each type
def fried_chicken_dinner_orders : ℕ := 2
def chicken_pasta_orders : ℕ := 6
def barbecue_chicken_orders : ℕ := 3

-- Theorem stating the total number of chicken pieces needed
theorem total_chicken_pieces :
  fried_chicken_dinner_orders * fried_chicken_dinner_pieces +
  chicken_pasta_orders * chicken_pasta_pieces +
  barbecue_chicken_orders * barbecue_chicken_pieces = 37 :=
by sorry

end NUMINAMATH_CALUDE_total_chicken_pieces_l644_64472


namespace NUMINAMATH_CALUDE_seconds_in_hours_l644_64492

theorem seconds_in_hours : 
  (∀ (hours : ℝ), hours * 60 * 60 = hours * 3600) →
  3.5 * 3600 = 12600 := by sorry

end NUMINAMATH_CALUDE_seconds_in_hours_l644_64492


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l644_64442

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l644_64442


namespace NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l644_64429

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the 20th and 30th triangular numbers is 675 -/
theorem sum_of_20th_and_30th_triangular : triangular_number 20 + triangular_number 30 = 675 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l644_64429


namespace NUMINAMATH_CALUDE_dads_strawberries_weight_l644_64467

/-- The weight of Marco's dad's strawberries -/
def dads_strawberries (total_weight marco_weight : ℕ) : ℕ :=
  total_weight - marco_weight

/-- Theorem stating that Marco's dad's strawberries weigh 9 pounds -/
theorem dads_strawberries_weight :
  dads_strawberries 23 14 = 9 := by
  sorry

end NUMINAMATH_CALUDE_dads_strawberries_weight_l644_64467


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l644_64456

/-- Given two lines that intersect at a specific point, prove the sum of their y-intercepts. -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∀ x y : ℝ, x = 3 * y + a ∧ y = 3 * x + b → x = 4 ∧ y = 1) →
  a + b = -10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l644_64456


namespace NUMINAMATH_CALUDE_scrap_cookie_radius_l644_64491

theorem scrap_cookie_radius 
  (R : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h1 : R = 3.5) 
  (h2 : r = 1) 
  (h3 : n = 9) : 
  ∃ (x : ℝ), x^2 = R^2 * π - n * r^2 * π ∧ x = Real.sqrt 3.25 := by
  sorry

end NUMINAMATH_CALUDE_scrap_cookie_radius_l644_64491


namespace NUMINAMATH_CALUDE_line_intersection_l644_64462

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ (c : ℝ), l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The problem statement -/
theorem line_intersection (p : ℝ) : 
  let line1 : Line2D := ⟨(2, 3), (5, -8)⟩
  let line2 : Line2D := ⟨(-1, 4), (3, p)⟩
  parallel line1 line2 → p = -24/5 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l644_64462


namespace NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l644_64485

-- Define variables
variable (x y : ℝ)

-- Theorem 1
theorem combine_like_terms_1 : 2*x - (x - y) + (x + y) = 2*x + 2*y := by
  sorry

-- Theorem 2
theorem combine_like_terms_2 : 3*x^2 - 9*x + 2 - x^2 + 4*x - 6 = 2*x^2 - 5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l644_64485


namespace NUMINAMATH_CALUDE_two_polygons_edges_l644_64441

theorem two_polygons_edges (a b : ℕ) : 
  a + b = 2014 →
  a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053 →
  a ≤ b →
  a = 952 := by sorry

end NUMINAMATH_CALUDE_two_polygons_edges_l644_64441


namespace NUMINAMATH_CALUDE_problem_statement_l644_64495

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ((a - 1) * (b - 1) = 1) ∧
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 4*b ≤ x + 4*y) ∧
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y = 1 → 1/a^2 + 2/b^2 ≤ 1/x^2 + 2/y^2) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ a + 4*b = x + 4*y ∧ a + 4*b = 9) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 ∧ 1/a^2 + 2/b^2 = 1/x^2 + 2/y^2 ∧ 1/a^2 + 2/b^2 = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l644_64495


namespace NUMINAMATH_CALUDE_alster_frogs_l644_64475

theorem alster_frogs (alster quinn bret : ℕ) 
  (h1 : quinn = 2 * alster)
  (h2 : bret = 3 * quinn)
  (h3 : bret = 12) :
  alster = 2 := by
sorry

end NUMINAMATH_CALUDE_alster_frogs_l644_64475


namespace NUMINAMATH_CALUDE_solve_and_prove_l644_64406

-- Given that |x+a| ≤ b has the solution set [-6, 2]
def has_solution_set (a b : ℝ) : Prop :=
  ∀ x, |x + a| ≤ b ↔ -6 ≤ x ∧ x ≤ 2

-- Define the conditions |am+n| < 1/3 and |m-bn| < 1/6
def conditions (a b m n : ℝ) : Prop :=
  |a * m + n| < 1/3 ∧ |m - b * n| < 1/6

theorem solve_and_prove (a b m n : ℝ) 
  (h1 : has_solution_set a b) 
  (h2 : conditions a b m n) : 
  (a = 2 ∧ b = 4) ∧ |n| < 2/27 :=
sorry

end NUMINAMATH_CALUDE_solve_and_prove_l644_64406


namespace NUMINAMATH_CALUDE_annas_number_l644_64407

theorem annas_number : ∃ x : ℚ, 5 * ((3 * x + 20) - 5) = 200 ∧ x = 25 / 3 := by sorry

end NUMINAMATH_CALUDE_annas_number_l644_64407


namespace NUMINAMATH_CALUDE_count_scalene_triangles_l644_64431

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c < 13 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem count_scalene_triangles :
  ∃! (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2) ∧
    S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_scalene_triangles_l644_64431


namespace NUMINAMATH_CALUDE_distance_traveled_correct_mrs_hilt_trip_distance_l644_64473

/-- Calculates the distance traveled given initial and final odometer readings -/
def distance_traveled (initial_reading final_reading : ℝ) : ℝ :=
  final_reading - initial_reading

/-- Theorem: The distance traveled is the difference between final and initial odometer readings -/
theorem distance_traveled_correct (initial_reading final_reading : ℝ) :
  distance_traveled initial_reading final_reading = final_reading - initial_reading :=
by sorry

/-- Mrs. Hilt's trip distance calculation -/
theorem mrs_hilt_trip_distance :
  distance_traveled 212.3 372 = 159.7 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_correct_mrs_hilt_trip_distance_l644_64473


namespace NUMINAMATH_CALUDE_inequality_solution_set_f_less_than_one_l644_64453

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem 1: Solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 2 - |x + 1|} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2/3} := by sorry

-- Theorem 2: Proof that f(x) < 1 under given conditions
theorem f_less_than_one (x y : ℝ) (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) :
  f x < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_f_less_than_one_l644_64453


namespace NUMINAMATH_CALUDE_unique_six_digit_number_divisibility_l644_64409

def is_valid_digit (d : Nat) : Prop := d ≥ 1 ∧ d ≤ 6

def all_digits_unique (p q r s t u : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧
  r ≠ s ∧ r ≠ t ∧ r ≠ u ∧
  s ≠ t ∧ s ≠ u ∧
  t ≠ u

def three_digit_number (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

theorem unique_six_digit_number_divisibility (p q r s t u : Nat) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧
  is_valid_digit s ∧ is_valid_digit t ∧ is_valid_digit u ∧
  all_digits_unique p q r s t u ∧
  (three_digit_number p q r) % 4 = 0 ∧
  (three_digit_number q r s) % 6 = 0 ∧
  (three_digit_number r s t) % 3 = 0 →
  p = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_divisibility_l644_64409


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l644_64421

theorem system_of_equations_solution :
  ∃! (x y : ℝ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 ∧ x = 6 ∧ y = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l644_64421


namespace NUMINAMATH_CALUDE_dot_product_theorem_l644_64422

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_theorem : (2 • a + b) • a = 1 := by sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l644_64422


namespace NUMINAMATH_CALUDE_football_cost_l644_64405

/-- The cost of a football given the total cost of a football and baseball, and the cost of the baseball. -/
theorem football_cost (total_cost baseball_cost : ℚ) : 
  total_cost = 20 - (4 + 5/100) → 
  baseball_cost = 6 + 81/100 → 
  total_cost - baseball_cost = 9 + 14/100 := by
sorry

end NUMINAMATH_CALUDE_football_cost_l644_64405


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l644_64477

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Main theorem
theorem fibonacci_divisibility (A B h k : ℕ) : 
  A > 0 → B > 0 → 
  (∃ m : ℕ, B^93 = m * A^19) →
  (∃ n : ℕ, A^93 = n * B^19) →
  (∃ i : ℕ, h = fib i ∧ k = fib (i + 1)) →
  (∃ p : ℕ, (A^4 + B^8)^k = p * (A * B)^h) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l644_64477


namespace NUMINAMATH_CALUDE_smallest_n_cookies_l644_64439

theorem smallest_n_cookies (n : ℕ) : (∃ k : ℕ, 15 * n - 1 = 11 * k) ↔ n ≥ 3 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_cookies_l644_64439


namespace NUMINAMATH_CALUDE_expected_heads_is_60_l644_64454

/-- The number of coins -/
def num_coins : ℕ := 64

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The number of possible tosses for each coin -/
def max_tosses : ℕ := 4

/-- The probability of a coin showing heads after up to four tosses -/
def prob_heads_after_four : ℚ :=
  p_heads + (1 - p_heads) * p_heads + 
  (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after the series of tosses -/
def expected_heads : ℚ := num_coins * prob_heads_after_four

theorem expected_heads_is_60 : expected_heads = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_is_60_l644_64454


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l644_64415

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (1 / (x + y)) + ((x + y) / z) ≥ 3 := by
  sorry

theorem min_value_achieved (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
  x' + y' + z' = 1 ∧ 
  (1 / (x' + y')) + ((x' + y') / z') = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achieved_l644_64415


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l644_64490

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the second term of the sequence is 3. -/
theorem second_term_of_geometric_series (a : ℝ) :
  (∑' n, a * (1/4)^n = 16) → a * (1/4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l644_64490


namespace NUMINAMATH_CALUDE_negation_of_proposition_l644_64460

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l644_64460


namespace NUMINAMATH_CALUDE_no_real_solutions_l644_64433

/-- 
Theorem: The system x^3 + y^3 = 2 and y = kx + d has no real solutions (x,y) 
if and only if k = -1 and 0 < d < 2√2.
-/
theorem no_real_solutions (k d : ℝ) : 
  (∀ x y : ℝ, x^3 + y^3 ≠ 2 ∨ y ≠ k*x + d) ↔ (k = -1 ∧ 0 < d ∧ d < 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_no_real_solutions_l644_64433


namespace NUMINAMATH_CALUDE_courier_net_pay_rate_l644_64402

def travel_time : ℝ := 3
def speed : ℝ := 65
def fuel_efficiency : ℝ := 28
def payment_rate : ℝ := 0.55
def gasoline_cost : ℝ := 2.50

theorem courier_net_pay_rate : 
  let total_distance := travel_time * speed
  let gasoline_used := total_distance / fuel_efficiency
  let earnings := payment_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := earnings - gasoline_expense
  let net_rate_per_hour := net_earnings / travel_time
  ⌊net_rate_per_hour⌋ = 30 := by sorry

end NUMINAMATH_CALUDE_courier_net_pay_rate_l644_64402


namespace NUMINAMATH_CALUDE_mehki_age_l644_64496

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove Mehki's age -/
theorem mehki_age (zrinka jordyn mehki : ℕ) 
  (h1 : mehki = jordyn + 10)
  (h2 : jordyn = 2 * zrinka)
  (h3 : zrinka = 6) :
  mehki = 22 := by sorry

end NUMINAMATH_CALUDE_mehki_age_l644_64496


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_right_angle_l644_64498

theorem triangle_angle_ratio_right_angle (α β γ : ℝ) (h_sum : α + β + γ = π) 
  (h_ratio : α = 3 * γ ∧ β = 2 * γ) : α = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_right_angle_l644_64498


namespace NUMINAMATH_CALUDE_imaginary_unit_seventh_power_l644_64487

theorem imaginary_unit_seventh_power :
  ∀ i : ℂ, i^2 = -1 → i^7 = -i :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_seventh_power_l644_64487


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l644_64403

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 8 = 0 → 
  x₂^2 - 14*x₂ + 8 = 0 → 
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = 7/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l644_64403


namespace NUMINAMATH_CALUDE_twenty_two_percent_of_300_prove_twenty_two_percent_of_300_l644_64493

theorem twenty_two_percent_of_300 : ℝ → Prop :=
  fun result => (22 / 100 : ℝ) * 300 = result

theorem prove_twenty_two_percent_of_300 : twenty_two_percent_of_300 66 := by
  sorry

end NUMINAMATH_CALUDE_twenty_two_percent_of_300_prove_twenty_two_percent_of_300_l644_64493


namespace NUMINAMATH_CALUDE_mother_age_four_times_yujeong_age_l644_64499

/-- Represents the age difference between the current year and the year in question -/
def yearDifference : ℕ := 2

/-- Yujeong's current age -/
def yujeongCurrentAge : ℕ := 12

/-- Yujeong's mother's current age -/
def motherCurrentAge : ℕ := 42

/-- Theorem stating that 2 years ago, Yujeong's mother's age was 4 times Yujeong's age -/
theorem mother_age_four_times_yujeong_age :
  (motherCurrentAge - yearDifference) = 4 * (yujeongCurrentAge - yearDifference) := by
  sorry

end NUMINAMATH_CALUDE_mother_age_four_times_yujeong_age_l644_64499


namespace NUMINAMATH_CALUDE_system_properties_l644_64450

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x + 3 * y = 4 - a ∧ x - y = 3 * a

-- Define the statements to be proven
theorem system_properties :
  -- Statement 1
  (∃ x y : ℝ, system x y 2 ∧ x = 5 ∧ y = -1) ∧
  -- Statement 2
  (∃ x y : ℝ, system x y (-2) ∧ x = -y) ∧
  -- Statement 3
  (∀ x y a : ℝ, system x y a → x + 2 * y = 3) ∧
  -- Statement 4
  (∃ x y : ℝ, system x y (-1) ∧ x + y ≠ 4 - (-1)) :=
by sorry

end NUMINAMATH_CALUDE_system_properties_l644_64450


namespace NUMINAMATH_CALUDE_rectangle_13_squares_ratio_l644_64444

/-- A rectangle that can be divided into 13 equal squares -/
structure Rectangle13Squares where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_divisible : ∃ s : ℝ, 0 < s ∧ (a = 13 * s ∧ b = s) ∨ (a = s ∧ b = 13 * s)

/-- The ratio of the longer side to the shorter side is 13:1 -/
theorem rectangle_13_squares_ratio (rect : Rectangle13Squares) :
  (max rect.a rect.b) / (min rect.a rect.b) = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_13_squares_ratio_l644_64444


namespace NUMINAMATH_CALUDE_tyson_basketball_score_l644_64484

theorem tyson_basketball_score (three_point_shots two_point_shots one_point_shots : ℕ) 
  (h1 : three_point_shots = 15)
  (h2 : two_point_shots = 12)
  (h3 : 3 * three_point_shots + 2 * two_point_shots + one_point_shots = 75) :
  one_point_shots = 6 := by
  sorry

end NUMINAMATH_CALUDE_tyson_basketball_score_l644_64484


namespace NUMINAMATH_CALUDE_triangle_formation_l644_64471

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation :
  can_form_triangle 4 4 7 ∧
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 5 8 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l644_64471


namespace NUMINAMATH_CALUDE_scientific_notation_218_million_l644_64417

theorem scientific_notation_218_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    218000000 = a * (10 : ℝ) ^ n ∧
    a = 2.18 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_218_million_l644_64417


namespace NUMINAMATH_CALUDE_intersection_of_sets_l644_64455

theorem intersection_of_sets (a : ℝ) : 
  let A : Set ℝ := {-a, a^2, a^2 + a}
  let B : Set ℝ := {-1, -1 - a, 1 + a^2}
  (A ∩ B).Nonempty → A ∩ B = {-1, 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l644_64455


namespace NUMINAMATH_CALUDE_sand_gravel_transport_l644_64469

theorem sand_gravel_transport :
  ∃ (x y : ℕ), 3 * x + 5 * y = 20 ∧ ((x = 5 ∧ y = 1) ∨ (x = 0 ∧ y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_sand_gravel_transport_l644_64469


namespace NUMINAMATH_CALUDE_function_identically_zero_l644_64418

/-- A function satisfying f(a · b) = a f(b) + b f(a) and |f(x)| ≤ 1 is identically zero -/
theorem function_identically_zero (f : ℝ → ℝ) 
  (h1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) 
  (h2 : ∀ x : ℝ, |f x| ≤ 1) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_identically_zero_l644_64418


namespace NUMINAMATH_CALUDE_olivia_wallet_remainder_l644_64446

/-- Calculates the remaining money in Olivia's wallet after visiting the supermarket. -/
def remaining_money (initial : ℕ) (collected : ℕ) (spent : ℕ) : ℕ :=
  initial + collected - spent

/-- Proves that given the initial amount, collected amount, and spent amount,
    the remaining money in Olivia's wallet is 159 dollars. -/
theorem olivia_wallet_remainder :
  remaining_money 100 148 89 = 159 := by
  sorry

end NUMINAMATH_CALUDE_olivia_wallet_remainder_l644_64446


namespace NUMINAMATH_CALUDE_three_digit_powers_of_three_l644_64443

theorem three_digit_powers_of_three :
  (∃! (s : Finset ℕ), s = {n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999} ∧ Finset.card s = 2) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_three_l644_64443


namespace NUMINAMATH_CALUDE_garden_shape_is_square_l644_64483

theorem garden_shape_is_square (cabbages_this_year : ℕ) (cabbage_increase : ℕ) 
  (h1 : cabbages_this_year = 11236)
  (h2 : cabbage_increase = 211)
  (h3 : ∃ (n : ℕ), n ^ 2 = cabbages_this_year)
  (h4 : ∃ (m : ℕ), m ^ 2 = cabbages_this_year - cabbage_increase) :
  ∃ (side : ℕ), side ^ 2 = cabbages_this_year := by
  sorry

end NUMINAMATH_CALUDE_garden_shape_is_square_l644_64483


namespace NUMINAMATH_CALUDE_water_consumption_percentage_difference_l644_64488

theorem water_consumption_percentage_difference : 
  let yesterday_consumption : ℝ := 48
  let two_days_ago_consumption : ℝ := 50
  let difference := two_days_ago_consumption - yesterday_consumption
  let percentage_difference := (difference / two_days_ago_consumption) * 100
  percentage_difference = 4 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_percentage_difference_l644_64488


namespace NUMINAMATH_CALUDE_cos_identity_l644_64468

theorem cos_identity : 
  (2 * (Real.cos (15 * π / 180))^2 - Real.cos (30 * π / 180) = 1) :=
by
  have h : Real.cos (30 * π / 180) = 2 * (Real.cos (15 * π / 180))^2 - 1 := by sorry
  sorry

end NUMINAMATH_CALUDE_cos_identity_l644_64468


namespace NUMINAMATH_CALUDE_x_intercept_is_two_l644_64432

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 2 -/
theorem x_intercept_is_two :
  let l : Line := { x₁ := 1, y₁ := -2, x₂ := 5, y₂ := 6 }
  xIntercept l = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_two_l644_64432


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l644_64489

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  first : ℚ
  diff : ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (a : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * a.first + (n - 1) * a.diff)

/-- The nth term of an arithmetic sequence -/
def nth_term (a : ArithmeticSequence) (n : ℕ) : ℚ :=
  a.first + (n - 1) * a.diff

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n_terms a n / sum_n_terms b n = n / (n + 1)) →
  nth_term a 4 / nth_term b 4 = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l644_64489


namespace NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l644_64401

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem folded_paper_perimeter_ratio :
  let original_side : ℝ := 8
  let large_rect : Rectangle := { width := original_side, height := original_side / 2 }
  let small_rect : Rectangle := { width := original_side / 2, height := original_side / 2 }
  (perimeter small_rect) / (perimeter large_rect) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l644_64401


namespace NUMINAMATH_CALUDE_evaluate_expression_l644_64413

theorem evaluate_expression : 6 - 9 * (10 - 4^2) * 5 = -264 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l644_64413
