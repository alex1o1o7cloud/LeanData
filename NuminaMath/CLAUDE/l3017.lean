import Mathlib

namespace NUMINAMATH_CALUDE_santiago_garrett_rose_difference_l3017_301776

/-- Mrs. Santiago has 58 red roses and Mrs. Garrett has 24 red roses. 
    The theorem proves that Mrs. Santiago has 34 more red roses than Mrs. Garrett. -/
theorem santiago_garrett_rose_difference :
  ∀ (santiago_roses garrett_roses : ℕ),
    santiago_roses = 58 →
    garrett_roses = 24 →
    santiago_roses - garrett_roses = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_santiago_garrett_rose_difference_l3017_301776


namespace NUMINAMATH_CALUDE_conference_handshakes_l3017_301741

/-- The number of handshakes in a conference of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 12 people where each person shakes hands
    exactly once with every other person, there are 66 handshakes. -/
theorem conference_handshakes :
  handshakes 12 = 66 := by
  sorry

#eval handshakes 12

end NUMINAMATH_CALUDE_conference_handshakes_l3017_301741


namespace NUMINAMATH_CALUDE_vinces_bus_ride_l3017_301743

theorem vinces_bus_ride (zachary_ride : ℝ) (vince_difference : ℝ) :
  zachary_ride = 0.5 →
  vince_difference = 0.125 →
  zachary_ride + vince_difference = 0.625 :=
by
  sorry

end NUMINAMATH_CALUDE_vinces_bus_ride_l3017_301743


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l3017_301701

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-2, 3)) : 
  abs (P.2) = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l3017_301701


namespace NUMINAMATH_CALUDE_obtuse_angle_is_in_second_quadrant_l3017_301730

/-- Definition of an obtuse angle -/
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

/-- Definition of an angle in the second quadrant -/
def is_in_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

/-- Theorem: An obtuse angle is an angle in the second quadrant -/
theorem obtuse_angle_is_in_second_quadrant (θ : ℝ) :
  is_obtuse_angle θ ↔ is_in_second_quadrant θ :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_is_in_second_quadrant_l3017_301730


namespace NUMINAMATH_CALUDE_age_difference_of_children_l3017_301715

/-- Proves that the age difference between children is 4 years given the conditions -/
theorem age_difference_of_children (n : ℕ) (sum_ages : ℕ) (eldest_age : ℕ) (d : ℕ) :
  n = 4 ∧ 
  sum_ages = 48 ∧ 
  eldest_age = 18 ∧ 
  sum_ages = n * eldest_age - (d * (n * (n - 1)) / 2) →
  d = 4 :=
by sorry


end NUMINAMATH_CALUDE_age_difference_of_children_l3017_301715


namespace NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l3017_301732

/-- The number of toothpicks in a grid with a gap -/
def toothpicks_in_grid_with_gap (length width gap_length gap_width : ℕ) : ℕ :=
  let vertical_toothpicks := (length + 1) * width
  let horizontal_toothpicks := (width + 1) * length
  let gap_vertical_toothpicks := (gap_length + 1) * gap_width
  let gap_horizontal_toothpicks := (gap_width + 1) * gap_length
  vertical_toothpicks + horizontal_toothpicks - gap_vertical_toothpicks - gap_horizontal_toothpicks

/-- Theorem stating the number of toothpicks in the specific grid described in the problem -/
theorem toothpicks_in_specific_grid :
  toothpicks_in_grid_with_gap 70 40 10 5 = 5595 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l3017_301732


namespace NUMINAMATH_CALUDE_direction_vector_c_value_l3017_301784

-- Define the two points on the line
def point1 : ℝ × ℝ := (-7, 3)
def point2 : ℝ × ℝ := (-3, -1)

-- Define the direction vector
def direction_vector (c : ℝ) : ℝ × ℝ := (4, c)

-- Theorem statement
theorem direction_vector_c_value :
  ∃ (c : ℝ), direction_vector c = (point2.1 - point1.1, point2.2 - point1.2) :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_c_value_l3017_301784


namespace NUMINAMATH_CALUDE_power_division_equals_729_l3017_301794

theorem power_division_equals_729 : 3^12 / 27^2 = 729 :=
by
  -- Define 27 as 3^3
  have h1 : 27 = 3^3 := by sorry
  
  -- Prove that 3^12 / 27^2 = 729
  sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l3017_301794


namespace NUMINAMATH_CALUDE_student_pet_difference_is_85_l3017_301735

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of pet rabbits in each classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of pet hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 2

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * students_per_classroom

/-- The total number of pets (rabbits and hamsters) in all classrooms -/
def total_pets : ℕ := num_classrooms * (rabbits_per_classroom + hamsters_per_classroom)

/-- The difference between the total number of students and the total number of pets -/
def student_pet_difference : ℕ := total_students - total_pets

theorem student_pet_difference_is_85 : student_pet_difference = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_is_85_l3017_301735


namespace NUMINAMATH_CALUDE_ninas_inheritance_l3017_301790

theorem ninas_inheritance (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧                    -- Both investments are positive
  0.06 * x + 0.08 * y = 860 ∧        -- Total yearly interest
  (x = 5000 ∨ y = 5000) →            -- $5000 invested at one rate
  x + y = 12000 :=                   -- Total inheritance
by sorry

end NUMINAMATH_CALUDE_ninas_inheritance_l3017_301790


namespace NUMINAMATH_CALUDE_x_equals_one_l3017_301703

theorem x_equals_one (y : ℝ) (a : ℝ) (x : ℝ) 
  (h1 : x + a * y = 10) 
  (h2 : y = 3) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_equals_one_l3017_301703


namespace NUMINAMATH_CALUDE_max_sum_rectangle_sides_l3017_301756

theorem max_sum_rectangle_sides (n : ℕ) (h : n = 10) :
  let total_sum := n * (n + 1) / 2
  let corner_sum := n + (n - 1) + (n - 2) + (n - 4)
  ∃ (side_sum : ℕ), 
    side_sum = (total_sum + corner_sum) / 4 ∧ 
    side_sum = 22 ∧
    ∀ (other_sum : ℕ), 
      (other_sum * 4 ≤ total_sum + corner_sum) → 
      other_sum ≤ side_sum :=
by sorry

end NUMINAMATH_CALUDE_max_sum_rectangle_sides_l3017_301756


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_32767_l3017_301708

def greatest_prime_divisor (n : ℕ) : ℕ :=
  sorry

def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem sum_digits_greatest_prime_divisor_32767 :
  sum_of_digits (greatest_prime_divisor 32767) = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_32767_l3017_301708


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3017_301793

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem mans_age_to_sons_age_ratio :
  ∀ (man_age son_age : ℕ),
  man_age = son_age + 18 →
  son_age = 16 →
  ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3017_301793


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_three_l3017_301740

theorem power_two_plus_one_div_three (n : ℕ+) :
  3 ∣ (2^n.val + 1) ↔ n.val % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_three_l3017_301740


namespace NUMINAMATH_CALUDE_initial_men_count_l3017_301762

/-- Represents the initial number of men working on the project -/
def initial_men : ℕ := sorry

/-- Represents the number of days to complete the work with the initial group -/
def initial_days : ℕ := 40

/-- Represents the number of men who leave the project -/
def men_who_leave : ℕ := 20

/-- Represents the number of days worked before some men leave -/
def days_before_leaving : ℕ := 10

/-- Represents the number of days to complete the remaining work after some men leave -/
def remaining_days : ℕ := 40

/-- Work rate of one man per day -/
def work_rate : ℚ := 1 / (initial_men * initial_days)

/-- Fraction of work completed before some men leave -/
def work_completed_before_leaving : ℚ := work_rate * initial_men * days_before_leaving

/-- Fraction of work remaining after some men leave -/
def remaining_work : ℚ := 1 - work_completed_before_leaving

/-- The theorem states that given the conditions, the initial number of men is 80 -/
theorem initial_men_count : initial_men = 80 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_l3017_301762


namespace NUMINAMATH_CALUDE_rational_sum_zero_l3017_301723

theorem rational_sum_zero (a b c : ℚ) 
  (h : (a + b + c) * (a + b - c) = 4 * c^2) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_zero_l3017_301723


namespace NUMINAMATH_CALUDE_vector_sign_sum_l3017_301754

/-- Given a 3-dimensional vector with nonzero components, the sum of the signs of its components
    plus the sign of their product can only be 4, 0, or -2. -/
theorem vector_sign_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x / |x| + y / |y| + z / |z| + (x * y * z) / |x * y * z|) ∈ ({4, 0, -2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_sign_sum_l3017_301754


namespace NUMINAMATH_CALUDE_charlie_running_steps_l3017_301745

/-- Given that Charlie makes 5350 steps on a 3-kilometer running field,
    prove that running 2 1/2 times around the field results in 13375 steps. -/
theorem charlie_running_steps (steps_per_field : ℕ) (field_length : ℝ) (laps : ℝ) :
  steps_per_field = 5350 →
  field_length = 3 →
  laps = 2.5 →
  (steps_per_field : ℝ) * laps = 13375 := by
  sorry

end NUMINAMATH_CALUDE_charlie_running_steps_l3017_301745


namespace NUMINAMATH_CALUDE_dice_roll_probability_l3017_301761

-- Define a dice roll
def DiceRoll : Type := Fin 6

-- Define a point as a pair of dice rolls
def Point : Type := DiceRoll × DiceRoll

-- Define the condition for a point to be inside the circle
def InsideCircle (p : Point) : Prop :=
  (p.1.val + 1)^2 + (p.2.val + 1)^2 < 17

-- Define the total number of possible outcomes
def TotalOutcomes : Nat := 36

-- Define the number of favorable outcomes
def FavorableOutcomes : Nat := 8

-- Theorem statement
theorem dice_roll_probability :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 2 / 9 :=
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l3017_301761


namespace NUMINAMATH_CALUDE_koolaid_mixture_l3017_301746

theorem koolaid_mixture (W : ℝ) : 
  W > 4 →
  (2 : ℝ) / (2 + 4 * (W - 4)) = 0.04 →
  W = 16 := by
sorry

end NUMINAMATH_CALUDE_koolaid_mixture_l3017_301746


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3017_301779

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  a = Real.sqrt 3 ∧ 
  b = Real.sqrt 13 ∧ 
  c = 4 ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3017_301779


namespace NUMINAMATH_CALUDE_solve_for_y_l3017_301759

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3017_301759


namespace NUMINAMATH_CALUDE_maria_gave_65_towels_l3017_301795

/-- The number of towels Maria gave to her mother -/
def towels_given_to_mother (green_towels white_towels remaining_towels : ℕ) : ℕ :=
  green_towels + white_towels - remaining_towels

/-- Proof that Maria gave 65 towels to her mother -/
theorem maria_gave_65_towels :
  towels_given_to_mother 40 44 19 = 65 := by
  sorry

end NUMINAMATH_CALUDE_maria_gave_65_towels_l3017_301795


namespace NUMINAMATH_CALUDE_oil_container_distribution_l3017_301777

theorem oil_container_distribution :
  ∃ (n m k : ℕ),
    n + m + k = 100 ∧
    n + 10 * m + 50 * k = 500 ∧
    n = 60 ∧ m = 39 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_oil_container_distribution_l3017_301777


namespace NUMINAMATH_CALUDE_budget_allocation_l3017_301758

theorem budget_allocation (transportation research_development utilities supplies salaries equipment : ℝ) :
  transportation = 20 →
  research_development = 9 →
  utilities = 5 →
  supplies = 2 →
  salaries = 216 / 360 * 100 →
  transportation + research_development + utilities + supplies + salaries + equipment = 100 →
  equipment = 4 :=
by sorry

end NUMINAMATH_CALUDE_budget_allocation_l3017_301758


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3017_301774

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 25) :
  (train_length + bridge_length) / crossing_time = 16 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l3017_301774


namespace NUMINAMATH_CALUDE_sum_of_digits_oneOver99Squared_l3017_301704

/-- Represents a repeating decimal expansion -/
structure RepeatingDecimal where
  digits : List Nat
  period : Nat

/-- The repeating decimal expansion of 1/(99^2) -/
def oneOver99Squared : RepeatingDecimal :=
  { digits := sorry
    period := sorry }

/-- The sum of digits in one period of the repeating decimal expansion of 1/(99^2) -/
def sumOfDigits (rd : RepeatingDecimal) : Nat :=
  (rd.digits.take rd.period).sum

theorem sum_of_digits_oneOver99Squared :
  sumOfDigits oneOver99Squared = 883 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_oneOver99Squared_l3017_301704


namespace NUMINAMATH_CALUDE_m_minus_n_value_l3017_301768

theorem m_minus_n_value (m n : ℤ) 
  (h1 : |m| = 4) 
  (h2 : |n| = 6) 
  (h3 : m + n = |m + n|) : 
  m - n = -2 ∨ m - n = -10 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_value_l3017_301768


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_equals_one_l3017_301760

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_square_equals_one :
  (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_equals_one_l3017_301760


namespace NUMINAMATH_CALUDE_unique_number_l3017_301764

theorem unique_number : ∃! n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧  -- three-digit number
  (n / 100 = 4) ∧  -- starts with 4
  ((n % 100) * 10 + 4 = (3 * n) / 4)  -- moving 4 to end results in 0.75 times original
  := by sorry

end NUMINAMATH_CALUDE_unique_number_l3017_301764


namespace NUMINAMATH_CALUDE_quadratic_two_positive_roots_l3017_301729

theorem quadratic_two_positive_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ 
   a * x^2 + 2*x - 1 = 0 ∧ 
   a * y^2 + 2*y - 1 = 0) ↔ 
  (-1 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_positive_roots_l3017_301729


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3017_301798

theorem system_of_equations_solution (x y z : ℝ) 
  (eq1 : 4 * x - 6 * y - 2 * z = 0)
  (eq2 : 2 * x + 6 * y - 28 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - 6*x*y) / (y^2 + 4*z^2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3017_301798


namespace NUMINAMATH_CALUDE_car_average_speed_l3017_301771

/-- Calculates the average speed of a car given specific conditions during a 4-hour trip -/
theorem car_average_speed : 
  let first_hour_speed : ℝ := 145
  let second_hour_speed : ℝ := 60
  let stop_duration : ℝ := 1/3
  let fourth_hour_min_speed : ℝ := 45
  let fourth_hour_max_speed : ℝ := 100
  let total_time : ℝ := 4 + stop_duration
  let fourth_hour_avg_speed : ℝ := (fourth_hour_min_speed + fourth_hour_max_speed) / 2
  let total_distance : ℝ := first_hour_speed + second_hour_speed + fourth_hour_avg_speed
  let average_speed : ℝ := total_distance / total_time
  ∃ ε > 0, |average_speed - 64.06| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_car_average_speed_l3017_301771


namespace NUMINAMATH_CALUDE_ink_covered_term_l3017_301720

variables {a b : ℝ}

theorem ink_covered_term (h : ∃ x, x * 3 * a * b = 6 * a * b - 3 * a * b ^ 3) :
  ∃ x, x = 2 - b ^ 2 ∧ x * 3 * a * b = 6 * a * b - 3 * a * b ^ 3 := by
sorry

end NUMINAMATH_CALUDE_ink_covered_term_l3017_301720


namespace NUMINAMATH_CALUDE_flea_meeting_configuration_l3017_301733

/-- Represents a small triangle on the infinite sheet of triangulated paper -/
structure SmallTriangle where
  x : ℤ
  y : ℤ

/-- Represents the equilateral triangle containing n^2 small triangles -/
def LargeTriangle (n : ℕ) : Set SmallTriangle :=
  { t : SmallTriangle | 0 ≤ t.x ∧ 0 ≤ t.y ∧ t.x + t.y < n }

/-- Represents the set of possible jumps a flea can make -/
def PossibleJumps : List (ℤ × ℤ) := [(1, 0), (-1, 1), (0, -1)]

/-- Defines a valid jump for a flea -/
def ValidJump (t1 t2 : SmallTriangle) : Prop :=
  (t2.x - t1.x, t2.y - t1.y) ∈ PossibleJumps

/-- Theorem: For which positive integers n does there exist an initial configuration
    such that after a finite number of jumps all the n fleas can meet in a single small triangle? -/
theorem flea_meeting_configuration (n : ℕ) :
  (∃ (initial_config : Fin n → SmallTriangle)
     (final_triangle : SmallTriangle)
     (num_jumps : ℕ),
   (∀ i j : Fin n, i ≠ j → initial_config i ≠ initial_config j) ∧
   (∀ i : Fin n, initial_config i ∈ LargeTriangle n) ∧
   (∃ (jump_sequence : Fin n → ℕ → SmallTriangle),
     (∀ i : Fin n, jump_sequence i 0 = initial_config i) ∧
     (∀ i : Fin n, ∀ k : ℕ, k < num_jumps →
       ValidJump (jump_sequence i k) (jump_sequence i (k+1))) ∧
     (∀ i : Fin n, jump_sequence i num_jumps = final_triangle))) ↔
  (n ≥ 1 ∧ n ≠ 2 ∧ n ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_flea_meeting_configuration_l3017_301733


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3017_301742

def diophantine_equation (x y : ℤ) : Prop :=
  2 * x^4 - 4 * y^4 - 7 * x^2 * y^2 - 27 * x^2 + 63 * y^2 + 85 = 0

def solution_set : Set (ℤ × ℤ) :=
  {(3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3)}

theorem diophantine_equation_solutions :
  ∀ (x y : ℤ), diophantine_equation x y ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3017_301742


namespace NUMINAMATH_CALUDE_percentage_increase_l3017_301728

theorem percentage_increase (x y p : ℝ) : 
  y = x * (1 + p / 100) →
  y = 150 →
  x = 120 →
  p = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l3017_301728


namespace NUMINAMATH_CALUDE_oil_distribution_l3017_301744

/-- Represents the problem of minimizing the number of small barrels --/
def MinimizeSmallBarrels (total_oil : ℕ) (large_barrel_capacity : ℕ) (small_barrel_capacity : ℕ) : Prop :=
  ∃ (large_barrels small_barrels : ℕ),
    large_barrel_capacity * large_barrels + small_barrel_capacity * small_barrels = total_oil ∧
    small_barrels = 1 ∧
    ∀ (l s : ℕ), large_barrel_capacity * l + small_barrel_capacity * s = total_oil →
      s ≥ small_barrels

theorem oil_distribution :
  MinimizeSmallBarrels 745 11 7 :=
sorry

end NUMINAMATH_CALUDE_oil_distribution_l3017_301744


namespace NUMINAMATH_CALUDE_people_in_house_l3017_301717

theorem people_in_house : 
  ∀ (initial_bedroom : ℕ) (entering_bedroom : ℕ) (living_room : ℕ),
    initial_bedroom = 2 →
    entering_bedroom = 5 →
    living_room = 8 →
    initial_bedroom + entering_bedroom + living_room = 14 := by
  sorry

end NUMINAMATH_CALUDE_people_in_house_l3017_301717


namespace NUMINAMATH_CALUDE_age_difference_l3017_301710

theorem age_difference (A B C : ℤ) (h : A + B = B + C + 11) : A - C = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3017_301710


namespace NUMINAMATH_CALUDE_scale_division_l3017_301711

/-- Given a scale of length 80 inches divided into 4 equal parts, 
    prove that each part is 20 inches long. -/
theorem scale_division (scale_length : ℕ) (num_parts : ℕ) (part_length : ℕ) : 
  scale_length = 80 ∧ num_parts = 4 ∧ scale_length = num_parts * part_length → part_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l3017_301711


namespace NUMINAMATH_CALUDE_green_balls_count_l3017_301727

theorem green_balls_count (total : ℕ) (red : ℕ) (green : ℕ) (prob_red : ℚ) : 
  red = 8 →
  green = total - red →
  prob_red = 1/3 →
  prob_red = red / total →
  green = 16 := by sorry

end NUMINAMATH_CALUDE_green_balls_count_l3017_301727


namespace NUMINAMATH_CALUDE_five_student_committees_l3017_301719

theorem five_student_committees (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_l3017_301719


namespace NUMINAMATH_CALUDE_circle_transformation_l3017_301757

/-- Coordinate transformation φ -/
def φ (x y : ℝ) : ℝ × ℝ :=
  (4 * x, 2 * y)

/-- Original circle equation -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Transformed equation -/
def transformed_equation (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 4) = 1

theorem circle_transformation (x y : ℝ) :
  original_circle x y ↔ transformed_equation (φ x y).1 (φ x y).2 :=
by sorry

end NUMINAMATH_CALUDE_circle_transformation_l3017_301757


namespace NUMINAMATH_CALUDE_product_of_numbers_l3017_301747

theorem product_of_numbers (x y : ℝ) : 
  |x - y| = 11 → x^2 + y^2 = 221 → x * y = 60 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3017_301747


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3017_301769

/-- Calculates the total wet surface area of a rectangular cistern --/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let side_area1 := 2 * (length * depth)
  let side_area2 := 2 * (width * depth)
  bottom_area + side_area1 + side_area2

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 m² --/
theorem cistern_wet_surface_area :
  total_wet_surface_area 4 8 1.25 = 62 := by
  sorry

#eval total_wet_surface_area 4 8 1.25

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3017_301769


namespace NUMINAMATH_CALUDE_P_120_l3017_301712

/-- 
P(n) represents the number of ways to express a positive integer n 
as a product of integers greater than 1, where the order matters.
-/
def P (n : ℕ) : ℕ := sorry

/-- The prime factorization of 120 -/
def primeFactors120 : List ℕ := [2, 2, 2, 3, 5]

/-- 120 is the product of its prime factors -/
axiom is120 : (primeFactors120.prod = 120)

/-- All elements in primeFactors120 are prime numbers -/
axiom allPrime : ∀ p ∈ primeFactors120, Nat.Prime p

theorem P_120 : P 120 = 29 := by sorry

end NUMINAMATH_CALUDE_P_120_l3017_301712


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3017_301788

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x | ∃ m ∈ A, x = 3 * m - 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3017_301788


namespace NUMINAMATH_CALUDE_john_door_replacement_l3017_301750

def outside_door_cost : ℕ := 20
def bedroom_door_count : ℕ := 3
def total_cost : ℕ := 70

def outside_door_count : ℕ := 2

theorem john_door_replacement :
  ∃ (x : ℕ),
    x * outside_door_cost + 
    bedroom_door_count * (outside_door_cost / 2) = 
    total_cost ∧
    x = outside_door_count :=
by sorry

end NUMINAMATH_CALUDE_john_door_replacement_l3017_301750


namespace NUMINAMATH_CALUDE_computer_literate_female_employees_l3017_301786

theorem computer_literate_female_employees 
  (total_employees : ℕ) 
  (female_percentage : ℚ) 
  (male_literate_percentage : ℚ) 
  (total_literate_percentage : ℚ) 
  (h1 : total_employees = 1400)
  (h2 : female_percentage = 60 / 100)
  (h3 : male_literate_percentage = 50 / 100)
  (h4 : total_literate_percentage = 62 / 100) :
  ↑(total_employees : ℚ) * female_percentage * total_literate_percentage - 
  (↑total_employees * (1 - female_percentage) * male_literate_percentage) = 588 := by
  sorry

#check computer_literate_female_employees

end NUMINAMATH_CALUDE_computer_literate_female_employees_l3017_301786


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3017_301718

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 9 ∧ x ≠ -6 →
  (4 * x - 3) / (x^2 - 3*x - 54) = (11/5) / (x - 9) + (9/5) / (x + 6) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3017_301718


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3017_301766

/-- The line equation passes through a fixed point for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2 + m) * (-1) + (1 - 2*m) * (-2) + 4 - 3*m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3017_301766


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l3017_301787

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the xoy plane --/
def symmetricXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetric_point_xoy_plane :
  let M : Point3D := { x := 2, y := 5, z := 8 }
  let N : Point3D := symmetricXOY M
  N = { x := 2, y := 5, z := -8 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l3017_301787


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3017_301782

theorem largest_prime_factor_of_expression : 
  (Nat.factors (12^3 + 8^4 - 4^5)).maximum = some 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l3017_301782


namespace NUMINAMATH_CALUDE_expression_value_l3017_301770

theorem expression_value (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 4) :
  (a^2 + b^2) / (2 - c) + (b^2 + c^2) / (2 - a) + (c^2 + a^2) / (2 - b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3017_301770


namespace NUMINAMATH_CALUDE_largest_angle_not_less_than_60_degrees_l3017_301781

open Real

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point) (b : Point)

/-- Calculates the angle between two lines -/
noncomputable def angle (l1 l2 : Line) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (a b c : Point) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (m : Point) (a b : Point) : Prop := sorry

/-- Main theorem -/
theorem largest_angle_not_less_than_60_degrees 
  (a b c : Point) 
  (h_equilateral : isEquilateral a b c)
  (c₁ : Point) (h_c₁_midpoint : isMidpoint c₁ a b)
  (a₁ : Point) (h_a₁_midpoint : isMidpoint a₁ b c)
  (b₁ : Point) (h_b₁_midpoint : isMidpoint b₁ c a)
  (p : Point) :
  let angle1 := angle (Line.mk a b) (Line.mk p c₁)
  let angle2 := angle (Line.mk b c) (Line.mk p a₁)
  let angle3 := angle (Line.mk c a) (Line.mk p b₁)
  max angle1 (max angle2 angle3) ≥ π/3 := by sorry

end NUMINAMATH_CALUDE_largest_angle_not_less_than_60_degrees_l3017_301781


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3017_301700

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

/-- A point is on the square if its coordinates are equal in absolute value -/
def on_square (x y : ℝ) : Prop := |x| = |y|

/-- The square is inscribed in the ellipse -/
def inscribed_square (t : ℝ) : Prop := 
  ellipse t t ∧ on_square t t ∧ t > 0

/-- The area of the inscribed square -/
def square_area (t : ℝ) : ℝ := (2*t)^2

theorem inscribed_square_area : 
  ∃ t : ℝ, inscribed_square t ∧ square_area t = 32/3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3017_301700


namespace NUMINAMATH_CALUDE_basketball_wins_needed_l3017_301763

/-- Calculates the number of additional games a basketball team needs to win to achieve a target win percentage -/
theorem basketball_wins_needed
  (games_played : ℕ)
  (games_won : ℕ)
  (games_remaining : ℕ)
  (target_percentage : ℚ)
  (h1 : games_played = 50)
  (h2 : games_won = 35)
  (h3 : games_remaining = 25)
  (h4 : target_percentage = 64 / 100) :
  ⌈(target_percentage * ↑(games_played + games_remaining) - ↑games_won)⌉ = 13 :=
by sorry

end NUMINAMATH_CALUDE_basketball_wins_needed_l3017_301763


namespace NUMINAMATH_CALUDE_total_length_of_T_l3017_301748

/-- The set T of points (x, y) in the Cartesian plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ‖‖|p.1| - 3‖ - 2‖ + ‖‖|p.2| - 3‖ - 2‖ = 2}

/-- The total length of all lines forming the set T -/
def total_length (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the total length of lines forming T is 64√2 -/
theorem total_length_of_T : total_length T = 64 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_total_length_of_T_l3017_301748


namespace NUMINAMATH_CALUDE_player_one_wins_with_2023_coins_l3017_301706

/-- Represents the possible moves for each player -/
inductive Move
| three : Move
| five : Move
| two : Move
| four : Move

/-- Represents a player in the game -/
inductive Player
| one : Player
| two : Player

/-- The game state -/
structure GameState where
  coins : ℕ
  currentPlayer : Player

/-- Determines if a move is valid for a given player -/
def validMove (player : Player) (move : Move) : Bool :=
  match player, move with
  | Player.one, Move.three => true
  | Player.one, Move.five => true
  | Player.two, Move.two => true
  | Player.two, Move.four => true
  | _, _ => false

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : Option GameState :=
  if validMove state.currentPlayer move then
    let newCoins := match move with
      | Move.three => state.coins - 3
      | Move.five => state.coins - 5
      | Move.two => state.coins - 2
      | Move.four => state.coins - 4
    let newPlayer := match state.currentPlayer with
      | Player.one => Player.two
      | Player.two => Player.one
    some { coins := newCoins, currentPlayer := newPlayer }
  else
    none

/-- Determines if a player has a winning strategy from a given game state -/
def hasWinningStrategy (state : GameState) : Prop :=
  sorry

/-- The main theorem: Player 1 has a winning strategy when starting with 2023 coins -/
theorem player_one_wins_with_2023_coins :
  hasWinningStrategy { coins := 2023, currentPlayer := Player.one } :=
  sorry

end NUMINAMATH_CALUDE_player_one_wins_with_2023_coins_l3017_301706


namespace NUMINAMATH_CALUDE_systematic_sample_valid_l3017_301773

def is_valid_systematic_sample (sample : List Nat) (population_size : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  ∀ i j, i < j → i < sample.length → j < sample.length →
    sample[i]! < sample[j]! ∧
    (sample[j]! - sample[i]!) % (population_size / sample_size) = 0 ∧
    sample[sample.length - 1]! ≤ population_size

theorem systematic_sample_valid :
  is_valid_systematic_sample [3, 13, 23, 33, 43, 53] 60 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_valid_l3017_301773


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l3017_301709

theorem geometric_to_arithmetic_sequence (a₁ a₂ a₃ a₄ q : ℝ) :
  q > 0 ∧ q ≠ 1 ∧ 
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧
  ((2 * a₃ = a₁ + a₄) ∨ (2 * a₂ = a₁ + a₄)) →
  q = ((-1 + Real.sqrt 5) / 2) ∨ q = ((1 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l3017_301709


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l3017_301789

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l3017_301789


namespace NUMINAMATH_CALUDE_equation_solution_l3017_301778

theorem equation_solution : 
  ∀ x : ℝ, (2010 + x)^2 = 4*x^2 ↔ x = 2010 ∨ x = -670 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3017_301778


namespace NUMINAMATH_CALUDE_shortest_distance_is_four_l3017_301772

/-- Represents the distances between three points A, B, and C. -/
structure TriangleDistances where
  ab : ℝ  -- Distance between A and B
  bc : ℝ  -- Distance between B and C
  ac : ℝ  -- Distance between A and C

/-- Given conditions for the problem -/
def problem_conditions (d : TriangleDistances) : Prop :=
  d.ab + d.bc = 10 ∧
  d.bc + d.ac = 13 ∧
  d.ac + d.ab = 11

/-- The theorem to be proved -/
theorem shortest_distance_is_four (d : TriangleDistances) 
  (h : problem_conditions d) : 
  min d.ab (min d.bc d.ac) = 4 := by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_is_four_l3017_301772


namespace NUMINAMATH_CALUDE_spinster_count_l3017_301702

theorem spinster_count : 
  ∀ (spinsters cats : ℕ), 
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 →
    cats = spinsters + 35 →
    spinsters = 14 := by
  sorry

end NUMINAMATH_CALUDE_spinster_count_l3017_301702


namespace NUMINAMATH_CALUDE_starting_lineup_count_starting_lineup_count_12_5_l3017_301707

/-- The number of ways to choose a starting lineup from a basketball team -/
theorem starting_lineup_count : ℕ → ℕ → ℕ
  | team_size, lineup_size =>
    if lineup_size > team_size then 0
    else team_size * (Nat.choose (team_size - 1) (lineup_size - 1))

/-- Proof that the number of ways to choose a starting lineup of 5 players 
    from a team of 12 members, where one player is designated as the captain 
    and the other four positions are interchangeable, is equal to 3960 -/
theorem starting_lineup_count_12_5 :
  starting_lineup_count 12 5 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_starting_lineup_count_12_5_l3017_301707


namespace NUMINAMATH_CALUDE_triangle_side_length_l3017_301724

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = 4 →
  B = π / 6 →
  Real.sin A = 1 / 3 →
  a = 8 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3017_301724


namespace NUMINAMATH_CALUDE_angle_calculation_l3017_301792

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Two angles are supplementary if their sum is 180 degrees -/
def supplementary (a b : ℝ) : Prop := a + b = 180

/-- Given that angle1 and angle2 are complementary, angle2 and angle3 are supplementary, 
    and angle1 is 20 degrees, prove that angle3 is 110 degrees -/
theorem angle_calculation (angle1 angle2 angle3 : ℝ) 
    (h1 : complementary angle1 angle2)
    (h2 : supplementary angle2 angle3)
    (h3 : angle1 = 20) : 
  angle3 = 110 := by sorry

end NUMINAMATH_CALUDE_angle_calculation_l3017_301792


namespace NUMINAMATH_CALUDE_average_annual_cost_reduction_l3017_301751

theorem average_annual_cost_reduction (total_reduction : Real) 
  (h : total_reduction = 0.36) : 
  ∃ x : Real, x > 0 ∧ x < 1 ∧ (1 - x)^2 = 1 - total_reduction :=
sorry

end NUMINAMATH_CALUDE_average_annual_cost_reduction_l3017_301751


namespace NUMINAMATH_CALUDE_chord_length_of_concentric_circles_l3017_301734

/-- Given two concentric circles with radii R and r, where the area of the annulus
    between them is 12½π square inches, the length of the chord of the larger circle
    which is tangent to the smaller circle is 5√2 inches. -/
theorem chord_length_of_concentric_circles (R r : ℝ) :
  R > r →
  π * R^2 - π * r^2 = 25 / 2 * π →
  ∃ (c : ℝ), c^2 = R^2 - r^2 ∧ c = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_of_concentric_circles_l3017_301734


namespace NUMINAMATH_CALUDE_inequality_proof_l3017_301767

theorem inequality_proof (x y z : ℝ) (h : 2 * x + y^2 + z^2 ≤ 2) : x + y + z ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3017_301767


namespace NUMINAMATH_CALUDE_garage_wheels_count_l3017_301780

def total_wheels (cars bicycles : Nat) (lawnmower tricycle unicycle skateboard wheelbarrow wagon : Nat) : Nat :=
  cars * 4 + bicycles * 2 + lawnmower * 4 + tricycle * 3 + unicycle + skateboard * 4 + wheelbarrow + wagon * 4

theorem garage_wheels_count :
  total_wheels 2 3 1 1 1 1 1 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_count_l3017_301780


namespace NUMINAMATH_CALUDE_number_ordering_l3017_301765

theorem number_ordering : (4 : ℚ) / 5 < (801 : ℚ) / 1000 ∧ (801 : ℚ) / 1000 < 81 / 100 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l3017_301765


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3017_301796

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 10) :
  1/p + 9/q + 25/r + 49/s + 81/t + 121/u ≥ 129.6 := by
  sorry


end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3017_301796


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3017_301755

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 40 →
  (360 : ℝ) / exterior_angle = n →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3017_301755


namespace NUMINAMATH_CALUDE_partnership_gain_l3017_301749

/-- Represents the investment and profit of a partnership --/
structure Partnership where
  x : ℝ  -- A's investment
  a_share : ℝ  -- A's share of the profit
  total_gain : ℝ  -- Total annual gain

/-- Calculates the total annual gain of the partnership --/
def calculate_total_gain (p : Partnership) : ℝ :=
  3 * p.a_share

/-- Theorem stating that given the investment conditions and A's share, 
    the total annual gain is 12000 --/
theorem partnership_gain (p : Partnership) 
  (h1 : p.x > 0)  -- A's investment is positive
  (h2 : p.a_share = 4000)  -- A's share is 4000
  : p.total_gain = 12000 :=
by
  sorry

#check partnership_gain

end NUMINAMATH_CALUDE_partnership_gain_l3017_301749


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3017_301736

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_line_through_point (P : Point) (l : Line) :
  P.x = 1 ∧ P.y = -1 ∧
  l.a = 1 ∧ l.b = -2 ∧ l.c = 1 →
  ∃ (m : Line), perpendicular m l ∧ pointOnLine P m ∧ m.a = 2 ∧ m.b = 1 ∧ m.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3017_301736


namespace NUMINAMATH_CALUDE_factorial_division_l3017_301721

theorem factorial_division (h : Nat.factorial 9 = 362880) :
  Nat.factorial 9 / Nat.factorial 4 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3017_301721


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l3017_301726

theorem lcm_gcd_problem : (Nat.lcm 12 9 * Nat.gcd 12 9) - Nat.gcd 15 9 = 105 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l3017_301726


namespace NUMINAMATH_CALUDE_fraction_equality_l3017_301799

theorem fraction_equality (a b : ℝ) (h : a ≠ b) :
  (a^2 - b^2) / (a - b)^2 = (a + b) / (a - b) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3017_301799


namespace NUMINAMATH_CALUDE_jack_additional_sweets_l3017_301753

theorem jack_additional_sweets (initial_sweets : ℕ) (remaining_sweets : ℕ) : 
  initial_sweets = 22 →
  remaining_sweets = 7 →
  (initial_sweets / 2 + (initial_sweets - remaining_sweets - initial_sweets / 2) = initial_sweets - remaining_sweets) →
  initial_sweets - remaining_sweets - initial_sweets / 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_jack_additional_sweets_l3017_301753


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3017_301731

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 3 + a 5 + a 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3017_301731


namespace NUMINAMATH_CALUDE_max_value_of_a_l3017_301738

theorem max_value_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 7)
  (prod_sum_eq : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 ∧ 
  ∃ (b' c' : ℝ), b' + c' = 7 - (7 + Real.sqrt 46) / 3 ∧ 
                 ((7 + Real.sqrt 46) / 3) * b' + ((7 + Real.sqrt 46) / 3) * c' + b' * c' = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3017_301738


namespace NUMINAMATH_CALUDE_monday_sales_l3017_301797

/-- Represents the sales and pricing of a shoe store -/
structure ShoeStore where
  shoe_price : ℕ
  boot_price : ℕ
  monday_shoe_sales : ℕ
  monday_boot_sales : ℕ
  tuesday_shoe_sales : ℕ
  tuesday_boot_sales : ℕ
  tuesday_total_sales : ℕ

/-- The conditions of the problem -/
def store_conditions (s : ShoeStore) : Prop :=
  s.boot_price = s.shoe_price + 15 ∧
  s.monday_shoe_sales = 22 ∧
  s.monday_boot_sales = 16 ∧
  s.tuesday_shoe_sales = 8 ∧
  s.tuesday_boot_sales = 32 ∧
  s.tuesday_total_sales = 560 ∧
  s.tuesday_shoe_sales * s.shoe_price + s.tuesday_boot_sales * s.boot_price = s.tuesday_total_sales

/-- The theorem to be proved -/
theorem monday_sales (s : ShoeStore) (h : store_conditions s) : 
  s.monday_shoe_sales * s.shoe_price + s.monday_boot_sales * s.boot_price = 316 := by
  sorry


end NUMINAMATH_CALUDE_monday_sales_l3017_301797


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l3017_301752

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) →
  n ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l3017_301752


namespace NUMINAMATH_CALUDE_linear_function_through_point_l3017_301716

def f (x : ℝ) : ℝ := x + 1

theorem linear_function_through_point :
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) ∧ f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_point_l3017_301716


namespace NUMINAMATH_CALUDE_line_does_not_intersect_circle_l3017_301783

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between a point and a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def lineIntersectsCircle (c : Circle) (l : Line) : Prop :=
  ∃ (p : ℝ × ℝ), distancePointToLine p l = 0 ∧ (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem line_does_not_intersect_circle
  (c : Circle) (l : Line) (h : distancePointToLine c.center l > c.radius) :
  ¬ lineIntersectsCircle c l :=
sorry

end NUMINAMATH_CALUDE_line_does_not_intersect_circle_l3017_301783


namespace NUMINAMATH_CALUDE_odd_function_sum_l3017_301714

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (λ x => f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3017_301714


namespace NUMINAMATH_CALUDE_number_of_paths_in_grid_l3017_301705

-- Define the grid dimensions
def grid_width : ℕ := 7
def grid_height : ℕ := 6

-- Define the total number of steps
def total_steps : ℕ := grid_width + grid_height

-- Theorem statement
theorem number_of_paths_in_grid : 
  (Nat.choose total_steps grid_height : ℕ) = 1716 := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_in_grid_l3017_301705


namespace NUMINAMATH_CALUDE_union_A_B_complement_intersection_A_B_l3017_301713

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set A
def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

-- Define set B
def B : Set ℝ := {x | 2*x - 9 ≥ 6 - 3*x}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for ∁ᵤ(A ∩ B)
theorem complement_intersection_A_B : (A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_intersection_A_B_l3017_301713


namespace NUMINAMATH_CALUDE_rectangle_horizontal_length_l3017_301737

/-- Proves that a rectangle with perimeter 54 cm and horizontal length 3 cm longer than vertical length has a horizontal length of 15 cm -/
theorem rectangle_horizontal_length : 
  ∀ (v h : ℝ), 
  (2 * v + 2 * h = 54) →  -- perimeter is 54 cm
  (h = v + 3) →           -- horizontal length is 3 cm longer than vertical length
  h = 15 := by            -- horizontal length is 15 cm
sorry

end NUMINAMATH_CALUDE_rectangle_horizontal_length_l3017_301737


namespace NUMINAMATH_CALUDE_count_numbers_with_three_l3017_301739

/-- Count of numbers from 1 to 800 without digit 3 -/
def count_without_three : ℕ := 729

/-- Count of numbers from 1 to 800 with at least one digit 3 -/
def count_with_three : ℕ := 800 - count_without_three

theorem count_numbers_with_three : count_with_three = 71 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_l3017_301739


namespace NUMINAMATH_CALUDE_compare_exponentials_l3017_301725

theorem compare_exponentials :
  (4 : ℝ) ^ (1/4) > (5 : ℝ) ^ (1/5) ∧
  (5 : ℝ) ^ (1/5) > (16 : ℝ) ^ (1/16) ∧
  (16 : ℝ) ^ (1/16) > (25 : ℝ) ^ (1/25) :=
by sorry

end NUMINAMATH_CALUDE_compare_exponentials_l3017_301725


namespace NUMINAMATH_CALUDE_trajectory_is_square_l3017_301785

-- Define the set of points (x, y) satisfying |x| + |y| = 1
def trajectory : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1| + |p.2| = 1}

-- Define a square in the plane
def isSquare (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), S = {p : ℝ × ℝ | max (|p.1 - a|) (|p.2 - b|) = 1/2}

-- Theorem statement
theorem trajectory_is_square : isSquare trajectory := by sorry

end NUMINAMATH_CALUDE_trajectory_is_square_l3017_301785


namespace NUMINAMATH_CALUDE_hexagonal_pattern_selections_l3017_301775

-- Define the structure of the hexagonal grid
def HexagonalGrid := Unit

-- Define the specific hexagonal pattern to be selected
def HexagonalPattern := Unit

-- Define the number of distinct positions without rotations
def distinctPositionsWithoutRotations : ℕ := 26

-- Define the number of distinct rotations for the hexagonal pattern
def distinctRotations : ℕ := 3

-- Theorem statement
theorem hexagonal_pattern_selections (grid : HexagonalGrid) (pattern : HexagonalPattern) :
  (distinctPositionsWithoutRotations * distinctRotations) = 78 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pattern_selections_l3017_301775


namespace NUMINAMATH_CALUDE_new_figure_has_five_sides_l3017_301791

/-- A regular polygon with n sides and side length 1 -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  sideLength_eq_one : sideLength = 1

/-- The new figure formed by connecting a hexagon and triangle -/
def NewFigure (hexagon triangle : RegularPolygon) : ℕ :=
  hexagon.sides + triangle.sides - 2

/-- Theorem stating that the new figure has 5 sides -/
theorem new_figure_has_five_sides
  (hexagon : RegularPolygon)
  (triangle : RegularPolygon)
  (hexagon_is_hexagon : hexagon.sides = 6)
  (triangle_is_triangle : triangle.sides = 3) :
  NewFigure hexagon triangle = 5 := by
  sorry

#eval NewFigure ⟨6, 1, rfl⟩ ⟨3, 1, rfl⟩

end NUMINAMATH_CALUDE_new_figure_has_five_sides_l3017_301791


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_set_l3017_301722

-- Define set M
def M : Set ℝ := {-1, 0, 1}

-- Define set N
def N : Set ℝ := {y | ∃ x ∈ M, y = Real.sin x}

-- Theorem statement
theorem M_intersect_N_equals_zero_set : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_set_l3017_301722
