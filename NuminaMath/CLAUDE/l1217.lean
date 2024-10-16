import Mathlib

namespace NUMINAMATH_CALUDE_least_cubes_from_cuboid_l1217_121798

/-- Given a cuboidal block with dimensions 6 cm x 9 cm x 12 cm,
    prove that the least possible number of equal cubes that can be cut from this block is 24. -/
theorem least_cubes_from_cuboid (length width height : ℕ) 
  (h_length : length = 6)
  (h_width : width = 9)
  (h_height : height = 12) :
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧
    length % cube_side = 0 ∧
    width % cube_side = 0 ∧
    height % cube_side = 0 ∧
    (length * width * height) / (cube_side ^ 3) = 24 ∧
    ∀ (other_side : ℕ), other_side > cube_side →
      ¬(length % other_side = 0 ∧
        width % other_side = 0 ∧
        height % other_side = 0)) :=
by sorry

end NUMINAMATH_CALUDE_least_cubes_from_cuboid_l1217_121798


namespace NUMINAMATH_CALUDE_final_pressure_is_three_l1217_121732

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
structure GasState where
  pressure : ℝ
  volume : ℝ
  constant : ℝ
  h : pressure * volume = constant

/-- The initial state of the hydrogen gas -/
def initial_state : GasState :=
  { pressure := 6
  , volume := 3
  , constant := 18
  , h := by sorry }

/-- The final state of the hydrogen gas after transfer -/
def final_state : GasState :=
  { pressure := 3
  , volume := 6
  , constant := 18
  , h := by sorry }

/-- Theorem stating that the final pressure is 3 kPa -/
theorem final_pressure_is_three :
  final_state.pressure = 3 :=
by sorry

end NUMINAMATH_CALUDE_final_pressure_is_three_l1217_121732


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1217_121730

/-- Calculates the selling price given the cost price and profit percentage -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem: The selling price is $1170 given a cost price of $975 and a 20% profit -/
theorem selling_price_calculation :
  selling_price 975 20 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1217_121730


namespace NUMINAMATH_CALUDE_fabric_requirement_l1217_121786

/-- The number of dresses to be made -/
def num_dresses : ℕ := 4

/-- The amount of fabric available in feet -/
def fabric_available : ℝ := 7

/-- The additional amount of fabric needed in feet -/
def fabric_needed : ℝ := 59

/-- The number of feet in a yard -/
def feet_per_yard : ℝ := 3

/-- The amount of fabric required for one dress in yards -/
def fabric_per_dress : ℝ := 5.5

theorem fabric_requirement :
  (fabric_available + fabric_needed) / feet_per_yard / num_dresses = fabric_per_dress := by
  sorry

end NUMINAMATH_CALUDE_fabric_requirement_l1217_121786


namespace NUMINAMATH_CALUDE_teacher_selection_plans_l1217_121787

theorem teacher_selection_plans (male_teachers female_teachers selected_teachers : ℕ) 
  (h1 : male_teachers = 5)
  (h2 : female_teachers = 4)
  (h3 : selected_teachers = 3) :
  (Nat.choose male_teachers 2 * Nat.choose female_teachers 1 * Nat.factorial selected_teachers) +
  (Nat.choose male_teachers 1 * Nat.choose female_teachers 2 * Nat.factorial selected_teachers) = 420 := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_plans_l1217_121787


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1217_121716

theorem rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h₁ : side_area = 12)
  (h₂ : front_area = 8)
  (h₃ : bottom_area = 6)
  : ∃ (length width height : ℝ),
    length * width = front_area ∧
    width * height = side_area ∧
    length * height = bottom_area ∧
    length * width * height = 24 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1217_121716


namespace NUMINAMATH_CALUDE_farmer_theorem_l1217_121765

def farmer_problem (initial_tomatoes initial_potatoes remaining_total : ℕ) : ℕ :=
  (initial_tomatoes + initial_potatoes) - remaining_total

theorem farmer_theorem (initial_tomatoes initial_potatoes remaining_total : ℕ) :
  farmer_problem initial_tomatoes initial_potatoes remaining_total =
  (initial_tomatoes + initial_potatoes) - remaining_total :=
by sorry

#eval farmer_problem 175 77 80

end NUMINAMATH_CALUDE_farmer_theorem_l1217_121765


namespace NUMINAMATH_CALUDE_mothers_carrots_l1217_121784

theorem mothers_carrots (faye_carrots good_carrots bad_carrots : ℕ) 
  (h1 : faye_carrots = 23)
  (h2 : good_carrots = 12)
  (h3 : bad_carrots = 16) :
  good_carrots + bad_carrots - faye_carrots = 5 :=
by sorry

end NUMINAMATH_CALUDE_mothers_carrots_l1217_121784


namespace NUMINAMATH_CALUDE_helen_lawn_gas_consumption_l1217_121785

/-- Represents the number of months with 2 cuts per month -/
def low_frequency_months : ℕ := 4

/-- Represents the number of months with 4 cuts per month -/
def high_frequency_months : ℕ := 4

/-- Represents the number of cuts per month in low frequency months -/
def low_frequency_cuts : ℕ := 2

/-- Represents the number of cuts per month in high frequency months -/
def high_frequency_cuts : ℕ := 4

/-- Represents the number of cuts before needing to refuel -/
def cuts_per_refuel : ℕ := 4

/-- Represents the number of gallons used per refuel -/
def gallons_per_refuel : ℕ := 2

/-- Theorem stating that Helen will need 12 gallons of gas for lawn cutting from March through October -/
theorem helen_lawn_gas_consumption : 
  (low_frequency_months * low_frequency_cuts + high_frequency_months * high_frequency_cuts) / cuts_per_refuel * gallons_per_refuel = 12 :=
by sorry

end NUMINAMATH_CALUDE_helen_lawn_gas_consumption_l1217_121785


namespace NUMINAMATH_CALUDE_wrong_divisor_problem_l1217_121703

theorem wrong_divisor_problem (correct_divisor correct_answer student_answer : ℕ) 
  (h1 : correct_divisor = 36)
  (h2 : correct_answer = 58)
  (h3 : student_answer = 24) :
  ∃ (wrong_divisor : ℕ), 
    (correct_divisor * correct_answer) / wrong_divisor = student_answer ∧ 
    wrong_divisor = 87 := by
  sorry

end NUMINAMATH_CALUDE_wrong_divisor_problem_l1217_121703


namespace NUMINAMATH_CALUDE_matrix_power_101_l1217_121778

open Matrix

/-- Given a 3x3 matrix A, prove that A^101 equals the given result -/
theorem matrix_power_101 (A : Matrix (Fin 3) (Fin 3) ℝ) :
  A = ![![0, 0, 1],
       ![1, 0, 0],
       ![0, 1, 0]] →
  A^101 = ![![0, 1, 0],
            ![0, 0, 1],
            ![1, 0, 0]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_101_l1217_121778


namespace NUMINAMATH_CALUDE_concurrent_diagonals_l1217_121758

/-- Regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- The spiral similarity center of two regular polygons -/
def spiralSimilarityCenter (p q : RegularPolygon 100) : ℝ × ℝ :=
  sorry

/-- Intersection point of two line segments -/
def intersectionPoint (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The R points defined by the intersection of sides of p and q -/
def R (p q : RegularPolygon 100) (i : Fin 100) : ℝ × ℝ :=
  intersectionPoint (p.vertices i) (p.vertices (i+1)) (q.vertices i) (q.vertices (i+1))

/-- A diagonal of the 200-gon formed by R points -/
def diagonal (p q : RegularPolygon 100) (i : Fin 50) : Set (ℝ × ℝ) :=
  sorry

theorem concurrent_diagonals (p q : RegularPolygon 100) :
  ∃ (center : ℝ × ℝ), ∀ (i : Fin 50), center ∈ diagonal p q i := by
  sorry

end NUMINAMATH_CALUDE_concurrent_diagonals_l1217_121758


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l1217_121742

theorem fraction_multiplication_addition : 
  (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 5 : ℚ) + (1 / 2 : ℚ) = 31 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l1217_121742


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l1217_121704

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

def binary : List Bool := [true, true, false, true, false, false, true, false]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal binary) = [3, 1, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l1217_121704


namespace NUMINAMATH_CALUDE_gumball_probability_l1217_121705

/-- Given a box of gumballs with blue, green, red, and purple colors, 
    prove that the probability of selecting either a red or a purple gumball is 0.45, 
    given that the probability of selecting a blue gumball is 0.3 
    and the probability of selecting a green gumball is 0.25. -/
theorem gumball_probability (blue green red purple : ℝ) 
  (h1 : blue = 0.3) 
  (h2 : green = 0.25) 
  (h3 : blue + green + red + purple = 1) : 
  red + purple = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l1217_121705


namespace NUMINAMATH_CALUDE_unique_solution_system_l1217_121710

theorem unique_solution_system (a b c : ℝ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  a^4 + b^2 * c^2 = 16 * a ∧
  b^4 + c^2 * a^2 = 16 * b ∧
  c^4 + a^2 * b^2 = 16 * c →
  a = 2 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1217_121710


namespace NUMINAMATH_CALUDE_marnie_chips_per_day_l1217_121739

/-- Calculates the number of chips Marnie eats each day starting from the second day -/
def chips_per_day (total_chips : ℕ) (first_day_chips : ℕ) (total_days : ℕ) : ℕ :=
  (total_chips - first_day_chips) / (total_days - 1)

/-- Theorem stating that Marnie eats 10 chips per day starting from the second day -/
theorem marnie_chips_per_day :
  chips_per_day 100 10 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_marnie_chips_per_day_l1217_121739


namespace NUMINAMATH_CALUDE_smoking_students_not_hospitalized_l1217_121759

theorem smoking_students_not_hospitalized 
  (total_students : ℕ) 
  (smoking_percentage : ℚ) 
  (hospitalized_percentage : ℚ) 
  (h1 : total_students = 300)
  (h2 : smoking_percentage = 40 / 100)
  (h3 : hospitalized_percentage = 70 / 100) :
  ⌊total_students * smoking_percentage - 
   (total_students * smoking_percentage * hospitalized_percentage)⌋ = 36 := by
sorry

end NUMINAMATH_CALUDE_smoking_students_not_hospitalized_l1217_121759


namespace NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l1217_121796

/-- Represents a cake that can be cut -/
structure Cake :=
  (volume : ℝ)
  (height : ℝ)
  (width : ℝ)
  (depth : ℝ)

/-- Represents a cut made to the cake -/
inductive Cut
  | Horizontal
  | Vertical
  | Parallel

/-- The number of pieces resulting from a series of cuts -/
def num_pieces (cuts : List Cut) : ℕ :=
  2 ^ (cuts.length)

/-- The maximum number of identical pieces obtainable with 3 cuts -/
def max_pieces : ℕ := 8

/-- Theorem: The maximum number of identical pieces obtainable from a cake with 3 cuts is 8 -/
theorem max_pieces_with_three_cuts (c : Cake) :
  ∀ (cuts : List Cut), cuts.length = 3 → num_pieces cuts ≤ max_pieces :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l1217_121796


namespace NUMINAMATH_CALUDE_farm_corn_cobs_l1217_121722

/-- The number of corn cobs in a row -/
def cobs_per_row : ℕ := 4

/-- The number of rows in the first field -/
def rows_field1 : ℕ := 13

/-- The number of rows in the second field -/
def rows_field2 : ℕ := 16

/-- The total number of corn cobs grown on the farm -/
def total_cobs : ℕ := rows_field1 * cobs_per_row + rows_field2 * cobs_per_row

theorem farm_corn_cobs : total_cobs = 116 := by
  sorry

end NUMINAMATH_CALUDE_farm_corn_cobs_l1217_121722


namespace NUMINAMATH_CALUDE_remainder_theorem_l1217_121726

def dividend (b x : ℝ) : ℝ := 12 * x^3 - 9 * x^2 + b * x + 8
def divisor (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

theorem remainder_theorem (b : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, dividend b x = divisor x * q x + 10) ↔ b = -31/3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1217_121726


namespace NUMINAMATH_CALUDE_remainder_sum_l1217_121723

theorem remainder_sum (p q : ℤ) (hp : p % 60 = 47) (hq : q % 45 = 36) : (p + q) % 30 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1217_121723


namespace NUMINAMATH_CALUDE_specific_window_height_l1217_121734

/-- Represents a rectangular window with glass panes. -/
structure Window where
  num_panes : ℕ
  rows : ℕ
  columns : ℕ
  pane_height_ratio : ℚ
  pane_width_ratio : ℚ
  border_width : ℕ

/-- Calculates the height of a window given its specifications. -/
def window_height (w : Window) : ℕ :=
  let pane_width := 4 * w.border_width
  let pane_height := 3 * w.border_width
  pane_height * w.rows + w.border_width * (w.rows + 1)

/-- The theorem stating the height of the specific window. -/
theorem specific_window_height :
  let w : Window := {
    num_panes := 8,
    rows := 4,
    columns := 2,
    pane_height_ratio := 3/4,
    pane_width_ratio := 4/3,
    border_width := 3
  }
  window_height w = 51 := by sorry

end NUMINAMATH_CALUDE_specific_window_height_l1217_121734


namespace NUMINAMATH_CALUDE_second_number_proof_l1217_121720

theorem second_number_proof (x y z : ℚ) 
  (sum_eq : x + y + z = 125)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 6) :
  y = 3500 / 73 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l1217_121720


namespace NUMINAMATH_CALUDE_bus_passengers_problem_l1217_121791

/-- Proves that the initial number of people on a bus was 50, given the conditions of passenger changes at three stops. -/
theorem bus_passengers_problem (initial : ℕ) : 
  (((initial - 15) - (8 - 2)) - (4 - 3) = 28) → initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_problem_l1217_121791


namespace NUMINAMATH_CALUDE_system_solution_l1217_121749

theorem system_solution :
  ∃ (x y : ℝ), x = -1 ∧ y = -2 ∧ x - 3*y = 5 ∧ 4*x - 3*y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1217_121749


namespace NUMINAMATH_CALUDE_students_on_field_trip_l1217_121774

/-- The number of students going on a field trip --/
def students_on_trip (seats_per_bus : ℕ) (num_buses : ℕ) : ℕ :=
  seats_per_bus * num_buses

/-- Theorem: The number of students on the trip is 28 given 7 seats per bus and 4 buses --/
theorem students_on_field_trip :
  students_on_trip 7 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_students_on_field_trip_l1217_121774


namespace NUMINAMATH_CALUDE_green_face_prob_half_l1217_121750

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  purple_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def green_face_probability (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a cube with 3 green faces and 3 purple faces is 1/2 -/
theorem green_face_prob_half :
  let cube : ColoredCube := { total_faces := 6, green_faces := 3, purple_faces := 3 }
  green_face_probability cube = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_green_face_prob_half_l1217_121750


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1217_121737

theorem point_in_fourth_quadrant (a b : ℝ) : 
  let A : ℝ × ℝ := (a^2 + 1, -1 - b^2)
  A.1 > 0 ∧ A.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1217_121737


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1217_121771

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l1217_121771


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1217_121797

/-- A quadratic function f(x) = ax^2 + bx + c with vertex (4, 10) and one x-intercept at (1, 0) -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem other_x_intercept_of_quadratic 
  (a b c : ℝ) 
  (h_vertex : QuadraticFunction a b c 4 = 10 ∧ (∀ x, QuadraticFunction a b c x ≥ 10 ∨ QuadraticFunction a b c x ≤ 10))
  (h_intercept : QuadraticFunction a b c 1 = 0) :
  ∃ x, x ≠ 1 ∧ QuadraticFunction a b c x = 0 ∧ x = 7 := by
sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1217_121797


namespace NUMINAMATH_CALUDE_crayon_count_theorem_l1217_121714

/-- Represents the number of crayons in various states --/
structure CrayonCounts where
  initial : ℕ
  givenAway : ℕ
  lost : ℕ
  remaining : ℕ

/-- Theorem stating the relationship between crayons lost, given away, and the total --/
theorem crayon_count_theorem (c : CrayonCounts) 
  (h1 : c.givenAway = 52)
  (h2 : c.lost = 535)
  (h3 : c.remaining = 492) :
  c.givenAway + c.lost = 587 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_theorem_l1217_121714


namespace NUMINAMATH_CALUDE_walking_speed_problem_l1217_121768

/-- The walking speed problem -/
theorem walking_speed_problem 
  (distance_between_homes : ℝ)
  (bob_speed : ℝ)
  (alice_distance : ℝ)
  (time_difference : ℝ)
  (h1 : distance_between_homes = 41)
  (h2 : bob_speed = 4)
  (h3 : alice_distance = 25)
  (h4 : time_difference = 1)
  : ∃ (alice_speed : ℝ), 
    alice_speed = 5 ∧ 
    alice_distance / alice_speed = (distance_between_homes - alice_distance) / bob_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l1217_121768


namespace NUMINAMATH_CALUDE_max_value_of_f_l1217_121793

def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + m

theorem max_value_of_f (m : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f m x ≥ 1) →
  (∃ x ∈ Set.Icc (-2) 2, f m x = 1) →
  (∃ x ∈ Set.Icc (-2) 2, f m x = 21) ∧
  (∀ x ∈ Set.Icc (-2) 2, f m x ≤ 21) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1217_121793


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1217_121763

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 2 → x ≠ 3 → x ≠ 4 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 4)) =
      A / (x - 2) + B / (x - 3) + C / (x - 4) ∧
      A = -5/2 ∧ B = 0 ∧ C = 7/2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1217_121763


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1217_121799

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1217_121799


namespace NUMINAMATH_CALUDE_julia_baking_days_l1217_121775

/-- The number of cakes Julia bakes per day -/
def cakes_per_day : ℕ := 4

/-- The number of cakes eaten every two days -/
def cakes_eaten_per_two_days : ℕ := 1

/-- The final number of cakes remaining -/
def final_cakes : ℕ := 21

/-- The number of days Julia baked cakes -/
def baking_days : ℕ := 6

/-- Proves that the number of days Julia baked cakes is 6 -/
theorem julia_baking_days :
  baking_days * cakes_per_day - (baking_days / 2) * cakes_eaten_per_two_days = final_cakes := by
  sorry


end NUMINAMATH_CALUDE_julia_baking_days_l1217_121775


namespace NUMINAMATH_CALUDE_jimmy_folders_l1217_121736

-- Define the variables
def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5
def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def paid_amount : ℕ := 50
def change_amount : ℕ := 25

-- Define the theorem
theorem jimmy_folders :
  (paid_amount - change_amount - (num_pens * pen_cost + num_notebooks * notebook_cost)) / folder_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_folders_l1217_121736


namespace NUMINAMATH_CALUDE_solution_value_l1217_121735

theorem solution_value (a b : ℝ) : 
  (a * 1 - b * 2 + 3 = 0) →
  (a * (-1) - b * 1 + 3 = 0) →
  a - 3 * b = -5 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l1217_121735


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1217_121782

theorem price_reduction_percentage (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1217_121782


namespace NUMINAMATH_CALUDE_quadratic_solution_inequality_solution_l1217_121761

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 1 = 0

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := x + 8 < 4*x - 1 ∧ (1/2)*x ≤ 8 - (3/2)*x

-- Theorem for the quadratic equation solution
theorem quadratic_solution :
  ∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧ 
              x₂ = (5 - Real.sqrt 21) / 2 ∧
              quadratic_equation x₁ ∧ 
              quadratic_equation x₂ :=
sorry

-- Theorem for the inequality system solution
theorem inequality_solution :
  ∀ x : ℝ, inequality_system x ↔ 3 < x ∧ x ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_inequality_solution_l1217_121761


namespace NUMINAMATH_CALUDE_wind_speed_calculation_l1217_121769

/-- Wind speed calculation for a helicopter flight --/
theorem wind_speed_calculation 
  (s v : ℝ) 
  (h_positive_s : 0 < s)
  (h_positive_v : 0 < v)
  (h_v_greater_s : s < v) :
  ∃ (x y vB : ℝ),
    x + y = 2 ∧                 -- Total flight time is 2 hours
    v + vB = s / x ∧            -- Speed from A to B (with wind)
    v - vB = s / y ∧            -- Speed from B to A (against wind)
    vB = Real.sqrt (v * (v - s)) -- Wind speed formula
  := by sorry

end NUMINAMATH_CALUDE_wind_speed_calculation_l1217_121769


namespace NUMINAMATH_CALUDE_solve_system_l1217_121795

theorem solve_system (C D : ℚ) 
  (eq1 : 3 * C - 4 * D = 18)
  (eq2 : C = 2 * D - 5) : 
  C = 28 ∧ D = 33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1217_121795


namespace NUMINAMATH_CALUDE_total_buttons_eq_1600_l1217_121717

/-- The number of 3-button shirts ordered -/
def shirts_3_button : ℕ := 200

/-- The number of 5-button shirts ordered -/
def shirts_5_button : ℕ := 200

/-- The number of buttons on a 3-button shirt -/
def buttons_per_3_button_shirt : ℕ := 3

/-- The number of buttons on a 5-button shirt -/
def buttons_per_5_button_shirt : ℕ := 5

/-- The total number of buttons used for the order -/
def total_buttons : ℕ := shirts_3_button * buttons_per_3_button_shirt + shirts_5_button * buttons_per_5_button_shirt

theorem total_buttons_eq_1600 : total_buttons = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_eq_1600_l1217_121717


namespace NUMINAMATH_CALUDE_function_evaluation_l1217_121757

theorem function_evaluation :
  ((-1)^4 + (-1)^3 + 1) / ((-1)^2 + 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_evaluation_l1217_121757


namespace NUMINAMATH_CALUDE_average_weight_of_group_l1217_121746

theorem average_weight_of_group (girls_count boys_count : ℕ) 
  (girls_avg_weight boys_avg_weight : ℝ) :
  girls_count = 5 →
  boys_count = 5 →
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  let total_count := girls_count + boys_count
  let total_weight := girls_count * girls_avg_weight + boys_count * boys_avg_weight
  (total_weight / total_count : ℝ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_group_l1217_121746


namespace NUMINAMATH_CALUDE_smallest_winning_number_l1217_121702

theorem smallest_winning_number : ∃ N : ℕ, N ≤ 999 ∧ 
  (∀ m : ℕ, m < N → (16 * m + 980 > 1200 ∨ 16 * m + 1050 ≤ 1200)) ∧
  16 * N + 980 ≤ 1200 ∧ 
  16 * N + 1050 > 1200 ∧
  N = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l1217_121702


namespace NUMINAMATH_CALUDE_eight_divided_by_recurring_third_l1217_121777

theorem eight_divided_by_recurring_third (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_recurring_third_l1217_121777


namespace NUMINAMATH_CALUDE_min_sum_of_intercepts_l1217_121764

/-- A line with positive x-intercept and y-intercept passing through (1,4) -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  passes_through : 1 / a + 4 / b = 1

/-- The minimum sum of intercepts for a line passing through (1,4) is 9 -/
theorem min_sum_of_intercepts (l : InterceptLine) : l.a + l.b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_intercepts_l1217_121764


namespace NUMINAMATH_CALUDE_evaluate_expression_l1217_121770

theorem evaluate_expression : (0.5^4 - 0.25^2) / (0.1^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1217_121770


namespace NUMINAMATH_CALUDE_complex_number_equation_l1217_121740

theorem complex_number_equation : ∃ z : ℂ, z / (1 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equation_l1217_121740


namespace NUMINAMATH_CALUDE_geometric_progression_sum_ratio_l1217_121741

theorem geometric_progression_sum_ratio (a : ℝ) (n : ℕ) : 
  let r : ℝ := 3
  let S_n := a * (1 - r^n) / (1 - r)
  let S_3 := a * (1 - r^3) / (1 - r)
  S_n / S_3 = 28 → n = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_ratio_l1217_121741


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1217_121729

theorem coin_flip_probability (oliver_prob jayden_prob mia_prob : ℚ) :
  oliver_prob = 1/3 →
  jayden_prob = 1/4 →
  mia_prob = 1/5 →
  (∑' n : ℕ, (1 - oliver_prob)^(n-1) * oliver_prob *
              (1 - jayden_prob)^(n-1) * jayden_prob *
              (1 - mia_prob)^(n-1) * mia_prob) = 1/36 := by
  sorry

#check coin_flip_probability

end NUMINAMATH_CALUDE_coin_flip_probability_l1217_121729


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1217_121733

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2*a - 1 = 5) :
  -2*a^2 - 4*a + 5 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1217_121733


namespace NUMINAMATH_CALUDE_min_sixth_graders_l1217_121721

theorem min_sixth_graders (x : ℕ) (hx : x > 0) : 
  let girls := x / 3
  let boys := x - girls
  let sixth_grade_girls := girls / 2
  let non_sixth_grade_boys := (boys * 5) / 7
  let sixth_grade_boys := boys - non_sixth_grade_boys
  let total_sixth_graders := sixth_grade_girls + sixth_grade_boys
  x % 3 = 0 ∧ girls % 2 = 0 ∧ boys % 7 = 0 →
  ∀ y : ℕ, y > 0 ∧ y < x ∧ 
    (let girls_y := y / 3
     let boys_y := y - girls_y
     let sixth_grade_girls_y := girls_y / 2
     let non_sixth_grade_boys_y := (boys_y * 5) / 7
     let sixth_grade_boys_y := boys_y - non_sixth_grade_boys_y
     let total_sixth_graders_y := sixth_grade_girls_y + sixth_grade_boys_y
     y % 3 = 0 ∧ girls_y % 2 = 0 ∧ boys_y % 7 = 0) →
    total_sixth_graders_y < total_sixth_graders →
  total_sixth_graders = 15 := by
sorry

end NUMINAMATH_CALUDE_min_sixth_graders_l1217_121721


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_100000_l1217_121712

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of all numbers from 1 to 100000 -/
theorem sum_of_digits_up_to_100000 : sumOfDigitsUpTo 100000 = 2443446 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_100000_l1217_121712


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l1217_121744

/-- Given a right-angled triangle with catheti of lengths a and b, and a circle
    with radius r inscribed such that it touches both catheti and has its center
    on the hypotenuse, prove that 1/a + 1/b = 1/r. -/
theorem inscribed_circle_theorem (a b r : ℝ) 
    (ha : a > 0) (hb : b > 0) (hr : r > 0)
    (h_right_triangle : ∃ c, a^2 + b^2 = c^2)
    (h_circle_inscribed : ∃ x y, x^2 + y^2 = r^2 ∧ x + y = r ∧ x < a ∧ y < b) :
    1/a + 1/b = 1/r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l1217_121744


namespace NUMINAMATH_CALUDE_acquainted_pairs_bound_l1217_121780

/-- Represents a company with n persons, where each person has no more than d acquaintances,
    and there exists a group of k persons (k ≥ d) who are not acquainted with each other. -/
structure Company where
  n : ℕ  -- Total number of persons
  d : ℕ  -- Maximum number of acquaintances per person
  k : ℕ  -- Size of the group of unacquainted persons
  h1 : k ≥ d  -- Condition that k is not less than d

/-- The number of acquainted pairs in the company -/
def acquaintedPairs (c : Company) : ℕ := sorry

/-- Theorem stating that the number of acquainted pairs is not greater than ⌊n²/4⌋ -/
theorem acquainted_pairs_bound (c : Company) : 
  acquaintedPairs c ≤ (c.n^2) / 4 := by sorry

end NUMINAMATH_CALUDE_acquainted_pairs_bound_l1217_121780


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l1217_121789

def num_tan_chips : ℕ := 4
def num_pink_chips : ℕ := 4
def num_violet_chips : ℕ := 3
def total_chips : ℕ := num_tan_chips + num_pink_chips + num_violet_chips

theorem consecutive_color_draw_probability :
  (2 * (num_tan_chips.factorial * num_pink_chips.factorial * num_violet_chips.factorial)) / 
  total_chips.factorial = 1 / 5760 := by sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l1217_121789


namespace NUMINAMATH_CALUDE_journey_portions_l1217_121743

/-- Proves that the journey is divided into 5 portions given the conditions -/
theorem journey_portions (total_distance : ℝ) (speed : ℝ) (time : ℝ) (portions_covered : ℕ) :
  total_distance = 35 →
  speed = 40 →
  time = 0.7 →
  portions_covered = 4 →
  (speed * time) / portions_covered = total_distance / 5 :=
by sorry

end NUMINAMATH_CALUDE_journey_portions_l1217_121743


namespace NUMINAMATH_CALUDE_ratio_of_q_r_to_p_l1217_121776

def p : ℝ := 47.99999999999999

theorem ratio_of_q_r_to_p : ∃ (f : ℝ), f = 1/6 ∧ 2 * f * p = p - 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_q_r_to_p_l1217_121776


namespace NUMINAMATH_CALUDE_candidate_a_democratic_votes_l1217_121790

theorem candidate_a_democratic_votes 
  (total_voters : ℝ) 
  (dem_percent : ℝ) 
  (rep_percent : ℝ) 
  (rep_for_a_percent : ℝ) 
  (total_for_a_percent : ℝ) 
  (h1 : dem_percent = 0.60)
  (h2 : rep_percent = 1 - dem_percent)
  (h3 : rep_for_a_percent = 0.20)
  (h4 : total_for_a_percent = 0.47) :
  let dem_for_a_percent := (total_for_a_percent * total_voters - rep_for_a_percent * rep_percent * total_voters) / (dem_percent * total_voters)
  dem_for_a_percent = 0.65 := by
sorry

end NUMINAMATH_CALUDE_candidate_a_democratic_votes_l1217_121790


namespace NUMINAMATH_CALUDE_max_coefficients_bound_l1217_121767

variable (p q x y A B C α β γ : ℝ)

theorem max_coefficients_bound 
  (h_p : 0 ≤ p ∧ p ≤ 1) 
  (h_q : 0 ≤ q ∧ q ≤ 1) 
  (h_eq1 : ∀ x y, (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2)
  (h_eq2 : ∀ x y, (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2) :
  max A (max B C) ≥ 4/9 ∧ max α (max β γ) ≥ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_max_coefficients_bound_l1217_121767


namespace NUMINAMATH_CALUDE_negation_equivalence_l1217_121794

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m < 0) ↔ (∀ x : ℤ, x^2 + 2*x + m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1217_121794


namespace NUMINAMATH_CALUDE_triangle_properties_l1217_121731

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions and the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.c = abc.a)
  (h2 : abc.c = Real.sqrt 3)
  (h3 : Real.sin abc.B ^ 2 = 2 * Real.sin abc.A * Real.sin abc.C) :
  Real.cos abc.B = 0 ∧ (1/2 * abc.a * abc.c * Real.sin abc.B = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1217_121731


namespace NUMINAMATH_CALUDE_discount_profit_percentage_l1217_121772

theorem discount_profit_percentage 
  (discount : Real) 
  (no_discount_profit : Real) 
  (h1 : discount = 0.05) 
  (h2 : no_discount_profit = 0.26) : 
  let marked_price := 1 + no_discount_profit
  let selling_price := marked_price * (1 - discount)
  let profit := selling_price - 1
  profit * 100 = 19.7 := by
sorry

end NUMINAMATH_CALUDE_discount_profit_percentage_l1217_121772


namespace NUMINAMATH_CALUDE_star_cell_is_one_l1217_121781

/-- Represents a 4x4 grid of natural numbers -/
def Grid := Fin 4 → Fin 4 → Nat

/-- Check if all numbers in the grid are nonzero -/
def all_nonzero (g : Grid) : Prop :=
  ∀ i j, g i j ≠ 0

/-- Calculate the product of a row -/
def row_product (g : Grid) (i : Fin 4) : Nat :=
  (g i 0) * (g i 1) * (g i 2) * (g i 3)

/-- Calculate the product of a column -/
def col_product (g : Grid) (j : Fin 4) : Nat :=
  (g 0 j) * (g 1 j) * (g 2 j) * (g 3 j)

/-- Calculate the product of the main diagonal -/
def main_diag_product (g : Grid) : Nat :=
  (g 0 0) * (g 1 1) * (g 2 2) * (g 3 3)

/-- Calculate the product of the anti-diagonal -/
def anti_diag_product (g : Grid) : Nat :=
  (g 0 3) * (g 1 2) * (g 2 1) * (g 3 0)

/-- Check if all products are equal -/
def all_products_equal (g : Grid) : Prop :=
  let p := row_product g 0
  (∀ i, row_product g i = p) ∧
  (∀ j, col_product g j = p) ∧
  (main_diag_product g = p) ∧
  (anti_diag_product g = p)

/-- The main theorem -/
theorem star_cell_is_one (g : Grid) 
  (h1 : all_nonzero g)
  (h2 : all_products_equal g)
  (h3 : g 1 1 = 2)
  (h4 : g 1 2 = 16)
  (h5 : g 2 1 = 8)
  (h6 : g 2 2 = 32) :
  g 1 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_star_cell_is_one_l1217_121781


namespace NUMINAMATH_CALUDE_vincent_stickers_l1217_121779

/-- The number of packs Vincent bought yesterday -/
def yesterday_packs : ℕ := 15

/-- The total number of packs Vincent has -/
def total_packs : ℕ := 40

/-- The number of additional packs Vincent bought today -/
def additional_packs : ℕ := total_packs - yesterday_packs

theorem vincent_stickers :
  additional_packs = 10 ∧ additional_packs > 0 := by
  sorry

end NUMINAMATH_CALUDE_vincent_stickers_l1217_121779


namespace NUMINAMATH_CALUDE_circle_equation_l1217_121773

theorem circle_equation (A B : ℝ × ℝ) (h_A : A = (4, 2)) (h_B : B = (-1, 3)) :
  ∃ (D E F : ℝ),
    (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 ↔ 
      ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ 
       ∃ (x1 x2 y1 y2 : ℝ), 
         x1 + x2 + y1 + y2 = 2 ∧
         x1^2 + D*x1 + F = 0 ∧
         x2^2 + D*x2 + F = 0 ∧
         y1^2 + E*y1 + F = 0 ∧
         y2^2 + E*y2 + F = 0)) →
    D = -2 ∧ E = 0 ∧ F = -12 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1217_121773


namespace NUMINAMATH_CALUDE_school_expansion_theorem_l1217_121788

/-- Calculates the total number of students after adding classes to a school -/
def total_students_after_adding_classes 
  (initial_classes : ℕ) 
  (students_per_class : ℕ) 
  (added_classes : ℕ) : ℕ :=
  (initial_classes + added_classes) * students_per_class

/-- Theorem: A school with 15 initial classes of 20 students each, 
    after adding 5 more classes, will have 400 students in total -/
theorem school_expansion_theorem : 
  total_students_after_adding_classes 15 20 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_expansion_theorem_l1217_121788


namespace NUMINAMATH_CALUDE_half_life_radioactive_substance_l1217_121753

/-- The half-life of a radioactive substance with an 8% annual decay rate -/
theorem half_life_radioactive_substance (a : ℝ) (t : ℝ) : 
  (∀ x : ℝ, x > 0 → a * (1 - 0.08)^x = a / 2 ↔ x = t) → 
  t = Real.log 0.5 / Real.log 0.92 :=
by sorry

end NUMINAMATH_CALUDE_half_life_radioactive_substance_l1217_121753


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1217_121783

theorem possible_values_of_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 8) + 1 = (x + b) * (x + c)) → 
  (a = 6 ∨ a = 10) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1217_121783


namespace NUMINAMATH_CALUDE_wage_payment_problem_l1217_121756

/-- Given a sum of money that can pay A's wages for 20 days and both A and B's wages for 12 days,
    prove that it can pay B's wages for 30 days. -/
theorem wage_payment_problem (total_sum : ℝ) (wage_A wage_B : ℝ) 
  (h1 : total_sum = 20 * wage_A)
  (h2 : total_sum = 12 * (wage_A + wage_B)) :
  total_sum = 30 * wage_B :=
by sorry

end NUMINAMATH_CALUDE_wage_payment_problem_l1217_121756


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l1217_121715

/-- Given a cubic equation x^3 - ax^2 + bx - c = 0 with roots r, s, and t,
    prove that r^2 + s^2 + t^2 = a^2 - 2b -/
theorem cubic_root_sum_squares (a b c r s t : ℝ) : 
  (r^3 - a*r^2 + b*r - c = 0) → 
  (s^3 - a*s^2 + b*s - c = 0) → 
  (t^3 - a*t^2 + b*t - c = 0) → 
  r^2 + s^2 + t^2 = a^2 - 2*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l1217_121715


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1217_121725

theorem greatest_distance_between_circle_centers 
  (circle_diameter : ℝ) 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ) 
  (h_diameter : circle_diameter = 8)
  (h_length : rectangle_length = 20)
  (h_width : rectangle_width = 16)
  (h_tangent : circle_diameter ≤ rectangle_width) :
  let circle_radius := circle_diameter / 2
  let horizontal_distance := 2 * circle_radius
  let vertical_distance := rectangle_width
  ∃ (max_distance : ℝ), 
    max_distance = (horizontal_distance^2 + vertical_distance^2).sqrt ∧
    max_distance = 8 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1217_121725


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1217_121748

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1217_121748


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1217_121738

theorem min_value_trigonometric_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  1 / (Real.sin θ)^2 + 9 / (Real.cos θ)^2 ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1217_121738


namespace NUMINAMATH_CALUDE_root_product_theorem_l1217_121747

theorem root_product_theorem (x₁ x₂ x₃ : ℝ) : 
  (Real.sqrt 2025 * x₁^3 - 4050 * x₁^2 + 4 = 0) →
  (Real.sqrt 2025 * x₂^3 - 4050 * x₂^2 + 4 = 0) →
  (Real.sqrt 2025 * x₃^3 - 4050 * x₃^2 + 4 = 0) →
  x₁ < x₂ → x₂ < x₃ →
  x₂ * (x₁ + x₃) = 90 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l1217_121747


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1217_121707

theorem sum_of_coefficients (x : ℝ) :
  ∃ (A B C D E : ℝ),
    125 * x^3 + 64 = (A * x + B) * (C * x^2 + D * x + E) ∧
    A + B + C + D + E = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1217_121707


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1217_121754

-- Define a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_perpendicular_to_plane_are_parallel 
  (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular l1 p) (h2 : perpendicular l2 p) : 
  parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1217_121754


namespace NUMINAMATH_CALUDE_divisors_of_8n_cubed_l1217_121727

theorem divisors_of_8n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 12) :
  (Nat.divisors (8 * n^3)).card = 280 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_8n_cubed_l1217_121727


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l1217_121755

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (α β : Plane) (m : Line) 
  (h1 : parallel_planes α β) 
  (h2 : contained_in m β) : 
  parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l1217_121755


namespace NUMINAMATH_CALUDE_open_box_volume_l1217_121751

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_square_side : ℝ) 
  (h1 : sheet_length = 50) 
  (h2 : sheet_width = 36) 
  (h3 : cut_square_side = 8) : 
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 5440 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l1217_121751


namespace NUMINAMATH_CALUDE_profit_at_80_max_profit_profit_range_l1217_121706

/-- Represents the clothing sale scenario with given constraints -/
structure ClothingSale where
  cost : ℝ
  demand : ℝ → ℝ
  profit_function : ℝ → ℝ
  max_profit_percentage : ℝ

/-- The specific clothing sale scenario from the problem -/
def sale : ClothingSale :=
  { cost := 60
  , demand := λ x => -x + 120
  , profit_function := λ x => (x - 60) * (-x + 120)
  , max_profit_percentage := 0.4 }

/-- Theorem stating the profit when selling price is 80 -/
theorem profit_at_80 (s : ClothingSale) (h : s = sale) :
  s.profit_function 80 = 800 :=
sorry

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem max_profit (s : ClothingSale) (h : s = sale) :
  ∃ x, x ≤ (1 + s.max_profit_percentage) * s.cost ∧
      s.profit_function x = 864 ∧
      ∀ y, y ≤ (1 + s.max_profit_percentage) * s.cost →
        s.profit_function y ≤ s.profit_function x :=
sorry

/-- Theorem stating the range of selling prices for profit not less than 500 -/
theorem profit_range (s : ClothingSale) (h : s = sale) :
  ∀ x, s.cost ≤ x ∧ x ≤ (1 + s.max_profit_percentage) * s.cost →
    (s.profit_function x ≥ 500 ↔ 70 ≤ x ∧ x ≤ 84) :=
sorry

end NUMINAMATH_CALUDE_profit_at_80_max_profit_profit_range_l1217_121706


namespace NUMINAMATH_CALUDE_age_ratio_simplified_l1217_121701

theorem age_ratio_simplified (kul_age saras_age : ℕ) 
  (h1 : kul_age = 22) 
  (h2 : saras_age = 33) : 
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ saras_age * b = kul_age * a :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_simplified_l1217_121701


namespace NUMINAMATH_CALUDE_arcsin_negative_one_l1217_121719

theorem arcsin_negative_one : Real.arcsin (-1) = -π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_negative_one_l1217_121719


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1217_121718

theorem sqrt_difference_equality (x a : ℝ) (m n : ℤ) (h : 0 < a) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : x + Real.sqrt (x^2 - 1) = a^((m - n : ℝ) / (2 * m * n : ℝ))) :
  x - Real.sqrt (x^2 - 1) = a^((n - m : ℝ) / (2 * m * n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1217_121718


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l1217_121713

/-- The function f that we want to prove fits the data points -/
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

/-- The set of data points given in the problem -/
def data_points : List (ℝ × ℝ) := [(1, 5), (2, 11), (3, 19), (4, 29), (5, 41)]

/-- Theorem stating that f satisfies all given data points -/
theorem f_satisfies_data_points : ∀ (point : ℝ × ℝ), point ∈ data_points → f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_data_points_l1217_121713


namespace NUMINAMATH_CALUDE_scale_length_theorem_l1217_121700

/-- A scale divided into equal parts -/
structure Scale where
  num_parts : ℕ
  part_length : ℝ

/-- The total length of a scale -/
def total_length (s : Scale) : ℝ := s.num_parts * s.part_length

/-- Theorem stating that a scale with 2 parts of 40 inches each has a total length of 80 inches -/
theorem scale_length_theorem :
  ∀ (s : Scale), s.num_parts = 2 ∧ s.part_length = 40 → total_length s = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_scale_length_theorem_l1217_121700


namespace NUMINAMATH_CALUDE_fraction_denominator_l1217_121709

theorem fraction_denominator (n : ℕ) (d : ℕ) (h1 : n = 325) (h2 : (n : ℚ) / d = 1 / 8) 
  (h3 : ∃ (seq : ℕ → ℕ), (∀ k, seq k < 10) ∧ 
    (∀ k, ((n : ℚ) / d - (n : ℚ) / d).floor + (seq k) / 10^(k+1) = ((n : ℚ) / d * 10^(k+1)).floor / 10^(k+1)) ∧ 
    seq 80 = 5) : 
  d = 8 := by sorry

end NUMINAMATH_CALUDE_fraction_denominator_l1217_121709


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l1217_121752

theorem two_digit_number_sum (n : ℕ) : 
  10 ≤ n ∧ n < 100 →  -- n is a two-digit number
  (n : ℚ) / 2 = n / 4 + 3 →  -- one half of n exceeds its one fourth by 3
  (n / 10 + n % 10 : ℕ) = 12  -- sum of digits is 12
  := by sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l1217_121752


namespace NUMINAMATH_CALUDE_unique_solution_inequality_system_l1217_121745

theorem unique_solution_inequality_system :
  ∃! (a b : ℤ), 11 > 2 * a - b ∧
                25 > 2 * b - a ∧
                42 < 3 * b - a ∧
                46 < 2 * a + b :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_system_l1217_121745


namespace NUMINAMATH_CALUDE_smallest_number_with_all_factors_l1217_121708

def alice_number : ℕ := 90

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ n → p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ m : ℕ, m > 0 ∧ has_all_prime_factors alice_number m ∧
  ∀ k : ℕ, k > 0 → has_all_prime_factors alice_number k → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_all_factors_l1217_121708


namespace NUMINAMATH_CALUDE_f1_not_unique_l1217_121724

-- Define the type of our functions
def F := ℝ → ℝ

-- Define the recursive relationship
def recursive_relation (f₁ : F) (n : ℕ) : F :=
  match n with
  | 0 => id
  | 1 => f₁
  | n + 2 => f₁ ∘ (recursive_relation f₁ (n + 1))

-- State the theorem
theorem f1_not_unique :
  ∃ (f₁ g₁ : F),
    f₁ ≠ g₁ ∧
    (∀ (n : ℕ), n ≥ 2 → (recursive_relation f₁ n) = f₁ ∘ (recursive_relation f₁ (n - 1))) ∧
    (∀ (n : ℕ), n ≥ 2 → (recursive_relation g₁ n) = g₁ ∘ (recursive_relation g₁ (n - 1))) ∧
    (recursive_relation f₁ 5) 2 = 33 ∧
    (recursive_relation g₁ 5) 2 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_f1_not_unique_l1217_121724


namespace NUMINAMATH_CALUDE_total_points_zach_and_ben_l1217_121760

theorem total_points_zach_and_ben (zach_points ben_points : ℝ) 
  (h1 : zach_points = 42.0) 
  (h2 : ben_points = 21.0) : 
  zach_points + ben_points = 63.0 := by
  sorry

end NUMINAMATH_CALUDE_total_points_zach_and_ben_l1217_121760


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l1217_121766

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : CartesianPoint :=
  { x := 2, y := -5 }

/-- Theorem stating that P is in the fourth quadrant -/
theorem P_in_fourth_quadrant : is_in_fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l1217_121766


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1217_121728

def number : Nat := 219257

theorem largest_prime_factors_difference (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ number ∧ q ∣ number ∧
  ∀ r, Nat.Prime r → r ∣ number → r ≤ p ∧ r ≤ q →
  p - q = 144 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1217_121728


namespace NUMINAMATH_CALUDE_greatest_common_factor_4050_12320_l1217_121711

theorem greatest_common_factor_4050_12320 : Nat.gcd 4050 12320 = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_4050_12320_l1217_121711


namespace NUMINAMATH_CALUDE_ball_color_difference_l1217_121792

theorem ball_color_difference (m n : ℕ) (h1 : m > n) (h2 : n > 0) :
  (m * (m - 1) + n * (n - 1) : ℚ) / ((m + n) * (m + n - 1)) = 
  (2 * m * n : ℚ) / ((m + n) * (m + n - 1)) →
  ∃ a : ℕ, a > 1 ∧ m - n = a :=
sorry

end NUMINAMATH_CALUDE_ball_color_difference_l1217_121792


namespace NUMINAMATH_CALUDE_chicken_farm_theorem_l1217_121762

/-- Represents the chicken farm problem -/
structure ChickenFarm where
  totalChicks : ℕ
  costA : ℕ
  costB : ℕ
  survivalRateA : ℚ
  survivalRateB : ℚ

/-- The solution to the chicken farm problem -/
def chickenFarmSolution (farm : ChickenFarm) : Prop :=
  -- Total number of chicks is 2000
  farm.totalChicks = 2000 ∧
  -- Cost of type A chick is 2 yuan
  farm.costA = 2 ∧
  -- Cost of type B chick is 3 yuan
  farm.costB = 3 ∧
  -- Survival rate of type A chicks is 94%
  farm.survivalRateA = 94/100 ∧
  -- Survival rate of type B chicks is 99%
  farm.survivalRateB = 99/100 ∧
  -- Question 1
  (∃ (x y : ℕ), x + y = farm.totalChicks ∧ 
    farm.costA * x + farm.costB * y = 4500 ∧
    x = 1500 ∧ y = 500) ∧
  -- Question 2
  (∃ (x : ℕ), x ≥ 1300 ∧
    ∀ (y : ℕ), y + x = farm.totalChicks →
      farm.costA * x + farm.costB * y ≤ 4700) ∧
  -- Question 3
  (∃ (x y : ℕ), x + y = farm.totalChicks ∧
    farm.survivalRateA * x + farm.survivalRateB * y ≥ 96/100 * farm.totalChicks ∧
    x = 1200 ∧ y = 800 ∧
    farm.costA * x + farm.costB * y = 4800 ∧
    ∀ (x' y' : ℕ), x' + y' = farm.totalChicks →
      farm.survivalRateA * x' + farm.survivalRateB * y' ≥ 96/100 * farm.totalChicks →
      farm.costA * x' + farm.costB * y' ≥ 4800)

theorem chicken_farm_theorem (farm : ChickenFarm) : chickenFarmSolution farm := by
  sorry

end NUMINAMATH_CALUDE_chicken_farm_theorem_l1217_121762
