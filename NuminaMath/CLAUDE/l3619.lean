import Mathlib

namespace NUMINAMATH_CALUDE_frank_hamburger_sales_l3619_361928

/-- The number of additional hamburgers Frank needs to sell to reach his target revenue -/
def additional_hamburgers (target_revenue : ℕ) (price_per_hamburger : ℕ) (initial_sales : ℕ) : ℕ :=
  (target_revenue - price_per_hamburger * initial_sales) / price_per_hamburger

theorem frank_hamburger_sales : additional_hamburgers 50 5 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_hamburger_sales_l3619_361928


namespace NUMINAMATH_CALUDE_sequence_divisibility_l3619_361980

def a (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def divides_sequence (p : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 1 ∧ p ∣ a n

theorem sequence_divisibility :
  {p : ℕ | p.Prime ∧ p ≤ 19 ∧ divides_sequence p} = {3, 7, 13, 17} := by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l3619_361980


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3619_361936

/-- The line y - 1 = k(x + 2) passes through the point (-2, 1) for all values of k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (1 : ℝ) - 1 = k * ((-2 : ℝ) + 2) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3619_361936


namespace NUMINAMATH_CALUDE_existence_of_floor_representation_l3619_361943

def is_valid_sequence (f : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, 1 ≤ i → 1 ≤ j → i + j ≤ 1997 →
    f i + f j ≤ f (i + j) ∧ f (i + j) ≤ f i + f j + 1

theorem existence_of_floor_representation (f : ℕ → ℕ) :
  is_valid_sequence f →
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → f n = ⌊n * x⌋ :=
sorry

end NUMINAMATH_CALUDE_existence_of_floor_representation_l3619_361943


namespace NUMINAMATH_CALUDE_red_team_score_l3619_361920

theorem red_team_score (chuck_team_score : ℕ) (score_difference : ℕ) :
  chuck_team_score = 95 →
  score_difference = 19 →
  chuck_team_score - score_difference = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_red_team_score_l3619_361920


namespace NUMINAMATH_CALUDE_zoo_trip_remaining_money_l3619_361981

/-- Calculates the amount of money left for lunch and snacks after a zoo trip -/
theorem zoo_trip_remaining_money 
  (ticket_price : ℚ)
  (bus_fare : ℚ)
  (total_money : ℚ)
  (num_people : ℕ)
  (h1 : ticket_price = 5)
  (h2 : bus_fare = 3/2)
  (h3 : total_money = 40)
  (h4 : num_people = 2)
  : total_money - (num_people * ticket_price + num_people * bus_fare * 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_remaining_money_l3619_361981


namespace NUMINAMATH_CALUDE_number_divided_by_five_l3619_361904

theorem number_divided_by_five (x : ℝ) : x - 5 = 35 → x / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_five_l3619_361904


namespace NUMINAMATH_CALUDE_village_population_equality_l3619_361982

theorem village_population_equality (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) :
  x_initial = 72000 →
  x_rate = 1200 →
  y_initial = 42000 →
  y_rate = 800 →
  ∃ n : ℕ, (x_initial - n * x_rate = y_initial + n * y_rate) ∧ n = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l3619_361982


namespace NUMINAMATH_CALUDE_p_iff_q_l3619_361907

-- Define the propositions
def p (a : ℝ) : Prop := a = -1

def q (a : ℝ) : Prop := ∀ (x y : ℝ), (a * x + y + 1 = 0) ↔ (x + a * y + 2 * a - 1 = 0)

-- State the theorem
theorem p_iff_q : ∀ (a : ℝ), p a ↔ q a := by sorry

end NUMINAMATH_CALUDE_p_iff_q_l3619_361907


namespace NUMINAMATH_CALUDE_floor_shorter_side_l3619_361938

/-- The length of the shorter side of a rectangular floor -/
def shorter_side_length : ℝ := 6

theorem floor_shorter_side :
  let floor_width : ℝ := 10
  let carpet_square_side : ℝ := 2
  let carpet_square_cost : ℝ := 15
  let total_cost : ℝ := 225
  let num_squares : ℝ := total_cost / carpet_square_cost
  let total_area : ℝ := num_squares * carpet_square_side^2
  shorter_side_length * floor_width = total_area := by sorry

end NUMINAMATH_CALUDE_floor_shorter_side_l3619_361938


namespace NUMINAMATH_CALUDE_pistachios_with_opened_shells_l3619_361995

/-- Given a bag of pistachios, calculate the number of pistachios with shells and opened shells -/
theorem pistachios_with_opened_shells
  (total : ℕ)
  (shell_percent : ℚ)
  (opened_percent : ℚ)
  (h_total : total = 80)
  (h_shell : shell_percent = 95 / 100)
  (h_opened : opened_percent = 75 / 100) :
  ⌊(total : ℚ) * shell_percent * opened_percent⌋ = 57 := by
sorry

end NUMINAMATH_CALUDE_pistachios_with_opened_shells_l3619_361995


namespace NUMINAMATH_CALUDE_problem_solution_l3619_361914

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - x + 2

def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x : ℝ, x > 0 → x < 1/Real.exp 1 → (deriv f) x < 0) ∧
  (m > 1/Real.exp 1 → ∀ x : ℝ, 1/Real.exp 1 < x → x < m → (deriv f) x > 0) ∧
  (∀ x : ℝ, x > 0 → 2 * f x ≤ (deriv (g (-2))) x + 2) ∧
  (∀ a : ℝ, a ≥ -2 → ∀ x : ℝ, x > 0 → 2 * f x ≤ (deriv (g a)) x + 2) ∧
  tangent_line (0 : ℝ) (g 1 0) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3619_361914


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3619_361956

theorem hyperbola_conjugate_axis_length :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧
  2 * b = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3619_361956


namespace NUMINAMATH_CALUDE_square_minus_floor_product_l3619_361918

/-- The floor function, which returns the greatest integer less than or equal to a given real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

/-- Theorem stating that for A = 50 + 19√7, A^2 - A⌊A⌋ = 27 -/
theorem square_minus_floor_product (A : ℝ) (h : A = 50 + 19 * Real.sqrt 7) :
  A^2 - A * (floor A) = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_floor_product_l3619_361918


namespace NUMINAMATH_CALUDE_unique_composite_with_special_divisor_property_l3619_361905

theorem unique_composite_with_special_divisor_property :
  ∃! (n : ℕ), 
    n > 1 ∧ 
    ¬(Nat.Prime n) ∧
    (∃ (k : ℕ) (d : ℕ → ℕ), 
      d 1 = 1 ∧ d k = n ∧
      (∀ i, 1 ≤ i → i < k → d i < d (i+1)) ∧
      (∀ i, 1 ≤ i → i < k → d i ∣ n) ∧
      (∀ i, 1 < i → i ≤ k → (d i - d (i-1)) = i * (d 2 - d 1))) ∧
    n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_composite_with_special_divisor_property_l3619_361905


namespace NUMINAMATH_CALUDE_fraction_simplification_l3619_361986

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3619_361986


namespace NUMINAMATH_CALUDE_solve_system_l3619_361931

theorem solve_system (a b : ℤ) 
  (eq1 : 2013 * a + 2015 * b = 2023)
  (eq2 : 2017 * a + 2019 * b = 2027) :
  a - b = -9 := by sorry

end NUMINAMATH_CALUDE_solve_system_l3619_361931


namespace NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l3619_361974

theorem max_consecutive_sum_less_than_1000 :
  ∀ n : ℕ, n > 0 → (n * (n + 1) / 2 < 1000 ↔ n ≤ 44) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l3619_361974


namespace NUMINAMATH_CALUDE_max_rented_trucks_is_twenty_l3619_361961

/-- Represents the truck rental scenario for a week -/
structure TruckRental where
  total : ℕ
  returned_percent : ℚ
  saturday_minimum : ℕ

/-- Calculates the maximum number of trucks that could have been rented out -/
def max_rented_trucks (rental : TruckRental) : ℕ :=
  min rental.total (2 * rental.saturday_minimum)

/-- Theorem stating the maximum number of trucks that could have been rented out -/
theorem max_rented_trucks_is_twenty (rental : TruckRental) :
    rental.total = 20 ∧ 
    rental.returned_percent = 1/2 ∧ 
    rental.saturday_minimum = 10 →
    max_rented_trucks rental = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_rented_trucks_is_twenty_l3619_361961


namespace NUMINAMATH_CALUDE_cat_finishes_food_on_sunday_l3619_361946

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the amount of food eaten by the cat -/
def cat_food_eaten (d : Day) : Rat :=
  match d with
  | Day.Monday => 5/6
  | Day.Tuesday => 10/6
  | Day.Wednesday => 15/6
  | Day.Thursday => 20/6
  | Day.Friday => 25/6
  | Day.Saturday => 30/6
  | Day.Sunday => 35/6

theorem cat_finishes_food_on_sunday :
  ∀ d : Day, cat_food_eaten d ≤ 9 ∧
  (d = Day.Sunday → cat_food_eaten d > 54/6) :=
by sorry

#check cat_finishes_food_on_sunday

end NUMINAMATH_CALUDE_cat_finishes_food_on_sunday_l3619_361946


namespace NUMINAMATH_CALUDE_five_integers_with_remainder_one_l3619_361978

theorem five_integers_with_remainder_one : 
  ∃! (S : Finset ℕ), 
    S.card = 5 ∧ 
    (∀ n ∈ S, n ≤ 50) ∧ 
    (∀ n ∈ S, n % 11 = 1) :=
by sorry

end NUMINAMATH_CALUDE_five_integers_with_remainder_one_l3619_361978


namespace NUMINAMATH_CALUDE_student_decrease_percentage_l3619_361972

theorem student_decrease_percentage
  (initial_students : ℝ)
  (initial_price : ℝ)
  (price_increase : ℝ)
  (consumption_decrease : ℝ)
  (h1 : price_increase = 0.20)
  (h2 : consumption_decrease = 0.074074074074074066)
  (h3 : initial_students > 0)
  (h4 : initial_price > 0) :
  let new_price := initial_price * (1 + price_increase)
  let new_consumption := 1 - consumption_decrease
  let new_students := initial_students * (1 - 0.10)
  initial_students * initial_price = new_students * new_price * new_consumption :=
by sorry

end NUMINAMATH_CALUDE_student_decrease_percentage_l3619_361972


namespace NUMINAMATH_CALUDE_electrons_gained_by_oxidizing_agent_l3619_361917

-- Define the redox reaction components
structure RedoxReaction where
  cu_io3_2 : ℕ
  ki : ℕ
  h2so4 : ℕ
  cui : ℕ
  i2 : ℕ
  k2so4 : ℕ
  h2o : ℕ

-- Define the valence changes
structure ValenceChanges where
  cu_initial : ℤ
  cu_final : ℤ
  i_initial : ℤ
  i_final : ℤ

-- Define the function to calculate electron moles gained
def electronMolesGained (vc : ValenceChanges) : ℤ :=
  (vc.cu_initial - vc.cu_final) + 2 * (vc.i_initial - vc.i_final)

-- Theorem statement
theorem electrons_gained_by_oxidizing_agent 
  (reaction : RedoxReaction)
  (valence_changes : ValenceChanges)
  (h1 : reaction.cu_io3_2 = 2)
  (h2 : reaction.ki = 24)
  (h3 : reaction.h2so4 = 12)
  (h4 : reaction.cui = 2)
  (h5 : reaction.i2 = 13)
  (h6 : reaction.k2so4 = 12)
  (h7 : reaction.h2o = 12)
  (h8 : valence_changes.cu_initial = 2)
  (h9 : valence_changes.cu_final = 1)
  (h10 : valence_changes.i_initial = 5)
  (h11 : valence_changes.i_final = 0) :
  electronMolesGained valence_changes = 11 := by
  sorry

end NUMINAMATH_CALUDE_electrons_gained_by_oxidizing_agent_l3619_361917


namespace NUMINAMATH_CALUDE_largest_divisor_n_plus_10_divisibility_condition_l3619_361911

theorem largest_divisor_n_plus_10 :
  ∀ n : ℕ, n > 0 → (n + 10) ∣ (n^3 + 2011) → n ≤ 1001 :=
by sorry

theorem divisibility_condition :
  (1001 + 10) ∣ (1001^3 + 2011) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_n_plus_10_divisibility_condition_l3619_361911


namespace NUMINAMATH_CALUDE_students_in_band_or_sports_l3619_361913

theorem students_in_band_or_sports
  (total : ℕ)
  (band : ℕ)
  (sports : ℕ)
  (both : ℕ)
  (h1 : total = 320)
  (h2 : band = 85)
  (h3 : sports = 200)
  (h4 : both = 60) :
  band + sports - both = 225 :=
by sorry

end NUMINAMATH_CALUDE_students_in_band_or_sports_l3619_361913


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3619_361968

-- Define the sets A and B
def A : Set ℝ := {x | (2*x + 3)/(x - 2) > 0}
def B : Set ℝ := {x | |x - 1| < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3619_361968


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2022_l3619_361900

theorem smallest_n_divisible_by_2022 :
  ∃ (n : ℕ), n > 1 ∧ n^7 - 1 % 2022 = 0 ∧
  ∀ (m : ℕ), m > 1 ∧ m < n → m^7 - 1 % 2022 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2022_l3619_361900


namespace NUMINAMATH_CALUDE_power_identity_l3619_361901

theorem power_identity (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : 
  a^(3*m + 2*n) = 72 := by
sorry

end NUMINAMATH_CALUDE_power_identity_l3619_361901


namespace NUMINAMATH_CALUDE_expression_value_l3619_361997

theorem expression_value (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |x| = 3) : 
  10 * a + 10 * b + c * d * x = 3 ∨ 10 * a + 10 * b + c * d * x = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3619_361997


namespace NUMINAMATH_CALUDE_triangle_existence_l3619_361976

/-- A triangle with semiperimeter s and two excircle radii r_a and r_b exists if and only if s^2 > r_a * r_b -/
theorem triangle_existence (s r_a r_b : ℝ) (h_s : s > 0) (h_ra : r_a > 0) (h_rb : r_b > 0) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 * s ∧
    ∃ (r_ea r_eb : ℝ), r_ea = r_a ∧ r_eb = r_b ∧
    r_ea = s * (b + c - a) / (b + c) ∧
    r_eb = s * (a + c - b) / (a + c)) ↔
  s^2 > r_a * r_b :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l3619_361976


namespace NUMINAMATH_CALUDE_median_BD_correct_altitude_CE_correct_l3619_361951

/-- Triangle with vertices A(2,3), B(-1,0), and C(5,-1) -/
structure Triangle where
  A : ℝ × ℝ := (2, 3)
  B : ℝ × ℝ := (-1, 0)
  C : ℝ × ℝ := (5, -1)

/-- Line equation in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Median BD of the triangle -/
def median_BD (t : Triangle) : LineEquation :=
  { a := 2, b := -9, c := 2 }

/-- Altitude CE of the triangle -/
def altitude_CE (t : Triangle) : LineEquation :=
  { a := 1, b := 1, c := -4 }

/-- A point (x, y) lies on a line if it satisfies the line equation -/
def point_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem median_BD_correct (t : Triangle) : 
  point_on_line t.B (median_BD t) ∧ 
  point_on_line ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2) (median_BD t) :=
sorry

theorem altitude_CE_correct (t : Triangle) : 
  point_on_line t.C (altitude_CE t) ∧ 
  (t.A.2 - t.B.2) * (t.C.1 - t.A.1) = (t.A.1 - t.B.1) * (t.C.2 - t.A.2) :=
sorry

end NUMINAMATH_CALUDE_median_BD_correct_altitude_CE_correct_l3619_361951


namespace NUMINAMATH_CALUDE_common_solution_of_linear_system_l3619_361958

theorem common_solution_of_linear_system :
  (∀ (a b : ℚ), ∃ (x y : ℚ), (a - b) * x - (a + b) * y = a + b) →
  (∃! (x y : ℚ), ∀ (a b : ℚ), (a - b) * x - (a + b) * y = a + b ∧ x = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_of_linear_system_l3619_361958


namespace NUMINAMATH_CALUDE_eighteen_picks_required_l3619_361902

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : Nat
  black : Nat
  yellow : Nat

/-- The minimum number of picks required to guarantee at least one ball of each color -/
def minPicksRequired (counts : BallCounts) : Nat :=
  counts.white + counts.black + 1

/-- Theorem stating that for the given ball counts, 18 picks are required -/
theorem eighteen_picks_required (counts : BallCounts) 
  (h_white : counts.white = 8)
  (h_black : counts.black = 9)
  (h_yellow : counts.yellow = 7) : 
  minPicksRequired counts = 18 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_picks_required_l3619_361902


namespace NUMINAMATH_CALUDE_disprove_square_implies_greater_l3619_361908

theorem disprove_square_implies_greater : ∃ a b : ℝ, a^2 > b^2 ∧ a ≤ b :=
  let a := -3
  let b := 2
  have h1 : a^2 > b^2 := by sorry
  have h2 : a ≤ b := by sorry
  ⟨a, b, h1, h2⟩

#check disprove_square_implies_greater

end NUMINAMATH_CALUDE_disprove_square_implies_greater_l3619_361908


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_seating_satisfies_spacing_l3619_361967

/-- The number of ways to seat people on chairs with spacing requirements -/
def seating_arrangements (people chairs : ℕ) : ℕ :=
  if people = 3 ∧ chairs = 8 then 36 else 0

/-- Theorem stating the correct number of seating arrangements -/
theorem correct_seating_arrangements :
  seating_arrangements 3 8 = 36 := by
  sorry

/-- Theorem proving the seating arrangement satisfies the spacing requirement -/
theorem seating_satisfies_spacing (arrangement : Fin 8 → Option (Fin 3)) :
  seating_arrangements 3 8 = 36 →
  (∀ i j : Fin 3, i ≠ j →
    ∀ s t : Fin 8, arrangement s = some i ∧ arrangement t = some j →
      (s : ℕ) + 1 < (t : ℕ) ∨ (t : ℕ) + 1 < (s : ℕ)) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_seating_satisfies_spacing_l3619_361967


namespace NUMINAMATH_CALUDE_quadratic_roots_negative_real_part_l3619_361953

theorem quadratic_roots_negative_real_part (p q : ℝ) :
  (∃ x : ℂ, p * x^2 + (p^2 - q) * x - (2*p - q - 1) = 0 ∧ x.re < 0) ↔
  (p = 0 ∧ -1 < q ∧ q < 0) ∨ (p > 0 ∧ q < p^2 ∧ q > 2*p - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_negative_real_part_l3619_361953


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3619_361990

theorem triangle_third_side_length (a b : ℝ) (cos_theta : ℝ) : 
  a = 5 → b = 3 → 
  (5 * cos_theta^2 - 7 * cos_theta - 6 = 0) →
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * cos_theta ∧ c = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3619_361990


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l3619_361948

def total_balls : ℕ := 11
def white_balls : ℕ := 4
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

theorem probability_three_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 4 / 165 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l3619_361948


namespace NUMINAMATH_CALUDE_sum_lower_bound_l3619_361992

theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l3619_361992


namespace NUMINAMATH_CALUDE_school_population_l3619_361966

/-- Represents the total number of students in the school -/
def total_students : ℕ := 50

/-- Represents the number of students of 8 years of age -/
def students_8_years : ℕ := 24

/-- Represents the fraction of students below 8 years of age -/
def fraction_below_8 : ℚ := 1/5

/-- Represents the ratio of students above 8 years to students of 8 years -/
def ratio_above_to_8 : ℚ := 2/3

theorem school_population :
  (students_8_years : ℚ) + 
  (ratio_above_to_8 * students_8_years) + 
  (fraction_below_8 * total_students) = total_students := by sorry

end NUMINAMATH_CALUDE_school_population_l3619_361966


namespace NUMINAMATH_CALUDE_product_of_roots_l3619_361979

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  ∃ p q r : ℝ, (x - p)*(x - q)*(x - r) = x^3 - 15*x^2 + 75*x - 50 ∧ p*q*r = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3619_361979


namespace NUMINAMATH_CALUDE_b_2016_equals_zero_l3619_361915

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sequence b_n
def b (n : ℕ) : ℕ := fib n % 4

-- Theorem statement
theorem b_2016_equals_zero : b 2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_2016_equals_zero_l3619_361915


namespace NUMINAMATH_CALUDE_tissues_left_is_1060_l3619_361942

/-- The number of tissues Tucker has left after all actions. -/
def tissues_left : ℕ :=
  let brand_a_per_box := 160
  let brand_b_per_box := 180
  let brand_c_per_box := 200
  let brand_a_boxes := 4
  let brand_b_boxes := 6
  let brand_c_boxes := 2
  let brand_a_used := 250
  let brand_b_used := 410
  let brand_c_used := 150
  let brand_b_given := 2
  let brand_c_received := 110

  let brand_a_left := brand_a_per_box * brand_a_boxes - brand_a_used
  let brand_b_left := brand_b_per_box * brand_b_boxes - brand_b_used - brand_b_per_box * brand_b_given
  let brand_c_left := brand_c_per_box * brand_c_boxes - brand_c_used + brand_c_received

  brand_a_left + brand_b_left + brand_c_left

theorem tissues_left_is_1060 : tissues_left = 1060 := by
  sorry

end NUMINAMATH_CALUDE_tissues_left_is_1060_l3619_361942


namespace NUMINAMATH_CALUDE_jessicas_allowance_l3619_361977

theorem jessicas_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 11) → allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_allowance_l3619_361977


namespace NUMINAMATH_CALUDE_local_extremum_implies_a_equals_four_l3619_361970

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_implies_a_equals_four :
  ∀ a b : ℝ,
  (f a b 1 = 10) →  -- f(1) = 10
  (f' a b 1 = 0) →  -- f'(1) = 0 (condition for local extremum)
  (∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → f a b x ≤ f a b 1) →  -- local maximum condition
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_local_extremum_implies_a_equals_four_l3619_361970


namespace NUMINAMATH_CALUDE_add_1723_minutes_to_midnight_l3619_361947

-- Define a custom datatype for date and time
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

-- Define the starting date and time
def startDateTime : DateTime :=
  { year := 2023, month := 5, day := 5, hour := 0, minute := 0 }

-- Define the number of minutes to add
def minutesToAdd : Nat := 1723

-- Theorem to prove
theorem add_1723_minutes_to_midnight :
  addMinutes startDateTime minutesToAdd =
    { year := 2023, month := 5, day := 6, hour := 4, minute := 43 } :=
  sorry

end NUMINAMATH_CALUDE_add_1723_minutes_to_midnight_l3619_361947


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3619_361971

/-- The surface area of a cylinder with lateral surface net as a rectangle with sides 6π and 4π -/
theorem cylinder_surface_area : 
  ∀ (r h : ℝ), 
  (2 * π * r = 6 * π) → 
  (h = 4 * π) → 
  (2 * π * r * h + 2 * π * r^2 = 24 * π^2 + 18 * π) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3619_361971


namespace NUMINAMATH_CALUDE_books_sum_is_95_l3619_361973

/-- The total number of books Tim, Mike, Sarah, and Emily have together -/
def total_books (tim_books mike_books sarah_books emily_books : ℕ) : ℕ :=
  tim_books + mike_books + sarah_books + emily_books

/-- Theorem stating that the total number of books is 95 -/
theorem books_sum_is_95 :
  total_books 22 20 35 18 = 95 := by
  sorry

end NUMINAMATH_CALUDE_books_sum_is_95_l3619_361973


namespace NUMINAMATH_CALUDE_triangle_theorem_l3619_361941

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem --/
theorem triangle_theorem (t : Triangle) :
  (t.b / (t.a + t.c) = (t.a + t.b - t.c) / (t.a + t.b)) →
  (t.A = π / 3) ∧
  (t.A = π / 3 ∧ t.a = 15 ∧ t.b = 10 → Real.cos t.B = Real.sqrt 6 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3619_361941


namespace NUMINAMATH_CALUDE_gcf_of_60_90_150_l3619_361960

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by sorry

end NUMINAMATH_CALUDE_gcf_of_60_90_150_l3619_361960


namespace NUMINAMATH_CALUDE_min_value_expression_l3619_361922

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1/(a*b) + 1/(a*(a-b)) ≥ 4 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > b₀ ∧ b₀ > 0 ∧ a₀^2 + 1/(a₀*b₀) + 1/(a₀*(a₀-b₀)) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3619_361922


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l3619_361952

/-- Represents the arrangement of numbers in the square with a center circle -/
structure Arrangement :=
  (top_left : ℕ)
  (top_right : ℕ)
  (bottom_left : ℕ)
  (bottom_right : ℕ)
  (center : ℕ)

/-- The set of numbers to be arranged -/
def numbers : Finset ℕ := {2, 4, 6, 8, 10}

/-- Checks if the given arrangement satisfies the diagonal and vertex sum condition -/
def is_valid_arrangement (a : Arrangement) : Prop :=
  a.top_left + a.center + a.bottom_right = 
  a.top_right + a.center + a.bottom_left ∧
  a.top_left + a.center + a.bottom_right = 
  a.top_left + a.top_right + a.bottom_left + a.bottom_right

/-- Checks if the given arrangement uses all the required numbers -/
def uses_all_numbers (a : Arrangement) : Prop :=
  {a.top_left, a.top_right, a.bottom_left, a.bottom_right, a.center} = numbers

/-- Theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : 
  ∃ (a : Arrangement), is_valid_arrangement a ∧ uses_all_numbers a :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l3619_361952


namespace NUMINAMATH_CALUDE_beef_price_calculation_l3619_361919

/-- The price per pound of beef, given the conditions of John's food order --/
def beef_price_per_pound : ℝ := 8

theorem beef_price_calculation (beef_amount : ℝ) (chicken_price : ℝ) (total_cost : ℝ) :
  beef_amount = 1000 →
  chicken_price = 3 →
  total_cost = 14000 →
  beef_price_per_pound * beef_amount + chicken_price * (2 * beef_amount) = total_cost := by
  sorry

#check beef_price_calculation

end NUMINAMATH_CALUDE_beef_price_calculation_l3619_361919


namespace NUMINAMATH_CALUDE_marble_difference_l3619_361934

theorem marble_difference (connie_marbles juan_marbles : ℕ) 
  (h1 : connie_marbles = 323)
  (h2 : juan_marbles = 498)
  (h3 : juan_marbles > connie_marbles) : 
  juan_marbles - connie_marbles = 175 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3619_361934


namespace NUMINAMATH_CALUDE_escalator_walking_speed_l3619_361983

/-- Proves that given an escalator moving at 12 ft/sec with a length of 160 feet,
    if a person covers the entire length in 8 seconds,
    then the person's walking speed on the escalator is 8 ft/sec. -/
theorem escalator_walking_speed
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (time_taken : ℝ)
  (person_speed : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 160)
  (h3 : time_taken = 8)
  (h4 : escalator_length = (person_speed + escalator_speed) * time_taken) :
  person_speed = 8 := by
  sorry

#check escalator_walking_speed

end NUMINAMATH_CALUDE_escalator_walking_speed_l3619_361983


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_l3619_361944

theorem integer_solutions_quadratic (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x y : ℤ, x^2 + p*x + q^4 = 0 ∧ y^2 + p*y + q^4 = 0 ∧ x ≠ y) ↔ 
  (p = 17 ∧ q = 2) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_l3619_361944


namespace NUMINAMATH_CALUDE_largest_square_with_four_interior_lattice_points_l3619_361962

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square in the plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Counts the number of lattice points strictly inside a square -/
def count_interior_lattice_points (s : Square) : ℕ :=
  sorry

/-- The theorem stating the area of the largest square with exactly 4 interior lattice points -/
theorem largest_square_with_four_interior_lattice_points :
  ∃ (s : Square),
    (count_interior_lattice_points s = 4) ∧
    (∀ (t : Square), count_interior_lattice_points t = 4 → t.side_length ≤ s.side_length) ∧
    (9 < s.side_length ^ 2) ∧ (s.side_length ^ 2 < 10) :=
  sorry

end NUMINAMATH_CALUDE_largest_square_with_four_interior_lattice_points_l3619_361962


namespace NUMINAMATH_CALUDE_vector_dot_product_equality_l3619_361921

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (2, 1)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def vector_BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_equality : 
  dot_product vector_AB (2 • vector_AC + vector_BC) = -14 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_equality_l3619_361921


namespace NUMINAMATH_CALUDE_thirty_times_multiple_of_every_integer_l3619_361957

theorem thirty_times_multiple_of_every_integer (n : ℤ) :
  (∀ m : ℤ, ∃ k : ℤ, n = 30 * k * m) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_thirty_times_multiple_of_every_integer_l3619_361957


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l3619_361965

/-- Proves that adding 5 liters of yellow tint to a 30-liter mixture
    with 30% yellow tint results in a new mixture with 40% yellow tint -/
theorem yellow_tint_percentage
  (original_volume : ℝ)
  (original_yellow_percent : ℝ)
  (added_yellow : ℝ)
  (h1 : original_volume = 30)
  (h2 : original_yellow_percent = 30)
  (h3 : added_yellow = 5) :
  let original_yellow := original_volume * (original_yellow_percent / 100)
  let new_yellow := original_yellow + added_yellow
  let new_volume := original_volume + added_yellow
  new_yellow / new_volume * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_yellow_tint_percentage_l3619_361965


namespace NUMINAMATH_CALUDE_jovana_shells_l3619_361954

/-- The amount of shells added to a bucket -/
def shells_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Jovana added 12 pounds of shells to her bucket -/
theorem jovana_shells : shells_added 5 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_l3619_361954


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3619_361949

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 5*x + 6 = 0
def equation2 (x : ℝ) : Prop := (x + 2)*(x - 1) = x + 2

-- Theorem for equation1
theorem solutions_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 3 ∧ x₂ = 2 :=
sorry

-- Theorem for equation2
theorem solutions_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation2 x₁ ∧ equation2 x₂ ∧ x₁ = -2 ∧ x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3619_361949


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3619_361999

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3619_361999


namespace NUMINAMATH_CALUDE_correct_product_l3619_361991

theorem correct_product (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  (∃ x y : ℕ, x * 10 + y = a ∧ y * 10 + x = (189 / b)) →  -- reversing digits of a and multiplying by b gives 189
  a * b = 108 := by
sorry

end NUMINAMATH_CALUDE_correct_product_l3619_361991


namespace NUMINAMATH_CALUDE_johns_father_age_l3619_361906

/-- Given the ages of John, his mother, and his father, we prove that the father's age is 40 years. -/
theorem johns_father_age (john mother father : ℕ) 
  (h1 : father = 2 * john)  -- John is half times younger than his father
  (h2 : father = mother + 4)  -- John's father is 4 years older than John's mother
  (h3 : mother = john + 16)  -- The age difference between John and his mother is 16 years
  : father = 40 := by
  sorry

end NUMINAMATH_CALUDE_johns_father_age_l3619_361906


namespace NUMINAMATH_CALUDE_sin_225_degrees_l3619_361916

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l3619_361916


namespace NUMINAMATH_CALUDE_sequence_general_term_l3619_361925

theorem sequence_general_term (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) = a n / (1 + 3 * a n)) →
  a 1 = 2 →
  ∀ n : ℕ, n ≥ 1 → a n = 2 / (6 * n - 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3619_361925


namespace NUMINAMATH_CALUDE_equation_real_solutions_l3619_361945

theorem equation_real_solutions :
  let f : ℝ → ℝ := λ x => 5*x/(x^2 + 2*x + 4) + 7*x/(x^2 - 7*x + 4) + 5/3
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ x, f x = 0 → x ∈ s :=
by sorry

end NUMINAMATH_CALUDE_equation_real_solutions_l3619_361945


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l3619_361969

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) (slower_speed : ℝ) :
  actual_distance = 24 →
  faster_speed = 5 →
  additional_distance = 6 →
  faster_speed * (actual_distance / slower_speed) = actual_distance + additional_distance →
  slower_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l3619_361969


namespace NUMINAMATH_CALUDE_never_prime_n_squared_plus_75_l3619_361940

theorem never_prime_n_squared_plus_75 (n : ℕ) (h : Prime n) : ¬ Prime (n^2 + 75) := by
  sorry

end NUMINAMATH_CALUDE_never_prime_n_squared_plus_75_l3619_361940


namespace NUMINAMATH_CALUDE_markese_earnings_l3619_361903

/-- Proves that Markese earned 16 dollars given the conditions of the problem -/
theorem markese_earnings (E : ℕ) 
  (h1 : E - 5 + E = 37) : 
  E - 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_markese_earnings_l3619_361903


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3619_361985

/-- The sum of the geometric series with a specific pattern -/
theorem geometric_series_sum : 
  ∑' k : ℕ, (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3619_361985


namespace NUMINAMATH_CALUDE_marina_extra_parks_l3619_361933

/-- The number of theme parks in Jamestown -/
def jamestown_parks : ℕ := 20

/-- The number of additional theme parks Venice has compared to Jamestown -/
def venice_extra_parks : ℕ := 25

/-- The total number of theme parks in all three towns -/
def total_parks : ℕ := 135

/-- The number of theme parks in Venice -/
def venice_parks : ℕ := jamestown_parks + venice_extra_parks

/-- The number of theme parks in Marina Del Ray -/
def marina_parks : ℕ := total_parks - (jamestown_parks + venice_parks)

/-- The difference in theme parks between Marina Del Ray and Jamestown -/
def marina_jamestown_difference : ℕ := marina_parks - jamestown_parks

theorem marina_extra_parks :
  marina_jamestown_difference = 50 := by sorry

end NUMINAMATH_CALUDE_marina_extra_parks_l3619_361933


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3619_361975

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (8, 0), radius := 2 }
  ∀ l : Line,
    (∃ p1 p2 : ℝ × ℝ,
      isTangent l c1 ∧
      isTangent l c2 ∧
      isInFirstQuadrant p1 ∧
      isInFirstQuadrant p2 ∧
      (p1.1 - 3)^2 + p1.2^2 = 9 ∧
      (p2.1 - 8)^2 + p2.2^2 = 4) →
    l.intercept = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3619_361975


namespace NUMINAMATH_CALUDE_height_sum_l3619_361924

/-- Given the heights of John, Lena, and Rebeca, prove that the sum of Lena's and Rebeca's heights is 295 cm. -/
theorem height_sum (john_height lena_height rebeca_height : ℕ) 
  (h1 : john_height = 152)
  (h2 : john_height = lena_height + 15)
  (h3 : rebeca_height = john_height + 6) :
  lena_height + rebeca_height = 295 := by
  sorry

end NUMINAMATH_CALUDE_height_sum_l3619_361924


namespace NUMINAMATH_CALUDE_medicine_parts_for_child_l3619_361910

/-- Calculates the number of equal parts a medicine dose should be divided into -/
def medicine_parts (weight : ℕ) (dosage_per_kg : ℕ) (mg_per_part : ℕ) : ℕ :=
  (weight * dosage_per_kg * 1000) / mg_per_part

/-- Theorem: For a 30 kg child, with 5 ml/kg dosage and 50 mg parts, the dose divides into 3000 parts -/
theorem medicine_parts_for_child : medicine_parts 30 5 50 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_medicine_parts_for_child_l3619_361910


namespace NUMINAMATH_CALUDE_kathleen_store_visits_l3619_361959

/-- The number of bottle caps Kathleen buys each time she goes to the store -/
def bottle_caps_per_visit : ℕ := 5

/-- The total number of bottle caps Kathleen bought last month -/
def total_bottle_caps : ℕ := 25

/-- The number of times Kathleen went to the store last month -/
def store_visits : ℕ := total_bottle_caps / bottle_caps_per_visit

theorem kathleen_store_visits : store_visits = 5 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_store_visits_l3619_361959


namespace NUMINAMATH_CALUDE_inequality_proof_l3619_361964

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ∧ 
  ((x + y) * (x^3 + y^3)) / ((x^2 + y^2)^2) ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3619_361964


namespace NUMINAMATH_CALUDE_six_people_arrangement_l3619_361909

/-- The number of ways to arrange n people in a row -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row, where two specific people must be adjacent in a fixed order -/
def arrangements_with_fixed_pair (n : ℕ) : ℕ := (n - 1).factorial

theorem six_people_arrangement : arrangements_with_fixed_pair 6 = 120 := by sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l3619_361909


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_l3619_361929

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_l3619_361929


namespace NUMINAMATH_CALUDE_andrew_sandwiches_l3619_361935

/-- Given a total number of sandwiches and number of friends, 
    calculate the number of sandwiches per friend -/
def sandwiches_per_friend (total_sandwiches : ℕ) (num_friends : ℕ) : ℕ :=
  total_sandwiches / num_friends

/-- Theorem: Given 12 sandwiches and 4 friends, 
    the number of sandwiches per friend is 3 -/
theorem andrew_sandwiches : 
  sandwiches_per_friend 12 4 = 3 := by
  sorry


end NUMINAMATH_CALUDE_andrew_sandwiches_l3619_361935


namespace NUMINAMATH_CALUDE_water_to_height_ratio_l3619_361932

def rons_height : ℝ := 12
def water_depth : ℝ := 60

theorem water_to_height_ratio : water_depth / rons_height = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_to_height_ratio_l3619_361932


namespace NUMINAMATH_CALUDE_root_implies_difference_l3619_361926

theorem root_implies_difference (a b : ℝ) :
  (∃ x, x^2 + 4*a^2*b^2*x = 4 ∧ x = (a^2 - b^2)^2) →
  (b^4 - a^4 = 2 ∨ b^4 - a^4 = -2) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_difference_l3619_361926


namespace NUMINAMATH_CALUDE_student_height_survey_is_comprehensive_l3619_361927

/-- Represents a survey --/
structure Survey where
  population : ℕ
  measurementType : Type
  isFeasible : Bool

/-- Defines the conditions for a comprehensive survey --/
def isComprehensiveSurvey (s : Survey) : Prop :=
  s.population ≤ 100 ∧ s.isFeasible = true

/-- Represents the survey of students' heights in a class --/
def studentHeightSurvey : Survey :=
  { population := 45,
    measurementType := ℝ,
    isFeasible := true }

/-- Theorem stating that the student height survey is a comprehensive survey --/
theorem student_height_survey_is_comprehensive :
  isComprehensiveSurvey studentHeightSurvey :=
by
  sorry


end NUMINAMATH_CALUDE_student_height_survey_is_comprehensive_l3619_361927


namespace NUMINAMATH_CALUDE_probability_two_hits_l3619_361912

def probability_at_least_one_hit : ℚ := 65/81

def number_of_shots : ℕ := 4

def probability_single_hit : ℚ := 1/3

theorem probability_two_hits :
  (1 - probability_at_least_one_hit) = (1 - probability_single_hit) ^ number_of_shots →
  Nat.choose number_of_shots 2 * probability_single_hit^2 * (1 - probability_single_hit)^2 = 8/27 := by
sorry

end NUMINAMATH_CALUDE_probability_two_hits_l3619_361912


namespace NUMINAMATH_CALUDE_cloth_selling_price_l3619_361950

/-- Calculates the total selling price of cloth given the quantity, profit per metre, and cost price per metre. -/
def total_selling_price (quantity : ℕ) (profit_per_metre : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  quantity * (cost_price_per_metre + profit_per_metre)

/-- Proves that the total selling price of 30 meters of cloth with a profit of Rs. 10 per metre
    and a cost price of Rs. 140 per metre is Rs. 4500. -/
theorem cloth_selling_price :
  total_selling_price 30 10 140 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l3619_361950


namespace NUMINAMATH_CALUDE_negation_equivalence_l3619_361939

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 0 ∧ 2 * x₀ = 3) ↔ (∀ x : ℝ, x ≥ 0 → 2 * x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3619_361939


namespace NUMINAMATH_CALUDE_game_draw_probability_l3619_361993

/-- In a game between two players, given the probabilities of not losing and losing for each player, 
    we can calculate the probability of a draw. -/
theorem game_draw_probability (p_not_losing p_losing : ℚ) : 
  p_not_losing = 3/4 → p_losing = 1/2 → p_not_losing - p_losing = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_game_draw_probability_l3619_361993


namespace NUMINAMATH_CALUDE_specific_prism_properties_l3619_361984

/-- A right prism with a triangular base -/
structure TriangularPrism where
  base_side_a : ℝ
  base_side_b : ℝ
  base_side_c : ℝ
  section_cut_a : ℝ
  section_cut_b : ℝ
  section_cut_c : ℝ

/-- Calculate the volume of the bounded figure -/
def bounded_volume (prism : TriangularPrism) : ℝ :=
  sorry

/-- Calculate the total surface area of the bounded figure -/
def bounded_surface_area (prism : TriangularPrism) : ℝ :=
  sorry

/-- Theorem stating the volume and surface area of the specific prism -/
theorem specific_prism_properties :
  let prism : TriangularPrism := {
    base_side_a := 6,
    base_side_b := 8,
    base_side_c := 10,
    section_cut_a := 12,
    section_cut_b := 12,
    section_cut_c := 18
  }
  bounded_volume prism = 336 ∧ bounded_surface_area prism = 396 :=
by sorry

end NUMINAMATH_CALUDE_specific_prism_properties_l3619_361984


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_value_l3619_361989

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a1_value
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a3 : a 3 = 1)
  (h_mean : (a 5 + (3/2) * a 4) / 2 = 1/2) :
  a 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_value_l3619_361989


namespace NUMINAMATH_CALUDE_inverse_B_cubed_l3619_361955

theorem inverse_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, 7; -2, -5]) : 
  (B^3)⁻¹ = !![13, 0; -42, -95] := by
  sorry

end NUMINAMATH_CALUDE_inverse_B_cubed_l3619_361955


namespace NUMINAMATH_CALUDE_arg_z_range_l3619_361930

theorem arg_z_range (z : ℂ) (h : |Complex.arg ((z + 1) / (z + 2))| = π / 6) :
  Complex.arg z ∈ Set.union
    (Set.Ioo (5 * π / 6 - Real.arcsin (Real.sqrt 3 / 3)) π)
    (Set.Ioo π (7 * π / 6 + Real.arcsin (Real.sqrt 3 / 3))) := by
  sorry

end NUMINAMATH_CALUDE_arg_z_range_l3619_361930


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3619_361963

theorem least_subtraction_for_divisibility (n m : ℕ) (h : n = 45678 ∧ m = 47) :
  ∃ k : ℕ, k ≤ m - 1 ∧ (n - k) % m = 0 ∧ ∀ j : ℕ, j < k → (n - j) % m ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3619_361963


namespace NUMINAMATH_CALUDE_g_comp_four_roots_l3619_361988

/-- The function g(x) defined as x^2 + 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 4 distinct real roots iff d < 4 -/
theorem g_comp_four_roots (d : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g_comp d x₁ = 0 ∧ g_comp d x₂ = 0 ∧ g_comp d x₃ = 0 ∧ g_comp d x₄ = 0 ∧
    ∀ (y : ℝ), g_comp d y = 0 → y = x₁ ∨ y = x₂ ∨ y = x₃ ∨ y = x₄) ↔
  d < 4 :=
sorry

end NUMINAMATH_CALUDE_g_comp_four_roots_l3619_361988


namespace NUMINAMATH_CALUDE_charity_run_donation_l3619_361996

theorem charity_run_donation (total_donation : ℕ) (race_length : ℕ) : 
  race_length = 5 ∧ 
  total_donation = 310 ∧ 
  (∃ initial_donation : ℕ, 
    total_donation = initial_donation * (2^race_length - 1)) →
  ∃ initial_donation : ℕ, initial_donation = 10 ∧
    total_donation = initial_donation * (2^race_length - 1) :=
by sorry

end NUMINAMATH_CALUDE_charity_run_donation_l3619_361996


namespace NUMINAMATH_CALUDE_midnight_temperature_l3619_361923

/-- Calculates the final temperature given initial temperature and temperature changes --/
def finalTemperature (initial : Int) (noonChange : Int) (midnightChange : Int) : Int :=
  initial + noonChange - midnightChange

/-- Theorem stating that the final temperature at midnight is -4°C --/
theorem midnight_temperature :
  finalTemperature (-2) 6 8 = -4 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l3619_361923


namespace NUMINAMATH_CALUDE_figure_to_square_l3619_361987

/-- Represents a figure on a grid --/
structure GridFigure where
  area : ℕ

/-- Represents a cut of the figure --/
inductive Cut
  | Part : Cut

/-- Represents the result of cutting and arranging --/
inductive Arrangement
  | Square : Arrangement

/-- Theorem: If a grid figure's area is a perfect square, 
    it can be cut into three parts and arranged into a square --/
theorem figure_to_square (f : GridFigure) 
  (h : ∃ n : ℕ, f.area = n * n) : 
  ∃ (c1 c2 c3 : Cut) (arr : Arrangement), 
    arr = Arrangement.Square := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l3619_361987


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3619_361994

theorem tan_alpha_value (α : Real) 
  (h : (Real.cos (π / 2 - α) - 3 * Real.cos α) / (Real.sin α - Real.cos (π + α)) = 2) : 
  Real.tan α = -5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3619_361994


namespace NUMINAMATH_CALUDE_percentage_problem_l3619_361998

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 70) : x = 280 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3619_361998


namespace NUMINAMATH_CALUDE_smallest_of_three_successive_integers_l3619_361937

/-- Given three successive integers whose product is 1,059,460, prove that the smallest of these integers is 101. -/
theorem smallest_of_three_successive_integers (n : ℤ) : 
  n * (n + 1) * (n + 2) = 1059460 → n = 101 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_successive_integers_l3619_361937
