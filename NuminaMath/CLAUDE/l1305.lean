import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1305_130596

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (((x - y) * x^4 < 0 → x < y) ∧
   ∃ a b : ℝ, a < b ∧ (a - b) * a^4 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1305_130596


namespace NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l1305_130518

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = (a n : ℚ) * r

theorem second_term_of_geometric_sequence
    (a : ℕ → ℕ)
    (is_geometric : IsGeometricSequence a)
    (first_term : a 1 = 5)
    (fifth_term : a 5 = 320) :
  a 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l1305_130518


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1305_130556

theorem min_value_trig_expression (θ : Real) (h : θ ∈ Set.Ioo 0 (π / 2)) :
  1 / (Real.sin θ)^2 + 4 / (Real.cos θ)^2 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1305_130556


namespace NUMINAMATH_CALUDE_expression_simplification_l1305_130567

theorem expression_simplification (x : ℝ) (h : x + 2 = Real.sqrt 2) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3*x)) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1305_130567


namespace NUMINAMATH_CALUDE_complex_modulus_l1305_130558

theorem complex_modulus (x y : ℝ) (z : ℂ) : 
  z = x + y * Complex.I → 
  (1/2 * x - y : ℂ) + (x + y) * Complex.I = 3 * Complex.I → 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1305_130558


namespace NUMINAMATH_CALUDE_train_ride_nap_time_l1305_130566

theorem train_ride_nap_time (total_time reading_time eating_time movie_time : ℕ) 
  (h1 : total_time = 9)
  (h2 : reading_time = 2)
  (h3 : eating_time = 1)
  (h4 : movie_time = 3) :
  total_time - (reading_time + eating_time + movie_time) = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_ride_nap_time_l1305_130566


namespace NUMINAMATH_CALUDE_count_special_arrangements_l1305_130583

/-- The number of 15-letter arrangements of 5 A's, 5 B's, and 5 C's,
    where no B's are in the first 5 letters, no C's are in the middle 5 letters,
    and no A's are in the last 5 letters. -/
def specialArrangements : ℕ :=
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) ^ 3)

/-- Theorem stating that the number of special arrangements
    is equal to the sum of (5 choose k)^3 for k from 0 to 5. -/
theorem count_special_arrangements :
  specialArrangements = (Finset.range 6).sum (fun k => (Nat.choose 5 k) ^ 3) := by
  sorry


end NUMINAMATH_CALUDE_count_special_arrangements_l1305_130583


namespace NUMINAMATH_CALUDE_printers_finish_time_l1305_130591

-- Define the start time of the first printer
def printer1_start : Real := 9

-- Define the time when half the tasks are completed
def half_tasks_time : Real := 12.5

-- Define the start time of the second printer
def printer2_start : Real := 13

-- Define the time taken by the second printer to complete its set amount
def printer2_duration : Real := 2

-- Theorem to prove
theorem printers_finish_time :
  let printer1_duration := 2 * (half_tasks_time - printer1_start)
  let printer1_finish := printer1_start + printer1_duration
  let printer2_finish := printer2_start + printer2_duration
  max printer1_finish printer2_finish = 16 := by
  sorry

end NUMINAMATH_CALUDE_printers_finish_time_l1305_130591


namespace NUMINAMATH_CALUDE_jack_apples_to_father_l1305_130503

/-- The number of apples Jack bought -/
def total_apples : ℕ := 55

/-- The number of Jack's friends -/
def num_friends : ℕ := 4

/-- The number of apples each person (Jack and his friends) gets -/
def apples_per_person : ℕ := 9

/-- The number of apples Jack wants to give to his father -/
def apples_to_father : ℕ := total_apples - (num_friends + 1) * apples_per_person

theorem jack_apples_to_father :
  apples_to_father = 10 := by sorry

end NUMINAMATH_CALUDE_jack_apples_to_father_l1305_130503


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1305_130512

/-- Represents a chess tournament -/
structure ChessTournament where
  /-- The number of participants in the tournament -/
  participants : ℕ
  /-- The total number of games played in the tournament -/
  total_games : ℕ
  /-- Each participant plays exactly one game with each other participant -/
  one_game_each : total_games = participants * (participants - 1) / 2

/-- Theorem: A chess tournament with 190 games has 20 participants -/
theorem chess_tournament_participants (t : ChessTournament) 
    (h : t.total_games = 190) : t.participants = 20 := by
  sorry

#check chess_tournament_participants

end NUMINAMATH_CALUDE_chess_tournament_participants_l1305_130512


namespace NUMINAMATH_CALUDE_orange_pear_total_cost_l1305_130573

def orange_pear_banana_problem (O P B : ℝ) : Prop :=
  (P - O = B) ∧
  (P = 90) ∧
  (200 * B + 400 * O = 24000)

theorem orange_pear_total_cost (O P B : ℝ) 
  (h : orange_pear_banana_problem O P B) : O + P = 120 := by
  sorry

end NUMINAMATH_CALUDE_orange_pear_total_cost_l1305_130573


namespace NUMINAMATH_CALUDE_family_composition_l1305_130590

/-- Represents a family with boys and girls -/
structure Family where
  boys : Nat
  girls : Nat

/-- A boy in the family has equal number of brothers and sisters -/
def equal_siblings (f : Family) : Prop :=
  f.boys - 1 = f.girls

/-- A girl in the family has twice as many brothers as sisters -/
def double_brothers (f : Family) : Prop :=
  f.boys = 2 * (f.girls - 1)

/-- The family satisfies both conditions and has 4 boys and 3 girls -/
theorem family_composition :
  ∃ (f : Family), equal_siblings f ∧ double_brothers f ∧ f.boys = 4 ∧ f.girls = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_family_composition_l1305_130590


namespace NUMINAMATH_CALUDE_power_of_two_difference_divisible_by_1987_l1305_130562

theorem power_of_two_difference_divisible_by_1987 :
  ∃ (a b : ℕ), 0 ≤ b ∧ b < a ∧ a ≤ 1987 ∧ (2^a - 2^b) % 1987 = 0 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_difference_divisible_by_1987_l1305_130562


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1305_130526

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (7/18, -1/6)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := y + 4 = 15 * x - 2

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1305_130526


namespace NUMINAMATH_CALUDE_largest_gcd_value_l1305_130550

theorem largest_gcd_value (n : ℕ) : 
  ∃ (m : ℕ), (∀ k : ℕ, Nat.gcd (k^2 + 3) ((k + 1)^2 + 3) ≤ m) ∧ 
             (Nat.gcd (n^2 + 3) ((n + 1)^2 + 3) = m) ∧
             m = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_gcd_value_l1305_130550


namespace NUMINAMATH_CALUDE_trapezoid_area_l1305_130506

/-- A trapezoid bounded by y = 2x, y = 12, y = 8, and the y-axis -/
structure Trapezoid where
  /-- The line y = 2x -/
  line : ℝ → ℝ
  /-- The upper bound y = 12 -/
  upper_bound : ℝ
  /-- The lower bound y = 8 -/
  lower_bound : ℝ
  /-- The line is y = 2x -/
  line_eq : ∀ x, line x = 2 * x
  /-- The upper bound is 12 -/
  upper_eq : upper_bound = 12
  /-- The lower bound is 8 -/
  lower_eq : lower_bound = 8

/-- The area of the trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem: The area of the specified trapezoid is 20 square units -/
theorem trapezoid_area : ∀ t : Trapezoid, area t = 20 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1305_130506


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1305_130531

/-- The eccentricity of a hyperbola whose focus coincides with the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | x^2 = -4 * Real.sqrt 5 * y}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a + y^2 / 4 = 1}
  let parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
  ∃ (c : ℝ), c > 0 ∧ (c, 0) ∈ hyperbola ∧ (-c, 0) ∈ hyperbola ∧ 
    (parabola_focus ∈ hyperbola → (Real.sqrt 5) / 2 = c / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1305_130531


namespace NUMINAMATH_CALUDE_largest_number_l1305_130578

theorem largest_number (a b c d : ℝ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = 2) :
  d ≥ a ∧ d ≥ b ∧ d ≥ c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1305_130578


namespace NUMINAMATH_CALUDE_currency_denomination_proof_l1305_130580

theorem currency_denomination_proof :
  let press_F_rate : ℚ := 1000 / 60  -- bills per second
  let press_F_value : ℚ := 5 * press_F_rate  -- dollars per second
  let press_T_rate : ℚ := 200 / 60  -- bills per second
  let time : ℚ := 3  -- seconds
  let extra_value : ℚ := 50  -- dollars
  ∃ x : ℚ, 
    (time * press_F_value = time * (x * press_T_rate) + extra_value) ∧ 
    x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_currency_denomination_proof_l1305_130580


namespace NUMINAMATH_CALUDE_cylinder_volume_l1305_130586

/-- The volume of a cylinder whose lateral surface unfolds into a square with side length 4 -/
theorem cylinder_volume (h : ℝ) (r : ℝ) : 
  h = 4 → 2 * Real.pi * r = 4 → Real.pi * r^2 * h = 16 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1305_130586


namespace NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l1305_130563

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2}

-- Define set A
def A : Set Int := {x ∈ U | x^2 + x - 2 = 0}

-- Define set B
def B : Set Int := {0, -2}

-- Theorem statement
theorem intersection_of_B_and_complement_of_A :
  B ∩ (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l1305_130563


namespace NUMINAMATH_CALUDE_arithmetic_problem_l1305_130579

theorem arithmetic_problem : (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l1305_130579


namespace NUMINAMATH_CALUDE_angle_is_2pi_3_l1305_130557

open Real

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_2pi_3 (a b : ℝ × ℝ) :
  b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 3 →
  a.1^2 + a.2^2 = 1 →
  b.1^2 + b.2^2 = 4 →
  angle_between_vectors a b = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_is_2pi_3_l1305_130557


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1305_130559

def arithmetic_sequence (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n+1 => arithmetic_sequence a d n + d

theorem ratio_a_to_b (a d : ℝ) :
  let b := a + 3 * d
  (arithmetic_sequence a d 0 = a) ∧
  (arithmetic_sequence a d 1 = a + 2*d) ∧
  (arithmetic_sequence a d 2 = a + 3*d) ∧
  (arithmetic_sequence a d 3 = a + 5*d) →
  a / b = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1305_130559


namespace NUMINAMATH_CALUDE_money_split_l1305_130528

theorem money_split (total : ℝ) (ratio_small : ℕ) (ratio_large : ℕ) (smaller_share : ℝ) :
  total = 125 →
  ratio_small = 2 →
  ratio_large = 3 →
  smaller_share = (ratio_small : ℝ) / ((ratio_small : ℝ) + (ratio_large : ℝ)) * total →
  smaller_share = 50 := by
sorry

end NUMINAMATH_CALUDE_money_split_l1305_130528


namespace NUMINAMATH_CALUDE_test_score_calculation_l1305_130530

theorem test_score_calculation (total_questions : Nat) (score : Int) 
  (h1 : total_questions = 100)
  (h2 : score = 61) :
  ∃ (correct : Nat),
    correct ≤ total_questions ∧ 
    (correct : Int) - 2 * (total_questions - correct) = score ∧ 
    correct = 87 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l1305_130530


namespace NUMINAMATH_CALUDE_inequalities_problem_l1305_130500

theorem inequalities_problem (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  (a + b < a * b) ∧
  (b/a + a/b > 2) ∧
  (a > b) ∧
  (abs a < abs b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_problem_l1305_130500


namespace NUMINAMATH_CALUDE_train_speed_l1305_130582

/-- Given a train that crosses a pole in a certain time, calculate its speed in km/hr -/
theorem train_speed (length : Real) (time : Real) (speed_km_hr : Real) : 
  length = 150 ∧ 
  time = 6 ∧ 
  speed_km_hr = (length / time) * 3.6 → 
  speed_km_hr = 90 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l1305_130582


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_of_repeated_digits_l1305_130505

/-- The integer consisting of n repetitions of digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_of_product_of_repeated_digits :
  let a := repeatedDigit 6 1000
  let b := repeatedDigit 7 1000
  sumOfDigits (9 * a * b) = 19986 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_of_repeated_digits_l1305_130505


namespace NUMINAMATH_CALUDE_loss_recording_l1305_130598

/-- Records a financial transaction as a number, where profits are positive and losses are negative. -/
def recordTransaction (amount : ℤ) : ℤ := amount

/-- Given that a profit of 100 yuan is recorded as +100, prove that a loss of 50 yuan is recorded as -50. -/
theorem loss_recording (h : recordTransaction 100 = 100) : recordTransaction (-50) = -50 := by
  sorry

end NUMINAMATH_CALUDE_loss_recording_l1305_130598


namespace NUMINAMATH_CALUDE_correct_product_l1305_130542

theorem correct_product (a b : ℚ) (a_int b_int : ℕ) (result : ℕ) : 
  a = 0.125 →
  b = 5.12 →
  a_int = 125 →
  b_int = 512 →
  result = 64000 →
  a_int * b_int = result →
  a * b = 0.64 := by
sorry

end NUMINAMATH_CALUDE_correct_product_l1305_130542


namespace NUMINAMATH_CALUDE_eggs_per_tray_calculation_l1305_130571

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 45

/-- The number of trays bought weekly -/
def trays_per_week : ℕ := 2

/-- The number of children -/
def num_children : ℕ := 2

/-- The number of eggs eaten by each child daily -/
def child_eggs_per_day : ℕ := 2

/-- The number of adults -/
def num_adults : ℕ := 2

/-- The number of eggs eaten by each adult daily -/
def adult_eggs_per_day : ℕ := 4

/-- The number of eggs left uneaten weekly -/
def uneaten_eggs_per_week : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem eggs_per_tray_calculation :
  eggs_per_tray * trays_per_week = 
    num_children * child_eggs_per_day * days_per_week +
    num_adults * adult_eggs_per_day * days_per_week +
    uneaten_eggs_per_week :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_tray_calculation_l1305_130571


namespace NUMINAMATH_CALUDE_no_solutions_to_sqrt_equation_l1305_130533

theorem no_solutions_to_sqrt_equation :
  ∀ x : ℝ, x ≥ 4 →
  ¬∃ y : ℝ, y = Real.sqrt (x + 5 - 6 * Real.sqrt (x - 4)) + Real.sqrt (x + 18 - 8 * Real.sqrt (x - 4)) ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_to_sqrt_equation_l1305_130533


namespace NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l1305_130593

/-- The number of wheels on each car -/
def wheels_per_car : ℕ := 4

/-- The number of cars brought by guests -/
def guest_cars : ℕ := 10

/-- The number of cars belonging to Dylan's parents -/
def parent_cars : ℕ := 2

/-- The total number of cars in the parking lot -/
def total_cars : ℕ := guest_cars + parent_cars

/-- Theorem stating the total number of car wheels in the parking lot -/
theorem total_wheels_in_parking_lot : 
  (total_cars * wheels_per_car) = 48 := by
sorry

end NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l1305_130593


namespace NUMINAMATH_CALUDE_halfway_fraction_l1305_130565

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) :
  (a + b) / 2 = 19/24 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1305_130565


namespace NUMINAMATH_CALUDE_saree_price_after_discounts_l1305_130517

def original_price : ℝ := 1000

def discount1 : ℝ := 0.30
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10
def discount4 : ℝ := 0.05

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ :=
  apply_discount (apply_discount (apply_discount (apply_discount original_price discount1) discount2) discount3) discount4

theorem saree_price_after_discounts :
  ⌊final_price⌋ = 509 := by sorry

end NUMINAMATH_CALUDE_saree_price_after_discounts_l1305_130517


namespace NUMINAMATH_CALUDE_correct_financial_equation_l1305_130502

/-- Represents Howard's financial transactions -/
def howards_finances (W D X Y : ℝ) : Prop :=
  let initial_money : ℝ := 26
  let final_money : ℝ := 52
  let window_washing_income : ℝ := W
  let dog_walking_income : ℝ := D
  let window_supplies_expense : ℝ := X
  let dog_treats_expense : ℝ := Y
  initial_money + window_washing_income + dog_walking_income - window_supplies_expense - dog_treats_expense = final_money

theorem correct_financial_equation (W D X Y : ℝ) :
  howards_finances W D X Y ↔ 26 + W + D - X - Y = 52 := by sorry

end NUMINAMATH_CALUDE_correct_financial_equation_l1305_130502


namespace NUMINAMATH_CALUDE_min_value_of_ab_l1305_130532

theorem min_value_of_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : 
  2 ≤ a * b ∧ ∃ (x y : ℝ), (1 / x) + (1 / y) = Real.sqrt (x * y) ∧ x * y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l1305_130532


namespace NUMINAMATH_CALUDE_max_lateral_area_inscribed_prism_l1305_130514

noncomputable section

-- Define the sphere's surface area
def sphere_surface_area : ℝ := 8 * Real.pi

-- Define the sphere's radius
def sphere_radius : ℝ := Real.sqrt (sphere_surface_area / (4 * Real.pi))

-- Define the base edge length of the prism
def base_edge_length : ℝ := Real.sqrt 3

-- Define the lateral area of the prism as a function of the base edge length
def lateral_area (x : ℝ) : ℝ := 
  6 * Real.sqrt (-(1/3) * (x^2 - 3)^2 + 3)

-- Theorem statement
theorem max_lateral_area_inscribed_prism :
  (lateral_area base_edge_length = 6 * Real.sqrt 3) ∧
  (∀ x : ℝ, 0 < x → x < Real.sqrt 6 → lateral_area x ≤ lateral_area base_edge_length) := by
  sorry

end

end NUMINAMATH_CALUDE_max_lateral_area_inscribed_prism_l1305_130514


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1305_130574

def U : Set Nat := {2, 4, 5, 7, 8}
def A : Set Nat := {4, 8}

theorem complement_of_A_in_U :
  (U \ A) = {2, 5, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1305_130574


namespace NUMINAMATH_CALUDE_xiao_hua_seat_l1305_130535

structure Classroom where
  rows : Nat
  columns : Nat

structure Seat where
  row : Nat
  column : Nat

def is_valid_seat (c : Classroom) (s : Seat) : Prop :=
  s.row ≤ c.rows ∧ s.column ≤ c.columns

theorem xiao_hua_seat (c : Classroom) (s : Seat) :
  c.rows = 7 →
  c.columns = 8 →
  is_valid_seat c s →
  s.row = 5 →
  s.column = 2 →
  s = ⟨5, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_xiao_hua_seat_l1305_130535


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l1305_130587

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the line x = -1
def line (x : ℝ) : Prop := x = -1

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define when a circle is tangent to a line
def is_tangent_to_line (c : Circle) (l : ℝ → Prop) : Prop :=
  abs (c.center.1 - (-1)) = c.radius

-- Define when a point is on a circle
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- The main theorem
theorem circle_passes_through_fixed_point :
  ∀ c : Circle,
    parabola c.center →
    is_tangent_to_line c line →
    point_on_circle (1, 0) c :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l1305_130587


namespace NUMINAMATH_CALUDE_classroom_desks_proof_l1305_130594

/-- The number of rows in the classroom -/
def num_rows : ℕ := 8

/-- The number of desks in the first row -/
def first_row_desks : ℕ := 10

/-- The maximum number of students that can be seated -/
def max_students : ℕ := 136

/-- The number of additional desks in each subsequent row -/
def additional_desks : ℕ := 2

/-- Calculates the total number of desks in the classroom -/
def total_desks (n : ℕ) : ℕ :=
  num_rows * first_row_desks + (num_rows - 1) * num_rows * n / 2

theorem classroom_desks_proof :
  total_desks additional_desks = max_students :=
sorry

end NUMINAMATH_CALUDE_classroom_desks_proof_l1305_130594


namespace NUMINAMATH_CALUDE_trees_along_road_l1305_130546

theorem trees_along_road (road_length : ℕ) (tree_spacing : ℕ) (h1 : road_length = 1000) (h2 : tree_spacing = 5) :
  road_length / tree_spacing + 1 = 201 := by
  sorry

end NUMINAMATH_CALUDE_trees_along_road_l1305_130546


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1305_130527

theorem cubic_equation_solutions :
  {x : ℝ | (18 * x - 2) ^ (1/3) + (16 * x + 2) ^ (1/3) = 5 * x ^ (1/3)} =
  {0, (-2 + Real.sqrt 1048) / 261, (-2 - Real.sqrt 1048) / 261} :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1305_130527


namespace NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l1305_130545

theorem smallest_value_of_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_omega_power : ω^4 = 1)
  (h_omega_not_one : ω ≠ 1) :
  ∃ (x y z w : ℤ), 
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    ∀ (p q r s : ℤ), p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
    Complex.abs (x + y*ω + z*ω^2 + w*ω^3) ≤ Complex.abs (p + q*ω + r*ω^2 + s*ω^3) ∧
    Complex.abs (x + y*ω + z*ω^2 + w*ω^3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_complex_sum_l1305_130545


namespace NUMINAMATH_CALUDE_fraction_inequality_l1305_130504

theorem fraction_inequality (x : ℝ) : (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1305_130504


namespace NUMINAMATH_CALUDE_sector_central_angle_l1305_130520

theorem sector_central_angle (r : ℝ) (α : ℝ) : 
  r > 0 → 
  r * α = 6 → 
  (1/2) * r * r * α = 6 → 
  α = 3 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1305_130520


namespace NUMINAMATH_CALUDE_total_metal_needed_l1305_130551

/-- Given that Charlie has 276 lbs of metal in storage and needs to buy an additional 359 lbs,
    prove that the total amount of metal he needs for the wings is 635 lbs. -/
theorem total_metal_needed (storage : ℕ) (additional : ℕ) (total : ℕ) 
    (h1 : storage = 276)
    (h2 : additional = 359)
    (h3 : total = storage + additional) : 
  total = 635 := by
  sorry

end NUMINAMATH_CALUDE_total_metal_needed_l1305_130551


namespace NUMINAMATH_CALUDE_quadratic_polynomial_symmetry_l1305_130595

theorem quadratic_polynomial_symmetry (P : ℝ → ℝ) (h : ∃ a b c : ℝ, P x = a * x^2 + b * x + c) :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    P (b + c) = P a ∧ P (c + a) = P b ∧ P (a + b) = P c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_symmetry_l1305_130595


namespace NUMINAMATH_CALUDE_message_spread_time_l1305_130561

/-- The number of people who have received the message after n minutes -/
def people_reached (n : ℕ) : ℕ := 2^(n + 1) - 1

/-- The time required for the message to reach 2047 people -/
def time_to_reach_2047 : ℕ := 10

theorem message_spread_time :
  people_reached time_to_reach_2047 = 2047 :=
sorry

end NUMINAMATH_CALUDE_message_spread_time_l1305_130561


namespace NUMINAMATH_CALUDE_allan_brought_six_balloons_l1305_130521

/-- The number of balloons Jake initially brought to the park -/
def jake_initial_balloons : ℕ := 2

/-- The number of balloons Jake bought at the park -/
def jake_bought_balloons : ℕ := 3

/-- The difference between Allan's and Jake's balloon count -/
def allan_jake_difference : ℕ := 1

/-- The total number of balloons Jake had in the park -/
def jake_total_balloons : ℕ := jake_initial_balloons + jake_bought_balloons

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := jake_total_balloons + allan_jake_difference

theorem allan_brought_six_balloons : allan_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_allan_brought_six_balloons_l1305_130521


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l1305_130564

theorem simplify_nested_roots : 
  (((1 / 65536)^(1/2))^(1/3))^(1/4) = 1 / (2^(2/3)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l1305_130564


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l1305_130511

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 3*x - 18 ≤ 0}
def B : Set ℝ := {x : ℝ | 1 / (x + 1) ≤ -1}

-- Define the complement of B
def complement_B : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem for part (1)
theorem complement_B_intersect_A :
  (complement_B ∩ A) = {x : ℝ | (-6 ≤ x ∧ x < -2) ∨ (-1 ≤ x ∧ x ≤ 3)} :=
sorry

-- Define set C
def C (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 1}

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (B ∪ C a = B) ↔ (a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_range_of_a_l1305_130511


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1305_130534

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1305_130534


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1305_130544

theorem line_passes_through_point :
  ∀ (t : ℝ), (t + 1) * (-4) - (2 * t + 5) * (-2) - 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1305_130544


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l1305_130523

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 9) 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l1305_130523


namespace NUMINAMATH_CALUDE_symmetric_complex_numbers_l1305_130599

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- If z₁ and z₂ are complex numbers symmetric with respect to the imaginary axis,
    and z₁ = 2 + i, then z₂ = -2 + i. -/
theorem symmetric_complex_numbers (z₁ z₂ : ℂ) 
    (h_sym : symmetric_wrt_imaginary_axis z₁ z₂) 
    (h_z₁ : z₁ = 2 + I) : 
    z₂ = -2 + I := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_numbers_l1305_130599


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1305_130568

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| - 2

theorem solution_set_part1 :
  {x : ℝ | f x + |2*x - 3| > 0} = {x : ℝ | x > 2 ∨ x < 2/3} := by sorry

-- Part 2
def g (a x : ℝ) : ℝ := |x - a| - 2

theorem range_of_a_part2 (a : ℝ) :
  (∃ x, g a x > |x - 3|) → a < 1 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1305_130568


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1305_130576

theorem fraction_sum_equality : 
  (4 : ℚ) / 3 + 13 / 9 + 40 / 27 + 121 / 81 - 8 / 3 = 171 / 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1305_130576


namespace NUMINAMATH_CALUDE_sqrt_factor_inside_l1305_130541

theorem sqrt_factor_inside (x : ℝ) (h : x > 0) :
  -2 * Real.sqrt (5/2) = -Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_factor_inside_l1305_130541


namespace NUMINAMATH_CALUDE_work_ratio_proof_l1305_130584

/-- Represents the work rate of a single cat -/
def single_cat_rate : ℝ := 1

/-- Represents the total work to be done -/
def total_work : ℝ := 10

/-- Represents the number of days the initial cats work -/
def initial_days : ℕ := 5

/-- Represents the total number of days to complete the work -/
def total_days : ℕ := 7

/-- Represents the initial number of cats -/
def initial_cats : ℕ := 2

/-- Represents the final number of cats -/
def final_cats : ℕ := 5

theorem work_ratio_proof :
  let initial_work := (initial_cats : ℝ) * single_cat_rate * initial_days
  let remaining_days := total_days - initial_days
  let remaining_work := (final_cats : ℝ) * single_cat_rate * remaining_days
  initial_work / (initial_work + remaining_work) = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_work_ratio_proof_l1305_130584


namespace NUMINAMATH_CALUDE_find_number_l1305_130548

theorem find_number (N : ℝ) (x : ℝ) (h1 : (N / 4) * x - 18 = 3 * x + 27) (h2 : x = 12) : N = 27 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1305_130548


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1305_130538

/-- A parabola passing through a specific point -/
def parabola_through_point (a : ℝ) (x y : ℝ) : Prop :=
  a ≠ 0 ∧ y = a * x^2

/-- Theorem: The parabola y = ax^2 passing through (2, -8) has a = -2 -/
theorem parabola_coefficient :
  ∀ a : ℝ, parabola_through_point a 2 (-8) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1305_130538


namespace NUMINAMATH_CALUDE_only_A_has_zero_constant_term_l1305_130524

def equation_A (x : ℝ) : ℝ := x^2 + x
def equation_B (x : ℝ) : ℝ := 2*x^2 - x - 12
def equation_C (x : ℝ) : ℝ := 2*(x^2 - 1) - 3*(x - 1)
def equation_D (x : ℝ) : ℝ := 2*(x^2 + 1) - (x + 4)

def has_zero_constant_term (f : ℝ → ℝ) : Prop := f 0 = 0

theorem only_A_has_zero_constant_term :
  has_zero_constant_term equation_A ∧
  ¬has_zero_constant_term equation_B ∧
  ¬has_zero_constant_term equation_C ∧
  ¬has_zero_constant_term equation_D :=
by sorry

end NUMINAMATH_CALUDE_only_A_has_zero_constant_term_l1305_130524


namespace NUMINAMATH_CALUDE_sequence_problem_l1305_130555

def sequence_rule (x y z : ℕ) : Prop := z = 2 * (x + y)

theorem sequence_problem : 
  ∀ (a b c : ℕ), 
  sequence_rule 10 a 30 → 
  sequence_rule a 30 b → 
  sequence_rule 30 b c → 
  c = 200 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1305_130555


namespace NUMINAMATH_CALUDE_draw_three_cards_not_same_color_l1305_130536

/-- Given a set of 16 cards with 4 of each color (red, yellow, blue, green),
    this theorem states that the number of ways to draw 3 cards such that
    they are not all the same color is equal to C(16,3) - 4 * C(4,3). -/
theorem draw_three_cards_not_same_color (total_cards : ℕ) (cards_per_color : ℕ) 
  (num_colors : ℕ) (draw : ℕ) (h1 : total_cards = 16) (h2 : cards_per_color = 4) 
  (h3 : num_colors = 4) (h4 : draw = 3) :
  (Nat.choose total_cards draw) - (num_colors * Nat.choose cards_per_color draw) = 544 := by
  sorry

end NUMINAMATH_CALUDE_draw_three_cards_not_same_color_l1305_130536


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l1305_130507

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l1305_130507


namespace NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_l1305_130539

theorem inscribed_rectangle_perimeter (circle_area : ℝ) (rect_area : ℝ) :
  circle_area = 32 * Real.pi →
  rect_area = 34 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    a * b = rect_area ∧
    a^2 + b^2 = 2 * circle_area / Real.pi ∧
    2 * (a + b) = 28 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_l1305_130539


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l1305_130552

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l1305_130552


namespace NUMINAMATH_CALUDE_point_A_in_transformed_plane_l1305_130508

/-- The similarity transformation coefficient -/
def k : ℚ := 1/2

/-- The original plane equation: 4x - 3y + 5z - 10 = 0 -/
def plane_a (x y z : ℚ) : Prop := 4*x - 3*y + 5*z - 10 = 0

/-- The transformed plane equation: 4x - 3y + 5z - 5 = 0 -/
def plane_a' (x y z : ℚ) : Prop := 4*x - 3*y + 5*z - 5 = 0

/-- Point A -/
def point_A : ℚ × ℚ × ℚ := (1/4, 1/3, 1)

/-- Theorem: Point A belongs to the image of plane a under the similarity transformation -/
theorem point_A_in_transformed_plane :
  plane_a' point_A.1 point_A.2.1 point_A.2.2 :=
by sorry

end NUMINAMATH_CALUDE_point_A_in_transformed_plane_l1305_130508


namespace NUMINAMATH_CALUDE_zero_in_interval_l1305_130553

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 4 ∧ f c = 0 := by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1305_130553


namespace NUMINAMATH_CALUDE_binary_octal_equivalence_l1305_130560

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 8^i) 0

/-- The binary number 1001101₂ is equal to the octal number 115₈ -/
theorem binary_octal_equivalence : 
  binary_to_decimal [1, 0, 1, 1, 0, 0, 1] = octal_to_decimal [5, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_octal_equivalence_l1305_130560


namespace NUMINAMATH_CALUDE_sequence_properties_l1305_130509

def S (n : ℕ+) : ℤ := 3 * n - 2 * n^2

def a (n : ℕ+) : ℤ := -4 * n + 5

theorem sequence_properties :
  ∀ n : ℕ+,
  (∀ k : ℕ+, k ≤ n → S k - S (k-1) = a k) ∧
  S n ≥ n * a n :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1305_130509


namespace NUMINAMATH_CALUDE_percentage_of_sum_l1305_130570

theorem percentage_of_sum (x y : ℝ) (P : ℝ) : 
  (0.5 * (x - y) = (P / 100) * (x + y)) → 
  (y = (11.11111111111111 / 100) * x) → 
  P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l1305_130570


namespace NUMINAMATH_CALUDE_certain_number_proof_l1305_130515

theorem certain_number_proof (X : ℝ) : 
  X / 3 = (169.4915254237288 / 100) * 236 → X = 1200 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1305_130515


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1305_130569

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_at_origin :
  let p : ℝ × ℝ := (0, 0)  -- The origin point
  let m : ℝ := f' p.1      -- The slope of the tangent line at the origin
  ∀ x y : ℝ, y = m * (x - p.1) + f p.1 → y = 0 :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_at_origin_l1305_130569


namespace NUMINAMATH_CALUDE_fraction_equality_l1305_130525

theorem fraction_equality : (5 * 7 + 3) / (3 * 5) = 38 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1305_130525


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1305_130547

theorem stratified_sample_size (n : ℕ) : 
  -- Define the ratio of products
  let ratio_A : ℕ := 2
  let ratio_B : ℕ := 3
  let ratio_C : ℕ := 5
  -- Define the total ratio
  let total_ratio : ℕ := ratio_A + ratio_B + ratio_C
  -- Define the number of items of product A in the sample
  let sample_A : ℕ := 16
  -- The stratified sampling condition
  (ratio_A : ℚ) / total_ratio = sample_A / n →
  -- The conclusion
  n = 80 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1305_130547


namespace NUMINAMATH_CALUDE_soccer_league_games_l1305_130588

/-- The number of games played in a soccer league -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 14 teams, where each team plays every other team once,
    the total number of games played is 91. -/
theorem soccer_league_games :
  num_games 14 = 91 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1305_130588


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1305_130519

/-- Given a curve y = ax + e^x - 1 and its tangent line y = 3x at (0,0), prove a = 2 -/
theorem tangent_line_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x + Real.exp x - 1) →  -- Curve equation
  (∃ m : ℝ, m = 3 ∧ ∀ x y : ℝ, y = m * x) →  -- Tangent line equation
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |((a * x + Real.exp x - 1) - 0) / (x - 0) - 3| < ε) →  -- Tangent condition
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1305_130519


namespace NUMINAMATH_CALUDE_stamps_ratio_after_gift_l1305_130529

/-- Proves that given the initial conditions, the new ratio of Kaye's stamps to Alberto's stamps is 4:3 -/
theorem stamps_ratio_after_gift (x : ℕ) 
  (h1 : 5 * x - 12 = 3 * x + 12 + 32) : 
  (5 * x - 12) / (3 * x + 12) = 4 / 3 := by
  sorry

#check stamps_ratio_after_gift

end NUMINAMATH_CALUDE_stamps_ratio_after_gift_l1305_130529


namespace NUMINAMATH_CALUDE_equation_equivalence_l1305_130540

theorem equation_equivalence (x y : ℝ) : 
  (3 * x^2 + 4 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 5 = 0) →
  4 * y^2 + 33 * y + 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1305_130540


namespace NUMINAMATH_CALUDE_final_basketball_count_l1305_130585

def initial_count : ℕ := 100

def transactions : List ℤ := [38, -42, 27, -33, -40]

theorem final_basketball_count : 
  initial_count + transactions.sum = 50 := by sorry

end NUMINAMATH_CALUDE_final_basketball_count_l1305_130585


namespace NUMINAMATH_CALUDE_no_rational_solution_l1305_130516

theorem no_rational_solution :
  ¬∃ (a b c d : ℚ) (n : ℕ), (a + b * Real.sqrt 2)^(2*n) + (c + d * Real.sqrt 2)^(2*n) = 5 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l1305_130516


namespace NUMINAMATH_CALUDE_divisibility_by_24_l1305_130537

theorem divisibility_by_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l1305_130537


namespace NUMINAMATH_CALUDE_greatest_x_value_l1305_130510

theorem greatest_x_value (x : ℤ) (h : (3.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 31000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → (3.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 31000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1305_130510


namespace NUMINAMATH_CALUDE_smallest_m_is_26_l1305_130522

def S : Finset Nat := Finset.range 100

-- Define the property we want to prove
def has_divisor (A : Finset Nat) : Prop :=
  ∃ x ∈ A, x ∣ (A.prod (fun y => if y ≠ x then y else 1))

theorem smallest_m_is_26 : 
  (∀ A : Finset Nat, A ⊆ S → A.card = 26 → has_divisor A) ∧ 
  (∀ m < 26, ∃ A : Finset Nat, A ⊆ S ∧ A.card = m ∧ ¬has_divisor A) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_26_l1305_130522


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1305_130589

theorem fractional_equation_solution :
  ∃ (x : ℝ), (3 / (x - 3) - 1 = 1 / (3 - x)) ∧ (x = 7) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1305_130589


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1305_130513

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  /-- Length of the first leg -/
  leg1 : ℝ
  /-- Length of the second leg -/
  leg2 : ℝ
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The triangle is right-angled -/
  right_angle : leg1 ^ 2 + leg2 ^ 2 = hypotenuse ^ 2
  /-- All sides are positive -/
  leg1_pos : leg1 > 0
  leg2_pos : leg2 > 0
  hypotenuse_pos : hypotenuse > 0
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed in the triangle -/
  inscribed : square_side > 0 ∧ square_side < leg1 ∧ square_side < leg2

/-- The side length of the inscribed square in the given right triangle is 12/5 -/
theorem inscribed_square_side_length (t : RightTriangleWithSquare) 
    (h1 : t.leg1 = 5) (h2 : t.leg2 = 12) (h3 : t.hypotenuse = 13) : 
    t.square_side = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1305_130513


namespace NUMINAMATH_CALUDE_max_fridays_12th_non_leap_year_max_fridays_12th_leap_year_l1305_130597

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

/-- Represents a month of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Returns the number of days in a given month for a given year -/
def daysInMonth (m : Month) (isLeapYear : Bool) : Nat :=
  match m with
  | .January => 31
  | .February => if isLeapYear then 29 else 28
  | .March => 31
  | .April => 30
  | .May => 31
  | .June => 30
  | .July => 31
  | .August => 31
  | .September => 30
  | .October => 31
  | .November => 30
  | .December => 31

/-- Returns the day of the week for the 12th of a given month, 
    given the day of the week of January 1st -/
def dayOfWeekOn12th (m : Month) (jan1 : DayOfWeek) (isLeapYear : Bool) : DayOfWeek :=
  sorry

/-- Counts the number of Fridays that fall on the 12th in a year -/
def countFridays12th (jan1 : DayOfWeek) (isLeapYear : Bool) : Nat :=
  sorry

/-- Theorem: In a non-leap year, there can be at most 3 Fridays 
    that fall on the 12th of a month -/
theorem max_fridays_12th_non_leap_year :
  ∀ (jan1 : DayOfWeek), countFridays12th jan1 false ≤ 3 :=
  sorry

/-- Theorem: In a leap year, there can be at most 4 Fridays 
    that fall on the 12th of a month -/
theorem max_fridays_12th_leap_year :
  ∀ (jan1 : DayOfWeek), countFridays12th jan1 true ≤ 4 :=
  sorry

end NUMINAMATH_CALUDE_max_fridays_12th_non_leap_year_max_fridays_12th_leap_year_l1305_130597


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l1305_130549

theorem rebus_puzzle_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C ∧
    100 * A + 10 * C + C = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l1305_130549


namespace NUMINAMATH_CALUDE_spheres_intersection_similar_triangles_l1305_130592

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Checks if a point lies on an edge of a tetrahedron -/
def on_edge (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Checks if a sphere passes through a point -/
def sphere_passes_through (s : Sphere) (p : Point3D) : Prop := sorry

/-- Checks if two triangles are similar -/
def triangles_similar (p1 p2 p3 q1 q2 q3 : Point3D) : Prop := sorry

/-- Main theorem -/
theorem spheres_intersection_similar_triangles 
  (ABCD : Tetrahedron) (G₁ G₂ : Sphere) 
  (K L M P Q R : Point3D) : 
  sphere_passes_through G₁ ABCD.A ∧ 
  sphere_passes_through G₁ ABCD.B ∧ 
  sphere_passes_through G₁ ABCD.C ∧
  sphere_passes_through G₂ ABCD.A ∧ 
  sphere_passes_through G₂ ABCD.B ∧ 
  sphere_passes_through G₂ ABCD.D ∧
  on_edge K ABCD ∧ collinear K ABCD.D ABCD.A ∧
  on_edge L ABCD ∧ collinear L ABCD.D ABCD.B ∧
  on_edge M ABCD ∧ collinear M ABCD.D ABCD.C ∧
  on_edge P ABCD ∧ collinear P ABCD.C ABCD.A ∧
  on_edge Q ABCD ∧ collinear Q ABCD.C ABCD.B ∧
  on_edge R ABCD ∧ collinear R ABCD.C ABCD.D
  →
  triangles_similar K L M P Q R := by
  sorry

end NUMINAMATH_CALUDE_spheres_intersection_similar_triangles_l1305_130592


namespace NUMINAMATH_CALUDE_prize_prices_and_min_cost_l1305_130581

/- Define the unit prices of prizes A and B -/
def price_A : ℝ := 20
def price_B : ℝ := 10

/- Define the total number of prizes and minimum number of prize A -/
def total_prizes : ℕ := 60
def min_prize_A : ℕ := 20

/- Define the cost function -/
def cost (m : ℕ) : ℝ := price_A * m + price_B * (total_prizes - m)

theorem prize_prices_and_min_cost :
  /- Condition 1: 1 A and 2 B cost $40 -/
  price_A + 2 * price_B = 40 ∧
  /- Condition 2: 2 A and 3 B cost $70 -/
  2 * price_A + 3 * price_B = 70 ∧
  /- The minimum cost occurs when m = min_prize_A -/
  (∀ m : ℕ, min_prize_A ≤ m → m ≤ total_prizes → cost min_prize_A ≤ cost m) ∧
  /- The minimum cost is $800 -/
  cost min_prize_A = 800 := by
  sorry

#check prize_prices_and_min_cost

end NUMINAMATH_CALUDE_prize_prices_and_min_cost_l1305_130581


namespace NUMINAMATH_CALUDE_stratified_sampling_l1305_130543

theorem stratified_sampling (total_employees : ℕ) (administrators : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 160)
  (h2 : administrators = 32)
  (h3 : sample_size = 20) :
  (administrators * sample_size) / total_employees = 4 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1305_130543


namespace NUMINAMATH_CALUDE_statement_true_except_two_and_five_l1305_130577

theorem statement_true_except_two_and_five (x : ℝ) :
  (x - 2) * (x - 5) ≠ 0 ↔ x ≠ 2 ∧ x ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_statement_true_except_two_and_five_l1305_130577


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1305_130554

/-- Given a trader selling cloth, calculate the cost price per metre. -/
theorem cloth_cost_price
  (total_metres : ℕ)
  (selling_price : ℕ)
  (profit_per_metre : ℕ)
  (h1 : total_metres = 30)
  (h2 : selling_price = 4500)
  (h3 : profit_per_metre = 10) :
  (selling_price - total_metres * profit_per_metre) / total_metres = 140 :=
sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1305_130554


namespace NUMINAMATH_CALUDE_all_transformations_pass_through_point_l1305_130572

def f (x : ℝ) := (x - 2)^2
def g (x : ℝ) := (x - 1)^2 - 1
def h (x : ℝ) := x^2 - 4
def k (x : ℝ) := -x^2 + 4

theorem all_transformations_pass_through_point :
  f 2 = 0 ∧ g 2 = 0 ∧ h 2 = 0 ∧ k 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_transformations_pass_through_point_l1305_130572


namespace NUMINAMATH_CALUDE_system_solution_l1305_130575

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 8) (eq2 : 2*x + y = 1) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1305_130575


namespace NUMINAMATH_CALUDE_work_ratio_l1305_130501

theorem work_ratio (a b : ℝ) (ha : a = 8) (hab : 1/a + 1/b = 0.375) : b/a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_work_ratio_l1305_130501
