import Mathlib

namespace a_minus_b_equals_15_l1799_179974

/-- Represents the division of money among A, B, and C -/
structure MoneyDivision where
  a : ℝ  -- Amount received by A
  b : ℝ  -- Amount received by B
  c : ℝ  -- Amount received by C

/-- Conditions for the money division problem -/
def validDivision (d : MoneyDivision) : Prop :=
  d.a = (1/3) * (d.b + d.c) ∧
  d.b = (2/7) * (d.a + d.c) ∧
  d.a > d.b ∧
  d.a + d.b + d.c = 540

/-- Theorem stating that A receives $15 more than B -/
theorem a_minus_b_equals_15 (d : MoneyDivision) (h : validDivision d) :
  d.a - d.b = 15 := by
  sorry

end a_minus_b_equals_15_l1799_179974


namespace value_of_a_l1799_179951

theorem value_of_a (a : ℝ) (S : Set ℝ) : 
  S = {x : ℝ | 3 * x + a = 0} → 
  (1 : ℝ) ∈ S → 
  a = -3 := by
sorry

end value_of_a_l1799_179951


namespace flight_time_theorem_l1799_179928

/-- Represents the flight time between two towns -/
structure FlightTime where
  against_wind : ℝ
  with_wind : ℝ
  no_wind : ℝ

/-- The flight time satisfies the given conditions -/
def satisfies_conditions (ft : FlightTime) : Prop :=
  ft.against_wind = 84 ∧ ft.with_wind = ft.no_wind - 9

/-- The theorem to be proved -/
theorem flight_time_theorem (ft : FlightTime) 
  (h : satisfies_conditions ft) : 
  ft.with_wind = 63 ∨ ft.with_wind = 12 := by
  sorry

end flight_time_theorem_l1799_179928


namespace intersection_M_N_l1799_179953

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 2) ≤ 0}
def N : Set ℝ := {x | x - 1 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by
  sorry

end intersection_M_N_l1799_179953


namespace reciprocal_of_three_halves_l1799_179938

-- Define the concept of reciprocal
def is_reciprocal (a b : ℚ) : Prop := a * b = 1

-- State the theorem
theorem reciprocal_of_three_halves : 
  is_reciprocal (3/2 : ℚ) (2/3 : ℚ) := by
  sorry

end reciprocal_of_three_halves_l1799_179938


namespace vector_subtraction_l1799_179906

def a : Fin 3 → ℝ := ![-3, 4, 2]
def b : Fin 3 → ℝ := ![5, -1, 3]

theorem vector_subtraction :
  (fun i => a i - 2 * b i) = ![-13, 6, -4] := by sorry

end vector_subtraction_l1799_179906


namespace vincent_book_cost_l1799_179920

/-- Calculates the total cost of Vincent's books --/
def total_cost (animal_books train_books history_books cooking_books : ℕ) 
  (animal_price outer_space_price train_price history_price cooking_price : ℕ) : ℕ :=
  animal_books * animal_price + 
  1 * outer_space_price + 
  train_books * train_price + 
  history_books * history_price + 
  cooking_books * cooking_price

/-- Theorem stating that Vincent's total book cost is $356 --/
theorem vincent_book_cost : 
  total_cost 10 3 5 2 16 20 14 18 22 = 356 := by
  sorry


end vincent_book_cost_l1799_179920


namespace bucket_weight_l1799_179955

/-- Given a bucket where:
    - The weight when half full (including the bucket) is c
    - The weight when completely full (including the bucket) is d
    This theorem proves that the weight when three-quarters full is (1/2)c + (1/2)d -/
theorem bucket_weight (c d : ℝ) : ℝ :=
  let half_full := c
  let full := d
  let three_quarters_full := (1/2 : ℝ) * c + (1/2 : ℝ) * d
  three_quarters_full

#check bucket_weight

end bucket_weight_l1799_179955


namespace square_fence_poles_l1799_179983

/-- Given a square fence with a total of 104 poles, prove that the number of poles on each side is 26. -/
theorem square_fence_poles (total_poles : ℕ) (h1 : total_poles = 104) :
  ∃ (side_poles : ℕ), side_poles * 4 = total_poles ∧ side_poles = 26 := by
  sorry

end square_fence_poles_l1799_179983


namespace units_digit_of_2189_power_1242_l1799_179970

theorem units_digit_of_2189_power_1242 : ∃ n : ℕ, 2189^1242 ≡ 1 [ZMOD 10] :=
sorry

end units_digit_of_2189_power_1242_l1799_179970


namespace number_ordering_l1799_179930

theorem number_ordering (a b c : ℝ) : 
  a = 9^(1/3) → b = 3^(2/5) → c = 4^(1/5) → a > b ∧ b > c := by
  sorry

end number_ordering_l1799_179930


namespace multiple_remainder_l1799_179956

theorem multiple_remainder (n m : ℤ) (h1 : n % 7 = 1) (h2 : ∃ k, (k * n) % 7 = 3) :
  m % 7 = 3 → (m * n) % 7 = 3 := by
sorry

end multiple_remainder_l1799_179956


namespace exclusive_or_implications_l1799_179971

theorem exclusive_or_implications (p q : Prop) 
  (h_or : p ∨ q) (h_not_and : ¬(p ∧ q)) : 
  (q ↔ ¬p) ∧ (p ↔ ¬q) := by
  sorry

end exclusive_or_implications_l1799_179971


namespace solve_for_q_l1799_179937

theorem solve_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
  sorry

end solve_for_q_l1799_179937


namespace pencil_count_l1799_179940

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 2

/-- The number of pencils Tim added to the drawer -/
def added_pencils : ℕ := 3

/-- The total number of pencils in the drawer after Tim's action -/
def total_pencils : ℕ := initial_pencils + added_pencils

theorem pencil_count : total_pencils = 5 := by
  sorry

end pencil_count_l1799_179940


namespace arithmetic_sequence_characterization_l1799_179931

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

theorem arithmetic_sequence_characterization (a : ℕ+ → ℝ) :
  is_arithmetic_sequence a ↔ ∀ n : ℕ+, 2 * a (n + 1) = a n + a (n + 2) :=
sorry

end arithmetic_sequence_characterization_l1799_179931


namespace swim_team_total_l1799_179922

theorem swim_team_total (girls : ℕ) (boys : ℕ) : 
  girls = 80 → girls = 5 * boys → girls + boys = 96 := by
  sorry

end swim_team_total_l1799_179922


namespace convention_handshakes_l1799_179913

theorem convention_handshakes (num_companies num_representatives_per_company : ℕ) :
  num_companies = 4 →
  num_representatives_per_company = 4 →
  let total_people := num_companies * num_representatives_per_company
  let handshakes_per_person := total_people - num_representatives_per_company
  (total_people * handshakes_per_person) / 2 = 96 :=
by
  sorry

end convention_handshakes_l1799_179913


namespace triangle_properties_l1799_179915

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C

-- Define the theorem
theorem triangle_properties (ABC : Triangle) 
  (h1 : Real.cos (ABC.A / 2) = 2 * Real.sqrt 5 / 5)
  (h2 : ABC.b * ABC.c * Real.cos ABC.A = 15)
  (h3 : Real.tan ABC.B = 2) : 
  (1/2 * ABC.b * ABC.c * Real.sin ABC.A = 10) ∧ 
  (ABC.a = 2 * Real.sqrt 5) := by
sorry


end triangle_properties_l1799_179915


namespace initial_overs_played_l1799_179950

/-- Proves that the number of overs played initially is 10, given the specified conditions --/
theorem initial_overs_played (total_target : ℝ) (initial_run_rate : ℝ) (remaining_overs : ℝ) (required_run_rate : ℝ)
  (h1 : total_target = 282)
  (h2 : initial_run_rate = 3.8)
  (h3 : remaining_overs = 40)
  (h4 : required_run_rate = 6.1)
  : ∃ (x : ℝ), x = 10 ∧ initial_run_rate * x + required_run_rate * remaining_overs = total_target :=
by
  sorry

end initial_overs_played_l1799_179950


namespace regular_polygon_sides_l1799_179905

theorem regular_polygon_sides (n : ℕ) (h_exterior : (360 : ℝ) / n = 30) 
  (h_interior : (180 : ℝ) - 30 = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l1799_179905


namespace geometric_sequence_roots_property_l1799_179991

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the property of a_4 and a_12 being roots of x^2 + 3x + 1 = 0
def roots_property (a : ℕ → ℝ) : Prop :=
  a 4 + a 12 = -3 ∧ a 4 * a 12 = 1

theorem geometric_sequence_roots_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (roots_property a → a 8 = -1) ∧
  ¬(a 8 = -1 → roots_property a) :=
sorry

end geometric_sequence_roots_property_l1799_179991


namespace display_window_problem_l1799_179935

/-- The number of configurations for two display windows --/
def total_configurations : ℕ := 36

/-- The number of non-fiction books in the right window --/
def non_fiction_books : ℕ := 3

/-- The number of fiction books in the left window --/
def fiction_books : ℕ := 3

theorem display_window_problem :
  fiction_books.factorial * non_fiction_books.factorial = total_configurations :=
sorry

end display_window_problem_l1799_179935


namespace gcd_lcm_45_75_l1799_179954

theorem gcd_lcm_45_75 :
  (Nat.gcd 45 75 = 15) ∧ (Nat.lcm 45 75 = 1125) := by
  sorry

end gcd_lcm_45_75_l1799_179954


namespace polynomial_simplification_l1799_179909

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (-x^4 + 4 * x^3 - 2 * x^2 + 8 * x - 7) =
  x^4 + x^3 + 3 * x^2 + 8 := by
  sorry

end polynomial_simplification_l1799_179909


namespace largest_bundle_size_correct_l1799_179923

def largest_bundle_size (john_notebooks emily_notebooks min_bundle_size : ℕ) : ℕ :=
  Nat.gcd john_notebooks emily_notebooks

theorem largest_bundle_size_correct 
  (john_notebooks : ℕ) 
  (emily_notebooks : ℕ) 
  (min_bundle_size : ℕ) 
  (h1 : john_notebooks = 36) 
  (h2 : emily_notebooks = 45) 
  (h3 : min_bundle_size = 5) :
  largest_bundle_size john_notebooks emily_notebooks min_bundle_size = 9 ∧ 
  largest_bundle_size john_notebooks emily_notebooks min_bundle_size > min_bundle_size := by
  sorry

#eval largest_bundle_size 36 45 5

end largest_bundle_size_correct_l1799_179923


namespace total_presents_equals_58_l1799_179993

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := ethan_presents - 22

/-- The number of presents Bella has -/
def bella_presents : ℕ := 2 * alissa_presents

/-- The total number of presents Bella, Ethan, and Alissa have -/
def total_presents : ℕ := ethan_presents + alissa_presents + bella_presents

theorem total_presents_equals_58 : total_presents = 58 := by
  sorry

end total_presents_equals_58_l1799_179993


namespace area_AEC_is_18_l1799_179934

-- Define the lengths BE and EC
def BE : ℝ := 3
def EC : ℝ := 2

-- Define the area of triangle ABE
def area_ABE : ℝ := 27

-- Theorem statement
theorem area_AEC_is_18 :
  let ratio := BE / EC
  let area_AEC := (EC / BE) * area_ABE
  area_AEC = 18 := by sorry

end area_AEC_is_18_l1799_179934


namespace triangles_forming_square_even_l1799_179921

theorem triangles_forming_square_even (n : ℕ) (a : ℕ) : 
  (n * 6 = a * a) → Even n := by sorry

end triangles_forming_square_even_l1799_179921


namespace spy_is_A_l1799_179964

/-- Represents the three defendants -/
inductive Defendant : Type
  | A
  | B
  | C

/-- Represents the role of each defendant -/
inductive Role : Type
  | Spy
  | Knight
  | Liar

/-- The statement made by each defendant -/
def statement (d : Defendant) : Prop :=
  match d with
  | Defendant.A => ∃ r, r = Role.Spy
  | Defendant.B => ∃ r, r = Role.Knight
  | Defendant.C => ∃ r, r = Role.Spy

/-- The role assigned to each defendant -/
def assigned_role : Defendant → Role := sorry

/-- A defendant tells the truth if they are the Knight or if they are the Spy and claim to be the Spy -/
def tells_truth (d : Defendant) : Prop :=
  (assigned_role d = Role.Knight) ∨
  (assigned_role d = Role.Spy ∧ statement d)

theorem spy_is_A :
  (∃! d : Defendant, assigned_role d = Role.Spy) ∧
  (∃! d : Defendant, assigned_role d = Role.Knight) ∧
  (∃! d : Defendant, assigned_role d = Role.Liar) ∧
  (tells_truth Defendant.B) →
  assigned_role Defendant.A = Role.Spy := by
  sorry


end spy_is_A_l1799_179964


namespace shaded_area_calculation_l1799_179973

/-- Represents the shaded area calculation problem on a grid with circles --/
theorem shaded_area_calculation (grid_size : ℕ) (small_circle_radius : ℝ) (large_circle_radius : ℝ) 
  (small_circle_count : ℕ) (large_circle_count : ℕ) :
  grid_size = 6 ∧ 
  small_circle_radius = 0.5 ∧ 
  large_circle_radius = 1 ∧
  small_circle_count = 4 ∧
  large_circle_count = 2 →
  ∃ (A C : ℝ), 
    (A - C * Real.pi = grid_size^2 - (small_circle_count * small_circle_radius^2 + large_circle_count * large_circle_radius^2) * Real.pi) ∧
    A + C = 39 := by
  sorry

end shaded_area_calculation_l1799_179973


namespace unique_six_digit_reverse_when_multiplied_by_nine_l1799_179908

/-- A function that returns the digits of a natural number in reverse order -/
def reverseDigits (n : ℕ) : List ℕ :=
  sorry

/-- A function that checks if a number is a six-digit number -/
def isSixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

/-- The main theorem stating that 109989 is the only six-digit number
    that, when multiplied by 9, has its digits arranged in reverse order -/
theorem unique_six_digit_reverse_when_multiplied_by_nine :
  ∀ n : ℕ, isSixDigitNumber n →
    (reverseDigits n = reverseDigits (9 * n) → n = 109989) ∧
    (n = 109989 → reverseDigits n = reverseDigits (9 * n)) :=
by sorry

end unique_six_digit_reverse_when_multiplied_by_nine_l1799_179908


namespace shaded_area_equals_36_plus_18pi_l1799_179977

-- Define the circle and its properties
def circle_radius : ℝ := 6

-- Define the triangles and their properties
def triangle_OAC_isosceles_right : Prop := sorry
def triangle_OBD_right : Prop := sorry

-- Define the areas
def area_triangle_OAC : ℝ := sorry
def area_triangle_OBD : ℝ := sorry
def area_sector_OAB : ℝ := sorry
def area_sector_OCD : ℝ := sorry

-- Theorem statement
theorem shaded_area_equals_36_plus_18pi :
  triangle_OAC_isosceles_right →
  triangle_OBD_right →
  area_triangle_OAC + area_triangle_OBD + area_sector_OAB + area_sector_OCD = 36 + 18 * Real.pi :=
by sorry

end shaded_area_equals_36_plus_18pi_l1799_179977


namespace exactly_one_incorrect_statement_l1799_179901

/-- Represents a statement about regression analysis -/
inductive RegressionStatement
  | residualBand
  | scatterPlotCorrelation
  | regressionLineInterpretation
  | sumSquaredResiduals

/-- Determines if a given statement about regression analysis is correct -/
def isCorrect (statement : RegressionStatement) : Prop :=
  match statement with
  | .residualBand => True
  | .scatterPlotCorrelation => False
  | .regressionLineInterpretation => True
  | .sumSquaredResiduals => True

theorem exactly_one_incorrect_statement :
  ∃! (s : RegressionStatement), ¬(isCorrect s) :=
sorry

end exactly_one_incorrect_statement_l1799_179901


namespace initial_total_marbles_l1799_179969

/-- Represents the number of parts in the ratio for each person -/
def brittany_ratio : ℕ := 3
def alex_ratio : ℕ := 5
def jamy_ratio : ℕ := 7

/-- Represents the total number of marbles Alex has after receiving half of Brittany's marbles -/
def alex_final_marbles : ℕ := 260

/-- The theorem stating the initial total number of marbles -/
theorem initial_total_marbles :
  ∃ (x : ℕ),
    (brittany_ratio * x + alex_ratio * x + jamy_ratio * x = 600) ∧
    (alex_ratio * x + (brittany_ratio * x) / 2 = alex_final_marbles) :=
by sorry

end initial_total_marbles_l1799_179969


namespace sum_of_fractions_simplification_l1799_179985

theorem sum_of_fractions_simplification (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h_sum : a + b + c + d = 0) :
  (1 / (b^2 + c^2 + d^2 - a^2)) + 
  (1 / (a^2 + c^2 + d^2 - b^2)) + 
  (1 / (a^2 + b^2 + d^2 - c^2)) + 
  (1 / (a^2 + b^2 + c^2 - d^2)) = 4 / d^2 := by
  sorry

end sum_of_fractions_simplification_l1799_179985


namespace rowing_speed_in_still_water_l1799_179929

/-- The speed of a man rowing in still water, given downstream conditions -/
theorem rowing_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 8.5)
  (h2 : distance = 45.5)
  (h3 : time = 9.099272058235341)
  : ∃ (still_water_speed : ℝ), still_water_speed = 9.5 := by
  sorry

end rowing_speed_in_still_water_l1799_179929


namespace quadratic_inequality_solution_l1799_179998

-- Define the quadratic function
def f (a b x : ℝ) := x^2 + b*x + a

-- State the theorem
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, f a b x > 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioi 5) : 
  a + b = -1 := by
  sorry

end quadratic_inequality_solution_l1799_179998


namespace symmetric_point_coordinates_l1799_179911

/-- Given a point P, return its symmetric point with respect to the y-axis -/
def symmetric_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Given a point P, return its symmetric point with respect to the x-axis -/
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := (-10, -1)
  let P₁ : ℝ × ℝ := symmetric_y P
  let P₂ : ℝ × ℝ := symmetric_x P₁
  P₂ = (10, 1) := by sorry

end symmetric_point_coordinates_l1799_179911


namespace existence_of_alpha_l1799_179975

theorem existence_of_alpha (p : Nat) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ α : Nat, 1 ≤ α ∧ α ≤ p - 2 ∧
    ¬(p^2 ∣ α^(p-1) - 1) ∧ ¬(p^2 ∣ (α+1)^(p-1) - 1) := by
  sorry

end existence_of_alpha_l1799_179975


namespace sector_area_l1799_179966

/-- The area of a sector with central angle 150° and radius 3 is 15π/4 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 150 * π / 180) (h2 : r = 3) :
  (1/2) * r^2 * θ = 15 * π / 4 := by
  sorry

end sector_area_l1799_179966


namespace calculate_required_hours_per_week_l1799_179984

/-- Proves that given an initial work plan and a period of unavailable work time,
    the required hours per week to meet the financial goal can be calculated. -/
theorem calculate_required_hours_per_week 
  (initial_hours_per_week : ℝ)
  (initial_weeks : ℝ)
  (financial_goal : ℝ)
  (unavailable_weeks : ℝ)
  (h1 : initial_hours_per_week = 25)
  (h2 : initial_weeks = 15)
  (h3 : financial_goal = 4500)
  (h4 : unavailable_weeks = 3)
  : (initial_hours_per_week * initial_weeks) / (initial_weeks - unavailable_weeks) = 31.25 := by
  sorry

end calculate_required_hours_per_week_l1799_179984


namespace percentage_decrease_l1799_179941

theorem percentage_decrease (w : ℝ) (x : ℝ) (h1 : w = 80) (h2 : w * (1 + 0.125) - w * (1 - x / 100) = 30) : x = 25 := by
  sorry

end percentage_decrease_l1799_179941


namespace sum_of_A_and_C_is_eight_l1799_179947

theorem sum_of_A_and_C_is_eight :
  ∀ (A B C D : ℕ),
    A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A : ℚ) / B - (C : ℚ) / D = 2 →
    A + C = 8 :=
by
  sorry


end sum_of_A_and_C_is_eight_l1799_179947


namespace g_negative_three_value_l1799_179960

theorem g_negative_three_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g (5 * x - 7) = 8 * x + 2) :
  g (-3) = 8.4 := by
  sorry

end g_negative_three_value_l1799_179960


namespace polynomial_simplification_l1799_179949

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) =
  2 * p^4 + 6 * p^3 + p^2 + p + 4 := by
  sorry

end polynomial_simplification_l1799_179949


namespace fraction_ordering_l1799_179944

theorem fraction_ordering : (4 : ℚ) / 17 < 6 / 25 ∧ 6 / 25 < 8 / 31 := by
  sorry

end fraction_ordering_l1799_179944


namespace factorial_calculation_l1799_179919

theorem factorial_calculation : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end factorial_calculation_l1799_179919


namespace min_unsuccessful_placements_l1799_179924

/-- Represents a cell in the grid -/
inductive Cell
| Plus : Cell
| Minus : Cell

/-- Represents an 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Cell

/-- Represents a T-shaped figure -/
structure TShape where
  row : Fin 8
  col : Fin 8
  orientation : Bool  -- True for horizontal, False for vertical

/-- Calculates the sum of a T-shape on the grid -/
def tShapeSum (g : Grid) (t : TShape) : Int :=
  sorry

/-- Counts the number of unsuccessful T-shape placements -/
def countUnsuccessful (g : Grid) : Nat :=
  sorry

/-- Theorem: The minimum number of unsuccessful T-shape placements is 132 -/
theorem min_unsuccessful_placements :
  ∀ g : Grid, countUnsuccessful g ≥ 132 :=
sorry

end min_unsuccessful_placements_l1799_179924


namespace inequality_proof_l1799_179996

theorem inequality_proof (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) :
  Real.exp x₂ * Real.log x₁ < Real.exp x₁ * Real.log x₂ := by
  sorry

end inequality_proof_l1799_179996


namespace sector_max_area_l1799_179982

/-- Given a sector with circumference 12 cm, its maximum area is 9 cm². -/
theorem sector_max_area (r l : ℝ) (h_circumference : 2 * r + l = 12) :
  (1/2 : ℝ) * l * r ≤ 9 := by
  sorry

end sector_max_area_l1799_179982


namespace billys_age_l1799_179988

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end billys_age_l1799_179988


namespace largest_perfect_square_factor_of_2800_l1799_179981

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_of_2800 :
  largest_perfect_square_factor 2800 = 400 := by sorry

end largest_perfect_square_factor_of_2800_l1799_179981


namespace lamp_sales_theorem_l1799_179990

/-- The monthly average growth rate of lamp sales -/
def monthly_growth_rate : ℝ := 0.2

/-- The price of lamps in April to achieve the target profit -/
def april_price : ℝ := 38

/-- Initial sales volume in January -/
def january_sales : ℕ := 400

/-- Sales volume in March -/
def march_sales : ℕ := 576

/-- Purchase cost per lamp -/
def purchase_cost : ℝ := 30

/-- Initial selling price -/
def initial_price : ℝ := 40

/-- Increase in sales volume per 0.5 yuan price reduction -/
def sales_increase_per_half_yuan : ℕ := 6

/-- Target profit in April -/
def target_profit : ℝ := 4800

/-- Theorem stating the correctness of the monthly growth rate and April price -/
theorem lamp_sales_theorem :
  (january_sales * (1 + monthly_growth_rate)^2 = march_sales) ∧
  ((april_price - purchase_cost) *
    (march_sales + 2 * sales_increase_per_half_yuan * (initial_price - april_price)) = target_profit) := by
  sorry


end lamp_sales_theorem_l1799_179990


namespace sum_of_integers_l1799_179904

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end sum_of_integers_l1799_179904


namespace scientific_notation_correct_l1799_179916

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 4600

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation :=
  { coefficient := 4.6
    exponent := 3
    property := by sorry }

/-- Theorem stating that the scientific notation form is correct -/
theorem scientific_notation_correct :
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end scientific_notation_correct_l1799_179916


namespace square_diagonal_and_area_l1799_179957

/-- Given a square with side length 30√3 cm, this theorem proves the length of its diagonal and its area. -/
theorem square_diagonal_and_area :
  let side_length : ℝ := 30 * Real.sqrt 3
  let diagonal : ℝ := side_length * Real.sqrt 2
  let area : ℝ := side_length ^ 2
  diagonal = 30 * Real.sqrt 6 ∧ area = 2700 := by
  sorry

end square_diagonal_and_area_l1799_179957


namespace rachel_furniture_assembly_time_l1799_179959

/-- Calculates the total time to assemble furniture -/
def total_assembly_time (chairs : ℕ) (tables : ℕ) (time_per_piece : ℕ) : ℕ :=
  (chairs + tables) * time_per_piece

/-- Theorem: The total assembly time for Rachel's furniture -/
theorem rachel_furniture_assembly_time :
  total_assembly_time 7 3 4 = 40 := by
  sorry

end rachel_furniture_assembly_time_l1799_179959


namespace mod_equivalence_problem_l1799_179917

theorem mod_equivalence_problem : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 8173 [ZMOD 15] ∧ n = 13 := by
  sorry

end mod_equivalence_problem_l1799_179917


namespace units_digit_sum_factorials_500_l1799_179939

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |> List.sum

theorem units_digit_sum_factorials_500 :
  units_digit (sum_factorials 500) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end units_digit_sum_factorials_500_l1799_179939


namespace solve_potato_problem_l1799_179903

def potato_problem (total_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) : Prop :=
  let remaining_potatoes := remaining_time / time_per_potato
  let cooked_potatoes := total_potatoes - remaining_potatoes
  cooked_potatoes = total_potatoes - (remaining_time / time_per_potato)

theorem solve_potato_problem :
  potato_problem 12 6 36 :=
by
  sorry

end solve_potato_problem_l1799_179903


namespace coordinate_plane_points_theorem_l1799_179943

theorem coordinate_plane_points_theorem (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 → ((x = 0 ∧ y = 0) ∨ y = 2)) ∧
  (x * y + 1 = x + y → (x = 1 ∨ y = 1)) := by
  sorry

end coordinate_plane_points_theorem_l1799_179943


namespace shipping_cost_invariant_l1799_179963

/-- Represents a settlement with its distance from the city and required goods weight -/
structure Settlement where
  distance : ℝ
  weight : ℝ
  distance_eq_weight : distance = weight

/-- Calculates the shipping cost for a given delivery order -/
def shipping_cost (settlements : List Settlement) : ℝ :=
  settlements.enum.foldl
    (fun acc (i, s) =>
      acc + s.weight * (settlements.take i).foldl (fun sum t => sum + t.distance) 0)
    0

/-- Theorem stating that the shipping cost is invariant under different delivery orders -/
theorem shipping_cost_invariant (settlements : List Settlement) :
  ∀ (perm : List Settlement), settlements.Perm perm →
    shipping_cost settlements = shipping_cost perm :=
  sorry

end shipping_cost_invariant_l1799_179963


namespace average_of_four_numbers_l1799_179986

theorem average_of_four_numbers (r s t u : ℝ) :
  (5 / 2) * (r + s + t + u) = 25 → (r + s + t + u) / 4 = 2.5 := by
sorry

end average_of_four_numbers_l1799_179986


namespace square_equality_base_is_ten_l1799_179926

/-- The base in which 34 squared equals 1296 -/
def base_b : ℕ := sorry

/-- The representation of 34 in base b -/
def thirty_four_b (b : ℕ) : ℕ := 3 * b + 4

/-- The representation of 1296 in base b -/
def twelve_ninety_six_b (b : ℕ) : ℕ := b^3 + 2*b^2 + 9*b + 6

/-- The theorem stating that the square of 34 in base b equals 1296 in base b -/
theorem square_equality (b : ℕ) : (thirty_four_b b)^2 = twelve_ninety_six_b b := by sorry

/-- The main theorem proving that the base b is 10 -/
theorem base_is_ten : base_b = 10 := by sorry

end square_equality_base_is_ten_l1799_179926


namespace incoming_scholars_count_l1799_179936

theorem incoming_scholars_count :
  ∃! n : ℕ, n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 ∧ n = 509 := by
  sorry

end incoming_scholars_count_l1799_179936


namespace max_d_is_401_l1799_179961

/-- The sequence a_n defined as n^2 + 100 -/
def a (n : ℕ+) : ℕ := n^2 + 100

/-- The sequence d_n defined as the gcd of a_n and a_{n+1} -/
def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The theorem stating that the maximum value of d_n is 401 -/
theorem max_d_is_401 : ∃ (n : ℕ+), d n = 401 ∧ ∀ (m : ℕ+), d m ≤ 401 := by
  sorry

end max_d_is_401_l1799_179961


namespace abs_z_minus_i_equals_sqrt2_over_2_l1799_179967

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z based on the given condition
def z : ℂ := by
  sorry

-- Theorem statement
theorem abs_z_minus_i_equals_sqrt2_over_2 :
  Complex.abs (z - i) = Real.sqrt 2 / 2 := by
  sorry

end abs_z_minus_i_equals_sqrt2_over_2_l1799_179967


namespace unique_four_digit_number_l1799_179932

theorem unique_four_digit_number :
  ∃! x : ℕ, 
    1000 ≤ x ∧ x < 10000 ∧
    x + (x % 10) = 5574 ∧
    x + ((x / 10) % 10) = 557 := by
  sorry

end unique_four_digit_number_l1799_179932


namespace trigonometric_simplification_l1799_179997

theorem trigonometric_simplification (α : ℝ) :
  Real.sin ((5 / 2) * Real.pi + 4 * α) - 
  Real.sin ((5 / 2) * Real.pi + 2 * α) ^ 6 + 
  Real.cos ((7 / 2) * Real.pi - 2 * α) ^ 6 = 
  (1 / 8) * Real.sin (8 * α) * Real.sin (4 * α) := by
  sorry

end trigonometric_simplification_l1799_179997


namespace solve_for_m_l1799_179968

theorem solve_for_m : ∀ m : ℚ, 
  (∃ x y : ℚ, m * x + y = 2 ∧ x = -2 ∧ y = 1) → m = -1/2 := by
  sorry

end solve_for_m_l1799_179968


namespace solution_set_when_a_is_one_range_of_a_for_inequality_l1799_179994

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x ∈ Set.Ioo 0 1, f a x > x} = Set.Ioc 0 2 := by sorry

end solution_set_when_a_is_one_range_of_a_for_inequality_l1799_179994


namespace range_of_g_l1799_179942

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := |x|

-- Define the domain
def domain : Set ℝ := {x | -4 ≤ x ∧ x ≤ 4}

-- Define the function g(x) = f(x) - x
def g (x : ℝ) : ℝ := f x - x

-- Theorem statement
theorem range_of_g :
  {y | ∃ x ∈ domain, g x = y} = {y | 0 ≤ y ∧ y ≤ 8} := by
  sorry

end range_of_g_l1799_179942


namespace ellipse_with_foci_on_y_axis_range_l1799_179927

/-- The equation of the curve -/
def equation (x y k : ℝ) : Prop := x^2 / (k - 5) + y^2 / (10 - k) = 1

/-- The condition for the equation to represent an ellipse -/
def is_ellipse (k : ℝ) : Prop := k - 5 > 0 ∧ 10 - k > 0

/-- The condition for the foci to be on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop := 10 - k > k - 5

/-- The theorem stating the range of k for which the equation represents an ellipse with foci on the y-axis -/
theorem ellipse_with_foci_on_y_axis_range (k : ℝ) :
  is_ellipse k ∧ foci_on_y_axis k ↔ k ∈ Set.Ioo 5 7.5 :=
sorry

end ellipse_with_foci_on_y_axis_range_l1799_179927


namespace rectangular_plot_breadth_l1799_179958

/-- 
Given a rectangular plot where:
- The area is 18 times the breadth
- The length is 10 meters more than the breadth
Prove that the breadth is 8 meters
-/
theorem rectangular_plot_breadth (b : ℝ) (l : ℝ) (A : ℝ) : 
  A = 18 * b →
  l = b + 10 →
  A = l * b →
  b = 8 := by
  sorry

end rectangular_plot_breadth_l1799_179958


namespace koolaid_water_increase_factor_l1799_179976

/-- Proves that the water increase factor is 4 given the initial conditions and final percentage --/
theorem koolaid_water_increase_factor : 
  ∀ (initial_koolaid initial_water evaporated_water : ℚ)
    (final_percentage : ℚ),
  initial_koolaid = 2 →
  initial_water = 16 →
  evaporated_water = 4 →
  final_percentage = 4/100 →
  ∃ (increase_factor : ℚ),
    increase_factor = 4 ∧
    initial_koolaid / (initial_koolaid + (initial_water - evaporated_water) * increase_factor) = final_percentage :=
by
  sorry

end koolaid_water_increase_factor_l1799_179976


namespace green_balls_count_l1799_179987

theorem green_balls_count (total : ℕ) (red blue green : ℕ) : 
  red + blue + green = total →
  red = total / 3 →
  blue = (2 * total) / 7 →
  green = 2 * blue - 8 →
  green = 16 :=
by
  sorry

end green_balls_count_l1799_179987


namespace divisibility_by_twelve_l1799_179907

theorem divisibility_by_twelve (m : Nat) : m ≤ 9 → (365 * 10 + m) % 12 = 0 ↔ m = 0 := by sorry

end divisibility_by_twelve_l1799_179907


namespace intersection_with_complement_l1799_179972

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {1, 3} := by sorry

end intersection_with_complement_l1799_179972


namespace quadratic_with_rational_roots_has_even_coefficient_l1799_179952

theorem quadratic_with_rational_roots_has_even_coefficient
  (a b c : ℕ+) -- a, b, c are positive integers
  (h_rational_roots : ∃ (p q r s : ℤ), (p * r ≠ 0 ∧ q * s ≠ 0) ∧
    (a * (p * s)^2 + b * (p * s) * (q * r) + c * (q * r)^2 = 0)) :
  Even a ∨ Even b ∨ Even c :=
sorry

end quadratic_with_rational_roots_has_even_coefficient_l1799_179952


namespace good_games_count_l1799_179910

def games_from_friend : ℕ := 41
def games_from_garage_sale : ℕ := 14
def non_working_games : ℕ := 31

theorem good_games_count : 
  games_from_friend + games_from_garage_sale - non_working_games = 24 := by
  sorry

end good_games_count_l1799_179910


namespace sin_2alpha_equals_3_5_l1799_179948

theorem sin_2alpha_equals_3_5 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : 
  Real.sin (2*α) = 3/5 := by sorry

end sin_2alpha_equals_3_5_l1799_179948


namespace upper_limit_of_multiples_l1799_179914

def average_of_multiples_of_10 (n : ℕ) : ℚ :=
  (n * (10 + n)) / (2 * n)

theorem upper_limit_of_multiples (n : ℕ) :
  n ≥ 10 → average_of_multiples_of_10 n = 55 → n = 100 := by
  sorry

end upper_limit_of_multiples_l1799_179914


namespace simplify_fraction_product_l1799_179918

theorem simplify_fraction_product : 15 * (18 / 5) * (-42 / 45) = -50.4 := by
  sorry

end simplify_fraction_product_l1799_179918


namespace value_of_b_l1799_179978

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 := by
  sorry

end value_of_b_l1799_179978


namespace series_sum_theorem_l1799_179962

/-- The sum of the infinite series (2n+1)x^n from n=0 to infinity -/
noncomputable def S (x : ℝ) : ℝ := ∑' n, (2 * n + 1) * x^n

/-- Theorem stating that if S(x) = 16, then x = (4 - √2) / 4 -/
theorem series_sum_theorem (x : ℝ) (hx : S x = 16) : x = (4 - Real.sqrt 2) / 4 := by
  sorry

end series_sum_theorem_l1799_179962


namespace right_triangle_area_l1799_179980

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≤ b) (h5 : b < c) (h6 : a + b = 13) (h7 : a = 5) (h8 : c^2 = a^2 + b^2) :
  (1/2) * a * b = 20 :=
by sorry

end right_triangle_area_l1799_179980


namespace square_area_from_vertices_l1799_179902

/-- The area of a square with adjacent vertices at (0,3) and (3,-4) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (0, 3)
  let p2 : ℝ × ℝ := (3, -4)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 58 := by sorry

end square_area_from_vertices_l1799_179902


namespace maximize_fruit_yield_l1799_179945

/-- Maximizing fruit yield in an orchard --/
theorem maximize_fruit_yield (x : ℝ) :
  let initial_trees : ℝ := 100
  let initial_yield_per_tree : ℝ := 600
  let yield_decrease_per_tree : ℝ := 5
  let total_trees : ℝ := x + initial_trees
  let new_yield_per_tree : ℝ := initial_yield_per_tree - yield_decrease_per_tree * x
  let total_yield : ℝ := total_trees * new_yield_per_tree
  (∀ z : ℝ, total_yield ≥ (z + initial_trees) * (initial_yield_per_tree - yield_decrease_per_tree * z)) →
  x = 10 := by
sorry

end maximize_fruit_yield_l1799_179945


namespace height_to_ad_l1799_179912

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- AB length
  ab : ℝ
  -- BC length
  bc : ℝ
  -- Height dropped to CD
  height_cd : ℝ
  -- Parallelogram property
  is_parallelogram : ab > 0 ∧ bc > 0 ∧ height_cd > 0

/-- Theorem: In a parallelogram ABCD where AB = 6, BC = 8, and the height dropped to CD is 4,
    the height dropped to AD is 3 -/
theorem height_to_ad (p : Parallelogram) 
    (h_ab : p.ab = 6)
    (h_bc : p.bc = 8)
    (h_height_cd : p.height_cd = 4) :
  ∃ (height_ad : ℝ), height_ad = 3 ∧ p.ab * p.height_cd = p.bc * height_ad :=
by sorry

end height_to_ad_l1799_179912


namespace f_monotonicity_and_positivity_l1799_179933

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x - k * Real.log x

-- State the theorem
theorem f_monotonicity_and_positivity (k : ℝ) (h_k : k > 0) :
  (∀ x > k, ∀ y > k, x < y → f k x < f k y) ∧ 
  (∀ x ∈ Set.Ioo 0 k, ∀ y ∈ Set.Ioo 0 k, x < y → f k x > f k y) ∧
  (∀ x ≥ 1, f k x > 0) → 
  0 < k ∧ k < Real.exp 1 := by
  sorry

end f_monotonicity_and_positivity_l1799_179933


namespace quinn_free_donuts_l1799_179989

/-- The number of free donuts Quinn is eligible for based on his summer reading --/
def free_donuts (books_per_donut : ℕ) (books_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  (books_per_week * num_weeks) / books_per_donut

/-- Theorem stating that Quinn is eligible for 4 free donuts --/
theorem quinn_free_donuts :
  free_donuts 5 2 10 = 4 := by
  sorry

end quinn_free_donuts_l1799_179989


namespace square_sum_reciprocal_l1799_179946

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end square_sum_reciprocal_l1799_179946


namespace exists_quadratic_with_2n_roots_l1799_179992

/-- Definition of function iteration -/
def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- A quadratic polynomial -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating the existence of a quadratic polynomial with the desired property -/
theorem exists_quadratic_with_2n_roots :
  ∃ (a b c : ℝ), ∀ (n : ℕ), n > 0 →
    (∃ (roots : Finset ℝ), roots.card = 2^n ∧
      (∀ x : ℝ, x ∈ roots ↔ iterate (quadratic a b c) n x = 0) ∧
      (∀ x y : ℝ, x ∈ roots → y ∈ roots → x ≠ y → x ≠ y)) :=
sorry

end exists_quadratic_with_2n_roots_l1799_179992


namespace num_divisors_2_pow_7_num_divisors_5_pow_4_num_divisors_2_pow_7_mul_5_pow_4_num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k_num_divisors_3600_num_divisors_42_pow_5_l1799_179965

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Theorem for 2^7
theorem num_divisors_2_pow_7 : num_divisors (2^7) = 8 := by sorry

-- Theorem for 5^4
theorem num_divisors_5_pow_4 : num_divisors (5^4) = 5 := by sorry

-- Theorem for 2^7 * 5^4
theorem num_divisors_2_pow_7_mul_5_pow_4 : num_divisors (2^7 * 5^4) = 40 := by sorry

-- Theorem for 2^m * 5^n * 3^k
theorem num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k (m n k : ℕ) :
  num_divisors (2^m * 5^n * 3^k) = (m + 1) * (n + 1) * (k + 1) := by sorry

-- Theorem for 3600
theorem num_divisors_3600 : num_divisors 3600 = 45 := by sorry

-- Theorem for 42^5
theorem num_divisors_42_pow_5 : num_divisors (42^5) = 216 := by sorry

end num_divisors_2_pow_7_num_divisors_5_pow_4_num_divisors_2_pow_7_mul_5_pow_4_num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k_num_divisors_3600_num_divisors_42_pow_5_l1799_179965


namespace conic_is_parabola_l1799_179925

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + y^2)

-- Define what it means for an equation to describe a parabola
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ) (h : a ≠ 0), 
    ∀ x y, f x y ↔ y = a * x^2 + b * x + c ∨ x = a * y^2 + b * y + d

-- Theorem statement
theorem conic_is_parabola : is_parabola conic_equation :=
sorry

end conic_is_parabola_l1799_179925


namespace police_force_competition_l1799_179995

theorem police_force_competition (x y : ℕ) : 
  (70 * x + 60 * y = 740) → 
  ((x = 8 ∧ y = 3) ∨ (x = 2 ∧ y = 10)) := by
sorry

end police_force_competition_l1799_179995


namespace union_of_sets_l1799_179900

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {1, 3, 5}
  A ∪ B = {1, 2, 3, 5} := by
  sorry

end union_of_sets_l1799_179900


namespace units_digit_sum_base8_l1799_179999

/-- The units digit of a number in a given base -/
def unitsDigit (n : ℕ) (base : ℕ) : ℕ := n % base

/-- Addition in a given base -/
def addInBase (a b base : ℕ) : ℕ := (a + b) % base

theorem units_digit_sum_base8 :
  unitsDigit (addInBase 45 37 8) 8 = 4 := by sorry

end units_digit_sum_base8_l1799_179999


namespace books_per_shelf_l1799_179979

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 315) (h2 : num_shelves = 7) : 
  total_books / num_shelves = 45 := by
  sorry

end books_per_shelf_l1799_179979
