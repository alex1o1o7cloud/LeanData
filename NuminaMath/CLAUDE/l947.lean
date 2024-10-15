import Mathlib

namespace NUMINAMATH_CALUDE_chair_cost_l947_94763

def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

theorem chair_cost (chair_cost : ℕ) 
  (h1 : chair_cost * num_chairs + table_cost = total_spent) 
  (h2 : chair_cost > 0) : chair_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l947_94763


namespace NUMINAMATH_CALUDE_smallest_n_l947_94758

/-- Given a positive integer k, N is the smallest positive integer such that
    there exists a set of 2k + 1 distinct positive integers whose sum is greater than N,
    but the sum of any k-element subset is at most N/2 -/
theorem smallest_n (k : ℕ+) : ∃ (N : ℕ),
  N = 2 * k.val^3 + 3 * k.val^2 + 3 * k.val ∧
  (∃ (S : Finset ℕ),
    S.card = 2 * k.val + 1 ∧
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y) ∧
    (S.sum id > N) ∧
    (∀ (T : Finset ℕ), T ⊆ S → T.card = k.val → T.sum id ≤ N / 2)) ∧
  (∀ (M : ℕ), M < N →
    ¬∃ (S : Finset ℕ),
      S.card = 2 * k.val + 1 ∧
      (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y) ∧
      (S.sum id > M) ∧
      (∀ (T : Finset ℕ), T ⊆ S → T.card = k.val → T.sum id ≤ M / 2)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_n_l947_94758


namespace NUMINAMATH_CALUDE_max_value_of_a_l947_94775

theorem max_value_of_a (a b c : ℝ) : 
  a^2 - b*c - 8*a + 7 = 0 → 
  b^2 + c^2 + b*c - 6*a + 6 = 0 → 
  a ≤ 9 ∧ ∃ b c : ℝ, a^2 - b*c - 8*a + 7 = 0 ∧ b^2 + c^2 + b*c - 6*a + 6 = 0 ∧ a = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l947_94775


namespace NUMINAMATH_CALUDE_triangle_existence_and_area_l947_94727

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 6 ∧
  Real.sin t.B ^ 2 + Real.sin t.C ^ 2 = Real.sin t.A ^ 2 + (2 * Real.sqrt 3 / 3) * Real.sin t.A * Real.sin t.B * Real.sin t.C

-- Define the theorem
theorem triangle_existence_and_area (t : Triangle) :
  triangle_conditions t → t.b + t.c = 2 * Real.sqrt 3 →
  ∃ (area : ℝ), area = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_and_area_l947_94727


namespace NUMINAMATH_CALUDE_problem_statement_l947_94782

theorem problem_statement (m n : ℝ) (a b : ℝ) 
  (h1 : m + n = 9)
  (h2 : 0 < a ∧ 0 < b)
  (h3 : a^2 + b^2 = 9) : 
  (∀ x : ℝ, |x - m| + |x + n| ≥ 9) ∧ 
  (a + b) * (a^3 + b^3) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l947_94782


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l947_94769

theorem sum_of_arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 107 →
  d = 10 →
  aₙ = 447 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℝ) / 2 * (a₁ + aₙ) = 9695 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l947_94769


namespace NUMINAMATH_CALUDE_fence_repair_boards_count_l947_94728

/-- Represents the number of boards nailed with a specific number of nails -/
structure BoardCount where
  count : ℕ
  nails_per_board : ℕ

/-- Represents a person's nailing work -/
structure NailingWork where
  first_type : BoardCount
  second_type : BoardCount

/-- Calculates the total number of nails used -/
def total_nails (work : NailingWork) : ℕ :=
  work.first_type.count * work.first_type.nails_per_board +
  work.second_type.count * work.second_type.nails_per_board

/-- Calculates the total number of boards nailed -/
def total_boards (work : NailingWork) : ℕ :=
  work.first_type.count + work.second_type.count

theorem fence_repair_boards_count :
  ∀ (petrov vasechkin : NailingWork),
    petrov.first_type.nails_per_board = 2 →
    petrov.second_type.nails_per_board = 3 →
    vasechkin.first_type.nails_per_board = 3 →
    vasechkin.second_type.nails_per_board = 5 →
    total_nails petrov = 87 →
    total_nails vasechkin = 94 →
    total_boards petrov = total_boards vasechkin →
    total_boards petrov = 30 :=
by sorry

end NUMINAMATH_CALUDE_fence_repair_boards_count_l947_94728


namespace NUMINAMATH_CALUDE_proposition_relationship_l947_94714

theorem proposition_relationship (a b : ℝ) :
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧
  (∃ a b : ℝ, a > 0 ∧ ¬(a > b ∧ a⁻¹ > b⁻¹)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l947_94714


namespace NUMINAMATH_CALUDE_min_participants_quiz_l947_94734

-- Define the number of correct answers for each question
def correct_q1 : ℕ := 90
def correct_q2 : ℕ := 50
def correct_q3 : ℕ := 40
def correct_q4 : ℕ := 20

-- Define the maximum number of questions a participant can answer correctly
def max_correct_per_participant : ℕ := 2

-- Define the total number of correct answers
def total_correct_answers : ℕ := correct_q1 + correct_q2 + correct_q3 + correct_q4

-- Theorem stating the minimum number of participants
theorem min_participants_quiz : 
  ∀ n : ℕ, 
  (n * max_correct_per_participant ≥ total_correct_answers) → 
  (∀ m : ℕ, m < n → m * max_correct_per_participant < total_correct_answers) → 
  n = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_participants_quiz_l947_94734


namespace NUMINAMATH_CALUDE_sara_pumpkins_l947_94722

/-- The number of pumpkins Sara has now -/
def pumpkins_left : ℕ := 20

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The initial number of pumpkins Sara grew -/
def initial_pumpkins : ℕ := pumpkins_left + pumpkins_eaten

theorem sara_pumpkins : initial_pumpkins = 43 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l947_94722


namespace NUMINAMATH_CALUDE_district_a_schools_l947_94787

/-- Represents the three types of schools in Veenapaniville -/
inductive SchoolType
  | Public
  | Parochial
  | PrivateIndependent

/-- Represents the three districts in Veenapaniville -/
inductive District
  | A
  | B
  | C

/-- The total number of high schools in Veenapaniville -/
def totalSchools : Nat := 50

/-- The number of public schools in Veenapaniville -/
def publicSchools : Nat := 25

/-- The number of parochial schools in Veenapaniville -/
def parochialSchools : Nat := 16

/-- The number of private independent schools in Veenapaniville -/
def privateIndependentSchools : Nat := 9

/-- The number of high schools in District B -/
def districtBSchools : Nat := 17

/-- The number of private independent schools in District B -/
def districtBPrivateIndependentSchools : Nat := 2

/-- Function to calculate the number of schools in District C -/
def districtCSchools : Nat := 3 * (min publicSchools (min parochialSchools privateIndependentSchools))

/-- Theorem stating that the number of high schools in District A is 6 -/
theorem district_a_schools :
  totalSchools - (districtBSchools + districtCSchools) = 6 := by
  sorry


end NUMINAMATH_CALUDE_district_a_schools_l947_94787


namespace NUMINAMATH_CALUDE_range_of_m_l947_94799

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) →
  (∀ x : ℝ, (2 / (x + 1) > 1) → x ≤ 1) →
  m ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l947_94799


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l947_94764

def vector_a : Fin 2 → ℝ := ![(-2), 3]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![3, m]

theorem perpendicular_vectors (m : ℝ) :
  (vector_a 0 * vector_b m 0 + vector_a 1 * vector_b m 1 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l947_94764


namespace NUMINAMATH_CALUDE_sum_base8_327_73_l947_94723

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent. -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits. -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimalToBase8 (n / 8)

/-- The sum of 327₈ and 73₈ in base 8 is equal to 422₈. -/
theorem sum_base8_327_73 :
  decimalToBase8 (base8ToDecimal [3, 2, 7] + base8ToDecimal [7, 3]) = [4, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_base8_327_73_l947_94723


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral_l947_94744

open Real

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π

/-- The sides of the triangle form an arithmetic sequence -/
def SidesArithmeticSequence (t : Triangle) : Prop :=
  2 / t.b = 1 / t.a + 1 / t.c

/-- The angles of the triangle form an arithmetic sequence -/
def AnglesArithmeticSequence (t : Triangle) : Prop :=
  2 * t.β = t.α + t.γ

/-- A triangle is equilateral if all its sides are equal -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral
  (t : Triangle)
  (h_sides : SidesArithmeticSequence t)
  (h_angles : AnglesArithmeticSequence t) :
  IsEquilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral_l947_94744


namespace NUMINAMATH_CALUDE_smallest_6digit_binary_palindrome_4digit_other_base_l947_94750

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Counts the number of digits in a number in a given base -/
def digitCount (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_6digit_binary_palindrome_4digit_other_base :
  ∀ n : ℕ,
  isPalindrome n 2 →
  digitCount n 2 = 6 →
  (∃ b : ℕ, b > 2 ∧ isPalindrome (baseConvert n 2 b) b ∧ digitCount (baseConvert n 2 b) b = 4) →
  n ≥ 33 := by sorry

end NUMINAMATH_CALUDE_smallest_6digit_binary_palindrome_4digit_other_base_l947_94750


namespace NUMINAMATH_CALUDE_spending_difference_l947_94770

/- Define the prices and quantities -/
def basketball_price : ℝ := 29
def basketball_quantity : ℕ := 10
def baseball_price : ℝ := 2.5
def baseball_quantity : ℕ := 14
def baseball_bat_price : ℝ := 18

/- Define the total spending for each coach -/
def coach_A_spending : ℝ := basketball_price * basketball_quantity
def coach_B_spending : ℝ := baseball_price * baseball_quantity + baseball_bat_price

/- Theorem statement -/
theorem spending_difference :
  coach_A_spending - coach_B_spending = 237 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_l947_94770


namespace NUMINAMATH_CALUDE_dislike_tv_and_books_l947_94766

theorem dislike_tv_and_books (total_population : ℕ) 
  (tv_dislike_percentage : ℚ) (book_dislike_percentage : ℚ) :
  total_population = 800 →
  tv_dislike_percentage = 25 / 100 →
  book_dislike_percentage = 15 / 100 →
  (tv_dislike_percentage * total_population : ℚ) * book_dislike_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_dislike_tv_and_books_l947_94766


namespace NUMINAMATH_CALUDE_timothy_read_300_pages_l947_94797

/-- The total number of pages Timothy read in a week -/
def total_pages_read : ℕ :=
  let monday_tuesday := 2 * 45
  let wednesday := 50
  let thursday_to_saturday := 3 * 40
  let sunday := 25 + 15
  monday_tuesday + wednesday + thursday_to_saturday + sunday

/-- Theorem stating that Timothy read 300 pages in total -/
theorem timothy_read_300_pages : total_pages_read = 300 := by
  sorry

end NUMINAMATH_CALUDE_timothy_read_300_pages_l947_94797


namespace NUMINAMATH_CALUDE_inequality_proof_l947_94796

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (16 * a^2 + 9) + Real.sqrt (16 * b^2 + 9) + Real.sqrt (16 * c^2 + 9) ≥ 3 + 4 * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l947_94796


namespace NUMINAMATH_CALUDE_shortest_distance_to_parabola_l947_94786

/-- The parabola defined by x = 2y^2 -/
def parabola (y : ℝ) : ℝ := 2 * y^2

/-- The point from which we measure the distance -/
def point : ℝ × ℝ := (8, 14)

/-- The shortest distance between the point and the parabola -/
def shortest_distance : ℝ := 26

/-- Theorem stating that the shortest distance between the point (8,14) and the parabola x = 2y^2 is 26 -/
theorem shortest_distance_to_parabola :
  ∃ (y : ℝ), 
    shortest_distance = 
      Real.sqrt ((parabola y - point.1)^2 + (y - point.2)^2) ∧
    ∀ (z : ℝ), 
      Real.sqrt ((parabola z - point.1)^2 + (z - point.2)^2) ≥ shortest_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_to_parabola_l947_94786


namespace NUMINAMATH_CALUDE_tub_capacity_l947_94738

/-- Calculates the capacity of a tub given specific filling conditions -/
theorem tub_capacity 
  (flow_rate : ℕ) 
  (escape_rate : ℕ) 
  (cycle_time : ℕ) 
  (total_time : ℕ) 
  (h1 : flow_rate = 12)
  (h2 : escape_rate = 1)
  (h3 : cycle_time = 2)
  (h4 : total_time = 24) :
  (total_time / cycle_time) * (flow_rate - escape_rate - escape_rate) = 120 :=
by sorry

end NUMINAMATH_CALUDE_tub_capacity_l947_94738


namespace NUMINAMATH_CALUDE_constant_speed_travel_time_l947_94718

/-- Given a constant speed, if a 120-mile trip takes 3 hours, then a 200-mile trip takes 5 hours. -/
theorem constant_speed_travel_time 
  (speed : ℝ) 
  (h₁ : speed > 0) 
  (h₂ : 120 / speed = 3) : 
  200 / speed = 5 := by
sorry

end NUMINAMATH_CALUDE_constant_speed_travel_time_l947_94718


namespace NUMINAMATH_CALUDE_stan_magician_payment_l947_94751

def magician_payment (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_per_week * num_weeks

theorem stan_magician_payment :
  magician_payment 60 3 7 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_stan_magician_payment_l947_94751


namespace NUMINAMATH_CALUDE_five_from_second_row_wading_l947_94795

/-- Represents the beach scenario with people in rows and some wading in the water -/
structure BeachScenario where
  initial_first_row : ℕ
  initial_second_row : ℕ
  third_row : ℕ
  first_row_wading : ℕ
  remaining_on_beach : ℕ

/-- Calculates the number of people from the second row who joined those wading in the water -/
def second_row_wading (scenario : BeachScenario) : ℕ :=
  scenario.initial_first_row + scenario.initial_second_row + scenario.third_row
  - scenario.first_row_wading - scenario.remaining_on_beach

/-- Theorem stating that 5 people from the second row joined those wading in the water -/
theorem five_from_second_row_wading (scenario : BeachScenario)
  (h1 : scenario.initial_first_row = 24)
  (h2 : scenario.initial_second_row = 20)
  (h3 : scenario.third_row = 18)
  (h4 : scenario.first_row_wading = 3)
  (h5 : scenario.remaining_on_beach = 54) :
  second_row_wading scenario = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_from_second_row_wading_l947_94795


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_equals_three_sqrt_two_l947_94730

theorem sqrt_three_times_sqrt_six_equals_three_sqrt_two :
  Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_equals_three_sqrt_two_l947_94730


namespace NUMINAMATH_CALUDE_composition_computation_stages_l947_94737

/-- A structure representing a linear function f(x) = px + q -/
structure LinearFunction where
  p : ℝ
  q : ℝ

/-- Computes the composition of n linear functions -/
def compose_linear_functions (fs : List LinearFunction) (x : ℝ) : ℝ := sorry

/-- Represents a computation stage -/
inductive Stage
  | init : Stage
  | next : Stage → Stage

/-- Counts the number of stages -/
def stage_count : Stage → Nat
  | Stage.init => 0
  | Stage.next s => stage_count s + 1

/-- Theorem stating that the composition can be computed in no more than 30 stages -/
theorem composition_computation_stages
  (fs : List LinearFunction)
  (h_length : fs.length = 1000)
  (x₀ : ℝ) :
  ∃ (s : Stage), stage_count s ≤ 30 ∧ compose_linear_functions fs x₀ = sorry :=
by sorry

end NUMINAMATH_CALUDE_composition_computation_stages_l947_94737


namespace NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l947_94743

/-- A letter lock with a given number of rings and letters per ring -/
structure LetterLock where
  num_rings : ℕ
  letters_per_ring : ℕ

/-- The number of distinct unsuccessful attempts for a given letter lock -/
def unsuccessfulAttempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem: For a letter lock with 3 rings and 6 letters per ring, 
    the number of distinct unsuccessful attempts is 215 -/
theorem letter_lock_unsuccessful_attempts :
  ∃ (lock : LetterLock), lock.num_rings = 3 ∧ lock.letters_per_ring = 6 ∧ 
  unsuccessfulAttempts lock = 215 := by
  sorry

end NUMINAMATH_CALUDE_letter_lock_unsuccessful_attempts_l947_94743


namespace NUMINAMATH_CALUDE_tamara_brownie_pans_l947_94736

def total_revenue : ℕ := 32
def brownie_price : ℕ := 2
def pieces_per_pan : ℕ := 8

theorem tamara_brownie_pans : 
  total_revenue / (brownie_price * pieces_per_pan) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tamara_brownie_pans_l947_94736


namespace NUMINAMATH_CALUDE_ellipse_max_value_l947_94785

theorem ellipse_max_value (x y : ℝ) :
  x^2 / 9 + y^2 = 1 → x + 3 * y ≤ 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l947_94785


namespace NUMINAMATH_CALUDE_equivalent_operations_l947_94755

theorem equivalent_operations (x : ℝ) : 
  (x * (5/6)) / (2/7) = x * (35/12) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operations_l947_94755


namespace NUMINAMATH_CALUDE_cubic_identities_l947_94762

/-- Prove algebraic identities for cubic expressions -/
theorem cubic_identities (x y : ℝ) : 
  ((x + y) * (x^2 - x*y + y^2) = x^3 + y^3) ∧
  ((x + 3) * (x^2 - 3*x + 9) = x^3 + 27) ∧
  ((x - 1) * (x^2 + x + 1) = x^3 - 1) ∧
  ((2*x - 3) * (4*x^2 + 6*x + 9) = 8*x^3 - 27) := by
  sorry


end NUMINAMATH_CALUDE_cubic_identities_l947_94762


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l947_94791

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l947_94791


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l947_94753

theorem geometric_sequence_product (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h2 : a 2 = 2) (h3 : a 6 = 8) : a 3 * a 4 * a 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l947_94753


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l947_94713

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 7 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l947_94713


namespace NUMINAMATH_CALUDE_baseball_average_hits_l947_94733

theorem baseball_average_hits (first_games : Nat) (first_avg : Nat) (remaining_games : Nat) (remaining_avg : Nat) : 
  first_games = 20 →
  first_avg = 2 →
  remaining_games = 10 →
  remaining_avg = 5 →
  let total_games := first_games + remaining_games
  let total_hits := first_games * first_avg + remaining_games * remaining_avg
  (total_hits : Rat) / total_games = 3 := by sorry

end NUMINAMATH_CALUDE_baseball_average_hits_l947_94733


namespace NUMINAMATH_CALUDE_unit_digit_15_power_100_l947_94726

theorem unit_digit_15_power_100 : (15 ^ 100) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_15_power_100_l947_94726


namespace NUMINAMATH_CALUDE_total_amount_distributed_l947_94740

/-- Given an equal distribution of money among 22 persons, where each person receives Rs 1950,
    prove that the total amount distributed is Rs 42900. -/
theorem total_amount_distributed (num_persons : ℕ) (amount_per_person : ℕ) 
  (h1 : num_persons = 22)
  (h2 : amount_per_person = 1950) : 
  num_persons * amount_per_person = 42900 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_distributed_l947_94740


namespace NUMINAMATH_CALUDE_watermelon_total_sold_l947_94773

def watermelon_problem (customers_one : Nat) (customers_three : Nat) (customers_two : Nat) : Nat :=
  customers_one * 1 + customers_three * 3 + customers_two * 2

theorem watermelon_total_sold :
  watermelon_problem 17 3 10 = 46 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_total_sold_l947_94773


namespace NUMINAMATH_CALUDE_direct_proportion_m_value_l947_94721

-- Define the function y as a direct proportion function
def is_direct_proportion (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → (m - 2) * x^(m^2 - 3) = k * x

-- Theorem statement
theorem direct_proportion_m_value :
  (∃ m : ℝ, is_direct_proportion m) → (∃ m : ℝ, is_direct_proportion m ∧ m = -2) :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_m_value_l947_94721


namespace NUMINAMATH_CALUDE_scenic_spot_assignment_l947_94748

/-- The number of scenic spots -/
def num_spots : ℕ := 3

/-- The number of people -/
def num_people : ℕ := 4

/-- The total number of possible assignments without restrictions -/
def total_assignments : ℕ := num_spots ^ num_people

/-- The number of assignments where A and B are in the same spot -/
def restricted_assignments : ℕ := num_spots * (num_spots ^ (num_people - 2))

/-- The number of valid assignments where A and B are not in the same spot -/
def valid_assignments : ℕ := total_assignments - restricted_assignments

theorem scenic_spot_assignment :
  valid_assignments = 54 := by sorry

end NUMINAMATH_CALUDE_scenic_spot_assignment_l947_94748


namespace NUMINAMATH_CALUDE_oil_fraction_after_replacements_l947_94706

def tank_capacity : ℚ := 20
def replacement_amount : ℚ := 5
def num_replacements : ℕ := 5

def fraction_remaining (n : ℕ) : ℚ := (3/4) ^ n

theorem oil_fraction_after_replacements :
  fraction_remaining num_replacements = 243/1024 := by
  sorry

end NUMINAMATH_CALUDE_oil_fraction_after_replacements_l947_94706


namespace NUMINAMATH_CALUDE_total_distance_is_202_l947_94741

/-- Represents the driving data for a single day -/
structure DailyDrive where
  hours : Float
  speed : Float

/-- Calculates the distance traveled in a day given the driving data -/
def distanceTraveled (drive : DailyDrive) : Float :=
  drive.hours * drive.speed

/-- The week's driving schedule -/
def weekSchedule : List DailyDrive := [
  { hours := 3, speed := 12 },    -- Monday
  { hours := 3.5, speed := 8 },   -- Tuesday
  { hours := 2.5, speed := 12 },  -- Wednesday
  { hours := 4, speed := 6 },     -- Thursday
  { hours := 2, speed := 12 },    -- Friday
  { hours := 3, speed := 15 },    -- Saturday
  { hours := 1.5, speed := 10 }   -- Sunday
]

/-- Theorem: The total distance traveled during the week is 202 km -/
theorem total_distance_is_202 :
  (weekSchedule.map distanceTraveled).sum = 202 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_202_l947_94741


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l947_94704

theorem complex_modulus_problem : Complex.abs (Complex.I / (1 - Complex.I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l947_94704


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l947_94729

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (9 * x) = 2 * Real.sin (6 * x) * Real.cos (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l947_94729


namespace NUMINAMATH_CALUDE_last_digit_fibonacci_mod12_l947_94788

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def fibonacci_mod12 (n : ℕ) : ℕ := fibonacci n % 12

def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ fibonacci_mod12 k = d

theorem last_digit_fibonacci_mod12 :
  ∀ d : ℕ, d < 12 →
    (digit_appears d 21 → digit_appears 11 22) ∧
    (¬ digit_appears 11 21) ∧
    digit_appears 11 22 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_fibonacci_mod12_l947_94788


namespace NUMINAMATH_CALUDE_bowling_score_ratio_l947_94700

theorem bowling_score_ratio (total_score : ℕ) (third_score : ℕ) : 
  total_score = 810 →
  third_score = 162 →
  ∃ (first_score second_score : ℕ),
    first_score + second_score + third_score = total_score ∧
    first_score = second_score / 3 →
    second_score / third_score = 3 := by
sorry

end NUMINAMATH_CALUDE_bowling_score_ratio_l947_94700


namespace NUMINAMATH_CALUDE_envelope_difference_l947_94712

theorem envelope_difference (blue_envelopes : ℕ) (total_envelopes : ℕ) (yellow_envelopes : ℕ) :
  blue_envelopes = 10 →
  total_envelopes = 16 →
  yellow_envelopes < blue_envelopes →
  yellow_envelopes + blue_envelopes = total_envelopes →
  blue_envelopes - yellow_envelopes = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_envelope_difference_l947_94712


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l947_94767

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = x + y → a + 4 * b ≤ x + 4 * y ∧ 
  (a + 4 * b = 9 ↔ a = 3 ∧ b = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l947_94767


namespace NUMINAMATH_CALUDE_perpendicular_length_l947_94779

/-- A parallelogram with a diagonal of length 'd' and area 'a' has a perpendicular of length 'h' dropped on that diagonal. -/
structure Parallelogram where
  d : ℝ  -- length of the diagonal
  a : ℝ  -- area of the parallelogram
  h : ℝ  -- length of the perpendicular dropped on the diagonal

/-- The area of a parallelogram is equal to the product of its diagonal and the perpendicular dropped on that diagonal. -/
axiom area_formula (p : Parallelogram) : p.a = p.d * p.h

/-- For a parallelogram with a diagonal of 30 meters and an area of 600 square meters, 
    the length of the perpendicular dropped on the diagonal is 20 meters. -/
theorem perpendicular_length : 
  ∀ (p : Parallelogram), p.d = 30 → p.a = 600 → p.h = 20 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_length_l947_94779


namespace NUMINAMATH_CALUDE_prob_select_boy_is_half_prob_same_gender_is_third_l947_94703

/-- The number of students in the class -/
def total_students : ℕ := 4

/-- The number of boys in the class -/
def num_boys : ℕ := 2

/-- The number of girls in the class -/
def num_girls : ℕ := 2

/-- The probability of selecting a boy when one student is randomly selected -/
def prob_select_boy : ℚ := num_boys / total_students

/-- The probability of selecting two students of the same gender when two students are randomly selected -/
def prob_same_gender : ℚ := 1 / 3

theorem prob_select_boy_is_half : prob_select_boy = 1 / 2 :=
sorry

theorem prob_same_gender_is_third : prob_same_gender = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_prob_select_boy_is_half_prob_same_gender_is_third_l947_94703


namespace NUMINAMATH_CALUDE_unit_circle_trig_values_l947_94798

theorem unit_circle_trig_values :
  ∀ y : ℝ,
  ((-Real.sqrt 3 / 2) ^ 2 + y ^ 2 = 1) →
  ∃ θ : ℝ,
  (0 < θ ∧ θ < 2 * Real.pi) ∧
  (Real.sin θ = y ∧ Real.cos θ = -Real.sqrt 3 / 2) ∧
  (y = 1 / 2 ∨ y = -1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_trig_values_l947_94798


namespace NUMINAMATH_CALUDE_two_successes_in_four_trials_l947_94707

def probability_of_two_successes_in_four_trials (p : ℝ) : ℝ :=
  6 * p^2 * (1 - p)^2

theorem two_successes_in_four_trials :
  probability_of_two_successes_in_four_trials 0.6 = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_two_successes_in_four_trials_l947_94707


namespace NUMINAMATH_CALUDE_fluorescent_tubes_count_l947_94715

theorem fluorescent_tubes_count :
  ∀ (x y : ℕ),
  x + y = 13 →
  x / 3 + y / 2 = 5 →
  x = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_fluorescent_tubes_count_l947_94715


namespace NUMINAMATH_CALUDE_g_injective_on_restricted_domain_c_is_smallest_l947_94732

/-- The function g(x) = (x+3)^2 - 6 -/
def g (x : ℝ) : ℝ := (x + 3)^2 - 6

/-- c is the lower bound of the restricted domain -/
def c : ℝ := -3

theorem g_injective_on_restricted_domain :
  ∀ x y, x ≥ c → y ≥ c → g x = g y → x = y :=
sorry

theorem c_is_smallest :
  ∀ c' < c, ∃ x y, x ≥ c' ∧ y ≥ c' ∧ x ≠ y ∧ g x = g y :=
sorry

end NUMINAMATH_CALUDE_g_injective_on_restricted_domain_c_is_smallest_l947_94732


namespace NUMINAMATH_CALUDE_decimal_365_to_octal_l947_94739

/-- Converts a natural number to its octal representation as a list of digits -/
def toOctal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- Theorem: The decimal number 365 is equal to 555₈ in octal representation -/
theorem decimal_365_to_octal :
  toOctal 365 = [5, 5, 5] := by
  sorry

end NUMINAMATH_CALUDE_decimal_365_to_octal_l947_94739


namespace NUMINAMATH_CALUDE_triangle_heights_order_l947_94702

/-- Given a triangle with sides a, b, c and corresponding heights ha, hb, hc,
    if a > b > c, then ha < hb < hc -/
theorem triangle_heights_order (a b c ha hb hc : ℝ) :
  a > 0 → b > 0 → c > 0 →  -- positive sides
  ha > 0 → hb > 0 → hc > 0 →  -- positive heights
  a > b → b > c →  -- order of sides
  a * ha = b * hb →  -- area equality
  b * hb = c * hc →  -- area equality
  ha < hb ∧ hb < hc := by
  sorry


end NUMINAMATH_CALUDE_triangle_heights_order_l947_94702


namespace NUMINAMATH_CALUDE_expression_evaluation_l947_94742

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l947_94742


namespace NUMINAMATH_CALUDE_linear_equation_condition_l947_94772

theorem linear_equation_condition (m : ℝ) :
  (∃ x, (3*m - 1)*x + 9 = 0) ∧ (∀ x y, (3*m - 1)*x + 9 = 0 ∧ (3*m - 1)*y + 9 = 0 → x = y) →
  m ≠ 1/3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l947_94772


namespace NUMINAMATH_CALUDE_no_natural_number_power_of_two_l947_94771

theorem no_natural_number_power_of_two : 
  ∀ n : ℕ, ¬∃ k : ℕ, n^2012 - 1 = 2^k := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_power_of_two_l947_94771


namespace NUMINAMATH_CALUDE_cube_geometry_l947_94720

-- Define a cube
def Cube : Type := Unit

-- Define a vertex of a cube
def Vertex (c : Cube) : Type := Unit

-- Define a set of 4 vertices
def FourVertices (c : Cube) : Type := Fin 4 → Vertex c

-- Define a spatial quadrilateral
def SpatialQuadrilateral (c : Cube) (v : FourVertices c) : Prop := sorry

-- Define a tetrahedron
def Tetrahedron (c : Cube) (v : FourVertices c) : Prop := sorry

-- Define an equilateral triangle
def EquilateralTriangle (c : Cube) (v1 v2 v3 : Vertex c) : Prop := sorry

-- Define an isosceles right-angled triangle
def IsoscelesRightTriangle (c : Cube) (v1 v2 v3 : Vertex c) : Prop := sorry

-- Theorem statement
theorem cube_geometry (c : Cube) : 
  (∃ v : FourVertices c, SpatialQuadrilateral c v) ∧ 
  (∃ v : FourVertices c, Tetrahedron c v ∧ 
    (∀ face : Fin 4 → Fin 3, EquilateralTriangle c (v (face 0)) (v (face 1)) (v (face 2)))) ∧
  (∃ v : FourVertices c, Tetrahedron c v ∧ 
    (∃ face : Fin 4 → Fin 3, EquilateralTriangle c (v (face 0)) (v (face 1)) (v (face 2))) ∧
    (∃ faces : Fin 3 → (Fin 4 → Fin 3), 
      ∀ i : Fin 3, IsoscelesRightTriangle c (v ((faces i) 0)) (v ((faces i) 1)) (v ((faces i) 2)))) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_geometry_l947_94720


namespace NUMINAMATH_CALUDE_percentage_relation_l947_94765

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l947_94765


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l947_94708

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l947_94708


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l947_94709

theorem cubic_root_equation_solution :
  ∃ x : ℝ, x = 1674 / 15 ∧ (30 * x + (30 * x + 27) ^ (1/3)) ^ (1/3) = 15 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l947_94709


namespace NUMINAMATH_CALUDE_lucky_lila_coincidence_l947_94754

theorem lucky_lila_coincidence (a b c d e f : ℚ) : 
  a = 2 → b = 3 → c = 4 → d = 5 → f = 6 →
  (a * b * c * d * e / f = a * (b - (c * (d - (e / f))))) →
  e = -51 / 28 := by
sorry

end NUMINAMATH_CALUDE_lucky_lila_coincidence_l947_94754


namespace NUMINAMATH_CALUDE_max_t_for_exponential_sequence_range_a_for_quadratic_sequence_l947_94735

/-- Definition of property P(t) for a sequence -/
def has_property_P (a : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ m n : ℕ, m ≠ n → (a m - a n) / (m - n : ℝ) ≥ t

/-- Theorem for part (i) -/
theorem max_t_for_exponential_sequence :
  ∃ t_max : ℝ, (∀ t : ℝ, has_property_P (λ n => (2 : ℝ) ^ n) t → t ≤ t_max) ∧
            has_property_P (λ n => (2 : ℝ) ^ n) t_max :=
sorry

/-- Theorem for part (ii) -/
theorem range_a_for_quadratic_sequence :
  ∃ a_min : ℝ, (∀ a : ℝ, has_property_P (λ n => n^2 - a / n) 10 → a ≥ a_min) ∧
            has_property_P (λ n => n^2 - a_min / n) 10 :=
sorry

end NUMINAMATH_CALUDE_max_t_for_exponential_sequence_range_a_for_quadratic_sequence_l947_94735


namespace NUMINAMATH_CALUDE_cyclic_product_sum_theorem_l947_94756

/-- A permutation of (1, 2, 3, 4, 5, 6) -/
def Permutation := Fin 6 → Fin 6

/-- The cyclic product sum for a given permutation -/
def cyclicProductSum (p : Permutation) : ℕ :=
  (p 0) * (p 1) + (p 1) * (p 2) + (p 2) * (p 3) + (p 3) * (p 4) + (p 4) * (p 5) + (p 5) * (p 0)

/-- Predicate to check if a function is a valid permutation of (1, 2, 3, 4, 5, 6) -/
def isValidPermutation (p : Permutation) : Prop :=
  Function.Injective p ∧ Function.Surjective p

/-- The maximum value of the cyclic product sum -/
def M : ℕ := 79

/-- The number of permutations that achieve the maximum value -/
def N : ℕ := 12

theorem cyclic_product_sum_theorem :
  (∀ p : Permutation, isValidPermutation p → cyclicProductSum p ≤ M) ∧
  (∃! (s : Finset Permutation), s.card = N ∧ 
    ∀ p ∈ s, isValidPermutation p ∧ cyclicProductSum p = M) :=
sorry

end NUMINAMATH_CALUDE_cyclic_product_sum_theorem_l947_94756


namespace NUMINAMATH_CALUDE_no_finite_vector_set_with_equal_sums_property_l947_94717

theorem no_finite_vector_set_with_equal_sums_property (n : ℕ) :
  ¬ ∃ (S : Finset (ℝ × ℝ)),
    (S.card = n) ∧
    (∀ (a b : ℝ × ℝ), a ∈ S → b ∈ S → a ≠ b →
      ∃ (c d : ℝ × ℝ), c ∈ S ∧ d ∈ S ∧ c ≠ d ∧ c ≠ a ∧ c ≠ b ∧ d ≠ a ∧ d ≠ b ∧
        a.1 + b.1 = c.1 + d.1 ∧ a.2 + b.2 = c.2 + d.2) :=
by sorry

end NUMINAMATH_CALUDE_no_finite_vector_set_with_equal_sums_property_l947_94717


namespace NUMINAMATH_CALUDE_total_gas_cost_l947_94784

-- Define the given parameters
def miles_per_gallon : ℝ := 50
def miles_per_day : ℝ := 75
def price_per_gallon : ℝ := 3
def number_of_days : ℝ := 10

-- Define the theorem
theorem total_gas_cost : 
  (number_of_days * miles_per_day / miles_per_gallon) * price_per_gallon = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_total_gas_cost_l947_94784


namespace NUMINAMATH_CALUDE_coefficient_x_seven_l947_94724

theorem coefficient_x_seven (x : ℝ) :
  ∃ (a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ),
    (x + 1)^5 * (2*x - 1)^3 = a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀ ∧
    a₇ = 28 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_seven_l947_94724


namespace NUMINAMATH_CALUDE_sum_ages_theorem_l947_94776

/-- The sum of Josiah's and Hans' ages after 3 years, given their current ages -/
def sum_ages_after_3_years (hans_current_age : ℕ) (josiah_current_age : ℕ) : ℕ :=
  (hans_current_age + 3) + (josiah_current_age + 3)

/-- Theorem stating the sum of Josiah's and Hans' ages after 3 years -/
theorem sum_ages_theorem (hans_current_age : ℕ) (josiah_current_age : ℕ) 
  (h1 : hans_current_age = 15)
  (h2 : josiah_current_age = 3 * hans_current_age) :
  sum_ages_after_3_years hans_current_age josiah_current_age = 66 := by
sorry

end NUMINAMATH_CALUDE_sum_ages_theorem_l947_94776


namespace NUMINAMATH_CALUDE_two_smallest_solutions_l947_94716

def is_solution (k : ℕ) : Prop :=
  (Real.cos ((k^2 + 7^2 : ℝ) * Real.pi / 180))^2 = 1

def smallest_solutions : Prop :=
  (is_solution 31 ∧ is_solution 37) ∧
  ∀ k : ℕ, 0 < k ∧ k < 31 → ¬is_solution k

theorem two_smallest_solutions : smallest_solutions := by
  sorry

end NUMINAMATH_CALUDE_two_smallest_solutions_l947_94716


namespace NUMINAMATH_CALUDE_probability_theorem_l947_94783

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧
  is_odd (a * b) ∧ (is_prime a ∨ is_prime b)

def total_pairs : ℕ := Nat.choose 20 2

def valid_pairs : ℕ := 42

theorem probability_theorem : 
  (valid_pairs : ℚ) / total_pairs = 21 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l947_94783


namespace NUMINAMATH_CALUDE_terminating_decimal_of_7_over_200_l947_94731

theorem terminating_decimal_of_7_over_200 : 
  ∃ (n : ℕ) (d : ℕ+), (7 : ℚ) / 200 = (n : ℚ) / d ∧ (n : ℚ) / d = 0.028 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_of_7_over_200_l947_94731


namespace NUMINAMATH_CALUDE_domain_transformation_l947_94760

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Ioo (1/3) 1

-- Define the domain of f(3^x)
def domain_f_exp : Set ℝ := Set.Ico (-1) 0

-- Theorem statement
theorem domain_transformation (h : ∀ x ∈ domain_f, f x ≠ 0) :
  ∀ x, f (3^x) ≠ 0 ↔ x ∈ domain_f_exp :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l947_94760


namespace NUMINAMATH_CALUDE_expression_evaluation_l947_94781

theorem expression_evaluation :
  let x : ℚ := -2
  (1 - 1 / (1 - x)) / (x^2 / (x^2 - 1)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l947_94781


namespace NUMINAMATH_CALUDE_unique_counterexample_l947_94777

-- Define geometric figures
inductive GeometricFigure
| Line
| Plane

-- Define spatial relationships
def perpendicular (a b : GeometricFigure) : Prop := sorry
def parallel (a b : GeometricFigure) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricFigure) : Prop :=
  (perpendicular x y ∧ parallel y z) → perpendicular x z

-- Theorem statement
theorem unique_counterexample :
  ∀ x y z : GeometricFigure,
    ¬proposition x y z ↔ 
      x = GeometricFigure.Line ∧ 
      y = GeometricFigure.Line ∧ 
      z = GeometricFigure.Plane :=
sorry

end NUMINAMATH_CALUDE_unique_counterexample_l947_94777


namespace NUMINAMATH_CALUDE_complex_number_properties_l947_94792

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 3*m + 2) (m^2 - 1)

theorem complex_number_properties :
  (∀ m : ℝ, z m = 0 ↔ m = 1) ∧
  (∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2) ∧
  (∀ m : ℝ, (z m).re < 0 ∧ (z m).im > 0 ↔ 1 < m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l947_94792


namespace NUMINAMATH_CALUDE_min_oranges_in_new_box_l947_94768

theorem min_oranges_in_new_box (m n x : ℕ) : 
  m + n ≤ 60 →
  59 * m = 60 * n + x →
  x > 0 →
  (∀ y : ℕ, y < x → ¬(∃ m' n' : ℕ, m' + n' ≤ 60 ∧ 59 * m' = 60 * n' + y)) →
  x = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_oranges_in_new_box_l947_94768


namespace NUMINAMATH_CALUDE_functional_equation_solution_l947_94745

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The functional equation property -/
def SatisfiesEquation (f : RationalFunction) : Prop :=
  ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y

/-- The theorem statement -/
theorem functional_equation_solution :
  ∀ f : RationalFunction, SatisfiesEquation f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l947_94745


namespace NUMINAMATH_CALUDE_problem_solution_l947_94710

theorem problem_solution (x y : ℝ) (h : 3 * x - 4 * y = 5) :
  (y = (3 * x - 5) / 4) ∧
  (y ≤ x → x ≥ -5) ∧
  (∀ a : ℝ, x + 2 * y = a ∧ x > 2 * y → a < 10) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l947_94710


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l947_94789

noncomputable def polynomial_remainder 
  (p : ℝ → ℝ) (a b : ℝ) (h : a ≠ b) : ℝ × ℝ × ℝ :=
  let r := (p a - p b - (b - a) * (deriv p a)) / ((b - a) * (b + a))
  let d := deriv p a - 2 * r * a
  let e := p a - r * a^2 - d * a
  (r, d, e)

theorem polynomial_division_theorem 
  (p : ℝ → ℝ) (a b : ℝ) (h : a ≠ b) :
  ∃ (q : ℝ → ℝ) (r d e : ℝ),
    (∀ x, p x = q x * (x - a)^2 * (x - b) + r * x^2 + d * x + e) ∧
    (r, d, e) = polynomial_remainder p a b h :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l947_94789


namespace NUMINAMATH_CALUDE_orange_juice_cartons_bought_l947_94719

def prove_orange_juice_cartons : Nat :=
  let initial_money : Nat := 86
  let bread_loaves : Nat := 3
  let bread_cost : Nat := 3
  let juice_cost : Nat := 6
  let remaining_money : Nat := 59
  let spent_money : Nat := initial_money - remaining_money
  let bread_total_cost : Nat := bread_loaves * bread_cost
  let juice_total_cost : Nat := spent_money - bread_total_cost
  juice_total_cost / juice_cost

theorem orange_juice_cartons_bought :
  prove_orange_juice_cartons = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_cartons_bought_l947_94719


namespace NUMINAMATH_CALUDE_book_arrangement_count_l947_94759

theorem book_arrangement_count :
  let math_books : ℕ := 4
  let english_books : ℕ := 6
  let particular_english_book : ℕ := 1
  let math_block_arrangements : ℕ := Nat.factorial math_books
  let english_block_arrangements : ℕ := Nat.factorial (english_books - particular_english_book)
  let block_arrangements : ℕ := 1  -- Only one way to arrange the two blocks due to the particular book constraint
  block_arrangements * math_block_arrangements * english_block_arrangements = 2880
  := by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l947_94759


namespace NUMINAMATH_CALUDE_garden_fence_columns_l947_94794

theorem garden_fence_columns (S C : ℕ) : 
  S * C + (S - 1) / 2 = 1223 → 
  S = 2 * C + 5 → 
  C = 23 := by sorry

end NUMINAMATH_CALUDE_garden_fence_columns_l947_94794


namespace NUMINAMATH_CALUDE_intersection_theorem_l947_94701

/-- The curve C₁ in Cartesian coordinates -/
def C₁ (k : ℝ) : ℝ → ℝ := λ x ↦ k * |x| + 2

/-- The curve C₂ in Cartesian coordinates -/
def C₂ : ℝ × ℝ → Prop := λ p ↦ (p.1 + 1)^2 + p.2^2 = 4

/-- The number of intersection points between C₁ and C₂ -/
def numIntersections (k : ℝ) : ℕ := sorry

theorem intersection_theorem (k : ℝ) :
  numIntersections k = 3 → k = -4/3 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l947_94701


namespace NUMINAMATH_CALUDE_fold_square_crease_l947_94778

/-- Given a square ABCD with side length 18 cm, if point B is folded to point E on AD
    such that DE = 6 cm, and the resulting crease intersects AB at point F,
    then the length of FB is 13 cm. -/
theorem fold_square_crease (A B C D E F : ℝ × ℝ) : 
  -- Square ABCD with side length 18
  (A = (0, 0) ∧ B = (18, 0) ∧ C = (18, 18) ∧ D = (0, 18)) →
  -- E is on AD and DE = 6
  (E.1 = 0 ∧ E.2 = 12) →
  -- F is on AB
  (F.2 = 0) →
  -- F is on the perpendicular bisector of BE
  (F.2 - 6 = (3/2) * (F.1 - 9)) →
  -- The length of FB is 13
  Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 13 :=
by sorry

end NUMINAMATH_CALUDE_fold_square_crease_l947_94778


namespace NUMINAMATH_CALUDE_abs_diff_neg_self_l947_94790

theorem abs_diff_neg_self (m : ℝ) (h : m < 0) : |m - (-m)| = -2*m := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_neg_self_l947_94790


namespace NUMINAMATH_CALUDE_is_center_of_hyperbola_l947_94793

/-- The equation of the hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, 4)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y ↔ hyperbola_eq (x - hyperbola_center.1) (y - hyperbola_center.2) :=
sorry

end NUMINAMATH_CALUDE_is_center_of_hyperbola_l947_94793


namespace NUMINAMATH_CALUDE_last_five_digits_of_product_l947_94725

theorem last_five_digits_of_product : 
  (99 * 10101 * 111 * 1001001) % (100000 : ℕ) = 88889 := by
  sorry

end NUMINAMATH_CALUDE_last_five_digits_of_product_l947_94725


namespace NUMINAMATH_CALUDE_seungho_original_marble_difference_l947_94780

/-- Proves that Seungho originally had 1023 more marbles than Hyukjin -/
theorem seungho_original_marble_difference (s h : ℕ) : 
  s - 273 = (h + 273) + 477 → s = h + 1023 := by
  sorry

end NUMINAMATH_CALUDE_seungho_original_marble_difference_l947_94780


namespace NUMINAMATH_CALUDE_bags_bought_l947_94761

def crayonPacks : ℕ := 5
def crayonPrice : ℚ := 5
def bookCount : ℕ := 10
def bookPrice : ℚ := 5
def calculatorCount : ℕ := 3
def calculatorPrice : ℚ := 5
def bookDiscount : ℚ := 0.2
def salesTax : ℚ := 0.05
def initialMoney : ℚ := 200
def bagPrice : ℚ := 10

def totalCost : ℚ :=
  crayonPacks * crayonPrice +
  bookCount * bookPrice * (1 - bookDiscount) +
  calculatorCount * calculatorPrice

def finalCost : ℚ := totalCost * (1 + salesTax)

def change : ℚ := initialMoney - finalCost

theorem bags_bought (h : change ≥ 0) : ⌊change / bagPrice⌋ = 11 := by
  sorry

#eval ⌊change / bagPrice⌋

end NUMINAMATH_CALUDE_bags_bought_l947_94761


namespace NUMINAMATH_CALUDE_betty_beads_l947_94705

theorem betty_beads (red blue green : ℕ) : 
  (5 * blue = 3 * red) →
  (5 * green = 2 * red) →
  (red = 50) →
  (blue + green = 50) := by
sorry

end NUMINAMATH_CALUDE_betty_beads_l947_94705


namespace NUMINAMATH_CALUDE_sum_of_three_reals_l947_94752

theorem sum_of_three_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + 2*(y-1)*(z-1) = 85)
  (eq2 : y^2 + 2*(z-1)*(x-1) = 84)
  (eq3 : z^2 + 2*(x-1)*(y-1) = 89) :
  x + y + z = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_reals_l947_94752


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l947_94747

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to arrange n crates with 3 possible orientations each -/
def totalArrangements (n : ℕ) : ℕ :=
  3^n

/-- Calculates the number of ways to arrange crates to reach a specific height -/
def validArrangements (n : ℕ) (target_height : ℕ) : ℕ :=
  sorry  -- Placeholder for the actual calculation

/-- The probability of achieving the target height -/
def probability (n : ℕ) (target_height : ℕ) : ℚ :=
  (validArrangements n target_height : ℚ) / (totalArrangements n : ℚ)

theorem crate_stacking_probability :
  let crate_dims : CrateDimensions := ⟨3, 5, 7⟩
  let num_crates : ℕ := 10
  let target_height : ℕ := 43
  probability num_crates target_height = 10 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l947_94747


namespace NUMINAMATH_CALUDE_age_difference_l947_94757

theorem age_difference (li_age zhang_age jung_age : ℕ) : 
  li_age = 12 →
  zhang_age = 2 * li_age →
  jung_age = 26 →
  jung_age - zhang_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l947_94757


namespace NUMINAMATH_CALUDE_indira_cricket_time_l947_94746

def total_minutes : ℕ := 15000

def amaya_cricket_minutes : ℕ := 75 * 5 * 4
def amaya_basketball_minutes : ℕ := 45 * 2 * 4
def sean_cricket_minutes : ℕ := 65 * 14
def sean_basketball_minutes : ℕ := 55 * 4

def amaya_sean_total : ℕ := amaya_cricket_minutes + amaya_basketball_minutes + sean_cricket_minutes + sean_basketball_minutes

theorem indira_cricket_time :
  total_minutes - amaya_sean_total = 12010 :=
sorry

end NUMINAMATH_CALUDE_indira_cricket_time_l947_94746


namespace NUMINAMATH_CALUDE_ellipse_max_b_l947_94749

/-- Given an ellipse x^2 + y^2/b^2 = 1 where 0 < b < 1, with foci F1 and F2 at distance 2c apart,
    if there exists a point P on the ellipse such that the distance from P to the line x = 1/c
    is the arithmetic mean of |PF1| and |PF2|, then the maximum value of b is √3/2. -/
theorem ellipse_max_b (b c : ℝ) (h1 : 0 < b) (h2 : b < 1) :
  (∃ (x y : ℝ), x^2 + y^2/b^2 = 1 ∧
    ∃ (PF1 PF2 : ℝ), |x - 1/c| = (PF1 + PF2)/2 ∧
      ∃ (c_foci : ℝ), c_foci = 2*c) →
  b ≤ Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_b_l947_94749


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l947_94774

theorem rectangular_solid_volume (a b c : ℝ) 
  (h_top : a * b = 15)
  (h_front : b * c = 10)
  (h_side : c * a = 6) :
  a * b * c = 30 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l947_94774


namespace NUMINAMATH_CALUDE_exam_total_boys_l947_94711

theorem exam_total_boys (average_all : ℚ) (average_passed : ℚ) (average_failed : ℚ) 
  (passed_count : ℕ) : 
  average_all = 40 ∧ average_passed = 39 ∧ average_failed = 15 ∧ passed_count = 125 → 
  ∃ (total_count : ℕ), total_count = 120 ∧ 
    average_all * total_count = average_passed * passed_count + 
      average_failed * (total_count - passed_count) :=
by sorry

end NUMINAMATH_CALUDE_exam_total_boys_l947_94711
