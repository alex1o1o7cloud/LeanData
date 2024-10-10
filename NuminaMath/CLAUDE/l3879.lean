import Mathlib

namespace elizabeth_study_time_l3879_387935

/-- Given that Elizabeth studied for a total of 60 minutes, including 35 minutes for math,
    prove that she studied for 25 minutes for science. -/
theorem elizabeth_study_time (total_time math_time science_time : ℕ) : 
  total_time = 60 ∧ math_time = 35 ∧ total_time = math_time + science_time →
  science_time = 25 := by
sorry

end elizabeth_study_time_l3879_387935


namespace trigonometric_equation_solution_l3879_387902

theorem trigonometric_equation_solution (t : ℝ) : 
  (16 * Real.sin (t / 2) - 25 * Real.cos (t / 2) ≠ 0) →
  (40 * (Real.sin (t / 2) ^ 3 - Real.cos (t / 2) ^ 3) / 
   (16 * Real.sin (t / 2) - 25 * Real.cos (t / 2)) = Real.sin t) ↔
  (∃ k : ℤ, t = 2 * Real.arctan (4 / 5) + 2 * Real.pi * ↑k) := by
  sorry

end trigonometric_equation_solution_l3879_387902


namespace factorization_equality_l3879_387971

theorem factorization_equality (x : ℝ) : x * (x + 2) + (x + 2)^2 = 2 * (x + 2) * (x + 1) := by
  sorry

end factorization_equality_l3879_387971


namespace original_class_size_l3879_387964

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ N : ℕ,
    N * original_avg + new_students * new_avg = (N + new_students) * (original_avg - avg_decrease) ∧
    N = 12 :=
by sorry

end original_class_size_l3879_387964


namespace plane_equation_correct_l3879_387926

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- Checks if a point lies on a plane -/
def Plane.contains (p : Plane) (x y z : ℤ) : Prop :=
  p.A * x + p.B * y + p.C * z + p.D = 0

/-- Checks if a vector is perpendicular to another vector -/
def isPerpendicular (x1 y1 z1 x2 y2 z2 : ℤ) : Prop :=
  x1 * x2 + y1 * y2 + z1 * z2 = 0

theorem plane_equation_correct : ∃ (p : Plane),
  p.contains 10 (-2) 5 ∧
  isPerpendicular p.A p.B p.C 10 (-2) 5 ∧
  p.A = 10 ∧ p.B = -2 ∧ p.C = 5 ∧ p.D = -129 := by
  sorry

end plane_equation_correct_l3879_387926


namespace cookie_bags_problem_l3879_387961

/-- Given a total number of cookies and the number of cookies per bag,
    calculate the number of bags. -/
def number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem cookie_bags_problem :
  let total_cookies : ℕ := 14
  let cookies_per_bag : ℕ := 2
  number_of_bags total_cookies cookies_per_bag = 7 := by
  sorry


end cookie_bags_problem_l3879_387961


namespace sum_of_repeating_decimals_l3879_387945

/-- Represents a repeating decimal with a single repeating digit -/
def repeating_decimal_single (d : ℕ) : ℚ :=
  d / 9

/-- Represents a repeating decimal with two repeating digits -/
def repeating_decimal_double (d : ℕ) : ℚ :=
  d / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_single 2 + repeating_decimal_double 2 = 8 / 33 := by
  sorry

end sum_of_repeating_decimals_l3879_387945


namespace reactor_rearrangements_count_l3879_387946

/-- The number of distinguishable rearrangements of REACTOR with vowels at the end -/
def rearrangements_reactor : ℕ :=
  let consonants := 4  -- R, C, T, R
  let vowels := 3      -- E, A, O
  let consonant_arrangements := Nat.factorial consonants / Nat.factorial 2  -- 4! / 2! due to repeated R
  let vowel_arrangements := Nat.factorial vowels
  consonant_arrangements * vowel_arrangements

/-- Theorem stating that the number of rearrangements is 72 -/
theorem reactor_rearrangements_count :
  rearrangements_reactor = 72 := by
  sorry

#eval rearrangements_reactor  -- Should output 72

end reactor_rearrangements_count_l3879_387946


namespace unique_complementary_digit_l3879_387960

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem unique_complementary_digit (N : ℕ) : 
  ∃! d : ℕ, 0 < d ∧ d < 9 ∧ (sum_of_digits N + d) % 9 = 0 := by sorry

end unique_complementary_digit_l3879_387960


namespace equal_absolute_values_imply_b_equals_two_l3879_387982

theorem equal_absolute_values_imply_b_equals_two (b : ℝ) :
  (|1 - b| = |3 - b|) → b = 2 := by
  sorry

end equal_absolute_values_imply_b_equals_two_l3879_387982


namespace ellipse_line_intersection_dot_product_l3879_387908

/-- The ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- A line with inclination angle 45° passing through a focus of the ellipse -/
def Line (f : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = p.1 - f.1}

/-- The dot product of two points in ℝ² -/
def dotProduct (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

theorem ellipse_line_intersection_dot_product :
  ∀ f A B : ℝ × ℝ,
  f ∈ Ellipse →
  f.2 = 0 →
  A ∈ Ellipse →
  B ∈ Ellipse →
  A ∈ Line f →
  B ∈ Line f →
  A ≠ B →
  dotProduct A B = -1/3 := by
sorry

end ellipse_line_intersection_dot_product_l3879_387908


namespace second_quadrant_trig_identity_l3879_387938

/-- For any angle α in the second quadrant, (sin α / cos α) * √(1 / sin²α - 1) = -1 -/
theorem second_quadrant_trig_identity (α : Real) (h : π / 2 < α ∧ α < π) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / Real.sin α ^ 2 - 1) = -1 := by
  sorry

end second_quadrant_trig_identity_l3879_387938


namespace correct_investment_equation_l3879_387906

/-- Represents the investment scenario over two years -/
def investment_scenario (initial_investment : ℝ) (total_investment : ℝ) (growth_rate : ℝ) : Prop :=
  initial_investment * (1 + growth_rate) + initial_investment * (1 + growth_rate)^2 = total_investment

/-- Theorem stating that the given equation correctly represents the investment scenario -/
theorem correct_investment_equation :
  investment_scenario 2500 6600 x = true :=
by
  sorry

end correct_investment_equation_l3879_387906


namespace sick_children_count_l3879_387956

/-- Calculates the number of children who called in sick given the initial number of jellybeans,
    normal class size, jellybeans eaten per child, and jellybeans left. -/
def children_called_sick (initial_jellybeans : ℕ) (normal_class_size : ℕ) 
                         (jellybeans_per_child : ℕ) (jellybeans_left : ℕ) : ℕ :=
  normal_class_size - (initial_jellybeans - jellybeans_left) / jellybeans_per_child

theorem sick_children_count : 
  children_called_sick 100 24 3 34 = 2 := by
  sorry

end sick_children_count_l3879_387956


namespace system_solution_l3879_387991

theorem system_solution (x y : ℝ) : 
  (2 * x^2 - 7 * x * y - 4 * y^2 + 9 * x - 18 * y + 10 = 0 ∧ x^2 + 2 * y^2 = 6) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) ∨ (x = -22/9 ∧ y = -1/9)) :=
by sorry

end system_solution_l3879_387991


namespace last_passenger_probability_l3879_387929

/-- Represents a bus with n seats and n passengers -/
structure Bus (n : ℕ) where
  seats : Fin n → Passenger
  tickets : Fin n → Passenger

/-- Represents a passenger -/
inductive Passenger
| scientist
| regular (i : ℕ)

/-- The seating strategy for passengers -/
def seatingStrategy (b : Bus n) : Bus n := sorry

/-- The probability that the last passenger sits in their assigned seat -/
def lastPassengerProbability (n : ℕ) : ℚ :=
  if n < 2 then 0 else 1/2

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 for n ≥ 2 -/
theorem last_passenger_probability (n : ℕ) (h : n ≥ 2) :
  lastPassengerProbability n = 1/2 := by sorry

end last_passenger_probability_l3879_387929


namespace correct_scientific_notation_l3879_387922

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 300670

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.0067
    exponent := 5
    h1 := by sorry }

/-- Theorem stating that the proposed notation correctly represents the original number -/
theorem correct_scientific_notation :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by
  sorry

end correct_scientific_notation_l3879_387922


namespace imaginary_part_of_z_l3879_387919

theorem imaginary_part_of_z (m : ℝ) (z : ℂ) : 
  z = 1 - m * I ∧ z = -2 * I → z.im = -1 := by sorry

end imaginary_part_of_z_l3879_387919


namespace qin_jiushao_v3_equals_71_l3879_387900

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

-- Define Qin Jiushao's algorithm for calculating V₃
def qin_jiushao_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 23

-- Theorem statement
theorem qin_jiushao_v3_equals_71 :
  qin_jiushao_v3 2 = 71 :=
by sorry

end qin_jiushao_v3_equals_71_l3879_387900


namespace sqrt_31_between_5_and_6_l3879_387904

theorem sqrt_31_between_5_and_6 : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := by
  sorry

end sqrt_31_between_5_and_6_l3879_387904


namespace torn_sheets_count_l3879_387933

/-- Represents a book with numbered pages -/
structure Book where
  /-- The last page number in the book -/
  lastPage : ℕ

/-- Represents a range of torn out pages -/
structure TornPages where
  /-- The first torn out page number -/
  first : ℕ
  /-- The last torn out page number -/
  last : ℕ

/-- Check if two numbers have the same digits in any order -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Calculate the number of sheets torn out -/
def sheetsTornOut (book : Book) (torn : TornPages) : ℕ :=
  (torn.last - torn.first + 1) / 2

/-- The main theorem -/
theorem torn_sheets_count (book : Book) (torn : TornPages) :
  torn.first = 185 →
  sameDigits torn.first torn.last →
  torn.last % 2 = 0 →
  torn.first < torn.last →
  sheetsTornOut book torn = 167 := by sorry

end torn_sheets_count_l3879_387933


namespace tournament_games_l3879_387975

/-- The number of games played in a single-elimination tournament -/
def games_played (n : ℕ) : ℕ :=
  n - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are played -/
theorem tournament_games : games_played 32 = 31 := by
  sorry

end tournament_games_l3879_387975


namespace smallest_number_divisibility_l3879_387968

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 1572 → ¬(
    (m + 3) % 9 = 0 ∧ 
    (m + 3) % 35 = 0 ∧ 
    (m + 3) % 25 = 0 ∧ 
    (m + 3) % 21 = 0
  )) ∧
  (1572 + 3) % 9 = 0 ∧ 
  (1572 + 3) % 35 = 0 ∧ 
  (1572 + 3) % 25 = 0 ∧ 
  (1572 + 3) % 21 = 0 :=
by sorry

end smallest_number_divisibility_l3879_387968


namespace parallel_condition_l3879_387932

def a : ℝ × ℝ := (1, -4)
def b (x : ℝ) : ℝ × ℝ := (-1, x)
def c (x : ℝ) : ℝ × ℝ := a + 3 • (b x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = k • w

theorem parallel_condition (x : ℝ) :
  parallel a (c x) ↔ x = 4 := by sorry

end parallel_condition_l3879_387932


namespace right_triangle_altitude_relation_l3879_387901

theorem right_triangle_altitude_relation (a b c x : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0)
  (h5 : c^2 = a^2 + b^2)  -- Pythagorean theorem
  (h6 : a * b = c * x)    -- Area relation
  : 1 / x^2 = 1 / a^2 + 1 / b^2 := by
  sorry

#check right_triangle_altitude_relation

end right_triangle_altitude_relation_l3879_387901


namespace line_intercepts_sum_l3879_387918

/-- Given a line with equation y - 7 = -3(x + 2), 
    prove that the sum of its x-intercept and y-intercept is 4/3 -/
theorem line_intercepts_sum (x y : ℝ) :
  (y - 7 = -3 * (x + 2)) →
  ∃ (x_int y_int : ℝ),
    (x_int - 7 = -3 * (x_int + 2)) ∧  -- x-intercept condition
    (0 - 7 = -3 * (x_int + 2)) ∧      -- x-intercept definition
    (y_int - 7 = -3 * (0 + 2)) ∧      -- y-intercept condition
    (x_int + y_int = 4/3) :=
by sorry

end line_intercepts_sum_l3879_387918


namespace snowball_partition_l3879_387912

/-- A directed graph where each vertex has an out-degree of exactly 1 -/
structure SnowballGraph (V : Type) :=
  (edges : V → V)

/-- A partition of vertices into three sets -/
def ThreeTeamPartition (V : Type) := V → Fin 3

theorem snowball_partition {V : Type} (G : SnowballGraph V) :
  ∃ (partition : ThreeTeamPartition V),
    ∀ (v w : V), G.edges v = w → partition v ≠ partition w :=
sorry

end snowball_partition_l3879_387912


namespace inequality_proof_l3879_387990

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  x^2 + y^2 + z^2 + 2 * Real.sqrt (3 * x * y * z) ≤ 1 := by
  sorry

end inequality_proof_l3879_387990


namespace no_square_divisible_by_six_between_55_and_120_l3879_387951

theorem no_square_divisible_by_six_between_55_and_120 : ¬ ∃ x : ℕ, 
  (∃ n : ℕ, x = n ^ 2) ∧ 
  (x % 6 = 0) ∧ 
  (55 < x) ∧ 
  (x < 120) := by
sorry

end no_square_divisible_by_six_between_55_and_120_l3879_387951


namespace credit_card_balance_proof_l3879_387999

def calculate_final_balance (initial_balance : ℝ)
  (month1_interest : ℝ)
  (month2_charges : ℝ) (month2_interest : ℝ)
  (month3_charges : ℝ) (month3_payment : ℝ) (month3_interest : ℝ)
  (month4_charges : ℝ) (month4_payment : ℝ) (month4_interest : ℝ) : ℝ :=
  let balance1 := initial_balance * (1 + month1_interest)
  let balance2 := (balance1 + month2_charges) * (1 + month2_interest)
  let balance3 := ((balance2 + month3_charges) - month3_payment) * (1 + month3_interest)
  let balance4 := ((balance3 + month4_charges) - month4_payment) * (1 + month4_interest)
  balance4

theorem credit_card_balance_proof :
  calculate_final_balance 50 0.2 20 0.2 30 10 0.25 40 20 0.15 = 189.75 := by
  sorry

end credit_card_balance_proof_l3879_387999


namespace birthday_cards_l3879_387973

theorem birthday_cards (initial_cards total_cards : ℕ) 
  (h1 : initial_cards = 64)
  (h2 : total_cards = 82) :
  total_cards - initial_cards = 18 := by
  sorry

end birthday_cards_l3879_387973


namespace our_ellipse_equation_l3879_387977

-- Define the ellipse
structure Ellipse where
  f1 : ℝ × ℝ  -- Focus 1
  f2 : ℝ × ℝ  -- Focus 2
  min_dist : ℝ -- Shortest distance from a point on the ellipse to F₁

-- Define our specific ellipse
def our_ellipse : Ellipse :=
  { f1 := (0, -4)
  , f2 := (0, 4)
  , min_dist := 2
  }

-- Define the equation of an ellipse
def is_ellipse_equation (e : Ellipse) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x, y) ∈ {p : ℝ × ℝ | dist p e.f1 + dist p e.f2 = 2 * (e.f2.1 - e.f1.1)}

-- Theorem statement
theorem our_ellipse_equation :
  is_ellipse_equation our_ellipse (fun x y => x^2/20 + y^2/36 = 1) :=
sorry

end our_ellipse_equation_l3879_387977


namespace arithmetic_geometric_ratio_l3879_387998

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h₁ : d ≠ 0
  h₂ : ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: If a₁, a₄, and a₅ of an arithmetic sequence form a geometric sequence,
    then the common ratio of this geometric sequence is 1/3 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h : (seq.a 4) ^ 2 = (seq.a 1) * (seq.a 5)) :
  (seq.a 4) / (seq.a 1) = 1/3 := by
  sorry

end arithmetic_geometric_ratio_l3879_387998


namespace sequence_comparison_l3879_387995

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, b (n + 1) = b n * q

/-- All terms of the sequence are positive -/
def all_positive (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0

theorem sequence_comparison
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (hpos : all_positive b)
  (h1 : a 1 = b 1)
  (h11 : a 11 = b 11) :
  a 6 > b 6 := by
  sorry

end sequence_comparison_l3879_387995


namespace johns_remaining_money_l3879_387949

/-- Calculates the remaining money after John's purchases -/
def remaining_money (initial_amount : ℚ) : ℚ :=
  let after_snacks := initial_amount * (1 - 1/5)
  let after_necessities := after_snacks * (1 - 3/4)
  after_necessities * (1 - 1/4)

/-- Theorem stating that John's remaining money is $3 -/
theorem johns_remaining_money :
  remaining_money 20 = 3 := by sorry

end johns_remaining_money_l3879_387949


namespace olivias_initial_amount_l3879_387970

/-- The amount of money Olivia had in her wallet initially -/
def initial_amount : ℕ := sorry

/-- The amount of money Olivia spent at the supermarket -/
def amount_spent : ℕ := 25

/-- The amount of money Olivia had left after visiting the supermarket -/
def amount_left : ℕ := 29

/-- Theorem stating that Olivia's initial amount of money was $54 -/
theorem olivias_initial_amount : initial_amount = 54 := by sorry

end olivias_initial_amount_l3879_387970


namespace laptop_installment_calculation_l3879_387985

/-- Calculates the monthly installment amount for a laptop purchase --/
theorem laptop_installment_calculation (laptop_cost : ℝ) (down_payment_percentage : ℝ) 
  (additional_down_payment : ℝ) (balance_after_four_months : ℝ) 
  (h1 : laptop_cost = 1000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : additional_down_payment = 20)
  (h4 : balance_after_four_months = 520) : 
  ∃ (monthly_installment : ℝ), monthly_installment = 65 := by
  sorry

#check laptop_installment_calculation

end laptop_installment_calculation_l3879_387985


namespace arithmetic_sequence_sum_l3879_387978

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (a 3)^2 - 6*(a 3) + 10 = 0 →                      -- a₃ is a root of x² - 6x + 10 = 0
  (a 15)^2 - 6*(a 15) + 10 = 0 →                    -- a₁₅ is a root of x² - 6x + 10 = 0
  (∀ n, S n = (n/2) * (2*(a 1) + (n - 1)*(a 2 - a 1))) →  -- sum formula
  S 17 = 51 := by
sorry

end arithmetic_sequence_sum_l3879_387978


namespace subtraction_of_decimals_l3879_387924

theorem subtraction_of_decimals : 2.5 - 0.32 = 2.18 := by sorry

end subtraction_of_decimals_l3879_387924


namespace balloons_given_to_sandy_l3879_387997

def initial_red_balloons : ℕ := 31
def remaining_red_balloons : ℕ := 7

theorem balloons_given_to_sandy :
  initial_red_balloons - remaining_red_balloons = 24 :=
by sorry

end balloons_given_to_sandy_l3879_387997


namespace monkey_climb_l3879_387936

theorem monkey_climb (tree_height : ℝ) (climb_rate : ℝ) (total_time : ℕ) 
  (h1 : tree_height = 19)
  (h2 : climb_rate = 3)
  (h3 : total_time = 17) :
  ∃ (slip_back : ℝ), 
    slip_back = 2 ∧ 
    (total_time - 1 : ℝ) * (climb_rate - slip_back) + climb_rate = tree_height :=
by sorry

end monkey_climb_l3879_387936


namespace multiplication_value_proof_l3879_387920

theorem multiplication_value_proof (n r : ℚ) (hn : n = 9) (hr : r = 18) :
  ∃ x : ℚ, (n / 6) * x = r ∧ x = 12 := by
  sorry

end multiplication_value_proof_l3879_387920


namespace inequality_proof_l3879_387913

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by
  sorry

end inequality_proof_l3879_387913


namespace pythagorean_preservation_l3879_387994

theorem pythagorean_preservation (a b c α β γ : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : α^2 + β^2 - γ^2 = 2)
  (s := a * α + b * β - c * γ)
  (p := a - α * s)
  (q := b - β * s)
  (r := c - γ * s) :
  p^2 + q^2 = r^2 := by
sorry

end pythagorean_preservation_l3879_387994


namespace quadratic_equation_solution_sum_l3879_387959

theorem quadratic_equation_solution_sum : ∀ c d : ℝ,
  (c^2 - 6*c + 15 = 27) →
  (d^2 - 6*d + 15 = 27) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end quadratic_equation_solution_sum_l3879_387959


namespace functional_equation_solution_l3879_387942

/-- A function g: ℝ → ℝ satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g x - y) = g x + g (g y - g (-x)) + 2 * x

/-- Theorem stating that any function satisfying the functional equation must be g(x) = -2x -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = -2 * x := by
  sorry

end functional_equation_solution_l3879_387942


namespace prob_king_ace_standard_deck_l3879_387939

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (kings : ℕ)
  (aces : ℕ)

/-- Calculates the probability of drawing a King first and an Ace second without replacement -/
def prob_king_ace (d : Deck) : ℚ :=
  (d.kings : ℚ) / d.total_cards * d.aces / (d.total_cards - 1)

/-- Theorem: The probability of drawing a King first and an Ace second from a standard deck is 4/663 -/
theorem prob_king_ace_standard_deck :
  let standard_deck : Deck := ⟨52, 4, 4⟩
  prob_king_ace standard_deck = 4 / 663 := by
sorry

end prob_king_ace_standard_deck_l3879_387939


namespace coloring_book_shelves_l3879_387907

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 27 → books_sold = 6 → books_per_shelf = 7 → 
  (initial_stock - books_sold) / books_per_shelf = 3 := by
sorry

end coloring_book_shelves_l3879_387907


namespace cubic_trinomial_condition_l3879_387928

/-- 
Given a polynomial of the form 3xy^(|m|) - (1/4)(m-2)xy + 1,
prove that for it to be a cubic trinomial, m must equal -2.
-/
theorem cubic_trinomial_condition (m : ℤ) : 
  (abs m = 2) ∧ ((1/4 : ℚ) * (m - 2) ≠ 0) → m = -2 := by sorry

end cubic_trinomial_condition_l3879_387928


namespace system_solution_l3879_387931

theorem system_solution : 
  let x : ℚ := -24/13
  let y : ℚ := 18/13
  let z : ℚ := -23/13
  (3*x + 2*y = z - 1) ∧ 
  (2*x - y = 4*z + 2) ∧ 
  (x + 4*y = 3*z + 9) := by
sorry

end system_solution_l3879_387931


namespace jogging_distance_three_weeks_l3879_387934

/-- Calculates the total miles jogged over a given number of weeks -/
def total_miles_jogged (miles_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  miles_per_day * days_per_week * num_weeks

/-- Theorem: A person jogging 5 miles per day on weekdays for three weeks covers 75 miles -/
theorem jogging_distance_three_weeks :
  total_miles_jogged 5 5 3 = 75 := by
  sorry

end jogging_distance_three_weeks_l3879_387934


namespace sqrt_equation_solution_l3879_387981

theorem sqrt_equation_solution (t : ℝ) : 
  (Real.sqrt (49 - (t - 3)^2) - 7 = 0) ↔ (t = 3) :=
by sorry

end sqrt_equation_solution_l3879_387981


namespace system_of_equations_l3879_387947

theorem system_of_equations (x y A : ℝ) : 
  2 * x + y = A → 
  x + 2 * y = 8 → 
  (x + y) / 3 = 1.6666666666666667 → 
  A = 7 := by
sorry

end system_of_equations_l3879_387947


namespace division_problem_l3879_387974

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 3086)
  (h2 : quotient = 36)
  (h3 : remainder = 26)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 85 := by
  sorry

end division_problem_l3879_387974


namespace set_problem_l3879_387953

theorem set_problem (U A B C : Finset ℕ) 
  (h_U : U.card = 300)
  (h_A : A.card = 80)
  (h_B : B.card = 70)
  (h_C : C.card = 60)
  (h_AB : (A ∩ B).card = 30)
  (h_AC : (A ∩ C).card = 25)
  (h_BC : (B ∩ C).card = 20)
  (h_ABC : (A ∩ B ∩ C).card = 15)
  (h_outside : (U \ (A ∪ B ∪ C)).card = 65)
  (h_subset : A ∪ B ∪ C ⊆ U) :
  (A \ (B ∪ C)).card = 40 := by
sorry

end set_problem_l3879_387953


namespace xray_cost_correct_l3879_387957

/-- The cost of an x-ray, given the conditions of the problem -/
def xray_cost : ℝ := 250

/-- The cost of an MRI, given that it's triple the x-ray cost -/
def mri_cost : ℝ := 3 * xray_cost

/-- The total cost of both procedures -/
def total_cost : ℝ := xray_cost + mri_cost

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.8

/-- The amount paid by the patient -/
def patient_payment : ℝ := 200

/-- Theorem stating that the x-ray cost is correct given the problem conditions -/
theorem xray_cost_correct : 
  mri_cost = 3 * xray_cost ∧ 
  (1 - insurance_coverage) * total_cost = patient_payment ∧
  xray_cost = 250 := by
  sorry

end xray_cost_correct_l3879_387957


namespace shape_partition_count_l3879_387987

/-- Represents a cell in the shape -/
structure Cell :=
  (x : ℕ) (y : ℕ)

/-- Represents a rectangle in the partition -/
inductive Rectangle
  | small : Cell → Rectangle  -- 1×1 square
  | large : Cell → Cell → Rectangle  -- 1×2 rectangle

/-- A partition of the shape -/
def Partition := List Rectangle

/-- The shape with 17 cells -/
def shape : List Cell := sorry

/-- Check if a partition is valid for the given shape -/
def is_valid_partition (p : Partition) (s : List Cell) : Prop := sorry

/-- Count the number of distinct valid partitions -/
def count_valid_partitions (s : List Cell) : ℕ := sorry

/-- The main theorem -/
theorem shape_partition_count :
  count_valid_partitions shape = 10 := by sorry

end shape_partition_count_l3879_387987


namespace unique_solution_trigonometric_equation_l3879_387925

theorem unique_solution_trigonometric_equation :
  ∃! (n k m : ℕ), 1 ≤ n ∧ n ≤ 5 ∧
                  1 ≤ k ∧ k ≤ 5 ∧
                  1 ≤ m ∧ m ≤ 5 ∧
                  (Real.sin (π * n / 12) * Real.sin (π * k / 12) * Real.sin (π * m / 12) = 1 / 8) :=
by sorry

end unique_solution_trigonometric_equation_l3879_387925


namespace share_yield_calculation_l3879_387909

/-- Calculates the effective interest rate (yield) for a share --/
theorem share_yield_calculation (face_value : ℝ) (dividend_rate : ℝ) (market_value : ℝ) :
  face_value = 60 ∧ dividend_rate = 0.09 ∧ market_value = 45 →
  (face_value * dividend_rate) / market_value = 0.12 := by
  sorry

end share_yield_calculation_l3879_387909


namespace leak_emptying_time_l3879_387992

theorem leak_emptying_time (tank_capacity : ℝ) (inlet_rate : ℝ) (emptying_time_with_inlet : ℝ) :
  tank_capacity = 6048 →
  inlet_rate = 6 →
  emptying_time_with_inlet = 12 →
  (tank_capacity / (tank_capacity / emptying_time_with_inlet + inlet_rate * 60)) = 7 := by
  sorry

end leak_emptying_time_l3879_387992


namespace unique_prime_solution_l3879_387944

theorem unique_prime_solution :
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ∧
    p = 5 ∧ q = 3 ∧ r = 19 := by
  sorry

end unique_prime_solution_l3879_387944


namespace lucky_larry_coincidence_l3879_387940

theorem lucky_larry_coincidence : ∃ e : ℝ, 
  let a : ℝ := 5
  let b : ℝ := 3
  let c : ℝ := 4
  let d : ℝ := 2
  (a + b - c + d - e) = (a + (b - (c + (d - e)))) := by
  sorry

end lucky_larry_coincidence_l3879_387940


namespace min_draws_for_red_specific_l3879_387966

/-- Given a bag with red, white, and black balls, we define the minimum number of draws
    required to guarantee drawing a red ball. -/
def min_draws_for_red (red white black : ℕ) : ℕ :=
  white + black + 1

/-- Theorem stating that for a bag with 10 red, 8 white, and 7 black balls,
    the minimum number of draws to guarantee a red ball is 16. -/
theorem min_draws_for_red_specific : min_draws_for_red 10 8 7 = 16 := by
  sorry

end min_draws_for_red_specific_l3879_387966


namespace moles_of_ch4_combined_l3879_387911

-- Define the chemical reaction
structure Reaction where
  ch4 : ℝ
  cl2 : ℝ
  ch3cl : ℝ
  hcl : ℝ

-- Define the stoichiometric coefficients
def stoichiometric_ratio : Reaction → Prop :=
  fun r => r.ch4 = r.cl2 ∧ r.ch4 = r.ch3cl ∧ r.ch4 = r.hcl

-- Define the theorem
theorem moles_of_ch4_combined 
  (r : Reaction) 
  (h1 : stoichiometric_ratio r) 
  (h2 : r.ch3cl = 2) 
  (h3 : r.cl2 = 2) : 
  r.ch4 = 2 := by
  sorry

end moles_of_ch4_combined_l3879_387911


namespace cost_reduction_proof_l3879_387903

theorem cost_reduction_proof (total_reduction : ℝ) (years : ℕ) (annual_reduction : ℝ) : 
  total_reduction = 0.36 ∧ years = 2 → 
  (1 - annual_reduction) ^ years = 1 - total_reduction →
  annual_reduction = 0.2 := by
sorry

end cost_reduction_proof_l3879_387903


namespace louisa_travel_l3879_387989

/-- Louisa's travel problem -/
theorem louisa_travel (average_speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  average_speed = 40 →
  second_day_distance = 280 →
  time_difference = 3 →
  let second_day_time := second_day_distance / average_speed
  let first_day_time := second_day_time - time_difference
  let first_day_distance := average_speed * first_day_time
  first_day_distance = 160 := by
  sorry

end louisa_travel_l3879_387989


namespace unique_quadratic_solution_l3879_387976

theorem unique_quadratic_solution (a c : ℤ) : 
  (∃! x : ℝ, a * x^2 - 6 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                        -- sum condition
  (a < c) →                             -- order condition
  (a = 3 ∧ c = 9) :=                    -- unique solution
by sorry

end unique_quadratic_solution_l3879_387976


namespace negation_equivalence_l3879_387943

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0) :=
by
  sorry

end negation_equivalence_l3879_387943


namespace largest_divisor_of_expression_l3879_387905

theorem largest_divisor_of_expression (n : ℕ+) : 
  ∃ (m : ℕ), m = 2448 ∧ 
  (∀ k : ℕ+, (9^(2*k.val) - 8^(2*k.val) - 17) % m = 0) ∧
  (∀ m' : ℕ, m' > m → ∃ k : ℕ+, (9^(2*k.val) - 8^(2*k.val) - 17) % m' ≠ 0) :=
sorry

end largest_divisor_of_expression_l3879_387905


namespace tangent_line_circle_l3879_387914

theorem tangent_line_circle (R : ℝ) : 
  R > 0 → 
  (∃ x y : ℝ, x + y = 2 * R ∧ (x - 1)^2 + y^2 = R ∧ 
    ∀ x' y' : ℝ, x' + y' = 2 * R → (x' - 1)^2 + y'^2 ≥ R) →
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 := by
sorry

end tangent_line_circle_l3879_387914


namespace arithmetic_sequence_length_arithmetic_sequence_has_671_terms_l3879_387923

/-- An arithmetic sequence starting at 2, with common difference 3, and last term 2014 -/
def ArithmeticSequence : ℕ → ℤ := fun n ↦ 2 + 3 * (n - 1)

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ArithmeticSequence n = 2014 ∧ ∀ m : ℕ, m > n → ArithmeticSequence m > 2014 :=
by
  sorry

theorem arithmetic_sequence_has_671_terms :
  ∃! n : ℕ, n > 0 ∧ ArithmeticSequence n = 2014 ∧ n = 671 :=
by
  sorry

end arithmetic_sequence_length_arithmetic_sequence_has_671_terms_l3879_387923


namespace total_amount_is_234_l3879_387993

/-- Represents the share of each person in rupees -/
structure Share where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The total amount distributed -/
def total_amount (s : Share) : ℝ := s.x + s.y + s.z

/-- The condition that y gets 45 paisa for each rupee x gets -/
def y_ratio (s : Share) : Prop := s.y = 0.45 * s.x

/-- The condition that z gets 50 paisa for each rupee x gets -/
def z_ratio (s : Share) : Prop := s.z = 0.50 * s.x

/-- The condition that y's share is 54 rupees -/
def y_share (s : Share) : Prop := s.y = 54

theorem total_amount_is_234 (s : Share) 
  (hy : y_ratio s) (hz : z_ratio s) (hy_share : y_share s) : 
  total_amount s = 234 := by
  sorry


end total_amount_is_234_l3879_387993


namespace fraction_problem_l3879_387955

theorem fraction_problem (x : ℚ) : 
  x^35 * (1/4)^18 = 1/(2*(10)^35) → x = 1/5 := by
sorry

end fraction_problem_l3879_387955


namespace negation_of_proposition_l3879_387952

theorem negation_of_proposition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end negation_of_proposition_l3879_387952


namespace negation_of_existence_negation_of_inequality_negation_of_proposition_l3879_387965

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x < 2, P x) ↔ (∀ x < 2, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) :
  ¬(x^2 - 2*x < 0) ↔ (x^2 - 2*x ≥ 0) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x < 2, x^2 - 2*x < 0) ↔ (∀ x < 2, x^2 - 2*x ≥ 0) := by sorry

end negation_of_existence_negation_of_inequality_negation_of_proposition_l3879_387965


namespace investment_income_is_648_l3879_387915

/-- Calculates the annual income from a stock investment given the total investment,
    share face value, quoted price, and dividend rate. -/
def annual_income (total_investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  let num_shares := total_investment / quoted_price
  let dividend_per_share := (dividend_rate / 100) * face_value
  num_shares * dividend_per_share

/-- Theorem stating that the annual income for the given investment scenario is 648. -/
theorem investment_income_is_648 :
  annual_income 4455 10 8.25 12 = 648 := by
  sorry

end investment_income_is_648_l3879_387915


namespace problem_solution_l3879_387950

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_solution_l3879_387950


namespace isosceles_triangle_base_length_l3879_387958

/-- An isosceles triangle with congruent sides of 8 cm and perimeter of 25 cm has a base length of 9 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 25 →
  base = 9 := by
sorry

end isosceles_triangle_base_length_l3879_387958


namespace max_constant_value_l3879_387984

theorem max_constant_value (c d : ℝ) : 
  (∃ (k : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) →
  (∃ (max_k : ℝ), ∀ (k : ℝ), (∃ (c d : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) → k ≤ max_k) →
  (∃ (max_k : ℝ), ∀ (k : ℝ), (∃ (c d : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) → k ≤ max_k) ∧
  (∃ (c d : ℝ), 5 * c + (d - 12)^2 = 235 ∧ c ≤ 47) :=
by sorry

end max_constant_value_l3879_387984


namespace smallest_cube_multiple_l3879_387937

theorem smallest_cube_multiple : 
  ∃ (x : ℕ+) (M : ℤ), 
    (1890 : ℤ) * (x : ℤ) = M^3 ∧ 
    (∀ (y : ℕ+) (N : ℤ), (1890 : ℤ) * (y : ℤ) = N^3 → x ≤ y) ∧
    x = 4900 := by
  sorry

end smallest_cube_multiple_l3879_387937


namespace inverse_mod_53_l3879_387969

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 31) : (36⁻¹ : ZMod 53) = 22 := by
  sorry

end inverse_mod_53_l3879_387969


namespace line_intersection_l3879_387948

theorem line_intersection : 
  let x : ℚ := 27/50
  let y : ℚ := -9/10
  let line1 (x : ℚ) : ℚ := -5/3 * x
  let line2 (x : ℚ) : ℚ := 15*x - 9
  (y = line1 x) ∧ (y = line2 x) :=
by sorry

end line_intersection_l3879_387948


namespace theater_ticket_price_l3879_387954

/-- Proves that the price of a balcony seat is $8 given the theater ticket sales conditions --/
theorem theater_ticket_price (total_tickets : ℕ) (total_revenue : ℕ) 
  (orchestra_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 360 →
  total_revenue = 3320 →
  orchestra_price = 12 →
  balcony_orchestra_diff = 140 →
  ∃ (balcony_price : ℕ), 
    balcony_price = 8 ∧
    balcony_price * (total_tickets / 2 + balcony_orchestra_diff / 2) + 
    orchestra_price * (total_tickets / 2 - balcony_orchestra_diff / 2) = total_revenue :=
by
  sorry

#check theater_ticket_price

end theater_ticket_price_l3879_387954


namespace senior_class_size_l3879_387963

/-- The number of students in the senior class at East High School -/
def total_students : ℕ := 400

/-- The proportion of students who play sports -/
def sports_proportion : ℚ := 52 / 100

/-- The proportion of sports-playing students who play soccer -/
def soccer_proportion : ℚ := 125 / 1000

/-- The number of students who play soccer -/
def soccer_players : ℕ := 26

theorem senior_class_size :
  (total_students : ℚ) * sports_proportion * soccer_proportion = soccer_players := by
  sorry

end senior_class_size_l3879_387963


namespace largest_divisor_of_expression_l3879_387980

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (10*x - 4) * (10*x) * (5*x + 15) = 1200 * k) ∧
  (∀ (m : ℤ), m > 1200 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (l : ℤ), (10*y - 4) * (10*y) * (5*y + 15) = m * l)) :=
by sorry

end largest_divisor_of_expression_l3879_387980


namespace min_surface_area_five_cubes_l3879_387962

/-- Represents a shape made of unit cubes -/
structure Shape :=
  (num_cubes : ℕ)
  (num_joins : ℕ)

/-- Calculates the surface area of a shape -/
def surface_area (s : Shape) : ℕ :=
  s.num_cubes * 6 - s.num_joins * 2

/-- Theorem: Among shapes with 5 unit cubes, the one with 5 joins has the smallest surface area -/
theorem min_surface_area_five_cubes (s : Shape) (h1 : s.num_cubes = 5) (h2 : s.num_joins ≤ 5) :
  surface_area s ≥ surface_area { num_cubes := 5, num_joins := 5 } :=
sorry

end min_surface_area_five_cubes_l3879_387962


namespace pie_distribution_problem_l3879_387910

theorem pie_distribution_problem :
  ∃! (p b a h : ℕ),
    p + b + a + h = 30 ∧
    b + p = a + h ∧
    p + a = 6 * (b + h) ∧
    h < p ∧ h < b ∧ h < a ∧
    h ≥ 1 ∧ p ≥ 1 ∧ b ≥ 1 ∧ a ≥ 1 := by
  sorry

end pie_distribution_problem_l3879_387910


namespace z_in_first_quadrant_l3879_387921

theorem z_in_first_quadrant : 
  ∀ z : ℂ, z / (1 + Complex.I) = 2 - Complex.I → 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 := by
  sorry

end z_in_first_quadrant_l3879_387921


namespace intersection_equals_interval_l3879_387986

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x + Real.log (1 - x)}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the half-open interval [0, 1)
def interval_zero_one : Set ℝ := {x | 0 ≤ x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_interval : M_intersect_N = interval_zero_one := by
  sorry

end intersection_equals_interval_l3879_387986


namespace arithmetic_equation_l3879_387996

theorem arithmetic_equation : 6 + 18 / 3 - 4^2 = -4 := by
  sorry

end arithmetic_equation_l3879_387996


namespace circle_area_tripled_l3879_387930

theorem circle_area_tripled (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 + 1)) := by
  sorry

end circle_area_tripled_l3879_387930


namespace men_who_left_job_l3879_387983

/-- Given information about tree cutting rates, prove the number of men who left the job -/
theorem men_who_left_job (initial_men : ℕ) (initial_trees : ℕ) (initial_hours : ℕ)
  (final_trees : ℕ) (final_hours : ℕ) (h1 : initial_men = 20)
  (h2 : initial_trees = 30) (h3 : initial_hours = 4)
  (h4 : final_trees = 36) (h5 : final_hours = 6) :
  ∃ (men_left : ℕ),
    men_left = 4 ∧
    (initial_trees : ℚ) / initial_hours / initial_men =
    (final_trees : ℚ) / final_hours / (initial_men - men_left) :=
by sorry

end men_who_left_job_l3879_387983


namespace joan_remaining_books_l3879_387988

/-- The number of books Joan has after selling some -/
def books_remaining (initial : ℕ) (sold : ℕ) : ℕ := initial - sold

/-- Theorem: Joan has 7 books remaining -/
theorem joan_remaining_books : books_remaining 33 26 = 7 := by
  sorry

end joan_remaining_books_l3879_387988


namespace reciprocal_comparison_reciprocal_comparison_with_condition_l3879_387967

theorem reciprocal_comparison :
  (-3/2 : ℚ) < (-2/3 : ℚ) ∧
  (2/3 : ℚ) < (3/2 : ℚ) ∧
  ¬((-1 : ℚ) < (-1 : ℚ)) ∧
  ¬((1 : ℚ) < (1 : ℚ)) ∧
  ¬((3 : ℚ) < (1/3 : ℚ)) :=
by
  sorry

-- Helper definition for the condition
def less_than_reciprocal (x : ℚ) : Prop :=
  x ≠ 0 ∧ x < 1 ∧ x < 1 / x

-- Theorem using the helper definition
theorem reciprocal_comparison_with_condition :
  less_than_reciprocal (-3/2) ∧
  less_than_reciprocal (2/3) ∧
  ¬less_than_reciprocal (-1) ∧
  ¬less_than_reciprocal 1 ∧
  ¬less_than_reciprocal 3 :=
by
  sorry

end reciprocal_comparison_reciprocal_comparison_with_condition_l3879_387967


namespace resulting_polygon_sides_l3879_387979

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon where
  sides : ℕ

/-- Represents the arrangement of regular polygons -/
structure PolygonArrangement where
  polygons : List RegularPolygon

/-- Calculates the number of exposed sides in the resulting polygon -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  sorry

/-- The specific arrangement of polygons in our problem -/
def ourArrangement : PolygonArrangement :=
  { polygons := [
      { sides := 5 },  -- pentagon
      { sides := 4 },  -- square
      { sides := 6 },  -- hexagon
      { sides := 7 },  -- heptagon
      { sides := 9 }   -- nonagon
    ] }

/-- Theorem stating that the resulting polygon has 23 sides -/
theorem resulting_polygon_sides : exposedSides ourArrangement = 23 :=
  sorry

end resulting_polygon_sides_l3879_387979


namespace tangent_line_equation_l3879_387916

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧
  y₀ = f x₀ ∧
  (Real.log x₀ + 1) * 0 - (-1) = (Real.log x₀ + 1) * x₀ - y₀ ∧
  ∀ (x y : ℝ), y = Real.log x₀ + 1 * (x - x₀) + y₀ ↔ x - y - 1 = 0 :=
sorry

end tangent_line_equation_l3879_387916


namespace easter_egg_count_l3879_387972

/-- The number of Easter eggs found in the club house -/
def club_house_eggs : ℕ := 40

/-- The number of Easter eggs found in the park -/
def park_eggs : ℕ := 25

/-- The number of Easter eggs found in the town hall -/
def town_hall_eggs : ℕ := 15

/-- The total number of Easter eggs found -/
def total_eggs : ℕ := club_house_eggs + park_eggs + town_hall_eggs

theorem easter_egg_count : total_eggs = 80 := by
  sorry

end easter_egg_count_l3879_387972


namespace inequality_theorem_largest_constant_l3879_387927

theorem inequality_theorem (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 :=
by sorry

theorem largest_constant :
  ∀ m : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) > m) → m ≤ 2 :=
by sorry

end inequality_theorem_largest_constant_l3879_387927


namespace exists_five_threes_equal_100_l3879_387917

/-- An arithmetic expression using only the number 3, parentheses, and arithmetic operations. -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression. -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Count the number of threes in an expression. -/
def countThrees : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

/-- There exists an arithmetic expression using five threes that evaluates to 100. -/
theorem exists_five_threes_equal_100 : ∃ e : Expr, countThrees e = 5 ∧ eval e = 100 := by
  sorry

end exists_five_threes_equal_100_l3879_387917


namespace circle_center_travel_distance_l3879_387941

-- Define the triangle
def triangle_sides : (ℝ × ℝ × ℝ) := (5, 12, 13)

-- Define the circle radius
def circle_radius : ℝ := 2

-- Define the function to calculate the perimeter of the inscribed triangle
def inscribed_triangle_perimeter (sides : ℝ × ℝ × ℝ) (radius : ℝ) : ℝ :=
  let (a, b, c) := sides
  (a - 2 * radius) + (b - 2 * radius) + (c - 2 * radius)

-- Theorem statement
theorem circle_center_travel_distance :
  inscribed_triangle_perimeter triangle_sides circle_radius = 18 := by
  sorry

end circle_center_travel_distance_l3879_387941
