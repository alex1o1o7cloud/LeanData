import Mathlib

namespace self_inverse_matrix_l3417_341796

/-- A 2x2 matrix is its own inverse if and only if p = 15/2 and q = -4 -/
theorem self_inverse_matrix (p q : ℚ) :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, p; -2, q]
  (A * A = 1) ↔ p = 15/2 ∧ q = -4 := by
  sorry

end self_inverse_matrix_l3417_341796


namespace probability_between_R_and_S_l3417_341763

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8RS,
    the probability of a randomly selected point on PQ being between R and S is 1/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) 
  (h_order : P ≤ R ∧ R ≤ S ∧ S ≤ Q)
  (h_PQ_PR : Q - P = 4 * (R - P))
  (h_PQ_RS : Q - P = 8 * (S - R)) :
  (S - R) / (Q - P) = 1 / 8 := by sorry

end probability_between_R_and_S_l3417_341763


namespace min_value_theorem_l3417_341720

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (2*a + 3*b + 4*c) * ((a + b)⁻¹ + (b + c)⁻¹ + (c + a)⁻¹) ≥ 4.5 := by
  sorry

end min_value_theorem_l3417_341720


namespace unique_pairs_count_l3417_341769

/-- Represents the colors of marbles Tom has --/
inductive MarbleColor
  | Red
  | Green
  | Blue
  | Yellow
  | Orange

/-- Represents Tom's collection of marbles --/
def toms_marbles : List MarbleColor :=
  [MarbleColor.Red, MarbleColor.Green, MarbleColor.Blue,
   MarbleColor.Yellow, MarbleColor.Yellow,
   MarbleColor.Orange, MarbleColor.Orange]

/-- Counts the number of unique pairs of marbles --/
def count_unique_pairs (marbles : List MarbleColor) : Nat :=
  sorry

/-- Theorem stating that the number of unique pairs Tom can choose is 12 --/
theorem unique_pairs_count :
  count_unique_pairs toms_marbles = 12 := by
  sorry

end unique_pairs_count_l3417_341769


namespace grain_equations_correct_l3417_341708

/-- Represents the amount of grain in sheng that one bundle can produce -/
structure GrainBundle where
  amount : ℝ

/-- High-quality grain bundle -/
def high_quality : GrainBundle := sorry

/-- Low-quality grain bundle -/
def low_quality : GrainBundle := sorry

/-- Theorem stating that the system of equations correctly represents the grain problem -/
theorem grain_equations_correct :
  (5 * high_quality.amount - 11 = 7 * low_quality.amount) ∧
  (7 * high_quality.amount - 25 = 5 * low_quality.amount) := by
  sorry

end grain_equations_correct_l3417_341708


namespace person_a_silver_cards_l3417_341754

/-- Represents the number of sheets of each type of card paper -/
structure CardPapers :=
  (red : ℕ)
  (gold : ℕ)
  (silver : ℕ)

/-- Represents the exchange rates between different types of card papers -/
structure ExchangeRates :=
  (red_to_gold : ℕ × ℕ)
  (gold_to_red_and_silver : ℕ × ℕ × ℕ)

/-- Function to perform exchanges and calculate the maximum number of silver cards obtainable -/
def max_silver_obtainable (initial : CardPapers) (rates : ExchangeRates) : ℕ :=
  sorry

/-- Theorem stating that person A can obtain 7 sheets of silver card paper -/
theorem person_a_silver_cards :
  let initial := CardPapers.mk 3 3 0
  let rates := ExchangeRates.mk (5, 2) (1, 1, 1)
  max_silver_obtainable initial rates = 7 :=
sorry

end person_a_silver_cards_l3417_341754


namespace quadratic_function_properties_l3417_341775

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h_odd : ∀ x, f a b c (-x) = -f a b c x)
  (h_f1 : f a b c 1 = b + c)
  (h_f2 : f a b c 2 = 4 * a + 2 * b + c) :
  (a = 2 ∧ b = -3 ∧ c = 0) ∧
  (∀ x, x > 0 → ∀ y, y > x → f a b c y < f a b c x) ∧
  (∃ m, m = 2 ∧ ∀ x, x > 0 → f a b c x ≥ m) := by
  sorry

end quadratic_function_properties_l3417_341775


namespace minimum_coins_l3417_341768

def nickel : ℚ := 5 / 100
def dime : ℚ := 10 / 100
def quarter : ℚ := 25 / 100
def half_dollar : ℚ := 50 / 100

def total_amount : ℚ := 3

theorem minimum_coins (n d q h : ℕ) : 
  n ≥ 1 → d ≥ 1 → q ≥ 1 → h ≥ 1 →
  n * nickel + d * dime + q * quarter + h * half_dollar = total_amount →
  n + d + q + h ≥ 9 :=
by sorry

end minimum_coins_l3417_341768


namespace birth_year_problem_l3417_341779

theorem birth_year_problem (x : ℕ) : 
  (1850 ≤ x^2 - 2*x + 1) ∧ (x^2 - 2*x + 1 < 1900) →
  (x^2 - x + 1 - (x^2 - 2*x + 1) = x) →
  x^2 - 2*x + 1 = 1849 := by
sorry

end birth_year_problem_l3417_341779


namespace smallest_x_for_perfect_cube_sum_l3417_341761

/-- The sum of arithmetic sequence with 5 terms, starting from x and with common difference 3 -/
def sequence_sum (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9) + (x + 12)

/-- A natural number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem smallest_x_for_perfect_cube_sum : 
  (∀ x : ℕ, x > 0 ∧ x < 19 → ¬(is_perfect_cube (sequence_sum x))) ∧ 
  (is_perfect_cube (sequence_sum 19)) := by
sorry

end smallest_x_for_perfect_cube_sum_l3417_341761


namespace min_people_for_hundred_chairs_is_minimum_people_l3417_341777

/-- The number of chairs in the circle -/
def num_chairs : ℕ := 100

/-- A function that calculates the minimum number of people needed -/
def min_people (chairs : ℕ) : ℕ :=
  (chairs + 2) / 3

/-- The theorem stating the minimum number of people for 100 chairs -/
theorem min_people_for_hundred_chairs :
  min_people num_chairs = 34 := by
  sorry

/-- The theorem proving that this is indeed the minimum -/
theorem is_minimum_people (n : ℕ) :
  n < min_people num_chairs →
  ∃ (m : ℕ), m > 2 ∧ m < num_chairs ∧
  ∀ (i : ℕ), i < n → ∃ (j : ℕ), j < n ∧ j ≠ i ∧
  (m + i) % num_chairs = (m + j) % num_chairs := by
  sorry

end min_people_for_hundred_chairs_is_minimum_people_l3417_341777


namespace circle_and_tangent_line_l3417_341797

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 2 = 0

-- Define the point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the line l (we'll use the point-slope form)
def line_l (x y : ℝ) (m : ℝ) : Prop := y - point_P.2 = m * (x - point_P.1)

-- State the theorem
theorem circle_and_tangent_line :
  ∃ (m : ℝ),
    -- The line l passes through P(1,1) and is tangent to C
    (∀ (x y : ℝ), line_l x y m → (circle_equation x y → x = y)) ∧
    -- The radius of C is √2
    (∃ (c_x c_y : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - c_x)^2 + (y - c_y)^2 = 2) ∧
    -- The equation of l is x - y = 0
    (∀ (x y : ℝ), line_l x y m ↔ x = y) := by
  sorry

end circle_and_tangent_line_l3417_341797


namespace like_terms_implies_value_l3417_341774

-- Define the condition for like terms
def are_like_terms (m n : ℕ) : Prop := m = 3 ∧ n = 2

-- State the theorem
theorem like_terms_implies_value (m n : ℕ) :
  are_like_terms m n → (-n : ℤ)^m = -8 := by
  sorry

end like_terms_implies_value_l3417_341774


namespace stating_smallest_n_no_arithmetic_progression_l3417_341767

/-- 
A function that checks if there exists an arithmetic progression of 
1999 terms containing exactly n integers
-/
def exists_arithmetic_progression (n : ℕ) : Prop :=
  ∃ (a d : ℝ), ∃ (k : ℕ), 
    k * n + k - 1 ≥ 1999 ∧
    (k + 1) * n - (k + 1) + 1 ≤ 1999

/-- 
Theorem stating that 70 is the smallest positive integer n such that 
there does not exist an arithmetic progression of 1999 terms of real 
numbers containing exactly n integers
-/
theorem smallest_n_no_arithmetic_progression : 
  (∀ m < 70, exists_arithmetic_progression m) ∧ 
  ¬ exists_arithmetic_progression 70 :=
sorry

end stating_smallest_n_no_arithmetic_progression_l3417_341767


namespace C_is_rotated_X_l3417_341716

/-- A shape in a 2D plane -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Rotation of a shape by 90 degrees clockwise around its center -/
def rotate90 (s : Shape) : Shape := sorry

/-- Two shapes are superimposable if they have the same set of points -/
def superimposable (s1 s2 : Shape) : Prop :=
  s1.points = s2.points

/-- The original shape X -/
def X : Shape := sorry

/-- The alternative shapes -/
def A : Shape := sorry
def B : Shape := sorry
def C : Shape := sorry
def D : Shape := sorry
def E : Shape := sorry

/-- The theorem stating that C is the only shape superimposable with X after rotation -/
theorem C_is_rotated_X : 
  superimposable (rotate90 X) C ∧ 
  (¬superimposable (rotate90 X) A ∧
   ¬superimposable (rotate90 X) B ∧
   ¬superimposable (rotate90 X) D ∧
   ¬superimposable (rotate90 X) E) :=
sorry

end C_is_rotated_X_l3417_341716


namespace cosine_identity_l3417_341742

theorem cosine_identity (α : ℝ) :
  3 - 4 * Real.cos (4 * α - 3 * Real.pi) - Real.cos (5 * Real.pi + 8 * α) = 8 * (Real.cos (2 * α))^4 := by
  sorry

end cosine_identity_l3417_341742


namespace negative_sqrt_ten_less_than_negative_three_l3417_341799

theorem negative_sqrt_ten_less_than_negative_three :
  -Real.sqrt 10 < -3 := by
  sorry

end negative_sqrt_ten_less_than_negative_three_l3417_341799


namespace valid_fractions_characterization_l3417_341740

def is_valid_fraction (n d : ℕ) : Prop :=
  0 < d ∧ d < 10 ∧ (7:ℚ)/9 < (n:ℚ)/d ∧ (n:ℚ)/d < (8:ℚ)/9

def valid_fractions : Set (ℕ × ℕ) :=
  {(n, d) | is_valid_fraction n d}

theorem valid_fractions_characterization :
  valid_fractions = {(5, 6), (6, 7), (7, 8), (4, 5)} := by sorry

end valid_fractions_characterization_l3417_341740


namespace quadratic_function_properties_l3417_341758

/-- A quadratic function -/
noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) :
  (∀ p q : ℝ, p ≠ q → f a b c p = f a b c q → f a b c (p + q) = c) ∧
  (∀ p q : ℝ, p ≠ q → f a b c (p + q) = c → (p + q = 0 ∨ f a b c p = f a b c q)) :=
sorry

end quadratic_function_properties_l3417_341758


namespace john_unintended_texts_l3417_341762

/-- The number of text messages John receives per week that are not intended for him -/
def unintended_texts_per_week (old_daily_texts old_daily_texts_from_friends new_daily_texts days_per_week : ℕ) : ℕ :=
  (new_daily_texts - old_daily_texts) * days_per_week

/-- Proof that John receives 245 unintended text messages per week -/
theorem john_unintended_texts :
  let old_daily_texts : ℕ := 20
  let new_daily_texts : ℕ := 55
  let days_per_week : ℕ := 7
  unintended_texts_per_week old_daily_texts old_daily_texts new_daily_texts days_per_week = 245 :=
by sorry

end john_unintended_texts_l3417_341762


namespace circle_equation_l3417_341778

/-- A circle passing through points A(0, -6) and B(1, -5) with center C on the line x-y+1=0 
    has the standard equation (x + 3)^2 + (y + 2)^2 = 25 -/
theorem circle_equation (C : ℝ × ℝ) : 
  (C.1 - C.2 + 1 = 0) →  -- C lies on the line x-y+1=0
  ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 = ((1 : ℝ) - C.1)^2 + ((-5 : ℝ) - C.2)^2 →  -- C is equidistant from A and B
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - C.1)^2 + (y - C.2)^2 = ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 :=
by sorry

end circle_equation_l3417_341778


namespace virus_infected_computers_office_virus_scenario_l3417_341784

/-- Represents the state of computers in an office before and after a virus infection. -/
structure ComputerNetwork where
  total : ℕ             -- Total number of computers
  infected : ℕ          -- Number of infected computers
  initialConnections : ℕ -- Number of initial connections per computer
  finalConnections : ℕ  -- Number of final connections per uninfected computer
  disconnectedCables : ℕ -- Number of cables disconnected due to virus

/-- The theorem stating the number of infected computers given the network conditions -/
theorem virus_infected_computers (network : ComputerNetwork) : 
  network.initialConnections = 5 ∧ 
  network.finalConnections = 3 ∧ 
  network.disconnectedCables = 26 →
  network.infected = 8 := by
  sorry

/-- Main theorem proving the number of infected computers in the given scenario -/
theorem office_virus_scenario : ∃ (network : ComputerNetwork), 
  network.initialConnections = 5 ∧
  network.finalConnections = 3 ∧
  network.disconnectedCables = 26 ∧
  network.infected = 8 := by
  sorry

end virus_infected_computers_office_virus_scenario_l3417_341784


namespace cube_eight_eq_two_power_ten_unique_solution_l3417_341715

theorem cube_eight_eq_two_power_ten :
  8^3 + 8^3 + 8^3 = 2^10 := by
sorry

theorem unique_solution (x : ℕ) :
  8^3 + 8^3 + 8^3 = 2^x → x = 10 := by
sorry

end cube_eight_eq_two_power_ten_unique_solution_l3417_341715


namespace white_bar_dimensions_l3417_341725

/-- Represents the dimensions of a bar in the cube -/
structure BarDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- The cube assembled from bars -/
structure Cube where
  edge_length : ℚ
  bar_count : ℕ
  gray_bar : BarDimensions
  white_bar : BarDimensions

/-- Theorem stating the dimensions of the white bar in the cube -/
theorem white_bar_dimensions (c : Cube) : 
  c.edge_length = 1 ∧ 
  c.bar_count = 8 ∧ 
  (c.gray_bar.length * c.gray_bar.width * c.gray_bar.height = 
   c.white_bar.length * c.white_bar.width * c.white_bar.height) →
  c.white_bar = ⟨7/10, 1/2, 5/14⟩ := by
  sorry

end white_bar_dimensions_l3417_341725


namespace at_least_three_babies_speak_l3417_341750

def probability_baby_speaks : ℚ := 1 / 3

def number_of_babies : ℕ := 6

def probability_at_least_three_speak (p : ℚ) (n : ℕ) : ℚ :=
  1 - (Nat.choose n 0 * (1 - p)^n + 
       Nat.choose n 1 * p * (1 - p)^(n-1) + 
       Nat.choose n 2 * p^2 * (1 - p)^(n-2))

theorem at_least_three_babies_speak : 
  probability_at_least_three_speak probability_baby_speaks number_of_babies = 353 / 729 := by
  sorry

end at_least_three_babies_speak_l3417_341750


namespace solve_quadratic_coefficients_l3417_341788

-- Define the universal set U
def U : Set ℤ := {2, 3, 5}

-- Define the set A
def A (b c : ℤ) : Set ℤ := {x ∈ U | x^2 + b*x + c = 0}

-- Define the theorem
theorem solve_quadratic_coefficients :
  ∀ b c : ℤ, (U \ A b c = {2}) → (b = -8 ∧ c = 15) :=
by
  sorry

end solve_quadratic_coefficients_l3417_341788


namespace three_at_five_equals_neg_six_l3417_341709

-- Define the @ operation
def at_op (a b : ℤ) : ℤ := 3 * a - 3 * b

-- Theorem statement
theorem three_at_five_equals_neg_six : at_op 3 5 = -6 := by
  sorry

end three_at_five_equals_neg_six_l3417_341709


namespace swimmer_speed_in_still_water_l3417_341703

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions of the swimmer's journey, his speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed) 
  (h1 : effectiveSpeed s true * 5 = 45)  -- Downstream condition
  (h2 : effectiveSpeed s false * 5 = 25) -- Upstream condition
  : s.swimmer = 7 := by
  sorry


end swimmer_speed_in_still_water_l3417_341703


namespace product_of_five_consecutive_integers_divisible_by_10_l3417_341755

theorem product_of_five_consecutive_integers_divisible_by_10 (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = 10 * k := by
  sorry

end product_of_five_consecutive_integers_divisible_by_10_l3417_341755


namespace three_digit_multiples_of_seven_l3417_341701

theorem three_digit_multiples_of_seven (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k ∧ 100 ≤ n ∧ n ≤ 999) ↔ n ∈ Finset.range 128 ∧ n ≠ 0 :=
sorry

end three_digit_multiples_of_seven_l3417_341701


namespace unique_solution_2011_l3417_341781

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem unique_solution_2011 :
  ∃! n : ℕ, n + sum_of_digits n = 2011 ∧ n = 1991 :=
sorry

end unique_solution_2011_l3417_341781


namespace pyramid_volume_l3417_341785

/-- The volume of a pyramid with a rectangular base and equal edge lengths from apex to base corners -/
theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) :
  base_length = 5 →
  base_width = 7 →
  edge_length = 15 →
  let base_area := base_length * base_width
  let base_diagonal := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (base_diagonal / 2)^2)
  (1 / 3 : ℝ) * base_area * height = (35 * Real.sqrt 188) / 3 :=
by sorry

end pyramid_volume_l3417_341785


namespace benny_stored_bales_l3417_341747

/-- The number of bales Benny stored in the barn -/
def bales_stored (initial_bales current_bales : ℕ) : ℕ :=
  current_bales - initial_bales

/-- Theorem stating that Benny stored 35 bales in the barn -/
theorem benny_stored_bales : 
  let initial_bales : ℕ := 47
  let current_bales : ℕ := 82
  bales_stored initial_bales current_bales = 35 := by
sorry

end benny_stored_bales_l3417_341747


namespace number_equation_proof_l3417_341772

theorem number_equation_proof (x : ℝ) (N : ℝ) : 
  x = 32 → 
  N - (23 - (15 - x)) = 12 * 2 / (1 / 2) → 
  N = 88 := by
sorry

end number_equation_proof_l3417_341772


namespace fort_food_duration_l3417_341743

/-- Calculates the initial number of days the food provision was meant to last given:
  * The initial number of men
  * The number of days after which some men left
  * The number of men who left
  * The number of days the food lasted after some men left
-/
def initialFoodDuration (initialMen : ℕ) (daysBeforeLeaving : ℕ) (menWhoLeft : ℕ) (remainingDays : ℕ) : ℕ :=
  (initialMen * daysBeforeLeaving + (initialMen - menWhoLeft) * remainingDays) / initialMen

theorem fort_food_duration :
  initialFoodDuration 150 10 25 42 = 45 := by
  sorry

#eval initialFoodDuration 150 10 25 42

end fort_food_duration_l3417_341743


namespace sum_of_x_and_y_equals_two_l3417_341776

theorem sum_of_x_and_y_equals_two (x y : ℝ) 
  (eq1 : 2 * x + 3 * y = 6)
  (eq2 : 3 * x + 2 * y = 4) : 
  x + y = 2 := by
sorry

end sum_of_x_and_y_equals_two_l3417_341776


namespace last_two_digits_factorial_sum_l3417_341793

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_factorial_sum :
  last_two_digits (sum_factorials 15) = last_two_digits (sum_factorials 9) :=
sorry

end last_two_digits_factorial_sum_l3417_341793


namespace am_gm_difference_bound_l3417_341791

theorem am_gm_difference_bound (a : ℝ) (h : 0 < a) :
  let b := a + 1
  let am := (a + b) / 2
  let gm := Real.sqrt (a * b)
  am - gm < (1 : ℝ) / 2 := by sorry

end am_gm_difference_bound_l3417_341791


namespace minimum_bailing_rate_l3417_341770

/-- Represents the minimum bailing rate problem --/
theorem minimum_bailing_rate 
  (distance_to_shore : ℝ) 
  (leaking_rate : ℝ) 
  (max_water_tolerance : ℝ) 
  (boat_speed : ℝ) 
  (h1 : distance_to_shore = 2) 
  (h2 : leaking_rate = 8) 
  (h3 : max_water_tolerance = 50) 
  (h4 : boat_speed = 3) :
  ∃ (bailing_rate : ℝ), 
    bailing_rate ≥ 7 ∧ 
    bailing_rate < 8 ∧
    (leaking_rate - bailing_rate) * (distance_to_shore / boat_speed * 60) ≤ max_water_tolerance :=
by sorry

end minimum_bailing_rate_l3417_341770


namespace prudence_sleep_is_200_l3417_341792

/-- Represents Prudence's sleep schedule and calculates total sleep over 4 weeks -/
def prudence_sleep : ℕ :=
  let weekday_sleep : ℕ := 5 * 6  -- 5 nights of 6 hours each
  let weekend_sleep : ℕ := 2 * 9  -- 2 nights of 9 hours each
  let nap_sleep : ℕ := 2 * 1      -- 2 days of 1 hour nap each
  let weekly_sleep : ℕ := weekday_sleep + weekend_sleep + nap_sleep
  4 * weekly_sleep                -- 4 weeks

/-- Theorem stating that Prudence's total sleep over 4 weeks is 200 hours -/
theorem prudence_sleep_is_200 : prudence_sleep = 200 := by
  sorry

#eval prudence_sleep  -- This will evaluate to 200

end prudence_sleep_is_200_l3417_341792


namespace constant_term_value_l3417_341710

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of the first two binomial coefficients equals 10 -/
def sum_first_two_coefficients (n : ℕ) : Prop :=
  binomial n 0 + binomial n 1 = 10

/-- The constant term in the expansion -/
def constant_term (n : ℕ) : ℕ :=
  2^(n - 6) * binomial n 6

theorem constant_term_value (n : ℕ) :
  sum_first_two_coefficients n → constant_term n = 672 := by
  sorry

end constant_term_value_l3417_341710


namespace polynomial_factorization_l3417_341765

theorem polynomial_factorization (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + x^3 - 5*x^2 + x - 6) →
  a + b + c + d = -4 := by
  sorry

end polynomial_factorization_l3417_341765


namespace complex_abs_from_square_l3417_341744

theorem complex_abs_from_square (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end complex_abs_from_square_l3417_341744


namespace total_strings_is_40_l3417_341771

/-- The number of strings on all instruments in Francis' family -/
def total_strings : ℕ :=
  let ukulele_count : ℕ := 2
  let guitar_count : ℕ := 4
  let violin_count : ℕ := 2
  let strings_per_ukulele : ℕ := 4
  let strings_per_guitar : ℕ := 6
  let strings_per_violin : ℕ := 4
  ukulele_count * strings_per_ukulele +
  guitar_count * strings_per_guitar +
  violin_count * strings_per_violin

theorem total_strings_is_40 : total_strings = 40 := by
  sorry

end total_strings_is_40_l3417_341771


namespace F_and_G_increasing_l3417_341726

-- Define the functions f, g, F, and G
variable (f g : ℝ → ℝ)
def F (x : ℝ) := f x + g x
def G (x : ℝ) := f x - g x

-- State the theorem
theorem F_and_G_increasing
  (h_incr : ∀ x y, x < y → f x < f y)
  (h_ineq : ∀ x y, x ≠ y → (f x - f y)^2 > (g x - g y)^2) :
  (∀ x y, x < y → F f g x < F f g y) ∧
  (∀ x y, x < y → G f g x < G f g y) :=
sorry

end F_and_G_increasing_l3417_341726


namespace kelsey_watched_160_l3417_341713

/-- The number of videos watched by three friends satisfies the given conditions -/
structure VideoWatching where
  total : ℕ
  kelsey_more_than_ekon : ℕ
  uma_more_than_ekon : ℕ
  h_total : total = 411
  h_kelsey_more : kelsey_more_than_ekon = 43
  h_uma_more : uma_more_than_ekon = 17

/-- Given the conditions, prove that Kelsey watched 160 videos -/
theorem kelsey_watched_160 (vw : VideoWatching) : 
  ∃ (ekon uma kelsey : ℕ), 
    ekon + uma + kelsey = vw.total ∧ 
    kelsey = ekon + vw.kelsey_more_than_ekon ∧ 
    uma = ekon + vw.uma_more_than_ekon ∧
    kelsey = 160 := by
  sorry

end kelsey_watched_160_l3417_341713


namespace closest_to_300_l3417_341746

def expression : ℝ := 3.25 * 9.252 * (6.22 + 3.78) - 10

def options : List ℝ := [250, 300, 350, 400, 450]

theorem closest_to_300 : 
  ∀ x ∈ options, x ≠ 300 → |expression - 300| < |expression - x| :=
sorry

end closest_to_300_l3417_341746


namespace proposition_p_or_q_is_true_l3417_341749

theorem proposition_p_or_q_is_true : 
  (1 ∈ { x : ℝ | x^2 - 2*x + 1 ≤ 0 }) ∨ (∀ x ∈ (Set.Icc 0 1 : Set ℝ), x^2 - 1 ≥ 0) := by
  sorry

end proposition_p_or_q_is_true_l3417_341749


namespace temperature_change_l3417_341748

/-- The temperature change problem -/
theorem temperature_change (initial temp_rise temp_drop final : Int) : 
  initial = -12 → 
  temp_rise = 8 → 
  temp_drop = 10 → 
  final = initial + temp_rise - temp_drop → 
  final = -14 := by sorry

end temperature_change_l3417_341748


namespace bible_yellow_tickets_l3417_341705

-- Define the conversion rates
def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10

-- Define Tom's current tickets
def tom_yellow : ℕ := 8
def tom_red : ℕ := 3
def tom_blue : ℕ := 7

-- Define the additional blue tickets needed
def additional_blue : ℕ := 163

-- Define the function to calculate the total blue tickets equivalent
def total_blue_equivalent (yellow red blue : ℕ) : ℕ :=
  yellow * yellow_to_red * red_to_blue + red * red_to_blue + blue

-- Theorem statement
theorem bible_yellow_tickets :
  ∃ (required_yellow : ℕ),
    required_yellow = 10 ∧
    total_blue_equivalent tom_yellow tom_red tom_blue + additional_blue =
    required_yellow * yellow_to_red * red_to_blue :=
by sorry

end bible_yellow_tickets_l3417_341705


namespace spherical_coordinate_transformation_l3417_341790

/-- Given a point in rectangular coordinates (3, -8, 6) with corresponding
    spherical coordinates (ρ, θ, φ), this theorem proves the rectangular
    coordinates of the point with spherical coordinates (ρ, θ + π/4, -φ). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  3 = ρ * Real.sin φ * Real.cos θ →
  -8 = ρ * Real.sin φ * Real.sin θ →
  6 = ρ * Real.cos φ →
  ∃ (x y : ℝ),
    x = -ρ * Real.sin φ * (Real.sqrt 2 / 2 * Real.cos θ - Real.sqrt 2 / 2 * Real.sin θ) ∧
    y = -ρ * Real.sin φ * (Real.sqrt 2 / 2 * Real.sin θ + Real.sqrt 2 / 2 * Real.cos θ) ∧
    6 = ρ * Real.cos φ :=
by sorry

end spherical_coordinate_transformation_l3417_341790


namespace vitya_older_probability_l3417_341795

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The probability that Vitya is at least one day older than Masha -/
def probability_vitya_older (june_days : ℕ) : ℚ :=
  (june_days * (june_days - 1) / 2) / (june_days * june_days)

theorem vitya_older_probability :
  probability_vitya_older june_days = 29 / 60 := by
  sorry

end vitya_older_probability_l3417_341795


namespace john_house_wall_planks_l3417_341757

/-- The number of planks John uses for the house wall -/
def num_planks : ℕ := 32 / 2

/-- Each plank needs 2 nails -/
def nails_per_plank : ℕ := 2

/-- The total number of nails needed for the wall -/
def total_nails : ℕ := 32

theorem john_house_wall_planks : num_planks = 16 := by
  sorry

end john_house_wall_planks_l3417_341757


namespace frequency_proportion_l3417_341733

theorem frequency_proportion (frequency sample_size : ℕ) 
  (h1 : frequency = 80) 
  (h2 : sample_size = 100) : 
  (frequency : ℚ) / sample_size = 0.8 := by
  sorry

end frequency_proportion_l3417_341733


namespace polynomial_rational_difference_l3417_341753

theorem polynomial_rational_difference (f : ℝ → ℝ) :
  (∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) →
  (∀ (x y : ℝ), ∃ (q : ℚ), x - y = q → ∃ (r : ℚ), f x - f y = r) →
  ∃ (b : ℚ) (c : ℝ), ∀ x, f x = b * x + c :=
by sorry

end polynomial_rational_difference_l3417_341753


namespace semicircle_to_cone_volume_l3417_341727

/-- The volume of a cone formed by rolling a semicircle of radius R --/
theorem semicircle_to_cone_volume (R : ℝ) (R_pos : R > 0) :
  let r : ℝ := R / 2
  let h : ℝ := R * (Real.sqrt 3) / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.sqrt 3 / 24) * Real.pi * R^3 := by
  sorry

end semicircle_to_cone_volume_l3417_341727


namespace inequality_solution_set_l3417_341751

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  S = if a > 0 then
        {x : ℝ | x < -a/4 ∨ x > a/3}
      else if a = 0 then
        {x : ℝ | x ≠ 0}
      else
        {x : ℝ | x < a/3 ∨ x > -a/4} :=
by sorry

end inequality_solution_set_l3417_341751


namespace fraction_range_l3417_341719

theorem fraction_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a ≤ 2*b ∧ 2*b ≤ 2*a + b) :
  (4/9 : ℝ) ≤ (2*a*b)/(a^2 + 2*b^2) ∧ (2*a*b)/(a^2 + 2*b^2) ≤ Real.sqrt 2 / 2 := by
  sorry

end fraction_range_l3417_341719


namespace min_value_theorem_l3417_341766

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 45) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 45 → 1/x + 4/y ≥ 1/5) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 45 ∧ 1/x + 4/y = 1/5) :=
by sorry

end min_value_theorem_l3417_341766


namespace club_diamond_heart_probability_l3417_341702

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total : Nat)
  (clubs : Nat)
  (diamonds : Nat)
  (hearts : Nat)

/-- The probability of drawing the sequence: club, diamond, heart -/
def sequence_probability (d : Deck) : ℚ :=
  (d.clubs : ℚ) / d.total *
  (d.diamonds : ℚ) / (d.total - 1) *
  (d.hearts : ℚ) / (d.total - 2)

theorem club_diamond_heart_probability :
  let standard_deck : Deck := ⟨52, 13, 13, 13⟩
  sequence_probability standard_deck = 2197 / 132600 := by
  sorry

end club_diamond_heart_probability_l3417_341702


namespace daves_painted_area_l3417_341728

theorem daves_painted_area 
  (total_area : ℝ) 
  (cathy_ratio : ℝ) 
  (dave_ratio : ℝ) 
  (h1 : total_area = 330) 
  (h2 : cathy_ratio = 4) 
  (h3 : dave_ratio = 7) : 
  dave_ratio / (cathy_ratio + dave_ratio) * total_area = 210 := by
sorry

end daves_painted_area_l3417_341728


namespace smallest_number_l3417_341712

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, 1, -5}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -5 :=
sorry

end smallest_number_l3417_341712


namespace special_polynomial_root_l3417_341732

/-- A fourth degree polynomial with specific root properties -/
structure SpecialPolynomial where
  P : ℝ → ℝ
  degree_four : ∃ (a b c d e : ℝ), ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e
  root_one : P 1 = 0
  root_three : P 3 = 0
  root_five : P 5 = 0
  derivative_root_seven : (deriv P) 7 = 0

/-- The remaining root of a SpecialPolynomial is 89/11 -/
theorem special_polynomial_root (p : SpecialPolynomial) : 
  ∃ (x : ℝ), x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ p.P x = 0 ∧ x = 89/11 := by
  sorry

end special_polynomial_root_l3417_341732


namespace single_intersection_l3417_341780

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 2 * y + 4

/-- The line function -/
def line (k : ℝ) : ℝ := k

/-- Theorem stating the condition for single intersection -/
theorem single_intersection (k : ℝ) : 
  (∃! y, parabola y = line k) ↔ k = 13/3 := by sorry

end single_intersection_l3417_341780


namespace intersection_of_A_and_B_l3417_341731

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B : Set ℝ := {x | 2*x - 1 > 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (1/2) 1 := by
  sorry

end intersection_of_A_and_B_l3417_341731


namespace process_600_parts_l3417_341734

/-- The regression line equation for processing parts -/
def regression_line (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem process_600_parts : regression_line 600 = 6.5 := by
  sorry

end process_600_parts_l3417_341734


namespace anthony_pencils_l3417_341798

theorem anthony_pencils (initial final added : ℕ) 
  (h1 : added = 56)
  (h2 : final = 65)
  (h3 : final = initial + added) :
  initial = 9 := by
sorry

end anthony_pencils_l3417_341798


namespace circle_center_correct_l3417_341783

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 (-6) 1 2 (-75)
  findCenter eq = CircleCenter.mk 3 (-1) := by sorry

end circle_center_correct_l3417_341783


namespace rectangle_area_difference_l3417_341745

theorem rectangle_area_difference : 
  let rect1_width : ℕ := 4
  let rect1_height : ℕ := 5
  let rect2_width : ℕ := 3
  let rect2_height : ℕ := 6
  let rect1_area := rect1_width * rect1_height
  let rect2_area := rect2_width * rect2_height
  rect1_area - rect2_area = 2 :=
by sorry

end rectangle_area_difference_l3417_341745


namespace quadratic_roots_range_l3417_341724

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0) → m < (1/2) :=
by sorry

end quadratic_roots_range_l3417_341724


namespace f_difference_l3417_341711

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as the sum of all positive divisors of n divided by n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(640) - f(320) = 3/320 -/
theorem f_difference : f 640 - f 320 = 3 / 320 := by sorry

end f_difference_l3417_341711


namespace chord_length_line_circle_intersection_l3417_341782

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection :
  let line : ℝ → ℝ → Prop := λ x y ↦ 2 * x - y + 1 = 0
  let circle : ℝ → ℝ → Prop := λ x y ↦ (x - 1)^2 + (y - 1)^2 = 1
  ∃ (A B : ℝ × ℝ),
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle A.1 A.2 ∧ circle B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5 / 5 :=
by sorry

end chord_length_line_circle_intersection_l3417_341782


namespace inequality_solution_set_l3417_341738

theorem inequality_solution_set (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 := by
  sorry

end inequality_solution_set_l3417_341738


namespace initial_water_percentage_in_milk_l3417_341760

/-- The initial percentage of water in milk, given that adding 15 liters of pure milk to 10 liters
    of the initial milk reduces the water content to 2%. -/
theorem initial_water_percentage_in_milk :
  ∀ (initial_water_percentage : ℝ),
    (initial_water_percentage ≥ 0) →
    (initial_water_percentage ≤ 100) →
    (10 * (100 - initial_water_percentage) / 100 + 15 = 0.98 * 25) →
    initial_water_percentage = 5 := by
  sorry

end initial_water_percentage_in_milk_l3417_341760


namespace quadratic_abs_value_analysis_l3417_341739

theorem quadratic_abs_value_analysis (x a : ℝ) :
  (x ≥ a → x^2 + 4*x - 2*|x - a| + 2 - a = x^2 + 2*x + a + 2) ∧
  (x < a → x^2 + 4*x - 2*|x - a| + 2 - a = x^2 + 6*x - 3*a + 2) :=
by sorry

end quadratic_abs_value_analysis_l3417_341739


namespace adams_earnings_l3417_341706

theorem adams_earnings (daily_wage : ℝ) (tax_rate : ℝ) (work_days : ℕ) :
  daily_wage = 40 →
  tax_rate = 0.1 →
  work_days = 30 →
  (daily_wage * (1 - tax_rate) * work_days : ℝ) = 1080 := by
  sorry

end adams_earnings_l3417_341706


namespace P_sufficient_not_necessary_Q_l3417_341704

/-- Triangle inequality condition -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Proposition P: segments can form a triangle -/
def P (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Proposition Q: sum of squares inequality -/
def Q (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a)

/-- P is sufficient but not necessary for Q -/
theorem P_sufficient_not_necessary_Q :
  (∀ a b c : ℝ, P a b c → Q a b c) ∧
  (∃ a b c : ℝ, Q a b c ∧ ¬P a b c) := by
  sorry

end P_sufficient_not_necessary_Q_l3417_341704


namespace perpendicular_lines_slope_l3417_341794

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - 2 ∨ y = (a + 2) * x + 1) →
  (a * (a + 2) = -1) →
  a = -1 := by
sorry

end perpendicular_lines_slope_l3417_341794


namespace no_real_solution_for_matrix_equation_l3417_341737

theorem no_real_solution_for_matrix_equation :
  (∀ a b : ℝ, Matrix.det !![a, 2*b; 2*a, b] = a*b - 4*a*b) →
  ¬∃ x : ℝ, Matrix.det !![3*x, 2; 6*x, x] = 6 := by
  sorry

end no_real_solution_for_matrix_equation_l3417_341737


namespace right_angled_triangle_l3417_341759

theorem right_angled_triangle (α β γ : Real) (h1 : 0 < α) (h2 : 0 < β) (h3 : 0 < γ)
  (h4 : α + β + γ = Real.pi) 
  (h5 : (Real.sin α + Real.sin β) / (Real.cos α + Real.cos β) = Real.sin γ) : 
  γ = Real.pi / 2 := by
sorry

end right_angled_triangle_l3417_341759


namespace doritos_distribution_l3417_341789

theorem doritos_distribution (total_bags : ℕ) (doritos_fraction : ℚ) (num_piles : ℕ) : 
  total_bags = 80 →
  doritos_fraction = 1/4 →
  num_piles = 4 →
  (total_bags : ℚ) * doritos_fraction / num_piles = 5 := by
  sorry

end doritos_distribution_l3417_341789


namespace onion_harvest_weight_l3417_341735

-- Define the number of bags per trip
def bags_per_trip : ℕ := 10

-- Define the weight of each bag in kg
def weight_per_bag : ℕ := 50

-- Define the number of trips
def number_of_trips : ℕ := 20

-- Define the total weight of onions harvested
def total_weight : ℕ := bags_per_trip * weight_per_bag * number_of_trips

-- Theorem statement
theorem onion_harvest_weight :
  total_weight = 10000 := by sorry

end onion_harvest_weight_l3417_341735


namespace book_pages_calculation_l3417_341718

/-- Given a book with a specific number of chapters and pages per chapter, 
    calculate the total number of pages in the book. -/
theorem book_pages_calculation (num_chapters : ℕ) (pages_per_chapter : ℕ) 
    (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) : 
    num_chapters * pages_per_chapter = 1891 := by
  sorry

#check book_pages_calculation

end book_pages_calculation_l3417_341718


namespace seventh_term_is_10_4_l3417_341787

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first four terms is 20
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 20
  -- Fifth term is 8
  fifth_term : a + 4*d = 8

/-- The seventh term of the arithmetic sequence is 10.4 -/
theorem seventh_term_is_10_4 (seq : ArithmeticSequence) : 
  seq.a + 6*seq.d = 10.4 := by
  sorry

end seventh_term_is_10_4_l3417_341787


namespace special_function_value_l3417_341773

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y + 6 * x * y) ∧
  (f (-1) * f 1 ≥ 9)

/-- Theorem stating that for any function satisfying the special conditions,
    f(2/3) = 4/3 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) :
  f (2/3) = 4/3 := by
  sorry

end special_function_value_l3417_341773


namespace ln_neg_implies_a_less_than_one_a_less_than_one_not_sufficient_for_ln_neg_l3417_341723

theorem ln_neg_implies_a_less_than_one :
  ∀ a : ℝ, Real.log a < 0 → a < 1 :=
sorry

theorem a_less_than_one_not_sufficient_for_ln_neg :
  ∃ a : ℝ, a < 1 ∧ ¬(Real.log a < 0) :=
sorry

end ln_neg_implies_a_less_than_one_a_less_than_one_not_sufficient_for_ln_neg_l3417_341723


namespace sqrt_three_comparison_l3417_341752

theorem sqrt_three_comparison : 2 * Real.sqrt 3 > 3 := by
  sorry

end sqrt_three_comparison_l3417_341752


namespace same_solution_implies_k_equals_one_l3417_341729

theorem same_solution_implies_k_equals_one :
  (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 := by
  sorry

end same_solution_implies_k_equals_one_l3417_341729


namespace fraction_equality_l3417_341714

theorem fraction_equality : (3 : ℚ) / (1 + 3 / 5) = 15 / 8 := by
  sorry

end fraction_equality_l3417_341714


namespace negative_885_degrees_conversion_l3417_341764

theorem negative_885_degrees_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    -885 * (π / 180) = 2 * k * π + α ∧
    0 ≤ α ∧ α ≤ 2 * π ∧
    k = -6 ∧ α = 13 * π / 12 := by
  sorry

end negative_885_degrees_conversion_l3417_341764


namespace line_through_circle_centers_l3417_341756

/-- Given two circles in polar coordinates, C1: ρ = 2cos θ and C2: ρ = 2sin θ,
    the polar equation of the line passing through their centers is θ = π/4 -/
theorem line_through_circle_centers (θ : Real) :
  let c1 : Real → Real := fun θ => 2 * Real.cos θ
  let c2 : Real → Real := fun θ => 2 * Real.sin θ
  ∃ (ρ : Real), (ρ * Real.cos (π/4) = 1 ∧ ρ * Real.sin (π/4) = 1) :=
by sorry

end line_through_circle_centers_l3417_341756


namespace x_squared_plus_y_squared_equals_four_l3417_341700

theorem x_squared_plus_y_squared_equals_four (x y : ℝ) :
  (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6 → x^2 + y^2 = 4 := by
  sorry

end x_squared_plus_y_squared_equals_four_l3417_341700


namespace intersection_A_B_l3417_341721

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x < 0}

-- Define set B
def B : Set ℝ := {x | x > 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 2 3 := by sorry

end intersection_A_B_l3417_341721


namespace min_value_quadratic_l3417_341741

theorem min_value_quadratic : 
  ∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896 ∧ 
  ∃ x₀ : ℝ, 3 * x₀^2 - 12 * x₀ + 908 = 896 :=
by sorry

end min_value_quadratic_l3417_341741


namespace rudy_typing_speed_l3417_341722

def team_size : ℕ := 5
def team_average : ℕ := 80
def joyce_speed : ℕ := 76
def gladys_speed : ℕ := 91
def lisa_speed : ℕ := 80
def mike_speed : ℕ := 89

theorem rudy_typing_speed :
  ∃ (rudy_speed : ℕ),
    rudy_speed = team_size * team_average - (joyce_speed + gladys_speed + lisa_speed + mike_speed) :=
by sorry

end rudy_typing_speed_l3417_341722


namespace sine_cosine_sum_l3417_341730

theorem sine_cosine_sum (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.sin (2*α - π/6) + Real.cos (2*α) = 7/9 := by
  sorry

end sine_cosine_sum_l3417_341730


namespace smallest_divisible_by_10_and_24_l3417_341717

theorem smallest_divisible_by_10_and_24 : ∃ n : ℕ, n > 0 ∧ n % 10 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 0 → m % 24 = 0 → n ≤ m := by
  sorry

end smallest_divisible_by_10_and_24_l3417_341717


namespace hyperbola_eccentricity_l3417_341786

theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
  (h_equilateral : b / a = Real.sqrt 3 / 3) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end hyperbola_eccentricity_l3417_341786


namespace cube_root_negative_eight_properties_l3417_341707

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Main theorem
theorem cube_root_negative_eight_properties :
  let y := cubeRoot (-8)
  ∃ (z : ℝ),
    -- Statement A: y represents the cube root of -8
    z^3 = -8 ∧
    -- Statement B: y results in -2
    y = -2 ∧
    -- Statement C: y is equal to -cubeRoot(8)
    y = -(cubeRoot 8) :=
by sorry

end cube_root_negative_eight_properties_l3417_341707


namespace triangle_perimeter_l3417_341736

/-- A triangle with specific area and angles has a specific perimeter -/
theorem triangle_perimeter (A B C : ℝ) (h_area : A = 3 - Real.sqrt 3)
    (h_angle1 : B = 45 * π / 180) (h_angle2 : C = 60 * π / 180) (h_angle3 : A = 75 * π / 180) :
  let perimeter := Real.sqrt 2 * (3 + 2 * Real.sqrt 3 - Real.sqrt 6)
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = perimeter ∧
    (1/2) * a * b * Real.sin C = 3 - Real.sqrt 3 :=
by sorry

end triangle_perimeter_l3417_341736
