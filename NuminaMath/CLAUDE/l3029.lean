import Mathlib

namespace NUMINAMATH_CALUDE_fewer_football_boxes_l3029_302944

theorem fewer_football_boxes (total_cards : ℕ) (basketball_boxes : ℕ) (cards_per_basketball_box : ℕ) (cards_per_football_box : ℕ) 
  (h1 : total_cards = 255)
  (h2 : basketball_boxes = 9)
  (h3 : cards_per_basketball_box = 15)
  (h4 : cards_per_football_box = 20)
  (h5 : basketball_boxes * cards_per_basketball_box + (total_cards - basketball_boxes * cards_per_basketball_box) = total_cards)
  (h6 : (total_cards - basketball_boxes * cards_per_basketball_box) % cards_per_football_box = 0) :
  basketball_boxes - (total_cards - basketball_boxes * cards_per_basketball_box) / cards_per_football_box = 3 := by
  sorry

end NUMINAMATH_CALUDE_fewer_football_boxes_l3029_302944


namespace NUMINAMATH_CALUDE_percentage_of_pistachios_with_shells_l3029_302980

theorem percentage_of_pistachios_with_shells 
  (total_pistachios : ℕ)
  (opened_shell_ratio : ℚ)
  (opened_shell_count : ℕ)
  (h1 : total_pistachios = 80)
  (h2 : opened_shell_ratio = 3/4)
  (h3 : opened_shell_count = 57) :
  (↑opened_shell_count / (↑total_pistachios * opened_shell_ratio) : ℚ) = 95/100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_pistachios_with_shells_l3029_302980


namespace NUMINAMATH_CALUDE_percentage_of_boats_eaten_by_fish_l3029_302959

theorem percentage_of_boats_eaten_by_fish 
  (initial_boats : ℕ) 
  (shot_boats : ℕ) 
  (remaining_boats : ℕ) 
  (h1 : initial_boats = 30) 
  (h2 : shot_boats = 2) 
  (h3 : remaining_boats = 22) : 
  (initial_boats - shot_boats - remaining_boats) / initial_boats * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boats_eaten_by_fish_l3029_302959


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l3029_302985

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_waiting_time : ℕ
  total_active_shopping_time : ℕ
  total_trip_time : ℕ

/-- Calculates the time spent shopping and performing tasks -/
def time_shopping_and_tasks (trip : ShoppingTrip) : ℕ :=
  trip.total_trip_time - trip.total_waiting_time

/-- Theorem stating that the time spent shopping and performing tasks
    is equal to the total trip time minus the total waiting time -/
theorem shopping_time_calculation (trip : ShoppingTrip) 
  (h1 : trip.total_waiting_time = 58)
  (h2 : trip.total_active_shopping_time = 29)
  (h3 : trip.total_trip_time = 135) :
  time_shopping_and_tasks trip = 77 := by
  sorry

#eval time_shopping_and_tasks { total_waiting_time := 58, total_active_shopping_time := 29, total_trip_time := 135 }

end NUMINAMATH_CALUDE_shopping_time_calculation_l3029_302985


namespace NUMINAMATH_CALUDE_sum_of_seven_terms_l3029_302978

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_seven_terms (a : ℕ → ℝ) :
  arithmetic_sequence a → (a 3 + a 4 + a 5 = 12) → (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_terms_l3029_302978


namespace NUMINAMATH_CALUDE_age_difference_richard_david_l3029_302904

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrothersAges) : Prop :=
  ages.david > ages.scott ∧
  ages.richard > ages.david ∧
  ages.david = ages.scott + 8 ∧
  ages.david = 14 ∧
  ages.richard + 8 = 2 * (ages.scott + 8)

/-- The theorem to be proved -/
theorem age_difference_richard_david (ages : BrothersAges) :
  problem_conditions ages → ages.richard - ages.david = 6 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_richard_david_l3029_302904


namespace NUMINAMATH_CALUDE_substitution_insufficient_for_identity_proof_l3029_302928

/-- A mathematical identity is an equality that holds for all values of the variables involved. -/
def MathematicalIdentity (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x

/-- Substitution method verifies if an expression holds true for particular values. -/
def SubstitutionMethod (f g : ℝ → ℝ) (values : Set ℝ) : Prop :=
  ∀ x ∈ values, f x = g x

/-- Theorem: Substituting numerical values is insufficient to conclusively prove an identity. -/
theorem substitution_insufficient_for_identity_proof :
  ∃ (f g : ℝ → ℝ) (values : Set ℝ), 
    SubstitutionMethod f g values ∧ ¬MathematicalIdentity f g :=
  sorry

#check substitution_insufficient_for_identity_proof

end NUMINAMATH_CALUDE_substitution_insufficient_for_identity_proof_l3029_302928


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l3029_302957

theorem crazy_silly_school_series (num_books : ℕ) (movies_watched : ℕ) (books_read : ℕ) :
  num_books = 8 →
  movies_watched = 19 →
  books_read = 16 →
  movies_watched = books_read + 3 →
  ∃ (num_movies : ℕ), num_movies ≥ 19 :=
by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l3029_302957


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3029_302950

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x, (deriv f) x - 2 * f x < 0) ∧ 
  f 0 = 1

/-- The main theorem -/
theorem solution_set_characterization (f : ℝ → ℝ) (h : satisfies_conditions f) :
  ∀ x, f x > Real.exp (2 * x) ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3029_302950


namespace NUMINAMATH_CALUDE_same_solution_implies_a_and_b_l3029_302919

theorem same_solution_implies_a_and_b (a b : ℝ) :
  (∃ x y : ℝ, x - y = 0 ∧ 2*a*x + b*y = 4 ∧ 2*x + y = 3 ∧ a*x + b*y = 3) →
  a = 1 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_and_b_l3029_302919


namespace NUMINAMATH_CALUDE_power_sum_equation_l3029_302942

/-- Given two real numbers a and b satisfying certain conditions, prove that a^10 + b^10 = 123 -/
theorem power_sum_equation (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equation_l3029_302942


namespace NUMINAMATH_CALUDE_greg_ate_four_halves_l3029_302908

/-- Represents the number of whole cookies made -/
def total_cookies : ℕ := 14

/-- Represents the number of halves each cookie is cut into -/
def halves_per_cookie : ℕ := 2

/-- Represents the number of halves Brad ate -/
def brad_halves : ℕ := 6

/-- Represents the number of halves left -/
def left_halves : ℕ := 18

/-- Theorem stating that Greg ate 4 halves -/
theorem greg_ate_four_halves : 
  total_cookies * halves_per_cookie - brad_halves - left_halves = 4 := by
  sorry

end NUMINAMATH_CALUDE_greg_ate_four_halves_l3029_302908


namespace NUMINAMATH_CALUDE_permutation_game_winning_strategy_l3029_302916

/-- The game on permutation group S_n -/
def PermutationGame (n : ℕ) : Prop :=
  n > 1 ∧
  ∃ (strategy : ℕ → Bool),
    (n ≥ 4 ∧ Odd n → strategy n = false) ∧
    (n = 2 ∨ n = 3 → strategy n = true)

/-- Theorem stating the winning strategies for different values of n -/
theorem permutation_game_winning_strategy :
  ∀ n : ℕ, PermutationGame n :=
sorry

end NUMINAMATH_CALUDE_permutation_game_winning_strategy_l3029_302916


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l3029_302926

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 5

/-- The probability that five fair 6-sided dice won't all show the same number -/
theorem five_dice_not_same_probability :
  (1 - (num_sides : ℚ) / (num_sides ^ num_dice)) = 1295 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l3029_302926


namespace NUMINAMATH_CALUDE_closest_point_l3029_302922

def w (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 8*s
  | 1 => -2 + 6*s
  | 2 => -4 - 2*s

def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 5
  | 2 => 6

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 8
  | 1 => 6
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (s = 19/52) ↔ 
  (∀ t : ℝ, ‖w s - b‖ ≤ ‖w t - b‖) :=
sorry

end NUMINAMATH_CALUDE_closest_point_l3029_302922


namespace NUMINAMATH_CALUDE_even_function_property_l3029_302993

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the main theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_positive : ∀ x > 0, f x = x) : 
  ∀ x < 0, f x = -x :=
by
  sorry


end NUMINAMATH_CALUDE_even_function_property_l3029_302993


namespace NUMINAMATH_CALUDE_original_number_proof_l3029_302903

theorem original_number_proof : ∃ x : ℝ, 3 * (2 * x + 9) = 81 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3029_302903


namespace NUMINAMATH_CALUDE_first_job_men_l3029_302955

/-- The number of men who worked on the first job -/
def M : ℕ := 250

/-- The number of days for the first job -/
def days_job1 : ℕ := 16

/-- The number of men working on the second job -/
def men_job2 : ℕ := 600

/-- The number of days for the second job -/
def days_job2 : ℕ := 20

/-- The ratio of work between the second and first job -/
def work_ratio : ℕ := 3

theorem first_job_men :
  M * days_job1 * work_ratio = men_job2 * days_job2 := by
  sorry

#check first_job_men

end NUMINAMATH_CALUDE_first_job_men_l3029_302955


namespace NUMINAMATH_CALUDE_gcd_count_for_360_l3029_302988

theorem gcd_count_for_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), (∀ d ∈ S, ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) ∧ 
                      (∀ d : ℕ, (∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) → d ∈ S) ∧
                      S.card = 10) :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_360_l3029_302988


namespace NUMINAMATH_CALUDE_m_minus_n_equals_six_l3029_302977

theorem m_minus_n_equals_six (m n : ℤ) 
  (h1 : |m| = 2)
  (h2 : |n| = 4)
  (h3 : m > 0)
  (h4 : n < 0) :
  m - n = 6 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_six_l3029_302977


namespace NUMINAMATH_CALUDE_fast_pulsar_period_scientific_notation_l3029_302952

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fast_pulsar_period_scientific_notation :
  toScientificNotation 0.00519 = ScientificNotation.mk 5.19 (-3) sorry := by
  sorry

end NUMINAMATH_CALUDE_fast_pulsar_period_scientific_notation_l3029_302952


namespace NUMINAMATH_CALUDE_printing_machines_equation_l3029_302981

theorem printing_machines_equation (x : ℝ) : x > 0 → 
  (1000 / 15 : ℝ) + 1000 / x = 1000 / 5 ↔ 1 / 15 + 1 / x = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_printing_machines_equation_l3029_302981


namespace NUMINAMATH_CALUDE_remainder_problem_l3029_302968

theorem remainder_problem (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3029_302968


namespace NUMINAMATH_CALUDE_cost_for_23_days_l3029_302963

/-- Calculates the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 13
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Proves that the cost for a 23-day stay is $334.00 -/
theorem cost_for_23_days : hostelCost 23 = 334 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_cost_for_23_days_l3029_302963


namespace NUMINAMATH_CALUDE_triangle_ab_length_l3029_302939

/-- Given a triangle ABC with angles B and C both 45 degrees and side BC of length 10,
    prove that the length of side AB is 5√2. -/
theorem triangle_ab_length (A B C : ℝ × ℝ) : 
  let triangle := (A, B, C)
  let angle (X Y Z : ℝ × ℝ) := Real.arccos ((X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2)) / 
    (((X.1 - Y.1)^2 + (X.2 - Y.2)^2).sqrt * ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2).sqrt)
  let distance (X Y : ℝ × ℝ) := ((X.1 - Y.1)^2 + (X.2 - Y.2)^2).sqrt
  angle B A C = π/4 →
  angle C B A = π/4 →
  distance B C = 10 →
  distance A B = 5 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_ab_length_l3029_302939


namespace NUMINAMATH_CALUDE_circle_M_equation_l3029_302994

-- Define the circle M
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
axiom center_on_line (M : CircleM) : M.center.2 = -2 * M.center.1

axiom passes_through_A (M : CircleM) :
  (2 - M.center.1)^2 + (-1 - M.center.2)^2 = M.radius^2

axiom tangent_to_line (M : CircleM) :
  |M.center.1 + M.center.2 - 1| / Real.sqrt 2 = M.radius

-- Define the theorem to be proved
theorem circle_M_equation (M : CircleM) :
  ∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2 ↔
    (x - M.center.1)^2 + (y - M.center.2)^2 = M.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l3029_302994


namespace NUMINAMATH_CALUDE_dog_bird_time_difference_l3029_302925

def dogs : ℕ := 3
def dog_hours : ℕ := 7
def holes : ℕ := 9
def birds : ℕ := 5
def bird_minutes : ℕ := 40
def nests : ℕ := 2

def dog_dig_time : ℚ := (dog_hours * 60 : ℚ) * holes / dogs
def bird_build_time : ℚ := (bird_minutes : ℚ) * birds / nests

theorem dog_bird_time_difference :
  dog_dig_time - bird_build_time = 40 := by sorry

end NUMINAMATH_CALUDE_dog_bird_time_difference_l3029_302925


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3029_302909

theorem complex_equation_solution (i : ℂ) (n : ℝ) 
  (h1 : i * i = -1) 
  (h2 : (2 : ℂ) / (1 - i) = 1 + n * i) : 
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3029_302909


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l3029_302907

theorem complex_roots_quadratic (a b : ℝ) : 
  (∃ z₁ z₂ : ℂ, z₁ = a + 3*I ∧ z₂ = b + 7*I ∧ 
   z₁^2 - (10 + 10*I)*z₁ + (70 + 16*I) = 0 ∧
   z₂^2 - (10 + 10*I)*z₂ + (70 + 16*I) = 0) →
  a = -3.5 ∧ b = 13.5 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l3029_302907


namespace NUMINAMATH_CALUDE_final_digit_is_two_l3029_302934

/-- Represents the state of the board with counts of zeros, ones, and twos -/
structure BoardState where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents an operation on the board -/
inductive Operation
  | ZeroOne   -- Erase 0 and 1, write 2
  | ZeroTwo   -- Erase 0 and 2, write 1
  | OneTwo    -- Erase 1 and 2, write 0

/-- Applies an operation to the board state -/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.ZeroOne => ⟨state.zeros - 1, state.ones - 1, state.twos + 1⟩
  | Operation.ZeroTwo => ⟨state.zeros - 1, state.ones + 1, state.twos - 1⟩
  | Operation.OneTwo => ⟨state.zeros + 1, state.ones - 1, state.twos - 1⟩

/-- Checks if the board state has only one digit remaining -/
def isFinalState (state : BoardState) : Bool :=
  (state.zeros + state.ones + state.twos = 1)

/-- Theorem: The final digit is always 2, regardless of the order of operations -/
theorem final_digit_is_two (initialState : BoardState) (ops : List Operation) :
  isFinalState (ops.foldl applyOperation initialState) →
  (ops.foldl applyOperation initialState).twos = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_digit_is_two_l3029_302934


namespace NUMINAMATH_CALUDE_jon_website_earnings_l3029_302954

/-- Calculates Jon's earnings from his website in a 30-day month -/
theorem jon_website_earnings : 
  let pay_per_visit : ℚ := 0.1
  let visits_per_hour : ℕ := 50
  let hours_per_day : ℕ := 24
  let days_in_month : ℕ := 30
  (pay_per_visit * visits_per_hour * hours_per_day * days_in_month : ℚ) = 3600 := by
  sorry

end NUMINAMATH_CALUDE_jon_website_earnings_l3029_302954


namespace NUMINAMATH_CALUDE_problem_statement_l3029_302982

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h1 : a + 2 / b = b + 2 / c) (h2 : b + 2 / c = c + 2 / a) :
  (a + 2 / b)^2 + (b + 2 / c)^2 + (c + 2 / a)^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3029_302982


namespace NUMINAMATH_CALUDE_min_nickels_needed_l3029_302983

def book_cost : ℚ := 27.37
def twenty_bill : ℚ := 20
def five_bill : ℚ := 5
def quarter_value : ℚ := 0.25
def nickel_value : ℚ := 0.05
def num_quarters : ℕ := 5

theorem min_nickels_needed :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), 
      twenty_bill + five_bill + num_quarters * quarter_value + m * nickel_value ≥ book_cost 
      → m ≥ n) ∧
    (twenty_bill + five_bill + num_quarters * quarter_value + n * nickel_value ≥ book_cost) ∧
    n = 23 :=
by sorry

end NUMINAMATH_CALUDE_min_nickels_needed_l3029_302983


namespace NUMINAMATH_CALUDE_distribute_6_3_l3029_302997

/-- The number of ways to distribute n items among k categories, 
    with each category receiving at least one item. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 10 ways to distribute 6 items among 3 categories, 
    with each category receiving at least one item. -/
theorem distribute_6_3 : distribute 6 3 = 10 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_l3029_302997


namespace NUMINAMATH_CALUDE_other_number_proof_l3029_302984

theorem other_number_proof (a b : ℕ+) (h1 : Nat.lcm a b = 2310) (h2 : Nat.gcd a b = 30) (h3 : a = 231) : b = 300 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3029_302984


namespace NUMINAMATH_CALUDE_quadrilateral_area_relations_integer_areas_perfect_square_product_l3029_302946

/-- Given a convex quadrilateral ABCD with diagonals intersecting at point P,
    S_ABP, S_BCP, S_CDP, and S_ADP are the areas of triangles ABP, BCP, CDP, and ADP respectively. -/
def QuadrilateralAreas (S_ABP S_BCP S_CDP S_ADP : ℝ) : Prop :=
  S_ABP > 0 ∧ S_BCP > 0 ∧ S_CDP > 0 ∧ S_ADP > 0

theorem quadrilateral_area_relations
  (S_ABP S_BCP S_CDP S_ADP : ℝ)
  (h : QuadrilateralAreas S_ABP S_BCP S_CDP S_ADP) :
  S_ADP = (S_ABP * S_CDP) / S_BCP ∧
  S_ABP * S_BCP * S_CDP * S_ADP = (S_ADP * S_BCP)^2 := by
  sorry

/-- If the areas of the four triangles are integers, their product is a perfect square. -/
theorem integer_areas_perfect_square_product
  (S_ABP S_BCP S_CDP S_ADP : ℤ)
  (h : QuadrilateralAreas (S_ABP : ℝ) (S_BCP : ℝ) (S_CDP : ℝ) (S_ADP : ℝ)) :
  ∃ (n : ℤ), S_ABP * S_BCP * S_CDP * S_ADP = n^2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_relations_integer_areas_perfect_square_product_l3029_302946


namespace NUMINAMATH_CALUDE_number_ratio_proof_l3029_302905

theorem number_ratio_proof (N P : ℚ) (h1 : N = 280) (h2 : (1/5) * N + 7 = P - 7) :
  (P - 7) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_proof_l3029_302905


namespace NUMINAMATH_CALUDE_distance_O_to_B_l3029_302902

/-- Represents a person moving on the road -/
structure Person where
  name : String
  startPosition : ℝ
  speed : ℝ

/-- Represents the road with three locations -/
structure Road where
  a : ℝ
  o : ℝ
  b : ℝ

/-- The main theorem stating the distance between O and B -/
theorem distance_O_to_B 
  (road : Road)
  (jia yi : Person)
  (h1 : road.o = 0)  -- Set O as the origin
  (h2 : road.a = -1360)  -- A is 1360 meters to the left of O
  (h3 : jia.startPosition = road.a)
  (h4 : yi.startPosition = road.o)
  (h5 : jia.speed * 10 + road.a = yi.speed * 10)  -- Equidistant at 10 minutes
  (h6 : jia.speed * 40 + road.a = road.b)  -- Meet at B at 40 minutes
  (h7 : yi.speed * 40 = road.b)  -- Yi also reaches B at 40 minutes
  : road.b = 2040 := by
  sorry


end NUMINAMATH_CALUDE_distance_O_to_B_l3029_302902


namespace NUMINAMATH_CALUDE_sum_of_four_twos_to_fourth_l3029_302927

theorem sum_of_four_twos_to_fourth (n : ℕ) : 
  (2^4 : ℕ) + (2^4 : ℕ) + (2^4 : ℕ) + (2^4 : ℕ) = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_twos_to_fourth_l3029_302927


namespace NUMINAMATH_CALUDE_last_three_average_l3029_302913

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 6 →
  numbers.sum / 6 = 60 →
  (numbers.take 3).sum / 3 = 55 →
  (numbers.drop 3).sum = 195 →
  (numbers.drop 3).sum / 3 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l3029_302913


namespace NUMINAMATH_CALUDE_sum_of_largest_odd_factors_l3029_302965

/-- The largest odd factor of a natural number -/
def largest_odd_factor (n : ℕ) : ℕ := sorry

/-- The sum of the first n terms of the sequence of largest odd factors -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem: The sum of the first 2^2016 - 1 terms of the sequence
    of largest odd factors is equal to (4^2016 - 1) / 3 -/
theorem sum_of_largest_odd_factors :
  S (2^2016 - 1) = (4^2016 - 1) / 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_largest_odd_factors_l3029_302965


namespace NUMINAMATH_CALUDE_quadratic_properties_l3029_302995

def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem quadratic_properties :
  ∀ (a b m : ℝ),
  (quadratic_function a b 2 = 0) →
  (quadratic_function a b 1 = m) →
  (
    (m = 3 → a = -2 ∧ b = 3) ∧
    (m = 3 → ∀ x, -1 ≤ x ∧ x ≤ 2 → -3 ≤ quadratic_function a b x ∧ quadratic_function a b x ≤ 25/8) ∧
    (a > 0 → m < 1)
  ) := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3029_302995


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l3029_302992

theorem min_sum_absolute_values : ∀ x : ℝ, 
  |x + 3| + |x + 5| + |x + 6| ≥ 5 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l3029_302992


namespace NUMINAMATH_CALUDE_students_not_enrolled_l3029_302930

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 94) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l3029_302930


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3029_302996

-- Problem 1
theorem problem_1 : 
  (2 + 1/4)^(1/2) + (-3.8)^0 - Real.sqrt 3 * (3/2)^(1/3) * (12^(1/6)) = -1/2 := by sorry

-- Problem 2
theorem problem_2 : 
  2 * (Real.log 2 / Real.log 3) - Real.log (32/9) / Real.log 3 + Real.log 8 / Real.log 3 - 
  (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3029_302996


namespace NUMINAMATH_CALUDE_andrei_apple_spending_l3029_302979

/-- Calculates Andrei's monthly spending on apples after price increase and discount -/
def andreiMonthlySpending (originalPrice : ℚ) (priceIncrease : ℚ) (discount : ℚ) (kgPerMonth : ℚ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease)
  let discountedPrice := newPrice * (1 - discount)
  discountedPrice * kgPerMonth

/-- Theorem stating that Andrei's monthly spending on apples is 99 rubles -/
theorem andrei_apple_spending :
  andreiMonthlySpending 50 (1/10) (1/10) 2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_andrei_apple_spending_l3029_302979


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3029_302975

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 + 0.000012 = 3858 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l3029_302975


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3029_302960

def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 4}
def B : Set ℤ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3029_302960


namespace NUMINAMATH_CALUDE_perimeter_of_20_rectangles_l3029_302962

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Creates a list of rectangles following the given pattern -/
def createRectangles (n : ℕ) : List Rectangle :=
  List.range n |>.map (fun i => ⟨i + 1, i + 2⟩)

/-- Calculates the perimeter of a polygon formed by arranging rectangles -/
def polygonPerimeter (rectangles : List Rectangle) : ℕ :=
  sorry

theorem perimeter_of_20_rectangles :
  let rectangles := createRectangles 20
  polygonPerimeter rectangles = 462 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_20_rectangles_l3029_302962


namespace NUMINAMATH_CALUDE_expression_simplification_l3029_302941

theorem expression_simplification (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 2 * a * b - a^2 ≠ 0) :
  (a^2 - 2*a*b + b^2) / (a*b) - (2*a*b - b^2) / (2*a*b - a^2) = (a^2 - 2*a*b + 2*b^2) / (a*b) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3029_302941


namespace NUMINAMATH_CALUDE_floor_inequality_l3029_302910

theorem floor_inequality (x y : ℝ) : 
  ⌊2*x⌋ + ⌊2*y⌋ ≥ ⌊x⌋ + ⌊y⌋ + ⌊x + y⌋ :=
sorry

end NUMINAMATH_CALUDE_floor_inequality_l3029_302910


namespace NUMINAMATH_CALUDE_height_difference_ruby_xavier_l3029_302986

-- Define heights as natural numbers (in centimeters)
def janet_height : ℕ := 62
def charlene_height : ℕ := 2 * janet_height
def pablo_height : ℕ := charlene_height + 70
def ruby_height : ℕ := pablo_height - 2
def xavier_height : ℕ := charlene_height + 84
def paul_height : ℕ := ruby_height + 45

-- Theorem statement
theorem height_difference_ruby_xavier : 
  xavier_height - ruby_height = 7 := by sorry

end NUMINAMATH_CALUDE_height_difference_ruby_xavier_l3029_302986


namespace NUMINAMATH_CALUDE_order_of_rational_numbers_l3029_302933

theorem order_of_rational_numbers (a b : ℚ) 
  (ha : a > 0) (hb : b < 0) (hab : |a| < |b|) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end NUMINAMATH_CALUDE_order_of_rational_numbers_l3029_302933


namespace NUMINAMATH_CALUDE_juan_number_puzzle_l3029_302920

theorem juan_number_puzzle (n : ℝ) : ((((n + 2) * 2) - 2) / 2) = 7 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_juan_number_puzzle_l3029_302920


namespace NUMINAMATH_CALUDE_farm_sheep_count_l3029_302990

/-- Represents the farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  racehorses : ℕ
  draft_horses : ℕ

/-- The ratio of sheep to total horses is 7:8 -/
def sheep_horse_ratio (f : Farm) : Prop :=
  7 * (f.racehorses + f.draft_horses) = 8 * f.sheep

/-- Total horse food consumption per day -/
def total_horse_food (f : Farm) : ℕ :=
  250 * f.racehorses + 300 * f.draft_horses

/-- There is 1/3 more racehorses than draft horses -/
def racehorse_draft_ratio (f : Farm) : Prop :=
  f.racehorses = f.draft_horses + (f.draft_horses / 3)

/-- The farm satisfies all given conditions -/
def valid_farm (f : Farm) : Prop :=
  sheep_horse_ratio f ∧
  total_horse_food f = 21000 ∧
  racehorse_draft_ratio f

theorem farm_sheep_count :
  ∃ f : Farm, valid_farm f ∧ f.sheep = 67 :=
sorry

end NUMINAMATH_CALUDE_farm_sheep_count_l3029_302990


namespace NUMINAMATH_CALUDE_no_divisible_sum_difference_l3029_302915

theorem no_divisible_sum_difference : 
  ¬∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ 
  ((∃ k : ℤ, A = k * (A + B) ∧ ∃ m : ℤ, B = m * (A - B)) ∨
   (∃ k : ℤ, B = k * (A + B) ∧ ∃ m : ℤ, A = m * (A - B))) :=
by sorry

end NUMINAMATH_CALUDE_no_divisible_sum_difference_l3029_302915


namespace NUMINAMATH_CALUDE_min_value_expression_l3029_302976

theorem min_value_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a + 2 * b = 1) :
  ∃ (m : ℝ), m = 2/3 ∧ ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 3 * x + 2 * y = 1 → 
    1 / (12 * x + 1) + 1 / (8 * y + 1) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3029_302976


namespace NUMINAMATH_CALUDE_max_mondays_in_51_days_l3029_302931

theorem max_mondays_in_51_days : ∀ (start_day : Nat),
  (start_day < 7) →
  (∃ (monday_count : Nat),
    monday_count = (51 / 7 : Nat) + (if start_day ≤ 1 then 1 else 0) ∧
    monday_count ≤ 8 ∧
    ∀ (other_count : Nat),
      (∃ (other_start : Nat), other_start < 7 ∧
        other_count = (51 / 7 : Nat) + (if other_start ≤ 1 then 1 else 0)) →
      other_count ≤ monday_count) :=
by sorry

end NUMINAMATH_CALUDE_max_mondays_in_51_days_l3029_302931


namespace NUMINAMATH_CALUDE_cloth_selling_price_l3029_302964

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Theorem stating that the total selling price for 85 meters of cloth with a profit of 10 Rs per meter and a cost price of 95 Rs per meter is 8925 Rs. -/
theorem cloth_selling_price :
  totalSellingPrice 85 10 95 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l3029_302964


namespace NUMINAMATH_CALUDE_last_three_average_l3029_302991

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 4).sum / 4 = 54 →
  (list.drop 4).sum / 3 = 72.67 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l3029_302991


namespace NUMINAMATH_CALUDE_even_quadratic_function_l3029_302938

/-- A quadratic function f(x) = ax^2 + (2a^2 - a)x + 1 is even if and only if a = 1/2 -/
theorem even_quadratic_function (a : ℝ) :
  (∀ x, (a * x^2 + (2 * a^2 - a) * x + 1) = (a * (-x)^2 + (2 * a^2 - a) * (-x) + 1)) ↔
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l3029_302938


namespace NUMINAMATH_CALUDE_construction_time_theorem_l3029_302921

/-- Represents the time taken to construct a wall given the number of boys and girls -/
def constructionTime (boys : ℕ) (girls : ℕ) : ℝ :=
  sorry

/-- Theorem stating that if 16 boys or 24 girls can construct a wall in 6 days,
    then 8 boys and 4 girls will take 12 days to construct the same wall -/
theorem construction_time_theorem :
  (constructionTime 16 0 = 6 ∧ constructionTime 0 24 = 6) →
  constructionTime 8 4 = 12 :=
sorry

end NUMINAMATH_CALUDE_construction_time_theorem_l3029_302921


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_specific_value_2A_minus_3B_l3029_302989

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := 3 * b^2 - 2 * a^2 + 5 * a * b

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := 4 * a * b + 2 * b^2 - a^2

/-- Theorem stating that 2A - 3B simplifies to -a² - 2ab for all real a and b -/
theorem simplify_2A_minus_3B (a b : ℝ) : 2 * A a b - 3 * B a b = -a^2 - 2*a*b := by
  sorry

/-- Theorem stating that when a = -1 and b = 4, 2A - 3B equals 7 -/
theorem specific_value_2A_minus_3B : 2 * A (-1) 4 - 3 * B (-1) 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_specific_value_2A_minus_3B_l3029_302989


namespace NUMINAMATH_CALUDE_sum_of_divisors_30_l3029_302951

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_30 : sum_of_divisors 30 = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_30_l3029_302951


namespace NUMINAMATH_CALUDE_octopus_family_total_l3029_302900

/-- Represents the number of children of each color in the octopus family -/
structure OctopusFamily :=
  (white : ℕ)
  (blue : ℕ)
  (striped : ℕ)

/-- The conditions of the octopus family problem -/
def octopusFamilyConditions (initial final : OctopusFamily) : Prop :=
  -- Initially, there were equal numbers of white, blue, and striped octopus children
  initial.white = initial.blue ∧ initial.blue = initial.striped
  -- Some blue octopus children became striped
  ∧ final.blue < initial.blue
  ∧ final.striped > initial.striped
  -- After the change, the total number of blue and white octopus children was 10
  ∧ final.blue + final.white = 10
  -- After the change, the total number of white and striped octopus children was 18
  ∧ final.white + final.striped = 18
  -- The total number of children remains constant
  ∧ initial.white + initial.blue + initial.striped = final.white + final.blue + final.striped

/-- The theorem stating that under the given conditions, the total number of children is 21 -/
theorem octopus_family_total (initial final : OctopusFamily) :
  octopusFamilyConditions initial final →
  final.white + final.blue + final.striped = 21 :=
by
  sorry


end NUMINAMATH_CALUDE_octopus_family_total_l3029_302900


namespace NUMINAMATH_CALUDE_smallest_square_area_l3029_302973

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The smallest square that can contain two non-overlapping rectangles -/
def smallest_containing_square (r1 r2 : Rectangle) : ℕ := 
  max (r1.width + r2.width) (max r1.height r2.height)

/-- Theorem: The smallest square containing a 3×5 and a 4×6 rectangle has area 49 -/
theorem smallest_square_area : 
  let r1 : Rectangle := ⟨3, 5⟩
  let r2 : Rectangle := ⟨4, 6⟩
  (smallest_containing_square r1 r2)^2 = 49 := by
sorry

#eval (smallest_containing_square ⟨3, 5⟩ ⟨4, 6⟩)^2

end NUMINAMATH_CALUDE_smallest_square_area_l3029_302973


namespace NUMINAMATH_CALUDE_zoo_arrangements_l3029_302970

/-- The number of letters in the word "ZOO₁M₁O₂M₂O₃" -/
def word_length : ℕ := 7

/-- The number of distinct arrangements of the letters in "ZOO₁M₁O₂M₂O₃" -/
def num_arrangements : ℕ := Nat.factorial word_length

theorem zoo_arrangements :
  num_arrangements = 5040 := by sorry

end NUMINAMATH_CALUDE_zoo_arrangements_l3029_302970


namespace NUMINAMATH_CALUDE_square_sum_implies_fourth_power_sum_l3029_302974

theorem square_sum_implies_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 7) : 
  r^4 + 1/r^4 = 23 := by
sorry

end NUMINAMATH_CALUDE_square_sum_implies_fourth_power_sum_l3029_302974


namespace NUMINAMATH_CALUDE_limit_special_function_l3029_302917

/-- The limit of (x^2 + 2x - 3) / (x^2 + 4x - 5) raised to the power of 1 / (2-x) as x approaches 1 is equal to 2/3 -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(((x^2 + 2*x - 3) / (x^2 + 4*x - 5))^(1/(2-x))) - (2/3)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_special_function_l3029_302917


namespace NUMINAMATH_CALUDE_football_lineup_combinations_l3029_302918

def team_size : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

def lineup_combinations : ℕ := 31680

theorem football_lineup_combinations :
  team_size = 12 ∧ 
  offensive_linemen = 4 ∧ 
  positions = 5 →
  lineup_combinations = offensive_linemen * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) :=
by sorry

end NUMINAMATH_CALUDE_football_lineup_combinations_l3029_302918


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3029_302923

theorem geometric_sequence_ratio_sum (m a₂ a₃ b₂ b₃ x y : ℝ) 
  (hm : m ≠ 0)
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (hxy : x ≠ y)
  (ha₂ : a₂ = m * x)
  (ha₃ : a₃ = m * x^2)
  (hb₂ : b₂ = m * y)
  (hb₃ : b₃ = m * y^2)
  (heq : a₃ - b₃ = 3 * (a₂ - b₂)) :
  x + y = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3029_302923


namespace NUMINAMATH_CALUDE_red_balls_count_l3029_302987

/-- The number of red balls in a box with specific conditions -/
def num_red_balls (total : ℕ) (blue : ℕ) : ℕ :=
  let green := 3 * blue
  let red := (total - blue - green) / 3
  red

/-- Theorem stating that the number of red balls is 4 under given conditions -/
theorem red_balls_count :
  num_red_balls 36 6 = 4 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l3029_302987


namespace NUMINAMATH_CALUDE_student_selection_l3029_302943

theorem student_selection (male_count : Nat) (female_count : Nat) :
  male_count = 5 →
  female_count = 4 →
  (Nat.choose (male_count + female_count) 3 -
   Nat.choose male_count 3 -
   Nat.choose female_count 3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_l3029_302943


namespace NUMINAMATH_CALUDE_ratio_p_to_r_l3029_302945

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 4 / 3)
  (h3 : s / q = 1 / 5) :
  p / r = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_p_to_r_l3029_302945


namespace NUMINAMATH_CALUDE_library_books_l3029_302911

theorem library_books (initial_books : ℕ) : 
  initial_books - 124 + 22 = 234 → initial_books = 336 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l3029_302911


namespace NUMINAMATH_CALUDE_baking_time_ratio_l3029_302949

def usual_assembly_time : ℝ := 1
def usual_baking_time : ℝ := 1.5
def usual_decorating_time : ℝ := 1
def total_time_on_failed_day : ℝ := 5

theorem baking_time_ratio :
  let usual_total_time := usual_assembly_time + usual_baking_time + usual_decorating_time
  let baking_time_on_failed_day := total_time_on_failed_day - usual_assembly_time - usual_decorating_time
  baking_time_on_failed_day / usual_baking_time = 2 := by
sorry

end NUMINAMATH_CALUDE_baking_time_ratio_l3029_302949


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3029_302947

/-- The asymptotes of the hyperbola x²/9 - y²/16 = 1 are given by y = ±(4/3)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → ℝ := λ x y => x^2 / 9 - y^2 / 16 - 1
  ∃ (k : ℝ), k > 0 ∧ ∀ (x y : ℝ), h x y = 0 → y = k * x ∨ y = -k * x :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3029_302947


namespace NUMINAMATH_CALUDE_rotation_of_point_A_l3029_302914

-- Define the rotation function
def rotate_clockwise_90 (x y : ℝ) : ℝ × ℝ := (y, -x)

-- Define the theorem
theorem rotation_of_point_A : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := rotate_clockwise_90 A.1 A.2
  B = (1, -2) := by sorry

end NUMINAMATH_CALUDE_rotation_of_point_A_l3029_302914


namespace NUMINAMATH_CALUDE_infinitely_many_special_integers_l3029_302912

theorem infinitely_many_special_integers (k : ℕ) (hk : k > 1) :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ x ∈ S, 
    (∃ (a b : ℕ), x = a^k - b^k) ∧ 
    (¬∃ (c d : ℕ), x = c^k + d^k)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_integers_l3029_302912


namespace NUMINAMATH_CALUDE_dividend_divisor_calculation_l3029_302999

/-- Given a dividend of 73648 and a divisor of 874, prove that the result of subtracting
    the product of the divisor and the sum of the quotient's digits from the dividend
    is equal to 63160. -/
theorem dividend_divisor_calculation : 
  let dividend : Nat := 73648
  let divisor : Nat := 874
  let quotient : Nat := dividend / divisor
  let remainder : Nat := dividend % divisor
  let sum_of_digits : Nat := (quotient / 10) + (quotient % 10)
  73648 - (sum_of_digits * 874) = 63160 := by
  sorry

#eval 73648 - ((73648 / 874 / 10 + 73648 / 874 % 10) * 874)

end NUMINAMATH_CALUDE_dividend_divisor_calculation_l3029_302999


namespace NUMINAMATH_CALUDE_y_coordinate_of_P_l3029_302966

/-- The y-coordinate of point P given specific conditions -/
theorem y_coordinate_of_P (A B C D P : ℝ × ℝ) : 
  A = (-4, 0) →
  B = (-3, 2) →
  C = (3, 2) →
  D = (4, 0) →
  dist P A + dist P D = 10 →
  dist P B + dist P C = 10 →
  P.2 = 6/7 := by
  sorry

#check y_coordinate_of_P

end NUMINAMATH_CALUDE_y_coordinate_of_P_l3029_302966


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3029_302961

-- Define the given constants
def train_length : ℝ := 390  -- in meters
def man_speed : ℝ := 2       -- in km/h
def crossing_time : ℝ := 52  -- in seconds

-- Define the theorem
theorem train_speed_calculation :
  ∃ (train_speed : ℝ),
    train_speed > 0 ∧
    train_speed = 25 ∧
    (train_speed + man_speed) * (crossing_time / 3600) = train_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3029_302961


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3029_302956

theorem quadratic_equation_roots (m : ℝ) :
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3029_302956


namespace NUMINAMATH_CALUDE_woods_width_l3029_302937

theorem woods_width (area : ℝ) (length : ℝ) (width : ℝ) 
  (h1 : area = 24) 
  (h2 : length = 3) 
  (h3 : area = length * width) : width = 8 := by
sorry

end NUMINAMATH_CALUDE_woods_width_l3029_302937


namespace NUMINAMATH_CALUDE_february_bill_increase_l3029_302901

def january_bill : ℝ := 179.99999999999991

theorem february_bill_increase (february_bill : ℝ) 
  (h1 : february_bill / january_bill = 3 / 2) 
  (h2 : ∃ (increased_bill : ℝ), increased_bill / january_bill = 5 / 3) : 
  ∃ (increased_bill : ℝ), increased_bill - february_bill = 30 :=
sorry

end NUMINAMATH_CALUDE_february_bill_increase_l3029_302901


namespace NUMINAMATH_CALUDE_razorback_shop_profit_l3029_302948

/-- The amount the shop makes off each jersey in dollars. -/
def jersey_profit : ℕ := 34

/-- The additional cost of a t-shirt compared to a jersey in dollars. -/
def tshirt_additional_cost : ℕ := 158

/-- The amount the shop makes off each t-shirt in dollars. -/
def tshirt_profit : ℕ := jersey_profit + tshirt_additional_cost

theorem razorback_shop_profit : tshirt_profit = 192 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_profit_l3029_302948


namespace NUMINAMATH_CALUDE_volume_expanded_parallelepiped_eq_l3029_302998

/-- The volume of a set of points inside or within one unit of a 2x3x4 rectangular parallelepiped -/
def volume_expanded_parallelepiped : ℝ := sorry

/-- The dimension of the parallelepiped along the x-axis -/
def x_dim : ℝ := 2

/-- The dimension of the parallelepiped along the y-axis -/
def y_dim : ℝ := 3

/-- The dimension of the parallelepiped along the z-axis -/
def z_dim : ℝ := 4

/-- The radius of the expanded region around the parallelepiped -/
def expansion_radius : ℝ := 1

theorem volume_expanded_parallelepiped_eq :
  volume_expanded_parallelepiped = (228 + 31 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_expanded_parallelepiped_eq_l3029_302998


namespace NUMINAMATH_CALUDE_min_x_plus_y_l3029_302958

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l3029_302958


namespace NUMINAMATH_CALUDE_product_of_three_rationals_l3029_302971

theorem product_of_three_rationals (a b c : ℚ) :
  a * b * c < 0 → (a < 0 ∧ b ≥ 0 ∧ c ≥ 0) ∨
                   (a ≥ 0 ∧ b < 0 ∧ c ≥ 0) ∨
                   (a ≥ 0 ∧ b ≥ 0 ∧ c < 0) ∨
                   (a < 0 ∧ b < 0 ∧ c < 0) :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_rationals_l3029_302971


namespace NUMINAMATH_CALUDE_quadratic_root_bound_l3029_302935

theorem quadratic_root_bound (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_real_roots : ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_bound_l3029_302935


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l3029_302906

theorem binomial_equation_solution (x : ℕ) : 
  (Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4) → (x = 5 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l3029_302906


namespace NUMINAMATH_CALUDE_root_difference_squared_l3029_302936

theorem root_difference_squared (p q : ℚ) : 
  (6 * p^2 - 7 * p - 20 = 0) → 
  (6 * q^2 - 7 * q - 20 = 0) → 
  (p - q)^2 = 529 / 36 := by
sorry

end NUMINAMATH_CALUDE_root_difference_squared_l3029_302936


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3029_302972

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3029_302972


namespace NUMINAMATH_CALUDE_paving_rate_calculation_l3029_302953

/-- Calculates the rate of paving per square meter given room dimensions and total cost -/
theorem paving_rate_calculation (length width total_cost : ℝ) :
  length = 5.5 ∧ width = 4 ∧ total_cost = 17600 →
  total_cost / (length * width) = 800 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_calculation_l3029_302953


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3029_302969

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3029_302969


namespace NUMINAMATH_CALUDE_jessica_has_62_marbles_l3029_302967

-- Define the number of marbles each person has
def dennis_marbles : ℕ := 70
def kurt_marbles : ℕ := dennis_marbles - 45
def laurie_marbles : ℕ := kurt_marbles + 12
def jessica_marbles : ℕ := laurie_marbles + 25

-- Theorem to prove
theorem jessica_has_62_marbles : jessica_marbles = 62 := by
  sorry

end NUMINAMATH_CALUDE_jessica_has_62_marbles_l3029_302967


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3029_302940

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3029_302940


namespace NUMINAMATH_CALUDE_cement_mixture_percentage_l3029_302929

/-- Calculates the percentage of cement in the second mixture for concrete production --/
theorem cement_mixture_percentage 
  (total_concrete : Real) 
  (final_cement_percentage : Real)
  (first_mixture_percentage : Real)
  (second_mixture_amount : Real) :
  let total_cement := total_concrete * final_cement_percentage / 100
  let first_mixture_amount := total_concrete - second_mixture_amount
  let first_mixture_cement := first_mixture_amount * first_mixture_percentage / 100
  let second_mixture_cement := total_cement - first_mixture_cement
  second_mixture_cement / second_mixture_amount * 100 = 80 :=
by
  sorry

#check cement_mixture_percentage 10 62 20 7

end NUMINAMATH_CALUDE_cement_mixture_percentage_l3029_302929


namespace NUMINAMATH_CALUDE_inequality_solution_l3029_302932

theorem inequality_solution (x : ℝ) : (x + 1) / x > 1 ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3029_302932


namespace NUMINAMATH_CALUDE_probability_two_pairs_is_5_21_l3029_302924

-- Define the total number of socks and colors
def total_socks : ℕ := 10
def num_colors : ℕ := 5
def socks_per_color : ℕ := 2
def socks_drawn : ℕ := 5

-- Define the probability function
def probability_two_pairs : ℚ :=
  let total_combinations := Nat.choose total_socks socks_drawn
  let favorable_combinations := Nat.choose num_colors 2 * Nat.choose (num_colors - 2) 1 * socks_per_color
  (favorable_combinations : ℚ) / total_combinations

-- Theorem statement
theorem probability_two_pairs_is_5_21 : 
  probability_two_pairs = 5 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_two_pairs_is_5_21_l3029_302924
