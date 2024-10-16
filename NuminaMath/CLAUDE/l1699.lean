import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1699_169921

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  a^2 / (b + c + d) + b^2 / (c + d + a) + c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1699_169921


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1699_169964

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 10 = 16 → a 4 + a 6 + a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1699_169964


namespace NUMINAMATH_CALUDE_fabric_C_required_is_120_l1699_169988

/-- Calculates the amount of fabric C required for pants production every week -/
def fabric_C_required (
  kingsley_pants_per_day : ℕ)
  (kingsley_work_days : ℕ)
  (fabric_C_per_pants : ℕ) : ℕ :=
  kingsley_pants_per_day * kingsley_work_days * fabric_C_per_pants

/-- Proves that the amount of fabric C required for pants production every week is 120 yards -/
theorem fabric_C_required_is_120 :
  fabric_C_required 4 6 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_fabric_C_required_is_120_l1699_169988


namespace NUMINAMATH_CALUDE_triangle_area_product_l1699_169991

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (1/2 * (8/a) * (8/b) = 8) → a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_product_l1699_169991


namespace NUMINAMATH_CALUDE_cyclist_meeting_oncoming_buses_l1699_169979

/-- The time interval between a cyclist meeting oncoming buses, given constant speeds and specific time intervals -/
theorem cyclist_meeting_oncoming_buses 
  (overtake_interval : ℝ) 
  (bus_interval : ℝ) 
  (h1 : overtake_interval > 0)
  (h2 : bus_interval > 0)
  (h3 : bus_interval = overtake_interval / 2) :
  overtake_interval / 2 = bus_interval := by
sorry

end NUMINAMATH_CALUDE_cyclist_meeting_oncoming_buses_l1699_169979


namespace NUMINAMATH_CALUDE_correct_lineup_count_l1699_169947

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of players in the starting lineup
def lineup_size : ℕ := 6

-- Define the number of guaranteed players (All-Stars)
def guaranteed_players : ℕ := 3

-- Define the number of goalkeepers
def goalkeepers : ℕ := 1

-- Define the function to calculate the number of possible lineups
def possible_lineups : ℕ := Nat.choose (total_players - guaranteed_players - goalkeepers) (lineup_size - guaranteed_players - goalkeepers)

-- Theorem statement
theorem correct_lineup_count : possible_lineups = 55 := by
  sorry

end NUMINAMATH_CALUDE_correct_lineup_count_l1699_169947


namespace NUMINAMATH_CALUDE_first_part_length_l1699_169961

/-- Given a trip with the following conditions:
  * Total distance is 50 km
  * First part is traveled at 66 km/h
  * Second part is traveled at 33 km/h
  * Average speed of the entire trip is 44 km/h
  Prove that the length of the first part of the trip is 25 km -/
theorem first_part_length (total_distance : ℝ) (speed1 speed2 avg_speed : ℝ) 
  (h1 : total_distance = 50)
  (h2 : speed1 = 66)
  (h3 : speed2 = 33)
  (h4 : avg_speed = 44)
  (h5 : ∃ x : ℝ, x > 0 ∧ x < total_distance ∧ 
        avg_speed = total_distance / (x / speed1 + (total_distance - x) / speed2)) :
  ∃ x : ℝ, x = 25 ∧ x > 0 ∧ x < total_distance ∧ 
    avg_speed = total_distance / (x / speed1 + (total_distance - x) / speed2) := by
  sorry

end NUMINAMATH_CALUDE_first_part_length_l1699_169961


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1699_169972

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 3 + a 4 + a 5 = 3)
    (h_a8 : a 8 = 8) : 
  a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1699_169972


namespace NUMINAMATH_CALUDE_skips_per_second_l1699_169927

def minutes_jumped : ℕ := 10
def total_skips : ℕ := 1800

def seconds_jumped : ℕ := minutes_jumped * 60

theorem skips_per_second : total_skips / seconds_jumped = 3 := by
  sorry

end NUMINAMATH_CALUDE_skips_per_second_l1699_169927


namespace NUMINAMATH_CALUDE_max_children_spell_names_l1699_169946

/-- Represents the available letters in the bag -/
def LetterBag : Finset Char := {'A', 'A', 'A', 'A', 'B', 'B', 'D', 'I', 'I', 'M', 'M', 'N', 'N', 'N', 'Y', 'Y'}

/-- Represents the names of the children -/
inductive Child
| Anna
| Vanya
| Dani
| Dima

/-- Returns the set of letters needed to spell a child's name -/
def lettersNeeded (c : Child) : Finset Char :=
  match c with
  | Child.Anna => {'A', 'N', 'N', 'A'}
  | Child.Vanya => {'V', 'A', 'N', 'Y'}
  | Child.Dani => {'D', 'A', 'N', 'Y'}
  | Child.Dima => {'D', 'I', 'M', 'A'}

/-- Theorem stating the maximum number of children who can spell their names -/
theorem max_children_spell_names :
  ∃ (S : Finset Child), (∀ c ∈ S, lettersNeeded c ⊆ LetterBag) ∧ 
                        (∀ T : Finset Child, (∀ c ∈ T, lettersNeeded c ⊆ LetterBag) → T.card ≤ S.card) ∧
                        S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_children_spell_names_l1699_169946


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l1699_169944

theorem a_is_perfect_square (a b : ℤ) (h : a = a^2 + b^2 - 8*b - 2*a*b + 16) :
  ∃ k : ℤ, a = k^2 := by sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l1699_169944


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l1699_169923

-- Define the geometric sequences and their properties
def geometric_sequence (k a₂ a₃ : ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2

-- Define the theorem
theorem sum_of_common_ratios_is_three
  (k a₂ a₃ b₂ b₃ : ℝ)
  (h₁ : geometric_sequence k a₂ a₃)
  (h₂ : geometric_sequence k b₂ b₃)
  (h₃ : ∃ p r : ℝ, p ≠ r ∧ a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2)
  (h₄ : a₃ - b₃ = 3 * (a₂ - b₂))
  (h₅ : k ≠ 0) :
  ∃ p r : ℝ, p + r = 3 ∧ 
    geometric_sequence k a₂ a₃ ∧
    geometric_sequence k b₂ b₃ ∧
    a₂ = k * p ∧ a₃ = k * p^2 ∧
    b₂ = k * r ∧ b₃ = k * r^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l1699_169923


namespace NUMINAMATH_CALUDE_performing_arts_school_l1699_169967

theorem performing_arts_school (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 150 ∧
  cant_sing = 80 ∧
  cant_dance = 110 ∧
  cant_act = 60 →
  ∃ (all_talents : ℕ),
    all_talents = total - ((total - cant_sing) + (total - cant_dance) + (total - cant_act) - total) ∧
    all_talents = 50 := by
  sorry

end NUMINAMATH_CALUDE_performing_arts_school_l1699_169967


namespace NUMINAMATH_CALUDE_bowling_ball_difference_l1699_169939

theorem bowling_ball_difference :
  ∀ (red green : ℕ),
  red = 30 →
  green > red →
  red + green = 66 →
  green - red = 6 :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_difference_l1699_169939


namespace NUMINAMATH_CALUDE_quadratic_form_completion_constant_term_value_l1699_169908

theorem quadratic_form_completion (x : ℝ) : 
  x^2 - 6*x = (x - 3)^2 - 9 :=
sorry

theorem constant_term_value : 
  ∃ k, ∀ x, x^2 - 6*x = (x - 3)^2 + k ∧ k = -9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_form_completion_constant_term_value_l1699_169908


namespace NUMINAMATH_CALUDE_joyce_apples_l1699_169959

/-- The number of apples Joyce ends up with after giving some away -/
def apples_remaining (starting_apples given_away : ℕ) : ℕ :=
  starting_apples - given_away

/-- Theorem stating that Joyce ends up with 23 apples -/
theorem joyce_apples : apples_remaining 75 52 = 23 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_l1699_169959


namespace NUMINAMATH_CALUDE_fruit_amount_proof_l1699_169913

/-- The cost of blueberries in dollars per carton -/
def blueberry_cost : ℚ := 5

/-- The weight of blueberries in ounces per carton -/
def blueberry_weight : ℚ := 6

/-- The cost of raspberries in dollars per carton -/
def raspberry_cost : ℚ := 3

/-- The weight of raspberries in ounces per carton -/
def raspberry_weight : ℚ := 8

/-- The number of batches of muffins -/
def num_batches : ℕ := 4

/-- The total savings in dollars by using raspberries instead of blueberries -/
def total_savings : ℚ := 22

/-- The amount of fruit in ounces required for each batch of muffins -/
def fruit_per_batch : ℚ := 12

theorem fruit_amount_proof :
  (total_savings / (num_batches : ℚ)) / 
  ((blueberry_cost / blueberry_weight) - (raspberry_cost / raspberry_weight)) = fruit_per_batch :=
sorry

end NUMINAMATH_CALUDE_fruit_amount_proof_l1699_169913


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1699_169914

theorem complex_fraction_equality : 
  (Complex.I : ℂ) ^ 2 = -1 → 
  (1 + Complex.I) ^ 3 / (1 - Complex.I) ^ 2 = -1 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1699_169914


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1699_169919

theorem geometric_series_ratio (a : ℝ) (r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r)) = 64 * (a * r^4 / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1699_169919


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l1699_169977

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_50th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 7)
  (h_fifteenth : a 15 = 41) :
  a 50 = 126 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l1699_169977


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1699_169924

theorem unique_solution_lcm_gcd_equation :
  ∃! n : ℕ+, n.val > 0 ∧ Nat.lcm n.val 180 = Nat.gcd n.val 180 + 360 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1699_169924


namespace NUMINAMATH_CALUDE_divisibility_criterion_l1699_169953

theorem divisibility_criterion (x : ℕ) : 
  (x ≥ 10 ∧ x ≤ 99) →
  (1207 % x = 0 ↔ (x / 10)^3 + (x % 10)^3 = 344) :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l1699_169953


namespace NUMINAMATH_CALUDE_arthur_reading_challenge_l1699_169917

/-- Arthur's summer reading challenge -/
theorem arthur_reading_challenge
  (total_goal : ℕ)
  (book1_pages : ℕ)
  (book2_pages : ℕ)
  (book1_read_percent : ℚ)
  (book2_read_fraction : ℚ)
  (h1 : total_goal = 800)
  (h2 : book1_pages = 500)
  (h3 : book2_pages = 1000)
  (h4 : book1_read_percent = 80 / 100)
  (h5 : book2_read_fraction = 1 / 5)
  : total_goal - (↑book1_pages * book1_read_percent + ↑book2_pages * book2_read_fraction) = 200 := by
  sorry

#check arthur_reading_challenge

end NUMINAMATH_CALUDE_arthur_reading_challenge_l1699_169917


namespace NUMINAMATH_CALUDE_freshman_psych_majors_percentage_l1699_169954

theorem freshman_psych_majors_percentage
  (total_students : ℕ)
  (freshman_ratio : ℚ)
  (liberal_arts_ratio : ℚ)
  (psych_major_ratio : ℚ)
  (h1 : freshman_ratio = 2/5)
  (h2 : liberal_arts_ratio = 1/2)
  (h3 : psych_major_ratio = 1/2)
  : (freshman_ratio * liberal_arts_ratio * psych_major_ratio : ℚ) = 1/10 := by
  sorry

#check freshman_psych_majors_percentage

end NUMINAMATH_CALUDE_freshman_psych_majors_percentage_l1699_169954


namespace NUMINAMATH_CALUDE_mean_home_runs_l1699_169910

def home_runs_data : List (Nat × Nat) := [(5, 6), (6, 8), (4, 10)]

theorem mean_home_runs :
  let total_home_runs := (home_runs_data.map (λ (players, hrs) => players * hrs)).sum
  let total_players := (home_runs_data.map (λ (players, _) => players)).sum
  (total_home_runs : ℚ) / total_players = 118 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l1699_169910


namespace NUMINAMATH_CALUDE_system_three_solutions_l1699_169945

/-- The system of equations has exactly three solutions if and only if a = 9 or a = 23 + 4√15 -/
theorem system_three_solutions (a : ℝ) :
  (∃! x y z : ℝ × ℝ, 
    ((abs (y.2 + 9) + abs (x.1 + 2) - 2) * (x.1^2 + x.2^2 - 3) = 0 ∧
     (x.1 + 2)^2 + (x.2 + 4)^2 = a) ∧
    ((abs (y.2 + 9) + abs (y.1 + 2) - 2) * (y.1^2 + y.2^2 - 3) = 0 ∧
     (y.1 + 2)^2 + (y.2 + 4)^2 = a) ∧
    ((abs (z.2 + 9) + abs (z.1 + 2) - 2) * (z.1^2 + z.2^2 - 3) = 0 ∧
     (z.1 + 2)^2 + (z.2 + 4)^2 = a) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔
  (a = 9 ∨ a = 23 + 4 * Real.sqrt 15) :=
by sorry

end NUMINAMATH_CALUDE_system_three_solutions_l1699_169945


namespace NUMINAMATH_CALUDE_real_number_operations_closure_l1699_169970

theorem real_number_operations_closure :
  (∀ (a b : ℝ), ∃ (c : ℝ), a + b = c) ∧
  (∀ (a b : ℝ), ∃ (c : ℝ), a - b = c) ∧
  (∀ (a b : ℝ), ∃ (c : ℝ), a * b = c) ∧
  (∀ (a b : ℝ), b ≠ 0 → ∃ (c : ℝ), a / b = c) :=
by sorry

end NUMINAMATH_CALUDE_real_number_operations_closure_l1699_169970


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_height_l1699_169966

/-- Represents a right hexagonal pyramid with three parallel cross sections. -/
structure HexagonalPyramid where
  /-- Height of the smallest cross section from the apex -/
  x : ℝ
  /-- Area of the smallest cross section -/
  area₁ : ℝ
  /-- Area of the middle cross section -/
  area₂ : ℝ
  /-- Area of the largest cross section -/
  area₃ : ℝ
  /-- The areas are in the correct ratio -/
  area_ratio₁ : area₁ / area₂ = 9 / 20
  /-- The areas are in the correct ratio -/
  area_ratio₂ : area₂ / area₃ = 5 / 9
  /-- The heights are in arithmetic progression -/
  height_progression : x + 20 - (x + 10) = (x + 10) - x

/-- The height of the smallest cross section from the apex in a right hexagonal pyramid
    with specific cross-sectional areas at specific heights. -/
theorem hexagonal_pyramid_height (p : HexagonalPyramid) : p.x = 100 / (10 - 3 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_height_l1699_169966


namespace NUMINAMATH_CALUDE_new_average_weight_l1699_169931

theorem new_average_weight (initial_count : ℕ) (initial_avg : ℚ) (new_weight : ℚ) :
  initial_count = 19 →
  initial_avg = 15 →
  new_weight = 13 →
  let total_weight := initial_count * initial_avg + new_weight
  let new_count := initial_count + 1
  new_count * (total_weight / new_count) = 298 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l1699_169931


namespace NUMINAMATH_CALUDE_charlies_metal_storage_l1699_169999

/-- The amount of metal Charlie has in storage -/
def metal_in_storage (total_needed : ℕ) (to_buy : ℕ) : ℕ :=
  total_needed - to_buy

/-- Theorem: Charlie's metal in storage is the difference between total needed and amount to buy -/
theorem charlies_metal_storage :
  metal_in_storage 635 359 = 276 := by
  sorry

end NUMINAMATH_CALUDE_charlies_metal_storage_l1699_169999


namespace NUMINAMATH_CALUDE_average_income_B_C_l1699_169922

def average_income (x y : ℕ) : ℚ := (x + y) / 2

theorem average_income_B_C 
  (h1 : average_income A_income B_income = 4050)
  (h2 : average_income A_income C_income = 4200)
  (h3 : A_income = 3000) :
  average_income B_income C_income = 5250 :=
by
  sorry


end NUMINAMATH_CALUDE_average_income_B_C_l1699_169922


namespace NUMINAMATH_CALUDE_sugar_bag_weight_l1699_169998

/-- The weight of a bag of sugar, given the weight of a bag of salt and their combined weight after removing 4 kg. -/
theorem sugar_bag_weight (salt_weight : ℝ) (combined_weight_minus_four : ℝ) 
  (h1 : salt_weight = 30)
  (h2 : combined_weight_minus_four = 42)
  (h3 : combined_weight_minus_four = salt_weight + sugar_weight - 4) :
  sugar_weight = 16 := by
  sorry

#check sugar_bag_weight

end NUMINAMATH_CALUDE_sugar_bag_weight_l1699_169998


namespace NUMINAMATH_CALUDE_present_age_of_B_prove_present_age_of_B_l1699_169965

theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 30 = 2 * (b - 30)) →  -- In 30 years, A will be twice as old as B was 30 years ago
    (a = b + 5) →              -- A is now 5 years older than B
    (b = 95)                   -- B's current age is 95 years

-- The proof of the theorem
theorem prove_present_age_of_B : ∃ a b : ℕ, present_age_of_B a b :=
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_prove_present_age_of_B_l1699_169965


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_common_difference_l1699_169900

theorem prime_arithmetic_sequence_common_difference (p : ℕ) (a : ℕ → ℕ) (d : ℕ) :
  Prime p →
  (∀ i, i ∈ Finset.range p → Prime (a i)) →
  (∀ i j, i < j → j < p → a i < a j) →
  (∀ i, i < p - 1 → a (i + 1) - a i = d) →
  a 0 > p →
  p ∣ d :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_common_difference_l1699_169900


namespace NUMINAMATH_CALUDE_swimming_practice_months_l1699_169985

def total_required_hours : ℕ := 1500
def completed_hours : ℕ := 180
def monthly_practice_hours : ℕ := 220

theorem swimming_practice_months :
  (total_required_hours - completed_hours) / monthly_practice_hours = 6 :=
by sorry

end NUMINAMATH_CALUDE_swimming_practice_months_l1699_169985


namespace NUMINAMATH_CALUDE_lakers_win_probability_l1699_169982

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 2/3

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 1/3

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The minimum number of games in the series -/
def min_games : ℕ := 5

/-- The probability of the Lakers winning the NBA Finals given that the series lasts at least 5 games -/
theorem lakers_win_probability : 
  (Finset.sum (Finset.range 3) (λ i => 
    (Nat.choose (games_to_win + i) i) * 
    (p_lakers ^ games_to_win) * 
    (p_celtics ^ i))) = 1040/729 := by sorry

end NUMINAMATH_CALUDE_lakers_win_probability_l1699_169982


namespace NUMINAMATH_CALUDE_fence_price_per_foot_l1699_169911

/-- Given a square plot with area and total fencing cost, calculate the price per foot of fencing --/
theorem fence_price_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3740) : 
  total_cost / (4 * Real.sqrt area) = 55 := by
  sorry

end NUMINAMATH_CALUDE_fence_price_per_foot_l1699_169911


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1699_169994

theorem cubic_equation_roots :
  let a : ℝ := 5
  let b : ℝ := (5 + Real.sqrt 61) / 2
  let f (x : ℝ) := x^3 - 5*x^2 - 9*x + 45
  (f a = 0 ∧ f b = 0 ∧ f (-b) = 0) ∧
  (∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ = -r₂) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1699_169994


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1699_169971

theorem triangle_max_perimeter (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  ∃ (n : ℕ), n = 62 ∧ ∀ (s : ℝ), s > 0 → a + s > b → b + s > a → n > a + b + s :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1699_169971


namespace NUMINAMATH_CALUDE_puzzle_solution_l1699_169920

theorem puzzle_solution (x y : ℤ) (h1 : 3 * x + 4 * y = 150) (h2 : x = 15 ∨ y = 15) : 
  (x ≠ 15 → x = 30) ∧ (y ≠ 15 → y = 30) :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1699_169920


namespace NUMINAMATH_CALUDE_fourth_score_proof_l1699_169909

/-- Given four test scores with an average of 94, where three scores are known to be 85, 100, and 94,
    prove that the fourth score must be 97. -/
theorem fourth_score_proof (score1 score2 score3 score4 : ℕ) : 
  score1 = 85 → score2 = 100 → score3 = 94 → 
  (score1 + score2 + score3 + score4) / 4 = 94 →
  score4 = 97 := by sorry

end NUMINAMATH_CALUDE_fourth_score_proof_l1699_169909


namespace NUMINAMATH_CALUDE_field_planting_fraction_l1699_169989

theorem field_planting_fraction :
  ∀ (a b c x : ℝ),
  a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
  x^2 * c = 3 * (a * b) →
  (a * b - x^2) / (a * b) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_field_planting_fraction_l1699_169989


namespace NUMINAMATH_CALUDE_pool_cleaning_tip_percentage_l1699_169990

/-- Calculates the tip percentage for pool cleaning sessions -/
theorem pool_cleaning_tip_percentage
  (days_between_cleanings : ℕ)
  (cost_per_cleaning : ℕ)
  (chemical_cost : ℕ)
  (chemical_frequency : ℕ)
  (total_monthly_cost : ℕ)
  (days_in_month : ℕ := 30)  -- Assumption from the problem
  (h1 : days_between_cleanings = 3)
  (h2 : cost_per_cleaning = 150)
  (h3 : chemical_cost = 200)
  (h4 : chemical_frequency = 2)
  (h5 : total_monthly_cost = 2050)
  : (total_monthly_cost - (days_in_month / days_between_cleanings * cost_per_cleaning + chemical_frequency * chemical_cost)) / (days_in_month / days_between_cleanings * cost_per_cleaning) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_pool_cleaning_tip_percentage_l1699_169990


namespace NUMINAMATH_CALUDE_house_sale_gain_l1699_169984

/-- Calculates the net gain from selling a house at a profit and buying it back at a loss -/
def netGainFromHouseSale (initialValue : ℝ) (profitPercent : ℝ) (lossPercent : ℝ) : ℝ :=
  let sellingPrice := initialValue * (1 + profitPercent)
  let buybackPrice := sellingPrice * (1 - lossPercent)
  sellingPrice - buybackPrice

/-- Theorem stating that selling a $200,000 house at 15% profit and buying it back at 5% loss results in $11,500 gain -/
theorem house_sale_gain :
  netGainFromHouseSale 200000 0.15 0.05 = 11500 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_gain_l1699_169984


namespace NUMINAMATH_CALUDE_ellipse_point_properties_l1699_169904

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-4, 0)
def right_focus : ℝ × ℝ := (4, 0)

-- Define the angle between PF₁ and PF₂
def angle_PF1F2 (P : ℝ × ℝ) : ℝ := 60

-- Theorem statement
theorem ellipse_point_properties (P : ℝ × ℝ) 
  (h_on_ellipse : is_on_ellipse P.1 P.2) 
  (h_angle : angle_PF1F2 P = 60) :
  (∃ (S : ℝ), S = 3 * Real.sqrt 3 ∧ 
    S = (1/2) * Real.sqrt ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) *
              Real.sqrt ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2) *
              Real.sin (angle_PF1F2 P * π / 180)) ∧
  (P.1 = 5 * Real.sqrt 13 / 4 ∨ P.1 = -5 * Real.sqrt 13 / 4) ∧
  (P.2 = 4 * Real.sqrt 3 / 4 ∨ P.2 = -4 * Real.sqrt 3 / 4) := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_properties_l1699_169904


namespace NUMINAMATH_CALUDE_line_plane_relations_l1699_169952

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- Represents a line in 3D space -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  z₁ : ℝ
  m : ℝ
  n : ℝ
  p : ℝ

/-- Determines if a line is parallel to a plane -/
def isParallel (plane : Plane) (line : Line) : Prop :=
  plane.A * line.m + plane.B * line.n + plane.C * line.p = 0

/-- Determines if a line is perpendicular to a plane -/
def isPerpendicular (plane : Plane) (line : Line) : Prop :=
  plane.A / line.m = plane.B / line.n ∧ plane.B / line.n = plane.C / line.p

theorem line_plane_relations (plane : Plane) (line : Line) :
  (isParallel plane line ↔ plane.A * line.m + plane.B * line.n + plane.C * line.p = 0) ∧
  (isPerpendicular plane line ↔ plane.A / line.m = plane.B / line.n ∧ plane.B / line.n = plane.C / line.p) :=
sorry

end NUMINAMATH_CALUDE_line_plane_relations_l1699_169952


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1699_169993

/-- A trapezoid with a line joining the midpoints of its diagonals. -/
structure Trapezoid where
  /-- The length of the longer base of the trapezoid. -/
  longer_base : ℝ
  /-- The length of the shorter base of the trapezoid. -/
  shorter_base : ℝ
  /-- The length of the line joining the midpoints of the diagonals. -/
  midline_length : ℝ
  /-- The midline length is half the difference of the bases. -/
  midline_property : midline_length = (longer_base - shorter_base) / 2

/-- 
Given a trapezoid where the line joining the midpoints of the diagonals has length 5
and the longer base is 105, the shorter base has length 95.
-/
theorem trapezoid_shorter_base (t : Trapezoid) 
    (h1 : t.longer_base = 105)
    (h2 : t.midline_length = 5) : 
    t.shorter_base = 95 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1699_169993


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1699_169902

theorem smaller_number_proof (x y : ℝ) : 
  y = 3 * x + 11 → x + y = 55 → x = 11 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1699_169902


namespace NUMINAMATH_CALUDE_weeks_to_save_for_coat_l1699_169903

/-- Calculates the number of weeks needed to save for a coat given specific conditions -/
theorem weeks_to_save_for_coat (weekly_savings : ℚ) (bill_fraction : ℚ) (gift : ℚ) (coat_cost : ℚ) :
  weekly_savings = 25 ∧ 
  bill_fraction = 1/3 ∧ 
  gift = 70 ∧ 
  coat_cost = 170 →
  ∃ w : ℕ, w * weekly_savings - (bill_fraction * 7 * weekly_savings) + gift = coat_cost ∧ w = 19 :=
by sorry

end NUMINAMATH_CALUDE_weeks_to_save_for_coat_l1699_169903


namespace NUMINAMATH_CALUDE_mean_daily_profit_l1699_169957

theorem mean_daily_profit (days_in_month : ℕ) (first_half_mean : ℚ) (second_half_mean : ℚ) :
  days_in_month = 30 →
  first_half_mean = 225 →
  second_half_mean = 475 →
  (first_half_mean * 15 + second_half_mean * 15) / days_in_month = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_daily_profit_l1699_169957


namespace NUMINAMATH_CALUDE_polyhedron_space_diagonals_l1699_169987

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces 
    (of which 32 are triangular and 12 are quadrilateral) has 339 space diagonals -/
theorem polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 32,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 339 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_space_diagonals_l1699_169987


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1699_169996

theorem complete_square_quadratic (x : ℝ) : 
  x^2 - 2*x - 2 = 0 → (x - 1)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1699_169996


namespace NUMINAMATH_CALUDE_inscribed_triangles_inequality_l1699_169976

/-- Two equilateral triangles inscribed in a circle -/
structure InscribedTriangles where
  r : ℝ
  S : ℝ

/-- Theorem: For two equilateral triangles inscribed in a circle with radius r,
    where S is the area of their common part, 2S ≥ √3 r² holds. -/
theorem inscribed_triangles_inequality (t : InscribedTriangles) : 2 * t.S ≥ Real.sqrt 3 * t.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangles_inequality_l1699_169976


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l1699_169925

theorem min_value_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 9) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 ∧
  ((a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) = 9 ↔ a = 3 ∧ b = 3 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l1699_169925


namespace NUMINAMATH_CALUDE_difference_of_squares_l1699_169942

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1699_169942


namespace NUMINAMATH_CALUDE_tom_new_books_l1699_169901

/-- Calculates the number of new books Tom bought given his initial, sold, and final book counts. -/
def new_books (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Tom bought 38 new books given the problem conditions. -/
theorem tom_new_books : new_books 5 4 39 = 38 := by
  sorry

end NUMINAMATH_CALUDE_tom_new_books_l1699_169901


namespace NUMINAMATH_CALUDE_reverse_difference_for_253_l1699_169905

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ∈ Finset.range 10
  t_range : tens ∈ Finset.range 10
  o_range : ones ∈ Finset.range 10

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  h_range := n.o_range
  t_range := n.t_range
  o_range := n.h_range

def ThreeDigitNumber.sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

theorem reverse_difference_for_253 (n : ThreeDigitNumber) 
    (h_253 : n.toNat = 253)
    (h_sum : n.sumOfDigits = 10)
    (h_middle : n.tens = n.hundreds + n.ones) :
    (n.reverse.toNat - n.toNat) = 99 := by
  sorry

#check reverse_difference_for_253

end NUMINAMATH_CALUDE_reverse_difference_for_253_l1699_169905


namespace NUMINAMATH_CALUDE_sequence_divergence_criterion_l1699_169995

/-- Given a sequence xₙ and a limit point a, prove that for every ε > 0,
    there exists a number k such that for all n > k, |xₙ - a| ≥ ε -/
theorem sequence_divergence_criterion 
  (x : ℕ → ℝ) (a : ℝ) : 
  ∀ ε > 0, ∃ k : ℕ, ∀ n > k, |x n - a| ≥ ε := by
  sorry

end NUMINAMATH_CALUDE_sequence_divergence_criterion_l1699_169995


namespace NUMINAMATH_CALUDE_remainder_91_power_91_mod_100_l1699_169907

theorem remainder_91_power_91_mod_100 : 91^91 % 100 = 91 := by
  sorry

end NUMINAMATH_CALUDE_remainder_91_power_91_mod_100_l1699_169907


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l1699_169974

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_strictly_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l1699_169974


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1699_169973

/-- The equation of an ellipse with foci at (4,0) and (-4,0) -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

/-- The simplified equation of the ellipse -/
def simplified_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

/-- Theorem stating the equivalence of the two equations -/
theorem ellipse_equation_equivalence :
  ∀ x y : ℝ, ellipse_equation x y ↔ simplified_equation x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1699_169973


namespace NUMINAMATH_CALUDE_adoption_time_proof_l1699_169941

/-- The number of days required to adopt all puppies in a pet shelter -/
def adoptionDays (initialPuppies additionalPuppies adoptionRate : ℕ) : ℕ :=
  (initialPuppies + additionalPuppies) / adoptionRate

/-- Theorem: It takes 7 days to adopt all puppies given the specified conditions -/
theorem adoption_time_proof :
  adoptionDays 9 12 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_adoption_time_proof_l1699_169941


namespace NUMINAMATH_CALUDE_remainder_problem_l1699_169933

theorem remainder_problem (n : ℕ) : 
  n % 6 = 4 ∧ n / 6 = 124 → (n + 24) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1699_169933


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l1699_169937

/-- Given a rectangular metallic sheet, this theorem proves that if the length is 48 meters,
    a 3-meter square is cut from each corner, and the resulting open box has a volume of 3780 m³,
    then the original width of the sheet must be 36 meters. -/
theorem metallic_sheet_width (length : ℝ) (width : ℝ) (cut_size : ℝ) (volume : ℝ) :
  length = 48 →
  cut_size = 3 →
  volume = 3780 →
  volume = (length - 2 * cut_size) * (width - 2 * cut_size) * cut_size →
  width = 36 := by
  sorry

#check metallic_sheet_width

end NUMINAMATH_CALUDE_metallic_sheet_width_l1699_169937


namespace NUMINAMATH_CALUDE_divisibility_implies_value_l1699_169948

theorem divisibility_implies_value (p q r : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^4 + 6*x^3 + 8*p*x^2 + 6*q*x + r = (x^3 + 4*x^2 + 16*x + 4) * k) →
  (p + q) * r = 56 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_value_l1699_169948


namespace NUMINAMATH_CALUDE_smallest_bob_number_l1699_169997

def alice_number : ℕ := 36

def has_all_prime_factors_plus_one (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ n → p ∣ m) ∧
  ∃ q : ℕ, Prime q ∧ q ∣ m ∧ ¬(q ∣ n)

theorem smallest_bob_number :
  ∃ m : ℕ, has_all_prime_factors_plus_one alice_number m ∧
  ∀ k : ℕ, has_all_prime_factors_plus_one alice_number k → m ≤ k :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l1699_169997


namespace NUMINAMATH_CALUDE_binomial_600_600_l1699_169962

theorem binomial_600_600 : Nat.choose 600 600 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_600_600_l1699_169962


namespace NUMINAMATH_CALUDE_rainy_day_probability_l1699_169928

theorem rainy_day_probability (A B : Set ℝ) (P : Set ℝ → ℝ) 
  (hA : P A = 0.06)
  (hB : P B = 0.08)
  (hAB : P (A ∩ B) = 0.02) :
  P B / P A = 1/3 :=
sorry

end NUMINAMATH_CALUDE_rainy_day_probability_l1699_169928


namespace NUMINAMATH_CALUDE_tuesday_rainfall_l1699_169975

/-- Given that it rained 0.9 inches on Monday and Tuesday's rainfall was 0.7 inches less than Monday's,
    prove that it rained 0.2 inches on Tuesday. -/
theorem tuesday_rainfall (monday_rain : ℝ) (tuesday_difference : ℝ) 
  (h1 : monday_rain = 0.9)
  (h2 : tuesday_difference = 0.7) :
  monday_rain - tuesday_difference = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_l1699_169975


namespace NUMINAMATH_CALUDE_program_output_correct_l1699_169983

def program_execution (initial_A initial_B : ℤ) : ℤ × ℤ × ℤ :=
  let A₁ := if initial_A < 0 then -initial_A else initial_A
  let B₁ := initial_B ^ 2
  let A₂ := A₁ + B₁
  let C  := A₂ - 2 * B₁
  let A₃ := A₂ / C
  let B₂ := B₁ * C + 1
  (A₃, B₂, C)

theorem program_output_correct :
  program_execution (-6) 2 = (5, 9, 2) := by
  sorry

end NUMINAMATH_CALUDE_program_output_correct_l1699_169983


namespace NUMINAMATH_CALUDE_sum_of_digits_of_k_l1699_169949

def k : ℕ := 10^30 - 54

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_k : sum_of_digits k = 11 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_k_l1699_169949


namespace NUMINAMATH_CALUDE_barn_paint_area_l1699_169916

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a rectangular barn -/
def totalPaintArea (dims : BarnDimensions) : ℝ :=
  2 * (2 * dims.width * dims.height + 2 * dims.length * dims.height) + dims.width * dims.length

/-- Theorem stating that the total area to be painted for the given barn is 654 sq yd -/
theorem barn_paint_area :
  let dims : BarnDimensions := { width := 11, length := 14, height := 6 }
  totalPaintArea dims = 654 := by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l1699_169916


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1699_169915

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 + a 6 = 3 →
  a 6 + a 10 = 12 →
  a 8 + a 12 = 24 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1699_169915


namespace NUMINAMATH_CALUDE_expression_simplification_l1699_169935

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3) :
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1699_169935


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l1699_169938

theorem mixed_fraction_product (X Y : ℤ) : 
  (5 + 1 / X : ℚ) * (Y + 1 / 2 : ℚ) = 43 →
  5 < 5 + 1 / X →
  5 + 1 / X ≤ 11 / 2 →
  X = 17 ∧ Y = 8 := by
sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l1699_169938


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1699_169912

theorem largest_integer_inequality : ∀ y : ℤ, y ≤ 3 ↔ (y : ℚ) / 4 + 6 / 7 < 7 / 4 := by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1699_169912


namespace NUMINAMATH_CALUDE_equation_implies_m_equals_zero_l1699_169929

theorem equation_implies_m_equals_zero (m n : ℝ) :
  21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_m_equals_zero_l1699_169929


namespace NUMINAMATH_CALUDE_exists_unique_q_l1699_169968

/-- Polynomial function g(x) -/
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p*x^4 + q*x^3 + r*x^2 + s*x + t

/-- Theorem stating the existence and uniqueness of q -/
theorem exists_unique_q :
  ∃! q : ℝ, ∃ p r s t : ℝ,
    g p q r s t 0 = 3 ∧
    g p q r s t (-2) = 0 ∧
    g p q r s t 1 = 0 ∧
    g p q r s t (-1) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_exists_unique_q_l1699_169968


namespace NUMINAMATH_CALUDE_negative_a_squared_times_b_over_a_squared_l1699_169940

theorem negative_a_squared_times_b_over_a_squared (a b : ℝ) (h : a ≠ 0) :
  ((-a)^2 * b) / (a^2) = b := by sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_b_over_a_squared_l1699_169940


namespace NUMINAMATH_CALUDE_circle_radius_l1699_169930

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 36 = 6*x + 24*y) → 
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = Real.sqrt 117 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1699_169930


namespace NUMINAMATH_CALUDE_choir_composition_l1699_169934

theorem choir_composition (initial_total : ℕ) (initial_blonde : ℕ) (added_blonde : ℕ) :
  initial_total = 80 →
  initial_blonde = 30 →
  added_blonde = 10 →
  initial_total - initial_blonde + added_blonde = 50 :=
by sorry

end NUMINAMATH_CALUDE_choir_composition_l1699_169934


namespace NUMINAMATH_CALUDE_triangle_area_l1699_169943

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 - t.b * t.c ∧
  t.b * t.c * Real.cos t.A = -4

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : satisfies_conditions t) : 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1699_169943


namespace NUMINAMATH_CALUDE_fraction_equality_l1699_169981

theorem fraction_equality : (2 - 4 + 8 - 16 + 32 + 64) / (4 - 8 + 16 - 32 + 64 + 128) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1699_169981


namespace NUMINAMATH_CALUDE_min_value_of_f_l1699_169958

/-- The function f(x) = 3x^2 - 6x + 9 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 9

/-- The minimum value of f(x) is 6 -/
theorem min_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1699_169958


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l1699_169960

/-- Calculates the perimeter of a rectangular field enclosed by evenly spaced posts -/
def field_perimeter (num_posts : ℕ) (post_width : ℚ) (gap_width : ℚ) : ℚ :=
  let short_side_posts := num_posts / 3
  let short_side_gaps := short_side_posts - 1
  let short_side_length := short_side_gaps * gap_width + short_side_posts * (post_width / 12)
  let long_side_length := 2 * short_side_length
  2 * (short_side_length + long_side_length)

theorem rectangular_field_perimeter :
  field_perimeter 36 (4 / 1) (7 / 2) = 238 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l1699_169960


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1699_169992

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (x * y * z + 2 * x + 3 * y + 6 * z = x * y + 2 * x * z + 3 * y * z) →
    (x = 4 ∧ y = 3 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1699_169992


namespace NUMINAMATH_CALUDE_function_exponent_proof_l1699_169950

theorem function_exponent_proof (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) → f 3 = Real.sqrt 3 → n = 1/2 := by
sorry

end NUMINAMATH_CALUDE_function_exponent_proof_l1699_169950


namespace NUMINAMATH_CALUDE_f_2008_equals_zero_l1699_169918

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property f(1-x) = f(1+x)
def symmetric_around_one (f : ℝ → ℝ) : Prop := ∀ x, f (1-x) = f (1+x)

theorem f_2008_equals_zero 
  (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_sym : symmetric_around_one f) : 
  f 2008 = 0 := by
sorry

end NUMINAMATH_CALUDE_f_2008_equals_zero_l1699_169918


namespace NUMINAMATH_CALUDE_max_value_theorem_l1699_169951

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  (a^2 * b^2) / (a + b) + (a^2 * c^2) / (a + c) + (b^2 * c^2) / (b + c) ≤ 1/6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1699_169951


namespace NUMINAMATH_CALUDE_cleaner_flow_rate_l1699_169980

/-- Represents the rate of cleaner flow through a pipe over time --/
structure CleanerFlow where
  initial_rate : ℝ
  middle_rate : ℝ
  final_rate : ℝ
  total_time : ℝ
  first_change_time : ℝ
  second_change_time : ℝ
  total_amount : ℝ

/-- The cleaner flow satisfies the problem conditions --/
def satisfies_conditions (flow : CleanerFlow) : Prop :=
  flow.initial_rate = 2 ∧
  flow.final_rate = 4 ∧
  flow.total_time = 30 ∧
  flow.first_change_time = 15 ∧
  flow.second_change_time = 25 ∧
  flow.total_amount = 80 ∧
  flow.initial_rate * flow.first_change_time +
  flow.middle_rate * (flow.second_change_time - flow.first_change_time) +
  flow.final_rate * (flow.total_time - flow.second_change_time) = flow.total_amount

theorem cleaner_flow_rate (flow : CleanerFlow) :
  satisfies_conditions flow → flow.middle_rate = 3 := by
  sorry


end NUMINAMATH_CALUDE_cleaner_flow_rate_l1699_169980


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1699_169963

theorem right_triangle_hypotenuse (a b c : ℕ) : 
  a * a + b * b = c * c →  -- Pythagorean theorem
  c - b = 1575 →           -- One leg is 1575 units shorter than hypotenuse
  a < 1991 →               -- The other leg is less than 1991 units
  c = 1799 :=              -- The hypotenuse length is 1799
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1699_169963


namespace NUMINAMATH_CALUDE_common_ratio_equation_l1699_169906

/-- A geometric progression with positive terms where the first term is equal to the sum of the next three terms -/
structure GeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_condition : a = a * r + a * r^2 + a * r^3

/-- The common ratio of the geometric progression satisfies the equation r^3 + r^2 + r - 1 = 0 -/
theorem common_ratio_equation (gp : GeometricProgression) : gp.r^3 + gp.r^2 + gp.r - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_equation_l1699_169906


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l1699_169932

theorem largest_number_with_given_hcf_lcm_factors :
  ∀ a b c : ℕ+,
  (∃ (hcf : ℕ+) (lcm : ℕ+), 
    (Nat.gcd a b = hcf) ∧ 
    (Nat.gcd (Nat.gcd a b) c = hcf) ∧
    (Nat.lcm (Nat.lcm a b) c = lcm) ∧
    (hcf = 59) ∧
    (∃ (k : ℕ+), lcm = hcf * 13 * (2^4) * 23 * k)) →
  max a (max b c) = 282256 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_lcm_factors_l1699_169932


namespace NUMINAMATH_CALUDE_quadratic_properties_l1699_169955

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a > 0)
  (hc : c > 0)
  (hf : f a b c c = 0)
  (h_positive : ∀ x, 0 < x → x < c → f a b c x > 0)
  (h_distinct : ∃ x, x ≠ c ∧ f a b c x = 0) :
  -- 1. The other x-intercept is at x = 1/a
  (∃ x, x ≠ c ∧ f a b c x = 0 ∧ x = 1/a) ∧
  -- 2. f(x) < 0 for x ∈ (c, 1/a)
  (∀ x, c < x → x < 1/a → f a b c x < 0) ∧
  -- 3. If the area of the triangle is 8, then 0 < a ≤ 1/8
  (((1/a - c) * c / 2 = 8) → (0 < a ∧ a ≤ 1/8)) ∧
  -- 4. If m^2 - 2km + 1 + b + ac ≥ 0 for all k ∈ [-1, 1], then m ≤ -2 or m = 0 or m ≥ 2
  ((∀ k, -1 ≤ k → k ≤ 1 → ∀ m, m^2 - 2*k*m + 1 + b + a*c ≥ 0) →
   ∀ m, m ≤ -2 ∨ m = 0 ∨ m ≥ 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1699_169955


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l1699_169956

theorem rug_overlap_problem (total_rug_area single_coverage double_coverage triple_coverage : ℝ) :
  total_rug_area = 200 →
  single_coverage + double_coverage + triple_coverage = 140 →
  double_coverage = 24 →
  triple_coverage = 18 :=
by sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l1699_169956


namespace NUMINAMATH_CALUDE_min_both_composers_l1699_169926

theorem min_both_composers (total : ℕ) (like_beethoven : ℕ) (like_chopin : ℕ)
  (h_total : total = 200)
  (h_beethoven : like_beethoven = 160)
  (h_chopin : like_chopin = 150)
  (h_beethoven_le : like_beethoven ≤ total)
  (h_chopin_le : like_chopin ≤ total) :
  ∃ (both : ℕ), both ≥ like_beethoven + like_chopin - total ∧
    (∀ (x : ℕ), x ≥ like_beethoven + like_chopin - total → x ≥ both) ∧
    both = 110 :=
sorry

end NUMINAMATH_CALUDE_min_both_composers_l1699_169926


namespace NUMINAMATH_CALUDE_quartet_performance_count_l1699_169936

/-- Represents the number of songs sung by each friend -/
structure SongCounts where
  lucy : ℕ
  sarah : ℕ
  beth : ℕ
  jane : ℕ

/-- The total number of songs performed by the quartets -/
def total_songs (counts : SongCounts) : ℕ :=
  (counts.lucy + counts.sarah + counts.beth + counts.jane) / 3

theorem quartet_performance_count (counts : SongCounts) :
  counts.lucy = 8 →
  counts.sarah = 5 →
  counts.beth > counts.sarah →
  counts.beth < counts.lucy →
  counts.jane > counts.sarah →
  counts.jane < counts.lucy →
  total_songs counts = 9 := by
  sorry

#eval total_songs ⟨8, 5, 7, 7⟩

end NUMINAMATH_CALUDE_quartet_performance_count_l1699_169936


namespace NUMINAMATH_CALUDE_benny_work_hours_l1699_169969

theorem benny_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 5 → days_worked = 12 → total_hours = hours_per_day * days_worked → total_hours = 60 := by
  sorry

end NUMINAMATH_CALUDE_benny_work_hours_l1699_169969


namespace NUMINAMATH_CALUDE_cycle_cost_price_l1699_169978

-- Define the cost price and selling price
def cost_price : ℝ := 1600
def selling_price : ℝ := 1360

-- Define the loss percentage
def loss_percentage : ℝ := 15

-- Theorem statement
theorem cycle_cost_price : 
  selling_price = cost_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cycle_cost_price_l1699_169978


namespace NUMINAMATH_CALUDE_factorization_proof_l1699_169986

theorem factorization_proof (x : ℝ) : 72 * x^5 - 90 * x^9 = 18 * x^5 * (4 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1699_169986
