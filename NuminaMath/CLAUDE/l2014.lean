import Mathlib

namespace NUMINAMATH_CALUDE_impossible_table_l2014_201441

/-- Represents a 6x6 table of integers -/
def Table := Fin 6 → Fin 6 → ℤ

/-- Checks if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

/-- Checks if the sum of a 1x5 rectangle is valid (2022 or 2023) -/
def valid_sum (s : ℤ) : Prop := s = 2022 ∨ s = 2023

/-- Checks if all 1x5 rectangles (horizontal and vertical) have valid sums -/
def all_rectangles_valid (t : Table) : Prop :=
  (∀ i j, valid_sum (t i j + t i (j+1) + t i (j+2) + t i (j+3) + t i (j+4))) ∧
  (∀ i j, valid_sum (t i j + t (i+1) j + t (i+2) j + t (i+3) j + t (i+4) j))

/-- The main theorem: it's impossible to fill the table satisfying all conditions -/
theorem impossible_table : ¬∃ (t : Table), all_distinct t ∧ all_rectangles_valid t := by
  sorry

end NUMINAMATH_CALUDE_impossible_table_l2014_201441


namespace NUMINAMATH_CALUDE_cubic_difference_l2014_201409

theorem cubic_difference (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 2 * x + y = 16) : 
  x^3 - y^3 = -448 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l2014_201409


namespace NUMINAMATH_CALUDE_min_value_ab_l2014_201491

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 5/a + 20/b = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → 5/x + 20/y = 4 → a * b ≤ x * y ∧ a * b = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l2014_201491


namespace NUMINAMATH_CALUDE_min_value_of_a_l2014_201443

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioc (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2014_201443


namespace NUMINAMATH_CALUDE_angle_triple_complement_l2014_201459

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l2014_201459


namespace NUMINAMATH_CALUDE_train_passing_platform_l2014_201496

/-- Time taken for a train to pass a platform -/
theorem train_passing_platform (train_length platform_length : ℝ) (train_speed : ℝ) : 
  train_length = 360 →
  platform_length = 140 →
  train_speed = 45 →
  (train_length + platform_length) / (train_speed * 1000 / 3600) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_train_passing_platform_l2014_201496


namespace NUMINAMATH_CALUDE_completing_square_equiv_l2014_201436

/-- Proves that y = -x^2 + 2x + 3 can be rewritten as y = -(x-1)^2 + 4 -/
theorem completing_square_equiv :
  ∀ x y : ℝ, y = -x^2 + 2*x + 3 ↔ y = -(x-1)^2 + 4 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equiv_l2014_201436


namespace NUMINAMATH_CALUDE_percentage_problem_l2014_201429

theorem percentage_problem (P : ℝ) : P = 0.7 ↔ 
  0.8 * 90 = P * 60.00000000000001 + 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2014_201429


namespace NUMINAMATH_CALUDE_constant_sequence_l2014_201433

theorem constant_sequence (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i > j → ((i - j)^(2*(i - j)) + 1) ∣ (a i - a j)) :
  ∀ n : ℕ, n ≥ 1 → a n = a 1 :=
by sorry

end NUMINAMATH_CALUDE_constant_sequence_l2014_201433


namespace NUMINAMATH_CALUDE_removed_triangles_area_l2014_201468

-- Define the square side length
def square_side : ℝ := 16

-- Define the ratio of r to s
def r_to_s_ratio : ℝ := 3

-- Theorem statement
theorem removed_triangles_area (r s : ℝ) : 
  r / s = r_to_s_ratio →
  (r + s)^2 + (r - s)^2 = square_side^2 →
  4 * (1/2 * r * s) = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l2014_201468


namespace NUMINAMATH_CALUDE_g_neg_two_l2014_201469

def g (x : ℝ) : ℝ := x^3 - x^2 + x

theorem g_neg_two : g (-2) = -14 := by sorry

end NUMINAMATH_CALUDE_g_neg_two_l2014_201469


namespace NUMINAMATH_CALUDE_runners_meet_at_start_l2014_201493

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the race -/
structure RaceState where
  runner_a : Runner
  runner_b : Runner
  time : ℝ

def track_length : ℝ := 300

/-- Function to update the race state after each meeting -/
def update_race_state (state : RaceState) : RaceState :=
  sorry

/-- Function to check if both runners are at the starting point -/
def at_start (state : RaceState) : Bool :=
  sorry

/-- Theorem stating that the runners meet at the starting point after 250 seconds -/
theorem runners_meet_at_start :
  let initial_state : RaceState := {
    runner_a := { speed := 2, direction := true },
    runner_b := { speed := 4, direction := false },
    time := 0
  }
  let final_state := update_race_state initial_state
  (at_start final_state ∧ final_state.time = 250) := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_at_start_l2014_201493


namespace NUMINAMATH_CALUDE_paper_clips_in_2_cases_l2014_201487

/-- The number of paper clips in 2 cases -/
def paperClipsIn2Cases (c b : ℕ) : ℕ := 2 * c * b * 400

/-- Theorem: The number of paper clips in 2 cases is 2 * c * b * 400 -/
theorem paper_clips_in_2_cases (c b : ℕ) : paperClipsIn2Cases c b = 2 * c * b * 400 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_in_2_cases_l2014_201487


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2014_201417

theorem min_value_of_expression (a b : ℕ+) (h : a > b) :
  let E := |(a + 2*b : ℝ) / (a - b : ℝ) + (a - b : ℝ) / (a + 2*b : ℝ)|
  ∀ x : ℝ, E ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2014_201417


namespace NUMINAMATH_CALUDE_triangle_side_length_l2014_201407

theorem triangle_side_length (b c : ℝ) (cosA : ℝ) (h1 : b = 3) (h2 : c = 5) (h3 : cosA = -1/2) :
  ∃ a : ℝ, a^2 = b^2 + c^2 - 2*b*c*cosA ∧ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2014_201407


namespace NUMINAMATH_CALUDE_closest_fraction_l2014_201437

def medals_won : ℚ := 23 / 150

def fractions : List ℚ := [1/6, 1/7, 1/8, 1/9, 1/10]

theorem closest_fraction :
  (fractions.argmin (λ x => |x - medals_won|)).get! = 1/7 := by sorry

end NUMINAMATH_CALUDE_closest_fraction_l2014_201437


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_l2014_201479

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_parallel_plane
  (α β : Plane) (l : Line)
  (h1 : perpendicular l α)
  (h2 : parallel α β) :
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_l2014_201479


namespace NUMINAMATH_CALUDE_committee_formation_possibilities_l2014_201405

/-- The number of ways to choose k elements from a set of n elements --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of members in the club --/
def club_size : ℕ := 25

/-- The size of the executive committee --/
def committee_size : ℕ := 4

/-- Theorem stating that choosing 4 people from 25 results in 12650 possibilities --/
theorem committee_formation_possibilities :
  choose club_size committee_size = 12650 := by sorry

end NUMINAMATH_CALUDE_committee_formation_possibilities_l2014_201405


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l2014_201461

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine (contract_days : ℕ) (daily_pay : ℚ) (absent_days : ℕ) (total_payment : ℚ) : ℚ :=
  let worked_days := contract_days - absent_days
  let earned_amount := worked_days * daily_pay
  (earned_amount - total_payment) / absent_days

theorem contractor_fine_calculation :
  let contract_days : ℕ := 30
  let daily_pay : ℚ := 25
  let absent_days : ℕ := 6
  let total_payment : ℚ := 555
  calculate_fine contract_days daily_pay absent_days total_payment = 7.5 := by
  sorry

#eval calculate_fine 30 25 6 555

end NUMINAMATH_CALUDE_contractor_fine_calculation_l2014_201461


namespace NUMINAMATH_CALUDE_average_price_is_16_l2014_201484

/-- The average price of books bought by Rahim -/
def average_price_per_book (books_shop1 books_shop2 : ℕ) (price_shop1 price_shop2 : ℕ) : ℚ :=
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2)

/-- Theorem stating that the average price per book is 16 given the problem conditions -/
theorem average_price_is_16 :
  average_price_per_book 55 60 1500 340 = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_price_is_16_l2014_201484


namespace NUMINAMATH_CALUDE_ratio_of_P_and_Q_l2014_201455

theorem ratio_of_P_and_Q (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4*x) = (x^2 + x + 15) / (x^3 + x^2 - 20*x)) →
  (Q : ℚ) / P = -45 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_P_and_Q_l2014_201455


namespace NUMINAMATH_CALUDE_max_stamps_purchased_l2014_201476

/-- Given a stamp price of 45 cents and $50 to spend, 
    the maximum number of stamps that can be purchased is 111. -/
theorem max_stamps_purchased (stamp_price : ℕ) (budget : ℕ) : 
  stamp_price = 45 → budget = 5000 → 
  (∀ n : ℕ, n * stamp_price ≤ budget → n ≤ 111) ∧ 
  111 * stamp_price ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchased_l2014_201476


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2014_201410

theorem unique_two_digit_number : 
  ∃! (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (Nat.factorial a - a * b) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l2014_201410


namespace NUMINAMATH_CALUDE_visited_none_count_l2014_201424

/-- Represents the number of people who have visited a country or combination of countries. -/
structure VisitCount where
  total : Nat
  iceland : Nat
  norway : Nat
  sweden : Nat
  all_three : Nat
  iceland_norway : Nat
  iceland_sweden : Nat
  norway_sweden : Nat

/-- Calculates the number of people who have visited neither Iceland, Norway, nor Sweden. -/
def people_visited_none (vc : VisitCount) : Nat :=
  vc.total - (vc.iceland + vc.norway + vc.sweden - vc.iceland_norway - vc.iceland_sweden - vc.norway_sweden + vc.all_three)

/-- Theorem stating that given the conditions, 42 people have visited neither country. -/
theorem visited_none_count (vc : VisitCount) 
  (h_total : vc.total = 100)
  (h_iceland : vc.iceland = 45)
  (h_norway : vc.norway = 37)
  (h_sweden : vc.sweden = 21)
  (h_all_three : vc.all_three = 12)
  (h_iceland_norway : vc.iceland_norway = 20)
  (h_iceland_sweden : vc.iceland_sweden = 15)
  (h_norway_sweden : vc.norway_sweden = 10) :
  people_visited_none vc = 42 := by
  sorry

end NUMINAMATH_CALUDE_visited_none_count_l2014_201424


namespace NUMINAMATH_CALUDE_max_sum_scores_max_sum_scores_achievable_l2014_201454

/-- Represents the scoring system for an exam -/
structure ExamScoring where
  m : ℕ             -- number of questions
  n : ℕ             -- number of students
  x : Fin m → ℕ     -- number of students who answered each question incorrectly
  h_m : m ≥ 2
  h_n : n ≥ 2
  h_x : ∀ k, x k ≤ n

/-- The score of a student -/
def student_score (E : ExamScoring) : ℕ → ℕ := sorry

/-- The highest score in the exam -/
def max_score (E : ExamScoring) : ℕ := sorry

/-- The lowest score in the exam -/
def min_score (E : ExamScoring) : ℕ := sorry

/-- Theorem: The maximum possible sum of the highest and lowest scores is m(n-1) -/
theorem max_sum_scores (E : ExamScoring) : 
  max_score E + min_score E ≤ E.m * (E.n - 1) :=
sorry

/-- Theorem: The maximum sum of scores is achievable -/
theorem max_sum_scores_achievable (m n : ℕ) (h_m : m ≥ 2) (h_n : n ≥ 2) : 
  ∃ E : ExamScoring, E.m = m ∧ E.n = n ∧ max_score E + min_score E = m * (n - 1) :=
sorry

end NUMINAMATH_CALUDE_max_sum_scores_max_sum_scores_achievable_l2014_201454


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2014_201494

theorem inequality_equivalence (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  (x * z^2 / z > y * z^2 / z) ↔ (x > y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2014_201494


namespace NUMINAMATH_CALUDE_rabbit_weeks_calculation_l2014_201489

/-- The number of weeks Julia has had the rabbit -/
def weeks_with_rabbit : ℕ := 2

/-- The total weekly cost for both animals' food -/
def total_weekly_cost : ℕ := 30

/-- The number of weeks Julia has had the parrot -/
def weeks_with_parrot : ℕ := 3

/-- The total spent on food so far -/
def total_spent : ℕ := 114

/-- The weekly cost of rabbit food -/
def weekly_rabbit_cost : ℕ := 12

theorem rabbit_weeks_calculation :
  weeks_with_rabbit * weekly_rabbit_cost + weeks_with_parrot * total_weekly_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_rabbit_weeks_calculation_l2014_201489


namespace NUMINAMATH_CALUDE_papers_per_envelope_l2014_201422

theorem papers_per_envelope 
  (total_papers : ℕ) 
  (num_envelopes : ℕ) 
  (h1 : total_papers = 120) 
  (h2 : num_envelopes = 12) : 
  total_papers / num_envelopes = 10 := by
sorry

end NUMINAMATH_CALUDE_papers_per_envelope_l2014_201422


namespace NUMINAMATH_CALUDE_cindys_calculation_l2014_201456

theorem cindys_calculation (x : ℝ) : 
  (x - 4) / 7 = 43 → (x - 7) / 4 = 74.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l2014_201456


namespace NUMINAMATH_CALUDE_sum_of_squared_pairs_l2014_201401

theorem sum_of_squared_pairs (p q r : ℝ) : 
  (p^3 - 18*p^2 + 25*p - 6 = 0) →
  (q^3 - 18*q^2 + 25*q - 6 = 0) →
  (r^3 - 18*r^2 + 25*r - 6 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 598 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_pairs_l2014_201401


namespace NUMINAMATH_CALUDE_insect_leg_paradox_l2014_201485

theorem insect_leg_paradox (total_legs : ℕ) (six_leg_insects : ℕ) (eight_leg_insects : ℕ) 
  (h1 : total_legs = 190)
  (h2 : 6 * six_leg_insects = 78)
  (h3 : 8 * eight_leg_insects = 24) :
  ¬∃ (ten_leg_insects : ℕ), 
    6 * six_leg_insects + 8 * eight_leg_insects + 10 * ten_leg_insects = total_legs :=
by sorry

end NUMINAMATH_CALUDE_insect_leg_paradox_l2014_201485


namespace NUMINAMATH_CALUDE_cyclic_matrix_squared_identity_l2014_201448

/-- A 4x4 complex matrix with a cyclic structure -/
def CyclicMatrix (a b c d : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  !![a, b, c, d;
     b, c, d, a;
     c, d, a, b;
     d, a, b, c]

theorem cyclic_matrix_squared_identity
  (a b c d : ℂ)
  (h1 : (CyclicMatrix a b c d) ^ 2 = 1)
  (h2 : a * b * c * d = 1) :
  a ^ 4 + b ^ 4 + c ^ 4 + d ^ 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_matrix_squared_identity_l2014_201448


namespace NUMINAMATH_CALUDE_complement_of_complement_l2014_201408

theorem complement_of_complement (α : ℝ) (h : α = 35) :
  90 - (90 - α) = α := by sorry

end NUMINAMATH_CALUDE_complement_of_complement_l2014_201408


namespace NUMINAMATH_CALUDE_gold_bar_weight_l2014_201411

/-- Proves that in an arithmetic sequence of 5 terms where the first term is 4 
    and the last term is 2, the second term is 7/2. -/
theorem gold_bar_weight (a : Fin 5 → ℚ) 
  (h_arith : ∀ i j : Fin 5, a j - a i = (j - i : ℚ) * (a 1 - a 0))
  (h_first : a 0 = 4)
  (h_last : a 4 = 2) : 
  a 1 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_gold_bar_weight_l2014_201411


namespace NUMINAMATH_CALUDE_water_displaced_squared_l2014_201430

/-- The volume of water displaced by a cube submerged in a cylindrical barrel -/
def water_displaced (cube_side : ℝ) (barrel_radius : ℝ) (barrel_height : ℝ) : ℝ :=
  cube_side ^ 3

/-- Theorem: The square of the volume of water displaced by a 10-foot cube
    in a cylindrical barrel is 1,000,000 cubic feet -/
theorem water_displaced_squared :
  let cube_side : ℝ := 10
  let barrel_radius : ℝ := 5
  let barrel_height : ℝ := 15
  (water_displaced cube_side barrel_radius barrel_height) ^ 2 = 1000000 := by
sorry

end NUMINAMATH_CALUDE_water_displaced_squared_l2014_201430


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2014_201472

/-- Represents the selling and buying of a bicycle through two transactions -/
def bicycle_sales (initial_cost : ℝ) : Prop :=
  let first_sale := initial_cost * 1.5
  let final_sale := first_sale * 1.25
  final_sale = 225

theorem bicycle_cost_price : ∃ (initial_cost : ℝ), 
  bicycle_sales initial_cost ∧ initial_cost = 120 := by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l2014_201472


namespace NUMINAMATH_CALUDE_equation_solution_l2014_201420

theorem equation_solution : 
  ∃ x : ℝ, (5 + 3.2 * x = 2.4 * x - 15) ∧ (x = -25) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2014_201420


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l2014_201439

theorem little_john_money_distribution 
  (initial_amount : ℚ)
  (spent_on_sweets : ℚ)
  (num_friends : ℕ)
  (remaining_amount : ℚ)
  (h1 : initial_amount = 10.10)
  (h2 : spent_on_sweets = 3.25)
  (h3 : num_friends = 2)
  (h4 : remaining_amount = 2.45) :
  (initial_amount - spent_on_sweets - remaining_amount) / num_friends = 2.20 :=
by sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l2014_201439


namespace NUMINAMATH_CALUDE_sum_of_inverse_conjugates_l2014_201499

theorem sum_of_inverse_conjugates (m n : ℝ) : 
  m = (Real.sqrt 2 - 1)⁻¹ → n = (Real.sqrt 2 + 1)⁻¹ → m + n = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inverse_conjugates_l2014_201499


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l2014_201419

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogram_area (v w : Fin 2 → ℝ) : ℝ :=
  |v 0 * w 1 - v 1 * w 0|

theorem parallelogram_area_specific_vectors :
  let v : Fin 2 → ℝ := ![7, -4]
  let w : Fin 2 → ℝ := ![12, -1]
  parallelogram_area v w = 41 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_vectors_l2014_201419


namespace NUMINAMATH_CALUDE_farm_animals_l2014_201412

/-- Given a farm with hens and cows, prove the number of hens -/
theorem farm_animals (total_heads : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) :
  total_heads = 44 →
  total_feet = 140 →
  total_heads = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 18 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2014_201412


namespace NUMINAMATH_CALUDE_basketball_games_count_l2014_201463

/-- Proves that a basketball team played 94 games in a season given specific conditions -/
theorem basketball_games_count :
  ∀ (total_games : ℕ) 
    (first_40_wins : ℕ) 
    (remaining_wins : ℕ),
  first_40_wins = 14 →  -- 35% of 40 games
  remaining_wins ≥ (0.7 : ℝ) * (total_games - 40) →  -- At least 70% of remaining games
  first_40_wins + remaining_wins = (0.55 : ℝ) * total_games →  -- 55% total win rate
  total_games = 94 := by
sorry

end NUMINAMATH_CALUDE_basketball_games_count_l2014_201463


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2014_201462

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def line (x y : ℝ) : Prop := y = 2*x - 4

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the length of chord AB
def chordLength : ℝ := sorry

-- Define point P on the parabola
def P : ℝ × ℝ := sorry

-- Define the area of triangle ABP
def triangleArea : ℝ := sorry

theorem parabola_line_intersection :
  (parabola A.1 A.2 ∧ line A.1 A.2) ∧
  (parabola B.1 B.2 ∧ line B.1 B.2) ∧
  chordLength = 3 * Real.sqrt 5 ∧
  parabola P.1 P.2 ∧
  triangleArea = 12 →
  (P = (9, 6) ∨ P = (4, -4)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2014_201462


namespace NUMINAMATH_CALUDE_max_garden_area_l2014_201470

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the fencing required for three sides of a rectangular garden -/
def fencingRequired (d : GardenDimensions) : ℝ :=
  d.length + 2 * d.width

/-- Theorem: The maximum area of a rectangular garden with 400 feet of fencing
    for three sides is 20000 square feet -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    fencingRequired d = 400 ∧
    ∀ (d' : GardenDimensions), fencingRequired d' = 400 →
      gardenArea d' ≤ gardenArea d ∧
      gardenArea d = 20000 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l2014_201470


namespace NUMINAMATH_CALUDE_log_condition_equivalence_l2014_201403

theorem log_condition_equivalence (m n : ℝ) (hm : m > 0 ∧ m ≠ 1) (hn : n > 0) :
  Real.log n / Real.log m < 0 ↔ (m - 1) * (n - 1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_condition_equivalence_l2014_201403


namespace NUMINAMATH_CALUDE_lindas_tv_cost_l2014_201478

/-- The cost of Linda's TV purchase, given her original savings and furniture expenses -/
theorem lindas_tv_cost (original_savings : ℝ) (furniture_fraction : ℝ) : 
  original_savings = 800 →
  furniture_fraction = 3/4 →
  original_savings * (1 - furniture_fraction) = 200 := by
sorry

end NUMINAMATH_CALUDE_lindas_tv_cost_l2014_201478


namespace NUMINAMATH_CALUDE_jasons_house_paintable_area_l2014_201440

/-- The total area to be painted in multiple identical bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_rooms * paintable_area

/-- Theorem stating the total area to be painted in Jason's house -/
theorem jasons_house_paintable_area :
  total_paintable_area 4 14 11 9 80 = 1480 := by
  sorry

end NUMINAMATH_CALUDE_jasons_house_paintable_area_l2014_201440


namespace NUMINAMATH_CALUDE_distance_between_squares_l2014_201402

/-- Given two squares where the smaller square has a perimeter of 8 cm and the larger square has an area of 36 cm², 
    prove that the distance between opposite corners of the two squares is approximately 8.9 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ) 
  (h1 : small_square_perimeter = 8) 
  (h2 : large_square_area = 36) : 
  ∃ (distance : ℝ), abs (distance - Real.sqrt 80) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_squares_l2014_201402


namespace NUMINAMATH_CALUDE_lcm_of_proportional_numbers_l2014_201450

def A : ℕ := 18
def B : ℕ := 24
def C : ℕ := 30

theorem lcm_of_proportional_numbers :
  (A : ℕ) / gcd A B = 3 ∧
  (B : ℕ) / gcd A B = 4 ∧
  (C : ℕ) / gcd A B = 5 ∧
  gcd A (gcd B C) = 6 ∧
  12 ∣ lcm A (lcm B C) →
  lcm A (lcm B C) = 360 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_proportional_numbers_l2014_201450


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l2014_201488

/-- Given a polynomial function g(x) = ax^5 + bx^3 + cx - 3 where g(-5) = 3, prove that g(5) = -9 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let g : ℝ → ℝ := λ x => a * x^5 + b * x^3 + c * x - 3
  g (-5) = 3 → g 5 = -9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l2014_201488


namespace NUMINAMATH_CALUDE_line_shift_l2014_201483

/-- The vertical shift of a line -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := fun x ↦ f x + shift

/-- The original line equation -/
def original_line : ℝ → ℝ := fun x ↦ 3 * x - 2

/-- Theorem: Moving the line y = 3x - 2 up by 6 units results in y = 3x + 4 -/
theorem line_shift :
  vertical_shift original_line 6 = fun x ↦ 3 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_line_shift_l2014_201483


namespace NUMINAMATH_CALUDE_raghu_investment_l2014_201418

/-- Represents the investment amounts of Raghu, Trishul, and Vishal --/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ

/-- Defines the conditions of the investment problem --/
def InvestmentConditions (i : Investments) : Prop :=
  i.trishul = 0.9 * i.raghu ∧
  i.vishal = 1.1 * i.trishul ∧
  i.raghu + i.trishul + i.vishal = 7225

/-- Theorem stating that under the given conditions, Raghu's investment is 2500 --/
theorem raghu_investment (i : Investments) (h : InvestmentConditions i) : i.raghu = 2500 := by
  sorry


end NUMINAMATH_CALUDE_raghu_investment_l2014_201418


namespace NUMINAMATH_CALUDE_consecutive_squares_divisors_l2014_201432

theorem consecutive_squares_divisors :
  ∃ (n : ℕ), 
    (∃ (a : ℕ), a > 1 ∧ a * a ∣ n) ∧
    (∃ (b : ℕ), b > 1 ∧ b * b ∣ (n + 1)) ∧
    (∃ (c : ℕ), c > 1 ∧ c * c ∣ (n + 2)) ∧
    (∃ (d : ℕ), d > 1 ∧ d * d ∣ (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_divisors_l2014_201432


namespace NUMINAMATH_CALUDE_bracelet_price_is_4_l2014_201431

/-- The price of a bracelet in dollars -/
def bracelet_price : ℝ := sorry

/-- The price of a keychain in dollars -/
def keychain_price : ℝ := 5

/-- The price of a coloring book in dollars -/
def coloring_book_price : ℝ := 3

/-- The total cost of the purchases -/
def total_cost : ℝ := 20

theorem bracelet_price_is_4 :
  2 * bracelet_price + keychain_price + bracelet_price + coloring_book_price = total_cost →
  bracelet_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_price_is_4_l2014_201431


namespace NUMINAMATH_CALUDE_max_AB_is_five_l2014_201447

/-- Represents a convex quadrilateral ABCD inscribed in a circle -/
structure CyclicQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  AB_shortest : AB ≤ BC ∧ AB ≤ CD ∧ AB ≤ DA
  distinct_sides : AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA
  max_side_10 : AB ≤ 10 ∧ BC ≤ 10 ∧ CD ≤ 10 ∧ DA ≤ 10
  area_ratio_int : ∃ k : ℕ, BC * CD = k * DA * AB

/-- The maximum possible value of AB in a CyclicQuadrilateral is 5 -/
theorem max_AB_is_five (q : CyclicQuadrilateral) : q.AB ≤ 5 :=
  sorry

end NUMINAMATH_CALUDE_max_AB_is_five_l2014_201447


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2014_201400

theorem necessary_but_not_sufficient_condition 
  (A B C : Set α) 
  (h_nonempty_A : A.Nonempty) 
  (h_nonempty_B : B.Nonempty) 
  (h_nonempty_C : C.Nonempty)
  (h_union : A ∪ B = C) 
  (h_not_subset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2014_201400


namespace NUMINAMATH_CALUDE_no_intersection_at_roots_l2014_201434

theorem no_intersection_at_roots : ∀ x : ℝ, 
  (x^2 - 3*x + 2 = 0) → 
  ¬(x^2 - 1 = 3*x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_intersection_at_roots_l2014_201434


namespace NUMINAMATH_CALUDE_sequence_properties_l2014_201438

/-- Proof of properties of sequences A, G, and H -/
theorem sequence_properties
  (x y k : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hk : k > 0)
  (hxy : x ≠ y)
  (hk1 : k ≠ 1)
  (A : ℕ → ℝ)
  (G : ℕ → ℝ)
  (H : ℕ → ℝ)
  (hA1 : A 1 = (k * x + y) / (k + 1))
  (hG1 : G 1 = (x^k * y)^(1 / (k + 1)))
  (hH1 : H 1 = ((k + 1) * x * y) / (k * x + y))
  (hAn : ∀ n ≥ 2, A n = (A (n-1) + H (n-1)) / 2)
  (hGn : ∀ n ≥ 2, G n = (A (n-1) * H (n-1))^(1/2))
  (hHn : ∀ n ≥ 2, H n = 2 / (1 / A (n-1) + 1 / H (n-1))) :
  (∀ n ≥ 1, A (n+1) < A n) ∧
  (∀ n ≥ 1, G (n+1) = G n) ∧
  (∀ n ≥ 1, H n < H (n+1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2014_201438


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l2014_201497

theorem smallest_x_abs_equation : ∃ x : ℝ, (∀ y : ℝ, |y + 3| = 15 → x ≤ y) ∧ |x + 3| = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l2014_201497


namespace NUMINAMATH_CALUDE_max_value_h_two_roots_range_max_positive_integer_a_l2014_201425

noncomputable section

open Real

-- Define the functions
def f (x : ℝ) := exp x
def g (a b : ℝ) (x : ℝ) := (a / 2) * x + b
def h (a b : ℝ) (x : ℝ) := f x * g a b x

-- Statement 1
theorem max_value_h (a b : ℝ) :
  a = -4 → b = 1 - a / 2 →
  ∃ (M : ℝ), M = 2 * exp (1 / 2) ∧ ∀ x ∈ Set.Icc 0 1, h a b x ≤ M :=
sorry

-- Statement 2
theorem two_roots_range (b : ℝ) :
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ f x₁ = g 4 b x₁ ∧ f x₂ = g 4 b x₂) ↔
  b ∈ Set.Ioo (2 - 2 * log 2) 1 :=
sorry

-- Statement 3
theorem max_positive_integer_a :
  ∃ (a : ℕ), a = 14 ∧ ∀ x : ℝ, f x > g a (-15/2) x ∧
  ∀ n : ℕ, n > a → ∃ y : ℝ, f y ≤ g n (-15/2) y :=
sorry

end

end NUMINAMATH_CALUDE_max_value_h_two_roots_range_max_positive_integer_a_l2014_201425


namespace NUMINAMATH_CALUDE_someone_next_to_two_economists_l2014_201404

/-- Represents the profession of a person -/
inductive Profession
| Accountant
| Manager
| Economist

/-- Represents a circular arrangement of people -/
def CircularArrangement := List Profession

/-- Counts the number of accountants sitting next to at least one economist -/
def accountantsNextToEconomist (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of managers sitting next to at least one economist -/
def managersNextToEconomist (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if there's someone sitting next to two economists -/
def someoneNextToTwoEconomists (arrangement : CircularArrangement) : Bool :=
  sorry

theorem someone_next_to_two_economists 
  (arrangement : CircularArrangement) : 
  accountantsNextToEconomist arrangement = 20 →
  managersNextToEconomist arrangement = 25 →
  someoneNextToTwoEconomists arrangement = true :=
by sorry

end NUMINAMATH_CALUDE_someone_next_to_two_economists_l2014_201404


namespace NUMINAMATH_CALUDE_total_birds_on_fence_l2014_201406

def birds_on_fence (initial : ℕ) (additional : ℕ) : ℕ := initial + additional

theorem total_birds_on_fence :
  birds_on_fence 12 8 = 20 := by sorry

end NUMINAMATH_CALUDE_total_birds_on_fence_l2014_201406


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2014_201498

theorem polynomial_factorization (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2014_201498


namespace NUMINAMATH_CALUDE_minimum_parents_needed_tour_parents_theorem_l2014_201477

theorem minimum_parents_needed (num_children : ℕ) (car_capacity : ℕ) : ℕ :=
  let total_people := num_children
  let cars_needed := (total_people + car_capacity - 1) / car_capacity
  cars_needed

theorem tour_parents_theorem :
  minimum_parents_needed 50 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_parents_needed_tour_parents_theorem_l2014_201477


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2014_201473

theorem magic_8_ball_probability : 
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 3  -- number of positive answers
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2014_201473


namespace NUMINAMATH_CALUDE_all_reachable_l2014_201452

def step (x : ℚ) : Set ℚ := {x + 1, -1 / x}

def reachable : Set ℚ → Prop :=
  λ S => ∀ y ∈ S, ∃ n : ℕ, ∃ f : ℕ → ℚ,
    f 0 = 1 ∧ (∀ i < n, f (i + 1) ∈ step (f i)) ∧ f n = y

theorem all_reachable : reachable {-2, 1/2, 5/3, 7} := by
  sorry

end NUMINAMATH_CALUDE_all_reachable_l2014_201452


namespace NUMINAMATH_CALUDE_smallest_quotient_l2014_201444

/-- Represents a three-digit number with different non-zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_nonzero : hundreds ≠ 0
  t_nonzero : tens ≠ 0
  o_nonzero : ones ≠ 0
  h_lt_ten : hundreds < 10
  t_lt_ten : tens < 10
  o_lt_ten : ones < 10
  all_different : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

/-- The value of a ThreeDigitNumber -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a ThreeDigitNumber -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The quotient of a ThreeDigitNumber divided by its digit sum -/
def quotient (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digitSum n : Rat)

theorem smallest_quotient :
  ∃ (n : ThreeDigitNumber), ∀ (m : ThreeDigitNumber), quotient n ≤ quotient m ∧ quotient n = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_quotient_l2014_201444


namespace NUMINAMATH_CALUDE_possible_signs_l2014_201490

theorem possible_signs (a b c : ℝ) : 
  a + b + c = 0 → 
  abs a > abs b → 
  abs b > abs c → 
  ∃ (a' b' c' : ℝ), a' + b' + c' = 0 ∧ 
                     abs a' > abs b' ∧ 
                     abs b' > abs c' ∧ 
                     c' > 0 ∧ 
                     a' < 0 :=
sorry

end NUMINAMATH_CALUDE_possible_signs_l2014_201490


namespace NUMINAMATH_CALUDE_product_selection_events_l2014_201449

structure ProductSelection where
  total : Nat
  genuine : Nat
  defective : Nat
  selected : Nat

def is_random_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∃ (outcome : Nat), event outcome ∧
  ∃ (outcome : Nat), ¬event outcome

def is_impossible_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∀ (outcome : Nat), ¬event outcome

def is_certain_event (ps : ProductSelection) (event : Nat → Prop) : Prop :=
  ∀ (outcome : Nat), event outcome

def all_genuine (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome = ps.genuine.choose ps.selected

def at_least_one_defective (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome > 0

def all_defective (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome = ps.defective.choose ps.selected

def at_least_one_genuine (ps : ProductSelection) (outcome : Nat) : Prop :=
  outcome < ps.selected

theorem product_selection_events (ps : ProductSelection) 
  (h1 : ps.total = 12)
  (h2 : ps.genuine = 10)
  (h3 : ps.defective = 2)
  (h4 : ps.selected = 3)
  (h5 : ps.total = ps.genuine + ps.defective) :
  is_random_event ps (all_genuine ps) ∧
  is_random_event ps (at_least_one_defective ps) ∧
  is_impossible_event ps (all_defective ps) ∧
  is_certain_event ps (at_least_one_genuine ps) := by
  sorry

end NUMINAMATH_CALUDE_product_selection_events_l2014_201449


namespace NUMINAMATH_CALUDE_potential_solution_check_l2014_201471

theorem potential_solution_check (x y : ℕ+) (h : 1 + 2^x.val + 2^(2*x.val+1) = y.val^2) : 
  x = 3 ∨ ∃ z : ℕ+, (1 + 2^z.val + 2^(2*z.val+1) = y.val^2 ∧ z ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_potential_solution_check_l2014_201471


namespace NUMINAMATH_CALUDE_train_crossing_time_l2014_201467

/-- Given a train and two platforms, calculate the time to cross the first platform -/
theorem train_crossing_time (Lt Lp1 Lp2 Tp2 : ℝ) (h1 : Lt = 30)
    (h2 : Lp1 = 180) (h3 : Lp2 = 250) (h4 : Tp2 = 20) :
  (Lt + Lp1) / ((Lt + Lp2) / Tp2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2014_201467


namespace NUMINAMATH_CALUDE_min_groups_for_30_students_max_6_l2014_201475

/-- Given a total number of students and a maximum group size, 
    calculate the smallest number of equal-sized groups. -/
def minGroups (totalStudents : ℕ) (maxGroupSize : ℕ) : ℕ :=
  (totalStudents + maxGroupSize - 1) / maxGroupSize

/-- Theorem: For 30 students and a maximum group size of 6, 
    the smallest number of equal-sized groups is 5. -/
theorem min_groups_for_30_students_max_6 :
  minGroups 30 6 = 5 := by sorry

end NUMINAMATH_CALUDE_min_groups_for_30_students_max_6_l2014_201475


namespace NUMINAMATH_CALUDE_line_is_intersection_l2014_201445

/-- The line of intersection of two planes -/
def line_of_intersection (p₁ p₂ : ℝ → ℝ → ℝ → Prop) : ℝ → ℝ → ℝ → Prop :=
  λ x y z => (x + 3) / (-3) = y / (-4) ∧ y / (-4) = z / (-9)

/-- First plane equation -/
def plane1 : ℝ → ℝ → ℝ → Prop :=
  λ x y z => 2*x + 3*y - 2*z + 6 = 0

/-- Second plane equation -/
def plane2 : ℝ → ℝ → ℝ → Prop :=
  λ x y z => x - 3*y + z + 3 = 0

/-- Theorem stating that the line is the intersection of the two planes -/
theorem line_is_intersection :
  ∀ x y z, line_of_intersection plane1 plane2 x y z ↔ (plane1 x y z ∧ plane2 x y z) :=
sorry

end NUMINAMATH_CALUDE_line_is_intersection_l2014_201445


namespace NUMINAMATH_CALUDE_angle_bisector_length_l2014_201482

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 8 ∧ qr = 15 ∧ pr = 17

-- Define the angle bisector QS
def AngleBisector (P Q R S : ℝ × ℝ) : Prop :=
  let ps := Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2)
  let rs := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  ps / rs = 8 / 15

-- Theorem statement
theorem angle_bisector_length (P Q R S : ℝ × ℝ) :
  Triangle P Q R → AngleBisector P Q R S →
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 4 * Real.sqrt 3272 / 23 :=
by sorry


end NUMINAMATH_CALUDE_angle_bisector_length_l2014_201482


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2014_201421

theorem regular_polygon_sides : ∀ n : ℕ, 
  n > 2 → (3 * (n * (n - 3) / 2) - n = 21 ↔ n = 6) := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2014_201421


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2014_201480

/-- Given a right triangle with sides 5, 12, and 13, x is the side length of a square
    inscribed with one vertex at the right angle, and y is the side length of a square
    inscribed with one side on the longest leg (12). -/
def triangle_with_squares (x y : ℝ) : Prop :=
  -- Right triangle condition
  5^2 + 12^2 = 13^2 ∧
  -- Condition for square with side x
  x / 5 = x / 12 ∧
  -- Condition for square with side y
  y + y = 12

/-- The ratio of the side lengths of the two inscribed squares is 10/17. -/
theorem inscribed_squares_ratio :
  ∀ x y : ℝ, triangle_with_squares x y → x / y = 10 / 17 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2014_201480


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2014_201442

/-- Given an arithmetic sequence with certain properties, prove its maximum sum -/
theorem arithmetic_sequence_max_sum (k : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) :
  k ≥ 2 →
  S (k - 1) = 8 →
  S k = 0 →
  S (k + 1) = -10 →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∃ d : ℤ, ∀ n, a (n + 1) - a n = d) →
  (∃ n : ℕ, ∀ m : ℕ, S m ≤ S n) →
  (∃ n : ℕ, S n = 20) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l2014_201442


namespace NUMINAMATH_CALUDE_chocolate_cuts_l2014_201492

/-- The minimum number of cuts required to divide a single piece into n pieces -/
def min_cuts (n : ℕ) : ℕ := n - 1

/-- Theorem stating that the minimum number of cuts to get 24 pieces is 23 -/
theorem chocolate_cuts : min_cuts 24 = 23 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cuts_l2014_201492


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2014_201465

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 1) :
  let expr := ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / ((x - 1)^2))
  expr = x - 1 ∧ (x = 5 → expr = 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2014_201465


namespace NUMINAMATH_CALUDE_min_value_theorem_l2014_201423

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 1) + 9 / y = 1) : 
  4 * x + y ≥ 21 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1 / (x₀ + 1) + 9 / y₀ = 1 ∧ 4 * x₀ + y₀ = 21 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2014_201423


namespace NUMINAMATH_CALUDE_monkey_bird_problem_l2014_201495

theorem monkey_bird_problem (initial_birds : ℕ) (eaten_birds : ℕ) (monkey_percentage : ℚ) : 
  initial_birds = 6 →
  eaten_birds = 2 →
  monkey_percentage = 6/10 →
  ∃ (initial_monkeys : ℕ), 
    initial_monkeys = 6 ∧
    (initial_monkeys : ℚ) / ((initial_monkeys : ℚ) + (initial_birds - eaten_birds : ℚ)) = monkey_percentage :=
by sorry

end NUMINAMATH_CALUDE_monkey_bird_problem_l2014_201495


namespace NUMINAMATH_CALUDE_solution_value_l2014_201426

theorem solution_value (a : ℝ) : (2 * (-1) + 3 * a = 4) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2014_201426


namespace NUMINAMATH_CALUDE_exactly_one_topic_not_chosen_l2014_201428

/-- The number of ways for n teachers to choose from m topics with replacement. -/
def choose_with_replacement (n m : ℕ) : ℕ := m ^ n

/-- The number of ways to arrange n items. -/
def arrangement (n : ℕ) : ℕ := n.factorial

/-- The number of ways for n teachers to choose from m topics with replacement,
    such that exactly one topic is not chosen. -/
def one_topic_not_chosen (n m : ℕ) : ℕ :=
  choose_with_replacement n m -
  (m * choose_with_replacement (n - 1) (m - 1)) -
  arrangement m

theorem exactly_one_topic_not_chosen :
  one_topic_not_chosen 4 4 = 112 := by sorry

end NUMINAMATH_CALUDE_exactly_one_topic_not_chosen_l2014_201428


namespace NUMINAMATH_CALUDE_ralph_cards_l2014_201413

/-- The number of cards Ralph has after various changes. -/
def final_cards (initial : ℕ) (from_father : ℕ) (from_sister : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_father + from_sister - traded - lost

/-- Theorem stating that Ralph ends up with 12 cards given the specific card changes. -/
theorem ralph_cards : final_cards 4 8 5 3 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ralph_cards_l2014_201413


namespace NUMINAMATH_CALUDE_multiplication_and_division_l2014_201427

theorem multiplication_and_division : 
  (8 * 4 = 32) ∧ (36 / 9 = 4) := by sorry

end NUMINAMATH_CALUDE_multiplication_and_division_l2014_201427


namespace NUMINAMATH_CALUDE_number_multiplying_a_l2014_201415

theorem number_multiplying_a (a b : ℝ) (h1 : ∃ x, x * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 8 = b / 7) :
  ∃ x, x * a = 8 * b ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplying_a_l2014_201415


namespace NUMINAMATH_CALUDE_more_books_than_maddie_l2014_201414

/-- Proves that Amy and Luisa have 9 more books than Maddie -/
theorem more_books_than_maddie 
  (maddie_books : ℕ) 
  (luisa_books : ℕ) 
  (amy_books : ℕ)
  (h1 : maddie_books = 15)
  (h2 : luisa_books = 18)
  (h3 : amy_books = 6) :
  luisa_books + amy_books - maddie_books = 9 := by
sorry

end NUMINAMATH_CALUDE_more_books_than_maddie_l2014_201414


namespace NUMINAMATH_CALUDE_select_defective_products_l2014_201458

def total_products : ℕ := 100
def defective_products : ℕ := 6
def products_to_select : ℕ := 3

theorem select_defective_products :
  Nat.choose total_products products_to_select -
  Nat.choose (total_products - defective_products) products_to_select =
  Nat.choose total_products products_to_select -
  Nat.choose 94 products_to_select :=
by sorry

end NUMINAMATH_CALUDE_select_defective_products_l2014_201458


namespace NUMINAMATH_CALUDE_temperatures_median_and_range_l2014_201486

def temperatures : List ℝ := [12, 9, 10, 6, 11, 12, 17]

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem temperatures_median_and_range :
  median temperatures = 11 ∧ range temperatures = 11 := by
  sorry

end NUMINAMATH_CALUDE_temperatures_median_and_range_l2014_201486


namespace NUMINAMATH_CALUDE_couple_ticket_cost_l2014_201453

theorem couple_ticket_cost (single_ticket_cost : ℚ) (total_sales : ℚ) 
  (total_attendance : ℕ) (couple_tickets_sold : ℕ) :
  single_ticket_cost = 20 →
  total_sales = 2280 →
  total_attendance = 128 →
  couple_tickets_sold = 16 →
  ∃ couple_ticket_cost : ℚ,
    couple_ticket_cost = 22.5 ∧
    total_sales = (total_attendance - 2 * couple_tickets_sold) * single_ticket_cost + 
                  couple_tickets_sold * couple_ticket_cost :=
by
  sorry


end NUMINAMATH_CALUDE_couple_ticket_cost_l2014_201453


namespace NUMINAMATH_CALUDE_tan_600_degrees_l2014_201416

theorem tan_600_degrees : Real.tan (600 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_600_degrees_l2014_201416


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l2014_201460

theorem lcm_of_ratio_and_sum (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 → a + b = 40 → Nat.lcm a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l2014_201460


namespace NUMINAMATH_CALUDE_income_ratio_l2014_201451

/-- Represents a person's financial information -/
structure Person where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- The problem setup -/
def problem_setup (p1 p2 : Person) : Prop :=
  p1.income = 5000 ∧
  p1.savings = 2000 ∧
  p2.savings = 2000 ∧
  3 * p2.expenditure = 2 * p1.expenditure ∧
  p1.income = p1.expenditure + p1.savings ∧
  p2.income = p2.expenditure + p2.savings

/-- The theorem to prove -/
theorem income_ratio (p1 p2 : Person) :
  problem_setup p1 p2 → 5 * p2.income = 4 * p1.income :=
by
  sorry


end NUMINAMATH_CALUDE_income_ratio_l2014_201451


namespace NUMINAMATH_CALUDE_product_evaluation_l2014_201446

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2014_201446


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2014_201464

def consecutive_integers (n : ℕ) (start : ℤ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

theorem largest_divisor_five_consecutive_integers :
  ∀ start : ℤ, 
  ∃ m : ℕ, m = 240 ∧ 
  (m : ℤ) ∣ (List.prod (consecutive_integers 5 start)) ∧
  ∀ k : ℕ, k > m → ¬((k : ℤ) ∣ (List.prod (consecutive_integers 5 start))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2014_201464


namespace NUMINAMATH_CALUDE_function_properties_l2014_201435

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem function_properties (a : ℝ) (h_min : ∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f a x ≤ f a y) 
  (h_min_value : ∃ (x : ℝ), x ∈ interval ∧ f a x = -37) :
  a = 3 ∧ ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), x ∈ interval → f a x ≤ m := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2014_201435


namespace NUMINAMATH_CALUDE_medium_stores_selected_is_ten_l2014_201457

/-- Represents the number of stores to be selected in a stratified sampling -/
def total_sample : ℕ := 30

/-- Represents the total number of stores -/
def total_stores : ℕ := 1500

/-- Represents the ratio of large stores -/
def large_ratio : ℕ := 1

/-- Represents the ratio of medium stores -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores -/
def small_ratio : ℕ := 9

/-- Calculates the number of medium-sized stores to be selected in the stratified sampling -/
def medium_stores_selected : ℕ := 
  (total_sample * medium_ratio) / (large_ratio + medium_ratio + small_ratio)

/-- Theorem stating that the number of medium-sized stores to be selected is 10 -/
theorem medium_stores_selected_is_ten : medium_stores_selected = 10 := by
  sorry


end NUMINAMATH_CALUDE_medium_stores_selected_is_ten_l2014_201457


namespace NUMINAMATH_CALUDE_inequality_proof_l2014_201466

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9)
  (h5 : a^2 + b^2 + c^2 + d^2 = 21) :
  a * b - c * d ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2014_201466


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2014_201474

theorem two_numbers_problem :
  ∃ (x y : ℕ), 
    x = y + 75 ∧
    x * y = (227 * y + 113) + 1000 ∧
    x > y ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2014_201474


namespace NUMINAMATH_CALUDE_homer_investment_interest_l2014_201481

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Proves that the interest earned on a $2000 investment at 2% for 3 years is $122.416 -/
theorem homer_investment_interest :
  let principal : ℝ := 2000
  let rate : ℝ := 0.02
  let time : ℕ := 3
  abs (interest_earned principal rate time - 122.416) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_homer_investment_interest_l2014_201481
