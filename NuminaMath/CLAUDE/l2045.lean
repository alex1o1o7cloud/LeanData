import Mathlib

namespace second_batch_weight_is_100_l2045_204509

-- Define the initial stock
def initial_stock : ℝ := 400

-- Define the percentage of decaf in initial stock
def initial_decaf_percent : ℝ := 0.20

-- Define the percentage of decaf in the second batch
def second_batch_decaf_percent : ℝ := 0.50

-- Define the final percentage of decaf in total stock
def final_decaf_percent : ℝ := 0.26

-- Define the weight of the second batch as a variable
variable (second_batch_weight : ℝ)

-- Theorem statement
theorem second_batch_weight_is_100 :
  (initial_stock * initial_decaf_percent + second_batch_weight * second_batch_decaf_percent) / 
  (initial_stock + second_batch_weight) = final_decaf_percent →
  second_batch_weight = 100 := by
  sorry

end second_batch_weight_is_100_l2045_204509


namespace typing_time_l2045_204563

/-- Proves that given Tom's typing speed and page length, it takes 50 minutes to type 10 pages -/
theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (pages : ℕ) : 
  typing_speed = 90 → words_per_page = 450 → pages = 10 → 
  (pages * words_per_page) / typing_speed = 50 := by
  sorry

end typing_time_l2045_204563


namespace calculate_product_l2045_204599

theorem calculate_product : 500 * 1986 * 0.3972 * 100 = 20 * 1986^2 := by
  sorry

end calculate_product_l2045_204599


namespace two_distinct_roots_implies_k_values_l2045_204564

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| + 1
def g (k : ℝ) (x : ℝ) : ℝ := k * x

-- State the theorem
theorem two_distinct_roots_implies_k_values (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g k x₁ ∧ f x₂ = g k x₂) →
  (k = 1/2 ∨ k = 2/3) :=
by sorry

end two_distinct_roots_implies_k_values_l2045_204564


namespace quadratic_roots_implies_composite_l2045_204579

/-- A number is composite if it's the product of two integers each greater than 1 -/
def IsComposite (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ n = p * q

/-- The roots of the quadratic x^2 + ax + b + 1 are positive integers -/
def HasPositiveIntegerRoots (a b : ℤ) : Prop :=
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c^2 + a*c + b + 1 = 0 ∧ d^2 + a*d + b + 1 = 0

/-- If x^2 + ax + b + 1 has positive integer roots, then a^2 + b^2 is composite -/
theorem quadratic_roots_implies_composite (a b : ℤ) :
  HasPositiveIntegerRoots a b → IsComposite (Int.natAbs (a^2 + b^2)) :=
sorry

end quadratic_roots_implies_composite_l2045_204579


namespace cube_sum_implies_sum_l2045_204585

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cube_sum_implies_sum_l2045_204585


namespace tournament_handshakes_correct_l2045_204515

/-- The number of handshakes in a tennis tournament with 3 teams of 2 players each --/
def tournament_handshakes : ℕ := 12

/-- The number of teams in the tournament --/
def num_teams : ℕ := 3

/-- The number of players per team --/
def players_per_team : ℕ := 2

/-- The total number of players in the tournament --/
def total_players : ℕ := num_teams * players_per_team

/-- The number of handshakes per player --/
def handshakes_per_player : ℕ := total_players - 2

theorem tournament_handshakes_correct :
  tournament_handshakes = (total_players * handshakes_per_player) / 2 :=
by sorry

end tournament_handshakes_correct_l2045_204515


namespace three_sequence_non_decreasing_indices_l2045_204505

theorem three_sequence_non_decreasing_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
sorry

end three_sequence_non_decreasing_indices_l2045_204505


namespace clock_equivalent_hours_l2045_204574

theorem clock_equivalent_hours : 
  ∃ n : ℕ, n > 5 ∧ 
           n * n - n ≡ 0 [MOD 12] ∧ 
           ∀ m : ℕ, m > 5 ∧ m < n → ¬(m * m - m ≡ 0 [MOD 12]) :=
by
  -- The proof goes here
  sorry

end clock_equivalent_hours_l2045_204574


namespace quadratic_distinct_roots_l2045_204524

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 9 = 0 ∧ x₂^2 + m*x₂ + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end quadratic_distinct_roots_l2045_204524


namespace b_8_equals_162_l2045_204581

/-- Given sequences {aₙ} and {bₙ} satisfying the specified conditions, b₈ equals 162 -/
theorem b_8_equals_162 (a b : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → (a n) * (a (n + 1)) = 3^n)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a n) + (a (n + 1)) = b n)
  (h4 : ∀ n : ℕ, n ≥ 1 → (a n)^2 - (b n) * (a n) + 3^n = 0) :
  b 8 = 162 := by
  sorry

end b_8_equals_162_l2045_204581


namespace divide_by_seven_l2045_204504

theorem divide_by_seven (x : ℚ) (h : x = 5/2) : x / 7 = 5/14 := by
  sorry

end divide_by_seven_l2045_204504


namespace custom_operation_solution_l2045_204556

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- State the theorem
theorem custom_operation_solution :
  ∃ x : ℝ, star 3 (star 4 x) = 10 ∧ x = 7.5 := by
  sorry

end custom_operation_solution_l2045_204556


namespace whatsapp_messages_l2045_204571

theorem whatsapp_messages (monday tuesday wednesday thursday : ℕ) :
  monday = 300 →
  tuesday = 200 →
  thursday = 2 * wednesday →
  monday + tuesday + wednesday + thursday = 2000 →
  wednesday - tuesday = 300 :=
by
  sorry

end whatsapp_messages_l2045_204571


namespace B_minus_A_equality_l2045_204506

def A : Set ℝ := {y | ∃ x, 1/3 ≤ x ∧ x ≤ 1 ∧ y = 1/x}
def B : Set ℝ := {y | ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ y = x^2 - 1}

theorem B_minus_A_equality : 
  B \ A = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end B_minus_A_equality_l2045_204506


namespace power_nine_mod_seven_l2045_204523

theorem power_nine_mod_seven : 9^123 % 7 = 1 := by
  sorry

end power_nine_mod_seven_l2045_204523


namespace optionB_is_suitable_only_optionB_is_suitable_l2045_204520

/-- Represents a sampling experiment --/
structure SamplingExperiment where
  sampleSize : Nat
  populationSize : Nat
  numFactories : Nat
  numBoxes : Nat

/-- Criteria for lottery method suitability --/
def isLotteryMethodSuitable (exp : SamplingExperiment) : Prop :=
  exp.sampleSize < 20 ∧ 
  exp.populationSize < 100 ∧ 
  exp.numFactories = 1 ∧
  exp.numBoxes > 1

/-- The four options given in the problem --/
def optionA : SamplingExperiment := ⟨600, 3000, 1, 1⟩
def optionB : SamplingExperiment := ⟨6, 30, 1, 2⟩
def optionC : SamplingExperiment := ⟨6, 30, 2, 2⟩
def optionD : SamplingExperiment := ⟨10, 3000, 1, 1⟩

/-- Theorem stating that option B is suitable for the lottery method --/
theorem optionB_is_suitable : isLotteryMethodSuitable optionB := by
  sorry

/-- Theorem stating that option B is the only suitable option --/
theorem only_optionB_is_suitable : 
  isLotteryMethodSuitable optionB ∧ 
  ¬isLotteryMethodSuitable optionA ∧ 
  ¬isLotteryMethodSuitable optionC ∧ 
  ¬isLotteryMethodSuitable optionD := by
  sorry

end optionB_is_suitable_only_optionB_is_suitable_l2045_204520


namespace sum_of_decimals_l2045_204569

theorem sum_of_decimals : 0.001 + 1.01 + 0.11 = 1.121 := by
  sorry

end sum_of_decimals_l2045_204569


namespace min_n_for_S_gt_1020_l2045_204590

def S (n : ℕ) : ℕ := 2^(n+1) - 2 - n

theorem min_n_for_S_gt_1020 : ∀ k : ℕ, k ≥ 10 ↔ S k > 1020 :=
sorry

end min_n_for_S_gt_1020_l2045_204590


namespace geometric_series_ratio_l2045_204544

/-- 
Given a geometric series with first term a and common ratio r,
prove that if the sum of the series is 24 and the sum of terms
with odd powers of r is 10, then r = 5/7.
-/
theorem geometric_series_ratio (a r : ℝ) : 
  (∑' n, a * r^n) = 24 →
  (∑' n, a * r^(2*n+1)) = 10 →
  r = 5/7 := by
sorry

end geometric_series_ratio_l2045_204544


namespace system_solution_existence_l2045_204549

theorem system_solution_existence (a : ℝ) :
  (∃ (x y b : ℝ), y = x^2 - a ∧ x^2 + y^2 + 8*b^2 = 4*b*(y - x) + 1) ↔ 
  a ≥ -Real.sqrt 2 - 1/4 := by sorry

end system_solution_existence_l2045_204549


namespace some_students_not_fraternity_members_l2045_204553

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Honest : U → Prop)
variable (FraternityMember : U → Prop)

-- Define the given conditions
axiom some_students_not_honest : ∃ x, Student x ∧ ¬Honest x
axiom all_fraternity_members_honest : ∀ x, FraternityMember x → Honest x

-- Theorem to prove
theorem some_students_not_fraternity_members : 
  ∃ x, Student x ∧ ¬FraternityMember x :=
sorry

end some_students_not_fraternity_members_l2045_204553


namespace train_speed_l2045_204593

/-- Given a train and a platform, calculate the speed of the train -/
theorem train_speed (train_length platform_length time : ℝ) 
  (h1 : train_length = 150)
  (h2 : platform_length = 250)
  (h3 : time = 8) :
  (train_length + platform_length) / time = 50 :=
by
  sorry

end train_speed_l2045_204593


namespace square_perimeter_problem_l2045_204501

/-- Given squares A, B, and C, prove that the perimeter of C is 48 -/
theorem square_perimeter_problem (A B C : ℝ) : 
  (4 * A = 16) →  -- Perimeter of A is 16
  (4 * B = 32) →  -- Perimeter of B is 32
  (C = A + B) →   -- Side length of C is sum of side lengths of A and B
  (4 * C = 48) := by  -- Perimeter of C is 48
sorry

end square_perimeter_problem_l2045_204501


namespace possible_values_of_a_l2045_204500

theorem possible_values_of_a : 
  {a : ℤ | ∃ b c : ℤ, ∀ x : ℝ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)} = {10, 14} := by
  sorry

end possible_values_of_a_l2045_204500


namespace catch_up_equation_l2045_204542

/-- The number of days it takes for a good horse to catch up with a slow horse -/
def catch_up_days (good_horse_speed slow_horse_speed : ℕ) (head_start : ℕ) : ℕ → Prop :=
  λ x => good_horse_speed * x = slow_horse_speed * x + slow_horse_speed * head_start

/-- Theorem stating the equation for the number of days it takes for the good horse to catch up -/
theorem catch_up_equation :
  let good_horse_speed := 240
  let slow_horse_speed := 150
  let head_start := 12
  ∃ x : ℕ, catch_up_days good_horse_speed slow_horse_speed head_start x :=
by
  sorry

end catch_up_equation_l2045_204542


namespace jackies_tree_climbing_ratio_l2045_204529

/-- Given the following conditions about Jackie's tree climbing:
  - Jackie climbed 4 trees in total
  - The first tree is 1000 feet tall
  - Two trees are of equal height
  - The fourth tree is 200 feet taller than the first tree
  - The average height of all trees is 800 feet

  Prove that the ratio of the height of the two equal trees to the height of the first tree is 1:2.
-/
theorem jackies_tree_climbing_ratio :
  ∀ (h₁ h₂ h₄ : ℝ),
  h₁ = 1000 →
  h₄ = h₁ + 200 →
  (h₁ + 2 * h₂ + h₄) / 4 = 800 →
  h₂ / h₁ = 1 / 2 :=
by sorry


end jackies_tree_climbing_ratio_l2045_204529


namespace find_x_l2045_204591

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 18 ∧ x = 11 := by
  sorry

end find_x_l2045_204591


namespace animus_tower_workers_l2045_204527

theorem animus_tower_workers (beavers spiders : ℕ) 
  (h1 : beavers = 318) 
  (h2 : spiders = 544) : 
  beavers + spiders = 862 := by
  sorry

end animus_tower_workers_l2045_204527


namespace train_length_l2045_204562

theorem train_length (v : ℝ) (L : ℝ) : 
  v > 0 → -- The train's speed is positive
  (L + 120) / 60 = v → -- It takes 60 seconds to pass through a 120m tunnel
  L / 20 = v → -- It takes 20 seconds to be completely inside the tunnel
  L = 60 := by
sorry

end train_length_l2045_204562


namespace factorization_of_2a_squared_minus_8_l2045_204543

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_of_2a_squared_minus_8_l2045_204543


namespace tan_value_from_ratio_l2045_204535

theorem tan_value_from_ratio (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan α = -3 := by
  sorry

end tan_value_from_ratio_l2045_204535


namespace ln_sufficient_not_necessary_for_exp_l2045_204541

theorem ln_sufficient_not_necessary_for_exp (x : ℝ) :
  (∀ x, (Real.log x > 0 → Real.exp x > 1)) ∧
  (∃ x, Real.exp x > 1 ∧ ¬(Real.log x > 0)) :=
sorry

end ln_sufficient_not_necessary_for_exp_l2045_204541


namespace max_advancing_teams_for_specific_tournament_l2045_204516

/-- Represents a football tournament with specified rules --/
structure FootballTournament where
  num_teams : Nat
  min_points_to_advance : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculates the maximum number of teams that can advance in the tournament --/
def max_advancing_teams (tournament : FootballTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of advancing teams for the specific tournament --/
theorem max_advancing_teams_for_specific_tournament :
  let tournament : FootballTournament := {
    num_teams := 7,
    min_points_to_advance := 12,
    points_for_win := 3,
    points_for_draw := 1,
    points_for_loss := 0
  }
  max_advancing_teams tournament = 5 := by sorry

end max_advancing_teams_for_specific_tournament_l2045_204516


namespace max_time_between_happy_moments_l2045_204508

/-- A happy moment on a 24-hour digital clock --/
structure HappyMoment where
  hours : Fin 24
  minutes : Fin 60
  is_happy : (hours = 6 * minutes) ∨ (minutes = 6 * hours)

/-- The time difference between two happy moments in minutes --/
def time_difference (h1 h2 : HappyMoment) : ℕ :=
  let total_minutes1 := h1.hours * 60 + h1.minutes
  let total_minutes2 := h2.hours * 60 + h2.minutes
  if total_minutes2 ≥ total_minutes1 then
    total_minutes2 - total_minutes1
  else
    (24 * 60) - (total_minutes1 - total_minutes2)

/-- Theorem stating the maximum time difference between consecutive happy moments --/
theorem max_time_between_happy_moments :
  ∃ (max : ℕ), max = 361 ∧
  ∀ (h1 h2 : HappyMoment), time_difference h1 h2 ≤ max :=
sorry

end max_time_between_happy_moments_l2045_204508


namespace interest_rate_equation_l2045_204577

/-- Proves that the interest rate R satisfies the equation for the given conditions -/
theorem interest_rate_equation (P : ℝ) (n : ℝ) (R : ℝ) : 
  P = 10000 → n = 2 → P * ((1 + R/100)^n - (1 + n*R/100)) = 36 → R = 6 := by
  sorry

end interest_rate_equation_l2045_204577


namespace percentage_lost_is_25_percent_l2045_204540

/-- Represents the number of kettles of hawks -/
def num_kettles : ℕ := 6

/-- Represents the average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- Represents the number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- Represents the expected number of babies this season -/
def expected_babies : ℕ := 270

/-- Calculates the percentage of baby hawks lost -/
def percentage_lost : ℚ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let lost_babies := total_babies - expected_babies
  (lost_babies : ℚ) / (total_babies : ℚ) * 100

/-- Theorem stating that the percentage of baby hawks lost is 25% -/
theorem percentage_lost_is_25_percent : percentage_lost = 25 := by
  sorry

end percentage_lost_is_25_percent_l2045_204540


namespace cashier_bills_problem_l2045_204511

theorem cashier_bills_problem (total_bills : ℕ) (total_value : ℕ) 
  (h_total_bills : total_bills = 126)
  (h_total_value : total_value = 840) :
  ∃ (some_dollar_bills ten_dollar_bills : ℕ),
    some_dollar_bills + ten_dollar_bills = total_bills ∧
    some_dollar_bills + 10 * ten_dollar_bills = total_value ∧
    some_dollar_bills = 47 := by
  sorry

end cashier_bills_problem_l2045_204511


namespace polygon_with_900_degree_sum_is_heptagon_l2045_204595

theorem polygon_with_900_degree_sum_is_heptagon :
  ∀ n : ℕ, n ≥ 3 → (n - 2) * 180 = 900 → n = 7 :=
by
  sorry

end polygon_with_900_degree_sum_is_heptagon_l2045_204595


namespace fractional_unit_problem_l2045_204514

def fractional_unit (n : ℕ) (d : ℕ) : ℚ := 1 / d

theorem fractional_unit_problem (n d : ℕ) (h1 : n = 5) (h2 : d = 11) :
  let u := fractional_unit n d
  (u = 1 / 11) ∧
  (n / d + 6 * u = 2) ∧
  (n / d - 5 * u = 1) :=
sorry

end fractional_unit_problem_l2045_204514


namespace ellipse_foci_distance_l2045_204517

/-- Given an ellipse with equation x²/16 + y²/9 = 1, the distance between its foci is 2√7. -/
theorem ellipse_foci_distance :
  ∀ (F₁ F₂ : ℝ × ℝ),
  (∀ (x y : ℝ), x^2/16 + y^2/9 = 1 → (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = 4 * (4 + 3)) →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 28 :=
by sorry

end ellipse_foci_distance_l2045_204517


namespace expand_and_simplify_l2045_204530

theorem expand_and_simplify (x : ℝ) : (2*x - 1)^2 - x*(4*x - 1) = -3*x + 1 := by
  sorry

end expand_and_simplify_l2045_204530


namespace height_difference_is_half_l2045_204576

/-- A circle tangent to the parabola y = x^2 + 1 at two points -/
structure TangentCircle where
  /-- x-coordinate of one tangent point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_condition : (a^2 + ((a^2 + 1) - b)^2 = r^2) ∧ 
                      ((-a)^2 + (((-a)^2 + 1) - b)^2 = r^2)
  /-- The circle's center is on the y-axis -/
  center_on_y_axis : b > 0

/-- The difference in height between the circle's center and tangent points -/
def height_difference (c : TangentCircle) : ℝ :=
  c.b - (c.a^2 + 1)

/-- Theorem: The height difference is always 1/2 -/
theorem height_difference_is_half (c : TangentCircle) : 
  height_difference c = 1/2 := by
  sorry

end height_difference_is_half_l2045_204576


namespace transformed_roots_equation_l2045_204539

/-- Given that a, b, c, and d are the solutions of x^4 + 2x^3 - 5 = 0,
    prove that abc/d, abd/c, acd/b, and bcd/a are the solutions of the same equation. -/
theorem transformed_roots_equation (a b c d : ℂ) : 
  (a^4 + 2*a^3 - 5 = 0) ∧ 
  (b^4 + 2*b^3 - 5 = 0) ∧ 
  (c^4 + 2*c^3 - 5 = 0) ∧ 
  (d^4 + 2*d^3 - 5 = 0) →
  ((a*b*c/d)^4 + 2*(a*b*c/d)^3 - 5 = 0) ∧
  ((a*b*d/c)^4 + 2*(a*b*d/c)^3 - 5 = 0) ∧
  ((a*c*d/b)^4 + 2*(a*c*d/b)^3 - 5 = 0) ∧
  ((b*c*d/a)^4 + 2*(b*c*d/a)^3 - 5 = 0) := by
  sorry


end transformed_roots_equation_l2045_204539


namespace compare_powers_l2045_204565

theorem compare_powers : 2^2023 * 7^2023 < 3^2023 * 5^2023 := by
  sorry

end compare_powers_l2045_204565


namespace pool_capacity_pool_capacity_is_2000_liters_l2045_204552

theorem pool_capacity (water_loss_per_jump : ℝ) (cleaning_threshold : ℝ) (jumps_before_cleaning : ℕ) : ℝ :=
  let total_water_loss := water_loss_per_jump * jumps_before_cleaning
  let water_loss_percentage := 1 - cleaning_threshold
  total_water_loss / water_loss_percentage

#check pool_capacity 0.4 0.8 1000 = 2000

theorem pool_capacity_is_2000_liters :
  pool_capacity 0.4 0.8 1000 = 2000 := by sorry

end pool_capacity_pool_capacity_is_2000_liters_l2045_204552


namespace min_omega_value_l2045_204587

open Real

/-- Given a function f(x) = 2sin(ωx + φ) where ω > 0, 
    if the graph is symmetrical about the line x = π/3 and f(π/12) = 0, 
    then the minimum value of ω is 2. -/
theorem min_omega_value (ω φ : ℝ) (hω : ω > 0) :
  (∀ x, 2 * sin (ω * x + φ) = 2 * sin (ω * (2 * π/3 - x) + φ)) →
  2 * sin (ω * π/12 + φ) = 0 →
  ω ≥ 2 :=
sorry

end min_omega_value_l2045_204587


namespace sum_squared_geq_three_l2045_204531

theorem sum_squared_geq_three (a b c : ℝ) (h : a * b + b * c + a * c = 1) :
  (a + b + c)^2 ≥ 3 := by
  sorry

end sum_squared_geq_three_l2045_204531


namespace angle_ABH_measure_l2045_204555

/-- A regular octagon ABCDEFGH -/
structure RegularOctagon where
  -- Define the octagon (we don't need to specify all vertices, just declare it's regular)
  vertices : Fin 8 → ℝ × ℝ
  is_regular : True  -- We assume it's regular without specifying the conditions

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- Angle ABH in the regular octagon -/
def angle_ABH (octagon : RegularOctagon) : ℝ :=
  22.5

/-- Theorem: In a regular octagon ABCDEFGH, the measure of angle ABH is 22.5° -/
theorem angle_ABH_measure (octagon : RegularOctagon) :
  angle_ABH octagon = 22.5 := by
  sorry


end angle_ABH_measure_l2045_204555


namespace solution_satisfies_system_l2045_204560

theorem solution_satisfies_system :
  let x : ℝ := -8/3
  let y : ℝ := -4/5
  (Real.sqrt (1 - 3*x) - 1 = Real.sqrt (5*y - 3*x)) ∧
  (Real.sqrt (5 - 5*y) + Real.sqrt (5*y - 3*x) = 5) := by
sorry

end solution_satisfies_system_l2045_204560


namespace sine_cosine_inequality_l2045_204567

theorem sine_cosine_inequality (x a b : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ((a + b) / 2) ^ 2 := by
  sorry

end sine_cosine_inequality_l2045_204567


namespace inequality_one_inequality_two_l2045_204533

-- Problem 1
theorem inequality_one (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := by sorry

-- Problem 2
theorem inequality_two (a b : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) :
  |1 - a * b| > |a - b| := by sorry

end inequality_one_inequality_two_l2045_204533


namespace room_width_calculation_l2045_204522

/-- Proves that given a rectangular room with a length of 5.5 meters, where the cost of paving the floor at a rate of 800 Rs/m² is 17,600 Rs, the width of the room is 4 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 17600 →
  cost_per_sqm = 800 →
  total_cost / cost_per_sqm / length = 4 := by
  sorry

end room_width_calculation_l2045_204522


namespace group_composition_l2045_204573

/-- Proves that in a group of 300 people, where the number of men is twice the number of women,
    and the number of women is 3 times the number of children, the number of children is 30. -/
theorem group_composition (total : ℕ) (children : ℕ) (women : ℕ) (men : ℕ) 
    (h1 : total = 300)
    (h2 : men = 2 * women)
    (h3 : women = 3 * children)
    (h4 : total = children + women + men) : 
  children = 30 := by
sorry

end group_composition_l2045_204573


namespace commute_days_is_22_l2045_204578

/-- Represents the commuting options for a day -/
inductive CommuteOption
  | MorningCarEveningBike
  | MorningBikeEveningCar
  | BothCar

/-- Represents the commute data over a period of days -/
structure CommuteData where
  totalDays : ℕ
  morningCar : ℕ
  eveningBike : ℕ
  totalCarCommutes : ℕ

/-- The commute data satisfies the given conditions -/
def validCommuteData (data : CommuteData) : Prop :=
  data.morningCar = 10 ∧
  data.eveningBike = 12 ∧
  data.totalCarCommutes = 14

theorem commute_days_is_22 (data : CommuteData) (h : validCommuteData data) :
  data.totalDays = 22 := by
  sorry

#check commute_days_is_22

end commute_days_is_22_l2045_204578


namespace square_root_sum_fractions_l2045_204503

theorem square_root_sum_fractions : 
  Real.sqrt (1/25 + 1/36 + 1/49) = Real.sqrt 7778 / 297 := by
  sorry

end square_root_sum_fractions_l2045_204503


namespace specific_seating_arrangements_l2045_204521

/-- Represents the seating arrangement in a theater -/
structure TheaterSeating where
  front_row : ℕ
  back_row : ℕ
  unusable_middle_seats : ℕ

/-- Calculates the number of ways to seat two people in the theater -/
def seating_arrangements (theater : TheaterSeating) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements for the given problem -/
theorem specific_seating_arrangements :
  let theater : TheaterSeating := {
    front_row := 10,
    back_row := 11,
    unusable_middle_seats := 3
  }
  seating_arrangements theater = 276 := by
  sorry

end specific_seating_arrangements_l2045_204521


namespace marty_painting_combinations_l2045_204586

/-- The number of available colors -/
def num_colors : ℕ := 5

/-- The number of available painting methods -/
def num_methods : ℕ := 4

/-- The number of restricted combinations (white paint with spray) -/
def num_restricted : ℕ := 1

theorem marty_painting_combinations :
  (num_colors - 1) * num_methods + (num_methods - 1) = 19 := by
  sorry

end marty_painting_combinations_l2045_204586


namespace f_inequality_l2045_204583

-- Define f as a differentiable function on ℝ
variable (f : ℝ → ℝ)

-- State the condition that f'(x) - f(x) < 0 for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv f) x - f x < 0)

-- Define e as the mathematical constant e
noncomputable def e : ℝ := Real.exp 1

-- State the theorem to be proved
theorem f_inequality : e * f 2015 > f 2016 :=
sorry

end f_inequality_l2045_204583


namespace success_permutations_l2045_204594

/-- The number of letters in the word "SUCCESS" -/
def total_letters : ℕ := 7

/-- The number of times 'S' appears in "SUCCESS" -/
def s_count : ℕ := 3

/-- The number of times 'C' appears in "SUCCESS" -/
def c_count : ℕ := 2

/-- The number of times 'U' appears in "SUCCESS" -/
def u_count : ℕ := 1

/-- The number of times 'E' appears in "SUCCESS" -/
def e_count : ℕ := 1

/-- The number of unique arrangements of the letters in "SUCCESS" -/
def success_arrangements : ℕ := 420

theorem success_permutations :
  Nat.factorial total_letters / (Nat.factorial s_count * Nat.factorial c_count * Nat.factorial u_count * Nat.factorial e_count) = success_arrangements :=
by sorry

end success_permutations_l2045_204594


namespace a_5_value_l2045_204588

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = -9 →
  a 7 = -1 →
  a 5 = -3 := by
sorry

end a_5_value_l2045_204588


namespace prism_volume_l2045_204510

/-- The volume of a right rectangular prism with face areas 15, 10, and 30 -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 30) :
  l * w * h = 30 * Real.sqrt 5 := by
  sorry

end prism_volume_l2045_204510


namespace quadratic_real_roots_l2045_204572

theorem quadratic_real_roots (a : ℝ) : 
  a > 1 → ∃ x : ℝ, x^2 - (2*a + 1)*x + a^2 = 0 :=
by
  sorry

#check quadratic_real_roots

end quadratic_real_roots_l2045_204572


namespace pet_store_puppies_sold_l2045_204554

/-- Proves that the number of puppies sold is 1, given the conditions of the pet store problem. -/
theorem pet_store_puppies_sold :
  let kittens_sold : ℕ := 2
  let kitten_price : ℕ := 6
  let puppy_price : ℕ := 5
  let total_earnings : ℕ := 17
  let puppies_sold : ℕ := (total_earnings - kittens_sold * kitten_price) / puppy_price
  puppies_sold = 1 := by
  sorry

end pet_store_puppies_sold_l2045_204554


namespace point_coordinates_l2045_204538

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- Predicate for a point being in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y > 0

/-- Theorem: If a point M is in the first quadrant, its distance to the x-axis is 3,
    and its distance to the y-axis is 2, then its coordinates are (2, 3) -/
theorem point_coordinates (M : Point) 
  (h1 : inFirstQuadrant M) 
  (h2 : distToXAxis M = 3) 
  (h3 : distToYAxis M = 2) : 
  M.x = 2 ∧ M.y = 3 := by
  sorry

end point_coordinates_l2045_204538


namespace infinite_solutions_iff_c_eq_five_l2045_204519

/-- The equation has infinitely many solutions if and only if c = 5 -/
theorem infinite_solutions_iff_c_eq_five :
  (∃ c : ℝ, ∀ x : ℝ, 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
sorry

end infinite_solutions_iff_c_eq_five_l2045_204519


namespace expansion_without_x3_x2_implies_m_plus_n_eq_neg_4_l2045_204596

theorem expansion_without_x3_x2_implies_m_plus_n_eq_neg_4 
  (m n : ℝ) 
  (h1 : (1 + m) = 0)
  (h2 : (-3*m + n) = 0) :
  m + n = -4 := by
  sorry

end expansion_without_x3_x2_implies_m_plus_n_eq_neg_4_l2045_204596


namespace line_direction_vector_l2045_204532

theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) :
  p1 = (4, -3) →
  p2 = (-1, 6) →
  ∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1)) →
  b = 5 / 9 := by
sorry

end line_direction_vector_l2045_204532


namespace ratio_m_n_l2045_204526

theorem ratio_m_n (m n : ℕ) (h1 : m > n) (h2 : ¬(n ∣ m)) 
  (h3 : m % n = (m + n) % (m - n)) : m / n = 5 / 2 := by
  sorry

end ratio_m_n_l2045_204526


namespace susan_drinks_eight_l2045_204548

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℚ := 2

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℚ := 2 * paul_bottles + 3

/-- The number of juice bottles Susan drinks per day -/
def susan_bottles : ℚ := 1.5 * donald_bottles - 2.5

/-- Theorem stating that Susan drinks 8 bottles of juice per day -/
theorem susan_drinks_eight : susan_bottles = 8 := by
  sorry

end susan_drinks_eight_l2045_204548


namespace wrong_number_difference_l2045_204589

/-- The number of elements in the set of numbers --/
def n : ℕ := 10

/-- The original average of the numbers --/
def original_average : ℚ := 402/10

/-- The correct average of the numbers --/
def correct_average : ℚ := 403/10

/-- The second wrongly copied number --/
def wrong_second : ℕ := 13

/-- The correct second number --/
def correct_second : ℕ := 31

/-- Theorem stating the difference between the wrongly copied number and the actual number --/
theorem wrong_number_difference (first_wrong : ℚ) (first_actual : ℚ) 
  (h1 : first_wrong > first_actual)
  (h2 : n * original_average = (n - 2) * correct_average + first_wrong + wrong_second)
  (h3 : n * correct_average = (n - 2) * correct_average + first_actual + correct_second) :
  first_wrong - first_actual = 19 := by
  sorry

end wrong_number_difference_l2045_204589


namespace square_difference_divided_by_nine_l2045_204545

theorem square_difference_divided_by_nine : (108^2 - 99^2) / 9 = 207 := by
  sorry

end square_difference_divided_by_nine_l2045_204545


namespace student_number_factor_l2045_204580

theorem student_number_factor (x f : ℝ) : 
  x = 110 → x * f - 220 = 110 → f = 3 := by
  sorry

end student_number_factor_l2045_204580


namespace irrational_product_l2045_204584

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Define the property of being rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem irrational_product (x : ℝ) : 
  IsIrrational x → 
  IsRational ((x - 2) * (x + 6)) → 
  IsIrrational ((x + 2) * (x - 6)) := by
  sorry

end irrational_product_l2045_204584


namespace p_necessary_not_sufficient_for_q_l2045_204557

theorem p_necessary_not_sufficient_for_q :
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) ∧
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) :=
by sorry

end p_necessary_not_sufficient_for_q_l2045_204557


namespace solution_characterization_l2045_204528

/-- A function satisfying the given differential equation for all real x and positive integers n -/
def SatisfiesDiffEq (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (n : ℕ), n > 0 → Differentiable ℝ f ∧ 
    deriv f x = (f (x + n) - f x) / n

/-- The main theorem stating that any function satisfying the differential equation
    is of the form f(x) = ax + b for some real constants a and b -/
theorem solution_characterization (f : ℝ → ℝ) :
  SatisfiesDiffEq f → ∃ (a b : ℝ), ∀ x, f x = a * x + b :=
sorry

end solution_characterization_l2045_204528


namespace product_equals_zero_l2045_204582

theorem product_equals_zero (b : ℤ) (h : b = 3) :
  (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * 
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end product_equals_zero_l2045_204582


namespace gcd_1337_382_l2045_204558

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end gcd_1337_382_l2045_204558


namespace expand_and_simplify_l2045_204570

theorem expand_and_simplify (x : ℝ) : (x + 3) * (x - 4) = x^2 - x - 12 := by
  sorry

end expand_and_simplify_l2045_204570


namespace range_of_m_l2045_204534

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the property that ¬p is sufficient but not necessary for ¬q
def neg_p_sufficient_not_necessary_for_neg_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ∃ x, ¬(q x m) ∧ p x

-- Theorem statement
theorem range_of_m :
  ∀ m, neg_p_sufficient_not_necessary_for_neg_q m ↔ -3 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l2045_204534


namespace P_plus_8_divisible_P_minus_8_divisible_P_unique_l2045_204507

/-- A fifth-degree polynomial P(x) that satisfies specific divisibility conditions -/
def P (x : ℝ) : ℝ := 3*x^5 - 10*x^3 + 15*x

/-- P(x) + 8 is divisible by (x+1)^3 -/
theorem P_plus_8_divisible (x : ℝ) : ∃ (q : ℝ → ℝ), P x + 8 = (x + 1)^3 * q x := by sorry

/-- P(x) - 8 is divisible by (x-1)^3 -/
theorem P_minus_8_divisible (x : ℝ) : ∃ (r : ℝ → ℝ), P x - 8 = (x - 1)^3 * r x := by sorry

/-- P(x) is the unique fifth-degree polynomial satisfying both divisibility conditions -/
theorem P_unique : ∀ (Q : ℝ → ℝ), 
  (∃ (q r : ℝ → ℝ), (∀ x, Q x + 8 = (x + 1)^3 * q x) ∧ (∀ x, Q x - 8 = (x - 1)^3 * r x)) →
  (∃ (a b c d e f : ℝ), ∀ x, Q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∀ x, Q x = P x) := by sorry

end P_plus_8_divisible_P_minus_8_divisible_P_unique_l2045_204507


namespace seashell_difference_l2045_204561

theorem seashell_difference (craig_shells : ℕ) (craig_ratio : ℕ) (brian_ratio : ℕ) : 
  craig_shells = 54 → 
  craig_ratio = 9 → 
  brian_ratio = 7 → 
  craig_shells - (craig_shells / craig_ratio * brian_ratio) = 12 := by
sorry

end seashell_difference_l2045_204561


namespace coverable_polyhedron_exists_l2045_204550

/-- A polyhedron that can be covered by a square and an equilateral triangle -/
structure CoverablePolyhedron where
  /-- Side length of the square -/
  s : ℝ
  /-- Side length of the equilateral triangle -/
  t : ℝ
  /-- The perimeters of the square and triangle are equal -/
  h_perimeter : 4 * s = 3 * t
  /-- The polyhedron exists and can be covered -/
  h_exists : Prop

/-- Theorem stating that there exists a polyhedron that can be covered by a square and an equilateral triangle with equal perimeters -/
theorem coverable_polyhedron_exists : ∃ (p : CoverablePolyhedron), p.h_exists := by
  sorry

end coverable_polyhedron_exists_l2045_204550


namespace perfect_squares_is_good_l2045_204502

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def perfect_squares : Set ℕ := {n : ℕ | is_perfect_square n}

def is_good (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → 
    ∀ p q : ℕ, Prime p → Prime q → p ≠ q → p ∣ n → q ∣ n →
      ¬(n - p ∈ A ∧ n - q ∈ A)

theorem perfect_squares_is_good : is_good perfect_squares :=
sorry

end perfect_squares_is_good_l2045_204502


namespace existence_of_abcd_l2045_204512

theorem existence_of_abcd (n : ℕ) (h : n > 1) : 
  ∃ (a b c d : ℕ), (a + b = c + d) ∧ (a * b - c * d = 4 * n) := by
  sorry

end existence_of_abcd_l2045_204512


namespace circle_area_from_circumference_l2045_204537

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 2 * π * r = 24 * π → π * r^2 = 144 * π := by
  sorry

end circle_area_from_circumference_l2045_204537


namespace snakes_not_hiding_l2045_204592

/-- Given a cage with snakes, some of which are hiding, calculate the number of snakes not hiding. -/
theorem snakes_not_hiding (total_snakes hiding_snakes : ℕ) 
  (h1 : total_snakes = 95)
  (h2 : hiding_snakes = 64) :
  total_snakes - hiding_snakes = 31 := by
  sorry

end snakes_not_hiding_l2045_204592


namespace equal_area_intersection_sum_l2045_204598

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℚ :=
  (1/2) * abs (a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y
             - (b.x * a.y + c.x * b.y + d.x * c.y + a.x * d.y))

/-- Checks if a fraction is in its lowest terms -/
def isLowestTerms (p q : ℤ) : Prop :=
  ∀ (d : ℤ), d > 1 → ¬(d ∣ p ∧ d ∣ q)

/-- Main theorem -/
theorem equal_area_intersection_sum (p q r s : ℤ) :
  let a := Point.mk 0 0
  let b := Point.mk 1 3
  let c := Point.mk 4 4
  let d := Point.mk 5 0
  let intersectionPoint := Point.mk (p/q) (r/s)
  quadrilateralArea a b intersectionPoint d = quadrilateralArea b c d intersectionPoint →
  isLowestTerms p q →
  isLowestTerms r s →
  p + q + r + s = 200 := by
  sorry

end equal_area_intersection_sum_l2045_204598


namespace sequence_formulas_l2045_204547

/-- Given a geometric sequence {a_n} and another sequence {b_n}, prove the formulas for a_n and b_n -/
theorem sequence_formulas (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 3 * a n) →  -- geometric sequence condition
  (a 1 = 4) →  -- initial condition for a_n
  (2 * a 2 + a 3 = 60) →  -- additional condition for a_n
  (∀ n, b (n + 1) = b n + a n) →  -- recurrence relation for b_n
  (b 1 = a 2) →  -- initial condition for b_n
  (b 1 > 0) →  -- positivity condition for b_1
  (∀ n, a n = 4 * 3^(n - 1)) ∧ 
  (∀ n, b n = 2 * 3^n + 10) := by
sorry


end sequence_formulas_l2045_204547


namespace percentage_problem_l2045_204568

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 400) : 1.2 * x = 2400 := by
  sorry

end percentage_problem_l2045_204568


namespace remainder_19_pow_60_mod_7_l2045_204525

theorem remainder_19_pow_60_mod_7 : 19^60 % 7 = 1 := by
  sorry

end remainder_19_pow_60_mod_7_l2045_204525


namespace intersection_and_union_of_sets_l2045_204551

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_and_union_of_sets :
  ∃ (a : ℝ), (A a ∩ B a = {9}) ∧ (a = -3) ∧ (A a ∪ B a = {-8, -7, -4, 4, 9}) := by
  sorry

end intersection_and_union_of_sets_l2045_204551


namespace negation_of_existence_log_negation_equivalence_l2045_204513

theorem negation_of_existence (p : Real → Prop) :
  (¬∃ x, x > 1 ∧ p x) ↔ (∀ x, x > 1 → ¬p x) := by sorry

theorem log_negation_equivalence :
  (¬∃ x₀, x₀ > 1 ∧ Real.log x₀ > 1) ↔ (∀ x, x > 1 → Real.log x ≤ 1) := by sorry

end negation_of_existence_log_negation_equivalence_l2045_204513


namespace right_triangle_leg_identity_l2045_204546

theorem right_triangle_leg_identity (a b : ℝ) : 2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := by
  sorry

end right_triangle_leg_identity_l2045_204546


namespace janessa_keeps_twenty_cards_l2045_204518

/-- The number of cards Janessa keeps for herself given the initial conditions -/
def janessas_kept_cards (initial_cards : ℕ) (father_cards : ℕ) (ebay_cards : ℕ) (bad_cards : ℕ) (given_cards : ℕ) : ℕ :=
  initial_cards + father_cards + ebay_cards - bad_cards - given_cards

/-- Theorem stating that Janessa keeps 20 cards for herself -/
theorem janessa_keeps_twenty_cards :
  janessas_kept_cards 4 13 36 4 29 = 20 := by sorry

end janessa_keeps_twenty_cards_l2045_204518


namespace four_Z_three_l2045_204559

-- Define the Z operation
def Z (x y : ℤ) : ℤ := x^2 - 3*x*y + y^2

-- Theorem to prove
theorem four_Z_three : Z 4 3 = -11 := by
  sorry

end four_Z_three_l2045_204559


namespace texas_integrated_school_student_count_l2045_204597

theorem texas_integrated_school_student_count 
  (initial_classes : ℕ) 
  (students_per_class : ℕ) 
  (additional_classes : ℕ) : 
  initial_classes = 15 → 
  students_per_class = 20 → 
  additional_classes = 5 → 
  (initial_classes + additional_classes) * students_per_class = 400 := by
sorry

end texas_integrated_school_student_count_l2045_204597


namespace average_odd_one_digit_l2045_204575

def is_odd_one_digit (n : ℕ) : Prop := n % 2 = 1 ∧ n ≥ 1 ∧ n ≤ 9

def odd_one_digit_numbers : List ℕ := [1, 3, 5, 7, 9]

theorem average_odd_one_digit : 
  (List.sum odd_one_digit_numbers) / (List.length odd_one_digit_numbers) = 5 := by
  sorry

end average_odd_one_digit_l2045_204575


namespace max_value_chord_intersection_l2045_204566

theorem max_value_chord_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (2*a*x + b*y = 2) ∧
   ∃ (x1 y1 x2 y2 : ℝ), x1^2 + y1^2 = 4 ∧ x2^2 + y2^2 = 4 ∧
   2*a*x1 + b*y1 = 2 ∧ 2*a*x2 + b*y2 = 2 ∧
   (x1 - x2)^2 + (y1 - y2)^2 = 12) →
  (∀ c : ℝ, c ≤ (9 * Real.sqrt 2) / 8 ∨ ∃ d : ℝ, d > c ∧ d = a * Real.sqrt (1 + 2*b^2)) :=
by sorry

end max_value_chord_intersection_l2045_204566


namespace floor_length_is_20_l2045_204536

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintingCost : ℝ
  paintingRate : ℝ

/-- Theorem stating the length of the floor under given conditions. -/
theorem floor_length_is_20 (floor : RectangularFloor)
  (h1 : floor.length = 3 * floor.breadth)
  (h2 : floor.paintingCost = 400)
  (h3 : floor.paintingRate = 3)
  (h4 : floor.paintingCost / floor.paintingRate = floor.length * floor.breadth) :
  floor.length = 20 := by
  sorry


end floor_length_is_20_l2045_204536
