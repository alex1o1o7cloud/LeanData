import Mathlib

namespace distance_between_points_l2419_241940

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 5 := by
  sorry

end distance_between_points_l2419_241940


namespace sample_definition_l2419_241963

/-- Represents a student's math score -/
def MathScore : Type := ℝ

/-- Represents a sample of math scores -/
def Sample : Type := List MathScore

structure SurveyData where
  totalStudents : ℕ
  sampleSize : ℕ
  scores : Sample
  h_sampleSize : sampleSize ≤ totalStudents

/-- Definition of a valid sample for the survey -/
def isValidSample (data : SurveyData) : Prop :=
  data.scores.length = data.sampleSize

theorem sample_definition (data : SurveyData) 
  (h_total : data.totalStudents = 960)
  (h_sample : data.sampleSize = 120)
  (h_valid : isValidSample data) :
  ∃ (sample : Sample), sample = data.scores ∧ sample.length = 120 :=
sorry

end sample_definition_l2419_241963


namespace chicken_theorem_l2419_241998

/-- The number of chickens Colten has -/
def colten_chickens : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar_chickens : ℕ := 3 * colten_chickens - 4

/-- The number of chickens Quentin has -/
def quentin_chickens : ℕ := 2 * skylar_chickens + 25

theorem chicken_theorem : 
  colten_chickens + skylar_chickens + quentin_chickens = 383 :=
by sorry

end chicken_theorem_l2419_241998


namespace claire_photos_l2419_241918

theorem claire_photos (c : ℕ) 
  (h1 : 3 * c = c + 10) : c = 5 := by
  sorry

#check claire_photos

end claire_photos_l2419_241918


namespace choose_three_from_nine_l2419_241988

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end choose_three_from_nine_l2419_241988


namespace three_digit_product_sum_l2419_241955

theorem three_digit_product_sum (P A U : ℕ) : 
  P ≠ A → P ≠ U → A ≠ U →
  P ≥ 1 → P ≤ 9 →
  A ≥ 0 → A ≤ 9 →
  U ≥ 0 → U ≤ 9 →
  100 * P + 10 * A + U ≥ 100 →
  100 * P + 10 * A + U ≤ 999 →
  (P + A + U) * P * A * U = 300 →
  ∃ (PAU : ℕ), PAU = 100 * P + 10 * A + U ∧ 
               (PAU.div 100 + (PAU.mod 100).div 10 + PAU.mod 10) * 
               PAU.div 100 * (PAU.mod 100).div 10 * PAU.mod 10 = 300 :=
by sorry

end three_digit_product_sum_l2419_241955


namespace lcm_of_ratio_and_hcf_l2419_241987

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → Nat.gcd a b = 4 → Nat.lcm a b = 80 := by
  sorry

end lcm_of_ratio_and_hcf_l2419_241987


namespace train_passing_jogger_time_l2419_241976

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 * (1000 / 3600))  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * (1000 / 3600))  -- 45 km/hr in m/s
  (h3 : train_length = 120)                -- 120 m
  (h4 : initial_distance = 180)            -- 180 m
  : (initial_distance + train_length) / (train_speed - jogger_speed) = 30 := by
  sorry


end train_passing_jogger_time_l2419_241976


namespace min_area_over_sqrt_t_l2419_241935

/-- The area bounded by the tangent lines and the parabola -/
noncomputable def S (t : ℝ) : ℝ := (2 / 3) * (1 + t^2)^(3/2)

/-- The main theorem statement -/
theorem min_area_over_sqrt_t (t : ℝ) (ht : t > 0) :
  ∃ (min_value : ℝ), min_value = (2 * 6^(3/2)) / (3 * 5^(5/4)) ∧
  ∀ (t : ℝ), t > 0 → S t / Real.sqrt t ≥ min_value :=
sorry

end min_area_over_sqrt_t_l2419_241935


namespace equation_solution_l2419_241931

theorem equation_solution : ∃ x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end equation_solution_l2419_241931


namespace investment_calculation_l2419_241905

/-- Calculates the total investment given share details and dividend income -/
def calculate_investment (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * face_value
  let number_of_shares := annual_income / dividend_per_share
  number_of_shares * quoted_price

/-- Theorem stating that the investment is 4940 given the problem conditions -/
theorem investment_calculation :
  calculate_investment 10 9.5 14 728 = 4940 := by
  sorry

#eval calculate_investment 10 9.5 14 728

end investment_calculation_l2419_241905


namespace platform_length_platform_length_is_340_l2419_241954

/-- The length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- The platform length is 340 meters -/
theorem platform_length_is_340 :
  platform_length 360 45 56 = 340 := by
  sorry

end platform_length_platform_length_is_340_l2419_241954


namespace direct_proportion_properties_l2419_241980

/-- A function representing direct proportion --/
def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Theorem stating the properties of the function f --/
theorem direct_proportion_properties :
  ∀ k : ℝ, k ≠ 0 →
  (f k 2 = 4) →
  ∃ a : ℝ, (f k a = 3 ∧ k = 2 ∧ a = 3/2) := by
  sorry

#check direct_proportion_properties

end direct_proportion_properties_l2419_241980


namespace facebook_group_removal_l2419_241946

/-- Proves the number of removed members from a Facebook group --/
theorem facebook_group_removal (initial_members : ℕ) (messages_per_day : ℕ) (total_messages_week : ℕ) : 
  initial_members = 150 →
  messages_per_day = 50 →
  total_messages_week = 45500 →
  (initial_members - (initial_members - 20)) * messages_per_day * 7 = total_messages_week :=
by
  sorry

end facebook_group_removal_l2419_241946


namespace parallelogram_altitude_base_ratio_l2419_241924

/-- 
Given a parallelogram with area 98 square meters and base length 7 meters,
prove that the ratio of its altitude to its base is 2.
-/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
  area = 98 →
  base = 7 →
  area = base * altitude →
  altitude / base = 2 := by
sorry

end parallelogram_altitude_base_ratio_l2419_241924


namespace range_of_m_l2419_241991

theorem range_of_m (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b - a * b = 0)
  (h_log : ∀ a b, 0 < a → 0 < b → a + b - a * b = 0 → Real.log (m ^ 2 / (a + b)) ≤ 0) :
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end range_of_m_l2419_241991


namespace salt_solution_concentration_l2419_241985

/-- Proves that the concentration of the salt solution is 50% given the specified conditions. -/
theorem salt_solution_concentration
  (water_volume : Real)
  (salt_solution_volume : Real)
  (total_volume : Real)
  (mixture_concentration : Real)
  (h1 : water_volume = 1)
  (h2 : salt_solution_volume = 0.25)
  (h3 : total_volume = water_volume + salt_solution_volume)
  (h4 : mixture_concentration = 0.1)
  (h5 : salt_solution_volume * (concentration / 100) = total_volume * mixture_concentration) :
  concentration = 50 := by
  sorry

end salt_solution_concentration_l2419_241985


namespace correct_conclusions_l2419_241911

-- Define the vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the parallel relation
def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

-- Define the dot product
variable (dot : V → V → ℝ)

-- Statement of the theorem
theorem correct_conclusions :
  (∀ (a b c : V), a = b → b = c → a = c) ∧ 
  (∃ (a b c : V), parallel a b → parallel b c → ¬ parallel a c) ∧
  (∃ (a b : V), |dot a b| ≠ |dot a (1 • b)|) ∧
  (∀ (a b c : V), b = c → dot a b = dot a c) :=
sorry

end correct_conclusions_l2419_241911


namespace laptop_discount_l2419_241947

theorem laptop_discount (initial_discount additional_discount : ℝ) 
  (h1 : initial_discount = 0.3)
  (h2 : additional_discount = 0.5) : 
  1 - (1 - initial_discount) * (1 - additional_discount) = 0.65 := by
  sorry

end laptop_discount_l2419_241947


namespace smallest_n_with_equal_digits_sum_l2419_241949

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Check if all digits of a number are equal -/
def all_digits_equal (n : ℕ) : Prop := 
  ∃ (d : ℕ) (k : ℕ), d ∈ Finset.range 10 ∧ n = d * (10^k - 1) / 9

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_n_with_equal_digits_sum : 
  ∃ (N : ℕ), 
    (∀ m : ℕ, m < N → ¬(all_digits_equal (m * sum_first_n 9))) ∧
    (all_digits_equal (N * sum_first_n 9)) ∧
    (digit_sum N = 37) := by sorry

end smallest_n_with_equal_digits_sum_l2419_241949


namespace chess_tournament_schedule_ways_l2419_241962

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  /-- Number of players per school -/
  players_per_school : Nat
  /-- Number of games each player plays against each opponent -/
  games_per_opponent : Nat
  /-- Number of rounds in the tournament -/
  num_rounds : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat

/-- Calculate the number of ways to schedule a chess tournament -/
def scheduleWays (t : ChessTournament) : Nat :=
  (t.num_rounds.factorial) + (t.num_rounds.factorial / (2^(t.num_rounds / 2)))

/-- Theorem stating the number of ways to schedule the specific chess tournament -/
theorem chess_tournament_schedule_ways :
  let t : ChessTournament := {
    players_per_school := 4,
    games_per_opponent := 2,
    num_rounds := 8,
    games_per_round := 4
  }
  scheduleWays t = 42840 := by
  sorry

end chess_tournament_schedule_ways_l2419_241962


namespace fraction_simplification_l2419_241945

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) : a / (a - 1) + 1 / (1 - a) = 1 := by
  sorry

end fraction_simplification_l2419_241945


namespace special_polyhedron_value_l2419_241979

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  T : ℕ
  P : ℕ
  is_convex : Prop
  face_count : faces = 32
  face_types : faces = triangles + pentagons
  vertex_config : Prop
  euler_formula : vertices - edges + faces = 2

/-- Theorem stating the specific value for the polyhedron -/
theorem special_polyhedron_value (poly : SpecialPolyhedron) :
  100 * poly.P + 10 * poly.T + poly.vertices = 250 := by
  sorry

end special_polyhedron_value_l2419_241979


namespace negation_of_existential_proposition_l2419_241942

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end negation_of_existential_proposition_l2419_241942


namespace johns_and_brothers_age_sum_l2419_241975

/-- Given that John's age is four less than six times his brother's age,
    and his brother is 8 years old, prove that the sum of their ages is 52. -/
theorem johns_and_brothers_age_sum :
  ∀ (john_age brother_age : ℕ),
    brother_age = 8 →
    john_age = 6 * brother_age - 4 →
    john_age + brother_age = 52 :=
by
  sorry

end johns_and_brothers_age_sum_l2419_241975


namespace ab_value_l2419_241968

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end ab_value_l2419_241968


namespace function_characterization_l2419_241973

theorem function_characterization (f : ℝ → ℤ) 
  (h1 : ∀ x y : ℝ, f (x + y) < f x + f y)
  (h2 : ∀ x : ℝ, f (f x) = ⌊x⌋ + 2) :
  ∀ x : ℤ, f x = x + 1 := by sorry

end function_characterization_l2419_241973


namespace calculate_principal_l2419_241916

/-- Given simple interest, rate, and time, calculate the principal amount --/
theorem calculate_principal
  (simple_interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : simple_interest = 4016.25)
  (h2 : rate = 3)
  (h3 : time = 5)
  : (simple_interest * 100) / (rate * time) = 26775 := by
  sorry

end calculate_principal_l2419_241916


namespace gcd_of_polynomials_l2419_241948

theorem gcd_of_polynomials (x : ℤ) (h : ∃ k : ℤ, x = 2 * k * 2027) :
  Int.gcd (3 * x^2 + 47 * x + 101) (x + 23) = 1 := by
  sorry

end gcd_of_polynomials_l2419_241948


namespace simplify_expression_l2419_241967

theorem simplify_expression :
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := by
  sorry

end simplify_expression_l2419_241967


namespace sum_of_roots_l2419_241903

/-- The function f(x) = x^3 + 3x^2 + 6x + 14 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

/-- Theorem: If f(a) = 1 and f(b) = 19, then a + b = -2 -/
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end sum_of_roots_l2419_241903


namespace tournament_cycle_l2419_241908

def TournamentGraph := Fin 12 → Fin 12 → Bool

theorem tournament_cycle (g : TournamentGraph) : 
  (∀ i j : Fin 12, i ≠ j → (g i j ≠ g j i) ∧ (g i j ∨ g j i)) →
  (∀ i : Fin 12, ∃ j : Fin 12, g i j) →
  ∃ a b c : Fin 12, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ g a b ∧ g b c ∧ g c a :=
by sorry

end tournament_cycle_l2419_241908


namespace work_completion_proof_l2419_241907

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 10

/-- The number of days it takes with fewer workers -/
def fewer_workers_days : ℕ := 20

/-- The reduction in the number of workers -/
def worker_reduction : ℕ := 10

/-- The original number of workers -/
def original_workers : ℕ := 20

theorem work_completion_proof :
  (original_workers * original_days = (original_workers - worker_reduction) * fewer_workers_days) ∧
  (original_workers > worker_reduction) :=
sorry

end work_completion_proof_l2419_241907


namespace sqrt_8_times_sqrt_50_l2419_241943

theorem sqrt_8_times_sqrt_50 : Real.sqrt 8 * Real.sqrt 50 = 20 := by sorry

end sqrt_8_times_sqrt_50_l2419_241943


namespace alyssa_bought_224_cards_l2419_241951

/-- The number of Pokemon cards Jason initially had -/
def initial_cards : ℕ := 676

/-- The number of Pokemon cards Jason has after Alyssa bought some -/
def remaining_cards : ℕ := 452

/-- The number of Pokemon cards Alyssa bought -/
def cards_bought : ℕ := initial_cards - remaining_cards

theorem alyssa_bought_224_cards : cards_bought = 224 := by
  sorry

end alyssa_bought_224_cards_l2419_241951


namespace orange_calorie_distribution_l2419_241972

theorem orange_calorie_distribution :
  ∀ (num_oranges : ℕ) 
    (pieces_per_orange : ℕ) 
    (num_people : ℕ) 
    (calories_per_orange : ℕ),
  num_oranges = 5 →
  pieces_per_orange = 8 →
  num_people = 4 →
  calories_per_orange = 80 →
  (num_oranges * pieces_per_orange / num_people) * (calories_per_orange / pieces_per_orange) = 100 :=
by
  sorry

end orange_calorie_distribution_l2419_241972


namespace sum_of_five_consecutive_odd_integers_l2419_241938

theorem sum_of_five_consecutive_odd_integers (n : ℤ) : 
  (n + (n + 8) = 156) → (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390) :=
by sorry

end sum_of_five_consecutive_odd_integers_l2419_241938


namespace equation_with_prime_solutions_l2419_241933

theorem equation_with_prime_solutions (m : ℕ) : 
  (∃ x y : ℕ, Prime x ∧ Prime y ∧ x ≠ y ∧ x^2 - 1999*x + m = 0 ∧ y^2 - 1999*y + m = 0) → 
  m = 3994 := by
sorry

end equation_with_prime_solutions_l2419_241933


namespace profit_equals_cost_of_three_toys_l2419_241986

/-- Proves that the number of toys whose cost price equals the profit is 3 -/
theorem profit_equals_cost_of_three_toys 
  (total_toys : ℕ) 
  (selling_price : ℕ) 
  (cost_per_toy : ℕ) 
  (h1 : total_toys = 18)
  (h2 : selling_price = 25200)
  (h3 : cost_per_toy = 1200) :
  (selling_price - total_toys * cost_per_toy) / cost_per_toy = 3 := by
  sorry

end profit_equals_cost_of_three_toys_l2419_241986


namespace inequality_proof_l2419_241917

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end inequality_proof_l2419_241917


namespace tangent_problem_l2419_241969

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (1 + Real.tan α) / (1 - Real.tan α) = 3/22 := by
sorry

end tangent_problem_l2419_241969


namespace sweeties_leftover_l2419_241953

theorem sweeties_leftover (m : ℕ) (h : m % 12 = 11) : (4 * m) % 12 = 8 := by
  sorry

end sweeties_leftover_l2419_241953


namespace circle_symmetry_l2419_241958

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 1 = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

/-- Symmetrical circle -/
def symmetrical_circle (x y : ℝ) : Prop :=
  (x + 7/5)^2 + (y - 6/5)^2 = 2

/-- Theorem stating that the symmetrical circle is indeed symmetrical to the given circle
    with respect to the line of symmetry -/
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    given_circle x₁ y₁ →
    symmetrical_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end circle_symmetry_l2419_241958


namespace log_identity_l2419_241970

theorem log_identity (x y : ℝ) 
  (hx : Real.log 5 / Real.log 4 = x)
  (hy : Real.log 7 / Real.log 5 = y) : 
  Real.log 7 / Real.log 10 = (2 * x * y) / (2 * x + 1) := by
  sorry

end log_identity_l2419_241970


namespace cube_roots_of_unity_sum_l2419_241936

theorem cube_roots_of_unity_sum (ω ω_bar : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω_bar = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω_bar^3 = 1 →
  ω^9 + ω_bar^9 = 2 := by sorry

end cube_roots_of_unity_sum_l2419_241936


namespace min_value_squared_differences_l2419_241906

theorem min_value_squared_differences (a b c d : ℝ) 
  (h1 : a * b = 3) 
  (h2 : c + 3 * d = 0) : 
  (a - c)^2 + (b - d)^2 ≥ 18/5 := by
  sorry

end min_value_squared_differences_l2419_241906


namespace evaluate_expression_max_value_function_max_value_function_achievable_l2419_241982

-- Part 1
theorem evaluate_expression : 
  Real.sqrt 3 * Real.cos (π / 12) - Real.sin (π / 12) = Real.sqrt 2 := by sorry

-- Part 2
theorem max_value_function : 
  ∀ θ : ℝ, Real.sqrt 3 * Real.cos θ - Real.sin θ ≤ 2 := by sorry

theorem max_value_function_achievable : 
  ∃ θ : ℝ, Real.sqrt 3 * Real.cos θ - Real.sin θ = 2 := by sorry

end evaluate_expression_max_value_function_max_value_function_achievable_l2419_241982


namespace angle_measure_in_triangle_l2419_241959

theorem angle_measure_in_triangle (D E F : ℝ) : 
  D = 90 →  -- Angle D is 90 degrees
  E = 4 * F - 10 →  -- Angle E is 10 degrees less than four times angle F
  D + E + F = 180 →  -- Sum of angles in a triangle is 180 degrees
  F = 20 :=  -- Measure of angle F is 20 degrees
by sorry

end angle_measure_in_triangle_l2419_241959


namespace range_of_p_l2419_241964

-- Define set A
def A (p : ℝ) : Set ℝ := {x : ℝ | x^2 + (p+2)*x + 1 = 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem range_of_p (p : ℝ) : (A p ∩ B = ∅) → p > -4 := by
  sorry

end range_of_p_l2419_241964


namespace balloon_height_theorem_l2419_241961

/-- Calculates the maximum height a balloon can fly given the budget and costs --/
def maxBalloonHeight (budget initialCost heliumPrice1 heliumPrice2 heliumPrice3 : ℚ) 
  (threshold1 threshold2 : ℚ) (heightPerOunce : ℚ) : ℚ :=
  let remainingBudget := budget - initialCost
  let ounces1 := min (remainingBudget / heliumPrice1) threshold1
  let ounces2 := min ((remainingBudget - ounces1 * heliumPrice1) / heliumPrice2) (threshold2 - threshold1)
  let totalOunces := ounces1 + ounces2
  totalOunces * heightPerOunce

/-- The maximum height the balloon can fly is 11,000 feet --/
theorem balloon_height_theorem : 
  maxBalloonHeight 200 74 1.2 1.1 1 50 120 100 = 11000 := by
  sorry

end balloon_height_theorem_l2419_241961


namespace min_operations_2_to_400_l2419_241966

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | MultiplyTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.MultiplyTwo => n * 2

/-- Checks if a sequence of operations transforms start into target --/
def transformsTo (start target : ℕ) (ops : List Operation) : Prop :=
  ops.foldl applyOperation start = target

/-- The minimum number of operations to transform 2 into 400 is 9 --/
theorem min_operations_2_to_400 :
  ∃ (ops : List Operation),
    transformsTo 2 400 ops ∧
    ops.length = 9 ∧
    (∀ (other_ops : List Operation),
      transformsTo 2 400 other_ops → other_ops.length ≥ 9) :=
by sorry

end min_operations_2_to_400_l2419_241966


namespace population_growth_l2419_241960

theorem population_growth (p₀ : ℝ) : 
  let p₁ := p₀ * 1.1
  let p₂ := p₁ * 1.2
  let p₃ := p₂ * 1.3
  (p₃ - p₀) / p₀ * 100 = 71.6 := by
sorry

end population_growth_l2419_241960


namespace f_expression_on_negative_interval_l2419_241922

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_expression_on_negative_interval
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_known : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
sorry

end f_expression_on_negative_interval_l2419_241922


namespace inequality_solution_range_l2419_241990

theorem inequality_solution_range (k : ℝ) : 
  (1 : ℝ) ∈ {x : ℝ | k^2 * x^2 - 6*k*x + 8 ≥ 0} ↔ k ≥ 4 ∨ k ≤ 2 := by
  sorry

end inequality_solution_range_l2419_241990


namespace permutation_problem_l2419_241900

theorem permutation_problem (n : ℕ) : n * (n - 1) = 132 → n = 12 := by sorry

end permutation_problem_l2419_241900


namespace fraction_sum_equal_decimal_l2419_241993

theorem fraction_sum_equal_decimal : 
  (2 / 20 : ℝ) + (8 / 200 : ℝ) + (3 / 300 : ℝ) + 2 * (5 / 40000 : ℝ) = 0.15025 := by
  sorry

end fraction_sum_equal_decimal_l2419_241993


namespace man_upstream_speed_l2419_241984

/-- Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem man_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 20)
  (h2 : speed_downstream = 25) :
  speed_still - (speed_downstream - speed_still) = 15 := by
  sorry


end man_upstream_speed_l2419_241984


namespace puppy_cost_l2419_241934

/-- Calculates the cost of a puppy given the total cost, food requirements, and food prices. -/
theorem puppy_cost (total_cost : ℚ) (weeks : ℕ) (daily_food : ℚ) (bag_size : ℚ) (bag_cost : ℚ) : 
  total_cost = 14 →
  weeks = 3 →
  daily_food = 1/3 →
  bag_size = 7/2 →
  bag_cost = 2 →
  total_cost - (((weeks * 7 * daily_food) / bag_size).ceil * bag_cost) = 10 := by
  sorry

end puppy_cost_l2419_241934


namespace apollonian_circle_apollonian_circle_specific_case_l2419_241923

/-- The locus of points with a constant ratio of distances to two fixed points is a circle. -/
theorem apollonian_circle (k : ℝ) (hk : k > 0 ∧ k ≠ 1) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ (x y : ℝ),
      (Real.sqrt ((x + 1)^2 + y^2)) / (Real.sqrt ((x - 2)^2 + y^2)) = k ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

/-- The specific case where the ratio is 2 and the fixed points are A(-1,0) and B(2,0). -/
theorem apollonian_circle_specific_case :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, 0) ∧ radius = 2 ∧
    ∀ (x y : ℝ),
      (Real.sqrt ((x + 1)^2 + y^2)) / (Real.sqrt ((x - 2)^2 + y^2)) = 2 ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end apollonian_circle_apollonian_circle_specific_case_l2419_241923


namespace tangent_line_to_parabola_l2419_241994

theorem tangent_line_to_parabola (x y : ℝ) :
  (y = x^2) →  -- The curve equation
  (∃ (m b : ℝ), (y = m*x + b) ∧  -- The tangent line equation
                (-1 = m*1 + b) ∧  -- The line passes through (1, -1)
                (∃ (a : ℝ), y = (2*a)*x - a^2 - a)) →  -- Tangent line touches the curve
  ((y = (2 + 2*Real.sqrt 2)*x - (3 + 2*Real.sqrt 2)) ∨
   (y = (2 - 2*Real.sqrt 2)*x - (3 - 2*Real.sqrt 2))) :=
by sorry

end tangent_line_to_parabola_l2419_241994


namespace soda_price_calculation_l2419_241928

theorem soda_price_calculation (remy_morning : ℕ) (nick_diff : ℕ) (evening_sales : ℚ) (evening_increase : ℚ) :
  remy_morning = 55 →
  nick_diff = 6 →
  evening_sales = 55 →
  evening_increase = 3 →
  ∃ (price : ℚ), price = 1/2 ∧ 
    (remy_morning + (remy_morning - nick_diff)) * price + evening_increase = evening_sales :=
by
  sorry

end soda_price_calculation_l2419_241928


namespace division_multiplication_problem_l2419_241977

theorem division_multiplication_problem : 
  let number : ℚ := 4
  let divisor : ℚ := 6
  let multiplier : ℚ := 12
  let result : ℚ := 8
  (number / divisor) * multiplier = result := by sorry

end division_multiplication_problem_l2419_241977


namespace angle_in_fourth_quadrant_l2419_241957

theorem angle_in_fourth_quadrant (α : Real) :
  (0 < α) ∧ (α < π / 2) → (3 * π / 2 < (2 * π - α)) ∧ ((2 * π - α) < 2 * π) := by
  sorry

end angle_in_fourth_quadrant_l2419_241957


namespace sum_of_three_numbers_l2419_241932

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 179) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 21 := by
sorry

end sum_of_three_numbers_l2419_241932


namespace online_price_is_6_l2419_241992

/-- The price of an item online -/
def online_price : ℝ := 6

/-- The price of an item in the regular store -/
def regular_price : ℝ := online_price + 2

/-- The total amount spent in the regular store -/
def regular_total : ℝ := 96

/-- The total amount spent online -/
def online_total : ℝ := 90

/-- The number of additional items bought online compared to the regular store -/
def additional_items : ℕ := 3

theorem online_price_is_6 :
  (online_total / online_price) = (regular_total / regular_price) + additional_items ∧
  online_price = 6 := by sorry

end online_price_is_6_l2419_241992


namespace sum_a_d_equals_two_l2419_241995

theorem sum_a_d_equals_two 
  (a b c d : ℤ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 2 := by
sorry

end sum_a_d_equals_two_l2419_241995


namespace range_of_a_l2419_241926

open Set

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, 2 * x₀^2 - 3 * a * x₀ + 9 < 0) ↔ 
  a ∈ (Iio (-2 * Real.sqrt 2) ∪ Ioi (2 * Real.sqrt 2)) := by
sorry

end range_of_a_l2419_241926


namespace smallest_twice_cube_thrice_square_l2419_241927

theorem smallest_twice_cube_thrice_square :
  (∃ k : ℕ, k > 0 ∧
    (∃ n : ℕ, k = 2 * n^3) ∧
    (∃ m : ℕ, k = 3 * m^2) ∧
    (∀ j : ℕ, j > 0 →
      (∃ p : ℕ, j = 2 * p^3) →
      (∃ q : ℕ, j = 3 * q^2) →
      j ≥ k)) →
  (∃ k : ℕ, k = 432 ∧
    (∃ n : ℕ, k = 2 * n^3) ∧
    (∃ m : ℕ, k = 3 * m^2) ∧
    (∀ j : ℕ, j > 0 →
      (∃ p : ℕ, j = 2 * p^3) →
      (∃ q : ℕ, j = 3 * q^2) →
      j ≥ k)) :=
by sorry

end smallest_twice_cube_thrice_square_l2419_241927


namespace sum_of_xyz_l2419_241910

theorem sum_of_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y + x) : x + y + z = 17 * x := by
  sorry

end sum_of_xyz_l2419_241910


namespace monkey_reach_top_l2419_241983

/-- The time it takes for a monkey to climb a greased pole -/
def monkey_climb_time (pole_height : ℕ) (ascend : ℕ) (slip : ℕ) : ℕ :=
  let effective_progress := ascend - slip
  let full_cycles := (pole_height - ascend) / effective_progress
  2 * full_cycles + 1

/-- Theorem stating that the monkey will reach the top of the pole in 17 minutes -/
theorem monkey_reach_top : monkey_climb_time 10 2 1 = 17 := by
  sorry

end monkey_reach_top_l2419_241983


namespace sufficient_not_necessary_condition_l2419_241944

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end sufficient_not_necessary_condition_l2419_241944


namespace cells_with_three_neighbors_count_l2419_241904

/-- Represents a rectangular grid --/
structure RectangularGrid where
  a : ℕ
  b : ℕ
  h_a : a ≥ 3
  h_b : b ≥ 3

/-- Two cells are neighboring if they share a common side --/
def neighboring (grid : RectangularGrid) : Prop := sorry

/-- The number of cells with exactly four neighboring cells --/
def cells_with_four_neighbors (grid : RectangularGrid) : ℕ :=
  (grid.a - 2) * (grid.b - 2)

/-- The number of cells with exactly three neighboring cells --/
def cells_with_three_neighbors (grid : RectangularGrid) : ℕ :=
  2 * (grid.a - 2) + 2 * (grid.b - 2)

/-- Main theorem: In a rectangular grid where 23 cells have exactly four neighboring cells,
    the number of cells with exactly three neighboring cells is 48 --/
theorem cells_with_three_neighbors_count
  (grid : RectangularGrid)
  (h : cells_with_four_neighbors grid = 23) :
  cells_with_three_neighbors grid = 48 := by
  sorry

end cells_with_three_neighbors_count_l2419_241904


namespace quotient_invariance_l2419_241920

theorem quotient_invariance (a b k : ℝ) (hb : b ≠ 0) (hk : k ≠ 0) :
  (a * k) / (b * k) = a / b := by
  sorry

end quotient_invariance_l2419_241920


namespace point_A_location_l2419_241929

theorem point_A_location (A : ℝ) : 
  (A + 2 = -2 ∨ A - 2 = -2) → (A = 0 ∨ A = -4) := by
sorry

end point_A_location_l2419_241929


namespace power_sum_difference_l2419_241901

theorem power_sum_difference : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  sorry

end power_sum_difference_l2419_241901


namespace parallel_line_plane_not_imply_parallel_lines_l2419_241989

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem parallel_line_plane_not_imply_parallel_lines 
  (l m : Line) (α : Plane) : 
  ¬(parallel_line_plane l α ∧ contains α m → parallel_lines l m) := by
  sorry

end parallel_line_plane_not_imply_parallel_lines_l2419_241989


namespace october_price_correct_l2419_241912

/-- The price of a mobile phone after a certain number of months, given an initial price and a monthly decrease rate. -/
def price_after_months (initial_price : ℝ) (decrease_rate : ℝ) (months : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ months

/-- Theorem stating that the price of a mobile phone in October is correct, given the initial price in January and a 3% monthly decrease. -/
theorem october_price_correct (a : ℝ) : price_after_months a 0.03 9 = a * 0.97^9 := by
  sorry

#check october_price_correct

end october_price_correct_l2419_241912


namespace opposite_of_two_minus_sqrt_five_l2419_241919

theorem opposite_of_two_minus_sqrt_five :
  -(2 - Real.sqrt 5) = Real.sqrt 5 - 2 := by
  sorry

end opposite_of_two_minus_sqrt_five_l2419_241919


namespace xy_value_l2419_241965

theorem xy_value (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x^y = -8 := by
  sorry

end xy_value_l2419_241965


namespace total_groom_time_is_210_l2419_241914

/-- The time it takes to groom a poodle, in minutes. -/
def poodle_groom_time : ℕ := 30

/-- The time it takes to groom a terrier, in minutes. -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- The number of poodles to be groomed. -/
def num_poodles : ℕ := 3

/-- The number of terriers to be groomed. -/
def num_terriers : ℕ := 8

/-- The total grooming time for all dogs. -/
def total_groom_time : ℕ := num_poodles * poodle_groom_time + num_terriers * terrier_groom_time

theorem total_groom_time_is_210 : total_groom_time = 210 := by
  sorry

end total_groom_time_is_210_l2419_241914


namespace age_problem_l2419_241974

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 27) : 
  b = 10 := by
sorry

end age_problem_l2419_241974


namespace fraction_equality_l2419_241902

theorem fraction_equality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_neq_xy : x ≠ y) (h_neq_yz : y ≠ z) (h_neq_xz : x ≠ z)
  (h_eq1 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h_eq2 : (y + 1) / (x + z) = (x + 1) / y) :
  (x + 1) / y = 1 := by
  sorry

end fraction_equality_l2419_241902


namespace parabola_equation_holds_l2419_241937

/-- A parabola with vertex at (2, 9) intersecting the x-axis to form a segment of length 6 -/
structure Parabola where
  vertex : ℝ × ℝ
  intersection_length : ℝ
  vertex_condition : vertex = (2, 9)
  length_condition : intersection_length = 6

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) (x y : ℝ) : Prop :=
  y = -(x - 2)^2 + 9

/-- Theorem stating that the given parabola satisfies the equation -/
theorem parabola_equation_holds (p : Parabola) :
  ∀ x y : ℝ, parabola_equation p x y ↔ 
    ∃ a : ℝ, y = a * (x - p.vertex.1)^2 + p.vertex.2 ∧
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
      a * (x₁ - p.vertex.1)^2 + p.vertex.2 = 0 ∧
      a * (x₂ - p.vertex.1)^2 + p.vertex.2 = 0 ∧
      |x₁ - x₂| = p.intersection_length :=
by
  sorry

end parabola_equation_holds_l2419_241937


namespace expression_simplification_l2419_241950

theorem expression_simplification :
  (((1 + 2 + 3) * 2)^2 / 3) + ((3 * 4 + 6 + 2) / 5) = 52 := by
  sorry

end expression_simplification_l2419_241950


namespace unique_sum_preceding_numbers_l2419_241921

theorem unique_sum_preceding_numbers : 
  ∃! n : ℕ, n > 0 ∧ n = (n * (n - 1)) / 2 :=
by
  sorry

end unique_sum_preceding_numbers_l2419_241921


namespace equation_solution_exists_l2419_241981

theorem equation_solution_exists (a : Real) : 
  a ∈ Set.Icc 0.5 1.5 →
  ∃ t ∈ Set.Icc 0 (Real.pi / 2), 
    (abs (Real.cos t - 0.5) + abs (Real.sin t) - a) / (Real.sqrt 3 * Real.sin t - Real.cos t) = 0 :=
by sorry

end equation_solution_exists_l2419_241981


namespace unattainable_value_l2419_241939

/-- The function f(x) = (1-2x) / (3x+4) cannot attain the value -2/3 for any real x ≠ -4/3. -/
theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  (1 - 2*x) / (3*x + 4) ≠ -2/3 := by
  sorry


end unattainable_value_l2419_241939


namespace smallest_root_of_unity_order_l2419_241913

open Complex

theorem smallest_root_of_unity_order : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^3 - z + 1 = 0 → z^n = 1) ∧ 
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^3 - z + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 5 := by
  sorry

end smallest_root_of_unity_order_l2419_241913


namespace shoe_pairing_probability_l2419_241978

/-- A permutation of n elements -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The number of permutations of n elements with all cycle lengths ≥ k -/
def numLongCyclePerms (n k : ℕ) : ℕ := sorry

/-- The probability of a random permutation of n elements having all cycle lengths ≥ k -/
def probLongCycles (n k : ℕ) : ℚ :=
  (numLongCyclePerms n k : ℚ) / (n.factorial : ℚ)

theorem shoe_pairing_probability :
  probLongCycles 8 5 = 1 / 8 := by sorry

end shoe_pairing_probability_l2419_241978


namespace same_number_on_cards_l2419_241999

theorem same_number_on_cards (n : ℕ) (cards : Fin n → ℕ) : 
  (∀ i, cards i ∈ Finset.range n) →
  (∀ s : Finset (Fin n), (s.sum cards) % (n + 1) ≠ 0) →
  ∀ i j, cards i = cards j :=
by sorry

end same_number_on_cards_l2419_241999


namespace three_X_four_equals_31_l2419_241915

-- Define the operation X
def X (a b : ℤ) : ℤ := b + 12 * a - a^2

-- Theorem statement
theorem three_X_four_equals_31 : X 3 4 = 31 := by
  sorry

end three_X_four_equals_31_l2419_241915


namespace angle_set_impossibility_l2419_241952

/-- Represents a set of angles formed by lines through a single point -/
structure AngleSet where
  odd : ℕ  -- number of angles with odd integer measures
  even : ℕ -- number of angles with even integer measures

/-- The property that the number of odd-measure angles is 15 more than even-measure angles -/
def has_15_more_odd (as : AngleSet) : Prop :=
  as.odd = as.even + 15

/-- The property that both odd and even counts are even numbers due to vertical angles -/
def vertical_angle_property (as : AngleSet) : Prop :=
  Even as.odd ∧ Even as.even

theorem angle_set_impossibility : 
  ¬∃ (as : AngleSet), has_15_more_odd as ∧ vertical_angle_property as :=
sorry

end angle_set_impossibility_l2419_241952


namespace chord_equation_through_midpoint_l2419_241971

/-- The equation of a chord passing through a point on an ellipse --/
theorem chord_equation_through_midpoint (x y : ℝ) :
  (4 * x^2 + 9 * y^2 = 144) →  -- Ellipse equation
  (3 : ℝ)^2 * 4 + 1^2 * 9 < 144 →  -- P(3,1) is inside the ellipse
  (∃ (x₁ y₁ x₂ y₂ : ℝ),  -- Existence of chord endpoints
    (4 * x₁^2 + 9 * y₁^2 = 144) ∧  -- A is on the ellipse
    (4 * x₂^2 + 9 * y₂^2 = 144) ∧  -- B is on the ellipse
    (x₁ + x₂ = 6) ∧  -- P is midpoint (x-coordinate)
    (y₁ + y₂ = 2)) →  -- P is midpoint (y-coordinate)
  (4 * x + 3 * y - 15 = 0)  -- Equation of the chord
  := by sorry

end chord_equation_through_midpoint_l2419_241971


namespace integer_solution_x4_y4_eq_3x3y_l2419_241996

theorem integer_solution_x4_y4_eq_3x3y :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y ↔ x = 0 ∧ y = 0 := by
sorry

end integer_solution_x4_y4_eq_3x3y_l2419_241996


namespace cristinas_pace_cristinas_pace_is_five_l2419_241925

/-- Cristina's pace in a race with Nicky -/
theorem cristinas_pace (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let total_distance := head_start + nickys_pace * catch_up_time
  total_distance / catch_up_time

/-- Prove that Cristina's pace is 5 meters per second -/
theorem cristinas_pace_is_five :
  cristinas_pace 48 3 24 = 5 := by
  sorry

end cristinas_pace_cristinas_pace_is_five_l2419_241925


namespace pasture_feeding_duration_l2419_241997

/-- Represents the daily grass consumption of a single cow -/
def daily_consumption_per_cow : ℝ := sorry

/-- Represents the initial amount of grass in the pasture -/
def initial_grass : ℝ := sorry

/-- Represents the daily growth rate of grass -/
def daily_growth_rate : ℝ := sorry

/-- The grass consumed by a number of cows over a period of days
    equals the initial grass plus the grass grown during that period -/
def grass_consumption (cows : ℝ) (days : ℝ) : Prop :=
  cows * daily_consumption_per_cow * days = initial_grass + daily_growth_rate * days

theorem pasture_feeding_duration :
  grass_consumption 20 40 ∧ grass_consumption 35 10 →
  grass_consumption 25 20 := by sorry

end pasture_feeding_duration_l2419_241997


namespace fraction_equality_l2419_241941

theorem fraction_equality : (1 / 5 - 1 / 6) / (1 / 4 - 1 / 5) = 2 / 3 := by
  sorry

end fraction_equality_l2419_241941


namespace school_students_l2419_241909

/-- Prove that the total number of students in a school is 1000, given the conditions described. -/
theorem school_students (S : ℕ) : 
  (S / 2) / 2 = 250 → S = 1000 := by
  sorry

end school_students_l2419_241909


namespace smallest_marble_count_l2419_241956

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing two marbles of one color and two of another --/
def prob_two_two (m : MarbleCount) (c1 c2 : ℕ) : ℚ :=
  (c1.choose 2 * c2.choose 2 : ℚ) / (m.red + m.white + m.blue + m.green).choose 4

/-- Calculates the probability of drawing one marble of each color --/
def prob_one_each (m : MarbleCount) : ℚ :=
  (m.red * m.white * m.blue * m.green : ℚ) / (m.red + m.white + m.blue + m.green).choose 4

/-- Checks if the probabilities of the three events are equal --/
def probabilities_equal (m : MarbleCount) : Prop :=
  prob_two_two m m.red m.blue = prob_two_two m m.white m.green ∧
  prob_two_two m m.red m.blue = prob_one_each m

/-- The theorem stating that 10 is the smallest number of marbles satisfying the conditions --/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), 
    (m.red + m.white + m.blue + m.green = 10) ∧ 
    probabilities_equal m ∧
    (∀ (n : MarbleCount), 
      (n.red + n.white + n.blue + n.green < 10) → ¬probabilities_equal n) :=
  sorry

end smallest_marble_count_l2419_241956


namespace children_bridge_problem_l2419_241930

/-- The problem of three children crossing a bridge --/
theorem children_bridge_problem (bridge_capacity : ℝ) (kelly_weight : ℝ) :
  bridge_capacity = 100 →
  kelly_weight = 34 →
  ∃ (megan_weight : ℝ) (mike_weight : ℝ),
    kelly_weight = 0.85 * megan_weight ∧
    mike_weight = megan_weight + 5 ∧
    kelly_weight + megan_weight + mike_weight - bridge_capacity = 19 :=
by sorry

end children_bridge_problem_l2419_241930
