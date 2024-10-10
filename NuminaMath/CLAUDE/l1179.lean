import Mathlib

namespace factor_cubic_expression_l1179_117933

theorem factor_cubic_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) 
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end factor_cubic_expression_l1179_117933


namespace sufficient_not_necessary_condition_l1179_117969

-- Define sets A and B
def A : Set ℝ := {x | x > 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Define the theorem
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) → a < 5 := by
  sorry

end sufficient_not_necessary_condition_l1179_117969


namespace max_value_constrained_l1179_117974

/-- Given non-negative real numbers x and y satisfying the constraints
x + 2y ≤ 6 and 2x + y ≤ 6, the maximum value of x + y is 4. -/
theorem max_value_constrained (x y : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) 
  (h1 : x + 2*y ≤ 6) (h2 : 2*x + y ≤ 6) : 
  x + y ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ x₀ + 2*y₀ ≤ 6 ∧ 2*x₀ + y₀ ≤ 6 ∧ x₀ + y₀ = 4 :=
by sorry

end max_value_constrained_l1179_117974


namespace right_triangle_with_constraints_l1179_117962

/-- A right-angled triangle with perimeter 5 and shortest altitude 1 has side lengths 5/3, 5/4, and 25/12. -/
theorem right_triangle_with_constraints (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  a^2 + b^2 = c^2 →  -- right-angled triangle (Pythagorean theorem)
  a + b + c = 5 →  -- perimeter is 5
  min (a*b/c) (min (b*c/a) (c*a/b)) = 1 →  -- shortest altitude is 1
  ((a = 5/3 ∧ b = 5/4 ∧ c = 25/12) ∨ (a = 5/4 ∧ b = 5/3 ∧ c = 25/12)) := by
sorry


end right_triangle_with_constraints_l1179_117962


namespace tan_seventeen_pi_fourths_l1179_117918

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end tan_seventeen_pi_fourths_l1179_117918


namespace pigeonhole_birthday_l1179_117973

theorem pigeonhole_birthday (n : ℕ) :
  (∀ f : Fin n → Fin 366, ∃ i j, i ≠ j ∧ f i = f j) ↔ n ≥ 367 := by
  sorry

end pigeonhole_birthday_l1179_117973


namespace problem_solution_l1179_117955

theorem problem_solution (m n : ℝ) (h : |3*m - 15| + ((n/3 + 1)^2) = 0) : 2*m - n = 13 := by
  sorry

end problem_solution_l1179_117955


namespace fraction_comparison_and_differences_l1179_117935

theorem fraction_comparison_and_differences :
  (1 / 3 : ℚ) < (1 / 2 : ℚ) ∧ (1 / 2 : ℚ) < (3 / 5 : ℚ) ∧
  (1 / 2 : ℚ) - (1 / 3 : ℚ) = (1 / 6 : ℚ) ∧
  (3 / 5 : ℚ) - (1 / 2 : ℚ) = (1 / 10 : ℚ) :=
by sorry

end fraction_comparison_and_differences_l1179_117935


namespace lindas_coins_l1179_117998

/-- Represents the number of coins Linda has initially -/
structure InitialCoins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Represents the number of coins Linda's mother gives her -/
structure AdditionalCoins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- The problem statement -/
theorem lindas_coins (initial : InitialCoins) (additional : AdditionalCoins) 
    (h1 : initial.quarters = 6)
    (h2 : initial.nickels = 5)
    (h3 : additional.dimes = 2)
    (h4 : additional.quarters = 10)
    (h5 : additional.nickels = 2 * initial.nickels)
    (h6 : initial.dimes + initial.quarters + initial.nickels + 
          additional.dimes + additional.quarters + additional.nickels = 35) :
    initial.dimes = 4 := by
  sorry


end lindas_coins_l1179_117998


namespace milk_replacement_problem_l1179_117980

theorem milk_replacement_problem (initial_volume : ℝ) (final_pure_milk : ℝ) :
  initial_volume = 45 ∧ final_pure_milk = 28.8 →
  ∃ (x : ℝ), x = 9 ∧ 
  (initial_volume - x) * (initial_volume - x) / initial_volume = final_pure_milk :=
by sorry

end milk_replacement_problem_l1179_117980


namespace second_machine_rate_l1179_117979

/-- Represents a copy machine with a constant rate of copies per minute -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Represents two copy machines working together -/
structure TwoMachines where
  machine1 : CopyMachine
  machine2 : CopyMachine

/-- The total number of copies produced by two machines in a given time -/
def total_copies (machines : TwoMachines) (minutes : ℕ) : ℕ :=
  (machines.machine1.copies_per_minute + machines.machine2.copies_per_minute) * minutes

theorem second_machine_rate (machines : TwoMachines) 
  (h1 : machines.machine1.copies_per_minute = 25)
  (h2 : total_copies machines 30 = 2400) :
  machines.machine2.copies_per_minute = 55 := by
sorry

end second_machine_rate_l1179_117979


namespace min_squared_distance_to_origin_l1179_117951

theorem min_squared_distance_to_origin (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → 
  ∃ (min : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 1 :=
sorry

end min_squared_distance_to_origin_l1179_117951


namespace probability_3400_is_3_32_l1179_117987

/-- The number of non-bankrupt outcomes on the spinner -/
def num_outcomes : ℕ := 4

/-- The total number of possible combinations in three spins -/
def total_combinations : ℕ := num_outcomes ^ 3

/-- The number of ways to arrange three specific amounts that sum to $3400 -/
def favorable_arrangements : ℕ := 6

/-- The probability of earning exactly $3400 in three spins -/
def probability_3400 : ℚ := favorable_arrangements / total_combinations

theorem probability_3400_is_3_32 : probability_3400 = 3 / 32 := by
  sorry

end probability_3400_is_3_32_l1179_117987


namespace complex_root_modulus_sqrt5_l1179_117916

theorem complex_root_modulus_sqrt5 (k : ℝ) :
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  k = -1 ∨ k = 3 := by
sorry

end complex_root_modulus_sqrt5_l1179_117916


namespace min_side_arithmetic_angles_l1179_117924

/-- Given a triangle ABC where the internal angles form an arithmetic sequence and the area is 2√3,
    the minimum value of side AB is 2√2. -/
theorem min_side_arithmetic_angles (A B C : ℝ) (a b c : ℝ) :
  -- Angles form an arithmetic sequence
  2 * C = A + B →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  -- Area of the triangle is 2√3
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 →
  -- AB is the side opposite to angle C
  c = (a^2 + b^2 - 2*a*b*(Real.cos C))^(1/2) →
  -- Minimum value of AB (c) is 2√2
  c ≥ 2 * Real.sqrt 2 ∧ ∃ (a' b' : ℝ), c = 2 * Real.sqrt 2 := by
  sorry

end min_side_arithmetic_angles_l1179_117924


namespace total_money_problem_l1179_117928

theorem total_money_problem (brad : ℝ) (josh : ℝ) (doug : ℝ) 
  (h1 : brad = 12.000000000000002)
  (h2 : josh = 2 * brad)
  (h3 : josh = (3/4) * doug) : 
  brad + josh + doug = 68.00000000000001 := by
  sorry

end total_money_problem_l1179_117928


namespace ming_ladybugs_l1179_117972

/-- The number of spiders Sami found -/
def spiders : ℕ := 3

/-- The number of ants Hunter saw -/
def ants : ℕ := 12

/-- The number of ladybugs that flew away -/
def flown_ladybugs : ℕ := 2

/-- The number of insects remaining in the playground -/
def remaining_insects : ℕ := 21

/-- The number of ladybugs Ming discovered initially -/
def initial_ladybugs : ℕ := remaining_insects + flown_ladybugs - (spiders + ants)

theorem ming_ladybugs : initial_ladybugs = 8 := by
  sorry

end ming_ladybugs_l1179_117972


namespace greatest_multiple_under_1000_l1179_117940

theorem greatest_multiple_under_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by sorry

end greatest_multiple_under_1000_l1179_117940


namespace system_solution_ratio_l1179_117927

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 →
  8 * x - 6 * y = c →
  12 * y - 18 * x = d →
  c / d = -4 / 9 := by
sorry

end system_solution_ratio_l1179_117927


namespace alice_bob_earnings_l1179_117914

/-- Given the working hours and hourly rates of Alice and Bob, prove that the value of t that makes their earnings equal is 7.8 -/
theorem alice_bob_earnings (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 := by
  sorry

end alice_bob_earnings_l1179_117914


namespace f_derivative_at_2_l1179_117939

-- Define the function f
def f (f'2 : ℝ) : ℝ → ℝ := λ x ↦ 3 * x^2 - 2 * x * f'2

-- State the theorem
theorem f_derivative_at_2 : 
  ∃ f'2 : ℝ, (deriv (f f'2)) 2 = 4 := by sorry

end f_derivative_at_2_l1179_117939


namespace not_equal_implies_not_both_zero_l1179_117965

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem not_equal_implies_not_both_zero (a b : V) (h : a ≠ b) : ¬(a = 0 ∧ b = 0) := by
  sorry

end not_equal_implies_not_both_zero_l1179_117965


namespace max_cube_path_length_l1179_117942

/-- Represents a cube with edges of a given length -/
structure Cube where
  edgeLength : ℝ
  edgeCount : ℕ

/-- Represents a path on the cube -/
structure CubePath where
  length : ℝ
  edgeCount : ℕ

/-- The maximum path length on a cube without retracing -/
def maxPathLength (c : Cube) : ℝ := sorry

theorem max_cube_path_length 
  (c : Cube) 
  (h1 : c.edgeLength = 3)
  (h2 : c.edgeCount = 12) :
  maxPathLength c = 24 := by sorry

end max_cube_path_length_l1179_117942


namespace circle_line_intersection_l1179_117930

theorem circle_line_intersection (a b : ℝ) : 
  (a^2 + b^2 > 1) →
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y + 2 = 0) →
  (a^2 + b^2 > 1) ∧
  ¬(∀ (a b : ℝ), a^2 + b^2 > 1 → ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y + 2 = 0) :=
by sorry

end circle_line_intersection_l1179_117930


namespace inequality_equivalence_l1179_117945

theorem inequality_equivalence (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := by
  sorry

end inequality_equivalence_l1179_117945


namespace playground_boys_count_l1179_117954

theorem playground_boys_count (total_children girls : ℕ) 
  (h1 : total_children = 63) 
  (h2 : girls = 28) : 
  total_children - girls = 35 := by
sorry

end playground_boys_count_l1179_117954


namespace vote_ratio_proof_l1179_117978

def candidate_A_votes : ℕ := 14
def total_votes : ℕ := 21

theorem vote_ratio_proof :
  let candidate_B_votes := total_votes - candidate_A_votes
  (candidate_A_votes : ℚ) / candidate_B_votes = 2 := by
  sorry

end vote_ratio_proof_l1179_117978


namespace amy_game_score_l1179_117915

theorem amy_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ)
  (h1 : points_per_treasure = 4)
  (h2 : treasures_level1 = 6)
  (h3 : treasures_level2 = 2) :
  points_per_treasure * treasures_level1 + points_per_treasure * treasures_level2 = 32 := by
  sorry

end amy_game_score_l1179_117915


namespace min_largest_group_size_l1179_117990

theorem min_largest_group_size (total_boxes : ℕ) (min_apples max_apples : ℕ) : 
  total_boxes = 128 →
  min_apples = 120 →
  max_apples = 144 →
  ∃ (n : ℕ), n = 6 ∧ 
    (∀ (group_size : ℕ), 
      (group_size * (max_apples - min_apples + 1) ≥ total_boxes → group_size ≥ n) ∧
      (∃ (distribution : List ℕ), 
        distribution.length = max_apples - min_apples + 1 ∧
        distribution.sum = total_boxes ∧
        ∀ (x : ℕ), x ∈ distribution → x ≤ n)) :=
by sorry

end min_largest_group_size_l1179_117990


namespace symmetry_implies_difference_l1179_117966

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_implies_difference (a b : ℝ) :
  symmetric_wrt_origin (-2, b) (a, 3) → a - b = 5 := by
  sorry

end symmetry_implies_difference_l1179_117966


namespace largest_integer_with_remainder_l1179_117902

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 5 = 2 ∧ ∀ m : ℕ, m < 100 ∧ m % 5 = 2 → m ≤ n → n = 97 := by
  sorry

end largest_integer_with_remainder_l1179_117902


namespace equal_playing_time_l1179_117984

theorem equal_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) % total_players = 0 →
  (players_on_field * match_duration) / total_players = 36 := by
  sorry

end equal_playing_time_l1179_117984


namespace inequalities_proof_l1179_117971

theorem inequalities_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) : 
  (a * b > a * c) ∧ (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) := by
  sorry

end inequalities_proof_l1179_117971


namespace gcd_problem_l1179_117961

/-- The greatest common divisor of (123^2 + 235^2 + 347^2) and (122^2 + 234^2 + 348^2) is 1 -/
theorem gcd_problem : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end gcd_problem_l1179_117961


namespace fruit_baskets_count_l1179_117957

/-- The number of non-empty fruit baskets -/
def num_fruit_baskets (num_apples num_oranges : ℕ) : ℕ :=
  (num_apples + 1) * (num_oranges + 1) - 1

/-- Theorem: The number of non-empty fruit baskets with 6 apples and 12 oranges is 90 -/
theorem fruit_baskets_count :
  num_fruit_baskets 6 12 = 90 := by
sorry

end fruit_baskets_count_l1179_117957


namespace billy_crayons_l1179_117908

theorem billy_crayons (initial remaining eaten : ℕ) 
  (h1 : eaten = 52)
  (h2 : remaining = 10)
  (h3 : initial = remaining + eaten) :
  initial = 62 := by
  sorry

end billy_crayons_l1179_117908


namespace quadratic_inequality_roots_l1179_117989

theorem quadratic_inequality_roots (k : ℝ) : 
  (∀ x : ℝ, -x^2 + k*x + 4 < 0 ↔ x < 2 ∨ x > 3) → k = 5 := by
  sorry

end quadratic_inequality_roots_l1179_117989


namespace train_speed_l1179_117948

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 900) (h2 : time = 12) :
  length / time = 75 := by
  sorry

#check train_speed

end train_speed_l1179_117948


namespace difference_between_point_eight_and_half_l1179_117922

theorem difference_between_point_eight_and_half : 0.8 - (1/2 : ℚ) = 0.3 := by
  sorry

end difference_between_point_eight_and_half_l1179_117922


namespace ellipse_special_point_l1179_117938

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

def line_intersects_ellipse (m t x y : ℝ) : Prop :=
  ellipse x y ∧ x = t*y + m

def distance_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2

theorem ellipse_special_point :
  ∃ (m : ℝ), 
    (∀ (t x₁ y₁ x₂ y₂ : ℝ),
      line_intersects_ellipse m t x₁ y₁ ∧ 
      line_intersects_ellipse m t x₂ y₂ ∧ 
      (x₁, y₁) ≠ (x₂, y₂) →
      ∃ (k : ℝ), 
        1 / distance_squared m 0 x₁ y₁ + 1 / distance_squared m 0 x₂ y₂ = k) ∧
    m = 2 * Real.sqrt 15 / 5 ∧
    (∀ (t x₁ y₁ x₂ y₂ : ℝ),
      line_intersects_ellipse m t x₁ y₁ ∧ 
      line_intersects_ellipse m t x₂ y₂ ∧ 
      (x₁, y₁) ≠ (x₂, y₂) →
      1 / distance_squared m 0 x₁ y₁ + 1 / distance_squared m 0 x₂ y₂ = 5) :=
sorry

end ellipse_special_point_l1179_117938


namespace last_two_digits_of_A_power_20_l1179_117937

theorem last_two_digits_of_A_power_20 (A : ℤ) 
  (h1 : A % 2 = 0) 
  (h2 : A % 10 ≠ 0) : 
  A^20 % 100 = 76 := by
sorry

end last_two_digits_of_A_power_20_l1179_117937


namespace circle_ratio_proof_l1179_117992

theorem circle_ratio_proof (b a c : ℝ) (h1 : b > 0) (h2 : a > 0) (h3 : c > 0)
  (h4 : b^2 - c^2 = 2 * a^2) (h5 : c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 := by
sorry

end circle_ratio_proof_l1179_117992


namespace trigonometric_inequalities_l1179_117958

theorem trigonometric_inequalities (α β γ : ℝ) : 
  (|Real.cos (α + β)| ≤ |Real.cos α| + |Real.sin β|) ∧ 
  (|Real.sin (α + β)| ≤ |Real.cos α| + |Real.cos β|) ∧ 
  (α + β + γ = 0 → |Real.cos α| + |Real.cos β| + |Real.cos γ| ≥ 1) :=
by sorry

end trigonometric_inequalities_l1179_117958


namespace find_liar_in_17_questions_l1179_117920

/-- Represents a person who can be either a knight or a liar -/
inductive Person
  | knight : Person
  | liar : Person

/-- Represents the response to a question -/
inductive Response
  | yes : Response
  | no : Response

/-- A function that simulates asking a question to a person -/
def ask (p : Person) (cardNumber : Nat) (askedNumber : Nat) : Response :=
  match p with
  | Person.knight => if cardNumber = askedNumber then Response.yes else Response.no
  | Person.liar => if cardNumber ≠ askedNumber then Response.yes else Response.no

/-- The main theorem statement -/
theorem find_liar_in_17_questions 
  (people : Fin 10 → Person) 
  (cards : Fin 10 → Nat) 
  (h1 : ∃! i, people i = Person.liar) 
  (h2 : ∀ i j, i ≠ j → cards i ≠ cards j) 
  (h3 : ∀ i, cards i ∈ Set.range (fun n : Nat => n + 1) ∩ Set.range (fun n : Nat => 11 - n)) :
  ∃ (strategy : Nat → Fin 10 × Nat), 
    (∀ n, n < 17 → (strategy n).2 ∈ Set.range (fun n : Nat => n + 1) ∩ Set.range (fun n : Nat => 11 - n)) →
    ∃ (result : Fin 10), 
      (∀ i, i ≠ result → people i = Person.knight) ∧ 
      (people result = Person.liar) :=
sorry

end find_liar_in_17_questions_l1179_117920


namespace expression_factorization_l1179_117913

theorem expression_factorization (x : ℝ) :
  (9 * x^5 + 25 * x^3 - 4) - (x^5 - 3 * x^3 - 4) = 4 * x^3 * (2 * x^2 + 7) := by
  sorry

end expression_factorization_l1179_117913


namespace bowling_team_size_l1179_117952

theorem bowling_team_size (original_avg : ℝ) (new_player1_weight : ℝ) (new_player2_weight : ℝ) (new_avg : ℝ) :
  original_avg = 121 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  new_avg = 113 →
  ∃ n : ℕ, n > 0 ∧ 
    (n * original_avg + new_player1_weight + new_player2_weight) / (n + 2) = new_avg ∧
    n = 7 :=
by sorry

end bowling_team_size_l1179_117952


namespace sugar_calculation_l1179_117903

/-- Proves that the total amount of sugar the owner started with is 14100 grams --/
theorem sugar_calculation (total_packs : ℕ) (pack_weight : ℝ) (remaining_sugar : ℝ) :
  total_packs = 35 →
  pack_weight = 400 →
  remaining_sugar = 100 →
  (total_packs : ℝ) * pack_weight + remaining_sugar = 14100 := by
  sorry

end sugar_calculation_l1179_117903


namespace smallest_positive_multiple_of_45_l1179_117968

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l1179_117968


namespace max_product_with_sum_and_even_l1179_117907

theorem max_product_with_sum_and_even (x y : ℤ) : 
  x + y = 280 → (Even x ∨ Even y) → x * y ≤ 19600 := by
  sorry

end max_product_with_sum_and_even_l1179_117907


namespace range_of_a_l1179_117944

theorem range_of_a (a : ℝ) : 
  (∀ x₀ : ℝ, ∀ x : ℝ, x + a * x₀ + 1 ≥ 0) → a ∈ Set.Icc (-2) 2 :=
by sorry

end range_of_a_l1179_117944


namespace division_multiplication_equivalence_l1179_117909

theorem division_multiplication_equivalence : 
  ∀ (x : ℚ), x * (9 / 3) * (5 / 6) = x / (2 / 5) :=
by sorry

end division_multiplication_equivalence_l1179_117909


namespace percentage_ratio_theorem_l1179_117977

theorem percentage_ratio_theorem (y : ℝ) : 
  let x := 7 * y
  let z := 3 * (x - y)
  let percentage := (x - y) / x * 100
  let ratio := z / (x + y)
  percentage / ratio = 800 / 21 := by
sorry

end percentage_ratio_theorem_l1179_117977


namespace roots_sum_of_powers_l1179_117904

theorem roots_sum_of_powers (α β : ℝ) : 
  α^2 - 2*α - 1 = 0 → β^2 - 2*β - 1 = 0 → 5*α^4 + 12*β^3 = 169 := by
  sorry

end roots_sum_of_powers_l1179_117904


namespace joanne_first_hour_coins_l1179_117994

/-- Represents the number of coins Joanne collected in the first hour -/
def first_hour_coins : ℕ := sorry

/-- Represents the total number of coins collected in the second and third hours -/
def second_third_hour_coins : ℕ := 35

/-- Represents the number of coins collected in the fourth hour -/
def fourth_hour_coins : ℕ := 50

/-- Represents the number of coins given to the coworker -/
def coins_given_away : ℕ := 15

/-- Represents the total number of coins after the fourth hour -/
def total_coins : ℕ := 120

/-- Theorem stating that Joanne collected 15 coins in the first hour -/
theorem joanne_first_hour_coins : 
  first_hour_coins = 15 :=
by
  sorry

#check joanne_first_hour_coins

end joanne_first_hour_coins_l1179_117994


namespace share_ratio_l1179_117941

/-- Given a total amount divided among three people (a, b, c), prove the ratio of a's share to the sum of b's and c's shares -/
theorem share_ratio (total a b c : ℚ) : 
  total = 100 →
  a = 20 →
  b = (3 / 5) * (a + c) →
  total = a + b + c →
  a / (b + c) = 1 / 4 := by
sorry

end share_ratio_l1179_117941


namespace triangle_properties_l1179_117949

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b + c = 3 →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  (∃ (R : ℝ), R > 0 ∧ R * (a + b + c) = a * b * Real.sin C) →
  C = π / 3 ∧
  (∀ (S : ℝ), S = π * R^2 → S ≤ π / 12) :=
by sorry

end triangle_properties_l1179_117949


namespace unique_number_property_l1179_117981

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end unique_number_property_l1179_117981


namespace binomial_12_choose_10_l1179_117921

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by sorry

end binomial_12_choose_10_l1179_117921


namespace average_first_30_multiples_of_29_l1179_117956

theorem average_first_30_multiples_of_29 : 
  let n : ℕ := 30
  let base : ℕ := 29
  let sum : ℕ := n * (base + n * base) / 2
  (sum : ℚ) / n = 449.5 := by sorry

end average_first_30_multiples_of_29_l1179_117956


namespace cube_tank_volume_l1179_117997

/-- Represents the number of metal sheets required to make the cube-shaped tank -/
def required_sheets : ℝ := 74.99999999999997

/-- Represents the length of a metal sheet in meters -/
def sheet_length : ℝ := 4

/-- Represents the width of a metal sheet in meters -/
def sheet_width : ℝ := 2

/-- Represents the number of faces in a cube -/
def cube_faces : ℕ := 6

/-- Represents the conversion factor from cubic meters to liters -/
def cubic_meter_to_liter : ℝ := 1000

/-- Theorem stating that the volume of the cube-shaped tank is 1,000,000 liters -/
theorem cube_tank_volume :
  let sheet_area := sheet_length * sheet_width
  let sheets_per_face := required_sheets / cube_faces
  let face_area := sheets_per_face * sheet_area
  let side_length := Real.sqrt face_area
  let volume_cubic_meters := side_length ^ 3
  let volume_liters := volume_cubic_meters * cubic_meter_to_liter
  volume_liters = 1000000 := by
  sorry

end cube_tank_volume_l1179_117997


namespace mona_unique_players_l1179_117959

/-- Represents the number of groups Mona joined --/
def total_groups : ℕ := 18

/-- Represents the number of groups where Mona encountered 2 previous players --/
def groups_with_two_previous : ℕ := 6

/-- Represents the number of groups where Mona encountered 1 previous player --/
def groups_with_one_previous : ℕ := 4

/-- Represents the number of players in the first large group --/
def first_large_group : ℕ := 9

/-- Represents the number of previous players in the first large group --/
def previous_in_first_large : ℕ := 4

/-- Represents the number of players in the second large group --/
def second_large_group : ℕ := 12

/-- Represents the number of previous players in the second large group --/
def previous_in_second_large : ℕ := 5

/-- Theorem stating that Mona grouped with at least 20 unique players --/
theorem mona_unique_players : ℕ := by
  sorry

end mona_unique_players_l1179_117959


namespace floor_sqrt_18_squared_l1179_117975

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end floor_sqrt_18_squared_l1179_117975


namespace rectangular_plot_longer_side_l1179_117931

theorem rectangular_plot_longer_side 
  (width : ℝ) 
  (num_poles : ℕ) 
  (pole_distance : ℝ) :
  width = 40 ∧ 
  num_poles = 36 ∧ 
  pole_distance = 5 →
  ∃ length : ℝ, 
    length > width ∧
    2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance ∧
    length = 47.5 := by
  sorry

end rectangular_plot_longer_side_l1179_117931


namespace candy_distribution_l1179_117964

theorem candy_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (Nat.choose (n + k - 1) (k - 1)) = 66 := by
  sorry

end candy_distribution_l1179_117964


namespace expression_value_l1179_117947

theorem expression_value (x y : ℝ) (h : x - 2*y + 3 = 0) : 
  (2*y - x)^2 - 2*x + 4*y - 1 = 14 := by
sorry

end expression_value_l1179_117947


namespace sqrt_AB_value_l1179_117991

def A : ℕ := 10^9 - 987654321
def B : ℚ := (123456789 + 1) / 10

theorem sqrt_AB_value : Real.sqrt (A * B) = 12345679 := by sorry

end sqrt_AB_value_l1179_117991


namespace fermat_like_theorem_l1179_117995

theorem fermat_like_theorem (k : ℕ) : ¬ ∃ (x y z : ℤ), 
  (x^k + y^k = z^k) ∧ (z > 0) ∧ (0 < x) ∧ (x < k) ∧ (0 < y) ∧ (y < k) := by
  sorry

end fermat_like_theorem_l1179_117995


namespace unique_remainder_mod_10_l1179_117917

theorem unique_remainder_mod_10 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := by
  sorry

end unique_remainder_mod_10_l1179_117917


namespace dan_has_five_marbles_l1179_117923

/-- The number of blue marbles Dan has -/
def dans_marbles : ℕ := sorry

/-- The number of blue marbles Mary has -/
def marys_marbles : ℕ := 10

/-- Mary has 2 times more blue marbles than Dan -/
axiom mary_double_dan : marys_marbles = 2 * dans_marbles

theorem dan_has_five_marbles : dans_marbles = 5 := by
  sorry

end dan_has_five_marbles_l1179_117923


namespace min_value_fraction_equality_condition_l1179_117934

theorem min_value_fraction (x : ℝ) (h : x > 6) : x^2 / (x - 6) ≥ 18 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 6) : x^2 / (x - 6) = 18 ↔ x = 12 := by
  sorry

end min_value_fraction_equality_condition_l1179_117934


namespace stratified_sampling_result_l1179_117900

def grade10_students : ℕ := 300
def grade11_students : ℕ := 200
def grade12_students : ℕ := 400
def total_selected : ℕ := 18

def total_students : ℕ := grade10_students + grade11_students + grade12_students

def stratified_sample (grade_students : ℕ) : ℕ :=
  (total_selected * grade_students) / total_students

theorem stratified_sampling_result :
  (stratified_sample grade10_students,
   stratified_sample grade11_students,
   stratified_sample grade12_students) = (6, 4, 8) := by
  sorry

end stratified_sampling_result_l1179_117900


namespace min_value_of_a2_plus_b2_l1179_117986

theorem min_value_of_a2_plus_b2 (a b : ℝ) :
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) → a'^2 + b'^2 ≥ 4/5) ∧
  (∃ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) ∧ a'^2 + b'^2 = 4/5) :=
by sorry


end min_value_of_a2_plus_b2_l1179_117986


namespace loan_amount_calculation_l1179_117905

/-- Calculates the total loan amount given the loan term, down payment, and monthly payment. -/
def total_loan_amount (loan_term_years : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) : ℕ :=
  down_payment + loan_term_years * 12 * monthly_payment

/-- Theorem stating that given the specific loan conditions, the total loan amount is $46,000. -/
theorem loan_amount_calculation :
  total_loan_amount 5 10000 600 = 46000 := by
  sorry

end loan_amount_calculation_l1179_117905


namespace blue_chip_value_l1179_117946

theorem blue_chip_value (yellow_value : ℕ) (green_value : ℕ) (yellow_count : ℕ) (blue_count : ℕ) (total_product : ℕ) :
  yellow_value = 2 →
  green_value = 5 →
  yellow_count = 4 →
  blue_count = blue_count →  -- This represents that blue and green chip counts are equal
  total_product = 16000 →
  total_product = yellow_value ^ yellow_count * blue_value ^ blue_count * green_value ^ blue_count →
  blue_value = 8 := by
  sorry

#check blue_chip_value

end blue_chip_value_l1179_117946


namespace distribute_six_balls_three_boxes_l1179_117910

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes 
    such that each box contains at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 540 ways to distribute 6 distinguishable balls
    into 3 distinguishable boxes such that each box contains at least one ball -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 540 := by
  sorry

end distribute_six_balls_three_boxes_l1179_117910


namespace expected_boy_girl_pairs_l1179_117901

/-- The expected number of adjacent boy-girl pairs when 10 boys and 14 girls
    are seated randomly around a circular table with 24 seats. -/
theorem expected_boy_girl_pairs :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 14
  let total_seats : ℕ := 24
  let prob_boy_girl : ℚ := (num_boys : ℚ) * num_girls / (total_seats * (total_seats - 1))
  let prob_girl_boy : ℚ := (num_girls : ℚ) * num_boys / (total_seats * (total_seats - 1))
  let prob_adjacent_pair : ℚ := prob_boy_girl + prob_girl_boy
  let expected_pairs : ℚ := (total_seats : ℚ) * prob_adjacent_pair
  expected_pairs = 280 / 23 :=
by sorry

end expected_boy_girl_pairs_l1179_117901


namespace min_stamps_theorem_l1179_117953

/-- The minimum number of stamps needed to make 48 cents using only 5 cent and 7 cent stamps -/
def min_stamps : ℕ := 8

/-- The value of stamps in cents -/
def total_value : ℕ := 48

/-- Represents a combination of 5 cent and 7 cent stamps -/
structure StampCombination where
  five_cent : ℕ
  seven_cent : ℕ

/-- Calculates the total value of a stamp combination -/
def combination_value (c : StampCombination) : ℕ :=
  5 * c.five_cent + 7 * c.seven_cent

/-- Calculates the total number of stamps in a combination -/
def total_stamps (c : StampCombination) : ℕ :=
  c.five_cent + c.seven_cent

/-- Predicate for a valid stamp combination that sums to the total value -/
def is_valid_combination (c : StampCombination) : Prop :=
  combination_value c = total_value

theorem min_stamps_theorem :
  ∃ (c : StampCombination), is_valid_combination c ∧
  (∀ (d : StampCombination), is_valid_combination d → total_stamps c ≤ total_stamps d) ∧
  total_stamps c = min_stamps :=
sorry

end min_stamps_theorem_l1179_117953


namespace school_girls_count_l1179_117982

theorem school_girls_count (total_students : ℕ) (boys_girls_difference : ℕ) 
  (h1 : total_students = 1250)
  (h2 : boys_girls_difference = 124) : 
  ∃ (girls : ℕ), girls = 563 ∧ 
  girls + (girls + boys_girls_difference) = total_students :=
sorry

end school_girls_count_l1179_117982


namespace sum_of_ratios_l1179_117919

theorem sum_of_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x/y + y/z + z/x = Real.sqrt ((x/y)^2 + (y/z)^2 + (z/x)^2 + 3) := by
  sorry

end sum_of_ratios_l1179_117919


namespace olivia_napkins_l1179_117932

theorem olivia_napkins (initial_napkins final_napkins : ℕ) 
  (h1 : initial_napkins = 15)
  (h2 : final_napkins = 45)
  (h3 : ∃ (o : ℕ), final_napkins = initial_napkins + o + 2*o) :
  ∃ (o : ℕ), o = 10 ∧ final_napkins = initial_napkins + o + 2*o :=
by sorry

end olivia_napkins_l1179_117932


namespace milk_for_cookies_l1179_117925

/-- Given that 18 cookies require 3 quarts of milk, 1 quart equals 2 pints,
    prove that 9 cookies require 3 pints of milk. -/
theorem milk_for_cookies (cookies_large : ℕ) (milk_quarts : ℕ) (cookies_small : ℕ) :
  cookies_large = 18 →
  milk_quarts = 3 →
  cookies_small = 9 →
  cookies_small * 2 = cookies_large →
  ∃ (milk_pints : ℕ),
    milk_pints = milk_quarts * 2 ∧
    milk_pints / 2 = 3 :=
by sorry

end milk_for_cookies_l1179_117925


namespace percent_of_300_l1179_117963

theorem percent_of_300 : (22 : ℝ) / 100 * 300 = 66 := by sorry

end percent_of_300_l1179_117963


namespace kevins_calculation_l1179_117985

theorem kevins_calculation (k : ℝ) : 
  (20 + 1) * (6 + k) = 20 + 1 * 6 + k → 20 + 1 * 6 + k = 21 :=
by
  sorry

end kevins_calculation_l1179_117985


namespace sum_first_six_primes_gt_10_l1179_117993

def first_six_primes_gt_10 : List Nat :=
  [11, 13, 17, 19, 23, 29]

theorem sum_first_six_primes_gt_10 :
  first_six_primes_gt_10.sum = 112 := by
  sorry

end sum_first_six_primes_gt_10_l1179_117993


namespace reciprocal_sum_l1179_117912

theorem reciprocal_sum (x y : ℝ) (h1 : x * y > 0) (h2 : 1 / (x * y) = 5) (h3 : (x + y) / 5 = 0.6) :
  1 / x + 1 / y = 15 := by
  sorry

end reciprocal_sum_l1179_117912


namespace willsons_work_hours_l1179_117929

theorem willsons_work_hours : 
  let monday : ℚ := 3/4
  let tuesday : ℚ := 1/2
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  monday + tuesday + wednesday + thursday + friday = 4 := by
  sorry

end willsons_work_hours_l1179_117929


namespace faye_halloween_candy_l1179_117996

/-- Represents the number of candy pieces Faye scored on Halloween. -/
def initial_candy : ℕ := 47

/-- Represents the number of candy pieces Faye ate on the first night. -/
def eaten_candy : ℕ := 25

/-- Represents the number of candy pieces Faye's sister gave her. -/
def received_candy : ℕ := 40

/-- Represents the number of candy pieces Faye has now. -/
def current_candy : ℕ := 62

theorem faye_halloween_candy : 
  initial_candy - eaten_candy + received_candy = current_candy := by
  sorry

end faye_halloween_candy_l1179_117996


namespace vector_addition_l1179_117943

def vector_AB : ℝ × ℝ := (1, 2)
def vector_BC : ℝ × ℝ := (3, 4)

theorem vector_addition :
  let vector_AC := (vector_AB.1 + vector_BC.1, vector_AB.2 + vector_BC.2)
  vector_AC = (4, 6) := by sorry

end vector_addition_l1179_117943


namespace parabola_properties_l1179_117911

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 3)^2 + 5

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  (∀ x : ℝ, parabola x ≥ 5) ∧ 
  (∀ x : ℝ, parabola (3 + x) = parabola (3 - x)) ∧
  (parabola 3 = 5) := by
  sorry


end parabola_properties_l1179_117911


namespace two_books_from_three_genres_l1179_117970

/-- The number of ways to select 2 books of different genres from 3 genres with 4 books each -/
def select_two_books (num_genres : ℕ) (books_per_genre : ℕ) : ℕ :=
  let total_books := num_genres * books_per_genre
  let books_in_other_genres := (num_genres - 1) * books_per_genre
  (total_books * books_in_other_genres) / 2

/-- Theorem stating that selecting 2 books of different genres from 3 genres with 4 books each results in 48 possibilities -/
theorem two_books_from_three_genres : 
  select_two_books 3 4 = 48 := by
  sorry

#eval select_two_books 3 4

end two_books_from_three_genres_l1179_117970


namespace rectangular_prism_volume_l1179_117976

theorem rectangular_prism_volume 
  (top_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ) 
  (h₁ : top_area = 20) 
  (h₂ : side_area = 15) 
  (h₃ : front_area = 12) : 
  ∃ (x y z : ℝ), 
    x * y = top_area ∧ 
    y * z = side_area ∧ 
    x * z = front_area ∧ 
    x * y * z = 60 := by
  sorry

end rectangular_prism_volume_l1179_117976


namespace speed_conversion_l1179_117967

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph_factor : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed_mps : ℝ := 45

/-- Theorem: Converting 45 meters per second to kilometers per hour results in 162 km/h -/
theorem speed_conversion :
  given_speed_mps * mps_to_kmph_factor = 162 := by sorry

end speed_conversion_l1179_117967


namespace fifteen_people_on_boats_l1179_117936

/-- Given a lake with boats and people, calculate the total number of people on the boats. -/
def total_people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) : ℕ :=
  num_boats * people_per_boat

/-- Theorem: In a lake with 5 boats and 3 people per boat, there are 15 people on boats in total. -/
theorem fifteen_people_on_boats :
  total_people_on_boats 5 3 = 15 := by
  sorry

end fifteen_people_on_boats_l1179_117936


namespace sin_330_degrees_l1179_117950

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l1179_117950


namespace a_is_zero_l1179_117983

/-- If a and b are natural numbers such that for every natural number n, 
    2^n * a + b is a perfect square, then a = 0. -/
theorem a_is_zero (a b : ℕ) 
    (h : ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2) : 
  a = 0 := by
  sorry

end a_is_zero_l1179_117983


namespace complex_magnitude_difference_zero_l1179_117988

theorem complex_magnitude_difference_zero : Complex.abs (3 - 5*Complex.I) - Complex.abs (3 + 5*Complex.I) = 0 := by
  sorry

end complex_magnitude_difference_zero_l1179_117988


namespace train_exit_time_l1179_117926

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Represents a train passing through a tunnel -/
structure TrainPassage where
  trainSpeed : Real  -- in km/h
  trainLength : Real  -- in km
  tunnelLength : Real  -- in km
  entryTime : Time

/-- Calculates the exit time of a train passing through a tunnel -/
def calculateExitTime (passage : TrainPassage) : Time :=
  sorry  -- Proof omitted

/-- Theorem stating that the given train leaves the tunnel at 6:05:15 am -/
theorem train_exit_time (passage : TrainPassage) 
  (h1 : passage.trainSpeed = 80)
  (h2 : passage.trainLength = 1)
  (h3 : passage.tunnelLength = 70)
  (h4 : passage.entryTime = ⟨5, 12, 0⟩) :
  calculateExitTime passage = ⟨6, 5, 15⟩ :=
by sorry

end train_exit_time_l1179_117926


namespace no_solution_to_system_l1179_117906

theorem no_solution_to_system :
  ¬∃ (x : ℝ), 
    (|Real.log x / Real.log 2| + (4 * x^2 / 15) - (16/15) = 0) ∧ 
    (Real.log (x + 2/3) / Real.log 7 + 12*x - 5 = 0) := by
  sorry

end no_solution_to_system_l1179_117906


namespace not_cheap_necessary_for_good_quality_l1179_117960

-- Define the propositions
variable (cheap : Prop) (good_quality : Prop)

-- Define the given condition
axiom cheap_implies_not_good : cheap → ¬good_quality

-- Theorem to prove
theorem not_cheap_necessary_for_good_quality :
  good_quality → ¬cheap :=
sorry

end not_cheap_necessary_for_good_quality_l1179_117960


namespace tree_spacing_l1179_117999

/-- Given a yard of length 300 meters with 26 equally spaced trees, including one at each end,
    the distance between consecutive trees is 12 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) : 
  yard_length = 300 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 12 := by
  sorry

end tree_spacing_l1179_117999
