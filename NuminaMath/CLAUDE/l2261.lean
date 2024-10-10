import Mathlib

namespace part_one_part_two_l2261_226125

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one : 
  {x : ℝ | |x - 1| ≥ |x + 1| + 1} = {x : ℝ | x ≤ -0.5} := by sorry

-- Part II
theorem part_two :
  {a : ℝ | ∀ x ≤ -1, f a x + 3 * x ≤ 0} = {a : ℝ | -4 ≤ a ∧ a ≤ 2} := by sorry

end part_one_part_two_l2261_226125


namespace cone_lateral_area_l2261_226121

/-- The lateral area of a cone with height 3 and slant height 5 is 20π. -/
theorem cone_lateral_area (h : ℝ) (s : ℝ) (r : ℝ) :
  h = 3 →
  s = 5 →
  r^2 + h^2 = s^2 →
  (1/2 : ℝ) * (2 * π * r) * s = 20 * π :=
by sorry

end cone_lateral_area_l2261_226121


namespace last_remaining_number_l2261_226160

/-- Represents the marking process on a list of numbers -/
def MarkingProcess (n : ℕ) (skip : ℕ) (l : List ℕ) : List ℕ :=
  sorry

/-- Represents a single pass of the marking process -/
def SinglePass (n : ℕ) (skip : ℕ) (l : List ℕ) : List ℕ :=
  sorry

/-- Represents the entire process of marking and skipping -/
def FullProcess (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the last remaining number is 21 -/
theorem last_remaining_number : FullProcess 50 = 21 :=
  sorry

end last_remaining_number_l2261_226160


namespace sin_sum_to_product_l2261_226156

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_to_product_l2261_226156


namespace congruence_solutions_count_l2261_226170

theorem congruence_solutions_count :
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x > 0 ∧ x < 150 ∧ (x + 15) % 45 = 75 % 45) ∧
    (∀ x, x > 0 → x < 150 → (x + 15) % 45 = 75 % 45 → x ∈ S) ∧
    Finset.card S = 3 := by
  sorry

end congruence_solutions_count_l2261_226170


namespace max_value_of_c_l2261_226138

theorem max_value_of_c (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : 2 * a * b = 2 * a + b) (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 := by
sorry

end max_value_of_c_l2261_226138


namespace average_increase_is_five_l2261_226151

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ

/-- Calculates the average runs per inning -/
def average (bp : BatsmanPerformance) : ℚ :=
  bp.totalRuns / bp.innings

/-- Theorem: The increase in average is 5 runs -/
theorem average_increase_is_five (bp : BatsmanPerformance) 
  (h1 : bp.innings = 11)
  (h2 : bp.lastInningRuns = 85)
  (h3 : average bp = 35) :
  average bp - average { bp with 
    innings := bp.innings - 1,
    totalRuns := bp.totalRuns - bp.lastInningRuns
  } = 5 := by
  sorry

#check average_increase_is_five

end average_increase_is_five_l2261_226151


namespace current_speed_l2261_226171

/-- The speed of the current given boat speeds upstream and downstream -/
theorem current_speed (upstream_speed downstream_speed : ℝ) : 
  upstream_speed = 1 / (20 / 60) →
  downstream_speed = 1 / (15 / 60) →
  (downstream_speed - upstream_speed) / 2 = 0.5 := by
  sorry

end current_speed_l2261_226171


namespace min_side_length_l2261_226137

theorem min_side_length (AB EC AC BE : ℝ) (hAB : AB = 7) (hEC : EC = 10) (hAC : AC = 15) (hBE : BE = 25) :
  ∃ (BC : ℕ), BC ≥ 15 ∧ ∀ (BC' : ℕ), (BC' ≥ 15 → BC' ≥ BC) :=
by sorry

end min_side_length_l2261_226137


namespace time_ratio_third_to_first_l2261_226101

-- Define the distances and speed ratios
def distance_first : ℝ := 60
def distance_second : ℝ := 240
def distance_third : ℝ := 180
def speed_ratio_second : ℝ := 4
def speed_ratio_third : ℝ := 2

-- Define the theorem
theorem time_ratio_third_to_first :
  let time_first := distance_first / (distance_first / time_first)
  let time_third := distance_third / (speed_ratio_third * (distance_first / time_first))
  time_third / time_first = 1.5 := by
  sorry

end time_ratio_third_to_first_l2261_226101


namespace completing_square_transform_l2261_226150

theorem completing_square_transform (x : ℝ) :
  (2 * x^2 - 4 * x - 3 = 0) ↔ ((x - 1)^2 - 5/2 = 0) :=
by sorry

end completing_square_transform_l2261_226150


namespace fourth_root_of_four_powers_l2261_226182

theorem fourth_root_of_four_powers : (4^7 + 4^7 + 4^7 + 4^7 : ℝ)^(1/4) = 16 := by
  sorry

end fourth_root_of_four_powers_l2261_226182


namespace quadratic_inequality_solution_set_l2261_226197

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 2*x + 3 > 0} = Set.Ioo (-1 : ℝ) 3 := by sorry

end quadratic_inequality_solution_set_l2261_226197


namespace sum_b_c_is_48_l2261_226192

/-- An arithmetic sequence with six terms -/
structure ArithmeticSequence :=
  (a₁ : ℝ)
  (a₂ : ℝ)
  (a₃ : ℝ)
  (a₄ : ℝ)
  (a₅ : ℝ)
  (a₆ : ℝ)
  (is_arithmetic : ∃ d : ℝ, a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d ∧ a₅ - a₄ = d ∧ a₆ - a₅ = d)

/-- The sum of the third and fifth terms in the specific arithmetic sequence -/
def sum_b_c (seq : ArithmeticSequence) : ℝ := seq.a₃ + seq.a₅

/-- Theorem stating that for the given arithmetic sequence, the sum of b and c is 48 -/
theorem sum_b_c_is_48 (seq : ArithmeticSequence) 
  (h₁ : seq.a₁ = 3)
  (h₂ : seq.a₂ = 10)
  (h₃ : seq.a₄ = 24)
  (h₄ : seq.a₆ = 38) :
  sum_b_c seq = 48 := by
  sorry

end sum_b_c_is_48_l2261_226192


namespace smallest_S_value_l2261_226145

def is_valid_arrangement (a b c d : Fin 4 → ℕ) : Prop :=
  ∀ i : Fin 16, ∃! j : Fin 4, ∃! k : Fin 4,
    i.val + 1 = a j ∨ i.val + 1 = b j ∨ i.val + 1 = c j ∨ i.val + 1 = d j

def S (a b c d : Fin 4 → ℕ) : ℕ :=
  (a 0) * (a 1) * (a 2) * (a 3) +
  (b 0) * (b 1) * (b 2) * (b 3) +
  (c 0) * (c 1) * (c 2) * (c 3) +
  (d 0) * (d 1) * (d 2) * (d 3)

theorem smallest_S_value :
  ∀ a b c d : Fin 4 → ℕ, is_valid_arrangement a b c d → S a b c d ≥ 2074 :=
by sorry

end smallest_S_value_l2261_226145


namespace january_salary_l2261_226133

/-- Given the average salaries for two sets of four months and the salary for May,
    prove that the salary for January is 4100. -/
theorem january_salary (jan feb mar apr may : ℕ)
  (h1 : (jan + feb + mar + apr) / 4 = 8000)
  (h2 : (feb + mar + apr + may) / 4 = 8600)
  (h3 : may = 6500) :
  jan = 4100 := by
  sorry

end january_salary_l2261_226133


namespace fourth_number_in_list_l2261_226178

theorem fourth_number_in_list (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 48, 507, 684, 42] →
  average = 223 →
  ∃ x : ℕ, (List.sum known_numbers + x) / 6 = average ∧ x = 2 :=
by sorry

end fourth_number_in_list_l2261_226178


namespace quadratic_root_condition_l2261_226191

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (3 * x₁^2 - 4*(3*a-2)*x₁ + a^2 + 2*a = 0) ∧ 
    (3 * x₂^2 - 4*(3*a-2)*x₂ + a^2 + 2*a = 0) ∧ 
    (x₁ < a ∧ a < x₂)) 
  ↔ 
  (a < 0 ∨ a > 5/4) :=
by sorry

end quadratic_root_condition_l2261_226191


namespace lcm_of_numbers_in_ratio_l2261_226152

def are_in_ratio (a b c : ℕ) (x y z : ℕ) : Prop :=
  ∃ (k : ℕ), a = k * x ∧ b = k * y ∧ c = k * z

theorem lcm_of_numbers_in_ratio (a b c : ℕ) 
  (h_ratio : are_in_ratio a b c 5 7 9)
  (h_hcf : Nat.gcd a (Nat.gcd b c) = 11) :
  Nat.lcm a (Nat.lcm b c) = 99 := by
  sorry

end lcm_of_numbers_in_ratio_l2261_226152


namespace probability_b_draws_red_l2261_226157

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

theorem probability_b_draws_red :
  let prob_b_red : ℚ := 
    (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) +
    (white_balls : ℚ) / total_balls * (red_balls : ℚ) / (total_balls - 1)
  prob_b_red = 2 / 5 :=
by sorry

end probability_b_draws_red_l2261_226157


namespace percentage_increase_problem_l2261_226143

theorem percentage_increase_problem (a b x m : ℝ) (k : ℝ) (h1 : a > 0) (h2 : b > 0) :
  a = 4 * k ∧ b = 5 * k ∧ k > 0 →
  (∃ p : ℝ, x = a * (1 + p / 100)) →
  m = b * 0.4 →
  m / x = 0.4 →
  ∃ p : ℝ, x = a * (1 + p / 100) ∧ p = 25 := by
sorry

end percentage_increase_problem_l2261_226143


namespace junior_fraction_l2261_226147

theorem junior_fraction (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h3 : J * 3 = S * 4) :
  J / (J + S) = 4 / 7 :=
by sorry

end junior_fraction_l2261_226147


namespace max_potential_salary_is_440000_l2261_226194

/-- Represents a soccer team with its payroll constraints -/
structure SoccerTeam where
  numPlayers : ℕ
  minSalary : ℕ
  maxPayroll : ℕ

/-- Calculates the maximum potential salary for an individual player on a team -/
def maxPotentialSalary (team : SoccerTeam) : ℕ :=
  team.maxPayroll - (team.numPlayers - 1) * team.minSalary

/-- Theorem stating the maximum potential salary for an individual player -/
theorem max_potential_salary_is_440000 :
  let team : SoccerTeam := ⟨19, 20000, 800000⟩
  maxPotentialSalary team = 440000 := by
  sorry

#eval maxPotentialSalary ⟨19, 20000, 800000⟩

end max_potential_salary_is_440000_l2261_226194


namespace square_prime_equivalence_l2261_226106

theorem square_prime_equivalence (N : ℕ) (h : N ≥ 2) :
  (∀ n : ℕ, n < N → ¬∃ s : ℕ, 4*n*(N-n)+1 = s^2) ↔ Nat.Prime (N^2 + 1) :=
by sorry

end square_prime_equivalence_l2261_226106


namespace expression_bounds_l2261_226144

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + 
    Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + 
    Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by sorry

end expression_bounds_l2261_226144


namespace exp_two_ln_two_equals_four_l2261_226131

theorem exp_two_ln_two_equals_four : Real.exp (2 * Real.log 2) = 4 := by
  sorry

end exp_two_ln_two_equals_four_l2261_226131


namespace uber_cost_is_22_l2261_226166

/-- The cost of a taxi ride --/
def taxi_cost : ℝ := 15

/-- The cost of a Lyft ride --/
def lyft_cost : ℝ := taxi_cost + 4

/-- The cost of an Uber ride --/
def uber_cost : ℝ := lyft_cost + 3

/-- The total cost of a taxi ride including a 20% tip --/
def taxi_total_cost : ℝ := taxi_cost * 1.2

theorem uber_cost_is_22 :
  (taxi_total_cost = 18) →
  (uber_cost = 22) :=
by
  sorry

#eval uber_cost

end uber_cost_is_22_l2261_226166


namespace lcm_of_150_and_490_l2261_226193

theorem lcm_of_150_and_490 : Nat.lcm 150 490 = 7350 := by
  sorry

end lcm_of_150_and_490_l2261_226193


namespace flag_designs_count_l2261_226169

/-- The number of colors available for the flag design. -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag. -/
def num_stripes : ℕ := 3

/-- Calculate the number of possible flag designs. -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the number of possible flag designs is 27. -/
theorem flag_designs_count : num_flag_designs = 27 := by
  sorry

end flag_designs_count_l2261_226169


namespace adam_shelf_count_l2261_226195

/-- The number of shelves in Adam's room -/
def num_shelves : ℕ := sorry

/-- The number of action figures that can fit on each shelf -/
def figures_per_shelf : ℕ := 9

/-- The total number of action figures that can be held by all shelves -/
def total_figures : ℕ := 27

/-- Theorem stating that the number of shelves is 3 -/
theorem adam_shelf_count : num_shelves = 3 := by sorry

end adam_shelf_count_l2261_226195


namespace oil_leak_calculation_l2261_226123

/-- The amount of oil leaked before engineers started fixing the pipe -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked while engineers were working -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem oil_leak_calculation :
  total_oil_leaked = 11687 := by
  sorry

end oil_leak_calculation_l2261_226123


namespace four_friends_same_group_probability_l2261_226124

/-- The total number of students -/
def total_students : ℕ := 900

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The number of students in each group -/
def students_per_group : ℕ := total_students / num_groups

/-- The probability of a single student being assigned to a specific group -/
def prob_single_student : ℚ := 1 / num_groups

theorem four_friends_same_group_probability :
  (prob_single_student ^ 3 : ℚ) = 1 / 64 :=
sorry

end four_friends_same_group_probability_l2261_226124


namespace sum_of_roots_quadratic_l2261_226189

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 = 2*x₁ + 1) → (x₂^2 = 2*x₂ + 1) → x₁ + x₂ = 2 := by
  sorry

end sum_of_roots_quadratic_l2261_226189


namespace inverse_variation_problem_l2261_226174

-- Define the inverse relationship between two quantities
def inverse_relation (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Define the given conditions
def conditions : Prop :=
  ∃ (k m : ℝ),
    inverse_relation 1500 0.4 k ∧
    inverse_relation 1500 2.5 m ∧
    inverse_relation 3000 0.2 k ∧
    inverse_relation 3000 1.25 m

-- State the theorem
theorem inverse_variation_problem :
  conditions → (∃ (s t : ℝ), s = 0.2 ∧ t = 1.25) :=
by
  sorry

end inverse_variation_problem_l2261_226174


namespace four_positive_integers_sum_l2261_226196

theorem four_positive_integers_sum (a b c d : ℕ+) 
  (sum1 : a + b + c = 6)
  (sum2 : a + b + d = 7)
  (sum3 : a + c + d = 8)
  (sum4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
  sorry

end four_positive_integers_sum_l2261_226196


namespace stating_max_wickets_theorem_l2261_226140

/-- Represents the maximum number of wickets a bowler can take in one over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the number of overs bowled by the bowler in an innings -/
def overs_bowled : ℕ := 6

/-- Represents the number of players in a cricket team -/
def players_per_team : ℕ := 11

/-- Represents the maximum number of wickets that can be taken in an innings -/
def max_wickets_in_innings : ℕ := players_per_team - 1

/-- 
Theorem stating that the maximum number of wickets a bowler can take in an innings
is the minimum of the theoretical maximum (max_wickets_per_over * overs_bowled) 
and the actual maximum (max_wickets_in_innings)
-/
theorem max_wickets_theorem : 
  min (max_wickets_per_over * overs_bowled) max_wickets_in_innings = max_wickets_in_innings := by
  sorry

end stating_max_wickets_theorem_l2261_226140


namespace officer_election_proof_l2261_226159

def total_candidates : ℕ := 18
def past_officers : ℕ := 8
def positions_available : ℕ := 6

theorem officer_election_proof :
  (Nat.choose total_candidates positions_available) -
  (Nat.choose (total_candidates - past_officers) positions_available) -
  (Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions_available - 1)) = 16338 := by
  sorry

end officer_election_proof_l2261_226159


namespace unique_number_between_9_and_9_1_cube_root_l2261_226104

theorem unique_number_between_9_and_9_1_cube_root (n : ℕ+) : 
  (∃ k : ℕ, n = 21 * k) ∧ 
  (9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.1) ↔ 
  n = 735 :=
sorry

end unique_number_between_9_and_9_1_cube_root_l2261_226104


namespace quadratic_equation_problem_l2261_226135

theorem quadratic_equation_problem (m : ℤ) (a : ℝ) 
  (h1 : ∃ x y : ℝ, x ≠ y ∧ (m^2 - m) * x^2 - 2*m*x + 1 = 0 ∧ (m^2 - m) * y^2 - 2*m*y + 1 = 0)
  (h2 : m < 3)
  (h3 : (m^2 - m) * a^2 - 2*m*a + 1 = 0) :
  m = 2 ∧ (2*a^2 - 3*a - 3 = (-6 + Real.sqrt 2) / 2 ∨ 2*a^2 - 3*a - 3 = (-6 - Real.sqrt 2) / 2) := by
  sorry

end quadratic_equation_problem_l2261_226135


namespace leah_lost_money_proof_l2261_226179

def leah_lost_money (initial_earnings : ℝ) (milkshake_fraction : ℝ) (comic_book_fraction : ℝ) (savings_fraction : ℝ) (not_shredded_fraction : ℝ) : ℝ :=
  let remaining_after_milkshake := initial_earnings - milkshake_fraction * initial_earnings
  let remaining_after_comic := remaining_after_milkshake - comic_book_fraction * remaining_after_milkshake
  let remaining_after_savings := remaining_after_comic - savings_fraction * remaining_after_comic
  let not_shredded := not_shredded_fraction * remaining_after_savings
  remaining_after_savings - not_shredded

theorem leah_lost_money_proof :
  leah_lost_money 28 (1/7) (1/5) (3/8) 0.1 = 10.80 := by
  sorry

end leah_lost_money_proof_l2261_226179


namespace find_number_l2261_226198

theorem find_number : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end find_number_l2261_226198


namespace student_calculation_l2261_226175

theorem student_calculation (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 := by
  sorry

end student_calculation_l2261_226175


namespace square_area_74_l2261_226187

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the square
def Square (p q : Point) :=
  {s : Set Point | ∃ (a b : ℝ), s = {(x, y) | min p.1 q.1 ≤ x ∧ x ≤ max p.1 q.1 ∧ min p.2 q.2 ≤ y ∧ y ≤ max p.2 q.2}}

-- Calculate the area of the square
def area (p q : Point) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem square_area_74 :
  let p : Point := (-2, -1)
  let q : Point := (3, 6)
  area p q = 74 := by
sorry

end square_area_74_l2261_226187


namespace geometric_sequence_properties_l2261_226112

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_properties 
  (a : ℕ → ℚ) 
  (h_geometric : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end geometric_sequence_properties_l2261_226112


namespace prob_two_qualified_bottles_l2261_226199

/-- The probability of a single bottle of beverage being qualified -/
def qualified_rate : ℝ := 0.8

/-- The probability of two bottles both being qualified -/
def both_qualified_prob : ℝ := qualified_rate * qualified_rate

/-- Theorem: The probability of drinking two qualified bottles is 0.64 -/
theorem prob_two_qualified_bottles : both_qualified_prob = 0.64 := by sorry

end prob_two_qualified_bottles_l2261_226199


namespace choir_average_age_l2261_226154

theorem choir_average_age (female_count : ℕ) (male_count : ℕ) (children_count : ℕ)
  (female_avg_age : ℝ) (male_avg_age : ℝ) (children_avg_age : ℝ)
  (h_female_count : female_count = 12)
  (h_male_count : male_count = 18)
  (h_children_count : children_count = 10)
  (h_female_avg : female_avg_age = 28)
  (h_male_avg : male_avg_age = 36)
  (h_children_avg : children_avg_age = 10) :
  let total_count := female_count + male_count + children_count
  let total_age := female_count * female_avg_age + male_count * male_avg_age + children_count * children_avg_age
  total_age / total_count = 27.1 := by
  sorry

end choir_average_age_l2261_226154


namespace arcade_spending_equals_allowance_l2261_226113

def dress_cost : ℕ := 80
def initial_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weeks_to_save : ℕ := 3

theorem arcade_spending_equals_allowance :
  ∃ (arcade_spending : ℕ),
    arcade_spending = weekly_allowance ∧
    initial_savings + weeks_to_save * weekly_allowance - weeks_to_save * arcade_spending = dress_cost :=
by sorry

end arcade_spending_equals_allowance_l2261_226113


namespace motorcycle_price_is_correct_l2261_226163

/-- Represents the factory's production and profit information -/
structure FactoryInfo where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycles_produced : ℕ
  profit_difference : ℕ

/-- Calculates the price per motorcycle based on the given factory information -/
def calculate_motorcycle_price (info : FactoryInfo) : ℕ :=
  (info.profit_difference + (info.car_material_cost + info.motorcycle_material_cost - info.cars_produced * info.car_price) + info.motorcycle_material_cost) / info.motorcycles_produced

/-- Theorem stating that the calculated motorcycle price is correct -/
theorem motorcycle_price_is_correct (info : FactoryInfo) 
  (h1 : info.car_material_cost = 100)
  (h2 : info.cars_produced = 4)
  (h3 : info.car_price = 50)
  (h4 : info.motorcycle_material_cost = 250)
  (h5 : info.motorcycles_produced = 8)
  (h6 : info.profit_difference = 50) :
  calculate_motorcycle_price info = 50 := by
  sorry

end motorcycle_price_is_correct_l2261_226163


namespace exponentiation_puzzle_l2261_226111

theorem exponentiation_puzzle : 3^(1^(0^2)) - ((3^1)^0)^2 = 2 := by
  sorry

end exponentiation_puzzle_l2261_226111


namespace least_number_remainder_l2261_226176

theorem least_number_remainder : ∃ (r : ℕ), r > 0 ∧ r < 3 ∧ r < 38 ∧ 115 % 38 = r ∧ 115 % 3 = r := by
  sorry

end least_number_remainder_l2261_226176


namespace vector_sum_magnitude_l2261_226177

/-- Given two plane vectors a and b, with the angle between them being 60°,
    a = (2,0), and |b| = 1, prove that |a + 2b| = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60° in radians
  a = (2, 0) ∧ 
  ‖b‖ = 1 ∧ 
  a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * Real.cos angle →
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l2261_226177


namespace max_points_theorem_l2261_226183

/-- Represents a football tournament with the given conditions -/
structure Tournament where
  teams : Nat
  total_points : Nat
  draw_points : Nat
  win_points : Nat

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : Nat :=
  t.teams * (t.teams - 1) / 2

/-- Represents the result of solving the tournament equations -/
structure TournamentResult where
  draws : Nat
  wins : Nat

/-- Solves the tournament equations to find the number of draws and wins -/
def solve_tournament (t : Tournament) : TournamentResult :=
  { draws := 23, wins := 5 }

/-- Calculates the maximum points a single team can obtain -/
def max_points (t : Tournament) (result : TournamentResult) : Nat :=
  (result.wins * t.win_points) + (t.teams - 1 - result.wins) * t.draw_points

/-- The main theorem stating the maximum points obtainable by a single team -/
theorem max_points_theorem (t : Tournament) 
  (h1 : t.teams = 8)
  (h2 : t.total_points = 61)
  (h3 : t.draw_points = 1)
  (h4 : t.win_points = 3) :
  max_points t (solve_tournament t) = 17 := by
  sorry

#eval max_points 
  { teams := 8, total_points := 61, draw_points := 1, win_points := 3 } 
  (solve_tournament { teams := 8, total_points := 61, draw_points := 1, win_points := 3 })

end max_points_theorem_l2261_226183


namespace brownie_pieces_count_l2261_226172

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the tray of brownies -/
def tray : Dimensions := ⟨24, 20⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the tray can be divided into exactly 80 pieces -/
theorem brownie_pieces_count : (area tray) / (area piece) = 80 := by
  sorry

end brownie_pieces_count_l2261_226172


namespace sum_of_roots_quadratic_l2261_226185

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ α β : ℝ, (α + β = 6) ∧ (α * β = 8) ∧ (α ≠ β → (α - β)^2 = 36 - 4*8)) :=
by sorry

end sum_of_roots_quadratic_l2261_226185


namespace roots_expression_value_l2261_226119

theorem roots_expression_value (m n : ℝ) : 
  m^2 + 2*m - 2027 = 0 → 
  n^2 + 2*n - 2027 = 0 → 
  2*m - m*n + 2*n = 2023 := by
sorry

end roots_expression_value_l2261_226119


namespace compare_sizes_l2261_226180

theorem compare_sizes (a b : ℝ) (ha : a = 0.2^(1/2)) (hb : b = 0.5^(1/5)) :
  0 < a ∧ a < b ∧ b < 1 := by sorry

end compare_sizes_l2261_226180


namespace max_consecutive_sum_30_l2261_226148

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive integers starting from 2 -/
def sum_from_2 (n : ℕ) : ℕ := sum_first_n (n + 1) - 1

/-- 30 is the maximum number of consecutive positive integers 
    starting from 2 that can be added together without exceeding 500 -/
theorem max_consecutive_sum_30 :
  (∀ k : ℕ, k ≤ 30 → sum_from_2 k ≤ 500) ∧
  (∀ k : ℕ, k > 30 → sum_from_2 k > 500) :=
sorry

end max_consecutive_sum_30_l2261_226148


namespace tank_capacities_l2261_226161

/-- Given three tanks with capacities T1, T2, and T3, prove that the total amount of water is 10850 gallons. -/
theorem tank_capacities (T1 T2 T3 : ℝ) : 
  (3/4 : ℝ) * T1 + (4/5 : ℝ) * T2 + (1/2 : ℝ) * T3 = 10850 := by
  sorry

#check tank_capacities

end tank_capacities_l2261_226161


namespace union_A_B_intersection_complement_A_B_l2261_226118

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2*x - 1 ∧ 2*x - 1 < 19}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

-- Theorem for (CₙA) ∩ B
theorem intersection_complement_A_B : (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

end union_A_B_intersection_complement_A_B_l2261_226118


namespace g_50_eq_zero_l2261_226158

-- Define φ(n) as the number of positive integers not exceeding n that are coprime to n
def phi (n : ℕ) : ℕ := sorry

-- Define g(n) to satisfy the condition that for any positive integer n, 
-- the sum of g(d) over all positive divisors d of n equals φ(n)
def g (n : ℕ) : ℤ :=
  sorry

-- Theorem to prove
theorem g_50_eq_zero : g 50 = 0 := by
  sorry

end g_50_eq_zero_l2261_226158


namespace alteredLucas_53_mod_5_l2261_226114

def alteredLucas : ℕ → ℕ
  | 0 => 1
  | 1 => 4
  | n + 2 => alteredLucas n + alteredLucas (n + 1)

theorem alteredLucas_53_mod_5 : alteredLucas 52 % 5 = 0 := by
  sorry

end alteredLucas_53_mod_5_l2261_226114


namespace tv_production_average_l2261_226100

theorem tv_production_average (total_days : ℕ) (first_period : ℕ) (first_avg : ℝ) (total_avg : ℝ) :
  total_days = 30 →
  first_period = 25 →
  first_avg = 50 →
  total_avg = 45 →
  (total_days * total_avg - first_period * first_avg) / (total_days - first_period) = 20 := by
sorry

end tv_production_average_l2261_226100


namespace smallest_non_prime_non_square_l2261_226165

def is_prime (n : ℕ) : Prop := sorry

def is_square (n : ℕ) : Prop := sorry

def has_prime_factor_less_than (n k : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square : 
  ∀ n : ℕ, n < 3599 → 
    (is_prime n ∨ is_square n ∨ has_prime_factor_less_than n 55) ∧
    (¬ is_prime 3599 ∧ ¬ is_square 3599 ∧ ¬ has_prime_factor_less_than 3599 55) :=
by sorry

end smallest_non_prime_non_square_l2261_226165


namespace triangle_area_fraction_l2261_226107

/-- The area of a triangle given its three vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

/-- The theorem stating that the area of the given triangle divided by the area of the grid equals 13/96 -/
theorem triangle_area_fraction :
  let a_x := 2
  let a_y := 4
  let b_x := 7
  let b_y := 2
  let c_x := 6
  let c_y := 5
  let grid_width := 8
  let grid_height := 6
  (triangleArea a_x a_y b_x b_y c_x c_y) / (grid_width * grid_height) = 13 / 96 := by
  sorry


end triangle_area_fraction_l2261_226107


namespace multiplication_problem_solution_l2261_226130

theorem multiplication_problem_solution :
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    1000 ≤ a * b ∧ a * b < 10000 ∧
    10 ≤ a * 8 ∧ a * 8 < 100 ∧
    100 ≤ a * 9 ∧ a * 9 < 1000 ∧
    a * b = 1068 := by
  sorry

end multiplication_problem_solution_l2261_226130


namespace lcm_problem_l2261_226162

theorem lcm_problem (n : ℕ+) 
  (h1 : Nat.lcm 40 n = 120) 
  (h2 : Nat.lcm n 45 = 180) : 
  n = 12 := by
  sorry

end lcm_problem_l2261_226162


namespace quadratic_inequality_implies_a_range_l2261_226132

theorem quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → x^2 + 2*a*x + 1 ≥ 0) → a ≥ -1 := by
  sorry

end quadratic_inequality_implies_a_range_l2261_226132


namespace scarves_per_box_l2261_226108

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_items : ℕ) : 
  num_boxes = 4 → 
  mittens_per_box = 6 → 
  total_items = 32 → 
  (total_items - num_boxes * mittens_per_box) / num_boxes = 2 := by
  sorry

end scarves_per_box_l2261_226108


namespace trigonometric_problem_l2261_226168

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (Real.pi / 4 + α) = 1 / 3)
  (h4 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  (Real.cos α = (Real.sqrt 2 + 4) / 6) ∧
  (Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9) := by
  sorry

end trigonometric_problem_l2261_226168


namespace assignment_count_proof_l2261_226190

/-- The number of ways to assign doctors and nurses to schools -/
def assignment_count : ℕ := 540

/-- The number of doctors -/
def num_doctors : ℕ := 3

/-- The number of nurses -/
def num_nurses : ℕ := 6

/-- The number of schools -/
def num_schools : ℕ := 3

/-- The number of doctors assigned to each school -/
def doctors_per_school : ℕ := 1

/-- The number of nurses assigned to each school -/
def nurses_per_school : ℕ := 2

theorem assignment_count_proof : 
  assignment_count = 
    (num_doctors.factorial * (num_nurses.factorial / (nurses_per_school.factorial ^ num_schools))) := by
  sorry

end assignment_count_proof_l2261_226190


namespace max_carlson_jars_l2261_226115

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamState where
  carlson_weight : ℕ  -- Total weight of Carlson's jars
  baby_weight : ℕ    -- Total weight of Baby's jars
  carlson_jars : ℕ   -- Number of Carlson's jars

/-- Conditions of the jam problem -/
def jam_conditions (state : JamState) : Prop :=
  state.carlson_weight = 13 * state.baby_weight ∧
  ∃ (lightest : ℕ), 
    lightest > 0 ∧
    (state.carlson_weight - lightest) = 8 * (state.baby_weight + lightest)

/-- The theorem to be proved -/
theorem max_carlson_jars : 
  ∀ (state : JamState), 
    jam_conditions state → 
    state.carlson_jars ≤ 23 :=
sorry

end max_carlson_jars_l2261_226115


namespace expression_equals_500_l2261_226127

theorem expression_equals_500 : 88 * 4 + 37 * 4 = 500 := by
  sorry

end expression_equals_500_l2261_226127


namespace calculate_walking_speed_l2261_226141

/-- Given two people walking towards each other, this theorem calculates the speed of one person given the total distance, the speed of the other person, and the distance traveled by the first person. -/
theorem calculate_walking_speed 
  (total_distance : ℝ) 
  (brad_speed : ℝ) 
  (maxwell_distance : ℝ) 
  (h1 : total_distance = 40) 
  (h2 : brad_speed = 5) 
  (h3 : maxwell_distance = 15) : 
  ∃ (maxwell_speed : ℝ), maxwell_speed = 3 := by
  sorry

#check calculate_walking_speed

end calculate_walking_speed_l2261_226141


namespace solution_range_l2261_226188

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop :=
  3 * x - (a * x + 1) / 2 < 4 * x / 3

-- State the theorem
theorem solution_range (a : ℝ) : 
  (inequality 3 a) → a > 3 :=
by
  sorry

end solution_range_l2261_226188


namespace inequality_implies_product_l2261_226181

theorem inequality_implies_product (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) 
  (h3 : 4 * Real.log x + 2 * Real.log y ≥ x^2 + 4*y - 4) : 
  x * y = Real.sqrt 2 / 2 := by
sorry

end inequality_implies_product_l2261_226181


namespace phillip_and_paula_numbers_l2261_226142

theorem phillip_and_paula_numbers (a b : ℚ) 
  (h1 : a = b + 12)
  (h2 : a^2 + b^2 = 169/2)
  (h3 : a^4 - b^4 = 5070) : 
  a + b = 5 := by
  sorry

end phillip_and_paula_numbers_l2261_226142


namespace xy_max_value_l2261_226134

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2 * y = 8) :
  x * y ≤ 8 := by
sorry

end xy_max_value_l2261_226134


namespace angle_CBO_is_20_degrees_l2261_226149

-- Define the triangle ABC
variable (A B C O : Point) (ABC : Triangle A B C)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem angle_CBO_is_20_degrees 
  (h1 : angle B A O = angle C A O)
  (h2 : angle C B O = angle A B O)
  (h3 : angle A C O = angle B C O)
  (h4 : angle A O C = 110)
  (h5 : ∀ P Q R : Point, angle P Q R + angle Q R P + angle R P Q = 180) :
  angle C B O = 20 := by sorry

end angle_CBO_is_20_degrees_l2261_226149


namespace parallelogram_bisecting_line_slope_l2261_226102

/-- A parallelogram with vertices at (8,35), (8,90), (25,125), and (25,70) -/
structure Parallelogram where
  v1 : ℝ × ℝ := (8, 35)
  v2 : ℝ × ℝ := (8, 90)
  v3 : ℝ × ℝ := (25, 125)
  v4 : ℝ × ℝ := (25, 70)

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- The line cuts the parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop := sorry

theorem parallelogram_bisecting_line_slope (p : Parallelogram) (l : Line) :
  cuts_into_congruent_polygons p l → l.slope = 25 / 4 := by
  sorry

end parallelogram_bisecting_line_slope_l2261_226102


namespace ivan_revival_time_l2261_226109

/-- Represents the scenario of Wolf, Ivan, and Raven --/
structure Scenario where
  distance : ℝ
  wolf_speed : ℝ
  water_needed : ℝ
  water_flow_rate : ℝ
  raven_speed : ℝ
  water_spill_rate : ℝ

/-- Checks if Ivan can be revived within the given time --/
def can_revive (s : Scenario) (time : ℝ) : Prop :=
  let wolf_travel_time := s.distance / s.wolf_speed
  let water_collect_time := s.water_needed / s.water_flow_rate
  let total_time := wolf_travel_time + water_collect_time
  let raven_travel_distance := s.distance / 2
  let raven_travel_time := raven_travel_distance / s.raven_speed
  let water_lost := raven_travel_time * s.water_spill_rate
  let water_remaining := s.water_needed - water_lost
  
  time ≥ total_time ∧ water_remaining > 0

/-- The main theorem to prove --/
theorem ivan_revival_time (s : Scenario) :
  s.distance = 20 ∧
  s.wolf_speed = 3 ∧
  s.water_needed = 1 ∧
  s.water_flow_rate = 0.5 ∧
  s.raven_speed = 6 ∧
  s.water_spill_rate = 0.25 →
  can_revive s 4 :=
by
  sorry


end ivan_revival_time_l2261_226109


namespace problem_solution_l2261_226103

def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

theorem problem_solution :
  (∀ x : ℝ, x ≤ -1 → f 2 x = 0 ↔ x = (-1 - Real.sqrt 3) / 2) ∧
  (∀ k : ℝ, (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
    -7/2 < k ∧ k < -1) ∧
  (∀ k x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 →
    1/x₁ + 1/x₂ < 4) :=
by sorry

end problem_solution_l2261_226103


namespace inequality_proof_l2261_226126

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) :=
by sorry

end inequality_proof_l2261_226126


namespace m_value_l2261_226120

def A (m : ℝ) : Set ℝ := {-1, 2, 2*m-1}
def B (m : ℝ) : Set ℝ := {2, m^2}

theorem m_value (m : ℝ) : B m ⊆ A m → m = 1 := by
  sorry

end m_value_l2261_226120


namespace smallest_n_congruence_l2261_226146

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, 0 < m → m < n → ¬(537 * m ≡ 1073 * m [ZMOD 30])) → 
  (537 * n ≡ 1073 * n [ZMOD 30]) → 
  n = 15 := by
sorry

end smallest_n_congruence_l2261_226146


namespace polygon_interior_angle_sum_not_polygon_interior_angle_sum_l2261_226153

theorem polygon_interior_angle_sum (n : ℕ) (sum : ℕ) : sum = (n - 2) * 180 → n ≥ 3 :=
by sorry

theorem not_polygon_interior_angle_sum : ¬ ∃ (n : ℕ), 800 = (n - 2) * 180 ∧ n ≥ 3 :=
by sorry

end polygon_interior_angle_sum_not_polygon_interior_angle_sum_l2261_226153


namespace norm_took_110_photos_l2261_226155

/-- The number of photos taken by each photographer --/
structure PhotoCount where
  lisa : ℕ
  mike : ℕ
  norm : ℕ

/-- The conditions of the problem --/
def satisfies_conditions (p : PhotoCount) : Prop :=
  p.lisa + p.mike = p.mike + p.norm - 60 ∧
  p.norm = 2 * p.lisa + 10

/-- The theorem stating that Norm took 110 photos --/
theorem norm_took_110_photos (p : PhotoCount) 
  (h : satisfies_conditions p) : p.norm = 110 := by
  sorry

end norm_took_110_photos_l2261_226155


namespace quadratic_equation_solution_l2261_226184

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 3 ∧ x₁^2 - 2*x₁ = 2) ∧ 
  (x₂ = 1 - Real.sqrt 3 ∧ x₂^2 - 2*x₂ = 2) := by
  sorry

end quadratic_equation_solution_l2261_226184


namespace invisibility_elixir_combinations_l2261_226167

/-- The number of valid combinations for the invisibility elixir. -/
def valid_combinations (roots : ℕ) (minerals : ℕ) (incompatible : ℕ) : ℕ :=
  roots * minerals - incompatible

/-- Theorem: Given 4 roots, 6 minerals, and 3 incompatible combinations,
    the number of valid combinations for the invisibility elixir is 21. -/
theorem invisibility_elixir_combinations :
  valid_combinations 4 6 3 = 21 := by
  sorry

end invisibility_elixir_combinations_l2261_226167


namespace only_cylinder_not_polyhedron_l2261_226164

-- Define the set of given figures
inductive Figure
  | ObliquePrism
  | Cube
  | Cylinder
  | Tetrahedron

-- Define what a polyhedron is
def isPolyhedron (f : Figure) : Prop :=
  match f with
  | Figure.ObliquePrism => true
  | Figure.Cube => true
  | Figure.Cylinder => false
  | Figure.Tetrahedron => true

-- Theorem statement
theorem only_cylinder_not_polyhedron :
  ∀ f : Figure, ¬(isPolyhedron f) ↔ f = Figure.Cylinder :=
sorry

end only_cylinder_not_polyhedron_l2261_226164


namespace trisector_inequality_l2261_226117

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Define trisectors
def trisectors (t : AcuteTriangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem trisector_inequality (t : AcuteTriangle) : 
  let (f, g) := trisectors t
  (f + g) / 2 < 2 / (1 / t.a + 1 / t.b) := by sorry

end trisector_inequality_l2261_226117


namespace sum_of_integers_l2261_226139

theorem sum_of_integers : (-1) + 2 + (-3) + 1 + (-2) + 3 = 0 := by sorry

end sum_of_integers_l2261_226139


namespace class_average_weight_l2261_226105

theorem class_average_weight (num_boys : ℕ) (num_girls : ℕ) 
  (avg_weight_boys : ℝ) (avg_weight_girls : ℝ) :
  num_boys = 5 →
  num_girls = 3 →
  avg_weight_boys = 60 →
  avg_weight_girls = 50 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 56.25 := by
  sorry

end class_average_weight_l2261_226105


namespace log_equation_solution_l2261_226136

theorem log_equation_solution (x : ℝ) :
  x > 0 → (4 * Real.log x / Real.log 3 = Real.log (6 * x) / Real.log 3) ↔ x = (6 : ℝ) ^ (1/3) :=
by sorry

end log_equation_solution_l2261_226136


namespace range_of_ratio_l2261_226173

theorem range_of_ratio (x y : ℝ) : 
  x^2 - 8*x + y^2 - 4*y + 16 ≤ 0 → 
  0 ≤ y/x ∧ y/x ≤ 4/3 :=
by sorry

end range_of_ratio_l2261_226173


namespace brothers_identity_l2261_226128

-- Define the types for brothers and card suits
inductive Brother
| First
| Second

inductive Suit
| Black
| Red

-- Define the statements made by each brother
def firstBrotherStatement (secondBrotherName : String) (secondBrotherSuit : Suit) : Prop :=
  secondBrotherName = "Tweedledee" ∧ secondBrotherSuit = Suit.Black

def secondBrotherStatement (firstBrotherName : String) (firstBrotherSuit : Suit) : Prop :=
  firstBrotherName = "Tweedledum" ∧ firstBrotherSuit = Suit.Red

-- Define the theorem
theorem brothers_identity :
  ∃ (firstBrotherName secondBrotherName : String) 
    (firstBrotherSuit secondBrotherSuit : Suit),
    (firstBrotherName = "Tweedledee" ∧ secondBrotherName = "Tweedledum") ∧
    (firstBrotherSuit = Suit.Black ∧ secondBrotherSuit = Suit.Red) ∧
    (firstBrotherStatement secondBrotherName secondBrotherSuit ≠ 
     secondBrotherStatement firstBrotherName firstBrotherSuit) :=
by
  sorry

end brothers_identity_l2261_226128


namespace quadratic_roots_property_l2261_226186

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3*a - 4) * (6*b - 8) = 14 := by
sorry

end quadratic_roots_property_l2261_226186


namespace lilliputian_matchboxes_in_gulliverian_l2261_226129

/-- Represents the scale factor between Lilliput and Gulliver's homeland -/
def scale_factor : ℝ := 12

/-- Calculates the volume of a matchbox given its dimensions -/
def matchbox_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The number of Lilliputian matchboxes that fit into one Gulliverian matchbox is 1728 -/
theorem lilliputian_matchboxes_in_gulliverian (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  (matchbox_volume l w h) / (matchbox_volume (l / scale_factor) (w / scale_factor) (h / scale_factor)) = 1728 := by
  sorry

#check lilliputian_matchboxes_in_gulliverian

end lilliputian_matchboxes_in_gulliverian_l2261_226129


namespace ladder_problem_l2261_226116

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 15)
  (h2 : height = 9) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 12 := by
  sorry

end ladder_problem_l2261_226116


namespace max_price_changes_l2261_226110

/-- Represents the price of the souvenir after n changes -/
def price (initial : ℕ) (x : ℚ) (n : ℕ) : ℚ :=
  initial * ((1 - x/100)^n * (1 + x/100)^n)

/-- The problem statement -/
theorem max_price_changes (initial : ℕ) (x : ℚ) : 
  initial = 10000 →
  0 < x →
  x < 100 →
  (∃ n : ℕ, ¬(price initial x n).isInt ∧ (price initial x (n-1)).isInt) →
  (∃ max_changes : ℕ, 
    (∀ n : ℕ, n ≤ max_changes → (price initial x n).isInt) ∧
    ¬(price initial x (max_changes + 1)).isInt ∧
    max_changes = 5) :=
sorry

end max_price_changes_l2261_226110


namespace solve_for_k_l2261_226122

theorem solve_for_k : ∀ k : ℝ, (2 * k * 1 - (-7) = -1) → k = -4 := by
  sorry

end solve_for_k_l2261_226122
