import Mathlib

namespace swimmers_pass_count_l3115_311515

/-- Represents the swimming scenario with two swimmers in a pool. -/
structure SwimmingScenario where
  poolLength : ℝ
  swimmer1Speed : ℝ
  swimmer2Speed : ℝ
  totalTime : ℝ

/-- Calculates the number of times the swimmers pass each other. -/
def numberOfPasses (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, the swimmers pass each other 20 times. -/
theorem swimmers_pass_count (scenario : SwimmingScenario) 
  (h1 : scenario.poolLength = 90)
  (h2 : scenario.swimmer1Speed = 3)
  (h3 : scenario.swimmer2Speed = 2)
  (h4 : scenario.totalTime = 12 * 60) : -- 12 minutes in seconds
  numberOfPasses scenario = 20 := by
  sorry

end swimmers_pass_count_l3115_311515


namespace max_value_quadratic_l3115_311542

theorem max_value_quadratic :
  (∃ (r : ℝ), -3 * r^2 + 36 * r - 9 = 99) ∧
  (∀ (r : ℝ), -3 * r^2 + 36 * r - 9 ≤ 99) :=
by sorry

end max_value_quadratic_l3115_311542


namespace equation_solution_exists_l3115_311545

theorem equation_solution_exists (m : ℕ+) :
  ∃ n : ℕ+, (n : ℚ) / m = ⌊(n^2 : ℚ)^(1/3)⌋ + ⌊(n : ℚ)^(1/2)⌋ + 1 := by
  sorry

end equation_solution_exists_l3115_311545


namespace must_divide_p_l3115_311531

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  5 ∣ p := by
  sorry

end must_divide_p_l3115_311531


namespace triple_lcm_equation_l3115_311529

theorem triple_lcm_equation (a b c n : ℕ+) :
  (a.val^2 + b.val^2 = n.val * Nat.lcm a.val b.val + n.val^2) ∧
  (b.val^2 + c.val^2 = n.val * Nat.lcm b.val c.val + n.val^2) ∧
  (c.val^2 + a.val^2 = n.val * Nat.lcm c.val a.val + n.val^2) →
  a = b ∧ b = c := by
  sorry

end triple_lcm_equation_l3115_311529


namespace inequality_proof_l3115_311593

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end inequality_proof_l3115_311593


namespace tax_reduction_theorem_l3115_311596

/-- Proves that a tax reduction of 15% results in a 6.5% revenue decrease
    when consumption increases by 10% -/
theorem tax_reduction_theorem (T C : ℝ) (X : ℝ) 
  (h_positive_T : T > 0) 
  (h_positive_C : C > 0) 
  (h_consumption_increase : 1.1 * C = C + 0.1 * C) 
  (h_revenue_decrease : (T * (1 - X / 100) * (C * 1.1)) = T * C * 0.935) :
  X = 15 := by
  sorry

end tax_reduction_theorem_l3115_311596


namespace square_sum_lower_bound_l3115_311541

theorem square_sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + b^2 ≥ 1/2 := by sorry

end square_sum_lower_bound_l3115_311541


namespace max_students_is_eight_l3115_311566

/-- Represents the relationship between students -/
def KnowsRelation (n : ℕ) := Fin n → Fin n → Prop

/-- Property: Among any 3 students, there are 2 who know each other -/
def ThreeKnowTwo (n : ℕ) (knows : KnowsRelation n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- Property: Among any 4 students, there are 2 who do not know each other -/
def FourDontKnowTwo (n : ℕ) (knows : KnowsRelation n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students_is_eight :
  ∃ (knows : KnowsRelation 8), ThreeKnowTwo 8 knows ∧ FourDontKnowTwo 8 knows ∧
  ∀ n > 8, ¬∃ (knows : KnowsRelation n), ThreeKnowTwo n knows ∧ FourDontKnowTwo n knows :=
sorry

end max_students_is_eight_l3115_311566


namespace subtraction_minimizes_l3115_311550

-- Define the set of operators
inductive Operator : Type
  | add : Operator
  | sub : Operator
  | mul : Operator
  | div : Operator

-- Function to apply the operator
def apply_operator (op : Operator) (a b : ℤ) : ℤ :=
  match op with
  | Operator.add => a + b
  | Operator.sub => a - b
  | Operator.mul => a * b
  | Operator.div => a / b

-- Theorem statement
theorem subtraction_minimizes :
  ∀ op : Operator, apply_operator Operator.sub (-3) 1 ≤ apply_operator op (-3) 1 := by
  sorry

end subtraction_minimizes_l3115_311550


namespace train_crossing_time_l3115_311558

theorem train_crossing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 320 →
  platform_crossing_time = 34 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let train_length := train_speed_mps * platform_crossing_time - platform_length
  let man_crossing_time := train_length / train_speed_mps
  man_crossing_time = 18 := by
  sorry

end train_crossing_time_l3115_311558


namespace arithmetic_sequence_proof_l3115_311559

/-- 
Given three consecutive terms in an arithmetic sequence: 4x, 2x-3, and 4x-3,
prove that x = -3/4
-/
theorem arithmetic_sequence_proof (x : ℚ) : 
  (∃ (d : ℚ), (2*x - 3) - 4*x = d ∧ (4*x - 3) - (2*x - 3) = d) → 
  x = -3/4 := by
sorry

end arithmetic_sequence_proof_l3115_311559


namespace sum_of_roots_and_constant_l3115_311567

theorem sum_of_roots_and_constant (a b c : ℝ) : 
  (1^2 + a*1 + 2 = 0) → 
  (a^2 + 5*a + c = 0) → 
  (b^2 + 5*b + c = 0) → 
  a + b + c = 1 := by
  sorry

end sum_of_roots_and_constant_l3115_311567


namespace problem_statement_l3115_311597

theorem problem_statement (θ : ℝ) (h : (Real.sin θ)^2 + 4 = 2 * (Real.cos θ + 1)) :
  (Real.cos θ + 1) * (Real.sin θ + 1) = 2 := by
  sorry

end problem_statement_l3115_311597


namespace student_average_less_than_actual_average_l3115_311519

theorem student_average_less_than_actual_average (a b c : ℝ) (h : a < b ∧ b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 := by
  sorry

end student_average_less_than_actual_average_l3115_311519


namespace luka_water_needed_l3115_311574

/-- Represents the recipe ratios and amount of lemon juice used --/
structure Recipe where
  water_sugar_ratio : ℚ
  sugar_lemon_ratio : ℚ
  lemon_juice : ℚ

/-- Calculates the amount of water needed based on the recipe --/
def water_needed (r : Recipe) : ℚ :=
  r.water_sugar_ratio * r.sugar_lemon_ratio * r.lemon_juice

/-- Theorem stating that Luka needs 24 cups of water --/
theorem luka_water_needed :
  let r : Recipe := {
    water_sugar_ratio := 4,
    sugar_lemon_ratio := 2,
    lemon_juice := 3
  }
  water_needed r = 24 := by sorry

end luka_water_needed_l3115_311574


namespace complex_fraction_sum_l3115_311525

theorem complex_fraction_sum (x y : ℝ) : 
  (∃ (z : ℂ), z = (1 + y * Complex.I) / (1 + Complex.I) ∧ (z : ℂ).re = x) → x + y = 2 := by
  sorry

end complex_fraction_sum_l3115_311525


namespace larger_number_l3115_311569

theorem larger_number (P Q : ℝ) (h1 : P = Real.sqrt 2) (h2 : Q = Real.sqrt 6 - Real.sqrt 2) : P > Q := by
  sorry

end larger_number_l3115_311569


namespace exponent_multiplication_l3115_311546

theorem exponent_multiplication (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end exponent_multiplication_l3115_311546


namespace derivative_of_exponential_l3115_311520

variable (a : ℝ) (ha : a > 0)

theorem derivative_of_exponential (x : ℝ) : 
  deriv (fun x => a^x) x = a^x * Real.log a := by sorry

end derivative_of_exponential_l3115_311520


namespace cubic_equation_real_root_l3115_311571

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b*x + 25 = 0 := by
  sorry

#check cubic_equation_real_root

end cubic_equation_real_root_l3115_311571


namespace integer_fraction_condition_l3115_311547

theorem integer_fraction_condition (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k.val) ↔
  (∃ l : ℕ+, (a = l ∧ b = 2 * l) ∨ (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
by sorry

end integer_fraction_condition_l3115_311547


namespace equal_probability_wsw_more_advantageous_l3115_311564

-- Define the probabilities of winning against strong and weak players
variable (Ps Pw : ℝ)

-- Define the condition that Ps < Pw
variable (h : Ps < Pw)

-- Define the probability of winning two consecutive games in the sequence Strong, Weak, Strong
def prob_sws : ℝ := Ps * Pw

-- Define the probability of winning two consecutive games in the sequence Weak, Strong, Weak
def prob_wsw : ℝ := Pw * Ps

-- Theorem stating that both sequences have equal probability
theorem equal_probability : prob_sws Ps Pw = prob_wsw Ps Pw := by
  sorry

-- Theorem stating that Weak, Strong, Weak is more advantageous
theorem wsw_more_advantageous (h : Ps < Pw) : prob_wsw Ps Pw ≥ prob_sws Ps Pw := by
  sorry

end equal_probability_wsw_more_advantageous_l3115_311564


namespace farm_ratio_change_l3115_311589

theorem farm_ratio_change (H C : ℕ) : 
  H = 6 * C →  -- Initial ratio of horses to cows is 6:1
  H - 15 = (C + 15) + 70 →  -- After transaction, 70 more horses than cows
  (H - 15) / (C + 15) = 3  -- New ratio of horses to cows is 3:1
  := by sorry

end farm_ratio_change_l3115_311589


namespace proportion_problem_l3115_311537

/-- Given that a, b, c, and d are in proportion, where a = 3, b = 2, and c = 6, prove that d = 4. -/
theorem proportion_problem (a b c d : ℝ) : 
  a = 3 → b = 2 → c = 6 → (a * d = b * c) → d = 4 := by
  sorry

end proportion_problem_l3115_311537


namespace chef_wage_percentage_increase_l3115_311557

/-- Proves that the percentage increase in the hourly wage of a chef compared to a dishwasher is 20% -/
theorem chef_wage_percentage_increase (manager_wage : ℝ) (chef_wage : ℝ) (dishwasher_wage : ℝ) :
  manager_wage = 7.5 →
  chef_wage = manager_wage - 3 →
  dishwasher_wage = manager_wage / 2 →
  (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 := by
  sorry

end chef_wage_percentage_increase_l3115_311557


namespace largest_n_for_product_1764_l3115_311579

/-- Represents an arithmetic sequence with integer terms -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDifference : ℤ

/-- The n-th term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1 : ℤ) * seq.commonDifference

theorem largest_n_for_product_1764 (c d : ArithmeticSequence)
    (h1 : c.firstTerm = 1)
    (h2 : d.firstTerm = 1)
    (h3 : nthTerm c 2 ≤ nthTerm d 2)
    (h4 : ∃ n : ℕ, nthTerm c n * nthTerm d n = 1764) :
    (∃ n : ℕ, nthTerm c n * nthTerm d n = 1764) ∧
    (∀ m : ℕ, nthTerm c m * nthTerm d m = 1764 → m ≤ 1764) := by
  sorry

end largest_n_for_product_1764_l3115_311579


namespace problem_solution_l3115_311534

theorem problem_solution :
  ∀ M : ℝ, (5 + 6 + 7) / 3 = (1988 + 1989 + 1990) / M → M = 994.5 := by
sorry

end problem_solution_l3115_311534


namespace max_factors_of_power_l3115_311503

-- Define the type for positive integers from 1 to 15
def PositiveIntegerTo15 : Type := {x : ℕ // 1 ≤ x ∧ x ≤ 15}

-- Define the function to count factors
def countFactors (m : ℕ) : ℕ := sorry

-- Define the function to calculate b^n
def powerFunction (b n : PositiveIntegerTo15) : ℕ := sorry

-- Theorem statement
theorem max_factors_of_power :
  ∃ (b n : PositiveIntegerTo15),
    ∀ (b' n' : PositiveIntegerTo15),
      countFactors (powerFunction b n) ≥ countFactors (powerFunction b' n') ∧
      countFactors (powerFunction b n) = 496 :=
sorry

end max_factors_of_power_l3115_311503


namespace inscribed_circle_theorem_l3115_311561

-- Define the triangle and circle
structure Triangle :=
  (A B C : ℝ × ℝ)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the inscribed circle property
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define the point of tangency
def pointOfTangency (t : Triangle) (c : Circle) : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two vectors
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem inscribed_circle_theorem (t : Triangle) (c : Circle) (M : ℝ × ℝ) :
  isInscribed t c →
  M = pointOfTangency t c →
  distance t.A M = 1 →
  distance t.B M = 4 →
  angle (t.B - t.A) (t.C - t.A) = 2 * π / 3 →
  distance t.C M = Real.sqrt 273 := by sorry

end inscribed_circle_theorem_l3115_311561


namespace remainder_theorem_l3115_311511

theorem remainder_theorem (n : ℤ) : (5 * n^2 + 7) - (3 * n - 2) ≡ 2 * n + 4 [ZMOD 5] := by
  sorry

end remainder_theorem_l3115_311511


namespace scooter_initial_value_l3115_311580

/-- Proves that the initial value of a scooter is 40000 given its depreciation rate and value after 2 years -/
theorem scooter_initial_value (depreciation_rate : ℚ) (value_after_two_years : ℚ) :
  depreciation_rate = 3 / 4 →
  value_after_two_years = 22500 →
  depreciation_rate * (depreciation_rate * 40000) = value_after_two_years :=
by sorry

end scooter_initial_value_l3115_311580


namespace equal_products_l3115_311587

def numbers : List Nat := [12, 15, 33, 44, 51, 85]
def group1 : List Nat := [12, 33, 85]
def group2 : List Nat := [44, 51, 15]

theorem equal_products :
  (List.prod group1 = List.prod group2) ∧
  (group1.toFinset ∪ group2.toFinset = numbers.toFinset) ∧
  (group1.toFinset ∩ group2.toFinset = ∅) :=
sorry

end equal_products_l3115_311587


namespace biology_group_size_l3115_311586

theorem biology_group_size : 
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) = 210 ∧ ∀ m : ℕ, m > 0 ∧ m * (m - 1) = 210 → m = n :=
by sorry

end biology_group_size_l3115_311586


namespace medical_team_selection_l3115_311543

theorem medical_team_selection (m n : ℕ) (hm : m = 6) (hn : n = 5) :
  (m.choose 2) * (n.choose 1) = 75 := by
  sorry

end medical_team_selection_l3115_311543


namespace wallpaper_area_proof_l3115_311554

theorem wallpaper_area_proof (total_area overlap_area double_layer triple_layer : ℝ) : 
  overlap_area = 180 →
  double_layer = 30 →
  triple_layer = 45 →
  total_area - 2 * double_layer - 3 * triple_layer = overlap_area →
  total_area = 375 := by
  sorry

end wallpaper_area_proof_l3115_311554


namespace supermarket_spending_l3115_311563

theorem supermarket_spending (total : ℚ) : 
  (2/5 : ℚ) * total + (1/4 : ℚ) * total + (1/10 : ℚ) * total + 
  (1/8 : ℚ) * total + (1/20 : ℚ) * total + 12 = total → 
  total = 160 := by
sorry

end supermarket_spending_l3115_311563


namespace min_reciprocal_sum_l3115_311555

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ (1 / 5 : ℝ) :=
by sorry

end min_reciprocal_sum_l3115_311555


namespace partial_fraction_decomposition_l3115_311565

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ), ∀ (x : ℚ), x ≠ 4 ∧ x ≠ 2 →
    (6 * x + 2) / ((x - 4) * (x - 2)^3) = 
    P / (x - 4) + Q / (x - 2) + R / (x - 2)^3 ∧
    P = 13 / 4 ∧ Q = -13 / 2 ∧ R = -7 := by
  sorry

end partial_fraction_decomposition_l3115_311565


namespace smallest_Y_value_l3115_311548

/-- A function that checks if a positive integer consists only of 0s and 1s -/
def onlyZerosAndOnes (n : ℕ+) : Prop := sorry

/-- The theorem stating the smallest possible value of Y -/
theorem smallest_Y_value (S : ℕ+) (hS : onlyZerosAndOnes S) (hDiv : 18 ∣ S) :
  (S / 18 : ℕ) ≥ 6172839500 :=
sorry

end smallest_Y_value_l3115_311548


namespace theater_ticket_pricing_l3115_311560

/-- Theorem: Theater Ticket Pricing
  Given:
  - Total tickets sold is 340
  - Total revenue is $3,320
  - Orchestra seat price is $12
  - Number of balcony seats sold is 40 more than orchestra seats
  Prove that the cost of a balcony seat is $8
-/
theorem theater_ticket_pricing 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (orchestra_price : ℕ) 
  (balcony_excess : ℕ) 
  (h1 : total_tickets = 340)
  (h2 : total_revenue = 3320)
  (h3 : orchestra_price = 12)
  (h4 : balcony_excess = 40) :
  let orchestra_seats := (total_tickets - balcony_excess) / 2
  let balcony_seats := orchestra_seats + balcony_excess
  let balcony_revenue := total_revenue - orchestra_price * orchestra_seats
  balcony_revenue / balcony_seats = 8 := by
  sorry

end theater_ticket_pricing_l3115_311560


namespace determinant_trig_matrix_l3115_311533

open Real Matrix

theorem determinant_trig_matrix (α β : ℝ) : 
  det ![![sin α * sin β, -sin α * cos β, cos α],
        ![cos β, sin β, 0],
        ![-cos α * sin β, -cos α * cos β, sin α]] = 1 - cos α := by
  sorry

end determinant_trig_matrix_l3115_311533


namespace largest_n_for_product_4021_l3115_311524

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1 : ℤ) * seq.diff

theorem largest_n_for_product_4021 (a b : ArithmeticSequence)
    (h1 : a.first = 1)
    (h2 : b.first = 1)
    (h3 : a.diff ≤ b.diff)
    (h4 : ∃ n : ℕ, a.nthTerm n * b.nthTerm n = 4021) :
    (∀ m : ℕ, a.nthTerm m * b.nthTerm m = 4021 → m ≤ 11) ∧
    (∃ n : ℕ, n = 11 ∧ a.nthTerm n * b.nthTerm n = 4021) := by
  sorry

end largest_n_for_product_4021_l3115_311524


namespace power_equation_solution_l3115_311562

theorem power_equation_solution (m n : ℕ) (h1 : (1/5)^m * (1/4)^n = 1/(10^4)) (h2 : m = 4) : n = 2 := by
  sorry

end power_equation_solution_l3115_311562


namespace lcm_gcf_relation_l3115_311573

theorem lcm_gcf_relation (n : ℕ) :
  Nat.lcm n 24 = 48 ∧ Nat.gcd n 24 = 8 → n = 16 := by
  sorry

end lcm_gcf_relation_l3115_311573


namespace symmetric_point_of_A_l3115_311507

def line_equation (x y : ℝ) : Prop := 2*x - 4*y + 9 = 0

def is_symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the given line
  (y₂ - y₁) / (x₂ - x₁) = -1 / (1/2) ∧
  -- The midpoint of the two points lies on the given line
  line_equation ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

theorem symmetric_point_of_A : is_symmetric_point 2 2 1 4 := by sorry

end symmetric_point_of_A_l3115_311507


namespace actual_distance_l3115_311549

/-- Calculates the actual distance between two cities given the map distance and scale. -/
theorem actual_distance (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : 
  map_distance * (scale_miles / scale_distance) = 240 :=
  by
  -- Assuming map_distance = 20, scale_distance = 0.5, and scale_miles = 6
  have h1 : map_distance = 20 := by sorry
  have h2 : scale_distance = 0.5 := by sorry
  have h3 : scale_miles = 6 := by sorry
  
  -- Proof goes here
  sorry

end actual_distance_l3115_311549


namespace union_equals_N_implies_a_in_range_l3115_311576

/-- Given sets M and N, if their union equals N, then a is in the interval [-2, 2] -/
theorem union_equals_N_implies_a_in_range (a : ℝ) :
  let M := {x : ℝ | x * (x - a - 1) < 0}
  let N := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
  (M ∪ N = N) → a ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end union_equals_N_implies_a_in_range_l3115_311576


namespace josh_remaining_money_l3115_311582

/-- Calculates the remaining money after Josh's shopping trip. -/
def remaining_money (initial_amount hat_cost pencil_cost cookie_cost cookie_count : ℚ) : ℚ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * cookie_count)

/-- Theorem stating that Josh has $3 left after his purchases. -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end josh_remaining_money_l3115_311582


namespace asphalt_cost_per_truckload_l3115_311504

/-- Calculates the cost per truckload of asphalt before tax -/
theorem asphalt_cost_per_truckload
  (road_length : ℝ)
  (road_width : ℝ)
  (coverage_per_truckload : ℝ)
  (tax_rate : ℝ)
  (total_cost_with_tax : ℝ)
  (h1 : road_length = 2000)
  (h2 : road_width = 20)
  (h3 : coverage_per_truckload = 800)
  (h4 : tax_rate = 0.2)
  (h5 : total_cost_with_tax = 4500) :
  (road_length * road_width) / coverage_per_truckload *
  (total_cost_with_tax / (1 + tax_rate)) /
  ((road_length * road_width) / coverage_per_truckload) = 75 := by
sorry

end asphalt_cost_per_truckload_l3115_311504


namespace symmetric_point_l3115_311518

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := 5*x + 4*y + 21 = 0

/-- The original point P --/
def P : ℝ × ℝ := (4, 0)

/-- The symmetric point P' --/
def P' : ℝ × ℝ := (-6, -8)

/-- Theorem stating that P' is symmetric to P with respect to the line of symmetry --/
theorem symmetric_point : 
  let midpoint := ((P.1 + P'.1) / 2, (P.2 + P'.2) / 2)
  line_of_symmetry midpoint.1 midpoint.2 ∧ 
  (P'.2 - P.2) * 5 = -(P'.1 - P.1) * 4 :=
sorry

end symmetric_point_l3115_311518


namespace dice_probability_l3115_311535

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The number of dice that should show numbers less than 5 -/
def num_less_than_five : ℕ := 4

/-- The probability of rolling a number less than 5 on a single die -/
def prob_less_than_five : ℚ := 1 / 2

/-- The probability of rolling a number 5 or greater on a single die -/
def prob_five_or_greater : ℚ := 1 - prob_less_than_five

theorem dice_probability :
  (Nat.choose num_dice num_less_than_five : ℚ) *
  (prob_less_than_five ^ num_less_than_five) *
  (prob_five_or_greater ^ (num_dice - num_less_than_five)) =
  35 / 128 := by
  sorry

end dice_probability_l3115_311535


namespace fraction_equality_l3115_311591

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 3 := by
  sorry

end fraction_equality_l3115_311591


namespace radii_product_l3115_311528

/-- Two circles C₁ and C₂ with centers (2, 2) and (-1, -1) respectively, 
    radii r₁ and r₂ (both positive), that are tangent to each other, 
    and have an external common tangent line with a slope of 7. -/
structure TangentCircles where
  r₁ : ℝ
  r₂ : ℝ
  h₁ : r₁ > 0
  h₂ : r₂ > 0
  h_tangent : (r₁ + r₂)^2 = (2 - (-1))^2 + (2 - (-1))^2  -- Distance between centers equals sum of radii
  h_slope : ∃ t : ℝ, (7 * 2 - 2 + t)^2 / 50 = r₁^2 ∧ (7 * (-1) - (-1) + t)^2 / 50 = r₂^2

/-- The product of the radii of two tangent circles with the given properties is 72/25. -/
theorem radii_product (c : TangentCircles) : c.r₁ * c.r₂ = 72 / 25 := by
  sorry

end radii_product_l3115_311528


namespace baking_time_proof_l3115_311572

/-- Alice's pie-baking time in minutes -/
def alice_time : ℕ := 5

/-- Bob's pie-baking time in minutes -/
def bob_time : ℕ := 6

/-- The time period in which Alice bakes 2 more pies than Bob -/
def time_period : ℕ := 60

theorem baking_time_proof :
  (time_period / alice_time : ℚ) = (time_period / bob_time : ℚ) + 2 :=
by sorry

end baking_time_proof_l3115_311572


namespace linear_relationship_scaling_l3115_311585

/-- Given a linear relationship between x and y, this theorem proves that
    if an increase of 5 units in x results in an increase of 11 units in y,
    then an increase of 20 units in x will result in an increase of 44 units in y. -/
theorem linear_relationship_scaling (f : ℝ → ℝ) (h : ∀ x, f (x + 5) = f x + 11) :
  ∀ x, f (x + 20) = f x + 44 := by
  sorry

end linear_relationship_scaling_l3115_311585


namespace symmetric_points_on_parabola_l3115_311514

/-- Given two points on a parabola that are symmetric with respect to a line, prove the value of m -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →  -- A is on the parabola
  y₂ = 2 * x₂^2 →  -- B is on the parabola
  (∃ (x₀ y₀ : ℝ), x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 ∧ y₀ = x₀ + m) →  -- midpoint condition for symmetry
  x₁ * x₂ = -1/2 →  -- given condition
  m = 3/2 := by
sorry

end symmetric_points_on_parabola_l3115_311514


namespace terrell_total_hike_distance_l3115_311577

/-- Represents a hike with distance, duration, and calorie expenditure -/
structure Hike where
  distance : ℝ
  duration : ℝ
  calories : ℝ

/-- Calculates the total distance of two hikes -/
def total_distance (h1 h2 : Hike) : ℝ :=
  h1.distance + h2.distance

theorem terrell_total_hike_distance :
  let saturday_hike : Hike := { distance := 8.2, duration := 5, calories := 4000 }
  let sunday_hike : Hike := { distance := 1.6, duration := 2, calories := 1500 }
  total_distance saturday_hike sunday_hike = 9.8 := by
  sorry

end terrell_total_hike_distance_l3115_311577


namespace trig_inequality_l3115_311592

theorem trig_inequality (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α < Real.pi / 2)
  (h2 : 0 ≤ β ∧ β < Real.pi / 2)
  (h3 : 0 ≤ γ ∧ γ < Real.pi / 2)
  (h4 : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3/8 := by
sorry

end trig_inequality_l3115_311592


namespace intern_teacher_assignment_l3115_311532

theorem intern_teacher_assignment :
  (∀ n m : ℕ, n = 4 ∧ m = 3 ∧ n > m) →
  (number_of_assignments : ℕ) →
  number_of_assignments = 36 :=
by
  sorry

end intern_teacher_assignment_l3115_311532


namespace linear_equation_solution_l3115_311521

theorem linear_equation_solution (m : ℝ) :
  (∃ k : ℝ, ∀ x, 3 * x^(m-1) + 2 = k * x + (-3)) →
  (∀ y, 3 * m * y + 2 * y = 3 + m ↔ y = 5/8) :=
by sorry

end linear_equation_solution_l3115_311521


namespace train_speed_proof_l3115_311540

/-- Proves that a train with given parameters has a specific speed -/
theorem train_speed_proof (train_length bridge_length time_to_cross : Real)
  (h1 : train_length = 110)
  (h2 : bridge_length = 136)
  (h3 : time_to_cross = 12.299016078713702) :
  (train_length + bridge_length) / time_to_cross * 3.6 = 72 := by
  sorry

end train_speed_proof_l3115_311540


namespace max_value_inequality_max_value_achievable_l3115_311527

theorem max_value_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  (a * b + 2 * b * c + 2 * c * d + d * e) / (a^2 + 3 * b^2 + 3 * c^2 + 5 * d^2 + e^2) ≤ Real.sqrt 2 :=
by sorry

theorem max_value_achievable :
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    (a * b + 2 * b * c + 2 * c * d + d * e) / (a^2 + 3 * b^2 + 3 * c^2 + 5 * d^2 + e^2) = Real.sqrt 2 :=
by sorry

end max_value_inequality_max_value_achievable_l3115_311527


namespace accessories_total_cost_l3115_311539

def mouse_cost : ℕ := 20

def keyboard_cost : ℕ := 2 * mouse_cost

def headphones_cost : ℕ := mouse_cost + 15

def usb_hub_cost : ℕ := 36 - mouse_cost

def total_cost : ℕ := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost

theorem accessories_total_cost : total_cost = 111 := by
  sorry

end accessories_total_cost_l3115_311539


namespace minimum_questionnaires_to_mail_l3115_311523

theorem minimum_questionnaires_to_mail 
  (response_rate : ℝ) 
  (required_responses : ℕ) 
  (h1 : response_rate = 0.7) 
  (h2 : required_responses = 300) : 
  ℕ := by
  
  sorry

#check minimum_questionnaires_to_mail

end minimum_questionnaires_to_mail_l3115_311523


namespace line_inclination_l3115_311595

/-- The inclination angle of a line given by parametric equations -/
def inclination_angle (x_eq : ℝ → ℝ) (y_eq : ℝ → ℝ) : ℝ :=
  sorry

theorem line_inclination :
  let x_eq := λ t : ℝ => 3 + t * Real.sin (25 * π / 180)
  let y_eq := λ t : ℝ => -t * Real.cos (25 * π / 180)
  inclination_angle x_eq y_eq = 115 * π / 180 :=
sorry

end line_inclination_l3115_311595


namespace algebraic_expression_value_l3115_311583

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 
  6 * x - 3 * y + 1 = 7 := by
  sorry

end algebraic_expression_value_l3115_311583


namespace train_car_estimate_l3115_311578

/-- Represents the number of cars that pass in a given time interval -/
structure CarPassage where
  cars : ℕ
  seconds : ℕ

/-- Calculates the estimated number of cars in a train given initial observations and total passage time -/
def estimateTrainCars (initialObservation : CarPassage) (totalPassageTime : ℕ) : ℕ :=
  (initialObservation.cars * totalPassageTime) / initialObservation.seconds

theorem train_car_estimate :
  let initialObservation : CarPassage := { cars := 8, seconds := 12 }
  let totalPassageTime : ℕ := 210
  estimateTrainCars initialObservation totalPassageTime = 140 := by
  sorry

end train_car_estimate_l3115_311578


namespace f_range_of_a_l3115_311590

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * 2^(x-1) - 1/a else (a-2)*x + 5/3

theorem f_range_of_a (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) →
  a ∈ Set.Ioo 2 3 := by sorry

end f_range_of_a_l3115_311590


namespace simplest_quadratic_radical_l3115_311598

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ n > 1 ∧ ∀ (m : ℕ), m ^ 2 ∣ n → m = 1

theorem simplest_quadratic_radical : 
  ¬ is_simplest_quadratic_radical (Real.sqrt 4) ∧ 
  is_simplest_quadratic_radical (Real.sqrt 5) ∧ 
  ¬ is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧ 
  ¬ is_simplest_quadratic_radical (Real.sqrt 8) :=
by sorry

end simplest_quadratic_radical_l3115_311598


namespace events_mutually_exclusive_events_not_complementary_l3115_311506

-- Define the sample space for a standard six-sided die
def DieOutcome : Type := Fin 6

-- Define the event "the number is odd"
def isOdd (n : DieOutcome) : Prop := n.val % 2 = 1

-- Define the event "the number is greater than 5"
def isGreaterThan5 (n : DieOutcome) : Prop := n.val = 6

-- Theorem stating that the events are mutually exclusive
theorem events_mutually_exclusive :
  ∀ (n : DieOutcome), ¬(isOdd n ∧ isGreaterThan5 n) :=
sorry

-- Theorem stating that the events are not complementary
theorem events_not_complementary :
  ¬(∀ (n : DieOutcome), isOdd n ↔ ¬isGreaterThan5 n) :=
sorry

end events_mutually_exclusive_events_not_complementary_l3115_311506


namespace regular_tetrahedron_properties_l3115_311594

-- Define a regular tetrahedron
structure RegularTetrahedron where
  -- Add any necessary fields here
  
-- Define the properties of a regular tetrahedron
def has_equal_edges_and_vertex_angles (t : RegularTetrahedron) : Prop :=
  sorry

def has_congruent_faces_and_equal_dihedral_angles (t : RegularTetrahedron) : Prop :=
  sorry

def has_congruent_faces_and_equal_vertex_angles (t : RegularTetrahedron) : Prop :=
  sorry

-- Theorem stating that a regular tetrahedron satisfies all three properties
theorem regular_tetrahedron_properties (t : RegularTetrahedron) :
  has_equal_edges_and_vertex_angles t ∧
  has_congruent_faces_and_equal_dihedral_angles t ∧
  has_congruent_faces_and_equal_vertex_angles t :=
sorry

end regular_tetrahedron_properties_l3115_311594


namespace negation_of_universal_proposition_l3115_311599

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 ≥ 1) ↔ ∃ x₀ : ℝ, x₀^2 < 1 := by sorry

end negation_of_universal_proposition_l3115_311599


namespace bin_game_expected_value_l3115_311551

theorem bin_game_expected_value (k : ℕ) : 
  let total_balls : ℕ := 10 + k
  let prob_green : ℚ := 10 / total_balls
  let prob_purple : ℚ := k / total_balls
  let expected_value : ℚ := 3 * prob_green - 1 * prob_purple
  (expected_value = 3/4) → (k = 13) :=
by sorry

end bin_game_expected_value_l3115_311551


namespace max_product_2017_l3115_311526

def sumToN (n : ℕ) := {l : List ℕ | l.sum = n}

def productOfList (l : List ℕ) := l.prod

def optimalSumProduct (n : ℕ) : List ℕ := 
  List.replicate 671 3 ++ List.replicate 2 2

theorem max_product_2017 :
  ∀ l ∈ sumToN 2017, 
    productOfList l ≤ productOfList (optimalSumProduct 2017) :=
sorry

end max_product_2017_l3115_311526


namespace lily_correct_answers_percentage_l3115_311538

theorem lily_correct_answers_percentage 
  (t : ℝ) -- total number of problems
  (h_t_pos : t > 0) -- t is positive
  (h_max_alone : 0.85 * (2/3 * t) = 17/30 * t) -- Max's correct answers alone
  (h_max_total : 0.90 * t = 0.90 * t) -- Max's total correct answers
  (h_together : 0.75 * (1/3 * t) = 0.25 * t) -- Correct answers together
  (h_lily_alone : 0.95 * (2/3 * t) = 19/30 * t) -- Lily's correct answers alone
  : (19/30 * t + 0.25 * t) / t = 49/60 := by
  sorry

end lily_correct_answers_percentage_l3115_311538


namespace min_value_theorem_l3115_311509

theorem min_value_theorem (x y z : ℝ) (h : 2 * x + 2 * y + z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 9 := by
  sorry

end min_value_theorem_l3115_311509


namespace opposite_of_negative_2023_l3115_311536

-- Define the concept of opposite
def opposite (x : ℤ) : ℤ := -x

-- Theorem statement
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end opposite_of_negative_2023_l3115_311536


namespace probability_one_white_one_black_l3115_311556

/-- The probability of drawing one white ball and one black ball from a box -/
theorem probability_one_white_one_black (w b : ℕ) (hw : w = 7) (hb : b = 8) :
  let total := w + b
  let favorable := w * b
  let total_combinations := (total * (total - 1)) / 2
  (favorable : ℚ) / total_combinations = 56 / 105 := by sorry

end probability_one_white_one_black_l3115_311556


namespace xyz_relation_theorem_l3115_311544

/-- A structure representing the relationship between x, y, and z -/
structure XYZRelation where
  x : ℝ
  y : ℝ
  z : ℝ
  c : ℝ
  d : ℝ
  h1 : y^2 = c * z^2  -- y² varies directly with z²
  h2 : y = d / x      -- y varies inversely with x

/-- The theorem statement -/
theorem xyz_relation_theorem (r : XYZRelation) (h3 : r.y = 3) (h4 : r.x = 4) (h5 : r.z = 6) :
  ∃ (r' : XYZRelation), r'.y = 2 ∧ r'.z = 12 ∧ r'.x = 6 ∧ r'.c = r.c ∧ r'.d = r.d :=
sorry


end xyz_relation_theorem_l3115_311544


namespace product_invariance_l3115_311553

theorem product_invariance (a b : ℝ) (h : a * b = 300) :
  (6 * a) * (b / 6) = 300 := by
  sorry

end product_invariance_l3115_311553


namespace triangle_angle_C_l3115_311568

/-- Given a triangle with angle A = 30°, side a = 1, and side b = √2,
    prove that the angle C is either 105° or 15°. -/
theorem triangle_angle_C (A : Real) (a b : Real) :
  A = 30 * π / 180 →
  a = 1 →
  b = Real.sqrt 2 →
  ∃ (C : Real), (C = 105 * π / 180 ∨ C = 15 * π / 180) :=
by sorry

end triangle_angle_C_l3115_311568


namespace decagon_diagonals_l3115_311508

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l3115_311508


namespace stamp_trade_l3115_311584

theorem stamp_trade (anna_initial alison_initial jeff_initial anna_final : ℕ) 
  (h1 : anna_initial = 37)
  (h2 : alison_initial = 28)
  (h3 : jeff_initial = 31)
  (h4 : anna_final = 50) : 
  (anna_initial + alison_initial / 2) - anna_final = 1 := by
  sorry

end stamp_trade_l3115_311584


namespace functional_equation_solution_l3115_311552

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end functional_equation_solution_l3115_311552


namespace tissue_used_count_l3115_311530

def initial_tissue_count : ℕ := 97
def remaining_tissue_count : ℕ := 93

theorem tissue_used_count : initial_tissue_count - remaining_tissue_count = 4 := by
  sorry

end tissue_used_count_l3115_311530


namespace percent_relation_l3115_311502

/-- Given that x is p percent of y, prove that p = 100x / y -/
theorem percent_relation (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y := by
  sorry

end percent_relation_l3115_311502


namespace min_triangles_in_configuration_l3115_311517

/-- A configuration of lines in a plane. -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersect : Bool

/-- The number of triangular regions formed by a line configuration. -/
def num_triangles (config : LineConfiguration) : ℕ := sorry

/-- Theorem: Given 3000 lines drawn on a plane where no two lines are parallel
    and no three lines intersect at a single point, the number of triangular
    regions formed is at least 2000. -/
theorem min_triangles_in_configuration :
  ∀ (config : LineConfiguration),
    config.num_lines = 3000 →
    config.no_parallel = true →
    config.no_triple_intersect = true →
    num_triangles config ≥ 2000 := by sorry

end min_triangles_in_configuration_l3115_311517


namespace f_at_four_equals_zero_l3115_311500

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_at_four_equals_zero : f 4 = 0 := by
  sorry

end f_at_four_equals_zero_l3115_311500


namespace max_value_3x_4y_l3115_311513

theorem max_value_3x_4y (x y : ℝ) (h : x^2 + y^2 = 14*x + 6*y + 6) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + b^2 = 14*a + 6*b + 6 → 3*a + 4*b ≤ max) ∧ max = 73 := by
  sorry

end max_value_3x_4y_l3115_311513


namespace soccer_game_total_goals_l3115_311575

theorem soccer_game_total_goals :
  let team_a_first_half : ℕ := 8
  let team_b_first_half : ℕ := team_a_first_half / 2
  let team_b_second_half : ℕ := team_a_first_half
  let team_a_second_half : ℕ := team_b_second_half - 2
  team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half = 26 :=
by sorry

end soccer_game_total_goals_l3115_311575


namespace birthday_age_proof_l3115_311510

theorem birthday_age_proof (A : ℤ) : A = 4 * (A - 10) - 5 ↔ A = 15 := by
  sorry

end birthday_age_proof_l3115_311510


namespace problem_statement_l3115_311522

theorem problem_statement (a b : ℝ) 
  (h1 : a + 1 / (a + 1) = b + 1 / (b - 1) - 2)
  (h2 : a ≠ -1)
  (h3 : b ≠ 1)
  (h4 : a - b + 2 ≠ 0) :
  a * b - a + b = 2 := by
sorry

end problem_statement_l3115_311522


namespace pizza_delivery_gas_theorem_l3115_311581

/-- The amount of gas remaining after a pizza delivery route. -/
def gas_remaining (start : Float) (used : Float) : Float :=
  start - used

/-- Theorem stating that given the starting amount and used amount of gas,
    the remaining amount is correctly calculated. -/
theorem pizza_delivery_gas_theorem :
  gas_remaining 0.5 0.3333333333333333 = 0.1666666666666667 := by
  sorry

end pizza_delivery_gas_theorem_l3115_311581


namespace middle_term_value_l3115_311516

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℤ  -- First term
  b : ℤ  -- Second term
  c : ℤ  -- Third term
  is_arithmetic : b - a = c - b

/-- The problem statement -/
theorem middle_term_value (seq : ArithmeticSequence3) 
  (h1 : seq.a = 2^3)
  (h2 : seq.c = 2^5) : 
  seq.b = 20 := by
  sorry

end middle_term_value_l3115_311516


namespace monday_children_count_l3115_311501

/-- The number of children who went to the zoo on Monday -/
def monday_children : ℕ := sorry

/-- The number of adults who went to the zoo on Monday -/
def monday_adults : ℕ := 5

/-- The number of children who went to the zoo on Tuesday -/
def tuesday_children : ℕ := 4

/-- The number of adults who went to the zoo on Tuesday -/
def tuesday_adults : ℕ := 2

/-- The cost of a child ticket -/
def child_ticket_cost : ℕ := 3

/-- The cost of an adult ticket -/
def adult_ticket_cost : ℕ := 4

/-- The total revenue for both days -/
def total_revenue : ℕ := 61

theorem monday_children_count : 
  monday_children = 7 ∧
  monday_children * child_ticket_cost + 
  monday_adults * adult_ticket_cost +
  tuesday_children * child_ticket_cost +
  tuesday_adults * adult_ticket_cost = total_revenue :=
sorry

end monday_children_count_l3115_311501


namespace equation_one_solution_l3115_311588

theorem equation_one_solution (a : ℝ) : 
  (∃! x : ℝ, (Real.log (x + 1) + Real.log (3 - x) = Real.log (1 - a * x)) ∧ 
   (-1 < x ∧ x < 3)) ↔ 
  (-1 ≤ a ∧ a ≤ 1/3) :=
sorry

end equation_one_solution_l3115_311588


namespace apple_basket_theorem_l3115_311505

/-- Represents the number of apples in each basket -/
def baskets : List ℕ := [20, 30, 40, 60, 90]

/-- The total number of apples initially -/
def total : ℕ := baskets.sum

/-- Checks if a number is divisible by 3 -/
def divisibleBy3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

/-- Checks if removing a basket results in a valid 2:1 ratio -/
def validRemoval (n : ℕ) : Prop :=
  n ∈ baskets ∧ divisibleBy3 (total - n) ∧
  ∃ x y : ℕ, x + y = total - n ∧ x = 2 * y ∧
  (x ∈ baskets.filter (· ≠ n) ∨ y ∈ baskets.filter (· ≠ n))

/-- The main theorem -/
theorem apple_basket_theorem :
  ∀ n : ℕ, validRemoval n → n = 60 ∨ n = 90 := by sorry

end apple_basket_theorem_l3115_311505


namespace intersection_M_N_l3115_311570

-- Define set M
def M : Set ℝ := {x | x^2 - x ≤ 0}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ico 0 1 := by
  sorry

end intersection_M_N_l3115_311570


namespace todd_snow_cone_stand_l3115_311512

/-- Todd's snow-cone stand problem -/
theorem todd_snow_cone_stand (borrowed : ℝ) (repay : ℝ) (equipment : ℝ) (ingredients : ℝ) 
  (marketing : ℝ) (snow_cones : ℕ) (price : ℝ) : 
  borrowed = 200 →
  repay = 220 →
  equipment = 100 →
  ingredients = 45 →
  marketing = 30 →
  snow_cones = 350 →
  price = 1.5 →
  snow_cones * price - (equipment + ingredients + marketing) - repay = 130 := by
  sorry

end todd_snow_cone_stand_l3115_311512
