import Mathlib

namespace marbles_lost_ratio_l3511_351155

/-- Represents the number of marbles Beth has initially -/
def total_marbles : ℕ := 72

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the number of red marbles lost -/
def red_lost : ℕ := 5

/-- Represents the number of marbles Beth has left after losing some -/
def marbles_left : ℕ := 42

/-- Represents the ratio of yellow marbles lost to red marbles lost -/
def yellow_to_red_ratio : ℕ := 3

theorem marbles_lost_ratio :
  ∃ (blue_lost : ℕ),
    (total_marbles / num_colors = total_marbles / num_colors) ∧
    (total_marbles - red_lost - blue_lost - (yellow_to_red_ratio * red_lost) = marbles_left) ∧
    (blue_lost : ℚ) / red_lost = 2 := by
  sorry

end marbles_lost_ratio_l3511_351155


namespace cubic_roots_sum_cubes_l3511_351104

theorem cubic_roots_sum_cubes (u v w : ℝ) : 
  (5 * u^3 + 500 * u + 1005 = 0) →
  (5 * v^3 + 500 * v + 1005 = 0) →
  (5 * w^3 + 500 * w + 1005 = 0) →
  (u + v)^3 + (v + w)^3 + (w + u)^3 = 603 := by
sorry

end cubic_roots_sum_cubes_l3511_351104


namespace max_value_of_x_plus_y_l3511_351139

theorem max_value_of_x_plus_y (x y : ℝ) 
  (h1 : 5 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  x + y ≤ 18 / 7 := by
  sorry

end max_value_of_x_plus_y_l3511_351139


namespace complex_average_equals_three_halves_l3511_351143

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- The main theorem to prove -/
theorem complex_average_equals_three_halves :
  avg3 (avg3 (avg2 2 2) 3 1) (avg2 1 2) 1 = 3/2 := by
  sorry

end complex_average_equals_three_halves_l3511_351143


namespace original_red_marbles_l3511_351123

-- Define the initial number of red and green marbles
variable (r g : ℚ)

-- Define the conditions
def initial_ratio : Prop := r / g = 3 / 2
def new_ratio : Prop := (r - 15) / (g + 25) = 2 / 5

-- State the theorem
theorem original_red_marbles 
  (h1 : initial_ratio r g) 
  (h2 : new_ratio r g) : 
  r = 375 / 11 := by
  sorry

end original_red_marbles_l3511_351123


namespace parallel_lines_condition_l3511_351146

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ : ℝ} : 
  (m₁ = m₂) ↔ (∀ (x y : ℝ), m₁ * x + y = 0 ↔ m₂ * x + y = 0)

/-- The slope of a line ax + by = c is -a/b -/
axiom slope_of_line {a b c : ℝ} (hb : b ≠ 0) : 
  ∀ (x y : ℝ), a * x + b * y = c → -a/b * x + y = c/b

theorem parallel_lines_condition (m : ℝ) :
  (∀ (x y : ℝ), (m - 1) * x + y = 4 * m - 1 ↔ 2 * x - 3 * y = 5) ↔ m = 1/3 :=
by sorry

end parallel_lines_condition_l3511_351146


namespace exists_circuit_with_rational_resistance_l3511_351176

/-- Represents an electrical circuit composed of unit resistances -/
inductive Circuit
  | unit : Circuit
  | series : Circuit → Circuit → Circuit
  | parallel : Circuit → Circuit → Circuit

/-- Calculates the resistance of a circuit -/
def resistance : Circuit → ℚ
  | Circuit.unit => 1
  | Circuit.series c1 c2 => resistance c1 + resistance c2
  | Circuit.parallel c1 c2 => 1 / (1 / resistance c1 + 1 / resistance c2)

/-- Theorem: For any rational number a/b (where a and b are positive integers),
    there exists an electrical circuit composed of unit resistances
    whose total resistance is equal to a/b -/
theorem exists_circuit_with_rational_resistance (a b : ℕ) (h : b > 0) :
  ∃ c : Circuit, resistance c = a / b := by sorry

end exists_circuit_with_rational_resistance_l3511_351176


namespace geometric_sequence_product_l3511_351168

/-- Given a geometric sequence {a_n} where a_1 = 1/9 and a_4 = 3, 
    the product of the first five terms is equal to 1 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 4 / a 1)^(1/3)) → -- Geometric sequence condition
  a 1 = 1/9 →                                  -- First term condition
  a 4 = 3 →                                    -- Fourth term condition
  a 1 * a 2 * a 3 * a 4 * a 5 = 1 :=            -- Product of first five terms
by sorry

end geometric_sequence_product_l3511_351168


namespace inequality_proof_l3511_351112

theorem inequality_proof (x y : ℝ) (n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  x^n / (x + y^3) + y^n / (x^3 + y) ≥ 2^(4-n) / 5 := by
  sorry

end inequality_proof_l3511_351112


namespace existence_of_five_numbers_l3511_351182

theorem existence_of_five_numbers : ∃ (a₁ a₂ a₃ a₄ a₅ : ℝ), 
  (a₁ + a₂ < 0) ∧ (a₂ + a₃ < 0) ∧ (a₃ + a₄ < 0) ∧ (a₄ + a₅ < 0) ∧ 
  (a₁ + a₂ + a₃ + a₄ + a₅ > 0) := by
  sorry

end existence_of_five_numbers_l3511_351182


namespace blue_card_value_is_five_l3511_351116

/-- The value of a blue card in credits -/
def blue_card_value (total_credits : ℕ) (total_cards : ℕ) (red_card_value : ℕ) (red_cards : ℕ) : ℕ :=
  (total_credits - red_card_value * red_cards) / (total_cards - red_cards)

/-- Theorem stating that the value of a blue card is 5 credits -/
theorem blue_card_value_is_five :
  blue_card_value 84 20 3 8 = 5 := by
  sorry

end blue_card_value_is_five_l3511_351116


namespace analects_reasoning_is_common_sense_l3511_351196

-- Define the types of reasoning
inductive ReasoningType
  | CommonSense
  | Inductive
  | Analogical
  | Deductive

-- Define the structure of a statement in the reasoning chain
structure Statement :=
  (premise : String)
  (conclusion : String)

-- Define the passage from The Analects as a list of statements
def analectsPassage : List Statement :=
  [⟨"Names are not correct", "Language will not be in accordance with the truth of things"⟩,
   ⟨"Language is not in accordance with the truth of things", "Affairs cannot be carried out successfully"⟩,
   ⟨"Affairs cannot be carried out successfully", "Rites and music will not flourish"⟩,
   ⟨"Rites and music do not flourish", "Punishments will not be properly executed"⟩,
   ⟨"Punishments are not properly executed", "The people will have nowhere to put their hands and feet"⟩]

-- Define a function to determine the type of reasoning
def determineReasoningType (passage : List Statement) : ReasoningType := sorry

-- Theorem stating that the reasoning in the Analects passage is common sense reasoning
theorem analects_reasoning_is_common_sense :
  determineReasoningType analectsPassage = ReasoningType.CommonSense := by sorry

end analects_reasoning_is_common_sense_l3511_351196


namespace no_solution_PP_QQ_l3511_351195

-- Define the type of polynomials over ℝ
variable (P Q : ℝ → ℝ)

-- Hypothesis: P and Q are polynomials
axiom P_polynomial : Polynomial ℝ
axiom Q_polynomial : Polynomial ℝ

-- Hypothesis: ∀x ∈ ℝ, P(Q(x)) = Q(P(x))
axiom functional_equality : ∀ x : ℝ, P (Q x) = Q (P x)

-- Hypothesis: P(x) = Q(x) has no solutions
axiom no_solution_PQ : ∀ x : ℝ, P x ≠ Q x

-- Theorem: P(P(x)) = Q(Q(x)) has no solutions
theorem no_solution_PP_QQ : ∀ x : ℝ, P (P x) ≠ Q (Q x) := by
  sorry

end no_solution_PP_QQ_l3511_351195


namespace square_sum_is_one_l3511_351172

/-- Given two real numbers A and B, we define two functions f and g. -/
def f (A B x : ℝ) : ℝ := A * x^2 + B

def g (A B x : ℝ) : ℝ := B * x^2 + A

/-- The main theorem stating that under certain conditions, A^2 + B^2 = 1 -/
theorem square_sum_is_one (A B : ℝ) (h1 : A ≠ B) 
    (h2 : ∀ x, f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A^2 + B^2 = 1 := by
  sorry


end square_sum_is_one_l3511_351172


namespace interest_rate_difference_l3511_351118

/-- Proves that for a sum of $700 at simple interest for 4 years, 
    if a higher rate fetches $56 more interest, 
    then the difference between the higher rate and the original rate is 2 percentage points. -/
theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (original_rate : ℝ) 
  (higher_rate : ℝ) 
  (h1 : principal = 700) 
  (h2 : time = 4) 
  (h3 : higher_rate * principal * time / 100 = original_rate * principal * time / 100 + 56) : 
  higher_rate - original_rate = 2 := by
sorry

end interest_rate_difference_l3511_351118


namespace disjoint_sets_cardinality_relation_l3511_351141

theorem disjoint_sets_cardinality_relation 
  (a b : ℕ+) 
  (A B : Finset ℤ) 
  (h_disjoint : Disjoint A B)
  (h_membership : ∀ i ∈ A ∪ B, (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := by
  sorry

end disjoint_sets_cardinality_relation_l3511_351141


namespace largest_integer_below_sqrt_sum_power_l3511_351163

theorem largest_integer_below_sqrt_sum_power : 
  ∃ n : ℕ, n = 7168 ∧ n < (Real.sqrt 5 + Real.sqrt (3/2))^8 ∧ 
  ∀ m : ℕ, m < (Real.sqrt 5 + Real.sqrt (3/2))^8 → m ≤ n :=
sorry

end largest_integer_below_sqrt_sum_power_l3511_351163


namespace max_candy_eaten_l3511_351175

def board_operation (board : List Nat) : Nat → Nat → List Nat :=
  fun i j => (board.removeNth i).removeNth j ++ [board[i]! + board[j]!]

def candy_eaten (board : List Nat) : Nat → Nat → Nat :=
  fun i j => board[i]! * board[j]!

theorem max_candy_eaten :
  ∃ (operations : List (Nat × Nat)),
    operations.length = 33 ∧
    (operations.foldl
      (fun (acc : List Nat × Nat) (op : Nat × Nat) =>
        (board_operation acc.1 op.1 op.2, acc.2 + candy_eaten acc.1 op.1 op.2))
      (List.replicate 34 1, 0)).2 = 561 :=
sorry

end max_candy_eaten_l3511_351175


namespace first_worker_time_l3511_351122

/-- Given three workers who make parts with the following conditions:
    1. They need to make 80 identical parts in total.
    2. Together, they produce 20 parts per hour.
    3. The first worker makes 20 parts, taking more than 3 hours.
    4. The remaining work is completed by the second and third workers together.
    5. The total time taken to complete the work is 8 hours.
    
    This theorem proves that it would take the first worker 16 hours to make all 80 parts by himself. -/
theorem first_worker_time (x y z : ℝ) (h1 : x + y + z = 20) 
  (h2 : 20 / x > 3) (h3 : 20 / x + 60 / (y + z) = 8) : 80 / x = 16 := by
  sorry

end first_worker_time_l3511_351122


namespace smallest_proportional_part_l3511_351198

theorem smallest_proportional_part (total : ℚ) (p1 p2 p3 : ℚ) (h1 : total = 130) 
  (h2 : p1 = 1) (h3 : p2 = 1/4) (h4 : p3 = 1/5) 
  (h5 : ∃ x : ℚ, x * p1 + x * p2 + x * p3 = total) : 
  min (x * p1) (min (x * p2) (x * p3)) = 2600/145 :=
sorry

end smallest_proportional_part_l3511_351198


namespace number_list_difference_l3511_351190

theorem number_list_difference (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : (x₁ + x₂ + x₃) / 3 = -3)
  (h2 : (x₁ + x₂ + x₃ + x₄) / 4 = 4)
  (h3 : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = -5) :
  x₄ - x₅ = 66 := by
sorry

end number_list_difference_l3511_351190


namespace arithmetic_verification_l3511_351144

theorem arithmetic_verification (A B C M N P : ℝ) : 
  (A - B = C → C + B = A ∧ A - C = B) ∧ 
  (M * N = P → P / N = M ∧ P / M = N) := by
  sorry

end arithmetic_verification_l3511_351144


namespace parallel_condition_l3511_351109

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ * c₂ ≠ m₂ * c₁

/-- The theorem stating that a=1 is a necessary and sufficient condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  parallel a 2 (-1) 1 2 4 ↔ a = 1 := by
  sorry


end parallel_condition_l3511_351109


namespace intersection_sum_l3511_351103

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y - 5)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 10 :=
by sorry

end intersection_sum_l3511_351103


namespace tax_revenue_decrease_l3511_351179

theorem tax_revenue_decrease (T C : ℝ) (T_positive : T > 0) (C_positive : C > 0) :
  let new_tax := 0.8 * T
  let new_consumption := 1.05 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.16 := by
  sorry

end tax_revenue_decrease_l3511_351179


namespace triple_application_equals_six_l3511_351130

/-- The function f defined as f(p) = 2p - 20 --/
def f (p : ℝ) : ℝ := 2 * p - 20

/-- Theorem stating that there exists a unique real number p such that f(f(f(p))) = 6 --/
theorem triple_application_equals_six :
  ∃! p : ℝ, f (f (f p)) = 6 := by
  sorry

end triple_application_equals_six_l3511_351130


namespace alberts_earnings_increase_l3511_351140

/-- Proves that given Albert's earnings of $495 after a 36% increase and $454.96 after an unknown percentage increase, the unknown percentage increase is 25%. -/
theorem alberts_earnings_increase (original : ℝ) (increased : ℝ) (percentage : ℝ) : 
  (original * 1.36 = 495) → 
  (original * (1 + percentage) = 454.96) → 
  percentage = 0.25 := by
  sorry

end alberts_earnings_increase_l3511_351140


namespace profit_percent_from_cost_price_ratio_l3511_351105

/-- Profit percent calculation given cost price as a percentage of selling price -/
theorem profit_percent_from_cost_price_ratio (selling_price : ℝ) (cost_price_ratio : ℝ) 
  (h : cost_price_ratio = 0.8) : 
  (selling_price - cost_price_ratio * selling_price) / (cost_price_ratio * selling_price) * 100 = 25 := by
  sorry

end profit_percent_from_cost_price_ratio_l3511_351105


namespace ternary_to_decimal_l3511_351152

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

def ternary_number : List Nat := [1, 0, 2, 0, 1, 2]

theorem ternary_to_decimal :
  to_decimal ternary_number 3 = 320 := by
  sorry

end ternary_to_decimal_l3511_351152


namespace systematic_sample_theorem_l3511_351151

def systematic_sample_count (total_population : ℕ) (sample_size : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ :=
  let group_size := total_population / sample_size
  ((range_end - range_start + 1) / group_size)

theorem systematic_sample_theorem :
  systematic_sample_count 800 20 121 400 = 7 := by
  sorry

end systematic_sample_theorem_l3511_351151


namespace shopping_money_l3511_351127

theorem shopping_money (initial_amount : ℝ) : 
  (0.7 * initial_amount = 350) → initial_amount = 500 := by
  sorry

end shopping_money_l3511_351127


namespace haley_final_lives_l3511_351149

/-- Calculates the final number of lives in a video game scenario. -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Proves that for the given scenario, the final number of lives is 46. -/
theorem haley_final_lives : final_lives 14 4 36 = 46 := by
  sorry

end haley_final_lives_l3511_351149


namespace point_A_on_curve_l3511_351153

/-- The equation of curve C is x^2 + x + y - 1 = 0 -/
def curve_equation (x y : ℝ) : Prop := x^2 + x + y - 1 = 0

/-- Point A has coordinates (0, 1) -/
def point_A : ℝ × ℝ := (0, 1)

/-- Theorem: Point A lies on curve C -/
theorem point_A_on_curve : curve_equation point_A.1 point_A.2 := by sorry

end point_A_on_curve_l3511_351153


namespace reciprocal_of_complex_l3511_351188

theorem reciprocal_of_complex (z : ℂ) (h : z = 5 + I) : 
  z⁻¹ = 5 / 26 - (1 / 26) * I :=
by sorry

end reciprocal_of_complex_l3511_351188


namespace polynomial_division_remainder_l3511_351100

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^6 + 2*x^5 - 3*x^4 - 4*x^3 + 5*x^2 - 6*x + 7 = 
  (x^2 - 1) * (x - 2) * q + (-3*x^2 - 8*x + 13) := by
  sorry

end polynomial_division_remainder_l3511_351100


namespace min_clerks_needed_is_84_l3511_351121

/-- The number of forms a clerk can process per hour -/
def forms_per_hour : ℕ := 25

/-- The time in minutes to process a type A form -/
def time_per_type_a : ℕ := 3

/-- The time in minutes to process a type B form -/
def time_per_type_b : ℕ := 4

/-- The number of type A forms to process -/
def num_type_a : ℕ := 3000

/-- The number of type B forms to process -/
def num_type_b : ℕ := 4000

/-- The number of hours in a workday -/
def hours_per_day : ℕ := 5

/-- The function to calculate the minimum number of clerks needed -/
def min_clerks_needed : ℕ :=
  let total_minutes := num_type_a * time_per_type_a + num_type_b * time_per_type_b
  let total_hours := (total_minutes + 59) / 60  -- Ceiling division
  (total_hours + hours_per_day - 1) / hours_per_day  -- Ceiling division

theorem min_clerks_needed_is_84 : min_clerks_needed = 84 := by
  sorry

end min_clerks_needed_is_84_l3511_351121


namespace yard_length_l3511_351131

theorem yard_length (n : ℕ) (d : ℝ) (h1 : n = 26) (h2 : d = 14) : 
  (n - 1) * d = 350 := by
  sorry

end yard_length_l3511_351131


namespace lcm_problem_l3511_351156

theorem lcm_problem (n : ℕ) (h1 : n > 0) (h2 : Nat.lcm 24 n = 72) (h3 : Nat.lcm n 27 = 108) :
  n = 36 := by
  sorry

end lcm_problem_l3511_351156


namespace leo_marbles_l3511_351178

theorem leo_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) 
  (manny_fraction : ℚ) (neil_fraction : ℚ) : 
  total_marbles = 400 →
  marbles_per_pack = 10 →
  manny_fraction = 1/4 →
  neil_fraction = 1/8 →
  (total_marbles / marbles_per_pack : ℚ) * (1 - manny_fraction - neil_fraction) = 25 := by
  sorry

end leo_marbles_l3511_351178


namespace bob_weighs_165_l3511_351142

/-- Bob's weight given the conditions -/
def bobs_weight (jim_weight bob_weight : ℕ) : Prop :=
  (jim_weight + bob_weight = 220) ∧ 
  (bob_weight - jim_weight = 2 * jim_weight) ∧
  (bob_weight = 165)

/-- Theorem stating that Bob's weight is 165 pounds given the conditions -/
theorem bob_weighs_165 :
  ∃ (jim_weight bob_weight : ℕ), bobs_weight jim_weight bob_weight :=
by
  sorry

end bob_weighs_165_l3511_351142


namespace tunnel_length_l3511_351177

/-- The length of a tunnel given a train passing through it -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) : 
  train_length = 2 →
  exit_time = 4 →
  train_speed = 90 →
  (train_speed / 60 * exit_time) - train_length = 4 :=
by
  sorry

end tunnel_length_l3511_351177


namespace vector_angle_theorem_l3511_351193

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def angle (a b : E) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

theorem vector_angle_theorem (a b : E) (k : ℝ) (hk : k ≠ 0) 
  (h : norm (a + k • b) = norm (a - b)) : 
  (k = -1 → angle a b = Real.pi / 2) ∧ 
  (k ≠ -1 → angle a b = Real.arccos (-1 / (k + 1))) :=
sorry

end

end vector_angle_theorem_l3511_351193


namespace last_four_digits_of_5_pow_2013_l3511_351154

def last_four_digits (n : ℕ) : ℕ := n % 10000

def cycle_pattern : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2013 :
  last_four_digits (5^2013) = 3125 := by
  sorry

end last_four_digits_of_5_pow_2013_l3511_351154


namespace zig_book_count_l3511_351173

/-- Given that Zig wrote four times as many books as Flo and they wrote 75 books in total,
    prove that Zig wrote 60 books. -/
theorem zig_book_count (flo_books : ℕ) (zig_books : ℕ) : 
  zig_books = 4 * flo_books →  -- Zig wrote four times as many books as Flo
  zig_books + flo_books = 75 →  -- They wrote 75 books altogether
  zig_books = 60 :=  -- Prove that Zig wrote 60 books
by sorry

end zig_book_count_l3511_351173


namespace nested_radical_equality_l3511_351183

theorem nested_radical_equality : ∃! (a b : ℕ), 
  0 < a ∧ a < b ∧ 
  (Real.sqrt (1 + Real.sqrt (24 + 15 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) ∧
  a = 2 ∧ b = 3 := by
sorry

end nested_radical_equality_l3511_351183


namespace triangle_angle_measure_l3511_351108

theorem triangle_angle_measure (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : C - A = 80) 
  (h3 : A + B + C = 180) : C = 100 := by
  sorry

end triangle_angle_measure_l3511_351108


namespace orange_profit_theorem_l3511_351102

/-- Represents the orange selling scenario --/
structure OrangeSelling where
  buy_price : ℚ  -- Price to buy 4 oranges in cents
  sell_price : ℚ  -- Price to sell 7 oranges in cents
  free_oranges : ℕ  -- Number of free oranges per 8 bought
  target_profit : ℚ  -- Target profit in cents
  oranges_to_sell : ℕ  -- Number of oranges to sell

/-- Calculates the profit from selling oranges --/
def calculate_profit (scenario : OrangeSelling) : ℚ :=
  let cost_per_9 := scenario.buy_price * 2  -- Cost for 8 bought + 1 free
  let cost_per_orange := cost_per_9 / 9
  let revenue_per_orange := scenario.sell_price / 7
  let profit_per_orange := revenue_per_orange - cost_per_orange
  profit_per_orange * scenario.oranges_to_sell

/-- Theorem: Selling 120 oranges results in a profit of at least 200 cents --/
theorem orange_profit_theorem (scenario : OrangeSelling) 
  (h1 : scenario.buy_price = 15)
  (h2 : scenario.sell_price = 35)
  (h3 : scenario.free_oranges = 1)
  (h4 : scenario.target_profit = 200)
  (h5 : scenario.oranges_to_sell = 120) :
  calculate_profit scenario ≥ scenario.target_profit :=
sorry

end orange_profit_theorem_l3511_351102


namespace triangle_side_length_l3511_351197

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : Real)
  (x y z : Real)

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (h1 : t.y = 7) 
  (h2 : t.z = 6) 
  (h3 : Real.cos (t.Y - t.Z) = 15/16) : 
  t.x = Real.sqrt 22 := by
  sorry

end triangle_side_length_l3511_351197


namespace min_translation_for_symmetry_l3511_351101

theorem min_translation_for_symmetry :
  let f (x m : ℝ) := Real.sin (2 * (x - m) - π / 6)
  ∀ m : ℝ, m > 0 →
    (∀ x : ℝ, f x m = f (-x) m) →
    m ≥ π / 6 :=
by sorry

end min_translation_for_symmetry_l3511_351101


namespace kite_area_is_40_l3511_351125

/-- A point in 2D space represented by its coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A kite defined by its four vertices -/
structure Kite where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculate the area of a kite given its vertices -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The specific kite from the problem -/
def problemKite : Kite := {
  v1 := { x := 0, y := 6 }
  v2 := { x := 4, y := 10 }
  v3 := { x := 8, y := 6 }
  v4 := { x := 4, y := 0 }
}

theorem kite_area_is_40 : kiteArea problemKite = 40 := by sorry

end kite_area_is_40_l3511_351125


namespace missing_donuts_percentage_l3511_351194

def initial_donuts : ℕ := 30
def remaining_donuts : ℕ := 9

theorem missing_donuts_percentage :
  (initial_donuts - remaining_donuts) / initial_donuts * 100 = 70 := by
  sorry

end missing_donuts_percentage_l3511_351194


namespace line_chart_best_for_temperature_l3511_351185

/-- Represents different types of charts --/
inductive ChartType
| BarChart
| LineChart
| PieChart

/-- Represents the characteristics a chart can show --/
structure ChartCharacteristics where
  showsAmount : Bool
  showsChangeOverTime : Bool
  showsPartToWhole : Bool

/-- Defines the characteristics of different chart types --/
def chartTypeCharacteristics : ChartType → ChartCharacteristics
| ChartType.BarChart => ⟨true, false, false⟩
| ChartType.LineChart => ⟨true, true, false⟩
| ChartType.PieChart => ⟨false, false, true⟩

/-- Defines what characteristics are needed for temperature representation --/
def temperatureRepresentationNeeds : ChartCharacteristics :=
  ⟨true, true, false⟩

/-- Theorem: Line chart is the most appropriate for representing temperature changes --/
theorem line_chart_best_for_temperature : 
  ∀ (ct : ChartType), 
    (chartTypeCharacteristics ct = temperatureRepresentationNeeds) → 
    (ct = ChartType.LineChart) :=
by sorry

end line_chart_best_for_temperature_l3511_351185


namespace series_sum_l3511_351184

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The series term -/
noncomputable def series_term (k : ℕ) : ℤ := 
  floor ((1 + Real.sqrt (2000000 / 4^k)) / 2)

/-- The theorem statement -/
theorem series_sum : ∑' k, series_term k = 1414 := by
  sorry

end series_sum_l3511_351184


namespace abcdef_hex_bits_l3511_351119

def hex_to_decimal (hex : String) : ℕ :=
  -- Convert hexadecimal string to decimal
  sorry

def bits_required (n : ℕ) : ℕ :=
  -- Calculate the number of bits required to represent n
  sorry

theorem abcdef_hex_bits :
  bits_required (hex_to_decimal "ABCDEF") = 24 := by
  sorry

end abcdef_hex_bits_l3511_351119


namespace ac_length_l3511_351159

/-- Two triangles ABC and ADE are similar with given side lengths. -/
structure SimilarTriangles where
  AB : ℝ
  BC : ℝ
  CA : ℝ
  AD : ℝ
  DE : ℝ
  EA : ℝ
  similar : True  -- Represents that the triangles are similar
  h_AB : AB = 18
  h_BC : BC = 24
  h_CA : CA = 20
  h_AD : AD = 9
  h_DE : DE = 12
  h_EA : EA = 15

/-- The length of AC in the similar triangles is 20. -/
theorem ac_length (t : SimilarTriangles) : t.CA = 20 := by
  sorry

end ac_length_l3511_351159


namespace contrapositive_equivalence_l3511_351132

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
sorry

end contrapositive_equivalence_l3511_351132


namespace proposition_d_is_true_l3511_351166

theorem proposition_d_is_true (a b : ℝ) : a + b ≠ 5 → a ≠ 2 ∨ b ≠ 3 := by
  sorry

end proposition_d_is_true_l3511_351166


namespace complex_equation_solution_l3511_351120

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end complex_equation_solution_l3511_351120


namespace isosceles_triangle_perimeter_l3511_351113

-- Define an isosceles triangle with sides of lengths 3 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3)

-- Triangle inequality theorem
axiom triangle_inequality (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  IsoscelesTriangle a b c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 7 ∨ b = 7 ∨ c = 7) →
  a + b + c = 17 :=
sorry

end isosceles_triangle_perimeter_l3511_351113


namespace right_triangle_properties_l3511_351106

theorem right_triangle_properties (a b c h : ℝ) : 
  a = 12 → b = 5 → c^2 = a^2 + b^2 → (1/2) * a * b = (1/2) * c * h →
  c = 13 ∧ h = 60/13 := by
  sorry

end right_triangle_properties_l3511_351106


namespace fraction_denominator_l3511_351180

theorem fraction_denominator (x : ℕ) : 
  (4128 : ℚ) / x = 0.9411764705882353 → x = 4387 := by
  sorry

end fraction_denominator_l3511_351180


namespace symmetric_points_sum_l3511_351111

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

/-- The problem statement -/
theorem symmetric_points_sum (m n : ℝ) 
  (h : symmetric_wrt_origin (-3, m) (n, 2)) : 
  m + n = 1 := by
  sorry

end symmetric_points_sum_l3511_351111


namespace fishing_problem_solution_l3511_351136

/-- Represents the fishing problem scenario -/
structure FishingProblem where
  totalCatch : ℝ
  plannedDays : ℝ
  dailyCatch : ℝ
  stormDuration : ℝ
  stormCatchReduction : ℝ
  normalCatchIncrease : ℝ
  daysAheadOfSchedule : ℝ

/-- Theorem stating the solution to the fishing problem -/
theorem fishing_problem_solution (p : FishingProblem) 
  (h1 : p.totalCatch = 1800)
  (h2 : p.stormDuration = p.plannedDays / 3)
  (h3 : p.stormCatchReduction = 20)
  (h4 : p.normalCatchIncrease = 20)
  (h5 : p.daysAheadOfSchedule = 1)
  (h6 : p.plannedDays * p.dailyCatch = p.totalCatch)
  (h7 : p.stormDuration * (p.dailyCatch - p.stormCatchReduction) + 
        (p.plannedDays - p.stormDuration - p.daysAheadOfSchedule) * 
        (p.dailyCatch + p.normalCatchIncrease) = p.totalCatch) :
  p.dailyCatch = 100 := by
  sorry


end fishing_problem_solution_l3511_351136


namespace smallest_valid_number_last_four_digits_l3511_351165

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 7

def has_both_digits (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 5 = 0 ∧
    m % 7 = 0 ∧
    is_valid_number m ∧
    has_both_digits m ∧
    (∀ k : ℕ, k > 0 ∧ k % 5 = 0 ∧ k % 7 = 0 ∧ is_valid_number k ∧ has_both_digits k → m ≤ k) ∧
    last_four_digits m = 2772 :=
sorry

end smallest_valid_number_last_four_digits_l3511_351165


namespace swim_team_capacity_difference_l3511_351158

/-- Represents the number of each type of vehicle --/
structure Vehicles where
  cars : Nat
  vans : Nat
  minibuses : Nat

/-- Represents the maximum capacity of each type of vehicle --/
structure VehicleCapacities where
  car : Nat
  van : Nat
  minibus : Nat

/-- Represents the actual number of people in each vehicle --/
structure ActualOccupancy where
  car1 : Nat
  car2 : Nat
  van1 : Nat
  van2 : Nat
  van3 : Nat
  minibus : Nat

def vehicles : Vehicles := {
  cars := 2,
  vans := 3,
  minibuses := 1
}

def capacities : VehicleCapacities := {
  car := 6,
  van := 8,
  minibus := 15
}

def occupancy : ActualOccupancy := {
  car1 := 5,
  car2 := 4,
  van1 := 3,
  van2 := 3,
  van3 := 5,
  minibus := 10
}

def totalMaxCapacity (v : Vehicles) (c : VehicleCapacities) : Nat :=
  v.cars * c.car + v.vans * c.van + v.minibuses * c.minibus

def actualTotalOccupancy (o : ActualOccupancy) : Nat :=
  o.car1 + o.car2 + o.van1 + o.van2 + o.van3 + o.minibus

theorem swim_team_capacity_difference :
  totalMaxCapacity vehicles capacities - actualTotalOccupancy occupancy = 21 := by
  sorry

end swim_team_capacity_difference_l3511_351158


namespace max_abs_z_quadratic_equation_l3511_351126

open Complex

theorem max_abs_z_quadratic_equation (a b c z : ℂ) 
  (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : arg c = arg a + arg b)
  (h5 : a * z^2 + b * z + c = 0) :
  abs z ≤ (1 + Real.sqrt 5) / 2 :=
sorry

end max_abs_z_quadratic_equation_l3511_351126


namespace shaded_area_calculation_l3511_351169

theorem shaded_area_calculation (R : ℝ) (h : R = 9) :
  let r : ℝ := R / 2
  let larger_circle_area : ℝ := π * R^2
  let smaller_circle_area : ℝ := π * r^2
  let shaded_area : ℝ := larger_circle_area - 3 * smaller_circle_area
  shaded_area = 20.25 * π := by sorry

end shaded_area_calculation_l3511_351169


namespace additional_discount_percentage_l3511_351135

/-- Proves that the additional discount percentage for mothers with 3 or more children is 4% -/
theorem additional_discount_percentage : 
  let original_price : ℚ := 125
  let mothers_day_discount : ℚ := 10 / 100
  let final_price : ℚ := 108
  let price_after_initial_discount : ℚ := original_price * (1 - mothers_day_discount)
  let additional_discount_amount : ℚ := price_after_initial_discount - final_price
  let additional_discount_percentage : ℚ := additional_discount_amount / price_after_initial_discount * 100
  additional_discount_percentage = 4 := by sorry

end additional_discount_percentage_l3511_351135


namespace units_digit_problem_l3511_351192

theorem units_digit_problem : ∃ n : ℕ, (8 * 13 * 1989 - 8^3) % 10 = 4 ∧ n * 10 ≤ 8 * 13 * 1989 - 8^3 ∧ 8 * 13 * 1989 - 8^3 < (n + 1) * 10 := by
  sorry

end units_digit_problem_l3511_351192


namespace contractor_work_completion_l3511_351181

/-- Represents the problem of determining when 1/4 of the work was completed. -/
theorem contractor_work_completion (total_days : ℕ) (initial_workers : ℕ) (remaining_days : ℕ) (fired_workers : ℕ) : 
  total_days = 100 →
  initial_workers = 10 →
  remaining_days = 75 →
  fired_workers = 2 →
  ∃ (x : ℕ), 
    (x * initial_workers = remaining_days * (initial_workers - fired_workers)) ∧
    x = 60 :=
by sorry

end contractor_work_completion_l3511_351181


namespace second_shirt_price_l3511_351170

/-- Proves that the price of the second shirt must be $100 given the conditions --/
theorem second_shirt_price (total_shirts : Nat) (first_shirt_price third_shirt_price : ℝ)
  (remaining_shirts_min_avg : ℝ) (overall_avg : ℝ) :
  total_shirts = 10 →
  first_shirt_price = 82 →
  third_shirt_price = 90 →
  remaining_shirts_min_avg = 104 →
  overall_avg = 100 →
  ∃ (second_shirt_price : ℝ),
    second_shirt_price = 100 ∧
    (first_shirt_price + second_shirt_price + third_shirt_price +
      (total_shirts - 3) * remaining_shirts_min_avg) / total_shirts ≥ overall_avg :=
by sorry

end second_shirt_price_l3511_351170


namespace range_of_m_l3511_351107

/-- The quadratic function p(x) = x^2 + 2x - m -/
def p (m : ℝ) (x : ℝ) : Prop := x^2 + 2*x - m > 0

/-- Given p(x): x^2 + 2x - m > 0, if p(1) is false and p(2) is true, 
    then the range of values for m is [3, 8) -/
theorem range_of_m (m : ℝ) : 
  (¬ p m 1) ∧ (p m 2) → 3 ≤ m ∧ m < 8 := by
  sorry

end range_of_m_l3511_351107


namespace cube_painting_theorem_l3511_351160

/-- The number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of available colors -/
def available_colors : ℕ := 7

/-- The number of distinguishable ways to paint a cube -/
def distinguishable_cubes : ℕ := 210

theorem cube_painting_theorem :
  (Nat.choose available_colors cube_faces * Nat.factorial cube_faces) / cube_symmetries = distinguishable_cubes :=
sorry

end cube_painting_theorem_l3511_351160


namespace train_crossing_time_l3511_351128

/-- The time taken for a train to cross a man running in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 450 ∧ 
  train_speed = 60 * (1000 / 3600) ∧ 
  man_speed = 6 * (1000 / 3600) → 
  train_length / (train_speed - man_speed) = 30 := by
  sorry

end train_crossing_time_l3511_351128


namespace total_flowers_l3511_351133

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) (h1 : num_pots = 141) (h2 : flowers_per_pot = 71) : 
  num_pots * flowers_per_pot = 10011 := by
  sorry

end total_flowers_l3511_351133


namespace remaining_distance_l3511_351148

def total_distance : ℝ := 300
def speed : ℝ := 60
def time : ℝ := 2

theorem remaining_distance : total_distance - speed * time = 180 := by
  sorry

end remaining_distance_l3511_351148


namespace probability_three_blue_marbles_l3511_351171

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 4

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := 5

/-- The number of white marbles in the jar -/
def white_marbles : ℕ := 8

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + blue_marbles + white_marbles + green_marbles

/-- The number of marbles drawn -/
def marbles_drawn : ℕ := 3

theorem probability_three_blue_marbles :
  (blue_marbles : ℚ) / total_marbles *
  ((blue_marbles - 1) : ℚ) / (total_marbles - 1) *
  ((blue_marbles - 2) : ℚ) / (total_marbles - 2) = 1 / 114 :=
by sorry

end probability_three_blue_marbles_l3511_351171


namespace garden_width_l3511_351137

theorem garden_width (playground_side : ℕ) (garden_length : ℕ) (total_fencing : ℕ) :
  playground_side = 27 →
  garden_length = 12 →
  total_fencing = 150 →
  4 * playground_side + 2 * garden_length + 2 * (total_fencing - 4 * playground_side - 2 * garden_length) / 2 = 150 →
  (total_fencing - 4 * playground_side - 2 * garden_length) / 2 = 9 := by
  sorry

#check garden_width

end garden_width_l3511_351137


namespace motorcyclist_distance_l3511_351189

/-- Represents the motion of a motorcyclist --/
structure Motion where
  initial_speed : ℝ
  acceleration : ℝ
  time_to_b : ℝ
  time_b_to_c : ℝ
  speed_at_c : ℝ

/-- Calculates the distance between points A and C --/
def distance_a_to_c (m : Motion) : ℝ :=
  let speed_at_b := m.initial_speed + m.acceleration * m.time_to_b
  let distance_a_to_b := m.initial_speed * m.time_to_b + 0.5 * m.acceleration * m.time_to_b^2
  let distance_b_to_c := speed_at_b * m.time_b_to_c - 0.5 * m.acceleration * m.time_b_to_c^2
  distance_a_to_b - distance_b_to_c

/-- The main theorem to prove --/
theorem motorcyclist_distance (m : Motion) 
  (h1 : m.initial_speed = 90)
  (h2 : m.time_to_b = 3)
  (h3 : m.time_b_to_c = 2)
  (h4 : m.speed_at_c = 110)
  (h5 : m.acceleration = (m.speed_at_c - m.initial_speed) / (m.time_to_b + m.time_b_to_c)) :
  distance_a_to_c m = 92 := by
  sorry


end motorcyclist_distance_l3511_351189


namespace no_leftover_eggs_l3511_351157

/-- The number of eggs Abigail has -/
def abigail_eggs : ℕ := 58

/-- The number of eggs Beatrice has -/
def beatrice_eggs : ℕ := 35

/-- The number of eggs Carson has -/
def carson_eggs : ℕ := 27

/-- The size of each egg carton -/
def carton_size : ℕ := 10

/-- The theorem stating that there are no leftover eggs -/
theorem no_leftover_eggs : (abigail_eggs + beatrice_eggs + carson_eggs) % carton_size = 0 := by
  sorry

end no_leftover_eggs_l3511_351157


namespace quadratic_equations_solutions_l3511_351167

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 - 4 * x - 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x - 3)^2 = 3 * x * (x - 3)
  let sol1 : Set ℝ := {(2 + Real.sqrt 6) / 2, (2 - Real.sqrt 6) / 2}
  let sol2 : Set ℝ := {3, -3/2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) := by
  sorry

#check quadratic_equations_solutions

end quadratic_equations_solutions_l3511_351167


namespace company_employees_l3511_351164

theorem company_employees (female_managers : ℕ) (male_female_ratio : ℚ) 
  (total_manager_ratio : ℚ) (male_manager_ratio : ℚ) :
  female_managers = 200 →
  male_female_ratio = 3 / 2 →
  total_manager_ratio = 2 / 5 →
  male_manager_ratio = 2 / 5 →
  ∃ (female_employees : ℕ) (total_employees : ℕ),
    female_employees = 500 ∧
    total_employees = 1250 := by
  sorry

end company_employees_l3511_351164


namespace ice_cream_volume_l3511_351174

/-- The volume of ice cream in a cone and hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (1/2) * (4/3) * π * r^3
  h = 8 ∧ r = 2 → cone_volume + hemisphere_volume = 16 * π := by sorry

end ice_cream_volume_l3511_351174


namespace dividend_division_theorem_l3511_351138

theorem dividend_division_theorem : ∃ (q r : ℕ), 
  220025 = (555 + 445) * q + r ∧ 
  r < (555 + 445) ∧ 
  r = 25 ∧ 
  q = 2 * (555 - 445) := by
  sorry

end dividend_division_theorem_l3511_351138


namespace remainder_of_x_l3511_351147

theorem remainder_of_x (x : ℤ) 
  (h1 : (2 + x) % (4^3) = 3^2 % (4^3))
  (h2 : (3 + x) % (5^3) = 5^2 % (5^3))
  (h3 : (6 + x) % (11^3) = 7^2 % (11^3)) :
  x % 220 = 192 := by
sorry

end remainder_of_x_l3511_351147


namespace popsicle_stick_sum_l3511_351117

theorem popsicle_stick_sum : 
  ∀ (gino ana sam speaker : ℕ),
    gino = 63 →
    ana = 128 →
    sam = 75 →
    speaker = 50 →
    gino + ana + sam + speaker = 316 :=
by
  sorry

end popsicle_stick_sum_l3511_351117


namespace intersection_points_polar_equations_l3511_351115

/-- The number of intersection points between r = 3 cos θ and r = 6 sin θ -/
theorem intersection_points_polar_equations : ∃ (n : ℕ), n = 2 ∧
  ∀ (x y : ℝ),
    ((x - 3/2)^2 + y^2 = 9/4 ∨ x^2 + (y - 3)^2 = 9) →
    (∃ (θ : ℝ), 
      (x = 3 * Real.cos θ * Real.cos θ ∧ y = 3 * Real.sin θ * Real.cos θ) ∨
      (x = 6 * Real.sin θ * Real.cos θ ∧ y = 6 * Real.sin θ * Real.sin θ)) :=
by sorry


end intersection_points_polar_equations_l3511_351115


namespace quadratic_discriminant_l3511_351187

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 5x^2 - 9x + 2 has discriminant 41 -/
theorem quadratic_discriminant : discriminant 5 (-9) 2 = 41 := by
  sorry

end quadratic_discriminant_l3511_351187


namespace equation_one_solution_equation_two_no_solution_l3511_351124

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  x / (2 * x - 5) + 5 / (5 - 2 * x) = 1 ↔ x = 0 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
sorry

end equation_one_solution_equation_two_no_solution_l3511_351124


namespace polygon_contains_half_unit_segment_l3511_351114

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon
  area : ℝ
  isConvex : Bool

/-- A square with side length 1 -/
structure UnitSquare where
  -- Add necessary fields for a unit square

/-- Represents the placement of a polygon inside a square -/
structure PolygonInSquare where
  polygon : ConvexPolygon
  square : UnitSquare
  isInside : Bool

/-- A line segment -/
structure LineSegment where
  length : ℝ
  isParallelToSquareSide : Bool
  isInsidePolygon : Bool

/-- The main theorem -/
theorem polygon_contains_half_unit_segment 
  (p : PolygonInSquare) 
  (h1 : p.polygon.area > 0.5) 
  (h2 : p.polygon.isConvex) 
  (h3 : p.isInside) :
  ∃ (s : LineSegment), s.length = 0.5 ∧ s.isParallelToSquareSide ∧ s.isInsidePolygon :=
by sorry

end polygon_contains_half_unit_segment_l3511_351114


namespace light_glows_165_times_l3511_351110

/-- Represents the glow pattern of the light in seconds -/
def glowPattern : List Nat := [15, 25, 35, 45]

/-- Calculates the total seconds in the glow pattern -/
def patternDuration : Nat := glowPattern.sum

/-- Converts a time in hours, minutes, seconds to total seconds -/
def timeToSeconds (hours minutes seconds : Nat) : Nat :=
  hours * 3600 + minutes * 60 + seconds

/-- Calculates the duration between two times in seconds -/
def durationBetween (startHours startMinutes startSeconds endHours endMinutes endSeconds : Nat) : Nat :=
  timeToSeconds endHours endMinutes endSeconds - timeToSeconds startHours startMinutes startSeconds

/-- Calculates the number of complete cycles in a given duration -/
def completeCycles (duration : Nat) : Nat :=
  duration / patternDuration

/-- Calculates the remaining seconds after complete cycles -/
def remainingSeconds (duration : Nat) : Nat :=
  duration % patternDuration

/-- Counts the number of glows in the remaining seconds -/
def countRemainingGlows (seconds : Nat) : Nat :=
  glowPattern.foldl (fun count interval => if seconds ≥ interval then count + 1 else count) 0

/-- Theorem: The light glows 165 times between 1:57:58 AM and 3:20:47 AM -/
theorem light_glows_165_times : 
  (completeCycles (durationBetween 1 57 58 3 20 47) * glowPattern.length) + 
  countRemainingGlows (remainingSeconds (durationBetween 1 57 58 3 20 47)) = 165 := by
  sorry


end light_glows_165_times_l3511_351110


namespace inscribed_square_side_length_l3511_351162

/-- The side length of a square inscribed in a right triangle with sides 6, 8, and 10 -/
def inscribedSquareSideLength : ℚ := 60 / 31

/-- Right triangle ABC with square XYZW inscribed -/
structure InscribedSquare where
  -- Triangle side lengths
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Square side length
  s : ℝ
  -- Conditions
  right_triangle : AB^2 + BC^2 = AC^2
  AB_eq : AB = 6
  BC_eq : BC = 8
  AC_eq : AC = 10
  inscribed : s > 0 -- The square is inscribed (side length is positive)
  on_AC : s ≤ AC -- X and Y are on AC
  on_AB : s ≤ AB -- W is on AB
  on_BC : s ≤ BC -- Z is on BC

/-- The side length of the inscribed square is equal to 60/31 -/
theorem inscribed_square_side_length (square : InscribedSquare) :
  square.s = inscribedSquareSideLength := by sorry

end inscribed_square_side_length_l3511_351162


namespace system_solution_l3511_351150

theorem system_solution (a b x y : ℝ) 
  (h1 : (x - y) / (1 - x * y) = 2 * a / (1 + a^2))
  (h2 : (x + y) / (1 + x * y) = 2 * b / (1 + b^2))
  (ha : a^2 ≠ 1)
  (hb : b^2 ≠ 1)
  (hab : a ≠ b)
  (hnr : a * b ≠ 1) :
  ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨
   (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1))) := by
  sorry

end system_solution_l3511_351150


namespace lcm_from_product_and_hcf_l3511_351191

theorem lcm_from_product_and_hcf (a b : ℕ+) (h1 : a * b = 987153000) (h2 : Nat.gcd a b = 440) :
  Nat.lcm a b = 2243525 := by
  sorry

end lcm_from_product_and_hcf_l3511_351191


namespace isosceles_triangle_cosine_l3511_351145

/-- Theorem: In an isosceles triangle with two sides of length 3 and the third side of length √15 - √3,
    the cosine of the angle opposite the third side is equal to √5/3. -/
theorem isosceles_triangle_cosine (a b c : ℝ) (h1 : a = 3) (h2 : b = 3) (h3 : c = Real.sqrt 15 - Real.sqrt 3) :
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  cosC = Real.sqrt 5 / 3 := by
  sorry

end isosceles_triangle_cosine_l3511_351145


namespace euston_carriages_l3511_351161

/-- The number of carriages in different towns --/
structure Carriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions of the carriage problem --/
def carriage_conditions (c : Carriages) : Prop :=
  c.euston = c.norfolk + 20 ∧
  c.norwich = 100 ∧
  c.flying_scotsman = c.norwich + 20 ∧
  c.euston + c.norfolk + c.norwich + c.flying_scotsman = 460

/-- Theorem stating that under the given conditions, Euston had 130 carriages --/
theorem euston_carriages (c : Carriages) (h : carriage_conditions c) : c.euston = 130 := by
  sorry

end euston_carriages_l3511_351161


namespace train_bus_cost_difference_l3511_351129

/-- The cost difference between a train ride and a bus ride -/
def cost_difference (train_cost bus_cost : ℝ) : ℝ := train_cost - bus_cost

/-- Theorem stating the cost difference between a train ride and a bus ride -/
theorem train_bus_cost_difference :
  ∀ (train_cost bus_cost : ℝ),
    train_cost > bus_cost →
    train_cost + bus_cost = 9.85 →
    bus_cost = 1.75 →
    cost_difference train_cost bus_cost = 6.35 := by
  sorry

end train_bus_cost_difference_l3511_351129


namespace gerald_wood_pieces_l3511_351134

/-- The number of pieces of wood needed to make a table -/
def wood_per_table : ℕ := 12

/-- The number of pieces of wood needed to make a chair -/
def wood_per_chair : ℕ := 8

/-- The number of chairs Gerald can make -/
def chairs : ℕ := 48

/-- The number of tables Gerald can make -/
def tables : ℕ := 24

/-- Theorem stating the total number of wood pieces Gerald has -/
theorem gerald_wood_pieces : 
  wood_per_table * tables + wood_per_chair * chairs = 672 := by
  sorry

end gerald_wood_pieces_l3511_351134


namespace distinct_triangles_in_2x4_grid_l3511_351199

/-- Represents a 2x4 grid of points -/
def Grid := Fin 2 × Fin 4

/-- Checks if three points in the grid are collinear -/
def collinear (p q r : Grid) : Prop := sorry

/-- The number of ways to choose 3 points from 8 points -/
def total_combinations : ℕ := Nat.choose 8 3

/-- The number of collinear triples in a 2x4 grid -/
def collinear_triples : ℕ := sorry

/-- The number of distinct triangles in a 2x4 grid -/
def distinct_triangles : ℕ := total_combinations - collinear_triples

theorem distinct_triangles_in_2x4_grid :
  distinct_triangles = 44 := by sorry

end distinct_triangles_in_2x4_grid_l3511_351199


namespace min_value_of_x2_plus_2y2_l3511_351186

theorem min_value_of_x2_plus_2y2 (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 2) :
  ∃ (m : ℝ), m = 4 - 2*Real.sqrt 2 ∧ ∀ (a b : ℝ), a^2 - 2*a*b + 2*b^2 = 2 → x^2 + 2*y^2 ≥ m :=
by sorry

end min_value_of_x2_plus_2y2_l3511_351186
