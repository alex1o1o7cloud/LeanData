import Mathlib

namespace most_likely_red_balls_l1901_190154

theorem most_likely_red_balls 
  (total_balls : ℕ) 
  (red_frequency : ℝ) 
  (h1 : total_balls = 20) 
  (h2 : 0 ≤ red_frequency ∧ red_frequency ≤ 1) 
  (h3 : red_frequency = 0.8) : 
  ⌊total_balls * red_frequency⌋ = 16 := by
sorry

end most_likely_red_balls_l1901_190154


namespace job_completion_time_l1901_190193

/-- Given two people working on a job, where the first person takes 3 hours and their combined
    work rate is 5/12 of the job per hour, prove that the second person takes 12 hours to
    complete the job individually. -/
theorem job_completion_time
  (time_person1 : ℝ)
  (combined_rate : ℝ)
  (h1 : time_person1 = 3)
  (h2 : combined_rate = 5 / 12)
  : ∃ (time_person2 : ℝ),
    time_person2 = 12 ∧
    1 / time_person1 + 1 / time_person2 = combined_rate :=
by sorry

end job_completion_time_l1901_190193


namespace expected_wealth_difference_10_days_l1901_190160

/-- Represents the daily outcome for an agent --/
inductive DailyOutcome
  | Win
  | Lose
  | Reset

/-- Represents the state of wealth for both agents --/
structure WealthState :=
  (cat : ℤ)
  (fox : ℤ)

/-- Defines the probability distribution for daily outcomes --/
def dailyProbability : DailyOutcome → ℝ
  | DailyOutcome.Win => 0.25
  | DailyOutcome.Lose => 0.25
  | DailyOutcome.Reset => 0.5

/-- Updates the wealth state based on the daily outcome --/
def updateWealth (state : WealthState) (outcome : DailyOutcome) : WealthState :=
  match outcome with
  | DailyOutcome.Win => { cat := state.cat + 1, fox := state.fox }
  | DailyOutcome.Lose => { cat := state.cat, fox := state.fox + 1 }
  | DailyOutcome.Reset => { cat := 0, fox := 0 }

/-- Calculates the expected value of the absolute difference in wealth after n days --/
def expectedWealthDifference (n : ℕ) : ℝ :=
  sorry

theorem expected_wealth_difference_10_days :
  expectedWealthDifference 10 = 1 :=
sorry

end expected_wealth_difference_10_days_l1901_190160


namespace only_5_6_10_forms_triangle_l1901_190107

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: Among the given sets, only (5, 6, 10) can form a triangle -/
theorem only_5_6_10_forms_triangle :
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 5 6 11 ∧
  can_form_triangle 5 6 10 ∧
  ¬ can_form_triangle 4 4 8 :=
sorry

end only_5_6_10_forms_triangle_l1901_190107


namespace geometric_sequence_product_l1901_190157

/-- Given real numbers x, y, and z forming a geometric sequence with -1 and -3,
    prove that their product equals -3√3 -/
theorem geometric_sequence_product (x y z : ℝ) 
  (h1 : ∃ (r : ℝ), x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * y * z = -3 * Real.sqrt 3 := by
  sorry

end geometric_sequence_product_l1901_190157


namespace thirtieth_term_of_sequence_l1901_190196

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 4) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 119 :=
by sorry

end thirtieth_term_of_sequence_l1901_190196


namespace three_questions_identify_l1901_190153

/-- Represents a geometric figure -/
inductive GeometricFigure
  | Circle
  | Ellipse
  | Triangle
  | Square
  | Rectangle
  | Parallelogram
  | Trapezoid

/-- Represents a yes/no question about a geometric figure -/
def Question := GeometricFigure → Bool

/-- The set of all geometric figures -/
def FigureSet : Set GeometricFigure := {
  GeometricFigure.Circle,
  GeometricFigure.Ellipse,
  GeometricFigure.Triangle,
  GeometricFigure.Square,
  GeometricFigure.Rectangle,
  GeometricFigure.Parallelogram,
  GeometricFigure.Trapezoid
}

/-- A sequence of three questions -/
structure ThreeQuestions where
  q1 : Question
  q2 : Question
  q3 : Question

/-- Checks if a sequence of three questions can uniquely identify a figure -/
def canIdentify (qs : ThreeQuestions) (f : GeometricFigure) : Prop :=
  ∀ g : GeometricFigure, g ∈ FigureSet →
    (qs.q1 f = qs.q1 g ∧ qs.q2 f = qs.q2 g ∧ qs.q3 f = qs.q3 g) → f = g

/-- The main theorem: there exists a sequence of three questions that can identify any figure -/
theorem three_questions_identify :
  ∃ qs : ThreeQuestions, ∀ f : GeometricFigure, f ∈ FigureSet → canIdentify qs f := by
  sorry


end three_questions_identify_l1901_190153


namespace total_population_l1901_190136

theorem total_population (b g t : ℕ) : 
  b = 4 * g ∧ g = 8 * t → b + g + t = 41 * t :=
by sorry

end total_population_l1901_190136


namespace power_mod_23_l1901_190189

theorem power_mod_23 : 17^2001 % 23 = 11 := by
  sorry

end power_mod_23_l1901_190189


namespace orange_shelves_l1901_190129

/-- The number of oranges on the nth shelf -/
def oranges_on_shelf (n : ℕ) : ℕ := 3 + 5 * (n - 1)

/-- The total number of oranges on n shelves -/
def total_oranges (n : ℕ) : ℕ := n * (oranges_on_shelf 1 + oranges_on_shelf n) / 2

theorem orange_shelves :
  ∃ n : ℕ, n > 0 ∧ total_oranges n = 325 :=
sorry

end orange_shelves_l1901_190129


namespace fraction_multiplication_invariance_l1901_190128

theorem fraction_multiplication_invariance (a b m : ℝ) (h : b ≠ 0) :
  ∀ x : ℝ, (a * (x - m)) / (b * (x - m)) = a / b ↔ x ≠ m :=
sorry

end fraction_multiplication_invariance_l1901_190128


namespace probability_of_red_ball_l1901_190115

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the number of black balls
def black_balls : ℕ := 6

-- Theorem statement
theorem probability_of_red_ball :
  (red_balls : ℚ) / total_balls = 2 / 5 :=
by sorry

end probability_of_red_ball_l1901_190115


namespace cubic_fraction_equals_10_04_l1901_190112

theorem cubic_fraction_equals_10_04 :
  let a : ℝ := 6
  let b : ℝ := 3
  let c : ℝ := 2
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2) = 10.04 := by
  sorry

end cubic_fraction_equals_10_04_l1901_190112


namespace unique_solution_l1901_190172

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (heq : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
sorry

end unique_solution_l1901_190172


namespace partition_infinite_multiples_l1901_190120

-- Define a partition of Natural Numbers
def Partition (A : ℕ → Set ℕ) (k : ℕ) : Prop :=
  (∀ n, ∃! i, i ≤ k ∧ n ∈ A i) ∧
  (∀ i, i ≤ k → Set.Nonempty (A i))

-- Define what it means for a set to contain infinitely many multiples of a number
def InfiniteMultiples (S : Set ℕ) (x : ℕ) : Prop :=
  Set.Infinite {n ∈ S | ∃ k, n = k * x}

-- Main theorem
theorem partition_infinite_multiples 
  {A : ℕ → Set ℕ} {k : ℕ} (h : Partition A k) :
  ∃ i, i ≤ k ∧ ∀ x : ℕ, x > 0 → InfiniteMultiples (A i) x :=
sorry

end partition_infinite_multiples_l1901_190120


namespace desk_purchase_price_l1901_190108

/-- Given a desk with a selling price that includes a 25% markup and results in a gross profit of $33.33, prove that the purchase price of the desk is $99.99. -/
theorem desk_purchase_price (purchase_price selling_price : ℝ) : 
  selling_price = purchase_price + 0.25 * selling_price →
  selling_price - purchase_price = 33.33 →
  purchase_price = 99.99 := by
sorry

end desk_purchase_price_l1901_190108


namespace cubic_roots_relationship_l1901_190127

/-- The cubic polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- The cubic polynomial h(x) -/
def h (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem stating the relationship between f and h and the values of a, b, and c -/
theorem cubic_roots_relationship (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ x : ℝ, f x = 0 → h a b c (x^3) = 0) →
  a = -6 ∧ b = -9 ∧ c = 20 := by
sorry

end cubic_roots_relationship_l1901_190127


namespace arithmetic_sequence_common_difference_l1901_190191

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_a1 : a 1 = 2)  -- given a_1 = 2
  (h_a3 : a 3 = 8)  -- given a_3 = 8
  : a 2 - a 1 = 3 :=  -- prove that the common difference is 3
by
  sorry

end arithmetic_sequence_common_difference_l1901_190191


namespace isosceles_triangle_from_lines_l1901_190100

/-- Given three lines in a plane, prove that they form an isosceles triangle -/
theorem isosceles_triangle_from_lines :
  let line1 : ℝ → ℝ := λ x => 4 * x + 3
  let line2 : ℝ → ℝ := λ x => -4 * x + 3
  let line3 : ℝ → ℝ := λ _ => -3
  let point1 : ℝ × ℝ := (0, 3)
  let point2 : ℝ × ℝ := (-3/2, -3)
  let point3 : ℝ × ℝ := (3/2, -3)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  true →
  ∃ (a b : ℝ), a = distance point1 point2 ∧ 
                a = distance point1 point3 ∧ 
                b = distance point2 point3 ∧
                a ≠ b :=
by sorry

end isosceles_triangle_from_lines_l1901_190100


namespace people_left_of_kolya_l1901_190161

/-- Given a class lineup with the following conditions:
  * There are 12 people to the right of Kolya
  * There are 20 people to the left of Sasha
  * There are 8 people to the right of Sasha
  Prove that there are 16 people to the left of Kolya -/
theorem people_left_of_kolya
  (right_of_kolya : ℕ)
  (left_of_sasha : ℕ)
  (right_of_sasha : ℕ)
  (h1 : right_of_kolya = 12)
  (h2 : left_of_sasha = 20)
  (h3 : right_of_sasha = 8) :
  left_of_sasha + right_of_sasha + 1 - right_of_kolya - 1 = 16 := by
  sorry

end people_left_of_kolya_l1901_190161


namespace existence_of_counterexample_l1901_190146

theorem existence_of_counterexample : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ |a - b| + (1 / (a - b)) < 2 := by
  sorry

end existence_of_counterexample_l1901_190146


namespace correct_quotient_calculation_l1901_190111

theorem correct_quotient_calculation (dividend : ℕ) (incorrect_quotient : ℕ) : 
  dividend % 21 = 0 →
  dividend = 12 * incorrect_quotient →
  incorrect_quotient = 56 →
  dividend / 21 = 32 := by
sorry

end correct_quotient_calculation_l1901_190111


namespace shirt_difference_l1901_190140

theorem shirt_difference (alex_shirts joe_shirts ben_shirts : ℕ) : 
  alex_shirts = 4 → 
  ben_shirts = 15 → 
  ben_shirts = joe_shirts + 8 → 
  joe_shirts - alex_shirts = 3 := by
sorry

end shirt_difference_l1901_190140


namespace prob_more_heads_than_tails_fair_coin_l1901_190192

/-- A fair coin is a coin with equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability of getting more heads than tails in 3 flips of a fair coin -/
def prob_more_heads_than_tails (p : ℝ) : ℝ :=
  prob_k_heads 3 2 p + prob_k_heads 3 3 p

theorem prob_more_heads_than_tails_fair_coin :
  ∀ p : ℝ, fair_coin p → prob_more_heads_than_tails p = 1/2 :=
by sorry

end prob_more_heads_than_tails_fair_coin_l1901_190192


namespace simplify_complex_expression_l1901_190179

theorem simplify_complex_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 5)) =
  Real.sqrt (224 - 22 * Real.sqrt 105) / 8 := by
  sorry

end simplify_complex_expression_l1901_190179


namespace smallest_solution_of_equation_l1901_190118

theorem smallest_solution_of_equation :
  let f : ℝ → ℝ := λ x => 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ x : ℝ, f x = 0 ∧ x = 5 - 2 * Real.sqrt 2 ∧ ∀ y : ℝ, f y = 0 → y ≥ x := by
  sorry

end smallest_solution_of_equation_l1901_190118


namespace sum_mod_nine_l1901_190183

theorem sum_mod_nine : (2469 + 2470 + 2471 + 2472 + 2473 + 2474) % 9 = 6 := by
  sorry

end sum_mod_nine_l1901_190183


namespace some_number_value_l1901_190137

theorem some_number_value (some_number : ℝ) : 
  |9 - 8 * (3 - some_number)| - |5 - 11| = 75 → some_number = 12 := by
  sorry

end some_number_value_l1901_190137


namespace solve_abc_l1901_190180

def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

theorem solve_abc (a b c : ℝ) :
  A a ≠ B b c ∧
  A a ∩ B b c = {-3} ∧
  A a ∪ B b c = {-3, 1, 4} →
  a = -1 ∧ b = 2 ∧ c = -3 := by
sorry

end solve_abc_l1901_190180


namespace domain_of_f_l1901_190162

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((Real.log x - 2) * (x - Real.log x - 1))

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {1} ∪ Set.Ici (Real.exp 2) := by sorry

end domain_of_f_l1901_190162


namespace apples_given_theorem_l1901_190181

/-- Represents the number of apples Joan gave to Melanie -/
def apples_given_to_melanie (initial_apples current_apples : ℕ) : ℕ :=
  initial_apples - current_apples

theorem apples_given_theorem (initial_apples current_apples : ℕ) 
  (h1 : initial_apples = 43)
  (h2 : current_apples = 16) :
  apples_given_to_melanie initial_apples current_apples = 27 := by
  sorry

end apples_given_theorem_l1901_190181


namespace product_of_numbers_l1901_190167

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 16) : x * y = 836 := by
  sorry

end product_of_numbers_l1901_190167


namespace unique_solution_fourth_root_equation_l1901_190134

/-- The equation √⁴(58 - 3x) + √⁴(26 + 3x) = 5 has a unique solution -/
theorem unique_solution_fourth_root_equation :
  ∃! x : ℝ, (58 - 3*x)^(1/4) + (26 + 3*x)^(1/4) = 5 :=
by sorry

end unique_solution_fourth_root_equation_l1901_190134


namespace polynomial_division_remainder_l1901_190147

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (3 * X ^ 3 - 4 * X ^ 2 + 17 * X + 34 : Polynomial ℤ) = 
  (X - 7) * q + 986 := by
  sorry

end polynomial_division_remainder_l1901_190147


namespace perpendicular_planes_from_lines_l1901_190121

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_lines_l1901_190121


namespace functional_equation_solution_l1901_190101

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x ∨ f x = 2 - x) :=
by sorry

end functional_equation_solution_l1901_190101


namespace cutting_theorem_l1901_190188

/-- Represents a string of pearls -/
structure PearlString where
  color : Bool  -- true for black, false for white
  length : Nat

/-- State of the cutting process -/
structure CuttingState where
  strings : List PearlString

/-- Cutting rules -/
def cut_strings (k : Nat) (state : CuttingState) : CuttingState := sorry

/-- Predicate to check if a state has a white pearl of length 1 -/
def has_single_white_pearl (state : CuttingState) : Prop := sorry

/-- Predicate to check if a state has a black pearl string of length > 1 -/
def has_multiple_black_pearls (state : CuttingState) : Prop := sorry

/-- The cutting process -/
def cutting_process (k : Nat) (b w : Nat) : CuttingState := sorry

/-- Main theorem -/
theorem cutting_theorem (k : Nat) (b w : Nat) 
  (h1 : k > 0) (h2 : b > w) (h3 : w > 1) :
  let final_state := cutting_process k b w
  has_single_white_pearl final_state → has_multiple_black_pearls final_state :=
by
  sorry


end cutting_theorem_l1901_190188


namespace age_difference_proof_l1901_190186

theorem age_difference_proof (patrick michael monica : ℕ) 
  (h1 : patrick * 5 = michael * 3)  -- Patrick and Michael's age ratio
  (h2 : michael * 4 = monica * 3)   -- Michael and Monica's age ratio
  (h3 : patrick + michael + monica = 88) -- Sum of ages
  : monica - patrick = 22 := by
  sorry

end age_difference_proof_l1901_190186


namespace function_extrema_l1901_190131

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - 1/x + 1/x^2

theorem function_extrema (a : ℝ) :
  a ≠ 0 →
  (∃ (xmax xmin : ℝ), xmax > 0 ∧ xmin > 0 ∧
    (∀ x > 0, f a x ≤ f a xmax) ∧
    (∀ x > 0, f a x ≥ f a xmin)) ↔
  -1/8 < a ∧ a < 0 :=
by sorry

end function_extrema_l1901_190131


namespace inequality_system_integer_solutions_l1901_190142

theorem inequality_system_integer_solutions (x : ℤ) :
  (2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1) ↔ (x = 3 ∨ x = 4 ∨ x = 5) := by
  sorry

end inequality_system_integer_solutions_l1901_190142


namespace light_reflection_l1901_190113

-- Define points M and P
def M : ℝ × ℝ := (-1, 3)
def P : ℝ × ℝ := (1, 0)

-- Define the reflecting lines
def x_axis (x y : ℝ) : Prop := y = 0
def reflecting_line (x y : ℝ) : Prop := x + y = 4

-- Define the light rays
def l1 : ℝ × ℝ → Prop := sorry
def l2 : ℝ × ℝ → Prop := sorry
def l3 : ℝ × ℝ → Prop := sorry

-- Define the reflection operation
def reflect (line : ℝ × ℝ → Prop) (ray : ℝ × ℝ → Prop) : ℝ × ℝ → Prop := sorry

-- State the theorem
theorem light_reflection :
  (∀ x y, l2 (x, y) ↔ y = 3/2 * (x - 1)) ∧
  (∀ x y, l3 (x, y) ↔ 2*x - 3*y + 1 = 0) := by
  sorry

end light_reflection_l1901_190113


namespace smallest_n_satisfying_conditions_l1901_190178

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 5 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m % 8 = 5 → m ≥ n) ∧
  n = 220 :=
by sorry

end smallest_n_satisfying_conditions_l1901_190178


namespace absolute_value_inequality_l1901_190182

theorem absolute_value_inequality (x : ℝ) : 
  3 ≤ |x - 3| ∧ |x - 3| ≤ 7 ↔ ((-4 ≤ x ∧ x ≤ 0) ∨ (6 ≤ x ∧ x ≤ 10)) := by
  sorry

end absolute_value_inequality_l1901_190182


namespace cake_payment_dimes_l1901_190197

/-- Represents the number of each type of coin used in the payment -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The value of the payment in cents -/
def payment_value (c : CoinCount) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

theorem cake_payment_dimes :
  ∃ (c : CoinCount),
    c.pennies + c.nickels + c.dimes = 50 ∧
    payment_value c = 200 ∧
    c.dimes = 14 := by
  sorry

end cake_payment_dimes_l1901_190197


namespace vector_properties_l1901_190125

-- Define plane vectors a and b
variable (a b : ℝ × ℝ)

-- Define the conditions
def condition1 : Prop := norm a = 1
def condition2 : Prop := norm b = 1
def condition3 : Prop := norm (2 • a + b) = Real.sqrt 6

-- Define the theorem
theorem vector_properties (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) :
  (a • b = 1/4) ∧ (norm (a + b) = Real.sqrt 10 / 2) := by
  sorry

end vector_properties_l1901_190125


namespace jimmy_stair_climbing_l1901_190156

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stair_climbing : arithmetic_sum 30 10 8 = 520 := by
  sorry

end jimmy_stair_climbing_l1901_190156


namespace stating_trapezoid_equal_area_segment_length_trapezoid_floor_x_squared_div_150_l1901_190195

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  -- Length of the shorter base
  b : ℝ
  -- Height of the trapezoid
  h : ℝ
  -- The segment joining midpoints of legs divides area in 3:4 ratio
  midpoint_divides_area : (2 * b + 75) / (2 * b + 225) = 3 / 4
  -- Length of segment parallel to bases dividing area equally
  x : ℝ
  -- x divides the trapezoid into two equal areas
  x_divides_equally : x * (b + x / 2) = 150

/-- 
Theorem stating that for a trapezoid with the given properties,
the length of the segment dividing it into equal areas is 75.
-/
theorem trapezoid_equal_area_segment_length (t : Trapezoid) : t.x = 75 := by
  sorry

/-- 
Corollary stating that the greatest integer not exceeding x^2/150 is 37.
-/
theorem trapezoid_floor_x_squared_div_150 (t : Trapezoid) : 
  ⌊(t.x^2 / 150 : ℝ)⌋ = 37 := by
  sorry

end stating_trapezoid_equal_area_segment_length_trapezoid_floor_x_squared_div_150_l1901_190195


namespace max_z_value_l1901_190187

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 3) (prod_eq : x*y + y*z + z*x = 2) :
  z ≤ 5/3 ∧ ∃ (x' y' z' : ℝ), x' + y' + z' = 3 ∧ x'*y' + y'*z' + z'*x' = 2 ∧ z' = 5/3 :=
by sorry

end max_z_value_l1901_190187


namespace largest_common_divisor_462_330_l1901_190150

theorem largest_common_divisor_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end largest_common_divisor_462_330_l1901_190150


namespace garden_perimeter_l1901_190109

/-- Represents a rectangular shape with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Proves that the perimeter of the garden is 56 meters -/
theorem garden_perimeter (garden : Rectangle) (playground : Rectangle) : 
  garden.width = 16 → 
  playground.length = 16 → 
  garden.area = playground.area → 
  garden.perimeter = 56 → 
  garden.perimeter = 56 := by
  sorry

#check garden_perimeter

end garden_perimeter_l1901_190109


namespace sum_of_four_consecutive_integers_divisible_by_two_l1901_190123

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n - 2) + (n - 1) + n + (n + 1)) :=
sorry

end sum_of_four_consecutive_integers_divisible_by_two_l1901_190123


namespace exponent_simplification_l1901_190173

theorem exponent_simplification : 8^6 * 27^6 * 8^27 * 27^8 = 216^14 * 8^19 := by sorry

end exponent_simplification_l1901_190173


namespace max_intersections_pentagon_circle_l1901_190199

/-- A regular pentagon -/
structure RegularPentagon where
  -- We don't need to define the structure, just declare it exists
  
/-- A circle -/
structure Circle where
  -- We don't need to define the structure, just declare it exists

/-- The maximum number of intersections between a line segment and a circle is 2 -/
axiom max_intersections_line_circle : ℕ

/-- The number of sides in a regular pentagon -/
def pentagon_sides : ℕ := 5

/-- Theorem: The maximum number of intersections between a regular pentagon and a circle is 10 -/
theorem max_intersections_pentagon_circle (p : RegularPentagon) (c : Circle) :
  (max_intersections_line_circle * pentagon_sides : ℕ) = 10 := by
  sorry

#check max_intersections_pentagon_circle

end max_intersections_pentagon_circle_l1901_190199


namespace circle_diameter_from_area_l1901_190145

/-- The diameter of a circle with area 50.26548245743669 square meters is 8 meters. -/
theorem circle_diameter_from_area :
  let area : Real := 50.26548245743669
  let diameter : Real := 8
  diameter = 2 * Real.sqrt (area / Real.pi) := by sorry

end circle_diameter_from_area_l1901_190145


namespace isosceles_triangle_perimeter_l1901_190175

/-- An isosceles triangle with sides 6 and 3 has perimeter 15 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 → b = 3 → c = 6 →  -- Two sides are 6, one is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a = c →  -- Isosceles condition
  a + b + c = 15 := by
  sorry

end isosceles_triangle_perimeter_l1901_190175


namespace orange_distribution_l1901_190144

theorem orange_distribution (x : ℚ) : 
  (x/2 + 1/2) + (1/2 * (x/2 - 1/2) + 1/2) + (1/2 * (x/4 - 3/4) + 1/2) = x → x = 7 := by
  sorry

end orange_distribution_l1901_190144


namespace fourth_group_size_l1901_190135

theorem fourth_group_size (total : ℕ) (group1 group2 group3 : ℕ) 
  (h1 : total = 24) 
  (h2 : group1 = 5) 
  (h3 : group2 = 8) 
  (h4 : group3 = 7) : 
  total - (group1 + group2 + group3) = 4 := by
  sorry

end fourth_group_size_l1901_190135


namespace intersection_M_N_l1901_190126

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by sorry

end intersection_M_N_l1901_190126


namespace bicycle_cost_price_l1901_190122

theorem bicycle_cost_price (profit_A_to_B : ℝ) (loss_B_to_C : ℝ) (profit_C_to_D : ℝ) (loss_D_to_E : ℝ) (price_E : ℝ)
  (h1 : profit_A_to_B = 0.20)
  (h2 : loss_B_to_C = 0.15)
  (h3 : profit_C_to_D = 0.30)
  (h4 : loss_D_to_E = 0.10)
  (h5 : price_E = 285) :
  price_E / ((1 + profit_A_to_B) * (1 - loss_B_to_C) * (1 + profit_C_to_D) * (1 - loss_D_to_E)) =
  285 / (1.20 * 0.85 * 1.30 * 0.90) := by
sorry

#eval 285 / (1.20 * 0.85 * 1.30 * 0.90)

end bicycle_cost_price_l1901_190122


namespace min_four_digit_satisfying_condition_l1901_190105

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  let ab := 10 * a + b
  let cd := 10 * c + d
  (n + ab * cd) % 1111 = 0

theorem min_four_digit_satisfying_condition :
  ∃ (n : ℕ), is_four_digit n ∧ satisfies_condition n ∧
  ∀ (m : ℕ), is_four_digit m → satisfies_condition m → n ≤ m :=
by
  use 1729
  sorry

end min_four_digit_satisfying_condition_l1901_190105


namespace rational_segment_existence_l1901_190151

theorem rational_segment_existence (f : ℚ → ℤ) : ∃ x y : ℚ, f x + f y ≤ 2 * f ((x + y) / 2) := by
  sorry

end rational_segment_existence_l1901_190151


namespace arithmetic_sequence_problem_l1901_190176

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

/-- Given an arithmetic sequence with S₈ = 30 and S₄ = 7, prove that a₄ = 13/4. -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : S seq 8 = 30)
    (h₂ : S seq 4 = 7) : 
  seq.a 4 = 13/4 := by
  sorry

end arithmetic_sequence_problem_l1901_190176


namespace equal_sum_sequence_2017_sum_l1901_190143

/-- An equal sum sequence with a given first term and common sum. -/
def EqualSumSequence (a : ℕ → ℕ) (first_term : ℕ) (common_sum : ℕ) : Prop :=
  a 1 = first_term ∧ ∀ n : ℕ, a n + a (n + 1) = common_sum

/-- The sum of the first n terms of a sequence. -/
def SequenceSum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

theorem equal_sum_sequence_2017_sum :
    ∀ a : ℕ → ℕ, EqualSumSequence a 2 5 → SequenceSum a 2017 = 5042 := by
  sorry

end equal_sum_sequence_2017_sum_l1901_190143


namespace f_neg_x_properties_l1901_190138

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem f_neg_x_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f (-x) > f (-y)) := by
  sorry

#check f_neg_x_properties

end f_neg_x_properties_l1901_190138


namespace probability_kings_or_aces_l1901_190163

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Probability of drawing three kings or at least two aces -/
def prob_kings_or_aces : ℚ := 6 / 425

/-- Theorem stating the probability of drawing three kings or at least two aces -/
theorem probability_kings_or_aces :
  (KingsInDeck.choose CardsDrawn) / (StandardDeck.choose CardsDrawn) +
  ((AcesInDeck.choose 2 * (StandardDeck - AcesInDeck).choose 1) +
   AcesInDeck.choose 3) / (StandardDeck.choose CardsDrawn) = prob_kings_or_aces := by
  sorry

end probability_kings_or_aces_l1901_190163


namespace first_hit_not_binomial_l1901_190106

/-- A random variable follows a binomial distribution -/
def is_binomial_distribution (X : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ) (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

/-- Computer virus infection scenario -/
def computer_infection (n : ℕ) : ℕ → ℝ := sorry

/-- First hit scenario -/
def first_hit : ℕ → ℝ := sorry

/-- Multiple shots scenario -/
def multiple_shots (n : ℕ) : ℕ → ℝ := sorry

/-- Car refueling scenario -/
def car_refueling : ℕ → ℝ := sorry

/-- Theorem stating that the first hit scenario is not a binomial distribution -/
theorem first_hit_not_binomial :
  is_binomial_distribution (computer_infection 10) ∧
  is_binomial_distribution (multiple_shots 10) ∧
  is_binomial_distribution car_refueling →
  ¬ is_binomial_distribution first_hit := by sorry

end first_hit_not_binomial_l1901_190106


namespace square_area_ratio_l1901_190116

theorem square_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (4 * a = 4 * (4 * b)) → a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l1901_190116


namespace largest_n_with_conditions_l1901_190132

theorem largest_n_with_conditions : 
  ∃ (n : ℕ), n = 4513 ∧ 
  (∃ (m : ℕ), n^2 = (m+1)^3 - m^3) ∧
  (∃ (k : ℕ), 2*n + 99 = k^2) ∧
  (∀ (n' : ℕ), n' > n → 
    (¬∃ (m : ℕ), n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℕ), 2*n' + 99 = k^2)) := by
  sorry

#check largest_n_with_conditions

end largest_n_with_conditions_l1901_190132


namespace book_page_difference_l1901_190159

/-- The number of pages in Selena's book -/
def selena_pages : ℕ := 400

/-- The number of pages in Harry's book -/
def harry_pages : ℕ := 180

/-- The difference between half of Selena's pages and Harry's pages -/
def page_difference : ℕ := selena_pages / 2 - harry_pages

theorem book_page_difference : page_difference = 20 := by
  sorry

end book_page_difference_l1901_190159


namespace steven_peaches_difference_l1901_190148

theorem steven_peaches_difference (jake steven jill : ℕ) 
  (h1 : jake + 5 = steven)
  (h2 : jill = 87)
  (h3 : jake = jill + 13) :
  steven - jill = 18 := by sorry

end steven_peaches_difference_l1901_190148


namespace solution_set_linear_inequalities_l1901_190164

theorem solution_set_linear_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end solution_set_linear_inequalities_l1901_190164


namespace max_area_rectangle_with_perimeter_60_max_area_225_achievable_l1901_190114

/-- The maximum area of a rectangle with perimeter 60 is 225 -/
theorem max_area_rectangle_with_perimeter_60 :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 60 →
  x * y ≤ 225 :=
by
  sorry

/-- The maximum area of 225 is achievable -/
theorem max_area_225_achievable :
  ∃ x y : ℝ,
  x > 0 ∧ y > 0 ∧
  2 * x + 2 * y = 60 ∧
  x * y = 225 :=
by
  sorry

end max_area_rectangle_with_perimeter_60_max_area_225_achievable_l1901_190114


namespace custom_mult_example_l1901_190170

/-- Custom multiplication operation for rational numbers -/
def custom_mult (a b : ℚ) : ℚ := (a + b) / (1 - b)

/-- Theorem stating that (5 * 4) * 2 = 1 using the custom multiplication -/
theorem custom_mult_example : custom_mult (custom_mult 5 4) 2 = 1 := by
  sorry

end custom_mult_example_l1901_190170


namespace square_root_problem_l1901_190130

theorem square_root_problem (a : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : Real.sqrt n = a + 3) (h3 : Real.sqrt n = 2*a - 15) : n = 49 := by
  sorry

end square_root_problem_l1901_190130


namespace cube_root_equation_solution_l1901_190149

theorem cube_root_equation_solution (x p : ℝ) : 
  (Real.rpow (1 - x) (1/3 : ℝ)) + (Real.rpow (1 + x) (1/3 : ℝ)) = p → 
  (x = 0 ∧ p = -1) → True :=
by sorry

end cube_root_equation_solution_l1901_190149


namespace s₁_less_than_s₂_l1901_190184

/-- Centroid of a triangle -/
structure Centroid (Point : Type*) (Triangle : Type*) where
  center : Point
  triangle : Triangle

/-- Calculate s₁ for a triangle with its centroid -/
def s₁ {Point : Type*} {Triangle : Type*} (c : Centroid Point Triangle) (distance : Point → Point → ℝ) : ℝ :=
  let G := c.center
  let A := sorry
  let B := sorry
  let C := sorry
  2 * (distance G A + distance G B + distance G C)

/-- Calculate s₂ for a triangle -/
def s₂ {Point : Type*} {Triangle : Type*} (t : Triangle) (distance : Point → Point → ℝ) : ℝ :=
  let A := sorry
  let B := sorry
  let C := sorry
  3 * (distance A B + distance B C + distance C A)

/-- The main theorem: s₁ < s₂ for any triangle with its centroid -/
theorem s₁_less_than_s₂ {Point : Type*} {Triangle : Type*} 
  (c : Centroid Point Triangle) (distance : Point → Point → ℝ) :
  s₁ c distance < s₂ c.triangle distance :=
sorry

end s₁_less_than_s₂_l1901_190184


namespace grunters_win_probability_l1901_190158

/-- The probability of the Grunters winning a single game -/
def p : ℚ := 3/5

/-- The number of games played -/
def n : ℕ := 5

/-- The probability of winning all games -/
def win_all : ℚ := p^n

theorem grunters_win_probability : win_all = 243/3125 := by
  sorry

end grunters_win_probability_l1901_190158


namespace quadratic_max_value_l1901_190174

/-- A quadratic function that takes specific values for consecutive natural numbers. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = -9 ∧ f (n + 1) = -9 ∧ f (n + 2) = -15

/-- The maximum value of a quadratic function with the given properties. -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = -33/4 := by
  sorry

end quadratic_max_value_l1901_190174


namespace quadratic_inequality_range_l1901_190141

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ a ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end quadratic_inequality_range_l1901_190141


namespace multiplication_simplification_l1901_190133

theorem multiplication_simplification : 12 * (1 / 26) * 52 * 4 = 96 := by
  sorry

end multiplication_simplification_l1901_190133


namespace polynomial_negative_roots_l1901_190168

theorem polynomial_negative_roots (q : ℝ) (hq : q > 2) :
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + q*x₁^3 + 2*x₁^2 + q*x₁ + 1 = 0 ∧
  x₂^4 + q*x₂^3 + 2*x₂^2 + q*x₂ + 1 = 0 :=
sorry

end polynomial_negative_roots_l1901_190168


namespace betty_age_l1901_190171

theorem betty_age (albert : ℕ) (mary : ℕ) (betty : ℕ)
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 22) :
  betty = 11 := by
  sorry

end betty_age_l1901_190171


namespace sector_area_l1901_190110

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 4) (h2 : θ = π / 3) :
  (1 / 2) * r * θ * r = (8 * π) / 3 := by
  sorry

end sector_area_l1901_190110


namespace solve_system_l1901_190139

theorem solve_system (a b c d : ℤ) 
  (eq1 : a + b = c)
  (eq2 : b + c = 7)
  (eq3 : c + d = 10)
  (eq4 : c = 4) :
  a = 1 ∧ d = 6 := by
  sorry

end solve_system_l1901_190139


namespace min_value_and_inequality_l1901_190102

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x - 3|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 3) : 
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 2) ∧ 
  (1/a^2 + 2/b^2 ≥ 2) := by
  sorry

end min_value_and_inequality_l1901_190102


namespace position_interpretation_is_false_l1901_190169

/-- Represents a position in a grid -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Interprets a position as (column, row) -/
def interpret (p : Position) : String :=
  s!"the {p.x}th column and the {p.y}th row"

/-- The statement to be proven false -/
def statement (p : Position) : String :=
  s!"the {p.y}th row and the {p.x}th column"

theorem position_interpretation_is_false : 
  statement (Position.mk 5 1) ≠ interpret (Position.mk 5 1) :=
sorry

end position_interpretation_is_false_l1901_190169


namespace proportion_proof_l1901_190190

theorem proportion_proof (a b c d : ℝ) : 
  a = 3 →
  a / b = 0.6 →
  a * d = 12 →
  a / b = c / d →
  (a, b, c, d) = (3, 5, 2.4, 4) := by
  sorry

end proportion_proof_l1901_190190


namespace cubic_roots_sum_l1901_190152

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 + p^2 - 2*p - 1 = 0) →
  (q^3 + q^2 - 2*q - 1 = 0) →
  (r^3 + r^2 - 2*r - 1 = 0) →
  p*(q-r)^2 + q*(r-p)^2 + r*(p-q)^2 = -1 := by
  sorry

end cubic_roots_sum_l1901_190152


namespace jellybean_probability_l1901_190119

/-- The probability of selecting exactly 2 red and 1 green jellybean when picking 4 jellybeans randomly without replacement from a bowl containing 5 red, 3 blue, 2 green, and 5 white jellybeans (15 total) -/
theorem jellybean_probability : 
  let total := 15
  let red := 5
  let blue := 3
  let green := 2
  let white := 5
  let pick := 4
  Nat.choose total pick ≠ 0 →
  (Nat.choose red 2 * Nat.choose green 1 * Nat.choose (blue + white) 1) / Nat.choose total pick = 32 / 273 := by
  sorry

end jellybean_probability_l1901_190119


namespace combined_mean_score_l1901_190165

/-- Given two sections of algebra students with different mean scores and a ratio of students between sections, calculate the combined mean score. -/
theorem combined_mean_score (mean1 mean2 : ℚ) (ratio : ℚ) : 
  mean1 = 92 →
  mean2 = 78 →
  ratio = 5/7 →
  (mean1 * ratio + mean2) / (ratio + 1) = 1006/12 := by
  sorry

end combined_mean_score_l1901_190165


namespace product_of_fractions_equals_one_l1901_190124

theorem product_of_fractions_equals_one :
  (5 / 3 : ℚ) * (6 / 10 : ℚ) * (15 / 9 : ℚ) * (12 / 20 : ℚ) *
  (25 / 15 : ℚ) * (18 / 30 : ℚ) * (35 / 21 : ℚ) * (24 / 40 : ℚ) = 1 := by
  sorry

end product_of_fractions_equals_one_l1901_190124


namespace negation_and_absolute_value_l1901_190194

theorem negation_and_absolute_value : 
  (-(-2) = 2) ∧ (-|(-2)| = -2) := by
  sorry

end negation_and_absolute_value_l1901_190194


namespace sequence_value_l1901_190103

/-- Given a sequence {aₙ} where a₁ = 3 and 2aₙ₊₁ - 2aₙ = 1 for all n ≥ 1,
    prove that a₉₉ = 52. -/
theorem sequence_value (a : ℕ → ℝ) (h1 : a 1 = 3) 
    (h2 : ∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 1) : 
  a 99 = 52 := by
  sorry

end sequence_value_l1901_190103


namespace simplify_fraction_l1901_190177

theorem simplify_fraction : (36 : ℚ) / 4536 = 1 / 126 := by
  sorry

end simplify_fraction_l1901_190177


namespace total_time_at_least_5400_seconds_l1901_190166

/-- Represents an observer's record of lap times -/
structure Observer where
  lap_times : List Int
  time_difference : Int

/-- The proposition to be proved -/
theorem total_time_at_least_5400_seconds
  (observer1 observer2 : Observer)
  (h1 : observer1.time_difference = 1)
  (h2 : observer2.time_difference = -1)
  (h3 : observer1.lap_times.length = observer2.lap_times.length)
  (h4 : observer1.lap_times.length ≥ 29) :
  (List.sum observer1.lap_times + List.sum observer2.lap_times) ≥ 5400 :=
sorry

end total_time_at_least_5400_seconds_l1901_190166


namespace max_pieces_in_box_l1901_190155

theorem max_pieces_in_box : 
  ∃ n : ℕ, n < 50 ∧ 
  (∃ k : ℕ, n = 4 * k + 2) ∧ 
  (∃ m : ℕ, n = 6 * m) ∧
  ∀ x : ℕ, x < 50 → 
    ((∃ k : ℕ, x = 4 * k + 2) ∧ (∃ m : ℕ, x = 6 * m)) → 
    x ≤ n :=
by sorry

end max_pieces_in_box_l1901_190155


namespace richard_numbers_l1901_190185

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_distinct (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → i ≠ j →
    (n / (10^i) % 10) ≠ (n / (10^j) % 10)

def all_digits_odd (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → (n / (10^i) % 10) % 2 = 1

def all_digits_even (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → (n / (10^i) % 10) % 2 = 0

def sum_starts_with_11_ends_with_1 (a b : ℕ) : Prop :=
  let sum := a + b
  110000 ≤ sum ∧ sum < 120000 ∧ sum % 10 = 1

def diff_starts_with_2_ends_with_11 (a b : ℕ) : Prop :=
  let diff := a - b
  20000 ≤ diff ∧ diff < 30000 ∧ diff % 100 = 11

theorem richard_numbers :
  ∃ (A B : ℕ),
    is_five_digit A ∧
    is_five_digit B ∧
    all_digits_distinct A ∧
    all_digits_distinct B ∧
    all_digits_odd A ∧
    all_digits_even B ∧
    sum_starts_with_11_ends_with_1 A B ∧
    diff_starts_with_2_ends_with_11 A B ∧
    A = 73591 ∧
    B = 46280 :=
by sorry

end richard_numbers_l1901_190185


namespace least_five_digit_congruent_to_9_mod_18_l1901_190198

theorem least_five_digit_congruent_to_9_mod_18 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 9 → n ≥ 10008 :=
by sorry

end least_five_digit_congruent_to_9_mod_18_l1901_190198


namespace thirteenth_result_l1901_190104

theorem thirteenth_result (total_count : Nat) (total_avg first_avg last_avg : ℚ) :
  total_count = 25 →
  total_avg = 19 →
  first_avg = 14 →
  last_avg = 17 →
  (total_count * total_avg - 12 * first_avg - 12 * last_avg : ℚ) = 103 :=
by sorry

end thirteenth_result_l1901_190104


namespace hyperbola_eccentricity_l1901_190117

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and an asymptote y = 4/3 * x is 5/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = 4 / 3) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5 / 3 := by sorry

end hyperbola_eccentricity_l1901_190117
