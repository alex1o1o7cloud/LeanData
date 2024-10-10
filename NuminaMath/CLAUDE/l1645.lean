import Mathlib

namespace positive_integer_triple_characterization_l1645_164568

theorem positive_integer_triple_characterization :
  ∀ (a b c : ℕ+),
    (a.val^2 = 2^b.val + c.val^4) →
    (a.val % 2 = 1 ∨ b.val % 2 = 1 ∨ c.val % 2 = 1) →
    (a.val % 2 = 0 ∨ b.val % 2 = 0) →
    (a.val % 2 = 0 ∨ c.val % 2 = 0) →
    (b.val % 2 = 0 ∨ c.val % 2 = 0) →
    ∃ (n : ℕ+), a.val = 3 * 2^(2*n.val) ∧ b.val = 4*n.val + 3 ∧ c.val = 2^n.val :=
by sorry

end positive_integer_triple_characterization_l1645_164568


namespace allan_correct_answers_l1645_164596

theorem allan_correct_answers (total_questions : ℕ) 
  (correct_points : ℚ) (incorrect_penalty : ℚ) (final_score : ℚ) :
  total_questions = 120 →
  correct_points = 1 →
  incorrect_penalty = 1/4 →
  final_score = 100 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    (correct_answers : ℚ) * correct_points + 
    ((total_questions - correct_answers) : ℚ) * (-incorrect_penalty) = final_score ∧
    correct_answers = 104 := by
  sorry

end allan_correct_answers_l1645_164596


namespace three_digit_divisibility_by_nine_l1645_164550

/-- Function to calculate the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem stating that for all three-digit numbers, if the sum of digits is divisible by 9, then the number is divisible by 9 -/
theorem three_digit_divisibility_by_nine :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → (sumOfDigits n % 9 = 0 → n % 9 = 0) :=
by
  sorry

#check three_digit_divisibility_by_nine

end three_digit_divisibility_by_nine_l1645_164550


namespace total_jumps_in_3min_l1645_164516

/-- The number of jumps Jung-min can do in 4 minutes -/
def jung_min_jumps_4min : ℕ := 256

/-- The number of jumps Jimin can do in 3 minutes -/
def jimin_jumps_3min : ℕ := 111

/-- The duration we want to calculate the total jumps for (in minutes) -/
def duration : ℕ := 3

/-- Theorem stating that the sum of Jung-min's and Jimin's jumps in 3 minutes is 303 -/
theorem total_jumps_in_3min :
  (jung_min_jumps_4min * duration) / 4 + jimin_jumps_3min = 303 := by
  sorry

#eval (jung_min_jumps_4min * duration) / 4 + jimin_jumps_3min

end total_jumps_in_3min_l1645_164516


namespace twentiethTerm_eq_97_l1645_164528

/-- The nth term of an arithmetic sequence -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence -/
def twentiethTerm : ℝ :=
  arithmeticSequence 2 5 20

theorem twentiethTerm_eq_97 : twentiethTerm = 97 := by
  sorry

end twentiethTerm_eq_97_l1645_164528


namespace gcd_840_1764_l1645_164580

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1645_164580


namespace quartic_sum_theorem_l1645_164509

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (m : ℝ) where
  Q : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, Q x = a * x^4 + b * x^3 + c * x^2 + d * x + m
  at_zero : Q 0 = m
  at_one : Q 1 = 2 * m
  at_neg_one : Q (-1) = 4 * m
  at_two : Q 2 = 5 * m

/-- Theorem: For a quartic polynomial Q satisfying specific conditions, Q(2) + Q(-2) = 66m -/
theorem quartic_sum_theorem (m : ℝ) (qp : QuarticPolynomial m) : qp.Q 2 + qp.Q (-2) = 66 * m := by
  sorry

end quartic_sum_theorem_l1645_164509


namespace work_rate_problem_l1645_164585

theorem work_rate_problem (a b : ℝ) (h1 : a = (1/2) * b) (h2 : (a + b) * 20 = 1) :
  1 / b = 30 := by sorry

end work_rate_problem_l1645_164585


namespace fruit_count_correct_l1645_164531

structure FruitBasket :=
  (plums : ℕ)
  (oranges : ℕ)
  (apples : ℕ)
  (pears : ℕ)
  (cherries : ℕ)

def initial_basket : FruitBasket :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def exchanges (basket : FruitBasket) : FruitBasket :=
  { plums := basket.plums - 4 + 2,
    oranges := basket.oranges - 3 + 1,
    apples := basket.apples - 5 + 2,
    pears := basket.pears + 1 + 3,
    cherries := basket.cherries + 2 }

def final_basket : FruitBasket :=
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 }

theorem fruit_count_correct : exchanges initial_basket = final_basket := by
  sorry

end fruit_count_correct_l1645_164531


namespace subset_implies_membership_l1645_164557

theorem subset_implies_membership {α : Type*} (A B : Set α) (h : A ⊆ B) :
  ∀ x, x ∈ A → x ∈ B := by
  sorry

end subset_implies_membership_l1645_164557


namespace car_price_proof_l1645_164592

/-- Calculates the price of a car given loan terms and payments -/
def carPrice (loanYears : ℕ) (downPayment : ℕ) (monthlyPayment : ℕ) : ℕ :=
  downPayment + loanYears * 12 * monthlyPayment

/-- Proves that the price of the car is $20,000 given the specified conditions -/
theorem car_price_proof :
  carPrice 5 5000 250 = 20000 := by
  sorry

#eval carPrice 5 5000 250

end car_price_proof_l1645_164592


namespace cube_root_equation_solution_l1645_164510

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x / 2) ^ (1/3 : ℝ) = -4 :=
by
  -- The unique solution is x = 134
  use 134
  sorry

end cube_root_equation_solution_l1645_164510


namespace map_distance_to_actual_distance_l1645_164511

theorem map_distance_to_actual_distance 
  (map_distance : ℝ) 
  (scale_map : ℝ) 
  (scale_actual : ℝ) 
  (h1 : map_distance = 18) 
  (h2 : scale_map = 0.5) 
  (h3 : scale_actual = 8) : 
  map_distance * (scale_actual / scale_map) = 288 := by
sorry

end map_distance_to_actual_distance_l1645_164511


namespace negation_of_proposition_l1645_164549

theorem negation_of_proposition :
  (∀ a b : ℝ, ab > 0 → a > 0) ↔ (∀ a b : ℝ, ab ≤ 0 → a ≤ 0) :=
by sorry

end negation_of_proposition_l1645_164549


namespace jims_gross_pay_l1645_164582

theorem jims_gross_pay (G : ℝ) : 
  G - 0.25 * G - 100 = 740 → G = 1120 := by
  sorry

end jims_gross_pay_l1645_164582


namespace car_braking_distance_l1645_164521

/-- Represents the braking sequence of a car -/
def brakingSequence (initial : ℕ) (decrease : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => max 0 (brakingSequence initial decrease n - decrease)

/-- Calculates the total distance traveled during braking -/
def totalDistance (initial : ℕ) (decrease : ℕ) : ℕ :=
  (List.range 100).foldl (λ acc n => acc + brakingSequence initial decrease n) 0

/-- Theorem stating the total braking distance for the given conditions -/
theorem car_braking_distance :
  totalDistance 36 8 = 108 := by
  sorry


end car_braking_distance_l1645_164521


namespace symmetric_point_wrt_origin_l1645_164533

/-- Given a point A with coordinates (1, 2), its symmetric point A' with respect to the origin has coordinates (-1, -2) -/
theorem symmetric_point_wrt_origin :
  let A : ℝ × ℝ := (1, 2)
  let A' : ℝ × ℝ := (-A.1, -A.2)
  A' = (-1, -2) := by sorry

end symmetric_point_wrt_origin_l1645_164533


namespace smallest_block_size_l1645_164505

/-- Given a rectangular block formed by N congruent 1-cm cubes,
    where 252 cubes are invisible when three faces are viewed,
    the smallest possible value of N is 392. -/
theorem smallest_block_size (N : ℕ) : 
  (∃ l m n : ℕ, 
    l > 0 ∧ m > 0 ∧ n > 0 ∧
    (l - 1) * (m - 1) * (n - 1) = 252 ∧
    N = l * m * n) →
  N ≥ 392 :=
by sorry

end smallest_block_size_l1645_164505


namespace quadratic_perfect_square_l1645_164530

theorem quadratic_perfect_square (x : ℝ) (d : ℝ) :
  (∃ b : ℝ, ∀ x, x^2 + 60*x + d = (x + b)^2) ↔ d = 900 := by
  sorry

end quadratic_perfect_square_l1645_164530


namespace intersection_empty_implies_a_geq_one_l1645_164575

theorem intersection_empty_implies_a_geq_one (a : ℝ) : 
  let A : Set ℝ := {0, 1}
  let B : Set ℝ := {x | x > a}
  A ∩ B = ∅ → a ≥ 1 := by
sorry

end intersection_empty_implies_a_geq_one_l1645_164575


namespace geometric_sequence_15th_term_l1645_164503

/-- Given a geometric sequence where the 8th term is 8 and the 11th term is 64,
    prove that the 15th term is 1024. -/
theorem geometric_sequence_15th_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (a 11 / a 8)^(1/3)) →
  a 8 = 8 →
  a 11 = 64 →
  a 15 = 1024 := by
sorry

end geometric_sequence_15th_term_l1645_164503


namespace binomial_largest_coefficient_l1645_164525

/-- 
Given a positive integer n, if the binomial coefficient in the expansion of (2+x)^n 
is largest in the 4th and 5th terms, then n = 7.
-/
theorem binomial_largest_coefficient (n : ℕ+) : 
  (∀ k : ℕ, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 3) ∧
  (∀ k : ℕ, k ≠ 3 ∧ k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) ∧
  Nat.choose n 3 = Nat.choose n 4 →
  n = 7 := by
sorry

end binomial_largest_coefficient_l1645_164525


namespace labourer_savings_l1645_164544

/-- Calculates the amount saved by a labourer after clearing debt -/
def amount_saved (monthly_income : ℕ) (initial_expense : ℕ) (initial_months : ℕ) (reduced_expense : ℕ) (reduced_months : ℕ) : ℕ :=
  let initial_total_expense := initial_expense * initial_months
  let initial_total_income := monthly_income * initial_months
  let debt := if initial_total_expense > initial_total_income then initial_total_expense - initial_total_income else 0
  let reduced_total_expense := reduced_expense * reduced_months
  let reduced_total_income := monthly_income * reduced_months
  reduced_total_income - (reduced_total_expense + debt)

/-- Theorem stating the amount saved by the labourer -/
theorem labourer_savings :
  amount_saved 81 90 6 60 4 = 30 :=
by sorry

end labourer_savings_l1645_164544


namespace wage_payment_theorem_l1645_164590

/-- Represents the daily wage of a worker -/
structure DailyWage where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a sum of money -/
def SumOfMoney : Type := ℝ

/-- Given two workers a and b, and a sum of money S, 
    prove that if S can pay a's wages for 21 days and 
    both a and b's wages for 12 days, then S can pay 
    b's wages for 28 days -/
theorem wage_payment_theorem 
  (a b : DailyWage) 
  (S : SumOfMoney) 
  (h1 : S = 21 * a.amount)
  (h2 : S = 12 * (a.amount + b.amount)) :
  S = 28 * b.amount := by
  sorry


end wage_payment_theorem_l1645_164590


namespace negation_of_union_membership_l1645_164563

theorem negation_of_union_membership {α : Type*} (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B :=
by sorry

end negation_of_union_membership_l1645_164563


namespace odd_square_not_representable_l1645_164562

def divisor_count (k : ℕ+) : ℕ := (Nat.divisors k.val).card

theorem odd_square_not_representable (M : ℕ+) (h_odd : Odd M.val) (h_square : ∃ k : ℕ+, M = k * k) :
  ¬∃ n : ℕ+, (M : ℚ) = (2 * Real.sqrt n.val / divisor_count n) ^ 2 := by
  sorry

end odd_square_not_representable_l1645_164562


namespace least_positive_integer_with_remainders_l1645_164551

theorem least_positive_integer_with_remainders : ∃ b : ℕ+, 
  (b : ℤ) % 4 = 1 ∧ 
  (b : ℤ) % 5 = 2 ∧ 
  (b : ℤ) % 6 = 3 ∧ 
  (∀ c : ℕ+, c < b → 
    (c : ℤ) % 4 ≠ 1 ∨ 
    (c : ℤ) % 5 ≠ 2 ∨ 
    (c : ℤ) % 6 ≠ 3) :=
by
  -- The proof goes here
  sorry

end least_positive_integer_with_remainders_l1645_164551


namespace emily_beads_used_l1645_164552

/-- The number of beads Emily has used so far -/
def beads_used (total_made : ℕ) (beads_per_necklace : ℕ) (given_away : ℕ) : ℕ :=
  total_made * beads_per_necklace - given_away * beads_per_necklace

/-- Theorem stating that Emily has used 92 beads -/
theorem emily_beads_used :
  beads_used 35 4 12 = 92 := by
  sorry

end emily_beads_used_l1645_164552


namespace impossible_continuous_coverage_l1645_164599

/-- Represents a runner on the track -/
structure Runner where
  speed : ℕ
  startPosition : ℝ

/-- Represents the circular track with runners -/
structure Track where
  length : ℝ
  spectatorStandLength : ℝ
  runners : List Runner

/-- Checks if a runner is passing the spectator stands at a given time -/
def isPassingStands (runner : Runner) (track : Track) (time : ℝ) : Prop :=
  let position := (runner.startPosition + runner.speed * time) % track.length
  0 ≤ position ∧ position < track.spectatorStandLength

/-- Main theorem statement -/
theorem impossible_continuous_coverage (track : Track) : 
  track.length = 2000 ∧ 
  track.spectatorStandLength = 100 ∧ 
  track.runners.length = 20 ∧
  (∀ i, i ∈ Finset.range 20 → 
    ∃ r ∈ track.runners, r.speed = i + 10) →
  ¬ (∀ t : ℝ, ∃ r ∈ track.runners, isPassingStands r track t) :=
by sorry

end impossible_continuous_coverage_l1645_164599


namespace knight_returns_to_start_l1645_164529

/-- A castle in Mara -/
structure Castle where
  id : ℕ

/-- The graph of castles and roads in Mara -/
structure MaraGraph where
  castles : Set Castle
  roads : Castle → Set Castle
  finite_castles : Set.Finite castles
  three_roads : ∀ c, (roads c).ncard = 3

/-- A turn direction -/
inductive Turn
| Left
| Right

/-- A path through the castles -/
structure KnightPath (G : MaraGraph) where
  path : ℕ → Castle
  turns : ℕ → Turn
  valid_path : ∀ n, G.roads (path n) (path (n + 1))
  alternating_turns : ∀ n, turns n ≠ turns (n + 1)

/-- The theorem stating that the knight will return to the original castle -/
theorem knight_returns_to_start (G : MaraGraph) (p : KnightPath G) :
  ∃ n m, n < m ∧ p.path n = p.path m := by sorry

end knight_returns_to_start_l1645_164529


namespace equation_solution_l1645_164548

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ 9*x - 3 = k*x + 14) ↔ (k = 8 ∨ k = -8) :=
by sorry

end equation_solution_l1645_164548


namespace book_distribution_theorem_l1645_164565

/-- The number of ways to divide n distinct objects into k groups of size m each. -/
def divide_into_groups (n k m : ℕ) : ℕ :=
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m) / (Nat.factorial k)

/-- The number of ways to distribute n distinct objects among k people, with each person receiving m objects. -/
def distribute_among_people (n k m : ℕ) : ℕ :=
  divide_into_groups n k m * (Nat.factorial k)

theorem book_distribution_theorem :
  let n : ℕ := 6  -- number of books
  let k : ℕ := 3  -- number of groups/people
  let m : ℕ := 2  -- number of books per group/person
  divide_into_groups n k m = 15 ∧
  distribute_among_people n k m = 90 := by
  sorry


end book_distribution_theorem_l1645_164565


namespace gumdrops_problem_l1645_164577

/-- The maximum number of gumdrops that can be bought with a given amount of money and cost per gumdrop. -/
def max_gumdrops (total_money : ℕ) (cost_per_gumdrop : ℕ) : ℕ :=
  total_money / cost_per_gumdrop

/-- Theorem stating that with 80 cents and gumdrops costing 4 cents each, the maximum number of gumdrops that can be bought is 20. -/
theorem gumdrops_problem :
  max_gumdrops 80 4 = 20 := by
  sorry

end gumdrops_problem_l1645_164577


namespace octagon_dissection_and_reassembly_l1645_164523

/-- Represents a regular octagon -/
structure RegularOctagon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a section of a regular octagon -/
structure OctagonSection where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Checks if two OctagonSections are similar -/
def are_similar (s1 s2 : OctagonSection) : Prop :=
  sorry

/-- Checks if two RegularOctagons are congruent -/
def are_congruent (o1 o2 : RegularOctagon) : Prop :=
  sorry

/-- Represents the dissection of a RegularOctagon into OctagonSections -/
def dissect (o : RegularOctagon) : List OctagonSection :=
  sorry

/-- Represents the reassembly of OctagonSections into RegularOctagons -/
def reassemble (sections : List OctagonSection) : List RegularOctagon :=
  sorry

theorem octagon_dissection_and_reassembly 
  (o : RegularOctagon) : 
  let sections := dissect o
  ∃ (reassembled : List RegularOctagon),
    (reassembled = reassemble sections) ∧ 
    (sections.length = 8) ∧
    (∀ (s1 s2 : OctagonSection), s1 ∈ sections → s2 ∈ sections → are_similar s1 s2) ∧
    (reassembled.length = 8) ∧
    (∀ (o1 o2 : RegularOctagon), o1 ∈ reassembled → o2 ∈ reassembled → are_congruent o1 o2) :=
by
  sorry

end octagon_dissection_and_reassembly_l1645_164523


namespace debby_pancakes_count_l1645_164527

/-- The number of pancakes Debby made with blueberries -/
def blueberry_pancakes : ℕ := 20

/-- The number of pancakes Debby made with bananas -/
def banana_pancakes : ℕ := 24

/-- The number of plain pancakes Debby made -/
def plain_pancakes : ℕ := 23

/-- The total number of pancakes Debby made -/
def total_pancakes : ℕ := blueberry_pancakes + banana_pancakes + plain_pancakes

theorem debby_pancakes_count : total_pancakes = 67 := by
  sorry

end debby_pancakes_count_l1645_164527


namespace circle_range_m_value_l1645_164573

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the condition for a point (x, y) to be on the circle
def on_circle (x y m : ℝ) : Prop :=
  circle_equation x y m

-- Define the condition for a point (x, y) to be on the line
def on_line (x y : ℝ) : Prop :=
  line_equation x y

-- Define the condition for the origin to be on the circle with diameter MN
def origin_on_diameter (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Range of m for which the equation represents a circle
theorem circle_range (m : ℝ) :
  (∃ x y, circle_equation x y m) → m < 5 :=
sorry

-- Theorem 2: Value of m when the circle intersects the line and origin is on the diameter
theorem m_value (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, 
    on_circle x₁ y₁ m ∧ 
    on_circle x₂ y₂ m ∧ 
    on_line x₁ y₁ ∧ 
    on_line x₂ y₂ ∧ 
    origin_on_diameter x₁ y₁ x₂ y₂) 
  → m = 8/5 :=
sorry

end circle_range_m_value_l1645_164573


namespace subtraction_amount_l1645_164537

theorem subtraction_amount (N : ℕ) (A : ℕ) : N = 32 → (N - A) / 13 = 2 → A = 6 := by
  sorry

end subtraction_amount_l1645_164537


namespace standard_ellipse_foci_l1645_164576

/-- Represents an ellipse with equation (x^2 / 10) + y^2 = 1 -/
structure StandardEllipse where
  equation : ∀ (x y : ℝ), (x^2 / 10) + y^2 = 1

/-- Represents the foci of an ellipse -/
structure EllipseFoci where
  x : ℝ
  y : ℝ

/-- Theorem: The foci of the standard ellipse are at (3, 0) and (-3, 0) -/
theorem standard_ellipse_foci (e : StandardEllipse) : 
  ∃ (f1 f2 : EllipseFoci), f1.x = 3 ∧ f1.y = 0 ∧ f2.x = -3 ∧ f2.y = 0 :=
sorry

end standard_ellipse_foci_l1645_164576


namespace oarsmen_count_l1645_164559

theorem oarsmen_count (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 1.8 →
  old_weight = 53 →
  new_weight = 71 →
  (new_weight - old_weight) / average_increase = 10 := by
sorry

end oarsmen_count_l1645_164559


namespace integer_triple_problem_l1645_164514

theorem integer_triple_problem (a b c : ℤ) :
  let N := ((a - b) * (b - c) * (c - a)) / 2 + 2
  (∃ m : ℕ, N = 1729^m ∧ N > 0) →
  ∃ k : ℤ, a = k + 2 ∧ b = k + 1 ∧ c = k :=
by sorry

end integer_triple_problem_l1645_164514


namespace intersection_tangent_line_constant_l1645_164545

/-- Given two curves f(x) = √x and g(x) = a ln x that intersect and have the same tangent line
    at the point of intersection, prove that a = e/2 -/
theorem intersection_tangent_line_constant (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.sqrt x₀ = a * Real.log x₀ ∧ 
    (1 / (2 * Real.sqrt x₀) : ℝ) = a / x₀) →
  a = Real.exp 1 / 2 := by
sorry

end intersection_tangent_line_constant_l1645_164545


namespace arithmetic_sequence_product_l1645_164554

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = -11) :=
by sorry

end arithmetic_sequence_product_l1645_164554


namespace compound_molecular_weight_l1645_164534

/-- The atomic mass of Deuterium (H-2) in atomic mass units (amu) -/
def mass_deuterium : ℝ := 2.014

/-- The atomic mass of Carbon-13 (C-13) in atomic mass units (amu) -/
def mass_carbon13 : ℝ := 13.003

/-- The atomic mass of Oxygen-16 (O-16) in atomic mass units (amu) -/
def mass_oxygen16 : ℝ := 15.995

/-- The atomic mass of Oxygen-18 (O-18) in atomic mass units (amu) -/
def mass_oxygen18 : ℝ := 17.999

/-- The number of Deuterium molecules in the compound -/
def num_deuterium : ℕ := 2

/-- The number of Carbon-13 molecules in the compound -/
def num_carbon13 : ℕ := 1

/-- The number of Oxygen-16 molecules in the compound -/
def num_oxygen16 : ℕ := 1

/-- The number of Oxygen-18 molecules in the compound -/
def num_oxygen18 : ℕ := 2

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  num_deuterium * mass_deuterium +
  num_carbon13 * mass_carbon13 +
  num_oxygen16 * mass_oxygen16 +
  num_oxygen18 * mass_oxygen18

theorem compound_molecular_weight :
  ∃ ε > 0, |molecular_weight - 69.024| < ε :=
sorry

end compound_molecular_weight_l1645_164534


namespace circle_equation_radius_l1645_164558

/-- The radius of a circle given its equation in standard form -/
def circle_radius (h : ℝ) (k : ℝ) (r : ℝ) : ℝ := r

theorem circle_equation_radius :
  circle_radius 1 0 3 = 3 :=
by sorry

end circle_equation_radius_l1645_164558


namespace rice_grains_difference_l1645_164564

def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ :=
  (List.range n).map grains_on_square |> List.sum

theorem rice_grains_difference : 
  grains_on_square 12 - sum_first_n_squares 9 = 501693 := by
  sorry

end rice_grains_difference_l1645_164564


namespace marketing_cost_per_book_l1645_164506

/-- The marketing cost per book for a publishing company --/
theorem marketing_cost_per_book 
  (fixed_cost : ℝ) 
  (selling_price : ℝ) 
  (break_even_quantity : ℕ) 
  (h1 : fixed_cost = 50000)
  (h2 : selling_price = 9)
  (h3 : break_even_quantity = 10000) :
  (selling_price * break_even_quantity - fixed_cost) / break_even_quantity = 4 := by
sorry


end marketing_cost_per_book_l1645_164506


namespace concentric_circles_area_ratio_l1645_164589

theorem concentric_circles_area_ratio 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : (π * R^2) / (π * r^2) = 4) : 
  R = 2 * r ∧ R - r = r := by
sorry

end concentric_circles_area_ratio_l1645_164589


namespace collinear_points_implies_b_value_l1645_164546

/-- Three points (x₁, y₁), (x₂, y₂), and (x₃, y₃) are collinear if and only if
    (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁) -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- If the points (5, -3), (2b + 4, 5), and (-3b + 6, -1) are collinear, then b = 5/14 -/
theorem collinear_points_implies_b_value :
  ∀ b : ℝ, collinear 5 (-3) (2*b + 4) 5 (-3*b + 6) (-1) → b = 5/14 := by
  sorry

end collinear_points_implies_b_value_l1645_164546


namespace jodi_walking_schedule_l1645_164542

/-- Represents Jodi's walking schedule over 4 weeks -/
structure WalkingSchedule where
  week1_distance : ℝ
  week2_distance : ℝ
  week3_distance : ℝ
  week4_distance : ℝ
  days_per_week : ℕ
  total_distance : ℝ

/-- Theorem stating that given Jodi's walking schedule, she walked 2 miles per day in the second week -/
theorem jodi_walking_schedule (schedule : WalkingSchedule) 
  (h1 : schedule.week1_distance = 1)
  (h2 : schedule.week3_distance = 3)
  (h3 : schedule.week4_distance = 4)
  (h4 : schedule.days_per_week = 6)
  (h5 : schedule.total_distance = 60)
  : schedule.week2_distance = 2 := by
  sorry

end jodi_walking_schedule_l1645_164542


namespace sean_bedroom_bulbs_l1645_164591

/-- The number of light bulbs Sean needs to replace in his bedroom. -/
def bedroom_bulbs : ℕ := 2

/-- The number of light bulbs Sean needs to replace in the bathroom. -/
def bathroom_bulbs : ℕ := 1

/-- The number of light bulbs Sean needs to replace in the kitchen. -/
def kitchen_bulbs : ℕ := 1

/-- The number of light bulbs Sean needs to replace in the basement. -/
def basement_bulbs : ℕ := 4

/-- The number of light bulbs per pack. -/
def bulbs_per_pack : ℕ := 2

/-- The number of packs Sean needs. -/
def packs_needed : ℕ := 6

/-- The total number of light bulbs Sean needs. -/
def total_bulbs : ℕ := packs_needed * bulbs_per_pack

theorem sean_bedroom_bulbs :
  bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs +
  (bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs) / 2 = total_bulbs :=
by sorry

end sean_bedroom_bulbs_l1645_164591


namespace inverse_variation_l1645_164543

/-- Given that p and q vary inversely, prove that when p = 400, q = 1, 
    given that when p = 800, q = 0.5 -/
theorem inverse_variation (p q : ℝ) (h : p * q = 800 * 0.5) :
  p = 400 → q = 1 := by
  sorry

end inverse_variation_l1645_164543


namespace divide_fractions_three_sevenths_div_two_and_half_l1645_164507

theorem divide_fractions (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem three_sevenths_div_two_and_half :
  (3 : ℚ) / 7 / (5 / 2) = 6 / 35 := by sorry

end divide_fractions_three_sevenths_div_two_and_half_l1645_164507


namespace smallest_negative_integer_congruence_l1645_164588

theorem smallest_negative_integer_congruence :
  ∃ (x : ℤ), x < 0 ∧ (45 * x + 8) % 24 = 5 ∧
  ∀ (y : ℤ), y < 0 ∧ (45 * y + 8) % 24 = 5 → x ≥ y :=
by sorry

end smallest_negative_integer_congruence_l1645_164588


namespace distance_between_squares_l1645_164526

/-- Given two squares, one with perimeter 8 cm and another with area 36 cm²,
    prove that the distance between opposite corners is √80 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ)
    (h1 : small_square_perimeter = 8)
    (h2 : large_square_area = 36) :
    Real.sqrt ((small_square_perimeter / 4 + Real.sqrt large_square_area) ^ 2 +
               (Real.sqrt large_square_area - small_square_perimeter / 4) ^ 2) = Real.sqrt 80 :=
by sorry

end distance_between_squares_l1645_164526


namespace geometric_sequence_seventh_term_l1645_164593

/-- Determinant of a 2x2 matrix -/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_third : a 3 = 1)
  (h_det : det (a 6) 8 8 (a 8) = 0) :
  a 7 = 8 := by
sorry

end geometric_sequence_seventh_term_l1645_164593


namespace min_value_fraction_sum_min_value_fraction_sum_achievable_l1645_164586

theorem min_value_fraction_sum (a b : ℤ) (h : a > b) :
  (((2 * a + b) : ℚ) / (a - b : ℚ)) + ((a - b : ℚ) / ((2 * a + b) : ℚ)) ≥ 13 / 6 :=
by sorry

theorem min_value_fraction_sum_achievable :
  ∃ (a b : ℤ), a > b ∧ (((2 * a + b) : ℚ) / (a - b : ℚ)) + ((a - b : ℚ) / ((2 * a + b) : ℚ)) = 13 / 6 :=
by sorry

end min_value_fraction_sum_min_value_fraction_sum_achievable_l1645_164586


namespace parallelogram_height_l1645_164595

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = base * height → area = 320 → base = 20 → height = 16 := by
  sorry

end parallelogram_height_l1645_164595


namespace square_sum_from_means_l1645_164572

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = 10) : 
  x^2 + y^2 = 1400 := by
sorry

end square_sum_from_means_l1645_164572


namespace min_score_for_maria_l1645_164517

def min_score_for_advanced_class (scores : List ℚ) (required_average : ℚ) : ℚ :=
  let total_terms := 5
  let current_sum := scores.sum
  max ((required_average * total_terms) - current_sum) 0

theorem min_score_for_maria : 
  min_score_for_advanced_class [84/100, 80/100, 82/100, 83/100] (85/100) = 96/100 := by
  sorry

end min_score_for_maria_l1645_164517


namespace grass_field_width_l1645_164587

/-- Proves that the width of a rectangular grass field is 192 meters, given specific conditions --/
theorem grass_field_width : 
  ∀ (w : ℝ),
  (82 * (w + 7) - 75 * w = 1918) →
  w = 192 := by
  sorry

end grass_field_width_l1645_164587


namespace tom_score_l1645_164571

/-- Calculates the score for regular enemies --/
def regularScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 10
  if kills ≥ 200 then baseScore * 2
  else if kills ≥ 150 then baseScore + (baseScore * 3 / 4)
  else if kills ≥ 100 then baseScore + (baseScore / 2)
  else baseScore

/-- Calculates the score for elite enemies --/
def eliteScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 25
  if kills ≥ 35 then baseScore + (baseScore * 7 / 10)
  else if kills ≥ 25 then baseScore + (baseScore / 2)
  else if kills ≥ 15 then baseScore + (baseScore * 3 / 10)
  else baseScore

/-- Calculates the score for boss enemies --/
def bossScore (kills : ℕ) : ℕ :=
  let baseScore := kills * 50
  if kills ≥ 10 then baseScore + (baseScore * 2 / 5)
  else if kills ≥ 5 then baseScore + (baseScore / 5)
  else baseScore

/-- Calculates the total score --/
def totalScore (regularKills eliteKills bossKills : ℕ) : ℕ :=
  regularScore regularKills + eliteScore eliteKills + bossScore bossKills

theorem tom_score : totalScore 160 20 8 = 3930 := by
  sorry

end tom_score_l1645_164571


namespace sphere_radius_l1645_164508

/-- Given two spheres A and B, where A has radius 40 cm and the ratio of their surface areas is 16,
    prove that the radius of sphere B is 20 cm. -/
theorem sphere_radius (r : ℝ) : 
  let surface_area (radius : ℝ) := 4 * Real.pi * radius^2
  surface_area 40 / surface_area r = 16 → r = 20 := by
  sorry

end sphere_radius_l1645_164508


namespace austin_bicycle_weeks_l1645_164538

/-- The number of weeks Austin needs to work to buy a bicycle -/
def weeks_to_buy_bicycle (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (friday_hours : ℚ) (bicycle_cost : ℚ) : ℚ :=
  bicycle_cost / (hourly_rate * (monday_hours + wednesday_hours + friday_hours))

/-- Proof that Austin needs 6 weeks to buy the bicycle -/
theorem austin_bicycle_weeks : 
  weeks_to_buy_bicycle 5 2 1 3 180 = 6 := by
  sorry

end austin_bicycle_weeks_l1645_164538


namespace lady_walking_distance_l1645_164522

theorem lady_walking_distance (x y : ℝ) (h1 : y = 2 * x) (h2 : x + y = 12) : x = 4 := by
  sorry

end lady_walking_distance_l1645_164522


namespace candy_ratio_in_bowl_l1645_164570

-- Define the properties of each bag
def bag1_total : ℕ := 27
def bag1_red_ratio : ℚ := 1/3

def bag2_total : ℕ := 36
def bag2_red_ratio : ℚ := 1/4

def bag3_total : ℕ := 45
def bag3_red_ratio : ℚ := 1/5

-- Define the theorem
theorem candy_ratio_in_bowl :
  let total_candies := bag1_total + bag2_total + bag3_total
  let total_red := bag1_total * bag1_red_ratio + bag2_total * bag2_red_ratio + bag3_total * bag3_red_ratio
  total_red / total_candies = 1/4 := by
  sorry

end candy_ratio_in_bowl_l1645_164570


namespace quadratic_equation_root_l1645_164583

theorem quadratic_equation_root (x : ℝ) : x^2 - 6*x - 4 = 0 ↔ x = Real.sqrt 5 - 3 := by sorry

end quadratic_equation_root_l1645_164583


namespace diagonal_passes_through_800_cubes_l1645_164581

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 240 × 360 × 400 rectangular solid passes through 800 unit cubes -/
theorem diagonal_passes_through_800_cubes :
  cubes_passed_by_diagonal 240 360 400 = 800 := by
  sorry

end diagonal_passes_through_800_cubes_l1645_164581


namespace number_of_fractions_l1645_164539

/-- A function that determines if an expression is a fraction in the form a/b -/
def isFraction (expr : String) : Bool :=
  match expr with
  | "5/(a-x)" => true
  | "(m+n)/(mn)" => true
  | "5x^2/x" => true
  | _ => false

/-- The list of expressions given in the problem -/
def expressions : List String :=
  ["1/5(1-x)", "5/(a-x)", "4x/(π-3)", "(m+n)/(mn)", "(x^2-y^2)/2", "5x^2/x"]

/-- Theorem stating that the number of fractions in the given list is 3 -/
theorem number_of_fractions : 
  (expressions.filter isFraction).length = 3 := by sorry

end number_of_fractions_l1645_164539


namespace divisibility_by_eight_l1645_164501

theorem divisibility_by_eight : ∃ k : ℤ, 5^2001 + 7^2002 + 9^2003 + 11^2004 = 8 * k := by
  sorry

end divisibility_by_eight_l1645_164501


namespace power_mod_equivalence_l1645_164574

theorem power_mod_equivalence (x : ℤ) (h : x^77 % 7 = 6) : x^5 % 7 = 6 := by
  sorry

end power_mod_equivalence_l1645_164574


namespace factorization_equality_l1645_164540

theorem factorization_equality (x y : ℝ) : (x + y)^2 - 14*(x + y) + 49 = (x + y - 7)^2 := by
  sorry

end factorization_equality_l1645_164540


namespace sum_first_5_even_numbers_l1645_164504

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.map (fun i => 2 * i) (List.range n)

theorem sum_first_5_even_numbers :
  (first_n_even_numbers 5).sum = 30 := by
  sorry

end sum_first_5_even_numbers_l1645_164504


namespace two_thousand_second_non_diff_of_squares_non_diff_of_squares_form_eight_thousand_six_form_main_theorem_l1645_164524

/-- A function that returns true if a number is the difference of two squares -/
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, n = x^2 - y^2

/-- A function that returns the nth number of the form 4k + 2 -/
def nth_non_diff_of_squares (n : ℕ) : ℕ :=
  4 * n - 2

/-- Theorem stating that 8006 is the 2002nd positive integer that is not the difference of two squares -/
theorem two_thousand_second_non_diff_of_squares :
  nth_non_diff_of_squares 2002 = 8006 ∧ ¬(is_diff_of_squares 8006) :=
sorry

/-- Theorem stating that numbers of the form 4k + 2 cannot be expressed as the difference of two squares -/
theorem non_diff_of_squares_form (k : ℕ) :
  ¬(is_diff_of_squares (4 * k + 2)) :=
sorry

/-- Theorem stating that 8006 is of the form 4k + 2 -/
theorem eight_thousand_six_form :
  ∃ k : ℕ, 8006 = 4 * k + 2 :=
sorry

/-- Main theorem combining the above results -/
theorem main_theorem :
  ∃ n : ℕ, n = 2002 ∧ nth_non_diff_of_squares n = 8006 ∧ ¬(is_diff_of_squares 8006) :=
sorry

end two_thousand_second_non_diff_of_squares_non_diff_of_squares_form_eight_thousand_six_form_main_theorem_l1645_164524


namespace coin_flip_probability_l1645_164515

theorem coin_flip_probability (p_tails : ℝ) (p_sequence : ℝ) : 
  p_tails = 1/2 → 
  p_sequence = 0.0625 →
  (1 - p_tails) = 1/2 := by
  sorry

end coin_flip_probability_l1645_164515


namespace company_uniforms_l1645_164520

theorem company_uniforms (num_stores : ℕ) (uniforms_per_store : ℕ) 
  (h1 : num_stores = 32) (h2 : uniforms_per_store = 4) : 
  num_stores * uniforms_per_store = 128 := by
  sorry

end company_uniforms_l1645_164520


namespace labourer_absence_proof_l1645_164584

def total_days : ℕ := 25
def daily_wage : ℚ := 2
def daily_fine : ℚ := 1/2
def total_received : ℚ := 75/2

def days_absent : ℕ := 5

theorem labourer_absence_proof :
  ∃ (days_worked : ℕ),
    days_worked + days_absent = total_days ∧
    daily_wage * days_worked - daily_fine * days_absent = total_received :=
by sorry

end labourer_absence_proof_l1645_164584


namespace min_coins_for_distribution_l1645_164518

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 70) :
  min_additional_coins friends initial_coins = 50 := by
  sorry

#eval min_additional_coins 15 70

end min_coins_for_distribution_l1645_164518


namespace fraction_simplification_l1645_164594

theorem fraction_simplification (b x : ℝ) (h : b^2 + x^2 ≠ 0) :
  (Real.sqrt (b^2 + x^2) - (x^2 - b^2) / Real.sqrt (b^2 + x^2)) / (2 * (b^2 + x^2)^2) =
  b^2 / (b^2 + x^2)^(5/2) := by
  sorry

end fraction_simplification_l1645_164594


namespace largest_valid_sample_size_l1645_164536

def population : ℕ := 36

def is_valid_sample_size (X : ℕ) : Prop :=
  (population % X = 0) ∧ (population % (X + 1) ≠ 0)

theorem largest_valid_sample_size :
  ∃ (X : ℕ), is_valid_sample_size X ∧ ∀ (Y : ℕ), Y > X → ¬is_valid_sample_size Y :=
by
  sorry

end largest_valid_sample_size_l1645_164536


namespace min_value_theorem_l1645_164555

theorem min_value_theorem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 1/a' + 1/b' + 1/c' ≥ 9/4) :=
by sorry

end min_value_theorem_l1645_164555


namespace multiple_root_equation_l1645_164553

/-- The equation x^4 + p^2*x + q = 0 has a multiple root if and only if p = 2 and q = 3, where p and q are positive prime numbers. -/
theorem multiple_root_equation (p q : ℕ) : 
  (Prime p ∧ Prime q ∧ 0 < p ∧ 0 < q) →
  (∃ (x : ℝ), (x^4 + p^2*x + q = 0 ∧ 
    ∃ (y : ℝ), y ≠ x ∧ y^4 + p^2*y + q = 0 ∧
    (∀ (z : ℝ), z^4 + p^2*z + q = 0 → z = x ∨ z = y))) ↔ 
  (p = 2 ∧ q = 3) :=
by sorry

end multiple_root_equation_l1645_164553


namespace beef_weight_loss_percentage_l1645_164513

/-- Proves that a side of beef with given initial and final weights loses approximately 30% of its weight during processing -/
theorem beef_weight_loss_percentage (initial_weight final_weight : ℝ) 
  (h1 : initial_weight = 714.2857142857143)
  (h2 : final_weight = 500) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((initial_weight - final_weight) / initial_weight * 100) - 30| < ε :=
sorry

end beef_weight_loss_percentage_l1645_164513


namespace steve_has_dimes_l1645_164579

/-- Represents the types of coins in US currency --/
inductive USCoin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a US coin in cents --/
def coin_value (c : USCoin) : ℕ :=
  match c with
  | USCoin.Penny => 1
  | USCoin.Nickel => 5
  | USCoin.Dime => 10
  | USCoin.Quarter => 25

/-- Theorem: Given the conditions, Steve must have 26 dimes --/
theorem steve_has_dimes (total_coins : ℕ) (total_value : ℕ) (majority_coin_count : ℕ)
    (h_total_coins : total_coins = 36)
    (h_total_value : total_value = 310)
    (h_majority_coin_count : majority_coin_count = 26)
    (h_two_types : ∃ (c1 c2 : USCoin), c1 ≠ c2 ∧
      ∃ (n1 n2 : ℕ), n1 + n2 = total_coins ∧
        n1 * coin_value c1 + n2 * coin_value c2 = total_value ∧
        (n1 = majority_coin_count ∨ n2 = majority_coin_count)) :
    ∃ (other_coin : USCoin), other_coin ≠ USCoin.Dime ∧
      majority_coin_count * coin_value USCoin.Dime +
      (total_coins - majority_coin_count) * coin_value other_coin = total_value :=
  sorry

end steve_has_dimes_l1645_164579


namespace poll_size_l1645_164567

theorem poll_size (total : ℕ) (women_in_favor_percent : ℚ) (women_opposed : ℕ) : 
  (2 * women_opposed : ℚ) / (1 - women_in_favor_percent) = total →
  women_in_favor_percent = 35 / 100 →
  women_opposed = 39 →
  total = 120 := by
sorry

end poll_size_l1645_164567


namespace sum_of_divisors_36_l1645_164541

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_36 : sum_of_divisors 36 = 91 := by
  sorry

end sum_of_divisors_36_l1645_164541


namespace equal_distribution_l1645_164560

theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) : 
  total_amount = 42900 →
  num_persons = 22 →
  amount_per_person = total_amount / num_persons →
  amount_per_person = 1950 := by
  sorry

end equal_distribution_l1645_164560


namespace deductive_reasoning_examples_l1645_164502

-- Define a type for the reasoning examples
inductive ReasoningExample
  | example1
  | example2
  | example3
  | example4

-- Define a function to check if a reasoning example is deductive
def isDeductive : ReasoningExample → Bool
  | ReasoningExample.example1 => false
  | ReasoningExample.example2 => true
  | ReasoningExample.example3 => false
  | ReasoningExample.example4 => true

-- Theorem statement
theorem deductive_reasoning_examples :
  ∀ (e : ReasoningExample), isDeductive e ↔ (e = ReasoningExample.example2 ∨ e = ReasoningExample.example4) := by
  sorry


end deductive_reasoning_examples_l1645_164502


namespace sqrt_fifteen_over_two_equals_half_sqrt_thirty_l1645_164532

theorem sqrt_fifteen_over_two_equals_half_sqrt_thirty : 
  Real.sqrt (15 / 2) = (1 / 2) * Real.sqrt 30 := by
  sorry

end sqrt_fifteen_over_two_equals_half_sqrt_thirty_l1645_164532


namespace dividend_divisor_remainder_l1645_164597

theorem dividend_divisor_remainder (x y : ℕ+) :
  (x : ℝ) / (y : ℝ) = 96.12 →
  (x : ℝ) % (y : ℝ) = 1.44 →
  y = 12 := by
  sorry

end dividend_divisor_remainder_l1645_164597


namespace extremal_points_sum_bound_l1645_164500

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + 1)

theorem extremal_points_sum_bound {a : ℝ} (ha : a > 0) 
  (x₁ x₂ : ℝ) (h_extremal : ∀ x, x ≠ x₁ → x ≠ x₂ → (deriv (f a)) x ≠ 0) :
  f a x₁ + f a x₂ < Real.exp 1 := by
  sorry

end extremal_points_sum_bound_l1645_164500


namespace log_greater_than_square_near_zero_l1645_164547

theorem log_greater_than_square_near_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x → x < δ → Real.log (1 + x) > x^2 := by
  sorry

end log_greater_than_square_near_zero_l1645_164547


namespace binomial_unique_parameters_l1645_164561

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a linear transformation of a binomial random variable -/
def expectation (X : BinomialRV) (a b : ℝ) : ℝ := a * X.n * X.p + b

/-- The variance of a linear transformation of a binomial random variable -/
def variance (X : BinomialRV) (a : ℝ) : ℝ := a^2 * X.n * X.p * (1 - X.p)

/-- Theorem: If E(3X + 2) = 9.2 and D(3X + 2) = 12.96 for X ~ B(n, p), then n = 6 and p = 0.4 -/
theorem binomial_unique_parameters (X : BinomialRV) 
  (h2 : expectation X 3 2 = 9.2)
  (h3 : variance X 3 = 12.96) : 
  X.n = 6 ∧ X.p = 0.4 := by
  sorry

end binomial_unique_parameters_l1645_164561


namespace expected_threes_eight_sided_dice_l1645_164566

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The probability of rolling a 3 on a single die -/
def p : ℚ := 1 / n

/-- The probability of not rolling a 3 on a single die -/
def q : ℚ := 1 - p

/-- The expected number of 3's when rolling two n-sided dice -/
def expected_threes (n : ℕ) : ℚ := 
  2 * (p * p) + 1 * (2 * p * q) + 0 * (q * q)

/-- Theorem: The expected number of 3's when rolling two 8-sided dice is 1/4 -/
theorem expected_threes_eight_sided_dice : expected_threes n = 1/4 := by
  sorry

end expected_threes_eight_sided_dice_l1645_164566


namespace stratified_sampling_total_components_l1645_164535

theorem stratified_sampling_total_components :
  let total_sample_size : ℕ := 45
  let sample_size_A : ℕ := 20
  let sample_size_C : ℕ := 10
  let num_B : ℕ := 300
  let num_C : ℕ := 200
  let num_A : ℕ := (total_sample_size * (num_B + num_C)) / (total_sample_size - sample_size_A - sample_size_C)
  num_A + num_B + num_C = 900 := by
  sorry


end stratified_sampling_total_components_l1645_164535


namespace exactly_one_true_iff_or_and_not_and_l1645_164569

theorem exactly_one_true_iff_or_and_not_and (p q : Prop) :
  ((p ∨ q) ∧ ¬(p ∧ q)) ↔ (p ∨ q) ∧ ¬(p ↔ q) := by
  sorry

end exactly_one_true_iff_or_and_not_and_l1645_164569


namespace base_prime_representation_441_l1645_164512

/-- Base prime representation of a natural number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ := sorry

/-- The list of primes up to a given number -/
def PrimesUpTo (n : ℕ) : List ℕ := sorry

theorem base_prime_representation_441 :
  let n := 441
  let primes := PrimesUpTo 7
  BasePrimeRepresentation n = [0, 2, 2, 0] ∧ 
  n = 3^2 * 7^2 ∧
  primes = [2, 3, 5, 7] := by sorry

end base_prime_representation_441_l1645_164512


namespace de_Bruijn_Erdos_l1645_164598

/-- A graph is a pair of a vertex set and an edge relation -/
structure Graph (V : Type) :=
  (edge : V → V → Prop)

/-- The chromatic number of a graph is the smallest number of colors needed to color the graph -/
def chromaticNumber {V : Type} (G : Graph V) : ℕ := sorry

/-- A subgraph of G induced by a subset of vertices -/
def inducedSubgraph {V : Type} (G : Graph V) (S : Set V) : Graph S := sorry

/-- A graph is finite if its vertex set is finite -/
def isFinite {V : Type} (G : Graph V) : Prop := sorry

theorem de_Bruijn_Erdos {V : Type} (G : Graph V) (k : ℕ) :
  (∀ (S : Set V), isFinite (inducedSubgraph G S) → chromaticNumber (inducedSubgraph G S) ≤ k) →
  chromaticNumber G ≤ k := by sorry

end de_Bruijn_Erdos_l1645_164598


namespace element_in_set_implies_a_values_l1645_164519

theorem element_in_set_implies_a_values (a : ℝ) : 
  -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end element_in_set_implies_a_values_l1645_164519


namespace book_reading_fraction_l1645_164556

theorem book_reading_fraction (total_pages : ℝ) (pages_read_more : ℝ) : 
  total_pages = 270.00000000000006 →
  pages_read_more = 90 →
  (total_pages / 2 + pages_read_more / 2) / total_pages = 2 / 3 := by
sorry

end book_reading_fraction_l1645_164556


namespace no_prime_sum_53_l1645_164578

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_sum_53 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 := by
  sorry

end no_prime_sum_53_l1645_164578
