import Mathlib

namespace function_always_positive_l4076_407673

theorem function_always_positive (k : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (k - 2) * x + 2 * |k| - 1 > 0) ↔ k > 5/4 := by
  sorry

end function_always_positive_l4076_407673


namespace garden_flowers_count_l4076_407697

/-- Represents a rectangular garden with a rose planted in it. -/
structure Garden where
  columns : ℕ
  rows : ℕ
  rose_col_left : ℕ
  rose_col_right : ℕ
  rose_row_front : ℕ
  rose_row_back : ℕ

/-- The total number of flowers in the garden. -/
def total_flowers (g : Garden) : ℕ := g.columns * g.rows

/-- Theorem stating the total number of flowers in the specific garden configuration. -/
theorem garden_flowers_count :
  ∀ g : Garden,
  g.rose_col_left = 9 →
  g.rose_col_right = 13 →
  g.rose_row_front = 7 →
  g.rose_row_back = 16 →
  g.columns = g.rose_col_left + g.rose_col_right - 1 →
  g.rows = g.rose_row_front + g.rose_row_back - 1 →
  total_flowers g = 462 := by
  sorry

#check garden_flowers_count

end garden_flowers_count_l4076_407697


namespace c_paisa_per_a_rupee_l4076_407692

/-- Represents the share of money for each person in rupees -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : Shares) : Prop :=
  s.b = 0.65 * s.a ∧  -- For each Rs. A has, B has 65 paisa
  s.c = 32 ∧  -- C's share is Rs. 32
  s.a + s.b + s.c = 164  -- The total sum of money is Rs. 164

/-- The theorem to be proved -/
theorem c_paisa_per_a_rupee (s : Shares) 
  (h : problem_conditions s) : (s.c * 100) / s.a = 40 := by
  sorry


end c_paisa_per_a_rupee_l4076_407692


namespace infinite_series_sum_l4076_407689

theorem infinite_series_sum : 
  (∑' n : ℕ, n / (5 ^ n : ℝ)) = 5 / 16 := by sorry

end infinite_series_sum_l4076_407689


namespace total_people_is_803_l4076_407616

/-- The number of parents in the program -/
def num_parents : ℕ := 105

/-- The number of pupils in the program -/
def num_pupils : ℕ := 698

/-- The total number of people in the program -/
def total_people : ℕ := num_parents + num_pupils

/-- Theorem stating that the total number of people in the program is 803 -/
theorem total_people_is_803 : total_people = 803 := by
  sorry

end total_people_is_803_l4076_407616


namespace train_speed_l4076_407674

/-- The speed of a train passing a platform -/
theorem train_speed (train_length platform_length : ℝ) (time : ℝ) 
  (h1 : train_length = 50)
  (h2 : platform_length = 100)
  (h3 : time = 10) :
  (train_length + platform_length) / time = 15 := by
  sorry

end train_speed_l4076_407674


namespace white_tshirts_per_pack_is_five_l4076_407627

/-- The number of white T-shirts in one pack -/
def white_tshirts_per_pack : ℕ := sorry

/-- The number of packs of white T-shirts bought -/
def white_packs : ℕ := 2

/-- The number of packs of blue T-shirts bought -/
def blue_packs : ℕ := 4

/-- The number of blue T-shirts in one pack -/
def blue_tshirts_per_pack : ℕ := 3

/-- The cost of one T-shirt in dollars -/
def cost_per_tshirt : ℕ := 3

/-- The total cost of all T-shirts in dollars -/
def total_cost : ℕ := 66

theorem white_tshirts_per_pack_is_five :
  white_tshirts_per_pack = 5 :=
by
  sorry

#check white_tshirts_per_pack_is_five

end white_tshirts_per_pack_is_five_l4076_407627


namespace soda_price_calculation_l4076_407626

def initial_amount : ℕ := 500
def rice_packets : ℕ := 2
def rice_price : ℕ := 20
def wheat_packets : ℕ := 3
def wheat_price : ℕ := 25
def remaining_balance : ℕ := 235

theorem soda_price_calculation :
  ∃ (soda_price : ℕ),
    initial_amount - (rice_packets * rice_price + wheat_packets * wheat_price + soda_price) = remaining_balance ∧
    soda_price = 150 := by
  sorry

end soda_price_calculation_l4076_407626


namespace vector_equation_vectors_parallel_l4076_407609

/-- Given vectors in R² --/
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![-1, 2]
def c : Fin 2 → ℚ := ![4, 1]

/-- Theorem for part 1 --/
theorem vector_equation :
  a = (5/9 : ℚ) • b + (8/9 : ℚ) • c := by sorry

/-- Helper function to check if two vectors are parallel --/
def are_parallel (v w : Fin 2 → ℚ) : Prop :=
  v 0 * w 1 = v 1 * w 0

/-- Theorem for part 2 --/
theorem vectors_parallel :
  are_parallel (a + (-16/3 : ℚ) • c) (2 • b - a) := by sorry

end vector_equation_vectors_parallel_l4076_407609


namespace root_range_implies_a_range_l4076_407687

theorem root_range_implies_a_range (a : ℝ) :
  (∃ α β : ℝ, 5 * α^2 - 7 * α - a = 0 ∧
              5 * β^2 - 7 * β - a = 0 ∧
              -1 < α ∧ α < 0 ∧
              1 < β ∧ β < 2) →
  (0 < a ∧ a < 6) := by
sorry

end root_range_implies_a_range_l4076_407687


namespace internet_price_difference_l4076_407686

/-- Represents the internet service with speed and price -/
structure InternetService where
  speed : ℕ  -- in Mbps
  price : ℕ  -- in dollars

/-- The problem setup -/
def internetProblem : Prop :=
  ∃ (current twentyMbps thirtyMbps : InternetService),
    -- Current service
    current.speed = 10 ∧ current.price = 20 ∧
    -- 30 Mbps service
    thirtyMbps.speed = 30 ∧ thirtyMbps.price = 2 * current.price ∧
    -- 20 Mbps service
    twentyMbps.speed = 20 ∧ twentyMbps.price > current.price ∧
    -- Yearly savings
    (thirtyMbps.price - twentyMbps.price) * 12 = 120 ∧
    -- The statement to prove
    twentyMbps.price = current.price + 10

theorem internet_price_difference :
  internetProblem :=
sorry

end internet_price_difference_l4076_407686


namespace remainder_theorem_l4076_407667

theorem remainder_theorem (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end remainder_theorem_l4076_407667


namespace geometric_progression_identity_l4076_407683

/-- If a, b, c form a geometric progression, then (a+b+c)(a-b+c) = a^2 + b^2 + c^2 -/
theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a*c) :
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := by
  sorry


end geometric_progression_identity_l4076_407683


namespace red_yellow_flowers_l4076_407670

theorem red_yellow_flowers (total : ℕ) (yellow_white : ℕ) (red_white : ℕ) (red_excess : ℕ) :
  total = 44 →
  yellow_white = 13 →
  red_white = 14 →
  red_excess = 4 →
  ∃ (red_yellow : ℕ), red_yellow = 17 ∧
    total = yellow_white + red_white + red_yellow ∧
    red_white + red_yellow = yellow_white + red_white + red_excess :=
by sorry

end red_yellow_flowers_l4076_407670


namespace pushup_sets_l4076_407601

theorem pushup_sets (total_pushups : ℕ) (sets : ℕ) (reduction : ℕ) : 
  total_pushups = 40 → sets = 3 → reduction = 5 → 
  ∃ x : ℕ, x + x + (x - reduction) = total_pushups ∧ x = 15 := by
  sorry

end pushup_sets_l4076_407601


namespace train_speed_l4076_407651

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 300 → 
  time = 18 → 
  speed = (length / time) * 3.6 → 
  speed = 60 := by sorry

end train_speed_l4076_407651


namespace sum_sequence_existence_l4076_407656

theorem sum_sequence_existence (n : ℕ) (h : n ≤ 2^1000000) :
  ∃ (k : ℕ) (x : ℕ → ℕ),
    x 0 = 1 ∧
    k ≤ 1100000 ∧
    x k = n ∧
    ∀ i ∈ Finset.range (k + 1), i ≠ 0 →
      ∃ r s, r ≤ s ∧ s < i ∧ x i = x r + x s :=
by sorry

end sum_sequence_existence_l4076_407656


namespace circle_radius_l4076_407671

theorem circle_radius (P Q : ℝ) (h : P / Q = 40 / Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ P = Real.pi * r^2 ∧ Q = 2 * Real.pi * r ∧ r = 80 / Real.pi :=
sorry

end circle_radius_l4076_407671


namespace jose_investment_is_4500_l4076_407690

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given shop investment scenario --/
def calculate_jose_investment (s : ShopInvestment) : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that Jose's investment is 4500 given the problem conditions --/
theorem jose_investment_is_4500 (s : ShopInvestment) 
  (h1 : s.tom_investment = 3000)
  (h2 : s.jose_join_delay = 2)
  (h3 : s.total_profit = 5400)
  (h4 : s.jose_profit = 3000) :
  calculate_jose_investment s = 4500 := by
  sorry

#check jose_investment_is_4500

end jose_investment_is_4500_l4076_407690


namespace x950x_divisible_by_36_l4076_407632

def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def form_x950x (x : ℕ) : ℕ :=
  x * 10000 + 9500 + x

theorem x950x_divisible_by_36 :
  ∃ (x : ℕ), 
    x < 10 ∧ 
    is_five_digit_number (form_x950x x) ∧ 
    (form_x950x x) % 36 = 0 ↔ 
    x = 2 := by
  sorry

end x950x_divisible_by_36_l4076_407632


namespace circle_area_three_fourths_l4076_407652

/-- Given a circle where three times the reciprocal of its circumference 
    equals its diameter, prove that its area is 3/4 -/
theorem circle_area_three_fourths (r : ℝ) (h : 3 * (1 / (2 * π * r)) = 2 * r) : 
  π * r^2 = 3/4 := by
  sorry

end circle_area_three_fourths_l4076_407652


namespace intersection_locus_is_circle_l4076_407665

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hand of the watch -/
structure WatchHand where
  axis : Point
  angularVelocity : ℝ
  initialAngle : ℝ

/-- Represents the watch configuration -/
structure Watch where
  secondHand : WatchHand
  stopwatchHand : WatchHand

/-- The locus of intersection points between extended watch hands -/
def intersectionLocus (w : Watch) (t : ℝ) : Point :=
  sorry

/-- Theorem stating that the intersection locus forms a circle -/
theorem intersection_locus_is_circle (w : Watch) : 
  ∃ (center : Point) (radius : ℝ), 
    ∀ t, ∃ θ, intersectionLocus w t = Point.mk (center.x + radius * Real.cos θ) (center.y + radius * Real.sin θ) :=
  sorry

end intersection_locus_is_circle_l4076_407665


namespace halloween_jelly_beans_l4076_407680

/-- Given the conditions of a Halloween jelly bean distribution, 
    prove that the total number of children at the celebration is 40. -/
theorem halloween_jelly_beans 
  (initial_jelly_beans : ℕ)
  (remaining_jelly_beans : ℕ)
  (allowed_percentage : ℚ)
  (jelly_beans_per_child : ℕ)
  (h1 : initial_jelly_beans = 100)
  (h2 : remaining_jelly_beans = 36)
  (h3 : allowed_percentage = 4/5)
  (h4 : jelly_beans_per_child = 2) :
  (initial_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child / allowed_percentage = 40 := by
  sorry

end halloween_jelly_beans_l4076_407680


namespace distribution_count_l4076_407617

theorem distribution_count (num_items : ℕ) (num_recipients : ℕ) : 
  num_items = 6 → num_recipients = 8 → num_recipients ^ num_items = 262144 := by
  sorry

end distribution_count_l4076_407617


namespace linear_function_m_value_l4076_407672

/-- Given a linear function y = (m^2 + 2m)x + m^2 + m - 1 + (2m - 3), prove that m = 1 -/
theorem linear_function_m_value (m : ℝ) : 
  (∃ k b, ∀ x, (m^2 + 2*m)*x + (m^2 + m - 1 + (2*m - 3)) = k*x + b) → 
  (m^2 + 2*m ≠ 0) → 
  m = 1 := by
sorry

end linear_function_m_value_l4076_407672


namespace infinite_monotone_subsequence_l4076_407699

-- Define an infinite sequence of distinct real numbers
def InfiniteSequence := ℕ → ℝ

-- Define the property that all elements in the sequence are distinct
def AllDistinct (seq : InfiniteSequence) : Prop :=
  ∀ i j : ℕ, i ≠ j → seq i ≠ seq j

-- Define a strictly increasing subsequence
def StrictlyIncreasing (subseq : ℕ → ℕ) (seq : InfiniteSequence) : Prop :=
  ∀ i j : ℕ, i < j → seq (subseq i) < seq (subseq j)

-- Define a strictly decreasing subsequence
def StrictlyDecreasing (subseq : ℕ → ℕ) (seq : InfiniteSequence) : Prop :=
  ∀ i j : ℕ, i < j → seq (subseq i) > seq (subseq j)

-- The main theorem
theorem infinite_monotone_subsequence
  (seq : InfiniteSequence) (h : AllDistinct seq) :
  (∃ subseq : ℕ → ℕ, StrictlyIncreasing subseq seq) ∨
  (∃ subseq : ℕ → ℕ, StrictlyDecreasing subseq seq) :=
sorry

end infinite_monotone_subsequence_l4076_407699


namespace absolute_value_equation_solution_difference_l4076_407610

theorem absolute_value_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ + 3| = 15) ∧ 
  (|x₂ + 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧ 
  (x₁ - x₂ = 30 ∨ x₂ - x₁ = 30) := by
  sorry

end absolute_value_equation_solution_difference_l4076_407610


namespace tree_height_problem_l4076_407661

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 24 →  -- One tree is 24 feet taller than the other
  h₂ / h₁ = 2 / 3 →  -- The heights are in the ratio 2:3
  h₁ = 72 :=  -- The height of the taller tree is 72 feet
by
  sorry

end tree_height_problem_l4076_407661


namespace max_value_of_derived_function_l4076_407628

/-- Given a function f(x) = a * sin(x) + b with max value 1 and min value -7,
    prove that the max value of b * sin²(x) - a * cos²(x) is either 4 or -3 -/
theorem max_value_of_derived_function 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * Real.sin x + b)
  (h_max : ∀ x, f x ≤ 1)
  (h_min : ∀ x, f x ≥ -7)
  : (∃ x, b * Real.sin x ^ 2 - a * Real.cos x ^ 2 = 4) ∨ 
    (∃ x, b * Real.sin x ^ 2 - a * Real.cos x ^ 2 = -3) :=
sorry

end max_value_of_derived_function_l4076_407628


namespace sin_n_equals_cos_675_l4076_407676

theorem sin_n_equals_cos_675 (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * π / 180) = Real.cos (675 * π / 180) → n = 45 := by
  sorry

end sin_n_equals_cos_675_l4076_407676


namespace simplify_and_ratio_l4076_407633

theorem simplify_and_ratio (k : ℝ) : ∃ (a b : ℝ), 
  (6 * k^2 + 18) / 6 = a * k^2 + b ∧ a / b = 1 / 3 :=
by
  sorry

end simplify_and_ratio_l4076_407633


namespace expression_simplification_l4076_407636

theorem expression_simplification (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  3 * m^2 * n + 2 * (2 * m * n^2 - 3 * m^2 * n) - 3 * (m * n^2 - m^2 * n) = 4 := by
  sorry

end expression_simplification_l4076_407636


namespace honeydews_left_l4076_407606

/-- Represents the problem of Darryl's melon sales --/
structure MelonSales where
  cantaloupe_price : ℕ
  honeydew_price : ℕ
  initial_cantaloupes : ℕ
  initial_honeydews : ℕ
  dropped_cantaloupes : ℕ
  rotten_honeydews : ℕ
  final_cantaloupes : ℕ
  total_revenue : ℕ

/-- Theorem stating the number of honeydews left at the end of the day --/
theorem honeydews_left (sale : MelonSales)
  (h1 : sale.cantaloupe_price = 2)
  (h2 : sale.honeydew_price = 3)
  (h3 : sale.initial_cantaloupes = 30)
  (h4 : sale.initial_honeydews = 27)
  (h5 : sale.dropped_cantaloupes = 2)
  (h6 : sale.rotten_honeydews = 3)
  (h7 : sale.final_cantaloupes = 8)
  (h8 : sale.total_revenue = 85) :
  sale.initial_honeydews - sale.rotten_honeydews -
  ((sale.total_revenue - (sale.initial_cantaloupes - sale.dropped_cantaloupes - sale.final_cantaloupes) * sale.cantaloupe_price) / sale.honeydew_price) = 9 :=
sorry

end honeydews_left_l4076_407606


namespace toll_booth_traffic_l4076_407619

theorem toll_booth_traffic (total : ℕ) (mon : ℕ) (tues : ℕ) (wed : ℕ) (thur : ℕ) :
  total = 450 →
  mon = 50 →
  tues = mon →
  wed = 2 * mon →
  thur = wed →
  ∃ (remaining : ℕ), 
    remaining * 3 = total - (mon + tues + wed + thur) ∧
    remaining = 50 :=
by sorry

end toll_booth_traffic_l4076_407619


namespace odd_implies_abs_symmetric_abs_symmetric_not_sufficient_for_odd_l4076_407669

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The graph of |f(x)| is symmetric about the y-axis if |f(-x)| = |f(x)| for all x ∈ ℝ -/
def IsAbsSymmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem odd_implies_abs_symmetric (f : ℝ → ℝ) :
  IsOdd f → IsAbsSymmetric f :=
sorry

theorem abs_symmetric_not_sufficient_for_odd :
  ∃ f : ℝ → ℝ, IsAbsSymmetric f ∧ ¬IsOdd f :=
sorry

end odd_implies_abs_symmetric_abs_symmetric_not_sufficient_for_odd_l4076_407669


namespace box_ratio_l4076_407662

/-- Represents the number of cardboards of each type -/
structure Cardboards where
  square : ℕ
  rectangular : ℕ

/-- Represents the number of boxes of each type -/
structure Boxes where
  vertical : ℕ
  horizontal : ℕ

/-- Represents the number of cardboards used for each type of box -/
structure BoxRequirements where
  vertical_square : ℕ
  vertical_rectangular : ℕ
  horizontal_square : ℕ
  horizontal_rectangular : ℕ

/-- The main theorem stating the ratio of vertical to horizontal boxes -/
theorem box_ratio 
  (c : Cardboards) 
  (b : Boxes) 
  (r : BoxRequirements) 
  (h1 : c.rectangular = 2 * c.square)  -- Ratio of cardboards is 1:2
  (h2 : r.vertical_square * b.vertical + r.horizontal_square * b.horizontal = c.square)  -- All square cardboards are used
  (h3 : r.vertical_rectangular * b.vertical + r.horizontal_rectangular * b.horizontal = c.rectangular)  -- All rectangular cardboards are used
  : b.vertical = b.horizontal / 2 := by
  sorry

end box_ratio_l4076_407662


namespace unknown_number_proof_l4076_407666

theorem unknown_number_proof (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 192)
  (h_hcf : Nat.gcd a b = 16)
  (h_a : a = 64) :
  b = 48 := by
sorry

end unknown_number_proof_l4076_407666


namespace trigonometric_identity_l4076_407685

theorem trigonometric_identity : 
  (Real.sin (20 * π / 180) / Real.cos (20 * π / 180)) + 
  (Real.sin (40 * π / 180) / Real.cos (40 * π / 180)) + 
  Real.tan (60 * π / 180) * Real.tan (20 * π / 180) * Real.tan (40 * π / 180) = 
  Real.sqrt 3 := by sorry

end trigonometric_identity_l4076_407685


namespace only_zhong_symmetrical_l4076_407664

/-- Represents a Chinese character --/
inductive ChineseCharacter
| ai    -- 爱
| wo    -- 我
| zhong -- 中
| guo   -- 国

/-- Determines if a Chinese character is symmetrical --/
def is_symmetrical (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.zhong => True
  | _ => False

/-- Theorem stating that among the given characters, only 中 (zhong) is symmetrical --/
theorem only_zhong_symmetrical :
  ∀ c : ChineseCharacter,
    is_symmetrical c ↔ c = ChineseCharacter.zhong :=
by sorry

end only_zhong_symmetrical_l4076_407664


namespace two_true_propositions_l4076_407693

theorem two_true_propositions :
  let P : ℝ → Prop := λ x => x > -3
  let Q : ℝ → Prop := λ x => x > -6
  let original := ∀ x, P x → Q x
  let converse := ∀ x, Q x → P x
  let inverse := ∀ x, ¬(P x) → ¬(Q x)
  let contrapositive := ∀ x, ¬(Q x) → ¬(P x)
  (original ∧ contrapositive ∧ ¬converse ∧ ¬inverse) ∨
  (original ∧ contrapositive ∧ converse ∧ ¬inverse) ∨
  (original ∧ contrapositive ∧ ¬converse ∧ inverse) :=
by
  sorry


end two_true_propositions_l4076_407693


namespace not_necessarily_congruent_with_two_sides_one_angle_l4076_407698

/-- Triangle represented by three points in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Predicate for triangle congruence -/
def IsCongruent (t1 t2 : Triangle) : Prop :=
  sorry

/-- Predicate for two sides and one angle being equal -/
def HasTwoSidesOneAngleEqual (t1 t2 : Triangle) : Prop :=
  sorry

/-- Theorem stating that triangles with two corresponding sides and one corresponding angle equal
    are not necessarily congruent -/
theorem not_necessarily_congruent_with_two_sides_one_angle :
  ∃ t1 t2 : Triangle, HasTwoSidesOneAngleEqual t1 t2 ∧ ¬IsCongruent t1 t2 :=
sorry

end not_necessarily_congruent_with_two_sides_one_angle_l4076_407698


namespace log_base_three_squared_l4076_407641

theorem log_base_three_squared (m : ℝ) (b : ℝ) (h : 3^m = b) :
  Real.log b / Real.log (3^2) = m / 2 := by
  sorry

end log_base_three_squared_l4076_407641


namespace problem_solution_l4076_407658

theorem problem_solution : ∃ x : ℕ, x = 13 ∧ (4 * x) / 8 = 6 ∧ (4 * x) % 8 = 4 := by
  sorry

end problem_solution_l4076_407658


namespace remainder_sum_mod_seven_l4076_407637

theorem remainder_sum_mod_seven : 
  (2 * (4561 + 4562 + 4563 + 4564 + 4565)) % 7 = 6 := by
  sorry

end remainder_sum_mod_seven_l4076_407637


namespace initial_distance_is_50_l4076_407657

/-- The initial distance between two people walking towards each other -/
def initial_distance (speed : ℝ) (distance_walked : ℝ) : ℝ :=
  2 * distance_walked

/-- Theorem: The initial distance between Fred and Sam is 50 miles -/
theorem initial_distance_is_50 (fred_speed sam_speed : ℝ) (sam_distance : ℝ) :
  fred_speed = 5 →
  sam_speed = 5 →
  sam_distance = 25 →
  initial_distance sam_speed sam_distance = 50 :=
by
  sorry

#check initial_distance_is_50

end initial_distance_is_50_l4076_407657


namespace integer_pair_sum_l4076_407605

theorem integer_pair_sum (m n : ℤ) (h : (m^2 + m*n + n^2) / (m + 2*n) = 13/3) : 
  m + 2*n = 9 := by
sorry

end integer_pair_sum_l4076_407605


namespace shortest_side_of_right_triangle_l4076_407643

theorem shortest_side_of_right_triangle (a b c : ℝ) :
  a = 5 →
  b = 12 →
  c^2 = a^2 + b^2 →
  c ≥ a ∧ c ≥ b →
  a = min a b := by
  sorry

end shortest_side_of_right_triangle_l4076_407643


namespace percentage_problem_l4076_407660

theorem percentage_problem : 
  ∃ (P : ℝ), (0.1 * 30 + P * 50 = 10.5) ∧ (P = 0.15) := by
  sorry

end percentage_problem_l4076_407660


namespace cuboid_edge_length_l4076_407653

/-- Given a cuboid with edges of 4 cm, x cm, and 6 cm, and a volume of 120 cm³, prove that x = 5 cm. -/
theorem cuboid_edge_length (x : ℝ) : 
  x > 0 → 4 * x * 6 = 120 → x = 5 := by sorry

end cuboid_edge_length_l4076_407653


namespace sequence_properties_l4076_407621

def S (n : ℕ) : ℤ := -n^2 + 7*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem sequence_properties :
  (∀ n : ℕ, a n = -2*n + 8) ∧
  (∀ n : ℕ, n > 4 → a n < 0) ∧
  (∀ n : ℕ, S n ≤ S 3 ∧ S n ≤ S 4) :=
by sorry

end sequence_properties_l4076_407621


namespace polynomial_value_constraint_l4076_407642

theorem polynomial_value_constraint (a b c : ℤ) : 
  (b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) → 
  (b + c + a ≠ 2009) := by
sorry

end polynomial_value_constraint_l4076_407642


namespace limit_at_negative_four_l4076_407694

/-- The limit of (2x^2 + 6x - 8)/(x + 4) as x approaches -4 is -10 -/
theorem limit_at_negative_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ →
    |(2*x^2 + 6*x - 8)/(x + 4) + 10| < ε :=
by sorry

end limit_at_negative_four_l4076_407694


namespace gcd_g_102_103_l4076_407675

def g (x : ℤ) : ℤ := x^2 - x + 2007

theorem gcd_g_102_103 : Int.gcd (g 102) (g 103) = 3 := by
  sorry

end gcd_g_102_103_l4076_407675


namespace fixed_point_of_exponential_function_l4076_407696

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 2)
  f 2 = 3 := by
  sorry

end fixed_point_of_exponential_function_l4076_407696


namespace benjie_margo_age_difference_l4076_407629

/-- The age difference between Benjie and Margo -/
def ageDifference (benjieAge : ℕ) (margoFutureAge : ℕ) (yearsTillMargoFutureAge : ℕ) : ℕ :=
  benjieAge - (margoFutureAge - yearsTillMargoFutureAge)

/-- Theorem stating the age difference between Benjie and Margo -/
theorem benjie_margo_age_difference :
  ageDifference 6 4 3 = 5 := by
  sorry

end benjie_margo_age_difference_l4076_407629


namespace complex_product_pure_imaginary_l4076_407620

theorem complex_product_pure_imaginary (a : ℝ) : 
  (Complex.I + 1) * (Complex.I * a + 1) = Complex.I * (Complex.I.im * b) → a = 1 := by
  sorry

end complex_product_pure_imaginary_l4076_407620


namespace quadrilateral_diagonal_intersection_l4076_407622

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_intersection 
  (q : Quadrilateral) 
  (hConvex : isConvex q) 
  (hAB : distance q.A q.B = 12)
  (hCD : distance q.C q.D = 15)
  (hAC : distance q.A q.C = 18)
  (E : Point)
  (hE : E = lineIntersection q.A q.C q.B q.D)
  (hAreas : triangleArea q.A E q.D = triangleArea q.B E q.C) :
  distance q.A E = 8 := by sorry

end quadrilateral_diagonal_intersection_l4076_407622


namespace square_sum_given_diff_and_product_l4076_407618

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 12) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 162 := by
sorry

end square_sum_given_diff_and_product_l4076_407618


namespace emilys_coin_collection_value_l4076_407650

/-- Proves that given the conditions of Emily's coin collection, the total value is $128 -/
theorem emilys_coin_collection_value :
  ∀ (total_coins : ℕ) 
    (first_type_count : ℕ) 
    (first_type_total_value : ℝ) 
    (second_type_count : ℕ),
  total_coins = 20 →
  first_type_count = 8 →
  first_type_total_value = 32 →
  second_type_count = total_coins - first_type_count →
  (second_type_count * (first_type_total_value / first_type_count) * 2 + first_type_total_value = 128) :=
by
  sorry


end emilys_coin_collection_value_l4076_407650


namespace sequence_constant_condition_general_term_l4076_407612

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The sequence a_n -/
noncomputable def a (x y : ℝ) : ℕ → ℝ
| 0 => x
| 1 => y
| (n + 2) => (a x y (n + 1) * a x y n + 1) / (a x y (n + 1) + a x y n)

theorem sequence_constant_condition (x y : ℝ) :
  (∃ n₀ : ℕ, ∀ n ≥ n₀, a x y (n + 1) = a x y n) ↔ (abs x = 1 ∧ y ≠ -x) :=
sorry

theorem general_term (x y : ℝ) (n : ℕ) :
  a x y n = ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) + (x + 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) /
            ((x + 1)^(fib (n - 2)) * (y + 1)^(fib (n - 1)) - (x - 1)^(fib (n - 2)) * (y - 1)^(fib (n - 1))) :=
sorry

end sequence_constant_condition_general_term_l4076_407612


namespace zero_success_probability_l4076_407640

/-- The probability of 0 successes in 7 Bernoulli trials with success probability 2/7 -/
def prob_zero_success (n : ℕ) (p : ℚ) : ℚ :=
  (1 - p) ^ n

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The probability of success in a single trial -/
def success_prob : ℚ := 2/7

theorem zero_success_probability :
  prob_zero_success num_trials success_prob = (5/7) ^ 7 := by
  sorry

end zero_success_probability_l4076_407640


namespace part1_part2_l4076_407691

-- Define the concept of l-increasing function
def is_l_increasing (f : ℝ → ℝ) (D : Set ℝ) (M : Set ℝ) (l : ℝ) : Prop :=
  l ≠ 0 ∧ (∀ x ∈ M, x + l ∈ D ∧ f (x + l) ≥ f x)

-- Part 1
theorem part1 (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Ici (-1), f x = x^2) →
  is_l_increasing f (Set.Ici (-1)) (Set.Ici (-1)) m →
  m ≥ 2 := by sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x ≥ 0, f x = |x - a^2| - a^2) →
  is_l_increasing f Set.univ Set.univ 8 →
  -2 ≤ a ∧ a ≤ 2 := by sorry

end part1_part2_l4076_407691


namespace bernoulli_inequality_l4076_407630

theorem bernoulli_inequality (x : ℝ) (m : ℕ+) (h : x > -1) :
  (1 + x)^(m : ℕ) ≥ 1 + m * x :=
by sorry

end bernoulli_inequality_l4076_407630


namespace student_composition_l4076_407668

/-- The number of ways to select participants from a group of students -/
def selectionWays (males females : ℕ) : ℕ :=
  males * (males - 1) * females

theorem student_composition :
  ∃ (males females : ℕ),
    males + females = 8 ∧
    selectionWays males females = 90 →
    males = 3 ∧ females = 5 := by
  sorry

end student_composition_l4076_407668


namespace imaginary_part_of_square_l4076_407644

theorem imaginary_part_of_square : Complex.im ((1 - 4 * Complex.I) ^ 2) = -8 := by
  sorry

end imaginary_part_of_square_l4076_407644


namespace bobby_candy_count_l4076_407695

/-- The number of candy pieces Bobby ate initially -/
def initial_candy : ℕ := 26

/-- The number of additional candy pieces Bobby ate -/
def additional_candy : ℕ := 17

/-- The total number of candy pieces Bobby ate -/
def total_candy : ℕ := initial_candy + additional_candy

theorem bobby_candy_count : total_candy = 43 := by
  sorry

end bobby_candy_count_l4076_407695


namespace geometric_sum_value_l4076_407663

/-- Sum of a geometric series with 15 terms, first term 4/5, and common ratio 4/5 -/
def geometricSum : ℚ :=
  let a : ℚ := 4/5
  let r : ℚ := 4/5
  let n : ℕ := 15
  a * (1 - r^n) / (1 - r)

/-- The sum of the geometric series is equal to 117775277204/30517578125 -/
theorem geometric_sum_value : geometricSum = 117775277204/30517578125 := by
  sorry

end geometric_sum_value_l4076_407663


namespace peter_erasers_l4076_407607

theorem peter_erasers (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 35 → received = 17 → total = initial + received → total = 52 := by
  sorry

end peter_erasers_l4076_407607


namespace f_minimum_and_range_l4076_407634

/-- The function f(x) = |2x+1| + |2x-1| -/
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

theorem f_minimum_and_range :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m) ∧
  (∀ (x : ℝ), (∀ (a b : ℝ), |2*a + b| + |a| - 1/2 * |a + b| * f x ≥ 0) →
    x ∈ Set.Icc (-1/2) (1/2)) :=
by sorry

end f_minimum_and_range_l4076_407634


namespace number_problem_l4076_407638

theorem number_problem (x : ℝ) : 0.7 * x - 40 = 30 → x = 100 := by
  sorry

end number_problem_l4076_407638


namespace rental_distance_theorem_l4076_407613

/-- Calculates the distance driven given rental parameters and total cost -/
def distance_driven (daily_rate : ℚ) (mile_rate : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - daily_rate) / mile_rate

theorem rental_distance_theorem (daily_rate mile_rate total_cost : ℚ) :
  daily_rate = 29 →
  mile_rate = 0.08 →
  total_cost = 46.12 →
  distance_driven daily_rate mile_rate total_cost = 214 := by
  sorry

end rental_distance_theorem_l4076_407613


namespace expression_value_l4076_407648

theorem expression_value : 
  let a : ℤ := 2025
  let b : ℤ := a + 1
  let k : ℤ := 1
  (a^3 - 2*k*a^2*b + 3*k*a*b^2 - b^3 + k) / (a*b) = 2025 := by
  sorry

end expression_value_l4076_407648


namespace rectangle_color_theorem_l4076_407647

/-- A cell in the rectangle can be either white or black -/
inductive CellColor
  | White
  | Black

/-- The rectangle is represented as a 3 × 7 matrix of cell colors -/
def Rectangle := Matrix (Fin 3) (Fin 7) CellColor

/-- A point in the rectangle, represented by its row and column -/
structure Point where
  row : Fin 3
  col : Fin 7

/-- Check if four points form a rectangle parallel to the sides of the original rectangle -/
def isParallelRectangle (p1 p2 p3 p4 : Point) : Prop :=
  (p1.row = p2.row ∧ p3.row = p4.row ∧ p1.col = p3.col ∧ p2.col = p4.col) ∨
  (p1.row = p3.row ∧ p2.row = p4.row ∧ p1.col = p2.col ∧ p3.col = p4.col)

/-- Check if all four points have the same color in the given rectangle -/
def sameColor (rect : Rectangle) (p1 p2 p3 p4 : Point) : Prop :=
  rect p1.row p1.col = rect p2.row p2.col ∧
  rect p2.row p2.col = rect p3.row p3.col ∧
  rect p3.row p3.col = rect p4.row p4.col

theorem rectangle_color_theorem (rect : Rectangle) :
  ∃ p1 p2 p3 p4 : Point,
    isParallelRectangle p1 p2 p3 p4 ∧
    sameColor rect p1 p2 p3 p4 := by
  sorry

end rectangle_color_theorem_l4076_407647


namespace team_leaders_problem_l4076_407600

theorem team_leaders_problem (m n : ℕ) :
  (10 ≥ m ∧ m > n ∧ n ≥ 4) →
  (Nat.choose (m + n) 2 * (Nat.choose m 2 + Nat.choose n 2) = 
   Nat.choose (m + n) 2 * (m * n)) →
  (m = 10 ∧ n = 6) :=
by sorry

end team_leaders_problem_l4076_407600


namespace arithmetic_harmonic_means_equal_implies_equal_values_l4076_407611

theorem arithmetic_harmonic_means_equal_implies_equal_values (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 2) 
  (h_harmonic : 2 / (1/a + 1/b) = 2) : 
  a = 2 ∧ b = 2 := by
  sorry

end arithmetic_harmonic_means_equal_implies_equal_values_l4076_407611


namespace fifteen_point_five_minutes_in_hours_l4076_407688

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ :=
  minutes * (1 / 60)

theorem fifteen_point_five_minutes_in_hours : 
  minutes_to_hours 15.5 = 930 / 3600 := by
sorry

end fifteen_point_five_minutes_in_hours_l4076_407688


namespace bellas_dancer_friends_l4076_407649

theorem bellas_dancer_friends (total_roses : ℕ) (parent_roses : ℕ) (roses_per_friend : ℕ) 
  (h1 : total_roses = 44)
  (h2 : parent_roses = 2 * 12)
  (h3 : roses_per_friend = 2) :
  (total_roses - parent_roses) / roses_per_friend = 10 := by
  sorry

end bellas_dancer_friends_l4076_407649


namespace inequality_proof_l4076_407654

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l4076_407654


namespace morning_trip_fare_correct_afternoon_trip_fare_formula_afternoon_trip_fare_specific_l4076_407602

/- Define the time periods and corresponding rates -/
def normal_mileage_rate : ℝ := 2.20
def early_morning_mileage_rate : ℝ := 2.80
def peak_mileage_rate : ℝ := 2.75
def normal_time_rate : ℝ := 0.38
def peak_time_rate : ℝ := 0.47

/- Define the fare calculation function -/
def calculate_fare (distance : ℝ) (time : ℝ) (mileage_rate : ℝ) (time_rate : ℝ) : ℝ :=
  distance * mileage_rate + time * time_rate

/- Theorem for the morning trip -/
theorem morning_trip_fare_correct :
  calculate_fare 6 10 early_morning_mileage_rate normal_time_rate = 20.6 := by sorry

/- Theorem for the afternoon trip (general formula) -/
theorem afternoon_trip_fare_formula (x : ℝ) (h : x ≤ 30) :
  calculate_fare x (x / 30 * 60) peak_mileage_rate peak_time_rate = 3.69 * x := by sorry

/- Theorem for the afternoon trip when x = 8 -/
theorem afternoon_trip_fare_specific :
  calculate_fare 8 16 peak_mileage_rate peak_time_rate = 29.52 := by sorry

end morning_trip_fare_correct_afternoon_trip_fare_formula_afternoon_trip_fare_specific_l4076_407602


namespace lines_parallel_iff_a_eq_neg_three_l4076_407645

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel_lines (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  B1 ≠ 0 ∧ B2 ≠ 0 ∧ A1 / B1 = A2 / B2

/-- The first line: 3x + ay + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + a * y + 1 = 0

/-- The second line: (a+2)x + y + a = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  (a + 2) * x + y + a = 0

/-- The theorem stating that the lines are parallel if and only if a = -3 -/
theorem lines_parallel_iff_a_eq_neg_three :
  ∃ (a : ℝ), parallel_lines 3 a 1 (a + 2) 1 a ↔ a = -3 := by sorry

end lines_parallel_iff_a_eq_neg_three_l4076_407645


namespace angle_sum_theorem_l4076_407659

-- Define the sum of angles around a point
def sum_of_angles : ℝ := 360

-- Define the four angles as functions of x
def angle1 (x : ℝ) : ℝ := 5 * x
def angle2 (x : ℝ) : ℝ := 4 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem angle_sum_theorem :
  ∃ x : ℝ, angle1 x + angle2 x + angle3 x + angle4 x = sum_of_angles ∧ x = 30 :=
by sorry

end angle_sum_theorem_l4076_407659


namespace cubic_expansion_coefficient_l4076_407684

theorem cubic_expansion_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end cubic_expansion_coefficient_l4076_407684


namespace price_per_square_foot_l4076_407624

def house_area : ℝ := 2400
def barn_area : ℝ := 1000
def total_property_value : ℝ := 333200

theorem price_per_square_foot :
  total_property_value / (house_area + barn_area) = 98 := by
sorry

end price_per_square_foot_l4076_407624


namespace smallest_number_with_conditions_l4076_407603

theorem smallest_number_with_conditions : ∃! n : ℕ, 
  (n % 11 = 0) ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) ∧
  (∀ m : ℕ, 
    (m % 11 = 0) ∧ 
    (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → m % k = 1) → 
    n ≤ m) ∧
  n = 6721 :=
by sorry

end smallest_number_with_conditions_l4076_407603


namespace solutions_for_20_l4076_407615

/-- The number of integer solutions for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

/-- Given conditions -/
axiom solution_1 : num_solutions 1 = 4
axiom solution_2 : num_solutions 2 = 8
axiom solution_3 : num_solutions 3 = 12

/-- Theorem: The number of different integer solutions for |x| + |y| = 20 is 80 -/
theorem solutions_for_20 : num_solutions 20 = 80 := by sorry

end solutions_for_20_l4076_407615


namespace sin_sum_upper_bound_l4076_407631

theorem sin_sum_upper_bound (x y z : ℝ) (hx : x ∈ Set.Icc 0 Real.pi) 
  (hy : y ∈ Set.Icc 0 Real.pi) (hz : z ∈ Set.Icc 0 Real.pi) : 
  Real.sin (x - y) + Real.sin (y - z) + Real.sin (z - x) ≤ 2 := by
  sorry

end sin_sum_upper_bound_l4076_407631


namespace second_cat_weight_l4076_407677

theorem second_cat_weight (total_weight first_weight third_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : first_weight = 2)
  (h3 : third_weight = 4) :
  total_weight - first_weight - third_weight = 7 := by
  sorry

end second_cat_weight_l4076_407677


namespace machine_a_time_proof_l4076_407646

/-- The time it takes for Machine A to finish the job alone -/
def machine_a_time : ℝ := 4

/-- The time it takes for Machine B to finish the job alone -/
def machine_b_time : ℝ := 12

/-- The time it takes for Machine C to finish the job alone -/
def machine_c_time : ℝ := 6

/-- The time it takes for all machines to finish the job together -/
def combined_time : ℝ := 2

theorem machine_a_time_proof :
  (1 / machine_a_time + 1 / machine_b_time + 1 / machine_c_time) * combined_time = 1 :=
by sorry

end machine_a_time_proof_l4076_407646


namespace chord_length_concentric_circles_l4076_407604

theorem chord_length_concentric_circles (a b : ℝ) (h1 : a > b) (h2 : a^2 - b^2 = 20) :
  ∃ c : ℝ, c = 4 * Real.sqrt 5 ∧ c^2 / 4 + b^2 = a^2 :=
by sorry

end chord_length_concentric_circles_l4076_407604


namespace even_mono_increasing_negative_l4076_407625

-- Define an even function that is monotonically increasing on [0, +∞)
def EvenMonoIncreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x ≤ f y)

-- Theorem statement
theorem even_mono_increasing_negative (f : ℝ → ℝ) (a b : ℝ) 
  (hf : EvenMonoIncreasing f) (hab : a < b) (hneg : b < 0) : 
  f a > f b := by
  sorry

end even_mono_increasing_negative_l4076_407625


namespace dot_product_ab_bc_l4076_407623

/-- Given two vectors AB and AC in 2D space, prove that their dot product with BC is -8. -/
theorem dot_product_ab_bc (AB AC : ℝ × ℝ) (h1 : AB = (4, 2)) (h2 : AC = (1, 4)) :
  AB • (AC - AB) = -8 := by
  sorry

end dot_product_ab_bc_l4076_407623


namespace bons_winning_probability_l4076_407635

/-- The probability of rolling a six on a six-sided die. -/
def probSix : ℚ := 1/6

/-- The probability of not rolling a six on a six-sided die. -/
def probNotSix : ℚ := 1 - probSix

/-- The probability that B. Bons wins the game. -/
noncomputable def probBonsWins : ℚ :=
  (probNotSix * probSix) / (1 - probNotSix * probNotSix)

theorem bons_winning_probability :
  probBonsWins = 5/11 := by sorry

end bons_winning_probability_l4076_407635


namespace playground_count_l4076_407682

theorem playground_count (x : ℤ) : 
  let known_numbers : List ℤ := [12, 1, 12, 7, 3, 8]
  let all_numbers : List ℤ := x :: known_numbers
  (all_numbers.sum / all_numbers.length : ℚ) = 7 → x = -1 :=
by sorry

end playground_count_l4076_407682


namespace range_of_a_l4076_407608

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → a ∈ Set.Iic 1 ∪ Set.Ici 3 := by
  sorry

end range_of_a_l4076_407608


namespace absolute_value_and_opposite_l4076_407679

theorem absolute_value_and_opposite :
  (|-2/5| = 2/5) ∧ (-(2023 : ℤ) = -2023) := by
  sorry

end absolute_value_and_opposite_l4076_407679


namespace monic_cubic_polynomial_unique_l4076_407681

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_unique
  (q : ℝ → ℝ)
  (h_monic : ∃ a b c : ℝ, q = MonicCubicPolynomial a b c)
  (h_root : q (2 - 3*I) = 0)
  (h_value : q 1 = 26) :
  q = MonicCubicPolynomial (-2.4) 6.6 20.8 := by sorry

end monic_cubic_polynomial_unique_l4076_407681


namespace students_per_bus_l4076_407655

theorem students_per_bus 
  (total_students : ℕ) 
  (num_buses : ℕ) 
  (students_in_cars : ℕ) 
  (h1 : total_students = 396) 
  (h2 : num_buses = 7) 
  (h3 : students_in_cars = 4) 
  (h4 : num_buses > 0) : 
  (total_students - students_in_cars) / num_buses = 56 := by
sorry

end students_per_bus_l4076_407655


namespace notched_circle_distance_l4076_407678

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 72}

def B : ℝ × ℝ := (1, -4)
def A : ℝ × ℝ := (1, 4)
def C : ℝ × ℝ := (7, -4)

-- State the theorem
theorem notched_circle_distance :
  B ∈ Circle ∧
  A ∈ Circle ∧
  C ∈ Circle ∧
  A.1 = B.1 ∧
  A.2 - B.2 = 8 ∧
  C.1 - B.1 = 6 ∧
  C.2 = B.2 ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →
  B.1^2 + B.2^2 = 17 := by
  sorry

end notched_circle_distance_l4076_407678


namespace calen_excess_pencils_l4076_407639

/-- The number of pencils each person has -/
structure PencilCount where
  calen : ℕ
  caleb : ℕ
  candy : ℕ

/-- The conditions of the problem -/
def pencil_problem (p : PencilCount) : Prop :=
  p.calen > p.caleb ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.candy = 9 ∧
  p.calen - 10 = 10

/-- The theorem to prove -/
theorem calen_excess_pencils (p : PencilCount) :
  pencil_problem p → p.calen - p.caleb = 5 := by
  sorry

end calen_excess_pencils_l4076_407639


namespace rational_roots_condition_l4076_407614

theorem rational_roots_condition (p : ℤ) : 
  (∃ x : ℚ, 4 * x^4 + 4 * p * x^3 = (p - 4) * x^2 - 4 * p * x + p) ↔ (p = 0 ∨ p = -1) :=
by sorry

end rational_roots_condition_l4076_407614
