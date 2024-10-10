import Mathlib

namespace general_term_is_2n_l2549_254993

/-- An increasing arithmetic sequence with specific properties -/
def IncreasingArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∃ d > 0, ∀ n, a (n + 1) = a n + d) ∧
  (a 1 = 2) ∧
  (a 2 ^ 2 = a 5 + 6)

/-- The general term of the sequence is 2n -/
theorem general_term_is_2n (a : ℕ → ℝ) 
    (h : IncreasingArithmeticSequence a) : 
    ∀ n : ℕ, a n = 2 * n := by
  sorry

end general_term_is_2n_l2549_254993


namespace smallest_discount_value_l2549_254944

theorem smallest_discount_value : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → 
    (1 - (m : ℝ) / 100 ≥ (1 - 0.20)^2 ∨ 
     1 - (m : ℝ) / 100 ≥ (1 - 0.15)^3 ∨ 
     1 - (m : ℝ) / 100 ≥ (1 - 0.30) * (1 - 0.10))) ∧ 
  (1 - (n : ℝ) / 100 < (1 - 0.20)^2 ∧ 
   1 - (n : ℝ) / 100 < (1 - 0.15)^3 ∧ 
   1 - (n : ℝ) / 100 < (1 - 0.30) * (1 - 0.10)) ∧ 
  n = 39 := by
  sorry

end smallest_discount_value_l2549_254944


namespace train_length_l2549_254962

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 * (5/18) → time = 40 → speed * time = 400 := by sorry

end train_length_l2549_254962


namespace doug_initial_marbles_l2549_254905

theorem doug_initial_marbles (ed_marbles : ℕ) (ed_more_than_doug : ℕ) (doug_lost : ℕ)
  (h1 : ed_marbles = 27)
  (h2 : ed_more_than_doug = 5)
  (h3 : doug_lost = 3) :
  ed_marbles - ed_more_than_doug + doug_lost = 25 := by
  sorry

end doug_initial_marbles_l2549_254905


namespace function_properties_imply_specific_form_and_result_l2549_254900

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties_imply_specific_form_and_result 
  (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  (∀ x : ℝ, f ω φ (x + Real.pi / 2) = f ω φ (Real.pi / 2 - x)) →
  (∀ x : ℝ, ∃ k : ℤ, f ω φ (x + Real.pi / (2 * ω)) = f ω φ (x + k * Real.pi / ω)) →
  (∃ α : ℝ, 0 < α ∧ α < Real.pi / 2 ∧ f ω φ (α / 2 + Real.pi / 12) = 3 / 5) →
  (∀ x : ℝ, f ω φ x = Real.cos (2 * x)) ∧
  (∀ α : ℝ, 0 < α → α < Real.pi / 2 → f ω φ (α / 2 + Real.pi / 12) = 3 / 5 → 
    Real.sin (2 * α) = (24 + 7 * Real.sqrt 3) / 50) := by
  sorry

end function_properties_imply_specific_form_and_result_l2549_254900


namespace range_of_m_l2549_254919

/-- The function f(x) = x^2 - 3x + 4 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (7/4) 4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 7/4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 4) →
  m ∈ Set.Icc (3/2) 3 :=
by sorry

end range_of_m_l2549_254919


namespace child_support_owed_amount_l2549_254956

/-- Calculates the amount owed in child support given the specified conditions --/
def child_support_owed (
  support_rate : ℝ)
  (initial_salary : ℝ)
  (initial_years : ℕ)
  (raise_percentage : ℝ)
  (raise_years : ℕ)
  (amount_paid : ℝ) : ℝ :=
  let total_initial_income := initial_salary * initial_years
  let raised_salary := initial_salary * (1 + raise_percentage)
  let total_raised_income := raised_salary * raise_years
  let total_income := total_initial_income + total_raised_income
  let total_support_due := total_income * support_rate
  total_support_due - amount_paid

/-- Theorem stating that the amount owed in child support is $69,000 --/
theorem child_support_owed_amount : 
  child_support_owed 0.3 30000 3 0.2 4 1200 = 69000 := by
  sorry

end child_support_owed_amount_l2549_254956


namespace pradeep_exam_marks_l2549_254952

/-- The maximum marks in Pradeep's exam -/
def maximum_marks : ℕ := 928

/-- The percentage required to pass the exam -/
def pass_percentage : ℚ := 55 / 100

/-- The marks Pradeep obtained -/
def pradeep_marks : ℕ := 400

/-- The number of marks Pradeep fell short by -/
def shortfall : ℕ := 110

theorem pradeep_exam_marks :
  (pass_percentage * maximum_marks : ℚ) = pradeep_marks + shortfall ∧
  maximum_marks * pass_percentage = (pradeep_marks + shortfall : ℚ) ∧
  ∀ m : ℕ, m > maximum_marks → 
    (pass_percentage * m : ℚ) > (pradeep_marks + shortfall : ℚ) :=
by sorry

end pradeep_exam_marks_l2549_254952


namespace jose_peanuts_l2549_254970

theorem jose_peanuts (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : kenya_peanuts = 133)
  (h2 : kenya_peanuts = jose_peanuts + 48) : 
  jose_peanuts = 85 := by
  sorry

end jose_peanuts_l2549_254970


namespace parabola_properties_line_parabola_intersection_l2549_254948

/-- Parabola C: y^2 = -4x -/
def parabola (x y : ℝ) : Prop := y^2 = -4*x

/-- Line l: y = kx - k + 2, passing through (1, 2) -/
def line (k x y : ℝ) : Prop := y = k*x - k + 2

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (-1, 0)

/-- Directrix of the parabola -/
def directrix (x : ℝ) : Prop := x = 1

/-- Distance from focus to directrix -/
def focus_directrix_distance : ℝ := 2

/-- Theorem about the parabola and its properties -/
theorem parabola_properties :
  (∀ x y, parabola x y → (focus.1 = -1 ∧ focus.2 = 0)) ∧
  (∀ x, directrix x ↔ x = 1) ∧
  focus_directrix_distance = 2 :=
sorry

/-- Theorem about the intersection of the line and parabola -/
theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, parabola x y ∧ line k x y →
    (k = 0 ∨ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) ↔
      (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2)) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) ↔
    (∃ p q : ℝ × ℝ, p ≠ q ∧ parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ line k p.1 p.2 ∧ line k q.1 q.2)) ∧
  (k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) ↔
    (∀ x y, ¬(parabola x y ∧ line k x y)) :=
sorry

end parabola_properties_line_parabola_intersection_l2549_254948


namespace factorization_equality_l2549_254977

theorem factorization_equality (a y : ℝ) : a^2 * y - 4 * y = y * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l2549_254977


namespace dirk_amulet_selling_days_l2549_254961

/-- Represents the problem of calculating the number of days Dirk sold amulets. -/
def amulet_problem (amulets_per_day : ℕ) (selling_price : ℚ) (cost_price : ℚ) 
  (faire_cut_percentage : ℚ) (total_profit : ℚ) : Prop :=
  let revenue_per_amulet : ℚ := selling_price
  let profit_per_amulet : ℚ := selling_price - cost_price
  let faire_cut_per_amulet : ℚ := faire_cut_percentage * revenue_per_amulet
  let net_profit_per_amulet : ℚ := profit_per_amulet - faire_cut_per_amulet
  let net_profit_per_day : ℚ := net_profit_per_amulet * amulets_per_day
  let days : ℚ := total_profit / net_profit_per_day
  days = 2

/-- Theorem stating the solution to Dirk's amulet selling problem. -/
theorem dirk_amulet_selling_days : 
  amulet_problem 25 40 30 (1/10) 300 := by
  sorry

end dirk_amulet_selling_days_l2549_254961


namespace line_intercepts_sum_l2549_254935

/-- The sum of the intercepts of the line 2x - 3y + 6 = 0 on the coordinate axes is -1 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (2 * x - 3 * y + 6 = 0) → 
  (∃ x_intercept y_intercept : ℝ, 
    (2 * x_intercept + 6 = 0) ∧ 
    (-3 * y_intercept + 6 = 0) ∧ 
    (x_intercept + y_intercept = -1)) := by
  sorry

end line_intercepts_sum_l2549_254935


namespace ladder_length_l2549_254927

theorem ladder_length (initial_distance : ℝ) (pull_distance : ℝ) (slide_distance : ℝ) :
  initial_distance = 15 →
  pull_distance = 9 →
  slide_distance = 13 →
  ∃ (ladder_length : ℝ) (initial_height : ℝ),
    ladder_length ^ 2 = initial_distance ^ 2 + initial_height ^ 2 ∧
    ladder_length ^ 2 = (initial_distance + pull_distance) ^ 2 + (initial_height - slide_distance) ^ 2 ∧
    ladder_length = 25 := by
sorry

end ladder_length_l2549_254927


namespace cyclic_sum_inequality_l2549_254903

theorem cyclic_sum_inequality (k : ℕ) (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x^(k+2) / (x^(k+1) + y^k + z^k)) + 
  (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
  (z^(k+2) / (z^(k+1) + x^k + y^k)) ≥ 1/7 ∧
  ((x^(k+2) / (x^(k+1) + y^k + z^k)) + 
   (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
   (z^(k+2) / (z^(k+1) + x^k + y^k)) = 1/7 ↔ 
   x = 1/3 ∧ y = 1/3 ∧ z = 1/3) := by
sorry

end cyclic_sum_inequality_l2549_254903


namespace savings_calculation_l2549_254950

theorem savings_calculation (total : ℚ) (furniture_fraction : ℚ) (tv_cost : ℚ) : 
  furniture_fraction = 3 / 4 →
  tv_cost = 300 →
  (1 - furniture_fraction) * total = tv_cost →
  total = 1200 := by
sorry

end savings_calculation_l2549_254950


namespace average_speed_calculation_l2549_254994

theorem average_speed_calculation (d₁ d₂ d₃ v₁ v₂ v₃ : ℝ) 
  (h₁ : d₁ = 30) (h₂ : d₂ = 50) (h₃ : d₃ = 40)
  (h₄ : v₁ = 30) (h₅ : v₂ = 50) (h₆ : v₃ = 60) : 
  (d₁ + d₂ + d₃) / ((d₁ / v₁) + (d₂ / v₂) + (d₃ / v₃)) = 45 := by
  sorry

end average_speed_calculation_l2549_254994


namespace gcd_765432_654321_l2549_254937

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l2549_254937


namespace sum_of_selected_flowerbeds_l2549_254930

/-- The number of seeds in each flowerbed -/
def seeds : Fin 9 → ℕ
  | 0 => 18  -- 1st flowerbed
  | 1 => 22  -- 2nd flowerbed
  | 2 => 30  -- 3rd flowerbed
  | 3 => 2 * seeds 0  -- 4th flowerbed
  | 4 => seeds 2  -- 5th flowerbed
  | 5 => seeds 1 / 2  -- 6th flowerbed
  | 6 => seeds 0  -- 7th flowerbed
  | 7 => seeds 3  -- 8th flowerbed
  | 8 => seeds 2 - 1  -- 9th flowerbed

theorem sum_of_selected_flowerbeds : seeds 0 + seeds 4 + seeds 8 = 77 := by
  sorry

end sum_of_selected_flowerbeds_l2549_254930


namespace triangle_area_l2549_254991

/-- Triangle XYZ with given properties has area 35√7/2 -/
theorem triangle_area (X Y Z : Real) (r R : Real) (h1 : r = 3) (h2 : R = 12) 
  (h3 : 3 * Real.cos Y = Real.cos X + Real.cos Z) : 
  ∃ (area : Real), area = (35 * Real.sqrt 7) / 2 ∧ 
  area = r * (Real.sin X * R + Real.sin Y * R + Real.sin Z * R) / 2 := by
  sorry

end triangle_area_l2549_254991


namespace binary_1011_is_11_l2549_254960

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

theorem binary_1011_is_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end binary_1011_is_11_l2549_254960


namespace solve_for_b_l2549_254973

theorem solve_for_b (b : ℚ) (h : b + b/4 = 10/4) : b = 2 := by
  sorry

end solve_for_b_l2549_254973


namespace composite_blackboard_theorem_l2549_254906

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ 1 < d ∧ d < n

def blackboard_numbers (n : ℕ) : Set ℕ :=
  {x | ∃ d, proper_divisor d n ∧ x = d + 1}

theorem composite_blackboard_theorem (n : ℕ) :
  is_composite n →
  (∃ m, blackboard_numbers n = {x | proper_divisor x m}) ↔
  n = 4 ∨ n = 8 := by
  sorry

end composite_blackboard_theorem_l2549_254906


namespace proposition_logic_l2549_254986

theorem proposition_logic (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by sorry

end proposition_logic_l2549_254986


namespace triangle_inequality_l2549_254992

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * (-a + b + c) + b^2 * (a - b + c) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l2549_254992


namespace green_light_probability_theorem_l2549_254990

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of arriving during the green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (d.red + d.yellow + d.green)

/-- Theorem stating the probability of arriving during the green light
    for the given traffic light durations -/
theorem green_light_probability_theorem (d : TrafficLightDuration) 
  (h1 : d.red = 30)
  (h2 : d.yellow = 5)
  (h3 : d.green = 40) :
  greenLightProbability d = 8/15 := by
  sorry

end green_light_probability_theorem_l2549_254990


namespace chessboard_divisibility_theorem_l2549_254915

/-- Represents a chessboard with natural numbers -/
def Chessboard := Matrix (Fin 8) (Fin 8) ℕ

/-- Represents an operation on the chessboard -/
inductive Operation
  | inc_3x3 (i j : Fin 6) : Operation
  | inc_4x4 (i j : Fin 5) : Operation

/-- Applies an operation to a chessboard -/
def apply_operation (board : Chessboard) (op : Operation) : Chessboard :=
  match op with
  | Operation.inc_3x3 i j => sorry
  | Operation.inc_4x4 i j => sorry

/-- Checks if all elements in the chessboard are divisible by 10 -/
def all_divisible_by_10 (board : Chessboard) : Prop :=
  ∀ i j, board i j % 10 = 0

/-- Main theorem: There exists a sequence of operations that makes all numbers divisible by 10 -/
theorem chessboard_divisibility_theorem (initial_board : Chessboard) :
  ∃ (ops : List Operation), all_divisible_by_10 (ops.foldl apply_operation initial_board) :=
sorry

end chessboard_divisibility_theorem_l2549_254915


namespace exp_greater_or_equal_e_l2549_254913

theorem exp_greater_or_equal_e : ∀ x : ℝ, Real.exp x ≥ Real.exp 1 := by sorry

end exp_greater_or_equal_e_l2549_254913


namespace staircase_extension_l2549_254989

/-- Calculates the number of additional toothpicks needed to extend a staircase -/
def additional_toothpicks (initial_steps : ℕ) (final_steps : ℕ) (initial_toothpicks : ℕ) : ℕ :=
  let base_increase := initial_toothpicks / initial_steps + 2
  let num_new_steps := final_steps - initial_steps
  (num_new_steps * (2 * base_increase + (num_new_steps - 1) * 2)) / 2

theorem staircase_extension :
  additional_toothpicks 4 7 28 = 42 :=
by sorry

end staircase_extension_l2549_254989


namespace set_union_problem_l2549_254976

theorem set_union_problem (A B : Set ℕ) (m : ℕ) :
  A = {1, 2, 3, 4} →
  B = {m, 4, 7, 8} →
  A ∩ B = {1, 4} →
  A ∪ B = {1, 2, 3, 4, 7, 8} := by
sorry

end set_union_problem_l2549_254976


namespace factor_expression_l2549_254929

theorem factor_expression (c : ℝ) : 189 * c^2 + 27 * c - 36 = 9 * (3 * c - 1) * (7 * c + 4) := by
  sorry

end factor_expression_l2549_254929


namespace max_value_product_max_value_achieved_l2549_254951

theorem max_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_two : a + b + c = 2) : 
  a^2 * b^3 * c^2 ≤ 128/2187 := by
sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ 
  a^2 * b^3 * c^2 > 128/2187 - ε := by
sorry

end max_value_product_max_value_achieved_l2549_254951


namespace count_perfect_square_factors_7560_l2549_254914

def prime_factorization (n : Nat) : List (Nat × Nat) :=
  [(2, 3), (3, 3), (5, 1), (7, 1)]

def is_perfect_square (factor : List (Nat × Nat)) : Bool :=
  factor.all (fun (p, e) => e % 2 = 0)

def count_perfect_square_factors (n : Nat) : Nat :=
  let factors := List.filter is_perfect_square 
    (List.map (fun l => List.map (fun (p, e) => (p, Nat.min e l.2)) (prime_factorization n)) 
      [(2, 0), (2, 2), (3, 0), (3, 2), (5, 0), (7, 0)])
  factors.length

theorem count_perfect_square_factors_7560 :
  count_perfect_square_factors 7560 = 4 := by
  sorry

end count_perfect_square_factors_7560_l2549_254914


namespace divisibility_equivalence_l2549_254971

theorem divisibility_equivalence (x y : ℤ) :
  (2 * x + 3 * y) % 7 = 0 ↔ (5 * x + 4 * y) % 7 = 0 := by
  sorry

end divisibility_equivalence_l2549_254971


namespace arithmetic_sequence_first_term_l2549_254985

/-- An arithmetic sequence satisfying a specific condition -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ d : ℝ, ∀ k : ℕ, a (k + 1) = a k + d

/-- The specific condition for the sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) + a n = 4 * n

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : SequenceCondition a) :
  a 1 = 1 := by
  sorry

end arithmetic_sequence_first_term_l2549_254985


namespace rabbit_carrots_l2549_254995

theorem rabbit_carrots : ∀ (rabbit_holes fox_holes : ℕ),
  rabbit_holes = fox_holes + 2 →
  5 * rabbit_holes = 6 * fox_holes →
  5 * rabbit_holes = 60 :=
by
  sorry

end rabbit_carrots_l2549_254995


namespace chocolate_division_l2549_254909

/-- The amount of chocolate Shaina receives when Jordan divides his chocolate -/
theorem chocolate_division (total : ℚ) (keep_fraction : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) : 
  total = 60 / 7 →
  keep_fraction = 1 / 3 →
  num_piles = 5 →
  piles_to_shaina = 2 →
  (1 - keep_fraction) * total * (piles_to_shaina / num_piles) = 16 / 7 := by
  sorry

end chocolate_division_l2549_254909


namespace max_area_rectangle_l2549_254966

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of the rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of the rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

/-- Predicate to check if a number is even. -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Theorem stating the maximum area of a rectangle with given constraints. -/
theorem max_area_rectangle :
  ∀ r : Rectangle,
    r.perimeter = 40 →
    isEven r.length →
    r.area ≤ 100 ∧
    (r.area = 100 ↔ r.length = 10 ∧ r.width = 10) := by
  sorry

#check max_area_rectangle

end max_area_rectangle_l2549_254966


namespace hyperbola_asymptote_l2549_254954

/-- Theorem: For a hyperbola x^2 - y^2/a^2 = 1 with a > 0, if its asymptotes are y = ± 2x, then a = 2 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 - y^2/a^2 = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 := by
  sorry

end hyperbola_asymptote_l2549_254954


namespace calculate_expression_l2549_254912

theorem calculate_expression : 2^2 + |-3| - Real.sqrt 25 = 2 := by
  sorry

end calculate_expression_l2549_254912


namespace smallest_number_l2549_254940

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -1) (hc : c = 1) (hd : d = -5) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ d :=
by sorry

end smallest_number_l2549_254940


namespace difference_of_squares_l2549_254998

theorem difference_of_squares (a b : ℝ) : (a + 2*b) * (a - 2*b) = a^2 - 4*b^2 := by
  sorry

end difference_of_squares_l2549_254998


namespace range_of_m_l2549_254972

/-- Given the equation (m+3)/(x-1) = 1 where x is a positive number, 
    prove that the range of m is m > -4 and m ≠ -3 -/
theorem range_of_m (m : ℝ) (x : ℝ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) :
  m > -4 ∧ m ≠ -3 := by
  sorry

end range_of_m_l2549_254972


namespace florist_roses_count_l2549_254939

/-- Calculates the total number of roses after picking two batches -/
def total_roses (initial : Float) (batch1 : Float) (batch2 : Float) : Float :=
  initial + batch1 + batch2

/-- Theorem stating that given the specific numbers from the problem, 
    the total number of roses is 72.0 -/
theorem florist_roses_count : 
  total_roses 37.0 16.0 19.0 = 72.0 := by
  sorry

end florist_roses_count_l2549_254939


namespace min_value_alpha_gamma_l2549_254941

open Complex

theorem min_value_alpha_gamma (f : ℂ → ℂ) (α γ : ℂ) :
  (∀ z, f z = (4 + I) * z^2 + α * z + γ) →
  (f 1).im = 0 →
  (f I).im = 0 →
  ∃ (α₀ γ₀ : ℂ), abs α₀ + abs γ₀ = Real.sqrt 2 ∧ 
    ∀ (α' γ' : ℂ), (∀ z, f z = (4 + I) * z^2 + α' * z + γ') →
      (f 1).im = 0 → (f I).im = 0 → abs α' + abs γ' ≥ Real.sqrt 2 :=
by sorry

end min_value_alpha_gamma_l2549_254941


namespace polynomial_real_root_l2549_254933

theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, x^3 + b*x^2 - x + b = 0) ↔ b ≤ 0 := by sorry

end polynomial_real_root_l2549_254933


namespace seth_sold_78_candy_bars_l2549_254904

def max_candy_bars : ℕ := 24

def seth_candy_bars : ℕ := 3 * max_candy_bars + 6

theorem seth_sold_78_candy_bars : seth_candy_bars = 78 := by
  sorry

end seth_sold_78_candy_bars_l2549_254904


namespace solution_difference_l2549_254949

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 + 3*r - 18) = r + 3) →
  ((6 * s - 18) / (s^2 + 3*s - 18) = s + 3) →
  r ≠ s →
  r > s →
  r - s = 1 :=
by
  sorry

end solution_difference_l2549_254949


namespace tangent_line_problem_l2549_254984

theorem tangent_line_problem (a : ℝ) : 
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → (x = 1 ∨ (y = m * (x - 1) ∧ 3 * x^2 = m)))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → (x = 1 ∨ (y = m * (x - 1) ∧ 2 * a * x + 15/4 = m)))))
  → a = -25/64 ∨ a = -1 := by
  sorry

#check tangent_line_problem

end tangent_line_problem_l2549_254984


namespace molecular_weight_proof_l2549_254965

/-- The molecular weight of C7H6O2 -/
def molecular_weight_C7H6O2 : ℝ := 122

/-- The number of moles in the given sample -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given sample -/
def total_weight : ℝ := 1098

/-- Theorem: The molecular weight of one mole of C7H6O2 is 122 g/mol -/
theorem molecular_weight_proof :
  molecular_weight_C7H6O2 = total_weight / num_moles :=
by sorry

end molecular_weight_proof_l2549_254965


namespace comic_collection_overtake_l2549_254911

/-- The number of months after which LaShawn's collection becomes 1.5 times Kymbrea's --/
def months_to_overtake : ℕ := 70

/-- Kymbrea's initial number of comic books --/
def kymbrea_initial : ℕ := 40

/-- LaShawn's initial number of comic books --/
def lashawn_initial : ℕ := 25

/-- Kymbrea's monthly collection rate --/
def kymbrea_rate : ℕ := 3

/-- LaShawn's monthly collection rate --/
def lashawn_rate : ℕ := 5

/-- Theorem stating that after the specified number of months, 
    LaShawn's collection is 1.5 times Kymbrea's --/
theorem comic_collection_overtake :
  (lashawn_initial + lashawn_rate * months_to_overtake : ℚ) = 
  1.5 * (kymbrea_initial + kymbrea_rate * months_to_overtake) :=
by sorry

end comic_collection_overtake_l2549_254911


namespace optimal_selection_uses_golden_ratio_l2549_254969

/-- The optimal selection method popularized by Hua Luogeng --/
def OptimalSelectionMethod : Type := Unit

/-- The concept used in the optimal selection method --/
def ConceptUsed : Type := Unit

/-- The golden ratio --/
def GoldenRatio : Type := Unit

/-- The optimal selection method was popularized by Hua Luogeng --/
axiom hua_luogeng_popularized : OptimalSelectionMethod

/-- The concept used in the optimal selection method is the golden ratio --/
theorem optimal_selection_uses_golden_ratio : 
  ConceptUsed = GoldenRatio := by sorry

end optimal_selection_uses_golden_ratio_l2549_254969


namespace solve_equation_l2549_254945

theorem solve_equation (x : ℝ) (h : 9 - (4/x) = 7 + (8/x)) : x = 6 := by
  sorry

end solve_equation_l2549_254945


namespace smallest_x_divisible_l2549_254916

theorem smallest_x_divisible : ∃ (x : ℤ), x = 36629 ∧ 
  (∀ (y : ℤ), y < x → ¬(33 ∣ (2 * y + 2) ∧ 44 ∣ (2 * y + 2) ∧ 55 ∣ (2 * y + 2) ∧ 666 ∣ (2 * y + 2))) ∧
  (33 ∣ (2 * x + 2) ∧ 44 ∣ (2 * x + 2) ∧ 55 ∣ (2 * x + 2) ∧ 666 ∣ (2 * x + 2)) :=
by sorry

end smallest_x_divisible_l2549_254916


namespace tank_capacity_l2549_254987

/-- The capacity of a tank with specific inlet and outlet pipe properties -/
theorem tank_capacity
  (outlet_time : ℝ)
  (inlet_rate : ℝ)
  (both_pipes_time : ℝ)
  (h1 : outlet_time = 5)
  (h2 : inlet_rate = 8)
  (h3 : both_pipes_time = 8) :
  ∃ (capacity : ℝ), capacity = 1280 ∧
    capacity / outlet_time - (inlet_rate * 60) = capacity / both_pipes_time :=
by sorry

end tank_capacity_l2549_254987


namespace pentagon_perimeter_l2549_254938

/-- Given a pentagon ABCDE where:
  - ΔABE, ΔBCE, and ΔCDE are right-angled triangles
  - ∠AEB = 45°
  - ∠BEC = 60°
  - ∠CED = 45°
  - AE = 40
Prove that the perimeter of pentagon ABCDE is 140 + (40√3)/3 -/
theorem pentagon_perimeter (A B C D E : ℝ × ℝ) : 
  let angle (p q r : ℝ × ℝ) := Real.arccos ((p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2)) / 
    (((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt * ((r.1 - q.1)^2 + (r.2 - q.2)^2).sqrt)
  let dist (p q : ℝ × ℝ) := ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  let perimeter := dist A B + dist B C + dist C D + dist D E + dist E A
  angle A E B = π/4 ∧ 
  angle B E C = π/3 ∧ 
  angle C E D = π/4 ∧
  angle B A E = π/2 ∧
  angle C B E = π/2 ∧
  angle D C E = π/2 ∧
  dist A E = 40 →
  perimeter = 140 + 40 * Real.sqrt 3 / 3 := by
sorry


end pentagon_perimeter_l2549_254938


namespace tan_inequality_solution_set_l2549_254934

theorem tan_inequality_solution_set : 
  let S := {x : ℝ | ∃ k : ℤ, k * π - π / 3 < x ∧ x < k * π + Real.arctan 2}
  ∀ x : ℝ, x ∈ S ↔ -Real.sqrt 3 < Real.tan x ∧ Real.tan x < 2 :=
by sorry

end tan_inequality_solution_set_l2549_254934


namespace is_circle_center_l2549_254958

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 55 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : ∀ (x y : ℝ), 
  circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 65 :=
by sorry

end is_circle_center_l2549_254958


namespace baseball_team_points_l2549_254907

/- Define the structure of the team -/
structure BaseballTeam where
  totalPlayers : Nat
  totalPoints : Nat
  startingPlayers : Nat
  reservePlayers : Nat
  rookiePlayers : Nat
  totalGames : Nat

/- Define the theorem -/
theorem baseball_team_points 
  (team : BaseballTeam)
  (h1 : team.totalPlayers = 15)
  (h2 : team.totalPoints = 900)
  (h3 : team.startingPlayers = 7)
  (h4 : team.reservePlayers = 3)
  (h5 : team.rookiePlayers = 5)
  (h6 : team.totalGames = 20) :
  ∃ (startingAvg reserveAvg rookieAvg : ℕ),
    (startingAvg * team.startingPlayers * team.totalGames +
     reserveAvg * team.reservePlayers * 15 +
     rookieAvg * team.rookiePlayers * ((20 + 10 + 10 + 5 + 5) / 5) + 
     (10 + 15 + 15)) = team.totalPoints := by
  sorry


end baseball_team_points_l2549_254907


namespace surrounding_circles_radius_l2549_254978

/-- The radius of the central circle -/
def central_radius : ℝ := 2

/-- The number of surrounding circles -/
def num_surrounding_circles : ℕ := 4

/-- Predicate that checks if all circles are touching each other -/
def circles_touching (r : ℝ) : Prop :=
  ∃ (centers : Fin num_surrounding_circles → ℝ × ℝ),
    ∀ (i j : Fin num_surrounding_circles),
      i ≠ j → ‖centers i - centers j‖ = 2 * r ∧
    ∀ (i : Fin num_surrounding_circles),
      ‖centers i‖ = central_radius + r

/-- Theorem stating that the radius of surrounding circles is 2 -/
theorem surrounding_circles_radius :
  ∃ (r : ℝ), r > 0 ∧ circles_touching r → r = 2 :=
sorry

end surrounding_circles_radius_l2549_254978


namespace compute_fraction_power_l2549_254928

theorem compute_fraction_power : 9 * (1/7)^4 = 9/2401 := by
  sorry

end compute_fraction_power_l2549_254928


namespace negation_of_existence_negation_of_proposition_l2549_254926

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > Real.sin x) ↔ (∀ x : ℝ, x ≤ Real.sin x) := by sorry

end negation_of_existence_negation_of_proposition_l2549_254926


namespace point_coordinates_l2549_254943

def is_valid_point (x y : ℝ) : Prop :=
  |y| = 1 ∧ |x| = 2

theorem point_coordinates :
  ∀ x y : ℝ, is_valid_point x y ↔ (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by sorry

end point_coordinates_l2549_254943


namespace impossible_tiling_l2549_254982

/-- Represents a T-tetromino placement on a checkerboard -/
structure TTetromino where
  blackMajor : ℕ  -- number of T-tetrominoes with 3 black squares
  whiteMajor : ℕ  -- number of T-tetrominoes with 3 white squares

/-- The size of the grid -/
def gridSize : ℕ := 10

/-- The total number of squares in the grid -/
def totalSquares : ℕ := gridSize * gridSize

/-- Represents the coloring constraint of the checkerboard pattern -/
def colorConstraint (t : TTetromino) : Prop :=
  3 * t.blackMajor + t.whiteMajor = totalSquares / 2 ∧
  t.blackMajor + 3 * t.whiteMajor = totalSquares / 2

/-- Theorem stating the impossibility of tiling the grid -/
theorem impossible_tiling : ¬ ∃ t : TTetromino, colorConstraint t := by
  sorry

end impossible_tiling_l2549_254982


namespace total_commission_is_4200_l2549_254918

def coupe_price : ℝ := 30000
def suv_price : ℝ := 2 * coupe_price
def luxury_sedan_price : ℝ := 80000
def commission_rate_coupe_suv : ℝ := 0.02
def commission_rate_luxury : ℝ := 0.03

def total_commission : ℝ :=
  coupe_price * commission_rate_coupe_suv +
  suv_price * commission_rate_coupe_suv +
  luxury_sedan_price * commission_rate_luxury

theorem total_commission_is_4200 :
  total_commission = 4200 := by sorry

end total_commission_is_4200_l2549_254918


namespace project_hours_theorem_l2549_254957

/-- Represents the hours worked by three people on a project -/
structure ProjectHours where
  least : ℕ
  middle : ℕ
  most : ℕ

/-- The total hours worked on the project -/
def total_hours (h : ProjectHours) : ℕ := h.least + h.middle + h.most

/-- The condition that the working times are in the ratio 1:2:3 -/
def ratio_condition (h : ProjectHours) : Prop :=
  h.middle = 2 * h.least ∧ h.most = 3 * h.least

/-- The condition that the hardest working person worked 40 hours more than the person who worked the least -/
def difference_condition (h : ProjectHours) : Prop :=
  h.most = h.least + 40

theorem project_hours_theorem (h : ProjectHours) 
  (hc1 : ratio_condition h) 
  (hc2 : difference_condition h) : 
  total_hours h = 120 := by
  sorry

#check project_hours_theorem

end project_hours_theorem_l2549_254957


namespace boat_upstream_downstream_distance_l2549_254931

/-- Proves that a boat with a given speed in still water, traveling a certain distance upstream in one hour, will travel a specific distance downstream in one hour. -/
theorem boat_upstream_downstream_distance 
  (v : ℝ) -- Speed of the boat in still water (km/h)
  (d_upstream : ℝ) -- Distance traveled upstream in one hour (km)
  (h1 : v = 8) -- The boat's speed in still water is 8 km/h
  (h2 : d_upstream = 5) -- The boat travels 5 km upstream in one hour
  : ∃ d_downstream : ℝ, d_downstream = 11 ∧ d_downstream = v + (v - d_upstream) := by
  sorry

end boat_upstream_downstream_distance_l2549_254931


namespace chocolate_milk_amount_l2549_254925

/-- Represents the ingredients for making chocolate milk -/
structure Ingredients where
  milk : ℕ
  chocolate_syrup : ℕ
  whipped_cream : ℕ

/-- Represents the recipe for one glass of chocolate milk -/
structure Recipe where
  milk : ℕ
  chocolate_syrup : ℕ
  whipped_cream : ℕ
  total : ℕ

/-- Calculates the number of full glasses that can be made with given ingredients and recipe -/
def fullGlasses (i : Ingredients) (r : Recipe) : ℕ :=
  min (i.milk / r.milk) (min (i.chocolate_syrup / r.chocolate_syrup) (i.whipped_cream / r.whipped_cream))

/-- Theorem: Charles will drink 96 ounces of chocolate milk -/
theorem chocolate_milk_amount (i : Ingredients) (r : Recipe) :
  i.milk = 130 ∧ i.chocolate_syrup = 60 ∧ i.whipped_cream = 25 ∧
  r.milk = 4 ∧ r.chocolate_syrup = 2 ∧ r.whipped_cream = 2 ∧ r.total = 8 →
  fullGlasses i r * r.total = 96 := by
  sorry

end chocolate_milk_amount_l2549_254925


namespace gain_percent_for_50_and_28_l2549_254923

/-- Calculates the gain percent given the number of articles at cost price and selling price that are equal in total value -/
def gainPercent (costArticles : ℕ) (sellArticles : ℕ) : ℚ :=
  let gain := (costArticles : ℚ) / sellArticles - 1
  gain * 100

/-- Proves that when 50 articles at cost price equal 28 articles at selling price, the gain percent is (11/14) * 100 -/
theorem gain_percent_for_50_and_28 :
  gainPercent 50 28 = 11 / 14 * 100 := by
  sorry

#eval gainPercent 50 28

end gain_percent_for_50_and_28_l2549_254923


namespace wall_bricks_l2549_254936

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 360

/-- Represents Brenda's time to build the wall alone (in hours) -/
def brenda_time : ℕ := 8

/-- Represents Brandon's time to build the wall alone (in hours) -/
def brandon_time : ℕ := 12

/-- Represents the decrease in combined output (in bricks per hour) -/
def output_decrease : ℕ := 15

/-- Represents the time taken to build the wall together (in hours) -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 360 -/
theorem wall_bricks : 
  (combined_time : ℚ) * ((total_bricks / brenda_time + total_bricks / brandon_time) - output_decrease) = total_bricks := by
  sorry

#check wall_bricks

end wall_bricks_l2549_254936


namespace decreasing_f_implies_a_bound_l2549_254963

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 1

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

-- Theorem statement
theorem decreasing_f_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2/3) (-1/3), f_deriv a x < 0) →
  a ≥ 7/4 := by
  sorry

end decreasing_f_implies_a_bound_l2549_254963


namespace sum_last_two_digits_fibonacci_factorial_series_l2549_254996

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_last_two_digits_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum = 5 := by
  sorry

end sum_last_two_digits_fibonacci_factorial_series_l2549_254996


namespace inscribed_rectangle_area_bounds_l2549_254964

/-- A rectangle inscribed in a unit square -/
structure InscribedRectangle where
  width : ℝ
  height : ℝ
  width_positive : 0 < width
  height_positive : 0 < height
  fits_in_square : width ≤ 1 ∧ height ≤ 1

/-- The area of an inscribed rectangle -/
def area (r : InscribedRectangle) : ℝ := r.width * r.height

/-- An inscribed rectangle is a square if its width equals its height -/
def is_square (r : InscribedRectangle) : Prop := r.width = r.height

theorem inscribed_rectangle_area_bounds (r : InscribedRectangle) :
  (¬ is_square r → 0 < area r ∧ area r < 1/2) ∧
  (is_square r → area r = 1/2) := by
  sorry

end inscribed_rectangle_area_bounds_l2549_254964


namespace solve_marbles_problem_l2549_254974

def marbles_problem (wolfgang_marbles : ℕ) : Prop :=
  let ludo_marbles := wolfgang_marbles + (wolfgang_marbles / 4)
  let total_wolfgang_ludo := wolfgang_marbles + ludo_marbles
  let michael_marbles := (2 * total_wolfgang_ludo) / 3
  let total_marbles := total_wolfgang_ludo + michael_marbles
  (wolfgang_marbles = 16) →
  (total_marbles / 3 = 20)

theorem solve_marbles_problem :
  marbles_problem 16 := by sorry

end solve_marbles_problem_l2549_254974


namespace direction_vectors_of_line_l2549_254920

/-- Given a line with equation 3x - 4y + 1 = 0, prove that (4, 3) and (1, 3/4) are valid direction vectors. -/
theorem direction_vectors_of_line (x y : ℝ) : 
  (3 * x - 4 * y + 1 = 0) →
  (∃ (k : ℝ), k ≠ 0 ∧ (k * 4, k * 3) = (3, -4)) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ (k * 1, k * (3/4)) = (3, -4)) :=
by sorry

end direction_vectors_of_line_l2549_254920


namespace hyperbola_with_foci_on_y_axis_l2549_254902

theorem hyperbola_with_foci_on_y_axis 
  (m n : ℝ) 
  (h : m * n < 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), m * x^2 - m * y^2 = n ↔ y^2 / a^2 - x^2 / b^2 = 1) ∧
    (∀ (c : ℝ), c > a → ∃ (f₁ f₂ : ℝ), 
      f₁ = 0 ∧ f₂ = 0 ∧ 
      ∀ (x y : ℝ), m * x^2 - m * y^2 = n → 
        (x - f₁)^2 + (y - f₂)^2 - ((x - f₁)^2 + (y + f₂)^2) = 4 * c^2) :=
sorry

end hyperbola_with_foci_on_y_axis_l2549_254902


namespace unique_aabb_perfect_square_l2549_254917

/-- A 4-digit number of the form aabb in base 10 -/
def aabb (a b : ℕ) : ℕ := 1000 * a + 100 * a + 10 * b + b

/-- Predicate for a number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem unique_aabb_perfect_square :
  ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
    (is_perfect_square (aabb a b) ↔ a = 7 ∧ b = 4) :=
sorry

end unique_aabb_perfect_square_l2549_254917


namespace coefficient_of_x_squared_l2549_254901

def original_expression (x : ℝ) : ℝ :=
  2 * (x - 6) + 5 * (10 - 3 * x^2 + 4 * x) - 7 * (3 * x^2 - 2 * x + 1)

theorem coefficient_of_x_squared :
  ∃ (a b c : ℝ), ∀ x, original_expression x = a * x^2 + b * x + c ∧ a = -36 :=
sorry

end coefficient_of_x_squared_l2549_254901


namespace knight_moves_equal_for_7x7_l2549_254999

/-- Represents a position on a chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents a knight's move on a chessboard -/
inductive KnightMove : Position → Position → Prop where
  | move_1 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 1, y + 2⟩
  | move_2 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 2, y + 1⟩
  | move_3 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 2, y - 1⟩
  | move_4 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 1, y - 2⟩
  | move_5 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 1, y - 2⟩
  | move_6 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 2, y - 1⟩
  | move_7 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 2, y + 1⟩
  | move_8 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 1, y + 2⟩

/-- The minimum number of moves for a knight to reach a target position from a start position -/
def minKnightMoves (start target : Position) : Nat :=
  sorry

theorem knight_moves_equal_for_7x7 :
  let start := Position.mk 0 0
  let upperRight := Position.mk 6 6
  let lowerRight := Position.mk 6 0
  minKnightMoves start upperRight = minKnightMoves start lowerRight :=
by
  sorry

end knight_moves_equal_for_7x7_l2549_254999


namespace gcd_45123_32768_l2549_254975

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end gcd_45123_32768_l2549_254975


namespace volleyball_team_combinations_l2549_254980

theorem volleyball_team_combinations (n : ℕ) (k : ℕ) (h1 : n = 14) (h2 : k = 6) :
  Nat.choose n k = 3003 := by
  sorry

end volleyball_team_combinations_l2549_254980


namespace polynomial_remainder_l2549_254997

theorem polynomial_remainder (f : ℝ → ℝ) (a b c d : ℝ) (h : a ≠ b) :
  (∃ g : ℝ → ℝ, ∀ x, f x = (x - a) * g x + c) →
  (∃ h : ℝ → ℝ, ∀ x, f x = (x - b) * h x + d) →
  ∃ k : ℝ → ℝ, ∀ x, f x = (x - a) * (x - b) * k x + ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b)) :=
by sorry

end polynomial_remainder_l2549_254997


namespace arithmetic_mean_problem_l2549_254968

theorem arithmetic_mean_problem (x : ℚ) : 
  (x + 10 + 20 + 3*x + 18 + (3*x + 6)) / 5 = 30 → x = 96/7 := by
  sorry

end arithmetic_mean_problem_l2549_254968


namespace bridge_length_l2549_254988

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 255 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end bridge_length_l2549_254988


namespace problem_solution_l2549_254924

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2*a + b = 1) :
  (∀ a b, a*b ≥ 1/8) ∧
  (∀ a b, 1/a + 2/b ≥ 8) ∧
  (∀ a b, Real.sqrt (2*a) + Real.sqrt b ≤ Real.sqrt 2) ∧
  (∀ a b, (a+1)*(b+1) < 2) :=
by sorry

end problem_solution_l2549_254924


namespace card_selection_count_l2549_254955

/-- Represents a standard deck of cards with an additional special suit -/
structure Deck :=
  (standard_cards : Nat)
  (special_suit_cards : Nat)
  (ace_count : Nat)

/-- Represents the selection criteria for the cards -/
structure Selection :=
  (total_cards : Nat)
  (min_aces : Nat)
  (different_suits : Bool)

/-- Calculates the number of ways to choose cards according to the given criteria -/
def choose_cards (d : Deck) (s : Selection) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem card_selection_count (d : Deck) (s : Selection) :
  d.standard_cards = 52 →
  d.special_suit_cards = 13 →
  d.ace_count = 4 →
  s.total_cards = 5 →
  s.min_aces = 1 →
  s.different_suits = true →
  choose_cards d s = 114244 :=
sorry

end card_selection_count_l2549_254955


namespace banana_arrangement_count_l2549_254947

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 120

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of 'A's in "BANANA" -/
def num_a : ℕ := 3

/-- Theorem stating that the number of unique arrangements of "BANANA" is correct -/
theorem banana_arrangement_count :
  banana_arrangements = (total_letters.factorial) / (num_a.factorial) :=
by sorry

end banana_arrangement_count_l2549_254947


namespace max_roses_proof_l2549_254967

-- Define the pricing structure
def individual_price : ℚ := 6.3
def dozen_price : ℚ := 36
def two_dozen_price : ℚ := 50
def five_dozen_price : ℚ := 110

-- Define Maria's budget constraints
def total_budget : ℚ := 680
def min_red_roses_budget : ℚ := 200

-- Define the function to calculate the maximum number of roses
def max_roses : ℕ := 360

-- Theorem statement
theorem max_roses_proof :
  ∀ (purchase_strategy : ℕ → ℕ → ℕ → ℕ → ℚ),
  (∀ a b c d, purchase_strategy a b c d * individual_price +
              purchase_strategy a b c d * dozen_price / 12 +
              purchase_strategy a b c d * two_dozen_price / 24 +
              purchase_strategy a b c d * five_dozen_price / 60 ≤ total_budget) →
  (∀ a b c d, purchase_strategy a b c d * five_dozen_price / 60 ≥ min_red_roses_budget) →
  (∀ a b c d, purchase_strategy a b c d + purchase_strategy a b c d * 12 +
              purchase_strategy a b c d * 24 + purchase_strategy a b c d * 60 ≤ max_roses) :=
by sorry

end max_roses_proof_l2549_254967


namespace distance_between_locations_l2549_254922

/-- The distance between two locations A and B given the conditions of two couriers --/
theorem distance_between_locations (x : ℝ) (y : ℝ) : 
  (x > 0) →  -- x is the number of days until the couriers meet
  (y > 0) →  -- y is the total distance between A and B
  (y / (x + 9) + y / (x + 16) = y) →  -- sum of distances traveled equals total distance
  (y / (x + 9) - y / (x + 16) = 12) →  -- difference in distances traveled is 12 miles
  (x^2 = 144) →  -- derived from solving the equations
  y = 84 := by
  sorry

end distance_between_locations_l2549_254922


namespace xiao_ying_performance_l2549_254983

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50
def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

def semester_performance : ℝ :=
  regular_weight * regular_score +
  midterm_weight * midterm_score +
  final_weight * final_score

theorem xiao_ying_performance :
  semester_performance = 90 := by sorry

end xiao_ying_performance_l2549_254983


namespace second_week_rainfall_l2549_254979

/-- Rainfall during the first two weeks of January in Springdale -/
def total_rainfall : ℝ := 20

/-- Ratio of second week's rainfall to first week's rainfall -/
def rainfall_ratio : ℝ := 1.5

/-- Theorem: The rainfall during the second week was 12 inches -/
theorem second_week_rainfall : 
  ∃ (first_week second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = rainfall_ratio * first_week ∧
    second_week = 12 := by
  sorry

end second_week_rainfall_l2549_254979


namespace twenty_percent_value_l2549_254942

theorem twenty_percent_value (x : ℝ) : 1.2 * x = 600 → 0.2 * x = 100 := by
  sorry

end twenty_percent_value_l2549_254942


namespace modern_literature_marks_l2549_254932

theorem modern_literature_marks
  (geography : ℕ) (history_gov : ℕ) (art : ℕ) (comp_sci : ℕ) (avg : ℚ) :
  geography = 56 →
  history_gov = 60 →
  art = 72 →
  comp_sci = 85 →
  avg = 70.6 →
  ∃ (modern_lit : ℕ),
    (geography + history_gov + art + comp_sci + modern_lit : ℚ) / 5 = avg ∧
    modern_lit = 80 := by
  sorry

end modern_literature_marks_l2549_254932


namespace smallest_positive_integer_with_remainders_l2549_254959

theorem smallest_positive_integer_with_remainders : ∃ (b : ℕ), b > 0 ∧
  b % 3 = 2 ∧ b % 5 = 3 ∧ 
  ∀ (x : ℕ), x > 0 ∧ x % 3 = 2 ∧ x % 5 = 3 → b ≤ x :=
by sorry

end smallest_positive_integer_with_remainders_l2549_254959


namespace ball_path_on_5x2_table_l2549_254910

/-- A rectangular table with integer dimensions -/
structure RectTable where
  length : ℕ
  width : ℕ

/-- The path of a ball on a rectangular table -/
def BallPath (table : RectTable) :=
  { bounces : ℕ // bounces ≤ table.length + table.width }

/-- Theorem: A ball on a 5x2 table reaches the opposite corner in 5 bounces -/
theorem ball_path_on_5x2_table :
  ∀ (table : RectTable),
    table.length = 5 →
    table.width = 2 →
    ∃ (path : BallPath table),
      path.val = 5 ∧
      (∀ (other_path : BallPath table), other_path.val ≥ 5) :=
sorry

end ball_path_on_5x2_table_l2549_254910


namespace four_right_angles_implies_plane_figure_less_than_four_right_angles_can_be_non_planar_l2549_254908

-- Define a quadrilateral
structure Quadrilateral :=
  (is_plane : Bool)
  (right_angles : Nat)

-- Define the property of being a plane figure
def is_plane_figure (q : Quadrilateral) : Prop :=
  q.is_plane = true

-- Define the property of having four right angles
def has_four_right_angles (q : Quadrilateral) : Prop :=
  q.right_angles = 4

-- Theorem stating that a quadrilateral with four right angles must be a plane figure
theorem four_right_angles_implies_plane_figure (q : Quadrilateral) :
  has_four_right_angles q → is_plane_figure q :=
by sorry

-- Theorem stating that quadrilaterals with less than four right angles can be non-planar
theorem less_than_four_right_angles_can_be_non_planar :
  ∃ (q : Quadrilateral), q.right_angles < 4 ∧ ¬(is_plane_figure q) :=
by sorry

end four_right_angles_implies_plane_figure_less_than_four_right_angles_can_be_non_planar_l2549_254908


namespace problem_solution_l2549_254946

theorem problem_solution (p q : ℕ) (hp : p < 30) (hq : q < 30) (h_eq : p + q + p * q = 119) :
  p + q = 20 := by
  sorry

end problem_solution_l2549_254946


namespace number_of_adults_at_play_l2549_254953

/-- The number of adults attending a play, given ticket prices and conditions. -/
theorem number_of_adults_at_play : ℕ :=
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  let extra_adult_cost : ℕ := 50
  9

#check number_of_adults_at_play

end number_of_adults_at_play_l2549_254953


namespace imaginary_part_of_complex_fraction_l2549_254981

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.im ((5 + i) / (2 - i)) = 7 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l2549_254981


namespace lizard_comparison_l2549_254921

/-- Represents a three-eyed lizard with wrinkles and spots -/
structure Lizard where
  eyes : Nat
  wrinkle_multiplier : Nat
  spot_multiplier : Nat

/-- Calculates the number of wrinkles for a lizard -/
def wrinkles (l : Lizard) : Nat :=
  l.eyes * l.wrinkle_multiplier

/-- Calculates the number of spots for a lizard -/
def spots (l : Lizard) : Nat :=
  l.spot_multiplier * (wrinkles l) ^ 2

/-- Calculates the total number of spots and wrinkles for a lizard -/
def total_spots_and_wrinkles (l : Lizard) : Nat :=
  spots l + wrinkles l

/-- The main theorem to prove -/
theorem lizard_comparison : 
  let jans_lizard : Lizard := { eyes := 3, wrinkle_multiplier := 3, spot_multiplier := 7 }
  let cousin_lizard : Lizard := { eyes := 3, wrinkle_multiplier := 2, spot_multiplier := 5 }
  total_spots_and_wrinkles jans_lizard + total_spots_and_wrinkles cousin_lizard - 
  (jans_lizard.eyes + cousin_lizard.eyes) = 756 := by
  sorry

end lizard_comparison_l2549_254921
