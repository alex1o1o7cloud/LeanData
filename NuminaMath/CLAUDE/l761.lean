import Mathlib

namespace rationalize_denominator_l761_76168

theorem rationalize_denominator :
  ∃ (A B : ℚ) (C : ℕ),
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
    A = 11 / 4 ∧
    B = 5 / 4 ∧
    C = 5 := by
  sorry

end rationalize_denominator_l761_76168


namespace xy_value_l761_76181

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end xy_value_l761_76181


namespace larger_number_problem_l761_76108

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : 
  max x y = 23 := by
  sorry

end larger_number_problem_l761_76108


namespace f_inequality_l761_76128

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the conditions
axiom period : ∀ x : ℝ, f (x + 3) = f (x - 3)
axiom even_shifted : ∀ x : ℝ, f (x + 3) = f (-x + 3)
axiom decreasing : ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f y < f x

-- State the theorem
theorem f_inequality : f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 := by
  sorry

end f_inequality_l761_76128


namespace johns_money_l761_76159

/-- The problem of determining John's money given the conditions --/
theorem johns_money (total money : ℕ) (ali nada john : ℕ) : 
  total = 67 →
  ali = nada - 5 →
  john = 4 * nada →
  total = ali + nada + john →
  john = 48 := by
  sorry

end johns_money_l761_76159


namespace original_savings_l761_76176

def lindas_savings : ℝ → Prop := λ s =>
  (3/4 * s + 1/4 * s = s) ∧  -- Total spending equals savings
  (1/4 * s = 200)            -- TV cost is 1/4 of savings and equals $200

theorem original_savings : ∃ s : ℝ, lindas_savings s ∧ s = 800 := by
  sorry

end original_savings_l761_76176


namespace problem_statement_l761_76121

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a^2*Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a^2*x - a*Real.log x

theorem problem_statement :
  (∀ a : ℝ, (∃ x_min : ℝ, x_min > 0 ∧ f a x_min = 0 ∧ ∀ x : ℝ, x > 0 → f a x ≥ 0) →
    a = 1 ∨ a = -2 * Real.exp (3/4)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → g a x ≥ 0) →
    0 ≤ a ∧ a ≤ 1) :=
by sorry

end problem_statement_l761_76121


namespace polygon_reassembly_l761_76138

-- Define a polygon as a set of points in 2D space
def Polygon : Type := Set (ℝ × ℝ)

-- Define the area of a polygon
def area (P : Polygon) : ℝ := sorry

-- Define a function to represent cutting and reassembling a polygon
def can_reassemble (P Q : Polygon) : Prop := sorry

-- Define a rectangle with one side of length 1
def rectangle_with_unit_side (R : Polygon) : Prop := sorry

theorem polygon_reassembly (P Q : Polygon) (h : area P = area Q) :
  (∃ R : Polygon, can_reassemble P R ∧ rectangle_with_unit_side R) ∧
  can_reassemble P Q := by sorry

end polygon_reassembly_l761_76138


namespace horner_rule_v2_equals_14_l761_76166

def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

def horner_v2 (a b c d e x : ℝ) : ℝ := ((a * x + b) * x + c) * x + d

theorem horner_rule_v2_equals_14 : 
  horner_v2 2 3 0 5 (-4) 2 = 14 := by
  sorry

end horner_rule_v2_equals_14_l761_76166


namespace m_range_proof_l761_76178

/-- Definition of p -/
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

/-- Definition of q -/
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

/-- ¬p is a sufficient but not necessary condition for ¬q -/
def not_p_sufficient_not_necessary_for_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := 0 < m ∧ m ≤ 3

/-- Main theorem: Given the conditions, prove the range of m -/
theorem m_range_proof :
  ∀ m, not_p_sufficient_not_necessary_for_not_q m → m_range m :=
by sorry

end m_range_proof_l761_76178


namespace vieta_sum_product_l761_76188

theorem vieta_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 15) →
  p + q = 72 :=
by sorry

end vieta_sum_product_l761_76188


namespace envelope_addressing_equation_l761_76170

theorem envelope_addressing_equation : 
  ∀ (x : ℝ), 
  (∃ (machine1_time machine2_time combined_time : ℝ),
    machine1_time = 12 ∧ 
    combined_time = 4 ∧
    machine2_time = x ∧
    (1 / machine1_time + 1 / machine2_time = 1 / combined_time)) ↔
  (1 / 12 + 1 / x = 1 / 4) :=
by sorry

end envelope_addressing_equation_l761_76170


namespace apple_cost_l761_76156

theorem apple_cost (initial_apples : ℕ) (initial_oranges : ℕ) (orange_cost : ℚ) 
  (final_apples : ℕ) (final_oranges : ℕ) (total_earnings : ℚ) :
  initial_apples = 50 →
  initial_oranges = 40 →
  orange_cost = 1/2 →
  final_apples = 10 →
  final_oranges = 6 →
  total_earnings = 49 →
  ∃ (apple_cost : ℚ), apple_cost = 4/5 := by
  sorry

#check apple_cost

end apple_cost_l761_76156


namespace total_distance_calculation_l761_76169

/-- Represents the problem of calculating the total distance traveled by a person
    given specific conditions. -/
theorem total_distance_calculation (d : ℝ) : 
  (d / 6 + d / 12 + d / 18 + d / 24 + d / 30 = 17 / 60) → 
  (5 * d = 425 / 114) := by
  sorry

#check total_distance_calculation

end total_distance_calculation_l761_76169


namespace sum_of_solutions_quadratic_l761_76139

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∀ x, equation x = 0) → sum_of_roots = 5/2 :=
by
  sorry

end sum_of_solutions_quadratic_l761_76139


namespace purely_imaginary_complex_number_l761_76127

/-- Given that (x^2 - 1) + (x^2 + 3x + 2)i is a purely imaginary number, prove that x = 1 -/
theorem purely_imaginary_complex_number (x : ℝ) : 
  (x^2 - 1 : ℂ) + (x^2 + 3*x + 2 : ℂ)*I = (0 : ℂ) + (y : ℝ)*I ∧ y ≠ 0 → x = 1 := by
  sorry


end purely_imaginary_complex_number_l761_76127


namespace power_of_three_mod_five_l761_76150

theorem power_of_three_mod_five : 3^304 % 5 = 1 := by
  sorry

end power_of_three_mod_five_l761_76150


namespace equation_roots_range_l761_76129

theorem equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2*k*x₁^2 + (8*k+1)*x₁ = -8*k ∧ 2*k*x₂^2 + (8*k+1)*x₂ = -8*k) →
  (k > -1/16 ∧ k ≠ 0) :=
by sorry

end equation_roots_range_l761_76129


namespace third_one_is_13th_a_2015_is_31_l761_76177

-- Define the sequence a_n
def a : ℕ → ℚ
| 0 => 0  -- We start from index 1, so define 0 as a placeholder
| n => 
  let k := (n.sqrt + 1) / 2  -- Calculate which group the term belongs to
  let m := n - (k - 1) * k   -- Calculate position within the group
  m.succ / (k + 1 - m)       -- Return the fraction

-- Third term equal to 1
theorem third_one_is_13th : ∃ n₁ n₂ : ℕ, n₁ < n₂ ∧ n₂ < 13 ∧ a n₁ = 1 ∧ a n₂ = 1 ∧ a 13 = 1 :=
sorry

-- 2015th term
theorem a_2015_is_31 : a 2015 = 31 :=
sorry

end third_one_is_13th_a_2015_is_31_l761_76177


namespace sufficient_not_necessary_condition_l761_76102

-- Define the line and hyperbola
def line (a x y : ℝ) : Prop := 2 * a * x - y + 2 * a^2 = 0
def hyperbola (a x y : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1

-- Define the condition for no focus
def no_focus (a : ℝ) : Prop := ∀ x y : ℝ, line a x y → hyperbola a x y → False

-- State the theorem
theorem sufficient_not_necessary_condition (a : ℝ) (h : a > 0) :
  (a ≥ 2 → no_focus a) ∧ ¬(no_focus a → a ≥ 2) :=
by sorry

end sufficient_not_necessary_condition_l761_76102


namespace collinear_points_sum_l761_76180

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define collinearity for four points in 3D space
def collinear (p q r s : Point3D) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), 
    q.x - p.x = t₁ * (r.x - p.x) ∧
    q.y - p.y = t₁ * (r.y - p.y) ∧
    q.z - p.z = t₁ * (r.z - p.z) ∧
    s.x - p.x = t₂ * (r.x - p.x) ∧
    s.y - p.y = t₂ * (r.y - p.y) ∧
    s.z - p.z = t₂ * (r.z - p.z) ∧
    t₃ * (q.x - p.x) = s.x - p.x ∧
    t₃ * (q.y - p.y) = s.y - p.y ∧
    t₃ * (q.z - p.z) = s.z - p.z

theorem collinear_points_sum (a b : ℝ) : 
  collinear 
    (Point3D.mk 2 a b) 
    (Point3D.mk a 3 b) 
    (Point3D.mk a b 4) 
    (Point3D.mk 5 b a) → 
  a + b = 9 := by
  sorry

end collinear_points_sum_l761_76180


namespace inequality_range_of_a_l761_76100

theorem inequality_range_of_a (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |x^2 - a*x| + b < 0) ↔
  ((b ≥ -1 ∧ b < 0 ∧ a ∈ Set.Ioo (1 + b) (2 * Real.sqrt (-b))) ∨
   (b < -1 ∧ a ∈ Set.Ioo (1 + b) (1 - b))) :=
by sorry

end inequality_range_of_a_l761_76100


namespace black_queen_thought_l761_76123

-- Define the possible states for each character
inductive State
  | Asleep
  | Awake

-- Define the characters
structure Character where
  name : String
  state : State
  thought : State

-- Define the perverse judgment property
def perverseJudgment (c : Character) : Prop :=
  (c.state = State.Asleep ∧ c.thought = State.Awake) ∨
  (c.state = State.Awake ∧ c.thought = State.Asleep)

-- Define the rational judgment property
def rationalJudgment (c : Character) : Prop :=
  (c.state = State.Asleep ∧ c.thought = State.Asleep) ∨
  (c.state = State.Awake ∧ c.thought = State.Awake)

-- Theorem statement
theorem black_queen_thought (blackKing blackQueen : Character) :
  blackKing.name = "Black King" →
  blackQueen.name = "Black Queen" →
  blackKing.thought = State.Asleep →
  (perverseJudgment blackKing ∨ rationalJudgment blackKing) →
  (perverseJudgment blackQueen ∨ rationalJudgment blackQueen) →
  blackQueen.thought = State.Asleep :=
by
  sorry

end black_queen_thought_l761_76123


namespace extended_tile_ratio_l761_76107

/-- The ratio of black tiles to white tiles in an extended rectangular pattern -/
theorem extended_tile_ratio (orig_width orig_height : ℕ) 
  (orig_black orig_white : ℕ) : 
  orig_width = 5 → 
  orig_height = 6 → 
  orig_black = 12 → 
  orig_white = 18 → 
  (orig_black : ℚ) / ((orig_white : ℚ) + 2 * (orig_width + orig_height + 2)) = 3 / 11 := by
  sorry

end extended_tile_ratio_l761_76107


namespace exists_b_for_even_f_l761_76162

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = 2x^2 - bx where b is a real number -/
def f (b : ℝ) : ℝ → ℝ := λ x ↦ 2 * x^2 - b * x

/-- There exists a real number b such that f(x) = 2x^2 - bx is an even function -/
theorem exists_b_for_even_f : ∃ b : ℝ, IsEven (f b) := by
  sorry

end exists_b_for_even_f_l761_76162


namespace base_number_problem_l761_76184

theorem base_number_problem (x : ℝ) (k : ℕ) 
  (h1 : x^k = 5) 
  (h2 : x^(2*k + 2) = 400) : 
  x = 5 := by sorry

end base_number_problem_l761_76184


namespace product_selection_probabilities_l761_76110

def totalProducts : ℕ := 5
def authenticProducts : ℕ := 3
def defectiveProducts : ℕ := 2

theorem product_selection_probabilities :
  let totalSelections := totalProducts.choose 2
  let bothAuthenticSelections := authenticProducts.choose 2
  let mixedSelections := authenticProducts * defectiveProducts
  (bothAuthenticSelections : ℚ) / totalSelections = 3 / 10 ∧
  (mixedSelections : ℚ) / totalSelections = 3 / 5 ∧
  1 - (bothAuthenticSelections : ℚ) / totalSelections = 7 / 10 := by
  sorry

end product_selection_probabilities_l761_76110


namespace sum_of_digits_1_to_10000_l761_76147

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sequence_sum_of_digits (n : Nat) : Nat :=
  (List.range n).map sum_of_digits |> List.sum

theorem sum_of_digits_1_to_10000 :
  sequence_sum_of_digits 10000 = 180001 := by
  sorry

end sum_of_digits_1_to_10000_l761_76147


namespace parallelepiped_coverage_l761_76142

/-- Represents a parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a square -/
structure Square where
  side : ℕ

/-- Calculates the surface area of a parallelepiped -/
def surface_area (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.length * p.height + p.width * p.height)

/-- Calculates the area of a square -/
def square_area (s : Square) : ℕ :=
  s.side * s.side

/-- Theorem stating that a 1x1x4 parallelepiped can be covered by two 4x4 squares and one 1x1 square -/
theorem parallelepiped_coverage :
  ∃ (p : Parallelepiped) (s1 s2 s3 : Square),
    p.length = 1 ∧ p.width = 1 ∧ p.height = 4 ∧
    s1.side = 4 ∧ s2.side = 4 ∧ s3.side = 1 ∧
    surface_area p = square_area s1 + square_area s2 + square_area s3 :=
sorry

end parallelepiped_coverage_l761_76142


namespace area_of_EFGH_l761_76160

-- Define the properties of the smaller rectangles
def short_side : ℝ := 4
def long_side : ℝ := 2 * short_side

-- Define the properties of the larger rectangle EFGH
def EFGH_width : ℝ := 4 * long_side
def EFGH_length : ℝ := short_side

-- State the theorem
theorem area_of_EFGH : EFGH_width * EFGH_length = 128 := by
  sorry

end area_of_EFGH_l761_76160


namespace base_r_transaction_l761_76164

/-- Converts a number from base r to base 10 -/
def to_base_10 (digits : List Nat) (r : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * r ^ i) 0

/-- The problem statement -/
theorem base_r_transaction (r : Nat) : r > 1 →
  (to_base_10 [0, 6, 5] r) + (to_base_10 [0, 2, 4] r) = (to_base_10 [0, 0, 1, 1] r) ↔ r = 8 := by
  sorry

end base_r_transaction_l761_76164


namespace cost_price_example_l761_76119

/-- Given a selling price and a profit percentage, calculate the cost price -/
def cost_price (selling_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  selling_price / (1 + profit_percentage / 100)

/-- Theorem: Given a selling price of 500 and a profit of 25%, the cost price is 400 -/
theorem cost_price_example : cost_price 500 25 = 400 := by
  sorry

end cost_price_example_l761_76119


namespace consecutive_page_numbers_l761_76182

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 15300 → n + (n + 1) = 247 := by
  sorry

end consecutive_page_numbers_l761_76182


namespace positive_y_floor_product_l761_76134

theorem positive_y_floor_product (y : ℝ) : 
  y > 0 → y * ⌊y⌋ = 90 → y = 10 := by sorry

end positive_y_floor_product_l761_76134


namespace problem_statement_l761_76153

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x * y > a * b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → x^2 + y^2 ≥ 1/2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → 4/x + 1/y ≥ 9) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y < Real.sqrt 2) :=
by sorry

end problem_statement_l761_76153


namespace interchange_difference_for_62_l761_76101

def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def interchange_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem interchange_difference_for_62 :
  is_two_digit_number 62 ∧ digit_sum 62 = 8 →
  62 - interchange_digits 62 = 36 := by
  sorry

end interchange_difference_for_62_l761_76101


namespace alex_original_seat_l761_76186

/-- Represents a seat in the movie theater --/
inductive Seat
| one | two | three | four | five | six | seven

/-- Represents the possible movements of friends --/
inductive Movement
| left : ℕ → Movement
| right : ℕ → Movement
| switch : Movement
| none : Movement

/-- Represents a friend in the theater --/
structure Friend :=
  (name : String)
  (initial_seat : Seat)
  (movement : Movement)

/-- The state of the theater --/
structure TheaterState :=
  (friends : List Friend)
  (alex_initial : Seat)
  (alex_final : Seat)

def is_end_seat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.seven

def move_left (s : Seat) (n : ℕ) : Seat :=
  match s, n with
  | Seat.one, _ => Seat.one
  | Seat.two, 1 => Seat.one
  | Seat.three, 1 => Seat.two
  | Seat.three, 2 => Seat.one
  | Seat.four, 1 => Seat.three
  | Seat.four, 2 => Seat.two
  | Seat.four, 3 => Seat.one
  | Seat.five, 1 => Seat.four
  | Seat.five, 2 => Seat.three
  | Seat.five, 3 => Seat.two
  | Seat.five, 4 => Seat.one
  | Seat.six, 1 => Seat.five
  | Seat.six, 2 => Seat.four
  | Seat.six, 3 => Seat.three
  | Seat.six, 4 => Seat.two
  | Seat.six, 5 => Seat.one
  | Seat.seven, 1 => Seat.six
  | Seat.seven, 2 => Seat.five
  | Seat.seven, 3 => Seat.four
  | Seat.seven, 4 => Seat.three
  | Seat.seven, 5 => Seat.two
  | Seat.seven, 6 => Seat.one
  | s, _ => s

theorem alex_original_seat (state : TheaterState) :
  state.friends = [
    ⟨"Bob", Seat.three, Movement.right 3⟩,
    ⟨"Cara", Seat.five, Movement.left 2⟩,
    ⟨"Dana", Seat.four, Movement.switch⟩,
    ⟨"Eve", Seat.two, Movement.switch⟩,
    ⟨"Fiona", Seat.six, Movement.right 1⟩,
    ⟨"Greg", Seat.seven, Movement.none⟩
  ] →
  is_end_seat state.alex_final →
  state.alex_initial = Seat.three :=
by sorry


end alex_original_seat_l761_76186


namespace ordered_pair_solution_l761_76146

theorem ordered_pair_solution :
  ∀ (c d : ℤ),
  Real.sqrt (16 - 12 * Real.cos (30 * π / 180)) = c + d * (1 / Real.cos (30 * π / 180)) →
  c = 4 ∧ d = -1 := by
sorry

end ordered_pair_solution_l761_76146


namespace ferris_wheel_ticket_cost_l761_76157

theorem ferris_wheel_ticket_cost 
  (initial_tickets : ℕ) 
  (remaining_tickets : ℕ) 
  (total_spent : ℕ) 
  (h1 : initial_tickets = 13)
  (h2 : remaining_tickets = 4)
  (h3 : total_spent = 81) :
  total_spent / (initial_tickets - remaining_tickets) = 9 := by
sorry

end ferris_wheel_ticket_cost_l761_76157


namespace increase_decrease_percentage_l761_76106

theorem increase_decrease_percentage (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) : 
  let increased := initial * (1 + increase_percent / 100)
  let final := increased * (1 - decrease_percent / 100)
  initial = 80 ∧ increase_percent = 150 ∧ decrease_percent = 25 → final = 150 := by
  sorry

end increase_decrease_percentage_l761_76106


namespace wolf_hunger_theorem_l761_76163

/-- Represents the satiety value of a food item -/
structure SatietyValue (α : Type) where
  value : ℝ

/-- Represents the satiety state of the wolf -/
inductive SatietyState
  | Hunger
  | Satisfied
  | Overeating

/-- The satiety value of a piglet -/
def piglet_satiety : SatietyValue ℝ := ⟨1⟩

/-- The satiety value of a kid -/
def kid_satiety : SatietyValue ℝ := ⟨1⟩

/-- Calculates the total satiety value of a meal -/
def meal_satiety (piglets kids : ℕ) : ℝ :=
  (piglets : ℝ) * piglet_satiety.value + (kids : ℝ) * kid_satiety.value

/-- Determines the satiety state based on the meal satiety -/
def get_satiety_state (meal : ℝ) : SatietyState := sorry

/-- The theorem to be proved -/
theorem wolf_hunger_theorem :
  (get_satiety_state (meal_satiety 3 7) = SatietyState.Hunger) →
  (get_satiety_state (meal_satiety 7 1) = SatietyState.Overeating) →
  (get_satiety_state (meal_satiety 0 11) = SatietyState.Hunger) :=
by sorry

end wolf_hunger_theorem_l761_76163


namespace cubic_equation_real_root_l761_76183

theorem cubic_equation_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) * (x + 2) := by
sorry

end cubic_equation_real_root_l761_76183


namespace nancys_payment_is_384_l761_76122

/-- Nancy's annual payment for her daughter's car insurance -/
def nancys_annual_payment (monthly_cost : ℝ) (nancys_percentage : ℝ) : ℝ :=
  monthly_cost * nancys_percentage * 12

/-- Theorem: Nancy's annual payment for her daughter's car insurance is $384 -/
theorem nancys_payment_is_384 :
  nancys_annual_payment 80 0.4 = 384 := by
  sorry

end nancys_payment_is_384_l761_76122


namespace bread_slices_proof_l761_76113

theorem bread_slices_proof (S : ℕ) : S ≥ 20 → (∃ T : ℕ, S = 2 * T + 10 ∧ S - 7 = 2 * T + 3) → S ≥ 20 := by
  sorry

end bread_slices_proof_l761_76113


namespace expression_evaluation_l761_76132

/-- Given a = -2 and b = -1/2, prove that the expression 3(2a²-4ab)-[a²-3(4a+ab)] evaluates to -13 -/
theorem expression_evaluation (a b : ℚ) (h1 : a = -2) (h2 : b = -1/2) :
  3 * (2 * a^2 - 4 * a * b) - (a^2 - 3 * (4 * a + a * b)) = -13 := by
  sorry

end expression_evaluation_l761_76132


namespace tens_digit_of_13_pow_2023_l761_76165

theorem tens_digit_of_13_pow_2023 :
  ∃ k : ℕ, 13^2023 = 100 * k + 97 :=
by
  -- We assume 13^20 ≡ 1 (mod 100) as a hypothesis
  have h1 : ∃ m : ℕ, 13^20 = 100 * m + 1 := sorry
  
  -- We use the division algorithm to write 2023 = 20q + r
  have h2 : ∃ q r : ℕ, 2023 = 20 * q + r ∧ r < 20 := sorry
  
  -- We prove that r = 3
  have h3 : ∃ q : ℕ, 2023 = 20 * q + 3 := sorry
  
  -- We prove that 13^3 ≡ 97 (mod 100)
  have h4 : ∃ n : ℕ, 13^3 = 100 * n + 97 := sorry
  
  -- Main proof
  sorry

end tens_digit_of_13_pow_2023_l761_76165


namespace becketts_age_l761_76190

theorem becketts_age (beckett olaf shannen jack : ℕ) 
  (h1 : beckett = olaf - 3)
  (h2 : shannen = olaf - 2)
  (h3 : jack = 2 * shannen + 5)
  (h4 : beckett + olaf + shannen + jack = 71) :
  beckett = 12 := by
  sorry

end becketts_age_l761_76190


namespace find_y_l761_76199

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (4 * b) = a ^ b * y ^ (3 * b)) : 
  y = 2 ^ (4 / 3) * a :=
sorry

end find_y_l761_76199


namespace pairwise_sum_problem_l761_76194

/-- Given four numbers that when added pairwise result in specific sums, 
    prove the remaining sums and possible sets of numbers -/
theorem pairwise_sum_problem (a b c d : ℝ) : 
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
  d + c = 20 ∧ d + b = 16 ∧ 
  ((d + a = 13 ∧ c + b = 9) ∨ (d + a = 9 ∧ c + b = 13)) →
  (a + b = 2 ∧ a + c = 6) ∧
  ((a = -0.5 ∧ b = 2.5 ∧ c = 6.5 ∧ d = 13.5) ∨
   (a = -2.5 ∧ b = 4.5 ∧ c = 8.5 ∧ d = 11.5)) := by
  sorry

end pairwise_sum_problem_l761_76194


namespace tims_score_is_56_l761_76191

/-- The sum of the first n even numbers -/
def sum_first_n_even (n : ℕ) : ℕ :=
  (2 * n * (n + 1)) / 2

/-- Tim's math score -/
def tims_score : ℕ := sum_first_n_even 7

theorem tims_score_is_56 : tims_score = 56 := by
  sorry

end tims_score_is_56_l761_76191


namespace product_of_roots_plus_one_l761_76189

theorem product_of_roots_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 20*a - 8 = 0) → 
  (b^3 - 15*b^2 + 20*b - 8 = 0) → 
  (c^3 - 15*c^2 + 20*c - 8 = 0) → 
  (1 + a) * (1 + b) * (1 + c) = 28 := by
sorry

end product_of_roots_plus_one_l761_76189


namespace even_function_implies_a_equals_four_l761_76179

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x + a)(x - 4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x + a)(x - 4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four :
  ∀ a : ℝ, IsEven (f a) → a = 4 := by
  sorry

end even_function_implies_a_equals_four_l761_76179


namespace arithmetic_sequence_common_difference_l761_76145

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 2) 
  (h_sum_condition : a 2 + a 4 = a 6) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) - a n = d) ∧ d = 2 := by
sorry

end arithmetic_sequence_common_difference_l761_76145


namespace common_external_tangent_y_intercept_l761_76161

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: The y-intercept of the common external tangent with positive slope for two given circles --/
theorem common_external_tangent_y_intercept 
  (c1 : Circle) 
  (c2 : Circle) 
  (h1 : c1.center = (5, -2)) 
  (h2 : c1.radius = 5)
  (h3 : c2.center = (20, 6))
  (h4 : c2.radius = 12) :
  ∃ (m b : ℝ), m > 0 ∧ b = -2100/161 ∧ 
  (∀ (x y : ℝ), y = m * x + b ↔ 
    (y - c1.center.2)^2 + (x - c1.center.1)^2 = (c1.radius + c2.radius)^2 ∧
    (y - c2.center.2)^2 + (x - c2.center.1)^2 = (c1.radius + c2.radius)^2) :=
sorry

end common_external_tangent_y_intercept_l761_76161


namespace least_k_value_l761_76117

theorem least_k_value (p q k : ℕ) : 
  p > 1 → 
  q > 1 → 
  p + q = 36 → 
  17 * (p + 1) = k * (q + 1) → 
  k ≥ 2 ∧ (∃ (p' q' : ℕ), p' > 1 ∧ q' > 1 ∧ p' + q' = 36 ∧ 17 * (p' + 1) = 2 * (q' + 1)) :=
by sorry

end least_k_value_l761_76117


namespace particle_motion_l761_76120

/-- Height of the particle in meters after t seconds -/
def s (t : ℝ) : ℝ := 180 * t - 18 * t^2

/-- Time at which the particle reaches its highest point -/
def t_max : ℝ := 5

/-- The highest elevation reached by the particle -/
def h_max : ℝ := 450

theorem particle_motion :
  (∀ t : ℝ, s t ≤ h_max) ∧
  s t_max = h_max :=
sorry

end particle_motion_l761_76120


namespace johns_number_l761_76175

theorem johns_number (n : ℕ) : 
  (125 ∣ n) ∧ 
  (180 ∣ n) ∧ 
  1000 ≤ n ∧ 
  n ≤ 3000 ∧
  (∀ m : ℕ, (125 ∣ m) ∧ (180 ∣ m) ∧ 1000 ≤ m ∧ m ≤ 3000 → n ≤ m) →
  n = 1800 := by
sorry

end johns_number_l761_76175


namespace thomas_monthly_earnings_l761_76198

/-- Calculates Thomas's total earnings for one month --/
def thomasEarnings (initialWage : ℝ) (weeklyIncrease : ℝ) (overtimeHours : ℕ) (overtimeRate : ℝ) (deduction : ℝ) : ℝ :=
  let week1 := initialWage
  let week2 := initialWage * (1 + weeklyIncrease)
  let week3 := week2 * (1 + weeklyIncrease)
  let week4 := week3 * (1 + weeklyIncrease)
  let overtimePay := (overtimeHours : ℝ) * overtimeRate
  week1 + week2 + week3 + week4 + overtimePay - deduction

/-- Theorem stating that Thomas's earnings for the month equal $19,761.07 --/
theorem thomas_monthly_earnings :
  thomasEarnings 4550 0.05 10 25 100 = 19761.07 := by
  sorry

end thomas_monthly_earnings_l761_76198


namespace unique_prime_twice_squares_l761_76131

theorem unique_prime_twice_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x : ℕ), p + 1 = 2 * x^2) ∧ 
    (∃ (y : ℕ), p^2 + 1 = 2 * y^2) ∧ 
    p = 7 :=
by sorry

end unique_prime_twice_squares_l761_76131


namespace bisection_next_point_l761_76103

theorem bisection_next_point 
  (f : ℝ → ℝ) 
  (h_continuous : ContinuousOn f (Set.Icc 1 2))
  (h_f1 : f 1 < 0)
  (h_f1_5 : f 1.5 > 0) :
  (1 + 1.5) / 2 = 1.25 := by sorry

end bisection_next_point_l761_76103


namespace polynomial_value_relation_l761_76136

theorem polynomial_value_relation (m n : ℝ) : 
  -m^2 + 3*n = 2 → m^2 - 3*n - 1 = -3 := by
sorry

end polynomial_value_relation_l761_76136


namespace circle_through_points_center_on_line_l761_76173

/-- A circle passing through two points with its center on a given line -/
theorem circle_through_points_center_on_line (A B O : ℝ × ℝ) (r : ℝ) :
  A = (1, -1) →
  B = (-1, 1) →
  O.1 + O.2 = 2 →
  r = 2 →
  ∀ (x y : ℝ), (x - O.1)^2 + (y - O.2)^2 = r^2 ↔
    ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) ∨
    (x - O.1)^2 + (y - O.2)^2 = r^2 :=
by sorry

end circle_through_points_center_on_line_l761_76173


namespace sqrt_pi_squared_minus_6pi_plus_9_l761_76197

theorem sqrt_pi_squared_minus_6pi_plus_9 : 
  Real.sqrt (π^2 - 6*π + 9) = π - 3 := by
  sorry

end sqrt_pi_squared_minus_6pi_plus_9_l761_76197


namespace circle_radius_when_area_is_250_percent_of_circumference_l761_76151

theorem circle_radius_when_area_is_250_percent_of_circumference (r : ℝ) : 
  r > 0 → π * r^2 = 2.5 * (2 * π * r) → r = 5 := by
  sorry

end circle_radius_when_area_is_250_percent_of_circumference_l761_76151


namespace towel_average_price_l761_76126

theorem towel_average_price :
  let towel_group1 : ℕ := 3
  let price1 : ℕ := 100
  let towel_group2 : ℕ := 5
  let price2 : ℕ := 150
  let towel_group3 : ℕ := 2
  let price3 : ℕ := 400
  let total_towels := towel_group1 + towel_group2 + towel_group3
  let total_cost := towel_group1 * price1 + towel_group2 * price2 + towel_group3 * price3
  (total_cost : ℚ) / (total_towels : ℚ) = 185 := by
  sorry

end towel_average_price_l761_76126


namespace arithmetic_sequence_problem_l761_76149

theorem arithmetic_sequence_problem (d a_n n : ℤ) (h1 : d = 2) (h2 : n = 15) (h3 : a_n = -10) :
  ∃ (a_1 S_n : ℤ),
    a_1 = -38 ∧
    S_n = -360 ∧
    a_n = a_1 + (n - 1) * d ∧
    S_n = n * (a_1 + a_n) / 2 :=
by sorry


end arithmetic_sequence_problem_l761_76149


namespace min_distance_exp_curve_to_line_l761_76130

/-- The minimum distance between a point on y = e^x and a point on y = x is √2/2 -/
theorem min_distance_exp_curve_to_line : 
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
    ∀ (p q : ℝ × ℝ), p.2 = Real.exp p.1 → q.2 = q.1 → 
      dist p q ≥ d :=
sorry

end min_distance_exp_curve_to_line_l761_76130


namespace perfect_square_condition_l761_76174

theorem perfect_square_condition (a : ℕ+) 
  (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ (n^2 * a.val - 1) % d = 0) : 
  ∃ k : ℕ, a.val = k^2 := by
  sorry

end perfect_square_condition_l761_76174


namespace faiths_weekly_earnings_l761_76185

/-- Calculates the total weekly earnings for Faith given her work conditions --/
def total_weekly_earnings (
  hourly_wage : ℝ)
  (regular_hours_per_day : ℝ)
  (regular_days_per_week : ℝ)
  (overtime_hours_per_day : ℝ)
  (overtime_days_per_week : ℝ)
  (overtime_rate_multiplier : ℝ)
  (commission_rate : ℝ)
  (total_sales : ℝ) : ℝ :=
  let regular_earnings := hourly_wage * regular_hours_per_day * regular_days_per_week
  let overtime_earnings := hourly_wage * overtime_rate_multiplier * overtime_hours_per_day * overtime_days_per_week
  let commission := commission_rate * total_sales
  regular_earnings + overtime_earnings + commission

/-- Theorem stating that Faith's total weekly earnings are $1,062.50 --/
theorem faiths_weekly_earnings :
  total_weekly_earnings 13.5 8 5 2 5 1.5 0.1 3200 = 1062.5 := by
  sorry

end faiths_weekly_earnings_l761_76185


namespace evaluate_f_l761_76192

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

-- State the theorem
theorem evaluate_f : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end evaluate_f_l761_76192


namespace probability_play_exactly_one_l761_76154

def total_people : ℕ := 800
def play_at_least_one_ratio : ℚ := 1 / 5
def play_two_or_more : ℕ := 64

theorem probability_play_exactly_one (total_people : ℕ) (play_at_least_one_ratio : ℚ) (play_two_or_more : ℕ) :
  (play_at_least_one_ratio * total_people - play_two_or_more : ℚ) / total_people = 12 / 100 :=
by sorry

end probability_play_exactly_one_l761_76154


namespace divisibility_by_20p_l761_76124

theorem divisibility_by_20p (p : ℕ) (h_prime : Prime p) (h_odd : Odd p) :
  ∃ k : ℤ, (⌊(Real.sqrt 5 + 2)^p - 2^(p+1)⌋ : ℤ) = 20 * p * k := by
  sorry

end divisibility_by_20p_l761_76124


namespace unique_congruence_in_range_l761_76196

theorem unique_congruence_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -1872 [ZMOD 9] ∧ n = 0 := by
  sorry

end unique_congruence_in_range_l761_76196


namespace jessica_bank_account_l761_76144

theorem jessica_bank_account (B : ℝ) : 
  B > 0 →
  (3/5) * B = B - 200 →
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    (3/5) * B + (x/y) * ((3/5) * B) = 450 →
    x/y = 1/2 := by
  sorry

end jessica_bank_account_l761_76144


namespace no_real_m_for_equal_roots_l761_76148

/-- The equation whose roots we're analyzing -/
def equation (x m : ℝ) : Prop :=
  (3 * x^2 * (x - 2) - (2*m + 3)) / ((x - 2) * (m - 2)) = 2 * x^2 / m

/-- Theorem stating that there are no real values of m for which the roots of the equation are equal -/
theorem no_real_m_for_equal_roots :
  ¬ ∃ (m : ℝ), ∃ (x : ℝ), ∀ (y : ℝ), equation y m → y = x :=
sorry

end no_real_m_for_equal_roots_l761_76148


namespace equation_solutions_l761_76158

theorem equation_solutions : 
  ∀ n m : ℕ+, 3 * 2^(m : ℕ) + 1 = (n : ℕ)^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) := by
  sorry

end equation_solutions_l761_76158


namespace sponge_city_philosophy_l761_76112

/-- Represents a sponge city -/
structure SpongeCity where
  resilience : Bool
  waterManagement : Bool
  pilotProject : Bool

/-- Philosophical perspectives on sponge cities -/
inductive PhilosophicalPerspective
  | overall_function_greater
  | integrated_thinking
  | new_connections
  | internal_structure_optimization

/-- Checks if a given philosophical perspective applies to sponge cities -/
def applies_to_sponge_cities (sc : SpongeCity) (pp : PhilosophicalPerspective) : Prop :=
  match pp with
  | PhilosophicalPerspective.overall_function_greater => true
  | PhilosophicalPerspective.integrated_thinking => true
  | _ => false

/-- Theorem: Sponge cities reflect specific philosophical perspectives -/
theorem sponge_city_philosophy (sc : SpongeCity) 
  (h1 : sc.resilience = true) 
  (h2 : sc.waterManagement = true) 
  (h3 : sc.pilotProject = true) :
  (applies_to_sponge_cities sc PhilosophicalPerspective.overall_function_greater) ∧
  (applies_to_sponge_cities sc PhilosophicalPerspective.integrated_thinking) :=
by
  sorry

end sponge_city_philosophy_l761_76112


namespace common_chord_of_circles_l761_76143

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 25 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3*y - 10 = 0

/-- The equation of the potential common chord -/
def common_chord (x y : ℝ) : Prop := 4*x - 3*y - 15 = 0

/-- Theorem stating that the given equation represents the common chord of the two circles -/
theorem common_chord_of_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end common_chord_of_circles_l761_76143


namespace jennas_eel_length_l761_76171

theorem jennas_eel_length (j b : ℝ) (h1 : j = b / 3) (h2 : j + b = 64) :
  j = 16 := by
  sorry

end jennas_eel_length_l761_76171


namespace shaded_area_square_with_quarter_circles_l761_76104

/-- The area of the shaded region inside a square with side length 16 cm but outside
    four quarter circles with radius 6 cm at each corner is 256 - 36π cm². -/
theorem shaded_area_square_with_quarter_circles (π : ℝ) :
  let square_side : ℝ := 16
  let circle_radius : ℝ := 6
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_area : ℝ := π * circle_radius ^ 2 / 4
  let total_quarter_circles_area : ℝ := 4 * quarter_circle_area
  let shaded_area : ℝ := square_area - total_quarter_circles_area
  shaded_area = 256 - 36 * π :=
by sorry

end shaded_area_square_with_quarter_circles_l761_76104


namespace sandy_marks_problem_l761_76125

theorem sandy_marks_problem (marks_per_correct : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_sums : ℕ) :
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 65 →
  correct_sums = 25 →
  (marks_per_correct * correct_sums - total_marks) / (total_attempts - correct_sums) = 2 :=
by sorry

end sandy_marks_problem_l761_76125


namespace instagram_followers_after_year_l761_76140

/-- Calculates the final number of followers for an Instagram influencer after a year --/
theorem instagram_followers_after_year 
  (initial_followers : ℕ) 
  (new_followers_per_day : ℕ) 
  (days_in_year : ℕ) 
  (unfollowers : ℕ) 
  (h1 : initial_followers = 100000)
  (h2 : new_followers_per_day = 1000)
  (h3 : days_in_year = 365)
  (h4 : unfollowers = 20000) :
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers = 445000 :=
by sorry

end instagram_followers_after_year_l761_76140


namespace power_equation_solution_l761_76167

theorem power_equation_solution (n : ℕ) : (3^n)^2 = 3^16 → n = 8 := by
  sorry

end power_equation_solution_l761_76167


namespace range_of_t_below_line_l761_76105

/-- A point (x, y) is below a line ax + by + c = 0 if ax + by + c > 0 -/
def IsBelowLine (x y a b c : ℝ) : Prop := a * x + b * y + c > 0

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t_below_line :
  ∀ t : ℝ, IsBelowLine 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
by sorry

end range_of_t_below_line_l761_76105


namespace units_digit_of_fraction_l761_76115

theorem units_digit_of_fraction (n : ℕ) : n = 30 * 31 * 32 * 33 * 34 / 120 → n % 10 = 4 := by
  sorry

end units_digit_of_fraction_l761_76115


namespace triangle_condition_l761_76114

theorem triangle_condition (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 →
  (x + y > z ∧ x + z > y ∧ y + z > x) ↔ (x < 1 ∧ y < 1 ∧ z < 1) :=
by sorry

end triangle_condition_l761_76114


namespace candy_bar_price_is_correct_l761_76109

/-- The selling price of a candy bar that results in a $25 profit when selling 5 boxes of 10 candy bars, each bought for $1. -/
def candy_bar_price : ℚ :=
  let boxes : ℕ := 5
  let bars_per_box : ℕ := 10
  let cost_price : ℚ := 1
  let total_profit : ℚ := 25
  let total_bars : ℕ := boxes * bars_per_box
  (total_profit / total_bars + cost_price)

theorem candy_bar_price_is_correct : candy_bar_price = 3/2 := by
  sorry

end candy_bar_price_is_correct_l761_76109


namespace number_of_combinations_max_probability_sums_l761_76195

-- Define the structure of a box
structure Box :=
  (ball1 : Nat)
  (ball2 : Nat)

-- Define the set of boxes
def boxes : List Box := [
  { ball1 := 1, ball2 := 2 },
  { ball1 := 1, ball2 := 2 },
  { ball1 := 1, ball2 := 2 }
]

-- Define a combination of drawn balls
def Combination := Nat × Nat × Nat

-- Function to generate all possible combinations
def generateCombinations (boxes : List Box) : List Combination := sorry

-- Function to calculate the sum of a combination
def sumCombination (c : Combination) : Nat := sorry

-- Function to count occurrences of a sum
def countSum (sum : Nat) (combinations : List Combination) : Nat := sorry

-- Theorem: The number of possible combinations is 8
theorem number_of_combinations :
  (generateCombinations boxes).length = 8 := by sorry

-- Theorem: The sums 4 and 5 have the highest probability
theorem max_probability_sums (combinations : List Combination := generateCombinations boxes) :
  ∀ (s : Nat), s ≠ 4 ∧ s ≠ 5 →
    countSum s combinations ≤ countSum 4 combinations ∧
    countSum s combinations ≤ countSum 5 combinations := by sorry

end number_of_combinations_max_probability_sums_l761_76195


namespace max_value_y_plus_one_squared_l761_76116

theorem max_value_y_plus_one_squared (y : ℝ) : 
  (4 * y^2 + 4 * y + 3 = 1) → ((y + 1)^2 ≤ (1/4 : ℝ)) ∧ (∃ y : ℝ, 4 * y^2 + 4 * y + 3 = 1 ∧ (y + 1)^2 = (1/4 : ℝ)) := by
  sorry

end max_value_y_plus_one_squared_l761_76116


namespace badminton_team_combinations_l761_76118

theorem badminton_team_combinations : 
  ∀ (male_players female_players : ℕ), 
    male_players = 6 → 
    female_players = 7 → 
    (male_players.choose 1) * (female_players.choose 1) = 42 := by
sorry

end badminton_team_combinations_l761_76118


namespace combined_weight_l761_76137

/-- The weight of a peach in grams -/
def peach_weight : ℝ := sorry

/-- The weight of a bun in grams -/
def bun_weight : ℝ := sorry

/-- Condition 1: One peach weighs the same as 2 buns plus 40 grams -/
axiom condition1 : peach_weight = 2 * bun_weight + 40

/-- Condition 2: One peach plus 80 grams weighs the same as one bun plus 200 grams -/
axiom condition2 : peach_weight + 80 = bun_weight + 200

/-- Theorem: The combined weight of 1 peach and 1 bun is 280 grams -/
theorem combined_weight : peach_weight + bun_weight = 280 := by sorry

end combined_weight_l761_76137


namespace problem_statement_l761_76187

-- Define proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, (f x₁ - f x₂) * (x₁ - x₂) ≥ 0

-- Define proposition q
def q : Prop :=
  ∀ x y : ℝ, x + y > 2 → x > 1 ∨ y > 1

-- Define decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem problem_statement (f : ℝ → ℝ) :
  (¬(p f ∧ q)) ∧ q → is_decreasing f :=
by sorry

end problem_statement_l761_76187


namespace independence_test_conclusions_not_always_correct_l761_76135

-- Define the concept of independence tests
def IndependenceTest : Type := Unit

-- Define the properties of independence tests
axiom small_probability_principle : IndependenceTest → Prop
axiom conclusions_vary_with_samples : IndependenceTest → Prop
axiom not_only_method : IndependenceTest → Prop

-- Define the statement we want to prove false
def conclusions_always_correct (test : IndependenceTest) : Prop :=
  ∀ (sample : Type), true

-- Theorem statement
theorem independence_test_conclusions_not_always_correct :
  ∃ (test : IndependenceTest),
    small_probability_principle test ∧
    conclusions_vary_with_samples test ∧
    not_only_method test ∧
    ¬(conclusions_always_correct test) :=
by
  sorry

end independence_test_conclusions_not_always_correct_l761_76135


namespace unique_positive_solution_l761_76141

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^10 + 7*x^9 + 14*x^8 + 1729*x^7 - 1379*x^6 = 0 :=
by sorry

end unique_positive_solution_l761_76141


namespace decreasing_function_l761_76111

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem statement
theorem decreasing_function : 
  (∀ x : ℝ, HasDerivAt f4 (-2) x) ∧ 
  (∀ x : ℝ, (HasDerivAt f1 (2*x) x) ∨ (HasDerivAt f2 (-2*x) x) ∨ (HasDerivAt f3 2 x)) :=
by sorry

end decreasing_function_l761_76111


namespace sum_not_prime_l761_76193

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : 
  ¬(Nat.Prime (a + b + c + d)) :=
by sorry

end sum_not_prime_l761_76193


namespace pizza_toppings_l761_76172

/-- Represents a pizza with a given number of slices and topping distribution -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  both_toppings : ℕ
  pepperoni_only : ℕ
  mushroom_only : ℕ
  h_total : total_slices = pepperoni_only + mushroom_only + both_toppings
  h_pepperoni : pepperoni_slices = pepperoni_only + both_toppings
  h_mushroom : mushroom_slices = mushroom_only + both_toppings

/-- Theorem stating that a pizza with the given conditions has 2 slices with both toppings -/
theorem pizza_toppings (p : Pizza) 
  (h_total : p.total_slices = 18)
  (h_pep : p.pepperoni_slices = 10)
  (h_mush : p.mushroom_slices = 10) :
  p.both_toppings = 2 := by
  sorry

end pizza_toppings_l761_76172


namespace cosine_equality_l761_76152

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (830 * π / 180) → n = 70 := by
  sorry

end cosine_equality_l761_76152


namespace linda_current_age_l761_76133

/-- Represents the ages of Sarah, Jake, and Linda -/
structure Ages where
  sarah : ℚ
  jake : ℚ
  linda : ℚ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 11
  (ages.sarah + ages.jake + ages.linda) / 3 = 11 ∧
  -- Five years ago, Linda was the same age as Sarah is now
  ages.linda - 5 = ages.sarah ∧
  -- In 4 years, Jake's age will be 3/4 of Sarah's age at that time
  ages.jake + 4 = 3 / 4 * (ages.sarah + 4)

/-- The theorem stating Linda's current age -/
theorem linda_current_age (ages : Ages) (h : age_conditions ages) : 
  ages.linda = 14 := by
  sorry

end linda_current_age_l761_76133


namespace wendy_trip_miles_l761_76155

def three_day_trip (day1_miles day2_miles total_miles : ℕ) : Prop :=
  ∃ day3_miles : ℕ, day1_miles + day2_miles + day3_miles = total_miles

theorem wendy_trip_miles :
  three_day_trip 125 223 493 →
  ∃ day3_miles : ℕ, day3_miles = 145 :=
by sorry

end wendy_trip_miles_l761_76155
