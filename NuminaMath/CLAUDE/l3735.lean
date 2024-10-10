import Mathlib

namespace max_value_theorem_l3735_373522

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0) :
  (∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 ∧
    z'/(x'*y') ≤ z/(x*y) ∧
    x + 2*y - z ≤ x' + 2*y' - z') ∧
  (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
    z'/(x'*y') ≤ z/(x*y) →
    x' + 2*y' - z' ≤ 2) :=
by sorry

end max_value_theorem_l3735_373522


namespace candy_bar_profit_l3735_373530

theorem candy_bar_profit :
  let total_bars : ℕ := 1200
  let buy_price : ℚ := 3 / 4
  let sell_price : ℚ := 2 / 3
  let discount_threshold : ℕ := 1000
  let discount_per_bar : ℚ := 1 / 10
  let cost : ℚ := total_bars * buy_price
  let revenue_before_discount : ℚ := total_bars * sell_price
  let discounted_bars : ℕ := total_bars - discount_threshold
  let discount : ℚ := discounted_bars * discount_per_bar
  let revenue_after_discount : ℚ := revenue_before_discount - discount
  let profit : ℚ := revenue_after_discount - cost
  profit = -116
:= by sorry

end candy_bar_profit_l3735_373530


namespace no_function_exists_for_part_a_function_exists_for_part_b_l3735_373548

-- Part a
theorem no_function_exists_for_part_a :
  ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1 :=
sorry

-- Part b
theorem function_exists_for_part_b :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = 2 * n :=
sorry

end no_function_exists_for_part_a_function_exists_for_part_b_l3735_373548


namespace paths_count_is_40_l3735_373518

/-- Represents the arrangement of letters and numerals --/
structure Arrangement where
  centralA : Unit
  adjacentM : Fin 4
  adjacentC : Fin 4 → Fin 3
  adjacent1 : Unit
  adjacent0 : Fin 2

/-- Counts the number of paths to spell AMC10 in the given arrangement --/
def countPaths (arr : Arrangement) : ℕ :=
  let pathsFromM (m : Fin 4) := arr.adjacentC m * 1 * 2
  (pathsFromM 0 + pathsFromM 1 + pathsFromM 2 + pathsFromM 3)

/-- The theorem stating that the number of paths is 40 --/
theorem paths_count_is_40 (arr : Arrangement) : countPaths arr = 40 := by
  sorry

#check paths_count_is_40

end paths_count_is_40_l3735_373518


namespace martha_lasagna_cost_l3735_373599

/-- The cost of ingredients for Martha's lasagna -/
def lasagna_cost (cheese_quantity : Real) (cheese_price : Real) 
                 (meat_quantity : Real) (meat_price : Real) : Real :=
  cheese_quantity * cheese_price + meat_quantity * meat_price

/-- Theorem: The cost of ingredients for Martha's lasagna is $13 -/
theorem martha_lasagna_cost : 
  lasagna_cost 1.5 6 0.5 8 = 13 := by
  sorry

end martha_lasagna_cost_l3735_373599


namespace fraction_equality_l3735_373576

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by sorry

end fraction_equality_l3735_373576


namespace quadratic_factorization_sum_l3735_373571

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 15*x + 54 = (x + a) * (x + b)) →
  (∀ x, x^2 - 17*x + 72 = (x - b) * (x - c)) →
  a + b + c = 23 := by
sorry

end quadratic_factorization_sum_l3735_373571


namespace mr_callen_loss_l3735_373596

def paintings_count : ℕ := 15
def paintings_price : ℚ := 60
def wooden_toys_count : ℕ := 12
def wooden_toys_price : ℚ := 25
def hats_count : ℕ := 20
def hats_price : ℚ := 15

def paintings_loss_percentage : ℚ := 18 / 100
def wooden_toys_loss_percentage : ℚ := 25 / 100
def hats_loss_percentage : ℚ := 10 / 100

def total_cost : ℚ := 
  paintings_count * paintings_price + 
  wooden_toys_count * wooden_toys_price + 
  hats_count * hats_price

def total_selling_price : ℚ := 
  paintings_count * paintings_price * (1 - paintings_loss_percentage) +
  wooden_toys_count * wooden_toys_price * (1 - wooden_toys_loss_percentage) +
  hats_count * hats_price * (1 - hats_loss_percentage)

def total_loss : ℚ := total_cost - total_selling_price

theorem mr_callen_loss : total_loss = 267 := by
  sorry

end mr_callen_loss_l3735_373596


namespace root_sum_reciprocal_products_l3735_373586

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  p^4 + 8*p^3 + 16*p^2 + 5*p + 2 = 0 →
  q^4 + 8*q^3 + 16*q^2 + 5*q + 2 = 0 →
  r^4 + 8*r^3 + 16*r^2 + 5*r + 2 = 0 →
  s^4 + 8*s^3 + 16*s^2 + 5*s + 2 = 0 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 8 := by
sorry

end root_sum_reciprocal_products_l3735_373586


namespace identity_function_satisfies_conditions_l3735_373533

theorem identity_function_satisfies_conditions (f : ℕ → ℕ) 
  (h1 : ∀ n, f (2 * n) = 2 * f n) 
  (h2 : ∀ n, f (2 * n + 1) = 2 * f n + 1) : 
  ∀ n, f n = n := by sorry

end identity_function_satisfies_conditions_l3735_373533


namespace square_diagonal_l3735_373545

theorem square_diagonal (A : ℝ) (h : A = 800) :
  ∃ d : ℝ, d = 40 ∧ d^2 = 2 * A :=
by sorry

end square_diagonal_l3735_373545


namespace prime_squared_plus_two_l3735_373581

theorem prime_squared_plus_two (p : ℕ) : 
  Nat.Prime p → (Nat.Prime (p^2 + 2) ↔ p = 3) := by sorry

end prime_squared_plus_two_l3735_373581


namespace four_digit_harmonious_divisible_by_11_l3735_373550

/-- A four-digit harmonious number with 'a' as the first and last digit, and 'b' as the second and third digit. -/
def four_digit_harmonious (a b : ℕ) : ℕ := 1000 * a + 100 * b + 10 * b + a

/-- Proposition: All four-digit harmonious numbers are divisible by 11. -/
theorem four_digit_harmonious_divisible_by_11 (a b : ℕ) :
  ∃ k : ℕ, four_digit_harmonious a b = 11 * k := by
  sorry

end four_digit_harmonious_divisible_by_11_l3735_373550


namespace smallest_n_for_good_sequence_2014_l3735_373582

/-- A sequence of real numbers is good if it satisfies certain conditions. -/
def IsGoodSequence (a : ℕ → ℝ) : Prop :=
  (∃ k : ℕ+, a k = 2014) ∧
  (∃ n : ℕ+, a 0 = n) ∧
  ∀ i : ℕ, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

/-- The smallest positive integer n such that there exists a good sequence with aₙ = 2014 is 60. -/
theorem smallest_n_for_good_sequence_2014 :
  ∃ (a : ℕ → ℝ), IsGoodSequence a ∧ a 60 = 2014 ∧
  ∀ (b : ℕ → ℝ) (m : ℕ), m < 60 → IsGoodSequence b → b m ≠ 2014 := by
  sorry

#check smallest_n_for_good_sequence_2014

end smallest_n_for_good_sequence_2014_l3735_373582


namespace complex_power_of_one_plus_i_l3735_373510

theorem complex_power_of_one_plus_i : (1 + Complex.I) ^ 6 = -8 * Complex.I := by
  sorry

end complex_power_of_one_plus_i_l3735_373510


namespace correct_factorization_l3735_373546

theorem correct_factorization (x y : ℝ) : x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end correct_factorization_l3735_373546


namespace inequality_proof_l3735_373568

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (a * c / (b + a * c)) ≤ 3 / 2 := by
  sorry

end inequality_proof_l3735_373568


namespace starting_number_proof_l3735_373558

theorem starting_number_proof (x : ℕ) : 
  (∃ (l : List ℕ), l.length = 12 ∧ 
    (∀ n ∈ l, x ≤ n ∧ n ≤ 47 ∧ n % 3 = 0) ∧
    (∀ m, x ≤ m ∧ m ≤ 47 ∧ m % 3 = 0 → m ∈ l)) ↔ 
  x = 12 := by
  sorry

#check starting_number_proof

end starting_number_proof_l3735_373558


namespace people_not_playing_sports_l3735_373525

theorem people_not_playing_sports (total_people : ℕ) (tennis_players : ℕ) (baseball_players : ℕ) (both_players : ℕ) :
  total_people = 310 →
  tennis_players = 138 →
  baseball_players = 255 →
  both_players = 94 →
  total_people - (tennis_players + baseball_players - both_players) = 11 :=
by sorry

end people_not_playing_sports_l3735_373525


namespace diophantine_equation_solutions_l3735_373500

theorem diophantine_equation_solutions :
  (∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (m n p : ℕ), (m, n, p) ∈ S ↔ 4 * m * n - m - n = p^2 - 1) ∧ 
    Set.Infinite S) ∧
  (∀ (m n p : ℕ), 4 * m * n - m - n ≠ p^2) := by
  sorry

end diophantine_equation_solutions_l3735_373500


namespace z_in_third_quadrant_l3735_373592

/-- Given that i is the imaginary unit and i · z = 1 - 2i, 
    prove that z is located in the third quadrant of the complex plane. -/
theorem z_in_third_quadrant (i z : ℂ) : 
  i * i = -1 →  -- i is the imaginary unit
  i * z = 1 - 2*i →  -- given equation
  z.re < 0 ∧ z.im < 0  -- z is in the third quadrant
  := by sorry

end z_in_third_quadrant_l3735_373592


namespace range_of_a_l3735_373573

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  a < 0 ∧
  (∀ x, p x a → q x) ∧
  (∃ x, q x ∧ ¬p x a) →
  -2/3 ≤ a ∧ a < 0 :=
sorry

end range_of_a_l3735_373573


namespace susan_spending_l3735_373536

def carnival_spending (initial_amount food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let total_spent := food_cost + ride_cost
  initial_amount - total_spent

theorem susan_spending :
  carnival_spending 50 12 = 14 := by
  sorry

end susan_spending_l3735_373536


namespace complex_pure_imaginary_l3735_373579

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I * (Complex.I * (a + 1) - 2 * a) / 5 = (Complex.I * (a + Complex.I) / (1 + 2 * Complex.I))) → 
  a = -2 := by
sorry

end complex_pure_imaginary_l3735_373579


namespace boxes_in_carton_l3735_373538

/-- The number of boxes in a carton -/
def boxes_per_carton : ℕ := sorry

/-- The number of packs of cheese cookies in each box -/
def packs_per_box : ℕ := 10

/-- The price of a pack of cheese cookies in dollars -/
def price_per_pack : ℕ := 1

/-- The cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- Theorem stating the number of boxes in a carton -/
theorem boxes_in_carton : boxes_per_carton = 12 := by sorry

end boxes_in_carton_l3735_373538


namespace increase_by_percentage_increase_1500_by_20_percent_l3735_373583

theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) : 
  initial + (initial * percentage) = initial * (1 + percentage) := by sorry

theorem increase_1500_by_20_percent : 
  1500 + (1500 * (20 / 100)) = 1800 := by sorry

end increase_by_percentage_increase_1500_by_20_percent_l3735_373583


namespace two_digit_prime_sum_20180500_prime_l3735_373563

theorem two_digit_prime_sum_20180500_prime (n : ℕ) : 
  (10 ≤ n ∧ n < 100) →  -- n is a two-digit number
  Nat.Prime n →         -- n is prime
  Nat.Prime (n + 20180500) → -- n + 20180500 is prime
  n = 61 := by
sorry

end two_digit_prime_sum_20180500_prime_l3735_373563


namespace intersection_when_a_is_one_union_equals_A_iff_l3735_373569

def A : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 0 ≤ x ∧ x ≤ 3} := by sorry

theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

end intersection_when_a_is_one_union_equals_A_iff_l3735_373569


namespace cos_750_degrees_l3735_373501

theorem cos_750_degrees : Real.cos (750 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_750_degrees_l3735_373501


namespace inequality_preservation_l3735_373561

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end inequality_preservation_l3735_373561


namespace not_perfect_power_l3735_373593

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ¬ ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ 10^k - 1 = m^n := by
  sorry

end not_perfect_power_l3735_373593


namespace parallel_lines_m_value_l3735_373553

/-- Two lines are parallel if their slopes are equal (when they exist) -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ (a₁ / b₁ = a₂ / b₂)

theorem parallel_lines_m_value :
  ∀ m : ℝ,
  parallel 3 (m + 1) (-(m - 7)) m 2 (-3 * m) →
  m = -3 :=
by sorry

end parallel_lines_m_value_l3735_373553


namespace expression_evaluation_l3735_373572

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  ((x + 3) / (x + 2)) * ((y - 1) / (y - 3)) * ((z + 9) / (z + 7)) = 2.4 := by
  sorry

end expression_evaluation_l3735_373572


namespace triangle_existence_l3735_373588

/-- Given an angle and two segments representing differences between sides,
    prove the existence of a triangle with these properties. -/
theorem triangle_existence (A : Real) (d e : ℝ) : ∃ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- positive side lengths
  a - c = d ∧              -- given difference d
  b - c = e ∧              -- given difference e
  0 < A ∧ A < π ∧          -- valid angle measure
  -- The angle A is the smallest in the triangle
  A ≤ Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧
  A ≤ Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) :=
by sorry


end triangle_existence_l3735_373588


namespace factorial_equality_l3735_373591

theorem factorial_equality : 7 * 6 * 4 * 2160 = Nat.factorial 9 := by
  sorry

end factorial_equality_l3735_373591


namespace roberts_markers_count_l3735_373594

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := 217

/-- The total number of markers Megan has now -/
def total_markers : ℕ := 326

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := total_markers - initial_markers

theorem roberts_markers_count : roberts_markers = 109 := by
  sorry

end roberts_markers_count_l3735_373594


namespace paperclip_capacity_l3735_373515

/-- Given a box of volume 16 cm³ that holds 50 paperclips, 
    prove that a box of volume 48 cm³ will hold 150 paperclips, 
    assuming a direct proportion between volume and paperclip capacity. -/
theorem paperclip_capacity (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) :
  v₁ = 16 → v₂ = 48 → c₁ = 50 →
  (v₁ * c₂ = v₂ * c₁) →
  c₂ = 150 := by
  sorry

#check paperclip_capacity

end paperclip_capacity_l3735_373515


namespace burger_cost_is_five_l3735_373507

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℝ := 4

/-- The cost of a smoothie in dollars -/
def smoothie_cost : ℝ := 4

/-- The number of smoothies ordered -/
def num_smoothies : ℕ := 2

/-- The total cost of the order in dollars -/
def total_cost : ℝ := 17

/-- The cost of the burger in dollars -/
def burger_cost : ℝ := total_cost - (sandwich_cost + num_smoothies * smoothie_cost)

theorem burger_cost_is_five :
  burger_cost = 5 := by sorry

end burger_cost_is_five_l3735_373507


namespace parallelogram_area_l3735_373508

/-- The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 24 → height = 16 → area = base * height → area = 384 := by
sorry

end parallelogram_area_l3735_373508


namespace abs_plus_one_nonzero_l3735_373598

theorem abs_plus_one_nonzero (a : ℚ) : |a| + 1 ≠ 0 := by
  sorry

end abs_plus_one_nonzero_l3735_373598


namespace solution_set_of_inequality_range_of_m_l3735_373549

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |1 - x|

-- Theorem for the first part of the problem
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ 3*x + 4 ↔ x ≥ -1/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ (m^2 - 3*m + 3) * |x|) ↔ 1 ≤ m ∧ m ≤ 2 :=
sorry

end solution_set_of_inequality_range_of_m_l3735_373549


namespace line_equation_proof_l3735_373578

/-- Given a line defined by (-3, 4) · ((x, y) - (2, -6)) = 0, 
    prove that its slope-intercept form is y = (3/4)x - 7.5 
    and consequently, m = 3/4 and b = -7.5 -/
theorem line_equation_proof (x y : ℝ) : 
  (-3 : ℝ) * (x - 2) + 4 * (y + 6) = 0 → 
  y = (3/4 : ℝ) * x - (15/2 : ℝ) ∧ 
  (3/4 : ℝ) = (3/4 : ℝ) ∧ 
  -(15/2 : ℝ) = -(15/2 : ℝ) := by
  sorry

end line_equation_proof_l3735_373578


namespace integer_remainder_properties_l3735_373552

theorem integer_remainder_properties (n : ℤ) (h : n % 20 = 13) :
  (n % 4 + n % 5 = 4) ∧ n % 2 = 1 := by
  sorry

end integer_remainder_properties_l3735_373552


namespace banana_cost_theorem_l3735_373502

/-- The cost of bananas given a specific rate and quantity -/
def banana_cost (rate_price : ℚ) (rate_quantity : ℚ) (buy_quantity : ℚ) : ℚ :=
  (rate_price / rate_quantity) * buy_quantity

/-- Theorem stating that 20 pounds of bananas cost $30 given the rate of $6 for 4 pounds -/
theorem banana_cost_theorem :
  banana_cost 6 4 20 = 30 := by
  sorry

end banana_cost_theorem_l3735_373502


namespace race_head_start_l3735_373506

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (51 / 44) * Vb) :
  let H := L * (7 / 51)
  L / Va = (L - H) / Vb := by sorry

end race_head_start_l3735_373506


namespace smallest_number_with_rearranged_double_l3735_373529

def digits_to_num (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

def num_to_digits (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

def rearrange_digits (digits : List Nat) : List Nat :=
  (digits.take 2).reverse ++ (digits.drop 2).reverse

theorem smallest_number_with_rearranged_double :
  ∃ (n : Nat),
    n = 263157894736842105 ∧
    (∀ m : Nat, m < n →
      let digits_m := num_to_digits m
      let r_m := digits_to_num (rearrange_digits digits_m)
      r_m ≠ 2 * m) ∧
    let digits_n := num_to_digits n
    let r_n := digits_to_num (rearrange_digits digits_n)
    r_n = 2 * n :=
by sorry

end smallest_number_with_rearranged_double_l3735_373529


namespace christine_siri_money_difference_l3735_373539

theorem christine_siri_money_difference :
  ∀ (christine_amount siri_amount : ℝ),
    christine_amount + siri_amount = 21 →
    christine_amount = 20.5 →
    christine_amount - siri_amount = 20 :=
by
  sorry

end christine_siri_money_difference_l3735_373539


namespace quarterback_passes_l3735_373531

theorem quarterback_passes (total passes_left passes_right passes_center : ℕ) : 
  total = 50 → 
  passes_left = 12 → 
  passes_right = 2 * passes_left → 
  total = passes_left + passes_right + passes_center → 
  passes_center - passes_left = 2 := by
  sorry

end quarterback_passes_l3735_373531


namespace simplify_sqrt_expression_l3735_373584

theorem simplify_sqrt_expression :
  (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 147 / Real.sqrt 63) = (42 - 7 * Real.sqrt 21) / 21 := by
  sorry

end simplify_sqrt_expression_l3735_373584


namespace square_circle_area_ratio_l3735_373575

theorem square_circle_area_ratio :
  ∀ (r : ℝ) (s₁ s₂ : ℝ),
  r > 0 → s₁ > 0 → s₂ > 0 →
  2 * π * r = 4 * s₁ →  -- Circle and first square have same perimeter
  2 * r = s₂ * Real.sqrt 2 →  -- Diameter of circle is diagonal of second square
  (s₂^2) / (s₁^2) = 8 :=
by sorry

end square_circle_area_ratio_l3735_373575


namespace infinite_special_integers_l3735_373567

theorem infinite_special_integers :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, (n ∣ 2^(2^n + 1) + 1) ∧ ¬(n ∣ 2^n + 1) := by
  sorry

end infinite_special_integers_l3735_373567


namespace last_segment_speed_l3735_373580

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (first_segment_speed : ℝ) (second_segment_speed : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_segment_speed = 50 →
  second_segment_speed = 70 →
  ∃ (last_segment_speed : ℝ),
    last_segment_speed = 60 ∧
    (first_segment_speed * (total_time / 3) + 
     second_segment_speed * (total_time / 3) + 
     last_segment_speed * (total_time / 3)) = total_distance :=
by sorry

end last_segment_speed_l3735_373580


namespace village_population_l3735_373554

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.8 = 4554 → P = 6325 := by
  sorry

end village_population_l3735_373554


namespace largest_multiple_of_15_under_500_l3735_373587

theorem largest_multiple_of_15_under_500 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 ∧ 5 ∣ n → n ≤ 495 :=
by sorry

end largest_multiple_of_15_under_500_l3735_373587


namespace perimeter_after_adding_tiles_l3735_373528

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { tiles := initial.tiles + added, perimeter := initial.perimeter + 1 }

theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration) 
  (h1 : initial.tiles = 9)
  (h2 : initial.perimeter = 16) :
  ∃ (final : TileConfiguration), 
    final = add_tiles initial 3 ∧ 
    final.perimeter = 17 :=
sorry

end perimeter_after_adding_tiles_l3735_373528


namespace solve_exponent_equation_l3735_373543

theorem solve_exponent_equation (n : ℕ) : 2 * 2^2 * 2^n = 2^10 → n = 7 := by
  sorry

end solve_exponent_equation_l3735_373543


namespace binary_101101110_equals_octal_556_l3735_373555

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation as a list of digits. -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

theorem binary_101101110_equals_octal_556 :
  let binary : List Bool := [true, false, true, true, false, true, true, true, false]
  let octal : List ℕ := [6, 5, 5]
  binary_to_natural binary = (natural_to_octal (binary_to_natural binary)).reverse.foldl (fun acc d => acc * 8 + d) 0 ∧
  natural_to_octal (binary_to_natural binary) = octal.reverse :=
by sorry

end binary_101101110_equals_octal_556_l3735_373555


namespace diophantine_equation_solutions_l3735_373534

theorem diophantine_equation_solutions :
  {(a, b) : ℕ × ℕ | 12 * a + 11 * b = 2002} =
    {(11, 170), (22, 158), (33, 146), (44, 134), (55, 122), (66, 110),
     (77, 98), (88, 86), (99, 74), (110, 62), (121, 50), (132, 38),
     (143, 26), (154, 14), (165, 2)} := by
  sorry

end diophantine_equation_solutions_l3735_373534


namespace congruence_solution_l3735_373557

theorem congruence_solution (n : ℤ) : 
  (13 * n) % 47 = 9 % 47 ↔ n % 47 = 39 % 47 := by
  sorry

end congruence_solution_l3735_373557


namespace min_cost_for_nine_hamburgers_l3735_373513

/-- Represents the cost of hamburgers under a "buy two, get one free" promotion -/
def hamburger_cost (unit_price : ℕ) (quantity : ℕ) : ℕ :=
  let sets := quantity / 3
  let remainder := quantity % 3
  sets * (2 * unit_price) + remainder * unit_price

/-- Theorem stating the minimum cost for 9 hamburgers under the given promotion -/
theorem min_cost_for_nine_hamburgers :
  hamburger_cost 10 9 = 60 := by
  sorry

end min_cost_for_nine_hamburgers_l3735_373513


namespace sum_of_squares_of_roots_l3735_373560

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (12 * x₁^2 + 16 * x₁ - 21 = 0) → 
  (12 * x₂^2 + 16 * x₂ - 21 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 95/18) :=
by sorry

end sum_of_squares_of_roots_l3735_373560


namespace square_difference_one_l3735_373514

theorem square_difference_one : (726 : ℕ) * 726 - 725 * 727 = 1 := by
  sorry

end square_difference_one_l3735_373514


namespace min_buses_for_field_trip_l3735_373547

def min_buses (total_students : ℕ) (bus_capacity_1 : ℕ) (bus_capacity_2 : ℕ) : ℕ :=
  let large_buses := total_students / bus_capacity_1
  let remaining_students := total_students % bus_capacity_1
  if remaining_students = 0 then
    large_buses
  else
    large_buses + 1

theorem min_buses_for_field_trip :
  min_buses 530 45 40 = 12 := by
  sorry

end min_buses_for_field_trip_l3735_373547


namespace fort_block_count_l3735_373512

/-- Calculates the number of one-foot cubical blocks required to construct a rectangular fort -/
def fort_blocks (length width height : ℕ) (wall_thickness : ℕ) : ℕ :=
  length * width * height - 
  (length - 2 * wall_thickness) * (width - 2 * wall_thickness) * (height - wall_thickness)

/-- Proves that a fort with given dimensions requires 430 blocks -/
theorem fort_block_count : fort_blocks 15 12 6 1 = 430 := by
  sorry

end fort_block_count_l3735_373512


namespace quadrilaterals_from_nine_points_l3735_373559

theorem quadrilaterals_from_nine_points : ∀ n : ℕ, n = 9 →
  (Nat.choose n 4 : ℕ) = 126 := by sorry

end quadrilaterals_from_nine_points_l3735_373559


namespace inscribed_square_area_l3735_373519

/-- The area of a square inscribed in the ellipse x^2/4 + y^2/8 = 1, 
    with sides parallel to the coordinate axes. -/
theorem inscribed_square_area : 
  ∃ (s : ℝ), s > 0 ∧ 
  (s^2 / 4 + s^2 / 8 = 1) ∧ 
  (4 * s^2 = 32 / 3) := by sorry

end inscribed_square_area_l3735_373519


namespace quadratic_inequality_solution_l3735_373540

theorem quadratic_inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := by
  sorry

end quadratic_inequality_solution_l3735_373540


namespace base_eight_to_ten_l3735_373505

theorem base_eight_to_ten : 
  (1 * 8^3 + 7 * 8^2 + 2 * 8^1 + 4 * 8^0 : ℕ) = 980 := by
  sorry

end base_eight_to_ten_l3735_373505


namespace spinner_prob_C_or_D_l3735_373597

/-- Represents a circular spinner with four parts -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The probability of landing on either C or D -/
def probCorD (s : Spinner) : ℚ := s.probC + s.probD

theorem spinner_prob_C_or_D (s : Spinner) 
  (h1 : s.probA = 1/4)
  (h2 : s.probB = 1/3)
  (h3 : s.probA + s.probB + s.probC + s.probD = 1) :
  probCorD s = 5/12 := by
    sorry

end spinner_prob_C_or_D_l3735_373597


namespace sum_due_calculation_l3735_373537

/-- The relationship between banker's discount, true discount, and sum due -/
def banker_discount_relation (bd td sd : ℝ) : Prop :=
  bd = td + td^2 / sd

/-- The problem statement -/
theorem sum_due_calculation (bd td : ℝ) (h1 : bd = 36) (h2 : td = 30) :
  ∃ sd : ℝ, banker_discount_relation bd td sd ∧ sd = 150 := by
  sorry

end sum_due_calculation_l3735_373537


namespace fraction_equivalence_l3735_373520

theorem fraction_equivalence (a b c x : ℝ) 
  (h1 : x = a / b + 2)
  (h2 : a ≠ b)
  (h3 : b ≠ 0) :
  (a + 2*b) / (a - 2*b) = x / (x - 4) := by
  sorry

end fraction_equivalence_l3735_373520


namespace polygon_side_length_theorem_l3735_373556

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon

/-- Represents a way to divide a polygon into equilateral triangles and squares. -/
structure Division where
  -- Add necessary fields for a division

/-- Counts the number of ways to divide a polygon into equilateral triangles and squares. -/
def countDivisions (M : ConvexPolygon) : ℕ :=
  sorry

/-- Checks if a number is prime. -/
def isPrime (n : ℕ) : Prop :=
  sorry

/-- Gets the length of a side of a polygon. -/
def sideLength (M : ConvexPolygon) (side : ℕ) : ℕ :=
  sorry

theorem polygon_side_length_theorem (M : ConvexPolygon) (p : ℕ) :
  isPrime p → countDivisions M = p → ∃ side, sideLength M side = p - 1 :=
by sorry

end polygon_side_length_theorem_l3735_373556


namespace least_m_for_no_real_roots_l3735_373503

theorem least_m_for_no_real_roots : 
  ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ (k : ℤ), k < m → ∃ (x : ℝ), 3 * x * (k * x + 6) - 2 * x^2 + 8 = 0) ∧
  m = 4 := by
  sorry

end least_m_for_no_real_roots_l3735_373503


namespace star_sum_squared_l3735_373564

/-- The star operation defined on real numbers -/
def star (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the result of (x+y)² ⋆ (y+x)² -/
theorem star_sum_squared (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end star_sum_squared_l3735_373564


namespace pyramid_solution_l3735_373504

/-- Represents a pyramid of numbers --/
structure Pyramid :=
  (bottom_row : List ℝ)
  (is_valid : bottom_row.length = 4)

/-- Checks if a pyramid satisfies the given conditions --/
def satisfies_conditions (p : Pyramid) : Prop :=
  ∃ x : ℝ,
    p.bottom_row = [13, x, 11, 2*x] ∧
    (13 + x) + (11 + 2*x) = 42

/-- The main theorem to prove --/
theorem pyramid_solution {p : Pyramid} (h : satisfies_conditions p) :
  ∃ x : ℝ, x = 6 ∧ p.bottom_row = [13, x, 11, 2*x] := by
  sorry

end pyramid_solution_l3735_373504


namespace calculate_Y_l3735_373570

theorem calculate_Y : ∀ P Q Y : ℚ,
  P = 4050 / 5 →
  Q = P / 4 →
  Y = P - Q →
  Y = 607.5 := by
sorry

end calculate_Y_l3735_373570


namespace unique_element_condition_l3735_373527

def M (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

theorem unique_element_condition (a : ℝ) : 
  (∃! x, x ∈ M a) ↔ (a = 0 ∨ a = 1) := by
sorry

end unique_element_condition_l3735_373527


namespace tree_distance_l3735_373562

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 10) (h2 : d = 100) :
  let tree_spacing := d / 5
  tree_spacing * (n - 1) = 180 := by
  sorry

end tree_distance_l3735_373562


namespace smallest_four_digit_divisible_by_first_four_primes_l3735_373521

theorem smallest_four_digit_divisible_by_first_four_primes : ∃ n : ℕ,
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧
  n = 1050 :=
by sorry

end smallest_four_digit_divisible_by_first_four_primes_l3735_373521


namespace box_volume_solutions_l3735_373565

def box_volume (x : ℤ) : ℤ :=
  (x + 3) * (x - 3) * (x^3 - 5*x + 25)

def satisfies_condition (x : ℤ) : Prop :=
  x > 0 ∧ box_volume x < 1500

theorem box_volume_solutions :
  (∃ (S : Finset ℤ), (∀ x ∈ S, satisfies_condition x) ∧
                     (∀ x : ℤ, satisfies_condition x → x ∈ S) ∧
                     Finset.card S = 4) := by
  sorry

end box_volume_solutions_l3735_373565


namespace sequence_even_terms_l3735_373589

def sequence_property (x : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ d : ℕ, d > 0 ∧ d < 10 ∧ d ∣ x (n-1) ∧ x n = x (n-1) + d

theorem sequence_even_terms (x : ℕ → ℕ) (h : sequence_property x) :
  (∃ n : ℕ, Even (x n)) ∧ (∀ m : ℕ, ∃ n : ℕ, n > m ∧ Even (x n)) :=
sorry

end sequence_even_terms_l3735_373589


namespace arithmetic_sequence_common_difference_l3735_373585

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  product_condition : a 7 * a 11 = 6
  sum_condition : a 4 + a 14 = 5

/-- The common difference of an arithmetic sequence is either 1/4 or -1/4 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  (∃ d : ℚ, (∀ n : ℕ, seq.a (n + 1) - seq.a n = d) ∧ (d = 1/4 ∨ d = -1/4)) := by
  sorry

end arithmetic_sequence_common_difference_l3735_373585


namespace cos_36_minus_cos_72_eq_half_l3735_373511

theorem cos_36_minus_cos_72_eq_half :
  Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end cos_36_minus_cos_72_eq_half_l3735_373511


namespace dilation_rotation_composition_l3735_373524

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

theorem dilation_rotation_composition :
  rotation_matrix * dilation_matrix = !![0, 2; -2, 0] := by sorry

end dilation_rotation_composition_l3735_373524


namespace first_chapter_pages_l3735_373509

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter2_pages : ℕ

/-- The number of pages in the first chapter of a book -/
def pages_in_chapter1 (b : Book) : ℕ := b.total_pages - b.chapter2_pages

/-- Theorem: For a book with 81 total pages and 68 pages in the second chapter,
    the first chapter has 13 pages -/
theorem first_chapter_pages :
  ∀ (b : Book), b.total_pages = 81 → b.chapter2_pages = 68 →
  pages_in_chapter1 b = 13 := by
  sorry

end first_chapter_pages_l3735_373509


namespace book_price_difference_l3735_373532

def necklace_price : ℕ := 34
def spending_limit : ℕ := 70
def overspent_amount : ℕ := 3

theorem book_price_difference (book_price : ℕ) : 
  book_price > necklace_price →
  book_price + necklace_price = spending_limit + overspent_amount →
  book_price - necklace_price = 5 := by
sorry

end book_price_difference_l3735_373532


namespace watermelon_cost_proof_l3735_373541

/-- Represents the number of fruits a container can hold -/
def ContainerCapacity : ℕ := 150

/-- Represents the total value of fruits in rubles -/
def TotalValue : ℕ := 24000

/-- Represents the capacity of the container in terms of melons -/
def MelonCapacity : ℕ := 120

/-- Represents the capacity of the container in terms of watermelons -/
def WatermelonCapacity : ℕ := 160

/-- Represents the cost of a single watermelon in rubles -/
def WatermelonCost : ℕ := 100

theorem watermelon_cost_proof :
  ∃ (num_watermelons num_melons : ℕ),
    num_watermelons + num_melons = ContainerCapacity ∧
    num_watermelons * WatermelonCost = num_melons * (TotalValue / num_melons) ∧
    num_watermelons * WatermelonCost + num_melons * (TotalValue / num_melons) = TotalValue ∧
    num_watermelons * (1 / WatermelonCapacity) + num_melons * (1 / MelonCapacity) = 1 :=
by sorry

end watermelon_cost_proof_l3735_373541


namespace simplify_fraction_difference_quotient_l3735_373595

theorem simplify_fraction_difference_quotient (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (1 / (a + 2) - 1 / (a - 2)) / (1 / (a - 2)) = -4 / (a + 2) := by
  sorry

end simplify_fraction_difference_quotient_l3735_373595


namespace union_of_A_and_B_l3735_373590

def A : Set ℕ := {1, 3}
def B : Set ℕ := {0, 3}

theorem union_of_A_and_B : A ∪ B = {0, 1, 3} := by sorry

end union_of_A_and_B_l3735_373590


namespace arctan_tan_difference_l3735_373574

theorem arctan_tan_difference (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
  Real.arctan (Real.tan x - 2 * Real.tan y) = 25 * π / 180 :=
by sorry

end arctan_tan_difference_l3735_373574


namespace sqrt_two_sufficient_not_necessary_l3735_373544

/-- The line x + y = 0 is tangent to the circle x^2 + (y - a)^2 = 1 -/
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x + y = 0 ∧ x^2 + (y - a)^2 = 1 ∧
  ∀ (x' y' : ℝ), x' + y' = 0 → x'^2 + (y' - a)^2 ≥ 1

/-- a = √2 is a sufficient but not necessary condition for the line to be tangent to the circle -/
theorem sqrt_two_sufficient_not_necessary :
  (∀ a : ℝ, a = Real.sqrt 2 → is_tangent a) ∧
  ¬(∀ a : ℝ, is_tangent a → a = Real.sqrt 2) :=
by sorry

end sqrt_two_sufficient_not_necessary_l3735_373544


namespace ball_placement_theorem_l3735_373542

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process described in the problem -/
def ballPlacementProcess (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of balls after n steps is equal to
    the sum of digits in the base 6 representation of n -/
theorem ball_placement_theorem (n : ℕ) :
  ballPlacementProcess n = sumDigits (toBase6 n) :=
  sorry

end ball_placement_theorem_l3735_373542


namespace valid_solutions_l3735_373517

/-- Defines a function that checks if a triple of digits forms a valid solution --/
def is_valid_solution (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  ∃ (k : ℤ), k * (10*a + b + 10*b + c + 10*c + a) = 100*a + 10*b + c + a + b + c

/-- The main theorem stating the valid solutions --/
theorem valid_solutions :
  ∀ a b c : ℕ,
    is_valid_solution a b c ↔
      (a = 5 ∧ b = 1 ∧ c = 6) ∨
      (a = 9 ∧ b = 1 ∧ c = 2) ∨
      (a = 6 ∧ b = 4 ∧ c = 5) ∨
      (a = 3 ∧ b = 7 ∧ c = 8) ∨
      (a = 5 ∧ b = 7 ∧ c = 6) ∨
      (a = 7 ∧ b = 7 ∧ c = 4) ∨
      (a = 9 ∧ b = 7 ∧ c = 2) :=
by sorry

end valid_solutions_l3735_373517


namespace dress_cost_difference_l3735_373516

theorem dress_cost_difference (patty ida jean pauline : ℕ) : 
  patty = ida + 10 →
  ida = jean + 30 →
  jean < pauline →
  pauline = 30 →
  patty + ida + jean + pauline = 160 →
  pauline - jean = 10 := by
sorry

end dress_cost_difference_l3735_373516


namespace min_sum_reciprocals_l3735_373551

theorem min_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  (1 / x + 1 / y + 1 / z) ≥ 9 := by
  sorry

end min_sum_reciprocals_l3735_373551


namespace fixed_point_exponential_function_l3735_373526

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ (x y : ℝ), x = 2 ∧ y = 2 ∧ y = a^(x - 2) + 1 := by
  sorry

end fixed_point_exponential_function_l3735_373526


namespace range_of_m_l3735_373577

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines a line in the form 2x + y + m = 0 -/
def Line (m : ℝ) (p : Point) : Prop :=
  2 * p.x + p.y + m = 0

/-- Defines when two points are on opposite sides of a line -/
def OppositesSides (m : ℝ) (p1 p2 : Point) : Prop :=
  (2 * p1.x + p1.y + m) * (2 * p2.x + p2.y + m) < 0

/-- The main theorem -/
theorem range_of_m (p1 p2 : Point) (h : OppositesSides m p1 p2) 
  (h1 : p1 = ⟨1, 3⟩) (h2 : p2 = ⟨-4, -2⟩) : 
  -5 < m ∧ m < 10 := by
  sorry

end range_of_m_l3735_373577


namespace quadratic_minimum_l3735_373523

def f (x : ℝ) := 5 * x^2 - 15 * x + 2

theorem quadratic_minimum :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9.25 :=
sorry

end quadratic_minimum_l3735_373523


namespace circle_center_and_difference_l3735_373566

/-- 
Given a circle described by the equation x^2 + y^2 - 10x + 4y + 13 = 0,
prove that its center is (5, -2) and x - y = 7.
-/
theorem circle_center_and_difference (x y : ℝ) :
  x^2 + y^2 - 10*x + 4*y + 13 = 0 →
  (∃ (r : ℝ), (x - 5)^2 + (y + 2)^2 = r^2) ∧
  x - y = 7 := by
sorry

end circle_center_and_difference_l3735_373566


namespace janet_total_cost_l3735_373535

/-- Calculates the total cost for Janet's group at the waterpark -/
def waterpark_cost (adult_price : ℚ) (group_size : ℕ) (child_count : ℕ) (soda_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let adult_count := group_size - child_count
  let total_ticket_cost := adult_price * adult_count + child_price * child_count
  let discount := total_ticket_cost * (1/5)
  (total_ticket_cost - discount) + soda_price

/-- Proves that Janet's total cost is $197 -/
theorem janet_total_cost : 
  waterpark_cost 30 10 4 5 = 197 := by
  sorry

end janet_total_cost_l3735_373535
