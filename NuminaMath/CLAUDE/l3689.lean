import Mathlib

namespace set_A_equals_roster_l3689_368941

def A : Set ℤ := {x | ∃ (n : ℕ+), 6 / (5 - x) = n}

theorem set_A_equals_roster : A = {-1, 2, 3, 4} := by sorry

end set_A_equals_roster_l3689_368941


namespace f_max_min_on_interval_l3689_368913

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem f_max_min_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 5) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -15) :=
sorry

end f_max_min_on_interval_l3689_368913


namespace right_triangle_shorter_leg_l3689_368923

theorem right_triangle_shorter_leg : ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 39 →           -- Hypotenuse is 39 units
  a ≤ b →            -- a is the shorter leg
  a = 15 :=          -- The shorter leg is 15 units
by
  sorry

end right_triangle_shorter_leg_l3689_368923


namespace initial_adults_on_train_l3689_368921

theorem initial_adults_on_train (adults_children_diff : ℕ)
  (adults_boarding : ℕ) (children_boarding : ℕ) (people_leaving : ℕ) (final_count : ℕ)
  (h1 : adults_children_diff = 17)
  (h2 : adults_boarding = 57)
  (h3 : children_boarding = 18)
  (h4 : people_leaving = 44)
  (h5 : final_count = 502) :
  ∃ (initial_adults initial_children : ℕ),
    initial_adults = initial_children + adults_children_diff ∧
    initial_adults + initial_children + adults_boarding + children_boarding - people_leaving = final_count ∧
    initial_adults = 244 := by
  sorry

end initial_adults_on_train_l3689_368921


namespace g_composition_15_l3689_368903

def g (x : ℤ) : ℤ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_composition_15 : g (g (g (g 15))) = 3 := by
  sorry

end g_composition_15_l3689_368903


namespace black_larger_than_gray_l3689_368912

/-- The gray area of the rectangles -/
def gray_area (a b c : ℝ) : ℝ := (10 - a) + (7 - b) - c

/-- The black area of the rectangles -/
def black_area (a b c : ℝ) : ℝ := (13 - a) - b + (5 - c)

/-- Theorem stating that the black area is larger than the gray area by 1 square unit -/
theorem black_larger_than_gray (a b c : ℝ) : 
  black_area a b c - gray_area a b c = 1 := by sorry

end black_larger_than_gray_l3689_368912


namespace cost_per_page_is_five_l3689_368978

/-- The cost per page in cents when buying notebooks -/
def cost_per_page (num_notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (num_notebooks * pages_per_notebook)

/-- Theorem: The cost per page is 5 cents when buying 2 notebooks with 50 pages each for $5 -/
theorem cost_per_page_is_five :
  cost_per_page 2 50 5 = 5 := by
  sorry

#eval cost_per_page 2 50 5

end cost_per_page_is_five_l3689_368978


namespace xy_plus_2y_value_l3689_368945

theorem xy_plus_2y_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x*y + 2*y = 12 := by
  sorry

end xy_plus_2y_value_l3689_368945


namespace hippopotamus_cards_l3689_368971

theorem hippopotamus_cards (initial_cards remaining_cards : ℕ) : 
  initial_cards = 72 → remaining_cards = 11 → initial_cards - remaining_cards = 61 := by
  sorry

end hippopotamus_cards_l3689_368971


namespace intersection_area_theorem_l3689_368972

/-- A rectangular prism with side lengths a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A quadrilateral formed by the intersection of a plane with a rectangular prism -/
structure IntersectionQuadrilateral where
  prism : RectangularPrism
  -- Assume A and C are diagonally opposite vertices
  -- Assume B and D are midpoints of opposite edges not containing A or C

/-- The area of the quadrilateral formed by the intersection -/
noncomputable def intersection_area (quad : IntersectionQuadrilateral) : ℝ := sorry

/-- Theorem stating the area of the specific intersection quadrilateral -/
theorem intersection_area_theorem (quad : IntersectionQuadrilateral) 
  (h1 : quad.prism.a = 2)
  (h2 : quad.prism.b = 3)
  (h3 : quad.prism.c = 5) :
  intersection_area quad = 7 * Real.sqrt 26 / 2 := by sorry

end intersection_area_theorem_l3689_368972


namespace octal_567_equals_decimal_375_l3689_368907

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let hundreds := octal / 100
  let tens := (octal / 10) % 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal number 567 is equal to the decimal number 375 --/
theorem octal_567_equals_decimal_375 : octal_to_decimal 567 = 375 := by
  sorry

end octal_567_equals_decimal_375_l3689_368907


namespace power_equality_implies_exponent_l3689_368995

theorem power_equality_implies_exponent (a : ℝ) (m : ℕ) (h : (a^2)^m = a^6) : m = 3 := by
  sorry

end power_equality_implies_exponent_l3689_368995


namespace line_slope_l3689_368943

theorem line_slope (x y : ℝ) :
  (x / 2 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/2) :=
by
  sorry

end line_slope_l3689_368943


namespace black_white_pieces_difference_l3689_368925

theorem black_white_pieces_difference (B W : ℕ) : 
  (B - 1) / W = 9 / 7 →
  B / (W - 1) = 7 / 5 →
  B - W = 7 :=
by sorry

end black_white_pieces_difference_l3689_368925


namespace cubic_fraction_equals_fifteen_l3689_368942

theorem cubic_fraction_equals_fifteen :
  let a : ℤ := 7
  let b : ℤ := 5
  let c : ℤ := 3
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2) = 15 := by
  sorry

end cubic_fraction_equals_fifteen_l3689_368942


namespace max_sum_arithmetic_sequence_l3689_368918

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem max_sum_arithmetic_sequence
  (a₁ : ℚ)
  (h1 : a₁ = 13)
  (h2 : sum_arithmetic_sequence a₁ d 3 = sum_arithmetic_sequence a₁ d 11) :
  ∃ (n : ℕ), ∀ (m : ℕ), sum_arithmetic_sequence a₁ d n ≥ sum_arithmetic_sequence a₁ d m ∧ n = 7 :=
sorry

end max_sum_arithmetic_sequence_l3689_368918


namespace absolute_value_inequality_l3689_368977

theorem absolute_value_inequality (x : ℝ) : |x - 1| < 2 ↔ -1 < x ∧ x < 3 := by
  sorry

end absolute_value_inequality_l3689_368977


namespace sequence_properties_l3689_368906

def sequence_a (n : ℕ) : ℝ := sorry
def sequence_b (n : ℕ) : ℝ := sorry
def sequence_c (n : ℕ) : ℝ := sequence_a n * sequence_b n

def sum_S (n : ℕ) : ℝ := sorry
def sum_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, sequence_a n = (sum_S n + 2) / 2) ∧
  (sequence_b 1 = 1) ∧
  (∀ n : ℕ, sequence_b n - sequence_b (n + 1) + 2 = 0) →
  (sequence_a 1 = 2) ∧
  (sequence_a 2 = 4) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^n) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_b n = 2*n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → sum_T n = (2*n - 3) * 2^(n+1) + 6) :=
by sorry

end sequence_properties_l3689_368906


namespace cubic_equation_q_expression_l3689_368973

theorem cubic_equation_q_expression (a b q r : ℝ) (h1 : b ≠ 0) :
  (∃ (x : ℂ), x^3 + q*x + r = 0 ∧ (x = a + b*I ∨ x = a - b*I)) →
  q = b^2 - 3*a^2 := by
sorry

end cubic_equation_q_expression_l3689_368973


namespace bank_profit_maximization_l3689_368996

/-- The bank's profit maximization problem -/
theorem bank_profit_maximization
  (k : ℝ) (h_k_pos : k > 0) :
  let deposit_amount (x : ℝ) := k * x
  let profit (x : ℝ) := 0.048 * deposit_amount x - x * deposit_amount x
  ∃ (x_max : ℝ), x_max ∈ Set.Ioo 0 0.048 ∧
    ∀ (x : ℝ), x ∈ Set.Ioo 0 0.048 → profit x ≤ profit x_max ∧
    x_max = 0.024 :=
by sorry

end bank_profit_maximization_l3689_368996


namespace screening_methods_count_l3689_368955

/-- The number of units showing the documentary -/
def num_units : ℕ := 4

/-- The number of different screening methods -/
def screening_methods : ℕ := num_units ^ num_units

/-- Theorem stating that the number of different screening methods
    is equal to 4^4 when there are 4 units each showing the film once -/
theorem screening_methods_count :
  screening_methods = 4^4 :=
by sorry

end screening_methods_count_l3689_368955


namespace probability_log_integer_l3689_368931

def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 15 ∧ n = 3^k}

def is_valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ k : ℕ, b = a^k

def total_pairs : ℕ := Nat.choose 15 2

def valid_pairs : ℕ := 30

theorem probability_log_integer :
  (valid_pairs : ℚ) / total_pairs = 2 / 7 := by sorry

end probability_log_integer_l3689_368931


namespace total_jellybeans_needed_l3689_368981

/-- The number of jellybeans needed to fill a large glass -/
def large_glass_beans : ℕ := 50

/-- The number of jellybeans needed to fill a small glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The number of large glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small glasses -/
def num_small_glasses : ℕ := 3

/-- Theorem: The total number of jellybeans needed to fill all glasses is 325 -/
theorem total_jellybeans_needed : 
  num_large_glasses * large_glass_beans + num_small_glasses * small_glass_beans = 325 := by
  sorry

end total_jellybeans_needed_l3689_368981


namespace two_red_balls_in_bag_l3689_368993

/-- Represents the contents of a bag of balls -/
structure BagOfBalls where
  redBalls : ℕ
  yellowBalls : ℕ

/-- The probability of selecting a yellow ball given another yellow ball was selected -/
def probYellowGivenYellow (bag : BagOfBalls) : ℚ :=
  (bag.yellowBalls - 1) / (bag.redBalls + bag.yellowBalls - 1)

theorem two_red_balls_in_bag :
  ∀ (bag : BagOfBalls),
    bag.yellowBalls = 3 →
    probYellowGivenYellow bag = 1/2 →
    bag.redBalls = 2 := by
  sorry

end two_red_balls_in_bag_l3689_368993


namespace vector_parallel_if_negative_l3689_368965

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def parallel (a b : n) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_parallel_if_negative (a b : n) : a = -b → parallel a b := by
  sorry

end vector_parallel_if_negative_l3689_368965


namespace savings_comparison_l3689_368959

theorem savings_comparison (last_year_salary : ℝ) 
  (last_year_savings_rate : ℝ) (salary_increase : ℝ) (this_year_savings_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  salary_increase = 0.20 →
  this_year_savings_rate = 0.05 →
  (this_year_savings_rate * (1 + salary_increase) * last_year_salary) / 
  (last_year_savings_rate * last_year_salary) = 1 := by
  sorry

#check savings_comparison

end savings_comparison_l3689_368959


namespace part_one_part_two_l3689_368969

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one : 
  {x : ℝ | f 1 x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, a > 1 → (∀ x : ℝ, f a x + |x - 1| ≥ 2) ↔ a ≥ 3 := by sorry

end part_one_part_two_l3689_368969


namespace turtle_position_and_distance_l3689_368938

def turtle_movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

theorem turtle_position_and_distance :
  (List.sum turtle_movements = 5) ∧
  (List.sum (List.map Int.natAbs turtle_movements) = 47) := by
  sorry

end turtle_position_and_distance_l3689_368938


namespace trig_problem_l3689_368989

theorem trig_problem (α β : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi/2)
  (h_acute_β : 0 < β ∧ β < Real.pi/2)
  (h_sin_α : Real.sin α = 3/5)
  (h_tan_diff : Real.tan (α - β) = -1/3) :
  (Real.sin (α - β) = -Real.sqrt 10 / 10) ∧ 
  (Real.cos β = 9 * Real.sqrt 10 / 50) := by
sorry

end trig_problem_l3689_368989


namespace geometric_sequence_product_l3689_368908

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 2 * a 4 = 36 := by
  sorry

end geometric_sequence_product_l3689_368908


namespace problem_1_problem_2_l3689_368948

theorem problem_1 : (-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21 := by sorry

theorem problem_2 : -99*(8/9) * 8 = -799*(1/9) := by sorry

end problem_1_problem_2_l3689_368948


namespace sum_of_squares_in_ratio_l3689_368940

theorem sum_of_squares_in_ratio (a b c : ℚ) : 
  (a : ℚ) + b + c = 9 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 1701 / 49 := by
  sorry

end sum_of_squares_in_ratio_l3689_368940


namespace correct_operation_l3689_368901

theorem correct_operation (x y : ℝ) : 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0 := by
  sorry

end correct_operation_l3689_368901


namespace four_numbers_with_equal_sums_l3689_368922

theorem four_numbers_with_equal_sums (A : Finset ℕ) 
  (h1 : A.card = 12)
  (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 30)
  (h3 : A.card = Finset.card (Finset.image id A)) :
  ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d := by
  sorry


end four_numbers_with_equal_sums_l3689_368922


namespace marble_prob_diff_l3689_368957

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1200

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 800

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- Theorem stating the absolute difference between the probabilities -/
theorem marble_prob_diff :
  |prob_same_color - prob_diff_color| = 7900 / 199900 := by
  sorry


end marble_prob_diff_l3689_368957


namespace betty_orange_boxes_l3689_368999

theorem betty_orange_boxes (oranges_per_box : ℕ) (total_oranges : ℕ) (h1 : oranges_per_box = 24) (h2 : total_oranges = 72) :
  total_oranges / oranges_per_box = 3 :=
by sorry

end betty_orange_boxes_l3689_368999


namespace square_area_ratio_l3689_368924

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 12.5) (h2 : side_D = 18.5) :
  (side_C^2) / (side_D^2) = 625 / 1369 := by
  sorry

end square_area_ratio_l3689_368924


namespace sphere_volume_ratio_l3689_368958

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 8 / 27 := by
sorry

end sphere_volume_ratio_l3689_368958


namespace rotation_theorem_l3689_368952

/-- Triangle in 2D plane -/
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

/-- Rotation of a point around another point by 120° clockwise -/
def rotate120 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Generate the sequence of A points -/
def A (n : ℕ) (t : Triangle) : ℝ × ℝ :=
  match n % 3 with
  | 0 => t.A₃
  | 1 => t.A₁
  | _ => t.A₂

/-- Generate the sequence of P points -/
def P (n : ℕ) (t : Triangle) (P₀ : ℝ × ℝ) : ℝ × ℝ :=
  match n with
  | 0 => P₀
  | n + 1 => rotate120 (A (n + 1) t) (P n t P₀)

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

theorem rotation_theorem (t : Triangle) (P₀ : ℝ × ℝ) :
  P 1986 t P₀ = P₀ → isEquilateral t := by sorry

end rotation_theorem_l3689_368952


namespace ratio_sum_over_y_l3689_368974

theorem ratio_sum_over_y (x y : ℚ) (h : x / y = 1 / 2) : (x + y) / y = 3 / 2 := by
  sorry

end ratio_sum_over_y_l3689_368974


namespace grinder_purchase_price_l3689_368953

theorem grinder_purchase_price
  (mobile_cost : ℝ)
  (grinder_loss_percent : ℝ)
  (mobile_profit_percent : ℝ)
  (total_profit : ℝ)
  (h1 : mobile_cost = 8000)
  (h2 : grinder_loss_percent = 0.05)
  (h3 : mobile_profit_percent = 0.10)
  (h4 : total_profit = 50) :
  ∃ (grinder_cost : ℝ),
    grinder_cost * (1 - grinder_loss_percent) +
    mobile_cost * (1 + mobile_profit_percent) -
    (grinder_cost + mobile_cost) = total_profit ∧
    grinder_cost = 15000 := by
  sorry

end grinder_purchase_price_l3689_368953


namespace frequency_of_score_range_l3689_368954

theorem frequency_of_score_range (total_students : ℕ) (high_scorers : ℕ) 
  (h1 : total_students = 50) (h2 : high_scorers = 10) : 
  (high_scorers : ℚ) / total_students = 1 / 5 := by
  sorry

end frequency_of_score_range_l3689_368954


namespace integral_reciprocal_x_l3689_368988

theorem integral_reciprocal_x : ∫ x in (1:ℝ)..2, (1:ℝ) / x = Real.log 2 := by sorry

end integral_reciprocal_x_l3689_368988


namespace tan_eleven_pi_sixths_l3689_368951

theorem tan_eleven_pi_sixths : Real.tan (11 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_eleven_pi_sixths_l3689_368951


namespace units_digit_of_fib_F_15_l3689_368997

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- State that the units digit of Fibonacci numbers repeats every 60 terms
axiom fib_units_period (n : ℕ) : fib n % 10 = fib (n % 60) % 10

-- Define F_15
def F_15 : ℕ := fib 15

-- Theorem to prove
theorem units_digit_of_fib_F_15 : fib (fib 15) % 10 = 5 := by
  sorry

end units_digit_of_fib_F_15_l3689_368997


namespace amount_per_friend_is_correct_l3689_368928

/-- The amount each friend pays when 6 friends split a $400 bill equally after applying a 5% discount -/
def amount_per_friend : ℚ :=
  let total_bill : ℚ := 400
  let discount_rate : ℚ := 5 / 100
  let num_friends : ℕ := 6
  let discounted_bill : ℚ := total_bill * (1 - discount_rate)
  discounted_bill / num_friends

/-- Theorem stating that the amount each friend pays is $63.33 (repeating) -/
theorem amount_per_friend_is_correct :
  amount_per_friend = 190 / 3 := by
  sorry

#eval amount_per_friend

end amount_per_friend_is_correct_l3689_368928


namespace stirling_second_kind_l3689_368946

/-- Stirling number of the second kind -/
def S (n k : ℕ) : ℚ :=
  sorry

/-- Main theorem for Stirling numbers of the second kind -/
theorem stirling_second_kind (n : ℕ) (h : n ≥ 2) :
  (∀ k, k ≥ 2 → S n k = k * S (n-1) k + S (n-1) (k-1)) ∧
  S n 1 = 1 ∧
  S n 2 = 2^(n-1) - 1 ∧
  S n 3 = (1/6) * 3^n - (1/2) * 2^n + 1/2 ∧
  S n 4 = (1/24) * 4^n - (1/6) * 3^n + (1/4) * 2^n - 1/6 :=
by
  sorry

end stirling_second_kind_l3689_368946


namespace gumball_price_l3689_368980

/-- Given that Melanie sells 4 gumballs for a total of 32 cents, prove that each gumball costs 8 cents. -/
theorem gumball_price (num_gumballs : ℕ) (total_cents : ℕ) (price_per_gumball : ℕ) 
  (h1 : num_gumballs = 4)
  (h2 : total_cents = 32)
  (h3 : price_per_gumball * num_gumballs = total_cents) :
  price_per_gumball = 8 := by
  sorry

end gumball_price_l3689_368980


namespace part1_part2_l3689_368915

-- Define the inequality function
def inequality (a x : ℝ) : Prop := (a * x - 1) * (x + 1) > 0

-- Part 1: If the solution set is {x | -1 < x < -1/2}, then a = -2
theorem part1 (a : ℝ) : 
  (∀ x, inequality a x ↔ (-1 < x ∧ x < -1/2)) → a = -2 := 
sorry

-- Part 2: Solution sets for a ≤ 0
theorem part2 (a : ℝ) (h : a ≤ 0) : 
  (∀ x, inequality a x ↔ 
    (a < -1 ∧ -1 < x ∧ x < 1/a) ∨
    (a = -1 ∧ False) ∨
    (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
    (a = 0 ∧ x < -1)) :=
sorry

end part1_part2_l3689_368915


namespace quadratic_roots_always_positive_implies_a_zero_l3689_368960

theorem quadratic_roots_always_positive_implies_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, p > 0 →
    ∀ x : ℝ, a * x^2 + b * x + c + p = 0 →
      x > 0 ∧ (∃ y : ℝ, y ≠ x ∧ a * y^2 + b * y + c + p = 0 ∧ y > 0)) :
  a = 0 :=
sorry

end quadratic_roots_always_positive_implies_a_zero_l3689_368960


namespace complex_equal_parts_l3689_368968

/-- Given a complex number z = (2+ai)/(1+2i) where a is real,
    if the real part of z equals its imaginary part, then a = -6 -/
theorem complex_equal_parts (a : ℝ) :
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  Complex.re z = Complex.im z → a = -6 := by
  sorry

end complex_equal_parts_l3689_368968


namespace smallest_n_all_digits_odd_l3689_368963

/-- Function to check if all digits of a natural number are odd -/
def allDigitsOdd (n : ℕ) : Prop := sorry

/-- The smallest integer greater than 1 such that all digits of 9997n are odd -/
def smallestN : ℕ := 3335

theorem smallest_n_all_digits_odd :
  smallestN > 1 ∧
  allDigitsOdd (9997 * smallestN) ∧
  ∀ m : ℕ, m > 1 → m < smallestN → ¬(allDigitsOdd (9997 * m)) :=
sorry

end smallest_n_all_digits_odd_l3689_368963


namespace coin_flip_probability_l3689_368934

theorem coin_flip_probability (n : ℕ) : 
  (n.choose 2 : ℚ) / 2^n = 1/32 → n = 8 := by
  sorry

end coin_flip_probability_l3689_368934


namespace smaller_square_area_equals_larger_l3689_368962

/-- A square inscribed in a circle with another smaller square -/
structure SquaresInCircle where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square -/
  s : ℝ
  /-- Side length of the smaller square -/
  x : ℝ
  /-- The larger square is inscribed in the circle -/
  h1 : s = 2 * r
  /-- The smaller square has one side coinciding with the larger square -/
  h2 : x ≤ s
  /-- Two vertices of the smaller square are on the circle -/
  h3 : x^2 + (r + x/2)^2 = r^2

/-- The area of the smaller square is equal to the area of the larger square -/
theorem smaller_square_area_equals_larger (sq : SquaresInCircle) :
  sq.x^2 = sq.s^2 := by
  sorry

end smaller_square_area_equals_larger_l3689_368962


namespace dalton_savings_proof_l3689_368991

/-- The amount of money Dalton saved from his allowance -/
def dalton_savings : ℕ := sorry

/-- The cost of all items Dalton wants to buy -/
def total_cost : ℕ := 23

/-- The amount Dalton's uncle gave him -/
def uncle_contribution : ℕ := 13

/-- The additional amount Dalton needs -/
def additional_needed : ℕ := 4

theorem dalton_savings_proof :
  dalton_savings = total_cost - uncle_contribution - additional_needed :=
by sorry

end dalton_savings_proof_l3689_368991


namespace alice_bob_sum_l3689_368990

theorem alice_bob_sum : ∀ (a b : ℕ),
  1 ≤ a ∧ a ≤ 50 ∧                     -- Alice's number is between 1 and 50
  1 ≤ b ∧ b ≤ 50 ∧                     -- Bob's number is between 1 and 50
  a ≠ b ∧                              -- Numbers are drawn without replacement
  a ≠ 1 ∧ a ≠ 50 ∧                     -- Alice can't tell who has the larger number
  b > a ∧                              -- Bob knows he has the larger number
  ∃ (d : ℕ), d > 1 ∧ d < b ∧ d ∣ b ∧   -- Bob's number is composite
  ∃ (k : ℕ), 50 * b + a = k * k →      -- 50 * Bob's number + Alice's number is a perfect square
  a + b = 29 := by
sorry

end alice_bob_sum_l3689_368990


namespace log_product_equals_two_l3689_368976

theorem log_product_equals_two (y : ℝ) (h : y > 0) : 
  (Real.log y / Real.log 3) * (Real.log 9 / Real.log y) = 2 → y = 9 := by
  sorry

end log_product_equals_two_l3689_368976


namespace girls_percentage_less_than_boys_l3689_368905

theorem girls_percentage_less_than_boys (boys girls : ℝ) 
  (h : boys = girls * 1.25) : 
  (boys - girls) / boys = 0.2 := by
sorry

end girls_percentage_less_than_boys_l3689_368905


namespace angle_with_supplement_four_times_complement_l3689_368987

theorem angle_with_supplement_four_times_complement (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_with_supplement_four_times_complement_l3689_368987


namespace tommy_wheels_count_l3689_368911

/-- The number of wheels Tommy saw during his run -/
def total_wheels (truck_wheels car_wheels bicycle_wheels bus_wheels : ℕ)
                 (num_trucks num_cars num_bicycles num_buses : ℕ) : ℕ :=
  truck_wheels * num_trucks + car_wheels * num_cars +
  bicycle_wheels * num_bicycles + bus_wheels * num_buses

theorem tommy_wheels_count :
  total_wheels 4 4 2 6 12 13 8 3 = 134 := by
  sorry

end tommy_wheels_count_l3689_368911


namespace expand_expression_l3689_368900

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 - 2 * y) = 36 * x + 48 - 24 * y := by
  sorry

end expand_expression_l3689_368900


namespace toothpaste_amount_l3689_368914

/-- The amount of toothpaste used by Anne's dad per brushing -/
def dadUsage : ℕ := 3

/-- The amount of toothpaste used by Anne's mom per brushing -/
def momUsage : ℕ := 2

/-- The amount of toothpaste used by Anne or her brother per brushing -/
def childUsage : ℕ := 1

/-- The number of times each family member brushes their teeth per day -/
def brushingsPerDay : ℕ := 3

/-- The number of days it takes for the toothpaste to run out -/
def daysUntilEmpty : ℕ := 5

/-- The number of children (Anne and her brother) -/
def numberOfChildren : ℕ := 2

/-- Theorem stating that the amount of toothpaste in the tube is 105 grams -/
theorem toothpaste_amount : 
  dadUsage * brushingsPerDay * daysUntilEmpty + 
  momUsage * brushingsPerDay * daysUntilEmpty + 
  childUsage * brushingsPerDay * daysUntilEmpty * numberOfChildren = 105 := by
  sorry

end toothpaste_amount_l3689_368914


namespace unique_solution_l3689_368927

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  lg x + lg (x - 2) = lg 3 + lg (x + 2)

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x > 2 ∧ equation x :=
by
  sorry

end unique_solution_l3689_368927


namespace min_value_of_expression_l3689_368985

theorem min_value_of_expression (x y : ℝ) : (x^3*y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end min_value_of_expression_l3689_368985


namespace sequence_not_ap_or_gp_l3689_368967

-- Define the sequence
def a : ℕ → ℕ
  | n => if n % 2 = 0 then ((n / 2) + 1)^2 else (n / 2 + 1) * (n / 2 + 2)

-- State the theorem
theorem sequence_not_ap_or_gp :
  -- The sequence is increasing
  (∀ n : ℕ, a n < a (n + 1)) ∧
  -- Each even-indexed term is the arithmetic mean of its neighbors
  (∀ n : ℕ, a (2 * n) = (a (2 * n - 1) + a (2 * n + 1)) / 2) ∧
  -- Each odd-indexed term is the geometric mean of its neighbors
  (∀ n : ℕ, n > 0 → a (2 * n - 1) = Int.sqrt (a (2 * n - 2) * a (2 * n))) ∧
  -- The sequence never becomes an arithmetic progression
  (∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ a (n + 2) - a (n + 1) ≠ a (n + 1) - a n) ∧
  -- The sequence never becomes a geometric progression
  (∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ a (n + 2) * a n ≠ (a (n + 1))^2) :=
by sorry

end sequence_not_ap_or_gp_l3689_368967


namespace quartic_equation_roots_l3689_368944

theorem quartic_equation_roots (a b : ℝ) :
  let x : ℝ → Prop := λ x => x^4 - 2*a*x^2 + b^2 = 0
  ∃ (ε₁ ε₂ : {r : ℝ // r = 1 ∨ r = -1}),
    x (ε₁ * (Real.sqrt ((a + b)/2) + ε₂ * Real.sqrt ((a - b)/2))) :=
by
  sorry

end quartic_equation_roots_l3689_368944


namespace ellipse_minor_axis_length_l3689_368930

/-- An ellipse with axes parallel to the coordinate axes passing through 
    the given points has a minor axis length of 4. -/
theorem ellipse_minor_axis_length : 
  ∀ (e : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ e ↔ ((x - 3/2)^2 / 3^2) + ((y - 1)^2 / b^2) = 1) →
  (0, 0) ∈ e →
  (0, 2) ∈ e →
  (3, 0) ∈ e →
  (3, 2) ∈ e →
  (3/2, 3) ∈ e →
  ∃ (b : ℝ), b = 2 := by
  sorry

end ellipse_minor_axis_length_l3689_368930


namespace secret_codes_count_l3689_368947

/-- The number of colors available for the secret code -/
def num_colors : ℕ := 7

/-- The number of slots in the secret code -/
def num_slots : ℕ := 4

/-- The number of possible secret codes -/
def num_codes : ℕ := num_colors ^ num_slots

/-- Theorem: The number of possible secret codes is 2401 -/
theorem secret_codes_count : num_codes = 2401 := by
  sorry

end secret_codes_count_l3689_368947


namespace max_value_product_l3689_368919

theorem max_value_product (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b*c) * (b^2 - c*a) * (c^2 - a*b) ≤ 1/8 :=
by sorry

end max_value_product_l3689_368919


namespace least_six_digit_multiple_l3689_368929

theorem least_six_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  (12 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (23 ∣ n) ∧ (29 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100000 ∧ m < n → ¬((12 ∣ m) ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (23 ∣ m) ∧ (29 ∣ m))) :=
by
  use 120060
  sorry

end least_six_digit_multiple_l3689_368929


namespace price_per_kg_correct_l3689_368975

/-- The price per kilogram of rooster -/
def price_per_kg : ℝ := 0.5

/-- The weight of the first rooster in kilograms -/
def weight1 : ℝ := 30

/-- The weight of the second rooster in kilograms -/
def weight2 : ℝ := 40

/-- The total earnings from selling both roosters -/
def total_earnings : ℝ := 35

/-- Theorem stating that the price per kilogram is correct -/
theorem price_per_kg_correct : 
  price_per_kg * (weight1 + weight2) = total_earnings := by sorry

end price_per_kg_correct_l3689_368975


namespace cube_root_problem_l3689_368904

theorem cube_root_problem (x : ℝ) : (x + 6) ^ (1/3 : ℝ) = 3 → (x + 6) ^ 3 = 19683 := by
  sorry

end cube_root_problem_l3689_368904


namespace max_value_interval_max_value_at_one_l3689_368916

/-- The function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

/-- f(x) is monotonically decreasing in (-∞, 2] -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 2 → f a x ≥ f a y

theorem max_value_interval (a : ℝ) (h : is_monotone_decreasing a) :
  (∀ x ∈ Set.Icc 3 5, f a x ≤ 8) ∧ (∃ x ∈ Set.Icc 3 5, f a x = 8) :=
sorry

theorem max_value_at_one (a : ℝ) (h : is_monotone_decreasing a) :
  f a 1 ≤ 0 :=
sorry

end max_value_interval_max_value_at_one_l3689_368916


namespace mean_of_data_is_10_l3689_368902

def data : List ℝ := [8, 12, 10, 11, 9]

theorem mean_of_data_is_10 :
  (data.sum / data.length : ℝ) = 10 := by
  sorry

end mean_of_data_is_10_l3689_368902


namespace average_pencils_is_111_75_l3689_368966

def anna_pencils : ℕ := 50

def harry_pencils : ℕ := 2 * anna_pencils - 19

def lucy_pencils : ℕ := 3 * anna_pencils - 13

def david_pencils : ℕ := 4 * anna_pencils - 21

def total_pencils : ℕ := anna_pencils + harry_pencils + lucy_pencils + david_pencils

def average_pencils : ℚ := total_pencils / 4

theorem average_pencils_is_111_75 : average_pencils = 111.75 := by
  sorry

end average_pencils_is_111_75_l3689_368966


namespace prob_two_heads_in_four_tosses_l3689_368926

/-- The probability of getting exactly 2 heads when tossing a fair coin 4 times -/
theorem prob_two_heads_in_four_tosses : 
  let n : ℕ := 4  -- number of tosses
  let k : ℕ := 2  -- number of desired heads
  let p : ℚ := 1/2  -- probability of getting heads in a single toss
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 :=
sorry

end prob_two_heads_in_four_tosses_l3689_368926


namespace balloon_arrangements_count_l3689_368961

/-- The number of unique arrangements of letters in "BALLOON" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of "BALLOON" is 1260 -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry

end balloon_arrangements_count_l3689_368961


namespace sqrt_three_minus_pi_squared_l3689_368917

theorem sqrt_three_minus_pi_squared : Real.sqrt ((3 - Real.pi) ^ 2) = Real.pi - 3 := by
  sorry

end sqrt_three_minus_pi_squared_l3689_368917


namespace right_triangle_third_side_product_l3689_368910

theorem right_triangle_third_side_product (a b c d : ℝ) :
  a = 3 ∧ b = 6 ∧ 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) ∧
  (c > 0) ∧ (d > 0) →
  c * d = Real.sqrt 1215 := by
  sorry

end right_triangle_third_side_product_l3689_368910


namespace speed_limit_exceeders_l3689_368933

/-- Represents the percentage of motorists who receive speeding tickets -/
def speeding_ticket_percentage : ℝ := 10

/-- Represents the percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 30

/-- Represents the total percentage of motorists exceeding the speed limit -/
def exceeding_speed_limit_percentage : ℝ := 14

theorem speed_limit_exceeders (total_motorists : ℝ) (total_motorists_pos : total_motorists > 0) :
  (speeding_ticket_percentage / 100) * total_motorists =
  ((100 - no_ticket_percentage) / 100) * (exceeding_speed_limit_percentage / 100) * total_motorists :=
by sorry

end speed_limit_exceeders_l3689_368933


namespace largest_number_l3689_368932

theorem largest_number : 
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  b > a ∧ b > c ∧ b > d := by
  sorry

end largest_number_l3689_368932


namespace complex_fraction_equals_25_l3689_368984

theorem complex_fraction_equals_25 :
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := by
  sorry

end complex_fraction_equals_25_l3689_368984


namespace christopher_stroll_l3689_368986

theorem christopher_stroll (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 4 → time = 1.25 → distance = speed * time → distance = 5 := by
  sorry

end christopher_stroll_l3689_368986


namespace quadratic_inequality_l3689_368956

/-- Represents a quadratic function of the form f(x) = -2ax^2 + ax - 4 where a > 0 -/
def QuadraticFunction (a : ℝ) : ℝ → ℝ := 
  fun x ↦ -2 * a * x^2 + a * x - 4

theorem quadratic_inequality (a : ℝ) (ha : a > 0) :
  let f := QuadraticFunction a
  f 2 < f (-1) ∧ f (-1) < f 1 := by
  sorry

end quadratic_inequality_l3689_368956


namespace other_number_proof_l3689_368994

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 6300)
  (h2 : Nat.gcd a b = 15)
  (h3 : a = 210) : 
  b = 450 := by
sorry

end other_number_proof_l3689_368994


namespace pirates_escape_strategy_l3689_368920

-- Define the type for colors (0 to 9)
def Color := Fin 10

-- Define the type for the sequence of hat colors
def HatSequence := ℕ → Color

-- Define the type for a pirate's strategy
def Strategy := (ℕ → Color) → Color

-- Define the property of a valid strategy
def ValidStrategy (s : Strategy) (h : HatSequence) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m ≥ n → s (fun i => h (i + m + 1)) = h m

-- Theorem statement
theorem pirates_escape_strategy :
  ∃ (s : Strategy), ∀ (h : HatSequence), ValidStrategy s h :=
sorry

end pirates_escape_strategy_l3689_368920


namespace no_max_min_value_l3689_368939

/-- The function f(x) = x³ - (3/2)x² + 1 has neither a maximum value nor a minimum value -/
theorem no_max_min_value (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - (3/2)*x^2 + 1) :
  (¬ ∃ y, ∀ x, f x ≤ f y) ∧ (¬ ∃ y, ∀ x, f x ≥ f y) := by
  sorry

end no_max_min_value_l3689_368939


namespace no_double_by_digit_move_l3689_368983

theorem no_double_by_digit_move :
  ¬ ∃ (x : ℕ) (n : ℕ), n ≥ 1 ∧
    (∃ (a : ℕ) (N : ℕ),
      x = a * 10^n + N ∧
      0 < a ∧ a < 10 ∧
      N < 10^n ∧
      10 * N + a = 2 * x) :=
sorry

end no_double_by_digit_move_l3689_368983


namespace five_student_committees_l3689_368979

theorem five_student_committees (n k : ℕ) (h1 : n = 8) (h2 : k = 5) : 
  Nat.choose n k = 56 := by
  sorry

end five_student_committees_l3689_368979


namespace one_tree_baskets_l3689_368935

/-- The number of apples that can fit in one basket -/
def apples_per_basket : ℕ := 15

/-- The number of apples produced by 10 trees -/
def apples_from_ten_trees : ℕ := 3000

/-- The number of trees -/
def number_of_trees : ℕ := 10

/-- Theorem: One apple tree can fill 20 baskets -/
theorem one_tree_baskets : 
  (apples_from_ten_trees / number_of_trees) / apples_per_basket = 20 := by
  sorry

end one_tree_baskets_l3689_368935


namespace perpendicular_line_to_plane_l3689_368909

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (perpendicularLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_to_plane 
  (α β : Plane) (m n : Line) :
  perpendicular α β →
  intersect α β m →
  contains α n →
  perpendicularLines n m →
  perpendicularLineToPlane n β :=
sorry

end perpendicular_line_to_plane_l3689_368909


namespace mod_equivalence_unique_solution_l3689_368998

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15827 [ZMOD 16] ∧ n = 3 := by
  sorry

end mod_equivalence_unique_solution_l3689_368998


namespace group_size_is_factor_l3689_368964

def num_cows : ℕ := 24
def num_sheep : ℕ := 7
def num_goats : ℕ := 113

def total_animals : ℕ := num_cows + num_sheep + num_goats

theorem group_size_is_factor :
  ∀ (group_size : ℕ), 
    group_size > 1 ∧ 
    group_size < total_animals ∧ 
    total_animals % group_size = 0 →
    ∃ (num_groups : ℕ), num_groups * group_size = total_animals :=
by sorry

end group_size_is_factor_l3689_368964


namespace power_equation_solution_l3689_368936

theorem power_equation_solution :
  ∃ x : ℝ, (5 : ℝ)^(x + 2) = 625 ∧ x = 2 := by sorry

end power_equation_solution_l3689_368936


namespace smaller_factor_of_4851_l3689_368982

theorem smaller_factor_of_4851 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4851 → 
  min a b = 53 := by
sorry

end smaller_factor_of_4851_l3689_368982


namespace cake_comparison_l3689_368970

theorem cake_comparison : (1 : ℚ) / 3 > (1 : ℚ) / 4 ∧ (1 : ℚ) / 3 > (1 : ℚ) / 5 := by
  sorry

end cake_comparison_l3689_368970


namespace chord_intersection_triangles_l3689_368992

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of chords is the number of ways to choose 2 points from n points -/
def num_chords : ℕ := n.choose 2

/-- The number of intersection points is the number of ways to choose 4 points from n points -/
def num_intersections : ℕ := n.choose 4

/-- The number of triangles is the number of ways to choose 3 intersection points -/
def num_triangles : ℕ := num_intersections.choose 3

/-- Theorem stating the number of triangles formed by chord intersections -/
theorem chord_intersection_triangles :
  num_triangles = 1524180 :=
sorry

end chord_intersection_triangles_l3689_368992


namespace exists_sequence_to_target_state_l3689_368949

-- Define the state of the urn
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

-- Define the replacement rules
inductive ReplacementRule
| Rule1 -- 3 black → 1 black + 2 white
| Rule2 -- 2 black + 1 white → 3 black
| Rule3 -- 1 black + 2 white → 2 white
| Rule4 -- 3 white → 2 white + 1 black

-- Define a function to apply a rule to an urn state
def applyRule (state : UrnState) (rule : ReplacementRule) : UrnState :=
  match rule with
  | ReplacementRule.Rule1 => 
      if state.black ≥ 3 then UrnState.mk (state.white + 2) (state.black - 2) else state
  | ReplacementRule.Rule2 => 
      if state.black ≥ 2 ∧ state.white ≥ 1 then UrnState.mk (state.white - 1) (state.black + 1) else state
  | ReplacementRule.Rule3 => 
      if state.black ≥ 1 ∧ state.white ≥ 2 then UrnState.mk state.white (state.black - 1) else state
  | ReplacementRule.Rule4 => 
      if state.white ≥ 3 then UrnState.mk (state.white - 1) (state.black + 1) else state

-- Define the initial state
def initialState : UrnState := UrnState.mk 50 50

-- Define the target state
def targetState : UrnState := UrnState.mk 2 0

-- Theorem to prove
theorem exists_sequence_to_target_state : 
  ∃ (sequence : List ReplacementRule), 
    (sequence.foldl applyRule initialState) = targetState :=
sorry

end exists_sequence_to_target_state_l3689_368949


namespace hyperbola_asymptote_slopes_l3689_368950

/-- The slopes of the asymptotes of a hyperbola -/
def asymptote_slopes (a b : ℝ) : Set ℝ :=
  {m : ℝ | m = b / a ∨ m = -b / a}

/-- Theorem: The slopes of the asymptotes of the hyperbola (x^2/16) - (y^2/25) = 1 are ±5/4 -/
theorem hyperbola_asymptote_slopes :
  asymptote_slopes 4 5 = {5/4, -5/4} := by
  sorry

#check hyperbola_asymptote_slopes

end hyperbola_asymptote_slopes_l3689_368950


namespace binomial_variance_calculation_l3689_368937

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial variable -/
def expectedValue (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial variable -/
def variance (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_variance_calculation (ξ : BinomialVariable) 
  (h_n : ξ.n = 36) 
  (h_exp : expectedValue ξ = 12) : 
  variance ξ = 8 := by
  sorry

end binomial_variance_calculation_l3689_368937
