import Mathlib

namespace count_irrational_numbers_l30_30176

def is_irrational (x : Real) : Prop := ¬ ∃ (a b : Int), b ≠ 0 ∧ x = a / b

-- Given list of numbers
def nums : List Real := [3.14159, -Real.cbrt 9, 0.131131113, -Real.pi, Real.sqrt 25, Real.cbrt 64, -1/7]

-- List of irrational numbers from provided list
def irrational_nums : List Real := [nums.nthLe 1 (by norm_num), nums.nthLe 2 (by norm_num), nums.nthLe 3 (by norm_num)]

theorem count_irrational_numbers : irrational_nums.length = 3 := 
by 
  -- Justification of each irrational number
  have h1 : is_irrational (nums.nthLe 1 (by norm_num)) := sorry,
  have h2 : is_irrational (nums.nthLe 2 (by norm_num)) := sorry,
  have h3 : is_irrational (nums.nthLe 3 (by norm_num)) := sorry,
  sorry

end count_irrational_numbers_l30_30176


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30503

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30503


namespace rectangle_diagonal_length_l30_30655

theorem rectangle_diagonal_length (L W : ℝ) (h1 : 2 * (L + W) = 72) (h2 : L / W = 5 / 2) : 
    (sqrt (L^2 + W^2)) = 194 / 7 := 
by 
  sorry

end rectangle_diagonal_length_l30_30655


namespace angle_bisector_inequality_l30_30626

-- Definitions and setup
variables {A B C D : Type} -- Consider A, B, C, D as points.

-- Assume the given conditions
axiom cyclic_quadrilateral (h : CyclicQuadrilateral A B C D) : Prop
axiom angle_bisector_of_A (h : AngleBisector A B C D) : Prop

-- Theorem statement
theorem angle_bisector_inequality
  (h_cyclic: cyclic_quadrilateral (A := A) (B := B) (C := C) (D := D))
  (h_angle_bisector: angle_bisector_of_A (A := A) (B := B) (C := C) (D := D)):
  distance A B + distance A C ≤ 2 * distance A D :=
sorry

end angle_bisector_inequality_l30_30626


namespace shopkeeper_loss_l30_30165

def total_fruits : ℕ := 700
def apples : ℕ := 280
def oranges : ℕ := 210
def bananas : ℕ := total_fruits - apples - oranges

def morning_prices : ℕ → ℕ
| n := if n = 0 then 5 else if n = 1 then 4 else 2

def afternoon_prices : ℕ → ℕ
| n := if n = 0 then 5.6 else if n = 1 then 4.6 else 2.16

def overhead_cost : ℕ := 320

def sold_percent_morning : ℕ → ℕ
| n := if n = 0 then 50 else if n = 1 then 60 else 80

def remaining_percent_afternoon : ℕ → ℕ
| n := 100 - sold_percent_morning n

noncomputable def revenue_morning :=
  (sold_percent_morning 0 / 100 * apples * morning_prices 0) +
  (sold_percent_morning 1 / 100 * oranges * morning_prices 1) +
  (sold_percent_morning 2 / 100 * bananas * morning_prices 2)

noncomputable def revenue_afternoon :=
  (remaining_percent_afternoon 0 / 100 * apples * afternoon_prices 0) +
  (remaining_percent_afternoon 1 / 100 * oranges * afternoon_prices 1) +
  (remaining_percent_afternoon 2 / 100 * bananas * afternoon_prices 2)

noncomputable def total_revenue :=
  revenue_morning + revenue_afternoon

noncomputable def cost_of_fruits :=
  (apples * morning_prices 0) +
  (oranges * morning_prices 1) +
  (bananas * morning_prices 2)

noncomputable def total_cost :=
  cost_of_fruits + overhead_cost

noncomputable def profit :=
  total_revenue - total_cost

theorem shopkeeper_loss : profit = -178.88 := by
  sorry

end shopkeeper_loss_l30_30165


namespace smallest_positive_period_max_min_values_l30_30957

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem smallest_positive_period (x : ℝ) :
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ T', T' > 0 ∧ ∀ x, f (x + T') = f x → T ≤ T' :=
  sorry

theorem max_min_values : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
  min ≤ f x ∧ f x ≤ max :=
  sorry

end smallest_positive_period_max_min_values_l30_30957


namespace arithmetic_sequence_geometric_condition_l30_30304

noncomputable def a_n (n : ℕ) := 1 + (n - 1) * d

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℕ) (d : ℕ) (h_arith : ∀ n, a (n + 1) - a n = d)
  (h_1 : a 1 = 1) (h_geo : a 1 * (a 1 + 8 * d) = (a 1 + 2 * d) * (a 1 + 2 * d)) :
  (∀ n, a n = n) ∨ (∀ n, a n = 1) := 
  sorry

end arithmetic_sequence_geometric_condition_l30_30304


namespace product_of_divisors_has_three_prime_factors_l30_30457

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30457


namespace problem_1_problem_2_l30_30294

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Conditions
axiom h1 : ∀ n : ℕ, 2 * S n = a (n + 1) - 2^(n + 1) + 1
axiom h2 : a 2 + 5 = a 1 + (a 3 - a 2)

-- Problem 1: Prove the value of a₁
theorem problem_1 : a 1 = 1 := sorry

-- Problem 2: Find the general term formula for the sequence {aₙ}
theorem problem_2 : ∀ n : ℕ, a n = 3^n - 2^n := sorry

end problem_1_problem_2_l30_30294


namespace patty_vs_ida_cost_difference_l30_30605

-- Definitions as per the conditions provided
variables {P I J Paul : ℕ}

-- Condition 1: Ida's dress was $30 more than Jean's dress
def ida_more_than_jean := I = J + 30

-- Condition 2: Jean's dress was $10 less than Pauline's dress
def jean_less_than_pauline := J = Paul - 10

-- Condition 3: Pauline's dress was $30
def pauline_cost := Paul = 30

-- Condition 4: All the ladies spent $160 on dresses put together
def total_spent := P + I + J + Paul = 160

-- The proof problem
theorem patty_vs_ida_cost_difference
  (h1 : ida_more_than_jean)
  (h2 : jean_less_than_pauline)
  (h3 : pauline_cost)
  (h4 : total_spent) : P = I + 10 := by
  sorry

end patty_vs_ida_cost_difference_l30_30605


namespace sin_sum_angles_36_108_l30_30611

theorem sin_sum_angles_36_108 (A B C : ℝ) (h_sum : A + B + C = 180)
  (h_angle : A = 36 ∨ A = 108 ∨ B = 36 ∨ B = 108 ∨ C = 36 ∨ C = 108) :
  Real.sin (5 * A) + Real.sin (5 * B) + Real.sin (5 * C) = 0 :=
by
  sorry

end sin_sum_angles_36_108_l30_30611


namespace sum_of_factors_24_l30_30714

theorem sum_of_factors_24 : ∑ i in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), i = 60 := 
by
  sorry

end sum_of_factors_24_l30_30714


namespace personal_planner_cost_l30_30166

variable (P : ℝ)
variable (C_spiral_notebook : ℝ := 15)
variable (total_cost_with_discount : ℝ := 112)
variable (discount_rate : ℝ := 0.20)
variable (num_spiral_notebooks : ℝ := 4)
variable (num_personal_planners : ℝ := 8)

theorem personal_planner_cost : (4 * C_spiral_notebook + 8 * P) * (1 - 0.20) = 112 → 
  P = 10 :=
by
  sorry

end personal_planner_cost_l30_30166


namespace arithmetic_mean_of_set_l30_30909

theorem arithmetic_mean_of_set (n : ℕ) (h : n > 2) (s : List ℝ) :
  (List.count (1 - 2 / n) s = 2) ∧ (List.count 1 s = n - 2) → 
  (s.sum / n = 1 - 4 / n^2) :=
by
  sorry

end arithmetic_mean_of_set_l30_30909


namespace hexagons_after_cuts_l30_30681

theorem hexagons_after_cuts (rectangles_initial : ℕ) (cuts : ℕ) (sheets_total : ℕ)
  (initial_sides : ℕ) (additional_sides : ℕ) 
  (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (final_sides : ℕ) (number_of_hexagons : ℕ) :
  rectangles_initial = 15 →
  cuts = 60 →
  sheets_total = rectangles_initial + cuts →
  initial_sides = rectangles_initial * 4 →
  additional_sides = cuts * 4 →
  final_sides = initial_sides + additional_sides →
  triangle_sides = 3 →
  hexagon_sides = 6 →
  (sheets_total * 4 = final_sides) →
  number_of_hexagons = (final_sides - 225) / 3 →
  number_of_hexagons = 25 :=
by
  intros
  sorry

end hexagons_after_cuts_l30_30681


namespace cone_heights_l30_30804

theorem cone_heights (H x r1 r2 : ℝ) (H_frustum : H - x = 18)
  (A_lower : 400 * Real.pi = Real.pi * r1^2)
  (A_upper : 100 * Real.pi = Real.pi * r2^2)
  (ratio_radii : r2 / r1 = 1 / 2)
  (ratio_heights : x / H = 1 / 2) :
  x = 18 ∧ H = 36 :=
by
  sorry

end cone_heights_l30_30804


namespace third_number_correct_l30_30111

-- Given that the row of Pascal's triangle with 51 numbers corresponds to the binomial coefficients of 50.
def third_number_in_51_pascal_row : ℕ := Nat.choose 50 2

-- Prove that the third number in this row of Pascal's triangle is 1225.
theorem third_number_correct : third_number_in_51_pascal_row = 1225 := 
by 
  -- Calculation part can be filled in for the full proof.
  sorry

end third_number_correct_l30_30111


namespace cistern_fill_time_l30_30148

theorem cistern_fill_time (fillA emptyB : ℕ) (hA : fillA = 8) (hB : emptyB = 12) : (24 : ℕ) = 24 :=
by
  sorry

end cistern_fill_time_l30_30148


namespace probability_A_probability_B_l30_30122

def dice_outcomes : List (ℕ × ℕ) := 
  [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
   (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
   
def event_A (outcome : ℕ × ℕ) : Prop := 
  outcome.fst = outcome.snd
  
def event_B (outcome : ℕ × ℕ) : Prop := 
  outcome.fst + outcome.snd = 8
  
theorem probability_A : 
  (List.filter event_A dice_outcomes).length = 6 ∧ (List.length dice_outcomes) = 36 → 
  (List.filter event_A dice_outcomes).length / (List.length dice_outcomes) = (1 / 6) := by
  sorry
  
theorem probability_B : 
  (List.filter event_B dice_outcomes).length = 5 ∧ (List.length dice_outcomes) = 36 → 
  (List.filter event_B dice_outcomes).length / (List.length dice_outcomes) = (5 / 36) := by
  sorry

end probability_A_probability_B_l30_30122


namespace smallest_palindrome_base2_base4_gt10_l30_30201

/--
  Compute the smallest base-10 positive integer greater than 10 that is a palindrome 
  when written in both base 2 and base 4.
-/
theorem smallest_palindrome_base2_base4_gt10 : 
  ∃ n : ℕ, 10 < n ∧ is_palindrome (nat_to_base 2 n) ∧ is_palindrome (nat_to_base 4 n) ∧ n = 15 :=
sorry

def nat_to_base (b n : ℕ) : list ℕ :=
sorry

def is_palindrome {α : Type} [DecidableEq α] (l : list α) : bool :=
sorry

end smallest_palindrome_base2_base4_gt10_l30_30201


namespace hide_and_seek_l30_30770

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30770


namespace mountain_number_count_l30_30857

noncomputable def isMountainNumber (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  let sorted_digits := digits.qsort (λ x y => x > y)
  digits.length = 4 ∧
  digits.all (λ d => 1 ≤ d ∧ d ≤ 9) ∧
  digits.nodup ∧
  sorted_digits[1] > sorted_digits[0] ∧ 
  sorted_digits[1] > sorted_digits[2] ∧ 
  sorted_digits[1] > sorted_digits[3] ∧
  digits[0] ≠ digits[3]

noncomputable def countMountainNumbers : ℕ :=
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ isMountainNumber n}.card

theorem mountain_number_count : countMountainNumbers = 3024 := by
  sorry

end mountain_number_count_l30_30857


namespace mutually_exclusive_not_opposite_l30_30277

-- Define the context: a bag containing 2 red balls and 2 black balls
constant Bag : Type
def red : Bag := sorry
def black : Bag := sorry
def ball_set : set Bag := {red, red, black, black}

-- Define the condition where two balls are drawn from the bag
def draw_two (s : set Bag) : set (Bag × Bag) :=
  { (a, b) | a ∈ s ∧ b ∈ s ∧ a ≠ b }

-- Define the events of interest
def one_black (pair : Bag × Bag) : Prop := (pair.fst = black ∧ pair.snd ≠ black) ∨ (pair.snd = black ∧ pair.fst ≠ black)
def two_black (pair : Bag × Bag) : Prop := pair.fst = black ∧ pair.snd = black

-- Define mutually exclusive but not opposite events
def not_opposite_and_exclusive (s : set Bag) : Prop :=
  ∃ (a b : Bag × Bag), one_black a ∧ two_black b ∧ (a ≠ b)

-- Prove the assertion that "exactly one black ball" and "exactly two black balls" are mutually exclusive but not opposite
theorem mutually_exclusive_not_opposite : not_opposite_and_exclusive ball_set := 
  sorry

end mutually_exclusive_not_opposite_l30_30277


namespace find_f_neg_two_l30_30903

variable {R : Type} [Real]

-- Define the function f and its properties.
def f (x : R) : R

-- Specify conditions for the function f.
axiom f_additive_property : ∀ (x y : R), f(x + y) = f(x) + f(y) + 4 * x * y
axiom f_at_one : f(1) = 2

-- State the theorem to be proven.
theorem find_f_neg_two : f(-2) = 8 := 
by 
  sorry

end find_f_neg_two_l30_30903


namespace triangle_sine_ratio_l30_30789

theorem triangle_sine_ratio (ABC : Triangle) (A1 B1 C1 : Point) 
  (hA1 : A1 ∈ line_segment(BC))
  (hB1 : B1 ∈ line_segment(CA))
  (hC1 : C1 ∈ line_segment(AB)) :
  (dist(AC1) / dist(C1B)) * (dist(BA1) / dist(A1C)) * (dist(CB1) / dist(B1A)) = 
  (real.sin(angle(ACC1)) / real.sin(angle(C1CB))) * 
  (real.sin(angle(BAA1)) / real.sin(angle(A1AC))) * 
  (real.sin(angle(CBB1)) / real.sin(angle(B1BA))) :=
sorry

end triangle_sine_ratio_l30_30789


namespace smallest_unreachable_integer_l30_30265

/-- The smallest positive integer that cannot be expressed in the form (2^a - 2^b) / (2^c - 2^d) where a, b, c, d are non-negative integers is 11. -/
theorem smallest_unreachable_integer : 
  ∀ (a b c d : ℕ), 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
  ∃ (n : ℕ), n = 11 ∧ ¬ ∃ (a b c d : ℕ), (2^a - 2^b) / (2^c - 2^d) = n :=
by
  sorry

end smallest_unreachable_integer_l30_30265


namespace distinct_prime_factors_of_B_l30_30442

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30442


namespace smallest_palindromic_integer_l30_30210

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n
  digits = List.reverse digits

theorem smallest_palindromic_integer :
  ∃ (n : ℕ), n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ (∀ m, m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 → n ≤ m) :=
begin
  sorry
end

end smallest_palindromic_integer_l30_30210


namespace distinct_prime_factors_B_l30_30546

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30546


namespace max_value_of_expr_l30_30106

noncomputable def max_value (t : ℕ) : ℝ := (3^t - 2*t)*t / 9^t

theorem max_value_of_expr :
  ∃ t : ℕ, max_value t = 1 / 8 :=
sorry

end max_value_of_expr_l30_30106


namespace rectangle_diagonal_length_l30_30643

theorem rectangle_diagonal_length (P : ℝ) (r : ℝ) (perimeter_eq : P = 72) (ratio_eq : r = 5/2) :
  let k := P / 14
  let l := 5 * k
  let w := 2 * k
  let d := Real.sqrt (l^2 + w^2)
  d = 194 / 7 :=
sorry

end rectangle_diagonal_length_l30_30643


namespace sum_of_factors_24_l30_30715

theorem sum_of_factors_24 : ∑ i in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), i = 60 := 
by
  sorry

end sum_of_factors_24_l30_30715


namespace distinct_prime_factors_product_divisors_60_l30_30484

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30484


namespace product_of_divisors_has_three_prime_factors_l30_30456

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30456


namespace wheat_flour_packets_correct_l30_30699

-- Define the initial amount of money Victoria had.
def initial_amount : ℕ := 500

-- Define the cost and quantity of rice packets Victoria bought.
def rice_packet_cost : ℕ := 20
def rice_packets : ℕ := 2

-- Define the cost and quantity of soda Victoria bought.
def soda_cost : ℕ := 150
def soda_quantity : ℕ := 1

-- Define the remaining balance after shopping.
def remaining_balance : ℕ := 235

-- Define the cost of one packet of wheat flour.
def wheat_flour_packet_cost : ℕ := 25

-- Define the total amount spent on rice and soda.
def total_spent_on_rice_and_soda : ℕ :=
  (rice_packets * rice_packet_cost) + (soda_quantity * soda_cost)

-- Define the total amount spent on wheat flour.
def total_spent_on_wheat_flour : ℕ :=
  initial_amount - remaining_balance - total_spent_on_rice_and_soda

-- Define the expected number of wheat flour packets bought.
def wheat_flour_packets_expected : ℕ := 3

-- The statement we want to prove: the number of wheat flour packets bought is 3.
theorem wheat_flour_packets_correct : total_spent_on_wheat_flour / wheat_flour_packet_cost = wheat_flour_packets_expected :=
  sorry

end wheat_flour_packets_correct_l30_30699


namespace expected_ones_three_dice_l30_30080

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30080


namespace smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30209

def is_palindrome_in_base (n : ℕ) (b : ℕ) : Prop :=
  let digits : List ℕ := (nat.to_digits b n)
  digits = digits.reverse

theorem smallest_palindrome_in_bases_2_and_4 :
  ∀ n : ℕ, 10 < n ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 → n ≥ 17 :=
begin
  intros n hn,
  have := (by norm_num : 16 < 17),
  cases hn with hn1 hn2,
  cases hn2 with hn3 hn4,
  linarith,
end

theorem smallest_palindrome_in_bases_2_and_4_17 :
  is_palindrome_in_base 17 2 ∧ is_palindrome_in_base 17 4 :=
begin
  split;
  { unfold is_palindrome_in_base, norm_num, refl },
end

end smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30209


namespace rectangle_diagonal_length_l30_30659

theorem rectangle_diagonal_length (L W : ℝ) (h1 : 2 * (L + W) = 72) (h2 : L / W = 5 / 2) : 
    (sqrt (L^2 + W^2)) = 194 / 7 := 
by 
  sorry

end rectangle_diagonal_length_l30_30659


namespace units_digit_of_product_of_special_set_l30_30717

-- Define the set of integers satisfying the given conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_between_20_and_50 (n : ℕ) : Prop := 20 < n ∧ n < 50
def is_not_multiple_of_5 (n : ℕ) : Prop := ¬ (n % 5 = 0)

-- Define the specific set of odd positive integers between 20 and 50 that are not multiples of 5
def special_set : set ℕ := {n | is_odd n ∧ is_between_20_and_50 n ∧ is_not_multiple_of_5 n}

-- Define the function to get the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the product of all elements in the special_set
noncomputable def product_of_special_set : ℕ :=
  finset.prod (finset.filter (λ n, is_odd n ∧ is_between_20_and_50 n ∧ is_not_multiple_of_5 n) (finset.Ico 21 50)) id

-- Prove that the units digit of the product of the special_set is 9
theorem units_digit_of_product_of_special_set : units_digit product_of_special_set = 9 := by
  sorry

end units_digit_of_product_of_special_set_l30_30717


namespace distinct_prime_factors_product_divisors_60_l30_30483

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30483


namespace remainder_sum_first_six_primes_div_seventh_prime_l30_30708

-- Define the first six prime numbers
def firstSixPrimes : List ℕ := [2, 3, 5, 7, 11, 13]

-- Define the sum of the first six prime numbers
def sumOfFirstSixPrimes : ℕ := firstSixPrimes.sum

-- Define the seventh prime number
def seventhPrime : ℕ := 17

-- Proof statement that the remainder of the division is 7
theorem remainder_sum_first_six_primes_div_seventh_prime :
  (sumOfFirstSixPrimes % seventhPrime) = 7 :=
by
  sorry

end remainder_sum_first_six_primes_div_seventh_prime_l30_30708


namespace angle_bisectors_intersect_at_one_point_l30_30612

theorem angle_bisectors_intersect_at_one_point
  (A B C : Type u)
  [metric_space A] [metric_space B] [metric_space C]
  [is_triangle A B C]
  (O : Type u)
  [is_external_angle_bisector O B C]
  [is_external_angle_bisector O C B]
  [is_internal_angle_bisector O A B C] :
  ∃ O, (is_external_angle_bisector O B C) ∧ (is_external_angle_bisector O C B) ∧ (is_internal_angle_bisector O A B C) :=
sorry

end angle_bisectors_intersect_at_one_point_l30_30612


namespace expected_number_of_ones_when_three_dice_rolled_l30_30031

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30031


namespace expected_ones_on_three_dice_l30_30088

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30088


namespace most_likely_dissatisfied_correct_expected_dissatisfied_correct_variance_dissatisfied_correct_l30_30680

noncomputable def most_likely_dissatisfied (n : ℕ) : ℕ :=
1 -- 1 dissatisfied passenger is the most probable number.

noncomputable def expected_dissatisfied (n : ℕ) : ℝ :=
0.564 * real.sqrt n -- Expected number of dissatisfied passengers is approximately 0.564 * sqrt(n).

noncomputable def variance_dissatisfied (n : ℕ) : ℝ :=
0.182 * n -- Variance of the number of dissatisfied passengers is approximately 0.182 * n.

theorem most_likely_dissatisfied_correct (n : ℕ) :
  most_likely_dissatisfied n = 1 :=
by sorry

theorem expected_dissatisfied_correct (n : ℕ) :
  expected_dissatisfied n ≈ 0.564 * real.sqrt n :=
by sorry

theorem variance_dissatisfied_correct (n : ℕ) :
  variance_dissatisfied n ≈ 0.182 * n :=
by sorry

end most_likely_dissatisfied_correct_expected_dissatisfied_correct_variance_dissatisfied_correct_l30_30680


namespace point_A_in_second_quadrant_l30_30379

def point_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_A_in_second_quadrant : 
  point_in_second_quadrant (-2) 3 :=
by
  split
  -- proof of x < 0
  show -2 < 0
  sorry
  -- proof of y > 0
  show 3 > 0
  sorry

end point_A_in_second_quadrant_l30_30379


namespace increasing_integers_divisible_by_15_l30_30361

def is_increasing_integer (n : Nat) : Prop :=
  let digits := n.digits 10
  ∀ i, i < digits.length - 1 → digits.get i < digits.get (i + 1)

def is_divisible_by_15 (n : Nat) : Prop :=
  n % 15 = 0

theorem increasing_integers_divisible_by_15 :
  {n : Nat | is_increasing_integer n ∧ is_divisible_by_15 n}.toFinset.card = 6 :=
by
  sorry

end increasing_integers_divisible_by_15_l30_30361


namespace foci_coordinates_l30_30628

-- Define the parameters for the hyperbola
def a_squared : ℝ := 3
def b_squared : ℝ := 1
def c_squared : ℝ := a_squared + b_squared

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

-- State the theorem about the coordinates of the foci
theorem foci_coordinates : {foci : ℝ × ℝ // foci = (-2, 0) ∨ foci = (2, 0)} :=
by 
  have ha : a_squared = 3 := rfl
  have hb : b_squared = 1 := rfl
  have hc : c_squared = a_squared + b_squared := rfl
  have c := Real.sqrt c_squared
  have hc' : c = 2 := 
  -- sqrt part can be filled if detailed, for now, just direct conclusion
  sorry
  exact ⟨(2, 0), Or.inr rfl⟩

end foci_coordinates_l30_30628


namespace rectangle_diagonal_length_l30_30658

theorem rectangle_diagonal_length (L W : ℝ) (h1 : 2 * (L + W) = 72) (h2 : L / W = 5 / 2) : 
    (sqrt (L^2 + W^2)) = 194 / 7 := 
by 
  sorry

end rectangle_diagonal_length_l30_30658


namespace distinct_prime_factors_of_B_l30_30473

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30473


namespace intersection_M_N_l30_30338

def M := { x : ℤ | -x^2 + 3x > 0 }
def N := { x : ℝ | x^2 - 4 < 0 }

theorem intersection_M_N : M ∩ (N ∩ ℤ) = {1} := 
by sorry

end intersection_M_N_l30_30338


namespace orthocentres_form_right_angle_l30_30890

-- Definitions for the points and circles
variables {A B C D E : Point}
variables {r : ℝ}

-- Conditions: points are on a circle with radius r and AC = BD = CE = r
axiom h1 : circle_radius (circle A B C D E) = r
axiom h2 : distance A C = r
axiom h3 : distance B D = r
axiom h4 : distance C E = r

-- The theorem to prove
theorem orthocentres_form_right_angle 
(h1 : circle_radius (circle A B C D E) = r)
(h2 : distance A C = r) 
(h3 : distance B D = r) 
(h4 : distance C E = r) : 
right_angled_triangle (orthocentre A C D) (orthocentre B C D) (orthocentre B C E) := 
sorry

end orthocentres_form_right_angle_l30_30890


namespace alex_plays_with_friends_l30_30749

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30749


namespace expected_ones_three_standard_dice_l30_30060

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30060


namespace proof_problem_l30_30301

noncomputable def focus : ℝ × ℝ := (1, 0)

def is_chord (focus : ℝ × ℝ) (line : ℝ → ℝ) (parabola : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), line x₁ = focus.2 ∧ line x₂ = focus.2 ∧
    parabola (line x₁) = x₁ ∧ parabola (line x₂) = x₂

def perpendicular (line1 line2 : ℝ → ℝ) : Prop :=
  ∃ (k1 k2 : ℝ), (∀ x, line1 x = k1 * (x - 1)) ∧
                 (∀ x, line2 x = k2 * (x - 1)) ∧
                 k1 * k2 = -1

def parabola (y : ℝ) : ℝ := (y^2 : ℝ) / 4

theorem proof_problem :
  ∀ (AB DE : ℝ → ℝ),
  is_chord focus AB parabola ∧
  is_chord focus DE parabola ∧
  perpendicular AB DE →
  (1 / (abs (AB (parabola(focus.2)))) + 1 / (abs (DE (parabola(focus.2))))) = 1 / 4 :=
by
  sorry

end proof_problem_l30_30301


namespace distinct_prime_factors_of_B_l30_30454

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30454


namespace coupon_savings_difference_l30_30816

theorem coupon_savings_difference (P : ℝ) (hP : P > 120) :
  let p := P - 120 in
  20 * P / 100 ≥ 35 → 20 * P / 100 ≥ 30 * p / 100 →
  ∃ x y : ℝ, x ≥ 175 ∧ y ≤ 360 ∧ (y - x = 185) :=
by
  sorry

end coupon_savings_difference_l30_30816


namespace intersection_is_correct_l30_30337

noncomputable def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
noncomputable def B := { x : ℝ | 0 < x ∧ x ≤ 3 }

theorem intersection_is_correct : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_is_correct_l30_30337


namespace expected_ones_in_three_dice_rolls_l30_30046

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30046


namespace alex_play_friends_with_l30_30735

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30735


namespace smallest_palindromic_integer_l30_30211

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n
  digits = List.reverse digits

theorem smallest_palindromic_integer :
  ∃ (n : ℕ), n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ (∀ m, m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 → n ≤ m) :=
begin
  sorry
end

end smallest_palindromic_integer_l30_30211


namespace percentage_of_third_number_l30_30157

theorem percentage_of_third_number (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A / C = 0.06 := 
by
  sorry

end percentage_of_third_number_l30_30157


namespace incorrect_statement_B_l30_30722

def f (x : ℝ) := log x (x + 1)

theorem incorrect_statement_B (a : ℝ) (x : ℝ) (hx : x > 0 ∧ x ≠ 1) : 
  ¬ ∀ a ≠ 0 ∧ a ≠ 1, ∃ x : ℝ, f x = a :=
by 
  sorry

end incorrect_statement_B_l30_30722


namespace intersection_M_N_l30_30928

noncomputable def f (x : ℝ) := Real.log (1 + x)
noncomputable def g (x : ℝ) := 2^x + 1

def M : Set ℝ := {x | -1 < x}
def N : Set ℝ := {y | 1 < y}

theorem intersection_M_N : M ∩ N = {y | 1 < y} :=
by 
  unfold M N
  ext y 
  simp
sorry

end intersection_M_N_l30_30928


namespace CDHcHd_is_parallelogram_l30_30615

-- Define points and quadrilateral
variables {A B C D H_c H_d : Type}

-- Assume ABCD is an inscribed quadrilateral
axiom inscribed_quad (AB : A ≠ B) (BC : B ≠ C) (CD : C ≠ D) (DA : D ≠ A) (A_angle_eq : ∡A ≠ ∡B) : ABCD ∈ cyclic_quadrilaterals

-- Assume H_c is the orthocenter of triangle ABD
axiom Hc_orthocenter : orthocenter (triangle A B D) H_c

-- Assume H_d is the orthocenter of triangle ABC
axiom Hd_orthocenter : orthocenter (triangle A B C) H_d

-- Prove that CDH_cH_d is a parallelogram
theorem CDHcHd_is_parallelogram :
  parallelogram (quadrilateral C D H_c H_d) :=
by
sorry

end CDHcHd_is_parallelogram_l30_30615


namespace sum_of_binomial_coefficients_excluding_a0_l30_30625

theorem sum_of_binomial_coefficients_excluding_a0 :
  let a := (2 - x)^10
  ∑ k in range (10 + 1), coeff (2 - x)^10 k * x^k - coeff (2 - x)^10 0 * x^0 = -1023 :=
by
  sorry

end sum_of_binomial_coefficients_excluding_a0_l30_30625


namespace range_of_g_l30_30260

-- Define the function g(x)
def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arccot x

-- Define the range of the function g(x)
def range_g : Set ℝ := { y | ∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ y = g x }

-- State the theorem to prove the range of g(x)
theorem range_of_g : range_g = Set.Icc (3 * Real.pi / 4) (5 * Real.pi / 4) := sorry

end range_of_g_l30_30260


namespace log_base_three_of_one_over_nine_eq_neg_two_l30_30245

theorem log_base_three_of_one_over_nine_eq_neg_two :
  (∃ y, y = Real.log 3 (1 / 9)) → ∃ y, y = -2 :=
by
  sorry

end log_base_three_of_one_over_nine_eq_neg_two_l30_30245


namespace angle_C_triangle_area_l30_30366

theorem angle_C 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) :
  C = 2 * Real.pi / 3 :=
sorry

theorem triangle_area 
  (a b c : ℝ) (C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C)
  (h2 : c = Real.sqrt 7)
  (h3 : b = 2) :
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 :=
sorry

end angle_C_triangle_area_l30_30366


namespace electromagnetic_storm_time_l30_30600

structure Time :=
(hh : ℕ)
(mm : ℕ)
(valid_hour : 0 ≤ hh ∧ hh < 24)
(valid_minute : 0 ≤ mm ∧ mm < 60)

def possible_digits (d : ℕ) : set ℕ :=
  {x | x = d + 1 ∨ x = d - 1}

theorem electromagnetic_storm_time :
  (∃ t : Time, t.hh = 18 ∧ t.mm = 18) →
  (∀ (orig : Time), 
    orig.hh ∈ possible_digits 0 ∧ 
    (orig.hh % 10) ∈ possible_digits 9 ∧
    orig.mm ∈ possible_digits 0 ∧ 
    (orig.mm % 10) ∈ possible_digits 9 →
      false) :=
by
  sorry

end electromagnetic_storm_time_l30_30600


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30561

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30561


namespace triangle_area_l30_30844

/-- A math problem to prove the area of triangle ABC given centers of circles and their radii. -/
theorem triangle_area (A B C : (ℝ × ℝ)) (A_r B_r C_r : ℝ)
  (hA : A_r = 2) (hB : B_r = 3) (hC : C_r = 4)
  (hB_coords : B = (3, 3))
  (hDistances : dist A B = A_r + B_r ∧ dist B C = B_r + C_r) :
  let area := ½ * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|
  in area = 1 := by
  sorry

end triangle_area_l30_30844


namespace find_fourth_vertex_of_square_l30_30996

-- Given the vertices of the square as complex numbers
def vertex1 : ℂ := 1 + 2 * Complex.I
def vertex2 : ℂ := -2 + Complex.I
def vertex3 : ℂ := -1 - 2 * Complex.I

-- The fourth vertex (to be proved)
def vertex4 : ℂ := 2 - Complex.I

-- The mathematically equivalent proof problem statement
theorem find_fourth_vertex_of_square :
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  -- Define vectors from the vertices
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4
  vector_ab = vector_dc :=
by {
  -- Definitions already provided above
  let v1 := vertex1
  let v2 := vertex2
  let v3 := vertex3
  let v4 := vertex4
  let vector_ab := v2 - v1
  let vector_dc := v3 - v4

  -- Placeholder for proof
  sorry
}

end find_fourth_vertex_of_square_l30_30996


namespace distinct_prime_factors_B_l30_30547

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30547


namespace inverse_proportion_function_l30_30363

theorem inverse_proportion_function (m x : ℝ) (h : (m ≠ 0)) (A : (m, m / 8) ∈ {p : ℝ × ℝ | p.snd = (m / p.fst)}) :
    ∃ f : ℝ → ℝ, (∀ x, f x = 8 / x) :=
by
  use (fun x => 8 / x)
  intros x
  rfl

end inverse_proportion_function_l30_30363


namespace volume_calculation_l30_30850

noncomputable def volume_parallelepiped_inclusive : ℝ :=
  let base_volume := 2 * 3 * 7
  let extended_faces_volume := 2 * (1 * 2 * 7 + 1 * 3 * 7 + 1 * 2 * 3)
  let quarter_cylinders_volume := 4 * (1 / 4 * Real.pi * 1^2) * (2 + 3 + 7)
  let eighth_spheres_volume := 8 * (1 / 8 * 4 / 3 * Real.pi * 1^3)
  (base_volume + extended_faces_volume + quarter_cylinders_volume + eighth_spheres_volume)

theorem volume_calculation : 
  let volume := volume_parallelepiped_inclusive in
  volume = 372 + 112 * Real.pi / 3 → 
  let m := 372
  let n := 112
  let p := 3
  (m + n + p) = 487 :=
by
  sorry

end volume_calculation_l30_30850


namespace second_month_earnings_relationship_l30_30347

theorem second_month_earnings_relationship :
  ∃ (x : ℝ), 350 + x + 4 * (350 + x) = 5500 ∧ x / 350 ≈ 2.14 :=
by
  sorry

end second_month_earnings_relationship_l30_30347


namespace expected_value_of_ones_on_three_dice_l30_30050

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30050


namespace tan_angle_sum_l30_30321

-- Given conditions
def point (α : ℝ) : Prop :=
  let P := (-1, 2) in 
  True  -- Given that P lies on the terminal side of angle α (indirectly stating we can find tan(α) from this)

-- The proof statement we need to show
theorem tan_angle_sum (α : ℝ) (h : point α) : Real.tan (α + Real.pi / 4) = -1 / 3 :=
  sorry

end tan_angle_sum_l30_30321


namespace range_values_pf1_pf2_l30_30609

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem range_values_pf1_pf2 :
  let P_line (x : ℝ) := (x, (3/4) * x)
  ∀ (P : ℝ × ℝ), P = P_line P.1 →
  let PF1 := distance P (−5, 0)
  let PF2 := distance P (5, 0)
  0 ≤ |PF1 - PF2| ∧ |PF1 - PF2| ≤ 8.5 :=
  sorry

end range_values_pf1_pf2_l30_30609


namespace imaginary_part_of_conjugate_l30_30901

def complex_number_problem (z : ℂ) : Prop :=
  z = (2 - 3 * complex.I) / complex.I → complex.im (conj z) = 2

theorem imaginary_part_of_conjugate :
  complex_number_problem ((2 - 3 * complex.I) / complex.I) :=
by
  sorry

end imaginary_part_of_conjugate_l30_30901


namespace percentage_decrease_in_breadth_l30_30821

theorem percentage_decrease_in_breadth :
  ∀ (L B : ℝ), (L > 0) → (B > 0) →
  ∃ (x : ℝ), 
    let initial_area := L * B in
    let final_length := 0.7 * L in
    let final_area := 0.42 * initial_area in
    let final_breadth := (1 - x / 100) * B in
    (final_length * final_breadth = final_area) → x = 40 := 
by
  intros L B hLpos hBpos
  use 40
  sorry

end percentage_decrease_in_breadth_l30_30821


namespace coloring_time_saved_percentage_l30_30396

variable (n : ℕ := 10) -- number of pictures
variable (draw_time : ℝ := 2) -- time to draw each picture in hours
variable (total_time : ℝ := 34) -- total time spent on drawing and coloring in hours

/-- 
  Prove the percentage of time saved on coloring each picture compared to drawing 
  given the specified conditions.
-/
theorem coloring_time_saved_percentage (n : ℕ) (draw_time total_time : ℝ) 
  (h1 : draw_time > 0)
  (draw_total_time : draw_time * n = 20)
  (total_picture_time : draw_time * n + coloring_total_time = total_time) :
  (draw_time - (coloring_total_time / n)) / draw_time * 100 = 30 := 
by
  sorry

end coloring_time_saved_percentage_l30_30396


namespace tangent_slope_at_point_through_1_0_l30_30168

open Real

def curve (x : ℝ) : ℝ := exp (x - 1)

def tangent_slope_at_point : ℝ :=
  let x0 := 2 in
  deriv curve x0

theorem tangent_slope_at_point_through_1_0 :
  tangent_slope_at_point = exp 1 := sorry

end tangent_slope_at_point_through_1_0_l30_30168


namespace distinct_prime_factors_of_B_l30_30451

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30451


namespace function_properties_l30_30636

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ x y, x ∈ I → y ∈ I → x < y → f x > f y

theorem function_properties :
  let f := λ x : ℝ, exp (-x) - exp x in
  is_odd_function f ∧ is_decreasing_on f (set.Ioi 0) :=
by
  sorry

end function_properties_l30_30636


namespace problem_l30_30403

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30403


namespace find_original_number_l30_30805

theorem find_original_number (x : ℕ) 
    (h1 : (73 * x - 17) / 5 - (61 * x + 23) / 7 = 183) : x = 32 := 
by
  sorry

end find_original_number_l30_30805


namespace geometric_sequence_common_ratio_l30_30272

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 2 * a 0 + a 1 = a 2)
  : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l30_30272


namespace scientific_notation_of_number_l30_30145

theorem scientific_notation_of_number : 15300000000 = 1.53 * (10 : ℝ)^10 := sorry

end scientific_notation_of_number_l30_30145


namespace alex_play_friends_with_l30_30732

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30732


namespace target_percentage_is_six_l30_30189

-- Definitions based on the problem's conditions
def initial_investment : ℝ := 1400
def initial_rate : ℝ := 0.05
def additional_investment : ℝ := 700
def additional_rate : ℝ := 0.08
def total_investment : ℝ := initial_investment + additional_investment
def total_income : ℝ := (initial_rate * initial_investment) + (additional_rate * additional_investment)
def target_percentage := (total_income / total_investment) * 100

-- Lean statement to prove the target percentage is 6%
theorem target_percentage_is_six : target_percentage = 6 :=
by
  sorry

end target_percentage_is_six_l30_30189


namespace triangle_is_isosceles_right_l30_30285

theorem triangle_is_isosceles_right
  (a b c : ℝ)
  (A B C : ℕ)
  (h1 : c = a * Real.cos B)
  (h2 : b = a * Real.sin C) :
  C = 90 ∧ B = 90 ∧ A = 90 :=
sorry

end triangle_is_isosceles_right_l30_30285


namespace locus_eq_l30_30916

def line_eq (x y : ℝ) : Prop := (x / 4) + (y / 3) = 1

def point_on_line (M : ℝ × ℝ) (x y : ℝ) : Prop :=
  line_eq (M.1) (M.2)

def perpendicular_projection (M : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  ((M.1, 0), (0, M.2))

def point_on_segment (A B P : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ P = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

def vector_eq (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  (P.1 - A.1, P.2 - A.2) = (2 * (-P.1), 2 * (B.2 - P.2))

theorem locus_eq {x y m n : ℝ} (M : ℝ × ℝ) (P : ℝ × ℝ)
  (hM : point_on_line M x y)
  (hm : M = (3 * x, (3 * y) / 2))
  (hP : point_on_segment (M.1, 0) (0, M.2) P)
  (hAP : vector_eq P (M.1, 0) (0, M.2)) :
  (3 * x / 4) + (y / 2) = 1 :=
sorry

end locus_eq_l30_30916


namespace infinite_sum_evaluation_l30_30248

noncomputable def infinite_sum := ∑' n : ℕ, if n = 0 then 0 else n / (n ^ 4 + 4 * n ^ 2 + 1)

theorem infinite_sum_evaluation : infinite_sum = Real.sqrt π / 2 :=
sorry

end infinite_sum_evaluation_l30_30248


namespace find_line_eq_l30_30254

-- Define the type for the line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def given_point : ℝ × ℝ := (-3, -1)
def given_parallel_line : Line := { a := 1, b := -3, c := -1 }

-- Define what it means for two lines to be parallel
def are_parallel (L1 L2 : Line) : Prop :=
  L1.a * L2.b = L1.b * L2.a

-- Define what it means for a point to lie on the line
def lies_on_line (P : ℝ × ℝ) (L : Line) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

-- Define the result line we need to prove
def result_line : Line := { a := 1, b := -3, c := 0 }

-- The final theorem statement
theorem find_line_eq : 
  ∃ (L : Line), are_parallel L given_parallel_line ∧ lies_on_line given_point L ∧ L = result_line := 
sorry

end find_line_eq_l30_30254


namespace replace_question_with_division_l30_30862

theorem replace_question_with_division :
  ∃ op: (ℤ → ℤ → ℤ), (op 8 2) + 5 - (3 - 2) = 8 ∧ 
  (∀ a b, op = Int.div ∧ ((op a b) = a / b)) :=
by
  sorry

end replace_question_with_division_l30_30862


namespace angles_on_coordinate_axes_l30_30674

theorem angles_on_coordinate_axes :
  let Sx := {α | ∃ k : ℤ, α = k * Real.pi},
      Sy := {α | ∃ k : ℤ, α = k * Real.pi + (Real.pi / 2)} in
  (Sx ∪ Sy) = {α | ∃ n : ℤ, α = (n * Real.pi) / 2} :=
by
  sorry

end angles_on_coordinate_axes_l30_30674


namespace max_log_product_l30_30288

theorem max_log_product (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : log 10 x + log 10 y = 4) :
  ∃ (M : ℝ), M = 4 ∧ ∀ (u v : ℝ), (log 10 u + log 10 v = 4) → u > 1 → v > 1 → log 10 u * log 10 v ≤ M :=
begin
  sorry
end

end max_log_product_l30_30288


namespace smallest_palindrome_in_base2_and_4_l30_30221

-- Define a function to check if a number's representation is a palindrome.
def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

-- Convert a number n to a given base and represent as a list of digits.
def to_base (n : ℕ) (b : ℕ) : List ℕ :=
  if h : b > 1 then
    let rec convert (n : ℕ) : List ℕ :=
      if n = 0 then [] else (n % b) :: convert (n / b)
    convert n 
  else []

-- Convert the list of digits to a list of characters.
def digits_to_chars (ds : List ℕ) : List Char :=
  ds.map (λ d => (Char.ofNat (d + 48))) -- Adjust ASCII value for digit representation

-- Define a function to check if a number is a palindrome in a specified base.
def is_palindromic_in_base (n base : ℕ) : Bool :=
  is_palindrome (digits_to_chars (to_base n base))

-- Define the main claim.
theorem smallest_palindrome_in_base2_and_4 : ∃ n, n > 10 ∧ is_palindromic_in_base n 2 ∧ is_palindromic_in_base n 4 ∧ ∀ m, m > 10 ∧ is_palindromic_in_base m 2 ∧ is_palindromic_in_base m 4 -> n ≤ m :=
by
  exists 15
  sorry

end smallest_palindrome_in_base2_and_4_l30_30221


namespace alex_plays_with_friends_l30_30755

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30755


namespace parabola_eq_chord_length_l30_30948

theorem parabola_eq_chord_length
  (p : ℝ) (A B : ℝ × ℝ)
  (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x)
  (h_line_angle : ∃ k : ℝ, k = 1)
  (h_midpoint : ∃ (Q : ℝ × ℝ), Q = (3, 2) ∧ (fst A + snd A)/2 = fst Q ∧ (fst B + snd B)/2 = snd Q)
  (h_points_A : ∃ (x₁ y₁ : ℝ), A = (x₁, y₁))
  (h_points_B : ∃ (x₂ y₂ : ℝ), B = (x₂, y₂))
  (h_ab_mid : fst A + fst B = 6 ∧ snd A + snd B = 4) :
  (p = 2 ∧ (∀ (x y : ℝ), y^2 = 4 * x) ∧ (|fst A - fst B| = 8)) := by
  sorry

end parabola_eq_chord_length_l30_30948


namespace athlete_difference_l30_30806

-- Define the conditions
def initial_athletes : ℕ := 300
def rate_of_leaving : ℕ := 28
def time_of_leaving : ℕ := 4
def rate_of_arriving : ℕ := 15
def time_of_arriving : ℕ := 7

-- Define intermediary calculations
def number_leaving : ℕ := rate_of_leaving * time_of_leaving
def remaining_athletes : ℕ := initial_athletes - number_leaving
def number_arriving : ℕ := rate_of_arriving * time_of_arriving
def total_sunday_night : ℕ := remaining_athletes + number_arriving

-- Theorem statement
theorem athlete_difference : initial_athletes - total_sunday_night = 7 :=
by
  sorry

end athlete_difference_l30_30806


namespace length_of_BE_l30_30608

/-- 
Given a parallelogram ABCD:
- Point F is on the extension of side AD.
- Line BF intersects diagonal AC at point E and side DC at point G.
- Length EF = 40.
- Length GF = 30.

We need to prove the length of BE is 20.
-/
theorem length_of_BE {A B C D E F G : Type}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  [MetricSpace E] [MetricSpace F] [MetricSpace G]
  [AD_line_extends_to_F : Line A D F]
  [BF_intersects_AC_at_E : Intersect B F A C E]
  [BF_intersects_DC_at_G : Intersect B F D C G]
  (EF : dist E F = 40)
  (GF : dist G F = 30) :
  dist B E = 20 := by
  sorry

end length_of_BE_l30_30608


namespace is_quadratic_l30_30123

theorem is_quadratic (A B C D : Prop) :
  (A = (∀ x : ℝ, x + (1 / x) = 0)) ∧
  (B = (∀ x y : ℝ, x + x * y + 1 = 0)) ∧
  (C = (∀ x : ℝ, 3 * x + 2 = 0)) ∧
  (D = (∀ x : ℝ, x^2 + 2 * x = 1)) →
  D := 
by
  sorry

end is_quadratic_l30_30123


namespace janet_initial_number_l30_30393

-- Define the conditions using Lean definitions
def janetProcess (x : ℕ) : ℕ :=
  (2 * (x + 7)) - 4

-- The theorem that expresses the statement of the problem: If the final result of the process is 28, then x = 9
theorem janet_initial_number (x : ℕ) (h : janetProcess x = 28) : x = 9 :=
sorry

end janet_initial_number_l30_30393


namespace range_of_f_when_a_0_range_of_a_for_three_zeros_l30_30332

noncomputable def f_part1 (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x else x ^ 2

theorem range_of_f_when_a_0 : Set.range f_part1 = {y : ℝ | 0 < y} := by
  sorry

noncomputable def f_part2 (a : ℝ) (x : ℝ) : ℝ :=
if h : x ≤ 0 then 2 ^ x - a else x ^ 2 - 3 * a * x + a

def discriminant (a : ℝ) (x : ℝ) : ℝ := (3 * a) ^ 2 - 4 * a

theorem range_of_a_for_three_zeros (a : ℝ) :
  (∀ x : ℝ, f_part2 a x = 0) → (4 / 9 < a ∧ a ≤ 1) := by
  sorry

end range_of_f_when_a_0_range_of_a_for_three_zeros_l30_30332


namespace probability_x_squared_minus_2x_minus_3_leq_0_l30_30161

theorem probability_x_squared_minus_2x_minus_3_leq_0 : 
  let interval := set.Icc (-4 : ℝ) 4,
      subinterval := set.Icc (-1 : ℝ) 3
  in (subinterval.nonempty ∧ subinterval ⊆ interval) → 
     ((subinterval.measure / interval.measure) = (1 / 2)) :=
by
  sorry

end probability_x_squared_minus_2x_minus_3_leq_0_l30_30161


namespace percentage_profit_l30_30812

variable (total_crates : ℕ)
variable (total_cost : ℕ)
variable (lost_crates : ℕ)
variable (sell_price_per_crate : ℕ)

theorem percentage_profit (h1 : total_crates = 10) (h2 : total_cost = 160)
  (h3 : lost_crates = 2) (h4 : sell_price_per_crate = 25) :
  (8 * sell_price_per_crate - total_cost) * 100 / total_cost = 25 :=
by
  -- Definitions and steps to prove this can be added here.
  sorry

end percentage_profit_l30_30812


namespace friends_who_participate_l30_30772

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30772


namespace distinct_prime_factors_product_divisors_60_l30_30488

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30488


namespace expected_ones_three_standard_dice_l30_30059

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30059


namespace area_of_triangle_intersection_l30_30925

noncomputable def F1 : (ℝ × ℝ) := (-5/2, 0)
noncomputable def F2 : (ℝ × ℝ) := (5/2, 0)

def isOnHyperbola (P : ℝ × ℝ) (m : ℝ) : Prop :=
  P.1^2 / 4 - P.2^2 / m = 1

def isOnEllipse (P : ℝ × ℝ) (n : ℝ) : Prop :=
  P.1^2 / 9 + P.2^2 / n = 1

noncomputable def areaTrianglePF1F2 (P : ℝ × ℝ) : ℝ :=
  let | (x1, y1), (x2, y2), (x3, y3) := (P, F1, F2)
  let a := Math.sqrt((x2 - x1)^2 + (y2 - y1)^2)
  let b := Math.sqrt((x3 - x1)^2 + (y3 - y1)^2)
  let c := Math.sqrt((x3 - x2)^2 + (y3 - y2)^2)
  let s := (a + b + c) / 2
  Math.sqrt(s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_intersection
  {P : ℝ × ℝ}
  (m n : ℝ)
  (h1 : isOnHyperbola P m)
  (h2 : isOnEllipse P n)
  (h3 : m - n = 4)
  (h4 : m + n = 6) : 
  areaTrianglePF1F2 P = 3 * Math.sqrt(11) / 4 :=
sorry

end area_of_triangle_intersection_l30_30925


namespace number_of_distinct_prime_factors_of_B_l30_30538

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30538


namespace expected_ones_three_standard_dice_l30_30056

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30056


namespace hyperbola_eccentricity_l30_30290

-- Definitions based on the problem's conditions
def circle : set (ℝ × ℝ) := { p | (p.1 - real.sqrt 2)^2 + p.2^2 = 1 }

def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : set (ℝ × ℝ) :=
  { p | p.1^2 / a^2 - p.2^2 / b^2 = 1 }

-- Defining the problem statement
theorem hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (tangent_to_same_asymptote : ∀ x y, x.1^2 / a^2 - x.2^2 / b^2 = 1 → (x.1 - real.sqrt 2)^2 + x.2^2 = 1 → ∥x - y∥ = 0)
  : ∃ e : ℝ, e = real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l30_30290


namespace find_a8_l30_30589

noncomputable def a : ℕ → ℝ
| 0     => 3
| (n+1) => (b n)^2 / a n

noncomputable def b : ℕ → ℝ
| 0     => 5
| (n+1) => (a n)^2 / b n

theorem find_a8 :
  a 8 = (5 ^ (3 ^ 8)) / (3 ^ (1 + 3 ^ 8)) := 
sorry

end find_a8_l30_30589


namespace expected_ones_on_three_dice_l30_30090

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30090


namespace find_radius_l30_30998

-- Definitions based on given conditions
def TP := 5
def TQ := 12
def angle := 30

-- Hypothetical radius
variable r : ℝ

-- Lean statement to prove r = 5/2 based on given conditions
theorem find_radius (h₁ : TP = 5) (h₂ : TQ = 12) (h₃ : angle = 30) : r = 5 / 2 := 
by 
  sorry

end find_radius_l30_30998


namespace rectangle_diagonal_length_l30_30646

theorem rectangle_diagonal_length (P : ℝ) (r : ℝ) (perimeter_eq : P = 72) (ratio_eq : r = 5/2) :
  let k := P / 14
  let l := 5 * k
  let w := 2 * k
  let d := Real.sqrt (l^2 + w^2)
  d = 194 / 7 :=
sorry

end rectangle_diagonal_length_l30_30646


namespace part_a_l30_30698

theorem part_a (A B C : Type) (hits : ℕ) (original_positions : list (A × B × C)) :
  hits = 7 → pucks_return_after_hits original_positions hits → false :=
by
  -- Assuming the required conditions and definitions
  sorry

end part_a_l30_30698


namespace maximum_partial_sum_l30_30225

theorem maximum_partial_sum (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
    (h_arith_seq : ∀ n, a n = a 0 + n * d)
    (h8_13 : 3 * a 8 = 5 * a 13)
    (h_pos : a 0 > 0)
    (h_sn_def : ∀ n, S n = n * (2 * a 0 + (n - 1) * d) / 2) :
  S 20 = max (max (S 10) (S 11)) (max (S 20) (S 21)) := 
sorry

end maximum_partial_sum_l30_30225


namespace distinct_prime_factors_of_B_l30_30480

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30480


namespace thirteen_cards_sufficient_twelve_cards_insufficient_l30_30811

def card := ℕ × ℕ

def mark_cells (c : Set card) (n : ℕ) : Prop := c.card = n

def losing_cells (c : Set card) : Prop := c.card = 10

def is_winning (c_marked c_losing : Set card) : Prop :=
  ∀ x ∈ c_marked, x ∉ c_losing

theorem thirteen_cards_sufficient :
  ∃ (cards : Fin 13 → Set card),
  (∀ i, mark_cells (cards i) 10) ∧
  ∀ l_set, losing_cells l_set → ∃ i, is_winning (cards i) l_set :=
sorry

theorem twelve_cards_insufficient :
  ∀ (cards : Fin 12 → Set card),
  (∀ i, mark_cells (cards i) 10) →
  ∃ l_set, losing_cells l_set ∧ ∀ i, ¬ is_winning (cards i) l_set :=
sorry

end thirteen_cards_sufficient_twelve_cards_insufficient_l30_30811


namespace candidate_loss_approx_833_l30_30794

theorem candidate_loss_approx_833 (V : ℝ) (hV : V ≈ 2450) :
    0.34 * V ≈ 833 := sorry

end candidate_loss_approx_833_l30_30794


namespace second_team_size_proof_l30_30693

noncomputable def number_of_people_in_second_team (first_team_size : ℕ) (second_team_size : ℚ) (hours_first_team_alone : ℚ) (hours_both_teams_together : ℚ) : Prop :=
  let total_worker_hours := first_team_size * hours_first_team_alone in
  let total_people := first_team_size + second_team_size in
  total_worker_hours = total_people * hours_both_teams_together

theorem second_team_size_proof (n : ℚ) : number_of_people_in_second_team 4 n 8 3 := by
  let total_worker_hours := 4 * 8
  have h1 : total_worker_hours = 32 := by norm_num
  let total_people := 4 + n
  let total_hours := total_people * 3
  have h2 : 32 = total_hours := by sorry
  have h3 : n = (20 : ℚ) / 3 := by sorry
  show number_of_people_in_second_team 4 n 8 3 from by sorry

end second_team_size_proof_l30_30693


namespace tax_free_amount_is_600_l30_30820

variable ( total_value : ℝ ) (tax_paid : ℝ )
variable ( tax_rate : ℝ := 0.10 )
variable ( tax_free_amount : ℝ )

-- Conditions from the problem
axiom total_value_is_1720 : total_value = 1720
axiom tax_paid_is_112 : tax_paid = 112
axiom tax_rate_is_0_10 : tax_rate = 0.10

-- The equation from the condition
axiom tax_calculation : tax_paid = tax_rate * (total_value - tax_free_amount)

-- The proof goal
theorem tax_free_amount_is_600 :
  tax_free_amount = 600 :=
by
  rw [total_value_is_1720, tax_paid_is_112, tax_rate_is_0_10] at tax_calculation
  sorry

end tax_free_amount_is_600_l30_30820


namespace two_n_plus_m_value_l30_30637

theorem two_n_plus_m_value (n m : ℤ) :
  3 * n - m < 5 ∧ n + m > 26 ∧ 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
sorry

end two_n_plus_m_value_l30_30637


namespace total_change_in_percentage_is_correct_l30_30173

def percentage_change_from_fall_to_spring (initial : ℕ) : ℝ :=
  let fall_increase := initial * 1.09
  let spring_decrease := fall_increase * 0.81
  ((spring_decrease - initial) / initial) * 100

theorem total_change_in_percentage_is_correct :
  percentage_change_from_fall_to_spring 100 = -11.71 :=
by
  sorry

end total_change_in_percentage_is_correct_l30_30173


namespace trapezoid_area_l30_30385

theorem trapezoid_area (ABCD : Trapezoid) (ω : Circle) (L : Point)
  (h_inscribed : inscribed_circle ABCD ω)
  (h_tangency : tangency_point ω CD L)
  (h_ratio : CL / LD = 1 / 4)
  (h_BC : BC = 9)
  (h_CD : CD = 30) :
  area ABCD = 972 := sorry

end trapezoid_area_l30_30385


namespace tangent_line_inclination_angle_at_minus_one_l30_30874

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + 2 * x - 1

theorem tangent_line_inclination_angle_at_minus_one :
  let point := (-1, f (-1))
  let slope := deriv f (-1)
  slope = 1 ∧ arctan slope = real.pi / 4 :=
by sorry

end tangent_line_inclination_angle_at_minus_one_l30_30874


namespace pow_mod_remainder_l30_30884

theorem pow_mod_remainder (n : ℕ) : 5 ^ 2023 % 11 = 4 :=
by sorry

end pow_mod_remainder_l30_30884


namespace distinct_prime_factors_B_l30_30553

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30553


namespace rectangle_diagonal_l30_30661

theorem rectangle_diagonal (k : ℚ)
  (h1 : 2 * (5 * k + 2 * k) = 72)
  (h2 : k = 36 / 7) :
  let l := 5 * k,
      w := 2 * k,
      d := Real.sqrt ((l ^ 2) + (w ^ 2)) in
  d = 194 / 7 := by
  sorry

end rectangle_diagonal_l30_30661


namespace max_value_f_when_a_eq_1_min_value_max_f_on_interval_range_possible_values_for_a_l30_30945

section

variable (a : ℝ) (x : ℝ)

-- Condition 1
def f (x : ℝ) (a : ℝ) := |x^2 - a|
def g (x : ℝ) (a : ℝ) := x^2 - a * x

-- Proof of maximum value of f(x) for a=1
theorem max_value_f_when_a_eq_1 : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x 1 ≤ 1 := by
  sorry

-- Proof of minimum value of the maximum value of f(x) on [-1,1]
theorem min_value_max_f_on_interval :
  ∃ a : ℝ, ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x a ≤ max (|x^2 - a|) := by
  sorry

-- Proof of the range of possible values for a given f(x) + g(x) = 0
theorem range_possible_values_for_a :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ x1 ≠ x2 ∧ f x1 a + g x1 a = 0 ∧ f x2 a + g x2 a = 0) →
  1 ≤ a ∧ a ≤ 8 / 3 := by
  sorry

end

end max_value_f_when_a_eq_1_min_value_max_f_on_interval_range_possible_values_for_a_l30_30945


namespace count_irrational_numbers_l30_30178

def is_irrational (x : Real) : Prop := ¬ ∃ (a b : Int), b ≠ 0 ∧ x = a / b

-- Given list of numbers
def nums : List Real := [3.14159, -Real.cbrt 9, 0.131131113, -Real.pi, Real.sqrt 25, Real.cbrt 64, -1/7]

-- List of irrational numbers from provided list
def irrational_nums : List Real := [nums.nthLe 1 (by norm_num), nums.nthLe 2 (by norm_num), nums.nthLe 3 (by norm_num)]

theorem count_irrational_numbers : irrational_nums.length = 3 := 
by 
  -- Justification of each irrational number
  have h1 : is_irrational (nums.nthLe 1 (by norm_num)) := sorry,
  have h2 : is_irrational (nums.nthLe 2 (by norm_num)) := sorry,
  have h3 : is_irrational (nums.nthLe 3 (by norm_num)) := sorry,
  sorry

end count_irrational_numbers_l30_30178


namespace percentage_of_smartphone_boxes_l30_30373

theorem percentage_of_smartphone_boxes (total_boxes smartphone_boxes : ℕ)
  (h_total_boxes : total_boxes = 3680)
  (h_smartphone_boxes : smartphone_boxes = 2156) :
  (smartphone_boxes : ℚ) / total_boxes * 100 ≈ 58.59 :=
by
  rw [h_total_boxes, h_smartphone_boxes]
  sorry

end percentage_of_smartphone_boxes_l30_30373


namespace expected_ones_three_standard_dice_l30_30062

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30062


namespace hide_and_seek_friends_l30_30756

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30756


namespace floor_sub_le_l30_30574

theorem floor_sub_le : ∀ (x y : ℝ), ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ :=
by sorry

end floor_sub_le_l30_30574


namespace algebra_problem_l30_30355

theorem algebra_problem
  (x : ℝ)
  (h : 59 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = Real.sqrt 61 :=
sorry

end algebra_problem_l30_30355


namespace ben_remaining_bonus_l30_30839

theorem ben_remaining_bonus :
  let total_bonus := 1496
  let kitchen_expense := total_bonus * (1/22 : ℚ)
  let holiday_expense := total_bonus * (1/4 : ℚ)
  let gift_expense := total_bonus * (1/8 : ℚ)
  let total_expense := kitchen_expense + holiday_expense + gift_expense
  total_bonus - total_expense = 867 :=
by
  let total_bonus := 1496
  let kitchen_expense := total_bonus * (1/22 : ℚ)
  let holiday_expense := total_bonus * (1/4 : ℚ)
  let gift_expense := total_bonus * (1/8 : ℚ)
  let total_expense := kitchen_expense + holiday_expense + gift_expense
  have h1 : kitchen_expense = 68 := by sorry
  have h2 : holiday_expense = 374 := by sorry
  have h3 : gift_expense = 187 := by sorry
  have h4 : total_expense = 629 := by sorry
  show total_bonus - total_expense = 867 from by
    calc
      total_bonus - total_expense
      = 1496 - 629 : by rw [h4]
      ... = 867 : by sorry

end ben_remaining_bonus_l30_30839


namespace inv_g_inv_5_l30_30359

noncomputable def g (x : ℝ) : ℝ := 25 / (2 + 5 * x)
noncomputable def g_inv (y : ℝ) : ℝ := (15 - 10) / 25  -- g^{-1}(5) as shown in the derivation above

theorem inv_g_inv_5 : (g_inv 5)⁻¹ = 5 / 3 := by
  have h_g_inv_5 : g_inv 5 = 3 / 5 := by sorry
  rw [h_g_inv_5]
  exact inv_div 3 5

end inv_g_inv_5_l30_30359


namespace max_boxes_l30_30696

theorem max_boxes (sheets : ℕ) (bodies_per_sheet : ℕ) (lids_per_sheet : ℕ)
  (bodies_per_box : ℕ) (lids_per_box : ℕ) 
  (hb : bodies_per_sheet = 2) (hl : lids_per_sheet = 3) 
  (hbb : bodies_per_box = 1) (hlb : lids_per_box = 2) (total_sheets : ℕ)
  (htotal : total_sheets = 20) :
  ∃ (body_sheets lid_sheets : ℕ), body_sheets = 8 ∧ lid_sheets = 11 ∧
  body_sheets + lid_sheets = total_sheets ∧
  let body_boxes := body_sheets * bodies_per_sheet in
  let lid_boxes := lid_sheets * lids_per_sheet in
  let total_boxes := min (body_boxes / bodies_per_box) (lid_boxes / lids_per_box) in
  total_boxes = 16 :=
sorry

end max_boxes_l30_30696


namespace expected_ones_in_three_dice_rolls_l30_30041

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30041


namespace quadratic_function_property_minimum_value_property_l30_30934

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) + f (x - 1) = 2 * x^2 - 4 * x

def minimize_on_interval (f : ℝ → ℝ) (m : ℝ) : ℝ :=
  let I := set.Icc m (m + 2) in
  Inf (f '' I)

theorem quadratic_function_property :
  ∀ (f : ℝ → ℝ),
    is_quadratic f →
    satisfies_condition f →
    f = (λ x, x^2 - 2 * x - 1) :=
by
  sorry

theorem minimum_value_property (m : ℝ) :
  let f := (λ x : ℝ, x^2 - 2 * x - 1) in
  minimize_on_interval f m =
    (if m ≤ -1 then m^2 + 2 * m - 1
     else if -1 < m ∧ m < 1 then -2
     else m^2 - 2 * m - 1) :=
by
  sorry

end quadratic_function_property_minimum_value_property_l30_30934


namespace distinct_prime_factors_of_product_of_divisors_l30_30426

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30426


namespace expected_ones_three_dice_l30_30085

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30085


namespace family_ages_l30_30991

theorem family_ages 
  (youngest : ℕ)
  (middle : ℕ := youngest + 2)
  (eldest : ℕ := youngest + 4)
  (mother : ℕ := 3 * youngest + 16)
  (father : ℕ := 4 * youngest + 18)
  (total_sum : youngest + middle + eldest + mother + father = 90) :
  youngest = 5 ∧ middle = 7 ∧ eldest = 9 ∧ mother = 31 ∧ father = 38 := 
by 
  sorry

end family_ages_l30_30991


namespace problem_l30_30412

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30412


namespace beka_flies_more_l30_30835

-- Definitions
def beka_flight_distance : ℕ := 873
def jackson_flight_distance : ℕ := 563

-- The theorem we need to prove
theorem beka_flies_more : beka_flight_distance - jackson_flight_distance = 310 :=
by
  sorry

end beka_flies_more_l30_30835


namespace white_wins_with_perfect_play_l30_30127

-- Define the initial positions of the kings
def WhiteInitialPosition : (ℕ × ℕ) := (1, 1) -- a1 square
def BlackInitialPosition : (ℕ × ℕ) := (8, 8) -- h8 square

-- Define the distance as the number of king's moves
def king_distance (p1 p2 : (ℕ × ℕ)) : ℕ :=
  max (abs (p1.1 - p2.1)) (abs (p1.2 - p2.2))

-- Define the winning condition for White
def white_wins (p : (ℕ × ℕ)) : Prop :=
  p.1 = 8 ∨ p.2 = 8

-- Define the winning condition for Black
def black_wins (p : (ℕ × ℕ)) : Prop :=
  p.1 = 1 ∨ p.2 = 1

-- Define the game state and movement rules
structure GameState :=
  (white : (ℕ × ℕ)) -- Position of White king
  (black : (ℕ × ℕ)) -- Position of Black king
  (distance : ℕ)    -- Distance between the kings

-- Initial game state
def initial_state : GameState :=
  { white := WhiteInitialPosition, black := BlackInitialPosition, distance := 7 }

-- Predicate to determine if a move is valid.
def valid_move (gs : GameState) (new_white new_black : (ℕ × ℕ)) : Prop :=
  (king_distance gs.white new_white ≤ gs.distance) ∧
  (king_distance gs.black new_black ≤ gs.distance) ∧
  (king_distance new_white new_black ≤ gs.distance)

-- Theorem statement: In the initial state, with perfect play, White will win.
theorem white_wins_with_perfect_play : ∀ (gs : GameState),
  gs = initial_state →
  (∃ (p : (ℕ × ℕ)), white_wins p) →
  (∀ (ws bs : (ℕ × ℕ)), valid_move gs ws bs) →
  (∃ (p : (ℕ × ℕ)), white_wins p) :=
by sorry

end white_wins_with_perfect_play_l30_30127


namespace PA_perp_BC_l30_30833

variables {A B C D E F G P : Point}
variables {O_B O_C : Circle}
variables (h_tangent_O_B : tangent O_B BC E)
variables (h_tangent_O_B' : tangent O_B BA F)
variables (h_tangent_O_C : tangent O_C CB D)
variables (h_tangent_O_C' : tangent O_C CA G)
variables (h_intersect : line_intersects DG EF P)
variables [TriangleExists : triangle A B C]

theorem PA_perp_BC 
  (exists_PA : ∃ P, line_intersects DG EF P) :
  perpendicular (line_through P A) BC :=
sorry

end PA_perp_BC_l30_30833


namespace mb_eq_five_l30_30008

theorem mb_eq_five (m b : ℝ) (h1 : b = -2) 
  (h2 : (1, -5) ∈ (λ x, m * x + b)) 
  (h3 : (-3, 5) ∈ (λ x, m * x + b)) : m * b = 5 := 
by 
  sorry

end mb_eq_five_l30_30008


namespace latin_student_sophomore_probability_l30_30372

variable (F S J SE : ℕ) -- freshmen, sophomores, juniors, seniors total
variable (FL SL JL SEL : ℕ) -- freshmen, sophomores, juniors, seniors taking latin
variable (p : ℚ) -- probability fraction
variable (m n : ℕ) -- relatively prime integers

-- Let the total number of students be 100 for simplicity in percentage calculations
-- Let us encode the given conditions
def conditions := 
  F = 40 ∧ 
  S = 30 ∧ 
  J = 20 ∧ 
  SE = 10 ∧ 
  FL = 40 ∧ 
  SL = S * 80 / 100 ∧ 
  JL = J * 50 / 100 ∧ 
  SEL = SE * 20 / 100

-- The probability calculation
def probability_sophomore (SL : ℕ) (FL SL JL SEL : ℕ) : ℚ := SL / (FL + SL + JL + SEL)

-- Target probability as a rational number
def target_probability := (6 : ℚ) / 19

theorem latin_student_sophomore_probability : 
  conditions F S J SE FL SL JL SEL → 
  probability_sophomore SL FL SL JL SEL = target_probability ∧ 
  m + n = 25 := 
by 
  sorry

end latin_student_sophomore_probability_l30_30372


namespace determine_t_l30_30323

noncomputable def circle (x y : ℝ) : Prop := x^2 + y^2 = 4
noncomputable def curve (x t : ℝ) : ℝ := 3 * |x - t|

variables {m n s p k t : ℝ}
variables (A : ℝ × ℝ) (B : ℝ × ℝ)
variables (x y : ℝ)
variables (on_circle : circle x y)

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1) * (P.1 - Q.1) + (P.2 - Q.2) * (P.2 - Q.2))

theorem determine_t
  (hA_on_curve : A.2 = curve A.1 t)
  (hB_on_curve : B.2 = curve B.1 t)
  (h_ratio : ∀ P : ℝ × ℝ, circle P.1 P.2 → distance P A = k * distance P B)
  (h_cond : 1 < k)
  : t = 4 / 3 :=
sorry

end determine_t_l30_30323


namespace carlson_candies_l30_30843

theorem carlson_candies (N : ℝ)
  (h1 : Carlson ate 20% of all the candies, of which 25% were caramels)
  (h2 : After that, Carlson ate three more chocolate candies)
  (h3 : The proportion of caramels among the candies Carlson had eaten decreased to 20%) :
  N = 60 :=
begin
  -- Definitions according to conditions:
  let total_initially_eaten := 0.2 * N,
  let caramels_initially_eaten := 0.25 * total_initially_eaten,
  let chocolates_initially_eaten := total_initially_eaten - caramels_initially_eaten,
  let carlson_ate := total_initially_eaten + 3,
  let caramel_fraction := caramels_initially_eaten / carlson_ate,

  -- Using given conditions:
  have fraction_car := caramel_fraction = 0.2, from sorry,
  have fryp : caramels_initially_eaten = 0.25 * 0.2 * N → caramels_initially_eaten = 0.05 * N, from sorry,
  have chocolat_ate := chocolates_initially_eaten = total_initially_eaten - caramels_initially_eaten, from sorry,
  have eqn := fraction_car.rec_on (by linarith),

  -- Solving for N:
  sorry  -- include the mathematical steps if necessary
end

end carlson_candies_l30_30843


namespace expected_ones_three_dice_l30_30079

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30079


namespace neg_p_sufficient_not_necessary_for_neg_q_l30_30357

variable (x : ℝ)
def p := x < -1
def q := x < -4

-- Statement: \neg p is a sufficient but not necessary condition for \neg q
theorem neg_p_sufficient_not_necessary_for_neg_q : ¬ p → ¬ q :=
by sorry

end neg_p_sufficient_not_necessary_for_neg_q_l30_30357


namespace number_of_distinct_prime_factors_of_B_l30_30441

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30441


namespace emmy_rosa_ipods_total_l30_30239

theorem emmy_rosa_ipods_total :
  ∃ (emmy_initial rosa_current : ℕ), 
    emmy_initial = 14 ∧ 
    (emmy_initial - 6) / 2 = rosa_current ∧ 
    (emmy_initial - 6) + rosa_current = 12 :=
by
  sorry

end emmy_rosa_ipods_total_l30_30239


namespace distinct_prime_factors_of_product_of_divisors_l30_30427

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30427


namespace range_of_linear_function_l30_30904

theorem range_of_linear_function (x : ℝ) (h : -1 < x ∧ x < 1) : 
  3 < -2 * x + 5 ∧ -2 * x + 5 < 7 :=
by {
  sorry
}

end range_of_linear_function_l30_30904


namespace range_of_a_l30_30317

theorem range_of_a (f : ℝ → ℝ) (h_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y)
  (h_exists : ∃ x0 : ℝ, f (|x0 + 1|) ≤ f (Real.log 2 a - |x0 + 2|)) : 2 ≤ a :=
by
  sorry

end range_of_a_l30_30317


namespace complement_of_A_in_U_l30_30341

def U : set ℝ := {x : ℝ | -3 < x ∧ x < 3}
def A : set ℝ := {x : ℝ | x^2 + x - 2 < 0}

theorem complement_of_A_in_U : (U \ A) = ({x : ℝ | -3 < x ∧ x ≤ -2} ∪ {x : ℝ | 1 ≤ x ∧ x < 3}) := by
  sorry

end complement_of_A_in_U_l30_30341


namespace expected_number_of_ones_l30_30068

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30068


namespace max_red_surface_area_l30_30030

theorem max_red_surface_area (total_cubes : ℕ) (adjacent_red_cubes : ℕ) (opposite_red_cubes : ℕ) 
  (cube_edge_length : ℕ) (red_faces_per_cube : ℕ) 
  (total_cubes = 64) (adjacent_red_cubes = 20) (opposite_red_cubes = 44) (cube_edge_length = 1) 
  (red_faces_per_cube = 2) :
  ∃ (max_red_area : ℕ), max_red_area = 76 :=
begin
  sorry
end

end max_red_surface_area_l30_30030


namespace find_f_at_1_l30_30330

theorem find_f_at_1 (m : ℝ) (h : ∀ x y : ℝ, x ≥ -2 → y ≥ -2 → x < y → 2 * y^2 - m * y + 5 ≥ 2 * x^2 - m * x + 5) : 
    (let f := λ x : ℝ, 2 * x^2 - m * x + 5 in f 1 = 15) :=
sorry

end find_f_at_1_l30_30330


namespace expected_number_of_ones_when_three_dice_rolled_l30_30035

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30035


namespace largest_two_digit_multiple_of_17_l30_30860

theorem largest_two_digit_multiple_of_17 : ∃ n, n ≤ 99 ∧ n ≥ 10 ∧ (n % 17 = 0) ∧ ∀ m, m ≤ 99 ∧ m ≥ 10 ∧ (m % 17 = 0) → m ≤ n :=
begin
  use 85,
  split, {
    norm_num,
  },
  split, {
    norm_num,
  },
  split, {
    norm_num,
  },
  intros m hm1 hm2 hm3,
  have hm : m ≤ 99 ∧ m ≥ 10 ∧ m % 17 = 0 := ⟨hm1, hm2, hm3⟩,
  calc m ≤ 85 : sorry,
end

end largest_two_digit_multiple_of_17_l30_30860


namespace part1_part2_l30_30005

-- Definitions for the given conditions
def prob_A1 : ℚ := 3/5
def prob_A2 : ℚ := 3/5
def prob_B1 : ℚ := 3/4
def prob_B2 : ℚ := 1/2

-- Proof problem formulation
theorem part1: 
  let prob_A := prob_A1 * prob_A2 in
  let prob_B := prob_B1 * prob_B2 in
  prob_B > prob_A := 
by
  sorry

theorem part2:
  let prob_A := prob_A1 * prob_A2 in
  let prob_B := prob_B1 * prob_B2 in
  let P_notA := 1 - prob_A in
  let P_notB := 1 - prob_B in
  1 - (P_notA * P_notB) = 3/5 := 
by
  sorry

end part1_part2_l30_30005


namespace athlete_difference_l30_30807

-- Define the conditions
def initial_athletes : ℕ := 300
def rate_of_leaving : ℕ := 28
def time_of_leaving : ℕ := 4
def rate_of_arriving : ℕ := 15
def time_of_arriving : ℕ := 7

-- Define intermediary calculations
def number_leaving : ℕ := rate_of_leaving * time_of_leaving
def remaining_athletes : ℕ := initial_athletes - number_leaving
def number_arriving : ℕ := rate_of_arriving * time_of_arriving
def total_sunday_night : ℕ := remaining_athletes + number_arriving

-- Theorem statement
theorem athlete_difference : initial_athletes - total_sunday_night = 7 :=
by
  sorry

end athlete_difference_l30_30807


namespace hide_and_seek_friends_l30_30759

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30759


namespace smallest_palindrome_in_bases_2_4_l30_30215

def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

def to_digits (n : Nat) (base : Nat) : List Char :=
  Nat.digits base n |> List.map (λ d, Char.ofNat (d + 48))

def is_palindrome_in_base (n base : Nat) : Bool :=
  is_palindrome (to_digits n base)

theorem smallest_palindrome_in_bases_2_4 :
  ∃ n : Nat, n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧
  ∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m :=
  ∃ n : Nat, n = 15 ∧ n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧ (∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m) :=
  sorry

end smallest_palindrome_in_bases_2_4_l30_30215


namespace find_difference_x_y_l30_30160

def cyclic_quadrilateral (a b c d : ℕ) := true -- Placeholder for the actual cyclic quadrilateral condition.
def circle_inscribed (a b c d : ℕ) := true -- Placeholder for the actual condition for an inscribed circle.

theorem find_difference_x_y :
  ∀ (a b c d x y : ℕ), 
  cyclic_quadrilateral a b c d → 
  circle_inscribed a b c d →
  a = 80 → b = 100 → c = 150 → d = 120 → 
  x + y = 150 → 
  |x - y| = 9 :=
by
  intros; sorry

end find_difference_x_y_l30_30160


namespace right_triangle_hypotenuse_length_l30_30163

theorem right_triangle_hypotenuse_length (x y : ℝ)
  (h1 : x + y = 7)
  (h2 : x * y = 12) : 
  (x^2 + y^2 = 25) :=
by {
  -- Begin proof, assume length of legs x and y satisfy the conditions given
  have h_leg1 : ∃ x y, x + y = 7 ∧ x * y = 12,
  {
    use 3,
    use 4,
    split,
    { exact h1 },
    { exact h2 }
  },
  -- use Pythagorean identity to prove hypotenuse length
  sorry,
}

end right_triangle_hypotenuse_length_l30_30163


namespace sum_a_is_100_l30_30326

def f (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n ^ 2) else n ^ 2
 
def a (n : ℕ) : ℤ := f n + f (n + 1)

theorem sum_a_is_100 : (∑ n in Finset.range 100, a (n + 1)) = 100 := 
by 
  sorry

end sum_a_is_100_l30_30326


namespace hyperbola_center_l30_30808

-- Define the points representing the foci
def F1 : ℝ × ℝ := (-3, 5)
def F2 : ℝ × ℝ := (1, -1)

-- Define a function to compute the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Prove that the midpoint of F1 and F2 is the center of the hyperbola
theorem hyperbola_center : midpoint F1 F2 = (-1, 2) :=
  sorry

end hyperbola_center_l30_30808


namespace ball_travel_distance_l30_30143

def ball_diameter : ℝ := 6 -- inches
def radius_A : ℝ := 80 -- inches
def radius_B : ℝ := 90 -- inches
def radius_C : ℝ := 70 -- inches
def radius_ball : ℝ := ball_diameter / 2

def adjusted_A := radius_A - radius_ball
def adjusted_B := radius_B + radius_ball
def adjusted_C := radius_C - radius_ball

def distance_A := 1 / 2 * 2 * Real.pi * adjusted_A
def distance_B := 2 * Real.pi * adjusted_B
def distance_C := 1 / 2 * 2 * Real.pi * adjusted_C

def total_distance : ℝ := distance_A + distance_B + distance_C

theorem ball_travel_distance :
  total_distance = 330 * Real.pi :=
sorry

end ball_travel_distance_l30_30143


namespace angle_between_c_and_b_l30_30930

variables (a b : ℝ^3) 

noncomputable def c : ℝ^3 := 
  1/2 • a + 1/4 • b

theorem angle_between_c_and_b 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (h_ab_angle : real.angle a b = 2 * real.pi / 3) :
  real.angle c b = real.arccos (1 / real.sqrt 6) :=
sorry

end angle_between_c_and_b_l30_30930


namespace rectangle_diagonal_l30_30666

theorem rectangle_diagonal (k : ℚ)
  (h1 : 2 * (5 * k + 2 * k) = 72)
  (h2 : k = 36 / 7) :
  let l := 5 * k,
      w := 2 * k,
      d := Real.sqrt ((l ^ 2) + (w ^ 2)) in
  d = 194 / 7 := by
  sorry

end rectangle_diagonal_l30_30666


namespace product_of_divisors_has_three_prime_factors_l30_30466

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30466


namespace product_of_divisors_has_three_prime_factors_l30_30458

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30458


namespace tangent_line_count_l30_30640

noncomputable def circle1 : set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 + 2 * p.2 - 4 = 0 }
noncomputable def circle2 : set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 - 4 * p.1 - 16 = 0 }

theorem tangent_line_count :
  let center1 := (0, -1) in
  let center2 := (2, 0) in
  let radius1 := real.sqrt 5 in
  let radius2 := 2 * (real.sqrt 5) in
  let dist_centers := real.sqrt 5 in
  dist_centers = radius2 - radius1 →
  ∃! (l : set (ℝ × ℝ)), (∀ x ∈ l, x ∈ circle1 ∨ x ∈ circle2) :=
begin
  sorry
end

end tangent_line_count_l30_30640


namespace find_other_number_l30_30004

theorem find_other_number (hcf lcm num1 num2 : ℕ) 
  (h_hcf : nat.gcd num1 num2 = hcf) 
  (h_lcm : nat.lcm num1 num2 = lcm) 
  (hcf_val : hcf = 16) 
  (lcm_val : lcm = 396) 
  (num1_val : num1 = 36) : 
  num2 = 176 :=
by
  -- Given conditions:
  -- hcf = 16, lcm = 396, num1 = 36
  -- num2 = (hcf * lcm) / num1
  sorry

end find_other_number_l30_30004


namespace perpendiculars_intersect_at_common_point_l30_30915

-- Definitions of the geometric constructs
variables {A B C D A1 B1 C1 : Type}
variables [InCircleCenter A B C D A1 B1 C1]

-- Assumptions of the problem
axiom equilateral_triangle_ABC : EquilateralTriangle A B C
axiom point_D_inside_ABC : PointInsideTriangle D A B C
axiom centers_of_incircles : Centers A1 B1 C1 A B C D

-- The main theorem to be proven
theorem perpendiculars_intersect_at_common_point :
  PerpendicularsIntersectAtCommonPoint A B C A1 B1 C1 :=
begin
  sorry
end

end perpendiculars_intersect_at_common_point_l30_30915


namespace least_pos_int_with_12_factors_l30_30104

theorem least_pos_int_with_12_factors : ∃ k : ℕ, (nat.num_factors k = 12) ∧ (∀ n : ℕ, nat.num_factors n = 12 → k ≤ n) ∧ k = 72 := 
sorry

end least_pos_int_with_12_factors_l30_30104


namespace number_of_distinct_prime_factors_of_B_l30_30437

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30437


namespace proof_supplies_proof_transportation_cost_proof_min_cost_condition_l30_30990

open Real

noncomputable def supplies_needed (a b : ℕ) := a = 200 ∧ b = 300

noncomputable def transportation_cost (x : ℝ) := 60 ≤ x ∧ x ≤ 260 ∧ ∀ w : ℝ, w = 10 * x + 10200

noncomputable def min_cost_condition (m x : ℝ) := 
  (0 < m ∧ m ≤ 8) ∧ (∀ w : ℝ, (10 - m) * x + 10200 ≥ 10320)

theorem proof_supplies : ∃ a b : ℕ, supplies_needed a b := 
by
  use 200, 300
  sorry

theorem proof_transportation_cost : ∃ x : ℝ, transportation_cost x := 
by
  use 60
  sorry

theorem proof_min_cost_condition : ∃ m x : ℝ, min_cost_condition m x := 
by
  use 8, 60
  sorry

end proof_supplies_proof_transportation_cost_proof_min_cost_condition_l30_30990


namespace Bela_wins_strategy_l30_30684

/-- 
There are 99 sticks with lengths 1, 2, 3, ..., 99 units.
Andrea and Béla take turns removing one stick, Andrea starts.
The game ends when exactly three sticks remain.
Andrea wins if the three remaining sticks can form a triangle.
Béla wins if the three remaining sticks cannot form a triangle.
Prove that Béla has a winning strategy.
-/

theorem Bela_wins_strategy : 
  (∃ (sticks : list ℕ), sticks = list.range 1 100) ∧ 
  (∀ (turn : ℕ), turn % 2 = 0 → turn = Andrea ∨ turn = Béla) ∧ 
  (take_turn_start: ℕ) ∧ 
  (game_ends : sticks.length = 3) ∧ 
  (Andrea_wins : ∀ a b c : ℕ, (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → "Andrea wins") ∧ 
  (Bela_wins : ∀ abc : list ℕ, sticks.length = 3 → (¬ (∃ (a b c : ℕ), a + b > c ∧ a + c > b ∧ b + c > a)) → "Béla wins")
  → "Béla has a winning strategy"
:=
sorry

end Bela_wins_strategy_l30_30684


namespace least_whole_number_greater_than_9_satisfying_condition_l30_30705

theorem least_whole_number_greater_than_9_satisfying_condition : 
  ∃ (n : ℕ), n > 9 ∧ (n^2 % 12 = n) ∧ ∀ (m : ℕ), m > 9 ∧ (m^2 % 12 = m) → n ≤ m :=
begin
  sorry
end

end least_whole_number_greater_than_9_satisfying_condition_l30_30705


namespace profit_percentage_not_sold_at_18_percent_l30_30156

theorem profit_percentage_not_sold_at_18_percent 
    (total_weight : ℝ)
    (weight_18_percent : ℝ)
    (overall_profit : ℝ)
    (profit_18_percent : ℝ)
    (weight_p_percent : ℝ)
    (initial_price : ℝ)
    (total_selling_price : ℝ)
    (selling_price_18_percent : ℝ) :
    total_weight = 1000 →
    weight_18_percent = 600 →
    weight_p_percent = total_weight - weight_18_percent →
    overall_profit = 0.14 →
    profit_18_percent = 0.18 →
    initial_price = 1 →
    total_selling_price = total_weight * initial_price * (1 + overall_profit) →
    selling_price_18_percent = weight_18_percent * initial_price * (1 + profit_18_percent) →
    (∃ P : ℝ, 400 * (1 + P / 100) = total_selling_price - selling_price_18_percent ∧ P = 8) :=
begin
  sorry
end

end profit_percentage_not_sold_at_18_percent_l30_30156


namespace prime_factors_of_B_l30_30522

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30522


namespace f_at_1_equals_2_l30_30943

noncomputable def f : ℝ → ℝ := λ x, x^2 + x

theorem f_at_1_equals_2 : f 1 = 2 := by
  sorry

end f_at_1_equals_2_l30_30943


namespace min_value_l30_30927

theorem min_value (a : ℝ) (h : a > 0) : a + 4 / a ≥ 4 :=
by sorry

end min_value_l30_30927


namespace square_side_length_l30_30709

theorem square_side_length (A : ℝ) (h : A = 100) : ∃ s : ℝ, s * s = A ∧ s = 10 := by
  sorry

end square_side_length_l30_30709


namespace sum_of_interiors_l30_30849

-- Let's define the conditions 
def is_regular_pentagon (polygon : Type) : Prop :=
  ∀ (n : ℕ), n = 5 → 180 * (n - 2) / n = 108

def is_rectangle (polygon : Type) : Prop :=
  ∀ (n : ℕ), n = 4 → 90 = 90

-- Angle calculation for pentagon and rectangle
def interior_angle_pentagon : ℝ := 108
def interior_angle_rectangle : ℝ := 90

def sum_of_angles_ABC_ABD (pentagon : Type) (rectangle : Type) [is_regular_pentagon pentagon] [is_rectangle rectangle] : ℝ :=
  interior_angle_pentagon + interior_angle_rectangle

-- The theorem to prove the sum of angles ABC and ABD is 198 degrees
theorem sum_of_interiors (pentagon : Type) (rectangle : Type) [is_regular_pentagon pentagon] [is_rectangle rectangle] :
  sum_of_angles_ABC_ABD pentagon rectangle = 198 :=
by
  unfold sum_of_angles_ABC_ABD
  unfold is_regular_pentagon is_rectangle
  simp
  sorry

end sum_of_interiors_l30_30849


namespace problem_min_value_l30_30311

noncomputable def min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : ℝ :=
  1 / x^2 + 1 / y^2 + 1 / (x * y)

theorem problem_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  min_value x y hx hy hxy = 3 := 
sorry

end problem_min_value_l30_30311


namespace distinct_prime_factors_of_B_l30_30449

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30449


namespace minimum_ratio_l30_30377

-- conditions
variables (a θ : ℝ) (h_a : a > 0)

-- definition of the areas of the shapes
noncomputable def area_triangle_ABC : ℝ := (1/2) * a^2 * tan θ
noncomputable def area_square_PQRS : ℝ := (a^2 * sin (2*θ)^2) / (sin (2*θ) + 2)^2

-- the ratio
noncomputable def ratio : ℝ := area_triangle_ABC a θ / area_square_PQRS a θ

-- proof problem statement
theorem minimum_ratio : (∃ θ : ℝ, 0 ≤ θ ∧ θ < π/2 ∧ ratio a θ = 9/4) :=
sorry

end minimum_ratio_l30_30377


namespace area_of_tangent_segments_region_l30_30798

noncomputable def radius : ℝ := 3
noncomputable def segment_length : ℝ := 4
noncomputable def area_of_region : ℝ := π * 4

theorem area_of_tangent_segments_region : area_of_region = π * 4 :=
by
  sorry

end area_of_tangent_segments_region_l30_30798


namespace count_irrational_numbers_l30_30184

theorem count_irrational_numbers :
  let numbers := [3.14159, real.cbrt 9 * -1, (7 + 1 / 7) / 1000 - 1 / 10, - real.pi, real.sqrt 25, real.cbrt 64, - (1 / 7)]
  let irrational_numbers := numbers.filter (λ x, ¬ (∃ (a b : ℚ), x = (a : ℝ) / b))
  irrational_numbers.length = 3 :=
by
  let numbers := 
  (3.14159 : ℝ) ::
  (- real.cbrt 9 : ℝ) ::
  (7 + 1 / 7) / 1000 - 1 / 10 ::
  (- real.pi : ℝ) ::
  (real.sqrt 25 : ℝ) ::
  (real.cbrt 64 : ℝ) ::
  (- (1 / 7) : ℝ) ::
  []
  let irrational_numbers := numbers.filter (λ x, ¬ ∃ a b : ℚ, x = (a : ℝ) / b)
  show irrational_numbers.length = 3, from sorry

end count_irrational_numbers_l30_30184


namespace general_terms_sum_first_n_terms_l30_30913

open Nat

-- Define the sequences and conditions
def a₁ : ℕ := 1
def a (n : ℕ) : ℕ := 3^(n-1)  -- General term of sequence {a_n}
def b (n : ℕ) : ℕ := 2*n - 1  -- General term of sequence {b_n}
def sum_seq (n : ℕ) : ℕ := (n-1) * 3^n + 1  -- Condition sequence sum

-- Assumptions
axiom a₃_condition : 2 * a 3 = a₁ + a 2 + 14 -- Condition involving a₃
axiom b_condition (n : ℕ) : ∑ k in range (n+1), a k * b k = sum_seq (n + 1)

-- Prove general terms of sequences {a_n} and {b_n}
theorem general_terms (n : ℕ) :
  (a n = 3^(n-1)) ∧ (b n = 2*n - 1) :=
sorry

-- Define the sequence {b_n / a_n}
def c (n : ℕ) : ℝ := (b n : ℝ) / (a n : ℝ)

-- Define the sum of the first n terms of the sequence {b_n / a_n}
def sum_terms (n : ℕ) : ℝ := ∑ k in range n, c k

-- Prove the sum of the first n terms of the sequence {b_n / a_n}
theorem sum_first_n_terms (n : ℕ) :
  sum_terms n = 3 - (n + 1) / 3^(n-1) :=
sorry

end general_terms_sum_first_n_terms_l30_30913


namespace find_lambda_l30_30322

open Set

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b : V}

theorem find_lambda (λ : ℝ) (h1 : λ • a + b = 1/2 • (a + 2 • b)) (h2 : ¬ Collinear ℝ {0, a, b}) : λ = 1/2 :=
sorry

end find_lambda_l30_30322


namespace modulus_product_l30_30247

theorem modulus_product (a b : ℂ) (ha : a = 3 - 5 * complex.i) (hb : b = 3 + 5 * complex.i) : 
  complex.abs a * complex.abs b = 34 :=
by
  rw [ha, hb]
  simp [complex.abs]
  sorry  -- proof is not required as per instructions

end modulus_product_l30_30247


namespace expected_value_of_ones_on_three_dice_l30_30051

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30051


namespace polynomial_coefficients_l30_30968

theorem polynomial_coefficients (a : ℕ → ℤ) :
  (∀ x, (x - 2) ^ 11 = ∑ k in Finset.range 12, a k * (x - 1) ^ k) →
  a 10 = -11 ∧ a 2 + a 4 + a 6 + a 8 + a 10 = -1023 :=
by
  intro h
  have h0 : a 10 = -11, from sorry
  have h1 : a 2 + a 4 + a 6 + a 8 + a 10 = -1023, from sorry
  exact ⟨h0, h1⟩

end polynomial_coefficients_l30_30968


namespace distinct_prime_factors_of_product_of_divisors_l30_30508

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30508


namespace distinct_prime_factors_of_B_l30_30475

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30475


namespace domain_f_l30_30632

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - 2 * real.log x / real.log 9)

theorem domain_f :
  {x : ℝ | 1 - 2 * real.log x / real.log 9 ≥ 0 ∧ x > 0} = {x : ℝ | 0 < x ∧ x ≤ 3} :=
by
  sorry

end domain_f_l30_30632


namespace solution_set_f_l30_30308

noncomputable def f (x : ℝ) : ℝ := if h : x > 0 then x^2 - x else -f (-x)

theorem solution_set_f (x : ℝ) : f x > 0 ↔ x ∈ (-1, 0) ∪ (1, +∞) :=
by
  sorry

end solution_set_f_l30_30308


namespace product_of_divisors_has_three_prime_factors_l30_30459

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30459


namespace expected_number_of_ones_on_three_dice_l30_30075

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30075


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30494

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30494


namespace price_reduction_l30_30727

theorem price_reduction (P : ℝ) : 
  let first_day_reduction := 0.91 * P
  let second_day_reduction := 0.90 * first_day_reduction
  second_day_reduction = 0.819 * P :=
by 
  sorry

end price_reduction_l30_30727


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30562

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30562


namespace power_addition_identity_l30_30191

theorem power_addition_identity : 
  (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end power_addition_identity_l30_30191


namespace part_a_part_b_l30_30630

theorem part_a (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * (Real.sqrt n + 1) / (n - 1))) :
  μ < (2 * (Real.sqrt n + 1) / (n - 1)) :=
by 
  exact h_μ.2

theorem part_b (n : ℕ) (h_n : 1 < n) (d : ℝ) (h_d : d = 1) (μ : ℝ) (h_μ : 0 < μ ∧ μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1)))) :
  μ < (2 * Real.sqrt 3 * (Real.sqrt n + 1) / (3 * (n - 1))) :=
by
  exact h_μ.2

end part_a_part_b_l30_30630


namespace rectangle_diagonal_l30_30663

theorem rectangle_diagonal (k : ℚ)
  (h1 : 2 * (5 * k + 2 * k) = 72)
  (h2 : k = 36 / 7) :
  let l := 5 * k,
      w := 2 * k,
      d := Real.sqrt ((l ^ 2) + (w ^ 2)) in
  d = 194 / 7 := by
  sorry

end rectangle_diagonal_l30_30663


namespace distinct_prime_factors_of_B_l30_30479

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30479


namespace distinct_prime_factors_of_B_l30_30471

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30471


namespace expected_number_of_ones_when_three_dice_rolled_l30_30036

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30036


namespace Justin_Tim_Emily_Play_Together_126_Times_l30_30855

theorem Justin_Tim_Emily_Play_Together_126_Times :
  ∀ (players : Finset ℕ)
    (Justin Tim Emily : ℕ),
    players.card = 12 ∧
    Justin ∈ players ∧
    Tim ∈ players ∧
    Emily ∈ players →
    (∃ play_count : ℕ, play_count = choose (players.erase Justin).erase Tim).erase Emily).card 7) ∧
    play_count = 126 :=
sorry

end Justin_Tim_Emily_Play_Together_126_Times_l30_30855


namespace distinct_prime_factors_B_l30_30555

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30555


namespace cube_in_tetrahedron_max_edge_length_l30_30105

theorem cube_in_tetrahedron_max_edge_length
  (a : ℝ) : (∀ (T : RegularTetrahedron), T.edge_length = 6 → 
              (a = (λ C : Cube, C.max_edge_length_that_can_freely_rotate_in_Tetrahedron T).a)) :=
by
  intros T hT
  sorry

end cube_in_tetrahedron_max_edge_length_l30_30105


namespace pascal_triangle_row_num_l30_30119

theorem pascal_triangle_row_num (n k : ℕ) (hn : n = 50) (hk : k = 2) : 
  nat.choose 50 2 = 1225 :=
by
  rw [nat.choose, hn, hk]
  sorry

end pascal_triangle_row_num_l30_30119


namespace distinct_prime_factors_of_B_l30_30477

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30477


namespace train_length_is_135_l30_30826

noncomputable def speed_km_per_hr : ℝ := 54
noncomputable def time_seconds : ℝ := 9
noncomputable def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
noncomputable def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem train_length_is_135 : length_of_train = 135 := by
  sorry

end train_length_is_135_l30_30826


namespace chess_tournament_games_l30_30370

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 
  2 * n * (n - 1) = 1200 :=
by
  rw h
  dsimp
  norm_num
  sorry

end chess_tournament_games_l30_30370


namespace expected_ones_three_standard_dice_l30_30058

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30058


namespace num_valid_squares_within_bounds_l30_30964

def is_valid_square(x1 y1 x2 y2: ℤ) : Prop :=
  let side_length := abs (x2 - x1)
  ∧ let side_length = abs (y2 - y1)
  in (side_length = 1 ∨ side_length = 2)
  ∧ x1 ≠ x2 ∧ y1 ≠ y2
  ∧ x1 % 1 = 0 ∧ x2 % 1 = 0 ∧ y1 % 1 = 0 ∧ y2 % 1 = 0
  ∧ x1 ≤ x2 ∧ y1 ≤ y2
  ∧ y2 ≤ (3/2) * x2
  ∧ y1 ≥ -1
  ∧ x2 ≤ 6

theorem num_valid_squares_within_bounds : 
  (finset.filter (λ (square : (ℤ × ℤ) × (ℤ × ℤ)),
      is_valid_square (square.1.1) (square.1.2) (square.2.1) (square.2.2))
          (finset.product 
             (finset.Icc (0, 0) (6, 9))
             (finset.Icc (0, 0) (6, 9)))).card = 94 :=
  sorry

end num_valid_squares_within_bounds_l30_30964


namespace time_equal_l30_30687

noncomputable def S : ℝ := sorry 
noncomputable def S_flat : ℝ := S
noncomputable def S_uphill : ℝ := (1 / 3) * S
noncomputable def S_downhill : ℝ := (2 / 3) * S
noncomputable def V_flat : ℝ := sorry 
noncomputable def V_uphill : ℝ := (1 / 2) * V_flat
noncomputable def V_downhill : ℝ := 2 * V_flat
noncomputable def t_flat: ℝ := S / V_flat
noncomputable def t_uphill: ℝ := S_uphill / V_uphill
noncomputable def t_downhill: ℝ := S_downhill / V_downhill
noncomputable def t_hill: ℝ := t_uphill + t_downhill

theorem time_equal: t_flat = t_hill := 
  by sorry

end time_equal_l30_30687


namespace TimeToPaintHouseTogether_l30_30390

/-- Define the work rates -/
def ShawnWorkRate := 1 / 18
def KarenWorkRate := 1 / 12

/-- Combined work rate when both work together -/
def CombinedWorkRate := ShawnWorkRate + KarenWorkRate

/-- Time taken by Shawn and Karen working together to paint one house -/
def TimeToPaintTogether := 1 / CombinedWorkRate

/-- Main theorem to prove the correct time -/
theorem TimeToPaintHouseTogether : TimeToPaintTogether = 7.2 :=
by
  sorry

end TimeToPaintHouseTogether_l30_30390


namespace course_selection_schemes_l30_30153

theorem course_selection_schemes (courses students : ℕ) (h_courses : courses = 3) (h_students : students = 3) (h_single_choice : ∀ s : ℕ, s ∈ (finset.range students) → ∃ c : ℕ, c ∈ (finset.range courses) ∧ ∀ t : ℕ, t ≠ s → c ≠ c) :
  (finset.card {(p : finset (finset ℕ)) | p.card = courses ∧ ∀ s ∈ (finset.range students), ∃ c ∈ p, s ∈ c ∧ (∀ t ∈ (finset.range students), t ≠ s → t ∉ c)} = 18)
:= sorry

end course_selection_schemes_l30_30153


namespace email_sending_ways_l30_30158

theorem email_sending_ways (emails addresses : ℕ) (h_emails : emails = 5) (h_addresses : addresses = 3) :
  (addresses ^ emails = 3 ^ 5) :=
by
  rw [h_emails, h_addresses]
  rfl

end email_sending_ways_l30_30158


namespace electromagnetic_storm_time_l30_30601

structure Time :=
(hh : ℕ)
(mm : ℕ)
(valid_hour : 0 ≤ hh ∧ hh < 24)
(valid_minute : 0 ≤ mm ∧ mm < 60)

def possible_digits (d : ℕ) : set ℕ :=
  {x | x = d + 1 ∨ x = d - 1}

theorem electromagnetic_storm_time :
  (∃ t : Time, t.hh = 18 ∧ t.mm = 18) →
  (∀ (orig : Time), 
    orig.hh ∈ possible_digits 0 ∧ 
    (orig.hh % 10) ∈ possible_digits 9 ∧
    orig.mm ∈ possible_digits 0 ∧ 
    (orig.mm % 10) ∈ possible_digits 9 →
      false) :=
by
  sorry

end electromagnetic_storm_time_l30_30601


namespace intersection_of_M_and_N_l30_30920

-- Definitions of the sets M and N
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Statement of the theorem proving the intersection of M and N
theorem intersection_of_M_and_N :
  M ∩ N = {2, 3} :=
by sorry

end intersection_of_M_and_N_l30_30920


namespace num_valid_x_l30_30892

theorem num_valid_x (x : ℕ) (hPR : ℕ) (hPV : ℕ) : 
  (∃ (a b c : ℕ),
    hPR = a^2 + b^2 ∧
    hPV = a^2 + c^2 ∧
    x^2 = b^2 + c^2 ∧ 
    x > 0 
  ) → 
  ∃ N : ℕ, N = 1981 :=
by {
  let h1 : ℕ := 1867,
  let h2 : ℕ := 2019,
  have h3 := (h1^2 + h2^2)^0.5,
  exact sorry,
}

end num_valid_x_l30_30892


namespace number_of_distinct_prime_factors_of_B_l30_30440

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30440


namespace hide_and_seek_l30_30768

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30768


namespace rectangle_diagonal_length_l30_30644

theorem rectangle_diagonal_length (P : ℝ) (r : ℝ) (perimeter_eq : P = 72) (ratio_eq : r = 5/2) :
  let k := P / 14
  let l := 5 * k
  let w := 2 * k
  let d := Real.sqrt (l^2 + w^2)
  d = 194 / 7 :=
sorry

end rectangle_diagonal_length_l30_30644


namespace log_positive_interval_l30_30328

noncomputable def f (a x : ℝ) : ℝ := Real.log (2 * x - a) / Real.log a

theorem log_positive_interval (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1 / 2) (2 / 3) → f a x > 0) ↔ (1 / 3 < a ∧ a < 1) := by
  sorry

end log_positive_interval_l30_30328


namespace part_I_part_II_part_III_l30_30329

noncomputable theory

-- Definitions and Conditions
def f (x : ℝ) (m : ℝ) : ℝ := (1/2) * x^2 + m * x + m * Real.log x

-- Part I: Discuss Monotonicity of f(x)
theorem part_I (m : ℝ) :
  (m ≥ 0 → ∀ x > 0, f x m > 0) ∧
  (m < 0 → ∀ x > 0, (f' x m > 0 ↔ x > (-m + Real.sqrt (m^2 - 4 * m))/2) ∧ (f' x m < 0 ↔ x < (-m + Real.sqrt (m^2 - 4 * m))/2)) := 
sorry

-- Part II: Range of a given unique solution
theorem part_II (a : ℝ) : 
  (∃! x ∈ Icc (1/Real.exp 1) ⊤, f x 1 = (1/2) * x^2 + a * x) ↔ (1 - Real.exp 1 ≤ a ∧ a ≤ 1) :=
sorry

-- Part III: Maximum value of m when m > 0
theorem part_III : 
  (∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → x2 ≤ 2 → |f x1 m - f x2 m| < x2^2 - x1^2) → m ≤ 1/2 :=
sorry

end part_I_part_II_part_III_l30_30329


namespace perp_FI_GH_l30_30585

variables {A B C D E F G H I : Type} [add_comm_group A] [module ℝ A]
variables (ABCD : parallelogram A B C D)
variables (E : A) (F : A)
variables (EG : line A) (BF : line A) (AF : line A) (BE : line A) (DH : line A) (BC : line A)

-- Definitions of the problem conditions
variables (hE_AD : point_on E (line_join A D))
variables (hF_CD : point_on F (line_join C D))
variables (hAEB : ∠ A E B = 90)
variables (hAFB : ∠ A F B = 90)
variables (hEG_AB : parallel EG (line_join A B))
variables (hEG_BF : intersect EG BF G)
variables (hAF_BE : intersect AF BE H)
variables (hDH_BC : intersect DH BC I)

-- Statement of the problem to prove
theorem perp_FI_GH : ⊥ (line_join F I) (line_join G H) :=
sorry

end perp_FI_GH_l30_30585


namespace count_irrational_numbers_l30_30183

theorem count_irrational_numbers :
  let numbers := [3.14159, real.cbrt 9 * -1, (7 + 1 / 7) / 1000 - 1 / 10, - real.pi, real.sqrt 25, real.cbrt 64, - (1 / 7)]
  let irrational_numbers := numbers.filter (λ x, ¬ (∃ (a b : ℚ), x = (a : ℝ) / b))
  irrational_numbers.length = 3 :=
by
  let numbers := 
  (3.14159 : ℝ) ::
  (- real.cbrt 9 : ℝ) ::
  (7 + 1 / 7) / 1000 - 1 / 10 ::
  (- real.pi : ℝ) ::
  (real.sqrt 25 : ℝ) ::
  (real.cbrt 64 : ℝ) ::
  (- (1 / 7) : ℝ) ::
  []
  let irrational_numbers := numbers.filter (λ x, ¬ ∃ a b : ℚ, x = (a : ℝ) / b)
  show irrational_numbers.length = 3, from sorry

end count_irrational_numbers_l30_30183


namespace minimum_value_of_expression_l30_30305

noncomputable def min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
(h3 : log 4 (2*a + b) = log 2 (sqrt (a*b))) : ℝ :=
if 2*a + b = 8 then 2*a + b else min_surrogate a b

theorem minimum_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
(h3 : log 4 (2*a + b) = log 2 (sqrt (a*b))) :
  8 ≤ 2*a + b :=
sorry

end minimum_value_of_expression_l30_30305


namespace force_water_on_dam_l30_30132

theorem force_water_on_dam (ρ g a b h : ℝ) (hρ : ρ = 1000) (hg : g = 10) (ha : a = 6.9) (hb : b = 11.4) (hh : h = 5.0) :
  let x := h in
  let P := ρ * g * x in
  let c := b - x * (b - a) / h in
  let F := ∫ x in 0..h, ρ * g * x * c in
  F = 1050000 := by
    sorry

end force_water_on_dam_l30_30132


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30496

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30496


namespace student_average_gt_true_average_l30_30167

variables {a b c : ℝ} (h : a < b) (h2 : b < c) (k : ℝ)
noncomputable def true_average : ℝ := (a + b + c) / 3
noncomputable def student_average (k : ℝ) : ℝ := ((b + c) * k + a) / 2

theorem student_average_gt_true_average (h : a < b) (h2 : b < c) (hk : k = 2) :
  student_average a b c k > true_average a b c :=
by
  sorry

end student_average_gt_true_average_l30_30167


namespace hide_and_seek_l30_30767

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30767


namespace distinct_prime_factors_of_product_of_divisors_l30_30511

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30511


namespace expected_ones_three_standard_dice_l30_30061

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30061


namespace train_length_is_135_l30_30825

noncomputable def speed_km_per_hr : ℝ := 54
noncomputable def time_seconds : ℝ := 9
noncomputable def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
noncomputable def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem train_length_is_135 : length_of_train = 135 := by
  sorry

end train_length_is_135_l30_30825


namespace hide_and_seek_friends_l30_30760

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30760


namespace vessel_p_alcohol_percentage_l30_30099

theorem vessel_p_alcohol_percentage (x : ℝ) :
  (87.5 / 100) * 4 + (x / 100) * 4 = 6 → x = 62.5 :=
by
  intro h
  have h' : (4 * x) / 100 + (87.5 * 4) / 100 = 6 := by
    rw [mul_div_assoc', mul_div_assoc'] 
    exact h
  sorry

end vessel_p_alcohol_percentage_l30_30099


namespace oranges_to_sell_l30_30144

theorem oranges_to_sell (cost_per_4_oranges : ℕ) (sell_per_6_oranges : ℕ) (desired_profit : ℕ) (cost_per_4_oranges = 14) (sell_per_6_oranges = 25) (desired_profit = 200) : 
  ∃ n, n = 300 :=
by 
  sorry

end oranges_to_sell_l30_30144


namespace intersection_of_sets_l30_30336

open Set

namespace Example

theorem intersection_of_sets :
  let A := ({-1, 1, 3, 5} : Set ℝ)
  let B := { x : ℝ | x^2 - 4 < 0 }
  A ∩ B = {-1, 1} :=
by
  let A := ({-1, 1, 3, 5} : Set ℝ)
  let B := { x : ℝ | x^2 - 4 < 0 }
  have h1 : A = {-1, 1, 3, 5} := rfl
  have h2 : B = { x : ℝ | -2 < x ∧ x < 2 } := by
    ext x
    simp
    split
    intros h
    linarith
    intros h
    nlinarith
  rw [h1, h2]
  ext x
  simp
  split
  intros h
  cases h with hA hB
  cases hA
  repeat {split; linarith}
  intros h
  split
  repeat {cases h, linarith}
      
end Example

end intersection_of_sets_l30_30336


namespace distinct_prime_factors_of_product_of_divisors_l30_30418

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30418


namespace emmy_rosa_ipods_total_l30_30240

theorem emmy_rosa_ipods_total :
  ∃ (emmy_initial rosa_current : ℕ), 
    emmy_initial = 14 ∧ 
    (emmy_initial - 6) / 2 = rosa_current ∧ 
    (emmy_initial - 6) + rosa_current = 12 :=
by
  sorry

end emmy_rosa_ipods_total_l30_30240


namespace distinct_prime_factors_of_product_of_divisors_l30_30424

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30424


namespace trigonometric_expression_identity_l30_30199

theorem trigonometric_expression_identity :
  (1 - 1 / Real.cos (37 * Real.pi / 180)) * (1 + 1 / Real.sin (53 * Real.pi / 180)) *
  (1 - 1 / Real.sin (37 * Real.pi / 180)) * (1 + 1 / Real.cos (53 * Real.pi / 180)) = 1 :=
by
  have h1 : Real.cos (37 * Real.pi / 180) = Real.sin (53 * Real.pi / 180), from sorry,
  have h2 : Real.sin (37 * Real.pi / 180) = Real.cos (53 * Real.pi / 180), from sorry,
  sorry

end trigonometric_expression_identity_l30_30199


namespace distinct_prime_factors_product_divisors_60_l30_30489

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30489


namespace bisection_approx_solution_l30_30695

-- Define the function f(x) = x^3 + 2x - 9
def f (x : ℝ) : ℝ := x^3 + 2 * x - 9

-- Define the partial function values as given in the table
def partial_values : list (ℝ × ℝ) := [
  (1,     -6),
  (2,     3),
  (1.5,   -2.625),
  (1.625, -1.459),
  (1.75,  -0.14),
  (1.875,  1.3418),
  (1.8125, 0.5793)
]

-- Define the desired accuracy
def accuracy : ℝ := 0.1

-- The statement to prove the approximate solution using bisection method
theorem bisection_approx_solution :
  ∃ (approx : ℝ), abs (approx - 1.8) < accuracy ∧ 
  (1.75 ≤ approx ∧ approx ≤ 1.8125) ∧
  (∃ a b, a < b ∧ f a * f b < 0 ∧ f approx = f a + (f b - f a) * (approx - a) / (b - a)) :=
by sorry

end bisection_approx_solution_l30_30695


namespace common_solutions_l30_30258

theorem common_solutions (x y : ℂ) : (y = (x - 1)^2 ∧ x * y + y = 2) ↔ 
  ((∃ t : ℝ, x = complex.exp (complex.I * (2 * t * real.pi / 3)) * 2^(1/3 : ℝ) + 1 ∧ y = (complex.exp (complex.I * (2 * t * real.pi / 3)) * 2^(1/3 : ℝ))^2) ∨ 
   (∃ t : ℝ, t = 0 ∧ x = 2^(1/3 : ℝ) + 1 ∧ y = 2^(2/3 : ℝ))) := 
by
  sorry

end common_solutions_l30_30258


namespace distinct_prime_factors_of_product_of_divisors_l30_30514

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30514


namespace pascal_third_number_in_51_row_l30_30116

-- Definition and conditions
def pascal_row_num := 50
def third_number_index := 2

-- Statement of the problem
theorem pascal_third_number_in_51_row : 
  (nat.choose pascal_row_num third_number_index) = 1225 :=
by {
  -- The proof step will be skipped using sorry
  sorry
}

end pascal_third_number_in_51_row_l30_30116


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30497

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30497


namespace sequence_next_term_l30_30141

theorem sequence_next_term (a b c d e : ℕ) (h1 : a = 34) (h2 : b = 45) (h3 : c = 56) (h4 : d = 67) (h5 : e = 78) (h6 : b = a + 11) (h7 : c = b + 11) (h8 : d = c + 11) (h9 : e = d + 11) : e + 11 = 89 :=
by
  sorry

end sequence_next_term_l30_30141


namespace product_fraction_eq_l30_30190

theorem product_fraction_eq :
  (∏ n in Finset.range 13, (n + 1) * (n + 3) / (n + 5) ^ 2) = (3 / 161840) :=
by
  sorry

end product_fraction_eq_l30_30190


namespace range_of_a_l30_30271

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ∈ [-2, +∞) :=
sorry

end range_of_a_l30_30271


namespace carla_sunflowers_l30_30842

theorem carla_sunflowers :
  ∀ (S : ℕ),
  let seeds_per_sunflower := 9 in
  let dandelions := 8 in
  let seeds_per_dandelion := 12 in
  let total_dandelion_seeds := dandelions * seeds_per_dandelion in
  let total_seeds := 96 / 0.64 in
  let total_sunflower_seeds := total_seeds - total_dandelion_seeds in
  S = total_sunflower_seeds / seeds_per_sunflower → S = 6 :=
by 
  intros S seeds_per_sunflower dandelions seeds_per_dandelion total_dandelion_seeds total_seeds total_sunflower_seeds hS
  exact hS

end carla_sunflowers_l30_30842


namespace planes_parallel_l30_30310

variables {Plane : Type} [NonEmpty Plane]
variables {Line : Type} [NonEmpty Line]
variables (α β γ : Plane) (l m : Line)
variables (plane_parallel : Plane → Plane → Prop)
variables (line_parallel : Line → Line → Prop)
variables (line_on_plane : Line → Plane → Prop)
variables (line_perpendicular : Line → Plane → Prop)

-- Conditions in problem D
axiom l_perp_α : line_perpendicular l α
axiom m_perp_β : line_perpendicular m β
axiom l_par_m : line_parallel l m

-- The goal to prove: α is parallel to β
theorem planes_parallel : plane_parallel α β :=
sorry

end planes_parallel_l30_30310


namespace expected_number_of_ones_when_three_dice_rolled_l30_30037

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30037


namespace trajectory_is_line_segment_l30_30280

theorem trajectory_is_line_segment : 
  ∃ (P : ℝ × ℝ) (F1 F2: ℝ × ℝ), 
    F1 = (-3, 0) ∧ F2 = (3, 0) ∧ (|F1.1 - P.1|^2 + |F1.2 - P.2|^2).sqrt + (|F2.1 - P.1|^2 + |F2.2 - P.2|^2).sqrt = 6
  → (P.1 = F1.1 ∨ P.1 = F2.1) ∧ (P.2 = F1.2 ∨ P.2 = F2.2) :=
by sorry

end trajectory_is_line_segment_l30_30280


namespace smallest_palindrome_base2_base4_gt10_l30_30200

/--
  Compute the smallest base-10 positive integer greater than 10 that is a palindrome 
  when written in both base 2 and base 4.
-/
theorem smallest_palindrome_base2_base4_gt10 : 
  ∃ n : ℕ, 10 < n ∧ is_palindrome (nat_to_base 2 n) ∧ is_palindrome (nat_to_base 4 n) ∧ n = 15 :=
sorry

def nat_to_base (b n : ℕ) : list ℕ :=
sorry

def is_palindrome {α : Type} [DecidableEq α] (l : list α) : bool :=
sorry

end smallest_palindrome_base2_base4_gt10_l30_30200


namespace student_correct_answers_l30_30993

theorem student_correct_answers 
(C W : ℕ) 
(h1 : C + W = 80) 
(h2 : 4 * C - W = 120) : 
C = 40 :=
by
  sorry 

end student_correct_answers_l30_30993


namespace problem_l30_30410

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30410


namespace meeting_time_l30_30142

noncomputable def combined_speed : ℕ := 10 -- km/h
noncomputable def distance_to_cover : ℕ := 50 -- km
noncomputable def start_time : ℕ := 6 -- pm (in hours)
noncomputable def speed_a : ℕ := 6 -- km/h
noncomputable def speed_b : ℕ := 4 -- km/h

theorem meeting_time : start_time + (distance_to_cover / combined_speed) = 11 :=
by
  sorry

end meeting_time_l30_30142


namespace prime_mod_congruence_l30_30134

theorem prime_mod_congruence (p : ℕ) (hp : Nat.Prime p) (x : Fin p → ℤ) :
  (∀ n : ℕ, (∑ i : Fin p, x i ^ n) % p = 0) →
  (∀ i j : Fin p, x i % p = x j % p) :=
by
  sorry

end prime_mod_congruence_l30_30134


namespace minimum_n_integers_l30_30299

theorem minimum_n_integers (n : ℕ) (x : ℕ → ℤ) 
  (h : ∑ i in finset.range n, (-1) ^ i * (x i)^4 = 1599) : 
  n ≥ 15 :=
sorry

end minimum_n_integers_l30_30299


namespace sum_of_factors_24_l30_30712

theorem sum_of_factors_24 : (∑ n in Finset.filter (λ d, 24 % d = 0) (Finset.range 25), n) = 60 := 
by 
  sorry

end sum_of_factors_24_l30_30712


namespace problem_l30_30593

theorem problem (a b c d : ℝ) (h1 : 2 + real.sqrt 2 = a + b) (h2 : 4 - real.sqrt 2 = c + d) 
  (ha : a = 3) (hb : b = real.sqrt 2 - 1) (hc : c = 2) (hd : d = 2 - real.sqrt 2) : 
  (b + d) / (a * c) = 1 / 6 :=
by
  rw [hb, hd, ha, hc]
  sorry

end problem_l30_30593


namespace product_of_coordinates_of_D_l30_30302

theorem product_of_coordinates_of_D 
  (x y : ℝ)
  (midpoint_x : (5 + x) / 2 = 4)
  (midpoint_y : (3 + y) / 2 = 7) : 
  x * y = 33 := 
by 
  sorry

end product_of_coordinates_of_D_l30_30302


namespace emmy_journey_total_length_l30_30236

/-- 
Emmy's journey conditions:
- The first 1/4 of the journey is on a gravel road.
- 30 miles are on pavement.
- The remaining 1/6 is on a sandy track.
-/

theorem emmy_journey_total_length :
  let x := 30 * 12 / 7 in
  1 / 4 * x + 30 + 1 / 6 * x = x :=
by
  sorry

end emmy_journey_total_length_l30_30236


namespace prime_factors_of_B_l30_30524

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30524


namespace distinct_prime_factors_of_B_l30_30448

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30448


namespace hide_and_seek_problem_l30_30745

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30745


namespace no_natural_numbers_satisfying_conditions_l30_30866

theorem no_natural_numbers_satisfying_conditions :
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ k : ℕ, b^2 + 4*a = k^2 := by
  sorry

end no_natural_numbers_satisfying_conditions_l30_30866


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30571

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30571


namespace ellipse_equation_unique_y0_l30_30937

theorem ellipse_equation_unique_y0
  (a b : ℝ) (e : ℝ) (rhombus_area : ℝ) (A Q : ℝ × ℝ) (dot_product : ℝ) :
  e = sqrt 3 / 2 → 
  rhombus_area = 4 → 
  A = (-a, 0) → 
  ∃ y₀ : ℝ, Q = (0, y₀) ∧ 
    (y₀ = 2 * sqrt 2 ∨ y₀ = -2 * sqrt 2 ∨ y₀ = 2 * sqrt 14 / 5 ∨ y₀ = -2 * sqrt 14 / 5) ∧ 
    (dot_product = 4) → 
  (a = 2 ∧ b = 1 ∧ ellipse_equation = (λ x y : ℝ, (x^2) / 4 + y^2 = 1)) :=
by
  sorry

end ellipse_equation_unique_y0_l30_30937


namespace diagonal_length_l30_30653

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l30_30653


namespace number_of_distinct_prime_factors_of_B_l30_30535

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30535


namespace distinct_prime_factors_of_product_of_divisors_l30_30423

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30423


namespace distinct_prime_factors_of_B_l30_30445

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30445


namespace initial_deck_card_count_l30_30371

theorem initial_deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 := by
  sorry

end initial_deck_card_count_l30_30371


namespace minimum_value_a_squared_l30_30967

open Classical
noncomputable theory

variables {X : Type} [Probability X]
variables {p : ℝ} (h : 0 < p ∧ p < 1)
variables (a : ℝ) (hx : Binomial X 16 p)

theorem minimum_value_a_squared (h_variance : D(a * X) = 16) : a^2 = 4 :=
sorry

end minimum_value_a_squared_l30_30967


namespace second_rectangle_area_l30_30006

theorem second_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hbx : x < h):
  2 * b * x * (h - 3 * x) / h = (2 * b * x * (h - 3 * x))/h := 
sorry

end second_rectangle_area_l30_30006


namespace find_three_digit_number_l30_30022

theorem find_three_digit_number (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : 0 ≤ c ∧ c ≤ 9)
    (h₄ : (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 33 / 37) :
    100 * a + 10 * b + c = 447 :=
sorry

end find_three_digit_number_l30_30022


namespace sum_of_factors_24_l30_30713

theorem sum_of_factors_24 : (∑ n in Finset.filter (λ d, 24 % d = 0) (Finset.range 25), n) = 60 := 
by 
  sorry

end sum_of_factors_24_l30_30713


namespace sum_binomial_series_eq_l30_30198

-- Define a helper function for binomial coefficients
def binomial_coeff (n k : ℕ) : ℚ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Define the sum from 2009 to infinity of the reciprocal of the binomial coefficient
def sum_binomial_reciprocal (k : ℕ) : ℚ :=
  ∑' n in finset.range (2009 + k), (1 / binomial_coeff n 2009) 

theorem sum_binomial_series_eq (k : ℕ) (h : k = 0) : 
  sum_binomial_reciprocal k = (2009 / 2008) := 
by sorry

end sum_binomial_series_eq_l30_30198


namespace probability_divisible_by_7_l30_30360

def digit_sum_42 (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ (n.digits.sum = 42)

def divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

theorem probability_divisible_by_7 :
  let S := {n : ℕ | digit_sum_42 n} in
  ∃! (p : ℚ), p = 2 / 15 ∧ p = (S.filter divisible_by_7).card / S.card :=
sorry

end probability_divisible_by_7_l30_30360


namespace perfect_square_trinomial_k_l30_30975

theorem perfect_square_trinomial_k (k : ℤ) : (∃ a b : ℤ, (a*x + b)^2 = x^2 + k*x + 9) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_k_l30_30975


namespace solution_exists_l30_30017

noncomputable def proof_problem (m p : Real) : Prop :=
  let B : Matrix (Fin 2) (Fin 2) Real := ![![1, 4], ![6, m]]
  let B_inv := B⁻¹
  let pB := p • B
  B_inv = pB ∧ p * 1 = 3 * m ∧
  ( (m ≈ 24.02 ∧ p ≈ -83.33) ∨ (m ≈ -0.017 ∧ p ≈ 0.0416) )

theorem solution_exists : ∃ m p : Real, proof_problem m p := by
  sorry

end solution_exists_l30_30017


namespace units_digit_of_expression_l30_30195

theorem units_digit_of_expression :
  (9 * 19 * 1989 - 9 ^ 3) % 10 = 0 :=
by
  sorry

end units_digit_of_expression_l30_30195


namespace inequality_holds_l30_30899

variable {R : Type} [linear_ordered_field R] (a b : R)

noncomputable def condition1 (a b : R) : Prop := (0 < a) ∧ (0 < b)
noncomputable def condition2 (a b : R) : Prop := (1 / a + 1 / b = 1)

theorem inequality_holds (n : ℕ) (hn : n > 0)
  (h1 : condition1 a b)
  (h2 : condition2 a b)
  : (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry

end inequality_holds_l30_30899


namespace coefficient_x2_in_expansion_l30_30997

theorem coefficient_x2_in_expansion :
  let p1 := (1 - 2 * x) ^ 5
  let p2 := (1 + 3 * x) ^ 4
  let expansion := p1 * p2
  coeff expansion 2 = -26 := by sorry

end coefficient_x2_in_expansion_l30_30997


namespace paula_remaining_money_l30_30607

theorem paula_remaining_money (initial_amount cost_per_shirt cost_of_pants : ℕ) 
                             (num_shirts : ℕ) (H1 : initial_amount = 109)
                             (H2 : cost_per_shirt = 11) (H3 : num_shirts = 2)
                             (H4 : cost_of_pants = 13) :
  initial_amount - (num_shirts * cost_per_shirt + cost_of_pants) = 74 := 
by
  -- Calculation of total spent and remaining would go here.
  sorry

end paula_remaining_money_l30_30607


namespace sequence_sum_l30_30908

theorem sequence_sum :
  let a : ℕ → ℕ := λ n => if n = 0 then 0 else ∑ i in finset.range n, (i + 1),
      s : ℕ → ℚ := λ n => ∑ i in finset.range (n+1), (1 : ℚ) / a (i + 1)
  in
  (∀ m n : ℕ, a (m + n) = a m + a n + m * n) →
  a 1 = 1 →
  s 2016 = 2017 / 1009 :=
by
  sorry

end sequence_sum_l30_30908


namespace distinct_prime_factors_of_B_l30_30443

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30443


namespace expectation_decreases_variance_increases_l30_30686

noncomputable def boxA_initial_red_balls := 1
noncomputable def boxB_total_red_balls := 3
noncomputable def boxB_total_balls := 6

def E_Y (n : ℕ) (h : 1 ≤ n ∧ n ≤ 6) : ℚ :=
  (1 : ℚ) / 2 + 1 / (2 * n + 2)

def D_Y (n : ℕ) (h : 1 ≤ n ∧ n ≤ 6) : ℚ :=
  let p := E_Y n h in
  p * (1 - p)

theorem expectation_decreases (n m : ℕ) (hn : 1 ≤ n ∧ n ≤ 6) (hm : 1 ≤ m ∧ m ≤ 6) (h : n < m) :
  E_Y m hm < E_Y n hn :=
  sorry

theorem variance_increases (n m : ℕ) (hn : 1 ≤ n ∧ n ≤ 6) (hm : 1 ≤ m ∧ m ≤ 6) (h : n < m) :
  D_Y n hn < D_Y m hm :=
  sorry

end expectation_decreases_variance_increases_l30_30686


namespace smallest_palindrome_base2_base4_gt10_l30_30204

/--
  Compute the smallest base-10 positive integer greater than 10 that is a palindrome 
  when written in both base 2 and base 4.
-/
theorem smallest_palindrome_base2_base4_gt10 : 
  ∃ n : ℕ, 10 < n ∧ is_palindrome (nat_to_base 2 n) ∧ is_palindrome (nat_to_base 4 n) ∧ n = 15 :=
sorry

def nat_to_base (b n : ℕ) : list ℕ :=
sorry

def is_palindrome {α : Type} [DecidableEq α] (l : list α) : bool :=
sorry

end smallest_palindrome_base2_base4_gt10_l30_30204


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30495

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30495


namespace smallest_six_digit_odd_div_by_125_l30_30676

theorem smallest_six_digit_odd_div_by_125 : 
  ∃ n : ℕ, n = 111375 ∧ 
           100000 ≤ n ∧ n < 1000000 ∧ 
           (∀ d : ℕ, d ∈ (n.digits 10) → d % 2 = 1) ∧ 
           n % 125 = 0 :=
by
  sorry

end smallest_six_digit_odd_div_by_125_l30_30676


namespace friends_who_participate_l30_30778

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30778


namespace base_10_to_base_7_l30_30702

-- We state the problem as a theorem in Lean.
theorem base_10_to_base_7 (n : ℕ) (h : n = 1234) : 
  ∃ a b c d : ℕ, a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 = n ∧ a = 3 ∧ b = 4 ∧ c = 1 ∧ d = 2 :=
by
  use 3, 4, 1, 2
  split
  . calc 3 * 7^3 + 4 * 7^2 + 1 * 7^1 + 2 * 7^0
       = 3 * 343 + 4 * 49 + 1 * 7 + 2 * 1 : by norm_num
    ... = 1029 + 196 + 7 + 2 : by norm_num
    ... = 1234 : by norm_num
  . split
    . rfl
    . split
      . rfl
      . split
        . rfl
        . rfl
  sorry

end base_10_to_base_7_l30_30702


namespace expected_ones_three_dice_l30_30086

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30086


namespace monotonically_increasing_on_interval_range_of_values_for_a_approximate_bisection_solution_l30_30940

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x

-- 1. Prove that f(x) is monotonically increasing on the interval (-1,1]
theorem monotonically_increasing_on_interval :
  ∀ x : ℝ, x ∈ Set.Ioo (-1:ℝ) (1:ℝ) → 0 ≤ 3 * (1 - x^2) :=
sorry

-- 2. Determine the range of values for a such that f(x) = a has a solution for x ∈ (-1,1]
theorem range_of_values_for_a :
  ∀ a : ℝ, a > -2 ∧ a ≤ 2 →
    ∃ x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ f x = a :=
sorry

-- 3. Use the bisection method to find an approximate solution to the equation f(x)=1 within (-1,1) with accuracy up to 0.1
theorem approximate_bisection_solution :
  ∃ x_0 : ℝ, x_0 ∈ Set.Ioo (-1) 1 ∧ |x_0 - 0.3| < 0.1 ∧ f(x_0) = 1 :=
sorry

end monotonically_increasing_on_interval_range_of_values_for_a_approximate_bisection_solution_l30_30940


namespace chord_length_l30_30797

theorem chord_length (r : ℝ) (h : r = 12) : 
  ∃ (l : ℝ), l = 12 * real.sqrt 3 :=
by 
  use 12 * real.sqrt 3
  have h_r := h.symm
  rw h_r
  sorry

end chord_length_l30_30797


namespace colored_midpoints_at_least_2016_l30_30292

theorem colored_midpoints_at_least_2016 (n : ℕ) (h : n = 3025) 
(color_midpoint : (ℕ × ℕ → Prop) → ℕ → ℕ → Prop) 
(Green Blue Red : ℕ → ℕ → Prop) :
  ∃ c : ℕ → ℕ → Prop, (c = Green ∨ c = Blue ∨ c = Red) ∧ 
  (∃ k ≥ 2016, ∀ i j, color_midpoint c i j → i ≠ j → i < n → j < n → k ≤ folded Points) := 
sorry

end colored_midpoints_at_least_2016_l30_30292


namespace total_cats_and_kittens_l30_30688

theorem total_cats_and_kittens (cats_total : ℕ) (perc_females : ℝ) (half_litter : ℝ) (avg_kittens : ℕ) 
(h_cat : cats_total = 150)
(h_female : perc_females = 0.6)
(h_half_litter : half_litter = 0.5)
(h_avg_kittens : avg_kittens = 5) : 
  let females := perc_females * cats_total in 
  let litters := females * half_litter in
  let kittens := litters * avg_kittens in
  let total := cats_total + kittens in
  total = 375 :=
by
  sorry

end total_cats_and_kittens_l30_30688


namespace eval_f_f3_l30_30894

def f (x : ℝ) : ℝ :=
if x < 3 then 3 * Real.exp (x - 1) else Real.log (x ^ 2 - 6) / Real.log 3

theorem eval_f_f3 : f (f 3) = 3 :=
by
  sorry

end eval_f_f3_l30_30894


namespace rectangle_diagonal_l30_30664

theorem rectangle_diagonal (k : ℚ)
  (h1 : 2 * (5 * k + 2 * k) = 72)
  (h2 : k = 36 / 7) :
  let l := 5 * k,
      w := 2 * k,
      d := Real.sqrt ((l ^ 2) + (w ^ 2)) in
  d = 194 / 7 := by
  sorry

end rectangle_diagonal_l30_30664


namespace S_2016_eq_l30_30334

-- Definitions
def a (n : ℕ) : ℝ :=
  1 / ((Real.sqrt (n-1) + Real.sqrt n) * (Real.sqrt (n-1) + Real.sqrt (n+1)) * (Real.sqrt n + Real.sqrt (n+1)))

def S (n : ℕ) : ℝ := ∑ k in Finset.range n, a (k + 1)

-- Theorem to be proved
theorem S_2016_eq : S 2016 = (1 + Real.sqrt 2016 - Real.sqrt 2017) / 2 :=
  by
    sorry

end S_2016_eq_l30_30334


namespace distinct_prime_factors_of_product_of_divisors_l30_30417

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30417


namespace product_of_divisors_has_three_prime_factors_l30_30467

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30467


namespace distinct_prime_factors_product_divisors_60_l30_30491

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30491


namespace HCF_48_99_l30_30621

-- definitions and theorem stating the problem
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_48_99 : HCF 48 99 = 3 :=
by
  sorry

end HCF_48_99_l30_30621


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30500

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30500


namespace max_sum_of_differences_on_rhombus_segments_l30_30100

theorem max_sum_of_differences_on_rhombus_segments (n : ℕ) :
  let k := 2*n^2 in 
  let segment_diff_sum := ∑ i in range k.pred, (|i - (i + 1)|) in
  -- Assuming this representation captures the setup of the problem
  segment_diff_sum.max = 3*n^4 - 4*n^2 + 4*n - 2 :=
sorry

end max_sum_of_differences_on_rhombus_segments_l30_30100


namespace distinct_nonzero_real_product_l30_30579

noncomputable section
open Real

theorem distinct_nonzero_real_product
  (a b c d : ℝ)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hcd : c ≠ d)
  (hda : d ≠ a)
  (ha_ne_0 : a ≠ 0)
  (hb_ne_0 : b ≠ 0)
  (hc_ne_0 : c ≠ 0)
  (hd_ne_0 : d ≠ 0)
  (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/d ∧ c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 :=
sorry

end distinct_nonzero_real_product_l30_30579


namespace perfect_square_trinomial_k_l30_30976

theorem perfect_square_trinomial_k (k : ℤ) : (∃ a b : ℤ, (a*x + b)^2 = x^2 + k*x + 9) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_k_l30_30976


namespace range_of_a_l30_30576

def f (x a : ℝ) : ℝ := x^2 - 2*x + a^2 + 3*a - 3

def min_f (a : ℝ) : ℝ := (a^2 + 3*a - 4)

def distance_MF (a : ℝ) : ℝ := (a^2 / 4) + 1

theorem range_of_a (a : ℝ) (p q : Prop) (h₁: ¬ p = false) (h₂: (p ∧ q) = false)
  : p = (∃ a : ℝ, ¬(a^2 + 3*a - 4 < 0)) → 
    q = ¬(distance_MF a > 2) →
    -2 <= a ∧ a < 1 := 
sorry

end range_of_a_l30_30576


namespace fruit_basket_count_l30_30354

theorem fruit_basket_count :
  let apples := 7,
  let oranges := 12,
  ∃ n : ℕ, n = apples * oranges ∧ n = 84 := 
by
  sorry

end fruit_basket_count_l30_30354


namespace prime_factors_of_B_l30_30520

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30520


namespace diagonal_length_l30_30652

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l30_30652


namespace workshop_employees_l30_30150

theorem workshop_employees (x y : ℕ) 
  (H1 : (x + y) - ((1 / 2) * x + (1 / 3) * y + (1 / 3) * x + (1 / 2) * y) = 120)
  (H2 : (1 / 2) * x + (1 / 3) * y = (1 / 7) * ((1 / 3) * x + (1 / 2) * y) + (1 / 3) * x + (1 / 2) * y) : 
  x = 480 ∧ y = 240 := 
by
  sorry

end workshop_employees_l30_30150


namespace circle_area_l30_30875

theorem circle_area : 
    (∃ x y : ℝ, 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
    (∃ A : ℝ, A = (7 / 4) * Real.pi) :=
by
  sorry

end circle_area_l30_30875


namespace axis_angle_set_l30_30672

def is_x_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def is_y_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

def is_axis_angle (α : ℝ) : Prop := ∃ n : ℤ, α = (n * Real.pi) / 2

theorem axis_angle_set : 
  (∀ α : ℝ, is_x_axis_angle α ∨ is_y_axis_angle α ↔ is_axis_angle α) :=
by 
  sorry

end axis_angle_set_l30_30672


namespace shale_mix_per_pound_is_5_l30_30795

noncomputable def cost_of_shale_mix_per_pound 
  (cost_limestone : ℝ) (cost_compound : ℝ) (weight_limestone : ℝ) (total_weight : ℝ) : ℝ :=
  let total_cost_limestone := weight_limestone * cost_limestone 
  let weight_shale := total_weight - weight_limestone
  let total_cost := total_weight * cost_compound
  let total_cost_shale := total_cost - total_cost_limestone
  total_cost_shale / weight_shale

theorem shale_mix_per_pound_is_5 :
  cost_of_shale_mix_per_pound 3 4.25 37.5 100 = 5 := 
by 
  sorry

end shale_mix_per_pound_is_5_l30_30795


namespace total_number_of_handshakes_l30_30188

theorem total_number_of_handshakes 
    (players_per_team : ℕ)
    (teams : ℕ)
    (referees : ℕ)
    (coaches : ℕ)
    (handshakes_between_teams : ℕ)
    (handshakes_between_players_and_referees : ℕ)
    (handshakes_involving_coaches : ℕ) :
    players_per_team = 6 →
    teams = 2 →
    referees = 3 →
    coaches = 2 →
    handshakes_between_teams = (players_per_team * players_per_team) →
    handshakes_between_players_and_referees = ((players_per_team * teams) * referees) →
    handshakes_involving_coaches = (coaches * (players_per_team * teams + referees)) →
    handshakes_between_teams + handshakes_between_players_and_referees + handshakes_involving_coaches = 102 :=
by
    intros h_players_per_team h_teams h_referees h_coaches h_handshakes_between_teams h_handshakes_between_players_and_referees h_handshakes_involving_coaches
    rw [h_players_per_team, h_teams, h_referees, h_coaches, h_handshakes_between_teams, h_handshakes_between_players_and_referees, h_handshakes_involving_coaches]
    sorry

end total_number_of_handshakes_l30_30188


namespace intersection_length_at_least_l30_30730

variables (m n : ℕ) (h : ℝ)
variables (Hmn : 0 < m ∧ 0 < n)
variables (X : set ℝ) (Y : set ℝ)
variables (X_properties : set.subset X (set.Icc 0 (2*π)) ∧ set.measure X = 2 * π * h)
variables (Y_properties : set.subset Y (set.Icc 0 (2*π)) ∧ set.measure Y = m * (π / n))

noncomputable def rotation (s : set ℝ) (i : ℕ) : set ℝ :=
  {x | ∃ y ∈ s, x = (y + i * (π / n)) % (2 * π)}

theorem intersection_length_at_least (m n : ℕ) (h : ℝ)
  (Hmn : 0 < m ∧ 0 < n) (X : set ℝ) 
  (Y : set ℝ) (X_properties : set.subset X (set.Icc 0 (2 * π)) ∧ set.measure X = 2 * π * h)
  (Y_properties : set.subset Y (set.Icc 0 (2 * π)) ∧ set.measure Y = m * (π / n)) :
  ∃ i : ℕ, set.measure (rotation X i ∩ Y) ≥ h * m * (π / n) := 
sorry

end intersection_length_at_least_l30_30730


namespace total_ticket_sales_cost_l30_30819

theorem total_ticket_sales_cost
  (num_orchestra num_balcony : ℕ)
  (price_orchestra price_balcony : ℕ)
  (total_tickets total_revenue : ℕ)
  (h1 : num_orchestra + num_balcony = 370)
  (h2 : num_balcony = num_orchestra + 190)
  (h3 : price_orchestra = 12)
  (h4 : price_balcony = 8)
  (h5 : total_tickets = 370)
  : total_revenue = 3320 := by
  sorry

end total_ticket_sales_cost_l30_30819


namespace max_PF1_PF2_l30_30015

noncomputable def a := 2 * Real.sqrt 2
noncomputable def ellipse_eq (x y : ℝ) : Prop := x^2 / 8 + y^2 = 1
noncomputable def PF1_plus_PF2 (P F1 F2 : ℝ × ℝ) : ℝ :=
  Real.dist P F1 + Real.dist P F2

theorem max_PF1_PF2
  (F1 F2 P : ℝ × ℝ)
  (hP : ellipse_eq P.1 P.2)
  (h_foci : F1 = (-2, 0) ∧ F2 = (2, 0)) :
  ∃ P, PF1_plus_PF2 P F1 F2 = 4 * Real.sqrt 2 ∧ Real.dist P F1 * Real.dist P F2 = 8 := 
sorry

end max_PF1_PF2_l30_30015


namespace subset_ne_l30_30572

open Set

namespace MathProof

def P := {1, 2, 3, 4, 5, 6, 7}
def Q := {2, 3, 5, 6}

theorem subset_ne (P Q : Set ℕ) (hP : P = {1, 2, 3, 4, 5, 6, 7}) (hQ : Q = {2, 3, 5, 6}) : Q ⊂ P :=
by
  rw [hP, hQ]
  exact sorry

end MathProof

end subset_ne_l30_30572


namespace number_of_recipes_l30_30796

-- Let's define the necessary conditions.
def cups_per_recipe : ℕ := 2
def total_cups_needed : ℕ := 46

-- Prove that the number of recipes required is 23.
theorem number_of_recipes : total_cups_needed / cups_per_recipe = 23 :=
by
  sorry

end number_of_recipes_l30_30796


namespace friends_who_participate_l30_30779

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30779


namespace distance_of_point_P_to_base_AB_l30_30095

theorem distance_of_point_P_to_base_AB :
  ∀ (P : ℝ) (A B C : ℝ → ℝ)
    (h : ∀ (x : ℝ), A x = B x)
    (altitude : ℝ)
    (area_ratio : ℝ),
  altitude = 6 →
  area_ratio = 1 / 3 →
  (∃ d : ℝ, d = 6 - (2 / 3) * 6 ∧ d = 2) := 
  sorry

end distance_of_point_P_to_base_AB_l30_30095


namespace solve_for_x_l30_30864

theorem solve_for_x (x : ℚ) (h : 5 * (x - 4) = 3 * (3 - 3 * x) + 6) : x = 5 / 2 :=
by {
  sorry
}

end solve_for_x_l30_30864


namespace number_of_distinct_prime_factors_of_B_l30_30533

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30533


namespace max_value_of_a_l30_30634

theorem max_value_of_a {a : ℝ} :
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → (x1 - a * real.sqrt x1 < x2 - a * real.sqrt x2)) →
  a ≤ 2 :=
begin
  sorry
end

end max_value_of_a_l30_30634


namespace friends_who_participate_l30_30774

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30774


namespace expected_ones_in_three_dice_rolls_l30_30044

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30044


namespace angles_on_coordinate_axes_l30_30675

theorem angles_on_coordinate_axes :
  let Sx := {α | ∃ k : ℤ, α = k * Real.pi},
      Sy := {α | ∃ k : ℤ, α = k * Real.pi + (Real.pi / 2)} in
  (Sx ∪ Sy) = {α | ∃ n : ℤ, α = (n * Real.pi) / 2} :=
by
  sorry

end angles_on_coordinate_axes_l30_30675


namespace x_seq_difference_l30_30588

noncomputable def x_seq (n : ℕ) : ℝ :=
  if n = 2 then sqrt (2 + 1/2)
  else if n = 3 then sqrt (2 + real.cbrt(3 + 1/3))
  else if n >= 4 then sqrt (2 + real.cbrt(3 + ( List.range(n-2).map (λ k, real.root (k+4) ((k+4) + 1/(k+4)))).foldl (λ acc x, acc + real.cbrt x) 0 )) 
  else 0

theorem x_seq_difference (n : ℕ) (hn : n ≥ 2) : x_seq (n+1) - x_seq n < 1/(nat.factorial n) :=
by {
  sorry
}

end x_seq_difference_l30_30588


namespace jeans_customer_price_percentage_increase_l30_30803

def manufacturing_cost (x : ℝ) : ℝ := x
def designer_price (x : ℝ) : ℝ := x * 1.35
def distributor_price (y : ℝ) : ℝ := y * 1.25
def customer_price (z : ℝ) : ℝ := z * 1.45
def percentage_increase (original final : ℝ) : ℝ := ((final - original) / original) * 100

theorem jeans_customer_price_percentage_increase (x : ℝ) (hx : 0 < x) :
  percentage_increase x (customer_price (distributor_price (designer_price x))) = 144.69 :=
by
  sorry

end jeans_customer_price_percentage_increase_l30_30803


namespace proof_problem_l30_30917

def proposition_p (a : ℝ) (x : ℝ) : Prop :=
  ax * x + a * x + 1 > 0

def condition_of_p (a : ℝ) : Prop :=
  0 < a ∧ a < 4

def proposition_q : Prop :=
  (∀ x : ℝ, x^2 - 2 * x - 8 > 0 → (x > 4 ∨ x < -2)) ∧
  ¬ (∀ x : ℝ, x > 5 → x^2 - 2 * x - 8 > 0)

def correct_proposition : Prop :=
  (¬ ∀ a : ℝ, 0 < a ∧ a < 4 → ∀ x : ℝ, ax * x + a * x + 1 > 0) ∧ proposition_q

theorem proof_problem : correct_proposition :=
by {
    sorry
}

end proof_problem_l30_30917


namespace distinct_prime_factors_B_l30_30558

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30558


namespace product_of_solutions_is_neg_four_l30_30581

-- Defining the problem and conditions
variable (f : ℝ → ℝ)
variable (hf_continuous : Continuous f)
variable (hf_even : ∀ x : ℝ, f x = f (-x))
variable (hf_monotonic : ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f x < f y)

-- Stating the problem as a Lean theorem
theorem product_of_solutions_is_neg_four :
  (∏ x in {x | f x = f (1 - 1 / (x + 3))}.to_finset.to_list) = -4 :=
sorry

end product_of_solutions_is_neg_four_l30_30581


namespace num_pos_integers_divisible_by_3_4_5_under_500_l30_30352

theorem num_pos_integers_divisible_by_3_4_5_under_500 : 
  (finset.filter (λ n, n % (nat.lcm (nat.lcm 3 4) 5) = 0) (finset.range 500)).card = 8 :=
by
  sorry

end num_pos_integers_divisible_by_3_4_5_under_500_l30_30352


namespace num_of_three_digit_integers_greater_than_217_l30_30865

theorem num_of_three_digit_integers_greater_than_217 : 
  ∃ n : ℕ, n = 82 ∧ ∀ x : ℕ, (217 < x ∧ x < 300) → 200 ≤ x ∧ x ≤ 299 → n = 82 := 
by
  sorry

end num_of_three_digit_integers_greater_than_217_l30_30865


namespace difference_between_x_and_y_l30_30980

theorem difference_between_x_and_y (x y : ℕ) (h₁ : 3 ^ x * 4 ^ y = 59049) (h₂ : x = 10) : x - y = 10 := by
  sorry

end difference_between_x_and_y_l30_30980


namespace marble_problem_l30_30368

-- Define the given conditions
def ratio (red blue green : ℕ) : Prop := red * 3 * 4 = blue * 2 * 4 ∧ blue * 2 * 4 = green * 2 * 3

-- The total number of marbles
def total_marbles (red blue green : ℕ) : ℕ := red + blue + green

-- The number of green marbles is given
def green_marbles : ℕ := 36

-- Proving the number of marbles and number of red marbles
theorem marble_problem
  (red blue green : ℕ)
  (h_ratio : ratio red blue green)
  (h_green : green = green_marbles) :
  total_marbles red blue green = 81 ∧ red = 18 :=
by
  sorry

end marble_problem_l30_30368


namespace number_of_distinct_prime_factors_of_B_l30_30431

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30431


namespace ellipse_equation_of_given_conditions_l30_30186

theorem ellipse_equation_of_given_conditions :
  ∃ a b : ℝ, 
  (a > b) ∧ 
  (a > 0) ∧ 
  (b > 0) ∧ 
  (let c := real.sqrt 2 in a^2 = b^2 + c^2) ∧ 
  (let k := real.sqrt 2 in (abs (c)) = k) ∧ 
  (x_mid = -2 / 3) ∧ 
  (y_mid = 1 / 3) ∧ 
  (- (4 / 3) / a^2 + (2 / 3) / b^2 = 0) → 
  (∀ x y : ℝ, (x^2) / (4) + (y^2) / (2) = 1) :=
by 
  sorry

end ellipse_equation_of_given_conditions_l30_30186


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30565

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30565


namespace which_option_suitable_l30_30185

def optionA := ("Draw 600 pieces", 600)
def optionB := ("Draw 6 pieces", 6)
def optionC := ("Draw 6 pieces (from different factories)", 6)
def optionD := ("Draw 10 pieces", 10)
def numberOfProductsA := 5000
def numberOfProductsB := 2 * 18
def numberOfProductsC := 2 * 18
def numberOfProductsD := 5000
def isMixed (option : String) : Bool :=
  optionB.contains("from same factory")
def isSmallSample (samplesize totalnumber : Nat) : Bool :=
  samplesize < 0.1 * totalnumber

theorem which_option_suitable :
  isMixed optionB ∧ 
  isSmallSample (optionB.snd) numberOfProductsB → 
  optionB = "Suitable option" :=
by sorry

end which_option_suitable_l30_30185


namespace cumulative_profit_higher_than_800_max_average_daily_profit_l30_30793

variable (x : ℕ) (y : ℝ)

-- Define the cumulative profit function
def cumulative_profit (x : ℕ) : ℝ := -0.5 * (x : ℝ)^2 + 60 * (x : ℝ) - 800

-- Define the average daily operating profit function
def average_daily_profit (x : ℕ) : ℝ := cumulative_profit x / (x : ℝ)

-- Prove that for the cumulative profit to be higher than 800, the operating days range should be 40 < x < 80.
theorem cumulative_profit_higher_than_800 (h : cumulative_profit x > 800) : 40 < x ∧ x < 80 :=
sorry

-- Prove that the maximum average daily operating profit is achieved at 400 days.
theorem max_average_daily_profit : 
  (∀ x : ℕ, x > 0 → average_daily_profit x ≤ 20) ∧ (average_daily_profit 400 = 20) :=
sorry

end cumulative_profit_higher_than_800_max_average_daily_profit_l30_30793


namespace problem_l30_30409

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30409


namespace perfect_square_trinomial_k_l30_30977

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), (a * x + b) ^ 2 = x ^ 2 + k * x + 9) → (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l30_30977


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30560

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30560


namespace simplify_exp_l30_30196

theorem simplify_exp : (10^8 / (10 * 10^5)) = 100 := 
by
  -- The proof is omitted; we are stating the problem.
  sorry

end simplify_exp_l30_30196


namespace log_eq_to_roots_l30_30889

theorem log_eq_to_roots (x : ℝ) : log 5 (x^2 - 12 * x) = 3 ↔ (x = 6 + sqrt 161 ∨ x = 6 - sqrt 161) := 
by
  sorry

end log_eq_to_roots_l30_30889


namespace product_of_divisors_has_three_prime_factors_l30_30455

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30455


namespace find_a_if_f_odd_l30_30362

noncomputable def f (a : ℝ) (x : ℝ) := x * log (2^x + 1) / log 2 + a * x

theorem find_a_if_f_odd :
  (∀ x : ℝ, f a (-x) = - f a x) → a = -1 / 2 :=
by
  sorry

end find_a_if_f_odd_l30_30362


namespace find_magnitude_AP_l30_30387

variables {A B C D P : Type} [InnerProductSpace ℝ A B C D P] -- Define the type of points and the inner product space (Euclidean space)
variables (AC AB BC : ℝ) (D_mid_AB : A = 0.5 * B + 0.5 * C) (CD_half_BC : C = 0.5 * B) (P_on_CD : A = (1 - 2 / 3) * C + (2 / 3) * D) -- Additional definitions related to the problem conditions
variables (AP_eqn : A = 2 * A + (1 / 3) * B)

theorem find_magnitude_AP :
  (| 2 * A + \frac{1}{3} * B | = \frac{2 * sqrt 13}{3}) := sorry

end find_magnitude_AP_l30_30387


namespace log_three_one_ninth_l30_30243

theorem log_three_one_ninth : log 3 (1 / 9) = -2 := by
  -- Definitions needed for the proof
  have h1 : 1 / 9 = 3 ^ (-2) := by 
    sorry
  have h2 : log 3 (3 ^ (-2)) = -2 * log 3 3 := by 
    sorry
  have h3 : log 3 3 = 1 := by 
    sorry
  -- Using the above definitions to prove the theorem
  have h3_implies : -2 * log 3 3 = -2 := by 
    rw [h3]
    rw [mul_one]
  rw [← h1, h2]
  exact h3_implies

end log_three_one_ninth_l30_30243


namespace expected_ones_in_three_dice_rolls_l30_30043

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30043


namespace incorrect_relation_l30_30124

theorem incorrect_relation
  (Q R Z N : Set ℝ)
  (hQinR : Q ∈ R)
  (hNsubZ : N ⊆ Z)
  (hNsubneqR : N ⊂ R)
  (hNcapQeqN : N ∩ Q = N) :
  False :=
by
  sorry

end incorrect_relation_l30_30124


namespace number_of_distinct_prime_factors_of_B_l30_30543

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30543


namespace distinct_prime_factors_B_l30_30551

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30551


namespace binary_addition_l30_30701

def bin_to_dec1 := 511  -- 111111111_2 in decimal
def bin_to_dec2 := 127  -- 1111111_2 in decimal

theorem binary_addition : bin_to_dec1 + bin_to_dec2 = 638 := by
  sorry

end binary_addition_l30_30701


namespace polynomial_remainder_zero_l30_30262

open Polynomial

noncomputable def poly1 : Polynomial ℝ := x ^ 68 + x ^ 51 + x ^ 34 + x ^ 17 + 1
noncomputable def poly2 : Polynomial ℝ := x ^ 6 + x ^ 5 + x ^ 4 + x ^ 3 + x ^ 2 + x + 1

theorem polynomial_remainder_zero :
  (poly2 ∣ poly1) := sorry

end polynomial_remainder_zero_l30_30262


namespace sqrt_ineq_l30_30689

theorem sqrt_ineq : (sqrt 3 + sqrt 7) < 2 * sqrt 5 :=
by
  have h1 : (sqrt 3 + sqrt 7) ^ 2 < (2 * sqrt 5) ^ 2,
  {
    calc
      (sqrt 3 + sqrt 7) ^ 2 = 10 + 2 * sqrt 21 : by ring
      ... < 20 : by linarith,
  },
  exact lt_of_pow_lt_pow' h1 one_ne_zero two_pos,

end sqrt_ineq_l30_30689


namespace cost_of_18_pounds_of_apples_l30_30832

theorem cost_of_18_pounds_of_apples :
  (∀ (rate : ℚ), rate = 6 ↔ 6 / 6 = 1) ->
  let rate := 6 : ℚ,
      pounds := 18 : ℚ in
  pounds / 6 * rate = 18 :=
by
  sorry

end cost_of_18_pounds_of_apples_l30_30832


namespace value_of_a4_plus_a_inv4_l30_30002

theorem value_of_a4_plus_a_inv4 (a : ℝ) (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 := by
  sorry

end value_of_a4_plus_a_inv4_l30_30002


namespace lattice_points_count_l30_30962

-- Define a predicate to check if a pair (x, y) is a lattice point on the graph
def isLatticePoint (x y : ℤ) : Prop := x^2 - y^2 = 75

-- Define the count of lattice points on the graph
def countLatticePoints : ℕ :=
  Finset.card (Finset.filter (λ xy : ℤ × ℤ, isLatticePoint xy.fst xy.snd) 
    ((Finset.Icc (-100) 100).product (Finset.Icc (-100) 100)))

-- The statement to be proven is that this count equals 6
theorem lattice_points_count : countLatticePoints = 6 := by
  sorry

end lattice_points_count_l30_30962


namespace smallest_palindrome_in_bases_2_4_l30_30219

def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

def to_digits (n : Nat) (base : Nat) : List Char :=
  Nat.digits base n |> List.map (λ d, Char.ofNat (d + 48))

def is_palindrome_in_base (n base : Nat) : Bool :=
  is_palindrome (to_digits n base)

theorem smallest_palindrome_in_bases_2_4 :
  ∃ n : Nat, n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧
  ∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m :=
  ∃ n : Nat, n = 15 ∧ n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧ (∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m) :=
  sorry

end smallest_palindrome_in_bases_2_4_l30_30219


namespace increasing_interval_of_f_l30_30639

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem about the increasing interval of the function
theorem increasing_interval_of_f : 
  set_of (λ x, ∃ y, f' x > 0) = Iic 0 :=
sorry

end increasing_interval_of_f_l30_30639


namespace maximum_profit_l30_30146

def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def units_sold_per_month : ℕ := 210
def price_increase_effect (x : ℕ) : ℕ := units_sold_per_month - 10 * x
def profit_function (x : ℕ) : ℕ := (price_increase_effect x) * (initial_selling_price_per_unit + x - cost_price_per_unit)

theorem maximum_profit :
  profit_function 5 = 2400 ∧ profit_function 6 = 2400 :=
by
  sorry

end maximum_profit_l30_30146


namespace count_irrational_numbers_l30_30180

def is_irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

noncomputable def num_list : List ℝ := [
  3.14159,
  -real.cbrt 9,
  0.131131113.repr.to_real,
  -real.pi,
  real.sqrt 25,
  real.cbrt 64,
  -1 / 7
]

theorem count_irrational_numbers :
  (num_list.filter is_irrational).length = 3 := 
sorry

end count_irrational_numbers_l30_30180


namespace number_of_friends_is_five_l30_30723

def total_cards : ℕ := 455
def cards_per_friend : ℕ := 91

theorem number_of_friends_is_five (n : ℕ) (h : total_cards = n * cards_per_friend) : n = 5 := 
sorry

end number_of_friends_is_five_l30_30723


namespace ab_greater_than_1_l30_30939

noncomputable def log10_abs (x : ℝ) : ℝ :=
  abs (Real.logb 10 x)

theorem ab_greater_than_1
  {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a < b)
  (hf : log10_abs a < log10_abs b) : a * b > 1 := by
  sorry

end ab_greater_than_1_l30_30939


namespace smallest_palindromic_integer_l30_30214

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n
  digits = List.reverse digits

theorem smallest_palindromic_integer :
  ∃ (n : ℕ), n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ (∀ m, m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 → n ≤ m) :=
begin
  sorry
end

end smallest_palindromic_integer_l30_30214


namespace ellipse_equation_constant_chord_length_l30_30293

-- Define the ellipse with given conditions
def ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ∃ (x y : ℝ), (x = 1 ∧ y = 3/2 ∧ (x^2 / a^2 + y^2 / b^2 = 1)) ∧ 
    ∃ (c : ℝ), (c = 1 ∧ c^2 = a^2 - b^2)

-- Define the point P and the focus F
def Point (x y : ℝ) := (x, y)
def Focus (x y : ℝ) := (x, y)

-- Main theorem for part (Ⅰ)
theorem ellipse_equation (a b : ℝ) (h : ellipse a b) :
  ∀ x y, x = 1 → y = 3/2 → (x^2 / 4 + y^2 / 3 = 1) :=
sorry

-- Supplementary definitions for part (Ⅱ)
def intersects_y_axis (P D : Point) (M : Point) : Prop :=
  ∃ (x_PD : ℝ), P.1 + x_PD * (D.1 - P.1) = 0 ∧ M = (0, P.2 + x_PD * (D.2 - P.2))

def symmetrical_points (D E : Point) : Prop :=
  D.1 = -E.1 ∧ D.2 = -E.2

-- Main theorem for part (Ⅱ)
theorem constant_chord_length (P D E M N : Point) 
  (h_sym : symmetrical_points D E) (h1 : intersects_y_axis P D M) (h2 : intersects_y_axis P E N)
  (h_eq : P.2 = 3/2) :
  ∀ a b : ℝ, ellipse a b → 
    let l := (3 * (3 / 4)^1/2) in -- This is the length of the chord MN intercepted by y = 3/2
    l = 3 * (3 ^ 1/2) / 2 :=
sorry

end ellipse_equation_constant_chord_length_l30_30293


namespace number_of_distinct_prime_factors_of_B_l30_30537

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30537


namespace product_of_divisors_has_three_prime_factors_l30_30465

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30465


namespace num_idempotent_functions_l30_30348

open Finset Function

theorem num_idempotent_functions :
  let n := 5
  let f_set := finset.fin_range n
  let count := ∑ k in f_set.Powerset, k.card.factorial * (n - k.card) ^ (n - k.card)
  count = 196 :=
by
  sorry

end num_idempotent_functions_l30_30348


namespace distinct_prime_factors_of_product_of_divisors_l30_30420

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30420


namespace product_of_divisors_has_three_prime_factors_l30_30464

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30464


namespace monotonic_intervals_unique_extreme_point_l30_30331

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := ln x - a * x - 1

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * (f x a) + (1 / 2) * x^2 + 2 * x

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (∀ x > 0, (f x a) = ln x - a * x - 1 ∧ 
    (0 < x ∧ x < 1 / a → (f x a)' > 0) ∧ 
    (x > 1 / a → (f x a)' < 0)) ∨ 
  (a ≤ 0 → ∀ x > 0, (f x a)' > 0) :=
sorry

theorem unique_extreme_point (h : g (λ x, x * (ln x - x - 1) + (1 / 2) * x^2 + 2 * x) = g x 1) :
  let m := ⌊λ x, g x 1 ∈ (m, m+1) ∧ unique_extreme_pt x m⌋ in
  m = 0 ∨ m = 3 :=
sorry

end monotonic_intervals_unique_extreme_point_l30_30331


namespace hide_and_seek_problem_l30_30742

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30742


namespace no_triangle_with_two_heights_greater_than_100_and_area_less_than_1_l30_30389

theorem no_triangle_with_two_heights_greater_than_100_and_area_less_than_1 (h1 h2 : ℝ) (a b : ℝ) :
  h1 > 100 → h2 > 100 → ¬(∃ (S : ℝ), S = (1 / 2) * a * h1 ∧ S < 1) :=
by {
  intros h1_gt_100 h2_gt_100,
  have a_geq_100 : a ≥ 100 := sorry,
  have S_geq_5000 : (1 / 2) * a * h1 ≥ 5000 := sorry,
  have S_geq_1 : (1 / 2) * a * h1 > 1 := sorry,
  contradiction
}

end no_triangle_with_two_heights_greater_than_100_and_area_less_than_1_l30_30389


namespace find_f_prime_at_1_l30_30314

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 * f' 1
noncomputable def f' (x : ℝ) : ℝ := HasDerivAt f (1 / x + 2 * x * f' 1) x

theorem find_f_prime_at_1 : f' 1 = -1 :=
sorry

end find_f_prime_at_1_l30_30314


namespace remainder_correct_l30_30720

noncomputable def p (z : ℚ) := 4 * z^3 + 5 * z^2 - 20 * z + 7
noncomputable def d (z : ℚ) := 4 * z - 3
noncomputable def q (z : ℚ) := z^2 + 2 * z + 1 / 4
noncomputable def r (z : ℚ) := -15 * z + 31 / 4

theorem remainder_correct :
  ∀ z : ℚ, p(z) = d(z) * q(z) + r(z) :=
by
  intro z
  sorry

end remainder_correct_l30_30720


namespace smallest_n_terminating_decimal_l30_30107

theorem smallest_n_terminating_decimal : ∃ n : ℕ, (∀ m : ℕ, m < n → (∀ k : ℕ, (n = 103 + k) → (∃ a b : ℕ, k = 2^a * 5^b)) → (k ≠ 0 → k = 125)) ∧ n = 22 := 
sorry

end smallest_n_terminating_decimal_l30_30107


namespace number_of_distinct_prime_factors_of_B_l30_30432

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30432


namespace complex_fraction_addition_l30_30841

theorem complex_fraction_addition :
  ( (1 + Complex.i)^2 / (1 + 2 * Complex.i) + (1 - Complex.i)^2 / (2 - Complex.i) = (6 - 2 * Complex.i) / 5) := 
by 
  sorry

end complex_fraction_addition_l30_30841


namespace sum_of_powers_of_four_to_50_l30_30194

theorem sum_of_powers_of_four_to_50 :
  2 * (Finset.sum (Finset.range 51) (λ x => x^4)) = 1301700 := by
  sorry

end sum_of_powers_of_four_to_50_l30_30194


namespace number_of_distinct_prime_factors_of_B_l30_30539

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30539


namespace mean_of_solutions_l30_30257

open Polynomial

noncomputable def poly : Polynomial ℚ :=
  Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 3 * Polynomial.X ^ 2 + Polynomial.C (-10) * Polynomial.X

theorem mean_of_solutions : 
  ( (roots poly).map (λ x, x / (roots poly).length)).sum = -1 := 
sorry

end mean_of_solutions_l30_30257


namespace black_to_white_ratio_l30_30851

/-- 
Given:
- The original square pattern consists of 13 black tiles and 23 white tiles
- Attaching a border of black tiles around the original 6x6 square pattern results in an 8x8 square pattern

To prove:
- The ratio of black tiles to white tiles in the extended 8x8 pattern is 41/23.
-/
theorem black_to_white_ratio (b_orig w_orig b_added b_total w_total : ℕ) 
  (h_black_orig: b_orig = 13)
  (h_white_orig: w_orig = 23)
  (h_size_orig: 6 * 6 = b_orig + w_orig)
  (h_size_ext: 8 * 8 = (b_orig + b_added) + w_orig)
  (h_b_added: b_added = 28)
  (h_b_total: b_total = b_orig + b_added)
  (h_w_total: w_total = w_orig)
  :
  b_total / w_total = 41 / 23 :=
by
  sorry

end black_to_white_ratio_l30_30851


namespace sum_of_intersections_l30_30642

theorem sum_of_intersections :
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (∀ x y : ℝ, y = (x - 2)^2 ↔ x + 1 = (y - 2)^2) ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 20) :=
sorry

end sum_of_intersections_l30_30642


namespace number_of_distinct_prime_factors_of_B_l30_30541

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30541


namespace alex_play_friends_with_l30_30739

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30739


namespace maximum_warehouse_area_and_iron_bars_length_l30_30139

noncomputable def warehouse_max_area : ℝ :=
  100

noncomputable def front_iron_bar_length : ℝ :=
  15

theorem maximum_warehouse_area_and_iron_bars_length (budget height back_wall_front front_cost side_cost roof_cost : ℝ)
  (h1 : budget = 3200)
  (h2 : height = height) -- height is constant
  (h3 : back_wall_front = 0) -- no cost for back wall
  (h4 : front_cost = 40) -- cost per meter of front iron bars
  (h5 : side_cost = 45) -- cost per meter of side walls
  (h6 : roof_cost = 20) -- cost per square meter of roof
  :
  ∃ S L, S = warehouse_max_area ∧ L = front_iron_bar_length :=
begin
  sorry
end

end maximum_warehouse_area_and_iron_bars_length_l30_30139


namespace man_l30_30813

theorem man's_age_twice_son_in_2_years 
  (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = 38) (h3 : M = S + 20) : 
  ∃ X : ℕ, (M + X = 2 * (S + X)) ∧ X = 2 :=
by
  sorry

end man_l30_30813


namespace train_P_speed_calculation_l30_30692

noncomputable def speed_of_train_from_P
  (distance_PQ : ℕ)
  (time_P_start : ℕ)
  (time_Q_start : ℕ)
  (speed_Q : ℕ)
  (meeting_time : ℕ)
  : ℕ :=
if h : 2 * 25 + (meeting_time - time_P_start) * (65 - 2 * 25) = 65
then (65 - speed_Q) / 2
else 0

theorem train_P_speed_calculation :
  ∀ (distance_PQ : ℕ) (time_P_start : ℕ) (time_Q_start : ℕ) (speed_Q : ℕ) (meeting_time : ℕ),
  distance_PQ = 65 → 
  time_P_start = 7 → 
  time_Q_start = 8 → 
  speed_Q = 25 → 
  meeting_time = 9 → 
  speed_of_train_from_P distance_PQ time_P_start time_Q_start speed_Q meeting_time = 20 := 
by {
  intros,
  rw [distance_PQ, time_P_start, time_Q_start, speed_Q, meeting_time],
  exact rfl,
  sorry -- skipping the detailed proof
}

end train_P_speed_calculation_l30_30692


namespace product_of_divisors_has_three_prime_factors_l30_30461

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30461


namespace cos_double_angle_l30_30278

theorem cos_double_angle (θ : ℝ) (h₁ : sin θ + cos θ = 1/5) (h₂ : π/2 ≤ θ ∧ θ ≤ 3*π/4) :
  cos (2*θ) = -7/25 :=
by
  sorry

end cos_double_angle_l30_30278


namespace integer_solution_zero_l30_30613

theorem integer_solution_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_solution_zero_l30_30613


namespace sum_of_factors_24_l30_30711

theorem sum_of_factors_24 : (∑ n in Finset.filter (λ d, 24 % d = 0) (Finset.range 25), n) = 60 := 
by 
  sorry

end sum_of_factors_24_l30_30711


namespace sum_first_2010_terms_eq_zero_l30_30335

noncomputable def sequence : ℕ → ℤ
| 0       := 2009
| 1        := 2010
| 2        := 1
| 3        := -2009
| 4        := -2010
| 5        := -1
| (n + 6) := sequence n

theorem sum_first_2010_terms_eq_zero :
  (Finset.range 2010).sum (λ n, sequence n) = 0 :=
sorry

end sum_first_2010_terms_eq_zero_l30_30335


namespace distinct_prime_factors_of_product_of_divisors_l30_30509

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30509


namespace range_of_m_l30_30933

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (hf : ∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ y, f y = x) :
  (∀ x, ∃ y, y = f (x + m) - f (x - m)) →
  -1 ≤ m ∧ m ≤ 1 :=
by
  intro hF
  sorry

end range_of_m_l30_30933


namespace probability_divisible_by_8_l30_30869

def spinner_outcomes : set ℕ := {1, 2, 3}

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def three_digit_number (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

theorem probability_divisible_by_8 :
  (∑ x in spinner_outcomes, 
     ∑ y in spinner_outcomes, 
       ∑ z in spinner_outcomes, 
         if is_divisible_by_8 (three_digit_number x y z) then (1 : ℝ) else 0) / 
  (∑ x in spinner_outcomes, 
     ∑ y in spinner_outcomes, 
       ∑ z in spinner_outcomes, 
         1) = 1 / 9 :=
by sorry

end probability_divisible_by_8_l30_30869


namespace must_divide_l30_30583

-- Proving 5 is a divisor of q

variables {p q r s : ℕ}

theorem must_divide (h1 : Nat.gcd p q = 30) (h2 : Nat.gcd q r = 42)
                   (h3 : Nat.gcd r s = 66) (h4 : 80 < Nat.gcd s p)
                   (h5 : Nat.gcd s p < 120) :
                   5 ∣ q :=
sorry

end must_divide_l30_30583


namespace emmy_rosa_ipods_l30_30238

theorem emmy_rosa_ipods :
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  Emmy_left + Rosa_ipods = 12 :=
by
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  sorry

end emmy_rosa_ipods_l30_30238


namespace floor_ineq_solution_l30_30573

-- Defining the problem statement
theorem floor_ineq_solution {x : ℝ} (h : (floor x)^2 - 5 * floor x + 6 ≤ 0) :
  2 ≤ x ∧ x < 4 :=
by 
  sorry  -- Placeholder for the proof

end floor_ineq_solution_l30_30573


namespace problem_l30_30592

theorem problem (a b c d : ℝ) (h1 : 2 + real.sqrt 2 = a + b) (h2 : 4 - real.sqrt 2 = c + d) 
  (ha : a = 3) (hb : b = real.sqrt 2 - 1) (hc : c = 2) (hd : d = 2 - real.sqrt 2) : 
  (b + d) / (a * c) = 1 / 6 :=
by
  rw [hb, hd, ha, hc]
  sorry

end problem_l30_30592


namespace inequality_solution_l30_30873

theorem inequality_solution (y : ℝ) : 
  (y^3 / (y + 2) ≥ (3 / (y - 2)) + (9 / 4)) ↔ (y ∈ set.Ioo (-2 : ℝ) (2 : ℝ) ∪ set.Ici (3 : ℝ)) := 
by
  sorry

end inequality_solution_l30_30873


namespace janos_walked_distance_l30_30397

-- Definitions based on the conditions:
def average_speed : ℝ := 42 -- in km/h
def usual_departure_time : ℝ := 16 -- 4:00 PM as hours
def usual_arrival_home_time : ℝ := 17 -- 5:00 PM as hours
def saved_time : ℝ := 10 / 60 -- saved 10 minutes in hours
def arrival_home_time_on_this_day : ℝ := usual_arrival_home_time - saved_time
def meeting_time_reduction : ℝ := saved_time / 2 -- met 5 minutes earlier

-- Distance calculation based on reduced meeting time:
def distance_walked : ℝ := average_speed * meeting_time_reduction

-- Proof statement:
theorem janos_walked_distance :
  distance_walked = 3.5 :=
by
  -- Inserting the actual mathematical statement to check correctness
  have h1 : distance_walked = average_speed * 5 / 60, by sorry
  rw h1
  have h2 : 42 * 5 / 60 = 3.5, by norm_num
  rw h2
  sorry

end janos_walked_distance_l30_30397


namespace sum_l_j_leq_mn_l30_30298

theorem sum_l_j_leq_mn (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n)
  (a : ℕ → ℕ)
  (a_periodic : ∀ i > n, a i = a (i - n))
  (l : Π j, 1 ≤ j → j ≤ n → ℕ)
  (l_def : ∀ j (h1 : 1 ≤ j) (h2 : j ≤ n), m ∣ ∑ k in range (l j h1 h2), a (j + k) - 1)
  : ∑ j in range (n - 1), l j (by linarith) (by linarith) ≤ m * n := sorry

end sum_l_j_leq_mn_l30_30298


namespace number_of_distinct_prime_factors_of_B_l30_30536

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30536


namespace area_of_triangle_l30_30009

theorem area_of_triangle {a b c : ℝ} (S : ℝ) (h1 : (a^2) * (Real.sin C) = 4 * (Real.sin A))
                          (h2 : (a + c)^2 = 12 + b^2)
                          (h3 : S = Real.sqrt ((1/4) * (a^2 * c^2 - ( (a^2 + c^2 - b^2)/2 )^2))) :
  S = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l30_30009


namespace problem_l30_30413

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30413


namespace log_base_4_of_64_l30_30246

theorem log_base_4_of_64 : ∃ (x : ℝ), (4 ^ x = 64) ∧ (x = 3) :=
by 
  have h : 4 ^ 3 = 64 := by norm_num
  use 3
  exact ⟨h, rfl⟩

end log_base_4_of_64_l30_30246


namespace percentage_problem_l30_30888

theorem percentage_problem 
    (y : ℝ)
    (h₁ : 0.47 * 1442 = 677.74)
    (h₂ : (677.74 - (y / 100) * 1412) + 63 = 3) :
    y = 52.25 :=
by sorry

end percentage_problem_l30_30888


namespace number_of_distinct_prime_factors_of_B_l30_30534

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30534


namespace triangle_max_perimeter_l30_30911

theorem triangle_max_perimeter (a b c : ℝ) (h1 : a = 1) (h2 : 2 * Real.cos (Real.acos ((a^2 + b^2 - c^2) / (2*a*b))) + c = 2 * b) :
  1 + b + c ≤ 3 :=
by {
  sorry
}

end triangle_max_perimeter_l30_30911


namespace orthogonal_projection_ratio_constant_and_equal_sine_ratio_l30_30234

theorem orthogonal_projection_ratio_constant_and_equal_sine_ratio
  (a b c : Line) (S P : Point) (Pa Pb : Point) (m n : ℝ)
  (hP_on_c : P ∈ c) 
  (orthogonal_proj_a : orthogonal_projection P a Pa)
  (orthogonal_proj_b : orthogonal_projection P b Pb)
  (hm : m = |P - Pa|)
  (hn : n = |P - Pb|)
  (same_half_plane : same_half_plane Pa Pb c) :
  (abc = m/n = sin (angle_between c a) / sin (angle_between c b)) := by
  sorry

end orthogonal_projection_ratio_constant_and_equal_sine_ratio_l30_30234


namespace problem_l30_30411

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30411


namespace prime_factors_of_B_l30_30523

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30523


namespace hide_and_seek_l30_30766

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30766


namespace friends_who_participate_l30_30775

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30775


namespace right_triangle_condition_l30_30912

theorem right_triangle_condition (a b c : ℝ) (h : c^2 - a^2 = b^2) : 
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A = 90 ∧ B + C = 90 :=
by sorry

end right_triangle_condition_l30_30912


namespace smallest_palindrome_in_base2_and_4_l30_30224

-- Define a function to check if a number's representation is a palindrome.
def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

-- Convert a number n to a given base and represent as a list of digits.
def to_base (n : ℕ) (b : ℕ) : List ℕ :=
  if h : b > 1 then
    let rec convert (n : ℕ) : List ℕ :=
      if n = 0 then [] else (n % b) :: convert (n / b)
    convert n 
  else []

-- Convert the list of digits to a list of characters.
def digits_to_chars (ds : List ℕ) : List Char :=
  ds.map (λ d => (Char.ofNat (d + 48))) -- Adjust ASCII value for digit representation

-- Define a function to check if a number is a palindrome in a specified base.
def is_palindromic_in_base (n base : ℕ) : Bool :=
  is_palindrome (digits_to_chars (to_base n base))

-- Define the main claim.
theorem smallest_palindrome_in_base2_and_4 : ∃ n, n > 10 ∧ is_palindromic_in_base n 2 ∧ is_palindromic_in_base n 4 ∧ ∀ m, m > 10 ∧ is_palindromic_in_base m 2 ∧ is_palindromic_in_base m 4 -> n ≤ m :=
by
  exists 15
  sorry

end smallest_palindrome_in_base2_and_4_l30_30224


namespace axis_angle_set_l30_30673

def is_x_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def is_y_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

def is_axis_angle (α : ℝ) : Prop := ∃ n : ℤ, α = (n * Real.pi) / 2

theorem axis_angle_set : 
  (∀ α : ℝ, is_x_axis_angle α ∨ is_y_axis_angle α ↔ is_axis_angle α) :=
by 
  sorry

end axis_angle_set_l30_30673


namespace probability_red_is_two_fifths_l30_30171

-- Define the durations
def red_light_duration : ℕ := 30
def yellow_light_duration : ℕ := 5
def green_light_duration : ℕ := 40

-- Define total cycle duration
def total_cycle_duration : ℕ :=
  red_light_duration + yellow_light_duration + green_light_duration

-- Define the probability function
def probability_of_red_light : ℚ :=
  red_light_duration / total_cycle_duration

-- The theorem statement to prove
theorem probability_red_is_two_fifths :
  probability_of_red_light = 2/5 := sorry

end probability_red_is_two_fifths_l30_30171


namespace log_three_one_ninth_l30_30242

theorem log_three_one_ninth : log 3 (1 / 9) = -2 := by
  -- Definitions needed for the proof
  have h1 : 1 / 9 = 3 ^ (-2) := by 
    sorry
  have h2 : log 3 (3 ^ (-2)) = -2 * log 3 3 := by 
    sorry
  have h3 : log 3 3 = 1 := by 
    sorry
  -- Using the above definitions to prove the theorem
  have h3_implies : -2 * log 3 3 = -2 := by 
    rw [h3]
    rw [mul_one]
  rw [← h1, h2]
  exact h3_implies

end log_three_one_ninth_l30_30242


namespace joan_mortgage_payment_total_l30_30395

theorem joan_mortgage_payment_total 
  (a₁ : ℕ) (r : ℕ) (n : ℕ)
  (h1 : a₁ = 100)
  (h2 : r = 3)
  (h3 : n = 7) :
  let S := ∑ i in Finset.range n, a₁ * r^i
  in S = 109300 :=
by
  sorry

end joan_mortgage_payment_total_l30_30395


namespace third_number_correct_l30_30112

-- Given that the row of Pascal's triangle with 51 numbers corresponds to the binomial coefficients of 50.
def third_number_in_51_pascal_row : ℕ := Nat.choose 50 2

-- Prove that the third number in this row of Pascal's triangle is 1225.
theorem third_number_correct : third_number_in_51_pascal_row = 1225 := 
by 
  -- Calculation part can be filled in for the full proof.
  sorry

end third_number_correct_l30_30112


namespace distinct_prime_factors_of_B_l30_30452

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30452


namespace problem_l30_30414

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30414


namespace distinct_prime_factors_of_product_of_divisors_l30_30416

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30416


namespace hide_and_seek_problem_l30_30746

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30746


namespace midpoint_in_good_set_iff_l30_30401

def is_good_set (S : set (ℝ × ℝ)) (θ : ℝ) : Prop :=
  ∀ (P : ℝ × ℝ), P ∈ S → ∀ (Q : ℝ × ℝ), (Q ∈ S → P = Q ∨ ∃ γ, (Q.1 = P.1 + γ * (Q.1 - P.1) - γ * (Q.2 - P.2) * cos θ - γ * (Q.2 - P.2) * sin θ) ∧
                                          (Q.2 = P.2 + γ * (Q.1 - P.1) * cos θ - γ * (Q.2 - P.2) * cos θ + γ * (Q.2 - P.2) * sin θ))

def midpoint_in_good_set {S : set (ℝ × ℝ)} (θ : ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ), A ∈ S → B ∈ S → ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∈ S

theorem midpoint_in_good_set_iff {r : ℝ} (h : r ∈ set.Icc (-1:ℝ) 1) (θ := real.arccos r) :
  midpoint_in_good_set θ ↔ ∃ n : ℕ, n > 0 ∧ r = 1 - (1:ℝ) / (4 * n) :=
by
  sorry

end midpoint_in_good_set_iff_l30_30401


namespace alex_plays_with_friends_l30_30751

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30751


namespace smallest_palindrome_in_base2_and_4_l30_30220

-- Define a function to check if a number's representation is a palindrome.
def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

-- Convert a number n to a given base and represent as a list of digits.
def to_base (n : ℕ) (b : ℕ) : List ℕ :=
  if h : b > 1 then
    let rec convert (n : ℕ) : List ℕ :=
      if n = 0 then [] else (n % b) :: convert (n / b)
    convert n 
  else []

-- Convert the list of digits to a list of characters.
def digits_to_chars (ds : List ℕ) : List Char :=
  ds.map (λ d => (Char.ofNat (d + 48))) -- Adjust ASCII value for digit representation

-- Define a function to check if a number is a palindrome in a specified base.
def is_palindromic_in_base (n base : ℕ) : Bool :=
  is_palindrome (digits_to_chars (to_base n base))

-- Define the main claim.
theorem smallest_palindrome_in_base2_and_4 : ∃ n, n > 10 ∧ is_palindromic_in_base n 2 ∧ is_palindromic_in_base n 4 ∧ ∀ m, m > 10 ∧ is_palindromic_in_base m 2 ∧ is_palindromic_in_base m 4 -> n ≤ m :=
by
  exists 15
  sorry

end smallest_palindrome_in_base2_and_4_l30_30220


namespace hide_and_seek_problem_l30_30741

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30741


namespace radius_of_circle_l30_30984

theorem radius_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) (h_center : center = (2, -3)) (h_point : point = (5, -7)) :
  sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = 5 :=
by
  rw [h_center, h_point]
  sorry

end radius_of_circle_l30_30984


namespace tan_neg_five_pi_over_three_l30_30268

theorem tan_neg_five_pi_over_three : Real.tan (-5 * Real.pi / 3) = Real.sqrt 3 := 
by 
  sorry

end tan_neg_five_pi_over_three_l30_30268


namespace sum_of_squares_of_consecutive_even_numbers_l30_30131

theorem sum_of_squares_of_consecutive_even_numbers :
  ∃ (x : ℤ), x + (x + 2) + (x + 4) + (x + 6) = 36 → (x ^ 2 + (x + 2) ^ 2 + (x + 4) ^ 2 + (x + 6) ^ 2 = 344) :=
by
  sorry

end sum_of_squares_of_consecutive_even_numbers_l30_30131


namespace pascal_third_number_in_51_row_l30_30114

-- Definition and conditions
def pascal_row_num := 50
def third_number_index := 2

-- Statement of the problem
theorem pascal_third_number_in_51_row : 
  (nat.choose pascal_row_num third_number_index) = 1225 :=
by {
  -- The proof step will be skipped using sorry
  sorry
}

end pascal_third_number_in_51_row_l30_30114


namespace friends_who_participate_l30_30773

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30773


namespace expected_ones_in_three_dice_rolls_l30_30042

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30042


namespace distinct_prime_factors_B_l30_30556

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30556


namespace alex_plays_with_friends_l30_30750

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30750


namespace total_limes_picked_l30_30175

theorem total_limes_picked (Alyssa_limes Mike_limes : ℕ) 
        (hAlyssa : Alyssa_limes = 25) (hMike : Mike_limes = 32) : 
       Alyssa_limes + Mike_limes = 57 :=
by {
  sorry
}

end total_limes_picked_l30_30175


namespace distinct_prime_factors_of_B_l30_30469

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30469


namespace expected_number_of_ones_l30_30063

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30063


namespace hide_and_seek_l30_30783

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30783


namespace modulus_of_complex_fraction_l30_30861

theorem modulus_of_complex_fraction :
  ( ∀ a b : ℂ, abs (a / b) = abs a / abs b ) →
  ( ∀ a b : ℂ, abs (a + b * I) = (a^2 + b^2)^0.5 ) →
  abs ((-5 + I) / (2 - 3 * I)) = real.sqrt 2 :=
by
  intros h1 h2
  have h3 := h2 (-5) 1
  have h4 := h2 2 (-3)
  rw h1 (-5 + I) (2 - 3 * I)
  rw h3
  rw h4
  sorry

end modulus_of_complex_fraction_l30_30861


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30501

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30501


namespace slope_through_points_l30_30710

/- 
  Prove that the slope of the line passing through the points (4, -3) and (-1, 7) is -2.
-/

theorem slope_through_points : 
  let P1 := (4 : ℝ, -3 : ℝ)
  let P2 := (-1 : ℝ, 7 : ℝ)
  let x1 := P1.1
  let y1 := P1.2
  let x2 := P2.1
  let y2 := P2.2
  (y2 - y1) / (x2 - x1) = -2 :=
by 
  sorry

end slope_through_points_l30_30710


namespace diagonal_length_l30_30650

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l30_30650


namespace water_added_l30_30169

theorem water_added (x : ℝ) (salt_percent_initial : ℝ) (evaporation_fraction : ℝ) 
(salt_added : ℝ) (resulting_salt_percent : ℝ) 
(hx : x = 119.99999999999996) (h_initial_salt : salt_percent_initial = 0.20) 
(h_evap_fraction : evaporation_fraction = 1/4) (h_salt_added : salt_added = 16)
(h_resulting_salt_percent : resulting_salt_percent = 1/3) : 
∃ (water_added : ℝ), water_added = 30 :=
by
  sorry

end water_added_l30_30169


namespace smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30205

def is_palindrome_in_base (n : ℕ) (b : ℕ) : Prop :=
  let digits : List ℕ := (nat.to_digits b n)
  digits = digits.reverse

theorem smallest_palindrome_in_bases_2_and_4 :
  ∀ n : ℕ, 10 < n ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 → n ≥ 17 :=
begin
  intros n hn,
  have := (by norm_num : 16 < 17),
  cases hn with hn1 hn2,
  cases hn2 with hn3 hn4,
  linarith,
end

theorem smallest_palindrome_in_bases_2_and_4_17 :
  is_palindrome_in_base 17 2 ∧ is_palindrome_in_base 17 4 :=
begin
  split;
  { unfold is_palindrome_in_base, norm_num, refl },
end

end smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30205


namespace expected_ones_on_three_dice_l30_30091

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30091


namespace sufficient_not_necessary_l30_30966

variables (A B : Prop)

theorem sufficient_not_necessary (h : B → A) : ¬(A → B) :=
by sorry

end sufficient_not_necessary_l30_30966


namespace y_plus_q_eq_2q_plus_5_l30_30979

theorem y_plus_q_eq_2q_plus_5 (y q : ℝ) (h1 : |y - 5| = q) (h2 : y > 5) : y + q = 2q + 5 :=
sorry

end y_plus_q_eq_2q_plus_5_l30_30979


namespace number_of_distinct_prime_factors_of_B_l30_30540

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30540


namespace alpha_irrational_l30_30227

noncomputable def alpha : ℚ :=
  let seq := (λ n : ℕ, 1987 ^ n) in
  (0.1 + seq 1 / 10 + seq 2 / 10^2 + seq 3 / 10^3 + seq 4 / 10^4 + seq 5 / 10^5 + ··· + seq n / 10^n)

theorem alpha_irrational : ¬ (∃ r : ℚ, r = α) := sorry

end alpha_irrational_l30_30227


namespace unique_function_f_l30_30232

-- Define the type N* which is the set of positive natural numbers
def N_star := {n : ℕ // n > 0}

-- Define the function type
def function_condition (f : N_star → N_star) : Prop :=
  ∀ (x y : N_star), ∃ (k : ℕ), x.val * f(x).val + 2 * x.val * f(y).val + (f(y).val)^2 = k^2

-- Define the main problem statement
theorem unique_function_f (f : N_star → N_star) : function_condition f → (∀ x, f(x) = x) :=
  by 
    sorry

end unique_function_f_l30_30232


namespace sum_of_first_n_terms_l30_30721

-- Define the sequence aₙ
def a (n : ℕ) : ℕ := 2 * n - 1

-- Prove that the sum of the first n terms of the sequence is n²
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range (n+1)).sum a = n^2 :=
by sorry -- Proof is skipped

end sum_of_first_n_terms_l30_30721


namespace expected_ones_on_three_dice_l30_30087

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30087


namespace distance_from_point_to_plane_example_l30_30953

structure Point where
  x : ℝ
  y : ℝ
  z: ℝ

def distance_point_to_plane (P A : Point) (n : Point) : ℝ :=
  abs ((n.x * (A.x - P.x)) + (n.y * (A.y - P.y)) + (n.z * (A.z - P.z))) / (Real.sqrt (n.x^2 + n.y^2 + n.z^2))

theorem distance_from_point_to_plane_example :
  let P := {x := 4, y := 3, z := 2}
  let A := {x := 2, y := 3, z := 1}
  let n := {x := 1, y := 0, z := -1}
  distance_point_to_plane P A n = Real.sqrt 2 / 2 :=
by
  sorry

end distance_from_point_to_plane_example_l30_30953


namespace distinct_prime_factors_B_l30_30557

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30557


namespace train_length_l30_30823

theorem train_length (speed_kmh : ℕ) (cross_time : ℕ) (h_speed : speed_kmh = 54) (h_time : cross_time = 9) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_m := speed_ms * cross_time
  length_m = 135 := by
  sorry

end train_length_l30_30823


namespace expected_number_of_ones_l30_30069

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30069


namespace distinct_prime_factors_of_B_l30_30444

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30444


namespace find_ab_for_equation_l30_30261

theorem find_ab_for_equation (a b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (∃ x, x = 12 - x1 - x2) ∧ (a * x1^2 - 24 * x1 + b) / (x1^2 - 1) = x1
  ∧ (a * x2^2 - 24 * x2 + b) / (x2^2 - 1) = x2) ∧ (a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819) := sorry

end find_ab_for_equation_l30_30261


namespace hide_and_seek_l30_30764

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30764


namespace surface_area_of_cube_inscribed_in_sphere_l30_30266

theorem surface_area_of_cube_inscribed_in_sphere (r : ℝ) (h : r = 5) : 
  let s := (10 * Real.sqrt 3) / 3 in
  let area := 6 * s^2 in
  area = 200 :=
by
  sorry

end surface_area_of_cube_inscribed_in_sphere_l30_30266


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30499

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30499


namespace expected_ones_three_dice_l30_30082

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30082


namespace sequence_properties_l30_30319

noncomputable def S_n (n : ℕ) : ℕ := 2 * n ^ 2

noncomputable def a (n : ℕ) : ℕ := 4 * n - 2

noncomputable def b (n : ℕ) : ℝ := 2 / (4 ^ (n - 1))

noncomputable def c (n : ℕ) : ℝ := a n / b n

noncomputable def T (n : ℕ) : ℝ := ∑ i in finset.range n, c (i + 1)

theorem sequence_properties (n : ℕ) (hn : n ≥ 1) :
  a n = 4 * n - 2 ∧
  b n = 2 / (4 ^ (n - 1)) ∧
  T n = (1 / 9) * ((6 * n - 5) * 4 ^ n + 5) := by
  sorry

end sequence_properties_l30_30319


namespace inheritance_amount_l30_30398

theorem inheritance_amount (x : ℝ) 
    (federal_tax : ℝ := 0.25 * x) 
    (remaining_after_federal_tax : ℝ := x - federal_tax) 
    (state_tax : ℝ := 0.15 * remaining_after_federal_tax) 
    (total_taxes : ℝ := federal_tax + state_tax) 
    (taxes_paid : total_taxes = 15000) : 
    x = 41379 :=
sorry

end inheritance_amount_l30_30398


namespace smallest_possible_gcd_l30_30973

theorem smallest_possible_gcd (m n p : ℕ) (h1 : Nat.gcd m n = 180) (h2 : Nat.gcd m p = 240) :
  ∃ k, k = Nat.gcd n p ∧ k = 60 := by
  sorry

end smallest_possible_gcd_l30_30973


namespace find_a99_l30_30320

-- Define the arithmetic sequence.
def arithmetic_seq (a1 d : ℚ) (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence.
def sum_arithmetic_seq (a1 d : ℚ) (n : ℕ) : ℚ := n / 2 * (2 * a1 + (n - 1) * d)

-- Define the conditions given in the problem.
def conditions (a1 d : ℚ) : Prop :=
  sum_arithmetic_seq a1 d 9 = 27 ∧ arithmetic_seq a1 d 10 = 8

-- Define the theorem to be proved.
theorem find_a99 (a1 d : ℚ)
  (h : conditions a1 d) : arithmetic_seq a1 d 99 = 218 / 5 :=
by
  -- Introduce the conditions into the proof.
  cases h with h_sum h_a10,
  -- Simplify the given conditions
  
  -- Proving the resulting condition
  sorry

end find_a99_l30_30320


namespace eiffel_tower_vs_burj_khalifa_l30_30958

-- Define the heights of the structures
def height_eiffel_tower : ℕ := 324
def height_burj_khalifa : ℕ := 830

-- Define the statement to be proven
theorem eiffel_tower_vs_burj_khalifa :
  height_burj_khalifa - height_eiffel_tower = 506 :=
by
  sorry

end eiffel_tower_vs_burj_khalifa_l30_30958


namespace product_of_25_digit_numbers_l30_30858

-- Define the two 25-digit numbers
def X : ℕ := 3333333333333333333333333 -- 25 times digit 3
def Y : ℕ := 6666666666666666666666666 -- 25 times digit 6

-- Define the expected product result as a string since it has a specific digit pattern
def expected_result := "222222222222222222222222177777777777777777777778"

-- Formalize the proof goal
theorem product_of_25_digit_numbers :
  -- Convert expected_result to natural number format, assuming base 10 calculations
  let expected_num := BigInt.ofString ("222222222222222222222222177777777777777777777778") in
  X * Y = expected_num :=
by
  sorry

end product_of_25_digit_numbers_l30_30858


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30564

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30564


namespace mango_ratio_correct_l30_30845

variable (kg_to_mangoes : Int → Int)
variable (total_kg_mangoes : Int := 60)
variable (sold_to_market_kg : Int := 20)
variable (mangoes_left : Int := 160)
variable (total_mangoes, sold_to_market, sold_to_community : Int)
variable (ratio : Int × Int := (1, 3))

def calculate_total_mangoes (kg : Int) : Int :=
  kg_to_mangoes kg

def calculate_sold_to_market_mangoes (kg : Int) : Int :=
  kg_to_mangoes kg

def calculate_sold_to_community_mangoes (total : Int) (sold : Int) (left : Int) : Int :=
  total - sold - left

theorem mango_ratio_correct :
  let total_mangoes := calculate_total_mangoes total_kg_mangoes in
  let sold_to_market := calculate_sold_to_market_mangoes sold_to_market_kg in
  let sold_to_community := calculate_sold_to_community_mangoes total_mangoes sold_to_market mangoes_left in
  8 = kg_to_mangoes 1 →
  ratio = (sold_to_community / total_mangoes, 3) :=
by
  sorry

end mango_ratio_correct_l30_30845


namespace smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30206

def is_palindrome_in_base (n : ℕ) (b : ℕ) : Prop :=
  let digits : List ℕ := (nat.to_digits b n)
  digits = digits.reverse

theorem smallest_palindrome_in_bases_2_and_4 :
  ∀ n : ℕ, 10 < n ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 → n ≥ 17 :=
begin
  intros n hn,
  have := (by norm_num : 16 < 17),
  cases hn with hn1 hn2,
  cases hn2 with hn3 hn4,
  linarith,
end

theorem smallest_palindrome_in_bases_2_and_4_17 :
  is_palindrome_in_base 17 2 ∧ is_palindrome_in_base 17 4 :=
begin
  split;
  { unfold is_palindrome_in_base, norm_num, refl },
end

end smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30206


namespace ellie_shoes_count_l30_30871

variable (E R : ℕ)

def ellie_shoes (E R : ℕ) : Prop :=
  E + R = 13 ∧ E = R + 3

theorem ellie_shoes_count (E R : ℕ) (h : ellie_shoes E R) : E = 8 :=
  by sorry

end ellie_shoes_count_l30_30871


namespace min_height_to_cover_snaps_l30_30848

-- Conditions:
axiom cube : Type
axiom is_stackable : cube → cube → Prop
axiom has_snap : cube → Prop
axiom has_receptacle : cube → Prop

-- Each cube has six sides: one protruding snap (top) and five receptacle holes.
-- When cubes are stacked, each cube covers the snap of the cube directly beneath it.

-- Define what it means for a structure to cover all snaps:
def covers_all_snaps (structure : list cube) : Prop :=
  ∀ (n : ℕ), n < structure.length - 1 → is_stackable (structure.nth_le n sorry) (structure.nth_le (n + 1) sorry) ∧ 
            has_snap (structure.nth_le n sorry) →
            ¬has_snap (structure.nth_le (n + 1) sorry)

-- Proof statement:
theorem min_height_to_cover_snaps (structure : list cube) : covers_all_snaps structure → structure.length ≥ 4 :=
sorry

end min_height_to_cover_snaps_l30_30848


namespace prime_factors_of_B_l30_30526

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30526


namespace num_zeros_of_f_l30_30020
noncomputable def f : ℝ → ℝ := 
λ x, if x ≤ 0 then x^2 + 2 * x - 3 else -2 + real.log x

theorem num_zeros_of_f : 
  {x : ℝ | f x = 0}.finite.to_finset.card = 2 :=
by sorry

end num_zeros_of_f_l30_30020


namespace emmy_rosa_ipods_l30_30237

theorem emmy_rosa_ipods :
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  Emmy_left + Rosa_ipods = 12 :=
by
  let Emmy_initial := 14
  let Emmy_lost := 6
  let Emmy_left := Emmy_initial - Emmy_lost
  let Rosa_ipods := Emmy_left / 2
  sorry

end emmy_rosa_ipods_l30_30237


namespace rectangle_diagonal_l30_30665

theorem rectangle_diagonal (k : ℚ)
  (h1 : 2 * (5 * k + 2 * k) = 72)
  (h2 : k = 36 / 7) :
  let l := 5 * k,
      w := 2 * k,
      d := Real.sqrt ((l ^ 2) + (w ^ 2)) in
  d = 194 / 7 := by
  sorry

end rectangle_diagonal_l30_30665


namespace smallest_n_terminating_decimal_l30_30109

theorem smallest_n_terminating_decimal : 
  ∃ (n : ℕ), (n > 0) ∧ (∀ p, prime p → p ∣ (n + 103) → (p = 2 ∨ p = 5)) ∧ (n = 22) := 
by
  sorry

end smallest_n_terminating_decimal_l30_30109


namespace distinct_prime_factors_of_product_of_divisors_l30_30507

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30507


namespace alex_play_friends_with_l30_30738

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30738


namespace common_difference_is_half_l30_30380

variable (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ) (q p : ℕ)

-- Conditions
def condition1 : Prop := a p = 4
def condition2 : Prop := a q = 2
def condition3 : Prop := p = 4 + q
def arithmetic_sequence : Prop := ∀ n : ℕ, a n = a₁ + (n - 1) * d

-- Proof statement
theorem common_difference_is_half 
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q)
  (as : arithmetic_sequence a a₁ d)
  : d = 1 / 2 := 
sorry

end common_difference_is_half_l30_30380


namespace graph_translation_l30_30986

variable (f : ℝ → ℝ)

theorem graph_translation (h : f 1 = 3) : f (-1) + 1 = 4 :=
sorry

end graph_translation_l30_30986


namespace parabola_intersection_square_l30_30988

theorem parabola_intersection_square (p : ℝ) :
   (∃ (x : ℝ), (x = 1 ∨ x = 2) ∧ x^2 * p = 1 ∨ x^2 * p = 2)
   → (1 / 4 ≤ p ∧ p ≤ 2) :=
by
  sorry

end parabola_intersection_square_l30_30988


namespace max_min_difference_of_f_l30_30631

noncomputable def f (x : Real) : Real := Real.abs sin x + (sin (2 * x))^4 + Real.abs cos x

theorem max_min_difference_of_f : 
  (Real.sqrt 2 - 1) = (Real.sqrt 2 - 1) :=
sorry

end max_min_difference_of_f_l30_30631


namespace expected_number_of_ones_on_three_dice_l30_30076

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30076


namespace min_abs_diff_is_11_l30_30312

noncomputable def min_abs_diff (k l : ℕ) : ℤ := abs (36^k - 5^l)

theorem min_abs_diff_is_11 :
  ∃ k l : ℕ, min_abs_diff k l = 11 :=
by
  sorry

end min_abs_diff_is_11_l30_30312


namespace expected_value_of_ones_on_three_dice_l30_30049

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30049


namespace average_weekly_labor_time_properties_l30_30961

-- Define the frequencies and group ranges
def group_frequencies : List ℕ := [10, 20, 12, 8]
def group_ranges : List (Set ℚ) := 
  [setOf (λ x, 0 ≤ x ∧ x < 1), 
   setOf (λ x, 1 ≤ x ∧ x < 2),
   setOf (λ x, 2 ≤ x ∧ x < 3),
   setOf (λ x, 3 ≤ x)]

-- Define the total number of students for the estimation problem
def total_students : ℕ := 500

theorem average_weekly_labor_time_properties :
  let sample_size := group_frequencies.sum
  let median_group := group_frequencies.ilast (0, 30)
  let mode_group := group_frequencies.ilast (0, 20)
  let estimated_students_in_fourth_group := total_students * (8 / 50)
  sample_size = 50 ∧
  median_group = (group_frequencies.nth 1).getD 0 ∧
  mode_group = (group_frequencies.nth 1).getD 0 ∧
  estimated_students_in_fourth_group = 80 :=
by
  sorry

end average_weekly_labor_time_properties_l30_30961


namespace coordinates_of_T_l30_30251

def is_square (O P Q R : (ℝ × ℝ)) : Prop :=
  O = (0, 0) ∧ Q = (3, 3) ∧ dist O Q = dist O P ∧ dist O Q = dist O R ∧
  (dist O P) + (dist O R) = dist O Q -- This checks if O, P, Q, R form the vertices of a square

theorem coordinates_of_T (T : ℝ × ℝ) (O P Q R : ℝ × ℝ) 
  (h_square : is_square O P Q R) 
  (h_area_tri_criteria : ∃ T, (dist P Q) * T.2 / 2 = 4.5) : 
  T = (3, 3) :=
sorry

end coordinates_of_T_l30_30251


namespace percentage_with_repeated_digits_l30_30358

-- Define the total number of five-digit numbers with the first digit non-zero.
def total_numbers : ℕ := 90000

-- Define the number of five-digit numbers without repeated digits.
def non_repeated_numbers : ℕ := 27216

-- Define the percentage of five-digit numbers with at least one repeated digit.
def x : ℝ := (total_numbers - non_repeated_numbers).toReal / total_numbers.toReal * 100

-- State the theorem that x is approximately 69.8 to the nearest tenth.
theorem percentage_with_repeated_digits : abs (x - 69.8) < 0.1 :=
by sorry

end percentage_with_repeated_digits_l30_30358


namespace expected_number_of_ones_when_three_dice_rolled_l30_30038

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30038


namespace hide_and_seek_l30_30786

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30786


namespace least_number_to_add_l30_30731

theorem least_number_to_add (n : ℕ) (m : ℕ) : (1156 + 19) % 25 = 0 :=
by
  sorry

end least_number_to_add_l30_30731


namespace drug_price_reduction_l30_30641

theorem drug_price_reduction (x : ℝ) :
    36 * (1 - x)^2 = 25 :=
sorry

end drug_price_reduction_l30_30641


namespace iterative_average_difference_is_4_25_l30_30295

-- Given a sequence of numbers 2, 4, 6, 8, and 10
def sequence : List ℕ := [2, 4, 6, 8, 10]

-- Each number is incremented by 1
def modified_sequence : List ℕ :=
  List.map (λ x => x + 1) sequence

-- Function to compute the iterative average of a list of numbers
def iterative_average : List ℝ → ℝ
| [] => 0
| [a] => a
| a :: b :: rest => 
    let mean_ab := (a + b) / 2
    iterative_average (mean_ab :: rest)

-- Let's find the minimum and maximum iterative averages 
-- by testing all permutations of the modified sequence
noncomputable def min_max_difference (l : List ℕ) : ℝ :=
  let perms := List.permutations l
  let iter_avgs := List.map (iterative_average ∘ List.map (λ x => (x : ℝ))) perms
  let min_avg := List.minimum iter_avgs
  let max_avg := List.maximum iter_avgs
  max_avg - min_avg

-- Prove that the difference between the largest and smallest possible values is 4.25
theorem iterative_average_difference_is_4_25 :
  min_max_difference modified_sequence = 4.25 :=
by
  sorry

end iterative_average_difference_is_4_25_l30_30295


namespace distinct_prime_factors_of_product_of_divisors_l30_30510

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30510


namespace mass_percentage_H_in_H2O_l30_30256

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def num_H_atoms_in_H2O : ℕ := 2
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def num_O_atoms_in_H2O : ℕ := 1

theorem mass_percentage_H_in_H2O :
  let mass_H_in_H2O := num_H_atoms_in_H2O * molar_mass_H,
      mass_H2O := mass_H_in_H2O + num_O_atoms_in_H2O * molar_mass_O,
      percentage_H := (mass_H_in_H2O / mass_H2O) * 100 in
  percentage_H = 11.21 :=
by
  sorry

end mass_percentage_H_in_H2O_l30_30256


namespace third_number_correct_l30_30113

-- Given that the row of Pascal's triangle with 51 numbers corresponds to the binomial coefficients of 50.
def third_number_in_51_pascal_row : ℕ := Nat.choose 50 2

-- Prove that the third number in this row of Pascal's triangle is 1225.
theorem third_number_correct : third_number_in_51_pascal_row = 1225 := 
by 
  -- Calculation part can be filled in for the full proof.
  sorry

end third_number_correct_l30_30113


namespace different_kinds_of_hamburgers_l30_30960

theorem different_kinds_of_hamburgers 
  (n_condiments : ℕ) 
  (condiment_choices : ℕ)
  (meat_patty_choices : ℕ)
  (h1 : n_condiments = 8)
  (h2 : condiment_choices = 2 ^ n_condiments)
  (h3 : meat_patty_choices = 3)
  : condiment_choices * meat_patty_choices = 768 := 
by
  sorry

end different_kinds_of_hamburgers_l30_30960


namespace two_digit_multiples_of_4_and_9_l30_30353

theorem two_digit_multiples_of_4_and_9 :
  {x : ℕ | 10 ≤ x ∧ x ≤ 99 ∧ x % 36 = 0}.card = 2 :=
by
  sorry

end two_digit_multiples_of_4_and_9_l30_30353


namespace distinct_prime_factors_product_divisors_60_l30_30487

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30487


namespace distinct_prime_factors_product_divisors_60_l30_30486

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30486


namespace num_team_formations_l30_30671

def num_boys : ℕ := 4
def num_girls : ℕ := 4
def total_team_members : ℕ := 4
def cannot_be_first_debater (boy : ℕ) (boy_A : ℕ) : Prop := boy ≠ boy_A
def cannot_be_fourth_debater (girl : ℕ) (girl_B : ℕ) : Prop := girl ≠ girl_B
def boy_A_selected_requires_girl_B (selected_boys : list ℕ) (selected_girls : list ℕ) (boy_A girl_B : ℕ) : Prop := boy_A ∈ selected_boys → girl_B ∈ selected_girls

theorem num_team_formations 
  (boy_A girl_B : ℕ)
  (hb : boy_A < num_boys)
  (hg : girl_B < num_girls)
  (team : list ℕ × list ℕ)
  (h_team_size : team.1.length + team.2.length = total_team_members)
  (h1 : cannot_be_first_debater team.1.head boy_A)
  (h4 : cannot_be_fourth_debater team.2.last girl_B)
  (hAB : boy_A_selected_requires_girl_B team.1 team.2 boy_A girl_B) : 
  (count_all_formations : ℕ) := 930
sorry

end num_team_formations_l30_30671


namespace vertical_asymptote_at_5_l30_30233

noncomputable def function_with_asymptote (x : ℝ) : ℝ := (x^2 + 3*x + 9) / (x - 5)

theorem vertical_asymptote_at_5 : ∃ x : ℝ, (x = 5) ∧ asymptote (function_with_asymptote x) :=
begin
  use 5,
  split,
  { refl },
  { sorry } -- Proof of asymptote goes here
end

end vertical_asymptote_at_5_l30_30233


namespace find_two_digit_numbers_l30_30252

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits_sum (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

def satisfies_condition (n : ℕ) : Prop :=
  is_prime (n - 7 * digits_sum n)

theorem find_two_digit_numbers :
  { n : ℕ | is_two_digit_number n ∧ satisfies_condition n } = {10, 31, 52, 73, 94} :=
  sorry

end find_two_digit_numbers_l30_30252


namespace rectangle_diagonal_length_l30_30657

theorem rectangle_diagonal_length (L W : ℝ) (h1 : 2 * (L + W) = 72) (h2 : L / W = 5 / 2) : 
    (sqrt (L^2 + W^2)) = 194 / 7 := 
by 
  sorry

end rectangle_diagonal_length_l30_30657


namespace value_of_expression_l30_30356

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * b / (c * d) = 180 :=
by
  sorry

end value_of_expression_l30_30356


namespace hide_and_seek_l30_30780

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30780


namespace rectangle_diagonal_length_l30_30645

theorem rectangle_diagonal_length (P : ℝ) (r : ℝ) (perimeter_eq : P = 72) (ratio_eq : r = 5/2) :
  let k := P / 14
  let l := 5 * k
  let w := 2 * k
  let d := Real.sqrt (l^2 + w^2)
  d = 194 / 7 :=
sorry

end rectangle_diagonal_length_l30_30645


namespace problem_solution_l30_30590

theorem problem_solution 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h₁ : a = ⌊2 + Real.sqrt 2⌋) 
  (h₂ : b = (2 + Real.sqrt 2) - ⌊2 + Real.sqrt 2⌋)
  (h₃ : c = ⌊4 - Real.sqrt 2⌋)
  (h₄ : d = (4 - Real.sqrt 2) - ⌊4 - Real.sqrt 2⌋) :
  (b + d) / (a * c) = 1 / 6 :=
by
  sorry

end problem_solution_l30_30590


namespace count_irrational_numbers_l30_30182

theorem count_irrational_numbers :
  let numbers := [3.14159, real.cbrt 9 * -1, (7 + 1 / 7) / 1000 - 1 / 10, - real.pi, real.sqrt 25, real.cbrt 64, - (1 / 7)]
  let irrational_numbers := numbers.filter (λ x, ¬ (∃ (a b : ℚ), x = (a : ℝ) / b))
  irrational_numbers.length = 3 :=
by
  let numbers := 
  (3.14159 : ℝ) ::
  (- real.cbrt 9 : ℝ) ::
  (7 + 1 / 7) / 1000 - 1 / 10 ::
  (- real.pi : ℝ) ::
  (real.sqrt 25 : ℝ) ::
  (real.cbrt 64 : ℝ) ::
  (- (1 / 7) : ℝ) ::
  []
  let irrational_numbers := numbers.filter (λ x, ¬ ∃ a b : ℚ, x = (a : ℝ) / b)
  show irrational_numbers.length = 3, from sorry

end count_irrational_numbers_l30_30182


namespace number_of_distinct_prime_factors_of_B_l30_30430

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30430


namespace even_distance_between_trees_with_signs_l30_30187

theorem even_distance_between_trees_with_signs
  (i j k : ℕ)
  (hi : i ≠ j)
  (hj : j ≠ k)
  (hk : k ≠ i) :
  ∃ (m n : ℕ), m ≠ n ∧ (m = 3 * i ∨ m = 3 * j ∨ m = 3 * k) ∧ (n = 3 * i ∨ n = 3 * j ∨ n = 3 * k) ∧ even (n - m) :=
sorry

end even_distance_between_trees_with_signs_l30_30187


namespace rectangle_diagonal_length_l30_30660

theorem rectangle_diagonal_length (L W : ℝ) (h1 : 2 * (L + W) = 72) (h2 : L / W = 5 / 2) : 
    (sqrt (L^2 + W^2)) = 194 / 7 := 
by 
  sorry

end rectangle_diagonal_length_l30_30660


namespace minimum_transportation_cost_l30_30170

theorem minimum_transportation_cost :
  ∀ (x : ℕ), 
    (17 - x) + (x - 3) = 12 → 
    (18 - x) + (17 - x) = 14 → 
    (200 * x + 19300 = 19900) → 
    (x = 3) 
:= by sorry

end minimum_transportation_cost_l30_30170


namespace find_L_l30_30801

open Real

noncomputable def surface_area_of_cube (a : ℝ) : ℝ :=
  6 * a^2

noncomputable def radius_of_sphere (A : ℝ) : ℝ :=
  sqrt (A / (4 * π))

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * π * r^3

theorem find_L (L : ℝ) :
  let a := 3
  let A := surface_area_of_cube a
  let r := radius_of_sphere A
  let V := volume_of_sphere r
  V = L * sqrt(15) / sqrt(π) -> L = 48 :=
by
  sorry

end find_L_l30_30801


namespace problem_l30_30407

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30407


namespace count_irrational_numbers_l30_30177

def is_irrational (x : Real) : Prop := ¬ ∃ (a b : Int), b ≠ 0 ∧ x = a / b

-- Given list of numbers
def nums : List Real := [3.14159, -Real.cbrt 9, 0.131131113, -Real.pi, Real.sqrt 25, Real.cbrt 64, -1/7]

-- List of irrational numbers from provided list
def irrational_nums : List Real := [nums.nthLe 1 (by norm_num), nums.nthLe 2 (by norm_num), nums.nthLe 3 (by norm_num)]

theorem count_irrational_numbers : irrational_nums.length = 3 := 
by 
  -- Justification of each irrational number
  have h1 : is_irrational (nums.nthLe 1 (by norm_num)) := sorry,
  have h2 : is_irrational (nums.nthLe 2 (by norm_num)) := sorry,
  have h3 : is_irrational (nums.nthLe 3 (by norm_num)) := sorry,
  sorry

end count_irrational_numbers_l30_30177


namespace find_a_plus_b_l30_30580

noncomputable def f (a b : ℝ) (x : ℝ) := a * x + b

noncomputable def h (x : ℝ) := 3 * x + 2

theorem find_a_plus_b (a b : ℝ) (x : ℝ) (h_condition : ∀ x, h (f a b x) = 4 * x - 1) :
  a + b = 1 / 3 := 
by
  sorry

end find_a_plus_b_l30_30580


namespace find_k_l30_30346

def vector := (ℝ × ℝ)

noncomputable def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude_squared (v : vector) : ℝ :=
  dot_product v v

noncomputable def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

noncomputable def vector_sub (v1 v2 : vector) : vector :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_k (k : ℝ) :
  let a : vector := (4, 2)
  let b : vector := (2 - k, k - 1) in
  magnitude_squared (vector_add a b) = magnitude_squared (vector_sub a b) → k = 3 :=
begin
  intros a b,
  sorry -- Proof is skipped 
end

end find_k_l30_30346


namespace expected_ones_on_three_dice_l30_30092

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30092


namespace smallest_palindromic_integer_l30_30212

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n
  digits = List.reverse digits

theorem smallest_palindromic_integer :
  ∃ (n : ℕ), n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ (∀ m, m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 → n ≤ m) :=
begin
  sorry
end

end smallest_palindromic_integer_l30_30212


namespace _l30_30369

noncomputable def probability_event_b_given_a : ℕ → ℕ → ℕ → ℕ × ℕ → ℚ
| zeros, ones, twos, (1, drawn_label) =>
  if drawn_label = 1 then
    (ones * (ones - 1)) / (zeros + ones + twos).choose 2
  else 0
| _, _, _, _ => 0

lemma probability_theorem :
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  (1 - 1) * (ones - 1)/(total.choose 2) = 1/7 :=
by
  let zeros := 1
  let ones := 2
  let twos := 2
  let total := zeros + ones + twos
  let draw_label := 1
  let event_b_given_a := probability_event_b_given_a zeros ones twos (1, draw_label)
  have pos_cases : (ones * (ones - 1))/(total.choose 2) = 1 / 7 := by sorry
  exact pos_cases

end _l30_30369


namespace distinct_prime_factors_of_product_of_divisors_l30_30518

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30518


namespace least_number_to_add_l30_30704

theorem least_number_to_add (x : ℕ) : (1053 + x) % 23 = 0 ↔ x = 5 := by
  sorry

end least_number_to_add_l30_30704


namespace circle_with_diameter_l30_30343

def midpoint (P1 P2 : ℝ × ℝ) := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

theorem circle_with_diameter (P1 : ℝ × ℝ) (P2 : ℝ × ℝ) :
  P1 = (4, 9) → P2 = (6, 3) → 
  let C := midpoint P1 P2 in
  let r := real.sqrt (((C.1 - P1.1) ^ 2) + ((C.2 - P1.2) ^ 2)) in
  ∀ x y : ℝ, (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2 ↔ (x - 5) ^ 2 + (y - 6) ^ 2 = 10 :=
by sorry

end circle_with_diameter_l30_30343


namespace correct_statements_l30_30852

theorem correct_statements : {1, 2} = { n | 
  n = 1 ∧ ∀ x : ℝ, ¬(x^2 - x > 0) ↔ x^2 - x ≤ 0 ∨
  n = 2 ∧ ∀ p q : Prop, p ∧ q → p ∨ q ∧ ¬(p ∨ q → p ∧ q) ∨
  n = 3 ∧ (∀ a b m : ℝ, a * m^2 < b * m^2 → a < b ↔ a < b → a * m^2 < b * m^2) ∨
  n = 4 ∧ ∀ (A B C D : Set), (A ∪ B = A ∧ C ∩ D = C → A ⊆ B ∧ C ⊆ D) } :=
by
  sorry

end correct_statements_l30_30852


namespace alex_play_friends_with_l30_30736

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30736


namespace remainder_1493827_division_l30_30101

theorem remainder_1493827_division :
  let n := 1493827 in
  n % 4 = 3 ∧ n % 3 = 1 :=
by
  let n := 1493827
  split
  { -- Prove that 1493827 % 4 = 3
    have : n % 4 = 27 % 4,
    { sorry },
    show n % 4 = 3,
    { exact this },
  }
  { -- Prove that 1493827 % 3 = 1
    have : (1 + 4 + 9 + 3 + 8 + 2 + 7) = 34,
    { sorry },
    show n % 3 = 1,
    { sorry },
  }

end remainder_1493827_division_l30_30101


namespace primes_have_property_P_infinitely_many_composites_have_property_P_l30_30270

def has_property_P (n : ℕ) : Prop :=
∀ a : ℕ, a > 0 → n ∣ a^n - 1 → n^2 ∣ a^n - 1

theorem primes_have_property_P :
∀ p : ℕ, p.prime → has_property_P p := sorry

theorem infinitely_many_composites_have_property_P : 
∃ᶠ n in at_top, ∃ m : ℕ, n = 2 * m ∧ has_property_P n := sorry

end primes_have_property_P_infinitely_many_composites_have_property_P_l30_30270


namespace exist_yz_in_S_l30_30586

open_locale classical

variables (Q : Type) [linear_ordered_field Q] (S : set Q)
variables (h1 : ¬ (0 : Q) ∈ S)
variables (h2 : ∀ s1 ∈ S, ∀ s2 ∈ S, s2 ≠ 0 → (s1 / s2) ∈ S)
variables (h3 : ∃ q ∈ Q, q ≠ 0 ∧ q ∉ S ∧ (∀ r ∈ Q, r ≠ 0 ∧ r ∉ S → ∃ s ∈ S, r = q * s))

theorem exist_yz_in_S (x : Q) (hx : x ∈ S) : ∃ y z ∈ S, x = y + z :=
sorry

end exist_yz_in_S_l30_30586


namespace expected_ones_in_three_dice_rolls_l30_30040

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30040


namespace chemical_compound_molar_mass_l30_30700

-- Define the given conditions
def mass : ℝ := 168
def moles : ℝ := 3

-- Define the formula for molar mass given mass and moles
def molar_mass (mass : ℝ) (moles : ℝ) : ℝ := mass / moles

-- The theorem that states the molar mass given the conditions
theorem chemical_compound_molar_mass : 
  molar_mass mass moles = 56 :=
by
  -- the proof is replaced with a placeholder
  sorry

end chemical_compound_molar_mass_l30_30700


namespace common_chord_length_eq_sqrt_11_l30_30007

theorem common_chord_length_eq_sqrt_11 :
  let c1 := (0, 0)
  let r1 := 2
  let c2 := (-1, 2)
  let r2 := sqrt 3
  let d := abs ((2 * c1.1 + (-4) * c1.2 + 5) / sqrt (2^2 + (-4)^2))
  chord_length_eq c1 r1 c2 r2 d = sqrt 11 := sorry

end common_chord_length_eq_sqrt_11_l30_30007


namespace distinct_prime_factors_product_divisors_60_l30_30490

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30490


namespace distinct_prime_factors_of_product_of_divisors_l30_30513

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30513


namespace hide_and_seek_friends_l30_30762

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30762


namespace distinct_prime_factors_of_B_l30_30450

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30450


namespace range_of_a_l30_30279

open Set Real

noncomputable def f (x a : ℝ) := x ^ 2 + 2 * x + a

theorem range_of_a (a : ℝ) :
  (∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≥ 0) → a ≥ -8 :=
by
  intro h
  sorry

end range_of_a_l30_30279


namespace expected_number_of_ones_l30_30066

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30066


namespace distinct_prime_factors_product_divisors_60_l30_30485

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30485


namespace alpha_values_for_odd_function_l30_30281

theorem alpha_values_for_odd_function {α : ℝ} :
  α ∈ {-1, 1, 2, 3/5, 7/2} →
  ((∀ x : ℝ, x ^ α ≠ 0) ∧ (∀ x : ℝ, x < 0 → x ^ α < 0) ∧ (∀ x : ℝ, (x ^ α = -( (-x) ^ α)))) ↔ α ∈ {1, 3/5} := 
sorry

end alpha_values_for_odd_function_l30_30281


namespace problem_l30_30276

noncomputable def number_of_regions_four_planes (h1 : True) (h2 : True) : ℕ := 14

theorem problem (h1 : True) (h2 : True) : number_of_regions_four_planes h1 h2 = 14 :=
by sorry

end problem_l30_30276


namespace expected_number_of_ones_when_three_dice_rolled_l30_30032

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30032


namespace verify_arithmetic_sequence_l30_30926

noncomputable def arithmetic_sequence_proof : Prop :=
  ∃ (a : ℕ → ℚ) (d : ℚ),
    (∀ n, a (n + 1) > a n) ∧ 
    a 3 = 5 / 2 ∧ 
    a 2 * a 4 = 6 ∧ 
    a 1 = 3 / 2 ∧ 
    d = 1 / 2 ∧ 
    (∀ n, a n = 1 / 2 * (n + 2)) ∧ 
    (∀ n, ∑ i in finset.range n, a i = 1 / 4 * n * n + 5 / 4 * n)

theorem verify_arithmetic_sequence :
  arithmetic_sequence_proof :=
sorry

end verify_arithmetic_sequence_l30_30926


namespace closest_to_fraction_is_2000_l30_30791

-- Define the original fractions and their approximations
def numerator : ℝ := 410
def denominator : ℝ := 0.21
def approximated_numerator : ℝ := 400
def approximated_denominator : ℝ := 0.2

-- Define the options to choose from
def options : List ℝ := [100, 500, 1900, 2000, 2500]

-- Statement to prove that the closest value to numerator / denominator is 2000
theorem closest_to_fraction_is_2000 : 
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 100) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 500) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 1900) ∧
  abs ((numerator / denominator) - 2000) < abs ((numerator / denominator) - 2500) :=
sorry

end closest_to_fraction_is_2000_l30_30791


namespace decrease_angle_equilateral_l30_30690

theorem decrease_angle_equilateral (D E F : ℝ) (h : D = 60) (h_equilateral : D = E ∧ E = F) (h_decrease : D' = D - 20) :
  ∃ max_angle : ℝ, max_angle = 70 :=
by
  sorry

end decrease_angle_equilateral_l30_30690


namespace prime_factors_of_B_l30_30525

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30525


namespace alex_play_friends_with_l30_30737

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30737


namespace first_shaded_square_for_all_columns_l30_30162

-- Let T be the function that generates the n-th triangular number
def T (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the goal: prove that, for n = 32, T(n) is 528
theorem first_shaded_square_for_all_columns :
  T 32 = 528 := by
  sorry

end first_shaded_square_for_all_columns_l30_30162


namespace smallest_palindrome_in_base2_and_4_l30_30222

-- Define a function to check if a number's representation is a palindrome.
def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

-- Convert a number n to a given base and represent as a list of digits.
def to_base (n : ℕ) (b : ℕ) : List ℕ :=
  if h : b > 1 then
    let rec convert (n : ℕ) : List ℕ :=
      if n = 0 then [] else (n % b) :: convert (n / b)
    convert n 
  else []

-- Convert the list of digits to a list of characters.
def digits_to_chars (ds : List ℕ) : List Char :=
  ds.map (λ d => (Char.ofNat (d + 48))) -- Adjust ASCII value for digit representation

-- Define a function to check if a number is a palindrome in a specified base.
def is_palindromic_in_base (n base : ℕ) : Bool :=
  is_palindrome (digits_to_chars (to_base n base))

-- Define the main claim.
theorem smallest_palindrome_in_base2_and_4 : ∃ n, n > 10 ∧ is_palindromic_in_base n 2 ∧ is_palindromic_in_base n 4 ∧ ∀ m, m > 10 ∧ is_palindromic_in_base m 2 ∧ is_palindromic_in_base m 4 -> n ≤ m :=
by
  exists 15
  sorry

end smallest_palindrome_in_base2_and_4_l30_30222


namespace persons_attended_total_l30_30827

theorem persons_attended_total (p q : ℕ) (a : ℕ) (c : ℕ) (total_amount : ℕ) (adult_ticket : ℕ) (child_ticket : ℕ) 
  (h1 : adult_ticket = 60) (h2 : child_ticket = 25) (h3 : total_amount = 14000) 
  (h4 : a = 200) (h5 : p = a + c)
  (h6 : a * adult_ticket + c * child_ticket = total_amount):
  p = 280 :=
by
  sorry

end persons_attended_total_l30_30827


namespace same_monotonicity_and_parity_l30_30828

-- Definitions of functions involved
def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := exp x - exp (-x)

-- Prove that g has the same monotonicity and parity as f
theorem same_monotonicity_and_parity :
  (∀ x : ℝ, g(-x) = -g(x)) ∧ (∀ x y : ℝ, x < y → g(x) < g(y)) :=
by
  sorry

end same_monotonicity_and_parity_l30_30828


namespace value_of_a4_plus_a_inv4_l30_30003

theorem value_of_a4_plus_a_inv4 (a : ℝ) (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 := by
  sorry

end value_of_a4_plus_a_inv4_l30_30003


namespace normal_distribution_probability_l30_30318

noncomputable def normalProb (ξ : ℝ → ℝ) (μ σ : ℝ) : Prop := 
  ∀ x, ξ x = (1 / (σ * sqrt (2 * π))) * exp (-(x - μ)^2 / (2 * σ^2))

theorem normal_distribution_probability (ξ : ℝ → ℝ) 
    (h₁ : normalProb ξ 2 1) 
    (h₂ : ∀ P, P (λ x, x > 3) = 0.1587) :
    ∀ P, P (λ x, x > 1) = 0.8413 := 
sorry

end normal_distribution_probability_l30_30318


namespace solve_system_of_congruences_l30_30619

theorem solve_system_of_congruences {x : ℤ} 
  (h1 : x ≡ 1 [MOD 7]) 
  (h2 : x ≡ 1 [MOD 8]) 
  (h3 : x ≡ 3 [MOD 9]) : 
  x ≡ 57 [MOD 504] := sorry

end solve_system_of_congruences_l30_30619


namespace number_of_distinct_prime_factors_of_B_l30_30438

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30438


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30502

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30502


namespace closest_fraction_medals_won_l30_30834

theorem closest_fraction_medals_won :
  let actual_fraction := (24 : ℚ) / 150
  let candidates := [1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9]
  let dist := λ x, abs (x - actual_fraction)
  (argmin dist candidates) = 1 / 6 :=
by
  sorry

end closest_fraction_medals_won_l30_30834


namespace number_of_distinct_prime_factors_of_B_l30_30434

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30434


namespace exactly_one_true_l30_30309

variables (l m : Line) (α β : Plane)

-- Propositions
def prop1 := (l ⊂ β) ∧ (α ⊥ β) → (l ⊥ α)
def prop2 := (l ⊥ β) ∧ (α ∥ β) → (l ⊥ α)
def prop3 := (l ⊥ β) ∧ (α ⊥ β) → (l ∥ α)
def prop4 := (α ∩ β = m) ∧ (l ∥ m) → (l ∥ α)

theorem exactly_one_true (l m : Line) (α β : Plane) :
  ¬ prop1 l α β ∧ prop2 l α β ∧ ¬ prop3 l α β ∧ ¬ prop4 l α β :=
sorry

end exactly_one_true_l30_30309


namespace at_most_one_positive_root_l30_30021

theorem at_most_one_positive_root 
  {n : ℕ} 
  (a : fin (n+1) → ℝ) 
  (h_pos : ∀ i, 1 ≤ i → i ≤ n → 0 < a i) : 
  ∃ (at_most_one_root : ℝ), 
  (∀ x > 0, x^n + a 0 * x^(n-1) - ∑ i in finset.range (n-1), a ↑(i + 1) * x^(n-i-2) = 0 → x = at_most_one_root) := 
sorry

end at_most_one_positive_root_l30_30021


namespace range_f_range_g_l30_30259

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem range_f : set.range f = {y : ℝ | -1 ≤ y} :=
by {sorry}

def g (x : ℝ) (h : 1 ≤ x ∧ x < 3) : ℝ := 1 / x

theorem range_g : set.range (λ (x : {x // 1 ≤ x ∧ x < 3}), g x x.property) = {y : ℝ | (1 / 3 : ℝ) < y ∧ y ≤ 1} :=
by {sorry}

end range_f_range_g_l30_30259


namespace sequence_properties_l30_30906

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = 2 * a n

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a (n+2) - a (n+1)) = (a (n+1) - a n)

theorem sequence_properties :
  (∃ (a : ℕ → ℝ), geometric_sequence a ∧ arithmetic_sequence (fun n => if n = 1 then a 1 else if n = 2 then a 1*2 else a 1 * 4)) →
  (∃ (a : ℕ → ℝ), ∀ n : ℕ, a n = 2^(n-1)) ∧
  (∃ (b : ℕ → ℝ), ∀ n : ℕ, b n = n * 2^(n-1) + n ∧ ∑ i in finset.range n, b i = (n-1) * 2^n + 1 + (n * (n + 1)) / 2) :=
begin
  sorry
end

end sequence_properties_l30_30906


namespace numberOfGirls_l30_30867

def totalStudents : ℕ := 1600
def sampleSize : ℕ := 200
def sampleDifference : ℕ := 20

theorem numberOfGirls (x : ℕ) (y : ℕ) (h1 : x + y = totalStudents) (h2 : sampleSize = 200) (h3 : y - x = sampleDifference) : x = 720 :=
  by
    have h4 : 110 * x = 90 * y, sorry
    have h5 : 110 * x = 90 * (1600 - x), sorry
    have h6 : 110 * x = 144000 - 90 * x, sorry
    have h7 : 200 * x = 144000, sorry
    have h8 : x = 720, sorry
    exact h8

end numberOfGirls_l30_30867


namespace x_coordinate_of_equidistant_point_l30_30102

theorem x_coordinate_of_equidistant_point (x : ℝ) : 
  ((-3 - x)^2 + (-2 - 0)^2) = ((2 - x)^2 + (-6 - 0)^2) → x = 2.7 :=
by
  sorry

end x_coordinate_of_equidistant_point_l30_30102


namespace max_types_of_painted_blocks_l30_30822

/-- A toy factory produces cubic building blocks of the same size, 
    with each of the six faces painted in one of three colors: red, yellow, and blue, 
    with each color appearing on two faces. If two blocks can be rotated 
    to match the position of each color on the faces, they are considered 
    the same type of block. This theorem proves that the maximum number 
    of different types of blocks that can be painted is 6. --/
theorem max_types_of_painted_blocks : 6 :=
by sorry

end max_types_of_painted_blocks_l30_30822


namespace angle_A_sum_b_c_l30_30910

theorem angle_A (A : ℝ): 
  (2 * sqrt 3 * sin A * cos A - 2 * (sin A)^2 + 2 = 2) -> 
  (A = π / 3) := sorry

theorem sum_b_c (a b c : ℝ) (C B : ℝ) :
  (a = 3) ->
  (sin C = 2 * sin B) -> 
  (b^2 + c^2 - b * c = 9) -> 
  (b + c = 3 * sqrt 3) := sorry

end angle_A_sum_b_c_l30_30910


namespace expression_value_l30_30718

theorem expression_value (a b : ℚ) (h₁ : a = -1/2) (h₂ : b = 3/2) : -a - 2 * b^2 + 3 * a * b = -25/4 :=
by
  sorry

end expression_value_l30_30718


namespace standard_equation_of_line_standard_equation_of_curve_C_find_value_of_a_l30_30870

noncomputable def parametric_line_equation (t : ℝ) : ℝ × ℝ :=
  (-1 + t * real.cos (real.pi / 4), -2 + t * real.sin (real.pi / 4))

theorem standard_equation_of_line :
  ∀ t : ℝ, (parametric_line_equation t).1 - (parametric_line_equation t).2 - 1 = 0 :=
sorry

theorem standard_equation_of_curve_C (a : ℝ) (h : 0 < a) :
  ∀ (ρ θ : ℝ), ρ * real.sin θ * real.tan θ = 2 * a ↔ (ρ * real.sin θ) ^ 2 = 2 * a * (ρ * real.cos θ) :=
sorry

theorem find_value_of_a (a : ℝ) (h : 0 < a) :
  (∃ t₁ t₂ : ℝ, t₁ > 0 ∧ t₂ > 0 ∧ |t₁ - t₂/2| = 0 ∧ (parametric_line_equation t₁).2 ^ 2 = 2 * a * (parametric_line_equation t₁).1 ∧
   (parametric_line_equation t₂).2 ^ 2 = 2 * a * (parametric_line_equation t₂).1) ↔ a = 1 / 4 :=
sorry

end standard_equation_of_line_standard_equation_of_curve_C_find_value_of_a_l30_30870


namespace eagles_win_26_games_l30_30028

-- Define the win counts for each team: Eagles, Lions, Sharks, Wolves, and Falcons
def win_counts := List ℕ

-- Conditions from the problem
def ascending_order_ms := [26, 30, 35, 40, 45]
def lions_more_than_eagles (wins : win_counts) := wins[1] > wins[0]
def sharks_more_than_wolves (wins : win_counts) := wins[2] > wins[3]
def sharks_less_than_falcons (wins : win_counts) := wins[2] < wins[4]
def wolves_more_than_25 (wins : win_counts) := wins[3] > 25

-- Proof statement
theorem eagles_win_26_games (wins : win_counts)
  (h_order : wins = ascending_order_ms)
  (h_lions : lions_more_than_eagles wins)
  (h_sharks_wolves : sharks_more_than_wolves wins)
  (h_sharks_falcons : sharks_less_than_falcons wins)
  (h_wolves : wolves_more_than_25 wins) :
  wins[0] = 26 :=
by
  sorry

end eagles_win_26_games_l30_30028


namespace prime_factors_of_B_l30_30521

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30521


namespace problem_l30_30404

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30404


namespace distinct_prime_factors_B_l30_30549

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30549


namespace b_2016_value_l30_30956

theorem b_2016_value : 
  ∃ (a b : ℕ → ℝ), 
    a 1 = 1 / 2 ∧ 
    (∀ n : ℕ, 0 < n → a n + b n = 1) ∧
    (∀ n : ℕ, 0 < n → b (n + 1) = b n / (1 - (a n)^2)) → 
    b 2016 = 2016 / 2017 :=
by
  sorry

end b_2016_value_l30_30956


namespace a4_plus_a_neg4_l30_30001

variable (a : ℝ)
-- Condition given in the problem
def condition : Prop := 5 = a + a⁻¹

-- Target statement to prove
theorem a4_plus_a_neg4 : condition a → a^4 + a^(-4) = 527 :=
by
  intro h
  sorry

end a4_plus_a_neg4_l30_30001


namespace problem_solution_l30_30914

open Real

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

def area_triangle_ABF1 (a c b : ℝ) : ℝ :=
  1/2 * (a - c) * b

def range_PQ_F1Q (k : ℝ) : Prop :=
  if k = 0 then 0 < 1 * 2 ∧ 1 * 2 ≤ 2
  else ∀ (t : ℝ), 1 < t → 0 < 2 * (1 + 3 * k^2) / ((1 + 2 * k^2) * (k^2 + 1)) ∧ 2 * (1 + 3 * k^2) / ((1 + 2 * k^2) * (k^2 + 1)) < 2

theorem problem_solution
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b = 1) 
  (h4 : a^2 = b^2 + c^2) (h5 : a - c = sqrt 2 - 1)
  (k : ℝ) :
  (ellipse_eq a b ∧ area_triangle_ABF1 a c b = (sqrt 2 - 1) / 2 ∧ range_PQ_F1Q k) := 
sorry

end problem_solution_l30_30914


namespace relation_y₁_y₂_y₃_l30_30923

def parabola (x : ℝ) : ℝ := - x^2 - 2 * x + 2
noncomputable def y₁ : ℝ := parabola (-2)
noncomputable def y₂ : ℝ := parabola (1)
noncomputable def y₃ : ℝ := parabola (2)

theorem relation_y₁_y₂_y₃ : y₁ > y₂ ∧ y₂ > y₃ := by
  have h₁ : y₁ = 2 := by
    unfold y₁ parabola
    norm_num
    
  have h₂ : y₂ = -1 := by
    unfold y₂ parabola
    norm_num
    
  have h₃ : y₃ = -6 := by
    unfold y₃ parabola
    norm_num
    
  rw [h₁, h₂, h₃]
  exact ⟨by norm_num, by norm_num⟩

end relation_y₁_y₂_y₃_l30_30923


namespace rectangle_diagonal_length_l30_30648

theorem rectangle_diagonal_length (P : ℝ) (r : ℝ) (perimeter_eq : P = 72) (ratio_eq : r = 5/2) :
  let k := P / 14
  let l := 5 * k
  let w := 2 * k
  let d := Real.sqrt (l^2 + w^2)
  d = 194 / 7 :=
sorry

end rectangle_diagonal_length_l30_30648


namespace number_of_distinct_prime_factors_of_B_l30_30542

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30542


namespace equivalent_relationship_l30_30306

theorem equivalent_relationship (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 3 ^ a = 4 ^ b ∧ 4 ^ b = 6 ^ c) : (1 / a) + (1 / (2 * b)) = (1 / c) :=
by
  sorry

end equivalent_relationship_l30_30306


namespace probabilityAcute_l30_30969
open Real Set

def isAcuteAngle (θ : ℝ) : Prop := θ > 0 ∧ θ < (π / 2)

def vectorAB (x y : ℤ) : Prop := x ∈ {-2, -1, 0, 1, 2} ∧ y ∈ {-2, -1, 0, 1, 2}

def vectorA : (ℤ × ℤ) := (1, -1)

noncomputable def probabilityAcuteAngle (x y : ℤ) : ℚ :=
  if x ∈ {-2, -1, 0, 1, 2} ∧ y ∈ {-2, -1, 0, 1, 2} ∧ (x - y > 0 ∧ x + y ≠ 0)
  then 1 / 25 else 0

theorem probabilityAcute (p : ℚ) : p = 8 / 25 :=
by
  let total_count := 25
  let favorable_count := 8
  have h : ∑ x in finset.product (finset.range 5) (finset.range 5), probabilityAcuteAngle x.1 x.2 = favorable_count / total_count := sorry
  exact h

end probabilityAcute_l30_30969


namespace required_force_18_inch_wrench_l30_30633

def inverse_force (l : ℕ) (k : ℕ) : ℕ := k / l

def extra_force : ℕ := 50

def initial_force : ℕ := 300

noncomputable
def handle_length_1 : ℕ := 12

noncomputable
def handle_length_2 : ℕ := 18

noncomputable
def adjusted_force : ℕ := inverse_force handle_length_2 (initial_force * handle_length_1)

theorem required_force_18_inch_wrench : 
  adjusted_force + extra_force = 250 := 
by
  sorry

end required_force_18_inch_wrench_l30_30633


namespace area_N1N2N3_one_sixteenth_area_ABC_l30_30129

variables (A B C D E F N1 N2 N3 : Type) [add_comm_group A] [module ℝ A]
variables [affine_space ℝ A] (A B C D E F N1 N2 N3 : A)
variables (K : ℝ)

-- Given conditions
noncomputable def CD_one_fourth_BC (hBC : B ≠ C) : collinear ℝ ({B, C, D}) ∧ (D = B + 1/4 • (C - B)) := sorry
noncomputable def AE_three_fourths_AC (hAC : A ≠ C) : collinear ℝ ({A, C, E}) ∧ (E = A + 3/4 • (C - A)) := sorry
noncomputable def BF_three_fourths_AB (hAB : A ≠ B) : collinear ℝ ({A, B, F}) ∧ (F = A + 3/4 • (B - A)) := sorry

-- Intersections
noncomputable def N1_intersection (hAD : affine_combo ℝ {A, D}) (hCF : affine_combo ℝ {C, F}) : affine_independent ℝ ({A, D, C, F}) ∧ N1 = line_inter ℝ ({A, D}) ({C, F}) := sorry
noncomputable def N2_intersection (hBE : affine_combo ℝ {B, E}) (hAD : affine_combo ℝ {A, D}) : affine_independent ℝ ({B, E, A, D}) ∧ N2 = line_inter ℝ ({B, E}) ({A, D}) := sorry
noncomputable def N3_intersection (hBE : affine_combo ℝ {B, E}) (hCF : affine_combo ℝ {C, F}) : affine_independent ℝ ({B, E, C, F}) ∧ N3 = line_inter ℝ ({B, E}) ({C, F}) := sorry

-- Proof statement
theorem area_N1N2N3_one_sixteenth_area_ABC
  (hBC : B ≠ C) (hAC : A ≠ C) (hAB : A ≠ B)
  (col_DC : collinear ℝ ({B, C, D})) (col_EA : collinear ℝ ({A, C, E})) (col_FB : collinear ℝ ({A, B, F}))
  (int_AD : affine_combo ℝ {A, D}) (int_CF : affine_combo ℝ {C, F})
  (int_BE : affine_combo ℝ {B, E})
  (hN1 : affine_independent ℝ ({A, D, C, F})) (hN2 : affine_independent ℝ ({B, E, A, D})) (hN3 : affine_independent ℝ ({B, E, C, F})) :
  let [ABC] := K in
  let [N1N2N3] := K/16 in
  by sorry := sorry

end area_N1N2N3_one_sixteenth_area_ABC_l30_30129


namespace no_special_70_digit_divisible_l30_30847

def isSpecial70DigitNumber (n : ℤ) : Prop :=
  (∀ d : ℕ, d ∈ [8, 9, 0] → ¬(10 ≤ n.digit d ∧ n.digit d ≤ 70)) ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7] → (n.digit d = 10))

theorem no_special_70_digit_divisible (n_1 n_2 : ℤ) (h1: isSpecial70DigitNumber n_1) (h2: isSpecial70DigitNumber n_2) :
  ¬ (∃ k : ℕ, n_1 * k = n_2) :=
sorry

end no_special_70_digit_divisible_l30_30847


namespace expected_value_of_ones_on_three_dice_l30_30047

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30047


namespace distinct_prime_factors_of_B_l30_30472

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30472


namespace hide_and_seek_l30_30787

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30787


namespace trains_cross_time_l30_30098

theorem trains_cross_time
  (length_each_train : ℝ)
  (speed_each_train_kmh : ℝ)
  (relative_speed_m_s : ℝ)
  (total_distance : ℝ)
  (conversion_factor : ℝ) :
  length_each_train = 120 →
  speed_each_train_kmh = 27 →
  conversion_factor = 1000 / 3600 →
  relative_speed_m_s = speed_each_train_kmh * conversion_factor →
  total_distance = 2 * length_each_train →
  total_distance / relative_speed_m_s = 16 :=
by
  sorry

end trains_cross_time_l30_30098


namespace estimated_white_balls_is_correct_l30_30685

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of trials
def trials : ℕ := 100

-- Define the number of times a red ball is drawn
def red_draws : ℕ := 80

-- Define the function to estimate the number of red balls based on the frequency
def estimated_red_balls (total_balls : ℕ) (red_draws : ℕ) (trials : ℕ) : ℕ :=
  total_balls * red_draws / trials

-- Define the function to estimate the number of white balls
def estimated_white_balls (total_balls : ℕ) (estimated_red_balls : ℕ) : ℕ :=
  total_balls - estimated_red_balls

-- State the theorem to prove the estimated number of white balls
theorem estimated_white_balls_is_correct : 
  estimated_white_balls total_balls (estimated_red_balls total_balls red_draws trials) = 2 :=
by
  sorry

end estimated_white_balls_is_correct_l30_30685


namespace geometric_locus_is_two_annuli_l30_30878

noncomputable def locus_of_vertices_of_isosceles_right_triangles
  (O1 O2 : ℝ × ℝ) (r1 r2 : ℝ) : set (ℝ × ℝ) :=
  { P : ℝ × ℝ |
    let d := dist O1 O2 in
    let s := (r1 + r2) * real.sqrt 2 / 2 in
    let t := abs (r1 - r2) * real.sqrt 2 / 2 in
    dist P O1 = d / real.sqrt 2 ∧ dist P O2 = d / real.sqrt 2 }

theorem geometric_locus_is_two_annuli
  {O1 O2 : ℝ × ℝ} {r1 r2 : ℝ} :
  exists (O O' : ℝ × ℝ),
    locus_of_vertices_of_isosceles_right_triangles O1 O2 r1 r2 =
    { P : ℝ × ℝ |
      let s := (r1 + r2) * real.sqrt 2 / 2 in
      let t := abs (r1 - r2) * real.sqrt 2 / 2 in
      dist P O = s ∨ dist P O' = s ∨ dist P O = t ∨ dist P O' = t } :=
sorry

end geometric_locus_is_two_annuli_l30_30878


namespace number_of_distinct_prime_factors_of_B_l30_30436

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30436


namespace alex_play_friends_with_l30_30733

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30733


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30563

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30563


namespace modulus_of_z_l30_30900

noncomputable def z_modulus {z : ℂ} (h : (3 + 4 * Complex.i) * z = 1) : ℝ :=
  Complex.abs z

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * Complex.i) * z = 1) : z_modulus h = 1 / 5 :=
by
  sorry

end modulus_of_z_l30_30900


namespace expected_ones_on_three_dice_l30_30093

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30093


namespace length_BC_correct_l30_30342

noncomputable def length_BC : ℝ :=
  let f := λ x : ℝ, real.cos x
  let g := λ x : ℝ, real.sqrt 3 * real.sin x
  let A : ℝ × ℝ := (real.pi / 6, real.sqrt 3 / 2)
  let f' := λ x : ℝ, -real.sin x
  let g' := λ x : ℝ, real.sqrt 3 * real.cos x
  let slope_f := f' (real.pi / 6)
  let slope_g := g' (real.pi / 6)
  let tangent_f (x : ℝ) := slope_f * (x - real.pi / 6) + real.sqrt 3 / 2
  let tangent_g (x : ℝ) := slope_g * (x - real.pi / 6) + real.sqrt 3 / 2
  let x_B := real.pi / 6 + real.sqrt 3
  let x_C := real.pi / 6 - real.sqrt 3 / 3
  abs (x_B - x_C)

theorem length_BC_correct : length_BC = 4 * real.sqrt 3 / 3 := 
by
  sorry

end length_BC_correct_l30_30342


namespace rectangle_diagonal_length_l30_30647

theorem rectangle_diagonal_length (P : ℝ) (r : ℝ) (perimeter_eq : P = 72) (ratio_eq : r = 5/2) :
  let k := P / 14
  let l := 5 * k
  let w := 2 * k
  let d := Real.sqrt (l^2 + w^2)
  d = 194 / 7 :=
sorry

end rectangle_diagonal_length_l30_30647


namespace number_of_subsets_P_times_Q_l30_30339

-- Define the sets P and Q
def P : Set ℕ := {3, 4, 5}
def Q : Set ℕ := {6, 7}

-- Define the set product P × Q
def P_times_Q : Set (ℕ × ℕ) := { (a, b) | a ∈ P ∧ b ∈ Q }

-- Proof statement
theorem number_of_subsets_P_times_Q : (P_times_Q.card = 6) → (2 ^ 6 = 64) :=
by sorry

end number_of_subsets_P_times_Q_l30_30339


namespace sequence_max_length_y_l30_30250

theorem sequence_max_length_y (y : ℤ) (h1 : 2000 - y > 0) (h2 : 2 * y - 2000 > 0) (h3 : 4000 - 3 * y > 0) 
(h4 : 5 * y - 6000 > 0) (h5 : 10000 - 8 * y > 0) (h6 : 13 * y - 16000 > 0) (h7 : 26000 - 21 * y > 0)
(h8 : 34 * y - 42000 > 0) (h9 : 68000 - 55 * y > 0) :
2000 < y ∧ y = 1236 := 
by {
  calc 
  let lower_bound := 42000 / 34;
  let upper_bound := 68000 / 55;
  have : lower_bound < y := by sorry,
  have : y < upper_bound := by sorry,
  exact sorry }

end sequence_max_length_y_l30_30250


namespace Igor_colored_all_cells_l30_30788

theorem Igor_colored_all_cells (m n : ℕ) (h1 : 9 * m = 12 * n) (h2 : 0 < m ∧ m ≤ 4) (h3 : 0 < n ∧ n ≤ 3) :
  m = 4 ∧ n = 3 :=
by {
  sorry
}

end Igor_colored_all_cells_l30_30788


namespace triangle_area_l30_30375

noncomputable theory

open_locale classical

variables {A B C E : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space E]

structure Triangle (A B C : Type) :=
(angle_ABC_90 : ∠A B C = 90)
(midpoint_E : midpoint A C = E)
(perpendicular_BE_AC : perp BE AC)
(length_AE : |A - E| = 5)
(length_EC : |E - C| = 10)

theorem triangle_area : ∀ t : Triangle A B C,
  area (triangle A B C) = 37.5 :=
by sorry

end triangle_area_l30_30375


namespace triangle_angles_sin_A_plus_sin_B_l30_30386

theorem triangle_angles (A B C : ℝ) (h1 : tan ((A + B) / 2) + tan (C / 2) = (4 * real.sqrt 3) / 3) (h2 : sin B * sin C = cos (A / 2) ^ 2) (h3 : A + B + C = π) :
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 := 
sorry

theorem sin_A_plus_sin_B (A B : ℝ) (h4 : A + B = 2 * π / 3) (h5 : 0 < B ∧ B < π / 2) :
  (sqrt 3 / 2 < sin A + sin B) ∧ (sin A + sin B ≤ sqrt 3) := 
sorry

end triangle_angles_sin_A_plus_sin_B_l30_30386


namespace problem_l30_30406

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30406


namespace new_avg_salary_correct_l30_30623

variables {average_salary_old avg_old avg_new : ℝ}
variables {worker_salary_old worker_salary_new : ℝ}
variables {supervisor_salary_old supervisor_salary_new : ℝ}
variables {num_workers num_people : ℝ}

-- conditions
def average_salary_old := 430
def supervisor_salary_old := 870
def supervisor_salary_new := 690
def num_workers := 8
def num_people := num_workers + 1

-- calculations
def total_salary_old := average_salary_old * num_people
def worker_salary_old := total_salary_old - supervisor_salary_old
def total_salary_new := worker_salary_old + supervisor_salary_new
def avg_new := total_salary_new / num_people

theorem new_avg_salary_correct :
  avg_new = 410 :=
by
  unfold average_salary_old supervisor_salary_old supervisor_salary_new num_workers num_people total_salary_old worker_salary_old total_salary_new avg_new
  norm_num
  sorry

end new_avg_salary_correct_l30_30623


namespace product_of_divisors_has_three_prime_factors_l30_30460

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30460


namespace expected_number_of_ones_on_three_dice_l30_30077

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30077


namespace dinner_cost_l30_30840

theorem dinner_cost (tax_rate tip_rate total_cost : ℝ) (h_tax : tax_rate = 0.12) (h_tip : tip_rate = 0.20) (h_total : total_cost = 30.60) :
  let meal_cost := total_cost / (1 + tax_rate + tip_rate)
  meal_cost = 23.18 :=
by
  sorry

end dinner_cost_l30_30840


namespace distinct_prime_factors_of_B_l30_30468

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30468


namespace num_best_friends_l30_30725

theorem num_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) : total_cards / cards_per_friend = 5 :=
by
  -- We assume the proof is going to be done here
  sorry

end num_best_friends_l30_30725


namespace electronics_weight_l30_30130

variable (B C E : ℝ)

-- Conditions
def initial_ratio : Prop := B / 5 = C / 4 ∧ C / 4 = E / 2
def removed_clothes : Prop := B / 10 = (C - 9) / 4

-- Proof statement
theorem electronics_weight (h1 : initial_ratio B C E) (h2 : removed_clothes B C) : E = 9 := 
by
  sorry

end electronics_weight_l30_30130


namespace sum_formula_induction_l30_30694

theorem sum_formula_induction (n : ℕ) :
    (∑ i in Finset.range (n + 1), i * (3 * i + 1)) = n * (n + 1) ^ 2 := by
  induction n with k hk
  case zero =>
    simp
  case succ =>
    rw [Finset.sum_range_succ, hk]
    ring
    sorry

end sum_formula_induction_l30_30694


namespace distinct_prime_factors_of_product_of_divisors_l30_30516

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30516


namespace domain_of_f_min_value_of_f_l30_30941

noncomputable def f_dom (a : ℝ) (x : ℝ) : ℝ :=
  Real.log a (1 - x) + Real.log a (x + 3)

theorem domain_of_f (a : ℝ) : 0 < a ∧ a < 1 → (∀ x : ℝ, (f_dom a x).implies (-3 < x ∧ x < 1)) :=
  sorry

theorem min_value_of_f (a : ℝ) : 0 < a ∧ a < 1 → (∀ x : ℝ, (f_dom a x = -2).implies (a = 1 / 2)) :=
  sorry

end domain_of_f_min_value_of_f_l30_30941


namespace distinct_prime_factors_product_divisors_60_l30_30482

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30482


namespace two_thirds_greater_l30_30010

theorem two_thirds_greater :
  let epsilon : ℚ := (2 : ℚ) / (3 * 10^8)
  let decimal_part : ℚ := 66666666 / 10^8
  (2 / 3) - decimal_part = epsilon := by
  sorry

end two_thirds_greater_l30_30010


namespace log_7_over_5_not_expressible_l30_30303

theorem log_7_over_5_not_expressible (log2 log3 : ℝ) (h2 : log 2 = log2) (h3 : log 3 = log3) :
  ¬ (∃ a b : ℝ, log (7/5) = a * log2 + b * log3) :=
sorry

end log_7_over_5_not_expressible_l30_30303


namespace not_similar_equilateral_triangle_regular_pentagon_l30_30126

-- Definitions for equilateral triangles and regular pentagons
structure EquilateralTriangle :=
(angle : ℝ := 60)

structure RegularPentagon :=
(angle : ℝ := 108)

def areSimilar (P Q : Type) [hasEqualSides : P → Prop] [hasEqualAngles : P → Prop] (p1 p2 : P) : Prop :=
  sorry -- One can define "similar" polygons via transformations and proportions of corresponding sides and angles.

-- We state the theorem we need to prove
theorem not_similar_equilateral_triangle_regular_pentagon :
  ¬ areSimilar EquilateralTriangle RegularPentagon :=
sorry

end not_similar_equilateral_triangle_regular_pentagon_l30_30126


namespace vector_at_t5_l30_30810

theorem vector_at_t5 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) 
  (h1 : (a, b) = (2, 5)) 
  (h2 : (a + 3 * c, b + 3 * d) = (8, -7)) :
  (a + 5 * c, b + 5 * d) = (10, -11) :=
by
  sorry

end vector_at_t5_l30_30810


namespace complex_number_quadrant_l30_30667

noncomputable def z : ℂ := (-3 - 4 * complex.I) * complex.I

theorem complex_number_quadrant :
  let p := (z.re, z.im)
  in p.1 > 0 ∧ p.2 < 0 :=
by
  sorry

end complex_number_quadrant_l30_30667


namespace year_population_below_five_percent_l30_30249

def population (P0 : ℕ) (years : ℕ) : ℕ :=
  P0 / 2^years

theorem year_population_below_five_percent (P0 : ℕ) :
  ∃ n, population P0 n < P0 / 20 ∧ (2005 + n) = 2010 := 
by {
  sorry
}

end year_population_below_five_percent_l30_30249


namespace find_lambda_l30_30345

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_lambda (λ : ℝ) :
  let a := (1, λ)
  let b := (2, 1)
  let c := (1, -2)
  let d := (4, 2 * λ + 1)
  (collinear d c) → (λ = - 9 / 2) :=
by
  sorry

end find_lambda_l30_30345


namespace min_value_of_f_l30_30879

noncomputable def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x * y + y^2) - 3 * (x + y) + 5

theorem min_value_of_f :
  ∃ x y : ℝ, (x > 0) ∧ (y > 0) ∧ (∀ u v : ℝ, (u > 0) ∧ (v > 0) → f u v ≥ 2) ∧ f x y = 2 :=
by
  sorry

end min_value_of_f_l30_30879


namespace diagonal_length_l30_30654

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l30_30654


namespace roots_are_rational_l30_30610

theorem roots_are_rational (p q n : ℚ) : 
    let a := p + q + n
    let b := -2 * (p + q)
    let c := p + q - n
    let Δ := b^2 - 4 * a * c
    Δ = (2 * n)^2 → ∃ x y : ℚ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by {
  intro h,
  use [(b - (2 * n))/2/a, (b + (2 * n))/2/a],
  split,
  { sorry },
  { sorry }
}

end roots_are_rational_l30_30610


namespace calculate_expression_l30_30719

theorem calculate_expression :
  (2^(1^(0^2)))^3 + (3^(1^2))^0 + 4^(0^1) = 10 :=
by
  have h1 : 0^2 = 0 := by norm_num
  have h2 : 1^0 = 1 := by norm_num
  have h3 : 2^1 = 2 := by norm_num
  have h4 : 2^1^((1^(0^2))) = 2 := by rw [h1, h2]
  have h5 : (2^1)^3 = 8 := by norm_num
  have h6 : 1^2 = 1 := by norm_num
  have h7 : 3^1 = 3 := by norm_num
  have h8 : 3^0 = 1 := by norm_num
  have h9 : 0^1 = 0 := by norm_num
  have h10 : 4^0 = 1 := by norm_num
  calc (2^(1^(0^2)))^3 + (3^(1^2))^0 + 4^(0^1)
      = (2^1)^3 + (3^1)^0 + 4^0 := by rw [h4, h5, h6, h7, h8, h9, h10]
  ... = 8 + 1 + 1 := by norm_num
  ... = 10 := by norm_num

end calculate_expression_l30_30719


namespace wall_penetrating_skill_l30_30999

theorem wall_penetrating_skill (k : ℕ) (h : k≠0) (eqn : k * Real.sqrt (k / (k^2 - 1 : ℝ)) = Real.sqrt (k * (k / (k^2 - 1 : ℝ)))) : 
  ∃ n : ℕ, n = k^2 - 1 :=
by
  exists k^2 - 1
  sorry

end wall_penetrating_skill_l30_30999


namespace find_x_l30_30638

variable (a : ℝ) (x : ℝ)

def square_a := a * a
def square_b := (2 * a) * (2 * a)
def square_c := (2 * a * (1 + x / 100)) * (2 * a * (1 + x / 100))
def area_condition := square_c = 10.24 * a * a

theorem find_x : x = 60 := by
  sorry

end find_x_l30_30638


namespace scientific_notation_of_2_8_million_l30_30367

theorem scientific_notation_of_2_8_million :
  (∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ 2.8 * 10^6 = a * 10^n ∧
    (if 1 ≤ |2.8 * 10^6| then n ≥ 0 else n ≤ 0)) :=
sorry

end scientific_notation_of_2_8_million_l30_30367


namespace expected_ones_on_three_dice_l30_30089

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30089


namespace find_common_difference_l30_30297

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
∀ n m : ℕ, n < m → (a m - a n) = (m - n) * (a 1 - a 0)

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, S n = n * a 1 + (n * (n - 1)) / 2 * (a 1 - a 0)

noncomputable def quadratic_roots (c : ℚ) (x1 x2 : ℚ) : Prop :=
2 * x1^2 - 12 * x1 + c = 0 ∧ 2 * x2^2 - 12 * x2 + c = 0

theorem find_common_difference
  (a : ℕ → ℚ) (S : ℕ → ℚ) (c : ℚ)
  (h_arith_seq: is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_roots : quadratic_roots c (a 3) (a 7))
  (h_S13 : S 13 = c) :
  (a 1 - a 0 = -3/2) ∨ (a 1 - a 0 = -7/4) :=
sorry

end find_common_difference_l30_30297


namespace alex_plays_with_friends_l30_30753

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30753


namespace prime_factors_of_B_l30_30531

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30531


namespace fraction_decimal_equivalence_l30_30267

-- Define the fraction 7/343
def fraction := 7 / 343

-- Define the decimal equivalent
def decimal_equivalent := 0.056

-- Prove that the fraction is equivalent to the decimal
theorem fraction_decimal_equivalence : fraction = decimal_equivalent :=
sorry

end fraction_decimal_equivalence_l30_30267


namespace expected_number_of_ones_on_three_dice_l30_30071

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30071


namespace expected_ones_three_standard_dice_l30_30057

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30057


namespace sum_of_c_and_d_l30_30972

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := (x - 3) / (x^2 + c * x + d)

theorem sum_of_c_and_d (c d : ℝ) (h_asymptote1 : (2:ℝ)^2 + c * 2 + d = 0) (h_asymptote2 : (-1:ℝ)^2 - c + d = 0) :
  c + d = -3 :=
by
-- theorem body (proof omitted)
sorry

end sum_of_c_and_d_l30_30972


namespace complex_modulus_problem_l30_30325

theorem complex_modulus_problem (z : ℂ) (hz : z = -1 - I) : 
  |(1 - z) * conj(z)| = Real.sqrt(10) :=
by
  have hconj : conj(z) = -1 + I := by
    rw [hz, Complex.conj, Complex.neg_re, Complex.neg_im, Complex.of_real_neg, Complex.of_real_neg]
    rfl
  rw [hconj]
  sorry

end complex_modulus_problem_l30_30325


namespace total_sum_of_grid_is_745_l30_30383

theorem total_sum_of_grid_is_745 :
  let top_row := [12, 13, 15, 17, 19]
  let left_column := [12, 14, 16, 18]
  let total_sum := 360 + 375 + 10
  total_sum = 745 :=
by
  -- The theorem establishes the total sum calculation.
  sorry

end total_sum_of_grid_is_745_l30_30383


namespace draw_3_odd_balls_from_15_is_336_l30_30140

-- Define the problem setting as given in the conditions
def odd_balls : Finset ℕ := {1, 3, 5, 7, 9, 11, 13, 15}

-- Define the function that calculates the number of ways to draw 3 balls
noncomputable def draw_3_odd_balls (S : Finset ℕ) : ℕ :=
  S.card * (S.card - 1) * (S.card - 2)

-- Prove that the drawing of 3 balls results in 336 ways
theorem draw_3_odd_balls_from_15_is_336 : draw_3_odd_balls odd_balls = 336 := by
  sorry

end draw_3_odd_balls_from_15_is_336_l30_30140


namespace find_certain_number_l30_30264

theorem find_certain_number (n : ℕ)
  (h1 : 3153 + 3 = 3156)
  (h2 : 3156 % 9 = 0)
  (h3 : 3156 % 70 = 0)
  (h4 : 3156 % 25 = 0) :
  3156 % 37 = 0 :=
by
  sorry

end find_certain_number_l30_30264


namespace total_writing_instruments_l30_30602

theorem total_writing_instruments 
 (bags : ℕ) (compartments_per_bag : ℕ) (empty_compartments : ℕ) (one_compartment : ℕ) (remaining_compartments : ℕ) 
 (writing_instruments_per_compartment : ℕ) (writing_instruments_in_one : ℕ) : 
 bags = 16 → 
 compartments_per_bag = 6 → 
 empty_compartments = 5 → 
 one_compartment = 1 → 
 remaining_compartments = 90 →
 writing_instruments_per_compartment = 8 → 
 writing_instruments_in_one = 6 → 
 (remaining_compartments * writing_instruments_per_compartment + one_compartment * writing_instruments_in_one) = 726 := 
  by
   sorry

end total_writing_instruments_l30_30602


namespace distinct_prime_factors_of_B_l30_30470

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30470


namespace expected_number_of_ones_when_three_dice_rolled_l30_30034

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30034


namespace total_profit_correct_l30_30728

noncomputable def total_profit (c_share : ℝ) (c_percentage : ℝ) : ℝ :=
  (c_share * 100) / c_percentage

theorem total_profit_correct (c_share : ℝ) (c_percentage : ℝ) (P : ℝ) :
  c_share = (c_percentage / 100) * P → 
  c_share = 43000 → 
  c_percentage = 25 → 
  P = 172000 :=
by
  intros h1 h2 h3
  simp [h2, h3, total_profit]
  sorry

end total_profit_correct_l30_30728


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30506

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30506


namespace linear_function_quadrants_l30_30983

theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) : 
  (∀ x : ℝ, (k < 0 ∧ b > 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) ∧ 
  (∀ x : ℝ, (k > 0 ∧ b < 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) :=
sorry

end linear_function_quadrants_l30_30983


namespace probability_range_l30_30315

noncomputable def probability_distribution (K : ℕ) : ℝ :=
  if K > 0 then 1 / (2^K) else 0

theorem probability_range (h2 : 2 < 3) (h3 : 3 ≤ 4) :
  probability_distribution 3 + probability_distribution 4 = 3 / 16 :=
by
  sorry

end probability_range_l30_30315


namespace rectangle_dimensions_l30_30817

theorem rectangle_dimensions (a1 a2 : ℝ) (h1 : a1 * a2 = 216) (h2 : a1 + a2 = 30 - 6)
  (h3 : 6 * 6 = 36) : (a1 = 12 ∧ a2 = 18) ∨ (a1 = 18 ∧ a2 = 12) :=
by
  -- The conditions are set; now we need the proof, which we'll replace with sorry for now.
  sorry

end rectangle_dimensions_l30_30817


namespace problem_condition_l30_30307

theorem problem_condition (f : ℝ → ℝ) (a : ℝ)
  (f_even : ∀ x, f x = f (-x))
  (f_def_pos : ∀ x, 0 < x → if 4 < x then f x = a * x + log 5 x else f x = x ^ 2 + 2 ^ x + 3)
  (hyp : f (-5) < f 2) : a < 2 :=
by
  sorry

end problem_condition_l30_30307


namespace expected_ones_on_three_dice_l30_30094

theorem expected_ones_on_three_dice : (expected_number_of_ones 3) = 1 / 2 :=
by
  sorry

def expected_number_of_ones (n : ℕ) : ℚ :=
  (n : ℚ) * (1 / 6)

end expected_ones_on_three_dice_l30_30094


namespace one_vertical_asymptote_l30_30893

noncomputable def f (c : ℝ) := 
  (λ x, (x^2 + 2 * x + c) / (x^2 - 5 * x + 6))

theorem one_vertical_asymptote (c : ℝ) :
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x^2 + 2 * x + c = 0)) ∧ 
  (¬∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x^2 + 2 * x + c ≠ 0)) ↔ 
  (c = -8 ∨ c = -15) :=
by 
  sorry

end one_vertical_asymptote_l30_30893


namespace sum_of_factors_24_l30_30716

theorem sum_of_factors_24 : ∑ i in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), i = 60 := 
by
  sorry

end sum_of_factors_24_l30_30716


namespace count_irrational_numbers_l30_30179

def is_irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

noncomputable def num_list : List ℝ := [
  3.14159,
  -real.cbrt 9,
  0.131131113.repr.to_real,
  -real.pi,
  real.sqrt 25,
  real.cbrt 64,
  -1 / 7
]

theorem count_irrational_numbers :
  (num_list.filter is_irrational).length = 3 := 
sorry

end count_irrational_numbers_l30_30179


namespace trajectory_of_P_is_lambda_plus_mu_is_constant_l30_30954

noncomputable def trajectory_eqn (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

theorem trajectory_of_P_is :
  ∀ (x y : ℝ), 
    (∃ (x1 y1 x2 y2 : ℝ), y1 = (sqrt 3 / 3) * x1 ∧ y2 = -(sqrt 3 / 3) * x2 ∧
     x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2 ∧ (x1 - x2)^2 + (y1 - y2)^2 = 12) →
    trajectory_eqn x y :=
sorry

theorem lambda_plus_mu_is_constant :
  ∀ (k : ℝ) (x3 y3 x4 y4 y5 : ℝ), 
    (x3, y3) ∈ trajectory_eqn ∧ (x4, y4) ∈ trajectory_eqn ∧
    y3 = k * (x3 - 1) ∧ y4 = k * (x4 - 1) ∧ y5 ∈ /(0, ∞) ∧
    ∃ (λ μ : ℝ), x3 = λ / (1 + λ) ∧ x4 = μ / (1 + μ) →
  ∃ (λ μ : ℝ), λ + μ = -9 / 4 :=
sorry

end trajectory_of_P_is_lambda_plus_mu_is_constant_l30_30954


namespace power_mod_l30_30882

theorem power_mod : (5 ^ 2023) % 11 = 4 := 
by 
  sorry

end power_mod_l30_30882


namespace alex_plays_with_friends_l30_30752

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30752


namespace ben_remaining_bonus_l30_30838

theorem ben_remaining_bonus :
  let total_bonus := 1496
  let kitchen_expense := total_bonus * (1/22 : ℚ)
  let holiday_expense := total_bonus * (1/4 : ℚ)
  let gift_expense := total_bonus * (1/8 : ℚ)
  let total_expense := kitchen_expense + holiday_expense + gift_expense
  total_bonus - total_expense = 867 :=
by
  let total_bonus := 1496
  let kitchen_expense := total_bonus * (1/22 : ℚ)
  let holiday_expense := total_bonus * (1/4 : ℚ)
  let gift_expense := total_bonus * (1/8 : ℚ)
  let total_expense := kitchen_expense + holiday_expense + gift_expense
  have h1 : kitchen_expense = 68 := by sorry
  have h2 : holiday_expense = 374 := by sorry
  have h3 : gift_expense = 187 := by sorry
  have h4 : total_expense = 629 := by sorry
  show total_bonus - total_expense = 867 from by
    calc
      total_bonus - total_expense
      = 1496 - 629 : by rw [h4]
      ... = 867 : by sorry

end ben_remaining_bonus_l30_30838


namespace problem_l30_30137

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the given conditions in Lean
def condition1 (f : ℝ → ℝ) := odd_function f
def condition2 (f : ℝ → ℝ) := ∀ x, x > 0 → x * (f x)' + 2 * f x > 0

-- Define the problem statement in Lean
theorem problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) : 4 * f 2 < 9 * f 3 :=
sorry

end problem_l30_30137


namespace median_salary_is_28000_l30_30174

def number_of_employees : List (String × Nat) :=
  [("CEO", 1),
   ("Senior Vice-President", 4),
   ("Manager", 12),
   ("Senior Consultant", 8),
   ("Consultant", 49)]

def salary_by_position (position : String) : Nat :=
  match position with
  | "CEO" => 140000
  | "Senior Vice-President" => 95000
  | "Manager" => 80000
  | "Senior Consultant" => 55000
  | "Consultant" => 28000
  | _ => 0

def total_employees : Nat :=
  number_of_employees.map (λ (_, n) => n).sum

def median_position : Nat :=
  (total_employees + 1) / 2

def find_position_title (num_list : List (String × Nat)) (pos : Nat) : String :=
  match num_list with
  | [] => ""
  | (title, count) :: rest =>
    if pos ≤ count then title else find_position_title rest (pos - count)

theorem median_salary_is_28000 : 
  find_position_title number_of_employees median_position = "Consultant" → salary_by_position "Consultant" = 28000 :=
by 
  intros h 
  rw h
  exact rfl

end median_salary_is_28000_l30_30174


namespace min_value_expression_l30_30578

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + ((b / a) - 1)^2 + ((c / b) - 1)^2 + ((5 / c) - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := 
by
  sorry

end min_value_expression_l30_30578


namespace student_comprehensive_score_l30_30147

def comprehensive_score (t_score i_score d_score : ℕ) (t_ratio i_ratio d_ratio : ℕ) :=
  (t_score * t_ratio + i_score * i_ratio + d_score * d_ratio) / (t_ratio + i_ratio + d_ratio)

theorem student_comprehensive_score :
  comprehensive_score 95 88 90 2 5 3 = 90 :=
by
  -- The proof goes here
  sorry

end student_comprehensive_score_l30_30147


namespace solution_set_inequality_l30_30025

noncomputable def solution_set := {x : ℝ | (x + 1) * (x - 2) ≤ 0 ∧ x ≠ -1}

theorem solution_set_inequality :
  solution_set = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by {
-- Insert proof here
sorry
}

end solution_set_inequality_l30_30025


namespace rosy_days_to_complete_work_l30_30596

-- Definitions based on conditions
def mary_days : ℕ := 11
def rosy_efficiency_factor : ℝ := 1.10

-- Main statement to be proved
theorem rosy_days_to_complete_work : 
  ∀ (mary_days : ℕ) (rosy_efficiency_factor : ℝ), 
  mary_days = 11 → 
  rosy_efficiency_factor = 1.10 → 
  let rosy_days := 1 / ((1 / mary_days.toReal) * rosy_efficiency_factor) in
  rosy_days = 10 := 
by
  intros mary_days rosy_efficiency_factor h_md h_ref
  -- Here we would provide the proof steps
  sorry

end rosy_days_to_complete_work_l30_30596


namespace cube_root_identity_l30_30136

theorem cube_root_identity (π : Real) : (∛((π - 2) ^ 3)) = π - 2 :=
  sorry

end cube_root_identity_l30_30136


namespace smallest_palindrome_in_bases_2_4_l30_30216

def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

def to_digits (n : Nat) (base : Nat) : List Char :=
  Nat.digits base n |> List.map (λ d, Char.ofNat (d + 48))

def is_palindrome_in_base (n base : Nat) : Bool :=
  is_palindrome (to_digits n base)

theorem smallest_palindrome_in_bases_2_4 :
  ∃ n : Nat, n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧
  ∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m :=
  ∃ n : Nat, n = 15 ∧ n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧ (∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m) :=
  sorry

end smallest_palindrome_in_bases_2_4_l30_30216


namespace position_of_2021_over_2019_l30_30951

def sequence_fraction_position (m n : ℕ) : ℕ :=
  let k := m + n
  let total_up_to_k_minus_1 := (k - 2) * (k - 1) / 2
  let current_index := m - 1
  total_up_to_k_minus_1 + current_index

theorem position_of_2021_over_2019 :
  sequence_fraction_position 2021 2019 = 8159741 :=
by
  unfold sequence_fraction_position
  have k_eq : 2021 + 2019 = 4040 := by norm_num
  have total_up_to_4039 := (4040 - 2) * (4040 - 1) / 2 := by norm_num
  have current_pos := 2021 - 1 := by norm_num
  rw [k_eq, total_up_to_4039, current_pos]
  norm_num
  sorry

end position_of_2021_over_2019_l30_30951


namespace solve_for_a_l30_30974

theorem solve_for_a (x a : ℝ) (h : x = -2) (hx : 2 * x + 3 * a = 0) : a = 4 / 3 :=
by
  sorry

end solve_for_a_l30_30974


namespace friends_who_participate_l30_30777

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30777


namespace clock_angle_9_15_l30_30707

/-- Definition of minute hand position at 15 minutes past the hour -/
def minute_hand_position (time_minutes : Int) : ℝ :=
  (time_minutes / 60) * 360

/-- Definition of hour hand position at 9:15 -/
def hour_hand_position (time_hour : Int) (time_minutes : Int) : ℝ :=
  let hour_part := time_hour * 30
  let minute_part := (time_minutes / 60) * 30
  hour_part + minute_part

/-- Definition to calculate acute angle between hands -/
def acute_angle (angle1 angle2 : ℝ) : ℝ :=
  let angle := abs (angle1 - angle2)
  if angle > 180 then 360 - angle else angle

/-- Theorem to prove the acute angle formed by the hands of a clock at 9:15 is 172.5 degrees. -/
theorem clock_angle_9_15 : 
  let time_hour := 9
  let time_minutes := 15
  acute_angle (hour_hand_position time_hour time_minutes) (minute_hand_position time_minutes) = 172.5 :=
by
  sorry

end clock_angle_9_15_l30_30707


namespace distinct_prime_factors_of_product_of_divisors_l30_30428

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30428


namespace distinct_prime_factors_product_divisors_60_l30_30492

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30492


namespace quadrilateral_area_correct_l30_30159

def is_in_quadrilateral (x y : ℝ) : Prop :=
  (x = 0 ∨ x = 4) ∧ (y = x - 2 ∨ y = x + 3)

def quadrilateral_area : ℝ :=
  20

theorem quadrilateral_area_correct :
  ∀ A B C D : (ℝ × ℝ),
  is_in_quadrilateral A.1 A.2 →
  is_in_quadrilateral B.1 B.2 →
  is_in_quadrilateral C.1 C.2 →
  is_in_quadrilateral D.1 D.2 →
  -- conditions to ensure A, B, C, D form the quadrilateral
  A = (0, -2) →
  B = (0, 3) →
  C = (4, 2) →
  D = (4, 7) →
  let area := quadrilateral_area
  in  area = 20 := sorry

end quadrilateral_area_correct_l30_30159


namespace infinitely_many_no_zero_divisible_by_sum_of_digits_l30_30618

theorem infinitely_many_no_zero_divisible_by_sum_of_digits :
  ∀ n : ℕ, ∃ a : ℕ, (digits a).count (0 : ℕ) = 0 ∧ (digits a).sum = 3 * n ∧ a % (digits a).sum = 0 :=
by
  sorry

end infinitely_many_no_zero_divisible_by_sum_of_digits_l30_30618


namespace distinct_prime_factors_product_divisors_60_l30_30493

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30493


namespace linear_combination_of_coprimes_l30_30013

theorem linear_combination_of_coprimes (
  a : ℕ → ℕ, n : ℕ) (h_gcd : nat.gcd (list.of_fn a) = 1) (N : ℕ) :
  ∃ (x : fin n → ℕ), N = ∑ i, a i * x i := 
by sorry

end linear_combination_of_coprimes_l30_30013


namespace sum_of_geometric_sequence_l30_30381

theorem sum_of_geometric_sequence :
  ∀ (a : ℕ → ℝ) (r : ℝ),
  (∃ a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ,
   a 1 = a_1 ∧ a 2 = a_2 ∧ a 3 = a_3 ∧ a 4 = a_4 ∧ a 5 = a_5 ∧ a 6 = a_6 ∧ a 7 = a_7 ∧ a 8 = a_8 ∧ a 9 = a_9 ∧
   a_1 * r^1 = a_2 ∧ a_1 * r^2 = a_3 ∧ a_1 * r^3 = a_4 ∧ a_1 * r^4 = a_5 ∧ a_1 * r^5 = a_6 ∧ a_1 * r^6 = a_7 ∧ a_1 * r^7 = a_8 ∧ a_1 * r^8 = a_9 ∧
   a_1 + a_2 + a_3 = 8 ∧
   a_4 + a_5 + a_6 = -4) →
  a 7 + a 8 + a 9 = 2 :=
sorry

end sum_of_geometric_sequence_l30_30381


namespace geometric_sequence_common_ratio_simple_sequence_general_term_l30_30138

-- Question 1
theorem geometric_sequence_common_ratio (a_3 : ℝ) (S_3 : ℝ) (q : ℝ) (h1 : a_3 = 3 / 2) (h2 : S_3 = 9 / 2) :
    q = -1 / 2 ∨ q = 1 :=
sorry

-- Question 2
theorem simple_sequence_general_term (S : ℕ → ℝ) (a : ℕ → ℝ) (h : ∀ n, S n = n^2) :
    ∀ n, a n = S n - S (n - 1) → ∀ n, a n = 2 * n - 1 :=
sorry

end geometric_sequence_common_ratio_simple_sequence_general_term_l30_30138


namespace positive_difference_100_l30_30394

theorem positive_difference_100 :
  let jo_sum := (100 * 101) / 2
  let alex_sum := (5 * 200)
  abs (jo_sum - alex_sum) = 4050 :=
by
  let jo_sum := (100 * 101) / 2
  let alex_sum := (5 * 200)
  have h1 : jo_sum = 5050 := by norm_num
  have h2 : alex_sum = 1000 := by norm_num
  rw [h1, h2]
  exact (abs (5050 - 1000)).symm
  sorry

end positive_difference_100_l30_30394


namespace five_digit_number_l30_30382

open Nat

noncomputable def problem_statement : Prop :=
  ∃ A B C D E F : ℕ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A + B + C + D + E + F = 25 ∧
    (A, B, C, D, E, F) = (3, 4, 2, 1, 6, 9)

theorem five_digit_number : problem_statement := 
  sorry

end five_digit_number_l30_30382


namespace intersects_to_equilateral_on_sphere_l30_30376

-- Definitions of point, line, and geometric constructs
def Point : Type := ℝ × ℝ × ℝ
def Line (p1 p2 : Point) : Type := { k : ℝ // ∃ t : ℝ, k = (p1.1 + t * (p2.1 - p1.1), p1.2 + t * (p2.2 - p1.2), p1.3 + t * (p2.3 - p1.3)) }

-- Definition of a triangle in 3D space
structure Triangle :=
  (a b c : Point)

-- Definition of a Tetrahedron in 3D space
structure Tetrahedron :=
  (a b c d : Point)

-- Conditions for the problem setup
variables (A B C D K L M : Point)
variables (tetra : Tetrahedron)
variables (sphere : Point → Prop)
variables (height_A height_B height_C : ℝ)

-- Hypotheses
def conditions : Prop :=
  -- Tetrahedron vertices
  tetra.a = A ∧ tetra.b = B ∧ tetra.c = C ∧ tetra.d = D ∧
  -- Heights conditions
  (distance D A) = height_A ∧ (distance D B) = height_B ∧ (distance D C) = height_C ∧
  -- Sphere intersection points
  sphere A ∧ sphere B ∧ sphere C ∧
  sphere K ∧ sphere L ∧ sphere M ∧
  -- Intersection points on the appropriate edges
  (∃ tA : ℝ, Line D A tA = K) ∧
  (∃ tB : ℝ, Line D B tB = L) ∧
  (∃ tC : ℝ, Line D C tC = M)

-- Theorem to prove
theorem intersects_to_equilateral_on_sphere :
  conditions A B C D K L M tetra sphere height_A height_B height_C →
  Equilateral △ (K,L,M) :=
sorry

end intersects_to_equilateral_on_sphere_l30_30376


namespace minimum_box_surface_area_l30_30683

theorem minimum_box_surface_area :
  let l := 10 in
  let w := 5 in
  let h := 3 in
  let num_bars := 4 in
  let stacked_height := num_bars * h in
  let A := 2 * l * w + 2 * l * stacked_height + 2 * w * stacked_height in
  A = 460 := by sorry

end minimum_box_surface_area_l30_30683


namespace vehicle_worth_l30_30616

-- Definitions from the conditions
def monthlyEarnings : ℕ := 4000
def savingFraction : ℝ := 0.5
def savingMonths : ℕ := 8

-- Theorem statement
theorem vehicle_worth : (monthlyEarnings * savingFraction * savingMonths : ℝ) = 16000 := 
by
  sorry

end vehicle_worth_l30_30616


namespace measure_of_B_l30_30388

theorem measure_of_B (A B C : ℝ) (h1 : B = A + 20) (h2 : C = 50) (h3 : A + B + C = 180) : B = 75 := by
  sorry

end measure_of_B_l30_30388


namespace fraction_calculation_l30_30027

theorem fraction_calculation : (36 - 12) / (12 - 4) = 3 :=
by
  sorry

end fraction_calculation_l30_30027


namespace sin_square_range_l30_30024

theorem sin_square_range (a : ℝ) : 
  (∃ x : ℝ, sin x * sin x - 2 * sin x = a) ↔ a ∈ set.Icc (-1 : ℝ) 3 :=
by
  sorry

end sin_square_range_l30_30024


namespace equation_of_line_l30_30995

variable {M : ℝ × ℝ} (hxM : M = (-4, 0))
variable {circle_center : ℝ × ℝ} (hxC : circle_center = (1, 0))
variable {circle_radius : ℝ} (hx_radius : circle_radius ^ 2 = 5)
variable {A B : ℝ × ℝ}
variable {l : ℝ × ℝ → Prop} (hl : l M ∧ l A ∧ l B)
variable (h_midpoint : (A.1 + B.1) / 2 = M.1 ∧ (A.2 + B.2) / 2 = M.2)

theorem equation_of_line :
  l = λ P, P.2 = (1 / 3) * (P.1 + 4) ∨ P.2 = -(1 / 3) * (P.1 + 4) := sorry

end equation_of_line_l30_30995


namespace quadratic_function_range_of_a_l30_30012

theorem quadratic_function_range_of_a (a b c : ℝ) 
    (h1 : (-2, 1) ∈ set_of (λ p : ℝ × ℝ, p.2 = a * p.1^2 + b * p.1 + c))
    (h2 : (2, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = a * p.1^2 + b * p.1 + c))
    (h3 : 0 < c ∧ c < 1) :
    (1 / 4 < a ∧ a < 1 / 2) :=
by {
  sorry
}

end quadratic_function_range_of_a_l30_30012


namespace smallest_palindrome_base2_base4_gt10_l30_30202

/--
  Compute the smallest base-10 positive integer greater than 10 that is a palindrome 
  when written in both base 2 and base 4.
-/
theorem smallest_palindrome_base2_base4_gt10 : 
  ∃ n : ℕ, 10 < n ∧ is_palindrome (nat_to_base 2 n) ∧ is_palindrome (nat_to_base 4 n) ∧ n = 15 :=
sorry

def nat_to_base (b n : ℕ) : list ℕ :=
sorry

def is_palindrome {α : Type} [DecidableEq α] (l : list α) : bool :=
sorry

end smallest_palindrome_base2_base4_gt10_l30_30202


namespace max_sqrt_sum_l30_30284

theorem max_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 10) : 
  sqrt (3 * b) + sqrt (2 * a) ≤ 2 * sqrt 5 :=
sorry

end max_sqrt_sum_l30_30284


namespace modulus_of_z_l30_30324

-- Define the complex number z
def z : ℂ := (2 - Complex.i) / (2 + Complex.i)

-- State the theorem
theorem modulus_of_z : Complex.abs z = 1 := by
  sorry

end modulus_of_z_l30_30324


namespace acute_triangle_locus_l30_30905

theorem acute_triangle_locus {A B C : Point} (AB : Segment A B)
    (S : Circle (midpoint A B) (dist A B / 2))
    (lA lB : Line)
    (hA : tangent lA S A)
    (hB : tangent lB S B) :
  (inside_band C lA lB) ∧ (outside_circle C S) ↔ (acute_triangle A B C) := sorry

end acute_triangle_locus_l30_30905


namespace distinct_prime_factors_of_B_l30_30453

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30453


namespace expected_ones_in_three_dice_rolls_l30_30045

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30045


namespace num_satisfying_permutations_l30_30587

def is_permutation {α : Type*} [DecidableEq α] (l1 l2 : List α) : Prop :=
  l1 ~ l2

def satisfies_condition (a : Fin 106 → ℕ) : Prop :=
  ∀ (m : ℕ), m ∈ {3, 5, 7} → ∀ n, 1 ≤ n ∧ n + m ≤ 105 → m ∣ (a ⟨n + m, sorry⟩ - a ⟨n, sorry⟩)

theorem num_satisfying_permutations :
  ∃! (a : Fin 106 → ℕ), is_permutation (List.ofFn a) (List.range 106) ∧ satisfies_condition a :=
3628800 :=
sorry

end num_satisfying_permutations_l30_30587


namespace hide_and_seek_l30_30782

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30782


namespace last_digit_3_pow_1991_plus_1991_pow_3_l30_30120

theorem last_digit_3_pow_1991_plus_1991_pow_3 :
  (3 ^ 1991 + 1991 ^ 3) % 10 = 8 :=
  sorry

end last_digit_3_pow_1991_plus_1991_pow_3_l30_30120


namespace convert_angle_to_rad_form_l30_30854

theorem convert_angle_to_rad_form :
  ∃ (α : ℝ) (k : ℤ), 0 ≤ α ∧ α < 2 * Real.pi ∧ -1485 * Real.pi / 180 = α + 2 * k * Real.pi ∧
   α - 2 * k * Real.pi = - (33* Real.pi) / 4 :=
begin
  sorry,
end

end convert_angle_to_rad_form_l30_30854


namespace area_of_parallelogram_l30_30253

variables (v w : ℝ × ℝ × ℝ)

-- Definitions of the vectors
def vector_v := (2, -4, 3)
def vector_w := (-1, 5, -2)

-- The proof statement
theorem area_of_parallelogram : 
  let cross_product := 
    (vector_v.2 * vector_w.3 - vector_v.3 * vector_w.2, 
     vector_v.3 * vector_w.1 - vector_v.1 * vector_w.3, 
     vector_v.1 * vector_w.2 - vector_v.2 * vector_w.1) in
  let magnitude := real.sqrt (cross_product.1 ^ 2 + cross_product.2 ^ 2 + cross_product.3 ^ 2) in
  magnitude = real.sqrt 86 := 
by 
  sorry

end area_of_parallelogram_l30_30253


namespace stock_price_end_of_second_year_l30_30868

theorem stock_price_end_of_second_year : 
  let P_0 := 50 in
  let P_1 := P_0 + P_0 * 1.5 in
  let P_2 := P_1 - P_1 * 0.3 in
  P_2 = 87.5 :=
by
  sorry

end stock_price_end_of_second_year_l30_30868


namespace pow_mod_remainder_l30_30885

theorem pow_mod_remainder (n : ℕ) : 5 ^ 2023 % 11 = 4 :=
by sorry

end pow_mod_remainder_l30_30885


namespace problem_l30_30408

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30408


namespace find_circle_equation_l30_30627

def point := ℝ × ℝ

def C : point := (-1/2, 3)

def line_l (x y : ℝ) : Prop := x + 2 * y - 3 = 0

def perpendicular_vectors (P Q : point) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  px * qx + py * qy = 0

theorem find_circle_equation (P Q : point) 
  (hP : line_l P.1 P.2) (hQ : line_l Q.1 Q.2) 
  (hperpendicular : perpendicular_vectors P Q) :
  ∃ r, ∀ x y, (x + 1/2)^2 + (y - 3)^2 = r^2 ∧ r = 5/2 :=
sorry

end find_circle_equation_l30_30627


namespace truncated_tetrahedron_volume_is_correct_l30_30800

noncomputable def volume_of_truncated_tetrahedron : ℝ :=
  let original_tetrahedron_volume := (3^3 * real.sqrt(2)) / 12 in
  let small_tetrahedron_volume := (1^3 * real.sqrt(2)) / 12 in
  original_tetrahedron_volume - 4 * small_tetrahedron_volume

theorem truncated_tetrahedron_volume_is_correct :
  volume_of_truncated_tetrahedron = (23 * real.sqrt(2)) / 12 :=
by
  sorry

end truncated_tetrahedron_volume_is_correct_l30_30800


namespace angle_equality_DBC_ACD_l30_30400

-- Define the points and their relationships based on the problem conditions
variables {A B C M D N : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M] [Inhabited D] [Inhabited N]

-- Define the midpoint property and angle conditions
axiom midpoint_AB (M A B : Type) [grp: Group M] : B ∈ midpoint A B
axiom right_angle (A B C : Type) [grp : Group A] : A ∈ right_angle B C
axiom point_on_median (D N M A C : Type) : D ∈ median M A C
axiom angle_equality (N A C D B M : Type) [grp: Group A] : angle N A C = angle N D B
axiom angle_equality2 (N A C D B M : Type) [grp: Group A] : angle A C D = angle B C M

-- State that ∠DBC = ∠ACD 
theorem angle_equality_DBC_ACD {M A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M] [Inhabited D] :
  (midpoint_AB M A B) → (right_angle A B C) → (angle_equality N A C D B M) →  (angle_equality2 N A C D B M) → 
  angle D B C = angle A C D :=
  by {
    sorry
  }

end angle_equality_DBC_ACD_l30_30400


namespace smallest_n_terminating_decimal_l30_30108

theorem smallest_n_terminating_decimal : ∃ n : ℕ, (∀ m : ℕ, m < n → (∀ k : ℕ, (n = 103 + k) → (∃ a b : ℕ, k = 2^a * 5^b)) → (k ≠ 0 → k = 125)) ∧ n = 22 := 
sorry

end smallest_n_terminating_decimal_l30_30108


namespace number_of_distinct_prime_factors_of_B_l30_30429

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30429


namespace hide_and_seek_problem_l30_30747

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30747


namespace not_monotonic_on_interval_l30_30985

noncomputable def f (x : ℝ) : ℝ := (x^2 / 2) - Real.log x

theorem not_monotonic_on_interval (m : ℝ) : 
  (∃ x y : ℝ, m < x ∧ x < m + 1/2 ∧ m < y ∧ y < m + 1/2 ∧ (x ≠ y) ∧ f x ≠ f y ) ↔ (1/2 < m ∧ m < 1) :=
sorry

end not_monotonic_on_interval_l30_30985


namespace expected_number_of_ones_l30_30064

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30064


namespace value_range_of_func_l30_30679

noncomputable def func (x : ℝ) : ℝ :=
  real.cos (2 * x) - 8 * real.cos x

theorem value_range_of_func :
  ∃ a b, (∀ x, real.cos x ∈ Icc (-1 : ℝ) 1 → func x ∈ Icc a b) ∧ a = -7 ∧ b = 9 :=
by
  sorry

end value_range_of_func_l30_30679


namespace pascal_third_number_in_51_row_l30_30115

-- Definition and conditions
def pascal_row_num := 50
def third_number_index := 2

-- Statement of the problem
theorem pascal_third_number_in_51_row : 
  (nat.choose pascal_row_num third_number_index) = 1225 :=
by {
  -- The proof step will be skipped using sorry
  sorry
}

end pascal_third_number_in_51_row_l30_30115


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30568

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30568


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30504

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30504


namespace distinct_prime_factors_B_l30_30548

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30548


namespace nth_term_formula_l30_30678

theorem nth_term_formula (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * n^2 + n)
  (h2 : a 1 = S 1)
  (h3 : ∀ n ≥ 2, a n = S n - S (n - 1))
  : ∀ n, a n = 4 * n - 1 := by
  sorry

end nth_term_formula_l30_30678


namespace union_is_012_l30_30952

def M := {1, x}
def N := {0, 2}
def condition := M ∩ N = {2}
def union_M_N := M ∪ N

theorem union_is_012 (x : ℕ) (h : condition) : union_M_N = {0, 1, 2} :=
sorry

end union_is_012_l30_30952


namespace smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30208

def is_palindrome_in_base (n : ℕ) (b : ℕ) : Prop :=
  let digits : List ℕ := (nat.to_digits b n)
  digits = digits.reverse

theorem smallest_palindrome_in_bases_2_and_4 :
  ∀ n : ℕ, 10 < n ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 → n ≥ 17 :=
begin
  intros n hn,
  have := (by norm_num : 16 < 17),
  cases hn with hn1 hn2,
  cases hn2 with hn3 hn4,
  linarith,
end

theorem smallest_palindrome_in_bases_2_and_4_17 :
  is_palindrome_in_base 17 2 ∧ is_palindrome_in_base 17 4 :=
begin
  split;
  { unfold is_palindrome_in_base, norm_num, refl },
end

end smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30208


namespace employee_B_payment_l30_30691

theorem employee_B_payment (x : ℝ) (h1 : ∀ A B : ℝ, A + B = 580) (h2 : A = 1.5 * B) : B = 232 :=
by
  sorry

end employee_B_payment_l30_30691


namespace alex_plays_with_friends_l30_30748

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30748


namespace smallest_palindrome_base2_base4_gt10_l30_30203

/--
  Compute the smallest base-10 positive integer greater than 10 that is a palindrome 
  when written in both base 2 and base 4.
-/
theorem smallest_palindrome_base2_base4_gt10 : 
  ∃ n : ℕ, 10 < n ∧ is_palindrome (nat_to_base 2 n) ∧ is_palindrome (nat_to_base 4 n) ∧ n = 15 :=
sorry

def nat_to_base (b n : ℕ) : list ℕ :=
sorry

def is_palindrome {α : Type} [DecidableEq α] (l : list α) : bool :=
sorry

end smallest_palindrome_base2_base4_gt10_l30_30203


namespace range_of_a_l30_30364

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 3 4, 2^(x^2 + 1) ≤ (1/4)^(3/2 - a * x)) ↔ a ∈ set.Ici (5 / 2) := 
sorry

end range_of_a_l30_30364


namespace multiple_of_960_l30_30577

theorem multiple_of_960 (a : ℤ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) :
  ∃ k : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * k :=
  sorry

end multiple_of_960_l30_30577


namespace smallest_factorization_c_l30_30886

theorem smallest_factorization_c : ∃ (c : ℤ), (∀ (r s : ℤ), r * s = 2016 → r + s = c) ∧ c > 0 ∧ c = 108 :=
by 
  sorry

end smallest_factorization_c_l30_30886


namespace expected_ones_in_three_dice_rolls_l30_30039

open ProbabilityTheory

theorem expected_ones_in_three_dice_rolls :
  let p := (1 / 6 : ℝ)
  let q := (5 / 6 : ℝ)
  let expected_value := (0 * (q ^ 3) + 1 * (3 * p * (q ^ 2)) + 2 * (3 * (p ^ 2) * q) + 3 * (p ^ 3))
  in expected_value = 1 / 2 :=
by
  -- Sorry, full proof is not provided.
  sorry

end expected_ones_in_three_dice_rolls_l30_30039


namespace number_of_ordered_pairs_l30_30963

def satisfies_equation (x y : ℤ) : Prop :=
  x^3 + y^2 = 2 * y + 1

theorem number_of_ordered_pairs : 
  (finset.univ.image (λ p : ℤ × ℤ, if satisfies_equation p.1 p.2 then some p else none)).filter_map id).card = 2 :=
sorry

end number_of_ordered_pairs_l30_30963


namespace expected_ones_three_dice_l30_30083

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30083


namespace ben_bonus_leftover_l30_30837

theorem ben_bonus_leftover (b : ℝ) (k h c : ℝ) (bk : k = 1/22 * b) (bh : h = 1/4 * b) (bc : c = 1/8 * b) :
  b - (k + h + c) = 867 :=
by
  sorry

end ben_bonus_leftover_l30_30837


namespace no_rank_3_necessary_l30_30809

open Matrix

def is_closed_under_operations (A : set (Matrix (Fin 5) (Fin 7) ℝ)) : Prop :=
  ∀ (x y : Matrix (Fin 5) (Fin 7) ℝ) (c : ℝ), (x ∈ A ∧ y ∈ A) → (c • x + y) ∈ A

def contains_given_ranks (A : set (Matrix (Fin 5) (Fin 7) ℝ)) : Prop :=
  (∃ (x : Matrix (Fin 5) (Fin 7) ℝ), rank x = 0 ∧ x ∈ A) ∧
  (∃ (x : Matrix (Fin 5) (Fin 7) ℝ), rank x = 1 ∧ x ∈ A) ∧
  (∃ (x : Matrix (Fin 5) (Fin 7) ℝ), rank x = 2 ∧ x ∈ A) ∧
  (∃ (x : Matrix (Fin 5) (Fin 7) ℝ), rank x = 4 ∧ x ∈ A) ∧
  (∃ (x : Matrix (Fin 5) (Fin 7) ℝ), rank x = 5 ∧ x ∈ A)

theorem no_rank_3_necessary (A : set (Matrix (Fin 5) (Fin 7) ℝ)) : 
  is_closed_under_operations A →
  contains_given_ranks A →
  ¬ (∀ x : Matrix (Fin 5) (Fin 7) ℝ, x ∈ A → rank x = 3) :=
sorry

end no_rank_3_necessary_l30_30809


namespace power_mod_l30_30883

theorem power_mod : (5 ^ 2023) % 11 = 4 := 
by 
  sorry

end power_mod_l30_30883


namespace distinct_prime_factors_B_l30_30554

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30554


namespace ben_bonus_leftover_l30_30836

theorem ben_bonus_leftover (b : ℝ) (k h c : ℝ) (bk : k = 1/22 * b) (bh : h = 1/4 * b) (bc : c = 1/8 * b) :
  b - (k + h + c) = 867 :=
by
  sorry

end ben_bonus_leftover_l30_30836


namespace distinct_prime_factors_of_B_l30_30476

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30476


namespace smallest_palindrome_in_bases_2_4_l30_30217

def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

def to_digits (n : Nat) (base : Nat) : List Char :=
  Nat.digits base n |> List.map (λ d, Char.ofNat (d + 48))

def is_palindrome_in_base (n base : Nat) : Bool :=
  is_palindrome (to_digits n base)

theorem smallest_palindrome_in_bases_2_4 :
  ∃ n : Nat, n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧
  ∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m :=
  ∃ n : Nat, n = 15 ∧ n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧ (∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m) :=
  sorry

end smallest_palindrome_in_bases_2_4_l30_30217


namespace friends_who_participate_l30_30776

/-- Definitions for the friends' participation in hide and seek -/
variables (A B V G D : Prop)

/-- Conditions given in the problem -/
axiom axiom1 : A → (B ∧ ¬V)
axiom axiom2 : B → (G ∨ D)
axiom axiom3 : ¬V → (¬B ∧ ¬D)
axiom axiom4 : ¬A → (B ∧ ¬G)

/-- Proof that B, V, and D will participate in hide and seek -/
theorem friends_who_participate : B ∧ V ∧ D :=
sorry

end friends_who_participate_l30_30776


namespace hide_and_seek_friends_l30_30763

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30763


namespace distinct_prime_factors_of_product_of_divisors_l30_30419

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30419


namespace distinct_prime_factors_of_B_l30_30478

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30478


namespace current_rate_is_4_l30_30677

-- Definition for the conditions
def boat_speed_still_water : ℝ := 18
def time_downstream_hr : ℝ := 14 / 60
def distance_downstream_km : ℝ := 5.133333333333334

-- Definition for the rate of the current
def rate_of_current (c : ℝ) : Prop :=
  distance_downstream_km = (boat_speed_still_water + c) * time_downstream_hr

-- Lean 4 statement for the proof
theorem current_rate_is_4 : rate_of_current 4 :=
by
  -- Here we would provide the proof steps
  sorry

end current_rate_is_4_l30_30677


namespace find_97th_digit_of_1_div_13_l30_30103

theorem find_97th_digit_of_1_div_13 :
  let dec_repr_1_13 := "076923" in
  (dec_repr_1_13.length = 6) →
  (97 % dec_repr_1_13.length = 1) →
  dec_repr_1_13[0] = '0'
:= by
  intros dec_repr_1_13 h_length h_mod
  simp only [String.length, h_length, Nat.mod_eq_of_lt]
  sorry

end find_97th_digit_of_1_div_13_l30_30103


namespace probability_of_prime_ball_l30_30235

def is_prime (n : Nat) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

def balls : List Nat := [3, 4, 5, 6, 7, 8, 11, 13]

def count_prime_balls : ℕ := (balls.filter is_prime).length

def total_balls : ℕ := balls.length

theorem probability_of_prime_ball : (count_prime_balls / total_balls : ℚ) = 5 / 8 := by
  sorry

end probability_of_prime_ball_l30_30235


namespace circle_radius_l30_30946

theorem circle_radius (x y : ℝ) :
  x^2 + 2 * x + y^2 = 0 → 1 = 1 :=
by sorry

end circle_radius_l30_30946


namespace no_real_k_for_coplanar_lines_l30_30228

theorem no_real_k_for_coplanar_lines :
  ∀ k : ℝ,
  let p1 := (3 : ℝ, 2 : ℝ, 7 : ℝ)
  let d1 := (2 : ℝ, 1 : ℝ, -k)
  let p2 := (4 : ℝ, 5 : ℝ, 6 : ℝ)
  let d2 := (k, 3, 2) in
  let a := (2 : ℝ, 1 : ℝ, -k) in
  let b := (k, 3, 2) in
  let c := (1 : ℝ, 3 : ℝ, -1 : ℝ) in
  det3 a b c ≠ 0 :=
by
  sorry

end no_real_k_for_coplanar_lines_l30_30228


namespace prime_factors_of_B_l30_30530

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30530


namespace good_number_power_l30_30982

def is_good_number (x : ℝ) : Prop :=
  ∃ n : ℕ+, x = Real.sqrt n + Real.sqrt (n + 1)

theorem good_number_power (x : ℝ) (r : ℕ+) (hx : is_good_number x) : is_good_number (x ^ r) :=
sorry

end good_number_power_l30_30982


namespace largest_value_is_B_l30_30125

def exprA := 1 + 2 * 3 + 4
def exprB := 1 + 2 + 3 * 4
def exprC := 1 + 2 + 3 + 4
def exprD := 1 * 2 + 3 + 4
def exprE := 1 * 2 + 3 * 4

theorem largest_value_is_B : exprB = 15 ∧ exprB > exprA ∧ exprB > exprC ∧ exprB > exprD ∧ exprB > exprE := 
by
  sorry

end largest_value_is_B_l30_30125


namespace equilateral_given_inequality_l30_30931

open Real

-- Define the primary condition to be used in the theorem
def inequality (a b c : ℝ) : Prop :=
  (1 / a * sqrt (1 / b + 1 / c) + 1 / b * sqrt (1 / c + 1 / a) + 1 / c * sqrt (1 / a + 1 / b)) ≥
  (3 / 2 * sqrt ((1 / a + 1 / b) * (1 / b + 1 / c) * (1 / c + 1 / a)))

-- Define the theorem that states the sides form an equilateral triangle under the given condition
theorem equilateral_given_inequality (a b c : ℝ) (habc : inequality a b c) (htriangle : a > 0 ∧ b > 0 ∧ c > 0):
  a = b ∧ b = c ∧ c = a := 
sorry

end equilateral_given_inequality_l30_30931


namespace alex_plays_with_friends_l30_30754

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end alex_plays_with_friends_l30_30754


namespace volume_ratio_l30_30296

noncomputable def base_diameter_cone := 2 * r
noncomputable def slant_height_cone := 2 * r
noncomputable def height_cone := Math.sqrt 3 * r
noncomputable def V1 := (1 / 3) * Real.pi * r^2 * (Math.sqrt 3 * r)
noncomputable def radius_inscribed_sphere := (Math.sqrt 3 / 3) * r
noncomputable def V2 := (4 / 3) * Real.pi * (radius_inscribed_sphere)^3

theorem volume_ratio (r : ℝ) : V1 / V2 = 9 / 4 :=
by
  sorry

end volume_ratio_l30_30296


namespace intersection_correct_l30_30790

variable (A B : Set ℝ)  -- Define variables A and B as sets of real numbers

-- Define set A as {x | -3 ≤ x < 4}
def setA : Set ℝ := {x | -3 ≤ x ∧ x < 4}

-- Define set B as {x | -2 ≤ x ≤ 5}
def setB : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- The goal is to prove the intersection of A and B is {x | -2 ≤ x < 4}
theorem intersection_correct : setA ∩ setB = {x : ℝ | -2 ≤ x ∧ x < 4} := sorry

end intersection_correct_l30_30790


namespace number_of_distinct_prime_factors_of_B_l30_30439

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30439


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30567

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30567


namespace distinct_prime_factors_of_B_l30_30474

-- Definitions from the problem statement
def B : ℕ := ∏ d in (multiset.powerset_len (nat.factors 60)).val, d

-- The theorem statement
theorem distinct_prime_factors_of_B : (nat.factors B).to_finset.card = 3 := by
  sorry

end distinct_prime_factors_of_B_l30_30474


namespace shifted_graph_equiv_l30_30096

-- Definitions to be used (based on the conditions provided):
def given_function (x : ℝ) : ℝ := 4 * sin (x - π / 6) * cos (x - π / 6)
def target_function (x : ℝ) : ℝ := 2 * sin (2 * x)
def shifted_target_function (x : ℝ) : ℝ := target_function (x - π / 6)

-- The proof problem to solve:
theorem shifted_graph_equiv :
  ∀ x : ℝ, given_function x = shifted_target_function x :=
by
  -- Proof omitted
  sorry

end shifted_graph_equiv_l30_30096


namespace tangent_lines_count_l30_30350

-- Definitions of the conditions
def point (x y : ℝ) := (x, y)
def circle (center : point ℝ ℝ) (radius : ℝ) : set (point ℝ ℝ) := 
  {p | (fst p - fst center)^2 + (snd p - snd center)^2 = radius^2}

-- Conditions
def point_P := point 2 3
def circle_C := circle (point 0 0) 2

-- Theorem statement
theorem tangent_lines_count : 
  ∃(l1 l2 : set (point ℝ ℝ)), 
    (∀p, p ∈ l1 → (fst p - 2)^2 + (snd p - 3)^2 = 4) ∧
    (∀p, p ∈ l2 → (fst p - 2)^2 + (snd p - 3)^2 = 4) ∧
    (∀p, p ∈ l1 → ∃q, q ∈ circle_C ∧ (fst p - fst q)*(snd p - snd q) = 1) ∧
    (∀p, p ∈ l2 → ∃q, q ∈ circle_C ∧ (fst p - fst q)*(snd p - snd q) = 1) ∧
    ∃q1 q2, q1 ∈ circle_C ∧ q2 ∈ circle_C ∧ (fst q1 = 2 ∨ fst q2 = 2) :=
sorry

end tangent_lines_count_l30_30350


namespace cesaro_sum_of_100_terms_l30_30891

noncomputable def CesaroSum (p : List ℝ) : ℝ :=
  let S := List.scanl (+) 0 p.tail
  (S.tail.sum / p.length)

theorem cesaro_sum_of_100_terms {p : List ℝ} (h_len : p.length = 99) (h_cesaro : CesaroSum p = 1000) :
  CesaroSum (9::p) = 999 :=
by
  sorry

end cesaro_sum_of_100_terms_l30_30891


namespace marbles_left_l30_30682

theorem marbles_left (red_marble_count blue_marble_count broken_marble_count : ℕ)
  (h1 : red_marble_count = 156)
  (h2 : blue_marble_count = 267)
  (h3 : broken_marble_count = 115) :
  red_marble_count + blue_marble_count - broken_marble_count = 308 :=
by
  sorry

end marbles_left_l30_30682


namespace PQ_over_AB_l30_30932

variable (P Q A B C : Type) [NormedAddCommGroup P] [NormedAddCommGroup Q] [NormedAddCommGroup A] 
                              [NormedAddCommGroup B] [NormedAddCommGroup C]
                              (PA PB PC QA QB QC: P → Q)
                              (PQ AB : P)

-- Conditions
def P_condition (P A B C: P) := PA + (2 * PB) + (3 * PC) = 0
def Q_condition (Q A B C: Q) := (2 * QA) + (3 * QB) + (5 * QC) = 0

-- Question to Prove
theorem PQ_over_AB (hP : P_condition P A B C) (hQ : Q_condition Q A B C):
  ∥PQ∥ / ∥AB∥ = 1/30 :=
begin
  sorry
end

end PQ_over_AB_l30_30932


namespace diagonal_length_l30_30649

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l30_30649


namespace rectangle_diagonal_l30_30662

theorem rectangle_diagonal (k : ℚ)
  (h1 : 2 * (5 * k + 2 * k) = 72)
  (h2 : k = 36 / 7) :
  let l := 5 * k,
      w := 2 * k,
      d := Real.sqrt ((l ^ 2) + (w ^ 2)) in
  d = 194 / 7 := by
  sorry

end rectangle_diagonal_l30_30662


namespace inequality_may_not_hold_l30_30965

theorem inequality_may_not_hold (a b : ℝ) (h : 0 < b ∧ b < a) :
  ¬(∀ x y : ℝ,  x = 1 / (a - b) → y = 1 / b → x > y) :=
sorry

end inequality_may_not_hold_l30_30965


namespace limit_n_b_n_l30_30856

def M (x : ℝ) : ℝ := x - (x^3) / 3

def b_n (n : ℕ) : ℝ :=
  let rec iter (k : ℕ) (x : ℝ) : ℝ :=
    if k = 0 then x else iter (k - 1) (M x)
  iter n (23 / n)

theorem limit_n_b_n : 
  filter.tendsto (λ n : ℕ, n * b_n n) filter.at_top (filter.const ℝ (69 / 20)) :=
sorry

end limit_n_b_n_l30_30856


namespace dragon_can_cover_center_l30_30830

-- Definitions of the conditions
structure Disc where
  radius : ℝ

structure Dragon where
  covers_center : Bool

-- Given conditions
axiom D1 : Disc
axiom D2 : Disc
axiom dragon_D1 : Dragon
axiom dragon_D2 : Dragon
axiom discs_identical : D1 = D2
axiom dragon_on_D1_covers_center : dragon_D1.covers_center = true
axiom dragon_on_D2_does_not_cover_center : dragon_D2.covers_center = false

-- Proof statement
theorem dragon_can_cover_center :
  ∃ P1 P2 : Disc, (P1 ∪ P2 = D2) ∧ P1 ∩ P2 = ∅ ∧
  ((∃Pr1 Pr2 : Disc, (Pr1 = D2 ∪ P1) ∧ (Pr2 = D2 ∪ P2)) ∧ dragon_D2.covers_center = true) := 
  sorry

end dragon_can_cover_center_l30_30830


namespace max_volume_rectangular_solid_l30_30697

noncomputable def max_volume : ℝ :=
  let V : ℝ → ℝ := λ x, -6 * x ^ 3 + 9 * x ^ 2 in
  if h : 0 < 1 ∧ 1 < 3 / 2 then V 1 else 0

theorem max_volume_rectangular_solid (h1 : true) (h2 : true) (h3 : ∀ x, (V x) = -6 * x ^ 3 + 9 * x ^ 2) :
  max_volume = 3 :=
by
  sorry

end max_volume_rectangular_solid_l30_30697


namespace min_val_x_y_l30_30918

theorem min_val_x_y (x y : ℝ) (h1 : xy + 1 = 4x + y) (h2 : x > 1) : (x + 1) * (y + 2) ≥ 27 :=
by sorry

end min_val_x_y_l30_30918


namespace pascal_triangle_row_num_l30_30117

theorem pascal_triangle_row_num (n k : ℕ) (hn : n = 50) (hk : k = 2) : 
  nat.choose 50 2 = 1225 :=
by
  rw [nat.choose, hn, hk]
  sorry

end pascal_triangle_row_num_l30_30117


namespace smallest_palindromic_integer_l30_30213

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := Nat.digits b n
  digits = List.reverse digits

theorem smallest_palindromic_integer :
  ∃ (n : ℕ), n > 10 ∧ is_palindrome n 2 ∧ is_palindrome n 4 ∧ (∀ m, m > 10 ∧ is_palindrome m 2 ∧ is_palindrome m 4 → n ≤ m) :=
begin
  sorry
end

end smallest_palindromic_integer_l30_30213


namespace smallest_n_terminating_decimal_l30_30110

theorem smallest_n_terminating_decimal : 
  ∃ (n : ℕ), (n > 0) ∧ (∀ p, prime p → p ∣ (n + 103) → (p = 2 ∨ p = 5)) ∧ (n = 22) := 
by
  sorry

end smallest_n_terminating_decimal_l30_30110


namespace wise_men_strategy_exists_l30_30622

-- Define the number of wise men and colors
def num_wise_men : ℕ := 300
def num_colors : ℕ := 25

-- Condition: Each wise man sees all other hats except his own
-- Condition: All hat color counts are unique from 0 to 24

-- Theorem statement
theorem wise_men_strategy_exists :
  ∃ (strategy : (vector (fin num_colors) (num_wise_men - 1) → fin num_colors) → (fin num_wise_men → fin num_colors)), 
  ∀ (hat_colors : fin num_wise_men → fin num_colors),
  (∀ i j, i ≠ j → hat_colors i ≠ hat_colors j) →
  (finset.card {i | strategy (λ x, if x < i then hat_colors (fin.of_nat (x.1)) else hat_colors (fin.of_nat (x.1 + 1))) = hat_colors i} ≥ 150) :=
begin
  sorry
end

end wise_men_strategy_exists_l30_30622


namespace distinct_prime_factors_of_product_of_divisors_l30_30517

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30517


namespace pascal_triangle_row_num_l30_30118

theorem pascal_triangle_row_num (n k : ℕ) (hn : n = 50) (hk : k = 2) : 
  nat.choose 50 2 = 1225 :=
by
  rw [nat.choose, hn, hk]
  sorry

end pascal_triangle_row_num_l30_30118


namespace prime_factors_of_B_l30_30527

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30527


namespace inequality_proof_l30_30614

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 3 ≥ ab + real.sqrt 3 * (a + b) :=
  sorry

end inequality_proof_l30_30614


namespace least_value_of_g_2016_l30_30152

def proper_divisor (m n : ℕ) : Prop := m ∣ n ∧ m ≠ n

noncomputable def g : ℕ → ℕ := sorry

theorem least_value_of_g_2016 :
  (∀ (m n : ℕ), proper_divisor m n → g(m) < g(n)) →
  (∀ (m n : ℕ), Nat.gcd m n = 1 ∧ m > 1 ∧ n > 1 → g(m * n) = g(m) g(n) + (n + 1) g(m) + (m + 1) g(n) + m + n) →
  g(2016) = 3053 :=
sorry

end least_value_of_g_2016_l30_30152


namespace capacity_of_each_bucket_in_second_case_final_proof_l30_30792

def tank_volume (buckets: ℕ) (bucket_capacity: ℝ) : ℝ := buckets * bucket_capacity

theorem capacity_of_each_bucket_in_second_case
  (total_volume: ℝ)
  (first_case_buckets : ℕ)
  (first_case_capacity : ℝ)
  (second_case_buckets : ℕ) :
  first_case_buckets * first_case_capacity = total_volume → 
  (total_volume / second_case_buckets) = 9 :=
by
  intros h
  sorry

-- Given the conditions:
noncomputable def total_volume := tank_volume 28 13.5

theorem final_proof :
  (tank_volume 28 13.5 = total_volume) → 
  (total_volume / 42 = 9) :=
by
  intro h
  exact capacity_of_each_bucket_in_second_case total_volume 28 13.5 42 h

end capacity_of_each_bucket_in_second_case_final_proof_l30_30792


namespace expected_ones_three_dice_l30_30084

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30084


namespace problem_l30_30405

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30405


namespace sqrt_n_P_triangle_inequality_l30_30133

variable {R : Type*} [LinearOrderedField R]

noncomputable def P (x : R) : R := a_n * x ^ n + a_(n-1) * x ^ (n-1) + ... + a_1 * x + a_0

theorem sqrt_n_P_triangle_inequality 
(a b c : R) (n : ℕ) (h1 : n ≥ 2) 
(hP : ∀ x, P x ≥ 0) 
(h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) :
  (√[n] (P a) < √[n] (P b) + √[n] (P c) ∧
   √[n] (P b) < √[n] (P a) + √[n] (P c) ∧
   √[n] (P c) < √[n] (P a) + √[n] (P b)) :=
by
  sorry

end sqrt_n_P_triangle_inequality_l30_30133


namespace total_seats_in_theater_l30_30992

theorem total_seats_in_theater 
    (n : ℕ) 
    (a1 : ℕ)
    (an : ℕ)
    (d : ℕ)
    (h1 : a1 = 12)
    (h2 : d = 2)
    (h3 : an = 48)
    (h4 : an = a1 + (n - 1) * d) :
    (n = 19) →
    (2 * (a1 + an) * n / 2 = 570) :=
by
  intros
  sorry

end total_seats_in_theater_l30_30992


namespace sum_of_p_equals_26_l30_30887

theorem sum_of_p_equals_26 :
  (∑ p in {p : ℤ | ∃ q r : ℤ, q ≠ r ∧ (λ (x : ℤ), (x - p) * (x - 13) + 4) = (λ (x : ℤ), (x + q) * (x + r)) ∧ p > 0}.to_finset, p) = 26 :=
by
  sorry

end sum_of_p_equals_26_l30_30887


namespace expected_value_of_ones_on_three_dice_l30_30054

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30054


namespace problem_l30_30415

theorem problem (B : ℕ) (h : B = (∏ d in (finset.divisors 60), d)) : (nat.factors B).to_finset.card = 3 :=
sorry

end problem_l30_30415


namespace num_best_friends_l30_30726

theorem num_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) : total_cards / cards_per_friend = 5 :=
by
  -- We assume the proof is going to be done here
  sorry

end num_best_friends_l30_30726


namespace initial_books_l30_30606

theorem initial_books (sold_books : ℕ) (given_books : ℕ) (remaining_books : ℕ) 
                      (h1 : sold_books = 11)
                      (h2 : given_books = 35)
                      (h3 : remaining_books = 62) :
  (sold_books + given_books + remaining_books = 108) :=
by
  -- Proof skipped
  sorry

end initial_books_l30_30606


namespace tangent_lines_through_M_find_a_value_l30_30950

def point_M : ℝ × ℝ := (3, 1)

def circle_center : ℝ × ℝ := (1, 2)

def radius : ℝ := 2

def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

def line_eq (a x y : ℝ) : Prop := ax - y + 4 = 0

theorem tangent_lines_through_M :
  ∀ (x y : ℝ),
    (circle_eq x y → (x = 3 ∨ 3 * x - 4 * y - 5 = 0)) := sorry

theorem find_a_value (a : ℝ) :
  (∃ (x y : ℝ), (circle_eq x y ∧ line_eq a x y)) →
  (√((circle_center.1 - a)^2 + (circle_center.2)^2) = 2 * √3) →
  a = -3 / 4 := sorry

end tangent_lines_through_M_find_a_value_l30_30950


namespace hide_and_seek_l30_30769

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30769


namespace expected_value_of_ones_on_three_dice_l30_30053

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30053


namespace log_base_three_of_one_over_nine_eq_neg_two_l30_30244

theorem log_base_three_of_one_over_nine_eq_neg_two :
  (∃ y, y = Real.log 3 (1 / 9)) → ∃ y, y = -2 :=
by
  sorry

end log_base_three_of_one_over_nine_eq_neg_two_l30_30244


namespace distinct_prime_factors_of_product_of_divisors_l30_30515

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30515


namespace hide_and_seek_problem_l30_30743

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30743


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30569

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30569


namespace find_x2_l30_30929

theorem find_x2 (a x1 x2 : ℝ) (h1 : x1 + x2 = 2) (h2 : x1 + 2 * x2 = 3 - sqrt 2) (h3 : (x1 - x2) = (sqrt 2 - 1)) : 
  x2 = 1 - sqrt 2 :=
by 
  sorry

end find_x2_l30_30929


namespace product_of_divisors_has_three_prime_factors_l30_30462

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30462


namespace square_circle_tangent_relation_square_circle_minimum_areas_rectangle_circle_minimum_areas_l30_30994

variable {r1 r2 : ℝ}
variable {s : ℝ := 1}
variable {l : ℝ := 3/2}

theorem square_circle_tangent_relation
  (ABCD : Square s) 
  (O1 O2 : Circle) 
  (h1 : IsTangent O1 O2 Externally) 
  (h2 : IsTangentToSides O1 A B D) 
  (h3 : IsTangentToSides O2 B C D)
  (r1 := O1.radius) 
  (r2 := O2.radius)
  : r1 + r2 = 2 - Real.sqrt 2 := 
sorry

theorem square_circle_minimum_areas
  (ABCD : Square s) 
  (O1 O2 : Circle) 
  (h1 : IsTangent O1 O2 Externally) 
  (h2 : IsTangentToSides O1 A B D) 
  (h3 : IsTangentToSides O2 B C D)
  (r1 := O1.radius) 
  (r2 := O2.radius)
  : π * (r1^2 + r2^2) = (3 - 2 * Real.sqrt 2) * π := 
sorry

theorem rectangle_circle_minimum_areas
  (ABCD : Rectangle s l) 
  (O1 O2 : Circle) 
  (h1 : IsTangent O1 O2 Externally) 
  (h2 : IsTangentToSides O1 A B D) 
  (h3 : IsTangentToSides O2 B C D)
  (r1 := O1.radius) 
  (r2 := O2.radius)
  : π * (r1^2 + r2^2) = π * (37 / 8 - 5 * Real.sqrt 3 / 4) := 
sorry

end square_circle_tangent_relation_square_circle_minimum_areas_rectangle_circle_minimum_areas_l30_30994


namespace smallest_odd_prime_factor_2021_8_plus_1_l30_30863

theorem smallest_odd_prime_factor_2021_8_plus_1 :
  ∃ p : ℕ, p > 1 ∧ p.prime ∧ p % 2 = 1 ∧ p ∣ (2021^8 + 1) ∧
           ∀ q : ℕ, q > 1 ∧ q.prime ∧ q % 2 = 1 ∧ q ∣ (2021^8 + 1) → p ≤ q :=
  sorry

end smallest_odd_prime_factor_2021_8_plus_1_l30_30863


namespace infinite_series_sine_square_l30_30971

theorem infinite_series_sine_square (θ : ℝ) (h : ∑' n : ℕ, (sin θ)^(2*n) = 4) : sin (2 * θ) = sqrt 3 / 2 := by
  sorry

end infinite_series_sine_square_l30_30971


namespace coordinates_equidistant_l30_30846

-- Define the condition of equidistance
theorem coordinates_equidistant (x y : ℝ) :
  (x + 2) ^ 2 + (y - 2) ^ 2 = (x - 2) ^ 2 + y ^ 2 →
  y = 2 * x + 1 :=
  sorry  -- Proof is omitted

end coordinates_equidistant_l30_30846


namespace quadratic_trinomial_constant_l30_30935

theorem quadratic_trinomial_constant (m : ℝ) (h : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
sorry

end quadratic_trinomial_constant_l30_30935


namespace distinct_prime_factors_of_product_of_divisors_l30_30425

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30425


namespace area_of_triangle_l30_30895

-- Defining the problem conditions
variable (A : ℝ) (a : ℝ) (sin_B sin_C : ℝ)

-- Given conditions
def cond_A : A = 60 := by sorry
def cond_a : a = sqrt 3 := by sorry
def cond_sin_relationship : sin_B + sin_C = 6 * sqrt 2 * sin_B * sin_C := by sorry

-- The required proof statement
theorem area_of_triangle (A_eq_60 : A = 60)
                        (a_eq_sqrt3 : a = sqrt 3)
                        (sin_relationship : sin_B + sin_C = 6 * sqrt 2 * sin_B * sin_C) :
  let b := 2 * sin_B
  let c := 2 * sin_C
  let bc := b * c
  let area := (1 / 2) * bc * (sqrt 3 / 2)
  area = sqrt 3 / 8 
  := by sorry

end area_of_triangle_l30_30895


namespace positive_t_value_l30_30275

theorem positive_t_value (t : ℝ) (ht : t > 0) (h : abs (Complex.mk (-5) t) = 3 * Real.sqrt 6) : 
  t = Real.sqrt 29 :=
by
  sorry

end positive_t_value_l30_30275


namespace rope_folded_three_times_parts_l30_30815

theorem rope_folded_three_times_parts (total_length : ℕ) :
  ∀ parts : ℕ, parts = (total_length / 8) →
  ∀ n : ℕ, n = 3 →
  (∀ length_each_part : ℚ, length_each_part = 1 / (2 ^ n) →
  length_each_part = 1 / 8) :=
by
  sorry

end rope_folded_three_times_parts_l30_30815


namespace hide_and_seek_l30_30781

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30781


namespace rectangle_AE_length_l30_30374

theorem rectangle_AE_length (AB BC : ℝ) (h₀ : AB = 30) (h₁ : BC = 15) (E : ℝ × ℝ) (h₂ : E.2 = 0) (h₃ : ∃ t, E = (t, 0)) (h₄ : ∃ θ, θ = 30 ∧ ∃ BE : ℝ, sin θ = 15 / BE) : 
    let BE := sqrt ((E.1 - BC)^2 + BC^2) in 
    let AE := sqrt (AB^2 + (E.1 - BC)^2) in 
  AE = 30 := sorry

end rectangle_AE_length_l30_30374


namespace expected_number_of_ones_on_three_dice_l30_30074

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30074


namespace prime_factors_of_B_l30_30532

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30532


namespace distinct_prime_factors_B_l30_30550

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30550


namespace coefficient_of_x3_is_zero_l30_30876

noncomputable def coefficient_of_x3 : ℚ :=
  let expr := (x^3 / 3 - 1 / x^2) ^ 9 in
  (coeff 3 (expand expr))

theorem coefficient_of_x3_is_zero : coefficient_of_x3 = 0 := sorry

end coefficient_of_x3_is_zero_l30_30876


namespace ceil_sqrt_sum_l30_30241

theorem ceil_sqrt_sum :
  (Int.ceil (Real.sqrt 20) + Int.ceil (Real.sqrt 200) + Int.ceil (Real.sqrt 2000)) = 65 := 
by
  calc
    Int.ceil (Real.sqrt 20) = 5 := 
      sorry
    Int.ceil (Real.sqrt 200) = 15 := 
      sorry
    Int.ceil (Real.sqrt 2000) = 45 := 
      sorry
    5 + 15 + 45 = 65 := 
      sorry

end ceil_sqrt_sum_l30_30241


namespace units_digit_of_153_base_3_l30_30229

theorem units_digit_of_153_base_3 :
  (153 % 3 ^ 1) = 2 := by
sorry

end units_digit_of_153_base_3_l30_30229


namespace parallelogram_isosceles_equilateral_l30_30399

variables {α : Type*} [euclidean_space α]

structure Parallelogram (A B C D : α) : Prop :=
(parallelogram : true) -- This should represent the parallelogram properties abstractly.

-- Define isosceles and equilateral triangles
def is_isosceles (A E B : α) (base : α) : Prop := dist A E = dist B E
def is_equilateral (D E F : α) : Prop := dist D E = dist E F ∧ dist E F = dist F D

-- Problem Statement
theorem parallelogram_isosceles_equilateral (A B C D E F : α) (h : Parallelogram A B C D)
(h_angle : ¬ ∃ θ : ℝ, θ = 60) 
(h_iso1 : is_isosceles A E B A) 
(h_iso2 : is_isosceles B F C B) 
(h_eq : is_equilateral D E F) :
∃ E F : α, is_equilateral D E F :=
sorry

end parallelogram_isosceles_equilateral_l30_30399


namespace diagonal_length_l30_30651

noncomputable def rectangle_diagonal (p : ℝ) (r : ℝ) (d : ℝ) : Prop :=
  ∃ k : ℝ, p = 2 * ((5 * k) + (2 * k)) ∧ r = 5 / 2 ∧ 
           d = Real.sqrt (((5 * k)^2 + (2 * k)^2)) 

theorem diagonal_length 
  (p : ℝ) (r : ℝ) (d : ℝ)
  (h₁ : p = 72) 
  (h₂ : r = 5 / 2)
  : rectangle_diagonal p r d ↔ d = 194 / 7 := 
sorry

end diagonal_length_l30_30651


namespace move_point_right_l30_30378

theorem move_point_right (x y : ℝ) (h₁ : x = 1) (h₂ : y = 1) (dx : ℝ) (h₃ : dx = 2) : (x + dx, y) = (3, 1) :=
by
  rw [h₁, h₂, h₃]
  simp
  sorry

end move_point_right_l30_30378


namespace proof_part1_proof_part2_l30_30802

noncomputable def part1 : Prop :=
  let mu : ℝ := 10
  let sigma : ℝ := 0.5
  let n : ℕ := 15
  let p_qualified : ℝ := 0.9973
  let p_all_qualified : ℝ := p_qualified ^ n
  let p_at_least_one_defective : ℝ := 1 - p_all_qualified
  p_at_least_one_defective = 0.0397

noncomputable def part2 : Prop :=
  let n : ℕ := 100
  let p : ℝ := 0.0027
  let k : ℕ := 0
  k = 0

theorem proof_part1 : part1 := by
  sorry

theorem proof_part2 : part2 := by
  sorry

end proof_part1_proof_part2_l30_30802


namespace find_n_l30_30575

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 1) / 2
noncomputable def T_n (n : ℕ) : ℕ := 2^n - 1
noncomputable def T_sum (n : ℕ) : ℕ := 2^(n+1) - n - 2
noncomputable def a_n (n : ℕ) : ℕ := n
noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

theorem find_n :
  ∃ (n : ℕ), S_n n + T_sum n = a_n n + 4 * b_n n ∧ n > 0 :=
begin
  -- We aim to show that the provided values lead to n = 4.
  use 4,
  split,
  { -- proving the equation holds for n = 4
    sorry },
  { -- proving n = 4 is a positive integer
    trivial, }
end

end find_n_l30_30575


namespace number_of_distinct_prime_factors_of_B_l30_30435

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30435


namespace prime_factors_of_B_l30_30529

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30529


namespace intervals_of_increase_min_max_values_l30_30902

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x + cos x)

theorem intervals_of_increase (k : ℤ) :
  ∃ I : set ℝ, (∀ x ∈ I, k * π - 3 * π / 8 ≤ x ∧ x ≤ k * π + π / 8) ∧
  (∀ x1 x2 ∈ I, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

theorem min_max_values :
  ∃ min_val max_val, min_val = 0 ∧ max_val = sqrt 2 + 1 ∧
    (∀ x ∈ set.Icc (-π/4) (π/4), f x ≥ min_val ∧ f x ≤ max_val) :=
sorry

end intervals_of_increase_min_max_values_l30_30902


namespace pyramid_volume_change_l30_30814

theorem pyramid_volume_change (s h : ℝ) (V : ℝ) (hsq_h : s^2 * h = 180) :
  let new_s := 3 * s,
      new_h := 2 * h,
      new_V := (1/3) * (new_s^2) * new_h in
  new_V = 1080 :=
by
  sorry

end pyramid_volume_change_l30_30814


namespace equilateral_equiangular_parallelogram_is_square_l30_30151

theorem equilateral_equiangular_parallelogram_is_square
  (P : Type) [parallelogram P] [equilateral P] [equiangular P] : square P :=
by
  sorry

end equilateral_equiangular_parallelogram_is_square_l30_30151


namespace hide_and_seek_l30_30784

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30784


namespace no_outliers_in_data_set_l30_30629

theorem no_outliers_in_data_set :
  let data_set := [7, 22, 31, 31, 42, 44, 44, 47, 55, 60]
  let Q1 := 31
  let Q3 := 47
  let IQR := Q3 - Q1
  let lower_threshold := Q1 - 1.5 * IQR
  let upper_threshold := Q3 + 1.5 * IQR
  (∀ x ∈ data_set, x ≥ lower_threshold ∧ x ≤ upper_threshold) →
  ([x ∈ data_set | x < lower_threshold ∨ x > upper_threshold].length = 0) := by
  sorry

end no_outliers_in_data_set_l30_30629


namespace increasing_function_odd_function_l30_30942

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem increasing_function (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
sorry

theorem odd_function (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) ↔ a = 1 :=
sorry

end increasing_function_odd_function_l30_30942


namespace part1_part2_l30_30300

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) := {x : ℝ | -3 * m + 4 ≤ x ∧ x ≤ 2 * m - 1}

def p (m : ℝ) := ∀ x : ℝ, x ∈ A → x ∈ B m
def q (m : ℝ) := ∃ x : ℝ, x ∈ B m ∧ x ∈ A

theorem part1 (m : ℝ) : p m → m ≥ 4 := by
  sorry

theorem part2 (m : ℝ) : q m → m ≥ 3/2 := by
  sorry

end part1_part2_l30_30300


namespace count_of_sequence_l30_30351

-- Definitions based on the conditions
def first_term : ℤ := 165
def common_difference : ℤ := -5
def last_term : ℤ := 40

-- Proof statement
theorem count_of_sequence : 
  ∃ n : ℕ, (seq = list.range (n + 1).map (λ i, first_term + i * common_difference)) → n + 1 = 26 :=
sorry

end count_of_sequence_l30_30351


namespace number_of_sides_of_polygon_l30_30019

theorem number_of_sides_of_polygon :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 2 * n + 7 ∧ n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l30_30019


namespace order_of_abc_l30_30313

def a : ℝ := Real.log 3 / Real.log (1/2)
def b : ℝ := (1/3 : ℝ) ^ 0.2
def c : ℝ := (2) ^ (1/3 : ℝ)

theorem order_of_abc : a < b ∧ b < c := by
  have ha : a < 0 := sorry
  have hb : 0 < b ∧ b < 1 := sorry
  have hc : 1 < c := sorry
  exact ⟨ha, hb.2.trans hc⟩

end order_of_abc_l30_30313


namespace center_of_circle_coordinates_l30_30291

noncomputable def center_of_circle_polar (rho theta : ℝ) : Prop :=
  rho^2 = 2 * rho * (Math.cos (theta + (Real.pi / 4)))

theorem center_of_circle_coordinates :
  (center_of_circle_polar 1 (-Real.pi / 4)) :=
by
  -- statement and proof related content
  sorry

end center_of_circle_coordinates_l30_30291


namespace dvd_rent_cost_l30_30594

theorem dvd_rent_cost (n : ℕ) (p d t : ℝ) (h_n : n = 4) (h_p : p = 4.80) (h_d : d = 0.10) (h_t : t = 0.07) : 
  let discount := p * d,
      discounted_price := p - discount,
      sales_tax := discounted_price * t,
      total_price := discounted_price + sales_tax,
      price_per_dvd := total_price / n in
  Real.round(price_per_dvd * 100) / 100 = 1.16 :=
by sorry

end dvd_rent_cost_l30_30594


namespace actual_time_before_storm_is_18_18_l30_30599

theorem actual_time_before_storm_is_18_18 :
  ∃ h m : ℕ, (h = 18) ∧ (m = 18) ∧ 
            ((09 = (if h == 0 then 1 else h - 1) ∨ 09 = (if h == 23 then 0 else h + 1)) ∧ 
             (09 = (if m == 0 then 1 else m - 1) ∨ 09 = (if m == 59 then 0 else m + 1))) := 
  sorry

end actual_time_before_storm_is_18_18_l30_30599


namespace distinct_prime_factors_of_product_of_divisors_l30_30512

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30512


namespace Ark5_ensures_metabolic_energy_l30_30014

variables (Ark5_inhibited : Prop)
variables (energy_scarcity : Prop)
variables (cancer_cells_proliferate_without_limit : Prop)
variables (cancer_cells_die_due_to_energy_lack : Prop)

-- Define the conditions based on the problem description
def inhibiting_Ark5_disrupts_balance (Ark5_inhibited : Prop) : Prop :=
  Ark5_inhibited → cancer_cells_proliferate_without_limit ∧ cancer_cells_die_due_to_energy_lack

def energy_scarcity_causes_metabolic_failure (energy_scarcity : Prop) : Prop :=
  energy_scarcity → cancer_cells_proliferate_without_limit

-- The goal to prove:
theorem Ark5_ensures_metabolic_energy (Ark5_inhibited : Prop) (energy_scarcity : Prop) :
  inhibiting_Ark5_disrupts_balance Ark5_inhibited ∧ 
  energy_scarcity_causes_metabolic_failure energy_scarcity → 
  Ark5_ensures_metabolic_energy Ark5 :=
sorry

end Ark5_ensures_metabolic_energy_l30_30014


namespace correct_intervals_valid_range_b_l30_30938

noncomputable def f (x : ℝ) : ℝ := (2 / x) + Real.log x - 2

def monotonicity_intervals : Prop :=
  (∀ x, 2 < x → (0 < (f x - f 2))) ∧ (∀ x, 0 < x ∧ x < 2 → (f x < f 2))

def range_b (b : ℝ) : Prop :=
  let fa := λ x : ℝ, (2 / x) + Real.log x - 2 + x - b in
  (fa (1 / Real.exp 1) ≥ 0) ∧ (fa (Real.exp 1) ≥ 0) ∧ (fa 1 < 0) →
  (1 < b ∧ b ≤ (2 / Real.exp 1 + Real.exp 1 - 1))

theorem correct_intervals :
  monotonicity_intervals :=
by
  sorry

theorem valid_range_b (b : ℝ) (h : ∀ x : ℝ, g x = x → ¬((1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (∃ c ≠ x, c ≠ 1))) :
  range_b b :=
by
  sorry

end correct_intervals_valid_range_b_l30_30938


namespace expected_number_of_ones_l30_30065

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30065


namespace division_quotient_correct_l30_30880

-- Define the polynomials involved
def dividend : ℤ[X] := 3 * X^4 - 5 * X^3 + 6 * X^2 - 8 * X + 3
def divisor : ℤ[X] := X^2 + X + 1
def expected_quotient : ℤ[X] := 3 * X^2 - 8 * X

-- Main statement: Prove that the quotient of dividend by divisor is the expected_quotient
theorem division_quotient_correct :
  polynomial.div dividend divisor = expected_quotient :=
by
  sorry

end division_quotient_correct_l30_30880


namespace expected_ones_three_dice_l30_30081

-- Define the scenario: rolling three standard dice
def roll_three_dice : List (Set (Fin 6)) :=
  [classical.decorated_of Fin.mk, classical.decorated_of Fin.mk, classical.decorated_of Fin.mk]

-- Define the event of rolling a '1'
def event_one (die : Set (Fin 6)) : Event (Fin 6) :=
  die = { Fin.of_nat 1 }

-- Probability of the event 'rolling a 1' for each die
def probability_one : ℚ :=
  1 / 6

-- Expected number of 1's when three dice are rolled
def expected_number_of_ones : ℚ :=
  3 * probability_one

theorem expected_ones_three_dice (h1 : probability_one = 1 / 6) :
  expected_number_of_ones = 1 / 2 :=
by
  have h1: probability_one = 1 / 6 := sorry 
  calc
    expected_number_of_ones
        = 3 * 1 / 6 : by rw [h1, expected_number_of_ones]
    ... = 1 / 2 : by norm_num

end expected_ones_three_dice_l30_30081


namespace problem1_problem2_l30_30197

-- First problem statement
theorem problem1 :
  log 8 + log 125 - (1 / 7) ^ (-2 : ℝ) + 16 ^ (3 / 4 : ℝ) + (real.sqrt 3 - 1) ^ 0 = -37 :=
sorry

-- Second problem statement
theorem problem2 :
  sin (25 * π / 6) + cos (25 * π / 3) + tan (-25 * π / 4) = 0 :=
sorry

end problem1_problem2_l30_30197


namespace expected_number_of_ones_on_three_dice_l30_30072

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30072


namespace expected_number_of_ones_l30_30067

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30067


namespace binary_to_decimal_l30_30853

theorem binary_to_decimal : 
  (0 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4) = 18 := 
by
  -- The proof is skipped
  sorry

end binary_to_decimal_l30_30853


namespace num_idempotent_functions_l30_30349

open Finset Function

theorem num_idempotent_functions :
  let n := 5
  let f_set := finset.fin_range n
  let count := ∑ k in f_set.Powerset, k.card.factorial * (n - k.card) ^ (n - k.card)
  count = 196 :=
by
  sorry

end num_idempotent_functions_l30_30349


namespace number_of_statements_implying_neg_statement_l30_30226

-- Define the statements as logical propositions
def statement1 (p q : Prop) : Prop := p ∨ q
def statement2 (p q : Prop) : Prop := p ∧ ¬ q
def statement3 (p q : Prop) : Prop := ¬ p ∨ q
def statement4 (p q : Prop) : Prop := ¬ p ∧ ¬ q

-- Define the negation of "p or q is true"
def neg_statement (p q : Prop) : Prop := ¬ (p ∨ q)

-- The problem to prove
theorem number_of_statements_implying_neg_statement (p q : Prop) :
  (if statement4 p q then 1 else 0) = 1 :=
by
  sorry

end number_of_statements_implying_neg_statement_l30_30226


namespace conjugate_in_fourth_quadrant_l30_30898

theorem conjugate_in_fourth_quadrant (z : ℂ) (h : z * (1 + complex.i) = complex.i) : 
    (z.conj.re > 0 ∧ z.conj.im < 0) :=
by
  sorry

end conjugate_in_fourth_quadrant_l30_30898


namespace bd_plus_da_equals_bc_l30_30402

noncomputable def triangle_abc (A B C : Type) := True

variables {A B C D : Type} 
variables (m : Type → Type) (angle_b angle_c : A → B)
variables (AC BC BD DA : A → ℝ) (intersects : A → B → C)
variables [triangle_abc A B C]

-- Conditions
def is_triangle (A B C : Type) : Prop := True

def angle_b_eq_40 (B : Type) : Prop := B = 40
def angle_c_eq_40 (C : Type) : Prop := C = 40

def angle_bisector_intersects (x D y : Type) : Prop := intersects x D y = True

-- Question
theorem bd_plus_da_equals_bc :
  ∀ {A B C D : Type}, 
    is_triangle A B C → 
    angle_b_eq_40 B → 
    angle_c_eq_40 C →
    angle_bisector_intersects A D C → 
    BD + DA = BC :=
begin
  intros,
  sorry
end

end bd_plus_da_equals_bc_l30_30402


namespace distinct_prime_factors_product_divisors_60_l30_30481

-- Definitions based on conditions identified in a)
def divisors (n : ℕ) : List ℕ := List.filter (λ d, n % d = 0) (List.range (n + 1))

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (Nat.Prime)

-- Given problem in Lean 4
theorem distinct_prime_factors_product_divisors_60 :
  let B := product (divisors 60)
  (prime_factors B).card = 3 := by
  sorry

end distinct_prime_factors_product_divisors_60_l30_30481


namespace joe_cars_after_getting_more_l30_30135

-- Defining the initial conditions as Lean variables
def initial_cars : ℕ := 50
def additional_cars : ℕ := 12

-- Stating the proof problem
theorem joe_cars_after_getting_more : initial_cars + additional_cars = 62 := by
  sorry

end joe_cars_after_getting_more_l30_30135


namespace expected_number_of_ones_l30_30070

theorem expected_number_of_ones (n : ℕ) (rolls : ℕ) (p : ℚ) (dice : ℕ) : expected_number_of_ones n rolls p dice = 1/2 :=
by
  -- n is the number of possible outcomes on a single die (6 for a standard die)
  have h_n : n = 6, from sorry,
  -- rolls is the number of dice being rolled
  have h_rolls : rolls = 3, from sorry,
  -- p is the probability of rolling a 1 on a single die
  have h_p : p = 1/6, from sorry,
  -- dice is the number of dice rolled
  have h_dice : dice = 3, from sorry,
  sorry

end expected_number_of_ones_l30_30070


namespace distinct_prime_factors_of_product_of_divisors_l30_30519

theorem distinct_prime_factors_of_product_of_divisors :
  let B := ∏ d in (finset.filter (λ n, 60 % n = 0) (finset.range 61)), d
  nat.factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30519


namespace problem_b_l30_30949

theorem problem_b (r : ℝ) (h : r > sqrt 2) :
  ∃ (l : Line) (A B C D : Point) (M N : Curve),
    M = { p | p.2^2 = 2*p.1 } ∧
    N = { p | (p.1-1)^2 + p.2^2 = r^2 } ∧
    l.contains (1, 0) ∧
    IntersectionPoints l N = [C, D] ∧
    IntersectionPoints l M = [A, B] ∧
    |distance A C| = |distance B D| ∧
    exactThreeLines := ∃ l1 l2 l3, l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 :=
sorry

end problem_b_l30_30949


namespace parallelogram_area_15_l30_30192

def point := (ℝ × ℝ)

def base_length (p1 p2 : point) : ℝ :=
  abs (p2.1 - p1.1)

def height_length (p3 p4 : point) : ℝ :=
  abs (p3.2 - p4.2)

def parallelogram_area (p1 p2 p3 p4 : point) : ℝ :=
  base_length p1 p2 * height_length p1 p3

theorem parallelogram_area_15 :
  parallelogram_area (0, 0) (3, 0) (1, 5) (4, 5) = 15 := by
  sorry

end parallelogram_area_15_l30_30192


namespace rectangle_diagonal_length_l30_30706

theorem rectangle_diagonal_length
  (a b : ℝ)
  (h1 : a = 40 * Real.sqrt 2)
  (h2 : b = 2 * a) :
  Real.sqrt (a^2 + b^2) = 160 := by
  sorry

end rectangle_diagonal_length_l30_30706


namespace product_of_divisors_has_three_prime_factors_l30_30463

theorem product_of_divisors_has_three_prime_factors :
  let B := ∏ d in (finset.filter (λ x, x ∣ 60) (finset.range 61)), d in
  (factors B).to_finset.card = 3 := sorry

end product_of_divisors_has_three_prime_factors_l30_30463


namespace range_of_a_l30_30283

noncomputable
def isDecreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x₁ x₂ ∈ I, x₁ < x₂ → f x₁ > f x₂

noncomputable
def p (a : ℝ) : Prop :=
a > 0 ∧ a < 1 ∧ isDecreasing (λ x => Real.log (x + 1)) (set.Ioi 0)

def discriminant (a : ℝ) : ℝ :=
(2 * a - 3) ^ 2 - 4

noncomputable
def q (a : ℝ) : Prop :=
a > 0 ∧ discriminant a > 0

noncomputable
def pq_condition (a : ℝ) : Prop :=
(p a ∨ q a) ∧ ¬(p a ∧ q a)

noncomputable
def validRange (a : ℝ) : Prop :=
(1 / 2 ≤ a ∧ a < 1) ∨ (a > 5 / 2)

theorem range_of_a (a : ℝ) :
  a > 0 → a ≠ 1 → pq_condition a → validRange a :=
sorry

end range_of_a_l30_30283


namespace problem_solution_l30_30591

theorem problem_solution 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)
  (h₁ : a = ⌊2 + Real.sqrt 2⌋) 
  (h₂ : b = (2 + Real.sqrt 2) - ⌊2 + Real.sqrt 2⌋)
  (h₃ : c = ⌊4 - Real.sqrt 2⌋)
  (h₄ : d = (4 - Real.sqrt 2) - ⌊4 - Real.sqrt 2⌋) :
  (b + d) / (a * c) = 1 / 6 :=
by
  sorry

end problem_solution_l30_30591


namespace tangents_and_triangle_area_l30_30936

theorem tangents_and_triangle_area
  (A : ℝ × ℝ) (C : ℝ × ℝ → Prop) (O : ℝ × ℝ) :
  A = (3, 5) →
  C = (λ (p : ℝ × ℝ), (p.1 - 2)^2 + (p.2 - 3)^2 = 1) →
  O = (0, 0) →
  (∀ p : ℝ × ℝ, p ∈ C → tangent_line p A = 3 * p.1 - 4 * p.2 + 11 = 0 ∨ p.1 = 3) →
  (triangle_area O A (2, 3) = 1 / 2) :=
by
sorry

noncomputable def tangent_line := sorry

noncomputable def triangle_area : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

end tangents_and_triangle_area_l30_30936


namespace hide_and_seek_problem_l30_30740

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30740


namespace number_of_distinct_prime_factors_of_B_l30_30433

theorem number_of_distinct_prime_factors_of_B
  (B : ℕ)
  (hB : B = 1 * 2 * 3 * 4 * 5 * 6 * 10 * 12 * 15 * 20 * 30 * 60) : 
  (3 : ℕ) := by
  -- Proof goes here
  sorry

end number_of_distinct_prime_factors_of_B_l30_30433


namespace train_length_l30_30824

theorem train_length (speed_kmh : ℕ) (cross_time : ℕ) (h_speed : speed_kmh = 54) (h_time : cross_time = 9) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_m := speed_ms * cross_time
  length_m = 135 := by
  sorry

end train_length_l30_30824


namespace parabola_points_relation_l30_30921

theorem parabola_points_relation :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = -(-2)^2 - 2*(-2) + 2) ∧ 
  (y2 = -(1)^2 - 2*(1) + 2) ∧ 
  (y3 = -(2)^2 - 2*(2) + 2) → 
  y1 > y2 ∧ y2 > y3 :=
by {
  intros y1 y2 y3 h,
  obtain ⟨h1, h2, h3⟩ := h,
  rw [h1, h2, h3],
  -- This is the placeholder for the proof
  sorry
}

end parabola_points_relation_l30_30921


namespace expected_value_of_ones_on_three_dice_l30_30052

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30052


namespace distinct_prime_factors_B_l30_30552

def num_divisors_60 := ∏ d in (multiset.to_finset {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}).toFinset.val, d
def B := num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60 * num_divisors_60
def factorization (n : ℕ) := multiset.nodup (int.factors n)
noncomputable def prime_divisors_B := {p ∈ int.factors B | prime p}

theorem distinct_prime_factors_B : multiset.card prime_divisors_B = 3 := sorry

end distinct_prime_factors_B_l30_30552


namespace sequence_sum_equals_l30_30872

-- Define the cyclical properties of \(i\)
def i := Complex.I

-- Define the sequence sum
noncomputable def sequence_sum : Complex :=
  ∑ k in Finset.range 3003, (k + 1) * (i ^ (k + 1))

-- The proof statement
theorem sequence_sum_equals :
  sequence_sum = -1504 - 1500 * i := sorry

end sequence_sum_equals_l30_30872


namespace zero_in_interval_l30_30327

def f (x : ℝ) : ℝ := log x / log 2 + x - 4

theorem zero_in_interval :
  ∃ n : ℕ+, f (real.to_nnreal ((n : ℕ) + 1)) > 0 ∧ f (real.to_nnreal n) < 0 ∧ n = 2 :=
by
  sorry

end zero_in_interval_l30_30327


namespace fossil_age_unique_permutations_l30_30149

-- The digits 2, 2, 5, 5, 7, 9
def digits : List ℕ := [2, 2, 5, 5, 7, 9]

-- Predicate to check if a number is a prime
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

-- Count permutations of the age starting with a prime
noncomputable def count_permutations_starting_with_prime (ds : List ℕ) (hd_prime : is_prime ds.head) : ℕ :=
  let remaining_digits := ds.tail
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else List.prod (List.range (1 + n))
  let count (l : List ℕ) := factorial l.length / (factorial (l.count 2) * factorial (l.count 5))
  3 * count remaining_digits

theorem fossil_age_unique_permutations : count_permutations_starting_with_prime digits sorry = 90 := sorry

end fossil_age_unique_permutations_l30_30149


namespace range_of_x2_y2_l30_30919

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x^4

theorem range_of_x2_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x) : 
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27 / 16 :=
sorry

end range_of_x2_y2_l30_30919


namespace number_of_distinct_prime_factors_of_B_l30_30544

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30544


namespace perpendicular_condition_l30_30287

-- Declaration of lines and plane
variables {l m : Type} [line l] [line m] {α : Type} [plane α]

-- Given conditions
axiom different_lines : l ≠ m
axiom m_in_plane : m ∈ α

-- Theorem for proof statement
theorem perpendicular_condition :
  (l ⊥ m) ↔ (necessary_but_not_sufficient_condition_for (l ⊥ α)) :=
begin
  -- proof steps would go here
  sorry
end

end perpendicular_condition_l30_30287


namespace general_term_correct_exists_arithmetic_lambda_l30_30907

variable {ℕ : Type} [AddCommGroup ℕ] [MulAction ℝ ℕ]

-- Given sequence condition
def sequence_condition (a_n S_n : ℕ → ℝ) (n : ℕ) : Prop :=
  a_n n - (1/2) * S_n n - 1 = 0

-- Sum of first n terms
def sum_of_first_n_terms (a_n S_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (Σ i in finset.range (n+1), a_n i)

-- General term to prove
def general_term (a_n : ℕ → ℝ) :=
  ∀ n, a_n n = 2^n

-- Arithmetic sequence check
def is_arithmetic_seq (b_n : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b_n (n + 1) - b_n n = d

theorem general_term_correct (a_n S_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, sequence_condition a_n S_n n) →
  (sum_of_first_n_terms a_n S_n) →
  general_term a_n :=
by
  intro h_cond h_sum
  sorry

theorem exists_arithmetic_lambda (S_n : ℕ → ℝ) :
  ∃ λ, ∀ n, is_arithmetic_seq (λ n, S_n n + (n + 2^n) * λ) :=
by
  sorry

end general_term_correct_exists_arithmetic_lambda_l30_30907


namespace brenda_david_scrabble_game_l30_30230

theorem brenda_david_scrabble_game :
  let brenda_first_turn := 18,
      david_first_turn := 10,
      brenda_second_turn := 25,
      david_second_turn := 35,
      brenda_third_turn_play := 15,
      brenda_third_turn_lead := 22,
      david_third_turn := 32 in
  let brenda_total := brenda_first_turn + brenda_second_turn + brenda_third_turn_play in
  let david_total := david_first_turn + david_second_turn + david_third_turn in
  david_total - brenda_total = 19 := 
by
  sorry

end brenda_david_scrabble_game_l30_30230


namespace distinct_prime_factors_of_B_l30_30447

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30447


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30505

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30505


namespace highest_digit_a_divisible_by_eight_l30_30877

theorem highest_digit_a_divisible_by_eight :
  ∃ a : ℕ, a ≤ 9 ∧ 8 ∣ (100 * a + 16) ∧ ∀ b : ℕ, b > a → b ≤ 9 → ¬ (8 ∣ (100 * b + 16)) := by
  sorry

end highest_digit_a_divisible_by_eight_l30_30877


namespace inequality_solution_set_maximum_m_value_l30_30944

section problem_1
  def f (x : ℝ) := |x - 1| + |x - 2|

  theorem inequality_solution_set :
    {x : ℝ | f x ≥ 2} = {x : ℝ | x ≤ 1 / 2} ∪ {x : ℝ | x ≥ 5 / 2} :=
  by
    sorry
end problem_1

section problem_2
  def f (x : ℝ) := |x - 1| + |x - 2|

  theorem maximum_m_value :
    ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ -2 * x ^ 2 + m) ∧ (∀ (m' : ℝ), (∀ (x : ℝ), f x ≥ -2 * x ^ 2 + m') → m ≤ m') ∧ m = 5 / 2 :=
  by
    sorry
end problem_2

end inequality_solution_set_maximum_m_value_l30_30944


namespace valid_parameterizations_l30_30016

-- Define the parameterization as a structure
structure LineParameterization where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

-- Define the line equation
def line_eq (p : ℝ × ℝ) : Prop :=
  p.snd = -(2/3) * p.fst + 4

-- Proving which parameterizations are valid
theorem valid_parameterizations :
  (line_eq (3 + t * 3, 4 + t * (-2)) ∧
   line_eq (0 + t * 1.5, 4 + t * (-1)) ∧
   line_eq (1 + t * (-6), 3.33 + t * 4) ∧
   line_eq (5 + t * 1.5, (2/3) + t * (-1)) ∧
   line_eq (-6 + t * 9, 8 + t * (-6))) = 
  false ∧ true ∧ false ∧ true ∧ false :=
by
  sorry

end valid_parameterizations_l30_30016


namespace number_of_friends_is_five_l30_30724

def total_cards : ℕ := 455
def cards_per_friend : ℕ := 91

theorem number_of_friends_is_five (n : ℕ) (h : total_cards = n * cards_per_friend) : n = 5 := 
sorry

end number_of_friends_is_five_l30_30724


namespace find_annual_income_of_A_l30_30018

theorem find_annual_income_of_A (ratio_A_B_D : ℕ → ℕ → ℕ → Prop)
  (percentage_increase : ℕ → ℕ → ℕ → Prop)
  (monthly_income_C : ℕ)
  (percentage_decrease : ℕ → ℕ → ℕ → Prop)
  (annual_income : ℕ → ℕ)
  : ∃ A_a, 
  ratio_A_B_D 5 2 3 ∧ 
  percentage_increase 12000 12 13440 ∧ 
  percentage_decrease 33600 20 26880 ∧ 
  (∀ n, annual_income n = 12 * 33600) → 
  A_a = 403200 :=
begin
  let A_m := 5 * (13440 / 2),
  let A_a := A_m * 12,
  exact
    ⟨A_a, ratio_A_B_D 5 2 3, percentage_increase 12000 12 13440, percentage_decrease 33600 20 26880,
    λ n, by simp; sorry⟩
end

end find_annual_income_of_A_l30_30018


namespace emma_reaches_jack_after_33_minutes_l30_30392

-- Definitions from conditions
def distance_initial : ℝ := 30  -- 30 km apart initially
def combined_speed : ℝ := 2     -- combined speed is 2 km/min
def time_before_breakdown : ℝ := 6 -- Jack biked for 6 minutes before breaking down

-- Assume speeds
def v_J (v_E : ℝ) : ℝ := 2 * v_E  -- Jack's speed is twice Emma's speed

-- Assertion to prove
theorem emma_reaches_jack_after_33_minutes :
  ∀ v_E : ℝ, ((v_J v_E + v_E = combined_speed) → 
              (distance_initial - combined_speed * time_before_breakdown = 18) → 
              (v_E > 0) → 
              (time_before_breakdown + 18 / v_E = 33)) :=
by 
  intro v_E 
  intros h1 h2 h3 
  have h4 : v_J v_E = 2 * v_E := rfl
  sorry

end emma_reaches_jack_after_33_minutes_l30_30392


namespace trajectory_equation_of_point_M_l30_30155

variables {x y a b : ℝ}

theorem trajectory_equation_of_point_M :
  (a^2 + b^2 = 100) →
  (x = a / (1 + 4)) →
  (y = 4 * b / (1 + 4)) →
  16 * x^2 + y^2 = 64 :=
by
  intros h1 h2 h3
  sorry

end trajectory_equation_of_point_M_l30_30155


namespace solution_set_l30_30582

def f (x : ℝ) : ℝ := x * (x^2 - Real.cos (x / 3) + 2)

theorem solution_set (x : ℝ) (h1 : -3 < x) (h2 : x < 3) :
  f (1 + x) + f 2 < f (1 - x) ↔ -2 < x ∧ x < -1 :=
sorry

end solution_set_l30_30582


namespace max_d_value_l30_30269

def a (n : ℕ) : ℕ := n^2 + 100

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ n : ℕ, ∀ m : ℕ, d n ≤ d m := 401 :=
by 
  sorry

end max_d_value_l30_30269


namespace smallest_palindrome_in_base2_and_4_l30_30223

-- Define a function to check if a number's representation is a palindrome.
def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

-- Convert a number n to a given base and represent as a list of digits.
def to_base (n : ℕ) (b : ℕ) : List ℕ :=
  if h : b > 1 then
    let rec convert (n : ℕ) : List ℕ :=
      if n = 0 then [] else (n % b) :: convert (n / b)
    convert n 
  else []

-- Convert the list of digits to a list of characters.
def digits_to_chars (ds : List ℕ) : List Char :=
  ds.map (λ d => (Char.ofNat (d + 48))) -- Adjust ASCII value for digit representation

-- Define a function to check if a number is a palindrome in a specified base.
def is_palindromic_in_base (n base : ℕ) : Bool :=
  is_palindrome (digits_to_chars (to_base n base))

-- Define the main claim.
theorem smallest_palindrome_in_base2_and_4 : ∃ n, n > 10 ∧ is_palindromic_in_base n 2 ∧ is_palindromic_in_base n 4 ∧ ∀ m, m > 10 ∧ is_palindromic_in_base m 2 ∧ is_palindromic_in_base m 4 -> n ≤ m :=
by
  exists 15
  sorry

end smallest_palindrome_in_base2_and_4_l30_30223


namespace third_root_of_cubic_l30_30097

theorem third_root_of_cubic (a b : ℚ) (h1 : a ≠ 0) 
  (h2 : eval (-2 : ℚ) (a * X^3 + (a + 2 * b) * X^2 + (b - 3 * a) * X + (8 - a)) = 0)
  (h3 : eval (3 : ℚ) (a * X^3 + (a + 2 * b) * X^2 + (b - 3 * a) * X + (8 - a)) = 0) 
  : a * (4 / 3) ^ 3 + (a + 2 * b) * (4 / 3) ^ 2 + (b - 3 * a) * (4 / 3) + (8 - a) = 0 :=
by
  sorry

end third_root_of_cubic_l30_30097


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30566

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30566


namespace largest_n_divisible_by_n_plus_12_l30_30859

theorem largest_n_divisible_by_n_plus_12 :
  ∃ n ∈ ℕ, (n + 12) ∣ (n^3 + 144) ∧ (∀ m ∈ ℕ, (m + 12) ∣ (m^3 + 144) → m ≤ 780) ∧ n = 780 :=
begin
  sorry
end

end largest_n_divisible_by_n_plus_12_l30_30859


namespace range_of_g_l30_30881

theorem range_of_g (x : ℝ) : 
  let u := sin x
      v := cos x
      g := u^4 - u^3 * v + v^4
  in (∀u v, u^2 + v^2 = 1 → 0.316 <= g ∧ g <= 1) :=
by
  let u := sin x
  let v := cos x
  let g := u^4 - u^3 * v + v^4
  have h1 : ∀u v, u^2 + v^2 = 1 → g = 1 - 2 * u^2 * v^2 - u^3 * v :=
    sorry
  have h2 : ∀u v, u^2 + v^2 = 1 → 0.316 <= g :=
    sorry
  have h3 : ∀u v, u^2 + v^2 = 1 → g <= 1 :=
    sorry
  split
  exact h1
  split
  exact h2
  exact h3

end range_of_g_l30_30881


namespace monotonic_increasing_interval_l30_30635

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  sqrt 3 * sin (ω * x) - cos (ω * x)

theorem monotonic_increasing_interval (ω : ℝ) (k : ℤ) (x : ℝ) 
  (h_omega_pos : ω > 0) (h_omega_eq_1 : ω = 1) :
  (2 * k * π - π / 3) ≤ x ∧ x ≤ (2 * k * π + 2 * π / 3) → monotonic_increasing_on (f x ω) :=
begin
  sorry
end

end monotonic_increasing_interval_l30_30635


namespace parallel_vectors_l30_30970

theorem parallel_vectors {x y : ℝ} (a b : ℝ × ℝ × ℝ) 
  (ha : a = (2 * x, 1, 3)) (hb : b = (1, -2 * y, 9)) 
  (h_parallel : ∃ k : ℝ, a = (k * (1, -2 * y, 9)) ∨ b = (k * (2 * x, 1, 3))) : 
  x * y = -1 / 4 := 
by 
  sorry

end parallel_vectors_l30_30970


namespace hide_and_seek_l30_30785

theorem hide_and_seek
  (A B V G D : Prop)
  (h1 : A → (B ∧ ¬V))
  (h2 : B → (G ∨ D))
  (h3 : ¬V → (¬B ∧ ¬D))
  (h4 : ¬A → (B ∧ ¬G)) :
  (B ∧ V ∧ D) :=
by
  sorry

end hide_and_seek_l30_30785


namespace expected_ones_three_standard_dice_l30_30055

noncomputable def expected_num_ones (dice_faces : ℕ) (num_rolls : ℕ) : ℚ := 
  let p_one := 1 / dice_faces
  let p_not_one := (dice_faces - 1) / dice_faces
  let zero_one_prob := p_not_one ^ num_rolls
  let one_one_prob := num_rolls * p_one * p_not_one ^ (num_rolls - 1)
  let two_one_prob := (num_rolls * (num_rolls - 1) / 2) * p_one ^ 2 * p_not_one ^ (num_rolls - 2)
  let three_one_prob := p_one ^ 3
  0 * zero_one_prob + 1 * one_one_prob + 2 * two_one_prob + 3 * three_one_prob

theorem expected_ones_three_standard_dice : expected_num_ones 6 3 = 1 / 2 := 
  sorry

end expected_ones_three_standard_dice_l30_30055


namespace angle_between_a_and_b_l30_30289

noncomputable def angle_between_vectors (a b : Euclidean_Space ℝ 2) : ℝ := 
  real.acos ((inner a b) / (∥a∥ * ∥b∥))

noncomputable def vector_magnitude (v : Euclidean_Space ℝ 2) : ℝ := ∥v∥

theorem angle_between_a_and_b
  (a b : Euclidean_Space ℝ 2)
  (ha : vector_magnitude a = 1)
  (hb : vector_magnitude b = 2)
  (h_perp : inner (a + b) a = 0) : 
  angle_between_vectors a b = real.pi * 2 / 3 :=
sorry

end angle_between_a_and_b_l30_30289


namespace number_of_distinct_prime_factors_of_B_l30_30545

theorem number_of_distinct_prime_factors_of_B :
  let B := ∏ d in (finset.filter (λ d, d ∣ 60) (finset.range (60 + 1))), d
  (nat.factors (60 ^ (number_of_divisors 60))).to_finset.card = 3 := by
sorry

end number_of_distinct_prime_factors_of_B_l30_30545


namespace club_additional_members_l30_30799

theorem club_additional_members (current_members additional_members future_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 15) 
  (h3 : future_members = current_members + additional_members) : 
  future_members - current_members = 15 :=
by
  sorry

end club_additional_members_l30_30799


namespace ben_points_l30_30128

theorem ben_points (B : ℕ) 
  (h1 : 42 = B + 21) : B = 21 := 
by 
-- Proof can be filled in here
sorry

end ben_points_l30_30128


namespace find_room_height_l30_30255

theorem find_room_height (l b d : ℕ) (h : ℕ) (hl : l = 12) (hb : b = 8) (hd : d = 17) :
  d = Int.sqrt (l^2 + b^2 + h^2) → h = 9 :=
by
  sorry

end find_room_height_l30_30255


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30570

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30570


namespace cyclist_speeds_l30_30670

variables {a b t : ℝ}

theorem cyclist_speeds (hx : 0 < t) : 
  ∃ x, 
    x = (4 * b - 3 * a * t + sqrt (9 * t^2 * a^2 + 16 * b^2)) / (6 * t) ∧
    (x + a) = (4 * b + 3 * a * t + sqrt (9 * t^2 * a^2 + 16 * b^2)) / (6 * t) :=
by
  sorry

end cyclist_speeds_l30_30670


namespace rope_speed_ratio_l30_30121

theorem rope_speed_ratio (r : ℝ) : 
  let v1 := (2 * π * r) / 0.5 in
  let v2 := (4 * π * r) / 0.6 in
  v1 / v2 = 3 / 5 :=
by {
  have v1_def : v1 = 4 * π * r, by {
    calc
      v1 = (2 * π * r) / 0.5 : rfl
         ... = 4 * π * r       : by linarith,
  },
  have v2_def : v2 = (20 * π * r) / 3, by {
    calc
      v2 = (4 * π * r) / 0.6   : rfl
         ... = (4 * π * r) * (10 / 6) : by rw [division_def, mul_div_assoc _ (4 * π * r) (10 : ℝ) 6]
         ... = (20 * π * r) / 3 : by ring,
  },
  rw [v1_def, v2_def],
  calc
    (4 * π * r) / ((20 * π * r) / 3) = 4 * π * r * (3 / (20 * π * r)) : by rw div_mul_div
                                   ... = (4 * 3) / 20                : by field_simp
                                   ... = 12 / 20                     : by ring
                                   ... = 3 / 5                       : by norm_num,
}

end rope_speed_ratio_l30_30121


namespace equation_of_curve_area_of_triangle_constant_lambda_exists_l30_30384

section Geometry

-- Define the points A, E, F, and the condition for moving point P
variables {A E F P Q : ℝ × ℝ}
def point_A := (-1, 0)
def point_E := (-2, 0)
def point_F := (2, 0)

-- Define the condition for P lying on the hyperbola Gamma
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 - y^2 / 3 = 1 ∧ x > 0

-- Define the condition for Q and its relation to P
def related_points (P Q : ℝ × ℝ) : Prop :=
  let (Px, Py) := P in let (Qx, Qy) := Q in 
  (Px, Py) = (8 - 3 * Qx, -3 * Qy)

-- 1. Prove the equation of the curve Gamma
theorem equation_of_curve :
  on_hyperbola P ↔ let (x, y) := P in x^2 - y^2 / 3 = 1 ∧ x > 0 := sorry

-- 2. Prove the area of triangle APQ given the vector condition
theorem area_of_triangle :
  on_hyperbola P → on_hyperbola Q → related_points P Q → 
  let (Px, Py) := P in let (Qx, Qy) := Q in let (Ax, Ay) := A in
  abs ((Ax * (Py - Qy) + Px * (Qy - Ay) + Qx * (Ay - Py)) / 2) = 3 * real.sqrt 15 := sorry

-- 3. Prove the existence of a constant lambda for the given angle condition
theorem constant_lambda_exists :
  (∃ λ : ℝ, ∀ P, on_hyperbola P → let (Px, Py) := P in
  let PF := (Px - point_F.1, Py - point_F.2) in let PA := (Px - point_A.1, Py - point_A.2) in
  ∠ PF = λ * ∠ PA) :=
  (∃ λ = 2 ∀ P, on_hyperbola P → let (Px, Py) := P in
  let PF := (Px - point_F.1, Py - point_F.2) in let PA := (Px - point_A.1, Py - point_A.2) in
  ∠ PF = 2 * ∠ PA) := sorry

end Geometry

end equation_of_curve_area_of_triangle_constant_lambda_exists_l30_30384


namespace expected_number_of_ones_when_three_dice_rolled_l30_30033

noncomputable def expected_number_of_ones : ℚ :=
  let num_dice := 3
  let prob_not_one := (5 : ℚ) / 6
  let prob_one := (1 : ℚ) / 6
  let prob_zero_ones := prob_not_one^num_dice
  let prob_one_one := (num_dice.choose 1) * prob_one * prob_not_one^(num_dice - 1)
  let prob_two_ones := (num_dice.choose 2) * (prob_one^2) * prob_not_one^(num_dice - 2)
  let prob_three_ones := (num_dice.choose 3) * (prob_one^3)
  let expected_value := (0 * prob_zero_ones + 
                         1 * prob_one_one + 
                         2 * prob_two_ones + 
                         3 * prob_three_ones)
  expected_value

theorem expected_number_of_ones_when_three_dice_rolled :
  expected_number_of_ones = (1 : ℚ) / 2 := by
  sorry

end expected_number_of_ones_when_three_dice_rolled_l30_30033


namespace vacuum_ratio_l30_30391

theorem vacuum_ratio (x : ℕ) (time_upstairs time_downstairs total_time : ℕ) :
  time_upstairs = 27 ∧ time_downstairs = x ∧ total_time = 38 ∧ 
  time_upstairs = time_downstairs + 5 ∧ 
  time_upstairs + time_downstairs = total_time → 
  time_upstairs.toRat / time_downstairs.toRat = 27 / 22 :=
by
  intro h
  sorry

end vacuum_ratio_l30_30391


namespace intersect_with_at_least_one_l30_30274

-- Let's define the necessary concepts first.
variables {α β ℓ : Type}
variables [Plane α] [Plane β]
variables (a b : Line) (la : a ∈ α) (lb : b ∈ β)

-- Say that α and β intersect in a line ℓ 
axiom intersect_planes : α ∩ β = ℓ

-- a and b are skew lines
axiom skew_lines : ∀ a b, skew a b

-- Define a proof for the intersection with at least one of line a or b.
theorem intersect_with_at_least_one (h1 : ∀ (la : a ∈ α), ∀ (lb : b ∈ β), skew a b) (h2 : α ∩ β = ℓ) :
  (a ∩ ℓ ≠ ∅ ∨ b ∩ ℓ ≠ ∅) :=
sorry

end intersect_with_at_least_one_l30_30274


namespace most_suitable_for_census_l30_30829

def Survey :=
  | A
  | B
  | C
  | D

def is_census_suitable (s: Survey) : Prop :=
  match s with
  | Survey.A => False
  | Survey.B => True
  | Survey.C => False
  | Survey.D => False

theorem most_suitable_for_census : ∀ s, is_census_suitable s ↔ s = Survey.B :=
by
  intro s
  cases s
  repeat
  sorry

end most_suitable_for_census_l30_30829


namespace hide_and_seek_friends_l30_30757

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30757


namespace greatest_prime_factor_of_f_36_l30_30729

def f (m : ℕ) : ℕ := ∏ i in (finset.filter (λ n, n % 2 = 0) (finset.range (m + 1))), i

theorem greatest_prime_factor_of_f_36 : (Nat.greatestPrimeFactor (f 36)) = 17 := 
by 
  -- Prove directly that the greatest prime factor of f(36) is 17
  sorry

end greatest_prime_factor_of_f_36_l30_30729


namespace product_of_divisors_of_60_has_3_prime_factors_l30_30559

theorem product_of_divisors_of_60_has_3_prime_factors :
  let B := ∏ d in (finset.divisors 60), d
  in ∏ p in (finset.prime_divisors B), p = 3 :=
by 
  -- conditions and steps will be handled here if proof is needed
  sorry

end product_of_divisors_of_60_has_3_prime_factors_l30_30559


namespace teachers_neither_condition_l30_30818

open Finset

theorem teachers_neither_condition (T : Finset ℕ) (A B : Finset ℕ)
  (hT : |T| = 150) (hA : |A| = 80) (hB : |B| = 50) (hAB : |A ∩ B| = 30) :
  ((|T \ (A ∪ B)| : ℚ) / |T| * 100) = 33.33 :=
by sorry

end teachers_neither_condition_l30_30818


namespace tetrahedron_volume_l30_30365

theorem tetrahedron_volume 
  (R S₁ S₂ S₃ S₄ : ℝ) : 
  V = R * (S₁ + S₂ + S₃ + S₄) :=
sorry

end tetrahedron_volume_l30_30365


namespace ladybugs_with_spots_l30_30620

theorem ladybugs_with_spots (total_ladybugs : ℕ) (ladybugs_without_spots : ℕ) : total_ladybugs = 67082 ∧ ladybugs_without_spots = 54912 → total_ladybugs - ladybugs_without_spots = 12170 := by
  sorry

end ladybugs_with_spots_l30_30620


namespace expected_value_of_ones_on_three_dice_l30_30048

theorem expected_value_of_ones_on_three_dice : 
  (∑ i in (finset.range 4), i * ( nat.choose 3 i * (1 / 6 : ℚ) ^ i * (5 / 6 : ℚ) ^ (3 - i) )) = 1 / 2 :=
sorry

end expected_value_of_ones_on_three_dice_l30_30048


namespace set_intersection_eq_l30_30340

universe u
noncomputable theory

variable {α : Type u}
variables (U A B : Set α)

def complement (univ : Set α) (s : Set α) := univ \ s

theorem set_intersection_eq {U : Set ℕ} {A : Set ℕ} {B : Set ℕ} 
  (hU : U = {1, 2, 3, 4, 5}) 
  (hA : A = {1, 3, 4}) 
  (hB : B = {2, 3}) : 
  (complement U A) ∩ B = {2} := 
  by sorry

end set_intersection_eq_l30_30340


namespace scale_division_l30_30164

theorem scale_division (ft_inch : ℕ) (inch : ℕ) (scale_parts : ℕ) (foot_to_inch : ℕ) :
  ft_inch = 7 → inch = 12 → scale_parts = 4 → foot_to_inch = 12 →
  ((ft_inch * foot_to_inch + inch) / scale_parts) = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end scale_division_l30_30164


namespace domain_f_shifted_correct_l30_30316

variable {α : Type} [OrderedField α]

def domain_f (x : α) : Prop := 0 < x ∧ x < 1
def domain_f_shifted (x : α) : Prop := -1 < x ∧ x < 0

theorem domain_f_shifted_correct :
  (∀ x, domain_f (x + 1)) ↔ (∀ x, domain_f_shifted x) :=
by
  sorry

end domain_f_shifted_correct_l30_30316


namespace alex_play_friends_with_l30_30734

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l30_30734


namespace distinct_prime_factors_of_B_l30_30446

theorem distinct_prime_factors_of_B :
  let B := ∏ d in (Finset.filter (λ x => x ∣ 60) (Finset.range 61)), d
  nat.prime_factors B = {2, 3, 5} :=
by {
  sorry
}

end distinct_prime_factors_of_B_l30_30446


namespace find_Luisa_books_l30_30595

structure Books where
  Maddie : ℕ
  Amy : ℕ
  Amy_and_Luisa : ℕ
  Luisa : ℕ

theorem find_Luisa_books (L M A : ℕ) (hM : M = 15) (hA : A = 6) (hAL : L + A = M + 9) : L = 18 := by
  sorry

end find_Luisa_books_l30_30595


namespace sum_powers_l30_30597

theorem sum_powers {a b : ℝ}
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end sum_powers_l30_30597


namespace relation_y₁_y₂_y₃_l30_30924

def parabola (x : ℝ) : ℝ := - x^2 - 2 * x + 2
noncomputable def y₁ : ℝ := parabola (-2)
noncomputable def y₂ : ℝ := parabola (1)
noncomputable def y₃ : ℝ := parabola (2)

theorem relation_y₁_y₂_y₃ : y₁ > y₂ ∧ y₂ > y₃ := by
  have h₁ : y₁ = 2 := by
    unfold y₁ parabola
    norm_num
    
  have h₂ : y₂ = -1 := by
    unfold y₂ parabola
    norm_num
    
  have h₃ : y₃ = -6 := by
    unfold y₃ parabola
    norm_num
    
  rw [h₁, h₂, h₃]
  exact ⟨by norm_num, by norm_num⟩

end relation_y₁_y₂_y₃_l30_30924


namespace train_stop_times_l30_30026

theorem train_stop_times :
  ∀ (speed_without_stops_A speed_with_stops_A speed_without_stops_B speed_with_stops_B : ℕ),
  speed_without_stops_A = 45 →
  speed_with_stops_A = 30 →
  speed_without_stops_B = 60 →
  speed_with_stops_B = 40 →
  (60 * (speed_without_stops_A - speed_with_stops_A) / speed_without_stops_A = 20) ∧
  (60 * (speed_without_stops_B - speed_with_stops_B) / speed_without_stops_B = 20) :=
by
  intros
  sorry

end train_stop_times_l30_30026


namespace gunny_bag_capacity_l30_30603

def packets_weight_per_packet : ℝ := 16 + 4 / 16

def total_weight_pounds (num_packets : ℕ) : ℝ := num_packets * packets_weight_per_packet

theorem gunny_bag_capacity 
  (one_ton_pounds : ℝ) 
  (one_pound_ounces : ℝ) 
  (packet_count : ℕ) 
  (packet_weight_pounds : ℝ) 
  (packet_weight_ounces : ℝ) :
  one_ton_pounds = 2100 →
  one_pound_ounces = 16 →
  packet_count = 1680 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  total_weight_pounds 1680 / 2100 = 13
:= by
  intro h1 h2 h3 h4 h5
  sorry

end gunny_bag_capacity_l30_30603


namespace find_minimum_value_l30_30011

-- Define the function and conditions
def function_y (a x : ℝ) : ℝ := real.log (a * (x + 3)) - 1

lemma line_through_A (m n : ℝ) (h : m * n > 0) : 2 * m + n = 2 :=
  by sorry

theorem find_minimum_value (m n : ℝ) (hmn : m * n > 0) (key : 2 * m + n = 2) :
  (2 / m) + (1 / n) ≥ 9 / 2 :=
  by sorry

end find_minimum_value_l30_30011


namespace value_of_f_l30_30286

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then cos (π * x) else f (x - 1)

theorem value_of_f : f (4 / 3) + f (- (4 / 3)) = 1 :=
by
  sorry

end value_of_f_l30_30286


namespace compare_abc_l30_30896

noncomputable def a : ℝ := 3 ^ 0.6
noncomputable def b : ℝ := Real.log (2 / 3) / Real.log 2
noncomputable def c : ℝ := Real.cos (300 * Real.pi / 180)

theorem compare_abc : b < c ∧ c < a :=
by 
  sorry

end compare_abc_l30_30896


namespace line_intersects_circle_l30_30584

noncomputable def quadratic_roots_condition (m : ℝ) : Prop :=
  (m > 0 ∧ m < 4/3) ∧ 
  ∃ x1 x2 : ℝ, 
    (x1 ≠ x2) ∧ 
    (x1^2 + m * x1 + m^2 - m = 0) ∧ 
    (x2^2 + m * x2 + m^2 - m = 0) 

theorem line_intersects_circle (m : ℝ) (h1 : quadratic_roots_condition m) :
  ∃ x1 x2 : ℝ, 
    (x1 ≠ x2) ∧ 
    (x1^2 + m * x1 + m^2 - m = 0) ∧ 
    (x2^2 + m * x2 + m^2 - m = 0) ∧
    let k := (x1 + x2) in
    let midpoint := ((x1 + x2) / 2, (x1^2 + x2^2) / 2) in
    let line_eq := λ x y : ℝ, y + m * x + m^2 - m = 0 in
    let circle_center := (1 : ℝ, -1 : ℝ) in
    let radius := 1 in
    let d := abs (m^2 - 1) / sqrt (m^2 + 1) in
    d < radius := sorry

end line_intersects_circle_l30_30584


namespace three_planes_intersection_l30_30989

theorem three_planes_intersection (P1 P2 P3 : set (set ℝ^3))
  (h1 : ∃ l1 : set ℝ^3, is_line l1 ∧ l1 ⊆ P1 ∩ P2)
  (h2 : ∃ l2 : set ℝ^3, is_line l2 ∧ l2 ⊆ P2 ∩ P3)
  (h3 : ∃ l3 : set ℝ^3, is_line l3 ∧ l3 ⊆ P3 ∩ P1) :
  (∃! l : set ℝ^3, is_line l ∧ l ⊆ P1 ∩ P2 ∩ P3) ∨
  (∀ l1 l2 l3 : set ℝ^3,
    is_line l1 → is_line l2 → is_line l3 →
    l1 ⊆ P1 ∩ P2 → l2 ⊆ P2 ∩ P3 → l3 ⊆ P3 ∩ P1 →
    disjoint l1 l2 → disjoint l2 l3 → disjoint l3 l1) :=
sorry

end three_planes_intersection_l30_30989


namespace actual_time_before_storm_is_18_18_l30_30598

theorem actual_time_before_storm_is_18_18 :
  ∃ h m : ℕ, (h = 18) ∧ (m = 18) ∧ 
            ((09 = (if h == 0 then 1 else h - 1) ∨ 09 = (if h == 23 then 0 else h + 1)) ∧ 
             (09 = (if m == 0 then 1 else m - 1) ∨ 09 = (if m == 59 then 0 else m + 1))) := 
  sorry

end actual_time_before_storm_is_18_18_l30_30598


namespace smallest_number_divisible_by_5_with_digit_sum_100_l30_30263

theorem smallest_number_divisible_by_5_with_digit_sum_100 :
  ∃ N : ℕ, (N % 5 = 0) ∧ (Nat.digits 10 N).sum = 100 ∧
  ∀ M : ℕ, (M % 5 = 0) ∧ (Nat.digits 10 M).sum = 100 → N ≤ M :=
begin
  use 599999999995,
  split,
  { norm_num },
  split,
  { norm_num,
    sorry },
  { intros M h1 h2,
    sorry }
end

end smallest_number_divisible_by_5_with_digit_sum_100_l30_30263


namespace find_equation_of_line_l30_30154

noncomputable def point (x y : ℝ) : Prop := true
noncomputable def circle (x y k : ℝ) : Prop := x^2 + y^2 + 2 * x + 2 * y - k = 0
noncomputable def chord_length (length : ℝ) : Prop := length = 2 * Real.sqrt 7
noncomputable def equation_of_line (a b c : ℝ) (x y : ℝ) := a * x + b * y + c = 0

noncomputable def line_passing_through_point (l : ℝ → ℝ → Prop) (P : ℝ → ℝ → Prop) : Prop := 
  ∃ x y, P x y ∧ l x y

noncomputable def line_intersects_circle_to_form_chord (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → ℝ → Prop) (length : ℝ) : Prop :=
  ∃ a b, (∃ x y z t, C x y 14 ∧ l x z ∧ l y t ∧ Real.dist (x, z) (y, t) = length)

theorem find_equation_of_line :
  ∃ l : ℝ → ℝ → Prop,
    line_passing_through_point l (point 2 (-3)) ∧
    line_intersects_circle_to_form_chord l (circle x y k) (2 * Real.sqrt 7) ∧
    (equation_of_line 5 (-12) (-46) x y ∨ equation_of_line 1 0 (-2) x y) :=
sorry

end find_equation_of_line_l30_30154


namespace distinct_prime_factors_of_product_of_divisors_l30_30421

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30421


namespace fraction_identity_l30_30282

theorem fraction_identity (a b : ℚ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 :=
by
  sorry

end fraction_identity_l30_30282


namespace hide_and_seek_l30_30771

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30771


namespace hyperbola_eccentricity_l30_30987

variable {a b : ℝ}

def hyperbola := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

def line1 := {p : ℝ × ℝ | p.2 = 2 * p.1 + 1}

theorem hyperbola_eccentricity (ha : a > 0) (hb : b > 0) (perpendicular : (b / a) = (1 / 2)) : 
  let e := Real.sqrt (1 + (b^2) / (a^2)) in
  e = Real.sqrt(5) / 2 :=
sorry

end hyperbola_eccentricity_l30_30987


namespace perfect_square_k_value_l30_30981

-- Given condition:
def is_perfect_square (P : ℤ) : Prop := ∃ (z : ℤ), P = z * z

-- Theorem to prove:
theorem perfect_square_k_value (a b k : ℤ) (h : is_perfect_square (4 * a^2 + k * a * b + 9 * b^2)) :
  k = 12 ∨ k = -12 :=
sorry

end perfect_square_k_value_l30_30981


namespace angles_same_terminal_side_l30_30703

theorem angles_same_terminal_side (k : ℤ) : ∃ (n : ℤ), (k * 360 + 250 : ℤ) ≡ 610 [MOD 360] :=
by
  sorry

end angles_same_terminal_side_l30_30703


namespace find_a2_l30_30023

-- Define the sequence a_n
def seq (n : ℕ) : ℕ := sorry

-- The conditions
axiom a1 : seq 1 = 19
axiom a9 : seq 9 = 99
axiom arithmetic_mean : ∀ n : ℕ, n ≥ 3 → seq n = (list.sum (list.map seq (list.range (n - 1)))) / (n - 1)

-- The statement we want to prove
theorem find_a2 : seq 2 = 179 :=
sorry

end find_a2_l30_30023


namespace total_cost_is_correct_l30_30604

-- Conditions
def cost_per_object : ℕ := 11
def objects_per_person : ℕ := 5  -- 2 shoes, 2 socks, 1 mobile per person
def number_of_people : ℕ := 3

-- Expected total cost
def expected_total_cost : ℕ := 165

-- Proof problem: Prove that the total cost for storing all objects is 165 dollars
theorem total_cost_is_correct :
  (number_of_people * objects_per_person * cost_per_object) = expected_total_cost :=
by
  sorry

end total_cost_is_correct_l30_30604


namespace hide_and_seek_problem_l30_30744

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l30_30744


namespace hide_and_seek_l30_30765

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end hide_and_seek_l30_30765


namespace sequence_formula_l30_30947

noncomputable def l_n (n : ℕ+) : ℝ → ℝ := λ x, x - real.sqrt (2 * n)
noncomputable def C_n (n : ℕ+) : ℝ × ℝ → ℝ := λ ⟨x, y⟩, x^2 + y^2 - (2 * a n + n)

def a : ℕ+ → ℝ
| 1     := 1
| (n+1) := 1 / 4 * (dist (C_n n) (l_n n)) ^ 2

theorem sequence_formula (n : ℕ+) : 
  a n = 2^(n - 1) := 
sorry

end sequence_formula_l30_30947


namespace sum_of_first_8_terms_seq_l30_30333

def a (n : ℕ) : ℕ := n

def S (n : ℕ) : ℚ := (n * (n + 1)) / 2

noncomputable def seq_term (n : ℕ) : ℚ :=
(a (n + 1)) / (S n) / (S (n + 1))

theorem sum_of_first_8_terms_seq :
  (∑ k in Finset.range 8, seq_term k) = 44 / 45 :=
sorry

end sum_of_first_8_terms_seq_l30_30333


namespace count_irrational_numbers_l30_30181

def is_irrational (x : ℝ) : Prop := ¬ (∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

noncomputable def num_list : List ℝ := [
  3.14159,
  -real.cbrt 9,
  0.131131113.repr.to_real,
  -real.pi,
  real.sqrt 25,
  real.cbrt 64,
  -1 / 7
]

theorem count_irrational_numbers :
  (num_list.filter is_irrational).length = 3 := 
sorry

end count_irrational_numbers_l30_30181


namespace expected_number_of_ones_on_three_dice_l30_30073

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30073


namespace perfect_square_trinomial_k_l30_30978

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), (a * x + b) ^ 2 = x ^ 2 + k * x + 9) → (k = 6 ∨ k = -6) :=
by
  sorry

end perfect_square_trinomial_k_l30_30978


namespace prime_factors_of_B_l30_30528

-- Definition of the product of the divisors of a natural number
noncomputable def prod_of_divisors (n : ℕ) : ℕ :=
  ∏ d in (finset.filter (λ d, d ∣ n) (finset.range (n + 1))), d

-- Conditions
def sixty : ℕ := 60
def B : ℕ := prod_of_divisors sixty

-- Question: Proving that B has exactly 3 distinct prime factors
theorem prime_factors_of_B (h : B = prod_of_divisors 60) : nat.distinct_prime_factors B = 3 :=
sorry

end prime_factors_of_B_l30_30528


namespace total_meters_examined_l30_30831

-- Define the conditions
def proportion_defective : ℝ := 0.1
def defective_meters : ℕ := 10

-- The statement to prove
theorem total_meters_examined (T : ℝ) (h : proportion_defective * T = defective_meters) : T = 100 :=
by
  sorry

end total_meters_examined_l30_30831


namespace first_year_students_selected_l30_30669

theorem first_year_students_selected (ratio1 ratio2 ratio3 total_students : ℕ) (h_ratio : ratio1 = 7 ∧ ratio2 = 3 ∧ ratio3 = 4) (h_total_students : total_students = 56) : 
    let total_ratio := ratio1 + ratio2 + ratio3 in
    let first_year_students := total_students * ratio1 / total_ratio in
    first_year_students = 28 :=
by
    rw [h_ratio.left, h_ratio.right.left, h_ratio.right.right, h_total_students]
    simp [nat.div_eq_of_eq_mul]
    rfl
    sorry

end first_year_students_selected_l30_30669


namespace games_given_to_neil_is_five_l30_30959

variable (x : ℕ)

def initial_games_henry : ℕ := 33
def initial_games_neil : ℕ := 2
def games_given_to_neil : ℕ := x

theorem games_given_to_neil_is_five
  (H : initial_games_henry - games_given_to_neil = 4 * (initial_games_neil + games_given_to_neil)) :
  games_given_to_neil = 5 := by
  sorry

end games_given_to_neil_is_five_l30_30959


namespace num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30498

def is_divisor (a b : ℕ) : Prop := b % a = 0

def product_of_divisors (n : ℕ) : ℕ :=
(n.divisors).prod

theorem num_distinct_prime_factors_of_product_of_divisors_of_60 :
  ∃ B, B = product_of_divisors 60 ∧ B.distinct_prime_factors.count = 3 :=
begin
  sorry
end

end num_distinct_prime_factors_of_product_of_divisors_of_60_l30_30498


namespace smallest_palindrome_in_bases_2_4_l30_30218

def is_palindrome (s : List Char) : Bool :=
  s = s.reverse

def to_digits (n : Nat) (base : Nat) : List Char :=
  Nat.digits base n |> List.map (λ d, Char.ofNat (d + 48))

def is_palindrome_in_base (n base : Nat) : Bool :=
  is_palindrome (to_digits n base)

theorem smallest_palindrome_in_bases_2_4 :
  ∃ n : Nat, n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧
  ∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m :=
  ∃ n : Nat, n = 15 ∧ n > 10 ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 ∧ (∀ m : Nat, m > 10 → is_palindrome_in_base m 2 → is_palindrome_in_base m 4 → n ≤ m) :=
  sorry

end smallest_palindrome_in_bases_2_4_l30_30218


namespace polynomial_coefficients_even_or_odd_l30_30955

-- Define the problem conditions as Lean definitions
variables {P Q : Polynomial ℤ}

-- Theorem: Given the conditions, prove the required statement
theorem polynomial_coefficients_even_or_odd
  (hP : ∀ n : ℕ, P.coeff n % 2 = 0)
  (hQ : ∀ n : ℕ, Q.coeff n % 2 = 0)
  (hProd : ¬ ∀ n : ℕ, (P * Q).coeff n % 4 = 0) :
  (∀ n : ℕ, P.coeff n % 2 = 0 ∧ ∃ k : ℕ, Q.coeff k % 2 ≠ 0) ∨
  (∀ n : ℕ, Q.coeff n % 2 = 0 ∧ ∃ k: ℕ, P.coeff k % 2 ≠ 0) :=
sorry

end polynomial_coefficients_even_or_odd_l30_30955


namespace age_when_sum_is_20_l30_30029

noncomputable def age_diff (x y : ℕ) : ℕ := y - x
noncomputable def years_ago (x y k : ℕ)  : ℕ := (x - k) + (y - k)

theorem age_when_sum_is_20 :
  ∀ (x y : ℕ), x = 18 ∧ y = 26 →
  ∃ k, (years_ago x y k) = 20 ∧ (y - k) = 14 :=
by
  intros x y h
  cases h with hx hy
  use 12
  simp [years_ago, age_diff, hx, hy]
  split
  {
    -- calculate the sum of ages 12 years ago
    calc (18 - 12) + (26 - 12) = 6 + 14 : by simp
                             ... = 20 : by simp,
  }
  {
    -- calculate the age of 姐姐 12 years ago
    calc 26 - 12 = 14 : by simp,
  }

end age_when_sum_is_20_l30_30029


namespace solve_x_l30_30231

def star (a b : ℝ) : ℝ :=
  if a ≥ b then a * b + b else a * b - a

theorem solve_x (x : ℝ) :
  (star (2 * x - 1) (x+2) = 0) →
  (x = -1 ∨ x = 1 / 2) :=
by
  sorry

end solve_x_l30_30231


namespace rectangle_diagonal_length_l30_30656

theorem rectangle_diagonal_length (L W : ℝ) (h1 : 2 * (L + W) = 72) (h2 : L / W = 5 / 2) : 
    (sqrt (L^2 + W^2)) = 194 / 7 := 
by 
  sorry

end rectangle_diagonal_length_l30_30656


namespace train_carriages_l30_30172

theorem train_carriages
  (carriage_length engine_length : ℕ)
  (bridge_length_km time_min  speed_kmph : ℕ)
  (carriage_length = 60)
  (engine_length = 60)
  (bridge_length_km = 35 / 10)  -- 3.5 km 
  (time_min = 5)
  (speed_kmph = 60) :
  let speed_mpm := speed_kmph * 1000 / 60 in
  let bridge_length_m := bridge_length_km * 1000 in
  let total_distance_covered := speed_mpm * time_min in
  let train_length := total_distance_covered - bridge_length_m in
  let n := (train_length - engine_length) / carriage_length in
  n = 24 :=
by
  sorry

end train_carriages_l30_30172


namespace abs_x_minus_y_zero_l30_30624

theorem abs_x_minus_y_zero (x y : ℝ) 
  (h_avg : (x + y + 30 + 29 + 31) / 5 = 30)
  (h_var : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) : 
  |x - y| = 0 :=
  sorry

end abs_x_minus_y_zero_l30_30624


namespace expected_number_of_ones_on_three_dice_l30_30078

noncomputable def expectedOnesInThreeDice : ℚ := 
  let p1 : ℚ := 1/6
  let pNot1 : ℚ := 5/6
  0 * (pNot1 ^ 3) + 
  1 * (3 * p1 * (pNot1 ^ 2)) + 
  2 * (3 * (p1 ^ 2) * pNot1) + 
  3 * (p1 ^ 3)

theorem expected_number_of_ones_on_three_dice :
  expectedOnesInThreeDice = 1 / 2 :=
by 
  sorry

end expected_number_of_ones_on_three_dice_l30_30078


namespace trig_identity_proof_l30_30897

theorem trig_identity_proof (α : ℝ)
  (hα1 : real.sin α = 3/5)
  (hα2 : π / 2 < α ∧ α < π) :
  let f (x : ℝ) := real.sin (x + π / 6) in
  f (α + π / 12) = - sqrt 2 / 10 := 
by
  sorry

end trig_identity_proof_l30_30897


namespace probability_of_union_l30_30344

variables {Ω : Type} [MeasureSpace Ω]
variables (A B : Set Ω) (P : Measure Ω)
variable [ProbabilityMeasure P]

theorem probability_of_union (h1 : A ⊆ B) (h2 : P A = 0.05) (h3 : P B = 0.15) : P (A ∪ B) = 0.15 :=
by
  sorry

end probability_of_union_l30_30344


namespace distinct_prime_factors_of_product_of_divisors_l30_30422

theorem distinct_prime_factors_of_product_of_divisors (B : ℕ) (h : B = ∏ d in (finset.univ.filter (λ d, d ∣ 60)), d) : 
  finset.card (finset.univ.filter (λ p, nat.prime p ∧ p ∣ B)) = 3 :=
by {
  sorry
}

end distinct_prime_factors_of_product_of_divisors_l30_30422


namespace hide_and_seek_friends_l30_30761

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30761


namespace smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30207

def is_palindrome_in_base (n : ℕ) (b : ℕ) : Prop :=
  let digits : List ℕ := (nat.to_digits b n)
  digits = digits.reverse

theorem smallest_palindrome_in_bases_2_and_4 :
  ∀ n : ℕ, 10 < n ∧ is_palindrome_in_base n 2 ∧ is_palindrome_in_base n 4 → n ≥ 17 :=
begin
  intros n hn,
  have := (by norm_num : 16 < 17),
  cases hn with hn1 hn2,
  cases hn2 with hn3 hn4,
  linarith,
end

theorem smallest_palindrome_in_bases_2_and_4_17 :
  is_palindrome_in_base 17 2 ∧ is_palindrome_in_base 17 4 :=
begin
  split;
  { unfold is_palindrome_in_base, norm_num, refl },
end

end smallest_palindrome_in_bases_2_and_4_smallest_palindrome_in_bases_2_and_4_17_l30_30207


namespace quadratic_scaling_l30_30668

theorem quadratic_scaling (a b c : ℝ) (h : ℝ) (n k : ℝ) 
  (h1 : quadratic_expr : (a * x^2 + b * x + c) = 3 * (x - 5)^2 + 7) :
  ∃ (h : ℝ), (n * (x - h)^2 + k) = 12 * (x - 5)^2 + 28 ∧ h = 5 :=
by 
  have h2 : 4 * (a * x^2 + b * x + c) = 12 * (x - 5)^2 + 28,
    -- multiply by 4
    sorry,
  
  use 5,
  exact ⟨12, 28, h2, rfl⟩

end quadratic_scaling_l30_30668


namespace parabola_points_relation_l30_30922

theorem parabola_points_relation :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = -(-2)^2 - 2*(-2) + 2) ∧ 
  (y2 = -(1)^2 - 2*(1) + 2) ∧ 
  (y3 = -(2)^2 - 2*(2) + 2) → 
  y1 > y2 ∧ y2 > y3 :=
by {
  intros y1 y2 y3 h,
  obtain ⟨h1, h2, h3⟩ := h,
  rw [h1, h2, h3],
  -- This is the placeholder for the proof
  sorry
}

end parabola_points_relation_l30_30922


namespace calculation_l30_30193

def fraction1 : Real := (15 : Real) / (4 : Real)
def fraction2 : Real := (8 : Real) / 3
def sum_sequence : Nat := 1 + 3 + 5 + 7 + 9
def denominator : Real := (sum_sequence : Real) * 20 + 3
def numerator : Real := fraction1 * 1.3 + (3 / fraction2)
def expression : Real := 2012 * (numerator / denominator)

theorem calculation : expression = 24 := by
  sorry

end calculation_l30_30193


namespace a4_plus_a_neg4_l30_30000

variable (a : ℝ)
-- Condition given in the problem
def condition : Prop := 5 = a + a⁻¹

-- Target statement to prove
theorem a4_plus_a_neg4 : condition a → a^4 + a^(-4) = 527 :=
by
  intro h
  sorry

end a4_plus_a_neg4_l30_30000


namespace geometric_shapes_on_cube_l30_30617

-- Definitions of the geometric shapes
def isRectangle (vertices : set Point) : Prop :=
  -- Define the condition for vertices forming a rectangle
  sorry

def isParallelogramButNotRectangle (vertices : set Point) : Prop :=
  -- Define the condition for vertices forming a parallelogram but not a rectangle
  sorry

def isTetrahedronWithSpecificProperties (vertices : set Point) (property : Prop) : Prop :=
  -- Define the condition for vertices forming a tetrahedron with specific face properties
  property

def isTetrahedronWithIsoscelesAndEquilateral (vertices : set Point) : Prop :=
  isTetrahedronWithSpecificProperties vertices (
    -- Define the condition for vertices forming a tetrahedron with 3 isosceles right triangle faces and 1 equilateral triangle face
    sorry
  )

def isTetrahedronWithEquilateralFaces (vertices : set Point) : Prop :=
  isTetrahedronWithSpecificProperties vertices (
    -- Define the condition for vertices forming a tetrahedron with all equilateral triangle faces
    sorry
  )

def isTetrahedronWithRightTriangleFaces (vertices : set Point) : Prop :=
  isTetrahedronWithSpecificProperties vertices (
    -- Define the condition for vertices forming a tetrahedron with all right triangle faces
    sorry
  )

-- Theorem statement
theorem geometric_shapes_on_cube :
  ∀ vertices : set Point, (∃ (a b c d : Point), {a, b, c, d} = vertices ∧
    isVertexOfCube a ∧ isVertexOfCube b ∧ isVertexOfCube c ∧ isVertexOfCube d) →
  isRectangle vertices ∨
  isTetrahedronWithIsoscelesAndEquilateral vertices ∨
  isTetrahedronWithEquilateralFaces vertices ∨
  isTetrahedronWithRightTriangleFaces vertices :=
by
  sorry

end geometric_shapes_on_cube_l30_30617


namespace abcd_mul_equals_2004_l30_30273

variables (a b c d : ℝ)

def condition_a (a : ℝ) : Prop := a = real.sqrt (45 - real.sqrt (21 - a))
def condition_b (b : ℝ) : Prop := b = real.sqrt (45 + real.sqrt (21 - b))
def condition_c (c : ℝ) : Prop := c = real.sqrt (45 - real.sqrt (21 + c))
def condition_d (d : ℝ) : Prop := d = real.sqrt (45 + real.sqrt (21 + d))

theorem abcd_mul_equals_2004 (ha : condition_a a) (hb : condition_b b) (hc : condition_c c) (hd : condition_d d) : 
  a * b * c * d = 2004 :=
sorry

end abcd_mul_equals_2004_l30_30273


namespace hide_and_seek_friends_l30_30758

open Classical

variables (A B V G D : Prop)

/-- Conditions -/
axiom cond1 : A → (B ∧ ¬V)
axiom cond2 : B → (G ∨ D)
axiom cond3 : ¬V → (¬B ∧ ¬D)
axiom cond4 : ¬A → (B ∧ ¬G)

/-- Proof that Alex played hide and seek with Boris, Vasya, and Denis -/
theorem hide_and_seek_friends : B ∧ V ∧ D := by
  sorry

end hide_and_seek_friends_l30_30758
