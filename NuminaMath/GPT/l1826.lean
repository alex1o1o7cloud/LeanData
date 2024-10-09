import Mathlib

namespace average_of_four_l1826_182660

-- Define the variables
variables {p q r s : ℝ}

-- Conditions as hypotheses
theorem average_of_four (h : (5 / 4) * (p + q + r + s) = 15) : (p + q + r + s) / 4 = 3 := 
by
  sorry

end average_of_four_l1826_182660


namespace triangle_area_l1826_182693

noncomputable def area_of_triangle (l1 l2 l3 : ℝ × ℝ → Prop) (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem triangle_area :
  let A := (1, 6)
  let B := (-1, 6)
  let C := (0, 4)
  ∀ x y : ℝ, 
    (y = 6 → l1 (x, y)) ∧ 
    (y = 2 * x + 4 → l2 (x, y)) ∧ 
    (y = -2 * x + 4 → l3 (x, y)) →
  area_of_triangle l1 l2 l3 A B C = 1 :=
by 
  intros
  unfold area_of_triangle
  sorry

end triangle_area_l1826_182693


namespace find_n_infinitely_many_squares_find_n_no_squares_l1826_182673

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P (n k l m : ℕ) : ℕ := n^k + n^l + n^m

theorem find_n_infinitely_many_squares :
  ∃ k, ∃ l, ∃ m, is_square (P 7 k l m) :=
by
  sorry

theorem find_n_no_squares :
  ∀ (k l m : ℕ) n, n ∈ [5, 6] → ¬is_square (P n k l m) :=
by
  sorry

end find_n_infinitely_many_squares_find_n_no_squares_l1826_182673


namespace total_spending_in_CAD_proof_l1826_182662

-- Define Jayda's spending
def Jayda_spending_stall1 : ℤ := 400
def Jayda_spending_stall2 : ℤ := 120
def Jayda_spending_stall3 : ℤ := 250

-- Define the factor by which Aitana spends more
def Aitana_factor : ℚ := 2 / 5

-- Define the sales tax rate
def sales_tax_rate : ℚ := 0.10

-- Define the exchange rate from USD to CAD
def exchange_rate : ℚ := 1.25

-- Calculate Jayda's total spending in USD before tax
def Jayda_total_spending : ℤ := Jayda_spending_stall1 + Jayda_spending_stall2 + Jayda_spending_stall3

-- Calculate Aitana's spending at each stall
def Aitana_spending_stall1 : ℚ := Jayda_spending_stall1 + (Aitana_factor * Jayda_spending_stall1)
def Aitana_spending_stall2 : ℚ := Jayda_spending_stall2 + (Aitana_factor * Jayda_spending_stall2)
def Aitana_spending_stall3 : ℚ := Jayda_spending_stall3 + (Aitana_factor * Jayda_spending_stall3)

-- Calculate Aitana's total spending in USD before tax
def Aitana_total_spending : ℚ := Aitana_spending_stall1 + Aitana_spending_stall2 + Aitana_spending_stall3

-- Calculate the combined total spending in USD before tax
def combined_total_spending_before_tax : ℚ := Jayda_total_spending + Aitana_total_spending

-- Calculate the sales tax amount
def sales_tax : ℚ := sales_tax_rate * combined_total_spending_before_tax

-- Calculate the total spending including sales tax
def total_spending_including_tax : ℚ := combined_total_spending_before_tax + sales_tax

-- Convert the total spending to Canadian dollars
def total_spending_in_CAD : ℚ := total_spending_including_tax * exchange_rate

-- The theorem to be proven
theorem total_spending_in_CAD_proof : total_spending_in_CAD = 2541 := sorry

end total_spending_in_CAD_proof_l1826_182662


namespace max_of_a_l1826_182620

theorem max_of_a (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a + b + c + d = 4) (h6 : a^2 + b^2 + c^2 + d^2 = 8) : a ≤ 1 + Real.sqrt 3 :=
sorry

end max_of_a_l1826_182620


namespace incorrect_statement_A_l1826_182696

def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def has_real_roots (a b c : ℝ) : Prop :=
  let delta := b^2 - 4 * a * c
  delta ≥ 0

theorem incorrect_statement_A (a b c : ℝ) (h₀ : a ≠ 0) :
  (∃ x : ℝ, parabola a b c x = 0) ∧ (parabola a b c (-b/(2*a)) < 0) → ¬ has_real_roots a b c := 
by
  sorry -- proof required here if necessary

end incorrect_statement_A_l1826_182696


namespace value_fraction_l1826_182617

variables {x y : ℝ}
variables (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + 2 * y) / (x - 4 * y) = 3)

theorem value_fraction : (x + 4 * y) / (4 * x - y) = 10 / 57 :=
by { sorry }

end value_fraction_l1826_182617


namespace identity_function_l1826_182607

theorem identity_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ y : ℝ, f y = y :=
by 
  sorry

end identity_function_l1826_182607


namespace largest_natural_divisible_power_l1826_182663

theorem largest_natural_divisible_power (p q : ℤ) (hp : p % 5 = 0) (hq : q % 5 = 0) (hdiscr : p^2 - 4*q > 0) :
  ∀ (α β : ℂ), (α^2 + p*α + q = 0 ∧ β^2 + p*β + q = 0) → (α^100 + β^100) % 5^50 = 0 :=
sorry

end largest_natural_divisible_power_l1826_182663


namespace total_tagged_numbers_l1826_182635

theorem total_tagged_numbers:
  let W := 200
  let X := W / 2
  let Y := X + W
  let Z := 400
  W + X + Y + Z = 1000 := by 
    sorry

end total_tagged_numbers_l1826_182635


namespace sum_of_cubes_l1826_182644

def cubic_eq (x : ℝ) : Prop := x^3 - 2 * x^2 + 3 * x - 4 = 0

variables (a b c : ℝ)

axiom a_root : cubic_eq a
axiom b_root : cubic_eq b
axiom c_root : cubic_eq c

axiom sum_roots : a + b + c = 2
axiom sum_products_roots : a * b + a * c + b * c = 3
axiom product_roots : a * b * c = 4

theorem sum_of_cubes : a^3 + b^3 + c^3 = 2 :=
by
  sorry

end sum_of_cubes_l1826_182644


namespace proof_problem_l1826_182606

def polar_curve_C (ρ : ℝ) : Prop := ρ = 5

def point_P (x y : ℝ) : Prop := x = -3 ∧ y = -3 / 2

def line_l_through_P (x y : ℝ) (k : ℝ) : Prop := y + 3 / 2 = k * (x + 3)

def distance_AB (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64

theorem proof_problem
  (ρ : ℝ) (x y : ℝ) (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : polar_curve_C ρ)
  (h2 : point_P (-3) (-3 / 2))
  (h3 : ∃ k, line_l_through_P x y k)
  (h4 : distance_AB A B) :
  ∃ (x y : ℝ), (x^2 + y^2 = 25) ∧ ((x = -3) ∨ (3 * x + 4 * y + 15 = 0)) := 
sorry

end proof_problem_l1826_182606


namespace complete_square_transform_l1826_182683

theorem complete_square_transform (x : ℝ) :
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := 
sorry

end complete_square_transform_l1826_182683


namespace equal_binomial_terms_l1826_182614

theorem equal_binomial_terms (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : p + q = 1)
    (h4 : 55 * p^9 * q^2 = 165 * p^8 * q^3) : p = 3 / 4 :=
by
  sorry

end equal_binomial_terms_l1826_182614


namespace area_D_meets_sign_l1826_182665

-- Definition of conditions as given in the question
def condition_A (mean median : ℝ) : Prop := mean = 3 ∧ median = 4
def condition_B (mean : ℝ) (variance_pos : Prop) : Prop := mean = 1 ∧ variance_pos
def condition_C (median mode : ℝ) : Prop := median = 2 ∧ mode = 3
def condition_D (mean variance : ℝ) : Prop := mean = 2 ∧ variance = 3

-- Theorem stating that Area D satisfies the condition to meet the required sign
theorem area_D_meets_sign (mean variance : ℝ) (h : condition_D mean variance) : 
  (∀ day_increase, day_increase ≤ 7) :=
sorry

end area_D_meets_sign_l1826_182665


namespace sufficient_but_not_necessary_condition_l1826_182689

variables {a b : ℝ}

theorem sufficient_but_not_necessary_condition (h₁ : b < -4) : |a| + |b| > 4 :=
by {
    sorry
}

end sufficient_but_not_necessary_condition_l1826_182689


namespace arithmetic_sequence_common_difference_l1826_182631

theorem arithmetic_sequence_common_difference  (a_n : ℕ → ℝ)
  (h1 : a_n 1 + a_n 6 = 12)
  (h2 : a_n 4 = 7) :
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 1 + (n - 1) * d ∧ d = 2 := 
sorry

end arithmetic_sequence_common_difference_l1826_182631


namespace find_x_l1826_182629

def operation (a b : Int) : Int := 2 * a + b

theorem find_x :
  ∃ x : Int, operation 3 (operation 4 x) = -1 :=
  sorry

end find_x_l1826_182629


namespace number_of_common_terms_between_arithmetic_sequences_l1826_182671

-- Definitions for the sequences
def seq1 (n : Nat) := 2 + 3 * n
def seq2 (n : Nat) := 4 + 5 * n

theorem number_of_common_terms_between_arithmetic_sequences
  (A : Finset Nat := Finset.range 673)  -- There are 673 terms in seq1 from 2 to 2015
  (B : Finset Nat := Finset.range 403)  -- There are 403 terms in seq2 from 4 to 2014
  (common_terms : Finset Nat := (A.image seq1) ∩ (B.image seq2)) :
  common_terms.card = 134 := by
  sorry

end number_of_common_terms_between_arithmetic_sequences_l1826_182671


namespace number_of_keepers_l1826_182647

theorem number_of_keepers
  (h₁ : 50 * 2 = 100)
  (h₂ : 45 * 4 = 180)
  (h₃ : 8 * 4 = 32)
  (h₄ : 12 * 8 = 96)
  (h₅ : 6 * 8 = 48)
  (h₆ : 100 + 180 + 32 + 96 + 48 = 456)
  (h₇ : 50 + 45 + 8 + 12 + 6 = 121)
  (h₈ : ∀ K : ℕ, (2 * (K - 5) + 6 + 2 = 2 * K - 2))
  (h₉ : ∀ K : ℕ, 121 + K + 372 = 456 + (2 * K - 2)) :
  ∃ K : ℕ, K = 39 :=
by
  sorry

end number_of_keepers_l1826_182647


namespace trajectory_of_P_l1826_182619

theorem trajectory_of_P (M P : ℝ × ℝ) (OM OP : ℝ) (x y : ℝ) :
  (M = (4, y)) →
  (P = (x, y)) →
  (OM = Real.sqrt (4^2 + y^2)) →
  (OP = Real.sqrt ((x - 4)^2 + y^2)) →
  (OM * OP = 16) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end trajectory_of_P_l1826_182619


namespace three_digit_subtraction_l1826_182604

theorem three_digit_subtraction (c d : ℕ) (H1 : 0 ≤ c ∧ c ≤ 9) (H2 : 0 ≤ d ∧ d ≤ 9) :
  (745 - (300 + c * 10 + 4) = (400 + d * 10 + 1)) ∧ ((4 + 1) - d % 11 = 0) → 
  c + d = 14 := 
sorry

end three_digit_subtraction_l1826_182604


namespace stan_weighs_5_more_than_steve_l1826_182652

theorem stan_weighs_5_more_than_steve
(S V J : ℕ) 
(h1 : J = 110)
(h2 : V = J - 8)
(h3 : S + V + J = 319) : 
(S - V = 5) :=
by
  sorry

end stan_weighs_5_more_than_steve_l1826_182652


namespace find_abc_solutions_l1826_182650

theorem find_abc_solutions
    (a b c : ℕ)
    (h_pos : (a > 0) ∧ (b > 0) ∧ (c > 0))
    (h1 : a < b)
    (h2 : a < 4 * c)
    (h3 : b * c ^ 3 ≤ a * c ^ 3 + b) :
    ((a = 7) ∧ (b = 8) ∧ (c = 2)) ∨
    ((a = 1 ∨ a = 2 ∨ a = 3) ∧ (b > a) ∧ (c = 1)) :=
by
  sorry

end find_abc_solutions_l1826_182650


namespace number_of_lines_l1826_182605

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero: a ≠ 0 ∨ b ≠ 0

-- Definition of a line passing through a point P
def passes_through (l : Line) (P : Point) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Definition of a line having equal intercepts on x-axis and y-axis
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.a = l.b

-- Definition of a specific point P
def P : Point := { x := 1, y := 2 }

-- The theorem statement
theorem number_of_lines : ∃ (lines : Finset Line), (∀ l ∈ lines, passes_through l P ∧ equal_intercepts l) ∧ lines.card = 2 := by
  sorry

end number_of_lines_l1826_182605


namespace jerry_total_miles_l1826_182633

def monday : ℕ := 15
def tuesday : ℕ := 18
def wednesday : ℕ := 25
def thursday : ℕ := 12
def friday : ℕ := 10

def total : ℕ := monday + tuesday + wednesday + thursday + friday

theorem jerry_total_miles : total = 80 := by
  sorry

end jerry_total_miles_l1826_182633


namespace partial_fraction_sum_eq_zero_l1826_182626

theorem partial_fraction_sum_eq_zero (A B C D E : ℂ) :
  (∀ x : ℂ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 4 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_eq_zero_l1826_182626


namespace sum_of_two_primes_unique_l1826_182610

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l1826_182610


namespace find_a_l1826_182681

variable (A B : Set ℤ) (a : ℤ)
variable (elem1 : 0 ∈ A) (elem2 : 1 ∈ A)
variable (elem3 : -1 ∈ B) (elem4 : 0 ∈ B) (elem5 : a + 3 ∈ B)

theorem find_a (h : A ⊆ B) : a = -2 := sorry

end find_a_l1826_182681


namespace tim_movie_marathon_duration_is_9_l1826_182685

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l1826_182685


namespace mixture_replacement_l1826_182616

theorem mixture_replacement (A B x : ℕ) (hA : A = 32) (h_ratio1 : A / B = 4) (h_ratio2 : A / (B + x) = 2 / 3) : x = 40 :=
by
  sorry

end mixture_replacement_l1826_182616


namespace strawberry_cost_l1826_182618

theorem strawberry_cost (price_per_basket : ℝ) (num_baskets : ℕ) (total_cost : ℝ)
  (h1 : price_per_basket = 16.50) (h2 : num_baskets = 4) : total_cost = 66.00 :=
by
  sorry

end strawberry_cost_l1826_182618


namespace range_of_a_for_monotonicity_l1826_182601

noncomputable def f (x : ℝ) (a : ℝ) := (Real.sqrt (x^2 + 1)) - a * x

theorem range_of_a_for_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x a < f y a) ↔ a ≥ 1 := sorry

end range_of_a_for_monotonicity_l1826_182601


namespace negation_equiv_l1826_182675

theorem negation_equiv (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l1826_182675


namespace at_least_2_boys_and_1_girl_l1826_182668

noncomputable def probability_at_least_2_boys_and_1_girl (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_with_0_boys := Nat.choose girls committee_size
  let ways_with_1_boy := Nat.choose boys 1 * Nat.choose girls (committee_size - 1)
  let ways_with_fewer_than_2_boys := ways_with_0_boys + ways_with_1_boy
  1 - (ways_with_fewer_than_2_boys / total_ways)

theorem at_least_2_boys_and_1_girl :
  probability_at_least_2_boys_and_1_girl 32 14 18 6 = 767676 / 906192 :=
by
  sorry

end at_least_2_boys_and_1_girl_l1826_182668


namespace river_width_l1826_182603

noncomputable def width_of_river (d: ℝ) (f: ℝ) (v: ℝ) : ℝ :=
  v / (d * (f * 1000 / 60))

theorem river_width : width_of_river 2 2 3000 = 45 := by
  sorry

end river_width_l1826_182603


namespace total_guppies_l1826_182678

noncomputable def initial_guppies : Nat := 7
noncomputable def baby_guppies_first_set : Nat := 3 * 12
noncomputable def baby_guppies_additional : Nat := 9

theorem total_guppies : initial_guppies + baby_guppies_first_set + baby_guppies_additional = 52 :=
by
  sorry

end total_guppies_l1826_182678


namespace gigi_mushrooms_l1826_182682

-- Define the conditions
def pieces_per_mushroom := 4
def kenny_pieces := 38
def karla_pieces := 42
def remaining_pieces := 8

-- Main theorem
theorem gigi_mushrooms : (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry

end gigi_mushrooms_l1826_182682


namespace counting_indistinguishable_boxes_l1826_182643

def distinguishable_balls := 5
def indistinguishable_boxes := 3

theorem counting_indistinguishable_boxes :
  (∃ ways : ℕ, ways = 66) := sorry

end counting_indistinguishable_boxes_l1826_182643


namespace parabola_distance_l1826_182612

theorem parabola_distance (p : ℝ) (hp : 0 < p) (hf : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  dist P (0, p / 2) = 16) (hx : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  P.2 = 10) : p = 12 :=
sorry

end parabola_distance_l1826_182612


namespace neznaika_mistake_correct_numbers_l1826_182613

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l1826_182613


namespace maximize_revenue_l1826_182640

theorem maximize_revenue (p : ℝ) (h : p ≤ 30) : 
  (∀ q : ℝ, q ≤ 30 → (150 * 18.75 - 4 * (18.75:ℝ)^2) ≥ (150 * q - 4 * q^2)) ↔ p = 18.75 := 
sorry

end maximize_revenue_l1826_182640


namespace problem_statement_l1826_182649

def U : Set Int := {x | |x| < 5}
def A : Set Int := {-2, 1, 3, 4}
def B : Set Int := {0, 2, 4}

theorem problem_statement : (A ∩ (U \ B)) = {-2, 1, 3} := by
  sorry

end problem_statement_l1826_182649


namespace parallel_lines_condition_l1826_182637

theorem parallel_lines_condition (a : ℝ) :
  ( ∀ x y : ℝ, (a * x + 2 * y + 2 = 0 → ∃ C₁ : ℝ, x - 2 * y = C₁) 
  ∧ (x + (a - 1) * y + 1 = 0 → ∃ C₂ : ℝ, x - 2 * y = C₂) )
  ↔ a = -1 :=
sorry

end parallel_lines_condition_l1826_182637


namespace total_number_of_shells_l1826_182621

variable (David Mia Ava Alice : ℕ)
variable (hd : David = 15)
variable (hm : Mia = 4 * David)
variable (ha : Ava = Mia + 20)
variable (hAlice : Alice = Ava / 2)

theorem total_number_of_shells :
  David + Mia + Ava + Alice = 195 :=
by
  sorry

end total_number_of_shells_l1826_182621


namespace floor_sqrt_23_squared_l1826_182639

theorem floor_sqrt_23_squared : (Int.floor (Real.sqrt 23))^2 = 16 := 
by
  -- conditions
  have h1 : 4^2 = 16 := by norm_num
  have h2 : 5^2 = 25 := by norm_num
  have h3 : 16 < 23 := by norm_num
  have h4 : 23 < 25 := by norm_num
  -- statement (goal)
  sorry

end floor_sqrt_23_squared_l1826_182639


namespace polygon_sides_l1826_182628

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1080) : n = 8 :=
sorry

end polygon_sides_l1826_182628


namespace totalCats_l1826_182655

def whiteCats : Nat := 2
def blackCats : Nat := 10
def grayCats : Nat := 3

theorem totalCats : whiteCats + blackCats + grayCats = 15 := by
  sorry

end totalCats_l1826_182655


namespace Anne_height_l1826_182672

-- Define the conditions
variables (S : ℝ)   -- Height of Anne's sister
variables (A : ℝ)   -- Height of Anne
variables (B : ℝ)   -- Height of Bella

-- Define the relations according to the problem's conditions
def condition1 (S : ℝ) := A = 2 * S
def condition2 (S : ℝ) := B = 3 * A
def condition3 (S : ℝ) := B - S = 200

-- Theorem statement to prove Anne's height
theorem Anne_height (S : ℝ) (A : ℝ) (B : ℝ)
(h1 : A = 2 * S) (h2 : B = 3 * A) (h3 : B - S = 200) : A = 80 :=
by sorry

end Anne_height_l1826_182672


namespace find_C_value_l1826_182642

theorem find_C_value (A B C : ℕ) 
  (cond1 : A + B + C = 10) 
  (cond2 : B + A = 9)
  (cond3 : A + 1 = 3) :
  C = 1 :=
by
  sorry

end find_C_value_l1826_182642


namespace geometric_product_Pi8_l1826_182664

def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

variables {a : ℕ → ℝ}
variable (h_geom : geometric_sequence a)
variable (h_prod : a 4 * a 5 = 2)

theorem geometric_product_Pi8 :
  (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = 16 :=
by
  sorry

end geometric_product_Pi8_l1826_182664


namespace max_ratio_is_99_over_41_l1826_182657

noncomputable def max_ratio (x y : ℕ) (h1 : x > y) (h2 : x + y = 140) : ℚ :=
  if h : y ≠ 0 then (x / y : ℚ) else 0

theorem max_ratio_is_99_over_41 : ∃ (x y : ℕ), x > y ∧ x + y = 140 ∧ max_ratio x y (by sorry) (by sorry) = (99 / 41 : ℚ) :=
by
  sorry

end max_ratio_is_99_over_41_l1826_182657


namespace tangent_line_eq_area_independent_of_a_l1826_182659

open Real

section TangentLineAndArea

def curve (x : ℝ) := x^2 - 1

def tangentCurvey (x : ℝ) := x^2

noncomputable def tangentLine (a : ℝ) (ha : a > 0) : (ℝ → ℝ) :=
  if a > 1 then λ x => (2*(a + 1)) * x - (a+1)^2
  else λ x => (2*(a - 1)) * x - (a-1)^2

theorem tangent_line_eq (a : ℝ) (ha : a > 0) :
  ∃ (line : ℝ → ℝ), (line = tangentLine a ha) :=
sorry

theorem area_independent_of_a (a : ℝ) (ha : a > 0) :
  (∫ x in (a - 1)..a, (tangentCurvey x - tangentLine a ha x)) +
  (∫ x in a..(a + 1), (tangentCurvey x - tangentLine a ha x)) = (2 / 3 : Real) :=
sorry

end TangentLineAndArea

end tangent_line_eq_area_independent_of_a_l1826_182659


namespace speed_boat_in_still_water_l1826_182684

-- Define the conditions
def speed_of_current := 20
def speed_upstream := 30

-- Define the effective speed given conditions
def effective_speed (speed_in_still_water : ℕ) := speed_in_still_water - speed_of_current

-- Theorem stating the problem
theorem speed_boat_in_still_water : 
  ∃ (speed_in_still_water : ℕ), effective_speed speed_in_still_water = speed_upstream ∧ speed_in_still_water = 50 := 
by 
  -- Proof to be filled in
  sorry

end speed_boat_in_still_water_l1826_182684


namespace percentage_of_annual_decrease_is_10_l1826_182651

-- Define the present population and future population
def P_present : ℕ := 500
def P_future : ℕ := 450 

-- Calculate the percentage decrease
def percentage_decrease (P_present P_future : ℕ) : ℕ :=
  ((P_present - P_future) * 100) / P_present

-- Lean statement to prove the percentage decrease is 10%
theorem percentage_of_annual_decrease_is_10 :
  percentage_decrease P_present P_future = 10 :=
by
  unfold percentage_decrease
  sorry

end percentage_of_annual_decrease_is_10_l1826_182651


namespace ratio_problem_l1826_182609

theorem ratio_problem 
  (a b c d : ℚ)
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 :=
by
  sorry

end ratio_problem_l1826_182609


namespace pq_plus_four_mul_l1826_182615

open Real

theorem pq_plus_four_mul {p q : ℝ} (h1 : (x - 4) * (3 * x + 11) = x ^ 2 - 19 * x + 72) 
  (hpq1 : 2 * p ^ 2 + 18 * p - 116 = 0) (hpq2 : 2 * q ^ 2 + 18 * q - 116 = 0) (hpq_ne : p ≠ q) : 
  (p + 4) * (q + 4) = -78 := 
sorry

end pq_plus_four_mul_l1826_182615


namespace find_U_l1826_182630

-- Declare the variables and conditions
def digits : Set ℤ := {1, 2, 3, 4, 5, 6}

theorem find_U (P Q R S T U : ℤ) :
  -- Condition: Digits are distinct and each is in {1, 2, 3, 4, 5, 6}
  (P ∈ digits) ∧ (Q ∈ digits) ∧ (R ∈ digits) ∧ (S ∈ digits) ∧ (T ∈ digits) ∧ (U ∈ digits) ∧
  (P ≠ Q) ∧ (P ≠ R) ∧ (P ≠ S) ∧ (P ≠ T) ∧ (P ≠ U) ∧
  (Q ≠ R) ∧ (Q ≠ S) ∧ (Q ≠ T) ∧ (Q ≠ U) ∧
  (R ≠ S) ∧ (R ≠ T) ∧ (R ≠ U) ∧ (S ≠ T) ∧ (S ≠ U) ∧ (T ≠ U) ∧
  -- Condition: The three-digit number PQR is divisible by 9
  (100 * P + 10 * Q + R) % 9 = 0 ∧
  -- Condition: The three-digit number QRS is divisible by 4
  (10 * Q + R) % 4 = 0 ∧
  -- Condition: The three-digit number RST is divisible by 3
  (10 * R + S) % 3 = 0 ∧
  -- Condition: The sum of the digits is divisible by 5
  (P + Q + R + S + T + U) % 5 = 0
  -- Conclusion: U = 4
  → U = 4 :=
by sorry

end find_U_l1826_182630


namespace percentage_dried_fruit_of_combined_mix_l1826_182656

theorem percentage_dried_fruit_of_combined_mix :
  ∀ (weight_sue weight_jane : ℝ),
  (weight_sue * 0.3 + weight_jane * 0.6) / (weight_sue + weight_jane) = 0.45 →
  100 * (weight_sue * 0.7) / (weight_sue + weight_jane) = 35 :=
by
  intros weight_sue weight_jane H
  sorry

end percentage_dried_fruit_of_combined_mix_l1826_182656


namespace geometric_sequence_S6_l1826_182611

noncomputable def sum_of_first_n_terms (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S6 (a r : ℝ) (h1 : sum_of_first_n_terms a r 2 = 6) (h2 : sum_of_first_n_terms a r 4 = 30) : 
  sum_of_first_n_terms a r 6 = 126 :=
sorry

end geometric_sequence_S6_l1826_182611


namespace fraction_division_l1826_182623

theorem fraction_division (a b c d : ℚ) (h1 : a = 3) (h2 : b = 8) (h3 : c = 5) (h4 : d = 12) :
  (a / b) / (c / d) = 9 / 10 :=
by
  sorry

end fraction_division_l1826_182623


namespace percentage_increase_l1826_182674

-- Defining the problem constants
def price (P : ℝ) : ℝ := P
def assets_A (A : ℝ) : ℝ := A
def assets_B (B : ℝ) : ℝ := B
def percentage (X : ℝ) : ℝ := X

-- Conditions
axiom price_company_B_double_assets : ∀ (P B: ℝ), price P = 2 * assets_B B
axiom price_seventy_five_percent_combined_assets : ∀ (P A B: ℝ), price P = 0.75 * (assets_A A + assets_B B)
axiom price_percentage_more_than_A : ∀ (P A X: ℝ), price P = assets_A A * (1 + percentage X / 100)

-- Theorem to prove
theorem percentage_increase : ∀ (P A B X : ℝ)
  (h1 : price P = 2 * assets_B B)
  (h2 : price P = 0.75 * (assets_A A + assets_B B))
  (h3 : price P = assets_A A * (1 + percentage X / 100)),
  percentage X = 20 :=
by
  intros P A B X h1 h2 h3
  -- Proof steps would go here
  sorry

end percentage_increase_l1826_182674


namespace family_of_four_children_includes_one_boy_one_girl_l1826_182686

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l1826_182686


namespace three_term_inequality_l1826_182667

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l1826_182667


namespace original_pumpkins_count_l1826_182608

def pumpkins_eaten_by_rabbits : ℕ := 23
def pumpkins_left : ℕ := 20
def original_pumpkins : ℕ := pumpkins_left + pumpkins_eaten_by_rabbits

theorem original_pumpkins_count :
  original_pumpkins = 43 :=
sorry

end original_pumpkins_count_l1826_182608


namespace complex_number_problem_l1826_182677

theorem complex_number_problem (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
by {
  -- provide proof here
  sorry
}

end complex_number_problem_l1826_182677


namespace total_pencils_children_l1826_182632

theorem total_pencils_children :
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  c1 + c2 + c3 + c4 + c5 = 60 :=
by
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  show c1 + c2 + c3 + c4 + c5 = 60
  sorry

end total_pencils_children_l1826_182632


namespace circumradius_of_triangle_ABC_l1826_182645

noncomputable def circumradius (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * K)

theorem circumradius_of_triangle_ABC :
  (circumradius 12 10 7 = 6) :=
by
  sorry

end circumradius_of_triangle_ABC_l1826_182645


namespace parallel_vectors_sin_cos_l1826_182679

theorem parallel_vectors_sin_cos (θ : ℝ) (a := (6, 3)) (b := (Real.sin θ, Real.cos θ))
  (h : (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2)) :
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = 2 / 5 :=
by
  sorry

end parallel_vectors_sin_cos_l1826_182679


namespace fraction_product_l1826_182600

theorem fraction_product : (1/2) * (3/5) * (7/11) * (4/13) = 84/1430 := by
  sorry

end fraction_product_l1826_182600


namespace linear_func_is_direct_proportion_l1826_182648

theorem linear_func_is_direct_proportion (m : ℝ) : (∀ x : ℝ, (y : ℝ) → y = m * x + m - 2 → (m - 2 = 0) → y = 0) → m = 2 :=
by
  intros h
  have : m - 2 = 0 := sorry
  exact sorry

end linear_func_is_direct_proportion_l1826_182648


namespace area_AOC_is_1_l1826_182680

noncomputable def point := (ℝ × ℝ) -- Define a point in 2D space

def vector_add (v1 v2 : point) : point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_zero : point := (0, 0)

def scalar_mul (r : ℝ) (v : point) : point :=
  (r * v.1, r * v.2)

def vector_eq (v1 v2 : point) : Prop := 
  v1.1 = v2.1 ∧ v1.2 = v2.2

variables (A B C O : point)
variable (area_ABC : ℝ)

-- Conditions:
-- Point O is a point inside triangle ABC with an area of 4
-- \(\overrightarrow {OA} + \overrightarrow {OB} + 2\overrightarrow {OC} = \overrightarrow {0}\)
axiom condition_area : area_ABC = 4
axiom condition_vector : vector_eq (vector_add (vector_add O A) (vector_add O B)) (scalar_mul (-2) O)

-- Theorem to prove: the area of triangle AOC is 1
theorem area_AOC_is_1 : (area_ABC / 4) = 1 := 
sorry

end area_AOC_is_1_l1826_182680


namespace find_triples_l1826_182698

theorem find_triples (a b c : ℕ) :
  (∃ n : ℕ, 2^a + 2^b + 2^c + 3 = n^2) ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end find_triples_l1826_182698


namespace average_value_of_T_l1826_182699

noncomputable def expected_value_T (B G : ℕ) : ℚ :=
  let total_pairs := 19
  let prob_bg := (B / (B + G)) * (G / (B + G))
  2 * total_pairs * prob_bg

theorem average_value_of_T 
  (B G : ℕ) (hB : B = 8) (hG : G = 12) : 
  expected_value_T B G = 9 :=
by
  rw [expected_value_T, hB, hG]
  norm_num
  sorry

end average_value_of_T_l1826_182699


namespace min_students_with_same_score_l1826_182661

noncomputable def highest_score : ℕ := 83
noncomputable def lowest_score : ℕ := 30
noncomputable def total_students : ℕ := 8000
noncomputable def range_scores : ℕ := (highest_score - lowest_score + 1)

theorem min_students_with_same_score :
  ∃ k : ℕ, k = Nat.ceil (total_students / range_scores) ∧ k = 149 :=
by
  sorry

end min_students_with_same_score_l1826_182661


namespace math_problem_l1826_182691

theorem math_problem :
  (50 - (4050 - 450)) * (4050 - (450 - 50)) = -12957500 := 
by
  sorry

end math_problem_l1826_182691


namespace negation_of_p_l1826_182669

variable (x : ℝ)

def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

theorem negation_of_p : ¬ (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end negation_of_p_l1826_182669


namespace brenda_more_than_jeff_l1826_182694

def emma_amount : ℕ := 8
def daya_amount : ℕ := emma_amount + (emma_amount * 25 / 100)
def jeff_amount : ℕ := (2 / 5) * daya_amount
def brenda_amount : ℕ := 8

theorem brenda_more_than_jeff :
  brenda_amount - jeff_amount = 4 :=
sorry

end brenda_more_than_jeff_l1826_182694


namespace minimum_ticket_cost_correct_l1826_182688

noncomputable def minimum_ticket_cost : Nat :=
let adults := 8
let children := 4
let adult_ticket_price := 100
let child_ticket_price := 50
let group_ticket_price := 70
let group_size := 10
-- Calculate the cost of group tickets for 10 people and regular tickets for 2 children
let total_cost := (group_size * group_ticket_price) + (2 * child_ticket_price)
total_cost

theorem minimum_ticket_cost_correct :
  minimum_ticket_cost = 800 := by
  sorry

end minimum_ticket_cost_correct_l1826_182688


namespace expand_expression_l1826_182670

variable (y : ℝ)

theorem expand_expression : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end expand_expression_l1826_182670


namespace sum_of_digits_base2_315_l1826_182676

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l1826_182676


namespace probability_not_face_card_l1826_182697

-- Definitions based on the conditions
def total_cards : ℕ := 52
def face_cards  : ℕ := 12
def non_face_cards : ℕ := total_cards - face_cards

-- Statement of the theorem
theorem probability_not_face_card : (non_face_cards : ℚ) / (total_cards : ℚ) = 10 / 13 := by
  sorry

end probability_not_face_card_l1826_182697


namespace solve_conjugate_l1826_182624
open Complex

-- Problem definition:
def Z (a : ℝ) : ℂ := ⟨a, 1⟩  -- Z = a + i

def conj_Z (a : ℝ) : ℂ := ⟨a, -1⟩  -- conjugate of Z

theorem solve_conjugate (a : ℝ) (h : Z a + conj_Z a = 4) : conj_Z 2 = 2 - I := by
  sorry

end solve_conjugate_l1826_182624


namespace cylinder_base_area_l1826_182658

-- Definitions: Adding variables and hypotheses based on the problem statement.
variable (A_c A_r : ℝ) -- Base areas of the cylinder and the rectangular prism
variable (h1 : 8 * A_c = 6 * A_r) -- Condition from the rise in water levels
variable (h2 : A_c + A_r = 98) -- Sum of the base areas
variable (h3 : A_c / A_r = 3 / 4) -- Ratio of the base areas

-- Statement: The goal is to prove that the base area of the cylinder is 42.
theorem cylinder_base_area : A_c = 42 :=
by
  sorry

end cylinder_base_area_l1826_182658


namespace Ian_money_left_l1826_182622

-- Definitions based on the conditions
def hours_worked : ℕ := 8
def rate_per_hour : ℕ := 18
def total_money_made : ℕ := hours_worked * rate_per_hour
def money_left : ℕ := total_money_made / 2

-- The statement to be proved 
theorem Ian_money_left : money_left = 72 :=
by
  sorry

end Ian_money_left_l1826_182622


namespace least_number_with_remainder_4_l1826_182654

theorem least_number_with_remainder_4 : ∃ n : ℕ, n = 184 ∧ 
  (∀ d ∈ [5, 9, 12, 18], (n - 4) % d = 0) ∧
  (∀ m : ℕ, (∀ d ∈ [5, 9, 12, 18], (m - 4) % d = 0) → m ≥ n) :=
by
  sorry

end least_number_with_remainder_4_l1826_182654


namespace min_value_of_expression_l1826_182690

theorem min_value_of_expression (α β : ℝ) (h : α + β = π / 2) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 65 := 
sorry

end min_value_of_expression_l1826_182690


namespace no_angle_sat_sin_cos_eq_sin_40_l1826_182625

open Real

theorem no_angle_sat_sin_cos_eq_sin_40 :
  ¬∃ α : ℝ, sin α * cos α = sin (40 * π / 180) := 
by 
  sorry

end no_angle_sat_sin_cos_eq_sin_40_l1826_182625


namespace rate_of_current_is_5_l1826_182627

theorem rate_of_current_is_5 
  (speed_still_water : ℕ)
  (distance_travelled : ℕ)
  (time_travelled : ℚ) 
  (effective_speed_with_current : ℚ) : 
  speed_still_water = 20 ∧ distance_travelled = 5 ∧ time_travelled = 1/5 ∧ 
  effective_speed_with_current = (speed_still_water + 5) →
  effective_speed_with_current * time_travelled = distance_travelled :=
by
  sorry

end rate_of_current_is_5_l1826_182627


namespace cooking_time_l1826_182602

theorem cooking_time
  (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) (remaining_potatoes : ℕ)
  (h_total : total_potatoes = 15)
  (h_cooked : cooked_potatoes = 8)
  (h_remaining_time : remaining_time = 63)
  (h_remaining_potatoes : remaining_potatoes = total_potatoes - cooked_potatoes) :
  remaining_time / remaining_potatoes = 9 :=
by
  sorry

end cooking_time_l1826_182602


namespace Marilyn_has_40_bananas_l1826_182634

-- Definitions of the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- Statement of the proof problem
theorem Marilyn_has_40_bananas : (boxes * bananas_per_box) = 40 := by
  sorry

end Marilyn_has_40_bananas_l1826_182634


namespace seven_digit_number_l1826_182687

theorem seven_digit_number (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
(h1 : a_1 + a_2 = 9)
(h2 : a_2 + a_3 = 7)
(h3 : a_3 + a_4 = 9)
(h4 : a_4 + a_5 = 2)
(h5 : a_5 + a_6 = 8)
(h6 : a_6 + a_7 = 11)
(h_digits : ∀ (i : ℕ), i ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] → i < 10) :
a_1 = 9 ∧ a_2 = 0 ∧ a_3 = 7 ∧ a_4 = 2 ∧ a_5 = 0 ∧ a_6 = 8 ∧ a_7 = 3 :=
by sorry

end seven_digit_number_l1826_182687


namespace probability_prime_and_cube_is_correct_l1826_182638

-- Conditions based on the problem
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_cube (n : ℕ) : Prop :=
  n = 1 ∨ n = 8

def possible_outcomes := 8 * 8
def successful_outcomes := 4 * 2

noncomputable def probability_of_prime_and_cube :=
  (successful_outcomes : ℝ) / (possible_outcomes : ℝ)

theorem probability_prime_and_cube_is_correct :
  probability_of_prime_and_cube = 1 / 8 :=
by
  sorry

end probability_prime_and_cube_is_correct_l1826_182638


namespace area_of_field_l1826_182692

theorem area_of_field (L W A : ℝ) (hL : L = 20) (hP : L + 2 * W = 25) : A = 50 :=
by
  sorry

end area_of_field_l1826_182692


namespace cubeRootThree_expression_value_l1826_182666

-- Define the approximate value of cube root of 3
def cubeRootThree : ℝ := 1.442

-- Lean theorem statement
theorem cubeRootThree_expression_value :
  cubeRootThree - 3 * cubeRootThree - 98 * cubeRootThree = -144.2 := by
  sorry

end cubeRootThree_expression_value_l1826_182666


namespace quadratic_expression_value_l1826_182646

theorem quadratic_expression_value (x₁ x₂ : ℝ) (h₁ : x₁^2 - 3 * x₁ + 1 = 0) (h₂ : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^2 + 3 * x₂ + x₁ * x₂ - 2 = 7 :=
by
  sorry

end quadratic_expression_value_l1826_182646


namespace evaluate_expression_l1826_182641

theorem evaluate_expression :
  12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 :=
by
  sorry

end evaluate_expression_l1826_182641


namespace ratio_15_to_1_l1826_182695

theorem ratio_15_to_1 (x : ℕ) (h : 15 / 1 = x / 10) : x = 150 := 
by sorry

end ratio_15_to_1_l1826_182695


namespace Matt_overall_profit_l1826_182636

def initialValue : ℕ := 8 * 6

def valueGivenAwayTrade1 : ℕ := 2 * 6
def valueReceivedTrade1 : ℕ := 3 * 2 + 9

def valueGivenAwayTrade2 : ℕ := 2 + 6
def valueReceivedTrade2 : ℕ := 2 * 5 + 8

def valueGivenAwayTrade3 : ℕ := 5 + 9
def valueReceivedTrade3 : ℕ := 3 * 3 + 10 + 1

def valueGivenAwayTrade4 : ℕ := 2 * 3 + 8
def valueReceivedTrade4 : ℕ := 2 * 7 + 4

def overallProfit : ℕ :=
  (valueReceivedTrade1 - valueGivenAwayTrade1) +
  (valueReceivedTrade2 - valueGivenAwayTrade2) +
  (valueReceivedTrade3 - valueGivenAwayTrade3) +
  (valueReceivedTrade4 - valueGivenAwayTrade4)

theorem Matt_overall_profit : overallProfit = 23 :=
by
  unfold overallProfit valueReceivedTrade1 valueGivenAwayTrade1 valueReceivedTrade2 valueGivenAwayTrade2 valueReceivedTrade3 valueGivenAwayTrade3 valueReceivedTrade4 valueGivenAwayTrade4
  linarith

end Matt_overall_profit_l1826_182636


namespace quadratic_roots_l1826_182653

theorem quadratic_roots (x : ℝ) : x^2 + 4 * x + 3 = 0 → x = -3 ∨ x = -1 :=
by
  intro h
  have h1 : (x + 3) * (x + 1) = 0 := by sorry
  have h2 : (x = -3 ∨ x = -1) := by sorry
  exact h2

end quadratic_roots_l1826_182653
