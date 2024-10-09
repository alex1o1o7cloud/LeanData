import Mathlib

namespace first_snail_time_proof_l1335_133551

-- Define the conditions
def first_snail_speed := 2 -- speed in feet per minute
def second_snail_speed := 2 * first_snail_speed
def third_snail_speed := 5 * second_snail_speed
def third_snail_time := 2 -- time in minutes
def distance := third_snail_speed * third_snail_time

-- Define the time it took the first snail
def first_snail_time := distance / first_snail_speed

-- Define the theorem to be proven
theorem first_snail_time_proof : first_snail_time = 20 := 
by
  -- Proof should be filled here
  sorry

end first_snail_time_proof_l1335_133551


namespace find_q_l1335_133535

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 17) (h2 : 6 * p + 5 * q = 20) : q = 2 / 11 :=
by
  sorry

end find_q_l1335_133535


namespace point_outside_circle_l1335_133597

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l1335_133597


namespace geometric_sequence_a8_value_l1335_133527

variable {a : ℕ → ℕ}

-- Assuming a is a geometric sequence, provide the condition a_3 * a_9 = 4 * a_4
def geometric_sequence_condition (a : ℕ → ℕ) :=
  (a 3) * (a 9) = 4 * (a 4)

-- Prove that a_8 = 4 under the given condition
theorem geometric_sequence_a8_value (a : ℕ → ℕ) (h : geometric_sequence_condition a) : a 8 = 4 :=
  sorry

end geometric_sequence_a8_value_l1335_133527


namespace smallest_radius_squared_of_sphere_l1335_133518

theorem smallest_radius_squared_of_sphere :
  ∃ (x y z : ℤ), 
  (x - 2)^2 + y^2 + z^2 = (x^2 + (y - 4)^2 + z^2) ∧
  (x - 2)^2 + y^2 + z^2 = (x^2 + y^2 + (z - 6)^2) ∧
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
  (∃ r, r^2 = (x - 2)^2 + (0 - y)^2 + (0 - z)^2) ∧
  51 = r^2 :=
sorry

end smallest_radius_squared_of_sphere_l1335_133518


namespace pattern_C_not_foldable_without_overlap_l1335_133516

-- Define the four patterns, denoted as PatternA, PatternB, PatternC, and PatternD.
inductive Pattern
| A : Pattern
| B : Pattern
| C : Pattern
| D : Pattern

-- Define a predicate for a pattern being foldable into a cube without overlap.
def foldable_into_cube (p : Pattern) : Prop := sorry

theorem pattern_C_not_foldable_without_overlap : ¬ foldable_into_cube Pattern.C := sorry

end pattern_C_not_foldable_without_overlap_l1335_133516


namespace simplify_expression_l1335_133510

theorem simplify_expression :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) *
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 :=
by 
  sorry

end simplify_expression_l1335_133510


namespace red_candies_remain_percentage_l1335_133574

noncomputable def percent_red_candies_remain (N : ℝ) : ℝ :=
let total_initial_candies : ℝ := 5 * N
let green_candies_eat : ℝ := N
let remaining_after_green : ℝ := total_initial_candies - green_candies_eat

let half_orange_candies_eat : ℝ := N / 2
let remaining_after_half_orange : ℝ := remaining_after_green - half_orange_candies_eat

let half_all_remaining_candies_eat : ℝ := (N / 2) + (N / 4) + (N / 2) + (N / 2)
let remaining_after_half_all : ℝ := remaining_after_half_orange - half_all_remaining_candies_eat

let final_remaining_candies : ℝ := 0.32 * total_initial_candies
let candies_to_eat_finally : ℝ := remaining_after_half_all - final_remaining_candies
let each_color_final_eat : ℝ := candies_to_eat_finally / 2

let remaining_red_candies : ℝ := (N / 2) - each_color_final_eat

(remaining_red_candies / N) * 100

theorem red_candies_remain_percentage (N : ℝ) : percent_red_candies_remain N = 42.5 := by
  -- Proof skipped
  sorry

end red_candies_remain_percentage_l1335_133574


namespace dress_designs_count_l1335_133520

-- Define the number of colors, fabric types, and patterns
def num_colors : Nat := 3
def num_fabric_types : Nat := 4
def num_patterns : Nat := 3

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_fabric_types * num_patterns

-- Define the theorem to prove the equivalence
theorem dress_designs_count :
  total_dress_designs = 36 :=
by
  -- This is to show the theorem's structure; proof will be added here.
  sorry

end dress_designs_count_l1335_133520


namespace greatest_common_multiple_of_10_and_15_lt_120_l1335_133503

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end greatest_common_multiple_of_10_and_15_lt_120_l1335_133503


namespace find_t_l1335_133566

theorem find_t (s t : ℤ) (h1 : 9 * s + 5 * t = 108) (h2 : s = t - 2) : t = 9 :=
sorry

end find_t_l1335_133566


namespace remainder_8354_11_l1335_133587

theorem remainder_8354_11 : 8354 % 11 = 6 := sorry

end remainder_8354_11_l1335_133587


namespace smaller_of_two_digit_product_l1335_133584

theorem smaller_of_two_digit_product (a b : ℕ) (h1 : a * b = 4896) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 32 :=
sorry

end smaller_of_two_digit_product_l1335_133584


namespace length_of_platform_l1335_133559

theorem length_of_platform (L : ℕ) :
  (∀ (V : ℚ), V = 600 / 52 → V = (600 + L) / 78) → L = 300 :=
by
  sorry

end length_of_platform_l1335_133559


namespace school_allocation_methods_l1335_133540

-- Define the conditions
def doctors : ℕ := 3
def nurses : ℕ := 6
def schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- The combinatorial function for binomial coefficient
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Verify the number of allocation methods
theorem school_allocation_methods : 
  C doctors doctors_per_school * C nurses nurses_per_school *
  C (doctors - 1) doctors_per_school * C (nurses - 2) nurses_per_school *
  C (doctors - 2) doctors_per_school * C (nurses - 4) nurses_per_school = 540 := 
sorry

end school_allocation_methods_l1335_133540


namespace sum_of_first_15_terms_l1335_133506

variable (a d : ℕ)

def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_15_terms (h : nth_term 4 + nth_term 12 = 16) : sum_of_first_n_terms 15 = 120 :=
by
  sorry

end sum_of_first_15_terms_l1335_133506


namespace y_n_is_square_of_odd_integer_l1335_133538

-- Define the sequences and the initial conditions
def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

-- Helper function to check if a number is odd
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The theorem to prove
theorem y_n_is_square_of_odd_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, y n = k ^ 2 ∧ is_odd k := by
  sorry

end y_n_is_square_of_odd_integer_l1335_133538


namespace smallest_among_5_neg7_0_neg53_l1335_133550

-- Define the rational numbers involved as constants
def a : ℚ := 5
def b : ℚ := -7
def c : ℚ := 0
def d : ℚ := -5 / 3

-- Define the conditions as separate lemmas
lemma positive_greater_than_zero (x : ℚ) (hx : x > 0) : x > c := by sorry
lemma zero_greater_than_negative (x : ℚ) (hx : x < 0) : c > x := by sorry
lemma compare_negative_by_absolute_value (x y : ℚ) (hx : x < 0) (hy : y < 0) (habs : |x| > |y|) : x < y := by sorry

-- Prove the main assertion
theorem smallest_among_5_neg7_0_neg53 : 
    b < a ∧ b < c ∧ b < d := by
    -- Here we apply the defined conditions to show b is the smallest
    sorry

end smallest_among_5_neg7_0_neg53_l1335_133550


namespace consecutive_integer_sum_l1335_133570

theorem consecutive_integer_sum (a b c : ℕ) 
  (h1 : b = a + 2) 
  (h2 : c = a + 4) 
  (h3 : a + c = 140) 
  (h4 : b - a = 2) : a + b + c = 210 := 
sorry

end consecutive_integer_sum_l1335_133570


namespace cougar_ratio_l1335_133575

theorem cougar_ratio (lions tigers total_cats cougars : ℕ) 
  (h_lions : lions = 12) 
  (h_tigers : tigers = 14) 
  (h_total : total_cats = 39) 
  (h_cougars : cougars = total_cats - (lions + tigers)) 
  : cougars * 2 = lions + tigers := 
by 
  rw [h_lions, h_tigers] 
  norm_num at * 
  sorry

end cougar_ratio_l1335_133575


namespace arithmetic_sequence_a6_value_l1335_133578

theorem arithmetic_sequence_a6_value (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_roots : ∀ x, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) :
  a 6 = -6 :=
by
  -- Definitions and given conditions would go here in a fully elaborated proof.
  sorry

end arithmetic_sequence_a6_value_l1335_133578


namespace race_runners_l1335_133545

theorem race_runners (n : ℕ) (h1 : 5 * 8 + (n - 5) * 10 = 70) : n = 8 :=
sorry

end race_runners_l1335_133545


namespace find_general_formula_l1335_133563

theorem find_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h₀ : n > 0)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, S (n + 1) = 2 * S n + n + 1)
  (h₃ : ∀ n, S (n + 1) - S n = a (n + 1)) :
  a n = 2^n - 1 :=
sorry

end find_general_formula_l1335_133563


namespace arithmetic_sequence_fifth_term_l1335_133536

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15)
  (h2 : a + 10 * d = 18) : 
  a + 4 * d = 0 := 
sorry

end arithmetic_sequence_fifth_term_l1335_133536


namespace nonagon_angles_l1335_133553

/-- Determine the angles of the nonagon given specified conditions -/
theorem nonagon_angles (a : ℝ) (x : ℝ) 
  (h_angle_eq : ∀ (AIH BCD HGF : ℝ), AIH = x → BCD = x → HGF = x)
  (h_internal_sum : 7 * 180 = 1260)
  (h_tessellation : x + x + x + (360 - x) + (360 - x) + (360 - x) = 1080) :
  True := sorry

end nonagon_angles_l1335_133553


namespace smallest_x_value_l1335_133593

theorem smallest_x_value : ∃ x : ℤ, ∃ y : ℤ, (xy + 7 * x + 6 * y = -8) ∧ x = -40 :=
by
  sorry

end smallest_x_value_l1335_133593


namespace probability_same_color_l1335_133583

-- Define the total combinations function
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- The given values from the problem
def whiteBalls := 2
def blackBalls := 3
def totalBalls := whiteBalls + blackBalls
def drawnBalls := 2

-- Calculate combinations
def comb_white_2 := comb whiteBalls drawnBalls
def comb_black_2 := comb blackBalls drawnBalls
def comb_total_2 := comb totalBalls drawnBalls

-- The correct answer given in the solution
def correct_probability := 2 / 5

-- Statement for the proof in Lean
theorem probability_same_color : (comb_white_2 + comb_black_2) / comb_total_2 = correct_probability := by
  sorry

end probability_same_color_l1335_133583


namespace math_proof_problem_l1335_133522

-- Define the function and its properties
variable (f : ℝ → ℝ)
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity : ∀ x : ℝ, f (x + 1) = -f x
axiom increasing_on_interval : ∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y

-- Theorem statement expressing the questions and answers
theorem math_proof_problem :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧
  (f 2 = f 0) :=
by
  sorry

end math_proof_problem_l1335_133522


namespace amount_of_silver_l1335_133561

-- Definitions
def total_silver (x : ℕ) : Prop :=
  (x - 4) % 7 = 0 ∧ (x + 8) % 9 = 1

-- Theorem to be proven
theorem amount_of_silver (x : ℕ) (h : total_silver x) : (x - 4)/7 = (x + 8)/9 :=
by sorry

end amount_of_silver_l1335_133561


namespace triangle_PZQ_area_is_50_l1335_133526

noncomputable def area_triangle_PZQ (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) : ℝ :=
  let RS := PQ -- since PQRS is a rectangle, RS = PQ
  let XY := RS - RX - SY
  let height := 2 * QR -- height is doubled due to triangle similarity ratio
  let area := 0.5 * PQ * height
  area

theorem triangle_PZQ_area_is_50 (PQ QR RX SY : ℝ) (hPQ : PQ = 10) (hQR : QR = 5) (hRX : RX = 2) (hSY : SY = 3) :
  area_triangle_PZQ PQ QR RX SY hPQ hQR hRX hSY = 50 :=
  sorry

end triangle_PZQ_area_is_50_l1335_133526


namespace no_difference_of_squares_equals_222_l1335_133586

theorem no_difference_of_squares_equals_222 (a b : ℤ) : a^2 - b^2 ≠ 222 := 
  sorry

end no_difference_of_squares_equals_222_l1335_133586


namespace cone_from_sector_l1335_133542

def cone_can_be_formed (θ : ℝ) (r_sector : ℝ) (r_cone_base : ℝ) (l_slant_height : ℝ) : Prop :=
  θ = 270 ∧ r_sector = 12 ∧ ∃ L, L = θ / 360 * (2 * Real.pi * r_sector) ∧ 2 * Real.pi * r_cone_base = L ∧ l_slant_height = r_sector

theorem cone_from_sector (base_radius slant_height : ℝ) :
  cone_can_be_formed 270 12 base_radius slant_height ↔ base_radius = 9 ∧ slant_height = 12 :=
by
  sorry

end cone_from_sector_l1335_133542


namespace possible_slopes_of_line_intersects_ellipse_l1335_133573

/-- 
A line whose y-intercept is (0, 3) intersects the ellipse 4x^2 + 9y^2 = 36. 
Find all possible slopes of this line. 
-/
theorem possible_slopes_of_line_intersects_ellipse :
  (∀ m : ℝ, ∃ x : ℝ, 4 * x^2 + 9 * (m * x + 3)^2 = 36) ↔ 
  (m <= - (Real.sqrt 5) / 3 ∨ m >= (Real.sqrt 5) / 3) :=
sorry

end possible_slopes_of_line_intersects_ellipse_l1335_133573


namespace simplify_expression_l1335_133567

theorem simplify_expression (n : ℤ) :
  (2 : ℝ) ^ (-(3 * n + 1)) + (2 : ℝ) ^ (-(3 * n - 2)) - 3 * (2 : ℝ) ^ (-3 * n) = (3 / 2) * (2 : ℝ) ^ (-3 * n) :=
by
  sorry

end simplify_expression_l1335_133567


namespace encounter_count_l1335_133572

theorem encounter_count (vA vB d : ℝ) (h₁ : 5 * d / vA = 9 * d / vB) :
  ∃ encounters : ℝ, encounters = 3023 :=
by
  sorry

end encounter_count_l1335_133572


namespace geometric_sequence_property_l1335_133534

-- Define the sequence and the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the main property we are considering
def given_property (a: ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n - 1) = (a n) ^ 2

-- State the theorem
theorem geometric_sequence_property {a : ℕ → ℝ} (n : ℕ) (hn : n ≥ 2) :
  (is_geometric_sequence a → given_property a n ∧ ∀ a, given_property a n → ¬ is_geometric_sequence a) := sorry

end geometric_sequence_property_l1335_133534


namespace trigonometric_unique_solution_l1335_133564

theorem trigonometric_unique_solution :
  (∃ x : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ 0 ≤ y ∧ y < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 ∧
    Real.sin y = 0.6 ∧ Real.cos y = 0.8 → x = y) :=
by
  sorry

end trigonometric_unique_solution_l1335_133564


namespace same_answer_l1335_133529

structure Person :=
(name : String)
(tellsTruth : Bool)

def Fedya : Person :=
{ name := "Fedya",
  tellsTruth := true }

def Vadim : Person :=
{ name := "Vadim",
  tellsTruth := false }

def question (p : Person) (q : String) : Bool :=
if p.tellsTruth then q = p.name else q ≠ p.name

theorem same_answer (q : String) :
  (question Fedya q = question Vadim q) :=
sorry

end same_answer_l1335_133529


namespace min_value_4x2_plus_y2_l1335_133581

theorem min_value_4x2_plus_y2 {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) : 
  4 * x^2 + y^2 ≥ 18 := by
  sorry

end min_value_4x2_plus_y2_l1335_133581


namespace mother_age_is_correct_l1335_133505

variable (D M : ℕ)

theorem mother_age_is_correct:
  (D + 3 = 26) → (M - 5 = 2 * (D - 5)) → M = 41 := by
  intros h1 h2
  sorry

end mother_age_is_correct_l1335_133505


namespace arc_length_of_curve_l1335_133580

noncomputable def arc_length : ℝ :=
∫ t in (0 : ℝ)..(Real.pi / 3),
  (Real.sqrt ((t^2 * Real.cos t)^2 + (t^2 * Real.sin t)^2))

theorem arc_length_of_curve :
  arc_length = (Real.pi^3 / 81) :=
by
  sorry

end arc_length_of_curve_l1335_133580


namespace find_larger_number_l1335_133502

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1355) (h2 : L = 6 * S + 15) : L = 1623 :=
sorry

end find_larger_number_l1335_133502


namespace difference_of_two_smallest_integers_divisors_l1335_133590

theorem difference_of_two_smallest_integers_divisors (n m : ℕ) (h₁ : n > 1) (h₂ : m > 1) 
(h₃ : n % 2 = 1) (h₄ : n % 3 = 1) (h₅ : n % 4 = 1) (h₆ : n % 5 = 1) 
(h₇ : n % 6 = 1) (h₈ : n % 7 = 1) (h₉ : n % 8 = 1) (h₁₀ : n % 9 = 1) 
(h₁₁ : n % 10 = 1) (h₃' : m % 2 = 1) (h₄' : m % 3 = 1) (h₅' : m % 4 = 1) 
(h₆' : m % 5 = 1) (h₇' : m % 6 = 1) (h₈' : m % 7 = 1) (h₉' : m % 8 = 1) 
(h₁₀' : m % 9 = 1) (h₁₁' : m % 10 = 1): m - n = 2520 :=
sorry

end difference_of_two_smallest_integers_divisors_l1335_133590


namespace arithmetic_sequence_term_l1335_133552

theorem arithmetic_sequence_term (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 4 = 6)
    (h2 : 2 * (a 3) - (a 2) = 6)
    (h_sum : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  a 1 = -3 := 
by sorry

end arithmetic_sequence_term_l1335_133552


namespace somu_current_age_l1335_133589

variable (S F : ℕ)

theorem somu_current_age
  (h1 : S = F / 3)
  (h2 : S - 10 = (F - 10) / 5) :
  S = 20 := by
  sorry

end somu_current_age_l1335_133589


namespace total_games_played_l1335_133548

noncomputable def win_ratio : ℝ := 5.5
noncomputable def lose_ratio : ℝ := 4.5
noncomputable def tie_ratio : ℝ := 2.5
noncomputable def rained_out_ratio : ℝ := 1
noncomputable def higher_league_ratio : ℝ := 3.5
noncomputable def lost_games : ℝ := 13.5

theorem total_games_played :
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  total_parts * games_per_part = 51 :=
by
  let total_parts := win_ratio + lose_ratio + tie_ratio + rained_out_ratio + higher_league_ratio
  let games_per_part := lost_games / lose_ratio
  have : total_parts * games_per_part = 51 := sorry
  exact this

end total_games_played_l1335_133548


namespace scholars_number_l1335_133568

theorem scholars_number (n : ℕ) : n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 → n = 509 :=
by
  intro h
  sorry

end scholars_number_l1335_133568


namespace probability_of_diamond_king_ace_l1335_133512

noncomputable def probability_three_cards : ℚ :=
  (11 / 52) * (4 / 51) * (4 / 50) + 
  (1 / 52) * (3 / 51) * (4 / 50) + 
  (1 / 52) * (4 / 51) * (3 / 50)

theorem probability_of_diamond_king_ace :
  probability_three_cards = 284 / 132600 := 
by
  sorry

end probability_of_diamond_king_ace_l1335_133512


namespace find_first_number_l1335_133533

theorem find_first_number (HCF LCM number2 number1 : ℕ) 
    (hcf_condition : HCF = 12) 
    (lcm_condition : LCM = 396) 
    (number2_condition : number2 = 198) 
    (number1_condition : number1 * number2 = HCF * LCM) : 
    number1 = 24 := 
by 
    sorry

end find_first_number_l1335_133533


namespace paving_stones_correct_l1335_133509

def paving_stone_area : ℕ := 3 * 2
def courtyard_breadth : ℕ := 6
def number_of_paving_stones : ℕ := 15
def courtyard_length : ℕ := 15

theorem paving_stones_correct : 
  number_of_paving_stones * paving_stone_area = courtyard_length * courtyard_breadth :=
by
  sorry

end paving_stones_correct_l1335_133509


namespace find_a_of_ellipse_foci_l1335_133525

theorem find_a_of_ellipse_foci (a : ℝ) :
  (∀ x y : ℝ, a^2 * x^2 - (a / 2) * y^2 = 1) →
  (a^2 - (2 / a) = 4) →
  a = (1 - Real.sqrt 5) / 4 :=
by 
  intros h1 h2
  sorry

end find_a_of_ellipse_foci_l1335_133525


namespace expression_value_l1335_133594

theorem expression_value (a b : ℚ) (h₁ : a = -1/2) (h₂ : b = 3/2) : -a - 2 * b^2 + 3 * a * b = -25/4 :=
by
  sorry

end expression_value_l1335_133594


namespace inequality_solution_l1335_133544

theorem inequality_solution (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x > 1) ∧ 
  (0 < a ∧ a < 1 ∧ a^(2*x-1) > (1/a)^(x-2) → x < 1) :=
by {
  sorry
}

end inequality_solution_l1335_133544


namespace part1_part2_part3_l1335_133599

-- Part 1
theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem part2 (a m n : ℤ) (h1 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) (h2 : 0 < a) (h3 : 0 < m) (h4 : 0 < n) : 
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem part3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end part1_part2_part3_l1335_133599


namespace sum_min_values_eq_zero_l1335_133585

-- Definitions of the polynomials
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c*x + d

-- Main theorem statement
theorem sum_min_values_eq_zero (b d : ℝ) :
  let a := -16
  let c := -8
  (-64 + b = 0) ∧ (-16 + d = 0) → (-64 + b + (-16 + d) = 0) :=
by
  intros
  rw [add_assoc]
  sorry

end sum_min_values_eq_zero_l1335_133585


namespace pairs_sold_l1335_133515

theorem pairs_sold (total_sales : ℝ) (avg_price_per_pair : ℝ) (h1 : total_sales = 490) (h2 : avg_price_per_pair = 9.8) :
  total_sales / avg_price_per_pair = 50 :=
by
  rw [h1, h2]
  norm_num

end pairs_sold_l1335_133515


namespace XF_XG_value_l1335_133571

-- Define the given conditions
noncomputable def AB := 4
noncomputable def BC := 3
noncomputable def CD := 7
noncomputable def DA := 9

noncomputable def DX (BD : ℚ) := (1 / 3) * BD
noncomputable def BY (BD : ℚ) := (1 / 4) * BD

-- Variables and points in the problem
variables (BD p q : ℚ)
variables (A B C D X Y E F G : Point)

-- Proof statement
theorem XF_XG_value 
(AB_eq : AB = 4) (BC_eq : BC = 3) (CD_eq : CD = 7) (DA_eq : DA = 9)
(DX_eq : DX BD = (1 / 3) * BD) (BY_eq : BY BD = (1 / 4) * BD)
(AC_BD_prod : p * q = 55) :
  XF * XG = (110 / 9) := 
by
  sorry

end XF_XG_value_l1335_133571


namespace integral_even_odd_l1335_133521

open Real

theorem integral_even_odd (a : ℝ) :
  (∫ x in -a..a, x^2 + sin x) = 18 → a = 3 :=
by
  intros h
  -- We'll skip the proof
  sorry

end integral_even_odd_l1335_133521


namespace largest_digit_B_divisible_by_4_l1335_133598

theorem largest_digit_B_divisible_by_4 :
  ∃ (B : ℕ), B ≤ 9 ∧ ∀ B', (B' ≤ 9 ∧ (4 * 10^5 + B' * 10^4 + 5 * 10^3 + 7 * 10^2 + 8 * 10 + 4) % 4 = 0) → B' ≤ B :=
by
  sorry

end largest_digit_B_divisible_by_4_l1335_133598


namespace total_oak_trees_after_planting_l1335_133500

-- Definitions based on conditions
def initial_oak_trees : ℕ := 5
def new_oak_trees : ℕ := 4

-- Statement of the problem and solution
theorem total_oak_trees_after_planting : initial_oak_trees + new_oak_trees = 9 := by
  sorry

end total_oak_trees_after_planting_l1335_133500


namespace infinite_integers_repr_l1335_133519

theorem infinite_integers_repr : ∀ (k : ℕ), k > 1 →
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
by
  intros k hk
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  sorry

end infinite_integers_repr_l1335_133519


namespace red_ballpoint_pens_count_l1335_133504

theorem red_ballpoint_pens_count (R B : ℕ) (h1: R + B = 240) (h2: B = R - 2) : R = 121 :=
by
  sorry

end red_ballpoint_pens_count_l1335_133504


namespace sum4_l1335_133569

noncomputable def alpha : ℂ := sorry
noncomputable def beta : ℂ := sorry
noncomputable def gamma : ℂ := sorry

axiom sum1 : alpha + beta + gamma = 1
axiom sum2 : alpha^2 + beta^2 + gamma^2 = 5
axiom sum3 : alpha^3 + beta^3 + gamma^3 = 9

theorem sum4 : alpha^4 + beta^4 + gamma^4 = 56 := by
  sorry

end sum4_l1335_133569


namespace find_ratios_sum_l1335_133582
noncomputable def Ana_biking_rate : ℝ := 8.6
noncomputable def Bob_biking_rate : ℝ := 6.2
noncomputable def CAO_biking_rate : ℝ := 5

variable (a b c : ℝ)

-- Conditions  
def Ana_distance := 2 * a + b + c = Ana_biking_rate
def Bob_distance := b + c = Bob_biking_rate
def Cao_distance := Real.sqrt (b^2 + c^2) = CAO_biking_rate

-- Main statement
theorem find_ratios_sum : 
  Ana_distance a b c ∧ 
  Bob_distance b c ∧ 
  Cao_distance b c →
  ∃ (p q r : ℕ), p + q + r = 37 ∧ Nat.gcd p q = 1 ∧ ((a / c) = p / r) ∧ ((b / c) = q / r) ∧ ((a / b) = p / q) :=
sorry

end find_ratios_sum_l1335_133582


namespace arithmetic_sequence_m_value_l1335_133554

theorem arithmetic_sequence_m_value (m : ℝ) (h : 2 + 6 = 2 * m) : m = 4 :=
by sorry

end arithmetic_sequence_m_value_l1335_133554


namespace find_m_and_y_range_l1335_133592

open Set

noncomputable def y (m x : ℝ) := (6 + 2 * m) * x^2 - 5 * x^((abs (m + 2))) + 3 

theorem find_m_and_y_range :
  (∃ m : ℝ, (∀ x : ℝ, y m x = (6 + 2*m) * x^2 - 5*x^((abs (m+2))) + 3) ∧ (∀ x : ℝ, y m x = -5 * x + 3 → m = -3)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 → y (-3) x ∈ Icc (-22 : ℝ) (8 : ℝ)) :=
by
  sorry

end find_m_and_y_range_l1335_133592


namespace infinite_sequence_exists_l1335_133528

noncomputable def has_k_distinct_positive_divisors (n k : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card ≥ k ∧ ∀ d ∈ S, d ∣ n

theorem infinite_sequence_exists :
    ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, 0 < k → ∃ n : ℕ, (a n > 0) ∧ has_k_distinct_positive_divisors (a n ^ 2 + a n + 2023) k) :=
  sorry

end infinite_sequence_exists_l1335_133528


namespace value_of_fraction_pow_l1335_133546

theorem value_of_fraction_pow (a b : ℤ) 
  (h1 : ∀ x, (x^2 + (a + 1)*x + a*b) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 4) : 
  ((1 / 2 : ℚ) ^ (a + 2*b) = 4) :=
sorry

end value_of_fraction_pow_l1335_133546


namespace young_fish_per_pregnant_fish_l1335_133588

-- Definitions based on conditions
def tanks := 3
def fish_per_tank := 4
def total_young_fish := 240

-- Calculations based on conditions
def total_pregnant_fish := tanks * fish_per_tank

-- The proof statement
theorem young_fish_per_pregnant_fish : total_young_fish / total_pregnant_fish = 20 := by
  sorry

end young_fish_per_pregnant_fish_l1335_133588


namespace quadratic_roots_condition_l1335_133541

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k*x^2 + 2*x + 1 = 0 ∧ k*y^2 + 2*y + 1 = 0) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_roots_condition_l1335_133541


namespace find_b_for_smallest_c_l1335_133565

theorem find_b_for_smallest_c (c b : ℝ) (h_c_pos : 0 < c) (h_b_pos : 0 < b)
  (polynomial_condition : ∀ x : ℝ, (x^4 - c*x^3 + b*x^2 - c*x + 1 = 0) → real) :
  c = 4 → b = 6 :=
by
  intros h_c_eq_4
  sorry

end find_b_for_smallest_c_l1335_133565


namespace temperature_difference_l1335_133596

variable (high_temp : ℝ) (low_temp : ℝ)

theorem temperature_difference (h1 : high_temp = 15) (h2 : low_temp = 7) : high_temp - low_temp = 8 :=
by {
  sorry
}

end temperature_difference_l1335_133596


namespace Alyssa_cookie_count_l1335_133530

/--
  Alyssa had some cookies.
  Aiyanna has 140 cookies.
  Aiyanna has 11 more cookies than Alyssa.
  How many cookies does Alyssa have? 
-/
theorem Alyssa_cookie_count 
  (aiyanna_cookies : ℕ) 
  (more_cookies : ℕ)
  (h1 : aiyanna_cookies = 140)
  (h2 : more_cookies = 11)
  (h3 : aiyanna_cookies = alyssa_cookies + more_cookies) :
  alyssa_cookies = 129 := 
sorry

end Alyssa_cookie_count_l1335_133530


namespace coin_problem_l1335_133543

theorem coin_problem : ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5) ∧ n % 9 = 0 :=
by
  sorry

end coin_problem_l1335_133543


namespace moles_of_NaNO3_formed_l1335_133539

/- 
  Define the reaction and given conditions.
  The following assumptions and definitions will directly come from the problem's conditions.
-/

/-- 
  Represents a chemical reaction: 1 molecule of AgNO3,
  1 molecule of NaOH producing 1 molecule of NaNO3 and 1 molecule of AgOH.
-/
def balanced_reaction (agNO3 naOH naNO3 agOH : ℕ) := agNO3 = 1 ∧ naOH = 1 ∧ naNO3 = 1 ∧ agOH = 1

/-- 
  Proves that the number of moles of NaNO3 formed is 1,
  given 1 mole of AgNO3 and 1 mole of NaOH.
-/
theorem moles_of_NaNO3_formed (agNO3 naOH naNO3 agOH : ℕ)
  (h : balanced_reaction agNO3 naOH naNO3 agOH) :
  naNO3 = 1 := 
by
  sorry  -- Proof will be added here later

end moles_of_NaNO3_formed_l1335_133539


namespace at_least_two_same_books_l1335_133560

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def satisfied (n : Nat) : Prop :=
  n / sum_of_digits n = 13

theorem at_least_two_same_books (n1 n2 n3 n4 : Nat) (h1 : satisfied n1) (h2 : satisfied n2) (h3 : satisfied n3) (h4 : satisfied n4) :
  n1 = n2 ∨ n1 = n3 ∨ n1 = n4 ∨ n2 = n3 ∨ n2 = n4 ∨ n3 = n4 :=
sorry

end at_least_two_same_books_l1335_133560


namespace Cindy_coins_l1335_133595

theorem Cindy_coins (n : ℕ) (h1 : ∃ X Y : ℕ, n = X * Y ∧ Y > 1 ∧ Y < n) (h2 : ∀ Y, Y > 1 ∧ Y < n → ¬Y ∣ n → False) : n = 65536 :=
by
  sorry

end Cindy_coins_l1335_133595


namespace range_of_a_l1335_133537

theorem range_of_a (a : ℝ) : (∀ x : ℤ, x > 2 * a - 3 ∧ 2 * (x : ℝ) ≥ 3 * ((x : ℝ) - 2) + 5) ↔ (1 / 2 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l1335_133537


namespace yoongi_age_l1335_133532

theorem yoongi_age
  (H Y : ℕ)
  (h1 : Y = H - 2)
  (h2 : Y + H = 18) :
  Y = 8 :=
by
  sorry

end yoongi_age_l1335_133532


namespace find_b_for_continuity_at_2_l1335_133517

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if h : x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem find_b_for_continuity_at_2 (b : ℝ) : (∀ x, f x b = if x ≤ 2 then 4 * x^2 + 5 else b * x + 3) ∧ 
  (f 2 b = 21) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x b - f 2 b| < ε) → 
  b = 9 :=
by
  sorry

end find_b_for_continuity_at_2_l1335_133517


namespace leo_class_girls_l1335_133523

theorem leo_class_girls (g b : ℕ) 
  (h_ratio : 3 * b = 4 * g) 
  (h_total : g + b = 35) : g = 15 := 
by
  sorry

end leo_class_girls_l1335_133523


namespace factor_polynomial_l1335_133547

theorem factor_polynomial : 
  (∀ x : ℝ, (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)) :=
by
  intro x
  -- Left-hand side
  let lhs := x^2 + 6 * x + 9 - 64 * x^4
  -- Right-hand side after factorization
  let rhs := (-8 * x^2 + x + 3) * (8 * x^2 + x + 3)
  -- Prove the equality
  show lhs = rhs
  sorry

end factor_polynomial_l1335_133547


namespace fraction_of_64_l1335_133558

theorem fraction_of_64 : (7 / 8) * 64 = 56 :=
sorry

end fraction_of_64_l1335_133558


namespace find_y_l1335_133513

theorem find_y (x y : ℤ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 :=
sorry

end find_y_l1335_133513


namespace range_of_a_l1335_133576

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 :=
by 
  sorry

end range_of_a_l1335_133576


namespace triangle_base_angles_eq_l1335_133501

theorem triangle_base_angles_eq
  (A B C C1 C2 : ℝ)
  (h1 : A > B)
  (h2 : C1 = 2 * C2)
  (h3 : A + B + C = 180)
  (h4 : B + C2 = 90)
  (h5 : C = C1 + C2) :
  A = B := by
  sorry

end triangle_base_angles_eq_l1335_133501


namespace find_original_number_l1335_133556

theorem find_original_number : ∃ (N : ℤ), (∃ (k : ℤ), N - 30 = 87 * k) ∧ N = 117 :=
by
  sorry

end find_original_number_l1335_133556


namespace part1_part2_l1335_133549

def y (x : ℝ) : ℝ := -x^2 + 8*x - 7

-- Part (1) Lean statement
theorem part1 : ∀ x : ℝ, x < 4 → y x < y (x + 1) := sorry

-- Part (2) Lean statement
theorem part2 : ∀ x : ℝ, (x < 1 ∨ x > 7) → y x < 0 := sorry

end part1_part2_l1335_133549


namespace bradley_travel_time_l1335_133577

theorem bradley_travel_time (T : ℕ) (h1 : T / 4 = 20) (h2 : T / 3 = 45) : T - 20 = 280 :=
by
  -- Placeholder for proof
  sorry

end bradley_travel_time_l1335_133577


namespace bears_in_stock_initially_l1335_133579

theorem bears_in_stock_initially 
  (shipment_bears : ℕ) (shelf_bears : ℕ) (shelves_used : ℕ)
  (total_bears_shelved : shipment_bears + shelf_bears * shelves_used = 24) : 
  (24 - shipment_bears = 6) :=
by
  exact sorry

end bears_in_stock_initially_l1335_133579


namespace measure_of_angle_B_find_a_and_c_find_perimeter_l1335_133514

theorem measure_of_angle_B (a b c : ℝ) (A B C : ℝ) 
    (h : c / (b - a) = (Real.sin A + Real.sin B) / (Real.sin A + Real.sin C)) 
    (cos_B : Real.cos B = -1 / 2) : B = 2 * Real.pi / 3 :=
by
  sorry

theorem find_a_and_c (a c A C : ℝ) (S : ℝ) 
    (h1 : Real.sin C = 2 * Real.sin A) (h2 : S = 2 * Real.sqrt 3) 
    (A' : a * c = 8) : a = 2 ∧ c = 4 :=
by
  sorry

theorem find_perimeter (a b c : ℝ) 
    (h1 : b = Real.sqrt 3) (h2 : a * c = 1) 
    (h3 : a + c = 2) : a + b + c = 2 + Real.sqrt 3 :=
by
  sorry

end measure_of_angle_B_find_a_and_c_find_perimeter_l1335_133514


namespace angles_of_terminal_side_on_line_y_equals_x_l1335_133562

noncomputable def set_of_angles_on_y_equals_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 180 + 45

theorem angles_of_terminal_side_on_line_y_equals_x (α : ℝ) :
  (∃ k : ℤ, α = k * 360 + 45) ∨ (∃ k : ℤ, α = k * 360 + 225) ↔ set_of_angles_on_y_equals_x α :=
by
  sorry

end angles_of_terminal_side_on_line_y_equals_x_l1335_133562


namespace find_pairs_l1335_133511

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ n : ℕ, (n > 0) ∧ (a = n ∧ b = n) ∨ (a = n ∧ b = 1)) ↔ 
  (a^3 ∣ b^2) ∧ ((b - 1) ∣ (a - 1)) :=
by {
  sorry
}

end find_pairs_l1335_133511


namespace max_int_difference_l1335_133507

theorem max_int_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) : 
  y - x = 5 :=
sorry

end max_int_difference_l1335_133507


namespace fraction_of_students_with_partner_l1335_133557

theorem fraction_of_students_with_partner
  (a b : ℕ)
  (condition1 : ∀ seventh, seventh ≠ 0 → ∀ tenth, tenth ≠ 0 → a * b = 0)
  (condition2 : b / 4 = (3 * a) / 7) :
  (b / 4 + 3 * a / 7) / (b + a) = 6 / 19 :=
by
  sorry

end fraction_of_students_with_partner_l1335_133557


namespace first_place_prize_is_200_l1335_133555

-- Define the conditions from the problem
def total_prize_money : ℤ := 800
def num_winners : ℤ := 18
def second_place_prize : ℤ := 150
def third_place_prize : ℤ := 120
def fourth_to_eighteenth_prize : ℤ := 22
def fourth_to_eighteenth_winners : ℤ := num_winners - 3

-- Define the amount awarded to fourth to eighteenth place winners
def total_fourth_to_eighteenth_prize : ℤ := fourth_to_eighteenth_winners * fourth_to_eighteenth_prize

-- Define the total amount awarded to second and third place winners
def total_second_and_third_prize : ℤ := second_place_prize + third_place_prize

-- Define the total amount awarded to second to eighteenth place winners
def total_second_to_eighteenth_prize : ℤ := total_fourth_to_eighteenth_prize + total_second_and_third_prize

-- Define the amount awarded to first place
def first_place_prize : ℤ := total_prize_money - total_second_to_eighteenth_prize

-- Statement for proof required
theorem first_place_prize_is_200 : first_place_prize = 200 :=
by
  -- Assuming the conditions are correct
  sorry

end first_place_prize_is_200_l1335_133555


namespace ratio_addition_l1335_133508

theorem ratio_addition (x : ℝ) : 
  (2 + x) / (3 + x) = 4 / 5 → x = 2 :=
by
  sorry

end ratio_addition_l1335_133508


namespace quadratic_inequality_solution_set_l1335_133591

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (x^2 - 2 * x < 0) ↔ (0 < x ∧ x < 2) := 
sorry

end quadratic_inequality_solution_set_l1335_133591


namespace boots_cost_more_l1335_133524

theorem boots_cost_more (S B : ℝ) 
  (h1 : 22 * S + 16 * B = 460) 
  (h2 : 8 * S + 32 * B = 560) : B - S = 5 :=
by
  -- Here we provide the statement only, skipping the proof
  sorry

end boots_cost_more_l1335_133524


namespace uma_income_l1335_133531

theorem uma_income
  (x y : ℝ)
  (h1 : 4 * x - 3 * y = 5000)
  (h2 : 3 * x - 2 * y = 5000) :
  4 * x = 20000 :=
by
  sorry

end uma_income_l1335_133531
