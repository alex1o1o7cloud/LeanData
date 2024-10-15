import Mathlib

namespace NUMINAMATH_GPT_geometric_series_properties_l189_18933

noncomputable def first_term := (7 : ℚ) / 8
noncomputable def common_ratio := (-1 : ℚ) / 2

theorem geometric_series_properties : 
  common_ratio = -1 / 2 ∧ 
  (first_term * (1 - common_ratio^4) / (1 - common_ratio)) = 35 / 64 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_series_properties_l189_18933


namespace NUMINAMATH_GPT_least_positive_nine_n_square_twelve_n_cube_l189_18928

theorem least_positive_nine_n_square_twelve_n_cube :
  ∃ (n : ℕ), 0 < n ∧ (∃ (k1 k2 : ℕ), 9 * n = k1^2 ∧ 12 * n = k2^3) ∧ n = 144 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_nine_n_square_twelve_n_cube_l189_18928


namespace NUMINAMATH_GPT_proof1_proof2_monotonically_increasing_interval_l189_18921

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x + Real.pi / 3), Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).fst * (vector_b x).fst + (vector_a x).snd * (vector_b x).snd - 0.5 * Real.cos (2 * x)

theorem proof1 : ∀ x : ℝ, f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

theorem proof2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 3 → -0.5 ≤ f x ∧ f x ≤ 0 :=
sorry

theorem monotonically_increasing_interval (k : ℤ) : 
∃ lb ub : ℝ, lb = Real.pi / 6 + k * Real.pi ∧ ub = 2 * Real.pi / 3 + k * Real.pi ∧ ∀ x : ℝ, lb ≤ x ∧ x ≤ ub → f x = -Real.sin (2 * x + Real.pi / 6) + 0.5 :=
sorry

end NUMINAMATH_GPT_proof1_proof2_monotonically_increasing_interval_l189_18921


namespace NUMINAMATH_GPT_power_of_b_l189_18971

theorem power_of_b (b n : ℕ) (hb : b > 1) (hn : n > 1) (h : ∀ k > 1, ∃ a_k : ℤ, k ∣ (b - a_k ^ n)) :
  ∃ A : ℤ, b = A ^ n :=
by
  sorry

end NUMINAMATH_GPT_power_of_b_l189_18971


namespace NUMINAMATH_GPT_min_packs_for_soda_l189_18997

theorem min_packs_for_soda (max_packs : ℕ) (packs : List ℕ) : 
  let num_cans := 95
  let max_each_pack := 4
  let pack_8 := packs.count 8 
  let pack_15 := packs.count 15
  let pack_18 := packs.count 18
  pack_8 ≤ max_each_pack ∧ pack_15 ≤ max_each_pack ∧ pack_18 ≤ max_each_pack ∧ 
  pack_8 * 8 + pack_15 * 15 + pack_18 * 18 = num_cans ∧ 
  pack_8 + pack_15 + pack_18 = max_packs → max_packs = 6 :=
sorry

end NUMINAMATH_GPT_min_packs_for_soda_l189_18997


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l189_18977

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l189_18977


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l189_18935

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l189_18935


namespace NUMINAMATH_GPT_negative_integer_example_l189_18953

def is_negative_integer (n : ℤ) := n < 0

theorem negative_integer_example : is_negative_integer (-2) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_negative_integer_example_l189_18953


namespace NUMINAMATH_GPT_additional_books_l189_18963

theorem additional_books (initial_books total_books additional_books : ℕ)
  (h_initial : initial_books = 54)
  (h_total : total_books = 77) :
  additional_books = total_books - initial_books :=
by
  sorry

end NUMINAMATH_GPT_additional_books_l189_18963


namespace NUMINAMATH_GPT_fish_per_person_l189_18940

theorem fish_per_person (eyes_per_fish : ℕ) (fish_caught : ℕ) (total_eyes : ℕ) (dog_eyes : ℕ) (oomyapeck_eyes : ℕ) (n_people : ℕ) :
  total_eyes = oomyapeck_eyes + dog_eyes →
  total_eyes = fish_caught * eyes_per_fish →
  n_people = 3 →
  oomyapeck_eyes = 22 →
  dog_eyes = 2 →
  eyes_per_fish = 2 →
  fish_caught / n_people = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_fish_per_person_l189_18940


namespace NUMINAMATH_GPT_repeating_decimals_expr_as_fraction_l189_18996

-- Define the repeating decimals as fractions
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Lean 4 statement to prove the equivalence
theorem repeating_decimals_expr_as_fraction : a - b + c = 186 / 333 := by
  sorry

end NUMINAMATH_GPT_repeating_decimals_expr_as_fraction_l189_18996


namespace NUMINAMATH_GPT_theater_seats_l189_18994

theorem theater_seats
  (A : ℕ) -- Number of adult tickets
  (C : ℕ) -- Number of child tickets
  (hC : C = 63) -- 63 child tickets sold
  (total_revenue : ℕ) -- Total Revenue
  (hRev : total_revenue = 519) -- Total revenue is 519
  (adult_ticket_price : ℕ := 12) -- Price per adult ticket
  (child_ticket_price : ℕ := 5) -- Price per child ticket
  (hRevEq : adult_ticket_price * A + child_ticket_price * C = total_revenue) -- Revenue equation
  : A + C = 80 := sorry

end NUMINAMATH_GPT_theater_seats_l189_18994


namespace NUMINAMATH_GPT_p_sufficient_for_not_q_l189_18998

variable (x : ℝ)
def p : Prop := 0 < x ∧ x ≤ 1
def q : Prop := 1 / x < 1

theorem p_sufficient_for_not_q : p x → ¬q x :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_for_not_q_l189_18998


namespace NUMINAMATH_GPT_base6_arithmetic_l189_18910

def base6_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  let n4 := n3 / 10
  let d4 := n4 % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0

def base10_to_base6 (n : ℕ) : ℕ :=
  let b4 := n / 6^4
  let r4 := n % 6^4
  let b3 := r4 / 6^3
  let r3 := r4 % 6^3
  let b2 := r3 / 6^2
  let r2 := r3 % 6^2
  let b1 := r2 / 6^1
  let b0 := r2 % 6^1
  b4 * 10000 + b3 * 1000 + b2 * 100 + b1 * 10 + b0

theorem base6_arithmetic : 
  base10_to_base6 ((base6_to_base10 45321 - base6_to_base10 23454) + base6_to_base10 14553) = 45550 :=
by
  sorry

end NUMINAMATH_GPT_base6_arithmetic_l189_18910


namespace NUMINAMATH_GPT_kathleen_allowance_l189_18936

theorem kathleen_allowance (x : ℝ) (h1 : Kathleen_middleschool_allowance = x + 2)
(h2 : Kathleen_senior_allowance = 5 + 2 * (x + 2))
(h3 : Kathleen_senior_allowance = 2.5 * Kathleen_middleschool_allowance) :
x = 8 :=
by sorry

end NUMINAMATH_GPT_kathleen_allowance_l189_18936


namespace NUMINAMATH_GPT_other_acute_angle_is_60_l189_18922

theorem other_acute_angle_is_60 (a b c : ℝ) (h_triangle : a + b + c = 180) (h_right : c = 90) (h_acute : a = 30) : b = 60 :=
by 
  -- inserting proof later
  sorry

end NUMINAMATH_GPT_other_acute_angle_is_60_l189_18922


namespace NUMINAMATH_GPT_max_product_of_roots_l189_18931

noncomputable def max_prod_roots_m : ℝ :=
  let m := 4.5
  m

theorem max_product_of_roots (m : ℕ) (h : 36 - 8 * m ≥ 0) : m = max_prod_roots_m :=
  sorry

end NUMINAMATH_GPT_max_product_of_roots_l189_18931


namespace NUMINAMATH_GPT_range_of_x_when_a_equals_1_range_of_a_l189_18911

variable {a x : ℝ}

-- Definitions for conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part (1): Prove the range of x when a = 1 and p ∨ q is true.
theorem range_of_x_when_a_equals_1 (h : a = 1) (h1 : p 1 x ∨ q x) : 1 < x ∧ x < 3 :=
by sorry

-- Part (2): Prove the range of a when p is a necessary but not sufficient condition for q.
theorem range_of_a (h2 : ∀ x, q x → p a x) (h3 : ¬ ∀ x, p a x → q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_range_of_x_when_a_equals_1_range_of_a_l189_18911


namespace NUMINAMATH_GPT_width_of_foil_covered_prism_l189_18901

theorem width_of_foil_covered_prism (L W H : ℝ) 
    (hW1 : W = 2 * L)
    (hW2 : W = 2 * H)
    (hvol : L * W * H = 128) :
    W + 2 = 8 := 
sorry

end NUMINAMATH_GPT_width_of_foil_covered_prism_l189_18901


namespace NUMINAMATH_GPT_sum_b_n_l189_18916

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), (∀ n : ℕ, a (n + 1) = q * a n)

theorem sum_b_n (h_geo : is_geometric a) (h_a1 : a 1 = 3) (h_sum_a : ∑' n, a n = 9) (h_bn : ∀ n, b n = a (2 * n)) :
  ∑' n, b n = 18 / 5 :=
sorry

end NUMINAMATH_GPT_sum_b_n_l189_18916


namespace NUMINAMATH_GPT_minimize_expression_l189_18943

variable (a b c : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : a ≠ 0)

theorem minimize_expression : 
  (a > b) → (b > c) → (a ≠ 0) → 
  ∃ x : ℝ, x = 4 ∧ ∀ y, y = (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 → x ≤ y := sorry

end NUMINAMATH_GPT_minimize_expression_l189_18943


namespace NUMINAMATH_GPT_rose_part_payment_l189_18946

-- Defining the conditions
def total_cost (T : ℝ) := 0.95 * T = 5700
def part_payment (x : ℝ) (T : ℝ) := x = 0.05 * T

-- The proof problem: Prove that the part payment Rose made is $300
theorem rose_part_payment : ∃ T x, total_cost T ∧ part_payment x T ∧ x = 300 :=
by
  sorry

end NUMINAMATH_GPT_rose_part_payment_l189_18946


namespace NUMINAMATH_GPT_Rickey_took_30_minutes_l189_18960

variables (R P : ℝ)

-- Define the conditions
def Prejean_speed_is_three_quarters_of_Rickey := P = 4 / 3 * R
def total_time_is_70 := R + P = 70

-- Define the statement to prove
theorem Rickey_took_30_minutes 
  (h1 : Prejean_speed_is_three_quarters_of_Rickey R P) 
  (h2 : total_time_is_70 R P) : R = 30 :=
by
  sorry

end NUMINAMATH_GPT_Rickey_took_30_minutes_l189_18960


namespace NUMINAMATH_GPT_original_price_of_book_l189_18974

-- Define the conditions as Lean 4 statements
variable (P : ℝ)  -- Original price of the book
variable (P_new : ℝ := 480)  -- New price of the book
variable (increase_percentage : ℝ := 0.60)  -- Percentage increase in the price

-- Prove the question: original price equals to $300
theorem original_price_of_book :
  P + increase_percentage * P = P_new → P = 300 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_book_l189_18974


namespace NUMINAMATH_GPT_simplify_expression_l189_18981

theorem simplify_expression :
  (1 / (Real.sqrt 8 + Real.sqrt 11) +
   1 / (Real.sqrt 11 + Real.sqrt 14) +
   1 / (Real.sqrt 14 + Real.sqrt 17) +
   1 / (Real.sqrt 17 + Real.sqrt 20) +
   1 / (Real.sqrt 20 + Real.sqrt 23) +
   1 / (Real.sqrt 23 + Real.sqrt 26) +
   1 / (Real.sqrt 26 + Real.sqrt 29) +
   1 / (Real.sqrt 29 + Real.sqrt 32)) = 
  (2 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l189_18981


namespace NUMINAMATH_GPT_sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l189_18957

-- Problem 1: Define the sum of the first n odd numbers and prove it equals n^2 when n = 5.
theorem sum_first_five_odds_equals_25 : (1 + 3 + 5 + 7 + 9 = 5^2) := 
sorry

-- Problem 2: Prove that if the smallest number in the decomposition of m^3 is 21, then m = 5.
theorem smallest_in_cube_decomposition_eq_21 : 
  (∃ m : ℕ, m > 0 ∧ 21 = 2 * m - 1 ∧ m = 5) := 
sorry

end NUMINAMATH_GPT_sum_first_five_odds_equals_25_smallest_in_cube_decomposition_eq_21_l189_18957


namespace NUMINAMATH_GPT_six_digit_number_divisible_by_504_l189_18905

theorem six_digit_number_divisible_by_504 : 
  ∃ a b c : ℕ, (523 * 1000 + 100 * a + 10 * b + c) % 504 = 0 := by 
sorry

end NUMINAMATH_GPT_six_digit_number_divisible_by_504_l189_18905


namespace NUMINAMATH_GPT_line_positional_relationship_l189_18955

variables {Point Line Plane : Type}

-- Definitions of the conditions
def is_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
def is_within_plane (b : Line) (α : Plane) : Prop := sorry
def no_common_point (a b : Line) : Prop := sorry
def parallel_or_skew (a b : Line) : Prop := sorry

-- Proof statement in Lean
theorem line_positional_relationship
  (a b : Line) (α : Plane)
  (h₁ : is_parallel_to_plane a α)
  (h₂ : is_within_plane b α)
  (h₃ : no_common_point a b) :
  parallel_or_skew a b :=
sorry

end NUMINAMATH_GPT_line_positional_relationship_l189_18955


namespace NUMINAMATH_GPT_xyz_eq_neg10_l189_18959

noncomputable def complex_numbers := {z : ℂ // z ≠ 0}

variables (a b c x y z : complex_numbers)

def condition1 := a.val = (b.val + c.val) / (x.val - 3)
def condition2 := b.val = (a.val + c.val) / (y.val - 3)
def condition3 := c.val = (a.val + b.val) / (z.val - 3)
def condition4 := x.val * y.val + x.val * z.val + y.val * z.val = 9
def condition5 := x.val + y.val + z.val = 6

theorem xyz_eq_neg10 (a b c x y z : complex_numbers) :
  condition1 a b c x ∧ condition2 a b c y ∧ condition3 a b c z ∧
  condition4 x y z ∧ condition5 x y z → x.val * y.val * z.val = -10 :=
by sorry

end NUMINAMATH_GPT_xyz_eq_neg10_l189_18959


namespace NUMINAMATH_GPT_triangle_area_CO_B_l189_18932

-- Define the conditions as given in the problem
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def Q : Point := ⟨0, 15⟩

variable (p : ℝ)
def C : Point := ⟨0, p⟩
def B : Point := ⟨15, 0⟩

-- Prove the area of triangle COB is 15p / 2
theorem triangle_area_CO_B :
  p ≥ 0 → p ≤ 15 → 
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  area = (15 * p) / 2 := 
by
  intros hp0 hp15
  let base := 15
  let height := p
  let area := (1 / 2) * base * height
  have : area = (15 * p) / 2 := sorry
  exact this

end NUMINAMATH_GPT_triangle_area_CO_B_l189_18932


namespace NUMINAMATH_GPT_parker_daily_earning_l189_18942

-- Definition of conditions
def total_earned : ℕ := 2646
def weeks_worked : ℕ := 6
def days_per_week : ℕ := 7
def total_days (weeks : ℕ) (days_in_week : ℕ) : ℕ := weeks * days_in_week

-- Proof statement
theorem parker_daily_earning (h : total_days weeks_worked days_per_week = 42) : (total_earned / 42) = 63 :=
by
  sorry

end NUMINAMATH_GPT_parker_daily_earning_l189_18942


namespace NUMINAMATH_GPT_hyperbola_asymptote_l189_18917

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x, - y^2 = - x^2 / a^2 + 1) ∧ 
  (∀ x y, y + 2 * x = 0) → 
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l189_18917


namespace NUMINAMATH_GPT_road_length_10_trees_10_intervals_l189_18958

theorem road_length_10_trees_10_intervals 
  (n_trees : ℕ) (n_intervals : ℕ) (tree_interval : ℕ) 
  (h_trees : n_trees = 10) (h_intervals : n_intervals = 9) (h_interval_length : tree_interval = 10) : 
  n_intervals * tree_interval = 90 := 
by 
  sorry

end NUMINAMATH_GPT_road_length_10_trees_10_intervals_l189_18958


namespace NUMINAMATH_GPT_inequality_sum_squares_products_l189_18907

theorem inequality_sum_squares_products {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_inequality_sum_squares_products_l189_18907


namespace NUMINAMATH_GPT_imaginary_roots_iff_l189_18930

theorem imaginary_roots_iff {k m : ℝ} (hk : k ≠ 0) : (exists (x : ℝ), k * x^2 + m * x + k = 0 ∧ ∃ (y : ℝ), y * 0 = 0 ∧ y ≠ 0) ↔ m ^ 2 < 4 * k ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_roots_iff_l189_18930


namespace NUMINAMATH_GPT_smallest_four_digit_number_l189_18939

theorem smallest_four_digit_number : 
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (∃ (AB CD : ℕ), 
      n = 1000 * (AB / 10) + 100 * (AB % 10) + CD ∧
      ((AB / 10) * 10 + (AB % 10) + 2) * CD = 100 ∧ 
      n / CD = ((AB / 10) * 10 + (AB % 10) + 1)^2) ∧
    n = 1805 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_number_l189_18939


namespace NUMINAMATH_GPT_team_A_champion_probability_l189_18918

/-- Teams A and B are playing a volleyball match.
Team A needs to win one more game to become the champion, while Team B needs to win two more games to become the champion.
The probability of each team winning each game is 0.5. -/
theorem team_A_champion_probability :
  let p_win := (0.5 : ℝ)
  let prob_A_champion := 1 - p_win * p_win
  prob_A_champion = 0.75 := by
  sorry

end NUMINAMATH_GPT_team_A_champion_probability_l189_18918


namespace NUMINAMATH_GPT_workers_complete_time_l189_18934

theorem workers_complete_time
  (A : ℝ) -- Total work
  (x1 x2 x3 : ℝ) -- Productivities of the workers
  (h1 : x3 = (x1 + x2) / 2)
  (h2 : 10 * x1 = 15 * x2) :
  (A / x1 = 50) ∧ (A / x2 = 75) ∧ (A / x3 = 60) :=
by
  sorry  -- Proof not required

end NUMINAMATH_GPT_workers_complete_time_l189_18934


namespace NUMINAMATH_GPT_range_of_g_l189_18976

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f (x))))

theorem range_of_g : ∀ x, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_range_of_g_l189_18976


namespace NUMINAMATH_GPT_range_of_a_l189_18929

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0 ↔ a ∈ Set.Ioo (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l189_18929


namespace NUMINAMATH_GPT_Rover_has_46_spots_l189_18954

theorem Rover_has_46_spots (G C R : ℕ) 
  (h1 : G = 5 * C)
  (h2 : C = (1/2 : ℝ) * R - 5)
  (h3 : G + C = 108) : 
  R = 46 :=
by
  sorry

end NUMINAMATH_GPT_Rover_has_46_spots_l189_18954


namespace NUMINAMATH_GPT_S_21_equals_4641_l189_18973

-- Define the first element of the nth set
def first_element_of_set (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

-- Define the last element of the nth set
def last_element_of_set (n : ℕ) : ℕ :=
  (first_element_of_set n) + n - 1

-- Define the sum of the nth set
def S (n : ℕ) : ℕ :=
  n * ((first_element_of_set n) + (last_element_of_set n)) / 2

-- The goal statement we want to prove
theorem S_21_equals_4641 : S 21 = 4641 := by
  sorry

end NUMINAMATH_GPT_S_21_equals_4641_l189_18973


namespace NUMINAMATH_GPT_geom_series_common_ratio_l189_18952

theorem geom_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hNewS : (ar^3) / (1 - r) = S / 27) : r = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geom_series_common_ratio_l189_18952


namespace NUMINAMATH_GPT_sum_of_coefficients_no_y_l189_18992

-- Defining the problem conditions
def expansion (a b c : ℤ) (n : ℕ) : ℤ := (a - b + c)^n

-- Summing the coefficients of the terms that do not contain y
noncomputable def coefficients_sum (a b : ℤ) (n : ℕ) : ℤ :=
  (a - b)^n

theorem sum_of_coefficients_no_y (n : ℕ) (h : 0 < n) : 
  coefficients_sum 4 3 n = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_no_y_l189_18992


namespace NUMINAMATH_GPT_sci_not_218000_l189_18903

theorem sci_not_218000 : 218000 = 2.18 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_sci_not_218000_l189_18903


namespace NUMINAMATH_GPT_subtraction_888_55_555_55_l189_18985

theorem subtraction_888_55_555_55 : 888.88 - 555.55 = 333.33 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_888_55_555_55_l189_18985


namespace NUMINAMATH_GPT_find_range_f_l189_18919

noncomputable def greatestIntegerLessEqual (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def f (x y : ℝ) : ℝ :=
  (x + y) / (greatestIntegerLessEqual x * greatestIntegerLessEqual y + greatestIntegerLessEqual x + greatestIntegerLessEqual y + 1)

theorem find_range_f (x y : ℝ) (h1: 0 < x) (h2: 0 < y) (h3: x * y = 1) : 
  ∃ r : ℝ, r = f x y := 
by
  sorry

end NUMINAMATH_GPT_find_range_f_l189_18919


namespace NUMINAMATH_GPT_least_subtracted_number_l189_18969

def is_sum_of_digits_at_odd_places (n : ℕ) : ℕ :=
  (n / 100000) % 10 + (n / 1000) % 10 + (n / 10) % 10

def is_sum_of_digits_at_even_places (n : ℕ) : ℕ :=
  (n / 10000) % 10 + (n / 100) % 10 + (n % 10)

def diff_digits_odd_even (n : ℕ) : ℕ :=
  is_sum_of_digits_at_odd_places n - is_sum_of_digits_at_even_places n

theorem least_subtracted_number :
  ∃ x : ℕ, (427398 - x) % 11 = 0 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_least_subtracted_number_l189_18969


namespace NUMINAMATH_GPT_S10_value_l189_18978

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ :=
  x^m + (1 / x)^m

theorem S10_value (x : ℝ) (h : x + 1/x = 5) : 
  S_m x 10 = 6430223 := by 
  sorry

end NUMINAMATH_GPT_S10_value_l189_18978


namespace NUMINAMATH_GPT_percent_Asian_in_West_l189_18944

noncomputable def NE_Asian := 2
noncomputable def MW_Asian := 2
noncomputable def South_Asian := 2
noncomputable def West_Asian := 6

noncomputable def total_Asian := NE_Asian + MW_Asian + South_Asian + West_Asian

theorem percent_Asian_in_West (h1 : total_Asian = 12) : (West_Asian / total_Asian) * 100 = 50 := 
by sorry

end NUMINAMATH_GPT_percent_Asian_in_West_l189_18944


namespace NUMINAMATH_GPT_compare_trig_values_l189_18913

noncomputable def a : ℝ := Real.tan (-7 * Real.pi / 6)
noncomputable def b : ℝ := Real.cos (23 * Real.pi / 4)
noncomputable def c : ℝ := Real.sin (-33 * Real.pi / 4)

theorem compare_trig_values : c < a ∧ a < b := sorry

end NUMINAMATH_GPT_compare_trig_values_l189_18913


namespace NUMINAMATH_GPT_problem1_problem2_l189_18902

-- Problem 1: Prove that (2a^2 b) * a b^2 / 4a^3 = 1/2 b^3
theorem problem1 (a b : ℝ) : (2 * a^2 * b) * (a * b^2) / (4 * a^3) = (1 / 2) * b^3 :=
  sorry

-- Problem 2: Prove that (2x + 5)(x - 3) = 2x^2 - x - 15
theorem problem2 (x : ℝ): (2 * x + 5) * (x - 3) = 2 * x^2 - x - 15 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l189_18902


namespace NUMINAMATH_GPT_caitlin_bracelets_l189_18941

/-- 
Caitlin makes bracelets to sell at the farmer’s market every weekend. 
Each bracelet takes twice as many small beads as it does large beads. 
If each bracelet uses 12 large beads, and Caitlin has 528 beads with equal amounts of large and small beads, 
prove that Caitlin can make 11 bracelets for this weekend.
-/
theorem caitlin_bracelets (total_beads large_beads_per_bracelet small_beads_per_bracelet total_large_beads total_small_beads bracelets : ℕ)
  (h1 : total_beads = 528)
  (h2 : total_beads = total_large_beads + total_small_beads)
  (h3 : total_large_beads = total_small_beads)
  (h4 : large_beads_per_bracelet = 12)
  (h5 : small_beads_per_bracelet = 2 * large_beads_per_bracelet)
  (h6 : bracelets = total_small_beads / small_beads_per_bracelet) : 
  bracelets = 11 := 
by {
  sorry
}

end NUMINAMATH_GPT_caitlin_bracelets_l189_18941


namespace NUMINAMATH_GPT_billiard_angle_correct_l189_18937

-- Definitions for the problem conditions
def center_O : ℝ × ℝ := (0, 0)
def point_P : ℝ × ℝ := (0.5, 0)
def radius : ℝ := 1

-- The angle to be proven
def strike_angle (α x : ℝ) := x = (90 - 2 * α)

-- Main theorem statement
theorem billiard_angle_correct :
  ∃ α x : ℝ, (strike_angle α x) ∧ x = 47 + (4 / 60) :=
sorry

end NUMINAMATH_GPT_billiard_angle_correct_l189_18937


namespace NUMINAMATH_GPT_inequality_sqrt_sum_ge_2_l189_18927
open Real

theorem inequality_sqrt_sum_ge_2 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  sqrt (a^3 / (1 + b * c)) + sqrt (b^3 / (1 + a * c)) + sqrt (c^3 / (1 + a * b)) ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_sqrt_sum_ge_2_l189_18927


namespace NUMINAMATH_GPT_pizza_ratio_l189_18904

/-- Define a function that represents the ratio calculation -/
def ratio (a b : ℕ) : ℕ × ℕ := (a / (Nat.gcd a b), b / (Nat.gcd a b))

/-- State the main problem to be proved -/
theorem pizza_ratio (total_slices friend_eats james_eats remaining_slices gcd : ℕ)
  (h1 : total_slices = 8)
  (h2 : friend_eats = 2)
  (h3 : james_eats = 3)
  (h4 : remaining_slices = total_slices - friend_eats)
  (h5 : gcd = Nat.gcd james_eats remaining_slices)
  (h6 : ratio james_eats remaining_slices = (1, 2)) :
  ratio james_eats remaining_slices = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_pizza_ratio_l189_18904


namespace NUMINAMATH_GPT_photo_arrangements_l189_18967

-- The description of the problem conditions translated into definitions
def num_positions := 6  -- Total positions (1 teacher + 5 students)

def teacher_positions := 4  -- Positions where teacher can stand (not at either end)

def student_permutations : ℕ := Nat.factorial 5  -- Number of ways to arrange 5 students

-- The total number of valid arrangements where the teacher does not stand at either end
def total_valid_arrangements : ℕ := teacher_positions * student_permutations

-- Statement to be proven
theorem photo_arrangements:
  total_valid_arrangements = 480 :=
by
  sorry

end NUMINAMATH_GPT_photo_arrangements_l189_18967


namespace NUMINAMATH_GPT_derivative_at_0_l189_18925

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x * Real.sin x - 7 * x

theorem derivative_at_0 : deriv f 0 = -6 := 
by
  sorry

end NUMINAMATH_GPT_derivative_at_0_l189_18925


namespace NUMINAMATH_GPT_Aimee_escalator_time_l189_18948

theorem Aimee_escalator_time (d : ℝ) (v_esc : ℝ) (v_walk : ℝ) :
  v_esc = d / 60 → v_walk = d / 90 → (d / (v_esc + v_walk)) = 36 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_Aimee_escalator_time_l189_18948


namespace NUMINAMATH_GPT_plane_equation_l189_18980

def point := ℝ × ℝ × ℝ
def vector := ℝ × ℝ × ℝ

def point_on_plane (P : point) (a b c d : ℝ) : Prop :=
  match P with
  | (x, y, z) => a * x + b * y + c * z + d = 0

def normal_to_plane (n : vector) (a b c : ℝ) : Prop :=
  match n with
  | (nx, ny, nz) => (a, b, c) = (nx, ny, nz)

theorem plane_equation
  (P₀ : point) (u : vector)
  (x₀ y₀ z₀ : ℝ) (a b c d : ℝ)
  (h1 : P₀ = (1, 2, 1))
  (h2 : u = (-2, 1, 3))
  (h3 : point_on_plane (1, 2, 1) a b c d)
  (h4 : normal_to_plane (-2, 1, 3) a b c)
  : (2 : ℝ) * (x₀ : ℝ) - (y₀ : ℝ) - (3 : ℝ) * (z₀ : ℝ) + (3 : ℝ) = 0 :=
sorry

end NUMINAMATH_GPT_plane_equation_l189_18980


namespace NUMINAMATH_GPT_find_other_root_l189_18951

theorem find_other_root (b : ℝ) (h : ∀ x : ℝ, x^2 - b * x + 3 = 0 → x = 3 ∨ ∃ y, y = 1) :
  ∃ y, y = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_other_root_l189_18951


namespace NUMINAMATH_GPT_factor_M_l189_18983

theorem factor_M (a b c d : ℝ) : 
  ((a - c)^2 + (b - d)^2) * (a^2 + b^2) - (a * d - b * c)^2 =
  (a * c + b * d - a^2 - b^2)^2 :=
by
  sorry

end NUMINAMATH_GPT_factor_M_l189_18983


namespace NUMINAMATH_GPT_ratio_of_triangle_side_to_rectangle_width_l189_18962

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end NUMINAMATH_GPT_ratio_of_triangle_side_to_rectangle_width_l189_18962


namespace NUMINAMATH_GPT_copy_pages_count_l189_18956

-- Definitions and conditions
def cost_per_page : ℕ := 5  -- Cost per page in cents
def total_money : ℕ := 50 * 100  -- Total money in cents

-- Proof goal
theorem copy_pages_count : total_money / cost_per_page = 1000 := 
by sorry

end NUMINAMATH_GPT_copy_pages_count_l189_18956


namespace NUMINAMATH_GPT_mary_age_proof_l189_18990

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end NUMINAMATH_GPT_mary_age_proof_l189_18990


namespace NUMINAMATH_GPT_prob_first_three_heads_all_heads_l189_18993

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end NUMINAMATH_GPT_prob_first_three_heads_all_heads_l189_18993


namespace NUMINAMATH_GPT_range_of_f_l189_18995

def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f :
  Set.range f = {0, -1} := 
sorry

end NUMINAMATH_GPT_range_of_f_l189_18995


namespace NUMINAMATH_GPT_summer_camp_students_l189_18950

theorem summer_camp_students (x : ℕ)
  (h1 : (1 / 6) * x = n_Shanghai)
  (h2 : n_Tianjin = 24)
  (h3 : (1 / 4) * x = n_Chongqing)
  (h4 : n_Beijing = (3 / 2) * (n_Shanghai + n_Tianjin)) :
  x = 180 :=
by
  sorry

end NUMINAMATH_GPT_summer_camp_students_l189_18950


namespace NUMINAMATH_GPT_sickness_temperature_increase_l189_18961

theorem sickness_temperature_increase :
  ∀ (normal_temp fever_threshold current_temp : ℕ), normal_temp = 95 → fever_threshold = 100 →
  current_temp = fever_threshold + 5 → (current_temp - normal_temp = 10) :=
by
  intros normal_temp fever_threshold current_temp h1 h2 h3
  sorry

end NUMINAMATH_GPT_sickness_temperature_increase_l189_18961


namespace NUMINAMATH_GPT_find_a5_l189_18979

-- Definitions related to the conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a5 (a : ℕ → ℕ) (h_arith : arithmetic_sequence a) (h_a3 : a 3 = 3)
  (h_geo : geometric_sequence (a 1) (a 2) (a 4)) :
  a 5 = 5 ∨ a 5 = 3 :=
  sorry

end NUMINAMATH_GPT_find_a5_l189_18979


namespace NUMINAMATH_GPT_sales_in_fourth_month_l189_18965

theorem sales_in_fourth_month
  (sale1 : ℕ)
  (sale2 : ℕ)
  (sale3 : ℕ)
  (sale5 : ℕ)
  (sale6 : ℕ)
  (average : ℕ)
  (h_sale1 : sale1 = 2500)
  (h_sale2 : sale2 = 6500)
  (h_sale3 : sale3 = 9855)
  (h_sale5 : sale5 = 7000)
  (h_sale6 : sale6 = 11915)
  (h_average : average = 7500) :
  ∃ sale4 : ℕ, sale4 = 14230 := by
  sorry

end NUMINAMATH_GPT_sales_in_fourth_month_l189_18965


namespace NUMINAMATH_GPT_find_a1_l189_18912

noncomputable def sum_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then n * a 0 else a 0 * (1 - q ^ n) / (1 - q)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Definitions from conditions
def S_3_eq_a2_plus_10a1 (a_1 a_2 S_3 : ℝ) : Prop :=
S_3 = a_2 + 10 * a_1

def a_5_eq_9 (a_5 : ℝ) : Prop :=
a_5 = 9

-- Main theorem statement
theorem find_a1 (h1 : S_3_eq_a2_plus_10a1 (a 1) (a 2) (sum_of_geometric_sequence a q 3))
                (h2 : a_5_eq_9 (a 5))
                (h3 : q ≠ 0 ∧ q ≠ 1) :
    a 1 = 1 / 9 :=
sorry

end NUMINAMATH_GPT_find_a1_l189_18912


namespace NUMINAMATH_GPT_xiaoming_total_money_l189_18949

def xiaoming_money (x : ℕ) := 9 * x

def fresh_milk_cost (y : ℕ) := 6 * y

def yogurt_cost_equation (x y : ℕ) := y = x + 6

theorem xiaoming_total_money (x : ℕ) (y : ℕ)
  (h1: fresh_milk_cost y = xiaoming_money x)
  (h2: yogurt_cost_equation x y) : xiaoming_money x = 108 := 
  sorry

end NUMINAMATH_GPT_xiaoming_total_money_l189_18949


namespace NUMINAMATH_GPT_amber_total_cost_l189_18964

/-
Conditions:
1. Base cost of the plan: $25.
2. Cost for text messages with different rates for the first 120 messages and additional messages.
3. Cost for additional talk time.
4. Given specific usage data for Amber in January.

Objective:
Prove that the total monthly cost for Amber is $47.
-/
noncomputable def base_cost : ℕ := 25
noncomputable def text_message_cost (total_messages : ℕ) : ℕ :=
  if total_messages <= 120 then
    3 * total_messages
  else
    3 * 120 + 2 * (total_messages - 120)

noncomputable def talk_time_cost (talk_hours : ℕ) : ℕ :=
  if talk_hours <= 25 then
    0
  else
    15 * 60 * (talk_hours - 25)

noncomputable def total_monthly_cost (total_messages : ℕ) (talk_hours : ℕ) : ℕ :=
  base_cost + ((text_message_cost total_messages) / 100) + ((talk_time_cost talk_hours) / 100)

theorem amber_total_cost : total_monthly_cost 140 27 = 47 := by
  sorry

end NUMINAMATH_GPT_amber_total_cost_l189_18964


namespace NUMINAMATH_GPT_zander_construction_cost_l189_18975

noncomputable def cost_of_cement (num_bags : ℕ) (price_per_bag : ℕ) : ℕ :=
  num_bags * price_per_bag

noncomputable def amount_of_sand (num_lorries : ℕ) (tons_per_lorry : ℕ) : ℕ :=
  num_lorries * tons_per_lorry

noncomputable def cost_of_sand (total_tons : ℕ) (price_per_ton : ℕ) : ℕ :=
  total_tons * price_per_ton

noncomputable def total_cost (cost_cement : ℕ) (cost_sand : ℕ) : ℕ :=
  cost_cement + cost_sand

theorem zander_construction_cost :
  total_cost (cost_of_cement 500 10) (cost_of_sand (amount_of_sand 20 10) 40) = 13000 :=
by
  sorry

end NUMINAMATH_GPT_zander_construction_cost_l189_18975


namespace NUMINAMATH_GPT_initial_food_days_l189_18984

theorem initial_food_days (x : ℕ) (h : 760 * (x - 2) = 3040 * 5) : x = 22 := by
  sorry

end NUMINAMATH_GPT_initial_food_days_l189_18984


namespace NUMINAMATH_GPT_speed_of_stream_l189_18914

theorem speed_of_stream
  (v : ℝ)
  (h1 : ∀ t : ℝ, t = 7)
  (h2 : ∀ d : ℝ, d = 72)
  (h3 : ∀ s : ℝ, s = 21)
  : (72 / (21 - v) + 72 / (21 + v) = 7) → v = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_speed_of_stream_l189_18914


namespace NUMINAMATH_GPT_no_base_131_cubed_l189_18906

open Nat

theorem no_base_131_cubed (n : ℕ) (k : ℕ) : 
  (4 ≤ n ∧ n ≤ 12) ∧ (1 * n^2 + 3 * n + 1 = k^3) → False := by
  sorry

end NUMINAMATH_GPT_no_base_131_cubed_l189_18906


namespace NUMINAMATH_GPT_determine_y_l189_18947

-- Define the main problem in a Lean theorem
theorem determine_y (y : ℕ) : 9^10 + 9^10 + 9^10 = 3^y → y = 21 :=
by
  -- proof not required, so we add sorry
  sorry

end NUMINAMATH_GPT_determine_y_l189_18947


namespace NUMINAMATH_GPT_number_of_people_l189_18972

theorem number_of_people (total_eggs : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : 
  total_eggs = 36 → eggs_per_omelet = 4 → omelets_per_person = 3 → 
  (total_eggs / eggs_per_omelet) / omelets_per_person = 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_of_people_l189_18972


namespace NUMINAMATH_GPT_retail_price_of_washing_machine_l189_18924

variable (a : ℝ)

theorem retail_price_of_washing_machine :
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price 
  retail_price = 1.04 * a :=
by
  let increased_price := 1.3 * a
  let retail_price := 0.8 * increased_price
  sorry -- Proof skipped

end NUMINAMATH_GPT_retail_price_of_washing_machine_l189_18924


namespace NUMINAMATH_GPT_most_cost_effective_years_l189_18920

noncomputable def total_cost (x : ℕ) : ℝ := 100000 + 15000 * x + 1000 + 2000 * ((x * (x - 1)) / 2)

noncomputable def average_annual_cost (x : ℕ) : ℝ := total_cost x / x

theorem most_cost_effective_years : ∃ (x : ℕ), x = 10 ∧
  (∀ y : ℕ, y ≠ 10 → average_annual_cost x ≤ average_annual_cost y) :=
by
  sorry

end NUMINAMATH_GPT_most_cost_effective_years_l189_18920


namespace NUMINAMATH_GPT_like_terms_correct_l189_18966

theorem like_terms_correct : 
  (¬(∀ x y z w : ℝ, (x * y^2 = z ∧ x^2 * y = w)) ∧ 
   ¬(∀ x y : ℝ, (x * y = -2 * y)) ∧ 
    (2^3 = 8 ∧ 3^2 = 9) ∧ 
   ¬(∀ x y z w : ℝ, (5 * x * y = z ∧ 6 * x * y^2 = w))) :=
by
  sorry

end NUMINAMATH_GPT_like_terms_correct_l189_18966


namespace NUMINAMATH_GPT_tournament_participants_l189_18900

theorem tournament_participants (x : ℕ) (h1 : ∀ g b : ℕ, g = 2 * b)
  (h2 : ∀ p : ℕ, p = 3 * x) 
  (h3 : ∀ G B : ℕ, G + B = (3 * x * (3 * x - 1)) / 2)
  (h4 : ∀ G B : ℕ, G / B = 7 / 9) 
  (h5 : x = 11) :
  3 * x = 33 :=
by
  sorry

end NUMINAMATH_GPT_tournament_participants_l189_18900


namespace NUMINAMATH_GPT_degrees_to_minutes_l189_18999

theorem degrees_to_minutes (d : ℚ) (fractional_part : ℚ) (whole_part : ℤ) :
  1 ≤ d ∧ d = fractional_part + whole_part ∧ fractional_part = 0.45 ∧ whole_part = 1 →
  (whole_part + fractional_part) * 60 = 1 * 60 + 27 :=
by { sorry }

end NUMINAMATH_GPT_degrees_to_minutes_l189_18999


namespace NUMINAMATH_GPT_least_n_ge_100_divides_sum_of_powers_l189_18926

theorem least_n_ge_100_divides_sum_of_powers (n : ℕ) (h₁ : n ≥ 100) :
    77 ∣ (Finset.sum (Finset.range (n + 1)) (λ k => 2^k) - 1) ↔ n = 119 :=
by
  sorry

end NUMINAMATH_GPT_least_n_ge_100_divides_sum_of_powers_l189_18926


namespace NUMINAMATH_GPT_fraction_simplification_l189_18915

theorem fraction_simplification : 
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l189_18915


namespace NUMINAMATH_GPT_x_coordinate_at_2005th_stop_l189_18987

theorem x_coordinate_at_2005th_stop :
 (∃ (f : ℕ → ℤ × ℤ),
    f 0 = (0, 0) ∧
    f 1 = (1, 0) ∧
    f 2 = (1, 1) ∧
    f 3 = (0, 1) ∧
    f 4 = (-1, 1) ∧
    f 5 = (-1, 0) ∧
    f 9 = (2, -1))
  → (∃ (f : ℕ → ℤ × ℤ), f 2005 = (3, -n)) := sorry

end NUMINAMATH_GPT_x_coordinate_at_2005th_stop_l189_18987


namespace NUMINAMATH_GPT_sheets_in_total_l189_18945

theorem sheets_in_total (boxes_needed : ℕ) (sheets_per_box : ℕ) (total_sheets : ℕ) 
  (h1 : boxes_needed = 7) (h2 : sheets_per_box = 100) : total_sheets = boxes_needed * sheets_per_box := by
  sorry

end NUMINAMATH_GPT_sheets_in_total_l189_18945


namespace NUMINAMATH_GPT_height_of_tank_B_l189_18909

noncomputable def height_tank_A : ℝ := 5
noncomputable def circumference_tank_A : ℝ := 4
noncomputable def circumference_tank_B : ℝ := 10
noncomputable def capacity_ratio : ℝ := 0.10000000000000002

theorem height_of_tank_B {h_B : ℝ} 
  (h_tank_A : height_tank_A = 5)
  (c_tank_A : circumference_tank_A = 4)
  (c_tank_B : circumference_tank_B = 10)
  (capacity_percentage : capacity_ratio = 0.10000000000000002)
  (V_A : ℝ := π * (2 / π)^2 * height_tank_A)
  (V_B : ℝ := π * (5 / π)^2 * h_B)
  (capacity_relation : V_A = capacity_ratio * V_B) :
  h_B = 8 :=
sorry

end NUMINAMATH_GPT_height_of_tank_B_l189_18909


namespace NUMINAMATH_GPT_pipe_q_fill_time_l189_18989

theorem pipe_q_fill_time :
  ∀ (T : ℝ), (2 * (1 / 10 + 1 / T) + 10 * (1 / T) = 1) → T = 15 :=
by
  intro T
  intro h
  sorry

end NUMINAMATH_GPT_pipe_q_fill_time_l189_18989


namespace NUMINAMATH_GPT_geometric_arithmetic_series_difference_l189_18938

theorem geometric_arithmetic_series_difference :
  let a := 1
  let r := 1 / 2
  let S := a / (1 - r)
  let T := 1 + 2 + 3
  S - T = -4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_series_difference_l189_18938


namespace NUMINAMATH_GPT_cookie_store_expense_l189_18970

theorem cookie_store_expense (B D: ℝ) 
  (h₁: D = (1 / 2) * B)
  (h₂: B = D + 20):
  B + D = 60 := by
  sorry

end NUMINAMATH_GPT_cookie_store_expense_l189_18970


namespace NUMINAMATH_GPT_two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l189_18986

open Nat

theorem two_pow_m_minus_one_not_divide_three_pow_n_minus_one 
  (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hmo : Odd m) (hno : Odd n) : ¬ (∃ k : ℕ, 2^m - 1 = k * (3^n - 1)) := by
  sorry

end NUMINAMATH_GPT_two_pow_m_minus_one_not_divide_three_pow_n_minus_one_l189_18986


namespace NUMINAMATH_GPT_find_enclosed_area_l189_18991

def area_square (side_length : ℕ) : ℕ :=
  side_length * side_length

def area_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem find_enclosed_area :
  let side1 := 3
  let side2 := 6
  let area1 := area_square side1
  let area2 := area_square side2
  let area_tri := 2 * area_triangle side1 side2
  area1 + area2 + area_tri = 63 :=
by
  sorry

end NUMINAMATH_GPT_find_enclosed_area_l189_18991


namespace NUMINAMATH_GPT_equal_cubic_values_l189_18923

theorem equal_cubic_values (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 3) 
  (h3 : a * b * c + b * c * d + c * d * a + d * a * b = 1) :
  a * (1 - a)^3 = b * (1 - b)^3 ∧ 
  b * (1 - b)^3 = c * (1 - c)^3 ∧ 
  c * (1 - c)^3 = d * (1 - d)^3 :=
sorry

end NUMINAMATH_GPT_equal_cubic_values_l189_18923


namespace NUMINAMATH_GPT_seven_pow_fifty_one_mod_103_l189_18908

theorem seven_pow_fifty_one_mod_103 : (7^51 - 1) % 103 = 0 := 
by
  -- Fermat's Little Theorem: If p is a prime number and a is an integer not divisible by p,
  -- then a^(p-1) ≡ 1 ⧸ p.
  -- 103 is prime, so for 7 which is not divisible by 103, we have 7^102 ≡ 1 ⧸ 103.
sorry

end NUMINAMATH_GPT_seven_pow_fifty_one_mod_103_l189_18908


namespace NUMINAMATH_GPT_problem_fraction_of_complex_numbers_l189_18968

/--
Given \(i\) is the imaginary unit, prove that \(\frac {1-i}{1+i} = -i\).
-/
theorem problem_fraction_of_complex_numbers (i : ℂ) (h_i : i^2 = -1) : 
  ((1 - i) / (1 + i)) = -i := 
sorry

end NUMINAMATH_GPT_problem_fraction_of_complex_numbers_l189_18968


namespace NUMINAMATH_GPT_people_sitting_between_same_l189_18988

theorem people_sitting_between_same 
  (n : ℕ) (h_even : n % 2 = 0) 
  (f : Fin (2 * n) → Fin (2 * n)) :
  ∃ (a b : Fin (2 * n)), 
  ∃ (k k' : ℕ), k < 2 * n ∧ k' < 2 * n ∧ (a : ℕ) < (b : ℕ) ∧ 
  ((b - a = k) ∧ (f b - f a = k)) ∨ ((a - b + 2*n = k') ∧ ((f a - f b + 2 * n) % (2 * n) = k')) :=
by
  sorry

end NUMINAMATH_GPT_people_sitting_between_same_l189_18988


namespace NUMINAMATH_GPT_determine_OP_l189_18982

variables (a b c d q : ℝ)
variables (P : ℝ)
variables (h_ratio : (|a - P| / |P - d| = |b - P| / |P - c|))
variables (h_twice : P = 2 * q)

theorem determine_OP : P = 2 * q :=
sorry

end NUMINAMATH_GPT_determine_OP_l189_18982
