import Mathlib

namespace NUMINAMATH_GPT_upper_limit_of_multiples_of_10_l449_44967

theorem upper_limit_of_multiples_of_10 (n : ℕ) (hn : 10 * n = 100) (havg : (10 * n + 10) / (n + 1) = 55) : 10 * n = 100 :=
by
  sorry

end NUMINAMATH_GPT_upper_limit_of_multiples_of_10_l449_44967


namespace NUMINAMATH_GPT_soccer_balls_percentage_holes_l449_44925

variable (x : ℕ)

theorem soccer_balls_percentage_holes 
    (h1 : ∃ x, 0 ≤ x ∧ x ≤ 100)
    (h2 : 48 = 80 * (100 - x) / 100) : 
  x = 40 := sorry

end NUMINAMATH_GPT_soccer_balls_percentage_holes_l449_44925


namespace NUMINAMATH_GPT_relative_error_comparison_l449_44991

theorem relative_error_comparison :
  let error1 := 0.05
  let length1 := 25
  let error2 := 0.25
  let length2 := 125
  (error1 / length1) = (error2 / length2) :=
by
  sorry

end NUMINAMATH_GPT_relative_error_comparison_l449_44991


namespace NUMINAMATH_GPT_license_plate_count_l449_44966

noncomputable def num_license_plates : Nat :=
  let num_digit_possibilities := 10
  let num_letter_possibilities := 26
  let num_letter_pairs := num_letter_possibilities * num_letter_possibilities
  let num_positions_for_block := 6
  num_positions_for_block * (num_digit_possibilities ^ 5) * num_letter_pairs

theorem license_plate_count :
  num_license_plates = 40560000 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_count_l449_44966


namespace NUMINAMATH_GPT_friends_receive_pens_l449_44969

-- Define the given conditions
def packs_kendra : ℕ := 4
def packs_tony : ℕ := 2
def pens_per_pack : ℕ := 3
def pens_kept_per_person : ℕ := 2

-- Define the proof problem
theorem friends_receive_pens :
  (packs_kendra * pens_per_pack + packs_tony * pens_per_pack - (pens_kept_per_person * 2)) = 14 :=
by sorry

end NUMINAMATH_GPT_friends_receive_pens_l449_44969


namespace NUMINAMATH_GPT_remaining_apples_l449_44947

-- Define the initial number of apples
def initialApples : ℕ := 356

-- Define the number of apples given away as a mixed number converted to a fraction
def applesGivenAway : ℚ := 272 + 3/5

-- Prove that the remaining apples after giving away are 83
theorem remaining_apples
  (initialApples : ℕ)
  (applesGivenAway : ℚ) :
  initialApples - applesGivenAway = 83 := 
sorry

end NUMINAMATH_GPT_remaining_apples_l449_44947


namespace NUMINAMATH_GPT_square_eq_four_implies_two_l449_44946

theorem square_eq_four_implies_two (x : ℝ) (h : x^2 = 4) : x = 2 := 
sorry

end NUMINAMATH_GPT_square_eq_four_implies_two_l449_44946


namespace NUMINAMATH_GPT_find_t_l449_44934

-- conditions
def quadratic_eq (x : ℝ) : Prop := 25 * x^2 + 20 * x - 1000 = 0

-- statement to prove
theorem find_t (x : ℝ) (p t : ℝ) (h1 : p = 2/5) (h2 : t = 104/25) : 
  (quadratic_eq x) → (x + p)^2 = t :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_t_l449_44934


namespace NUMINAMATH_GPT_cost_of_pen_l449_44958

-- define the conditions
def notebook_cost (pen_cost : ℝ) : ℝ := 3 * pen_cost
def total_cost (notebook_cost : ℝ) : ℝ := 4 * notebook_cost

-- theorem stating the problem we need to prove
theorem cost_of_pen (pen_cost : ℝ) (h1 : total_cost (notebook_cost pen_cost) = 18) : pen_cost = 1.5 :=
by
  -- proof to be constructed
  sorry

end NUMINAMATH_GPT_cost_of_pen_l449_44958


namespace NUMINAMATH_GPT_line_circle_no_intersection_l449_44974

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → ¬ (x^2 + y^2 = 4)) :=
by
  sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l449_44974


namespace NUMINAMATH_GPT_intersection_M_N_eq_l449_44935

open Set

theorem intersection_M_N_eq :
  let M := {x : ℝ | x - 2 > 0}
  let N := {y : ℝ | ∃ (x : ℝ), y = Real.sqrt (x^2 + 1)}
  M ∩ N = {x : ℝ | x > 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_l449_44935


namespace NUMINAMATH_GPT_find_fourth_number_l449_44921

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end NUMINAMATH_GPT_find_fourth_number_l449_44921


namespace NUMINAMATH_GPT_jane_coffees_l449_44901

open Nat

theorem jane_coffees (b m c n : Nat) 
  (h1 : b + m + c = 6)
  (h2 : 75 * b + 60 * m + 100 * c = 100 * n) :
  c = 1 :=
by sorry

end NUMINAMATH_GPT_jane_coffees_l449_44901


namespace NUMINAMATH_GPT_segment_length_OI_is_3_l449_44954

-- Define the points along the path
def point (n : ℕ) : ℝ × ℝ := (n, n)

-- Use the Pythagorean theorem to calculate the distance from point O to point I
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the points O and I
def O : ℝ × ℝ := point 0
def I : ℝ × ℝ := point 3

-- The proposition to prove: 
-- The distance between points O and I is 3
theorem segment_length_OI_is_3 : distance O I = 3 := 
  sorry

end NUMINAMATH_GPT_segment_length_OI_is_3_l449_44954


namespace NUMINAMATH_GPT_cos_75_degree_l449_44997

theorem cos_75_degree (cos : ℝ → ℝ) (sin : ℝ → ℝ) :
    cos 75 = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_75_degree_l449_44997


namespace NUMINAMATH_GPT_union_of_A_and_B_l449_44907

open Set

-- Definitions for the conditions
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Statement of the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l449_44907


namespace NUMINAMATH_GPT_constant_remainder_polynomial_division_l449_44996

theorem constant_remainder_polynomial_division (b : ℚ) :
  (∃ (r : ℚ), ∀ x : ℚ, r = (8 * x^3 - 9 * x^2 + b * x + 10) % (3 * x^2 - 2 * x + 5)) ↔ b = 118 / 9 :=
by
  sorry

end NUMINAMATH_GPT_constant_remainder_polynomial_division_l449_44996


namespace NUMINAMATH_GPT_tan_half_difference_l449_44994

-- Given two angles a and b with the following conditions
variables (a b : ℝ)
axiom cos_cond : (Real.cos a + Real.cos b = 3 / 5)
axiom sin_cond : (Real.sin a + Real.sin b = 2 / 5)

-- Prove that tan ((a - b) / 2) = 2 / 3
theorem tan_half_difference (a b : ℝ) (cos_cond : Real.cos a + Real.cos b = 3 / 5) 
  (sin_cond : Real.sin a + Real.sin b = 2 / 5) : 
  Real.tan ((a - b) / 2) = 2 / 3 := 
sorry

end NUMINAMATH_GPT_tan_half_difference_l449_44994


namespace NUMINAMATH_GPT_athlete_groups_l449_44939

/-- A school has athletes divided into groups.
   - If there are 7 people per group, there will be 3 people left over.
   - If there are 8 people per group, there will be a shortage of 5 people.
The goal is to prove that the system of equations is valid --/
theorem athlete_groups (x y : ℕ) :
  7 * y = x - 3 ∧ 8 * y = x + 5 := 
by 
  sorry

end NUMINAMATH_GPT_athlete_groups_l449_44939


namespace NUMINAMATH_GPT_competition_votes_l449_44961

/-- 
In a revival competition, if B's number of votes is 20/21 of A's, and B wins by
gaining at least 4 votes more than A, prove the possible valid votes counts.
-/
theorem competition_votes (x : ℕ) 
  (hx : x > 0) 
  (hx_mod_21 : x % 21 = 0) 
  (hB_wins : ∀ b : ℕ, b = (20 * x / 21) + 4 → b > x - 4) :
  (x = 147 ∧ 140 = 20 * x / 21) ∨ (x = 126 ∧ 120 = 20 * x / 21) := 
by 
  sorry

end NUMINAMATH_GPT_competition_votes_l449_44961


namespace NUMINAMATH_GPT_derivative_f_eq_l449_44970

noncomputable def f (x : ℝ) : ℝ := (Real.exp (2 * x)) / x

theorem derivative_f_eq :
  (deriv f) = fun x ↦ ((2 * x - 1) * (Real.exp (2 * x))) / (x ^ 2) := by
  sorry

end NUMINAMATH_GPT_derivative_f_eq_l449_44970


namespace NUMINAMATH_GPT_birth_year_1849_l449_44909

theorem birth_year_1849 (x : ℕ) (h1 : 1850 ≤ x^2 - 2 * x + 1) (h2 : x^2 - 2 * x + 1 < 1900) (h3 : x^2 - x + 1 = x) : x = 44 ↔ x^2 - 2 * x + 1 = 1849 := 
sorry

end NUMINAMATH_GPT_birth_year_1849_l449_44909


namespace NUMINAMATH_GPT_total_weight_marble_purchased_l449_44962

theorem total_weight_marble_purchased (w1 w2 w3 : ℝ) (h1 : w1 = 0.33) (h2 : w2 = 0.33) (h3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 := by
  sorry

end NUMINAMATH_GPT_total_weight_marble_purchased_l449_44962


namespace NUMINAMATH_GPT_total_tshirts_bought_l449_44957

-- Given conditions
def white_packs : ℕ := 3
def white_tshirts_per_pack : ℕ := 6
def blue_packs : ℕ := 2
def blue_tshirts_per_pack : ℕ := 4

-- Theorem statement: Total number of T-shirts Dave bought
theorem total_tshirts_bought : white_packs * white_tshirts_per_pack + blue_packs * blue_tshirts_per_pack = 26 := by
  sorry

end NUMINAMATH_GPT_total_tshirts_bought_l449_44957


namespace NUMINAMATH_GPT_probability_of_draw_l449_44960

noncomputable def P_A_winning : ℝ := 0.4
noncomputable def P_A_not_losing : ℝ := 0.9

theorem probability_of_draw : P_A_not_losing - P_A_winning = 0.5 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_draw_l449_44960


namespace NUMINAMATH_GPT_smallest_lucky_number_theorem_specific_lucky_number_theorem_l449_44932

-- Definitions based on the given conditions
def is_lucky_number (M : ℕ) : Prop :=
  ∃ (A B : ℕ), (M = A * B) ∧
               (A ≥ B) ∧
               (A ≥ 10 ∧ A ≤ 99) ∧
               (B ≥ 10 ∧ B ≤ 99) ∧
               (A / 10 = B / 10) ∧
               (A % 10 + B % 10 = 6)

def smallest_lucky_number : ℕ :=
  165

def P (M A B : ℕ) := A + B
def Q (M A B : ℕ) := A - B

def specific_lucky_number (M A B : ℕ) : Prop :=
  M = A * B ∧ (P M A B) / (Q M A B) % 7 = 0

-- Theorems to prove
theorem smallest_lucky_number_theorem :
  ∃ M, is_lucky_number M ∧ M = smallest_lucky_number := by
  sorry

theorem specific_lucky_number_theorem :
  ∃ M A B, is_lucky_number M ∧ specific_lucky_number M A B ∧ M = 3968 := by
  sorry

end NUMINAMATH_GPT_smallest_lucky_number_theorem_specific_lucky_number_theorem_l449_44932


namespace NUMINAMATH_GPT_find_b_l449_44928

theorem find_b (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 40) (h3 : a - 2 * b = 10) (ha : a = 4) : b = 75 :=
  sorry

end NUMINAMATH_GPT_find_b_l449_44928


namespace NUMINAMATH_GPT_breadth_of_rectangle_l449_44944

noncomputable def length (radius : ℝ) : ℝ := (1/4) * radius
noncomputable def side (sq_area : ℝ) : ℝ := Real.sqrt sq_area
noncomputable def radius (side : ℝ) : ℝ := side
noncomputable def breadth (rect_area length : ℝ) : ℝ := rect_area / length

theorem breadth_of_rectangle :
  breadth 200 (length (radius (side 1225))) = 200 / (1/4 * Real.sqrt 1225) :=
by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangle_l449_44944


namespace NUMINAMATH_GPT_complex_number_is_3i_quadratic_equation_roots_l449_44906

open Complex

-- Given complex number z satisfies 2z + |z| = 3 + 6i
-- We need to prove that z = 3i
theorem complex_number_is_3i (z : ℂ) (h : 2 * z + abs z = 3 + 6 * I) : z = 3 * I :=
sorry

-- Given that z = 3i is a root of the quadratic equation with real coefficients
-- Prove that b - c = -9
theorem quadratic_equation_roots (b c : ℝ) (h1 : 3 * I + -3 * I = -b)
  (h2 : 3 * I * -3 * I = c) : b - c = -9 :=
sorry

end NUMINAMATH_GPT_complex_number_is_3i_quadratic_equation_roots_l449_44906


namespace NUMINAMATH_GPT_d_minus_b_equals_757_l449_44971

theorem d_minus_b_equals_757 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := 
by 
  sorry

end NUMINAMATH_GPT_d_minus_b_equals_757_l449_44971


namespace NUMINAMATH_GPT_sin_expression_l449_44945

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

theorem sin_expression (a b x₀ : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : ∀ x, f a b x = f a b (π / 6 - x)) 
  (h₃ : f a b x₀ = (8 / 5) * a) 
  (h₄ : b = Real.sqrt 3 * a) :
  Real.sin (2 * x₀ + π / 6) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_expression_l449_44945


namespace NUMINAMATH_GPT_woman_wait_time_l449_44941
noncomputable def time_for_man_to_catch_up (man_speed woman_speed distance: ℝ) : ℝ :=
  distance / man_speed

theorem woman_wait_time 
    (man_speed : ℝ)
    (woman_speed : ℝ)
    (wait_time_minutes : ℝ) 
    (woman_time : ℝ)
    (distance : ℝ)
    (man_time : ℝ) :
    man_speed = 5 -> 
    woman_speed = 15 -> 
    wait_time_minutes = 2 -> 
    woman_time = woman_speed * (1 / 60) * wait_time_minutes -> 
    woman_time = distance -> 
    man_speed * (1 / 60) = 0.0833 -> 
    man_time = distance / 0.0833 -> 
    man_time = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_woman_wait_time_l449_44941


namespace NUMINAMATH_GPT_rectangle_area_in_cm_l449_44915

theorem rectangle_area_in_cm (length_in_m : ℝ) (width_in_m : ℝ) 
  (h_length : length_in_m = 0.5) (h_width : width_in_m = 0.36) : 
  (100 * length_in_m) * (100 * width_in_m) = 1800 :=
by
  -- We skip the proof for now
  sorry

end NUMINAMATH_GPT_rectangle_area_in_cm_l449_44915


namespace NUMINAMATH_GPT_net_change_in_price_l449_44938

-- Define the initial price of the TV
def initial_price (P : ℝ) := P

-- Define the price after a 20% decrease
def decreased_price (P : ℝ) := 0.80 * P

-- Define the final price after a 50% increase on the decreased price
def final_price (P : ℝ) := 1.20 * P

-- Prove that the net change is 20% of the original price
theorem net_change_in_price (P : ℝ) : final_price P - initial_price P = 0.20 * P := by
  sorry

end NUMINAMATH_GPT_net_change_in_price_l449_44938


namespace NUMINAMATH_GPT_johns_change_l449_44910

/-- Define the cost of Slurpees and amount given -/
def cost_per_slurpee : ℕ := 2
def amount_given : ℕ := 20
def slurpees_bought : ℕ := 6

/-- Define the total cost of the Slurpees -/
def total_cost : ℕ := cost_per_slurpee * slurpees_bought

/-- Define the change John gets -/
def change (amount_given total_cost : ℕ) : ℕ := amount_given - total_cost

/-- The statement for Lean 4 that proves the change John gets is $8 given the conditions -/
theorem johns_change : change amount_given total_cost = 8 :=
by 
  -- Rest of the proof omitted
  sorry

end NUMINAMATH_GPT_johns_change_l449_44910


namespace NUMINAMATH_GPT_determine_N_l449_44912

theorem determine_N (N : ℕ) :
    995 + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := 
by 
  sorry

end NUMINAMATH_GPT_determine_N_l449_44912


namespace NUMINAMATH_GPT_tangent_ln_at_origin_l449_44949

theorem tangent_ln_at_origin {k : ℝ} (h : ∀ x : ℝ, (k * x = Real.log x) → k = 1 / x) : k = 1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_ln_at_origin_l449_44949


namespace NUMINAMATH_GPT_find_integer_n_l449_44923

noncomputable def cubic_expr_is_pure_integer (n : ℤ) : Prop :=
  (729 * n ^ 6 - 540 * n ^ 4 + 240 * n ^ 2 - 64 : ℂ).im = 0

theorem find_integer_n :
  ∃! n : ℤ, cubic_expr_is_pure_integer n := 
sorry

end NUMINAMATH_GPT_find_integer_n_l449_44923


namespace NUMINAMATH_GPT_range_of_f_gt_f_of_quadratic_l449_44964

-- Define the function f and its properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Define the problem statement
theorem range_of_f_gt_f_of_quadratic (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_inc : is_increasing_on_pos f) :
  {x : ℝ | f x > f (x^2 - 2*x + 2)} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_range_of_f_gt_f_of_quadratic_l449_44964


namespace NUMINAMATH_GPT_outlier_attribute_l449_44952

/-- Define the given attributes of the Dragon -/
def one_eyed := "одноокий"
def two_eared := "двуухий"
def three_tailed := "треххвостый"
def four_legged := "четырехлапый"
def five_spiked := "пятиглый"

/-- Define a predicate to check if an attribute contains doubled letters -/
def has_doubled_letters (s : String) : Bool :=
  let chars := s.toList
  chars.any (fun ch => chars.count ch > 1)

/-- Prove that "четырехлапый" (four-legged) does not fit the pattern of containing doubled letters -/
theorem outlier_attribute : ¬ has_doubled_letters four_legged :=
by
  -- Proof would be inserted here
  sorry

end NUMINAMATH_GPT_outlier_attribute_l449_44952


namespace NUMINAMATH_GPT_sqrt_four_l449_44924

theorem sqrt_four : {x : ℝ | x ^ 2 = 4} = {-2, 2} := by
  sorry

end NUMINAMATH_GPT_sqrt_four_l449_44924


namespace NUMINAMATH_GPT_Kelsey_watched_537_videos_l449_44956

-- Definitions based on conditions
def total_videos : ℕ := 1222
def delilah_videos : ℕ := 78

-- Declaration of variables representing the number of videos each friend watched
variables (Kelsey Ekon Uma Ivan Lance : ℕ)

-- Conditions from the problem
def cond1 : Kelsey = 3 * Ekon := sorry
def cond2 : Ekon = Uma - 23 := sorry
def cond3 : Uma = 2 * Ivan := sorry
def cond4 : Lance = Ivan + 19 := sorry
def cond5 : delilah_videos = 78 := sorry
def cond6 := Kelsey + Ekon + Uma + Ivan + Lance + delilah_videos = total_videos

-- The theorem to prove
theorem Kelsey_watched_537_videos : Kelsey = 537 :=
  by
  sorry

end NUMINAMATH_GPT_Kelsey_watched_537_videos_l449_44956


namespace NUMINAMATH_GPT_parallel_lines_condition_l449_44981

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y - 1 = 0) → (x + (a + 1) * y + 4 = 0) → a = 1) ↔
  (∀ x y : ℝ, (a = 1 ∧ a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0) ∨
   (a ≠ 1 ∧ a = -2 ∧ a * x + 2 * y - 1 ≠ 0 → x + (a + 1) * y + 4 ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l449_44981


namespace NUMINAMATH_GPT_expression_value_l449_44965

theorem expression_value : 200 * (200 - 8) - (200 * 200 + 8) = -1608 := 
by 
  -- We will put the proof here
  sorry

end NUMINAMATH_GPT_expression_value_l449_44965


namespace NUMINAMATH_GPT_rectangle_area_given_diagonal_l449_44968

noncomputable def area_of_rectangle (x : ℝ) : ℝ :=
  1250 - x^2 / 2

theorem rectangle_area_given_diagonal (P : ℝ) (x : ℝ) (A : ℝ) :
  P = 100 → x^2 = (P / 2)^2 - 2 * A → A = area_of_rectangle x :=
by
  intros hP hx
  sorry

end NUMINAMATH_GPT_rectangle_area_given_diagonal_l449_44968


namespace NUMINAMATH_GPT_cost_of_3600_pens_l449_44980

theorem cost_of_3600_pens
  (pack_size : ℕ)
  (pack_cost : ℝ)
  (n_pens : ℕ)
  (pen_cost : ℝ)
  (total_cost : ℝ)
  (h1: pack_size = 150)
  (h2: pack_cost = 45)
  (h3: n_pens = 3600)
  (h4: pen_cost = pack_cost / pack_size)
  (h5: total_cost = n_pens * pen_cost) :
  total_cost = 1080 :=
sorry

end NUMINAMATH_GPT_cost_of_3600_pens_l449_44980


namespace NUMINAMATH_GPT_weight_increase_percentage_l449_44979

theorem weight_increase_percentage :
  ∀ (x : ℝ), (2 * x * 1.1 + 5 * x * 1.17 = 82.8) →
    ((82.8 - (2 * x + 5 * x)) / (2 * x + 5 * x)) * 100 = 15.06 := 
by 
  intro x 
  intro h
  sorry

end NUMINAMATH_GPT_weight_increase_percentage_l449_44979


namespace NUMINAMATH_GPT_total_students_l449_44988

theorem total_students (boys girls : ℕ) (h_ratio : 5 * girls = 7 * boys) (h_girls : girls = 140) :
  boys + girls = 240 :=
sorry

end NUMINAMATH_GPT_total_students_l449_44988


namespace NUMINAMATH_GPT_Danny_more_than_Larry_l449_44963

/-- Keith scored 3 points. --/
def Keith_marks : Nat := 3

/-- Larry scored 3 times as many marks as Keith. --/
def Larry_marks : Nat := 3 * Keith_marks

/-- The total marks scored by Keith, Larry, and Danny is 26. --/
def total_marks (D : Nat) : Prop := Keith_marks + Larry_marks + D = 26

/-- Prove the number of more marks Danny scored than Larry is 5. --/
theorem Danny_more_than_Larry (D : Nat) (h : total_marks D) : D - Larry_marks = 5 :=
sorry

end NUMINAMATH_GPT_Danny_more_than_Larry_l449_44963


namespace NUMINAMATH_GPT_average_rate_of_reduction_l449_44927

theorem average_rate_of_reduction
  (original_price final_price : ℝ)
  (h1 : original_price = 200)
  (h2 : final_price = 128)
  : ∃ (x : ℝ), 0 ≤ x ∧ x < 1 ∧ 200 * (1 - x) * (1 - x) = 128 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_reduction_l449_44927


namespace NUMINAMATH_GPT_savings_correct_l449_44916

def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem savings_correct : final_savings = 1340840 :=
by
  sorry

end NUMINAMATH_GPT_savings_correct_l449_44916


namespace NUMINAMATH_GPT_zachary_cans_first_day_l449_44926

theorem zachary_cans_first_day :
  ∃ (first_day_cans : ℕ),
    ∃ (second_day_cans : ℕ),
      ∃ (third_day_cans : ℕ),
        ∃ (seventh_day_cans : ℕ),
          second_day_cans = 9 ∧
          third_day_cans = 14 ∧
          (∀ (n : ℕ), 2 ≤ n ∧ n < 7 → third_day_cans = second_day_cans + 5) →
          seventh_day_cans = 34 ∧
          first_day_cans = second_day_cans - 5 ∧
          first_day_cans = 4 :=

by
  sorry

end NUMINAMATH_GPT_zachary_cans_first_day_l449_44926


namespace NUMINAMATH_GPT_pattern_expression_equality_l449_44998

theorem pattern_expression_equality (n : ℕ) : ((n - 1) * (n + 1)) + 1 = n^2 :=
  sorry

end NUMINAMATH_GPT_pattern_expression_equality_l449_44998


namespace NUMINAMATH_GPT_find_principal_sum_l449_44986

def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem find_principal_sum (CI SI : ℝ) (t : ℕ)
  (h1 : CI = 11730) 
  (h2 : SI = 10200) 
  (h3 : t = 2) :
  ∃ P r, P = 17000 ∧
  compound_interest P r t = CI ∧
  simple_interest P r t = SI :=
by
  sorry

end NUMINAMATH_GPT_find_principal_sum_l449_44986


namespace NUMINAMATH_GPT_benny_pays_l449_44905

theorem benny_pays (cost_per_lunch : ℕ) (number_of_people : ℕ) (total_cost : ℕ) :
  cost_per_lunch = 8 → number_of_people = 3 → total_cost = number_of_people * cost_per_lunch → total_cost = 24 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_benny_pays_l449_44905


namespace NUMINAMATH_GPT_correct_operation_l449_44904

theorem correct_operation (a : ℕ) : a ^ 3 * a ^ 2 = a ^ 5 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l449_44904


namespace NUMINAMATH_GPT_number_of_men_in_third_group_l449_44983

theorem number_of_men_in_third_group (m w : ℝ) (x : ℕ) :
  3 * m + 8 * w = 6 * m + 2 * w →
  x * m + 5 * w = 0.9285714285714286 * (6 * m + 2 * w) →
  x = 4 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_number_of_men_in_third_group_l449_44983


namespace NUMINAMATH_GPT_scheduled_conference_games_l449_44995

-- Definitions based on conditions
def num_divisions := 3
def teams_per_division := 4
def games_within_division := 3
def games_across_divisions := 2

-- Proof statement
theorem scheduled_conference_games :
  let teams_in_division := teams_per_division
  let div_game_count := games_within_division * (teams_in_division * (teams_in_division - 1) / 2) 
  let total_within_division := div_game_count * num_divisions
  let cross_div_game_count := (teams_in_division * games_across_divisions * (num_divisions - 1) * teams_in_division * num_divisions) / 2
  total_within_division + cross_div_game_count = 102 := 
by {
  sorry
}

end NUMINAMATH_GPT_scheduled_conference_games_l449_44995


namespace NUMINAMATH_GPT_fraction_nonnegative_for_all_reals_l449_44992

theorem fraction_nonnegative_for_all_reals (x : ℝ) : 
  (x^2 + 2 * x + 1) / (x^2 + 4 * x + 8) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_fraction_nonnegative_for_all_reals_l449_44992


namespace NUMINAMATH_GPT_deduction_from_third_l449_44929

-- Define the conditions
def avg_10_consecutive_eq_20 (x : ℝ) : Prop :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 20

def new_avg_10_numbers_eq_15_5 (x y : ℝ) : Prop :=
  ((x - 9) + (x - 7) + (x + 2 - y) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 = 15.5

-- Define the theorem to be proved
theorem deduction_from_third (x y : ℝ) (h1 : avg_10_consecutive_eq_20 x) (h2 : new_avg_10_numbers_eq_15_5 x y) : y = 6 :=
sorry

end NUMINAMATH_GPT_deduction_from_third_l449_44929


namespace NUMINAMATH_GPT_analects_deductive_reasoning_l449_44936

theorem analects_deductive_reasoning :
  (∀ (P Q R S T U V : Prop), 
    (P → Q) → 
    (Q → R) → 
    (R → S) → 
    (S → T) → 
    (T → U) → 
    ((P → U) ↔ deductive_reasoning)) :=
sorry

end NUMINAMATH_GPT_analects_deductive_reasoning_l449_44936


namespace NUMINAMATH_GPT_average_student_headcount_is_correct_l449_44999

noncomputable def average_student_headcount : ℕ :=
  let a := 11000
  let b := 10200
  let c := 10800
  let d := 11300
  (a + b + c + d) / 4

theorem average_student_headcount_is_correct :
  average_student_headcount = 10825 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_average_student_headcount_is_correct_l449_44999


namespace NUMINAMATH_GPT_find_m_l449_44977

variables {m : ℝ}
def vec_a : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c : ℝ × ℝ := (m, 3)
def vec_a_plus_c := (1 + m, 3 + m)
def vec_a_minus_b := (1 - 2, m - 5)

theorem find_m (h : (1 + m) * (m - 5) = -1 * (m + 3)) : m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := 
sorry

end NUMINAMATH_GPT_find_m_l449_44977


namespace NUMINAMATH_GPT_negation_proof_l449_44920

theorem negation_proof (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l449_44920


namespace NUMINAMATH_GPT_num_sol_and_sum_sol_l449_44911

-- Definition of the main problem condition
def equation (x : ℝ) := (4 * x^2 - 9)^2 = 49

-- Proof problem statement
theorem num_sol_and_sum_sol :
  (∃ s : Finset ℝ, (∀ x, equation x ↔ x ∈ s) ∧ s.card = 4 ∧ s.sum id = 0) :=
sorry

end NUMINAMATH_GPT_num_sol_and_sum_sol_l449_44911


namespace NUMINAMATH_GPT_intersect_count_l449_44914

noncomputable def f (x : ℝ) : ℝ := sorry  -- Function f defined for all real x.
noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Inverse function of f.

theorem intersect_count : 
  (∃ a b : ℝ, a ≠ b ∧ f (a^2) = f (a^3) ∧ f (b^2) = f (b^3)) :=
by sorry

end NUMINAMATH_GPT_intersect_count_l449_44914


namespace NUMINAMATH_GPT_john_shower_duration_l449_44990

variable (days_per_week : ℕ := 7)
variable (weeks : ℕ := 4)
variable (total_days : ℕ := days_per_week * weeks)
variable (shower_frequency : ℕ := 2) -- every other day
variable (number_of_showers : ℕ := total_days / shower_frequency)
variable (total_gallons_used : ℕ := 280)
variable (gallons_per_shower : ℕ := total_gallons_used / number_of_showers)
variable (gallons_per_minute : ℕ := 2)

theorem john_shower_duration 
  (h_cond : total_gallons_used = number_of_showers * gallons_per_shower)
  (h_shower_eq : total_days / shower_frequency = number_of_showers)
  : gallons_per_shower / gallons_per_minute = 10 :=
by
  sorry

end NUMINAMATH_GPT_john_shower_duration_l449_44990


namespace NUMINAMATH_GPT_sum_of_squares_edges_l449_44930

-- Define Points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define given conditions (4 vertices each on two parallel planes)
def A1 : Point := { x := 0, y := 0, z := 0 }
def A2 : Point := { x := 1, y := 0, z := 0 }
def A3 : Point := { x := 1, y := 1, z := 0 }
def A4 : Point := { x := 0, y := 1, z := 0 }

def B1 : Point := { x := 0, y := 0, z := 1 }
def B2 : Point := { x := 1, y := 0, z := 1 }
def B3 : Point := { x := 1, y := 1, z := 1 }
def B4 : Point := { x := 0, y := 1, z := 1 }

-- Function to calculate distance squared between two points
def dist_sq (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2

-- The Theorem to be proven
theorem sum_of_squares_edges : dist_sq A1 B2 + dist_sq A2 B3 + dist_sq A3 B4 + dist_sq A4 B1 = 8 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_edges_l449_44930


namespace NUMINAMATH_GPT_rectangular_prism_volume_l449_44959

theorem rectangular_prism_volume (a b c V : ℝ) (h1 : a * b = 20) (h2 : b * c = 12) (h3 : a * c = 15) (hb : b = 5) : V = 75 :=
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l449_44959


namespace NUMINAMATH_GPT_sqrt_nested_eq_l449_44955

theorem sqrt_nested_eq (y : ℝ) (hy : 0 ≤ y) :
  Real.sqrt (y * Real.sqrt (y * Real.sqrt (y * Real.sqrt y))) = y ^ (9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_nested_eq_l449_44955


namespace NUMINAMATH_GPT_value_of_expression_l449_44918

theorem value_of_expression (x : ℤ) (h : x = 4) : (3 * x + 7) ^ 2 = 361 := by
  rw [h] -- Replace x with 4
  norm_num -- Simplify the expression
  done

end NUMINAMATH_GPT_value_of_expression_l449_44918


namespace NUMINAMATH_GPT_sum_of_terms_l449_44953

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sum_of_terms (h : ∀ n, S n = n^2) : a 5 + a 6 + a 7 = 33 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_terms_l449_44953


namespace NUMINAMATH_GPT_total_pets_combined_l449_44973

def teddy_dogs : ℕ := 7
def teddy_cats : ℕ := 8
def ben_dogs : ℕ := teddy_dogs + 9
def dave_cats : ℕ := teddy_cats + 13
def dave_dogs : ℕ := teddy_dogs - 5

def teddy_pets : ℕ := teddy_dogs + teddy_cats
def ben_pets : ℕ := ben_dogs
def dave_pets : ℕ := dave_cats + dave_dogs

def total_pets : ℕ := teddy_pets + ben_pets + dave_pets

theorem total_pets_combined : total_pets = 54 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_pets_combined_l449_44973


namespace NUMINAMATH_GPT_smallest_x_250_multiple_1080_l449_44978

theorem smallest_x_250_multiple_1080 : (∃ x : ℕ, x > 0 ∧ (250 * x) % 1080 = 0) ∧ ¬(∃ y : ℕ, y > 0 ∧ y < 54 ∧ (250 * y) % 1080 = 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_250_multiple_1080_l449_44978


namespace NUMINAMATH_GPT_transforming_sin_curve_l449_44948

theorem transforming_sin_curve :
  ∀ x : ℝ, (2 * Real.sin (x + (Real.pi / 3))) = (2 * Real.sin ((1/3) * x + (Real.pi / 3))) :=
by
  sorry

end NUMINAMATH_GPT_transforming_sin_curve_l449_44948


namespace NUMINAMATH_GPT_minimize_quadratic_expression_l449_44913

theorem minimize_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_expression_l449_44913


namespace NUMINAMATH_GPT_distinct_ints_sum_to_4r_l449_44985

theorem distinct_ints_sum_to_4r 
  (a b c d r : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d := 
by sorry

end NUMINAMATH_GPT_distinct_ints_sum_to_4r_l449_44985


namespace NUMINAMATH_GPT_correct_transformation_l449_44993

-- Conditions given in the problem
def cond_A (a : ℤ) : Prop := a + 3 = 9 → a = 3 + 9
def cond_B (x : ℤ) : Prop := 4 * x = 7 * x - 2 → 4 * x - 7 * x = 2
def cond_C (a : ℤ) : Prop := 2 * a - 2 = -6 → 2 * a = 6 + 2
def cond_D (x : ℤ) : Prop := 2 * x - 5 = 3 * x + 3 → 2 * x - 3 * x = 3 + 5

-- Prove that the transformation in condition D is correct
theorem correct_transformation : (∀ a : ℤ, ¬cond_A a) ∧ (∀ x : ℤ, ¬cond_B x) ∧ (∀ a : ℤ, ¬cond_C a) ∧ (∀ x : ℤ, cond_D x) :=
by {
  -- Proof is provided in the solution and skipped here
  sorry
}

end NUMINAMATH_GPT_correct_transformation_l449_44993


namespace NUMINAMATH_GPT_greatest_coloring_integer_l449_44989

theorem greatest_coloring_integer (α β : ℝ) (h1 : 1 < α) (h2 : α < β) :
  ∃ r : ℕ, r = 2 ∧ ∀ (f : ℕ → ℕ), ∃ x y : ℕ, x ≠ y ∧ f x = f y ∧ α ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β := 
sorry

end NUMINAMATH_GPT_greatest_coloring_integer_l449_44989


namespace NUMINAMATH_GPT_pool_length_calc_l449_44908

variable (total_water : ℕ) (drinking_cooking_water : ℕ) (shower_water : ℕ) (shower_count : ℕ)
variable (pool_width : ℕ) (pool_height : ℕ) (pool_volume : ℕ)

theorem pool_length_calc (h1 : total_water = 1000)
    (h2 : drinking_cooking_water = 100)
    (h3 : shower_water = 20)
    (h4 : shower_count = 15)
    (h5 : pool_width = 10)
    (h6 : pool_height = 6)
    (h7 : pool_volume = total_water - (drinking_cooking_water + shower_water * shower_count)) :
    pool_volume = 600 →
    pool_volume = 60 * length →
    length = 10 :=
by
  sorry

end NUMINAMATH_GPT_pool_length_calc_l449_44908


namespace NUMINAMATH_GPT_shooter_with_more_fluctuation_l449_44972

noncomputable def variance (scores : List ℕ) (mean : ℕ) : ℚ :=
  (List.sum (List.map (λ x => (x - mean) * (x - mean)) scores) : ℚ) / scores.length

theorem shooter_with_more_fluctuation :
  let scores_A := [7, 9, 8, 6, 10]
  let scores_B := [7, 8, 9, 8, 8]
  let mean := 8
  variance scores_A mean > variance scores_B mean :=
by
  sorry

end NUMINAMATH_GPT_shooter_with_more_fluctuation_l449_44972


namespace NUMINAMATH_GPT_unique_solution_l449_44942

theorem unique_solution (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  12^x + 13^y - 14^z = 2013^t → (x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_unique_solution_l449_44942


namespace NUMINAMATH_GPT_system_no_five_distinct_solutions_system_four_distinct_solutions_l449_44950

theorem system_no_five_distinct_solutions (a : ℤ) :
  ¬ ∃ x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ z₁ z₂ z₃ z₄ z₅ : ℤ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅) ∧
    (y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₁ ≠ y₄ ∧ y₁ ≠ y₅ ∧ y₂ ≠ y₃ ∧ y₂ ≠ y₄ ∧ y₂ ≠ y₅ ∧ y₃ ≠ y₄ ∧ y₃ ≠ y₅ ∧ y₄ ≠ y₅) ∧
    (z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ z₄ ≠ z₅) ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a) ∧
    (2 * y₃ * z₃ + x₃ - y₃ - z₃ = a) ∧ (2 * x₃ * z₃ - x₃ + y₃ - z₃ = a) ∧ (2 * x₃ * y₃ - x₃ - y₃ + z₃ = a) ∧
    (2 * y₄ * z₄ + x₄ - y₄ - z₄ = a) ∧ (2 * x₄ * z₄ - x₄ + y₄ - z₄ = a) ∧ (2 * x₄ * y₄ - x₄ - y₄ + z₄ = a) ∧
    (2 * y₅ * z₅ + x₅ - y₅ - z₅ = a) ∧ (2 * x₅ * z₅ - x₅ + y₅ - z₅ = a) ∧ (2 * x₅ * y₅ - x₅ - y₅ + z₅ = a) :=
sorry

theorem system_four_distinct_solutions (a : ℤ) :
  (∃ x₁ x₂ y₁ y₂ z₁ z₂ : ℤ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ z₁ ≠ z₂ ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a)) ↔
  ∃ k : ℤ, k % 2 = 1 ∧ a = (k^2 - 1) / 8 :=
sorry

end NUMINAMATH_GPT_system_no_five_distinct_solutions_system_four_distinct_solutions_l449_44950


namespace NUMINAMATH_GPT_perimeter_of_isosceles_triangle_l449_44982

theorem perimeter_of_isosceles_triangle (a b : ℕ) (h_isosceles : (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3)) :
  ∃ p : ℕ, p = 10 ∨ p = 11 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_isosceles_triangle_l449_44982


namespace NUMINAMATH_GPT_least_number_of_cubes_is_10_l449_44903

noncomputable def volume_of_block (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

noncomputable def volume_of_cube (side : ℕ) : ℕ :=
  side ^ 3

noncomputable def least_number_of_cubes (length width height : ℕ) : ℕ := 
  volume_of_block length width height / volume_of_cube (gcd_three_numbers length width height)

theorem least_number_of_cubes_is_10 : least_number_of_cubes 15 30 75 = 10 := by
  sorry

end NUMINAMATH_GPT_least_number_of_cubes_is_10_l449_44903


namespace NUMINAMATH_GPT_inequality_solution_l449_44917

theorem inequality_solution (x : ℝ) : 5 * x > 4 * x + 2 → x > 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l449_44917


namespace NUMINAMATH_GPT_unique_positive_b_discriminant_zero_l449_44933

theorem unique_positive_b_discriminant_zero (c : ℚ) : 
  (∃! b : ℚ, b > 0 ∧ (b^2 + 3*b + 1/b)^2 - 4*c = 0) ↔ c = -1/2 :=
sorry

end NUMINAMATH_GPT_unique_positive_b_discriminant_zero_l449_44933


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l449_44919

theorem relationship_y1_y2_y3 
  (y_1 y_2 y_3 : ℝ)
  (h1 : y_1 = (-2)^2 + 2*(-2) + 2)
  (h2 : y_2 = (-1)^2 + 2*(-1) + 2)
  (h3 : y_3 = 2^2 + 2*2 + 2) :
  y_2 < y_1 ∧ y_1 < y_3 := 
sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l449_44919


namespace NUMINAMATH_GPT_triangle_formation_segments_l449_44922

theorem triangle_formation_segments (a b c : ℝ) (h_sum : a + b + c = 1) (h_a : a < 1/2) (h_b : b < 1/2) (h_c : c < 1/2) : 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := 
by
  sorry

end NUMINAMATH_GPT_triangle_formation_segments_l449_44922


namespace NUMINAMATH_GPT_acute_triangle_inequality_l449_44987

theorem acute_triangle_inequality
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (h1 : 0 < A ∧ A < π/2)
  (h2 : 0 < B ∧ B < π/2)
  (h3 : 0 < C ∧ C < π/2)
  (h4 : A + B + C = π)
  (h5 : R = 1)
  (h6 : a = 2 * R * Real.sin A)
  (h7 : b = 2 * R * Real.sin B)
  (h8 : c = 2 * R * Real.sin C) :
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_inequality_l449_44987


namespace NUMINAMATH_GPT_ln_abs_a_even_iff_a_eq_zero_l449_44900

theorem ln_abs_a_even_iff_a_eq_zero (a : ℝ) :
  (∀ x : ℝ, Real.log (abs (x - a)) = Real.log (abs (-x - a))) ↔ (a = 0) :=
by
  sorry

end NUMINAMATH_GPT_ln_abs_a_even_iff_a_eq_zero_l449_44900


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l449_44902

theorem solve_eq1 (x : ℤ) : x - 2 * (5 + x) = -4 → x = -6 := by
  sorry

theorem solve_eq2 (x : ℤ) : (2 * x - 1) / 2 = 1 - (3 - x) / 4 → x = 1 := by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l449_44902


namespace NUMINAMATH_GPT_problem_l449_44943

theorem problem:
  ∀ k : Real, (2 - Real.sqrt 2 / 2 ≤ k ∧ k ≤ 2 + Real.sqrt 2 / 2) →
  (11 - 6 * Real.sqrt 2) / 4 ≤ (3 / 2 * (k - 1)^2 + 1 / 2) ∧ 
  (3 / 2 * (k - 1)^2 + 1 / 2 ≤ (11 + 6 * Real.sqrt 2) / 4) :=
by
  intros k hk
  sorry

end NUMINAMATH_GPT_problem_l449_44943


namespace NUMINAMATH_GPT_rectangle_area_l449_44951

theorem rectangle_area (x : ℝ) :
  let large_rectangle_area := (2 * x + 14) * (2 * x + 10)
  let hole_area := (4 * x - 6) * (2 * x - 4)
  let square_area := (x + 3) * (x + 3)
  large_rectangle_area - hole_area + square_area = -3 * x^2 + 82 * x + 125 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l449_44951


namespace NUMINAMATH_GPT_combo_discount_is_50_percent_l449_44976

noncomputable def combo_discount_percentage
  (ticket_cost : ℕ) (combo_cost : ℕ) (ticket_discount : ℕ) (total_savings : ℕ) : ℕ :=
  let ticket_savings := ticket_cost * ticket_discount / 100
  let combo_savings := total_savings - ticket_savings
  (combo_savings * 100) / combo_cost

theorem combo_discount_is_50_percent:
  combo_discount_percentage 10 10 20 7 = 50 :=
by
  sorry

end NUMINAMATH_GPT_combo_discount_is_50_percent_l449_44976


namespace NUMINAMATH_GPT_travel_time_reduction_l449_44940

theorem travel_time_reduction : 
  let t_initial := 19.5
  let factor_1998 := 1.30
  let factor_1999 := 1.25
  let factor_2000 := 1.20
  t_initial / factor_1998 / factor_1999 / factor_2000 = 10 := by
  sorry

end NUMINAMATH_GPT_travel_time_reduction_l449_44940


namespace NUMINAMATH_GPT_girls_points_l449_44975

theorem girls_points (g b : ℕ) (total_points : ℕ) (points_g : ℕ) (points_b : ℕ) :
  b = 9 * g ∧
  total_points = 10 * g * (10 * g - 1) ∧
  points_g = 2 * g * (10 * g - 1) ∧
  points_b = 4 * points_g ∧
  total_points = points_g + points_b
  → points_g = 18 := 
by
  sorry

end NUMINAMATH_GPT_girls_points_l449_44975


namespace NUMINAMATH_GPT_shaded_fraction_l449_44937

theorem shaded_fraction (rectangle_length rectangle_width : ℕ) (h_length : rectangle_length = 15) (h_width : rectangle_width = 20)
                        (total_area : ℕ := rectangle_length * rectangle_width)
                        (shaded_quarter : ℕ := total_area / 4)
                        (h_shaded_quarter : shaded_quarter = total_area / 5) :
  shaded_quarter / total_area = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_shaded_fraction_l449_44937


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_eq_one_third_l449_44984

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio_eq_one_third
  (h_geom : geometric_sequence a_n q)
  (h_increasing : ∀ n, a_n n < a_n (n + 1))
  (h_a1 : a_n 1 = -2)
  (h_recurrence : ∀ n, 3 * (a_n n + a_n (n + 2)) = 10 * a_n (n + 1)) :
  q = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_eq_one_third_l449_44984


namespace NUMINAMATH_GPT_total_votes_l449_44931

theorem total_votes (V : ℝ) (h1 : 0.60 * V = V - 240) : V = 600 :=
sorry

end NUMINAMATH_GPT_total_votes_l449_44931
