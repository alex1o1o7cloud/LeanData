import Mathlib

namespace NUMINAMATH_GPT_area_of_triangle_is_correct_l2359_235990

def point := ℚ × ℚ

def A : point := (4, -4)
def B : point := (-1, 1)
def C : point := (2, -7)

def vector_sub (p1 p2 : point) : point :=
(p1.1 - p2.1, p1.2 - p2.2)

def determinant (v w : point) : ℚ :=
v.1 * w.2 - v.2 * w.1

def area_of_triangle (A B C : point) : ℚ :=
(abs (determinant (vector_sub C A) (vector_sub C B))) / 2

theorem area_of_triangle_is_correct :
  area_of_triangle A B C = 12.5 :=
by sorry

end NUMINAMATH_GPT_area_of_triangle_is_correct_l2359_235990


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l2359_235997

noncomputable def arithmetic_sequence_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
n * a_1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a_1 d : ℝ) (p q : ℕ) (h₁ : p ≠ q) (h₂ : arithmetic_sequence_sum a_1 d p = q) (h₃ : arithmetic_sequence_sum a_1 d q = p) : 
arithmetic_sequence_sum a_1 d (p + q) = - (p + q) := sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l2359_235997


namespace NUMINAMATH_GPT_compare_costs_l2359_235903

def cost_X (copies: ℕ) : ℝ :=
  if copies >= 40 then
    (copies * 1.25) * 0.95
  else
    copies * 1.25

def cost_Y (copies: ℕ) : ℝ :=
  if copies >= 100 then
    copies * 2.00
  else if copies >= 60 then
    copies * 2.25
  else
    copies * 2.75

def cost_Z (copies: ℕ) : ℝ :=
  if copies >= 50 then
    (copies * 3.00) * 0.90
  else
    copies * 3.00

def cost_W (copies: ℕ) : ℝ :=
  let bulk_groups := copies / 25
  let remainder := copies % 25
  (bulk_groups * 40) + (remainder * 2.00)

theorem compare_costs : 
  cost_X 60 < cost_Y 60 ∧ 
  cost_X 60 < cost_Z 60 ∧ 
  cost_X 60 < cost_W 60 ∧
  cost_Y 60 - cost_X 60 = 63.75 ∧
  cost_Z 60 - cost_X 60 = 90.75 ∧
  cost_W 60 - cost_X 60 = 28.75 :=
  sorry

end NUMINAMATH_GPT_compare_costs_l2359_235903


namespace NUMINAMATH_GPT_paintable_wall_area_l2359_235924

/-- Given 4 bedrooms each with length 15 feet, width 11 feet, and height 9 feet,
and doorways and windows occupying 80 square feet in each bedroom,
prove that the total paintable wall area is 1552 square feet. -/
theorem paintable_wall_area
  (bedrooms : ℕ) (length width height doorway_window_area : ℕ) :
  bedrooms = 4 →
  length = 15 →
  width = 11 →
  height = 9 →
  doorway_window_area = 80 →
  4 * (2 * (length * height) + 2 * (width * height) - doorway_window_area) = 1552 :=
by
  intros bedrooms_eq length_eq width_eq height_eq doorway_window_area_eq
  -- Definition of the problem conditions
  have bedrooms_def : bedrooms = 4 := bedrooms_eq
  have length_def : length = 15 := length_eq
  have width_def : width = 11 := width_eq
  have height_def : height = 9 := height_eq
  have doorway_window_area_def : doorway_window_area = 80 := doorway_window_area_eq
  -- Assertion of the correct answer
  sorry

end NUMINAMATH_GPT_paintable_wall_area_l2359_235924


namespace NUMINAMATH_GPT_largest_three_digit_multiple_of_six_with_sum_fifteen_l2359_235930

theorem largest_three_digit_multiple_of_six_with_sum_fifteen : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n % 6 = 0) ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ (m % 6 = 0) ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
  sorry

end NUMINAMATH_GPT_largest_three_digit_multiple_of_six_with_sum_fifteen_l2359_235930


namespace NUMINAMATH_GPT_total_emails_received_l2359_235959

theorem total_emails_received (E : ℝ)
    (h1 : (3/5) * (3/4) * E = 180) :
    E = 400 :=
sorry

end NUMINAMATH_GPT_total_emails_received_l2359_235959


namespace NUMINAMATH_GPT_days_gumballs_last_l2359_235987

def pairs_day_1 := 3
def gumballs_per_pair := 9
def gumballs_day_1 := pairs_day_1 * gumballs_per_pair

def pairs_day_2 := pairs_day_1 * 2
def gumballs_day_2 := pairs_day_2 * gumballs_per_pair

def pairs_day_3 := pairs_day_2 - 1
def gumballs_day_3 := pairs_day_3 * gumballs_per_pair

def total_gumballs := gumballs_day_1 + gumballs_day_2 + gumballs_day_3
def gumballs_eaten_per_day := 3

theorem days_gumballs_last : total_gumballs / gumballs_eaten_per_day = 42 :=
by
  sorry

end NUMINAMATH_GPT_days_gumballs_last_l2359_235987


namespace NUMINAMATH_GPT_gloria_turtle_time_l2359_235934

theorem gloria_turtle_time (g_time : ℕ) (george_time : ℕ) (gloria_time : ℕ) 
  (h1 : g_time = 6) 
  (h2 : george_time = g_time - 2)
  (h3 : gloria_time = 2 * george_time) : 
  gloria_time = 8 :=
sorry

end NUMINAMATH_GPT_gloria_turtle_time_l2359_235934


namespace NUMINAMATH_GPT_find_m_range_l2359_235947

def proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0) ∧ (1 > 0)

def proposition_q (m : ℝ) : Prop :=
  16 * (m - 2)^2 - 16 < 0

theorem find_m_range : {m : ℝ // proposition_p m ∧ proposition_q m} = {m : ℝ // 2 < m ∧ m < 3} :=
by
  sorry

end NUMINAMATH_GPT_find_m_range_l2359_235947


namespace NUMINAMATH_GPT_inequality_holds_l2359_235923

theorem inequality_holds (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) (h_mul : a * b * c * d = 1) :
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (12 / (a + b + c + d)) ≥ 7 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l2359_235923


namespace NUMINAMATH_GPT_dorothy_needs_more_money_l2359_235951

structure Person :=
  (age : ℕ)

def Discount (age : ℕ) : ℝ :=
  if age <= 11 then 0.5 else
  if age >= 65 then 0.8 else
  if 12 <= age && age <= 18 then 0.7 else 1.0

def ticketCost (age : ℕ) : ℝ :=
  (10 : ℝ) * Discount age

def specialExhibitCost : ℝ := 5

def totalCost (family : List Person) : ℝ :=
  (family.map (λ p => ticketCost p.age + specialExhibitCost)).sum

def salesTaxRate : ℝ := 0.1

def finalCost (family : List Person) : ℝ :=
  let total := totalCost family
  total + (total * salesTaxRate)

def dorothy_money_after_trip (dorothy_money : ℝ) (family : List Person) : ℝ :=
  dorothy_money - finalCost family

theorem dorothy_needs_more_money :
  dorothy_money_after_trip 70 [⟨15⟩, ⟨10⟩, ⟨40⟩, ⟨42⟩, ⟨65⟩] = -1.5 := by
  sorry

end NUMINAMATH_GPT_dorothy_needs_more_money_l2359_235951


namespace NUMINAMATH_GPT_find_second_divisor_l2359_235942

theorem find_second_divisor :
  ∃ y : ℝ, (320 / 2) / y = 53.33 ∧ y = 160 / 53.33 :=
by
  sorry

end NUMINAMATH_GPT_find_second_divisor_l2359_235942


namespace NUMINAMATH_GPT_right_triangle_area_is_integer_l2359_235963

theorem right_triangle_area_is_integer (a b : ℕ) (h1 : ∃ (A : ℕ), A = (1 / 2 : ℚ) * ↑a * ↑b) : (a % 2 = 0) ∨ (b % 2 = 0) :=
sorry

end NUMINAMATH_GPT_right_triangle_area_is_integer_l2359_235963


namespace NUMINAMATH_GPT_intersection_A_B_l2359_235901

def set_A (x : ℝ) : Prop := 2 * x + 1 > 0
def set_B (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_A_B : 
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2359_235901


namespace NUMINAMATH_GPT_sum_of_first_n_terms_l2359_235946

theorem sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + 2 * a 2 = 3)
  (h2 : ∀ n, a (n + 1) = a n + 2) :
  ∀ n, S n = n * (n - 4 / 3) := 
sorry

end NUMINAMATH_GPT_sum_of_first_n_terms_l2359_235946


namespace NUMINAMATH_GPT_part2_l2359_235952

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) - x

theorem part2 (a : ℝ) (h : a > 0) (x : ℝ) : f a x < (a - 1) * Real.log a + a^2 := 
  sorry

end NUMINAMATH_GPT_part2_l2359_235952


namespace NUMINAMATH_GPT_golden_ratio_problem_l2359_235994

noncomputable def m := 2 * Real.sin (Real.pi * 18 / 180)
noncomputable def n := 4 - m^2
noncomputable def target_expression := m * Real.sqrt n / (2 * (Real.cos (Real.pi * 27 / 180))^2 - 1)

theorem golden_ratio_problem :
  target_expression = 2 :=
by
  -- Proof will be placed here
  sorry

end NUMINAMATH_GPT_golden_ratio_problem_l2359_235994


namespace NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l2359_235971

theorem factorize_problem_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := 
by sorry

theorem factorize_problem_2 (x y : ℝ) : 2 * x^3 - 12 * x^2 * y + 18 * x * y^2 = 2 * x * (x - 3 * y)^2 :=
by sorry

end NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l2359_235971


namespace NUMINAMATH_GPT_range_of_a_l2359_235960

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
by sorry

end NUMINAMATH_GPT_range_of_a_l2359_235960


namespace NUMINAMATH_GPT_problem_l2359_235926

-- Define the concept of reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the conditions in the problem
def condition1 : Prop := reciprocal 1.5 = 2/3
def condition2 : Prop := reciprocal 1 = 1

-- Theorem stating our goals
theorem problem : condition1 ∧ condition2 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l2359_235926


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l2359_235999

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25 / 6 := 
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l2359_235999


namespace NUMINAMATH_GPT_clock_angle_7_35_l2359_235908

noncomputable def hour_angle (hours : ℤ) (minutes : ℤ) : ℝ :=
  (hours * 30 + (minutes * 30) / 60 : ℝ)

noncomputable def minute_angle (minutes : ℤ) : ℝ :=
  (minutes * 360 / 60 : ℝ)

noncomputable def angle_between (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

theorem clock_angle_7_35 : angle_between (hour_angle 7 35) (minute_angle 35) = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_7_35_l2359_235908


namespace NUMINAMATH_GPT_remainder_when_divided_by_11_l2359_235936

theorem remainder_when_divided_by_11 {k x : ℕ} (h : x = 66 * k + 14) : x % 11 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_11_l2359_235936


namespace NUMINAMATH_GPT_solve_equation_l2359_235996

theorem solve_equation (m x : ℝ) (hm_pos : m > 0) (hm_ne_one : m ≠ 1) :
  7.320 * m^(1 + Real.log x / Real.log 3) + m^(1 - Real.log x / Real.log 3) = m^2 + 1 ↔ x = 3 ∨ x = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2359_235996


namespace NUMINAMATH_GPT_largest_piece_length_l2359_235956

theorem largest_piece_length (v : ℝ) (hv : v + (3/2) * v + (9/4) * v = 95) : 
  (9/4) * v = 45 :=
by sorry

end NUMINAMATH_GPT_largest_piece_length_l2359_235956


namespace NUMINAMATH_GPT_total_fish_purchased_l2359_235916

/-- Definition of the conditions based on Roden's visits to the pet shop. -/
def first_visit_goldfish := 15
def first_visit_bluefish := 7
def second_visit_goldfish := 10
def second_visit_bluefish := 12
def second_visit_greenfish := 5
def third_visit_goldfish := 3
def third_visit_bluefish := 7
def third_visit_greenfish := 9

/-- Proof statement in Lean 4. -/
theorem total_fish_purchased :
  first_visit_goldfish + first_visit_bluefish +
  second_visit_goldfish + second_visit_bluefish + second_visit_greenfish +
  third_visit_goldfish + third_visit_bluefish + third_visit_greenfish = 68 :=
by
  sorry

end NUMINAMATH_GPT_total_fish_purchased_l2359_235916


namespace NUMINAMATH_GPT_solve_inequality_l2359_235975

variable {c : ℝ}
variable (h_c_ne_2 : c ≠ 2)

theorem solve_inequality :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - (1 + 2) * x + 2 ≤ 0) ∧
  (c > 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x > c ∨ x < 2)) ∧
  (c < 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x < c ∨ x > 2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2359_235975


namespace NUMINAMATH_GPT_baker_cakes_l2359_235950

theorem baker_cakes (C : ℕ) (h1 : 154 = 78 + 76) (h2 : C = 78) : C = 78 :=
sorry

end NUMINAMATH_GPT_baker_cakes_l2359_235950


namespace NUMINAMATH_GPT_simplify_sqrt_eight_l2359_235945

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_simplify_sqrt_eight_l2359_235945


namespace NUMINAMATH_GPT_intersection_A_B_l2359_235958

def A : Set ℤ := {-1, 1, 3, 5, 7}
def B : Set ℝ := { x | 2^x > 2 * Real.sqrt 2 }

theorem intersection_A_B :
  A ∩ { x : ℤ | x > 3 / 2 } = {3, 5, 7} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2359_235958


namespace NUMINAMATH_GPT_sum_equidistant_terms_l2359_235988

def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n m : ℕ, (n < m) → a (n+1) - a n = a (m+1) - a m

variable {a : ℕ → ℤ}

theorem sum_equidistant_terms (h_seq : is_arithmetic_sequence a)
  (h_4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end NUMINAMATH_GPT_sum_equidistant_terms_l2359_235988


namespace NUMINAMATH_GPT_division_addition_l2359_235973

theorem division_addition : (-300) / (-75) + 10 = 14 := by
  sorry

end NUMINAMATH_GPT_division_addition_l2359_235973


namespace NUMINAMATH_GPT_problem1_part1_problem1_part2_l2359_235900

theorem problem1_part1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) := 
sorry

theorem problem1_part2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 := 
sorry

end NUMINAMATH_GPT_problem1_part1_problem1_part2_l2359_235900


namespace NUMINAMATH_GPT_sum_of_numbers_l2359_235941

theorem sum_of_numbers (avg : ℝ) (count : ℕ) (h_avg : avg = 5.7) (h_count : count = 8) : (avg * count = 45.6) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l2359_235941


namespace NUMINAMATH_GPT_geometric_series_sum_eq_l2359_235970

theorem geometric_series_sum_eq :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5
  (∀ S_n, S_n = a * (1 - r^n) / (1 - r) → S_n = 1 / 3) :=
by
  intro a r n S_n
  sorry

end NUMINAMATH_GPT_geometric_series_sum_eq_l2359_235970


namespace NUMINAMATH_GPT_marbles_selection_l2359_235937

theorem marbles_selection : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ marbles : Finset ℕ, marbles.card = 15 ∧
  ∃ rgb : Finset ℕ, rgb ⊆ marbles ∧ rgb.card = 3 ∧
  ∃ yellow : ℕ, yellow ∈ marbles ∧ yellow ∉ rgb ∧ 
  ∀ (selection : Finset ℕ), selection.card = 5 →
  (∃ red green blue : ℕ, red ∈ rgb ∧ green ∈ rgb ∧ blue ∈ rgb ∧ 
  (red ∈ selection ∨ green ∈ selection ∨ blue ∈ selection) ∧ yellow ∉ selection) → 
  (selection.card = 5) :=
by
  sorry

end NUMINAMATH_GPT_marbles_selection_l2359_235937


namespace NUMINAMATH_GPT_triangle_sin_a_triangle_area_l2359_235957

theorem triangle_sin_a (B : ℝ) (a b c : ℝ) (hB : B = π / 4)
  (h_bc : b = Real.sqrt 5 ∧ c = Real.sqrt 2 ∨ a = 3 ∧ c = Real.sqrt 2) :
  Real.sin A = (3 * Real.sqrt 10) / 10 :=
sorry

theorem triangle_area (B a b c : ℝ) (hB : B = π / 4) (hb : b = Real.sqrt 5)
  (h_ac : a + c = 3) : 1 / 2 * a * c * Real.sin B = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_triangle_sin_a_triangle_area_l2359_235957


namespace NUMINAMATH_GPT_four_digit_not_multiples_of_4_or_9_l2359_235929

theorem four_digit_not_multiples_of_4_or_9 (h1 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 4 ∣ n ↔ (250 ≤ n / 4 ∧ n / 4 ≤ 2499))
                                         (h2 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 9 ∣ n ↔ (112 ≤ n / 9 ∧ n / 9 ≤ 1111))
                                         (h3 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 36 ∣ n ↔ (28 ≤ n / 36 ∧ n / 36 ≤ 277)) :
                                         (9000 - ((2250 : ℕ) + 1000 - 250)) = 6000 :=
by sorry

end NUMINAMATH_GPT_four_digit_not_multiples_of_4_or_9_l2359_235929


namespace NUMINAMATH_GPT_closed_polygon_inequality_l2359_235920

noncomputable def length_eq (A B C D : ℝ × ℝ × ℝ) (l : ℝ) : Prop :=
  dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D A = l

theorem closed_polygon_inequality 
  (A B C D P : ℝ × ℝ × ℝ) (l : ℝ)
  (hABCD : length_eq A B C D l) :
  dist P A < dist P B + dist P C + dist P D :=
sorry

end NUMINAMATH_GPT_closed_polygon_inequality_l2359_235920


namespace NUMINAMATH_GPT_domain_of_sqrt_sin_l2359_235982

open Real Set

noncomputable def domain_sqrt_sine : Set ℝ :=
  {x | ∃ (k : ℤ), 2 * π * k + π / 6 ≤ x ∧ x ≤ 2 * π * k + 5 * π / 6}

theorem domain_of_sqrt_sin (x : ℝ) :
  (∃ y, y = sqrt (2 * sin x - 1)) ↔ x ∈ domain_sqrt_sine :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_sin_l2359_235982


namespace NUMINAMATH_GPT_distributive_laws_none_hold_l2359_235983

def star (a b : ℝ) : ℝ := a + b + a * b

theorem distributive_laws_none_hold (x y z : ℝ) :
  ¬ (x * (y + z) = (x * y) + (x * z)) ∧
  ¬ (x + (y * z) = (x + y) * (x + z)) ∧
  ¬ (x * (y * z) = (x * y) * (x * z)) :=
by
  sorry

end NUMINAMATH_GPT_distributive_laws_none_hold_l2359_235983


namespace NUMINAMATH_GPT_tom_tim_typing_ratio_l2359_235961

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
by
  sorry

end NUMINAMATH_GPT_tom_tim_typing_ratio_l2359_235961


namespace NUMINAMATH_GPT_roots_of_quadratic_function_l2359_235932

variable (a b x : ℝ)

theorem roots_of_quadratic_function (h : a + b = 0) : (b * x * x + a * x = 0) → (x = 0 ∨ x = 1) :=
by {sorry}

end NUMINAMATH_GPT_roots_of_quadratic_function_l2359_235932


namespace NUMINAMATH_GPT_transformed_roots_l2359_235939

noncomputable def specific_polynomial : Polynomial ℝ :=
  Polynomial.C 1 - Polynomial.C 4 * Polynomial.X + Polynomial.C 6 * Polynomial.X ^ 2 - Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 4

theorem transformed_roots (a b c d : ℝ) :
  (a^4 - b*a - 5 = 0) ∧ (b^4 - b*b - 5 = 0) ∧ (c^4 - b*c - 5 = 0) ∧ (d^4 - b*d - 5 = 0) →
  specific_polynomial.eval ((a + b + c) / d)^2 = 0 ∧
  specific_polynomial.eval ((a + b + d) / c)^2 = 0 ∧
  specific_polynomial.eval ((a + c + d) / b)^2 = 0 ∧
  specific_polynomial.eval ((b + c + d) / a)^2 = 0 :=
  by
    sorry

end NUMINAMATH_GPT_transformed_roots_l2359_235939


namespace NUMINAMATH_GPT_div_by_240_l2359_235993

theorem div_by_240 (a b c d : ℕ) : 240 ∣ (a ^ (4 * b + d) - a ^ (4 * c + d)) :=
sorry

end NUMINAMATH_GPT_div_by_240_l2359_235993


namespace NUMINAMATH_GPT_sum_ab_equals_five_l2359_235915

-- Definitions for conditions
variables {a b : ℝ}

-- Assumption that establishes the solution set for the quadratic inequality
axiom quadratic_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ x^2 + b * x - a < 0

-- Statement to be proved
theorem sum_ab_equals_five : a + b = 5 :=
sorry

end NUMINAMATH_GPT_sum_ab_equals_five_l2359_235915


namespace NUMINAMATH_GPT_completing_the_square_l2359_235984

theorem completing_the_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) → ((x - 1)^2 = 6) :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_l2359_235984


namespace NUMINAMATH_GPT_power_modulo_remainder_l2359_235968

theorem power_modulo_remainder :
  (17 ^ 2046) % 23 = 22 := 
sorry

end NUMINAMATH_GPT_power_modulo_remainder_l2359_235968


namespace NUMINAMATH_GPT_pizza_area_difference_l2359_235979

def hueys_hip_pizza (small_size : ℕ) (small_cost : ℕ) (large_size : ℕ) (large_cost : ℕ) : ℕ :=
  let small_area := small_size * small_size
  let large_area := large_size * large_size
  let individual_money := 30
  let pooled_money := 2 * individual_money

  let individual_small_total_area := (individual_money / small_cost) * small_area * 2
  let pooled_large_total_area := (pooled_money / large_cost) * large_area

  pooled_large_total_area - individual_small_total_area

theorem pizza_area_difference :
  hueys_hip_pizza 6 10 9 20 = 27 :=
by
  sorry

end NUMINAMATH_GPT_pizza_area_difference_l2359_235979


namespace NUMINAMATH_GPT_sum_of_asymptotes_l2359_235906

theorem sum_of_asymptotes :
  let c := -3/2
  let d := -1
  c + d = -5/2 :=
by
  -- Definitions corresponding to the problem conditions
  let c := -3/2
  let d := -1
  -- Statement of the theorem
  show c + d = -5/2
  sorry

end NUMINAMATH_GPT_sum_of_asymptotes_l2359_235906


namespace NUMINAMATH_GPT_find_integer_n_l2359_235986

theorem find_integer_n (n : ℤ) :
  (⌊ (n^2 : ℤ) / 9 ⌋ - ⌊ n / 3 ⌋^2 = 3) → (n = 8 ∨ n = 10) :=
  sorry

end NUMINAMATH_GPT_find_integer_n_l2359_235986


namespace NUMINAMATH_GPT_min_n_Sn_greater_1020_l2359_235989

theorem min_n_Sn_greater_1020 : ∃ n : ℕ, (n ≥ 0) ∧ (2^(n+1) - 2 - n > 1020) ∧ ∀ m : ℕ, (m ≥ 0) ∧ (m < n) → (2^(m+1) - 2 - m ≤ 1020) :=
by
  sorry

end NUMINAMATH_GPT_min_n_Sn_greater_1020_l2359_235989


namespace NUMINAMATH_GPT_mixture_problem_l2359_235962

theorem mixture_problem :
  ∀ (x P : ℝ), 
    let initial_solution := 70
    let initial_percentage := 0.20
    let final_percentage := 0.40
    let final_amount := 70
    (x = 70) →
    (initial_percentage * initial_solution + P * x = final_percentage * (initial_solution + x)) →
    (P = 0.60) :=
by
  intros x P initial_solution initial_percentage final_percentage final_amount hx h_eq
  sorry

end NUMINAMATH_GPT_mixture_problem_l2359_235962


namespace NUMINAMATH_GPT_project_completion_time_l2359_235918

theorem project_completion_time
  (A_time B_time : ℕ) 
  (hA : A_time = 20)
  (hB : B_time = 20)
  (A_quit_days : ℕ) 
  (hA_quit : A_quit_days = 10) :
  ∃ x : ℕ, (x - A_quit_days) * (1 / A_time : ℚ) + (x * (1 / B_time : ℚ)) = 1 ∧ x = 15 := by
  sorry

end NUMINAMATH_GPT_project_completion_time_l2359_235918


namespace NUMINAMATH_GPT_total_non_overlapping_area_of_squares_l2359_235992

theorem total_non_overlapping_area_of_squares 
  (side_length : ℕ) 
  (num_squares : ℕ)
  (overlapping_areas_count : ℕ)
  (overlapping_width : ℕ)
  (overlapping_height : ℕ)
  (total_area_with_overlap: ℕ)
  (final_missed_patch_ratio: ℕ)
  (final_adjustment: ℕ) 
  (total_area: ℕ :=  total_area_with_overlap-final_missed_patch_ratio ):
  side_length = 2 ∧ 
  num_squares = 4 ∧ 
  overlapping_areas_count = 3 ∧ 
  overlapping_width = 1 ∧ 
  overlapping_height = 2 ∧
  total_area_with_overlap = 16- 3  ∧
  final_missed_patch_ratio = 3-> 
  total_area = 13 := 
 by sorry

end NUMINAMATH_GPT_total_non_overlapping_area_of_squares_l2359_235992


namespace NUMINAMATH_GPT_simplify_expression_l2359_235938

theorem simplify_expression :
  (123 / 999) * 27 = 123 / 37 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2359_235938


namespace NUMINAMATH_GPT_birdseed_weekly_consumption_l2359_235974

def parakeets := 3
def parakeet_consumption := 2
def parrots := 2
def parrot_consumption := 14
def finches := 4
def finch_consumption := parakeet_consumption / 2
def canaries := 5
def canary_consumption := 3
def african_grey_parrots := 2
def african_grey_parrot_consumption := 18
def toucans := 3
def toucan_consumption := 25

noncomputable def daily_consumption := 
  parakeets * parakeet_consumption +
  parrots * parrot_consumption +
  finches * finch_consumption +
  canaries * canary_consumption +
  african_grey_parrots * african_grey_parrot_consumption +
  toucans * toucan_consumption

noncomputable def weekly_consumption := 7 * daily_consumption

theorem birdseed_weekly_consumption : weekly_consumption = 1148 := by
  sorry

end NUMINAMATH_GPT_birdseed_weekly_consumption_l2359_235974


namespace NUMINAMATH_GPT_no_solution_exists_l2359_235969

theorem no_solution_exists :
  ¬ ∃ a b : ℝ, a^2 + 3 * b^2 + 2 = 3 * a * b :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l2359_235969


namespace NUMINAMATH_GPT_lily_typing_speed_l2359_235933

-- Define the conditions
def wordsTyped : ℕ := 255
def totalMinutes : ℕ := 19
def breakTime : ℕ := 2
def typingInterval : ℕ := 10
def effectiveMinutes : ℕ := totalMinutes - breakTime

-- Define the number of words typed in effective minutes
def wordsPerMinute (words : ℕ) (minutes : ℕ) : ℕ := words / minutes

-- Statement to be proven
theorem lily_typing_speed : wordsPerMinute wordsTyped effectiveMinutes = 15 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_lily_typing_speed_l2359_235933


namespace NUMINAMATH_GPT_find_a_values_l2359_235995

def setA (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.snd - 3) / (p.fst - 2) = a + 1}

def setB (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (a^2 - 1) * p.fst + (a - 1) * p.snd = 15}

def sets_disjoint (A B : Set (ℝ × ℝ)) : Prop := ∀ p : ℝ × ℝ, p ∉ A ∪ B

theorem find_a_values (a : ℝ) :
  sets_disjoint (setA a) (setB a) ↔ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
sorry

end NUMINAMATH_GPT_find_a_values_l2359_235995


namespace NUMINAMATH_GPT_arun_weight_average_l2359_235981

theorem arun_weight_average :
  ∀ (w : ℝ), (w > 61 ∧ w < 72) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 64) →
  (w = 62 ∨ w = 63) →
  (62 + 63) / 2 = 62.5 :=
by
  intros w h1 h2
  sorry

end NUMINAMATH_GPT_arun_weight_average_l2359_235981


namespace NUMINAMATH_GPT_stream_speed_l2359_235913

theorem stream_speed :
  ∀ (v : ℝ),
  (12 - v) / (12 + v) = 1 / 2 →
  v = 4 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l2359_235913


namespace NUMINAMATH_GPT_other_x_intercept_l2359_235904

noncomputable def quadratic_function_vertex :=
  ∃ (a b c : ℝ), ∀ (x : ℝ), (a ≠ 0) →
  (5, -3) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) ∧
  (x = 1) ∧ (a * x^2 + b * x + c = 0) →
  ∃ (x2 : ℝ), x2 = 9

theorem other_x_intercept :
  quadratic_function_vertex :=
sorry

end NUMINAMATH_GPT_other_x_intercept_l2359_235904


namespace NUMINAMATH_GPT_campers_afternoon_l2359_235976

theorem campers_afternoon (x : ℕ) 
  (h1 : 44 = x + 5) : 
  x = 39 := 
by
  sorry

end NUMINAMATH_GPT_campers_afternoon_l2359_235976


namespace NUMINAMATH_GPT_cube_difference_div_l2359_235944

theorem cube_difference_div (a b : ℕ) (h_a : a = 64) (h_b : b = 27) : 
  (a^3 - b^3) / (a - b) = 6553 := by
  sorry

end NUMINAMATH_GPT_cube_difference_div_l2359_235944


namespace NUMINAMATH_GPT_seq_eq_a1_b1_l2359_235917

theorem seq_eq_a1_b1 {a b : ℕ → ℝ} 
  (h1 : ∀ n, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n, a n > 0) :
  a 1 = b 1 := 
sorry

end NUMINAMATH_GPT_seq_eq_a1_b1_l2359_235917


namespace NUMINAMATH_GPT_samantha_probability_l2359_235972

noncomputable def probability_of_selecting_yellow_apples 
  (total_apples : ℕ) (yellow_apples : ℕ) (selection_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_apples selection_size
  let yellow_ways := Nat.choose yellow_apples selection_size
  yellow_ways / total_ways

theorem samantha_probability : 
  probability_of_selecting_yellow_apples 10 5 3 = 1 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_samantha_probability_l2359_235972


namespace NUMINAMATH_GPT_ce_length_l2359_235948

noncomputable def CE_in_parallelogram (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) : ℝ :=
  280

theorem ce_length (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) :
  CE_in_parallelogram AB AD BD AB_eq AD_eq BD_eq = 280 :=
by
  sorry

end NUMINAMATH_GPT_ce_length_l2359_235948


namespace NUMINAMATH_GPT_find_principal_amount_l2359_235991

noncomputable def principal_amount (SI R T : ℝ) : ℝ :=
  SI / (R * T / 100)

theorem find_principal_amount :
  principal_amount 4052.25 9 5 = 9005 := by
sorry

end NUMINAMATH_GPT_find_principal_amount_l2359_235991


namespace NUMINAMATH_GPT_olympic_high_school_amc10_l2359_235949

/-- At Olympic High School, 2/5 of the freshmen and 4/5 of the sophomores took the AMC-10.
    Given that the number of freshmen and sophomore contestants was the same, there are twice as many freshmen as sophomores. -/
theorem olympic_high_school_amc10 (f s : ℕ) (hf : f > 0) (hs : s > 0)
  (contest_equal : (2 / 5 : ℚ)*f = (4 / 5 : ℚ)*s) : f = 2 * s :=
by
  sorry

end NUMINAMATH_GPT_olympic_high_school_amc10_l2359_235949


namespace NUMINAMATH_GPT_profit_difference_l2359_235953

-- Define the initial investments
def investment_A : ℚ := 8000
def investment_B : ℚ := 10000
def investment_C : ℚ := 12000

-- Define B's profit share
def profit_B : ℚ := 1700

-- Prove that the difference between A and C's profit shares is Rs. 680
theorem profit_difference (investment_A investment_B investment_C profit_B: ℚ) (hA : investment_A = 8000) (hB : investment_B = 10000) (hC : investment_C = 12000) (pB : profit_B = 1700) :
    let ratio_A : ℚ := 4
    let ratio_B : ℚ := 5
    let ratio_C : ℚ := 6
    let part_value : ℚ := profit_B / ratio_B
    let profit_A : ℚ := ratio_A * part_value
    let profit_C : ℚ := ratio_C * part_value
    profit_C - profit_A = 680 := 
by
  sorry

end NUMINAMATH_GPT_profit_difference_l2359_235953


namespace NUMINAMATH_GPT_total_number_of_animals_is_304_l2359_235921

theorem total_number_of_animals_is_304
    (dogs frogs : ℕ) 
    (h1 : frogs = 160) 
    (h2 : frogs = 2 * dogs) 
    (cats : ℕ) 
    (h3 : cats = dogs - (dogs / 5)) :
  cats + dogs + frogs = 304 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_animals_is_304_l2359_235921


namespace NUMINAMATH_GPT_mark_initial_fries_l2359_235928

variable (Sally_fries_before : ℕ)
variable (Sally_fries_after : ℕ)
variable (Mark_fries_given : ℕ)
variable (Mark_fries_initial : ℕ)

theorem mark_initial_fries (h1 : Sally_fries_before = 14) (h2 : Sally_fries_after = 26) (h3 : Mark_fries_given = Sally_fries_after - Sally_fries_before) (h4 : Mark_fries_given = 1/3 * Mark_fries_initial) : Mark_fries_initial = 36 :=
by sorry

end NUMINAMATH_GPT_mark_initial_fries_l2359_235928


namespace NUMINAMATH_GPT_inequality_proof_l2359_235967

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c) ^ 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2359_235967


namespace NUMINAMATH_GPT_more_balloons_l2359_235964

theorem more_balloons (you_balloons : ℕ) (friend_balloons : ℕ) (h_you : you_balloons = 7) (h_friend : friend_balloons = 5) : 
  you_balloons - friend_balloons = 2 :=
sorry

end NUMINAMATH_GPT_more_balloons_l2359_235964


namespace NUMINAMATH_GPT_cost_difference_proof_l2359_235912

noncomputable def sailboat_daily_rent : ℕ := 60
noncomputable def ski_boat_hourly_rent : ℕ := 80
noncomputable def sailboat_hourly_fuel_cost : ℕ := 10
noncomputable def ski_boat_hourly_fuel_cost : ℕ := 20
noncomputable def discount : ℕ := 10

noncomputable def rent_time : ℕ := 3
noncomputable def rent_days : ℕ := 2

noncomputable def ken_sailboat_rent_cost :=
  sailboat_daily_rent * rent_days - sailboat_daily_rent * discount / 100

noncomputable def ken_sailboat_fuel_cost :=
  sailboat_hourly_fuel_cost * rent_time * rent_days

noncomputable def ken_total_cost :=
  ken_sailboat_rent_cost + ken_sailboat_fuel_cost

noncomputable def aldrich_ski_boat_rent_cost :=
  ski_boat_hourly_rent * rent_time * rent_days - (ski_boat_hourly_rent * rent_time * discount / 100)

noncomputable def aldrich_ski_boat_fuel_cost :=
  ski_boat_hourly_fuel_cost * rent_time * rent_days

noncomputable def aldrich_total_cost :=
  aldrich_ski_boat_rent_cost + aldrich_ski_boat_fuel_cost

noncomputable def cost_difference :=
  aldrich_total_cost - ken_total_cost

theorem cost_difference_proof : cost_difference = 402 := by
  sorry

end NUMINAMATH_GPT_cost_difference_proof_l2359_235912


namespace NUMINAMATH_GPT_ellipse_area_l2359_235985

theorem ellipse_area
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (a : { endpoints_major_axis : (ℝ × ℝ) × (ℝ × ℝ) // endpoints_major_axis = ((x1, y1), (x2, y2)) })
  (b : { point_on_ellipse : ℝ × ℝ // point_on_ellipse = (x3, y3) }) :
  (-5 : ℝ) = x1 ∧ (2 : ℝ) = y1 ∧ (15 : ℝ) = x2 ∧ (2 : ℝ) = y2 ∧
  (8 : ℝ) = x3 ∧ (6 : ℝ) = y3 → 
  100 * Real.pi * Real.sqrt (16 / 91) = 100 * Real.pi * Real.sqrt (16 / 91) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_area_l2359_235985


namespace NUMINAMATH_GPT_sum_four_digit_integers_ending_in_zero_l2359_235935

def arithmetic_series_sum (a l d : ℕ) : ℕ := 
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_four_digit_integers_ending_in_zero : 
  arithmetic_series_sum 1000 9990 10 = 4945500 :=
by
  sorry

end NUMINAMATH_GPT_sum_four_digit_integers_ending_in_zero_l2359_235935


namespace NUMINAMATH_GPT_fewest_printers_l2359_235977

theorem fewest_printers (x y : ℕ) (h1 : 350 * x = 200 * y) : x + y = 11 := 
by
  sorry

end NUMINAMATH_GPT_fewest_printers_l2359_235977


namespace NUMINAMATH_GPT_joan_socks_remaining_l2359_235943

-- Definitions based on conditions
def total_socks : ℕ := 1200
def white_socks : ℕ := total_socks / 4
def blue_socks : ℕ := total_socks * 3 / 8
def red_socks : ℕ := total_socks / 6
def green_socks : ℕ := total_socks / 12
def white_socks_lost : ℕ := white_socks / 3
def blue_socks_sold : ℕ := blue_socks / 2
def remaining_white_socks : ℕ := white_socks - white_socks_lost
def remaining_blue_socks : ℕ := blue_socks - blue_socks_sold

-- Theorem to prove the total number of remaining socks
theorem joan_socks_remaining :
  remaining_white_socks + remaining_blue_socks + red_socks + green_socks = 725 := by
  sorry

end NUMINAMATH_GPT_joan_socks_remaining_l2359_235943


namespace NUMINAMATH_GPT_study_days_l2359_235922

theorem study_days (chapters worksheets : ℕ) (chapter_hours worksheet_hours daily_study_hours hourly_break
                     snack_breaks_count snack_break time_lunch effective_hours : ℝ)
  (h1 : chapters = 2) 
  (h2 : worksheets = 4) 
  (h3 : chapter_hours = 3) 
  (h4 : worksheet_hours = 1.5) 
  (h5 : daily_study_hours = 4) 
  (h6 : hourly_break = 10 / 60) 
  (h7 : snack_breaks_count = 3) 
  (h8 : snack_break = 10 / 60) 
  (h9 : time_lunch = 30 / 60)
  (h10 : effective_hours = daily_study_hours - (hourly_break * (daily_study_hours - 1)) - (snack_breaks_count * snack_break) - time_lunch)
  : (chapters * chapter_hours + worksheets * worksheet_hours) / effective_hours = 4.8 :=
by
  sorry

end NUMINAMATH_GPT_study_days_l2359_235922


namespace NUMINAMATH_GPT_value_of_expression_l2359_235927

theorem value_of_expression (x : ℝ) (h : x = 5) : (x^2 + x - 12) / (x - 4) = 18 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l2359_235927


namespace NUMINAMATH_GPT_find_a_l2359_235925

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : (min (log_a a 2) (log_a a 4)) * (max (log_a a 2) (log_a a 4)) = 2) : 
  a = (1 / 2) ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l2359_235925


namespace NUMINAMATH_GPT_sunflower_packets_correct_l2359_235965

namespace ShyneGarden

-- Define the given conditions
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def eggplant_packets_bought := 4
def total_plants := 116

-- Define the function to calculate the number of sunflower packets bought
def sunflower_packets_bought (eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants : ℕ) : ℕ :=
  (total_plants - (eggplant_packets_bought * eggplants_per_packet)) / sunflowers_per_packet

-- State the theorem to prove the number of sunflower packets
theorem sunflower_packets_correct :
  sunflower_packets_bought eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants = 6 :=
by
  sorry

end ShyneGarden

end NUMINAMATH_GPT_sunflower_packets_correct_l2359_235965


namespace NUMINAMATH_GPT_fertilizer_percentage_l2359_235911

theorem fertilizer_percentage (total_volume : ℝ) (vol_74 : ℝ) (vol_53 : ℝ) (perc_74 : ℝ) (perc_53 : ℝ) (final_perc : ℝ) :
  total_volume = 42 ∧ vol_74 = 20 ∧ vol_53 = total_volume - vol_74 ∧ perc_74 = 0.74 ∧ perc_53 = 0.53 
  → final_perc = ((vol_74 * perc_74 + vol_53 * perc_53) / total_volume) * 100
  → final_perc = 63.0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fertilizer_percentage_l2359_235911


namespace NUMINAMATH_GPT_inequality_condition_l2359_235978

theorem inequality_condition {a b c : ℝ} :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (Real.sqrt (a^2 + b^2) < c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_condition_l2359_235978


namespace NUMINAMATH_GPT_largest_fraction_consecutive_primes_l2359_235909

theorem largest_fraction_consecutive_primes (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h0 : 0 < p) (h1 : p < q) (h2 : q < r) (h3 : r < s)
  (hconsec : p + 2 = q ∧ q + 2 = r ∧ r + 2 = s) :
  (r + s) / (p + q) > max ((p + q) / (r + s)) (max ((p + s) / (q + r)) (max ((q + r) / (p + s)) ((q + s) / (p + r)))) :=
sorry

end NUMINAMATH_GPT_largest_fraction_consecutive_primes_l2359_235909


namespace NUMINAMATH_GPT_scientific_notation_l2359_235905

def given_number : ℝ := 632000

theorem scientific_notation : given_number = 6.32 * 10^5 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_l2359_235905


namespace NUMINAMATH_GPT_average_contribution_increase_l2359_235940

theorem average_contribution_increase
  (average_old : ℝ)
  (num_people_old : ℕ)
  (john_donation : ℝ)
  (increase_percentage : ℝ) :
  average_old = 75 →
  num_people_old = 3 →
  john_donation = 150 →
  increase_percentage = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_contribution_increase_l2359_235940


namespace NUMINAMATH_GPT_union_A_B_inter_A_B_comp_int_B_l2359_235914

open Set

variable (x : ℝ)

def A := {x : ℝ | 2 ≤ x ∧ x < 4}
def B := {x : ℝ | 3 ≤ x}

theorem union_A_B : A ∪ B = (Ici 2) :=
by
  sorry

theorem inter_A_B : A ∩ B = Ico 3 4 :=
by
  sorry

theorem comp_int_B : (univ \ A) ∩ B = Ici 4 :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_inter_A_B_comp_int_B_l2359_235914


namespace NUMINAMATH_GPT_prove_remaining_area_is_24_l2359_235954

/-- A rectangular piece of paper with length 12 cm and width 8 cm has four identical isosceles 
right triangles with legs of 6 cm cut from it. Prove that the remaining area is 24 cm². --/
def remaining_area : ℕ := 
  let length := 12
  let width := 8
  let rect_area := length * width
  let triangle_leg := 6
  let triangle_area := (triangle_leg * triangle_leg) / 2
  let total_triangle_area := 4 * triangle_area
  rect_area - total_triangle_area

theorem prove_remaining_area_is_24 : (remaining_area = 24) :=
  by sorry

end NUMINAMATH_GPT_prove_remaining_area_is_24_l2359_235954


namespace NUMINAMATH_GPT_greatest_integer_with_gcd_6_l2359_235919

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end NUMINAMATH_GPT_greatest_integer_with_gcd_6_l2359_235919


namespace NUMINAMATH_GPT_people_per_car_l2359_235966

theorem people_per_car (total_people cars : ℕ) (h1 : total_people = 63) (h2 : cars = 9) :
  total_people / cars = 7 :=
by
  sorry

end NUMINAMATH_GPT_people_per_car_l2359_235966


namespace NUMINAMATH_GPT_height_at_age_10_is_around_146_l2359_235910

noncomputable def predicted_height (x : ℝ) : ℝ :=
  7.2 * x + 74

theorem height_at_age_10_is_around_146 :
  abs (predicted_height 10 - 146) < ε :=
by
  let ε := 10
  sorry

end NUMINAMATH_GPT_height_at_age_10_is_around_146_l2359_235910


namespace NUMINAMATH_GPT_book_distribution_l2359_235955

theorem book_distribution (a b : ℕ) (h1 : a + b = 282) (h2 : (3 / 4) * a = (5 / 9) * b) : a = 120 ∧ b = 162 := by
  sorry

end NUMINAMATH_GPT_book_distribution_l2359_235955


namespace NUMINAMATH_GPT_least_positive_four_digit_multiple_of_6_l2359_235998

theorem least_positive_four_digit_multiple_of_6 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0 → n ≤ m) := 
sorry

end NUMINAMATH_GPT_least_positive_four_digit_multiple_of_6_l2359_235998


namespace NUMINAMATH_GPT_Lizzie_has_27_crayons_l2359_235980

variable (Lizzie Bobbie Billie : ℕ)

axiom Billie_crayons : Billie = 18
axiom Bobbie_crayons : Bobbie = 3 * Billie
axiom Lizzie_crayons : Lizzie = Bobbie / 2

theorem Lizzie_has_27_crayons : Lizzie = 27 :=
by
  sorry

end NUMINAMATH_GPT_Lizzie_has_27_crayons_l2359_235980


namespace NUMINAMATH_GPT_base8_to_base10_362_eq_242_l2359_235902

theorem base8_to_base10_362_eq_242 : 
  let digits := [3, 6, 2]
  let base := 8
  let base10_value := (digits[2] * base^0) + (digits[1] * base^1) + (digits[0] * base^2) 
  base10_value = 242 :=
by
  sorry

end NUMINAMATH_GPT_base8_to_base10_362_eq_242_l2359_235902


namespace NUMINAMATH_GPT_midpoint_of_points_l2359_235907

theorem midpoint_of_points (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 10) (h3 : x2 = 8) (h4 : y2 = 4) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 7) := 
by
  rw [h1, h2, h3, h4]
  norm_num

end NUMINAMATH_GPT_midpoint_of_points_l2359_235907


namespace NUMINAMATH_GPT_min_value_of_box_l2359_235931

theorem min_value_of_box (a b : ℤ) (h_ab : a * b = 30) : 
  ∃ (m : ℤ), m = 61 ∧ (∀ (c : ℤ), a * b = 30 → a^2 + b^2 = c → c ≥ m) := 
sorry

end NUMINAMATH_GPT_min_value_of_box_l2359_235931
