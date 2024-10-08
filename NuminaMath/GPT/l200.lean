import Mathlib

namespace option_d_correct_l200_200609

theorem option_d_correct (x : ℝ) : (-3 * x + 2) * (-3 * x - 2) = 9 * x^2 - 4 := 
  sorry

end option_d_correct_l200_200609


namespace radius_of_circle_from_chord_and_line_l200_200016

theorem radius_of_circle_from_chord_and_line (r : ℝ) (t θ : ℝ) 
    (param_line : ℝ × ℝ) (param_circle : ℝ × ℝ)
    (chord_length : ℝ) 
    (h1 : param_line = (3 + 3 * t, 1 - 4 * t))
    (h2 : param_circle = (r * Real.cos θ, r * Real.sin θ))
    (h3 : chord_length = 4) 
    : r = Real.sqrt 13 :=
sorry

end radius_of_circle_from_chord_and_line_l200_200016


namespace circle_trajectory_l200_200526

theorem circle_trajectory (x y : ℝ) (h1 : (x-5)^2 + (y+7)^2 = 16) (h2 : ∃ c : ℝ, c = ((x + 1 - 5)^2 + (y + 1 + 7)^2)): 
    ((x-5)^2+(y+7)^2 = 25 ∨ (x-5)^2+(y+7)^2 = 9) :=
by
  -- Proof is omitted
  sorry

end circle_trajectory_l200_200526


namespace cheetahs_pandas_ratio_l200_200035

-- Let C denote the number of cheetahs 5 years ago.
-- Let P denote the number of pandas 5 years ago.
-- The conditions given are:
-- 1. The ratio of cheetahs to pandas 5 years ago was the same as it is now.
-- 2. The number of cheetahs has increased by 2.
-- 3. The number of pandas has increased by 6.
-- We need to prove that the current ratio of cheetahs to pandas is C / P.

theorem cheetahs_pandas_ratio
  (C P : ℕ)
  (h1 : C / P = (C + 2) / (P + 6)) :
  (C + 2) / (P + 6) = C / P :=
by sorry

end cheetahs_pandas_ratio_l200_200035


namespace solve_for_x_l200_200783

theorem solve_for_x : ∃ x k l : ℕ, (3 * 22 = k) ∧ (66 + l = 90) ∧ (160 * 3 / 4 = x - l) → x = 144 :=
by
  sorry

end solve_for_x_l200_200783


namespace rosie_circles_track_24_l200_200311

-- Definition of the problem conditions
def lou_distance := 3 -- Lou's total distance in miles
def track_length := 1 / 4 -- Length of the circular track in miles
def rosie_speed_factor := 2 -- Rosie runs at twice the speed of Lou

-- Define the number of times Rosie circles the track as a result
def rosie_circles_the_track : Nat :=
  let lou_circles := lou_distance / track_length
  let rosie_distance := lou_distance * rosie_speed_factor
  let rosie_circles := rosie_distance / track_length
  rosie_circles

-- The theorem stating that Rosie circles the track 24 times
theorem rosie_circles_track_24 : rosie_circles_the_track = 24 := by
  sorry

end rosie_circles_track_24_l200_200311


namespace base_of_first_term_is_two_l200_200102

-- Define h as a positive integer
variable (h : ℕ) (a b c : ℕ)

-- Conditions
variables 
  (h_positive : h > 0)
  (divisor_225 : 225 ∣ h)
  (divisor_216 : 216 ∣ h)

-- Given h can be expressed as specified and a + b + c = 8
variable (h_expression : ∃ k : ℕ, h = k^a * 3^b * 5^c)
variable (sum_eight : a + b + c = 8)

-- Prove the base of the first term in the expression for h is 2.
theorem base_of_first_term_is_two : (∃ k : ℕ, k^a * 3^b * 5^c = h) → k = 2 :=
by 
  sorry

end base_of_first_term_is_two_l200_200102


namespace number_of_relatively_prime_to_18_l200_200602

theorem number_of_relatively_prime_to_18 : 
  ∃ N : ℕ, N = 30 ∧ ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → Nat.gcd n 18 = 1 ↔ false :=
by
  sorry

end number_of_relatively_prime_to_18_l200_200602


namespace no_such_sequence_exists_l200_200153

theorem no_such_sequence_exists (a : ℕ → ℝ) :
  (∀ i, 1 ≤ i ∧ i ≤ 13 → a i + a (i + 1) + a (i + 2) > 0) →
  (∀ i, 1 ≤ i ∧ i ≤ 12 → a i + a (i + 1) + a (i + 2) + a (i + 3) < 0) →
  False :=
by
  sorry

end no_such_sequence_exists_l200_200153


namespace proportional_value_l200_200567

theorem proportional_value :
  ∃ (x : ℝ), 18 / 60 / (12 / 60) = x / 6 ∧ x = 9 := sorry

end proportional_value_l200_200567


namespace third_derivative_correct_l200_200976

noncomputable def func (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_correct :
  (deriv^[3] func) x = (4 / (1 + x^2)^2) :=
sorry

end third_derivative_correct_l200_200976


namespace handshakes_at_networking_event_l200_200965

noncomputable def total_handshakes (n : ℕ) (exclude : ℕ) : ℕ :=
  (n * (n - 1 - exclude)) / 2

theorem handshakes_at_networking_event : total_handshakes 12 1 = 60 := by
  sorry

end handshakes_at_networking_event_l200_200965


namespace solve_inequality_l200_200453

open Set Real

noncomputable def inequality_solution_set : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 4) * (x - 6) ^ 2 ≤ 0 ↔ x ∈ inequality_solution_set := 
sorry

end solve_inequality_l200_200453


namespace find_P_at_1_l200_200834

noncomputable def P (x : ℝ) : ℝ := x ^ 2 + x + 1008

theorem find_P_at_1 :
  (∀ x : ℝ, P (P x) - (P x) ^ 2 = x ^ 2 + x + 2016) →
  P 1 = 1010 := by
  intros H
  sorry

end find_P_at_1_l200_200834


namespace remaining_flour_needed_l200_200042

-- Define the required total amount of flour
def total_flour : ℕ := 8

-- Define the amount of flour already added
def flour_added : ℕ := 2

-- Define the remaining amount of flour needed
def remaining_flour : ℕ := total_flour - flour_added

-- The theorem we need to prove
theorem remaining_flour_needed : remaining_flour = 6 := by
  sorry

end remaining_flour_needed_l200_200042


namespace Lin_finishes_reading_on_Monday_l200_200449

theorem Lin_finishes_reading_on_Monday :
  let start_day := "Tuesday"
  let book_days : ℕ → ℕ := fun n => n
  let total_books := 10
  let total_days := (total_books * (total_books + 1)) / 2
  let days_in_a_week := 7
  let finish_day_offset := total_days % days_in_a_week
  let day_names := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  (day_names.indexOf start_day + finish_day_offset) % days_in_a_week = day_names.indexOf "Monday" :=
by
  sorry

end Lin_finishes_reading_on_Monday_l200_200449


namespace simplify_sqrt7_pow6_l200_200242

theorem simplify_sqrt7_pow6 : (Real.sqrt 7)^6 = (343 : Real) :=
by 
  -- we'll fill in the proof later
  sorry

end simplify_sqrt7_pow6_l200_200242


namespace tan_angle_sum_l200_200902

variable (α β : ℝ)

theorem tan_angle_sum (h1 : Real.tan (α - Real.pi / 6) = 3 / 7)
                      (h2 : Real.tan (Real.pi / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
by
  sorry

end tan_angle_sum_l200_200902


namespace total_pints_l200_200571

variables (Annie Kathryn Ben Sam : ℕ)

-- Conditions
def condition1 := Annie = 16
def condition2 (Annie : ℕ) := Kathryn = 2 * Annie + 2
def condition3 (Kathryn : ℕ) := Ben = Kathryn / 2 - 3
def condition4 (Ben Kathryn : ℕ) := Sam = 2 * (Ben + Kathryn) / 3

-- Statement to prove
theorem total_pints (Annie Kathryn Ben Sam : ℕ) 
  (h1 : condition1 Annie) 
  (h2 : condition2 Annie Kathryn) 
  (h3 : condition3 Kathryn Ben) 
  (h4 : condition4 Ben Kathryn Sam) : 
  Annie + Kathryn + Ben + Sam = 96 :=
sorry

end total_pints_l200_200571


namespace triangle_cut_20_sided_polygon_l200_200878

-- Definitions based on the conditions
def is_triangle (T : Type) : Prop := ∃ (a b c : ℝ), a + b + c = 180 

def can_form_20_sided_polygon (pieces : List (ℝ × ℝ)) : Prop := pieces.length = 20

-- Theorem statement
theorem triangle_cut_20_sided_polygon (T : Type) (P1 P2 : (ℝ × ℝ)) :
  is_triangle T → 
  (P1 ≠ P2) → 
  can_form_20_sided_polygon [P1, P2] :=
sorry

end triangle_cut_20_sided_polygon_l200_200878


namespace equal_expressions_l200_200966

theorem equal_expressions : (-2)^3 = -(2^3) :=
by sorry

end equal_expressions_l200_200966


namespace line_intersects_hyperbola_l200_200959

theorem line_intersects_hyperbola 
  (k : ℝ)
  (hyp : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) :
  -Real.sqrt 15 / 3 < k ∧ k < -1 := 
sorry


end line_intersects_hyperbola_l200_200959


namespace parallelogram_side_length_l200_200906

theorem parallelogram_side_length (s : ℝ) (h : 3 * s * s * (1 / 2) = 27 * Real.sqrt 3) : 
  s = 3 * Real.sqrt (2 * Real.sqrt 3) :=
sorry

end parallelogram_side_length_l200_200906


namespace constant_function_odd_iff_zero_l200_200419

theorem constant_function_odd_iff_zero (k : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = k) 
  (h2 : ∀ x, f (-x) = -f x) : 
  k = 0 :=
sorry

end constant_function_odd_iff_zero_l200_200419


namespace find_a10_l200_200583

-- Define the arithmetic sequence with its common difference and initial term
axiom a_seq : ℕ → ℝ
axiom a1 : ℝ
axiom d : ℝ

-- Conditions
axiom a3 : a_seq 3 = a1 + 2 * d
axiom a5_a8 : a_seq 5 + a_seq 8 = 15

-- Theorem statement
theorem find_a10 : a_seq 10 = 13 :=
by sorry

end find_a10_l200_200583


namespace cards_given_l200_200223

def initial_cards : ℕ := 304
def remaining_cards : ℕ := 276
def given_cards : ℕ := initial_cards - remaining_cards

theorem cards_given :
  given_cards = 28 :=
by
  unfold given_cards
  unfold initial_cards
  unfold remaining_cards
  sorry

end cards_given_l200_200223


namespace Jaylen_total_vegetables_l200_200859

def Jaylen_vegetables (J_bell_peppers J_green_beans J_carrots J_cucumbers : Nat) : Nat :=
  J_bell_peppers + J_green_beans + J_carrots + J_cucumbers

theorem Jaylen_total_vegetables :
  let Kristin_bell_peppers := 2
  let Kristin_green_beans := 20
  let Jaylen_bell_peppers := 2 * Kristin_bell_peppers
  let Jaylen_green_beans := (Kristin_green_beans / 2) - 3
  let Jaylen_carrots := 5
  let Jaylen_cucumbers := 2
  Jaylen_vegetables Jaylen_bell_peppers Jaylen_green_beans Jaylen_carrots Jaylen_cucumbers = 18 := 
by
  sorry

end Jaylen_total_vegetables_l200_200859


namespace valueOf_seq_l200_200254

variable (a : ℕ → ℝ)
variable (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
variable (h_positive : ∀ n : ℕ, a n > 0)
variable (h_arith_subseq : 2 * a 5 = a 3 + a 6)

theorem valueOf_seq (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arith_subseq : 2 * a 5 = a 3 + a 6) :
  (∃ q : ℝ, q = 1 ∨ q = (1 + Real.sqrt 5) / 2 ∧ (a 3 + a 5) / (a 4 + a 6) = 1 / q) → 
  (∃ q : ℝ, (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2) :=
by
  sorry

end valueOf_seq_l200_200254


namespace line_transformation_equiv_l200_200961

theorem line_transformation_equiv :
  (∀ x y: ℝ, (2 * x - y - 3 = 0) ↔
    (7 * (x + 2 * y) - 5 * (-x + 4 * y) - 18 = 0)) :=
sorry

end line_transformation_equiv_l200_200961


namespace evaluate_expression_l200_200257

theorem evaluate_expression :
  - (20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 :=
by
  sorry

end evaluate_expression_l200_200257


namespace moving_circle_trajectory_is_ellipse_l200_200775

noncomputable def trajectory_of_center (x y : ℝ) : Prop :=
  let ellipse_eq := x^2 / 4 + y^2 / 3 = 1 
  ellipse_eq ∧ x ≠ -2

theorem moving_circle_trajectory_is_ellipse
  (M_1 M_2 center : ℝ × ℝ)
  (r1 r2 R : ℝ)
  (h1 : M_1 = (-1, 0))
  (h2 : M_2 = (1, 0))
  (h3 : r1 = 1)
  (h4 : r2 = 3)
  (h5 : (center.1 + 1)^2 + center.2^2 = (1 + R)^2)
  (h6 : (center.1 - 1)^2 + center.2^2 = (3 - R)^2) :
  trajectory_of_center center.1 center.2 :=
by sorry

end moving_circle_trajectory_is_ellipse_l200_200775


namespace probability_drawing_balls_l200_200979

theorem probability_drawing_balls :
  let total_balls := 15
  let red_balls := 10
  let blue_balls := 5
  let drawn_balls := 4
  let num_ways_to_draw_4_balls := Nat.choose total_balls drawn_balls
  let num_ways_to_draw_3_red_1_blue := (Nat.choose red_balls 3) * (Nat.choose blue_balls 1)
  let num_ways_to_draw_1_red_3_blue := (Nat.choose red_balls 1) * (Nat.choose blue_balls 3)
  let total_favorable_outcomes := num_ways_to_draw_3_red_1_blue + num_ways_to_draw_1_red_3_blue
  let probability := total_favorable_outcomes / num_ways_to_draw_4_balls
  probability = (140 : ℚ) / 273 :=
sorry

end probability_drawing_balls_l200_200979


namespace trig_identity_l200_200184

theorem trig_identity :
  (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_l200_200184


namespace solution_set_interval_l200_200879

theorem solution_set_interval (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} :=
sorry

end solution_set_interval_l200_200879


namespace Integers_and_fractions_are_rational_numbers_l200_200054

-- Definitions from conditions
def is_fraction (x : ℚ) : Prop :=
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

def is_integer (x : ℤ) : Prop := 
  ∃n : ℤ, x = n

def is_rational (x : ℚ) : Prop := 
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

-- The statement to be proven
theorem Integers_and_fractions_are_rational_numbers (x : ℚ) : 
  (∃n : ℤ, x = (n : ℚ)) ∨ is_fraction x ↔ is_rational x :=
by sorry

end Integers_and_fractions_are_rational_numbers_l200_200054


namespace rate_percent_calculation_l200_200563

theorem rate_percent_calculation (SI P T : ℝ) (R : ℝ) : SI = 640 ∧ P = 4000 ∧ T = 2 → SI = P * R * T / 100 → R = 8 :=
by
  intros
  sorry

end rate_percent_calculation_l200_200563


namespace equivalent_annual_rate_correct_l200_200982

noncomputable def quarterly_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

noncomputable def effective_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate / 100)^4

noncomputable def equivalent_annual_rate (annual_rate : ℝ) : ℝ :=
  (effective_annual_rate (quarterly_rate annual_rate) - 1) * 100

theorem equivalent_annual_rate_correct :
  equivalent_annual_rate 8 = 8.24 := 
by
  sorry

end equivalent_annual_rate_correct_l200_200982


namespace vector_condition_l200_200415

def vec_a : ℝ × ℝ := (5, 2)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (-23, -12)

theorem vector_condition : 3 • (vec_a.1, vec_a.2) - 2 • (vec_b.1, vec_b.2) + vec_c = (0, 0) :=
by
  sorry

end vector_condition_l200_200415


namespace volume_of_rectangular_prism_l200_200121

theorem volume_of_rectangular_prism (x y z : ℝ) 
  (h1 : x * y = 30) 
  (h2 : x * z = 45) 
  (h3 : y * z = 75) : 
  x * y * z = 150 :=
sorry

end volume_of_rectangular_prism_l200_200121


namespace train_speed_correct_l200_200509

def train_length : ℝ := 100
def crossing_time : ℝ := 12
def expected_speed : ℝ := 8.33

theorem train_speed_correct : (train_length / crossing_time) = expected_speed :=
by
  -- Proof goes here
  sorry

end train_speed_correct_l200_200509


namespace geom_cos_sequence_l200_200535

open Real

theorem geom_cos_sequence (b : ℝ) (hb : 0 < b ∧ b < 360) (h : cos (2*b) / cos b = cos (3*b) / cos (2*b)) : b = 180 :=
by
  sorry

end geom_cos_sequence_l200_200535


namespace packs_of_snacks_l200_200823

theorem packs_of_snacks (kyle_bike_hours : ℝ) (pack_cost : ℝ) (ryan_budget : ℝ) :
  kyle_bike_hours = 2 →
  10 * (2 * kyle_bike_hours) = pack_cost →
  ryan_budget = 2000 →
  ryan_budget / pack_cost = 50 :=
by 
  sorry

end packs_of_snacks_l200_200823


namespace glasses_total_l200_200013

theorem glasses_total :
  ∃ (S L e : ℕ), 
    (L = S + 16) ∧ 
    (12 * S + 16 * L) / (S + L) = 15 ∧ 
    (e = 12 * S + 16 * L) ∧ 
    e = 480 :=
by
  sorry

end glasses_total_l200_200013


namespace inequality_solution_set_minimum_value_mn_squared_l200_200413

noncomputable def f (x : ℝ) := |x - 2| + |x + 1|

theorem inequality_solution_set : 
  (∀ x, f x > 7 ↔ x > 4 ∨ x < -3) :=
by sorry

theorem minimum_value_mn_squared (m n : ℝ) (hm : n > 0) (hmin : ∀ x, f x ≥ m + n) :
  m^2 + n^2 = 9 / 2 ∧ m = 3 / 2 ∧ n = 3 / 2 :=
by sorry

end inequality_solution_set_minimum_value_mn_squared_l200_200413


namespace two_lines_perpendicular_to_same_plane_are_parallel_l200_200750

variables {Plane Line : Type} 
variables (perp : Line → Plane → Prop) (parallel : Line → Line → Prop)

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane) (ha : perp a α) (hb : perp b α) : parallel a b :=
sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l200_200750


namespace odd_expression_l200_200942

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_expression (k m : ℤ) (o := 2 * k + 3) (n := 2 * m) :
  is_odd (o^2 + n * o) :=
by sorry

end odd_expression_l200_200942


namespace arithmetic_sequence_ratio_l200_200732

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 d : ℝ)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) (h2 : ∀ n, S n = n * (2 * a1 + (n - 1) * d) / 2)
  (h_nonzero: ∀ n, a n ≠ 0):
  (S 5) / (a 3) = 5 :=
by
  sorry

end arithmetic_sequence_ratio_l200_200732


namespace total_property_price_l200_200135

theorem total_property_price :
  let price_per_sqft : ℝ := 98
  let house_sqft : ℝ := 2400
  let barn_sqft : ℝ := 1000
  let house_price : ℝ := house_sqft * price_per_sqft
  let barn_price : ℝ := barn_sqft * price_per_sqft
  let total_price : ℝ := house_price + barn_price
  total_price = 333200 := by
  sorry

end total_property_price_l200_200135


namespace xy_nonzero_implies_iff_l200_200041

variable {x y : ℝ}

theorem xy_nonzero_implies_iff (h : x * y ≠ 0) : (x + y = 0) ↔ (x / y + y / x = -2) :=
sorry

end xy_nonzero_implies_iff_l200_200041


namespace distance_AB_l200_200731

-- Definitions and conditions taken from part a)
variables (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0)

-- The main theorem statement
theorem distance_AB (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0) : 
  ∃ s : ℝ, s = Real.sqrt ((a * b * c) / (a + c - b)) := 
sorry

end distance_AB_l200_200731


namespace part1_solution_part2_solution_l200_200717

section Part1

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

theorem part1_solution (x : ℝ) : f x > 0 ↔ x < -3 ∨ x > 2 :=
sorry

end Part1

section Part2

variables (a : ℝ) (ha : a < 0)
noncomputable def g (x : ℝ) : ℝ := a*x^2 + (3 - 2*a)*x - 6

theorem part2_solution (x : ℝ) :
  if h1 : a < -3/2 then g x < 0 ↔ x < -3/a ∨ x > 2
  else if h2 : a = -3/2 then g x < 0 ↔ x ≠ 2
  else -3/2 < a ∧ a < 0 → g x < 0 ↔ x < 2 ∨ x > -3/a :=
sorry

end Part2

end part1_solution_part2_solution_l200_200717


namespace range_of_a_l200_200608

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) ∧ (∃ x : ℝ, x^2 - 4 * x + a ≤ 0) →
  a ∈ Set.Icc (Real.exp 1) 4 :=
by
  sorry

end range_of_a_l200_200608


namespace children_count_l200_200499

noncomputable def king_age := 35
noncomputable def queen_age := 35
noncomputable def num_sons := 3
noncomputable def initial_children_age := 35
noncomputable def total_combined_age := 70
noncomputable def max_children := 20

theorem children_count :
  ∃ d n, (king_age + queen_age + 2 * n = initial_children_age + (d + num_sons) * n) ∧ 
         (king_age + queen_age = total_combined_age) ∧
         (initial_children_age = 35) ∧
         (d + num_sons ≤ max_children) ∧
         (d + num_sons = 7 ∨ d + num_sons = 9)
:= sorry

end children_count_l200_200499


namespace matrix_identity_l200_200901

noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![-2, 1]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  N * N = 4 • N + -11 • I :=
by
  sorry

end matrix_identity_l200_200901


namespace largest_root_vieta_l200_200676

theorem largest_root_vieta 
  (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = -6) : 
  max a (max b c) = 3 :=
sorry

end largest_root_vieta_l200_200676


namespace find_triples_of_positive_integers_l200_200695

theorem find_triples_of_positive_integers (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn_pos : 0 < n) 
  (equation : p * (p + 3) + q * (q + 3) = n * (n + 3)) : 
  (p = 3 ∧ q = 2 ∧ n = 4) :=
sorry

end find_triples_of_positive_integers_l200_200695


namespace arithmetic_series_remainder_l200_200357

-- Define the sequence parameters
def a : ℕ := 2
def l : ℕ := 12
def d : ℕ := 1
def n : ℕ := (l - a) / d + 1

-- Define the sum of the arithmetic series
def S : ℕ := n * (a + l) / 2

-- The final theorem statement
theorem arithmetic_series_remainder : S % 9 = 5 := 
by sorry

end arithmetic_series_remainder_l200_200357


namespace obtuse_triangle_side_range_l200_200663

theorem obtuse_triangle_side_range (a : ℝ) :
  (a > 0) ∧
  ((a < 3 ∧ a > -1) ∧ 
  (2 * a + 1 > a + 2) ∧ 
  (a > 1)) → 1 < a ∧ a < 3 := 
by
  sorry

end obtuse_triangle_side_range_l200_200663


namespace flat_odot_length_correct_l200_200852

noncomputable def sides : ℤ × ℤ × ℤ := (4, 5, 6)

noncomputable def semiperimeter (a b c : ℤ) : ℚ :=
  (a + b + c) / 2

noncomputable def length_flat_odot (a b c : ℤ) : ℚ :=
  (semiperimeter a b c) - b

theorem flat_odot_length_correct : length_flat_odot 4 5 6 = 2.5 := by
  sorry

end flat_odot_length_correct_l200_200852


namespace find_a_l200_200481

def F (a b c : ℤ) : ℤ := a * b^2 + c

theorem find_a (a : ℤ) (h : F a 3 (-1) = F a 5 (-3)) : a = 1 / 8 := by
  sorry

end find_a_l200_200481


namespace problem_solution_l200_200631

noncomputable def positiveIntPairsCount : ℕ :=
  sorry

theorem problem_solution :
  positiveIntPairsCount = 2 :=
sorry

end problem_solution_l200_200631


namespace number_of_cells_after_9_days_l200_200369

theorem number_of_cells_after_9_days : 
  let initial_cells := 4 
  let doubling_period := 3 
  let total_duration := 9 
  ∀ cells_after_9_days, cells_after_9_days = initial_cells * 2^(total_duration / doubling_period) 
  → cells_after_9_days = 32 :=
by
  sorry

end number_of_cells_after_9_days_l200_200369


namespace sum_invested_eq_2000_l200_200525

theorem sum_invested_eq_2000 (P : ℝ) (R1 R2 T : ℝ) (H1 : R1 = 18) (H2 : R2 = 12) 
  (H3 : T = 2) (H4 : (P * R1 * T / 100) - (P * R2 * T / 100) = 240): 
  P = 2000 :=
by 
  sorry

end sum_invested_eq_2000_l200_200525


namespace find_g_l200_200347

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 4

theorem find_g :
  (∀ x : ℝ, g x = x + 5) ∨ (∀ x : ℝ, g x = -x - 3) := 
by
  sorry

end find_g_l200_200347


namespace linear_regression_eq_l200_200404

noncomputable def x_vals : List ℝ := [3, 7, 11]
noncomputable def y_vals : List ℝ := [10, 20, 24]

theorem linear_regression_eq :
  ∃ a b : ℝ, (a = 5.75) ∧ (b = 1.75) ∧ (∀ x, ∃ y, y = a + b * x) := sorry

end linear_regression_eq_l200_200404


namespace find_a_b_l200_200188

theorem find_a_b (a b : ℝ) (h1 : b - a = -7) (h2 : 64 * (a + b) = 20736) :
  a = 165.5 ∧ b = 158.5 :=
by
  sorry

end find_a_b_l200_200188


namespace num_dimes_l200_200397

/--
Given eleven coins consisting of pennies, nickels, dimes, quarters, and half-dollars,
having a total value of $1.43, with at least one coin of each type,
prove that there must be exactly 4 dimes.
-/
theorem num_dimes (p n d q h : ℕ) :
  1 ≤ p ∧ 1 ≤ n ∧ 1 ≤ d ∧ 1 ≤ q ∧ 1 ≤ h ∧ 
  p + n + d + q + h = 11 ∧ 
  (1 * p + 5 * n + 10 * d + 25 * q + 50 * h) = 143
  → d = 4 :=
by
  sorry

end num_dimes_l200_200397


namespace find_a_l200_200027

theorem find_a
  (a : ℝ)
  (h1 : ∃ P Q : ℝ × ℝ, (P.1 ^ 2 + P.2 ^ 2 - 2 * P.1 + 4 * P.2 + 1 = 0) ∧ (Q.1 ^ 2 + Q.2 ^ 2 - 2 * Q.1 + 4 * Q.2 + 1 = 0) ∧
                         (a * P.1 + 2 * P.2 + 6 = 0) ∧ (a * Q.1 + 2 * Q.2 + 6 = 0) ∧
                         ((P.1 - 1) * (Q.1 - 1) + (P.2 + 2) * (Q.2 + 2) = 0)) :
  a = 2 :=
by
  sorry

end find_a_l200_200027


namespace inequality_proof_l200_200587

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31 :=
sorry

end inequality_proof_l200_200587


namespace find_asymptotes_l200_200923

def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

def shifted_hyperbola_asymptotes (x y : ℝ) : Prop :=
  y = 4 / 3 * x + 5 ∨ y = -4 / 3 * x + 5

theorem find_asymptotes (x y : ℝ) :
  (∃ y', y = y' + 5 ∧ hyperbola_eq x y')
  ↔ shifted_hyperbola_asymptotes x y :=
by
  sorry

end find_asymptotes_l200_200923


namespace contrapositive_property_l200_200520

def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def is_divisible_by_2 (n : ℤ) : Prop := n % 2 = 0

theorem contrapositive_property :
  (∀ n : ℤ, is_divisible_by_6 n → is_divisible_by_2 n) ↔ (∀ n : ℤ, ¬ is_divisible_by_2 n → ¬ is_divisible_by_6 n) :=
by
  sorry

end contrapositive_property_l200_200520


namespace car_mpg_city_l200_200936

theorem car_mpg_city (h c t : ℕ) (H1 : 560 = h * t) (H2 : 336 = c * t) (H3 : c = h - 6) : c = 9 :=
by
  sorry

end car_mpg_city_l200_200936


namespace point_coordinates_in_second_quadrant_l200_200425

theorem point_coordinates_in_second_quadrant
    (P : ℝ × ℝ)
    (h1 : P.1 < 0)
    (h2 : P.2 > 0)
    (h3 : |P.2| = 4)
    (h4 : |P.1| = 5) :
    P = (-5, 4) :=
sorry

end point_coordinates_in_second_quadrant_l200_200425


namespace correct_option_C_l200_200972

variable (a : ℝ)

theorem correct_option_C : (a^2 * a = a^3) :=
by sorry

end correct_option_C_l200_200972


namespace oranges_to_put_back_l200_200067

theorem oranges_to_put_back
  (price_apple price_orange : ℕ)
  (A_all O_all : ℕ)
  (mean_initial_fruit mean_final_fruit : ℕ)
  (A O x : ℕ)
  (h_price_apple : price_apple = 40)
  (h_price_orange : price_orange = 60)
  (h_total_fruit : A_all + O_all = 10)
  (h_mean_initial : mean_initial_fruit = 54)
  (h_mean_final : mean_final_fruit = 50)
  (h_total_cost_initial : price_apple * A_all + price_orange * O_all = mean_initial_fruit * (A_all + O_all))
  (h_total_cost_final : price_apple * A + price_orange * (O - x) = mean_final_fruit * (A + (O - x)))
  : x = 4 := 
  sorry

end oranges_to_put_back_l200_200067


namespace probability_of_pink_l200_200690

-- Given conditions
variables (B P : ℕ) (h : (B : ℚ) / (B + P) = 3 / 7)

-- To prove
theorem probability_of_pink (h_pow : (B : ℚ) ^ 2 / (B + P) ^ 2 = 9 / 49) :
  (P : ℚ) / (B + P) = 4 / 7 :=
sorry

end probability_of_pink_l200_200690


namespace solve_for_x_l200_200754

theorem solve_for_x (x : ℚ) (h : (3 - x)/(x + 2) + (3 * x - 6)/(3 - x) = 2) : x = -7/6 := 
by 
  sorry

end solve_for_x_l200_200754


namespace exponentiation_division_l200_200243

variable (a b : ℝ)

theorem exponentiation_division (a b : ℝ) : ((2 * a) / b) ^ 4 = (16 * a ^ 4) / (b ^ 4) := by
  sorry

end exponentiation_division_l200_200243


namespace alphametic_puzzle_l200_200159

theorem alphametic_puzzle (I D A M E R O : ℕ) 
  (h1 : R = 0) 
  (h2 : D + E = 10)
  (h3 : I + M + 1 = O)
  (h4 : A = D + 1) :
  I + 1 + M + 10 + 1 = O + 0 + A := sorry

end alphametic_puzzle_l200_200159


namespace conic_sections_l200_200496

theorem conic_sections (x y : ℝ) (h : y^4 - 6 * x^4 = 3 * y^2 - 2) :
  (∃ a b : ℝ, y^2 = a + b * x^2) ∨ (∃ c d : ℝ, y^2 = c - d * x^2) :=
sorry

end conic_sections_l200_200496


namespace joes_total_weight_l200_200050

theorem joes_total_weight (F S : ℕ) (h1 : F = 700) (h2 : 2 * F = S + 300) :
  F + S = 1800 :=
by
  sorry

end joes_total_weight_l200_200050


namespace previous_year_height_l200_200236

noncomputable def previous_height (H_current : ℝ) (g : ℝ) : ℝ :=
  H_current / (1 + g)

theorem previous_year_height :
  previous_height 147 0.05 = 140 :=
by
  unfold previous_height
  -- Proof steps would go here
  sorry

end previous_year_height_l200_200236


namespace remaining_adults_fed_l200_200886

theorem remaining_adults_fed 
  (cans : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (remaining_cans : ℕ)
  (remaining_adults : ℕ) :
  (adults_per_can = 4) →
  (children_per_can = 6) →
  (initial_cans = 7) →
  (children_fed = 18) →
  (remaining_cans = initial_cans - children_fed / children_per_can) →
  (remaining_adults = remaining_cans * adults_per_can) →
  remaining_adults = 16 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end remaining_adults_fed_l200_200886


namespace arcsin_one_half_l200_200354

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  -- Conditions
  have h1 : -Real.pi / 2 ≤ Real.pi / 6 ∧ Real.pi / 6 ≤ Real.pi / 2 := by
    -- Proof the range of pi/6 is within [-pi/2, pi/2]
    sorry
  have h2 : ∀ x, Real.sin x = 1 / 2 → x = Real.pi / 6 := by
    -- Proof sin(pi/6) = 1 / 2
    sorry
  show Real.arcsin (1 / 2) = Real.pi / 6
  -- Proof arcsin(1/2) = pi/6 based on the above conditions
  sorry

end arcsin_one_half_l200_200354


namespace cyclist_speed_ratio_is_4_l200_200291

noncomputable def ratio_of_speeds (v_a v_b v_c : ℝ) : ℝ :=
  if v_a ≤ v_b ∧ v_b ≤ v_c then v_c / v_a else 0

theorem cyclist_speed_ratio_is_4
  (v_a v_b v_c : ℝ)
  (h1 : v_a + v_b = d / 5)
  (h2 : v_b + v_c = 15)
  (h3 : 15 = (45 - d) / 3)
  (d : ℝ) : 
  ratio_of_speeds v_a v_b v_c = 4 :=
by
  sorry

end cyclist_speed_ratio_is_4_l200_200291


namespace Cole_drive_time_to_work_l200_200903

theorem Cole_drive_time_to_work :
  ∀ (D T_work T_home : ℝ),
    (T_work = D / 80) →
    (T_home = D / 120) →
    (T_work + T_home = 3) →
    (T_work * 60 = 108) :=
by
  intros D T_work T_home h1 h2 h3
  sorry

end Cole_drive_time_to_work_l200_200903


namespace concentration_of_salt_solution_l200_200376

-- Conditions:
def total_volume : ℝ := 1 + 0.25
def concentration_of_mixture : ℝ := 0.15
def volume_of_salt_solution : ℝ := 0.25

-- Expression for the concentration of the salt solution used, $C$:
theorem concentration_of_salt_solution (C : ℝ) :
  (volume_of_salt_solution * (C / 100)) = (total_volume * concentration_of_mixture) → C = 75 := by
  sorry

end concentration_of_salt_solution_l200_200376


namespace simplify_expression_evaluate_l200_200426

theorem simplify_expression_evaluate : 
  let x := 1
  let y := 2
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 :=
by
  sorry

end simplify_expression_evaluate_l200_200426


namespace infinite_product_eq_four_four_thirds_l200_200888

theorem infinite_product_eq_four_four_thirds :
  ∏' n : ℕ, (4^(n+1)^(1/(2^(n+1)))) = 4^(4/3) :=
sorry

end infinite_product_eq_four_four_thirds_l200_200888


namespace sum_of_digits_of_d_l200_200045

noncomputable section

def exchange_rate : ℚ := 8/5
def euros_after_spending (d : ℚ) : ℚ := exchange_rate * d - 80

theorem sum_of_digits_of_d {d : ℚ} (h : euros_after_spending d = d) : 
  d = 135 ∧ 1 + 3 + 5 = 9 := 
by 
  sorry

end sum_of_digits_of_d_l200_200045


namespace marks_lost_per_wrong_answer_l200_200220

theorem marks_lost_per_wrong_answer
    (total_questions : ℕ)
    (correct_questions : ℕ)
    (total_marks : ℕ)
    (marks_per_correct : ℕ)
    (marks_lost : ℕ)
    (x : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_questions = 44)
    (h3 : total_marks = 160)
    (h4 : marks_per_correct = 4)
    (h5 : marks_lost = 176 - total_marks)
    (h6 : marks_lost = x * (total_questions - correct_questions)) :
    x = 1 := by
  sorry

end marks_lost_per_wrong_answer_l200_200220


namespace find_numbers_l200_200551

theorem find_numbers (a b : ℝ) (h1 : a - b = 7.02) (h2 : a = 10 * b) : a = 7.8 ∧ b = 0.78 :=
by
  sorry

end find_numbers_l200_200551


namespace charles_remaining_skittles_l200_200591

def c : ℕ := 25
def d : ℕ := 7
def remaining_skittles : ℕ := c - d

theorem charles_remaining_skittles : remaining_skittles = 18 := by
  sorry

end charles_remaining_skittles_l200_200591


namespace mrs_sheridan_fish_count_l200_200028

/-
  Problem statement: 
  Prove that the total number of fish Mrs. Sheridan has now is 69, 
  given that she initially had 22 fish and she received 47 more from her sister.
-/

theorem mrs_sheridan_fish_count :
  let initial_fish : ℕ := 22
  let additional_fish : ℕ := 47
  initial_fish + additional_fish = 69 := by
sorry

end mrs_sheridan_fish_count_l200_200028


namespace unit_digit_product_is_zero_l200_200528

-- Definitions based on conditions in (a)
def a_1 := 6245
def a_2 := 7083
def a_3 := 9137
def a_4 := 4631
def a_5 := 5278
def a_6 := 3974

-- Helper function to get the unit digit of a number
def unit_digit (n : Nat) : Nat := n % 10

-- Main theorem to prove
theorem unit_digit_product_is_zero :
  unit_digit (a_1 * a_2 * a_3 * a_4 * a_5 * a_6) = 0 := by
  sorry

end unit_digit_product_is_zero_l200_200528


namespace suitable_for_sampling_l200_200858

-- Definitions based on conditions
def optionA_requires_comprehensive : Prop := true
def optionB_requires_comprehensive : Prop := true
def optionC_requires_comprehensive : Prop := true
def optionD_allows_sampling : Prop := true

-- Problem in Lean: Prove that option D is suitable for a sampling survey
theorem suitable_for_sampling : optionD_allows_sampling := by
  sorry

end suitable_for_sampling_l200_200858


namespace proof_problem_l200_200043

theorem proof_problem (x : ℤ) (h : (x - 34) / 10 = 2) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l200_200043


namespace A_equals_half_C_equals_half_l200_200359

noncomputable def A := 2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)
noncomputable def C := Real.sin (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - Real.cos (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180)

theorem A_equals_half : A = 1 / 2 := 
by
  sorry

theorem C_equals_half : C = 1 / 2 := 
by
  sorry

end A_equals_half_C_equals_half_l200_200359


namespace pow137_mod8_l200_200459

theorem pow137_mod8 : (5 ^ 137) % 8 = 5 := by
  -- Use the provided conditions
  have h1: 5 % 8 = 5 := by norm_num
  have h2: (5 ^ 2) % 8 = 1 := by norm_num
  sorry

end pow137_mod8_l200_200459


namespace m_and_n_relationship_l200_200234

-- Define the function f
def f (x m : ℝ) := x^2 - 4*x + 4 + m

-- State the conditions and required proof
theorem m_and_n_relationship (m n : ℝ) (h_domain : ∀ x, 2 ≤ x ∧ x ≤ n → 2 ≤ f x m ∧ f x m ≤ n) :
  m^n = 8 :=
by
  -- Placeholder for the actual proof
  sorry

end m_and_n_relationship_l200_200234


namespace buffy_less_brittany_by_40_seconds_l200_200033

/-
The following statement proves that Buffy's breath-holding time was 40 seconds less than Brittany's, 
given the initial conditions about their breath-holding times.
-/
theorem buffy_less_brittany_by_40_seconds 
  (kelly_time : ℕ) 
  (brittany_time : ℕ) 
  (buffy_time : ℕ) 
  (h_kelly : kelly_time = 180) 
  (h_brittany : brittany_time = kelly_time - 20) 
  (h_buffy : buffy_time = 120)
  :
  brittany_time - buffy_time = 40 :=
sorry

end buffy_less_brittany_by_40_seconds_l200_200033


namespace find_m_for_unique_solution_l200_200088

theorem find_m_for_unique_solution :
  ∃ m : ℝ, (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) ∧ 
  ∀ x : ℝ, (mx - 2 ≠ 0 → (x + 3) / (mx - 2) = x + 1 ↔ ∃! x : ℝ, (mx - 2) * (x + 1) = (x + 3)) :=
sorry

end find_m_for_unique_solution_l200_200088


namespace divisible_by_17_l200_200278

theorem divisible_by_17 (a b c d : ℕ) (h1 : a + b + c + d = 2023)
    (h2 : 2023 ∣ (a * b - c * d))
    (h3 : 2023 ∣ (a^2 + b^2 + c^2 + d^2))
    (h4 : ∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 7 ∣ x) :
    (∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 17 ∣ x) := 
sorry

end divisible_by_17_l200_200278


namespace simplify_poly_l200_200585

-- Define the polynomial expressions
def poly1 (r : ℝ) := 2 * r^3 + 4 * r^2 + 5 * r - 3
def poly2 (r : ℝ) := r^3 + 6 * r^2 + 8 * r - 7

-- Simplification goal
theorem simplify_poly (r : ℝ) : (poly1 r) - (poly2 r) = r^3 - 2 * r^2 - 3 * r + 4 :=
by 
  -- We declare the proof is omitted using sorry
  sorry

end simplify_poly_l200_200585


namespace cars_each_remaining_day_l200_200742

theorem cars_each_remaining_day (total_cars : ℕ) (monday_cars : ℕ) (tuesday_cars : ℕ)
  (wednesday_cars : ℕ) (thursday_cars : ℕ) (remaining_days : ℕ)
  (h_total : total_cars = 450)
  (h_mon : monday_cars = 50)
  (h_tue : tuesday_cars = 50)
  (h_wed : wednesday_cars = 2 * monday_cars)
  (h_thu : thursday_cars = 2 * monday_cars)
  (h_remaining : remaining_days = (total_cars - (monday_cars + tuesday_cars + wednesday_cars + thursday_cars)) / 3)
  :
  remaining_days = 50 := sorry

end cars_each_remaining_day_l200_200742


namespace solve_for_x_l200_200295

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) :
  x = 37 :=
sorry

end solve_for_x_l200_200295


namespace find_a_l200_200589

theorem find_a (a t : ℝ) 
    (h1 : (a + t) / 2 = 2020) 
    (h2 : t / 2 = 11) : 
    a = 4018 := 
by 
    sorry

end find_a_l200_200589


namespace slope_of_line_l200_200429

theorem slope_of_line (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 3)) :
  (Q.snd - P.snd) / (Q.fst - P.fst) = 1 / 3 := by
  sorry

end slope_of_line_l200_200429


namespace number_of_arrangements_word_l200_200293

noncomputable def factorial (n : Nat) : Nat := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem number_of_arrangements_word (letters : List Char) (n : Nat) (r1 r2 r3 : Nat) 
  (h1 : letters = ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'])
  (h2 : 2 = r1) (h3 : 2 = r2) (h4 : 2 = r3) :
  n = 11 → 
  factorial n / (factorial r1 * factorial r2 * factorial r3) = 4989600 := 
by
  sorry

end number_of_arrangements_word_l200_200293


namespace amount_after_two_years_l200_200150

def amount_after_years (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * ((r + 1) ^ n) / (r ^ n)

theorem amount_after_two_years :
  let P : ℕ := 70400
  let r : ℕ := 8
  amount_after_years P r 2 = 89070 :=
  by
    sorry

end amount_after_two_years_l200_200150


namespace product_of_possible_x_l200_200770

theorem product_of_possible_x : 
  (∀ x : ℚ, abs ((18 / x) + 4) = 3 → x = -18 ∨ x = -18 / 7) → 
  ((-18) * (-18 / 7) = 324 / 7) :=
by
  sorry

end product_of_possible_x_l200_200770


namespace sales_volume_relation_maximize_profit_l200_200647

-- Define the conditions as given in the problem
def cost_price : ℝ := 6
def sales_data : List (ℝ × ℝ) := [(10, 4000), (11, 3900), (12, 3800)]
def price_range (x : ℝ) : Prop := 6 ≤ x ∧ x ≤ 32

-- Define the functional relationship y in terms of x
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

-- Define the profit function w in terms of x
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - cost_price)

-- Prove that the functional relationship holds within the price range
theorem sales_volume_relation (x : ℝ) (h : price_range x) :
  ∀ (y : ℝ), (x, y) ∈ sales_data → y = sales_volume x := by
  sorry

-- Prove that the profit is maximized when x = 28 and the profit is 48400 yuan
theorem maximize_profit :
  ∃ x, price_range x ∧ x = 28 ∧ profit x = 48400 := by
  sorry

end sales_volume_relation_maximize_profit_l200_200647


namespace penny_initial_money_l200_200868

theorem penny_initial_money
    (pairs_of_socks : ℕ)
    (cost_per_pair : ℝ)
    (number_of_pairs : ℕ)
    (cost_of_hat : ℝ)
    (money_left : ℝ)
    (initial_money : ℝ)
    (H1 : pairs_of_socks = 4)
    (H2 : cost_per_pair = 2)
    (H3 : number_of_pairs = pairs_of_socks)
    (H4 : cost_of_hat = 7)
    (H5 : money_left = 5)
    (H6 : initial_money = (number_of_pairs * cost_per_pair) + cost_of_hat + money_left) : initial_money = 20 :=
sorry

end penny_initial_money_l200_200868


namespace g_domain_l200_200416

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

theorem g_domain : { x : ℝ | -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 } = (Set.Icc (-1) 0 ∪ Set.Icc 0 1) \ {0} :=
by
  sorry

end g_domain_l200_200416


namespace quotient_with_zero_in_middle_l200_200387

theorem quotient_with_zero_in_middle : 
  ∃ (op : ℕ → ℕ → ℕ), 
  (op = Nat.add ∧ ((op 6 4) / 3).digits 10 = [3, 0, 3]) := 
by 
  sorry

end quotient_with_zero_in_middle_l200_200387


namespace euclidean_steps_arbitrarily_large_l200_200970

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib (n + 1) + fib n

theorem euclidean_steps_arbitrarily_large (n : ℕ) (h : n ≥ 2) :
  gcd (fib (n+1)) (fib n) = gcd (fib 1) (fib 0) := 
sorry

end euclidean_steps_arbitrarily_large_l200_200970


namespace original_denominator_l200_200032

theorem original_denominator (d : ℤ) (h1 : 5 = d + 3) : d = 12 := 
by 
  sorry

end original_denominator_l200_200032


namespace find_actual_weights_l200_200490

noncomputable def melon_weight : ℝ := 4.5
noncomputable def watermelon_weight : ℝ := 3.5
noncomputable def scale_error : ℝ := 0.5

def weight_bounds (actual_weight measured_weight error_margin : ℝ) :=
  (measured_weight - error_margin ≤ actual_weight) ∧ (actual_weight ≤ measured_weight + error_margin)

theorem find_actual_weights (x y : ℝ) 
  (melon_measured : x = 4)
  (watermelon_measured : y = 3)
  (combined_measured : x + y = 8.5)
  (hx : weight_bounds melon_weight x scale_error)
  (hy : weight_bounds watermelon_weight y scale_error)
  (h_combined : weight_bounds (melon_weight + watermelon_weight) (x + y) (2 * scale_error)) :
  x = melon_weight ∧ y = watermelon_weight := 
sorry

end find_actual_weights_l200_200490


namespace negation_of_existential_l200_200396

theorem negation_of_existential :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 - 3 > 0) = (∀ x : ℝ, x^2 + 2 * x - 3 ≤ 0) := 
by
  sorry

end negation_of_existential_l200_200396


namespace x_y_difference_l200_200101

theorem x_y_difference
    (x y : ℚ)
    (h1 : x + y = 780)
    (h2 : x / y = 1.25) :
    x - y = 86.66666666666667 :=
by
  sorry

end x_y_difference_l200_200101


namespace candied_apple_price_l200_200071

theorem candied_apple_price
  (x : ℝ) -- price of each candied apple in dollars
  (h1 : 15 * x + 12 * 1.5 = 48) -- total earnings equation
  : x = 2 := 
sorry

end candied_apple_price_l200_200071


namespace find_M_l200_200527

def grid_conditions :=
  ∃ (M : ℤ), 
  ∀ d1 d2 d3 d4, 
    (d1 = 22) ∧ (d2 = 6) ∧ (d3 = -34 / 6) ∧ (d4 = (8 - M) / 6) ∧
    (10 = 32 - d2) ∧ 
    (16 = 10 + d2) ∧ 
    (-2 = 10 - d2) ∧
    (32 - M = 34 / 6 * 6) ∧ 
    (M = -34 / 6 - (-17 / 3))

theorem find_M : grid_conditions → ∃ M : ℤ, M = 17 :=
by
  intros
  existsi (17 : ℤ) 
  sorry

end find_M_l200_200527


namespace geometric_sequence_problem_l200_200319

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (r : ℝ)
  (h₀ : ∀ n, a n > 0)
  (h₁ : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)
  (h₂ : ∀ n, a (n + 1) = a n * r) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sequence_problem_l200_200319


namespace intersection_points_A_B_segment_length_MN_l200_200853

section PolarCurves

-- Given conditions
def curve1 (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 8
def curve2 (θ : ℝ) : Prop := θ = Real.pi / 6
def is_on_line (x y t : ℝ) : Prop := x = 2 + Real.sqrt 3 / 2 * t ∧ y = 1 / 2 * t

-- Polar coordinates of points A and B
theorem intersection_points_A_B :
  ∃ (ρ₁ ρ₂ θ₁ θ₂ : ℝ), curve1 ρ₁ θ₁ ∧ curve2 θ₁ ∧ curve1 ρ₂ θ₂ ∧ curve2 θ₂ ∧
    (ρ₁, θ₁) = (4, Real.pi / 6) ∧ (ρ₂, θ₂) = (4, -Real.pi / 6) :=
sorry

-- Length of the segment MN
theorem segment_length_MN :
  ∀ t : ℝ, curve1 (2 + Real.sqrt 3 / 2 * t) (1 / 2 * t) →
    ∃ t₁ t₂ : ℝ, (is_on_line (2 + Real.sqrt 3 / 2 * t₁) (1 / 2 * t₁) t₁) ∧
                (is_on_line (2 + Real.sqrt 3 / 2 * t₂) (1 / 2 * t₂) t₂) ∧
                Real.sqrt ((2 * -Real.sqrt 3 * 4)^2 - 4 * (-8)) = 4 * Real.sqrt 5 :=
sorry

end PolarCurves

end intersection_points_A_B_segment_length_MN_l200_200853


namespace area_of_triangle_BQW_l200_200932

theorem area_of_triangle_BQW (AZ WC AB : ℝ) (h_trap_area : ℝ) (h_eq : AZ = WC) (AZ_val : AZ = 8) (AB_val : AB = 16) (trap_area_val : h_trap_area = 160) : 
  ∃ (BQW_area: ℝ), BQW_area = 48 :=
by
  let h_2 := 2 * h_trap_area / (AZ + AB)
  let h := AZ + h_2
  let BZW_area := h_trap_area - (1 / 2) * AZ * AB
  let BQW_area := 1 / 2 * BZW_area
  have AZ_eq : AZ = 8 := AZ_val
  have AB_eq : AB = 16 := AB_val
  have trap_area_eq : h_trap_area = 160 := trap_area_val
  let h_2_val : ℝ := 10 -- Calculated from h_2 = 2 * 160 / 32
  let h_val : ℝ := AZ + h_2_val -- full height
  let BZW_area_val : ℝ := 96 -- BZW area from 160 - 64
  let BQW_area_val : ℝ := 48 -- Half of BZW
  exact ⟨48, by sorry⟩ -- To complete the theorem

end area_of_triangle_BQW_l200_200932


namespace length_of_ae_l200_200421

-- Define the given consecutive points
variables (a b c d e : ℝ)

-- Conditions from the problem
-- 1. Points a, b, c, d, e are 5 consecutive points on a straight line - implicitly assumed on the same line
-- 2. bc = 2 * cd
-- 3. de = 4
-- 4. ab = 5
-- 5. ac = 11

theorem length_of_ae 
  (h1 : b - a = 5) -- ab = 5
  (h2 : c - a = 11) -- ac = 11
  (h3 : c - b = 2 * (d - c)) -- bc = 2 * cd
  (h4 : e - d = 4) -- de = 4
  : (e - a) = 18 := sorry

end length_of_ae_l200_200421


namespace soccer_team_lineups_l200_200204

-- Define the number of players in the team
def numPlayers : Nat := 16

-- Define the number of regular players to choose (excluding the goalie)
def numRegularPlayers : Nat := 10

-- Define the total number of starting lineups, considering the goalie and the combination of regular players
def totalStartingLineups : Nat :=
  numPlayers * Nat.choose (numPlayers - 1) numRegularPlayers

-- The theorem to prove
theorem soccer_team_lineups : totalStartingLineups = 48048 := by
  sorry

end soccer_team_lineups_l200_200204


namespace number_of_dolls_combined_l200_200317

-- Defining the given conditions as variables
variables (aida sophie vera : ℕ)

-- Given conditions
def condition1 : Prop := aida = 2 * sophie
def condition2 : Prop := sophie = 2 * vera
def condition3 : Prop := vera = 20

-- The final proof statement we need to prove
theorem number_of_dolls_combined (h1 : condition1 aida sophie) (h2 : condition2 sophie vera) (h3 : condition3 vera) : 
  aida + sophie + vera = 140 :=
  by sorry

end number_of_dolls_combined_l200_200317


namespace production_days_l200_200866

-- Definitions of the conditions
variables (n : ℕ) (P : ℕ)
variable (H1 : P = n * 50)
variable (H2 : (P + 60) / (n + 1) = 55)

-- Theorem to prove that n = 1 given the conditions
theorem production_days (n : ℕ) (P : ℕ) (H1 : P = n * 50) (H2 : (P + 60) / (n + 1) = 55) : n = 1 :=
by
  sorry

end production_days_l200_200866


namespace correct_choice_is_C_l200_200755

def first_quadrant_positive_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def right_angle_is_axial (θ : ℝ) : Prop :=
  θ = 90

def obtuse_angle_second_quadrant (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

def terminal_side_initial_side_same (θ : ℝ) : Prop :=
  θ = 0 ∨ θ = 360

theorem correct_choice_is_C : obtuse_angle_second_quadrant 120 :=
by
  sorry

end correct_choice_is_C_l200_200755


namespace time_after_seconds_l200_200411

def initial_time : Nat × Nat × Nat := (4, 45, 0)
def seconds_to_add : Nat := 12345
def final_time : Nat × Nat × Nat := (8, 30, 45)

theorem time_after_seconds (h : initial_time = (4, 45, 0) ∧ seconds_to_add = 12345) : 
  ∃ (h' : Nat × Nat × Nat), h' = final_time := by
  sorry

end time_after_seconds_l200_200411


namespace total_seats_l200_200122

theorem total_seats (s : ℕ) 
  (h1 : 30 + (0.20 * s : ℝ) + (0.60 * s : ℝ) = s) : s = 150 :=
  sorry

end total_seats_l200_200122


namespace set_membership_proof_l200_200854

variable (A : Set ℕ) (B : Set (Set ℕ))

theorem set_membership_proof :
  A = {0, 1} → B = {x | x ⊆ A} → A ∈ B :=
by
  intros hA hB
  rw [hA, hB]
  sorry

end set_membership_proof_l200_200854


namespace number_of_beakers_calculation_l200_200251

-- Conditions
def solution_per_test_tube : ℕ := 7
def number_of_test_tubes : ℕ := 6
def solution_per_beaker : ℕ := 14

-- Total amount of solution
def total_solution : ℕ := solution_per_test_tube * number_of_test_tubes

-- Number of beakers is the fraction of total solution and solution per beaker
def number_of_beakers : ℕ := total_solution / solution_per_beaker

-- Statement of the problem
theorem number_of_beakers_calculation : number_of_beakers = 3 :=
by 
  -- Proof goes here
  sorry

end number_of_beakers_calculation_l200_200251


namespace competition_end_time_is_5_35_am_l200_200368

def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes
def duration : Nat := 875  -- competition duration in minutes
def end_time : Nat := (start_time + duration) % (24 * 60)  -- competition end time in minutes

theorem competition_end_time_is_5_35_am :
  end_time = 5 * 60 + 35 :=  -- 5:35 a.m. in minutes
sorry

end competition_end_time_is_5_35_am_l200_200368


namespace trig_identity_l200_200388

theorem trig_identity (A : ℝ) (h : Real.cos (π + A) = -1/2) : Real.sin (π / 2 + A) = 1/2 :=
by 
sorry

end trig_identity_l200_200388


namespace min_xy_min_x_plus_y_l200_200981

theorem min_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : xy ≥ 36 :=
sorry  

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end min_xy_min_x_plus_y_l200_200981


namespace largest_sum_ABC_l200_200702

theorem largest_sum_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 3003) : 
  A + B + C ≤ 105 :=
sorry

end largest_sum_ABC_l200_200702


namespace maria_mushrooms_l200_200521

theorem maria_mushrooms (potatoes carrots onions green_beans bell_peppers mushrooms : ℕ) 
  (h1 : carrots = 6 * potatoes)
  (h2 : onions = 2 * carrots)
  (h3 : green_beans = onions / 3)
  (h4 : bell_peppers = 4 * green_beans)
  (h5 : mushrooms = 3 * bell_peppers)
  (h0 : potatoes = 3) : 
  mushrooms = 144 :=
by
  sorry

end maria_mushrooms_l200_200521


namespace find_triples_l200_200898

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x^2 + y^2 = 3 * 2016^z + 77) :
  (x = 4 ∧ y = 8 ∧ z = 0) ∨ (x = 8 ∧ y = 4 ∧ z = 0) ∨
  (x = 14 ∧ y = 77 ∧ z = 1) ∨ (x = 77 ∧ y = 14 ∧ z = 1) ∨
  (x = 35 ∧ y = 70 ∧ z = 1) ∨ (x = 70 ∧ y = 35 ∧ z = 1) :=
sorry

end find_triples_l200_200898


namespace find_f_3_l200_200390

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : 
  (∀ (x : ℝ), x ≠ 0 → 27 * f (-x) / x - x^2 * f (1 / x) = - 2 * x^2) →
  f 3 = 2 :=
sorry

end find_f_3_l200_200390


namespace managers_salary_l200_200329

-- Definitions based on conditions
def avg_salary_50_employees : ℝ := 2000
def num_employees : ℕ := 50
def new_avg_salary : ℝ := 2150
def num_employees_with_manager : ℕ := 51

-- Condition statement: The manager's salary such that when added, average salary increases as given.
theorem managers_salary (M : ℝ) :
  (num_employees * avg_salary_50_employees + M) / num_employees_with_manager = new_avg_salary →
  M = 9650 := sorry

end managers_salary_l200_200329


namespace geometric_sequence_problem_l200_200277

variable {a : ℕ → ℝ} -- Considering the sequence is a real number sequence
variable {q : ℝ} -- Common ratio

-- Conditions
axiom a2a6_eq_16 : a 2 * a 6 = 16
axiom a4_plus_a8_eq_8 : a 4 + a 8 = 8

-- Geometric sequence definition
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_problem : a 20 / a 10 = 1 :=
  by
  sorry

end geometric_sequence_problem_l200_200277


namespace division_addition_correct_l200_200203

theorem division_addition_correct : 0.2 / 0.005 + 0.1 = 40.1 :=
by
  sorry

end division_addition_correct_l200_200203


namespace total_cost_correct_l200_200644

variables (gravel_cost_per_ton : ℝ) (gravel_tons : ℝ)
variables (sand_cost_per_ton : ℝ) (sand_tons : ℝ)
variables (cement_cost_per_ton : ℝ) (cement_tons : ℝ)

noncomputable def total_cost : ℝ :=
  (gravel_cost_per_ton * gravel_tons) + (sand_cost_per_ton * sand_tons) + (cement_cost_per_ton * cement_tons)

theorem total_cost_correct :
  gravel_cost_per_ton = 30.5 → gravel_tons = 5.91 →
  sand_cost_per_ton = 40.5 → sand_tons = 8.11 →
  cement_cost_per_ton = 55.6 → cement_tons = 4.35 →
  total_cost gravel_cost_per_ton gravel_tons sand_cost_per_ton sand_tons cement_cost_per_ton cement_tons = 750.57 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end total_cost_correct_l200_200644


namespace unique_a_exists_for_prime_p_l200_200083

theorem unique_a_exists_for_prime_p (p : ℕ) [Fact p.Prime] :
  (∃! (a : ℕ), a ∈ Finset.range (p + 1) ∧ (a^3 - 3*a + 1) % p = 0) ↔ p = 3 := by
  sorry

end unique_a_exists_for_prime_p_l200_200083


namespace first_term_and_common_difference_l200_200302

theorem first_term_and_common_difference (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) :
  a 1 = 1 ∧ (a 2 - a 1) = 4 :=
by
  sorry

end first_term_and_common_difference_l200_200302


namespace complex_expr_evaluation_l200_200197

def complex_expr : ℤ :=
  2 * (3 * (2 * (3 * (2 * (3 * (2 + 1) * 2) + 2) * 2) + 2) * 2) + 2

theorem complex_expr_evaluation : complex_expr = 5498 := by
  sorry

end complex_expr_evaluation_l200_200197


namespace max_value_ab_bc_cd_da_l200_200280

theorem max_value_ab_bc_cd_da (a b c d : ℝ) (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d) (sum_eq_200 : a + b + c + d = 200) : 
  ab + bc + cd + 0.5 * d * a ≤ 11250 := 
sorry


end max_value_ab_bc_cd_da_l200_200280


namespace min_sum_reciprocals_of_roots_l200_200498

theorem min_sum_reciprocals_of_roots (k : ℝ) 
  (h_roots_positive : ∀ x : ℝ, (x^2 - k * x + k + 3 = 0) → 0 < x) :
  (k ≥ 6) → 
  ∀ x1 x2 : ℝ, (x1*x2 = k + 3) ∧ (x1 + x2 = k) ∧ (x1 > 0) ∧ (x2 > 0) → 
  (1 / x1 + 1 / x2) = 2 / 3 :=
by 
  -- proof steps go here
  sorry

end min_sum_reciprocals_of_roots_l200_200498


namespace printer_Z_time_l200_200020

theorem printer_Z_time (T_Z : ℝ) (h1 : (1.0 / 15.0 : ℝ) = (15.0 * ((1.0 / 12.0) + (1.0 / T_Z))) / 2.0833333333333335) : 
  T_Z = 18.0 :=
sorry

end printer_Z_time_l200_200020


namespace angle_E_measure_l200_200491

-- Definition of degrees for each angle in the quadrilateral
def angle_measure (E F G H : ℝ) : Prop :=
  E = 3 * F ∧ E = 4 * G ∧ E = 6 * H ∧ E + F + G + H = 360

-- Prove the measure of angle E
theorem angle_E_measure (E F G H : ℝ) (h : angle_measure E F G H) : E = 360 * (4 / 7) :=
by {
  sorry
}

end angle_E_measure_l200_200491


namespace problem1_problem2_problem3_l200_200176

theorem problem1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 :=
by
  sorry

theorem problem2 (a : ℝ) : (-2 * a^2)^3 * a^2 + a^8 = -7 * a^8 :=
by
  sorry

theorem problem3 : 2023^2 - 2024 * 2022 = 1 :=
by
  sorry

end problem1_problem2_problem3_l200_200176


namespace product_in_third_quadrant_l200_200829

def z1 : ℂ := 1 - 3 * Complex.I
def z2 : ℂ := 3 - 2 * Complex.I
def z := z1 * z2

theorem product_in_third_quadrant : z.re < 0 ∧ z.im < 0 := 
sorry

end product_in_third_quadrant_l200_200829


namespace total_packs_l200_200360

noncomputable def robyn_packs : ℕ := 16
noncomputable def lucy_packs : ℕ := 19

theorem total_packs : robyn_packs + lucy_packs = 35 := by
  sorry

end total_packs_l200_200360


namespace find_angle_A_l200_200950

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 2) (h3 : B = Real.pi / 4) : A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l200_200950


namespace ravi_prakash_finish_together_l200_200686

theorem ravi_prakash_finish_together (ravi_days prakash_days : ℕ) (h_ravi : ravi_days = 15) (h_prakash : prakash_days = 30) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 10 := 
by
  sorry

end ravi_prakash_finish_together_l200_200686


namespace Jeff_pays_when_picking_up_l200_200937

-- Definition of the conditions
def deposit_rate : ℝ := 0.10
def increase_rate : ℝ := 0.40
def last_year_cost : ℝ := 250
def this_year_cost : ℝ := last_year_cost * (1 + increase_rate)
def deposit : ℝ := this_year_cost * deposit_rate

-- Lean statement of the proof
theorem Jeff_pays_when_picking_up : this_year_cost - deposit = 315 := by
  sorry

end Jeff_pays_when_picking_up_l200_200937


namespace fisherman_total_fish_l200_200953

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end fisherman_total_fish_l200_200953


namespace inequality_proof_l200_200355

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 := 
by 
  sorry

end inequality_proof_l200_200355


namespace smallest_number_of_students_l200_200971

/--
At a school, the ratio of 10th-graders to 8th-graders is 3:2, 
and the ratio of 10th-graders to 9th-graders is 5:3. 
Prove that the smallest number of students from these grades is 34.
-/
theorem smallest_number_of_students {G8 G9 G10 : ℕ} 
  (h1 : 3 * G8 = 2 * G10) (h2 : 5 * G9 = 3 * G10) : 
  G10 + G8 + G9 = 34 :=
by
  sorry

end smallest_number_of_students_l200_200971


namespace min_amount_for_free_shipping_l200_200361

def book1 : ℝ := 13.00
def book2 : ℝ := 15.00
def book3 : ℝ := 10.00
def book4 : ℝ := 10.00
def discount_rate : ℝ := 0.25
def shipping_threshold : ℝ := 9.00

def total_cost_before_discount : ℝ := book1 + book2 + book3 + book4
def discount_amount : ℝ := book1 * discount_rate + book2 * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

theorem min_amount_for_free_shipping : total_cost_after_discount + shipping_threshold = 50.00 :=
by
  sorry

end min_amount_for_free_shipping_l200_200361


namespace log21_requires_additional_information_l200_200037

noncomputable def log3 : ℝ := 0.4771
noncomputable def log5 : ℝ := 0.6990

theorem log21_requires_additional_information
  (log3_given : log3 = 0.4771)
  (log5_given : log5 = 0.6990) :
  ¬ (∃ c₁ c₂ : ℝ, log21 = c₁ * log3 + c₂ * log5) :=
sorry

end log21_requires_additional_information_l200_200037


namespace range_of_a_l200_200023

open Real

theorem range_of_a (k a : ℝ) : 
  (∀ k : ℝ, ∀ x y : ℝ, k * x - y - k + 2 = 0 → x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-7 : ℝ) (-2) ∪ Set.Ioi 1) := 
sorry

end range_of_a_l200_200023


namespace tangent_ellipse_hyperbola_l200_200916

-- Definitions of the curves
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y+3)^2 = 1

-- Condition for tangency: the curves must meet and the discriminant must be zero
noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Prove the given curves are tangent at some x and y for m = 8/9
theorem tangent_ellipse_hyperbola : 
    (∃ x y : ℝ, ellipse x y ∧ hyperbola x y (8 / 9)) ∧ 
    quadratic_discriminant ((8 / 9) + 9) (6 * (8 / 9)) ((-8/9) * (8 * (8/9)) - 8) = 0 :=
sorry

end tangent_ellipse_hyperbola_l200_200916


namespace hexagon_largest_angle_l200_200617

variable (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ)
theorem hexagon_largest_angle (h : a₁ = 3)
                             (h₀ : a₂ = 3)
                             (h₁ : a₃ = 3)
                             (h₂ : a₄ = 4)
                             (h₃ : a₅ = 5)
                             (h₄ : a₆ = 6)
                             (sum_angles : 3*a₁ + 3*a₀ + 3*a₁ + 4*a₂ + 5*a₃ + 6*a₄ = 720) :
                             6 * 30 = 180 := by
    sorry

end hexagon_largest_angle_l200_200617


namespace impossible_path_2018_grid_l200_200395

theorem impossible_path_2018_grid :
  ¬((∃ (path : Finset (Fin 2018 × Fin 2018)), 
    (0, 0) ∈ path ∧ (2017, 2017) ∈ path ∧ 
    (∀ {x y}, (x, y) ∈ path → (x + 1, y) ∈ path ∨ (x, y + 1) ∈ path ∨ (x - 1, y) ∈ path ∨ (x, y - 1) ∈ path) ∧ 
    (∀ {x y}, (x, y) ∈ path → (Finset.card path = 2018 * 2018)))) :=
by 
  sorry

end impossible_path_2018_grid_l200_200395


namespace remainder_of_349_divided_by_17_l200_200857

theorem remainder_of_349_divided_by_17 : 
  (349 % 17 = 9) := 
by
  sorry

end remainder_of_349_divided_by_17_l200_200857


namespace MissAisha_height_l200_200052

theorem MissAisha_height (H : ℝ)
  (legs_length : ℝ := H / 3)
  (head_length : ℝ := H / 4)
  (rest_body_length : ℝ := 25) :
  H = 60 :=
by sorry

end MissAisha_height_l200_200052


namespace inequality_2_pow_ge_n_sq_l200_200219

theorem inequality_2_pow_ge_n_sq (n : ℕ) (hn : n ≠ 3) : 2^n ≥ n^2 :=
sorry

end inequality_2_pow_ge_n_sq_l200_200219


namespace quadratic_transformation_l200_200896

theorem quadratic_transformation (a b c : ℝ) (h : a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) : 
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end quadratic_transformation_l200_200896


namespace correct_answer_is_B_l200_200247

-- Definitions for each set of line segments
def setA := (2, 2, 4)
def setB := (8, 6, 3)
def setC := (2, 6, 3)
def setD := (11, 4, 6)

-- Triangle inequality theorem checking function
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statements to verify each set
lemma check_setA : ¬ is_triangle 2 2 4 := by sorry
lemma check_setB : is_triangle 8 6 3 := by sorry
lemma check_setC : ¬ is_triangle 2 6 3 := by sorry
lemma check_setD : ¬ is_triangle 11 4 6 := by sorry

-- Final theorem combining all checks to match the given problem
theorem correct_answer_is_B : 
  ¬ is_triangle 2 2 4 ∧ is_triangle 8 6 3 ∧ ¬ is_triangle 2 6 3 ∧ ¬ is_triangle 11 4 6 :=
by sorry

end correct_answer_is_B_l200_200247


namespace cube_pyramid_volume_l200_200158

theorem cube_pyramid_volume (s b h : ℝ) 
  (hcube : s = 6) 
  (hbase : b = 10)
  (eq_volumes : (s ^ 3) = (1 / 3) * (b ^ 2) * h) : 
  h = 162 / 25 := 
by 
  sorry

end cube_pyramid_volume_l200_200158


namespace simple_interest_rate_l200_200899

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 800) (hSI : SI = 128) (hT : T = 4) : 
  (SI = P * (R : ℝ) * T / 100) → R = 4 := 
by {
  -- Proof goes here.
  sorry
}

end simple_interest_rate_l200_200899


namespace parking_fines_l200_200260

theorem parking_fines (total_citations littering_citations offleash_dog_citations parking_fines : ℕ) 
  (h1 : total_citations = 24) 
  (h2 : littering_citations = 4) 
  (h3 : offleash_dog_citations = 4) 
  (h4 : total_citations = littering_citations + offleash_dog_citations + parking_fines) : 
  parking_fines = 16 := 
by 
  sorry

end parking_fines_l200_200260


namespace combined_difference_is_correct_l200_200883

-- Define the number of cookies each person has
def alyssa_cookies : Nat := 129
def aiyanna_cookies : Nat := 140
def carl_cookies : Nat := 167

-- Define the differences between each pair of people's cookies
def diff_alyssa_aiyanna : Nat := aiyanna_cookies - alyssa_cookies
def diff_alyssa_carl : Nat := carl_cookies - alyssa_cookies
def diff_aiyanna_carl : Nat := carl_cookies - aiyanna_cookies

-- Define the combined difference
def combined_difference : Nat := diff_alyssa_aiyanna + diff_alyssa_carl + diff_aiyanna_carl

-- State the theorem to be proved
theorem combined_difference_is_correct : combined_difference = 76 := by
  sorry

end combined_difference_is_correct_l200_200883


namespace cosine_greater_sine_cosine_cos_greater_sine_sin_l200_200166

variable {f g : ℝ → ℝ}

-- Problem 1
theorem cosine_greater_sine (h : ∀ x, - (Real.pi / 2) < f x + g x ∧ f x + g x < Real.pi / 2
                            ∧ - (Real.pi / 2) < f x - g x ∧ f x - g x < Real.pi / 2) :
  ∀ x, Real.cos (f x) > Real.sin (g x) :=
sorry

-- Problem 2
theorem cosine_cos_greater_sine_sin (x : ℝ) :  Real.cos (Real.cos x) > Real.sin (Real.sin x) :=
sorry

end cosine_greater_sine_cosine_cos_greater_sine_sin_l200_200166


namespace inequality_solutions_l200_200794

theorem inequality_solutions (n : ℕ) (h : n > 0) : n^3 - n < n! ↔ (n = 1 ∨ n ≥ 6) := 
by
  sorry

end inequality_solutions_l200_200794


namespace solve_pounds_l200_200933

def price_per_pound_corn : ℝ := 1.20
def price_per_pound_beans : ℝ := 0.60
def price_per_pound_rice : ℝ := 0.80
def total_weight : ℕ := 30
def total_cost : ℝ := 24.00
def equal_beans_rice (b r : ℕ) : Prop := b = r

theorem solve_pounds (c b r : ℕ) (h1 : price_per_pound_corn * ↑c + price_per_pound_beans * ↑b + price_per_pound_rice * ↑r = total_cost)
    (h2 : c + b + r = total_weight) (h3 : equal_beans_rice b r) : c = 6 ∧ b = 12 ∧ r = 12 := by
  sorry

end solve_pounds_l200_200933


namespace find_smaller_number_l200_200206

-- Define the conditions as hypotheses and the goal as a proposition
theorem find_smaller_number (x y : ℕ) (h1 : x + y = 77) (h2 : x = 42 ∨ y = 42) (h3 : 5 * x = 6 * y) : x = 35 :=
sorry

end find_smaller_number_l200_200206


namespace train_cross_platform_time_l200_200245

noncomputable def kmph_to_mps (s : ℚ) : ℚ :=
  (s * 1000) / 3600

theorem train_cross_platform_time :
  let train_length := 110
  let speed_kmph := 52
  let platform_length := 323.36799999999994
  let speed_mps := kmph_to_mps 52
  let total_distance := train_length + platform_length
  let time := total_distance / speed_mps
  time = 30 := 
by
  sorry

end train_cross_platform_time_l200_200245


namespace complement_set_l200_200383

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x | x < -2 ∨ x > 2}

-- The mathematical proof to be stated
theorem complement_set :
  U \ M = complement_M_in_U := sorry

end complement_set_l200_200383


namespace probability_of_shaded_triangle_l200_200240

def triangle (name: String) := name

def triangles := ["AEC", "AEB", "BED", "BEC", "BDC", "ABD"]
def shaded_triangles := ["BEC", "BDC", "ABD"]

theorem probability_of_shaded_triangle :
  (shaded_triangles.length : ℚ) / (triangles.length : ℚ) = 1 / 2 := 
by
  sorry

end probability_of_shaded_triangle_l200_200240


namespace smallest_m_l200_200977

theorem smallest_m (m : ℕ) (h1 : 7 ≡ 2 [MOD 5]) : 
  (7^m ≡ m^7 [MOD 5]) ↔ (m = 7) :=
by sorry

end smallest_m_l200_200977


namespace triple_overlap_area_correct_l200_200785

-- Define the dimensions of the auditorium and carpets
def auditorium_dim : ℕ × ℕ := (10, 10)
def carpet1_dim : ℕ × ℕ := (6, 8)
def carpet2_dim : ℕ × ℕ := (6, 6)
def carpet3_dim : ℕ × ℕ := (5, 7)

-- The coordinates and dimensions of the overlap regions are derived based on the given positions
-- Here we assume derivations as described in the solution steps without recalculating them

-- Overlap area of the second and third carpets
def overlap23 : ℕ × ℕ := (5, 3)

-- Intersection of this overlap with the first carpet
def overlap_all : ℕ × ℕ := (2, 3)

-- Calculate the area of the region where all three carpets overlap
def triple_overlap_area : ℕ :=
  (overlap_all.1 * overlap_all.2)

theorem triple_overlap_area_correct :
  triple_overlap_area = 6 := by
  -- Expected result should be 6 square meters
  sorry

end triple_overlap_area_correct_l200_200785


namespace divisibility_by_cube_greater_than_1_l200_200447

theorem divisibility_by_cube_greater_than_1 (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hdiv : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) :
  ∃ k : ℕ, 1 < k ∧ k^3 ∣ a^2 + 3 * a * b + 3 * b^2 - 1 := 
by {
  sorry
}

end divisibility_by_cube_greater_than_1_l200_200447


namespace cos_17_pi_over_6_l200_200476

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * 180 / Real.pi

theorem cos_17_pi_over_6 : Real.cos (17 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_17_pi_over_6_l200_200476


namespace domain_of_f_eq_l200_200115

noncomputable def domain_of_f (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_f_eq :
  { x : ℝ | domain_of_f x} = { x : ℝ | -1 ≤ x ∧ x < 0 } ∪ { x : ℝ | 0 < x } :=
by
  sorry

end domain_of_f_eq_l200_200115


namespace least_pos_int_satisfies_conditions_l200_200483

theorem least_pos_int_satisfies_conditions :
  ∃ x : ℕ, x > 0 ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  x = 419 :=
by
  sorry

end least_pos_int_satisfies_conditions_l200_200483


namespace initial_files_count_l200_200806

theorem initial_files_count (deleted_files folders files_per_folder total_files initial_files : ℕ)
    (h1 : deleted_files = 21)
    (h2 : folders = 9)
    (h3 : files_per_folder = 8)
    (h4 : total_files = folders * files_per_folder)
    (h5 : initial_files = total_files + deleted_files) :
    initial_files = 93 :=
by
  sorry

end initial_files_count_l200_200806


namespace num_blue_balls_l200_200765

theorem num_blue_balls (total_balls blue_balls : ℕ) 
  (prob_all_blue : ℚ)
  (h_total : total_balls = 12)
  (h_prob : prob_all_blue = 1 / 55)
  (h_prob_eq : (blue_balls / 12) * ((blue_balls - 1) / 11) * ((blue_balls - 2) / 10) = prob_all_blue) :
  blue_balls = 4 :=
by
  -- Placeholder for proof
  sorry

end num_blue_balls_l200_200765


namespace correct_result_l200_200489

theorem correct_result (x : ℕ) (h: (325 - x) * 5 = 1500) : 325 - x * 5 = 200 := 
by
  -- placeholder for proof
  sorry

end correct_result_l200_200489


namespace new_price_of_sugar_l200_200442

theorem new_price_of_sugar (C : ℝ) (H : 10 * C = P * (0.7692307692307693 * C)) : P = 13 := by
  sorry

end new_price_of_sugar_l200_200442


namespace sandy_spent_on_repairs_l200_200250

theorem sandy_spent_on_repairs (initial_cost : ℝ) (selling_price : ℝ) (gain_percent : ℝ) (repair_cost : ℝ) :
  initial_cost = 800 → selling_price = 1400 → gain_percent = 40 → selling_price = 1.4 * (initial_cost + repair_cost) → repair_cost = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_spent_on_repairs_l200_200250


namespace f_shift_l200_200745

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the main theorem
theorem f_shift (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) :=
by
  sorry

end f_shift_l200_200745


namespace inequality_solution_l200_200917

theorem inequality_solution (x : ℝ) : 
  (x < -4 ∨ x > 2) ↔ (x^2 + 3 * x - 4) / (x^2 - x - 2) > 0 :=
sorry

end inequality_solution_l200_200917


namespace horizontal_asymptote_l200_200477

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 7 * x^3 + 10 * x^2 + 6 * x + 4) / (4 * x^4 + 3 * x^3 + 9 * x^2 + 4 * x + 2)

theorem horizontal_asymptote :
  ∃ L : ℝ, (∀ ε > 0, ∃ M > 0, ∀ x > M, |rational_function x - L| < ε) → L = 15 / 4 :=
by
  sorry

end horizontal_asymptote_l200_200477


namespace pizza_toppings_combination_l200_200931

def num_combinations {α : Type} (s : Finset α) (k : ℕ) : ℕ :=
  (s.card.choose k)

theorem pizza_toppings_combination (s : Finset ℕ) (h : s.card = 7) : num_combinations s 3 = 35 :=
by
  sorry

end pizza_toppings_combination_l200_200931


namespace hyperbola_standard_equation_l200_200753

def ellipse_equation (x y : ℝ) : Prop :=
  (y^2) / 16 + (x^2) / 12 = 1

def hyperbola_equation (x y : ℝ) : Prop :=
  (y^2) / 2 - (x^2) / 2 = 1

def passes_through_point (x y : ℝ) : Prop :=
  x = 1 ∧ y = Real.sqrt 3

theorem hyperbola_standard_equation (x y : ℝ) (hx : passes_through_point x y)
  (ellipse_foci_shared : ∀ x y : ℝ, ellipse_equation x y → ellipse_equation x y)
  : hyperbola_equation x y := 
sorry

end hyperbola_standard_equation_l200_200753


namespace line_passes_fixed_point_l200_200954

theorem line_passes_fixed_point (k b : ℝ) (h : -1 = (k + b) / 2) :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ y = k * x + b :=
by
  sorry

end line_passes_fixed_point_l200_200954


namespace constant_term_of_second_eq_l200_200607

theorem constant_term_of_second_eq (x y : ℝ) 
  (h1 : 7*x + y = 19) 
  (h2 : 2*x + y = 5) : 
  ∃ k : ℝ, x + 3*y = k ∧ k = 15 := 
by
  sorry

end constant_term_of_second_eq_l200_200607


namespace tangents_secant_intersect_l200_200202

variable {A B C O1 P Q R : Type} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]
variables (AB AC : Set (MetricSpace A)) (t : Tangent AB) (s : Tangent AC)

variable (BC : line ( Set A))
variable (APQ : secant A P Q) 

theorem tangents_secant_intersect { AR AP AQ : ℝ } :
  2 / AR = 1 / AP + 1 / AQ :=
by
  sorry

end tangents_secant_intersect_l200_200202


namespace not_perfect_square_2023_l200_200406

theorem not_perfect_square_2023 : ¬ (∃ x : ℤ, x^2 = 5^2023) := 
sorry

end not_perfect_square_2023_l200_200406


namespace larger_segment_of_triangle_l200_200272

theorem larger_segment_of_triangle (x y : ℝ) (h1 : 40^2 = x^2 + y^2) 
  (h2 : 90^2 = (100 - x)^2 + y^2) :
  100 - x = 82.5 :=
by {
  sorry
}

end larger_segment_of_triangle_l200_200272


namespace complementary_event_target_l200_200034

theorem complementary_event_target (S : Type) (hit miss : S) (shoots : ℕ → S) :
  (∀ n : ℕ, (shoots n = hit ∨ shoots n = miss)) →
  (∃ n : ℕ, shoots n = hit) ↔ (∀ n : ℕ, shoots n ≠ hit) :=
by
sorry

end complementary_event_target_l200_200034


namespace calculate_non_defective_m3_percentage_l200_200629

def percentage_non_defective_m3 : ℝ := 93

theorem calculate_non_defective_m3_percentage 
  (P : ℝ) -- Total number of products
  (P_pos : 0 < P) -- Total number of products is positive
  (percentage_m1 : ℝ := 0.40)
  (percentage_m2 : ℝ := 0.30)
  (percentage_m3 : ℝ := 0.30)
  (defective_m1 : ℝ := 0.03)
  (defective_m2 : ℝ := 0.01)
  (total_defective : ℝ := 0.036) :
  percentage_non_defective_m3 = 93 :=
by sorry -- The actual proof is omitted

end calculate_non_defective_m3_percentage_l200_200629


namespace graph_of_f_4_minus_x_l200_200167

theorem graph_of_f_4_minus_x (f : ℝ → ℝ) (h : f 0 = 1) : f (4 - 4) = 1 :=
by
  rw [sub_self]
  exact h

end graph_of_f_4_minus_x_l200_200167


namespace find_density_of_gold_l200_200497

theorem find_density_of_gold
  (side_length : ℝ)
  (gold_cost_per_gram : ℝ)
  (sale_factor : ℝ)
  (profit : ℝ)
  (density_of_gold : ℝ) :
  side_length = 6 →
  gold_cost_per_gram = 60 →
  sale_factor = 1.5 →
  profit = 123120 →
  density_of_gold = 19 :=
sorry

end find_density_of_gold_l200_200497


namespace problem_statement_l200_200226

open Complex

noncomputable def z : ℂ := ((1 - I)^2 + 3 * (1 + I)) / (2 - I)

theorem problem_statement :
  z = 1 + I ∧ (∀ (a b : ℝ), (z^2 + a * z + b = 1 - I) → (a = -3 ∧ b = 4)) :=
by
  sorry

end problem_statement_l200_200226


namespace election_votes_l200_200836

noncomputable def third_candidate_votes (total_votes first_candidate_votes second_candidate_votes : ℕ) (winning_fraction : ℚ) : ℕ :=
  total_votes - (first_candidate_votes + second_candidate_votes)

theorem election_votes :
  ∃ total_votes : ℕ, 
  ∃ first_candidate_votes : ℕ,
  ∃ second_candidate_votes : ℕ,
  ∃ winning_fraction : ℚ,
  first_candidate_votes = 5000 ∧ 
  second_candidate_votes = 15000 ∧ 
  winning_fraction = 2/3 ∧ 
  total_votes = 60000 ∧ 
  third_candidate_votes total_votes first_candidate_votes second_candidate_votes winning_fraction = 40000 :=
    sorry

end election_votes_l200_200836


namespace thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l200_200565

theorem thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five :
  (35 * 99 ≠ 35 * 100 + 35) :=
by
  sorry

end thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l200_200565


namespace minimum_trucks_needed_l200_200002

theorem minimum_trucks_needed (total_weight : ℝ) (box_weight : ℕ → ℝ) 
  (n : ℕ) (H_total_weight : total_weight = 10) 
  (H_box_weight : ∀ i, box_weight i ≤ 1) 
  (truck_capacity : ℝ) 
  (H_truck_capacity : truck_capacity = 3) : 
  n = 5 :=
by {
  sorry
}

end minimum_trucks_needed_l200_200002


namespace vector_magnitude_difference_l200_200367

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

-- Statement to prove that the magnitude of the difference of vectors a and b is 5
theorem vector_magnitude_difference : ‖a - b‖ = 5 := 
sorry -- Proof omitted

end vector_magnitude_difference_l200_200367


namespace find_initial_mice_l200_200450

theorem find_initial_mice : 
  ∃ x : ℕ, (∀ (h1 : ∀ (m : ℕ), m * 2 = m + m), (35 * x = 280) → x = 8) :=
by
  existsi 8
  intro h1 h2
  sorry

end find_initial_mice_l200_200450


namespace original_money_l200_200183
noncomputable def original_amount (x : ℝ) :=
  let after_first_loss := (2/3) * x
  let after_first_win := after_first_loss + 10
  let after_second_loss := after_first_win - (1/3) * after_first_win
  let after_second_win := after_second_loss + 20
  after_second_win

theorem original_money (x : ℝ) (h : original_amount x = x) : x = 48 :=
by {
  sorry
}

end original_money_l200_200183


namespace find_p8_l200_200662

noncomputable def p (x : ℝ) : ℝ := sorry -- p is a monic polynomial of degree 7

def monic_degree_7 (p : ℝ → ℝ) : Prop := sorry -- p is monic polynomial of degree 7
def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 4 ∧ p 4 = 5 ∧ p 5 = 6 ∧ p 6 = 7 ∧ p 7 = 8

theorem find_p8 (h_monic : monic_degree_7 p) (h_conditions : satisfies_conditions p) : p 8 = 5049 :=
by
  sorry

end find_p8_l200_200662


namespace degree_polynomial_is_13_l200_200391

noncomputable def degree_polynomial (a b c d e f g h j : ℝ) : ℕ :=
  (7 + 4 + 2)

theorem degree_polynomial_is_13 (a b c d e f g h j : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0) (hh : h ≠ 0) (hj : j ≠ 0) : 
  degree_polynomial a b c d e f g h j = 13 :=
by
  rfl

end degree_polynomial_is_13_l200_200391


namespace minimum_students_lost_all_items_l200_200222

def smallest_number (N A B C : ℕ) (x : ℕ) : Prop :=
  N = 30 ∧ A = 26 ∧ B = 23 ∧ C = 21 → x ≥ 10

theorem minimum_students_lost_all_items (N A B C : ℕ) : 
  smallest_number N A B C 10 := 
by {
  sorry
}

end minimum_students_lost_all_items_l200_200222


namespace find_A_l200_200058

theorem find_A (A B : ℚ) (h1 : B - A = 211.5) (h2 : B = 10 * A) : A = 23.5 :=
by sorry

end find_A_l200_200058


namespace chips_cost_l200_200656

noncomputable def cost_of_each_bag_of_chips (amount_paid_per_friend : ℕ) (number_of_friends : ℕ) (number_of_bags : ℕ) : ℕ :=
  (amount_paid_per_friend * number_of_friends) / number_of_bags

theorem chips_cost
  (amount_paid_per_friend : ℕ := 5)
  (number_of_friends : ℕ := 3)
  (number_of_bags : ℕ := 5) :
  cost_of_each_bag_of_chips amount_paid_per_friend number_of_friends number_of_bags = 3 :=
by
  sorry

end chips_cost_l200_200656


namespace a_gt_one_l200_200935

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 - x - 1

theorem a_gt_one (a : ℝ) :
  (∃! x, 0 < x ∧ x < 1 ∧ f a x = 0) → 1 < a :=
by
  sorry

end a_gt_one_l200_200935


namespace equal_distribution_l200_200511

theorem equal_distribution (total_cookies bags : ℕ) (h_total : total_cookies = 14) (h_bags : bags = 7) : total_cookies / bags = 2 := by
  sorry

end equal_distribution_l200_200511


namespace calculate_expression_l200_200284

theorem calculate_expression : (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end calculate_expression_l200_200284


namespace original_price_l200_200427

variable (P : ℝ)
variable (S : ℝ := 140)
variable (discount : ℝ := 0.60)

theorem original_price :
  (S = P * (1 - discount)) → (P = 350) :=
by
  sorry

end original_price_l200_200427


namespace triangle_proof_l200_200378

-- Declare a structure for a triangle with given conditions
structure TriangleABC :=
  (a b c : ℝ) -- sides opposite to angles A, B, and C
  (A B C : ℝ) -- angles A, B, and C
  (R : ℝ) -- circumcircle radius
  (r : ℝ := 3) -- inradius is given as 3
  (area : ℝ := 6) -- area of the triangle is 6
  (h1 : a * Real.cos A + b * Real.cos B + c * Real.cos C = R / 3) -- given condition
  (h2 : ∀ a b c A B C, a * Real.sin A + b * Real.sin B + c * Real.sin C = 2 * area / (a+b+c)) -- implied area condition

-- Define the theorem using the above conditions
theorem triangle_proof (t : TriangleABC) :
  t.a + t.b + t.c = 4 ∧
  (Real.sin (2 * t.A) + Real.sin (2 * t.B) + Real.sin (2 * t.C)) = 1/3 ∧
  t.R = 6 :=
by
  sorry

end triangle_proof_l200_200378


namespace total_surface_area_correct_l200_200877

-- Definitions for side lengths of the cubes
def side_length_large := 5
def side_length_medium := 2
def side_length_small := 1

-- Surface area calculation for a single cube
def surface_area (side_length : ℕ) : ℕ := 6 * side_length^2

-- Surface areas for each size of the cube
def surface_area_large := surface_area side_length_large
def surface_area_medium := surface_area side_length_medium
def surface_area_small := surface_area side_length_small

-- Total surface areas for medium and small cubes
def surface_area_medium_total := 4 * surface_area_medium
def surface_area_small_total := 4 * surface_area_small

-- Total surface area of the structure
def total_surface_area := surface_area_large + surface_area_medium_total + surface_area_small_total

-- Expected result
def expected_surface_area := 270

-- Proof statement
theorem total_surface_area_correct : total_surface_area = expected_surface_area := by
  sorry

end total_surface_area_correct_l200_200877


namespace time_to_reach_madison_l200_200064

-- Definitions based on the conditions
def map_distance : ℝ := 5 -- inches
def average_speed : ℝ := 60 -- miles per hour
def map_scale : ℝ := 0.016666666666666666 -- inches per mile

-- The time taken by Pete to arrive in Madison
noncomputable def time_to_madison := map_distance / map_scale / average_speed

-- The theorem to prove
theorem time_to_reach_madison : time_to_madison = 5 := 
by
  sorry

end time_to_reach_madison_l200_200064


namespace geometric_sequence_a3_value_l200_200636

theorem geometric_sequence_a3_value
  {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 82)
  (h2 : a 2 * a 4 = 81)
  (h3 : ∀ n : ℕ, a (n + 1) = a n * a 3 / a 2) :
  a 3 = 9 :=
sorry

end geometric_sequence_a3_value_l200_200636


namespace y_intercept_of_line_l200_200949

def equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

theorem y_intercept_of_line : equation 0 (-2) :=
by
  sorry

end y_intercept_of_line_l200_200949


namespace seats_scientific_notation_l200_200408

theorem seats_scientific_notation : 
  (13000 = 1.3 * 10^4) := 
by 
  sorry 

end seats_scientific_notation_l200_200408


namespace number_of_cows_l200_200552

variable {D C : ℕ}

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 :=
by sorry

end number_of_cows_l200_200552


namespace truck_tank_percentage_increase_l200_200114

-- Declaration of the initial conditions (as given in the problem)
def service_cost : ℝ := 2.20
def fuel_cost_per_liter : ℝ := 0.70
def num_minivans : ℕ := 4
def num_trucks : ℕ := 2
def total_cost : ℝ := 395.40
def minivan_tank_size : ℝ := 65.0

-- Proof statement with the conditions declared above
theorem truck_tank_percentage_increase :
  ∃ p : ℝ, p = 120 ∧ (minivan_tank_size * (p + 100) / 100 = 143) :=
sorry

end truck_tank_percentage_increase_l200_200114


namespace circles_equal_or_tangent_l200_200400

theorem circles_equal_or_tangent (a b c : ℝ) 
  (h : (2 * a)^2 - 4 * (b^2 - c * (b - a)) = 0) : 
  a = b ∨ c = a + b :=
by
  -- Will fill the proof later
  sorry

end circles_equal_or_tangent_l200_200400


namespace find_absolute_difference_l200_200771

def condition_avg_sum (m n : ℝ) : Prop :=
  m + n + 5 + 6 + 4 = 25

def condition_variance (m n : ℝ) : Prop :=
  (m - 5) ^ 2 + (n - 5) ^ 2 = 8

theorem find_absolute_difference (m n : ℝ) (h1 : condition_avg_sum m n) (h2 : condition_variance m n) : |m - n| = 4 :=
sorry

end find_absolute_difference_l200_200771


namespace sum_gcd_lcm_is_39_l200_200984

theorem sum_gcd_lcm_is_39 : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by 
  sorry

end sum_gcd_lcm_is_39_l200_200984


namespace negation_of_at_most_one_obtuse_l200_200812

-- Defining a predicate to express the concept of an obtuse angle
def is_obtuse (θ : ℝ) : Prop := θ > 90

-- Defining a triangle with three interior angles α, β, and γ
structure Triangle :=
  (α β γ : ℝ)
  (sum_angles : α + β + γ = 180)

-- Defining the condition that "At most, only one interior angle of a triangle is obtuse"
def at_most_one_obtuse (T : Triangle) : Prop :=
  (is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ is_obtuse T.β ∧ ¬ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ is_obtuse T.γ) ∨
  (¬ is_obtuse T.α ∧ ¬ is_obtuse T.β ∧ ¬ is_obtuse T.γ)

-- The theorem we want to prove: Negation of "At most one obtuse angle" is "At least two obtuse angles"
theorem negation_of_at_most_one_obtuse (T : Triangle) :
  ¬ at_most_one_obtuse T ↔ (is_obtuse T.α ∧ is_obtuse T.β) ∨ (is_obtuse T.α ∧ is_obtuse T.γ) ∨ (is_obtuse T.β ∧ is_obtuse T.γ) := by
  sorry

end negation_of_at_most_one_obtuse_l200_200812


namespace problem_complement_intersection_l200_200678

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

def complement (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

theorem problem_complement_intersection :
  (complement U M) ∩ N = {3} :=
by
  sorry

end problem_complement_intersection_l200_200678


namespace Dacid_weighted_average_l200_200394

noncomputable def DacidMarks := 86 * 3 + 85 * 4 + 92 * 4 + 87 * 3 + 95 * 3 + 89 * 2 + 75 * 1
noncomputable def TotalCreditHours := 3 + 4 + 4 + 3 + 3 + 2 + 1
noncomputable def WeightedAverageMarks := (DacidMarks : ℝ) / (TotalCreditHours : ℝ)

theorem Dacid_weighted_average :
  WeightedAverageMarks = 88.25 :=
sorry

end Dacid_weighted_average_l200_200394


namespace find_second_number_l200_200095

def sum_of_three (a b c : ℚ) : Prop :=
  a + b + c = 120

def ratio_first_to_second (a b : ℚ) : Prop :=
  a / b = 3 / 4

def ratio_second_to_third (b c : ℚ) : Prop :=
  b / c = 3 / 5

theorem find_second_number (a b c : ℚ) 
  (h_sum : sum_of_three a b c)
  (h_ratio_ab : ratio_first_to_second a b)
  (h_ratio_bc : ratio_second_to_third b c) : 
  b = 1440 / 41 := 
sorry

end find_second_number_l200_200095


namespace tree_planting_l200_200300

/-- The city plans to plant 500 thousand trees. The original plan 
was to plant x thousand trees per day. Due to volunteers, the actual number 
of trees planted per day exceeds the original plan by 30%. As a result, 
the task is completed 2 days ahead of schedule. Prove the equation. -/
theorem tree_planting
    (x : ℝ) 
    (hx : x > 0) : 
    (500 / x) - (500 / ((1 + 0.3) * x)) = 2 :=
sorry

end tree_planting_l200_200300


namespace equation1_solution_equation2_solution_l200_200720

theorem equation1_solution (x : ℝ) : (x - 1) ^ 3 = 64 ↔ x = 5 := sorry

theorem equation2_solution (x : ℝ) : 25 * x ^ 2 + 3 = 12 ↔ x = 3 / 5 ∨ x = -3 / 5 := sorry

end equation1_solution_equation2_solution_l200_200720


namespace probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l200_200157

noncomputable def probability_correct_answers : ℚ :=
  let pA := (1/5 : ℚ)
  let pB := (3/5 : ℚ)
  let pC := (1/5 : ℚ)
  ((pA * (3/9 : ℚ) * (2/3)^2 * (1/3)) + (pB * (6/9 : ℚ) * (2/3) * (1/3)^2) + (pC * (1/9 : ℚ) * (1/3)^3))

theorem probability_of_3_correct_answers_is_31_over_135 :
  probability_correct_answers = 31 / 135 := by
  sorry

noncomputable def expected_score : ℚ :=
  let E_m := (1/5 * 1 + 3/5 * 2 + 1/5 * 3 : ℚ)
  let E_n := (3 * (2/3 : ℚ))
  (15 * E_m + 10 * E_n)

theorem expected_value_of_total_score_is_50 :
  expected_score = 50 := by
  sorry

end probability_of_3_correct_answers_is_31_over_135_expected_value_of_total_score_is_50_l200_200157


namespace minimize_rental_cost_l200_200333

def travel_agency (x y : ℕ) : ℕ := 1600 * x + 2400 * y

theorem minimize_rental_cost :
    ∃ (x y : ℕ), (x + y ≤ 21) ∧ (y ≤ x + 7) ∧ (36 * x + 60 * y = 900) ∧ 
    (∀ (a b : ℕ), (a + b ≤ 21) ∧ (b ≤ a + 7) ∧ (36 * a + 60 * b = 900) → travel_agency a b ≥ travel_agency x y) ∧
    travel_agency x y = 36800 :=
sorry

end minimize_rental_cost_l200_200333


namespace grandmother_dolls_l200_200179

-- Define the conditions
variable (S G : ℕ)

-- Rene has three times as many dolls as her sister
def rene_dolls : ℕ := 3 * S

-- The sister has two more dolls than their grandmother
def sister_dolls_eq : Prop := S = G + 2

-- Together they have a total of 258 dolls
def total_dolls : Prop := (rene_dolls S) + S + G = 258

-- Prove that the grandmother has 50 dolls given the conditions
theorem grandmother_dolls : sister_dolls_eq S G → total_dolls S G → G = 50 :=
by
  intros h1 h2
  sorry

end grandmother_dolls_l200_200179


namespace number_of_boys_l200_200815

theorem number_of_boys {total_students : ℕ} (h1 : total_students = 49)
  (ratio_girls_boys : ℕ → ℕ → Prop)
  (h2 : ratio_girls_boys 4 3) :
  ∃ boys : ℕ, boys = 21 := by
  sorry

end number_of_boys_l200_200815


namespace angles_equal_l200_200381

theorem angles_equal (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : Real.sin A = 2 * Real.cos B * Real.sin C) : B = C :=
by sorry

end angles_equal_l200_200381


namespace determine_c_l200_200281

theorem determine_c (c y : ℝ) : (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 := by
  sorry

end determine_c_l200_200281


namespace number_of_remaining_grandchildren_l200_200786

-- Defining the given values and conditions
def total_amount : ℕ := 124600
def half_amount : ℕ := total_amount / 2
def amount_per_remaining_grandchild : ℕ := 6230

-- Defining the goal to prove the number of remaining grandchildren
theorem number_of_remaining_grandchildren : (half_amount / amount_per_remaining_grandchild) = 10 := by
  sorry

end number_of_remaining_grandchildren_l200_200786


namespace remaining_tickets_l200_200213

-- Define initial tickets and used tickets
def initial_tickets := 13
def used_tickets := 6

-- Declare the theorem we want to prove
theorem remaining_tickets (initial_tickets used_tickets : ℕ) (h1 : initial_tickets = 13) (h2 : used_tickets = 6) : initial_tickets - used_tickets = 7 :=
by
  sorry

end remaining_tickets_l200_200213


namespace a3_equals_1_div_12_l200_200612

-- Definition of the sequence
def seq (n : Nat) : Rat :=
  1 / (n * (n + 1))

-- Assertion to be proved
theorem a3_equals_1_div_12 : seq 3 = 1 / 12 := 
sorry

end a3_equals_1_div_12_l200_200612


namespace book_cost_is_2_l200_200519

-- Define initial amount of money
def initial_amount : ℕ := 48

-- Define the number of books purchased
def num_books : ℕ := 5

-- Define the amount of money left after purchasing the books
def amount_left : ℕ := 38

-- Define the cost per book
def cost_per_book (initial amount_left : ℕ) (num_books : ℕ) : ℕ := (initial - amount_left) / num_books

-- The theorem to prove
theorem book_cost_is_2
    (initial_amount : ℕ := 48) 
    (amount_left : ℕ := 38) 
    (num_books : ℕ := 5) :
    cost_per_book initial_amount amount_left num_books = 2 :=
by
  sorry

end book_cost_is_2_l200_200519


namespace parts_processed_per_hour_l200_200816

theorem parts_processed_per_hour (x : ℕ) (y : ℕ) (h1 : y = x + 10) (h2 : 150 / y = 120 / x) :
  x = 40 ∧ y = 50 :=
by {
  sorry
}

end parts_processed_per_hour_l200_200816


namespace quadrilateral_ABCD_is_rectangle_l200_200422

noncomputable def point := (ℤ × ℤ)

def A : point := (-2, 0)
def B : point := (1, 6)
def C : point := (5, 4)
def D : point := (2, -2)

def vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def dot_product (v1 v2 : point) : ℤ := (v1.1 * v2.1) + (v1.2 * v2.2)

def is_perpendicular (v1 v2 : point) : Prop := dot_product v1 v2 = 0

def is_rectangle (A B C D : point) :=
  vector A B = vector C D ∧ is_perpendicular (vector A B) (vector A D)

theorem quadrilateral_ABCD_is_rectangle : is_rectangle A B C D :=
by
  sorry

end quadrilateral_ABCD_is_rectangle_l200_200422


namespace complex_expression_proof_l200_200722

open Complex

theorem complex_expression_proof {x y z : ℂ}
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 18 :=
by
  sorry

end complex_expression_proof_l200_200722


namespace max_value_f_zero_points_range_k_l200_200779

noncomputable def f (x k : ℝ) : ℝ := 3 * x^2 + 2 * (k - 1) * x + (k + 5)

theorem max_value_f (k : ℝ) (h : k < -7/2 ∨ k ≥ -7/2) :
  ∃ max_val : ℝ, max_val = if k < -7/2 then k + 5 else 7 * k + 26 :=
sorry

theorem zero_points_range_k :
  ∀ k : ℝ, (f 0 k) * (f 3 k) ≤ 0 ↔ (-5 ≤ k ∧ k ≤ -2) :=
sorry

end max_value_f_zero_points_range_k_l200_200779


namespace ratio_of_carpets_l200_200149

theorem ratio_of_carpets (h1 h2 h3 h4 : ℕ) (total : ℕ) 
  (H1 : h1 = 12) (H2 : h2 = 20) (H3 : h3 = 10) (H_total : total = 62) 
  (H_all_houses : h1 + h2 + h3 + h4 = total) : h4 / h3 = 2 :=
by
  sorry

end ratio_of_carpets_l200_200149


namespace product_of_fractions_is_3_div_80_l200_200725

def product_fractions (a b c d e f : ℚ) : ℚ := (a / b) * (c / d) * (e / f)

theorem product_of_fractions_is_3_div_80 
  (h₁ : product_fractions 3 8 2 5 1 4 = 3 / 80) : True :=
by
  sorry

end product_of_fractions_is_3_div_80_l200_200725


namespace positive_integer_solution_inequality_l200_200865

theorem positive_integer_solution_inequality (x : ℕ) (h : 2 * (x + 1) ≥ 5 * x - 3) : x = 1 :=
by {
  sorry
}

end positive_integer_solution_inequality_l200_200865


namespace value_of_y_l200_200943

theorem value_of_y (y : ℝ) (h : |y| = |y - 3|) : y = 3 / 2 :=
sorry

end value_of_y_l200_200943


namespace sequence_sum_after_operations_l200_200029

-- Define the initial sequence length
def initial_sequence := [1, 9, 8, 8]

-- Define the sum of initial sequence
def initial_sum := initial_sequence.sum

-- Define the number of operations
def ops := 100

-- Define the increase per operation
def increase_per_op := 7

-- Define the final sum after operations
def final_sum := initial_sum + (increase_per_op * ops)

-- Prove the final sum is 726 after 100 operations
theorem sequence_sum_after_operations : final_sum = 726 := by
  -- Proof omitted as per instructions
  sorry

end sequence_sum_after_operations_l200_200029


namespace calculate_f_at_pi_div_6_l200_200711

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem calculate_f_at_pi_div_6 (ω φ : ℝ) 
  (h : ∀ x : ℝ, f (π / 3 + x) ω φ = f (-x) ω φ) :
  f (π / 6) ω φ = 2 ∨ f (π / 6) ω φ = -2 :=
sorry

end calculate_f_at_pi_div_6_l200_200711


namespace distance_A_B_l200_200112

noncomputable def distance_3d (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_A_B :
  distance_3d 4 1 9 10 (-1) 6 = 7 :=
by
  sorry

end distance_A_B_l200_200112


namespace sine_double_angle_inequality_l200_200212

theorem sine_double_angle_inequality {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 4) : 
  Real.sin (2 * α) < 2 * Real.sin α :=
by
  sorry

end sine_double_angle_inequality_l200_200212


namespace nat_perfect_square_l200_200470

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l200_200470


namespace salary_increase_l200_200094

theorem salary_increase (prev_income : ℝ) (prev_percentage : ℝ) (new_percentage : ℝ) (rent_utilities : ℝ) (new_income : ℝ) :
  prev_income = 1000 ∧ prev_percentage = 0.40 ∧ new_percentage = 0.25 ∧ rent_utilities = prev_percentage * prev_income ∧
  rent_utilities = new_percentage * new_income → new_income - prev_income = 600 :=
by 
  sorry

end salary_increase_l200_200094


namespace seats_selection_l200_200620

theorem seats_selection (n k d : ℕ) (hn : n ≥ 4) (hk : k ≥ 2) (hd : d ≥ 2) (hkd : k * d ≤ n) :
  ∃ ways : ℕ, ways = (n / k) * Nat.choose (n - k * d + k - 1) (k - 1) :=
sorry

end seats_selection_l200_200620


namespace rabbit_count_l200_200539

theorem rabbit_count (r1 r2 : ℕ) (h1 : r1 = 8) (h2 : r2 = 5) : r1 + r2 = 13 := 
by 
  sorry

end rabbit_count_l200_200539


namespace problem_1_problem_2_l200_200344

def f (x : ℝ) : ℝ := |(1 - 2 * x)| - |(1 + x)|

theorem problem_1 :
  {x | f x ≥ 4} = {x | x ≤ -2 ∨ x ≥ 6} :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, a^2 + 2 * a + |(1 + x)| > f x) → (a < -3 ∨ a > 1) :=
sorry

end problem_1_problem_2_l200_200344


namespace roots_equal_and_real_l200_200374

theorem roots_equal_and_real (a c : ℝ) (h : 32 - 4 * a * c = 0) :
  ∃ x : ℝ, x = (2 * Real.sqrt 2) / a := 
by sorry

end roots_equal_and_real_l200_200374


namespace diameter_of_outer_edge_l200_200128

-- Defining the conditions as variables
variable (pathWidth gardenWidth statueDiameter fountainDiameter : ℝ)
variable (hPathWidth : pathWidth = 10)
variable (hGardenWidth : gardenWidth = 12)
variable (hStatueDiameter : statueDiameter = 6)
variable (hFountainDiameter : fountainDiameter = 14)

-- Lean statement to prove the diameter
theorem diameter_of_outer_edge :
  2 * ((fountainDiameter / 2) + gardenWidth + pathWidth) = 58 :=
by
  rw [hPathWidth, hGardenWidth, hFountainDiameter]
  sorry

end diameter_of_outer_edge_l200_200128


namespace solve_for_x_l200_200309

theorem solve_for_x (x : ℝ) (h : (1 / 4) + (5 / x) = (12 / x) + (1 / 15)) : x = 420 / 11 := 
by
  sorry

end solve_for_x_l200_200309


namespace prob_truth_same_time_l200_200342

theorem prob_truth_same_time (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.60) :
  pA * pB = 0.51 :=
by
  rw [hA, hB]
  norm_num

end prob_truth_same_time_l200_200342


namespace remainder_of_product_of_odd_primes_mod_32_l200_200469

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l200_200469


namespace maximum_xyz_l200_200811

theorem maximum_xyz {x y z : ℝ} (hx: 0 < x) (hy: 0 < y) (hz: 0 < z) 
  (h : (x * y) + z = (x + z) * (y + z)) : xyz ≤ (1 / 27) :=
by
  sorry

end maximum_xyz_l200_200811


namespace imo1965_cmo6511_l200_200346

theorem imo1965_cmo6511 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ∧
  |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ≤ Real.sqrt 2 ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
sorry

end imo1965_cmo6511_l200_200346


namespace cube_inequality_l200_200996

theorem cube_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cube_inequality_l200_200996


namespace sum_of_fourth_powers_l200_200075

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := 
by 
  sorry

end sum_of_fourth_powers_l200_200075


namespace number_of_solutions_eq_six_l200_200846

/-- 
The number of ordered pairs (m, n) of positive integers satisfying the equation
6/m + 3/n = 1 is 6.
-/
theorem number_of_solutions_eq_six : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ p ∈ s, (1 < p.1 ∧ 1 < p.2) ∧ 6 / p.1 + 3 / p.2 = 1) ∧ s.card = 6 :=
sorry

end number_of_solutions_eq_six_l200_200846


namespace largest_undefined_x_value_l200_200046

theorem largest_undefined_x_value :
  ∃ x : ℝ, (6 * x^2 - 65 * x + 54 = 0) ∧ (∀ y : ℝ, (6 * y^2 - 65 * y + 54 = 0) → y ≤ x) :=
sorry

end largest_undefined_x_value_l200_200046


namespace find_AC_find_area_l200_200141

theorem find_AC (BC : ℝ) (angleA : ℝ) (cosB : ℝ) 
(hBC : BC = Real.sqrt 7) (hAngleA : angleA = 60) (hCosB : cosB = Real.sqrt 6 / 3) :
  (AC : ℝ) → (hAC : AC = 2 * Real.sqrt 7 / 3) → Prop :=
by
  sorry

theorem find_area (BC AB : ℝ) (angleA : ℝ) 
(hBC : BC = Real.sqrt 7) (hAB : AB = 2) (hAngleA : angleA = 60) :
  (area : ℝ) → (hArea : area = 3 * Real.sqrt 3 / 2) → Prop :=
by
  sorry

end find_AC_find_area_l200_200141


namespace geom_seq_sum_3000_l200_200796

noncomputable
def sum_geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n
  else a * (1 - r ^ n) / (1 - r)

theorem geom_seq_sum_3000 (a r : ℝ) (h1: sum_geom_seq a r 1000 = 300) (h2: sum_geom_seq a r 2000 = 570) :
  sum_geom_seq a r 3000 = 813 :=
sorry

end geom_seq_sum_3000_l200_200796


namespace moles_of_C2H6_formed_l200_200630

-- Definitions of the quantities involved
def moles_H2 : ℕ := 3
def moles_C2H4 : ℕ := 3
def moles_C2H6 : ℕ := 3

-- Stoichiometry condition stated in a way that Lean can understand.
axiom stoichiometry : moles_H2 = moles_C2H4

theorem moles_of_C2H6_formed : moles_C2H6 = 3 :=
by
  -- Assume the constraints and state the final result
  have h : moles_H2 = moles_C2H4 := stoichiometry
  show moles_C2H6 = 3
  sorry

end moles_of_C2H6_formed_l200_200630


namespace song_distribution_l200_200084

-- Let us define the necessary conditions and the result as a Lean statement.

theorem song_distribution :
    ∃ (AB BC CA A B C N : Finset ℕ),
    -- Six different songs.
    (AB ∪ BC ∪ CA ∪ A ∪ B ∪ C ∪ N) = {1, 2, 3, 4, 5, 6} ∧
    -- No song is liked by all three.
    (∀ song, ¬(song ∈ AB ∩ BC ∩ CA)) ∧
    -- Each girl dislikes at least one song.
    (N ≠ ∅) ∧
    -- For each pair of girls, at least one song liked by those two but disliked by the third.
    (AB ≠ ∅ ∧ BC ≠ ∅ ∧ CA ≠ ∅) ∧
    -- The total number of ways this can be done is 735.
    True := sorry

end song_distribution_l200_200084


namespace average_of_last_six_l200_200040

theorem average_of_last_six (avg_13 : ℕ → ℝ) (avg_first_6 : ℕ → ℝ) (middle_number : ℕ → ℝ) :
  (∀ n, avg_13 n = 9) →
  (∀ n, n ≤ 6 → avg_first_6 n = 5) →
  (middle_number 7 = 45) →
  ∃ (A : ℝ), (∀ n, n > 6 → n < 13 → avg_13 n = A) ∧ A = 7 :=
by
  sorry

end average_of_last_six_l200_200040


namespace sum_of_first_17_terms_l200_200412

variable {α : Type*} [LinearOrderedField α] 

-- conditions
def arithmetic_sequence (a : ℕ → α) : Prop := 
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

variable {a : ℕ → α}
variable {S : ℕ → α}

-- main theorem
theorem sum_of_first_17_terms (h_arith : arithmetic_sequence a)
  (h_S : sum_of_first_n_terms a S)
  (h_condition : a 7 + a 12 = 12 - a 8) :
  S 17 = 68 := sorry

end sum_of_first_17_terms_l200_200412


namespace symmetric_line_eq_l200_200221

theorem symmetric_line_eq (x y : ℝ) (c : ℝ) (P : ℝ × ℝ)
  (h₁ : 3 * x - y - 4 = 0)
  (h₂ : P = (2, -1))
  (h₃ : 3 * x - y + c = 0)
  (h : 3 * 2 - (-1) + c = 0) : 
  c = -7 :=
by
  sorry

end symmetric_line_eq_l200_200221


namespace polygon_problem_l200_200581

theorem polygon_problem
  (sum_interior_angles : ℕ → ℝ)
  (sum_exterior_angles : ℝ)
  (condition : ∀ n, sum_interior_angles n = (3 * sum_exterior_angles) - 180) :
  (∃ n : ℕ, sum_interior_angles n = 180 * (n - 2) ∧ n = 7) ∧
  (∃ n : ℕ, n = 7 → (n * (n - 3) / 2) = 14) :=
by
  sorry

end polygon_problem_l200_200581


namespace eccentricity_of_ellipse_l200_200402

open Real

def ellipse_eq (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def foci_dist_eq (a c : ℝ) : Prop :=
  2 * c / (2 * a) = sqrt 6 / 2

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_ellipse (a b x y c : ℝ)
  (h1 : ellipse_eq a b x y)
  (h2 : foci_dist_eq a c) :
  eccentricity c a = sqrt 6 / 3 :=
sorry

end eccentricity_of_ellipse_l200_200402


namespace sale_in_second_month_l200_200762

theorem sale_in_second_month
  (sale1 sale3 sale4 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (total_months : ℕ)
  (h_sale1 : sale1 = 5420)
  (h_sale3 : sale3 = 6200)
  (h_sale4 : sale4 = 6350)
  (h_sale5 : sale5 = 6500)
  (h_sale6 : sale6 = 6470)
  (h_average_sale : average_sale = 6100)
  (h_total_months : total_months = 6) :
  ∃ sale2 : ℕ, sale2 = 5660 := 
by
  sorry

end sale_in_second_month_l200_200762


namespace inversions_range_l200_200895

/-- Given any permutation of 10 elements, 
    the number of inversions (or disorders) in the permutation 
    can take any value from 0 to 45.
-/
theorem inversions_range (perm : List ℕ) (h_length : perm.length = 10):
  ∃ S, 0 ≤ S ∧ S ≤ 45 :=
sorry

end inversions_range_l200_200895


namespace totalNumberOfCrayons_l200_200375

def numOrangeCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numBlueCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numRedCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

theorem totalNumberOfCrayons :
  numOrangeCrayons 6 8 + numBlueCrayons 7 5 + numRedCrayons 1 11 = 94 :=
by
  sorry

end totalNumberOfCrayons_l200_200375


namespace min_value_of_sum_l200_200177

theorem min_value_of_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x * y + 2 * x + y = 4) : x + y ≥ 2 * Real.sqrt 6 - 3 :=
sorry

end min_value_of_sum_l200_200177


namespace evaluate_nested_operation_l200_200603

def operation (a b c : ℕ) : ℕ := (a + b) / c

theorem evaluate_nested_operation : operation (operation 72 36 108) (operation 4 2 6) (operation 12 6 18) = 2 := by
  -- Here we assume all operations are valid (c ≠ 0 for each case)
  sorry

end evaluate_nested_operation_l200_200603


namespace museum_revenue_l200_200126

theorem museum_revenue (V : ℕ) (H : V = 500)
  (R : ℕ) (H_R : R = 60 * V / 100)
  (C_p : ℕ) (H_C_p : C_p = 40 * R / 100)
  (S_p : ℕ) (H_S_p : S_p = 30 * R / 100)
  (A_p : ℕ) (H_A_p : A_p = 30 * R / 100)
  (C_t S_t A_t : ℕ) (H_C_t : C_t = 4) (H_S_t : S_t = 6) (H_A_t : A_t = 12) :
  C_p * C_t + S_p * S_t + A_p * A_t = 2100 :=
by 
  sorry

end museum_revenue_l200_200126


namespace SomeAthletesNotHonorSociety_l200_200006

variable (Athletes HonorSociety : Type)
variable (Discipline : Athletes → Prop)
variable (isMember : Athletes → HonorSociety → Prop)

-- Some athletes are not disciplined
axiom AthletesNotDisciplined : ∃ a : Athletes, ¬Discipline a

-- All members of the honor society are disciplined
axiom AllHonorSocietyDisciplined : ∀ h : HonorSociety, ∀ a : Athletes, isMember a h → Discipline a

-- The theorem to be proved
theorem SomeAthletesNotHonorSociety : ∃ a : Athletes, ∀ h : HonorSociety, ¬isMember a h :=
  sorry

end SomeAthletesNotHonorSociety_l200_200006


namespace slope_of_line_through_focus_of_parabola_l200_200766

theorem slope_of_line_through_focus_of_parabola
  (C : (x y : ℝ) → y^2 = 4 * x)
  (F : (ℝ × ℝ) := (1, 0))
  (A B : (ℝ × ℝ))
  (l : ℝ → ℝ)
  (intersects : (x : ℝ) → (l x) ^ 2 = 4 * x)
  (passes_through_focus : l 1 = 0)
  (distance_condition : ∀ (d1 d2 : ℝ), d1 = 4 * d2 → dist F A = d1 ∧ dist F B = d2) :
  ∃ k : ℝ, (∀ (x : ℝ), l x = k * (x - 1)) ∧ (k = 4 / 3 ∨ k = -4 / 3) :=
by
  sorry

end slope_of_line_through_focus_of_parabola_l200_200766


namespace relationship_of_points_l200_200797

variable (y k b x : ℝ)
variable (y1 y2 : ℝ)

noncomputable def linear_func (x : ℝ) : ℝ := k * x - b

theorem relationship_of_points
  (h_pos_k : k > 0)
  (h_point1 : linear_func k b (-1) = y1)
  (h_point2 : linear_func k b 2 = y2):
  y1 < y2 := 
sorry

end relationship_of_points_l200_200797


namespace solve_quadratic_l200_200788

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
by
  sorry

end solve_quadratic_l200_200788


namespace terminal_side_in_third_quadrant_l200_200713

open Real

theorem terminal_side_in_third_quadrant (θ : ℝ) (h1 : sin θ < 0) (h2 : cos θ < 0) : 
    θ ∈ Set.Ioo (π : ℝ) (3 * π / 2) := 
sorry

end terminal_side_in_third_quadrant_l200_200713


namespace calculate_negative_subtraction_l200_200438

theorem calculate_negative_subtraction : -2 - (-3) = 1 :=
by sorry

end calculate_negative_subtraction_l200_200438


namespace intersection_union_complement_l200_200003

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def universal_set := U = univ
def set_A := A = {x : ℝ | -1 ≤ x ∧ x < 2}
def set_B := B = {x : ℝ | 1 < x ∧ x ≤ 3}

theorem intersection (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := sorry

theorem union (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) (hB : B = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := sorry

theorem complement (hU : U = univ) (hA : A = {x : ℝ | -1 ≤ x ∧ x < 2}) :
  U \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := sorry

end intersection_union_complement_l200_200003


namespace three_digit_integers_product_30_l200_200005

theorem three_digit_integers_product_30 : 
  ∃ (n : ℕ), 
    (100 ≤ n ∧ n < 1000) ∧ 
    (∀ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 → 
    (1 ≤ d1 ∧ d1 ≤ 9) ∧ 
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    d1 * d2 * d3 = 30) ∧ 
    n = 12 :=
sorry

end three_digit_integers_product_30_l200_200005


namespace evaluate_fraction_expression_l200_200048

theorem evaluate_fraction_expression :
  ( (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) ) = 2 / 5 :=
by
  sorry

end evaluate_fraction_expression_l200_200048


namespace value_of_expression_l200_200637

-- Given conditions
variable (n : ℤ)
def m : ℤ := 4 * n + 3

-- Main theorem statement
theorem value_of_expression (n : ℤ) : 
  (m n)^2 - 8 * (m n) * n + 16 * n^2 = 9 := 
  sorry

end value_of_expression_l200_200637


namespace rachel_wrote_six_pages_l200_200178

theorem rachel_wrote_six_pages
  (write_rate : ℕ)
  (research_time : ℕ)
  (editing_time : ℕ)
  (total_time : ℕ)
  (total_time_in_minutes : ℕ := total_time * 60)
  (actual_time_writing : ℕ := total_time_in_minutes - (research_time + editing_time))
  (pages_written : ℕ := actual_time_writing / write_rate) :
  write_rate = 30 →
  research_time = 45 →
  editing_time = 75 →
  total_time = 5 →
  pages_written = 6 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  subst h4
  have h5 : total_time_in_minutes = 300 := by sorry
  have h6 : actual_time_writing = 180 := by sorry
  have h7 : pages_written = 6 := by sorry
  exact h7

end rachel_wrote_six_pages_l200_200178


namespace salary_of_A_l200_200056

theorem salary_of_A (x y : ℝ) (h₁ : x + y = 4000) (h₂ : 0.05 * x = 0.15 * y) : x = 3000 :=
by {
    sorry
}

end salary_of_A_l200_200056


namespace problem_statement_l200_200190

open Complex

noncomputable def a : ℂ := 5 - 3 * I
noncomputable def b : ℂ := 2 + 4 * I

theorem problem_statement : 3 * a - 4 * b = 7 - 25 * I :=
by { sorry }

end problem_statement_l200_200190


namespace smallest_m_exists_l200_200580

theorem smallest_m_exists :
  ∃ (m : ℕ), 0 < m ∧ (∃ k : ℕ, 5 * m = k^2) ∧ (∃ l : ℕ, 3 * m = l^3) ∧ m = 243 :=
by
  sorry

end smallest_m_exists_l200_200580


namespace union_sets_l200_200196

def A := { x : ℝ | x^2 ≤ 1 }
def B := { x : ℝ | 0 < x }

theorem union_sets : A ∪ B = { x | -1 ≤ x } :=
by {
  sorry -- Proof is omitted as per the instructions
}

end union_sets_l200_200196


namespace find_n_if_roots_opposite_signs_l200_200693

theorem find_n_if_roots_opposite_signs :
  ∃ n : ℝ, (∀ x : ℝ, (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) → x = -x) →
    (n = (-1 + Real.sqrt 5) / 2 ∨ n = (-1 - Real.sqrt 5) / 2) :=
by
  sorry

end find_n_if_roots_opposite_signs_l200_200693


namespace trapezoid_inequality_l200_200107

theorem trapezoid_inequality (a b R : ℝ) (h : a > 0) (h1 : b > 0) (h2 : R > 0) 
  (circumscribed : ∃ (x y : ℝ), x + y = a ∧ R^2 * (1/x + 1/y) = b) : 
  a * b ≥ 4 * R^2 :=
by
  sorry

end trapezoid_inequality_l200_200107


namespace isabel_pop_albums_l200_200650

theorem isabel_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) (pop_albums : ℕ)
  (h1 : total_songs = 72)
  (h2 : country_albums = 4)
  (h3 : songs_per_album = 8)
  (h4 : total_songs - country_albums * songs_per_album = pop_albums * songs_per_album) :
  pop_albums = 5 :=
by
  sorry

end isabel_pop_albums_l200_200650


namespace unique_four_digit_square_l200_200379

theorem unique_four_digit_square (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 10 = (n / 10) % 10) ∧ 
  ((n / 100) % 10 = (n / 1000) % 10) ∧ 
  (∃ k : ℕ, n = k^2) ↔ n = 7744 := 
by 
  sorry

end unique_four_digit_square_l200_200379


namespace B_completes_work_in_n_days_l200_200494

-- Define the conditions
def can_complete_work_A_in_d_days (d : ℕ) : Prop := d = 15
def fraction_of_work_left_after_working_together (t : ℕ) (fraction : ℝ) : Prop :=
  t = 5 ∧ fraction = 0.41666666666666663

-- Define the theorem to be proven
theorem B_completes_work_in_n_days (d t : ℕ) (fraction : ℝ) (x : ℕ) 
  (hA : can_complete_work_A_in_d_days d) 
  (hB : fraction_of_work_left_after_working_together t fraction) : x = 20 :=
sorry

end B_completes_work_in_n_days_l200_200494


namespace tanvi_min_candies_l200_200685

theorem tanvi_min_candies : 
  ∃ c : ℕ, 
  (c % 6 = 5) ∧ 
  (c % 8 = 7) ∧ 
  (c % 9 = 6) ∧ 
  (c % 11 = 0) ∧ 
  (∀ d : ℕ, 
    (d % 6 = 5) ∧ 
    (d % 8 = 7) ∧ 
    (d % 9 = 6) ∧ 
    (d % 11 = 0) → 
    c ≤ d) → 
  c = 359 :=
by sorry

end tanvi_min_candies_l200_200685


namespace quadratic_has_distinct_real_roots_find_k_l200_200544

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (k : ℝ) : 
  let a := 1
  let b := 2 * k - 1
  let c := -k - 2
  let Δ := b^2 - 4 * a * c
  (Δ > 0) :=
by
  sorry

-- Part 2: Given the roots condition, find k
theorem find_k (x1 x2 k : ℝ)
  (h1 : x1 + x2 = -(2 * k - 1))
  (h2 : x1 * x2 = -k - 2)
  (h3 : x1 + x2 - 4 * x1 * x2 = 1) : 
  k = -4 :=
by
  sorry

end quadratic_has_distinct_real_roots_find_k_l200_200544


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l200_200106

open Set -- Open the Set namespace for convenience

-- Define the universal set U, and sets A and B
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof statements
theorem complement_U_A : U \ A = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 3} :=
by sorry

theorem complement_U_intersection_A_B : U \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2} :=
by sorry

theorem complement_A_intersection_B : (U \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} :=
by sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_A_intersection_B_l200_200106


namespace eight_points_in_circle_l200_200111

theorem eight_points_in_circle :
  ∀ (P : Fin 8 → ℝ × ℝ), 
  (∀ i, (P i).1^2 + (P i).2^2 ≤ 1) → 
  ∃ (i j : Fin 8), i ≠ j ∧ ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2 < 1 :=
by
  sorry

end eight_points_in_circle_l200_200111


namespace area_triangle_ABC_l200_200596

-- Definitions of the lengths and height
def BD : ℝ := 3
def DC : ℝ := 2 * BD
def BC : ℝ := BD + DC
def h_A_BC : ℝ := 4

-- The triangle area formula
def areaOfTriangle (base height : ℝ) : ℝ := 0.5 * base * height

-- The goal to prove that the area of triangle ABC is 18 square units
theorem area_triangle_ABC : areaOfTriangle BC h_A_BC = 18 := by
  sorry

end area_triangle_ABC_l200_200596


namespace mike_cards_remaining_l200_200529

-- Define initial condition
def mike_initial_cards : ℕ := 87

-- Define the cards bought by Sam
def sam_bought_cards : ℕ := 13

-- Define the expected remaining cards
def mike_final_cards := mike_initial_cards - sam_bought_cards

-- Theorem to prove the final count of Mike's baseball cards
theorem mike_cards_remaining : mike_final_cards = 74 := by
  sorry

end mike_cards_remaining_l200_200529


namespace bus_trip_times_l200_200522

/-- Given two buses traveling towards each other from points A and B which are 120 km apart.
The first bus stops for 10 minutes and the second bus stops for 5 minutes. The first bus reaches 
its destination 25 minutes before the second bus. The first bus travels 20 km/h faster than the 
second bus. Prove that the travel times for the buses are 
1 hour 40 minutes and 2 hours 5 minutes respectively. -/
theorem bus_trip_times (d : ℕ) (v1 v2 : ℝ) (t1 t2 t : ℝ) (h1 : d = 120) (h2 : v1 = v2 + 20) 
(h3 : t1 = d / v1 + 10) (h4 : t2 = d / v2 + 5) (h5 : t2 - t1 = 25) :
t1 = 100 ∧ t2 = 125 := 
by 
  sorry

end bus_trip_times_l200_200522


namespace find_f_condition_l200_200060

theorem find_f_condition {f : ℂ → ℂ} (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) :
  ∀ z : ℂ, f z = 1 :=
by
  sorry

end find_f_condition_l200_200060


namespace part1_solution_set_l200_200454

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2 * x|

theorem part1_solution_set : {x : ℝ | f x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by sorry

end part1_solution_set_l200_200454


namespace kanul_spent_on_machinery_l200_200031

theorem kanul_spent_on_machinery (total raw_materials cash M : ℝ) 
  (h_total : total = 7428.57) 
  (h_raw_materials : raw_materials = 5000) 
  (h_cash : cash = 0.30 * total) 
  (h_expenditure : total = raw_materials + M + cash) :
  M = 200 := 
by
  sorry

end kanul_spent_on_machinery_l200_200031


namespace sufficient_but_not_necessary_condition_l200_200321

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (floor x = floor y) → (abs (x - y) < 1) ∧ (¬ (abs (x - y) < 1) → (floor x ≠ floor y)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l200_200321


namespace smallest_root_of_quadratic_l200_200201

theorem smallest_root_of_quadratic :
  ∃ x : ℝ, (12 * x^2 - 50 * x + 48 = 0) ∧ x = 1.333 := 
sorry

end smallest_root_of_quadratic_l200_200201


namespace repeating_decimal_to_fraction_l200_200876

theorem repeating_decimal_to_fraction :
  7.4646464646 = (739 / 99) :=
  sorry

end repeating_decimal_to_fraction_l200_200876


namespace greatest_possible_perimeter_l200_200761

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l200_200761


namespace system_of_equations_inconsistent_l200_200724

theorem system_of_equations_inconsistent :
  ¬∃ (x1 x2 x3 x4 x5 : ℝ), 
    (x1 + 2 * x2 - x3 + 3 * x4 - x5 = 0) ∧ 
    (2 * x1 - x2 + 3 * x3 + x4 - x5 = -1) ∧
    (x1 - x2 + x3 + 2 * x4 = 2) ∧
    (4 * x1 + 3 * x3 + 6 * x4 - 2 * x5 = 5) := 
sorry

end system_of_equations_inconsistent_l200_200724


namespace max_value_f_at_a1_f_div_x_condition_l200_200524

noncomputable def f (a x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem max_value_f_at_a1 :
  ∀ x : ℝ, (f 1 0) = 0 ∧ ( ∀ y : ℝ, y ≠ 0 → f 1 y < f 1 0) := 
sorry

theorem f_div_x_condition :
  ∀ x : ℝ, x ≠ 0 → (((f 1 x) / x) < 1) :=
sorry

end max_value_f_at_a1_f_div_x_condition_l200_200524


namespace taxi_fare_l200_200969

theorem taxi_fare :
  ∀ (initial_fee rate_per_increment increment_distance total_distance : ℝ),
    initial_fee = 2.35 →
    rate_per_increment = 0.35 →
    increment_distance = (2 / 5) →
    total_distance = 3.6 →
    (initial_fee + rate_per_increment * (total_distance / increment_distance)) = 5.50 :=
by
  intros initial_fee rate_per_increment increment_distance total_distance
  intro h1 h2 h3 h4
  sorry -- Proof is not required.

end taxi_fare_l200_200969


namespace min_value_frac_add_x_l200_200573

theorem min_value_frac_add_x (x : ℝ) (h : x > 3) : (∃ m, (∀ (y : ℝ), y > 3 → (4 / y - 3 + y) ≥ m) ∧ m = 7) :=
sorry

end min_value_frac_add_x_l200_200573


namespace andrew_age_l200_200677

theorem andrew_age 
  (g a : ℚ)
  (h1: g = 16 * a)
  (h2: g - 20 - (a - 20) = 45) : 
 a = 17 / 3 := by 
  sorry

end andrew_age_l200_200677


namespace find_a_l200_200304

noncomputable def p (a : ℝ) : Prop := 3 < a ∧ a < 7/2
noncomputable def q (a : ℝ) : Prop := a > 3 ∧ a ≠ 7/2
theorem find_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) (hpq : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : a > 7/2 :=
sorry

end find_a_l200_200304


namespace greatest_number_of_fruit_baskets_l200_200356

def number_of_oranges : ℕ := 18
def number_of_pears : ℕ := 27
def number_of_bananas : ℕ := 12

theorem greatest_number_of_fruit_baskets :
  Nat.gcd (Nat.gcd number_of_oranges number_of_pears) number_of_bananas = 3 :=
by
  sorry

end greatest_number_of_fruit_baskets_l200_200356


namespace integer_multiplication_l200_200675

theorem integer_multiplication :
  ∃ A : ℤ, (999999999 : ℤ) * A = (111111111 : ℤ) :=
by {
  sorry
}

end integer_multiplication_l200_200675


namespace charlie_third_week_data_l200_200132

theorem charlie_third_week_data (d3 : ℕ) : 
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  overage_GB = total_extra_GB -> d3 = 5 := 
by
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  have : overage_GB = total_extra_GB := sorry
  have : d3 = 5 := sorry
  sorry

end charlie_third_week_data_l200_200132


namespace find_n_l200_200628

-- Define the polynomial function
def polynomial (n : ℤ) : ℤ :=
  n^4 + 2 * n^3 + 6 * n^2 + 12 * n + 25

-- Define the condition that n is a positive integer
def is_positive_integer (n : ℤ) : Prop :=
  n > 0

-- Define the condition that polynomial is a perfect square
def is_perfect_square (k : ℤ) : Prop :=
  ∃ m : ℤ, m^2 = k

-- The theorem we need to prove
theorem find_n (n : ℤ) (h1 : is_positive_integer n) (h2 : is_perfect_square (polynomial n)) : n = 8 :=
sorry

end find_n_l200_200628


namespace point_in_fourth_quadrant_l200_200980

def point : ℝ × ℝ := (3, -2)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l200_200980


namespace envelope_width_l200_200180

theorem envelope_width (Area Height Width : ℝ) (h_area : Area = 36) (h_height : Height = 6) (h_area_formula : Area = Width * Height) : Width = 6 :=
by
  sorry

end envelope_width_l200_200180


namespace c_minus_b_seven_l200_200403

theorem c_minus_b_seven {a b c d : ℕ} (ha : a^6 = b^5) (hb : c^4 = d^3) (hc : c - a = 31) : c - b = 7 :=
sorry

end c_minus_b_seven_l200_200403


namespace equal_real_roots_possible_values_l200_200616

theorem equal_real_roots_possible_values (a : ℝ): 
  (∀ x : ℝ, x^2 + a * x + 1 = 0) → (a = 2 ∨ a = -2) :=
by
  sorry

end equal_real_roots_possible_values_l200_200616


namespace average_percentage_popped_average_percentage_kernels_l200_200322

theorem average_percentage_popped (
  pops1 total1 pops2 total2 pops3 total3 : ℕ
) (h1 : pops1 = 60) (h2 : total1 = 75) 
  (h3 : pops2 = 42) (h4 : total2 = 50) 
  (h5 : pops3 = 82) (h6 : total3 = 100) : 
  ((pops1 : ℝ) / total1) * 100 + ((pops2 : ℝ) / total2) * 100 + ((pops3 : ℝ) / total3) * 100 = 246 := 
by
  sorry

theorem average_percentage_kernels (pops1 total1 pops2 total2 pops3 total3 : ℕ)
  (h1 : pops1 = 60) (h2 : total1 = 75)
  (h3 : pops2 = 42) (h4 : total2 = 50)
  (h5 : pops3 = 82) (h6 : total3 = 100) :
  ((
      (((pops1 : ℝ) / total1) * 100) + 
       (((pops2 : ℝ) / total2) * 100) + 
       (((pops3 : ℝ) / total3) * 100)
    ) / 3 = 82) :=
by
  sorry

end average_percentage_popped_average_percentage_kernels_l200_200322


namespace first_diamond_second_spade_prob_l200_200760

/--
Given a standard deck of 52 cards, there are 13 cards of each suit.
What is the probability that the first card dealt is a diamond (♦) 
and the second card dealt is a spade (♠)?
-/
theorem first_diamond_second_spade_prob : 
  let total_cards := 52
  let diamonds := 13
  let spades := 13
  let first_diamond_prob := diamonds / total_cards
  let second_spade_prob_after_diamond := spades / (total_cards - 1)
  let combined_prob := first_diamond_prob * second_spade_prob_after_diamond
  combined_prob = 13 / 204 := 
by
  sorry

end first_diamond_second_spade_prob_l200_200760


namespace cherries_count_l200_200409

theorem cherries_count (b s r c : ℝ) 
  (h1 : b + s + r + c = 360)
  (h2 : s = 2 * b)
  (h3 : r = 4 * s)
  (h4 : c = 2 * r) : 
  c = 640 / 3 :=
by 
  sorry

end cherries_count_l200_200409


namespace sum_of_cube_faces_l200_200847

theorem sum_of_cube_faces (a b c d e f : ℕ) (h1 : a % 2 = 0) (h2 : b = a + 2) (h3 : c = b + 2) (h4 : d = c + 2) (h5 : e = d + 2) (h6 : f = e + 2)
(h_pairs : (a + f + 2) = (b + e + 2) ∧ (b + e + 2) = (c + d + 2)) :
  a + b + c + d + e + f = 90 :=
  sorry

end sum_of_cube_faces_l200_200847


namespace transformation_power_of_two_l200_200502

theorem transformation_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ s : ℕ, 2 ^ s ≥ n :=
by sorry

end transformation_power_of_two_l200_200502


namespace trig_identity_l200_200480

noncomputable def trig_expr := 
  4.34 * (Real.cos (28 * Real.pi / 180) * Real.cos (56 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) + 
  (Real.cos (2 * Real.pi / 180) * Real.cos (4 * Real.pi / 180) / Real.sin (28 * Real.pi / 180))

theorem trig_identity : 
  trig_expr = (Real.sqrt 3 * Real.sin (38 * Real.pi / 180)) / (4 * Real.sin (2 * Real.pi / 180) * Real.sin (28 * Real.pi / 180)) :=
by 
  sorry

end trig_identity_l200_200480


namespace abs_neg_seven_l200_200598

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end abs_neg_seven_l200_200598


namespace find_digit_l200_200674

theorem find_digit (a : ℕ) (n1 n2 n3 : ℕ) (h1 : n1 = a * 1000) (h2 : n2 = a * 1000 + 998) (h3 : n3 = a * 1000 + 999) (h4 : n1 + n2 + n3 = 22997) :
  a = 7 :=
by
  sorry

end find_digit_l200_200674


namespace proof_value_of_expression_l200_200508

theorem proof_value_of_expression (a b c d m : ℝ) 
  (h1: a + b = 0)
  (h2: c * d = 1)
  (h3: |m| = 4) : 
  m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end proof_value_of_expression_l200_200508


namespace ricciana_jump_distance_l200_200819

theorem ricciana_jump_distance (R : ℕ) :
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1
  Total_distance_Margarita = Total_distance_Ricciana → R = 22 :=
by
  -- Definitions
  let Ricciana_run := 20
  let Margarita_run := 18
  let Margarita_jump := 2 * R - 1
  let Total_distance_Margarita := Margarita_run + Margarita_jump
  let Total_distance_Ricciana := Ricciana_run + R + 1

  -- Given condition
  intro h
  sorry

end ricciana_jump_distance_l200_200819


namespace molecular_weight_of_10_moles_l200_200967

-- Define the molecular weight of a compound as a constant
def molecular_weight (compound : Type) : ℝ := 840

-- Prove that the molecular weight of 10 moles of the compound is the same as the molecular weight of 1 mole of the compound
theorem molecular_weight_of_10_moles (compound : Type) :
  molecular_weight compound = 840 :=
by
  -- Proof
  sorry

end molecular_weight_of_10_moles_l200_200967


namespace cost_of_magazine_l200_200918

theorem cost_of_magazine (B M : ℝ) 
  (h1 : 2 * B + 2 * M = 26) 
  (h2 : B + 3 * M = 27) : 
  M = 7 := 
by 
  sorry

end cost_of_magazine_l200_200918


namespace remaining_volume_of_cube_with_hole_l200_200929

theorem remaining_volume_of_cube_with_hole : 
  let side_length_cube := 8 
  let side_length_hole := 4 
  let volume_cube := side_length_cube ^ 3 
  let cross_section_hole := side_length_hole ^ 2
  let volume_hole := cross_section_hole * side_length_cube
  let remaining_volume := volume_cube - volume_hole
  remaining_volume = 384 := by {
    sorry
  }

end remaining_volume_of_cube_with_hole_l200_200929


namespace average_income_P_Q_l200_200103

   variable (P Q R : ℝ)

   theorem average_income_P_Q
     (h1 : (Q + R) / 2 = 6250)
     (h2 : (P + R) / 2 = 5200)
     (h3 : P = 4000) :
     (P + Q) / 2 = 5050 := by
   sorry
   
end average_income_P_Q_l200_200103


namespace relationship_f_minus_a2_f_minus_1_l200_200138

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement translation
theorem relationship_f_minus_a2_f_minus_1 (a : ℝ) : f (-a^2) ≤ f (-1) := 
sorry

end relationship_f_minus_a2_f_minus_1_l200_200138


namespace ratio_of_p_to_q_l200_200146

theorem ratio_of_p_to_q (p q r : ℚ) (h1: p = r * q) (h2: 18 / 7 + (2 * q - p) / (2 * q + p) = 3) : r = 29 / 10 :=
by
  sorry

end ratio_of_p_to_q_l200_200146


namespace cobbler_hours_per_day_l200_200340

-- Defining some conditions based on our problem statement
def cobbler_rate : ℕ := 3  -- pairs of shoes per hour
def friday_hours : ℕ := 3  -- number of hours worked on Friday
def friday_pairs : ℕ := cobbler_rate * friday_hours  -- pairs mended on Friday
def weekly_pairs : ℕ := 105  -- total pairs mended in a week
def mon_thu_pairs : ℕ := weekly_pairs - friday_pairs  -- pairs mended from Monday to Thursday
def mon_thu_hours : ℕ := mon_thu_pairs / cobbler_rate  -- total hours worked from Monday to Thursday

-- Thm statement: If a cobbler works h hours daily from Mon to Thu, then h = 8 implies total = 105 pairs
theorem cobbler_hours_per_day (h : ℕ) : (4 * h = mon_thu_hours) ↔ (h = 8) :=
by
  sorry

end cobbler_hours_per_day_l200_200340


namespace fraction_zero_implies_x_is_minus_one_l200_200951

variable (x : ℝ)

theorem fraction_zero_implies_x_is_minus_one (h : (x^2 - 1) / (1 - x) = 0) : x = -1 :=
sorry

end fraction_zero_implies_x_is_minus_one_l200_200951


namespace distance_between_red_lights_l200_200809

def position_of_nth_red (n : ℕ) : ℕ :=
  7 * (n - 1) / 3 + n

def in_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem distance_between_red_lights :
  in_feet ((position_of_nth_red 30 - position_of_nth_red 5) * 8) = 41 :=
by
  sorry

end distance_between_red_lights_l200_200809


namespace min_f_l200_200780

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x then (x + 1) * Real.log x
else 2 * x + 3

noncomputable def f' (x : ℝ) : ℝ :=
if 0 < x then Real.log x + (x + 1) / x
else 2

theorem min_f'_for_x_pos : ∃ (c : ℝ), c = 2 ∧ ∀ x > 0, f' x ≥ c := 
  sorry

end min_f_l200_200780


namespace determine_b_l200_200181

theorem determine_b (a b c y1 y2 : ℝ) 
  (h1 : y1 = a * 2^2 + b * 2 + c)
  (h2 : y2 = a * (-2)^2 + b * (-2) + c)
  (h3 : y1 - y2 = -12) : 
  b = -3 := 
by
  sorry

end determine_b_l200_200181


namespace find_xyz_l200_200089

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := 
by 
  sorry

end find_xyz_l200_200089


namespace sum_of_a5_a6_l200_200152

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

noncomputable def geometric_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
a 1 + a 2 = 1 ∧ a 3 + a 4 = 4 ∧ q^2 = 4

theorem sum_of_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) (h_cond : geometric_conditions a q) :
  a 5 + a 6 = 16 :=
sorry

end sum_of_a5_a6_l200_200152


namespace price_after_discounts_l200_200515

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * (1 - 0.10)
  let second_discount := first_discount * (1 - 0.20)
  second_discount

theorem price_after_discounts (initial_price : ℝ) (h : final_price initial_price = 174.99999999999997) : 
  final_price initial_price = 175 := 
by {
  sorry
}

end price_after_discounts_l200_200515


namespace original_price_of_trouser_l200_200706

theorem original_price_of_trouser (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 30) (h2 : discount = 0.70) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l200_200706


namespace clusters_of_oats_l200_200182

-- Define conditions:
def clusters_per_spoonful : Nat := 4
def spoonfuls_per_bowl : Nat := 25
def bowls_per_box : Nat := 5

-- Define the question and correct answer:
def clusters_per_box : Nat :=
  clusters_per_spoonful * spoonfuls_per_bowl * bowls_per_box

-- Theorem statement for the proof problem:
theorem clusters_of_oats:
  clusters_per_box = 500 :=
by
  sorry

end clusters_of_oats_l200_200182


namespace graphs_intersect_at_one_point_l200_200830

theorem graphs_intersect_at_one_point (a : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 3 * x + 1 = -x - 1) ↔ a = 2) :=
by
  sorry

end graphs_intersect_at_one_point_l200_200830


namespace num_true_propositions_l200_200170

theorem num_true_propositions (x : ℝ) :
  (∀ x, x > -3 → x > -6) ∧
  (∀ x, x > -6 → x > -3 = false) ∧
  (∀ x, x ≤ -3 → x ≤ -6 = false) ∧
  (∀ x, x ≤ -6 → x ≤ -3) →
  2 = 2 :=
by
  sorry

end num_true_propositions_l200_200170


namespace correct_calculation_l200_200231

variable (a b : ℝ)

theorem correct_calculation : ((-a^2)^3 = -a^6) :=
by sorry

end correct_calculation_l200_200231


namespace sum_of_fractions_equals_l200_200802

theorem sum_of_fractions_equals :
  (1 / 15 + 2 / 25 + 3 / 35 + 4 / 45 : ℚ) = 0.32127 :=
  sorry

end sum_of_fractions_equals_l200_200802


namespace inequality_abc_l200_200705

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_abc :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) :=
  sorry

end inequality_abc_l200_200705


namespace sum_of_roots_of_quadratic_l200_200501

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : 2 * (X^2) - 8 * X + 6 = 0) : 
  (-b / a) = 4 :=
sorry

end sum_of_roots_of_quadratic_l200_200501


namespace tensor_op_correct_l200_200144

-- Define the operation ⊗
def tensor_op (x y : ℝ) : ℝ := x^2 + y

-- Goal: Prove h ⊗ (h ⊗ h) = 2h^2 + h for some h in ℝ
theorem tensor_op_correct (h : ℝ) : tensor_op h (tensor_op h h) = 2 * h^2 + h :=
by
  sorry

end tensor_op_correct_l200_200144


namespace systematic_sampling_eighth_group_l200_200559

theorem systematic_sampling_eighth_group (total_students : ℕ) (groups : ℕ) (group_size : ℕ)
(start_number : ℕ) (group_number : ℕ)
(h1 : total_students = 480)
(h2 : groups = 30)
(h3 : group_size = 16)
(h4 : start_number = 5)
(h5 : group_number = 8) :
  (group_number - 1) * group_size + start_number = 117 := by
  sorry

end systematic_sampling_eighth_group_l200_200559


namespace average_30_matches_is_25_l200_200274

noncomputable def average_runs_in_30_matches (average_20_matches average_10_matches : ℝ) (total_matches_20 total_matches_10 : ℕ) : ℝ :=
  let total_runs_20 := total_matches_20 * average_20_matches
  let total_runs_10 := total_matches_10 * average_10_matches
  (total_runs_20 + total_runs_10) / (total_matches_20 + total_matches_10)

theorem average_30_matches_is_25 (h1 : average_runs_in_30_matches 30 15 20 10 = 25) : 
  average_runs_in_30_matches 30 15 20 10 = 25 := 
  by
    exact h1

end average_30_matches_is_25_l200_200274


namespace line_third_quadrant_l200_200820

theorem line_third_quadrant (A B C : ℝ) (h_origin : C = 0)
  (h_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ A * x - B * y = 0) :
  A * B < 0 :=
by
  sorry

end line_third_quadrant_l200_200820


namespace domain_all_real_l200_200147

theorem domain_all_real (p : ℝ) : 
  (∀ x : ℝ, -3 * x ^ 2 + 3 * x + p ≠ 0) ↔ p < -3 / 4 := 
by
  sorry

end domain_all_real_l200_200147


namespace sum_of_areas_of_rectangles_l200_200908

theorem sum_of_areas_of_rectangles :
  let width := 2
  let lengths := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => l * width)
  let total_area := areas.sum
  total_area = 182 := by
  sorry

end sum_of_areas_of_rectangles_l200_200908


namespace percent_profit_l200_200134

theorem percent_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) (final_profit_percent : ℝ)
  (h1 : cost = 50)
  (h2 : markup_percent = 30)
  (h3 : discount_percent = 10)
  (h4 : final_profit_percent = 17)
  : (markup_percent / 100 * cost - discount_percent / 100 * (cost + markup_percent / 100 * cost)) / cost * 100 = final_profit_percent := 
by
  sorry

end percent_profit_l200_200134


namespace trigonometric_inequality_proof_l200_200536

theorem trigonometric_inequality_proof : 
  ∀ (sin cos : ℝ → ℝ), 
  (∀ θ, 0 ≤ θ ∧ θ ≤ π/2 → sin θ = cos (π/2 - θ)) → 
  sin (π * 11 / 180) < sin (π * 12 / 180) ∧ sin (π * 12 / 180) < sin (π * 80 / 180) :=
by 
  intros sin cos identity
  sorry

end trigonometric_inequality_proof_l200_200536


namespace max_value_f_period_f_l200_200256

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 - (Real.cos x) ^ 4

theorem max_value_f : ∃ x : ℝ, (f x) = 1 / 4 :=
sorry

theorem period_f : ∃ p : ℝ, p = π / 2 ∧ ∀ x : ℝ, f (x + p) = f x :=
sorry

end max_value_f_period_f_l200_200256


namespace arithmetic_seq_finite_negative_terms_l200_200712

theorem arithmetic_seq_finite_negative_terms (a d : ℝ) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → a + n * d ≥ 0) ↔ (a < 0 ∧ d > 0) :=
by
  sorry

end arithmetic_seq_finite_negative_terms_l200_200712


namespace molecular_weight_BaCl2_l200_200363

theorem molecular_weight_BaCl2 (mw8 : ℝ) (n : ℝ) (h : mw8 = 1656) : (mw8 / n = 207) ↔ n = 8 := 
by
  sorry

end molecular_weight_BaCl2_l200_200363


namespace value_of_expression_l200_200756

-- Definitions of the variables x and y along with their assigned values
def x : ℕ := 20
def y : ℕ := 8

-- The theorem that asserts the value of (x - y) * (x + y) equals 336
theorem value_of_expression : (x - y) * (x + y) = 336 := by 
  -- Skipping proof
  sorry

end value_of_expression_l200_200756


namespace odd_function_behavior_l200_200592

variable {f : ℝ → ℝ}

theorem odd_function_behavior (h1 : ∀ x : ℝ, f (-x) = -f x) 
                             (h2 : ∀ x : ℝ, 0 < x → f x = x * (1 + x)) 
                             (x : ℝ)
                             (hx : x < 0) : 
  f x = x * (1 - x) :=
by
  -- Insert proof here
  sorry

end odd_function_behavior_l200_200592


namespace lcm_quadruples_count_l200_200985

-- Define the problem conditions
variables (r s : ℕ) (hr : r > 0) (hs : s > 0)

-- Define the mathematical problem statement
theorem lcm_quadruples_count :
  ( ∀ (a b c d : ℕ),
    lcm (lcm a b) c = lcm (lcm a b) d ∧
    lcm (lcm a b) c = lcm (lcm a c) d ∧
    lcm (lcm a b) c = lcm (lcm b c) d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a = 3 ^ r * 7 ^ s ∧
    b = 3 ^ r * 7 ^ s ∧
    c = 3 ^ r * 7 ^ s ∧
    d = 3 ^ r * 7 ^ s 
  → ∃ n, n = (1 + 4 * r + 6 * r^2) * (1 + 4 * s + 6 * s^2)) :=
sorry

end lcm_quadruples_count_l200_200985


namespace intersection_M_P_l200_200259

def is_natural (x : ℤ) : Prop := x ≥ 0

def M (x : ℤ) : Prop := (x - 1)^2 < 4 ∧ is_natural x

def P := ({-1, 0, 1, 2, 3} : Set ℤ)

theorem intersection_M_P :
  {x : ℤ | M x} ∩ P = {0, 1, 2} :=
  sorry

end intersection_M_P_l200_200259


namespace profit_per_unit_and_minimum_units_l200_200749

noncomputable def conditions (x y m : ℝ) : Prop :=
  2 * x + 7 * y = 41 ∧
  x + 3 * y = 18 ∧
  0.5 * m + 0.3 * (30 - m) ≥ 13.1

theorem profit_per_unit_and_minimum_units (x y m : ℝ) :
  conditions x y m → x = 3 ∧ y = 5 ∧ m ≥ 21 :=
by
  sorry

end profit_per_unit_and_minimum_units_l200_200749


namespace parallel_implies_not_contained_l200_200570

variables {Line Plane : Type} (l : Line) (α : Plane)

-- Define the predicate for a line being parallel to a plane
def parallel (l : Line) (α : Plane) : Prop := sorry

-- Define the predicate for a line not being contained in a plane
def not_contained (l : Line) (α : Plane) : Prop := sorry

theorem parallel_implies_not_contained (l : Line) (α : Plane) (h : parallel l α) : not_contained l α :=
sorry

end parallel_implies_not_contained_l200_200570


namespace scientific_notation_l200_200941

def significant_digits : ℝ := 4.032
def exponent : ℤ := 11
def original_number : ℝ := 403200000000

theorem scientific_notation : original_number = significant_digits * 10 ^ exponent := 
by
  sorry

end scientific_notation_l200_200941


namespace count_even_numbers_between_500_and_800_l200_200428

theorem count_even_numbers_between_500_and_800 :
  let a := 502
  let d := 2
  let last_term := 798
  ∃ n, a + (n - 1) * d = last_term ∧ n = 149 :=
by
  sorry

end count_even_numbers_between_500_and_800_l200_200428


namespace gcf_50_75_l200_200924

theorem gcf_50_75 : Nat.gcd 50 75 = 25 := by
  sorry

end gcf_50_75_l200_200924


namespace max_difference_primes_l200_200417

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even_integer : ℕ := 138

theorem max_difference_primes (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ p + q = even_integer ∧ p ≠ q →
  (q - p) = 124 :=
by
  sorry

end max_difference_primes_l200_200417


namespace actual_area_l200_200990

open Real

theorem actual_area
  (scale : ℝ)
  (mapped_area_cm2 : ℝ)
  (actual_area_cm2 : ℝ)
  (actual_area_m2 : ℝ)
  (h_scale : scale = 1 / 50000)
  (h_mapped_area : mapped_area_cm2 = 100)
  (h_proportion : mapped_area_cm2 / actual_area_cm2 = scale ^ 2)
  : actual_area_m2 = 2.5 * 10^7 :=
by
  sorry

end actual_area_l200_200990


namespace num_comics_liked_by_males_l200_200692

-- Define the problem conditions
def num_comics : ℕ := 300
def percent_liked_by_females : ℕ := 30
def percent_disliked_by_both : ℕ := 30

-- Define the main theorem to prove
theorem num_comics_liked_by_males :
  let percent_liked_by_at_least_one_gender := 100 - percent_disliked_by_both
  let num_comics_liked_by_females := percent_liked_by_females * num_comics / 100
  let num_comics_liked_by_at_least_one_gender := percent_liked_by_at_least_one_gender * num_comics / 100
  num_comics_liked_by_at_least_one_gender - num_comics_liked_by_females = 120 :=
by
  sorry

end num_comics_liked_by_males_l200_200692


namespace sum_of_solutions_l200_200337

theorem sum_of_solutions (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 225) : 2 * x = 0 :=
by
  sorry

end sum_of_solutions_l200_200337


namespace classify_abc_l200_200264

theorem classify_abc (a b c : ℝ) 
  (h1 : (a > 0 ∨ a < 0 ∨ a = 0) ∧ (b > 0 ∨ b < 0 ∨ b = 0) ∧ (c > 0 ∨ c < 0 ∨ c = 0))
  (h2 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ (a > 0 ∧ b = 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c = 0) ∨
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ b > 0 ∧ c < 0) ∨ (a = 0 ∧ b < 0 ∧ c > 0))
  (h3 : |a| = b^2 * (b - c)) : 
  a < 0 ∧ b > 0 ∧ c = 0 :=
by 
  sorry

end classify_abc_l200_200264


namespace find_lighter_ball_min_weighings_l200_200120

noncomputable def min_weighings_to_find_lighter_ball (balls : Fin 9 → ℕ) : ℕ :=
  2

-- Given: 9 balls, where 8 weigh 10 grams and 1 weighs 9 grams, and a balance scale.
theorem find_lighter_ball_min_weighings :
  (∃ i : Fin 9, balls i = 9 ∧ (∀ j : Fin 9, j ≠ i → balls j = 10)) 
  → min_weighings_to_find_lighter_ball balls = 2 :=
by
  intros
  sorry

end find_lighter_ball_min_weighings_l200_200120


namespace thursday_to_wednesday_ratio_l200_200070

-- Let M, T, W, Th be the number of messages sent on Monday, Tuesday, Wednesday, and Thursday respectively.
variables (M T W Th : ℕ)

-- Conditions are given as follows
axiom hM : M = 300
axiom hT : T = 200
axiom hW : W = T + 300
axiom hSum : M + T + W + Th = 2000

-- Define the function to compute the ratio
def ratio (a b : ℕ) : ℚ := a / b

-- The target is to prove that the ratio of the messages sent on Thursday to those sent on Wednesday is 2 / 1
theorem thursday_to_wednesday_ratio : ratio Th W = 2 :=
by {
  sorry
}

end thursday_to_wednesday_ratio_l200_200070


namespace solution_unique_l200_200008

def satisfies_equation (x y : ℝ) : Prop :=
  (x - 7)^2 + (y - 8)^2 + (x - y)^2 = 1 / 3

theorem solution_unique (x y : ℝ) :
  satisfies_equation x y ↔ x = 7 + 1/3 ∧ y = 8 - 1/3 :=
by {
  sorry
}

end solution_unique_l200_200008


namespace triangle_angles_geometric_progression_l200_200053

-- Theorem: If the sides of a triangle whose angles form an arithmetic progression are in geometric progression, then all three angles are 60°.
theorem triangle_angles_geometric_progression (A B C : ℝ) (a b c : ℝ)
  (h_arith_progression : 2 * B = A + C)
  (h_sum_angles : A + B + C = 180)
  (h_geo_progression : (a / b) = (b / c))
  (h_b_angle : B = 60) :
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_angles_geometric_progression_l200_200053


namespace find_X_plus_Y_in_base_8_l200_200073

theorem find_X_plus_Y_in_base_8 (X Y : ℕ) (h1 : 3 * 8^2 + X * 8 + Y + 5 * 8 + 2 = 4 * 8^2 + X * 8 + 3) : X + Y = 1 :=
sorry

end find_X_plus_Y_in_base_8_l200_200073


namespace programs_produce_same_result_l200_200479

-- Define Program A's computation
def programA_sum : ℕ := (List.range (1000 + 1)).sum -- Sum of numbers from 0 to 1000

-- Define Program B's computation
def programB_sum : ℕ := (List.range (1000 + 1)).reverse.sum -- Sum of numbers from 1000 down to 0

theorem programs_produce_same_result : programA_sum = programB_sum :=
  sorry

end programs_produce_same_result_l200_200479


namespace number_of_triangles_l200_200790

/-!
# Problem Statement
Given a square with 20 interior points connected such that the lines do not intersect and divide the square into triangles,
prove that the number of triangles formed is 42.
-/

theorem number_of_triangles (V E F : ℕ) (hV : V = 24) (hE : E = (3 * F + 1) / 2) (hF : V - E + F = 2) :
  (F - 1) = 42 :=
by
  sorry

end number_of_triangles_l200_200790


namespace apples_for_juice_l200_200566

def totalApples : ℝ := 6
def exportPercentage : ℝ := 0.25
def juicePercentage : ℝ := 0.60

theorem apples_for_juice : 
  let remainingApples := totalApples * (1 - exportPercentage)
  let applesForJuice := remainingApples * juicePercentage
  applesForJuice = 2.7 :=
by
  sorry

end apples_for_juice_l200_200566


namespace distance_from_hyperbola_focus_to_line_l200_200448

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l200_200448


namespace sum_of_arithmetic_sequence_2008_terms_l200_200211

theorem sum_of_arithmetic_sequence_2008_terms :
  let a := -1776
  let d := 11
  let n := 2008
  let l := a + (n - 1) * d
  let S := (n / 2) * (a + l)
  S = 18599100 := by
  sorry

end sum_of_arithmetic_sequence_2008_terms_l200_200211


namespace simplified_expression_l200_200638

noncomputable def simplify_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ :=
  (x⁻¹ - z⁻¹)⁻¹

theorem simplified_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : 
  simplify_expression x z hx hz = x * z / (z - x) := 
by
  sorry

end simplified_expression_l200_200638


namespace cards_probability_ratio_l200_200455

theorem cards_probability_ratio :
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  q / p = 44 :=
by
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  have : q / p = 44 := sorry
  exact this

end cards_probability_ratio_l200_200455


namespace greatest_ab_sum_l200_200463

theorem greatest_ab_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) :
  a + b = Real.sqrt 220 ∨ a + b = -Real.sqrt 220 :=
sorry

end greatest_ab_sum_l200_200463


namespace max_k_exists_l200_200735

noncomputable def max_possible_k (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) : ℝ :=
sorry

theorem max_k_exists (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) :
  ∃ k_max : ℝ, k_max = max_possible_k x y k h_pos h_eq :=
sorry

end max_k_exists_l200_200735


namespace tax_budget_level_correct_l200_200327

-- Definitions for tax types and their corresponding budget levels
inductive TaxType where
| property_tax_organizations : TaxType
| federal_tax : TaxType
| profit_tax_organizations : TaxType
| tax_subjects_RF : TaxType
| transport_collecting : TaxType
deriving DecidableEq

inductive BudgetLevel where
| federal_budget : BudgetLevel
| subjects_RF_budget : BudgetLevel
deriving DecidableEq

def tax_to_budget_level : TaxType → BudgetLevel
| TaxType.property_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.federal_tax => BudgetLevel.federal_budget
| TaxType.profit_tax_organizations => BudgetLevel.subjects_RF_budget
| TaxType.tax_subjects_RF => BudgetLevel.subjects_RF_budget
| TaxType.transport_collecting => BudgetLevel.subjects_RF_budget

theorem tax_budget_level_correct :
  tax_to_budget_level TaxType.property_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.federal_tax = BudgetLevel.federal_budget ∧
  tax_to_budget_level TaxType.profit_tax_organizations = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.tax_subjects_RF = BudgetLevel.subjects_RF_budget ∧
  tax_to_budget_level TaxType.transport_collecting = BudgetLevel.subjects_RF_budget :=
by
  sorry

end tax_budget_level_correct_l200_200327


namespace find_a_geometric_sequence_l200_200948

theorem find_a_geometric_sequence (a : ℤ) (T : ℕ → ℤ) (b : ℕ → ℤ) :
  (∀ n, T n = 3 ^ n + a) →
  b 1 = T 1 →
  (∀ n, n ≥ 2 → b n = T n - T (n - 1)) →
  (∀ n, n ≥ 2 → (∃ r, r * b n = b (n - 1))) →
  a = -1 :=
by
  sorry

end find_a_geometric_sequence_l200_200948


namespace impossible_permuted_sum_l200_200606

def isPermutation (X Y : ℕ) : Prop :=
  -- Define what it means for two numbers to be permutations of each other.
  sorry

theorem impossible_permuted_sum (X Y : ℕ) (h1 : isPermutation X Y) (h2 : X + Y = (10^1111 - 1)) : false :=
  sorry

end impossible_permuted_sum_l200_200606


namespace quadratic_to_standard_form_l200_200920

theorem quadratic_to_standard_form (a b c : ℝ) (x : ℝ) :
  (20 * x^2 + 240 * x + 3200 = a * (x + b)^2 + c) → (a + b + c = 2506) :=
  sorry

end quadratic_to_standard_form_l200_200920


namespace elaine_earnings_increase_l200_200315

variable (E : ℝ) (P : ℝ)

theorem elaine_earnings_increase
  (h1 : E > 0) 
  (h2 : 0.30 * E * (1 + P / 100) = 1.80 * 0.20 * E) : 
  P = 20 :=
by
  sorry

end elaine_earnings_increase_l200_200315


namespace selina_sells_5_shirts_l200_200741

theorem selina_sells_5_shirts
    (pants_price shorts_price shirts_price : ℕ)
    (pants_sold shorts_sold shirts_bought remaining_money : ℕ)
    (total_earnings : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  pants_sold = 3 →
  shorts_sold = 5 →
  shirts_bought = 2 →
  remaining_money = 30 →
  total_earnings = remaining_money + shirts_bought * 10 →
  total_earnings = 50 →
  total_earnings = pants_sold * pants_price + shorts_sold * shorts_price + 20 →
  20 / shirts_price = 5 :=
by
  sorry

end selina_sells_5_shirts_l200_200741


namespace possible_values_of_a_l200_200007

theorem possible_values_of_a (x y a : ℝ) (h1 : x + y = a) (h2 : x^3 + y^3 = a) (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 :=
by sorry

end possible_values_of_a_l200_200007


namespace number_of_intersection_points_l200_200199

-- Define the standard parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define what it means for a line to be tangent to the parabola
def is_tangent (m : ℝ) (c : ℝ) : Prop :=
  ∃ x0 : ℝ, parabola x0 = m * x0 + c ∧ 2 * x0 = m

-- Define what it means for a line to intersect the parabola
def line_intersects_parabola (m : ℝ) (c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, parabola x1 = m * x1 + c ∧ parabola x2 = m * x2 + c

-- Main theorem statement
theorem number_of_intersection_points :
  (∃ m c : ℝ, is_tangent m c) → (∃ m' c' : ℝ, line_intersects_parabola m' c') →
  ∃ n : ℕ, n = 1 ∨ n = 2 ∨ n = 3 :=
sorry

end number_of_intersection_points_l200_200199


namespace find_initial_alison_stamps_l200_200659

-- Define initial number of stamps Anna, Jeff, and Alison had
def initial_anna_stamps : ℕ := 37
def initial_jeff_stamps : ℕ := 31
def final_anna_stamps : ℕ := 50

-- Define the assumption that Alison gave Anna half of her stamps
def alison_gave_anna_half (a : ℕ) : Prop :=
  initial_anna_stamps + a / 2 = final_anna_stamps

-- Define the problem of finding the initial number of stamps Alison had
def alison_initial_stamps : ℕ := 26

theorem find_initial_alison_stamps :
  ∃ a : ℕ, alison_gave_anna_half a ∧ a = alison_initial_stamps :=
by
  sorry

end find_initial_alison_stamps_l200_200659


namespace sum_arith_seq_l200_200814

theorem sum_arith_seq (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h₁ : ∀ n, S n = n * a 1 + (n * (n - 1)) * d / 2)
    (h₂ : S 10 = S 20)
    (h₃ : d > 0) :
    a 10 + a 22 > 0 := 
sorry

end sum_arith_seq_l200_200814


namespace common_internal_tangent_length_l200_200307

-- Definitions based on given conditions
def center_distance : ℝ := 50
def radius_small : ℝ := 7
def radius_large : ℝ := 10

-- Target theorem
theorem common_internal_tangent_length :
  let AB := center_distance
  let BE := radius_small + radius_large 
  let AE := Real.sqrt (AB^2 - BE^2)
  AE = Real.sqrt 2211 :=
by
  sorry

end common_internal_tangent_length_l200_200307


namespace problem_statement_l200_200267

variable {α : Type*} [LinearOrderedCommRing α]

theorem problem_statement (a b c d e : α) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : a * b^2 * c * d^4 * e < 0 :=
by
  sorry

end problem_statement_l200_200267


namespace parabola_directrix_l200_200380

theorem parabola_directrix (x y : ℝ) : 
    (x^2 = (1/2) * y) -> (y = -1/8) :=
sorry

end parabola_directrix_l200_200380


namespace no_preimage_iff_k_less_than_neg2_l200_200248

theorem no_preimage_iff_k_less_than_neg2 (k : ℝ) :
  ¬∃ x : ℝ, x^2 - 2 * x - 1 = k ↔ k < -2 :=
sorry

end no_preimage_iff_k_less_than_neg2_l200_200248


namespace max_value_Tn_l200_200824

noncomputable def geom_seq (a : ℕ → ℝ) : Prop := 
∀ n : ℕ, a (n+1) = 2 * a n

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 0 * (1 - (2 : ℝ)^n) / (1 - (2 : ℝ))

noncomputable def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(9 * sum_first_n_terms a n - sum_first_n_terms a (2 * n)) / (a n * (2 : ℝ)^n)

theorem max_value_Tn (a : ℕ → ℝ) (h : geom_seq a) : 
  ∃ n, T_n a n ≤ 3 :=
sorry

end max_value_Tn_l200_200824


namespace turtles_still_on_sand_l200_200022

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l200_200022


namespace solve_quadratic_equation_l200_200085

theorem solve_quadratic_equation : 
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by
  sorry


end solve_quadratic_equation_l200_200085


namespace tea_sales_l200_200687

theorem tea_sales (L T : ℕ) (h1 : L = 32) (h2 : L = 4 * T + 8) : T = 6 :=
by
  sorry

end tea_sales_l200_200687


namespace bank_robbery_participants_l200_200017

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l200_200017


namespace hyperbola_eq_from_conditions_l200_200651

-- Conditions of the problem
def hyperbola_center : Prop := ∃ (h : ℝ → ℝ → Prop), h 0 0
def hyperbola_eccentricity : Prop := ∃ e : ℝ, e = 2
def parabola_focus : Prop := ∃ p : ℝ × ℝ, p = (4, 0)
def parabola_equation : Prop := ∀ x y : ℝ, y^2 = 8 * x

-- Hyperbola equation to be proved
def hyperbola_equation : Prop := ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1

-- Lean 4 theorem statement
theorem hyperbola_eq_from_conditions 
  (h_center : hyperbola_center) 
  (h_eccentricity : hyperbola_eccentricity) 
  (p_focus : parabola_focus) 
  (p_eq : parabola_equation) 
  : hyperbola_equation :=
by
  sorry

end hyperbola_eq_from_conditions_l200_200651


namespace find_f500_l200_200371

variable (f : ℕ → ℕ)
variable (h : ∀ x y : ℕ, f (x * y) = f x + f y)
variable (h₁ : f 10 = 16)
variable (h₂ : f 40 = 24)

theorem find_f500 : f 500 = 44 :=
sorry

end find_f500_l200_200371


namespace perfect_square_trinomial_m_eq_6_or_neg6_l200_200269

theorem perfect_square_trinomial_m_eq_6_or_neg6
  (m : ℤ) :
  (∃ a : ℤ, x * x + m * x + 9 = (x + a) * (x + a)) → (m = 6 ∨ m = -6) :=
by
  sorry

end perfect_square_trinomial_m_eq_6_or_neg6_l200_200269


namespace largest_circle_at_A_l200_200154

/--
Given a pentagon with side lengths AB = 16 cm, BC = 14 cm, CD = 17 cm, DE = 13 cm, and EA = 14 cm,
and given five circles with centers A, B, C, D, and E such that each pair of circles with centers at
the ends of a side of the pentagon touch on that side, the circle with center A
has the largest radius.
-/
theorem largest_circle_at_A
  (rA rB rC rD rE : ℝ) 
  (hAB : rA + rB = 16)
  (hBC : rB + rC = 14)
  (hCD : rC + rD = 17)
  (hDE : rD + rE = 13)
  (hEA : rE + rA = 14) :
  rA ≥ rB ∧ rA ≥ rC ∧ rA ≥ rD ∧ rA ≥ rE := 
sorry

end largest_circle_at_A_l200_200154


namespace y1_lt_y2_of_linear_graph_l200_200805

/-- In the plane rectangular coordinate system xOy, if points A(2, y1) and B(5, y2) 
    lie on the graph of a linear function y = x + b (where b is a constant), then y1 < y2. -/
theorem y1_lt_y2_of_linear_graph (y1 y2 b : ℝ) (hA : y1 = 2 + b) (hB : y2 = 5 + b) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_graph_l200_200805


namespace sum_of_money_l200_200827

noncomputable def Patricia : ℕ := 60
noncomputable def Jethro : ℕ := Patricia / 3
noncomputable def Carmen : ℕ := 2 * Jethro - 7

theorem sum_of_money : Patricia + Jethro + Carmen = 113 := by
  sorry

end sum_of_money_l200_200827


namespace sum_of_midpoint_coordinates_l200_200200

theorem sum_of_midpoint_coordinates :
  let (x1, y1) := (8, 16)
  let (x2, y2) := (2, -8)
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 9 :=
by
  sorry

end sum_of_midpoint_coordinates_l200_200200


namespace range_of_a_l200_200873

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then (a - 5) * x + 8 else 2 * a / x

theorem range_of_a (a : ℝ) : 
  (∀ x y, x < y → f a x ≥ f a y) → (2 ≤ a ∧ a < 5) :=
sorry

end range_of_a_l200_200873


namespace find_min_value_l200_200595

noncomputable def min_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem find_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (cond : 2/x + 3/y + 5/z = 10) : min_value x y z = 390625 / 1296 :=
sorry

end find_min_value_l200_200595


namespace find_value_of_m_l200_200373

open Real

theorem find_value_of_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = sqrt 10 := by
  sorry

end find_value_of_m_l200_200373


namespace smallest_number_of_students_l200_200837

theorem smallest_number_of_students (n : ℕ) : 
  (6 * n + 2 > 40) → (∃ n, 4 * n + 2 * (n + 1) = 44) :=
 by
  intro h
  exact sorry

end smallest_number_of_students_l200_200837


namespace correct_omega_l200_200869

theorem correct_omega (Ω : ℕ) (h : Ω * Ω = 2 * 2 * 2 * 2 * 3 * 3) : Ω = 2 * 2 * 3 :=
by
  sorry

end correct_omega_l200_200869


namespace point_P_distance_l200_200635

variable (a b c d x : ℝ)

-- Define the points on the line
def O := 0
def A := a
def B := b
def C := c
def D := d

-- Define the conditions for point P
def AP_PDRatio := (|a - x| / |x - d| = 2 * |b - x| / |x - c|)

theorem point_P_distance : AP_PDRatio a b c d x → b + c - a = x :=
by
  sorry

end point_P_distance_l200_200635


namespace arithmetic_sequence_properties_l200_200233

noncomputable def arithmetic_sequence (a3 a5_a7_sum : ℝ) : Prop :=
  ∃ (a d : ℝ), a + 2*d = a3 ∧ 2*a + 10*d = a5_a7_sum

noncomputable def sequence_a_n (a d n : ℝ) : ℝ := a + (n - 1)*d

noncomputable def sum_S_n (a d n : ℝ) : ℝ := n/2 * (2*a + (n-1)*d)

noncomputable def sequence_b_n (a d n : ℝ) : ℝ := 1 / (sequence_a_n a d n ^ 2 - 1)

noncomputable def sum_T_n (a d n : ℝ) : ℝ :=
  (1 / 4) * (1 - 1/(n+1))

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 7 26) →
  (∀ n : ℕ+, sequence_a_n 3 2 n = 2 * n + 1) ∧
  (∀ n : ℕ+, sum_S_n 3 2 n = n^2 + 2 * n) ∧
  (∀ n : ℕ+, sum_T_n 3 2 n = n / (4 * (n + 1))) :=
by
  sorry

end arithmetic_sequence_properties_l200_200233


namespace incorrect_conclusion_C_l200_200077

variable {a : ℕ → ℝ} {q : ℝ}

-- Conditions
def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q

theorem incorrect_conclusion_C 
  (h_geo: geo_seq a q)
  (h_cond: a 1 * a 2 < 0) : 
  a 1 * a 5 > 0 :=
by 
  sorry

end incorrect_conclusion_C_l200_200077


namespace perimeters_positive_difference_l200_200645

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l200_200645


namespace total_area_of_removed_triangles_l200_200352

theorem total_area_of_removed_triangles (a b : ℝ)
  (square_side : ℝ := 16)
  (triangle_hypotenuse : ℝ := 8)
  (isosceles_right_triangle : a = b ∧ a^2 + b^2 = triangle_hypotenuse^2) :
  4 * (1 / 2 * a * b) = 64 :=
by
  -- Sketch of the proof:
  -- From the isosceles right triangle property and Pythagorean theorem,
  -- a^2 + b^2 = 8^2 ⇒ 2 * a^2 = 64 ⇒ a^2 = 32 ⇒ a = b = 4√2
  -- The area of one triangle is (1/2) * a * b = 16
  -- Total area of four such triangles is 4 * 16 = 64
  sorry

end total_area_of_removed_triangles_l200_200352


namespace average_infection_l200_200148

theorem average_infection (x : ℕ) (h : 1 + 2 * x + x^2 = 121) : x = 10 :=
by
  sorry -- Proof to be filled.

end average_infection_l200_200148


namespace maximum_area_of_rectangle_with_given_perimeter_l200_200441

noncomputable def perimeter : ℝ := 30
noncomputable def area (length width : ℝ) : ℝ := length * width
noncomputable def max_area : ℝ := 56.25

theorem maximum_area_of_rectangle_with_given_perimeter :
  ∃ length width : ℝ, 2 * length + 2 * width = perimeter ∧ area length width = max_area :=
sorry

end maximum_area_of_rectangle_with_given_perimeter_l200_200441


namespace solve_real_eq_l200_200253

theorem solve_real_eq (x : ℝ) :
  (8 * x ^ 2 + 150 * x + 3) / (3 * x + 56) = 4 * x + 2 ↔ x = -1.5 ∨ x = -18.5 :=
by
  sorry

end solve_real_eq_l200_200253


namespace line_log_intersection_l200_200634

theorem line_log_intersection (a b : ℤ) (k : ℝ)
  (h₁ : k = a + Real.sqrt b)
  (h₂ : k > 0)
  (h₃ : Real.log k / Real.log 2 - Real.log (k + 2) / Real.log 2 = 1
    ∨ Real.log (k + 2) / Real.log 2 - Real.log k / Real.log 2 = 1) :
  a + b = 2 :=
sorry

end line_log_intersection_l200_200634


namespace rectangle_area_diagonal_l200_200192

theorem rectangle_area_diagonal (r l w d : ℝ) (h_ratio : r = 5 / 2) (h_diag : d^2 = l^2 + w^2) : ∃ k : ℝ, (k = 10 / 29) ∧ (l / w = r) ∧ (l^2 + w^2 = d^2) :=
by
  sorry

end rectangle_area_diagonal_l200_200192


namespace odd_function_value_at_neg2_l200_200193

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_ge_one : ∀ x, 1 ≤ x → f x = 3 * x - 7)

theorem odd_function_value_at_neg2 : f (-2) = 1 :=
by
  -- Proof goes here
  sorry

end odd_function_value_at_neg2_l200_200193


namespace least_positive_three_digit_multiple_of_8_l200_200225

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l200_200225


namespace sales_worth_l200_200694

theorem sales_worth (S: ℝ) : 
  (1300 + 0.025 * (S - 4000) = 0.05 * S + 600) → S = 24000 :=
by
  sorry

end sales_worth_l200_200694


namespace empty_vessel_mass_l200_200864

theorem empty_vessel_mass
  (m1 : ℝ) (m2 : ℝ) (rho_K : ℝ) (rho_B : ℝ) (V : ℝ) (m_c : ℝ)
  (h1 : m1 = m_c + rho_K * V)
  (h2 : m2 = m_c + rho_B * V)
  (h_mass_kerosene : m1 = 31)
  (h_mass_water : m2 = 33)
  (h_rho_K : rho_K = 800)
  (h_rho_B : rho_B = 1000) :
  m_c = 23 :=
by
  -- Proof skipped
  sorry

end empty_vessel_mass_l200_200864


namespace Adam_bought_9_cat_food_packages_l200_200065

def num_cat_food_packages (c : ℕ) : Prop :=
  let cat_cans := 10 * c
  let dog_cans := 7 * 5
  cat_cans = dog_cans + 55

theorem Adam_bought_9_cat_food_packages : num_cat_food_packages 9 :=
by
  unfold num_cat_food_packages
  sorry

end Adam_bought_9_cat_food_packages_l200_200065


namespace initial_books_in_library_l200_200418

theorem initial_books_in_library
  (books_out_tuesday : ℕ)
  (books_in_thursday : ℕ)
  (books_out_friday : ℕ)
  (final_books : ℕ)
  (h1 : books_out_tuesday = 227)
  (h2 : books_in_thursday = 56)
  (h3 : books_out_friday = 35)
  (h4 : final_books = 29) : 
  initial_books = 235 :=
by
  sorry

end initial_books_in_library_l200_200418


namespace hungarian_license_plates_l200_200648

/-- 
In Hungarian license plates, digits can be identical. Based on observations, 
someone claimed that on average, approximately 3 out of every 10 vehicles 
have such license plates. Is this statement true?
-/
theorem hungarian_license_plates : 
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  abs (probability - 0.3) < 0.05 :=
by {
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  sorry
}

end hungarian_license_plates_l200_200648


namespace money_left_l200_200890

def initial_money : ℝ := 18
def spent_on_video_games : ℝ := 6
def spent_on_snack : ℝ := 3
def toy_original_cost : ℝ := 4
def toy_discount : ℝ := 0.25

theorem money_left (initial_money spent_on_video_games spent_on_snack toy_original_cost toy_discount : ℝ) :
  initial_money = 18 →
  spent_on_video_games = 6 →
  spent_on_snack = 3 →
  toy_original_cost = 4 →
  toy_discount = 0.25 →
  (initial_money - (spent_on_video_games + spent_on_snack + (toy_original_cost * (1 - toy_discount)))) = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end money_left_l200_200890


namespace price_increase_decrease_eq_l200_200512

theorem price_increase_decrease_eq (x : ℝ) (p : ℝ) (hx : x ≠ 0) :
  x * (1 + p / 100) * (1 - p / 200) = x * (1 + p / 300) → p = 100 / 3 :=
by
  intro h
  -- The proof would go here
  sorry

end price_increase_decrease_eq_l200_200512


namespace geometric_sequence_general_formula_arithmetic_sequence_sum_l200_200974

-- Problem (I)
theorem geometric_sequence_general_formula (a : ℕ → ℝ) (q a1 : ℝ)
  (h1 : ∀ n, a (n + 1) = q * a n)
  (h2 : a 1 + a 2 = 6)
  (h3 : a 1 * a 2 = a 3) :
  a n = 2 ^ n :=
sorry

-- Problem (II)
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) (S T : ℕ → ℝ)
  (h1 : ∀ n, a n = 2 ^ n)
  (h2 : ∀ n, S n = (n * (b 1 + b n)) / 2)
  (h3 : ∀ n, S (2 * n + 1) = b n * b (n + 1))
  (h4 : ∀ n, b n = 2 * n + 1) :
  T n = 5 - (2 * n + 5) / 2 ^ n :=
sorry

end geometric_sequence_general_formula_arithmetic_sequence_sum_l200_200974


namespace whitewashing_cost_l200_200117

noncomputable def cost_of_whitewashing (l w h : ℝ) (c : ℝ) (door_area window_area : ℝ) (num_windows : ℝ) : ℝ :=
  let perimeter := 2 * (l + w)
  let total_wall_area := perimeter * h
  let total_window_area := num_windows * window_area
  let total_paintable_area := total_wall_area - (door_area + total_window_area)
  total_paintable_area * c

theorem whitewashing_cost:
  cost_of_whitewashing 25 15 12 6 (6 * 3) (4 * 3) 3 = 5436 := by
  sorry

end whitewashing_cost_l200_200117


namespace circle_y_axis_intersection_range_l200_200684

theorem circle_y_axis_intersection_range (m : ℝ) : (4 - 4 * (m + 6) > 0) → (-2 < 0) → (m + 6 > 0) → (-6 < m ∧ m < -5) :=
by 
  intros h1 h2 h3 
  sorry

end circle_y_axis_intersection_range_l200_200684


namespace zoe_pop_albums_l200_200205

theorem zoe_pop_albums (total_songs country_albums songs_per_album : ℕ) (h1 : total_songs = 24) (h2 : country_albums = 3) (h3 : songs_per_album = 3) :
  total_songs - (country_albums * songs_per_album) = 15 ↔ (total_songs - (country_albums * songs_per_album)) / songs_per_album = 5 :=
by
  sorry

end zoe_pop_albums_l200_200205


namespace parabola_directrix_equation_l200_200365

theorem parabola_directrix_equation (x y a : ℝ) : 
  (x^2 = 4 * y) → (a = 1) → (y = -a) := by
  intro h1 h2
  rw [h2] -- given a = 1
  sorry

end parabola_directrix_equation_l200_200365


namespace y_value_l200_200665

theorem y_value (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := 
by 
  sorry

end y_value_l200_200665


namespace nth_term_of_sequence_99_l200_200826

def sequence_rule (n : ℕ) : ℕ :=
  if n < 20 then n * 9
  else if n % 2 = 0 then n / 2
  else if n > 19 ∧ n % 7 ≠ 0 then n - 5
  else n + 7

noncomputable def sequence_nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.repeat sequence_rule n start

theorem nth_term_of_sequence_99 :
  sequence_nth_term 65 98 = 30 :=
sorry

end nth_term_of_sequence_99_l200_200826


namespace math_proof_problem_l200_200325

variables {a b c d e f k : ℝ}

theorem math_proof_problem 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (a + b + c + (d + k) + (e + k) + (f + k) = d + e + f + (a + k) + (b + k) + (c + k) ∧
   a^2 + b^2 + c^2 + (d + k)^2 + (e + k)^2 + (f + k)^2 = d^2 + e^2 + f^2 + (a + k)^2 + (b + k)^2 + (c + k)^2 ∧
   a^3 + b^3 + c^3 + (d + k)^3 + (e + k)^3 + (f + k)^3 = d^3 + e^3 + f^3 + (a + k)^3 + (b + k)^3 + (c + k)^3) 
   ∧ 
  (a^4 + b^4 + c^4 + (d + k)^4 + (e + k)^4 + (f + k)^4 ≠ d^4 + e^4 + f^4 + (a + k)^4 + (b + k)^4 + (c + k)^4) := 
  sorry

end math_proof_problem_l200_200325


namespace square_tiles_count_l200_200688

theorem square_tiles_count (p s : ℕ) (h1 : p + s = 30) (h2 : 5 * p + 4 * s = 122) : s = 28 :=
sorry

end square_tiles_count_l200_200688


namespace volunteers_distribution_l200_200461

theorem volunteers_distribution:
  let num_volunteers := 5
  let group_distribution := (2, 2, 1)
  ∃ (ways : ℕ), ways = 15 :=
by
  sorry

end volunteers_distribution_l200_200461


namespace arithmetic_sequence_sum_l200_200887

variable (a : ℕ → ℝ) (d : ℝ)
-- Conditions
def is_arithmetic_sequence : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def condition : Prop := a 4 + a 8 = 8

-- Question
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a d →
  condition a →
  (11 / 2) * (a 1 + a 11) = 44 :=
by
  sorry

end arithmetic_sequence_sum_l200_200887


namespace deepak_age_l200_200912

-- Defining the problem with the given conditions in Lean:
theorem deepak_age (x : ℕ) (rahul_current : ℕ := 4 * x) (deepak_current : ℕ := 3 * x) :
  (rahul_current + 6 = 38) → (deepak_current = 24) :=
by
  sorry

end deepak_age_l200_200912


namespace influenza_probability_l200_200752

theorem influenza_probability :
  let flu_rate_A := 0.06
  let flu_rate_B := 0.05
  let flu_rate_C := 0.04
  let population_ratio_A := 6
  let population_ratio_B := 5
  let population_ratio_C := 4
  (population_ratio_A * flu_rate_A + population_ratio_B * flu_rate_B + population_ratio_C * flu_rate_C) / 
  (population_ratio_A + population_ratio_B + population_ratio_C) = 77 / 1500 :=
by
  sorry

end influenza_probability_l200_200752


namespace problem_statement_l200_200776

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem problem_statement : ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 := by
  sorry

end problem_statement_l200_200776


namespace max_triangles_formed_l200_200066

-- Define the triangles and their properties
structure EquilateralTriangle (α : Type) :=
(midpoint_segment : α) -- Each triangle has a segment connecting the midpoints of two sides

variables {α : Type} [OrderedSemiring α]

-- Define the condition of being mirrored horizontally
def areMirroredHorizontally (A B : EquilateralTriangle α) : Prop := 
  -- Placeholder for any formalization needed to specify mirrored horizontally
  sorry

-- Movement conditions and number of smaller triangles
def numberOfSmallerTrianglesAtMaxOverlap (A B : EquilateralTriangle α) (move_horizontally : α) : ℕ :=
  -- Placeholder function/modeling for counting triangles during movement
  sorry

-- Statement of our main theorem
theorem max_triangles_formed (A B : EquilateralTriangle α) (move_horizontally : α) 
  (h_mirrored : areMirroredHorizontally A B) :
  numberOfSmallerTrianglesAtMaxOverlap A B move_horizontally = 11 :=
sorry

end max_triangles_formed_l200_200066


namespace sheela_monthly_income_l200_200793

-- Definitions from the conditions
def deposited_amount : ℝ := 5000
def percentage_of_income : ℝ := 0.20

-- The theorem to be proven
theorem sheela_monthly_income : (deposited_amount / percentage_of_income) = 25000 := by
  sorry

end sheela_monthly_income_l200_200793


namespace combined_area_of_three_walls_l200_200772

theorem combined_area_of_three_walls (A : ℝ) :
  (A - 2 * 30 - 3 * 45 = 180) → (A = 375) :=
by
  intro h
  sorry

end combined_area_of_three_walls_l200_200772


namespace arithmetic_geometric_sequence_l200_200714

theorem arithmetic_geometric_sequence
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (b : ℕ → ℕ)
  (T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : S 3 = 12)
  (h3 : (a 1 + (a 2 - a 1))^2 = a 1 * (a 1 + 2 * (a 2 - a 1) + 2))
  (h4 : ∀ n, b n = (3 ^ n) * a n) :
  (∀ n, a n = 2 * n) ∧ 
  (∀ n, T n = (2 * n - 1) * 3^(n + 1) / 2 + 3 / 2) :=
sorry

end arithmetic_geometric_sequence_l200_200714


namespace trig_identity_l200_200557

theorem trig_identity : 2 * Real.sin (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 1 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l200_200557


namespace total_crosswalk_lines_l200_200062

theorem total_crosswalk_lines 
  (intersections : ℕ) 
  (crosswalks_per_intersection : ℕ) 
  (lines_per_crosswalk : ℕ)
  (h1 : intersections = 10)
  (h2 : crosswalks_per_intersection = 8)
  (h3 : lines_per_crosswalk = 30) :
  intersections * crosswalks_per_intersection * lines_per_crosswalk = 2400 := 
by {
  sorry
}

end total_crosswalk_lines_l200_200062


namespace abs_eq_four_l200_200019

theorem abs_eq_four (x : ℝ) (h : |x| = 4) : x = 4 ∨ x = -4 :=
by
  sorry

end abs_eq_four_l200_200019


namespace john_toy_store_fraction_l200_200910

theorem john_toy_store_fraction
  (allowance : ℝ)
  (spent_at_arcade_fraction : ℝ)
  (remaining_allowance : ℝ)
  (spent_at_candy_store : ℝ)
  (spent_at_toy_store : ℝ)
  (john_allowance : allowance = 3.60)
  (arcade_fraction : spent_at_arcade_fraction = 3 / 5)
  (arcade_amount : remaining_allowance = allowance - (spent_at_arcade_fraction * allowance))
  (candy_store_amount : spent_at_candy_store = 0.96)
  (remaining_after_candy_store : spent_at_toy_store = remaining_allowance - spent_at_candy_store)
  : spent_at_toy_store / remaining_allowance = 1 / 3 :=
by
  sorry

end john_toy_store_fraction_l200_200910


namespace side_length_of_square_l200_200218

theorem side_length_of_square (s : ℝ) (h : s^2 = 2 * (4 * s)) : s = 8 := 
by
  sorry

end side_length_of_square_l200_200218


namespace production_today_l200_200870

theorem production_today (n x: ℕ) (avg_past: ℕ) 
  (h1: avg_past = 50) 
  (h2: n = 1) 
  (h3: (avg_past * n + x) / (n + 1) = 55): 
  x = 60 := 
by 
  sorry

end production_today_l200_200870


namespace prime_neighbor_divisible_by_6_l200_200639

theorem prime_neighbor_divisible_by_6 (p : ℕ) (h_prime: Prime p) (h_gt3: p > 3) :
  ∃ k : ℕ, k ≠ 0 ∧ ((p - 1) % 6 = 0 ∨ (p + 1) % 6 = 0) :=
by
  sorry

end prime_neighbor_divisible_by_6_l200_200639


namespace find_multiplier_l200_200301

theorem find_multiplier 
  (x : ℝ)
  (number : ℝ)
  (condition1 : 4 * number + x * number = 55)
  (condition2 : number = 5.0) :
  x = 7 :=
by
  sorry

end find_multiplier_l200_200301


namespace loisa_saves_70_l200_200118

def tablet_cash_price : ℕ := 450
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 40
def next_4_months_payment : ℕ := 35
def last_4_months_payment : ℕ := 30
def total_installment_payment : ℕ := down_payment + (4 * first_4_months_payment) + (4 * next_4_months_payment) + (4 * last_4_months_payment)
def savings : ℕ := total_installment_payment - tablet_cash_price

theorem loisa_saves_70 : savings = 70 := by
  sorry

end loisa_saves_70_l200_200118


namespace gcd_102_238_eq_34_l200_200215

theorem gcd_102_238_eq_34 :
  Int.gcd 102 238 = 34 :=
sorry

end gcd_102_238_eq_34_l200_200215


namespace solve_for_x_l200_200561

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l200_200561


namespace square_root_value_l200_200808

-- Define the problem conditions
def x : ℝ := 5

-- Prove the solution
theorem square_root_value : (Real.sqrt (x - 3)) = Real.sqrt 2 :=
by
  -- Proof steps skipped
  sorry

end square_root_value_l200_200808


namespace students_not_taken_test_l200_200326

theorem students_not_taken_test 
  (num_enrolled : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (answered_both : ℕ) 
  (H_num_enrolled : num_enrolled = 40) 
  (H_answered_q1 : answered_q1 = 30) 
  (H_answered_q2 : answered_q2 = 29) 
  (H_answered_both : answered_both = 29) : 
  num_enrolled - (answered_q1 + answered_q2 - answered_both) = 10 :=
by {
  sorry
}

end students_not_taken_test_l200_200326


namespace fragment_probability_l200_200392

noncomputable def probability_fragment_in_21_digit_code : ℚ :=
  (12 * 10^11 - 30) / 10^21

theorem fragment_probability:
  ∀ (code : Fin 10 → Fin 21 → Fin 10),
  (∃ (i : Fin 12), ∀ (j : Fin 10), code (i + j) = j) → 
  probability_fragment_in_21_digit_code = (12 * 10^11 - 30) / 10^21 :=
sorry

end fragment_probability_l200_200392


namespace librarians_all_work_together_l200_200323

/-- Peter works every 5 days -/
def Peter_days := 5

/-- Quinn works every 8 days -/
def Quinn_days := 8

/-- Rachel works every 10 days -/
def Rachel_days := 10

/-- Sam works every 14 days -/
def Sam_days := 14

/-- Least common multiple of the intervals at which Peter, Quinn, Rachel, and Sam work -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem librarians_all_work_together : LCM (LCM (LCM Peter_days Quinn_days) Rachel_days) Sam_days = 280 :=
  by
  sorry

end librarians_all_work_together_l200_200323


namespace pythagorean_theorem_example_l200_200670

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 8
noncomputable def c : ℕ := 10

theorem pythagorean_theorem_example :
  c = Real.sqrt (a^2 + b^2) := 
by
  sorry

end pythagorean_theorem_example_l200_200670


namespace milk_transfer_proof_l200_200994

theorem milk_transfer_proof :
  ∀ (A B C x : ℝ), 
  A = 1232 →
  B = A - 0.625 * A → 
  C = A - B → 
  B + x = C - x → 
  x = 154 :=
by
  intros A B C x hA hB hC hEqual
  sorry

end milk_transfer_proof_l200_200994


namespace share_of_y_l200_200716

-- Define the conditions as hypotheses
variables (n : ℝ) (x y z : ℝ)

-- The main theorem we need to prove
theorem share_of_y (h1 : x = n) 
                   (h2 : y = 0.45 * n) 
                   (h3 : z = 0.50 * n) 
                   (h4 : x + y + z = 78) : 
  y = 18 :=
by 
  -- insert proof here (not required as per instructions)
  sorry

end share_of_y_l200_200716


namespace part_a_l200_200364

theorem part_a {d m b : ℕ} (h_d : d = 41) (h_m : m = 28) (h_b : b = 15) :
    d - b + m - b + b = 54 :=
  by sorry

end part_a_l200_200364


namespace inconsistent_equation_system_l200_200804

variables {a x c : ℝ}

theorem inconsistent_equation_system (h1 : (a + x) / 2 = 110) (h2 : (x + c) / 2 = 170) (h3 : a - c = 120) : false :=
by
  sorry

end inconsistent_equation_system_l200_200804


namespace find_number_l200_200715

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l200_200715


namespace min_value_of_a3_l200_200885

open Real

theorem min_value_of_a3 (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n) (hgeo : ∀ n, a (n + 1) / a n = a 1 / a 0)
    (h : a 1 * a 2 * a 3 = a 1 + a 2 + a 3) : a 2 ≥ sqrt 3 := by {
  sorry
}

end min_value_of_a3_l200_200885


namespace third_side_length_l200_200774

theorem third_side_length (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < x) (h₄ : x < 11) : x = 6 :=
sorry

end third_side_length_l200_200774


namespace transform_equation_l200_200577

theorem transform_equation (x y : ℝ) (h : y = x + x⁻¹) :
  x^4 + x^3 - 5 * x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 := 
sorry

end transform_equation_l200_200577


namespace can_invent_1001_sad_stories_l200_200618

-- Definitions
def is_natural (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 17

def is_sad_story (a b c : ℕ) : Prop :=
  ∀ x y : ℤ, a * x + b * y ≠ c

-- The Statement
theorem can_invent_1001_sad_stories :
  ∃ stories : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ stories → is_natural a ∧ is_natural b ∧ is_natural c ∧ is_sad_story a b c) ∧
    stories.card ≥ 1001 :=
by
  sorry

end can_invent_1001_sad_stories_l200_200618


namespace three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l200_200082

theorem three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one (n : ℕ) (h : n > 1) : ¬(2^n - 1) ∣ (3^n - 1) :=
sorry

end three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l200_200082


namespace train_length_l200_200767

theorem train_length :
  ∃ L : ℝ, 
    (∀ V : ℝ, V = L / 24 ∧ V = (L + 650) / 89) → 
    L = 240 :=
by
  sorry

end train_length_l200_200767


namespace nonagon_diagonals_l200_200946

-- Define the number of sides for a nonagon.
def n : ℕ := 9

-- Define the formula for the number of diagonals in a polygon.
def D (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem to prove that the number of diagonals in a nonagon is 27.
theorem nonagon_diagonals : D n = 27 := by
  sorry

end nonagon_diagonals_l200_200946


namespace tan_theta_eq_neg3_then_expr_eq_5_div_2_l200_200945

theorem tan_theta_eq_neg3_then_expr_eq_5_div_2
  (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5 / 2 := 
sorry

end tan_theta_eq_neg3_then_expr_eq_5_div_2_l200_200945


namespace sally_out_of_pocket_cost_l200_200733

/-- Definitions of the given conditions -/
def given_money : Int := 320
def cost_per_book : Int := 15
def number_of_students : Int := 35

/-- Theorem to prove the amount Sally needs to pay out of pocket -/
theorem sally_out_of_pocket_cost : 
  let total_cost := number_of_students * cost_per_book
  let amount_given := given_money
  let out_of_pocket_cost := total_cost - amount_given
  out_of_pocket_cost = 205 := by
  sorry

end sally_out_of_pocket_cost_l200_200733


namespace range_of_a_l200_200718

def f (x : ℝ) : ℝ := 3 * x * |x|

theorem range_of_a : {a : ℝ | f (1 - a) + f (2 * a) < 0 } = {a : ℝ | a < -1} :=
by
  sorry

end range_of_a_l200_200718


namespace base_b_addition_correct_base_b_l200_200851

theorem base_b_addition (b : ℕ) (hb : b > 5) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 :=
  by
    sorry

theorem correct_base_b : ∃ (b : ℕ), b > 5 ∧ 
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 ∧
  (4 + 5 = b + 1) ∧
  (2 + 1 + 1 = 4) :=
  ⟨8, 
   by decide,
   base_b_addition 8 (by decide),
   by decide,
   by decide⟩ 

end base_b_addition_correct_base_b_l200_200851


namespace hyperbola_foci_on_x_axis_l200_200169

theorem hyperbola_foci_on_x_axis (a : ℝ) 
  (h1 : 1 - a < 0)
  (h2 : a - 3 > 0)
  (h3 : ∀ c, c = 2 → 2 * c = 4) : 
  a = 4 := 
sorry

end hyperbola_foci_on_x_axis_l200_200169


namespace ordered_pairs_unique_solution_l200_200988

theorem ordered_pairs_unique_solution :
  ∃! (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (b^2 - 4 * c = 0) ∧ (c^2 - 4 * b = 0) :=
sorry

end ordered_pairs_unique_solution_l200_200988


namespace min_value_frac_gcd_l200_200554

theorem min_value_frac_gcd {N k : ℕ} (hN_substring : N % 10^5 = 11235) (hN_pos : 0 < N) (hk_pos : 0 < k) (hk_bound : 10^k > N) : 
  (10^k - 1) / Nat.gcd N (10^k - 1) = 89 :=
by
  -- proof goes here
  sorry

end min_value_frac_gcd_l200_200554


namespace volume_of_square_pyramid_l200_200057

theorem volume_of_square_pyramid (a r : ℝ) : 
  a > 0 → r > 0 → volume = (1 / 3) * a^2 * r :=
by 
    sorry

end volume_of_square_pyramid_l200_200057


namespace correct_grammatical_phrase_l200_200232

-- Define the conditions as lean definitions 
def number_of_cars_produced_previous_year : ℕ := sorry  -- number of cars produced in previous year
def number_of_cars_produced_2004 : ℕ := 3 * number_of_cars_produced_previous_year  -- number of cars produced in 2004

-- Define the theorem stating the correct phrase to describe the production numbers
theorem correct_grammatical_phrase : 
  (3 * number_of_cars_produced_previous_year = number_of_cars_produced_2004) → 
  ("three times as many cars" = "three times as many cars") := 
by
  sorry

end correct_grammatical_phrase_l200_200232


namespace min_u_value_l200_200623

theorem min_u_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x + 1 / x) * (y + 1 / (4 * y)) ≥ 25 / 8 :=
by
  sorry

end min_u_value_l200_200623


namespace winning_post_distance_l200_200849

theorem winning_post_distance (v_A v_B D : ℝ) (hvA : v_A = (5 / 3) * v_B) (head_start : 80 ≤ D) :
  (D / v_A = (D - 80) / v_B) → D = 200 :=
by
  sorry

end winning_post_distance_l200_200849


namespace books_not_read_l200_200880

theorem books_not_read (total_books read_books : ℕ) (h1 : total_books = 20) (h2 : read_books = 15) : total_books - read_books = 5 := by
  sorry

end books_not_read_l200_200880


namespace total_students_in_class_l200_200189

theorem total_students_in_class (R S : ℕ) (h1 : 2 + 12 + 14 + R = S) (h2 : 2 * S = 40 + 3 * R) : S = 44 :=
by
  sorry

end total_students_in_class_l200_200189


namespace meals_for_children_l200_200958

theorem meals_for_children (C : ℕ)
  (H1 : 70 * C = 70 * 45)
  (H2 : 70 * 45 = 2 * 45 * 35) :
  C = 90 :=
by
  sorry

end meals_for_children_l200_200958


namespace soccer_ball_cost_l200_200530

theorem soccer_ball_cost (x : ℝ) (soccer_balls basketballs : ℕ) 
  (soccer_ball_cost basketball_cost : ℝ) 
  (h1 : soccer_balls = 2 * basketballs)
  (h2 : 5000 = soccer_balls * soccer_ball_cost)
  (h3 : 4000 = basketballs * basketball_cost)
  (h4 : basketball_cost = soccer_ball_cost + 30)
  (eqn : 5000 / soccer_ball_cost = 2 * (4000 / basketball_cost)) :
  soccer_ball_cost = x :=
by
  sorry

end soccer_ball_cost_l200_200530


namespace find_b_l200_200681

theorem find_b (k a b : ℝ) (h1 : 1 + a + b = 3) (h2 : k = 3 + a) :
  b = 3 := 
sorry

end find_b_l200_200681


namespace coordinates_of_P_l200_200734

theorem coordinates_of_P (a : ℝ) (h : 2 * a - 6 = 0) : (2 * a - 6, a + 1) = (0, 4) :=
by 
  have ha : a = 3 := by linarith
  rw [ha]
  sorry

end coordinates_of_P_l200_200734


namespace prime_sum_l200_200437

theorem prime_sum (m n : ℕ) (hm : Prime m) (hn : Prime n) (h : 5 * m + 7 * n = 129) :
  m + n = 19 ∨ m + n = 25 := by
  sorry

end prime_sum_l200_200437


namespace reduced_price_per_kg_l200_200036

variable (P R Q : ℝ)

theorem reduced_price_per_kg :
  R = 0.75 * P →
  1200 = (Q + 5) * R →
  Q * P = 1200 →
  R = 60 :=
by
  intro h₁ h₂ h₃
  sorry

end reduced_price_per_kg_l200_200036


namespace geometric_sequence_const_k_l200_200584

noncomputable def sum_of_terms (n : ℕ) (k : ℤ) : ℤ := 3 * 2^n + k
noncomputable def a1 (k : ℤ) : ℤ := sum_of_terms 1 k
noncomputable def a2 (k : ℤ) : ℤ := sum_of_terms 2 k - sum_of_terms 1 k
noncomputable def a3 (k : ℤ) : ℤ := sum_of_terms 3 k - sum_of_terms 2 k

theorem geometric_sequence_const_k :
  (∀ (k : ℤ), (a1 k * a3 k = a2 k * a2 k) → k = -3) :=
by
  sorry

end geometric_sequence_const_k_l200_200584


namespace union_A_B_intersection_complement_A_B_l200_200439

open Set Real

noncomputable def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}
noncomputable def B : Set ℝ := {x : ℝ | abs (2 * x + 1) ≤ 1}

theorem union_A_B : A ∪ B = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by
  sorry

theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end union_A_B_intersection_complement_A_B_l200_200439


namespace triangle_inscribed_angle_l200_200784

theorem triangle_inscribed_angle 
  (y : ℝ)
  (arc_PQ arc_QR arc_RP : ℝ)
  (h1 : arc_PQ = 2 * y + 40)
  (h2 : arc_QR = 3 * y + 15)
  (h3 : arc_RP = 4 * y - 40)
  (h4 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_P : ℝ, angle_P = 64.995 := 
by 
  sorry

end triangle_inscribed_angle_l200_200784


namespace smallest_x_value_l200_200341

theorem smallest_x_value :
  ∃ x, (x ≠ 9) ∧ (∀ y, (y ≠ 9) → ((x^2 - x - 72) / (x - 9) = 3 / (x + 6)) → x ≤ y) ∧ x = -9 :=
by
  sorry

end smallest_x_value_l200_200341


namespace speed_of_stream_l200_200532

theorem speed_of_stream
  (D : ℝ) (v : ℝ)
  (h : D / (72 - v) = 2 * D / (72 + v)) :
  v = 24 := by
  sorry

end speed_of_stream_l200_200532


namespace problem_l200_200109

noncomputable def g (x : ℝ) : ℝ := 3^x + 2

theorem problem (x : ℝ) : g (x + 1) - g x = 2 * g x - 2 := sorry

end problem_l200_200109


namespace square_perimeter_increase_l200_200691

theorem square_perimeter_increase (s : ℝ) : (4 * (s + 2) - 4 * s) = 8 := 
by
  sorry

end square_perimeter_increase_l200_200691


namespace height_of_right_triangle_on_parabola_equals_one_l200_200049

theorem height_of_right_triangle_on_parabola_equals_one 
    (x0 x1 x2 : ℝ) 
    (h0 : x0 ≠ x1)
    (h1 : x0 ≠ x2) 
    (h2 : x1 ≠ x2) 
    (h3 : x0^2 = x1^2) 
    (h4 : x0^2 < x2^2):
    x2^2 - x0^2 = 1 := by
  sorry

end height_of_right_triangle_on_parabola_equals_one_l200_200049


namespace sammy_offer_l200_200773

-- Declaring the given constants and assumptions
def peggy_records : ℕ := 200
def bryan_interested_records : ℕ := 100
def bryan_uninterested_records : ℕ := 100
def bryan_interested_offer : ℕ := 6
def bryan_uninterested_offer : ℕ := 1
def sammy_offer_diff : ℕ := 100

-- The problem to be proved
theorem sammy_offer:
    ∃ S : ℝ, 
    (200 * S) - 
    (bryan_interested_records * bryan_interested_offer +
    bryan_uninterested_records * bryan_uninterested_offer) = sammy_offer_diff → 
    S = 4 :=
sorry

end sammy_offer_l200_200773


namespace theater_seat_count_l200_200012

theorem theater_seat_count (number_of_people : ℕ) (empty_seats : ℕ) (total_seats : ℕ) 
  (h1 : number_of_people = 532) 
  (h2 : empty_seats = 218) 
  (h3 : total_seats = number_of_people + empty_seats) : 
  total_seats = 750 := 
by 
  sorry

end theater_seat_count_l200_200012


namespace probability_exactly_half_red_balls_l200_200443

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) * p^k * (1 - p)^(n - k)

theorem probability_exactly_half_red_balls :
  binomial_probability 8 4 (1/2) = 35/128 :=
by
  sorry

end probability_exactly_half_red_balls_l200_200443


namespace S_is_positive_rationals_l200_200235

variable {S : Set ℚ}

-- Defining the conditions as axioms
axiom cond1 (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : (a + b ∈ S) ∧ (a * b ∈ S)
axiom cond2 {r : ℚ} : (r ∈ S) ∨ (-r ∈ S) ∨ (r = 0)

-- The theorem to prove
theorem S_is_positive_rationals : S = { r : ℚ | r > 0 } := sorry

end S_is_positive_rationals_l200_200235


namespace quadratic_discriminant_single_solution_l200_200871

theorem quadratic_discriminant_single_solution :
  ∃ (n : ℝ), (∀ x : ℝ, 9 * x^2 + n * x + 36 = 0 → x = (-n) / (2 * 9)) → n = 36 :=
by
  sorry

end quadratic_discriminant_single_solution_l200_200871


namespace initial_number_of_professors_l200_200163

theorem initial_number_of_professors (p : ℕ) :
  (∃ p, (6480 / p : ℚ) < (11200 / (p + 3) : ℚ) ∧ 
   6480 % p = 0 ∧ 11200 % (p + 3) = 0 ∧ p > 4) → 
  p = 5 := 
sorry

end initial_number_of_professors_l200_200163


namespace line_through_A_with_equal_intercepts_l200_200555

theorem line_through_A_with_equal_intercepts (x y : ℝ) (A : ℝ × ℝ) (hx : A = (2, 1)) :
  (∃ k : ℝ, x + y = k ∧ x + y - 3 = 0) ∨ (x - 2 * y = 0) :=
sorry

end line_through_A_with_equal_intercepts_l200_200555


namespace entry_exit_ways_l200_200072

theorem entry_exit_ways (n : ℕ) (h : n = 8) : n * (n - 1) = 56 :=
by {
  sorry
}

end entry_exit_ways_l200_200072


namespace domain_of_function_l200_200286

def function_domain : Set ℝ := { x : ℝ | x + 1 ≥ 0 ∧ 2 - x ≠ 0 }

theorem domain_of_function :
  function_domain = { x : ℝ | x ≥ -1 ∧ x ≠ 2 } :=
sorry

end domain_of_function_l200_200286


namespace least_positive_x_multiple_l200_200550

theorem least_positive_x_multiple (x : ℕ) : 
  (∃ k : ℕ, (2 * x + 41) = 53 * k) → 
  x = 6 :=
sorry

end least_positive_x_multiple_l200_200550


namespace Nikka_stamp_collection_l200_200721

theorem Nikka_stamp_collection (S : ℝ) 
  (h1 : 0.35 * S ≥ 0) 
  (h2 : 0.2 * S ≥ 0) 
  (h3 : 0 < S) 
  (h4 : 0.45 * S = 45) : S = 100 :=
sorry

end Nikka_stamp_collection_l200_200721


namespace xy_sum_is_one_l200_200216

theorem xy_sum_is_one (x y : ℤ) (h1 : 2021 * x + 2025 * y = 2029) (h2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 :=
by sorry

end xy_sum_is_one_l200_200216


namespace joy_can_choose_17_rods_for_quadrilateral_l200_200249

theorem joy_can_choose_17_rods_for_quadrilateral :
  ∃ (possible_rods : Finset ℕ), 
    possible_rods.card = 17 ∧
    ∀ rod ∈ possible_rods, 
      rod > 0 ∧ rod <= 30 ∧
      (rod ≠ 3 ∧ rod ≠ 7 ∧ rod ≠ 15) ∧
      (rod > 15 - (3 + 7)) ∧
      (rod < 3 + 7 + 15) :=
by
  sorry

end joy_can_choose_17_rods_for_quadrilateral_l200_200249


namespace sum_of_ages_l200_200997

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l200_200997


namespace range_of_m_l200_200431

theorem range_of_m (m : ℝ) (H : ∀ x, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) : m < -1 / 2 :=
sorry

end range_of_m_l200_200431


namespace apples_problem_l200_200097

variable (K A : ℕ)

theorem apples_problem (K A : ℕ) (h1 : K + (3 / 4) * K + 600 = 2600) (h2 : A + (3 / 4) * A + 600 = 2600) :
  K = 1142 ∧ A = 1142 :=
by
  sorry

end apples_problem_l200_200097


namespace range_of_a_l200_200081

-- Definitions of the conditions
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0
def q (x : ℝ) (a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0 ∧ a > 0

-- Statement of the theorem that proves the range of a
theorem range_of_a (x : ℝ) (a : ℝ) :
  (¬ (p x) → ¬ (q x a)) ∧ (¬ (q x a) → ¬ (p x)) → (a ≥ 9) :=
by
  sorry

end range_of_a_l200_200081


namespace min_marked_cells_l200_200241

theorem min_marked_cells (marking : Fin 15 → Fin 15 → Prop) :
  (∀ i : Fin 15, ∃ j : Fin 15, ∀ k : Fin 10, marking i (j + k % 15)) ∧
  (∀ j : Fin 15, ∃ i : Fin 15, ∀ k : Fin 10, marking (i + k % 15) j) →
  ∃s : Finset (Fin 15 × Fin 15), s.card = 20 ∧ ∀ i : Fin 15, (∃ j, (i, j) ∈ s ∨ (j, i) ∈ s) :=
sorry

end min_marked_cells_l200_200241


namespace garden_roller_length_l200_200751

noncomputable def length_of_garden_roller (d : ℝ) (A : ℝ) (revolutions : ℕ) (π : ℝ) : ℝ :=
  let r := d / 2
  let area_in_one_revolution := A / revolutions
  let L := area_in_one_revolution / (2 * π * r)
  L

theorem garden_roller_length :
  length_of_garden_roller 1.2 37.714285714285715 5 (22 / 7) = 2 := by
  sorry

end garden_roller_length_l200_200751


namespace sum_of_digits_1_to_1000_l200_200168

/--  sum_of_digits calculates the sum of digits of a given number n -/
def sum_of_digits(n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- sum_of_digits_in_range calculates the sum of the digits 
of all numbers in the inclusive range from 1 to m -/
def sum_of_digits_in_range (m : ℕ) : ℕ :=
  (Finset.range (m + 1)).sum sum_of_digits

theorem sum_of_digits_1_to_1000 : sum_of_digits_in_range 1000 = 13501 :=
by
  sorry

end sum_of_digits_1_to_1000_l200_200168


namespace factorize_expression_l200_200594

theorem factorize_expression (m : ℝ) : 
  4 * m^2 - 64 = 4 * (m + 4) * (m - 4) :=
sorry

end factorize_expression_l200_200594


namespace exists_integers_greater_than_N_l200_200385

theorem exists_integers_greater_than_N (N : ℝ) : 
  ∃ (x1 x2 x3 x4 : ℤ), (x1 > N) ∧ (x2 > N) ∧ (x3 > N) ∧ (x4 > N) ∧ 
  (x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4 = x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4) := 
sorry

end exists_integers_greater_than_N_l200_200385


namespace soccer_score_combinations_l200_200744

theorem soccer_score_combinations :
  ∃ (x y z : ℕ), x + y + z = 14 ∧ 3 * x + y = 19 ∧ x + y + z ≥ 0 ∧ 
    ({ (3, 10, 1), (4, 7, 3), (5, 4, 5), (6, 1, 7) } = 
      { (x, y, z) | x + y + z = 14 ∧ 3 * x + y = 19 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 }) :=
by 
  sorry

end soccer_score_combinations_l200_200744


namespace prime_polynomial_l200_200339

theorem prime_polynomial (n : ℕ) (h1 : 2 ≤ n)
  (h2 : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end prime_polynomial_l200_200339


namespace monotonic_increasing_interval_l200_200446

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x + 6)

theorem monotonic_increasing_interval : 
  ∀ x y : ℝ, x < y → y < 1 → f x < f y :=
by
  sorry

end monotonic_increasing_interval_l200_200446


namespace least_three_digit_multiple_of_3_4_5_l200_200484

def is_multiple_of (a b : ℕ) : Prop := b % a = 0

theorem least_three_digit_multiple_of_3_4_5 : 
  ∃ n : ℕ, is_multiple_of 3 n ∧ is_multiple_of 4 n ∧ is_multiple_of 5 n ∧ 100 ≤ n ∧ n < 1000 ∧ (∀ m : ℕ, is_multiple_of 3 m ∧ is_multiple_of 4 m ∧ is_multiple_of 5 m ∧ 100 ≤ m ∧ m < 1000 → n ≤ m) ∧ n = 120 :=
by
  sorry

end least_three_digit_multiple_of_3_4_5_l200_200484


namespace difference_of_sums_l200_200297

noncomputable def sum_of_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

noncomputable def sum_of_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem difference_of_sums : 
  sum_of_first_n_even 2004 - sum_of_first_n_odd 2003 = 6017 := 
by sorry

end difference_of_sums_l200_200297


namespace ratio_of_democrats_l200_200507

variable (F M D_F D_M : ℕ)

theorem ratio_of_democrats (h1 : F + M = 750)
    (h2 : D_F = 1 / 2 * F)
    (h3 : D_F = 125)
    (h4 : D_M = 1 / 4 * M) :
    (D_F + D_M) / 750 = 1 / 3 :=
sorry

end ratio_of_democrats_l200_200507


namespace jonah_profit_is_correct_l200_200927

noncomputable def jonah_profit : Real :=
  let pineapples := 6
  let pricePerPineapple := 3
  let pineappleCostWithoutDiscount := pineapples * pricePerPineapple
  let discount := if pineapples > 4 then 0.20 * pineappleCostWithoutDiscount else 0
  let totalCostAfterDiscount := pineappleCostWithoutDiscount - discount
  let ringsPerPineapple := 10
  let totalRings := pineapples * ringsPerPineapple
  let ringsSoldIndividually := 2
  let pricePerIndividualRing := 5
  let revenueFromIndividualRings := ringsSoldIndividually * pricePerIndividualRing
  let ringsLeft := totalRings - ringsSoldIndividually
  let ringsPerSet := 4
  let setsSold := ringsLeft / ringsPerSet -- This should be interpreted as an integer division
  let pricePerSet := 16
  let revenueFromSets := setsSold * pricePerSet
  let totalRevenue := revenueFromIndividualRings + revenueFromSets
  let profit := totalRevenue - totalCostAfterDiscount
  profit
  
theorem jonah_profit_is_correct :
  jonah_profit = 219.60 := by
  sorry

end jonah_profit_is_correct_l200_200927


namespace part_a_l200_200613

theorem part_a (n : ℕ) (hn : 0 < n) : 
  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end part_a_l200_200613


namespace equilateral_triangle_t_gt_a_squared_l200_200537

theorem equilateral_triangle_t_gt_a_squared {a x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ a) :
  2 * x^2 - 2 * a * x + 3 * a^2 > a^2 :=
by {
  sorry
}

end equilateral_triangle_t_gt_a_squared_l200_200537


namespace abs_diff_of_solutions_l200_200506

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l200_200506


namespace intersection_complement_l200_200622

def real_set_M : Set ℝ := {x | 1 < x}
def real_set_N : Set ℝ := {x | x > 4}

theorem intersection_complement (x : ℝ) : x ∈ (real_set_M ∩ (real_set_Nᶜ)) ↔ 1 < x ∧ x ≤ 4 :=
by
  sorry

end intersection_complement_l200_200622


namespace guilty_prob_l200_200798

-- Defining suspects
inductive Suspect
| A
| B
| C

open Suspect

-- Constants for the problem
def looks_alike (x y : Suspect) : Prop :=
(x = A ∧ y = B) ∨ (x = B ∧ y = A)

def timid (x : Suspect) : Prop :=
x = A ∨ x = B

def bold (x : Suspect) : Prop :=
x = C

def alibi_dover (x : Suspect) : Prop :=
x = A ∨ x = B

def needs_accomplice (x : Suspect) : Prop :=
timid x

def works_alone (x : Suspect) : Prop :=
bold x

def in_bar_during_robbery (x : Suspect) : Prop :=
x = A ∨ x = B

-- Theorem to be proved
theorem guilty_prob :
  ∃ x : Suspect, (x = B) ∧ ∀ y : Suspect, y ≠ B → 
    ((y = A ∧ timid y ∧ needs_accomplice y ∧ in_bar_during_robbery y) ∨
    (y = C ∧ bold y ∧ works_alone y)) :=
by
  sorry

end guilty_prob_l200_200798


namespace course_choice_gender_related_l200_200642
open scoped Real

theorem course_choice_gender_related :
  let a := 40 -- Males choosing Calligraphy
  let b := 10 -- Males choosing Paper Cutting
  let c := 30 -- Females choosing Calligraphy
  let d := 20 -- Females choosing Paper Cutting
  let n := a + b + c + d -- Total number of students
  let χ_squared := (n * (a*d - b*c)^2) / ((a+b) * (c+d) * (a+c) * (b+d))
  χ_squared > 3.841 := 
by
  sorry

end course_choice_gender_related_l200_200642


namespace min_value_my_function_l200_200171

noncomputable def my_function (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x - 2) + 3 * abs (x - 3) + 4 * abs (x - 4)

theorem min_value_my_function :
  ∃ (x : ℝ), my_function x = 8 ∧ (∀ y : ℝ, my_function y ≥ 8) :=
sorry

end min_value_my_function_l200_200171


namespace find_base_l200_200614
-- Import the necessary library

-- Define the conditions and the result
theorem find_base (x y b : ℕ) (h1 : x - y = 9) (h2 : x = 9) (h3 : b^x * 4^y = 19683) : b = 3 :=
by
  sorry

end find_base_l200_200614


namespace cupboard_cost_price_l200_200343

theorem cupboard_cost_price
  (C : ℝ)
  (h1 : ∀ (S : ℝ), S = 0.84 * C) -- Vijay sells a cupboard at 84% of the cost price.
  (h2 : ∀ (S_new : ℝ), S_new = 1.16 * C) -- If Vijay got Rs. 1200 more, he would have made a profit of 16%.
  (h3 : ∀ (S_new S : ℝ), S_new - S = 1200) -- The difference between new selling price and original selling price is Rs. 1200.
  : C = 3750 := 
sorry -- Proof is not required.

end cupboard_cost_price_l200_200343


namespace dissimilar_terms_expansion_count_l200_200209

noncomputable def num_dissimilar_terms_in_expansion (a b c d : ℝ) : ℕ :=
  let n := 8
  let k := 4
  Nat.choose (n + k - 1) (k - 1)

theorem dissimilar_terms_expansion_count : 
  num_dissimilar_terms_in_expansion a b c d = 165 := by
  sorry

end dissimilar_terms_expansion_count_l200_200209


namespace books_difference_l200_200456

theorem books_difference (maddie_books luisa_books amy_books total_books : ℕ) 
  (h1 : maddie_books = 15) 
  (h2 : luisa_books = 18) 
  (h3 : amy_books = 6) 
  (h4 : total_books = amy_books + luisa_books) :
  total_books - maddie_books = 9 := 
sorry

end books_difference_l200_200456


namespace find_y_l200_200287

theorem find_y (x y : ℤ) 
  (h1 : x^2 + 4 = y - 2) 
  (h2 : x = 6) : 
  y = 42 := 
by 
  sorry

end find_y_l200_200287


namespace count_even_numbers_between_250_and_600_l200_200161

theorem count_even_numbers_between_250_and_600 : 
  ∃ n : ℕ, (n = 175 ∧ 
    ∀ k : ℕ, (250 < 2 * k ∧ 2 * k ≤ 600) ↔ (126 ≤ k ∧ k ≤ 300)) :=
by
  sorry

end count_even_numbers_between_250_and_600_l200_200161


namespace largest_integer_less_than_100_div_8_rem_5_l200_200926

theorem largest_integer_less_than_100_div_8_rem_5 : ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n := by
  sorry

end largest_integer_less_than_100_div_8_rem_5_l200_200926


namespace find_missing_term_l200_200795

theorem find_missing_term (a b : ℕ) : ∃ x, (2 * a - b) * x = 4 * a^2 - b^2 :=
by
  use (2 * a + b)
  sorry

end find_missing_term_l200_200795


namespace one_third_of_nine_times_seven_l200_200540

theorem one_third_of_nine_times_seven : (1 / 3) * (9 * 7) = 21 := 
by
  sorry

end one_third_of_nine_times_seven_l200_200540


namespace postcards_per_day_l200_200039

variable (income_per_card total_income days : ℕ)
variable (x : ℕ)

theorem postcards_per_day
  (h1 : income_per_card = 5)
  (h2 : total_income = 900)
  (h3 : days = 6)
  (h4 : total_income = income_per_card * x * days) :
  x = 30 :=
by
  rw [h1, h2, h3] at h4
  linarith

end postcards_per_day_l200_200039


namespace distance_PF_l200_200423

-- Definitions for the given conditions
structure Rectangle :=
  (EF GH: ℝ)
  (interior_point : ℝ × ℝ)
  (PE : ℝ)
  (PH : ℝ)
  (PG : ℝ)

-- The theorem to prove PF equals 12 under the given conditions
theorem distance_PF 
  (r : Rectangle)
  (hPE : r.PE = 5)
  (hPH : r.PH = 12)
  (hPG : r.PG = 13) :
  ∃ PF, PF = 12 := 
sorry

end distance_PF_l200_200423


namespace find_difference_l200_200137

theorem find_difference (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.30 * y) : x - y = 10 := 
by
  sorry

end find_difference_l200_200137


namespace find_BA_prime_l200_200285

theorem find_BA_prime (BA BC A_prime C_1 : ℝ) 
  (h1 : BA = 3)
  (h2 : BC = 2)
  (h3 : A_prime < BA)
  (h4 : A_prime * C_1 = 3) : A_prime = 3 / 2 := 
by 
  sorry

end find_BA_prime_l200_200285


namespace arithmetic_sequence_sum_l200_200590

theorem arithmetic_sequence_sum :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
  (∀ n, a n = a 0 + n * d) ∧ 
  (∃ b c, b^2 - 6*b + 5 = 0 ∧ c^2 - 6*c + 5 = 0 ∧ a 3 = b ∧ a 15 = c) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l200_200590


namespace no_adjacent_teachers_l200_200568

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

theorem no_adjacent_teachers (students teachers : ℕ)
  (h_students : students = 4)
  (h_teachers : teachers = 3) :
  ∃ (arrangements : ℕ), arrangements = (factorial students) * (permutation (students + 1) teachers) :=
by
  sorry

end no_adjacent_teachers_l200_200568


namespace relationship_m_n_l200_200578

variable (a b : ℝ)
variable (m n : ℝ)

theorem relationship_m_n (h1 : a > b) (h2 : b > 0) (hm : m = Real.sqrt a - Real.sqrt b) (hn : n = Real.sqrt (a - b)) : m < n := sorry

end relationship_m_n_l200_200578


namespace find_numbers_l200_200435

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

noncomputable def number1 := 986
noncomputable def number2 := 689

theorem find_numbers :
  is_three_digit_number number1 ∧ is_three_digit_number number2 ∧
  hundreds_digit number1 = units_digit number2 ∧ hundreds_digit number2 = units_digit number1 ∧
  number1 - number2 = 297 ∧ (hundreds_digit number2 + tens_digit number2 + units_digit number2) = 23 :=
by
  sorry

end find_numbers_l200_200435


namespace each_child_plays_40_minutes_l200_200151

variable (TotalMinutes : ℕ)
variable (NumChildren : ℕ)
variable (ChildPairs : ℕ)

theorem each_child_plays_40_minutes (h1 : TotalMinutes = 120) 
                                    (h2 : NumChildren = 6) 
                                    (h3 : ChildPairs = 2) :
  (ChildPairs * TotalMinutes) / NumChildren = 40 :=
by
  sorry

end each_child_plays_40_minutes_l200_200151


namespace avg_consecutive_integers_l200_200324

theorem avg_consecutive_integers (a : ℝ) (b : ℝ) 
  (h₁ : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5)) / 6) :
  (a + 5) = (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5)) / 6 :=
by sorry

end avg_consecutive_integers_l200_200324


namespace fraction_area_outside_circle_l200_200160

theorem fraction_area_outside_circle (r : ℝ) (h1 : r > 0) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := π * r ^ 2
  let area_outside := area_square - area_circle
  (area_outside / area_square) = 1 - ↑π / 4 :=
by
  sorry

end fraction_area_outside_circle_l200_200160


namespace find_widgets_l200_200542

theorem find_widgets (a b c d e f : ℕ) : 
  (3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255) →
  (3 ^ a * 11 ^ b * 5 ^ c * 7 ^ d * 13 ^ e * 17 ^ f = 351125648000) →
  c = 3 :=
by
  sorry

end find_widgets_l200_200542


namespace rice_and_flour_bags_l200_200881

theorem rice_and_flour_bags (x : ℕ) (y : ℕ) 
  (h1 : x + y = 351)
  (h2 : x + 20 = 3 * (y - 50) + 1) : 
  x = 221 ∧ y = 130 :=
by
  sorry

end rice_and_flour_bags_l200_200881


namespace exponent_equality_l200_200078

theorem exponent_equality (n : ℕ) : 
    5^n = 5 * (5^2)^2 * (5^3)^3 → n = 14 := by
    sorry

end exponent_equality_l200_200078


namespace remaining_water_l200_200893

def initial_water : ℚ := 3
def water_used : ℚ := 4 / 3

theorem remaining_water : initial_water - water_used = 5 / 3 := 
by sorry -- skipping the proof for now

end remaining_water_l200_200893


namespace baba_yagas_savings_plan_l200_200116

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l200_200116


namespace debt_amount_is_40_l200_200191

theorem debt_amount_is_40 (l n t debt remaining : ℕ) (h_l : l = 6)
  (h_n1 : n = 5 * l) (h_n2 : n = 3 * t) (h_remaining : remaining = 6) 
  (h_share : ∀ x y z : ℕ, x = y ∧ y = z ∧ z = 2) :
  debt = 40 := 
by
  sorry

end debt_amount_is_40_l200_200191


namespace find_y_l200_200389

theorem find_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 18) : y = 5 :=
sorry

end find_y_l200_200389


namespace problem1_problem2_l200_200059

-- Problem (1)
theorem problem1 (f : ℝ → ℝ) (h : ∀ x ≠ 0, f (2 / x + 2) = x + 1) : 
  ∀ x ≠ 2, f x = x / (x - 2) :=
sorry

-- Problem (2)
theorem problem2 (f : ℝ → ℝ) (h : ∃ k b, ∀ x, f x = k * x + b ∧ k ≠ 0)
  (h' : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) :
  ∀ x, f x = 2 * x + 7 :=
sorry

end problem1_problem2_l200_200059


namespace wage_difference_l200_200289

theorem wage_difference (P Q H: ℝ) (h1: P = 1.5 * Q) (h2: P * H = 300) (h3: Q * (H + 10) = 300) : P - Q = 5 :=
by
  sorry

end wage_difference_l200_200289


namespace max_value_function_l200_200569

theorem max_value_function (x : ℝ) (h : x > 4) : -x + (1 / (4 - x)) ≤ -6 :=
sorry

end max_value_function_l200_200569


namespace range_of_2a_minus_b_l200_200615

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : 2 < b) (h4 : b < 4) : 
  -2 < 2 * a - b ∧ 2 * a - b < 4 := 
by 
  sorry

end range_of_2a_minus_b_l200_200615


namespace Beth_and_Jan_total_money_l200_200271

theorem Beth_and_Jan_total_money (B J : ℝ) 
  (h1 : B + 35 = 105)
  (h2 : J - 10 = B) : 
  B + J = 150 :=
by
  -- Proof omitted
  sorry

end Beth_and_Jan_total_money_l200_200271


namespace expand_and_simplify_l200_200076

theorem expand_and_simplify (x : ℝ) : 6 * (x - 3) * (x + 10) = 6 * x^2 + 42 * x - 180 :=
by
  sorry

end expand_and_simplify_l200_200076


namespace min_students_l200_200833

noncomputable def smallest_possible_number_of_students (b g : ℕ) : ℕ :=
if 3 * (3 * b) = 5 * (4 * g) then b + g else 0

theorem min_students (b g : ℕ) (h1 : 0 < b) (h2 : 0 < g) (h3 : 3 * (3 * b) = 5 * (4 * g)) :
  smallest_possible_number_of_students b g = 29 := sorry

end min_students_l200_200833


namespace M_geq_N_l200_200707

variable (x y : ℝ)
def M : ℝ := x^2 + y^2 + 1
def N : ℝ := x + y + x * y

theorem M_geq_N (x y : ℝ) : M x y ≥ N x y :=
by
sorry

end M_geq_N_l200_200707


namespace rectangle_sides_l200_200244

theorem rectangle_sides (a b : ℝ) (h1 : a < b) (h2 : a * b = 2 * a + 2 * b) : a < 4 ∧ b > 4 :=
by
  sorry

end rectangle_sides_l200_200244


namespace amount_saved_per_person_l200_200995

-- Definitions based on the conditions
def original_price := 60
def discounted_price := 48
def number_of_people := 3
def discount := original_price - discounted_price

-- Proving that each person paid 4 dollars less.
theorem amount_saved_per_person : discount / number_of_people = 4 :=
by
  sorry

end amount_saved_per_person_l200_200995


namespace intersection_A_B_l200_200787

-- Define sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 4 }
def B : Set ℝ := { 2, 3, 4, 5 }

-- State the theorem about the intersection A ∩ B
theorem intersection_A_B : A ∩ B = { 2, 3 } :=
by
  sorry

end intersection_A_B_l200_200787


namespace pranks_combinations_correct_l200_200863

noncomputable def pranks_combinations : ℕ := by
  let monday_choice := 1
  let tuesday_choice := 2
  let wednesday_choice := 4
  let thursday_choice := 5
  let friday_choice := 1
  let total_combinations := monday_choice * tuesday_choice * wednesday_choice * thursday_choice * friday_choice
  exact 40

theorem pranks_combinations_correct : pranks_combinations = 40 := by
  unfold pranks_combinations
  sorry -- Proof omitted

end pranks_combinations_correct_l200_200863


namespace birdhouse_volume_difference_l200_200533

-- Definitions to capture the given conditions
def sara_width_ft : ℝ := 1
def sara_height_ft : ℝ := 2
def sara_depth_ft : ℝ := 2

def jake_width_in : ℝ := 16
def jake_height_in : ℝ := 20
def jake_depth_in : ℝ := 18

-- Convert Sara's dimensions to inches
def ft_to_in (x : ℝ) : ℝ := x * 12
def sara_width_in := ft_to_in sara_width_ft
def sara_height_in := ft_to_in sara_height_ft
def sara_depth_in := ft_to_in sara_depth_ft

-- Volume calculations
def volume (width height depth : ℝ) := width * height * depth
def sara_volume := volume sara_width_in sara_height_in sara_depth_in
def jake_volume := volume jake_width_in jake_height_in jake_depth_in

-- The theorem to prove the difference in volume
theorem birdhouse_volume_difference : sara_volume - jake_volume = 1152 := by
  -- Proof goes here
  sorry

end birdhouse_volume_difference_l200_200533


namespace find_multiple_l200_200473

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_eq : m * n - 15 = 2 * n + 10) : m = 7 :=
by
  sorry

end find_multiple_l200_200473


namespace grace_charges_for_pulling_weeds_l200_200874

theorem grace_charges_for_pulling_weeds :
  (∃ (W : ℕ ), 63 * 6 + 9 * W + 10 * 9 = 567 → W = 11) :=
by
  use 11
  intro h
  sorry

end grace_charges_for_pulling_weeds_l200_200874


namespace quadratic_inequality_solution_l200_200964

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end quadratic_inequality_solution_l200_200964


namespace sum_of_radical_conjugates_l200_200599

theorem sum_of_radical_conjugates : 
  (8 - Real.sqrt 1369) + (8 + Real.sqrt 1369) = 16 :=
by
  sorry

end sum_of_radical_conjugates_l200_200599


namespace cos_of_angle_through_point_l200_200777

-- Define the point P and the angle α
def P : ℝ × ℝ := (4, 3)
def α : ℝ := sorry  -- α is an angle such that its terminal side passes through P

-- Define the squared distance from the origin to the point P
noncomputable def distance_squared : ℝ := P.1^2 + P.2^2

-- Define cos α
noncomputable def cosα : ℝ := P.1 / (Real.sqrt distance_squared)

-- State the theorem
theorem cos_of_angle_through_point : cosα = 4 / 5 := 
by sorry

end cos_of_angle_through_point_l200_200777


namespace connections_required_l200_200000

theorem connections_required (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end connections_required_l200_200000


namespace geom_seq_min_value_proof_l200_200652

noncomputable def geom_seq_min_value : ℝ := 3 / 2

theorem geom_seq_min_value_proof (a : ℕ → ℝ) (a1 : ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  a 2017 = a 2016 + 2 * a 2015 →
  a m * a n = 16 * a1^2 →
  (4 / m + 1 / n) = geom_seq_min_value :=
by {
  sorry
}

end geom_seq_min_value_proof_l200_200652


namespace determine_a_b_l200_200142

-- Definitions
def num (a b : ℕ) := 10000*a + 1000*6 + 100*7 + 10*9 + b

def divisible_by_72 (n : ℕ) : Prop := n % 72 = 0

noncomputable def a : ℕ := 3
noncomputable def b : ℕ := 2

-- Theorem statement
theorem determine_a_b : divisible_by_72 (num a b) :=
by
  -- The proof will be inserted here
  sorry

end determine_a_b_l200_200142


namespace quadratic_equation_only_option_B_l200_200493

theorem quadratic_equation_only_option_B (a b c : ℝ) (x : ℝ):
  (a ≠ 0 → (a * x^2 + b * x + c = 0)) ∧              -- Option A
  (3 * (x + 1)^2 = 2 * (x - 2) ↔ 3 * x^2 + 4 * x + 7 = 0) ∧  -- Option B
  (1 / x^2 + 1 = x^2 + 1 → False) ∧         -- Option C
  (1 / x^2 + 1 / x - 2 = 0 → False) →       -- Option D
  -- Option B is the only quadratic equation.
  (3 * (x + 1)^2 = 2 * (x - 2)) :=
sorry

end quadratic_equation_only_option_B_l200_200493


namespace min_q_difference_l200_200516

theorem min_q_difference (p q : ℕ) (hpq : 0 < p ∧ 0 < q) (ineq1 : (7:ℚ)/12 < p/q) (ineq2 : p/q < (5:ℚ)/8) (hmin : ∀ r s : ℕ, 0 < r ∧ 0 < s ∧ (7:ℚ)/12 < r/s ∧ r/s < (5:ℚ)/8 → q ≤ s) : q - p = 2 :=
sorry

end min_q_difference_l200_200516


namespace vampire_needs_7_gallons_per_week_l200_200410

-- Define conditions given in the problem
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Prove the vampire needs 7 gallons of blood per week to survive
theorem vampire_needs_7_gallons_per_week :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := 
by 
  sorry

end vampire_needs_7_gallons_per_week_l200_200410


namespace intersection_of_lines_l200_200099

theorem intersection_of_lines : ∃ x y : ℚ, y = 3 * x ∧ y - 5 = -7 * x ∧ x = 1 / 2 ∧ y = 3 / 2 :=
by
  sorry

end intersection_of_lines_l200_200099


namespace relationship_among_abc_l200_200723

noncomputable def a : ℝ := (1/4)^(1/2)
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := (1/3)^(1/2)

theorem relationship_among_abc : b > c ∧ c > a :=
by
  -- Proof will go here
  sorry

end relationship_among_abc_l200_200723


namespace average_difference_l200_200963

theorem average_difference :
  let avg1 := (10 + 30 + 50) / 3
  let avg2 := (20 + 40 + 6) / 3
  avg1 - avg2 = 8 := by
  sorry

end average_difference_l200_200963


namespace B_days_to_complete_job_alone_l200_200358

theorem B_days_to_complete_job_alone (x : ℝ) : 
  (1 / 15 + 1 / x) * 4 = 0.4666666666666667 → x = 20 :=
by
  intro h
  -- Note: The proof is omitted as we only need the statement here.
  sorry

end B_days_to_complete_job_alone_l200_200358


namespace initial_population_l200_200009

variable (P : ℕ)

theorem initial_population
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℚ := 1.2) :
  (P = 3000) :=
by
  sorry

end initial_population_l200_200009


namespace connie_start_marbles_l200_200093

variable (marbles_total marbles_given marbles_left : ℕ)

theorem connie_start_marbles :
  marbles_given = 73 → marbles_left = 70 → marbles_total = marbles_given + marbles_left → marbles_total = 143 :=
by intros; sorry

end connie_start_marbles_l200_200093


namespace eggs_left_in_jar_l200_200467

variable (initial_eggs : ℝ) (removed_eggs : ℝ)

theorem eggs_left_in_jar (h1 : initial_eggs = 35.3) (h2 : removed_eggs = 4.5) :
  initial_eggs - removed_eggs = 30.8 :=
by
  sorry

end eggs_left_in_jar_l200_200467


namespace players_started_first_half_l200_200014

variable (total_players : Nat)
variable (first_half_substitutions : Nat)
variable (second_half_substitutions : Nat)
variable (players_not_playing : Nat)

theorem players_started_first_half :
  total_players = 24 →
  first_half_substitutions = 2 →
  second_half_substitutions = 2 * first_half_substitutions →
  players_not_playing = 7 →
  let total_substitutions := first_half_substitutions + second_half_substitutions 
  let players_played := total_players - players_not_playing
  ∃ S, S + total_substitutions = players_played ∧ S = 11 := 
by
  sorry

end players_started_first_half_l200_200014


namespace primes_p_p2_p4_l200_200817

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem primes_p_p2_p4 (p : ℕ) (hp : is_prime p) (hp2 : is_prime (p + 2)) (hp4 : is_prime (p + 4)) :
  p = 3 :=
sorry

end primes_p_p2_p4_l200_200817


namespace probability_interval_l200_200139

/-- 
The probability of event A occurring is 4/5, the probability of event B occurring is 3/4,
and the probability of event C occurring is 2/3. The smallest interval necessarily containing
the probability q that all three events occur is [0, 2/3].
-/
theorem probability_interval (P_A P_B P_C q : ℝ)
  (hA : P_A = 4 / 5) (hB : P_B = 3 / 4) (hC : P_C = 2 / 3)
  (h_q_le_A : q ≤ P_A) (h_q_le_B : q ≤ P_B) (h_q_le_C : q ≤ P_C) :
  0 ≤ q ∧ q ≤ 2 / 3 := by
  sorry

end probability_interval_l200_200139


namespace prime_arithmetic_sequence_l200_200572

theorem prime_arithmetic_sequence {p1 p2 p3 d : ℕ} 
  (hp1 : Nat.Prime p1) 
  (hp2 : Nat.Prime p2) 
  (hp3 : Nat.Prime p3)
  (h3_p1 : 3 < p1)
  (h3_p2 : 3 < p2)
  (h3_p3 : 3 < p3)
  (h_seq1 : p2 = p1 + d)
  (h_seq2 : p3 = p1 + 2 * d) : 
  d % 6 = 0 :=
by sorry

end prime_arithmetic_sequence_l200_200572


namespace opposite_of_neg3_l200_200314

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end opposite_of_neg3_l200_200314


namespace inequality_solution_l200_200548

theorem inequality_solution (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := 
sorry

end inequality_solution_l200_200548


namespace total_people_at_fair_l200_200465

theorem total_people_at_fair (num_children : ℕ) (num_adults : ℕ) 
  (children_attended : num_children = 700) 
  (adults_attended : num_adults = 1500) : 
  num_children + num_adults = 2200 := by
  sorry

end total_people_at_fair_l200_200465


namespace additional_hours_q_l200_200495

variable (P Q : ℝ)

theorem additional_hours_q (h1 : P = 1.5 * Q) 
                           (h2 : P = Q + 8) 
                           (h3 : 480 / P = 20):
  (480 / Q) - (480 / P) = 10 :=
by
  sorry

end additional_hours_q_l200_200495


namespace minimize_at_five_halves_five_sixths_l200_200011

noncomputable def minimize_expression (x y : ℝ) : ℝ :=
  (y - 1)^2 + (x + y - 3)^2 + (2 * x + y - 6)^2

theorem minimize_at_five_halves_five_sixths (x y : ℝ) :
  minimize_expression x y = 1 / 6 ↔ (x = 5 / 2 ∧ y = 5 / 6) :=
sorry

end minimize_at_five_halves_five_sixths_l200_200011


namespace total_food_per_day_l200_200462

theorem total_food_per_day 
  (first_soldiers : ℕ)
  (second_soldiers : ℕ)
  (food_first_side_per_soldier : ℕ)
  (food_second_side_per_soldier : ℕ) :
  first_soldiers = 4000 →
  second_soldiers = first_soldiers - 500 →
  food_first_side_per_soldier = 10 →
  food_second_side_per_soldier = food_first_side_per_soldier - 2 →
  (first_soldiers * food_first_side_per_soldier + second_soldiers * food_second_side_per_soldier = 68000) :=
by
  intros h1 h2 h3 h4
  sorry

end total_food_per_day_l200_200462


namespace initial_price_of_phone_l200_200224

theorem initial_price_of_phone (P : ℝ) (h : 0.20 * P = 480) : P = 2400 :=
sorry

end initial_price_of_phone_l200_200224


namespace must_be_nonzero_l200_200382

noncomputable def Q (a b c d : ℝ) : ℝ → ℝ :=
  λ x => x^5 + a * x^4 + b * x^3 + c * x^2 + d * x

theorem must_be_nonzero (a b c d : ℝ)
  (h_roots : ∃ p q r s : ℝ, (∀ y : ℝ, Q a b c d y = 0 → y = 0 ∨ y = -1 ∨ y = p ∨ y = q ∨ y = r ∨ y = s) ∧ p ≠ 0 ∧ p ≠ -1 ∧ q ≠ 0 ∧ q ≠ -1 ∧ r ≠ 0 ∧ r ≠ -1 ∧ s ≠ 0 ∧ s ≠ -1)
  (h_distinct : (∀ x₁ x₂ : ℝ, Q a b c d x₁ = 0 ∧ Q a b c d x₂ = 0 → x₁ ≠ x₂ ∨ x₁ = x₂) → False)
  (h_f_zero : Q a b c d 0 = 0) :
  d ≠ 0 := by
  sorry

end must_be_nonzero_l200_200382


namespace rotated_curve_eq_l200_200593

theorem rotated_curve_eq :
  let θ := Real.pi / 4  -- Rotation angle 45 degrees in radians
  let cos_theta := Real.sqrt 2 / 2
  let sin_theta := Real.sqrt 2 / 2
  let x' := cos_theta * x - sin_theta * y
  let y' := sin_theta * x + cos_theta * y
  x + y^2 = 1 → x' ^ 2 + y' ^ 2 - 2 * x' * y' + Real.sqrt 2 * x' + Real.sqrt 2 * y' - 2 = 0 := 
sorry  -- Proof to be provided.

end rotated_curve_eq_l200_200593


namespace shadow_boundary_eqn_l200_200625

noncomputable def boundary_of_shadow (x : ℝ) : ℝ := x^2 / 10 - 1

theorem shadow_boundary_eqn (radius : ℝ) (center : ℝ × ℝ × ℝ) (light_source : ℝ × ℝ × ℝ) (x y: ℝ) :
  radius = 2 →
  center = (0, 0, 2) →
  light_source = (0, -2, 3) →
  y = boundary_of_shadow x :=
by
  intros hradius hcenter hlight
  sorry

end shadow_boundary_eqn_l200_200625


namespace neg_p_iff_exists_ge_zero_l200_200704

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + x + 1 < 0

theorem neg_p_iff_exists_ge_zero : ¬ p ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by 
   sorry

end neg_p_iff_exists_ge_zero_l200_200704


namespace reflected_parabola_equation_l200_200283

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := x^2

-- Define the line of reflection
def reflection_line (x : ℝ) : ℝ := x + 2

-- The reflected equation statement to be proved
theorem reflected_parabola_equation (x y : ℝ) :
  (parabola x = y) ∧ (reflection_line x = y) →
  (∃ y' x', x = y'^2 - 4 * y' + 2 ∧ y = x' + 2 ∧ x' = y - 2) :=
sorry

end reflected_parabola_equation_l200_200283


namespace F_2021_F_integer_F_divisibility_l200_200962

/- Part 1 -/
def F (n : ℕ) : ℕ := 
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  let n' := 1000 * c + 100 * d + 10 * a + b
  (n + n') / 101

theorem F_2021 : F 2021 = 41 :=
  sorry

/- Part 2 -/
theorem F_integer (a b c d : ℕ) (ha : 1 ≤ a) (hb : a ≤ 9) (hc : 0 ≤ b) (hd : b ≤ 9)
(hc' : 0 ≤ c) (hd' : c ≤ 9) (hc'' : 0 ≤ d) (hd'' : d ≤ 9) :
  let n := 1000 * a + 100 * b + 10 * c + d
  let n' := 1000 * c + 100 * d + 10 * a + b
  F n = (101 * (10 * a + b + 10 * c + d)) / 101 :=
  sorry

/- Part 3 -/
theorem F_divisibility (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 5 ≤ b ∧ b ≤ 9) :
  let s := 3800 + 10 * a + b
  let t := 1000 * b + 100 * a + 13
  (3 * F t - F s) % 8 = 0 ↔ s = 3816 ∨ s = 3847 ∨ s = 3829 :=
  sorry

end F_2021_F_integer_F_divisibility_l200_200962


namespace min_value_of_expression_l200_200252

noncomputable def minValueExpr (a b c : ℝ) : ℝ :=
  a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem min_value_of_expression (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minValueExpr a b c >= 60 :=
by
  sorry

end min_value_of_expression_l200_200252


namespace moles_of_H2O_formed_l200_200835

theorem moles_of_H2O_formed (moles_NH4NO3 moles_NaOH : ℕ) (percent_NaOH_reacts : ℝ)
  (h_decomposition : moles_NH4NO3 = 2) (h_NaOH : moles_NaOH = 2) 
  (h_percent : percent_NaOH_reacts = 0.85) : 
  (moles_NaOH * percent_NaOH_reacts = 1.7) :=
by
  sorry

end moles_of_H2O_formed_l200_200835


namespace lines_perpendicular_to_same_line_l200_200268

-- Definitions for lines and relationship types
structure Line := (name : String)
inductive RelType
| parallel 
| intersect
| skew

-- Definition stating two lines are perpendicular to the same line
def perpendicular_to_same_line (l1 l2 l3 : Line) : Prop :=
  -- (dot product or a similar condition could be specified, leaving abstract here)
  sorry

-- Theorem statement
theorem lines_perpendicular_to_same_line (l1 l2 l3 : Line) (h1 : perpendicular_to_same_line l1 l2 l3) : 
  RelType :=
by
  -- Proof to be filled in
  sorry

end lines_perpendicular_to_same_line_l200_200268


namespace equilibrium_constant_relationship_l200_200831

def given_problem (K1 K2 : ℝ) : Prop :=
  K2 = (1 / K1)^(1 / 2)

theorem equilibrium_constant_relationship (K1 K2 : ℝ) (h : given_problem K1 K2) :
  K1 = 1 / K2^2 :=
by sorry

end equilibrium_constant_relationship_l200_200831


namespace solution_to_inequality_l200_200474

-- Define the combination function C(n, k)
def combination (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the permutation function A(n, k)
def permutation (n k : ℕ) : ℕ :=
  n.factorial / (n - k).factorial

-- State the final theorem
theorem solution_to_inequality : 
  ∀ x : ℕ, (combination 5 x + permutation x 3 < 30) ↔ (x = 3 ∨ x = 4) :=
by
  -- The actual proof is not required as per the instructions
  sorry

end solution_to_inequality_l200_200474


namespace starting_current_ratio_l200_200350

theorem starting_current_ratio (running_current : ℕ) (units : ℕ) (total_current : ℕ)
    (h1 : running_current = 40) 
    (h2 : units = 3) 
    (h3 : total_current = 240) 
    (h4 : total_current = running_current * (units * starter_ratio)) :
    starter_ratio = 2 := 
sorry

end starting_current_ratio_l200_200350


namespace airlines_routes_l200_200051

open Function

theorem airlines_routes
  (n_regions m_regions : ℕ)
  (h_n_regions : n_regions = 18)
  (h_m_regions : m_regions = 10)
  (A B : Fin n_regions → Fin n_regions → Bool)
  (h_flight : ∀ r1 r2 : Fin n_regions, r1 ≠ r2 → (A r1 r2 = true ∨ B r1 r2 = true) ∧ ¬(A r1 r2 = true ∧ B r1 r2 = true)) :
  ∃ (routes_A routes_B : List (List (Fin n_regions))),
    (∀ route ∈ routes_A, 2 ∣ route.length) ∧
    (∀ route ∈ routes_B, 2 ∣ route.length) ∧
    routes_A ≠ [] ∧
    routes_B ≠ [] :=
sorry

end airlines_routes_l200_200051


namespace insphere_radius_l200_200239

theorem insphere_radius (V S : ℝ) (hV : V > 0) (hS : S > 0) : 
  ∃ r : ℝ, r = 3 * V / S := by
  sorry

end insphere_radius_l200_200239


namespace inequality_solution_set_l200_200672

theorem inequality_solution_set (a b x : ℝ) (h1 : a > 0) (h2 : b = a) : 
  ((a * x + b) * (x - 3) > 0 ↔ x < -1 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l200_200672


namespace number_of_correct_answers_l200_200840

theorem number_of_correct_answers (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 110) : c = 34 :=
by
  -- placeholder for proof
  sorry

end number_of_correct_answers_l200_200840


namespace decreased_value_l200_200884

noncomputable def original_expression (x y: ℝ) : ℝ :=
  x * y^2

noncomputable def decreased_expression (x y: ℝ) : ℝ :=
  (1 / 2) * x * (1 / 2 * y) ^ 2

theorem decreased_value (x y: ℝ) :
  decreased_expression x y = (1 / 8) * original_expression x y :=
by
  sorry

end decreased_value_l200_200884


namespace seeds_in_pots_l200_200246

theorem seeds_in_pots (x : ℕ) (total_seeds : ℕ) (seeds_fourth_pot : ℕ) 
  (h1 : total_seeds = 10) (h2 : seeds_fourth_pot = 1) 
  (h3 : 3 * x + seeds_fourth_pot = total_seeds) : x = 3 :=
by
  sorry

end seeds_in_pots_l200_200246


namespace n_cubed_minus_n_plus_one_is_square_l200_200556

theorem n_cubed_minus_n_plus_one_is_square (n : ℕ) (h : (n^5 + n^4 + 1).divisors.card = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
sorry

end n_cubed_minus_n_plus_one_is_square_l200_200556


namespace probability_five_dice_same_l200_200582

-- Define a function that represents the probability problem
noncomputable def probability_all_dice_same : ℚ :=
  (1 / 6) * (1 / 6) * (1 / 6) * (1 / 6)

-- The main theorem to state the proof problem
theorem probability_five_dice_same : probability_all_dice_same = 1 / 1296 :=
by
  sorry

end probability_five_dice_same_l200_200582


namespace compute_expression_l200_200960

theorem compute_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (h3 : x = 1 / z^2) : 
  (x - 1 / x) * (z^2 + 1 / z^2) = x^2 - z^4 :=
by
  sorry

end compute_expression_l200_200960


namespace graphs_intersect_once_l200_200882

variable {a b c d : ℝ}

theorem graphs_intersect_once 
(h1: ∃ x, (2 * a + 1 / (x - b)) = (2 * c + 1 / (x - d)) ∧ 
∃ y₁ y₂: ℝ, ∀ x, (2 * a + 1 / (x - b)) ≠ 2 * c + 1 / (x - d)) : 
∃ x, ((2 * b + 1 / (x - a)) = (2 * d + 1 / (x - c))) ∧ 
∃ y₁ y₂: ℝ, ∀ x, 2 * b + 1 / (x - a) ≠ 2 * d + 1 / (x - c) := 
sorry

end graphs_intersect_once_l200_200882


namespace statement_A_statement_B_statement_C_l200_200579

variable {a b : ℝ}
variable (ha : a > 0) (hb : b > 0)

theorem statement_A : (ab ≤ 1) → (1/a + 1/b ≥ 2) :=
by
  sorry

theorem statement_B : (a + b = 4) → (∀ x, (x = 1/a + 9/b) → (x ≥ 4)) :=
by
  sorry

theorem statement_C : (a^2 + b^2 = 4) → (ab ≤ 2) :=
by
  sorry

end statement_A_statement_B_statement_C_l200_200579


namespace multiple_of_15_bounds_and_difference_l200_200485

theorem multiple_of_15_bounds_and_difference :
  ∃ (n : ℕ), 15 * n ≤ 2016 ∧ 2016 < 15 * (n + 1) ∧ (15 * (n + 1) - 2016) = 9 :=
by
  sorry

end multiple_of_15_bounds_and_difference_l200_200485


namespace part_a_part_b_part_c_l200_200682

-- Part a
theorem part_a (n: ℕ) (h: n = 1): (n^2 - 5 * n + 4) / (n - 4) = 0 := by sorry

-- Part b
theorem part_b (n: ℕ) (h: (n^2 - 5 * n + 4) / (n - 4) = 5): n = 6 := 
  by sorry

-- Part c
theorem part_c (n: ℕ) (h : n ≠ 4): (n^2 - 5 * n + 4) / (n - 4) ≠ 3 := 
  by sorry

end part_a_part_b_part_c_l200_200682


namespace Jenny_has_6_cards_l200_200275

variable (J : ℕ)

noncomputable def Jenny_number := J
noncomputable def Orlando_number := J + 2
noncomputable def Richard_number := 3 * (J + 2)
noncomputable def Total_number := J + (J + 2) + 3 * (J + 2)

theorem Jenny_has_6_cards
  (h1 : Orlando_number J = J + 2)
  (h2 : Richard_number J = 3 * (J + 2))
  (h3 : Total_number J = 38) : J = 6 :=
by
  sorry

end Jenny_has_6_cards_l200_200275


namespace thought_number_is_24_l200_200973

variable (x : ℝ)

theorem thought_number_is_24 (h : x / 4 + 9 = 15) : x = 24 := by
  sorry

end thought_number_is_24_l200_200973


namespace loom_weaving_rate_l200_200362

noncomputable def total_cloth : ℝ := 27
noncomputable def total_time : ℝ := 210.9375

theorem loom_weaving_rate :
  (total_cloth / total_time) = 0.128 :=
by
  sorry

end loom_weaving_rate_l200_200362


namespace squats_day_after_tomorrow_l200_200737

theorem squats_day_after_tomorrow (initial_squats : ℕ) (daily_increase : ℕ) (today : ℕ) (tomorrow : ℕ) (day_after_tomorrow : ℕ)
  (h1 : initial_squats = 30)
  (h2 : daily_increase = 5)
  (h3 : today = initial_squats + daily_increase)
  (h4 : tomorrow = today + daily_increase)
  (h5 : day_after_tomorrow = tomorrow + daily_increase) : 
  day_after_tomorrow = 45 := 
sorry

end squats_day_after_tomorrow_l200_200737


namespace minimum_value_expression_l200_200792

theorem minimum_value_expression (F M N : ℝ × ℝ) (x y : ℝ) (a : ℝ) (k : ℝ) :
  (y ^ 2 = 16 * x ∧ F = (4, 0) ∧ l = (k * (x - 4), y) ∧ (M = (x₁, y₁) ∧ N = (x₂, y₂)) ∧
  0 ≤ x₁ ∧ y₁ ^ 2 = 16 * x₁ ∧ 0 ≤ x₂ ∧ y₂ ^ 2 = 16 * x₂) →
  (abs (dist F N) / 9 - 4 / abs (dist F M) ≥ 1 / 3) :=
sorry -- proof will be provided

end minimum_value_expression_l200_200792


namespace congruence_a_b_mod_1008_l200_200619

theorem congruence_a_b_mod_1008
  (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : a ^ b - b ^ a = 1008) : a ≡ b [MOD 1008] :=
sorry

end congruence_a_b_mod_1008_l200_200619


namespace bus_travel_time_kimovsk_moscow_l200_200975

noncomputable def travel_time_kimovsk_moscow (d1 d2 d3: ℝ) (max_speed: ℝ) (t_kt: ℝ) (t_nm: ℝ) : Prop :=
  35 ≤ d1 ∧ d1 ≤ 35 ∧
  60 ≤ d2 ∧ d2 ≤ 60 ∧
  200 ≤ d3 ∧ d3 ≤ 200 ∧
  max_speed <= 60 ∧
  2 ≤ t_kt ∧ t_kt ≤ 2 ∧
  5 ≤ t_nm ∧ t_nm ≤ 5 ∧
  (5 + 7/12 : ℝ) ≤ t_kt + t_nm ∧ t_kt + t_nm ≤ 6

theorem bus_travel_time_kimovsk_moscow
  (d1 d2 d3 : ℝ) (max_speed : ℝ) (t_kt : ℝ) (t_nm : ℝ) :
  travel_time_kimovsk_moscow d1 d2 d3 max_speed t_kt t_nm := 
by
  sorry

end bus_travel_time_kimovsk_moscow_l200_200975


namespace smallest_angle_of_triangle_l200_200004

theorem smallest_angle_of_triangle (x : ℝ) (h : 3 * x + 4 * x + 5 * x = 180) : 3 * x = 45 :=
by
  sorry

end smallest_angle_of_triangle_l200_200004


namespace triangle_perimeter_l200_200909

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 6) (h3 : c = 7) :
  a + b + c = 23 := by
  sorry

end triangle_perimeter_l200_200909


namespace complex_division_example_l200_200471

theorem complex_division_example : (2 : ℂ) / (I * (3 - I)) = (1 - 3 * I) / 5 := 
by {
  sorry
}

end complex_division_example_l200_200471


namespace value_of_z_l200_200505

theorem value_of_z (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x * y - 9) : z = 0 :=
by
  sorry

end value_of_z_l200_200505


namespace proof_solution_arithmetic_progression_l200_200164

noncomputable def system_has_solution (a b c m : ℝ) : Prop :=
  (m = 1 → a = b ∧ b = c) ∧
  (m = -2 → a + b + c = 0) ∧ 
  (m ≠ -2 ∧ m ≠ 1 → ∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c)

def abc_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem proof_solution_arithmetic_progression (a b c m : ℝ) : 
  system_has_solution a b c m → 
  (∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c ∧ 2 * y = x + z) ↔
  abc_arithmetic_progression a b c := 
by 
  sorry

end proof_solution_arithmetic_progression_l200_200164


namespace division_problem_l200_200911

theorem division_problem : 0.05 / 0.0025 = 20 := 
sorry

end division_problem_l200_200911


namespace expected_coins_basilio_20_l200_200320

noncomputable def binomialExpectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

noncomputable def expectedCoinsDifference : ℚ :=
  0.5

noncomputable def expectedCoinsBasilio (n : ℕ) (p : ℚ) : ℚ :=
  (binomialExpectation n p + expectedCoinsDifference) / 2

theorem expected_coins_basilio_20 :
  expectedCoinsBasilio 20 (1/2) = 5.25 :=
by
  -- here you would fill in the proof steps
  sorry

end expected_coins_basilio_20_l200_200320


namespace find_m_l200_200227

theorem find_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 :=
by
  intros h
  sorry

end find_m_l200_200227


namespace number_of_intersections_l200_200683

def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1
def vertical_line (x : ℝ) : Prop := x = 4

theorem number_of_intersections : 
    (∃ y : ℝ, ellipse 4 y ∧ vertical_line 4) ∧ 
    ∀ y1 y2, (ellipse 4 y1 ∧ vertical_line 4) → (ellipse 4 y2 ∧ vertical_line 4) → y1 = y2 :=
by
  sorry

end number_of_intersections_l200_200683


namespace correct_log_values_l200_200763

theorem correct_log_values (a b c : ℝ)
                          (log_027 : ℝ) (log_21 : ℝ) (log_1_5 : ℝ) (log_2_8 : ℝ)
                          (log_3 : ℝ) (log_5 : ℝ) (log_6 : ℝ) (log_7 : ℝ)
                          (log_8 : ℝ) (log_9 : ℝ) (log_14 : ℝ) :
  (log_3 = 2 * a - b) →
  (log_5 = a + c) →
  (log_6 = 1 + a - b - c) →
  (log_7 = 2 * (b + c)) →
  (log_9 = 4 * a - 2 * b) →
  (log_1_5 = 3 * a - b + c) →
  (log_14 = 1 - c + 2 * b) →
  (log_1_5 = 3 * a - b + c - 1) ∧ (log_7 = 2 * b + c) := sorry

end correct_log_values_l200_200763


namespace largest_power_of_2_that_divides_n_l200_200845

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_2_that_divides_n :
  ∃ k : ℕ, 2^k ∣ n ∧ ¬ (2^(k+1) ∣ n) ∧ k = 5 := sorry

end largest_power_of_2_that_divides_n_l200_200845


namespace distribution_ways_l200_200290

def count_distributions (n : ℕ) (k : ℕ) : ℕ :=
-- Calculation for count distributions will be implemented here
sorry

theorem distribution_ways (items bags : ℕ) (cond : items = 6 ∧ bags = 3):
  count_distributions items bags = 75 :=
by
  -- Proof would be implemented here
  sorry

end distribution_ways_l200_200290


namespace people_who_cannot_do_either_l200_200799

def people_total : ℕ := 120
def can_dance : ℕ := 88
def can_write_calligraphy : ℕ := 32
def can_do_both : ℕ := 18

theorem people_who_cannot_do_either : 
  people_total - (can_dance + can_write_calligraphy - can_do_both) = 18 := 
by
  sorry

end people_who_cannot_do_either_l200_200799


namespace chocolate_bar_cost_l200_200915

theorem chocolate_bar_cost :
  ∀ (total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips : ℕ),
  total = 150 →
  gummy_bear_cost = 2 →
  chocolate_chip_cost = 5 →
  num_chocolate_bars = 10 →
  num_gummy_bears = 10 →
  num_chocolate_chips = 20 →
  ((total - (num_gummy_bears * gummy_bear_cost + num_chocolate_chips * chocolate_chip_cost)) / num_chocolate_bars = 3) := 
by
  intros total gummy_bear_cost chocolate_chip_cost num_chocolate_bars num_gummy_bears num_chocolate_chips 
  intros htotal hgb_cost hcc_cost hncb hngb hncc
  sorry

end chocolate_bar_cost_l200_200915


namespace car_cost_l200_200217

/--
A group of six friends planned to buy a car. They plan to share the cost equally. 
They had a car wash to help raise funds, which would be taken out of the total cost. 
The remaining cost would be split between the six friends. At the car wash, they earn $500. 
However, Brad decided not to join in the purchase of the car, and now each friend has to pay $40 more. 
What is the cost of the car?
-/
theorem car_cost 
  (C : ℝ) 
  (h1 : 6 * ((C - 500) / 5) = 5 * (C / 6 + 40)) : 
  C = 4200 := 
by 
  sorry

end car_cost_l200_200217


namespace no_valid_n_lt_200_l200_200366

noncomputable def roots_are_consecutive (n m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * (k + 1) ∧ n = 2 * k + 1

theorem no_valid_n_lt_200 :
  ¬∃ n m : ℕ, n < 200 ∧
              m % 4 = 0 ∧
              ∃ t : ℕ, t^2 = m ∧
              roots_are_consecutive n m := 
by
  sorry

end no_valid_n_lt_200_l200_200366


namespace find_a5_l200_200925

variable (a_n : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 
  (h1 : is_arithmetic_sequence a_n d)
  (h2 : a_n 3 + a_n 8 = 22)
  (h3 : a_n 6 = 7) :
  a_n 5 = 15 :=
sorry

end find_a5_l200_200925


namespace fraction_neither_cell_phones_nor_pagers_l200_200310

theorem fraction_neither_cell_phones_nor_pagers
  (E : ℝ) -- total number of employees (E must be positive)
  (h1 : 0 < E)
  (frac_cell_phones : ℝ)
  (H1 : frac_cell_phones = (2 / 3))
  (frac_pagers : ℝ)
  (H2 : frac_pagers = (2 / 5))
  (frac_both : ℝ)
  (H3 : frac_both = 0.4) :
  (1 / 3) = (1 - frac_cell_phones - frac_pagers + frac_both) :=
by
  -- setup definitions, conditions and final proof
  sorry

end fraction_neither_cell_phones_nor_pagers_l200_200310


namespace prime_factorization_of_expression_l200_200475

theorem prime_factorization_of_expression (p n : ℕ) (hp : Nat.Prime p) (hdiv : p^2 ∣ 2^(p-1) - 1) : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) ∧ 
  a ∣ (p-1) ∧ b ∣ (p! + 2^n) ∧ c ∣ (p! + 2^n) := 
sorry

end prime_factorization_of_expression_l200_200475


namespace system_of_equations_solution_l200_200514

theorem system_of_equations_solution :
  ∃ x y : ℝ, (x + y = 3) ∧ (2 * x - 3 * y = 1) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end system_of_equations_solution_l200_200514


namespace find_x_l200_200487

theorem find_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  -- proof is not required, so we insert sorry
  sorry

end find_x_l200_200487


namespace count_ordered_pairs_l200_200061

theorem count_ordered_pairs : 
  ∃ n, n = 719 ∧ 
    (∀ (a b : ℕ), a + b = 1100 → 
      (∀ d ∈ [a, b], 
        ¬(∃ k : ℕ, d = 10 * k ∨ d % 10 = 0 ∨ d / 10 % 10 = 0 ∨ d % 5 = 0))) -> n = 719 :=
by
  sorry

end count_ordered_pairs_l200_200061


namespace min_value_of_a_plus_2b_l200_200155

theorem min_value_of_a_plus_2b (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_a_plus_2b_l200_200155


namespace actual_average_height_l200_200769

theorem actual_average_height (average_height : ℝ) (num_students : ℕ)
  (incorrect_heights actual_heights : Fin 3 → ℝ)
  (h_avg : average_height = 165)
  (h_num : num_students = 50)
  (h_incorrect : incorrect_heights 0 = 150 ∧ incorrect_heights 1 = 175 ∧ incorrect_heights 2 = 190)
  (h_actual : actual_heights 0 = 135 ∧ actual_heights 1 = 170 ∧ actual_heights 2 = 185) :
  (average_height * num_students 
   - (incorrect_heights 0 + incorrect_heights 1 + incorrect_heights 2) 
   + (actual_heights 0 + actual_heights 1 + actual_heights 2))
   / num_students = 164.5 :=
by
  -- proof steps here
  sorry

end actual_average_height_l200_200769


namespace recipe_butter_per_cup_l200_200604

theorem recipe_butter_per_cup (coconut_oil_to_butter_substitution : ℝ)
  (remaining_butter : ℝ)
  (planned_baking_mix : ℝ)
  (used_coconut_oil : ℝ)
  (butter_per_cup : ℝ)
  (h1 : coconut_oil_to_butter_substitution = 1)
  (h2 : remaining_butter = 4)
  (h3 : planned_baking_mix = 6)
  (h4 : used_coconut_oil = 8) :
  butter_per_cup = 4 / 3 := 
by 
  sorry

end recipe_butter_per_cup_l200_200604


namespace triangle_inequality_for_min_segments_l200_200414

theorem triangle_inequality_for_min_segments
  (a b c d : ℝ)
  (a1 b1 c1 : ℝ)
  (h1 : a1 = min a d)
  (h2 : b1 = min b d)
  (h3 : c1 = min c d)
  (h_triangle : c < a + b) :
  a1 + b1 > c1 ∧ a1 + c1 > b1 ∧ b1 + c1 > a1 := sorry

end triangle_inequality_for_min_segments_l200_200414


namespace conquering_Loulan_necessary_for_returning_home_l200_200768

theorem conquering_Loulan_necessary_for_returning_home : 
  ∀ (P Q : Prop), (¬ Q → ¬ P) → (P → Q) :=
by sorry

end conquering_Loulan_necessary_for_returning_home_l200_200768


namespace digit_sum_9_l200_200541

def digits := {n : ℕ // n < 10}

theorem digit_sum_9 (a b : digits) 
  (h1 : (4 * 100) + (a.1 * 10) + 3 + 984 = (1 * 1000) + (3 * 100) + (b.1 * 10) + 7) 
  (h2 : (1 + b.1) - (3 + 7) % 11 = 0) 
: a.1 + b.1 = 9 :=
sorry

end digit_sum_9_l200_200541


namespace tagged_fish_in_second_catch_l200_200758

theorem tagged_fish_in_second_catch 
  (total_fish : ℕ := 3200) 
  (initial_tagged : ℕ := 80) 
  (second_catch : ℕ := 80) 
  (T : ℕ) 
  (h : (T : ℚ) / second_catch = initial_tagged / total_fish) :
  T = 2 :=
by 
  sorry

end tagged_fish_in_second_catch_l200_200758


namespace runner_speed_ratio_l200_200208

noncomputable def speed_ratio (u1 u2 : ℝ) : ℝ := u1 / u2

theorem runner_speed_ratio (u1 u2 : ℝ) (h1 : u1 > u2) (h2 : u1 + u2 = 5) (h3 : u1 - u2 = 5/3) :
  speed_ratio u1 u2 = 2 :=
by
  sorry

end runner_speed_ratio_l200_200208


namespace xiaoming_xiaoqiang_common_visit_l200_200782

-- Define the initial visit dates and subsequent visit intervals
def xiaoming_initial_visit : ℕ := 3 -- The first Wednesday of January
def xiaoming_interval : ℕ := 4

def xiaoqiang_initial_visit : ℕ := 4 -- The first Thursday of January
def xiaoqiang_interval : ℕ := 3

-- Prove that the only common visit date is January 7
theorem xiaoming_xiaoqiang_common_visit : 
  ∃! d, (d < 32) ∧ ∃ n m, d = xiaoming_initial_visit + n * xiaoming_interval ∧ d = xiaoqiang_initial_visit + m * xiaoqiang_interval :=
  sorry

end xiaoming_xiaoqiang_common_visit_l200_200782


namespace compound_p_and_q_false_l200_200668

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1) /- The function y = a^x is monotonically decreasing. -/
def q : Prop := (a > 1/2) /- The function y = log(ax^2 - x + a) has the range R. -/

theorem compound_p_and_q_false : 
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) → (0 < a ∧ a ≤ 1/2) ∨ (a > 1) :=
by {
  -- this part will contain the proof steps, omitted here.
  sorry
}

end compound_p_and_q_false_l200_200668


namespace marcus_scored_50_percent_l200_200306

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l200_200306


namespace train_speed_approx_l200_200983

noncomputable def distance_in_kilometers (d : ℝ) : ℝ :=
d / 1000

noncomputable def time_in_hours (t : ℝ) : ℝ :=
t / 3600

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ :=
distance_in_kilometers d / time_in_hours t

theorem train_speed_approx (d t : ℝ) (h_d : d = 200) (h_t : t = 5.80598713393251) :
  abs (speed_in_kmh d t - 124.019) < 1e-3 :=
by
  rw [h_d, h_t]
  simp only [distance_in_kilometers, time_in_hours, speed_in_kmh]
  norm_num
  -- We're using norm_num to deal with numerical approximations and constants
  -- The actual calculations can be verified through manual checks or external tools but in Lean we skip this step.
  sorry

end train_speed_approx_l200_200983


namespace eq_solutions_count_l200_200068

def f (x a : ℝ) : ℝ := abs (abs (abs (x - a) - 1) - 1)

theorem eq_solutions_count (a b : ℝ) : 
  ∃ count : ℕ, (∀ x : ℝ, f x a = abs b → true) ∧ count = 4 :=
by
  sorry

end eq_solutions_count_l200_200068


namespace greatest_possible_difference_in_rectangles_area_l200_200080

theorem greatest_possible_difference_in_rectangles_area :
  ∃ (l1 w1 l2 w2 l3 w3 : ℤ),
    2 * l1 + 2 * w1 = 148 ∧
    2 * l2 + 2 * w2 = 150 ∧
    2 * l3 + 2 * w3 = 152 ∧
    (∃ (A1 A2 A3 : ℤ),
      A1 = l1 * w1 ∧
      A2 = l2 * w2 ∧
      A3 = l3 * w3 ∧
      (max (abs (A1 - A2)) (max (abs (A1 - A3)) (abs (A2 - A3))) = 1372)) :=
by
  sorry

end greatest_possible_difference_in_rectangles_area_l200_200080


namespace coral_must_read_pages_to_finish_book_l200_200110

theorem coral_must_read_pages_to_finish_book
  (total_pages first_week_read second_week_percentage pages_remaining first_week_left second_week_read : ℕ)
  (initial_pages_read : ℕ := total_pages / 2)
  (remaining_after_first_week : ℕ := total_pages - initial_pages_read)
  (read_second_week : ℕ := remaining_after_first_week * second_week_percentage / 100)
  (remaining_after_second_week : ℕ := remaining_after_first_week - read_second_week)
  (final_pages_to_read : ℕ := remaining_after_second_week):
  total_pages = 600 → first_week_read = 300 → second_week_percentage = 30 →
  pages_remaining = 300 → first_week_left = 300 → second_week_read = 90 →
  remaining_after_first_week = 300 - 300 →
  remaining_after_second_week = remaining_after_first_week - second_week_read →
  third_week_read = remaining_after_second_week →
  third_week_read = 210 := by
  sorry

end coral_must_read_pages_to_finish_book_l200_200110


namespace evaluate_polynomial_at_2_l200_200079

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 2) = 32 := 
by
  sorry

end evaluate_polynomial_at_2_l200_200079


namespace number_is_46000050_l200_200850

-- Define the corresponding place values for the given digit placements
def ten_million (n : ℕ) : ℕ := n * 10000000
def hundred_thousand (n : ℕ) : ℕ := n * 100000
def hundred (n : ℕ) : ℕ := n * 100

-- Define the specific numbers given in the conditions.
def digit_4 : ℕ := ten_million 4
def digit_60 : ℕ := hundred_thousand 6
def digit_500 : ℕ := hundred 5

-- Combine these values to form the number
def combined_number : ℕ := digit_4 + digit_60 + digit_500

-- The theorem, stating the number equals 46000050
theorem number_is_46000050 : combined_number = 46000050 := by
  sorry

end number_is_46000050_l200_200850


namespace rectangle_perimeter_l200_200015

open Real

def triangle_DEF_sides : ℝ × ℝ × ℝ := (9, 12, 15) -- sides of the triangle DEF

def rectangle_width : ℝ := 6 -- width of the rectangle

theorem rectangle_perimeter (a b c width : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : width = 6) :
  2 * (54 / width + width) = 30 :=
by
  sorry -- Proof is omitted as required

end rectangle_perimeter_l200_200015


namespace long_show_episodes_correct_l200_200038

variable {short_show_episodes : ℕ} {short_show_duration : ℕ} {total_watched_time : ℕ} {long_show_episode_duration : ℕ}

def episodes_long_show (short_episodes_duration total_duration long_episode_duration : ℕ) : ℕ :=
  (total_duration - short_episodes_duration) / long_episode_duration

theorem long_show_episodes_correct :
  ∀ (short_show_episodes short_show_duration total_watched_time long_show_episode_duration : ℕ),
  short_show_episodes = 24 →
  short_show_duration = 1 / 2 →
  total_watched_time = 24 →
  long_show_episode_duration = 1 →
  episodes_long_show (short_show_episodes * short_show_duration) total_watched_time long_show_episode_duration = 12 := by
  intros
  sorry

end long_show_episodes_correct_l200_200038


namespace smallest_possible_value_of_other_integer_l200_200991

theorem smallest_possible_value_of_other_integer (x b : ℕ) (h_gcd_lcm : ∀ m n : ℕ, m = 36 → gcd m n = x + 5 → lcm m n = x * (x + 5)) : 
  b > 0 → ∃ b, b = 1 ∧ gcd 36 b = x + 5 ∧ lcm 36 b = x * (x + 5) := 
by {
   sorry 
}

end smallest_possible_value_of_other_integer_l200_200991


namespace max_n_positive_l200_200451

theorem max_n_positive (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : S 15 > 0)
  (h2 : S 16 < 0)
  (hs1 : S 15 = 15 * (a 8))
  (hs2 : S 16 = 8 * (a 8 + a 9)) :
  (∀ n, a n > 0 → n ≤ 8) :=
by {
    sorry
}

end max_n_positive_l200_200451


namespace solve_abs_equation_l200_200956

theorem solve_abs_equation (x : ℝ) :
  |2 * x - 1| + |x - 2| = |x + 1| ↔ 1 / 2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end solve_abs_equation_l200_200956


namespace highest_qualification_number_possible_l200_200957

theorem highest_qualification_number_possible (n : ℕ) (qualifies : ℕ → ℕ → Prop)
    (h512 : n = 512)
    (hqualifies : ∀ a b, qualifies a b ↔ (a < b ∧ b - a ≤ 2)): 
    ∃ k, k = 18 ∧ (∀ m, qualifies m k → m < k) :=
by
  sorry

end highest_qualification_number_possible_l200_200957


namespace express_y_in_terms_of_x_l200_200654

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 1) : y = -3 * x + 1 := 
by
  sorry

end express_y_in_terms_of_x_l200_200654


namespace percentage_customers_return_books_l200_200127

theorem percentage_customers_return_books 
  (total_customers : ℕ) (price_per_book : ℕ) (sales_after_returns : ℕ) 
  (h1 : total_customers = 1000) 
  (h2 : price_per_book = 15) 
  (h3 : sales_after_returns = 9450) : 
  ((total_customers - (sales_after_returns / price_per_book)) / total_customers) * 100 = 37 := 
by
  sorry

end percentage_customers_return_books_l200_200127


namespace largest_3_digit_sum_l200_200377

theorem largest_3_digit_sum : ∃ A B : ℕ, A ≠ B ∧ A < 10 ∧ B < 10 ∧ 100 ≤ 111 * A + 12 * B ∧ 111 * A + 12 * B = 996 := by
  sorry

end largest_3_digit_sum_l200_200377


namespace dad_vacuum_time_l200_200430

theorem dad_vacuum_time (x : ℕ) (h1 : 2 * x + 5 = 27) (h2 : x + (2 * x + 5) = 38) :
  (2 * x + 5) = 27 := by
  sorry

end dad_vacuum_time_l200_200430


namespace find_fraction_l200_200588

theorem find_fraction (f : ℝ) (n : ℝ) (h : n = 180) (eqn : f * ((1 / 3) * (1 / 5) * n) + 6 = (1 / 15) * n) : f = 1 / 2 :=
by
  -- Definitions and assumptions provided above will be used here.
  sorry

end find_fraction_l200_200588


namespace simple_interest_proof_l200_200534

def simple_interest (P R T: ℝ) : ℝ :=
  P * R * T

theorem simple_interest_proof :
  simple_interest 810 (4.783950617283951 / 100) 4 = 154.80 :=
by
  sorry

end simple_interest_proof_l200_200534


namespace algebraic_expression_value_l200_200210

theorem algebraic_expression_value 
  (x1 x2 : ℝ)
  (h1 : x1^2 - x1 - 2022 = 0)
  (h2 : x2^2 - x2 - 2022 = 0) :
  x1^3 - 2022 * x1 + x2^2 = 4045 :=
by 
  sorry

end algebraic_expression_value_l200_200210


namespace trapezoid_segment_length_l200_200696

theorem trapezoid_segment_length (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a^2 + b^2) / 2) :=
sorry

end trapezoid_segment_length_l200_200696


namespace find_sum_u_v_l200_200900

theorem find_sum_u_v : ∃ (u v : ℚ), 5 * u - 6 * v = 35 ∧ 3 * u + 5 * v = -10 ∧ u + v = -40 / 43 :=
by
  sorry

end find_sum_u_v_l200_200900


namespace solution_set_non_empty_iff_l200_200641

theorem solution_set_non_empty_iff (a : ℝ) : (∃ x : ℝ, |x - 1| + |x + 2| < a) ↔ (a > 3) := 
sorry

end solution_set_non_empty_iff_l200_200641


namespace probability_passing_exam_l200_200313

-- Define probabilities for sets A, B, and C, and passing conditions
def P_A := 0.3
def P_B := 0.3
def P_C := 1 - P_A - P_B
def P_D_given_A := 0.8
def P_D_given_B := 0.6
def P_D_given_C := 0.8

-- Total probability of passing
def P_D := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

-- Proof that the total probability of passing is 0.74
theorem probability_passing_exam : P_D = 0.74 :=
by
  -- (skip the proof steps)
  sorry

end probability_passing_exam_l200_200313


namespace cube_edge_length_l200_200338

theorem cube_edge_length
  (length_base : ℝ) (width_base : ℝ) (rise_level : ℝ) (volume_displaced : ℝ) (volume_cube : ℝ) (edge_length : ℝ)
  (h_base : length_base = 20) (h_width : width_base = 15) (h_rise : rise_level = 3.3333333333333335)
  (h_volume_displaced : volume_displaced = length_base * width_base * rise_level)
  (h_volume_cube : volume_cube = volume_displaced)
  (h_edge_length_eq : volume_cube = edge_length ^ 3)
  : edge_length = 10 :=
by
  sorry

end cube_edge_length_l200_200338


namespace find_x_l200_200653

theorem find_x (x : ℚ) (h1 : 3 * x + (4 * x - 10) = 90) : x = 100 / 7 :=
by {
  sorry
}

end find_x_l200_200653


namespace expected_value_coin_flip_l200_200759

def probability_heads : ℚ := 2 / 3
def probability_tails : ℚ := 1 / 3
def gain_heads : ℤ := 5
def loss_tails : ℤ := -9

theorem expected_value_coin_flip : (2 / 3 : ℚ) * 5 + (1 / 3 : ℚ) * (-9) = 1 / 3 :=
by sorry

end expected_value_coin_flip_l200_200759


namespace circle_tangent_line_l200_200237

theorem circle_tangent_line 
    (center : ℝ × ℝ) (line_eq : ℝ → ℝ → ℝ) 
    (tangent_eq : ℝ) :
    center = (-1, 1) →
    line_eq 1 (-1)= 0 →
    tangent_eq = 2 :=
  let h := -1;
  let k := 1;
  let radius := Real.sqrt 2;
  sorry

end circle_tangent_line_l200_200237


namespace min_value_of_f_in_interval_l200_200145

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) ∧ f x = -Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_in_interval_l200_200145


namespace find_point_A_l200_200821

theorem find_point_A :
  (∃ A : ℤ, A + 2 = -2) ∨ (∃ A : ℤ, A - 2 = -2) → (∃ A : ℤ, A = 0 ∨ A = -4) :=
by
  sorry

end find_point_A_l200_200821


namespace washer_total_cost_l200_200818

variable (C : ℝ)
variable (h : 0.25 * C = 200)

theorem washer_total_cost : C = 800 :=
by
  sorry

end washer_total_cost_l200_200818


namespace regular_polygon_interior_angle_160_l200_200523

theorem regular_polygon_interior_angle_160 (n : ℕ) (h : 160 * n = 180 * (n - 2)) : n = 18 :=
by {
  sorry
}

end regular_polygon_interior_angle_160_l200_200523


namespace gcd_1729_1337_l200_200699

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := 
by
  sorry

end gcd_1729_1337_l200_200699


namespace sale_on_day_five_l200_200740

def sale1 : ℕ := 435
def sale2 : ℕ := 927
def sale3 : ℕ := 855
def sale6 : ℕ := 741
def average_sale : ℕ := 625
def total_days : ℕ := 5

theorem sale_on_day_five : 
  average_sale * total_days - (sale1 + sale2 + sale3 + sale6) = 167 :=
by
  sorry

end sale_on_day_five_l200_200740


namespace exists_rationals_leq_l200_200892

theorem exists_rationals_leq (f : ℚ → ℤ) : ∃ a b : ℚ, (f a + f b) / 2 ≤ f (a + b) / 2 :=
by
  sorry

end exists_rationals_leq_l200_200892


namespace ratio_H_G_l200_200810

theorem ratio_H_G (G H : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (G / (x + 3) + H / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x^3 + x^2 - 15 * x))) :
    H / G = 64 :=
sorry

end ratio_H_G_l200_200810


namespace angle_C_modified_l200_200998

theorem angle_C_modified (A B C : ℝ) (h_eq_triangle: A = B) (h_C_modified: C = A + 40) (h_sum_angles: A + B + C = 180) : 
  C = 86.67 := 
by 
  sorry

end angle_C_modified_l200_200998


namespace parabola_focus_hyperbola_equation_l200_200238

-- Problem 1
theorem parabola_focus (p : ℝ) (h₀ : p > 0) (h₁ : 2 * p - 0 - 4 = 0) : p = 2 :=
by
  sorry

-- Problem 2
theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : b / a = 3 / 4) (h₃ : a^2 / a = 16 / 5) (h₄ : a^2 + b^2 = 1) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
by
  sorry

end parabola_focus_hyperbola_equation_l200_200238


namespace find_k_value_l200_200791

theorem find_k_value (k : ℝ) (h₁ : ∀ x, k * x^2 - 5 * x - 12 = 0 → (x = 3 ∨ x = -4 / 3)) : k = 3 :=
sorry

end find_k_value_l200_200791


namespace count_valid_A_l200_200255

theorem count_valid_A : 
  ∃! (count : ℕ), count = 4 ∧ ∀ A : ℕ, (1 ≤ A ∧ A ≤ 9) → 
  (∃ x1 x2 : ℕ, x1 + x2 = 2 * A + 1 ∧ x1 * x2 = 2 * A ∧ x1 > 0 ∧ x2 > 0) → A = 1 ∨ A = 2 ∨ A = 3 ∨ A = 4 :=
sorry

end count_valid_A_l200_200255


namespace find_width_of_chalkboard_l200_200266

variable (w : ℝ) (l : ℝ)

-- Given conditions
def length_eq_twice_width (w l : ℝ) : Prop := l = 2 * w
def area_eq_eighteen (w l : ℝ) : Prop := w * l = 18

-- Theorem statement
theorem find_width_of_chalkboard (h1 : length_eq_twice_width w l) (h2 : area_eq_eighteen w l) : w = 3 :=
by sorry

end find_width_of_chalkboard_l200_200266


namespace min_n_constant_term_exists_l200_200504

theorem min_n_constant_term_exists (n : ℕ) (h : 0 < n) :
  (∃ r : ℕ, (2 * n = 3 * r) ∧ n > 0) ↔ n = 3 :=
by
  sorry

end min_n_constant_term_exists_l200_200504


namespace find_total_salary_l200_200298

noncomputable def total_salary (salary_left : ℕ) : ℚ :=
  salary_left * (120 / 19)

theorem find_total_salary
  (food : ℚ) (house_rent : ℚ) (clothes : ℚ) (transport : ℚ) (remaining : ℕ) :
  food = 1 / 4 →
  house_rent = 1 / 8 →
  clothes = 3 / 10 →
  transport = 1 / 6 →
  remaining = 35000 →
  total_salary remaining = 210552.63 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_total_salary_l200_200298


namespace max_books_single_student_l200_200318

theorem max_books_single_student (total_students : ℕ) (students_0_books : ℕ) (students_1_book : ℕ) (students_2_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 20 →
  students_0_books = 3 →
  students_1_book = 9 →
  students_2_books = 4 →
  avg_books_per_student = 2 →
  ∃ max_books : ℕ, max_books = 14 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_books_single_student_l200_200318


namespace total_price_of_houses_l200_200113

theorem total_price_of_houses (price_first price_second total_price : ℝ)
    (h1 : price_first = 200000)
    (h2 : price_second = 2 * price_first)
    (h3 : total_price = price_first + price_second) :
  total_price = 600000 := by
  sorry

end total_price_of_houses_l200_200113


namespace bridge_toll_fees_for_annie_are_5_l200_200531

-- Conditions
def start_fee : ℝ := 2.50
def cost_per_mile : ℝ := 0.25
def mike_miles : ℕ := 36
def annie_miles : ℕ := 16
def total_cost_mike : ℝ := start_fee + cost_per_mile * mike_miles

-- Hypothesis from conditions
axiom both_charged_same : ∀ (bridge_fees : ℝ), total_cost_mike = start_fee + cost_per_mile * annie_miles + bridge_fees

-- Proof problem
theorem bridge_toll_fees_for_annie_are_5 : ∃ (bridge_fees : ℝ), bridge_fees = 5 :=
by
  existsi 5
  sorry

end bridge_toll_fees_for_annie_are_5_l200_200531


namespace shop_width_l200_200433

theorem shop_width 
  (monthly_rent : ℝ) 
  (shop_length : ℝ) 
  (annual_rent_per_sqft : ℝ) 
  (width : ℝ) 
  (monthly_rent_eq : monthly_rent = 2244) 
  (shop_length_eq : shop_length = 22) 
  (annual_rent_per_sqft_eq : annual_rent_per_sqft = 68) 
  (width_eq : width = 18) : 
  (12 * monthly_rent) / annual_rent_per_sqft / shop_length = width := 
by 
  sorry

end shop_width_l200_200433


namespace max_cake_boxes_l200_200633

theorem max_cake_boxes 
  (L_carton W_carton H_carton : ℕ) (L_box W_box H_box : ℕ)
  (h_carton : L_carton = 25 ∧ W_carton = 42 ∧ H_carton = 60)
  (h_box : L_box = 8 ∧ W_box = 7 ∧ H_box = 5) : 
  (L_carton * W_carton * H_carton) / (L_box * W_box * H_box) = 225 := by 
  sorry

end max_cake_boxes_l200_200633


namespace factorial_equation_solution_l200_200398

theorem factorial_equation_solution (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → a = 3 ∧ b = 3 ∧ c = 4 := by
  sorry

end factorial_equation_solution_l200_200398


namespace express_114_as_ones_and_threes_with_min_ten_ones_l200_200726

theorem express_114_as_ones_and_threes_with_min_ten_ones :
  ∃n: ℕ, n = 35 ∧ ∃ x y : ℕ, x + 3 * y = 114 ∧ x ≥ 10 := sorry

end express_114_as_ones_and_threes_with_min_ten_ones_l200_200726


namespace problem_statement_l200_200803

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) :=
  ∀ x y, x ≤ y → f x ≤ f y

noncomputable def isOddFunction (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def isArithmeticSeq (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem problem_statement (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) (a3 : ℝ):
  isMonotonicIncreasing f →
  isOddFunction f →
  isArithmeticSeq a →
  a 3 = a3 →
  a3 > 0 →
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by
  -- proof will go here
  sorry

end problem_statement_l200_200803


namespace length_of_bridge_l200_200136

theorem length_of_bridge 
    (length_of_train : ℕ)
    (speed_of_train_km_per_hr : ℕ)
    (time_to_cross_seconds : ℕ)
    (bridge_length : ℕ) 
    (h_train_length : length_of_train = 130)
    (h_speed_train : speed_of_train_km_per_hr = 54)
    (h_time_cross : time_to_cross_seconds = 30)
    (h_bridge_length : bridge_length = 320) : 
    bridge_length = 320 :=
by sorry

end length_of_bridge_l200_200136


namespace evaluate_expression_l200_200655

theorem evaluate_expression : 
    (1 / ( (-5 : ℤ) ^ 4) ^ 2 ) * (-5 : ℤ) ^ 9 = -5 :=
by sorry

end evaluate_expression_l200_200655


namespace smallest_positive_integer_with_12_divisors_l200_200562

theorem smallest_positive_integer_with_12_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≠ n) → n ≤ m) ∧ ∃ d : ℕ → ℕ, (d n = 12) :=
by
  sorry

end smallest_positive_integer_with_12_divisors_l200_200562


namespace probability_of_condition_l200_200090

def Q_within_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

def condition (x y : ℝ) : Prop :=
  y > (1/2) * x

theorem probability_of_condition : 
  ∀ x y, Q_within_square x y → (0.75 = 3 / 4) :=
by
  sorry

end probability_of_condition_l200_200090


namespace niki_money_l200_200123

variables (N A : ℕ)

def condition1 (N A : ℕ) : Prop := N = 2 * A + 15
def condition2 (N A : ℕ) : Prop := N - 30 = (A + 30) / 2

theorem niki_money : condition1 N A ∧ condition2 N A → N = 55 :=
by
  sorry

end niki_money_l200_200123


namespace ellipse_eccentricity_l200_200575

theorem ellipse_eccentricity :
  (∃ (e : ℝ), (∀ (x y : ℝ), ((x^2 / 9) + y^2 = 1) → (e = 2 * Real.sqrt 2 / 3))) :=
by
  sorry

end ellipse_eccentricity_l200_200575


namespace false_statement_l200_200074

-- Define propositions p and q
def p := ∀ x : ℝ, (|x| = x) ↔ (x ≥ 0)
def q := ∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → (∃ origin : ℝ, ∀ y : ℝ, f (origin + y) = f (origin - y))

-- Define the possible answers
def option_A := p ∨ q
def option_B := p ∧ q
def option_C := ¬p ∧ q
def option_D := ¬p ∨ q

-- Define the false option (the correct answer was B)
def false_proposition := option_B

-- The statement to prove
theorem false_statement : false_proposition = false :=
by sorry

end false_statement_l200_200074


namespace max_height_reached_l200_200801

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 120 * t + 36

theorem max_height_reached :
  ∃ t : ℝ, h t = 216 ∧ t = 3 :=
sorry

end max_height_reached_l200_200801


namespace range_of_a_l200_200757

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x - (Real.cos x)^2 ≤ 3) : -3 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l200_200757


namespace driving_time_eqn_l200_200558

open Nat

-- Define the variables and constants
def avg_speed_before := 80 -- km/h
def stop_time := 1 / 3 -- hour
def avg_speed_after := 100 -- km/h
def total_distance := 250 -- km
def total_time := 3 -- hours

variable (t : ℝ) -- the time in hours before the stop

-- State the main theorem
theorem driving_time_eqn :
  avg_speed_before * t + avg_speed_after * (total_time - stop_time - t) = total_distance := by
  sorry

end driving_time_eqn_l200_200558


namespace cost_price_for_a_l200_200312

-- Definitions from the conditions
def selling_price_c : ℝ := 225
def profit_b : ℝ := 0.25
def profit_a : ℝ := 0.60

-- To prove: The cost price of the bicycle for A (cp_a) is 112.5
theorem cost_price_for_a : 
  ∃ (cp_a : ℝ), 
  (∃ (cp_b : ℝ), cp_b = (selling_price_c / (1 + profit_b)) ∧ 
   cp_a = (cp_b / (1 + profit_a))) ∧ 
   cp_a = 112.5 :=
by
  sorry

end cost_price_for_a_l200_200312


namespace tea_bags_l200_200657

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l200_200657


namespace b_and_c_work_days_l200_200661

theorem b_and_c_work_days
  (A B C : ℝ)
  (h1 : A + B = 1 / 8)
  (h2 : A + C = 1 / 8)
  (h3 : A + B + C = 1 / 6) :
  B + C = 1 / 24 :=
sorry

end b_and_c_work_days_l200_200661


namespace difference_in_perimeter_is_50_cm_l200_200778

-- Define the lengths of the four ribbons
def ribbon_lengths (x : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, x + 25, x + 50, x + 75)

-- Define the perimeter of the first shape
def perimeter_first_shape (x : ℕ) : ℕ :=
  2 * x + 230

-- Define the perimeter of the second shape
def perimeter_second_shape (x : ℕ) : ℕ :=
  2 * x + 280

-- Define the main theorem that the difference in perimeter is 50 cm
theorem difference_in_perimeter_is_50_cm (x : ℕ) :
  perimeter_second_shape x - perimeter_first_shape x = 50 := by
  sorry

end difference_in_perimeter_is_50_cm_l200_200778


namespace pow2_gt_square_for_all_n_ge_5_l200_200621

theorem pow2_gt_square_for_all_n_ge_5 (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
by
  sorry

end pow2_gt_square_for_all_n_ge_5_l200_200621


namespace red_ball_probability_l200_200669

noncomputable def Urn1_blue : ℕ := 5
noncomputable def Urn1_red : ℕ := 3
noncomputable def Urn2_blue : ℕ := 4
noncomputable def Urn2_red : ℕ := 4
noncomputable def Urn3_blue : ℕ := 8
noncomputable def Urn3_red : ℕ := 0

noncomputable def P_urn (n : ℕ) : ℝ := 1 / 3
noncomputable def P_red_urn1 : ℝ := (Urn1_red : ℝ) / (Urn1_blue + Urn1_red)
noncomputable def P_red_urn2 : ℝ := (Urn2_red : ℝ) / (Urn2_blue + Urn2_red)
noncomputable def P_red_urn3 : ℝ := (Urn3_red : ℝ) / (Urn3_blue + Urn3_red)

theorem red_ball_probability : 
  (P_urn 1 * P_red_urn1 + P_urn 2 * P_red_urn2 + P_urn 3 * P_red_urn3) = 7 / 24 :=
  by sorry

end red_ball_probability_l200_200669


namespace campaign_fliers_l200_200944

theorem campaign_fliers (total_fliers : ℕ) (fraction_morning : ℚ) (fraction_afternoon : ℚ) 
  (remaining_fliers_after_morning : ℕ) (remaining_fliers_after_afternoon : ℕ) :
  total_fliers = 1000 → fraction_morning = 1/5 → fraction_afternoon = 1/4 → 
  remaining_fliers_after_morning = total_fliers - total_fliers * fraction_morning → 
  remaining_fliers_after_afternoon = remaining_fliers_after_morning - remaining_fliers_after_morning * fraction_afternoon → 
  remaining_fliers_after_afternoon = 600 := 
by
  sorry

end campaign_fliers_l200_200944


namespace find_a_l200_200119

theorem find_a 
  (a b c : ℚ) 
  (h1 : b = 4 * a) 
  (h2 : b = 15 - 4 * a - c) 
  (h3 : c = a + 2) : 
  a = 13 / 9 := 
by 
  sorry

end find_a_l200_200119


namespace number_of_lamps_bought_l200_200018

-- Define the given conditions
def price_of_lamp : ℕ := 7
def price_of_bulb : ℕ := price_of_lamp - 4
def bulbs_bought : ℕ := 6
def total_spent : ℕ := 32

-- Define the statement to prove
theorem number_of_lamps_bought : 
  ∃ (L : ℕ), (price_of_lamp * L + price_of_bulb * bulbs_bought = total_spent) ∧ (L = 2) :=
sorry

end number_of_lamps_bought_l200_200018


namespace sum_of_three_numbers_l200_200510

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 20 :=
sorry

end sum_of_three_numbers_l200_200510


namespace chris_leftover_money_l200_200229

def chris_will_have_leftover : Prop :=
  let video_game_cost := 60
  let candy_cost := 5
  let hourly_wage := 8
  let hours_worked := 9
  let total_earned := hourly_wage * hours_worked
  let total_cost := video_game_cost + candy_cost
  let leftover := total_earned - total_cost
  leftover = 7

theorem chris_leftover_money : chris_will_have_leftover := 
  by
    sorry

end chris_leftover_money_l200_200229


namespace polynomial_m_n_values_l200_200789

theorem polynomial_m_n_values :
  ∀ (m n : ℝ), ((x - 1) * (x + m) = x^2 - n * x - 6) → (m = 6 ∧ n = -5) := 
by
  intros m n h
  sorry

end polynomial_m_n_values_l200_200789


namespace marcus_percentage_of_team_points_l200_200842

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l200_200842


namespace infinite_series_sum_l200_200305

theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / ((4 * n - 1)^3 * (4 * n + 3)^3)) = 1 / 972 := 
by 
  sorry

end infinite_series_sum_l200_200305


namespace years_passed_l200_200947

def initial_ages : List ℕ := [19, 34, 37, 42, 48]

def new_ages (x : ℕ) : List ℕ :=
  initial_ages.map (λ age => age + x)

-- Hypothesis: The new ages fit the following stem-and-leaf plot structure
def valid_stem_and_leaf (ages : List ℕ) : Bool :=
  ages = [25, 31, 34, 37, 43, 48]

theorem years_passed : ∃ x : ℕ, valid_stem_and_leaf (new_ages x) := by
  sorry

end years_passed_l200_200947


namespace original_balance_l200_200025

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

theorem original_balance (decrease_percentage : ℝ) (current_balance : ℝ) (original_balance : ℝ) :
  decrease_percentage = 0.10 → current_balance = 90000 → 
  current_balance = (1 - decrease_percentage) * original_balance → 
  original_balance = 100000 := by
  sorry

end original_balance_l200_200025


namespace arithmetic_sequence_sum_l200_200407

theorem arithmetic_sequence_sum {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 6) 
  (h2 : a 2 + a 14 = 26) :
  (10 / 2) * (a 1 + a 10) = 80 :=
by sorry

end arithmetic_sequence_sum_l200_200407


namespace not_consecutive_l200_200825

theorem not_consecutive (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) : 
  ¬ (∃ n : ℕ, (2023 + a - b = n ∧ 2023 + b - c = n + 1 ∧ 2023 + c - a = n + 2) ∨ 
    (2023 + a - b = n ∧ 2023 + b - c = n - 1 ∧ 2023 + c - a = n - 2)) :=
by
  sorry

end not_consecutive_l200_200825


namespace weight_gain_ratio_l200_200839

variable (J O F : ℝ)

theorem weight_gain_ratio :
  O = 5 ∧ F = (1/2) * J - 3 ∧ 5 + J + F = 20 → J / O = 12 / 5 :=
by
  intros h
  cases' h with hO h'
  cases' h' with hF hTotal
  sorry

end weight_gain_ratio_l200_200839


namespace lily_pads_half_lake_l200_200162

noncomputable def size (n : ℕ) : ℝ := sorry

theorem lily_pads_half_lake {n : ℕ} (h : size 48 = size 0 * 2^48) : size 47 = (size 48) / 2 :=
by 
  sorry

end lily_pads_half_lake_l200_200162


namespace parabola_through_points_with_h_l200_200968

noncomputable def quadratic_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_through_points_with_h (
    a h k : ℝ) 
    (H0 : quadratic_parabola a h k 0 = 4)
    (H1 : quadratic_parabola a h k 6 = 5)
    (H2 : a < 0)
    (H3 : 0 < h)
    (H4 : h < 6) : 
    h = 4 := 
sorry

end parabola_through_points_with_h_l200_200968


namespace dan_stationery_spent_l200_200069

def total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def notebook_cost : ℕ := 3
def number_of_notebooks : ℕ := 5
def stationery_cost_each : ℕ := 1

theorem dan_stationery_spent : 
  (total_spent - (backpack_cost + notebook_cost * number_of_notebooks)) = 2 :=
by
  sorry

end dan_stationery_spent_l200_200069


namespace vertex_x_coord_l200_200999

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Conditions based on given points
def conditions (a b c : ℝ) : Prop :=
  quadratic a b c 2 = 4 ∧
  quadratic a b c 8 =4 ∧
  quadratic a b c 10 = 13

-- Statement to prove the x-coordinate of the vertex is 5
theorem vertex_x_coord (a b c : ℝ) (h : conditions a b c) : 
  (-(b) / (2 * a)) = 5 :=
by
  sorry

end vertex_x_coord_l200_200999


namespace solution_of_inequality_l200_200063

open Set

theorem solution_of_inequality (x : ℝ) :
  x^2 - 2 * x - 3 > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_of_inequality_l200_200063


namespace general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l200_200905
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 3*n - 1
noncomputable def c_n (n : ℕ) : ℚ := (3*n - 1) / 2^(n-1)

-- 1. Prove that the sequence {a_n} is given by a_n = 2^(n-1) and {b_n} is given by b_n = 3n - 1
theorem general_formulas :
  (∀ n : ℕ, n > 0 → a_n n = 2^(n-1)) ∧
  (∀ n : ℕ, n > 0 → b_n n = 3*n - 1) :=
sorry

-- 2. Prove that the values of n for which c_n > 1 are n = 1, 2, 3, 4
theorem values_of_n_for_c_n_gt_one :
  { n : ℕ | n > 0 ∧ c_n n > 1 } = {1, 2, 3, 4} :=
sorry

-- 3. Prove that no three terms from {a_n} can form an arithmetic sequence
theorem no_three_terms_arithmetic_seq :
  ∀ p q r : ℕ, p < q ∧ q < r ∧ p > 0 ∧ q > 0 ∧ r > 0 →
  ¬ (2 * a_n q = a_n p + a_n r) :=
sorry

end general_formulas_values_of_n_for_c_n_gt_one_no_three_terms_arithmetic_seq_l200_200905


namespace simplify_expression_l200_200546

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) : 
  (x^2 - x) / (x^2 - 2 * x + 1) = 2 + Real.sqrt 2 :=
by
  sorry

end simplify_expression_l200_200546


namespace find_a_minus_b_l200_200922

theorem find_a_minus_b (a b : ℝ) (h1 : a + b = 12) (h2 : a^2 - b^2 = 48) : a - b = 4 :=
by
  sorry

end find_a_minus_b_l200_200922


namespace alloy_parts_separation_l200_200643

theorem alloy_parts_separation {p q x : ℝ} (h0 : p ≠ q)
  (h1 : 6 * p ≠ 16 * q)
  (h2 : 6 * x * p + 2 * (8 - 2 * x) * q = 8 * (8 - x) * p + 6 * x * q) :
  x = 2.4 :=
by
  sorry

end alloy_parts_separation_l200_200643


namespace sqrt_of_16_is_4_l200_200610

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 :=
sorry

end sqrt_of_16_is_4_l200_200610


namespace buying_beams_l200_200586

theorem buying_beams (x : ℕ) (h : 3 * (x - 1) * x = 6210) :
  3 * (x - 1) * x = 6210 :=
by {
  sorry
}

end buying_beams_l200_200586


namespace find_b_l200_200992

theorem find_b (a c S : ℝ) (h₁ : a = 5) (h₂ : c = 2) (h₃ : S = 4) : 
  b = Real.sqrt 17 ∨ b = Real.sqrt 41 := by
  sorry

end find_b_l200_200992


namespace triangle_is_isosceles_right_l200_200574

theorem triangle_is_isosceles_right (a b c : ℝ) (A B C : ℝ) (h1 : b = a * Real.sin C) (h2 : c = a * Real.cos B) : 
  A = π / 2 ∧ b = c := 
sorry

end triangle_is_isosceles_right_l200_200574


namespace sequence_decreasing_l200_200914

theorem sequence_decreasing : 
  ∀ (n : ℕ), n ≥ 1 → (1 / 2^(n - 1)) > (1 / 2^n) := 
by {
  sorry
}

end sequence_decreasing_l200_200914


namespace sum_of_fraction_parts_l200_200332

theorem sum_of_fraction_parts (x : ℝ) (hx : x = 0.45) : 
  (∃ (a b : ℕ), x = a / b ∧ Nat.gcd a b = 1 ∧ a + b = 16) :=
by
  sorry

end sum_of_fraction_parts_l200_200332


namespace quadratic_has_distinct_real_roots_l200_200384

theorem quadratic_has_distinct_real_roots :
  ∃ (x y : ℝ), x ≠ y ∧ (x^2 - 3 * x - 1 = 0) ∧ (y^2 - 3 * y - 1 = 0) :=
by {
  sorry
}

end quadratic_has_distinct_real_roots_l200_200384


namespace Rachel_picked_apples_l200_200862

theorem Rachel_picked_apples :
  let apples_from_first_tree := 8
  let apples_from_second_tree := 10
  let apples_from_third_tree := 12
  let apples_from_fifth_tree := 6
  apples_from_first_tree + apples_from_second_tree + apples_from_third_tree + apples_from_fifth_tree = 36 :=
by
  sorry

end Rachel_picked_apples_l200_200862


namespace fraction_arithmetic_l200_200605

theorem fraction_arithmetic :
  (3 / 4) / (5 / 8) + (1 / 8) = 53 / 40 :=
by
  sorry

end fraction_arithmetic_l200_200605


namespace geometric_sequence_sum_correct_l200_200860

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 2 then 2^(n + 1) - 2
else 64 * (1 - (1 / 2)^n)

theorem geometric_sequence_sum_correct (a1 q : ℝ) (n : ℕ) 
  (h1 : q > 0) 
  (h2 : a1 + a1 * q^4 = 34) 
  (h3 : a1^2 * q^4 = 64) :
  geometric_sequence_sum a1 q n = 
  if q = 2 then 2^(n + 1) - 2 else 64 * (1 - (1 / 2)^n) :=
sorry

end geometric_sequence_sum_correct_l200_200860


namespace cost_per_person_l200_200549

theorem cost_per_person 
  (total_cost : ℕ) 
  (total_people : ℕ) 
  (total_cost_in_billion : total_cost = 40000000000) 
  (total_people_in_million : total_people = 200000000) :
  total_cost / total_people = 200 := 
sorry

end cost_per_person_l200_200549


namespace f_even_f_increasing_f_range_l200_200919

variables {R : Type*} [OrderedRing R] (f : R → R)

-- Conditions
axiom f_mul : ∀ x y : R, f (x * y) = f x * f y
axiom f_neg1 : f (-1) = 1
axiom f_27 : f 27 = 9
axiom f_lt_1 : ∀ x : R, 0 ≤ x → x < 1 → 0 ≤ f x ∧ f x < 1

-- Questions
theorem f_even (x : R) : f x = f (-x) :=
by sorry

theorem f_increasing (x1 x2 : R) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 < x2) : f x1 < f x2 :=
by sorry

theorem f_range (a : R) (h1 : 0 ≤ a) (h2 : f (a + 1) ≤ 39) : 0 ≤ a ∧ a ≤ 2 :=
by sorry

end f_even_f_increasing_f_range_l200_200919


namespace find_t_of_quadratic_root_l200_200553

variable (a t : ℝ)

def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ t : ℝ, Complex.ofReal a + Complex.I * 3 = Complex.ofReal a - Complex.I * 3 ∧
           (Complex.ofReal a + Complex.I * 3).re * (Complex.ofReal a - Complex.I * 3).re = t

theorem find_t_of_quadratic_root (h : quadratic_root_condition a) : t = 13 :=
sorry

end find_t_of_quadratic_root_l200_200553


namespace strictly_positive_integer_le_36_l200_200611

theorem strictly_positive_integer_le_36 (n : ℕ) (h_pos : n > 0) :
  (∀ a : ℤ, (a % 2 = 1) → (a * a ≤ n) → (a ∣ n)) → n ≤ 36 := by
  sorry

end strictly_positive_integer_le_36_l200_200611


namespace arithmetic_sequence_sum_l200_200336

variable {a : ℕ → ℕ}

noncomputable def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h_arith : is_arithmetic_seq a) (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
by
  sorry

end arithmetic_sequence_sum_l200_200336


namespace remainder_250_div_k_l200_200303

theorem remainder_250_div_k {k : ℕ} (h1 : 0 < k) (h2 : 180 % (k * k) = 12) : 250 % k = 10 := by
  sorry

end remainder_250_div_k_l200_200303


namespace ratio_paperback_fiction_to_nonfiction_l200_200328

-- Definitions
def total_books := 160
def hardcover_nonfiction := 25
def paperback_nonfiction := hardcover_nonfiction + 20
def paperback_fiction := total_books - hardcover_nonfiction - paperback_nonfiction

-- Theorem statement
theorem ratio_paperback_fiction_to_nonfiction : paperback_fiction / paperback_nonfiction = 2 :=
by
  -- proof details would go here
  sorry

end ratio_paperback_fiction_to_nonfiction_l200_200328


namespace average_rate_of_change_correct_l200_200646

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_correct :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_correct_l200_200646


namespace intersection_point_of_lines_l200_200600

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 
    2 * x + y - 7 = 0 ∧ 
    x + 2 * y - 5 = 0 ∧ 
    x = 3 ∧ 
    y = 1 := 
by {
  sorry
}

end intersection_point_of_lines_l200_200600


namespace problem1_problem2_part1_problem2_part2_l200_200891

-- Problem 1
theorem problem1 (x : ℚ) (h : x = 11 / 12) : 
  (2 * x - 5) * (2 * x + 5) - (2 * x - 3) ^ 2 = -23 := 
by sorry

-- Problem 2
theorem problem2_part1 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  a^2 + b^2 = 22 := 
by sorry

theorem problem2_part2 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  (a - b)^2 = 8 := 
by sorry

end problem1_problem2_part1_problem2_part2_l200_200891


namespace factor_expression_l200_200024

theorem factor_expression (x : ℝ) : 
  (9 * x^5 + 25 * x^3 - 4) - (x^5 - 3 * x^3 - 4) = 4 * x^3 * (2 * x^2 + 7) :=
by
  sorry

end factor_expression_l200_200024


namespace vacant_student_seats_given_to_parents_l200_200108

-- Definitions of the conditions
def total_seats : Nat := 150

def awardees_seats : Nat := 15
def admins_teachers_seats : Nat := 45
def students_seats : Nat := 60
def parents_seats : Nat := 30

def awardees_occupied_seats : Nat := 15
def admins_teachers_occupied_seats : Nat := 9 * admins_teachers_seats / 10
def students_occupied_seats : Nat := 4 * students_seats / 5
def parents_occupied_seats : Nat := 7 * parents_seats / 10

-- Vacant seats calculation
def awardees_vacant_seats : Nat := awardees_seats - awardees_occupied_seats
def admins_teachers_vacant_seats : Nat := admins_teachers_seats - admins_teachers_occupied_seats
def students_vacant_seats : Nat := students_seats - students_occupied_seats
def parents_vacant_seats : Nat := parents_seats - parents_occupied_seats

-- Theorem statement
theorem vacant_student_seats_given_to_parents :
  students_vacant_seats = 12 →
  parents_vacant_seats = 9 →
  9 ≤ students_vacant_seats ∧ 9 ≤ parents_vacant_seats :=
by
  sorry

end vacant_student_seats_given_to_parents_l200_200108


namespace constant_term_binomial_expansion_l200_200353

theorem constant_term_binomial_expansion :
  ∀ (x : ℝ), ((2 / x) + x) ^ 4 = 24 :=
by
  sorry

end constant_term_binomial_expansion_l200_200353


namespace distinct_paths_l200_200855

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem distinct_paths (right_steps up_steps : ℕ) : right_steps = 7 → up_steps = 3 →
  binom (right_steps + up_steps) up_steps = 120 := 
by
  intros h1 h2
  rw [h1, h2]
  unfold binom
  simp
  norm_num
  sorry

end distinct_paths_l200_200855


namespace positive_divisors_d17_l200_200172

theorem positive_divisors_d17 (n : ℕ) (d : ℕ → ℕ) (k : ℕ) (h_order : d 1 = 1 ∧ ∀ i, 1 ≤ i → i ≤ k → d i < d (i + 1)) 
  (h_last : d k = n) (h_pythagorean : d 7 ^ 2 + d 15 ^ 2 = d 16 ^ 2) : 
  d 17 = 28 :=
sorry

end positive_divisors_d17_l200_200172


namespace trajectory_eq_of_midpoint_l200_200282

theorem trajectory_eq_of_midpoint (x y m n : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : (2*x = 3 + m) ∧ (2*y = n)) :
  (2*x - 3)^2 + 4*y^2 = 1 := 
sorry

end trajectory_eq_of_midpoint_l200_200282


namespace school_year_length_l200_200279

theorem school_year_length
  (children : ℕ)
  (juice_boxes_per_child_per_day : ℕ)
  (days_per_week : ℕ)
  (total_juice_boxes : ℕ)
  (w : ℕ)
  (h1 : children = 3)
  (h2 : juice_boxes_per_child_per_day = 1)
  (h3 : days_per_week = 5)
  (h4 : total_juice_boxes = 375)
  (h5 : total_juice_boxes = children * juice_boxes_per_child_per_day * days_per_week * w)
  : w = 25 :=
by
  sorry

end school_year_length_l200_200279


namespace Mrs_Brown_points_l200_200738

-- Conditions given
variables (points_William points_Adams points_Daniel points_mean: ℝ) (num_classes: ℕ)

-- Define the conditions
def Mrs_William_points := points_William = 50
def Mr_Adams_points := points_Adams = 57
def Mrs_Daniel_points := points_Daniel = 57
def mean_condition := points_mean = 53.3
def num_classes_condition := num_classes = 4

-- Define the problem to prove
theorem Mrs_Brown_points :
  Mrs_William_points points_William ∧ Mr_Adams_points points_Adams ∧ Mrs_Daniel_points points_Daniel ∧ mean_condition points_mean ∧ num_classes_condition num_classes →
  ∃ (points_Brown: ℝ), points_Brown = 49 :=
by
  sorry

end Mrs_Brown_points_l200_200738


namespace range_of_m_l200_200680

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) ↔ (m ∈ Set.Icc (-6:ℝ) 2) :=
by
  sorry

end range_of_m_l200_200680


namespace sufficient_but_not_necessary_condition_l200_200026

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ ¬ (|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l200_200026


namespace danny_marks_in_math_l200_200764

theorem danny_marks_in_math
  (english_marks : ℕ := 76)
  (physics_marks : ℕ := 82)
  (chemistry_marks : ℕ := 67)
  (biology_marks : ℕ := 75)
  (average_marks : ℕ := 73)
  (num_subjects : ℕ := 5) :
  ∃ (math_marks : ℕ), math_marks = 65 :=
by
  let total_marks := average_marks * num_subjects
  let other_subjects_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  have math_marks := total_marks - other_subjects_marks
  use math_marks
  sorry

end danny_marks_in_math_l200_200764


namespace solution_set_ineq1_solution_set_ineq2_l200_200626

theorem solution_set_ineq1 (x : ℝ) : 
  (-3 * x ^ 2 + x + 1 > 0) ↔ (x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) := 
sorry

theorem solution_set_ineq2 (x : ℝ) : 
  (x ^ 2 - 2 * x + 1 ≤ 0) ↔ (x = 1) := 
sorry

end solution_set_ineq1_solution_set_ineq2_l200_200626


namespace goat_cow_difference_l200_200889

-- Given the number of pigs (P), cows (C), and goats (G) on a farm
variables (P C G : ℕ)

-- Conditions:
def pig_count := P = 10
def cow_count_relationship := C = 2 * P - 3
def total_animals := P + C + G = 50

-- Theorem: The difference between the number of goats and cows
theorem goat_cow_difference (h1 : pig_count P)
                           (h2 : cow_count_relationship P C)
                           (h3 : total_animals P C G) :
  G - C = 6 := 
  sorry

end goat_cow_difference_l200_200889


namespace infinite_primes_of_form_l200_200928

theorem infinite_primes_of_form (p : ℕ) (hp : Nat.Prime p) (hpodd : p % 2 = 1) :
  ∃ᶠ n in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_l200_200928


namespace pentagon_angle_T_l200_200701

theorem pentagon_angle_T (P Q R S T : ℝ) 
  (hPRT: P = R ∧ R = T)
  (hQS: Q + S = 180): 
  T = 120 :=
by
  sorry

end pentagon_angle_T_l200_200701


namespace find_a_plus_b_l200_200989

theorem find_a_plus_b (a b : ℝ) (h_sum : 2 * a = -6) (h_prod : a^2 - b = 1) : a + b = 5 :=
by {
  -- Proof would go here; we assume the theorem holds true.
  sorry
}

end find_a_plus_b_l200_200989


namespace original_class_strength_l200_200813

variable (x : ℕ)

/-- The average age of an adult class is 40 years.
  18 new students with an average age of 32 years join the class, 
  therefore decreasing the average by 4 years.
  Find the original strength of the class.
-/
theorem original_class_strength (h1 : 40 * x + 18 * 32 = (x + 18) * 36) : x = 18 := 
by sorry

end original_class_strength_l200_200813


namespace division_of_8_identical_books_into_3_piles_l200_200098

-- Definitions for the conditions
def identical_books_division_ways (n : ℕ) (p : ℕ) : ℕ :=
  if n = 8 ∧ p = 3 then 5 else sorry

-- Theorem statement
theorem division_of_8_identical_books_into_3_piles :
  identical_books_division_ways 8 3 = 5 := by
  sorry

end division_of_8_identical_books_into_3_piles_l200_200098


namespace weekly_earnings_l200_200401

theorem weekly_earnings (total_earnings : ℕ) (weeks : ℕ) (h1 : total_earnings = 133) (h2 : weeks = 19) : 
  round (total_earnings / weeks : ℝ) = 7 := 
by 
  sorry

end weekly_earnings_l200_200401


namespace both_selected_prob_l200_200129

noncomputable def prob_X : ℚ := 1 / 3
noncomputable def prob_Y : ℚ := 2 / 7
noncomputable def combined_prob : ℚ := prob_X * prob_Y

theorem both_selected_prob :
  combined_prob = 2 / 21 :=
by
  unfold combined_prob prob_X prob_Y
  sorry

end both_selected_prob_l200_200129


namespace solution_set_fraction_inequality_l200_200743

theorem solution_set_fraction_inequality : 
  { x : ℝ | 0 < x ∧ x < 1/3 } = { x : ℝ | 1/x > 3 } :=
by
  sorry

end solution_set_fraction_inequality_l200_200743


namespace find_b_l200_200466

theorem find_b (b : ℝ) (x : ℝ) (hx : x^2 + b * x - 45 = 0) (h_root : x = -5) : b = -4 :=
by
  sorry

end find_b_l200_200466


namespace find_missing_number_l200_200207

theorem find_missing_number :
  ∀ (x y : ℝ),
    (12 + x + 42 + 78 + 104) / 5 = 62 →
    (128 + y + 511 + 1023 + x) / 5 = 398.2 →
    y = 255 :=
by
  intros x y h1 h2
  sorry

end find_missing_number_l200_200207


namespace math_homework_pages_l200_200330

-- Define Rachel's total pages, math homework pages, and reading homework pages
def total_pages : ℕ := 13
def reading_homework : ℕ := sorry
def math_homework (r : ℕ) : ℕ := r + 3

-- State the main theorem that needs to be proved
theorem math_homework_pages :
  ∃ r : ℕ, r + (math_homework r) = total_pages ∧ (math_homework r) = 8 :=
by {
  sorry
}

end math_homework_pages_l200_200330


namespace no_rational_multiples_pi_tan_sum_two_l200_200424

theorem no_rational_multiples_pi_tan_sum_two (x y : ℚ) (hx : 0 < x * π ∧ x * π < y * π ∧ y * π < π / 2) (hxy : Real.tan (x * π) + Real.tan (y * π) = 2) : False :=
sorry

end no_rational_multiples_pi_tan_sum_two_l200_200424


namespace total_trees_in_gray_areas_l200_200907

theorem total_trees_in_gray_areas (white_region_first : ℕ) (white_region_second : ℕ)
    (total_first : ℕ) (total_second : ℕ)
    (h1 : white_region_first = 82) (h2 : white_region_second = 82)
    (h3 : total_first = 100) (h4 : total_second = 90) :
  (total_first - white_region_first) + (total_second - white_region_second) = 26 := by
  sorry

end total_trees_in_gray_areas_l200_200907


namespace orthogonal_vectors_l200_200517

open Real

variables (r s : ℝ)

def a : ℝ × ℝ × ℝ := (5, r, -3)
def b : ℝ × ℝ × ℝ := (-1, 2, s)

theorem orthogonal_vectors
  (orthogonality : 5 * (-1) + r * 2 + (-3) * s = 0)
  (magnitude_condition : 34 + r^2 = 4 * (5 + s^2)) :
  ∃ (r s : ℝ), (2 * r - 3 * s = 5) ∧ (r^2 - 4 * s^2 = -14) :=
  sorry

end orthogonal_vectors_l200_200517


namespace brick_wall_l200_200055

theorem brick_wall (y : ℕ) (h1 : ∀ y, 6 * ((y / 8) + (y / 12) - 12) = y) : y = 288 :=
sorry

end brick_wall_l200_200055


namespace sum_of_coefficients_l200_200987

theorem sum_of_coefficients:
  (∀ x : ℝ, (2*x - 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) →
  a_1 + a_3 + a_5 = -364 :=
by
  sorry

end sum_of_coefficients_l200_200987


namespace triangle_internal_angles_external_angle_theorem_l200_200513

theorem triangle_internal_angles {A B C : ℝ}
 (mA : A = 64) (mB : B = 33) (mC_ext : C = 120) :
  180 - A - B = 83 :=
by
  sorry

theorem external_angle_theorem {A C D : ℝ}
 (mA : A = 64) (mC_ext : C = 120) :
  C = A + D → D = 56 :=
by
  sorry

end triangle_internal_angles_external_angle_theorem_l200_200513


namespace rowing_students_l200_200194

theorem rowing_students (X Y : ℕ) (N : ℕ) :
  (17 * X + 6 = N) →
  (10 * Y + 2 = N) →
  100 < N →
  N < 200 →
  5 ≤ X ∧ X ≤ 11 →
  10 ≤ Y ∧ Y ≤ 19 →
  N = 142 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end rowing_students_l200_200194


namespace sum_of_squares_fraction_l200_200092

variable {x1 x2 x3 y1 y2 y3 : ℝ}

theorem sum_of_squares_fraction :
  x1 + x2 + x3 = 0 → y1 + y2 + y3 = 0 → x1 * y1 + x2 * y2 + x3 * y3 = 0 →
  (x1^2 / (x1^2 + x2^2 + x3^2)) + (y1^2 / (y1^2 + y2^2 + y3^2)) = 2 / 3 :=
by
  intros h1 h2 h3
  sorry

end sum_of_squares_fraction_l200_200092


namespace units_produced_today_eq_90_l200_200458

-- Define the average production and number of past days
def average_past_production (n : ℕ) (past_avg : ℕ) : ℕ :=
  n * past_avg

def average_total_production (n : ℕ) (current_avg : ℕ) : ℕ :=
  (n + 1) * current_avg

def units_produced_today (n : ℕ) (past_avg : ℕ) (current_avg : ℕ) : ℕ :=
  average_total_production n current_avg - average_past_production n past_avg

-- Given conditions
def n := 5
def past_avg := 60
def current_avg := 65

-- Statement to prove
theorem units_produced_today_eq_90 : units_produced_today n past_avg current_avg = 90 :=
by
  -- Declare which parts need proving
  sorry

end units_produced_today_eq_90_l200_200458


namespace Alex_final_silver_tokens_l200_200372

variable (x y : ℕ)

def final_red_tokens (x y : ℕ) : ℕ := 90 - 3 * x + 2 * y
def final_blue_tokens (x y : ℕ) : ℕ := 65 + 2 * x - 4 * y
def silver_tokens (x y : ℕ) : ℕ := x + y

theorem Alex_final_silver_tokens (h1 : final_red_tokens x y < 3)
                                 (h2 : final_blue_tokens x y < 4) :
  silver_tokens x y = 67 := 
sorry

end Alex_final_silver_tokens_l200_200372


namespace rate_percent_simple_interest_l200_200130

theorem rate_percent_simple_interest (P SI T : ℝ) (hP : P = 720) (hSI : SI = 180) (hT : T = 4) :
  (SI = P * (R / 100) * T) → R = 6.25 :=
by
  sorry

end rate_percent_simple_interest_l200_200130


namespace box_height_correct_l200_200486

noncomputable def box_height : ℕ :=
  8

theorem box_height_correct (box_width box_length block_height block_width block_length : ℕ) (num_blocks : ℕ) :
  box_width = 10 ∧
  box_length = 12 ∧
  block_height = 3 ∧
  block_width = 2 ∧
  block_length = 4 ∧
  num_blocks = 40 →
  (num_blocks * block_height * block_width * block_length) /
  (box_width * box_length) = box_height :=
  by
  sorry

end box_height_correct_l200_200486


namespace initial_percentage_of_grape_juice_l200_200897

theorem initial_percentage_of_grape_juice (P : ℝ) 
  (h₀ : 10 + 30 = 40)
  (h₁ : 40 * 0.325 = 13)
  (h₂ : 30 * P + 10 = 13) : 
  P = 0.1 :=
  by 
    sorry

end initial_percentage_of_grape_juice_l200_200897


namespace divisible_by_five_l200_200334

theorem divisible_by_five {x y z : ℤ} (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
sorry

end divisible_by_five_l200_200334


namespace george_initial_socks_l200_200838

theorem george_initial_socks (S : ℕ) (h : S - 4 + 36 = 60) : S = 28 :=
by
  sorry

end george_initial_socks_l200_200838


namespace pyramid_height_correct_l200_200270

noncomputable def pyramid_height (a α : ℝ) : ℝ :=
  a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))

theorem pyramid_height_correct (a α : ℝ) (hα : α ≠ 0 ∧ α ≠ π) :
  ∃ m : ℝ, m = pyramid_height a α := 
by
  use a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))
  sorry

end pyramid_height_correct_l200_200270


namespace min_value_xyz_l200_200703

open Real

theorem min_value_xyz
  (x y z : ℝ)
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : 5 * x + 16 * y + 33 * z ≥ 136) :
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 :=
sorry

end min_value_xyz_l200_200703


namespace circle_possible_values_l200_200436

theorem circle_possible_values (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 + a - 1 = 0 → -2 < a ∧ a < 2/3) := sorry

end circle_possible_values_l200_200436


namespace stratified_sampling_l200_200047

theorem stratified_sampling
  (students_class1 : ℕ)
  (students_class2 : ℕ)
  (formation_slots : ℕ)
  (total_students : ℕ)
  (prob_selected: ℚ)
  (selected_class1 : ℕ)
  (selected_class2 : ℕ)
  (h1 : students_class1 = 54)
  (h2 : students_class2 = 42)
  (h3 : formation_slots = 16)
  (h4 : total_students = students_class1 + students_class2)
  (h5 : prob_selected = formation_slots / total_students)
  (h6 : selected_class1 = students_class1 * prob_selected)
  (h7 : selected_class2 = students_class2 * prob_selected)
  : selected_class1 = 9 ∧ selected_class2 = 7 := by
  sorry

end stratified_sampling_l200_200047


namespace tessa_still_owes_greg_l200_200807

def initial_debt : ℝ := 40
def first_repayment : ℝ := 0.25 * initial_debt
def debt_after_first_repayment : ℝ := initial_debt - first_repayment
def second_borrowing : ℝ := 25
def debt_after_second_borrowing : ℝ := debt_after_first_repayment + second_borrowing
def second_repayment : ℝ := 0.5 * debt_after_second_borrowing
def debt_after_second_repayment : ℝ := debt_after_second_borrowing - second_repayment
def third_borrowing : ℝ := 30
def debt_after_third_borrowing : ℝ := debt_after_second_repayment + third_borrowing
def third_repayment : ℝ := 0.1 * debt_after_third_borrowing
def final_debt : ℝ := debt_after_third_borrowing - third_repayment

theorem tessa_still_owes_greg : final_debt = 51.75 := by
  sorry

end tessa_still_owes_greg_l200_200807


namespace percentage_rotten_apples_l200_200030

theorem percentage_rotten_apples
  (total_apples : ℕ)
  (smell_pct : ℚ)
  (non_smelling_rotten_apples : ℕ)
  (R : ℚ) :
  total_apples = 200 →
  smell_pct = 0.70 →
  non_smelling_rotten_apples = 24 →
  0.30 * (R / 100 * total_apples) = non_smelling_rotten_apples →
  R = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_rotten_apples_l200_200030


namespace more_pups_than_adult_dogs_l200_200331

def number_of_huskies := 5
def number_of_pitbulls := 2
def number_of_golden_retrievers := 4
def pups_per_husky := 3
def pups_per_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky + additional_pups_per_golden_retriever

def total_pups := (number_of_huskies * pups_per_husky) + (number_of_pitbulls * pups_per_pitbull) + (number_of_golden_retrievers * pups_per_golden_retriever)
def total_adult_dogs := number_of_huskies + number_of_pitbulls + number_of_golden_retrievers

theorem more_pups_than_adult_dogs : (total_pups - total_adult_dogs) = 30 :=
by
  -- proof steps, which we will skip
  sorry

end more_pups_than_adult_dogs_l200_200331


namespace hall_reunion_attendees_l200_200175

theorem hall_reunion_attendees
  (total_guests : ℕ)
  (oates_attendees : ℕ)
  (both_attendees : ℕ)
  (h : total_guests = 100 ∧ oates_attendees = 50 ∧ both_attendees = 12) :
  ∃ (hall_attendees : ℕ), hall_attendees = 62 :=
by
  sorry

end hall_reunion_attendees_l200_200175


namespace domain_of_f_2x_plus_1_l200_200875

theorem domain_of_f_2x_plus_1 {f : ℝ → ℝ} :
  (∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 3 → (-3 : ℝ) ≤ x - 1 ∧ x - 1 ≤ 2) →
  (∀ x, (-3 : ℝ) ≤ x ∧ x ≤ 2 → (-2 : ℝ) ≤ (x : ℝ) ∧ x ≤ 1/2) →
  ∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 1 / 2 → ∀ y, y = 2 * x + 1 → (-3 : ℝ) ≤ y ∧ y ≤ 2 :=
by
  sorry

end domain_of_f_2x_plus_1_l200_200875


namespace white_mice_count_l200_200445

variable (T W B : ℕ) -- Declare variables T (total), W (white), B (brown)

def W_condition := W = (2 / 3) * T  -- White mice condition
def B_condition := B = 7           -- Brown mice condition
def T_condition := T = W + B       -- Total mice condition

theorem white_mice_count : W = 14 :=
by
  sorry  -- Proof to be filled in

end white_mice_count_l200_200445


namespace min_value_of_fraction_sum_l200_200230

theorem min_value_of_fraction_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 :=
sorry

end min_value_of_fraction_sum_l200_200230


namespace find_M_range_of_a_l200_200689

def Δ (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

def A : Set ℝ := { x | 4 * x^2 + 9 * x + 2 < 0 }

def B : Set ℝ := { x | -1 < x ∧ x < 2 }

def M : Set ℝ := Δ B A

def P (a: ℝ) : Set ℝ := { x | (x - 2 * a) * (x + a - 2) < 0 }

theorem find_M :
  M = { x | -1/4 ≤ x ∧ x < 2 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ M → x ∈ P a) →
  a < -1/8 ∨ a > 9/4 :=
sorry

end find_M_range_of_a_l200_200689


namespace largest_common_number_in_arithmetic_sequences_l200_200288

theorem largest_common_number_in_arithmetic_sequences (x : ℕ)
  (h1 : x ≡ 2 [MOD 8])
  (h2 : x ≡ 5 [MOD 9])
  (h3 : x < 200) : x = 194 :=
by sorry

end largest_common_number_in_arithmetic_sequences_l200_200288


namespace shaded_region_perimeter_l200_200952

theorem shaded_region_perimeter (r : ℝ) (θ : ℝ) (h₁ : r = 2) (h₂ : θ = 90) : 
  (2 * r + (2 * π * r * (1 - θ / 180))) = π + 4 := 
by sorry

end shaded_region_perimeter_l200_200952


namespace time_worked_on_thursday_l200_200262

/-
  Given:
  - Monday: 3/4 hour
  - Tuesday: 1/2 hour
  - Wednesday: 2/3 hour
  - Friday: 75 minutes
  - Total (Monday to Friday): 4 hours = 240 minutes
  
  The time Mr. Willson worked on Thursday is 50 minutes.
-/

noncomputable def time_worked_monday : ℝ := (3 / 4) * 60
noncomputable def time_worked_tuesday : ℝ := (1 / 2) * 60
noncomputable def time_worked_wednesday : ℝ := (2 / 3) * 60
noncomputable def time_worked_friday : ℝ := 75
noncomputable def total_time_worked : ℝ := 4 * 60

theorem time_worked_on_thursday :
  time_worked_monday + time_worked_tuesday + time_worked_wednesday + time_worked_friday + 50 = total_time_worked :=
by
  sorry

end time_worked_on_thursday_l200_200262


namespace percent_water_evaporated_l200_200708

theorem percent_water_evaporated (W : ℝ) (E : ℝ) (T : ℝ) (hW : W = 10) (hE : E = 0.16) (hT : T = 75) : 
  ((min (E * T) W) / W) * 100 = 100 :=
by
  sorry

end percent_water_evaporated_l200_200708


namespace angle_QPS_l200_200746

-- Definitions of the points and angles
variables (P Q R S : Point)
variables (angle : Point → Point → Point → ℝ)

-- Conditions about the isosceles triangles and angles
variables (isosceles_PQR : PQ = QR)
variables (isosceles_PRS : PR = RS)
variables (R_inside_PQS : ¬(R ∈ convex_hull ℝ {P, Q, S}))
variables (angle_PQR : angle P Q R = 50)
variables (angle_PRS : angle P R S = 120)

-- The theorem we want to prove
theorem angle_QPS : angle Q P S = 35 :=
sorry -- Proof goes here

end angle_QPS_l200_200746


namespace odd_even_divisors_ratio_l200_200666

theorem odd_even_divisors_ratio (M : ℕ) (h1 : M = 2^5 * 3^5 * 5 * 7^3) :
  let sum_odd_divisors := (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_all_divisors := (1 + 2 + 4 + 8 + 16 + 32) * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  sum_odd_divisors / sum_even_divisors = 1 / 62 :=
by
  sorry

end odd_even_divisors_ratio_l200_200666


namespace coord_of_point_B_l200_200096
-- Necessary import for mathematical definitions and structures

-- Define the initial point A and the translation conditions
def point_A : ℝ × ℝ := (1, -2)
def translation_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1, p.2 + units)

-- The target point B after translation
def point_B := translation_up point_A 1

-- The theorem to prove that the coordinates of B are (1, -1)
theorem coord_of_point_B : point_B = (1, -1) :=
by
  -- Placeholder for proof
  sorry

end coord_of_point_B_l200_200096


namespace determine_k_l200_200841

variables (x y z k : ℝ)

theorem determine_k (h1 : (5 / (x - z)) = (k / (y + z))) 
                    (h2 : (k / (y + z)) = (12 / (x + y))) 
                    (h3 : y + z = 2 * x) : 
                    k = 17 := 
by 
  sorry

end determine_k_l200_200841


namespace area_ratio_of_circles_l200_200500

-- Define the circles and lengths of arcs
variables {R_C R_D : ℝ} (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D))

-- Theorem proving the ratio of the areas
theorem area_ratio_of_circles (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 := sorry

end area_ratio_of_circles_l200_200500


namespace fraction_meaningful_l200_200105

theorem fraction_meaningful (x : ℝ) : (x ≠ 5) ↔ (x-5 ≠ 0) :=
by simp [sub_eq_zero]

end fraction_meaningful_l200_200105


namespace inequality_proof_l200_200452

theorem inequality_proof (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a^3 * b + b^3 * c + c^3 * a ≥ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by {
  sorry
}

end inequality_proof_l200_200452


namespace domain_tan_x_plus_pi_over_3_l200_200393

open Real Set

theorem domain_tan_x_plus_pi_over_3 :
  ∀ x : ℝ, ¬ (∃ k : ℤ, x = k * π + π / 6) ↔ x ∈ {x : ℝ | ¬ ∃ k : ℤ, x = k * π + π / 6} :=
by {
  sorry
}

end domain_tan_x_plus_pi_over_3_l200_200393


namespace gcd_of_1237_and_1957_is_one_l200_200728

noncomputable def gcd_1237_1957 : Nat := Nat.gcd 1237 1957

theorem gcd_of_1237_and_1957_is_one : gcd_1237_1957 = 1 :=
by
  unfold gcd_1237_1957
  have : Nat.gcd 1237 1957 = 1 := sorry
  exact this

end gcd_of_1237_and_1957_is_one_l200_200728


namespace woman_born_second_half_20th_century_l200_200993

theorem woman_born_second_half_20th_century (x : ℕ) (hx : 45 < x ∧ x < 50) (h_year : x * x = 2025) :
  x * x - x = 1980 :=
by {
  -- Add the crux of the problem here.
  sorry
}

end woman_born_second_half_20th_century_l200_200993


namespace original_number_one_more_reciprocal_is_11_over_5_l200_200913

theorem original_number_one_more_reciprocal_is_11_over_5 (x : ℚ) (h : 1 + 1/x = 11/5) : x = 5/6 :=
by
  sorry

end original_number_one_more_reciprocal_is_11_over_5_l200_200913


namespace symmetric_circle_eq_l200_200739

open Real

-- Define the original circle equation and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation with respect to the line y = -x
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Define the new circle that is symmetric to the original circle with respect to y = -x
def new_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- The theorem to be proven
theorem symmetric_circle_eq :
  ∀ x y : ℝ, original_circle (-y) (-x) ↔ new_circle x y := 
by
  sorry

end symmetric_circle_eq_l200_200739


namespace male_contestants_l200_200649

theorem male_contestants (total_contestants : ℕ) (female_proportion : ℕ) (total_females : ℕ) :
  female_proportion = 3 ∧ total_contestants = 18 ∧ total_females = total_contestants / female_proportion →
  (total_contestants - total_females) = 12 :=
by
  sorry

end male_contestants_l200_200649


namespace power_sum_l200_200460

theorem power_sum : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end power_sum_l200_200460


namespace owen_turtles_l200_200457

theorem owen_turtles (o_initial : ℕ) (j_initial : ℕ) (o_after_month : ℕ) (j_remaining : ℕ) (o_final : ℕ) 
  (h1 : o_initial = 21)
  (h2 : j_initial = o_initial - 5)
  (h3 : o_after_month = 2 * o_initial)
  (h4 : j_remaining = j_initial / 2)
  (h5 : o_final = o_after_month + j_remaining) :
  o_final = 50 :=
sorry

end owen_turtles_l200_200457


namespace isosceles_triangle_congruent_side_length_l200_200044

theorem isosceles_triangle_congruent_side_length 
  (base : ℝ) (area : ℝ) (a b c : ℝ) 
  (h1 : a = c)
  (h2 : a = base / 2)
  (h3 : (base * a) / 2 = area)
  : b = 5 * Real.sqrt 10 := 
by sorry

end isosceles_triangle_congruent_side_length_l200_200044


namespace fraction_blue_balls_l200_200700

theorem fraction_blue_balls (total_balls : ℕ) (red_fraction : ℚ) (other_balls : ℕ) (remaining_blue_fraction : ℚ) 
  (h1 : total_balls = 360) 
  (h2 : red_fraction = 1/4) 
  (h3 : other_balls = 216) 
  (h4 : remaining_blue_fraction = 1/5) :
  (total_balls - (total_balls / 4) - other_balls) = total_balls * (5 * red_fraction / 270) := 
by
  sorry

end fraction_blue_balls_l200_200700


namespace matrix_det_zero_l200_200308

variables {α β γ : ℝ}

theorem matrix_det_zero (h : α + β + γ = π) :
  Matrix.det ![
    ![Real.cos β, Real.cos α, -1],
    ![Real.cos γ, -1, Real.cos α],
    ![-1, Real.cos γ, Real.cos β]
  ] = 0 :=
sorry

end matrix_det_zero_l200_200308


namespace intersect_once_l200_200444

theorem intersect_once (x : ℝ) : 
  (∀ y, y = 3 * Real.log x ↔ y = Real.log (3 * x)) → (∃! x, 3 * Real.log x = Real.log (3 * x)) :=
by 
  sorry

end intersect_once_l200_200444


namespace eight_sharp_two_equals_six_thousand_l200_200420

def new_operation (a b : ℕ) : ℕ :=
  (a + b) ^ 3 * (a - b)

theorem eight_sharp_two_equals_six_thousand : new_operation 8 2 = 6000 := 
  by
    sorry

end eight_sharp_two_equals_six_thousand_l200_200420


namespace professionals_work_days_l200_200432

theorem professionals_work_days (cost_per_hour_1 cost_per_hour_2 hours_per_day total_cost : ℝ) (h_cost1: cost_per_hour_1 = 15) (h_cost2: cost_per_hour_2 = 15) (h_hours: hours_per_day = 6) (h_total: total_cost = 1260) : (∃ d : ℝ, total_cost = d * hours_per_day * (cost_per_hour_1 + cost_per_hour_2) ∧ d = 7) :=
by
  use 7
  rw [h_cost1, h_cost2, h_hours, h_total]
  simp
  sorry

end professionals_work_days_l200_200432


namespace average_temp_tues_to_fri_l200_200351

theorem average_temp_tues_to_fri (T W Th : ℕ) 
  (h1: (42 + T + W + Th) / 4 = 48) 
  (mon: 42 = 42) 
  (fri: 10 = 10) :
  (T + W + Th + 10) / 4 = 40 := by
  sorry

end average_temp_tues_to_fri_l200_200351


namespace find_a_range_l200_200547

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := 3 * Real.exp x + a

theorem find_a_range (a : ℝ) :
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x > g x a) → a < Real.exp 2 :=
by
  sorry

end find_a_range_l200_200547


namespace max_value_of_g_l200_200263

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_of_g : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 ∧ (∃ x0, x0 = 1 ∧ g x0 = 3) :=
by
  sorry

end max_value_of_g_l200_200263


namespace mix_alcohol_solutions_l200_200986

-- Definitions capturing the conditions from part (a)
def volume_solution_y : ℝ := 600
def percent_alcohol_x : ℝ := 0.1
def percent_alcohol_y : ℝ := 0.3
def desired_percent_alcohol : ℝ := 0.25

-- The resulting Lean statement to prove question == answer given conditions
theorem mix_alcohol_solutions (Vx : ℝ) (h : (percent_alcohol_x * Vx + percent_alcohol_y * volume_solution_y) / (Vx + volume_solution_y) = desired_percent_alcohol) : Vx = 200 :=
sorry

end mix_alcohol_solutions_l200_200986


namespace citizen_income_l200_200940

theorem citizen_income (tax_paid : ℝ) (base_income : ℝ) (base_rate excess_rate : ℝ) (income : ℝ) 
  (h1 : 0 < base_income) (h2 : base_rate * base_income = 4400) (h3 : tax_paid = 8000)
  (h4 : excess_rate = 0.20) (h5 : base_rate = 0.11)
  (h6 : tax_paid = base_rate * base_income + excess_rate * (income - base_income)) :
  income = 58000 :=
sorry

end citizen_income_l200_200940


namespace current_balance_after_deduction_l200_200125

theorem current_balance_after_deduction :
  ∀ (original_balance deduction_percent : ℕ), 
  original_balance = 100000 →
  deduction_percent = 10 →
  original_balance - (deduction_percent * original_balance / 100) = 90000 :=
by
  intros original_balance deduction_percent h1 h2
  sorry

end current_balance_after_deduction_l200_200125


namespace simple_random_sampling_methods_proof_l200_200848

-- Definitions based on conditions
def equal_probability (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
∀ s1 s2 : samples, p s1 = p s2

-- Define that Lottery Drawing Method and Random Number Table Method are part of simple random sampling
def is_lottery_drawing_method (samples : Type) : Prop := sorry
def is_random_number_table_method (samples : Type) : Prop := sorry

def simple_random_sampling_methods (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
  equal_probability samples p ∧ is_lottery_drawing_method samples ∧ is_random_number_table_method samples

-- Statement to be proven
theorem simple_random_sampling_methods_proof (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) :
  (∀ s1 s2 : samples, p s1 = p s2) → simple_random_sampling_methods samples p :=
by
  intro h
  unfold simple_random_sampling_methods
  constructor
  exact h
  constructor
  sorry -- Proof for is_lottery_drawing_method
  sorry -- Proof for is_random_number_table_method

end simple_random_sampling_methods_proof_l200_200848


namespace distance_ran_each_morning_l200_200021

-- Definitions based on conditions
def days_ran : ℕ := 3
def total_distance : ℕ := 2700

-- The goal is to prove the distance ran each morning
theorem distance_ran_each_morning : total_distance / days_ran = 900 :=
by
  sorry

end distance_ran_each_morning_l200_200021


namespace percentage_reduction_in_women_l200_200597

theorem percentage_reduction_in_women
    (total_people : Nat) (men_in_office : Nat) (women_in_office : Nat)
    (men_in_meeting : Nat) (women_in_meeting : Nat)
    (even_men_women : men_in_office = women_in_office)
    (total_people_condition : total_people = men_in_office + women_in_office)
    (meeting_condition : total_people = 60)
    (men_meeting_condition : men_in_meeting = 4)
    (women_meeting_condition : women_in_meeting = 6) :
    ((women_in_meeting * 100) / women_in_office) = 20 :=
by
  sorry

end percentage_reduction_in_women_l200_200597


namespace find_orange_juice_amount_l200_200730

variable (s y t oj : ℝ)

theorem find_orange_juice_amount (h1 : s = 0.2) (h2 : y = 0.1) (h3 : t = 0.5) (h4 : oj = t - (s + y)) : oj = 0.2 :=
by
  sorry

end find_orange_juice_amount_l200_200730


namespace necessary_but_not_sufficient_condition_l200_200729

variables {a b : ℤ}

theorem necessary_but_not_sufficient_condition : (¬(a = 1) ∨ ¬(b = 2)) ↔ ¬(a + b = 3) :=
by
  sorry

end necessary_but_not_sufficient_condition_l200_200729


namespace value_of_ab_l200_200228

theorem value_of_ab (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 :=
by
  sorry

end value_of_ab_l200_200228


namespace fraction_equation_solution_l200_200143

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) :
  (3 / (x - 3) = 4 / (x - 4)) → x = 0 :=
by
  sorry

end fraction_equation_solution_l200_200143


namespace choco_delight_remainder_l200_200386

theorem choco_delight_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := 
by 
  sorry

end choco_delight_remainder_l200_200386


namespace line_circle_intersections_l200_200265

-- Define the line equation as a predicate
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- The goal is to prove the number of intersections of the line and the circle
theorem line_circle_intersections : (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧ 
                                   (∃ x y : ℝ, line_eq x y ∧ circle_eq x y ∧ x ≠ y) :=
sorry

end line_circle_intersections_l200_200265


namespace total_length_XYZ_l200_200405

theorem total_length_XYZ :
  let straight_segments := 7
  let slanted_segments := 7 * Real.sqrt 2
  straight_segments + slanted_segments = 7 + 7 * Real.sqrt 2 :=
by
  sorry

end total_length_XYZ_l200_200405


namespace ThaboRatio_l200_200214

-- Define the variables
variables (P_f P_nf H_nf : ℕ)

-- Define the conditions as hypotheses
def ThaboConditions := P_f + P_nf + H_nf = 280 ∧ P_nf = H_nf + 20 ∧ H_nf = 55

-- State the theorem we want to prove
theorem ThaboRatio (h : ThaboConditions P_f P_nf H_nf) : (P_f / P_nf) = 2 :=
by sorry

end ThaboRatio_l200_200214


namespace simplify_fraction_l200_200939

theorem simplify_fraction :
  (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end simplify_fraction_l200_200939


namespace perfect_squares_between_2_and_20_l200_200697

-- Defining the conditions and problem statement
theorem perfect_squares_between_2_and_20 : 
  ∃ n, n = 3 ∧ ∀ m, (2 < m ∧ m < 20 ∧ ∃ k, k * k = m) ↔ m = 4 ∨ m = 9 ∨ m = 16 :=
by {
  -- Start the proof process
  sorry -- Placeholder for the proof
}

end perfect_squares_between_2_and_20_l200_200697


namespace noodles_given_to_William_l200_200938

def initial_noodles : ℝ := 54.0
def noodles_left : ℝ := 42.0
def noodles_given : ℝ := initial_noodles - noodles_left

theorem noodles_given_to_William : noodles_given = 12.0 := 
by
  sorry -- Proof to be filled in

end noodles_given_to_William_l200_200938


namespace cross_section_area_of_truncated_pyramid_l200_200273

-- Given conditions
variables (a b : ℝ) (α : ℝ)
-- Constraints
variable (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2)

-- Proposed theorem
theorem cross_section_area_of_truncated_pyramid (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2) :
    ∃ area : ℝ, area = (7 * a + 3 * b) / (144 * Real.cos α) * Real.sqrt (3 * (a^2 + b^2 + 2 * a * b * Real.cos (2 * α))) :=
sorry

end cross_section_area_of_truncated_pyramid_l200_200273


namespace incorrect_statement_l200_200100

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Definition of set M
def M : Set ℕ := {1, 2}

-- Definition of set N
def N : Set ℕ := {2, 4}

-- Complement of set in a universal set
def complement (S : Set ℕ) : Set ℕ := U \ S

-- Statement that D is incorrect
theorem incorrect_statement :
  M ∩ complement N ≠ {1, 2, 3} :=
by
  sorry

end incorrect_statement_l200_200100


namespace yule_log_surface_area_increase_l200_200370

theorem yule_log_surface_area_increase :
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initial_surface_area := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let slice_height := h / n
  let slice_surface_area := 2 * Real.pi * r * slice_height + 2 * Real.pi * r^2
  let total_surface_area_slices := n * slice_surface_area
  let delta_surface_area := total_surface_area_slices - initial_surface_area
  delta_surface_area = 100 * Real.pi :=
by
  sorry

end yule_log_surface_area_increase_l200_200370


namespace digit_encoding_problem_l200_200930

theorem digit_encoding_problem :
  ∃ (A B : ℕ), 0 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 21 * A + B = 111 * B ∧ A = 5 ∧ B = 5 :=
by
  sorry

end digit_encoding_problem_l200_200930


namespace find_fraction_l200_200664

theorem find_fraction (x y : ℝ) (h1 : (1/3) * (1/4) * x = 18) (h2 : y * x = 64.8) : y = 0.3 :=
sorry

end find_fraction_l200_200664


namespace percentage_alcohol_second_vessel_l200_200198

theorem percentage_alcohol_second_vessel :
  (∀ (x : ℝ),
    (0.25 * 3 + (x / 100) * 5 = 0.275 * 10) -> x = 40) :=
by
  intro x h
  sorry

end percentage_alcohol_second_vessel_l200_200198


namespace negation_proposition_equiv_l200_200345

open Classical

variable (R : Type) [OrderedRing R] (a x : R)

theorem negation_proposition_equiv :
  (¬ ∃ a : R, ∃ x : R, a * x^2 + 1 = 0) ↔ (∀ a : R, ∀ x : R, a * x^2 + 1 ≠ 0) :=
by
  sorry

end negation_proposition_equiv_l200_200345


namespace bella_age_is_five_l200_200872

-- Definitions from the problem:
def is_age_relation (bella_age brother_age : ℕ) : Prop :=
  brother_age = bella_age + 9 ∧ bella_age + brother_age = 19

-- The main proof statement:
theorem bella_age_is_five (bella_age brother_age : ℕ) (h : is_age_relation bella_age brother_age) :
  bella_age = 5 :=
by {
  -- Placeholder for proof steps
  sorry
}

end bella_age_is_five_l200_200872


namespace brick_length_is_20_cm_l200_200195

theorem brick_length_is_20_cm
    (courtyard_length_m : ℕ) (courtyard_width_m : ℕ)
    (brick_length_cm : ℕ) (brick_width_cm : ℕ)
    (total_bricks_required : ℕ)
    (h1 : courtyard_length_m = 25)
    (h2 : courtyard_width_m = 16)
    (h3 : brick_length_cm = 20)
    (h4 : brick_width_cm = 10)
    (h5 : total_bricks_required = 20000) :
    brick_length_cm = 20 := 
by
    sorry

end brick_length_is_20_cm_l200_200195


namespace arithmetic_progression_condition_l200_200861

theorem arithmetic_progression_condition
  (a b c : ℝ) (a1 d : ℝ) (p n k : ℕ) :
  a = a1 + (p - 1) * d →
  b = a1 + (n - 1) * d →
  c = a1 + (k - 1) * d →
  a * (n - k) + b * (k - p) + c * (p - n) = 0 :=
by
  intros h1 h2 h3
  sorry


end arithmetic_progression_condition_l200_200861


namespace wage_ratio_l200_200174

-- Define the conditions
variable (M W : ℝ) -- M stands for man's daily wage, W stands for woman's daily wage
variable (h1 : 40 * 10 * M = 14400) -- Condition 1: 40 men working for 10 days earn Rs. 14400
variable (h2 : 40 * 30 * W = 21600) -- Condition 2: 40 women working for 30 days earn Rs. 21600

-- The statement to prove
theorem wage_ratio (h1 : 40 * 10 * M = 14400) (h2 : 40 * 30 * W = 21600) : M / W = 2 := by
  sorry

end wage_ratio_l200_200174


namespace boys_speed_l200_200187

-- Define the conditions
def sideLength : ℕ := 50
def timeTaken : ℕ := 72

-- Define the goal
theorem boys_speed (sideLength timeTaken : ℕ) (D T : ℝ) :
  D = (4 * sideLength : ℕ) / 1000 ∧
  T = timeTaken / 3600 →
  (D / T = 10) := by
  sorry

end boys_speed_l200_200187


namespace urn_gold_coins_percentage_l200_200010

noncomputable def percentage_gold_coins_in_urn
  (total_objects : ℕ)
  (beads_percentage : ℝ)
  (rings_percentage : ℝ)
  (coins_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  : ℝ := 
  let gold_coins_percentage := 100 - silver_coins_percentage
  let coins_total_percentage := total_objects * coins_percentage / 100
  coins_total_percentage * gold_coins_percentage / 100

theorem urn_gold_coins_percentage 
  (total_objects : ℕ)
  (beads_percentage rings_percentage : ℝ)
  (silver_coins_percentage : ℝ)
  (h1 : beads_percentage = 15)
  (h2 : rings_percentage = 15)
  (h3 : beads_percentage + rings_percentage = 30)
  (h4 : coins_percentage = 100 - 30)
  (h5 : silver_coins_percentage = 35)
  : percentage_gold_coins_in_urn total_objects beads_percentage rings_percentage (100 - 30) 35 = 45.5 :=
sorry

end urn_gold_coins_percentage_l200_200010


namespace simplify_expression_l200_200472

variable (a : ℝ)

theorem simplify_expression (h₁ : a ≠ -3) (h₂ : a ≠ 1) :
  (1 - 4/(a + 3)) / ((a^2 - 2*a + 1) / (2*a + 6)) = 2 / (a - 1) :=
sorry

end simplify_expression_l200_200472


namespace odd_natural_of_form_l200_200844

/-- 
  Prove that the only odd natural number n in the form (p + q) / (p - q)
  where p and q are prime numbers and p > q is 5.
-/
theorem odd_natural_of_form (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p > q) 
  (h2 : ∃ n : ℕ, n = (p + q) / (p - q) ∧ n % 2 = 1) : ∃ n : ℕ, n = 5 :=
sorry

end odd_natural_of_form_l200_200844


namespace smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l200_200087

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- Statement A: The smallest positive period of f(x) is π.
theorem smallest_positive_period_pi : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
by sorry

-- Statement B: If f(x + θ) is an odd function, then one possible value of θ is π/4.
theorem not_odd_at_theta_pi_div_4 : 
  ¬ (∀ x : ℝ, f (x + Real.pi / 4) = -f x) :=
by sorry

-- Statement C: A possible axis of symmetry for f(x) is the line x = π / 3.
theorem axis_of_symmetry_at_pi_div_3 :
  ∀ x : ℝ, f (Real.pi / 3 - x) = f (Real.pi / 3 + x) :=
by sorry

-- Statement D: The maximum value of f(x) on [0, π / 4] is 1.
theorem max_value_not_1_on_interval : 
  ¬ (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 1) :=
by sorry

end smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l200_200087


namespace max_quotient_l200_200186

-- Define the given conditions
def conditions (a b : ℝ) :=
  100 ≤ a ∧ a ≤ 250 ∧ 700 ≤ b ∧ b ≤ 1400

-- State the theorem for the largest value of the quotient b / a
theorem max_quotient (a b : ℝ) (h : conditions a b) : b / a ≤ 14 :=
by
  sorry

end max_quotient_l200_200186


namespace integral_transform_eq_l200_200091

open MeasureTheory

variable (f : ℝ → ℝ)

theorem integral_transform_eq (hf_cont : Continuous f) (hL_exists : ∃ L, ∫ x in (Set.univ : Set ℝ), f x = L) :
  ∃ L, ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = L :=
by
  cases' hL_exists with L hL
  use L
  have h_transform : ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = ∫ x in (Set.univ : Set ℝ), f x := sorry
  rw [h_transform]
  exact hL

end integral_transform_eq_l200_200091


namespace compare_logs_l200_200518

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log (1 / 2)

theorem compare_logs : c < a ∧ a < b := by
  have h0 : a = Real.log 2 / Real.log 3 := rfl
  have h1 : b = Real.log 3 / Real.log 2 := rfl
  have h2 : c = Real.log 5 / Real.log (1 / 2) := rfl
  sorry

end compare_logs_l200_200518


namespace case_b_conditions_l200_200292

-- Definition of the polynomial
def polynomial (p q x : ℝ) : ℝ := x^2 + p * x + q

-- Main theorem
theorem case_b_conditions (p q: ℝ) (x1 x2: ℝ) (hx1: x1 ≤ 0) (hx2: x2 ≥ 2) :
    q ≤ 0 ∧ 2 * p + q + 4 ≤ 0 :=
sorry

end case_b_conditions_l200_200292


namespace even_function_periodic_odd_function_period_generalized_period_l200_200276

-- Problem 1
theorem even_function_periodic (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 2 * a) = f x :=
by sorry

-- Problem 2
theorem odd_function_period (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * a) = f x :=
by sorry

-- Problem 3
theorem generalized_period (f : ℝ → ℝ) (a m n : ℝ) (h₁ : ∀ x : ℝ, 2 * n - f x = f (2 * m - x)) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * (m - a)) = f x :=
by sorry

end even_function_periodic_odd_function_period_generalized_period_l200_200276


namespace subtraction_of_negatives_l200_200468

theorem subtraction_of_negatives :
  -2 - (-3) = 1 := 
by
  sorry

end subtraction_of_negatives_l200_200468


namespace solve_inequality_l200_200538

noncomputable def solution_set (a b : ℝ) (x : ℝ) : Prop :=
x < -1 / b ∨ x > 1 / a

theorem solve_inequality (a b : ℝ) (x : ℝ)
  (h_a : a > 0) (h_b : b > 0) :
  (-b < 1 / x ∧ 1 / x < a) ↔ solution_set a b x :=
by
  sorry

end solve_inequality_l200_200538


namespace find_a_if_f_is_even_l200_200867

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.exp x - a / Real.exp x)

theorem find_a_if_f_is_even
  (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 :=
sorry

end find_a_if_f_is_even_l200_200867


namespace target_avg_weekly_income_l200_200503

-- Define the weekly incomes for the past 5 weeks
def past_incomes : List ℤ := [406, 413, 420, 436, 395]

-- Define the average income over the next 2 weeks
def avg_income_next_two_weeks : ℤ := 365

-- Define the target average weekly income over the 7-week period
theorem target_avg_weekly_income : 
  ((past_incomes.sum + 2 * avg_income_next_two_weeks) / 7 = 400) :=
sorry

end target_avg_weekly_income_l200_200503


namespace right_triangle_unique_perimeter_18_l200_200316

theorem right_triangle_unique_perimeter_18 :
  ∃! (a b c : ℤ), a^2 + b^2 = c^2 ∧ a + b + c = 18 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end right_triangle_unique_perimeter_18_l200_200316


namespace guide_is_knight_l200_200434

-- Definitions
def knight (p : Prop) : Prop := p
def liar (p : Prop) : Prop := ¬p

-- Conditions
variable (GuideClaimsKnight : Prop)
variable (SecondResidentClaimsKnight : Prop)
variable (GuideReportsAccurately : Prop)

-- Proof problem
theorem guide_is_knight
  (GuideClaimsKnight : Prop)
  (SecondResidentClaimsKnight : Prop)
  (GuideReportsAccurately : (GuideClaimsKnight ↔ SecondResidentClaimsKnight)) :
  GuideClaimsKnight := 
sorry

end guide_is_knight_l200_200434


namespace k_value_five_l200_200781

theorem k_value_five (a b k : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a^2 + b^2) / (a * b - 1) = k) : k = 5 := 
sorry

end k_value_five_l200_200781


namespace find_q_l200_200673

theorem find_q (q : Nat) (h : 81 ^ 6 = 3 ^ q) : q = 24 :=
by
  sorry

end find_q_l200_200673


namespace min_omega_l200_200104

theorem min_omega (f : Real → Real) (ω φ : Real) (φ_bound : |φ| < π / 2) 
  (h1 : ω > 0) (h2 : f = fun x => Real.sin (ω * x + φ)) 
  (h3 : f 0 = 1/2) 
  (h4 : ∀ x, f x ≤ f (π / 12)) : ω = 4 := 
by
  sorry

end min_omega_l200_200104


namespace price_increase_eq_20_percent_l200_200727

theorem price_increase_eq_20_percent (a x : ℝ) (h : a * (1 + x) * (1 + x) = a * 1.44) : x = 0.2 :=
by {
  -- This part will contain the proof steps.
  sorry -- Placeholder
}

end price_increase_eq_20_percent_l200_200727


namespace dad_steps_l200_200131

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end dad_steps_l200_200131


namespace problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l200_200482

variable (a b : ℝ)

theorem problem_statement_part1 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a + 2 / b) ≥ 9 := sorry

theorem problem_statement_part2 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (2 ^ a + 4 ^ b) ≥ 2 * Real.sqrt 2 := sorry

theorem problem_statement_part3 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a * b) ≤ (1 / 8) := sorry

theorem problem_statement_part4 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2) ≥ (1 / 5) := sorry

end problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l200_200482


namespace translate_triangle_l200_200478

theorem translate_triangle (A B C A' : (ℝ × ℝ)) (hx_A : A = (2, 1)) (hx_B : B = (4, 3)) 
  (hx_C : C = (0, 2)) (hx_A' : A' = (-1, 5)) : 
  ∃ C' : (ℝ × ℝ), C' = (-3, 6) :=
by 
  sorry

end translate_triangle_l200_200478


namespace evaluate_f_at_3_l200_200335

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^7 + a * x^5 + b * x - 5

theorem evaluate_f_at_3 (a b : ℝ)
  (h : f (-3) a b = 5) : f 3 a b = -15 :=
by
  sorry

end evaluate_f_at_3_l200_200335


namespace max_value_of_3x_plus_4y_l200_200667

theorem max_value_of_3x_plus_4y (x y : ℝ) (h : x^2 + y^2 = 10) : 
  ∃ z, z = 5 * Real.sqrt 10 ∧ z = 3 * x + 4 * y :=
by
  sorry

end max_value_of_3x_plus_4y_l200_200667


namespace evaluate_expression_l200_200349

theorem evaluate_expression :
  (3 ^ 4 * 5 ^ 2 * 7 ^ 3 * 11) / (7 * 11 ^ 2) = 9025 :=
by 
  sorry

end evaluate_expression_l200_200349


namespace total_spent_on_clothing_l200_200258

-- Define the individual costs
def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

-- Define the proof problem to show the total cost
theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  sorry

end total_spent_on_clothing_l200_200258


namespace pyramid_four_triangular_faces_area_l200_200632

noncomputable def pyramid_total_area (base_edge lateral_edge : ℝ) : ℝ :=
  if base_edge = 8 ∧ lateral_edge = 7 then 16 * Real.sqrt 33 else 0

theorem pyramid_four_triangular_faces_area :
  pyramid_total_area 8 7 = 16 * Real.sqrt 33 :=
by
  -- Proof omitted
  sorry

end pyramid_four_triangular_faces_area_l200_200632


namespace parallel_vectors_perpendicular_vectors_l200_200719

/-- Given vectors a and b where a = (1, 2) and b = (x, 1),
    let u = a + b and v = a - b.
    Prove that if u is parallel to v, then x = 1/2. 
    Also, prove that if u is perpendicular to v, then x = 2 or x = -2. --/

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)
noncomputable def vector_u (x : ℝ) : ℝ × ℝ := (1 + x, 3)
noncomputable def vector_v (x : ℝ) : ℝ × ℝ := (1 - x, 1)

theorem parallel_vectors (x : ℝ) :
  (vector_u x).fst / (vector_v x).fst = (vector_u x).snd / (vector_v x).snd ↔ x = 1 / 2 :=
by
  sorry

theorem perpendicular_vectors (x : ℝ) :
  (vector_u x).fst * (vector_v x).fst + (vector_u x).snd * (vector_v x).snd = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end parallel_vectors_perpendicular_vectors_l200_200719


namespace find_sum_of_coordinates_of_other_endpoint_l200_200165

theorem find_sum_of_coordinates_of_other_endpoint :
  ∃ (x y : ℤ), (7, -5) = (10 + x / 2, 4 + y / 2) ∧ x + y = -10 :=
by
  sorry

end find_sum_of_coordinates_of_other_endpoint_l200_200165


namespace jelly_bean_problem_l200_200671

variables {p_r p_o p_y p_g : ℝ}

theorem jelly_bean_problem :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 :=
by
  intros p_r_eq p_o_eq sum_eq
  -- The proof would proceed here, but we avoid proof details
  sorry

end jelly_bean_problem_l200_200671


namespace sin_cos_power_four_l200_200921

theorem sin_cos_power_four (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 7 / 8 := 
sorry

end sin_cos_power_four_l200_200921


namespace f_2016_value_l200_200747

def f : ℝ → ℝ := sorry

axiom f_prop₁ : ∀ x : ℝ, (x + 6) + f x = 0
axiom f_symmetry : ∀ x : ℝ, f (-x) = -f x ∧ f 0 = 0

theorem f_2016_value : f 2016 = 0 :=
by
  sorry

end f_2016_value_l200_200747


namespace find_ABC_l200_200399

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  ∀ (A B C : ℝ),
  (∀ (x : ℝ), x > 2 → g x A B C > 0.3) →
  (∃ (A : ℤ), A = 4) →
  (∃ (B : ℤ), ∃ (C : ℤ), A = 4 ∧ B = 8 ∧ C = -12) →
  A + B + C = 0 :=
by
  intros A B C h1 h2 h3
  rcases h2 with ⟨intA, h2'⟩
  rcases h3 with ⟨intB, ⟨intC, h3'⟩⟩
  simp [h2', h3']
  sorry -- proof skipped

end find_ABC_l200_200399


namespace ratio_AB_PQ_f_half_func_f_l200_200576

-- Define given conditions
variables {m n : ℝ} -- Lengths of AB and PQ
variables {h : ℝ} -- Height of triangle and rectangle (both are 1)
variables {x : ℝ} -- Variable in the range [0, 1]

-- Same area and height conditions
axiom areas_equal : m / 2 = n
axiom height_equal : h = 1

-- Given the areas are equal and height is 1
theorem ratio_AB_PQ : m / n = 2 :=
by sorry -- Proof of the ratio 

-- Given the specific calculation for x = 1/2
theorem f_half (hx : x = 1 / 2) (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  f (1 / 2) = 3 / 4 :=
by sorry -- Proof of function value at 1/2

-- Prove the expression of the function f(x)
theorem func_f (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  ∀ x, 0 ≤ x → x ≤ 1 → f x = 2 * x - x^2 :=
by sorry -- Proof of the function expression


end ratio_AB_PQ_f_half_func_f_l200_200576


namespace original_rent_eq_l200_200560

theorem original_rent_eq (R : ℝ)
  (h1 : 4 * 800 = 3200)
  (h2 : 4 * 850 = 3400)
  (h3 : 3400 - 3200 = 200)
  (h4 : 200 = 0.25 * R) : R = 800 := by
  sorry

end original_rent_eq_l200_200560


namespace find_reading_l200_200124

variable (a_1 a_2 a_3 a_4 : ℝ) (x : ℝ)
variable (h1 : a_1 = 2) (h2 : a_2 = 2.1) (h3 : a_3 = 2) (h4 : a_4 = 2.2)
variable (mean : (a_1 + a_2 + a_3 + a_4 + x) / 5 = 2)

theorem find_reading : x = 1.7 :=
by
  sorry

end find_reading_l200_200124


namespace yearly_water_consumption_correct_l200_200173

def monthly_water_consumption : ℝ := 182.88
def months_in_a_year : ℕ := 12
def yearly_water_consumption : ℝ := monthly_water_consumption * (months_in_a_year : ℝ)

theorem yearly_water_consumption_correct :
  yearly_water_consumption = 2194.56 :=
by
  sorry

end yearly_water_consumption_correct_l200_200173


namespace equation_of_perpendicular_line_l200_200698

theorem equation_of_perpendicular_line :
  ∃ c : ℝ, (∀ x y : ℝ, (2 * x + y + c = 0 ↔ (x = 1 ∧ y = 1))) → (c = -3) := 
by
  sorry

end equation_of_perpendicular_line_l200_200698


namespace find_a_plus_b_l200_200658

theorem find_a_plus_b (a b x : ℝ) (h1 : x + 2 * a > 4) (h2 : 2 * x < b)
  (h3 : 0 < x) (h4 : x < 2) : a + b = 6 :=
by
  sorry

end find_a_plus_b_l200_200658


namespace bronze_medals_l200_200894

theorem bronze_medals (G S B : ℕ) 
  (h1 : G + S + B = 89) 
  (h2 : G + S = 4 * B - 6) :
  B = 19 :=
sorry

end bronze_medals_l200_200894


namespace largest_possible_b_l200_200185

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 :=
by
  sorry

end largest_possible_b_l200_200185


namespace coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l200_200800

theorem coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45 :
  let general_term (r : ℕ) := (Nat.choose 10 r) * (x^(10 - 3 * r)/2)
  ∃ r : ℕ, (general_term r) = 2 ∧ (Nat.choose 10 r) = 45 :=
by
  sorry

end coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l200_200800


namespace smallest_positive_integer_in_form_l200_200736

theorem smallest_positive_integer_in_form (m n : ℤ) : 
  ∃ m n : ℤ, 3001 * m + 24567 * n = 1 :=
by
  sorry

end smallest_positive_integer_in_form_l200_200736


namespace water_formed_l200_200564

theorem water_formed (CaOH2 CO2 CaCO3 H2O : Nat) 
  (h_balanced : ∀ n, n * CaOH2 + n * CO2 = n * CaCO3 + n * H2O)
  (h_initial : CaOH2 = 2 ∧ CO2 = 2) : 
  H2O = 2 :=
by
  sorry

end water_formed_l200_200564


namespace average_mark_of_excluded_students_l200_200601

theorem average_mark_of_excluded_students 
  (N A E A_remaining : ℕ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hA_remaining : A_remaining = 95) : 
  ∃ A_excluded : ℕ, A_excluded = 20 :=
by
  -- Use the conditions in the proof.
  sorry

end average_mark_of_excluded_students_l200_200601


namespace sum_of_squares_of_coefficients_l200_200934

theorem sum_of_squares_of_coefficients :
  ∃ a b c d e f : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 :=
by
  sorry

end sum_of_squares_of_coefficients_l200_200934


namespace total_revenue_correct_l200_200748

def items : Type := ℕ × ℝ

def magazines : items := (425, 2.50)
def newspapers : items := (275, 1.50)
def books : items := (150, 5.00)
def pamphlets : items := (75, 0.50)

def revenue (item : items) : ℝ := item.1 * item.2

def total_revenue : ℝ :=
  revenue magazines +
  revenue newspapers +
  revenue books +
  revenue pamphlets

theorem total_revenue_correct : total_revenue = 2262.50 := by
  sorry

end total_revenue_correct_l200_200748


namespace claudia_ratio_of_kids_l200_200627

def claudia_art_class :=
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  sunday_kids / saturday_kids = 1 / 2

theorem claudia_ratio_of_kids :
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  (sunday_kids / saturday_kids = 1 / 2) :=
by
  sorry

end claudia_ratio_of_kids_l200_200627


namespace solve_system_l200_200904

def inequality1 (x : ℝ) : Prop := 5 / (x + 3) ≥ 1

def inequality2 (x : ℝ) : Prop := x^2 + x - 2 ≥ 0

def solution (x : ℝ) : Prop := (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)

theorem solve_system (x : ℝ) : inequality1 x ∧ inequality2 x → solution x := by
  sorry

end solve_system_l200_200904


namespace non_officers_count_l200_200492

theorem non_officers_count 
    (avg_salary_employees : ℝ) 
    (avg_salary_officers : ℝ) 
    (avg_salary_non_officers : ℝ) 
    (num_officers : ℕ) : 
    avg_salary_employees = 120 ∧ avg_salary_officers = 470 ∧ avg_salary_non_officers = 110 ∧ num_officers = 15 → 
    ∃ N : ℕ, N = 525 ∧ 
    (num_officers * avg_salary_officers + N * avg_salary_non_officers) / (num_officers + N) = avg_salary_employees := 
by 
    sorry

end non_officers_count_l200_200492


namespace elevator_initial_floors_down_l200_200086

theorem elevator_initial_floors_down (x : ℕ) (h1 : 9 - x + 3 + 8 = 13) : x = 7 := 
by
  -- Proof
  sorry

end elevator_initial_floors_down_l200_200086


namespace max_profit_at_9_l200_200001

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 10.8 - (1 / 30) * x^2
else if h : x > 10 then 108 / x - 1000 / (3 * x^2)
else 0

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 8.1 * x - x^3 / 30 - 10
else if h : x > 10 then 98 - 1000 / (3 * x) - 2.7 * x
else 0

theorem max_profit_at_9 : W 9 = 38.6 :=
sorry

end max_profit_at_9_l200_200001


namespace line_through_points_on_parabola_l200_200296

theorem line_through_points_on_parabola
  (p q : ℝ)
  (hpq : p^2 - 4 * q > 0) :
  ∃ (A B : ℝ × ℝ),
    (exists (x₁ x₂ : ℝ), x₁^2 + p * x₁ + q = 0 ∧ x₂^2 + p * x₂ + q = 0 ∧
                         A = (x₁, x₁^2 / 3) ∧ B = (x₂, x₂^2 / 3) ∧
                         (∀ x y, (x, y) = A ∨ (x, y) = B → px + 3 * y + q = 0)) :=
sorry

end line_through_points_on_parabola_l200_200296


namespace Yi_visited_city_A_l200_200545

variable (visited : String -> String -> Prop) -- denote visited "Student" "City"
variables (Jia Yi Bing : String) (A B C : String)

theorem Yi_visited_city_A
  (h1 : visited Jia A ∧ visited Jia C ∧ ¬ visited Jia B)
  (h2 : ¬ visited Yi C)
  (h3 : visited Jia A ∧ visited Yi A ∧ visited Bing A) :
  visited Yi A :=
by
  sorry

end Yi_visited_city_A_l200_200545


namespace remainder_of_sum_l200_200261

theorem remainder_of_sum (a b c : ℕ) (h1 : a % 15 = 8) (h2 : b % 15 = 12) (h3 : c % 15 = 13) : (a + b + c) % 15 = 3 := 
by
  sorry

end remainder_of_sum_l200_200261


namespace christine_needs_32_tablespoons_l200_200624

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l200_200624


namespace complete_the_square_l200_200348

theorem complete_the_square (x : ℝ) : 
    (x^2 - 2 * x - 5 = 0) -> (x - 1)^2 = 6 :=
by sorry

end complete_the_square_l200_200348


namespace greatest_consecutive_sum_l200_200133

theorem greatest_consecutive_sum (S : ℤ) (hS : S = 105) : 
  ∃ N : ℤ, (∃ a : ℤ, (N * (2 * a + N - 1) = 2 * S)) ∧ 
  (∀ M : ℤ, (∃ b : ℤ, (M * (2 * b + M - 1) = 2 * S)) → M ≤ N) ∧ N = 210 := 
sorry

end greatest_consecutive_sum_l200_200133


namespace inequality_solution_l200_200856

noncomputable def solve_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo (-3 : ℝ) 3

theorem inequality_solution (x : ℝ) (h : x ≠ -3) :
  (x^2 - 9) / (x + 3) < 0 ↔ solve_inequality x :=
by
  sorry

end inequality_solution_l200_200856


namespace seeds_per_can_l200_200488

theorem seeds_per_can (total_seeds : ℕ) (num_cans : ℕ) (h1 : total_seeds = 54) (h2 : num_cans = 9) : total_seeds / num_cans = 6 :=
by {
  sorry
}

end seeds_per_can_l200_200488


namespace animals_not_like_either_l200_200140

def total_animals : ℕ := 75
def animals_eat_carrots : ℕ := 26
def animals_like_hay : ℕ := 56
def animals_like_both : ℕ := 14

theorem animals_not_like_either : (total_animals - (animals_eat_carrots - animals_like_both + animals_like_hay - animals_like_both + animals_like_both)) = 7 := by
  sorry

end animals_not_like_either_l200_200140


namespace union_is_correct_l200_200640

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem union_is_correct : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by
  sorry

end union_is_correct_l200_200640


namespace polynomial_factorization_example_l200_200156

open Polynomial

theorem polynomial_factorization_example
  (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) (hf : ∀ i ∈ [a_5, a_4, a_3, a_2, a_1, a_0], |i| ≤ 4)
  (b_3 b_2 b_1 b_0 : ℤ) (hg : ∀ i ∈ [b_3, b_2, b_1, b_0], |i| ≤ 1)
  (c_2 c_1 c_0 : ℤ) (hh : ∀ i ∈ [c_2, c_1, c_0], |i| ≤ 1)
  (h : (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0).eval 10 =
       ((C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0)).eval 10) :
  (C a_5 * X^5 + C a_4 * X^4 + C a_3 * X^3 + C a_2 * X^2 + C a_1 * X + C a_0) =
  (C b_3 * X^3 + C b_2 * X^2 + C b_1 * X + C b_0) * (C c_2 * X^2 + C c_1 * X + C c_0) :=
sorry

end polynomial_factorization_example_l200_200156


namespace inequality_problem_l200_200710

theorem inequality_problem
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  (b^2 / a + a^2 / b) ≥ (a + b) :=
sorry

end inequality_problem_l200_200710


namespace distinct_solutions_abs_eq_l200_200828

theorem distinct_solutions_abs_eq : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (|2 * x1 - 14| = |x1 + 4| ∧ |2 * x2 - 14| = |x2 + 4|) ∧ (∀ x, |2 * x - 14| = |x + 4| → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end distinct_solutions_abs_eq_l200_200828


namespace quadratic_no_real_roots_l200_200679

-- Define the quadratic polynomial f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions: f(x) = x has no real roots
theorem quadratic_no_real_roots (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
sorry

end quadratic_no_real_roots_l200_200679


namespace no_hexagonal_pyramid_with_equal_edges_l200_200822

theorem no_hexagonal_pyramid_with_equal_edges (edges : ℕ → ℝ)
  (regular_polygon : ℕ → ℝ → Prop)
  (equal_length_edges : ∀ (n : ℕ), regular_polygon n (edges n) → ∀ i j, edges i = edges j)
  (apex_above_centroid : ∀ (n : ℕ) (h : regular_polygon n (edges n)), True) :
  ¬ regular_polygon 6 (edges 6) :=
by
  sorry

end no_hexagonal_pyramid_with_equal_edges_l200_200822


namespace area_outside_two_small_squares_l200_200440

theorem area_outside_two_small_squares (L S : ℝ) (hL : L = 9) (hS : S = 4) :
  let large_square_area := L^2
  let small_square_area := S^2
  let combined_small_squares_area := 2 * small_square_area
  large_square_area - combined_small_squares_area = 49 :=
by
  sorry

end area_outside_two_small_squares_l200_200440


namespace d_share_l200_200299

theorem d_share (x : ℝ) (d c : ℝ)
  (h1 : c = 3 * x + 500)
  (h2 : d = 3 * x)
  (h3 : c = 4 * x) :
  d = 1500 := 
by 
  sorry

end d_share_l200_200299


namespace hyperbola_focus_l200_200955

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_focus
  (a b : ℝ)
  (hEq : ∀ x y : ℝ, ((x - 1)^2 / a^2) - ((y - 10)^2 / b^2) = 1):
  (1 + c 7 3, 10) = (1 + Real.sqrt (7^2 + 3^2), 10) :=
by
  sorry

end hyperbola_focus_l200_200955


namespace problem_solution_l200_200843

variable (a : ℝ)

theorem problem_solution (h : a ≠ 0) : a^2 + 1 > 1 :=
sorry

end problem_solution_l200_200843


namespace arabella_first_step_time_l200_200294

def time_first_step (x : ℝ) : Prop :=
  let time_second_step := x / 2
  let time_third_step := x + x / 2
  (x + time_second_step + time_third_step = 90)

theorem arabella_first_step_time (x : ℝ) (h : time_first_step x) : x = 30 :=
by
  sorry

end arabella_first_step_time_l200_200294


namespace dave_winfield_home_runs_correct_l200_200464

def dave_winfield_home_runs (W : ℕ) : Prop :=
  755 = 2 * W - 175

theorem dave_winfield_home_runs_correct : dave_winfield_home_runs 465 :=
by
  -- The proof is omitted as requested
  sorry

end dave_winfield_home_runs_correct_l200_200464


namespace tetrahedron_mistaken_sum_l200_200709

theorem tetrahedron_mistaken_sum :
  let edges := 6
  let vertices := 4
  let faces := 4
  let joe_count := vertices + 1  -- Joe counts one vertex twice
  edges + joe_count + faces = 15 := by
  sorry

end tetrahedron_mistaken_sum_l200_200709


namespace f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l200_200660
open Real

noncomputable def f : ℝ → ℝ := sorry

theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := sorry
theorem f_positive_lt_x_zero (x : ℝ) (h_pos : 0 < x) : f x < 0 := sorry
theorem f_at_one : f 1 = 1 := sorry

-- Prove that f is an odd function
theorem f_odd (x : ℝ) : f (-x) = -f x :=
  sorry

-- Solve the inequality: f((log2 x)^2 - log2 (x^2)) > 3
theorem f_inequality (x : ℝ) (h_pos : 0 < x) : (f ((log x / log 2)^2 - (log x^2 / log 2))) > 3 ↔ 1 / 2 < x ∧ x < 8 :=
  sorry

end f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l200_200660


namespace cannot_be_combined_with_sqrt2_l200_200832

def can_be_combined (x y : ℝ) : Prop := ∃ k : ℝ, k * x = y

theorem cannot_be_combined_with_sqrt2 :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 8
  let c := Real.sqrt 12
  let d := -Real.sqrt 18
  ¬ can_be_combined c (Real.sqrt 2) := 
by
  sorry

end cannot_be_combined_with_sqrt2_l200_200832


namespace find_triple_sum_l200_200978

theorem find_triple_sum (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = 1 - 4 * y)
  (h3 : x + y = -12 - 4 * z) :
  3 * x + 3 * y + 3 * z = 9 / 2 := 
sorry

end find_triple_sum_l200_200978


namespace base_angle_isosceles_triangle_l200_200543

theorem base_angle_isosceles_triangle
  (sum_angles : ∀ (α β γ : ℝ), α + β + γ = 180)
  (isosceles : ∀ (α β : ℝ), α = β)
  (one_angle_forty : ∃ α : ℝ, α = 40) :
  ∃ β : ℝ, β = 70 ∨ β = 40 :=
by
  sorry

end base_angle_isosceles_triangle_l200_200543
