import Mathlib

namespace NUMINAMATH_GPT_select_16_genuine_coins_l141_14171

theorem select_16_genuine_coins (coins : Finset ℕ) (h_coins_count : coins.card = 40) 
  (counterfeit : Finset ℕ) (h_counterfeit_count : counterfeit.card = 3)
  (h_counterfeit_lighter : ∀ c ∈ counterfeit, ∀ g ∈ (coins \ counterfeit), c < g) :
  ∃ genuine : Finset ℕ, genuine.card = 16 ∧ 
    (∀ h1 h2 h3 : Finset ℕ, h1.card = 20 → h2.card = 10 → h3.card = 8 →
      ((h1 ⊆ coins ∧ h2 ⊆ h1 ∧ h3 ⊆ (h1 \ counterfeit)) ∨
       (h1 ⊆ coins ∧ h2 ⊆ (h1 \ counterfeit) ∧ h3 ⊆ (h2 \ counterfeit))) →
      genuine ⊆ coins \ counterfeit) :=
sorry

end NUMINAMATH_GPT_select_16_genuine_coins_l141_14171


namespace NUMINAMATH_GPT_product_of_three_numbers_l141_14164

theorem product_of_three_numbers (p q r m : ℝ) (h1 : p + q + r = 180) (h2 : m = 8 * p)
  (h3 : m = q - 10) (h4 : m = r + 10) : p * q * r = 90000 := by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l141_14164


namespace NUMINAMATH_GPT_solution_set_of_inequality_l141_14168

variable {a x : ℝ}

theorem solution_set_of_inequality (h : 2 * a + 1 < 0) : 
  {x : ℝ | x^2 - 4 * a * x - 5 * a^2 > 0} = {x | x < 5 * a ∨ x > -a} := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l141_14168


namespace NUMINAMATH_GPT_jake_watched_friday_l141_14187

theorem jake_watched_friday
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (thursday_hours : ℕ)
  (total_hours : ℕ)
  (day_hours : ℕ := 24) :
  monday_hours = (day_hours / 2) →
  tuesday_hours = 4 →
  wednesday_hours = (day_hours / 4) →
  thursday_hours = ((monday_hours + tuesday_hours + wednesday_hours) / 2) →
  total_hours = 52 →
  (total_hours - (monday_hours + tuesday_hours + wednesday_hours + thursday_hours)) = 19 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_jake_watched_friday_l141_14187


namespace NUMINAMATH_GPT_num_correct_statements_l141_14149

def doubleAbsDiff (a b c d : ℝ) : ℝ :=
  |a - b| - |c - d|

theorem num_correct_statements : 
  (∀ a b c d : ℝ, (a, b, c, d) = (24, 25, 29, 30) → 
    (doubleAbsDiff a b c d = 0) ∨
    (doubleAbsDiff a c b d = 0) ∨
    (doubleAbsDiff a d b c = -0.5) ∨
    (doubleAbsDiff b c a d = 0.5)) → 
  (∀ x : ℝ, x ≥ 2 → 
    doubleAbsDiff (x^2) (2*x) 1 1 = 7 → 
    (x^4 + 2401 / x^4 = 226)) →
  (∀ x : ℝ, x ≥ -2 → 
    (doubleAbsDiff (2*x-5) (3*x-2) (4*x-1) (5*x+3)) ≠ 0) →
  (0 = 0)
:= by
  sorry

end NUMINAMATH_GPT_num_correct_statements_l141_14149


namespace NUMINAMATH_GPT_frac_calc_l141_14188

theorem frac_calc : (2 / 9) * (5 / 11) + 1 / 3 = 43 / 99 :=
by sorry

end NUMINAMATH_GPT_frac_calc_l141_14188


namespace NUMINAMATH_GPT_base_8_subtraction_l141_14167

theorem base_8_subtraction : 
  let x := 0o1234   -- 1234 in base 8
  let y := 0o765    -- 765 in base 8
  let result := 0o225 -- 225 in base 8
  x - y = result := by sorry

end NUMINAMATH_GPT_base_8_subtraction_l141_14167


namespace NUMINAMATH_GPT_distinct_real_nums_condition_l141_14105

theorem distinct_real_nums_condition 
  (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : r ≠ p)
  (h4 : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_nums_condition_l141_14105


namespace NUMINAMATH_GPT_fraction_of_seniors_study_japanese_l141_14144

variable (J S : ℝ)
variable (fraction_seniors fraction_juniors : ℝ)
variable (total_fraction_study_japanese : ℝ)

theorem fraction_of_seniors_study_japanese 
  (h1 : S = 2 * J)
  (h2 : fraction_juniors = 3 / 4)
  (h3 : total_fraction_study_japanese = 1 / 3) :
  fraction_seniors = 1 / 8 :=
by
  -- Here goes the proof.
  sorry

end NUMINAMATH_GPT_fraction_of_seniors_study_japanese_l141_14144


namespace NUMINAMATH_GPT_find_x_when_y_neg_10_l141_14141

def inversely_proportional (x y : ℝ) (k : ℝ) := x * y = k

theorem find_x_when_y_neg_10 (k : ℝ) (h₁ : inversely_proportional 4 (-2) k) (yval : y = -10) 
: ∃ x, inversely_proportional x y k ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_find_x_when_y_neg_10_l141_14141


namespace NUMINAMATH_GPT_circumscribed_quadrilateral_arc_sum_l141_14161

theorem circumscribed_quadrilateral_arc_sum 
  (a b c d : ℝ) 
  (h : a + b + c + d = 360) : 
  (1/2 * (b + c + d)) + (1/2 * (a + c + d)) + (1/2 * (a + b + d)) + (1/2 * (a + b + c)) = 540 :=
by
  sorry

end NUMINAMATH_GPT_circumscribed_quadrilateral_arc_sum_l141_14161


namespace NUMINAMATH_GPT_theo_cookies_eaten_in_9_months_l141_14178

-- Define the basic variable values as per the conditions
def cookiesPerTime : Nat := 25
def timesPerDay : Nat := 5
def daysPerMonth : Nat := 27
def numMonths : Nat := 9

-- Define the total number of cookies Theo can eat in 9 months
def totalCookiesIn9Months : Nat :=
  cookiesPerTime * timesPerDay * daysPerMonth * numMonths

-- The theorem stating the answer
theorem theo_cookies_eaten_in_9_months :
  totalCookiesIn9Months = 30375 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_theo_cookies_eaten_in_9_months_l141_14178


namespace NUMINAMATH_GPT_bears_in_shipment_l141_14160

theorem bears_in_shipment (initial_bears shipped_bears bears_per_shelf shelves_used : ℕ) 
  (h1 : initial_bears = 4) 
  (h2 : bears_per_shelf = 7) 
  (h3 : shelves_used = 2) 
  (total_bears_on_shelves : ℕ) 
  (h4 : total_bears_on_shelves = shelves_used * bears_per_shelf) 
  (total_bears_after_shipment : ℕ) 
  (h5 : total_bears_after_shipment = total_bears_on_shelves) 
  : shipped_bears = total_bears_on_shelves - initial_bears := 
sorry

end NUMINAMATH_GPT_bears_in_shipment_l141_14160


namespace NUMINAMATH_GPT_sequence_sum_after_6_steps_l141_14122

noncomputable def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else if n = 1 then 3
  else if n = 2 then 15
  else if n = 3 then 1435 -- would define how numbers sequence works recursively.
  else sorry -- next steps up to 6
  

theorem sequence_sum_after_6_steps : sequence_sum 6 = 191 := 
by
  sorry

end NUMINAMATH_GPT_sequence_sum_after_6_steps_l141_14122


namespace NUMINAMATH_GPT_part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l141_14174

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + 1
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

-- (Ⅰ) Equation of the tangent line to y = f(x) at x = 1
theorem part1_tangent_line_at_1 : ∀ x, (f 1 + (1 / 1) * (x - 1)) = x - 1 := sorry

-- (Ⅱ) Intervals where F(x) is monotonic
theorem part2_monotonic_intervals (a : ℝ) : 
  (a ≤ 0 → ∀ x > 0, F x a > 0) ∧ 
  (a > 0 → (∀ x > 0, x < (1 / a) → F x a > 0) ∧ (∀ x > 1 / a, F x a < 0)) := sorry

-- (Ⅲ) Range of a for which f(x) is below g(x) for all x > 0
theorem part3_range_of_a (a : ℝ) : (∀ x > 0, f x < g x a) ↔ a ∈ Set.Ioi (Real.exp (-2)) := sorry

end NUMINAMATH_GPT_part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l141_14174


namespace NUMINAMATH_GPT_greatest_possible_gcd_l141_14124

theorem greatest_possible_gcd (d : ℕ) (a : ℕ → ℕ) (h_sum : (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 595)
  (h_gcd : ∀ i, d ∣ a i) : d ≤ 35 :=
sorry

end NUMINAMATH_GPT_greatest_possible_gcd_l141_14124


namespace NUMINAMATH_GPT_trig_log_exp_identity_l141_14143

theorem trig_log_exp_identity : 
  (Real.sin (330 * Real.pi / 180) + 
   (Real.sqrt 2 - 1)^0 + 
   3^(Real.log 2 / Real.log 3)) = 5 / 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_trig_log_exp_identity_l141_14143


namespace NUMINAMATH_GPT_vans_capacity_l141_14181

def students : ℕ := 33
def adults : ℕ := 9
def vans : ℕ := 6

def total_people : ℕ := students + adults
def people_per_van : ℕ := total_people / vans

theorem vans_capacity : people_per_van = 7 := by
  sorry

end NUMINAMATH_GPT_vans_capacity_l141_14181


namespace NUMINAMATH_GPT_train_travel_time_l141_14114

def travel_time (departure arrival : Nat) : Nat :=
  arrival - departure

theorem train_travel_time : travel_time 425 479 = 54 := by
  sorry

end NUMINAMATH_GPT_train_travel_time_l141_14114


namespace NUMINAMATH_GPT_Cubs_home_runs_third_inning_l141_14121

variable (X : ℕ)

theorem Cubs_home_runs_third_inning 
  (h : X + 1 + 2 = 2 + 3) : 
  X = 2 :=
by 
  sorry

end NUMINAMATH_GPT_Cubs_home_runs_third_inning_l141_14121


namespace NUMINAMATH_GPT_max_tetrahedron_in_cube_l141_14129

open Real

noncomputable def cube_edge_length : ℝ := 6
noncomputable def max_tetrahedron_edge_length (a : ℝ) : Prop :=
  ∃ x : ℝ, x = 2 * sqrt 6 ∧ 
          (∃ R : ℝ, R = (a * sqrt 3) / 2 ∧ x / sqrt (2 / 3) = 4 * R / 3)

theorem max_tetrahedron_in_cube : max_tetrahedron_edge_length cube_edge_length :=
sorry

end NUMINAMATH_GPT_max_tetrahedron_in_cube_l141_14129


namespace NUMINAMATH_GPT_not_always_possible_triangle_sides_l141_14112

theorem not_always_possible_triangle_sides (α β γ δ : ℝ) 
  (h1 : α + β + γ + δ = 360) 
  (h2 : α < 180) 
  (h3 : β < 180) 
  (h4 : γ < 180) 
  (h5 : δ < 180) : 
  ¬ (∀ (x y z : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) ∧ (y = α ∨ y = β ∨ y = γ ∨ y = δ) ∧ (z = α ∨ z = β ∨ z = γ ∨ z = δ) ∧ (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) → x + y > z ∧ x + z > y ∧ y + z > x)
:= sorry

end NUMINAMATH_GPT_not_always_possible_triangle_sides_l141_14112


namespace NUMINAMATH_GPT_sacks_required_in_4_weeks_l141_14175

-- Definitions for the weekly requirements of each bakery
def weekly_sacks_bakery1 : Nat := 2
def weekly_sacks_bakery2 : Nat := 4
def weekly_sacks_bakery3 : Nat := 12

-- Total weeks considered
def weeks : Nat := 4

-- Calculating the total sacks needed for all bakeries over the given weeks
def total_sacks_needed : Nat :=
  (weekly_sacks_bakery1 * weeks) +
  (weekly_sacks_bakery2 * weeks) +
  (weekly_sacks_bakery3 * weeks)

-- The theorem to be proven
theorem sacks_required_in_4_weeks :
  total_sacks_needed = 72 :=
by
  sorry

end NUMINAMATH_GPT_sacks_required_in_4_weeks_l141_14175


namespace NUMINAMATH_GPT_johns_weekly_earnings_after_raise_l141_14184

theorem johns_weekly_earnings_after_raise 
  (original_weekly_earnings : ℕ) 
  (percentage_increase : ℝ) 
  (new_weekly_earnings : ℕ)
  (h1 : original_weekly_earnings = 60)
  (h2 : percentage_increase = 0.16666666666666664) :
  new_weekly_earnings = 70 :=
sorry

end NUMINAMATH_GPT_johns_weekly_earnings_after_raise_l141_14184


namespace NUMINAMATH_GPT_number_of_boys_l141_14193

-- Definitions of the conditions
def total_members (B G : ℕ) : Prop := B + G = 26
def meeting_attendance (B G : ℕ) : Prop := (1 / 2 : ℚ) * G + B = 16

-- Theorem statement
theorem number_of_boys (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : B = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l141_14193


namespace NUMINAMATH_GPT_triangle_is_either_isosceles_or_right_angled_l141_14111

theorem triangle_is_either_isosceles_or_right_angled
  (A B : Real)
  (a b c : Real)
  (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  : a = b ∨ a^2 + b^2 = c^2 :=
sorry

end NUMINAMATH_GPT_triangle_is_either_isosceles_or_right_angled_l141_14111


namespace NUMINAMATH_GPT_total_number_of_coins_is_336_l141_14116

theorem total_number_of_coins_is_336 (N20 : ℕ) (N25 : ℕ) (total_value_rupees : ℚ)
    (h1 : N20 = 260) (h2 : total_value_rupees = 71) (h3 : 20 * N20 + 25 * N25 = 7100) :
    N20 + N25 = 336 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_is_336_l141_14116


namespace NUMINAMATH_GPT_layla_earnings_l141_14155

def rate_donaldsons : ℕ := 15
def bonus_donaldsons : ℕ := 5
def hours_donaldsons : ℕ := 7
def rate_merck : ℕ := 18
def discount_merck : ℝ := 0.10
def hours_merck : ℕ := 6
def rate_hille : ℕ := 20
def bonus_hille : ℕ := 10
def hours_hille : ℕ := 3
def rate_johnson : ℕ := 22
def flat_rate_johnson : ℕ := 80
def hours_johnson : ℕ := 4
def rate_ramos : ℕ := 25
def bonus_ramos : ℕ := 20
def hours_ramos : ℕ := 2

def donaldsons_earnings := rate_donaldsons * hours_donaldsons + bonus_donaldsons
def merck_earnings := rate_merck * hours_merck - (rate_merck * hours_merck * discount_merck : ℝ)
def hille_earnings := rate_hille * hours_hille + bonus_hille
def johnson_earnings := rate_johnson * hours_johnson
def ramos_earnings := rate_ramos * hours_ramos + bonus_ramos

noncomputable def total_earnings : ℝ :=
  donaldsons_earnings + merck_earnings + hille_earnings + johnson_earnings + ramos_earnings

theorem layla_earnings : total_earnings = 435.2 :=
by
  sorry

end NUMINAMATH_GPT_layla_earnings_l141_14155


namespace NUMINAMATH_GPT_centroid_y_sum_zero_l141_14138

theorem centroid_y_sum_zero
  (x1 x2 x3 y2 y3 : ℝ)
  (h : y2 + y3 = 0) :
  (x1 + x2 + x3) / 3 = (x1 / 3 + x2 / 3 + x3 / 3) ∧ (y2 + y3) / 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_centroid_y_sum_zero_l141_14138


namespace NUMINAMATH_GPT_martian_right_angle_l141_14139

theorem martian_right_angle :
  ∀ (full_circle clerts_per_right_angle : ℕ),
  (full_circle = 600) →
  (clerts_per_right_angle = full_circle / 3) →
  clerts_per_right_angle = 200 :=
by
  intros full_circle clerts_per_right_angle h1 h2
  sorry

end NUMINAMATH_GPT_martian_right_angle_l141_14139


namespace NUMINAMATH_GPT_pentagon_coloring_valid_l141_14173

-- Define the colors
inductive Color
| Red
| Blue

-- Define the vertices as a type
inductive Vertex
| A | B | C | D | E

open Vertex Color

-- Define an edge as a pair of vertices
def Edge := Vertex × Vertex

-- Define the coloring function
def color : Edge → Color := sorry

-- Define the pentagon
def pentagon_edges : List Edge :=
  [(A, B), (B, C), (C, D), (D, E), (E, A), (A, C), (A, D), (A, E), (B, D), (B, E), (C, E)]

-- Define the condition for a valid triangle coloring
def valid_triangle_coloring (e1 e2 e3 : Edge) : Prop :=
  (color e1 = Red ∧ (color e2 = Blue ∨ color e3 = Blue)) ∨
  (color e2 = Red ∧ (color e1 = Blue ∨ color e3 = Blue)) ∨
  (color e3 = Red ∧ (color e1 = Blue ∨ color e2 = Blue))

-- Define the condition for all triangles formed by the vertices of the pentagon
def all_triangles_valid : Prop :=
  ∀ v1 v2 v3 : Vertex,
    v1 ≠ v2 → v2 ≠ v3 → v1 ≠ v3 →
    valid_triangle_coloring (v1, v2) (v2, v3) (v1, v3)

-- Statement: Prove that there are 12 valid ways to color the pentagon
theorem pentagon_coloring_valid : (∃ (coloring : Edge → Color), all_triangles_valid) :=
  sorry

end NUMINAMATH_GPT_pentagon_coloring_valid_l141_14173


namespace NUMINAMATH_GPT_fishes_per_body_of_water_l141_14102

-- Define the number of bodies of water
def n_b : Nat := 6

-- Define the total number of fishes
def n_f : Nat := 1050

-- Prove the number of fishes per body of water
theorem fishes_per_body_of_water : n_f / n_b = 175 := by 
  sorry

end NUMINAMATH_GPT_fishes_per_body_of_water_l141_14102


namespace NUMINAMATH_GPT_OH_over_ON_eq_2_no_other_common_points_l141_14159

noncomputable def coordinates (t p : ℝ) : ℝ × ℝ :=
  (t^2 / (2 * p), t)

noncomputable def symmetric_point (M P : ℝ × ℝ) : ℝ × ℝ :=
  let (xM, yM) := M;
  let (xP, yP) := P;
  (2 * xP - xM, 2 * yP - yM)

noncomputable def line_ON (p t : ℝ) : ℝ → ℝ :=
  λ x => (p / t) * x

noncomputable def line_MH (t p : ℝ) : ℝ → ℝ :=
  λ x => (p / (2 * t)) * x + t

noncomputable def point_H (t p : ℝ) : ℝ × ℝ :=
  (2 * t^2 / p, 2 * t)

theorem OH_over_ON_eq_2
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  (H.snd) / (N.snd) = 2 := by
  sorry

theorem no_other_common_points
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  ∀ y, (y ≠ H.snd → ¬ ∃ x, line_MH t p x = y ∧ y^2 = 2 * p * x) := by 
  sorry

end NUMINAMATH_GPT_OH_over_ON_eq_2_no_other_common_points_l141_14159


namespace NUMINAMATH_GPT_percentage_difference_l141_14119

theorem percentage_difference : (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l141_14119


namespace NUMINAMATH_GPT_gcd_of_all_elements_in_B_is_2_l141_14199

-- Define the set B as the set of all numbers that can be represented as the sum of four consecutive positive integers.
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2 ∧ x > 0}

-- Translate the question to a Lean statement.
theorem gcd_of_all_elements_in_B_is_2 : ∀ n ∈ B, gcd n 2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_all_elements_in_B_is_2_l141_14199


namespace NUMINAMATH_GPT_complex_multiplication_l141_14191

theorem complex_multiplication :
  ∀ (i : ℂ), i^2 = -1 → (1 - i) * i = 1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l141_14191


namespace NUMINAMATH_GPT_gain_percentage_l141_14198

theorem gain_percentage (C S : ℝ) (h : 80 * C = 25 * S) : 220 = ((S - C) / C) * 100 :=
by sorry

end NUMINAMATH_GPT_gain_percentage_l141_14198


namespace NUMINAMATH_GPT_subtract_real_numbers_l141_14113

theorem subtract_real_numbers : 3.56 - 1.89 = 1.67 :=
by
  sorry

end NUMINAMATH_GPT_subtract_real_numbers_l141_14113


namespace NUMINAMATH_GPT_find_constants_l141_14147

theorem find_constants (P Q R : ℚ) (h : ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) = (P / (x - 1)) + (Q / (x - 4)) + (R / (x - 6))) : 
  (P, Q, R) = (-4/5, -1/2, 23/10) := 
  sorry

end NUMINAMATH_GPT_find_constants_l141_14147


namespace NUMINAMATH_GPT_maria_uses_666_blocks_l141_14192

theorem maria_uses_666_blocks :
  let original_volume := 15 * 12 * 7
  let interior_length := 15 - 2 * 1.5
  let interior_width := 12 - 2 * 1.5
  let interior_height := 7 - 1.5
  let interior_volume := interior_length * interior_width * interior_height
  let blocks_volume := original_volume - interior_volume
  blocks_volume = 666 :=
by
  sorry

end NUMINAMATH_GPT_maria_uses_666_blocks_l141_14192


namespace NUMINAMATH_GPT_find_number_l141_14101

def four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_digit_is_three (n : ℕ) : Prop :=
  n / 1000 = 3

def last_digit_is_five (n : ℕ) : Prop :=
  n % 10 = 5

theorem find_number :
  ∃ (x : ℕ), four_digit_number (x^2) ∧ first_digit_is_three (x^2) ∧ last_digit_is_five (x^2) ∧ x = 55 :=
sorry

end NUMINAMATH_GPT_find_number_l141_14101


namespace NUMINAMATH_GPT_line_x_intercept_l141_14176

theorem line_x_intercept {x1 y1 x2 y2 : ℝ} (h : (x1, y1) = (4, 6)) (h2 : (x2, y2) = (8, 2)) :
  ∃ x : ℝ, (y1 - y2) / (x1 - x2) * x + 6 - ((y1 - y2) / (x1 - x2)) * 4 = 0 ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_line_x_intercept_l141_14176


namespace NUMINAMATH_GPT_sample_size_l141_14150

theorem sample_size 
  (n_A n_B n_C : ℕ)
  (h1 : n_A = 15)
  (h2 : 3 * n_B = 4 * n_A)
  (h3 : 3 * n_C = 7 * n_A) :
  n_A + n_B + n_C = 70 :=
by
sorry

end NUMINAMATH_GPT_sample_size_l141_14150


namespace NUMINAMATH_GPT_distance_from_hotel_l141_14127

def total_distance := 600
def speed1 := 50
def time1 := 3
def speed2 := 80
def time2 := 4

theorem distance_from_hotel :
  total_distance - (speed1 * time1 + speed2 * time2) = 130 := 
by
  sorry

end NUMINAMATH_GPT_distance_from_hotel_l141_14127


namespace NUMINAMATH_GPT_find_m_l141_14179

theorem find_m (m : ℝ) (a b : ℝ × ℝ) (k : ℝ) (ha : a = (1, 1)) (hb : b = (m, 2)) 
  (h_parallel : 2 • a + b = k • a) : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_l141_14179


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l141_14156

-- (1) Prove 1 - 2(x - y) + (x - y)^2 = (1 - x + y)^2
theorem problem1 (x y : ℝ) : 1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 :=
sorry

-- (2) Prove 25(a - 1)^2 - 10(a - 1) + 1 = (5a - 6)^2
theorem problem2 (a : ℝ) : 25 * (a - 1)^2 - 10 * (a - 1) + 1 = (5 * a - 6)^2 :=
sorry

-- (3) Prove (y^2 - 4y)(y^2 - 4y + 8) + 16 = (y - 2)^4
theorem problem3 (y : ℝ) : (y^2 - 4 * y) * (y^2 - 4 * y + 8) + 16 = (y - 2)^4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l141_14156


namespace NUMINAMATH_GPT_not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l141_14194

theorem not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles : 
  ¬ ∃ (rectangles : ℕ × ℕ), rectangles.1 = 1 ∧ rectangles.2 = 7 ∧ rectangles.1 * 4 + rectangles.2 * 3 = 25 :=
by
  sorry

end NUMINAMATH_GPT_not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l141_14194


namespace NUMINAMATH_GPT_methane_combined_l141_14157

def balancedEquation (CH₄ O₂ CO₂ H₂O : ℕ) : Prop :=
  CH₄ = 1 ∧ O₂ = 2 ∧ CO₂ = 1 ∧ H₂O = 2

theorem methane_combined {moles_CH₄ moles_O₂ moles_H₂O : ℕ}
  (h₁ : moles_O₂ = 2)
  (h₂ : moles_H₂O = 2)
  (h_eq : balancedEquation moles_CH₄ moles_O₂ 1 moles_H₂O) : 
  moles_CH₄ = 1 :=
by
  sorry

end NUMINAMATH_GPT_methane_combined_l141_14157


namespace NUMINAMATH_GPT_smallest_integer_to_make_perfect_square_l141_14130

-- Define the number y as specified
def y : ℕ := 2^5 * 3^6 * (2^2)^7 * 5^8 * (2 * 3)^9 * 7^10 * (2^3)^11 * (3^2)^12

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The goal statement
theorem smallest_integer_to_make_perfect_square : 
  ∃ z : ℕ, z > 0 ∧ is_perfect_square (y * z) ∧ ∀ w : ℕ, w > 0 → is_perfect_square (y * w) → z ≤ w := by
  sorry

end NUMINAMATH_GPT_smallest_integer_to_make_perfect_square_l141_14130


namespace NUMINAMATH_GPT_odd_function_a_eq_minus_1_l141_14151

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x + a) / x

theorem odd_function_a_eq_minus_1 (a : ℝ) :
  (∀ x : ℝ, f (-x) a = -f x a) → a = -1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_odd_function_a_eq_minus_1_l141_14151


namespace NUMINAMATH_GPT_quotient_of_powers_l141_14166

theorem quotient_of_powers:
  (50 : ℕ) = 2 * 5^2 →
  (25 : ℕ) = 5^2 →
  (50^50 / 25^25 : ℕ) = 100^25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_quotient_of_powers_l141_14166


namespace NUMINAMATH_GPT_original_fraction_is_two_thirds_l141_14185

theorem original_fraction_is_two_thirds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ)/(b + 3) = 2 * (a : ℚ)/b → (a : ℚ)/b = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_original_fraction_is_two_thirds_l141_14185


namespace NUMINAMATH_GPT_binary_ternary_product_base_10_l141_14162

theorem binary_ternary_product_base_10 :
  let b2 := 2
  let t3 := 3
  let n1 := 1011 -- binary representation
  let n2 := 122 -- ternary representation
  let a1 := (1 * b2^3) + (0 * b2^2) + (1 * b2^1) + (1 * b2^0)
  let a2 := (1 * t3^2) + (2 * t3^1) + (2 * t3^0)
  a1 * a2 = 187 :=
by
  sorry

end NUMINAMATH_GPT_binary_ternary_product_base_10_l141_14162


namespace NUMINAMATH_GPT_inequal_f_i_sum_mn_ii_l141_14128

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 3 / 2 then -2 
  else if x > -5 / 2 then -x - 1 / 2 
  else 2

theorem inequal_f_i (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) :=
sorry

theorem sum_mn_ii (m n : ℝ) (h1 : f m + f n = 4) (h2 : m < n) : m + n < -5 :=
sorry

end NUMINAMATH_GPT_inequal_f_i_sum_mn_ii_l141_14128


namespace NUMINAMATH_GPT_value_of_a_is_3_l141_14169

def symmetric_about_x1 (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| + |x - a| = |2 - x + 1| + |2 - x - a|

theorem value_of_a_is_3 : symmetric_about_x1 3 :=
sorry

end NUMINAMATH_GPT_value_of_a_is_3_l141_14169


namespace NUMINAMATH_GPT_how_many_whole_boxes_did_nathan_eat_l141_14135

-- Define the conditions
def gumballs_per_package := 5
def total_gumballs := 20

-- The problem to prove
theorem how_many_whole_boxes_did_nathan_eat : total_gumballs / gumballs_per_package = 4 :=
by sorry

end NUMINAMATH_GPT_how_many_whole_boxes_did_nathan_eat_l141_14135


namespace NUMINAMATH_GPT_brandon_textbooks_weight_l141_14153

-- Define the weights of Jon's textbooks
def weight_jon_book1 := 2
def weight_jon_book2 := 8
def weight_jon_book3 := 5
def weight_jon_book4 := 9

-- Calculate the total weight of Jon's textbooks
def total_weight_jon := weight_jon_book1 + weight_jon_book2 + weight_jon_book3 + weight_jon_book4

-- Define the condition where Jon's textbooks weigh three times as much as Brandon's textbooks
def jon_to_brandon_ratio := 3

-- Define the weight of Brandon's textbooks
def weight_brandon := total_weight_jon / jon_to_brandon_ratio

-- The goal is to prove that the weight of Brandon's textbooks is 8 pounds.
theorem brandon_textbooks_weight : weight_brandon = 8 := by
  sorry

end NUMINAMATH_GPT_brandon_textbooks_weight_l141_14153


namespace NUMINAMATH_GPT_mod_21_solution_l141_14137

theorem mod_21_solution (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n < 21) (h₂ : 47635 ≡ n [MOD 21]) : n = 19 :=
by
  sorry

end NUMINAMATH_GPT_mod_21_solution_l141_14137


namespace NUMINAMATH_GPT_problem_l141_14104

def seq (a : ℕ → ℤ) : Prop :=
∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

theorem problem (a : ℕ → ℤ) (h₁ : a 1 = 2010) (h₂ : a 2 = 2011) (h₃ : seq a) : a 1000 = 2343 :=
sorry

end NUMINAMATH_GPT_problem_l141_14104


namespace NUMINAMATH_GPT_no_positive_reals_satisfy_equations_l141_14196

theorem no_positive_reals_satisfy_equations :
  ¬ ∃ (a b c d : ℝ), (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧
  (a / b + b / c + c / d + d / a = 6) ∧ (b / a + c / b + d / c + a / d = 32) :=
by sorry

end NUMINAMATH_GPT_no_positive_reals_satisfy_equations_l141_14196


namespace NUMINAMATH_GPT_janet_earnings_per_hour_l141_14197

theorem janet_earnings_per_hour :
  let P := 0.25
  let T := 10
  3600 / T * P = 90 :=
by
  let P := 0.25
  let T := 10
  sorry

end NUMINAMATH_GPT_janet_earnings_per_hour_l141_14197


namespace NUMINAMATH_GPT_value_of_x_minus_y_l141_14177

theorem value_of_x_minus_y (x y : ℚ) 
    (h₁ : 3 * x - 5 * y = 5) 
    (h₂ : x / (x + y) = 5 / 7) : x - y = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l141_14177


namespace NUMINAMATH_GPT_sequence_match_l141_14190

-- Define the sequence sum S_n
def S_n (n : ℕ) : ℕ := 2^(n + 1) - 1

-- Define the sequence a_n based on the problem statement
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 2^n

-- The theorem stating that sequence a_n satisfies the given sum condition S_n
theorem sequence_match (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end NUMINAMATH_GPT_sequence_match_l141_14190


namespace NUMINAMATH_GPT_inequality_proof_l141_14117

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  ¬ (1 / (1 + x + x * y) > 1 / 3 ∧ 
     y / (1 + y + y * z) > 1 / 3 ∧
     (x * z) / (1 + z + x * z) > 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l141_14117


namespace NUMINAMATH_GPT_sin_double_angle_identity_l141_14152

open Real

theorem sin_double_angle_identity {α : ℝ} (h1 : π / 2 < α ∧ α < π) 
    (h2 : sin (α + π / 6) = 1 / 3) :
  sin (2 * α + π / 3) = -4 * sqrt 2 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l141_14152


namespace NUMINAMATH_GPT_total_pieces_ten_row_triangle_l141_14131

-- Definitions based on the conditions
def rods (n : ℕ) : ℕ :=
  (n * (2 * 4 + (n - 1) * 5)) / 2

def connectors (n : ℕ) : ℕ :=
  ((n + 1) * (2 * 1 + n * 1)) / 2

def support_sticks (n : ℕ) : ℕ := 
  if n >= 3 then ((n - 2) * (2 * 2 + (n - 3) * 2)) / 2 else 0

-- The theorem stating the total number of pieces is 395 for a ten-row triangle
theorem total_pieces_ten_row_triangle : rods 10 + connectors 10 + support_sticks 10 = 395 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_ten_row_triangle_l141_14131


namespace NUMINAMATH_GPT_sum_of_squares_diagonals_of_rhombus_l141_14180

theorem sum_of_squares_diagonals_of_rhombus (d1 d2 : ℝ) (h : (d1 / 2)^2 + (d2 / 2)^2 = 4) : d1^2 + d2^2 = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_diagonals_of_rhombus_l141_14180


namespace NUMINAMATH_GPT_vasya_mushrooms_l141_14146

-- Lean definition of the problem based on the given conditions
theorem vasya_mushrooms :
  ∃ (N : ℕ), 
    N ≥ 100 ∧ N < 1000 ∧
    (∃ (a b c : ℕ), a ≠ 0 ∧ N = 100 * a + 10 * b + c ∧ a + b + c = 14) ∧
    N % 50 = 0 ∧ 
    N = 950 :=
by
  sorry

end NUMINAMATH_GPT_vasya_mushrooms_l141_14146


namespace NUMINAMATH_GPT_amount_paid_after_discount_l141_14125

def phone_initial_price : ℝ := 600
def discount_percentage : ℝ := 0.2

theorem amount_paid_after_discount : (phone_initial_price - discount_percentage * phone_initial_price) = 480 :=
by
  sorry

end NUMINAMATH_GPT_amount_paid_after_discount_l141_14125


namespace NUMINAMATH_GPT_exponential_inequality_l141_14148

theorem exponential_inequality (k l m : ℕ) : 2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 :=
by
  sorry

end NUMINAMATH_GPT_exponential_inequality_l141_14148


namespace NUMINAMATH_GPT_not_divisible_2310_l141_14133

theorem not_divisible_2310 (n : ℕ) (h : n < 2310) : ¬ (2310 ∣ n * (2310 - n)) :=
sorry

end NUMINAMATH_GPT_not_divisible_2310_l141_14133


namespace NUMINAMATH_GPT_percentage_of_seniors_is_90_l141_14120

-- Definitions of the given conditions
def total_students : ℕ := 120
def students_in_statistics : ℕ := total_students / 2
def seniors_in_statistics : ℕ := 54

-- Statement to prove
theorem percentage_of_seniors_is_90 : 
  ( seniors_in_statistics / students_in_statistics : ℚ ) * 100 = 90 := 
by
  sorry  -- Proof will be provided here.

end NUMINAMATH_GPT_percentage_of_seniors_is_90_l141_14120


namespace NUMINAMATH_GPT_no_intersection_with_x_axis_l141_14142

open Real

theorem no_intersection_with_x_axis (m : ℝ) :
  (∀ x : ℝ, 3 ^ (-(|x - 1|)) + m ≠ 0) ↔ (m ≥ 0 ∨ m < -1) :=
by
  sorry

end NUMINAMATH_GPT_no_intersection_with_x_axis_l141_14142


namespace NUMINAMATH_GPT_find_n_l141_14195

theorem find_n : ∀ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) → (n = -7 / 2) := by
  intro n h
  sorry

end NUMINAMATH_GPT_find_n_l141_14195


namespace NUMINAMATH_GPT_abs_a_lt_abs_b_sub_abs_c_l141_14118

theorem abs_a_lt_abs_b_sub_abs_c (a b c : ℝ) (h : |a + c| < b) : |a| < |b| - |c| :=
sorry

end NUMINAMATH_GPT_abs_a_lt_abs_b_sub_abs_c_l141_14118


namespace NUMINAMATH_GPT_min_value_frac_sum_l141_14103

theorem min_value_frac_sum (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  (∃ m, ∀ x y, m = 1 ∧ (
      (1 / (x + y)^2) + (1 / (x - y)^2) ≥ m)) :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l141_14103


namespace NUMINAMATH_GPT_g_value_at_5_l141_14140

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_5 (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x ^ 2) : g 5 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_g_value_at_5_l141_14140


namespace NUMINAMATH_GPT_find_x_value_l141_14110

theorem find_x_value (x : ℤ)
    (h1 : (5 + 9) / 2 = 7)
    (h2 : (5 + x) / 2 = 10)
    (h3 : (x + 9) / 2 = 12) : 
    x = 15 := 
sorry

end NUMINAMATH_GPT_find_x_value_l141_14110


namespace NUMINAMATH_GPT_sqrt_squared_l141_14136

theorem sqrt_squared (n : ℕ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

example : (Real.sqrt 987654) ^ 2 = 987654 := 
  sqrt_squared 987654 (by norm_num)

end NUMINAMATH_GPT_sqrt_squared_l141_14136


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l141_14106

theorem ratio_of_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l141_14106


namespace NUMINAMATH_GPT_original_denominator_l141_14154

theorem original_denominator (d : ℕ) (h : 11 = 3 * (d + 8)) : d = 25 :=
by
  sorry

end NUMINAMATH_GPT_original_denominator_l141_14154


namespace NUMINAMATH_GPT_petya_addition_mistake_l141_14186

theorem petya_addition_mistake:
  ∃ (x y c : ℕ), x + y = 12345 ∧ (10 * x + c) + y = 44444 ∧ x = 3566 ∧ y = 8779 ∧ c = 5 := by
  sorry

end NUMINAMATH_GPT_petya_addition_mistake_l141_14186


namespace NUMINAMATH_GPT_alloy_cut_weight_l141_14109

variable (a b x : ℝ)
variable (ha : 0 ≤ a ∧ a ≤ 1) -- assuming copper content is a fraction between 0 and 1
variable (hb : 0 ≤ b ∧ b ≤ 1)
variable (h : a ≠ b)
variable (hx : 0 < x ∧ x < 40) -- x is strictly between 0 and 40 (since 0 ≤ x ≤ 40)

theorem alloy_cut_weight (A B : ℝ) (hA : A = 40) (hB : B = 60) (h1 : (a * x + b * (A - x)) / 40 = (b * x + a * (B - x)) / 60) : x = 24 :=
by
  sorry

end NUMINAMATH_GPT_alloy_cut_weight_l141_14109


namespace NUMINAMATH_GPT_largest_possible_value_l141_14172

variable (a b : ℝ)

theorem largest_possible_value (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) :
  2 * a + b ≤ 5 :=
sorry

end NUMINAMATH_GPT_largest_possible_value_l141_14172


namespace NUMINAMATH_GPT_find_three_digit_number_divisible_by_5_l141_14158

theorem find_three_digit_number_divisible_by_5 {n x : ℕ} (hx1 : 100 ≤ x) (hx2 : x < 1000) (hx3 : x % 5 = 0) (hx4 : x = n^3 + n^2) : x = 150 ∨ x = 810 := 
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_divisible_by_5_l141_14158


namespace NUMINAMATH_GPT_find_angle_y_l141_14115

theorem find_angle_y (angle_ABC angle_ABD angle_ADB y : ℝ)
  (h1 : angle_ABC = 115)
  (h2 : angle_ABD = 180 - angle_ABC)
  (h3 : angle_ADB = 30)
  (h4 : angle_ABD + angle_ADB + y = 180) :
  y = 85 := 
sorry

end NUMINAMATH_GPT_find_angle_y_l141_14115


namespace NUMINAMATH_GPT_percentage_failed_in_english_l141_14126

theorem percentage_failed_in_english
  (H_perc : ℝ) (B_perc : ℝ) (Passed_in_English_alone : ℝ) (Total_candidates : ℝ)
  (H_perc_eq : H_perc = 36)
  (B_perc_eq : B_perc = 15)
  (Passed_in_English_alone_eq : Passed_in_English_alone = 630)
  (Total_candidates_eq : Total_candidates = 3000) :
  ∃ E_perc : ℝ, E_perc = 85 := by
  sorry

end NUMINAMATH_GPT_percentage_failed_in_english_l141_14126


namespace NUMINAMATH_GPT_harold_august_tips_fraction_l141_14189

noncomputable def tips_fraction : ℚ :=
  let A : ℚ := sorry -- average monthly tips for March to July and September
  let august_tips := 6 * A -- Tips for August
  let total_tips := 6 * A + 6 * A -- Total tips for all months worked
  august_tips / total_tips

theorem harold_august_tips_fraction :
  tips_fraction = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_harold_august_tips_fraction_l141_14189


namespace NUMINAMATH_GPT_sin_cos_pi_over_12_l141_14132

theorem sin_cos_pi_over_12 :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
sorry

end NUMINAMATH_GPT_sin_cos_pi_over_12_l141_14132


namespace NUMINAMATH_GPT_math_pages_l141_14170

def total_pages := 7
def reading_pages := 2

theorem math_pages : total_pages - reading_pages = 5 := by
  sorry

end NUMINAMATH_GPT_math_pages_l141_14170


namespace NUMINAMATH_GPT_focus_of_parabola_x_squared_eq_neg_4_y_l141_14183

theorem focus_of_parabola_x_squared_eq_neg_4_y:
  (∃ F : ℝ × ℝ, (F = (0, -1)) ∧ (∀ x y : ℝ, x^2 = -4 * y → F = (0, y + 1))) :=
sorry

end NUMINAMATH_GPT_focus_of_parabola_x_squared_eq_neg_4_y_l141_14183


namespace NUMINAMATH_GPT_remainder_divisibility_l141_14108

theorem remainder_divisibility (n : ℕ) (d : ℕ) (r : ℕ) : 
  let n := 1234567
  let d := 256
  let r := n % d
  r = 933 ∧ ¬ (r % 7 = 0) := by
  sorry

end NUMINAMATH_GPT_remainder_divisibility_l141_14108


namespace NUMINAMATH_GPT_total_balloons_l141_14145

theorem total_balloons (fred_balloons : ℕ) (sam_balloons : ℕ) (mary_balloons : ℕ) :
  fred_balloons = 5 → sam_balloons = 6 → mary_balloons = 7 → fred_balloons + sam_balloons + mary_balloons = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_balloons_l141_14145


namespace NUMINAMATH_GPT_flour_quantity_l141_14123

-- Define the recipe ratio of eggs to flour
def recipe_ratio : ℚ := 3 / 2

-- Define the number of eggs needed
def eggs_needed := 9

-- Prove that the number of cups of flour needed is 6
theorem flour_quantity (r : ℚ) (n : ℕ) (F : ℕ) 
  (hr : r = 3 / 2) (hn : n = 9) : F = 6 :=
by
  sorry

end NUMINAMATH_GPT_flour_quantity_l141_14123


namespace NUMINAMATH_GPT_product_of_square_and_neighbor_is_divisible_by_12_l141_14182

theorem product_of_square_and_neighbor_is_divisible_by_12 (n : ℤ) : 12 ∣ (n^2 * (n - 1) * (n + 1)) :=
sorry

end NUMINAMATH_GPT_product_of_square_and_neighbor_is_divisible_by_12_l141_14182


namespace NUMINAMATH_GPT_sin_arith_seq_l141_14100

theorem sin_arith_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) :
  Real.sin (a 2 + a 8) = - (Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_sin_arith_seq_l141_14100


namespace NUMINAMATH_GPT_gumballs_per_pair_of_earrings_l141_14165

theorem gumballs_per_pair_of_earrings : 
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  (total_gumballs / total_earrings) = 9 :=
by
  -- Definitions
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  -- Theorem statement
  sorry

end NUMINAMATH_GPT_gumballs_per_pair_of_earrings_l141_14165


namespace NUMINAMATH_GPT_order_of_abc_l141_14134

theorem order_of_abc (a b c : ℝ) (h1 : a = 16 ^ (1 / 3))
                                 (h2 : b = 2 ^ (4 / 5))
                                 (h3 : c = 5 ^ (2 / 3)) :
  c > a ∧ a > b :=
by {
  sorry
}

end NUMINAMATH_GPT_order_of_abc_l141_14134


namespace NUMINAMATH_GPT_find_k_plus_a_l141_14107

theorem find_k_plus_a (k a : ℤ) (h1 : k > a) (h2 : a > 0) 
(h3 : 2 * (Int.natAbs (a - k)) * (Int.natAbs (a + k)) = 32) : k + a = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_k_plus_a_l141_14107


namespace NUMINAMATH_GPT_no_distinct_positive_integers_2007_l141_14163

theorem no_distinct_positive_integers_2007 (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) : 
  ¬ (x^2007 + y! = y^2007 + x!) :=
by
  sorry

end NUMINAMATH_GPT_no_distinct_positive_integers_2007_l141_14163
