import Mathlib

namespace unknown_angles_are_80_l1573_157371

theorem unknown_angles_are_80 (y : ℝ) (h1 : y + y + 200 = 360) : y = 80 :=
by
  sorry

end unknown_angles_are_80_l1573_157371


namespace cos_double_angle_l1573_157343

theorem cos_double_angle (x : ℝ) (h : Real.sin (x + Real.pi / 2) = 1 / 3) : Real.cos (2 * x) = -7 / 9 :=
sorry

end cos_double_angle_l1573_157343


namespace color_change_probability_l1573_157347

-- Definitions based directly on conditions in a)
def light_cycle_duration := 93
def change_intervals_duration := 15
def expected_probability := 5 / 31

-- The Lean 4 statement for the proof problem
theorem color_change_probability :
  (change_intervals_duration / light_cycle_duration) = expected_probability :=
by
  sorry

end color_change_probability_l1573_157347


namespace area_relationship_l1573_157399

theorem area_relationship (a b c : ℝ) (h : a^2 + b^2 = c^2) : (a + b)^2 = a^2 + 2*a*b + b^2 := 
by sorry

end area_relationship_l1573_157399


namespace cube_volume_from_surface_area_l1573_157392

theorem cube_volume_from_surface_area (s : ℝ) (h : 6 * s^2 = 54) : s^3 = 27 :=
sorry

end cube_volume_from_surface_area_l1573_157392


namespace geometric_seq_problem_l1573_157366

theorem geometric_seq_problem
  (a : Nat → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_cond : a 1 * a 99 = 16) :
  a 20 * a 80 = 16 := 
sorry

end geometric_seq_problem_l1573_157366


namespace johns_average_speed_last_hour_l1573_157395

theorem johns_average_speed_last_hour
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (distance_last_hour : ℕ)
  (average_speed_last_hour : ℕ)
  (H1 : total_distance = 120)
  (H2 : total_time = 3)
  (H3 : speed_first_hour = 40)
  (H4 : speed_second_hour = 50)
  (H5 : distance_last_hour = total_distance - (speed_first_hour + speed_second_hour))
  (H6 : average_speed_last_hour = distance_last_hour / 1)
  : average_speed_last_hour = 30 := 
by
  -- Placeholder for the proof
  sorry

end johns_average_speed_last_hour_l1573_157395


namespace rationalize_denominator_sum_l1573_157396

theorem rationalize_denominator_sum :
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  A + B + C + D + E + F = 210 :=
by
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  show 3 + -9 + -9 + 9 + 165 + 51 = 210
  sorry

end rationalize_denominator_sum_l1573_157396


namespace raspberry_pies_l1573_157328

theorem raspberry_pies (total_pies : ℕ) (r_peach : ℕ) (r_strawberry : ℕ) (r_raspberry : ℕ) (r_sum : ℕ) :
    total_pies = 36 → r_peach = 2 → r_strawberry = 5 → r_raspberry = 3 → r_sum = (r_peach + r_strawberry + r_raspberry) →
    (total_pies : ℝ) / (r_sum : ℝ) * (r_raspberry : ℝ) = 10.8 :=
by
    -- This theorem is intended to state the problem.
    sorry

end raspberry_pies_l1573_157328


namespace solution_set_abs_ineq_l1573_157354

theorem solution_set_abs_ineq (x : ℝ) : abs (2 - x) ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end solution_set_abs_ineq_l1573_157354


namespace sample_size_l1573_157386

variable (x n : ℕ)

-- Conditions as definitions
def staff_ratio : Prop := 15 * x + 3 * x + 2 * x = 20 * x
def sales_staff : Prop := 30 / n = 15 / 20

-- Main statement to prove
theorem sample_size (h1: staff_ratio x) (h2: sales_staff n) : n = 40 := by
  sorry

end sample_size_l1573_157386


namespace find_G_8_l1573_157360

noncomputable def G : Polynomial ℝ := sorry 

variable (x : ℝ)

theorem find_G_8 :
  G.eval 4 = 8 ∧ 
  (∀ x, (G.eval (2*x)) / (G.eval (x+2)) = 4 - (16 * x) / (x^2 + 2 * x + 2)) →
  G.eval 8 = 40 := 
sorry

end find_G_8_l1573_157360


namespace max_profit_achieved_at_180_l1573_157323

-- Definitions:
def cost (x : ℝ) : ℝ := 0.1 * x^2 - 11 * x + 3000  -- Condition 1
def selling_price_per_unit : ℝ := 25  -- Condition 2

-- Statement to prove that the maximum profit is achieved at x = 180
theorem max_profit_achieved_at_180 :
  ∃ (S : ℝ), ∀ (x : ℝ),
    S = -0.1 * (x - 180)^2 + 240 → S = 25 * 180 - cost 180 :=
by
  sorry

end max_profit_achieved_at_180_l1573_157323


namespace Jaymee_is_22_l1573_157307

-- Define Shara's age
def Shara_age : ℕ := 10

-- Define Jaymee's age according to the problem conditions
def Jaymee_age : ℕ := 2 + 2 * Shara_age

-- The proof statement to show that Jaymee's age is 22
theorem Jaymee_is_22 : Jaymee_age = 22 := by 
  -- The proof is omitted according to the instructions.
  sorry

end Jaymee_is_22_l1573_157307


namespace workshop_output_comparison_l1573_157346

theorem workshop_output_comparison (a x : ℝ)
  (h1 : ∀n:ℕ, n ≥ 0 → (1 + n * a) = (1 + x)^n) :
  (1 + 3 * a) > (1 + x)^3 := sorry

end workshop_output_comparison_l1573_157346


namespace fishing_tomorrow_l1573_157369

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end fishing_tomorrow_l1573_157369


namespace problem_statement_l1573_157378

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) : b < 0 ∧ |b| > |a| :=
by
  sorry

end problem_statement_l1573_157378


namespace smallest_r_l1573_157320

variables (p q r s : ℤ)

-- Define the conditions
def condition1 : Prop := p + 3 = q - 1
def condition2 : Prop := p + 3 = r + 5
def condition3 : Prop := p + 3 = s - 2

-- Prove that r is the smallest
theorem smallest_r (h1 : condition1 p q) (h2 : condition2 p r) (h3 : condition3 p s) : r < p ∧ r < q ∧ r < s :=
sorry

end smallest_r_l1573_157320


namespace vertex_of_parabola_l1573_157315

-- Define the statement of the problem
theorem vertex_of_parabola :
  ∀ (a h k : ℝ), (∀ x : ℝ, 3 * (x - 5) ^ 2 + 4 = a * (x - h) ^ 2 + k) → (h, k) = (5, 4) :=
by
  sorry

end vertex_of_parabola_l1573_157315


namespace infinite_set_divisor_l1573_157339

open Set

noncomputable def exists_divisor (A : Set ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ A → d ∣ a

theorem infinite_set_divisor (A : Set ℕ) (hA1 : ∀ (b : Finset ℕ), (↑b ⊆ A) → ∃ (d : ℕ), d > 1 ∧ ∀ (a : ℕ), a ∈ b → d ∣ a) :
  exists_divisor A :=
sorry

end infinite_set_divisor_l1573_157339


namespace find_remaining_score_l1573_157398

-- Define the problem conditions
def student_scores : List ℕ := [70, 80, 90]
def average_score : ℕ := 70

-- Define the remaining score to prove it equals 40
def remaining_score : ℕ := 40

-- The theorem statement
theorem find_remaining_score (scores : List ℕ) (avg : ℕ) (r : ℕ) 
    (h_scores : scores = [70, 80, 90]) 
    (h_avg : avg = 70) 
    (h_length : scores.length = 3) 
    (h_avg_eq : (scores.sum + r) / (scores.length + 1) = avg) 
    : r = 40 := 
by
  sorry

end find_remaining_score_l1573_157398


namespace product_of_squares_of_consecutive_even_integers_l1573_157342

theorem product_of_squares_of_consecutive_even_integers :
  ∃ (a : ℤ), (a - 2) * a * (a + 2) = 36 * a ∧ (a > 0) ∧ (a % 2 = 0) ∧
  ((a - 2)^2 * a^2 * (a + 2)^2) = 36864 :=
by
  sorry

end product_of_squares_of_consecutive_even_integers_l1573_157342


namespace fewer_spoons_l1573_157313

/--
Stephanie initially planned to buy 15 pieces of each type of silverware.
There are 4 types of silverware.
This totals to 60 pieces initially planned to be bought.
She only bought 44 pieces in total.
Show that she decided to purchase 4 fewer spoons.
-/
theorem fewer_spoons
  (initial_total : ℕ := 60)
  (final_total : ℕ := 44)
  (types : ℕ := 4)
  (pieces_per_type : ℕ := 15) :
  (initial_total - final_total) / types = 4 := 
by
  -- since initial_total = 60, final_total = 44, and types = 4
  -- we need to prove (60 - 44) / 4 = 4
  sorry

end fewer_spoons_l1573_157313


namespace lower_bound_third_inequality_l1573_157340

theorem lower_bound_third_inequality (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 8 > x ∧ x > 0)
  (h4 : x + 1 < 9) :
  x = 7 → ∃ l < 7, ∀ y, l < y ∧ y < 9 → y = x := 
sorry

end lower_bound_third_inequality_l1573_157340


namespace find_x_l1573_157327

theorem find_x (x : ℝ) (y : ℝ) : (∀ y, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 := by
  intros h
  -- At this point, you would include the necessary proof steps, but for now we skip it.
  sorry

end find_x_l1573_157327


namespace triangle_medians_and_area_l1573_157382

/-- Given a triangle with side lengths 13, 14, and 15,
    prove that the sum of the squares of the lengths of the medians is 385
    and the area of the triangle is 84. -/
theorem triangle_medians_and_area :
  let a := 13
  let b := 14
  let c := 15
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let m_a := Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
  let m_b := Real.sqrt (2 * c^2 + 2 * a^2 - b^2) / 2
  let m_c := Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2
  m_a^2 + m_b^2 + m_c^2 = 385 ∧ area = 84 := sorry

end triangle_medians_and_area_l1573_157382


namespace sum_series_eq_1_div_300_l1573_157335

noncomputable def sum_series : ℝ :=
  ∑' n, (6 * (n:ℝ) + 1) / ((6 * (n:ℝ) - 1) ^ 2 * (6 * (n:ℝ) + 5) ^ 2)

theorem sum_series_eq_1_div_300 : sum_series = 1 / 300 :=
  sorry

end sum_series_eq_1_div_300_l1573_157335


namespace weight_of_rod_l1573_157303

theorem weight_of_rod (w₆ : ℝ) (h₁ : w₆ = 6.1) : 
  w₆ / 6 * 12 = 12.2 := by
  sorry

end weight_of_rod_l1573_157303


namespace percent_increase_is_equivalent_l1573_157316

variable {P : ℝ}

theorem percent_increase_is_equivalent 
  (h1 : 1.0 + 15.0 / 100.0 = 1.15)
  (h2 : 1.15 * (1.0 + 25.0 / 100.0) = 1.4375)
  (h3 : 1.4375 * (1.0 + 10.0 / 100.0) = 1.58125) :
  (1.58125 - 1) * 100 = 58.125 :=
by
  sorry

end percent_increase_is_equivalent_l1573_157316


namespace conic_section_is_ellipse_l1573_157374

theorem conic_section_is_ellipse :
  ∀ x y : ℝ, 4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0 →
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧ (a * (x - h)^2 + b * (y - k)^2 = 1) :=
by
  sorry

end conic_section_is_ellipse_l1573_157374


namespace sum_of_slopes_range_l1573_157302

theorem sum_of_slopes_range (p b : ℝ) (hpb : 2 * p > b) (hp : p > 0) 
  (K1 K2 : ℝ) (A B : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1) (hB : B.2^2 = 2 * p * B.1)
  (hl1 : A.2 = A.1 + b) (hl2 : B.2 = B.1 + b) 
  (hA_pos : A.2 > 0) (hB_pos : B.2 > 0) :
  4 < K1 + K2 :=
sorry

end sum_of_slopes_range_l1573_157302


namespace no_value_of_n_l1573_157330

noncomputable def t1 (n : ℕ) : ℚ :=
3 * n * (n + 2)

noncomputable def t2 (n : ℕ) : ℚ :=
(3 * n^2 + 19 * n) / 2

theorem no_value_of_n (n : ℕ) (h : n > 0) : t1 n ≠ t2 n :=
by {
  sorry
}

end no_value_of_n_l1573_157330


namespace max_reciprocal_sum_eq_2_l1573_157375

theorem max_reciprocal_sum_eq_2 (r1 r2 t q : ℝ) (h1 : r1 + r2 = t) (h2 : r1 * r2 = q)
  (h3 : ∀ n : ℕ, n > 0 → r1 + r2 = r1^n + r2^n) :
  1 / r1^2010 + 1 / r2^2010 = 2 :=
by
  sorry

end max_reciprocal_sum_eq_2_l1573_157375


namespace problem_statement_l1573_157311

/-- Define the sequence of numbers spoken by Jo and Blair. -/
def next_number (n : ℕ) : ℕ :=
if n % 2 = 1 then (n + 1) / 2 else n / 2

/-- Helper function to compute the 21st number said. -/
noncomputable def twenty_first_number : ℕ :=
(21 + 1) / 2

/-- Statement of the problem in Lean 4. -/
theorem problem_statement : twenty_first_number = 11 := by
  sorry

end problem_statement_l1573_157311


namespace supplementary_angle_measure_l1573_157394

theorem supplementary_angle_measure (a b : ℝ) 
  (h1 : a + b = 180) 
  (h2 : a / 5 = b / 4) : b = 80 :=
by
  sorry

end supplementary_angle_measure_l1573_157394


namespace tan_identity_l1573_157384

theorem tan_identity (α β γ : ℝ) (h : α + β + γ = 45 * π / 180) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 :=
by
  sorry

end tan_identity_l1573_157384


namespace unique_positive_integers_exists_l1573_157380

theorem unique_positive_integers_exists (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) : 
  ∃! m n : ℕ, m^2 = n * (n + p) ∧ m = (p^2 - 1) / 2 ∧ n = (p - 1)^2 / 4 := by
  sorry

end unique_positive_integers_exists_l1573_157380


namespace age_sum_proof_l1573_157336

theorem age_sum_proof (a b c : ℕ) (h1 : a - (b + c) = 16) (h2 : a^2 - (b + c)^2 = 1632) : a + b + c = 102 :=
by
  sorry

end age_sum_proof_l1573_157336


namespace Mary_books_check_out_l1573_157359

theorem Mary_books_check_out
  (initial_books : ℕ)
  (returned_unhelpful_books : ℕ)
  (returned_later_books : ℕ)
  (checked_out_later_books : ℕ)
  (total_books_now : ℕ)
  (h1 : initial_books = 5)
  (h2 : returned_unhelpful_books = 3)
  (h3 : returned_later_books = 2)
  (h4 : checked_out_later_books = 7)
  (h5 : total_books_now = 12) :
  ∃ (x : ℕ), (initial_books - returned_unhelpful_books + x - returned_later_books + checked_out_later_books = total_books_now) ∧ x = 5 :=
by {
  sorry
}

end Mary_books_check_out_l1573_157359


namespace negate_exponential_inequality_l1573_157344

theorem negate_exponential_inequality :
  ¬ (∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x :=
by
  sorry

end negate_exponential_inequality_l1573_157344


namespace inequality_solution_l1573_157393

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, 2 * x + 7 > 3 * x + 2 ∧ 2 * x - 2 < 2 * m → x < 5) → m ≥ 4 :=
by
  sorry

end inequality_solution_l1573_157393


namespace expression_is_product_l1573_157338

def not_sum (a x : Int) : Prop :=
  ¬(a + x = -7 * x)

def not_difference (a x : Int) : Prop :=
  ¬(a - x = -7 * x)

def not_quotient (a x : Int) : Prop :=
  ¬(a / x = -7 * x)

theorem expression_is_product (x : Int) : 
  not_sum (-7) x ∧ not_difference (-7) x ∧ not_quotient (-7) x → (-7 * x = -7 * x) :=
by sorry

end expression_is_product_l1573_157338


namespace no_eight_consecutive_sums_in_circle_l1573_157325

theorem no_eight_consecutive_sums_in_circle :
  ¬ ∃ (arrangement : Fin 8 → ℕ) (sums : Fin 8 → ℤ),
      (∀ i, 1 ≤ arrangement i ∧ arrangement i ≤ 8) ∧
      (∀ i, sums i = arrangement i + arrangement (⟨(i + 1) % 8, sorry⟩)) ∧
      (∃ (n : ℤ), 
        (sums 0 = n - 3) ∧ 
        (sums 1 = n - 2) ∧ 
        (sums 2 = n - 1) ∧ 
        (sums 3 = n) ∧ 
        (sums 4 = n + 1) ∧ 
        (sums 5 = n + 2) ∧ 
        (sums 6 = n + 3) ∧ 
        (sums 7 = n + 4)) := 
sorry

end no_eight_consecutive_sums_in_circle_l1573_157325


namespace number_of_students_l1573_157312

theorem number_of_students (n : ℕ)
  (h1 : ∃ n, (175 * n) / n = 175)
  (h2 : 175 * n - 40 = 173 * n) :
  n = 20 :=
sorry

end number_of_students_l1573_157312


namespace number_of_intersections_l1573_157367

theorem number_of_intersections : 
  (∃ p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ 9 * p.1^2 + p.2^2 = 1) 
  ∧ (∃! p₁ p₂ : ℝ × ℝ, p₁ ≠ p₂ ∧ p₁.1^2 + 9 * p₁.2^2 = 9 ∧ 9 * p₁.1^2 + p₁.2^2 = 1 ∧
    p₂.1^2 + 9 * p₂.2^2 = 9 ∧ 9 * p₂.1^2 + p₂.2^2 = 1) :=
by
  -- The proof will be here
  sorry

end number_of_intersections_l1573_157367


namespace not_exists_cube_in_sequence_l1573_157350

-- Lean statement of the proof problem
theorem not_exists_cube_in_sequence : ∀ n : ℕ, ¬ ∃ k : ℤ, 2 ^ (2 ^ n) + 1 = k ^ 3 := 
by 
    intro n
    intro ⟨k, h⟩
    sorry

end not_exists_cube_in_sequence_l1573_157350


namespace john_total_water_usage_l1573_157358

-- Define the basic conditions
def total_days_in_weeks (weeks : ℕ) : ℕ := weeks * 7
def showers_every_other_day (days : ℕ) : ℕ := days / 2
def total_minutes_shower (showers : ℕ) (minutes_per_shower : ℕ) : ℕ := showers * minutes_per_shower
def total_water_usage (total_minutes : ℕ) (water_per_minute : ℕ) : ℕ := total_minutes * water_per_minute

-- Main statement
theorem john_total_water_usage :
  total_water_usage (total_minutes_shower (showers_every_other_day (total_days_in_weeks 4)) 10) 2 = 280 :=
by
  sorry

end john_total_water_usage_l1573_157358


namespace problem_l1573_157362

noncomputable def f (ω φ : ℝ) (x : ℝ) := 4 * Real.sin (ω * x + φ)

theorem problem (ω : ℝ) (φ : ℝ) (x1 x2 α : ℝ) (hω : 0 < ω) (hφ : |φ| < Real.pi / 2)
  (h0 : f ω φ 0 = 2 * Real.sqrt 3)
  (hx1 : f ω φ x1 = 0) (hx2 : f ω φ x2 = 0) (hx1x2 : |x1 - x2| = Real.pi / 2)
  (hα : α ∈ Set.Ioo (Real.pi / 12) (Real.pi / 2)) :
  f 2 (Real.pi / 3) α = 12 / 5 ∧ Real.sin (2 * α) = (3 + 4 * Real.sqrt 3) / 10 :=
sorry

end problem_l1573_157362


namespace solve_equation_l1573_157324

theorem solve_equation (x : ℝ) :
  (1 / (x ^ 2 + 14 * x - 10)) + (1 / (x ^ 2 + 3 * x - 10)) + (1 / (x ^ 2 - 16 * x - 10)) = 0
  ↔ (x = 5 ∨ x = -2 ∨ x = 2 ∨ x = -5) :=
sorry

end solve_equation_l1573_157324


namespace john_ate_half_package_l1573_157379

def fraction_of_package_john_ate (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving : ℚ)

theorem john_ate_half_package (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) 
    (h_servings : servings = 3) (h_calories_per_serving : calories_per_serving = 120) (h_calories_consumed : calories_consumed = 180) :
    fraction_of_package_john_ate servings calories_per_serving calories_consumed = 1 / 2 :=
by
  -- Replace the actual proof with sorry to ensure the statement compiles.
  sorry

end john_ate_half_package_l1573_157379


namespace term_10_of_sequence_l1573_157365

theorem term_10_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * n + 1)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 10 = 39 :=
by
  intros hS ha
  sorry

end term_10_of_sequence_l1573_157365


namespace find_g_l1573_157355

open Function

def linear_system (a b c d e f g : ℚ) :=
  a + b + c + d + e = 1 ∧
  b + c + d + e + f = 2 ∧
  c + d + e + f + g = 3 ∧
  d + e + f + g + a = 4 ∧
  e + f + g + a + b = 5 ∧
  f + g + a + b + c = 6 ∧
  g + a + b + c + d = 7

theorem find_g (a b c d e f g : ℚ) (h : linear_system a b c d e f g) : 
  g = 13 / 3 :=
sorry

end find_g_l1573_157355


namespace smallest_sum_ending_2050306_l1573_157318

/--
Given nine consecutive natural numbers starting at n,
prove that the smallest sum of these nine numbers ending in 2050306 is 22050306.
-/
theorem smallest_sum_ending_2050306 
  (n : ℕ) 
  (hn : ∃ m : ℕ, 9 * m = (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) ∧ 
                 (9 * m) % 10^7 = 2050306) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) = 22050306 := 
sorry

end smallest_sum_ending_2050306_l1573_157318


namespace cristina_nicky_head_start_l1573_157388

theorem cristina_nicky_head_start (s_c s_n : ℕ) (t d : ℕ) 
  (h1 : s_c = 5) 
  (h2 : s_n = 3) 
  (h3 : t = 30)
  (h4 : d = s_n * t):
  d = 90 := 
by
  sorry

end cristina_nicky_head_start_l1573_157388


namespace total_payment_l1573_157300

-- Define the basic conditions
def hours_first_day : ℕ := 10
def hours_second_day : ℕ := 8
def hours_third_day : ℕ := 15
def hourly_wage : ℕ := 10
def number_of_workers : ℕ := 2

-- Define the proof problem
theorem total_payment : 
  (hours_first_day + hours_second_day + hours_third_day) * hourly_wage * number_of_workers = 660 := 
by
  sorry

end total_payment_l1573_157300


namespace valid_license_plates_count_l1573_157314

def validLicensePlates : Nat :=
  26 * 26 * 26 * 10 * 9 * 8

theorem valid_license_plates_count :
  validLicensePlates = 15818400 :=
by
  sorry

end valid_license_plates_count_l1573_157314


namespace interest_rate_l1573_157356

-- Definitions based on given conditions
def SumLent : ℝ := 1500
def InterestTime : ℝ := 4
def InterestAmount : ℝ := SumLent - 1260

-- Main theorem to prove the interest rate r is 4%
theorem interest_rate (r : ℝ) : (InterestAmount = SumLent * r / 100 * InterestTime) → r = 4 :=
by
  sorry

end interest_rate_l1573_157356


namespace two_zeros_of_cubic_polynomial_l1573_157331

theorem two_zeros_of_cubic_polynomial (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -x1^3 + 3*x1 + m = 0 ∧ -x2^3 + 3*x2 + m = 0) →
  (m = -2 ∨ m = 2) :=
by
  sorry

end two_zeros_of_cubic_polynomial_l1573_157331


namespace Shekar_average_marks_l1573_157326

theorem Shekar_average_marks 
  (math_marks : ℕ := 76)
  (science_marks : ℕ := 65)
  (social_studies_marks : ℕ := 82)
  (english_marks : ℕ := 67)
  (biology_marks : ℕ := 95)
  (num_subjects : ℕ := 5) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = 77 := 
sorry

end Shekar_average_marks_l1573_157326


namespace berries_per_bird_per_day_l1573_157391

theorem berries_per_bird_per_day (birds : ℕ) (total_berries : ℕ) (days : ℕ) (berries_per_bird_per_day : ℕ) 
  (h_birds : birds = 5)
  (h_total_berries : total_berries = 140)
  (h_days : days = 4) :
  berries_per_bird_per_day = 7 :=
  sorry

end berries_per_bird_per_day_l1573_157391


namespace percentage_material_B_new_mixture_l1573_157373

theorem percentage_material_B_new_mixture :
  let mixtureA := 8 -- kg of Mixture A
  let addOil := 2 -- kg of additional oil
  let addMixA := 6 -- kg of additional Mixture A
  let oil_percent := 0.20 -- 20% oil in Mixture A
  let materialB_percent := 0.80 -- 80% material B in Mixture A

  -- Initial amounts in 8 kg of Mixture A
  let initial_oil := oil_percent * mixtureA
  let initial_materialB := materialB_percent * mixtureA

  -- New mixture after adding 2 kg oil
  let new_oil := initial_oil + addOil
  let new_materialB := initial_materialB

  -- Adding 6 kg of Mixture A
  let added_oil := oil_percent * addMixA
  let added_materialB := materialB_percent * addMixA

  -- Total amounts in the new mixture
  let total_oil := new_oil + added_oil
  let total_materialB := new_materialB + added_materialB
  let total_weight := mixtureA + addOil + addMixA

  -- Percent calculation
  let percent_materialB := (total_materialB / total_weight) * 100

  percent_materialB = 70 := sorry

end percentage_material_B_new_mixture_l1573_157373


namespace complete_square_eq_l1573_157309

theorem complete_square_eq (b c : ℤ) (h : ∃ b c : ℤ, (∀ x : ℝ, (x - 5)^2 = b * x + c) ∧ b + c = 5) :
  b + c = 5 :=
sorry

end complete_square_eq_l1573_157309


namespace howard_rewards_l1573_157387

theorem howard_rewards (initial_bowls : ℕ) (customers : ℕ) (customers_bought_20 : ℕ) 
                       (bowls_remaining : ℕ) (rewards_per_bowl : ℕ) :
  initial_bowls = 70 → 
  customers = 20 → 
  customers_bought_20 = 10 → 
  bowls_remaining = 30 → 
  rewards_per_bowl = 2 →
  ∀ (bowls_bought_per_customer : ℕ), bowls_bought_per_customer = 20 → 
  2 * (200 / 20) = 10 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end howard_rewards_l1573_157387


namespace age_of_b_l1573_157368

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_of_b_l1573_157368


namespace problem_statement_l1573_157317

variable {f : ℝ → ℝ}

-- Assume the conditions provided in the problem statement.
def continuous_on_ℝ (f : ℝ → ℝ) : Prop := Continuous f
def condition_x_f_prime (f : ℝ → ℝ) (h : ℝ → ℝ) : Prop := ∀ x : ℝ, x * h x < 0

-- The main theorem statement based on the conditions and the correct answer.
theorem problem_statement (hf : continuous_on_ℝ f) (hf' : ∀ x : ℝ, x * (deriv f x) < 0) :
  f (-1) + f 1 < 2 * f 0 :=
sorry

end problem_statement_l1573_157317


namespace least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l1573_157364

def least_subtrahend (n m : ℕ) (k : ℕ) : Prop :=
  (n - k) % m = 0 ∧ ∀ k' : ℕ, k' < k → (n - k') % m ≠ 0

theorem least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22 :
  least_subtrahend 102932847 25 22 :=
sorry

end least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l1573_157364


namespace sequence_u5_value_l1573_157304

theorem sequence_u5_value (u : ℕ → ℝ) 
  (h_rec : ∀ n, u (n + 2) = 2 * u (n + 1) + u n)
  (h_u3 : u 3 = 9) 
  (h_u6 : u 6 = 128) : 
  u 5 = 53 := 
sorry

end sequence_u5_value_l1573_157304


namespace cost_per_chicken_l1573_157322

-- Definitions for conditions
def totalBirds : ℕ := 15
def ducks : ℕ := totalBirds / 3
def chickens : ℕ := totalBirds - ducks
def feed_cost : ℕ := 20

-- Theorem stating the cost per chicken
theorem cost_per_chicken : (feed_cost / chickens) = 2 := by
  sorry

end cost_per_chicken_l1573_157322


namespace peanut_butter_candy_count_l1573_157301

-- Definitions derived from the conditions
def grape_candy (banana_candy : ℕ) := banana_candy + 5
def peanut_butter_candy (grape_candy : ℕ) := 4 * grape_candy

-- Given condition for the banana jar
def banana_candy := 43

-- The main theorem statement
theorem peanut_butter_candy_count : peanut_butter_candy (grape_candy banana_candy) = 192 :=
by
  sorry

end peanut_butter_candy_count_l1573_157301


namespace a_left_after_working_days_l1573_157329

variable (x : ℕ)  -- x represents the days A worked 

noncomputable def A_work_rate := (1 : ℚ) / 21
noncomputable def B_work_rate := (1 : ℚ) / 28
noncomputable def B_remaining_work := (3 : ℚ) / 4
noncomputable def combined_work_rate := A_work_rate + B_work_rate

theorem a_left_after_working_days 
  (h : combined_work_rate * x + B_remaining_work = 1) : x = 3 :=
by 
  sorry

end a_left_after_working_days_l1573_157329


namespace average_speed_is_one_l1573_157370

-- Definition of distance and time
def distance : ℕ := 1800
def time_in_minutes : ℕ := 30
def time_in_seconds : ℕ := time_in_minutes * 60

-- Definition of average speed as distance divided by time
def average_speed (distance : ℕ) (time : ℕ) : ℚ :=
  distance / time

-- Theorem: Given the distance and time, the average speed is 1 meter per second
theorem average_speed_is_one : average_speed distance time_in_seconds = 1 :=
  by
    sorry

end average_speed_is_one_l1573_157370


namespace gcd_of_16_and_12_l1573_157390

theorem gcd_of_16_and_12 : Nat.gcd 16 12 = 4 := by
  sorry

end gcd_of_16_and_12_l1573_157390


namespace consecutive_integers_solution_l1573_157305

theorem consecutive_integers_solution :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) + 91 = n^2 + (n + 1)^2 ∧ n + 1 = 10 :=
by
  sorry

end consecutive_integers_solution_l1573_157305


namespace part_i_l1573_157348

theorem part_i (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  a + b > c ∧ a + c > b ∧ b + c > a := sorry

end part_i_l1573_157348


namespace sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l1573_157351

-- Part (a):
def sibling_of_frac (x : ℚ) : Prop :=
  x = 5/7

theorem sibling_of_5_over_7 : ∃ (y : ℚ), sibling_of_frac (y / (y + 1)) ∧ y + 1 = 7/2 :=
  sorry

-- Part (b):
def child (x y : ℚ) : Prop :=
  y = x + 1 ∨ y = x / (x + 1)

theorem child_unique_parent (x y z : ℚ) (hx : 0 < x) (hz : 0 < z) (hyx : child x y) (hyz : child z y) : x = z :=
  sorry

-- Part (c):
def descendent (x y : ℚ) : Prop :=
  ∃ n : ℕ, y = 1 / (x + n)

theorem one_over_2008_descendent_of_one : descendent 1 (1 / 2008) :=
  sorry

end sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l1573_157351


namespace angle_bisector_length_is_5_l1573_157308

open Real

noncomputable def triangleAngleBisectorLength (a b c : ℝ) : ℝ :=
  sqrt (a * b * (1 - (c * c) / ((a + b) * (a + b))))

theorem angle_bisector_length_is_5 :
  ∀ (A B C : ℝ), A = 20 ∧ C = 40 ∧ (b - c = 5) →
  triangleAngleBisectorLength a (2 * a * cos (A * π / 180) + 5) (2 * a * cos (A * π / 180)) = 5 :=
  by
  -- you can skip this part with sorry
  sorry

end angle_bisector_length_is_5_l1573_157308


namespace polyhedron_volume_correct_l1573_157321

-- Definitions of geometric shapes and their properties
def is_isosceles_right_triangle (A : Type) (a b c : ℝ) := 
  a = b ∧ c = a * Real.sqrt 2

def is_square (B : Type) (side : ℝ) := 
  side = 2

def is_equilateral_triangle (G : Type) (side : ℝ) := 
  side = Real.sqrt 8

noncomputable def polyhedron_volume (A E F B C D G : Type) (a b c d e f g : ℝ) := 
  let cube_volume := 8
  let tetrahedron_volume := 2 * Real.sqrt 2 / 3
  cube_volume - tetrahedron_volume

theorem polyhedron_volume_correct (A E F B C D G : Type) (a b c d e f g : ℝ) :
  (is_isosceles_right_triangle A a b c) →
  (is_isosceles_right_triangle E a b c) →
  (is_isosceles_right_triangle F a b c) →
  (is_square B d) →
  (is_square C e) →
  (is_square D f) →
  (is_equilateral_triangle G g) →
  a = 2 → d = 2 → e = 2 → f = 2 → g = Real.sqrt 8 →
  polyhedron_volume A E F B C D G a b c d e f g =
    8 - (2 * Real.sqrt 2 / 3) :=
by
  intros hA hE hF hB hC hD hG ha hd he hf hg
  sorry

end polyhedron_volume_correct_l1573_157321


namespace jerry_total_cost_l1573_157381

-- Definition of the costs and quantities
def cost_color : ℕ := 32
def cost_bw : ℕ := 27
def num_color : ℕ := 3
def num_bw : ℕ := 1

-- Definition of the total cost
def total_cost : ℕ := (cost_color * num_color) + (cost_bw * num_bw)

-- The theorem that needs to be proved
theorem jerry_total_cost : total_cost = 123 :=
by
  sorry

end jerry_total_cost_l1573_157381


namespace option_d_always_correct_l1573_157397

variable {a b : ℝ}

theorem option_d_always_correct (h1 : a < b) (h2 : b < 0) (h3 : a < 0) :
  (a + 1 / b)^2 > (b + 1 / a)^2 :=
by
  -- Lean proof code would go here.
  sorry

end option_d_always_correct_l1573_157397


namespace divisor_is_3_l1573_157376

theorem divisor_is_3 (divisor quotient remainder : ℕ) (h_dividend : 22 = (divisor * quotient) + remainder) 
  (h_quotient : quotient = 7) (h_remainder : remainder = 1) : divisor = 3 :=
by
  sorry

end divisor_is_3_l1573_157376


namespace total_students_l1573_157306

theorem total_students (S : ℕ) (h1 : S / 2 / 2 = 250) : S = 1000 :=
by
  sorry

end total_students_l1573_157306


namespace LCM_of_apple_and_cherry_pies_l1573_157372

theorem LCM_of_apple_and_cherry_pies :
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 :=
by
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  have h : (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 := sorry
  exact h

end LCM_of_apple_and_cherry_pies_l1573_157372


namespace unused_square_is_teal_l1573_157385

-- Define the set of colors
inductive Color
| Cyan
| Magenta
| Lime
| Purple
| Teal
| Silver
| Violet

open Color

-- Define the condition that Lime is opposite Purple in the cube
def opposite (a b : Color) : Prop :=
  (a = Lime ∧ b = Purple) ∨ (a = Purple ∧ b = Lime)

-- Define the problem: seven squares are colored and one color remains unused.
def seven_squares_set (hinge : List Color) : Prop :=
  hinge.length = 6 ∧ 
  opposite Lime Purple ∧
  Color.Cyan ∈ hinge ∧
  Color.Magenta ∈ hinge ∧ 
  Color.Lime ∈ hinge ∧ 
  Color.Purple ∈ hinge ∧ 
  Color.Teal ∈ hinge ∧ 
  Color.Silver ∈ hinge ∧ 
  Color.Violet ∈ hinge

theorem unused_square_is_teal :
  ∃ hinge : List Color, seven_squares_set hinge ∧ ¬ (Teal ∈ hinge) := 
by sorry

end unused_square_is_teal_l1573_157385


namespace cuboid_surface_area_l1573_157349

-- Definition of the problem with given conditions and the statement we need to prove.
theorem cuboid_surface_area (h l w: ℝ) (H1: 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100)
                            (H2: l = 2 * h)
                            (H3: w = 2 * h) :
                            (2 * (l * w + l * h + w * h) = 400) :=
by
  sorry

end cuboid_surface_area_l1573_157349


namespace probability_of_stopping_after_2nd_shot_l1573_157353

-- Definitions based on the conditions
def shootingProbability : ℚ := 2 / 3

noncomputable def scoring (n : ℕ) : ℕ := 12 - n

def stopShootingProbabilityAfterNthShot (n : ℕ) (probOfShooting : ℚ) : ℚ :=
  if n = 2 then (1 / 3) * (2 / 3) * sorry -- Note: Here, filling in the remaining calculation steps according to problem logic.
  else sorry -- placeholder for other cases

theorem probability_of_stopping_after_2nd_shot :
  stopShootingProbabilityAfterNthShot 2 shootingProbability = 8 / 729 :=
by
  sorry

end probability_of_stopping_after_2nd_shot_l1573_157353


namespace binomial_20_19_eq_20_l1573_157363

theorem binomial_20_19_eq_20 : Nat.choose 20 19 = 20 :=
by
  sorry

end binomial_20_19_eq_20_l1573_157363


namespace letter_puzzle_solutions_l1573_157352

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l1573_157352


namespace max_value_isosceles_triangle_l1573_157333

theorem max_value_isosceles_triangle (a b c : ℝ) (h_isosceles : b = c) :
  ∃ B, (∀ (a b c : ℝ), b = c → (b + c) / a ≤ B) ∧ B = 2 :=
by
  sorry

end max_value_isosceles_triangle_l1573_157333


namespace hoopit_toes_l1573_157332

theorem hoopit_toes (h : ℕ) : 
  (7 * (4 * h) + 8 * (2 * 5) = 164) -> h = 3 :=
by
  sorry

end hoopit_toes_l1573_157332


namespace avg_prime_factors_of_multiples_of_10_l1573_157357

theorem avg_prime_factors_of_multiples_of_10 : 
  (2 + 5) / 2 = 3.5 :=
by
  -- The prime factors of 10 are 2 and 5.
  -- Therefore, the average of these prime factors is (2 + 5) / 2.
  sorry

end avg_prime_factors_of_multiples_of_10_l1573_157357


namespace range_of_a_l1573_157383

def p (a : ℝ) : Prop := (a + 2) > 1
def q (a : ℝ) : Prop := (4 - 4 * a) ≥ 0
def prop_and (a : ℝ) : Prop := p a ∧ q a
def prop_or (a : ℝ) : Prop := p a ∨ q a
def valid_a (a : ℝ) : Prop := (a ∈ Set.Iic (-1)) ∨ (a ∈ Set.Ioi 1)

theorem range_of_a (a : ℝ) (h_and : ¬ prop_and a) (h_or : prop_or a) : valid_a a := 
sorry

end range_of_a_l1573_157383


namespace closest_points_to_A_l1573_157310

noncomputable def distance_squared (x y : ℝ) : ℝ :=
  x^2 + (y + 3)^2

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 9

theorem closest_points_to_A :
  ∃ (x y : ℝ),
    hyperbola x y ∧
    (distance_squared x y = distance_squared (-3 * Real.sqrt 5 / 2) (-3/2) ∨
     distance_squared x y = distance_squared (3 * Real.sqrt 5 / 2) (-3/2)) :=
sorry

end closest_points_to_A_l1573_157310


namespace speed_of_stream_l1573_157389

-- Definitions based on the conditions
def upstream_speed (c v : ℝ) : Prop := c - v = 4
def downstream_speed (c v : ℝ) : Prop := c + v = 12

-- Main theorem to prove
theorem speed_of_stream (c v : ℝ) (h1 : upstream_speed c v) (h2 : downstream_speed c v) : v = 4 :=
by
  sorry

end speed_of_stream_l1573_157389


namespace movie_store_additional_movie_needed_l1573_157337

theorem movie_store_additional_movie_needed (movies shelves : ℕ) (h_movies : movies = 999) (h_shelves : shelves = 5) : 
  (shelves - (movies % shelves)) % shelves = 1 :=
by
  sorry

end movie_store_additional_movie_needed_l1573_157337


namespace common_external_tangent_b_l1573_157341

def circle1_center := (1, 3)
def circle1_radius := 3
def circle2_center := (10, 6)
def circle2_radius := 7

theorem common_external_tangent_b :
  ∃ (b : ℝ), ∀ (m : ℝ), m = 3 / 4 ∧ b = 9 / 4 := sorry

end common_external_tangent_b_l1573_157341


namespace find_linear_function_and_unit_price_l1573_157334

def linear_function (k b x : ℝ) : ℝ := k * x + b

def profit (cost_price : ℝ) (selling_price : ℝ) (sales_volume : ℝ) : ℝ := 
  (selling_price - cost_price) * sales_volume

theorem find_linear_function_and_unit_price
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1 = 20) (h2 : y1 = 200)
  (h3 : x2 = 25) (h4 : y2 = 150)
  (h5 : x3 = 30) (h6 : y3 = 100)
  (cost_price := 10) (desired_profit := 2160) :
  ∃ k b x : ℝ, 
    (linear_function k b x1 = y1) ∧ 
    (linear_function k b x2 = y2) ∧ 
    (profit cost_price x (linear_function k b x) = desired_profit) ∧ 
    (linear_function k b x = -10 * x + 400) ∧ 
    (x = 22) :=
by
  sorry

end find_linear_function_and_unit_price_l1573_157334


namespace fraction_of_smaller_jar_l1573_157361

theorem fraction_of_smaller_jar (S L : ℝ) (W : ℝ) (F : ℝ) 
  (h1 : W = F * S) 
  (h2 : W = 1/2 * L) 
  (h3 : 2 * W = 2/3 * L) 
  (h4 : S = 2/3 * L) :
  F = 3 / 4 :=
by
  sorry

end fraction_of_smaller_jar_l1573_157361


namespace negate_proposition_l1573_157377

theorem negate_proposition :
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0)) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) :=
by
  sorry

end negate_proposition_l1573_157377


namespace algebraic_expression_value_l1573_157319

theorem algebraic_expression_value (x : ℝ) (hx : x = 2 * Real.cos 45 + 1) :
  (1 / (x - 1) - (x - 3) / (x ^ 2 - 2 * x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end algebraic_expression_value_l1573_157319


namespace number_of_guests_l1573_157345

-- Defining the given conditions
def appetizers_per_guest : ℕ := 6
def deviled_eggs_dozen : ℕ := 3
def pigs_in_blanket_dozen : ℕ := 2
def kebabs_dozen : ℕ := 2
def additional_appetizers_dozen : ℕ := 8

-- The main theorem to prove the number of guests Patsy is expecting
theorem number_of_guests : 
  (deviled_eggs_dozen + pigs_in_blanket_dozen + kebabs_dozen + additional_appetizers_dozen) * 12 / appetizers_per_guest = 30 :=
by
  sorry

end number_of_guests_l1573_157345
