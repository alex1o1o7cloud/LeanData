import Mathlib

namespace part_a_l786_786271

theorem part_a (α β : ℝ) (h₁ : α = 1.0000000004) (h₂ : β = 1.00000000002) (h₃ : α > β) :
  2.00000000002 / (β * β + 2.00000000002) > 2.00000000004 / α := 
sorry

end part_a_l786_786271


namespace angle_A_right_l786_786499

-- Define the sides of the triangle
def AB := Real.sqrt 5
def AC := Real.sqrt 3
def BC := 2 * Real.sqrt 2

-- Define a triangle with these sides
noncomputable def triangle_ABC := Triangle (a := AB) (b := AC) (c := BC)

-- State the theorem we want to prove: ∠A = 90°
theorem angle_A_right (h : AB = Real.sqrt 5 ∧ AC = Real.sqrt 3 ∧ BC = 2 * Real.sqrt 2) :
  triangle_ABC.angle_A = 90 := 
begin
  sorry,
end

end angle_A_right_l786_786499


namespace area_of_triangle_XYZ_l786_786181

-- Definitions to set up the problem
def is_isosceles_right_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ C = A * Real.sqrt 2) ∨ (A = C ∧ B = A * Real.sqrt 2) ∨ (B = C ∧ A = B * Real.sqrt 2)

variables (XY YZ XZ : ℝ)

-- Hypotheses based on conditions
hypothesis h1 : is_isosceles_right_triangle XY YZ XZ
hypothesis h2 : XY > YZ
hypothesis h3 : XY = 14

-- Theorem to prove the area
theorem area_of_triangle_XYZ : 
  (1/2 * YZ * YZ = 49) :=
by sorry

end area_of_triangle_XYZ_l786_786181


namespace fourth_quarter_profit_l786_786679

theorem fourth_quarter_profit
  (first_quarter_profit : ℝ)
  (third_quarter_profit : ℝ)
  (annual_profit : ℝ)
  (h_first : first_quarter_profit = 1500)
  (h_third : third_quarter_profit = 3000)
  (h_annual : annual_profit = 8000) :
  ∃ fourth_quarter_profit : ℝ, fourth_quarter_profit = 3500 :=
by
  let total_first_third := first_quarter_profit + third_quarter_profit
  have h_total : total_first_third = 4500 := by
    rw [h_first, h_third]
    norm_num
  let fourth_quarter_profit := annual_profit - total_first_third
  have h_fourth : fourth_quarter_profit = 3500 := by
    rw [h_annual, h_total]
    norm_num
  use fourth_quarter_profit
  exact h_fourth

end fourth_quarter_profit_l786_786679


namespace wicket_keeper_age_difference_l786_786965

/- 
Conditions: 
The captain is 27 years old.
The average age of the whole team is 24.
The team has 11 members.
The average age of the remaining players (excluding the captain and wicket keeper) is one year less than the average age of the whole team.
-/

def captain_age : ℕ := 27
def team_average_age : ℕ := 24
def team_size : ℕ := 11
def remaining_players_average_age : ℕ := team_average_age - 1

theorem wicket_keeper_age_difference :
  let wicket_keeper_age_diff := 3 in
  team_average_age * team_size = captain_age + (captain_age + wicket_keeper_age_diff) + (remaining_players_average_age * (team_size - 2)) :=
sorry

end wicket_keeper_age_difference_l786_786965


namespace value_of_a_l786_786470

theorem value_of_a (a b c : ℕ) (h1 : a + b = 12) (h2 : b + c = 16) (h3 : c = 7) : a = 3 := by
  sorry

end value_of_a_l786_786470


namespace original_cost_price_40_l786_786299

theorem original_cost_price_40
  (selling_price : ℝ)
  (decrease_rate : ℝ)
  (profit_increase_rate : ℝ)
  (new_selling_price := selling_price)
  (original_cost_price : ℝ)
  (new_cost_price := (1 - decrease_rate) * original_cost_price)
  (original_profit_margin := (selling_price - original_cost_price) / original_cost_price)
  (new_profit_margin := (new_selling_price - new_cost_price) / new_cost_price)
  (profit_margin_increase := profit_increase_rate)
  (h1 : selling_price = 48)
  (h2 : decrease_rate = 0.04)
  (h3 : profit_increase_rate = 0.05)
  (h4 : new_profit_margin = original_profit_margin + profit_margin_increase) :
  original_cost_price = 40 := 
by 
  sorry

end original_cost_price_40_l786_786299


namespace xiao_ming_average_score_l786_786276

theorem xiao_ming_average_score (x y z : ℕ) :
  let avg_first_two := 85 in
  let total_last_three := 270 in
  (2 * avg_first_two + total_last_three) / 5 = 88 := 
by
  sorry

end xiao_ming_average_score_l786_786276


namespace cosine_of_angle_between_diagonals_l786_786315

def a : ℝ × ℝ × ℝ := (3, 2, 1)
def b : ℝ × ℝ × ℝ := (1, 3, 2)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
def norm (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem cosine_of_angle_between_diagonals :
  dot_product (a.1 + b.1, a.2 + b.2, a.3 + b.3) (b.1 - a.1, b.2 - a.2, b.3 - a.3) /
  (norm (a.1 + b.1, a.2 + b.2, a.3 + b.3) * norm (b.1 - a.1, b.2 - a.2, b.3 - a.3)) = 0 :=
by {
  sorry -- proof would go here
}

end cosine_of_angle_between_diagonals_l786_786315


namespace find_a_plus_b_plus_c_l786_786183

open Classical

noncomputable def total_time_in_minutes : ℝ := 60
noncomputable def arrival_distribution := ℝ 
noncomputable def m := 60 - 30 * Real.sqrt 7

def conditions (S1 S2 : arrival_distribution) : Prop :=
  ∃ (m : ℝ), m = 60 - 30 * Real.sqrt 7 ∧ (|S1 - S2| ≤ m)

theorem find_a_plus_b_plus_c : 
  let a := 60
  let b := 30
  let c := 7
  a + b + c = 97 := 
by
  sorry

end find_a_plus_b_plus_c_l786_786183


namespace students_only_math_class_l786_786602

-- Define the problem as a Lean theorem statement.
theorem students_only_math_class 
  (total_students : ℕ)
  (students_math : ℕ)
  (students_science : ℕ)
  (students_language : ℕ)
  (students_union : total_students = 120)
  (students_union_math : students_math = 85)
  (students_union_science : students_science = 70)
  (students_union_language : students_language = 54)
  : (∀ s ∈ {s | s ∈ (students_math \ (students_science ∪ students_language))}, s = 45) :=
by {
  -- insert proof here
  sorry
}

end students_only_math_class_l786_786602


namespace cards_arrangement_l786_786556

theorem cards_arrangement :
  let cards := {1, 2, 3, 4, 5, 6, 7}
  let primes := {2, 3, 5, 7}
  (∃ f : finset ℕ → nat, 
    (∀ p ∈ primes, f (cards.erase p) = 1 + 1) ∧ 
    f ( ∅ ) = 0) := 
  ∑ p in primes, 2 = 8 :=
by sorry

end cards_arrangement_l786_786556


namespace smallest_angle_in_20_sided_polygon_is_143_l786_786579

theorem smallest_angle_in_20_sided_polygon_is_143
  (n : ℕ)
  (h_n : n = 20)
  (angles : ℕ → ℕ)
  (h_convex : ∀ i, 1 ≤ i → i ≤ n → angles i < 180)
  (h_arithmetic_seq : ∃ d : ℕ, ∀ i, 1 ≤ i → i < n → angles (i + 1) = angles i + d)
  (h_increasing : ∀ i, 1 ≤ i → i < n → angles (i + 1) > angles i)
  (h_sum : ∑ i in finset.range n, angles (i + 1) = (n - 2) * 180) :
  angles 1 = 143 :=
by
  sorry

end smallest_angle_in_20_sided_polygon_is_143_l786_786579


namespace find_y_l786_786132

theorem find_y (y : ℝ) (h : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) : y = 53 / 3 :=
by
  sorry

end find_y_l786_786132


namespace evaluate_expression_l786_786732

theorem evaluate_expression : (13! - 12!) / 10! = 1584 :=
by
  sorry

end evaluate_expression_l786_786732


namespace arc_length_identity_l786_786596

noncomputable def radius : ℝ := 18
noncomputable def central_angle : ℝ := π / 3
noncomputable def arc_length : ℝ := 6 * π

theorem arc_length_identity (a : ℝ) (h : arc_length = a * π) : a = 6 := by
  rw [← h]
  simp
  sorry

end arc_length_identity_l786_786596


namespace max_intersections_l786_786659

theorem max_intersections (circle : Type) (lines : list Type)
  (h_distinct_lines : lines.length = 3)
  (h_distinct : ∀ (l₁ l₂ : Type), l₁ ∈ lines → l₂ ∈ lines → l₁ ≠ l₂ → l₁ ∩ l₂ ≠ ∅)
  (h_max_points_circle : ∀ (l : Type), l ∈ lines → count_intersection_points l circle = 2)
  (h_max_points_lines : ∀ (l₁ l₂ : Type), l₁ ∈ lines → l₂ ∈ lines → l₁ ≠ l₂ → count_intersection_points l₁ l₂ = 1) :
  count_intersection_points circle lines = 9 :=
sorry

end max_intersections_l786_786659


namespace inverse_function_proof_l786_786909

theorem inverse_function_proof :
  (∀ x, x < 0 → f (f⁻¹ x) = x) → 
  (∀ y, y^2 + 2 = y⁻¹ → y < 0 → f(y^-1) = y) → 
  let log2_8 := Real.logb 2 8 in 
  f(log2_8) = -1 :=
by
  sorry

end inverse_function_proof_l786_786909


namespace find_z_l786_786840

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786840


namespace largest_initial_number_l786_786043

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786043


namespace problem_solution_l786_786719

theorem problem_solution (y : Fin 8 → ℝ)
  (h1 : y 0 + 4 * y 1 + 9 * y 2 + 16 * y 3 + 25 * y 4 + 36 * y 5 + 49 * y 6 + 64 * y 7 = 2)
  (h2 : 4 * y 0 + 9 * y 1 + 16 * y 2 + 25 * y 3 + 36 * y 4 + 49 * y 5 + 64 * y 6 + 81 * y 7 = 15)
  (h3 : 9 * y 0 + 16 * y 1 + 25 * y 2 + 36 * y 3 + 49 * y 4 + 64 * y 5 + 81 * y 6 + 100 * y 7 = 156)
  (h4 : 16 * y 0 + 25 * y 1 + 36 * y 2 + 49 * y 3 + 64 * y 4 + 81 * y 5 + 100 * y 6 + 121 * y 7 = 1305) :
  25 * y 0 + 36 * y 1 + 49 * y 2 + 64 * y 3 + 81 * y 4 + 100 * y 5 + 121 * y 6 + 144 * y 7 = 4360 :=
sorry

end problem_solution_l786_786719


namespace sum_lent_correct_l786_786661

noncomputable def sum_lent := 26635.94
def annual_interest_rate := 0.0625
def compounding_periods_per_year := 4
def years := 8
def interest_less_than_sum := 8600

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem sum_lent_correct :
  let P := sum_lent,
      r := annual_interest_rate,
      n := compounding_periods_per_year,
      t := years,
      A := compound_interest P r n t
  in A - P = interest_less_than_sum :=
by
  sorry

end sum_lent_correct_l786_786661


namespace sum_b_inverse_l786_786355

noncomputable
def average_reciprocal (n : ℕ) (p : ℕ → ℚ) : ℚ :=
  n / (finset.range n).sum p

noncomputable
def a_n (n : ℕ) : ℚ := 4 * n - 1

noncomputable
def b_n (n : ℕ) : ℚ := (a_n n + 1) / 4

theorem sum_b_inverse {n : ℕ} :
  n > 0 →
  (average_reciprocal n (λ k, (a_n k))) = 1 / (2 * n + 1) →
  ∑ i in finset.range n, 1 / (b_n i * b_n (i + 1)) = (↑n - 1) / ↑n :=
sorry

end sum_b_inverse_l786_786355


namespace find_d_l786_786374

noncomputable def floor_root (d : ℝ) : ℝ := int.floor d
noncomputable def fractional_part (d : ℝ) : ℝ := d - int.floor d

theorem find_d (d : ℝ) (h1 : 3 * (floor_root d)^2 + 19 * (floor_root d) - 63 = 0)
               (h2 : 4 * (fractional_part d)^2 - 21 * (fractional_part d) + 8 = 0) :
  d = -35 / 4 :=
by
  sorry

end find_d_l786_786374


namespace most_economical_speed_cost_l786_786907

noncomputable def optimal_speed (a m n : ℝ) : ℝ :=
  a * real.cbrt (n / (2 * m))

noncomputable def minimum_cost (a m n p : ℝ) : ℝ :=
  (3 * p / a) * real.cbrt (m * n^2 / 4)

theorem most_economical_speed_cost (a m n p : ℝ) :
  (optimal_speed a m n = a * real.cbrt (n / (2 * m))) ∧
  (minimum_cost a m n p = (3 * p / a) * real.cbrt (m * n^2 / 4)) :=
  by
  split
  -- Proof for optimal speed
  sorry
  -- Proof for minimum cost
  sorry

end most_economical_speed_cost_l786_786907


namespace find_phi_range_l786_786151

   -- Conditions 
   def is_intersection_point (ω φ x : ℝ) := sin (ω * x + φ) = cos (ω * x + φ)
   
   def interval := set.Icc 0 (5 * real.sqrt 2 / 2)
   
   def three_intersections (ω φ : ℝ) := 
     ∃ (P M N : ℝ), 
     P ∈ interval ∧ M ∈ interval ∧ N ∈ interval ∧
     P ≠ M ∧ M ≠ N ∧ P ≠ N ∧
     is_intersection_point ω φ P ∧ 
     is_intersection_point ω φ M ∧ 
     is_intersection_point ω φ N

   def is_right_triangle (P M N : ℝ) := 
     -- Placeholder for the right triangle property
     sorry 

   theorem find_phi_range (ω : ℝ) (φ : ℝ) 
       (h1 : φ > 0)
       (h2 : |φ| < real.pi / 2)
       (h3 : three_intersections ω φ)
       (h4 : ∃ (P M N : ℝ), is_right_triangle P M N) 
     : φ ∈ set.Icc (-real.pi / 4) (real.pi / 4)
   := sorry
   
end find_phi_range_l786_786151


namespace monotonic_intervals_when_a_is_4_range_of_a_inequality_ln_l786_786914

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  real.log x - a * (x - 1) / (x + 2)

-- 1. Proving intervals of monotonicity
theorem monotonic_intervals_when_a_is_4 :
  (∀ x ∈ Ioo 0 (4 - 2 * real.sqrt 3), 0 < (deriv (λ x, f x 4) x)) ∧
  (∀ x ∈ Ioo (4 - 2 * real.sqrt 3) (4 + 2 * real.sqrt 3), (deriv (λ x, f x 4) x) < 0) ∧
  (∀ x ∈ Ioo (4 + 2 * real.sqrt 3) (real.infinity), 0 < (deriv (λ x, f x 4) x)) :=
sorry

-- 2. Proving the range of a for which f(x) is increasing in (0,1]
theorem range_of_a :
  (∀ x ∈ Ico 0 1, 0 < (deriv (λ x, f x a) x)) → a ≤ 3 :=
sorry

-- 3. Proving the given inequality
theorem inequality_ln (x₁ x₂ : ℝ):
  x₁ > 0 → x₂ > 0 → x₁ ≤ x₂ → 
  (real.log x₁ - real.log x₂) * (x₁ + 2 * x₂) ≤ 3 * (x₁ - x₂) :=
sorry

end monotonic_intervals_when_a_is_4_range_of_a_inequality_ln_l786_786914


namespace solve_for_z_l786_786867

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786867


namespace Jodi_miles_per_day_in_second_week_l786_786990

theorem Jodi_miles_per_day_in_second_week :
  let first_week_miles := 1 * 6,
      third_week_miles := 3 * 6,
      fourth_week_miles := 4 * 6,
      total_weeks_miles := 60,
      second_week_total_miles := total_weeks_miles - first_week_miles - third_week_miles - fourth_week_miles,
      second_week_days := 6,
      second_week_miles_per_day := second_week_total_miles / second_week_days in
  second_week_miles_per_day = 2 := by
    sorry

end Jodi_miles_per_day_in_second_week_l786_786990


namespace descending_order_numbers_count_l786_786748

theorem descending_order_numbers_count : 
  ∃ (n : ℕ), (n = 1013) ∧ 
  ∀ (x : ℕ), (∃ (xs : list ℕ), 
                (∀ i, i < xs.length - 1 → xs.nth_le i sorry > xs.nth_le (i+1) sorry) ∧ 
                nat_digits_desc xs ∧
                1 < xs.length) → 
             x ∈ nat_digits xs →
             ∃ (refs : list ℕ), n = refs.length ∧ 
             ∀ ref, ref ∈ refs → ref < x :=
sorry

end descending_order_numbers_count_l786_786748


namespace number_of_students_not_good_either_l786_786168

def students_not_good_either (total students_english students_mandarin students_both : ℕ) : ℕ :=
  total - (students_english + students_mandarin - students_both)

theorem number_of_students_not_good_either :
  students_not_good_either 45 35 31 24 = 3 :=
by
  simp [students_not_good_either]
  sorry

end number_of_students_not_good_either_l786_786168


namespace solve_z_l786_786845

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786845


namespace smallest_four_digit_number_divisible_by_6_l786_786266

theorem smallest_four_digit_number_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m % 6 = 0) → n ≤ m :=
begin
  use 1002,
  split,
  { exact nat.le_succ 999,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.le_succ 1001,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  { intros m h1,
    exact le_of_lt_iff.2 (by linarith) }
end

end smallest_four_digit_number_divisible_by_6_l786_786266


namespace train_speed_kmph_l786_786328

/-- Given that the length of the train is 200 meters and it crosses a pole in 9 seconds,
the speed of the train in km/hr is 80. -/
theorem train_speed_kmph (length : ℝ) (time : ℝ) (length_eq : length = 200) (time_eq : time = 9) : 
  (length / time) * (3600 / 1000) = 80 :=
by
  sorry

end train_speed_kmph_l786_786328


namespace solve_for_z_l786_786864

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786864


namespace number_of_ways_l786_786217

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786217


namespace units_digit_of_modifiedLucas_379_l786_786357

-- Define the modified Lucas sequence
def modifiedLucas : ℕ → ℕ
| 0     := 3
| 1     := 1
| (n+2) := 2 * modifiedLucas (n + 1) + modifiedLucas n

-- Define the function to get the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- State the main theorem
theorem units_digit_of_modifiedLucas_379 : units_digit (modifiedLucas 379) = 3 :=
sorry

end units_digit_of_modifiedLucas_379_l786_786357


namespace crowns_count_l786_786397

theorem crowns_count (total_feathers : ℕ) (feathers_per_crown : ℕ) (h_total : total_feathers = 6538) (h_feathers : feathers_per_crown = 7) :
  total_feathers / feathers_per_crown = 934 := 
by
  rw [h_total, h_feathers]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end crowns_count_l786_786397


namespace expression_evaluation_l786_786343

theorem expression_evaluation : 70 + (105 / 15) + (19 * 11) - 250 - (360 / 12) = 6 := by
  have h1 : 105 / 15 = 7 := by norm_num
  have h2 : 19 * 11 = 209 := by norm_num
  have h3 : 360 / 12 = 30 := by norm_num
  calc
    70 + (105 / 15) + (19 * 11) - 250 - (360 / 12)
    = 70 + 7 + (19 * 11) - 250 - (360 / 12) : by rw [h1]
    ... = 70 + 7 + 209 - 250 - (360 / 12) : by rw [h2]
    ... = 70 + 7 + 209 - 250 - 30 : by rw [h3]
    ... = 6 : by norm_num

end expression_evaluation_l786_786343


namespace combination_sum_32_l786_786765

theorem combination_sum_32 (n : ℕ) (h : ∑ i in finset.range (n + 1), (2^i * nat.choose n i) = 729) :
  (nat.choose n 1 + nat.choose n 3 + nat.choose n 5) = 32 :=
sorry

end combination_sum_32_l786_786765


namespace largest_initial_number_l786_786022

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786022


namespace students_material_selection_l786_786223

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786223


namespace number_of_ways_l786_786213

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786213


namespace compute_expression_l786_786706

theorem compute_expression :
  3 * 3^4 - 9^60 / 9^57 = -486 :=
by
  sorry

end compute_expression_l786_786706


namespace exists_n_2000_prime_divisors_l786_786724

-- Definition of the problem conditions
def has_prime_divisors (n : ℕ) (k : ℕ) : Prop :=
  nat.prime_factors n = k

-- The theorem statement
theorem exists_n_2000_prime_divisors : ∃ (n : ℕ), has_prime_divisors n 2000 ∧ n ∣ 2^n + 1 :=
sorry

end exists_n_2000_prime_divisors_l786_786724


namespace product_of_real_parts_of_roots_l786_786390

noncomputable def quadratic_roots_real_product (a b c : ℂ) : ℝ :=
  let roots := {x | x^2 - (b : ℂ) * x + c = 0}
  in (roots.someRealPart * roots.anotherRealPart) -- simplified for brevity

theorem product_of_real_parts_of_roots : 
  quadratic_roots_real_product 1 (-2) (12 - 7 * complex.i) = -13.5 :=
sorry

end product_of_real_parts_of_roots_l786_786390


namespace solve_for_z_l786_786875

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786875


namespace students_material_selection_l786_786220

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786220


namespace value_of_40th_number_is_12_l786_786013

def sequence_value (n : ℕ) : ℕ :=
  let row_of_num := (n + 1) / 2 -- find the row that contains the nth number
  in 2 * row_of_num

theorem value_of_40th_number_is_12 : sequence_value 40 = 12 := 
by
  sorry

end value_of_40th_number_is_12_l786_786013


namespace negation_of_real_root_proposition_l786_786549

theorem negation_of_real_root_proposition :
  (¬ ∃ m : ℝ, ∃ (x : ℝ), x^2 + m * x + 1 = 0) ↔ (∀ m : ℝ, ∀ (x : ℝ), x^2 + m * x + 1 ≠ 0) :=
by
  sorry

end negation_of_real_root_proposition_l786_786549


namespace solve_for_y_l786_786130

theorem solve_for_y (y : ℝ) (h : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) : y = 53 / 3 := by
  sorry

end solve_for_y_l786_786130


namespace sum_of_coordinates_of_B_l786_786111

theorem sum_of_coordinates_of_B : 
  ∃ (x : ℚ), ∃ (B : ℚ × ℚ),
    B.1 = x ∧ B.2 = 5 ∧ (unit $ (5 - 0) / (x - 0) = 3/4) ∧ 
    B.1 + B.2 = 35 / 3 :=
begin
  -- Circle in sorry because only the statement is required.
  sorry
end

end sum_of_coordinates_of_B_l786_786111


namespace part1_part2_l786_786094

def seq (n : ℕ) : ℝ := 1 / (n * (n + 1))

theorem part1 (n : ℕ) (hn : n > 0) : 
  1 / n = 1 / (n + 1) + seq n := 
sorry

theorem part2 (n : ℕ) (hn : n > 1) :
  ∃ r s : ℕ, r < s ∧ 1 / n = (finset.range (s - r + 1)).sum (λ k, seq (r + k)) :=
sorry

end part1_part2_l786_786094


namespace solve_for_z_l786_786767

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786767


namespace maximize_x3y4_l786_786090

noncomputable def maximize_expr (x y : ℝ) : ℝ :=
x^3 * y^4

theorem maximize_x3y4 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 50 ∧ maximize_expr x y = maximize_expr 30 20 :=
by
  sorry

end maximize_x3y4_l786_786090


namespace compute_exponent_problem_l786_786705

noncomputable def exponent_problem : ℤ :=
  3 * (3^4) - (9^60) / (9^57)

theorem compute_exponent_problem : exponent_problem = -486 := by
  sorry

end compute_exponent_problem_l786_786705


namespace measure_of_y_l786_786384

variables (A B C D : Point) (y : ℝ)
-- Given conditions
def angle_ABC := 120
def angle_BAD := 30
def angle_BDA := 21
def angle_ABD := 180 - angle_ABC

-- Theorem to prove
theorem measure_of_y :
  angle_BAD + angle_ABD + angle_BDA + y = 180 → y = 69 :=
by
  sorry

end measure_of_y_l786_786384


namespace find_n_modulo_conditions_l786_786380

theorem find_n_modulo_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n % 7 = -3137 % 7 ∧ (n = 1 ∨ n = 8) := sorry

end find_n_modulo_conditions_l786_786380


namespace turtles_in_lake_l786_786177

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end turtles_in_lake_l786_786177


namespace smallest_angle_in_icosagon_l786_786582

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end smallest_angle_in_icosagon_l786_786582


namespace no_convex_quadrilateral_with_acute_diagonals_l786_786984

theorem no_convex_quadrilateral_with_acute_diagonals :
  ¬ ∃ (A B C D : Point), 
      ConvexQuadrilateral A B C D ∧ 
      (AcuteTriangle A B C ∧ AcuteTriangle A C D) ∧ 
      (AcuteTriangle B C D ∧ AcuteTriangle A B D) :=
by
  sorry

end no_convex_quadrilateral_with_acute_diagonals_l786_786984


namespace determinant_scaling_l786_786896

variable (p q r s : ℝ)

theorem determinant_scaling 
  (h : Matrix.det ![![p, q], ![r, s]] = 3) : 
  Matrix.det ![![2 * p, 2 * p + 5 * q], ![2 * r, 2 * r + 5 * s]] = 30 :=
by 
  sorry

end determinant_scaling_l786_786896


namespace proof_equivalent_problem_l786_786117

-- Definition of conditions
def cost_condition_1 (x y : ℚ) : Prop := 500 * x + 40 * y = 1250
def cost_condition_2 (x y : ℚ) : Prop := 1000 * x + 20 * y = 1000
def budget_condition (a b : ℕ) (total_masks : ℕ) (budget : ℕ) : Prop := 2 * a + (total_masks - a) / 2 + 25 * b = budget

-- Main theorem
theorem proof_equivalent_problem : 
  ∃ (x y : ℚ) (a b : ℕ), 
    cost_condition_1 x y ∧
    cost_condition_2 x y ∧
    (x = 1 / 2) ∧ 
    (y = 25) ∧
    (budget_condition a b 200 400) ∧
    ((a = 150 ∧ b = 3) ∨
     (a = 100 ∧ b = 6) ∨
     (a = 50 ∧ b = 9)) :=
by {
  sorry -- The proof steps are not required
}

end proof_equivalent_problem_l786_786117


namespace descending_digits_count_l786_786742

theorem descending_digits_count : 
  ∑ k in (finset.range 11).filter (λ k, 2 ≤ k), nat.choose 10 k = 1013 := 
sorry

end descending_digits_count_l786_786742


namespace largest_initial_number_l786_786050

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786050


namespace find_z_l786_786811

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786811


namespace solve_for_z_l786_786770

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786770


namespace correct_option_D_l786_786272

theorem correct_option_D (a b : ℝ) :
  (a^2 * a^3 = a^5) ∧ 
  ((-a^3 * b)^2 = a^6 * b^2) ∧ 
  (a^6 / a^3 = a^3) ∧ 
  ((a^2)^3 = a^6) → 
  ((a^2)^3 = a^6) :=
by
  intro h
  exact h.2.2.2
  sorry

end correct_option_D_l786_786272


namespace sqrt_product_eq_l786_786710

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l786_786710


namespace students_material_selection_l786_786222

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786222


namespace two_students_choose_materials_l786_786185

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786185


namespace triangle_properties_l786_786488

-- Defining the right triangle ∆ DEF
variables (DEF : Type) [triangle DEF] 
           (D E F N : DEF.Point) 
           (DF DE EF : Real)
           (HN : mid_point N E F)
           (right_angle_DFE : DEF.ang D F E = 90)
           (length_DF : DF = 6)
           (length_DE : DE = 8)

-- Goal: Prove DN = 5 cm and Area(∆ DEF) = 24 cm^2
theorem triangle_properties
  (EF_calc : EF = Real.sqrt (DF^2 + DE^2))
  (median_DN : DEF.seg_len D N = EF / 2)
  (area_DEF : DEF.area D E F = (1 / 2) * DF * DE) :
  DEF.seg_len D N = 5 ∧ DEF.area D E F = 24 :=
by {
  sorry
}

end triangle_properties_l786_786488


namespace number_of_ways_l786_786216

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786216


namespace number_of_ways_to_choose_materials_l786_786235

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786235


namespace minimal_closed_broken_line_non_self_intersecting_l786_786660

def is_closed_broken_line (L : List (ℝ × ℝ)) : Prop :=
  L.head = L.last

def length_of_line (L : List (ℝ × ℝ)) : ℝ :=
  L.zip (L.tail ++ [L.head]).sum (λ ((x1, y1), (x2, y2)), (x2 - x1)^2 + (y2 - y1)^2).sqrt

def is_non_self_intersecting (L : List (ℝ × ℝ)) : Prop :=
  -- define what it means for a polygon to be non-self-intersecting
  sorry

theorem minimal_closed_broken_line_non_self_intersecting (L : List (ℝ × ℝ))
  (hL_closed : is_closed_broken_line L)
  (hL_minimal : ∀ L', (is_closed_broken_line L') → (List.length L = List.length L') → length_of_line L ≤ length_of_line L') :
  is_non_self_intersecting L :=
sorry

end minimal_closed_broken_line_non_self_intersecting_l786_786660


namespace cos_identity_l786_786473

theorem cos_identity (α : ℝ) (h : sin α = -2 * cos α) : 
  cos (2 * α + π / 3) = (4 * real.sqrt 3 - 3) / 10 := 
sorry

end cos_identity_l786_786473


namespace largest_initial_number_l786_786066

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786066


namespace polynomial_value_l786_786253

noncomputable def polynomial_spec (p : ℝ) : Prop :=
  p^3 - 5 * p + 1 = 0

theorem polynomial_value (p : ℝ) (h : polynomial_spec p) : 
  p^4 - 3 * p^3 - 5 * p^2 + 16 * p + 2015 = 2018 := 
by
  sorry

end polynomial_value_l786_786253


namespace remainder_of_144_div_k_l786_786396

theorem remainder_of_144_div_k
  (k : ℕ)
  (h1 : 0 < k)
  (h2 : 120 % k^2 = 12) :
  144 % k = 0 :=
by
  sorry

end remainder_of_144_div_k_l786_786396


namespace baseball_team_wins_more_than_three_times_losses_l786_786312

theorem baseball_team_wins_more_than_three_times_losses
    (total_games : ℕ)
    (wins : ℕ)
    (losses : ℕ)
    (h1 : total_games = 130)
    (h2 : wins = 101)
    (h3 : wins + losses = total_games) :
    wins - 3 * losses = 14 :=
by
    -- Proof goes here
    sorry

end baseball_team_wins_more_than_three_times_losses_l786_786312


namespace largest_initial_number_l786_786020

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786020


namespace area_difference_of_squares_l786_786163

theorem area_difference_of_squares :
  let side1 := 22
  let side2 := 20
  let area1 := side1^2
  let area2 := side2^2
  area1 - area2 = 84 :=
by
  let side1 := 22
  let side2 := 20
  let area1 := side1^2
  let area2 := side2^2
  calc area1 - area2 = 484 - 400 : by sorry
  ... = 84 : by sorry

end area_difference_of_squares_l786_786163


namespace minimum_difference_l786_786959

def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem minimum_difference (x y z : ℤ) 
  (hx : even x) (hy : odd y) (hz : odd z)
  (hxy : x < y) (hyz : y < z) (hzx : z - x = 9) : y - x = 1 := 
sorry

end minimum_difference_l786_786959


namespace largest_initial_number_l786_786044

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786044


namespace range_of_m_length_of_chord_l786_786437

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

noncomputable def line_eq (x y m : ℝ) : Prop := y = x + m

theorem range_of_m (m : ℝ) : (∃ x y : ℝ, ellipse_eq x y ∧ line_eq x y m) ↔ -real.sqrt 3 ≤ m ∧ m ≤ real.sqrt 3 :=
sorry

theorem length_of_chord (x1 y1 x2 y2 : ℝ) (h1 : ellipse_eq x1 y1) (h2 : ellipse_eq x2 y2) 
(hline1 : line_eq x1 y1 (-1)) (hline2 : line_eq x2 y2 (-1)) : 
real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = (4 / 3) * real.sqrt 2 :=
sorry

end range_of_m_length_of_chord_l786_786437


namespace negation_P_l786_786925

open Real

theorem negation_P :
  (∀ x : ℝ, x > sin x) → (∃ x : ℝ, x ≤ sin x) :=
sorry

end negation_P_l786_786925


namespace descending_digits_count_l786_786744

theorem descending_digits_count : 
  ∑ k in (finset.range 11).filter (λ k, 2 ≤ k), nat.choose 10 k = 1013 := 
sorry

end descending_digits_count_l786_786744


namespace find_exponent_l786_786953

theorem find_exponent 
  (h1 : (1 : ℝ) / 9 = 3 ^ (-2 : ℝ))
  (h2 : (3 ^ (20 : ℝ) : ℝ) / 9 = 3 ^ x) : 
  x = 18 :=
by sorry

end find_exponent_l786_786953


namespace equidistant_points_form_parabola_l786_786756

theorem equidistant_points_form_parabola :
  let A := (5, -1)
  let line : ℝ × ℝ → Prop := λ p, p.1 + p.2 - 1 = 0
  ∀ p : ℝ × ℝ, (dist p A = dist p (5, -1, line)) → 
    p ∈ { q : ℝ × ℝ | 2 * q.1 + 0 * q.2 = 10 } :=
begin
  sorry
end

end equidistant_points_form_parabola_l786_786756


namespace linear_function_passing_through_point_l786_786908

theorem linear_function_passing_through_point (k : ℝ) :
  (∀ x : ℝ, y = k * x - 2 ∧ (∃ x : ℝ, y = 0) ∧ (area_of_triangle_with_axes line = 3)) ↔ 
  (∀ x : ℝ, (y = (2/3) * x - 2) ∨ (y = -(2/3) * x - 2)) :=
by 
sorry

end linear_function_passing_through_point_l786_786908


namespace number_of_ways_to_choose_materials_l786_786234

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786234


namespace calculation_correctness_l786_786344

theorem calculation_correctness : 15 - 14 * 3 + 11 / 2 - 9 * 4 + 18 = -39.5 := by
  sorry

end calculation_correctness_l786_786344


namespace students_material_selection_l786_786228

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786228


namespace lcm_of_8_and_15_l786_786739

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 :=
by
  sorry

end lcm_of_8_and_15_l786_786739


namespace largest_initial_number_l786_786060

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786060


namespace train_speed_l786_786667

-- Define the platform length in meters and the time taken to cross in seconds
def platform_length : ℝ := 260
def time_crossing : ℝ := 26

-- Define the length of the goods train in meters
def train_length : ℝ := 260.0416

-- Define the total distance covered by the train when crossing the platform
def total_distance : ℝ := platform_length + train_length

-- Define the speed of the train in meters per second
def speed_m_s : ℝ := total_distance / time_crossing

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the speed of the train in kilometers per hour
def speed_km_h : ℝ := speed_m_s * conversion_factor

-- State the theorem to be proved
theorem train_speed : speed_km_h = 72.00576 :=
by
  sorry

end train_speed_l786_786667


namespace inequality_sum_fractions_l786_786552

theorem inequality_sum_fractions (n : ℕ) (hn : n > 1) :
  ∑ i in Finset.range (n + 1) (λ k, 1 / (n + 1 + k)) > (13 / 24 : ℝ) := by
  sorry

end inequality_sum_fractions_l786_786552


namespace distinct_shell_arrangements_l786_786510

/--
John draws a regular five pointed star and places one of ten different sea shells at each of the 5 outward-pointing points and 5 inward-pointing points. 
Considering rotations and reflections of an arrangement as equivalent, prove that the number of ways he can place the shells is 362880.
-/
theorem distinct_shell_arrangements : 
  let total_arrangements := Nat.factorial 10
  let symmetries := 10
  total_arrangements / symmetries = 362880 :=
by
  sorry

end distinct_shell_arrangements_l786_786510


namespace probability_of_winning_pair_l786_786966

-- Define the cards and their properties
inductive Color
| Red
| Green

inductive Label
| A
| B
| C

structure Card :=
  (color : Color)
  (label : Label)

-- Define the deck of cards
def deck : Finset Card :=
  {⟨Color.Red, Label.A⟩, ⟨Color.Red, Label.B⟩, ⟨Color.Red, Label.C⟩,
   ⟨Color.Green, Label.A⟩, ⟨Color.Green, Label.B⟩, ⟨Color.Green, Label.C⟩}

-- Define what it means to draw a winning pair
def is_winning_pair (c1 c2 : Card) : Prop :=
  (c1.color = c2.color) ∨ (c1.label = c2.label)

-- Problem statement: prove the probability of drawing a winning pair
theorem probability_of_winning_pair : 
  (deck.cardinal.choose 2) = 15 →
  ∀ (winning_count : Nat), 
    (∃ (p : Nat), p = 9 ∧ p = winning_count) →
    ∃ p : ℚ, p = (9 : ℚ) / 15 :=
begin
  sorry
end

end probability_of_winning_pair_l786_786966


namespace triangle_inequalities_l786_786500

variable {A B C M : Type} -- Define the variables

-- Define the setup using Lean's definition and theorem statements
def bisector (A B C : Point) (M : Point) : Prop :=
  -- BM is the bisector of the angle ABC and M lies on AC
  sorry

def angle_bisector_theorem {A B C M : Point} (h : bisector A B C M) : Prop :=
  (∃ k : ℝ, k = (dist A M) / (dist M C) ∧ k = (dist A B) / (dist B C))

theorem triangle_inequalities
  {A B C M : Point} (h : bisector A B C M) (ht : angle_bisector_theorem h) :
  dist A M < dist A B ∧ dist M C < dist B C :=
begin
  sorry
end

end triangle_inequalities_l786_786500


namespace slice_of_bread_area_l786_786678

theorem slice_of_bread_area (total_area : ℝ) (number_of_parts : ℕ) (h1 : total_area = 59.6) (h2 : number_of_parts = 4) : 
  total_area / number_of_parts = 14.9 :=
by
  rw [h1, h2]
  norm_num


end slice_of_bread_area_l786_786678


namespace range_of_a_l786_786534

def setA (a : ℝ) : set ℝ := {x | x ≤ a}
def setB : set ℝ := {x | x^2 - 5 * x < 0}

theorem range_of_a (a : ℝ) : (setA a ∩ setB = setB) → a ≥ 5 :=
sorry

end range_of_a_l786_786534


namespace equation_result_l786_786634

theorem equation_result : 
  ∀ (n : ℝ), n = 5.0 → (4 * n + 7 * n) = 55.0 :=
by
  intro n h
  rw [h]
  norm_num

end equation_result_l786_786634


namespace find_z_l786_786817

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786817


namespace relationship_a_b_c_l786_786413

-- Define the variables with the given conditions
def a := 4 ^ 0.6
def b := 8 ^ 0.34
def c := (1 / 2) ^ -0.9

-- State the theorem that proves the relationship between a, b, and c
theorem relationship_a_b_c : a > b ∧ b > c := by
  -- Proof will be filled here
  sorry

end relationship_a_b_c_l786_786413


namespace solve_for_z_l786_786855

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786855


namespace value_of_m_l786_786405

theorem value_of_m : ∃ (m : ℕ), (3 * 4 * 5 * m = Nat.factorial 8) ∧ m = 672 := by
  sorry

end value_of_m_l786_786405


namespace smallest_four_digit_divisible_by_six_l786_786259

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end smallest_four_digit_divisible_by_six_l786_786259


namespace total_number_of_turtles_l786_786179

variable {T : Type} -- Define a variable for the type of turtles

-- Define the conditions as hypotheses
variable (total_turtles : ℕ)
variable (female_percentage : ℚ) (male_percentage : ℚ)
variable (striped_male_prop : ℚ)
variable (baby_striped_males : ℕ) (adult_striped_males_prop : ℚ)
variable (striped_male_percentage : ℚ)
variable (striped_males : ℕ)
variable (male_turtles : ℕ)

-- Condition definitions
def female_percentage_def := female_percentage = 60 / 100
def male_percentage_def := male_percentage = 1 - female_percentage
def striped_male_prop_def := striped_male_prop = 1 / 4
def adult_striped_males_prop_def := adult_striped_males_prop = 60 / 100
def baby_and_adult_striped_males_prop_def := (1 - adult_striped_males_prop) = 40 / 100
def striped_males_def := striped_males = baby_striped_males / (1 - adult_striped_males_prop)
def male_turtles_def := male_turtles = striped_males / striped_male_prop
def male_turtles_percentage_def := male_turtles = total_turtles * (1 - female_percentage)

-- The proof statement to show the total number of turtles is 100
theorem total_number_of_turtles (h_female : female_percentage_def)
                                (h_male : male_percentage_def)
                                (h_striped_male_prop : striped_male_prop_def)
                                (h_adult_striped_males_prop : adult_striped_males_prop_def)
                                (h_baby_and_adult_striped_males_prop : baby_and_adult_striped_males_prop_def)
                                (h_striped_males : striped_males_def)
                                (h_male_turtles : male_turtles_def)
                                (h_male_turtles_percentage : male_turtles_percentage_def):
  total_turtles = 100 := 
by sorry

end total_number_of_turtles_l786_786179


namespace animal_count_inconsistency_l786_786103

theorem animal_count_inconsistency (x y z g : ℕ) 
  (h1 : x = 2 * y) 
  (h2 : y = 310)
  (h3 : z = 180)
  (h4 : x + y + z + g = 900) : False :=
by 
  have hx : x = 2 * 310 := by rw [h2, h1]
  have hx_val : x = 620 := by norm_num [hx]
  have total_val : 620 + 310 + 180 + g = 900 := by rw [hx_val, h2, h3, h4]
  have sum_val : 1110 + g = 900 := by norm_num [hx_val, h2, h3]
  have inconsistency : 1110 ≤ 900 := by linarith
  exact not_le_of_gt (by norm_num : 1110 > 900) inconsistency

end animal_count_inconsistency_l786_786103


namespace students_behind_yoongi_l786_786650

theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) (behind_students : ℕ)
  (h1 : total_students = 20)
  (h2 : jungkook_position = 3)
  (h3 : yoongi_position = jungkook_position + 1)
  (h4 : behind_students = total_students - yoongi_position) :
  behind_students = 16 :=
by
  sorry

end students_behind_yoongi_l786_786650


namespace number_of_ways_to_choose_reading_materials_l786_786241

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786241


namespace largest_initial_number_l786_786041

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786041


namespace trains_pass_each_other_time_l786_786327

variables (v1 v2 l1 l2 : ℝ)

-- Conditions
def condition1 : Prop := 2 * v2 = v1
def condition2 : Prop := 3 * l2 = l1
def condition3 : Prop := l1 / v1 = 5

-- Problem statement
theorem trains_pass_each_other_time (hv1 : v1 ≠ 0) (hv2 : v2 ≠ 0) (hl1 : l1 ≠ 0) (hl2 : l2 ≠ 0) :
  condition1 v1 v2 → condition2 l1 l2 → condition3 l1 v1 →
  let v_relative := v1 + v2
      l_combined := l1 + l2 in
  v_relative ≠ 0 →
  (l_combined / v_relative) = 40 / 9 := sorry

end trains_pass_each_other_time_l786_786327


namespace students_material_selection_l786_786227

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786227


namespace modulus_of_complex_number_l786_786601

theorem modulus_of_complex_number :
  |(1 : ℂ) / (complex.I - 1)| = (real.sqrt 2) / 2 :=
by
  sorry

end modulus_of_complex_number_l786_786601


namespace mutual_exclusive_non_complementary_l786_786345

def all_even (l : List ℕ) : Prop := ∀ x ∈ l, x % 2 = 0
def all_odd  (l : List ℕ) : Prop := ∀ x ∈ l, x % 2 = 1
def at_least_one_odd (l : List ℕ) : Prop := ∃ x ∈ l, x % 2 = 1
def at_most_one_odd (l : List ℕ) : Prop := (l.filter (λ x, x % 2 = 1)).length ≤ 1
def one_even_two_odd (l : List ℕ) : Prop := (l.filter (λ x, x % 2 = 0)).length = 1 ∧ (l.filter (λ x, x % 2 = 1)).length = 2
def two_even_one_odd (l : List ℕ) : Prop := (l.filter (λ x, x % 2 = 0)).length = 2 ∧ (l.filter (λ x, x % 2 = 1)).length = 1

theorem mutual_exclusive_non_complementary :
  ∀ l : List ℕ, l.length = 3 ∧ (∀ x ∈ l, x ∈ (List.range' 1 9)) →
  (
    (all_even l ∧ all_odd l) ∨ 
    (one_even_two_odd l ∧ two_even_one_odd l)
  ) ↔ False :=
by sorry

end mutual_exclusive_non_complementary_l786_786345


namespace find_z_l786_786835

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786835


namespace train_speed_in_kph_l786_786682

-- Define the given conditions
def length_of_train : ℝ := 200 -- meters
def time_crossing_pole : ℝ := 16 -- seconds

-- Define conversion factor
def mps_to_kph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Statement of the theorem
theorem train_speed_in_kph : mps_to_kph (length_of_train / time_crossing_pole) = 45 := 
sorry

end train_speed_in_kph_l786_786682


namespace find_z_l786_786819

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786819


namespace solve_complex_equation_l786_786788

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786788


namespace solve_complex_equation_l786_786793

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786793


namespace ratio_AP_to_PC_l786_786971

variable {A B C D M N P : Type}
variable [HasSubtype ℝ]

-- Defining a parallelogram
def is_parallelogram (ABCD: Type) (A B C D : ℝ) := sorry

-- Point M on AB such that AM/MB = 2/5
def M_on_AB (A B M : ℝ) : Prop := (∃ k : ℝ, 0 < k ∧ k = 7 ∧ AM = k * 2 / 7 ∧ MB = k * 5 / 7)

-- Point N on AD such that AN/ND = 2/5
def N_on_AD (A D N : ℝ) : Prop := (∃ k : ℝ, 0 < k ∧ k = 7 ∧ AN = k * 2 / 7 ∧ ND = k * 5 / 7)

-- Defining intersection point P of AC and MN
def P_intersection (A C M N P : ℝ) : Prop := sorry

theorem ratio_AP_to_PC (ABCD : Type) (A B C D M N P : ℝ)
  (h_parallelogram : is_parallelogram ABCD A B C D)
  (h_M_on_AB : M_on_AB A B M)
  (h_N_on_AD : N_on_AD A D N)
  (h_P_intersection : P_intersection A C M N P) : 
  (AP / PC = 2 / 5) := 
sorry

end ratio_AP_to_PC_l786_786971


namespace terminal_side_quadrant_l786_786463

theorem terminal_side_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) :
  ∃ k : ℤ, (k % 2 = 0 ∧ (k * Real.pi + Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + Real.pi / 2)) ∨
           (k % 2 = 1 ∧ (k * Real.pi + 3 * Real.pi / 4 < α / 2 ∧ α / 2 < k * Real.pi + 5 * Real.pi / 4)) := sorry

end terminal_side_quadrant_l786_786463


namespace john_final_weight_l786_786512

-- Definitions based on the conditions provided

def initial_weight : ℝ := 220
def lose_percent (weight : ℝ) (percent : ℝ) : ℝ := weight - (weight * percent)
def gain (weight : ℝ) (amount : ℝ) : ℝ := weight + amount

-- Statement that encapsulates all the steps and final result

theorem john_final_weight :
  let weight_after_first_loss := lose_percent initial_weight 0.10 in
  let weight_after_first_gain := gain weight_after_first_loss 5 in
  let weight_after_second_loss := lose_percent weight_after_first_gain 0.15 in
  let weight_after_second_gain := gain weight_after_second_loss 8 in
  let final_weight := lose_percent weight_after_second_gain 0.20 in
  final_weight = 144.44 :=
by
  sorry

end john_final_weight_l786_786512


namespace problem_part1_problem_part2_l786_786900

-- Problem statements

theorem problem_part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 := 
sorry

theorem problem_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) : 
  a + b ≤ 2 := 
sorry

end problem_part1_problem_part2_l786_786900


namespace largest_initial_number_l786_786058

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786058


namespace solve_f_inequality_l786_786880

def f (x : ℝ) : ℝ :=
if x > 0 then ln (1 / x) else 1 / x

theorem solve_f_inequality :
  {x : ℝ | f x > -1} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < e} :=
by
  sorry

end solve_f_inequality_l786_786880


namespace collinear_points_A_P_Q_l786_786963

-- Define the circles and their intersection points
variables {Γ Γ' : Circle} {A B : Point} (hAB : A ∈ Γ ∧ A ∈ Γ' ∧ B ∈ Γ ∧ B ∈ Γ')

-- Define the tangents and their intersection points with the circles
variables {C D : Point} 
  (hC : tangent Γ A ∧ C ∈ Γ')
  (hD : tangent Γ' A ∧ D ∈ Γ)

-- Define the points of intersection of CD with the circles
variables {E F : Point}
  (hE : E ∈ lineSegment C D ∧ E ∈ Γ')
  (hF : F ∈ lineSegment C D ∧ F ∈ Γ)

-- Define the points where the perpendiculars intersect the circles
variables {P Q : Point}
  (hP : P ∈ perpFrom E (lineSegment A C) ∧ P ∈ Γ')
  (hQ : Q ∈ perpFrom F (lineSegment A D) ∧ Q ∈ Γ)

-- State that points A, P, and Q are on the same side of line CD
variables (hSide : sameSide A P Q (lineSegment C D))

#check Point A P Q

-- The goal is to prove that points A, P, and Q are collinear
theorem collinear_points_A_P_Q : collinear A P Q :=
by
  sorry

end collinear_points_A_P_Q_l786_786963


namespace number_of_ways_to_choose_materials_l786_786231

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786231


namespace proof_problem_l786_786428

theorem proof_problem (x y : ℝ) (hx : 2^x = 3) (hy : log 2 (8/9) = y) : 2 * x + y = 3 :=
sorry

end proof_problem_l786_786428


namespace find_z_l786_786808

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786808


namespace two_students_one_common_material_l786_786198

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786198


namespace one_plane_halves_rect_prism_l786_786320

theorem one_plane_halves_rect_prism :
  ∀ (T : Type) (a b c : ℝ)
  (x y z : ℝ) 
  (black_prisms_volume white_prisms_volume : ℝ),
  (black_prisms_volume = (x * y * z + x * (b - y) * (c - z) + (a - x) * y * (c - z) + (a - x) * (b - y) * z)) ∧
  (white_prisms_volume = ((a - x) * (b - y) * (c - z) + (a - x) * y * z + x * (b - y) * z + x * y * (c - z))) ∧
  (black_prisms_volume = white_prisms_volume) →
  (x = a / 2 ∨ y = b / 2 ∨ z = c / 2) :=
by
  sorry

end one_plane_halves_rect_prism_l786_786320


namespace smallest_positive_period_ω_range_of_f_l786_786100

noncomputable def f (ω λ : ℝ) (x : ℝ) : ℝ :=
  sin(ω * x) ^ 2 - cos(ω * x) ^ 2 + 2 * sqrt 3 * sin(ω * x) * cos(ω * x) + λ

theorem smallest_positive_period_ω (ω λ : ℝ) (hω : 1 / 2 < ω ∧ ω < 1) (h_symm : ∀ x, f ω λ (2 * π - x) = f ω λ x) :
  ω = 5 / 6 :=
sorry

theorem range_of_f (λ : ℝ) (hλ : 2 * sin(5 / 3 * π / 4 - π / 6) + λ = 0):
  ∀ x, 0 ≤ x ∧ x ≤ 3 * π / 5 → -1 - sqrt 2 ≤ f (5 / 6) λ x ∧ f (5 / 6) λ x ≤ 2 - sqrt 2 :=
sorry

end smallest_positive_period_ω_range_of_f_l786_786100


namespace solve_z_l786_786785

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786785


namespace smallest_four_digit_number_divisible_by_6_l786_786264

theorem smallest_four_digit_number_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m % 6 = 0) → n ≤ m :=
begin
  use 1002,
  split,
  { exact nat.le_succ 999,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.le_succ 1001,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  { intros m h1,
    exact le_of_lt_iff.2 (by linarith) }
end

end smallest_four_digit_number_divisible_by_6_l786_786264


namespace problem1_problem2_l786_786701

open Real

noncomputable def expr1 : ℝ := 
  ((2 ^ (1 / 3) * 3 ^ (1 / 2)) ^ 6 + ((2 ^ (3 / 2)) ^ (1 / 2 * 4 / 3)) - 4 * ((4 / 7) ^ (2 * (-1 / 2))) -
   2 ^ (1 / 4) * 2 ^ (3 / 4) - 1)

theorem problem1 : expr1 = 100 :=
by
  sorry

noncomputable def expr2 : ℝ := 
  (log 2.5 6.25 + log10 0.01 + log (sqrt exp 1) - 2 ^ (1 + (log 2 3)))

theorem problem2 : expr2 = -11 / 2 :=
by
  sorry

end problem1_problem2_l786_786701


namespace value_of_m_l786_786403

theorem value_of_m :
  ∃ m : ℕ, 3 * 4 * 5 * m = fact 8 ∧ m = 672 :=
by
  use 672
  split
  sorry

end value_of_m_l786_786403


namespace hyperbola_eccentricity_is_sqrt2_l786_786895

noncomputable def eccentricity_of_hyperbola {a b : ℝ} (h : a ≠ 0) (hb : b = a) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (c / a)

theorem hyperbola_eccentricity_is_sqrt2 {a : ℝ} (h : a ≠ 0) :
  eccentricity_of_hyperbola h (rfl) = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt2_l786_786895


namespace solve_for_z_l786_786766

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786766


namespace prism_volume_l786_786632

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) : a * b * c = 12 :=
by sorry

end prism_volume_l786_786632


namespace vector_subtraction_l786_786931

def vector_a : ℝ × ℝ := (3, 5)
def vector_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (7, 3) :=
sorry

end vector_subtraction_l786_786931


namespace BM_less_than_AC_l786_786501

open EuclideanGeometry

variable {A B C M : Point}

noncomputable def triangle_angle_A_eq_angle_B (A B C M : Point) : Prop :=
  ∠ A B M = ∠ B

noncomputable def angle_AMB_eq_100 (A B C M : Point) : Prop :=
  ∠ A M B = 100

noncomputable def angle_C_eq_70 (A B C M : Point) : Prop :=
  ∠ C = 70

theorem BM_less_than_AC {A B C M : Point} (h1 : triangle_angle_A_eq_angle_B A B C M)
                                                (h2 : angle_AMB_eq_100 A B C M)
                                                (h3 : angle_C_eq_70 A B C M) :
  length (segment B M) < length (segment A C) :=
  sorry

end BM_less_than_AC_l786_786501


namespace count_positive_integers_l786_786395

-- Define the conditions and their interpretations
def cond1 (x : ℕ) : Prop := 50 < x ∧ x < 70
def cond2 (x : ℕ) : Prop := (Real.log10 (x-50) + Real.log10 (70-x) ≤ 2)
def cond3 (x : ℕ) : Prop := (x^2 - 120 * x + 3500 ≤ 0)

-- Main proposition combining all conditions and the proof goal
theorem count_positive_integers :
  ∃! n : ℕ, n = 21 ∧ ∀ x : ℕ, cond1 x → cond2 x → cond3 x → ∃ pos_int_sol : ℕ, ∀ sol, sol = x ∧ pos_int_sol = 21 :=
begin
  sorry
end

end count_positive_integers_l786_786395


namespace range_of_a_l786_786468

open Real

theorem range_of_a (a : ℝ) (log_condition : log a (2 / 3) < 1) (a_pos : 0 < a) (a_ne_one : a ≠ 1) : 
  a ∈ (Set.Ioo 0 (2 / 3) ∪ Set.Ioi 1) :=
sorry

end range_of_a_l786_786468


namespace new_tax_rate_l786_786664

-- Condition definitions
def previous_tax_rate : ℝ := 0.20
def initial_income : ℝ := 1000000
def new_income : ℝ := 1500000
def additional_taxes_paid : ℝ := 250000

-- Theorem statement
theorem new_tax_rate : 
  ∃ T : ℝ, 
    (new_income * T = initial_income * previous_tax_rate + additional_taxes_paid) ∧ 
    T = 0.30 :=
by sorry

end new_tax_rate_l786_786664


namespace tutors_schedule_l786_786365

theorem tutors_schedule :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end tutors_schedule_l786_786365


namespace sequence_count_is_13_l786_786752

theorem sequence_count_is_13 :
  ∃ (s : Fin 2022 → ℕ), 
  (∀ i, s (⟨i + 1, lt_trans i.2⟩) ≥ s i) ∧
  (∃ i, s i = 2022) ∧ 
  (∀ i j, (∑ k in Finset.range 2022, s k) - s i - s j) % (s i) = 0 ∧ 
                      ((∑ k in Finset.range 2022, s k) - s i - s j) % (s j) = 0 := 
sorry

end sequence_count_is_13_l786_786752


namespace michael_card_count_l786_786537

variable (Lloyd Mark Michael : ℕ)
variable (L : ℕ)

-- Conditions from the problem
axiom condition1 : Mark = 3 * Lloyd
axiom condition2 : Mark + 10 = Michael
axiom condition3 : Lloyd + Mark + (Michael + 80) = 300

-- The correct answer we want to prove
theorem michael_card_count : Michael = 100 :=
by
  -- Proof will be here.
  sorry

end michael_card_count_l786_786537


namespace number_of_ways_to_choose_reading_materials_l786_786239

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786239


namespace find_p_l786_786069

open Complex

theorem find_p (q p : ℝ) (h_root : (4 + I : ℂ) is_root) : p = 51 :=
sorry

end find_p_l786_786069


namespace largest_initial_number_l786_786025

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786025


namespace total_surface_area_of_solid_is_11_l786_786313

def piece_heights : ℕ → ℝ
| 0 => 1/2
| 1 => 1/3
| 2 => 1/17
| _ => 1 - (1/2 + 1/3 + 1/17)

def total_surface_area (height_A height_B height_C height_D : ℝ) : ℝ :=
  let top_bottom_area := 4 * 1 -- top and bottom surfaces area
  let side_area := 2 * 1 -- side surfaces area
  let front_back_area := 1 * 1 -- front and back surfaces area
  top_bottom_area + side_area + front_back_area

theorem total_surface_area_of_solid_is_11 :
  total_surface_area (piece_heights 0) (piece_heights 1) (piece_heights 2) (piece_heights 3) = 11 := by
  sorry

end total_surface_area_of_solid_is_11_l786_786313


namespace two_students_one_common_material_l786_786194

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786194


namespace sum_of_coordinates_l786_786169

-- Defining the conditions of the problem
def is_distance_from_line (p : ℝ × ℝ) (line_y : ℝ) (distance : ℝ) : Prop := 
  abs (p.2 - line_y) = distance

def is_distance_from_point (p : ℝ × ℝ) (q : ℝ × ℝ) (distance : ℝ) : Prop := 
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) = distance

-- Four points needed
def points : list (ℝ × ℝ) := [
  (5 + 2 * real.sqrt 21, 6),
  (5 - 2 * real.sqrt 21, 6),
  (5 + 2 * real.sqrt 21, 14),
  (5 - 2 * real.sqrt 21, 14)
]

-- Statement to prove that the sum of the coordinates is 60
theorem sum_of_coordinates : 
  let p := (5, 10) in
  ∑ pt in points, pt.1 + pt.2 = 60 := 
sorry

end sum_of_coordinates_l786_786169


namespace sum_of_distinct_products_of_6_23H_508_3G4_l786_786603

theorem sum_of_distinct_products_of_6_23H_508_3G4 (G H : ℕ) : 
  (G < 10) → (H < 10) →
  (623 * 1000 + H * 100 + 508 * 10 + 3 * 10 + G * 1 + 4) % 72 = 0 →
  (if G = 0 then 0 + if G = 4 then 4 else 0 else 0) = 4 :=
by
  intros
  sorry

end sum_of_distinct_products_of_6_23H_508_3G4_l786_786603


namespace smallest_four_digit_divisible_by_6_l786_786261

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_divisible_by_6_l786_786261


namespace find_z_l786_786833

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786833


namespace problem_statement_l786_786557

noncomputable def martingale {Ω : Type*} {n : ℕ} 
  (ξ : Fin (n+1) → Ω → ℝ) (ℱ : Fin (n+1) → Ω → Set (Set Ω)) : Prop :=
∀ k (hk : k ≤ n), ξ k = (∫ ξ (n) ∂(ℱ k))

def stopping_time {Ω : Type*} {n : ℕ} 
  (τ : Ω → Fin (n+1)) (ℱ : Fin (n+1) → Ω → Set (Set Ω)) : Prop :=
∃ k, ∀ ω, (τ ω = k) → ℱ k ω = ℱ k ω

theorem problem_statement (Ω : Type*) {n : ℕ} 
  (ξ : Fin (n+1) → Ω → ℝ) 
  (ℱ : Fin (n+1) → Ω → Set (Set Ω)) 
  (τ : Ω → Fin (n+1)) 
  [Mart : martingale ξ ℱ] 
  [StopTime : stopping_time τ ℱ]
  (k : Fin (n+1)) (hk : k ≤ n) :
  (∫ (λ ω, ξ n ω * (τ ω = k)) ∂(ℱ n)) = (∫ (λ ω, ξ k ω * (τ ω = k)) ∂(ℱ k)) :=
sorry

end problem_statement_l786_786557


namespace total_number_of_turtles_l786_786178

variable {T : Type} -- Define a variable for the type of turtles

-- Define the conditions as hypotheses
variable (total_turtles : ℕ)
variable (female_percentage : ℚ) (male_percentage : ℚ)
variable (striped_male_prop : ℚ)
variable (baby_striped_males : ℕ) (adult_striped_males_prop : ℚ)
variable (striped_male_percentage : ℚ)
variable (striped_males : ℕ)
variable (male_turtles : ℕ)

-- Condition definitions
def female_percentage_def := female_percentage = 60 / 100
def male_percentage_def := male_percentage = 1 - female_percentage
def striped_male_prop_def := striped_male_prop = 1 / 4
def adult_striped_males_prop_def := adult_striped_males_prop = 60 / 100
def baby_and_adult_striped_males_prop_def := (1 - adult_striped_males_prop) = 40 / 100
def striped_males_def := striped_males = baby_striped_males / (1 - adult_striped_males_prop)
def male_turtles_def := male_turtles = striped_males / striped_male_prop
def male_turtles_percentage_def := male_turtles = total_turtles * (1 - female_percentage)

-- The proof statement to show the total number of turtles is 100
theorem total_number_of_turtles (h_female : female_percentage_def)
                                (h_male : male_percentage_def)
                                (h_striped_male_prop : striped_male_prop_def)
                                (h_adult_striped_males_prop : adult_striped_males_prop_def)
                                (h_baby_and_adult_striped_males_prop : baby_and_adult_striped_males_prop_def)
                                (h_striped_males : striped_males_def)
                                (h_male_turtles : male_turtles_def)
                                (h_male_turtles_percentage : male_turtles_percentage_def):
  total_turtles = 100 := 
by sorry

end total_number_of_turtles_l786_786178


namespace first_digits_of_powers_of_2_non_periodic_l786_786985

theorem first_digits_of_powers_of_2_non_periodic :
  ¬ ∃ d, ∀ n : ℕ, (first_digit (2 ^ n) = first_digit (2 ^ (n + d))) :=
sorry

end first_digits_of_powers_of_2_non_periodic_l786_786985


namespace solve_z_l786_786847

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786847


namespace largest_initial_number_l786_786039

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786039


namespace find_N_l786_786635

def f (N : ℕ) : ℕ :=
  if N % 2 = 0 then 5 * N else 3 * N + 2

theorem find_N (N : ℕ) :
  f (f (f (f (f N)))) = 542 ↔ N = 112500 := by
  sorry

end find_N_l786_786635


namespace num_ways_choose_materials_l786_786208

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786208


namespace wolf_hungry_if_eating_11_hares_l786_786479

variables {Q l c : ℝ}
def food_quantity (pigs hares : ℝ) := pigs * c + hares * l

theorem wolf_hungry_if_eating_11_hares
  (h1 : food_quantity 3 7 < Q)
  (h2 : food_quantity 7 1 > Q) :
  food_quantity 0 11 < Q :=
by {
  have h3 : 3 * c + 7 * l < 7 * c + l, from (lt_trans h1 h2),
  have h4 : 6 * l < 4 * c, from calc
    3 * c + 7 * l < 7 * c + l : h3
    ... - 3 * c - l < 7 * c + l - 3 * c - l: sub_lt_sub_left (lt_of_le_of_lt (le_refl _) h2),
  have h5 : 3 * l < 2 * c, from div_lt_div_of_mul_lt_mul_left h4 zero_lt_two (by norm_num),
  have h6 : 4 * l < 3 * c, from calc
    4 * l = 2 * l + 2 * l: by norm_num
    ... < c + c : add_lt_add h5 h5
    ... = 2 * c : by norm_num,
  have h7 : 11 * l < Q, from calc
    11 * l = 3 * l + 7 * l + l: by ring
    ... < 3 * l + 4 * c: add_lt_add_left h6 3 * l
    ... < 3 * l + 3 * c + l: add_lt_add h6 h5
    ... = Q: by ring,
  exact h7,
 sorry
}

end wolf_hungry_if_eating_11_hares_l786_786479


namespace part_one_part_two_l786_786921

variable {x m : ℝ}

theorem part_one (h1 : ∀ x : ℝ, ¬(m * x^2 - (m + 1) * x + (m + 1) ≥ 0)) : m < -1 := sorry

theorem part_two (h2 : ∀ x : ℝ, 1 < x → m * x^2 - (m + 1) * x + (m + 1) ≥ 0) : m ≥ 1 / 3 := sorry

end part_one_part_two_l786_786921


namespace total_arrangements_l786_786363

-- Definitions according to the conditions
def num_female_teachers := 2
def num_male_teachers := 4
def num_females_per_group := 1
def num_males_per_group := 2

-- The goal is to prove the total number of different arrangements
theorem total_arrangements : 
  (nat.choose num_female_teachers num_females_per_group) * 
  (nat.choose num_male_teachers (2 * num_males_per_group)) = 12 := 
by
  -- Calculation steps should go here, but we skip the proof with sorry
  sorry

end total_arrangements_l786_786363


namespace compound_interest_time_l786_786324

theorem compound_interest_time 
  (P : ℝ) (r : ℝ) (A₁ : ℝ) (A₂ : ℝ) (t₁ t₂ : ℕ)
  (h1 : r = 0.10)
  (h2 : A₁ = P * (1 + r) ^ t₁)
  (h3 : A₂ = P * (1 + r) ^ t₂)
  (h4 : A₁ = 2420)
  (h5 : A₂ = 2662)
  (h6 : t₂ = t₁ + 3) :
  t₁ = 3 := 
sorry

end compound_interest_time_l786_786324


namespace max_distance_of_line_with_slope_1_intersecting_ellipse_l786_786306

theorem max_distance_of_line_with_slope_1_intersecting_ellipse :
  (∀ l : Affine.Line ℝ, l.slope = 1 → ∃ A B : ℝ × ℝ,
    (A ≠ B ∧ (A.1^2 / 4 + A.2^2 = 1) ∧ (B.1^2 / 4 + B.2^2 = 1) ∧
      ∃ t : ℝ, A.2 = A.1 + t ∧ B.2 = B.1 + t)) →
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1^2 / 4 + A.2^2 = 1) ∧ (B.1^2 / 4 + B.2^2 = 1) →
    ∃ t : ℝ, A.2 = A.1 + t ∧ B.2 = B.1 + t ∧
    |√((B.1 - A.1)^2 + (B.2 - A.2)^2)| = 4 * sqrt 10 / 5) :=
by
  sorry

end max_distance_of_line_with_slope_1_intersecting_ellipse_l786_786306


namespace area_of_closed_figure_l786_786249

noncomputable def f (x : ℝ) : ℝ :=
  if x ^ 2 < sqrt x then sqrt x else x ^ 2

theorem area_of_closed_figure :
  ∫ x in (1 / 4 : ℝ)..1, sqrt x + ∫ x in (1 : ℝ)..2, x^2 = (35 / 12 : ℝ) :=
by
  sorry

end area_of_closed_figure_l786_786249


namespace lagrange_intermediate_value_l786_786595

open Set

variable {a b : ℝ} (f : ℝ → ℝ)

-- Ensure that a < b for the interval [a, b]
axiom hab : a < b

-- Assume f is differentiable on [a, b]
axiom differentiable_on_I : DifferentiableOn ℝ f (Icc a b)

theorem lagrange_intermediate_value :
  ∃ (x0 : ℝ), x0 ∈ Ioo a b ∧ (deriv f x0) = (f a - f b) / (a - b) :=
sorry

end lagrange_intermediate_value_l786_786595


namespace unique_real_value_for_equal_roots_l786_786751

theorem unique_real_value_for_equal_roots :
  ∃! p : ℝ, ∀ x : ℝ, x^2 - p * x + p^2 = 0 → discriminant x^2 - p * x + p^2 = 0 :=
by {
  sorry
}

end unique_real_value_for_equal_roots_l786_786751


namespace smallest_palindrome_base2_base4_l786_786314

/-- A palindrome is a number that reads the same forward and backward -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = List.reverse digits

/-- The smallest 5-digit palindrome in base 2 can be expressed as a 3-digit palindrome in base 4 -/
theorem smallest_palindrome_base2_base4 :
  ∃ n : ℕ, is_palindrome n 2 ∧ n < 2^5 ∧ n ≥ 2^4 ∧ is_palindrome n 4 ∧ n = 17 :=
by
  exists 17
  sorry

end smallest_palindrome_base2_base4_l786_786314


namespace find_z_l786_786799

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786799


namespace second_derivative_eq_l786_786912

variable (q r : ℝ → ℝ)
variable (q' r' : ℝ → ℝ)

axiom q'_def : ∀ t, q' t = 3 * q t - 3
axiom r'_def: ∀ t, r' t = 2 * r t + 1

theorem second_derivative_eq (t : ℝ) :
  (derivative (derivative (λ t, 7 * q t + r t))) t = 63 * q t + 4 * r t - 61 :=
by
  sorry

end second_derivative_eq_l786_786912


namespace coefficient_x5_in_expansion_l786_786979

theorem coefficient_x5_in_expansion : 
  (coeff (∑ i in range (11), (1 - x) * (x ^ i) * (finset.choose 10 i)) 5) = 42 :=
by
  sorry

end coefficient_x5_in_expansion_l786_786979


namespace find_value_in_sequence_l786_786612

theorem find_value_in_sequence 
: ∀ (a b c d x e f g : ℕ),
    a = 1 ∧ b = 1 ∧ c = a + b ∧ d = b + c ∧ e = x + d ∧ f = e + x + d ∧ g = f + e ∧
    c = 2 ∧ d = 3 ∧ e = 8 ∧ f = 13 ∧ g = 21 → x = 5 := 
by
  intros a b c d x e f g
  intro h
  have ha := h.1
  have hb := h.2.1
  have hc := h.2.2.1
  have hd := h.2.2.2.1
  have he := h.2.2.2.2.1
  have hf := h.2.2.2.2.2.1
  have hg := h.2.2.2.2.2.2
  have hcEq := h.2.2.2.2.2.2.1
  have hdEq := h.2.2.2.2.2.2.2.1
  have heEq := h.2.2.2.2.2.2.2.2.1
  have hfEq := h.2.2.2.2.2.2.2.2.2.1
  have hgEq := h.2.2.2.2.2.2.2.2.2.2
  rw [hcEq, hdEq] at he
  sorry

end find_value_in_sequence_l786_786612


namespace solve_z_l786_786848

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786848


namespace adamek_marbles_l786_786685

theorem adamek_marbles : ∃ n : ℕ, (∀ k : ℕ, n = 4 * k ∧ n = 3 * (k + 8)) → n = 96 :=
by
  sorry

end adamek_marbles_l786_786685


namespace inequality_proof_l786_786878

noncomputable def a := (1 / 4) * Real.logb 2 3
noncomputable def b := 1 / 2
noncomputable def c := (1 / 2) * Real.logb 5 3

theorem inequality_proof : c < a ∧ a < b :=
by
  sorry

end inequality_proof_l786_786878


namespace det_transformation_l786_786081

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (D : ℝ)

def det_column_matrix (a b c : V) : ℝ :=
  (a • b × c)

def det_new_matrix (a b c : V) : ℝ :=
  (a - b) • ((b - c) × (c - a))

theorem det_transformation (hD : D = det_column_matrix a b c) :
  det_new_matrix a b c = 2 * D :=
  sorry

end det_transformation_l786_786081


namespace negative_product_probability_l786_786642

noncomputable def m : Set ℤ := {-9, -7, -5, -3, 0, 5, 7}
noncomputable def t : Set ℤ := {-8, -6, -2, 0, 3, 4, 6, 7}

theorem negative_product_probability : 
  let total_ways := finite.to_finset m.card * finite.to_finset t.card in
  let neg_m_pos_t := (finite.to_finset (m.filter (λ x => x < 0))).card * (finite.to_finset (t.filter (λ x => x > 0))).card in
  let pos_m_neg_t := (finite.to_finset (m.filter (λ x => x > 0))).card * (finite.to_finset (t.filter (λ x => x < 0))).card in
  let neg_product_ways := neg_m_pos_t + pos_m_neg_t in
  (neg_product_ways : ℚ) / total_ways = 11 / 28 :=
by
  sorry

end negative_product_probability_l786_786642


namespace draw_white_ball_is_impossible_l786_786591

-- Definitions based on the conditions
def redBalls : Nat := 2
def blackBalls : Nat := 6
def totalBalls : Nat := redBalls + blackBalls

-- Definition for the white ball drawing event
def whiteBallDraw (redBalls blackBalls : Nat) : Prop :=
  ∀ (n : Nat), n ≠ 0 → n ≤ redBalls + blackBalls → false

-- Theorem to prove the event is impossible
theorem draw_white_ball_is_impossible : whiteBallDraw redBalls blackBalls :=
  by
  sorry

end draw_white_ball_is_impossible_l786_786591


namespace germs_per_dish_l786_786975

theorem germs_per_dish:
  let total_germs : ℝ := 0.037 * 10^5
  let total_petri_dishes : ℝ := 148000 * 10^(-3)
  let germs_per_dish : ℝ := total_germs / total_petri_dishes
  germs_per_dish ≈ 25 :=
by
  sorry

end germs_per_dish_l786_786975


namespace anton_food_cost_l786_786077

def food_cost_julie : ℝ := 10
def food_cost_letitia : ℝ := 20
def tip_per_person : ℝ := 4
def num_people : ℕ := 3
def tip_percentage : ℝ := 0.20

theorem anton_food_cost (A : ℝ) :
  tip_percentage * (food_cost_julie + food_cost_letitia + A) = tip_per_person * num_people →
  A = 30 :=
by
  intro h
  sorry

end anton_food_cost_l786_786077


namespace find_point_B_l786_786318

noncomputable def point : Type := ℝ × ℝ × ℝ

def A : point := (-2, 8, 12)
def C : point := (4, 6, 7)

def plane_eq (p : point) : ℝ := 2 * p.1 - p.2 + 3 * p.3 - 25

def point_B (B : point) : Prop :=
  B = (38/9, 55/9, 98/9) ∧
  plane_eq B = 0 ∧
  ∃ t, (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2), A.3 + t * (C.3 - A.3)) = B

theorem find_point_B : ∃ B : point, point_B B :=
  sorry

end find_point_B_l786_786318


namespace find_other_cat_weight_l786_786347

variable (cat1 cat2 dog : ℕ)

def weight_of_other_cat (cat1 cat2 dog : ℕ) : Prop :=
  cat1 = 7 ∧
  dog = 34 ∧
  dog = 2 * (cat1 + cat2) ∧
  cat2 = 10

theorem find_other_cat_weight (cat1 : ℕ) (cat2 : ℕ) (dog : ℕ) :
  weight_of_other_cat cat1 cat2 dog := by
  sorry

end find_other_cat_weight_l786_786347


namespace xn_correct_sn_correct_l786_786401

noncomputable def x_n (x_0 : ℝ) : ℕ → ℝ
| 0 => x_0
| n + 1 => x_0 * (4^(n) / 3^(n+1))

def S_n (x_0 : ℝ) : ℕ → ℝ
| 0 => x_0
| n + 1 => x_0 * (4 / 3)^(n + 1)

lemma sequence_relation (x_0 : ℝ) (n : ℕ) :
  n > 0 → 2 * x_n x_0 n = list.sum (list.map (x_n x_0) (list.range n)) - x_n x_0 n :=
sorry

theorem xn_correct (x_0 : ℝ) (n : ℕ) :
  n > 0 → x_n x_0 n = x_0 * 4^(n-1) / 3^n :=
sorry

theorem sn_correct (x_0 : ℝ) (n : ℕ) :
  S_n x_0 n = x_0 * (4 / 3)^n :=
sorry

end xn_correct_sn_correct_l786_786401


namespace solve_for_z_l786_786856

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786856


namespace area_enclosed_by_graph_l786_786143

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0

theorem area_enclosed_by_graph :
  ∫ x in 0..1, f x + ∫ x in 1..2, f x = 5 / 6 :=
by
  sorry

end area_enclosed_by_graph_l786_786143


namespace solve_for_z_l786_786866

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786866


namespace perpendicular_vectors_l786_786905

open Real EuclideanGeometry

theorem perpendicular_vectors (AB AC AP BC : ℝ^3) (λ : ℝ)
  (h₁ : ∠AB AC = π / 3)
  (h₂ : ‖AB‖ = 2)
  (h₃ : ‖AC‖ = 4)
  (h₄ : AP = AB + λ • AC)
  (h₅ : AP ⬝ BC = 0) :
  λ = 1 / 6 :=
sorry

end perpendicular_vectors_l786_786905


namespace chosen_number_l786_786322

theorem chosen_number (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 :=
sorry

end chosen_number_l786_786322


namespace min_value_l786_786424

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 1/(a-1) + 4/(b-1) ≥ 4 :=
by
  sorry

end min_value_l786_786424


namespace minimum_difference_l786_786960

def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem minimum_difference (x y z : ℤ) 
  (hx : even x) (hy : odd y) (hz : odd z)
  (hxy : x < y) (hyz : y < z) (hzx : z - x = 9) : y - x = 1 := 
sorry

end minimum_difference_l786_786960


namespace total_games_played_is_53_l786_786516

theorem total_games_played_is_53 :
  ∃ (ken_wins dave_wins jerry_wins larry_wins total_ties total_games_played : ℕ),
  jerry_wins = 7 ∧
  dave_wins = jerry_wins + 3 ∧
  ken_wins = dave_wins + 5 ∧
  larry_wins = 2 * jerry_wins ∧
  5 ≤ ken_wins ∧ 5 ≤ dave_wins ∧ 5 ≤ jerry_wins ∧ 5 ≤ larry_wins ∧
  total_ties = jerry_wins ∧
  total_games_played = ken_wins + dave_wins + jerry_wins + larry_wins + total_ties ∧
  total_games_played = 53 :=
by
  sorry

end total_games_played_is_53_l786_786516


namespace two_students_choose_materials_l786_786184

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786184


namespace intersection_of_M_and_N_l786_786082

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | x < 1}
def expected_intersection : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N :
  M ∩ N = expected_intersection :=
sorry

end intersection_of_M_and_N_l786_786082


namespace sum_log_floor_not_equal_91_l786_786757

def floor_log (x : Real) : Int := Int.floor (Real.log10 x)

theorem sum_log_floor_not_equal_91 (n : ℕ) (h : n > 0) : 
  (Finset.sum (Finset.range n.succ) (λ x => floor_log (↑x + 2))) ≠ 91 :=
sorry

end sum_log_floor_not_equal_91_l786_786757


namespace smallest_angle_in_20_sided_polygon_is_143_l786_786580

theorem smallest_angle_in_20_sided_polygon_is_143
  (n : ℕ)
  (h_n : n = 20)
  (angles : ℕ → ℕ)
  (h_convex : ∀ i, 1 ≤ i → i ≤ n → angles i < 180)
  (h_arithmetic_seq : ∃ d : ℕ, ∀ i, 1 ≤ i → i < n → angles (i + 1) = angles i + d)
  (h_increasing : ∀ i, 1 ≤ i → i < n → angles (i + 1) > angles i)
  (h_sum : ∑ i in finset.range n, angles (i + 1) = (n - 2) * 180) :
  angles 1 = 143 :=
by
  sorry

end smallest_angle_in_20_sided_polygon_is_143_l786_786580


namespace AM_eq_AN_l786_786335

noncomputable theory

variables {A B C D E F G M N : Type*}
variables [metric_space A] [metric_space B] [metric_space C]
variables (acute_triangle : ∀ (A B C : A), A ≠ 60) 
variables (tangents : ∀ (A B C : A), BD = CE = BC)
variables (lineDE : ∀ (A B C D E F G : A), (DE ∩ extensionAB) = F ∧ (DE ∩ extensionAC) = G)
variables (intersectionCF : ∀ (A B C F M : A), (CF ∩ BD) = M)
variables (intersectionCE : ∀ (A B C G N : A), (CE ∩ BG) = N)
variables (AM AN : A)

theorem AM_eq_AN : AM = AN :=
sorry

end AM_eq_AN_l786_786335


namespace arthur_amount_left_l786_786694

def initial_amount : ℝ := 200
def fraction_spent : ℝ := 4 / 5

def spent (initial : ℝ) (fraction : ℝ) : ℝ := fraction * initial

def amount_left (initial : ℝ) (spent_amount : ℝ) : ℝ := initial - spent_amount

theorem arthur_amount_left : amount_left initial_amount (spent initial_amount fraction_spent) = 40 := 
by
  sorry

end arthur_amount_left_l786_786694


namespace interleave_sequence_count_l786_786535

theorem interleave_sequence_count 
  (n1 n2 n3 : ℕ) : 
  ∃ (count : ℕ), count = ((n1 + n2 + n3)!.div (n1! * n2! * n3!)) :=
by
  sorry

end interleave_sequence_count_l786_786535


namespace two_students_choose_materials_l786_786191

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786191


namespace three_digit_max_l786_786627

theorem three_digit_max (n : ℕ) : 
  n % 9 = 1 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ 100 <= n ∧ n <= 999 → n = 793 :=
by
  sorry

end three_digit_max_l786_786627


namespace sum_of_digits_B_equals_4_l786_786091

theorem sum_of_digits_B_equals_4 (A B : ℕ) (N : ℕ) (hN : N = 4444 ^ 4444)
    (hA : A = (N.digits 10).sum) (hB : B = (A.digits 10).sum) :
    (B.digits 10).sum = 4 := by
  sorry

end sum_of_digits_B_equals_4_l786_786091


namespace total_opponent_runs_l786_786656

-- Define the scores in each game.
def scores_team : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the number of games lost by one run.
def num_games_lost_by_one_run : ℕ := 6

-- Define the scores difference for losing by one run.
def lost_by_one_run_difference : ℕ := 1

-- Define the function to calculate opponent's score when the team loses by one run.
def opponent_score_lost_by_one (team_score : ℕ) : ℕ := team_score + lost_by_one_run_difference

-- Define the function to calculate opponent's score when the team scores twice.
def opponent_score_scored_twice (team_score : ℕ) : ℕ := team_score / 2

-- Prove that total runs scored by opponents is 63.
theorem total_opponent_runs : List.sum (scores_team.map (λ team_score, 
  if team_score % 2 = 0 then opponent_score_lost_by_one team_score else opponent_score_scored_twice team_score)) = 63 :=
by
  sorry

end total_opponent_runs_l786_786656


namespace find_coordinates_of_B_find_equation_of_BC_l786_786016

-- Problem 1: Prove that the coordinates of B are (10, 5)
theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0) :
  B = (10, 5) :=
sorry

-- Problem 2: Prove that the equation of line BC is 2x + 9y - 65 = 0
theorem find_equation_of_BC (A B C : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0)
  (coordinates_B : B = (10, 5)) :
  ∃ k : ℝ, ∀ P : ℝ × ℝ, (P.1 - C.1) / (P.2 - C.2) = k → 2 * P.1 + 9 * P.2 - 65 = 0 :=
sorry

end find_coordinates_of_B_find_equation_of_BC_l786_786016


namespace sum_of_squares_of_root_pairs_eq_400_l786_786531

theorem sum_of_squares_of_root_pairs_eq_400
  (p q r : ℝ)
  (h : polynomial.eval p (polynomial.C 1 * polynomial.X^3 - polynomial.C 15 * polynomial.X^2 + polynomial.C 25 * polynomial.X - polynomial.C 10) = 0)
  (hq : polynomial.eval q (polynomial.C 1 * polynomial.X^3 - polynomial.C 15 * polynomial.X^2 + polynomial.C 25 * polynomial.X - polynomial.C 10) = 0)
  (hr : polynomial.eval r (polynomial.C 1 * polynomial.X^3 - polynomial.C 15 * polynomial.X^2 + polynomial.C 25 * polynomial.X - polynomial.C 10) = 0) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 := 
sorry

end sum_of_squares_of_root_pairs_eq_400_l786_786531


namespace xn_correct_sn_correct_l786_786400

noncomputable def x_n (x_0 : ℝ) : ℕ → ℝ
| 0 => x_0
| n + 1 => x_0 * (4^(n) / 3^(n+1))

def S_n (x_0 : ℝ) : ℕ → ℝ
| 0 => x_0
| n + 1 => x_0 * (4 / 3)^(n + 1)

lemma sequence_relation (x_0 : ℝ) (n : ℕ) :
  n > 0 → 2 * x_n x_0 n = list.sum (list.map (x_n x_0) (list.range n)) - x_n x_0 n :=
sorry

theorem xn_correct (x_0 : ℝ) (n : ℕ) :
  n > 0 → x_n x_0 n = x_0 * 4^(n-1) / 3^n :=
sorry

theorem sn_correct (x_0 : ℝ) (n : ℕ) :
  S_n x_0 n = x_0 * (4 / 3)^n :=
sorry

end xn_correct_sn_correct_l786_786400


namespace members_play_both_l786_786969

variable (N B T E' : ℕ)
variables (hN : N = 50) (hB : B = 25) (hT : T = 32) (hE' : E' = 5)

theorem members_play_both (h : B + T - 12 = N - E') : B + T - (N - E') = 12 := 
by
  rw [hN, hB, hT, hE'] at h 
  exact h

#check members_play_both

end members_play_both_l786_786969


namespace unique_real_solution_k_l786_786761

theorem unique_real_solution_k (k : ℝ) :
  (∃ x : ℝ, (3 * x + 7) * (x - 5) = -27 + k * x) ∧ 
  ∀ x1 x2 : ℝ, 
    ( (3 * x1 + 7) * (x1 - 5) = -27 + k * x1 ∧ 
      (3 * x2 + 7) * (x2 - 5) = -27 + k * x2 ) → x1 = x2 
  ↔ k = -8 + 4 * real.sqrt 6 ∨ k = -8 - 4 * real.sqrt 6 :=
by
  sorry

end unique_real_solution_k_l786_786761


namespace number_of_ways_l786_786219

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786219


namespace value_of_k_l786_786154

theorem value_of_k : ∃ (k : ℤ), (2 * (-5) + 20 = 10) ∧ (-3 * (-5) + 20 = k) ∧ (k = 35) :=
by
  use 35
  split
  { norm_num }
  { split
    { norm_num }
    { refl }
  }

end value_of_k_l786_786154


namespace trapezoid_perimeter_l786_786575

theorem trapezoid_perimeter (a b : ℝ) (h : ∃ c : ℝ, a * b = c^2) :
  ∃ K : ℝ, K = 2 * (a + b + Real.sqrt (a * b)) :=
by
  sorry

end trapezoid_perimeter_l786_786575


namespace find_x_l786_786110

-- Definitions for positions
def A := sorry
def B := sorry
def C := sorry
def D := 1
def E := 4
def F := sorry
def G := sorry
def H := sorry
def I := 9
def J := sorry

-- x is the number to place in the circle marked with "⭑"
def x := sorry

-- The sum of the three vertices for each of the 7 triangles must equal 15
axiom triangle_sum1 : A + B + C = 15
axiom triangle_sum2 : A + D + E = 15
axiom triangle_sum3 : B + F + G = 15
axiom triangle_sum4 : C + E + H = 15
axiom triangle_sum5 : D + G + H = 15
axiom triangle_sum6 : E + F + I = 15
axiom triangle_sum7 : F + H + J = 15

-- We need to prove that x (marked as ⭑) is 7
theorem find_x : x = 7 :=
  by sorry

end find_x_l786_786110


namespace solve_for_z_l786_786854

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786854


namespace number_of_descending_digit_numbers_l786_786747

theorem number_of_descending_digit_numbers : 
  (∑ k in Finset.range 8, Nat.choose 10 (k + 2)) + 1 = 1013 :=
by
  sorry

end number_of_descending_digit_numbers_l786_786747


namespace triangle_ABC_properties_l786_786755

theorem triangle_ABC_properties
  (A B C : ℝ) (a b c : ℝ)
  (h1 : sin B = sin A * cos C + cos A * sin C + sin A * cos C + (1 / 2) * sin (2 * A))
  (h2 : B = 2 + (4 / sqrt 3) * sin (C + π / 3))
  (h3 : 0 < C ∧ C < 2 * π / 3)
  (h4 : a + b + c = a + (4 / sqrt 3) * (sin C + sin B))
  (h5 : A = π / 3)
  (h6 : sin (C + π / 6) ≤ 1)
  (h7 : a = 24 * sin (C + π / 6))
  (h8 : a = 2 + (4 / sqrt 3) * (sin C + (sqrt 3 / 2) * cos C + (1 / 2) * sin C))
  (h9 : C = π / 3 ∧ b = 2 ∧ c = 2) :
  A = π / 3 ∧ max a (max b c) = 6 ∧ b = 2 ∧ c = 2 :=
sorry

end triangle_ABC_properties_l786_786755


namespace part1_part2i_part2ii_l786_786882

noncomputable def f (x k : ℝ) : ℝ := |x^2 - 1| + x^2 + k * x

theorem part1 (k : ℝ) (x : ℝ) (h_k : k = 2) (h_x : x ∈ Iic (-1)) :
  f x k = 0 ↔ x = (-1 - Real.sqrt 3) / 2 :=
begin
  sorry
end

theorem part2i (x1 x2 k : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < 2) (h4 : x2 < 2) (hx1 : f x1 k = 0) (hx2 : f x2 k = 0) :
  k ∈ Ioo (-7 / 2) (-1) :=
begin
  sorry
end

theorem part2ii (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < 2) (h4 : x2 < 2) (hx1 : f x1 (-1 / x1) = 0) (hx2 : f x2 (1 / x2 - 2 * x2) = 0) :
  (1 / x1 + 1 / x2) < 4 :=
begin
  sorry
end

end part1_part2i_part2ii_l786_786882


namespace solve_z_l786_786846

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786846


namespace triangle_inequality_l786_786153

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c R : ℝ) : ℝ := a * b * c / (4 * R)
noncomputable def inradius_area (a b c r : ℝ) : ℝ := semiperimeter a b c * r

theorem triangle_inequality (a b c R r : ℝ) (h₁ : a ≤ 1) (h₂ : b ≤ 1) (h₃ : c ≤ 1)
  (h₄ : area a b c R = semiperimeter a b c * r) : 
  semiperimeter a b c * (1 - 2 * R * r) ≥ 1 :=
by 
  -- Proof goes here
  sorry

end triangle_inequality_l786_786153


namespace BC_length_l786_786997

-- Definitions of lengths and conditions
variables (A B C E F G : Type*)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E] [MetricSpace F] [MetricSpace G]

-- Given conditions: lengths of sides and particular points
axiom h1 : dist A B = 20
axiom h2 : dist A C = 18
axiom h3 : dist A E = 8
axiom h4 : dist A F = 8

-- Quadrilateral AEGF is cyclic
axiom cyclic_AEGF : CyclicQuad A E G F

-- Proof problem to be solved
theorem BC_length :
  ∃ (m n : ℕ), BC = m * Real.sqrt n ∧ ¬ ∃ p : ℕ, p^2 ∣ n ∧ 100 * m + n = 305
:= sorry

end BC_length_l786_786997


namespace andy_location_after_10_moves_l786_786693

theorem andy_location_after_10_moves :
  let start_point := (10, -10)
  let initial_direction := (0, 1) -- north
  let movement_pattern := λ (step : ℕ), 2 * (step + 1)
  let right_turn := λ (dir : ℤ × ℤ), (-dir.2, dir.1)
  let move := λ (pos : ℤ × ℤ) (dir : ℤ × ℤ) (distance : ℤ), (pos.1 + dir.1 * distance, pos.2 + dir.2 * distance)
  let final_position := (22, 0)
  ∀ (n : ℕ) (pos dir : ℤ × ℤ), 
    n = 10 →
    (∀ k < n, 
      let distance := movement_pattern k in
      let new_dir := right_turn (List.foldl (λ d _ , right_turn d) initial_direction (List.range k)) in
      pos = List.foldl (λ p i, move p (right_turn (List.foldl (λ d _ , right_turn d) initial_direction (List.range i))) (movement_pattern i)) start_point (List.range (k+1))) →
    pos = final_position :=
by
  intros n pos dir hn move_correct
  -- Proof would go here
  sorry

end andy_location_after_10_moves_l786_786693


namespace geometric_sequence_formula_sequence_b_formula_l786_786886

-- Problem (Ⅰ)
variables {a_n : ℕ → ℝ} {q : ℝ}

-- Conditions for Problem (Ⅰ)
def geometric_seq (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n * q

def sum_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, a_n (i + 1)

def arithmetic_seq (a1 a2 a3 : ℝ) : Prop :=
  2 * a2 = a1 + a3

-- Main statement for Problem (Ⅰ)
theorem geometric_sequence_formula (a_n : ℕ → ℝ) (q > 1) (S3 = 7) 
  (h1: sum_first_n_terms a_n 3 = 7)
  (h2: arithmetic_seq (a_n 0 + 3) (3 * a_n 1) (a_n 2 + 4))
  (ha1 : a_n 0 = 1)
  (hq : q = 2) :
  a_n = λ n, 2^n := by
  sorry

-- Problem (Ⅱ)
variables {b_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- Conditions for Problem (Ⅱ)
def sum_bn (T_n : ℕ → ℝ) (b_n : ℕ → ℝ) :=
  ∀ n : ℕ, 6 * T_n n = (3 * n + 1) * b_n n + 2

def b1_value (b_n : ℕ → ℝ) :=
  b_n 1 = 1

-- Main statement for Problem (Ⅱ)
theorem sequence_b_formula (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h_sum : sum_bn T_n b_n)
  (h_b1 : b1_value b_n) :
  b_n = λ n, 3 * n - 2 := by
  sorry

end geometric_sequence_formula_sequence_b_formula_l786_786886


namespace smallest_three_digit_multiple_of_17_l786_786630

theorem smallest_three_digit_multiple_of_17 : 
  ∃ (n : ℕ), (n ≥ 100 ∧ n < 1000) ∧ (n % 17 = 0) ∧ ∀ (m : ℕ), (m ≥ 100 ∧ m < 1000) ∧ (m % 17 = 0) → n ≤ m :=
begin
  use 102,
  split,
  { 
    split,
    { linarith, },
    { linarith, }
  },
  split,
  { 
    norm_num,
  },
  {
    intros m hm,
    cases hm with h1 h2,
    exact nat.le_of_dvd (nat.sub_pos_of_lt (h1.left)) h1.right,
  }
end

end smallest_three_digit_multiple_of_17_l786_786630


namespace product_of_abcd_l786_786638

noncomputable def a (c : ℚ) : ℚ := 33 * c + 16
noncomputable def b (c : ℚ) : ℚ := 8 * c + 4
noncomputable def d (c : ℚ) : ℚ := c + 1

theorem product_of_abcd :
  (2 * a c + 3 * b c + 5 * c + 8 * d c = 45) →
  (4 * (d c + c) = b c) →
  (4 * (b c) + c = a c) →
  (c + 1 = d c) →
  a c * b c * c * d c = ((1511 : ℚ) / 103) * ((332 : ℚ) / 103) * (-(7 : ℚ) / 103) * ((96 : ℚ) / 103) :=
by
  intros
  sorry

end product_of_abcd_l786_786638


namespace range_of_a_l786_786915

noncomputable theory
open_locale classical

def f (a x : ℝ) : ℝ := Real.exp x - x + a

theorem range_of_a (a : ℝ) : (∃ x : ℝ, f a x = 0) → a ≤ -1 :=
begin
  sorry
end

end range_of_a_l786_786915


namespace meghan_coffee_order_cost_l786_786071

def drip_coffee_cost (count : ℕ) (price : ℚ) (discount : ℚ) : ℚ :=
  let total_cost := count * price
  total_cost - (total_cost * discount)

def espresso_cost (price : ℚ) (tax : ℚ) : ℚ :=
  let tax_amount := price * tax
  price + tax_amount

def latte_cost (price : ℚ) (count : ℕ) (discount : ℚ) (syrup_price : ℚ) (syrup_tax : ℚ) : ℚ :=
  let first_lattee_cost := price
  let second_latte_cost := price * discount
  let syrup_cost := syrup_price + (syrup_price * syrup_tax)
  first_lattee_cost + second_latte_cost + syrup_cost

def cold_brew_cost (count : ℕ) (price : ℚ) (total_discount : ℚ) : ℚ :=
  let total_cost := count * price
  total_cost - total_discount

def cappuccino_cost (price : ℚ) (tip : ℚ) : ℚ :=
  let tip_amount := price * tip
  price + tip_amount

theorem meghan_coffee_order_cost : 
  let total_cost := 
    drip_coffee_cost 2 2.25 0.10 +
    espresso_cost 3.50 0.15 +
    latte_cost 4.00 2 0.50 0.50 0.20 +
    cold_brew_cost 2 2.50 1 +
    cappuccino_cost 3.50 0.05
  in total_cost = 22.35 := 
by sorry

end meghan_coffee_order_cost_l786_786071


namespace problem_f_f2_eq_1_l786_786881

def f (x : ℝ) : ℝ :=
  if x < 2 then Real.exp (x - 1)
  else Real.log x^2 / Real.log 3 - Real.log 1 / Real.log 3

theorem problem_f_f2_eq_1 : f (f 2) = 1 := by
  sorry

end problem_f_f2_eq_1_l786_786881


namespace fraction_is_half_l786_786633

variable (N : ℕ) (F : ℚ)

theorem fraction_is_half (h1 : N = 90) (h2 : 3 + F * (1/3) * (1/5) * N = (1/15) * N) : F = 1/2 :=
by
  sorry

end fraction_is_half_l786_786633


namespace solve_for_z_l786_786768

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786768


namespace KLMN_is_parallelogram_and_area_less_than_8_over_27_l786_786523

variables (A B C D K L M N : Type*)

-- Definitions of the data given
hypothesis h1 : ∀ (A B C D : Type*), convex_quadrilateral A B C D → 
  (∃! K, is_intersection_point A C B D K)

hypothesis h2 : ∀ (L : Type*) (A D : Type*), lies_on_side L A D

hypothesis h3 : ∀ (N : Type*) (B C : Type*), lies_on_side N B C

hypothesis h4 : ∀ (M : Type*) (A C : Type*), lies_on_diagonal M A C

hypothesis h5 : parallel KL AB
hypothesis h6 : parallel MN AB
hypothesis h7 : parallel LM DC

-- Proof statements
theorem KLMN_is_parallelogram_and_area_less_than_8_over_27 (A B C D K L M N : Type*) 
  (h1 : ∀ (A B C D : Type*), convex_quadrilateral A B C D → (∃! K, is_intersection_point A C B D K))
  (h2 : ∀ (L : Type*) (A D : Type*), lies_on_side L A D)
  (h3 : ∀ (N : Type*) (B C : Type*), lies_on_side N B C)
  (h4 : ∀ (M : Type*) (A C : Type*), lies_on_diagonal M A C)
  (h5 : parallel KL AB)
  (h6 : parallel MN AB)
  (h7 : parallel LM DC) :
  is_parallelogram K L M N ∧ area K L M N < (8/27) * area A B C D :=
sorry

end KLMN_is_parallelogram_and_area_less_than_8_over_27_l786_786523


namespace equation_is_hyperbola_l786_786358

theorem equation_is_hyperbola :
  ∃ (a b : ℝ) (h k : ℝ), 4 * x^2 - 9 * y^2 - 8 * x + 36 = 0 ∧
  (frac (x - h)^2 a^2 - frac y^2 b^2 = 1) := sorry

end equation_is_hyperbola_l786_786358


namespace find_z_l786_786803

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786803


namespace tanya_addition_problem_l786_786033

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786033


namespace markup_percentage_correct_l786_786687

-- Define variables
def cost_price : Float := 250.0
def tax_rate : Float := 0.12
def additional_expense : Float := 25.0
def profit_rate : Float := 0.30
def discount_rate1 : Float := 0.15
def discount_amount2 : Float := 65.0
def total_cost_price : Float := cost_price * (1.0 + tax_rate) + additional_expense
def final_selling_price : Float := total_cost_price * (1.0 + profit_rate)

-- Define the initial selling price using an equation obtained from the conditions
def isp : Float := (final_selling_price + discount_amount2) / (1.0 - discount_rate1)

-- Define the markup and markup percentage
def markup : Float := isp - total_cost_price
def markup_percentage : Float := (markup / total_cost_price) * 100.0

-- The statement to be proved
theorem markup_percentage_correct : abs (markup_percentage - 78.21) < 0.01 :=
by sorry

end markup_percentage_correct_l786_786687


namespace find_z_l786_786837

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786837


namespace johns_last_segment_speed_l786_786992

theorem johns_last_segment_speed :
  ∃ (x : ℕ), (x = 70) ∧ 
    let total_distance := 150 in
    let total_time := 150 / 60 in
    let first_segment_speed := 50 in
    let second_segment_speed := 60 in
    let third_segment_time := 50 / 60 in
    (total_distance / total_time = 
      (first_segment_speed * third_segment_time + 
       second_segment_speed * third_segment_time + 
       x * third_segment_time) / (3 * third_segment_time)) :=
begin
  -- To be proved
  sorry
end

end johns_last_segment_speed_l786_786992


namespace min_rectangular_pieces_l786_786068

-- Define the rectangular sheet and the number of holes as conditions
def rectangular_sheet (n : ℕ) : Type := 
  { sheet : ℝ × ℝ // ∀ i : fin n, ℝ × ℝ } 

-- Conjecture: Prove the minimum number of rectangular pieces the perforated sheet can be guaranteed to be cut into
theorem min_rectangular_pieces (n : ℕ) (holes : rectangular_sheet n) : 
  ∃ m : ℕ, (m = 3 * n + 1 ∧ ∀ cuts : list (ℝ → Prop), valid_cuts cuts holes → sections_after_cuts cuts = m) :=
sorry

end min_rectangular_pieces_l786_786068


namespace ways_to_divide_week_l786_786684

-- Define the total number of seconds in a week
def total_seconds_in_week : ℕ := 604800

-- Define the math problem statement
theorem ways_to_divide_week (n m : ℕ) (h : n * m = total_seconds_in_week) (hn : 0 < n) (hm : 0 < m) : 
  (∃ (n_pairs : ℕ), n_pairs = 144) :=
sorry

end ways_to_divide_week_l786_786684


namespace sahil_selling_price_l786_786288

theorem sahil_selling_price (purchase_price repair_cost transport_cost profit_percentage : ℝ) 
  (h_purchase : purchase_price = 11000) 
  (h_repair : repair_cost = 5000)
  (h_transport : transport_cost = 1000)
  (h_profit : profit_percentage = 0.50) : 
  let total_cost := purchase_price + repair_cost + transport_cost 
  in total_cost * (1 + profit_percentage) = 25500 :=
by
  sorry

end sahil_selling_price_l786_786288


namespace solve_complex_equation_l786_786791

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786791


namespace probability_divisible_by_4_and_5_l786_786655

theorem probability_divisible_by_4_and_5 : 
  let digits := [2, 4, 5, 6, 8] in
  let num_digits := digits.length in
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * (factorial (n - 1)) in
  let total_permutations := factorial num_digits in
  let favorable_permutations := 6 in
  let probability := favorable_permutations / total_permutations in
  num_digits = 5 → probability = (1 / 20) := 
by
  intros digits num_digits factorial total_permutations favorable_permutations probability
  intros h
  sorry

end probability_divisible_by_4_and_5_l786_786655


namespace smallest_four_digit_number_divisible_by_6_l786_786265

theorem smallest_four_digit_number_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 6 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m % 6 = 0) → n ≤ m :=
begin
  use 1002,
  split,
  { exact nat.le_succ 999,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.le_succ 1001,
    exact nat.succ_le_of_lt (by norm_num) },
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  { intros m h1,
    exact le_of_lt_iff.2 (by linarith) }
end

end smallest_four_digit_number_divisible_by_6_l786_786265


namespace minimum_cos_sin_half_l786_786383

theorem minimum_cos_sin_half (A: ℝ) :
  (∃ A, (A = 450 ∨ A % 360 = 90) ∧ (cos(π / 4) = sqrt 2 / 2 ∧ sin(π / 4) = sqrt 2 / 2)) 
  → cos (A / 2) + sin (A / 2) ≥ -sqrt 2 :=
by
  -- Proof to be provided
  sorry

end minimum_cos_sin_half_l786_786383


namespace student_average_grade_last_year_l786_786323

theorem student_average_grade_last_year:
  ∃ x : ℚ, 
    (let total_points_last_year := 6 * x,
         total_points_year_before := 5 * 50,
         total_courses := 6 + 5,
         average_two_years := 77,
         total_points_two_years := total_courses * average_two_years in 
    total_points_last_year + total_points_year_before = total_points_two_years) ∧
    x = 99.5 := 
by
  sorry

end student_average_grade_last_year_l786_786323


namespace sum_of_intersection_points_l786_786394

theorem sum_of_intersection_points : 
  let possible_values := {0, 1, 3, 4, 5, 6, 7, 8, 9, 10},
  sum_eq_53 : Finset.sum possible_values id = 53 :=
  by
    sorry

end sum_of_intersection_points_l786_786394


namespace regular_21_gon_symmetry_calculation_l786_786349

theorem regular_21_gon_symmetry_calculation:
  let L := 21
  let R := 360 / 21
  L + R = 38 :=
by
  sorry

end regular_21_gon_symmetry_calculation_l786_786349


namespace range_of_a_is_correct_l786_786924

open Classical

variable {a x : ℝ}

def p := x ≤ 1 / 2 ∨ x ≥ 1

def q := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a_is_correct :
  (∀ (x : ℝ), (p x → ¬ q x)) ∧ ¬(∀ (x : ℝ), ¬q x → p x) ↔ (0 ≤ a ∧ a ≤ 1 / 2) :=
by
  sorry

end range_of_a_is_correct_l786_786924


namespace number_of_ways_to_choose_reading_materials_l786_786244

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786244


namespace cos_seven_pi_over_four_l786_786735

theorem cos_seven_pi_over_four : 
  cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := 
by sorry

end cos_seven_pi_over_four_l786_786735


namespace range_of_P_l786_786893

theorem range_of_P (x y : ℝ) (h : x^2 / 3 + y^2 = 1) : 
  ∃ a b, (a ≤ b ∧ ∀ z, z ∈ set.range (λ (x y : ℝ), |2*x + y - 4| + |4 - x - 2*y| ) → a ≤ z ∧ z ≤ b ∧ set.range (λ (x y : ℝ), |2*x + y - 4| + |4 - x - 2*y| ) = set.Icc a b) ∧ a = 2 ∧ b = 14 :=
sorry

end range_of_P_l786_786893


namespace find_z_l786_786804

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786804


namespace tanya_addition_problem_l786_786028

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786028


namespace find_number_l786_786295

theorem find_number (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 :=
sorry

end find_number_l786_786295


namespace line_quadrants_l786_786955

theorem line_quadrants
  (k b : ℝ)
  (h1 : k > 0)
  (h2 : b < 0) :
  let f : ℝ → ℝ := λ x, k * x + b 
  let g : ℝ → ℝ := λ x, -b * x + k 
  (∀ x : ℝ, ∃ y : ℝ, y = f x ∧ (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0 ∨ x > 0 ∧ y < 0)) →
  (∀ x : ℝ, ∃ y : ℝ, y = g x ∧ (x > 0 ∧ y > 0 ∨ x < 0 ∧ y > 0 ∨ x > 0 ∧ y < 0)) :=
by 
  sorry

end line_quadrants_l786_786955


namespace part1_part2_l786_786894

def setA (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ 3 - 2 * a}
def setB := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}

theorem part1 (a : ℝ) : (setA a ∪ setB = setB) ↔ (-(1 / 2) ≤ a) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ∈ setB ↔ x ∈ setA a) ↔ (a ≤ -1) :=
sorry

end part1_part2_l786_786894


namespace solve_z_l786_786787

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786787


namespace smallest_y_value_l786_786267

-- Define the original equation
def original_eq (y : ℝ) := 3 * y^2 + 36 * y - 90 = y * (y + 18)

-- Define the problem statement
theorem smallest_y_value : ∃ (y : ℝ), original_eq y ∧ y = -15 :=
by
  sorry

end smallest_y_value_l786_786267


namespace find_y_l786_786137

theorem find_y (y : Real) : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9 → y = 53 / 3 :=
by
  sorry

end find_y_l786_786137


namespace set_difference_M_N_l786_786927

def setM : Set ℝ := { x | -1 < x ∧ x < 1 }
def setN : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem set_difference_M_N :
  setM \ setN = { x | -1 < x ∧ x < 0 } := sorry

end set_difference_M_N_l786_786927


namespace ProblemStatement_l786_786913

variable (α β : Type) [Plane α] [Plane β]
variable (a b : Line) (c : Line)
variable [Hα : a ∈ α] [Hβ : b ∈ β] [Hc_int : c = α ∩ β]

def PropositionI : Prop :=
  ¬ (∃ x : Point, x ∈ a ∧ x ∈ c) ∨ ¬ (∃ y : Point, y ∈ b ∧ y ∈ c)

def PropositionII : Prop :=
  ¬ ∃ (f : ℕ → Line), ∀ (m n : ℕ), m ≠ n → skew (f m) (f n)

theorem ProblemStatement : ¬ PropositionI ∧ ¬ PropositionII :=
by
  sorry

end ProblemStatement_l786_786913


namespace find_z_l786_786815

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786815


namespace arithmetic_sequence_problems_l786_786689

noncomputable def general_formula (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a_n n = -2 + (n - 1) * d

noncomputable def formula1 (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n n = 3 * n - 5

noncomputable def formula2 (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n n = (1 / 4) * n - (9 / 4)

noncomputable def sum_of_first_n_terms (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
∀ n, S_n n = n / 2 * (a_n 1 + a_n n)

noncomputable def largest_n_formula1 (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : ℕ :=
  Nat.find (λ n, S_n n < a_n n)

noncomputable def largest_n_formula2 (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : ℕ :=
  Nat.find (λ n, S_n n < a_n n)

theorem arithmetic_sequence_problems :
  ∃ d, d ≠ 0 ∧ general_formula (λ n, -2 + (n - 1) * d) d ∧
         (formula1 (λ n, -2 + (n - 1) * d) ∨ formula2 (λ n, -2 + (n - 1) * d)) ∧
         (let S1 := λ n, n / 2 * (-2 + -2 + (n - 1) * 3);
          largest_n_formula1 S1 (λ n, 3 * n - 5) = 3) ∧
         (let S2 := λ n, n / 2 * ((1 / 4) * 1 - (9 / 4) + (1 / 4) * (n - 1) - (9 / 4));
          largest_n_formula2 S2 (λ n, (1 / 4) * n - (9 / 4)) = 17) :=
by sorry

end arithmetic_sequence_problems_l786_786689


namespace min_initial_apples_l786_786544

theorem min_initial_apples (n : ℕ) 
  (cond1 : ∃ (k1 : ℕ), n = 3 * k1 + 1) 
  (cond2 : ∃ (k2 : ℕ), 3 * (k1 + cond1.some_val − k1) = k2 / 3 + 1)
  (cond3 : ∃ (k3 : ℕ), 3 * (k2 + cond2.some_val − k1) = k3 / 3 + 1) : 
  n ≥ 25 := sorry

end min_initial_apples_l786_786544


namespace largest_initial_number_l786_786027

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786027


namespace pencil_partition_l786_786948

theorem pencil_partition (total_length green_fraction green_length remaining_length white_fraction half_remaining white_length gold_length : ℝ)
  (h1 : green_fraction = 7 / 10)
  (h2 : total_length = 2)
  (h3 : green_length = green_fraction * total_length)
  (h4 : remaining_length = total_length - green_length)
  (h5 : white_fraction = 1 / 2)
  (h6 : white_length = white_fraction * remaining_length)
  (h7 : gold_length = remaining_length - white_length) :
  (gold_length / remaining_length) = 1 / 2 :=
sorry

end pencil_partition_l786_786948


namespace at_least_two_boys_two_girls_l786_786608

-- Definitions of the problem constraints
def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 6

-- The number of ways to choose a k-combination from n elements
def combination (n k : ℕ) : ℕ :=
  nat.choose n k

-- Calculate the probabilities
def total_ways : ℕ := combination total_members committee_size
def unwanted_ways : ℕ := combination girls committee_size + boys * combination girls (committee_size - 1) + combination boys committee_size + girls * combination boys (committee_size - 1)
def desired_ways : ℕ := total_ways - unwanted_ways
def probability : ℚ := desired_ways / total_ways

-- The theorem we want to prove
theorem at_least_two_boys_two_girls :
  probability = 457215 / 593775 :=
by sorry

end at_least_two_boys_two_girls_l786_786608


namespace half_hour_half_circle_half_hour_statement_is_true_l786_786986

-- Definitions based on conditions
def half_circle_divisions : ℕ := 30
def small_divisions_per_minute : ℕ := 1
def total_small_divisions : ℕ := 60
def minutes_per_circle : ℕ := 60

-- Relation of small divisions and time taken
def time_taken_for_small_divisions (divs : ℕ) : ℕ := divs * small_divisions_per_minute

-- Theorem to prove the statement
theorem half_hour_half_circle : time_taken_for_small_divisions half_circle_divisions = 30 :=
by
  -- Given half circle covers 30 small divisions
  -- Each small division represents 1 minute
  -- Therefore, time taken for 30 divisions should be 30 minutes
  exact rfl

-- The final statement proving the truth of the condition
theorem half_hour_statement_is_true : 
  (time_taken_for_small_divisions half_circle_divisions = 30) → True :=
by
  intro h
  trivial

end half_hour_half_circle_half_hour_statement_is_true_l786_786986


namespace factor_difference_of_squares_l786_786371

theorem factor_difference_of_squares (x : ℝ) : 49 - 16 * x^2 = (7 - 4 * x) * (7 + 4 * x) :=
by
  sorry

end factor_difference_of_squares_l786_786371


namespace circle_tangent_l786_786145

theorem circle_tangent
    (Γ Γ₁ Γ₂ : Circle)
    (M N A B C D : Point)
    (touches_M : Γ₁.touches Γ M)
    (touches_N : Γ₂.touches Γ N)
    (center_Γ₂_in_Γ₁ : Γ₁.passesThroughCenterOf Γ₂)
    (intersects_AB : LineThroughIntersectionPoints Γ₁ Γ₂ Γ A B)
    (intersect_MA : MA_intersects_Γ₁_at C)
    (intersect_MB : MB_intersects_Γ₁_at D) :
    TangentTo (CD Line) Γ₂ := 
sorry

end circle_tangent_l786_786145


namespace father_l786_786616

/-- The problem conditions -/
variables {M F : ℕ}
axiom condition1 : F - 3 = 8 * (M - 3)
axiom condition2 : F = 5 * M

/-- The problem statement: to prove the father's age this year is 35. -/
theorem father's_age_this_year : F = 35 := by
  sorry

end father_l786_786616


namespace calculate_a10_l786_786889

noncomputable def a (n : ℕ) : ℕ → ℕ := λ d, a + d * n

theorem calculate_a10 
  (a : ℕ) 
  (d : ℕ) 
  (h1 : a + (a + d) + (a + 2 * d) = 15)
  (h2 : (a + 2) * ((a + 2d) + 13)  = (a + d + 5) ^ 2)
  (h3: ∀ n, a n d > 0):
  a 10 2 = 21 :=
by 
  sorry

end calculate_a10_l786_786889


namespace ones_digit_of_power_l786_786388

theorem ones_digit_of_power (a b : ℕ) : (34^{34 * (17^{17})}) % 10 = 4 :=
by
  sorry

end ones_digit_of_power_l786_786388


namespace power_function_value_l786_786597

noncomputable def f (x : ℝ) : ℝ := x ^ 3

theorem power_function_value :
  (∀ a : ℝ, ∃ (P : ℝ × ℝ), P ∈ set_of (λ p, (p.2 = log a (2 * p.1 - 3) + 8)) ∧ P ∈ set_of (λ q, (q.2 = f q.1))) →
  f 4 = 64 :=
by
  intro h
  sorry

end power_function_value_l786_786597


namespace two_students_choose_materials_l786_786186

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786186


namespace largest_initial_number_l786_786067

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786067


namespace magnitude_of_complex_number_l786_786884

theorem magnitude_of_complex_number (z : ℂ) (h : z * (1 + complex.i) = 3 + complex.i) :
  complex.abs z = real.sqrt 5 :=
sorry

end magnitude_of_complex_number_l786_786884


namespace mixed_operations_with_decimals_false_l786_786725

-- Definitions and conditions
def operations_same_level_with_decimals : Prop :=
  ∀ (a b c : ℝ), a + b - c = (a + b) - c

def calculate_left_to_right_with_decimals : Prop :=
  ∀ (a b c : ℝ), (a - b + c) = a - b + c ∧ (a + b - c) = a + b - c

-- Proposition we're proving
theorem mixed_operations_with_decimals_false :
  ¬ ∀ (a b c : ℝ), (a + b - c) ≠ (a - b + c) :=
by
  intro h
  sorry

end mixed_operations_with_decimals_false_l786_786725


namespace tangent_line_at_1_l786_786590

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, f 1)

-- Define the slope of the tangent line at x=1
def slope_at_1 : ℝ := f' 1

-- Define the tangent line equation at x=1
def tangent_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem that the tangent line to f at x=1 is 2x - y + 1 = 0
theorem tangent_line_at_1 :
  tangent_line 1 (f 1) :=
by
  sorry

end tangent_line_at_1_l786_786590


namespace combined_sum_l786_786342

-- Define the nth triangular number t_n
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of the reciprocals of the first 1000 triangular numbers
def sum_reciprocals_triangular : ℚ :=
  (∑ i in Finset.range 1000, 1 / (triangular (i+1) : ℚ))

-- Define the sum of the squares of the first 1000 natural numbers
def sum_squares : ℚ :=
  ∑ i in Finset.range 1000, (i + 1) ^ 2

-- Combined problem to prove the final sum equals the given value
theorem combined_sum :
  sum_reciprocals_triangular + sum_squares = 333500168.6667 := by
  sorry

end combined_sum_l786_786342


namespace two_students_one_common_material_l786_786200

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786200


namespace number_of_ways_l786_786215

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786215


namespace area_of_rectangle_l786_786119

-- Define data types for the triangle and rectangle
structure Triangle :=
  (P Q R : Point)
  (anglePQR : Angle)

structure Rectangle :=
  (L M N O : Point)

structure Point :=
  (x y : Real)

structure Angle :=
  (deg : Real)

-- Define the conditions provided in part a)
axiom altitude_P_QR : ∀ (P Q R : Point), (Triangle P Q R) → (altitude_P_QR = 8)

axiom length_QR : ∀ (P Q R : Point), (Triangle P Q R) → (QR = 12)

axiom LM_third_LN : ∀ (L M N O : Point), (Rectangle L M N O) → (LM = (1/3) * LN)

axiom inscribed_Rectangle : ∀ (L M N O P Q R : Point) (rect: Rectangle L M N O),
                            (Triangle P Q R) → 
                            (side LN on side QR of triangle PQR)

-- Define the main theorem (proof problem)
theorem area_of_rectangle :
  ∀ (L M N O P Q R : Point) (rect: Rectangle L M N O) (tri: Triangle P Q R),
  inscribed_Rectangle L M N O P Q R rect tri →
  altitude_P_QR P Q R tri →
  length_QR P Q R tri →
  LM_third_LN L M N O rect →
  area rectangle LMNO = 27 :=
sorry

end area_of_rectangle_l786_786119


namespace total_weight_of_plastic_rings_l786_786994

-- Conditions
def orange_ring_weight : ℝ := 0.08
def purple_ring_weight : ℝ := 0.33
def white_ring_weight : ℝ := 0.42

-- Proof Statement
theorem total_weight_of_plastic_rings :
  orange_ring_weight + purple_ring_weight + white_ring_weight = 0.83 := by
  sorry

end total_weight_of_plastic_rings_l786_786994


namespace number_of_campers_is_22_l786_786547

def trout_weight : ℕ := 8
def bass_individual_weight : ℕ := 2
def number_of_bass : ℕ := 6
def salmon_individual_weight : ℕ := 12
def number_of_salmon : ℕ := 2
def weight_per_person : ℕ := 2

def total_weight := trout_weight + (number_of_bass * bass_individual_weight) + (number_of_salmon * salmon_individual_weight)
def number_of_campers := total_weight / weight_per_person

theorem number_of_campers_is_22 : number_of_campers = 22 := 
by 
  have h1 : total_weight = 44 := by calc
    total_weight = trout_weight + (number_of_bass * bass_individual_weight) + (number_of_salmon * salmon_individual_weight) := rfl
    ... = 8 + (6 * 2) + (2 * 12) := rfl
    ... = 8 + 12 + 24 := rfl
    ... = 44 := rfl
  have h2 : number_of_campers = total_weight / weight_per_person := rfl
  have h3 : number_of_campers = 44 / 2 := by rw [h2, h1]
  show number_of_campers = 22
  calc 
    number_of_campers = 44 / 2 := h3
    ... = 22 := rfl

end number_of_campers_is_22_l786_786547


namespace locus_eq_closed_curve_l786_786350

noncomputable def locus_of_points (d : ℝ) : set (ℝ × ℝ) :=
  let A := (d, 0)
  let B := (0, d)
  let O := (0, 0)
  let segment_AB := { P | 0 ≤ P.1 ∧ P.1 ≤ d ∧ P.2 = d - P.1 }
  let arc_third_quadrant := { Q | Q.1^2 + Q.2^2 = (d/2)^2 ∧ Q.1 ≤ 0 ∧ Q.2 ≤ 0 }
  let parabola_second_quadrant := { X | (X.1^2 - X.2) = (d^2 / 4) }
  let parabola_fourth_quadrant := { Y | (Y.1^2 + Y.2) = (d^2 / 4) }
  in segment_AB ∪ arc_third_quadrant ∪ parabola_second_quadrant ∪ parabola_fourth_quadrant

theorem locus_eq_closed_curve (d : ℝ) :
  ∃ (closed_curve : set (ℝ × ℝ)), closed_curve = locus_of_points d :=
by
  sorry

end locus_eq_closed_curve_l786_786350


namespace unique_A_value_l786_786722

theorem unique_A_value (A : ℝ) (x1 x2 : ℂ) (hx1_ne : x1 ≠ x2) :
  (x1 * (x1 + 1) = A) ∧ (x2 * (x2 + 1) = A) ∧ (A * x1^4 + 3 * x1^3 + 5 * x1 = x2^4 + 3 * x2^3 + 5 * x2) 
  → A = -7 := by
  sorry

end unique_A_value_l786_786722


namespace monomials_like_terms_l786_786000

theorem monomials_like_terms (m n : ℕ) (h₁ : n = 1) (h₂ : m = 2) : (m - n) ^ 2023 = 1 := by
  rw [h₁, h₂]
  norm_num
  sorry

end monomials_like_terms_l786_786000


namespace jon_speed_gain_per_week_l786_786513

/-- Given Jon's initial speed of 80 mph, and after training 4 times for 4 weeks each,
his speed increases by 20%. Prove that he gained 1 mph per week on average. -/
theorem jon_speed_gain_per_week : 
  ∀ (initial_speed : ℕ) (speed_increase_percentage : ℚ) (training_sessions : ℕ) (weeks_per_session : ℕ),
  initial_speed = 80 →
  speed_increase_percentage = 0.20 →
  training_sessions = 4 →
  weeks_per_session = 4 →
  let total_speed_increase := initial_speed * speed_increase_percentage in
  let total_weeks := training_sessions * weeks_per_session in
  total_speed_increase / total_weeks = 1 :=
by
  intros initial_speed speed_increase_percentage training_sessions weeks_per_session
  intros h1 h2 h3 h4
  let total_speed_increase := initial_speed * speed_increase_percentage
  let total_weeks := training_sessions * weeks_per_session
  have h_total_speed_increase : total_speed_increase = 16 := by rw [h1, h2]; norm_num
  have h_total_weeks : total_weeks = 16 := by rw [h3, h4]; norm_num
  have h_division : 16 / 16 = 1 := by norm_num
  rw [h_total_speed_increase, h_total_weeks, h_division]
  sorry

end jon_speed_gain_per_week_l786_786513


namespace difference_in_percentage_blue_vs_striped_l786_786504

theorem difference_in_percentage_blue_vs_striped :
  ∀ (total_nails purple_nails blue_nails : ℕ),
    total_nails = 20 → purple_nails = 6 → blue_nails = 8 →
    let striped_nails := total_nails - purple_nails - blue_nails in
    let blue_percentage := (blue_nails : ℝ) / total_nails * 100 in
    let striped_percentage := (striped_nails : ℝ) / total_nails * 100 in
    blue_percentage - striped_percentage = 10 :=
by
  intros total_nails purple_nails blue_nails h_total h_purple h_blue
  let striped_nails := total_nails - purple_nails - blue_nails
  let blue_percentage := (blue_nails : ℝ) / total_nails * 100
  let striped_percentage := (striped_nails : ℝ) / total_nails * 100
  sorry

end difference_in_percentage_blue_vs_striped_l786_786504


namespace arrow_directions_from_2008_to_2010_l786_786418

theorem arrow_directions_from_2008_to_2010 :
  (multiple_of_4 2008) →
  (∀ n, multiple_of_4 n → appears_on_top_row n) →
  (∀ n, multiple_of_4 n → arrow_after n = ↓) →
  arrow_sequence 2008 2010 = [↓, →] :=
sorry

end arrow_directions_from_2008_to_2010_l786_786418


namespace cos2alpha_over_sin_alpha_pi4_l786_786876

theorem cos2alpha_over_sin_alpha_pi4 (α : ℝ) (h₁ : \sin(α) = \frac{1}{2} + \cos(α)) (h₂ : 0 < α ∧ α < \frac{\pi}{2}) : 
  \frac{\cos(2α)}{\sin(α + \frac{\pi}{4})} = -\frac{\sqrt{2}}{2} :=
sorry

end cos2alpha_over_sin_alpha_pi4_l786_786876


namespace find_z_l786_786816

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786816


namespace correct_answer_l786_786637

def f (x : ℝ) : ℝ := sin x * (sin x - cos x)

theorem correct_answer : ∀ x, f (x) = f (π/8 - x) :=
by
  intro x
  sorry

end correct_answer_l786_786637


namespace opposite_of_7_l786_786159

theorem opposite_of_7 :
  ∀ (x : ℤ), x = 7 → ∃ (y : ℤ), 7 + y = 0 ∧ y = -7 :=
by
  intros x hx
  use -7
  split
  · rw hx
    exact add_neg_self 7
  · refl

end opposite_of_7_l786_786159


namespace find_n_l786_786293

theorem find_n : ∃ n : ℕ, n < 200 ∧ ∃ k : ℕ, n^2 + (n + 1)^2 = k^2 ∧ (n = 3 ∨ n = 20 ∨ n = 119) := 
by
  sorry

end find_n_l786_786293


namespace solve_equation_l786_786124

-- Defining the original equation as a Lean function
def equation (x : ℝ) : Prop :=
  (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2))

theorem solve_equation :
  ∃ x : ℝ, equation x ∧ x = -13 / 2 :=
by
  -- Equation specification and transformations
  sorry

end solve_equation_l786_786124


namespace problem_2017_f_2017_eq_cos_l786_786088

-- Define the sequence of functions
def f : ℕ → (ℝ → ℝ)
| 0       := sin
| (n + 1) := (f n)' -- derivative of the previous function

-- Problem statement
theorem problem_2017_f_2017_eq_cos x : f 2017 x = cos x := 
sorry

end problem_2017_f_2017_eq_cos_l786_786088


namespace number_of_ways_to_choose_reading_materials_l786_786246

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786246


namespace central_angle_measure_l786_786142

-- Constants representing the arc length and the area of the sector.
def arc_length : ℝ := 5
def sector_area : ℝ := 5

-- Variables representing the central angle in radians and the radius.
variable (α r : ℝ)

-- Conditions given in the problem.
axiom arc_length_eq : arc_length = α * r
axiom sector_area_eq : sector_area = 1 / 2 * α * r^2

-- The goal to prove that the radian measure of the central angle α is 5 / 2.
theorem central_angle_measure : α = 5 / 2 := by sorry

end central_angle_measure_l786_786142


namespace find_z_l786_786883

def is_imaginary_unit (i : ℂ) : Prop :=
  i = complex.I

def is_correct_complex_value (z i : ℂ) : Prop :=
  z * (1 + i) = 1 + 3 * i

theorem find_z (z i : ℂ) (h1 : is_imaginary_unit i) (h2 : is_correct_complex_value z i) : z = 2 + i :=
sorry

end find_z_l786_786883


namespace fill_n_by_n_table_conditions_l786_786546

theorem fill_n_by_n_table_conditions (n : ℕ) (h : n > 1) : 
  (∃ (f : Fin n × Fin n → ℕ), 
    (∀ (i j : Fin n × Fin n), 
      (f i = f j + 1 ∨ f j = f i + 1 → (abs (i.1 - j.1) + abs (i.2 - j.2) = 1)) ∧ 
      (f i % n = f j % n → i.1 ≠ j.1 ∧ i.2 ≠ j.2))) ↔ Even n := 
by 
  sorry

end fill_n_by_n_table_conditions_l786_786546


namespace stripe_length_l786_786648

theorem stripe_length
  (C : ℝ) (hC : C = 18)
  (H : ℝ) (hH : H = 8)
  : ∃ L : ℝ, L = sqrt(1360) :=
by {
  use sqrt(1360),
  sorry
}

end stripe_length_l786_786648


namespace autumn_pencils_l786_786698

theorem autumn_pencils : 
  ∃ (x : ℕ), 
    let initial_pencils := 20 in
    let misplaced_pencils := 7 in
    let broken_pencils := 3 in
    let found_pencils := 4 in
    let final_pencils := 16 in
    initial_pencils - misplaced_pencils - broken_pencils + found_pencils + x = final_pencils :=
begin
  sorry
end

end autumn_pencils_l786_786698


namespace solve_for_z_l786_786769

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786769


namespace measure_of_angle_A_l786_786956

theorem measure_of_angle_A {A B C : ℝ} (hC : C = 2 * B) (hB : B = 21) :
  A = 180 - B - C := 
by 
  sorry

end measure_of_angle_A_l786_786956


namespace henry_finishes_on_thursday_l786_786938

theorem henry_finishes_on_thursday :
  let total_days := 210
  let start_day := 4  -- Assume Thursday is 4th day of the week in 0-indexed (0=Sunday, 1=Monday, ..., 6=Saturday)
  (start_day + total_days) % 7 = start_day :=
by
  sorry

end henry_finishes_on_thursday_l786_786938


namespace max_table_height_l786_786497

theorem max_table_height 
  (DE EF FD : ℝ)
  (h_d h_e h_f : ℝ)
  (PQ RS TU : set ℝ)
  (P Q R S T U : ℝ)
  (h : ℝ)
  (area_of_tr DEF_area : ℝ)
  (same_side_PQ : PQ = { res in \overline{DE} | P ∈ PQ ∧ Q ∈ PQ })
  (same_side_RS : RS = { res in \overline{EF} | R ∈ RS ∧ S ∈ RS })
  (same_side_TU : TU = { res in \overline{FD} | T ∈ TU ∧ U ∈ TU })
  (parallel_PQ_EF : ∀ P ∈ PQ, ∀ Q ∈ PQ, ∀ EF ∈  \overline{DE},  PQ ∥ EF)
  (parallel_RS_FD : ∀ R ∈ RS, ∀ S ∈ RS, ∀ FD ∈ \overline{EF},  RS ∥ FD)
  (parallel_TU_DE : ∀ T ∈ TU, ∀ U ∈ TU, ∀ DE ∈ \overline{FD},  TU ∥ DE):
  h = min ( (h_d * h_e) / (h_d + h_e), (h_e * h_f) / (h_e + h_f), (h_f * h_d) / (h_f + h_d) ) :=
by
  sorry

end max_table_height_l786_786497


namespace least_n_factorial_6930_l786_786256

theorem least_n_factorial_6930 (n : ℕ) (h : n! % 6930 = 0) : n ≥ 11 := by
  sorry

end least_n_factorial_6930_l786_786256


namespace audio_per_cd_l786_786333

theorem audio_per_cd (total_audio : ℕ) (max_per_cd : ℕ) (num_cds : ℕ) 
  (h1 : total_audio = 360) 
  (h2 : max_per_cd = 60) 
  (h3 : num_cds = total_audio / max_per_cd): 
  (total_audio / num_cds = max_per_cd) :=
by
  sorry

end audio_per_cd_l786_786333


namespace solve_for_y_l786_786131

theorem solve_for_y (y : ℝ) (h : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) : y = 53 / 3 := by
  sorry

end solve_for_y_l786_786131


namespace lettuce_price_1_l786_786250

theorem lettuce_price_1 (customers_per_month : ℕ) (lettuce_per_customer : ℕ) (tomatoes_per_customer : ℕ) 
(price_per_tomato : ℝ) (total_sales : ℝ)
  (h_customers : customers_per_month = 500)
  (h_lettuce_per_customer : lettuce_per_customer = 2)
  (h_tomatoes_per_customer : tomatoes_per_customer = 4)
  (h_price_per_tomato : price_per_tomato = 0.5)
  (h_total_sales : total_sales = 2000) :
  let heads_of_lettuce_sold := customers_per_month * lettuce_per_customer
  let tomato_sales := customers_per_month * tomatoes_per_customer * price_per_tomato
  let lettuce_sales := total_sales - tomato_sales
  let price_per_lettuce := lettuce_sales / heads_of_lettuce_sold
  price_per_lettuce = 1 := by
{
  sorry
}

end lettuce_price_1_l786_786250


namespace solve_equation_l786_786646

theorem solve_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := 
by
  sorry  -- Placeholder for the proof

end solve_equation_l786_786646


namespace tangent_segment_length_proof_l786_786527

-- Define the circles C1 and C2
def circle1 : set (ℝ × ℝ) := {p | (p.1 - 6)^2 + p.2^2 = 25}
def circle2 : set (ℝ × ℝ) := {p | (p.1 + 10)^2 + p.2^2 = 36}

-- A noncomputable definition of the length of the tangent segment PQ
noncomputable def tangent_segment_length : ℝ :=
  11

-- The theorem stating that the length of the shortest line segment PQ that
-- is tangent to both given circles is 11
theorem tangent_segment_length_proof :
  ∀ P Q : ℝ × ℝ, 
  P ∈ circle1 →
  Q ∈ circle2 →
  tangent P circle1 →
  tangent Q circle2 →
  dist P Q = tangent_segment_length :=
begin
  intros P Q hP hQ hTangentP hTangentQ,
  sorry
end

end tangent_segment_length_proof_l786_786527


namespace largest_initial_number_l786_786040

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786040


namespace arithmetic_sequence_fifth_term_l786_786294

theorem arithmetic_sequence_fifth_term (x y : ℝ) : 
  let a1 := x + 2y,
      a2 := x - 2y,
      a3 := 2xy,
      a4 := x^2 / y,
      d  := a2 - a1 in
  a1 + 4 * d = a4 - d :=
by {
  let a1 := x + 2y,
  let a2 := x - 2y,
  let a3 := 2 * x * y,
  let a4 := x^2 / y,
  let d  := a2 - a1,
  sorry,
}

end arithmetic_sequence_fifth_term_l786_786294


namespace heal_time_l786_786987

theorem heal_time (x : ℝ) (hx_pos : 0 < x) (h_total : 2.5 * x = 10) : x = 4 := 
by {
  -- Lean proof will be here
  sorry
}

end heal_time_l786_786987


namespace smallest_angle_in_icosagon_l786_786583

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end smallest_angle_in_icosagon_l786_786583


namespace solve_complex_equation_l786_786796

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786796


namespace crow_speed_l786_786639

-- Define the conditions
def distance_nest_to_ditch : ℝ := 200 -- distance in meters
def round_trip_distance : ℝ := 2 * distance_nest_to_ditch -- distance in meters
def total_round_trips : ℕ := 15
def total_distance : ℝ := total_round_trips * round_trip_distance / 1000 -- convert to kilometers
def total_time : ℝ := 1.5 -- time in hours

-- Theorem to prove the crow's speed
theorem crow_speed : (total_distance / total_time) = 4 := 
by
  sorry

end crow_speed_l786_786639


namespace solve_for_z_l786_786862

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786862


namespace determinant_is_correct_l786_786520

noncomputable def v : ℝ × ℝ × ℝ := (3, 2, -2)
noncomputable def w : ℝ × ℝ × ℝ := (-1, 1, 4)

noncomputable def direction_u : ℝ × ℝ × ℝ := (-1, 1, 0)
noncomputable def norm (x : ℝ × ℝ × ℝ) : ℝ := real.sqrt (x.1^2 + x.2^2 + x.3^2)

noncomputable def u : ℝ × ℝ × ℝ :=
let n := norm direction_u in
(direction_u.1 / n, direction_u.2 / n, direction_u.3 / n)

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem determinant_is_correct :
  let cross_vw := cross_product v w in
  dot_product u cross_vw = -4 * real.sqrt 2 := by
  sorry

end determinant_is_correct_l786_786520


namespace fraction_product_eq_l786_786369
-- Import the necessary library

-- Define the fractions and the product
def fraction_product : ℚ :=
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8)

-- State the theorem we want to prove
theorem fraction_product_eq : fraction_product = 3 / 8 := 
sorry

end fraction_product_eq_l786_786369


namespace log_base_2_y_l786_786951

theorem log_base_2_y (y : ℝ) (h : y = (Real.log 3 / Real.log 9) ^ Real.log 27 / Real.log 3) : 
  Real.log y = -3 :=
by
  sorry

end log_base_2_y_l786_786951


namespace mean_inequalities_l786_786526

variable (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x)
variable (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z)
variable (h7 : abs (x - y) < δ) (h8 : abs (y - z) < δ) (h9 : abs (z - x) < δ)

theorem mean_inequalities (δ : ℝ) (h : 0 < δ): 
  (x ≈ y) → (y ≈ z) → 
  (x + y) / 2 > √(x * y) ∧ √(x * y) > (2 * y * z) / (y + z) := 
by 
  sorry

end mean_inequalities_l786_786526


namespace line_through_incenter_intersects_circles_l786_786307

open_locale big_operators

theorem line_through_incenter_intersects_circles
  (ABC : Triangle)
  (I : Point) 
  (D E F G : Point) 
  (r R : ℝ)
  (hI_incenter : incenter ABC I)
  (h_line : ∀ p, p ∈ line_through I → (incircle ABC p ↔ p = D ∨ p = E) ∧ (circumcircle ABC p ↔ p = F ∨ p = G))
  (hD_between_I_F : between D I F)
  (hr_positive : 0 < r)
  (h_r_inradius : inradius ABC = r)
  (h_R_circumradius : circumradius ABC = R) :
  distance D F * distance E G ≥ r^2 :=
sorry

end line_through_incenter_intersects_circles_l786_786307


namespace solve_for_z_l786_786831

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786831


namespace two_students_choose_materials_l786_786188

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786188


namespace chi_square_confidence_l786_786952

theorem chi_square_confidence (chi_square : ℝ) (df : ℕ) (critical_value : ℝ) :
  chi_square = 6.825 ∧ df = 1 ∧ critical_value = 6.635 → confidence_level = 0.99 := 
by
  sorry

end chi_square_confidence_l786_786952


namespace a_2024_value_l786_786879

-- Defining the sequence using a general recursive formula
def a : ℕ → ℤ
| 1       := 0
| (n + 1) := -|(a n) + (n + 1)|

-- The theorem to prove the value of a_2024
theorem a_2024_value : a 2024 = -1012 := by
  sorry

end a_2024_value_l786_786879


namespace symmetric_slope_angle_l786_786444

theorem symmetric_slope_angle (α₁ : ℝ)
  (hα₁ : 0 ≤ α₁ ∧ α₁ < Real.pi) :
  ∃ α₂ : ℝ, (α₁ < Real.pi / 2 → α₂ = Real.pi - α₁) ∧
            (α₁ = Real.pi / 2 → α₂ = 0) :=
sorry

end symmetric_slope_angle_l786_786444


namespace h_is_odd_not_even_l786_786532
noncomputable def Q : set ℝ := {x : ℝ | ∃ q : ℚ, ↑q = x}

def f (x : ℝ) : ℝ :=
if x ∈ Q then 1 else -1

def g (x : ℝ) : ℝ :=
(e^x - 1) / (e^x + 1)

def h (x : ℝ) : ℝ :=
f x * g x

theorem h_is_odd_not_even : (∀ x : ℝ, h (-x) = -h x) ∧ ¬ (∀ x : ℝ, h (-x) = h x) :=
by
  sorry

end h_is_odd_not_even_l786_786532


namespace tims_integer_is_unique_l786_786617

theorem tims_integer_is_unique :
  ∃! n : ℕ, (2 ≤ n ∧ n ≤ 15) ∧ (number_of_factors n = 6) :=
begin
  sorry
end

end tims_integer_is_unique_l786_786617


namespace largest_initial_number_l786_786055

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786055


namespace pet_store_animals_left_l786_786317

def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5
def initial_spiders : Nat := 15

def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

def birds_left : Nat := initial_birds - birds_sold
def puppies_left : Nat := initial_puppies - puppies_adopted
def cats_left : Nat := initial_cats
def spiders_left : Nat := initial_spiders - spiders_loose

def total_animals_left : Nat := birds_left + puppies_left + cats_left + spiders_left

theorem pet_store_animals_left : total_animals_left = 25 :=
by
  sorry

end pet_store_animals_left_l786_786317


namespace find_x_value_l786_786375

noncomputable def floor_plus_2x_eq_33 (x : ℝ) : Prop :=
  ∃ n : ℤ, ⌊x⌋ = n ∧ n + 2 * x = 33 ∧  (0 : ℝ) ≤ x - n ∧ x - n < 1

theorem find_x_value : ∀ x : ℝ, floor_plus_2x_eq_33 x → x = 11 :=
by
  intro x
  intro h
  -- Proof skipped, included as 'sorry' to compile successfully.
  sorry

end find_x_value_l786_786375


namespace largest_initial_number_l786_786056

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786056


namespace difference_in_percentage_blue_vs_striped_l786_786505

theorem difference_in_percentage_blue_vs_striped :
  ∀ (total_nails purple_nails blue_nails : ℕ),
    total_nails = 20 → purple_nails = 6 → blue_nails = 8 →
    let striped_nails := total_nails - purple_nails - blue_nails in
    let blue_percentage := (blue_nails : ℝ) / total_nails * 100 in
    let striped_percentage := (striped_nails : ℝ) / total_nails * 100 in
    blue_percentage - striped_percentage = 10 :=
by
  intros total_nails purple_nails blue_nails h_total h_purple h_blue
  let striped_nails := total_nails - purple_nails - blue_nails
  let blue_percentage := (blue_nails : ℝ) / total_nails * 100
  let striped_percentage := (striped_nails : ℝ) / total_nails * 100
  sorry

end difference_in_percentage_blue_vs_striped_l786_786505


namespace area_of_trapezoid_l786_786626

-- Define the lines bounding the trapezoid
def line1 (x y : ℝ) : Prop := y = x
def line2 (y : ℝ) : Prop := y = 10
def line3 (y : ℝ) : Prop := y = 5
def y_axis (x : ℝ) : Prop := x = 0

-- Define the vertices of the trapezoid
def vertex1 : ℝ × ℝ := (10, 10)
def vertex2 : ℝ × ℝ := (5, 5)
def vertex3 : ℝ × ℝ := (0, 10)
def vertex4 : ℝ × ℝ := (0, 5)

-- Calculate the lengths of the bases
def base1 : ℝ := 10
def base2 : ℝ := 5

-- Calculate the height of the trapezoid
def height : ℝ := 5

-- State the problem to prove the area of the trapezoid
theorem area_of_trapezoid : (1/2) * (base1 + base2) * height = 37.5 := 
by 
  -- skipping the proof steps with sorry
  sorry

end area_of_trapezoid_l786_786626


namespace find_z_l786_786841

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786841


namespace distance_between_A_and_B_l786_786377

-- Define the points
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (-3, 6)

-- Define the distance function between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  Math.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Prove that the distance between A and B is sqrt(74)
theorem distance_between_A_and_B :
  distance A B = Real.sqrt 74 :=
by
  sorry

end distance_between_A_and_B_l786_786377


namespace centroid_coordinates_l786_786182

theorem centroid_coordinates (P Q R S : Point) (x y : ℝ)
    (hP : P = (2, 5))
    (hQ : Q = (-1, -3))
    (hR : R = (7, 0))
    (hS : S = centroid P Q R)
    (hx : x = centroid_x P Q R)
    (hy : y = centroid_y P Q R) :
    7 * x + 3 * y = 62 / 3 := 
  sorry

end centroid_coordinates_l786_786182


namespace solve_for_z_l786_786863

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786863


namespace find_prob_xi_leq_1_l786_786435

noncomputable def xi : Type := sorry
noncomputable def ξ : xi := sorry
noncomputable def normal_dist {μ σ : ℝ} (ξ : xi) : Prop := sorry

theorem find_prob_xi_leq_1 (δ : ℝ) (h1 : normal_dist 2 δ ξ) (h2 : P ξ ≤ 3 = 0.8413) :
  P ξ ≤ 1 = 0.1587 :=
by
  sorry

end find_prob_xi_leq_1_l786_786435


namespace students_material_selection_l786_786225

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786225


namespace part1_part2_l786_786084

-- Part 1: Prove A = π / 3
theorem part1 (a b c A B C : ℝ)
  (h1 : sqrt (b + c) = sqrt (2 * a * sin (C + π / 6)))
  (h2 : sin (A + B + C) = 0) :
  A = π / 3 := sorry

-- Part 2: Prove the range of a sin B
theorem part2 (a b c A B C : ℝ)
  (h3 : A + B + C = π)
  (h4 : sin (C + 2 * A / 3) = 1)
  (h5 : c = 2)
  (h6 : A < π / 2) :
  a * sin B ∈ Ioo (sqrt 3 / 2) (2 * sqrt 3) := sorry

end part1_part2_l786_786084


namespace f_is_odd_f_inequality_l786_786902

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 - 3^x else -1 + 3^(-x)

theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) :=
by
  intro x
  unfold f
  split_ifs;
  {
    sorry
  }

theorem f_inequality (a : ℝ) (h_a : a ≥ 6) : ∀ x : ℝ, (2 ≤ x ∧ x ≤ 8) → f ((Real.log x / Real.log 2) ^ 2) + f (5 - a * (Real.log x / Real.log 2)) ≥ 0 :=
by
  intros x hx
  have hlog1 : 1 ≤ Real.log x / Real.log 2 := sorry
  have hlog2 : Real.log x / Real.log 2 ≤ 3 := sorry
  let t := Real.log x / Real.log 2
  let g := λ t : ℝ, t^2 - a * t + 5
  have h_gmax : ∀ t : ℝ, (1 ≤ t ∧ t ≤ 3) → g t ≤ 0 := sorry
  exact h_gmax t ⟨hlog1, hlog2⟩

end f_is_odd_f_inequality_l786_786902


namespace percentage_more_than_6_years_l786_786662

theorem percentage_more_than_6_years
  (x : ℕ)
  (less_than_3_years : ℕ := 10 * x)
  (three_to_six_years : ℕ := 15 * x)
  (more_than_6_years : ℕ := 7 * x)
  (total_employees : ℕ := less_than_3_years + three_to_six_years + more_than_6_years) :
  (more_than_6_years.to_rat / total_employees.to_rat) * 100 = 21.875 := by
  sorry

end percentage_more_than_6_years_l786_786662


namespace x_minus_y_l786_786720

-- Definitions: Convert 253 from base 10 to base 2 representation 
def base2_representation_of_253 := "11111101"

def count_zeros (s : String) : Nat :=
  s.fold 0 (fun count ch => if ch = '0' then count + 1 else count)

def count_ones (s : String) : Nat :=
  s.fold 0 (fun count ch => if ch = '1' then count + 1 else count)

theorem x_minus_y : (count_zeros base2_representation_of_253) - (count_ones base2_representation_of_253) = -2 := 
  sorry

end x_minus_y_l786_786720


namespace number_of_ways_to_choose_reading_materials_l786_786240

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786240


namespace shaded_area_l786_786625

theorem shaded_area (side_length : ℝ) (h_side_length : side_length = 20) : 
  let r := side_length / 4 in
  let A_square := side_length ^ 2 in
  let A_circle := π * r ^ 2 in
  let total_area_circles := 4 * A_circle in
  A_square - total_area_circles = 400 - 100 * π :=
by
  sorry

end shaded_area_l786_786625


namespace smallest_angle_in_convex_20_gon_seq_l786_786584

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end smallest_angle_in_convex_20_gon_seq_l786_786584


namespace turtles_in_lake_l786_786176

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end turtles_in_lake_l786_786176


namespace ball_distribution_l786_786461

theorem ball_distribution :
  let balls := 7
  let boxes := 3
  let total_ways := 1 + 7 + 21 + 21 + 35 + 105 + 70 + 105
  balls = 7 ∧ boxes = 3 → total_ways = 365 :=
by
  intro h
  simp
  exact rfl

end ball_distribution_l786_786461


namespace garden_area_maximal_l786_786408

/-- Given a garden with sides 20 meters, 16 meters, 12 meters, and 10 meters, 
    prove that the area is approximately 194.4 square meters. -/
theorem garden_area_maximal (a b c d : ℝ) (h1 : a = 20) (h2 : b = 16) (h3 : c = 12) (h4 : d = 10) :
    ∃ A : ℝ, abs (A - 194.4) < 0.1 :=
by
  sorry

end garden_area_maximal_l786_786408


namespace sum_of_areas_of_disks_l786_786734

theorem sum_of_areas_of_disks (r : ℝ) (a b c : ℕ) (h : a + b + c = 123) :
  ∃ (r : ℝ), (15 * Real.pi * r^2 = Real.pi * ((105 / 4) - 15 * Real.sqrt 3) ∧ r = 1 - (Real.sqrt 3) / 2) := 
by
  sorry

end sum_of_areas_of_disks_l786_786734


namespace min_pairs_l786_786542

-- Define the types for knights and liars
inductive Residents
| Knight : Residents
| Liar : Residents

def total_residents : ℕ := 200
def knights : ℕ := 100
def liars : ℕ := 100

-- Additional conditions
def conditions (friend_claims_knights friend_claims_liars : ℕ) : Prop :=
  friend_claims_knights = 100 ∧
  friend_claims_liars = 100 ∧
  knights + liars = total_residents

-- Minimum number of knight-liar pairs to prove
def min_knight_liar_pairs : ℕ := 50

theorem min_pairs {friend_claims_knights friend_claims_liars : ℕ} (h : conditions friend_claims_knights friend_claims_liars) :
    min_knight_liar_pairs = 50 :=
sorry

end min_pairs_l786_786542


namespace number_of_ways_to_choose_materials_l786_786233

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786233


namespace find_y_l786_786135

theorem find_y (y : Real) : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9 → y = 53 / 3 :=
by
  sorry

end find_y_l786_786135


namespace number_of_ways_l786_786218

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786218


namespace largest_smallest_number_l786_786248

def is_ten_digit_number (n : ℕ) : Prop :=
  1000000000 ≤ n ∧ n < 10000000000

def no_repeated_digits (n : ℕ) : Prop :=
  let digits := (List.of_fn (λ i => (n / 10^i) % 10)).filter (λ x => x > 0);
  digits.length = digits.eraseDuplicates.length

def divisible_by_11 (n : ℕ) : Prop :=
  let digits := List.of_fn (λ i => (n / 10^i) % 10);
  let x := digits.enum.filter (λ ⟨i, _⟩ => i % 2 = 0).map Prod.snd |>.sum;
  let y := digits.enum.filter (λ ⟨i, _⟩ => i % 2 = 1).map Prod.snd |>.sum;
  (x - y) % 11 = 0

theorem largest_smallest_number : ∃ A_max A_min : ℕ,
  is_ten_digit_number A_max ∧ no_repeated_digits A_max ∧ divisible_by_11 A_max ∧
  is_ten_digit_number A_min ∧ no_repeated_digits A_min ∧ divisible_by_11 A_min ∧
  A_max = 9876524130 ∧ A_min = 1024375869 :=
sorry

end largest_smallest_number_l786_786248


namespace solve_z_l786_786779

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786779


namespace solve_complex_equation_l786_786790

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786790


namespace turtles_in_lake_l786_786175

-- Definitions based on conditions
def total_turtles : ℝ := 100
def percent_female : ℝ := 0.6
def percent_male : ℝ := 0.4
def percent_striped_male : ℝ := 0.25
def striped_turtle_babies : ℝ := 4
def percent_babies : ℝ := 0.4

-- Statement to prove
theorem turtles_in_lake : 
  (total_turtles * percent_male * percent_striped_male / percent_babies = striped_turtle_babies) →
  total_turtles = 100 :=
by
  sorry

end turtles_in_lake_l786_786175


namespace solve_for_y_l786_786129

theorem solve_for_y (y : ℝ) (h : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) : y = 53 / 3 := by
  sorry

end solve_for_y_l786_786129


namespace tan_add_pi_over_4_l786_786877

theorem tan_add_pi_over_4 (α : ℝ) (hα_range : α ∈ Ioo (π / 2) π) (h_sin : Real.sin α = 3 / 5) :
  Real.tan (α + π / 4) = 1 / 7 :=
sorry

end tan_add_pi_over_4_l786_786877


namespace length_bc_area_abc_l786_786331

theorem length_bc_area_abc :
  ∀ (A B C : ℝ × ℝ),
  (A = (2, 4)) ∧ 
  (B = (-12, 16)) ∧ 
  (C = (12, 16)) ∧
  (∀ x y, (x, y) ∈ ({B, C} : set (ℝ × ℝ)) → y = 16) ∧
  ((B.2 = 16) ∧ (C.2 = 16)) ∧
  ∀ (BC_area : ℝ), 
  (BC_area = 144) →
  (B.1 - C.1 = 2 * 12) ∧
  BC_area = (1 / 2) * abs(B.1 - C.1) * (16 - 4) :=
by
  intros A B C hA hB hC hBC hsx B_area ht
  have h1 : B.2 = 16 ∧ C.2 = 16 := by
    sorry
  have h2 : B.1 = -12 ∧ C.1 = 12 := by
    sorry
  have height : 12 := 16 - 4
  have base := 24
  have area := (1 / 2) * base * height
  calc
  base = 2 * 12  : by sorry
  and area = height : by sorry
  and 144 = area : by sorry


end length_bc_area_abc_l786_786331


namespace exam_maximum_marks_l786_786486

theorem exam_maximum_marks :
  (∃ M S E : ℕ, 
    (90 + 20 = 40 * M / 100) ∧ 
    (110 + 35 = 35 * S / 100) ∧ 
    (80 + 10 = 30 * E / 100) ∧ 
    M = 275 ∧ 
    S = 414 ∧ 
    E = 300) :=
by
  sorry

end exam_maximum_marks_l786_786486


namespace find_z_l786_786807

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786807


namespace arthur_money_left_l786_786696

theorem arthur_money_left {initial_amount spent_fraction : ℝ} (h_initial : initial_amount = 200) (h_fraction : spent_fraction = 4 / 5) : 
  (initial_amount - spent_fraction * initial_amount = 40) :=
by
  sorry

end arthur_money_left_l786_786696


namespace students_in_class_l786_786171

theorem students_in_class (S : ℕ) 
  (h1 : chess_students = S / 3)
  (h2 : tournament_students = chess_students / 2)
  (h3 : tournament_students = 4) : 
  S = 24 :=
by
  sorry

end students_in_class_l786_786171


namespace tan_angle_sum_l786_786411

variable (α β : ℝ)

theorem tan_angle_sum (h1 : Real.tan (α - Real.pi / 6) = 3 / 7)
                      (h2 : Real.tan (Real.pi / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
by
  sorry

end tan_angle_sum_l786_786411


namespace solve_complex_equation_l786_786789

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786789


namespace solve_for_z_l786_786858

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786858


namespace largest_initial_number_l786_786042

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786042


namespace problem_I_problem_II_problem_III_l786_786430

-- (I) Prove that if the radius of circle C is sqrt(3), then a = 2.
theorem problem_I (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + a = 0) ∧ (sqrt (5 - a) = sqrt 3) → a = 2 :=
by sorry

-- (II) Prove that if the length of chord AB is 6, then a = -6.
theorem problem_II (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + a = 0) ∧ (∃ P : ℝ × ℝ, P = (0,1)) ∧ (9 = (5 - a - 2)) → a = -6 :=
by sorry

-- (III) Prove that when a = 1, the length of the common chord MN of circles O and C is sqrt(11).
theorem problem_III :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧ (∀ x y : ℝ, x^2 + y^2 = 4) ∧ (∃ P, P = (2*x - 4*y + 5 = 0)) → 2 * sqrt(4 - 5/4) = sqrt 11 :=
by sorry

end problem_I_problem_II_problem_III_l786_786430


namespace sum_of_perimeters_of_triangle_AFD_l786_786112

theorem sum_of_perimeters_of_triangle_AFD (E F: ℝ) (AD AE ED AF EF x h: ℕ)
    (h1: E ∈ segment AD) (h2: AE = 15) (h3: ED = 25)
    (h4: AF = FD) (h5: AF = x) (h6: EF = y) 
    (h7: AF * AF = m * m) : 
    (s = 100) :=
begin
  sorry.
end

end sum_of_perimeters_of_triangle_AFD_l786_786112


namespace number_of_stamps_given_l786_786606

variable (x y : ℕ)

def initial_ratio (P Q : ℕ) : Prop := 7 * x = P ∧ 4 * x = Q

def new_ratio (P Q : ℕ) (y : ℕ) : Prop := 6 * (4 * x + y) = 5 * (7 * x - y)

def stamps_difference (P Q : ℕ) (y : ℕ) : Prop := (7 * x - y) = (4 * x + y) + 8 

theorem number_of_stamps_given (x y : ℕ) (h1 : initial_ratio x y) (h2 : new_ratio x y)
    (h3 : stamps_difference x y) :
    y = 8 := sorry

end number_of_stamps_given_l786_786606


namespace range_of_m_l786_786892

noncomputable def real : Type := ℝ

variable (m : real)

def p (m : real) : Prop := ∃ x : real, m * x^2 + 1 ≤ 0
def q (m : real) : Prop := ∀ x : real, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬ (p m ∨ q m)) : m ≥ 2 :=
sorry

end range_of_m_l786_786892


namespace distinct_shell_arrangements_l786_786073

def number_of_distinct_arrangements : Nat := 14.factorial / 14

theorem distinct_shell_arrangements : number_of_distinct_arrangements = 6227020800 := by
  have h1 : 14.factorial = 87178291200 := by norm_num
  have h2 : 87178291200 / 14 = 6227020800 := by norm_num
  rw [number_of_distinct_arrangements, h1, h2]
  norm_num

end distinct_shell_arrangements_l786_786073


namespace discount_percentage_is_25_l786_786155

-- Defining the conditions
def regular_rate : ℝ := 40.00
def number_of_mani_pedis : ℕ := 5
def total_discounted_cost : ℝ := 150.00

-- Statement of the problem
theorem discount_percentage_is_25 :
  let total_cost_without_discount := number_of_mani_pedis * regular_rate in
  let discount_amount := total_cost_without_discount - total_discounted_cost in
  let discount_percentage := (discount_amount / total_cost_without_discount) * 100 in
  discount_percentage = 25 :=
by
  sorry

end discount_percentage_is_25_l786_786155


namespace two_students_choose_materials_l786_786192

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786192


namespace sqrt_of_expression_l786_786716

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l786_786716


namespace salary_problem_l786_786614

theorem salary_problem :
  ∃ (Y X Z : ℝ), X = 1.10 * Y ∧ Z = 0.90 * Y ∧ (X + Y = 750) ∧ 
  X ≈ 392.86 ∧ Y ≈ 357.14 ∧ Z ≈ 321.43 :=
by
  sorry

end salary_problem_l786_786614


namespace expected_value_greater_than_median_l786_786303

noncomputable def pdf (f : ℝ → ℝ) (a b : ℝ) : Prop :=
(∀ x, x < a ∨ x ≥ b → f x = 0) ∧
(∀ x, a ≤ x ∧ x < b → 0 < f x) ∧
(∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 < b → f x1 ≥ f x2) ∧
(continuous_on f (set.Ico a b))

theorem expected_value_greater_than_median {a b : ℝ} {f : ℝ → ℝ} (hf : pdf f a b) :
  ∃ μ, is_median f μ ∧ ∫ x in set.Icc a b, x * f x > μ :=
begin
  sorry
end

end expected_value_greater_than_median_l786_786303


namespace visibility_times_l786_786305

-- Definitions based on conditions
def car_length : ℝ := 10
def num_cars : ℕ := 8
def loco_length : ℝ := 24
def fast_train_speed_kmh : ℝ := 60
def pass_train_speed_kmh : ℝ := 42

-- Convert speeds to meters per second
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

def fast_train_speed_mps := kmh_to_mps fast_train_speed_kmh
def pass_train_speed_mps := kmh_to_mps pass_train_speed_kmh

-- Calculate total length of the fast train
def total_length : ℝ := num_cars * car_length + loco_length

noncomputable def time_stationary : ℝ := total_length / fast_train_speed_mps
noncomputable def time_opposite : ℝ := total_length / (fast_train_speed_mps + pass_train_speed_mps)
noncomputable def time_same_direction : ℝ := total_length / (fast_train_speed_mps - pass_train_speed_mps)

theorem visibility_times :
  time_stationary = 6.24 ∧
  time_opposite = 3.67 ∧
  time_same_direction = 20.8 :=
by
  sorry

end visibility_times_l786_786305


namespace problem1_problem2_l786_786095

noncomputable def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

noncomputable def g' (x : ℝ) : ℝ := 15 * x^4 - 16 * x^3 + 6 * x^2 - 56 * x + 15

theorem problem1 : g 6 = 17568 := 
by {
  sorry
}

theorem problem2 : g' 6 = 15879 := 
by {
  sorry
}

end problem1_problem2_l786_786095


namespace balls_in_indistinguishable_boxes_l786_786457

theorem balls_in_indistinguishable_boxes : 
    ∀ (balls boxes : ℕ), balls = 7 ∧ boxes = 3 →
    (∑ d in finset.range (balls + 1), nat.choose balls d) = 64 := by
  intro balls boxes h
  sorry

end balls_in_indistinguishable_boxes_l786_786457


namespace solve_for_x_l786_786562

theorem solve_for_x (x : ℝ) : (x - 6)^4 = (1/16)^(-1) → x = 8 := by
  sorry

end solve_for_x_l786_786562


namespace solve_for_z_l786_786821

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786821


namespace sum_neighbors_invariant_l786_786105

open Finset

def is_neighbor (a b : Fin (8 × 8)) : Prop :=
  (abs (a.1 - b.1) ≤ 1) ∧ (abs (a.2 - b.2) ≤ 1) ∧ (a ≠ b)

def blank_neighbors_sum (board : Fin (8 × 8) → ℕ) : ℕ :=
  ∑ s in univ, board s

theorem sum_neighbors_invariant :
  ∃₀ board : Fin (8 × 8) → ℕ, 
  (∀ s, board s = ∑ t in univ.filter (is_neighbor s), if board t = 0 then 1 else 0) → 
  blank_neighbors_sum board = 210 :=
by sorry

end sum_neighbors_invariant_l786_786105


namespace trajectory_center_of_C_number_of_lines_l_l786_786416

noncomputable def trajectory_equation : Prop :=
  ∃ (a b : ℝ), a = 4 ∧ b^2 = 12 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def line_count : Prop :=
  ∀ (k m : ℤ), 
  ∃ (num_lines : ℕ), 
  (∀ (x : ℝ), (3 + 4 * k^2) * x^2 + 8 * k * m * x + 4 * m^2 - 48 = 0 → num_lines = 9 ∨ num_lines = 0) ∧
  (∀ (x : ℝ), (3 - k^2) * x^2 - 2 * k * m * x - m^2 - 12 = 0 → num_lines = 9 ∨ num_lines = 0)

theorem trajectory_center_of_C :
  trajectory_equation :=
sorry

theorem number_of_lines_l :
  line_count :=
sorry

end trajectory_center_of_C_number_of_lines_l_l786_786416


namespace sum_and_product_of_conjugates_l786_786957

theorem sum_and_product_of_conjugates (c d : ℚ) 
  (h1 : 2 * c = 6)
  (h2 : c^2 - 4 * d = 4) :
  c + d = 17 / 4 :=
by
  sorry

end sum_and_product_of_conjugates_l786_786957


namespace find_z_l786_786805

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786805


namespace area_of_triangle_PF1F2_l786_786518

theorem area_of_triangle_PF1F2
  (a b c : ℝ)
  (h_ellipse : a^2 / 49 + b^2 / 24 = 1)
  (F1 F2 : Point)
  (h_foci : dist F1 (0, 0) = 5 ∧ dist F2 (0, 0) = 5)
  (P : Point)
  (h_point : P.1 ^ 2 / 49 + P.2 ^ 2 / 24 = 1)
  (h_ratio : dist P F1 / dist P F2 = 4 / 3) :
  area (triangle (P, F1, F2)) = 24 :=
sorry

end area_of_triangle_PF1F2_l786_786518


namespace largest_initial_number_l786_786021

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786021


namespace problem_statement_l786_786114

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end problem_statement_l786_786114


namespace solve_for_z_l786_786828

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786828


namespace AC_over_AB_l786_786970

variables {A B C D F O : Type*} [IsoscelesTriangle ABC] [AngleBisectors BD AF]

def ratio_area_triangle_DOA_BOAF (ABC : IsoscelesTriangle) (BD AF : AngleBisectors) (O : Point) : Prop :=
  let area_ratio : ℝ :=
    (Triangle.area DOA) / (Triangle.area BOF)
  in area_ratio = 3 / 8

theorem AC_over_AB (ABC : IsoscelesTriangle) (BD AF : AngleBisectors) (O : Point) 
  (h1 : ratio_area_triangle_DOA_BOAF ABC BD AF O) :
  (AC / AB) = 1 / 2 :=
  sorry

end AC_over_AB_l786_786970


namespace equilateral_triangle_properties_l786_786718

theorem equilateral_triangle_properties 
  (A B C P : Type) [triangle A B C] 
  (h_eq : is_equilateral A B C)
  (h_centroid : is_centroid P A B C) :
  (∀ T1 T2 : Type, 
     (T1 = triangle_A_BP ∨ T1 = triangle_BP_A ∨ T1 = triangle_B_CP ∨ T1 = triangle_CP_B ∨ T1 = triangle_C_AP ∨ T1 = triangle_AP_C) →
     (T2 = triangle_A_BP ∨ T2 = triangle_BP_A ∨ T2 = triangle_B_CP ∨ T2 = triangle_CP_B ∨ T2 = triangle_C_AP ∨ T2 = triangle_AP_C) →
     (T1 ≠ T2 → congruent T1 T2 ∨ equal_area T1 T2)) :=
sorry

end equilateral_triangle_properties_l786_786718


namespace vector_parallel_calculate_l786_786448

variable {α : Type*} [RealField α]
noncomputable def vector_a (α : α) : α × α := (1, sin α)
noncomputable def vector_b (α : α) : α × α := (2, cos α)

def parallel (a b : α × α) : Prop := a.1 * b.2 = a.2 * b.1

theorem vector_parallel_calculate (α : α) (h : parallel (vector_a α) (vector_b α)) :
  (sin α + 2 * cos α) / (cos α - 3 * sin α) = -5 :=
  by
    sorry

end vector_parallel_calculate_l786_786448


namespace sum_first_six_terms_l786_786899

-- Definitions of arithmetic sequence properties
def arith_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- The sum of the first n terms of an arithmetic sequence
def sum_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (finset.range n).sum a

-- The given conditions
def a : ℕ → ℤ := λ n, 6 + n * (-2)  -- Since a_1 is 6 and d = -2
lemma a1_is_6 : a 1 = 6 := by simp [a]
lemma a3_a5_zero : a 3 + a 5 = 0 := by simp [a]

-- Proving the main result
theorem sum_first_six_terms : sum_n a 6 = 6 :=
by
  simp [sum_n, a]
  sorry

end sum_first_six_terms_l786_786899


namespace maximum_value_of_m_solve_inequality_l786_786758

theorem maximum_value_of_m (a b : ℝ) (h : a ≠ 0) : 
  ∃ m : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧ (m = 2) :=
by
  use 2
  sorry

theorem solve_inequality (x : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| ≤ 2 → (1/2 ≤ x ∧ x ≤ 5/2)) :=
by
  sorry

end maximum_value_of_m_solve_inequality_l786_786758


namespace minimum_value_fraction_l786_786475

variable (x y : ℝ)

-- Conditions
def conditions := x > y ∧ y > 0 ∧ log 2 x + log 2 y = 1

-- Statement
theorem minimum_value_fraction (h : conditions x y) : 
  ∀ x y, ((x^2 + y^2) / (x - y)) = 4 := 
sorry

end minimum_value_fraction_l786_786475


namespace min_value_geometric_sequence_l786_786420

theorem min_value_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)
  (h_geometric : ∃ q a1, ∀ k, a k = a1 * q ^ k . fromReal)
  (h_S : ∀ n, S n = a 1 * (1 - (h_geometric.snd.snd) ^ n) / (1 - (h_geometric.snd.snd))) 
  (h_initial : a 2 = 2 ∧ a 5 = 16) :
  ∃ n, (S (2 * n) + S n + 18) / 2^n = 9 :=
by
  sorry

end min_value_geometric_sequence_l786_786420


namespace new_average_l786_786888

theorem new_average (L : List ℝ) (hL : L.length = 10) (h_sum : L.sum = 200) :
  let new_L := 
    [2 * L[0], 
     3 * L[1], 
     4 * L[2], 
     5 * L[3], 
     6 * L[4], 
     L[5] / 2, 
     L[6] / 3, 
     L[7] / 4, 
     L[8] / 5, 
     L[9] / 6] in
  new_L.sum / 10 = 25 := 
by
  sorry

end new_average_l786_786888


namespace baseball_card_value_decrease_l786_786278

theorem baseball_card_value_decrease (initial_value : ℝ) :
  (1 - 0.70 * 0.90) * 100 = 37 := 
by sorry

end baseball_card_value_decrease_l786_786278


namespace cubic_root_conditions_l786_786588

noncomputable def isRoot (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

theorem cubic_root_conditions :
  ∃ a b c t : ℝ, 
    (∀ x : ℝ, isRoot (λ x, x^3 + a*x^2 + b*x + c) x ↔ x = t ∨ x = 0) 
    ∧ (t + 0 + 0 = -a)
    ∧ (t*0 + 0*t + t*0 = b)
    ∧ (t*0*0 = -c)
    ∧ (c = a * b)
    ∧ (b = 0) :=
sorry

end cubic_root_conditions_l786_786588


namespace waiting_time_boarding_l786_786106

noncomputable def time_taken_uber_to_house : ℕ := 10
noncomputable def time_taken_uber_to_airport : ℕ := 5 * time_taken_uber_to_house
noncomputable def time_taken_bag_check : ℕ := 15
noncomputable def time_taken_security : ℕ := 3 * time_taken_bag_check
noncomputable def total_process_time : ℕ := 180
noncomputable def remaining_time : ℕ := total_process_time - (time_taken_uber_to_house + time_taken_uber_to_airport + time_taken_bag_check + time_taken_security)
noncomputable def time_before_takeoff (B : ℕ) := 2 * B

theorem waiting_time_boarding : ∃ B : ℕ, B + time_before_takeoff B = remaining_time ∧ B = 20 := 
by 
  sorry

end waiting_time_boarding_l786_786106


namespace dad_strawberries_weight_l786_786104

variable (weight_Marco : ℕ) (total_weight : ℕ)

-- Given conditions
def Marco_strawberries_weight := (weight_Marco = 8)
def Total_strawberries_weight := (total_weight = 40)

-- Proof statement
theorem dad_strawberries_weight (weight_Marco := 8) (total_weight := 40) :
  Marco_strawberries_weight weight_Marco →
  Total_strawberries_weight total_weight →
  ∃ weight_dad : ℕ, weight_dad = 32 :=
by
  intros hMarco hTotal
  use total_weight - weight_Marco
  rw [hTotal, hMarco]
  norm_num
  sorry

end dad_strawberries_weight_l786_786104


namespace original_integer_if_and_only_if_fractional_sum_integer_l786_786393

variable (a b c : ℝ)

def greatest_integer_le (x : ℝ) : ℤ := Int.floor x

def fractional_part (x : ℝ) : ℝ := x - greatest_integer_le x

theorem original_integer_if_and_only_if_fractional_sum_integer
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) :
  (∃ k : ℤ, ∑ cyc [a, b, c], a ^ 3 / ((a - b) * (a - c)) = k) ↔
  (∃ m : ℤ, fractional_part a + fractional_part b + fractional_part c = m) :=
sorry

end original_integer_if_and_only_if_fractional_sum_integer_l786_786393


namespace chrysanthemum_arrangements_l786_786337

theorem chrysanthemum_arrangements :
  let varieties := ['A', 'B', 'C', 'D', 'E', 'F'] in
  let arrangements := { l : List Char // l.length = 6 ∧ l.nodup } in
  ∃(count: Nat), count = 480 ∧ 
  ∀ (l ∈ arrangements), 
   (list.index_of 'A' l < list.index_of 'C' l ∧ list.index_of 'B' l < list.index_of 'C' l) ∨ 
   (list.index_of 'A' l > list.index_of 'C' l ∧ list.index_of 'B' l > list.index_of 'C' l) → 
   l ∈ arrangements ∧
   ↑count = ∑ l in arrangements, if (list.index_of 'A' l < list.index_of 'C' l ∧ 
   list.index_of 'B' l < list.index_of 'C' l) ∨ 
   (list.index_of 'A' l > list.index_of 'C' l ∧ 
   list.index_of 'B' l > list.index_of 'C' l) then 1 else 0 :=
by
  sorry

end chrysanthemum_arrangements_l786_786337


namespace solve_for_x_l786_786567

theorem solve_for_x (x : ℝ) : (x - 6)^4 = (1 / 16)⁻¹ → x = 8 :=
by
  intro h
  have h1 : (1 / 16)⁻¹ = 16 := by norm_num
  rw [h1] at h
  have h2 : (x - 6)^4 = 16 := h
  have h3 : (x - 6) = 2 := by
    apply real.rpow_lt_rpow_iff
    norm_num
    exact h2
  linarith

end solve_for_x_l786_786567


namespace pure_imaginary_value_l786_786903

theorem pure_imaginary_value (a : ℝ) (ha : (a + complex.I) / (1 - complex.I) = complex.I * (a + 1) / 2) : a = 1 :=
sorry

end pure_imaginary_value_l786_786903


namespace problem_P_A_union_B_complement_l786_786429

variables (A B : Set ℝ) (PA PB : ℝ)

def mutually_exclusive (A B : Set ℝ) : Prop := A ∩ B = ∅
def complementary (B : Set ℝ) : Set ℝ := Bᶜ
def probability (s : Set ℝ) (p : ℝ) : Prop := p = P(s)

-- Given conditions
axiom h1 : mutually_exclusive A B
axiom h2 : probability A 0.6
axiom h3 : probability B 0.2

-- Goal
theorem problem_P_A_union_B_complement : probability (A ∪ complementary B) 0.8 :=
sorry

end problem_P_A_union_B_complement_l786_786429


namespace range_tan_squared_plus_tan_plus_one_l786_786753

theorem range_tan_squared_plus_tan_plus_one :
  (∀ y, ∃ x : ℝ, x ≠ (k : ℤ) * Real.pi + Real.pi / 2 → y = Real.tan x ^ 2 + Real.tan x + 1) ↔ 
  ∀ y, y ∈ Set.Ici (3 / 4) :=
sorry

end range_tan_squared_plus_tan_plus_one_l786_786753


namespace temperature_on_Friday_l786_786573

-- Definitions of the temperatures on the days
variables {M T W Th F : ℝ}

-- Conditions given in the problem
def avg_temp_mon_thu (M T W Th : ℝ) : Prop := (M + T + W + Th) / 4 = 48
def avg_temp_tue_fri (T W Th F : ℝ) : Prop := (T + W + Th + F) / 4 = 46
def temp_mon (M : ℝ) : Prop := M = 44

-- Statement to prove
theorem temperature_on_Friday (h1 : avg_temp_mon_thu M T W Th)
                               (h2 : avg_temp_tue_fri T W Th F)
                               (h3 : temp_mon M) : F = 36 :=
sorry

end temperature_on_Friday_l786_786573


namespace sin_add_7pi_over_6_eq_neg_four_fifths_l786_786410

theorem sin_add_7pi_over_6_eq_neg_four_fifths (a : ℝ)
  (h : cos(a - π / 6) + sin(a) = 4 * sqrt 3 / 5) :
  sin(a + 7 * π / 6) = -4 / 5 := 
sorry

end sin_add_7pi_over_6_eq_neg_four_fifths_l786_786410


namespace normals_to_parabola_l786_786096

noncomputable def count_normals (p a b : ℝ) : ℕ :=
if 4 * (2 * p - a)^2 + 27 * p * b^2 > 0 then 3
else if 4 * (2 * p - a)^2 + 27 * p * b^2 = 0 then 1
else 2

theorem normals_to_parabola (p a b : ℝ) (hp : 0 < p) :
  let S := λ x y : ℝ, y^2 = 4 * p * x in
  let P := (a, b) in
  1 ≤ count_normals p a b ∧ count_normals p a b ≤ 3 :=
by
  sorry

end normals_to_parabola_l786_786096


namespace integral_evaluation_l786_786700

theorem integral_evaluation :
  ∫ x in 0..6, (e^(Real.sqrt ((6 - x) / (6 + x))) / ((6 + x) * Real.sqrt (36 - x^2))) = (e - 1) / 6 :=
by
  sorry

end integral_evaluation_l786_786700


namespace smallest_and_largest_values_l786_786471

theorem smallest_and_largest_values (x : ℕ) (h : x < 100) :
  (x ≡ 2 [MOD 3]) ∧ (x ≡ 2 [MOD 4]) ∧ (x ≡ 2 [MOD 5]) ↔ (x = 2 ∨ x = 62) :=
by
  sorry

end smallest_and_largest_values_l786_786471


namespace probability_genuine_second_given_first_l786_786270

theorem probability_genuine_second_given_first :
  let total_items := 10
  let genuine_items := 6
  let defective_items := 4
  let first_genuine := 1
  P(first_genuine = 1) → 
  P(draw_second_genuine_given_first) = 5 / 9 :=
  sorry

end probability_genuine_second_given_first_l786_786270


namespace solve_for_z_l786_786771

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786771


namespace joe_prob_at_least_two_diff_fruits_l786_786509

noncomputable def probabilityAtLeastTwoDifferentFruits : ℝ :=
  1 - (4 * ((1 / 4) ^ 3))

theorem joe_prob_at_least_two_diff_fruits :
  probabilityAtLeastTwoDifferentFruits = 15 / 16 :=
by
  sorry

end joe_prob_at_least_two_diff_fruits_l786_786509


namespace find_a_l786_786423

/-- Define points A and B -/
structure Point (α : Type*) :=
(x : α)
(y : α)

def A (a : ℝ) : Point ℝ := ⟨a, 4⟩
def B (a : ℝ) : Point ℝ := ⟨-1, a⟩

/-- Define the slope between two points -/
def slope (P Q : Point ℝ) : ℝ :=
(P.y - Q.y) / (P.x - Q.x)

/-- Define the angle of slope is 45 degrees, which implies slope is 1 -/
def slope_45_degrees : ℝ := 1

/-- The theorem we want to prove -/
theorem find_a (a : ℝ) (H : slope (A a) (B a) = slope_45_degrees) : a = 3/2 :=
by
  sorry

end find_a_l786_786423


namespace surface_area_of_stacked_cubes_is_correct_l786_786121

-- Definition of the cubes' volumes
def cube_volumes : List ℕ := [1, 8, 27, 64, 125, 216, 343]

-- The surface area calculations and expected results 
noncomputable def stacked_surface_area (volumes : List ℕ) : ℕ :=
  let side_lengths := volumes.map (λ v => Nat.cbrt v) -- Compute side lengths
  let individual_surface_area := side_lengths.map (λ n => 6 * n^2) -- Surface areas of individual cubes
  let total_original_surface_area := individual_surface_area.sum
  let overlapping_areas := (side_lengths.init.map (λ n => n^2)).sum -- Overlapping areas (init excludes last)
  total_original_surface_area - 2 * overlapping_areas -- Subtracting overlapping areas

-- Theorem statement
theorem surface_area_of_stacked_cubes_is_correct :
  stacked_surface_area cube_volumes = 658 :=
  sorry

end surface_area_of_stacked_cubes_is_correct_l786_786121


namespace distinct_shell_arrangements_l786_786074

def number_of_distinct_arrangements : Nat := 14.factorial / 14

theorem distinct_shell_arrangements : number_of_distinct_arrangements = 6227020800 := by
  have h1 : 14.factorial = 87178291200 := by norm_num
  have h2 : 87178291200 / 14 = 6227020800 := by norm_num
  rw [number_of_distinct_arrangements, h1, h2]
  norm_num

end distinct_shell_arrangements_l786_786074


namespace inv_matrix_eq_l786_786431

variable (a : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 1, a])
variable (A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![a, -3; -1, a])

theorem inv_matrix_eq : (A⁻¹ = A_inv) → (a = 2) := 
by 
  sorry

end inv_matrix_eq_l786_786431


namespace num_solutions_to_quadratic_l786_786760

theorem num_solutions_to_quadratic (x : ℤ) :
  (40 < (x^2 + 8 * x + 16)) ∧ ((x^2 + 8 * x + 16) < 80) →
  2 := sorry

end num_solutions_to_quadratic_l786_786760


namespace highest_score_runs_l786_786289

theorem highest_score_runs 
  (avg : ℕ) (innings : ℕ) (total_runs : ℕ) (H L : ℕ)
  (diff_HL : ℕ) (excl_avg : ℕ) (excl_innings : ℕ) (excl_total_runs : ℕ) :
  avg = 60 → innings = 46 → total_runs = avg * innings →
  diff_HL = 180 → excl_avg = 58 → excl_innings = 44 → 
  excl_total_runs = excl_avg * excl_innings →
  H - L = diff_HL →
  total_runs = excl_total_runs + H + L →
  H = 194 :=
by
  intros h_avg h_innings h_total_runs h_diff_HL h_excl_avg h_excl_innings h_excl_total_runs h_H_minus_L h_total_eq
  sorry

end highest_score_runs_l786_786289


namespace solve_for_z_l786_786773

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786773


namespace fraction_irreducible_iff_l786_786762

-- Define the condition for natural number n
def is_natural (n : ℕ) : Prop :=
  True  -- All undergraduate natural numbers abide to True

-- Main theorem formalized in Lean 4
theorem fraction_irreducible_iff (n : ℕ) :
  (∃ (g : ℕ), g = 1 ∧ (∃ a b : ℕ, 2 * n * n + 11 * n - 18 = a * g ∧ n + 7 = b * g)) ↔ 
  (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end fraction_irreducible_iff_l786_786762


namespace stations_visited_l786_786346

-- Define the total number of nails
def total_nails : ℕ := 560

-- Define the number of nails left at each station
def nails_per_station : ℕ := 14

-- Main theorem statement
theorem stations_visited : total_nails / nails_per_station = 40 := by
  sorry

end stations_visited_l786_786346


namespace tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l786_786283

structure Tetrahedron :=
  (faces : Nat := 4)
  (vertices : Nat := 4)
  (valence : Nat := 3)
  (face_shape : String := "triangular")

structure Cube :=
  (faces : Nat := 6)
  (vertices : Nat := 8)
  (valence : Nat := 3)
  (face_shape : String := "square")

structure Octahedron :=
  (faces : Nat := 8)
  (vertices : Nat := 6)
  (valence : Nat := 4)
  (face_shape : String := "triangular")

structure Dodecahedron :=
  (faces : Nat := 12)
  (vertices : Nat := 20)
  (valence : Nat := 3)
  (face_shape : String := "pentagonal")

structure Icosahedron :=
  (faces : Nat := 20)
  (vertices : Nat := 12)
  (valence : Nat := 5)
  (face_shape : String := "triangular")

theorem tetrahedron_is_self_dual:
  Tetrahedron := by
  sorry

theorem cube_is_dual_to_octahedron:
  Cube × Octahedron := by
  sorry

theorem dodecahedron_is_dual_to_icosahedron:
  Dodecahedron × Icosahedron := by
  sorry

end tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l786_786283


namespace solve_for_z_l786_786825

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786825


namespace clock_angle_3_to_7_l786_786302

theorem clock_angle_3_to_7 : 
  let number_of_rays := 12
  let total_degrees := 360
  let degree_per_ray := total_degrees / number_of_rays
  let angle_3_to_7 := 4 * degree_per_ray
  angle_3_to_7 = 120 :=
by
  sorry

end clock_angle_3_to_7_l786_786302


namespace positive_difference_of_roots_l786_786629

theorem positive_difference_of_roots :
  let f := λ r : ℝ, (r^2 - 5*r - 26) / (r + 5) - 3*r - 8
  let discriminant := 10^2 - 4 * 1 * 33
  (-10 + real.sqrt discriminant) / 2 - (-10 - real.sqrt discriminant) / 2 = 8 := by
  sorry

end positive_difference_of_roots_l786_786629


namespace smallest_angle_in_20_sided_polygon_is_143_l786_786578

theorem smallest_angle_in_20_sided_polygon_is_143
  (n : ℕ)
  (h_n : n = 20)
  (angles : ℕ → ℕ)
  (h_convex : ∀ i, 1 ≤ i → i ≤ n → angles i < 180)
  (h_arithmetic_seq : ∃ d : ℕ, ∀ i, 1 ≤ i → i < n → angles (i + 1) = angles i + d)
  (h_increasing : ∀ i, 1 ≤ i → i < n → angles (i + 1) > angles i)
  (h_sum : ∑ i in finset.range n, angles (i + 1) = (n - 2) * 180) :
  angles 1 = 143 :=
by
  sorry

end smallest_angle_in_20_sided_polygon_is_143_l786_786578


namespace tanya_addition_problem_l786_786035

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786035


namespace limit_sin_cos_exponential_l786_786702

-- Mathematically equivalent proof problem in Lean 4 statement.
theorem limit_sin_cos_exponential :
  (∀ (f g : ℝ → ℝ), (∀ x, f x = 1 + sin x * cos (2 * x)) → (∀ x, g x = 1 + sin x * cos (3 * x)) → 
    ((λ x, (f x / g x) ^ (1 / sin x ^ 3)) ⟶ (exp (-5 / 2)) [at 0])) :=
by
  sorry

end limit_sin_cos_exponential_l786_786702


namespace production_days_l786_786285

theorem production_days (n : ℕ) (h1 : (40 * n + 90) / (n + 1) = 45) : n = 9 :=
by
  sorry

end production_days_l786_786285


namespace solve_z_l786_786782

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786782


namespace minutes_until_sunset_l786_786541

noncomputable def minutes_after_midnight (h : Nat) (m : Nat) : Nat :=
h * 60 + m

def sunset_time (initial_time : Nat) (days : Nat) (delay_per_day : Nat) : Nat :=
initial_time + days * delay_per_day

def time_until_sunset (current_time : Nat) (sunset_at : Nat) : Nat :=
if sunset_at > current_time then sunset_at - current_time else 0

theorem minutes_until_sunset 
  (initial_sunset_time : Nat := minutes_after_midnight 18 0) -- 6 PM is 1080 minutes after midnight
  (delay_per_day : Nat := 1.2)
  (days_passed : Nat := 40)
  (current_time : Nat := minutes_after_midnight 18 10) -- 6:10 PM is 1110 minutes after midnight
 :
  time_until_sunset current_time (sunset_time initial_sunset_time days_passed delay_per_day) = 38 :=
by
  sorry

end minutes_until_sunset_l786_786541


namespace solve_for_z_l786_786872

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786872


namespace tetrahedron_sphere_surface_area_l786_786681

noncomputable def circumsphere_surface_area (edge_length : ℝ) : ℝ :=
  let radius := (edge_length / 2) * (sqrt 3)
  4 * π * radius ^ 2

theorem tetrahedron_sphere_surface_area:
  circumsphere_surface_area (sqrt 2) = 3 * π :=
by
  sorry

end tetrahedron_sphere_surface_area_l786_786681


namespace num_ways_choose_materials_l786_786204

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786204


namespace train_speed_l786_786666

-- Define the platform length in meters and the time taken to cross in seconds
def platform_length : ℝ := 260
def time_crossing : ℝ := 26

-- Define the length of the goods train in meters
def train_length : ℝ := 260.0416

-- Define the total distance covered by the train when crossing the platform
def total_distance : ℝ := platform_length + train_length

-- Define the speed of the train in meters per second
def speed_m_s : ℝ := total_distance / time_crossing

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the speed of the train in kilometers per hour
def speed_km_h : ℝ := speed_m_s * conversion_factor

-- State the theorem to be proved
theorem train_speed : speed_km_h = 72.00576 :=
by
  sorry

end train_speed_l786_786666


namespace complement_union_l786_786446

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 4}

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {2, 4}) (hB : B = {1, 4}) :
  (U \ (A ∪ B)) = {3} :=
by
  simp [hU, hA, hB]
  sorry

end complement_union_l786_786446


namespace problem_l786_786466

def f (x : ℝ) : ℝ := sorry -- We assume f is defined as per the given condition but do not provide an implementation.

theorem problem (h : ∀ x : ℝ, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (Real.pi / 12)) = - (Real.sqrt 3) / 2 :=
by
  sorry -- The proof is omitted

end problem_l786_786466


namespace four_digit_numbers_using_0_and_9_l786_786536

theorem four_digit_numbers_using_0_and_9 :
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d, d ∈ Nat.digits 10 n → (d = 0 ∨ d = 9)} = {9000, 9009, 9090, 9099, 9900, 9909, 9990, 9999} :=
by
  sorry

end four_digit_numbers_using_0_and_9_l786_786536


namespace cyclist_arrives_first_l786_786622

variables (v d : ℝ)

-- Conditions
axiom cyclist_speed : v > 0
axiom total_distance : d > 0
axiom motorist_speed_accident : 5 * v > 0
axiom motorist_speed_walk : v / 2 > 0

-- Travel times for cyclist and motorist to determine who arrives first
theorem cyclist_arrives_first :
  let t_cyclist := d / v in
  let t_motorist := d / (10 * v) + d / v in
  t_cyclist < t_motorist :=
by
  sorry

end cyclist_arrives_first_l786_786622


namespace tanya_addition_problem_l786_786034

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786034


namespace find_z_l786_786820

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786820


namespace equal_functions_l786_786273

theorem equal_functions :
  ∀ (x : ℝ), (sqrt (x ^ 4) = x ^ 2) := by
  assume x,
  sorry

end equal_functions_l786_786273


namespace solve_z_l786_786784

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786784


namespace descending_order_numbers_count_l786_786749

theorem descending_order_numbers_count : 
  ∃ (n : ℕ), (n = 1013) ∧ 
  ∀ (x : ℕ), (∃ (xs : list ℕ), 
                (∀ i, i < xs.length - 1 → xs.nth_le i sorry > xs.nth_le (i+1) sorry) ∧ 
                nat_digits_desc xs ∧
                1 < xs.length) → 
             x ∈ nat_digits xs →
             ∃ (refs : list ℕ), n = refs.length ∧ 
             ∀ ref, ref ∈ refs → ref < x :=
sorry

end descending_order_numbers_count_l786_786749


namespace sqrt_product_eq_l786_786711

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l786_786711


namespace probability_three_out_of_five_dice_prime_l786_786699

noncomputable def probability_of_three_prime_dice : ℚ := 5 / 16

theorem probability_three_out_of_five_dice_prime :
  let dice_faces := 20
  let primes := {2, 3, 5, 7, 11, 13, 17, 19}
  let non_prime_probability := (dice_faces - primes.size : ℚ) / dice_faces
  let prime_probability := primes.size / dice_faces
  let combinations := (nat.choose 5 3)
  let probability_of_event := (prime_probability ^ 3) * (non_prime_probability ^ 2) * combinations
  in probability_of_event = probability_of_three_prime_dice :=
by
  sorry

end probability_three_out_of_five_dice_prime_l786_786699


namespace solve_for_z_l786_786829

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786829


namespace number_of_ways_to_choose_materials_l786_786237

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786237


namespace find_k_plus_l_l786_786982

noncomputable def triangle_condition : Prop :=
  ∃ (P Q R S : Type) (angle R P Q : ℕ) (PS : ℕ) (k l : ℕ),
    angle R = 90 ∧ PS = 17^3 ∧ coprime k l ∧ tan Q = k / l ∧ k + l = 161

theorem find_k_plus_l (P Q R S : Type) (angle R P Q : ℕ) (PS : ℕ) (k l : ℕ)
  (h1 : angle R = 90)
  (h2 : PS = 17^3)
  (h3 : coprime k l)
  (h4 : tan Q = k / l) :
  k + l = 161 :=
sorry

end find_k_plus_l_l786_786982


namespace sum_first_ten_multiples_of_13_l786_786268

theorem sum_first_ten_multiples_of_13 : 
  let n := 10 in 
  let sum_1_to_n := n * (n + 1) / 2 in 
  13 * sum_1_to_n = 715 :=
by
  let n := 10
  let sum_1_to_n := n * (n + 1) / 2
  show 13 * sum_1_to_n = 715
  -- Proof goes here
  sorry

end sum_first_ten_multiples_of_13_l786_786268


namespace tanya_addition_problem_l786_786029

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786029


namespace total_marbles_l786_786964

variable (red blue green yellow : ℕ)

def marble_ratio (red blue green yellow : ℕ) : Prop := red:blue:green:yellow = 1:3:4:2

def number_of_green_marble (green : ℕ) : Prop := green = 40

theorem total_marbles {red blue green yellow : ℕ} (h1 : marble_ratio red blue green yellow) (h2 : number_of_green_marble green) :
  red + blue + green + yellow = 100 :=
by 
  sorry

end total_marbles_l786_786964


namespace solve_for_x_l786_786559

/-- Prove that x = 8 under the condition that (x - 6)^4 = (1 / 16)^(-1) -/
theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)^(-1)) : x = 8 := 
by 
  sorry

end solve_for_x_l786_786559


namespace cone_base_radius_l786_786663

variables (r : ℝ) (h : ℝ)

def slant_height := 2 * r
def height := sqrt (4 * r * r - r * r)
def lateral_surface_area := π * r * slant_height
def volume := (1 / 3) * π * r * r * height

theorem cone_base_radius:
  (lateral_surface_area = (1 / 2) * volume) ->
    r = 4 * sqrt 3 :=
  by
  -- Assume that slant_height is defined as 2r
  have slant_height := 2 * r,
  -- Assume the height is defined using the Pythagorean theorem
  have height := sqrt (slant_height * slant_height - r * r),
  sorry

end cone_base_radius_l786_786663


namespace part1_part2_l786_786447

noncomputable def sn (a b n : ℕ) : ℕ := a^n + b^(n+1)

theorem part1 (a b : ℕ) (ha : 1 < a) (hb : 1 < b) : 
  ∃ᶠ n in at_top, ∃ k : ℕ, 1 < k ∧ sn a b n = k * k :=
sorry

theorem part2 (a b : ℕ) (ha : 1 < a) (hb : 1 < b) :
  ∃ᶠ p in at_top.filter is_prime, ∃ n : ℕ, p ∣ sn a b n :=
sorry

end part1_part2_l786_786447


namespace line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l786_786378

-- Definitions for the first condition
def P : ℝ × ℝ := (3, 2)
def passes_through_P (l : ℝ → ℝ) := l P.1 = P.2
def equal_intercepts (l : ℝ → ℝ) := ∃ a : ℝ, l a = 0 ∧ l (-a) = 0

-- Equation 1: Line passing through P with equal intercepts
theorem line_through_P_with_equal_intercepts :
  (∃ l : ℝ → ℝ, passes_through_P l ∧ equal_intercepts l ∧ 
   (∀ x y : ℝ, l x = y ↔ (2 * x - 3 * y = 0) ∨ (x + y - 5 = 0))) :=
sorry

-- Definitions for the second condition
def A : ℝ × ℝ := (-1, -3)
def passes_through_A (l : ℝ → ℝ) := l A.1 = A.2
def inclination_90 (l : ℝ → ℝ) := ∀ x : ℝ, l x = l 0

-- Equation 2: Line passing through A with inclination 90°
theorem line_through_A_with_inclination_90 :
  (∃ l : ℝ → ℝ, passes_through_A l ∧ inclination_90 l ∧ 
   (∀ x y : ℝ, l x = y ↔ (x + 1 = 0))) :=
sorry

end line_through_P_with_equal_intercepts_line_through_A_with_inclination_90_l786_786378


namespace point_P_in_fourth_quadrant_l786_786974

def point_in_which_quadrant (x : ℝ) (y : ℝ) : string :=
  if x > 0 ∧ y > 0 then "First"
  else if x < 0 ∧ y > 0 then "Second"
  else if x < 0 ∧ y < 0 then "Third"
  else if x > 0 ∧ y < 0 then "Fourth"
  else "On Axis"

theorem point_P_in_fourth_quadrant : point_in_which_quadrant 2023 (-2022) = "Fourth" := by
  sorry

end point_P_in_fourth_quadrant_l786_786974


namespace bag_contains_n_black_balls_l786_786297

theorem bag_contains_n_black_balls (n : ℕ) : (5 / (n + 5) = 1 / 3) → n = 10 := by
  sorry

end bag_contains_n_black_balls_l786_786297


namespace part_a_part_b_l786_786292

section BlockSimilarPolynomials

variable (n : ℕ) (hn : n ≥ 2)

def isBlockSimilar (P Q : ℝ[X]) (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → ∀ (k : ℕ), 0 ≤ k ∧ k < 2015,
    ∃ (perm : ℕ → ℕ), (perm k = (k + 1) % 2015) ∧ (P.eval (2015 * i - k) = Q.eval (2015 * i - perm k))

theorem part_a :
  ∃ (P Q : ℝ[X]), P ≠ Q ∧ P.degree = n + 1 ∧ Q.degree = n + 1 ∧ isBlockSimilar P Q n :=
sorry

theorem part_b :
  ¬ ∃ (P Q : ℝ[X]), P ≠ Q ∧ P.degree = n ∧ Q.degree = n ∧ isBlockSimilar P Q n :=
sorry

end BlockSimilarPolynomials

end part_a_part_b_l786_786292


namespace man_l786_786311

noncomputable def speed_in_still_water (current_speed_kmph : ℝ) (distance_m : ℝ) (time_seconds : ℝ) : ℝ :=
   let current_speed_mps := current_speed_kmph * 1000 / 3600
   let downstream_speed_mps := distance_m / time_seconds
   let still_water_speed_mps := downstream_speed_mps - current_speed_mps
   let still_water_speed_kmph := still_water_speed_mps * 3600 / 1000
   still_water_speed_kmph

theorem man's_speed_in_still_water :
  speed_in_still_water 6 100 14.998800095992323 = 18 := by
  sorry

end man_l786_786311


namespace max_effective_speed_at_1_and_3_l786_786690

def distance (hour: ℕ) : ℝ := 
  match hour with
  | 0 => 0
  | 1 => 100
  | 2 => 250
  | 3 => 350
  | 4 => 420
  | 5 => 500
  | _ => 0 -- Assuming data is only for the first 5 hours

def altitude (hour: ℕ) : ℝ := 
  match hour with
  | 0 => 0
  | 1 => 2000
  | 2 => 1000
  | 3 => 3000
  | 4 => 2000
  | 5 => 0
  | _ => 0

def effective_speed (n : ℕ) : ℝ :=
  if (n > 0 ∧ n ≤ 5) then 
    real.sqrt ((distance n - distance (n - 1)) ^ 2 + (altitude n - altitude (n - 1)) ^ 2)
  else 
    0

theorem max_effective_speed_at_1_and_3 :
  ∀ n: ℕ, (effective_speed 1 = effective_speed n) ∨ (effective_speed 3 = effective_speed n) ↔
    (effective_speed 2 ≤ effective_speed 1 ∧ effective_speed 4 ≤ effective_speed 1 ∧ effective_speed 5 ≤ effective_speed 1)
    ∧
    (effective_speed 2 ≤ effective_speed 3 ∧ effective_speed 4 ≤ effective_speed 3 ∧ effective_speed 5 ≤ effective_speed 3) :=
  sorry

end max_effective_speed_at_1_and_3_l786_786690


namespace two_students_one_common_material_l786_786199

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786199


namespace socks_difference_l786_786995

-- Definitions of the conditions
def week1 : ℕ := 12
def week2 (S : ℕ) : ℕ := S
def week3 (S : ℕ) : ℕ := (12 + S) / 2
def week4 (S : ℕ) : ℕ := (12 + S) / 2 - 3
def total (S : ℕ) : ℕ := week1 + week2 S + week3 S + week4 S

-- Statement of the theorem
theorem socks_difference (S : ℕ) (h : total S = 57) : S - week1 = 1 :=
by 
  -- Proof is not required
  sorry

end socks_difference_l786_786995


namespace solve_z_l786_786786

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786786


namespace ladder_distance_l786_786654

theorem ladder_distance (a b c : ℕ) (h1 : a = 12) (h2 : c = 13) (h3 : a^2 + b^2 = c^2) : b = 5 :=
by
  rw [h1, h2] at h3
  exact Nat.eq_of_succ_lt_succ
    (by norm_num at h3; assumption) sorry

end ladder_distance_l786_786654


namespace find_function_l786_786723

noncomputable def f (x : ℕ) : ℕ

theorem find_function (f : ℕ → ℕ) 
  (h1 : ∀ x y : ℕ, f x * f y ∣ (1 + 2 * x) * f y + (1 + 2 * y) * f x)
  (h2 : ∀ x y : ℕ, x < y → f x < f y) : 
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = 4 * x + 2) :=
sorry

end find_function_l786_786723


namespace balls_in_indistinguishable_boxes_l786_786458

theorem balls_in_indistinguishable_boxes : 
    ∀ (balls boxes : ℕ), balls = 7 ∧ boxes = 3 →
    (∑ d in finset.range (balls + 1), nat.choose balls d) = 64 := by
  intro balls boxes h
  sorry

end balls_in_indistinguishable_boxes_l786_786458


namespace largest_initial_number_l786_786063

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786063


namespace descending_digits_count_l786_786743

theorem descending_digits_count : 
  ∑ k in (finset.range 11).filter (λ k, 2 ≤ k), nat.choose 10 k = 1013 := 
sorry

end descending_digits_count_l786_786743


namespace line_properties_l786_786669

noncomputable def slope_angle_line (P Q : ℝ × ℝ) (d : ℝ) : Prop :=
  let L1 := (x : ℝ) = 2 in
  let L2 := (y : ℝ) = (sqrt 3 / 3) * (x - 2) in
  P = (2, 0) ∧
  Q = (-2, 4 * sqrt 3 / 3) ∧
  d = 4 ∧
  (distance P Q = d → slope_angle L1 = 90 ∧ equation L1 = "x - 2 = 0") ∧
  (distance P Q = d → slope_angle L2 = 30 ∧ equation L2 = "x - sqrt 3 y - 2 = 0")

theorem line_properties (P Q : ℝ × ℝ) (d : ℝ) :
  slope_angle_line P Q d := sorry

end line_properties_l786_786669


namespace solve_for_x_l786_786566

theorem solve_for_x (x : ℝ) : (x - 6)^4 = (1 / 16)⁻¹ → x = 8 :=
by
  intro h
  have h1 : (1 / 16)⁻¹ = 16 := by norm_num
  rw [h1] at h
  have h2 : (x - 6)^4 = 16 := h
  have h3 : (x - 6) = 2 := by
    apply real.rpow_lt_rpow_iff
    norm_num
    exact h2
  linarith

end solve_for_x_l786_786566


namespace max_min_values_on_interval_l786_786382

noncomputable def function_y (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

theorem max_min_values_on_interval :
  ∃ x₁ x₂ ∈ Icc (0 : ℝ) 2, 
    (∀ x ∈ Icc (0 : ℝ) 2, function_y x ≤ function_y x₁) ∧
    (∀ x ∈ Icc (0 : ℝ) 2, function_y x₂ ≤ function_y x) ∧
    function_y x₁ = 5/2 ∧
    function_y x₂ = 1/2 :=
by
  sorry

end max_min_values_on_interval_l786_786382


namespace possible_constant_ratios_l786_786009

noncomputable def is_arithmetic_seq (a : ℕ → ℕ) : Prop :=
∃ (a1 d : ℕ), ∀ n : ℕ, a n = a1 + (n-1) * d

noncomputable def constant_ratio (a : ℕ → ℕ) (r : ℚ) : Prop :=
∀ n : ℕ, a n / a (2 * n) = r

theorem possible_constant_ratios (a : ℕ → ℕ) (h_arith_seq : is_arithmetic_seq a) (h_constant_ratio : constant_ratio a (q : ℚ)) :
  set_of (λ r, constant_ratio a r) = {1, 1/2} := sorry

end possible_constant_ratios_l786_786009


namespace exists_set_b_l786_786348

-- Definitions of sets and the conditions
variables {n : ℕ}
variables (a : fin n → ℝ) -- finite set of vectors in R^n, represented as functions from finite indices to ℝ
variables (P : ℝ^3 → Prop) -- representing plane P as a proposition on ℝ^3

-- E is the set of non-negative linear combinations of vectors a_i
def E (a : fin n → ℝ^3) : set ℝ^3 :=
  {x | ∃ (λ : fin n → ℝ), (∀ i, 0 ≤ λ i) ∧ x = finset.univ.sum (λ i, λ i • a i)}

-- F includes all vectors in E and all vectors parallel to the plane P
def F (a : fin n → ℝ^3) (P : ℝ^3 → Prop) : set ℝ^3 :=
  {x | x ∈ E a ∨ (∃ (u v ∈ ℝ^3), P u ∧ P v ∧ x = u + v)}

-- Proof that F can be represented as a non-negative combination of some set of vectors {b_1, b_2, ..., b_p}
theorem exists_set_b {a : fin n → ℝ^3} {P : ℝ^3 → Prop} :
  ∃ (p : ℕ) (b : fin p → ℝ^3), ∀ (y : ℝ^3),
    (y ∈ F a P) ↔ (∃ (μ : fin p → ℝ), (∀ i, 0 ≤ μ i) ∧ y = finset.univ.sum (λ i, μ i • b i)) :=
sorry

end exists_set_b_l786_786348


namespace find_analytical_expression_of_f_l786_786901

theorem find_analytical_expression_of_f :
  (∃ (f : ℝ → ℝ), (∃ k b : ℝ, (f = λ x, k * x + b) ∧ 
    (∀ x, f (f x) = 4 * x - 1))) ↔ 
  (∃ (k b : ℝ), (k = 2 ∧ b = -1/3) ∨ (k = -2 ∧ b = 1)) :=
sorry

end find_analytical_expression_of_f_l786_786901


namespace ratio_sum_eq_four_l786_786099

variables (O G A1 B1 C1 D1 : Point)
variables (A B C D : Tetrahedron)

def is_centroid (G : Point) (A B C D : Tetrahedron) : Prop :=
  centroid_of_tetrahedron G A B C D = G

def line_intersects_face (O G : Point) (tet : Tetrahedron)
(face : Face) (P : Point) : Prop :=
  line_thru O G ⋂ face = {P}

theorem ratio_sum_eq_four
  (O G A1 B1 C1 D1 : Point)
  (t : Tetrahedron)
  (hG_centroid : is_centroid G t)
  (hA1_face : line_intersects_face O G t t.face1 A1)
  (hB1_face : line_intersects_face O G t t.face2 B1)
  (hC1_face : line_intersects_face O G t t.face3 C1)
  (hD1_face : line_intersects_face O G t t.face4 D1) :
  (dist A1 O) / (dist A1 G) + (dist B1 O) / (dist B1 G) +
  (dist C1 O) / (dist C1 G) + (dist D1 O) / (dist D1 G) = 4 := 
sorry

end ratio_sum_eq_four_l786_786099


namespace least_possible_value_of_f_l786_786477

theorem least_possible_value_of_f (x y z N : ℤ) (h1 : x < y) (h2 : y < z) 
    (h3 : y - x > 5) (h4 : even x) (h5 : x % 5 = 0) (h6 : odd y) (h7 : odd z) 
    (h8 : y ^ 2 + z ^ 2 = N) (h9 : N > 0) : 
    z - x = 9 :=
sorry

end least_possible_value_of_f_l786_786477


namespace number_of_descending_digit_numbers_l786_786746

theorem number_of_descending_digit_numbers : 
  (∑ k in Finset.range 8, Nat.choose 10 (k + 2)) + 1 = 1013 :=
by
  sorry

end number_of_descending_digit_numbers_l786_786746


namespace solve_for_z_l786_786776

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786776


namespace compute_expression_l786_786707

theorem compute_expression :
  3 * 3^4 - 9^60 / 9^57 = -486 :=
by
  sorry

end compute_expression_l786_786707


namespace log_diff_example_l786_786730

-- Definitions using the conditions in (a)
def log_base : ℕ → ℕ → ℤ
| b m => Int.log b m

axiom log_sub (b m n : ℕ) : b > 1 → m > 0 → n > 0 → log_base b m - log_base b n = log_base b (m / n)
axiom log_eq_one (b n : ℕ) : b > 1 → n = b → log_base b n = 1

-- Rewrite the problem statement in Lean
theorem log_diff_example : log_base 3 6 - log_base 3 2 = 1 :=
by
  have h₁ := log_sub 3 6 2 (dec_trivial) (dec_trivial) (dec_trivial)
  have h₂ : 6 / 2 = 3 := (dec_trivial : 6 / 2 = 3)
  rw [h₂] at h₁
  exact log_eq_one 3 3 (dec_trivial) (rfl)

end log_diff_example_l786_786730


namespace solve_for_z_l786_786871

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786871


namespace sum_of_table_equals_one_l786_786281

   -- Define conditions
   variables {m n : ℕ} (a : ℕ → ℕ → ℝ)
   def row_sum (i : ℕ) : ℝ := ∑ j in finset.range m, a i j
   def col_sum (j : ℕ) : ℝ := ∑ i in finset.range n, a i j

   -- Problem statement
   theorem sum_of_table_equals_one 
     (h : ∀ i j, a i j = row_sum a i * col_sum a j) : 
     (∑ i in finset.range n, ∑ j in finset.range m, a i j) = 1 := 
   sorry
   
end sum_of_table_equals_one_l786_786281


namespace value_of_m_l786_786404

theorem value_of_m :
  ∃ m : ℕ, 3 * 4 * 5 * m = fact 8 ∧ m = 672 :=
by
  use 672
  split
  sorry

end value_of_m_l786_786404


namespace tapB_fill_in_20_l786_786572

-- Conditions definitions
def tapA_rate (A: ℝ) : Prop := A = 3 -- Tap A fills 3 liters per minute
def total_volume (V: ℝ) : Prop := V = 36 -- Total bucket volume is 36 liters
def together_fill_time (t: ℝ) : Prop := t = 10 -- Both taps fill the bucket in 10 minutes

-- Tap B's rate can be derived from these conditions
def tapB_rate (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : Prop := V - (A * t) = B * t

-- The final question we need to prove
theorem tapB_fill_in_20 (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : 
  tapA_rate A → total_volume V → together_fill_time t → tapB_rate B A V t → B * 20 = 12 := by
  sorry

end tapB_fill_in_20_l786_786572


namespace num_ways_choose_materials_l786_786202

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786202


namespace calculate_second_rate_l786_786336

def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

def first_investment := 1800
def first_rate := 0.05
def first_time := 1
def first_interest := interest first_investment first_rate first_time -- 90

def second_investment := 2 * first_investment - 400 -- 3200
def total_interest := 298
def second_interest := total_interest - first_interest -- 208

theorem calculate_second_rate : 
  second_interest / (second_investment * first_time) = 0.065 := 
by 
  unfold interest first_investment first_rate first_time first_interest second_investment total_interest second_interest
  sorry

end calculate_second_rate_l786_786336


namespace amount_invested_is_6800_l786_786737

-- Define the parameters
def income : ℝ := 3000
def dividend_yield : ℝ := 0.60
def face_value : ℝ := 100
def market_price : ℝ := 136

-- Calculate the annual dividend per share
def annual_dividend_per_share : ℝ := dividend_yield * face_value

-- Define the theorem which states that the invested amount is $6800
theorem amount_invested_is_6800 (income eq_one: income = 3000) 
  (annual_dividend_per_share eq_two: annual_dividend_per_share = dividend_yield * face_value)
  (market_price eq_three: market_price = 136)
  : income / annual_dividend_per_share * market_price = 6800 :=
by 
  sorry

end amount_invested_is_6800_l786_786737


namespace value_of_business_l786_786640

variable (V : ℝ)
variable (h1 : (2 / 3) * V = S)
variable (h2 : (3 / 4) * S = 75000)

theorem value_of_business (h1 : (2 / 3) * V = S) (h2 : (3 / 4) * S = 75000) : V = 150000 :=
sorry

end value_of_business_l786_786640


namespace max_workers_l786_786304

variable {n : ℕ} -- number of workers on the smaller field
variable {S : ℕ} -- area of the smaller field
variable (a : ℕ) -- productivity of each worker

theorem max_workers 
  (h_area : ∀ large small : ℕ, large = 2 * small) 
  (h_workers : ∀ large small : ℕ, large = small + 4) 
  (h_inequality : ∀ (S : ℕ) (n a : ℕ), S / (a * n) > (2 * S) / (a * (n + 4))) :
  2 * n + 4 ≤ 10 :=
by
  -- h_area implies the area requirement
  -- h_workers implies the worker requirement
  -- h_inequality implies the time requirement
  sorry

end max_workers_l786_786304


namespace expectation_sum_ne_sum_expectation_l786_786764

open MeasureTheory

-- Define the indicator function
def indicator {α : Type*} (s : set α) [decidable_pred s] (x : α) : ℝ :=
if x ∈ s then 1 else 0

noncomputable def xi (n : ℕ) (U : ℝ) : ℝ :=
n * indicator {x : ℝ | n * x ≤ 1} U - (n - 1) * indicator {x : ℝ | (n - 1) * x ≤ 1} U

theorem expectation_sum_ne_sum_expectation (U : ℝ) (hU : U ∈ set.Icc 0.0 1.0) :
  ∃ (xi : ℕ → ℝ), (ℝ → ℝ) (∞) -> ¬(∑' n, (xi n)) = ∑' n, xi :=
begin
  sorry
end

end expectation_sum_ne_sum_expectation_l786_786764


namespace largest_initial_number_l786_786046

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786046


namespace permutations_count_l786_786385

open Finset

theorem permutations_count (b : Fin 7 → ℕ) :
  (∀ i : Fin 7, b i ∈ ({1, 2, 3, 4, 5, 6, 7} : Finset ℕ)) →
  (∃! (b : Fin 7 → ℕ) (hb : ∀ i : Fin 7, b i ∈ ({1, 2, 3, 4, 5, 6, 7} : Finset ℕ)), 
     (∏ i, (b i + i + 1) / 3) > fact 7)  :=
sorry

end permutations_count_l786_786385


namespace expr_is_irreducible_fraction_l786_786370

def a : ℚ := 3 / 2015
def b : ℚ := 11 / 2016

noncomputable def expr : ℚ := 
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a

theorem expr_is_irreducible_fraction : expr = 11 / 112 := by
  sorry

end expr_is_irreducible_fraction_l786_786370


namespace solve_for_z_l786_786865

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786865


namespace total_area_of_colored_paper_l786_786937

-- Definitions
def num_pieces : ℝ := 3.2
def side_length : ℝ := 8.5

-- Theorem statement
theorem total_area_of_colored_paper : 
  let area_one_piece := side_length * side_length
  let total_area := area_one_piece * num_pieces
  total_area = 231.2 := by
  sorry

end total_area_of_colored_paper_l786_786937


namespace sum_of_valid_m_values_l786_786476

theorem sum_of_valid_m_values : 
  (∀ x m, ( x - m ) / 6 ≥ 0 → x + 3 < 3 * ( x - 1 ) → x > 3 → m ≤ 3) →
  (∀ y m, ( 3 - y ) / (2 - y) = 3 - m / (y - 2) → 0 ≤ y → ∃ n, n ∈ ℤ∧ y = n) →
  ∑ m ∈ ({ m | ∃ y, (3 - y) / (2 - y) = 3 - m / (y - 2) ∧ 0 ≤ y ∧ y ∈ ℤ }), m = -1 :=
by
  sorry

end sum_of_valid_m_values_l786_786476


namespace interval_decreasing_l786_786381

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

def f_prime (x : ℝ) : ℝ := x * (2 * Real.log x + 1)

theorem interval_decreasing (x : ℝ) (h : 0 < x ∧ x < Real.sqrt Real.exp() / Real.exp()) :
  f_prime x < 0 :=
sorry

end interval_decreasing_l786_786381


namespace solve_z_l786_786850

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786850


namespace vertical_asymptote_l786_786947

theorem vertical_asymptote :
  ∀ (x : ℝ), (x = -5) → ∃ y, is_limit (λ y, y = (x^2 - 2*x + 12)/(x + 5)) ∞ := by
  sorry

end vertical_asymptote_l786_786947


namespace heptagon_exterior_angle_sum_l786_786593

-- Let a heptagon be defined as a polygon with exactly seven sides.
def is_heptagon (P : Finset ℕ) : Prop := P.card = 7

-- The sum of the exterior angles of any polygon is always 360 degrees.
axiom sum_of_exterior_angles (P : Finset ℕ) : 360° = ∑ i in P, exterior_angle i

-- Proof statement: Given that a polygon P is a heptagon, its exterior angle sum is 360 degrees.
theorem heptagon_exterior_angle_sum (P : Finset ℕ) (h : is_heptagon P) : ∑ i in P, exterior_angle i = 360° :=
by sorry

end heptagon_exterior_angle_sum_l786_786593


namespace price_per_large_bottle_l786_786511

theorem price_per_large_bottle 
  (P : ℝ) -- Price per large bottle.
  (price_per_small : ℝ := 1.35) -- Price per small bottle.
  (large_bottles : ℕ := 1375) -- Number of large bottles.
  (small_bottles : ℕ := 690) -- Number of small bottles.
  (average_price : ℝ := 1.6163438256658595) -- Given average price.
  (approx_price_per_large : ℝ := 1.74979773148) -- Correct answer.
  :
  P ≈ approx_price_per_large := 
by
  -- Given total number of bottles and the conditions, let's define the total cost, total bottles and average cost.
  let total_bottles := large_bottles + small_bottles
  let total_cost := large_bottles * P + small_bottles * price_per_small
  let average_cost := total_cost / total_bottles

  -- From here, the hypothesis should imply that the actual cost approximation is close to the given average cost
  -- Thus, proving P is approximately to the given correct answer derived above.
  have h1 : average_cost = average_price := sorry
  have h2 : total_cost = 1375 * P + 690 * 1.35 := sorry  
  have h3 : average_cost = (1375 * P + 931.5) / 2065 := sorry
  have h4 : 1375 * P + 931.5 ≈ 1.6163438256658595 * 2065 := sorry
  have h5 : P ≈ approx_price_per_large := sorry

  exact h5

end price_per_large_bottle_l786_786511


namespace find_other_endpoint_l786_786156

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ)
  (h_midpoint_x : x_m = (x_1 + x_2) / 2)
  (h_midpoint_y : y_m = (y_1 + y_2) / 2)
  (h_given_midpoint : x_m = 3 ∧ y_m = 0)
  (h_given_endpoint1 : x_1 = 7 ∧ y_1 = -4) :
  x_2 = -1 ∧ y_2 = 4 :=
sorry

end find_other_endpoint_l786_786156


namespace num_divisors_8820_mult_3_5_l786_786940

theorem num_divisors_8820_mult_3_5 : 
  let x := 8820;
  let count_divisors_multiples_3_5 := 
    (finset.range 3).card * 2 * 1 * (finset.range 3).card
  in
  count_divisors_multiples_3_5 = 18 := by
  let x := 8820
  let prime_factors := [2, 3, 5, 7]
  let powers := [2, 2, 1, 2]
  sorry

end num_divisors_8820_mult_3_5_l786_786940


namespace morgan_sats_percentage_improvement_l786_786539

variable (first_score second_score : ℕ)
variable (percentage_improvement : ℕ)

def calculate_improvement (first_score second_score : ℕ) : ℕ :=
  second_score - first_score

def calculate_percentage_improvement (improvement first_score : ℕ) : ℕ :=
  (improvement * 100) / first_score

theorem morgan_sats_percentage_improvement :
  first_score = 1000 → second_score = 1100 → percentage_improvement = 10 → 
  calculate_percentage_improvement (calculate_improvement first_score second_score) first_score = percentage_improvement :=
by
  intros h1 h2 h3
  rw [h1, h2]
  simp
  rw h3
  rfl

end morgan_sats_percentage_improvement_l786_786539


namespace proofProblem_answer_l786_786928

noncomputable def proofProblem : Prop :=
  ∀ (a b : ℝ) (θ : ℝ), a < 0 → b < 0 → a^2 + b^2 < 1 → 0 ≤ θ → θ ≤ π / 2 →
  1 ≤ (a - cos θ)^2 + (b - sin θ)^2 ∧ (a - cos θ)^2 + (b - sin θ)^2 ≤ 4

theorem proofProblem_answer : proofProblem :=
  sorry

end proofProblem_answer_l786_786928


namespace grant_total_earnings_l786_786936

def earnings_first_month : ℕ := 350
def earnings_second_month : ℕ := 2 * earnings_first_month + 50
def earnings_third_month : ℕ := 4 * (earnings_first_month + earnings_second_month)
def total_earnings : ℕ := earnings_first_month + earnings_second_month + earnings_third_month

theorem grant_total_earnings : total_earnings = 5500 := by
  sorry

end grant_total_earnings_l786_786936


namespace number_of_ways_l786_786211

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786211


namespace Tim_carrots_count_l786_786003

theorem Tim_carrots_count (initial_potatoes new_potatoes initial_carrots final_potatoes final_carrots : ℕ) 
  (h_ratio : 3 * final_potatoes = 4 * final_carrots)
  (h_initial_potatoes : initial_potatoes = 32)
  (h_new_potatoes : new_potatoes = 28)
  (h_final_potatoes : final_potatoes = initial_potatoes + new_potatoes)
  (h_initial_ratio : 3 * 32 = 4 * initial_carrots) : 
  final_carrots = 45 :=
by {
  sorry
}

end Tim_carrots_count_l786_786003


namespace general_term_l786_786926

noncomputable def a_sequence : ℕ → ℝ
| 0       := 1
| (n + 1) := (-2) / (a_sequence n + 3)

theorem general_term (n : ℕ) :
  a_sequence (n + 1) = (2 / (3 * 2^n - 2)) - 1 := 
sorry

end general_term_l786_786926


namespace num_ways_choose_materials_l786_786210

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786210


namespace solve_z_l786_786843

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786843


namespace solve_for_x_l786_786564

theorem solve_for_x (x : ℝ) : (x - 6)^4 = (1/16)^(-1) → x = 8 := by
  sorry

end solve_for_x_l786_786564


namespace ferry_travel_time_l786_786763

theorem ferry_travel_time:
  ∀ (v_P v_Q : ℝ) (d_P d_Q : ℝ) (t_P t_Q : ℝ),
    v_P = 8 →
    v_Q = v_P + 1 →
    d_Q = 3 * d_P →
    t_Q = t_P + 5 →
    d_P = v_P * t_P →
    d_Q = v_Q * t_Q →
    t_P = 3 := by
  sorry

end ferry_travel_time_l786_786763


namespace ratio_15_to_1_l786_786269

theorem ratio_15_to_1 (x : ℕ) (h : 15 / 1 = x / 10) : x = 150 := 
by sorry

end ratio_15_to_1_l786_786269


namespace consecutive_odd_integers_l786_786474

theorem consecutive_odd_integers (n : ℕ) (h1 : n > 0) (h2 : (1 : ℚ) / n * ((n : ℚ) * 154) = 154) : n = 10 :=
sorry

end consecutive_odd_integers_l786_786474


namespace isoperimetric_triangle_isoperimetric_quadrilateral_isoperimetric_ngon_l786_786652

theorem isoperimetric_triangle {p : ℝ} (h : p > 0) :
  ∃ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = p ∧
  ∀ (a' b' c' : ℝ), a' + b' + c' = p → triangle_area a b c ≥ triangle_area a' b' c' := sorry

theorem isoperimetric_quadrilateral {p : ℝ} (h : p > 0) :
  ∃ (a b c d : ℝ), a = b ∧ b = c ∧ c = d ∧ a + b + c + d = p ∧
  ∀ (a' b' c' d' : ℝ), a' + b' + c' + d' = p → quadrilateral_area a b c d ≥ quadrilateral_area a' b' c' d' := sorry

theorem isoperimetric_ngon {n : ℕ} (hn : n ≥ 3) {p : ℝ} (hp : p > 0) :
  ∃ (a : ℕ → ℝ), (∀ i, a i = p / n) ∧
  ∀ (a' : ℕ → ℝ), (∀ i, 0 ≤ a' i) ∧ (∑ i in finset.range n, a' i) = p →
  ngon_area n a ≥ ngon_area n a' := sorry

end isoperimetric_triangle_isoperimetric_quadrilateral_isoperimetric_ngon_l786_786652


namespace proof_f_sum_l786_786098

def f (x : ℝ) : ℝ :=
  if x > 3 then x^2 - 1
  else if x >= -3 then 3 * x + 2
  else -1

theorem proof_f_sum : f (-4) + f (0) + f (4) = 16 :=
by
  sorry

end proof_f_sum_l786_786098


namespace find_m_l786_786489

-- Parameter equation of the circle C
def circle_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos t, -2 + 3 * Real.sin t)

-- Standard equation of circle C
def circle_standard (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y + 2) ^ 2 = 9

-- Polar equation of the line l
def line_polar (ρ θ : ℝ) (m : ℝ) : Prop :=
  √2 * ρ * Real.sin (θ - Real.pi / 4) = m

-- Cartesian equation of the line l
def line_cartesian (x y m : ℝ) : Prop := 
  x - y + m = 0

-- Distance between the center of circle C and line l
def distance (c : ℝ × ℝ) (m : ℝ) : ℝ :=
  let (x₀, y₀) := c
  abs (x₀ - y₀ + m) / √2

-- Proof statement: Given the conditions, m must be -3 ± 2√2
theorem find_m (m : ℝ) : (distance (1, -2) m = 2) → (m = -3 + 2 * √2) ∨ (m = -3 - 2 * √2) :=
by
  sorry

end find_m_l786_786489


namespace gcd_ab_l786_786708

def a := 59^7 + 1
def b := 59^7 + 59^3 + 1

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l786_786708


namespace each_friend_gets_four_pieces_l786_786508

noncomputable def pieces_per_friend : ℕ :=
  let oranges := 80
  let pieces_per_orange := 10
  let friends := 200
  (oranges * pieces_per_orange) / friends

theorem each_friend_gets_four_pieces :
  pieces_per_friend = 4 :=
by
  sorry

end each_friend_gets_four_pieces_l786_786508


namespace solve_for_z_l786_786826

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786826


namespace sequences_of_length_21_l786_786943

def valid_sequences : ℕ → ℕ
| 3     := 1
| 4     := 1
| 5     := 1
| 6     := 2
| (n+7) := valid_sequences (n+3) + 2 * valid_sequences (n+2) + valid_sequences n
| _     := 0 -- default case for other numbers

theorem sequences_of_length_21 :
  valid_sequences 21 = 114 := 
sorry

end sequences_of_length_21_l786_786943


namespace sqrt_equiv_1715_l786_786714

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l786_786714


namespace largest_initial_number_l786_786064

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786064


namespace arthur_amount_left_l786_786695

def initial_amount : ℝ := 200
def fraction_spent : ℝ := 4 / 5

def spent (initial : ℝ) (fraction : ℝ) : ℝ := fraction * initial

def amount_left (initial : ℝ) (spent_amount : ℝ) : ℝ := initial - spent_amount

theorem arthur_amount_left : amount_left initial_amount (spent initial_amount fraction_spent) = 40 := 
by
  sorry

end arthur_amount_left_l786_786695


namespace probability_of_selecting_A_l786_786325

noncomputable def total_students : ℕ := 4
noncomputable def selected_student_A : ℕ := 1

theorem probability_of_selecting_A : 
  (selected_student_A : ℝ) / (total_students : ℝ) = 1 / 4 :=
by
  sorry

end probability_of_selecting_A_l786_786325


namespace solve_for_z_l786_786478

theorem solve_for_z (a b s z : ℝ) (h1 : z ≠ 0) (h2 : 1 - 6 * s ≠ 0) (h3 : z = a^3 * b^2 + 6 * z * s - 9 * s^2) :
  z = (a^3 * b^2 - 9 * s^2) / (1 - 6 * s) := 
 by
  sorry

end solve_for_z_l786_786478


namespace three_digit_number_l786_786929

theorem three_digit_number (a b c : ℕ) (h1 : a * (b + c) = 33) (h2 : b * (a + c) = 40) : 
  100 * a + 10 * b + c = 347 :=
by
  sorry

end three_digit_number_l786_786929


namespace shortest_perimeter_inscribed_triangle_l786_786983

theorem shortest_perimeter_inscribed_triangle (A B C D E F : Point)
(acute_ABC : is_acute_triangle A B C)
(on_side_D : lies_on D (segment B C))
(on_side_E : lies_on E (segment C A))
(on_side_F : lies_on F (segment A B))
: is_orthic_triangle A B C D E F :=
sorry

end shortest_perimeter_inscribed_triangle_l786_786983


namespace fraction_product_eq_l786_786368
-- Import the necessary library

-- Define the fractions and the product
def fraction_product : ℚ :=
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8)

-- State the theorem we want to prove
theorem fraction_product_eq : fraction_product = 3 / 8 := 
sorry

end fraction_product_eq_l786_786368


namespace largest_initial_number_l786_786048

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786048


namespace sufficient_but_not_necessary_to_increasing_l786_786610

theorem sufficient_but_not_necessary_to_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → (x^2 - 2*a*x) ≤ (y^2 - 2*a*y)) ↔ (a ≤ 1) := sorry

end sufficient_but_not_necessary_to_increasing_l786_786610


namespace dividend_paid_per_person_l786_786657

-- Define the expected and actual earnings, as well as the number of shares
def expected_earnings_per_share : ℝ := 0.80
def actual_earnings_per_share : ℝ := 1.10
def shares_owned : ℕ := 100

-- Extra earnings computations and dividend conditions
def half_earnings_distributed_as_dividend (e : ℝ) : ℝ := e / 2
def additional_dividend_rate : ℝ := 0.04
def extra_per_share_in_dollars : ℝ := 0.10

-- Calculate the exact dividend to prove
theorem dividend_paid_per_person :
  let expected_dividend_per_share := half_earnings_distributed_as_dividend expected_earnings_per_share,
      extra_earnings := actual_earnings_per_share - expected_earnings_per_share,
      additional_dividends := (extra_earnings / extra_per_share_in_dollars) * additional_dividend_rate,
      total_dividend_per_share := expected_dividend_per_share + additional_dividends,
      total_dividend := total_dividend_per_share * shares_owned
  in total_dividend = 52 :=
begin
  sorry
end

end dividend_paid_per_person_l786_786657


namespace hiker_speed_correct_l786_786668

variable (hikerSpeed : ℝ)
variable (cyclistSpeed : ℝ := 15)
variable (cyclistTravelTime : ℝ := 5 / 60)  -- Converted 5 minutes to hours
variable (hikerCatchUpTime : ℝ := 13.75 / 60)  -- Converted 13.75 minutes to hours
variable (cyclistDistance : ℝ := cyclistSpeed * cyclistTravelTime)

theorem hiker_speed_correct :
  (hikerSpeed * hikerCatchUpTime = cyclistDistance) →
  hikerSpeed = 60 / 11 :=
by
  intro hiker_eq_cyclist_distance
  sorry

end hiker_speed_correct_l786_786668


namespace cube_volume_l786_786545

theorem cube_volume (a : ℕ) (h : a^3 - ((a - 2) * a * (a + 2)) = 16) : a^3 = 64 := by
  sorry

end cube_volume_l786_786545


namespace solve_for_z_l786_786860

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786860


namespace cos_c_of_tan_c_l786_786002

theorem cos_c_of_tan_c (A B C : Type) 
  [triangle : triangleABC A B C] (h_right : angle A = 90) (h_tan : tan C = 4 / 3) : 
  cos C = 3 / 5 :=
sorry

end cos_c_of_tan_c_l786_786002


namespace odd_function_f_l786_786432

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x * (1 - x) else -f (-x)

theorem odd_function_f (x : ℝ) (h : x < 0) : f x = x * (1 + x) := 
by
  sorry

end odd_function_f_l786_786432


namespace range_of_real_number_a_l786_786935

theorem range_of_real_number_a (a : ℝ) : (∀ (x : ℝ), 0 < x → a < x + 1/x) → a < 2 := 
by
  sorry

end range_of_real_number_a_l786_786935


namespace binomial_sum_divisible_l786_786422

theorem binomial_sum_divisible 
  (m : ℤ) (k : ℤ) (p : ℕ) [Fact (Nat.Prime p)] 
  (hm : Odd m) (hm_gt : m > 1) 
  (hp : p > m * k + 1) 
  : p^2 ∣ ∑ x in Finset.range p \ Finset.range k, (Nat.choose x.to_nat k.to_nat)^m :=
by
  sorry

end binomial_sum_divisible_l786_786422


namespace total_distance_travelled_downstream_l786_786609

-- Define the given speeds and times
def boat_speed_still : ℝ := 15  -- speed of the boat in still water (km/hr)
def current_speed_1 : ℝ := 3    -- current speed for the first 4 minutes (km/hr)
def current_speed_2 : ℝ := 5    -- current speed for the next 4 minutes (km/hr)
def current_speed_3 : ℝ := 7    -- current speed for the last 4 minutes (km/hr)

-- Convert the given times from minutes to hours
def time_period : ℝ := 4 / 60   -- each segment is 4 minutes, which is 4/60 hours

-- Calculations for each segment
def speed_segment_1 : ℝ := boat_speed_still + current_speed_1
def speed_segment_2 : ℝ := boat_speed_still + current_speed_2
def speed_segment_3 : ℝ := boat_speed_still + current_speed_3

def distance_segment_1 : ℝ := speed_segment_1 * time_period
def distance_segment_2 : ℝ := speed_segment_2 * time_period
def distance_segment_3 : ℝ := speed_segment_3 * time_period

-- Summing up the distances for each segment
def total_distance : ℝ := distance_segment_1 + distance_segment_2 + distance_segment_3

-- Lean theorem statement
theorem total_distance_travelled_downstream : total_distance = 4.001 := by
  sorry

end total_distance_travelled_downstream_l786_786609


namespace a_seq_general_term_S_n_sum_l786_786015

-- Define the sequence a_n based on the given conditions
def a_seq (n : ℕ) : ℕ → ℝ
| 1 => 1
| k + 2 => (λ a : ℝ, (2 - (a + 1) * (3/2 + (a_seq k))))⁻¹ - 1 -- transformed from the given condition

-- Define the new sequence b_n
def b_seq (n : ℕ) : ℕ → ℝ
| n => 1 + (a_seq (2^n))

-- Define the sum S_n for the sequence 2nb_n
def S_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, 2 * (k + 1) * b_seq (n - (k + 1))) -- sum of sequence 2nb_n

theorem a_seq_general_term (n : ℕ) (hn : 0 < n) : a_seq n = (2 / n) - 1 := by
  sorry

theorem S_n_sum (n : ℕ) (hn : 0 < n) : S_n n = 8 - (n + 2) / (2 ^ (n - 2)) := by
  sorry

end a_seq_general_term_S_n_sum_l786_786015


namespace solve_for_z_l786_786861

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786861


namespace tanya_addition_problem_l786_786031

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786031


namespace sqrt_equiv_1715_l786_786712

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l786_786712


namespace sqrt_of_expression_l786_786717

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l786_786717


namespace dogs_legs_l786_786014

theorem dogs_legs (num_dogs : ℕ) (legs_per_dog : ℕ) (h1 : num_dogs = 109) (h2 : legs_per_dog = 4) : num_dogs * legs_per_dog = 436 :=
by {
  -- The proof is omitted as it's indicated that it should contain "sorry"
  sorry
}

end dogs_legs_l786_786014


namespace num_pieces_on_board_l786_786251

def num_pieces (board : matrix (fin 8) (fin 8) ℕ) : ℕ :=
∑ i j, board i j

def same_pieces_in_2x2_squares (board : matrix (fin 8) (fin 8) ℕ) : Prop :=
∀ i j, (i < 7) → (j < 7) →
  board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1) = m

def same_pieces_in_3x1_rectangles (board : matrix (fin 8) (fin 8) ℕ) : Prop :=
∀ i j, (i < 6) → board i j + board (i+1) j + board (i+2) j = n ∧
       (j < 6) → board i j + board i (j+1) + board i (j+2) = n

theorem num_pieces_on_board (board : matrix (fin 8) (fin 8) ℕ) 
  (h20 : same_pieces_in_2x2_squares board)
  (h31 : same_pieces_in_3x1_rectangles board) :
  num_pieces board = 0 ∨ num_pieces board = 64 := 
sorry

end num_pieces_on_board_l786_786251


namespace solve_for_y_l786_786126

theorem solve_for_y (y : ℝ) :
  16^(2 * y - 4) = (1 / 2)^(y + 3) → y = 13 / 9 :=
by sorry

end solve_for_y_l786_786126


namespace length_of_PT_l786_786721

-- Define the quadrilateral and points
variable {P Q R S T : Type}
variable [AddGroup Q] [AddGroup R] 

-- Define lengths as constants corresponding to the problem conditions
def PQ_length : ℝ := 10
def RS_length : ℝ := 15
def PR_length : ℝ := 18

-- Define the conditions required for the proof
def is_convex_quadrilateral (PQRS : Q → R → P → S → Prop) : Prop :=
  PQRS P Q R S

def diagonals_intersect_at (T : Type) : Prop :=
  ∃ T, T = QT ∧ T = TR

def equal_area_triangles : Prop :=
  ∃ T, area_of_triangle PTR = area_of_triangle QTS

-- The main statement to prove
theorem length_of_PT :
  is_convex_quadrilateral PQRS → 
  diagonals_intersect_at T → 
  equal_area_triangles → 
  PT = 36 / 5 :=
sorry

end length_of_PT_l786_786721


namespace solve_z_l786_786778

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786778


namespace part1_monotonic_increasing_interval_part2_max_k_l786_786439

-- Definitions for the given functions
def f (x : ℝ) : ℝ := Real.log x - x^2
def y (x : ℝ) : ℝ := f(x) + x * (1 / x - 2 * x)
def g (x : ℝ) (b : ℝ) : ℝ := f(x) + (3 / 2) * x^2 - (1 + b) * x

-- Proving the mathematically equivalent proof problem
theorem part1_monotonic_increasing_interval : 
  (∀ x (hx : 0 < x ∧ x < Real.sqrt 6 / 6), 
    1 - 6 * x^2 > 0) :=
sorry

theorem part2_max_k (b : ℝ) (hb : b ≥ (Real.exp 2 + 1) / Real.exp 1 - 1) :
  (∀ x1 x2 (hx : 0 < x1 ∧ x1 < x2 ∧ x2 < 1),
    let t := x1 + 1 / x1 in
    t ≥ Real.exp 1 + 1 / Real.exp 1 →
    g(x1) - g(x2) ≥ k →
    k ≤ Real.exp 2 / 2 - 1 / (2 * Real.exp 2) - 2) :=
sorry

end part1_monotonic_increasing_interval_part2_max_k_l786_786439


namespace solve_for_z_l786_786775

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786775


namespace average_weight_l786_786144

theorem average_weight :
  ∀ (A B C : ℝ),
    (A + B = 84) → 
    (B + C = 86) → 
    (B = 35) → 
    (A + B + C) / 3 = 45 :=
by
  intros A B C hab hbc hb
  -- proof omitted
  sorry

end average_weight_l786_786144


namespace task_probabilities_l786_786275

theorem task_probabilities :
  let p1 := 5 / 8
  let p2_not := 2 / 5
  let p3_not := 3 / 10
  let p4 := 3 / 4
  p1 * p2_not * p3_not * p4 = 9 / 160 :=
by
  let p1 := 5 / 8
  let p2_not := 2 / 5
  let p3_not := 3 / 10
  let p4 := 3 / 4
  calc
    p1 * p2_not * p3_not * p4 = (5 / 8) * (2 / 5) * (3 / 10) * (3 / 4) : by refl
    ... = (5 * 2 * 3 * 3) / (8 * 5 * 10 * 4) : by norm_num
    ... = (2 * 3 * 3) / (8 * 10 * 4) : by field_simp [mul_comm]
    ... = 9 / 160 : by norm_num

end task_probabilities_l786_786275


namespace total_cost_of_six_apples_three_oranges_l786_786558

def cost_of_apple : ℝ := 0.21
def cost_of_orange (c2a5o : ℝ) : ℝ := (c2a5o - 2 * cost_of_apple) / 5
def c2a5o : ℝ := 1.27
def cost_of_six_apples_three_oranges : ℝ := 6 * cost_of_apple + 3 * cost_of_orange c2a5o

theorem total_cost_of_six_apples_three_oranges : cost_of_six_apples_three_oranges = 1.77 :=
by
  hyperref sorry

end total_cost_of_six_apples_three_oranges_l786_786558


namespace solve_z_l786_786777

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786777


namespace chord_midpoint_line_eq_l786_786462

theorem chord_midpoint_line_eq (P : ℝ × ℝ)(O : ℝ × ℝ)(r : ℝ)(chord_eq : ℝ × ℝ → ℝ) : 
  P = (1, 1) →
  O = (3, 0) →
  (chord_eq (1, 1) = 0) → 
  (∀ x y : ℝ, (x - 3)^2 + y^2 = 9 → chord_eq (x, y) = 0) →
  chord_eq = (λ (p : ℝ × ℝ), 2 * p.1 - p.2 - 1) := 
by
  intros hP hO hPmid hcircle
  sorry

end chord_midpoint_line_eq_l786_786462


namespace largest_initial_number_l786_786061

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786061


namespace a6_value_l786_786442

def seq_rule : ℕ → ℕ → List ℕ → ℕ 
| n, a_n, written_vals :=
  if a_n - 2 ∈ written_vals then a_n + 3 
  else if a_n - 2 ∈ {0, 1, 2, 3, 4, 5, 6, 7}.toFinset then a_n - 2 else a_n + 3

theorem a6_value :
  let seq := ∀ (n : ℕ), ℕ → List ℕ → ℕ,
      a1 := 1,
      a2 := seq_rule 1 a1 [],
      a3 := seq_rule 2 a2 [a1],
      a4 := seq_rule 3 a3 [a1, a2],
      a5 := seq_rule 4 a4 [a1, a2, a3],
      a6 := seq_rule 5 a5 [a1, a2, a3, a4]
  in a6 = 6 := sorry

end a6_value_l786_786442


namespace point_above_line_l786_786607

theorem point_above_line (t : ℝ) : (∃ y : ℝ, y = (2 : ℝ)/3) → (t > (2 : ℝ)/3) :=
  by
  intro h
  sorry

end point_above_line_l786_786607


namespace pentagon_vertex_assignment_l786_786109

theorem pentagon_vertex_assignment :
  ∃ (x_A x_B x_C x_D x_E : ℝ),
    x_A + x_B = 1 ∧
    x_B + x_C = 2 ∧
    x_C + x_D = 3 ∧
    x_D + x_E = 4 ∧
    x_E + x_A = 5 ∧
    (x_A, x_B, x_C, x_D, x_E) = (1.5, -0.5, 2.5, 0.5, 3.5) := by
  sorry

end pentagon_vertex_assignment_l786_786109


namespace cylinder_diameter_height_l786_786649

-- Define the radius of the sphere
def r_s : ℝ := 4

-- Define the total volume of 12 spheres
def V_total_spheres : ℝ := 12 * (4 / 3) * Real.pi * r_s^3

-- Define the radius and height condition for the cylinder
def cylinder_condition (d h : ℝ) : Prop := 
  (pi * (d / 2)^2 * h = V_total_spheres) ∧ (d = h)

-- The goal is to prove that for the given conditions, diameter and height are both 16 cm.
theorem cylinder_diameter_height : ∃ d h : ℝ, cylinder_condition d h ∧ d = 16 ∧ h = 16 := by
  sorry

end cylinder_diameter_height_l786_786649


namespace find_hcf_l786_786598

-- Defining the conditions given in the problem
def hcf_of_two_numbers_is_H (A B H : ℕ) : Prop := Nat.gcd A B = H
def lcm_of_A_B (A B : ℕ) (H : ℕ) : Prop := Nat.lcm A B = H * 21 * 23
def larger_number_is_460 (A : ℕ) : Prop := A = 460

-- The propositional goal to prove that H = 20 given the above conditions
theorem find_hcf (A B H : ℕ) (hcf_cond : hcf_of_two_numbers_is_H A B H)
  (lcm_cond : lcm_of_A_B A B H) (larger_cond : larger_number_is_460 A) : H = 20 :=
sorry

end find_hcf_l786_786598


namespace min_value_trig_l786_786157

theorem min_value_trig (x : ℝ) : 
  ∀ (t : ℝ), t = x + 10 * real.pi / 180 →
  cos (x + 10 * real.pi / 180) + cos (x + 70 * real.pi / 180) = 
  sqrt 3 * cos (t + 30 * real.pi / 180) →
  (∀ y, y = cos (x + 10 * real.pi / 180) + cos (x + 70 * real.pi / 180) → y ≥ -sqrt 3) :=
begin
  sorry
end

end min_value_trig_l786_786157


namespace orthocenter_coincides_with_Nagel_point_l786_786621

-- Definitions of basic geometry entities
variable (ABC : Type) [triangle ABC]
variable (I : incenter ABC)
variable (Omega_A Omega_B Omega_C : excircle ABC)
variable (T_A T_B T_C : BC-related tangency_points ABC Omega_A Omega_B Omega_C)
variable (l_A l_B l_C : tangents_lines_through_feet ABC I Omega_A Omega_B Omega_C)

-- Definition of the Nagel point 
def Nagel_point := intersection [connected_segment T_A, connected_segment T_B, connected_segment T_C]

-- Assumption that these lines form a triangle
variable (orthocenter_of_triangle : orthocenter (triangle_of_lines l_A l_B l_C))

-- The theorem to be proved
theorem orthocenter_coincides_with_Nagel_point :
  orthocenter_of_triangle = Nagel_point ABC Omega_A Omega_B Omega_C T_A T_B T_C l_A l_B l_C :=
sorry

end orthocenter_coincides_with_Nagel_point_l786_786621


namespace find_z_l786_786838

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786838


namespace smallest_angle_in_convex_20_gon_seq_l786_786586

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end smallest_angle_in_convex_20_gon_seq_l786_786586


namespace tv_price_lower_l786_786538

-- Definitions based on conditions from the problem
def initial_price : ℝ := 1000
def discount_100 : ℝ := 100
def additional_discount_percentage : ℝ := 20

-- Theorem to prove the final answer
theorem tv_price_lower 
  (initial_price : ℝ) 
  (discount_100 : ℝ) 
  (additional_discount_percentage : ℝ) :
  let discounted_price := initial_price - discount_100 in
  let additional_discount := discounted_price * (additional_discount_percentage / 100) in
  let final_price := discounted_price - additional_discount in
  initial_price - final_price = 280 :=
by
  sorry

end tv_price_lower_l786_786538


namespace trajectory_is_parabola_line_PQ_fixed_point_l786_786417

noncomputable def dist (P Q : Point ℝ) : ℝ := 
  sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

structure Point (α : Type*) := 
  (x : α)
  (y : α)

def parabola (F : Point ℝ) (d : ℝ) (M : Point ℝ) : Prop :=
  dist M F = abs (M.x + d)

-- ** Part (Ⅰ) **
theorem trajectory_is_parabola :
  ∀ (M : Point ℝ), (dist M (Point.mk 1 0) = abs (M.x + 1)) →
  (M.y)^2 = 4 * M.x := 
sorry

-- ** Part (Ⅱ) **
theorem line_PQ_fixed_point :
  ∀ (M A B F P Q : Point ℝ) (k : ℝ),
  (dist M F = abs (M.x + 1)) →
  (dist A F = abs (A.x + 1)) →
  (dist B F = abs (B.x + 1)) →
  -- Assuming the parabola y^2 = 4x
  P = Point.mk (1 + 2/k^2) (2/k) →
  Q = Point.mk (1 + 2*k^2) (-2*k) →
  ∃ E : Point ℝ, E = Point.mk 3 0 ∧ (Q.x - P.x)*(1) = (Q.y - P.y)*(0) :=
sorry

end trajectory_is_parabola_line_PQ_fixed_point_l786_786417


namespace greatest_possible_integer_l786_786989

theorem greatest_possible_integer (n k l : ℕ) (h1 : n < 150) (h2 : n = 11 * k - 1) (h3 : n = 9 * l + 2) : n = 65 :=
by sorry

end greatest_possible_integer_l786_786989


namespace simson_line_l786_786551

noncomputable def points_collinear {α : Type*} [plane_geometry α]
  (P A B C U V W : α)
  (circumcircle : circle α)
  (P_on_circumcircle : P ∈ circumcircle)
  (P_perp_BC : ⟂ P U BC)
  (P_perp_CA : ⟂ P V CA)
  (P_perp_AB : ⟂ P W AB) : Prop := 
  collinear {U, V, W}

theorem simson_line
  {α : Type*} [plane_geometry α]
  (A B C P U V W : α)
  (circumcircle : circle α)
  (P_on_circumcircle : P ∈ circumcircle)
  (P_perp_BC : ⟂ P U BC)
  (P_perp_CA : ⟂ P V CA)
  (P_perp_AB : ⟂ P W AB) :
  points_collinear P A B C U V W circumcircle P_on_circumcircle P_perp_BC P_perp_CA P_perp_AB := 
  sorry

end simson_line_l786_786551


namespace descending_order_numbers_count_l786_786750

theorem descending_order_numbers_count : 
  ∃ (n : ℕ), (n = 1013) ∧ 
  ∀ (x : ℕ), (∃ (xs : list ℕ), 
                (∀ i, i < xs.length - 1 → xs.nth_le i sorry > xs.nth_le (i+1) sorry) ∧ 
                nat_digits_desc xs ∧
                1 < xs.length) → 
             x ∈ nat_digits xs →
             ∃ (refs : list ℕ), n = refs.length ∧ 
             ∀ ref, ref ∈ refs → ref < x :=
sorry

end descending_order_numbers_count_l786_786750


namespace distance_focus_directrix_l786_786587

theorem distance_focus_directrix (x y : ℝ) (h : x^2 = 4 * y) : 
  (distance_focus_to_directrix h = 2) := sorry

end distance_focus_directrix_l786_786587


namespace number_of_ways_to_choose_reading_materials_l786_786243

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786243


namespace sixth_purchase_cost_l786_786484

theorem sixth_purchase_cost (a b c d : ℕ) (h : {1900, 2070, 2110, 2330, 2500} ⊆ {a + b, b + c, c + d, d + a, a + c, b + d}) : 
  (∃ x ∈ {a + b, b + c, c + d, d + a, a + c, b + d}, x = 2290) := 
by 
  -- From given conditions, {a + b, b + c, c + d, d + a, a + c, b + d} is the set of sums,
  -- and given five sums {1900, 2070, 2110, 2330, 2500} are known.
  -- We need to prove that there exists an x ∈ {a + b, b + c, c + d, d + a, a + c, b + d} such that x = 2290.
  sorry

end sixth_purchase_cost_l786_786484


namespace open_box_problem_l786_786673

theorem open_box_problem (length width l1 l2 w1 w2 : ℕ) (l1_eq : l1 = 7) (l2_eq : l2 = 4) (w1_eq : w1 = 5) (w2_eq : w2 = 6) (length_eq : length = 48) (width_eq : width = 36) :
  let box_length := length - (l1 + l2) in
  let box_width := width - (w1 + w2) in
  let box_height := min l1 (min l2 (min w1 w2)) in
  let volume := box_length * box_width * box_height in
  box_length = 37 ∧ box_width = 25 ∧ box_height = 4 ∧ volume = 3700 :=
by
  sorry

end open_box_problem_l786_786673


namespace increasing_intervals_range_of_m_l786_786089

noncomputable def f (x : ℝ) : ℝ := sqrt (3) * sin(2 * x + π / 2) + sin(2 * x + 0)
noncomputable def g (x : ℝ) : ℝ := 2 * sin(2 * (x + π / 6) + π / 3) - 1

theorem increasing_intervals :
  ∀ x, 0 ≤ x ∧ x ≤ π →
  ((0 ≤ x ∧ x ≤ π / 12) ∨ (7 * π / 12 ≤ x ∧ x ≤ π)) →
  ∃ (I1 I2 : set ℝ),
    I1 = set.Icc 0 (π / 12) ∧ I2 = set.Icc (7 * π / 12) π ∧
    ∀ x ∈ I1 ∪ I2, monotone (f x) := 
sorry

theorem range_of_m (m : ℝ) :
  ∃ x ∈ set.Icc 0 (π / 2), g x = m →
  -3 ≤ m ∧ m ≤ sqrt (3) - 1 := 
sorry

end increasing_intervals_range_of_m_l786_786089


namespace num_ways_choose_materials_l786_786205

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786205


namespace determine_slope_l786_786891

variables {a_n : ℕ → ℤ} {s_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ a₁ d, ∀ (n : ℕ), a_n n = a₁ + n * d

def sum_of_first_n_terms (a_n : ℕ → ℤ) (s_n : ℕ → ℤ) :=
  ∀ (n : ℕ), s_n n = (n * (a_n 0 + a_n (n-1))) / 2

noncomputable def find_slope {a_n : ℕ → ℤ} (n : ℕ) : ℤ :=
  (a_n (n+2) - a_n n) / (2 : ℕ)

theorem determine_slope (a_n s_n : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a_n)
  (h2 : sum_of_first_n_terms a_n s_n)
  (S2 : s_n 2 = 10) (S5 : s_n 5 = 55)
  (n : ℕ) (hn : n ≠ 0) : find_slope a_n n = 4 :=
begin
  sorry
end

end determine_slope_l786_786891


namespace solve_complex_equation_l786_786797

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786797


namespace time_for_A_to_complete_race_l786_786483

theorem time_for_A_to_complete_race
  (V_A V_B : ℝ) (T_A : ℝ)
  (h1 : V_B = 975 / T_A) (h2 : V_B = 2.5) :
  T_A = 390 :=
by
  sorry

end time_for_A_to_complete_race_l786_786483


namespace roger_down_payment_percentage_l786_786554

-- Define conditions
variables (P : ℝ)
constants (price : ℝ := 100000) (remainingDebt : ℝ := 56000)

-- Roger's parents pay off 30% of the remaining balance.
def remainingBalance := price - P * price
def parentContribution := 0.30 * remainingBalance
def finalDebt := remainingBalance - parentContribution

-- Lean statement for the problem
theorem roger_down_payment_percentage (P : ℝ) (h : finalDebt = remainingDebt) : 
  P = 0.20 :=
by
  sorry

end roger_down_payment_percentage_l786_786554


namespace max_buns_transported_l786_786676

-- Definitions based on the conditions in a)
def total_buns : ℕ := 200
def buns_per_trip : ℕ := 40
def buns_eaten_each_way : ℕ := 1

-- Statement for the problem using the conditions and correct answer
theorem max_buns_transported : 
  ∀ (total_buns buns_per_trip buns_eaten_each_way : ℕ), 
  total_buns = 200 → 
  buns_per_trip = 40 → 
  buns_eaten_each_way = 1 → 
  ∃ max_buns : ℕ, max_buns = 191 :=
by
  intros
  use 191
  sorry

end max_buns_transported_l786_786676


namespace sequence_conjecture_l786_786070

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

def sequence_condition_1 : a 1 = 2 := by
  sorry
  
def sequence_condition_2 (n : ℕ) : 3 * (S n) = a n * (n + 2) := by
  sorry

def sum_condition (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, S n = ∑ k in range n + 1, a k) := by
  sorry

theorem sequence_conjecture (n : ℕ) : 
  (a n = n * (n + 1)) := by
  sorry

end sequence_conjecture_l786_786070


namespace students_material_selection_l786_786224

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786224


namespace angle_A_in_triangle_ABC_l786_786017

theorem angle_A_in_triangle_ABC :
  ∀ (A B C : ℝ) (a b c : ℝ) (hA : A > 0 ∧ A < π) (hB : cos B = 2 / 3) (h_b : b = sqrt 5) (h_c : c = 2),
  a^2 + c^2 - 2 * a * c * cos B = b^2 → A = π / 2 :=
by
  intros A B C a b c hA hB h_b h_c h1
  sorry

end angle_A_in_triangle_ABC_l786_786017


namespace quadrilateral_AD_BC_area_l786_786972

theorem quadrilateral_AD_BC_area (a b : ℝ) (α : ℝ) (ha : a > b) (hα : sin α > b / a) :
  let AD := (a - b * sin α) / (cos α)
  let BC := (a * sin α - b) / (cos α)
  let area := (a^2 - b^2) * tan α / 2
  AD = (a - b * sin α) / (cos α) 
  ∧ BC = (a * sin α - b) / (cos α) 
  ∧ area = (a^2 - b^2) * tan α / 2 :=
by
  sorry

end quadrilateral_AD_BC_area_l786_786972


namespace find_y_l786_786133

theorem find_y (y : ℝ) (h : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) : y = 53 / 3 :=
by
  sorry

end find_y_l786_786133


namespace solution_in_each_test_tube_l786_786170

theorem solution_in_each_test_tube (solution_total : ℕ) (num_test_tubes : ℕ) (num_beakers : ℕ) (solution_per_beaker : ℕ) (h1 : num_test_tubes = 6) (h2 : num_beakers = 3) (h3 : solution_per_beaker = 14) (h4 : solution_total = num_beakers * solution_per_beaker) : 
(solution_total / num_test_tubes = 7)  :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  rw [Nat.div_eq_of_eq_mul_right (by norm_num) h4]
  norm_num

end solution_in_each_test_tube_l786_786170


namespace zero_in_A_l786_786443

noncomputable def A : Set ℝ := {x | x * (x - 1) = 0}

theorem zero_in_A : 0 ∈ A :=
by
  have h : 0 * (0 - 1) = 0 := by norm_num
  show 0 ∈ A from h

end zero_in_A_l786_786443


namespace sum_two_triangular_numbers_iff_l786_786683

theorem sum_two_triangular_numbers_iff (m : ℕ) : 
  (∃ a b : ℕ, m = (a * (a + 1)) / 2 + (b * (b + 1)) / 2) ↔ 
  (∃ x y : ℕ, 4 * m + 1 = x * x + y * y) :=
by sorry

end sum_two_triangular_numbers_iff_l786_786683


namespace find_z_l786_786802

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786802


namespace total_number_of_turtles_l786_786180

variable {T : Type} -- Define a variable for the type of turtles

-- Define the conditions as hypotheses
variable (total_turtles : ℕ)
variable (female_percentage : ℚ) (male_percentage : ℚ)
variable (striped_male_prop : ℚ)
variable (baby_striped_males : ℕ) (adult_striped_males_prop : ℚ)
variable (striped_male_percentage : ℚ)
variable (striped_males : ℕ)
variable (male_turtles : ℕ)

-- Condition definitions
def female_percentage_def := female_percentage = 60 / 100
def male_percentage_def := male_percentage = 1 - female_percentage
def striped_male_prop_def := striped_male_prop = 1 / 4
def adult_striped_males_prop_def := adult_striped_males_prop = 60 / 100
def baby_and_adult_striped_males_prop_def := (1 - adult_striped_males_prop) = 40 / 100
def striped_males_def := striped_males = baby_striped_males / (1 - adult_striped_males_prop)
def male_turtles_def := male_turtles = striped_males / striped_male_prop
def male_turtles_percentage_def := male_turtles = total_turtles * (1 - female_percentage)

-- The proof statement to show the total number of turtles is 100
theorem total_number_of_turtles (h_female : female_percentage_def)
                                (h_male : male_percentage_def)
                                (h_striped_male_prop : striped_male_prop_def)
                                (h_adult_striped_males_prop : adult_striped_males_prop_def)
                                (h_baby_and_adult_striped_males_prop : baby_and_adult_striped_males_prop_def)
                                (h_striped_males : striped_males_def)
                                (h_male_turtles : male_turtles_def)
                                (h_male_turtles_percentage : male_turtles_percentage_def):
  total_turtles = 100 := 
by sorry

end total_number_of_turtles_l786_786180


namespace lines_coinicide_l786_786115

open Real

theorem lines_coinicide (k m n : ℝ) :
  (∃ (x y : ℝ), y = k * x + m ∧ y = m * x + n ∧ y = n * x + k) →
  k = m ∧ m = n :=
by
  sorry

end lines_coinicide_l786_786115


namespace find_z_l786_786834

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786834


namespace monomial_coefficient_and_degree_l786_786577

variable (x y z : ℝ)
def monomial := -3 * Real.pi * x * y^2 * z^3

theorem monomial_coefficient_and_degree :
  let coeff := -3 * Real.pi
  let degree := 1 + 2 + 3
  monomial x y z = coeff * x * y^2 * z^3 ∧ degree = 6 :=
by
  sorry

end monomial_coefficient_and_degree_l786_786577


namespace minimize_distance_on_ellipse_l786_786904

theorem minimize_distance_on_ellipse (a m n : ℝ) (hQ : 0 < a ∧ a ≠ Real.sqrt 3)
  (hP : m^2 / 3 + n^2 / 2 = 1) :
  |minimize_distance| = Real.sqrt 3 ∨ |minimize_distance| = 3 * a := sorry

end minimize_distance_on_ellipse_l786_786904


namespace man_speed_approx_l786_786296

noncomputable def speed_of_man : ℝ :=
  let L := 700    -- Length of the train in meters
  let u := 63 / 3.6  -- Speed of the train in meters per second (converted)
  let t := 41.9966402687785 -- Time taken to cross the man in seconds
  let v := (u * t - L) / t  -- Speed of the man
  v

-- The main theorem to prove that the speed of the man is approximately 0.834 m/s.
theorem man_speed_approx : abs (speed_of_man - 0.834) < 1e-3 :=
by
  -- Simplification and exact calculations will be handled by the Lean prover or could be manually done.
  sorry

end man_speed_approx_l786_786296


namespace find_z_l786_786839

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786839


namespace smallest_four_digit_divisible_by_6_l786_786262

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_divisible_by_6_l786_786262


namespace perpendicular_tangents_sum_x1_x2_gt_4_l786_786412

noncomputable def f (x : ℝ) : ℝ := (1 / 6) * x^3 - (1 / 2) * x^2 + (1 / 3)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def F (x : ℝ) : ℝ := (1 / 2) * x^2 - x - 2 * Real.log x

theorem perpendicular_tangents (a : ℝ) (b : ℝ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 1 / 3) (h₃ : c = 0) :
  let f' x := (1 / 2) * x^2 - x
  let g' x := 2 / x
  f' 1 * g' 1 = -1 :=
by sorry

theorem sum_x1_x2_gt_4 (x1 x2 : ℝ) (h₁ : 0 < x1 ∧ x1 < 4) (h₂ : 0 < x2 ∧ x2 < 4) (h₃ : x1 ≠ x2) (h₄ : F x1 = F x2) :
  x1 + x2 > 4 :=
by sorry

end perpendicular_tangents_sum_x1_x2_gt_4_l786_786412


namespace smallest_sum_of_three_l786_786352

open Finset

-- Define the set of numbers
def my_set : Finset ℤ := {10, 2, -4, 15, -7}

-- Statement of the problem: Prove the smallest sum of any three different numbers from the set is -9
theorem smallest_sum_of_three :
  ∃ (a b c : ℤ), a ∈ my_set ∧ b ∈ my_set ∧ c ∈ my_set ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
sorry

end smallest_sum_of_three_l786_786352


namespace makenna_garden_larger_by_375_l786_786515

variable (karl_length : ℕ) (karl_width : ℕ)
variable (makenna_length : ℕ) (makenna_width : ℕ)
variable (shed_length : ℕ) (shed_width : ℕ)

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def effective_makenna_area (makenna_length makenna_width shed_length shed_width : ℕ) : ℕ :=
  area makenna_length makenna_width - area shed_length shed_width

theorem makenna_garden_larger_by_375 (karl_length karl_width makenna_length makenna_width shed_length shed_width : ℕ) :
  karl_length = 30 →
  karl_width = 50 →
  makenna_length = 35 →
  makenna_width = 55 →
  shed_length = 5 →
  shed_width = 10 →
  effective_makenna_area makenna_length makenna_width shed_length shed_width = area karl_length karl_width + 375 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  dsimp [area, effective_makenna_area]
  -- Prove the expected result manually
  sorry

end makenna_garden_larger_by_375_l786_786515


namespace coloring_squares_l786_786503

/-- Jesse has ten squares, labeled 1, 2, ..., 10. Each square can be colored with one of four colors: red, green, yellow, or blue.
    We need to count the number of ways to color the squares such that for all 1 ≤ i < j ≤ 10, if i divides j, then the i-th and j-th squares have different colors. -/
theorem coloring_squares : 
  let colors := {0, 1, 2, 3}, 
      squares := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      colorings := list (fin 10) → fin 4
  in ∃ c : colorings, 
      (∀ (i j : fin 10), i.val < j.val ∧ j.val % i.val = 0 → c i ≠ c j) 
      ∧ (fintype.card { f : colorings // ∀ (i j : fin 10), i.val < j.val ∧ j.val % i.val = 0 → f i ≠ f j } = 324) :=
sorry

end coloring_squares_l786_786503


namespace range_of_a_l786_786441

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end range_of_a_l786_786441


namespace find_wyz_l786_786351

def N (w y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1, 3 * y, 0],
    ![w, y, -2 * z],
    ![w, -y, z]
  ]

theorem find_wyz (w y z : ℝ) (h : (N w y z)ᵀ ⬝ (N w y z) = 1) :
  w^2 + y^2 + z^2 = 4/5 := sorry

end find_wyz_l786_786351


namespace find_solutions_l786_786736

theorem find_solutions (x y z : ℝ) :
    (x^2 + y^2 - z * (x + y) = 2 ∧ y^2 + z^2 - x * (y + z) = 4 ∧ z^2 + x^2 - y * (z + x) = 8) ↔
    (x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2) := sorry

end find_solutions_l786_786736


namespace minimum_y_value_l786_786097

noncomputable def minValueY : ℝ :=
  27 - real.sqrt 745

theorem minimum_y_value (x y : ℝ) (h : x^2 + y^2 = 8 * x + 54 * y) : minValueY ≤ y :=
  sorry

end minimum_y_value_l786_786097


namespace simplify_trig_expression_l786_786123

theorem simplify_trig_expression (θ : ℝ) (h1 : θ = 160) (h2 : θ > 90 ∧ θ < 180) :
  sqrt(1 - sin θ ^ 2) = -cos θ :=
by
  sorry

end simplify_trig_expression_l786_786123


namespace no_identical_on_diagonal_l786_786726

def filled_grid (n : ℕ) : list (list ℕ) := sorry -- Represents the given grid

def valid_grid (grid : list (list ℕ)) : Prop := 
   (∀ i j, i < 15 → j < 15 → 1 ≤ grid[i][j] ∧ grid[i][j] ≤ 15) ∧  -- Each cell contains 1 to 15
   (∀ i j k, j ≠ k → grid[i][j] ≠ grid[i][k]) ∧  -- Rows are unique
   (∀ i j k, i ≠ k → grid[i][j] ≠ grid[k][j]) ∧  -- Columns are unique
   (∀ i j, i < 15 → j < 15 → grid[i][j] = grid[j][i])  -- Symmetric with respect to diagonals

def distinct_diagonal (grid : list (list ℕ)) : Prop := 
   (∀ i j, i ≠ j → i < 15 → j < 15 → grid[i][i] ≠ grid[j][j])  -- Main diagonal entries are distinct

theorem no_identical_on_diagonal :
  ∀ grid, valid_grid grid → distinct_diagonal grid :=
begin
  intros grid h,
  sorry
end

end no_identical_on_diagonal_l786_786726


namespace largest_initial_number_l786_786036

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786036


namespace central_angle_of_sector_l786_786434

theorem central_angle_of_sector
  (r : ℝ) (S_sector : ℝ) (alpha : ℝ) (h₁ : r = 2) (h₂ : S_sector = (2 / 5) * Real.pi)
  (h₃ : S_sector = (1 / 2) * alpha * r^2) : alpha = Real.pi / 5 :=
by
  sorry

end central_angle_of_sector_l786_786434


namespace marble_cut_in_third_week_l786_786321

def percentage_cut_third_week := 
  let initial_weight : ℝ := 250 
  let final_weight : ℝ := 105
  let percent_cut_first_week : ℝ := 0.30
  let percent_cut_second_week : ℝ := 0.20
  let weight_after_first_week := initial_weight * (1 - percent_cut_first_week)
  let weight_after_second_week := weight_after_first_week * (1 - percent_cut_second_week)
  (weight_after_second_week - final_weight) / weight_after_second_week * 100 = 25

theorem marble_cut_in_third_week :
  percentage_cut_third_week = true :=
by
  sorry

end marble_cut_in_third_week_l786_786321


namespace minimal_operations_bounds_l786_786525

-- Let n be an integer representing the dimension of the grid.
variable (n : ℤ)

-- Define the grid and its properties: each cell is either black or white.
def Grid (n : ℤ) := matrix (fin n) (fin n) bool

-- Define a function to determine if a vertex is red based on the number of black cells around it.
def isRedVertex (grid : Grid n) (i j : fin (n + 1)) : bool :=
  (if i.val < n ∧ j.val < n then
     (grid (fin.of_nat i.val) (fin.of_nat j.val) 
      + if i.val > 0 then grid (fin.pred i) (fin.of_nat j.val) else false
      + if j.val > 0 then grid (fin.of_nat i.val) (fin.pred j) else false
      + if i.val < n - 1 then grid (fin.succ i) (fin.of_nat j.val) else false
      + if j.val < n - 1 then grid (fin.of_nat i.val) (fin.succ j) else false) % 2)
  else
    false

-- Y is the number of red vertices.
def Y (grid : Grid n) : ℤ :=
  finset.card { ⟨i, j⟩ | isRedVertex grid i j }

-- X is the minimal number of operations needed to make the grid white.
def minimalOperations (grid : Grid n) : ℤ :=
  sorry -- Implementation required to calculate the minimal number of operations.

-- State the theorem about the bounds of X with respect to Y.
theorem minimal_operations_bounds (grid : Grid n) :
  ∃ X, minimalOperations grid = X ∧ 
  (Y grid / 4 : ℝ) ≤ (X : ℝ) ∧ (X : ℝ) ≤ (Y grid / 2 : ℝ) :=
sorry

end minimal_operations_bounds_l786_786525


namespace intersection_complement_correct_l786_786445

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A based on the condition given
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 3}

-- Define set B based on the condition given
def B : Set ℝ := {x | x > 3}

-- Define the complement of set B in the universal set U
def compl_B : Set ℝ := {x | x ≤ 3}

-- Define the expected result of A ∩ compl_B
def expected_result : Set ℝ := {x | x ≤ -3} ∪ {3}

-- State the theorem to be proven
theorem intersection_complement_correct :
  (A ∩ compl_B) = expected_result :=
sorry

end intersection_complement_correct_l786_786445


namespace frequency_of_2_in_20220420_l786_786493

theorem frequency_of_2_in_20220420 : 
  let num := "20220420",
      num_length := 8,
      count_2 := 4
  in count_2 * 2 = num_length :=
by 
  sorry

end frequency_of_2_in_20220420_l786_786493


namespace solve_for_z_l786_786869

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786869


namespace two_students_choose_materials_l786_786187

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786187


namespace solve_z_l786_786853

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786853


namespace solve_complex_equation_l786_786795

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786795


namespace vector_operations_find_k_l786_786449

variables (a b : ℝ × ℝ)
def k (a b : ℝ × ℝ) : ℝ := (4 : ℝ) * (2 * (a.1) + 1) / (2 * (a.1 + 2 * b.1))
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k • v)

theorem vector_operations (a b : ℝ × ℝ)
    (h_a : a = (2, 0)) (h_b : b = (1, 4)) : 
    2 • a + 3 • b = (7, 12) ∧ a - 2 • b = (0, -8) :=
  by
    sorry

theorem find_k (a b : ℝ × ℝ)
    (h_a : a = (2, 0)) (h_b : b = (1, 4))
    (hp : parallel (k a b • a + b) (a + 2 • b)) :
    k a b = 1 / 2 :=
  by
    sorry

end vector_operations_find_k_l786_786449


namespace find_z_l786_786812

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786812


namespace even_sum_exists_l786_786334

variable {N : ℕ}

/-- 
Given an \(N \times (N+1)\) integer matrix, some columns can be deleted 
such that each row sum of remaining elements is even.
-/

theorem even_sum_exists (M : Matrix (Fin N) (Fin (N + 1)) ℤ) :
    ∃ (cols_to_delete : Finset (Fin (N + 1))),
      ∀ i : Fin N, (∑ j in (Finset.univ \ cols_to_delete), M i j) % 2 = 0 := 
sorry

end even_sum_exists_l786_786334


namespace vacuuming_time_l786_786450

variable (time_dusting : ℕ) (time_mopping : ℕ)
variable (time_brushing_per_cat : ℕ) (num_cats : ℕ)
variable (total_free_time : ℕ) (free_time_left : ℕ)

theorem vacuuming_time :
  time_dusting = 60 →
  time_mopping = 30 →
  time_brushing_per_cat = 5 →
  num_cats = 3 →
  total_free_time = 180 →
  free_time_left = 30 →
  ((total_free_time - free_time_left) - (time_dusting + time_mopping + time_brushing_per_cat * num_cats) = 45) :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  apply rfl

end vacuuming_time_l786_786450


namespace blue_triangle_coloring_not_more_than_four_blue_triangles_l786_786494

theorem blue_triangle_coloring
  (P : Finset (ℝ × ℝ × ℝ)) (h₁ : P.card = 8)
  (h₂ : ∀ S ∈ P.powerset.filter (λ s, s.card = 4), ¬∃ a ∈ S, ∃ b ∈ S, ∃ c ∈ S, ∃ d ∈ S, affine_independent ℝ ![a, b, c, d])
  (blue_edges : Finset (P × P)) (h₃ : blue_edges.card = 17)
  (red_edges : Finset (P × P)) (h₄ : ∀ (e : P × P), e ∉ blue_edges ↔ e ∈ red_edges):
  ∃ S : Finset (P × P × P), S.card ≥ 4 ∧ ∀ (triangle ∈ S), ∀ e ∈ triangle, e ∈ blue_edges :=
sorry

theorem not_more_than_four_blue_triangles
  (P : Finset (ℝ × ℝ × ℝ)) (h₁ : P.card = 8)
  (h₂ : ∀ S ∈ P.powerset.filter (λ s, s.card = 4), ¬∃ a ∈ S, ∃ b ∈ S, ∃ c ∈ S, ∃ d ∈ S, affine_independent ℝ ![a, b, c, d])
  (blue_edges : Finset (P × P)) (h₃ : blue_edges.card = 17)
  (red_edges : Finset (P × P)) (h₄ : ∀ (e : P × P), e ∉ blue_edges ↔ e ∈ red_edges)
  (h₅ : ∀ S : Finset (P × P × P), S.card ≥ 5 → ∃ (triangle ∈ S), ∀ e ∈ triangle, e ∈ blue_edges) :
  false :=
sorry

end blue_triangle_coloring_not_more_than_four_blue_triangles_l786_786494


namespace number_of_obtuse_triangles_l786_786939

theorem number_of_obtuse_triangles : 
  let angles := { (α : ℕ) | 1 ≤ α ∧ α ≤ 44 }; 
  let num_obtuse_triangles := ∑ α in finset.range 45 \ finset.singleton 0, (89 - 2 * α);
  num_obtuse_triangles = 1936 := 
by
  sorry

end number_of_obtuse_triangles_l786_786939


namespace probability_of_point_in_triangle_l786_786658

noncomputable def radius := R
def side_length_of_triangle := √3 * radius
def area_of_circle := π * radius^2
def area_of_triangle := (√3 / 4) * side_length_of_triangle^2
def probability := area_of_triangle / area_of_circle

theorem probability_of_point_in_triangle :
  probability = (3 * √3) / (4 * π) := sorry

end probability_of_point_in_triangle_l786_786658


namespace sum_f_eq_328053_l786_786391

open Finset

noncomputable def f (x : ℕ) : ℕ := x^2 - 4 * x + 100
def g : Finset ℕ := (finset.range 101).filter (λ x, x ≠ 0)

theorem sum_f_eq_328053 : (∑ x in g, f x) = 328053 := by 
  sorry

end sum_f_eq_328053_l786_786391


namespace find_pqr_l786_786078

def A := ![
  ![0, 1, 2],
  ![1, 0, 1],
  ![2, 1, 0]
]

def I := ![
  ![1, 0, 0],
  ![0, 1, 0],
  ![0, 0, 1]
]

def Z := ![
  ![0, 0, 0],
  ![0, 0, 0],
  ![0, 0, 0]
]

def matrix_eqn (p q r : ℝ) :=
  let A2 := A * A
  let A3 := A * A2
  A3 + p • A2 + q • A + r • I = Z

theorem find_pqr : matrix_eqn 0 (-6) (-4) :=
by sorry

end find_pqr_l786_786078


namespace temperature_decrease_l786_786958

theorem temperature_decrease (rise_1_degC : ℝ) (decrease_2_degC : ℝ) 
  (h : rise_1_degC = 1) : decrease_2_degC = -2 :=
by 
  -- This is the statement with the condition and problem to be proven:
  sorry

end temperature_decrease_l786_786958


namespace hyperbola_standard_eq_proof_right_triangle_proof_l786_786887

-- Definitions based on the conditions
def hyperbola_center_origin (C : set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), (x, y) ∈ C ↔ (x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0)

def hyperbola_foci_x_axis (C : set (ℝ × ℝ)) (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (-5, 0) ∧ F2 = (5, 0)

def hyperbola_eccentricity (c e a : ℝ) : Prop :=
  e = c / a ∧ e = 5

def point_on_hyperbola_branch (P F1 F2 : ℝ × ℝ) (SP : ℝ) : Prop :=
  |P - F1| + |P - F2| = SP

def hyperbola_standard_eq (C : set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a = 1 ∧ b^2 = 24 ∧ ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 - y^2 / 24 = 1

def right_triangle (P F1 F2 : ℝ × ℝ) : Prop :=
  P.1^2 + F1.1^2 = F2.1^2

-- Theorem statements to be proved
theorem hyperbola_standard_eq_proof :
  ∀ (C : set (ℝ × ℝ)), hyperbola_center_origin C →
                      hyperbola_foci_x_axis C (-5, 0) (5, 0) →
                      hyperbola_eccentricity 5 5 1 →
                      hyperbola_standard_eq C :=
by sorry

theorem right_triangle_proof :
  ∀ (P F1 F2 : ℝ × ℝ), point_on_hyperbola_branch (8, 0) (-5, 0) (5, 0) 14 →
                        |(-5, 0) - (5, 0)| = 10 →
                        right_triangle (8, 0) (-5, 0) (5, 0) :=
by sorry

end hyperbola_standard_eq_proof_right_triangle_proof_l786_786887


namespace train_length_proof_l786_786326

/-- Given a train's speed of 45 km/hr, time to cross a bridge of 30 seconds, and the bridge length of 225 meters, prove that the length of the train is 150 meters. -/
theorem train_length_proof (speed_km_hr : ℝ) (time_sec : ℝ) (bridge_length_m : ℝ) (train_length_m : ℝ)
    (h_speed : speed_km_hr = 45) (h_time : time_sec = 30) (h_bridge_length : bridge_length_m = 225) :
  train_length_m = 150 :=
by
  sorry

end train_length_proof_l786_786326


namespace min_radius_circle_line_intersection_l786_786911

theorem min_radius_circle_line_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) (r : ℝ) (hr : r > 0)
    (intersect : ∃ (x y : ℝ), (x - Real.cos θ)^2 + (y - Real.sin θ)^2 = r^2 ∧ 2 * x - y - 10 = 0) :
    r ≥ 2 * Real.sqrt 5 - 1 :=
  sorry

end min_radius_circle_line_intersection_l786_786911


namespace min_positive_period_of_f_triangle_b_value_l786_786438

noncomputable def f (x : Real) : Real := Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem min_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) :=
by
  use Real.pi
  sorry

theorem triangle_b_value (c : Real) (cosB : Real) (f_val : f (C / 2) = -1 / 4)
  (c_value : c = Real.sqrt 6) (cosB_value : cosB = 1 / 3) :
  ∃ b, b = 8 / 3 :=
by
  have sinB := Real.sqrt (1 - (cosB ^ 2))
  have sinB_val : sinB = 2 * Real.sqrt 2 / 3 := sorry
  have sinC := Real.sqrt 3 / 2
  have b := c * sinB / sinC
  use b
  rw [c_value, sinB_val, mul_div_assoc, mul_comm (Real.sqrt 6), div_mul_cancel, eq_comm, eq_div_iff] at b
  simp at b
  have b_val : b = 8 / 3 := sorry
  exact ⟨b, b_val⟩
  sorry

end min_positive_period_of_f_triangle_b_value_l786_786438


namespace gcd_of_triangle_sides_l786_786998

-- Define a structure for a triangle with integer sides
structure Triangle :=
  (a b c : ℕ) -- sides are natural numbers
  (AB_lt_AC : b < c) -- AB < AC

-- Given: a triangle with integer sides and a specific condition on the sides
-- the tangent to the circumcircle at A intersects BC at D, and AD is an integer
noncomputable def gcd_condition (T : Triangle) : Prop :=
  ∃ D : ℕ, gcd T.b T.c > 1

-- The proof problem statement
theorem gcd_of_triangle_sides (T : Triangle) (D : ℕ) (AD_integer : D ∈ ℕ) : gcd_condition T :=
  sorry

end gcd_of_triangle_sides_l786_786998


namespace number_of_ways_to_choose_materials_l786_786236

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786236


namespace find_z_l786_786813

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786813


namespace dice_prob_at_least_one_three_l786_786665

theorem dice_prob_at_least_one_three :
  let outcomes := [(1, 1, 1), (1, 2, 2), (2, 1, 2), (1, 3, 3), (2, 2, 4), (3, 1, 4), 
                   (3, 3, 6), (4, 2, 6), (5, 1, 6)] in
  let valid_outcomes := filter (λ (t : ℕ × ℕ × ℕ), t.1 + t.2 = 2 * t.3) outcomes in
  let favorable_outcomes := filter (λ (t : ℕ × ℕ × ℕ), t.1 = 3 ∨ t.2 = 3 ∨ t.3 = 3) valid_outcomes in
  (favorable_outcomes.length : ℚ) / (valid_outcomes.length : ℚ) = 1 / 3 :=
by
  sorry

end dice_prob_at_least_one_three_l786_786665


namespace largest_initial_number_l786_786024

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786024


namespace circle_center_l786_786738

theorem circle_center (x y : ℝ) : ∀ (h k : ℝ), (x^2 - 6*x + y^2 + 2*y = 9) → (x - h)^2 + (y - k)^2 = 19 → h = 3 ∧ k = -1 :=
by
  intros h k h_eq c_eq
  sorry

end circle_center_l786_786738


namespace homogeneous_variances_l786_786407

noncomputable def sample_sizes : (ℕ × ℕ × ℕ) := (9, 13, 15)
noncomputable def sample_variances : (ℝ × ℝ × ℝ) := (3.2, 3.8, 6.3)
noncomputable def significance_level : ℝ := 0.05
noncomputable def degrees_of_freedom : ℕ := 2
noncomputable def V : ℝ := 1.43
noncomputable def critical_value : ℝ := 6.0

theorem homogeneous_variances :
  V < critical_value :=
by
  sorry

end homogeneous_variances_l786_786407


namespace find_cost_price_l786_786280

variable (CP : ℝ)

def SP1 : ℝ := 0.80 * CP
def SP2 : ℝ := 1.06 * CP

axiom cond1 : SP2 - SP1 = 520

theorem find_cost_price : CP = 2000 :=
by
  sorry

end find_cost_price_l786_786280


namespace sequence_properties_l786_786398

variable (x : ℕ → ℝ) (x0 : ℝ) (Sn : ℕ → ℝ)

-- The given condition on the sequence
axiom seq_rel : ∀ n, 2 * x (n) = (∑ i in Finset.range n, x i) - x n

-- Define the expressions for x_n and S_n according to the problem's answers
def x_n (n : ℕ) : ℝ := if n = 0 then x0 else x0 * (4^(n-1)) / (3^n)
def S_n (n : ℕ) : ℝ := x0 * (4 / 3)^n

-- Proof that the expressions derived for x_n and S_n correspond to the sequences
theorem sequence_properties : 
  (∀ n, (x n = x_n x0 n)) ∧ 
  (∀ n, (Sn n = S_n x0 n)) := 
by 
  sorry

end sequence_properties_l786_786398


namespace solution_set_of_inequality_l786_786164

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x^2 + 2 < x} = {x : ℝ | x < -2 / 3} ∪ {x : ℝ | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l786_786164


namespace go_quantity_range_l786_786618

variable (m : ℕ)

def chess_and_go_sets := 120
def chess_price := 25
def go_price := 30
def total_cost_limit := 3500

def condition1 := m ≥ 2 * (chess_and_go_sets - m)
def condition2 := go_price * m + chess_price * (chess_and_go_sets - m) ≤ total_cost_limit

theorem go_quantity_range
  (m : ℕ) (h1 : condition1) (h2 : condition2) :
  80 ≤ m ∧ m ≤ 100 :=
sorry

end go_quantity_range_l786_786618


namespace prod_a2_a6_l786_786150

variable {a : ℕ → ℝ} (q : ℝ)

def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 0 = 3 ∧ a 1 = 3*q ∧ a 2 = 3*q^2 ∧ a 3 = 3*q^3 ∧ a 4 = 3*q^4 ∧ a 5 = 3*q^5 ∧ a 6 = 3*q^6

def cond (a : ℕ → ℝ) : Prop := 
  a 0 = 3 ∧ a 0 + a 2 + a 4 = 21

theorem prod_a2_a6 (h : cond a) (hs : geometric_seq a q) :
  (a 1) * (a 5) = 72 :=
by sorry

end prod_a2_a6_l786_786150


namespace smallest_integer_divisible_by_2018_l786_786361

theorem smallest_integer_divisible_by_2018 (A B : ℕ) (h1 : odd_digits A) (h2 : divisible_by_2018 A) (h3 : removing_middle_digit A = B) (h4 : divisible_by_2018 B) : A = 100902018 :=
sorry

def odd_digits (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  digits.length % 2 = 1

def divisible_by_2018 (n : ℕ) : Prop :=
  n % 2018 = 0

def removing_middle_digit (n : ℕ) : ℕ :=
  let digits := n.to_digits 10
  let k := digits.length / 2
  list.foldl (λ (acc : ℕ) (d : ℕ), acc*10 + d) 0 (digits.take k ++ digits.drop (k+1))

end smallest_integer_divisible_by_2018_l786_786361


namespace ball_distribution_l786_786460

theorem ball_distribution :
  let balls := 7
  let boxes := 3
  let total_ways := 1 + 7 + 21 + 21 + 35 + 105 + 70 + 105
  balls = 7 ∧ boxes = 3 → total_ways = 365 :=
by
  intro h
  simp
  exact rfl

end ball_distribution_l786_786460


namespace angle_sum_eq_128_l786_786492

theorem angle_sum_eq_128 (A B C x y : ℝ) (hA : A = 28) (hB : B = 74) (hC : C = 26)
  (h_polygon_sum : 468 + A + B - x - y = 540) : x + y = 128 := by
  rw [hA, hB, hC] at h_polygon_sum
  linarith

end angle_sum_eq_128_l786_786492


namespace find_multiple_of_ron_l786_786553

variable (R_d R_g R_n m : ℕ)

def rodney_can_lift_146 : Prop := R_d = 146
def combined_weight_239 : Prop := R_d + R_g + R_n = 239
def rodney_twice_as_roger : Prop := R_d = 2 * R_g
def roger_seven_less_than_multiple_of_ron : Prop := R_g = m * R_n - 7

theorem find_multiple_of_ron (h1 : rodney_can_lift_146 R_d) 
                             (h2 : combined_weight_239 R_d R_g R_n) 
                             (h3 : rodney_twice_as_roger R_d R_g) 
                             (h4 : roger_seven_less_than_multiple_of_ron R_g R_n m) 
                             : m = 4 :=
by 
    sorry

end find_multiple_of_ron_l786_786553


namespace distinct_odd_rearrangements_l786_786359

-- Definitions based on the given conditions
def digits := {3, 4, 3, 9, 6}
def is_odd (n : Nat) := n % 2 = 1
def rearrangements := {n | ∃ (l : List Nat), l.perm digits.toList ∧ is_odd (List.head l.getLast)}

theorem distinct_odd_rearrangements :
  rearrangements.card = 36 :=
sorry

end distinct_odd_rearrangements_l786_786359


namespace order_of_numbers_l786_786360

theorem order_of_numbers : 0.4^3 < 3^(0.3) ∧ 3^(0.3) < 3^(0.4) 
:= by
  -- Given conditions
  have h1 : ∀ x y : ℝ, x < y → 3^x < 3^y := λ x y hxy, Real.rpow_lt_rpow_of_exponent_lt (by norm_num; linarith) (by norm_num; linarith) hxy,
  have h2 : 0.4^3 < 1 := by norm_num,
  -- Demonstrate our proof by combining the above conditions
  sorry

end order_of_numbers_l786_786360


namespace trigonometric_identity_l786_786759

noncomputable def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem trigonometric_identity :
  special_operation (Real.sin (Real.pi / 12)) (Real.cos (Real.pi / 12))
  = - (1 + 2 * Real.sqrt 3) / 4 :=
by
  sorry

end trigonometric_identity_l786_786759


namespace num_ways_choose_materials_l786_786209

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786209


namespace pq_over_ef_l786_786118

-- Define the conditions given in the problem
variables (A B C D E F G : ℝ × ℝ)
variables (AB BC EB CG DF : ℝ)
variables (P Q : ℝ × ℝ)
variables (EF PQ : ℝ)

-- Given conditions
axiom h1 : A = (0, 5) ∧ B = (6, 5) ∧ C = (6, 0) ∧ D = (0, 0)
axiom h2 : E = (4, 5) ∧ F = (3, 0) ∧ G = (6, 3)
axiom h3 : AB = 6 ∧ BC = 5
axiom h4 : EB = 2 ∧ CG = 2 ∧ DF = 3
axiom h5 : EF = √((4 - 3) ^ 2 + (5 - 0) ^ 2)
axiom h6 : P = ((24 / 7), (15 / 7))
axiom h7 : Q = ((15 / 4), (15 / 4))
axiom h8 : PQ = abs(((15 / 4) - (24 / 7)))

-- Prove that PQ / EF = 9 / (28 * √26)
theorem pq_over_ef : PQ / EF = 9 / (28 * √26) :=
sorry

end pq_over_ef_l786_786118


namespace largest_initial_number_l786_786038

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786038


namespace rect_area_from_max_min_expr_l786_786592

theorem rect_area_from_max_min_expr : 
  (∃ (x1 x2 : ℝ), x1 < 0 ∧ x2 > 0 ∧
    x1 + 1/x1 = -2 ∧ x2 + 1/x2 = 2 ∧
    ∃ (x3 x4 y1 y2 y3 y4 : ℝ), 
    (x3, y1) = (x2, 2) ∧
    (x4, y2) = (x1, -2) ∧
    (x3, y3) = (x2, -2) ∧
    (x4, y4) = (x1, 2)) → 
    8 := 
begin
  sorry
end

end rect_area_from_max_min_expr_l786_786592


namespace perfect_square_trinomial_l786_786469

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, a^2 = 1 ∧ b^2 = 1 ∧ x^2 + m * x * y + y^2 = (a * x + b * y)^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end perfect_square_trinomial_l786_786469


namespace largest_initial_number_l786_786047

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786047


namespace cross_section_is_regular_hexagon_l786_786362

-- Define the cube structure and conditions
structure Cube (α : Type*) :=
  (center : α)
  (diagonal : α × α)

-- Define the plane passing through the center and perpendicular to the diagonal
structure CuttingPlane (α : Type*) :=
  (cube : Cube α)
  (intersection : α)
  (is_center_of_cube : intersection = cube.center)
  (is_perpendicular_to_diagonal : ∃ vec, vec ∈ vector_space α ∧ angle vec cube.diagonal = 90)

-- Proof that the cross-section is a regular hexagon
theorem cross_section_is_regular_hexagon {α : Type*} [field α] 
  [vector_space α] (P : CuttingPlane α) : 
  ∃ hexagon : RegularHexagon α, 
    (hexagon.center = P.cube.center ∧ ∀ angle, angle ∈ [0, 60, 120, 180, 240, 300], rotate(angle, hexagon) = hexagon) :=
sorry

end cross_section_is_regular_hexagon_l786_786362


namespace sum_of_squares_of_root_pairs_eq_400_l786_786530

theorem sum_of_squares_of_root_pairs_eq_400
  (p q r : ℝ)
  (h : polynomial.eval p (polynomial.C 1 * polynomial.X^3 - polynomial.C 15 * polynomial.X^2 + polynomial.C 25 * polynomial.X - polynomial.C 10) = 0)
  (hq : polynomial.eval q (polynomial.C 1 * polynomial.X^3 - polynomial.C 15 * polynomial.X^2 + polynomial.C 25 * polynomial.X - polynomial.C 10) = 0)
  (hr : polynomial.eval r (polynomial.C 1 * polynomial.X^3 - polynomial.C 15 * polynomial.X^2 + polynomial.C 25 * polynomial.X - polynomial.C 10) = 0) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 := 
sorry

end sum_of_squares_of_root_pairs_eq_400_l786_786530


namespace not_in_set_M_in_set_M_for_any_b_l786_786419

-- Define the set M of functions satisfying the condition
def M (f : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, f(t + 2) = f(t) + f(2)

-- Problem statement (1): Prove that f(x) = 3x + 2 does not belong to set M
theorem not_in_set_M : ¬ M (λ x : ℝ, 3 * x + 2) :=
sorry

-- Problem statement (2): Prove that f(x) = 2^x + bx^2 belongs to set M for any real number b
theorem in_set_M_for_any_b (b : ℝ) : M (λ x : ℝ, 2^x + b * x^2) :=
sorry

end not_in_set_M_in_set_M_for_any_b_l786_786419


namespace two_students_choose_materials_l786_786189

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786189


namespace max_height_l786_786298

noncomputable def height (t : ℝ) : ℝ :=
  -16 * t^2 + 40 * t + 25

theorem max_height : ∃ t : ℝ, height t = 50 :=
by
  use 1.25
  -- Proof of maximum height computation skipped
  sorry

end max_height_l786_786298


namespace correct_calculation_l786_786636

theorem correct_calculation (a : ℝ) :
  (¬ (a^2 + a^2 = a^4)) ∧ (¬ (a^2 * a^3 = a^6)) ∧ (¬ ((a + 1)^2 = a^2 + 1)) ∧ ((-a^2)^2 = a^4) :=
by
  sorry

end correct_calculation_l786_786636


namespace largest_initial_number_l786_786062

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786062


namespace sum_of_squares_second_15_eq_8215_l786_786643

theorem sum_of_squares_second_15_eq_8215:
  (∑ i in Finset.range 30.succ, (i + 1) ^ 2) - (∑ i in Finset.range 15.succ, i ^ 2) = 8215 := by
  have sum_first_15 : (∑ i in Finset.range 15.succ, i ^ 2) = 1240 := sorry
  calc
    (∑ i in Finset.range 30.succ, i ^ 2) - 1240 = 8215 := sorry

end sum_of_squares_second_15_eq_8215_l786_786643


namespace probability_of_sum_18_is_1_over_216_l786_786615

noncomputable def probability_sum_18 : ℚ :=
  let die_faces := finset.range 6 + 1 -- Die faces numbered from 1 to 6
  let event := finset.filter (λ (tup : (ℕ × ℕ × ℕ)), tup.1 + tup.2 + tup.3 = 18) 
                            (finset.product (finset.product die_faces die_faces) die_faces)
  event.card * ((6 : ℚ)⁻¹) ^ 3 

theorem probability_of_sum_18_is_1_over_216
  (dice : list (finset ℕ)) (h : ∀ d ∈ dice, d = finset.range 6 + 1) :
  probability_sum_18 = 1 / 216 :=
sorry

end probability_of_sum_18_is_1_over_216_l786_786615


namespace find_number_l786_786949

theorem find_number (x : ℝ) : 
  (72 = 0.70 * x + 30) -> x = 60 :=
by
  sorry

end find_number_l786_786949


namespace set_A_enumeration_l786_786162

-- Define the conditions of the problem.
def A : Set ℕ := { x | ∃ (n : ℕ), 6 = n * (6 - x) }

-- State the theorem to be proved.
theorem set_A_enumeration : A = {0, 2, 3, 4, 5} :=
by
  sorry

end set_A_enumeration_l786_786162


namespace voting_problem_l786_786006

theorem voting_problem (x y x' y' : ℕ) (m : ℕ) (h1 : x + y = 500) (h2 : y > x)
    (h3 : y - x = m) (h4 : x' = (10 * y) / 9) (h5 : x' + y' = 500)
    (h6 : x' - y' = 3 * m) :
    x' - x = 59 := 
sorry

end voting_problem_l786_786006


namespace solve_for_z_l786_786827

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786827


namespace EF_squared_l786_786569

noncomputable def square_side : ℝ := 15
noncomputable def BE : ℝ := 6
noncomputable def DF : ℝ := 6
noncomputable def AE : ℝ := 13
noncomputable def CF : ℝ := 13

/-- Given a square with side length 15, and points E and F outside of the square such that BE = DF = 6 and AE = CF = 13,
    prove that the square of the distance between points E and F (EF squared) is 226.
 -/
theorem EF_squared : 
    let s := square_side,
        be := BE,
        df := DF,
        ae := AE,
        cf := CF in
    EF^2 = 226 := 
by
    sorry

end EF_squared_l786_786569


namespace min_positive_period_of_f_l786_786440

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + 0.5

theorem min_positive_period_of_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
by
    sorry

end min_positive_period_of_f_l786_786440


namespace shaded_region_area_is_15_l786_786491

noncomputable def area_of_shaded_region : ℝ :=
  let radius := 1
  let area_of_one_circle := Real.pi * (radius ^ 2)
  4 * area_of_one_circle + 3 * (4 - area_of_one_circle)

theorem shaded_region_area_is_15 : 
  abs (area_of_shaded_region - 15) < 1 :=
by
  exact sorry

end shaded_region_area_is_15_l786_786491


namespace angle_between_a_b_is_pi_l786_786932

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  let norm_a := Real.sqrt (a.1^2 + a.2^2) in
  let norm_b := Real.sqrt (b.1^2 + b.2^2) in
  Real.arccos (dot_product / (norm_a * norm_b))

theorem angle_between_a_b_is_pi
  (a b : ℝ × ℝ)
  (h1 : a.1 + 2 * b.1 = 2 ∧ a.2 + 2 * b.2 = -4)
  (h2 : 3 * a.1 - b.1 = -8 ∧ 3 * a.2 - b.2 = 16) :
  angle_between_vectors a b = Real.pi :=
by
  sorry -- Proof goes here

end angle_between_a_b_is_pi_l786_786932


namespace find_t_closest_to_a_l786_786392

-- Definitions for the vectors involved in the problem
def v (t : ℝ) : ℝ × ℝ × ℝ := (3 + 8 * t, - 1 + 2 * t, - 2 - 3 * t)
def a : ℝ × ℝ × ℝ := (1, 7, 1)

-- Definition of the dot product of two 3D vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Orthogonality condition: (v(t) - a) · direction = 0
def orthogonality_condition (t : ℝ) : Prop :=
  dot_product (v t - a) (8, 2, -3) = 0

-- The theorem to prove the value of t
theorem find_t_closest_to_a : 
  t : ℝ := -1/7
   :=
by {
  sorry
}


end find_t_closest_to_a_l786_786392


namespace max_real_roots_of_polynomial_l786_786740

theorem max_real_roots_of_polynomial (n : ℕ) (c : ℝ) (h1 : 0 < n) (h2 : c = -n - 1) :
  (∀ x : ℝ, x^n + x^(n - 1) + ... + x + c = 0 → (x = 1)) ∨ (∀ x : ℝ, x^n + x^(n - 1) + ... + x + c = 0 → (x ≠ -1)) :=
sorry

end max_real_roots_of_polynomial_l786_786740


namespace students_material_selection_l786_786221

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786221


namespace conformal_map_upper_half_plane_l786_786379

noncomputable def conformal_map (α : ℝ) (hα : 0 < α ∧ α < 2) (z : ℂ) (hz : 0 < z.im) : ℂ :=
z^α

theorem conformal_map_upper_half_plane (α : ℝ) (hα : 0 < α ∧ α < 2) (z : ℂ) (hz : 0 < z.im) :
  0 < (conformal_map α hα z hz).arg ∧ (conformal_map α hα z hz).arg < α * Real.pi :=
sorry

end conformal_map_upper_half_plane_l786_786379


namespace number_of_true_statements_l786_786973

-- Definitions of lines l, m, n, and planes α, β, γ
variables (l m n : Line) (α β γ : Plane)

-- Conditions
axiom diff_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Statements
def statement1 := (α ⊥ β ∧ β ⊥ γ ∧ α ∩ β = l) → l ⊥ γ
def statement2 := (l ∥ α ∧ l ∥ β ∧ α ∩ β = m) → l ∥ m
def statement3 := (α ∩ β = l ∧ β ∩ γ = m ∧ γ ∩ α = n ∧ l ∥ m) → l ∥ n
def statement4 := (α ⊥ γ ∧ β ⊥ γ) → (α ⊥ β ∨ α ∥ β)

-- Correct answer is 3 true statements out of 4
theorem number_of_true_statements : 
  (statement1 α β γ l m ∧ statement2 α β l m ∧ statement3 α β γ l m n ∧ ¬ statement4 α γ β) → true_statements = 3 :=
sorry

end number_of_true_statements_l786_786973


namespace chord_arithmetic_sequence_length_l786_786976

theorem chord_arithmetic_sequence_length :
  let circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 5 * x = 0
  in let point : ℝ × ℝ := (5/2, 3/2)
  in let valid_d : ℝ → Prop := λ d, -1/6 < d ∧ d ≤ 1/3
  in ∀ n : ℕ, 
       (∃ (chord_lengths : list ℝ), 
          ∀ i, 
              0 ≤ i → i < chord_lengths.length - 1 
              → (chord_lengths.get_or_else i 0) + d = chord_lengths.get_or_else (i + 1) 0 
              → circle_eq (chord_lengths.get_or_else i 0) (chord_lengths.get_or_else (i + 1) 0)) 
       → (valid_d d) 
       → n ∈ {3, 4, 5} :=
by
  sorry

end chord_arithmetic_sequence_length_l786_786976


namespace probability_of_tamika_greater_than_carlos_l786_786139

theorem probability_of_tamika_greater_than_carlos :
  let tamika_set := {7, 8, 9}
  let carlos_set := {2, 4, 5}
  let tamika_results := {abs (x - y) | x ∈ tamika_set, y ∈ tamika_set, x ≠ y}
  let carlos_results := {x * y | x ∈ carlos_set, y ∈ carlos_set, x ≠ y}
  let favorable_outcomes := { (t, c) | t ∈ tamika_results, c ∈ carlos_results, t > c }
  let total_outcomes := { (t, c) | t ∈ tamika_results, c ∈ carlos_results }
  (favorable_outcomes.card / total_outcomes.card) = 0 := by
    sorry

end probability_of_tamika_greater_than_carlos_l786_786139


namespace golden_matrix_only_exists_for_1_l786_786332

def isGoldenMatrix (n : ℕ) (A : Matrix (Fin n) (Fin n) ℤ) : Prop :=
  ∀ i j : Fin n, ((Finset.univ.biUnion (λ k : Fin n, {A i k} ∪ {A k j})) = (Finset.range (2 * n - 1)).map Fin.val)

theorem golden_matrix_only_exists_for_1 (n : ℕ) (A : Matrix (Fin n) (Fin n) ℤ) : 
  isGoldenMatrix n A → n = 1 :=
by
  sorry

end golden_matrix_only_exists_for_1_l786_786332


namespace max_x1_squared_plus_x2_squared_l786_786291

theorem max_x1_squared_plus_x2_squared (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = k - 2)
  (h2 : x₁ * x₂ = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) :
  x₁ ^ 2 + x₂ ^ 2 ≤ 18 :=
sorry

end max_x1_squared_plus_x2_squared_l786_786291


namespace evaluate_f_at_2_l786_786950

def f (x : ℝ) : ℝ := x^2 - x

theorem evaluate_f_at_2 : f 2 = 2 := by
  sorry

end evaluate_f_at_2_l786_786950


namespace num_ways_choose_materials_l786_786207

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786207


namespace ones_digit_of_power_l786_786389

theorem ones_digit_of_power (a b : ℕ) : (34^{34 * (17^{17})}) % 10 = 4 :=
by
  sorry

end ones_digit_of_power_l786_786389


namespace four_digit_numbers_count_l786_786453

def digits : Set ℕ := {1, 2, 3, 4, 5, 6}
def odd_digits : Set ℕ := {1, 3, 5}
def even_digits : Set ℕ := {2, 4, 6}

theorem four_digit_numbers_count :
  let even_count := (Set.card even_digits).choose 2
      odd_count := (Set.card odd_digits).choose 2
      arrangement := (4.factorial) in
  even_count * odd_count * arrangement = 216 :=
by sorry

end four_digit_numbers_count_l786_786453


namespace angle_RPS_given_angles_l786_786011

-- Definitions of angles
variables {α β γ Δ : ℝ}

-- Given the conditions
theorem angle_RPS_given_angles :
  α = 35 ∧ β = 80 ∧ γ = 25 →
  Δ = 180 - α - γ →
  RPS = Δ - β →
  RPS = 40 :=
by {
  intros h1 h2 h3,
  -- Using h1, h2, and h3, we are to show the answer
  sorry
}

end angle_RPS_given_angles_l786_786011


namespace tangent_log_line_value_a_l786_786954

theorem tangent_log_line_value_a :
  (∃ (a : ℝ), (∀ x : ℝ, x > 0 → f(x) = log a x) ∧
                 (line y = (1 / 3) * x) ∧ 
                 (∃ (m : ℝ), (log a m = 1/3 * m ∧
                             (1 / (m * log a) = 1 / 3))) → 
                 a = exp (3 / exp 1)) :=
by 
  sorry

end tangent_log_line_value_a_l786_786954


namespace students_count_at_least_l786_786481

theorem students_count_at_least (R B : Type) (hR : R → Prop) (hB : B → Prop) 
(hR_card : cardinal.mk R = 18) 
(hB_card : cardinal.mk B = 15) 
(hRB_card : cardinal.mk (set_of (λ x : R × B, hR x.1 ∧ hB x.2)) = 7) : 
cardinal.mk (set_of (λ x : R × B, hR x.1 ∨ hB x.2)) = 26 :=
sorry

end students_count_at_least_l786_786481


namespace total_players_l786_786287

def cricket_players : ℕ := 15
def hockey_players : ℕ := 12
def football_players : ℕ := 13
def softball_players : ℕ := 15

theorem total_players : cricket_players + hockey_players + football_players + softball_players = 55 :=
by {
  show 15 + 12 + 13 + 15 = 55,
  sorry
}

end total_players_l786_786287


namespace evaluate_fraction_l786_786731

theorem evaluate_fraction : (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = (8 / 21) :=
by
  sorry

end evaluate_fraction_l786_786731


namespace find_rate_squares_sum_l786_786728

theorem find_rate_squares_sum {b j s : ℤ} 
(H1 : 3 * b + 2 * j + 2 * s = 112)
(H2 : 2 * b + 3 * j + 4 * s = 129) : b^2 + j^2 + s^2 = 1218 :=
by sorry

end find_rate_squares_sum_l786_786728


namespace area_of_pointSet_l786_786255

open Set

def pointSet := { p : ℝ × ℝ | abs(p.1 + p.2) + abs(p.1 - p.2) ≤ 4 }

theorem area_of_pointSet : measure_theory.measure_space.volume (pointSet) = 16 :=
sorry

end area_of_pointSet_l786_786255


namespace ratio_of_areas_l786_786570

-- Definitions based on the conditions given
def square_side_length : ℕ := 48
def rectangle_width : ℕ := 56
def rectangle_height : ℕ := 63

-- Areas derived from the definitions
def square_area := square_side_length * square_side_length
def rectangle_area := rectangle_width * rectangle_height

-- Lean statement to prove the ratio of areas
theorem ratio_of_areas :
  (square_area : ℚ) / rectangle_area = 2 / 3 := 
sorry

end ratio_of_areas_l786_786570


namespace central_angle_of_sector_l786_786968

noncomputable def sector_radius : ℝ := 10
noncomputable def sector_area : ℝ := 100
noncomputable def sector_formula (r : ℝ) (area : ℝ) : ℝ := area / (π * r^2) * 2 * π

theorem central_angle_of_sector :
  sector_formula sector_radius sector_area = 2 :=
by
  sorry

end central_angle_of_sector_l786_786968


namespace math_problem_l786_786254

theorem math_problem
  (a b c d : ℚ)
  (h₁ : a = 1 / 3)
  (h₂ : b = 1 / 6)
  (h₃ : c = 1 / 9)
  (h₄ : d = 1 / 18) :
  9 * (a + b + c + d)⁻¹ = 27 / 2 := 
sorry

end math_problem_l786_786254


namespace smallest_four_digit_divisible_by_six_l786_786260

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end smallest_four_digit_divisible_by_six_l786_786260


namespace positive_difference_two_largest_prime_factors_of_380899_l786_786628

theorem positive_difference_two_largest_prime_factors_of_380899 :
  let prime_factors := [379, 3, 5, 67] in
  prime_factors.contains 379 ∧ prime_factors.contains 67 ∧ (379 - 67 = 312) := 
by
  let prime_factors := [379, 3, 5, 67]
  have h_pf_379 : 379 ∈ prime_factors := by simp
  have h_pf_67 : 67 ∈ prime_factors := by simp
  have h_diff : 379 - 67 = 312 := by norm_num
  exact ⟨h_pf_379, h_pf_67, h_diff⟩

end positive_difference_two_largest_prime_factors_of_380899_l786_786628


namespace shortest_side_of_right_triangle_l786_786329

theorem shortest_side_of_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) (c : ℝ) (right_triangle : a^2 + b^2 = c^2) :
  min a b = 5 :=
by
  rw [ha, hb],
  sorry

end shortest_side_of_right_triangle_l786_786329


namespace treats_total_l786_786686

theorem treats_total :
  let chewingGums := 60
  let chocolateBars := 55
  let lollipops := 70
  let cookies := 50
  let candies := 40
  let total := chewingGums + chocolateBars + lollipops + cookies + candies
  in total = 275 :=
by
  let chewingGums := 60
  let chocolateBars := 55
  let lollipops := 70
  let cookies := 50
  let candies := 40
  let total := chewingGums + chocolateBars + lollipops + cookies + candies
  show total = 275
  sorry

end treats_total_l786_786686


namespace points_of_tangency_correct_l786_786140

noncomputable def points_of_tangency (p q : ℝ) : set (ℝ × ℝ) :=
  { (x, y) | x = p + √(p^2 - 4*q) ∨ x = p - √(p^2 - 4*q) ∧ 
             y = (p^2 - 2*q + p * √(p^2 - 4*q)) / 2 ∨ y = (p^2 - 2*q - p * √(p^2 - 4*q)) / 2 }

theorem points_of_tangency_correct (p q : ℝ) (h : p^2 - 4*q = 0) : 
  points_of_tangency p q = 
    { (p + √(p^2 - 4*q), (p^2 - 2*q + p * √(p^2 - 4*q)) / 2), 
      (p - √(p^2 - 4*q), (p^2 - 2*q - p * √(p^2 - 4*q)) / 2) } := 
  sorry

end points_of_tangency_correct_l786_786140


namespace sequence_general_terms_and_sum_l786_786890

-- Definitions for the arithmetic and geometric sequences
def a_n (n : ℕ) : ℝ := 2 * n - 1
def b_n (n : ℕ) : ℝ := 3^(n-1)

-- Definition of c_n based on conditions
def c_n : ℕ → ℝ
| 1     := 3
| (n+1) := 2 * (3^n)

noncomputable def S_n (n : ℕ) : ℝ :=
  if n = 1 then c_n 1
  else c_n 1 + 2 * ((3 + 3^2 + ... + 3^(n-1)))

theorem sequence_general_terms_and_sum (n : ℕ) (h : n ≥ 2) :
  (∀ k, a_n k = 2 * k - 1) ∧ (∀ k, b_n k = 3^(k - 1)) ∧
  (S_n n = 3 ^ n) := by
  sorry

end sequence_general_terms_and_sum_l786_786890


namespace solve_for_x_l786_786563

theorem solve_for_x (x : ℝ) : (x - 6)^4 = (1/16)^(-1) → x = 8 := by
  sorry

end solve_for_x_l786_786563


namespace find_five_halves_l786_786906

noncomputable def f : ℝ+ → ℝ :=
  sorry

theorem find_five_halves (f : ℝ+ → ℝ) 
  (h_add : ∀ x y : ℝ+, f (x + y) = f x + f y)
  (h_eight : f 8 = 3) : 
  f (5 / 2) = 15 / 16 :=
sorry

end find_five_halves_l786_786906


namespace number_of_ways_to_choose_materials_l786_786229

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786229


namespace length_of_TU_l786_786981

-- Define the trapezoid PQRS and its properties
variables (P Q R S T U V : Point)
variables (h_parallel : QR ∥ PS)
variables (h_QR : QR = 800)
variables (h_PS : PS = 1600)
variables (h_angle_P : ∠P = 30)
variables (h_angle_S : ∠S = 60)
variables (h_midpoint_T : midPoint QR T)
variables (h_midpoint_U : midPoint PS U)

-- Define everything in a single structure to make the theorem statement
structure Trapezoid :=
(P Q R S T U V : Point)
(QR ∥ PS : Prop)
(QR_len : QR = 800)
(PS_len : PS = 1600)
(angle_P : ∠P = 30)
(angle_S : ∠S = 60)
(midPoint_T : midPoint QR T)
(midPoint_U : midPoint PS U)

noncomputable def length_TU : Trapezoid → ℝ
| ⟨P, Q, R, S, T, U, V, h_parallel, h_QR, h_PS, h_angle_P, h_angle_S, h_midpoint_T, h_midpoint_U⟩ :=
800 - 400

-- The theorem statement
theorem length_of_TU (trapezoid : Trapezoid) : length_TU trapezoid = 400 :=
sorry

end length_of_TU_l786_786981


namespace num_of_arithmetic_sequences_l786_786455

-- Define the set of digits {1, 2, ..., 15}
def digits := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define an arithmetic sequence condition 
def is_arithmetic_sequence (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

-- Define the count of valid sequences with a specific difference
def count_arithmetic_sequences_with_difference (d : ℕ) : ℕ :=
  if d = 1 then 13
  else if d = 5 then 6
  else 0

-- Define the total count of valid sequences
def total_arithmetic_sequences : ℕ :=
  count_arithmetic_sequences_with_difference 1 +
  count_arithmetic_sequences_with_difference 5

-- The final statement to prove
theorem num_of_arithmetic_sequences : total_arithmetic_sequences = 19 := 
  sorry

end num_of_arithmetic_sequences_l786_786455


namespace not_possible_one_lies_other_not_l786_786988

-- Variable definitions: Jean is lying (J), Pierre is lying (P)
variable (J P : Prop)

-- Conditions from the problem
def Jean_statement : Prop := P → J
def Pierre_statement : Prop := P → J

-- Theorem statement
theorem not_possible_one_lies_other_not (h1 : Jean_statement J P) (h2 : Pierre_statement J P) : ¬ ((J ∨ ¬ J) ∧ (P ∨ ¬ P) ∧ ((J ∧ ¬ P) ∨ (¬ J ∧ P))) :=
by
  sorry

end not_possible_one_lies_other_not_l786_786988


namespace two_students_one_common_material_l786_786196

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786196


namespace elise_saving_correct_l786_786366

-- Definitions based on the conditions
def initial_money : ℤ := 8
def spent_comic_book : ℤ := 2
def spent_puzzle : ℤ := 18
def final_money : ℤ := 1

-- The theorem to prove the amount saved
theorem elise_saving_correct (x : ℤ) : 
  initial_money + x - spent_comic_book - spent_puzzle = final_money → x = 13 :=
by
  sorry

end elise_saving_correct_l786_786366


namespace road_network_exists_l786_786644

theorem road_network_exists (u : Fin 51 → ℕ) :
  (∀ i, 0 < u i) ∧ (u.foldl (+) 0 = 100) →
  ∃ (network : List (Fin 51 × Fin 51)),
    (∀ r : Fin 51 × Fin 51, r ∈ network → r.1 ≠ r.2) ∧
    (∀ i, u i = network.count (λ r, r.1 = i ∨ r.2 = i)) ∧
    (∀ a b, a ≠ b →
      ∃! p, (a, b) ∈ p.edges_on network ∨ (b, a) ∈ p.edges_on network ∧
      ∀ c d, c ≠ d → p.edges_on network c = p.edges_on network d) :=
sorry

end road_network_exists_l786_786644


namespace range_of_a_l786_786421

noncomputable def f (x : ℝ) : ℝ := 
  -- f is an odd function and satisfies f(x+2) = -f(x)
  sorry

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) 
  (h2 : ∀ x : ℝ, f (x + 2) = -f x) 
  (h3 : f (-1) > -2) 
  (h4 : f (-7) = (a + 1) / (3 - 2 * a)
  ) : a ∈ set.Ioo (-∞) 1 ∪ set.Ioo (3/‌2) ∞ := 
begin
  sorry
end

end range_of_a_l786_786421


namespace number_of_ways_to_choose_materials_l786_786230

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786230


namespace necessary_but_not_sufficient_l786_786464

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a - b > 0 → a^2 - b^2 > 0) ∧ ¬(a^2 - b^2 > 0 → a - b > 0) := by
sorry

end necessary_but_not_sufficient_l786_786464


namespace min_value_expression_l786_786919

open Real

/-- 
  Given that the function y = log_a(2x+3) - 4 passes through a fixed point P and the fixed point P lies on the line l: ax + by + 7 = 0,
  prove the minimum value of 1/(a+2) + 1/(4b) is 4/9, where a > 0, a ≠ 1, and b > 0.
-/
theorem min_value_expression (a b : ℝ) (h_a : 0 < a) (h_a_ne_1 : a ≠ 1) (h_b : 0 < b)
  (h_eqn : (a * -1 + b * -4 + 7 = 0) → (a + 2 + 4 * b = 9)):
  (1 / (a + 2) + 1 / (4 * b)) = 4 / 9 :=
by
  sorry

end min_value_expression_l786_786919


namespace number_of_ways_to_choose_reading_materials_l786_786245

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786245


namespace leibniz_triangle_recursive_leibniz_triangle_formula_l786_786290

open BigOperators

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Definition of Leibniz Triangle element based on the binomial coefficient
def leibniz_triangle (n k : ℕ) : ℝ := 1 / ((n + 1) * binom n k)

-- State the recursive formula for elements in the Leibniz Triangle
theorem leibniz_triangle_recursive (n k : ℕ) (h1 : 0 < k) (h2 : k ≤ n) :
  leibniz_triangle (n + 1) (k - 1) + leibniz_triangle (n + 1) k = leibniz_triangle n (k - 1) :=
by {
  -- These conditions are sufficient to skip the proof.
  sorry
}

-- State the main formula connecting Pascal's Triangle and Leibniz's Triangle
theorem leibniz_triangle_formula (n k : ℕ) (h : k ≤ n) :
  leibniz_triangle n k = 1 / ((n + 1) * binom n k) :=
by {
  -- These conditions are sufficient to skip the proof.
  sorry
}

end leibniz_triangle_recursive_leibniz_triangle_formula_l786_786290


namespace length_of_train_is_approx_500_04_l786_786279

def speed_kmh : ℝ := 100
def time_seconds : ℝ := 18
def speed_mps : ℝ := 100 * 1000 / 3600
def length_of_train : ℝ := speed_mps * time_seconds

theorem length_of_train_is_approx_500_04 :
  length_of_train ≈ 500.04 := by
  sorry

end length_of_train_is_approx_500_04_l786_786279


namespace find_length_LM_l786_786018

open_locale real

variables {A B C K L M : Type} 
variables [normed_add_comm_group A]
variables [normed_add_comm_group B]
variables [normed_add_comm_group C]
variables [normed_add_comm_group K]
variables [normed_add_comm_group L]
variables [normed_add_comm_group M]

noncomputable def is_triangle (A B C : Type) := sorry
noncomputable def angle_A_eq_90 (A B C : Type) := sorry
noncomputable def angle_B_eq_30 (A B C : Type) := sorry
noncomputable def point_on_side (K : A) (AC : A → Prop) := sorry
noncomputable def points_on_side (L M : B) (BC : B → Prop) := sorry
noncomputable def equal_segments (KL KM : K) := sorry
noncomputable def point_in_segment (L : B) (BM : B → Prop) := sorry
noncomputable def segment_lengths (A K L M : A) (lenAK : ℝ) (lenBL : ℝ) (lenMC : ℝ) := sorry
noncomputable def length_LM (L M : A) (lenLM : ℝ) := sorry

theorem find_length_LM
{A B C K L M : Type}
[is_triangle A B C]
[angle_A_eq_90 A B C]
[angle_B_eq_30 A B C]
[point_on_side K (AC)]
[points_on_side L M (BC)]
[equal_segments KL KM]
[point_in_segment L (BM)]
[segment_lengths A K L M 4 31 3]
: length_LM L M 14 := sorry

end find_length_LM_l786_786018


namespace minimal_sum_of_roots_l786_786677

noncomputable def p (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem minimal_sum_of_roots :
  (∃ p : ℝ → ℝ, ∀ x : ℝ, p(x) = x^2 - 4*x + 4 ∧ p(p(x)) = 0 ∧ leading_coeff p = 1 ∧ 
  (∀ q : ℝ → ℝ, (leading_coeff q = 1) ∧ 
  (∀ x : ℝ, q(q(x)) = 0) → sum_of_roots q ≥ 4) ) →
  p(0) = 4 :=
by
  sorry

end minimal_sum_of_roots_l786_786677


namespace find_fx_l786_786086

noncomputable def f (x : ℝ) : ℝ

axiom condition : ∀ (x y : ℝ), (f(x) * f(y) - f(x * y)) / 3 = x + y + 2

theorem find_fx : ∀ x : ℝ, f(x) = x + 3 :=
sorry

end find_fx_l786_786086


namespace arc_length_polar_curve_l786_786341

/-- The arc length of the curve given by ρ = 2φ for 0 ≤ φ ≤ 3/4 is 15/8 + 2 ln(2). -/
theorem arc_length_polar_curve :
  (∫ (φ : ℝ) in 0 .. 3/4, sqrt ((2 * φ)^2 + (2:ℝ)^2)) = 15/8 + 2 * Real.log(2) := 
by 
  sorry

end arc_length_polar_curve_l786_786341


namespace find_number_l786_786651

theorem find_number (x : ℝ) (h : 0.60 * x - 40 = 50) : x = 150 := 
by
  sorry

end find_number_l786_786651


namespace number_of_valid_numbers_l786_786452

def digits_of_2025 : List ℕ := [2, 0, 2, 5]

def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def uses_digits (n : ℕ) (digits : List ℕ) : Prop :=
  ∀ digit ∈ digits, digit ∈ (n.digits 10)

def valid_number (n : ℕ) : Prop :=
  is_four_digit_number n ∧ uses_digits n digits_of_2025

theorem number_of_valid_numbers : 
  ∃! n : ℕ, valid_number n := 6 := 
sorry

end number_of_valid_numbers_l786_786452


namespace monkeys_bananas_l786_786138

theorem monkeys_bananas (m b t : ℕ) (h : 12 * t = some minutes) :
  (72 * t = 72 minutes) → (b = 72) → m = 72 :=
by
  sorry

end monkeys_bananas_l786_786138


namespace largest_initial_number_l786_786059

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786059


namespace compute_exponent_problem_l786_786704

noncomputable def exponent_problem : ℤ :=
  3 * (3^4) - (9^60) / (9^57)

theorem compute_exponent_problem : exponent_problem = -486 := by
  sorry

end compute_exponent_problem_l786_786704


namespace find_z_l786_786800

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786800


namespace cell_same_color_neighbors_l786_786576

theorem cell_same_color_neighbors (n : ℕ) (h50 : n = 50) (colors : Fin n → Fin n → Fin 4) :
  ∃ i j : Fin n, ∀ (d : (0, -1) | (0, 1) | (-1, 0) | (1, 0)),
    colors (i + d.fst) (j + d.snd) = colors i j := 
by
  sorry

end cell_same_color_neighbors_l786_786576


namespace solve_for_x_l786_786561

/-- Prove that x = 8 under the condition that (x - 6)^4 = (1 / 16)^(-1) -/
theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)^(-1)) : x = 8 := 
by 
  sorry

end solve_for_x_l786_786561


namespace smallest_four_digit_divisible_by_6_l786_786263

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end smallest_four_digit_divisible_by_6_l786_786263


namespace supply_without_leak_last_for_20_days_l786_786680

variable (C V : ℝ)

-- Condition 1: if there is a 10-liter leak per day, the supply lasts for 15 days
axiom h1 : C = 15 * (V + 10)

-- Condition 2: if there is a 20-liter leak per day, the supply lasts for 12 days
axiom h2 : C = 12 * (V + 20)

-- The problem to prove: without any leak, the tank can supply water to the village for 20 days
theorem supply_without_leak_last_for_20_days (C V : ℝ) (h1 : C = 15 * (V + 10)) (h2 : C = 12 * (V + 20)) : C / V = 20 := 
by 
  sorry

end supply_without_leak_last_for_20_days_l786_786680


namespace arithmetic_expression_proof_l786_786373

theorem arithmetic_expression_proof : 4 * 6 * 8 + 18 / 3 ^ 2 = 194 := by
  sorry

end arithmetic_expression_proof_l786_786373


namespace initial_marbles_l786_786120

theorem initial_marbles (initial_marbles: ℝ) (given_marbles: ℝ) (total_marbles: ℝ) 
  (h_given: given_marbles = 233.0) 
  (h_total: total_marbles = 1025) :
  initial_marbles = 792 :=
by {
  have h : initial_marbles = total_marbles - given_marbles, {
    sorry  -- This is where the actual proof would go.
  },
  rw [h_given, h_total] at h,
  sorry  -- This second "sorry" is necessary because we're not completing the proof here.
}

end initial_marbles_l786_786120


namespace square_of_leg_l786_786967

theorem square_of_leg (a c b : ℝ) (h1 : c = 2 * a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = 3 * a^2 + 4 * a + 1 :=
by
  sorry

end square_of_leg_l786_786967


namespace largest_prime_divisor_needed_for_primality_in_range_l786_786620

theorem largest_prime_divisor_needed_for_primality_in_range (m : ℕ) (h : 700 ≤ m ∧ m ≤ 750) :
  ∃ p, nat.prime p ∧ p ≤ nat.floor (real.sqrt 750) ∧ ∀ q, nat.prime q ∧ q ≤ nat.floor (real.sqrt 750) → q ≤ p := by
  sorry

end largest_prime_divisor_needed_for_primality_in_range_l786_786620


namespace parabola_focus_distance_l786_786923

open Real

def parabola_y2_neg2x (A : ℝ × ℝ) : Prop :=
  A.snd ^ 2 = -2 * A.fst

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.fst - Q.fst) ^ 2 + (P.snd - Q.snd) ^ 2)

theorem parabola_focus_distance (x₀ y₀ : ℝ) (h₀ : parabola_y2_neg2x (x₀, y₀)) :
  distance (x₀, y₀) (0, -1) = 3 / 2 → x₀ = -1 :=
by
  sorry

end parabola_focus_distance_l786_786923


namespace prob_visiting_C_l786_786672

-- Definitions of the problem
def intersection_A := (0, 0)
def intersection_B := (3, 2)
def intersection_C := (2, 1)
def intersection_D := (5, 3)

-- Define the paths and probabilities
def total_paths_AB := ℕ.choose 5 3 
def total_paths_BD := ℕ.choose 3 2
def total_paths_AD_via_B := total_paths_AB * total_paths_BD

def paths_AB_through_C := ℕ.choose 3 2 * 1
def paths_through_C := paths_AB_through_C * total_paths_BD

-- The probability of passing through C
def prob_through_C := paths_through_C / total_paths_AD_via_B

theorem prob_visiting_C (prob_eq : prob_through_C = 3 / 10) : 
    prob_through_C = 3 / 10 := sorry

end prob_visiting_C_l786_786672


namespace largest_initial_number_l786_786045

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786045


namespace min_diff_between_y_and_x_l786_786961

theorem min_diff_between_y_and_x (x y z : ℤ)
    (h1 : x < y)
    (h2 : y < z)
    (h3 : Even x)
    (h4 : Odd y)
    (h5 : Odd z)
    (h6 : z - x = 9) :
    y - x = 1 := 
  by sorry

end min_diff_between_y_and_x_l786_786961


namespace tanya_addition_problem_l786_786030

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786030


namespace inscribed_circle_ratio_l786_786300

theorem inscribed_circle_ratio :
  (1 : ℝ) / 18 = (let r : ℝ := 1 in
                  let initial_square_side := 2 * r in
                  let second_square_side := initial_square_side in
                  let first_diameter := second_square_side in
                  let larger_circle_radius := second_square_side * real.sqrt 2 / 2 in
                  let scaled_side := 3 * 2 * real.sqrt 2 * r in
                  let final_circle_radius := scaled_side / 2 in
                  r^2 * real.pi / (final_circle_radius^2 * real.pi)) :=
by sorry

end inscribed_circle_ratio_l786_786300


namespace number_of_ways_to_choose_materials_l786_786232

theorem number_of_ways_to_choose_materials :
  let total_materials := 6
  let students_choose := 2
  let one_common_material := 1
  let total_ways := nat.choose total_materials one_common_material * nat.choose (total_materials - 1) (students_choose - one_common_material)
  total_ways = 120 := by
  sorry

end number_of_ways_to_choose_materials_l786_786232


namespace solve_complex_equation_l786_786798

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786798


namespace product_of_points_l786_786004

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def Chris_rolls : List ℕ := [5, 2, 1, 6]
def Dana_rolls : List ℕ := [6, 2, 3, 3]

def Chris_points : ℕ := (Chris_rolls.map f).sum
def Dana_points : ℕ := (Dana_rolls.map f).sum

theorem product_of_points : Chris_points * Dana_points = 297 := by
  sorry

end product_of_points_l786_786004


namespace anne_cleaning_time_l786_786284

theorem anne_cleaning_time (B A : ℝ) 
  (h₁ : 4 * (B + A) = 1) 
  (h₂ : 3 * (B + 2 * A) = 1) : 
  1 / A = 12 :=
sorry

end anne_cleaning_time_l786_786284


namespace arc_segments_greater_one_third_l786_786613

theorem arc_segments_greater_one_third {n : ℕ} (h : n > 1) :
  ∀ (points : Finset ℕ) (hpoints : points.card = n + 2), 
  ∃ (p1 p2 : ℕ) (hp1 : p1 ∈ points) (hp2 : p2 ∈ points) (hneq : p1 ≠ p2),
    let arc1_length := min (p2 - p1 : ℕ) (3 * n - (p2 - p1 : ℕ)),
    let arc2_length := 3 * n - arc1_length
    in arc1_length > n ∧ arc2_length > n := 
sorry

end arc_segments_greater_one_third_l786_786613


namespace reading_proof_l786_786147

noncomputable def reading (arrow_pos : ℝ) : ℝ :=
  if arrow_pos > 9.75 ∧ arrow_pos < 10.0 then 9.95 else 0

theorem reading_proof
  (arrow_pos : ℝ)
  (h0 : 9.75 < arrow_pos)
  (h1 : arrow_pos < 10.0)
  (possible_readings : List ℝ)
  (h2 : possible_readings = [9.80, 9.90, 9.95, 10.0, 9.85]) :
  reading arrow_pos = 9.95 := by
  -- Proof would go here
  sorry

end reading_proof_l786_786147


namespace least_distinct_values_l786_786308

theorem least_distinct_values (lst : List ℕ) (h_len : lst.length = 2023) (h_mode : ∃ m, (∀ n ≠ m, lst.count n < lst.count m) ∧ lst.count m = 13) : ∃ x, x = 169 :=
by
  sorry

end least_distinct_values_l786_786308


namespace num_ways_choose_materials_l786_786206

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786206


namespace monotonicity_extreme_points_inequality_l786_786916

noncomputable def f (x a : ℝ) : ℝ := x - a * x^2 - real.log x

def critical_points_eqn (a : ℝ) := 2 * a * x^2 - x + 1 = 0

theorem monotonicity (a : ℝ) (h : a ≥ 1/8) :
  ∀ x > 0, ∀ y > 0, x < y → f x a ≥ f y a :=
sorry

theorem extreme_points_inequality (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1/8) :
  ∃ x₁ x₂ > 0, critical_points_eqn a x₁ ∧ critical_points_eqn a x₂ ∧ 
  f x₁ a + f x₂ a > 3 - 2 * real.log 2 :=
sorry

end monotonicity_extreme_points_inequality_l786_786916


namespace range_of_theta_l786_786946

theorem range_of_theta (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (h_ineq : 3 * (Real.sin θ ^ 5 + Real.cos (2 * θ) ^ 5) > 5 * (Real.sin θ ^ 3 + Real.cos (2 * θ) ^ 3)) :
    θ ∈ Set.Ico (7 * Real.pi / 6) (11 * Real.pi / 6) :=
sorry

end range_of_theta_l786_786946


namespace LaurynCompanyEmployees_l786_786996

noncomputable def LaurynTotalEmployees (men women total : ℕ) : Prop :=
  men = 80 ∧ women = men + 20 ∧ total = men + women

theorem LaurynCompanyEmployees : ∃ total, ∀ men women, LaurynTotalEmployees men women total → total = 180 :=
by 
  sorry

end LaurynCompanyEmployees_l786_786996


namespace gain_percentage_is_correct_l786_786310

def cost_prices : List ℝ := [18.50, 25.75, 42.60, 29.90, 56.20]
def selling_prices : List ℝ := [22.50, 32.25, 49.60, 36.40, 65.80]

def total_cp : ℝ := cost_prices.sum
def total_sp : ℝ := selling_prices.sum

def total_gain : ℝ := total_sp - total_cp
def gain_percentage : ℝ := (total_gain / total_cp) * 100

theorem gain_percentage_is_correct :
  gain_percentage = 19.35 := 
by
  -- Applying the given conditions
  have h_cp : total_cp = cost_prices.sum := rfl
  have h_sp : total_sp = selling_prices.sum := rfl
  have h_gain : total_gain = total_sp - total_cp := rfl
  have h_gain_percentage : gain_percentage = (total_gain / total_cp) * 100 := rfl
  
  -- Computing the exact values
  have cp_val : total_cp = 173.05 := by norm_num1 -- Assuming norm_num1 exists and works
  have sp_val : total_sp = 206.55 := by norm_num1 -- Assuming norm_num1 exists and works
  have gain_val : total_gain = 33.50 := by norm_num1 -- Assuming norm_num1 exists and works
  have percentage_val : gain_percentage = 19.35 := by -- Applying the fraction and multiplication directly
    calc
      gain_percentage = (33.50 / 173.05) * 100 := by exact h_gain_percentage
      ... = 19.35 := by norm_num1 -- Assuming division and multiplication works as norm_num1

  exact percentage_val

#csorry -- This comment is just to indicate the missing proof steps for direct calculation if required.

end gain_percentage_is_correct_l786_786310


namespace find_eccentricity_find_ellipse_equation_l786_786533

noncomputable def ellipse_equation (a b : ℝ) : (x y : ℝ) → Prop :=
  a > b → b > 0 → (x/a)^2 + (y/b)^2 = 1

noncomputable def point_O : (ℝ × ℝ) := (0, 0)

noncomputable def point_A (a : ℝ) : (ℝ × ℝ) := (a, 0)

noncomputable def point_B (b : ℝ) : (ℝ × ℝ) := (0, b)

noncomputable def point_M (a b : ℝ) : (ℝ × ℝ) := (2*a/3, b/3)

noncomputable def slope_OM : ℝ := real.sqrt 5 / 10

noncomputable def point_C (a : ℝ) : (ℝ × ℝ) := (-a, 0)

noncomputable def point_N (a b : ℝ) : (ℝ × ℝ) := (-real.sqrt 5 * b / 2, b / 2)

noncomputable def symmetric_point_N_ordinate : ℝ := 13 / 2

theorem find_eccentricity (a b : ℝ) (e : ℝ) :
  (a > b ∧ b > 0) →
  e = (real.sqrt 5 / 2) →
  e = (2 * b / (real.sqrt 5 * b)) :=
  sorry

theorem find_ellipse_equation (a b : ℝ) :
  a = 3 * real.sqrt 5 →
  b = 3 →
  ellipse_equation (3 * real.sqrt 5) 3 =
  λ x y, (x^2) / 45 + (y^2) / 9 = 1 :=
  sorry

end find_eccentricity_find_ellipse_equation_l786_786533


namespace largest_initial_number_l786_786037

theorem largest_initial_number :
  ∃ n : ℕ, (n + f n = 100 ∧
  ¬ ∃ k : ℕ, k ∣ n ∧ k ∣ f n) ∧
  ∀ m : ℕ, (m < n → ¬∃ f' : ℕ → ℕ, m + f' m = 100) :=
sorry

end largest_initial_number_l786_786037


namespace two_students_one_common_material_l786_786197

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786197


namespace number_of_descending_digit_numbers_l786_786745

theorem number_of_descending_digit_numbers : 
  (∑ k in Finset.range 8, Nat.choose 10 (k + 2)) + 1 = 1013 :=
by
  sorry

end number_of_descending_digit_numbers_l786_786745


namespace tetrahedron_area_l786_786999

theorem tetrahedron_area {A B C D : Type} 
  (p q r : ℝ)
  (h₁ : ∃ b d, p = 1/2 * b * d ∧ b > 0 ∧ d > 0)
  (h₂ : ∃ c d, r = 1/2 * c * d ∧ c > 0 ∧ d > 0)
  (h₃ : ∃ b c, q = 1/2 * b * c ∧ b > 0 ∧ c > 0) :
  ∃ K : ℝ, K = sqrt (p^2 + q^2 + r^2) :=
begin
  sorry
end

end tetrahedron_area_l786_786999


namespace harmonic_series_inequality_l786_786093

theorem harmonic_series_inequality (n : ℕ) (h : n > 2) : 
  n * (n + 1) ^ (1 / n : ℝ) - n < (∑ k in Finset.range (n + 1), (1 : ℝ) / k.succ) ∧ 
  (∑ k in Finset.range (n + 1), (1 : ℝ) / k.succ) < 
  n - (n - 1) * n ^ (-(1 / (n - 1) : ℝ)) :=
by
  sorry

end harmonic_series_inequality_l786_786093


namespace quadratic_root_value_l786_786402

theorem quadratic_root_value (a b : ℝ) (h : a * 4 + b * 2 = 6) : 4 * a + 2 * b = 6 :=
  by exact h

end quadratic_root_value_l786_786402


namespace total_students_l786_786482

theorem total_students (ratio_boys_girls : ℕ) (girls : ℕ) (boys : ℕ) (total_students : ℕ)
  (h1 : ratio_boys_girls = 2)     -- The simplified ratio of boys to girls
  (h2 : girls = 200)              -- There are 200 girls
  (h3 : boys = ratio_boys_girls * girls) -- Number of boys is ratio * number of girls
  (h4 : total_students = boys + girls)   -- Total number of students is the sum of boys and girls
  : total_students = 600 :=             -- Prove that the total number of students is 600
sorry

end total_students_l786_786482


namespace no_ordered_triples_exist_l786_786941

theorem no_ordered_triples_exist :
  ¬ ∃ (x y z : ℤ), 
    (x^2 - 3 * x * y + 2 * y^2 - z^2 = 39) ∧
    (-x^2 + 6 * y * z + 2 * z^2 = 40) ∧
    (x^2 + x * y + 8 * z^2 = 96) :=
sorry

end no_ordered_triples_exist_l786_786941


namespace six_tangent_circles_l786_786522

-- Define the circle structure and the tangent condition
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

def tangents (C1 C2 : Circle) : Prop :=
  let distance := real.sqrt ((C1.center.fst - C2.center.fst) ^ 2 + (C1.center.snd - C2.center.snd) ^ 2)
  distance = C1.radius + C2.radius ∨ distance = abs (C1.radius - C2.radius)

-- Define specific circles C1 and C2
def C1 : Circle := { center := (0, 0), radius := 1 }
def C2 : Circle := { center := (2, 0), radius := 1 }

-- Tangency conditions for C3 with radius 3
def tangency_conditions (C1 C2 C3 : Circle) : Prop :=
  tangents C1 C3 ∧ tangents C2 C3

-- Define the theorem stating that there are exactly 6 such circles C3
theorem six_tangent_circles :
  ∃ C3s : list Circle, C3s.length = 6 ∧ ∀ C3 ∈ C3s, C3.radius = 3 ∧ tangency_conditions C1 C2 C3 :=
sorry

end six_tangent_circles_l786_786522


namespace area_of_polygon_ABCDFH_l786_786978

theorem area_of_polygon_ABCDFH (
  A B C D E F G H : Type 
  [metric_space B] 
  (ABCD_is_square : square ABCD 5) 
  (EFGD_is_square : square EFGD 3) 
  (H_is_midpoint : is_midpoint_of H (B, C) ∧ is_midpoint_of H (E, F))
) : 
  area_of_polygon ABHFGD = 25.5 :=
sorry

end area_of_polygon_ABCDFH_l786_786978


namespace compute_expression_l786_786529

variable (p q r : ℝ)

-- Given conditions
def roots_of_polynomial : Prop :=
  p + q + r = 15 ∧ pq + qr + rp = 25 ∧ p^3 - 15*p^2 + 25*p - 10 = 0 ∧ q^3 - 15*q^2 + 25*q - 10 = 0 ∧ r^3 - 15*r^2 + 25*r - 10 = 0

theorem compute_expression (h : roots_of_polynomial p q r) : 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 :=
sorry

end compute_expression_l786_786529


namespace derivative_of_function_tangent_line_at_x_eq_1_l786_786920

-- Define the function y = x ln(x)
def y (x : ℝ) : ℝ := x * Real.log x

-- Statement 1: Derivative of the function
theorem derivative_of_function : (x : ℝ) (h : 0 < x) -> deriv y x = Real.log x + 1 :=
by
  sorry

-- Statement 2: Equation of the tangent line at x = 1
theorem tangent_line_at_x_eq_1 : 
  (∀ x : ℝ, y x = x * Real.log x) →
  let slope := (deriv y 1) in
  let tangent_point := (1, 0) in
  slope = 1 ∧ tangent_point.2 = y tangent_point.1 →
  ∀ x : ℝ, y tangent_point.2 = slope * (x - tangent_point.1) :=
by
  sorry

end derivative_of_function_tangent_line_at_x_eq_1_l786_786920


namespace conjugate_of_complex_number_l786_786521

theorem conjugate_of_complex_number : 
  (∀ (x y : ℝ) (i : ℂ), i * (x + y * i) = 5 * i / (2 - i) → conj (x + y * i) = 2 - i) :=
by
  intros x y i h
  sorry

end conjugate_of_complex_number_l786_786521


namespace statement_③_l786_786008

variable {l m : Type} [Line l] [Line m] -- Assuming Line is a type-class for lines
variable {α β : Type} [Plane α] [Plane β] -- Assuming Plane is a type-class for planes

open Plane Line

-- Given conditions
axiom lines_different : l ≠ m
axiom planes_different : α ≠ β

-- Specific condition for the problem
axiom l_in_alpha : l ⊆ α
axiom m_not_perpendicular_to_l : ¬ perpendicular m l

-- Required to prove the statement
theorem statement_③ : ¬ perpendicular m α :=
sorry

end statement_③_l786_786008


namespace tetrahedron_sphere_intersection_l786_786543

-- Definitions of points and tetrahedron
variables {α : Type*} [metric_space α] [normed_add_comm_group α] [normed_space ℝ α]
variables (A B C D K L M N P Q : α)

-- Conditions: Points K, L, M, N, P, Q are on the edges and different from vertices A, B, C, D
def on_edge_and_distinct (p1 p2 p3 : α) (pt : α) := 
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ pt = t • p1 + (1 - t) • p2 ∧ pt ≠ p1 ∧ pt ≠ p2 ∧ pt ≠ p3

axiom cond_K : on_edge_and_distinct A B D K
axiom cond_L : on_edge_and_distinct A C D L
axiom cond_M : on_edge_and_distinct A D B M
axiom cond_N : on_edge_and_distinct B C D N
axiom cond_P : on_edge_and_distinct C D B P
axiom cond_Q : on_edge_and_distinct B D C Q

-- The main goal: Prove that there exists a point F where the four spheres intersect
theorem tetrahedron_sphere_intersection :
  ∃ (F : α),
    (F ∈ (metric.sphere A (dist A K)) ∧ F ∈ (metric.sphere A (dist A L)) ∧ F ∈ (metric.sphere A (dist A M))) ∧
    (F ∈ (metric.sphere B (dist B K)) ∧ F ∈ (metric.sphere B (dist B N)) ∧ F ∈ (metric.sphere B (dist B Q))) ∧
    (F ∈ (metric.sphere C (dist C L)) ∧ F ∈ (metric.sphere C (dist C N)) ∧ F ∈ (metric.sphere C (dist C P))) ∧
    (F ∈ (metric.sphere D (dist D M)) ∧ F ∈ (metric.sphere D (dist D P)) ∧ F ∈ (metric.sphere D (dist D Q))) :=
sorry

end tetrahedron_sphere_intersection_l786_786543


namespace searchlight_parabola_l786_786574

theorem searchlight_parabola :
  ∃ p : ℝ, (p > 0) ∧ (∀ x y : ℝ, (y = 30 ∧ x = 40 → y^2 = 2 * p * x) ∧ (x^2 = -45 / 2 * y)) :=
begin
  sorry
end

end searchlight_parabola_l786_786574


namespace solve_complex_equation_l786_786792

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786792


namespace incidence_bounds_l786_786116

noncomputable def incidence_upper_bound (n : ℕ) : Prop :=
  ∃ c : ℝ, ∀ (n_points n_lines : ℕ), n_points = n ∧ n_lines = n → I(n_points, n_lines) ≤ c * (n : ℝ) ^ (4/3)

noncomputable def incidence_lower_bound (n : ℕ) : Prop :=
  ∃ c' : ℝ, ∀ (n_points n_lines : ℕ), n_points = n ∧ n_lines = n → I(n_points, n_lines) ≥ c' * (n : ℝ) ^ (4/3)

theorem incidence_bounds (n : ℕ) :
  incidence_upper_bound n ∧ incidence_lower_bound n := by
  sorry

end incidence_bounds_l786_786116


namespace star_shell_arrangements_l786_786076

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem star_shell_arrangements : nat :=
  let total_arrangements := factorial 14
  let symmetries := 14
  total_arrangements / symmetries = 6227020800

end star_shell_arrangements_l786_786076


namespace num_ways_choose_materials_l786_786203

theorem num_ways_choose_materials (n m : ℕ) (h₁ : n = 6) (h₂ : m = 2) :
  let ways := ((Nat.choose n 1) * (Nat.perm (n - 1) m))
  in ways = 120 :=
by
  have h₃ : ways = ((Nat.choose 6 1) * (Nat.perm 5 2)), by rw [h₁, h₂]
  rw [h₃, Nat.choose, Nat.perm]
  sorry

end num_ways_choose_materials_l786_786203


namespace distinct_zeros_in_interval_l786_786918

theorem distinct_zeros_in_interval (m : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧ a ∈ set.Icc (-1 : ℝ) (2 : ℝ) ∧ b ∈ set.Icc (-1 : ℝ) (2 : ℝ) ∧ 
    (m * a^2 + (m - 1) * a - 1 = 0) ∧ (m * b^2 + (m - 1) * b - 1 = 0)) ↔ (m = 3 ∨ m = 4) :=
by
  sorry

end distinct_zeros_in_interval_l786_786918


namespace solve_for_x_l786_786560

/-- Prove that x = 8 under the condition that (x - 6)^4 = (1 / 16)^(-1) -/
theorem solve_for_x (x : ℝ) (h : (x - 6)^4 = (1 / 16)^(-1)) : x = 8 := 
by 
  sorry

end solve_for_x_l786_786560


namespace divide_triangle_into_rhombus_l786_786007

noncomputable theory

variables {A B C M : Type} [T : Triangle A B C]

def is_median_longer_than_side (A B C M : Type) : Prop :=
  distance A M > distance A B

def is_acute_triangle (A B C : Type) [T : Triangle A B C] : Prop :=
  ∀ (α β γ : angle), α < π/2 ∧ β < π/2 ∧ γ < π/2

theorem divide_triangle_into_rhombus
  (h1 : is_acute_triangle A B C)
  (h2 : is_median_longer_than_side A B C M) :
  ∃ P Q R, (assemble_to_rhombus P Q R A B C) :=
sorry

end divide_triangle_into_rhombus_l786_786007


namespace proof_prob_5_or_more_calls_proof_mean_interval_between_such_seconds_l786_786727

noncomputable def poisson_prob_5_or_more_calls (λ : ℝ) : ℝ :=
  1 - (Real.exp (-λ) * (1 + 1 + λ / 2 + λ^2 / (2*3) + λ^3 / (2*3*4)))

noncomputable def mean_interval_between_such_seconds (prob : ℝ) : ℝ :=
  1 / prob

theorem proof_prob_5_or_more_calls :
  poisson_prob_5_or_more_calls 1 = 1 - 65 / 24 * Real.exp (-1) :=
by sorry

theorem proof_mean_interval_between_such_seconds :
  mean_interval_between_such_seconds (poisson_prob_5_or_more_calls 1) 
  = 1 / (1 - 65 / 24 * Real.exp (-1)) :=
by sorry

end proof_prob_5_or_more_calls_proof_mean_interval_between_such_seconds_l786_786727


namespace quadratic_function_unique_l786_786414

noncomputable def f (x : ℝ) : ℝ := (1/2)*x^2 + (1/2)*x + 1

theorem quadratic_function_unique
  (f : ℝ → ℝ)
  (hf_quad : ∀ x, ∃ a b c : ℝ, f x = a*x^2 + b*x + c)
  (hf0 : f 0 = 1)
  (hf_shift : ∀ x, f (x + 1) = f x + x + 1) :
  f = (λ x, (1/2)*x^2 + (1/2)*x + 1) :=
by
  sorry

end quadratic_function_unique_l786_786414


namespace find_z_l786_786806

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786806


namespace solve_for_x_l786_786631

theorem solve_for_x (x : ℝ) (h : 3375 = (1 / 4) * x + 144) : x = 12924 :=
by
  sorry

end solve_for_x_l786_786631


namespace largest_initial_number_l786_786049

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786049


namespace man_speed_against_current_with_wind_and_waves_l786_786671

-- Defining the given conditions as constants
def speed_with_current : ℝ := 20
def speed_of_current : ℝ := 5
def wind_effect : ℝ := 2
def waves_effect : ℝ := 1

-- The target statement to prove
theorem man_speed_against_current_with_wind_and_waves :
  let speed_in_still_water := speed_with_current - speed_of_current in
  let speed_against_wind := speed_in_still_water - wind_effect in
  let speed_against_current := speed_against_wind - speed_of_current in
  let final_speed := speed_against_current - waves_effect in
  final_speed = 7 := 
sorry

end man_speed_against_current_with_wind_and_waves_l786_786671


namespace find_g_53_l786_786594

variable (g : ℝ → ℝ)

-- Given conditions
def functional_equation (x y : ℝ) : Prop := g(x * y) = y * g(x)
def given_value: Prop := g(1) = 15

-- Theorem statement
theorem find_g_53 (h1 : functional_equation g) (h2 : given_value g) : g(53) = 795 :=
sorry

end find_g_53_l786_786594


namespace solve_for_z_l786_786822

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786822


namespace weighted_average_correct_l786_786353

-- Define the marks
def english_marks : ℝ := 76
def mathematics_marks : ℝ := 65
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 85

-- Define the weightages
def english_weightage : ℝ := 0.20
def mathematics_weightage : ℝ := 0.25
def physics_weightage : ℝ := 0.25
def chemistry_weightage : ℝ := 0.15
def biology_weightage : ℝ := 0.15

-- Define the weighted sum calculation
def weighted_sum : ℝ :=
  english_marks * english_weightage + 
  mathematics_marks * mathematics_weightage + 
  physics_marks * physics_weightage + 
  chemistry_marks * chemistry_weightage + 
  biology_marks * biology_weightage

-- Define the theorem statement: the weighted average marks
theorem weighted_average_correct : weighted_sum = 74.75 :=
by
  sorry

end weighted_average_correct_l786_786353


namespace solve_complex_equation_l786_786794

-- Definitions to outline the conditions and the theorem
def complex_solution (z : ℂ) : Prop := (1-𝑖)^2 * z = 3 + 2 * 𝑖

theorem solve_complex_equation : complex_solution (-1 + 3 / 2 * 𝑖) :=
by
  sorry

end solve_complex_equation_l786_786794


namespace largest_initial_number_l786_786052

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786052


namespace line_equation_l786_786146

theorem line_equation (t : ℝ) : 
  ∃ m b, (∀ x y : ℝ, (x, y) = (3 * t + 6, 5 * t - 7) → y = m * x + b) ∧
  m = 5 / 3 ∧ b = -17 :=
by
  use 5 / 3, -17
  sorry

end line_equation_l786_786146


namespace solve_z_l786_786851

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786851


namespace sin_squared_right_triangle_l786_786496

variable {A B C : ℝ}

def is_right_triangle (A B C : ℝ) : Prop :=
A^2 + B^2 = C^2 ∨ B^2 + C^2 = A^2 ∨ C^2 + A^2 = B^2

theorem sin_squared_right_triangle (h : sin A ^ 2 = sin B ^ 2 + sin C ^ 2) : is_right_triangle A B C :=
sorry

end sin_squared_right_triangle_l786_786496


namespace solve_for_z_l786_786873

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786873


namespace emily_required_sixth_score_is_99_l786_786729

/-- Emily's quiz scores and the required mean score -/
def emily_scores : List ℝ := [85, 90, 88, 92, 98]
def required_mean_score : ℝ := 92

/-- The function to calculate the required sixth quiz score for Emily -/
def required_sixth_score (scores : List ℝ) (mean : ℝ) : ℝ :=
  let sum_current := scores.sum
  let total_required := mean * (scores.length + 1)
  total_required - sum_current

/-- Emily needs to score 99 on her sixth quiz for an average of 92 -/
theorem emily_required_sixth_score_is_99 : 
  required_sixth_score emily_scores required_mean_score = 99 :=
by
  sorry

end emily_required_sixth_score_is_99_l786_786729


namespace instantaneous_velocity_at_2_l786_786675

def s (t : ℝ) : ℝ := 3 * t^2 + t

theorem instantaneous_velocity_at_2 : (deriv s 2) = 13 :=
by
  sorry

end instantaneous_velocity_at_2_l786_786675


namespace find_other_number_l786_786286

open BigOperators

noncomputable def other_number (n : ℕ) : Prop := n = 12

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 8 n = 24) (h_hcf : Nat.gcd 8 n = 4) : other_number n := 
by
  sorry

end find_other_number_l786_786286


namespace no_3_digit_even_sum_27_l786_786376

/-- Predicate for a 3-digit number -/
def is_3_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate for an even number -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Function to compute the digit sum of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Theorem: There are no 3-digit numbers with a digit sum of 27 that are even -/
theorem no_3_digit_even_sum_27 : 
  ∀ n : ℕ, is_3_digit n → digit_sum n = 27 → is_even n → false :=
by
  sorry

end no_3_digit_even_sum_27_l786_786376


namespace words_count_correct_l786_786944

def number_of_words (n : ℕ) : ℕ :=
if n % 2 = 0 then
  8 * 3^(n / 2 - 1)
else
  14 * 3^((n - 1) / 2)

theorem words_count_correct (n : ℕ) :
  number_of_words n = if n % 2 = 0 then 8 * 3^(n / 2 - 1) else 14 * 3^((n - 1) / 2) :=
by
  sorry

end words_count_correct_l786_786944


namespace number_of_ways_l786_786212

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786212


namespace candy_bar_cost_l786_786354

theorem candy_bar_cost (initial_amount : ℚ) (number_of_candy_bars : ℕ) (final_amount : ℚ)
  (h1 : initial_amount = 4) (h2 : number_of_candy_bars = 99) (h3 : final_amount = 1) :
  (initial_amount - final_amount) / number_of_candy_bars = 1 / 33 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end candy_bar_cost_l786_786354


namespace find_z_l786_786810

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786810


namespace kelly_baking_powder_l786_786274

variable (current_supply : ℝ) (additional_supply : ℝ)

theorem kelly_baking_powder (h1 : current_supply = 0.3)
                            (h2 : additional_supply = 0.1) :
                            current_supply + additional_supply = 0.4 := 
by
  sorry

end kelly_baking_powder_l786_786274


namespace problem_D_l786_786012

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b c : V}

def is_parallel (u v : V) : Prop := ∃ k : ℝ, u = k • v

theorem problem_D (h₁ : is_parallel a b) (h₂ : is_parallel b c) (h₃ : b ≠ 0) : is_parallel a c :=
sorry

end problem_D_l786_786012


namespace total_employees_l786_786480

variable (E : ℕ)
variable (employees_prefer_X employees_prefer_Y number_of_prefers : ℕ)
variable (X_percentage Y_percentage : ℝ)

-- Conditions based on the problem
axiom prefer_X : X_percentage = 0.60
axiom prefer_Y : Y_percentage = 0.40
axiom max_preference_relocation : number_of_prefers = 140

-- Defining the total number of employees who prefer city X or Y and get relocated accordingly:
axiom equation : X_percentage * E + Y_percentage * E = number_of_prefers

-- The theorem we are proving
theorem total_employees : E = 140 :=
by
  -- Proof placeholder
  sorry

end total_employees_l786_786480


namespace largest_initial_number_l786_786023

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786023


namespace no_divisibility_polynomial_l786_786550

theorem no_divisibility_polynomial (A : ℤ) (n m : ℕ) : 
  ¬ ∃ p : ℝ[X], (3 * X^(2*n) + (A:ℝ) * X^n + 2) = (2 * X^(2*m) + (A:ℝ) * X^m + 3) * p := by
  sorry

end no_divisibility_polynomial_l786_786550


namespace balls_in_indistinguishable_boxes_l786_786456

theorem balls_in_indistinguishable_boxes : 
    ∀ (balls boxes : ℕ), balls = 7 ∧ boxes = 3 →
    (∑ d in finset.range (balls + 1), nat.choose balls d) = 64 := by
  intro balls boxes h
  sorry

end balls_in_indistinguishable_boxes_l786_786456


namespace problem_statement_l786_786433

-- Define the function
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

-- The minimum value of difference between axes of symmetry is π/2
axiom min_diff_symmetry_axes (x1 x2 : ℝ) (f : ℝ → ℝ) : |x1 - x2| = Real.pi / 2

-- Define the intervals A and B
def A : Set ℝ := {x | Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2}
def B (m : ℝ) : Set ℝ := {x | |f x - m| < 1}

theorem problem_statement : 
  (∀ k : ℤ, f.interval_of_monotonic_increase_for_k k = (k * Real.pi - Real.pi / 12, k * Real.pi + 5 * Real.pi / 12)) ∧
  (∀ k : ℤ, f.center_of_symmetry_for_k k = (k * Real.pi / 2 + Real.pi / 6, 0)) ∧
  (A ⊆ B m → 1 < m ∧ m < 2) :=
by
  sorry

end problem_statement_l786_786433


namespace solve_exponent_eq_l786_786733

theorem solve_exponent_eq :
  ∃ (x : ℝ), (27^(x - 1)) / (9^(x + 1)) = 81^(-x) ∧ x = 1 :=
by
  have h1 : 27 = 3^3 := by norm_num
  have h2 : 9 = 3^2 := by norm_num
  have h3 : 81 = 3^4 := by norm_num
  use 1
  rw [h1, h2, h3]
  rw [← real.rpow_mul, ← real.rpow_mul, ← real.rpow_mul]
  norm_num
  sorry

end solve_exponent_eq_l786_786733


namespace range_of_a_squared_minus_2b_l786_786524

theorem range_of_a_squared_minus_2b 
  (a b : ℝ) 
  (h1 : f(0) = b ≥ 0) 
  (h2 : f(1) = 1 + a + b ≥ 0) 
  (h3 : a^2 - 4b ≥ 0) 
  (h4 : -2 ≤ a ∧ a ≤ 0) :
  0 ≤ a^2 - 2b ∧ a^2 - 2b ≤ 2 := 
sorry

end range_of_a_squared_minus_2b_l786_786524


namespace largest_initial_number_l786_786026

theorem largest_initial_number :
  ∃ n a1 a2 a3 a4 a5 : ℕ,
  (∀ i ∈ [a1, a2, a3, a4, a5], n + i ∣ n → False) ∧
  n + a1 + a2 + a3 + a4 + a5 = 100 ∧ 
  (∀ m, (∃ b1 b2 b3 b4 b5 : ℕ, 
         (∀ j ∈ [b1, b2, b3, b4, b5], m + j ∣ m → False) ∧
         m + b1 + b2 + b3 + b4 + b5 = 100 
         ) → 
       m ≤ n) :=
begin
  sorry
end

end largest_initial_number_l786_786026


namespace second_intersection_on_AC_l786_786080

open EuclideanGeometry

variables (A B C D E F : Point)
variables (circ1 : Circle)
variables (circ2 : Circle)
variables [trapezoid : Trapezoid A B C D]
variables (H1 : LiesOn E (segment A C))
variables (H2 : LineParallelTo BE (lineThrough C F))
variables (H3 : lineThrough C F = lineThrough B D)
variables (H4 : circumcircle circ1 A B F H1 H2 H3)
variables (H5 : circumcircle circ2 B E D)

theorem second_intersection_on_AC :
  ∃ G : Point, (G ≠ A ∧ G ≠ C) ∧ LiesOn G (segment A C) ∧ LiesOn G (circ1) ∧ LiesOn G (circ2) := sorry

end second_intersection_on_AC_l786_786080


namespace ones_digit_34_pow_34_pow_17_pow_17_l786_786386

-- Definitions from the conditions
def ones_digit (n : ℕ) : ℕ := n % 10

-- Translation of the original problem statement
theorem ones_digit_34_pow_34_pow_17_pow_17 :
  ones_digit (34 ^ (34 * 17 ^ 17)) = 4 :=
sorry

end ones_digit_34_pow_34_pow_17_pow_17_l786_786386


namespace ratio_B_to_C_l786_786555

-- Definitions for conditions
def total_amount : ℕ := 1440
def B_amt : ℕ := 270
def A_amt := (1 / 3) * B_amt
def C_amt := total_amount - A_amt - B_amt

-- Theorem statement
theorem ratio_B_to_C : (B_amt : ℚ) / C_amt = 1 / 4 :=
  by
    sorry

end ratio_B_to_C_l786_786555


namespace find_m_values_tangent_find_m_values_intersect_l786_786415

noncomputable def tangent_condition (m : ℝ) : Prop :=
  abs(3 + 3) / real.sqrt(1^2 + (-m)^2) = 2

noncomputable def intersect_condition (m : ℝ) : Prop :=
  abs(3 + 3) / real.sqrt(1^2 + (-m)^2) = 3 * real.sqrt(10) / 5

theorem find_m_values_tangent (m : ℝ) (x y : ℝ) :
  (x - y * m + 3 = 0) ∧ (x^2 + y^2 - 6 * x + 5 = 0) → 
  tangent_condition m → 
  (m = 2 * real.sqrt 2 ∨ m = -2 * real.sqrt 2) := 
sorry

theorem find_m_values_intersect (m : ℝ) (x y : ℝ) :
  (x - y * m + 3 = 0) ∧ (x^2 + y^2 - 6 * x + 5 = 0) ∧ 
  intersect_condition m → 
  (m = 3 ∨ m = -3) := 
sorry

end find_m_values_tangent_find_m_values_intersect_l786_786415


namespace farthest_vertex_coordinates_l786_786571

noncomputable def image_vertex_coordinates_farthest_from_origin 
    (center_EFGH : ℝ × ℝ) (area_EFGH : ℝ) (dilation_center : ℝ × ℝ) 
    (scale_factor : ℝ) : ℝ × ℝ := sorry

theorem farthest_vertex_coordinates 
    (center_EFGH : ℝ × ℝ := (10, -6)) (area_EFGH : ℝ := 16) 
    (dilation_center : ℝ × ℝ := (2, 2)) (scale_factor : ℝ := 3) : 
    image_vertex_coordinates_farthest_from_origin center_EFGH area_EFGH dilation_center scale_factor = (32, -28) := 
sorry

end farthest_vertex_coordinates_l786_786571


namespace solve_z_l786_786783

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786783


namespace total_lagaan_collected_l786_786517

variable (T : ℝ)
variable (_muttersPayment : ℝ) (percentageOfTotal : ℝ) 

-- Conditions from part a)
def muttersPayment : ℝ := 480
def percentageOfTotal : ℝ := 0.23255813953488372 / 100

-- Lean 4 statement for the proof problem
theorem total_lagaan_collected (h1 : muttersPayment = 480) (h2 : percentageOfTotal = 0.23255813953488372 / 100) :
  T = 206400000 := 
sorry

end total_lagaan_collected_l786_786517


namespace solve_z_l786_786852

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786852


namespace maximum_length_l786_786498

open_locale big_operators

variables {P Q R : Type*}
variables [add_comm_group P] [affine_space P Q]

structure right_triangle (P Q R : P) : Prop :=
(hypotenuse : distance P Q)
(right_angle : ∃ M : P, distance P R = 6 ∧ distance Q M = sqrt (10^2 - 3^2))

noncomputable def max_length_segment (P Q R : P) (hT : right_triangle P Q R) : ℝ :=
sqrt (10^2 - 3^2)

theorem maximum_length 
  {P Q R : P}
  (hT : right_triangle P Q R) :
  max_length_segment P Q R hT = sqrt 91 :=
sorry  -- Placeholder for proof

end maximum_length_l786_786498


namespace workshop_screws_nuts_l786_786485

open Nat Rat Real

theorem workshop_screws_nuts (
  x y : ℕ
  h1 : x + y = 26
  h2 : 1000 * ↑y = 2 * 800 * ↑x
) : true :=
by
  sorry

end workshop_screws_nuts_l786_786485


namespace ratio_of_wire_lengths_l786_786338

/-- 
Bonnie constructs a frame of a rectangular prism using 8 pieces of wire, each 10 inches long.
The prism has dimensions 10 inches, 5 inches, and 2 inches. Roark uses 1-inch-long pieces of wire
to make a collection of unit cube frames. The total volume of Roark's cubes is the same as the
volume of Bonnie's rectangular prism. Prove that the ratio of the total length of Bonnie's wire
to the total length of Roark's wire is 1/15.
-/
theorem ratio_of_wire_lengths :
  let bonnie_wire_length := 8 * 10 in
  let prism_volume := 10 * 5 * 2 in
  let roark_num_cubes := prism_volume in
  let roark_wire_length := roark_num_cubes * 12 in
  bonnie_wire_length / roark_wire_length = 1 / 15 :=
by
  let bonnie_wire_length := 8 * 10
  let prism_volume := 10 * 5 * 2
  let roark_num_cubes := prism_volume
  let roark_wire_length := roark_num_cubes * 12
  show bonnie_wire_length / roark_wire_length = 1 / 15
  sorry

end ratio_of_wire_lengths_l786_786338


namespace two_students_one_common_material_l786_786193

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786193


namespace solve_for_z_l786_786857

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786857


namespace smallest_angle_in_convex_20_gon_seq_l786_786585

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end smallest_angle_in_convex_20_gon_seq_l786_786585


namespace fraction_power_l786_786340

theorem fraction_power (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : (↑a / ↑b)^3 = 27 / 64 :=
by
  rw [h_a, h_b]
  norm_num
  sorry

end fraction_power_l786_786340


namespace minimum_ceiling_height_l786_786319

def is_multiple_of_0_1 (h : ℝ) : Prop := ∃ (k : ℤ), h = k / 10

def football_field_illuminated (h : ℝ) : Prop :=
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 80 →
  (x^2 + y^2 ≤ h^2) ∨ ((x - 100)^2 + y^2 ≤ h^2) ∨
  (x^2 + (y - 80)^2 ≤ h^2) ∨ ((x - 100)^2 + (y - 80)^2 ≤ h^2)

theorem minimum_ceiling_height :
  ∃ (h : ℝ), football_field_illuminated h ∧ is_multiple_of_0_1 h ∧ h = 32.1 :=
sorry

end minimum_ceiling_height_l786_786319


namespace largest_initial_number_l786_786057

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786057


namespace sum_of_consecutive_integers_l786_786160

theorem sum_of_consecutive_integers (n : ℕ) (h : n*(n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end sum_of_consecutive_integers_l786_786160


namespace min_f_abs_l786_786645

def f (x y : ℤ) : ℤ := 5 * x^2 + 11 * x * y - 5 * y^2

theorem min_f_abs (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : (∃ m, ∀ x y : ℤ, (x ≠ 0 ∨ y ≠ 0) → |f x y| ≥ m) ∧ 5 = 5 :=
by
  sorry -- proof goes here

end min_f_abs_l786_786645


namespace largest_initial_number_l786_786065

theorem largest_initial_number :
  ∃ (n : ℕ) (a_1 a_2 a_3 a_4 a_5 : ℕ),
  n + a_1 + a_2 + a_3 + a_4 + a_5 = 100 ∧ 
  (¬ n ∣ a_1) ∧ 
  (¬ (n + a_1) ∣ a_2) ∧ 
  (¬ (n + a_1 + a_2) ∣ a_3) ∧ 
  (¬ (n + a_1 + a_2 + a_3) ∣ a_4) ∧ 
  (¬ (n + a_1 + a_2 + a_3 + a_4) ∣ a_5) ∧ 
  n = 89 :=
begin
  sorry -- proof steps go here
end

end largest_initial_number_l786_786065


namespace no_integer_pairs_satisfying_equation_l786_786472

theorem no_integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), (x - 8) * (x - 10) = 2 ^ y → false :=
by
  intros x y h
  sorry

end no_integer_pairs_satisfying_equation_l786_786472


namespace solve_for_z_l786_786774

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786774


namespace sequence_sum_s10_l786_786161

noncomputable def a : ℕ → ℝ
| 0     := 1  -- Note that Lean indexing starts from 0, so a₁ is a 0
| (n+1) := a n * (1 - a (n+1))

def b (n : ℕ) : ℝ := a n * a (n + 1)

def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i, b i)

theorem sequence_sum_s10 : S 10 = 10 / 11 :=
by
  sorry

end sequence_sum_s10_l786_786161


namespace part1_part2_l786_786425

section
variables (x m : ℝ)
def A := { x | abs (x - 1) < 2 }
def B (m : ℝ) := { x | x^2 - 2 * m * x + m^2 - 1 < 0 }

theorem part1 (hm : m = 3) : A ∩ B m = { x | 2 < x ∧ x < 3 } :=
by sorry

theorem part2 (h : A ∪ B m = A) : 0 ≤ m ∧ m ≤ 2 :=
by sorry
end

end part1_part2_l786_786425


namespace shortest_distance_D_to_V_l786_786149

-- Define distances
def distance_A_to_G : ℕ := 12
def distance_G_to_B : ℕ := 10
def distance_A_to_B : ℕ := 8
def distance_D_to_G : ℕ := 15
def distance_V_to_G : ℕ := 17

-- Prove the shortest distance from Dasha to Vasya
theorem shortest_distance_D_to_V : 
  let dD_to_V := distance_D_to_G + distance_V_to_G
  let dAlt := dD_to_V + distance_A_to_B - distance_A_to_G - distance_G_to_B
  (dAlt < dD_to_V) -> dAlt = 18 :=
by
  sorry

end shortest_distance_D_to_V_l786_786149


namespace find_z_l786_786818

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786818


namespace sqrt_value_of_roots_l786_786427

def quadratic_roots (a b c : ℝ) (m n : ℝ) : Prop :=
  a * (m^2 + n^2 + 2 * b * m * n) + c = 0

theorem sqrt_value_of_roots :
  ∀ m n : ℝ,
  quadratic_roots 1 2 (2 * (2)) m n →
  (m + n = -2 * Real.sqrt 2 ∧ m * n = 1) →
  Real.sqrt (m^2 + n^2 + 3 * (m * n)) = 3 :=
λ m n hroots hmn, by {
  sorry
}

end sqrt_value_of_roots_l786_786427


namespace nail_painting_percentage_difference_l786_786506

theorem nail_painting_percentage_difference :
  let total_nails := 20
  let purple_nails := 6
  let blue_nails := 8
  let striped_nails := total_nails - purple_nails - blue_nails
  let blue_percentage := (blue_nails : ℝ) / (total_nails : ℝ) * 100
  let striped_percentage := (striped_nails : ℝ) / (total_nails : ℝ) * 100
  blue_percentage - striped_percentage = 10 :=
begin
  -- Steps 1 and 2 are calculations of striped_nails, blue_percentage.
  have striped_nails_calc : striped_nails = 6, by
  { simp [total_nails, purple_nails, blue_nails, striped_nails] },

  have blue_percentage_calc : blue_percentage = 40, by
  { norm_num [blue_nails, total_nails, blue_percentage] },

  -- Steps 3 and 4 are calculations of striped_percentage, and the final assertion.
  have striped_percentage_calc : striped_percentage = 30, by
  { norm_num [striped_nails, total_nails, striped_percentage] },

  -- Now we use the calculations to complete the proof.
  simp [blue_percentage_calc, striped_percentage_calc],
  norm_num,
end

end nail_painting_percentage_difference_l786_786506


namespace parallel_slope_l786_786257

theorem parallel_slope {x1 y1 x2 y2 : ℝ} (h : x1 = 3 ∧ y1 = -2 ∧ x2 = 1 ∧ y2 = 5) :
    let slope := (y2 - y1) / (x2 - x1)
    slope = -7 / 2 := 
by 
    sorry

end parallel_slope_l786_786257


namespace students_material_selection_l786_786226

open Finset

theorem students_material_selection {materials : Finset ℕ} (hmat : materials.card = 6) :
  (card {s1 : Finset ℕ // s1 ⊆ materials ∧ s1.card = 2} * card {s2 : Finset ℕ // s2 ⊆ materials ∧ s2.card = 2 ∧ ∃ a, a ∈ s1 ∧ a ∈ s2}) = 120 :=
by sorry

end students_material_selection_l786_786226


namespace two_students_one_common_material_l786_786195

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786195


namespace modulus_of_z_is_4_l786_786426

def complex (re : ℝ) (im : ℝ) := re + im * Complex.i

noncomputable def modulus (z : ℂ) : ℝ := Complex.abs z

noncomputable def given_condition (z : ℂ) : Prop := 
  z / (complex (Real.sqrt 3) (-1)) = complex 1 (Real.sqrt 3)

theorem modulus_of_z_is_4 (z : ℂ) (hz : given_condition z) : modulus z = 4 := 
  sorry

end modulus_of_z_is_4_l786_786426


namespace orthogonal_vectors_x_value_l786_786933

theorem orthogonal_vectors_x_value :
  ∀ (x : ℝ), (let a : ℝ × ℝ := (2, 1) in
               let b : ℝ × ℝ := (x, -1) in
               (a.1 * b.1 + a.2 * b.2 = 0) → x = 1/2) :=
by
  intros x a b h
  rw [←tuple₀_interp] at h
  sorry

end orthogonal_vectors_x_value_l786_786933


namespace wilsons_theorem_l786_786604

theorem wilsons_theorem (p : ℕ) (hp : p > 1) :
  (p.prime ↔ ∃ k, (fact (p - 1) + 1) = k * p) := sorry

end wilsons_theorem_l786_786604


namespace min_diff_between_y_and_x_l786_786962

theorem min_diff_between_y_and_x (x y z : ℤ)
    (h1 : x < y)
    (h2 : y < z)
    (h3 : Even x)
    (h4 : Odd y)
    (h5 : Odd z)
    (h6 : z - x = 9) :
    y - x = 1 := 
  by sorry

end min_diff_between_y_and_x_l786_786962


namespace approximate_reading_l786_786148

-- Define the given conditions
def arrow_location_between (a b : ℝ) : Prop := a < 42.3 ∧ 42.6 < b

-- Statement of the proof problem
theorem approximate_reading (a b : ℝ) (ha : arrow_location_between a b) :
  a = 42.3 :=
sorry

end approximate_reading_l786_786148


namespace sqrt_equiv_1715_l786_786713

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l786_786713


namespace tangents_intersection_on_median_l786_786010

-- Given: BK is the altitude of the acute triangle ABC.
-- Given: A circle is drawn with BK as the diameter.
-- Given: The circle intersects AB and BC at points E and F respectively.
-- Given: Tangents to the circle are drawn through E and F.
-- Prove: The intersection point of these tangents lies on the median of triangle ABC that passes through vertex B.

variables {A B C K E F : Type}
variables [triangle A B C]
variables [altitude BK A B C]
variables [circle_with_diameter BK]
variables [circle_intersect_points BK AB E]
variables [circle_intersect_points BK BC F]
variables [tangent_through_point circle E]
variables [tangent_through_point circle F]

theorem tangents_intersection_on_median :
  ∃ P, intersection_point_tangent EF P ∧ lies_on_median P B A C :=
sorry

end tangents_intersection_on_median_l786_786010


namespace nail_painting_percentage_difference_l786_786507

theorem nail_painting_percentage_difference :
  let total_nails := 20
  let purple_nails := 6
  let blue_nails := 8
  let striped_nails := total_nails - purple_nails - blue_nails
  let blue_percentage := (blue_nails : ℝ) / (total_nails : ℝ) * 100
  let striped_percentage := (striped_nails : ℝ) / (total_nails : ℝ) * 100
  blue_percentage - striped_percentage = 10 :=
begin
  -- Steps 1 and 2 are calculations of striped_nails, blue_percentage.
  have striped_nails_calc : striped_nails = 6, by
  { simp [total_nails, purple_nails, blue_nails, striped_nails] },

  have blue_percentage_calc : blue_percentage = 40, by
  { norm_num [blue_nails, total_nails, blue_percentage] },

  -- Steps 3 and 4 are calculations of striped_percentage, and the final assertion.
  have striped_percentage_calc : striped_percentage = 30, by
  { norm_num [striped_nails, total_nails, striped_percentage] },

  -- Now we use the calculations to complete the proof.
  simp [blue_percentage_calc, striped_percentage_calc],
  norm_num,
end

end nail_painting_percentage_difference_l786_786507


namespace probability_exactly_3_common_l786_786540

open BigOperators

theorem probability_exactly_3_common (S : Finset ℕ) (hS : S.card = 12) :
  let books : Finset (Finset ℕ) := S.powerset.filter (λ s, s.card = 6)
  ∃ p : ℚ, p = 100 / 231 ∧ 
  ∑ H in books, ∑ B in books, if (H ∩ B).card = 3 then 1 else 0 = 
  p * ∑ H in books, ∑ B in books, 1 :=
by 
  sorry

end probability_exactly_3_common_l786_786540


namespace find_y_l786_786134

theorem find_y (y : ℝ) (h : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) : y = 53 / 3 :=
by
  sorry

end find_y_l786_786134


namespace number_of_zeros_l786_786356

def f (x : ℝ) : ℝ := if x ∈ Set.Ioc (-1) 4 then x^2 - 2 * x else 16 - f (x - 5)

theorem number_of_zeros (f : ℝ → ℝ)
    (h1 : ∀ x, f x + f (x + 5) = 16)
    (h2 : ∀ x ∈ Set.Ioc (-1 : ℝ) 4, f x = x^2 - 2 * x) :
  (Set.Icc 0 2013).filter (λ x, f x = 0).card = 202 := sorry

end number_of_zeros_l786_786356


namespace sequence_adhesion_to_all_ones_iff_power_of_two_l786_786079

def adhesion (xs : List Int) : List Int :=
  match xs with
  | [] => []
  | x :: xs' => (x * xs'.headI) :: (adhesion (xs'.tail ++ [x]))

def power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem sequence_adhesion_to_all_ones_iff_power_of_two :
  ∀ (n : ℕ), n ≥ 2 →
    (∃ (xs : List Int), 
      (∀ (x ∈ xs), x = 1 ∨ x = -1) ∧
      (∃ k : ℕ, iterate adhesion k xs = List.replicate n 1))
    ↔ power_of_two n := 
by
  intros
  sorry

end sequence_adhesion_to_all_ones_iff_power_of_two_l786_786079


namespace solve_z_l786_786781

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786781


namespace intersection_of_A_and_B_l786_786647

def A : Set ℝ := { x | x^2 - x - 2 ≥ 0 }
def B : Set ℝ := { x | -2 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -2 ≤ x ∧ x ≤ -1 } := by
-- The proof would go here
sorry

end intersection_of_A_and_B_l786_786647


namespace ratio_of_sphere_to_cube_l786_786754

theorem ratio_of_sphere_to_cube (R a : ℝ) (h : 2 * R = a * sqrt 3) :
  let V_sphere := (4 / 3) * π * R^3
  let V_cube := (2 * R / sqrt 3)^3
  let S_sphere := 4 * π * R^2
  let S_cube := 6 * (2 * R / sqrt 3)^2
  (V_sphere / V_cube = π * sqrt 3 / 2) ∧ (S_sphere / S_cube = π / 2) :=
by {
  let V_sphere := (4 / 3) * π * R^3,
  let V_cube := (2 * R / sqrt 3)^3,
  let S_sphere := 4 * π * R^2,
  let S_cube := 6 * (2 * R / sqrt 3)^2,
  have := calc
    V_cube = ((2 * R / sqrt 3) ^ 3) : by sorry,
    S_cube = (6 * (2 * R / sqrt 3) ^ 2) : by sorry,
  split,
  -- Proof for volume ratio
  have h1 : V_sphere / V_cube = π * sqrt 3 / 2, by sorry,
  exact h1,
  -- Proof for surface area ratio
  have h2 : S_sphere / S_cube = π / 2, by sorry,
  exact h2,
}

end ratio_of_sphere_to_cube_l786_786754


namespace commercials_count_l786_786107

-- Given conditions as definitions
def total_airing_time : ℤ := 90         -- 1.5 hours in minutes
def commercial_time : ℤ := 10           -- each commercial lasts 10 minutes
def show_time : ℤ := 60                 -- TV show (without commercials) lasts 60 minutes

-- Statement: Prove that the number of commercials is 3
theorem commercials_count :
  (total_airing_time - show_time) / commercial_time = 3 :=
sorry

end commercials_count_l786_786107


namespace inscribed_circle_quadrilateral_l786_786301

theorem inscribed_circle_quadrilateral
  (AB CD BC AD AC BD E : ℝ)
  (r1 r2 r3 r4 : ℝ)
  (h1 : BC = AD)
  (h2 : AB + CD = BC + AD)
  (h3 : ∃ E, ∃ AC BD, AC * BD = E∧ AC > 0 ∧ BD > 0)
  (h_r1 : r1 > 0)
  (h_r2 : r2 > 0)
  (h_r3 : r3 > 0)
  (h_r4 : r4 > 0):
  1 / r1 + 1 / r3 = 1 / r2 + 1 / r4 := 
by
  sorry

end inscribed_circle_quadrilateral_l786_786301


namespace red_paint_intensity_l786_786568

variable (I : ℝ) -- Intensity of the original paint
variable (P : ℝ) -- Volume of the original paint
variable (fraction_replaced : ℝ := 1) -- Fraction of original paint replaced
variable (new_intensity : ℝ := 20) -- New paint intensity
variable (replacement_intensity : ℝ := 20) -- Replacement paint intensity

theorem red_paint_intensity : new_intensity = replacement_intensity :=
by
  -- Placeholder for the actual proof
  sorry

end red_paint_intensity_l786_786568


namespace find_m_l786_786930

variables (a b : ℝ × ℝ) (m : ℝ)

def vectors := (a = (3, 4)) ∧ (b = (2, -1))

def perpendicular (a b : ℝ × ℝ) : Prop :=
a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (h1 : vectors a b) (h2 : perpendicular (a.1 + m * b.1, a.2 + m * b.2) (a.1 - b.1, a.2 - b.2)) :
  m = 23 / 3 :=
sorry

end find_m_l786_786930


namespace total_turtles_in_lake_l786_786172

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end total_turtles_in_lake_l786_786172


namespace smallest_four_digit_divisible_by_six_l786_786258

theorem smallest_four_digit_divisible_by_six : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m, m ≥ 1000 ∧ m < n → ¬ (m % 6 = 0) :=
by
  sorry

end smallest_four_digit_divisible_by_six_l786_786258


namespace f_cos_x_l786_786467

noncomputable def f (x : ℝ) : ℝ := 3 - Math.cos (2 * Math.atan x)

theorem f_cos_x (x : ℝ) :
  (∀ x, f (Math.sin x) = 3 - Math.cos (2 * x)) ∧
  (∀ x, Math.cos (2 * x) = Math.cos x * Math.cos x - Math.sin x * Math.sin x) ∧
  (∀ x, Math.cos x * Math.cos x + Math.sin x * Math.sin x = 1) →
  f (Math.cos x) = 3 + Math.cos (2 * x) :=
by
  sorry

end f_cos_x_l786_786467


namespace angle_between_vectors_l786_786165

variables (u v w : ℝ^3)
variables (φ : ℝ)

def norm (x : ℝ^3) : ℝ := real.sqrt (x.x * x.x + x.y * x.y + x.z * x.z)
def dot (x y : ℝ^3) : ℝ := x.x * y.x + x.y * y.y + x.z * y.z
def cross (x y : ℝ^3) : ℝ^3 := ⟨x.y * y.z - x.z * y.y, x.z * y.x - x.x * y.z, x.x * y.y - x.y * y.x⟩

theorem angle_between_vectors
  (h1 : norm u = 2)
  (h2 : norm v = 1)
  (h3 : norm w = 4)
  (h4 : cross u (cross u w) + (2 : ℝ) • v = 0) :
  φ = real.arccos (sqrt 15 / 8) ∨ φ = real.arccos (-sqrt 15 / 8) := sorry

end angle_between_vectors_l786_786165


namespace parakeets_per_cage_l786_786316

theorem parakeets_per_cage (cages parrots_per_cage total_birds : ℕ)
  (h1 : cages = 6)
  (h2 : parrots_per_cage = 6)
  (h3 : total_birds = 48) :
  let total_parrots := cages * parrots_per_cage in
  let total_parakeets := total_birds - total_parrots in
  total_parakeets / cages = 2 :=
by
  sorry

end parakeets_per_cage_l786_786316


namespace ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l786_786113

theorem ab_cd_ge_ac_bd_squared (a b c d : ℝ) : ((a^2 + b^2) * (c^2 + d^2)) ≥ (a * c + b * d)^2 := 
by sorry

theorem eq_condition_ad_eq_bc (a b c d : ℝ) (h : a * d = b * c) : ((a^2 + b^2) * (c^2 + d^2)) = (a * c + b * d)^2 := 
by sorry

end ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l786_786113


namespace total_turtles_in_lake_l786_786173

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end total_turtles_in_lake_l786_786173


namespace ones_digit_34_pow_34_pow_17_pow_17_l786_786387

-- Definitions from the conditions
def ones_digit (n : ℕ) : ℕ := n % 10

-- Translation of the original problem statement
theorem ones_digit_34_pow_34_pow_17_pow_17 :
  ones_digit (34 ^ (34 * 17 ^ 17)) = 4 :=
sorry

end ones_digit_34_pow_34_pow_17_pow_17_l786_786387


namespace sequence_sum_a1_a3_l786_786611

theorem sequence_sum_a1_a3 (S : ℕ → ℕ) (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → S n + S (n - 1) = 2 * n - 1) 
  (h2 : S 2 = 3) : 
  a 1 + a 3 = -1 := by
  sorry

end sequence_sum_a1_a3_l786_786611


namespace sum_of_fraction_of_primes_l786_786514

theorem sum_of_fraction_of_primes (s : Finset ℕ) :
  (∀ n ∈ s, ∀ m < 1, m ≤ 9 → s.digit_sum n = 4) →
  (∀ n ∈ s, nat.prime n) →
  let a := s.filter nat.prime |>.card,
      b := s.card in gcd a b = 1 → a + b = 19 :=
by
  sorry

end sum_of_fraction_of_primes_l786_786514


namespace perp_planes_necessity_perp_planes_insufficiency_perp_planes_conclusion_l786_786409

variable (α β : Plane) (m : Line)
-- assuming these are declared elsewhere
-- α, β are two different planes and m is a line contained in α.
axiom α_neq_β : α ≠ β
axiom m_in_α  : m ∈ α

theorem perp_planes_necessity (h₁ : m ⊥ β) : α ⊥ β :=
sorry

theorem perp_planes_insufficiency (h₂ : α ⊥ β) : ¬(α ⊥ β ↔ m ⊥ β) :=
sorry

theorem perp_planes_conclusion : 
(α ⊥ β → m ⊥ β) ∧ (¬(m ⊥ β → α ⊥ β)) :=
  begin
    split,
    { exact perp_planes_necessity α β m },
    { exact perp_planes_insufficiency α β m },
  end

end perp_planes_necessity_perp_planes_insufficiency_perp_planes_conclusion_l786_786409


namespace number_of_triangles_l786_786605

noncomputable def num_right_angled_triangles : ℕ :=
  {b : ℕ // b < 2011}.filter (λ b, ∃ a : ℕ, a^2 = 2 * b + 1 ∧ odd (a^2) ∧ ∃ n : ℕ, a = 2 * n + 1).card

theorem number_of_triangles : num_right_angled_triangles = 31 := 
  sorry

end number_of_triangles_l786_786605


namespace solve_for_z_l786_786870

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786870


namespace sum_coordinates_of_D_is_10_l786_786548

open Real

def is_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem sum_coordinates_of_D_is_10 :
  ∀ (a b : ℝ),
  is_first_quadrant (a, b) →
  let A : ℝ × ℝ := (2, 8)
  let B : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (6, 4)
  let D : ℝ × ℝ := (a, b)
  let M_AB : ℝ × ℝ := midpoint A B
  let M_BC : ℝ × ℝ := midpoint B C
  let M_CD : ℝ × ℝ := midpoint C D
  let M_DA : ℝ × ℝ := midpoint D A
  (M_AB = (1, 5) ∧ M_BC = (3, 3) ∧
   M_CD = (5, 5) ∧ M_DA = (4, 6)) →
  a + b = 10 :=
by
  intros a b H1 H2
  sorry

end sum_coordinates_of_D_is_10_l786_786548


namespace total_turtles_in_lake_l786_786174

theorem total_turtles_in_lake
  (female_percent : ℝ) (male_with_stripes_fraction : ℝ) 
  (babies_with_stripes : ℝ) (adults_percentage : ℝ) : 
  female_percent = 0.6 → 
  male_with_stripes_fraction = 1/4 →
  babies_with_stripes = 4 →
  adults_percentage = 0.6 →
  ∃ (total_turtles : ℕ), total_turtles = 100 :=
  by
  -- Step-by-step proof to be filled here
  sorry

end total_turtles_in_lake_l786_786174


namespace cost_price_proof_l786_786691

noncomputable def cost_price (C : ℝ) : Prop :=
  let SP := 0.76 * C in
  let ISP := 1.18 * C in
  ISP = SP + 450 ∧ C = 1071.43

theorem cost_price_proof : ∃ C : ℝ, cost_price C :=
by
  use 1071.43
  unfold cost_price
  split
  . exact eq.symm (by norm_num : 1.18 * 1071.43 = 0.76 * 1071.43 + 450)
  . exact eq.refl 1071.43

end cost_price_proof_l786_786691


namespace no_real_solutions_l786_786127

theorem no_real_solutions (y : ℝ) : 
  (8 * y ^ 2 + 47 * y + 5) / (4 * y + 15) = 4 * y + 2 → false :=
begin
  intro h,
  -- We can add proof steps here if desired, but it's not required for the statement.
  sorry
end

end no_real_solutions_l786_786127


namespace train_crossing_time_l786_786641

theorem train_crossing_time
  (train_length : ℕ)
  (bridge_length : ℕ)
  (train_speed_kmph : ℕ)
  (train_speed_mps : ℕ)
  (length_unit_conversion : 1 km = 1000 m)
  (time_unit_conversion : 1 hour = 3600 s)
  (speed_conversion : train_speed_mps = (train_speed_kmph * 1000 / 3600)) :
  train_length = 100 ∧ bridge_length = 180 ∧ train_speed_kmph = 36 ∧ train_speed_mps = 10 →
  (train_length + bridge_length) / train_speed_mps = 28 :=
by
  sorry

end train_crossing_time_l786_786641


namespace sqrt_of_expression_l786_786715

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l786_786715


namespace sum_of_digits_of_smallest_divisible_is_6_l786_786519

noncomputable def smallest_divisible (n : ℕ) : ℕ :=
Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_smallest_divisible_is_6 : sum_of_digits (smallest_divisible 7) = 6 := 
by
  simp [smallest_divisible, sum_of_digits]
  sorry

end sum_of_digits_of_smallest_divisible_is_6_l786_786519


namespace solve_for_z_l786_786859

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l786_786859


namespace second_year_selection_l786_786005

noncomputable def students_from_first_year : ℕ := 30
noncomputable def students_from_second_year : ℕ := 40
noncomputable def selected_from_first_year : ℕ := 6
noncomputable def selected_from_second_year : ℕ := (selected_from_first_year * students_from_second_year) / students_from_first_year

theorem second_year_selection :
  students_from_second_year = 40 ∧ students_from_first_year = 30 ∧ selected_from_first_year = 6 →
  selected_from_second_year = 8 :=
by
  intros h
  sorry

end second_year_selection_l786_786005


namespace number_of_ways_l786_786214

theorem number_of_ways (n : ℕ) (r : ℕ) (A B : ℕ) : 
(n = 6) → (r = 2) → (A = 6) → (B = 20) → (A * B = 120) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3, h4]
  exact Eq.refl 120

end number_of_ways_l786_786214


namespace fraction_power_l786_786339

theorem fraction_power (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : (↑a / ↑b)^3 = 27 / 64 :=
by
  rw [h_a, h_b]
  norm_num
  sorry

end fraction_power_l786_786339


namespace sqrt_product_eq_l786_786709

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end sqrt_product_eq_l786_786709


namespace finish_fourth_task_at_9_40_PM_l786_786330

-- Given conditions
def start_time := Time.mk 15 20 -- 3:20 PM in 24-hour format

def watch_delay := 20 -- in minutes

def time_first_three_tasks := 5 * 60 -- 5 hours in minutes

/-- The fourth task takes as long as the average time of the first three tasks -/
def duration_fourth_task := time_first_three_tasks / 3 -- average time in minutes

/-- Determine the actual finish time accounting for the watch delay and task durations -/
def actual_start_time := start_time - watch_delay

def end_time_third_task := actual_start_time + time_first_three_tasks

def finish_time_fourth_task := end_time_third_task + duration_fourth_task

theorem finish_fourth_task_at_9_40_PM : 
  finish_time_fourth_task = Time.mk 21 40 -- 9:40 PM in 24-hour format
:=
sorry -- The proof is omitted

end finish_fourth_task_at_9_40_PM_l786_786330


namespace defectives_prob_lt_2_l786_786166

theorem defectives_prob_lt_2 :
  let n := 10 in
  let m := 3 in
  let r := 2 in
  let P (k : ℕ) : ℚ := (nat.choose (n - m) (r - k) * nat.choose m k) / nat.choose n r in
  (P 0 + P 1 = 14 / 15) :=
by
  sorry

end defectives_prob_lt_2_l786_786166


namespace number_of_steps_l786_786072

theorem number_of_steps (n : ℕ) :
  (∑ k in Finset.range (n + 1), 3 * k) = 360 ↔ n = 15 :=
by
  sorry

end number_of_steps_l786_786072


namespace factorize_a_cube_minus_nine_a_l786_786372

theorem factorize_a_cube_minus_nine_a (a : ℝ) : a^3 - 9 * a = a * (a + 3) * (a - 3) :=
by sorry

end factorize_a_cube_minus_nine_a_l786_786372


namespace sin_alpha_correct_complicated_expression_correct_l786_786436

open Real

variables {α : ℝ}
noncomputable def sin_alpha : ℝ := 4/5
noncomputable def cos_alpha : ℝ := -3/5

theorem sin_alpha_correct :
  α.vertex_at_origin ∧ α.initial_side_non_negative_x_axis ∧ α.terminal_side_intersects_unit_circle (-3/5) (4/5) →
  sin α = 4/5 :=
sorry

theorem complicated_expression_correct :
  α.vertex_at_origin ∧ α.initial_side_non_negative_x_axis ∧ α.terminal_side_intersects_unit_circle (-3/5) (4/5) →
  (sin (2*α) + cos (2*α) + 1) / (1 + tan α) = 6/5 :=
sorry

end sin_alpha_correct_complicated_expression_correct_l786_786436


namespace match_probability_l786_786309

/-- Given five celebrities and their corresponding baby pictures,
 the probability of randomly guessing all matches correctly is 1/120. -/
theorem match_probability :
  let n := 5 in
  let total_arrangements := nat.factorial n in
  let correct_arrangements := 1 in
  (correct_arrangements : ℚ) / (total_arrangements : ℚ) = 1 / 120 := 
by
  sorry

end match_probability_l786_786309


namespace symmedian_halves_segment_l786_786141

noncomputable def triangle := (A B C : Point)
def altitude (A : Point) (triangle : Type) := Point -- Define A_1, B_1, C_1
def symmedian (B : Point) (triangle : Type) := Point -- Define symmedian point
def projection (B : Point) (line : Type) := Point -- Define K
def midpoint (P Q : Point) := Point

theorem symmedian_halves_segment
  (A B C A1 B1 C1 K : Point)
  (h1 : altitude A triangle)
  (h2 : altitude B triangle)
  (h3 : altitude C triangle)
  (Hproj : K = projection B (line A1 C1))
  (Hsymm : symmedian B triangle)
  (Hmid : midpoint B1 K = midpoint B1 K) :
  midpoint B1 K = B :=
  sorry

end symmedian_halves_segment_l786_786141


namespace possible_values_of_r_l786_786703

-- Conditions given in the problem.
def circleA_radius := 150
def is_tangent (r : ℕ) : Prop := r < 50 ∧ 150 % r = 0

-- The statement to be proved.
theorem possible_values_of_r : {r : ℕ | is_tangent r}.to_finset.card = 9 :=
by
  sorry

end possible_values_of_r_l786_786703


namespace largest_initial_number_l786_786051

theorem largest_initial_number : ∃ (n : ℕ), (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ a : ℕ, ¬ (n + (i - 1) * a = n + (i - 1) * a) ∧ n + (i - 1) * a = 100) ∧ (∀ m, m ≥ n → m = 89) := 
sorry

end largest_initial_number_l786_786051


namespace max_m_existence_l786_786102

open Real

theorem max_m_existence :
  ∃ (f : ℝ → ℝ) (a b c : ℝ) (t : ℝ) (m : ℝ),
  (a ≠ 0) ∧
  f = λ x, a * x ^ 2 + b * x + c ∧
  (∀ x, f (x - 4) = f (2 - x)) ∧
  (∀ x, f x ≥ x) ∧
  (∀ x, x > 0 ∧ x < 2 → f x ≤ (x + 1) / 2 ^ 2) ∧
  (∀ x, x ∈ Icc 1 m → f (x + t) ≤ x) ∧
  m = 9 :=
sorry

end max_m_existence_l786_786102


namespace star_shell_arrangements_l786_786075

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem star_shell_arrangements : nat :=
  let total_arrangements := factorial 14
  let symmetries := 14
  total_arrangements / symmetries = 6227020800

end star_shell_arrangements_l786_786075


namespace solve_for_x_l786_786565

theorem solve_for_x (x : ℝ) : (x - 6)^4 = (1 / 16)⁻¹ → x = 8 :=
by
  intro h
  have h1 : (1 / 16)⁻¹ = 16 := by norm_num
  rw [h1] at h
  have h2 : (x - 6)^4 = 16 := h
  have h3 : (x - 6) = 2 := by
    apply real.rpow_lt_rpow_iff
    norm_num
    exact h2
  linarith

end solve_for_x_l786_786565


namespace small_loads_have_equal_pieces_l786_786993

def initial_clothing : ℕ := 39
def load_one_clothing : ℕ := 19
def remaining_clothing : ℕ := initial_clothing - load_one_clothing
def number_of_small_loads : ℕ := 5

theorem small_loads_have_equal_pieces :
  remaining_clothing / number_of_small_loads = 4 :=
by calc
  remaining_clothing = 20   : rfl
  remaining_clothing / number_of_small_loads = 4 : by norm_num

end small_loads_have_equal_pieces_l786_786993


namespace possible_to_cut_into_4_identical_parts_l786_786502

-- Define the shape and conditions for the grid cuts
variable (Shape : Type) [finite : Fintype Shape]
variable (GridLine : Shape → Shape → Prop)
variable (IdenticalParts : Set (Set Shape))

-- The given condition that GridLine represents permissible cuts between shapes
axiom grid_line_cuts : ∀ (s1 s2 : Shape), GridLine s1 s2 → ∃ parts: Set (Set Shape), parts ∈ IdenticalParts

-- The proof problem statement: it is possible to cut the shape into 4 identical parts along the grid lines
theorem possible_to_cut_into_4_identical_parts :
  ∃ parts : Set (Set Shape), parts ∈ IdenticalParts ∧ Fintype.card parts = 4 :=
by
  sorry

end possible_to_cut_into_4_identical_parts_l786_786502


namespace smallest_positive_period_range_value_of_a_l786_786934

noncomputable def period_range (x : ℝ) : ℝ :=
  let m := (sqrt 3 * sin (2 * x) + 2, cos x)
  let n := (1, 2 * cos x)
  let f := m.1 * n.1 + m.2 * n.2
  if x ∈ Ioo (-π / 6) (π / 2) then f else 0

theorem smallest_positive_period_range :
  (∀ x ∈ Ioo (-π / 6) (π / 2), period_range x ∈ Ioc (2 : ℝ) 5) ∧ θ = π :=
sorry

noncomputable def find_a (A : ℝ) :=
  let f := (3 + 2 * sin (2 * A + π / 6))
  let b := 4
  let area := sqrt 3
  let c := 1
  let a_sq := b^2 + c^2 - 2 * b * c * cos A
  sqrt a_sq

theorem value_of_a
  (A : ℝ)
  (f_A : period_range A = 4)
  (b : ℝ := 4)
  (area : ℝ := sqrt 3)
  : find_a A = sqrt 13 :=
sorry

end smallest_positive_period_range_value_of_a_l786_786934


namespace max_value_of_f_l786_786600

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - Real.log2 (x + 2)

theorem max_value_of_f : ∃ y ∈ Icc (-1:ℝ) (1:ℝ), ∀ x ∈ Icc (-1:ℝ) (1:ℝ), f x ≤ f y ∧ f y = 3 :=
by
  use -1
  split
  · exact by norm_num
  · split
    · intros x hx
      sorry
    · norm_num

end max_value_of_f_l786_786600


namespace solve_for_z_l786_786874

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786874


namespace find_z_l786_786842

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786842


namespace perpendicular_line_l786_786589

theorem perpendicular_line (p : ℝ) : 
    ∃ a b c : ℝ, (a, b, c) = (2, -1, -1) ∧
        (∃ x y : ℝ, (x, y) = (2, 3) ∧
        a * x + b * y + c = 0) ∧
        ∃ x' y' : ℝ, (x' + 2 * y' + p = 0) :=
begin
    sorry
end

end perpendicular_line_l786_786589


namespace Mr_Szürke_house_number_l786_786945

-- Definitions for conditions
noncomputable def house_surnames (n : ℕ) : Prop :=
  (odd n ↔ ∃ (s : String), s ∈ ["Zöld", "Fehér", "Fekete", "Barna", "Kalapos", "Szürke", "Sárga"]) ∧
  (even n ↔ ∃ (s : String), s ∈ ["Szabó", "Fazekas", "Kovács", "Lakatos", "Kádárné", "Pék", "Bordóné"])

noncomputable def opposite_houses (a b : ℕ) : Prop := abs a - b = 1

noncomputable def surname_positions : Prop :=
    (opposite_houses 1 2 ∧ opposite_houses 3 4 ∧ opposite_houses 5 6 ∧ opposite_houses 7 8 ∧ 
    opposite_houses 9 10 ∧ opposite_houses 11 12 ∧ opposite_houses 13 14) ∧
    (∃ n, n = 5 ∧ ∃ m, (opposite_houses (m+7) 5)) ∧
    (∃ n, (opposite_houses ((6-1)+ (4+1)) n) ∧ ∃ m, opposite_houses (m+1) 5 ∧ (m = 6)) ∧
    (opposite_houses 11 12) ∧ 
    (∃ n, ) 

-- Main theorem
theorem Mr_Szürke_house_number : 
    ∃ (n : ℕ), n = 13 ∧ house_surnames n ∧ surname_positions :=
by sorry


end Mr_Szürke_house_number_l786_786945


namespace relationship_among_abc_l786_786085

noncomputable def a : ℝ := 7 ^ 0.6
noncomputable def b : ℝ := 0.7 ^ 6
noncomputable def c : ℝ := Real.log 0.6 / Real.log 7

theorem relationship_among_abc : a > b ∧ b > c := by
  sorry

end relationship_among_abc_l786_786085


namespace min_value_dot_product_eq_4sqrt3_l786_786495

open Real

-- Define the given conditions in terms of Lean structures and notions
variables {A B C P : Point} (ABC_area : area A B C = 4)
  (E F : Point)
  (E_midpoint : E = midpoint A B)
  (F_midpoint : F = midpoint A C)
  (P_on_EF : P ∈ line_through E F)

-- Define the vectors and dot products
noncomputable def vector_PC : ℝ := sorry
noncomputable def vector_PB : ℝ := sorry
noncomputable def vector_BC : ℝ := sorry

-- State the theorem to prove the minimum value of the expression
theorem min_value_dot_product_eq_4sqrt3 :
  vector_PC ⋅ vector_PB + vector_BC ^ 2 ≥ 4 * sqrt 3 := sorry

end min_value_dot_product_eq_4sqrt3_l786_786495


namespace max_value_f_on_interval_l786_786917

noncomputable def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

theorem max_value_f_on_interval : 
  is_max_on f (Set.Icc (-2 : ℝ) (1 / 2)) 2 :=
sorry

end max_value_f_on_interval_l786_786917


namespace cost_price_proof_l786_786692

noncomputable def cost_price (C : ℝ) : Prop :=
  let SP := 0.76 * C in
  let ISP := 1.18 * C in
  ISP = SP + 450 ∧ C = 1071.43

theorem cost_price_proof : ∃ C : ℝ, cost_price C :=
by
  use 1071.43
  unfold cost_price
  split
  . exact eq.symm (by norm_num : 1.18 * 1071.43 = 0.76 * 1071.43 + 450)
  . exact eq.refl 1071.43

end cost_price_proof_l786_786692


namespace sequence_properties_l786_786399

variable (x : ℕ → ℝ) (x0 : ℝ) (Sn : ℕ → ℝ)

-- The given condition on the sequence
axiom seq_rel : ∀ n, 2 * x (n) = (∑ i in Finset.range n, x i) - x n

-- Define the expressions for x_n and S_n according to the problem's answers
def x_n (n : ℕ) : ℝ := if n = 0 then x0 else x0 * (4^(n-1)) / (3^n)
def S_n (n : ℕ) : ℝ := x0 * (4 / 3)^n

-- Proof that the expressions derived for x_n and S_n correspond to the sequences
theorem sequence_properties : 
  (∀ n, (x n = x_n x0 n)) ∧ 
  (∀ n, (Sn n = S_n x0 n)) := 
by 
  sorry

end sequence_properties_l786_786399


namespace predict_sales_correct_l786_786128

variable (x y : ℝ)

theorem predict_sales_correct (h1 : (-2, 20) ∈ {(-2, 20), (-3, 23), (-5, 27), (-6, 30)})
  (h2 : b = -2.4) (h3 : a = 15.2) :
  y = b * (-8) + a →
  y = 34.4 :=
by
  intro h
  rw [h2, h3] at h
  linarith

end predict_sales_correct_l786_786128


namespace number_of_valid_paintings_l786_786653

def grid (side_len : ℕ) : Type :=
  { squares : ℕ | squares = side_len * side_len }

def is_valid_painting (painting : grid 3 → bool) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 3 ∧ 0 ≤ j ∧ j < 3 →
    (painting (i, j) = painting (i-1, j) → i = 0) ∧
    (painting (i, j) = painting (i+1, j) → i = 2) ∧
    (painting (i, j) = painting (i, j-1) → j = 0) ∧
    (painting (i, j) = painting (i, j+1) → j = 2)

theorem number_of_valid_paintings : 
  ∃ n : ℕ, n = 10 ∧
  ∀ painting : grid 3 → bool, is_valid_painting painting → grid 3 →
    true :=
sorry

end number_of_valid_paintings_l786_786653


namespace solve_for_z_l786_786868

theorem solve_for_z : (1 - complex.i)^2 * z = 3 + 2 * complex.i → z = -1 + 3 / 2 * complex.i :=
by
  sorry

end solve_for_z_l786_786868


namespace number_of_ways_to_choose_reading_materials_l786_786238

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786238


namespace find_z_l786_786801

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786801


namespace two_students_choose_materials_l786_786190

noncomputable def number_of_ways_two_students_choose_materials (total_materials: ℕ) (materials_per_student: ℕ) (common_materials: ℕ): ℕ :=
  nat.choose total_materials common_materials * nat.choose (total_materials - common_materials) (materials_per_student - common_materials)

theorem two_students_choose_materials :
  number_of_ways_two_students_choose_materials 6 2 1 = 60 := 
by 
  -- This will include the necessary combinatorial calculations
  calc
    number_of_ways_two_students_choose_materials 6 2 1 
        = nat.choose 6 1 * nat.choose (6 - 1) (2 - 1) : rfl
    ... = 6 * nat.choose 5 1 : rfl
    ... = 6 * 5 : by rw nat.choose_succ_self
    ... = 30 : by norm_num
    ... = 60 / 2 : by norm_num
    ... = 60 : by norm_num

-- Note: This Lean proof step included the transformation steps for clarity.

end two_students_choose_materials_l786_786190


namespace distance_points_lt_2_over_3_r_l786_786980

theorem distance_points_lt_2_over_3_r (r : ℝ) (h_pos_r : 0 < r) (points : Fin 17 → ℝ × ℝ)
  (h_points_in_circle : ∀ i, (points i).1 ^ 2 + (points i).2 ^ 2 < r ^ 2) :
  ∃ i j : Fin 17, i ≠ j ∧ (dist (points i) (points j) < 2 * r / 3) :=
by
  sorry

end distance_points_lt_2_over_3_r_l786_786980


namespace find_z_l786_786832

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786832


namespace minimum_value_f_l786_786741

def f (x y: ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y

theorem minimum_value_f : ∃ x y: ℝ, (∀ x' y': ℝ, f x y ≤ f x' y') ∧ f x y = 7 :=
by
  use [-2, 3]
  split
  -- proof of the minimum condition
  -- sorry indicates this part can be proven later
  · sorry 
  -- proof of f -2 3 = 7
  · sorry 

end minimum_value_f_l786_786741


namespace largest_initial_number_l786_786053

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786053


namespace paths_length_difference_is_zero_l786_786490

-- Definitions of the problem conditions
def AC := 10 -- length of AC in meters
def CB := 10 -- length of CB in meters
def AB := AC + CB -- length of AB as sum of AC and CB
def radius_large_semi_circle := AB / 2
def radius_small_semi_circle := AC / 2

-- Lengths of arcs
def length_path1 := 1/2 * 2 * Real.pi * radius_large_semi_circle
def length_path2 := 2 * (1/2 * 2 * Real.pi * radius_small_semi_circle)

-- Problem statement in Lean
theorem paths_length_difference_is_zero :
  length_path1 - length_path2 = 0 := 
sorry

end paths_length_difference_is_zero_l786_786490


namespace ball_distribution_l786_786459

theorem ball_distribution :
  let balls := 7
  let boxes := 3
  let total_ways := 1 + 7 + 21 + 21 + 35 + 105 + 70 + 105
  balls = 7 ∧ boxes = 3 → total_ways = 365 :=
by
  intro h
  simp
  exact rfl

end ball_distribution_l786_786459


namespace time_to_eliminate_mice_l786_786670

def total_work : ℝ := 1
def work_done_by_2_cats_in_5_days : ℝ := 0.5
def initial_2_cats : ℕ := 2
def additional_cats : ℕ := 3
def total_initial_days : ℝ := 5
def total_cats : ℕ := initial_2_cats + additional_cats

theorem time_to_eliminate_mice (h : total_initial_days * (work_done_by_2_cats_in_5_days / total_initial_days) = work_done_by_2_cats_in_5_days) : 
  total_initial_days + (total_work - work_done_by_2_cats_in_5_days) / (total_cats * (work_done_by_2_cats_in_5_days / total_initial_days / initial_2_cats)) = 7 := 
by
  sorry

end time_to_eliminate_mice_l786_786670


namespace two_students_one_common_material_l786_786201

theorem two_students_one_common_material : 
  let total_materials := 6
  let common_material_ways := Nat.choose 6 1   -- Choosing 1 common material out of 6
  let remaining_materials := 5
  let remaining_choices_ways := Nat.perm 5 2   -- Choosing 2 remaining materials out of 5 (distinguished)
  common_material_ways * remaining_choices_ways = 120 := 
by
  simp [common_material_ways, remaining_choices_ways]
  sorry

end two_students_one_common_material_l786_786201


namespace compute_expression_l786_786528

variable (p q r : ℝ)

-- Given conditions
def roots_of_polynomial : Prop :=
  p + q + r = 15 ∧ pq + qr + rp = 25 ∧ p^3 - 15*p^2 + 25*p - 10 = 0 ∧ q^3 - 15*q^2 + 25*q - 10 = 0 ∧ r^3 - 15*r^2 + 25*r - 10 = 0

theorem compute_expression (h : roots_of_polynomial p q r) : 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 :=
sorry

end compute_expression_l786_786528


namespace magnitude_sum_squared_l786_786897

variables {a b : EuclideanSpace ℝ (Fin 3)}

noncomputable def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop :=
 ∥v∥ = 1

theorem magnitude_sum_squared (ha : is_unit_vector a) (hb : is_unit_vector b) (angle_ab : real.angle a b = real.pi * (2/3)) : 
  ∥a + 2 • b∥ = real.sqrt 3 :=
sorry

end magnitude_sum_squared_l786_786897


namespace rods_in_one_mile_l786_786898

-- Define the given conditions
def mile_to_chains : ℕ := 10
def chain_to_rods : ℕ := 4

-- Prove the number of rods in one mile
theorem rods_in_one_mile : (1 * mile_to_chains * chain_to_rods) = 40 := by
  sorry

end rods_in_one_mile_l786_786898


namespace germination_rate_sunflower_l786_786451

variable (s_d s_s f_d f_s p : ℕ) (g_d g_f : ℚ)

-- Define the conditions
def conditions :=
  s_d = 25 ∧ s_s = 25 ∧ g_d = 0.60 ∧ g_f = 0.80 ∧ p = 28 ∧ f_d = 12 ∧ f_s = 16

-- Define the statement to be proved
theorem germination_rate_sunflower (h : conditions s_d s_s f_d f_s p g_d g_f) : 
  (f_s / (g_f * (s_s : ℚ))) > 0.0 ∧ (f_s / (g_f * (s_s : ℚ)) * 100) = 80 := 
by
  sorry

end germination_rate_sunflower_l786_786451


namespace solution_set_of_f_g_l786_786087

variable {R : Type _} [LinearOrderedField R]

variable (f g : R → R)

-- Conditions from a)
def odd_function (h : R → R) := ∀ x, h (-x) = - h (x)
def increasing_on (h : R → R) (s : Set R) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → h x < h y
def decreasing_on (h : R → R) (s : Set R) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → h x > h y

-- Main statement proving the solution set
theorem solution_set_of_f_g (h : ∀ x, f (-x) = -f x ∧ g (-x) = -g x)
    (cond : ∀ x, x < 0 → f'' x * g x + f x * g'' x > 0)
    (f_neg_2_eq_zero : f (-2) = 0) :
    { x | f x * g x < 0 } = { x | x < -2 } ∪ { x | x > 2 } := sorry

end solution_set_of_f_g_l786_786087


namespace find_girls_last_names_l786_786364

-- Define the variables and conditions
variables (a b c d : ℕ)
variable (h1 : a + b + c + d = 10)
variable (h2 : 2 * a + 3 * b + 4 * c + 5 * d = 32)

-- Set the expected result
def correct_last_names : list string := ["Ivanov", "Grishin", "Andreev", "Sergeev"]

-- Prove the names match given the peaches distribution.
theorem find_girls_last_names (h : a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 4) :
  correct_last_names = ["Ivanov", "Grishin", "Andreev", "Sergeev"] :=
by {
  sorry
}

end find_girls_last_names_l786_786364


namespace max_digits_sum_l786_786465

theorem max_digits_sum (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) 
  (h4 : ∃ y ∈ {1, 2, 4, 5, 8, 10}, 0.abc = 1 / y) :
  a + b + c ≤ 8 :=
sorry

end max_digits_sum_l786_786465


namespace factorial_div_l786_786367

theorem factorial_div (n : ℕ) (h : n = 4) : ((n!)!) / n! = (n! - 1)! :=
by
  rw h
  sorry

end factorial_div_l786_786367


namespace solve_z_l786_786844

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786844


namespace polynomial_bound_l786_786092

variable {R : Type*} [LinearOrderedField R]

theorem polynomial_bound {P : R[X]} {n : ℕ}
  (degree_bound : P.degree ≤ 2 * n)
  (value_bound : ∀ k : ℤ, k ∈ Icc (-n) n → |P.eval k| ≤ 1) :
  ∀ x : R, x ∈ Icc (- (n : R)) n → |P.eval x| ≤ 2^(2 * n) :=
begin
  sorry
end

end polynomial_bound_l786_786092


namespace smallest_m_units_digit_of_smallest_m_l786_786991

-- Define the perfect square subtraction sequence
def seq (n : ℕ) := List.unfoldr (λ x => if x = 0 then none else some (x, x - Nat.sqrt x * Nat.sqrt x)) n

-- Define the condition for a sequence to have exactly 7 steps and end at 0
def valid_seq (m : ℕ) : Prop :=
  let s := seq m
  s.length = 7 ∧ s.getLast! = 0

theorem smallest_m : ∃ m, valid_seq m ∧ m = 21 ∧ m % 10 = 1 :=
  sorry

-- Alternatively, if stated in terms of the units digit explicitly
theorem units_digit_of_smallest_m : (∃ m, valid_seq m ∧ m = 21) → (21 % 10 = 1) :=
  sorry

end smallest_m_units_digit_of_smallest_m_l786_786991


namespace square_vector_addition_triangle_isosceles_or_right_l786_786083

-- Statement 1: Square and vector calculations
theorem square_vector_addition (a b c : ℝ) (h_a: a = 1) (h_b: b = 1) :
  |⟨a, 0⟩ + ⟨0, b⟩ + ⟨a, b⟩| = 2 * real.sqrt 2 :=
by sorry

-- Statement 2: Triangle and cosine condition
theorem triangle_isosceles_or_right (a b c A B C: ℝ)
  (h_cos_eq : a * real.cos A = b * real.cos B) :
  (A = B) ∨ (A + B = real.pi / 2) :=
by sorry

end square_vector_addition_triangle_isosceles_or_right_l786_786083


namespace number_of_ways_to_choose_reading_materials_l786_786242

theorem number_of_ways_to_choose_reading_materials 
  (students : Type) (materials : Finset ℕ) (h_stu : students = 2) (h_mat : materials.card = 6) 
  (common_material : ℕ) (h_common : common_material ∈ materials) :
  ∃ ways : ℕ, ways = 120 :=
by  sorry

end number_of_ways_to_choose_reading_materials_l786_786242


namespace find_y_l786_786136

theorem find_y (y : Real) : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9 → y = 53 / 3 :=
by
  sorry

end find_y_l786_786136


namespace total_short_trees_after_planting_l786_786167

def current_short_oak_trees := 3
def current_short_pine_trees := 4
def current_short_maple_trees := 5
def new_short_oak_trees := 9
def new_short_pine_trees := 6
def new_short_maple_trees := 4

theorem total_short_trees_after_planting :
  current_short_oak_trees + current_short_pine_trees + current_short_maple_trees +
  new_short_oak_trees + new_short_pine_trees + new_short_maple_trees = 31 := by
  sorry

end total_short_trees_after_planting_l786_786167


namespace simplify_144_over_1296_times_36_l786_786122

theorem simplify_144_over_1296_times_36 :
  (144 / 1296) * 36 = 4 :=
by
  sorry

end simplify_144_over_1296_times_36_l786_786122


namespace find_z_l786_786836

-- Defining the given condition
def cond : Prop := (1 - complex.i) ^ 2 * z = 3 + 2 * complex.i

-- Statement that proves z under the given condition
theorem find_z (z : ℂ) (h : cond) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786836


namespace tanya_addition_problem_l786_786032

noncomputable def largest_initial_number : ℕ :=
  let a (n : ℕ) (s : Fin 5 → ℕ) : Fin 5 → ℕ := λ i =>
    let m := n + (List.sum (List.ofFn (λ j : Fin i => s j)))
    classical.some (Nat.exists_lt_not_dvd m)
  in
  classical.some (Nat.exists_number_of_five_adds_to 100)

theorem tanya_addition_problem :
  ∃ n : ℕ, largest_initial_number = 89 :=
begin
  sorry -- we skip the actual proof
end

end tanya_addition_problem_l786_786032


namespace problem_part1_problem_part2_l786_786019

theorem problem_part1 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  2 * ((1 / x) + (1 / y) + (1 / z)) ≤ (1 / p) + (1 / q) + (1 / r) :=
sorry

theorem problem_part2 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  x * y + y * z + z * x ≥ 2 * (p * x + q * y + r * z) :=
sorry

end problem_part1_problem_part2_l786_786019


namespace michelles_savings_l786_786688

theorem michelles_savings (n : ℕ) (h : n = 8) : n * 100 = 800 :=
by
  rw [h]
  norm_num

end michelles_savings_l786_786688


namespace Mario_expected_doors_l786_786487

-- Define the parameters r and d
variables (r d : ℕ)

-- Define the expected number of doors E function, assuming r and d are given
noncomputable def expected_doors : ℕ → ℕ → ℕ :=
  λ r d, d * (d^r - 1) / (d - 1)

-- Prove that the expected number of doors Mario passes through is correct
theorem Mario_expected_doors (r d : ℕ) (h1 : d > 1) :
  expected_doors r d = (d * ((d^r) - 1)) / (d - 1) :=
by
  unfold expected_doors
  sorry

end Mario_expected_doors_l786_786487


namespace smallest_angle_in_icosagon_l786_786581

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end smallest_angle_in_icosagon_l786_786581


namespace apple_price_l786_786277

theorem apple_price (A W P : ℝ) 
  (price_relation_1 : 6 * A = 2 * W)
  (price_relation_2 : 3 * P = 2 * W)
  (pineapple_relation : P = 2 * A)
  (orange_price : 24 * 0.75)
  (total_bill : 24 * 0.75 + 18 * A + 12 * W + 18 * P = 165) : 
  A = 49 / 30 :=
by
  let oranges_cost := 24 * 0.75
  have W_value : W = 3 * A, by sorry
  have total_bill_eq : oranges_cost + 18 * A + 12 * (3 * A) + 18 * (2 * A) = 165, by
    rw [W_value, pineapple_relation, total_bill]
    sorry
  have : 165 = 24 * 0.75 + 90 * A, by
    calc
      165 = 165 : by sorry
      ... = oranges_cost + 18 * A + 36 * A + 36 * A : by sorry
  have simplify_fraction : 147 / 90 = 49 / 30, by sorry
  exact (eq_div_iff (ne_of_gt (show 90 > 0 by norm_num))).mp $
    eq.symm $ (add_eq_cancel_left oranges_cost).mp $
    eq.trans total_bill_eq $
    eq.symm (by ring_nf)
  sorry

end

end apple_price_l786_786277


namespace solve_for_x_l786_786125

theorem solve_for_x (x : ℝ) (d : ℝ) (h1 : x > 0) (h2 : x^2 = 4 + d) (h3 : 25 = x^2 + d) : x = Real.sqrt 14.5 := 
by 
  sorry

end solve_for_x_l786_786125


namespace largest_initial_number_l786_786054

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l786_786054


namespace solve_for_z_l786_786772

theorem solve_for_z (z : ℂ) (h : ((1 - complex.i)^2) * z = 3 + 2 * complex.i) : 
  z = -1 + (3 / 2) * complex.i :=
sorry

end solve_for_z_l786_786772


namespace arthur_money_left_l786_786697

theorem arthur_money_left {initial_amount spent_fraction : ℝ} (h_initial : initial_amount = 200) (h_fraction : spent_fraction = 4 / 5) : 
  (initial_amount - spent_fraction * initial_amount = 40) :=
by
  sorry

end arthur_money_left_l786_786697


namespace solve_geom_seq_and_sum_l786_786885

def geom_seq (a : ℕ → ℝ) : Prop := ∀ n, a n > 0 ∧ ∃ q, a (n+1) = q * a n
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = ∑ i in finset.range (n+1), a i
def arithmetic_seq (S1 S2 S3 : ℝ) : Prop := 2 * S3 = 2 * S1 + 4 * S2
def a_given_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  geom_seq a ∧
  (a 1) * (a 2) * (a 3) = 64 ∧
  sum_first_n_terms a S ∧
  arithmetic_seq (2 * S 0) (4 * S 1) (S 2)

theorem solve_geom_seq_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  a_given_conditions a S →
  (∀ n, a n = 2 ^ n) ∧ (∀ n, T n = 2 * n / (n + 1)) :=
by
  intros
  sorry

end solve_geom_seq_and_sum_l786_786885


namespace shift_sin2x_to_sin2x_pi_over_3_l786_786619

theorem shift_sin2x_to_sin2x_pi_over_3 :
  ∀ (x : ℝ), sin (2 * (x - π / 6)) = sin (2 * x - π / 3) :=
by
  intro x
  sorry

end shift_sin2x_to_sin2x_pi_over_3_l786_786619


namespace oblique_projection_parallelogram_l786_786624

theorem oblique_projection_parallelogram (figure : Type) (is_parallelogram : figure → Prop) : 
  is_parallelogram (oblique_projection figure) :=
sorry

end oblique_projection_parallelogram_l786_786624


namespace value_of_m_l786_786406

theorem value_of_m : ∃ (m : ℕ), (3 * 4 * 5 * m = Nat.factorial 8) ∧ m = 672 := by
  sorry

end value_of_m_l786_786406


namespace complex_midpoint_l786_786977

theorem complex_midpoint {i : ℂ} (h_i : i = complex.I) :
  (let A := (1 : ℂ) / (1 + i);
       B := (1 : ℂ) / (1 - i) in
   (A + B) / 2 = 1 / 2) :=
by {
  -- Instead of providing the proof, we place a sorry since proof isn't required here
  sorry
}

end complex_midpoint_l786_786977


namespace max_white_cubes_l786_786252

theorem max_white_cubes (cubes : set (fin 3 × fin 3 × fin 3)) (painted_gray_faces : set (fin 3 × fin 3 × fin 3)) :
  (∀ c ∈ cubes, (∃ f ∈ painted_gray_faces, f ≠ c) → false) ∧ (∀ c ∈ cubes, (c ∉ painted_gray_faces) → c ∈ cubes) :=
begin
  -- Define conditions
  have h1 : cubes = set.univ,
  -- Set of all small $1 \times 1 \times 1$ cubes in a $3 \times 3 \times 3$ arrangement.
  have h2 : ∀ c ∈ cubes, c ∉ painted_gray_faces
    → (∃ w ∈ cubes, w ∉ painted_gray_faces) × (painted_gray_faces ⊆ cubes),
    -- Establishing maximum all-white criteria correctly.
  exact sorry
end

end max_white_cubes_l786_786252


namespace min_triangle_perimeter_proof_l786_786101

noncomputable def min_triangle_perimeter (l m n : ℕ) : ℕ :=
  if l > m ∧ m > n ∧ (3^l % 10000 = 3^m % 10000) ∧ (3^m % 10000 = 3^n % 10000) then
    l + m + n
  else
    0

theorem min_triangle_perimeter_proof : ∃ (l m n : ℕ), l > m ∧ m > n ∧ 
  (3^l % 10000 = 3^m % 10000) ∧
  (3^m % 10000 = 3^n % 10000) ∧ min_triangle_perimeter l m n = 3003 :=
  sorry

end min_triangle_perimeter_proof_l786_786101


namespace sequences_to_CCAMT_l786_786454

theorem sequences_to_CCAMT : 
  let n := 5 
  in let freq_C := 2
  in let freq_A := 1
  in let freq_M := 1
  in let freq_T := 1
  in
  number_of_sequences_to_form_word "CCAMT" n freq_C freq_A freq_M freq_T = 60 := 
sorry

end sequences_to_CCAMT_l786_786454


namespace solve_for_z_l786_786823

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786823


namespace find_z_l786_786814

theorem find_z (z : ℂ) (h : (1 - complex.i)^2 * z = 3 + 2 * complex.i) : z = -1 + 3 / 2 * complex.i :=
sorry

end find_z_l786_786814


namespace solve_for_z_l786_786824

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786824


namespace find_z_l786_786809

theorem find_z 
  (z : ℂ) 
  (h : (1 - complex.I) ^ 2 * z = 3 + 2 * complex.I) : 
  z = -1 + (3 / 2) * complex.I :=
  sorry

end find_z_l786_786809


namespace triangle_angle_A_eq_pi_div_3_triangle_area_l786_786001

variable (A B C a b c : ℝ)
variable (S : ℝ)

-- First part: Proving A = π / 3
theorem triangle_angle_A_eq_pi_div_3 (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                                      (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : A > 0) (h6 : A < Real.pi) :
  A = Real.pi / 3 :=
sorry

-- Second part: Finding the area of the triangle
theorem triangle_area (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                      (h2 : b + c = Real.sqrt 10) (h3 : a = 2) (h4 : A = Real.pi / 3) :
  S = Real.sqrt 3 / 2 :=
sorry

end triangle_angle_A_eq_pi_div_3_triangle_area_l786_786001


namespace solve_z_l786_786849

-- Defining the given condition
def condition (z : ℂ) : Prop := (1 - complex.I)^2 * z = 3 + 2 * complex.I

-- Stating the theorem that needs to be proved
theorem solve_z : ∃ z : ℂ, condition z ∧ z = -1 + 3 / 2 * complex.I :=
by {
  -- Proof skipped
  sorry
}

end solve_z_l786_786849


namespace not_correct_tangent_definition_not_needs_curve_on_one_side_l786_786282

-- Define what it means for the definition to be correct
def correct_tangent_definition : Prop :=
  ∀ (C : ℝ → ℝ) (T : ℝ → ℝ), (∃ x : ℝ, ∀ y : ℝ, C y = T y ↔ y = x) →
  (∃ ε > 0, ∀ y : ℝ, y ≠ x → (y > x - ε ∧ y < x + ε → C y > T y ∨ C y < T y)) 

-- Define what it means for the curve to need to lie on one side
def needs_curve_on_one_side : Prop :=
  ∀ (C : ℝ → ℝ) (T : ℝ → ℝ) (x : ℝ), (C x = T x ∧ 
  (∃ ε > 0, ∀ y : ℝ, y ≠ x → (y > x - ε ∧ y < x + ε → (C y > T y ∨ C y < T y))))

theorem not_correct_tangent_definition : ¬ correct_tangent_definition :=
by sorry

theorem not_needs_curve_on_one_side : ¬ needs_curve_on_one_side :=
by sorry

end not_correct_tangent_definition_not_needs_curve_on_one_side_l786_786282


namespace positive_integer_solutions_count_l786_786942

theorem positive_integer_solutions_count : 
  {x : ℤ // x > 0 ∧ 15 < -2 * x + 20}.to_finset.card = 2 := by
  sorry

end positive_integer_solutions_count_l786_786942


namespace solve_for_z_l786_786830

theorem solve_for_z : ∃ (z : ℂ), ((1 - complex.i)^2 * z = 3 + 2 * complex.i) ∧ (z = -1 + complex.i * (3 / 2)) :=
by {
    use -1 + complex.i * (3 / 2),
    sorry
}

end solve_for_z_l786_786830


namespace angle_FMP_right_angle_l786_786152

theorem angle_FMP_right_angle 
  (A B C D E P F M : Point) 
  (ω : Circle) 
  (hω : ω.tangent_at D = A ∧ ω.tangent_at E = A)
  (hP : P ∈ larger_arc (D E))
  (hFD_symm : F = symmetric_point A (line_through D P))
  (hmid_DE : M = midpoint D E) 
  : angle F M P = 90 := 
sorry

end angle_FMP_right_angle_l786_786152


namespace solve_z_l786_786780

theorem solve_z (z : ℂ) : ((1 - complex.I)^2) * z = 3 + 2 * complex.I → z = -1 + (3 / 2) * complex.I :=
by sorry

end solve_z_l786_786780


namespace swimmers_pass_each_other_l786_786623

/-- Two swimmers in a 100-foot pool, one swimming at 4 feet per second, the other at 3 feet per second,
    continuously for 12 minutes, pass each other exactly 32 times. -/
theorem swimmers_pass_each_other 
  (pool_length : ℕ) 
  (time : ℕ) 
  (rate1 : ℕ)
  (rate2 : ℕ)
  (meet_times : ℕ)
  (hp : pool_length = 100) 
  (ht : time = 720) -- 12 minutes = 720 seconds
  (hr1 : rate1 = 4) 
  (hr2 : rate2 = 3)
  : meet_times = 32 := 
sorry

end swimmers_pass_each_other_l786_786623


namespace number_greater_than_neg_one_by_two_l786_786158

/-- Theorem: The number that is greater than -1 by 2 is 1. -/
theorem number_greater_than_neg_one_by_two : -1 + 2 = 1 :=
by
  sorry

end number_greater_than_neg_one_by_two_l786_786158


namespace max_min_values_l786_786599

theorem max_min_values (f : ℝ → ℝ) {a b : ℝ} (h : a ≤ b) (hf_def : ∀ x, f x = x^3 - 3 * x + 1) :
  (-3 ≤ 0) → ∃ (x_max x_min : ℝ), x_max = 3 ∧ x_min = -17 ∧ ∀ x ∈ set.Icc (-3) 0, f x_min ≤ f x ∧ f x ≤ f x_max :=
by {
  sorry
}

end max_min_values_l786_786599


namespace tangent_line_parabola_max_area_parabola_l786_786922

-- Part 1: proving b = -4
theorem tangent_line_parabola (b : ℝ) : 
  (∀ x y : ℝ, (y = 2*x + b) → (x^2 = 4*y) → (x^2 - 8*x - 4*b = 0)) → 
  b = -4 :=
sorry

-- Part 2: proving maximum area of Δ ABP 
theorem max_area_parabola (A B P : ℝ × ℝ) :
  (∀ (x y : ℝ), y = 2*x + 1 → x^2 = 4*y) →
  (A.1 + B.1 = 8 ∧ (A.1 * B.1 = -4)) →
  (∀ t, 4 - 2 * Real.sqrt 5 < t ∧ t < 4 + 2 * Real.sqrt 5 → P = (t, t^2/4)) →
  (∃ t, A = (4 - Real.sqrt 5, (4 - Real.sqrt 5)^2/4) ∧ B = (4 + Real.sqrt 5, (4 + Real.sqrt 5)^2/4)) →
  let d := (abs ((1/4) * (P.1 - 4)^2 - 5))/(Real.sqrt 5) in
  d = Real.sqrt 5 →
  (let |AB| := 20 in (1/2) * |AB| * d = 10 * Real.sqrt 5) :=
sorry

end tangent_line_parabola_max_area_parabola_l786_786922


namespace probability_2_to_4_l786_786910

noncomputable def normal_distribution_probability 
  (X : ℝ → ℝ) (μ σ : ℝ) (hX : ∀ u, X u = Real.gausspdf μ σ u) : Prop :=
∃ (P : Set ℝ → ℝ), P 2 < X ∧ X ≤ 4 = 0.34

theorem probability_2_to_4 
  (X : ℝ → ℝ) (μ σ : ℝ) (hX : ∀ u, X u = Real.gausspdf 2 σ u)
  (h1 : (Real.cdf 2 σ 0) = 0.16) :
  normal_distribution_probability X 2 σ hX :=
begin
  have : Real.cdf 2 σ 4 - Real.cdf 2 σ 2 = 0.34,
  { sorry }
end

end probability_2_to_4_l786_786910


namespace time_to_paint_one_room_l786_786674

variables (rooms_total rooms_painted : ℕ) (hours_to_paint_remaining : ℕ)

-- The conditions
def painter_conditions : Prop :=
  rooms_total = 10 ∧ rooms_painted = 8 ∧ hours_to_paint_remaining = 16

-- The goal is to find out the hours to paint one room
theorem time_to_paint_one_room (h : painter_conditions rooms_total rooms_painted hours_to_paint_remaining) : 
  let rooms_remaining := rooms_total - rooms_painted
  let hours_per_room := hours_to_paint_remaining / rooms_remaining
  hours_per_room = 8 :=
by sorry

end time_to_paint_one_room_l786_786674


namespace digit_sum_distances_l786_786108

theorem digit_sum_distances :
  ∀ (a b : ℕ), (100 ≤ a ∧ a < 1000) → (100 ≤ b ∧ b < 1000) →
  (∀ a, a = 100 * a.div 100 + 10 * ((a % 100) / 10) + a % 10 → (a.div 100 + (a % 100) / 10 + a % 10 = 16)) →
  let d_max := abs (970 - 169) in
  let d_min := abs (178 - 169) in
  max (abs (a - b)) d_max ∧ min (abs (a - b)) d_min := by
  sorry

end digit_sum_distances_l786_786108


namespace speed_of_faster_train_l786_786247

-- Define the conditions
def slower_train_speed_kmph : ℝ := 36                    -- Slower train speed in kmph
def crossing_time_seconds : ℝ := 18                      -- Time taken to cross in seconds
def faster_train_length_meters : ℝ := 180                -- Length of the faster train in meters

-- Define the conversion factor from kmph to m/s
def kmph_to_mps (v_kmph : ℝ) : ℝ := v_kmph / 3.6

-- Define the known speed of slower train in m/s
def slower_train_speed_mps : ℝ := kmph_to_mps slower_train_speed_kmph

-- Proof statement
theorem speed_of_faster_train (V_f : ℝ) : 
  (V_f - slower_train_speed_mps) = (faster_train_length_meters / crossing_time_seconds) 
  → V_f = 72 :=
by
  intro h
  sorry

end speed_of_faster_train_l786_786247
