import Mathlib

namespace ways_to_distribute_balls_l114_114360

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114360


namespace ways_to_distribute_balls_l114_114352

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114352


namespace value_of_expression_l114_114109

theorem value_of_expression (a b m n x : ℝ) 
    (hab : a * b = 1) 
    (hmn : m + n = 0) 
    (hxsq : x^2 = 1) : 
    2022 * (m + n) + 2018 * x^2 - 2019 * (a * b) = -1 := 
by 
    sorry

end value_of_expression_l114_114109


namespace mailman_total_delivered_l114_114568

def pieces_of_junk_mail : Nat := 6
def magazines : Nat := 5
def newspapers : Nat := 3
def bills : Nat := 4
def postcards : Nat := 2

def total_pieces_of_mail : Nat := pieces_of_junk_mail + magazines + newspapers + bills + postcards

theorem mailman_total_delivered : total_pieces_of_mail = 20 := by
  sorry

end mailman_total_delivered_l114_114568


namespace balls_into_boxes_l114_114401

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114401


namespace num_ways_to_distribute_balls_into_boxes_l114_114453

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114453


namespace digit_after_decimal_l114_114937

theorem digit_after_decimal (n : ℕ) : 
  (Nat.floor (10 * (Real.sqrt (n^2 + n) - Nat.floor (Real.sqrt (n^2 + n))))) = 4 :=
by
  sorry

end digit_after_decimal_l114_114937


namespace alice_total_distance_correct_l114_114862

noncomputable def alice_daily_morning_distance : ℕ := 10

noncomputable def alice_daily_afternoon_distance : ℕ := 12

noncomputable def alice_daily_distance : ℕ :=
  alice_daily_morning_distance + alice_daily_afternoon_distance

noncomputable def alice_weekly_distance : ℕ :=
  5 * alice_daily_distance

theorem alice_total_distance_correct :
  alice_weekly_distance = 110 :=
by
  unfold alice_weekly_distance alice_daily_distance alice_daily_morning_distance alice_daily_afternoon_distance
  norm_num

end alice_total_distance_correct_l114_114862


namespace jessica_and_sibling_age_l114_114549

theorem jessica_and_sibling_age
  (J M S : ℕ)
  (h1 : J = M / 2)
  (h2 : M + 10 = 70)
  (h3 : S = J + ((70 - M) / 2)) :
  J = 40 ∧ S = 45 :=
by
  sorry

end jessica_and_sibling_age_l114_114549


namespace quadratic_equation_solution_l114_114478

-- We want to prove that for the conditions given, the only possible value of m is 3
theorem quadratic_equation_solution (m : ℤ) (h1 : m^2 - 7 = 2) (h2 : m + 3 ≠ 0) : m = 3 :=
sorry

end quadratic_equation_solution_l114_114478


namespace ways_to_put_balls_in_boxes_l114_114426

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114426


namespace distinct_balls_boxes_l114_114405

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114405


namespace balloons_problem_l114_114629

variable (b_J b_S b_J_f b_g : ℕ)

theorem balloons_problem
  (h1 : b_J = 9)
  (h2 : b_S = 5)
  (h3 : b_J_f = 12)
  (h4 : b_g = (b_J + b_S) - b_J_f)
  : b_g = 2 :=
by {
  sorry
}

end balloons_problem_l114_114629


namespace problem_equivalent_l114_114782

noncomputable def h (y : ℝ) : ℝ := y^5 - y^3 + 2
noncomputable def k (y : ℝ) : ℝ := y^2 - 3

theorem problem_equivalent (y₁ y₂ y₃ y₄ y₅ : ℝ) (h_roots : ∀ y, h y = 0 ↔ y = y₁ ∨ y = y₂ ∨ y = y₃ ∨ y = y₄ ∨ y = y₅) :
  (k y₁) * (k y₂) * (k y₃) * (k y₄) * (k y₅) = 104 :=
sorry

end problem_equivalent_l114_114782


namespace tan_neg_405_eq_neg1_l114_114698

theorem tan_neg_405_eq_neg1 : tan (-405 * real.pi / 180) = -1 :=
by 
  -- Simplify representing -405 degrees in radians and use known angle properties
  sorry

end tan_neg_405_eq_neg1_l114_114698


namespace polygon_side_count_l114_114540

theorem polygon_side_count (s : ℝ) (hs : s ≠ 0) : 
  ∀ (side_length_ratio : ℝ) (sides_first sides_second : ℕ),
  sides_first = 50 ∧ side_length_ratio = 3 ∧ 
  sides_first * side_length_ratio * s = sides_second * s → sides_second = 150 :=
by
  sorry

end polygon_side_count_l114_114540


namespace put_balls_in_boxes_l114_114441

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114441


namespace george_run_speed_last_half_mile_l114_114304

theorem george_run_speed_last_half_mile :
  ∀ (distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance)
    (today_total_time normal_total_time remaining_time : ℝ),
    distance_school = 2 →
    normal_speed = 4 →
    first_segment_distance = 3 / 4 →
    first_segment_speed = 3 →
    second_segment_distance = 3 / 4 →
    second_segment_speed = 4 →
    remaining_distance = 1 / 2 →
    normal_total_time = distance_school / normal_speed →
    today_total_time = (first_segment_distance / first_segment_speed) + (second_segment_distance / second_segment_speed) →
    normal_total_time = today_total_time + remaining_time →
    (remaining_distance / remaining_time) = 8 :=
by
  intros distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance today_total_time normal_total_time remaining_time h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end george_run_speed_last_half_mile_l114_114304


namespace original_triangle_area_l114_114648

-- Define the variables
variable (A_new : ℝ) (r : ℝ)

-- The conditions from the problem
def conditions := r = 5 ∧ A_new = 100

-- Goal: Prove that the original area is 4
theorem original_triangle_area (A_orig : ℝ) (h : conditions r A_new) : A_orig = 4 := by
  sorry

end original_triangle_area_l114_114648


namespace eccentricity_of_ellipse_l114_114314

theorem eccentricity_of_ellipse 
  (a b : ℝ) (e : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x = 0 ∧ y > 0 ∧ (9 * b^2 = 16/7 * a^2)) :
  e = Real.sqrt (10) / 6 :=
sorry

end eccentricity_of_ellipse_l114_114314


namespace second_class_students_count_l114_114957

theorem second_class_students_count 
    (x : ℕ)
    (h1 : 12 * 40 = 480)
    (h2 : ∀ x, x * 60 = 60 * x)
    (h3 : (12 + x) * 54 = 480 + 60 * x) : 
    x = 28 :=
by
  sorry

end second_class_students_count_l114_114957


namespace combined_volume_cone_hemisphere_cylinder_l114_114067

theorem combined_volume_cone_hemisphere_cylinder (r h : ℝ)
  (vol_cylinder : ℝ) (vol_cone : ℝ) (vol_hemisphere : ℝ)
  (H1 : vol_cylinder = 72 * π)
  (H2 : vol_cylinder = π * r^2 * h)
  (H3 : vol_cone = (1/3) * π * r^2 * h)
  (H4 : vol_hemisphere = (2/3) * π * r^3)
  (H5 : vol_cylinder = vol_cone + vol_hemisphere) :
  vol_cylinder = 72 * π :=
by
  sorry

end combined_volume_cone_hemisphere_cylinder_l114_114067


namespace no_solution_to_a_l114_114017

theorem no_solution_to_a (x : ℝ) :
  (4 * x - 1) / 6 - (5 * x - 2 / 3) / 10 + (9 - x / 2) / 3 ≠ 101 / 20 := 
sorry

end no_solution_to_a_l114_114017


namespace sum_of_powers_of_4_l114_114857

theorem sum_of_powers_of_4 : 4^0 + 4^1 + 4^2 + 4^3 = 85 :=
by
  sorry

end sum_of_powers_of_4_l114_114857


namespace ways_to_place_balls_in_boxes_l114_114386

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114386


namespace number_of_triangles_in_decagon_l114_114265

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l114_114265


namespace tan_neg_405_eq_neg_1_l114_114701

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l114_114701


namespace distinct_balls_boxes_l114_114383

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114383


namespace remainder_n_plus_2023_l114_114550

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 :=
sorry

end remainder_n_plus_2023_l114_114550


namespace smallest_n_in_range_l114_114100

theorem smallest_n_in_range : ∃ n : ℕ, n > 1 ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 8 = 2) ∧ 120 ≤ n ∧ n ≤ 149 :=
by
  sorry

end smallest_n_in_range_l114_114100


namespace cleared_land_with_corn_is_630_acres_l114_114155

-- Definitions based on given conditions
def total_land : ℝ := 6999.999999999999
def cleared_fraction : ℝ := 0.90
def potato_fraction : ℝ := 0.20
def tomato_fraction : ℝ := 0.70

-- Calculate the cleared land
def cleared_land : ℝ := cleared_fraction * total_land

-- Calculate the land used for potato and tomato
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := tomato_fraction * cleared_land

-- Define the land planted with corn
def corn_land : ℝ := cleared_land - (potato_land + tomato_land)

-- The theorem to be proved
theorem cleared_land_with_corn_is_630_acres : corn_land = 630 := by
  sorry

end cleared_land_with_corn_is_630_acres_l114_114155


namespace speed_of_stream_l114_114672

-- Definitions
variable (b s : ℝ)
def downstream_distance : ℝ := 120
def downstream_time : ℝ := 4
def upstream_distance : ℝ := 90
def upstream_time : ℝ := 6

-- Equations
def downstream_eq : Prop := downstream_distance = (b + s) * downstream_time
def upstream_eq : Prop := upstream_distance = (b - s) * upstream_time

-- Main statement
theorem speed_of_stream (h₁ : downstream_eq b s) (h₂ : upstream_eq b s) : s = 7.5 :=
by
  sorry

end speed_of_stream_l114_114672


namespace trisha_take_home_pay_l114_114038

theorem trisha_take_home_pay
  (hourly_pay : ℝ := 15)
  (hours_per_week : ℝ := 40)
  (weeks_per_year : ℝ := 52)
  (withholding_percentage : ℝ := 0.20) :
  let annual_gross_pay := hourly_pay * hours_per_week * weeks_per_year,
      amount_withheld := annual_gross_pay * withholding_percentage,
      annual_take_home_pay := annual_gross_pay - amount_withheld
  in annual_take_home_pay = 24960 := by
    sorry

end trisha_take_home_pay_l114_114038


namespace balls_into_boxes_l114_114433

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114433


namespace distance_between_points_l114_114675

theorem distance_between_points :
  ∀ (D : ℝ), (10 + 2) * (5 / D) + (10 - 2) * (5 / D) = 24 ↔ D = 24 := 
sorry

end distance_between_points_l114_114675


namespace f_eq_f_inv_implies_x_eq_0_l114_114705

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1
noncomputable def f_inv (x : ℝ) : ℝ := (-1 + Real.sqrt (3 * x + 4)) / 3

theorem f_eq_f_inv_implies_x_eq_0 (x : ℝ) : f x = f_inv x → x = 0 :=
by
  sorry

end f_eq_f_inv_implies_x_eq_0_l114_114705


namespace num_triangles_in_decagon_l114_114281

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l114_114281


namespace price_of_each_sundae_l114_114833

theorem price_of_each_sundae (A B : ℝ) (x y z : ℝ) (hx : 200 * x = 80) (hy : A = y) (hz : y = 0.40)
  (hxy : A - 80 = z) (hyz : 200 * z = B) : y = 0.60 :=
by
  sorry

end price_of_each_sundae_l114_114833


namespace greatest_fraction_l114_114825

theorem greatest_fraction :
  (∃ frac, frac ∈ {
    (44444 : ℚ)/55555,
    (5555 : ℚ)/6666,
    (666 : ℚ)/777,
    (77 : ℚ)/88,
    (8 : ℚ)/9
  } ∧ frac = (8 : ℚ)/9 ∧ ∀ f ∈ {
    (44444 : ℚ)/55555,
    (5555 : ℚ)/6666,
    (666 : ℚ)/777,
    (77 : ℚ)/88,
    (8 : ℚ)/9
  }, frac ≥ f) :=
sorry

end greatest_fraction_l114_114825


namespace adoption_event_l114_114575

theorem adoption_event (c : ℕ) 
  (h1 : ∀ d : ℕ, d = 8) 
  (h2 : ∀ fees_dog : ℕ, fees_dog = 15) 
  (h3 : ∀ fees_cat : ℕ, fees_cat = 13)
  (h4 : ∀ donation : ℕ, donation = 53)
  (h5 : fees_dog * 8 + fees_cat * c = 159) :
  c = 3 :=
by 
  sorry

end adoption_event_l114_114575


namespace first_month_sale_l114_114209

def sale_second_month : ℕ := 5744
def sale_third_month : ℕ := 5864
def sale_fourth_month : ℕ := 6122
def sale_fifth_month : ℕ := 6588
def sale_sixth_month : ℕ := 4916
def average_sale_six_months : ℕ := 5750

def expected_total_sales : ℕ := 6 * average_sale_six_months
def known_sales : ℕ := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month

theorem first_month_sale :
  (expected_total_sales - (known_sales + sale_sixth_month)) = 5266 :=
by
  sorry

end first_month_sale_l114_114209


namespace max_distinct_colorings_5x5_l114_114814

theorem max_distinct_colorings_5x5 (n : ℕ) :
  ∃ N, N ≤ (n^25 + 4 * n^15 + n^13 + 2 * n^7) / 8 :=
sorry

end max_distinct_colorings_5x5_l114_114814


namespace num_triangles_from_decagon_l114_114241

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l114_114241


namespace arithmetic_sequence_15th_term_l114_114080

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_15th_term :
  arithmetic_sequence (-3) 4 15 = 53 :=
by
  sorry

end arithmetic_sequence_15th_term_l114_114080


namespace increase_in_daily_mess_expenses_l114_114537

theorem increase_in_daily_mess_expenses (A X : ℝ)
  (h1 : 35 * A = 420)
  (h2 : 42 * (A - 1) = 420 + X) :
  X = 42 :=
by
  sorry

end increase_in_daily_mess_expenses_l114_114537


namespace focus_of_parabola_l114_114715

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l114_114715


namespace put_balls_in_boxes_l114_114442

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114442


namespace balls_into_boxes_l114_114431

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114431


namespace circle_value_l114_114097

theorem circle_value (c d s : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y + d)^2 = s^2)
  (h2 : c = 4)
  (h3 : d = -8)
  (h4 : s = 2 * Real.sqrt 5) :
  c + d + s = -4 + 2 * Real.sqrt 5 := 
sorry

end circle_value_l114_114097


namespace compare_2_pow_n_n_sq_l114_114823

theorem compare_2_pow_n_n_sq (n : ℕ) (h : n > 0) :
  (n = 1 → 2^n > n^2) ∧
  (n = 2 → 2^n = n^2) ∧
  (n = 3 → 2^n < n^2) ∧
  (n = 4 → 2^n = n^2) ∧
  (n ≥ 5 → 2^n > n^2) :=
by sorry

end compare_2_pow_n_n_sq_l114_114823


namespace num_ways_to_distribute_balls_into_boxes_l114_114455

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114455


namespace am_gm_inequality_l114_114002

theorem am_gm_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧ (x + y) / 2 = Real.sqrt (x * y) ↔ x = y := by
  sorry

end am_gm_inequality_l114_114002


namespace exists_indices_divisible_2019_l114_114502

theorem exists_indices_divisible_2019 (x : Fin 2020 → ℤ) : 
  ∃ (i j : Fin 2020), i ≠ j ∧ (x j - x i) % 2019 = 0 := 
  sorry

end exists_indices_divisible_2019_l114_114502


namespace D_144_l114_114501

def D (n : ℕ) : ℕ :=
  if n = 1 then 1 else sorry

theorem D_144 : D 144 = 51 := by
  sorry

end D_144_l114_114501


namespace cos_diff_proof_l114_114744

theorem cos_diff_proof (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5 / 8 := 
by
  sorry

end cos_diff_proof_l114_114744


namespace tan_neg405_deg_l114_114695

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l114_114695


namespace min_folds_exceed_12mm_l114_114123

theorem min_folds_exceed_12mm : ∃ n : ℕ, 0.1 * (2: ℝ)^n > 12 ∧ ∀ m < n, 0.1 * (2: ℝ)^m ≤ 12 := 
by
  sorry

end min_folds_exceed_12mm_l114_114123


namespace simplify_and_evaluate_l114_114798

/-- 
Given the expression (1 + 1 / (x - 2)) ÷ ((x ^ 2 - 2 * x + 1) / (x - 2)), 
prove that it evaluates to -1 when x = 0.
-/
theorem simplify_and_evaluate (x : ℝ) (h : x = 0) :
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x - 2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l114_114798


namespace cafe_purchase_l114_114127

theorem cafe_purchase (s d : ℕ) (h_d : d ≥ 2) (h_cost : 5 * s + 125 * d = 4000) :  s + d = 11 :=
    -- Proof steps go here
    sorry

end cafe_purchase_l114_114127


namespace ball_in_boxes_l114_114341

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114341


namespace landA_area_and_ratio_l114_114560

/-
  a = 3, b = 5, c = 6
  p = 1/2 * (a + b + c)
  S = sqrt(p * (p - a) * (p - b) * (p - c))
  S_A = 2 * sqrt(14)
  S_B = 3/2 * sqrt(14)
  S_A / S_B = 4 / 3
-/
theorem landA_area_and_ratio :
  let a := 3
  let b := 5
  let c := 6
  let p := (a + b + c) / 2
  let S_A := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let S_B := 3 / 2 * Real.sqrt 14
  S_A = 2 * Real.sqrt 14 ∧ S_A / S_B = 4 / 3 :=
by
  sorry

end landA_area_and_ratio_l114_114560


namespace number_ordering_l114_114046

theorem number_ordering : (10^5 < 2^20) ∧ (2^20 < 5^10) :=
by {
  -- We place the proof steps here
  sorry
}

end number_ordering_l114_114046


namespace ways_to_distribute_balls_l114_114362

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114362


namespace fraction_of_female_attendees_on_time_l114_114081

theorem fraction_of_female_attendees_on_time (A : ℝ)
  (h1 : 3 / 5 * A = M)
  (h2 : 7 / 8 * M = M_on_time)
  (h3 : 0.115 * A = n_A_not_on_time) :
  0.9 * F = (A - M_on_time - n_A_not_on_time)/((2 / 5) * A - n_A_not_on_time) :=
by
  sorry

end fraction_of_female_attendees_on_time_l114_114081


namespace circle_properties_radius_properties_l114_114752

theorem circle_properties (m x y : ℝ) :
  (x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) ↔
    (-((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :=
sorry

theorem radius_properties (m : ℝ) (h : -((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :
  ∃ r : ℝ, (0 < r ∧ r ≤ (4 / Real.sqrt 7)) :=
sorry

end circle_properties_radius_properties_l114_114752


namespace width_of_lawn_is_60_l114_114677

-- Define the problem conditions in Lean
def length_of_lawn : ℕ := 70
def road_width : ℕ := 10
def total_road_cost : ℕ := 3600
def cost_per_sq_meter : ℕ := 3

-- Define the proof problem
theorem width_of_lawn_is_60 (W : ℕ) 
  (h1 : (road_width * W) + (road_width * length_of_lawn) - (road_width * road_width) 
        = total_road_cost / cost_per_sq_meter) : 
  W = 60 := 
by 
  sorry

end width_of_lawn_is_60_l114_114677


namespace part_I_part_II_l114_114901

noncomputable section

def f (x a : ℝ) : ℝ := |x + a| + |x - (1 / a)|

theorem part_I (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -5/2 ∨ x ≥ 5/2 := by
  sorry

theorem part_II (a m : ℝ) (h : ∀ x : ℝ, f x a ≥ |m - 1|) : -1 ≤ m ∧ m ≤ 3 := by
  sorry

end part_I_part_II_l114_114901


namespace number_of_triangles_in_regular_decagon_l114_114231

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l114_114231


namespace regular_decagon_triangle_count_l114_114258

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l114_114258


namespace possible_lengths_of_c_l114_114891

-- Definitions of the given conditions
variables (a b c : ℝ) (S : ℝ)
variables (h₁ : a = 4)
variables (h₂ : b = 5)
variables (h₃ : S = 5 * Real.sqrt 3)

-- The main theorem stating the possible lengths of c
theorem possible_lengths_of_c : c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
  sorry

end possible_lengths_of_c_l114_114891


namespace total_time_pushing_car_l114_114693

theorem total_time_pushing_car :
  let d1 := 3
  let s1 := 6
  let d2 := 3
  let s2 := 3
  let d3 := 4
  let s3 := 8
  let t1 := d1 / s1
  let t2 := d2 / s2
  let t3 := d3 / s3
  (t1 + t2 + t3) = 2 :=
by
  sorry

end total_time_pushing_car_l114_114693


namespace number_of_triangles_in_decagon_l114_114266

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l114_114266


namespace calculate_f_g_of_1_l114_114916

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem calculate_f_g_of_1 : f (g 1) = 39 :=
by
  -- Enable quick skippable proof with 'sorry'
  sorry

end calculate_f_g_of_1_l114_114916


namespace quadratic_not_divisible_by_49_l114_114512

theorem quadratic_not_divisible_by_49 (n : ℤ) : ¬ (n^2 + 3 * n + 4) % 49 = 0 := 
by
  sorry

end quadratic_not_divisible_by_49_l114_114512


namespace distinct_lines_through_point_and_parabola_l114_114612

noncomputable def num_distinct_lines : ℕ :=
  let num_divisors (n : ℕ) : ℕ :=
    have factors := [2^5, 3^2, 7]
    factors.foldl (fun acc f => acc * (f + 1)) 1
  (num_divisors 2016) / 2 -- as each pair (x_1, x_2) corresponds twice

theorem distinct_lines_through_point_and_parabola :
  num_distinct_lines = 36 :=
by
  sorry

end distinct_lines_through_point_and_parabola_l114_114612


namespace largest_possible_median_l114_114108

theorem largest_possible_median 
  (l : List ℕ)
  (h_l : l = [4, 5, 3, 7, 9, 6])
  (h_pos : ∀ n ∈ l, 0 < n)
  (additional : List ℕ)
  (h_additional_pos : ∀ n ∈ additional, 0 < n)
  (h_length : l.length + additional.length = 9) : 
  ∃ median, median = 7 :=
by
  sorry

end largest_possible_median_l114_114108


namespace ways_to_put_balls_in_boxes_l114_114423

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114423


namespace share_money_3_people_l114_114154

theorem share_money_3_people (total_money : ℝ) (amount_per_person : ℝ) (h1 : total_money = 3.75) (h2 : amount_per_person = 1.25) : 
  total_money / amount_per_person = 3 := by
  sorry

end share_money_3_people_l114_114154


namespace minimum_a_l114_114307

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 1)
noncomputable def g (x a : ℝ) : ℝ := f x - a

theorem minimum_a (a : ℝ) : (∃ x : ℝ, g x a = 0) ↔ (a ≥ 1) :=
by sorry

end minimum_a_l114_114307


namespace solve_for_x_l114_114987

theorem solve_for_x (x : ℝ) (h : (2 * x - 3) ^ (x + 3) = 1) : 
  x = -3 ∨ x = 2 ∨ x = 1 := 
sorry

end solve_for_x_l114_114987


namespace sum_of_x_intersections_l114_114113

theorem sum_of_x_intersections (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 + x) = f (3 - x))
  (m : ℕ) (xs : Fin m → ℝ) (ys : Fin m → ℝ)
  (h_intersection : ∀ i : Fin m, f (xs i) = |(xs i)^2 - 4 * (xs i) - 3|) :
  (Finset.univ.sum fun i => xs i) = 2 * m :=
by
  sorry

end sum_of_x_intersections_l114_114113


namespace ways_to_distribute_balls_l114_114350

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114350


namespace num_triangles_in_decagon_l114_114282

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l114_114282


namespace cos_double_angle_l114_114613

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2 * n) = 12) : Real.cos (2 * θ) = 5 / 6 := 
sorry

end cos_double_angle_l114_114613


namespace put_balls_in_boxes_l114_114438

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114438


namespace Tom_completes_wall_l114_114626

theorem Tom_completes_wall :
  let avery_rate_per_hour := (1:ℝ)/3
  let tom_rate_per_hour := (1:ℝ)/2
  let combined_rate_per_hour := avery_rate_per_hour + tom_rate_per_hour
  let portion_completed_together := combined_rate_per_hour * 1 
  let remaining_wall := 1 - portion_completed_together
  let time_for_tom := remaining_wall / tom_rate_per_hour
  time_for_tom = (1:ℝ)/3 := 
by 
  sorry

end Tom_completes_wall_l114_114626


namespace decagon_triangle_count_l114_114263

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l114_114263


namespace inequality_abc_l114_114633

theorem inequality_abc (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 2) (h₃ : 0 ≤ b) (h₄ : b ≤ 2) (h₅ : 0 ≤ c) (h₆ : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end inequality_abc_l114_114633


namespace increasing_function_solve_inequality_find_range_l114_114599

noncomputable def f : ℝ → ℝ := sorry
def a1 := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f x
def a2 := f 1 = 1
def a3 := ∀ m n : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ -1 ≤ n ∧ n ≤ 1 ∧ m + n ≠ 0 → (f m + f n) / (m + n) > 0

-- Statement for question (1)
theorem increasing_function : 
  (∀ x y : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x < f y) :=
by 
  apply sorry

-- Statement for question (2)
theorem solve_inequality (x : ℝ) :
  (f (x^2 - 1) + f (3 - 3*x) < 0 ↔ 1 < x ∧ x ≤ 4/3) :=
by 
  apply sorry

-- Statement for question (3)
theorem find_range (t : ℝ) :
  (∀ x a : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ a ∧ a ≤ 1 → f x ≤ t^2 - 2*a*t + 1) 
  ↔ (2 ≤ t ∨ t ≤ -2 ∨ t = 0) :=
by 
  apply sorry

end increasing_function_solve_inequality_find_range_l114_114599


namespace correct_system_of_equations_l114_114835

theorem correct_system_of_equations (x y : ℕ) (h1 : x + y = 145) (h2 : 10 * x + 12 * y = 1580) :
  (x + y = 145) ∧ (10 * x + 12 * y = 1580) :=
by
  sorry

end correct_system_of_equations_l114_114835


namespace double_burger_cost_l114_114854

theorem double_burger_cost (D : ℝ) : 
  let single_burger_cost := 1.00
  let total_burgers := 50
  let double_burgers := 37
  let total_cost := 68.50
  let single_burgers := total_burgers - double_burgers
  let singles_cost := single_burgers * single_burger_cost
  let doubles_cost := total_cost - singles_cost
  let burger_cost := doubles_cost / double_burgers
  burger_cost = D := 
by 
  sorry

end double_burger_cost_l114_114854


namespace wheel_rpm_is_approximately_5000_23_l114_114060

noncomputable def bus_wheel_rpm (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speed_cm_per_min := (speed * 1000 * 100) / 60
  speed_cm_per_min / circumference

-- Conditions
def radius := 35
def speed := 66

-- Question (to be proved)
theorem wheel_rpm_is_approximately_5000_23 : 
  abs (bus_wheel_rpm radius speed - 5000.23) < 0.01 :=
by
  sorry

end wheel_rpm_is_approximately_5000_23_l114_114060


namespace inequality_proof_l114_114783

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ 2 * (a^3 + b^3 + c^3) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l114_114783


namespace ways_to_put_balls_in_boxes_l114_114420

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114420


namespace SUVs_purchased_l114_114968

theorem SUVs_purchased (x : ℕ) (hToyota : ℕ) (hHonda : ℕ) (hNissan : ℕ) 
  (hRatioToyota : hToyota = 7 * x) 
  (hRatioHonda : hHonda = 5 * x) 
  (hRatioNissan : hNissan = 3 * x) 
  (hToyotaSUV : ℕ) (hHondaSUV : ℕ) (hNissanSUV : ℕ) 
  (hToyotaSUV_num : hToyotaSUV = (50 * hToyota) / 100) 
  (hHondaSUV_num : hHondaSUV = (40 * hHonda) / 100) 
  (hNissanSUV_num : hNissanSUV = (30 * hNissan) / 100) : 
  hToyotaSUV + hHondaSUV + hNissanSUV = 64 := 
by
  sorry

end SUVs_purchased_l114_114968


namespace simplify_expression_l114_114498

theorem simplify_expression (a b c x : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  ( ( (x + a)^4 ) / ( (a - b) * (a - c) ) 
  + ( (x + b)^4 ) / ( (b - a) * (b - c) ) 
  + ( (x + c)^4 ) / ( (c - a) * (c - b) ) ) = a + b + c + 4 * x := 
by
  sorry

end simplify_expression_l114_114498


namespace least_positive_three_digit_multiple_of_8_l114_114973

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l114_114973


namespace mean_of_two_means_eq_l114_114521

theorem mean_of_two_means_eq (z : ℚ) (h : (5 + 10 + 20) / 3 = (15 + z) / 2) : z = 25 / 3 :=
by
  sorry

end mean_of_two_means_eq_l114_114521


namespace grandpa_tomatoes_before_vacation_l114_114925

theorem grandpa_tomatoes_before_vacation 
  (tomatoes_after_vacation : ℕ) 
  (growth_factor : ℕ) 
  (actual_number : ℕ) 
  (h1 : growth_factor = 100) 
  (h2 : tomatoes_after_vacation = 3564) 
  (h3 : actual_number = tomatoes_after_vacation / growth_factor) : 
  actual_number = 36 := 
by
  -- Here would be the step-by-step proof, but we use sorry to skip it
  sorry

end grandpa_tomatoes_before_vacation_l114_114925


namespace sum_ineq_l114_114640

theorem sum_ineq (x y z t : ℝ) (h₁ : x + y + z + t = 0) (h₂ : x^2 + y^2 + z^2 + t^2 = 1) :
  -1 ≤ x * y + y * z + z * t + t * x ∧ x * y + y * z + z * t + t * x ≤ 0 :=
by
  sorry

end sum_ineq_l114_114640


namespace divisor_of_p_l114_114001

theorem divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40)
  (hqr : Nat.gcd q r = 45) (hrs : Nat.gcd r s = 60)
  (hspr : 100 < Nat.gcd s p ∧ Nat.gcd s p < 150)
  : 7 ∣ p :=
sorry

end divisor_of_p_l114_114001


namespace valid_set_example_l114_114051

def is_valid_set (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, x ≠ y

theorem valid_set_example : is_valid_set { x : ℝ | x > Real.sqrt 2 } :=
sorry

end valid_set_example_l114_114051


namespace mean_of_combined_sets_l114_114649

theorem mean_of_combined_sets
  (S₁ : Finset ℝ) (S₂ : Finset ℝ)
  (h₁ : S₁.card = 7) (h₂ : S₂.card = 8)
  (mean_S₁ : (S₁.sum id) / S₁.card = 15)
  (mean_S₂ : (S₂.sum id) / S₂.card = 26)
  : (S₁.sum id + S₂.sum id) / (S₁.card + S₂.card) = 20.8667 := 
by
  sorry

end mean_of_combined_sets_l114_114649


namespace total_charge_for_3_hours_l114_114828

namespace TherapyCharges

-- Conditions
variables (A F : ℝ)
variable (h1 : F = A + 20)
variable (h2 : F + 4 * A = 300)

-- Prove that the total charge for 3 hours of therapy is 188
theorem total_charge_for_3_hours : F + 2 * A = 188 :=
by
  sorry

end TherapyCharges

end total_charge_for_3_hours_l114_114828


namespace ads_minutes_l114_114686

-- Definitions and conditions
def videos_per_day : Nat := 2
def minutes_per_video : Nat := 7
def total_time_on_youtube : Nat := 17

-- The theorem to prove
theorem ads_minutes : (total_time_on_youtube - (videos_per_day * minutes_per_video)) = 3 :=
by
  sorry

end ads_minutes_l114_114686


namespace num_triangles_in_decagon_l114_114279

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l114_114279


namespace mass_percentage_Ba_in_BaI2_l114_114872

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 : 
  (molar_mass_Ba / molar_mass_BaI2) * 100 = 35.11 := 
  by 
    -- implementing the proof here would demonstrate that (137.33 / 391.13) * 100 = 35.11
    sorry

end mass_percentage_Ba_in_BaI2_l114_114872


namespace find_x_l114_114958

-- Define the condition as a theorem
theorem find_x (x : ℝ) (h : (1 + 3 + x) / 3 = 3) : x = 5 :=
by
  sorry  -- Placeholder for the proof

end find_x_l114_114958


namespace f_neg_l114_114936

-- Define the function f and its properties
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else sorry

-- Define the property of f being an odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of f for non-negative x
axiom f_nonneg : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

-- The theorem to be proven
theorem f_neg : ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := by
  sorry

end f_neg_l114_114936


namespace a4_plus_a5_eq_27_l114_114777

-- Define the geometric sequence conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a_2 : a 2 = 1 - a 1
axiom a_4 : a 4 = 9 - a 3

-- Define the geometric sequence property
axiom geom_seq : ∀ n, a (n + 1) = a n * q

theorem a4_plus_a5_eq_27 : a 4 + a 5 = 27 := sorry

end a4_plus_a5_eq_27_l114_114777


namespace distance_between_points_A_and_B_l114_114111

theorem distance_between_points_A_and_B :
  ∃ (d : ℝ), 
    -- Distance must be non-negative
    d ≥ 0 ∧
    -- Condition 1: Car 3 reaches point A at 10:00 AM (3 hours after 7:00 AM)
    (∃ V3 : ℝ, V3 = d / 6) ∧ 
    -- Condition 2: Car 2 reaches point A at 10:30 AM (3.5 hours after 7:00 AM)
    (∃ V2 : ℝ, V2 = 2 * d / 7) ∧ 
    -- Condition 3: When Car 1 and Car 3 meet, Car 2 has traveled exactly 3/8 of d
    (∃ V1 : ℝ, V1 = (d - 84) / 7 ∧ 2 * V1 + 2 * V3 = 8 * V2 / 3) ∧ 
    -- Required: The distance between A and B is 336 km
    d = 336 :=
by
  sorry

end distance_between_points_A_and_B_l114_114111


namespace power_calculation_l114_114546

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l114_114546


namespace power_calculation_l114_114544

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l114_114544


namespace given_cond_then_geq_eight_l114_114105

theorem given_cond_then_geq_eight (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 1) : 
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := 
  sorry

end given_cond_then_geq_eight_l114_114105


namespace problem1_problem2_problem3_problem4_l114_114853

theorem problem1 : 12 - (-1) + (-7) = 6 := by
  sorry

theorem problem2 : -3.5 * (-3 / 4) / (7 / 8) = 3 := by
  sorry

theorem problem3 : (1 / 3 - 1 / 6 - 1 / 12) * (-12) = -1 := by
  sorry

theorem problem4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by
  sorry

end problem1_problem2_problem3_problem4_l114_114853


namespace probability_all_boxes_non_empty_equals_4_over_9_l114_114213

structure PaintingPlacement :=
  (paintings : Finset ℕ)
  (boxes : Finset ℕ)
  (num_paintings : paintings.card = 4)
  (num_boxes : boxes.card = 3)

noncomputable def probability_non_empty_boxes (pp : PaintingPlacement) : ℚ :=
  let total_outcomes := 3^4
  let favorable_outcomes := Nat.choose 4 2 * Nat.factorial 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_all_boxes_non_empty_equals_4_over_9
  (pp : PaintingPlacement) : pp.paintings.card = 4 → pp.boxes.card = 3 →
  probability_non_empty_boxes pp = 4 / 9 :=
by
  intros h1 h2
  sorry

end probability_all_boxes_non_empty_equals_4_over_9_l114_114213


namespace net_gain_mr_A_l114_114005

def home_worth : ℝ := 12000
def sale1 : ℝ := home_worth * 1.2
def sale2 : ℝ := sale1 * 0.85
def sale3 : ℝ := sale2 * 1.1

theorem net_gain_mr_A : sale1 - sale2 + sale3 = 3384 := by
  sorry -- Proof will be provided here

end net_gain_mr_A_l114_114005


namespace pet_store_cages_l114_114212

theorem pet_store_cages (total_puppies sold_puppies puppies_per_cage : ℕ) (h1 : total_puppies = 45) (h2 : sold_puppies = 39) (h3 : puppies_per_cage = 2) :
  (total_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l114_114212


namespace triangles_from_decagon_l114_114271

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l114_114271


namespace mittens_in_each_box_l114_114657

theorem mittens_in_each_box (boxes scarves_per_box total_clothing : ℕ) (h1 : boxes = 8) (h2 : scarves_per_box = 4) (h3 : total_clothing = 80) :
  ∃ (mittens_per_box : ℕ), mittens_per_box = 6 :=
by
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  let mittens_per_box := total_mittens / boxes
  use mittens_per_box
  sorry

end mittens_in_each_box_l114_114657


namespace inequality_solution_l114_114173

noncomputable def solution_set (x : ℝ) : Prop := 
  (x < -1) ∨ (x > 3)

theorem inequality_solution :
  { x : ℝ | (3 - x) / (x + 1) < 0 } = { x : ℝ | solution_set x } :=
by
  sorry

end inequality_solution_l114_114173


namespace extreme_points_sum_gt_two_l114_114321

noncomputable def f (x : ℝ) (b : ℝ) := x^2 / 2 + b * Real.exp x
noncomputable def f_prime (x : ℝ) (b : ℝ) := x + b * Real.exp x

theorem extreme_points_sum_gt_two
  (b : ℝ)
  (h_b : -1 / Real.exp 1 < b ∧ b < 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : f_prime x₁ b = 0)
  (h_x₂ : f_prime x₂ b = 0)
  (h_x₁_lt_x₂ : x₁ < x₂) :
  x₁ + x₂ > 2 := by
  sorry

end extreme_points_sum_gt_two_l114_114321


namespace abs_gt_x_iff_x_lt_0_l114_114205

theorem abs_gt_x_iff_x_lt_0 (x : ℝ) : |x| > x ↔ x < 0 := 
by
  sorry

end abs_gt_x_iff_x_lt_0_l114_114205


namespace uniformity_comparison_l114_114513

theorem uniformity_comparison (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : S1^2 < S2^2 :=
by {
  sorry
}

end uniformity_comparison_l114_114513


namespace distinguish_ball_box_ways_l114_114473

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114473


namespace larger_number_is_eight_l114_114920

variable {x y : ℝ}

theorem larger_number_is_eight (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end larger_number_is_eight_l114_114920


namespace elephant_entry_duration_l114_114543

theorem elephant_entry_duration
  (initial_elephants : ℕ)
  (exodus_duration : ℕ)
  (leaving_rate : ℕ)
  (entering_rate : ℕ)
  (final_elephants : ℕ)
  (h_initial : initial_elephants = 30000)
  (h_exodus_duration : exodus_duration = 4)
  (h_leaving_rate : leaving_rate = 2880)
  (h_entering_rate : entering_rate = 1500)
  (h_final : final_elephants = 28980) :
  (final_elephants - (initial_elephants - (exodus_duration * leaving_rate))) / entering_rate = 7 :=
by
  sorry

end elephant_entry_duration_l114_114543


namespace probability_point_between_C_and_E_l114_114950

noncomputable def length_between_points (total_length : ℝ) (ratio : ℝ) : ℝ :=
ratio * total_length

theorem probability_point_between_C_and_E
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : C < E)
  (h3 : B - A = 4 * (D - A))
  (h4 : B - A = 8 * (B - C))
  (h5 : B - E = 2 * (E - C)) :
  (E - C) / (B - A) = 1 / 24 :=
by 
  sorry

end probability_point_between_C_and_E_l114_114950


namespace balls_into_boxes_l114_114429

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114429


namespace system_of_equations_correct_l114_114178

theorem system_of_equations_correct (x y : ℝ) (h1 : x + y = 2000) (h2 : y = x * 0.30) :
  x + y = 2000 ∧ y = x * 0.30 :=
by 
  exact ⟨h1, h2⟩

end system_of_equations_correct_l114_114178


namespace number_of_divisors_64n4_l114_114876

theorem number_of_divisors_64n4 
  (n : ℕ) 
  (h1 : (factors (120 * n^3)).length = 120) 
  (h2 : 120.nat_factors.prod * (n^3).nat_factors.prod = (120 * n^3)) :
  (factors (64 * n^4)).length = 675 := 
sorry

end number_of_divisors_64n4_l114_114876


namespace ratio_of_cows_to_bulls_l114_114176

-- Define the total number of cattle
def total_cattle := 555

-- Define the number of bulls
def number_of_bulls := 405

-- Compute the number of cows
def number_of_cows := total_cattle - number_of_bulls

-- Define the expected ratio of cows to bulls
def expected_ratio_cows_to_bulls := (10, 27)

-- Prove that the ratio of cows to bulls is equal to the expected ratio
theorem ratio_of_cows_to_bulls : 
  (number_of_cows / (gcd number_of_cows number_of_bulls), number_of_bulls / (gcd number_of_cows number_of_bulls)) = expected_ratio_cows_to_bulls :=
sorry

end ratio_of_cows_to_bulls_l114_114176


namespace geometric_mean_2_6_l114_114803

theorem geometric_mean_2_6 : ∃ x : ℝ, x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end geometric_mean_2_6_l114_114803


namespace ball_box_distribution_l114_114459

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114459


namespace find_number_l114_114556

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 64) : x = 160 :=
sorry

end find_number_l114_114556


namespace alice_total_distance_correct_l114_114863

noncomputable def alice_daily_morning_distance : ℕ := 10

noncomputable def alice_daily_afternoon_distance : ℕ := 12

noncomputable def alice_daily_distance : ℕ :=
  alice_daily_morning_distance + alice_daily_afternoon_distance

noncomputable def alice_weekly_distance : ℕ :=
  5 * alice_daily_distance

theorem alice_total_distance_correct :
  alice_weekly_distance = 110 :=
by
  unfold alice_weekly_distance alice_daily_distance alice_daily_morning_distance alice_daily_afternoon_distance
  norm_num

end alice_total_distance_correct_l114_114863


namespace total_earnings_l114_114200

noncomputable def daily_wage_a (C : ℝ) := (3 * C) / 5
noncomputable def daily_wage_b (C : ℝ) := (4 * C) / 5
noncomputable def daily_wage_c (C : ℝ) := C

noncomputable def earnings_a (C : ℝ) := daily_wage_a C * 6
noncomputable def earnings_b (C : ℝ) := daily_wage_b C * 9
noncomputable def earnings_c (C : ℝ) := daily_wage_c C * 4

theorem total_earnings (C : ℝ) (h : C = 115) : 
  earnings_a C + earnings_b C + earnings_c C = 1702 :=
by
  sorry

end total_earnings_l114_114200


namespace total_animals_made_it_to_shore_l114_114665

def boat (total_sheep total_cows total_dogs sheep_drowned cows_drowned dogs_saved : Nat) : Prop :=
  cows_drowned = sheep_drowned * 2 ∧
  dogs_saved = total_dogs ∧
  total_sheep + total_cows + total_dogs - sheep_drowned - cows_drowned = 35

theorem total_animals_made_it_to_shore :
  boat 20 10 14 3 6 14 :=
by
  sorry

end total_animals_made_it_to_shore_l114_114665


namespace least_positive_three_digit_multiple_of_8_l114_114975

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l114_114975


namespace smallest_positive_integer_form_3003_55555_l114_114190

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l114_114190


namespace intersection_one_point_l114_114738

open Set

def A (x y : ℝ) : Prop := x^2 - 3*x*y + 4*y^2 = 7 / 2
def B (k x y : ℝ) : Prop := k > 0 ∧ k*x + y = 2

theorem intersection_one_point (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, A x y ∧ B k x y) → (∀ x₁ y₁ x₂ y₂ : ℝ, (A x₁ y₁ ∧ B k x₁ y₁) ∧ (A x₂ y₂ ∧ B k x₂ y₂) → x₁ = x₂ ∧ y₁ = y₂) ↔ k = 1 / 4 :=
sorry

end intersection_one_point_l114_114738


namespace maximum_area_l114_114217

variable {l w : ℝ}

theorem maximum_area (h1 : l + w = 200) (h2 : l ≥ 90) (h3 : w ≥ 50) (h4 : l ≤ 2 * w) : l * w ≤ 10000 :=
sorry

end maximum_area_l114_114217


namespace log2_3_value_l114_114317

variables (a b log2 log3 : ℝ)

-- Define the conditions
axiom h1 : a = log2 + log3
axiom h2 : b = 1 + log2

-- Define the logarithmic requirement to be proved
theorem log2_3_value : log2 * log3 = (a - b + 1) / (b - 1) :=
sorry

end log2_3_value_l114_114317


namespace ratio_of_radii_l114_114129

theorem ratio_of_radii 
  (a b : ℝ)
  (h1 : ∀ (a b : ℝ), π * b^2 - π * a^2 = 4 * π * a^2) : 
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l114_114129


namespace total_price_of_hats_l114_114655

-- Declare the conditions as Lean definitions
def total_hats : Nat := 85
def green_hats : Nat := 38
def blue_hat_cost : Nat := 6
def green_hat_cost : Nat := 7

-- The question becomes proving the total cost of the hats is $548
theorem total_price_of_hats :
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  total_blue_cost + total_green_cost = 548 := by
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  show total_blue_cost + total_green_cost = 548
  sorry

end total_price_of_hats_l114_114655


namespace average_headcount_correct_l114_114040

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end average_headcount_correct_l114_114040


namespace rice_and_flour_bags_l114_114647

theorem rice_and_flour_bags (x : ℕ) (y : ℕ) 
  (h1 : x + y = 351)
  (h2 : x + 20 = 3 * (y - 50) + 1) : 
  x = 221 ∧ y = 130 :=
by
  sorry

end rice_and_flour_bags_l114_114647


namespace student_marks_problem_l114_114531

-- Define the variables
variables (M P C X : ℕ)

-- State the conditions
-- Condition 1: M + P = 70
def condition1 : Prop := M + P = 70

-- Condition 2: C = P + X
def condition2 : Prop := C = P + X

-- Condition 3: (M + C) / 2 = 45
def condition3 : Prop := (M + C) / 2 = 45

-- The theorem stating the problem
theorem student_marks_problem (h1 : condition1 M P) (h2 : condition2 C P X) (h3 : condition3 M C) : X = 20 :=
by sorry

end student_marks_problem_l114_114531


namespace least_three_digit_multiple_of_8_l114_114979

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l114_114979


namespace Beth_peas_count_l114_114084

-- Definitions based on conditions
def number_of_corn : ℕ := 10
def number_of_peas (number_of_corn : ℕ) : ℕ := 2 * number_of_corn + 15

-- Theorem that represents the proof problem
theorem Beth_peas_count : number_of_peas 10 = 35 :=
by
  sorry

end Beth_peas_count_l114_114084


namespace value_2_std_devs_less_than_mean_l114_114663

-- Define the arithmetic mean
def mean : ℝ := 15.5

-- Define the standard deviation
def standard_deviation : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value_2_std_less_than_mean : ℝ := mean - 2 * standard_deviation

-- The theorem we want to prove
theorem value_2_std_devs_less_than_mean : value_2_std_less_than_mean = 12.5 := by
  sorry

end value_2_std_devs_less_than_mean_l114_114663


namespace range_of_m_l114_114309

theorem range_of_m (m : ℝ) : 
  (¬ (∀ x : ℝ, x^2 + m * x + 1 = 0 → x > 0) → m ≥ -2) :=
by
  sorry

end range_of_m_l114_114309


namespace simplify_expression_l114_114517

theorem simplify_expression (b c : ℝ) : 
  (2 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4 * 7 * c^2 = 5040 * b^10 * c^2) :=
by sorry

end simplify_expression_l114_114517


namespace log_one_fifth_25_eq_neg2_l114_114587

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end log_one_fifth_25_eq_neg2_l114_114587


namespace number_of_triangles_in_regular_decagon_l114_114229

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l114_114229


namespace plaster_cost_correct_l114_114055

def length : ℝ := 25
def width : ℝ := 12
def depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.30

def area_longer_walls : ℝ := 2 * (length * depth)
def area_shorter_walls : ℝ := 2 * (width * depth)
def area_bottom : ℝ := length * width
def total_area : ℝ := area_longer_walls + area_shorter_walls + area_bottom

def calculated_cost : ℝ := total_area * cost_per_sq_meter
def correct_cost : ℝ := 223.2

theorem plaster_cost_correct : calculated_cost = correct_cost := by
  sorry

end plaster_cost_correct_l114_114055


namespace total_votes_400_l114_114058

theorem total_votes_400 
    (V : ℝ)
    (h1 : ∃ (c1_votes c2_votes : ℝ), c1_votes = 0.70 * V ∧ c2_votes = 0.30 * V)
    (h2 : ∃ (majority : ℝ), majority = 160)
    (h3 : ∀ (c1_votes c2_votes majority : ℝ), c1_votes - c2_votes = majority) : V = 400 :=
by 
  sorry

end total_votes_400_l114_114058


namespace ball_in_boxes_l114_114347

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114347


namespace sara_added_onions_l114_114515

theorem sara_added_onions
  (initial_onions X : ℤ) 
  (h : initial_onions + X - 5 + 9 = initial_onions + 8) :
  X = 4 :=
by
  sorry

end sara_added_onions_l114_114515


namespace main_problem_l114_114753

noncomputable def sin_func (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem main_problem 
  (ω : ℝ) 
  (φ : ℝ) 
  (hω : ω > 0) 
  (hφ : φ ∈ Set.Ioo (-Real.pi) Real.pi) 
  (zero1 : sin_func ω φ (π / 3) = 0) 
  (zero2 : sin_func ω φ (5 * π / 6) = 0) : 
  (∃ k : ℤ, k ∈ (7 * π / 12 + (k : ℝ) * (π / 2))) ∧ 
  (∃ k : ℤ, φ = k * π - (2 * π / 3)) :=
sorry

end main_problem_l114_114753


namespace number_of_triangles_in_decagon_l114_114264

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l114_114264


namespace find_value_l114_114896

theorem find_value (a b : ℝ) (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) : 2 * a^100 - 3 * b⁻¹ = 3 :=
by sorry

end find_value_l114_114896


namespace least_positive_three_digit_multiple_of_8_l114_114974

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l114_114974


namespace number_of_triangles_in_regular_decagon_l114_114232

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l114_114232


namespace sum_of_multiples_of_4_between_34_and_135_l114_114061

theorem sum_of_multiples_of_4_between_34_and_135 :
  let first := 36
  let last := 132
  let n := (last - first) / 4 + 1
  let sum := n * (first + last) / 2
  sum = 2100 := 
by
  sorry

end sum_of_multiples_of_4_between_34_and_135_l114_114061


namespace total_revenue_correct_l114_114053

noncomputable def total_revenue : ℚ := 
  let revenue_v1 := 23 * 5 * 0.50
  let revenue_v2 := 28 * 6 * 0.60
  let revenue_v3 := 35 * 7 * 0.50
  let revenue_v4 := 43 * 8 * 0.60
  let revenue_v5 := 50 * 9 * 0.50
  let revenue_v6 := 64 * 10 * 0.60
  revenue_v1 + revenue_v2 + revenue_v3 + revenue_v4 + revenue_v5 + revenue_v6

theorem total_revenue_correct : total_revenue = 1096.20 := 
by
  sorry

end total_revenue_correct_l114_114053


namespace ways_to_place_balls_in_boxes_l114_114385

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114385


namespace number_of_triangles_in_decagon_l114_114227

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l114_114227


namespace ways_to_distribute_balls_l114_114353

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114353


namespace reflected_rectangle_has_no_point_neg_3_4_l114_114028

structure Point where
  x : ℤ
  y : ℤ
  deriving DecidableEq, Repr

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def is_not_vertex (pts: List Point) (p: Point) : Prop :=
  ¬ (p ∈ pts)

theorem reflected_rectangle_has_no_point_neg_3_4 :
  let initial_pts := [ Point.mk 1 3, Point.mk 1 1, Point.mk 4 1, Point.mk 4 3 ]
  let reflected_pts := initial_pts.map reflect_y
  is_not_vertex reflected_pts (Point.mk (-3) 4) :=
by
  sorry

end reflected_rectangle_has_no_point_neg_3_4_l114_114028


namespace smallest_positive_integer_linear_combination_l114_114182

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l114_114182


namespace investment_time_period_l114_114729

theorem investment_time_period :
  ∀ (A P : ℝ) (R : ℝ) (T : ℝ),
  A = 896 → P = 799.9999999999999 → R = 5 →
  (A - P) = (P * R * T / 100) → T = 2.4 :=
by
  intros A P R T hA hP hR hSI
  sorry

end investment_time_period_l114_114729


namespace log_base_one_fifth_twenty_five_l114_114588

theorem log_base_one_fifth_twenty_five : log (1/5) 25 = -2 :=
by
  sorry

end log_base_one_fifth_twenty_five_l114_114588


namespace select_numbers_to_sum_713_l114_114811

open Set

-- Definitions based on the problem statement
def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_5 (n : ℕ) : Prop := n % 5 ≠ 0
def ends_in_713 (n : ℕ) : Prop := n % 10000 = 713

-- Main theorem statement
theorem select_numbers_to_sum_713 (S : Set ℕ) (h1 : S.card = 1000)
  (h2 : ∀ s ∈ S, is_odd s) (h3 : ∀ s ∈ S, not_divisible_by_5 s) :
  ∃ T ⊆ S, ends_in_713 (T.sum id) := sorry

end select_numbers_to_sum_713_l114_114811


namespace sequence_general_term_correct_l114_114900

open Nat

def S (n : ℕ) : ℤ := 3 * (n : ℤ) * (n : ℤ) - 2 * (n : ℤ) + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2
  else 6 * (n : ℤ) - 5

theorem sequence_general_term_correct : ∀ n, (S n - S (n - 1) = a n) :=
by
  intros
  sorry

end sequence_general_term_correct_l114_114900


namespace simplify_expression_l114_114016

variables (a b : ℝ)

theorem simplify_expression : 
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6)) = -9 * a := by
  -- proof here
  sorry

end simplify_expression_l114_114016


namespace distinct_balls_boxes_l114_114406

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114406


namespace negation_proposition_l114_114805

theorem negation_proposition :
  (¬ (∀ x : ℝ, abs x + x^2 ≥ 0)) ↔ (∃ x₀ : ℝ, abs x₀ + x₀^2 < 0) :=
by
  sorry

end negation_proposition_l114_114805


namespace rabbit_probability_l114_114772

theorem rabbit_probability (rabbits : Finset ℕ) (measured_rabbits : Finset ℕ) (h_rabbits_card : rabbits.card = 5) (h_measured_card : measured_rabbits.card = 3) :
  ∃ (selected : Finset (Finset ℕ)), ∃ (probability : ℚ),
  (∀ (sel ∈ selected), sel.card = 3) ∧
  (∃ (favorable : Finset (Finset ℕ)), (∀ (fav ∈ favorable, ∃ (measured_count : ℕ), ∃ (unmeasured_count : ℕ), 
    fav.card = 3 ∧ (measured_count + unmeasured_count = 3) ∧ measured_count = 2 ∧ unmeasured_count = 1))) ∧
  probability = (favorably.card : ℚ) / (selected.card : ℚ) ∧
  probability = 3 / 5 :=
sorry

end rabbit_probability_l114_114772


namespace balls_into_boxes_l114_114397

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114397


namespace annual_salary_is_20_l114_114760

-- Define the conditions
variable (months_worked : ℝ) (total_received : ℝ) (turban_price : ℝ)
variable (S : ℝ)

-- Actual values from the problem
axiom h1 : months_worked = 9 / 12
axiom h2 : total_received = 55
axiom h3 : turban_price = 50

-- Define the statement to prove
theorem annual_salary_is_20 : S = 20 := by
  -- Conditions derived from the problem
  have cash_received := total_received - turban_price
  have fraction_of_salary := months_worked * S
  -- Given the servant worked 9 months and received Rs. 55 including Rs. 50 turban
  have : cash_received = fraction_of_salary := by sorry
  -- Solving the equation 3/4 S = 5 for S
  have : S = 20 := by sorry
  sorry -- Final proof step

end annual_salary_is_20_l114_114760


namespace restaurant_chili_paste_needs_l114_114843

theorem restaurant_chili_paste_needs:
  let large_can_volume := 25
  let small_can_volume := 15
  let large_cans_required := 45
  let total_volume := large_cans_required * large_can_volume
  let small_cans_needed := total_volume / small_can_volume
  small_cans_needed - large_cans_required = 30 :=
by
  sorry

end restaurant_chili_paste_needs_l114_114843


namespace iesha_total_books_l114_114764

theorem iesha_total_books (schoolBooks sportsBooks : ℕ) (h1 : schoolBooks = 19) (h2 : sportsBooks = 39) : schoolBooks + sportsBooks = 58 :=
by
  sorry

end iesha_total_books_l114_114764


namespace tan_minus_405_eq_neg1_l114_114700

theorem tan_minus_405_eq_neg1 :
  let θ := 405
  in  tan (-θ : ℝ) = -1 :=
by
  sorry

end tan_minus_405_eq_neg1_l114_114700


namespace alloy_gold_content_l114_114201

theorem alloy_gold_content (x : ℝ) (w : ℝ) (p0 p1 : ℝ) (h_w : w = 16)
  (h_p0 : p0 = 0.50) (h_p1 : p1 = 0.80) (h_alloy : x = 24) :
  (p0 * w + x) / (w + x) = p1 :=
by sorry

end alloy_gold_content_l114_114201


namespace parabola_focus_l114_114518

theorem parabola_focus :
  ∀ (x y : ℝ), x^2 = 4 * y → (0, 1) = (0, (2 / 2)) :=
by
  intros x y h
  sorry

end parabola_focus_l114_114518


namespace ways_to_put_balls_in_boxes_l114_114421

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114421


namespace white_tshirts_per_pack_l114_114584

def packs_of_white := 3
def packs_of_blue := 2
def blue_in_each_pack := 4
def total_tshirts := 26

theorem white_tshirts_per_pack :
  ∃ W : ℕ, packs_of_white * W + packs_of_blue * blue_in_each_pack = total_tshirts ∧ W = 6 :=
by
  sorry

end white_tshirts_per_pack_l114_114584


namespace find_valid_pairs_l114_114654

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def distinct_two_digit_primes : List (ℕ × ℕ) :=
  [(13, 53), (19, 47), (23, 43), (29, 37)]

def average (p q : ℕ) : ℕ := (p + q) / 2

def number1 (p q : ℕ) : ℕ := 100 * p + q
def number2 (p q : ℕ) : ℕ := 100 * q + p

theorem find_valid_pairs (p q : ℕ)
  (hp : is_prime p) (hq : is_prime q)
  (hpq : p ≠ q)
  (havg : average p q ∣ number1 p q ∧ average p q ∣ number2 p q) :
  (p, q) ∈ distinct_two_digit_primes ∨ (q, p) ∈ distinct_two_digit_primes :=
sorry

end find_valid_pairs_l114_114654


namespace exists_divisible_by_3_on_circle_l114_114510

theorem exists_divisible_by_3_on_circle :
  ∃ a : ℕ → ℕ, (∀ i, a i ≥ 1) ∧
               (∀ i, i < 99 → (a (i + 1) < 99 → (a (i + 1) - a i = 1 ∨ a (i + 1) - a i = 2 ∨ a (i + 1) = 2 * a i))) ∧
               (∃ i, i < 99 ∧ a i % 3 = 0) := 
sorry

end exists_divisible_by_3_on_circle_l114_114510


namespace line_tangent_to_parabola_k_value_l114_114291

theorem line_tangent_to_parabola_k_value :
  ∃ k : ℝ, (∀ x y : ℝ, 4 * x + 7 * y + k = 0 → y ^ 2 = 16 * x) → k = 49 :=
by
  -- definitions of the line and parabola
  let line := (x y : ℝ) → 4 * x + 7 * y + k = 0
  let parabola := (y : ℝ) → y ^ 2 = 16 * (-7*y - k)/4
  -- proof to be filled
  sorry

end line_tangent_to_parabola_k_value_l114_114291


namespace stock_yield_percentage_l114_114062

theorem stock_yield_percentage
  (annual_dividend : ℝ)
  (market_price : ℝ)
  (face_value : ℝ)
  (yield_percentage : ℝ)
  (H1 : annual_dividend = 0.14 * face_value)
  (H2 : market_price = 175)
  (H3 : face_value = 100)
  (H4 : yield_percentage = (annual_dividend / market_price) * 100) :
  yield_percentage = 8 := sorry

end stock_yield_percentage_l114_114062


namespace distinguish_ball_box_ways_l114_114465

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114465


namespace smallest_positive_integer_form_3003_55555_l114_114189

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l114_114189


namespace problem1_problem2_problem3_problem4_problem5_l114_114179

-- Definitions and conditions
variable (a : ℝ) (b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a - 2 * Real.sqrt b > 0)

-- Problem 1: √(a - 2√b) = √m - √n
theorem problem1 (h₁ : a = 5) (h₂ : b = 6) : Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 := sorry

-- Problem 2: √(a + 2√b) = √m + √n
theorem problem2 (h₁ : a = 12) (h₂ : b = 35) : Real.sqrt (12 + 2 * Real.sqrt 35) = Real.sqrt 7 + Real.sqrt 5 := sorry

-- Problem 3: √(a + 6√b) = √m + √n
theorem problem3 (h₁ : a = 9) (h₂ : b = 6) : Real.sqrt (9 + 6 * Real.sqrt 2) = Real.sqrt 6 + Real.sqrt 3 := sorry

-- Problem 4: √(a - 4√b) = √m - √n
theorem problem4 (h₁ : a = 16) (h₂ : b = 60) : Real.sqrt (16 - 4 * Real.sqrt 15) = Real.sqrt 10 - Real.sqrt 6 := sorry

-- Problem 5: √(a - √b) + √(c + √d)
theorem problem5 (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 2) (h₄ : d = 3) 
  : Real.sqrt (3 - Real.sqrt 5) + Real.sqrt (2 + Real.sqrt 3) = (Real.sqrt 10 + Real.sqrt 6) / 2 := sorry

end problem1_problem2_problem3_problem4_problem5_l114_114179


namespace balls_in_boxes_l114_114367

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114367


namespace g_is_even_l114_114132

noncomputable def g (x : ℝ) : ℝ := 5^(x^2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g x = g (-x) :=
by
  sorry

end g_is_even_l114_114132


namespace ball_in_boxes_l114_114339

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114339


namespace number_of_triangles_in_decagon_l114_114267

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l114_114267


namespace frog_eyes_count_l114_114079

def total_frog_eyes (a b c : ℕ) (eyesA eyesB eyesC : ℕ) : ℕ :=
  a * eyesA + b * eyesB + c * eyesC

theorem frog_eyes_count :
  let a := 2
  let b := 1
  let c := 3
  let eyesA := 2
  let eyesB := 3
  let eyesC := 4
  total_frog_eyes a b c eyesA eyesB eyesC = 19 := by
  sorry

end frog_eyes_count_l114_114079


namespace num_triangles_from_decagon_l114_114242

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l114_114242


namespace find_a_l114_114597

theorem find_a (a : ℝ) (h : (2:ℝ)^2 + 2 * a - 3 * a = 0) : a = 4 :=
sorry

end find_a_l114_114597


namespace ball_box_distribution_l114_114463

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114463


namespace Ondra_problems_conditions_l114_114529

-- Define the conditions as provided
variables {a b : ℤ}

-- Define the first condition where the subtraction is equal to the product.
def condition1 : Prop := a + b = a * b

-- Define the second condition involving the relationship with 182.
def condition2 : Prop := a * b * (a + b) = 182

-- The statement to be proved: Ondra's problems (a, b) are (2, 2) and (1, 13) or (13, 1)
theorem Ondra_problems_conditions {a b : ℤ} (h1 : condition1) (h2 : condition2) :
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
sorry

end Ondra_problems_conditions_l114_114529


namespace find_starting_number_l114_114666

theorem find_starting_number (x : ℝ) (h : ((x - 2 + 4) / 1) / 2 * 8 = 77) : x = 17.25 := by
  sorry

end find_starting_number_l114_114666


namespace polygon_edges_l114_114808

theorem polygon_edges :
  ∃ a b : ℕ, a + b = 2014 ∧
              (a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053) ∧
              a ≤ b ∧
              a = 952 :=
by
  sorry

end polygon_edges_l114_114808


namespace part_one_part_two_l114_114068

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 5 then (16 / (9 - x) - 1) else (11 - (2 / 45) * x ^ 2)

theorem part_one (k : ℝ) (h : 1 ≤ k ∧ k ≤ 4) : k * (16 / (9 - 3) - 1) = 4 → k = 12 / 5 :=
by sorry

theorem part_two (y x : ℝ) (h_y : y = 4) :
  (1 ≤ x ∧ x ≤ 5 ∧ 4 * (16 / (9 - x) - 1) ≥ 4) ∨
  (5 < x ∧ x ≤ 15 ∧ 4 * (11 - (2/45) * x ^ 2) ≥ 4) :=
by sorry

end part_one_part_two_l114_114068


namespace tetrahedron_volume_l114_114315

variable {R : ℝ}
variable {S1 S2 S3 S4 : ℝ}
variable {V : ℝ}

theorem tetrahedron_volume (R : ℝ) (S1 S2 S3 S4 V : ℝ) :
  V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l114_114315


namespace balls_into_boxes_l114_114334

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114334


namespace number_of_beautiful_arrangements_l114_114664

noncomputable def count_beautiful_arrangements : ℕ :=
  let total_ways := Nat.choose 10 5
  let invalid_ways := Nat.choose 9 4
  total_ways - invalid_ways

theorem number_of_beautiful_arrangements : count_beautiful_arrangements = 126 := by
  sorry

end number_of_beautiful_arrangements_l114_114664


namespace largest_power_of_2_dividing_n_l114_114098

open Nat

-- Defining given expressions
def n : ℕ := 17^4 - 9^4 + 8 * 17^2

-- The theorem to prove
theorem largest_power_of_2_dividing_n : 2^3 ∣ n ∧ ∀ k, (k > 3 → ¬ 2^k ∣ n) :=
by
  sorry

end largest_power_of_2_dividing_n_l114_114098


namespace smallest_positive_integer_l114_114185

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l114_114185


namespace randy_quiz_goal_l114_114012

def randy_scores : List ℕ := [90, 98, 92, 94]
def randy_next_score : ℕ := 96
def randy_goal_average : ℕ := 94

theorem randy_quiz_goal :
  let total_score := randy_scores.sum
  let required_total_score := 470
  total_score + randy_next_score = required_total_score →
  required_total_score / randy_goal_average = 5 :=
by
  intro h
  sorry

end randy_quiz_goal_l114_114012


namespace milk_cost_correct_l114_114733

-- Definitions of the given conditions
def bagelCost : ℝ := 0.95
def orangeJuiceCost : ℝ := 0.85
def sandwichCost : ℝ := 4.65
def lunchExtraCost : ℝ := 4.0

-- Total cost of breakfast
def breakfastCost : ℝ := bagelCost + orangeJuiceCost

-- Total cost of lunch
def lunchCost : ℝ := breakfastCost + lunchExtraCost

-- Cost of milk
def milkCost : ℝ := lunchCost - sandwichCost

-- Theorem to prove the cost of milk
theorem milk_cost_correct : milkCost = 1.15 :=
by
  sorry

end milk_cost_correct_l114_114733


namespace small_poster_ratio_l114_114570

theorem small_poster_ratio (total_posters : ℕ) (medium_posters large_posters small_posters : ℕ)
  (h1 : total_posters = 50)
  (h2 : medium_posters = 50 / 2)
  (h3 : large_posters = 5)
  (h4 : small_posters = total_posters - medium_posters - large_posters)
  (h5 : total_posters ≠ 0) :
  small_posters = 20 ∧ (small_posters : ℚ) / total_posters = 2 / 5 := 
sorry

end small_poster_ratio_l114_114570


namespace find_ab_l114_114598

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 100) : a * b = -3 :=
by
sorry

end find_ab_l114_114598


namespace blue_lipstick_students_l114_114007

def total_students : ℕ := 200
def students_with_lipstick : ℕ := total_students / 2
def students_with_red_lipstick : ℕ := students_with_lipstick / 4
def students_with_blue_lipstick : ℕ := students_with_red_lipstick / 5

theorem blue_lipstick_students : students_with_blue_lipstick = 5 :=
by
  sorry

end blue_lipstick_students_l114_114007


namespace xy_diff_l114_114121

theorem xy_diff {x y : ℝ} (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  sorry

end xy_diff_l114_114121


namespace polygon_has_twelve_sides_l114_114126

theorem polygon_has_twelve_sides
  (sum_exterior_angles : ℝ)
  (sum_interior_angles : ℝ → ℝ)
  (n : ℝ)
  (h1 : sum_exterior_angles = 360)
  (h2 : ∀ n, sum_interior_angles n = 180 * (n - 2))
  (h3 : ∀ n, sum_interior_angles n = 5 * sum_exterior_angles) :
  n = 12 :=
by
  sorry

end polygon_has_twelve_sides_l114_114126


namespace remainder_three_n_l114_114661

theorem remainder_three_n (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 :=
by
  sorry

end remainder_three_n_l114_114661


namespace carrie_spent_money_l114_114203

variable (cost_per_tshirt : ℝ) (num_tshirts : ℕ)

theorem carrie_spent_money (h1 : cost_per_tshirt = 9.95) (h2 : num_tshirts = 20) :
  cost_per_tshirt * num_tshirts = 199 := by
  sorry

end carrie_spent_money_l114_114203


namespace solve_for_x_l114_114917

theorem solve_for_x (x : ℝ) (h : 3 / (x + 10) = 1 / (2 * x)) : x = 2 :=
sorry

end solve_for_x_l114_114917


namespace books_sold_l114_114511

-- Define the conditions
def initial_books : ℕ := 134
def books_given_away : ℕ := 39
def remaining_books : ℕ := 68

-- Define the intermediate calculation of books left after giving away
def books_after_giving_away : ℕ := initial_books - books_given_away

-- Prove the number of books sold
theorem books_sold (initial_books books_given_away remaining_books : ℕ) (h1 : books_after_giving_away = 95) (h2 : remaining_books = 68) :
  (books_after_giving_away - remaining_books) = 27 :=
by
  sorry

end books_sold_l114_114511


namespace exists_irrationals_floor_neq_l114_114990

-- Define irrationality of a number
def irrational (x : ℝ) : Prop :=
  ¬ ∃ (r : ℚ), x = r

theorem exists_irrationals_floor_neq :
  ∃ (a b : ℝ), irrational a ∧ irrational b ∧ 1 < a ∧ 1 < b ∧ 
  ∀ (m n : ℕ), ⌊a ^ m⌋ ≠ ⌊b ^ n⌋ :=
by
  sorry

end exists_irrationals_floor_neq_l114_114990


namespace least_positive_three_digit_multiple_of_8_l114_114980

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l114_114980


namespace number_of_triangles_in_decagon_l114_114244

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l114_114244


namespace ratatouille_cost_per_quart_l114_114163

theorem ratatouille_cost_per_quart:
  let eggplant_weight := 5.5
  let eggplant_price := 2.20
  let zucchini_weight := 3.8
  let zucchini_price := 1.85
  let tomatoes_weight := 4.6
  let tomatoes_price := 3.75
  let onions_weight := 2.7
  let onions_price := 1.10
  let basil_weight := 1.0
  let basil_price_per_quarter := 2.70
  let bell_peppers_weight := 0.75
  let bell_peppers_price := 3.15
  let yield_quarts := 4.5
  let eggplant_cost := eggplant_weight * eggplant_price
  let zucchini_cost := zucchini_weight * zucchini_price
  let tomatoes_cost := tomatoes_weight * tomatoes_price
  let onions_cost := onions_weight * onions_price
  let basil_cost := basil_weight * (basil_price_per_quarter * 4)
  let bell_peppers_cost := bell_peppers_weight * bell_peppers_price
  let total_cost := eggplant_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost + bell_peppers_cost
  let cost_per_quart := total_cost / yield_quarts
  cost_per_quart = 11.67 :=
by
  sorry

end ratatouille_cost_per_quart_l114_114163


namespace cos_sub_eq_five_over_eight_l114_114742

theorem cos_sub_eq_five_over_eight (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_sub_eq_five_over_eight_l114_114742


namespace masha_guessed_number_l114_114796

theorem masha_guessed_number (a b : ℕ) (h1 : a + b = 2002 ∨ a * b = 2002)
  (h2 : ∀ x y, x + y = 2002 → x ≠ 1001 → y ≠ 1001)
  (h3 : ∀ x y, x * y = 2002 → x ≠ 1001 → y ≠ 1001) :
  b = 1001 :=
by {
  sorry
}

end masha_guessed_number_l114_114796


namespace geometric_sequence_a7_l114_114488

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

-- Given condition
axiom geom_seq_condition : a 4 * a 10 = 9

-- proving the required result
theorem geometric_sequence_a7 (h : is_geometric_sequence a r) : a 7 = 3 ∨ a 7 = -3 :=
by
  sorry

end geometric_sequence_a7_l114_114488


namespace triangle_a_c_sin_A_minus_B_l114_114504

theorem triangle_a_c_sin_A_minus_B (a b c : ℝ) (A B C : ℝ):
  a + c = 6 → b = 2 → Real.cos B = 7/9 →
  a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27 :=
by
  intro h1 h2 h3
  sorry

end triangle_a_c_sin_A_minus_B_l114_114504


namespace power_calculation_l114_114545

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l114_114545


namespace select_numbers_with_sum_713_l114_114810

noncomputable def is_suitable_sum (numbers : List ℤ) : Prop :=
  ∃ subset : List ℤ, subset ⊆ numbers ∧ (subset.sum % 10000 = 713)

theorem select_numbers_with_sum_713 :
  ∀ numbers : List ℤ, 
  numbers.length = 1000 → 
  (∀ n ∈ numbers, n % 2 = 1 ∧ n % 5 ≠ 0) →
  is_suitable_sum numbers :=
sorry

end select_numbers_with_sum_713_l114_114810


namespace fraction_spent_on_DVDs_l114_114134

theorem fraction_spent_on_DVDs (initial_money spent_on_books additional_books_cost remaining_money_spent fraction remaining_money_after_DVDs : ℚ) : 
  initial_money = 320 ∧
  spent_on_books = initial_money / 4 ∧
  additional_books_cost = 10 ∧
  remaining_money_spent = 230 ∧
  remaining_money_after_DVDs = 130 ∧
  remaining_money_spent = initial_money - (spent_on_books + additional_books_cost) ∧
  remaining_money_after_DVDs = remaining_money_spent - (fraction * remaining_money_spent + 8) 
  → fraction = 46 / 115 :=
by
  intros
  sorry

end fraction_spent_on_DVDs_l114_114134


namespace exists_k_l114_114795

theorem exists_k (m n : ℕ) : ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end exists_k_l114_114795


namespace compute_expression_l114_114579

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l114_114579


namespace spending_Mar_Apr_May_l114_114802

-- Define the expenditures at given points
def e_Feb : ℝ := 0.7
def e_Mar : ℝ := 1.2
def e_May : ℝ := 4.4

-- Define the amount spent from March to May
def amount_spent_Mar_Apr_May := e_May - e_Feb

-- The main theorem to prove
theorem spending_Mar_Apr_May : amount_spent_Mar_Apr_May = 3.7 := by
  sorry

end spending_Mar_Apr_May_l114_114802


namespace distinguish_ball_box_ways_l114_114469

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114469


namespace number_of_triangles_in_decagon_l114_114284

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l114_114284


namespace total_letters_in_names_is_33_l114_114139

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l114_114139


namespace area_of_octagon_l114_114538

/--
Given:
- Two congruent squares share the same center O and have sides of length 2.
- The length of segment AB is 7/15.
- Points A, B, C, D, E, F, G, H are points of intersection of the inner rotated square with the outer square, arranged to maintain symmetry.

Prove that the area of octagon ABCDEFGH is 56/15.
-/
theorem area_of_octagon :
  let side_length := 2
  let center := (0, 0)
  let segment_AB := (7 : ℚ) / 15
  let area_octa := (56 : ℚ) / 15
  ∃ (a b c d e f g h : ℚ × ℚ),
    (a.1 = 0 ∧ a.2 = side_length / 2) ∧
    (b.1 = segment_AB / 2 ∧ b.2 = side_length / 2) ∧
    (area_of_octagon = area_octa) := by
      sorry

end area_of_octagon_l114_114538


namespace tasks_completed_correctly_l114_114622

theorem tasks_completed_correctly (x y : ℕ) (h1 : 9 * x - 5 * y = 57) (h2 : x + y ≤ 15) : x = 8 := 
by
  sorry

end tasks_completed_correctly_l114_114622


namespace monotonicity_f_intersection_point_inequality_ln_sum_l114_114902

/-- Part (1): Monotonicity of the function f(x) = 2a ln(x) - x^2 + a -/
theorem monotonicity_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → (a ≤ 0 → Deriv (λ x, 2 * a * log x - x ^ 2 + a) x < 0) ∧ 
  (a > 0 → (forall y: ℝ, 0 < y ∧ y < sqrt a → Deriv (λ y, 2 * a * log y - y ^ 2 + a) y > 0) ∧ 
  (∀ z : ℝ, z > sqrt a → Deriv (λ z, 2 * a * log z - z ^ 2 + a) z < 0))) := sorry

/-- Part (2): Intersection Point x0 is less than the arithmetic mean of x1 and x2 -/
theorem intersection_point (x1 x2 : ℝ) (a : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2)
  (h4 : f x1 = 0) (h5 : f x2 = 0) (hx0 : x0 = (x1 + x2) / 2):
  let x0 := (x1 + x2) / 2 in x0 < (x1 + x2) / 2 := sorry

/-- Part (3): Prove the inequality ln(n+1) < 1/2 + sum (1/(i+1)) for n ∈ ℕ*-/
theorem inequality_ln_sum (n : ℕ) (hn : 0 < n) : 
  log (n + 1) < 1 / 2 + ∑ i in range n, 1 / (i + 2) := sorry

end monotonicity_f_intersection_point_inequality_ln_sum_l114_114902


namespace output_is_three_l114_114048

-- Define the initial values
def initial_a : ℕ := 1
def initial_b : ℕ := 2

-- Define the final value of a after the computation
def final_a : ℕ := initial_a + initial_b

-- The theorem stating that the final value of a is 3
theorem output_is_three : final_a = 3 := by
  sorry

end output_is_three_l114_114048


namespace distinct_balls_boxes_l114_114403

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114403


namespace hourly_wage_difference_l114_114059

theorem hourly_wage_difference (P Q: ℝ) (H_p: ℝ) (H_q: ℝ) (h1: P = 1.5 * Q) (h2: H_q = H_p + 10) (h3: P * H_p = 420) (h4: Q * H_q = 420) : P - Q = 7 := by
  sorry

end hourly_wage_difference_l114_114059


namespace solve_system_of_equations_l114_114954

theorem solve_system_of_equations :
    ∃ x y : ℚ, 4 * x - 3 * y = 2 ∧ 6 * x + 5 * y = 1 ∧ x = 13 / 38 ∧ y = -4 / 19 :=
by
  sorry

end solve_system_of_equations_l114_114954


namespace Bobby_candy_chocolate_sum_l114_114852

/-
  Bobby ate 33 pieces of candy, then ate 4 more, and he also ate 14 pieces of chocolate.
  Prove that the total number of pieces of candy and chocolate he ate altogether is 51.
-/

theorem Bobby_candy_chocolate_sum :
  let initial_candy := 33
  let more_candy := 4
  let chocolate := 14
  let total_candy := initial_candy + more_candy
  total_candy + chocolate = 51 :=
by
  -- The theorem asserts the problem; apologies, the proof is not required here.
  sorry

end Bobby_candy_chocolate_sum_l114_114852


namespace tangent_line_equation_at_point_l114_114711

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem tangent_line_equation_at_point (x y : ℝ) (h : y = curve x) 
    (hx : 2) (hy : 5) (hpt : y = 5 ∧ x = 2) : 7 * x - y - 9 = 0 :=
by
  sorry

end tangent_line_equation_at_point_l114_114711


namespace balls_in_boxes_l114_114366

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114366


namespace tan_neg405_deg_l114_114696

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l114_114696


namespace compute_expression_l114_114582

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l114_114582


namespace put_balls_in_boxes_l114_114443

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114443


namespace total_number_of_letters_l114_114144

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l114_114144


namespace number_of_valid_digits_l114_114996

theorem number_of_valid_digits :
  let N := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let valid_digits := [n | n ∈ N ∧ (640 + n) % 4 = 0] in
  valid_digits.length = 5 :=
by sorry

end number_of_valid_digits_l114_114996


namespace least_positive_three_digit_multiple_of_8_l114_114972

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l114_114972


namespace median_possible_values_l114_114496

theorem median_possible_values (S : Finset ℤ)
  (h : S.card = 10)
  (h_contains : {5, 7, 12, 15, 18, 21} ⊆ S) :
  ∃! n : ℕ, n = 5 :=
by
   sorry

end median_possible_values_l114_114496


namespace num_triangles_from_decagon_l114_114239

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l114_114239


namespace nina_money_proof_l114_114638

def total_money_nina_has (W M : ℝ) : Prop :=
  (10 * W = M) ∧ (14 * (W - 1.75) = M)

theorem nina_money_proof (W M : ℝ) (h : total_money_nina_has W M) : M = 61.25 :=
by 
  sorry

end nina_money_proof_l114_114638


namespace range_of_a_l114_114022

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 - 3 * a * x^2 + (2 * a + 1) * x

noncomputable def f' (a x : ℝ) : ℝ :=
  3 * x^2 - 6 * a * x + (2 * a + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f' a x = 0 ∧ ∀ y : ℝ, f' a y ≠ 0) →
  (a > 1 ∨ a < -1 / 3) :=
sorry

end range_of_a_l114_114022


namespace count_mappings_l114_114634

open Finset

def M : Finset (Fin 3) := {0, 1, 2}
def N : Finset ℤ := {-1, 0, 1}

noncomputable def satisfies_condition (f : Fin 3 → ℤ) : Prop :=
  f 0 + f 1 + f 2 = 0

theorem count_mappings : (card {f : (Fin 3 → ℤ) // ∀ x ∈ M, f x ∈ N ∧ satisfies_condition f} = 7) :=
sorry

end count_mappings_l114_114634


namespace total_number_of_letters_l114_114142

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l114_114142


namespace suit_price_after_discount_l114_114523

-- Define the original price of the suit.
def original_price : ℝ := 150

-- Define the increase rate and the discount rate.
def increase_rate : ℝ := 0.20
def discount_rate : ℝ := 0.20

-- Define the increased price after the 20% increase.
def increased_price : ℝ := original_price * (1 + increase_rate)

-- Define the final price after applying the 20% discount.
def final_price : ℝ := increased_price * (1 - discount_rate)

-- Prove that the final price is $144.
theorem suit_price_after_discount : final_price = 144 := by
  sorry  -- Proof to be completed

end suit_price_after_discount_l114_114523


namespace second_smallest_packs_of_hot_dogs_l114_114093

theorem second_smallest_packs_of_hot_dogs
    (n : ℤ) 
    (h1 : ∃ m : ℤ, 12 * n = 8 * m + 6) :
    ∃ k : ℤ, n = 4 * k + 7 :=
sorry

end second_smallest_packs_of_hot_dogs_l114_114093


namespace monotonic_increasing_iff_monotonic_decreasing_on_interval_l114_114935

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 - a * x - 1

theorem monotonic_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a < f y a) ↔ a ≤ 0 :=
by 
  sorry

theorem monotonic_decreasing_on_interval (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → ∀ y : ℝ, -1 < y ∧ y < 1 → x < y → f y a < f x a) ↔ 3 ≤ a :=
by 
  sorry

end monotonic_increasing_iff_monotonic_decreasing_on_interval_l114_114935


namespace range_of_m_l114_114003

noncomputable def f (a x: ℝ) := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem range_of_m (a m x₁ x₂: ℝ) (h₁: a ∈ Set.Icc (-3) (0)) (h₂: x₁ ∈ Set.Icc (0) (2)) (h₃: x₂ ∈ Set.Icc (0) (2)) : m ∈ Set.Ici (5) → m - a * m^2 ≥ |f a x₁ - f a x₂| :=
sorry

end range_of_m_l114_114003


namespace total_letters_in_names_is_33_l114_114140

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l114_114140


namespace put_balls_in_boxes_l114_114446

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114446


namespace decreasing_interval_f_l114_114650

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (4*x - x^2)

theorem decreasing_interval_f : ∀ x, (2 < x) ∧ (x < 4) → f x < f (2 : ℝ) :=
by
sorry

end decreasing_interval_f_l114_114650


namespace percent_employed_females_in_employed_population_l114_114929

def percent_employed (population: ℝ) : ℝ := 0.64 * population
def percent_employed_males (population: ℝ) : ℝ := 0.50 * population
def percent_employed_females (population: ℝ) : ℝ := percent_employed population - percent_employed_males population

theorem percent_employed_females_in_employed_population (population: ℝ) : 
  (percent_employed_females population / percent_employed population) * 100 = 21.875 :=
by
  sorry

end percent_employed_females_in_employed_population_l114_114929


namespace cards_relationship_l114_114630

-- Definitions from the conditions given in the problem
variables (x y : ℕ)

-- Theorem statement proving the relationship
theorem cards_relationship (h : x + y = 8 * x) : y = 7 * x :=
sorry

end cards_relationship_l114_114630


namespace comprehensive_survey_option_l114_114554

def suitable_for_comprehensive_survey (survey : String) : Prop :=
  survey = "Survey on the components of the first large civil helicopter in China"

theorem comprehensive_survey_option (A B C D : String)
  (hA : A = "Survey on the number of waste batteries discarded in the city every day")
  (hB : B = "Survey on the quality of ice cream in the cold drink market")
  (hC : C = "Survey on the current mental health status of middle school students nationwide")
  (hD : D = "Survey on the components of the first large civil helicopter in China") :
  suitable_for_comprehensive_survey D :=
by
  sorry

end comprehensive_survey_option_l114_114554


namespace decagon_triangle_count_l114_114262

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l114_114262


namespace sum_of_first_5n_l114_114125

theorem sum_of_first_5n (n : ℕ) (h : (3 * n) * (3 * n + 1) / 2 = n * (n + 1) / 2 + 270) : (5 * n) * (5 * n + 1) / 2 = 820 :=
by
  sorry

end sum_of_first_5n_l114_114125


namespace trisha_take_home_pay_l114_114036

def hourly_wage : ℝ := 15
def hours_per_week : ℝ := 40
def weeks_per_year : ℝ := 52
def tax_rate : ℝ := 0.2

def annual_gross_pay : ℝ := hourly_wage * hours_per_week * weeks_per_year
def amount_withheld : ℝ := tax_rate * annual_gross_pay
def annual_take_home_pay : ℝ := annual_gross_pay - amount_withheld

theorem trisha_take_home_pay :
  annual_take_home_pay = 24960 := 
by
  sorry

end trisha_take_home_pay_l114_114036


namespace find_value_of_a_l114_114218

variables (a : ℚ)

-- Definitions based on the conditions
def Brian_has_mar_bles : ℚ := 3 * a
def Caden_original_mar_bles : ℚ := 4 * Brian_has_mar_bles a
def Daryl_original_mar_bles : ℚ := 2 * Caden_original_mar_bles a
def Caden_after_give_10 : ℚ := Caden_original_mar_bles a - 10
def Daryl_after_receive_10 : ℚ := Daryl_original_mar_bles a + 10

-- Together Caden and Daryl now have 190 marbles
def together_mar_bles : ℚ := Caden_after_give_10 a + Daryl_after_receive_10 a

theorem find_value_of_a : together_mar_bles a = 190 → a = 95 / 18 :=
by
  sorry

end find_value_of_a_l114_114218


namespace ways_to_distribute_balls_l114_114351

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114351


namespace cost_of_article_l114_114057

variable (C : ℝ) 
variable (G : ℝ)
variable (H1 : G = 380 - C)
variable (H2 : 1.05 * G = 420 - C)

theorem cost_of_article : C = 420 :=
by
  sorry

end cost_of_article_l114_114057


namespace focus_of_parabola_l114_114717

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l114_114717


namespace factorization_correct_l114_114095

theorem factorization_correct (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 :=
by
  sorry

end factorization_correct_l114_114095


namespace satisfy_conditions_l114_114859

variable (x : ℝ)

theorem satisfy_conditions :
  (3 * x^2 + 4 * x - 9 < 0) ∧ (x ≥ -2) ↔ (-2 ≤ x ∧ x < 1) := by
  sorry

end satisfy_conditions_l114_114859


namespace domain_of_expression_l114_114867

theorem domain_of_expression (x : ℝ) :
  (1 ≤ x ∧ x < 6) ↔ (∃ y : ℝ, y = (x-1) ∧ y = (6-x) ∧ 0 ≤ y) :=
sorry

end domain_of_expression_l114_114867


namespace quadratic_roots_l114_114730

theorem quadratic_roots (a b: ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0)
  (root_condition1 : a * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (root_condition2 : a * (1/3)^2 + b * (1/3) + 2 = 0) 
  : a - b = -10 := 
by {
  sorry
}

end quadratic_roots_l114_114730


namespace select_4_with_both_sexes_from_4_boys_3_girls_l114_114015

theorem select_4_with_both_sexes_from_4_boys_3_girls :
  (nat.choose 4 3 * nat.choose 3 1) + (nat.choose 4 2 * nat.choose 3 2) + (nat.choose 4 1 * nat.choose 3 3) = 34 :=
by
  sorry

end select_4_with_both_sexes_from_4_boys_3_girls_l114_114015


namespace jonathans_and_sisters_total_letters_l114_114138

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l114_114138


namespace magnitude_of_2a_plus_b_l114_114319

open Real

variables (a b : ℝ × ℝ) (angle : ℝ)

-- Conditions
axiom angle_between_a_b (a b : ℝ × ℝ) : angle = π / 3 -- 60 degrees in radians
axiom norm_a_eq_1 (a : ℝ × ℝ) : ‖a‖ = 1
axiom b_eq (b : ℝ × ℝ) : b = (3, 0)

-- Theorem
theorem magnitude_of_2a_plus_b (h1 : angle = π / 3) (h2 : ‖a‖ = 1) (h3 : b = (3, 0)) :
  ‖2 • a + b‖ = sqrt 19 :=
sorry

end magnitude_of_2a_plus_b_l114_114319


namespace focus_of_parabola_l114_114723

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l114_114723


namespace regular_decagon_triangle_count_l114_114256

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l114_114256


namespace larger_number_1655_l114_114801

theorem larger_number_1655 (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
by sorry

end larger_number_1655_l114_114801


namespace find_n_l114_114869

theorem find_n (k : ℤ) : 
  ∃ n : ℤ, (n = 35 * k + 24) ∧ (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) :=
by
  -- Proof goes here
  sorry

end find_n_l114_114869


namespace arc_segment_difference_l114_114776

noncomputable def arc_length_AB (r θ : ℝ) := r * θ

noncomputable def segment_length_AD : ℝ := 2 * Real.tan (Real.pi / 12) -- 15 degrees in radians

theorem arc_segment_difference :
  (|arc_length_AB 1 (Real.pi / 6) - segment_length_AD| = 0.0122) :=
by
  sorry

end arc_segment_difference_l114_114776


namespace triangles_from_decagon_l114_114236

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l114_114236


namespace distinct_balls_boxes_l114_114407

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114407


namespace find_focus_of_parabola_4x2_l114_114719

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l114_114719


namespace y_at_40_l114_114524

def y_at_x (x : ℤ) : ℤ :=
  3 * x + 4

theorem y_at_40 : y_at_x 40 = 124 :=
by {
  sorry
}

end y_at_40_l114_114524


namespace number_of_triangles_in_regular_decagon_l114_114250

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l114_114250


namespace quadratic_two_distinct_real_roots_find_k_l114_114755

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  ∀ k : ℝ, let a := 1, b := 2 * k + 1, c := k^2 + k in
  let Δ := b^2 - 4 * a * c in Δ > 0 :=
by
  intros
  let Δ := (2 * k + 1)^2 - 4 * 1 * (k^2 + k)
  show Δ = 1
  sorry

-- Part 2: Find the value of k if the roots satisfy the given condition
theorem find_k (k x1 x2 : ℝ) : 
  ∀ k : ℝ, (x1 + x2 = x1 * x2 - 1) ∧ x1 + x2 = -(2 * k + 1) ∧ x1 * x2 = k^2 + k -> k = 0 ∨ k = -3 :=
by
  intros k h
  have h_sum := h.2.1
  have h_product := h.2.2
  show x1 + x2 = -(2 * k + 1) and x1 * x2 = k^2 + k = k = 0 ∨ k = -3
  sorry

end quadratic_two_distinct_real_roots_find_k_l114_114755


namespace lattice_points_on_sphere_at_distance_5_with_x_1_l114_114928

theorem lattice_points_on_sphere_at_distance_5_with_x_1 :
  let points := [(1, 0, 4), (1, 0, -4), (1, 4, 0), (1, -4, 0),
                 (1, 2, 4), (1, 2, -4), (1, -2, 4), (1, -2, -4),
                 (1, 4, 2), (1, 4, -2), (1, -4, 2), (1, -4, -2),
                 (1, 2, 2), (1, 2, -2), (1, -2, 2), (1, -2, -2)]
  (hs : ∀ y z, (1, y, z) ∈ points → 1^2 + y^2 + z^2 = 25) →
  24 = points.length :=
sorry

end lattice_points_on_sphere_at_distance_5_with_x_1_l114_114928


namespace original_price_of_car_l114_114135

-- Let P be the original price of the car
variable (P : ℝ)

-- Condition: The car's value is reduced by 30%
-- Condition: The car's current value is $2800, which means 70% of the original price
def car_current_value_reduced (P : ℝ) : Prop :=
  0.70 * P = 2800

-- Theorem: Prove that the original price of the car is $4000
theorem original_price_of_car (P : ℝ) (h : car_current_value_reduced P) : P = 4000 := by
  sorry

end original_price_of_car_l114_114135


namespace num_integers_satisfying_inequality_l114_114906

theorem num_integers_satisfying_inequality : 
  ∃ (xs : Finset ℤ), (∀ x ∈ xs, -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9) ∧ xs.card = 5 := 
by 
  sorry

end num_integers_satisfying_inequality_l114_114906


namespace ways_to_distribute_balls_l114_114417

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114417


namespace ways_to_distribute_balls_l114_114418

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114418


namespace minimum_value_l114_114555

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℝ), (∃ (x : ℝ), x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) ∧ (a^2 + b^2 = 4 / 5)

-- This line states that the minimum possible value of a^2 + b^2, given the condition, is 4/5.
theorem minimum_value (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
  sorry

end minimum_value_l114_114555


namespace sin_four_alpha_l114_114593

theorem sin_four_alpha (α : ℝ) (h1 : Real.sin (2 * α) = -4 / 5) (h2 : -Real.pi / 4 < α ∧ α < Real.pi / 4) :
  Real.sin (4 * α) = -24 / 25 :=
sorry

end sin_four_alpha_l114_114593


namespace find_integer_n_l114_114870

theorem find_integer_n (n : ℤ) : 
  (∃ m : ℤ, n = 35 * m + 24) ↔ (5 ∣ (3 * n - 2) ∧ 7 ∣ (2 * n + 1)) :=
by sorry

end find_integer_n_l114_114870


namespace gus_eggs_l114_114607

theorem gus_eggs : 
  let eggs_breakfast := 2 in
  let eggs_lunch := 3 in
  let eggs_dinner := 1 in
  let total_eggs := eggs_breakfast + eggs_lunch + eggs_dinner in
  total_eggs = 6 :=
by
  sorry

end gus_eggs_l114_114607


namespace shirts_washed_total_l114_114086

theorem shirts_washed_total (short_sleeve_shirts long_sleeve_shirts : Nat) (h1 : short_sleeve_shirts = 4) (h2 : long_sleeve_shirts = 5) : short_sleeve_shirts + long_sleeve_shirts = 9 := by
  sorry

end shirts_washed_total_l114_114086


namespace dice_sum_prob_l114_114769

theorem dice_sum_prob :
  (3 / 6) * (3 / 6) * (2 / 5) * (1 / 6) * 2 = 13 / 216 :=
by sorry

end dice_sum_prob_l114_114769


namespace num_triangles_from_decagon_l114_114243

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l114_114243


namespace roots_reciprocal_l114_114547

theorem roots_reciprocal {a b c x y : ℝ} (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (a * x^2 + b * x + c = 0) ↔ (c * y^2 + b * y + a = 0) := by
sorry

end roots_reciprocal_l114_114547


namespace number_of_triangles_in_regular_decagon_l114_114252

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l114_114252


namespace people_visited_both_l114_114924

theorem people_visited_both (total iceland norway neither both : ℕ) (h_total: total = 100) (h_iceland: iceland = 55) (h_norway: norway = 43) (h_neither: neither = 63)
  (h_both_def: both = iceland + norway - (total - neither)) :
  both = 61 :=
by 
  rw [h_total, h_iceland, h_norway, h_neither] at h_both_def
  simp at h_both_def
  exact h_both_def

end people_visited_both_l114_114924


namespace find_b_for_inf_solutions_l114_114103

theorem find_b_for_inf_solutions (x : ℝ) (b : ℝ) : 5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 :=
by
  intro h
  sorry

end find_b_for_inf_solutions_l114_114103


namespace radius_of_inscribed_circle_in_quarter_circle_l114_114014

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ :=
  R * (Real.sqrt 2 - 1)

theorem radius_of_inscribed_circle_in_quarter_circle 
  (R : ℝ) (hR : R = 6) : inscribed_circle_radius R = 6 * Real.sqrt 2 - 6 :=
by
  rw [inscribed_circle_radius, hR]
  -- Apply the necessary simplifications and manipulations from the given solution steps here
  sorry

end radius_of_inscribed_circle_in_quarter_circle_l114_114014


namespace fourth_vertex_of_parallelogram_l114_114756

structure Point where
  x : ℤ
  y : ℤ

def midPoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def isMidpoint (M P Q : Point) : Prop :=
  M = midPoint P Q

theorem fourth_vertex_of_parallelogram (A B C D : Point)
  (hA : A = {x := -2, y := 1})
  (hB : B = {x := -1, y := 3})
  (hC : C = {x := 3, y := 4})
  (h1 : isMidpoint (midPoint A C) B D ∨
        isMidpoint (midPoint A B) C D ∨
        isMidpoint (midPoint B C) A D) :
  D = {x := 2, y := 2} ∨ D = {x := -6, y := 0} ∨ D = {x := 4, y := 6} := by
  sorry

end fourth_vertex_of_parallelogram_l114_114756


namespace log_base_one_fifth_of_25_l114_114586

theorem log_base_one_fifth_of_25 : log (1/5) 25 = -2 := by
  sorry

end log_base_one_fifth_of_25_l114_114586


namespace num_triangles_from_decagon_l114_114277

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l114_114277


namespace number_of_triangles_in_decagon_l114_114285

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l114_114285


namespace gabby_l114_114884

-- Define variables and conditions
variables (watermelons peaches plums total_fruit : ℕ)
variables (h_watermelons : watermelons = 1)
variables (h_peaches : peaches = watermelons + 12)
variables (h_plums : plums = 3 * peaches)
variables (h_total_fruit : total_fruit = watermelons + peaches + plums)

-- The theorem we aim to prove
theorem gabby's_fruit_count (h_watermelons : watermelons = 1)
                           (h_peaches : peaches = watermelons + 12)
                           (h_plums : plums = 3 * peaches)
                           (h_total_fruit : total_fruit = watermelons + peaches + plums) :
  total_fruit = 53 := by
sorry

end gabby_l114_114884


namespace prove_a_minus_c_l114_114124

-- Define the given conditions as hypotheses
def condition1 (a b d : ℝ) : Prop := (a + d + b + d) / 2 = 80
def condition2 (b c d : ℝ) : Prop := (b + d + c + d) / 2 = 180

-- State the theorem to be proven
theorem prove_a_minus_c (a b c d : ℝ) (h1 : condition1 a b d) (h2 : condition2 b c d) : a - c = -200 :=
by
  sorry

end prove_a_minus_c_l114_114124


namespace distinguish_ball_box_ways_l114_114472

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114472


namespace sum_of_six_angles_l114_114483

theorem sum_of_six_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle3 + angle5 = 180)
  (h2 : angle2 + angle4 + angle6 = 180) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 :=
by
  sorry

end sum_of_six_angles_l114_114483


namespace put_balls_in_boxes_l114_114440

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114440


namespace bags_sold_on_Thursday_l114_114931

theorem bags_sold_on_Thursday 
    (total_bags : ℕ) (sold_Monday : ℕ) (sold_Tuesday : ℕ) (sold_Wednesday : ℕ) (sold_Friday : ℕ) (percent_not_sold : ℕ) :
    total_bags = 600 →
    sold_Monday = 25 →
    sold_Tuesday = 70 →
    sold_Wednesday = 100 →
    sold_Friday = 145 →
    percent_not_sold = 25 →
    ∃ (sold_Thursday : ℕ), sold_Thursday = 110 :=
by
  sorry

end bags_sold_on_Thursday_l114_114931


namespace find_ratio_l114_114088

open Nat

def sequence_def (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 →
    (a ((n + 2)) / a ((n + 1))) - (a ((n + 1)) / a n) = d

def geometric_difference_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3 ∧ sequence_def a 2

theorem find_ratio (a : ℕ → ℕ) (h : geometric_difference_sequence a) :
  a 12 / a 10 = 399 := sorry

end find_ratio_l114_114088


namespace sequence_solution_l114_114313

-- Defining the sequence and the condition
def sequence_condition (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 1

-- Defining the sequence formula we need to prove
def sequence_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 ^ (n - 1)

theorem sequence_solution (a S : ℕ → ℝ) (h : sequence_condition a S) :
  sequence_formula a :=
by 
  sorry

end sequence_solution_l114_114313


namespace inequality_solution_l114_114851

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 > (3 * x - 2) / 2 - 1 → x < 2 :=
by
  intro h
  sorry

end inequality_solution_l114_114851


namespace trapezoid_perimeter_l114_114984

noncomputable def perimeter_of_trapezoid (AB CD BC AD AP DQ : ℕ) : ℕ :=
  AB + BC + CD + AD

theorem trapezoid_perimeter (AB CD BC AP DQ : ℕ) (hBC : BC = 50) (hAP : AP = 18) (hDQ : DQ = 7) :
  perimeter_of_trapezoid AB CD BC (AP + BC + DQ) AP DQ = 180 :=
by 
  unfold perimeter_of_trapezoid
  rw [hBC, hAP, hDQ]
  -- sorry to skip the proof
  sorry

end trapezoid_perimeter_l114_114984


namespace rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l114_114199

theorem rectangles_on_8x8_chessboard : 
  (Nat.choose 9 2) * (Nat.choose 9 2) = 1296 := by
  sorry

theorem rectangles_on_nxn_chessboard (n : ℕ) : 
  (Nat.choose (n + 1) 2) * (Nat.choose (n + 1) 2) = (n * (n + 1) / 2) * (n * (n + 1) / 2) := by 
  sorry

end rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l114_114199


namespace distinct_balls_boxes_l114_114375

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114375


namespace triangles_from_decagon_l114_114237

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l114_114237


namespace total_points_team_l114_114482

def T : ℕ := 4
def J : ℕ := 2 * T + 6
def S : ℕ := J / 2
def R : ℕ := T + J - 3
def A : ℕ := S + R + 4

theorem total_points_team : T + J + S + R + A = 66 := by
  sorry

end total_points_team_l114_114482


namespace ball_in_boxes_l114_114342

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114342


namespace num_passenger_cars_l114_114069

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end num_passenger_cars_l114_114069


namespace inequality_proof_l114_114119

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c :=
sorry

end inequality_proof_l114_114119


namespace beta_value_l114_114120

variable {α β : Real}
open Real

theorem beta_value :
  cos α = 1 / 7 ∧ cos (α + β) = -11 / 14 ∧ 0 < α ∧ α < π / 2 ∧ π / 2 < α + β ∧ α + β < π → 
  β = π / 3 := 
by
  -- Proof would go here
  sorry

end beta_value_l114_114120


namespace distinguish_ball_box_ways_l114_114471

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114471


namespace minimum_value_of_f_l114_114619

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∃ a > 2, (∀ x > 2, f x ≥ f a) ∧ a = 3 := by
sorry

end minimum_value_of_f_l114_114619


namespace difference_is_divisible_by_p_l114_114166

-- Lean 4 statement equivalent to the math proof problem
theorem difference_is_divisible_by_p
  (a : ℕ → ℕ) (p : ℕ) (d : ℕ)
  (h_prime : Nat.Prime p)
  (h_prog : ∀ i j: ℕ, 1 ≤ i ∧ i ≤ p ∧ 1 ≤ j ∧ j ≤ p ∧ i < j → a j = a (i + 1) + (j - 1) * d)
  (h_a_gt_p : a 1 > p)
  (h_arith_prog_primes : ∀ i: ℕ, 1 ≤ i ∧ i ≤ p → Nat.Prime (a i)) :
  d % p = 0 := sorry

end difference_is_divisible_by_p_l114_114166


namespace percentage_of_first_to_second_l114_114558

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) :
  first = 0.06 * X →
  second = 0.30 * X →
  (first / second) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_of_first_to_second_l114_114558


namespace alice_weekly_walk_distance_l114_114864

theorem alice_weekly_walk_distance :
  let miles_to_school_per_day := 10
  let miles_home_per_day := 12
  let days_per_week := 5
  let weekly_total_miles := (miles_to_school_per_day * days_per_week) + (miles_home_per_day * days_per_week)
  weekly_total_miles = 110 :=
by
  sorry

end alice_weekly_walk_distance_l114_114864


namespace triangles_from_decagon_l114_114234

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l114_114234


namespace triangles_from_decagon_l114_114270

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l114_114270


namespace projection_of_vector_l114_114874

open Real EuclideanSpace

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b

theorem projection_of_vector : 
  vector_projection (6, -3) (3, 0) = (6, 0) := 
by 
  sorry

end projection_of_vector_l114_114874


namespace factor_expression_l114_114223

theorem factor_expression (y : ℝ) : 
  (16 * y ^ 6 + 36 * y ^ 4 - 9) - (4 * y ^ 6 - 6 * y ^ 4 - 9) = 6 * y ^ 4 * (2 * y ^ 2 + 7) := 
by sorry

end factor_expression_l114_114223


namespace lcm_of_9_12_18_l114_114818

-- Let's declare the numbers involved
def num1 : ℕ := 9
def num2 : ℕ := 12
def num3 : ℕ := 18

-- Define what it means for a number to be the LCM of num1, num2, and num3
def is_lcm (a b c l : ℕ) : Prop :=
  l % a = 0 ∧ l % b = 0 ∧ l % c = 0 ∧
  ∀ m, (m % a = 0 ∧ m % b = 0 ∧ m % c = 0) → l ≤ m

-- Now state the theorem
theorem lcm_of_9_12_18 : is_lcm num1 num2 num3 36 :=
by
  sorry

end lcm_of_9_12_18_l114_114818


namespace num_ways_to_distribute_balls_into_boxes_l114_114450

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114450


namespace monotonicity_f_on_interval_l114_114522

def f (x : ℝ) : ℝ := |x + 2|

theorem monotonicity_f_on_interval :
  ∀ x1 x2 : ℝ, x1 < x2 → x1 < -4 → x2 < -4 → f x1 ≥ f x2 :=
by
  sorry

end monotonicity_f_on_interval_l114_114522


namespace total_hours_uploaded_l114_114574

def hours_June_1_to_10 : ℝ := 5 * 2 * 10
def hours_June_11_to_20 : ℝ := 10 * 1 * 10
def hours_June_21_to_25 : ℝ := 7 * 3 * 5
def hours_June_26_to_30 : ℝ := 15 * 0.5 * 5

def total_video_hours : ℝ :=
  hours_June_1_to_10 + hours_June_11_to_20 + hours_June_21_to_25 + hours_June_26_to_30

theorem total_hours_uploaded :
  total_video_hours = 342.5 :=
by
  sorry

end total_hours_uploaded_l114_114574


namespace die_top_face_after_path_l114_114204

def opposite_face (n : ℕ) : ℕ :=
  7 - n

def roll_die (start : ℕ) (sequence : List String) : ℕ :=
  sequence.foldl
    (λ top movement =>
      match movement with
      | "left" => opposite_face (7 - top) -- simplified assumption for movements
      | "forward" => opposite_face (top - 1)
      | "right" => opposite_face (7 - top + 1)
      | "back" => opposite_face (top + 1)
      | _ => top) start

theorem die_top_face_after_path : roll_die 3 ["left", "forward", "right", "back", "forward", "back"] = 4 :=
  by
  sorry

end die_top_face_after_path_l114_114204


namespace compute_expression_l114_114577

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l114_114577


namespace youngsville_population_l114_114196

def initial_population : ℕ := 684
def increase_rate : ℝ := 0.25
def decrease_rate : ℝ := 0.40

theorem youngsville_population : 
  let increased_population := initial_population + ⌊increase_rate * ↑initial_population⌋
  let decreased_population := increased_population - ⌊decrease_rate * increased_population⌋
  decreased_population = 513 :=
by
  sorry

end youngsville_population_l114_114196


namespace ways_to_distribute_balls_l114_114364

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114364


namespace fraction_simplification_l114_114703

theorem fraction_simplification:
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 :=
by {
  -- Proof goes here
  sorry
}

end fraction_simplification_l114_114703


namespace Ondra_problems_conditions_l114_114530

-- Define the conditions as provided
variables {a b : ℤ}

-- Define the first condition where the subtraction is equal to the product.
def condition1 : Prop := a + b = a * b

-- Define the second condition involving the relationship with 182.
def condition2 : Prop := a * b * (a + b) = 182

-- The statement to be proved: Ondra's problems (a, b) are (2, 2) and (1, 13) or (13, 1)
theorem Ondra_problems_conditions {a b : ℤ} (h1 : condition1) (h2 : condition2) :
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
sorry

end Ondra_problems_conditions_l114_114530


namespace value_of_abcg_defh_l114_114988

theorem value_of_abcg_defh
  (a b c d e f g h: ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6)
  (h6 : f / g = 5 / 2)
  (h7 : g / h = 3 / 4) :
  abcg / defh = 5 / 48 :=
by
  sorry

end value_of_abcg_defh_l114_114988


namespace fixed_point_exists_l114_114050

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y : ℝ, (x = 2 ∧ y = -2 ∧ (ax - 5 = y)) :=
by
  sorry

end fixed_point_exists_l114_114050


namespace remainder_of_sum_mod_13_l114_114548

theorem remainder_of_sum_mod_13 :
  ∀ (D : ℕ) (k1 k2 : ℕ),
    D = 13 →
    (242 = k1 * D + 8) →
    (698 = k2 * D + 9) →
    (242 + 698) % D = 4 :=
by
  intros D k1 k2 hD h242 h698
  sorry

end remainder_of_sum_mod_13_l114_114548


namespace smallest_positive_integer_linear_combination_l114_114184

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l114_114184


namespace car_speed_l114_114197

theorem car_speed (v : ℝ) (hv : 2 + (1 / v) * 3600 = (1 / 90) * 3600) :
  v = 600 / 7 :=
sorry

end car_speed_l114_114197


namespace balls_into_boxes_l114_114398

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114398


namespace ball_box_distribution_l114_114457

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114457


namespace lillian_candies_addition_l114_114505

noncomputable def lillian_initial_candies : ℕ := 88
noncomputable def lillian_father_candies : ℕ := 5
noncomputable def lillian_total_candies : ℕ := 93

theorem lillian_candies_addition : lillian_initial_candies + lillian_father_candies = lillian_total_candies := by
  sorry

end lillian_candies_addition_l114_114505


namespace range_of_a_l114_114754

noncomputable def f (x : ℝ) := x * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) → a ≤ 5 + Real.log 2 :=
by
  sorry

end range_of_a_l114_114754


namespace inequality_holds_for_all_x_in_interval_l114_114298

theorem inequality_holds_for_all_x_in_interval (a b : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^2 + a * x + b| ≤ 1 / 8) ↔ (a = -1 ∧ b = 1 / 8) :=
sorry

end inequality_holds_for_all_x_in_interval_l114_114298


namespace balls_in_boxes_l114_114372

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114372


namespace contrapositive_l114_114167

-- Definitions based on the conditions
def original_proposition (a b : ℝ) : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- The theorem to prove the contrapositive
theorem contrapositive (a b : ℝ) : original_proposition a b ↔ (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) :=
sorry

end contrapositive_l114_114167


namespace trucks_needed_for_coal_transport_l114_114565

def number_of_trucks (total_coal : ℕ) (capacity_per_truck : ℕ) (x : ℕ) : Prop :=
  capacity_per_truck * x = total_coal

theorem trucks_needed_for_coal_transport :
  number_of_trucks 47500 2500 19 :=
by
  sorry

end trucks_needed_for_coal_transport_l114_114565


namespace perpendicular_vectors_l114_114606

theorem perpendicular_vectors (k : ℝ) (a b : ℝ × ℝ) 
  (ha : a = (0, 2)) 
  (hb : b = (Real.sqrt 3, 1)) 
  (h : (a.1 - k * b.1) * (k * a.1 + b.1) + (a.2 - k * b.2) * (k * a.2 + b.2) = 0) :
  k = -1 ∨ k = 1 :=
sorry

end perpendicular_vectors_l114_114606


namespace find_coin_flips_l114_114562

theorem find_coin_flips (n : ℕ) (h : (Nat.choose n 2 : ℚ) / 2^n = 7 / 32) : n = 7 :=
by 
  sorry

end find_coin_flips_l114_114562


namespace balls_in_boxes_l114_114370

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114370


namespace non_congruent_triangles_perimeter_12_l114_114908

theorem non_congruent_triangles_perimeter_12 :
  ∃ S : finset (ℕ × ℕ × ℕ), S.card = 5 ∧ ∀ (abc ∈ S), 
  let (a, b, c) := abc in 
    a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    ∀ (abc' ∈ S), abc' ≠ abc → abc ≠ (λ t, (t.2.2, t.2.1, t.1)) abc' :=
by sorry

end non_congruent_triangles_perimeter_12_l114_114908


namespace remainder_of_x_divided_by_30_l114_114018

theorem remainder_of_x_divided_by_30:
  ∀ x : ℤ,
    (4 + x ≡ 9 [ZMOD 8]) ∧ 
    (6 + x ≡ 8 [ZMOD 27]) ∧ 
    (8 + x ≡ 49 [ZMOD 125]) ->
    (x ≡ 17 [ZMOD 30]) :=
by
  intros x h
  sorry

end remainder_of_x_divided_by_30_l114_114018


namespace share_expenses_l114_114329

theorem share_expenses (h l : ℕ) : 
  let henry_paid := 120
  let linda_paid := 150
  let jack_paid := 210
  let total_paid := henry_paid + linda_paid + jack_paid
  let each_should_pay := total_paid / 3
  let henry_owes := each_should_pay - henry_paid
  let linda_owes := each_should_pay - linda_paid
  (h = henry_owes) → 
  (l = linda_owes) → 
  h - l = 30 := by
  sorry

end share_expenses_l114_114329


namespace problem_statement_l114_114667

structure Pricing :=
  (price_per_unit_1 : ℕ) (threshold_1 : ℕ)
  (price_per_unit_2 : ℕ) (threshold_2 : ℕ)
  (price_per_unit_3 : ℕ)

def cost (units : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if units ≤ t1 then units * p1
  else if units ≤ t2 then t1 * p1 + (units - t1) * p2
  else t1 * p1 + (t2 - t1) * p2 + (units - t2) * p3 

def units_given_cost (c : ℕ) (pricing : Pricing) : ℕ :=
  let t1 := pricing.threshold_1
  let t2 := pricing.threshold_2
  let p1 := pricing.price_per_unit_1
  let p2 := pricing.price_per_unit_2
  let p3 := pricing.price_per_unit_3
  if c ≤ t1 * p1 then c / p1
  else if c ≤ t1 * p1 + (t2 - t1) * p2 then t1 + (c - t1 * p1) / p2
  else t2 + (c - t1 * p1 - (t2 - t1) * p2) / p3

def double_eleven_case (total_units total_cost : ℕ) (x_units : ℕ) (pricing : Pricing) : ℕ :=
  let y_units := total_units - x_units
  let case1_cost := cost x_units pricing + cost y_units pricing
  if case1_cost = total_cost then (x_units, y_units).fst
  else sorry

theorem problem_statement (pricing : Pricing):
  (cost 120 pricing = 420) ∧ 
  (cost 260 pricing = 868) ∧
  (units_given_cost 740 pricing = 220) ∧
  (double_eleven_case 400 1349 290 pricing = 290)
  := sorry

end problem_statement_l114_114667


namespace minimum_value_l114_114497

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 3)

theorem minimum_value : 
  (1 / (3 * a + 5 * b)) + (1 / (3 * b + 5 * c)) + (1 / (3 * c + 5 * a)) ≥ 9 / 8 :=
by
  sorry

end minimum_value_l114_114497


namespace annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l114_114779

-- Define principal amounts for Paul, Emma and Harry
def principalPaul : ℚ := 5000
def principalEmma : ℚ := 3000
def principalHarry : ℚ := 7000

-- Define time periods for Paul, Emma and Harry
def timePaul : ℚ := 2
def timeEmma : ℚ := 4
def timeHarry : ℚ := 3

-- Define interests received from Paul, Emma and Harry
def interestPaul : ℚ := 2200
def interestEmma : ℚ := 3400
def interestHarry : ℚ := 3900

-- Define the simple interest formula 
def simpleInterest (P : ℚ) (R : ℚ) (T : ℚ) : ℚ := P * R * T

-- Prove the annual interest rates for each loan 
theorem annual_interest_rate_Paul : 
  ∃ (R : ℚ), simpleInterest principalPaul R timePaul = interestPaul ∧ R = 0.22 := 
by
  sorry

theorem annual_interest_rate_Emma : 
  ∃ (R : ℚ), simpleInterest principalEmma R timeEmma = interestEmma ∧ R = 0.2833 := 
by
  sorry

theorem annual_interest_rate_Harry : 
  ∃ (R : ℚ), simpleInterest principalHarry R timeHarry = interestHarry ∧ R = 0.1857 := 
by
  sorry

end annual_interest_rate_Paul_annual_interest_rate_Emma_annual_interest_rate_Harry_l114_114779


namespace infinite_indices_exist_l114_114784

theorem infinite_indices_exist (a : ℕ → ℕ) (h_seq : ∀ n, a n < a (n + 1)) :
  ∃ᶠ m in ⊤, ∃ x y h k : ℕ, 0 < h ∧ h < k ∧ k < m ∧ a m = x * a h + y * a k :=
by sorry

end infinite_indices_exist_l114_114784


namespace quadratic_real_roots_iff_l114_114768

theorem quadratic_real_roots_iff (k : ℝ) : 
  (∃ x : ℝ, (k-1) * x^2 + 3 * x - 1 = 0) ↔ k ≥ -5 / 4 ∧ k ≠ 1 := sorry

end quadratic_real_roots_iff_l114_114768


namespace ways_to_place_balls_in_boxes_l114_114384

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114384


namespace trigonometric_inequality_proof_l114_114193

theorem trigonometric_inequality_proof : 
  ∀ (sin cos : ℝ → ℝ), 
  (∀ θ, 0 ≤ θ ∧ θ ≤ π/2 → sin θ = cos (π/2 - θ)) → 
  sin (π * 11 / 180) < sin (π * 12 / 180) ∧ sin (π * 12 / 180) < sin (π * 80 / 180) :=
by 
  intros sin cos identity
  sorry

end trigonometric_inequality_proof_l114_114193


namespace smallest_altitude_leq_three_l114_114026

theorem smallest_altitude_leq_three (a b c : ℝ) (r : ℝ) 
  (ha : a = max a (max b c)) 
  (r_eq : r = 1) 
  (area_eq : ∀ (S : ℝ), S = (a + b + c) / 2 ∧ S = a * h / 2) :
  ∃ h : ℝ, h ≤ 3 :=
by
  sorry

end smallest_altitude_leq_three_l114_114026


namespace exists_k_lt_ak_by_2001_fac_l114_114785

theorem exists_k_lt_ak_by_2001_fac (a : ℕ → ℝ) (H0 : a 0 = 1)
(Hn : ∀ n : ℕ, n > 0 → a n = a (⌊(7 * n / 9)⌋₊) + a (⌊(n / 9)⌋₊)) :
  ∃ k : ℕ, k > 0 ∧ a k < k / ↑(Nat.factorial 2001) := by
  sorry

end exists_k_lt_ak_by_2001_fac_l114_114785


namespace max_length_of_each_piece_l114_114506

theorem max_length_of_each_piece (a b c d : ℕ) (h1 : a = 48) (h2 : b = 72) (h3 : c = 108) (h4 : d = 120) : Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 12 := by
  sorry

end max_length_of_each_piece_l114_114506


namespace sum_of_natural_numbers_l114_114023

noncomputable def number_of_ways (n : ℕ) : ℕ :=
  2^(n-1)

theorem sum_of_natural_numbers (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, k = number_of_ways n :=
by
  use 2^(n-1)
  sorry

end sum_of_natural_numbers_l114_114023


namespace gabby_fruit_total_l114_114886

-- Definitions based on conditions
def watermelon : ℕ := 1
def peaches : ℕ := watermelon + 12
def plums : ℕ := peaches * 3
def total_fruit : ℕ := watermelon + peaches + plums

-- Proof statement
theorem gabby_fruit_total : total_fruit = 53 := 
by {
  sorry
}

end gabby_fruit_total_l114_114886


namespace sqrt_of_expression_l114_114824

theorem sqrt_of_expression (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 :=
by
  rw [h]
  simp
  sorry

end sqrt_of_expression_l114_114824


namespace equivalent_fractions_l114_114939

variable {x y a c : ℝ}

theorem equivalent_fractions (h_nonzero_c : c ≠ 0) (h_transform : x = (a / c) * y) :
  (x + a) / (y + c) = a / c :=
by
  sorry

end equivalent_fractions_l114_114939


namespace fraction_of_emilys_coins_l114_114294

theorem fraction_of_emilys_coins {total_states : ℕ} (h1 : total_states = 30)
    {states_from_1790_to_1799 : ℕ} (h2 : states_from_1790_to_1799 = 9) :
    (states_from_1790_to_1799 / total_states : ℚ) = 3 / 10 := by
  sorry

end fraction_of_emilys_coins_l114_114294


namespace valid_root_l114_114799

theorem valid_root:
  ∃ x : ℚ, 
    (3 * x^2 + 5) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 ∧ x = 2 / 3 := 
by
  sorry

end valid_root_l114_114799


namespace parabola_focus_l114_114726

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l114_114726


namespace polygon_properties_l114_114170

theorem polygon_properties
    (each_exterior_angle : ℝ)
    (h1 : each_exterior_angle = 24) :
    ∃ n : ℕ, n = 15 ∧ (180 * (n - 2) = 2340) :=
  by
    sorry

end polygon_properties_l114_114170


namespace irreducible_fraction_for_any_n_l114_114164

theorem irreducible_fraction_for_any_n (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := 
by {
  sorry
}

end irreducible_fraction_for_any_n_l114_114164


namespace least_pebbles_2021_l114_114011

noncomputable def least_pebbles (n : ℕ) : ℕ :=
  n + n / 2

theorem least_pebbles_2021 :
  least_pebbles 2021 = 3031 :=
by
  sorry

end least_pebbles_2021_l114_114011


namespace ball_in_boxes_l114_114343

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114343


namespace number_of_triangles_in_decagon_l114_114225

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l114_114225


namespace decagon_triangle_count_l114_114261

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l114_114261


namespace num_triangles_from_decagon_l114_114274

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l114_114274


namespace general_formula_l114_114927

def sequence_a (n : ℕ) : ℕ :=
by sorry

def partial_sum (n : ℕ) : ℕ :=
by sorry

axiom base_case : partial_sum 1 = 5

axiom recurrence_relation (n : ℕ) (h : 2 ≤ n) : partial_sum (n - 1) = sequence_a n

theorem general_formula (n : ℕ) : partial_sum n = 5 * 2^(n-1) :=
by
-- Proof will be provided here
sorry

end general_formula_l114_114927


namespace ways_to_distribute_balls_l114_114416

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114416


namespace survey_participants_l114_114485

-- Total percentage for option A and option B in bytes
def percent_A : ℝ := 0.50
def percent_B : ℝ := 0.30

-- Number of participants who chose option A
def participants_A : ℕ := 150

-- Target number of participants who chose option B (to be proved)
def participants_B : ℕ := 90

-- The theorem to prove the number of participants who chose option B
theorem survey_participants :
  (participants_B : ℝ) = participants_A * (percent_B / percent_A) :=
by
  sorry

end survey_participants_l114_114485


namespace distinct_balls_boxes_l114_114376

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114376


namespace bread_left_in_pond_l114_114175

theorem bread_left_in_pond (total_bread : ℕ) 
                           (half_bread_duck : ℕ)
                           (second_duck_bread : ℕ)
                           (third_duck_bread : ℕ)
                           (total_bread_thrown : total_bread = 100)
                           (half_duck_eats : half_bread_duck = total_bread / 2)
                           (second_duck_eats : second_duck_bread = 13)
                           (third_duck_eats : third_duck_bread = 7) :
                           total_bread - (half_bread_duck + second_duck_bread + third_duck_bread) = 30 :=
    by
    sorry

end bread_left_in_pond_l114_114175


namespace total_letters_in_names_is_33_l114_114141

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end total_letters_in_names_is_33_l114_114141


namespace sophomores_in_sample_l114_114210

-- Define the number of freshmen, sophomores, and juniors
def freshmen : ℕ := 400
def sophomores : ℕ := 600
def juniors : ℕ := 500

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the total number of students in the sample
def sample_size : ℕ := 100

-- Define the expected number of sophomores in the sample
def expected_sophomores : ℕ := (sample_size * sophomores) / total_students

-- Statement of the problem we want to prove
theorem sophomores_in_sample : expected_sophomores = 40 := by
  sorry

end sophomores_in_sample_l114_114210


namespace smallest_integer_n_satisfying_inequality_l114_114985

theorem smallest_integer_n_satisfying_inequality :
  ∃ n : ℤ, n^2 - 13 * n + 36 ≤ 0 ∧ (∀ m : ℤ, m^2 - 13 * m + 36 ≤ 0 → m ≥ n) ∧ n = 4 := 
by
  sorry

end smallest_integer_n_satisfying_inequality_l114_114985


namespace ways_to_put_balls_in_boxes_l114_114428

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114428


namespace degree_measure_of_subtracted_angle_l114_114817

def angle := 30

theorem degree_measure_of_subtracted_angle :
  let supplement := 180 - angle
  let complement_of_supplement := 90 - supplement
  let twice_complement := 2 * (90 - angle)
  twice_complement - complement_of_supplement = 180 :=
by
  sorry

end degree_measure_of_subtracted_angle_l114_114817


namespace triangles_from_decagon_l114_114238

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l114_114238


namespace balls_in_boxes_l114_114371

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114371


namespace geom_seq_necessary_not_sufficient_l114_114933

theorem geom_seq_necessary_not_sufficient (a1 : ℝ) (q : ℝ) (h1 : 0 < a1) :
  (∀ n : ℕ, n > 0 → a1 * q^(2*n - 1) + a1 * q^(2*n) < 0) ↔ q < 0 :=
sorry

end geom_seq_necessary_not_sufficient_l114_114933


namespace total_current_ages_l114_114809

theorem total_current_ages (T : ℕ) : (T - 12 = 54) → T = 66 :=
by
  sorry

end total_current_ages_l114_114809


namespace ball_box_distribution_l114_114461

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114461


namespace problem_solution_l114_114740

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem problem_solution (f : ℝ → ℝ)
  (H1 : even_function f)
  (H2 : ∀ x, f (x + 4) = -f x)
  (H3 : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 4 → f y < f x) :
  f 13 < f 10 ∧ f 10 < f 15 :=
  by
    sorry

end problem_solution_l114_114740


namespace find_weight_difference_l114_114646

variables (W_A W_B W_C W_D W_E : ℝ)

-- Definitions of the conditions
def average_weight_abc := (W_A + W_B + W_C) / 3 = 84
def average_weight_abcd := (W_A + W_B + W_C + W_D) / 4 = 80
def average_weight_bcde := (W_B + W_C + W_D + W_E) / 4 = 79
def weight_a := W_A = 77

-- The theorem statement
theorem find_weight_difference (h1 : average_weight_abc W_A W_B W_C)
                               (h2 : average_weight_abcd W_A W_B W_C W_D)
                               (h3 : average_weight_bcde W_B W_C W_D W_E)
                               (h4 : weight_a W_A) :
  W_E - W_D = 5 :=
sorry

end find_weight_difference_l114_114646


namespace find_N_l114_114941

theorem find_N (N : ℕ) (h_pos : N > 0) (h_small_factors : 1 + 3 = 4) 
  (h_large_factors : N + N / 3 = 204) : N = 153 :=
  by sorry

end find_N_l114_114941


namespace number_of_connections_l114_114104

theorem number_of_connections (n : ℕ) (d : ℕ) (h₀ : n = 40) (h₁ : d = 4) : 
  (n * d) / 2 = 80 :=
by
  sorry

end number_of_connections_l114_114104


namespace Lyle_percentage_of_chips_l114_114790

theorem Lyle_percentage_of_chips (total_chips : ℕ) (ratio_Ian_Lyle : ℕ × ℕ) (h_total_chips : total_chips = 100) (h_ratio : ratio_Ian_Lyle = (4, 6)) :
  let total_parts := ratio_Ian_Lyle.1 + ratio_Ian_Lyle.2 in
  let chips_per_part := total_chips / total_parts in
  let Lyle_chips := ratio_Ian_Lyle.2 * chips_per_part in
  let percentage_Lyle := (Lyle_chips * 100) / total_chips in
  percentage_Lyle = 60 :=
by
  intros
  sorry

end Lyle_percentage_of_chips_l114_114790


namespace balls_into_boxes_l114_114399

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114399


namespace compute_expression_l114_114578

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l114_114578


namespace least_three_digit_multiple_of_8_l114_114978

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l114_114978


namespace prove_smallest_geometric_third_term_value_l114_114844

noncomputable def smallest_value_geometric_third_term : ℝ :=
  let d_1 := -5 + 10 * Real.sqrt 2
  let d_2 := -5 - 10 * Real.sqrt 2
  let g3_1 := 39 + 2 * d_1
  let g3_2 := 39 + 2 * d_2
  min g3_1 g3_2

theorem prove_smallest_geometric_third_term_value :
  smallest_value_geometric_third_term = 29 - 20 * Real.sqrt 2 := by sorry

end prove_smallest_geometric_third_term_value_l114_114844


namespace initial_oranges_per_tree_l114_114861

theorem initial_oranges_per_tree (x : ℕ) (h1 : 8 * (5 * x - 2 * x) / 5 = 960) : x = 200 :=
sorry

end initial_oranges_per_tree_l114_114861


namespace mother_younger_than_father_l114_114923

variable (total_age : ℕ) (father_age : ℕ) (brother_age : ℕ) (sister_age : ℕ) (kaydence_age : ℕ) (mother_age : ℕ)

noncomputable def family_data : Prop :=
  total_age = 200 ∧
  father_age = 60 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  kaydence_age = 12 ∧
  mother_age = total_age - (father_age + brother_age + sister_age + kaydence_age)

theorem mother_younger_than_father :
  family_data total_age father_age brother_age sister_age kaydence_age mother_age →
  father_age - mother_age = 2 :=
sorry

end mother_younger_than_father_l114_114923


namespace principal_amount_l114_114820

theorem principal_amount (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ)
  (h1 : SI = (P * R * T) / 100)
  (h2 : SI = 640)
  (h3 : R = 8)
  (h4 : T = 2) :
  P = 4000 :=
sorry

end principal_amount_l114_114820


namespace greatest_monthly_drop_in_March_l114_114177

noncomputable def jan_price_change : ℝ := -3.00
noncomputable def feb_price_change : ℝ := 1.50
noncomputable def mar_price_change : ℝ := -4.50
noncomputable def apr_price_change : ℝ := 2.00
noncomputable def may_price_change : ℝ := -1.00
noncomputable def jun_price_change : ℝ := 0.50

theorem greatest_monthly_drop_in_March :
  mar_price_change < jan_price_change ∧
  mar_price_change < feb_price_change ∧
  mar_price_change < apr_price_change ∧
  mar_price_change < may_price_change ∧
  mar_price_change < jun_price_change :=
by {
  sorry
}

end greatest_monthly_drop_in_March_l114_114177


namespace find_minutes_per_mile_l114_114052

-- Conditions
def num_of_movies : ℕ := 2
def avg_length_of_movie_hours : ℝ := 1.5
def total_distance_miles : ℝ := 15

-- Question and proof target
theorem find_minutes_per_mile :
  (num_of_movies * avg_length_of_movie_hours * 60) / total_distance_miles = 12 :=
by
  -- Insert the proof here (not required as per the task instructions)
  sorry

end find_minutes_per_mile_l114_114052


namespace simplify_sqrt7_pow6_l114_114641

theorem simplify_sqrt7_pow6 : (Real.sqrt 7)^6 = (343 : Real) :=
by 
  -- we'll fill in the proof later
  sorry

end simplify_sqrt7_pow6_l114_114641


namespace number_of_non_congruent_triangles_l114_114909

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def non_congruent_triangles_with_perimeter_12 : ℕ :=
  { (a, b, c) | a ≤ b ∧ b ≤ c ∧ a + b + c = 12 ∧ is_triangle a b c }.to_finset.card

theorem number_of_non_congruent_triangles : non_congruent_triangles_with_perimeter_12 = 2 := sorry

end number_of_non_congruent_triangles_l114_114909


namespace time_to_pick_up_dog_l114_114778

def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_cleaning_time : ℕ := 10
def cooking_time : ℕ := 90
def dinner_time_in_minutes : ℕ := 180  -- 7:00 pm - 4:00 pm in minutes

def total_known_time : ℕ := commute_time + grocery_time + dry_cleaning_time + cooking_time

theorem time_to_pick_up_dog : (dinner_time_in_minutes - total_known_time) = 20 :=
by
  -- Proof goes here.
  sorry

end time_to_pick_up_dog_l114_114778


namespace num_ways_to_distribute_balls_into_boxes_l114_114454

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114454


namespace focus_of_parabola_l114_114718

theorem focus_of_parabola (x f d : ℝ) (h : ∀ x, y = 4 * x^2 → (x = 0 ∧ y = f) → PF^2 = PQ^2 ∧ 
(PF^2 = x^2 + (4 * x^2 - f) ^ 2) := (PQ^2 = (4 * x^2 - d) ^ 2)) : 
  f = 1 / 16 := 
sorry

end focus_of_parabola_l114_114718


namespace tan_minus_405_eq_neg1_l114_114699

theorem tan_minus_405_eq_neg1 :
  let θ := 405
  in  tan (-θ : ℝ) = -1 :=
by
  sorry

end tan_minus_405_eq_neg1_l114_114699


namespace length_of_room_l114_114520

theorem length_of_room 
  (width : ℝ) (cost : ℝ) (rate : ℝ) (area : ℝ) (length : ℝ) 
  (h1 : width = 3.75) 
  (h2 : cost = 24750) 
  (h3 : rate = 1200) 
  (h4 : area = cost / rate) 
  (h5 : area = length * width) : 
  length = 5.5 :=
sorry

end length_of_room_l114_114520


namespace share_money_3_people_l114_114153

theorem share_money_3_people (total_money : ℝ) (amount_per_person : ℝ) (h1 : total_money = 3.75) (h2 : amount_per_person = 1.25) : 
  total_money / amount_per_person = 3 := by
  sorry

end share_money_3_people_l114_114153


namespace sum_of_first_eight_terms_l114_114899

theorem sum_of_first_eight_terms (a : ℝ) (r : ℝ) 
  (h1 : r = 2) (h2 : a * (1 + 2 + 4 + 8) = 1) :
  a * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 17 :=
by
  -- sorry is used to skip the proof
  sorry

end sum_of_first_eight_terms_l114_114899


namespace point_b_in_third_quadrant_l114_114685

-- Definitions of the points with their coordinates
def PointA : ℝ × ℝ := (2, 3)
def PointB : ℝ × ℝ := (-1, -4)
def PointC : ℝ × ℝ := (-4, 1)
def PointD : ℝ × ℝ := (5, -3)

-- Definition of a point being in the third quadrant
def inThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- The main Theorem to prove that PointB is in the third quadrant
theorem point_b_in_third_quadrant : inThirdQuadrant PointB :=
by sorry

end point_b_in_third_quadrant_l114_114685


namespace balls_into_boxes_l114_114400

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114400


namespace log_eq_neg_two_l114_114585

theorem log_eq_neg_two : ∀ (x : ℝ), (1 / 5) ^ x = 25 → x = -2 :=
by
  intros x h
  sorry

end log_eq_neg_two_l114_114585


namespace sum_of_endpoints_l114_114930

noncomputable def triangle_side_length (PQ QR PR QS PS : ℝ) (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = (PS / QS)) : ℝ :=
  if 4 < PR ∧ PR < 18 then 4 + 18 else 0

theorem sum_of_endpoints {PQ PR QS PS : ℝ} (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = ( PS / QS)) :
  triangle_side_length PQ 0 PR QS PS h1 h2 h3 = 22 := by
  sorry

end sum_of_endpoints_l114_114930


namespace balls_into_boxes_l114_114336

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114336


namespace nancy_carrots_next_day_l114_114943

-- Definitions based on conditions
def carrots_picked_on_first_day : Nat := 12
def carrots_thrown_out : Nat := 2
def total_carrots_after_two_days : Nat := 31

-- Problem statement
theorem nancy_carrots_next_day :
  let carrots_left_after_first_day := carrots_picked_on_first_day - carrots_thrown_out
  let carrots_picked_next_day := total_carrots_after_two_days - carrots_left_after_first_day
  carrots_picked_next_day = 21 :=
by
  sorry

end nancy_carrots_next_day_l114_114943


namespace gabby_fruit_total_l114_114885

-- Definitions based on conditions
def watermelon : ℕ := 1
def peaches : ℕ := watermelon + 12
def plums : ℕ := peaches * 3
def total_fruit : ℕ := watermelon + peaches + plums

-- Proof statement
theorem gabby_fruit_total : total_fruit = 53 := 
by {
  sorry
}

end gabby_fruit_total_l114_114885


namespace parabola_focus_l114_114714

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l114_114714


namespace lyle_percentage_l114_114789

theorem lyle_percentage (chips : ℕ) (ian_ratio lyle_ratio : ℕ) (h_ratio_sum : ian_ratio + lyle_ratio = 10) (h_chips : chips = 100) :
  (lyle_ratio / (ian_ratio + lyle_ratio) : ℚ) * 100 = 60 := 
by
  sorry

end lyle_percentage_l114_114789


namespace balls_into_boxes_l114_114432

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114432


namespace obtuse_triangles_in_17_gon_l114_114911

noncomputable def number_of_obtuse_triangles (n : ℕ): ℕ := 
  if h : n ≥ 3 then (n * (n - 1) * (n - 2)) / 6 else 0

theorem obtuse_triangles_in_17_gon : number_of_obtuse_triangles 17 = 476 := sorry

end obtuse_triangles_in_17_gon_l114_114911


namespace balls_into_boxes_l114_114436

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114436


namespace find_b_vector_l114_114887

-- Define input vectors a, b, and their sum.
def vec_a : ℝ × ℝ × ℝ := (1, -2, 1)
def vec_b : ℝ × ℝ × ℝ := (-2, 4, -2)
def vec_sum : ℝ × ℝ × ℝ := (-1, 2, -1)

-- The theorem statement to prove that b is calculated correctly.
theorem find_b_vector :
  vec_a + vec_b = vec_sum →
  vec_b = (-2, 4, -2) :=
by
  sorry

end find_b_vector_l114_114887


namespace platform_length_l114_114063

variable (L : ℝ) -- The length of the platform
variable (train_length : ℝ := 300) -- The length of the train
variable (time_pole : ℝ := 26) -- Time to cross the signal pole
variable (time_platform : ℝ := 39) -- Time to cross the platform

theorem platform_length :
  (train_length / time_pole) = (train_length + L) / time_platform → L = 150 := sorry

end platform_length_l114_114063


namespace sum_of_operations_l114_114024

def operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem sum_of_operations : operation 12 5 + operation 8 3 = 174 := by
  sorry

end sum_of_operations_l114_114024


namespace average_headcount_l114_114042

theorem average_headcount 
  (h1 : ℕ := 11500) 
  (h2 : ℕ := 11600) 
  (h3 : ℕ := 11300) : 
  (Float.round ((h1 + h2 + h3 : ℕ : Float) / 3) = 11467) :=
sorry

end average_headcount_l114_114042


namespace neg_or_implication_l114_114918

theorem neg_or_implication {p q : Prop} : ¬(p ∨ q) → (¬p ∧ ¬q) :=
by
  intros h
  sorry

end neg_or_implication_l114_114918


namespace ten_faucets_fill_50_gallons_in_60_seconds_l114_114731

-- Define the rate of water dispensed by each faucet and time calculation
def fill_tub_time (num_faucets tub_volume faucet_rate : ℝ) : ℝ :=
  tub_volume / (num_faucets * faucet_rate)

-- Given conditions
def five_faucets_fill_200_gallons_in_8_minutes : Prop :=
  ∀ faucet_rate : ℝ, 5 * faucet_rate * 8 = 200

-- Main theorem: Ten faucets fill a 50-gallon tub in 60 seconds
theorem ten_faucets_fill_50_gallons_in_60_seconds 
  (faucet_rate : ℝ) 
  (h : five_faucets_fill_200_gallons_in_8_minutes) : 
  fill_tub_time 10 50 faucet_rate = 1 := by
  sorry

end ten_faucets_fill_50_gallons_in_60_seconds_l114_114731


namespace potassium_bromate_molecular_weight_l114_114180

def potassium_atomic_weight : Real := 39.10
def bromine_atomic_weight : Real := 79.90
def oxygen_atomic_weight : Real := 16.00
def oxygen_atoms : Nat := 3

theorem potassium_bromate_molecular_weight :
  potassium_atomic_weight + bromine_atomic_weight + oxygen_atoms * oxygen_atomic_weight = 167.00 :=
by
  sorry

end potassium_bromate_molecular_weight_l114_114180


namespace cos_sub_eq_five_over_eight_l114_114741

theorem cos_sub_eq_five_over_eight (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end cos_sub_eq_five_over_eight_l114_114741


namespace num_ways_to_distribute_balls_into_boxes_l114_114452

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114452


namespace area_of_quadrilateral_APQC_l114_114033

-- Define the geometric entities and conditions
structure RightTriangle (a b c : ℝ) :=
  (hypotenuse_eq: c = Real.sqrt (a ^ 2 + b ^ 2))

-- Triangles PAQ and PQC are right triangles with given sides
def PAQ := RightTriangle 9 12 (Real.sqrt (9^2 + 12^2))
def PQC := RightTriangle 12 9 (Real.sqrt (15^2 - 12^2))

-- Prove that the area of quadrilateral APQC is 108 square units
theorem area_of_quadrilateral_APQC :
  let area_PAQ := 1/2 * 9 * 12
  let area_PQC := 1/2 * 12 * 9
  area_PAQ + area_PQC = 108 :=
by
  sorry

end area_of_quadrilateral_APQC_l114_114033


namespace ways_to_place_balls_in_boxes_l114_114388

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114388


namespace number_of_triangles_in_regular_decagon_l114_114233

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l114_114233


namespace ball_box_distribution_l114_114462

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114462


namespace find_x1_plus_x2_l114_114308

def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem find_x1_plus_x2 (x1 x2 : ℝ) (hneq : x1 ≠ x2) (h1 : f x1 = 101) (h2 : f x2 = 101) : x1 + x2 = 2 := 
by 
  -- proof or sorry can be used; let's assume we use sorry to skip proof
  sorry

end find_x1_plus_x2_l114_114308


namespace polynomial_n_values_possible_num_values_of_n_l114_114025

theorem polynomial_n_values_possible :
  ∃ (n : ℤ), 
    (∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → x > 0) ∧
    (∃ a : ℤ, a > 0 ∧ ∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → 
      x = a ∨ x = a / 4 + r ∨ x = a / 4 - r) ∧
    1 ≤ r^2 ∧ r^2 ≤ 4090499 :=
sorry

theorem num_values_of_n : 
  ∃ (n_values : ℤ), n_values = 4088474 :=
sorry

end polynomial_n_values_possible_num_values_of_n_l114_114025


namespace shortest_distance_to_circle_l114_114049

def center : ℝ × ℝ := (8, 7)
def radius : ℝ := 5
def point : ℝ × ℝ := (1, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

theorem shortest_distance_to_circle :
  distance point center - radius = Real.sqrt 130 - 5 :=
by
  sorry

end shortest_distance_to_circle_l114_114049


namespace f_at_8_l114_114311

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem f_at_8 : f 8 = -1 := 
by
-- The following will be filled with the proof, hence sorry for now.
sorry

end f_at_8_l114_114311


namespace alice_weekly_walk_distance_l114_114865

theorem alice_weekly_walk_distance :
  let miles_to_school_per_day := 10
  let miles_home_per_day := 12
  let days_per_week := 5
  let weekly_total_miles := (miles_to_school_per_day * days_per_week) + (miles_home_per_day * days_per_week)
  weekly_total_miles = 110 :=
by
  sorry

end alice_weekly_walk_distance_l114_114865


namespace trisha_take_home_pay_l114_114037

def hourly_wage : ℝ := 15
def hours_per_week : ℝ := 40
def weeks_per_year : ℝ := 52
def tax_rate : ℝ := 0.2

def annual_gross_pay : ℝ := hourly_wage * hours_per_week * weeks_per_year
def amount_withheld : ℝ := tax_rate * annual_gross_pay
def annual_take_home_pay : ℝ := annual_gross_pay - amount_withheld

theorem trisha_take_home_pay :
  annual_take_home_pay = 24960 := 
by
  sorry

end trisha_take_home_pay_l114_114037


namespace sequence_geometric_progression_l114_114171

theorem sequence_geometric_progression (p : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = p * a n + 2^n)
  (h3 : ∀ n : ℕ, 0 < n → a (n + 1)^2 = a n * a (n + 2)): 
  ∃ p : ℝ, ∀ n : ℕ, a n = 2^n :=
by
  sorry

end sequence_geometric_progression_l114_114171


namespace gcd_12m_18n_l114_114475

theorem gcd_12m_18n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_gcd_mn : m.gcd n = 10) : (12 * m).gcd (18 * n) = 60 := by
  sorry

end gcd_12m_18n_l114_114475


namespace num_passenger_cars_l114_114070

noncomputable def passengerCars (p c : ℕ) : Prop :=
  c = p / 2 + 3 ∧ p + c = 69

theorem num_passenger_cars (p c : ℕ) (h : passengerCars p c) : p = 44 :=
by
  unfold passengerCars at h
  cases h
  sorry

end num_passenger_cars_l114_114070


namespace triangles_from_decagon_l114_114273

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l114_114273


namespace physics_class_size_l114_114219

variable (students : ℕ)
variable (physics math both : ℕ)

-- Conditions
def conditions := students = 75 ∧ physics = 2 * (math - both) + both ∧ both = 9

-- The proof goal
theorem physics_class_size : conditions students physics math both → physics = 56 := 
by 
  sorry

end physics_class_size_l114_114219


namespace tangent_line_eq_l114_114712

noncomputable def curve (x : ℝ) : ℝ := x^2 + 3 * x + 1

def point : ℝ × ℝ := (2, 5)

theorem tangent_line_eq : ∀ (x y : ℝ), 
  (y = x^2 + 3 * x + 1) ∧ (x = 2 ∧ y = 5) →
  7 * x - y = 9 :=
by
  intros x y h
  sorry

end tangent_line_eq_l114_114712


namespace ways_to_distribute_balls_l114_114361

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114361


namespace range_f_compare_sizes_final_comparison_l114_114322

noncomputable def f (x : ℝ) := |2 * x - 1| + |x + 1|

theorem range_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = {y : ℝ | y ∈ Set.Ici (3 / 2)} :=
sorry

theorem compare_sizes (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
sorry

theorem final_comparison (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
by
  exact compare_sizes a ha

end range_f_compare_sizes_final_comparison_l114_114322


namespace no_polynomial_transformation_l114_114491

-- Define the problem conditions: initial and target sequences
def initial_seq : List ℤ := [-3, -1, 1, 3]
def target_seq : List ℤ := [-3, -1, -3, 3]

-- State the main theorem to be proved
theorem no_polynomial_transformation :
  ¬ (∃ (P : ℤ → ℤ), ∀ x ∈ initial_seq, P x ∈ target_seq) :=
  sorry

end no_polynomial_transformation_l114_114491


namespace ways_to_distribute_balls_l114_114413

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114413


namespace ball_box_distribution_l114_114460

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114460


namespace sum_of_three_numbers_is_neg_fifteen_l114_114087

theorem sum_of_three_numbers_is_neg_fifteen
  (a b c : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 5)
  (h4 : (a + b + c) / 3 = c - 20)
  (h5 : b = 10) :
  a + b + c = -15 := by
  sorry

end sum_of_three_numbers_is_neg_fifteen_l114_114087


namespace Ram_Gohul_days_work_together_l114_114122

-- Define the conditions
def Ram_days := 10
def Gohul_days := 15

-- Define the work rates
def Ram_rate := 1 / Ram_days
def Gohul_rate := 1 / Gohul_days

-- Define the combined work rate
def Combined_rate := Ram_rate + Gohul_rate

-- Define the number of days to complete the job together
def Together_days := 1 / Combined_rate

-- State the proof problem
theorem Ram_Gohul_days_work_together : Together_days = 6 := by
  sorry

end Ram_Gohul_days_work_together_l114_114122


namespace ways_to_distribute_balls_l114_114415

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114415


namespace sack_flour_cost_l114_114541

theorem sack_flour_cost
  (x y : ℝ) 
  (h1 : 10 * x + 800 = 108 * y)
  (h2 : 4 * x - 800 = 36 * y) : x = 1600 := by
  -- Add your proof here
  sorry

end sack_flour_cost_l114_114541


namespace cards_given_by_Dan_l114_114162

def initial_cards : Nat := 27
def bought_cards : Nat := 20
def total_cards : Nat := 88

theorem cards_given_by_Dan :
  ∃ (cards_given : Nat), cards_given = total_cards - bought_cards - initial_cards :=
by
  use 41
  sorry

end cards_given_by_Dan_l114_114162


namespace ways_to_distribute_balls_l114_114354

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114354


namespace f_val_at_100_l114_114645

theorem f_val_at_100 (f : ℝ → ℝ) (h₀ : ∀ x, f x * f (x + 3) = 12) (h₁ : f 1 = 4) : f 100 = 3 :=
sorry

end f_val_at_100_l114_114645


namespace Q_polynomial_l114_114056

def cos3x_using_cos2x (cos_α : ℝ) := (2 * cos_α^2 - 1) * cos_α - 2 * (1 - cos_α^2) * cos_α

def Q (x : ℝ) := 4 * x^3 - 3 * x

theorem Q_polynomial (α : ℝ) : Q (Real.cos α) = Real.cos (3 * α) := by
  rw [Real.cos_three_mul]
  sorry

end Q_polynomial_l114_114056


namespace find_value_l114_114474

-- Given conditions of the problem
axiom condition : ∀ (a : ℝ), a - 1/a = 1

-- The mathematical proof problem
theorem find_value (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 :=
by
  sorry

end find_value_l114_114474


namespace ribbon_total_length_l114_114827

theorem ribbon_total_length (R : ℝ)
  (h_first : R - (1/2)*R = (1/2)*R)
  (h_second : (1/2)*R - (1/3)*((1/2)*R) = (1/3)*R)
  (h_third : (1/3)*R - (1/2)*((1/3)*R) = (1/6)*R)
  (h_remaining : (1/6)*R = 250) :
  R = 1500 :=
sorry

end ribbon_total_length_l114_114827


namespace sarah_speed_for_rest_of_trip_l114_114516

def initial_speed : ℝ := 15  -- miles per hour
def initial_time : ℝ := 1  -- hour
def total_distance : ℝ := 45  -- miles
def extra_time_if_same_speed : ℝ := 1  -- hour (late)
def arrival_early_time : ℝ := 0.5  -- hour (early)

theorem sarah_speed_for_rest_of_trip (remaining_distance remaining_time : ℝ) :
  remaining_distance = total_distance - initial_speed * initial_time →
  remaining_time = (remaining_distance / initial_speed - extra_time_if_same_speed) + arrival_early_time →
  remaining_distance / remaining_time = 20 :=
by
  intros h1 h2
  sorry

end sarah_speed_for_rest_of_trip_l114_114516


namespace find_square_sum_of_xy_l114_114708

theorem find_square_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y + x + y = 83) (h2 : x^2 * y + x * y^2 = 1056) : x^2 + y^2 = 458 :=
sorry

end find_square_sum_of_xy_l114_114708


namespace triangle_angles_l114_114486

theorem triangle_angles
  (A B C M : Type)
  (ortho_divides_height_A : ∀ (H_AA1 : ℝ), ∃ (H_AM : ℝ), H_AA1 = H_AM * 3 ∧ H_AM = 2 * H_AA1 / 3)
  (ortho_divides_height_B : ∀ (H_BB1 : ℝ), ∃ (H_BM : ℝ), H_BB1 = H_BM * 5 / 2 ∧ H_BM = 3 * H_BB1 / 5) :
  ∃ α β γ : ℝ, α = 60 + 40 / 60 ∧ β = 64 + 36 / 60 ∧ γ = 54 + 44 / 60 :=
by { 
  sorry 
}

end triangle_angles_l114_114486


namespace roots_quadratic_eq_a2_b2_l114_114989

theorem roots_quadratic_eq_a2_b2 (a b : ℝ) (h1 : a^2 - 5 * a + 5 = 0) (h2 : b^2 - 5 * b + 5 = 0) : a^2 + b^2 = 15 :=
by
  sorry

end roots_quadratic_eq_a2_b2_l114_114989


namespace bob_final_amount_l114_114692

noncomputable def final_amount (start: ℝ) : ℝ :=
  let day1 := start - (3/5) * start
  let day2 := day1 - (7/12) * day1
  let day3 := day2 - (2/3) * day2
  let day4 := day3 - (1/6) * day3
  let day5 := day4 - (5/8) * day4
  let day6 := day5 - (3/5) * day5
  day6

theorem bob_final_amount : final_amount 500 = 3.47 := by
  sorry

end bob_final_amount_l114_114692


namespace leo_peeled_potatoes_l114_114636

noncomputable def lucy_rate : ℝ := 4
noncomputable def leo_rate : ℝ := 6
noncomputable def total_potatoes : ℝ := 60
noncomputable def lucy_time_alone : ℝ := 6
noncomputable def total_potatoes_left : ℝ := total_potatoes - lucy_rate * lucy_time_alone
noncomputable def combined_rate : ℝ := lucy_rate + leo_rate
noncomputable def combined_time : ℝ := total_potatoes_left / combined_rate
noncomputable def leo_potatoes : ℝ := combined_time * leo_rate

theorem leo_peeled_potatoes :
  leo_potatoes = 22 :=
by
  sorry

end leo_peeled_potatoes_l114_114636


namespace greatest_possible_radius_of_circle_l114_114616

theorem greatest_possible_radius_of_circle
  (π : Real)
  (r : Real)
  (h : π * r^2 < 100 * π) :
  ∃ (n : ℕ), n = 9 ∧ (r : ℝ) ≤ 10 ∧ (r : ℝ) ≥ 9 :=
by
  sorry

end greatest_possible_radius_of_circle_l114_114616


namespace probability_red_ball_first_occurrence_l114_114735

theorem probability_red_ball_first_occurrence 
  (P : ℕ → ℝ) : 
  ∃ (P1 P2 P3 P4 : ℝ),
    P 1 = 0.4 ∧ P 2 = 0.3 ∧ P 3 = 0.2 ∧ P 4 = 0.1 :=
  sorry

end probability_red_ball_first_occurrence_l114_114735


namespace number_of_triangles_in_decagon_l114_114286

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l114_114286


namespace store_discount_problem_l114_114074

theorem store_discount_problem (original_price : ℝ) :
  let price_after_first_discount := original_price * 0.75
  let price_after_second_discount := price_after_first_discount * 0.90
  let true_discount := 1 - price_after_second_discount / original_price
  let claimed_discount := 0.40
  let difference := claimed_discount - true_discount
  true_discount = 0.325 ∧ difference = 0.075 :=
by
  sorry

end store_discount_problem_l114_114074


namespace non_congruent_triangles_with_perimeter_12_l114_114910

theorem non_congruent_triangles_with_perimeter_12 :
  ∃ (S : finset (ℤ × ℤ × ℤ)), S.card = 2 ∧ ∀ (a b c : ℤ), (a, b, c) ∈ S →
  a + b + c = 12 ∧ a ≤ b ∧ b ≤ c ∧ c < a + b :=
sorry

end non_congruent_triangles_with_perimeter_12_l114_114910


namespace time_lent_to_C_eq_l114_114567

variable (principal_B : ℝ := 5000)
variable (time_B : ℕ := 2)
variable (principal_C : ℝ := 3000)
variable (total_interest : ℝ := 1980)
variable (rate_of_interest_per_annum : ℝ := 0.09)

theorem time_lent_to_C_eq (n : ℝ) (H : principal_B * rate_of_interest_per_annum * time_B + principal_C * rate_of_interest_per_annum * n = total_interest) : 
  n = 2 / 3 :=
by
  sorry

end time_lent_to_C_eq_l114_114567


namespace adjacent_product_negative_l114_114926

noncomputable def a_seq : ℕ → ℚ
| 0 => 15
| (n+1) => (a_seq n) - (2 / 3)

theorem adjacent_product_negative :
  ∃ n : ℕ, a_seq 22 * a_seq 23 < 0 :=
by
  -- From the conditions, it is known that a_seq satisfies the recursive definition
  --
  -- We seek to prove that a_seq 22 * a_seq 23 < 0
  sorry

end adjacent_product_negative_l114_114926


namespace distinct_balls_boxes_l114_114409

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114409


namespace average_headcount_correct_l114_114041

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end average_headcount_correct_l114_114041


namespace scientific_notation_correct_l114_114156

noncomputable def scientific_notation_139000 : Prop :=
  139000 = 1.39 * 10^5

theorem scientific_notation_correct : scientific_notation_139000 :=
by
  -- The proof would be included here, but we add sorry to skip it
  sorry

end scientific_notation_correct_l114_114156


namespace smallest_positive_integer_form_3003_55555_l114_114188

theorem smallest_positive_integer_form_3003_55555 :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 57 :=
by {
  sorry
}

end smallest_positive_integer_form_3003_55555_l114_114188


namespace total_amount_spent_l114_114102

-- Define the variables B and D representing the amounts Ben and David spent.
variables (B D : ℝ)

-- Define the conditions based on the given problem.
def conditions : Prop :=
  (D = 0.60 * B) ∧ (B = D + 14)

-- The main theorem stating the total amount spent by Ben and David is 56.
theorem total_amount_spent (h : conditions B D) : B + D = 56 :=
sorry  -- Proof omitted.

end total_amount_spent_l114_114102


namespace train_cross_bridge_time_l114_114076

noncomputable def length_train : ℝ := 130
noncomputable def length_bridge : ℝ := 320
noncomputable def speed_kmh : ℝ := 54
noncomputable def speed_ms : ℝ := speed_kmh * 1000 / 3600

theorem train_cross_bridge_time :
  (length_train + length_bridge) / speed_ms = 30 := by
  sorry

end train_cross_bridge_time_l114_114076


namespace trapezoid_two_heights_l114_114681

-- Define trivially what a trapezoid is, in terms of having two parallel sides.
structure Trapezoid :=
(base1 base2 : ℝ)
(height1 height2 : ℝ)
(has_two_heights : height1 = height2)

theorem trapezoid_two_heights (T : Trapezoid) : ∃ h1 h2 : ℝ, h1 = h2 :=
by
  use T.height1
  use T.height2
  exact T.has_two_heights

end trapezoid_two_heights_l114_114681


namespace ondra_homework_problems_l114_114526

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end ondra_homework_problems_l114_114526


namespace problem_a_l114_114146

variable {S : Type*}
variables (a b : S)
variables [Inhabited S] -- Ensures S has at least one element
variables (op : S → S → S) -- Defines the binary operation

-- Condition: binary operation a * (b * a) = b holds for all a, b in S
axiom binary_condition : ∀ a b : S, op a (op b a) = b

-- Theorem to prove: (a * b) * a ≠ a
theorem problem_a : (op (op a b) a) ≠ a :=
sorry

end problem_a_l114_114146


namespace probability_red_jelly_bean_l114_114065

variable (r b g : Nat) (eaten_green eaten_blue : Nat)

theorem probability_red_jelly_bean
    (h_r : r = 15)
    (h_b : b = 20)
    (h_g : g = 16)
    (h_eaten_green : eaten_green = 1)
    (h_eaten_blue : eaten_blue = 1)
    (h_total : r + b + g = 51)
    (h_remaining_total : r + (b - eaten_blue) + (g - eaten_green) = 49) :
    (r : ℚ) / 49 = 15 / 49 :=
by
  sorry

end probability_red_jelly_bean_l114_114065


namespace balls_into_boxes_l114_114430

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114430


namespace geometric_series_sum_l114_114293

theorem geometric_series_sum {a r : ℚ} (n : ℕ) (h_a : a = 3/4) (h_r : r = 3/4) (h_n : n = 8) : 
       a * (1 - r^n) / (1 - r) = 176925 / 65536 :=
by
  -- Utilizing the provided conditions
  have h_a := h_a
  have h_r := h_r
  have h_n := h_n
  -- Proving the theorem using sorry as a placeholder for the detailed steps
  sorry

end geometric_series_sum_l114_114293


namespace parabola_focus_l114_114713

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end parabola_focus_l114_114713


namespace min_value_expression_l114_114500

noncomputable def f (x y : ℝ) : ℝ := 
  (x + 1 / y) * (x + 1 / y - 2023) + (y + 1 / x) * (y + 1 / x - 2023)

theorem min_value_expression : ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ f x y = -2048113 :=
sorry

end min_value_expression_l114_114500


namespace boat_travel_distance_l114_114492

variable (v c d : ℝ) (c_eq_1 : c = 1)

theorem boat_travel_distance : 
  (∀ (v : ℝ), d = (v + c) * 4 → d = (v - c) * 6) → d = 24 := 
by
  intro H
  sorry

end boat_travel_distance_l114_114492


namespace find_integer_n_l114_114871

theorem find_integer_n (n : ℤ) : 
  (∃ m : ℤ, n = 35 * m + 24) ↔ (5 ∣ (3 * n - 2) ∧ 7 ∣ (2 * n + 1)) :=
by sorry

end find_integer_n_l114_114871


namespace misha_problem_l114_114131

theorem misha_problem (N : ℕ) (h : ∀ a, a ∈ {a | a > 1 → ∃ b > 0, b ∈ {b' | b' < a ∧ a % b' = 0}}) :
  (∀ t : ℕ, (t > 1) → (1 / t ^ 2) < (1 / t * (t - 1))) →
  (∃ (n : ℕ), n = 1) → (N = 1 ↔ ∃ (k : ℕ), k = N^2) :=
by
  sorry

end misha_problem_l114_114131


namespace rectangles_with_one_gray_cell_l114_114761

-- Define the number of gray cells
def gray_cells : ℕ := 40

-- Define the total rectangles containing exactly one gray cell
def total_rectangles : ℕ := 176

-- The theorem we want to prove
theorem rectangles_with_one_gray_cell (h : gray_cells = 40) : total_rectangles = 176 := 
by 
  sorry

end rectangles_with_one_gray_cell_l114_114761


namespace tan_neg_405_eq_neg_1_l114_114702

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end tan_neg_405_eq_neg_1_l114_114702


namespace num_non_congruent_triangles_with_perimeter_12_l114_114907

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end num_non_congruent_triangles_with_perimeter_12_l114_114907


namespace num_triangles_from_decagon_l114_114275

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l114_114275


namespace ratio_of_percentages_l114_114660

theorem ratio_of_percentages (x y : ℝ) (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by
  sorry

end ratio_of_percentages_l114_114660


namespace total_weight_all_bags_sold_l114_114489

theorem total_weight_all_bags_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions morning_carrots afternoon_carrots : ℕ)
  (weight_potatoes weight_onions weight_carrots total_weight : ℕ)
  (h_morning_potatoes : morning_potatoes = 29)
  (h_afternoon_potatoes : afternoon_potatoes = 17)
  (h_morning_onions : morning_onions = 15)
  (h_afternoon_onions : afternoon_onions = 22)
  (h_morning_carrots : morning_carrots = 12)
  (h_afternoon_carrots : afternoon_carrots = 9)
  (h_weight_potatoes : weight_potatoes = 7)
  (h_weight_onions : weight_onions = 5)
  (h_weight_carrots : weight_carrots = 4)
  (h_total_weight : total_weight = 591) :
  morning_potatoes + afternoon_potatoes * weight_potatoes +
  morning_onions + afternoon_onions * weight_onions +
  morning_carrots + afternoon_carrots * weight_carrots = total_weight :=
by {
  sorry
}

end total_weight_all_bags_sold_l114_114489


namespace marsha_first_package_miles_l114_114942

noncomputable def total_distance (x : ℝ) : ℝ := x + 28 + 14

noncomputable def earnings (x : ℝ) : ℝ := total_distance x * 2

theorem marsha_first_package_miles : ∃ x : ℝ, earnings x = 104 ∧ x = 10 :=
by
  use 10
  sorry

end marsha_first_package_miles_l114_114942


namespace solve_for_y_l114_114090

theorem solve_for_y (y : ℚ) (h : |5 * y - 6| = 0) : y = 6 / 5 :=
by 
  sorry

end solve_for_y_l114_114090


namespace books_in_series_l114_114029

-- Define the number of movies
def M := 14

-- Define that the number of books is one more than the number of movies
def B := M + 1

-- Theorem statement to prove that the number of books is 15
theorem books_in_series : B = 15 :=
by
  sorry

end books_in_series_l114_114029


namespace remainder_T10_mod_5_l114_114734

noncomputable def T : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => T (n+1) + T n + T n

theorem remainder_T10_mod_5 :
  (T 10) % 5 = 4 :=
sorry

end remainder_T10_mod_5_l114_114734


namespace balls_into_boxes_l114_114333

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114333


namespace room_analysis_l114_114009

-- First person's statements
def statement₁ (n: ℕ) (liars: ℕ) :=
  n ≤ 3 ∧ liars = n

-- Second person's statements
def statement₂ (n: ℕ) (liars: ℕ) :=
  n ≤ 4 ∧ liars < n

-- Third person's statements
def statement₃ (n: ℕ) (liars: ℕ) :=
  n = 5 ∧ liars = 3

theorem room_analysis (n liars : ℕ) :
  (¬ statement₁ n liars) ∧ statement₂ n liars ∧ ¬ statement₃ n liars → (n = 4 ∧ liars = 2) :=
by
  sorry

end room_analysis_l114_114009


namespace probability_at_most_one_girl_l114_114670

theorem probability_at_most_one_girl (boys girls : ℕ) (total_selected : ℕ)
  (hb : boys = 3) (hg : girls = 2) (hts : total_selected = 2) : 
  let n := Nat.choose (boys + girls) total_selected in
  let m := Nat.choose boys total_selected + (Nat.choose boys 1 * Nat.choose girls 1) in
  m / n = 9 / 10 :=
by 
  exact sorry

end probability_at_most_one_girl_l114_114670


namespace probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l114_114774

/- Define number of boys and girls -/
def num_boys : ℕ := 5
def num_girls : ℕ := 3

/- Define number of students selected -/
def num_selected : ℕ := 2

/- Define the total number of ways to select -/
def total_ways : ℕ := Nat.choose (num_boys + num_girls) num_selected

/- Define the number of ways to select exactly one girl -/
def ways_one_girl : ℕ := Nat.choose num_girls 1 * Nat.choose num_boys 1

/- Define the number of ways to select at least one girl -/
def ways_at_least_one_girl : ℕ := total_ways - Nat.choose num_boys num_selected

/- Define the first probability: exactly one girl participates -/
def prob_one_girl : ℚ := ways_one_girl / total_ways

/- Define the second probability: exactly one girl given at least one girl -/
def prob_one_girl_given_at_least_one : ℚ := ways_one_girl / ways_at_least_one_girl

theorem probability_of_one_girl : prob_one_girl = 15 / 28 := by
  sorry

theorem conditional_probability_of_one_girl_given_at_least_one : prob_one_girl_given_at_least_one = 5 / 6 := by
  sorry

end probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l114_114774


namespace prime_dates_in_2008_l114_114858

noncomputable def num_prime_dates_2008 : Nat := 52

theorem prime_dates_in_2008 : 
  let prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_months_days := [(2, 29), (3, 31), (5, 31), (7, 31), (11, 30)]
  -- Count the prime days for each month considering the list
  let prime_day_count (days : Nat) := (prime_days.filter (λ d => d <= days)).length
  -- Sum the counts for each prime month
  (prime_months_days.map (λ (m, days) => prime_day_count days)).sum = num_prime_dates_2008 :=
by
  sorry

end prime_dates_in_2008_l114_114858


namespace union_A_B_union_complement_A_B_l114_114325

open Set

-- Definitions for sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {3, 5}

-- Statement 1: Prove that A ∪ B = {1, 3, 5, 7}
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by
  sorry

-- Definition for complement of A in U
def complement_A_U : Set ℕ := {x ∈ U | x ∉ A}

-- Statement 2: Prove that (complement of A in U) ∪ B = {2, 3, 4, 5, 6}
theorem union_complement_A_B : complement_A_U ∪ B = {2, 3, 4, 5, 6} := by
  sorry

end union_A_B_union_complement_A_B_l114_114325


namespace greatest_sundays_in_56_days_l114_114971

theorem greatest_sundays_in_56_days (days_in_first: ℕ) (days_in_week: ℕ) (sundays_in_week: ℕ) : ℕ :=
by 
  -- Given conditions
  have days_in_first := 56
  have days_in_week := 7
  have sundays_in_week := 1

  -- Conclusion
  let num_weeks := days_in_first / days_in_week

  -- Answer
  exact num_weeks * sundays_in_week

-- This theorem establishes that the greatest number of Sundays in 56 days is indeed 8.
-- Proof: The number of Sundays in 56 days is given by the number of weeks (which is 8) times the number of Sundays per week (which is 1).

example : greatest_sundays_in_56_days 56 7 1 = 8 := 
by 
  unfold greatest_sundays_in_56_days
  exact rfl

end greatest_sundays_in_56_days_l114_114971


namespace operation_commutative_operation_associative_l114_114290

def my_operation (a b : ℝ) : ℝ := a * b + a + b

theorem operation_commutative (a b : ℝ) : my_operation a b = my_operation b a := by
  sorry

theorem operation_associative (a b c : ℝ) : my_operation (my_operation a b) c = my_operation a (my_operation b c) := by
  sorry

end operation_commutative_operation_associative_l114_114290


namespace intersect_sets_l114_114604

def M : Set ℝ := { x | x ≥ -1 }
def N : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersect_sets :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } := by
  sorry

end intersect_sets_l114_114604


namespace number_of_triangles_in_regular_decagon_l114_114249

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l114_114249


namespace problem1_problem2_l114_114327

def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1
def f (a b : ℝ) : ℝ := 3 * A a b + 6 * B a b

theorem problem1 (a b : ℝ) : f a b = 15 * a * b - 6 * a - 9 :=
by 
  sorry

theorem problem2 (b : ℝ) : (∀ a : ℝ, f a b = -9) → b = 2 / 5 :=
by 
  sorry

end problem1_problem2_l114_114327


namespace expected_value_of_flipped_coins_l114_114673

theorem expected_value_of_flipped_coins :
  let p := 1
  let n := 5
  let d := 10
  let q := 25
  let f := 50
  let prob := (1:ℝ) / 2
  let V := prob * p + prob * n + prob * d + prob * q + prob * f
  V = 45.5 :=
by
  sorry

end expected_value_of_flipped_coins_l114_114673


namespace fractions_equivalence_l114_114704

theorem fractions_equivalence (k : ℝ) (h : k ≠ -5) : (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := 
by 
  sorry

end fractions_equivalence_l114_114704


namespace coffee_cream_ratio_l114_114609

theorem coffee_cream_ratio :
  let
    harry_initial_coffee := 20,
    harry_drunk_coffee := 4,
    harry_added_cream := 3,
    sally_initial_coffee := 20,
    sally_added_cream := 3,
    sally_drunk_mix := 4,
    harry_remaining_coffee := harry_initial_coffee - harry_drunk_coffee,
    harry_final_mixture := harry_remaining_coffee + harry_added_cream,
    sally_initial_mixture := sally_initial_coffee + sally_added_cream,
    sally_fraction_cream := sally_added_cream / sally_initial_mixture,
    sally_cream_drunk := sally_fraction_cream * sally_drunk_mix,
    sally_remaining_cream := sally_added_cream - sally_cream_drunk,
    harry_final_cream := harry_added_cream,
    cream_ratio := harry_final_cream / sally_remaining_cream in
  cream_ratio = (23 : ℚ) / 19 :=
by
  sorry

end coffee_cream_ratio_l114_114609


namespace ball_in_boxes_l114_114346

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114346


namespace number_of_triangles_in_decagon_l114_114246

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l114_114246


namespace hexagon_area_l114_114819

-- Define the area of a triangle
def triangle_area (base height: ℝ) : ℝ := 0.5 * base * height

-- Given dimensions for each triangle
def base_unit := 1
def original_height := 3
def new_height := 4

-- Calculate areas of each triangle in the new configuration
def single_triangle_area := triangle_area base_unit new_height
def total_triangle_area := 4 * single_triangle_area

-- The area of the rectangular region formed by the hexagon and triangles
def rectangular_region_area := (base_unit + original_height + original_height) * new_height

-- Prove the area of the hexagon
theorem hexagon_area : rectangular_region_area - total_triangle_area = 32 :=
by
  -- We will provide the proof here
  sorry

end hexagon_area_l114_114819


namespace difference_between_numbers_l114_114652

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 20000) (h2 : b = 2 * a + 6) (h3 : 9 ∣ a) : b - a = 6670 :=
by
  sorry

end difference_between_numbers_l114_114652


namespace find_focus_of_parabola_4x2_l114_114720

-- Defining what it means to be the focus of a parabola given an equation y = ax^2.
def is_focus (a : ℝ) (f : ℝ) : Prop :=
  ∀ (x : ℝ), (x ^ 2 + (a * x ^ 2 - f) ^ 2) = ((a * x ^ 2 - (f * 8)) ^ 2)

-- Specific instance of the parabola y = 4x^2.
def parabola_4x2 := (4 : ℝ)

theorem find_focus_of_parabola_4x2 : ∃ f : ℝ, is_focus parabola_4x2 f :=
begin
  use (1/16 : ℝ),
  sorry -- The proof will be filled in by the theorem prover.
end

end find_focus_of_parabola_4x2_l114_114720


namespace pages_printed_l114_114627

theorem pages_printed (P : ℕ) 
  (H1 : P % 7 = 0)
  (H2 : P % 3 = 0)
  (H3 : P - (P / 7 + P / 3 - P / 21) = 24) : 
  P = 42 :=
sorry

end pages_printed_l114_114627


namespace simplify_sqrt7_to_the_six_l114_114642

theorem simplify_sqrt7_to_the_six : (sqrt 7)^6 = 343 :=
by 
  sorry

end simplify_sqrt7_to_the_six_l114_114642


namespace polynomial_form_l114_114709

theorem polynomial_form (P : ℝ → ℝ) (h₁ : P 0 = 0) (h₂ : ∀ x, P x = (P (x + 1) + P (x - 1)) / 2) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end polynomial_form_l114_114709


namespace ondra_homework_problems_l114_114525

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end ondra_homework_problems_l114_114525


namespace cube_faces_one_third_blue_l114_114211

theorem cube_faces_one_third_blue (n : ℕ) (h1 : ∃ n, n > 0 ∧ (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 := by
  sorry

end cube_faces_one_third_blue_l114_114211


namespace solve_equation_l114_114644

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔ x = -2 ∨ x = 1 / 3 :=
by
  sorry

end solve_equation_l114_114644


namespace move_line_up_l114_114573

/-- Define the original line equation as y = 3x - 2 -/
def original_line (x : ℝ) : ℝ := 3 * x - 2

/-- Define the resulting line equation as y = 3x + 4 -/
def resulting_line (x : ℝ) : ℝ := 3 * x + 4

theorem move_line_up (x : ℝ) : resulting_line x = original_line x + 6 :=
by
  sorry

end move_line_up_l114_114573


namespace cost_to_replace_and_install_l114_114999

theorem cost_to_replace_and_install (s l : ℕ) 
  (h1 : l = 3 * s) (h2 : 2 * s + 2 * l = 640) 
  (cost_per_foot : ℕ) (cost_per_gate : ℕ) (installation_cost_per_gate : ℕ) 
  (h3 : cost_per_foot = 5) (h4 : cost_per_gate = 150) (h5 : installation_cost_per_gate = 75) : 
  (s * cost_per_foot + 2 * (cost_per_gate + installation_cost_per_gate)) = 850 := 
by 
  sorry

end cost_to_replace_and_install_l114_114999


namespace angle_ABC_40_degrees_l114_114689

theorem angle_ABC_40_degrees (ABC ABD CBD : ℝ) 
    (h1 : CBD = 90) 
    (h2 : ABD = 60)
    (h3 : ABC + ABD + CBD = 190) : 
    ABC = 40 := 
by {
  sorry
}

end angle_ABC_40_degrees_l114_114689


namespace compute_expression_l114_114581

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l114_114581


namespace average_rate_of_interest_l114_114682

theorem average_rate_of_interest (total_investment : ℝ) (rate1 rate2 average_rate : ℝ) (amount1 amount2 : ℝ)
  (H1 : total_investment = 6000)
  (H2 : rate1 = 0.03)
  (H3 : rate2 = 0.07)
  (H4 : average_rate = 0.042)
  (H5 : amount1 + amount2 = total_investment)
  (H6 : rate1 * amount1 = rate2 * amount2) :
  (rate1 * amount1 + rate2 * amount2) / total_investment = average_rate := 
sorry

end average_rate_of_interest_l114_114682


namespace regular_decagon_triangle_count_l114_114255

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l114_114255


namespace angle_C_obtuse_l114_114128

theorem angle_C_obtuse (a b c C : ℝ) (h1 : a^2 + b^2 < c^2) (h2 : Real.sin C = Real.sqrt 3 / 2) : C = 2 * Real.pi / 3 :=
by
  sorry

end angle_C_obtuse_l114_114128


namespace soccer_ball_selling_price_l114_114073

theorem soccer_ball_selling_price
  (cost_price_per_ball : ℕ)
  (num_balls : ℕ)
  (total_profit : ℕ)
  (h_cost_price : cost_price_per_ball = 60)
  (h_num_balls : num_balls = 50)
  (h_total_profit : total_profit = 1950) :
  (cost_price_per_ball + (total_profit / num_balls) = 99) :=
by 
  -- Note: Proof can be filled here
  sorry

end soccer_ball_selling_price_l114_114073


namespace least_positive_three_digit_multiple_of_8_l114_114982

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l114_114982


namespace xiaoGong_walking_speed_l114_114519

-- Defining the parameters for the problem
def distance : ℕ := 1200
def daChengExtraSpeedPerMinute : ℕ := 20
def timeUntilMeetingForDaCheng : ℕ := 12
def timeUntilMeetingForXiaoGong : ℕ := 6 + timeUntilMeetingForDaCheng

-- The main statement to prove Xiao Gong's speed
theorem xiaoGong_walking_speed : ∃ v : ℕ, 12 * (v + daChengExtraSpeedPerMinute) + 18 * v = distance ∧ v = 32 :=
by
  sorry

end xiaoGong_walking_speed_l114_114519


namespace jonathans_and_sisters_total_letters_l114_114136

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l114_114136


namespace sledding_small_hills_l114_114194

theorem sledding_small_hills (total_sleds tall_hills_sleds sleds_per_tall_hill sleds_per_small_hill small_hills : ℕ) 
  (h1 : total_sleds = 14)
  (h2 : tall_hills_sleds = 2)
  (h3 : sleds_per_tall_hill = 4)
  (h4 : sleds_per_small_hill = sleds_per_tall_hill / 2)
  (h5 : total_sleds = tall_hills_sleds * sleds_per_tall_hill + small_hills * sleds_per_small_hill)
  : small_hills = 3 := 
sorry

end sledding_small_hills_l114_114194


namespace distinguish_ball_box_ways_l114_114470

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114470


namespace vector_magnitude_problem_l114_114118

open Real

noncomputable def magnitude (x : ℝ × ℝ) : ℝ := sqrt (x.1 ^ 2 + x.2 ^ 2)

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h_a : a = (1, 3))
  (h_perp : (a.1 + b.1, a.2 + b.2) • (a.1 - b.1, a.2 - b.2) = 0) :
  magnitude b = sqrt 10 := 
sorry

end vector_magnitude_problem_l114_114118


namespace sugarCubeWeight_l114_114991

theorem sugarCubeWeight
  (ants1 : ℕ) (sugar_cubes1 : ℕ) (weight1 : ℕ) (hours1 : ℕ)
  (ants2 : ℕ) (sugar_cubes2 : ℕ) (hours2 : ℕ) :
  ants1 = 15 →
  sugar_cubes1 = 600 →
  weight1 = 10 →
  hours1 = 5 →
  ants2 = 20 →
  sugar_cubes2 = 960 →
  hours2 = 3 →
  ∃ weight2 : ℕ, weight2 = 5 := by
  sorry

end sugarCubeWeight_l114_114991


namespace sum_of_reciprocals_eq_three_l114_114653

theorem sum_of_reciprocals_eq_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1/x + 1/y) = 3 := 
by
  sorry

end sum_of_reciprocals_eq_three_l114_114653


namespace find_number_l114_114101

theorem find_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end find_number_l114_114101


namespace number_of_triangles_in_decagon_l114_114288

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l114_114288


namespace direction_vector_arithmetic_sequence_l114_114829

theorem direction_vector_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) 
    (n : ℕ) 
    (S2_eq_10 : S_n 2 = 10) 
    (S5_eq_55 : S_n 5 = 55)
    (arith_seq_sum : ∀ n, S_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))) / 2): 
    (a_n (n + 2) - a_n n) / (n + 2 - n) = 4 :=
by
  sorry

end direction_vector_arithmetic_sequence_l114_114829


namespace possible_values_of_polynomial_l114_114806

theorem possible_values_of_polynomial (x : ℝ) (h : x^2 - 7 * x + 12 < 0) : 
48 < x^2 + 7 * x + 12 ∧ x^2 + 7 * x + 12 < 64 :=
sorry

end possible_values_of_polynomial_l114_114806


namespace value_of_f_at_5_l114_114168

def f (x : ℤ) : ℤ := x^3 - x^2 + x

theorem value_of_f_at_5 : f 5 = 105 := by
  sorry

end value_of_f_at_5_l114_114168


namespace compute_expression_l114_114580

theorem compute_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end compute_expression_l114_114580


namespace lake_with_more_frogs_has_45_frogs_l114_114145

-- Definitions for the problem.
variable (F : ℝ) -- Number of frogs in the lake with more frogs.
variable (F_less : ℝ) -- Number of frogs in Lake Crystal (the lake with fewer frogs).

-- Conditions
axiom fewer_frogs_condition : F_less = 0.8 * F
axiom total_frogs_condition : F + F_less = 81

-- Theorem statement: Proving that the number of frogs in the lake with more frogs is 45.
theorem lake_with_more_frogs_has_45_frogs :
  F = 45 :=
by
  sorry

end lake_with_more_frogs_has_45_frogs_l114_114145


namespace lori_marbles_l114_114635

theorem lori_marbles (friends marbles_per_friend : ℕ) (h_friends : friends = 5) (h_marbles_per_friend : marbles_per_friend = 6) : friends * marbles_per_friend = 30 := sorry

end lori_marbles_l114_114635


namespace homework_problem1_homework_problem2_l114_114528

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end homework_problem1_homework_problem2_l114_114528


namespace balls_into_boxes_l114_114338

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114338


namespace a_2017_eq_2_l114_114174

variable (n : ℕ)
variable (S : ℕ → ℤ)

/-- Define the sequence sum Sn -/
def S_n (n : ℕ) : ℤ := 2 * n - 1

/-- Define the sequence term an -/
def a_n (n : ℕ) : ℤ := S_n n - S_n (n - 1)

theorem a_2017_eq_2 : a_n 2017 = 2 := 
by
  have hSn : ∀ n, S_n n = (2 * n - 1) := by intro; simp [S_n] 
  have ha : ∀ n, a_n n = (S_n n - S_n (n - 1)) := by intro; simp [a_n]
  simp only [ha, hSn] 
  sorry

end a_2017_eq_2_l114_114174


namespace height_of_brick_l114_114834

-- Definitions of given conditions
def length_brick : ℝ := 125
def width_brick : ℝ := 11.25
def length_wall : ℝ := 800
def height_wall : ℝ := 600
def width_wall : ℝ := 22.5
def number_bricks : ℝ := 1280

-- Prove that the height of each brick is 6.01 cm
theorem height_of_brick :
  ∃ H : ℝ,
    H = 6.01 ∧
    (number_bricks * (length_brick * width_brick * H) = length_wall * height_wall * width_wall) :=
by
  sorry

end height_of_brick_l114_114834


namespace intersection_point_l114_114300

def line_eq (x y z : ℝ) : Prop :=
  (x - 1) / 1 = (y + 1) / 0 ∧ (y + 1) / 0 = (z - 1) / -1

def plane_eq (x y z : ℝ) : Prop :=
  3 * x - 2 * y - 4 * z - 8 = 0

theorem intersection_point : 
  ∃ (x y z : ℝ), line_eq x y z ∧ plane_eq x y z ∧ x = -6 ∧ y = -1 ∧ z = 8 :=
by 
  sorry

end intersection_point_l114_114300


namespace find_PC_l114_114481

noncomputable def side_lengths : ℕ × ℕ × ℕ := (10, 8, 7)

noncomputable def similarity_ratios (PC PA : ℝ) := 
  PC / PA = 7 / 10 ∧ 
  PA / (PC + 8) = 7 / 10

theorem find_PC (PC : ℝ) (PA : ℝ) (AB BC CA : ℕ) (similar : similarity_ratios PC PA) :
  AB = 10 ∧ BC = 8 ∧ CA = 7 → 
  PC = 392 / 51 :=
by
  sorry

end find_PC_l114_114481


namespace combined_average_score_clubs_l114_114993

theorem combined_average_score_clubs
  (nA nB : ℕ) -- Number of members in each club
  (avgA avgB : ℝ) -- Average score of each club
  (hA : nA = 40)
  (hB : nB = 50)
  (hAvgA : avgA = 90)
  (hAvgB : avgB = 81) :
  (nA * avgA + nB * avgB) / (nA + nB) = 85 :=
by
  sorry -- Proof omitted

end combined_average_score_clubs_l114_114993


namespace jasmine_dinner_time_l114_114628

-- Define the conditions as variables and constants.
constant work_end_time : ℕ := 16  -- Representing 4:00 pm in 24-hour format (16:00)
constant commute_time : ℕ := 30   -- in minutes
constant grocery_time : ℕ := 30   -- in minutes
constant dry_cleaning_time : ℕ := 10  -- in minutes
constant dog_grooming_time : ℕ := 20  -- in minutes
constant cooking_time : ℕ := 90   -- in minutes

-- Define a function to sum up the times
def total_time_after_work : ℕ := commute_time + grocery_time + dry_cleaning_time + dog_grooming_time + cooking_time

def time_to_hour_minutes (total_minutes : ℕ) : (ℕ × ℕ) := 
  (total_minutes / 60, total_minutes % 60)

-- Prove that Jasmine will eat dinner at 7:00 pm (19:00 in 24-hour format)
theorem jasmine_dinner_time : total_time_after_work / 60 + work_end_time = 19 := by
  -- Leave the proof part as sorry since we don't need to provide proof steps
  sorry

end jasmine_dinner_time_l114_114628


namespace distinct_balls_boxes_l114_114382

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114382


namespace problem_statement_l114_114897

noncomputable def a : ℝ := 2.68 * 0.74
noncomputable def b : ℝ := a^2 + Real.cos a

theorem problem_statement : b = 2.96535 := 
by 
  sorry

end problem_statement_l114_114897


namespace value_of_expression_l114_114913

theorem value_of_expression (x : ℕ) (h : x = 3) : 2 * x + 3 = 9 :=
by 
  sorry

end value_of_expression_l114_114913


namespace gcd_of_168_56_224_l114_114542

theorem gcd_of_168_56_224 : (Nat.gcd 168 56 = 56) ∧ (Nat.gcd 56 224 = 56) ∧ (Nat.gcd 168 224 = 56) :=
by
  sorry

end gcd_of_168_56_224_l114_114542


namespace min_value_of_expression_l114_114110

noncomputable def f (x : ℝ) : ℝ :=
  2 / x + 9 / (1 - 2 * x)

theorem min_value_of_expression (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 2) : ∃ m, f x = m ∧ m = 25 :=
by
  sorry

end min_value_of_expression_l114_114110


namespace focus_of_parabola_l114_114716

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := 1 / 16 in
    (0, f)

theorem focus_of_parabola (x : ℝ) : 
  let focus := parabola_focus in
  focus = (0, 1 / 16) :=
by
  sorry

end focus_of_parabola_l114_114716


namespace initial_number_of_girls_is_31_l114_114303

-- Define initial number of boys and girls
variables (b g : ℕ)

-- Conditions
def first_condition (g b : ℕ) : Prop := b = 3 * (g - 18)
def second_condition (g b : ℕ) : Prop := 4 * (b - 36) = g - 18

-- Theorem statement
theorem initial_number_of_girls_is_31 (b g : ℕ) (h1 : first_condition g b) (h2 : second_condition g b) : g = 31 :=
by
  sorry

end initial_number_of_girls_is_31_l114_114303


namespace intersection_points_l114_114013

-- Define parameters: number of sides for each polygon
def n₆ := 6
def n₇ := 7
def n₈ := 8
def n₉ := 9

-- Condition: polygons are inscribed in the same circle, no shared vertices, no three sides intersect at a common point
def polygons_are_disjoint (n₁ n₂ : ℕ) (n₃ n₄ : ℕ) (n₅ : ℕ) : Prop :=
  true -- Assume this is a primitive condition encapsulating given constraints

-- Prove the number of intersection points is 80
theorem intersection_points : polygons_are_disjoint n₆ n₇ n₈ n₉ n₅ → 
  2 * (n₆ + n₇ + n₇ + n₈) + 2 * (n₇ + n₈) + 2 * n₉ = 80 :=
by  
  sorry

end intersection_points_l114_114013


namespace number_of_triangles_in_regular_decagon_l114_114253

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l114_114253


namespace calc_r_over_s_at_2_l114_114169

def r (x : ℝ) := 3 * (x - 4) * (x - 1)
def s (x : ℝ) := (x - 4) * (x + 3)

theorem calc_r_over_s_at_2 : (r 2) / (s 2) = 3 / 5 := by
  sorry

end calc_r_over_s_at_2_l114_114169


namespace find_cost_price_l114_114659

theorem find_cost_price 
  (C : ℝ)
  (h1 : 1.10 * C + 110 = 1.15 * C)
  : C = 2200 :=
sorry

end find_cost_price_l114_114659


namespace new_pressure_eq_l114_114082

-- Defining the initial conditions and values
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3.5
def new_volume : ℝ := 10.5
def k : ℝ := initial_pressure * initial_volume

-- The statement to prove
theorem new_pressure_eq :
  ∃ p_new : ℝ, new_volume * p_new = k ∧ p_new = 8 / 3 :=
by
  use (8 / 3)
  sorry

end new_pressure_eq_l114_114082


namespace radius_of_large_circle_l114_114032

noncomputable def small_circle_radius : ℝ := 2
noncomputable def larger_circle_radius : ℝ :=
  let side_length := 2 * small_circle_radius in
  let altitude := (side_length * Real.sqrt 3) / 2 in
  side_length + altitude

theorem radius_of_large_circle :
  let r := (2 + 4 * Real.sqrt 3) / 2 in
  larger_circle_radius = r := by
  sorry

end radius_of_large_circle_l114_114032


namespace orthogonal_vectors_y_value_l114_114860

theorem orthogonal_vectors_y_value (y : ℝ) :
  (3 : ℝ) * (-1) + (4 : ℝ) * y = 0 → y = 3 / 4 :=
by
  sorry

end orthogonal_vectors_y_value_l114_114860


namespace balls_into_boxes_l114_114394

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114394


namespace balls_into_boxes_l114_114393

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114393


namespace block_wall_min_blocks_l114_114831

theorem block_wall_min_blocks :
  ∃ n,
    n = 648 ∧
    ∀ (row_height wall_height block1_length block2_length wall_length: ℕ),
    row_height = 1 ∧
    wall_height = 8 ∧
    block1_length = 1 ∧
    block2_length = 3/2 ∧
    wall_length = 120 ∧
    (∀ i : ℕ, i < wall_height → ∃ k m : ℕ, k * block1_length + m * block2_length = wall_length) →
    n = (wall_height * (1 + 2 * 79))
:= by sorry

end block_wall_min_blocks_l114_114831


namespace ball_box_distribution_l114_114456

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114456


namespace sample_variance_is_two_l114_114030

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : (1 / 5) * ((-1 - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end sample_variance_is_two_l114_114030


namespace number_of_zeros_of_h_l114_114890

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := 3 - x^2
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem number_of_zeros_of_h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 = 0 ∧ h x2 = 0 ∧ ∀ x, h x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_zeros_of_h_l114_114890


namespace ways_to_place_balls_in_boxes_l114_114392

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114392


namespace ways_to_distribute_balls_l114_114355

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114355


namespace num_triangles_in_decagon_l114_114280

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l114_114280


namespace problem_statement_l114_114962

-- Define the power function f and the property that it is odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_cond : f 3 < f 2)

-- The statement we need to prove
theorem problem_statement : f (-3) > f (-2) := by
  sorry

end problem_statement_l114_114962


namespace perfect_square_trinomial_t_l114_114751

theorem perfect_square_trinomial_t (a b t : ℝ) :
  (∃ (x y : ℝ), x = a ∧ y = 2 * b ∧ a^2 + (2 * t - 1) * a * b + 4 * b^2 = (x + y)^2) →
  (t = 5 / 2 ∨ t = -3 / 2) :=
by
  sorry

end perfect_square_trinomial_t_l114_114751


namespace ways_to_put_balls_in_boxes_l114_114427

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114427


namespace distinct_balls_boxes_l114_114404

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114404


namespace surveyed_households_count_l114_114839

theorem surveyed_households_count 
  (neither : ℕ) (only_R : ℕ) (both_B : ℕ) (both : ℕ) (h_main : Ξ)
  (H1 : neither = 80)
  (H2 : only_R = 60)
  (H3 : both = 40)
  (H4 : both_B = 3 * both) : 
  neither + only_R + both_B + both = 300 :=
by
  sorry

end surveyed_households_count_l114_114839


namespace minute_hand_travel_distance_l114_114668

theorem minute_hand_travel_distance :
  ∀ (r : ℝ), r = 8 → (45 / 60) * (2 * Real.pi * r) = 12 * Real.pi :=
by
  intros r r_eq
  sorry

end minute_hand_travel_distance_l114_114668


namespace balls_into_boxes_l114_114434

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114434


namespace ways_to_place_balls_in_boxes_l114_114389

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114389


namespace ways_to_put_balls_in_boxes_l114_114422

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114422


namespace remainder_2017_div_89_l114_114181

theorem remainder_2017_div_89 : 2017 % 89 = 59 :=
by
  sorry

end remainder_2017_div_89_l114_114181


namespace find_f2_l114_114960

noncomputable def f : ℝ → ℝ := sorry

axiom function_property : ∀ (x : ℝ), f (2^x) + x * f (2^(-x)) = 1

theorem find_f2 : f 2 = 0 :=
by
  sorry

end find_f2_l114_114960


namespace function_properties_and_k_range_l114_114114

theorem function_properties_and_k_range :
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 ^ x) ∧ (∀ y, y > 0)) ∧
  (∀ k : ℝ, (∃ t : ℝ, t > 0 ∧ (t^2 - 2*t + k = 0)) ↔ (0 < k ∧ k < 1)) :=
by sorry

end function_properties_and_k_range_l114_114114


namespace min_value_of_f_l114_114873

noncomputable def f (x : ℝ) : ℝ :=
  x^2 / (x - 3)

theorem min_value_of_f : ∀ x > 3, f x ≥ 12 :=
by
  sorry

end min_value_of_f_l114_114873


namespace union_sets_l114_114494

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- State the theorem
theorem union_sets : A ∪ B = { x | -1 < x ∧ x ≤ 4 } := 
by
   sorry

end union_sets_l114_114494


namespace triangles_from_decagon_l114_114272

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l114_114272


namespace bacteria_after_time_l114_114092

def initial_bacteria : ℕ := 1
def division_time : ℕ := 20  -- time in minutes for one division
def total_time : ℕ := 180  -- total time in minutes

def divisions := total_time / division_time

theorem bacteria_after_time : (initial_bacteria * 2 ^ divisions) = 512 := by
  exact sorry

end bacteria_after_time_l114_114092


namespace trisha_take_home_pay_l114_114034

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end trisha_take_home_pay_l114_114034


namespace smallest_positive_integer_l114_114187

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l114_114187


namespace remainder_of_sum_modulo_9_l114_114875

theorem remainder_of_sum_modulo_9 : 
  (8230 + 8231 + 8232 + 8233 + 8234 + 8235) % 9 = 0 := by
  sorry

end remainder_of_sum_modulo_9_l114_114875


namespace number_of_triangles_in_decagon_l114_114228

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l114_114228


namespace problem_statement_l114_114148

theorem problem_statement (a b c : ℝ) (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0) (h_condition : a * b + b * c + c * a = 1 / 3) :
  1 / (a^2 - b * c + 1) + 1 / (b^2 - c * a + 1) + 1 / (c^2 - a * b + 1) ≤ 3 :=
by
  sorry

end problem_statement_l114_114148


namespace range_of_a_l114_114912

theorem range_of_a (m a : ℝ) (h1 : m < a) (h2 : m ≤ -1) : a > -1 :=
by sorry

end range_of_a_l114_114912


namespace triangles_from_decagon_l114_114235

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l114_114235


namespace digits_making_number_divisible_by_4_l114_114997

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end digits_making_number_divisible_by_4_l114_114997


namespace largest_subset_size_with_property_l114_114571

def no_four_times_property (S : Finset ℕ) : Prop := 
  ∀ {x y}, x ∈ S → y ∈ S → x = 4 * y → False

noncomputable def max_subset_size : ℕ := 145

theorem largest_subset_size_with_property :
  ∃ (S : Finset ℕ), (∀ x ∈ S, x ≤ 150) ∧ no_four_times_property S ∧ S.card = max_subset_size :=
sorry

end largest_subset_size_with_property_l114_114571


namespace gcd_of_expressions_l114_114047

theorem gcd_of_expressions :
  Nat.gcd (121^2 + 233^2 + 345^2) (120^2 + 232^2 + 346^2) = 5 :=
sorry

end gcd_of_expressions_l114_114047


namespace least_positive_three_digit_multiple_of_8_l114_114983

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l114_114983


namespace weights_in_pile_l114_114676

theorem weights_in_pile (a b c : ℕ) (h1 : a + b + c = 100) (h2 : a + 10 * b + 50 * c = 500) : 
  a = 60 ∧ b = 39 ∧ c = 1 :=
sorry

end weights_in_pile_l114_114676


namespace spacy_subsets_15_l114_114289

def spacy_subsets_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | (k + 5) => spacy_subsets_count (k + 4) + spacy_subsets_count k

theorem spacy_subsets_15 : spacy_subsets_count 15 = 181 :=
sorry

end spacy_subsets_15_l114_114289


namespace difference_between_mean_and_median_l114_114624

namespace MathProof

noncomputable def percentage_72 := 0.12
noncomputable def percentage_82 := 0.30
noncomputable def percentage_87 := 0.18
noncomputable def percentage_91 := 0.10
noncomputable def percentage_96 := 1 - (percentage_72 + percentage_82 + percentage_87 + percentage_91)

noncomputable def num_students := 20
noncomputable def scores := [72, 72, 82, 82, 82, 82, 82, 82, 87, 87, 87, 87, 91, 91, 96, 96, 96, 96, 96, 96]

noncomputable def mean_score : ℚ := (72 * 2 + 82 * 6 + 87 * 4 + 91 * 2 + 96 * 6) / num_students
noncomputable def median_score : ℚ := 87

theorem difference_between_mean_and_median :
  mean_score - median_score = 0.1 := by
  sorry

end MathProof

end difference_between_mean_and_median_l114_114624


namespace regular_decagon_triangle_count_l114_114257

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l114_114257


namespace binom_11_1_l114_114583

theorem binom_11_1 : Nat.choose 11 1 = 11 :=
by
  sorry

end binom_11_1_l114_114583


namespace bounded_sequence_iff_l114_114940

theorem bounded_sequence_iff (x : ℕ → ℝ) (h : ∀ n, x (n + 1) = (n^2 + 1) * x n ^ 2 / (x n ^ 3 + n^2)) :
  (∃ C, ∀ n, x n < C) ↔ (0 < x 0 ∧ x 0 ≤ (Real.sqrt 5 - 1) / 2) ∨ x 0 ≥ 1 := sorry

end bounded_sequence_iff_l114_114940


namespace ten_faucets_fill_50_gallon_in_60_seconds_l114_114732

-- Define the conditions
def five_faucets_fill_tub (faucet_rate : ℝ) : Prop :=
  5 * faucet_rate * 8 = 200

def all_faucets_same_rate (tub_capacity time : ℝ) (num_faucets : ℕ) (faucet_rate : ℝ) : Prop :=
  num_faucets * faucet_rate * time = tub_capacity

-- Define the main theorem to be proven
theorem ten_faucets_fill_50_gallon_in_60_seconds (faucet_rate : ℝ) :
  (∃ faucet_rate, five_faucets_fill_tub faucet_rate) →
  all_faucets_same_rate 50 1 10 faucet_rate →
  10 * faucet_rate * (1 / 60) = 50 :=
by
  sorry

end ten_faucets_fill_50_gallon_in_60_seconds_l114_114732


namespace ball_box_distribution_l114_114464

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114464


namespace polynomial_roots_l114_114301

theorem polynomial_roots :
  ∀ x, (3 * x^4 + 16 * x^3 - 36 * x^2 + 8 * x = 0) ↔ 
       (x = 0 ∨ x = 1 / 3 ∨ x = -3 + 2 * Real.sqrt 17 ∨ x = -3 - 2 * Real.sqrt 17) :=
by
  sorry

end polynomial_roots_l114_114301


namespace remaining_lawn_mowing_l114_114637

-- Definitions based on the conditions in the problem.
def Mary_mowing_time : ℝ := 3  -- Mary can mow the lawn in 3 hours
def John_mowing_time : ℝ := 6  -- John can mow the lawn in 6 hours
def John_work_time : ℝ := 3    -- John works for 3 hours

-- Question: How much of the lawn remains to be mowed?
theorem remaining_lawn_mowing : (Mary_mowing_time = 3) ∧ (John_mowing_time = 6) ∧ (John_work_time = 3) →
  (1 - (John_work_time / John_mowing_time) = 1 / 2) :=
by
  sorry

end remaining_lawn_mowing_l114_114637


namespace fifth_term_of_geometric_sequence_l114_114566

theorem fifth_term_of_geometric_sequence (a r : ℕ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h_a : a = 5) (h_fourth_term : a * r^3 = 405) :
  a * r^4 = 405 := by
  sorry

end fifth_term_of_geometric_sequence_l114_114566


namespace infinite_geometric_series_sum_l114_114295

theorem infinite_geometric_series_sum : 
  (∃ (a r : ℚ), a = 5/4 ∧ r = 1/3) → 
  ∑' n : ℕ, ((5/4 : ℚ) * (1/3 : ℚ) ^ n) = (15/8 : ℚ) :=
by
  sorry

end infinite_geometric_series_sum_l114_114295


namespace units_digit_8th_group_l114_114623

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_8th_group (t k : ℕ) (ht : t = 7) (hk : k = 8) : 
  units_digit (t + k) = 5 := 
by
  -- Proof step will go here.
  sorry

end units_digit_8th_group_l114_114623


namespace circle_area_with_diameter_CD_l114_114792

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_area_with_diameter_CD (C D E : ℝ × ℝ)
  (hC : C = (-1, 2)) (hD : D = (5, -6)) (hE : E = (2, -2))
  (hE_midpoint : E = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  ∃ (A : ℝ), A = 25 * Real.pi :=
by
  -- Define the coordinates of points C and D
  let Cx := -1
  let Cy := 2
  let Dx := 5
  let Dy := -6

  -- Calculate the distance (diameter) between C and D
  let diameter := distance Cx Cy Dx Dy

  -- Calculate the radius of the circle
  let radius := diameter / 2

  -- Calculate the area of the circle
  let area := Real.pi * radius^2

  -- Prove the area is 25π
  use area
  sorry

end circle_area_with_diameter_CD_l114_114792


namespace max_value_of_function_l114_114299

noncomputable def function_to_maximize (x : ℝ) : ℝ :=
  (Real.sin x)^4 + (Real.cos x)^4 + 1 / ((Real.sin x)^2 + (Real.cos x)^2 + 1)

theorem max_value_of_function :
  ∃ x : ℝ, function_to_maximize x = 7 / 4 :=
sorry

end max_value_of_function_l114_114299


namespace balls_into_boxes_l114_114435

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114435


namespace relationship_between_D_and_A_l114_114894

variable {A B C D : Prop}

theorem relationship_between_D_and_A
  (h1 : A → B)
  (h2 : B → C)
  (h3 : D ↔ C) :
  (A → D) ∧ ¬(D → A) :=
by
sorry

end relationship_between_D_and_A_l114_114894


namespace election_winner_votes_difference_l114_114850

theorem election_winner_votes_difference :
  ∃ W S T F, F = 199 ∧ W = S + 53 ∧ W = T + 79 ∧ W + S + T + F = 979 ∧ (W - F = 105) :=
by
  sorry

end election_winner_votes_difference_l114_114850


namespace evaluate_expression_l114_114986

theorem evaluate_expression :
  (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 :=
by
  sorry

end evaluate_expression_l114_114986


namespace scientific_notation_of_tourists_l114_114683

theorem scientific_notation_of_tourists : 
  (23766400 : ℝ) = 2.37664 * 10^7 :=
by 
  sorry

end scientific_notation_of_tourists_l114_114683


namespace smallest_n_l114_114514

theorem smallest_n {n : ℕ} (h1 : n ≡ 4 [MOD 6]) (h2 : n ≡ 3 [MOD 7]) (h3 : n > 10) : n = 52 :=
sorry

end smallest_n_l114_114514


namespace value_of_expression_l114_114532

theorem value_of_expression : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end value_of_expression_l114_114532


namespace distinguish_ball_box_ways_l114_114466

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114466


namespace solve_problem_l114_114618

variable (f : ℝ → ℝ)

axiom f_property : ∀ x : ℝ, f (x + 1) = x^2 - 2 * x

theorem solve_problem : f 2 = -1 :=
by
  sorry

end solve_problem_l114_114618


namespace balls_into_boxes_l114_114396

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114396


namespace average_headcount_l114_114043

theorem average_headcount 
  (h1 : ℕ := 11500) 
  (h2 : ℕ := 11600) 
  (h3 : ℕ := 11300) : 
  (Float.round ((h1 + h2 + h3 : ℕ : Float) / 3) = 11467) :=
sorry

end average_headcount_l114_114043


namespace fraction_simplification_l114_114821

theorem fraction_simplification : 
  (2222 - 2123) ^ 2 / 121 = 81 :=
by
  sorry

end fraction_simplification_l114_114821


namespace new_average_mark_l114_114956

theorem new_average_mark (average_mark : ℕ) (average_excluded : ℕ) (total_students : ℕ) (excluded_students: ℕ)
    (h1 : average_mark = 90)
    (h2 : average_excluded = 45)
    (h3 : total_students = 20)
    (h4 : excluded_students = 2) :
  ((total_students * average_mark - excluded_students * average_excluded) / (total_students - excluded_students)) = 95 := by
  sorry

end new_average_mark_l114_114956


namespace base_conversion_correct_l114_114296

def convert_base_9_to_10 (n : ℕ) : ℕ :=
  3 * 9^2 + 6 * 9^1 + 1 * 9^0

def convert_base_13_to_10 (n : ℕ) (C : ℕ) : ℕ :=
  4 * 13^2 + C * 13^1 + 5 * 13^0

theorem base_conversion_correct :
  convert_base_9_to_10 361 + convert_base_13_to_10 4 12 = 1135 :=
by
  sorry

end base_conversion_correct_l114_114296


namespace transactions_Mabel_l114_114509

variable {M A C J : ℝ}

theorem transactions_Mabel (h1 : A = 1.10 * M)
                          (h2 : C = 2 / 3 * A)
                          (h3 : J = C + 18)
                          (h4 : J = 84) :
  M = 90 :=
by
  sorry

end transactions_Mabel_l114_114509


namespace quadratic_reciprocal_roots_sum_min_positive_c_l114_114112

-- Problem 1
theorem quadratic_reciprocal (m n : ℝ) (h : n ≠ 0) :
  (∃ x1 x2 : ℝ, x1 + x2 = -m ∧ x1 * x2 = n) →
  ∃ x1 x2 : ℝ, (x1 = 1 / x1 ∨ x2 = 1 / x2) ∧
    (∃ p q : ℝ, x1^2 + p * x1 + q = 0 ∧ (n * x1^2 + m * x1 + 1 = 0)) :=
sorry

-- Problem 2
theorem roots_sum (a b : ℝ) (h1 : a^2 - 15 * a - 5 = 0) (h2 : b^2 - 15 * b - 5 = 0) :
  a + b = 15 :=
sorry

-- Problem 3
theorem min_positive_c (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 16) :
  c ≥ 4 :=
sorry

end quadratic_reciprocal_roots_sum_min_positive_c_l114_114112


namespace log_relationship_l114_114306

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem log_relationship :
  a > b ∧ b > c := by
  sorry

end log_relationship_l114_114306


namespace focus_of_parabola_l114_114724

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end focus_of_parabola_l114_114724


namespace josie_total_animals_is_correct_l114_114493

noncomputable def totalAnimals : Nat :=
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  let giraffes := antelopes + 15
  let lions := leopards + giraffes
  let elephants := 3 * lions
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

theorem josie_total_animals_is_correct : totalAnimals = 1308 := by
  sorry

end josie_total_animals_is_correct_l114_114493


namespace problem1_problem2_l114_114694

-- Problem 1
theorem problem1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := sorry

-- Problem 2
theorem problem2 (a : ℝ) (ha : a ≠ 1) : (1 - 1 / a) / ((a^2 - 2 * a + 1) / a) = 1 / (a - 1) := sorry

end problem1_problem2_l114_114694


namespace fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l114_114951

def fractional_eq_solution_1 (x : ℝ) : Prop :=
  x + 5 / x = -6

theorem fractional_eq_solutions_1 : fractional_eq_solution_1 (-1) ∧ fractional_eq_solution_1 (-5) := sorry

def fractional_eq_solution_2 (x : ℝ) : Prop :=
  x - 3 / x = 4

theorem fractional_eq_reciprocal_sum
  (m n : ℝ) (h₀ : fractional_eq_solution_2 m) (h₁ : fractional_eq_solution_2 n) :
  m * n = -3 → m + n = 4 → (1 / m + 1 / n = -4 / 3) := sorry

def fractional_eq_solution_3 (x : ℝ) (a : ℝ) : Prop :=
  x + (a^2 + 2 * a) / (x + 1) = 2 * a + 1

theorem fractional_eq_solution_diff_square (a : ℝ) (h₀ : a ≠ 0)
  (x1 x2 : ℝ) (hx1 : fractional_eq_solution_3 x1 a) (hx2 : fractional_eq_solution_3 x2 a) :
  x1 + 1 = a → x2 + 1 = a + 2 → (x1 - x2) ^ 2 = 4 := sorry

end fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l114_114951


namespace focus_of_parabola_y_eq_4x_sq_l114_114721

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l114_114721


namespace jonathans_and_sisters_total_letters_l114_114137

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l114_114137


namespace ways_to_place_balls_in_boxes_l114_114390

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114390


namespace smallest_positive_integer_l114_114186

-- We define the integers 3003 and 55555 as given in the conditions
def a : ℤ := 3003
def b : ℤ := 55555

-- The main theorem stating the smallest positive integer that can be written in the form ax + by is 1
theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, a * m + b * n = 1 :=
by
  -- We need not provide the proof steps here, just state it
  sorry

end smallest_positive_integer_l114_114186


namespace cm_bc_ratio_l114_114008

noncomputable def ratio_cm_bc (A B C K L M : Type) [Triangle ABC K L M] :=
∀ (A B C K L M) (AK KB AL LC : ℝ) (H1 : AK / KB = 4 / 7) (H2 : AL / LC = 3 / 2),
  ∃ (CM BC : ℝ), CM / BC = 8 / 13

theorem cm_bc_ratio :
  (∀ (A B C K L M : Point) (AK KB AL LC : ℝ), 
    AK / KB = 4 / 7 ∧ AL / LC = 3 / 2 →
    ∃ (CM BC : ℝ), CM / BC = 8 / 13) := by
  sorry

end cm_bc_ratio_l114_114008


namespace terminating_decimals_count_l114_114591

theorem terminating_decimals_count :
  (∀ m : ℤ, 1 ≤ m ∧ m ≤ 999 → ∃ k : ℕ, (m : ℝ) / 1000 = k / (2 ^ 3 * 5 ^ 3)) :=
by
  sorry

end terminating_decimals_count_l114_114591


namespace farey_sequence_mediant_l114_114632

theorem farey_sequence_mediant (a b x y c d : ℕ) (h₁ : a * y < b * x) (h₂ : b * x < y * c) (farey_consecutiveness: bx - ay = 1 ∧ cy - dx = 1) : (x / y) = (a+c) / (b+d) := 
by
  sorry

end farey_sequence_mediant_l114_114632


namespace min_fencing_cost_l114_114172

theorem min_fencing_cost {A B C : ℕ} (h1 : A = 25) (h2 : B = 35) (h3 : C = 40)
  (h_ratio : ∃ (x : ℕ), 3 * x * 4 * x = 8748) : 
  ∃ (total_cost : ℝ), total_cost = 87.75 :=
by
  sorry

end min_fencing_cost_l114_114172


namespace prudence_sleep_4_weeks_equals_200_l114_114880

-- Conditions
def sunday_to_thursday_sleep := 6 
def friday_saturday_sleep := 9 
def nap := 1 

-- Number of days in the mentioned periods per week
def sunday_to_thursday_days := 5
def friday_saturday_days := 2
def nap_days := 2

-- Calculate total sleep per week
def total_sleep_per_week : Nat :=
  (sunday_to_thursday_days * sunday_to_thursday_sleep) +
  (friday_saturday_days * friday_saturday_sleep) +
  (nap_days * nap)

-- Calculate total sleep in 4 weeks
def total_sleep_in_4_weeks : Nat :=
  4 * total_sleep_per_week

theorem prudence_sleep_4_weeks_equals_200 : total_sleep_in_4_weeks = 200 := by
  sorry

end prudence_sleep_4_weeks_equals_200_l114_114880


namespace system1_solution_system2_solution_l114_114165

-- System 1 Definitions
def eq1 (x y : ℝ) : Prop := 3 * x - 2 * y = 9
def eq2 (x y : ℝ) : Prop := 2 * x + 3 * y = 19

-- System 2 Definitions
def eq3 (x y : ℝ) : Prop := (2 * x + 1) / 5 - 1 = (y - 1) / 3
def eq4 (x y : ℝ) : Prop := 2 * (y - x) - 3 * (1 - y) = 6

-- Theorem Statements
theorem system1_solution (x y : ℝ) : eq1 x y ∧ eq2 x y ↔ x = 5 ∧ y = 3 := by
  sorry

theorem system2_solution (x y : ℝ) : eq3 x y ∧ eq4 x y ↔ x = 4 ∧ y = 17 / 5 := by
  sorry

end system1_solution_system2_solution_l114_114165


namespace milk_left_is_correct_l114_114220

def total_morning_milk : ℕ := 365
def total_evening_milk : ℕ := 380
def milk_sold : ℕ := 612
def leftover_milk_from_yesterday : ℕ := 15

def total_milk_left : ℕ :=
  (total_morning_milk + total_evening_milk - milk_sold) + leftover_milk_from_yesterday

theorem milk_left_is_correct : total_milk_left = 148 := by
  sorry

end milk_left_is_correct_l114_114220


namespace parabola_focus_l114_114725

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  (0, 1 / (4 * a)) = (0, 1 / 16) :=
by
  rw [h]
  norm_num
  sorry

end parabola_focus_l114_114725


namespace number_of_triangles_in_regular_decagon_l114_114251

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l114_114251


namespace list_price_proof_l114_114215

-- Define the list price of the item
noncomputable def list_price : ℝ := 33

-- Define the selling price and commission for Alice
def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * alice_selling_price x

-- Define the selling price and commission for Charles
def charles_selling_price (x : ℝ) : ℝ := x - 18
def charles_commission (x : ℝ) : ℝ := 0.18 * charles_selling_price x

-- The main theorem: proving the list price given Alice and Charles receive the same commission
theorem list_price_proof (x : ℝ) (h : alice_commission x = charles_commission x) : x = list_price :=
by 
  sorry

end list_price_proof_l114_114215


namespace number_of_large_boxes_l114_114813

theorem number_of_large_boxes (total_boxes : ℕ) (small_weight large_weight remaining_small remaining_large : ℕ) :
  total_boxes = 62 →
  small_weight = 5 →
  large_weight = 3 →
  remaining_small = 15 →
  remaining_large = 15 →
  ∀ (small_boxes large_boxes : ℕ),
    total_boxes = small_boxes + large_boxes →
    ((large_boxes * large_weight) + (remaining_small * small_weight) = (small_boxes * small_weight) + (remaining_large * large_weight)) →
    large_boxes = 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end number_of_large_boxes_l114_114813


namespace problem_l114_114781

theorem problem (a b : ℝ) : a^6 + b^6 ≥ a^4 * b^2 + a^2 * b^4 := 
by sorry

end problem_l114_114781


namespace measure_of_third_angle_l114_114539

-- Definitions based on given conditions
def angle_sum_of_triangle := 180
def angle1 := 30
def angle2 := 60

-- Problem Statement: Prove the third angle (angle3) in a triangle is 90 degrees
theorem measure_of_third_angle (angle_sum : ℕ := angle_sum_of_triangle) 
  (a1 : ℕ := angle1) (a2 : ℕ := angle2) : (angle_sum - (a1 + a2)) = 90 :=
by
  sorry

end measure_of_third_angle_l114_114539


namespace number_of_triangles_in_decagon_l114_114287

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l114_114287


namespace num_triangles_from_decagon_l114_114278

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l114_114278


namespace difference_in_number_of_girls_and_boys_l114_114019

def ratio_boys_girls (b g : ℕ) : Prop := b * 3 = g * 2

def total_students (b g : ℕ) : Prop := b + g = 30

theorem difference_in_number_of_girls_and_boys
  (b g : ℕ)
  (h1 : ratio_boys_girls b g)
  (h2 : total_students b g) :
  g - b = 6 :=
sorry

end difference_in_number_of_girls_and_boys_l114_114019


namespace mutually_exclusive_probability_zero_l114_114476

theorem mutually_exclusive_probability_zero {A B : Prop} (p1 p2 : ℝ) 
  (hA : 0 ≤ p1 ∧ p1 ≤ 1) 
  (hB : 0 ≤ p2 ∧ p2 ≤ 1) 
  (hAB : A ∧ B → False) : 
  (A ∧ B) = False :=
by
  sorry

end mutually_exclusive_probability_zero_l114_114476


namespace neil_halloween_candy_l114_114328

-- Definitions based on the conditions
def maggie_collected : ℕ := 50
def percentage_increase_harper : ℝ := 0.30
def percentage_increase_neil : ℝ := 0.40

-- Define the extra candy Harper collected
def extra_candy_harper (m : ℕ) (p : ℝ) : ℕ := (p * m).nat_abs

-- Define the total candy Harper collected
def harper_collected (m : ℕ) (p : ℝ) : ℕ := m + extra_candy_harper m p

-- Define the extra candy Neil collected
def extra_candy_neil (h : ℕ) (p : ℝ) : ℕ := (p * h).nat_abs

-- Define the total candy Neil collected
def neil_collected (h : ℕ) (p : ℝ) : ℕ := h + extra_candy_neil h p

-- Problem statement
theorem neil_halloween_candy : neil_collected (harper_collected maggie_collected percentage_increase_harper) percentage_increase_neil = 91 :=
by
  sorry

end neil_halloween_candy_l114_114328


namespace cos_diff_simplify_l114_114953

theorem cos_diff_simplify (x : ℝ) (y : ℝ) (h1 : x = Real.cos (Real.pi / 10)) (h2 : y = Real.cos (3 * Real.pi / 10)) : 
  x - y = 4 * x * (1 - x^2) := 
sorry

end cos_diff_simplify_l114_114953


namespace packages_per_box_l114_114966

theorem packages_per_box (P : ℕ) (h1 : 192 > 0) (h2 : 2 > 0) (total_soaps : 2304 > 0) (h : 2 * P * 192 = 2304) : P = 6 :=
by
  sorry

end packages_per_box_l114_114966


namespace additional_books_acquired_l114_114845

def original_stock : ℝ := 40.0
def shelves_used : ℕ := 15
def books_per_shelf : ℝ := 4.0

theorem additional_books_acquired :
  (shelves_used * books_per_shelf) - original_stock = 20.0 :=
by
  sorry

end additional_books_acquired_l114_114845


namespace min_value_expression_l114_114903

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem min_value_expression (a b : ℝ) (h1 : b > 0) (h2 : f a b 1 = 3) :
  ∃ x, x = (4 / (a - 1) + 1 / b) ∧ x = 9 / 2 :=
by
  sorry

end min_value_expression_l114_114903


namespace part1_max_value_l114_114602

variable (f : ℝ → ℝ)
def is_maximum (y : ℝ) := ∀ x : ℝ, f x ≤ y

theorem part1_max_value (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + m*x + 1) :
  m = 0 → (exists y, is_maximum f y ∧ y = 1) := 
sorry

end part1_max_value_l114_114602


namespace polynomial_value_sum_l114_114737

theorem polynomial_value_sum
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (Hf : ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d)
  (H1 : f 1 = 1) (H2 : f 2 = 2) (H3 : f 3 = 3) :
  f 0 + f 4 = 28 :=
sorry

end polynomial_value_sum_l114_114737


namespace complex_magnitude_add_reciprocals_l114_114503

open Complex

theorem complex_magnitude_add_reciprocals
  (z w : ℂ)
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hz_plus_w : Complex.abs (z + w) = 6) :
  Complex.abs (1 / z + 1 / w) = 3 / 4 := by
  sorry

end complex_magnitude_add_reciprocals_l114_114503


namespace prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l114_114595

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4 * x + a) / (x^2 + 1)

-- 1. Prove that a = 0 given that f(x) is an odd function
theorem prove_a_eq_0 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = - f x a) : a = 0 := sorry

-- 2. Prove that f(x) = 4x / (x^2 + 1) is monotonically decreasing on [1, +∞) for x > 0
theorem prove_monotonic_decreasing (x : ℝ) (hx : x > 0) :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (f x1 0) > (f x2 0) := sorry

-- 3. Prove that |f(x1) - f(x2)| ≤ m for all x1, x2 ∈ R implies m ≥ 4
theorem prove_m_ge_4 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, |f x1 0 - f x2 0| ≤ m) : m ≥ 4 := sorry

end prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l114_114595


namespace arrange_polynomial_l114_114688

theorem arrange_polynomial :
  ∀ (x y : ℝ), 2 * x^3 * y - 4 * y^2 + 5 * x^2 = 5 * x^2 + 2 * x^3 * y - 4 * y^2 :=
by
  sorry

end arrange_polynomial_l114_114688


namespace find_a_10_l114_114746

-- We define the arithmetic sequence and sum properties
def arithmetic_seq (a_1 d : ℚ) (a_n : ℕ → ℚ) :=
  ∀ n, a_n n = a_1 + d * n

def sum_arithmetic_seq (a : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  ∀ n, S_n n = n * (a 1 + a n) / 2

-- Conditions given in the problem
def given_conditions (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  arithmetic_seq a_1 1 a_n ∧ sum_arithmetic_seq a_n S_n ∧ S_n 6 = 4 * S_n 3

-- The theorem to prove
theorem find_a_10 (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) 
  (h : given_conditions a_1 a_n S_n) : a_n 10 = 19 / 2 :=
by sorry

end find_a_10_l114_114746


namespace gabby_l114_114883

-- Define variables and conditions
variables (watermelons peaches plums total_fruit : ℕ)
variables (h_watermelons : watermelons = 1)
variables (h_peaches : peaches = watermelons + 12)
variables (h_plums : plums = 3 * peaches)
variables (h_total_fruit : total_fruit = watermelons + peaches + plums)

-- The theorem we aim to prove
theorem gabby's_fruit_count (h_watermelons : watermelons = 1)
                           (h_peaches : peaches = watermelons + 12)
                           (h_plums : plums = 3 * peaches)
                           (h_total_fruit : total_fruit = watermelons + peaches + plums) :
  total_fruit = 53 := by
sorry

end gabby_l114_114883


namespace gear_revolutions_difference_l114_114856

noncomputable def gear_revolution_difference (t : ℕ) : ℕ :=
  let p := 10 * t
  let q := 40 * t
  q - p

theorem gear_revolutions_difference (t : ℕ) : gear_revolution_difference t = 30 * t :=
by
  sorry

end gear_revolutions_difference_l114_114856


namespace ball_in_boxes_l114_114345

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114345


namespace probability_one_hits_l114_114160

theorem probability_one_hits (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.6) :
  (P_A * (1 - P_B) + (1 - P_A) * P_B) = 0.48 :=
by
  sorry

end probability_one_hits_l114_114160


namespace min_button_presses_l114_114671

theorem min_button_presses :
  ∃ (a b : ℤ), 9 * a - 20 * b = 13 ∧  a + b = 24 := 
by
  sorry

end min_button_presses_l114_114671


namespace polygon_sides_l114_114898

theorem polygon_sides (h : ∀ (θ : ℕ), θ = 108) : ∃ n : ℕ, n = 5 :=
by
  sorry

end polygon_sides_l114_114898


namespace red_marbles_initial_count_l114_114763

theorem red_marbles_initial_count (r g : ℕ) 
  (h1 : 3 * r = 5 * g)
  (h2 : 4 * (r - 18) = g + 27) :
  r = 29 :=
sorry

end red_marbles_initial_count_l114_114763


namespace exists_digit_sum_div_11_in_39_succ_nums_l114_114794

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_digit_sum_div_11_in_39_succ_nums (n : ℕ) :
  ∃ k, k ∈ list.range' n 39 ∧ digit_sum k % 11 = 0 :=
by
  -- The proof would go here
  sorry

end exists_digit_sum_div_11_in_39_succ_nums_l114_114794


namespace coins_in_box_l114_114832

theorem coins_in_box (n : ℕ) 
    (h1 : n % 8 = 7) 
    (h2 : n % 7 = 5) : 
    n = 47 ∧ (47 % 9 = 2) :=
sorry

end coins_in_box_l114_114832


namespace inequality_solution_empty_l114_114479

theorem inequality_solution_empty {a : ℝ} :
  (∀ x : ℝ, ¬ (|x+2| + |x-1| < a)) ↔ a ≤ 3 :=
by
  sorry

end inequality_solution_empty_l114_114479


namespace number_of_triangles_in_regular_decagon_l114_114230

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l114_114230


namespace balls_into_boxes_l114_114331

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114331


namespace equilateral_triangle_min_perimeter_l114_114793

theorem equilateral_triangle_min_perimeter (a b c : ℝ) (S : ℝ) :
  let p := (a + b + c) / 2 in
  let area := sqrt (p * (p - a) * (p - b) * (p - c)) in
  area = S →
  ∀ a b c, p = (a + b + c) / 2 →
  sqrt (p * (p - a) * (p - b) * (p - c)) = S →
  a = b = c :=
by sorry

end equilateral_triangle_min_perimeter_l114_114793


namespace passenger_cars_count_l114_114071

theorem passenger_cars_count (P C : ℕ) 
    (h₁ : C = (1 / 2 : ℚ) * P + 3) 
    (h₂ : P + C + 2 = 71) : P = 44 :=
begin
  sorry
end

end passenger_cars_count_l114_114071


namespace minimum_reciprocal_sum_l114_114758

theorem minimum_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  (∃ z : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → z ≤ (1 / x + 2 / y)) ∧ z = 35 / 6) :=
  sorry

end minimum_reciprocal_sum_l114_114758


namespace paul_coins_difference_l114_114159

/-- Paul owes Paula 145 cents and has a pocket full of 10-cent coins, 
20-cent coins, and 50-cent coins. Prove that the difference between 
the largest and smallest number of coins he can use to pay her is 9. -/
theorem paul_coins_difference :
  ∃ min_coins max_coins : ℕ, 
    (min_coins = 5 ∧ max_coins = 14) ∧ (max_coins - min_coins = 9) :=
by
  sorry

end paul_coins_difference_l114_114159


namespace total_hours_played_l114_114945

-- Definitions based on conditions
def Nathan_hours_per_day : ℕ := 3
def Nathan_weeks : ℕ := 2
def days_per_week : ℕ := 7

def Tobias_hours_per_day : ℕ := 5
def Tobias_weeks : ℕ := 1

-- Calculating total hours
def Nathan_total_hours := Nathan_hours_per_day * days_per_week * Nathan_weeks
def Tobias_total_hours := Tobias_hours_per_day * days_per_week * Tobias_weeks

-- Theorem statement
theorem total_hours_played : Nathan_total_hours + Tobias_total_hours = 77 := by
  -- Proof would go here
  sorry

end total_hours_played_l114_114945


namespace number_of_triangles_in_decagon_l114_114268

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l114_114268


namespace factor_expression_l114_114882

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l114_114882


namespace intersection_S_T_eq_l114_114323

def S : Set ℝ := { x | (x - 2) * (x - 3) ≥ 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T_eq : (S ∩ T) = { x | (0 < x ∧ x ≤ 2) ∨ (x ≥ 3) } :=
by
  sorry

end intersection_S_T_eq_l114_114323


namespace decagon_triangle_count_l114_114260

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l114_114260


namespace balls_in_boxes_l114_114374

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114374


namespace sum_of_arithmetic_sequence_has_remainder_2_l114_114656

def arithmetic_sequence_remainder : ℕ := 
  let first_term := 1
  let common_difference := 6
  let last_term := 259
  -- Calculate number of terms
  let n := (last_term + 5) / common_difference
  -- Sum of remainders of each term when divided by 6
  let sum_of_remainders := n * 1
  -- The remainder when this sum is divided by 6
  sum_of_remainders % 6 
theorem sum_of_arithmetic_sequence_has_remainder_2 : 
  arithmetic_sequence_remainder = 2 := by 
  sorry

end sum_of_arithmetic_sequence_has_remainder_2_l114_114656


namespace ways_to_distribute_balls_l114_114348

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114348


namespace num_triangles_from_decagon_l114_114240

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l114_114240


namespace journey_speed_l114_114998

theorem journey_speed
  (v : ℝ) -- Speed during the first four hours
  (total_distance : ℝ) (total_time : ℝ) -- Total distance and time of the journey
  (distance_part1 : ℝ) (time_part1 : ℝ) -- Distance and time for the first part of journey
  (distance_part2 : ℝ) (time_part2 : ℝ) -- Distance and time for the second part of journey
  (speed_part2 : ℝ) : -- Speed during the second part of journey
  total_distance = 24 ∧ total_time = 8 ∧ speed_part2 = 2 ∧ 
  time_part1 = 4 ∧ time_part2 = 4 ∧ 
  distance_part1 = v * time_part1 ∧ distance_part2 = speed_part2 * time_part2 →
  v = 4 := 
by
  sorry

end journey_speed_l114_114998


namespace remaining_amount_is_correct_l114_114615

-- Define the original price based on the deposit paid
def original_price : ℝ := 1500

-- Define the discount percentage
def discount_percentage : ℝ := 0.05

-- Define the sales tax percentage
def tax_percentage : ℝ := 0.075

-- Define the deposit already paid
def deposit_paid : ℝ := 150

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_percentage)

-- Define the sales tax amount
def sales_tax : ℝ := discounted_price * tax_percentage

-- Define the final cost after adding sales tax
def final_cost : ℝ := discounted_price + sales_tax

-- Define the remaining amount to be paid
def remaining_amount : ℝ := final_cost - deposit_paid

-- The statement we need to prove
theorem remaining_amount_is_correct : remaining_amount = 1381.875 :=
by
  -- We'd normally write the proof here, but that's not required for this task.
  sorry

end remaining_amount_is_correct_l114_114615


namespace induction_inequality_term_added_l114_114551

theorem induction_inequality_term_added (k : ℕ) (h : k > 0) :
  let termAdded := (1 / (2 * (k + 1) - 1 : ℝ)) + (1 / (2 * (k + 1) : ℝ)) - (1 / (k + 1 : ℝ))
  ∃ h : ℝ, termAdded = h :=
by
  sorry

end induction_inequality_term_added_l114_114551


namespace final_position_is_negative_one_total_revenue_is_118_yuan_l114_114066

-- Define the distances
def distances : List Int := [9, -3, -6, 4, -8, 6, -3, -6, -4, 10]

-- Define the taxi price per kilometer
def price_per_km : Int := 2

-- Theorem to prove the final position of the taxi relative to Wu Zhong
theorem final_position_is_negative_one : 
  List.sum distances = -1 :=
by 
  sorry -- Proof omitted

-- Theorem to prove the total revenue for the afternoon
theorem total_revenue_is_118_yuan : 
  price_per_km * List.sum (List.map Int.natAbs distances) = 118 :=
by
  sorry -- Proof omitted

end final_position_is_negative_one_total_revenue_is_118_yuan_l114_114066


namespace chandra_valid_pairings_l114_114085

noncomputable def valid_pairings (total_items : Nat) (invalid_pairing : Nat) : Nat :=
total_items * total_items - invalid_pairing

theorem chandra_valid_pairings : valid_pairings 5 1 = 24 := by
  sorry

end chandra_valid_pairings_l114_114085


namespace proof_angles_constant_l114_114759

noncomputable theory
open_locale classical

def const_sum_angles (O1 O2 : Circle) (A A' : Point) (B C D E : Point) : Prop :=
(∃ (line : Line), is_on_line B line ∧ is_on_line C line ∧ is_on_circle B O1 ∧ is_on_circle C O1 ∧
 is_on_line D line ∧ is_on_line E line ∧ is_on_circle D O2 ∧ is_on_circle E O2 ∧
 (is_collinear C D B E ∨ is_collinear B E C D) → (angle B A D + angle C A E = const))

def const_abs_diff_angles (O1 O2 : Circle) (A A' : Point) (B C D E : Point) : Prop :=
(∃ (line : Line), is_on_line B line ∧ is_on_line C line ∧ is_on_circle B O1 ∧ is_on_circle C O1 ∧
 is_on_line D line ∧ is_on_line E line ∧ is_on_circle D O2 ∧ is_on_circle E O2 ∧
 (¬is_collinear C D B E ∧ ¬is_collinear B E C D) → (|angle B A D - angle C A E| = const))

theorem proof_angles_constant (O1 O2 : Circle) (A A' : Point) : 
  ∀ B C D E, const_sum_angles O1 O2 A A' B C D E ∨ const_abs_diff_angles O1 O2 A A' B C D E :=
by sorry

end proof_angles_constant_l114_114759


namespace trisha_take_home_pay_l114_114039

theorem trisha_take_home_pay
  (hourly_pay : ℝ := 15)
  (hours_per_week : ℝ := 40)
  (weeks_per_year : ℝ := 52)
  (withholding_percentage : ℝ := 0.20) :
  let annual_gross_pay := hourly_pay * hours_per_week * weeks_per_year,
      amount_withheld := annual_gross_pay * withholding_percentage,
      annual_take_home_pay := annual_gross_pay - amount_withheld
  in annual_take_home_pay = 24960 := by
    sorry

end trisha_take_home_pay_l114_114039


namespace number_of_triangles_in_decagon_l114_114245

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l114_114245


namespace decagon_triangle_count_l114_114259

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l114_114259


namespace ways_to_distribute_balls_l114_114414

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114414


namespace pizza_toppings_problem_l114_114830

theorem pizza_toppings_problem
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (pepperoni_mushroom_slices : ℕ)
  (pepperoni_olive_slices : ℕ)
  (mushroom_olive_slices : ℕ)
  (pepperoni_mushroom_olive_slices : ℕ) :
  total_slices = 20 →
  pepperoni_slices = 12 →
  mushroom_slices = 14 →
  olive_slices = 12 →
  pepperoni_mushroom_slices = 8 →
  pepperoni_olive_slices = 8 →
  mushroom_olive_slices = 8 →
  total_slices = pepperoni_slices + mushroom_slices + olive_slices
    - pepperoni_mushroom_slices - pepperoni_olive_slices - mushroom_olive_slices
    + pepperoni_mushroom_olive_slices →
  pepperoni_mushroom_olive_slices = 6 :=
by
  intros
  sorry

end pizza_toppings_problem_l114_114830


namespace eggs_total_l114_114608

-- Definitions based on the conditions
def breakfast_eggs : Nat := 2
def lunch_eggs : Nat := 3
def dinner_eggs : Nat := 1

-- Theorem statement
theorem eggs_total : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  -- This part will be filled in with proof steps, but it's omitted here
  sorry

end eggs_total_l114_114608


namespace ball_in_boxes_l114_114344

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114344


namespace construct_right_triangle_l114_114596

noncomputable def quadrilateral (A B C D : Type) : Prop :=
∃ (AB BC CA : ℝ), 
AB = BC ∧ BC = CA ∧ 
∃ (angle_D : ℝ), 
angle_D = 30

theorem construct_right_triangle (A B C D : Type) (angle_D: ℝ) (AB BC CA : ℝ) 
    (h1 : AB = BC) (h2 : BC = CA) (h3 : angle_D = 30) : 
    exists DA DB DC : ℝ, (DA * DA) + (DC * DC) = (AD * AD) :=
by sorry

end construct_right_triangle_l114_114596


namespace boys_neither_happy_nor_sad_l114_114946

theorem boys_neither_happy_nor_sad : 
  (∀ children total happy sad neither boys girls happy_boys sad_girls : ℕ,
    total = 60 →
    happy = 30 →
    sad = 10 →
    neither = 20 →
    boys = 19 →
    girls = 41 →
    happy_boys = 6 →
    sad_girls = 4 →
    (boys - (happy_boys + (sad - sad_girls))) = 7) :=
by
  intros children total happy sad neither boys girls happy_boys sad_girls
  sorry

end boys_neither_happy_nor_sad_l114_114946


namespace find_value_of_pow_function_l114_114739

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem find_value_of_pow_function :
  (∃ α : ℝ, power_function α 4 = 1/2) →
  ∃ α : ℝ, power_function α (1/4) = 2 :=
by
  sorry

end find_value_of_pow_function_l114_114739


namespace find_square_sum_l114_114914

theorem find_square_sum (x y z : ℝ)
  (h1 : x^2 - 6 * y = 10)
  (h2 : y^2 - 8 * z = -18)
  (h3 : z^2 - 10 * x = -40) :
  x^2 + y^2 + z^2 = 50 :=
sorry

end find_square_sum_l114_114914


namespace circle_area_l114_114804

/--
Given the polar equation of a circle r = -4 * cos θ + 8 * sin θ,
prove that the area of the circle is 20π.
-/
theorem circle_area (θ : ℝ) (r : ℝ) (cos : ℝ → ℝ) (sin : ℝ → ℝ) 
  (h_eq : ∀ θ : ℝ, r = -4 * cos θ + 8 * sin θ) : 
  ∃ A : ℝ, A = 20 * Real.pi :=
by
  sorry

end circle_area_l114_114804


namespace factorization_of_cubic_polynomial_l114_114866

-- Define the elements and the problem
variable (a : ℝ)

theorem factorization_of_cubic_polynomial :
  a^3 - 3 * a = a * (a + Real.sqrt 3) * (a - Real.sqrt 3) := by
  sorry

end factorization_of_cubic_polynomial_l114_114866


namespace balls_into_boxes_l114_114437

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l114_114437


namespace number_99_in_column_4_l114_114849

-- Definition of the arrangement rule
def column_of (num : ℕ) : ℕ :=
  ((num % 10) + 4) / 2 % 5 + 1

theorem number_99_in_column_4 : 
  column_of 99 = 4 :=
by
  sorry

end number_99_in_column_4_l114_114849


namespace calc_x_squared_plus_5xy_plus_y_squared_l114_114106

theorem calc_x_squared_plus_5xy_plus_y_squared 
  (x y : ℝ) 
  (h1 : x * y = 4)
  (h2 : x - y = 5) :
  x^2 + 5 * x * y + y^2 = 53 :=
by 
  sorry

end calc_x_squared_plus_5xy_plus_y_squared_l114_114106


namespace atomic_number_l114_114749

theorem atomic_number (mass_number : ℕ) (neutrons : ℕ) (protons : ℕ) :
  mass_number = 288 →
  neutrons = 169 →
  (protons = mass_number - neutrons) →
  protons = 119 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end atomic_number_l114_114749


namespace number_of_ways_to_select_books_l114_114535

theorem number_of_ways_to_select_books :
  let bag1 := 4
  let bag2 := 5
  bag1 * bag2 = 20 :=
by
  sorry

end number_of_ways_to_select_books_l114_114535


namespace relationship_between_x_and_y_l114_114594

theorem relationship_between_x_and_y (a b : ℝ) (x y : ℝ)
  (h1 : x = a^2 + b^2 + 20)
  (h2 : y = 4 * (2 * b - a)) :
  x ≥ y :=
by 
-- we need to prove x ≥ y
sorry

end relationship_between_x_and_y_l114_114594


namespace michael_earnings_l114_114152

theorem michael_earnings :
  let price_extra_large := 150
  let price_large := 100
  let price_medium := 80
  let price_small := 60
  let qty_extra_large := 3
  let qty_large := 5
  let qty_medium := 8
  let qty_small := 10
  let discount_large := 0.10
  let tax := 0.05
  let cost_materials := 300
  let commission_fee := 0.10

  let total_initial_sales := (qty_extra_large * price_extra_large) + 
                             (qty_large * price_large) + 
                             (qty_medium * price_medium) + 
                             (qty_small * price_small)

  let discount_on_large := discount_large * (qty_large * price_large)
  let sales_after_discount := total_initial_sales - discount_on_large

  let sales_tax := tax * sales_after_discount
  let total_collected := sales_after_discount + sales_tax

  let commission := commission_fee * sales_after_discount
  let total_deductions := cost_materials + commission
  let earnings := total_collected - total_deductions

  earnings = 1733 :=
by
  sorry

end michael_earnings_l114_114152


namespace maximize_revenue_l114_114075

-- Define the conditions
def price (p : ℝ) := p ≤ 30
def toys_sold (p : ℝ) : ℝ := 150 - 4 * p
def revenue (p : ℝ) := p * (toys_sold p)

-- State the theorem to solve the problem
theorem maximize_revenue : ∃ p : ℝ, price p ∧ 
  (∀ q : ℝ, price q → revenue q ≤ revenue p) ∧ p = 18.75 :=
by {
  sorry
}

end maximize_revenue_l114_114075


namespace minimum_minutes_for_planB_cheaper_l114_114078

-- Define the costs for Plan A and Plan B as functions of minutes
def planACost (x : Nat) : Nat := 1500 + 12 * x
def planBCost (x : Nat) : Nat := 3000 + 6 * x

-- Statement to prove
theorem minimum_minutes_for_planB_cheaper : 
  ∃ x : Nat, (planBCost x < planACost x) ∧ ∀ y : Nat, y < x → planBCost y ≥ planACost y :=
by
  sorry

end minimum_minutes_for_planB_cheaper_l114_114078


namespace field_trip_vans_l114_114944

-- Define the number of students and adults
def students := 12
def adults := 3

-- Define the capacity of each van
def van_capacity := 5

-- Total number of people
def total_people := students + adults

-- Calculate the number of vans needed
def vans_needed := (total_people + van_capacity - 1) / van_capacity  -- For rounding up division

theorem field_trip_vans : vans_needed = 3 :=
by
  -- Calculation and proof would go here
  sorry

end field_trip_vans_l114_114944


namespace find_n_l114_114893

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (n : ℕ)

def isArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sumTo (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem find_n 
  (h_arith : isArithmeticSeq a)
  (h_a2 : a 2 = 2) 
  (h_S_diff : ∀ n, n > 3 → S n - S (n - 3) = 54)
  (h_Sn : S n = 100)
  : n = 10 := 
by
  sorry

end find_n_l114_114893


namespace total_amount_spent_l114_114083

def cost_of_soft_drink : ℕ := 2
def cost_per_candy_bar : ℕ := 5
def number_of_candy_bars : ℕ := 5

theorem total_amount_spent : cost_of_soft_drink + cost_per_candy_bar * number_of_candy_bars = 27 := by
  sorry

end total_amount_spent_l114_114083


namespace department_store_earnings_l114_114836

theorem department_store_earnings :
  let original_price : ℝ := 1000000
  let discount_rate : ℝ := 0.1
  let prizes := [ (5, 1000), (10, 500), (20, 200), (40, 100), (5000, 10) ]
  let A_earnings := original_price * (1 - discount_rate)
  let total_prizes := prizes.foldl (fun sum (count, amount) => sum + count * amount) 0
  let B_earnings := original_price - total_prizes
  (B_earnings - A_earnings) >= 32000 := by
  sorry

end department_store_earnings_l114_114836


namespace value_of_k_l114_114651

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem value_of_k
  (a d : ℝ)
  (a1_eq_1 : a = 1)
  (sum_9_eq_sum_4 : 9/2 * (2*a + 8*d) = 4/2 * (2*a + 3*d))
  (k : ℕ)
  (a_k_plus_a_4_eq_0 : arithmetic_sequence a d k + arithmetic_sequence a d 4 = 0) :
  k = 10 :=
by
  sorry

end value_of_k_l114_114651


namespace average_student_headcount_l114_114044

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end average_student_headcount_l114_114044


namespace trig_identity_l114_114310

-- Define the given condition
def tan_half (α : ℝ) : Prop := Real.tan (α / 2) = 2

-- The main statement we need to prove
theorem trig_identity (α : ℝ) (h : tan_half α) : (1 + Real.cos α) / (Real.sin α) = 1 / 2 :=
  by
  sorry

end trig_identity_l114_114310


namespace weight_of_each_bag_is_correct_l114_114838

noncomputable def weightOfEachBag
    (days1 : ℕ := 60)
    (consumption1 : ℕ := 2)
    (days2 : ℕ := 305)
    (consumption2 : ℕ := 4)
    (ouncesPerPound : ℕ := 16)
    (numberOfBags : ℕ := 17) : ℝ :=
        let totalOunces := (days1 * consumption1) + (days2 * consumption2)
        let totalPounds := totalOunces / ouncesPerPound
        totalPounds / numberOfBags

theorem weight_of_each_bag_is_correct :
  weightOfEachBag = 4.93 :=
by
  sorry

end weight_of_each_bag_is_correct_l114_114838


namespace Lance_must_read_today_l114_114932

def total_pages : ℕ := 100
def pages_read_yesterday : ℕ := 35
def pages_read_tomorrow : ℕ := 27

noncomputable def pages_read_today : ℕ :=
  pages_read_yesterday - 5

noncomputable def pages_left_today : ℕ :=
  total_pages - (pages_read_yesterday + pages_read_today + pages_read_tomorrow)

theorem Lance_must_read_today :
  pages_read_today + pages_left_today = 38 :=
by 
  rw [pages_read_today, pages_left_today, pages_read_yesterday, pages_read_tomorrow, total_pages]
  simp
  sorry

end Lance_must_read_today_l114_114932


namespace abc_inequality_l114_114490

theorem abc_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l114_114490


namespace floyd_infinite_jumps_l114_114302

def sum_of_digits (n: Nat) : Nat := 
  n.digits 10 |>.sum 

noncomputable def jumpable (a b: Nat) : Prop := 
  b > a ∧ b ≤ 2 * a 

theorem floyd_infinite_jumps :
  ∃ f : ℕ → ℕ, 
    (∀ n : ℕ, jumpable (f n) (f (n + 1))) ∧
    (∀ m n : ℕ, m ≠ n → sum_of_digits (f m) ≠ sum_of_digits (f n)) :=
sorry

end floyd_infinite_jumps_l114_114302


namespace distinguish_ball_box_ways_l114_114467

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114467


namespace focus_of_parabola_y_eq_4x_sq_l114_114722

noncomputable def parabola_focus : (ℝ × ℝ) :=
  let f := (0 : ℝ, 1 / 16 : ℝ)
  in f

theorem focus_of_parabola_y_eq_4x_sq :
  (0, 1 / 16) = parabola_focus := by
  unfold parabola_focus
  sorry

end focus_of_parabola_y_eq_4x_sq_l114_114722


namespace calculate_polynomial_value_l114_114305

theorem calculate_polynomial_value (a a1 a2 a3 a4 a5 : ℝ) : 
  (∀ x : ℝ, (1 - x)^2 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) → 
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by 
  intro h
  sorry

end calculate_polynomial_value_l114_114305


namespace inequality_flip_l114_114788

theorem inequality_flip (a b : ℤ) (c : ℤ) (h1 : a < b) (h2 : c < 0) : 
  c * a > c * b :=
sorry

end inequality_flip_l114_114788


namespace minute_hand_length_l114_114947

theorem minute_hand_length 
  (arc_length : ℝ) (r : ℝ) (h : arc_length = 20 * (2 * Real.pi / 60) * r) :
  r = 1/2 :=
  sorry

end minute_hand_length_l114_114947


namespace commute_time_abs_diff_l114_114840

theorem commute_time_abs_diff (x y : ℝ) 
  (h1 : (x + y + 10 + 11 + 9)/5 = 10) 
  (h2 : (1/5 : ℝ) * ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) = 2) : 
  |x - y| = 4 :=
by
  sorry

end commute_time_abs_diff_l114_114840


namespace quadrilateral_pyramid_volume_l114_114589

theorem quadrilateral_pyramid_volume (h Q : ℝ) : 
  ∃ V : ℝ, V = (2 / 3 : ℝ) * h * (Real.sqrt (h^2 + 4 * Q^2) - h^2) :=
by
  sorry

end quadrilateral_pyramid_volume_l114_114589


namespace balls_into_boxes_l114_114332

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114332


namespace cristina_catches_up_l114_114508

theorem cristina_catches_up
  (t : ℝ)
  (cristina_speed : ℝ := 5)
  (nicky_speed : ℝ := 3)
  (nicky_head_start : ℝ := 54)
  (distance_cristina : ℝ := cristina_speed * t)
  (distance_nicky : ℝ := nicky_head_start + nicky_speed * t) :
  distance_cristina = distance_nicky → t = 27 :=
by
  intros h
  sorry

end cristina_catches_up_l114_114508


namespace minimize_dot_product_l114_114117

def vector := ℝ × ℝ

def OA : vector := (2, 2)
def OB : vector := (4, 1)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def AP (P : vector) : vector :=
  (P.1 - OA.1, P.2 - OA.2)

def BP (P : vector) : vector :=
  (P.1 - OB.1, P.2 - OB.2)

def is_on_x_axis (P : vector) : Prop :=
  P.2 = 0

theorem minimize_dot_product :
  ∃ (P : vector), is_on_x_axis P ∧ dot_product (AP P) (BP P) = ( (P.1 - 3) ^ 2 + 1) ∧ P = (3, 0) :=
by
  sorry

end minimize_dot_product_l114_114117


namespace simplify_cos18_minus_cos54_l114_114952

noncomputable def cos_54 : ℝ := 2 * (cos 27)^2 - 1
noncomputable def cos_27 : ℝ := sqrt ((1 + cos_54) / 2)
noncomputable def cos_18 : ℝ := 1 - 2 * (sin 9)^2
noncomputable def sin_9 : ℝ := sqrt ((1 - cos_18) / 2)

theorem simplify_cos18_minus_cos54 : (cos 18 - cos 54) = 0 :=
by
  have h_cos_54 : cos 54 = cos_54 := by sorry
  have h_cos_27 : cos 27 = cos_27 := by sorry
  have h_cos_18 : cos 18 = cos_18 := by sorry
  have h_sin_9 : sin 9 = sin_9 := by sorry
  sorry

end simplify_cos18_minus_cos54_l114_114952


namespace compound_interest_l114_114687

noncomputable def final_amount (P : ℕ) (r : ℚ) (t : ℕ) :=
  P * ((1 : ℚ) + r) ^ t

theorem compound_interest : 
  final_amount 20000 0.20 10 = 123834.73 := 
by 
  sorry

end compound_interest_l114_114687


namespace prudence_sleep_in_4_weeks_l114_114881

theorem prudence_sleep_in_4_weeks :
  let hours_per_night_from_sun_to_thu := 6
      nights_from_sun_to_thu := 5
      hours_per_night_fri_and_sat := 9
      nights_fri_and_sat := 2
      nap_hours_per_day_on_sat_and_sun := 1
      nap_days_on_sat_and_sun := 2
      weeks := 4
  in
  (nights_from_sun_to_thu * hours_per_night_from_sun_to_thu +
   nights_fri_and_sat * hours_per_night_fri_and_sat +
   nap_days_on_sat_and_sun * nap_hours_per_day_on_sat_and_sun) * weeks = 200 :=
by
  sorry

end prudence_sleep_in_4_weeks_l114_114881


namespace range_of_m_l114_114021

def f (x : ℝ) : ℝ := x^2 - 4 * x - 6

theorem range_of_m (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ m → -10 ≤ f x ∧ f x ≤ -6) →
  2 ≤ m ∧ m ≤ 4 := 
sorry

end range_of_m_l114_114021


namespace color_tv_cost_l114_114680

theorem color_tv_cost (x : ℝ) (y : ℝ) (z : ℝ)
  (h1 : y = x * 1.4)
  (h2 : z = y * 0.8)
  (h3 : z = 360 + x) :
  x = 3000 :=
sorry

end color_tv_cost_l114_114680


namespace option_A_is_translation_l114_114216

-- Define what constitutes a translation transformation
def is_translation (description : String) : Prop :=
  description = "Pulling open a drawer"

-- Define each option
def option_A : String := "Pulling open a drawer"
def option_B : String := "Viewing text through a magnifying glass"
def option_C : String := "The movement of the minute hand on a clock"
def option_D : String := "You and the image in a plane mirror"

-- The main theorem stating that option A is the translation transformation
theorem option_A_is_translation : is_translation option_A :=
by
  -- skip the proof, adding sorry
  sorry

end option_A_is_translation_l114_114216


namespace population_definition_l114_114967

variable (students : Type) (weights : students → ℝ) (sample : Fin 50 → students)
variable (total_students : Fin 300 → students)
variable (is_selected : students → Prop)

theorem population_definition :
    (∀ s, is_selected s ↔ ∃ i, sample i = s) →
    (population = {w : ℝ | ∃ s, w = weights s}) ↔
    (population = {w : ℝ | ∃ s, w = weights s ∧ ∃ i, total_students i = s}) := by
  sorry

end population_definition_l114_114967


namespace distinct_balls_boxes_l114_114377

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114377


namespace min_value_3x_4y_l114_114766

theorem min_value_3x_4y {x y : ℝ} (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
    3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_4y_l114_114766


namespace number_of_triangles_in_decagon_l114_114226

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l114_114226


namespace zongzi_cost_per_bag_first_batch_l114_114157

theorem zongzi_cost_per_bag_first_batch (x : ℝ)
  (h1 : 7500 / (x - 4) = 3 * (3000 / x))
  (h2 : 3000 > 0)
  (h3 : 7500 > 0)
  (h4 : x > 4) :
  x = 24 :=
by sorry

end zongzi_cost_per_bag_first_batch_l114_114157


namespace cos_diff_proof_l114_114743

theorem cos_diff_proof (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5 / 8 := 
by
  sorry

end cos_diff_proof_l114_114743


namespace min_colors_correctness_l114_114149

noncomputable def min_colors_no_monochromatic_cycle (n : ℕ) : ℕ :=
if n ≤ 2 then 1 else 2

theorem min_colors_correctness (n : ℕ) (h₀ : n > 0) :
  (min_colors_no_monochromatic_cycle n = 1 ∧ n ≤ 2) ∨
  (min_colors_no_monochromatic_cycle n = 2 ∧ n ≥ 3) :=
by
  sorry

end min_colors_correctness_l114_114149


namespace wicket_count_l114_114054

theorem wicket_count (initial_avg new_avg : ℚ) (runs_last_match wickets_last_match : ℕ) (delta_avg : ℚ) (W : ℕ) :
  initial_avg = 12.4 →
  new_avg = 12.0 →
  delta_avg = 0.4 →
  runs_last_match = 26 →
  wickets_last_match = 8 →
  initial_avg * W + runs_last_match = new_avg * (W + wickets_last_match) →
  W = 175 := by
  sorry

end wicket_count_l114_114054


namespace tom_pie_share_l114_114691

theorem tom_pie_share :
  (∃ (x : ℚ), 4 * x = (5 / 8) ∧ x = 5 / 32) :=
by
  sorry

end tom_pie_share_l114_114691


namespace ratio_of_area_to_breadth_l114_114955

theorem ratio_of_area_to_breadth (b l A : ℝ) (h₁ : b = 10) (h₂ : l - b = 10) (h₃ : A = l * b) : A / b = 20 := 
by
  sorry

end ratio_of_area_to_breadth_l114_114955


namespace maximum_value_of_function_l114_114707

noncomputable def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem maximum_value_of_function :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f x_max = 25 / 4 :=
by 
  sorry

end maximum_value_of_function_l114_114707


namespace sum_of_remaining_two_scores_l114_114643

open Nat

theorem sum_of_remaining_two_scores :
  ∃ x y : ℕ, x + y = 160 ∧ (65 + 75 + 85 + 95 + x + y) / 6 = 80 :=
by
  sorry

end sum_of_remaining_two_scores_l114_114643


namespace value_of_a3_l114_114961

def a_n (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem value_of_a3 : a_n 3 = -10 :=
by
  -- The proof would go here.
  sorry

end value_of_a3_l114_114961


namespace not_integer_division_l114_114915

def P : ℕ := 1
def Q : ℕ := 2

theorem not_integer_division : ¬ (∃ (n : ℤ), (P : ℤ) / (Q : ℤ) = n) := by
sorry

end not_integer_division_l114_114915


namespace regular_polygon_sides_l114_114842

theorem regular_polygon_sides (n : ℕ) (h : (180 * (n - 2) = 135 * n)) : n = 8 := by
  sorry

end regular_polygon_sides_l114_114842


namespace chocolate_discount_l114_114020

theorem chocolate_discount :
    let original_cost : ℝ := 2
    let final_price : ℝ := 1.43
    let discount := original_cost - final_price
    discount = 0.57 := by
  sorry

end chocolate_discount_l114_114020


namespace least_three_digit_multiple_of_8_l114_114977

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l114_114977


namespace find_AC_l114_114994

theorem find_AC (A B C : ℝ) (r1 r2 : ℝ) (AB : ℝ) (AC : ℝ) 
  (h_rad1 : r1 = 1) (h_rad2 : r2 = 3) (h_AB : AB = 2 * Real.sqrt 5) 
  (h_AC : AC = AB / 4) :
  AC = Real.sqrt 5 / 2 :=
by
  sorry

end find_AC_l114_114994


namespace balls_into_boxes_l114_114395

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l114_114395


namespace expression_parity_l114_114938

variable (o n c : ℕ)

def is_odd (x : ℕ) : Prop := ∃ k, x = 2 * k + 1

theorem expression_parity (ho : is_odd o) (hc : is_odd c) : 
  (o^2 + n * o + c) % 2 = 0 :=
  sorry

end expression_parity_l114_114938


namespace asymptotes_of_hyperbola_l114_114600

variable {a : ℝ}

/-- Given that the length of the real axis of the hyperbola x^2/a^2 - y^2 = 1 (a > 0) is 1,
    we want to prove that the equation of its asymptotes is y = ± 2x. -/
theorem asymptotes_of_hyperbola (ha : a > 0) (h_len : 2 * a = 1) :
  ∀ x y : ℝ, (y = 2 * x) ∨ (y = -2 * x) :=
by {
  sorry
}

end asymptotes_of_hyperbola_l114_114600


namespace ways_to_place_balls_in_boxes_l114_114391

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114391


namespace problem_part1_problem_part2_l114_114316

variable (A B : Set ℝ)
def C_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem problem_part1 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  C_R (A ∩ B) = { x : ℝ | x < 3 ∨ x ≥ 6 } :=
by
  intros hA hB
  sorry

theorem problem_part2 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  (C_R B) ∪ A = { x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by
  intros hA hB
  sorry

end problem_part1_problem_part2_l114_114316


namespace math_problem_l114_114553

-- Define the first part of the problem
def line_area_to_axes (line_eq : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  line_eq x y ∧ x = 4 ∧ y = -4

-- Define the second part of the problem
def line_through_fixed_point (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (m * x) + y + m = 0 ∧ x = -1 ∧ y = 0

-- Theorem combining both parts
theorem math_problem (line_eq : ℝ → ℝ → Prop) (m : ℝ) :
  (∃ x y, line_area_to_axes line_eq x y → 8 = (1 / 2) * 4 * 4) ∧ line_through_fixed_point m :=
sorry

end math_problem_l114_114553


namespace tan_of_angle_subtraction_l114_114888

theorem tan_of_angle_subtraction (a : ℝ) (h : Real.tan (a + Real.pi / 4) = 1 / 7) : Real.tan a = -3 / 4 :=
by
  sorry

end tan_of_angle_subtraction_l114_114888


namespace baseball_batter_at_bats_left_l114_114564

theorem baseball_batter_at_bats_left (L R H_L H_R : ℕ) (h1 : L + R = 600)
    (h2 : H_L + H_R = 192) (h3 : H_L = 25 / 100 * L) (h4 : H_R = 35 / 100 * R) : 
    L = 180 :=
by
  sorry

end baseball_batter_at_bats_left_l114_114564


namespace tan_neg_405_eq_neg1_l114_114697

theorem tan_neg_405_eq_neg1 : tan (-405 * real.pi / 180) = -1 :=
by 
  -- Simplify representing -405 degrees in radians and use known angle properties
  sorry

end tan_neg_405_eq_neg1_l114_114697


namespace balls_in_boxes_l114_114368

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114368


namespace trivia_team_original_members_l114_114572

theorem trivia_team_original_members (x : ℕ) (h1 : 6 * (x - 2) = 18) : x = 5 :=
by
  sorry

end trivia_team_original_members_l114_114572


namespace geometric_sequence_increasing_iff_l114_114934

variable {a : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem geometric_sequence_increasing_iff 
  (ha : is_geometric_sequence a q) 
  (h : a 0 < a 1 ∧ a 1 < a 2) : 
  is_increasing_sequence a ↔ (a 0 < a 1 ∧ a 1 < a 2) := 
sorry

end geometric_sequence_increasing_iff_l114_114934


namespace determine_weights_of_balls_l114_114970

theorem determine_weights_of_balls (A B C D E m1 m2 m3 m4 m5 m6 m7 m8 m9 : ℝ)
  (h1 : m1 = A)
  (h2 : m2 = B)
  (h3 : m3 = C)
  (h4 : m4 = A + D)
  (h5 : m5 = A + E)
  (h6 : m6 = B + D)
  (h7 : m7 = B + E)
  (h8 : m8 = C + D)
  (h9 : m9 = C + E) :
  ∃ (A' B' C' D' E' : ℝ), 
    ((A' = A ∨ B' = B ∨ C' = C ∨ D' = D ∨ E' = E) ∧
     (A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ A' ≠ E' ∧
      B' ≠ C' ∧ B' ≠ D' ∧ B' ≠ E' ∧
      C' ≠ D' ∧ C' ≠ E' ∧
      D' ≠ E')) :=
sorry

end determine_weights_of_balls_l114_114970


namespace passenger_cars_count_l114_114072

theorem passenger_cars_count (P C : ℕ) 
    (h₁ : C = (1 / 2 : ℚ) * P + 3) 
    (h₂ : P + C + 2 = 71) : P = 44 :=
begin
  sorry
end

end passenger_cars_count_l114_114072


namespace tangency_point_l114_114727

def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 18
def parabola2 (y : ℝ) : ℝ := y^2 + 60 * y + 910

theorem tangency_point (x y : ℝ) (h1 : y = parabola1 x) (h2 : x = parabola2 y) :
  x = -9 / 2 ∧ y = -59 / 2 :=
by
  sorry

end tangency_point_l114_114727


namespace sugar_flour_difference_l114_114004

theorem sugar_flour_difference :
  ∀ (flour_required_kg sugar_required_lb flour_added_kg kg_to_lb),
    flour_required_kg = 2.25 →
    sugar_required_lb = 5.5 →
    flour_added_kg = 1 →
    kg_to_lb = 2.205 →
    (sugar_required_lb / kg_to_lb * 1000) - ((flour_required_kg - flour_added_kg) * 1000) = 1244.8 :=
by
  intros flour_required_kg sugar_required_lb flour_added_kg kg_to_lb
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- sorry is used to skip the actual proof
  sorry

end sugar_flour_difference_l114_114004


namespace caterpillars_left_on_tree_l114_114812

-- Definitions based on conditions
def initialCaterpillars : ℕ := 14
def hatchedCaterpillars : ℕ := 4
def caterpillarsLeftToCocoon : ℕ := 8

-- The proof problem statement in Lean
theorem caterpillars_left_on_tree : initialCaterpillars + hatchedCaterpillars - caterpillarsLeftToCocoon = 10 :=
by
  -- solution steps will go here eventually
  sorry

end caterpillars_left_on_tree_l114_114812


namespace proof_problem_l114_114601

open Real

-- Definitions
noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Conditions
def eccentricity (c a : ℝ) : Prop :=
  c / a = (sqrt 2) / 2

def min_distance_to_focus (a c : ℝ) : Prop :=
  a - c = sqrt 2 - 1

-- Proof problem statement
theorem proof_problem (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (b_lt_a : b < a)
  (ecc : eccentricity c a) (min_dist : min_distance_to_focus a c)
  (x y k m : ℝ) (line_condition : y = k * x + m) :
  ellipse_equation x y a b → ellipse_equation x y (sqrt 2) 1 ∧
  (parabola_equation x y → (y = sqrt 2 / 2 * x + sqrt 2 ∨ y = -sqrt 2 / 2 * x - sqrt 2)) :=
sorry

end proof_problem_l114_114601


namespace number_times_one_fourth_squared_eq_four_cubed_l114_114822

theorem number_times_one_fourth_squared_eq_four_cubed :
  ∃ x : ℕ, x * (1 / 4: ℝ)^2 = (4: ℝ)^3 :=
by
  use 1024
  sorry

end number_times_one_fourth_squared_eq_four_cubed_l114_114822


namespace kona_distance_proof_l114_114592

-- Defining the distances as constants
def distance_to_bakery : ℕ := 9
def distance_from_grandmother_to_home : ℕ := 27
def additional_trip_distance : ℕ := 6

-- Defining the variable for the distance from bakery to grandmother's house
def x : ℕ := 30

-- Main theorem to prove the distance
theorem kona_distance_proof :
  distance_to_bakery + x + distance_from_grandmother_to_home = 2 * x + additional_trip_distance :=
by
  sorry

end kona_distance_proof_l114_114592


namespace abs_inequality_solution_l114_114728

theorem abs_inequality_solution (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 < x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x < -5) := 
by
  sorry

end abs_inequality_solution_l114_114728


namespace num_ways_to_distribute_balls_into_boxes_l114_114448

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114448


namespace find_k_max_product_l114_114320

theorem find_k_max_product : 
  (∃ k : ℝ, (3 : ℝ) * (x ^ 2) - 4 * x + k = 0 ∧ 16 - 12 * k ≥ 0 ∧ (∀ x1 x2 : ℝ, x1 * x2 = k / 3 → x1 + x2 = 4 / 3 → x1 * x2 ≤ (2 / 3) ^ 2)) →
  k = 4 / 3 :=
by 
  sorry

end find_k_max_product_l114_114320


namespace remainder_86592_8_remainder_8741_13_l114_114816

theorem remainder_86592_8 :
  86592 % 8 = 0 :=
by
  sorry

theorem remainder_8741_13 :
  8741 % 13 = 5 :=
by
  sorry

end remainder_86592_8_remainder_8741_13_l114_114816


namespace perfect_cube_probability_l114_114679

theorem perfect_cube_probability :
  ∃ p q : ℕ, p + q = 288 ∧ Nat.Coprime p q ∧ 
  ∃ (P : ℚ), P = (p : ℚ) / (q : ℚ) ∧ 
  (∀ (dices : Fin 5 → Fin 6), 
    let product := ∏ i, dices i + 1 
    in (Nat.isCube product) ↔ ((dices 0 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 1 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 2 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 3 = ⟨5, Nat.lt_succ_self 5⟩ ∧ dices 4 = ⟨5, Nat.lt_succ_self 5⟩) ∨ 
                            (natMod (product) 15 = 0))) :=
sorry

end perfect_cube_probability_l114_114679


namespace find_a9_l114_114895

variable (S : ℕ → ℚ) (a : ℕ → ℚ) (n : ℕ) (d : ℚ)

-- Conditions
axiom sum_first_six : S 6 = 3
axiom sum_first_eleven : S 11 = 18
axiom Sn_definition : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n)
axiom arithmetic_sequence : ∀ n, a (n + 1) = a 1 + n * d

-- Problem statement
theorem find_a9 : a 9 = 3 := sorry

end find_a9_l114_114895


namespace ways_to_distribute_balls_l114_114359

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114359


namespace trapezoid_median_l114_114847

theorem trapezoid_median
  (h : ℝ)
  (area_triangle : ℝ)
  (area_trapezoid : ℝ)
  (bt : ℝ)
  (bt_sum : ℝ)
  (ht_positive : h ≠ 0)
  (triangle_area : area_triangle = (1/2) * bt * h)
  (trapezoid_area : area_trapezoid = area_triangle)
  (trapezoid_bt_sum : bt_sum = 40)
  (triangle_bt : bt = 24)
  : (bt_sum / 2) = 20 :=
by
  sorry

end trapezoid_median_l114_114847


namespace minimum_resistors_required_l114_114221

-- Define the grid configuration and the connectivity condition
def isReliableGrid (m : ℕ) (n : ℕ) (failures : Finset (ℕ × ℕ)) : Prop :=
m * n > 9 ∧ (∀ (a b : ℕ), a ≠ b → (a, b) ∉ failures)

-- Minimum number of resistors ensuring connectivity with up to 9 failures
theorem minimum_resistors_required :
  ∃ (m n : ℕ), 5 * 5 = 25 ∧ isReliableGrid 5 5 ∅ :=
by
  let m : ℕ := 5
  let n : ℕ := 5
  have h₁ : m * n = 25 := by rfl
  have h₂ : isReliableGrid 5 5 ∅ := by
    unfold isReliableGrid
    exact ⟨by norm_num, sorry⟩ -- formal proof omitted for brevity
  exact ⟨m, n, h₁, h₂⟩

end minimum_resistors_required_l114_114221


namespace intersection_M_N_l114_114605

open Set

noncomputable def M : Set ℝ := {x | x ≥ 2}

noncomputable def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} :=
by
  sorry

end intersection_M_N_l114_114605


namespace grill_run_time_l114_114208

-- Definitions of conditions
def coals_burned_per_minute : ℕ := 15
def minutes_per_coal_burned : ℕ := 20
def coals_per_bag : ℕ := 60
def bags_burned : ℕ := 3

-- Theorems to prove the question
theorem grill_run_time (coals_burned_per_minute: ℕ) (minutes_per_coal_burned: ℕ) (coals_per_bag: ℕ) (bags_burned: ℕ): (coals_burned_per_minute * (minutes_per_coal_burned * bags_burned * coals_per_bag / (coals_burned_per_minute * coals_per_bag))) / 60 = 4 := 
by 
  -- Lean statement skips detailed proof steps for conciseness
  sorry

end grill_run_time_l114_114208


namespace regular_polygon_sides_l114_114841

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ m : ℕ, m = 360 / n → n ≠ 0 → m = 30) : n = 12 :=
  sorry

end regular_polygon_sides_l114_114841


namespace center_temperature_l114_114611

-- Define the conditions as a structure
structure SquareSheet (f : ℝ × ℝ → ℝ) :=
  (temp_0: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 0) = 0 ∧ f (0, x) = 0 ∧ f (1, x) = 0)
  (temp_100: ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x, 1) = 100)
  (no_radiation_loss: True) -- Just a placeholder since this condition is theoretical in nature

-- Define the claim as a theorem
theorem center_temperature (f : ℝ × ℝ → ℝ) (h : SquareSheet f) : f (0.5, 0.5) = 25 :=
by
  sorry -- Proof is not required and skipped

end center_temperature_l114_114611


namespace ball_in_boxes_l114_114340

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l114_114340


namespace number_of_triangles_in_decagon_l114_114248

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l114_114248


namespace exists_constant_C_inequality_for_difference_l114_114892

theorem exists_constant_C (a : ℕ → ℝ) (C : ℝ) (hC : 0 < C) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a n ≤ C * n^2) := sorry

theorem inequality_for_difference (a : ℕ → ℝ) :
  (a 1 = 1) →
  (a 2 = 8) →
  (∀ n : ℕ, 2 ≤ n → a (n + 1) = a (n - 1) + (4 / n) * a n) →
  (∀ n : ℕ, a (n + 1) - a n ≤ 4 * n + 3) := sorry

end exists_constant_C_inequality_for_difference_l114_114892


namespace number_of_triangles_in_decagon_l114_114224

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l114_114224


namespace correct_choices_are_bd_l114_114552

def is_correct_statement_b : Prop := 
  let S := (1 / 2) * 4 * |(-4)| in
  S = 8

def is_correct_statement_d : Prop :=
  ∀ m : ℝ, ∃ p : ℝ × ℝ, p = (-1, 0) ∧
    (p.1 = -1) → (m * p.1 + p.2 + m = 0)

theorem correct_choices_are_bd : is_correct_statement_b ∧ is_correct_statement_d :=
by 
  sorry

end correct_choices_are_bd_l114_114552


namespace total_return_at_x_50_optimal_investment_strategy_l114_114064

-- Definitions and conditions from the problem
def total_investment (x y : ℝ) : Prop := x + y = 120
def min_investment (x y : ℝ) : Prop := x ≥ 40 ∧ y ≥ 40
def return_A (x : ℝ) : ℝ := 3 * real.sqrt(2 * x) - 6
def return_B (y : ℝ) : ℝ := (1/4) * y + 2
def total_return (x : ℝ) : ℝ := return_A x + return_B (120 - x)

-- Q1: Total return when x = 50
theorem total_return_at_x_50 : total_return 50 = 43.5 := 
by 
  rw [total_return, return_A, return_B]; 
  -- specific calculations are done in actual proofs
  sorry

-- Q2: Optimal investment strategy
theorem optimal_investment_strategy : 
  (∀ x : ℝ, 40 ≤ x ∧ x ≤ 80 → total_return x ≤ 44) ∧ total_return 72 = 44 := 
by 
  -- actual optimization calculations and proofs
  sorry

end total_return_at_x_50_optimal_investment_strategy_l114_114064


namespace sufficient_but_not_necessary_l114_114561

theorem sufficient_but_not_necessary (x : ℝ) (h1 : x > 1 → x > 0) (h2 : ¬ (x > 0 → x > 1)) : 
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) := 
by 
  sorry

end sufficient_but_not_necessary_l114_114561


namespace divisors_of_64n4_l114_114877

theorem divisors_of_64n4 (n : ℕ) (hn : 0 < n) (hdiv : ∃ d, d = (120 * n^3) ∧ d.divisors.card = 120) : (64 * n^4).divisors.card = 375 := 
by 
  sorry

end divisors_of_64n4_l114_114877


namespace balls_in_boxes_l114_114373

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114373


namespace second_person_days_l114_114563

theorem second_person_days (P1 P2 : ℝ) (h1 : P1 = 1 / 24) (h2 : P1 + P2 = 1 / 8) : 1 / P2 = 12 :=
by
  sorry

end second_person_days_l114_114563


namespace horses_eat_oats_twice_a_day_l114_114791

-- Define the main constants and assumptions
def number_of_horses : ℕ := 4
def oats_per_meal : ℕ := 4
def grain_per_day : ℕ := 3
def total_food : ℕ := 132
def duration_in_days : ℕ := 3

-- Main theorem statement
theorem horses_eat_oats_twice_a_day (x : ℕ) (h : duration_in_days * number_of_horses * (oats_per_meal * x + grain_per_day) = total_food) : x = 2 := 
sorry

end horses_eat_oats_twice_a_day_l114_114791


namespace ways_to_put_balls_in_boxes_l114_114424

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114424


namespace expected_non_allergic_l114_114010

theorem expected_non_allergic (p : ℝ) (n : ℕ) (h : p = 1 / 4) (hn : n = 300) : n * p = 75 :=
by sorry

end expected_non_allergic_l114_114010


namespace find_number_l114_114192

theorem find_number:
  ∃ x: ℕ, (∃ k: ℕ, ∃ r: ℕ, 5 * (x + 3) = 8 * k + r ∧ k = 156 ∧ r = 2) ∧ x = 247 :=
by 
  sorry

end find_number_l114_114192


namespace fraction_given_to_jerry_l114_114815

-- Define the problem conditions
def initial_apples := 2
def slices_per_apple := 8
def total_slices := initial_apples * slices_per_apple -- 2 * 8 = 16

def remaining_slices_after_eating := 5
def slices_before_eating := remaining_slices_after_eating * 2 -- 5 * 2 = 10
def slices_given_to_jerry := total_slices - slices_before_eating -- 16 - 10 = 6

-- Define the proof statement to verify that the fraction of slices given to Jerry is 3/8
theorem fraction_given_to_jerry : (slices_given_to_jerry : ℚ) / total_slices = 3 / 8 :=
by
  -- skip the actual proof, just outline the goal
  sorry

end fraction_given_to_jerry_l114_114815


namespace moles_HCl_combination_l114_114710

-- Define the conditions:
def moles_HCl (C5H12O: ℕ) (H2O: ℕ) : ℕ :=
  if H2O = 18 then 18 else 0

-- The main statement to prove:
theorem moles_HCl_combination :
  moles_HCl 1 18 = 18 :=
sorry

end moles_HCl_combination_l114_114710


namespace find_age_of_B_l114_114771

-- Define A and B as natural numbers (assuming ages are non-negative integers)
variables (A B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- The goal is to prove that B = 36 given the conditions
theorem find_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 :=
sorry

end find_age_of_B_l114_114771


namespace range_of_b_l114_114631

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a x b : ℝ) (ha : -1 ≤ a) (ha' : a < 0) (hx : 0 < x) (hx' : x ≤ 1) 
  (h : f x a < b) : -3 / 2 < b := 
sorry

end range_of_b_l114_114631


namespace ways_to_distribute_balls_l114_114411

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114411


namespace half_guests_want_two_burgers_l114_114855

theorem half_guests_want_two_burgers 
  (total_guests : ℕ) (half_guests : ℕ)
  (time_per_side : ℕ) (time_per_burger : ℕ)
  (grill_capacity : ℕ) (total_time : ℕ)
  (guests_one_burger : ℕ) (total_burgers : ℕ) : 
  total_guests = 30 →
  time_per_side = 4 →
  time_per_burger = 8 →
  grill_capacity = 5 →
  total_time = 72 →
  guests_one_burger = 15 →
  total_burgers = 45 →
  half_guests * 2 = total_burgers - guests_one_burger →
  half_guests = 15 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end half_guests_want_two_burgers_l114_114855


namespace set_intersection_complement_l114_114324

def U : Set ℝ := Set.univ
def A : Set ℝ := { y | ∃ x, x > 0 ∧ y = 4 / x }
def B : Set ℝ := { y | ∃ x, x < 1 ∧ y = 2^x }
def comp_B : Set ℝ := { y | y ≤ 0 } ∪ { y | y ≥ 2 }
def intersection : Set ℝ := { y | y ≥ 2 }

theorem set_intersection_complement :
  A ∩ comp_B = intersection :=
by
  sorry

end set_intersection_complement_l114_114324


namespace photos_ratio_l114_114948

theorem photos_ratio (L R C : ℕ) (h1 : R = L) (h2 : C = 12) (h3 : R = C + 24) :
  L / C = 3 :=
by 
  sorry

end photos_ratio_l114_114948


namespace combined_capacity_eq_l114_114534

variable {x y z : ℚ}

-- Container A condition
def containerA_full (x : ℚ) := 0.75 * x
def containerA_initial (x : ℚ) := 0.30 * x
def containerA_diff (x : ℚ) := containerA_full x - containerA_initial x = 36

-- Container B condition
def containerB_full (y : ℚ) := 0.70 * y
def containerB_initial (y : ℚ) := 0.40 * y
def containerB_diff (y : ℚ) := containerB_full y - containerB_initial y = 20

-- Container C condition
def containerC_full (z : ℚ) := (2 / 3) * z
def containerC_initial (z : ℚ) := 0.50 * z
def containerC_diff (z : ℚ) := containerC_full z - containerC_initial z = 12

-- Theorem to prove the total capacity
theorem combined_capacity_eq : containerA_diff x → containerB_diff y → containerC_diff z → 
(218 + 2 / 3 = x + y + z) :=
by
  intros hA hB hC
  sorry

end combined_capacity_eq_l114_114534


namespace train_crosses_platform_in_39_seconds_l114_114207

-- Definitions based on the problem's conditions
def train_length : ℕ := 450
def time_to_cross_signal : ℕ := 18
def platform_length : ℕ := 525

-- The speed of the train
def train_speed : ℕ := train_length / time_to_cross_signal

-- The total distance the train has to cover
def total_distance : ℕ := train_length + platform_length

-- The time it takes for the train to cross the platform
def time_to_cross_platform : ℕ := total_distance / train_speed

-- The theorem we need to prove
theorem train_crosses_platform_in_39_seconds :
  time_to_cross_platform = 39 := by
  sorry

end train_crosses_platform_in_39_seconds_l114_114207


namespace total_number_of_letters_l114_114143

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l114_114143


namespace probability_X_eq_Y_correct_l114_114684

noncomputable def probability_X_eq_Y : ℝ :=
  let lower_bound := -20 * Real.pi
  let upper_bound := 20 * Real.pi
  let total_pairs := (upper_bound - lower_bound) * (upper_bound - lower_bound)
  let matching_pairs := 81
  matching_pairs / total_pairs

theorem probability_X_eq_Y_correct :
  probability_X_eq_Y = 81 / 1681 :=
by
  unfold probability_X_eq_Y
  sorry

end probability_X_eq_Y_correct_l114_114684


namespace sum_of_squares_l114_114536

def b1 : ℚ := 10 / 32
def b2 : ℚ := 0
def b3 : ℚ := -5 / 32
def b4 : ℚ := 0
def b5 : ℚ := 1 / 32

theorem sum_of_squares : b1^2 + b2^2 + b3^2 + b4^2 + b5^2 = 63 / 512 :=
by
  sorry

end sum_of_squares_l114_114536


namespace simultaneous_eq_solvable_l114_114878

theorem simultaneous_eq_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = m * x + 4 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_eq_solvable_l114_114878


namespace ways_to_distribute_balls_l114_114356

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114356


namespace vanilla_syrup_cost_l114_114151

theorem vanilla_syrup_cost :
  ∀ (unit_cost_drip : ℝ) (num_drip : ℕ)
    (unit_cost_espresso : ℝ) (num_espresso : ℕ)
    (unit_cost_latte : ℝ) (num_lattes : ℕ)
    (unit_cost_cold_brew : ℝ) (num_cold_brews : ℕ)
    (unit_cost_cappuccino : ℝ) (num_cappuccino : ℕ)
    (total_cost : ℝ) (vanilla_cost : ℝ),
  unit_cost_drip = 2.25 →
  num_drip = 2 →
  unit_cost_espresso = 3.50 →
  num_espresso = 1 →
  unit_cost_latte = 4.00 →
  num_lattes = 2 →
  unit_cost_cold_brew = 2.50 →
  num_cold_brews = 2 →
  unit_cost_cappuccino = 3.50 →
  num_cappuccino = 1 →
  total_cost = 25.00 →
  vanilla_cost =
    total_cost -
    ((unit_cost_drip * num_drip) +
    (unit_cost_espresso * num_espresso) +
    (unit_cost_latte * (num_lattes - 1)) +
    (unit_cost_cold_brew * num_cold_brews) +
    (unit_cost_cappuccino * num_cappuccino)) →
  vanilla_cost = 0.50 := sorry

end vanilla_syrup_cost_l114_114151


namespace largest_possible_square_area_l114_114569

def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 6
def largest_square_side : ℕ := rectangle_width
def largest_square_area : ℕ := largest_square_side * largest_square_side

theorem largest_possible_square_area :
  largest_square_area = 36 := by
    sorry

end largest_possible_square_area_l114_114569


namespace simplify_fraction_l114_114797

theorem simplify_fraction (x : ℤ) : 
    (2 * x + 3) / 4 + (5 - 4 * x) / 3 = (-10 * x + 29) / 12 := 
by
  sorry

end simplify_fraction_l114_114797


namespace parabola_standard_form_l114_114590

theorem parabola_standard_form (a : ℝ) (x y : ℝ) :
  (∀ a : ℝ, (2 * a + 3) * x + y - 4 * a + 2 = 0 → 
  x = 2 ∧ y = -8) → 
  (y^2 = 32 * x ∨ x^2 = - (1/2) * y) :=
by 
  intros h
  sorry

end parabola_standard_form_l114_114590


namespace proof_problem_l114_114904

noncomputable def p (a : ℝ) : Prop :=
∀ x : ℝ, x^2 + a * x + a^2 ≥ 0

noncomputable def q : Prop :=
∃ x₀ : ℕ, 0 < x₀ ∧ 2 * x₀^2 - 1 ≤ 0

theorem proof_problem (a : ℝ) (hp : p a) (hq : q) : p a ∨ q :=
by
  sorry

end proof_problem_l114_114904


namespace min_function_value_l114_114318

theorem min_function_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  (1/3 * x^3 + y^2 + z) = 13/12 :=
sorry

end min_function_value_l114_114318


namespace find_a_l114_114495

def A := { x : ℝ | x^2 + 4 * x = 0 }
def B (a : ℝ) := { x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 1) = 0 }

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x ∈ (A ∩ B a) ↔ x ∈ B a) → (a = 1 ∨ a ≤ -1) :=
by 
  sorry

end find_a_l114_114495


namespace dot_product_property_l114_114326

-- Definitions based on conditions
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

-- Required property
theorem dot_product_property : dot_product (vec_add (scalar_mult 2 vec_a) vec_b) vec_a = 6 :=
by sorry

end dot_product_property_l114_114326


namespace greatest_radius_l114_114617

theorem greatest_radius (r : ℤ) (h : π * r^2 < 100 * π) : r < 10 :=
sorry

example : ∃ r : ℤ, π * r^2 < 100 * π ∧ ∀ r' : ℤ, (π * r'^2 < 100 * π) → r' ≤ r :=
begin
  use 9,
  split,
  { linarith },
  { intros r' h',
    have hr' : r' < 10,
    { linarith },
    exact int.lt_of_le_of_lt (int.le_of_lt_add_one hr') (by linarith) }
end

end greatest_radius_l114_114617


namespace radius_large_circle_l114_114031

/-- Definitions for the problem context -/
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_circles (c1 c2 : Circle) : Prop :=
dist c1.center c2.center = c1.radius + c2.radius

/-- Theorem to prove the radius of the large circle -/
theorem radius_large_circle 
  (small_circle : Circle)
  (h_radius : small_circle.radius = 2)
  (large_circle : Circle)
  (h_tangency1 : tangent_circles small_circle large_circle)
  (small_circle2 : Circle)
  (small_circle3 : Circle)
  (h_tangency2 : tangent_circles small_circle small_circle2)
  (h_tangency3 : tangent_circles small_circle small_circle3)
  (h_tangency4 : tangent_circles small_circle2 large_circle)
  (h_tangency5 : tangent_circles small_circle3 large_circle)
  (h_tangency6 : tangent_circles small_circle2 small_circle3)
  : large_circle.radius = 2 * (Real.sqrt 3 + 1) :=
sorry

end radius_large_circle_l114_114031


namespace find_angle_A_l114_114620

theorem find_angle_A 
  (a b c A B C : ℝ)
  (h₀ : a = Real.sqrt 2)
  (h₁ : b = 2)
  (h₂ : Real.sin B - Real.cos B = Real.sqrt 2)
  (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  : A = Real.pi / 6 := 
  sorry

end find_angle_A_l114_114620


namespace winnie_lollipops_remainder_l114_114826

theorem winnie_lollipops_remainder :
  ∃ (k : ℕ), k = 505 % 14 ∧ k = 1 :=
by
  sorry

end winnie_lollipops_remainder_l114_114826


namespace profit_percentage_is_correct_l114_114557

noncomputable def cost_price (SP : ℝ) : ℝ := 0.81 * SP

noncomputable def profit (SP CP : ℝ) : ℝ := SP - CP

noncomputable def profit_percentage (profit CP : ℝ) : ℝ := (profit / CP) * 100

theorem profit_percentage_is_correct (SP : ℝ) (h : SP = 100) :
  profit_percentage (profit SP (cost_price SP)) (cost_price SP) = 23.46 :=
by
  sorry

end profit_percentage_is_correct_l114_114557


namespace distinct_balls_boxes_l114_114408

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114408


namespace ways_to_distribute_balls_l114_114365

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114365


namespace simple_interest_calculation_l114_114658

-- Define the principal (P), rate (R), and time (T)
def principal : ℝ := 10000
def rate : ℝ := 0.08
def time : ℝ := 1

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- The theorem to be proved
theorem simple_interest_calculation : simple_interest principal rate time = 800 :=
by
  -- Proof steps would go here, but this is left as an exercise
  sorry

end simple_interest_calculation_l114_114658


namespace sum_of_first_2n_terms_l114_114027

-- Definitions based on conditions
variable (n : ℕ) (S : ℕ → ℝ)

-- Conditions
def condition1 : Prop := S n = 24
def condition2 : Prop := S (3 * n) = 42

-- Statement to be proved
theorem sum_of_first_2n_terms {n : ℕ} (S : ℕ → ℝ) 
    (h1 : S n = 24) (h2 : S (3 * n) = 42) : S (2 * n) = 36 := by
  sorry

end sum_of_first_2n_terms_l114_114027


namespace sachin_age_l114_114161
-- Import the necessary library

-- Lean statement defining the problem conditions and result
theorem sachin_age :
  ∃ (S R : ℝ), (R = S + 7) ∧ (S / R = 7 / 9) ∧ (S = 24.5) :=
by
  sorry

end sachin_age_l114_114161


namespace A_inter_B_l114_114780

open Set

def A : Set ℤ := { x | -3 ≤ (2 * x - 1) ∧ (2 * x - 1) < 3 }

def B : Set ℤ := { x | ∃ k : ℤ, x = 2 * k + 1 }

theorem A_inter_B : A ∩ B = {-1, 1} :=
by
  sorry

end A_inter_B_l114_114780


namespace trajectory_of_A_l114_114625

theorem trajectory_of_A (A B C : (ℝ × ℝ)) (x y : ℝ) : 
  B = (-2, 0) ∧ C = (2, 0) ∧ (dist A (0, 0) = 3) → 
  (x, y) = A → 
  x^2 + y^2 = 9 ∧ y ≠ 0 := 
sorry

end trajectory_of_A_l114_114625


namespace current_population_l114_114195

def initial_population : ℕ := 684
def growth_rate : ℝ := 0.25
def moving_away_rate : ℝ := 0.40

theorem current_population (P0 : ℕ) (g : ℝ) (m : ℝ) : 
  P0 = initial_population → 
  g = growth_rate → 
  m = moving_away_rate → 
  (P0 + (P0 * g).to_nat - ((P0 + (P0 * g).to_nat) * m).to_nat) = 513 := 
by
  intros hP0 hg hm
  sorry

end current_population_l114_114195


namespace correct_proposition_l114_114848

theorem correct_proposition : 
  (¬ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 2 * x_0) ↔ (∀ x : ℝ, x^2 + 1 > 2 * x) := 
sorry

end correct_proposition_l114_114848


namespace ways_to_distribute_balls_l114_114363

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114363


namespace modulus_of_quotient_l114_114477

theorem modulus_of_quotient (z : ℂ) (hz : z = (1 - 2 * Complex.i) / (3 - Complex.i)) : Complex.abs z = Real.sqrt 2 / 2 :=
by
  rw [hz]
  sorry

end modulus_of_quotient_l114_114477


namespace factorize_expression_l114_114096

theorem factorize_expression (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) :=
by
  sorry

end factorize_expression_l114_114096


namespace ball_box_distribution_l114_114458

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l114_114458


namespace balls_in_boxes_l114_114369

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l114_114369


namespace tangent_line_parabola_k_l114_114292

theorem tangent_line_parabola_k :
  ∃ (k : ℝ), (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → (28 ^ 2 = 4 * 1 * 4 * k)) → k = 49 :=
by
  sorry

end tangent_line_parabola_k_l114_114292


namespace correct_answer_l114_114610

-- Definition of the correctness condition
def indicates_number (phrase : String) : Prop :=
  (phrase = "Noun + Cardinal Number") ∨ (phrase = "the + Ordinal Number + Noun")

-- Example phrases to be evaluated
def class_first : String := "Class First"
def the_class_one : String := "the Class One"
def class_one : String := "Class One"
def first_class : String := "First Class"

-- The goal is to prove that "Class One" meets the condition
theorem correct_answer : indicates_number "Class One" :=
by {
  -- Insert detailed proof steps here, currently omitted
  sorry
}

end correct_answer_l114_114610


namespace triangle_inequality_range_l114_114116

theorem triangle_inequality_range (x : ℝ) (h1 : 4 + 5 > x) (h2 : 4 + x > 5) (h3 : 5 + x > 4) :
  1 < x ∧ x < 9 := 
by
  sorry

end triangle_inequality_range_l114_114116


namespace least_positive_three_digit_multiple_of_8_l114_114981

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end least_positive_three_digit_multiple_of_8_l114_114981


namespace distinct_balls_boxes_l114_114380

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114380


namespace stream_speed_l114_114992

variable (B S : ℝ)

def downstream_eq : Prop := B + S = 13
def upstream_eq : Prop := B - S = 5

theorem stream_speed (h1 : downstream_eq B S) (h2 : upstream_eq B S) : S = 4 :=
by
  sorry

end stream_speed_l114_114992


namespace distinct_balls_boxes_l114_114402

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114402


namespace num_triangles_from_decagon_l114_114276

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l114_114276


namespace num_ways_to_distribute_balls_into_boxes_l114_114447

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114447


namespace angelina_speed_l114_114202

theorem angelina_speed (v : ℝ) (h1 : 840 / v - 40 = 240 / v) :
  2 * v = 30 :=
by
  sorry

end angelina_speed_l114_114202


namespace area_percent_difference_l114_114206

theorem area_percent_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  percent_difference = 4 := 
by
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  sorry

end area_percent_difference_l114_114206


namespace problem_statement_l114_114919

variable {P : ℕ → Prop}

theorem problem_statement
  (h1 : ∀ k, P k → P (k + 1))
  (h2 : ¬P 4)
  (n : ℕ) (hn : 1 ≤ n → n ≤ 4 → n ∈ Set.Icc 1 4) :
  ¬P n :=
by
  sorry

end problem_statement_l114_114919


namespace boys_camp_percentage_l114_114773

theorem boys_camp_percentage (x : ℕ) (total_boys : ℕ) (percent_science : ℕ) (not_science_boys : ℕ) 
    (percent_not_science : ℕ) (h1 : not_science_boys = percent_not_science * (x / 100) * total_boys) 
    (h2 : percent_not_science = 100 - percent_science) (h3 : percent_science = 30) 
    (h4 : not_science_boys = 21) (h5 : total_boys = 150) : x = 20 :=
by 
  sorry

end boys_camp_percentage_l114_114773


namespace quadrilateral_angle_l114_114775

theorem quadrilateral_angle (x y : ℝ) (h1 : 3 * x ^ 2 - x + 4 = 5) (h2 : x ^ 2 + y ^ 2 = 9) :
  x = (1 + Real.sqrt 13) / 6 :=
by
  sorry

end quadrilateral_angle_l114_114775


namespace mark_more_than_kate_by_100_l114_114949

variable (Pat Kate Mark : ℕ)
axiom total_hours : Pat + Kate + Mark = 180
axiom pat_twice_as_kate : Pat = 2 * Kate
axiom pat_third_of_mark : Pat = Mark / 3

theorem mark_more_than_kate_by_100 : Mark - Kate = 100 :=
by
  sorry

end mark_more_than_kate_by_100_l114_114949


namespace minimize_wire_length_l114_114969

theorem minimize_wire_length :
  ∃ (x : ℝ), (x > 0) ∧ (2 * (x + 4 / x) = 8) :=
by
  sorry

end minimize_wire_length_l114_114969


namespace fraction_zero_solution_l114_114480

theorem fraction_zero_solution (x : ℝ) (h : (x - 1) / (2 - x) = 0) : x = 1 :=
sorry

end fraction_zero_solution_l114_114480


namespace rooms_in_second_wing_each_hall_l114_114787

theorem rooms_in_second_wing_each_hall
  (floors_first_wing : ℕ)
  (halls_per_floor_first_wing : ℕ)
  (rooms_per_hall_first_wing : ℕ)
  (floors_second_wing : ℕ)
  (halls_per_floor_second_wing : ℕ)
  (total_rooms : ℕ)
  (h1 : floors_first_wing = 9)
  (h2 : halls_per_floor_first_wing = 6)
  (h3 : rooms_per_hall_first_wing = 32)
  (h4 : floors_second_wing = 7)
  (h5 : halls_per_floor_second_wing = 9)
  (h6 : total_rooms = 4248) :
  (total_rooms - floors_first_wing * halls_per_floor_first_wing * rooms_per_hall_first_wing) / 
  (floors_second_wing * halls_per_floor_second_wing) = 40 :=
  by {
  sorry
}

end rooms_in_second_wing_each_hall_l114_114787


namespace ways_to_stand_l114_114077

-- Definitions derived from conditions
def num_steps : ℕ := 7
def max_people_per_step : ℕ := 2

-- Define a function to count the number of different ways
noncomputable def count_ways : ℕ :=
  336

-- The statement to be proven in Lean 4
theorem ways_to_stand : count_ways = 336 :=
  sorry

end ways_to_stand_l114_114077


namespace necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l114_114736

variable (x y : ℝ)

theorem necessary_but_not_sufficient (hx : x < y ∧ y < 0) : x^2 > y^2 :=
sorry

theorem not_sufficient (hx : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
sorry

-- Optional: Combining the two to create a combined theorem statement
theorem x2_gt_y2_iff_x_lt_y_lt_0 : (∀ x y : ℝ, x < y ∧ y < 0 → x^2 > y^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬ (x < y ∧ y < 0)) :=
sorry

end necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l114_114736


namespace product_wavelengths_eq_n_cbrt_mn2_l114_114964

variable (m n : ℝ)

noncomputable def common_ratio (m n : ℝ) := (n / m)^(1/3)

noncomputable def wavelength_jiazhong (m n : ℝ) := (m^2 * n)^(1/3)
noncomputable def wavelength_nanlu (m n : ℝ) := (n^4 / m)^(1/3)

theorem product_wavelengths_eq_n_cbrt_mn2
  (h : n = m * (common_ratio m n)^3) :
  (wavelength_jiazhong m n) * (wavelength_nanlu m n) = n * (m * n^2)^(1/3) :=
by
  sorry

end product_wavelengths_eq_n_cbrt_mn2_l114_114964


namespace m_range_positive_solution_l114_114922

theorem m_range_positive_solution (m : ℝ) : (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) := by
  sorry

end m_range_positive_solution_l114_114922


namespace total_selling_price_is_correct_l114_114846

-- Define the given constants
def meters_of_cloth : ℕ := 85
def profit_per_meter : ℕ := 10
def cost_price_per_meter : ℕ := 95

-- Compute the selling price per meter
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- Calculate the total selling price
def total_selling_price : ℕ := selling_price_per_meter * meters_of_cloth

-- The theorem statement
theorem total_selling_price_is_correct : total_selling_price = 8925 := by
  sorry

end total_selling_price_is_correct_l114_114846


namespace school_trip_l114_114678

theorem school_trip (x : ℕ) (total_students : ℕ) :
  (28 * x + 13 = total_students) ∧ (32 * x - 3 = total_students) → 
  x = 4 ∧ total_students = 125 :=
by
  sorry

end school_trip_l114_114678


namespace smallest_digit_divisible_by_11_l114_114099

theorem smallest_digit_divisible_by_11 :
  ∃ (d : ℕ), d < 10 ∧ ∀ n : ℕ, (n + 45000 + 1000 + 457 + d) % 11 = 0 → d = 5 :=
by {
  sorry
}

end smallest_digit_divisible_by_11_l114_114099


namespace angle_between_vectors_is_30_degrees_l114_114757

open Real

noncomputable def vector_a : ℝ × ℝ := (cos (35 * pi / 180), sin (35 * pi / 180))
noncomputable def vector_b : ℝ × ℝ := (cos (65 * pi / 180), sin (65 * pi / 180))

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem angle_between_vectors_is_30_degrees :
  let θ := acos ((dot_product vector_a vector_b) / (magnitude vector_a * magnitude vector_b))
  θ = 30 * pi / 180 :=
by
  sorry

end angle_between_vectors_is_30_degrees_l114_114757


namespace num_ways_to_distribute_balls_into_boxes_l114_114451

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114451


namespace unique_solution_to_equation_l114_114297

theorem unique_solution_to_equation (a : ℝ) (h : ∀ x : ℝ, a * x^2 + Real.sin x ^ 2 = a^2 - a) : a = 1 :=
sorry

end unique_solution_to_equation_l114_114297


namespace cricket_team_matches_in_august_l114_114621

noncomputable def cricket_matches_played_in_august (M W W_new: ℕ) : Prop :=
  W = 26 * M / 100 ∧
  W_new = 52 * (M + 65) / 100 ∧ 
  W_new = W + 65

theorem cricket_team_matches_in_august (M W W_new: ℕ) : cricket_matches_played_in_august M W W_new → M = 120 := 
by
  sorry

end cricket_team_matches_in_august_l114_114621


namespace complement_of_A_eq_l114_114750

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x > 1}

theorem complement_of_A_eq {U : Set ℝ} (U_eq : U = Set.univ) {A : Set ℝ} (A_eq : A = {x | x > 1}) :
    U \ A = {x | x ≤ 1} :=
by
  sorry

end complement_of_A_eq_l114_114750


namespace remainder_div_7_l114_114559

theorem remainder_div_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k < 39) : k % 7 = 3 :=
sorry

end remainder_div_7_l114_114559


namespace smallest_sum_is_S5_l114_114130

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definitions of arithmetic sequence sum
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom h1 : a 3 + a 8 > 0
axiom h2 : S 9 < 0

-- Statements relating terms and sums in arithmetic sequence
axiom h3 : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem smallest_sum_is_S5 (seq_a : arithmetic_sequence a) : S 5 ≤ S 1 ∧ S 5 ≤ S 2 ∧ S 5 ≤ S 3 ∧ S 5 ≤ S 4 ∧ S 5 ≤ S 6 ∧ S 5 ≤ S 7 ∧ S 5 ≤ S 8 ∧ S 5 ≤ S 9 :=
by {
    sorry
}

end smallest_sum_is_S5_l114_114130


namespace midpoint_in_polar_coordinates_l114_114484

theorem midpoint_in_polar_coordinates :
  let A := (9, Real.pi / 3)
  let B := (9, 2 * Real.pi / 3)
  let mid := (Real.sqrt (3) * 9 / 2, Real.pi / 2)
  (mid = (Real.sqrt (3) * 9 / 2, Real.pi / 2)) :=
by 
  sorry

end midpoint_in_polar_coordinates_l114_114484


namespace multiple_of_larger_number_l114_114807

variables (S L M : ℝ)

-- Conditions
def small_num := S = 10.0
def sum_eq := S + L = 24
def multiplication_relation := 7 * S = M * L

-- Theorem statement
theorem multiple_of_larger_number (S L M : ℝ) 
  (h1 : small_num S) 
  (h2 : sum_eq S L) 
  (h3 : multiplication_relation S L M) : 
  M = 5 := by
  sorry

end multiple_of_larger_number_l114_114807


namespace average_student_headcount_l114_114045

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end average_student_headcount_l114_114045


namespace triangle_angles_l114_114089

-- Defining a structure for a triangle with angles
structure Triangle :=
(angleA angleB angleC : ℝ)

-- Define the condition for the triangle mentioned in the problem
def triangle_condition (t : Triangle) : Prop :=
  ∃ (α : ℝ), α = 22.5 ∧ t.angleA = 90 ∧ t.angleB = α ∧ t.angleC = 67.5

theorem triangle_angles :
  ∃ (t : Triangle), triangle_condition t :=
by
  -- The proof outline
  -- We need to construct a triangle with the given angle conditions
  -- angleA = 90°, angleB = 22.5°, angleC = 67.5°
  sorry

end triangle_angles_l114_114089


namespace largest_sum_digits_24_hour_watch_l114_114837

theorem largest_sum_digits_24_hour_watch : 
  (∃ h m : ℕ, 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ 
              (h / 10 + h % 10 + m / 10 + m % 10 = 24)) :=
by
  sorry

end largest_sum_digits_24_hour_watch_l114_114837


namespace ways_to_distribute_balls_l114_114419

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114419


namespace find_divisor_l114_114921

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 34) / y = 2) : y = 10 :=
by
  sorry

end find_divisor_l114_114921


namespace find_k_l114_114767

variable (k : ℝ) (t : ℝ) (a : ℝ)

theorem find_k (h1 : t = (5 / 9) * (k - 32) + a * k) (h2 : t = 20) (h3 : a = 3) : k = 10.625 := by
  sorry

end find_k_l114_114767


namespace num_triangles_in_decagon_l114_114283

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l114_114283


namespace relationship_between_b_and_c_l114_114115

-- Definitions based on the given conditions
def y1 (x a b : ℝ) : ℝ := (x + 2 * a) * (x - 2 * b)
def y2 (x b : ℝ) : ℝ := -x + 2 * b
def y (x a b : ℝ) : ℝ := y1 x a b + y2 x b

-- Lean theorem for the proof problem
theorem relationship_between_b_and_c
  (a b c : ℝ)
  (h : a + 2 = b)
  (h_y : y c a b = 0) :
  c = 5 - 2 * b ∨ c = 2 * b :=
by
  -- The proof will go here, currently omitted
  sorry

end relationship_between_b_and_c_l114_114115


namespace balls_into_boxes_l114_114330

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114330


namespace least_three_digit_multiple_of_8_l114_114976

theorem least_three_digit_multiple_of_8 : 
  ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ (n % 8 = 0) ∧ 
  (∀ m : ℕ, m >= 100 ∧ m < 1000 ∧ (m % 8 = 0) → n ≤ m) ∧ n = 104 :=
sorry

end least_three_digit_multiple_of_8_l114_114976


namespace M_eq_N_l114_114905

def M : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = (5/6) * Real.pi + 2 * k * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = -(7/6) * Real.pi + 2 * k * Real.pi}

theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l114_114905


namespace no_integer_b_satisfies_conditions_l114_114959

theorem no_integer_b_satisfies_conditions :
  ¬ ∃ b : ℕ, b^6 ≤ 196 ∧ 196 < b^7 :=
by
  sorry

end no_integer_b_satisfies_conditions_l114_114959


namespace find_s_l114_114000

theorem find_s (s : ℝ) :
  let P := (s - 3, 2)
  let Q := (1, s + 2)
  let M := ((s - 2) / 2, (s + 4) / 2)
  let dist_sq := (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2
  dist_sq = 3 * s^2 / 4 →
  s = -5 + 5 * Real.sqrt 2 ∨ s = -5 - 5 * Real.sqrt 2 :=
by
  intros P Q M dist_sq h
  sorry

end find_s_l114_114000


namespace smallest_positive_integer_linear_combination_l114_114183

theorem smallest_positive_integer_linear_combination : ∃ m n : ℤ, 3003 * m + 55555 * n = 1 :=
by
  sorry

end smallest_positive_integer_linear_combination_l114_114183


namespace packs_of_yellow_bouncy_balls_l114_114150

-- Define the conditions and the question in Lean
variables (GaveAwayGreen : ℝ) (BoughtGreen : ℝ) (BouncyBallsPerPack : ℝ) (TotalKeptBouncyBalls : ℝ) (Y : ℝ)

-- Assume the given conditions
axiom cond1 : GaveAwayGreen = 4.0
axiom cond2 : BoughtGreen = 4.0
axiom cond3 : BouncyBallsPerPack = 10.0
axiom cond4 : TotalKeptBouncyBalls = 80.0

-- Define the theorem statement
theorem packs_of_yellow_bouncy_balls (h1 : GaveAwayGreen = 4.0) (h2 : BoughtGreen = 4.0) (h3 : BouncyBallsPerPack = 10.0) (h4 : TotalKeptBouncyBalls = 80.0) : Y = 8 :=
sorry

end packs_of_yellow_bouncy_balls_l114_114150


namespace number_of_people_in_team_l114_114762

variable (x : ℕ) -- Number of people in the team

-- Conditions as definitions
def average_age_all (x : ℕ) : ℝ := 25
def leader_age : ℝ := 45
def average_age_without_leader (x : ℕ) : ℝ := 23

-- Proof problem statement
theorem number_of_people_in_team (h1 : (x : ℝ) * average_age_all x = x * (average_age_without_leader x - 1) + leader_age) : x = 11 := by
  sorry

end number_of_people_in_team_l114_114762


namespace num_possibilities_l114_114995

def last_digit_divisible_by_4 (n : Nat) : Prop := (60 + n) % 4 = 0

theorem num_possibilities : {n : Nat | n < 10 ∧ last_digit_divisible_by_4 n}.card = 3 := by
  sorry

end num_possibilities_l114_114995


namespace abs_x_minus_one_iff_x_in_interval_l114_114963

theorem abs_x_minus_one_iff_x_in_interval (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 :=
by
  sorry

end abs_x_minus_one_iff_x_in_interval_l114_114963


namespace inequality_solution_l114_114879

theorem inequality_solution (x : ℝ) (hx1 : x ≥ -1/2) (hx2 : x ≠ 0) :
  (4 * x^2 / (1 - Real.sqrt (1 + 2 * x))^2 < 2 * x + 9) ↔ 
  (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 45/8) :=
by
  sorry

end inequality_solution_l114_114879


namespace mrs_sheridan_fish_count_l114_114507

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

end mrs_sheridan_fish_count_l114_114507


namespace small_box_dolls_l114_114576

theorem small_box_dolls (x : ℕ) : 
  (5 * 7 + 9 * x = 71) → x = 4 :=
by
  sorry

end small_box_dolls_l114_114576


namespace triangles_from_decagon_l114_114269

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l114_114269


namespace ways_to_distribute_balls_l114_114357

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114357


namespace put_balls_in_boxes_l114_114439

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114439


namespace ways_to_distribute_balls_l114_114412

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l114_114412


namespace gain_per_year_is_correct_l114_114674

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem gain_per_year_is_correct :
  let borrowed_amount := 7000
  let borrowed_rate := 0.04
  let borrowed_time := 2
  let borrowed_compound_freq := 1 -- annually
  
  let lent_amount := 7000
  let lent_rate := 0.06
  let lent_time := 2
  let lent_compound_freq := 2 -- semi-annually
  
  let amount_owed := compound_interest borrowed_amount borrowed_rate borrowed_compound_freq borrowed_time
  let amount_received := compound_interest lent_amount lent_rate lent_compound_freq lent_time
  let total_gain := amount_received - amount_owed
  let gain_per_year := total_gain / lent_time
  
  gain_per_year = 153.65 :=
by
  sorry

end gain_per_year_is_correct_l114_114674


namespace distinct_balls_boxes_l114_114410

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l114_114410


namespace regular_decagon_triangle_count_l114_114254

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l114_114254


namespace age_of_other_replaced_man_l114_114800

theorem age_of_other_replaced_man (A B C D : ℕ) (h1 : A = 23) (h2 : ((52 + C + D) / 4 > (A + B + C + D) / 4)) :
  B < 29 := 
by
  sorry

end age_of_other_replaced_man_l114_114800


namespace max_bottles_drunk_l114_114747

theorem max_bottles_drunk (e b : ℕ) (h1 : e = 16) (h2 : b = 4) : 
  ∃ n : ℕ, n = 5 :=
by
  sorry

end max_bottles_drunk_l114_114747


namespace ways_to_place_balls_in_boxes_l114_114387

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l114_114387


namespace find_minimum_value_l114_114603

noncomputable def fixed_point_at_2_2 (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : Prop :=
∀ (x : ℝ), a^(2-x) + 1 = 2 ↔ x = 2

noncomputable def point_on_line (m n : ℝ) (hmn_pos : m * n > 0) : Prop :=
2 * m + 2 * n = 1

theorem find_minimum_value (m n : ℝ) (hmn_pos : m * n > 0) :
  (fixed_point_at_2_2 a ha_pos ha_ne) → (point_on_line m n hmn_pos) → (1/m + 1/n ≥ 8) :=
sorry

end find_minimum_value_l114_114603


namespace balls_into_boxes_l114_114337

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114337


namespace distinct_balls_boxes_l114_114379

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114379


namespace xy_product_approx_25_l114_114770

noncomputable def approx_eq (a b : ℝ) (ε : ℝ := 1e-6) : Prop :=
  |a - b| < ε

theorem xy_product_approx_25 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
  (hxy : x / y = 36) (hy : y = 0.8333333333333334) : approx_eq (x * y) 25 :=
by
  sorry

end xy_product_approx_25_l114_114770


namespace range_of_a_l114_114889

noncomputable def domain_f (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + a ≥ 0
noncomputable def range_g (a : ℝ) : Prop := ∀ x : ℝ, x ≤ 2 → 2^x - a ∈ Set.Ioi (0 : ℝ)

theorem range_of_a (a : ℝ) : (domain_f a ∨ range_g a) ∧ ¬(domain_f a ∧ range_g a) → (a ≥ 1 ∨ a ≤ 0) := by
  sorry

end range_of_a_l114_114889


namespace num_ways_to_distribute_balls_into_boxes_l114_114449

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l114_114449


namespace length_of_train_correct_l114_114198

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_sec

theorem length_of_train_correct :
  length_of_train 60 18 = 300.06 :=
by
  -- Placeholder for proof
  sorry

end length_of_train_correct_l114_114198


namespace number_of_triangles_in_decagon_l114_114247

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l114_114247


namespace trisha_take_home_pay_l114_114035

theorem trisha_take_home_pay :
  let hourly_wage := 15
  let hours_per_week := 40
  let weeks_per_year := 52
  let withholding_percentage := 0.2

  let annual_gross_pay := hourly_wage * hours_per_week * weeks_per_year
  let withholding_amount := annual_gross_pay * withholding_percentage
  let take_home_pay := annual_gross_pay - withholding_amount

  take_home_pay = 24960 :=
by
  sorry

end trisha_take_home_pay_l114_114035


namespace n_consecutive_even_sum_l114_114786

theorem n_consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) : 
  ∃ (a : ℕ), (n * (n - 1)^(k - 1)) = (2 * a + (2 * a + 2 * (n - 1))) / 2 * n :=
by
  sorry

end n_consecutive_even_sum_l114_114786


namespace find_n_l114_114868

theorem find_n (k : ℤ) : 
  ∃ n : ℤ, (n = 35 * k + 24) ∧ (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) :=
by
  -- Proof goes here
  sorry

end find_n_l114_114868


namespace salon_revenue_l114_114690

noncomputable def revenue (num_customers first_visit second_visit third_visit : ℕ) (first_charge second_charge : ℕ) : ℕ :=
  num_customers * first_charge + second_visit * second_charge + third_visit * second_charge

theorem salon_revenue : revenue 100 100 30 10 10 8 = 1320 :=
by
  unfold revenue
  -- The proof will continue here.
  sorry

end salon_revenue_l114_114690


namespace sin_complementary_angle_l114_114745

theorem sin_complementary_angle (θ : ℝ) (h1 : Real.tan θ = 2) (h2 : Real.cos θ < 0) : 
  Real.sin (Real.pi / 2 - θ) = -Real.sqrt 5 / 5 :=
sorry

end sin_complementary_angle_l114_114745


namespace sin_theta_value_l114_114765

theorem sin_theta_value (θ : ℝ) (h₁ : θ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h₂ : Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8) : Real.sin θ = 3 / 4 :=
sorry

end sin_theta_value_l114_114765


namespace find_annual_interest_rate_l114_114669

theorem find_annual_interest_rate (P0 P1 P2 : ℝ) (r1 r : ℝ) :
  P0 = 12000 →
  r1 = 10 →
  P1 = P0 * (1 + (r1 / 100) / 2) →
  P1 = 12600 →
  P2 = 13260 →
  P1 * (1 + (r / 200)) = P2 →
  r = 10.476 :=
by
  intros hP0 hr1 hP1 hP1val hP2 hP1P2
  sorry

end find_annual_interest_rate_l114_114669


namespace min_shirts_to_save_money_l114_114214

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 60 + 11 * x < 20 + 15 * x ∧ (∀ y : ℕ, 60 + 11 * y < 20 + 15 * y → y ≥ x) ∧ x = 11 :=
by
  sorry

end min_shirts_to_save_money_l114_114214


namespace distinct_balls_boxes_l114_114381

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114381


namespace exchange_ways_100_yuan_l114_114191

theorem exchange_ways_100_yuan : ∃ n : ℕ, n = 6 ∧ (∀ (x y : ℕ), 20 * x + 10 * y = 100 ↔ y = 10 - 2 * x):=
by
  sorry

end exchange_ways_100_yuan_l114_114191


namespace value_of_1_minus_a_l114_114614

theorem value_of_1_minus_a (a : ℤ) (h : a = -(-6)) : 1 - a = -5 := 
by 
  sorry

end value_of_1_minus_a_l114_114614


namespace cubic_polynomial_at_zero_l114_114147

noncomputable def f (x : ℝ) : ℝ := by sorry

theorem cubic_polynomial_at_zero :
  (∃ f : ℝ → ℝ, f 2 = 15 ∨ f 2 = -15 ∧
                 f 4 = 15 ∨ f 4 = -15 ∧
                 f 5 = 15 ∨ f 5 = -15 ∧
                 f 6 = 15 ∨ f 6 = -15 ∧
                 f 8 = 15 ∨ f 8 = -15 ∧
                 f 9 = 15 ∨ f 9 = -15 ∧
                 ∀ x, ∃ c a b d, f x = c * x^3 + a * x^2 + b * x + d ) →
  |f 0| = 135 :=
by sorry

end cubic_polynomial_at_zero_l114_114147


namespace arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l114_114533

-- Definitions based on conditions in A)
def students : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def A : Char := 'A'
def B : Char := 'B'
def C : Char := 'C'
def D : Char := 'D'
def E : Char := 'E'
def F : Char := 'F'
def G : Char := 'G'

-- Holistic theorem statements for each question derived from the correct answers in B)
theorem arrangement_A_and_B_adjacent :
  ∃ (n : ℕ), n = 1440 := sorry

theorem arrangement_A_B_and_C_adjacent :
  ∃ (n : ℕ), n = 720 := sorry

theorem arrangement_A_and_B_adjacent_C_not_ends :
  ∃ (n : ℕ), n = 960 := sorry

theorem arrangement_ABC_and_DEFG_units :
  ∃ (n : ℕ), n = 288 := sorry

end arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l114_114533


namespace acute_angle_sum_l114_114748

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = (2 * Real.sqrt 5) / 5) (h2 : Real.sin β = (3 * Real.sqrt 10) / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end acute_angle_sum_l114_114748


namespace balls_into_boxes_l114_114335

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l114_114335


namespace distinct_balls_boxes_l114_114378

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l114_114378


namespace bear_problem_l114_114222

variables (w b br : ℕ)

theorem bear_problem 
    (h1 : b = 2 * w)
    (h2 : br = b + 40)
    (h3 : w + b + br = 190) :
    b = 60 :=
by
  sorry

end bear_problem_l114_114222


namespace olly_needs_24_shoes_l114_114006

-- Define the number of paws for different types of pets
def dogs : ℕ := 3
def cats : ℕ := 2
def ferret : ℕ := 1

def paws_per_dog : ℕ := 4
def paws_per_cat : ℕ := 4
def paws_per_ferret : ℕ := 4

-- The theorem we want to prove
theorem olly_needs_24_shoes : 
  dogs * paws_per_dog + cats * paws_per_cat + ferret * paws_per_ferret = 24 :=
by
  sorry

end olly_needs_24_shoes_l114_114006


namespace ways_to_distribute_balls_l114_114358

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l114_114358


namespace isabel_reading_homework_pages_l114_114133

-- Definitions for the given problem
def num_math_pages := 2
def problems_per_page := 5
def total_problems := 30

-- Calculation based on conditions
def math_problems := num_math_pages * problems_per_page
def reading_problems := total_problems - math_problems

-- The statement to be proven
theorem isabel_reading_homework_pages : (reading_problems / problems_per_page) = 4 :=
by
  -- The proof would go here.
  sorry

end isabel_reading_homework_pages_l114_114133


namespace ways_to_distribute_balls_l114_114349

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l114_114349


namespace find_divisor_l114_114965

theorem find_divisor : exists d : ℕ, 
  (∀ x : ℕ, x ≥ 10 ∧ x ≤ 1000000 → x % d = 0) ∧ 
  (10 + 999990 * d/111110 = 1000000) ∧
  d = 9 := by
  sorry

end find_divisor_l114_114965


namespace put_balls_in_boxes_l114_114444

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114444


namespace find_abc_triplet_l114_114107

theorem find_abc_triplet (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_order : a < b ∧ b < c) 
  (h_eqn : (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = (a + b + c) / 2) :
  ∃ d : ℕ, d > 0 ∧ ((a = d ∧ b = 2 * d ∧ c = 3 * d) ∨ (a = d ∧ b = 3 * d ∧ c = 6 * d)) :=
  sorry

end find_abc_triplet_l114_114107


namespace homework_problem1_homework_problem2_l114_114527

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end homework_problem1_homework_problem2_l114_114527


namespace ways_to_put_balls_in_boxes_l114_114425

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l114_114425


namespace sandra_oranges_l114_114094

theorem sandra_oranges (S E B: ℕ) (h1: E = 7 * S) (h2: E = 252) (h3: B = 12) : S / B = 3 := by
  sorry

end sandra_oranges_l114_114094


namespace unique_pair_exists_l114_114706

theorem unique_pair_exists :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (a + b + (Nat.gcd a b)^2 = Nat.lcm a b) ∧
  (Nat.lcm a b = 2 * Nat.lcm (a - 1) b) ∧
  (a, b) = (6, 15) :=
sorry

end unique_pair_exists_l114_114706


namespace distinguish_ball_box_ways_l114_114468

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l114_114468


namespace problem_l114_114312

noncomputable def discriminant (p q : ℝ) : ℝ := p^2 - 4 * q
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (p q : ℝ) (hq : q = -2 * p - 5) :
  (quadratic 1 p (q + 1) 2 = 0) →
  q = -2 * p - 5 ∧
  discriminant p q > 0 ∧
  (discriminant p (q + 1) = 0 → 
    (p = -4 ∧ q = 3 ∧ ∀ x : ℝ, quadratic 1 p q x = 0 ↔ (x = 1 ∨ x = 3))) :=
by
  intro hroot_eq
  sorry

end problem_l114_114312


namespace expected_messages_xiaoli_l114_114091

noncomputable def expected_greeting_messages (probs : List ℝ) (counts : List ℕ) : ℝ :=
  List.sum (List.zipWith (λ p c => p * c) probs counts)

theorem expected_messages_xiaoli :
  expected_greeting_messages [1, 0.8, 0.5, 0] [8, 15, 14, 3] = 27 :=
by
  -- The proof will use the expected value formula
  sorry

end expected_messages_xiaoli_l114_114091


namespace trapezoid_diagonals_l114_114639

theorem trapezoid_diagonals (a c b d e f : ℝ) (h1 : a ≠ c):
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧ 
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := 
by
  sorry

end trapezoid_diagonals_l114_114639


namespace simplify_expression_l114_114499

variables {a b c x : ℝ}
hypothesis h₁ : a ≠ b
hypothesis h₂ : b ≠ c
hypothesis h₃ : c ≠ a

def p (x : ℝ) : ℝ :=
  (x + a)^4 / ((a - b) * (a - c)) + 
  (x + b)^4 / ((b - a) * (b - c)) + 
  (x + c)^4 / ((c - a) * (c - b))

theorem simplify_expression : p x = a + b + c + 3 * x^2 :=
by sorry

end simplify_expression_l114_114499


namespace num_students_play_cricket_l114_114158

theorem num_students_play_cricket 
  (total_students : ℕ)
  (play_football : ℕ)
  (play_both : ℕ)
  (play_neither : ℕ)
  (C : ℕ) :
  total_students = 450 →
  play_football = 325 →
  play_both = 100 →
  play_neither = 50 →
  (total_students - play_neither = play_football + C - play_both) →
  C = 175 := by
  intros h0 h1 h2 h3 h4
  sorry

end num_students_play_cricket_l114_114158


namespace joe_first_lift_weight_l114_114662

variable (x y : ℕ)

def conditions : Prop :=
  (x + y = 1800) ∧ (2 * x = y + 300)

theorem joe_first_lift_weight (h : conditions x y) : x = 700 := by
  sorry

end joe_first_lift_weight_l114_114662


namespace age_ratio_in_ten_years_l114_114487

-- Definitions of given conditions
variable (A : ℕ) (B : ℕ)
axiom age_condition : A = 20
axiom sum_of_ages : A + 10 + (B + 10) = 45

-- Theorem and proof skeleton for the ratio of ages in ten years.
theorem age_ratio_in_ten_years (A B : ℕ) (hA : A = 20) (hSum : A + 10 + (B + 10) = 45) :
  (A + 10) / (B + 10) = 2 := by
  sorry

end age_ratio_in_ten_years_l114_114487


namespace put_balls_in_boxes_l114_114445

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l114_114445
