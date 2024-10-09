import Mathlib

namespace wilfred_carrots_on_tuesday_l752_75205

theorem wilfred_carrots_on_tuesday :
  ∀ (carrots_eaten_Wednesday carrots_eaten_Thursday total_carrots desired_total: ℕ),
    carrots_eaten_Wednesday = 6 →
    carrots_eaten_Thursday = 5 →
    desired_total = 15 →
    desired_total - (carrots_eaten_Wednesday + carrots_eaten_Thursday) = 4 :=
by
  intros
  sorry

end wilfred_carrots_on_tuesday_l752_75205


namespace tan_of_angle_through_point_l752_75258

theorem tan_of_angle_through_point (α : ℝ) (hα : ∃ x y : ℝ, (x = 1) ∧ (y = 2) ∧ (y/x = (Real.sin α) / (Real.cos α))) :
  Real.tan α = 2 :=
sorry

end tan_of_angle_through_point_l752_75258


namespace negation_proposition_of_cube_of_odd_is_odd_l752_75252

def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_proposition_of_cube_of_odd_is_odd :
  (¬ ∀ n : ℤ, odd n → odd (n^3)) ↔ (∃ n : ℤ, odd n ∧ ¬ odd (n^3)) :=
by
  sorry

end negation_proposition_of_cube_of_odd_is_odd_l752_75252


namespace natalia_apartment_number_unit_digit_l752_75260

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def true_statements (n : ℕ) : Prop :=
  (n % 3 = 0 → true) ∧   -- Statement (1): divisible by 3
  (∃ k : ℕ, k^2 = n → true) ∧  -- Statement (2): square number
  (n % 2 = 1 → true) ∧   -- Statement (3): odd
  (n % 10 = 4 → true)     -- Statement (4): ends in 4

def three_out_of_four_true (n : ℕ) : Prop :=
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 ≠ 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 ≠ 1 ∧ n % 10 = 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 ≠ n) ∧ n % 2 = 1 ∧ n % 10 = 4) ∨
  (n % 3 ≠ 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 = 4)

theorem natalia_apartment_number_unit_digit :
  ∀ n : ℕ, two_digit_number n → three_out_of_four_true n → n % 10 = 1 :=
by sorry

end natalia_apartment_number_unit_digit_l752_75260


namespace find_a_l752_75294

theorem find_a (a : ℝ) (extreme_at_neg_2 : ∀ x : ℝ, (3 * a * x^2 + 2 * x) = 0 → x = -2) :
    a = 1 / 3 :=
sorry

end find_a_l752_75294


namespace number_of_integers_l752_75242

theorem number_of_integers (n : ℕ) (h₁ : 300 < n^2) (h₂ : n^2 < 1200) : ∃ k, k = 17 :=
by
  sorry

end number_of_integers_l752_75242


namespace find_k_values_l752_75273

theorem find_k_values (k : ℝ) : 
  ((2 * 1 + 3 * k = 0) ∨
   (1 * 2 + (3 - k) * 3 = 0) ∨
   (1 * 1 + (3 - k) * k = 0)) →
   (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2) := 
by
  sorry

end find_k_values_l752_75273


namespace tea_to_cheese_ratio_l752_75225

-- Definitions based on conditions
def total_cost : ℝ := 21
def tea_cost : ℝ := 10
def butter_to_cheese_ratio : ℝ := 0.8
def bread_to_butter_ratio : ℝ := 0.5

-- Main theorem statement
theorem tea_to_cheese_ratio (B C Br : ℝ) (hBr : Br = B * bread_to_butter_ratio) (hB : B = butter_to_cheese_ratio * C) (hTotal : B + Br + C + tea_cost = total_cost) :
  10 / C = 2 :=
  sorry

end tea_to_cheese_ratio_l752_75225


namespace complete_residue_system_l752_75233

theorem complete_residue_system {m n : ℕ} {a : ℕ → ℕ} {b : ℕ → ℕ}
  (h₁ : ∀ i j, 1 ≤ i → i ≤ m → 1 ≤ j → j ≤ n → (a i) * (b j) % (m * n) ≠ (a i) * (b j)) :
  (∀ i₁ i₂, 1 ≤ i₁ → i₁ ≤ m → 1 ≤ i₂ → i₂ ≤ m → i₁ ≠ i₂ → (a i₁ % m ≠ a i₂ % m)) ∧ 
  (∀ j₁ j₂, 1 ≤ j₁ → j₁ ≤ n → 1 ≤ j₂ → j₂ ≤ n → j₁ ≠ j₂ → (b j₁ % n ≠ b j₂ % n)) := sorry

end complete_residue_system_l752_75233


namespace isosceles_trapezoid_side_length_l752_75208

theorem isosceles_trapezoid_side_length (A b1 b2 : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 48) (hb1 : b1 = 9) (hb2 : b2 = 15) 
  (h_area : A = 1 / 2 * (b1 + b2) * h) 
  (h_h : h = 4)
  (h_s : s^2 = h^2 + ((b2 - b1) / 2)^2) :
  s = 5 :=
by sorry

end isosceles_trapezoid_side_length_l752_75208


namespace sum_xyz_l752_75200

variables (x y z : ℤ)

theorem sum_xyz (h1 : y = 3 * x) (h2 : z = 3 * y - x) : x + y + z = 12 * x :=
by 
  -- skip the proof
  sorry

end sum_xyz_l752_75200


namespace subset_zero_in_A_l752_75298

def A := { x : ℝ | x > -1 }

theorem subset_zero_in_A : {0} ⊆ A :=
by sorry

end subset_zero_in_A_l752_75298


namespace absolute_difference_volumes_l752_75276

/-- The absolute difference in volumes of the cylindrical tubes formed by Amy and Carlos' papers. -/
theorem absolute_difference_volumes :
  let h_A := 12
  let C_A := 10
  let r_A := C_A / (2 * Real.pi)
  let V_A := Real.pi * r_A^2 * h_A
  let h_C := 8
  let C_C := 14
  let r_C := C_C / (2 * Real.pi)
  let V_C := Real.pi * r_C^2 * h_C
  abs (V_C - V_A) = 92 / Real.pi :=
by
  sorry

end absolute_difference_volumes_l752_75276


namespace problem_min_ineq_range_l752_75291

theorem problem_min_ineq_range (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x, 1 / a + 4 / b ≥ |2 * x - 1| - |x + 1|) ∧ (1 / a + 4 / b = 9) ∧ (-7 ≤ x ∧ x ≤ 11) :=
sorry

end problem_min_ineq_range_l752_75291


namespace problem_l752_75280

variable (x y : ℝ)

theorem problem (h1 : x + 3 * y = 6) (h2 : x * y = -12) : x^2 + 9 * y^2 = 108 :=
sorry

end problem_l752_75280


namespace distance_between_foci_l752_75289

-- Define the conditions
def is_asymptote (y x : ℝ) (slope intercept : ℝ) : Prop := y = slope * x + intercept

def passes_through_point (x y x0 y0 : ℝ) : Prop := x = x0 ∧ y = y0

-- The hyperbola conditions
axiom asymptote1 : ∀ x y : ℝ, is_asymptote y x 2 3
axiom asymptote2 : ∀ x y : ℝ, is_asymptote y x (-2) 5
axiom hyperbola_passes : passes_through_point 2 9 2 9

-- The proof problem statement: distance between the foci
theorem distance_between_foci : ∀ {a b c : ℝ}, ∃ c, (c^2 = 22.75 + 22.75) → 2 * c = 2 * Real.sqrt 45.5 :=
by
  sorry

end distance_between_foci_l752_75289


namespace line_l_passes_fixed_point_line_l_perpendicular_value_a_l752_75293

variable (a : ℝ)

def line_l (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => (a + 1) * p.1 + p.2 + 2 - a = 0

def perpendicular_line : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - 3 * p.2 + 4 = 0

theorem line_l_passes_fixed_point :
  line_l a (1, -3) :=
by
  sorry

theorem line_l_perpendicular_value_a (a : ℝ) :
  (∀ p : ℝ × ℝ, perpendicular_line p → line_l a p) → 
  a = 1 / 2 :=
by
  sorry

end line_l_passes_fixed_point_line_l_perpendicular_value_a_l752_75293


namespace roots_quadratic_solution_l752_75265

theorem roots_quadratic_solution (α β : ℝ) (hα : α^2 - 3*α - 2 = 0) (hβ : β^2 - 3*β - 2 = 0) :
  3*α^3 + 8*β^4 = 1229 := by
  sorry

end roots_quadratic_solution_l752_75265


namespace factorize_eq_l752_75229

theorem factorize_eq (x : ℝ) : 2 * x^3 - 8 * x = 2 * x * (x + 2) * (x - 2) := 
by
  sorry

end factorize_eq_l752_75229


namespace fourier_series_decomposition_l752_75228

open Real

noncomputable def f : ℝ → ℝ :=
  λ x => if (x < 0) then -1 else (if (0 < x) then 1/2 else 0)

theorem fourier_series_decomposition :
    ∀ x, -π ≤ x ∧ x ≤ π →
         f x = -1/4 + (3/π) * ∑' k, (sin ((2*k+1)*x)) / (2*k+1) :=
by
  sorry

end fourier_series_decomposition_l752_75228


namespace angle_C_length_CD_area_range_l752_75274

-- 1. Prove C = π / 3 given (2a - b)cos C = c cos B
theorem angle_C (a b c : ℝ) (A B C : ℝ) (h : (2 * a - b) * Real.cos C = c * Real.cos B) : 
  C = Real.pi / 3 := sorry

-- 2. Prove the length of CD is 6√3 / 5 given a = 2, b = 3, and CD is the angle bisector of angle C
theorem length_CD (a b x : ℝ) (C D : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : x = (6 * Real.sqrt 3) / 5) : 
  x = (6 * Real.sqrt 3) / 5 := sorry

-- 3. Prove the range of values for the area of acute triangle ABC is (8√3 / 3, 4√3] given a cos B + b cos A = 4
theorem area_range (a b : ℝ) (A B C : ℝ) (S : Set ℝ) (h1 : a * Real.cos B + b * Real.cos A = 4) 
  (h2 : S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3)) : 
  S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3) := sorry

end angle_C_length_CD_area_range_l752_75274


namespace angles_terminal_yaxis_l752_75210

theorem angles_terminal_yaxis :
  {θ : ℝ | ∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 2 ∨ θ = 2 * k * Real.pi + 3 * Real.pi / 2} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end angles_terminal_yaxis_l752_75210


namespace problem_proof_l752_75221

variables (a b : ℝ) (n : ℕ)

theorem problem_proof (h1: a > 0) (h2: b > 0) (h3: a + b = 1) (h4: n >= 2) :
  3/2 < 1/(a^n + 1) + 1/(b^n + 1) ∧ 1/(a^n + 1) + 1/(b^n + 1) ≤ (2^(n+1))/(2^n + 1) := sorry

end problem_proof_l752_75221


namespace container_capacity_l752_75290

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 36 = 0.75 * C) : 
  C = 80 :=
sorry

end container_capacity_l752_75290


namespace max_balls_in_cube_l752_75235

noncomputable def volume_of_cube : ℝ := (5 : ℝ)^3

noncomputable def volume_of_ball : ℝ := (4 / 3) * Real.pi * (1 : ℝ)^3

theorem max_balls_in_cube (c_length : ℝ) (b_radius : ℝ) (h1 : c_length = 5)
  (h2 : b_radius = 1) : 
  ⌊volume_of_cube / volume_of_ball⌋ = 29 := 
by
  sorry

end max_balls_in_cube_l752_75235


namespace squares_difference_l752_75297

theorem squares_difference (n : ℕ) (h : n > 0) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := 
by 
  sorry

end squares_difference_l752_75297


namespace sequence_8th_term_is_sqrt23_l752_75243

noncomputable def sequence_term (n : ℕ) : ℝ := Real.sqrt (2 + 3 * (n - 1))

theorem sequence_8th_term_is_sqrt23 : sequence_term 8 = Real.sqrt 23 :=
by
  sorry

end sequence_8th_term_is_sqrt23_l752_75243


namespace photo_area_with_frame_l752_75248

-- Define the areas and dimensions given in the conditions
def paper_length : ℕ := 12
def paper_width : ℕ := 8
def frame_width : ℕ := 2

-- Define the dimensions of the photo including the frame
def photo_length_with_frame : ℕ := paper_length + 2 * frame_width
def photo_width_with_frame : ℕ := paper_width + 2 * frame_width

-- The theorem statement proving the area of the wall photo including the frame
theorem photo_area_with_frame :
  (photo_length_with_frame * photo_width_with_frame) = 192 := by
  sorry

end photo_area_with_frame_l752_75248


namespace find_evening_tickets_l752_75259

noncomputable def matinee_price : ℕ := 5
noncomputable def evening_price : ℕ := 12
noncomputable def threeD_price : ℕ := 20
noncomputable def matinee_tickets : ℕ := 200
noncomputable def threeD_tickets : ℕ := 100
noncomputable def total_revenue : ℕ := 6600

theorem find_evening_tickets (E : ℕ) (hE : total_revenue = matinee_tickets * matinee_price + E * evening_price + threeD_tickets * threeD_price) :
  E = 300 :=
by
  sorry

end find_evening_tickets_l752_75259


namespace find_price_of_100_apples_l752_75250

noncomputable def price_of_100_apples (P : ℕ) : Prop :=
  (12000 / P) - (12000 / (P + 4)) = 5

theorem find_price_of_100_apples : price_of_100_apples 96 :=
by
  sorry

end find_price_of_100_apples_l752_75250


namespace rectangle_measurement_error_l752_75262

theorem rectangle_measurement_error 
  (L W : ℝ)
  (measured_length : ℝ := 1.05 * L)
  (measured_width : ℝ := 0.96 * W)
  (actual_area : ℝ := L * W)
  (calculated_area : ℝ := measured_length * measured_width)
  (error : ℝ := calculated_area - actual_area) :
  ((error / actual_area) * 100) = 0.8 :=
sorry

end rectangle_measurement_error_l752_75262


namespace find_rate_of_current_l752_75240

noncomputable def rate_of_current : ℝ := 
  let speed_still_water := 42
  let distance_downstream := 33.733333333333334
  let time_hours := 44 / 60
  (distance_downstream / time_hours) - speed_still_water

theorem find_rate_of_current : rate_of_current = 4 :=
by sorry

end find_rate_of_current_l752_75240


namespace david_number_sum_l752_75204

theorem david_number_sum :
  ∃ (x y : ℕ), (10 ≤ x ∧ x < 100) ∧ (100 ≤ y ∧ y < 1000) ∧ (1000 * x + y = 4 * x * y) ∧ (x + y = 266) :=
sorry

end david_number_sum_l752_75204


namespace cone_height_l752_75249

theorem cone_height (S h H Vcone Vcylinder : ℝ)
  (hcylinder_height : H = 9)
  (hvolumes : Vcone = Vcylinder)
  (hbase_areas : S = S)
  (hV_cone : Vcone = (1 / 3) * S * h)
  (hV_cylinder : Vcylinder = S * H) : h = 27 :=
by
  -- sorry is used here to indicate missing proof steps which are predefined as unnecessary
  sorry

end cone_height_l752_75249


namespace line_slope_translation_l752_75270

theorem line_slope_translation (k : ℝ) (b : ℝ) :
  (∀ x y : ℝ, y = k * x + b → y = k * (x - 3) + (b + 2)) → k = 2 / 3 :=
by
  intro h
  sorry

end line_slope_translation_l752_75270


namespace colbert_materials_needed_l752_75269

def wooden_planks_needed (total_needed quarter_in_stock : ℕ) : ℕ :=
  let total_purchased := total_needed - quarter_in_stock / 4
  (total_purchased + 7) / 8 -- ceil division by 8

def iron_nails_needed (total_needed thirty_percent_provided : ℕ) : ℕ :=
  let total_purchased := total_needed - total_needed * thirty_percent_provided / 100
  (total_purchased + 24) / 25 -- ceil division by 25

def fabric_needed (total_needed third_provided : ℚ) : ℚ :=
  total_needed - total_needed / third_provided

def metal_brackets_needed (total_needed in_stock multiple : ℕ) : ℕ :=
  let total_purchased := total_needed - in_stock
  (total_purchased + multiple - 1) / multiple * multiple -- ceil to next multiple of 5

theorem colbert_materials_needed :
  wooden_planks_needed 250 62 = 24 ∧
  iron_nails_needed 500 30 = 14 ∧
  fabric_needed 10 3 = 6.67 ∧
  metal_brackets_needed 40 10 5 = 30 :=
by sorry

end colbert_materials_needed_l752_75269


namespace combined_weight_of_olivers_bags_l752_75271

theorem combined_weight_of_olivers_bags (w_james : ℕ) (w_oliver : ℕ) (w_combined : ℕ) 
  (h1 : w_james = 18) 
  (h2 : w_oliver = w_james / 6) 
  (h3 : w_combined = 2 * w_oliver) : 
  w_combined = 6 := 
by
  sorry

end combined_weight_of_olivers_bags_l752_75271


namespace interest_rate_first_year_l752_75246

theorem interest_rate_first_year (R : ℚ)
  (principal : ℚ := 7000)
  (final_amount : ℚ := 7644)
  (time_period_first_year : ℚ := 1)
  (time_period_second_year : ℚ := 1)
  (rate_second_year : ℚ := 5) :
  principal + (principal * R * time_period_first_year / 100) + 
  ((principal + (principal * R * time_period_first_year / 100)) * rate_second_year * time_period_second_year / 100) = final_amount →
  R = 4 := 
by {
  sorry
}

end interest_rate_first_year_l752_75246


namespace students_with_both_l752_75241

-- Define the problem conditions as given in a)
def total_students : ℕ := 50
def students_with_bike : ℕ := 28
def students_with_scooter : ℕ := 35

-- State the theorem
theorem students_with_both :
  ∃ (n : ℕ), n = 13 ∧ total_students = students_with_bike + students_with_scooter - n := by
  sorry

end students_with_both_l752_75241


namespace quotient_is_76_l752_75284

def original_number : ℕ := 12401
def divisor : ℕ := 163
def remainder : ℕ := 13

theorem quotient_is_76 : (original_number - remainder) / divisor = 76 :=
by
  sorry

end quotient_is_76_l752_75284


namespace star_intersections_l752_75203

theorem star_intersections (n k : ℕ) (h_coprime : Nat.gcd n k = 1) (h_n_ge_5 : 5 ≤ n) (h_k_lt_n_div_2 : k < n / 2) :
    k = 25 → n = 2018 → n * (k - 1) = 48432 := by
  intros
  sorry

end star_intersections_l752_75203


namespace worst_ranking_l752_75286

theorem worst_ranking (teams : Fin 25 → Nat) (A : Fin 25)
  (round_robin : ∀ i j, i ≠ j → teams i + teams j ≤ 4)
  (most_goals : ∀ i, i ≠ A → teams A > teams i)
  (fewest_goals : ∀ i, i ≠ A → teams i > teams A) :
  ∃ ranking : Fin 25 → Fin 25, ranking A = 24 :=
by
  sorry

end worst_ranking_l752_75286


namespace luke_money_last_weeks_l752_75283

theorem luke_money_last_weeks (earnings_mowing : ℕ) (earnings_weed_eating : ℕ) (weekly_spending : ℕ) 
  (h1 : earnings_mowing = 9) (h2 : earnings_weed_eating = 18) (h3 : weekly_spending = 3) :
  (earnings_mowing + earnings_weed_eating) / weekly_spending = 9 :=
by sorry

end luke_money_last_weeks_l752_75283


namespace vector_expression_evaluation_l752_75227

theorem vector_expression_evaluation (θ : ℝ) :
  let a := (2 * Real.cos θ, Real.sin θ)
  let b := (1, -6)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (2 * Real.cos θ + Real.sin θ) / (Real.cos θ + 3 * Real.sin θ) = 7 / 6 :=
by
  intros a b h
  sorry

end vector_expression_evaluation_l752_75227


namespace range_of_m_l752_75230

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x + 5 < 4 * x - 1 ∧ x > m → x > 2) → m ≤ 2 :=
by
  intro h
  have h₁ := h 2
  sorry

end range_of_m_l752_75230


namespace NineChaptersProblem_l752_75256

-- Conditions: Assign the given conditions to variables
variables (x y : Int)
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Proof problem: Prove that the system of equations is consistent with the given conditions
theorem NineChaptersProblem : condition1 x y ∧ condition2 x y := sorry

end NineChaptersProblem_l752_75256


namespace range_of_a_l752_75278

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x + x^2 - 2 * a * x

theorem range_of_a (a : ℝ) (h₀ : a > 0) 
  (h₁ h₂ : ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ a - f x₂ a ≥ (3/2) - 2 * Real.log 2) : 
  a ≥ (3/2) * Real.sqrt 2 :=
sorry

end range_of_a_l752_75278


namespace regions_first_two_sets_regions_all_sets_l752_75207

-- Definitions for the problem
def triangle_regions_first_two_sets (n : ℕ) : ℕ :=
  (n + 1) * (n + 1)

def triangle_regions_all_sets (n : ℕ) : ℕ :=
  3 * n * n + 3 * n + 1

-- Proof Problem 1: Given n points on AB and AC, prove the regions are (n + 1)^2
theorem regions_first_two_sets (n : ℕ) :
  (n * (n + 1) + (n + 1)) = (n + 1) * (n + 1) :=
by sorry

-- Proof Problem 2: Given n points on AB, AC, and BC, prove the regions are 3n^2 + 3n + 1
theorem regions_all_sets (n : ℕ) :
  ((n + 1) * (n + 1) + n * (2 * n + 1)) = 3 * n * n + 3 * n + 1 :=
by sorry

end regions_first_two_sets_regions_all_sets_l752_75207


namespace opposite_quotient_l752_75285

theorem opposite_quotient {a b : ℝ} (h1 : a ≠ b) (h2 : a = -b) : a / b = -1 := 
sorry

end opposite_quotient_l752_75285


namespace sum_of_cubes_divisible_l752_75234

theorem sum_of_cubes_divisible (a b c : ℤ) (h : (a + b + c) % 3 = 0) : 
  (a^3 + b^3 + c^3) % 3 = 0 := 
by sorry

end sum_of_cubes_divisible_l752_75234


namespace total_ages_l752_75268

variable (Craig_age Mother_age : ℕ)

theorem total_ages (h1 : Craig_age = 16) (h2 : Mother_age = Craig_age + 24) : Craig_age + Mother_age = 56 := by
  sorry

end total_ages_l752_75268


namespace chef_makes_10_cakes_l752_75296

def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

theorem chef_makes_10_cakes :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 := by
  sorry

end chef_makes_10_cakes_l752_75296


namespace gandalf_reachability_l752_75292

theorem gandalf_reachability :
  ∀ (k : ℕ), ∃ (s : ℕ → ℕ) (m : ℕ), (s 0 = 1) ∧ (s m = k) ∧ (∀ i < m, s (i + 1) = 2 * s i ∨ s (i + 1) = 3 * s i + 1) := 
by
  sorry

end gandalf_reachability_l752_75292


namespace problem4_l752_75245

theorem problem4 (a : ℝ) : (a-1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := 
by sorry

end problem4_l752_75245


namespace range_of_k_l752_75224

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > 0 → (k+4) * x < 0) → k < -4 :=
by
  sorry

end range_of_k_l752_75224


namespace range_of_4a_minus_2b_l752_75247

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b)
  (h2 : a - b ≤ 2)
  (h3 : 2 ≤ a + b)
  (h4 : a + b ≤ 4) : 
  5 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 10 :=
by
  sorry

end range_of_4a_minus_2b_l752_75247


namespace tan_105_eq_neg2_sub_sqrt3_l752_75222

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l752_75222


namespace number_of_interior_diagonals_of_dodecahedron_l752_75237

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end number_of_interior_diagonals_of_dodecahedron_l752_75237


namespace solve_for_x_l752_75253

theorem solve_for_x
  (x y : ℝ)
  (h1 : x + 2 * y = 100)
  (h2 : y = 25) :
  x = 50 :=
by
  sorry

end solve_for_x_l752_75253


namespace total_number_of_games_l752_75217

theorem total_number_of_games (n : ℕ) (k : ℕ) (teams : Finset ℕ)
  (h_n : n = 8) (h_k : k = 2) (h_teams : teams.card = n) :
  (teams.card.choose k) = 28 :=
by
  sorry

end total_number_of_games_l752_75217


namespace number_of_division_games_l752_75239

theorem number_of_division_games (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5) (h3 : 4 * N + 5 * M = 100) :
  4 * N = 60 :=
by
  sorry

end number_of_division_games_l752_75239


namespace maximal_s_value_l752_75288

noncomputable def max_tiles_sum (a b c : ℕ) : ℕ := a + c

theorem maximal_s_value :
  ∃ s : ℕ, 
    ∃ a b c : ℕ, 
      4 * a + 4 * c + 5 * b = 3986000 ∧ 
      s = max_tiles_sum a b c ∧ 
      s = 996500 := 
    sorry

end maximal_s_value_l752_75288


namespace selection_probabilities_l752_75212

-- Define the probabilities of selection for Ram, Ravi, and Rani
def prob_ram : ℚ := 5 / 7
def prob_ravi : ℚ := 1 / 5
def prob_rani : ℚ := 3 / 4

-- State the theorem that combines these probabilities
theorem selection_probabilities : prob_ram * prob_ravi * prob_rani = 3 / 28 :=
by
  sorry


end selection_probabilities_l752_75212


namespace man_walking_rate_l752_75295

theorem man_walking_rate (x : ℝ) 
  (woman_rate : ℝ := 15)
  (woman_time_after_passing : ℝ := 2 / 60)
  (man_time_to_catch_up : ℝ := 4 / 60)
  (distance_woman : ℝ := woman_rate * woman_time_after_passing)
  (distance_man : ℝ := x * man_time_to_catch_up)
  (h : distance_man = distance_woman) :
  x = 7.5 :=
sorry

end man_walking_rate_l752_75295


namespace employee_payment_correct_l752_75254

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the percentage markup for retail price
def markup_percentage : ℝ := 0.20

-- Define the retail_price based on wholesale cost and markup percentage
def retail_price : ℝ := wholesale_cost + (markup_percentage * wholesale_cost)

-- Define the employee discount percentage
def discount_percentage : ℝ := 0.20

-- Define the discount amount based on retail price and discount percentage
def discount_amount : ℝ := retail_price * discount_percentage

-- Define the final price the employee pays after applying the discount
def employee_price : ℝ := retail_price - discount_amount

-- State the theorem to prove
theorem employee_payment_correct :
  employee_price = 192 :=
  by
    sorry

end employee_payment_correct_l752_75254


namespace sock_combination_count_l752_75232

noncomputable def numSockCombinations : Nat :=
  let striped := 4
  let solid := 4
  let checkered := 4
  let striped_and_solid := striped * solid
  let striped_and_checkered := striped * checkered
  striped_and_solid + striped_and_checkered

theorem sock_combination_count :
  numSockCombinations = 32 :=
by
  unfold numSockCombinations
  sorry

end sock_combination_count_l752_75232


namespace Martiza_study_time_l752_75267

theorem Martiza_study_time :
  ∀ (x : ℕ),
  (30 * x + 30 * 25 = 20 * 60) →
  x = 15 :=
by
  intros x h
  sorry

end Martiza_study_time_l752_75267


namespace absolute_value_sum_l752_75220

theorem absolute_value_sum (a b : ℤ) (h_a : |a| = 5) (h_b : |b| = 3) : 
  (a + b = 8) ∨ (a + b = 2) ∨ (a + b = -2) ∨ (a + b = -8) :=
by
  sorry

end absolute_value_sum_l752_75220


namespace maria_trip_distance_l752_75266

theorem maria_trip_distance
  (D : ℝ)
  (h1 : D/2 = D/8 + 210) :
  D = 560 :=
sorry

end maria_trip_distance_l752_75266


namespace calc_correct_operation_l752_75275

theorem calc_correct_operation (a : ℕ) :
  (2 : ℕ) * a + (3 : ℕ) * a = (5 : ℕ) * a :=
by
  -- Proof
  sorry

end calc_correct_operation_l752_75275


namespace sum_of_extreme_T_l752_75282

theorem sum_of_extreme_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022)
  (h2 : B + M + T = 72) :
  ∃ Tmin Tmax, Tmin + Tmax = 48 ∧ Tmin ≤ T ∧ T ≤ Tmax :=
by
  sorry

end sum_of_extreme_T_l752_75282


namespace Mia_studied_fraction_l752_75213

-- Define the conditions
def total_minutes_per_day := 1440
def time_spent_watching_TV := total_minutes_per_day * 1 / 5
def time_spent_studying := 288
def remaining_time := total_minutes_per_day - time_spent_watching_TV
def fraction_studying := time_spent_studying / remaining_time

-- State the proof goal
theorem Mia_studied_fraction : fraction_studying = 1 / 4 := by
  sorry

end Mia_studied_fraction_l752_75213


namespace range_of_m_l752_75214

theorem range_of_m (x y m : ℝ) : (∃ (x y : ℝ), x + y^2 - x + y + m = 0) → m < 1/2 :=
by
  sorry

end range_of_m_l752_75214


namespace linoleum_cut_rearrange_l752_75231

def linoleum : Type := sorry -- placeholder for the specific type of the linoleum piece

def A : linoleum := sorry -- define piece A
def B : linoleum := sorry -- define piece B

def cut_and_rearrange (L : linoleum) (A B : linoleum) : Prop :=
  -- Define the proposition that pieces A and B can be rearranged into an 8x8 square
  sorry

theorem linoleum_cut_rearrange (L : linoleum) (A B : linoleum) :
  cut_and_rearrange L A B :=
sorry

end linoleum_cut_rearrange_l752_75231


namespace helen_cookies_till_last_night_l752_75223

theorem helen_cookies_till_last_night 
  (cookies_yesterday : Nat := 31) 
  (cookies_day_before_yesterday : Nat := 419) : 
  cookies_yesterday + cookies_day_before_yesterday = 450 := 
by
  sorry

end helen_cookies_till_last_night_l752_75223


namespace max_x4_y6_l752_75215

noncomputable def maximum_product (x y : ℝ) := x^4 * y^6

theorem max_x4_y6 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 100) :
  maximum_product x y ≤ maximum_product 40 60 := sorry

end max_x4_y6_l752_75215


namespace fraction_equation_correct_l752_75209

theorem fraction_equation_correct : (1 / 2 - 1 / 6) / (1 / 6009) = 2003 := by
  sorry

end fraction_equation_correct_l752_75209


namespace abc_inequality_l752_75277

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a^2 < 16 * b * c) (h2 : b^2 < 16 * c * a) (h3 : c^2 < 16 * a * b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l752_75277


namespace number_of_terms_in_ap_is_eight_l752_75281

theorem number_of_terms_in_ap_is_eight
  (n : ℕ) (a d : ℝ)
  (even : n % 2 = 0)
  (sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 24)
  (sum_even : (n / 2 : ℝ) * (2 * a + n * d) = 30)
  (last_exceeds_first : (n - 1) * d = 10.5) :
  n = 8 :=
by sorry

end number_of_terms_in_ap_is_eight_l752_75281


namespace art_of_passing_through_walls_l752_75299

theorem art_of_passing_through_walls (n : ℕ) :
  (2 * Real.sqrt (2 / 3) = Real.sqrt (2 * (2 / 3))) ∧
  (3 * Real.sqrt (3 / 8) = Real.sqrt (3 * (3 / 8))) ∧
  (4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15))) ∧
  (5 * Real.sqrt (5 / 24) = Real.sqrt (5 * (5 / 24))) →
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) →
  n = 63 :=
by
  sorry

end art_of_passing_through_walls_l752_75299


namespace estimated_students_in_sport_A_correct_l752_75226

noncomputable def total_students_surveyed : ℕ := 80
noncomputable def students_in_sport_A_surveyed : ℕ := 30
noncomputable def total_school_population : ℕ := 800
noncomputable def proportion_sport_A : ℚ := students_in_sport_A_surveyed / total_students_surveyed
noncomputable def estimated_students_in_sport_A : ℚ := total_school_population * proportion_sport_A

theorem estimated_students_in_sport_A_correct :
  estimated_students_in_sport_A = 300 :=
by
  sorry

end estimated_students_in_sport_A_correct_l752_75226


namespace find_principal_l752_75251

-- Problem conditions
variables (SI : ℚ := 4016.25) 
variables (R : ℚ := 0.08) 
variables (T : ℚ := 5)

-- The simple interest formula to find Principal
noncomputable def principal (SI : ℚ) (R : ℚ) (T : ℚ) : ℚ := SI * 100 / (R * T)

-- Lean statement to prove
theorem find_principal : principal SI R T = 10040.625 := by
  sorry

end find_principal_l752_75251


namespace hyperbola_equation_correct_l752_75211

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :=
  (x y : ℝ) -> (x^2 / 5) - (y^2 / 20) = 1

theorem hyperbola_equation_correct {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :
  hyperbola_equation a b a_pos b_pos focal_len asymptote_slope :=
by {
  sorry
}

end hyperbola_equation_correct_l752_75211


namespace angle_B_l752_75218

theorem angle_B (A B C a b c : ℝ) (h : 2 * b * (Real.cos A) = 2 * c - Real.sqrt 3 * a) :
  B = Real.pi / 6 :=
sorry

end angle_B_l752_75218


namespace candy_cost_l752_75202

theorem candy_cost (J H C : ℕ) (h1 : J + 7 = C) (h2 : H + 1 = C) (h3 : J + H < C) : C = 7 :=
by
  sorry

end candy_cost_l752_75202


namespace audrey_not_dreaming_fraction_l752_75244

theorem audrey_not_dreaming_fraction :
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  cycle1_not_dreaming + cycle2_not_dreaming + cycle3_not_dreaming + cycle4_not_dreaming = 227 / 84 :=
by
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  sorry

end audrey_not_dreaming_fraction_l752_75244


namespace calculate_value_l752_75257

theorem calculate_value 
  (a : Int) (b : Int) (c : Real) (d : Real)
  (h1 : a = -1)
  (h2 : b = 2)
  (h3 : c * d = 1) :
  a + b - c * d = 0 := 
by
  sorry

end calculate_value_l752_75257


namespace find_a_l752_75279

-- Define point
structure Point where
  x : ℝ
  y : ℝ

-- Define curves
def C1 (a x : ℝ) : ℝ := a * x^3 + 1
def C2 (P : Point) : Prop := P.x^2 + P.y^2 = 5 / 2

-- Define the tangent slope function for curve C1
def tangent_slope_C1 (a x : ℝ) : ℝ := 3 * a * x^2

-- State the problem that we need to prove
theorem find_a (a x₀ y₀ : ℝ) (h1 : y₀ = C1 a x₀) (h2 : C2 ⟨x₀, y₀⟩) (h3 : y₀ = 3 * a * x₀^3) 
  (ha_pos : 0 < a) : a = 4 := 
  by
    sorry

end find_a_l752_75279


namespace average_length_is_21_08_l752_75201

def lengths : List ℕ := [20, 21, 22]
def quantities : List ℕ := [23, 64, 32]

def total_length := List.sum (List.zipWith (· * ·) lengths quantities)
def total_quantity := List.sum quantities

def average_length := total_length / total_quantity

theorem average_length_is_21_08 :
  average_length = 2508 / 119 := by
  sorry

end average_length_is_21_08_l752_75201


namespace chess_group_players_count_l752_75216

theorem chess_group_players_count (n : ℕ)
  (h1 : ∀ (x y : ℕ), x ≠ y → ∃ k, k = 2)
  (h2 : n * (n - 1) / 2 = 45) :
  n = 10 := sorry

end chess_group_players_count_l752_75216


namespace donna_card_shop_hourly_wage_correct_l752_75236

noncomputable def donna_hourly_wage_at_card_shop : ℝ := 
  let total_earnings := 305.0
  let earnings_dog_walking := 2 * 10.0 * 5
  let earnings_babysitting := 4 * 10.0
  let earnings_card_shop := total_earnings - (earnings_dog_walking + earnings_babysitting)
  let hours_card_shop := 5 * 2
  earnings_card_shop / hours_card_shop

theorem donna_card_shop_hourly_wage_correct : donna_hourly_wage_at_card_shop = 16.50 :=
by 
  -- Skipping proof steps for the implementation
  sorry

end donna_card_shop_hourly_wage_correct_l752_75236


namespace trevor_brother_age_l752_75263

theorem trevor_brother_age :
  ∃ B : ℕ, Trevor_current_age = 11 ∧
           Trevor_future_age = 24 ∧
           Brother_future_age = 3 * Trevor_current_age ∧
           B = Brother_future_age - (Trevor_future_age - Trevor_current_age) :=
sorry

end trevor_brother_age_l752_75263


namespace general_term_sequence_l752_75255

theorem general_term_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : ∀ (m : ℕ), m ≥ 2 → a m - a (m - 1) + 1 = 0) : 
  a n = 3 - n :=
sorry

end general_term_sequence_l752_75255


namespace sum_eighth_row_l752_75287

-- Definitions based on the conditions
def sum_of_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

axiom sum_fifth_row : sum_of_interior_numbers 5 = 14
axiom sum_sixth_row : sum_of_interior_numbers 6 = 30

-- The proof problem statement
theorem sum_eighth_row : sum_of_interior_numbers 8 = 126 :=
by {
  sorry
}

end sum_eighth_row_l752_75287


namespace sequence_sum_l752_75219

-- Defining the sequence terms
variables (J K L M N O P Q R S : ℤ)
-- Condition N = 7
def N_value : Prop := N = 7
-- Condition sum of any four consecutive terms is 40
def sum_of_consecutive : Prop := 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40

-- The main theorem stating J + S = 40 given the conditions
theorem sequence_sum (N_value : N = 7) (sum_of_consecutive : 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40) : 
  J + S = 40 := sorry

end sequence_sum_l752_75219


namespace sufficient_but_not_necessary_condition_l752_75261

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : Prop) (q : Prop)
  (h₁ : p ↔ (x^2 - 1 > 0)) (h₂ : q ↔ (x < -2)) :
  (¬p → ¬q) ∧ ¬(¬q → ¬p) := 
by
  sorry

end sufficient_but_not_necessary_condition_l752_75261


namespace sum_of_roots_l752_75206

theorem sum_of_roots (x : ℝ) :
  (x + 2) * (x - 3) = 16 →
  ∃ a b : ℝ, (a ≠ x ∧ b ≠ x ∧ (x - a) * (x - b) = 0) ∧
             (a + b = 1) :=
by
  intro h
  sorry

end sum_of_roots_l752_75206


namespace greatest_divisor_l752_75238

theorem greatest_divisor (n : ℕ) (h1 : 1657 % n = 6) (h2 : 2037 % n = 5) : n = 127 :=
by
  sorry

end greatest_divisor_l752_75238


namespace all_div_by_25_form_no_div_by_35_l752_75272

noncomputable def exists_div_by_25 (M : ℕ) : Prop :=
∃ (M N : ℕ) (n : ℕ), M = 6 * 10 ^ (n - 1) + N ∧ M = 25 * N ∧ 4 * N = 10 ^ (n - 1)

theorem all_div_by_25_form :
  ∀ M, exists_div_by_25 M → (∃ k : ℕ, M = 625 * 10 ^ k) :=
by
  intro M
  intro h
  sorry

noncomputable def not_exists_div_by_35 (M : ℕ) : Prop :=
∀ (M N : ℕ) (n : ℕ), M ≠ 6 * 10 ^ (n - 1) + N ∨ M ≠ 35 * N

theorem no_div_by_35 :
  ∀ M, not_exists_div_by_35 M :=
by
  intro M
  intro h
  sorry

end all_div_by_25_form_no_div_by_35_l752_75272


namespace ratio_apples_peaches_l752_75264

theorem ratio_apples_peaches (total_fruits oranges peaches apples : ℕ)
  (h_total : total_fruits = 56)
  (h_oranges : oranges = total_fruits / 4)
  (h_peaches : peaches = oranges / 2)
  (h_apples : apples = 35) : apples / peaches = 5 := 
by
  sorry

end ratio_apples_peaches_l752_75264
