import Mathlib

namespace cube_sum_of_edges_corners_faces_eq_26_l192_192923

theorem cube_sum_of_edges_corners_faces_eq_26 :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 :=
by
  let edges := 12
  let corners := 8
  let faces := 6
  sorry

end cube_sum_of_edges_corners_faces_eq_26_l192_192923


namespace longer_diagonal_of_rhombus_l192_192504

theorem longer_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (h₁ : d1 = 12) (h₂ : area = 120) :
  d2 = 20 :=
by
  sorry

end longer_diagonal_of_rhombus_l192_192504


namespace problem_solution_l192_192478

open Real

def system_satisfied (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    (a (2 * k + 1) = (1 / a (2 * (k + n) - 1) + 1 / a (2 * k + 2))) ∧ 
    (a (2 * k + 2) = a (2 * k + 1) + a (2 * k + 3))

theorem problem_solution (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 4)
  (h2 : ∀ k, 0 ≤ k → k < 2 * n → a k > 0)
  (h3 : system_satisfied a n) :
  ∀ k, 0 ≤ k ∧ k < n → a (2 * k + 1) = 1 ∧ a (2 * k + 2) = 2 :=
sorry

end problem_solution_l192_192478


namespace route_comparison_l192_192483

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end route_comparison_l192_192483


namespace right_triangle_third_side_l192_192340

theorem right_triangle_third_side (x : ℝ) : 
  (∃ (a b c : ℝ), (a = 3 ∧ b = 4 ∧ (a^2 + b^2 = c^2 ∧ (c = x ∨ x^2 + a^2 = b^2)))) → (x = 5 ∨ x = Real.sqrt 7) :=
by 
  sorry

end right_triangle_third_side_l192_192340


namespace scaling_factor_is_2_l192_192705

-- Define the volumes of the original and scaled cubes
def V1 : ℕ := 343
def V2 : ℕ := 2744

-- Assume s1 cubed equals V1 and s2 cubed equals V2
def s1 : ℕ := 7  -- because 7^3 = 343
def s2 : ℕ := 14 -- because 14^3 = 2744

-- Scaling factor between the cubes
def scaling_factor : ℕ := s2 / s1 

-- The theorem stating the scaling factor is 2 given the volumes
theorem scaling_factor_is_2 (h1 : s1 ^ 3 = V1) (h2 : s2 ^ 3 = V2) : scaling_factor = 2 := by
  sorry

end scaling_factor_is_2_l192_192705


namespace compute_54_mul_46_l192_192866

theorem compute_54_mul_46 : (54 * 46 = 2484) :=
by sorry

end compute_54_mul_46_l192_192866


namespace sample_size_l192_192560

theorem sample_size (f r n : ℕ) (freq_def : f = 36) (rate_def : r = 25 / 100) (relation : r = f / n) : n = 144 :=
sorry

end sample_size_l192_192560


namespace remaining_wire_length_l192_192996

theorem remaining_wire_length (total_wire_length : ℝ) (square_side_length : ℝ) 
  (h₀ : total_wire_length = 60) (h₁ : square_side_length = 9) : 
  total_wire_length - 4 * square_side_length = 24 :=
by
  sorry

end remaining_wire_length_l192_192996


namespace area_overlap_of_triangles_l192_192976

structure Point where
  x : ℝ
  y : ℝ

def Triangle (p1 p2 p3 : Point) : Set Point :=
  { q | ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ (a * p1.x + b * p2.x + c * p3.x = q.x) ∧ (a * p1.y + b * p2.y + c * p3.y = q.y) }

def area_of_overlap (t1 t2 : Set Point) : ℝ :=
  -- Assume we have a function that calculates the overlap area
  sorry

def point1 : Point := ⟨0, 2⟩
def point2 : Point := ⟨2, 1⟩
def point3 : Point := ⟨0, 0⟩
def point4 : Point := ⟨2, 2⟩
def point5 : Point := ⟨0, 1⟩
def point6 : Point := ⟨2, 0⟩

def triangle1 : Set Point := Triangle point1 point2 point3
def triangle2 : Set Point := Triangle point4 point5 point6

theorem area_overlap_of_triangles :
  area_of_overlap triangle1 triangle2 = 1 :=
by
  -- Proof goes here, replacing sorry with actual proof steps
  sorry

end area_overlap_of_triangles_l192_192976


namespace greatest_divisor_of_620_and_180_l192_192687

/-- This theorem asserts that the greatest divisor of 620 that 
    is smaller than 100 and also a factor of 180 is 20. -/
theorem greatest_divisor_of_620_and_180 (d : ℕ) (h1 : d ∣ 620) (h2 : d ∣ 180) (h3 : d < 100) : d ≤ 20 :=
by
  sorry

end greatest_divisor_of_620_and_180_l192_192687


namespace not_all_x_ne_1_imp_x2_ne_0_l192_192083

theorem not_all_x_ne_1_imp_x2_ne_0 : ¬ (∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) :=
sorry

end not_all_x_ne_1_imp_x2_ne_0_l192_192083


namespace total_votes_cast_l192_192999

theorem total_votes_cast (V : ℝ) (h1 : V > 0) (h2 : 0.35 * V = candidate_votes) (h3 : candidate_votes + 2400 = rival_votes) (h4 : candidate_votes + rival_votes = V) : V = 8000 := 
by
  sorry

end total_votes_cast_l192_192999


namespace remainder_when_divided_by_30_l192_192825

theorem remainder_when_divided_by_30 (n k R m : ℤ) (h1 : 0 ≤ R ∧ R < 30) (h2 : 2 * n % 15 = 2) (h3 : n = 30 * k + R) : R = 1 := by
  sorry

end remainder_when_divided_by_30_l192_192825


namespace fraction_problem_l192_192323

theorem fraction_problem (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := 
by
  sorry

end fraction_problem_l192_192323


namespace rhombus_difference_l192_192225

theorem rhombus_difference (n : ℕ) (h : n > 3)
    (m : ℕ := 3 * (n - 1) * n / 2)
    (d : ℕ := 3 * (n - 3) * (n - 2) / 2) :
    m - d = 6 * n - 9 := by {
  -- Proof omitted
  sorry
}

end rhombus_difference_l192_192225


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l192_192877

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4 :
  let n := 7^7 - 7^4 in 
  (∑ p in (nat.factors n).to_finset, p) = 31 :=
by sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_l192_192877


namespace three_kids_savings_l192_192502

theorem three_kids_savings :
  (200 / 100) + (100 / 20) + (330 / 10) = 40 :=
by
  -- Proof goes here
  sorry

end three_kids_savings_l192_192502


namespace marbles_left_in_the_box_l192_192092

-- Define the main problem parameters.
def total_marbles : ℕ := 50
def white_marbles : ℕ := 20
def blue_marbles : ℕ := (total_marbles - white_marbles) / 2
def red_marbles : ℕ := blue_marbles
def removed_marbles : ℕ := 2 * (white_marbles - blue_marbles)
def remaining_marbles : ℕ := total_marbles - removed_marbles

-- The theorem to prove the number of marbles left in the box.
theorem marbles_left_in_the_box : remaining_marbles = 40 := by
  unfold total_marbles white_marbles blue_marbles red_marbles removed_marbles remaining_marbles
  -- Here goes the calculus step simplification
  sorry

end marbles_left_in_the_box_l192_192092


namespace value_range_of_2_sin_x_minus_1_l192_192260

theorem value_range_of_2_sin_x_minus_1 :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) →
  (∀ y : ℝ, y = 2 * Real.sin y - 1 → -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end value_range_of_2_sin_x_minus_1_l192_192260


namespace lucky_numbers_count_l192_192098

def isLuckyNumber (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3 = 6) && (100 ≤ n) && (n < 1000)

def countLuckyNumbers : ℕ :=
  (List.range' 100 900).filter isLuckyNumber |>.length

theorem lucky_numbers_count : countLuckyNumbers = 21 := 
  sorry

end lucky_numbers_count_l192_192098


namespace minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l192_192020

noncomputable def f (x m : ℝ) : ℝ := x - m * Real.log x - (m - 1) / x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem minimum_value_f_1 (m : ℝ) : (m ≤ 2) → f 1 m = 2 - m := sorry

theorem minimum_value_f_e (m : ℝ) : (m ≥ Real.exp 1 + 1) → f (Real.exp 1) m = Real.exp 1 - m - (m - 1) / Real.exp 1 := sorry

theorem minimum_value_f_m_minus_1 (m : ℝ) : (2 < m ∧ m < Real.exp 1 + 1) → 
  f (m - 1) m = m - 2 - m * Real.log (m - 1) := sorry

theorem range_of_m (m : ℝ) : 
  (m ≤ 2) → 
  (∃ x1 ∈ Set.Icc (Real.exp 1) (Real.exp 1 ^ 2), ∀ x2 ∈ Set.Icc (-2 : ℝ) 0, f x1 m ≤ g x2) → 
  Real.exp 1 - m - (m - 1) / Real.exp 1 ≤ 1 → 
  (m ≥ (Real.exp 1 ^ 2 - Real.exp 1 + 1) / (Real.exp 1 + 1) ∧ m ≤ 2) := sorry

end minimum_value_f_1_minimum_value_f_e_minimum_value_f_m_minus_1_range_of_m_l192_192020


namespace closest_integer_to_cube_root_of_250_l192_192118

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l192_192118


namespace prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l192_192643

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_gt_3_div_24 (p : ℕ) (hp : is_prime p) (h : p > 3) : 
  24 ∣ (p^2 - 1) :=
sorry

theorem num_form_6n_plus_minus_1_div_24 (n : ℕ) : 
  24 ∣ (6 * n + 1)^2 - 1 ∧ 24 ∣ (6 * n - 1)^2 - 1 :=
sorry

end prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l192_192643


namespace sqrt_sum_simplify_l192_192959

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l192_192959


namespace closest_integer_to_cube_root_250_l192_192110

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l192_192110


namespace score_difference_l192_192885

theorem score_difference (chuck_score red_score : ℕ) (h1 : chuck_score = 95) (h2 : red_score = 76) : chuck_score - red_score = 19 := by
  sorry

end score_difference_l192_192885


namespace union_of_sets_l192_192031

def setA : Set ℝ := { x | -5 ≤ x ∧ x < 1 }
def setB : Set ℝ := { x | x ≤ 2 }

theorem union_of_sets : setA ∪ setB = { x | x ≤ 2 } :=
by sorry

end union_of_sets_l192_192031


namespace B_joined_amount_l192_192830

theorem B_joined_amount (T : ℝ)
  (A_investment : ℝ := 45000)
  (B_time : ℝ := 2)
  (profit_ratio : ℝ := 2 / 1)
  (investment_ratio_rule : (A_investment * T) / (B_investment_amount * B_time) = profit_ratio) :
  B_investment_amount = 22500 :=
by
  sorry

end B_joined_amount_l192_192830


namespace necessary_but_not_sufficient_l192_192659

-- Definitions extracted from the problem conditions
def isEllipse (k : ℝ) : Prop := (9 - k > 0) ∧ (k - 7 > 0) ∧ (9 - k ≠ k - 7)

-- The necessary but not sufficient condition for the ellipse equation
theorem necessary_but_not_sufficient : 
  (7 < k ∧ k < 9) → isEllipse k → (isEllipse k ↔ (7 < k ∧ k < 9)) := 
by 
  sorry

end necessary_but_not_sufficient_l192_192659


namespace distribution_centers_count_l192_192704

theorem distribution_centers_count (n : ℕ) (h : n = 5) : n + (n * (n - 1)) / 2 = 15 :=
by
  subst h -- replace n with 5
  show 5 + (5 * (5 - 1)) / 2 = 15
  have : (5 * 4) / 2 = 10 := by norm_num
  show 5 + 10 = 15
  norm_num

end distribution_centers_count_l192_192704


namespace lilith_caps_collection_l192_192776

noncomputable def monthlyCollectionYear1 := 3
noncomputable def monthlyCollectionAfterYear1 := 5
noncomputable def christmasCaps := 40
noncomputable def yearlyCapsLost := 15
noncomputable def totalYears := 5

noncomputable def totalCapsCollectedByLilith :=
  let firstYearCaps := monthlyCollectionYear1 * 12
  let remainingYearsCaps := monthlyCollectionAfterYear1 * 12 * (totalYears - 1)
  let christmasCapsTotal := christmasCaps * totalYears
  let totalCapsBeforeLosses := firstYearCaps + remainingYearsCaps + christmasCapsTotal
  let lostCapsTotal := yearlyCapsLost * totalYears
  let totalCapsAfterLosses := totalCapsBeforeLosses - lostCapsTotal
  totalCapsAfterLosses

theorem lilith_caps_collection : totalCapsCollectedByLilith = 401 := by
  sorry

end lilith_caps_collection_l192_192776


namespace TomTotalWeight_l192_192982

def TomWeight : ℝ := 150
def HandWeight (personWeight: ℝ) : ℝ := 1.5 * personWeight
def VestWeight (personWeight: ℝ) : ℝ := 0.5 * personWeight
def TotalHandWeight (handWeight: ℝ) : ℝ := 2 * handWeight
def TotalWeight (totalHandWeight vestWeight: ℝ) : ℝ := totalHandWeight + vestWeight

theorem TomTotalWeight : TotalWeight (TotalHandWeight (HandWeight TomWeight)) (VestWeight TomWeight) = 525 := 
by
  sorry

end TomTotalWeight_l192_192982


namespace work_rate_l192_192284

theorem work_rate (A_rate : ℝ) (combined_rate : ℝ) (B_days : ℝ) :
  A_rate = 1 / 12 ∧ combined_rate = 1 / 6.461538461538462 → 1 / B_days = combined_rate - A_rate → B_days = 14 :=
by
  intros
  sorry

end work_rate_l192_192284


namespace find_a_values_l192_192790

theorem find_a_values (a x₁ x₂ : ℝ) (h1 : x^2 + a * x - 2 = 0)
                      (h2 : x₁ ≠ x₂)
                      (h3 : x₁^3 + 22 / x₂ = x₂^3 + 22 / x₁) :
                      a = 3 ∨ a = -3 :=
by
  sorry

end find_a_values_l192_192790


namespace vova_last_grades_l192_192522

theorem vova_last_grades (grades : Fin 19 → ℕ) 
  (first_four_2s : ∀ i : Fin 4, grades i = 2)
  (all_combinations_once : ∀ comb : Fin 4 → ℕ, 
    (∃ (start : Fin (19-3)), ∀ j : Fin 4, grades (start + j) = comb j) ∧
    (∀ i j : Fin (19-3), 
      (∀ k : Fin 4, grades (i + k) = grades (j + k)) → i = j)) :
  ∀ i : Fin 4, grades (15 + i) = if i = 0 then 3 else 2 :=
by
  sorry

end vova_last_grades_l192_192522


namespace function_parity_l192_192164

noncomputable def f : ℝ → ℝ := sorry

-- Condition: f satisfies the functional equation for all x, y in Real numbers
axiom functional_eqn (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y

-- Prove that the function could be either odd or even.
theorem function_parity : (∀ x, f (-x) = f x) ∨ (∀ x, f (-x) = -f x) := 
sorry

end function_parity_l192_192164


namespace dave_books_about_outer_space_l192_192868

theorem dave_books_about_outer_space (x : ℕ) 
  (H1 : 8 + 3 = 11) 
  (H2 : 11 * 6 = 66) 
  (H3 : 102 - 66 = 36) 
  (H4 : 36 / 6 = x) : 
  x = 6 := 
by
  sorry

end dave_books_about_outer_space_l192_192868


namespace closest_integer_to_cubert_of_250_is_6_l192_192107

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l192_192107


namespace probability_page_multiple_of_7_l192_192975

theorem probability_page_multiple_of_7 (total_pages : ℕ) (probability : ℚ)
  (h_total_pages : total_pages = 500) 
  (h_probability : probability = 71 / 500) :
  probability = 0.142 := 
sorry

end probability_page_multiple_of_7_l192_192975


namespace down_payment_calculation_l192_192493

theorem down_payment_calculation 
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (n : ℕ)
  (interest_rate : ℝ)
  (down_payment : ℝ) :
  purchase_price = 127 ∧ 
  monthly_payment = 10 ∧ 
  n = 12 ∧ 
  interest_rate = 0.2126 ∧
  down_payment + (n * monthly_payment) = purchase_price * (1 + interest_rate) 
  → down_payment = 34 := 
sorry

end down_payment_calculation_l192_192493


namespace dennis_pants_purchase_l192_192576

theorem dennis_pants_purchase
  (pants_cost : ℝ) 
  (pants_discount : ℝ) 
  (socks_cost : ℝ) 
  (socks_discount : ℝ) 
  (socks_quantity : ℕ)
  (total_spent : ℝ)
  (discounted_pants_cost : ℝ)
  (discounted_socks_cost : ℝ)
  (pants_quantity : ℕ) :
  pants_cost = 110.00 →
  pants_discount = 0.30 →
  socks_cost = 60.00 →
  socks_discount = 0.30 →
  socks_quantity = 2 →
  total_spent = 392.00 →
  discounted_pants_cost = pants_cost * (1 - pants_discount) →
  discounted_socks_cost = socks_cost * (1 - socks_discount) →
  total_spent = socks_quantity * discounted_socks_cost + pants_quantity * discounted_ppants_cost →
  pants_quantity = 4 :=
by
  intros
  sorry

end dennis_pants_purchase_l192_192576


namespace stack_of_logs_total_l192_192848

-- Define the given conditions as variables and constants in Lean
def bottom_row : Nat := 15
def top_row : Nat := 4
def rows : Nat := bottom_row - top_row + 1
def sum_arithmetic_series (a l n : Nat) : Nat := n * (a + l) / 2

-- Define the main theorem to prove
theorem stack_of_logs_total : sum_arithmetic_series top_row bottom_row rows = 114 :=
by
  -- Here you will normally provide the proof
  sorry

end stack_of_logs_total_l192_192848


namespace perpendicular_and_intersection_l192_192730

variables (x y : ℚ)

def line1 := 4 * y - 3 * x = 15
def line4 := 3 * y + 4 * x = 15

theorem perpendicular_and_intersection :
  (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 15) →
  let m1 := (3 : ℚ) / 4
  let m4 := -(4 : ℚ) / 3
  m1 * m4 = -1 ∧
  ∃ x y : ℚ, 4*y - 3*x = 15 ∧ 3*y + 4*x = 15 ∧ x = 15/32 ∧ y = 35/8 :=
by
  sorry

end perpendicular_and_intersection_l192_192730


namespace hotel_ticket_ratio_l192_192635

theorem hotel_ticket_ratio (initial_amount : ℕ) (remaining_amount : ℕ) (ticket_cost : ℕ) (hotel_cost : ℕ) :
  initial_amount = 760 →
  remaining_amount = 310 →
  ticket_cost = 300 →
  initial_amount - remaining_amount - ticket_cost = hotel_cost →
  (hotel_cost : ℚ) / (ticket_cost : ℚ) = 1 / 2 :=
by
  intros h_initial h_remaining h_ticket h_hotel
  sorry

end hotel_ticket_ratio_l192_192635


namespace workshop_male_workers_l192_192144

variables (F M : ℕ)

theorem workshop_male_workers :
  (M = F + 45) ∧ (M - 5 = 3 * F) → M = 65 :=
by
  intros h
  sorry

end workshop_male_workers_l192_192144


namespace find_a_l192_192892

noncomputable def f (x : ℝ) : ℝ := x^2 + 12
noncomputable def g (x : ℝ) : ℝ := x^2 - x - 4

theorem find_a (a : ℝ) (h_pos : a > 0) (h_fga : f (g a) = 12) : a = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_a_l192_192892


namespace distance_between_cities_l192_192283

theorem distance_between_cities
    (v_bus : ℕ) (v_car : ℕ) (t_bus_meet : ℚ) (t_car_wait : ℚ)
    (d_overtake : ℚ) (s : ℚ)
    (h_vb : v_bus = 40)
    (h_vc : v_car = 50)
    (h_tbm : t_bus_meet = 0.25)
    (h_tcw : t_car_wait = 0.25)
    (h_do : d_overtake = 20)
    (h_eq : (s - 10) / 50 + t_car_wait = (s - 30) / 40) :
    s = 160 :=
by
    exact sorry

end distance_between_cities_l192_192283


namespace cone_height_circular_sector_l192_192285

theorem cone_height_circular_sector (r : ℝ) (n : ℕ) (h : ℝ)
  (hr : r = 10)
  (hn : n = 3)
  (hradius : r > 0)
  (hcircumference : 2 * Real.pi * r / n = 2 * Real.pi * r / 3)
  : h = (20 * Real.sqrt 2) / 3 :=
by {
  sorry
}

end cone_height_circular_sector_l192_192285


namespace decreasing_f_range_l192_192019

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x - 2 * k * x - 1

theorem decreasing_f_range (k : ℝ) (x₁ x₂ : ℝ) (h₁ : 2 ≤ x₁) (h₂ : x₁ < x₂) (h₃ : x₂ ≤ 4) :
  k ≥ 1 / 4 → (x₁ - x₂) * (f x₁ k - f x₂ k) < 0 :=
sorry

end decreasing_f_range_l192_192019


namespace find_value_of_expression_l192_192891

noncomputable def roots_g : Set ℂ := { x | x^2 - 3*x - 2 = 0 }

theorem find_value_of_expression:
  ∀ γ δ : ℂ, γ ∈ roots_g → δ ∈ roots_g →
  (γ + δ = 3) → (7 * γ^4 + 10 * δ^3 = 1363) :=
by
  intros γ δ hγ hδ hsum
  -- Proof skipped
  sorry

end find_value_of_expression_l192_192891


namespace min_value_of_f_in_D_l192_192875

noncomputable def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x * y + y^2) - 3 * (x + y) + 5

def D (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem min_value_of_f_in_D : ∃ (x y : ℝ), D x y ∧ f x y = 2 ∧ (∀ (u v : ℝ), D u v → f u v ≥ 2) :=
by
  sorry

end min_value_of_f_in_D_l192_192875


namespace baker_sold_more_pastries_l192_192724

theorem baker_sold_more_pastries {cakes_made pastries_made pastries_sold cakes_sold : ℕ}
    (h1 : cakes_made = 105)
    (h2 : pastries_made = 275)
    (h3 : pastries_sold = 214)
    (h4 : cakes_sold = 163) :
    pastries_sold - cakes_sold = 51 := by
  sorry

end baker_sold_more_pastries_l192_192724


namespace range_of_m_l192_192749

theorem range_of_m {m : ℝ} (h1 : m^2 - 1 < 0) (h2 : m > 0) : 0 < m ∧ m < 1 :=
sorry

end range_of_m_l192_192749


namespace rectangle_width_l192_192261

theorem rectangle_width
  (L W : ℝ)
  (h1 : W = L + 2)
  (h2 : 2 * L + 2 * W = 16) :
  W = 5 :=
by
  sorry

end rectangle_width_l192_192261


namespace total_games_in_season_l192_192809

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end total_games_in_season_l192_192809


namespace suff_but_not_necess_condition_l192_192013

theorem suff_but_not_necess_condition (a b : ℝ) (h1 : a < 0) (h2 : -1 < b ∧ b < 0) : a + a * b < 0 :=
  sorry

end suff_but_not_necess_condition_l192_192013


namespace arithmetic_sequence_problem_l192_192210

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n, a n = a1 + (n - 1) * d

-- Given condition
def given_condition (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
3 * a 9 - a 15 - a 3 = 20

-- Question to prove
def question (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
2 * a 8 - a 7 = 20

-- Main theorem
theorem arithmetic_sequence_problem (a: ℕ → ℝ) (a1 d: ℝ):
  arithmetic_sequence a a1 d →
  given_condition a a1 d →
  question a a1 d :=
by
  sorry

end arithmetic_sequence_problem_l192_192210


namespace steve_speed_during_race_l192_192351

theorem steve_speed_during_race 
  (distance_gap : ℝ) 
  (john_speed : ℝ) 
  (time : ℝ) 
  (john_ahead : ℝ)
  (steve_speed : ℝ) :
  distance_gap = 16 →
  john_speed = 4.2 →
  time = 36 →
  john_ahead = 2 →
  steve_speed = (151.2 - 18) / 36 :=
by
  sorry

end steve_speed_during_race_l192_192351


namespace compacted_space_of_all_cans_l192_192490

def compacted_space_per_can (original_space: ℕ) (compaction_rate: ℕ) : ℕ :=
  original_space * compaction_rate / 100

def total_compacted_space (num_cans: ℕ) (compacted_space: ℕ) : ℕ :=
  num_cans * compacted_space

theorem compacted_space_of_all_cans :
  ∀ (num_cans original_space compaction_rate : ℕ),
  num_cans = 100 →
  original_space = 30 →
  compaction_rate = 35 →
  total_compacted_space num_cans (compacted_space_per_can original_space compaction_rate) = 1050 :=
by
  intros num_cans original_space compaction_rate h1 h2 h3
  rw [h1, h2, h3]
  dsimp [compacted_space_per_can, total_compacted_space]
  norm_num
  sorry

end compacted_space_of_all_cans_l192_192490


namespace intervals_of_monotonicity_of_f_l192_192480

noncomputable def f (a b c d : ℝ) (x : ℝ) := a * x^3 + b * x^2 + c * x + d

theorem intervals_of_monotonicity_of_f (a b c d : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P.1 = 0 ∧ d = P.2 ∧ (12 * P.1 - P.2 - 4 = 0))
  (h2 : ∃ x : ℝ, x = 2 ∧ (f a b c d x = 0) ∧ (∃ x : ℝ, x = 0 ∧ (3 * a * x^2 + 2 * b * x + c = 12))) 
  : ( ∃ a b c d : ℝ , (f a b c d) = (2 * x^3 - 9 * x^2 + 12 * x -4)) := 
  sorry

end intervals_of_monotonicity_of_f_l192_192480


namespace closest_cube_root_of_250_l192_192102

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l192_192102


namespace number_of_logs_in_stack_l192_192846

theorem number_of_logs_in_stack :
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  total_logs = 114 := by
{
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  sorry
}

end number_of_logs_in_stack_l192_192846


namespace average_goals_is_92_l192_192465

-- Definitions based on conditions
def layla_goals : ℕ := 104
def kristin_fewer_goals : ℕ := 24
def kristin_goals : ℕ := layla_goals - kristin_fewer_goals
def combined_goals : ℕ := layla_goals + kristin_goals
def average_goals : ℕ := combined_goals / 2

-- Theorem
theorem average_goals_is_92 : average_goals = 92 := 
  sorry

end average_goals_is_92_l192_192465


namespace non_neg_integer_solutions_l192_192738

theorem non_neg_integer_solutions (a b c : ℕ) :
  (∀ x : ℕ, x^2 - 2 * a * x + b = 0 → x ≥ 0) ∧ 
  (∀ y : ℕ, y^2 - 2 * b * y + c = 0 → y ≥ 0) ∧ 
  (∀ z : ℕ, z^2 - 2 * c * z + a = 0 → z ≥ 0) → 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end non_neg_integer_solutions_l192_192738


namespace total_games_in_season_l192_192810

-- Define the constants according to the conditions
def number_of_teams : ℕ := 25
def games_per_pair : ℕ := 15

-- Define the mathematical statement we want to prove
theorem total_games_in_season :
  let round_robin_games := (number_of_teams * (number_of_teams - 1)) / 2 in
  let total_games := round_robin_games * games_per_pair in
  total_games = 4500 :=
by
  sorry

end total_games_in_season_l192_192810


namespace max_sum_of_prices_l192_192513

theorem max_sum_of_prices (R P : ℝ) 
  (h1 : 4 * R + 5 * P ≥ 27) 
  (h2 : 6 * R + 3 * P ≤ 27) : 
  3 * R + 4 * P ≤ 36 :=
by 
  sorry

end max_sum_of_prices_l192_192513


namespace sum_remainders_l192_192680

theorem sum_remainders (a b c : ℕ) (h₁ : a % 30 = 7) (h₂ : b % 30 = 11) (h₃ : c % 30 = 23) : 
  (a + b + c) % 30 = 11 := 
by
  sorry

end sum_remainders_l192_192680


namespace composite_sum_of_squares_l192_192796

theorem composite_sum_of_squares (a b : ℤ) (h_roots : ∃ x1 x2 : ℕ, (x1 + x2 : ℤ) = -a ∧ (x1 * x2 : ℤ) = b + 1) :
  ∃ m n : ℕ, a^2 + b^2 = m * n ∧ 1 < m ∧ 1 < n :=
sorry

end composite_sum_of_squares_l192_192796


namespace hotel_bill_amount_l192_192698

-- Definition of the variables used in the conditions
def each_paid : ℝ := 124.11
def friends : ℕ := 9

-- The Lean 4 theorem statement
theorem hotel_bill_amount :
  friends * each_paid = 1116.99 := sorry

end hotel_bill_amount_l192_192698


namespace find_some_number_l192_192253

noncomputable def some_number : ℝ := 1000
def expr_approx (a b c d : ℝ) := (a * b) / c = d

theorem find_some_number :
  expr_approx 3.241 14 some_number 0.045374000000000005 :=
by sorry

end find_some_number_l192_192253


namespace dave_age_l192_192995

theorem dave_age (C D E : ℝ) (h1 : C = 4 * D) (h2 : E = D + 5) (h3 : C = E) : D = 5 / 3 :=
by
  sorry

end dave_age_l192_192995


namespace avg_goals_l192_192462

-- Let's declare the variables and conditions
def layla_goals : ℕ := 104
def games_played : ℕ := 4
def less_goals_kristin : ℕ := 24

-- Define the number of goals Kristin scored
def kristin_goals : ℕ := layla_goals - less_goals_kristin

-- Calculate the total number of goals scored by both
def total_goals : ℕ := layla_goals + kristin_goals

-- Calculate the average number of goals per game
def average_goals_per_game : ℕ := total_goals / games_played

-- The theorem statement
theorem avg_goals : average_goals_per_game = 46 := by
  -- proof skipped, assume correct by using sorry
  sorry

end avg_goals_l192_192462


namespace find_m_and_max_value_l192_192751

theorem find_m_and_max_value (m : ℝ) 
  (f : ℝ → ℝ := λ x, -x^3 + 3 * x^2 + 9 * x + m) 
  (h_max : ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 20) 
  (h_at_2 : f 2 = 20) : 
  m = -2 ∧ f 2 = 20 := 
by 
  sorry

end find_m_and_max_value_l192_192751


namespace fraction_irreducibility_l192_192741

theorem fraction_irreducibility (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducibility_l192_192741


namespace problem1_problem2_l192_192187

section
variable {x a : ℝ}

-- Definitions of the functions
def f (x : ℝ) : ℝ := |x + 1|
def g (x : ℝ) (a : ℝ) : ℝ := 2 * |x| + a

-- Problem 1
theorem problem1 (a : ℝ) (H : a = -1) : 
  ∀ x : ℝ, f x ≤ g x a ↔ (x ≤ -2/3 ∨ 2 ≤ x) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) : 
  (∃ x₀ : ℝ, f x₀ ≥ 1/2 * g x₀ a) → a ≤ 2 :=
sorry

end

end problem1_problem2_l192_192187


namespace sum_of_transformed_numbers_l192_192383

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_transformed_numbers_l192_192383


namespace polynomial_third_and_fourth_equal_l192_192081

theorem polynomial_third_and_fourth_equal (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1)
  (h_eq : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = (8 : ℝ) / 11 :=
by
  sorry

end polynomial_third_and_fourth_equal_l192_192081


namespace problem_l192_192677

theorem problem 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 996 / 1005 :=
sorry

end problem_l192_192677


namespace x_eq_sum_of_squares_of_two_consecutive_integers_l192_192978

noncomputable def x_seq (n : ℕ) : ℝ :=
  1 / 4 * ((2 + Real.sqrt 3) ^ (2 * n - 1) + (2 - Real.sqrt 3) ^ (2 * n - 1))

theorem x_eq_sum_of_squares_of_two_consecutive_integers (n : ℕ) : 
  ∃ y : ℤ, x_seq n = (y:ℝ)^2 + (y + 1)^2 :=
sorry

end x_eq_sum_of_squares_of_two_consecutive_integers_l192_192978


namespace prime_or_four_no_square_div_factorial_l192_192584

theorem prime_or_four_no_square_div_factorial (n : ℕ) :
  (n * n ∣ n!) = false ↔ Nat.Prime n ∨ n = 4 := by
  sorry

end prime_or_four_no_square_div_factorial_l192_192584


namespace count_valid_n_l192_192195

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, 300 < n^2 ∧ n^2 < 1200 ∧ n % 3 = 0) ∧
                     S.card = 6 := sorry

end count_valid_n_l192_192195


namespace route_comparison_l192_192484

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end route_comparison_l192_192484


namespace correct_answer_l192_192683

theorem correct_answer (A B C D : String) (sentence : String)
  (h1 : A = "us")
  (h2 : B = "we")
  (h3 : C = "our")
  (h4 : D = "ours")
  (h_sentence : sentence = "To save class time, our teacher has _ students do half of the exercise in class and complete the other half for homework.") :
  sentence = "To save class time, our teacher has " ++ A ++ " students do half of the exercise in class and complete the other half for homework." :=
by
  sorry

end correct_answer_l192_192683


namespace transylvanian_sanity_l192_192069

theorem transylvanian_sanity (sane : Prop) (belief : Prop) (h1 : sane) (h2 : sane → belief) : belief :=
by
  sorry

end transylvanian_sanity_l192_192069


namespace cost_price_of_computer_table_l192_192379

/-- The cost price \(C\) of a computer table is Rs. 7000 -/
theorem cost_price_of_computer_table : 
  ∃ (C : ℝ), (S = 1.20 * C) ∧ (S = 8400) → C = 7000 := 
by 
  sorry

end cost_price_of_computer_table_l192_192379


namespace intersection_x_coord_of_lines_l192_192375

theorem intersection_x_coord_of_lines (k b : ℝ) (h : k ≠ b) :
  ∃ x : ℝ, (kx + b = bx + k) ∧ x = 1 :=
by
  -- Proof is omitted.
  sorry

end intersection_x_coord_of_lines_l192_192375


namespace carla_drive_distance_l192_192157

theorem carla_drive_distance
    (d1 d3 : ℕ) (gpm : ℕ) (gas_price total_cost : ℕ) 
    (x : ℕ)
    (hx : 2 * gas_price = 1)
    (gallon_cost : ℕ := total_cost / gas_price)
    (total_distance   : ℕ := gallon_cost * gpm)
    (total_errand_distance : ℕ := d1 + x + d3 + 2 * x)
    (h_distance : total_distance = total_errand_distance) :
  x = 10 :=
by
  -- begin
  -- proof construction
  sorry

end carla_drive_distance_l192_192157


namespace find_x_l192_192620

def operation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) :
  operation 6 (operation 4 x) = 480 ↔ x = 5 := 
by
  sorry

end find_x_l192_192620


namespace exists_k_for_binary_operation_l192_192009

noncomputable def binary_operation (a b : ℤ) : ℤ := sorry

theorem exists_k_for_binary_operation :
  (∀ (a b c : ℤ), binary_operation a (b + c) = 
      binary_operation b a + binary_operation c a) →
  ∃ (k : ℤ), ∀ (a b : ℤ), binary_operation a b = k * a * b :=
by
  sorry

end exists_k_for_binary_operation_l192_192009


namespace triangle_ABC_is_right_triangle_l192_192214

theorem triangle_ABC_is_right_triangle (A B C : ℝ) (hA : A = 68) (hB : B = 22) :
  A + B + C = 180 → C = 90 :=
by
  intro hABC
  sorry

end triangle_ABC_is_right_triangle_l192_192214


namespace solve_for_q_l192_192241

theorem solve_for_q (n m q: ℚ)
  (h1 : 3 / 4 = n / 88)
  (h2 : 3 / 4 = (m + n) / 100)
  (h3 : 3 / 4 = (q - m) / 150) :
  q = 121.5 :=
sorry

end solve_for_q_l192_192241


namespace f_eq_l192_192902

noncomputable def a (n : ℕ) : ℚ := 1 / ((n + 1) ^ 2)

noncomputable def f : ℕ → ℚ
| 0     => 1
| (n+1) => f n * (1 - a (n+1))

theorem f_eq : ∀ n : ℕ, f n = (n + 2) / (2 * (n + 1)) :=
by
  sorry

end f_eq_l192_192902


namespace correctStatements_l192_192398

-- Definitions based on conditions
def isFunctionalRelationshipDeterministic (S1 : Prop) := 
  S1 = true

def isCorrelationNonDeterministic (S2 : Prop) := 
  S2 = true

def regressionAnalysisFunctionalRelation (S3 : Prop) :=
  S3 = false

def regressionAnalysisCorrelation (S4 : Prop) :=
  S4 = true

-- The translated proof problem statement
theorem correctStatements :
  ∀ (S1 S2 S3 S4 : Prop), 
    isFunctionalRelationshipDeterministic S1 →
    isCorrelationNonDeterministic S2 →
    regressionAnalysisFunctionalRelation S3 →
    regressionAnalysisCorrelation S4 →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) = (true ∧ true ∧ true ∧ true) :=
by
  intros S1 S2 S3 S4 H1 H2 H3 H4 H5
  sorry

end correctStatements_l192_192398


namespace max_ab_eq_one_quarter_l192_192908

theorem max_ab_eq_one_quarter (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_eq_one_quarter_l192_192908


namespace xy_square_sum_l192_192761

variable (x y : ℝ)

theorem xy_square_sum : (y + 6 = (x - 3)^2) →
                        (x + 6 = (y - 3)^2) →
                        (x ≠ y) →
                        x^2 + y^2 = 43 :=
by
  intros h₁ h₂ h₃
  sorry

end xy_square_sum_l192_192761


namespace polynomial_identity_l192_192747

theorem polynomial_identity (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 15) 
  (h3 : a^3 + b^3 + c^3 = 47) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) = 625 := 
by 
  sorry

end polynomial_identity_l192_192747


namespace find_smaller_number_l192_192514

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 3) : y = 28.5 :=
by
  sorry

end find_smaller_number_l192_192514


namespace simplify_expr1_simplify_expr2_l192_192649

variable {a b : ℝ}

theorem simplify_expr1 : 3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1 :=
by
  sorry

theorem simplify_expr2 : 2 * (5 * a - 3 * b) - 3 * (a ^ 2 - 2 * b) = 10 * a - 3 * a ^ 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l192_192649


namespace odd_tiling_numbers_l192_192593

def f (n k : ℕ) : ℕ := sorry -- Assume f(n, 2k) is defined appropriately.

theorem odd_tiling_numbers (n : ℕ) : (∀ k : ℕ, f n (2*k) % 2 = 1) ↔ ∃ i : ℕ, n = 2^i - 1 := sorry

end odd_tiling_numbers_l192_192593


namespace rolls_to_neighbor_l192_192434

theorem rolls_to_neighbor (total_needed rolls_to_grandmother rolls_to_uncle rolls_needed : ℕ) (h1 : total_needed = 45) (h2 : rolls_to_grandmother = 1) (h3 : rolls_to_uncle = 10) (h4 : rolls_needed = 28) :
  total_needed - rolls_needed - (rolls_to_grandmother + rolls_to_uncle) = 6 := by
  sorry

end rolls_to_neighbor_l192_192434


namespace range_of_a_l192_192592

theorem range_of_a 
  (a : ℝ) (h : ∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) 
  : -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l192_192592


namespace total_cost_of_phone_l192_192833

theorem total_cost_of_phone (cost_per_phone : ℕ) (monthly_cost : ℕ) (months : ℕ) (phone_count : ℕ) :
  cost_per_phone = 2 → monthly_cost = 7 → months = 4 → phone_count = 1 →
  (cost_per_phone * phone_count + monthly_cost * months) = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end total_cost_of_phone_l192_192833


namespace arithmetic_sqrt_25_l192_192656

-- Define the arithmetic square root condition
def is_arithmetic_sqrt (x a : ℝ) : Prop :=
  0 ≤ x ∧ x^2 = a

-- Lean statement to prove the arithmetic square root of 25 is 5
theorem arithmetic_sqrt_25 : is_arithmetic_sqrt 5 25 :=
by 
  sorry

end arithmetic_sqrt_25_l192_192656


namespace red_ball_expectation_variance_l192_192829

noncomputable def red_ball_problem (n : ℕ) (p : ℝ) := sorry

theorem red_ball_expectation_variance :
  let ξ_1 := binomial 2 (1/3)
  let ξ_2 := { 0 := 1/3, 1 := 2/3 }
  E ξ_1 = E ξ_2 ∧ D ξ_1 > D ξ_2 := by
  sorry

end red_ball_expectation_variance_l192_192829


namespace find_number_l192_192555

-- Statement of the problem in Lean 4
theorem find_number (n : ℝ) (h : n / 3000 = 0.008416666666666666) : n = 25.25 :=
sorry

end find_number_l192_192555


namespace hockey_league_games_l192_192812

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end hockey_league_games_l192_192812


namespace math_problem_l192_192909

theorem math_problem (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end math_problem_l192_192909


namespace bryce_raisins_l192_192568

theorem bryce_raisins:
  ∃ x : ℕ, (x - 8 = x / 3) ∧ x = 12 :=
by 
  sorry

end bryce_raisins_l192_192568


namespace range_of_a_l192_192595

variable (a x y : ℝ)

theorem range_of_a (h1 : 2 * x + y = 1 + 4 * a) (h2 : x + 2 * y = 2 - a) (h3 : x + y > 0) : a > -1 :=
sorry

end range_of_a_l192_192595


namespace find_interior_angles_l192_192215

theorem find_interior_angles (A B C : ℝ) (h1 : B = A + 10) (h2 : C = B + 10) (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 60 ∧ C = 70 := by
  sorry

end find_interior_angles_l192_192215


namespace op_value_l192_192575

noncomputable def op (a b c : ℝ) (k : ℤ) : ℝ :=
  b^2 - k * a^2 * c

theorem op_value : op 2 5 3 3 = -11 := by
  sorry

end op_value_l192_192575


namespace integer_average_problem_l192_192676

theorem integer_average_problem (a b c d : ℤ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
(h_max : max a (max b (max c d)) = 90) (h_min : min a (min b (min c d)) = 29) : 
(a + b + c + d) / 4 = 45 := 
sorry

end integer_average_problem_l192_192676


namespace find_m_when_z_is_real_l192_192101

theorem find_m_when_z_is_real (m : ℝ) (h : (m ^ 2 + 2 * m - 15 = 0)) : m = 3 :=
sorry

end find_m_when_z_is_real_l192_192101


namespace double_inequality_l192_192063

variable (a b c : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem double_inequality (h : triangle_sides a b c) : 
  3 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + b * c + c * a) :=
by
  sorry

end double_inequality_l192_192063


namespace question_solution_l192_192228

noncomputable def segment_ratio : (ℝ × ℝ) :=
  let m := 7
  let n := 2
  let x := - (2 / (m - n))
  let y := 7 / (m - n)
  (x, y)

theorem question_solution : segment_ratio = (-2/5, 7/5) :=
  by
  -- prove that the pair (x, y) calculated using given m and n equals (-2/5, 7/5)
  sorry

end question_solution_l192_192228


namespace line_through_point_parallel_l192_192663

theorem line_through_point_parallel (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) (hA : A = (2, 3)) (hl : ∀ x y, l x y ↔ 2 * x - 4 * y + 7 = 0) :
  ∃ m, (∀ x y, (2 * x - 4 * y + m = 0) ↔ (x - 2 * y + 4 = 0)) ∧ (2 * (A.1) - 4 * (A.2) + m = 0) := 
sorry

end line_through_point_parallel_l192_192663


namespace find_a17_a18_a19_a20_l192_192919

variable {α : Type*} [Field α]

-- Definitions based on the given conditions:
def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a n = a 0 * r ^ n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (Finset.range n).sum a

-- Problem statement based on the question and conditions:
theorem find_a17_a18_a19_a20 (a S : ℕ → α) (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S) (hS4 : S 4 = 1) (hS8 : S 8 = 3) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end find_a17_a18_a19_a20_l192_192919


namespace sum_of_geometric_ratios_l192_192356

theorem sum_of_geometric_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ)
  (hk : k ≠ 0) (hp : p ≠ r)
  (ha2 : a2 = k * p) (ha3 : a3 = k * p * p)
  (hb2 : b2 = k * r) (hb3 : b3 = k * r * r)
  (h : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 :=
by sorry

end sum_of_geometric_ratios_l192_192356


namespace find_triangle_sides_l192_192326

theorem find_triangle_sides (k : ℕ) (k_pos : k = 6) 
  {x y z : ℝ} (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) 
  (h : k * (x * y + y * z + z * x) > 5 * (x ^ 2 + y ^ 2 + z ^ 2)) :
  ∃ x' y' z', (x = x') ∧ (y = y') ∧ (z = z') ∧ ((x' + y' > z') ∧ (x' + z' > y') ∧ (y' + z' > x')) :=
by
  sorry

end find_triangle_sides_l192_192326


namespace customers_left_tip_l192_192716

-- Definition of the given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def customers_didnt_tip : ℕ := 34

-- Lean 4 statement proving that the number of customers who did leave a tip (answer) equals 15
theorem customers_left_tip : (initial_customers + added_customers - customers_didnt_tip) = 15 :=
by
  sorry

end customers_left_tip_l192_192716


namespace man_year_of_birth_l192_192409

theorem man_year_of_birth (x : ℕ) (hx1 : (x^2 + x >= 1850)) (hx2 : (x^2 + x < 1900)) : (1850 + (x^2 + x - x)) = 1892 :=
by {
  sorry
}

end man_year_of_birth_l192_192409


namespace possible_values_for_abc_l192_192182

theorem possible_values_for_abc (a b c : ℝ)
  (h : ∀ x y z : ℤ, (a * x + b * y + c * z) ∣ (b * x + c * y + a * z)) :
  (a, b, c) = (1, 0, 0) ∨ (a, b, c) = (0, 1, 0) ∨ (a, b, c) = (0, 0, 1) ∨
  (a, b, c) = (-1, 0, 0) ∨ (a, b, c) = (0, -1, 0) ∨ (a, b, c) = (0, 0, -1) :=
sorry

end possible_values_for_abc_l192_192182


namespace simplify_sum_of_square_roots_l192_192957

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l192_192957


namespace sqrt_sum_simplify_l192_192958

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l192_192958


namespace percent_decrease_area_pentagon_l192_192042

open Real

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * sqrt 3 / 2) * s ^ 2

noncomputable def area_pentagon (s : ℝ) : ℝ :=
  (sqrt (5 * (5 + 2 * sqrt 5)) / 4) * s ^ 2

noncomputable def diagonal_pentagon (s : ℝ) : ℝ :=
  (1 + sqrt 5) / 2 * s

theorem percent_decrease_area_pentagon :
  let s_p := sqrt (400 / sqrt (5 * (5 + 2 * sqrt 5)))
  let d := diagonal_pentagon s_p
  let new_d := 0.9 * d
  let new_s := new_d / ((1 + sqrt 5) / 2)
  let new_area := area_pentagon new_s
  (100 - new_area) / 100 * 100 = 20 :=
by
  sorry

end percent_decrease_area_pentagon_l192_192042


namespace age_of_student_who_left_l192_192072

variables
  (avg_age_students : ℝ)
  (num_students_before : ℕ)
  (num_students_after : ℕ)
  (age_teacher : ℝ)
  (new_avg_age_class : ℝ)

theorem age_of_student_who_left
  (h1 : avg_age_students = 14)
  (h2 : num_students_before = 45)
  (h3 : num_students_after = 44)
  (h4 : age_teacher = 45)
  (h5 : new_avg_age_class = 14.66)
: ∃ (age_student_left : ℝ), abs (age_student_left - 15.3) < 0.1 :=
sorry

end age_of_student_who_left_l192_192072


namespace trajectory_midpoint_l192_192989

theorem trajectory_midpoint (P Q M : ℝ × ℝ)
  (hP : P.1^2 + P.2^2 = 1)
  (hQ : Q.1 = 3 ∧ Q.2 = 0)
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end trajectory_midpoint_l192_192989


namespace range_of_g_l192_192052

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by
  sorry

end range_of_g_l192_192052


namespace largest_perfect_square_factor_of_882_l192_192688

theorem largest_perfect_square_factor_of_882 : ∃ n, n * n = 441 ∧ ∀ m, m * m ∣ 882 → m * m ≤ 441 := 
by 
 sorry

end largest_perfect_square_factor_of_882_l192_192688


namespace power_function_const_coeff_l192_192896

theorem power_function_const_coeff (m : ℝ) (h1 : m^2 + 2 * m - 2 = 1) (h2 : m ≠ 1) : m = -3 :=
  sorry

end power_function_const_coeff_l192_192896


namespace francis_violin_count_l192_192437

theorem francis_violin_count :
  let ukuleles := 2
  let guitars := 4
  let ukulele_strings := 4
  let guitar_strings := 6
  let violin_strings := 4
  let total_strings := 40
  ∃ (violins: ℕ), violins = 2 := by
    sorry

end francis_violin_count_l192_192437


namespace problem_statement_l192_192221

/-- Define the sequence of numbers spoken by Jo and Blair. -/
def next_number (n : ℕ) : ℕ :=
if n % 2 = 1 then (n + 1) / 2 else n / 2

/-- Helper function to compute the 21st number said. -/
noncomputable def twenty_first_number : ℕ :=
(21 + 1) / 2

/-- Statement of the problem in Lean 4. -/
theorem problem_statement : twenty_first_number = 11 := by
  sorry

end problem_statement_l192_192221


namespace derivative_at_2_l192_192538

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l192_192538


namespace evaluate_fraction_sqrt_l192_192872

theorem evaluate_fraction_sqrt :
  (Real.sqrt ((1 / 8) + (1 / 18)) = (Real.sqrt 26) / 12) :=
by
  sorry

end evaluate_fraction_sqrt_l192_192872


namespace part1_part2_l192_192899

noncomputable def f (x k : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2 + 2

theorem part1 {x : ℝ} (hx : x = 0) : 
    f x 0 = 1 :=
by
  sorry

theorem part2 {x k : ℝ} (hx : 0 ≤ x) (hxf : f x k ≥ 1) : 
    k ≤ 1 / 2 :=
by
  sorry

end part1_part2_l192_192899


namespace geometric_sequence_common_ratio_l192_192281

theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℝ) (q : ℝ),
  (∀ n, a (n + 1) = a n * q) →
  (∀ n m, n < m → a n < a m) →
  a 2 = 2 →
  a 4 - a 3 = 4 →
  q = 2 :=
by
  intros a q h_geo h_inc h_a2 h_a4_a3
  sorry

end geometric_sequence_common_ratio_l192_192281


namespace complex_number_in_fourth_quadrant_l192_192073

variable {a b : ℝ}

theorem complex_number_in_fourth_quadrant (a b : ℝ): 
  (a^2 + 1 > 0) ∧ (-b^2 - 1 < 0) → 
  ((a^2 + 1, -b^2 - 1).fst > 0 ∧ (a^2 + 1, -b^2 - 1).snd < 0) :=
by
  intro h
  exact h

#check complex_number_in_fourth_quadrant

end complex_number_in_fourth_quadrant_l192_192073


namespace business_total_profit_l192_192300

def total_profit (investmentB periodB profitB : ℝ) (investmentA periodA profitA : ℝ) (investmentC periodC profitC : ℝ) : ℝ :=
    (investmentA * periodA * profitA) + (investmentB * periodB * profitB) + (investmentC * periodC * profitC)

theorem business_total_profit 
    (investmentB periodB profitB : ℝ)
    (investmentA periodA profitA : ℝ)
    (investmentC periodC profitC : ℝ)
    (hA_inv : investmentA = 3 * investmentB)
    (hA_period : periodA = 2 * periodB)
    (hC_inv : investmentC = 2 * investmentB)
    (hC_period : periodC = periodB / 2)
    (hA_rate : profitA = 0.10)
    (hB_rate : profitB = 0.15)
    (hC_rate : profitC = 0.12)
    (hB_profit : investmentB * periodB * profitB = 4000) :
    total_profit investmentB periodB profitB investmentA periodA profitA investmentC periodC profitC = 23200 := 
sorry

end business_total_profit_l192_192300


namespace cow_cost_calculation_l192_192838

theorem cow_cost_calculation (C cow calf : ℝ) 
  (h1 : cow = 8 * calf) 
  (h2 : cow + calf = 990) : 
  cow = 880 :=
by
  sorry

end cow_cost_calculation_l192_192838


namespace sufficient_but_not_necessary_condition_l192_192280

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 3) → (x ≥ 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l192_192280


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l192_192878

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let expr := 7 ^ 7 - 7 ^ 4 in
  (7^4 * (7^3 - 1) = expr) ∧ (7^3 - 1 = 342) ∧ (Prime 2) ∧ (Prime 3) ∧ (Prime 7) ∧ (Prime 19) ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ 342 → p = 2 ∨ p = 3 ∨ p = 19) → 
  (∀ p : ℕ, Nat.Prime p → p ∣ expr → p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) → 
  (2 + 3 + 7 + 19 = 31) := 
by
  intro expr fact1 fact2 prime2 prime3 prime7 prime19 factors342 factorsExpr
  sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l192_192878


namespace largest_possible_s_l192_192929

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) 
  (h3 : ((r - 2) * 180 : ℚ) / r = (29 / 28) * ((s - 2) * 180 / s)) :
    s = 114 := by sorry

end largest_possible_s_l192_192929


namespace number_of_digits_in_expression_l192_192306

theorem number_of_digits_in_expression : 
  (Nat.digits 10 (2^12 * 5^8)).length = 10 := 
by
  sorry

end number_of_digits_in_expression_l192_192306


namespace sum_of_roots_eq_six_l192_192199

variable (a b : ℝ)

theorem sum_of_roots_eq_six (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_roots_eq_six_l192_192199


namespace simplify_sum_of_square_roots_l192_192955

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l192_192955


namespace probability_of_picking_combination_is_0_4_l192_192857

noncomputable def probability_at_least_19_rubles (total_coins total_value: ℕ) :=
  let coins := [10, 10, 5, 5, 2] in
  let all_combinations := (Finset.powersetLen 3 (coins.to_finset)).to_list in
  let favorable_combinations := all_combinations.filter (fun c => c.sum ≥ total_value) in
  (favorable_combinations.length : ℚ) / (all_combinations.length : ℚ)

theorem probability_of_picking_combination_is_0_4 :
  probability_at_least_19_rubles 5 19 = 0.4 :=
by
  sorry

end probability_of_picking_combination_is_0_4_l192_192857


namespace min_max_SX_SY_l192_192926

theorem min_max_SX_SY (n : ℕ) (hn : 2 ≤ n) (a : Finset ℕ) 
  (ha_sum : Finset.sum a id = 2 * n - 1) :
  ∃ (min_val max_val : ℕ), 
    (min_val = 2 * n - 2) ∧ 
    (max_val = n * (n - 1)) :=
sorry

end min_max_SX_SY_l192_192926


namespace statement_a_correct_statement_b_correct_l192_192212

open Real

theorem statement_a_correct (a b c : ℝ) (ha : a > b) (hc : c < 0) : a + c > b + c := by
  sorry

theorem statement_b_correct (a b : ℝ) (ha : a > b) (hb : b > 0) : (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end statement_a_correct_statement_b_correct_l192_192212


namespace total_bill_cost_l192_192653

-- Definitions of costs and conditions
def curtis_meal_cost : ℝ := 16.00
def rob_meal_cost : ℝ := 18.00
def total_cost_before_discount : ℝ := curtis_meal_cost + rob_meal_cost
def discount_rate : ℝ := 0.5
def time_of_meal : ℝ := 3.0

-- Condition for discount applicability
def discount_applicable : Prop := 2.0 ≤ time_of_meal ∧ time_of_meal ≤ 4.0

-- Total cost with discount applied
def cost_with_discount (total_cost : ℝ) (rate : ℝ) : ℝ := total_cost * rate

-- Theorem statement we need to prove
theorem total_bill_cost :
  discount_applicable →
  cost_with_discount total_cost_before_discount discount_rate = 17.00 :=
by
  sorry

end total_bill_cost_l192_192653


namespace group_count_l192_192712

theorem group_count (sample_capacity : ℕ) (frequency : ℝ) (h_sample_capacity : sample_capacity = 80) (h_frequency : frequency = 0.125) : sample_capacity * frequency = 10 := 
by
  sorry

end group_count_l192_192712


namespace correct_average_calculation_l192_192245

-- Conditions as definitions
def incorrect_average := 5
def num_values := 10
def incorrect_num := 26
def correct_num := 36

-- Statement to prove
theorem correct_average_calculation : 
  (incorrect_average * num_values + (correct_num - incorrect_num)) / num_values = 6 :=
by
  -- Placeholder for the proof
  sorry

end correct_average_calculation_l192_192245


namespace infinite_series_sum_l192_192591

theorem infinite_series_sum :
  ∑' (n : ℕ), (1 / (1 + 3^n : ℝ) - 1 / (1 + 3^(n+1) : ℝ)) = 1/2 := 
sorry

end infinite_series_sum_l192_192591


namespace measure_of_angle_E_l192_192628

theorem measure_of_angle_E
    (A B C D E F : ℝ)
    (h1 : A = B)
    (h2 : B = C)
    (h3 : C = D)
    (h4 : E = F)
    (h5 : A = E - 30)
    (h6 : A + B + C + D + E + F = 720) :
  E = 140 :=
by
  -- Proof goes here
  sorry

end measure_of_angle_E_l192_192628


namespace arc_length_of_given_curve_l192_192279

open Real

noncomputable def arc_length (f : ℝ → ℝ) (a b : ℝ) :=
  ∫ x in a..b, sqrt (1 + (deriv f x)^2)

noncomputable def given_function (x : ℝ) : ℝ :=
  arccos (sqrt x) - sqrt (x - x^2) + 4

theorem arc_length_of_given_curve :
  arc_length given_function 0 (1/2) = sqrt 2 :=
by
  sorry

end arc_length_of_given_curve_l192_192279


namespace probability_lt_2y_l192_192296

noncomputable def rectangle : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

noncomputable def region : set (ℝ × ℝ) := {p | p ∈ rectangle ∧ p.1 < 2 * p.2 }

noncomputable def area_rectangle : ℝ := measure_theory.measure_space.measure_univ (set.univ.restrict rectangle)

noncomputable def area_region : ℝ := measure_theory.measure_space.measure_univ (set.univ.restrict region)

theorem probability_lt_2y : area_region / area_rectangle = 1 / 6 :=
begin
  sorry
end

end probability_lt_2y_l192_192296


namespace all_points_equal_l192_192734

-- Define the problem conditions and variables
variable (P : Type) -- points in the plane
variable [MetricSpace P] -- the plane is a metric space
variable (f : P → ℝ) -- assignment of numbers to points
variable (incenter : P → P → P → P) -- calculates incenter of a nondegenerate triangle

-- Condition: the value at the incenter of a triangle is the arithmetic mean of the values at the vertices
axiom incenter_mean_property : ∀ (A B C : P), 
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  f (incenter A B C) = (f A + f B + f C) / 3

-- The theorem to be proved
theorem all_points_equal : ∀ x y : P, f x = f y :=
by
  sorry

end all_points_equal_l192_192734


namespace max_volumes_on_fedor_shelf_l192_192781

theorem max_volumes_on_fedor_shelf 
  (S s1 s2 n : ℕ) 
  (h1 : S + s1 ≥ (n - 2) / 2) 
  (h2 : S + s2 < (n - 2) / 3) 
  : n = 12 := 
sorry

end max_volumes_on_fedor_shelf_l192_192781


namespace simplify_sum_of_square_roots_l192_192956

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l192_192956


namespace x_y_z_sum_l192_192757

theorem x_y_z_sum :
  ∃ (x y z : ℕ), (16 / 3)^x * (27 / 25)^y * (5 / 4)^z = 256 ∧ x + y + z = 6 :=
by
  -- Proof can be completed here
  sorry

end x_y_z_sum_l192_192757


namespace popsicle_sticks_left_l192_192367

theorem popsicle_sticks_left (initial_sticks given_per_group groups : ℕ) 
  (h_initial : initial_sticks = 170)
  (h_given : given_per_group = 15)
  (h_groups : groups = 10) : 
  initial_sticks - (given_per_group * groups) = 20 := by
  rw [h_initial, h_given, h_groups]
  norm_num
  sorry -- Alternatively: exact eq.refl 20

end popsicle_sticks_left_l192_192367


namespace sum_of_three_numbers_l192_192554

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h : a + (b * c) = (a + b) * (a + c)) : a + b + c = 1 :=
by
  sorry

end sum_of_three_numbers_l192_192554


namespace probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l192_192883

-- Definitions and conditions from the problem
def success_probability_each_segment : ℚ := 3 / 4
def num_segments : ℕ := 4

-- Correct answers from the solution
def prob_both_success : ℚ := 9 / 16
def expected_successful_segments : ℚ := 3
def cond_prob_given_3_successful : ℚ := 3 / 4

theorem probability_both_segments_successful :
  (success_probability_each_segment * success_probability_each_segment) = prob_both_success :=
by
  sorry

theorem expected_number_of_successful_segments :
  (num_segments * success_probability_each_segment) = expected_successful_segments :=
by
  sorry

theorem probability_given_3_successful :
  let prob_M := 4 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  let prob_NM := 3 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  (prob_NM / prob_M) = cond_prob_given_3_successful :=
by
  sorry

end probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l192_192883


namespace det_is_18_l192_192803

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1],
    ![2, 5]]

theorem det_is_18 : det A = 18 := by
  sorry

end det_is_18_l192_192803


namespace acute_triangle_l192_192038

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ area, 
    (area = (1 / 2) * a * b * Real.sin C) ∧
    (a / Real.sin A = 2 * c / Real.sqrt 3) ∧
    (c = Real.sqrt 7) ∧
    (area = (3 * Real.sqrt 3) / 2)

theorem acute_triangle (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) :
  C = 60 ∧ a^2 + b^2 = 13 :=
by
  obtain ⟨_, h_area, h_sine, h_c, h_area_eq⟩ := h
  sorry

end acute_triangle_l192_192038


namespace route_time_difference_l192_192486

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end route_time_difference_l192_192486


namespace max_expr_on_circle_l192_192894

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 6 * y + 4 = 0

noncomputable def expr (x y : ℝ) : ℝ :=
  3 * x - 4 * y

theorem max_expr_on_circle : 
  ∃ (x y : ℝ), circle_eq x y ∧ ∀ (x' y' : ℝ), circle_eq x' y' → expr x y ≤ expr x' y' :=
sorry

end max_expr_on_circle_l192_192894


namespace solve_for_k_l192_192274

theorem solve_for_k :
  (∀ x : ℤ, (2 * x + 4 = 4 * (x - 2)) ↔ ( -x + 17 = 2 * x - 1 )) :=
by
  sorry

end solve_for_k_l192_192274


namespace probability_of_perfect_square_or_multiple_of_5_l192_192681

noncomputable def perfect_square_or_multiple_of_5_probability : ℚ :=
  let dice := {1, 2, 3, 4, 5, 6}
  let sum_count := (dice.product dice).product dice.count
    (λ t, let (d1, (d2, d3)) := t in
          let s := d1 + d2 + d3 in
          s = 4 ∨ s = 5 ∨ s = 9 ∨ s = 10 ∨ s = 15 ∨ s = 16)
  sum_count / 216

theorem probability_of_perfect_square_or_multiple_of_5 :
  perfect_square_or_multiple_of_5_probability = 77 / 216 := by
  sorry

end probability_of_perfect_square_or_multiple_of_5_l192_192681


namespace sphere_radius_and_volume_l192_192895

theorem sphere_radius_and_volume (A : ℝ) (d : ℝ) (π : ℝ) (r : ℝ) (R : ℝ) (V : ℝ) 
  (h_cross_section : A = π) (h_distance : d = 1) (h_radius : r = 1) :
  R = Real.sqrt (r^2 + d^2) ∧ V = (4 / 3) * π * R^3 := 
by
  sorry

end sphere_radius_and_volume_l192_192895


namespace jane_played_rounds_l192_192037

-- Define the conditions
def points_per_round := 10
def points_ended_with := 60
def points_lost := 20

-- Define the proof problem
theorem jane_played_rounds : (points_ended_with + points_lost) / points_per_round = 8 :=
by
  sorry

end jane_played_rounds_l192_192037


namespace directrix_of_parabola_l192_192506

theorem directrix_of_parabola (x y : ℝ) : (x ^ 2 = y) → (4 * y + 1 = 0) :=
sorry

end directrix_of_parabola_l192_192506


namespace num_positive_divisors_of_720_multiples_of_5_l192_192905

theorem num_positive_divisors_of_720_multiples_of_5 :
  (∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 2 ∧ c = 1) →
  ∃ (n : ℕ), n = 15 :=
by
  -- Proof will go here
  sorry

end num_positive_divisors_of_720_multiples_of_5_l192_192905


namespace congruence_problem_l192_192907

theorem congruence_problem (x : ℤ) (h : 5 * x + 9 ≡ 4 [ZMOD 18]) : 3 * x + 15 ≡ 12 [ZMOD 18] :=
sorry

end congruence_problem_l192_192907


namespace order_of_abc_l192_192598

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem order_of_abc : c < a ∧ a < b :=
by {
  -- The proof would go here
  sorry
}

end order_of_abc_l192_192598


namespace function_symmetry_extremum_l192_192763

noncomputable def f (x θ : ℝ) : ℝ := 3 * Real.cos (Real.pi * x + θ)

theorem function_symmetry_extremum {θ : ℝ} (H : ∀ x : ℝ, f x θ = f (2 - x) θ) : 
  f 1 θ = 3 ∨ f 1 θ = -3 :=
by
  sorry

end function_symmetry_extremum_l192_192763


namespace percentage_vehicles_updated_2003_l192_192094

theorem percentage_vehicles_updated_2003 (a : ℝ) (h1 : 1.1^4 = 1.46) (h2 : 1.1^5 = 1.61) :
  (a * 1 / (a * 1.61) * 100 = 16.4) :=
  by sorry

end percentage_vehicles_updated_2003_l192_192094


namespace initial_tax_rate_l192_192621

theorem initial_tax_rate 
  (income : ℝ)
  (differential_savings : ℝ)
  (final_tax_rate : ℝ)
  (initial_tax_rate : ℝ) 
  (h1 : income = 42400) 
  (h2 : differential_savings = 4240) 
  (h3 : final_tax_rate = 32)
  (h4 : differential_savings = (initial_tax_rate / 100) * income - (final_tax_rate / 100) * income) :
  initial_tax_rate = 42 :=
sorry

end initial_tax_rate_l192_192621


namespace penelope_food_intake_l192_192639

theorem penelope_food_intake
(G P M E : ℕ) -- Representing amount of food each animal eats per day
(h1 : P = 10 * G) -- Penelope eats 10 times Greta's food
(h2 : M = G / 100) -- Milton eats 1/100 of Greta's food
(h3 : E = 4000 * M) -- Elmer eats 4000 times what Milton eats
(h4 : E = P + 60) -- Elmer eats 60 pounds more than Penelope
(G_val : G = 2) -- Greta eats 2 pounds per day
: P = 20 := -- Prove Penelope eats 20 pounds per day
by
  rw [G_val] at h1 -- Replace G with 2 in h1
  norm_num at h1 -- Evaluate the expression in h1
  exact h1 -- Conclude P = 20

end penelope_food_intake_l192_192639


namespace perpendicular_lines_condition_l192_192600

theorem perpendicular_lines_condition (a : ℝ) :
  (¬ a = 1/2 ∨ ¬ a = -1/2) ∧ a * (-4 * a) = -1 ↔ a = 1/2 :=
by
  sorry

end perpendicular_lines_condition_l192_192600


namespace find_a_l192_192325

-- Definitions of the conditions
variables {a b c : ℤ} 

-- Theorem statement
theorem find_a (h1: a + b = c) (h2: b + c = 7) (h3: c = 4) : a = 1 :=
by
  -- Using sorry to skip the proof
  sorry

end find_a_l192_192325


namespace domain_f_l192_192869

noncomputable def f (x : ℝ) : ℝ := -2 / (Real.sqrt (x + 5)) + Real.log (2^x + 1)

theorem domain_f :
  {x : ℝ | (-5 ≤ x)} = {x : ℝ | f x ∈ Set.univ} := sorry

end domain_f_l192_192869


namespace closest_cube_root_l192_192116

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l192_192116


namespace autumn_sales_l192_192918

theorem autumn_sales (T : ℝ) (spring summer winter autumn : ℝ) 
    (h1 : spring = 3)
    (h2 : summer = 6)
    (h3 : winter = 5)
    (h4 : T = (3 / 0.2)) :
    autumn = 1 :=
by 
  -- Proof goes here
  sorry

end autumn_sales_l192_192918


namespace even_product_divisible_by_1947_l192_192273

theorem even_product_divisible_by_1947 (n : ℕ) (h_even : n % 2 = 0) :
  (∃ k: ℕ, 2 ≤ k ∧ k ≤ n / 2 ∧ 1947 ∣ (2 ^ k * k!)) → n ≥ 3894 :=
by
  sorry

end even_product_divisible_by_1947_l192_192273


namespace line_intersects_ellipse_l192_192509

theorem line_intersects_ellipse (b : ℝ) : (∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + 1 → ((x^2 / 5) + (y^2 / b) = 1))
  ↔ b ∈ (Set.Ico 1 5 ∪ Set.Ioi 5) := by
sorry

end line_intersects_ellipse_l192_192509


namespace original_number_of_employees_l192_192411

theorem original_number_of_employees (E : ℝ) :
  (E - 0.125 * E) - 0.09 * (E - 0.125 * E) = 12385 → E = 15545 := 
by  -- Start the proof
  sorry  -- Placeholder for the proof, which is not required

end original_number_of_employees_l192_192411


namespace find_coords_of_P_l192_192979

-- Definitions from the conditions
def line_eq (x y : ℝ) : Prop := x - y - 7 = 0
def is_midpoint (P Q M : ℝ × ℝ) : Prop := 
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Coordinates given in the problem
def P : ℝ × ℝ := (-2, 1)

-- The proof goal
theorem find_coords_of_P : ∃ Q : ℝ × ℝ,
  is_midpoint P Q (1, -1) ∧ 
  line_eq Q.1 Q.2 :=
sorry

end find_coords_of_P_l192_192979


namespace domain_myFunction_l192_192524

noncomputable def myFunction (x : ℝ) : ℝ :=
  (x^3 - 125) / (x + 125)

theorem domain_myFunction :
  {x : ℝ | ∀ y, y = myFunction x → x ≠ -125} = { x : ℝ | x ≠ -125 } := 
by
  sorry

end domain_myFunction_l192_192524


namespace find_f_prime_at_2_l192_192540

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l192_192540


namespace ratio_of_fruit_salads_l192_192849

theorem ratio_of_fruit_salads 
  (salads_Alaya : ℕ) 
  (total_salads : ℕ) 
  (h1 : salads_Alaya = 200) 
  (h2 : total_salads = 600) : 
  (total_salads - salads_Alaya) / salads_Alaya = 2 :=
by 
  sorry

end ratio_of_fruit_salads_l192_192849


namespace sqrt_72_plus_sqrt_32_l192_192965

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l192_192965


namespace find_a_if_even_function_l192_192204

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

theorem find_a_if_even_function (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 / 2 := by
  sorry

end find_a_if_even_function_l192_192204


namespace Penelope_daily_savings_l192_192638

theorem Penelope_daily_savings
  (total_savings : ℝ)
  (days_in_year : ℕ)
  (h1 : total_savings = 8760)
  (h2 : days_in_year = 365) :
  total_savings / days_in_year = 24 :=
by
  sorry

end Penelope_daily_savings_l192_192638


namespace cubic_with_root_p_sq_l192_192079

theorem cubic_with_root_p_sq (p : ℝ) (hp : p^3 + p - 3 = 0) : (p^2 : ℝ) ^ 3 + 2 * (p^2) ^ 2 + p^2 - 9 = 0 :=
sorry

end cubic_with_root_p_sq_l192_192079


namespace alok_age_proof_l192_192149

variable (A B C : ℕ)

theorem alok_age_proof (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  sorry

end alok_age_proof_l192_192149


namespace closest_integer_to_cubert_of_250_is_6_l192_192106

theorem closest_integer_to_cubert_of_250_is_6 :
  ∃ n : ℤ, n = 6 ∧ |250 - n^3| < |250 - 7^3| ∧ |250 - n^3| < |250 - 5^3| := 
by {
  use 6,
  split,
  { refl, },
  {
    split,
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    },
    {
      norm_num,
      exact abs_lt.mpr ⟨by norm_num, by norm_num⟩,
    }
  }
}

end closest_integer_to_cubert_of_250_is_6_l192_192106


namespace fraction_of_jumbo_tiles_l192_192141

-- Definitions for conditions
variables (L W : ℝ) -- Length and width of regular tiles
variables (n : ℕ) -- Number of regular tiles
variables (m : ℕ) -- Number of jumbo tiles

-- Conditions
def condition1 : Prop := (n : ℝ) * (L * W) = 40 -- Regular tiles cover 40 square feet
def condition2 : Prop := (n : ℝ) * (L * W) + (m : ℝ) * (3 * L * W) = 220 -- Entire wall is 220 square feet
def condition3 : Prop := ∃ (k : ℝ), (m : ℝ) = k * (n : ℝ) ∧ k = 1.5 -- Relationship ratio between jumbo and regular tiles

-- Theorem to be proved
theorem fraction_of_jumbo_tiles (L W : ℝ) (n m : ℕ)
  (h1 : condition1 L W n)
  (h2 : condition2 L W n m)
  (h3 : condition3 n m) :
  (m : ℝ) / ((n : ℝ) + (m : ℝ)) = 3 / 5 :=
sorry

end fraction_of_jumbo_tiles_l192_192141


namespace cheese_stick_problem_l192_192350

theorem cheese_stick_problem (cheddar pepperjack mozzarella : ℕ) (total : ℕ)
    (h1 : cheddar = 15)
    (h2 : pepperjack = 45)
    (h3 : 2 * pepperjack = total)
    (h4 : total = cheddar + pepperjack + mozzarella) :
    mozzarella = 30 :=
by
    sorry

end cheese_stick_problem_l192_192350


namespace sqrt_sum_simplify_l192_192962

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l192_192962


namespace matrix_cubic_l192_192773

noncomputable def matrix_entries (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

theorem matrix_cubic (x y z : ℝ) (N : Matrix (Fin 3) (Fin 3) ℝ)
    (hN : N = matrix_entries x y z)
    (hn : N ^ 2 = 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ))
    (hxyz : x * y * z = -2) :
  x^3 + y^3 + z^3 = -6 + 2 * Real.sqrt 2 ∨ x^3 + y^3 + z^3 = -6 - 2 * Real.sqrt 2 :=
by
  sorry

end matrix_cubic_l192_192773


namespace simplify_radicals_l192_192948

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l192_192948


namespace sqrt_72_plus_sqrt_32_l192_192967

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l192_192967


namespace circle_radius_order_l192_192421

theorem circle_radius_order (r_X r_Y r_Z : ℝ)
  (hX : r_X = π)
  (hY : 2 * π * r_Y = 8 * π)
  (hZ : π * r_Z^2 = 9 * π) :
  r_Z < r_X ∧ r_X < r_Y :=
by {
  sorry
}

end circle_radius_order_l192_192421


namespace percentage_of_water_in_fresh_grapes_l192_192317

theorem percentage_of_water_in_fresh_grapes
  (P : ℝ)  -- Let P be the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 5)  -- weight of fresh grapes in kg
  (dried_grapes_weight : ℝ := 0.625)  -- weight of dried grapes in kg
  (dried_water_percentage : ℝ := 20)  -- percentage of water in dried grapes
  (h1 : (100 - P) / 100 * fresh_grapes_weight = (100 - dried_water_percentage) / 100 * dried_grapes_weight) :
  P = 90 := 
sorry

end percentage_of_water_in_fresh_grapes_l192_192317


namespace part_a_l192_192223

def A (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x * y) = x * f y

theorem part_a (f : ℝ → ℝ) (h : A f) : ∀ x y : ℝ, f (x + y) = f x + f y :=
sorry

end part_a_l192_192223


namespace fraction_value_l192_192691

theorem fraction_value : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end fraction_value_l192_192691


namespace greatest_possible_length_l192_192646

-- Define the lengths of the ropes
def rope_lengths : List ℕ := [72, 48, 120, 96]

-- Define the gcd function to find the greatest common divisor of a list of numbers
def list_gcd (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- Define the target problem statement
theorem greatest_possible_length 
  (h : list_gcd rope_lengths = 24) : 
  ∀ length ∈ rope_lengths, length % 24 = 0 :=
by
  intros length h_length
  sorry

end greatest_possible_length_l192_192646


namespace determinant_inequality_l192_192782

theorem determinant_inequality (x : ℝ) (h : 2 * x - (3 - x) > 0) : 3 * x - 3 > 0 := 
by
  sorry

end determinant_inequality_l192_192782


namespace line_segment_length_is_0_7_l192_192292

def isLineSegment (length : ℝ) (finite : Bool) : Prop :=
  finite = true ∧ length = 0.7

theorem line_segment_length_is_0_7 : isLineSegment 0.7 true :=
by
  sorry

end line_segment_length_is_0_7_l192_192292


namespace maximize_S_n_l192_192627

variable {a : ℕ → ℝ} -- Sequence term definition
variable {S : ℕ → ℝ} -- Sum of first n terms

-- Definitions based on conditions
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * a 1 + (n * (n - 1) / 2) * ((a 2) - (a 1))

axiom a1_positive (a1 : ℝ) : 0 < a1 -- given a1 > 0
axiom S3_eq_S16 (a1 d : ℝ) : sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16

-- Problem Statement
theorem maximize_S_n (a : ℕ → ℝ) (d : ℝ) : is_arithmetic_sequence a d →
  a 1 > 0 →
  sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16 →
  (∀ n, sum_of_first_n_terms a n = sum_of_first_n_terms a 9 ∨ sum_of_first_n_terms a n = sum_of_first_n_terms a 10) :=
by
  sorry

end maximize_S_n_l192_192627


namespace probability_top_card_special_l192_192299

-- Definition of the problem conditions
def deck_size : ℕ := 52
def special_card_count : ℕ := 16

-- The statement we need to prove
theorem probability_top_card_special : 
  (special_card_count : ℚ) / deck_size = 4 / 13 := 
  by sorry

end probability_top_card_special_l192_192299


namespace range_of_m_l192_192451

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m) : -3 < m ∧ m < 1 := 
sorry

end range_of_m_l192_192451


namespace problem_example_l192_192477

theorem problem_example (a : ℕ) (H1 : a ∈ ({a, b, c} : Set ℕ)) (H2 : 0 ∈ ({x | x^2 ≠ 0} : Set ℕ)) :
  a ∈ ({a, b, c} : Set ℕ) ∧ 0 ∈ ({x | x^2 ≠ 0} : Set ℕ) :=
by
  sorry

end problem_example_l192_192477


namespace max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l192_192178

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a ≤ 2 :=
by
  -- Proof omitted
  sorry

theorem le_2_and_ge_neg_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : -2 ≤ a :=
by
  -- Proof omitted
  sorry

theorem max_a_is_2 (a : ℝ) (h3 : a ≤ 2) (h4 : -2 ≤ a) : a = 2 :=
by
  -- Proof omitted
  sorry

end max_value_of_a_le_2_and_ge_neg_2_max_a_is_2_l192_192178


namespace coefficient_x7_in_expansion_l192_192686

theorem coefficient_x7_in_expansion : 
  let n := 10
  let k := 7
  let binom := Nat.choose n k
  let coeff := 1
  coeff * binom = 120 :=
by
  sorry

end coefficient_x7_in_expansion_l192_192686


namespace rainfall_second_week_value_l192_192309

-- Define the conditions
variables (rainfall_first_week rainfall_second_week : ℝ)
axiom condition1 : rainfall_first_week + rainfall_second_week = 30
axiom condition2 : rainfall_second_week = 1.5 * rainfall_first_week

-- Define the theorem we want to prove
theorem rainfall_second_week_value : rainfall_second_week = 18 := by
  sorry

end rainfall_second_week_value_l192_192309


namespace david_marks_in_physics_l192_192731

theorem david_marks_in_physics 
  (english_marks mathematics_marks chemistry_marks biology_marks : ℕ)
  (num_subjects : ℕ)
  (average_marks : ℕ)
  (h1 : english_marks = 81)
  (h2 : mathematics_marks = 65)
  (h3 : chemistry_marks = 67)
  (h4 : biology_marks = 85)
  (h5 : num_subjects = 5)
  (h6 : average_marks = 76) :
  ∃ physics_marks : ℕ, physics_marks = 82 :=
by
  sorry

end david_marks_in_physics_l192_192731


namespace area_less_than_perimeter_probability_l192_192412

-- Define the side length s as the sum of a pair of dice rolls (ranging from 2 to 12)
noncomputable def sum_of_dice_rolls : ℕ := sorry

-- Define the probability function for a given outcome of dice rolls
noncomputable def probability_of_sum (s : ℕ) : ℚ := sorry

-- Define the probability that the side length s is less than 4
noncomputable def probability_s_less_than_4 : ℚ :=
  probability_of_sum 2 + probability_of_sum 3

-- State the theorem to prove the probability is 1/12
theorem area_less_than_perimeter_probability : probability_s_less_than_4 = 1/12 :=
by
  sorry

end area_less_than_perimeter_probability_l192_192412


namespace first_candidate_percentage_l192_192404

noncomputable
def passing_marks_approx : ℝ := 240

noncomputable
def total_marks (P : ℝ) : ℝ := (P + 30) / 0.45

noncomputable
def percentage_marks (T P : ℝ) : ℝ := ((P - 60) / T) * 100

theorem first_candidate_percentage :
  let P := passing_marks_approx
  let T := total_marks P
  percentage_marks T P = 30 :=
by
  sorry

end first_candidate_percentage_l192_192404


namespace total_cost_of_bill_l192_192655

def original_price_curtis := 16.00
def original_price_rob := 18.00
def time_of_meal := 3

def is_early_bird_discount_applicable (time : ℕ) : Prop :=
  2 ≤ time ∧ time ≤ 4

theorem total_cost_of_bill :
  is_early_bird_discount_applicable time_of_meal →
  original_price_curtis / 2 + original_price_rob / 2 = 17.00 :=
by
  sorry

end total_cost_of_bill_l192_192655


namespace evaluate_expression_l192_192425

theorem evaluate_expression :
  ((3.5 / 0.7) * (5 / 3) + (7.2 / 0.36) - ((5 / 3) * (0.75 / 0.25))) = 23.3335 :=
by
  sorry

end evaluate_expression_l192_192425


namespace solution_set_inequality_l192_192327

noncomputable def f (x : ℝ) := Real.exp (2 * x) - 1
noncomputable def g (x : ℝ) := Real.log (x + 1)

theorem solution_set_inequality :
  {x : ℝ | f (g x) - g (f x) ≤ 1} = Set.Icc (-1 : ℝ) 1 :=
sorry

end solution_set_inequality_l192_192327


namespace part_I_part_II_l192_192611

noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x
noncomputable def g (x k : ℝ) : ℝ := k * x / (x + 1)

theorem part_I (k : ℝ) : 
  (∃ x0, g x0 k = x0 + 4 ∧ (k / (x0 + 1)^2) = 1) ↔ (k = 1 ∨ k = 9) :=
by
  sorry

theorem part_II (k : ℕ) : (∀ x : ℝ, 1 < x → f x > g x k) → k ≤ 7 :=
by
  sorry

end part_I_part_II_l192_192611


namespace find_m_l192_192339

theorem find_m (x y m : ℝ)
  (h1 : 6 * x + 3 = 0)
  (h2 : 3 * y + m = 15)
  (h3 : x * y = 1) : m = 21 := 
sorry

end find_m_l192_192339


namespace mutual_independence_of_A_and_D_l192_192817

noncomputable theory

variables (Ω : Type) [ProbabilitySpace Ω]
-- Definition of events A, B, C, D as sets over Ω
def event_A : Event Ω := {ω | some_condition_for_A}
def event_B : Event Ω := {ω | some_condition_for_B}
def event_C : Event Ω := {ω | some_condition_for_C}
def event_D : Event Ω := {ω | some_condition_for_D}

-- Given probabilities
axiom P_A : P(event_A Ω) = 1 / 6
axiom P_B : P(event_B Ω) = 1 / 6
axiom P_C : P(event_C Ω) = 5 / 36
axiom P_D : P(event_D Ω) = 1 / 6

-- Independence definition
def are_independent (X Y : Event Ω) : Prop :=
  P(X ∩ Y) = P(X) * P(Y)

-- The problem statement: proving A and D are independent
theorem mutual_independence_of_A_and_D : are_independent Ω (event_A Ω) (event_D Ω) :=
sorry

end mutual_independence_of_A_and_D_l192_192817


namespace fractions_order_l192_192268

theorem fractions_order:
  (20 / 15) < (25 / 18) ∧ (25 / 18) < (23 / 16) ∧ (23 / 16) < (21 / 14) :=
by
  sorry

end fractions_order_l192_192268


namespace solution_set_f_le_1_l192_192014

variable {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_f_le_1 :
  is_even_function f →
  monotone_on_nonneg f →
  f (-2) = 1 →
  {x : ℝ | f x ≤ 1} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by
  intros h_even h_mono h_f_neg_2
  sorry

end solution_set_f_le_1_l192_192014


namespace arithmetic_sequence_sum_l192_192354

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h_arithmetic : ∀ n, a (n+1) = a n + d)
  (h_pos_diff : d > 0)
  (h_sum_3 : a 0 + a 1 + a 2 = 15)
  (h_prod_3 : a 0 * a 1 * a 2 = 80) :
  a 10 + a 11 + a 12 = 105 :=
sorry

end arithmetic_sequence_sum_l192_192354


namespace eric_has_9306_erasers_l192_192580

-- Define the conditions as constants
def number_of_friends := 99
def erasers_per_friend := 94

-- Define the total number of erasers based on the conditions
def total_erasers := number_of_friends * erasers_per_friend

-- Theorem stating the total number of erasers Eric has
theorem eric_has_9306_erasers : total_erasers = 9306 := by
  -- Proof to be filled in
  sorry

end eric_has_9306_erasers_l192_192580


namespace sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l192_192880

theorem sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31 :
  let n := 7^7 - 7^4 in
  let prime_factors := {2, 3, 7, 19} in
  finset.sum prime_factors id = 31 :=
by
  sorry

end sum_of_distinct_prime_factors_of_7_pow_7_minus_7_pow_4_eq_31_l192_192880


namespace find_pq_cube_l192_192188

theorem find_pq_cube (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 3) : (p + q) ^ 3 = 125 := 
by
  -- This is where the proof would go
  sorry

end find_pq_cube_l192_192188


namespace day_after_1999_cubed_days_is_tuesday_l192_192517

theorem day_after_1999_cubed_days_is_tuesday : 
    let today := "Monday"
    let days_in_week := 7
    let target_days := 1999 ^ 3
    ∃ remaining_days, remaining_days = (target_days % days_in_week) ∧ today = "Monday" ∧ remaining_days = 1 → 
    "Tuesday" = "Tuesday" := 
by
  sorry

end day_after_1999_cubed_days_is_tuesday_l192_192517


namespace polygon_interior_angle_sum_l192_192660

theorem polygon_interior_angle_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * (n - 2 + 3) = 2880 := by
  sorry

end polygon_interior_angle_sum_l192_192660


namespace remainder_sum_l192_192099

-- Define the conditions given in the problem.
def remainder_13_mod_5 : ℕ := 3
def remainder_12_mod_5 : ℕ := 2
def remainder_11_mod_5 : ℕ := 1

theorem remainder_sum :
  ((13 ^ 6 + 12 ^ 7 + 11 ^ 8) % 5) = 3 := by
  sorry

end remainder_sum_l192_192099


namespace min_unit_cubes_intersect_all_l192_192135

theorem min_unit_cubes_intersect_all (n : ℕ) : 
  let A_n := if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2
  A_n = if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2 :=
sorry

end min_unit_cubes_intersect_all_l192_192135


namespace probability_three_fair_coins_l192_192820

noncomputable def probability_one_head_two_tails (n : ℕ) : ℚ :=
  if n = 3 then 3 / 8 else 0

theorem probability_three_fair_coins :
  probability_one_head_two_tails 3 = 3 / 8 :=
by
  sorry

end probability_three_fair_coins_l192_192820


namespace largest_number_of_stores_visited_l192_192389

theorem largest_number_of_stores_visited
  (stores : ℕ) (total_visits : ℕ) (total_peopled_shopping : ℕ)
  (people_visiting_2_stores : ℕ) (people_visiting_3_stores : ℕ)
  (people_visiting_4_stores : ℕ) (people_visiting_1_store : ℕ)
  (everyone_visited_at_least_one_store : ∀ p : ℕ, 0 < people_visiting_1_store + people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores)
  (h1 : stores = 15) (h2 : total_visits = 60) (h3 : total_peopled_shopping = 30)
  (h4 : people_visiting_2_stores = 12) (h5 : people_visiting_3_stores = 6)
  (h6 : people_visiting_4_stores = 4) (h7 : people_visiting_1_store = total_peopled_shopping - (people_visiting_2_stores + people_visiting_3_stores + people_visiting_4_stores + 2)) :
  ∃ p : ℕ, ∀ person, person ≤ p ∧ p = 4 := sorry

end largest_number_of_stores_visited_l192_192389


namespace ark5_ensures_metabolic_energy_l192_192941

-- Define conditions
def inhibits_ark5_activity (inhibits: Bool) (balance: Bool): Prop :=
  if inhibits then ¬balance else balance

def cancer_cells_proliferate_without_energy (proliferate: Bool) (die_due_to_insufficient_energy: Bool) : Prop :=
  proliferate → die_due_to_insufficient_energy

-- Define the hypothesis based on conditions
def hypothesis (inhibits: Bool) (balance: Bool) (proliferate: Bool) (die_due_to_insufficient_energy: Bool): Prop :=
  inhibits_ark5_activity inhibits balance ∧ cancer_cells_proliferate_without_energy proliferate die_due_to_insufficient_energy

-- Define the theorem to be proved
theorem ark5_ensures_metabolic_energy
  (inhibits : Bool)
  (balance : Bool)
  (proliferate : Bool)
  (die_due_to_insufficient_energy : Bool)
  (h : hypothesis inhibits balance proliferate die_due_to_insufficient_energy) :
  ensures_metabolic_energy :=
  sorry

end ark5_ensures_metabolic_energy_l192_192941


namespace sturdy_square_impossible_l192_192349

def size : ℕ := 6
def dominos_used : ℕ := 18
def cells_per_domino : ℕ := 2
def total_cells : ℕ := size * size
def dividing_lines : ℕ := 10

def is_sturdy_square (grid_size : ℕ) (domino_count : ℕ) : Prop :=
  grid_size * grid_size = domino_count * cells_per_domino ∧ 
  ∀ line : ℕ, line < dividing_lines → ∃ domino : ℕ, domino < domino_count

theorem sturdy_square_impossible 
    (grid_size : ℕ) (domino_count : ℕ)
    (h1 : grid_size = size) (h2 : domino_count = dominos_used)
    (h3 : cells_per_domino = 2) (h4 : dividing_lines = 10) : 
  ¬ is_sturdy_square grid_size domino_count :=
by
  cases h1
  cases h2
  cases h3
  cases h4
  sorry

end sturdy_square_impossible_l192_192349


namespace operation_addition_x_l192_192393

theorem operation_addition_x (x : ℕ) (h : 106 + 106 + x + x = 19872) : x = 9830 :=
sorry

end operation_addition_x_l192_192393


namespace find_north_speed_l192_192096

-- Define the variables and conditions
variables (v : ℝ)  -- the speed of the cyclist going towards the north
def south_speed : ℝ := 25  -- the speed of the cyclist going towards the south is 25 km/h
def time_taken : ℝ := 1.4285714285714286  -- time taken to be 50 km apart
def distance_apart : ℝ := 50  -- distance apart after given time

-- Define the hypothesis based on the conditions
def relative_speed (v : ℝ) : ℝ := v + south_speed
def distance_formula (v : ℝ) : Prop :=
  distance_apart = relative_speed v * time_taken

-- The statement to prove
theorem find_north_speed : distance_formula v → v = 10 :=
  sorry

end find_north_speed_l192_192096


namespace student_history_mark_l192_192713

theorem student_history_mark
  (math_score : ℕ)
  (desired_average : ℕ)
  (third_subject_score : ℕ)
  (history_score : ℕ) :
  math_score = 74 →
  desired_average = 75 →
  third_subject_score = 70 →
  (math_score + history_score + third_subject_score) / 3 = desired_average →
  history_score = 81 :=
by
  intros h_math h_avg h_third h_equiv
  sorry

end student_history_mark_l192_192713


namespace smallest_number_l192_192987

theorem smallest_number (x : ℕ) : (∃ y : ℕ, y = x - 16 ∧ (y % 4 = 0) ∧ (y % 6 = 0) ∧ (y % 8 = 0) ∧ (y % 10 = 0)) → x = 136 := by
  sorry

end smallest_number_l192_192987


namespace fraction_value_l192_192321

variable (a b : ℚ)  -- Variables a and b are rational numbers

theorem fraction_value (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_value_l192_192321


namespace first_term_of_geometric_sequence_l192_192387

-- Define a geometric sequence
def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Initialize conditions
variable (a r : ℝ)

-- Provided that the 3rd term and the 6th term
def third_term : Prop := geometric_sequence a r 2 = 5
def sixth_term : Prop := geometric_sequence a r 5 = 40

-- The theorem to prove that a == 5/4 given the conditions
theorem first_term_of_geometric_sequence : third_term a r ∧ sixth_term a r → a = 5 / 4 :=
by 
  sorry

end first_term_of_geometric_sequence_l192_192387


namespace option_A_option_B_option_C_option_D_l192_192542

-- Option A
theorem option_A (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 
  (x-1)^2 + x*(x-4) + (x-2)*(x+2) ≠ 0 := 
sorry

-- Option B
theorem option_B (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^3 + (1/x)^3 - 3 = 15 := 
sorry

-- Option C
theorem option_C (x : ℝ) (a b c : ℝ) (h_a : a = 1 / 20 * x + 20) (h_b : b = 1 / 20 * x + 19) (h_c : c = 1 / 20 * x + 21) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := 
sorry

-- Option D
theorem option_D (x m n : ℝ) (h : 2*x^2 - 8*x + 7 = 0) (h_roots : m + n = 4 ∧ m * n = 7/2) : 
  Real.sqrt (m^2 + n^2) = 3 := 
sorry

end option_A_option_B_option_C_option_D_l192_192542


namespace solution_set_for_inequality_l192_192087

theorem solution_set_for_inequality :
  {x : ℝ | (1 / (x - 1) ≥ -1)} = {x : ℝ | x ≤ 0 ∨ x > 1} :=
by
  sorry

end solution_set_for_inequality_l192_192087


namespace derivative_at_2_l192_192535

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l192_192535


namespace simplify_radicals_l192_192952

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l192_192952


namespace sin_diff_l192_192445

variable (θ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5)

theorem sin_diff
  (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin θ = 2 * Real.sqrt 5 / 5) :
  Real.sin (θ - π / 4) = Real.sqrt 10 / 10 :=
sorry

end sin_diff_l192_192445


namespace pat_oj_consumption_l192_192239

def initial_oj : ℚ := 3 / 4
def alex_fraction : ℚ := 1 / 2
def pat_fraction : ℚ := 1 / 3

theorem pat_oj_consumption : pat_fraction * (initial_oj * (1 - alex_fraction)) = 1 / 8 := by
  -- This will be the proof part which can be filled later
  sorry

end pat_oj_consumption_l192_192239


namespace angela_spent_78_l192_192722

-- Definitions
def angela_initial_money : ℕ := 90
def angela_left_money : ℕ := 12
def angela_spent_money : ℕ := angela_initial_money - angela_left_money

-- Theorem statement
theorem angela_spent_78 : angela_spent_money = 78 := by
  -- Proof would go here, but it is not required.
  sorry

end angela_spent_78_l192_192722


namespace number_of_logs_in_stack_l192_192845

theorem number_of_logs_in_stack :
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  total_logs = 114 := by
{
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  sorry
}

end number_of_logs_in_stack_l192_192845


namespace scientific_notation_3050000_l192_192418

def scientific_notation (n : ℕ) : String :=
  "3.05 × 10^6"

theorem scientific_notation_3050000 :
  scientific_notation 3050000 = "3.05 × 10^6" :=
by
  sorry

end scientific_notation_3050000_l192_192418


namespace orthogonal_projection_magnitude_correct_l192_192022

-- Define the vectors a and b
def a : ℝ × ℝ := (real.sqrt 3, 1)
def b : ℝ × ℝ := (1, 0)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Define the orthogonal projection magnitude of a onto b
def orthogonal_projection_magnitude (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

-- The proof statement
theorem orthogonal_projection_magnitude_correct :
  orthogonal_projection_magnitude a b = real.sqrt 3 :=
by
  sorry

end orthogonal_projection_magnitude_correct_l192_192022


namespace closest_integer_to_cube_root_of_250_l192_192105

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l192_192105


namespace remaining_volume_is_21_l192_192287

-- Definitions of edge lengths and volumes
def edge_length_original : ℕ := 3
def edge_length_small : ℕ := 1
def volume (a : ℕ) : ℕ := a ^ 3

-- Volumes of the original cube and the small cubes
def volume_original : ℕ := volume edge_length_original
def volume_small : ℕ := volume edge_length_small
def number_of_faces : ℕ := 6
def total_volume_cut : ℕ := number_of_faces * volume_small

-- Volume of the remaining part
def volume_remaining : ℕ := volume_original - total_volume_cut

-- Proof statement
theorem remaining_volume_is_21 : volume_remaining = 21 := by
  sorry

end remaining_volume_is_21_l192_192287


namespace fraction_of_male_fish_l192_192302

def total_fish : ℕ := 45
def female_fish : ℕ := 15
def male_fish := total_fish - female_fish

theorem fraction_of_male_fish : (male_fish : ℚ) / total_fish = 2 / 3 := by
  sorry

end fraction_of_male_fish_l192_192302


namespace tomatoes_left_after_yesterday_correct_l192_192836

def farmer_initial_tomatoes := 160
def tomatoes_picked_yesterday := 56
def tomatoes_left_after_yesterday : ℕ := farmer_initial_tomatoes - tomatoes_picked_yesterday

theorem tomatoes_left_after_yesterday_correct :
  tomatoes_left_after_yesterday = 104 :=
by
  unfold tomatoes_left_after_yesterday
  -- Proof goes here
  sorry

end tomatoes_left_after_yesterday_correct_l192_192836


namespace hockey_league_games_l192_192813

theorem hockey_league_games (n : ℕ) (k : ℕ) (h_n : n = 25) (h_k : k = 15) : 
  (n * (n - 1) / 2) * k = 4500 := by
  sorry

end hockey_league_games_l192_192813


namespace p_x_range_l192_192029

variable (x : ℝ)

def inequality_condition := x^2 - 5*x + 6 < 0
def polynomial_function := x^2 + 5*x + 6

theorem p_x_range (x_ineq : inequality_condition x) : 
  20 < polynomial_function x ∧ polynomial_function x < 30 :=
sorry

end p_x_range_l192_192029


namespace solve_quadratic_l192_192256

theorem solve_quadratic (x : ℝ) : x^2 - 2*x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l192_192256


namespace mrs_hilt_initial_marbles_l192_192060

theorem mrs_hilt_initial_marbles (lost_marble : ℕ) (remaining_marble : ℕ) (h1 : lost_marble = 15) (h2 : remaining_marble = 23) : 
    (remaining_marble + lost_marble) = 38 :=
by
  sorry

end mrs_hilt_initial_marbles_l192_192060


namespace average_first_21_multiples_of_17_l192_192392

theorem average_first_21_multiples_of_17:
  let n := 21
  let a1 := 17
  let a21 := 17 * n
  let sum := n / 2 * (a1 + a21)
  (sum / n = 187) :=
by
  sorry

end average_first_21_multiples_of_17_l192_192392


namespace expected_value_is_correct_l192_192841

-- Define the probabilities of heads and tails
def P_H := 2 / 5
def P_T := 3 / 5

-- Define the winnings for heads and the loss for tails
def W_H := 5
def L_T := -4

-- Calculate the expected value
def expected_value := P_H * W_H + P_T * L_T

-- Prove that the expected value is -2/5
theorem expected_value_is_correct : expected_value = -2 / 5 := by
  sorry

end expected_value_is_correct_l192_192841


namespace items_per_friend_l192_192066

theorem items_per_friend (pencils : ℕ) (erasers : ℕ) (friends : ℕ) 
    (pencils_eq : pencils = 35) 
    (erasers_eq : erasers = 5) 
    (friends_eq : friends = 5) : 
    (pencils + erasers) / friends = 8 := 
by
  sorry

end items_per_friend_l192_192066


namespace min_value_geometric_sequence_l192_192915

theorem min_value_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h : 0 < q ∧ 0 < a 0) 
  (H : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) 
  (h_geom : ∀ n, a (n+1) = a n * q) : 
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end min_value_geometric_sequence_l192_192915


namespace sufficient_condition_l192_192758

theorem sufficient_condition (a b : ℝ) (h : a > b ∧ b > 0) : a + a^2 > b + b^2 :=
by
  sorry

end sufficient_condition_l192_192758


namespace closest_integer_to_cube_root_of_250_l192_192104

theorem closest_integer_to_cube_root_of_250 :
    ∃ k : ℤ, k = 6 ∧ (k^3 ≤ 250 ∧ 250 < (k + 1)^3) :=
by
  use 6
  split
  · sorry
  · split
    · sorry
    · sorry

end closest_integer_to_cube_root_of_250_l192_192104


namespace jeremy_oranges_l192_192934

theorem jeremy_oranges (M : ℕ) (h : M + 3 * M + 70 = 470) : M = 100 := 
by
  sorry

end jeremy_oranges_l192_192934


namespace oliver_final_amount_l192_192233

def initial_amount : ℤ := 33
def spent : ℤ := 4
def received : ℤ := 32

def final_amount (initial_amount spent received : ℤ) : ℤ :=
  initial_amount - spent + received

theorem oliver_final_amount : final_amount initial_amount spent received = 61 := 
by sorry

end oliver_final_amount_l192_192233


namespace problem_statement_l192_192180

noncomputable def general_term (a : ℕ → ℕ) (n : ℕ) : Prop :=
a n = n

noncomputable def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n, S n = (n * (n + 1)) / 2

noncomputable def b_def (S : ℕ → ℕ) (b : ℕ → ℚ) : Prop :=
∀ n, b n = (2 : ℚ) / (S n)

noncomputable def sum_b_first_n_terms (b : ℕ → ℚ) (T : ℕ → ℚ) : Prop :=
∀ n, T n = (4 * n) / (n + 1)

theorem problem_statement (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (∀ n, a n = 1 + (n - 1) * 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) - a n ≠ 0) →
  a 3 ^ 2 = a 1 * a 9 →
  general_term a 1 →
  sum_first_n_terms a S →
  b_def S b →
  sum_b_first_n_terms b T :=
by
  intro arithmetic_seq
  intro a_1_eq_1
  intro non_zero_diff
  intro geometric_seq
  intro gen_term_cond
  intro sum_terms_cond
  intro b_def_cond
  intro sum_b_terms_cond
  -- The proof goes here.
  sorry

end problem_statement_l192_192180


namespace sqrt_sum_simplify_l192_192945

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l192_192945


namespace sqrt_sum_simplify_l192_192943

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l192_192943


namespace isaac_ribbon_length_l192_192218

variable (part_length : ℝ) (total_length : ℝ := part_length * 6) (unused_length : ℝ := part_length * 2)

theorem isaac_ribbon_length
  (total_parts : ℕ := 6)
  (used_parts : ℕ := 4)
  (not_used_parts : ℕ := total_parts - used_parts)
  (not_used_length : Real := 10)
  (equal_parts : total_length / total_parts = part_length) :
  total_length = 30 := by
  sorry

end isaac_ribbon_length_l192_192218


namespace room_length_l192_192376

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) (length : ℝ)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 28875)
  (h_cost_per_sqm : cost_per_sqm = 1400)
  (h_length : length = total_cost / cost_per_sqm / width) :
  length = 5.5 := by
  sorry

end room_length_l192_192376


namespace inequality_proof_l192_192361

theorem inequality_proof (a b c d : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (a_geq_1 : 1 ≤ a) (b_geq_1 : 1 ≤ b) (c_geq_1 : 1 ≤ c)
  (abcd_eq_1 : a * b * c * d = 1)
  : 
  1 / (a^2 - a + 1)^2 + 1 / (b^2 - b + 1)^2 + 1 / (c^2 - c + 1)^2 + 1 / (d^2 - d + 1)^2 ≤ 4
  := sorry

end inequality_proof_l192_192361


namespace determine_function_l192_192423

noncomputable def functional_solution (f : ℝ → ℝ) : Prop := 
  ∃ (C₁ C₂ : ℝ), ∀ (x : ℝ), 0 < x → f x = C₁ * x + C₂ / x 

theorem determine_function (f : ℝ → ℝ) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (x + 1 / x) * f y = f (x * y) + f (y / x)) →
  functional_solution f :=
sorry

end determine_function_l192_192423


namespace power_equation_value_l192_192276

theorem power_equation_value (n : ℕ) (h : n = 20) : n ^ (n / 2) = 102400000000000000000 := by
  sorry

end power_equation_value_l192_192276


namespace sum_of_two_primes_l192_192456

theorem sum_of_two_primes (k : ℕ) (n : ℕ) (h : n = 1 + 10 * k) :
  (n = 1 ∨ ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ n = p1 + p2) :=
by
  sorry

end sum_of_two_primes_l192_192456


namespace tangent_identity_l192_192015

theorem tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2)
  = ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) :=
sorry

end tangent_identity_l192_192015


namespace part_a_part_b_l192_192053

-- Given distinct primes p and q
variables (p q : ℕ) [hp : Fact (Nat.Prime p)] [hq : Fact (Nat.Prime q)] (h : p ≠ q)

-- Prove p^q + q^p ≡ p + q (mod pq)
theorem part_a (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) :
  (p^q + q^p) % (p * q) = (p + q) % (p * q) := by
  sorry

-- Given distinct primes p and q, and neither are 2
theorem part_b (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q) (hp2 : p ≠ 2) (hq2 : q ≠ 2) :
  Even (Nat.floor ((p^q + q^p) / (p * q))) := by
  sorry

end part_a_part_b_l192_192053


namespace lucy_money_l192_192466

variable (L : ℕ) -- Value for Lucy's original amount of money

theorem lucy_money (h1 : ∀ (L : ℕ), L - 5 = 10 + 5 → L = 20) : L = 20 :=
by sorry

end lucy_money_l192_192466


namespace smallest_n_l192_192064

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n % 8 = 5) (h4 : n > 20) : n = 136 := by
  sorry

end smallest_n_l192_192064


namespace find_number_l192_192637

theorem find_number (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) (number : ℕ) :
  quotient = 9 ∧ remainder = 1 ∧ divisor = 30 → number = 271 := by
  intro h
  sorry

end find_number_l192_192637


namespace probability_abs_diff_less_than_one_third_l192_192645

def coin_flip_distribution : ProbabilityMeasure ℝ :=
  sorry -- This represents the probability distribution described in the problem

noncomputable def prob_diff_lt_one_third (x y : ℝ) : ℝ :=
  coin_flip_distribution.prob {z : ℝ × ℝ | abs (z.1 - z.2) < 1/3}

theorem probability_abs_diff_less_than_one_third :
  let x := coin_flip_distribution,
      y := coin_flip_distribution in
  prob_diff_lt_one_third x y = 1/8 :=
by
  sorry

end probability_abs_diff_less_than_one_third_l192_192645


namespace parallel_vectors_x_eq_one_l192_192191

/-- Given vectors a = (2x + 1, 3) and b = (2 - x, 1), prove that if they 
are parallel, then x = 1. -/
theorem parallel_vectors_x_eq_one (x : ℝ) :
  (∃ k : ℝ, (2 * x + 1) = k * (2 - x) ∧ 3 = k * 1) → x = 1 :=
by 
  sorry

end parallel_vectors_x_eq_one_l192_192191


namespace trees_in_yard_l192_192205

theorem trees_in_yard (L d : ℕ) (hL : L = 250) (hd : d = 5) : 
  (L / d + 1) = 51 := by
  sorry

end trees_in_yard_l192_192205


namespace count_five_digit_numbers_ending_in_6_divisible_by_3_l192_192025

theorem count_five_digit_numbers_ending_in_6_divisible_by_3 : 
  (∃ (n : ℕ), n = 3000 ∧
  ∀ (x : ℕ), (x ≥ 10000 ∧ x ≤ 99999) ∧ (x % 10 = 6) ∧ (x % 3 = 0) ↔ 
  (∃ (k : ℕ), x = 10026 + k * 30 ∧ k < 3000)) :=
by
  -- Proof is omitted
  sorry

end count_five_digit_numbers_ending_in_6_divisible_by_3_l192_192025


namespace janet_spending_difference_l192_192219

-- Defining hourly rates and weekly hours for each type of lessons
def clarinet_hourly_rate := 40
def clarinet_weekly_hours := 3
def piano_hourly_rate := 28
def piano_weekly_hours := 5
def violin_hourly_rate := 35
def violin_weekly_hours := 2
def singing_hourly_rate := 45
def singing_weekly_hours := 1

-- Calculating weekly costs
def clarinet_weekly_cost := clarinet_hourly_rate * clarinet_weekly_hours
def piano_weekly_cost := piano_hourly_rate * piano_weekly_hours
def violin_weekly_cost := violin_hourly_rate * violin_weekly_hours
def singing_weekly_cost := singing_hourly_rate * singing_weekly_hours
def combined_weekly_cost := piano_weekly_cost + violin_weekly_cost + singing_weekly_cost

-- Calculating annual costs with 52 weeks in a year
def weeks_per_year := 52
def clarinet_annual_cost := clarinet_weekly_cost * weeks_per_year
def combined_annual_cost := combined_weekly_cost * weeks_per_year

-- Proving the final statement
theorem janet_spending_difference :
  combined_annual_cost - clarinet_annual_cost = 7020 := by sorry

end janet_spending_difference_l192_192219


namespace symmetric_circle_l192_192662

variable (x y : ℝ)

def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem symmetric_circle :
  (∃ x y, original_circle x y) → (x^2 + (y + 2)^2 = 5) :=
sorry

end symmetric_circle_l192_192662


namespace find_A_B_l192_192882

theorem find_A_B (A B : ℝ) (h : ∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 → 
  (A / (x - 5) + B / (x + 2) = (5 * x - 4) / (x^2 - 3 * x - 10))) :
  A = 3 ∧ B = 2 :=
sorry

end find_A_B_l192_192882


namespace percentage_of_cars_with_no_features_l192_192091

theorem percentage_of_cars_with_no_features (N S W R SW SR WR SWR : ℕ)
  (hN : N = 120)
  (hS : S = 70)
  (hW : W = 40)
  (hR : R = 30)
  (hSW : SW = 20)
  (hSR : SR = 15)
  (hWR : WR = 10)
  (hSWR : SWR = 5) :
  (120 - (S + W + R - SW - SR - WR + SWR)) / (N : ℝ) * 100 = 16.67 :=
by
  sorry

end percentage_of_cars_with_no_features_l192_192091


namespace pq_r_zero_l192_192359

theorem pq_r_zero (p q r : ℝ) : 
  (∀ x : ℝ, x^4 + 6 * x^3 + 4 * p * x^2 + 2 * q * x + r = (x^3 + 4 * x^2 + 2 * x + 1) * (x - 2)) → 
  (p + q) * r = 0 :=
by
  sorry

end pq_r_zero_l192_192359


namespace melony_profit_l192_192231

theorem melony_profit (profit_3_shirts : ℝ)
  (profit_2_sandals : ℝ)
  (h1 : profit_3_shirts = 21)
  (h2 : profit_2_sandals = 4 * 21) : profit_3_shirts / 3 * 7 + profit_2_sandals / 2 * 3 = 175 := 
by 
  sorry

end melony_profit_l192_192231


namespace factorization_correct_l192_192310

variable (a b : ℝ)

theorem factorization_correct :
  12 * a ^ 3 * b - 12 * a ^ 2 * b + 3 * a * b = 3 * a * b * (2 * a - 1) ^ 2 :=
by 
  sorry

end factorization_correct_l192_192310


namespace chess_tournament_participants_l192_192206

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 136) : n = 17 :=
by {
  sorry -- Proof will be here.
}

end chess_tournament_participants_l192_192206


namespace simplify_expression_l192_192785

theorem simplify_expression (x : ℝ) : 
  (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 :=
by
  sorry

end simplify_expression_l192_192785


namespace solve_sqrt_equation_l192_192313

noncomputable def f (x : ℝ) : ℝ :=
  real.cbrt (3 - x) + real.sqrt (x - 2)

theorem solve_sqrt_equation :
  { x : ℝ | f x = 1 ∧ 2 ≤ x } = {2, 3, 11} := by
  sorry

end solve_sqrt_equation_l192_192313


namespace factor_poly_PQ_sum_l192_192461

theorem factor_poly_PQ_sum (P Q : ℝ) (h : (∀ x : ℝ, (x^2 + 3 * x + 4) * (x^2 + -3 * x + 4) = x^4 + P * x^2 + Q)) : P + Q = 15 :=
by
  sorry

end factor_poly_PQ_sum_l192_192461


namespace letters_by_30_typists_in_1_hour_l192_192198

-- Definitions from the conditions
def lettersTypedByOneTypistIn20Minutes := 44 / 20

def lettersTypedBy30TypistsIn20Minutes := 30 * (lettersTypedByOneTypistIn20Minutes)

def conversionToHours := 3

-- Theorem statement
theorem letters_by_30_typists_in_1_hour : lettersTypedBy30TypistsIn20Minutes * conversionToHours = 198 := by
  sorry

end letters_by_30_typists_in_1_hour_l192_192198


namespace man_total_pay_l192_192293

def regular_rate : ℕ := 3
def regular_hours : ℕ := 40
def overtime_hours : ℕ := 13

def regular_pay : ℕ := regular_rate * regular_hours
def overtime_rate : ℕ := 2 * regular_rate
def overtime_pay : ℕ := overtime_rate * overtime_hours

def total_pay : ℕ := regular_pay + overtime_pay

theorem man_total_pay : total_pay = 198 := by
  sorry

end man_total_pay_l192_192293


namespace find_divisor_l192_192935

theorem find_divisor (N D k : ℤ) (h1 : N = 5 * D) (h2 : N % 11 = 2) : D = 7 :=
by
  sorry

end find_divisor_l192_192935


namespace fraction_value_l192_192320

variable (a b : ℚ)  -- Variables a and b are rational numbers

theorem fraction_value (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_value_l192_192320


namespace total_games_in_season_l192_192811

-- Define the constants according to the conditions
def number_of_teams : ℕ := 25
def games_per_pair : ℕ := 15

-- Define the mathematical statement we want to prove
theorem total_games_in_season :
  let round_robin_games := (number_of_teams * (number_of_teams - 1)) / 2 in
  let total_games := round_robin_games * games_per_pair in
  total_games = 4500 :=
by
  sorry

end total_games_in_season_l192_192811


namespace meters_to_centimeters_l192_192377

theorem meters_to_centimeters : (3.5 : ℝ) * 100 = 350 :=
by
  sorry

end meters_to_centimeters_l192_192377


namespace third_term_arithmetic_sequence_l192_192090

variable (a d : ℤ)
variable (h1 : a + 20 * d = 12)
variable (h2 : a + 21 * d = 15)

theorem third_term_arithmetic_sequence : a + 2 * d = -42 := by
  sorry

end third_term_arithmetic_sequence_l192_192090


namespace tom_payment_l192_192095

theorem tom_payment :
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  total_amount = 1190 :=
by
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  sorry

end tom_payment_l192_192095


namespace Iggy_Tuesday_Run_l192_192623

def IggyRunsOnTuesday (total_miles : ℕ) (monday_miles : ℕ) (wednesday_miles : ℕ) (thursday_miles : ℕ) (friday_miles : ℕ) : ℕ :=
  total_miles - (monday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem Iggy_Tuesday_Run :
  let monday_miles := 3
  let wednesday_miles := 6
  let thursday_miles := 8
  let friday_miles := 3
  let total_miles := 240 / 10
  IggyRunsOnTuesday total_miles monday_miles wednesday_miles thursday_miles friday_miles = 4 :=
by
  sorry

end Iggy_Tuesday_Run_l192_192623


namespace rationalize_denominator_simplify_l192_192371

theorem rationalize_denominator_simplify :
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := 1
  let d : ℝ := 2
  ∀ (x y z : ℝ), 
  (x = 3 * Real.sqrt 2) → 
  (y = 3) → 
  (z = Real.sqrt 3) → 
  (x / (y - z) = (3 * Real.sqrt 2 + Real.sqrt 6) / 2) :=
by
  sorry

end rationalize_denominator_simplify_l192_192371


namespace problem_ratio_l192_192567

-- Define the conditions
variables 
  (R : ℕ) 
  (Bill_problems : ℕ := 20) 
  (Frank_problems_per_type : ℕ := 30)
  (types : ℕ := 4)

-- State the problem to prove
theorem problem_ratio (h1 : 3 * R = Frank_problems_per_type * types) :
  R / Bill_problems = 2 :=
by
  -- placeholder for proof
  sorry

end problem_ratio_l192_192567


namespace area_of_shape_is_correct_l192_192298

noncomputable def square_side_length : ℝ := 2 * Real.pi

noncomputable def semicircle_radius : ℝ := square_side_length / 2

noncomputable def area_of_resulting_shape : ℝ :=
  let area_square := square_side_length^2
  let area_semicircle := (1/2) * Real.pi * semicircle_radius^2
  let total_area := area_square + 4 * area_semicircle
  total_area

theorem area_of_shape_is_correct :
  area_of_resulting_shape = 2 * Real.pi^2 * (Real.pi + 2) :=
sorry

end area_of_shape_is_correct_l192_192298


namespace factorization_problem1_factorization_problem2_l192_192004

variables {a b x y : ℝ}

theorem factorization_problem1 (a b x y : ℝ) : a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) :=
by sorry

theorem factorization_problem2 (a b : ℝ) : a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 :=
by sorry

end factorization_problem1_factorization_problem2_l192_192004


namespace count_lucky_numbers_l192_192097

-- Definitions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_sum_to_six (n : ℕ) : Prop := (n / 100 + (n / 10) % 10 + n % 10 = 6)

-- Proposition to prove
theorem count_lucky_numbers : {n : ℕ | is_three_digit_number n ∧ digits_sum_to_six n}.to_finset.card = 21 := 
by 
  sorry

end count_lucky_numbers_l192_192097


namespace find_f_prime_at_two_l192_192532

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l192_192532


namespace lilith_caps_collection_l192_192779

theorem lilith_caps_collection :
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * 4
  let christmas_caps := 40 * 5
  let lost_caps := 15 * 5
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps - lost_caps
  total_caps = 401 := by
  sorry

end lilith_caps_collection_l192_192779


namespace cubic_root_sqrt_equation_l192_192312

theorem cubic_root_sqrt_equation (x : ℝ) (h1 : 3 - x = y^3) (h2 : x - 2 = z^2) (h3 : y + z = 1) : 
  x = 3 ∨ x = 2 ∨ x = 11 :=
sorry

end cubic_root_sqrt_equation_l192_192312


namespace heavy_equipment_pay_l192_192301

theorem heavy_equipment_pay
  (total_workers : ℕ)
  (total_payroll : ℕ)
  (laborers : ℕ)
  (laborer_pay : ℕ)
  (heavy_operator_pay : ℕ)
  (h1 : total_workers = 35)
  (h2 : total_payroll = 3950)
  (h3 : laborers = 19)
  (h4 : laborer_pay = 90)
  (h5 : (total_workers - laborers) * heavy_operator_pay + laborers * laborer_pay = total_payroll) :
  heavy_operator_pay = 140 :=
by
  sorry

end heavy_equipment_pay_l192_192301


namespace line_CD_area_triangle_equality_line_CD_midpoint_l192_192629

theorem line_CD_area_triangle_equality :
  ∃ k : ℝ, 4 * k - 1 = 1 - k := sorry

theorem line_CD_midpoint :
  ∃ k : ℝ, 9 * k - 2 = 1 := sorry

end line_CD_area_triangle_equality_line_CD_midpoint_l192_192629


namespace sum_abs_diff_ge_two_n_minus_two_l192_192496

open Int

theorem sum_abs_diff_ge_two_n_minus_two (n : ℕ) (h : n ≥ 2) (circle: Fin n → ℕ) (h_circle : ∀ i, circle i ∈ Finset.range (n + 1) ∧ (Finset.range (n + 1) \ Finset.singleton 0).card = n ∧ (∑' i, circle i) = (∑ i in Finset.range (n+1), i)) :

  (∑ i in Finset.range n, Int.natAbs ((circle i) - (circle ((i + 1) % n)))) ≥ 2 * n - 2 := 
begin
  sorry
end

end sum_abs_diff_ge_two_n_minus_two_l192_192496


namespace range_of_m_non_perpendicular_tangent_l192_192612

noncomputable def f (m x : ℝ) : ℝ := Real.exp x - m * x

theorem range_of_m_non_perpendicular_tangent (m : ℝ) :
  (∀ x : ℝ, (deriv (f m) x ≠ -2)) → m ≤ 2 :=
by
  sorry

end range_of_m_non_perpendicular_tangent_l192_192612


namespace intersection_P_Q_l192_192634

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the problem statement as a theorem
theorem intersection_P_Q : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_P_Q_l192_192634


namespace cody_final_tickets_l192_192862

def initial_tickets : ℝ := 56.5
def lost_tickets : ℝ := 6.3
def spent_tickets : ℝ := 25.75
def won_tickets : ℝ := 10.25
def dropped_tickets : ℝ := 3.1

theorem cody_final_tickets : 
  initial_tickets - lost_tickets - spent_tickets + won_tickets - dropped_tickets = 31.6 :=
by
  sorry

end cody_final_tickets_l192_192862


namespace last_digit_inverse_power_two_l192_192984

theorem last_digit_inverse_power_two :
  let n := 12
  let x := 5^n
  let y := 10^n
  (x % 10 = 5) →
  ((1 / (2^n)) * (5^n) / (5^n) == (5^n) / (10^n)) →
  (y % 10 = 0) →
  ((1 / (2^n)) % 10 = 5) :=
by
  intros n x y h1 h2 h3
  sorry

end last_digit_inverse_power_two_l192_192984


namespace min_units_l192_192835

theorem min_units (x : ℕ) (h1 : 5500 * 60 + 5000 * (x - 60) > 550000) : x ≥ 105 := 
by {
  sorry
}

end min_units_l192_192835


namespace intersection_A_B_l192_192903

theorem intersection_A_B :
  let A := {1, 3, 5, 7}
  let B := {x | x^2 - 2 * x - 5 ≤ 0}
  A ∩ B = {1, 3} := by
sorry

end intersection_A_B_l192_192903


namespace cherries_initially_l192_192234

theorem cherries_initially (x : ℕ) (h₁ : x - 6 = 10) : x = 16 :=
by
  sorry

end cherries_initially_l192_192234


namespace daisy_germination_rate_theorem_l192_192754

-- Define the conditions of the problem
variables (daisySeeds sunflowerSeeds : ℕ) (sunflowerGermination flowerProduction finalFlowerPlants : ℝ)
def conditions : Prop :=
  daisySeeds = 25 ∧ sunflowerSeeds = 25 ∧ sunflowerGermination = 0.80 ∧ flowerProduction = 0.80 ∧ finalFlowerPlants = 28

-- Define the statement that the germination rate of the daisy seeds is 60%
def germination_rate_of_daisy_seeds : Prop :=
  ∃ (daisyGerminationRate : ℝ), (conditions daisySeeds sunflowerSeeds sunflowerGermination flowerProduction finalFlowerPlants) →
  daisyGerminationRate = 0.60

-- The proof is omitted - note this is just the statement
theorem daisy_germination_rate_theorem : germination_rate_of_daisy_seeds 25 25 0.80 0.80 28 :=
sorry

end daisy_germination_rate_theorem_l192_192754


namespace find_f_prime_at_two_l192_192533

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l192_192533


namespace men_with_ac_at_least_12_l192_192917

-- Define the variables and conditions
variable (total_men : ℕ) (married_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (men_with_all_four : ℕ)

-- Assume the given conditions
axiom h1 : total_men = 100
axiom h2 : married_men = 82
axiom h3 : tv_men = 75
axiom h4 : radio_men = 85
axiom h5 : men_with_all_four = 12

-- Define the number of men with AC
variable (ac_men : ℕ)

-- State the proposition that the number of men with AC is at least 12
theorem men_with_ac_at_least_12 : ac_men ≥ 12 := sorry

end men_with_ac_at_least_12_l192_192917


namespace melissa_points_per_game_l192_192636

variable (t g p : ℕ)

theorem melissa_points_per_game (ht : t = 36) (hg : g = 3) : p = t / g → p = 12 :=
by
  intro h
  sorry

end melissa_points_per_game_l192_192636


namespace slow_train_speed_l192_192251

/-- Given the conditions of two trains traveling towards each other and their meeting times,
     prove the speed of the slow train. -/
theorem slow_train_speed :
  let distance_AB := 901
  let slow_train_departure := 5 + 30 / 60 -- 5:30 AM in decimal hours
  let fast_train_departure := 9 + 30 / 60 -- 9:30 AM in decimal hours
  let meeting_time := 16 + 30 / 60 -- 4:30 PM in decimal hours
  let fast_train_speed := 58 -- speed in km/h
  let slow_train_time := meeting_time - slow_train_departure
  let fast_train_time := meeting_time - fast_train_departure
  let fast_train_distance := fast_train_speed * fast_train_time
  let slow_train_distance := distance_AB - fast_train_distance
  let slow_train_speed := slow_train_distance / slow_train_time
  slow_train_speed = 45 := sorry

end slow_train_speed_l192_192251


namespace amc12a_2006_p24_l192_192051

theorem amc12a_2006_p24 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
by
  sorry

end amc12a_2006_p24_l192_192051


namespace closest_integer_to_cube_root_of_250_l192_192109

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l192_192109


namespace avg_amount_lost_per_loot_box_l192_192632

-- Define the conditions
def cost_per_loot_box : ℝ := 5
def avg_value_of_items : ℝ := 3.5
def total_amount_spent : ℝ := 40

-- Define the goal
theorem avg_amount_lost_per_loot_box : 
  (total_amount_spent / cost_per_loot_box) * (cost_per_loot_box - avg_value_of_items) / (total_amount_spent / cost_per_loot_box) = 1.5 := 
by 
  sorry

end avg_amount_lost_per_loot_box_l192_192632


namespace cats_remaining_l192_192843

theorem cats_remaining 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 := 
by
  sorry

end cats_remaining_l192_192843


namespace steven_more_peaches_l192_192043

variable (Jake Steven Jill : ℕ)

-- Conditions
axiom h1 : Jake + 6 = Steven
axiom h2 : Jill = 5
axiom h3 : Jake = 17

-- Goal
theorem steven_more_peaches : Steven - Jill = 18 := by
  sorry

end steven_more_peaches_l192_192043


namespace sum_of_reciprocals_l192_192675

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) : 
  (1 / x) + (1 / y) = 8 := 
by 
  sorry

end sum_of_reciprocals_l192_192675


namespace non_working_games_count_l192_192232

-- Definitions based on conditions
def totalGames : ℕ := 16
def pricePerGame : ℕ := 7
def totalEarnings : ℕ := 56

-- Statement to prove
theorem non_working_games_count : 
  totalGames - (totalEarnings / pricePerGame) = 8 :=
by
  sorry

end non_working_games_count_l192_192232


namespace equation_of_ellipse_area_of_OAPB_l192_192183

variables {a b x y : ℝ} {A B M P : Point}

def ellipse_eq (a b x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def midpoint (A B M : Point) : Prop := M = Point.midpoint A B

def slopes_product (A B : Point) (l OM : Line) : Prop := 
let k := (A.y - B.y) / (A.x - B.x) in
let k_OM := (M.y / M.x) in
k * k_OM = -1/4

theorem equation_of_ellipse (a b : ℝ) (A B M P : Point) (l OM : Line) (C := {p : Point | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hC : ellipse_eq a b x y) (hM : midpoint A B M) (hP : extension_OM A B P) (hS : slopes_product A B l OM)
  (h_major_axis : 2 * a = 4) :
  ∃ a b, C = {p : Point | p.1^2 / 4 + p.2^2 = 1} :=
sorry

theorem area_of_OAPB (a b : ℝ) (A B M P : Point) (l : Line) (C := {p : Point | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hC : ellipse_eq a b x y) (hM : midpoint A B M) (hP : extension_OM A B P) (hS : slopes_product A B l OM)
  (h_parallel : parallelogram O A P B) :
  area O A P B = (sqrt 3 / 2) * a * b :=
sorry

end equation_of_ellipse_area_of_OAPB_l192_192183


namespace largest_consecutive_multiple_l192_192673

theorem largest_consecutive_multiple (n : ℕ) (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 117) : 3 * (n + 2) = 42 :=
sorry

end largest_consecutive_multiple_l192_192673


namespace part_a_part_b_l192_192282

namespace ShaltaevBoltaev

variables {s b : ℕ}

-- Condition: 175s > 125b
def condition1 (s b : ℕ) : Prop := 175 * s > 125 * b

-- Condition: 175s < 126b
def condition2 (s b : ℕ) : Prop := 175 * s < 126 * b

-- Prove that 3s + b > 80
theorem part_a (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 80 := sorry

-- Prove that 3s + b > 100
theorem part_b (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 100 := sorry

end ShaltaevBoltaev

end part_a_part_b_l192_192282


namespace closest_integer_to_cube_root_of_250_l192_192108

theorem closest_integer_to_cube_root_of_250 :
  (abs (6 - real.cbrt 250) < abs (7 - real.cbrt 250)) ∧
  (abs (6 - real.cbrt 250) <= abs (5 - real.cbrt 250)) :=
by
  sorry

end closest_integer_to_cube_root_of_250_l192_192108


namespace lilith_caps_collection_l192_192777

noncomputable def monthlyCollectionYear1 := 3
noncomputable def monthlyCollectionAfterYear1 := 5
noncomputable def christmasCaps := 40
noncomputable def yearlyCapsLost := 15
noncomputable def totalYears := 5

noncomputable def totalCapsCollectedByLilith :=
  let firstYearCaps := monthlyCollectionYear1 * 12
  let remainingYearsCaps := monthlyCollectionAfterYear1 * 12 * (totalYears - 1)
  let christmasCapsTotal := christmasCaps * totalYears
  let totalCapsBeforeLosses := firstYearCaps + remainingYearsCaps + christmasCapsTotal
  let lostCapsTotal := yearlyCapsLost * totalYears
  let totalCapsAfterLosses := totalCapsBeforeLosses - lostCapsTotal
  totalCapsAfterLosses

theorem lilith_caps_collection : totalCapsCollectedByLilith = 401 := by
  sorry

end lilith_caps_collection_l192_192777


namespace first_car_gas_consumed_l192_192626

theorem first_car_gas_consumed 
    (sum_avg_mpg : ℝ) (g2_gallons : ℝ) (total_miles : ℝ) 
    (avg_mpg_car1 : ℝ) (avg_mpg_car2 : ℝ) (g1_gallons : ℝ) :
    sum_avg_mpg = avg_mpg_car1 + avg_mpg_car2 →
    g2_gallons = 35 →
    total_miles = 2275 →
    avg_mpg_car1 = 40 →
    avg_mpg_car2 = 35 →
    g1_gallons = (total_miles - (avg_mpg_car2 * g2_gallons)) / avg_mpg_car1 →
    g1_gallons = 26.25 :=
by
  intros h_sum_avg_mpg h_g2_gallons h_total_miles h_avg_mpg_car1 h_avg_mpg_car2 h_g1_gallons
  sorry

end first_car_gas_consumed_l192_192626


namespace solve_for_n_l192_192067

theorem solve_for_n (n : ℕ) : (8 ^ n) * (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 4 → n = 2 :=
by 
  intro h
  sorry

end solve_for_n_l192_192067


namespace smallest_n_for_divisibility_l192_192430

theorem smallest_n_for_divisibility (n : ℕ) : 
  (∀ m, m > 0 → (315^2 - m^2) ∣ (315^3 - m^3) → m ≥ n) → 
  (315^2 - n^2) ∣ (315^3 - n^3) → 
  n = 90 :=
by
  sorry

end smallest_n_for_divisibility_l192_192430


namespace smallest_three_digit_divisible_l192_192432

theorem smallest_three_digit_divisible :
  ∃ (A B C : Nat), A ≠ 0 ∧ 100 ≤ (100 * A + 10 * B + C) ∧ (100 * A + 10 * B + C) < 1000 ∧
  (10 * A + B) > 9 ∧ (10 * B + C) > 9 ∧ 
  (100 * A + 10 * B + C) % (10 * A + B) = 0 ∧ (100 * A + 10 * B + C) % (10 * B + C) = 0 ∧
  (100 * A + 10 * B + C) = 110 :=
by
  sorry

end smallest_three_digit_divisible_l192_192432


namespace functional_equation_solution_l192_192311

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 ≠ 0) ∧ (∀ x y : ℝ, f (x + y) * f (x + y) = 2 * f x * f y + max (f (x * x) + f (y * y)) (f (x * x + y * y)))

theorem functional_equation_solution (f : ℝ → ℝ) :
  satisfies_conditions f → (∀ x : ℝ, f x = -1 ∨ f x = x - 1) :=
by
  intros h
  sorry

end functional_equation_solution_l192_192311


namespace point_A_in_Quadrant_IV_l192_192641

-- Define the coordinates of point A
def A : ℝ × ℝ := (5, -4)

-- Define the quadrants based on x and y signs
def in_Quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_Quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_Quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_Quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Statement to prove that point A lies in Quadrant IV
theorem point_A_in_Quadrant_IV : in_Quadrant_IV A :=
by
  sorry

end point_A_in_Quadrant_IV_l192_192641


namespace initial_roses_in_vase_l192_192678

theorem initial_roses_in_vase (added_roses current_roses : ℕ) (h1 : added_roses = 8) (h2 : current_roses = 18) : 
  current_roses - added_roses = 10 :=
by
  sorry

end initial_roses_in_vase_l192_192678


namespace solve_y_l192_192968

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l192_192968


namespace correct_answer_is_B_l192_192665

def lack_of_eco_friendly_habits : Prop := true
def major_global_climate_change_cause (s : String) : Prop :=
  s = "cause"

theorem correct_answer_is_B :
  major_global_climate_change_cause "cause" ∧ lack_of_eco_friendly_habits → "B" = "cause" :=
by
  sorry

end correct_answer_is_B_l192_192665


namespace solution_set_f_l192_192448

noncomputable def f (x : ℝ) : ℝ := sorry -- The differentiable function f

axiom f_deriv_lt (x : ℝ) : deriv f x < x -- Condition on the derivative of f
axiom f_at_2 : f 2 = 1 -- Given f(2) = 1

theorem solution_set_f : ∀ x : ℝ, f x < (1 / 2) * x^2 - 1 ↔ x > 2 :=
by sorry

end solution_set_f_l192_192448


namespace match_graph_l192_192543

theorem match_graph (x : ℝ) (h : x ≤ 0) : 
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by
  sorry

end match_graph_l192_192543


namespace area_of_ABCD_l192_192942

noncomputable def AB := 6
noncomputable def BC := 8
noncomputable def CD := 15
noncomputable def DA := 17
def right_angle_BCD := true
def convex_ABCD := true

theorem area_of_ABCD : ∃ area : ℝ, area = 110 := by
  -- Given conditions
  have hAB : AB = 6 := rfl
  have hBC : BC = 8 := rfl
  have hCD : CD = 15 := rfl
  have hDA : DA = 17 := rfl
  have hAngle : right_angle_BCD = true := rfl
  have hConvex : convex_ABCD = true := rfl

  -- skip the proof
  sorry

end area_of_ABCD_l192_192942


namespace combined_platforms_length_is_correct_l192_192521

noncomputable def combined_length_of_platforms (lengthA lengthB speedA_kmph speedB_kmph timeA_sec timeB_sec : ℝ) : ℝ :=
  let speedA := speedA_kmph * (1000 / 3600)
  let speedB := speedB_kmph * (1000 / 3600)
  let distanceA := speedA * timeA_sec
  let distanceB := speedB * timeB_sec
  let platformA := distanceA - lengthA
  let platformB := distanceB - lengthB
  platformA + platformB

theorem combined_platforms_length_is_correct :
  combined_length_of_platforms 650 450 115 108 30 25 = 608.32 := 
by 
  sorry

end combined_platforms_length_is_correct_l192_192521


namespace sum_of_final_two_numbers_l192_192385

theorem sum_of_final_two_numbers (a b S : ℝ) (h : a + b = S) : 
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_final_two_numbers_l192_192385


namespace green_square_area_percentage_l192_192563

variable (s a : ℝ)
variable (h : a^2 + 4 * a * (s - 2 * a) = 0.49 * s^2)

theorem green_square_area_percentage :
  (a^2 / s^2) = 0.1225 :=
sorry

end green_square_area_percentage_l192_192563


namespace intersection_complement_eq_l192_192455

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 })
variable (B : Set ℝ := { x | x > -1 })

theorem intersection_complement_eq :
  A ∩ (U \ B) = { x | -2 ≤ x ∧ x ≤ -1 } :=
by {
  sorry
}

end intersection_complement_eq_l192_192455


namespace max_possible_number_under_operations_l192_192468

theorem max_possible_number_under_operations :
  ∀ x : ℕ, x < 17 →
    ∀ n : ℕ, (∃ k : ℕ, k < n ∧ (x + 17 * k) % 19 = 0) →
    ∃ m : ℕ, m = (304 : ℕ) :=
sorry

end max_possible_number_under_operations_l192_192468


namespace cats_not_eating_cheese_or_tuna_l192_192344

-- Define the given conditions
variables (n C T B : ℕ)

-- State the problem in Lean
theorem cats_not_eating_cheese_or_tuna 
  (h_n : n = 100)  
  (h_C : C = 25)  
  (h_T : T = 70)  
  (h_B : B = 15)
  : n - (C - B + T - B + B) = 20 := 
by {
  -- Insert proof here
  sorry
}

end cats_not_eating_cheese_or_tuna_l192_192344


namespace parallel_segment_length_l192_192624

/-- In \( \triangle ABC \), given side lengths AB = 500, BC = 550, and AC = 650,
there exists an interior point P such that each segment drawn parallel to the
sides of the triangle and passing through P splits the sides into segments proportional
to the overall sides of the triangle. Prove that the length \( d \) of each segment
parallel to the sides is 28.25 -/
theorem parallel_segment_length
  (A B C P : Type)
  (d AB BC AC : ℝ)
  (ha : AB = 500)
  (hb : BC = 550)
  (hc : AC = 650)
  (hp : AB * BC = AC * 550) -- This condition ensures proportionality of segments
  : d = 28.25 :=
sorry

end parallel_segment_length_l192_192624


namespace relationship_a_b_l192_192449

theorem relationship_a_b
  (m a b : ℝ)
  (h1 : ∃ m, ∀ x, -2 * x + m = y)
  (h2 : ∃ x₁ y₁, (x₁ = -2) ∧ (y₁ = a) ∧ (-2 * x₁ + m = y₁))
  (h3 : ∃ x₂ y₂, (x₂ = 2) ∧ (y₂ = b) ∧ (-2 * x₂ + m = y₂)) :
  a > b :=
sorry

end relationship_a_b_l192_192449


namespace find_three_digit_number_l192_192737

/-- 
  Define the three-digit number abc and show that for some digit d in the range of 1 to 9,
  the conditions are satisfied.
-/
theorem find_three_digit_number
  (a b c d : ℕ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : 1 ≤ d ∧ d ≤ 9)
  (h_abc : 100 * a + 10 * b + c = 627)
  (h_bcd : 100 * b + 10 * c + d = 627 * a)
  (h_1a4d : 1040 + 100 * a + d = 627 * a)
  : 100 * a + 10 * b + c = 627 := 
sorry

end find_three_digit_number_l192_192737


namespace log_equality_l192_192152

theorem log_equality : 2 * log 5 10 + log 5 0.25 = 2 := 
sorry

end log_equality_l192_192152


namespace decreasing_direct_proportion_l192_192018

theorem decreasing_direct_proportion (k : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 > k * x2) : k < 0 :=
by
  sorry

end decreasing_direct_proportion_l192_192018


namespace total_bill_cost_l192_192652

-- Definitions of costs and conditions
def curtis_meal_cost : ℝ := 16.00
def rob_meal_cost : ℝ := 18.00
def total_cost_before_discount : ℝ := curtis_meal_cost + rob_meal_cost
def discount_rate : ℝ := 0.5
def time_of_meal : ℝ := 3.0

-- Condition for discount applicability
def discount_applicable : Prop := 2.0 ≤ time_of_meal ∧ time_of_meal ≤ 4.0

-- Total cost with discount applied
def cost_with_discount (total_cost : ℝ) (rate : ℝ) : ℝ := total_cost * rate

-- Theorem statement we need to prove
theorem total_bill_cost :
  discount_applicable →
  cost_with_discount total_cost_before_discount discount_rate = 17.00 :=
by
  sorry

end total_bill_cost_l192_192652


namespace problem_eqn_l192_192033

theorem problem_eqn (a b c : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁^2 + 3 * r₁ - 1 = 0 ∧ r₂^2 + 3 * r₂ - 1 = 0) ∧
  (∀ x : ℝ, (x^2 + 3 * x - 1 = 0) → (x^4 + a * x^2 + b * x + c = 0)) →
  a + b + 4 * c = -7 :=
by
  sorry

end problem_eqn_l192_192033


namespace solve_quadratic_equation_l192_192255

theorem solve_quadratic_equation (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end solve_quadratic_equation_l192_192255


namespace yellow_paint_percentage_l192_192316

theorem yellow_paint_percentage 
  (total_gallons_mixture : ℝ)
  (light_green_paint_gallons : ℝ)
  (dark_green_paint_gallons : ℝ)
  (dark_green_paint_percentage : ℝ)
  (mixture_percentage : ℝ)
  (X : ℝ) 
  (h_total_gallons : total_gallons_mixture = light_green_paint_gallons + dark_green_paint_gallons)
  (h_dark_green_paint_yellow_amount : dark_green_paint_gallons * dark_green_paint_percentage = 1.66666666667 * 0.4)
  (h_mixture_yellow_amount : total_gallons_mixture * mixture_percentage = 5 * X + 1.66666666667 * 0.4) :
  X = 0.2 :=
by
  sorry

end yellow_paint_percentage_l192_192316


namespace Tile_in_rectangle_R_l192_192682

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def X : Tile := ⟨5, 3, 6, 2⟩
def Y : Tile := ⟨3, 6, 2, 5⟩
def Z : Tile := ⟨6, 0, 1, 5⟩
def W : Tile := ⟨2, 5, 3, 0⟩

theorem Tile_in_rectangle_R : 
  X.top = 5 ∧ X.right = 3 ∧ X.bottom = 6 ∧ X.left = 2 ∧ 
  Y.top = 3 ∧ Y.right = 6 ∧ Y.bottom = 2 ∧ Y.left = 5 ∧ 
  Z.top = 6 ∧ Z.right = 0 ∧ Z.bottom = 1 ∧ Z.left = 5 ∧ 
  W.top = 2 ∧ W.right = 5 ∧ W.bottom = 3 ∧ W.left = 0 → 
  (∀ rectangle_R : Tile, rectangle_R = W) :=
by sorry

end Tile_in_rectangle_R_l192_192682


namespace find_a1_l192_192674

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

end find_a1_l192_192674


namespace one_eighth_of_2_pow_44_eq_2_pow_x_l192_192030

theorem one_eighth_of_2_pow_44_eq_2_pow_x (x : ℕ) :
  (2^44 / 8 = 2^x) → x = 41 :=
by
  sorry

end one_eighth_of_2_pow_44_eq_2_pow_x_l192_192030


namespace sqrt_sum_simplify_l192_192947

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l192_192947


namespace sqrt_72_plus_sqrt_32_l192_192966

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l192_192966


namespace solve_quadratic_l192_192257

theorem solve_quadratic (x : ℝ) : x^2 - 2*x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l192_192257


namespace earnings_proof_l192_192692

theorem earnings_proof (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 300) (h3 : C = 100) : A + C = 400 :=
sorry

end earnings_proof_l192_192692


namespace sqrt_three_squared_l192_192824

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end sqrt_three_squared_l192_192824


namespace choose_most_suitable_l192_192990

def Survey := ℕ → Bool
structure Surveys :=
  (A B C D : Survey)
  (census_suitable : Survey)

theorem choose_most_suitable (s : Surveys) :
  s.census_suitable = s.C :=
sorry

end choose_most_suitable_l192_192990


namespace part1_part2_l192_192329

noncomputable def setA : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }
noncomputable def setB (m : ℝ) : Set ℝ := { x | m - 1 < x ∧ x < 2*m + 1 }

theorem part1 (x : ℝ) : 
  setA ∪ setB 3 = { x | -1 ≤ x ∧ x < 7 } :=
sorry

theorem part2 (m : ℝ) : 
  (∀ x, x ∈ setA → x ∈ setB m) ∧ ¬(∃ x, x ∈ setB m ∧ x ∉ setA) ↔ 
  m ≤ -2 ∨ (0 ≤ m ∧ m ≤ 1) :=
sorry

end part1_part2_l192_192329


namespace hanging_spheres_ratio_l192_192429

theorem hanging_spheres_ratio (m1 m2 g T_B T_H : ℝ)
  (h1 : T_B = 3 * T_H)
  (h2 : T_H = m2 * g)
  (h3 : T_B = m1 * g + T_H)
  : m1 / m2 = 2 :=
by
  sorry

end hanging_spheres_ratio_l192_192429


namespace geometric_sequence_third_term_l192_192859

theorem geometric_sequence_third_term :
  ∀ (a_1 a_5 : ℚ) (r : ℚ), 
    a_1 = 1 / 2 →
    (a_1 * r^4) = a_5 →
    a_5 = 16 →
    (a_1 * r^2) = 2 := 
by
  intros a_1 a_5 r h1 h2 h3
  sorry

end geometric_sequence_third_term_l192_192859


namespace length_of_FD_l192_192768

-- Define the conditions
def is_square (ABCD : ℝ) (side_length : ℝ) : Prop :=
  side_length = 8 ∧ ABCD = 4 * side_length

def point_E (x : ℝ) : Prop :=
  x = 8 / 3

def point_F (CD : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 8

-- State the theorem
theorem length_of_FD (side_length : ℝ) (x : ℝ) (CD ED FD : ℝ) :
  is_square 4 side_length → 
  point_E ED → 
  point_F CD x → 
  FD = 20 / 9 :=
by
  sorry

end length_of_FD_l192_192768


namespace locus_of_D_l192_192566

theorem locus_of_D 
  (a b : ℝ)
  (hA : 0 ≤ a ∧ a ≤ (2 * Real.sqrt 3 / 3))
  (hB : 0 ≤ b ∧ b ≤ (2 * Real.sqrt 3 / 3))
  (AB_eq : Real.sqrt ((b - 2 * a)^2 + (Real.sqrt 3 * b)^2)  = 2) :
  3 * (b - a / 2)^2 + (Real.sqrt 3 / 2 * (a + b))^2 / 3 = 1 :=
sorry

end locus_of_D_l192_192566


namespace part1_part2_l192_192614

section 
variable {a b : ℚ}

-- Define the new operation as given in the condition
def odot (a b : ℚ) : ℚ := a * (a + b) - 1

-- Prove the given results
theorem part1 : odot 3 (-2) = 2 :=
by
  -- Proof omitted
  sorry

theorem part2 : odot (-2) (odot 3 5) = -43 :=
by
  -- Proof omitted
  sorry

end

end part1_part2_l192_192614


namespace tom_age_ratio_l192_192518

variable (T N : ℕ)

theorem tom_age_ratio (h_sum : T = T) (h_relation : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end tom_age_ratio_l192_192518


namespace salary_of_thomas_l192_192695

variable (R Ro T : ℕ)

theorem salary_of_thomas 
  (h1 : R + Ro = 8000) 
  (h2 : R + Ro + T = 15000) : 
  T = 7000 := by
  sorry

end salary_of_thomas_l192_192695


namespace parabola_and_x4_value_l192_192039

theorem parabola_and_x4_value :
  (∀ P, dist P (0, 1/2) = dist P (x, -1/2) → ∃ y, P = (x, y) ∧ x^2 = 2 * y) ∧
  (∀ (x1 x2 : ℝ), x1 = 6 → x2 = 2 → ∃ x4, 1/x4 = 1/((3/2) : ℝ) + 1/x2 ∧ x4 = 6/7) :=
by
  sorry

end parabola_and_x4_value_l192_192039


namespace sum_and_product_roots_l192_192670

structure quadratic_data where
  m : ℝ
  n : ℝ

def roots_sum_eq (qd : quadratic_data) : Prop :=
  qd.m / 3 = 9

def roots_product_eq (qd : quadratic_data) : Prop :=
  qd.n / 3 = 20

theorem sum_and_product_roots (qd : quadratic_data) :
  roots_sum_eq qd → roots_product_eq qd → qd.m + qd.n = 87 := by
  sorry

end sum_and_product_roots_l192_192670


namespace physics_class_size_l192_192147

theorem physics_class_size (total_students physics_only math_only both : ℕ) 
  (h1 : total_students = 100)
  (h2 : physics_only + math_only + both = total_students)
  (h3 : both = 10)
  (h4 : physics_only + both = 2 * (math_only + both)) :
  physics_only + both = 62 := 
by sorry

end physics_class_size_l192_192147


namespace last_digit_of_one_over_two_pow_twelve_l192_192985

theorem last_digit_of_one_over_two_pow_twelve : 
  let x : ℚ := 1 / 2^12 in (x * 10^12).den = 244140625 → (x.toReal - floor x.toReal) * 10 ^ 12 = 244140625 :=
by
  sorry

end last_digit_of_one_over_two_pow_twelve_l192_192985


namespace monotone_f_iff_l192_192893

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x < 1 then a^x
  else x^2 + 4 / x + a * Real.log x

theorem monotone_f_iff (a : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ↔ 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end monotone_f_iff_l192_192893


namespace geometric_sequence_product_l192_192217

theorem geometric_sequence_product :
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 5 = 16 ∧ 
    (∀ n, a (n + 1) = a n * r) ∧
    ∃ r : ℝ, 
      a 2 * a 3 * a 4 = 64 :=
by
  sorry

end geometric_sequence_product_l192_192217


namespace problem1_problem2_problem3_problem4_l192_192590

-- Define predicate conditions and solutions in Lean 4 for each problem

theorem problem1 (x : ℝ) :
  -2 * x^2 + 3 * x + 9 > 0 ↔ (-3 / 2 < x ∧ x < 3) := by
  sorry

theorem problem2 (x : ℝ) :
  (8 - x) / (5 + x) > 1 ↔ (-5 < x ∧ x ≤ 3 / 2) := by
  sorry

theorem problem3 (x : ℝ) :
  ¬ (-x^2 + 2 * x - 3 > 0) ↔ True := by
  sorry

theorem problem4 (x : ℝ) :
  x^2 - 14 * x + 50 > 0 ↔ True := by
  sorry

end problem1_problem2_problem3_problem4_l192_192590


namespace prime_or_four_no_square_div_factorial_l192_192585

theorem prime_or_four_no_square_div_factorial (n : ℕ) :
  (n * n ∣ n!) = false ↔ Nat.Prime n ∨ n = 4 := by
  sorry

end prime_or_four_no_square_div_factorial_l192_192585


namespace find_interest_rate_l192_192277

theorem find_interest_rate (P r : ℝ) 
  (h1 : 100 = P * (1 + 2 * r)) 
  (h2 : 200 = P * (1 + 6 * r)) : 
  r = 0.5 :=
sorry

end find_interest_rate_l192_192277


namespace product_of_largest_integer_digits_l192_192315

theorem product_of_largest_integer_digits (u v : ℕ) :
  u^2 + v^2 = 45 ∧ u < v → u * v = 18 :=
sorry

end product_of_largest_integer_digits_l192_192315


namespace parametric_two_rays_l192_192247

theorem parametric_two_rays (t : ℝ) : (x = t + 1 / t ∧ y = 2) → (x ≤ -2 ∨ x ≥ 2) := by
  sorry

end parametric_two_rays_l192_192247


namespace batsman_average_after_17th_inning_l192_192544

-- Define the conditions and prove the required question.
theorem batsman_average_after_17th_inning (A : ℕ) (h1 : 17 * (A + 10) = 16 * A + 300) :
  (A + 10) = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l192_192544


namespace luke_pages_lemma_l192_192549

def number_of_new_cards : ℕ := 3
def number_of_old_cards : ℕ := 9
def cards_per_page : ℕ := 3
def total_number_of_cards := number_of_new_cards + number_of_old_cards
def total_number_of_pages := total_number_of_cards / cards_per_page

theorem luke_pages_lemma : total_number_of_pages = 4 := by
  sorry

end luke_pages_lemma_l192_192549


namespace distance_is_18_l192_192410

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  let faster := (x + 1) * (3 * t / 4) = d
  let slower := (x - 1) * (t + 3) = d
  let normal := x * t = d
  faster ∧ slower ∧ normal

theorem distance_is_18 : 
  ∃ (x t : ℝ), distance_walked x t 18 :=
by
  sorry

end distance_is_18_l192_192410


namespace complex_problem_l192_192743

open Complex

theorem complex_problem (a b : ℝ) (h : (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I) : a + b = 4 := by
  sorry

end complex_problem_l192_192743


namespace markup_percentage_l192_192137

theorem markup_percentage {C : ℝ} (hC0: 0 < C) (h1: 0 < 1.125 * C) : 
  ∃ (x : ℝ), 0.75 * (1.20 * C * (1 + x / 100)) = 1.125 * C ∧ x = 25 := 
by
  have h2 : 1.20 = (6 / 5 : ℝ) := by norm_num
  have h3 : 0.75 = (3 / 4 : ℝ) := by norm_num
  sorry

end markup_percentage_l192_192137


namespace A_share_in_profit_l192_192403

/-
Given:
1. a_contribution (A's amount contributed in Rs. 5000) and duration (in months 8)
2. b_contribution (B's amount contributed in Rs. 6000) and duration (in months 5)
3. total_profit (Total profit in Rs. 8400)

Prove that A's share in the total profit is Rs. 4800.
-/

theorem A_share_in_profit 
  (a_contribution : ℝ) (a_months : ℝ) 
  (b_contribution : ℝ) (b_months : ℝ) 
  (total_profit : ℝ) :
  a_contribution = 5000 → 
  a_months = 8 → 
  b_contribution = 6000 → 
  b_months = 5 → 
  total_profit = 8400 → 
  (a_contribution * a_months / (a_contribution * a_months + b_contribution * b_months) * total_profit) = 4800 := 
by {
  sorry
}

end A_share_in_profit_l192_192403


namespace initial_average_weight_l192_192390

theorem initial_average_weight (A : ℝ) (weight7th : ℝ) (new_avg_weight : ℝ) (initial_num : ℝ) (total_num : ℝ) 
  (h_weight7th : weight7th = 97) (h_new_avg_weight : new_avg_weight = 151) (h_initial_num : initial_num = 6) (h_total_num : total_num = 7) :
  initial_num * A + weight7th = total_num * new_avg_weight → A = 160 := 
by 
  intros h
  sorry

end initial_average_weight_l192_192390


namespace exists_hamiltonian_path_l192_192765

theorem exists_hamiltonian_path (n : ℕ) (cities : Fin n → Type) (roads : ∀ (i j : Fin n), cities i → cities j → Prop) 
(road_one_direction : ∀ i j (c1 : cities i) (c2 : cities j), roads i j c1 c2 → ¬ roads j i c2 c1) :
∃ start : Fin n, ∃ path : Fin n → Fin n, ∀ i j : Fin n, i ≠ j → path i ≠ path j :=
sorry

end exists_hamiltonian_path_l192_192765


namespace closest_integer_to_cube_root_of_250_l192_192114

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l192_192114


namespace cylinder_height_proof_l192_192146

noncomputable def cone_base_radius : ℝ := 15
noncomputable def cone_height : ℝ := 25
noncomputable def cylinder_base_radius : ℝ := 10
noncomputable def cylinder_water_height : ℝ := 18.75

theorem cylinder_height_proof :
  (1 / 3 * π * cone_base_radius^2 * cone_height) = π * cylinder_base_radius^2 * cylinder_water_height :=
by sorry

end cylinder_height_proof_l192_192146


namespace relationship_between_f_l192_192613

-- Given definitions
def quadratic_parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def axis_of_symmetry (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y ↔ x + y = 2 * α

-- The problem statement to prove in Lean 4
theorem relationship_between_f (a b c x : ℝ) (hpos : x > 0) (apos : a > 0) :
  axis_of_symmetry (quadratic_parabola a b c) 1 →
  quadratic_parabola a b c (3^x) > quadratic_parabola a b c (2^x) :=
by
  sorry

end relationship_between_f_l192_192613


namespace raise_percentage_to_original_l192_192547

-- Let original_salary be a variable representing the original salary.
-- Since the salary was reduced by 50%, the reduced_salary is half of the original_salary.
-- We need to prove that to get the reduced_salary back to the original_salary, 
-- it must be increased by 100%.

noncomputable def original_salary : ℝ := sorry
noncomputable def reduced_salary : ℝ := original_salary * 0.5

theorem raise_percentage_to_original :
  (original_salary - reduced_salary) / reduced_salary * 100 = 100 :=
sorry

end raise_percentage_to_original_l192_192547


namespace machines_work_together_time_l192_192275

theorem machines_work_together_time (rate1 rate2 : ℚ) (h1 : rate1 = 1 / 20) (h2 : rate2 = 1 / 30) :
  (1 / (rate1 + rate2)) = 12 :=
by
  sorry

end machines_work_together_time_l192_192275


namespace rectangle_divided_area_l192_192209

-- Define the parameters for the rectangle
def rectangle_sides (AD AB : ℝ) : Prop := AD = 4 ∧ AB = 2

-- Define the angle bisectors
def angle_bisectors (AK DL : ℝ → ℝ) : Prop := 
  ∀ (x : ℝ), (0 ≤ x) → (x ≤ AB) → (AK x = x / 2) ∧ (DL x = x / 2)

-- Have the condition about rectangle sides and angle bisectors and prove the division of area
theorem rectangle_divided_area (AD AB : ℝ) (AK DL : ℝ → ℝ) 
  (h_sides : rectangle_sides AD AB) (h_bisectors : angle_bisectors AK DL) :
  ∃ A1 A2 A3 : ℝ, A1 = 2 ∧ A2 = 2 ∧ A3 = 4 ∧ (A1 + A2 + A3 = AD * AB) :=
by
  sorry

end rectangle_divided_area_l192_192209


namespace length_of_diagonal_l192_192494

open Real

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, -a^2)
noncomputable def B (a : ℝ) : ℝ × ℝ := (-a, -a^2)
noncomputable def C (a : ℝ) : ℝ × ℝ := (a, -a^2)
def O : ℝ × ℝ := (0, 0)

noncomputable def is_square (A B O C : ℝ × ℝ) : Prop :=
  dist A B = dist B O ∧ dist B O = dist O C ∧ dist O C = dist C A

theorem length_of_diagonal (a : ℝ) (h_square : is_square (A a) (B a) O (C a)) : 
  dist (A a) (C a) = 2 * abs a :=
sorry

end length_of_diagonal_l192_192494


namespace product_remainder_div_5_l192_192150

theorem product_remainder_div_5 :
  (1234 * 1567 * 1912) % 5 = 1 :=
by
  sorry

end product_remainder_div_5_l192_192150


namespace probability_at_least_one_correct_l192_192473

-- Define the probability of missing a single question
def prob_miss_one : ℚ := 3 / 4

-- Define the probability of missing all six questions
def prob_miss_six : ℚ := prob_miss_one ^ 6

-- Define the probability of getting at least one correct answer
def prob_at_least_one : ℚ := 1 - prob_miss_six

-- The problem statement
theorem probability_at_least_one_correct :
  prob_at_least_one = 3367 / 4096 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_one_correct_l192_192473


namespace parallel_lines_solution_l192_192332

theorem parallel_lines_solution (m : ℝ) :
  (∀ x y : ℝ, (x + (1 + m) * y + (m - 2) = 0) → (m * x + 2 * y + 8 = 0)) → m = 1 :=
by
  sorry

end parallel_lines_solution_l192_192332


namespace range_of_a_l192_192452

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| - |x - 3| ≤ a) → a ≥ -5 := by
  sorry

end range_of_a_l192_192452


namespace g_of_3_eq_seven_over_two_l192_192745

theorem g_of_3_eq_seven_over_two :
  ∀ f g : ℝ → ℝ,
  (∀ x, f x = (2 * x + 3) / (x - 1)) →
  (∀ x, g x = (x + 4) / (x - 1)) →
  g 3 = 7 / 2 :=
by
  sorry

end g_of_3_eq_seven_over_two_l192_192745


namespace same_side_of_line_l192_192750

theorem same_side_of_line (a : ℝ) :
    let point1 := (3, -1)
    let point2 := (-4, -3)
    let line_eq (x y : ℝ) := 3 * x - 2 * y + a
    (line_eq point1.1 point1.2) * (line_eq point2.1 point2.2) > 0 ↔
        (a < -11 ∨ a > 6) := sorry

end same_side_of_line_l192_192750


namespace find_a9_a10_l192_192474

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

theorem find_a9_a10 (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 2) :
  a 9 + a 10 = 16 := 
sorry

end find_a9_a10_l192_192474


namespace derivative_at_2_l192_192531

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l192_192531


namespace count_neither_3_nor_4_l192_192756

def is_multiple_of_3_or_4 (n : Nat) : Bool := (n % 3 = 0) ∨ (n % 4 = 0)

def three_digit_numbers := List.range' 100 900 -- Generates a list from 100 to 999 (inclusive)

def count_multiples_of_3_or_4 : Nat := three_digit_numbers.filter is_multiple_of_3_or_4 |>.length

def count_total := 900 -- Since three-digit numbers range from 100 to 999

theorem count_neither_3_nor_4 : count_total - count_multiples_of_3_or_4 = 450 := by
  sorry

end count_neither_3_nor_4_l192_192756


namespace min_games_to_achieve_98_percent_l192_192503

-- Define initial conditions
def initial_games : ℕ := 5
def initial_sharks_wins : ℕ := 2
def initial_tigers_wins : ℕ := 3

-- Define the total number of games and the total number of wins by the Sharks after additional games
def total_games (N : ℕ) : ℕ := initial_games + N
def total_sharks_wins (N : ℕ) : ℕ := initial_sharks_wins + N

-- Define the Sharks' winning percentage
def sharks_winning_percentage (N : ℕ) : ℚ := total_sharks_wins N / total_games N

-- Define the minimum number of additional games needed
def minimum_N : ℕ := 145

-- Theorem: Prove that the Sharks' winning percentage is at least 98% when N = 145
theorem min_games_to_achieve_98_percent :
  sharks_winning_percentage minimum_N ≥ 49 / 50 :=
sorry

end min_games_to_achieve_98_percent_l192_192503


namespace sqrt_sum_simplify_l192_192961

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l192_192961


namespace solve_y_l192_192969

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l192_192969


namespace value_of_g_at_3_l192_192759

theorem value_of_g_at_3 (g : ℕ → ℕ) (h : ∀ x, g (x + 2) = 2 * x + 3) : g 3 = 5 := by
  sorry

end value_of_g_at_3_l192_192759


namespace solve_eq1_solve_eq2_l192_192068

theorem solve_eq1 (x : ℝ) : (12 * (x - 1) ^ 2 = 3) ↔ (x = 3/2 ∨ x = 1/2) := 
by sorry

theorem solve_eq2 (x : ℝ) : ((x + 1) ^ 3 = 0.125) ↔ (x = -0.5) := 
by sorry

end solve_eq1_solve_eq2_l192_192068


namespace new_average_doubled_l192_192070

theorem new_average_doubled (n : ℕ) (avg : ℝ) (h1 : n = 12) (h2 : avg = 50) :
  2 * avg = 100 := by
sorry

end new_average_doubled_l192_192070


namespace winning_votes_cast_l192_192679

theorem winning_votes_cast (V : ℝ) (h1 : 0.40 * V = 280) : 0.70 * V = 490 :=
by
  sorry

end winning_votes_cast_l192_192679


namespace absent_laborers_l192_192408

theorem absent_laborers (L : ℝ) (A : ℝ) (hL : L = 17.5) (h_work_done : (L - A) / 10 = L / 6) : A = 14 :=
by
  sorry

end absent_laborers_l192_192408


namespace customers_left_tip_l192_192717

-- Definition of the given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def customers_didnt_tip : ℕ := 34

-- Lean 4 statement proving that the number of customers who did leave a tip (answer) equals 15
theorem customers_left_tip : (initial_customers + added_customers - customers_didnt_tip) = 15 :=
by
  sorry

end customers_left_tip_l192_192717


namespace sum_distinct_prime_factors_of_7pow7_minus_7pow4_l192_192876

noncomputable def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  let factors := (Nat.factors n).erase_dup
  factors.sum

theorem sum_distinct_prime_factors_of_7pow7_minus_7pow4 :
  sum_of_distinct_prime_factors (7 ^ 7 - 7 ^ 4) = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7pow7_minus_7pow4_l192_192876


namespace solve_for_x_l192_192373

theorem solve_for_x (x : ℚ) (h : (x + 8) / (x - 4) = (x - 3) / (x + 6)) : 
  x = -12 / 7 :=
sorry

end solve_for_x_l192_192373


namespace exists_n_not_perfect_square_l192_192046

theorem exists_n_not_perfect_square (a b : ℤ) (h1 : a > 1) (h2 : b > 1) (h3 : a ≠ b) : 
  ∃ (n : ℕ), (n > 0) ∧ ¬∃ (k : ℤ), (a^n - 1) * (b^n - 1) = k^2 :=
by sorry

end exists_n_not_perfect_square_l192_192046


namespace simplify_expression_l192_192500

theorem simplify_expression (c : ℤ) : (3 * c + 6 - 6 * c) / 3 = -c + 2 := by
  sorry

end simplify_expression_l192_192500


namespace find_m_l192_192837

def g (n : Int) : Int :=
  if n % 2 ≠ 0 then n + 5 else 
  if n % 3 = 0 then n / 3 else n

theorem find_m (m : Int) 
  (h_odd : m % 2 ≠ 0) 
  (h_ggg : g (g (g m)) = 35) : 
  m = 85 := 
by
  sorry

end find_m_l192_192837


namespace maximum_profit_l192_192406

noncomputable def sales_volume (x : ℝ) : ℝ := -10 * x + 1000
noncomputable def profit (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

theorem maximum_profit : ∀ x : ℝ, 44 ≤ x ∧ x ≤ 46 → profit x ≤ 8640 :=
by
  intro x hx
  sorry

end maximum_profit_l192_192406


namespace find_slope_of_line_q_l192_192780

theorem find_slope_of_line_q
  (k : ℝ)
  (h₁ : ∀ (x y : ℝ), (y = 3 * x + 5) → (y = k * x + 3) → (x = -4 ∧ y = -7))
  : k = 2.5 :=
sorry

end find_slope_of_line_q_l192_192780


namespace midpoint_coordinates_l192_192314

theorem midpoint_coordinates :
  let x1 := 2
  let y1 := -3
  let z1 := 5
  let x2 := 8
  let y2 := 3
  let z2 := -1
  ( (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2 ) = (5, 0, 2) :=
by
  sorry

end midpoint_coordinates_l192_192314


namespace min_correct_answers_l192_192767

theorem min_correct_answers (x : ℕ) : 
  (∃ x, 0 ≤ x ∧ x ≤ 20 ∧ 5 * x - (20 - x) ≥ 88) :=
sorry

end min_correct_answers_l192_192767


namespace molecular_weight_X_l192_192834

theorem molecular_weight_X (Ba_weight : ℝ) (total_molecular_weight : ℝ) (X_weight : ℝ) 
  (h1 : Ba_weight = 137) 
  (h2 : total_molecular_weight = 171) 
  (h3 : total_molecular_weight - Ba_weight * 1 = 2 * X_weight) : 
  X_weight = 17 :=
by
  sorry

end molecular_weight_X_l192_192834


namespace part_A_part_B_l192_192922

-- Definitions for the setup
variables (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0)

-- Part (A): Specific distance 5d
theorem part_A (d : ℝ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = 25 * d^2 ∧ |y - d| = 5 * d → 
  (x = 3 * d ∧ y = -4 * d) ∨ (x = -3 * d ∧ y = -4 * d)) :=
sorry

-- Part (B): General distance nd
theorem part_B (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d → ∃ x y, (x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d)) :=
sorry

end part_A_part_B_l192_192922


namespace savings_same_l192_192699

theorem savings_same (A_salary B_salary total_salary : ℝ)
  (A_spend_perc B_spend_perc : ℝ)
  (h_total : A_salary + B_salary = total_salary)
  (h_A_salary : A_salary = 4500)
  (h_A_spend_perc : A_spend_perc = 0.95)
  (h_B_spend_perc : B_spend_perc = 0.85)
  (h_total_salary : total_salary = 6000) :
  ((1 - A_spend_perc) * A_salary) = ((1 - B_spend_perc) * B_salary) :=
by
  sorry

end savings_same_l192_192699


namespace triangle_sin_ratio_cos_side_l192_192764

noncomputable section

variables (A B C a b c : ℝ)
variables (h1 : a + b + c = 5)
variables (h2 : Real.cos B = 1 / 4)
variables (h3 : Real.cos A - 2 * Real.cos C = (2 * c - a) / b * Real.cos B)

theorem triangle_sin_ratio_cos_side :
  (Real.sin C / Real.sin A = 2) ∧ (b = 2) :=
  sorry

end triangle_sin_ratio_cos_side_l192_192764


namespace Bill_wins_at_least_once_in_three_games_l192_192419

noncomputable def probability_win_at_least_once : ℚ :=
  let P_win_single_game : ℚ := 4 / 36
  let P_lose_single_game := 1 - P_win_single_game
  let P_lose_all_three_games := P_lose_single_game * P_lose_single_game * P_lose_single_game
  let P_win_at_least_once := 1 - P_lose_all_three_games
  P_win_at_least_once

theorem Bill_wins_at_least_once_in_three_games :
  probability_win_at_least_once = 217 / 729 :=
by
  unfold probability_win_at_least_once
  sorry

end Bill_wins_at_least_once_in_three_games_l192_192419


namespace one_fourths_in_seven_halves_l192_192755

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end one_fourths_in_seven_halves_l192_192755


namespace abc_minus_def_l192_192177

def f (x y z : ℕ) : ℕ := 5^x * 2^y * 3^z

theorem abc_minus_def {a b c d e f : ℕ} (ha : a = d) (hb : b = e) (hc : c = f + 1) : 
  (100 * a + 10 * b + c) - (100 * d + 10 * e + f) = 1 :=
by
  -- Proof omitted
  sorry

end abc_minus_def_l192_192177


namespace proposition_A_proposition_B_proposition_C_proposition_D_l192_192396

theorem proposition_A (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a * b < 0 := 
sorry

theorem proposition_B (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬ (a^2 < a * b ∧ a * b < b^2) := 
sorry

theorem proposition_C (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) : ¬ (a / (c - a) < b / (c - b)) := 
sorry

theorem proposition_D (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : a / b > (a + c) / (b + c) := 
sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l192_192396


namespace margaret_mean_score_l192_192391

noncomputable def cyprian_scores : List ℕ := [82, 85, 89, 91, 95, 97]
noncomputable def cyprian_mean : ℕ := 88

theorem margaret_mean_score :
  let total_sum := List.sum cyprian_scores
  let cyprian_sum := cyprian_mean * 3
  let margaret_sum := total_sum - cyprian_sum
  let margaret_mean := (margaret_sum : ℚ) / 3
  margaret_mean = 91.66666666666667 := 
by 
  -- Definitions used in conditions, skipping steps.
  sorry

end margaret_mean_score_l192_192391


namespace rectangle_area_given_perimeter_l192_192471

theorem rectangle_area_given_perimeter (x : ℝ) (h_perim : 8 * x = 160) : (2 * x) * (2 * x) = 1600 := by
  -- Definitions derived from conditions
  let length := 2 * x
  let width := 2 * x
  -- Proof transformed to a Lean statement
  have h1 : length = 40 := by sorry
  have h2 : width = 40 := by sorry
  have h_area : length * width = 1600 := by sorry
  exact h_area

end rectangle_area_given_perimeter_l192_192471


namespace total_games_in_season_l192_192808

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end total_games_in_season_l192_192808


namespace total_apples_count_l192_192804

-- Definitions based on conditions
def red_apples := 16
def green_apples := red_apples + 12
def total_apples := green_apples + red_apples

-- Statement to prove
theorem total_apples_count : total_apples = 44 := 
by
  sorry

end total_apples_count_l192_192804


namespace problem_statement_l192_192901

noncomputable def f (x k : ℝ) := x^3 / (2^x + k * 2^(-x))

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def k2_eq_1_is_nec_but_not_suff (f : ℝ → ℝ) (k : ℝ) : Prop :=
  (k^2 = 1) → (is_even_function f → k = -1 ∧ ¬(k = 1))

theorem problem_statement (k : ℝ) :
  k2_eq_1_is_nec_but_not_suff (λ x => f x k) k :=
by
  sorry

end problem_statement_l192_192901


namespace fraction_irreducible_l192_192236

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 1) = 1 :=
sorry

end fraction_irreducible_l192_192236


namespace possible_values_of_c_l192_192762

theorem possible_values_of_c (a b c : ℕ) (n : ℕ) (h₀ : a ≠ 0) (h₁ : n = 729 * a + 81 * b + 36 + c) (h₂ : ∃ k, n = k^3) :
  c = 1 ∨ c = 8 :=
sorry

end possible_values_of_c_l192_192762


namespace basic_astrophysics_budget_percent_l192_192132

theorem basic_astrophysics_budget_percent
  (total_degrees : ℝ := 360)
  (astrophysics_degrees : ℝ := 108) :
  (astrophysics_degrees / total_degrees) * 100 = 30 := by
  sorry

end basic_astrophysics_budget_percent_l192_192132


namespace exists_excircle_radius_at_least_three_times_incircle_radius_l192_192647

variable (a b c s T r ra rb rc : ℝ)
variable (ha : ra = T / (s - a))
variable (hb : rb = T / (s - b))
variable (hc : rc = T / (s - c))
variable (hincircle : r = T / s)

theorem exists_excircle_radius_at_least_three_times_incircle_radius
  (ha : ra = T / (s - a)) (hb : rb = T / (s - b)) (hc : rc = T / (s - c)) (hincircle : r = T / s) :
  ∃ rc, rc ≥ 3 * r :=
by {
  use rc,
  sorry
}

end exists_excircle_radius_at_least_three_times_incircle_radius_l192_192647


namespace common_difference_l192_192011

theorem common_difference (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ)
    (h₁ : a 5 + a 6 = -10)
    (h₂ : S 14 = -14)
    (h₃ : ∀ n, S n = n * (a 1 + a n) / 2)
    (h₄ : ∀ n, a (n + 1) = a n + d) :
  d = 2 :=
sorry

end common_difference_l192_192011


namespace teena_distance_behind_poe_l192_192974

theorem teena_distance_behind_poe (D : ℝ)
    (teena_speed : ℝ) (poe_speed : ℝ)
    (time_hours : ℝ) (teena_ahead : ℝ) :
    teena_speed = 55 
    → poe_speed = 40 
    → time_hours = 1.5 
    → teena_ahead = 15 
    → D + teena_ahead = (teena_speed - poe_speed) * time_hours 
    → D = 7.5 := 
by 
    intros 
    sorry

end teena_distance_behind_poe_l192_192974


namespace sum_of_distances_l192_192574

theorem sum_of_distances (A B C D M P : ℝ × ℝ) 
    (hA : A = (0, 0))
    (hB : B = (4, 0))
    (hC : C = (4, 4))
    (hD : D = (0, 4))
    (hM : M = (2, 0))
    (hP : P = (0, 2)) :
    dist A M + dist A P = 4 :=
by
  sorry

end sum_of_distances_l192_192574


namespace trigonometric_identity_l192_192122

theorem trigonometric_identity
  (sin : ℝ → ℝ)
  (cos : ℝ → ℝ)
  (h_sin_cos_180: ∀ x, cos (180 - x) = - cos x)
  (h_cos_90: ∀ x, cos (90 + x) = - sin x)
  : (sin 20 * cos 10 + cos 160 * cos 100) / (sin 21 * cos 9 + cos 159 * cos 99) = 1 :=
by
  sorry

end trigonometric_identity_l192_192122


namespace miss_davis_sticks_left_l192_192364

theorem miss_davis_sticks_left (initial_sticks groups_per_class sticks_per_group : ℕ) 
(h1 : initial_sticks = 170) (h2 : groups_per_class = 10) (h3 : sticks_per_group = 15) : 
initial_sticks - (groups_per_class * sticks_per_group) = 20 :=
by sorry

end miss_davis_sticks_left_l192_192364


namespace correct_ordering_of_powers_l192_192822

theorem correct_ordering_of_powers :
  (6 ^ 8) < (3 ^ 15) ∧ (3 ^ 15) < (8 ^ 10) :=
by
  -- Define the expressions for each power
  let a := (8 : ℕ) ^ 10
  let b := (3 : ℕ) ^ 15
  let c := (6 : ℕ) ^ 8
  
  -- To utilize the values directly in inequalities
  have h1 : (c < b) := sorry -- Proof that 6^8 < 3^15
  have h2 : (b < a) := sorry -- Proof that 3^15 < 8^10

  exact ⟨h1, h2⟩ -- Conjunction of h1 and h2 to show 6^8 < 3^15 < 8^10

end correct_ordering_of_powers_l192_192822


namespace maximum_value_of_k_l192_192608

-- Define the variables and conditions
variables {a b c k : ℝ}
axiom h₀ : a > b
axiom h₁ : b > c
axiom h₂ : 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0

-- State the theorem
theorem maximum_value_of_k : k ≤ 9 := sorry

end maximum_value_of_k_l192_192608


namespace sandwiches_bought_l192_192690

theorem sandwiches_bought (sandwich_cost soda_cost total_cost_sodas total_cost : ℝ)
  (h1 : sandwich_cost = 2.45)
  (h2 : soda_cost = 0.87)
  (h3 : total_cost_sodas = 4 * soda_cost)
  (h4 : total_cost = 8.38) :
  ∃ (S : ℕ), sandwich_cost * S + total_cost_sodas = total_cost ∧ S = 2 :=
by
  use 2
  simp [h1, h2, h3, h4]
  sorry

end sandwiches_bought_l192_192690


namespace g_of_neg2_l192_192168

def g (x : ℤ) : ℤ := x^3 - x^2 + x

theorem g_of_neg2 : g (-2) = -14 := 
by
  sorry

end g_of_neg2_l192_192168


namespace expression_multiple_of_five_l192_192740

theorem expression_multiple_of_five (n : ℕ) (h : n ≥ 10) : 
  (∃ k : ℕ, (n + 2) * (n + 1) = 5 * k) :=
sorry

end expression_multiple_of_five_l192_192740


namespace annual_income_increase_l192_192243

variable (x y : ℝ)

-- Definitions of the conditions
def regression_line (x : ℝ) : ℝ := 0.254 * x + 0.321

-- The statement we want to prove
theorem annual_income_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 0.254 := 
sorry

end annual_income_increase_l192_192243


namespace number_of_pairs_of_shoes_l192_192700

/-- A box contains some pairs of shoes with a total of 10 shoes.
    If two shoes are selected at random, the probability that they are matching shoes is 1/9.
    Prove that the number of pairs of shoes in the box is 5. -/
theorem number_of_pairs_of_shoes (n : ℕ) (h1 : 2 * n = 10) 
  (h2 : ((n * (n - 1)) / (10 * (10 - 1))) = 1 / 9) : n = 5 := 
sorry

end number_of_pairs_of_shoes_l192_192700


namespace coterminal_angle_neg_60_eq_300_l192_192991

theorem coterminal_angle_neg_60_eq_300 :
  ∃ k : ℤ, 0 ≤ k * 360 - 60 ∧ k * 360 - 60 < 360 ∧ (k * 360 - 60 = 300) := by
  sorry

end coterminal_angle_neg_60_eq_300_l192_192991


namespace integer_solutions_for_even_ratio_l192_192424

theorem integer_solutions_for_even_ratio (a : ℤ) (h : ∃ k : ℤ, (a = 2 * k * (1011 - k))): 
  a = 1010 ∨ a = 1012 ∨ a = 1008 ∨ a = 1014 ∨ a = 674 ∨ a = 1348 ∨ a = 0 ∨ a = 2022 :=
sorry

end integer_solutions_for_even_ratio_l192_192424


namespace negation_proposition_false_l192_192792

variable {R : Type} [LinearOrderedField R]

theorem negation_proposition_false (x y : R) :
  ¬ (x > 2 ∧ y > 3 → x + y > 5) = false := by
sorry

end negation_proposition_false_l192_192792


namespace value_of_y_l192_192548

theorem value_of_y (x y : ℕ) (h1 : x % y = 6) (h2 : (x : ℝ) / y = 6.12) : y = 50 :=
sorry

end value_of_y_l192_192548


namespace double_theta_acute_l192_192444

theorem double_theta_acute (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_theta_acute_l192_192444


namespace mathematically_equivalent_proof_l192_192459

noncomputable def proof_problem (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ a^x = 2 ∧ a^y = 3 → a^(x - 2 * y) = 2 / 9

theorem mathematically_equivalent_proof (a : ℝ) (x y : ℝ) :
  proof_problem a x y :=
by
  sorry  -- Proof steps will go here

end mathematically_equivalent_proof_l192_192459


namespace tangent_lines_inequality_l192_192249

theorem tangent_lines_inequality (k k1 k2 b b1 b2 : ℝ)
  (h1 : k = - (b * b) / 4)
  (h2 : k1 = - (b1 * b1) / 4)
  (h3 : k2 = - (b2 * b2) / 4)
  (h4 : b = b1 + b2) :
  k ≥ 2 * (k1 + k2) := sorry

end tangent_lines_inequality_l192_192249


namespace division_correct_l192_192160

theorem division_correct : 0.45 / 0.005 = 90 := by
  sorry

end division_correct_l192_192160


namespace jane_played_8_rounds_l192_192034

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end jane_played_8_rounds_l192_192034


namespace limit_sum_infinite_geometric_series_l192_192890

noncomputable def infinite_geometric_series_limit (a_1 q : ℝ) :=
  if |q| < 1 then (a_1 / (1 - q)) else 0

theorem limit_sum_infinite_geometric_series :
  infinite_geometric_series_limit 1 (1 / 3) = 3 / 2 :=
by
  sorry

end limit_sum_infinite_geometric_series_l192_192890


namespace remaining_scoops_l192_192702

-- Define the initial scoops for each flavor
def initial_chocolate : Nat := 10
def initial_strawberry : Nat := 10
def initial_vanilla : Nat := 10

-- Define the scoops requested by each person
def ethan_chocolate : Nat := 1
def ethan_vanilla : Nat := 1

def lucas_danny_connor_chocolate : Nat := 3 * 2

def olivia_strawberry : Nat := 1
def olivia_vanilla : Nat := 1

def shannon_strawberry : Nat := 2
def shannon_vanilla : Nat := 2

-- Calculate and prove the remaining scoops for each flavor
theorem remaining_scoops :
  let remaining_chocolate := initial_chocolate - (ethan_chocolate + lucas_danny_connor_chocolate)
  let remaining_strawberry := initial_strawberry - (olivia_strawberry + shannon_strawberry)
  let remaining_vanilla := initial_vanilla - (ethan_vanilla + olivia_vanilla + shannon_vanilla)
  remaining_chocolate + remaining_strawberry + remaining_vanilla = 16 :=
by
  let remaining_chocolate := initial_chocolate - (ethan_chocolate + lucas_danny_connor_chocolate)
  let remaining_strawberry := initial_strawberry - (olivia_strawberry + shannon_strawberry)
  let remaining_vanilla := initial_vanilla - (ethan_vanilla + olivia_vanilla + shannon_vanilla)
  calc
    remaining_chocolate + remaining_strawberry + remaining_vanilla
    = (10 - (1 + 6)) + (10 - (1 + 2)) + (10 - (1 + 1 + 2)) : by rfl
    ... = 3 + 7 + 6 : by rfl
    ... = 16 : by rfl

end remaining_scoops_l192_192702


namespace symmetric_points_existence_l192_192609

-- Define the ellipse equation
def is_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define the line equation parameterized by m
def line_eq (x y m : ℝ) : Prop :=
  y = 4 * x + m

-- Define the range for m such that symmetric points exist
def m_in_range (m : ℝ) : Prop :=
  - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13

-- Prove the existence of symmetric points criteria for m
theorem symmetric_points_existence (m : ℝ) :
  (∀ (x y : ℝ), is_ellipse x y → line_eq x y m → 
    (∃ (x1 y1 x2 y2 : ℝ), is_ellipse x1 y1 ∧ is_ellipse x2 y2 ∧ line_eq x1 y1 m ∧ line_eq x2 y2 m ∧ 
      (x1 = x2) ∧ (y1 = -y2))) ↔ m_in_range m :=
sorry

end symmetric_points_existence_l192_192609


namespace closest_cube_root_of_250_l192_192103

theorem closest_cube_root_of_250 : 
  let n := 6 in 
  | (n^3 - 250) | < | ((n+1)^3 - 250) | ∧ | (n^3 - 250) | < | ((n-1)^3 - 250) :=
by
  let n := 6
  have h1 : n = 6 := rfl
  have h2 : (n + 1) = 7 := rfl
  have h3 : (n - 1) = 5 := rfl
  have h4 :  n^3 = 216 := by norm_num
  have h5 : ((n + 1)^3) = 343 := by norm_num
  have h6 : ((n - 1)^3) = 125 := by norm_num
  have h7 : | (n^3 - 250) | = 34 := by norm_num
  have h8 : | ((n + 1)^3 - 250) | = 93 := by norm_num
  have h9 : | ((n - 1)^3 - 250) | = 125 := by norm_num
  show 34 < 93 ∧ 34 < 125 from by norm_num
  sorry

end closest_cube_root_of_250_l192_192103


namespace evaluate_g_at_8_l192_192619

def g (n : ℕ) : ℕ := n^2 - 3 * n + 29

theorem evaluate_g_at_8 : g 8 = 69 := by
  unfold g
  calc
    8^2 - 3 * 8 + 29 = 64 - 24 + 29 := by simp
                      _ = 69 := by norm_num

end evaluate_g_at_8_l192_192619


namespace condition_sufficient_not_necessary_l192_192921

-- Define the triangle ABC
variables {A B C : Point ℝ}
variables {t : ℝ} (ht : t ≠ 1)

-- Define the vectors BA, BC, and AC
def BA := vector (B - A)
def BC := vector (B - C)
def AC := vector (A - C)

-- Condition
axiom h : ∀ t ≠ 1, ∥BA - t • BC∥ > ∥AC∥

-- Theorem: The given condition is sufficient but not necessary for a right-angled triangle at C
theorem condition_sufficient_not_necessary : 
  (∀ t ≠ 1, ∥BA - t • BC∥ > ∥AC∥) → ∃ (R : ℝ), right_angle_triangle R A B C :=
  sorry

end condition_sufficient_not_necessary_l192_192921


namespace rubiks_cube_repeats_l192_192165

theorem rubiks_cube_repeats (num_positions : ℕ) (H : num_positions = 43252003274489856000) 
  (moves : ℕ → ℕ) : 
  ∃ n, ∃ m, (∀ P, moves n = moves m → P = moves 0) :=
by
  sorry

end rubiks_cube_repeats_l192_192165


namespace yellow_curved_given_curved_l192_192343

variable (P_green : ℝ) (P_yellow : ℝ) (P_straight : ℝ) (P_curved : ℝ)
variable (P_red_given_straight : ℝ) 

-- Given conditions
variables (h1 : P_green = 3 / 4) 
          (h2 : P_yellow = 1 / 4) 
          (h3 : P_straight = 1 / 2) 
          (h4 : P_curved = 1 / 2)
          (h5 : P_red_given_straight = 1 / 3)

-- To be proven
theorem yellow_curved_given_curved : (P_yellow * P_curved) / P_curved = 1 / 4 :=
by
sorry

end yellow_curved_given_curved_l192_192343


namespace solve_problem_l192_192729

def problem_statement : Prop :=
  ⌊ (2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011) ⌋ = 8

theorem solve_problem : problem_statement := 
  by sorry

end solve_problem_l192_192729


namespace total_wire_length_l192_192831

theorem total_wire_length (S : ℕ) (L : ℕ)
  (hS : S = 20) 
  (hL : L = 2 * S) : S + L = 60 :=
by
  sorry

end total_wire_length_l192_192831


namespace min_cubes_needed_proof_l192_192983

noncomputable def min_cubes_needed_to_form_30_digit_number : ℕ :=
  sorry

theorem min_cubes_needed_proof : min_cubes_needed_to_form_30_digit_number = 50 :=
  sorry

end min_cubes_needed_proof_l192_192983


namespace correct_first_coupon_day_l192_192630

def is_redemption_valid (start_day : ℕ) (interval : ℕ) (num_coupons : ℕ) (closed_day : ℕ) : Prop :=
  ∀ n : ℕ, n < num_coupons → (start_day + n * interval) % 7 ≠ closed_day

def wednesday : ℕ := 3  -- Assuming Sunday = 0, Monday = 1, ..., Saturday = 6

theorem correct_first_coupon_day : 
  is_redemption_valid wednesday 10 6 0 :=
by {
  -- Proof goes here
  sorry
}

end correct_first_coupon_day_l192_192630


namespace train_speed_160m_6sec_l192_192130

noncomputable def train_speed (distance time : ℕ) : ℚ :=
(distance : ℚ) / (time : ℚ)

theorem train_speed_160m_6sec : train_speed 160 6 = 26.67 :=
by
  simp [train_speed]
  norm_num
  sorry

end train_speed_160m_6sec_l192_192130


namespace ratio_of_drinking_speeds_l192_192057

def drinking_ratio(mala_portion usha_portion : ℚ) (same_time: Bool) (usha_fraction: ℚ) : ℚ :=
if same_time then mala_portion / usha_portion else 0

theorem ratio_of_drinking_speeds
  (mala_portion : ℚ)
  (usha_portion : ℚ)
  (same_time : Bool)
  (usha_fraction : ℚ)
  (usha_drank : usha_fraction = 2 / 10)
  (mala_drank : mala_portion = 1 - usha_fraction)
  (equal_time : same_time = tt)
  (ratio : drinking_ratio mala_portion usha_portion same_time usha_fraction = 4) :
  mala_portion / usha_portion = 4 :=
by
  sorry

end ratio_of_drinking_speeds_l192_192057


namespace marked_price_percentage_l192_192294

theorem marked_price_percentage (L C M S : ℝ) 
  (h1 : C = 0.7 * L) 
  (h2 : C = 0.7 * S) 
  (h3 : S = 0.9 * M) 
  (h4 : S = L) 
  : M = (10 / 9) * L := 
by
  sorry

end marked_price_percentage_l192_192294


namespace inequalities_correct_l192_192440

open Real

-- Define the variables and conditions
variables (a b : ℝ) (h_pos : a * b > 0)

-- Define the inequalities
def inequality_B : Prop := 2 * (a^2 + b^2) >= (a + b)^2
def inequality_C : Prop := (b / a) + (a / b) >= 2
def inequality_D : Prop := (a + 1 / a) * (b + 1 / b) >= 4

-- The Lean statement
theorem inequalities_correct : inequality_B a b h_pos ∧ inequality_C a b h_pos ∧ inequality_D a b h_pos :=
by
  sorry

end inequalities_correct_l192_192440


namespace find_f_prime_at_2_l192_192536

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l192_192536


namespace continuity_of_f_at_3_l192_192772

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3*x^2 + 2*x - 4 else b*x + 7

theorem continuity_of_f_at_3 (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x b - f 3 b) < ε) ↔ b = 22 / 3 :=
by
  sorry

end continuity_of_f_at_3_l192_192772


namespace probability_x_y_gte_one_fourth_l192_192783

open ProbabilityTheory

noncomputable def probability_x_y_condition := sorry

theorem probability_x_y_gte_one_fourth:
  let E : Set (ℝ × ℝ) := {z : ℝ × ℝ | |z.1 - z.2| > 1 / 4} in
  P(E | probability_x_y_condition) = 1 / 2 := 
sorry

end probability_x_y_gte_one_fourth_l192_192783


namespace complement_of_intersection_l192_192454

open Set

def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}
def S : Set ℝ := univ -- S is the set of all real numbers

theorem complement_of_intersection :
  S \ (A ∩ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 3 < x } :=
by
  sorry

end complement_of_intersection_l192_192454


namespace arithmetic_sequence_common_difference_l192_192476

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
  (h_a2 : a 2 = 3)
  (h_a7 : a 7 = 13) : 
  ∃ d, ∀ n, a n = a 1 + (n - 1) * d ∧ d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l192_192476


namespace probability_shaded_is_one_third_l192_192557

-- Define the total number of regions as a constant
def total_regions : ℕ := 12

-- Define the number of shaded regions as a constant
def shaded_regions : ℕ := 4

-- The probability that the tip of a spinner stopping in a shaded region
def probability_shaded : ℚ := shaded_regions / total_regions

-- Main theorem stating the probability calculation is correct
theorem probability_shaded_is_one_third : probability_shaded = 1 / 3 :=
by
  sorry

end probability_shaded_is_one_third_l192_192557


namespace range_of_a_l192_192753

def proposition_P (a : ℝ) := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def proposition_Q (a : ℝ) := 5 - 2*a > 1

theorem range_of_a :
  (∃! (p : Prop), (p = proposition_P a ∨ p = proposition_Q a) ∧ p) →
  a ∈ Set.Iic (-2) :=
by
  sorry

end range_of_a_l192_192753


namespace sequence_values_l192_192016

variable {a1 a2 b2 : ℝ}

theorem sequence_values
  (arithmetic : 2 * a1 = 1 + a2 ∧ 2 * a2 = a1 + 4)
  (geometric : b2 ^ 2 = 1 * 4) :
  (a1 + a2) / b2 = 5 / 2 :=
by
  sorry

end sequence_values_l192_192016


namespace lilith_caps_collection_l192_192778

theorem lilith_caps_collection :
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * 4
  let christmas_caps := 40 * 5
  let lost_caps := 15 * 5
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps - lost_caps
  total_caps = 401 := by
  sorry

end lilith_caps_collection_l192_192778


namespace prove_original_sides_l192_192596

def original_parallelogram_sides (a b : ℕ) : Prop :=
  ∃ k : ℕ, (a, b) = (k * 1, k * 2) ∨ (a, b) = (1, 5) ∨ (a, b) = (4, 5) ∨ (a, b) = (3, 7) ∨ (a, b) = (4, 7) ∨ (a, b) = (3, 8) ∨ (a, b) = (5, 8) ∨ (a, b) = (5, 7) ∨ (a, b) = (2, 7)

theorem prove_original_sides (a b : ℕ) : original_parallelogram_sides a b → (1, 2) = (1, 2) :=
by
  intro h
  sorry

end prove_original_sides_l192_192596


namespace print_time_l192_192709

theorem print_time (P R: ℕ) (hR : R = 24) (hP : P = 360) (T : ℕ) : T = P / R → T = 15 := by
  intros h
  rw [hR, hP] at h
  exact h

end print_time_l192_192709


namespace line_through_point_parallel_to_l_l192_192664

theorem line_through_point_parallel_to_l {x y : ℝ} (l : ℝ → ℝ → Prop) (A : ℝ × ℝ) :
  l = (λ x y, 2 * x - 4 * y + 7 = 0) → A = (2, 3) →
  (∃ m, (λ x y, 2 * x - 4 * y + m = 0) = (λ x y, x - 2 * y + 4 = 0)) :=
by
  intros hl hA
  use 8
  -- Further proof steps would go here
  sorry

end line_through_point_parallel_to_l_l192_192664


namespace problem_to_prove_l192_192049

theorem problem_to_prove
  (a b c : ℝ)
  (h1 : a + b + c = -3)
  (h2 : a * b + b * c + c * a = -10)
  (h3 : a * b * c = -5) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 70 :=
by
  sorry

end problem_to_prove_l192_192049


namespace convert_yahs_to_bahs_l192_192617

theorem convert_yahs_to_bahs :
  (∀ (bahs rahs yahs : ℝ), (10 * bahs = 18 * rahs) 
    ∧ (6 * rahs = 10 * yahs) 
    → (1500 * yahs / (10 / 6) / (18 / 10) = 500 * bahs)) :=
by
  intros bahs rahs yahs h
  sorry

end convert_yahs_to_bahs_l192_192617


namespace incorrect_conclusions_l192_192331

variables (a b : ℝ)

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem incorrect_conclusions :
  a > 0 → b > 0 → a ≠ 1 → b ≠ 1 → log_base a b > 1 →
  (a < 1 ∧ b > a ∨ (¬ (b < 1 ∧ b < a) ∧ ¬ (a < 1 ∧ a < b))) :=
by intros ha hb ha_ne1 hb_ne1 hlog; sorry

end incorrect_conclusions_l192_192331


namespace distribution_centers_l192_192703

theorem distribution_centers (n : ℕ) (h : n = 5) : 
  (n + (nat.choose n 2) = 15) :=
by
  rw h
  -- simplifying n and the binomial coefficient for n = 5

  sorry

end distribution_centers_l192_192703


namespace power_of_power_l192_192570

theorem power_of_power (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := 
  sorry

end power_of_power_l192_192570


namespace complex_quadrant_l192_192599

open Complex

theorem complex_quadrant :
  let z := (1 - I) * (3 + I)
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l192_192599


namespace new_remainder_when_scaled_l192_192127

theorem new_remainder_when_scaled (a b c : ℕ) (h : a = b * c + 7) : (10 * a) % (10 * b) = 70 := by
  sorry

end new_remainder_when_scaled_l192_192127


namespace max_n_leq_V_l192_192467

theorem max_n_leq_V (n : ℤ) (V : ℤ) (h1 : 102 * n^2 <= V) (h2 : ∀ k : ℤ, (102 * k^2 <= V) → k <= 8) : V >= 6528 :=
sorry

end max_n_leq_V_l192_192467


namespace trigonometric_identity_l192_192450

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  1 + Real.sin α * Real.cos α = 7 / 5 :=
by
  sorry

end trigonometric_identity_l192_192450


namespace balance_difference_is_7292_83_l192_192850

noncomputable def angela_balance : ℝ := 7000 * (1 + 0.05)^15
noncomputable def bob_balance : ℝ := 9000 * (1 + 0.03)^30
noncomputable def balance_difference : ℝ := bob_balance - angela_balance

theorem balance_difference_is_7292_83 : balance_difference = 7292.83 := by
  sorry

end balance_difference_is_7292_83_l192_192850


namespace grassy_plot_width_l192_192711

noncomputable def gravel_cost (L w p : ℝ) : ℝ :=
  0.80 * ((L + 2 * p) * (w + 2 * p) - L * w)

theorem grassy_plot_width
  (L : ℝ) 
  (p : ℝ) 
  (cost : ℝ) 
  (hL : L = 110) 
  (hp : p = 2.5) 
  (hcost : cost = 680) :
  ∃ w : ℝ, gravel_cost L w p = cost ∧ w = 97.5 :=
by
  sorry

end grassy_plot_width_l192_192711


namespace function_neither_even_nor_odd_l192_192720

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 - x

theorem function_neither_even_nor_odd : ¬is_even_function f ∧ ¬is_odd_function f := by
  sorry

end function_neither_even_nor_odd_l192_192720


namespace wind_speed_l192_192138

theorem wind_speed (w : ℝ) (h : 420 / (253 + w) = 350 / (253 - w)) : w = 23 :=
by
  sorry

end wind_speed_l192_192138


namespace sum_17_20_l192_192920

-- Definitions for the conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ) 
variable (r : ℝ)

-- The sequence is geometric and sums are defined accordingly
axiom geo_seq : ∀ n, a (n + 1) = r * a n
axiom sum_def : ∀ n, S n = ∑ i in finset.range n, a (i + 1)

-- Given conditions
axiom S4 : S 4 = 1
axiom S8 : S 8 = 3

-- The value we need to prove
theorem sum_17_20 : a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end sum_17_20_l192_192920


namespace total_cookies_l192_192201

   -- Define the conditions
   def cookies_per_bag : ℕ := 41
   def number_of_bags : ℕ := 53

   -- Define the problem: Prove that the total number of cookies is 2173
   theorem total_cookies : cookies_per_bag * number_of_bags = 2173 :=
   by sorry
   
end total_cookies_l192_192201


namespace chris_score_l192_192372

variable (s g c : ℕ)

theorem chris_score  (h1 : s = g + 60) (h2 : (s + g) / 2 = 110) (h3 : c = 110 * 120 / 100) :
  c = 132 := by
  sorry

end chris_score_l192_192372


namespace ratio_new_radius_l192_192388

theorem ratio_new_radius (r R h : ℝ) (h₀ : π * r^2 * h = 6) (h₁ : π * R^2 * h = 186) : R / r = Real.sqrt 31 :=
by
  sorry

end ratio_new_radius_l192_192388


namespace find_f_at_2_l192_192744

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h1 : ∀ x, f x = x^3 + 3 * x * f'' 2)
variable (h2 : f' 2 = 12 + 3 * f' 2)

theorem find_f_at_2 : f 2 = -28 :=
by
  sorry

end find_f_at_2_l192_192744


namespace geometric_progression_common_ratio_l192_192914

-- Define the problem conditions in Lean 4
theorem geometric_progression_common_ratio (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
  (h_pos : ∀ n, a n > 0) 
  (h_rel : ∀ n, a n = (a (n + 1) + a (n + 2)) / 2 + 2 ) : 
  r = 1 :=
sorry

end geometric_progression_common_ratio_l192_192914


namespace ratio_of_diagonals_to_sides_l192_192378

-- Define the given parameters and formula
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem
theorem ratio_of_diagonals_to_sides (n : ℕ) (h : n = 5) : 
  (num_diagonals n) / n = 1 :=
by
  -- Proof skipped
  sorry

end ratio_of_diagonals_to_sides_l192_192378


namespace intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l192_192055

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + x^2 - 2 * a * x + a^2

-- Question Ⅰ
theorem intervals_of_monotonicity_when_a_eq_2 :
  (∀ x : ℝ, 0 < x ∧ x < (2 - Real.sqrt 2) / 2 → f x 2 > 0) ∧
  (∀ x : ℝ, (2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2 → f x 2 < 0) ∧
  (∀ x : ℝ, (2 + Real.sqrt 2) / 2 < x → f x 2 > 0) := sorry

-- Question Ⅱ
theorem no_increasing_intervals_on_1_3_implies_a_ge_19_over_6 (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 0) → a ≥ (19 / 6) := sorry

end intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l192_192055


namespace Rockham_Soccer_League_members_l192_192040

theorem Rockham_Soccer_League_members (sock_cost tshirt_cost cap_cost total_cost members : ℕ) (h1 : sock_cost = 6) (h2 : tshirt_cost = sock_cost + 10) (h3 : cap_cost = 3) (h4 : total_cost = 4620) (h5 : total_cost = 50 * members) : members = 92 :=
by
  sorry

end Rockham_Soccer_League_members_l192_192040


namespace sum_interior_angles_equal_diagonals_l192_192798

theorem sum_interior_angles_equal_diagonals (n : ℕ) (h : n = 4 ∨ n = 5) :
  (n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540 :=
by sorry

end sum_interior_angles_equal_diagonals_l192_192798


namespace average_gas_mileage_round_trip_l192_192708

noncomputable def average_gas_mileage
  (d1 d2 : ℕ) (m1 m2 : ℕ) : ℚ :=
  let total_distance := d1 + d2
  let total_fuel := (d1 / m1) + (d2 / m2)
  total_distance / total_fuel

theorem average_gas_mileage_round_trip :
  average_gas_mileage 150 180 25 15 = 18.3 := by
  sorry

end average_gas_mileage_round_trip_l192_192708


namespace power_of_power_l192_192571

theorem power_of_power (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := 
  sorry

end power_of_power_l192_192571


namespace elena_total_pens_l192_192579

theorem elena_total_pens (price_x price_y total_cost : ℝ) (num_x : ℕ) (hx1 : price_x = 4.0) (hx2 : price_y = 2.2) 
  (hx3 : total_cost = 42.0) (hx4 : num_x = 6) : 
  ∃ num_total : ℕ, num_total = 14 :=
by
  sorry

end elena_total_pens_l192_192579


namespace angle_x_in_triangle_l192_192578

theorem angle_x_in_triangle :
  ∀ (x : ℝ), x + 2 * x + 50 = 180 → x = 130 / 3 :=
by
  intro x h
  sorry

end angle_x_in_triangle_l192_192578


namespace closest_integer_to_cube_root_of_250_l192_192119

theorem closest_integer_to_cube_root_of_250 : 
  let a := 6 
  in (abs ((a: ℤ) ^ 3 - 250) < abs ((a - 1) ^ 3 - 250)) 
     ∧ (abs ((a: ℤ) ^ 3 - 250) < abs ((a + 1) ^ 3 - 250)) → 
     a = 6 :=
by 
  simp [abs]
  sorry

end closest_integer_to_cube_root_of_250_l192_192119


namespace maximize_area_CDFE_l192_192047

-- Given the side lengths of the rectangle
def AB : ℝ := 2
def AD : ℝ := 1

-- Definitions for points E and F
def AE (x : ℝ) : ℝ := x
def AF (x : ℝ) : ℝ := x

-- The formula for the area of quadrilateral CDFE
def area_CDFE (x : ℝ) : ℝ := 
  0.5 * x * (3 - 2 * x)

theorem maximize_area_CDFE : 
  ∃ x : ℝ, x = 3 / 4 ∧ area_CDFE x = 9 / 16 :=
by 
  sorry

end maximize_area_CDFE_l192_192047


namespace total_pumpkin_weight_l192_192059

-- Conditions
def weight_first_pumpkin : ℝ := 4
def weight_second_pumpkin : ℝ := 8.7

-- Statement
theorem total_pumpkin_weight :
  weight_first_pumpkin + weight_second_pumpkin = 12.7 :=
by
  -- Proof can be done manually or via some automation here
  sorry

end total_pumpkin_weight_l192_192059


namespace numerator_equals_denominator_l192_192175

theorem numerator_equals_denominator (x : ℝ) (h : 4 * x - 3 = 5 * x + 2) : x = -5 :=
  by
    sorry

end numerator_equals_denominator_l192_192175


namespace geometric_sequence_fifth_term_l192_192289

theorem geometric_sequence_fifth_term
  (a : ℕ) (r : ℕ)
  (h₁ : a = 3)
  (h₂ : a * r^3 = 243) :
  a * r^4 = 243 :=
by
  sorry

end geometric_sequence_fifth_term_l192_192289


namespace evaluate_64_pow_7_over_6_l192_192001

theorem evaluate_64_pow_7_over_6 : (64 : ℝ)^(7 / 6) = 128 := by
  have h : (64 : ℝ) = 2^6 := by norm_num
  rw [h]
  norm_num
  sorry

end evaluate_64_pow_7_over_6_l192_192001


namespace simplify_fraction_product_l192_192648

theorem simplify_fraction_product : 
  (256 / 20 : ℚ) * (10 / 160) * ((16 / 6) ^ 2) = 256 / 45 :=
by norm_num

end simplify_fraction_product_l192_192648


namespace journey_time_l192_192270

theorem journey_time
  (t_1 t_2 : ℝ)
  (h1 : t_1 + t_2 = 5)
  (h2 : 40 * t_1 + 60 * t_2 = 240) :
  t_1 = 3 :=
sorry

end journey_time_l192_192270


namespace hyperbola_eccentricity_range_l192_192252

theorem hyperbola_eccentricity_range {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end hyperbola_eccentricity_range_l192_192252


namespace divisible_by_five_solution_exists_l192_192123

theorem divisible_by_five_solution_exists
  (a b c d : ℤ)
  (h₀ : ∃ k : ℤ, d = 5 * k + d % 5 ∧ d % 5 ≠ 0)
  (h₁ : ∃ n : ℤ, (a * n^3 + b * n^2 + c * n + d) % 5 = 0) :
  ∃ m : ℤ, (a + b * m + c * m^2 + d * m^3) % 5 = 0 := 
sorry

end divisible_by_five_solution_exists_l192_192123


namespace solve_quadratic_equation_l192_192254

theorem solve_quadratic_equation (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end solve_quadratic_equation_l192_192254


namespace vector_dot_product_l192_192341

-- Definitions based on the given conditions
variables (A B C M : ℝ)  -- points in 2D or 3D space can be generalized as real numbers for simplicity
variables (BA BC BM : ℝ) -- vector magnitudes
variables (AC : ℝ) -- magnitude of AC

-- Hypotheses from the problem conditions
variable (hM : 2 * BM = BA + BC)  -- M is the midpoint of AC
variable (hAC : AC = 4)
variable (hBM : BM = 3)

-- Theorem statement asserting the desired result
theorem vector_dot_product :
  BA * BC = 5 :=
by {
  sorry
}

end vector_dot_product_l192_192341


namespace find_a5_l192_192041

variable {a : ℕ → ℝ}

-- Condition 1: {a_n} is an arithmetic sequence
def arithmetic_sequence (a: ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition 2: a1 + a9 = 10
axiom a1_a9_sum : a 1 + a 9 = 10

theorem find_a5 (h_arith : arithmetic_sequence a) : a 5 = 5 :=
by {
  sorry
}

end find_a5_l192_192041


namespace factor_poly_PQ_sum_l192_192460

theorem factor_poly_PQ_sum (P Q : ℝ) (h : (∀ x : ℝ, (x^2 + 3 * x + 4) * (x^2 + -3 * x + 4) = x^4 + P * x^2 + Q)) : P + Q = 15 :=
by
  sorry

end factor_poly_PQ_sum_l192_192460


namespace power_of_product_l192_192572

variable (x y : ℝ)

theorem power_of_product (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 :=
  sorry

end power_of_product_l192_192572


namespace max_value_at_one_l192_192526

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l192_192526


namespace min_S_l192_192065

variable {x y : ℝ}
def condition (x y : ℝ) : Prop := (4 * x^2 + 5 * x * y + 4 * y^2 = 5)
def S (x y : ℝ) : ℝ := x^2 + y^2
theorem min_S (hx : condition x y) : S x y = (10 / 13) :=
sorry

end min_S_l192_192065


namespace sum_of_digits_10pow97_minus_97_l192_192196

-- Define a function that computes the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main statement we want to prove
theorem sum_of_digits_10pow97_minus_97 :
  sum_of_digits (10^97 - 97) = 858 :=
by
  sorry

end sum_of_digits_10pow97_minus_97_l192_192196


namespace solve_a_solve_inequality_solution_set_l192_192338

theorem solve_a (a : ℝ) :
  (∀ x : ℝ, (1 / 2 < x ∧ x < 2) ↔ ax^2 + 5 * x - 2 > 0) →
  a = -2 :=
by
  sorry

theorem solve_inequality_solution_set (x : ℝ) :
  (a = -2) →
  (2 * x^2 + 5 * x - 3 < 0) ↔
  (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end solve_a_solve_inequality_solution_set_l192_192338


namespace range_of_c_over_a_l192_192458

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + 2 * b + c = 0) :
    -3 < c / a ∧ c / a < -(1 / 3) := 
sorry

end range_of_c_over_a_l192_192458


namespace complex_power_identity_l192_192760

theorem complex_power_identity (w : ℂ) (h : w + w⁻¹ = 2) : w^(2022 : ℕ) + (w⁻¹)^(2022 : ℕ) = 2 := by
  sorry

end complex_power_identity_l192_192760


namespace final_exam_mean_score_l192_192739

theorem final_exam_mean_score (μ σ : ℝ) 
  (h1 : 55 = μ - 1.5 * σ)
  (h2 : 75 = μ - 2 * σ)
  (h3 : 85 = μ + 1.5 * σ)
  (h4 : 100 = μ + 3.5 * σ) :
  μ = 115 :=
by
  sorry

end final_exam_mean_score_l192_192739


namespace square_assembly_possible_l192_192155

theorem square_assembly_possible (Area1 Area2 Area3 : ℕ) (h1 : Area1 = 29) (h2 : Area2 = 18) (h3 : Area3 = 10) (h_total : Area1 + Area2 + Area3 = 57) : 
  ∃ s : ℝ, s^2 = 57 ∧ true :=
by
  sorry

end square_assembly_possible_l192_192155


namespace popsicle_sticks_left_l192_192366

theorem popsicle_sticks_left (initial_sticks given_per_group groups : ℕ) 
  (h_initial : initial_sticks = 170)
  (h_given : given_per_group = 15)
  (h_groups : groups = 10) : 
  initial_sticks - (given_per_group * groups) = 20 := by
  rw [h_initial, h_given, h_groups]
  norm_num
  sorry -- Alternatively: exact eq.refl 20

end popsicle_sticks_left_l192_192366


namespace lattice_points_in_5_sphere_l192_192904

theorem lattice_points_in_5_sphere :
  (Finset.card {p : ℤ × ℤ × ℤ × ℤ × ℤ | p.1^2 + p.2.1^2 + p.2.2.1^2 + p.2.2.2.1^2 + p.2.2.2.2^2 ≤ 9}) = 1343 := sorry

end lattice_points_in_5_sphere_l192_192904


namespace mutual_independence_of_A_and_D_l192_192816

noncomputable theory

variables (Ω : Type) [ProbabilitySpace Ω]
-- Definition of events A, B, C, D as sets over Ω
def event_A : Event Ω := {ω | some_condition_for_A}
def event_B : Event Ω := {ω | some_condition_for_B}
def event_C : Event Ω := {ω | some_condition_for_C}
def event_D : Event Ω := {ω | some_condition_for_D}

-- Given probabilities
axiom P_A : P(event_A Ω) = 1 / 6
axiom P_B : P(event_B Ω) = 1 / 6
axiom P_C : P(event_C Ω) = 5 / 36
axiom P_D : P(event_D Ω) = 1 / 6

-- Independence definition
def are_independent (X Y : Event Ω) : Prop :=
  P(X ∩ Y) = P(X) * P(Y)

-- The problem statement: proving A and D are independent
theorem mutual_independence_of_A_and_D : are_independent Ω (event_A Ω) (event_D Ω) :=
sorry

end mutual_independence_of_A_and_D_l192_192816


namespace basketball_court_width_l192_192795

variable (width length : ℕ)

-- Given conditions
axiom h1 : length = width + 14
axiom h2 : 2 * length + 2 * width = 96

-- Prove the width is 17 meters
theorem basketball_court_width : width = 17 :=
by {
  sorry
}

end basketball_court_width_l192_192795


namespace lateral_surface_area_of_cone_l192_192184

theorem lateral_surface_area_of_cone (diameter height : ℝ) (h_d : diameter = 2) (h_h : height = 2) :
  let radius := diameter / 2
  let slant_height := Real.sqrt (radius ^ 2 + height ^ 2)
  π * radius * slant_height = Real.sqrt 5 * π := 
  by
    sorry

end lateral_surface_area_of_cone_l192_192184


namespace reinforcement_size_l192_192707

theorem reinforcement_size (R : ℕ) : 
  2000 * 39 = (2000 + R) * 20 → R = 1900 :=
by
  intro h
  sorry

end reinforcement_size_l192_192707


namespace customers_tipped_count_l192_192714

variable (initial_customers : ℕ)
variable (added_customers : ℕ)
variable (customers_no_tip : ℕ)

def total_customers (initial_customers added_customers : ℕ) : ℕ :=
  initial_customers + added_customers

theorem customers_tipped_count 
  (h_init : initial_customers = 29)
  (h_added : added_customers = 20)
  (h_no_tip : customers_no_tip = 34) :
  (total_customers initial_customers added_customers - customers_no_tip) = 15 :=
by
  sorry

end customers_tipped_count_l192_192714


namespace elena_bread_max_flour_l192_192871

variable (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
variable (available_butter available_sugar : ℕ)

def max_flour (butter_per_cup_flour butter sugar_per_cup_flour sugar : ℕ)
  (available_butter available_sugar : ℕ) : ℕ :=
  min (available_butter * sugar / butter_per_cup_flour) (available_sugar * butter / sugar_per_cup_flour)

theorem elena_bread_max_flour : 
  max_flour 3 4 2 5 24 30 = 32 := sorry

end elena_bread_max_flour_l192_192871


namespace miss_davis_sticks_left_l192_192365

theorem miss_davis_sticks_left (initial_sticks groups_per_class sticks_per_group : ℕ) 
(h1 : initial_sticks = 170) (h2 : groups_per_class = 10) (h3 : sticks_per_group = 15) : 
initial_sticks - (groups_per_class * sticks_per_group) = 20 :=
by sorry

end miss_davis_sticks_left_l192_192365


namespace smallest_sector_angle_24_l192_192924

theorem smallest_sector_angle_24
  (a : ℕ) (d : ℕ)
  (h1 : ∀ i, i < 8 → ((a + i * d) : ℤ) > 0)
  (h2 : (2 * a + 7 * d = 90)) : a = 24 :=
by
  sorry

end smallest_sector_angle_24_l192_192924


namespace three_rays_with_common_point_l192_192048

theorem three_rays_with_common_point (x y : ℝ) :
  (∃ (common : ℝ), ((5 = x - 1 ∧ y + 3 ≤ 5) ∨ 
                     (5 = y + 3 ∧ x - 1 ≤ 5) ∨ 
                     (x - 1 = y + 3 ∧ 5 ≤ x - 1 ∧ 5 ≤ y + 3)) 
  ↔ ((x = 6 ∧ y ≤ 2) ∨ (y = 2 ∧ x ≤ 6) ∨ (y = x - 4 ∧ x ≥ 6))) :=
sorry

end three_rays_with_common_point_l192_192048


namespace difference_of_squares_l192_192381

noncomputable def product_of_consecutive_integers (n : ℕ) := n * (n + 1)

theorem difference_of_squares (h : ∃ n : ℕ, product_of_consecutive_integers n = 2720) :
  ∃ a b : ℕ, product_of_consecutive_integers a = 2720 ∧ (b = a + 1) ∧ (b * b - a * a = 103) :=
by
  sorry

end difference_of_squares_l192_192381


namespace value_of_x_l192_192189

theorem value_of_x (x : ℕ) (M : Set ℕ) :
  M = {0, 1, 2} →
  M ∪ {x} = {0, 1, 2, 3} →
  x = 3 :=
by
  sorry

end value_of_x_l192_192189


namespace inequality_holds_l192_192235

-- Define parameters for the problem
variables (p q x y z : ℝ) (n : ℕ)

-- Define the conditions on x, y, and z
def condition1 : Prop := y = x^n + p*x + q
def condition2 : Prop := z = y^n + p*y + q
def condition3 : Prop := x = z^n + p*z + q

-- Define the statement of the inequality
theorem inequality_holds (h1 : condition1 p q x y n) (h2 : condition2 p q y z n) (h3 : condition3 p q x z n):
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y :=
sorry

end inequality_holds_l192_192235


namespace trig_identity_l192_192886

theorem trig_identity (α : ℝ) (h : Real.tan α = 1/3) :
  Real.cos α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10 := 
sorry

end trig_identity_l192_192886


namespace francis_violin_count_l192_192438

theorem francis_violin_count :
  let ukuleles := 2
  let guitars := 4
  let ukulele_strings := 4
  let guitar_strings := 6
  let violin_strings := 4
  let total_strings := 40
  ∃ (violins: ℕ), violins = 2 := by
    sorry

end francis_violin_count_l192_192438


namespace find_remainder_of_n_l192_192889

theorem find_remainder_of_n (n k d : ℕ) (hn_pos : n > 0) (hk_pos : k > 0) (hd_pos_digits : d < 10^k) 
  (h : n * 10^k + d = n * (n + 1) / 2) : n % 9 = 1 :=
sorry

end find_remainder_of_n_l192_192889


namespace find_f_prime_at_2_l192_192541

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * log x + b / x
noncomputable def f_der (x : ℝ) (a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_2 (a b : ℝ) (h₁ : f 1 a b = -2)
  (h₂ : f_der 1 a b = 0) : f_der 2 a b = -1 / 2 :=
by
    sorry

end find_f_prime_at_2_l192_192541


namespace meeting_point_distance_proof_l192_192265

theorem meeting_point_distance_proof :
  ∃ x t : ℕ, (5 * t + (t * (7 + t) / 2) = 85) ∧ x = 9 :=
by
  sorry

end meeting_point_distance_proof_l192_192265


namespace division_problem_solution_l192_192100

theorem division_problem_solution (x : ℝ) (h : (2.25 / x) * 12 = 9) : x = 3 :=
sorry

end division_problem_solution_l192_192100


namespace value_of_expression_l192_192126

theorem value_of_expression (n : ℕ) (a : ℝ) (h1 : 6 * 11 * n ≠ 0) (h2 : a ^ (2 * n) = 5) : 2 * a ^ (6 * n) - 4 = 246 :=
by
  sorry

end value_of_expression_l192_192126


namespace student_correct_ans_l192_192400

theorem student_correct_ans (c w : ℕ) (h1 : c + w = 80) (h2 : 4 * c - w = 120) : c = 40 :=
by
  sorry

end student_correct_ans_l192_192400


namespace length_BD_l192_192642

/-- Points A, B, C, and D lie on a line in that order. We are given:
  AB = 2 cm,
  AC = 5 cm, and
  CD = 3 cm.
Then, we need to show that the length of BD is 6 cm. -/
theorem length_BD :
  ∀ (A B C D : ℕ),
  A + B = 2 → A + C = 5 → C + D = 3 →
  D - B = 6 :=
by
  intros A B C D h1 h2 h3
  -- Proof steps to be filled in
  sorry

end length_BD_l192_192642


namespace numBaskets_l192_192304

noncomputable def numFlowersInitial : ℕ := 5 + 5
noncomputable def numFlowersAfterGrowth : ℕ := numFlowersInitial + 20
noncomputable def numFlowersFinal : ℕ := numFlowersAfterGrowth - 10
noncomputable def flowersPerBasket : ℕ := 4

theorem numBaskets : numFlowersFinal / flowersPerBasket = 5 := 
by
  sorry

end numBaskets_l192_192304


namespace score_comparison_l192_192469

theorem score_comparison :
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  combined_score - opponent_score = 143 :=
by
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  sorry

end score_comparison_l192_192469


namespace distance_between_parallel_lines_l192_192507

theorem distance_between_parallel_lines (a d : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 3 = 0 ∧ a * x - y + 4 = 0 → (2 = a ∧ d = |(3 - 4)| / Real.sqrt (2 ^ 2 + (-1) ^ 2))) → 
  (a = 2 ∧ d = Real.sqrt 5 / 5) :=
by 
  sorry

end distance_between_parallel_lines_l192_192507


namespace track_champion_races_l192_192345

theorem track_champion_races (total_sprinters : ℕ) (lanes : ℕ) (eliminations_per_race : ℕ)
  (h1 : total_sprinters = 216) (h2 : lanes = 6) (h3 : eliminations_per_race = 5) : 
  (total_sprinters - 1) / eliminations_per_race = 43 :=
by
  -- We acknowledge that a proof is needed here. Placeholder for now.
  sorry

end track_champion_races_l192_192345


namespace difference_of_numbers_l192_192258

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 34800) (h2 : b % 25 = 0) (h3 : b / 100 = a) : b - a = 32112 := by
  sorry

end difference_of_numbers_l192_192258


namespace pyramid_side_length_difference_l192_192058

theorem pyramid_side_length_difference (x : ℕ) (h1 : 1 + x^2 + (x + 1)^2 + (x + 2)^2 = 30) : x = 2 :=
by
  sorry

end pyramid_side_length_difference_l192_192058


namespace quincy_sold_more_than_jake_l192_192495

theorem quincy_sold_more_than_jake :
  ∀ (T Jake : ℕ), Jake = 2 * T + 15 → 4000 = 100 * (T + Jake) → 4000 - Jake = 3969 :=
by
  intros T Jake hJake hQuincy
  sorry

end quincy_sold_more_than_jake_l192_192495


namespace minimum_area_triangle_ABC_l192_192551

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0,0)
def B : ℤ × ℤ := (30,18)

-- Define a function to calculate the area of the triangle using the Shoelace formula
def area_of_triangle (A B C : ℤ × ℤ) : ℤ := 15 * (C.2).natAbs

-- State the theorem
theorem minimum_area_triangle_ABC : 
  ∀ C : ℤ × ℤ, C ≠ (0,0) → area_of_triangle A B C ≥ 15 :=
by
  sorry -- Skip the proof

end minimum_area_triangle_ABC_l192_192551


namespace three_layer_rug_area_l192_192176

theorem three_layer_rug_area 
  (A B C D : ℕ) 
  (hA : A = 350) 
  (hB : B = 250) 
  (hC : C = 45) 
  (h_formula : A = B + C + D) : 
  D = 55 :=
by
  sorry

end three_layer_rug_area_l192_192176


namespace solve_for_y_l192_192973

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l192_192973


namespace figure_can_form_square_l192_192154

-- Define the problem statement in Lean
theorem figure_can_form_square (n : ℕ) (h : is_perfect_square n) : 
  ∃ (parts : list (set (ℕ × ℕ))), parts.length = 3 ∧ 
  (⋃ p in parts, p) = set.univ.filter (fun i => i.1 * i.2 < n) ∧ 
  (∃ k : ℕ, is_square k = some (sqrt n) ∧ 
  (⋃ p in parts, ∃ (x_shift y_shift : ℤ), (λ (i : ℕ × ℕ), (i.1 + x_shift, i.2 + y_shift)) '' p = set.univ.filter (fun i => i.1 < k ∧ i.2 < k))) :=
sorry

end figure_can_form_square_l192_192154


namespace compare_powers_l192_192159

theorem compare_powers:
  (2 ^ 2023) * (7 ^ 2023) < (3 ^ 2023) * (5 ^ 2023) :=
  sorry

end compare_powers_l192_192159


namespace problem_divisibility_l192_192224

theorem problem_divisibility (k : ℕ) (hk : k > 1) (p : ℕ) (hp : p = 6 * k + 1) (hprime : Prime p) 
  (m : ℕ) (hm : m = 2^p - 1) : 
  127 * m ∣ 2^(m - 1) - 1 := 
sorry

end problem_divisibility_l192_192224


namespace solve_y_l192_192970

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l192_192970


namespace complex_number_solution_l192_192479

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (hz : z * (i - 1) = 2 * i) : 
z = 1 - i :=
by 
  sorry

end complex_number_solution_l192_192479


namespace remainder_of_16_pow_2048_mod_11_l192_192986

theorem remainder_of_16_pow_2048_mod_11 : (16^2048) % 11 = 4 := by
  sorry

end remainder_of_16_pow_2048_mod_11_l192_192986


namespace closest_integer_to_cube_root_of_250_l192_192115

def is_closer_to (n m k : ℤ) (x : ℝ) : Prop :=
  abs (x - m) < abs (x - k)

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, n = 6 ↔ ∃ m k : ℤ, m = 6 ∧ k ≠ 6 ∧ 
    is_closer_to 250 m k (real.cbrt 250) := sorry

end closest_integer_to_cube_root_of_250_l192_192115


namespace ratio_a_over_b_l192_192186

-- Definitions of conditions
def func (a b x : ℝ) : ℝ := a * x^2 + b
def derivative (a b x : ℝ) : ℝ := 2 * a * x

-- Given conditions
variables (a b : ℝ)
axiom tangent_slope : derivative a b 1 = 2
axiom point_on_graph : func a b 1 = 3

-- Statement to prove
theorem ratio_a_over_b : a / b = 1 / 2 :=
by sorry

end ratio_a_over_b_l192_192186


namespace fourth_derivative_at_0_l192_192930

noncomputable def f : ℝ → ℝ := sorry

axiom f_at_0 : f 0 = 1
axiom f_prime_at_0 : deriv f 0 = 2
axiom f_double_prime : ∀ t, deriv (deriv f) t = 4 * deriv f t - 3 * f t + 1

-- We want to prove that the fourth derivative of f at 0 equals 54
theorem fourth_derivative_at_0 : deriv (deriv (deriv (deriv f))) 0 = 54 :=
sorry

end fourth_derivative_at_0_l192_192930


namespace sum_of_squares_l192_192386

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 28) : x^2 + y^2 = 200 :=
by
  sorry

end sum_of_squares_l192_192386


namespace meal_combinations_l192_192399

def menu_items : ℕ := 12
def special_dish_chosen : Prop := true

theorem meal_combinations : (special_dish_chosen → (menu_items - 1) * (menu_items - 1) = 121) :=
by
  sorry

end meal_combinations_l192_192399


namespace sum_of_abc_is_40_l192_192027

theorem sum_of_abc_is_40 (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * b + c = 55) (h2 : b * c + a = 55) (h3 : c * a + b = 55) :
    a + b + c = 40 :=
by
  sorry

end sum_of_abc_is_40_l192_192027


namespace closest_integer_to_cube_root_of_250_l192_192121

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l192_192121


namespace dishonest_dealer_profit_percent_l192_192288

theorem dishonest_dealer_profit_percent
  (C : ℝ) -- assumed cost price for 1 kg of goods
  (SP_600 : ℝ := C) -- selling price for 600 grams is equal to the cost price for 1 kg
  (CP_600 : ℝ := 0.6 * C) -- cost price for 600 grams
  : (SP_600 - CP_600) / CP_600 * 100 = 66.67 := by
  sorry

end dishonest_dealer_profit_percent_l192_192288


namespace train_length_l192_192685

theorem train_length (speed_fast speed_slow : ℝ) (time_pass : ℝ)
  (L : ℝ)
  (hf : speed_fast = 46 * (1000/3600))
  (hs : speed_slow = 36 * (1000/3600))
  (ht : time_pass = 36)
  (hL : (2 * L = (speed_fast - speed_slow) * time_pass)) :
  L = 50 := by
  sorry

end train_length_l192_192685


namespace midpoint_count_bounds_l192_192442

theorem midpoint_count_bounds (n : ℕ) (h : n ≥ 2) :
  2 * n - 3 ≤ ∑ i in (Finset.range (n)).powerset.filter (λ s, s.card = 2), (by simp : ℤ) ≤ (n * (n - 1)) / 2 :=
sorry

end midpoint_count_bounds_l192_192442


namespace tables_count_l192_192407

theorem tables_count (c t : Nat) (h1 : c = 8 * t) (h2 : 3 * c + 5 * t = 580) : t = 20 :=
by
  sorry

end tables_count_l192_192407


namespace range_of_a_plus_b_l192_192216

noncomputable def range_of_sum_of_sides (a b : ℝ) (c : ℝ) : Prop :=
  (2 < a + b ∧ a + b ≤ 4)

theorem range_of_a_plus_b
  (a b c : ℝ)
  (h1 : (2 * (b ^ 2 - (1/2) * a * b) = b ^ 2 + 4 - a ^ 2))
  (h2 : c = 2) :
  range_of_sum_of_sides a b c :=
by
  -- Proof would go here, but it's omitted as per the instructions.
  sorry

end range_of_a_plus_b_l192_192216


namespace simplify_expression_l192_192240

theorem simplify_expression :
  (↑(Real.sqrt 648) / ↑(Real.sqrt 81) - ↑(Real.sqrt 245) / ↑(Real.sqrt 49)) = 2 * Real.sqrt 2 - Real.sqrt 5 := by
  -- proof omitted
  sorry

end simplify_expression_l192_192240


namespace total_apples_count_l192_192805

-- Definitions based on conditions
def red_apples := 16
def green_apples := red_apples + 12
def total_apples := green_apples + red_apples

-- Statement to prove
theorem total_apples_count : total_apples = 44 := 
by
  sorry

end total_apples_count_l192_192805


namespace probability_two_black_balls_l192_192131

theorem probability_two_black_balls (white_balls black_balls drawn_balls : ℕ) 
  (h_w : white_balls = 4) (h_b : black_balls = 7) (h_d : drawn_balls = 2) :
  let total_ways := Nat.choose (white_balls + black_balls) drawn_balls
  let black_ways := Nat.choose black_balls drawn_balls
  (black_ways / total_ways : ℚ) = 21 / 55 :=
by
  sorry

end probability_two_black_balls_l192_192131


namespace polygon_sides_l192_192089

theorem polygon_sides (n : ℕ) 
  (h₁ : (n - 2) * 180 + 360 = 1800) : 
  n = 10 :=
begin
  sorry
end

end polygon_sides_l192_192089


namespace average_goals_is_92_l192_192464

-- Definitions based on conditions
def layla_goals : ℕ := 104
def kristin_fewer_goals : ℕ := 24
def kristin_goals : ℕ := layla_goals - kristin_fewer_goals
def combined_goals : ℕ := layla_goals + kristin_goals
def average_goals : ℕ := combined_goals / 2

-- Theorem
theorem average_goals_is_92 : average_goals = 92 := 
  sorry

end average_goals_is_92_l192_192464


namespace glasses_per_pitcher_l192_192158

def total_glasses : Nat := 30
def num_pitchers : Nat := 6

theorem glasses_per_pitcher : total_glasses / num_pitchers = 5 := by
  sorry

end glasses_per_pitcher_l192_192158


namespace probability_different_plants_l192_192840

-- Define the types of plants as an enum
inductive Plant
| Pothos
| LuckyBamboo
| Jade
| Aloe

open Plant

def all_pairs (pl1 pl2 : Plant) :=
  [(Pothos, Pothos), (Pothos, LuckyBamboo), (Pothos, Jade), (Pothos, Aloe),
   (LuckyBamboo, Pothos), (LuckyBamboo, LuckyBamboo), (LuckyBamboo, Jade), (LuckyBamboo, Aloe),
   (Jade, Pothos), (Jade, LuckyBamboo), (Jade, Jade), (Jade, Aloe),
   (Aloe, Pothos), (Aloe, LuckyBamboo), (Aloe, Jade), (Aloe, Aloe)]

-- Condition: total number of pairs
def total_pairs : ℕ := 16

-- Condition: same plant pairs
def same_plant_pairs : List (Plant × Plant) :=
  [ (Pothos, Pothos), (LuckyBamboo, LuckyBamboo), (Jade, Jade), (Aloe, Aloe) ]

-- Theorem statement (proof omitted)
theorem probability_different_plants: 
  (total_pairs - List.length same_plant_pairs) / total_pairs = 13 / 16 := by
  sorry

end probability_different_plants_l192_192840


namespace probability_of_passing_through_correct_l192_192696

def probability_of_passing_through (n k : ℕ) : ℚ :=
(2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_of_passing_through_correct (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  probability_of_passing_through n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := 
by
  sorry

end probability_of_passing_through_correct_l192_192696


namespace jane_played_rounds_l192_192036

-- Define the conditions
def points_per_round := 10
def points_ended_with := 60
def points_lost := 20

-- Define the proof problem
theorem jane_played_rounds : (points_ended_with + points_lost) / points_per_round = 8 :=
by
  sorry

end jane_played_rounds_l192_192036


namespace find_B_l192_192248

theorem find_B (N : ℕ) (A B : ℕ) (H1 : N = 757000000 + A * 10000 + B * 1000 + 384) (H2 : N % 357 = 0) : B = 5 :=
sorry

end find_B_l192_192248


namespace matrix_condition_l192_192888

open Matrix

variable {R : Type*} [Field R]

noncomputable def matrix_B := λ (p q r s : R), matrix![
  [p, q],
  [r, s]
]

theorem matrix_condition 
(p q r s : R) (h : (matrix_B p q r s)ᵀ = 2 * (matrix_B p q r s)⁻¹) :
  p^2 + q^2 + r^2 + s^2 = 1 :=
sorry

end matrix_condition_l192_192888


namespace sum_of_squares_l192_192940

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 110) : x^2 + y^2 = 1380 := 
by sorry

end sum_of_squares_l192_192940


namespace nat_square_not_div_factorial_l192_192587

-- Define n as a natural number
def n : Nat := sorry  -- We assume n is given somewhere

-- Define a function to check if a number is prime
def is_prime (p : Nat) : Prop := sorry  -- Placeholder for prime checking function

-- The main theorem to prove
theorem nat_square_not_div_factorial (n : Nat) : (n = 4 ∨ is_prime n) → ¬ ((n * n) ∣ Nat.factorial n) := by
  sorry

end nat_square_not_div_factorial_l192_192587


namespace probability_of_winning_l192_192346

open Nat

theorem probability_of_winning (h : True) : 
  let num_cards := 3
  let num_books := 5
  (1 - (Nat.choose num_cards 2 * 2^num_books - num_cards) / num_cards^num_books) = 50 / 81 := sorry

end probability_of_winning_l192_192346


namespace geometric_sequence_terms_l192_192017

theorem geometric_sequence_terms
  (a_3 : ℝ) (a_4 : ℝ)
  (h1 : a_3 = 12)
  (h2 : a_4 = 18) :
  ∃ (a_1 a_2 : ℝ) (q: ℝ), 
    a_1 = 16 / 3 ∧ a_2 = 8 ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 := 
by
  sorry

end geometric_sequence_terms_l192_192017


namespace problem_equivalent_l192_192197

theorem problem_equivalent :
  ∀ m n : ℤ, |m - n| = n - m ∧ |m| = 4 ∧ |n| = 3 → m + n = -1 ∨ m + n = -7 :=
by
  intros m n h
  have h1 : |m - n| = n - m := h.1
  have h2 : |m| = 4 := h.2.1
  have h3 : |n| = 3 := h.2.2
  sorry

end problem_equivalent_l192_192197


namespace speed_conversion_l192_192002

-- Define the conversion factor
def conversion_factor := 3.6

-- Define the given speed in meters per second
def speed_mps := 16.668

-- Define the expected speed in kilometers per hour
def expected_speed_kmph := 60.0048

-- The theorem to prove that the given speed in m/s converts to the expected speed in km/h
theorem speed_conversion : speed_mps * conversion_factor = expected_speed_kmph := 
  by
    sorry

end speed_conversion_l192_192002


namespace find_m_max_value_l192_192334

noncomputable def f (x : ℝ) := |x - 1|

theorem find_m (m : ℝ) :
  (∀ x, f (x + 5) ≤ 3 * m) ∧ m > 0 ∧ (∀ x, -7 ≤ x ∧ x ≤ -1 → f (x + 5) ≤ 3 * m) →
  m = 1 :=
by
  sorry

theorem max_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h2 : 2 * a ^ 2 + b ^ 2 = 3) :
  ∃ x, (∀ a b, 2 * a * Real.sqrt (1 + b ^ 2) ≤ x) ∧ x = 2 * Real.sqrt 2 :=
by
  sorry

end find_m_max_value_l192_192334


namespace sqrt_sum_simplify_l192_192960

theorem sqrt_sum_simplify : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 :=
by sorry

end sqrt_sum_simplify_l192_192960


namespace perp_tangents_l192_192748

theorem perp_tangents (a b : ℝ) (h : a + b = 5) (tangent_perp : ∀ x y : ℝ, x = 1 ∧ y = 1) :
  a / b = 1 / 3 :=
sorry

end perp_tangents_l192_192748


namespace probability_of_picking_combination_is_0_4_l192_192858

noncomputable def probability_at_least_19_rubles (total_coins total_value: ℕ) :=
  let coins := [10, 10, 5, 5, 2] in
  let all_combinations := (Finset.powersetLen 3 (coins.to_finset)).to_list in
  let favorable_combinations := all_combinations.filter (fun c => c.sum ≥ total_value) in
  (favorable_combinations.length : ℚ) / (all_combinations.length : ℚ)

theorem probability_of_picking_combination_is_0_4 :
  probability_at_least_19_rubles 5 19 = 0.4 :=
by
  sorry

end probability_of_picking_combination_is_0_4_l192_192858


namespace max_students_on_field_trip_l192_192246

theorem max_students_on_field_trip 
  (bus_cost : ℕ := 100)
  (bus_capacity : ℕ := 25)
  (student_admission_cost_high : ℕ := 10)
  (student_admission_cost_low : ℕ := 8)
  (discount_threshold : ℕ := 20)
  (teacher_cost : ℕ := 0)
  (budget : ℕ := 350) :
  max_students ≤ bus_capacity ↔ bus_cost + 
  (if max_students ≥ discount_threshold then max_students * student_admission_cost_low
  else max_students * student_admission_cost_high) 
   ≤ budget := 
sorry

end max_students_on_field_trip_l192_192246


namespace new_planet_volume_eq_l192_192472

noncomputable def volume_of_new_planet (V_earth : ℝ) (scaling_factor : ℝ) : ℝ :=
  V_earth * (scaling_factor^3)

theorem new_planet_volume_eq 
  (V_earth : ℝ)
  (scaling_factor : ℝ)
  (hV_earth : V_earth = 1.08 * 10^12)
  (h_scaling_factor : scaling_factor = 10^4) :
  volume_of_new_planet V_earth scaling_factor = 1.08 * 10^24 :=
by
  sorry

end new_planet_volume_eq_l192_192472


namespace correct_average_l192_192694

theorem correct_average 
(n : ℕ) (avg1 avg2 avg3 : ℝ): 
  n = 10 
  → avg1 = 40.2 
  → avg2 = avg1
  → avg3 = avg1
  → avg1 = avg3 :=
by 
  intros hn h_avg1 h_avg2 h_avg3
  sorry

end correct_average_l192_192694


namespace required_tents_l192_192931

def numberOfPeopleInMattFamily : ℕ := 1 + 2
def numberOfPeopleInBrotherFamily : ℕ := 1 + 1 + 4
def numberOfPeopleInUncleJoeFamily : ℕ := 1 + 1 + 3
def totalNumberOfPeople : ℕ := numberOfPeopleInMattFamily + numberOfPeopleInBrotherFamily + numberOfPeopleInUncleJoeFamily
def numberOfPeopleSleepingInHouse : ℕ := 4
def numberOfPeopleSleepingInTents : ℕ := totalNumberOfPeople - numberOfPeopleSleepingInHouse
def peoplePerTent : ℕ := 2

def numberOfTentsNeeded : ℕ :=
  numberOfPeopleSleepingInTents / peoplePerTent

theorem required_tents : numberOfTentsNeeded = 5 := by
  sorry

end required_tents_l192_192931


namespace ratio_of_inverse_l192_192977

theorem ratio_of_inverse (a b c d : ℝ) (h : ∀ x, (3 * (a * x + b) / (c * x + d) - 2) / ((a * x + b) / (c * x + d) + 4) = x) : 
  a / c = -4 :=
sorry

end ratio_of_inverse_l192_192977


namespace truncated_pyramid_properties_l192_192661

noncomputable def truncatedPyramidSurfaceArea
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the surface area function

noncomputable def truncatedPyramidVolume
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the volume function

theorem truncated_pyramid_properties
  (a b c : ℝ) (theta m : ℝ)
  (h₀ : a = 148) 
  (h₁ : b = 156) 
  (h₂ : c = 208) 
  (h₃ : theta = 112.62) 
  (h₄ : m = 27) :
  (truncatedPyramidSurfaceArea a b c theta m = 74352) ∧
  (truncatedPyramidVolume a b c theta m = 395280) :=
by
  sorry -- The actual proof will go here

end truncated_pyramid_properties_l192_192661


namespace inequality_abc_l192_192360

variable {a b c : ℝ}

-- Assume a, b, c are positive real numbers
def positive_real_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Assume the sum of any two numbers is greater than the third
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Lean 4 statement for the proof problem
theorem inequality_abc (h1 : positive_real_numbers a b c) (h2 : triangle_inequality a b c) :
  abc ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
sorry

end inequality_abc_l192_192360


namespace tiffany_ate_pies_l192_192736

theorem tiffany_ate_pies (baking_days : ℕ) (pies_per_day : ℕ) (wc_per_pie : ℕ) 
                         (remaining_wc : ℕ) (total_pies : ℕ) (total_wc : ℕ) :
  baking_days = 11 → pies_per_day = 3 → wc_per_pie = 2 → remaining_wc = 58 →
  total_pies = pies_per_day * baking_days → total_wc = total_pies * wc_per_pie →
  (total_wc - remaining_wc) / wc_per_pie = 4 :=
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tiffany_ate_pies_l192_192736


namespace probability_one_absent_other_present_l192_192913

noncomputable def prob_absent_present (absent_rate : ℚ) : ℚ :=
  let present_rate := 1 - absent_rate in
  (present_rate * absent_rate + absent_rate * present_rate) * 100

theorem probability_one_absent_other_present :
  prob_absent_present (1/20) = 9.5 :=
by
  sorry

end probability_one_absent_other_present_l192_192913


namespace sum_of_coefficients_l192_192433

-- Define the polynomial
def polynomial (x : ℝ) : ℝ :=
  2 * (4 * x ^ 8 + 7 * x ^ 6 - 9 * x ^ 3 + 3) + 6 * (x ^ 7 - 2 * x ^ 4 + 8 * x ^ 2 - 2)

-- State the theorem to prove the sum of the coefficients
theorem sum_of_coefficients : polynomial 1 = 40 :=
by
  sorry

end sum_of_coefficients_l192_192433


namespace original_selling_price_l192_192086

theorem original_selling_price (P : ℝ) (h : 0.7 * P = 560) : P = 800 :=
by
  sorry

end original_selling_price_l192_192086


namespace cara_younger_than_mom_l192_192865

noncomputable def cara_grandmothers_age : ℤ := 75
noncomputable def cara_moms_age := cara_grandmothers_age - 15
noncomputable def cara_age : ℤ := 40

theorem cara_younger_than_mom :
  cara_moms_age - cara_age = 20 := by
  sorry

end cara_younger_than_mom_l192_192865


namespace gcd_of_items_l192_192510

def numPens : ℕ := 891
def numPencils : ℕ := 810
def numNotebooks : ℕ := 1080
def numErasers : ℕ := 972

theorem gcd_of_items :
  Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numNotebooks) numErasers = 27 :=
by
  sorry

end gcd_of_items_l192_192510


namespace mean_equality_l192_192078

theorem mean_equality (y z : ℝ)
  (h : (14 + y + z) / 3 = (8 + 15 + 21) / 3)
  (hyz : y = z) :
  y = 15 ∧ z = 15 :=
by sorry

end mean_equality_l192_192078


namespace wall_building_problem_l192_192997

theorem wall_building_problem 
    (num_workers_1 : ℕ) (length_wall_1 : ℕ) (days_1 : ℕ)
    (num_workers_2 : ℕ) (length_wall_2 : ℕ) (days_2 : ℕ) :
    num_workers_1 = 8 → length_wall_1 = 140 → days_1 = 42 →
    num_workers_2 = 30 → length_wall_2 = 100 →
    (work_done : ℕ → ℕ → ℕ) → 
    (work_done length_wall_1 days_1 = num_workers_1 * days_1 * length_wall_1) →
    (work_done length_wall_2 days_2 = num_workers_2 * days_2 * length_wall_2) →
    (days_2 = 8) :=
by
  intros h1 h2 h3 h4 h5 wf wlen1 wlen2
  sorry

end wall_building_problem_l192_192997


namespace solve_inequality_l192_192497

-- Define conditions
def valid_x (x : ℝ) : Prop := x ≠ -3 ∧ x ≠ -8/3

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (-3 < x ∧ x < -8/3) ∨ ((1 - Real.sqrt 89) / 4 < x ∧ x < (1 + Real.sqrt 89) / 4)

-- Prove the equivalence
theorem solve_inequality (x : ℝ) (h : valid_x x) : inequality x ↔ solution_set x :=
by
  sorry

end solve_inequality_l192_192497


namespace paraboloid_area_first_octant_bounded_plane_y6_l192_192863

open RealMeasureTheory Set Filter

noncomputable def paraboloid_surface_area : ℝ :=
  let f x z := sqrt (1 + (4 * (x^2 + z^2)) / 9)
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1^2 + p.2^2 ≤ 18}
  (∫ region (λ p, f p.1 p.2)) * π / 2
-- Use improper integral calculus to derive the result
theorem paraboloid_area_first_octant_bounded_plane_y6 :
  paraboloid_surface_area = (39 * π) / 4 := by
  -- The proof is lengthy and involves proper change of variables and evaluation.
  sorry

end paraboloid_area_first_octant_bounded_plane_y6_l192_192863


namespace sum_of_interior_angles_of_special_regular_polygon_l192_192801

theorem sum_of_interior_angles_of_special_regular_polygon (n : ℕ) (h1 : n = 4 ∨ n = 5) :
  ((n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540) :=
by sorry

end sum_of_interior_angles_of_special_regular_polygon_l192_192801


namespace minimum_value_l192_192927

noncomputable def min_expression (a b : ℝ) : ℝ :=
  a^2 + b^2 + 1 / (a + b)^2 + 1 / (a^2 * b^2)

theorem minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 + 3 ∧ min_expression a b ≥ c :=
by
  use 2 * Real.sqrt 2 + 3
  sorry

end minimum_value_l192_192927


namespace total_weight_moved_l192_192981

theorem total_weight_moved (tom_weight : ℝ) (vest_fraction : ℝ) (hold_fraction : ℝ) :
  tom_weight = 150 → vest_fraction = 0.5 → hold_fraction = 1.5 →
  let vest_weight := vest_fraction * tom_weight,
      hand_weight := hold_fraction * tom_weight,
      total_hand_weight := 2 * hand_weight,
      total_weight := tom_weight + vest_weight + total_hand_weight in
  total_weight = 675 :=
by
  sorry

end total_weight_moved_l192_192981


namespace race_runners_l192_192766

theorem race_runners (n : ℕ) (h1 : 5 * 8 + (n - 5) * 10 = 70) : n = 8 :=
sorry

end race_runners_l192_192766


namespace sum_of_transformed_numbers_l192_192382

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_transformed_numbers_l192_192382


namespace sum_of_final_two_numbers_l192_192384

theorem sum_of_final_two_numbers (a b S : ℝ) (h : a + b = S) : 
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 :=
by
  sorry

end sum_of_final_two_numbers_l192_192384


namespace count_numbers_divisible_by_12_not_20_l192_192721

theorem count_numbers_divisible_by_12_not_20 : 
  let N := 2017
  let a := Nat.floor (N / 12)
  let b := Nat.floor (N / 60)
  a - b = 135 := by
    -- Definitions used
    let N := 2017
    let a := Nat.floor (N / 12)
    let b := Nat.floor (N / 60)
    -- The desired statement
    show a - b = 135
    sorry

end count_numbers_divisible_by_12_not_20_l192_192721


namespace no_real_value_x_l192_192084

theorem no_real_value_x (R H : ℝ) (π : ℝ := Real.pi) :
  R = 10 → H = 5 →
  ¬∃ x : ℝ,  π * (R + x)^2 * H = π * R^2 * (H + x) ∧ x ≠ 0 :=
by
  intros hR hH; sorry

end no_real_value_x_l192_192084


namespace seventh_observation_l192_192993

-- Declare the conditions with their definitions
def average_of_six (sum6 : ℕ) : Prop := sum6 = 6 * 14
def new_average_decreased (sum6 sum7 : ℕ) : Prop := sum7 = sum6 + 7 ∧ 13 = (sum6 + 7) / 7

-- The main statement to prove that the seventh observation is 7
theorem seventh_observation (sum6 sum7 : ℕ) (h_avg6 : average_of_six sum6) (h_new_avg : new_average_decreased sum6 sum7) :
  sum7 - sum6 = 7 := 
  sorry

end seventh_observation_l192_192993


namespace sufficient_not_necessary_condition_l192_192082

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x^2 - a ≤ 0)) → (a ≥ 5) :=
sorry

end sufficient_not_necessary_condition_l192_192082


namespace ratio_sheep_horses_l192_192861

theorem ratio_sheep_horses (amount_food_per_horse : ℕ) (total_food_per_day : ℕ) (num_sheep : ℕ) (num_horses : ℕ) :
  amount_food_per_horse = 230 ∧ total_food_per_day = 12880 ∧ num_sheep = 24 ∧ num_horses = total_food_per_day / amount_food_per_horse →
  num_sheep / num_horses = 3 / 7 :=
by
  sorry

end ratio_sheep_horses_l192_192861


namespace roses_in_vase_now_l192_192093

-- Definitions of initial conditions and variables
def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def orchids_cut : ℕ := 19
def orchids_now : ℕ := 21

-- The proof problem to show that the number of roses now is still the same as initially.
theorem roses_in_vase_now : initial_roses = 12 :=
by
  -- The proof itself is left as an exercise (add proof here)
  sorry

end roses_in_vase_now_l192_192093


namespace blue_paint_cans_needed_l192_192062

theorem blue_paint_cans_needed (ratio_bg : ℤ × ℤ) (total_cans : ℤ) (r : ratio_bg = (4, 3)) (t : total_cans = 42) :
  let ratio_bw : ℚ := 4 / (4 + 3) 
  let blue_cans : ℚ := ratio_bw * total_cans 
  blue_cans = 24 :=
by
  sorry

end blue_paint_cans_needed_l192_192062


namespace mean_of_remaining_students_l192_192133

variable (k : ℕ) (h1 : k > 20)

def mean_of_class (mean : ℝ := 10) := mean
def mean_of_20_students (mean : ℝ := 16) := mean

theorem mean_of_remaining_students 
  (h2 : mean_of_class = 10)
  (h3 : mean_of_20_students = 16) :
  let remaining_students := (k - 20)
  let total_score_20 := 20 * mean_of_20_students
  let total_score_class := k * mean_of_class
  let total_score_remaining := total_score_class - total_score_20
  let mean_remaining := total_score_remaining / remaining_students
  mean_remaining = (10 * k - 320) / (k - 20) :=
sorry

end mean_of_remaining_students_l192_192133


namespace sqrt_72_plus_sqrt_32_l192_192963

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l192_192963


namespace polygon_sides_l192_192088

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n = 10 := by
  sorry

end polygon_sides_l192_192088


namespace apples_in_bowl_l192_192807

theorem apples_in_bowl (green_plus_red_diff red_count : ℕ) (h1 : green_plus_red_diff = 12) (h2 : red_count = 16) :
  red_count + (red_count + green_plus_red_diff) = 44 :=
by
  sorry

end apples_in_bowl_l192_192807


namespace similar_polygon_area_sum_l192_192867

theorem similar_polygon_area_sum 
  (t1 t2 a1 a2 b : ℝ)
  (h_ratio: t1 / t2 = a1^2 / a2^2)
  (t3 : ℝ := t1 + t2)
  (h_area_eq : t3 = b^2 * a1^2 / a2^2): 
  b = Real.sqrt (a1^2 + a2^2) :=
by
  sorry

end similar_polygon_area_sum_l192_192867


namespace derivative_at_2_l192_192534

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x
noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem derivative_at_2 
  (a b : ℝ)
  (h1 : f a b 1 = -2)
  (h2 : (∂ x, f a b x) 1 = 0) : 
  f' a b 2 = -1/2 := sorry

end derivative_at_2_l192_192534


namespace simplify_radicals_l192_192951

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l192_192951


namespace painting_time_eq_l192_192308

theorem painting_time_eq (t : ℚ) : 
  (1/6 + 1/8 + 1/10) * (t - 2) = 1 := 
sorry

end painting_time_eq_l192_192308


namespace quadratic_equation_in_one_variable_l192_192826

-- Definitions for each condition
def equation_A (x : ℝ) : Prop := x^2 = -1
def equation_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def equation_C (x : ℝ) : Prop := 2 * (x + 1)^2 = (Real.sqrt 2 * x - 1)^2
def equation_D (x : ℝ) : Prop := x + 1 / x = 1

-- Main theorem statement
theorem quadratic_equation_in_one_variable (x : ℝ) :
  equation_A x ∧ ¬(∃ a b c, equation_B a b c x ∧ a ≠ 0) ∧ ¬equation_C x ∧ ¬equation_D x :=
  sorry

end quadratic_equation_in_one_variable_l192_192826


namespace find_f_prime_at_2_l192_192528

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l192_192528


namespace sum_of_interior_angles_of_special_regular_polygon_l192_192800

theorem sum_of_interior_angles_of_special_regular_polygon (n : ℕ) (h1 : n = 4 ∨ n = 5) :
  ((n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540) :=
by sorry

end sum_of_interior_angles_of_special_regular_polygon_l192_192800


namespace priya_speed_l192_192238

theorem priya_speed (Riya_speed Priya_speed : ℝ) (time_separation distance_separation : ℝ)
  (h1 : Riya_speed = 30) 
  (h2 : time_separation = 45 / 60) -- 45 minutes converted to hours
  (h3 : distance_separation = 60)
  : Priya_speed = 50 :=
sorry

end priya_speed_l192_192238


namespace average_of_last_six_l192_192071

theorem average_of_last_six (avg_13 : ℕ → ℝ) (avg_first_6 : ℕ → ℝ) (middle_number : ℕ → ℝ) :
  (∀ n, avg_13 n = 9) →
  (∀ n, n ≤ 6 → avg_first_6 n = 5) →
  (middle_number 7 = 45) →
  ∃ (A : ℝ), (∀ n, n > 6 → n < 13 → avg_13 n = A) ∧ A = 7 :=
by
  sorry

end average_of_last_six_l192_192071


namespace division_correct_l192_192161

theorem division_correct : 0.45 / 0.005 = 90 := by
  sorry

end division_correct_l192_192161


namespace no_whole_numbers_satisfy_eqn_l192_192307

theorem no_whole_numbers_satisfy_eqn :
  ¬ ∃ (x y z : ℤ), (x - y) ^ 3 + (y - z) ^ 3 + (z - x) ^ 3 = 2021 :=
by
  sorry

end no_whole_numbers_satisfy_eqn_l192_192307


namespace german_mo_2016_problem_1_l192_192171

theorem german_mo_2016_problem_1 (a b : ℝ) :
  a^2 + b^2 = 25 ∧ 3 * (a + b) - a * b = 15 ↔
  (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨
  (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4) :=
sorry

end german_mo_2016_problem_1_l192_192171


namespace minimum_value_of_expression_l192_192008

theorem minimum_value_of_expression :
  ∀ x y : ℝ, x^2 - x * y + y^2 ≥ 0 :=
by
  sorry

end minimum_value_of_expression_l192_192008


namespace arith_seq_ninth_term_value_l192_192607

variable {a : Nat -> ℤ}
variable {S : Nat -> ℤ}

def arith_seq (a : Nat -> ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + a 1^2

def arith_sum (S : Nat -> ℤ) (a : Nat -> ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem arith_seq_ninth_term_value
  (h_seq : arith_seq a)
  (h_sum : arith_sum S a)
  (h_cond1 : a 1 + a 2^2 = -3)
  (h_cond2 : S 5 = 10) :
  a 9 = 20 :=
by
  sorry

end arith_seq_ninth_term_value_l192_192607


namespace hannahs_adblock_not_block_l192_192023

theorem hannahs_adblock_not_block (x : ℝ) (h1 : 0.8 * x = 0.16) : x = 0.2 :=
by {
  sorry
}

end hannahs_adblock_not_block_l192_192023


namespace arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l192_192347

-- Definitions and theorems for the given conditions

-- (1) General formula for the arithmetic sequence
theorem arithmetic_sequence_formula (a S : Nat → Int) (n : Nat) (h1 : a 2 = -1)
  (h2 : S 9 = 5 * S 5) : 
  ∀ n, a n = -8 * n + 15 := 
sorry

-- (2) Minimum value of t - s
theorem min_value_t_minus_s (b : Nat → Rat) (T : Nat → Rat) 
  (h3 : ∀ n, b n = 1 / ((-8 * (n + 1) + 15) * (-8 * (n + 2) + 15))) 
  (h4 : ∀ n, s ≤ T n ∧ T n ≤ t) : 
  t - s = 1 / 72 := 
sorry

-- (3) Maximum value of k
theorem max_value_k (S a : Nat → Int) (k : Rat)
  (h5 : ∀ n, n ≥ 3 → S n / a n ≤ n^2 / (n + k)) :
  k = 80 / 9 := 
sorry

end arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l192_192347


namespace incorrect_statement_d_l192_192633

noncomputable def x := Complex.mk (-1/2) (Real.sqrt 3 / 2)
noncomputable def y := Complex.mk (-1/2) (-Real.sqrt 3 / 2)

theorem incorrect_statement_d : (x^12 + y^12) ≠ 1 := by
  sorry

end incorrect_statement_d_l192_192633


namespace regular_polygon_interior_angle_l192_192558

theorem regular_polygon_interior_angle (S : ℝ) (n : ℕ) (h1 : S = 720) (h2 : (n - 2) * 180 = S) : 
  (S / n) = 120 := 
by
  sorry

end regular_polygon_interior_angle_l192_192558


namespace route_time_difference_l192_192485

-- Define the conditions of the problem
def first_route_time : ℕ := 
  let time_uphill := 6
  let time_path := 2 * time_uphill
  let time_finish := (time_uphill + time_path) / 3
  time_uphill + time_path + time_finish

def second_route_time : ℕ := 
  let time_flat_path := 14
  let time_finish := 2 * time_flat_path
  time_flat_path + time_finish

-- Prove the question
theorem route_time_difference : second_route_time - first_route_time = 18 :=
by
  sorry

end route_time_difference_l192_192485


namespace area_of_triangle_formed_by_lines_l192_192802

theorem area_of_triangle_formed_by_lines (x y : ℝ) (h1 : y = x) (h2 : x = -5) :
  let base := 5
  let height := 5
  let area := (1 / 2 : ℝ) * base * height
  area = 12.5 := 
by
  sorry

end area_of_triangle_formed_by_lines_l192_192802


namespace additional_weight_difference_l192_192352

theorem additional_weight_difference (raw_squat sleeves_add wraps_percentage : ℝ) 
  (raw_squat_val : raw_squat = 600) 
  (sleeves_add_val : sleeves_add = 30) 
  (wraps_percentage_val : wraps_percentage = 0.25) : 
  (wraps_percentage * raw_squat) - sleeves_add = 120 :=
by
  rw [ raw_squat_val, sleeves_add_val, wraps_percentage_val ]
  norm_num

end additional_weight_difference_l192_192352


namespace areas_of_triangles_l192_192348

-- Define the condition that the gcd of a, b, and c is 1
def gcd_one (a b c : ℤ) : Prop := Int.gcd (Int.gcd a b) c = 1

-- Define the set of possible areas for triangles in E
def f_E : Set ℝ :=
  { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) }

theorem areas_of_triangles : 
  f_E = { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) } :=
by {
  sorry
}

end areas_of_triangles_l192_192348


namespace fred_earned_from_car_wash_l192_192742

def weekly_allowance : ℕ := 16
def spent_on_movies : ℕ := weekly_allowance / 2
def amount_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 14
def earned_from_car_wash : ℕ := final_amount - amount_after_movies

theorem fred_earned_from_car_wash : earned_from_car_wash = 6 := by
  sorry

end fred_earned_from_car_wash_l192_192742


namespace A_and_D_independent_l192_192814

variable (Ω : Type) [Fintype Ω] [ProbabilitySpace Ω]

namespace BallDrawing

def events (ω₁ ω₂ : Ω) : Prop :=
  (ω₁ = 1 ∧ ω₂ = 2) ∨
  (ω₁ + ω₂ = 8) ∨
  (ω₁ + ω₂ = 7)

def A (ω₁ ω₂ : Ω) : Prop := ω₁ = 1
def B (ω₁ ω₂ : Ω) : Prop := ω₂ = 2
def C (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 8
def D (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 7

theorem A_and_D_independent : 
  ∀ Ω [Fintype Ω] [ProbabilitySpace Ω], 
  independence (event A) (event D) :=
by sorry

end BallDrawing

end A_and_D_independent_l192_192814


namespace ab_value_l192_192200

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry

end ab_value_l192_192200


namespace horatio_sonnets_l192_192024

theorem horatio_sonnets (num_lines_per_sonnet : ℕ) (heard_sonnets : ℕ) (unheard_lines : ℕ) (h1 : num_lines_per_sonnet = 16) (h2 : heard_sonnets = 9) (h3 : unheard_lines = 126) :
  ∃ total_sonnets : ℕ, total_sonnets = 16 :=
by
  -- Note: The proof is not required, hence 'sorry' is included to skip it.
  sorry

end horatio_sonnets_l192_192024


namespace mark_total_cans_l192_192487

theorem mark_total_cans (p1 p2 p3 p4 p5 p6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ)
  (h1 : p1 = 30) (h2 : p2 = 25) (h3 : p3 = 35) (h4 : p4 = 40) 
  (h5 : p5 = 28) (h6 : p6 = 32) (hc1 : c1 = 12) (hc2 : c2 = 10) 
  (hc3 : c3 = 15) (hc4 : c4 = 14) (hc5 : c5 = 11) (hc6 : c6 = 13) :
  p1 * c1 + p2 * c2 + p3 * c3 + p4 * c4 + p5 * c5 + p6 * c6 = 2419 := 
by 
  sorry

end mark_total_cans_l192_192487


namespace customers_tipped_count_l192_192715

variable (initial_customers : ℕ)
variable (added_customers : ℕ)
variable (customers_no_tip : ℕ)

def total_customers (initial_customers added_customers : ℕ) : ℕ :=
  initial_customers + added_customers

theorem customers_tipped_count 
  (h_init : initial_customers = 29)
  (h_added : added_customers = 20)
  (h_no_tip : customers_no_tip = 34) :
  (total_customers initial_customers added_customers - customers_no_tip) = 15 :=
by
  sorry

end customers_tipped_count_l192_192715


namespace max_value_at_one_l192_192527

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem max_value_at_one (a b : ℝ) 
  (h1 : f 1 a b = -2) 
  (h2 : ∀ x, f' x a b = (a * x + 2) / x^2 := by {
    sorry -- this can be the calculated derivative step
  }) 
  (h3 : f' 1 a b = 0) : 
  f' 2 (-2) (-2) = -1/2 := by
{
  sorry -- this needs to match the final correct answer
}

end max_value_at_one_l192_192527


namespace smallest_integer_solution_l192_192428

theorem smallest_integer_solution (x : ℤ) (h : 2 * (x : ℝ)^2 + 2 * |(x : ℝ)| + 7 < 25) : x = -2 :=
by
  sorry

end smallest_integer_solution_l192_192428


namespace mark_performance_length_l192_192229

theorem mark_performance_length :
  ∃ (x : ℕ), (x > 0) ∧ (6 * 5 * x = 90) ∧ (x = 3) :=
by
  sorry

end mark_performance_length_l192_192229


namespace sum_sin_double_angles_eq_l192_192272

theorem sum_sin_double_angles_eq (
  α β γ : ℝ
) (h : α + β + γ = Real.pi) :
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ :=
sorry

end sum_sin_double_angles_eq_l192_192272


namespace div_decimal_l192_192162

theorem div_decimal (a b : ℝ)  (h₁ : a = 0.45) (h₂ : b = 0.005):
  a / b = 90 :=
by {
  sorry
}

end div_decimal_l192_192162


namespace inequality_2n_1_lt_n_plus_1_sq_l192_192435

theorem inequality_2n_1_lt_n_plus_1_sq (n : ℕ) (h : 0 < n) : 2 * n - 1 < (n + 1) ^ 2 := 
by 
  sorry

end inequality_2n_1_lt_n_plus_1_sq_l192_192435


namespace closest_integer_to_cbrt_250_l192_192112

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l192_192112


namespace probability_of_at_least_19_l192_192851

-- Defining the possible coins in Anya's pocket
def coins : list ℕ := [10, 10, 5, 5, 2]

-- Function to calculate the sum of chosen coins
def sum_coins (l : list ℕ) := list.sum l

-- Function to check if the sum of chosen coins is at least 19 rubles
def at_least_19 (l : list ℕ) := (sum_coins l) ≥ 19

-- Extract all possible combinations of 3 coins from the list
def combinations (l : list ℕ) (n : ℕ) := 
  if h : n ≤ l.length then 
    (list.permutations l).dedup.map (λ p, p.take n).dedup
  else
    []

-- Specific combinations of 3 coins out of 5
def three_coin_combinations := combinations coins 3 

-- Count the number of favorable outcomes (combinations that sum to at least 19)
def favorable_combinations := list.filter at_least_19 three_coin_combinations

-- Calculate the probability
def probability := (favorable_combinations.length : ℚ) / (three_coin_combinations.length : ℚ)

-- Prove that the probability is 0.4
theorem probability_of_at_least_19 : probability = 0.4 :=
  sorry

end probability_of_at_least_19_l192_192851


namespace power_of_product_l192_192573

variable (x y : ℝ)

theorem power_of_product (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 :=
  sorry

end power_of_product_l192_192573


namespace goldfish_in_first_tank_l192_192788

-- Definitions of conditions
def num_fish_third_tank : Nat := 10
def num_fish_second_tank := 3 * num_fish_third_tank
def num_fish_first_tank := num_fish_second_tank / 2
def goldfish_and_beta_sum (G : Nat) : Prop := G + 8 = num_fish_first_tank

-- Theorem to prove the number of goldfish in the first fish tank
theorem goldfish_in_first_tank (G : Nat) (h : goldfish_and_beta_sum G) : G = 7 :=
by
  sorry

end goldfish_in_first_tank_l192_192788


namespace track_length_l192_192420

theorem track_length (x : ℝ) (b_speed s_speed : ℝ) (b_dist1 s_dist1 s_dist2 : ℝ)
  (h1 : b_dist1 = 80)
  (h2 : s_dist1 = x / 2 - 80)
  (h3 : s_dist2 = s_dist1 + 180)
  (h4 : x / 4 * b_speed = (x / 2 - 80) * s_speed)
  (h5 : x / 4 * ((x / 2) - 100) = (x / 2 + 100) * s_speed) :
  x = 520 := 
sorry

end track_length_l192_192420


namespace ratio_of_w_to_y_l192_192667

variables (w x y z : ℚ)

theorem ratio_of_w_to_y:
  (w / x = 5 / 4) →
  (y / z = 5 / 3) →
  (z / x = 1 / 5) →
  (w / y = 15 / 4) :=
by
  intros hwx hyz hzx
  sorry

end ratio_of_w_to_y_l192_192667


namespace trader_gain_percentage_is_25_l192_192569

noncomputable def trader_gain_percentage (C : ℝ) : ℝ :=
  ((22 * C) / (88 * C)) * 100

theorem trader_gain_percentage_is_25 (C : ℝ) (h : C ≠ 0) : trader_gain_percentage C = 25 := by
  unfold trader_gain_percentage
  field_simp [h]
  norm_num
  sorry

end trader_gain_percentage_is_25_l192_192569


namespace inequality_proof_l192_192237

theorem inequality_proof (x : ℝ) (hx : x ≥ 1) : x^5 - 1 / x^4 ≥ 9 * (x - 1) := 
by sorry

end inequality_proof_l192_192237


namespace cost_price_computer_table_l192_192278

theorem cost_price_computer_table (CP SP : ℝ) (h1 : SP = 1.15 * CP) (h2 : SP = 6400) : CP = 5565.22 :=
by sorry

end cost_price_computer_table_l192_192278


namespace Monroe_spiders_l192_192370

theorem Monroe_spiders (S : ℕ) (h1 : 12 * 6 + S * 8 = 136) : S = 8 :=
by
  sorry

end Monroe_spiders_l192_192370


namespace number_of_girls_l192_192561

variable (N n g : ℕ)
variable (h1 : N = 1600)
variable (h2 : n = 200)
variable (h3 : g = 95)

theorem number_of_girls (G : ℕ) (h : g * N = G * n) : G = 760 :=
by sorry

end number_of_girls_l192_192561


namespace tan_theta_determined_l192_192577

theorem tan_theta_determined (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4) (h_zero : Real.tan θ + Real.tan (4 * θ) = 0) :
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) :=
sorry

end tan_theta_determined_l192_192577


namespace triangle_side_height_inequality_l192_192666

theorem triangle_side_height_inequality (a b h_a h_b S : ℝ) (h1 : a > b) 
  (h2: h_a = 2 * S / a) (h3: h_b = 2 * S / b) :
  a + h_a ≥ b + h_b :=
by sorry

end triangle_side_height_inequality_l192_192666


namespace simplify_sum_of_square_roots_l192_192954

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l192_192954


namespace krishan_money_l192_192668

theorem krishan_money 
  (R G K : ℝ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 490) : K = 2890 :=
sorry

end krishan_money_l192_192668


namespace brick_surface_area_l192_192998

variable (X Y Z : ℝ)

#check 4 * X + 4 * Y + 2 * Z = 72 → 
       4 * X + 2 * Y + 4 * Z = 96 → 
       2 * X + 4 * Y + 4 * Z = 102 →
       2 * (X + Y + Z) = 54

theorem brick_surface_area (h1 : 4 * X + 4 * Y + 2 * Z = 72)
                           (h2 : 4 * X + 2 * Y + 4 * Z = 96)
                           (h3 : 2 * X + 4 * Y + 4 * Z = 102) :
                           2 * (X + Y + Z) = 54 := by
  sorry

end brick_surface_area_l192_192998


namespace friends_gift_l192_192044

-- Define the original number of balloons and the final number of balloons
def original_balloons := 8
def final_balloons := 10

-- The main theorem: Joan's friend gave her 2 orange balloons.
theorem friends_gift : (final_balloons - original_balloons) = 2 := by
  sorry

end friends_gift_l192_192044


namespace sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l192_192879

theorem sum_distinct_prime_factors_of_7_to_7_minus_7_to_4 : 
  let pfs := primeFactors (7 ^ 7 - 7 ^ 4)
  in (pfs = {2, 3, 19}) → sum pfs = 24 :=
by
  sorry

end sum_distinct_prime_factors_of_7_to_7_minus_7_to_4_l192_192879


namespace sum_of_first_seven_primes_with_units_digit_3_lt_150_l192_192174

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_less_than_150 (n : ℕ) : Prop :=
  n < 150

def first_seven_primes_with_units_digit_3 := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3_lt_150 :
  (has_units_digit_3 3) ∧ (is_less_than_150 3) ∧ (Prime 3) ∧
  (has_units_digit_3 13) ∧ (is_less_than_150 13) ∧ (Prime 13) ∧
  (has_units_digit_3 23) ∧ (is_less_than_150 23) ∧ (Prime 23) ∧
  (has_units_digit_3 43) ∧ (is_less_than_150 43) ∧ (Prime 43) ∧
  (has_units_digit_3 53) ∧ (is_less_than_150 53) ∧ (Prime 53) ∧
  (has_units_digit_3 73) ∧ (is_less_than_150 73) ∧ (Prime 73) ∧
  (has_units_digit_3 83) ∧ (is_less_than_150 83) ∧ (Prime 83) →
  (3 + 13 + 23 + 43 + 53 + 73 + 83 = 291) :=
by
  sorry

end sum_of_first_seven_primes_with_units_digit_3_lt_150_l192_192174


namespace distinct_natural_numbers_l192_192769

theorem distinct_natural_numbers (n : ℕ) (h : n = 100) : 
  ∃ (nums : Fin n → ℕ), 
    (∀ i j, i ≠ j → nums i ≠ nums j) ∧
    (∀ (a b c d e : Fin n), 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
     c ≠ d ∧ c ≠ e ∧ 
     d ≠ e →
      (nums a) * (nums b) * (nums c) * (nums d) * (nums e) % ((nums a) + (nums b) + (nums c) + (nums d) + (nums e)) = 0) :=
by
  sorry

end distinct_natural_numbers_l192_192769


namespace find_sum_of_numbers_l192_192693

-- Define the problem using the given conditions
def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

-- The main theorem we want to prove
theorem find_sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) :
  sum_of_three_numbers a b c = 22 :=
by
  sorry

end find_sum_of_numbers_l192_192693


namespace Marnie_can_make_9_bracelets_l192_192230

def number_of_beads : Nat :=
  (5 * 50) + (2 * 100)

def beads_per_bracelet : Nat := 50

def total_bracelets (total_beads : Nat) (beads_per_bracelet : Nat) : Nat :=
  total_beads / beads_per_bracelet

theorem Marnie_can_make_9_bracelets :
  total_bracelets number_of_beads beads_per_bracelet = 9 :=
by
  -- proof goes here
  sorry

end Marnie_can_make_9_bracelets_l192_192230


namespace determine_e_l192_192250

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

theorem determine_e (d f : ℝ) (h1 : f = 18) (h2 : -f/3 = -6) (h3 : -d/3 = -6) (h4 : 3 + d + e + f = -6) : e = -45 :=
sorry

end determine_e_l192_192250


namespace central_angle_of_sector_l192_192143

theorem central_angle_of_sector (r S : ℝ) (h_r : r = 2) (h_S : S = 4) : 
  ∃ α : ℝ, α = 2 ∧ S = (1/2) * α * r^2 := 
by 
  sorry

end central_angle_of_sector_l192_192143


namespace diana_bike_home_time_l192_192000

theorem diana_bike_home_time : 
  ∀ (dist total_dist speed1 speed2 time1 time2 : ℝ), 
  total_dist = 10 ∧ 
  speed1 = 3 ∧ time1 = 2 ∧ speed2 = 1 ∧
  dist = speed1 * time1 ∧
  (total_dist - dist) = speed2 * time2 → 
  time1 + time2 = 6 :=
by
  intros dist total_dist speed1 speed2 time1 time2 h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  cases h8 with h9 h10
  rw [h3, h4, h5, h6] at h10
  rw [h7, h8, h9, h10] at h10
  sorry

end diana_bike_home_time_l192_192000


namespace documentaries_count_l192_192582

def number_of_documents
  (novels comics albums crates capacity : ℕ)
  (total_items := crates * capacity)
  (known_items := novels + comics + albums)
  (documentaries := total_items - known_items) : ℕ :=
  documentaries

theorem documentaries_count
  : number_of_documents 145 271 209 116 9 = 419 :=
by
  sorry

end documentaries_count_l192_192582


namespace simplify_radicals_l192_192949

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l192_192949


namespace determinant_zero_l192_192735

noncomputable def A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]

theorem determinant_zero (α β : ℝ) : Matrix.det (A α β) = 0 := 
  sorry

end determinant_zero_l192_192735


namespace closest_integer_to_cube_root_250_l192_192111

theorem closest_integer_to_cube_root_250 : ∃ x : ℤ, (x = 6) ∧ (∀ y : ℤ, (abs (y - real.cbrt 250)) ≥ (abs (6 - real.cbrt 250))) :=
by
  sorry

end closest_integer_to_cube_root_250_l192_192111


namespace closest_integer_to_cbrt_250_l192_192113

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l192_192113


namespace measure_of_angle_4_l192_192417

theorem measure_of_angle_4 
  (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) : 
  angle4 = 110 :=
by
  sorry

end measure_of_angle_4_l192_192417


namespace negation_of_proposition_l192_192453

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0)) ↔ (∃ x : ℝ, x^2 + 2 * x + 3 < 0) :=
by sorry

end negation_of_proposition_l192_192453


namespace find_f_prime_at_2_l192_192537

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem find_f_prime_at_2 (a b : ℝ) 
  (h1 : f a b 1 = -2)
  (h2 : f' a b 1 = 0) :
  f' a b 2 = -1 / 2 :=
sorry

end find_f_prime_at_2_l192_192537


namespace first_even_number_of_8_sum_424_l192_192718

theorem first_even_number_of_8_sum_424 (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + 
                   (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) : x = 46 :=
by sorry

end first_even_number_of_8_sum_424_l192_192718


namespace negation_universal_proposition_l192_192794

theorem negation_universal_proposition {x : ℝ} : 
  (¬ ∀ x : ℝ, x^2 ≥ 2) ↔ (∃ x : ℝ, x^2 < 2) := 
sorry

end negation_universal_proposition_l192_192794


namespace count_wave_numbers_l192_192295

/-- 
A wave number is a 5-digit number such that the tens and thousands digits are each larger than their adjacent digits.
We prove that the number of 5-digit wave numbers that can be formed using the digits 1, 2, 3, 4, and 5 without repeating any digits is 16.
-/
theorem count_wave_numbers : 
  (Finset.univ.filter (λ n : Fin 5 → Fin 6, n 1 > n 0 ∧ n 1 > n 2 ∧ n 3 > n 2 ∧ n 3 > n 4)).card = 16 :=
sorry

end count_wave_numbers_l192_192295


namespace negation_of_proposition_l192_192793

variable (x : ℝ)

theorem negation_of_proposition (h : ∃ x : ℝ, x^2 + x - 1 < 0) : ¬ (∀ x : ℝ, x^2 + x - 1 ≥ 0) :=
sorry

end negation_of_proposition_l192_192793


namespace cory_chairs_l192_192167

theorem cory_chairs (total_cost table_cost chair_cost C : ℕ) (h1 : total_cost = 135) (h2 : table_cost = 55) (h3 : chair_cost = 20) (h4 : total_cost = table_cost + chair_cost * C) : C = 4 := 
by 
  sorry

end cory_chairs_l192_192167


namespace necessary_but_not_sufficient_condition_l192_192074

noncomputable def condition_sufficiency (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m*x + 1 > 0

theorem necessary_but_not_sufficient_condition (m : ℝ) : m < 2 → (¬ condition_sufficiency m ∨ condition_sufficiency m) :=
by
  sorry

end necessary_but_not_sufficient_condition_l192_192074


namespace twice_abs_difference_of_squares_is_4000_l192_192266

theorem twice_abs_difference_of_squares_is_4000 :
  2 * |(105:ℤ)^2 - (95:ℤ)^2| = 4000 :=
by sorry

end twice_abs_difference_of_squares_is_4000_l192_192266


namespace average_capacity_is_3_65_l192_192061

/-- Define the capacities of the jars as a list--/
def jarCapacities : List ℚ := [2, 1/4, 8, 1.5, 0.75, 3, 10]

/-- Calculate the average jar capacity --/
def averageCapacity (capacities : List ℚ) : ℚ :=
  (capacities.sum) / (capacities.length)

/-- The average jar capacity for the given list of jar capacities is 3.65 liters. --/
theorem average_capacity_is_3_65 :
  averageCapacity jarCapacities = 3.65 := 
by
  unfold averageCapacity
  dsimp [jarCapacities]
  norm_num
  sorry

end average_capacity_is_3_65_l192_192061


namespace fraction_problem_l192_192324

theorem fraction_problem (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := 
by
  sorry

end fraction_problem_l192_192324


namespace num_ways_distribute_plants_correct_l192_192644

def num_ways_to_distribute_plants : Nat :=
  let basil := 2
  let aloe := 1
  let cactus := 1
  let white_lamps := 2
  let red_lamp := 1
  let blue_lamp := 1
  let plants := basil + aloe + cactus
  let lamps := white_lamps + red_lamp + blue_lamp
  4
  
theorem num_ways_distribute_plants_correct :
  num_ways_to_distribute_plants = 4 :=
by
  sorry -- Proof of the correctness of the distribution

end num_ways_distribute_plants_correct_l192_192644


namespace arrangement_valid_l192_192303

def unique_digits (a b c d e f : Nat) : Prop :=
  (a = 4) ∧ (b = 1) ∧ (c = 2) ∧ (d = 5) ∧ (e = 6) ∧ (f = 3)

def sum_15 (x y z : Nat) : Prop :=
  x + y + z = 15

theorem arrangement_valid :
  ∃ a b c d e f : Nat, unique_digits a b c d e f ∧
  sum_15 a d e ∧
  sum_15 d b f ∧
  sum_15 f e c ∧
  sum_15 a b c ∧
  sum_15 a e f ∧
  sum_15 b d c :=
sorry

end arrangement_valid_l192_192303


namespace g_at_5_l192_192774

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_at_5 : g 5 = 74 := 
by {
  sorry
}

end g_at_5_l192_192774


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l192_192881

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l192_192881


namespace fraction_expression_l192_192003

theorem fraction_expression :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_expression_l192_192003


namespace fraction_solution_l192_192616

theorem fraction_solution (x : ℝ) (h : 4 - 9 / x + 4 / x^2 = 0) : 3 / x = 12 ∨ 3 / x = 3 / 4 :=
by
  -- Proof to be written here
  sorry

end fraction_solution_l192_192616


namespace relationship_between_m_and_n_l192_192185

theorem relationship_between_m_and_n (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x : ℝ, f x = f (-x)) 
  (h_mono_inc : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) 
  (m_def : f (-1) = f 1) 
  (n_def : f (a^2 + 2*a + 3) > f 1) :
  f (-1) < f (a^2 + 2*a + 3) := 
by 
  sorry

end relationship_between_m_and_n_l192_192185


namespace sufficient_not_necessary_condition_l192_192446

-- Definition of the conditions
def Q (x : ℝ) : Prop := x^2 - x - 2 > 0
def P (x a : ℝ) : Prop := |x| > a

-- Main statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, P x a → Q x) → a ≥ 2 :=
by
  sorry

end sufficient_not_necessary_condition_l192_192446


namespace Paul_seashells_l192_192194

namespace SeashellProblem

variables (P L : ℕ)

def initial_total_seashells (H P L : ℕ) : Prop := H + P + L = 59

def final_total_seashells (H P L : ℕ) : Prop := H + P + L - L / 4 = 53

theorem Paul_seashells : 
  (initial_total_seashells 11 P L) → (final_total_seashells 11 P L) → P = 24 :=
by
  intros h_initial h_final
  sorry

end SeashellProblem

end Paul_seashells_l192_192194


namespace gcd_a2_14a_49_a_7_l192_192330

theorem gcd_a2_14a_49_a_7 (a : ℤ) (k : ℤ) (h : a = 2100 * k) :
  Int.gcd (a^2 + 14*a + 49) (a + 7) = 7 := 
by
  sorry

end gcd_a2_14a_49_a_7_l192_192330


namespace molecular_weight_of_3_moles_l192_192823

namespace AscorbicAcid

def molecular_form : List (String × ℕ) := [("C", 6), ("H", 8), ("O", 6)]

def atomic_weight : String → ℝ
| "C" => 12.01
| "H" => 1.008
| "O" => 16.00
| _ => 0

noncomputable def molecular_weight (molecular_form : List (String × ℕ)) : ℝ :=
molecular_form.foldr (λ (x : (String × ℕ)) acc => acc + (x.snd * atomic_weight x.fst)) 0

noncomputable def weight_of_3_moles (mw : ℝ) : ℝ := mw * 3

theorem molecular_weight_of_3_moles :
  weight_of_3_moles (molecular_weight molecular_form) = 528.372 :=
by
  sorry

end AscorbicAcid

end molecular_weight_of_3_moles_l192_192823


namespace rectangle_area_problem_l192_192710

/--
Given a rectangle with dimensions \(3x - 4\) and \(4x + 6\),
show that the area of the rectangle equals \(12x^2 + 2x - 24\) if and only if \(x \in \left(\frac{4}{3}, \infty\right)\).
-/
theorem rectangle_area_problem 
  (x : ℝ) 
  (h1 : 3 * x - 4 > 0)
  (h2 : 4 * x + 6 > 0) :
  (3 * x - 4) * (4 * x + 6) = 12 * x^2 + 2 * x - 24 ↔ x > 4 / 3 :=
sorry

end rectangle_area_problem_l192_192710


namespace A_and_D_independent_l192_192815

variable (Ω : Type) [Fintype Ω] [ProbabilitySpace Ω]

namespace BallDrawing

def events (ω₁ ω₂ : Ω) : Prop :=
  (ω₁ = 1 ∧ ω₂ = 2) ∨
  (ω₁ + ω₂ = 8) ∨
  (ω₁ + ω₂ = 7)

def A (ω₁ ω₂ : Ω) : Prop := ω₁ = 1
def B (ω₁ ω₂ : Ω) : Prop := ω₂ = 2
def C (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 8
def D (ω₁ ω₂ : Ω) : Prop := ω₁ + ω₂ = 7

theorem A_and_D_independent : 
  ∀ Ω [Fintype Ω] [ProbabilitySpace Ω], 
  independence (event A) (event D) :=
by sorry

end BallDrawing

end A_and_D_independent_l192_192815


namespace no_lonely_points_eventually_l192_192602

structure Graph (α : Type) :=
(vertices : Finset α)
(edges : α → Finset α)

namespace Graph

def is_lonely {α : Type} (G : Graph α) (coloring : α → Bool) (v : α) : Prop :=
  let neighbors := G.edges v
  let different_color_neighbors := neighbors.filter (λ w => coloring w ≠ coloring v)
  2 * different_color_neighbors.card > neighbors.card

end Graph

theorem no_lonely_points_eventually
  {α : Type}
  (G : Graph α)
  (initial_coloring : α → Bool) :
  ∃ (steps : Nat),
  ∀ (coloring : α → Bool),
  (∃ (t : Nat), t ≤ steps ∧ 
    (∀ v, ¬ Graph.is_lonely G coloring v)) :=
sorry

end no_lonely_points_eventually_l192_192602


namespace thought_number_and_appended_digit_l192_192139

theorem thought_number_and_appended_digit (x y : ℕ) (hx : x > 0) (hy : y ≤ 9):
  (10 * x + y - x^2 = 8 * x) ↔ (x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8) := sorry

end thought_number_and_appended_digit_l192_192139


namespace arithmetic_problem_l192_192727

theorem arithmetic_problem : 
  let part1 := (20 / 100) * 120
  let part2 := (25 / 100) * 250
  let part3 := (15 / 100) * 80
  let sum := part1 + part2 + part3
  let subtract := (10 / 100) * 600
  sum - subtract = 38.5 := by
  sorry

end arithmetic_problem_l192_192727


namespace apples_in_bowl_l192_192806

theorem apples_in_bowl (green_plus_red_diff red_count : ℕ) (h1 : green_plus_red_diff = 12) (h2 : red_count = 16) :
  red_count + (red_count + green_plus_red_diff) = 44 :=
by
  sorry

end apples_in_bowl_l192_192806


namespace hours_of_work_l192_192684

variables (M W X : ℝ)

noncomputable def work_rate := 
  (2 * M + 3 * W) * X * 5 = 1 ∧ 
  (4 * M + 4 * W) * 3 * 7 = 1 ∧ 
  7 * M * 4 * 5.000000000000001 = 1

theorem hours_of_work (M W : ℝ) (h : work_rate M W 7) : X = 7 :=
sorry

end hours_of_work_l192_192684


namespace find_some_number_l192_192374

theorem find_some_number :
  ∃ (x : ℝ), abs (x - 0.004) < 0.0001 ∧ 9.237333333333334 = (69.28 * x) / 0.03 := by
  sorry

end find_some_number_l192_192374


namespace savings_sum_l192_192501

-- Define the values assigned to each coin type
def penny := 0.01
def nickel := 0.05
def dime := 0.10

-- Define the number of coins each person has
def teaganPennies := 200
def rexNickels := 100
def toniDimes := 330

-- Calculate the total amount saved by each person
def teaganSavings := teaganPennies * penny
def rexSavings := rexNickels * nickel
def toniSavings := toniDimes * dime

-- Calculate the total savings of all three persons together
def totalSavings := teaganSavings + rexSavings + toniSavings

theorem savings_sum : totalSavings = 40 := by
  -- the actual proof is omitted, indicated by sorry
  sorry

end savings_sum_l192_192501


namespace largest_divisor_of_n_l192_192124

theorem largest_divisor_of_n (n : ℕ) (hn : 0 < n) (h : 50 ∣ n^2) : 5 ∣ n :=
sorry

end largest_divisor_of_n_l192_192124


namespace curve_not_parabola_l192_192594

theorem curve_not_parabola (k : ℝ) : ¬(∃ (a b c d e f : ℝ), k * x^2 + y^2 = a * x^2 + b * x * y + c * y^2 + d * x + e * y + f ∧ b^2 = 4*a*c ∧ (a = 0 ∨ c = 0)) := sorry

end curve_not_parabola_l192_192594


namespace sticks_left_is_correct_l192_192369

-- Define the initial conditions
def initial_popsicle_sticks : ℕ := 170
def popsicle_sticks_per_group : ℕ := 15
def number_of_groups : ℕ := 10

-- Define the total number of popsicle sticks given out to the groups
def total_sticks_given : ℕ := popsicle_sticks_per_group * number_of_groups

-- Define the number of popsicle sticks left
def sticks_left : ℕ := initial_popsicle_sticks - total_sticks_given

-- Prove that the number of sticks left is 20
theorem sticks_left_is_correct : sticks_left = 20 :=
by
  sorry

end sticks_left_is_correct_l192_192369


namespace find_a_9_l192_192605

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)

-- Assumptions and definitions from the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def condition_one (a : ℕ → ℤ) : Prop := (a 1) + (a 2)^2 = -3
def condition_two (S : ℕ → ℤ) : Prop := S 5 = 10

-- Main theorem statement
theorem find_a_9 (h_arithmetic : arithmetic_sequence a d)
                 (h_sum : sum_of_arithmetic_sequence S a)
                 (h_cond1 : condition_one a)
                 (h_cond2 : condition_two S) : a 9 = 20 := 
sorry

end find_a_9_l192_192605


namespace find_fraction_l192_192498

-- Define the given variables and conditions
variables (x y : ℝ)
-- Assume x and y are nonzero
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
-- Assume the given condition
variable (h : (4*x + 2*y) / (2*x - 8*y) = 3)

-- Define the theorem to be proven
theorem find_fraction (h : (4*x + 2*y) / (2*x - 8*y) = 3) : (x + 4 * y) / (4 * x - y) = 1 / 3 := 
by
  sorry

end find_fraction_l192_192498


namespace sum_interior_angles_equal_diagonals_l192_192799

theorem sum_interior_angles_equal_diagonals (n : ℕ) (h : n = 4 ∨ n = 5) :
  (n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540 :=
by sorry

end sum_interior_angles_equal_diagonals_l192_192799


namespace nat_square_not_div_factorial_l192_192586

-- Define n as a natural number
def n : Nat := sorry  -- We assume n is given somewhere

-- Define a function to check if a number is prime
def is_prime (p : Nat) : Prop := sorry  -- Placeholder for prime checking function

-- The main theorem to prove
theorem nat_square_not_div_factorial (n : Nat) : (n = 4 ∨ is_prime n) → ¬ ((n * n) ∣ Nat.factorial n) := by
  sorry

end nat_square_not_div_factorial_l192_192586


namespace probability_of_at_least_19_l192_192852

-- Defining the possible coins in Anya's pocket
def coins : list ℕ := [10, 10, 5, 5, 2]

-- Function to calculate the sum of chosen coins
def sum_coins (l : list ℕ) := list.sum l

-- Function to check if the sum of chosen coins is at least 19 rubles
def at_least_19 (l : list ℕ) := (sum_coins l) ≥ 19

-- Extract all possible combinations of 3 coins from the list
def combinations (l : list ℕ) (n : ℕ) := 
  if h : n ≤ l.length then 
    (list.permutations l).dedup.map (λ p, p.take n).dedup
  else
    []

-- Specific combinations of 3 coins out of 5
def three_coin_combinations := combinations coins 3 

-- Count the number of favorable outcomes (combinations that sum to at least 19)
def favorable_combinations := list.filter at_least_19 three_coin_combinations

-- Calculate the probability
def probability := (favorable_combinations.length : ℚ) / (three_coin_combinations.length : ℚ)

-- Prove that the probability is 0.4
theorem probability_of_at_least_19 : probability = 0.4 :=
  sorry

end probability_of_at_least_19_l192_192852


namespace not_mutually_exclusive_probability_both_segments_success_expectation_successful_segments_conditional_prob_three_segments_given_l192_192884

open Classical

-- Condition definitions
variable (p : ℝ) (n : ℕ)
def segment_success_prob : ℝ := 3 / 4
def num_segments : ℕ := 4

noncomputable def prob_of_two_segments_success := (segment_success_prob * segment_success_prob : ℝ)
noncomputable def expected_successful_segments := num_segments * segment_success_prob
noncomputable def prob_three_successful_and_one_specific :=
  (3 / 4) ^ 3 * (1 / 4) * (3 choose 2)
noncomputable def exactly_three_successful :=
  (4 choose 3) * (3 / 4) ^ 3 * (1 / 4)
noncomputable def conditional_prob_welcoming_success :=
  prob_three_successful_and_one_specific / exactly_three_successful

theorem not_mutually_exclusive : ¬(prob_of_two_segments_success = 0) :=
sorry

theorem probability_both_segments_success :
  prob_of_two_segments_success = 9 / 16 :=
sorry

theorem expectation_successful_segments :
  expected_successful_segments = 3 :=
sorry

theorem conditional_prob_three_segments_given :
  conditional_prob_welcoming_success = 3 / 4 :=
sorry

end not_mutually_exclusive_probability_both_segments_success_expectation_successful_segments_conditional_prob_three_segments_given_l192_192884


namespace exists_quadratic_polynomial_distinct_remainders_l192_192733

theorem exists_quadratic_polynomial_distinct_remainders :
  ∃ (a b c : ℤ), 
    (¬ (2014 ∣ a)) ∧ 
    (∀ x y : ℤ, (1 ≤ x ∧ x ≤ 2014) ∧ (1 ≤ y ∧ y ≤ 2014) → x ≠ y → 
      (1007 * x^2 + 1008 * x + c) % 2014 ≠ (1007 * y^2 + 1008 * y + c) % 2014) :=
  sorry

end exists_quadratic_polynomial_distinct_remainders_l192_192733


namespace twelve_point_five_minutes_in_seconds_l192_192026

-- Definitions
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- Theorem: Prove that 12.5 minutes is 750 seconds
theorem twelve_point_five_minutes_in_seconds : minutes_to_seconds 12.5 = 750 :=
by 
  sorry

end twelve_point_five_minutes_in_seconds_l192_192026


namespace width_of_rect_prism_l192_192242

theorem width_of_rect_prism (w : ℝ) 
  (h : ℝ := 8) (l : ℝ := 5) (diagonal : ℝ := 17) 
  (h_diag : l^2 + w^2 + h^2 = diagonal^2) :
  w = 10 * Real.sqrt 2 :=
by
  sorry

end width_of_rect_prism_l192_192242


namespace base_conversion_zero_l192_192658

theorem base_conversion_zero (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 8 * A + B = 6 * B + A) : 8 * A + B = 0 :=
by
  sorry

end base_conversion_zero_l192_192658


namespace fraction_problem_l192_192322

theorem fraction_problem (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := 
by
  sorry

end fraction_problem_l192_192322


namespace ce_length_l192_192054

noncomputable def CE_in_parallelogram (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) : ℝ :=
  280

theorem ce_length (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) :
  CE_in_parallelogram AB AD BD AB_eq AD_eq BD_eq = 280 :=
by
  sorry

end ce_length_l192_192054


namespace cricketer_average_score_l192_192193

variable {A : ℤ} -- A represents the average score after 18 innings

theorem cricketer_average_score
  (h1 : (19 * (A + 4) = 18 * A + 98)) :
  A + 4 = 26 := by
  sorry

end cricketer_average_score_l192_192193


namespace courtyard_paving_l192_192134

noncomputable def length_of_brick (L : ℕ) := L = 12

theorem courtyard_paving  (courtyard_length : ℕ) (courtyard_width : ℕ) 
                           (brick_width : ℕ) (total_bricks : ℕ) 
                           (H1 : courtyard_length = 18) (H2 : courtyard_width = 12) 
                           (H3 : brick_width = 6) (H4 : total_bricks = 30000) 
                           : length_of_brick 12 := 
by 
  sorry

end courtyard_paving_l192_192134


namespace buffy_less_brittany_by_40_seconds_l192_192045

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

end buffy_less_brittany_by_40_seconds_l192_192045


namespace simplify_radicals_l192_192950

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
by
  sorry

end simplify_radicals_l192_192950


namespace route_difference_is_18_l192_192482

theorem route_difference_is_18 :
  let t_uphill := 6 in
  let t_path1 := 2 * t_uphill in
  let t_final1 := (t_uphill + t_path1)/3 in
  let t_route1 := t_uphill + t_path1 + t_final1 in
  let t_flat := 14 in
  let t_final2 := 2 * t_flat in
  let t_route2 := t_flat + t_final2 in
  t_route2 - t_route1 = 18 :=
by
  let t_uphill := 6
  let t_path1 := 2 * t_uphill
  let t_final1 := (t_uphill + t_path1)/3
  let t_route1 := t_uphill + t_path1 + t_final1
  let t_flat := 14
  let t_final2 := 2 * t_flat
  let t_route2 := t_flat + t_final2
  have h : t_route2 - t_route1 = 18
  sorry

end route_difference_is_18_l192_192482


namespace howard_rewards_l192_192860

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

end howard_rewards_l192_192860


namespace complex_expression_equality_l192_192170

-- Define the basic complex number properties and operations.
def i : ℂ := Complex.I -- Define the imaginary unit

theorem complex_expression_equality (a b : ℤ) :
  (3 - 4 * i) * ((-4 + 2 * i) ^ 2) = -28 - 96 * i :=
by
  -- Syntactical proof placeholders
  sorry

end complex_expression_equality_l192_192170


namespace perpendicular_sum_value_of_m_l192_192336

-- Let a and b be defined as vectors in R^2
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product for vectors in R^2
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors using dot product
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the sum of two vectors
def vector_sum (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- State our proof problem
theorem perpendicular_sum_value_of_m :
  is_perpendicular (vector_sum vector_a (vector_b (-7 / 2))) vector_a :=
by
  -- Proof omitted
  sorry

end perpendicular_sum_value_of_m_l192_192336


namespace taehyung_walks_more_than_minyoung_l192_192651

def taehyung_distance_per_minute : ℕ := 114
def minyoung_distance_per_minute : ℕ := 79
def minutes_per_hour : ℕ := 60

theorem taehyung_walks_more_than_minyoung :
  (taehyung_distance_per_minute * minutes_per_hour) -
  (minyoung_distance_per_minute * minutes_per_hour) = 2100 := by
  sorry

end taehyung_walks_more_than_minyoung_l192_192651


namespace find_a_values_l192_192436

theorem find_a_values (a t t₁ t₂ : ℝ) :
  (t^2 + (a - 6) * t + (9 - 3 * a) = 0) ∧
  (t₁ = 4 * t₂) ∧
  (t₁ + t₂ = 6 - a) ∧
  (t₁ * t₂ = 9 - 3 * a)
  ↔ (a = -2 ∨ a = 2) := sorry

end find_a_values_l192_192436


namespace tickets_sold_l192_192980

theorem tickets_sold (S G : ℕ) (hG : G = 388) (h_total : 4 * S + 6 * G = 2876) :
  S + G = 525 := by
  sorry

end tickets_sold_l192_192980


namespace ledi_age_10_in_years_l192_192402

-- Definitions of ages of Duoduo and Ledi
def duoduo_current_age : ℝ := 10
def years_ago : ℝ := 12.3
def sum_ages_years_ago : ℝ := 12

-- Function to calculate Ledi's current age
def ledi_current_age :=
  (sum_ages_years_ago + years_ago + years_ago) + (duoduo_current_age - years_ago)

-- Function to calculate years from now for Ledi to be 10 years old
def years_until_ledi_age_10 (ledi_age_now : ℝ) : ℝ :=
  10 - ledi_age_now

-- Main statement we need to prove
theorem ledi_age_10_in_years : years_until_ledi_age_10 ledi_current_age = 6.3 :=
by
  -- Proof goes here
  sorry

end ledi_age_10_in_years_l192_192402


namespace area_of_triangle_ABC_l192_192832

theorem area_of_triangle_ABC 
  (r : ℝ) (R : ℝ) (ACB : ℝ) 
  (hr : r = 2) 
  (hR : R = 4) 
  (hACB : ACB = 120) : 
  let s := (2 * (2 + 4 * Real.sqrt 3)) / Real.sqrt 3 
  let S := s * r 
  S = 56 / Real.sqrt 3 :=
sorry

end area_of_triangle_ABC_l192_192832


namespace greatest_m_value_l192_192422

theorem greatest_m_value (x y z u : ℕ) (hx : x ≥ y) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ m ≤ x / y :=
sorry

end greatest_m_value_l192_192422


namespace proposition_A_proposition_B_proposition_C_proposition_D_l192_192395

-- Definitions and conditions for proposition A
def propA_conditions (a b : ℝ) : Prop :=
  a > b ∧ (1 / a) > (1 / b)

def propA (a b : ℝ) : Prop :=
  a * b < 0

-- Definitions and conditions for proposition B
def propB_conditions (a b : ℝ) : Prop :=
  a < b ∧ b < 0

def propB (a b : ℝ) : Prop :=
  a^2 < a * b ∧ a * b < b^2

-- Definitions and conditions for proposition C
def propC_conditions (c a b : ℝ) : Prop :=
  c > a ∧ a > b ∧ b > 0

def propC (c a b : ℝ) : Prop :=
  (a / (c - a)) < (b / (c - b))

-- Definitions and conditions for proposition D
def propD_conditions (a b c : ℝ) : Prop :=
  a > b ∧ b > c ∧ c > 0

def propD (a b c : ℝ) : Prop :=
  (a / b) > ((a + c) / (b + c))

-- The propositions
theorem proposition_A (a b : ℝ) (h : propA_conditions a b) : propA a b := 
sorry

theorem proposition_B (a b : ℝ) (h : propB_conditions a b) : ¬ propB a b :=
sorry

theorem proposition_C (c a b : ℝ) (h : propC_conditions c a b) : ¬ propC c a b :=
sorry

theorem proposition_D (a b c : ℝ) (h : propD_conditions a b c) : propD a b c :=
sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l192_192395


namespace relationship_among_ys_l192_192447

-- Define the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ :=
  -2 * x + b

-- Define the points on the graph
def y1 (b : ℝ) : ℝ :=
  linear_function (-2) b

def y2 (b : ℝ) : ℝ :=
  linear_function (-1) b

def y3 (b : ℝ) : ℝ :=
  linear_function 1 b

-- Theorem to prove the relation among y1, y2, y3
theorem relationship_among_ys (b : ℝ) : y1 b > y2 b ∧ y2 b > y3 b :=
by
  sorry

end relationship_among_ys_l192_192447


namespace tournament_participants_l192_192202

theorem tournament_participants (n : ℕ) (h : (n * (n - 1)) / 2 = 171) : n = 19 :=
by
  sorry

end tournament_participants_l192_192202


namespace sum_of_roots_of_quadratic_l192_192525

theorem sum_of_roots_of_quadratic :
  let a := 2
  let b := -8
  let c := 6
  let sum_of_roots := (-b / a)
  2 * (sum_of_roots) * sum_of_roots - 8 * sum_of_roots + 6 = 0 :=
by
  sorry

end sum_of_roots_of_quadratic_l192_192525


namespace age_difference_in_decades_l192_192546

-- Declare the ages of x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the condition
def age_condition (x y z : ℝ) : Prop := x + y = y + z + 18

-- The proof problem statement
theorem age_difference_in_decades (h : age_condition x y z) : (x - z) / 10 = 1.8 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end age_difference_in_decades_l192_192546


namespace anya_probability_l192_192854

open Finset

def possible_coins := {10, 10, 5, 5, 2}
def target := 19

noncomputable def combinations := (possible_coins.vals.ctype_power 3).val.filter (λ s, Finset.sum s >= target)

noncomputable def probability : ℝ :=
  (combinations.card : ℝ) / (possible_coins.vals.ctype_power 3).card

theorem anya_probability : probability = 0.4 := sorry

end anya_probability_l192_192854


namespace aluminum_carbonate_weight_l192_192689

-- Define the atomic weights
def Al : ℝ := 26.98
def C : ℝ := 12.01
def O : ℝ := 16.00

-- Define the molecular weight of aluminum carbonate
def molecularWeightAl2CO3 : ℝ := (2 * Al) + (3 * C) + (9 * O)

-- Define the number of moles
def moles : ℝ := 5

-- Calculate the total weight of 5 moles of aluminum carbonate
def totalWeight : ℝ := moles * molecularWeightAl2CO3

-- Statement to prove
theorem aluminum_carbonate_weight : totalWeight = 1169.95 :=
by {
  sorry
}

end aluminum_carbonate_weight_l192_192689


namespace distinct_flavors_count_l192_192499

-- Define the number of available candies
def red_candies := 3
def green_candies := 2
def blue_candies := 4

-- Define what it means for a flavor to be valid: includes at least one candy of each color.
def is_valid_flavor (x y z : Nat) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x ≤ red_candies ∧ y ≤ green_candies ∧ z ≤ blue_candies

-- Define what it means for two flavors to have the same ratio
def same_ratio (x1 y1 z1 x2 y2 z2 : Nat) : Prop :=
  x1 * y2 * z2 = x2 * y1 * z1

-- Define the proof problem: the number of distinct flavors
theorem distinct_flavors_count :
  ∃ n, n = 21 ∧ ∀ (x y z : Nat), is_valid_flavor x y z ↔ (∃ x' y' z', is_valid_flavor x' y' z' ∧ ¬ same_ratio x y z x' y' z') :=
sorry

end distinct_flavors_count_l192_192499


namespace solve_for_k_in_quadratic_l192_192441

theorem solve_for_k_in_quadratic :
  ∃ k : ℝ, (∀ x1 x2 : ℝ,
    x1 + x2 = 3 ∧
    x1 * x2 + 2 * x1 + 2 * x2 = 1 ∧
    (x1^2 - 3*x1 + k = 0) ∧ (x2^2 - 3*x2 + k = 0)) →
  k = -5 :=
sorry

end solve_for_k_in_quadratic_l192_192441


namespace last_box_weight_l192_192220

theorem last_box_weight (a b c : ℕ) (h1 : a = 2) (h2 : b = 11) (h3 : a + b + c = 18) : c = 5 :=
by
  sorry

end last_box_weight_l192_192220


namespace jim_gas_gallons_l192_192625

theorem jim_gas_gallons (G : ℕ) (C_NC C_VA : ℕ → ℕ) 
  (h₁ : ∀ G, C_NC G = 2 * G)
  (h₂ : ∀ G, C_VA G = 3 * G)
  (h₃ : C_NC G + C_VA G = 50) :
  G = 10 := 
sorry

end jim_gas_gallons_l192_192625


namespace range_of_a_l192_192032

noncomputable def isNotPurelyImaginary (a : ℝ) : Prop :=
  let re := a^2 - a - 2
  re ≠ 0

theorem range_of_a (a : ℝ) (h : isNotPurelyImaginary a) : a ≠ -1 :=
  sorry

end range_of_a_l192_192032


namespace pairs_satisfied_condition_l192_192269

def set_A : Set ℕ := {1, 2, 3, 4, 5, 6, 10, 11, 12, 15, 20, 22, 30, 33, 44, 55, 60, 66, 110, 132, 165, 220, 330, 660}
def set_B : Set ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def is_valid_pair (a b : ℕ) := a ∈ set_A ∧ b ∈ set_B ∧ (a - b = 4)

def valid_pairs : Set (ℕ × ℕ) := 
  {(6, 2), (10, 6), (12, 8), (22, 18)}

theorem pairs_satisfied_condition :
  { (a, b) | is_valid_pair a b } = valid_pairs := 
sorry

end pairs_satisfied_condition_l192_192269


namespace det_scaled_matrix_l192_192597

variable {R : Type*} [CommRing R]

def det2x2 (a b c d : R) : R := a * d - b * c

theorem det_scaled_matrix 
  (x y z w : R) 
  (h : det2x2 x y z w = 3) : 
  det2x2 (3 * x) (3 * y) (6 * z) (6 * w) = 54 := by
  sorry

end det_scaled_matrix_l192_192597


namespace sum_of_two_digit_factors_of_8060_l192_192732

theorem sum_of_two_digit_factors_of_8060 : ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 8060) ∧ (a + b = 127) :=
by sorry

end sum_of_two_digit_factors_of_8060_l192_192732


namespace expand_and_simplify_l192_192873

theorem expand_and_simplify :
  ∀ (x : ℝ), 2 * x * (3 * x ^ 2 - 4 * x + 5) - (x ^ 2 - 3 * x) * (4 * x + 5) = 2 * x ^ 3 - x ^ 2 + 25 * x :=
by
  intro x
  sorry

end expand_and_simplify_l192_192873


namespace reading_enhusiasts_not_related_to_gender_l192_192565

noncomputable def contingency_table (boys_scores : List Nat) (girls_scores : List Nat) :
  (Nat × Nat × Nat × Nat × Nat × Nat) × (Nat × Nat × Nat × Nat × Nat × Nat) :=
  let boys_range := (2, 3, 5, 15, 18, 12)
  let girls_range := (0, 5, 10, 10, 7, 13)
  ((2, 3, 5, 15, 18, 12), (0, 5, 10, 10, 7, 13))

theorem reading_enhusiasts_not_related_to_gender (boys_scores : List Nat) (girls_scores : List Nat) :
  let table := contingency_table boys_scores girls_scores
  let (boys_range, girls_range) := table
  let a := 45 -- Boys who are reading enthusiasts
  let b := 10 -- Boys who are non-reading enthusiasts
  let c := 30 -- Girls who are reading enthusiasts
  let d := 15 -- Girls who are non-reading enthusiasts
  let n := a + b + c + d
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  k_squared < 3.841 := 
sorry

end reading_enhusiasts_not_related_to_gender_l192_192565


namespace code_XYZ_to_base_10_l192_192556

def base_6_to_base_10 (x y z : ℕ) : ℕ :=
  x * 6^2 + y * 6^1 + z * 6^0

theorem code_XYZ_to_base_10 :
  ∀ (X Y Z : ℕ), 
    X = 5 ∧ Y = 0 ∧ Z = 4 →
    base_6_to_base_10 X Y Z = 184 :=
by
  intros X Y Z h
  cases' h with hX hYZ
  cases' hYZ with hY hZ
  rw [hX, hY, hZ]
  exact rfl

end code_XYZ_to_base_10_l192_192556


namespace rain_probability_l192_192125

theorem rain_probability :
  let PM : ℝ := 0.62
  let PT : ℝ := 0.54
  let PMcTc : ℝ := 0.28
  let PMT : ℝ := PM + PT - (1 - PMcTc)
  PMT = 0.44 :=
by
  sorry

end rain_probability_l192_192125


namespace g_of_neg_2_l192_192928

def f (x : ℚ) : ℚ := 4 * x - 9

def g (y : ℚ) : ℚ :=
  3 * ((y + 9) / 4)^2 - 4 * ((y + 9) / 4) + 2

theorem g_of_neg_2 : g (-2) = 67 / 16 :=
by
  sorry

end g_of_neg_2_l192_192928


namespace closest_cube_root_l192_192117

theorem closest_cube_root :
  ∀ n : ℤ, abs (n^3 - 250) ≥ abs (6^3 - 250) := by
  sorry

end closest_cube_root_l192_192117


namespace intersection_of_sets_l192_192190

open Set

theorem intersection_of_sets (M N : Set ℕ) (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) :
  M ∩ N = {2, 3} :=
by
  sorry

end intersection_of_sets_l192_192190


namespace probability_is_0_4_l192_192856

def coin_values : List ℕ := [10, 10, 5, 5, 2]

def valid_combination (comb : List ℕ) : Prop :=
  comb.sum ≥ 19

def favorable_outcomes : Finset (Finset ℕ) :=
  {s ∈ coin_values.to_finset.powerset.filter (λ s, s.card = 3) | valid_combination s.val.to_list}

def total_outcomes : Finset (Finset ℕ) :=
  coin_values.to_finset.powerset.filter (λ s, s.card = 3)

def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

theorem probability_is_0_4 : probability = 2 / 5 :=
by
  -- Proof will go here
  sorry

end probability_is_0_4_l192_192856


namespace wife_catch_up_l192_192839

/-- A man drives at a speed of 40 miles/hr.
His wife left 30 minutes late with a speed of 50 miles/hr.
Prove that they will meet 2 hours after the wife starts driving. -/
theorem wife_catch_up (t : ℝ) (speed_man speed_wife : ℝ) (late_time : ℝ) :
  speed_man = 40 →
  speed_wife = 50 →
  late_time = 0.5 →
  50 * t = 40 * (t + 0.5) →
  t = 2 :=
by
  intros h_man h_wife h_late h_eq
  -- Actual proof goes here. 
  -- (Skipping the proof as requested, leaving it as a placeholder)
  sorry

end wife_catch_up_l192_192839


namespace find_center_of_circle_l192_192173

theorem find_center_of_circle (x y : ℝ) :
  4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 16 = 0 →
  (x + 1)^2 + (y - 3)^2 = 6 :=
by
  intro h
  sorry

end find_center_of_circle_l192_192173


namespace find_difference_between_larger_and_fraction_smaller_l192_192427

theorem find_difference_between_larger_and_fraction_smaller
  (x y : ℝ) 
  (h1 : x + y = 147)
  (h2 : x - 0.375 * y = 4) : x - 0.375 * y = 4 :=
by
  sorry

end find_difference_between_larger_and_fraction_smaller_l192_192427


namespace log_sum_equality_l192_192153

noncomputable def log_base_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_sum_equality :
  2 * log_base_5 10 + log_base_5 0.25 = 2 :=
by
  sorry -- proof goes here

end log_sum_equality_l192_192153


namespace volume_displacement_square_l192_192136

-- Define the given conditions
def radius_cylinder := 5
def height_cylinder := 12
def side_length_cube := 10

theorem volume_displacement_square :
  let r := radius_cylinder
  let h := height_cylinder
  let s := side_length_cube
  let cube_diagonal := s * Real.sqrt 3
  let w := (125 * Real.sqrt 6) / 8
  w^2 = 1464.0625 :=
by
  sorry

end volume_displacement_square_l192_192136


namespace abs_diff_roots_quad_eq_one_l192_192874

theorem abs_diff_roots_quad_eq_one : 
  ∀ r1 r2 : ℝ, (r1 + r2 = 7) ∧ (r1 * r2 = 12) → |r1 - r2| = 1 :=
by
  intro r1 r2 h
  have h_sum := h.1
  have h_prod := h.2
  sorry

end abs_diff_roots_quad_eq_one_l192_192874


namespace solution_exists_l192_192870

def divide_sum_of_squares_and_quotient_eq_seventy_two (x : ℝ) : Prop :=
  (10 - x)^2 + x^2 + (10 - x) / x = 72

theorem solution_exists (x : ℝ) : divide_sum_of_squares_and_quotient_eq_seventy_two x → x = 2 := sorry

end solution_exists_l192_192870


namespace hyperbola_asymptote_l192_192076

theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (y = (1/2) * x) ∨ (y = -(1/2) * x) :=
by
  intros x y h
  sorry

end hyperbola_asymptote_l192_192076


namespace total_fish_l192_192227

def LillyFish : ℕ := 10
def RosyFish : ℕ := 8
def MaxFish : ℕ := 15

theorem total_fish : LillyFish + RosyFish + MaxFish = 33 := by
  sorry

end total_fish_l192_192227


namespace total_ttaki_count_l192_192291

noncomputable def total_ttaki_used (n : ℕ): ℕ := n * n

theorem total_ttaki_count {n : ℕ} (h : 4 * n - 4 = 240) : total_ttaki_used n = 3721 := by
  sorry

end total_ttaki_count_l192_192291


namespace probability_X_eq_4_l192_192142

-- Define the number of students and boys
def total_students := 15
def total_boys := 7
def selected_students := 10

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := n.choose k

-- Calculate the probability
def P_X_eq_4 := (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students

-- The statement to be proven
theorem probability_X_eq_4 :
  P_X_eq_4 = (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students := by
  sorry

end probability_X_eq_4_l192_192142


namespace rectangle_perimeter_l192_192505

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) : 2 * (L + B) = 186 :=
by
  sorry

end rectangle_perimeter_l192_192505


namespace zoe_candy_bars_needed_l192_192271

def total_cost : ℝ := 485
def grandma_contribution : ℝ := 250
def per_candy_earning : ℝ := 1.25
def required_candy_bars : ℕ := 188

theorem zoe_candy_bars_needed :
  (total_cost - grandma_contribution) / per_candy_earning = required_candy_bars :=
by
  sorry

end zoe_candy_bars_needed_l192_192271


namespace total_flowers_bouquets_l192_192771

-- Define the number of tulips Lana picked
def tulips : ℕ := 36

-- Define the number of roses Lana picked
def roses : ℕ := 37

-- Define the number of extra flowers Lana picked
def extra_flowers : ℕ := 3

-- Prove that the total number of flowers used by Lana for the bouquets is 76
theorem total_flowers_bouquets : (tulips + roses + extra_flowers) = 76 :=
by
  sorry

end total_flowers_bouquets_l192_192771


namespace permutation_value_l192_192550

theorem permutation_value : ∀ (n r : ℕ), n = 5 → r = 3 → (n.choose r) * r.factorial = 60 := 
by
  intros n r hn hr 
  rw [hn, hr]
  -- We use the permutation formula A_{n}^{r} = n! / (n-r)!
  -- A_{5}^{3} = 5! / 2!
  -- Simplifies to 5 * 4 * 3 = 60.
  sorry

end permutation_value_l192_192550


namespace volume_of_cut_cone_l192_192706

theorem volume_of_cut_cone (V_frustum : ℝ) (A_bottom : ℝ) (A_top : ℝ) (V_cut_cone : ℝ) :
  V_frustum = 52 ∧ A_bottom = 9 * A_top → V_cut_cone = 54 :=
by
  sorry

end volume_of_cut_cone_l192_192706


namespace total_chickens_l192_192939

open Nat

theorem total_chickens 
  (Q S C : ℕ) 
  (h1 : Q = 2 * S + 25) 
  (h2 : S = 3 * C - 4) 
  (h3 : C = 37) : 
  Q + S + C = 383 := by
  sorry

end total_chickens_l192_192939


namespace cakes_served_dinner_l192_192559

def total_cakes_today : Nat := 15
def cakes_served_lunch : Nat := 6

theorem cakes_served_dinner : total_cakes_today - cakes_served_lunch = 9 :=
by
  -- Define what we need to prove
  sorry -- to skip the proof

end cakes_served_dinner_l192_192559


namespace sonny_cookie_problem_l192_192787

theorem sonny_cookie_problem 
  (total_boxes : ℕ) (boxes_sister : ℕ) (boxes_cousin : ℕ) (boxes_left : ℕ) (boxes_brother : ℕ) : 
  total_boxes = 45 → boxes_sister = 9 → boxes_cousin = 7 → boxes_left = 17 → 
  boxes_brother = total_boxes - boxes_left - boxes_sister - boxes_cousin → 
  boxes_brother = 12 :=
by
  intros h_total h_sister h_cousin h_left h_brother
  rw [h_total, h_sister, h_cousin, h_left] at h_brother
  exact h_brother

end sonny_cookie_problem_l192_192787


namespace prob_at_least_one_heart_spade_or_king_l192_192405

theorem prob_at_least_one_heart_spade_or_king :
  let total_cards := 52
  let hearts := 13
  let spades := 13
  let kings := 4
  let unique_hsk := hearts + spades + 2  -- Two unique kings from other suits
  let prob_not_hsk := (total_cards - unique_hsk) / total_cards
  let prob_not_hsk_two_draws := prob_not_hsk * prob_not_hsk
  let prob_at_least_one_hsk := 1 - prob_not_hsk_two_draws
  prob_at_least_one_hsk = 133 / 169 :=
by sorry

end prob_at_least_one_heart_spade_or_king_l192_192405


namespace fraction_of_quarters_from_1860_to_1869_l192_192784

theorem fraction_of_quarters_from_1860_to_1869
  (total_quarters : ℕ) (quarters_from_1860s : ℕ)
  (h1 : total_quarters = 30) (h2 : quarters_from_1860s = 15) :
  (quarters_from_1860s : ℚ) / (total_quarters : ℚ) = 1 / 2 := by
  sorry

end fraction_of_quarters_from_1860_to_1869_l192_192784


namespace inequality_solution_sets_l192_192911

theorem inequality_solution_sets (a : ℝ)
  (h1 : ∀ x : ℝ, (1/2) < x ∧ x < 2 ↔ ax^2 + 5*x - 2 > 0) :
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) ↔ ax^2 - 5*x + a^2 - 1 > 0) :=
by {
  sorry
}

end inequality_solution_sets_l192_192911


namespace mixture_replacement_l192_192290

theorem mixture_replacement (A B x : ℕ) (hA : A = 32) (h_ratio1 : A / B = 4) (h_ratio2 : A / (B + x) = 2 / 3) : x = 40 :=
by
  sorry

end mixture_replacement_l192_192290


namespace arith_seq_ninth_term_value_l192_192606

variable {a : Nat -> ℤ}
variable {S : Nat -> ℤ}

def arith_seq (a : Nat -> ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + a 1^2

def arith_sum (S : Nat -> ℤ) (a : Nat -> ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem arith_seq_ninth_term_value
  (h_seq : arith_seq a)
  (h_sum : arith_sum S a)
  (h_cond1 : a 1 + a 2^2 = -3)
  (h_cond2 : S 5 = 10) :
  a 9 = 20 :=
by
  sorry

end arith_seq_ninth_term_value_l192_192606


namespace avg_goals_l192_192463

-- Let's declare the variables and conditions
def layla_goals : ℕ := 104
def games_played : ℕ := 4
def less_goals_kristin : ℕ := 24

-- Define the number of goals Kristin scored
def kristin_goals : ℕ := layla_goals - less_goals_kristin

-- Calculate the total number of goals scored by both
def total_goals : ℕ := layla_goals + kristin_goals

-- Calculate the average number of goals per game
def average_goals_per_game : ℕ := total_goals / games_played

-- The theorem statement
theorem avg_goals : average_goals_per_game = 46 := by
  -- proof skipped, assume correct by using sorry
  sorry

end avg_goals_l192_192463


namespace percent_of_div_l192_192552

theorem percent_of_div (P: ℝ) (Q: ℝ) (R: ℝ) : ( ( P / 100 ) * Q ) / R = 354.2 :=
by
  -- Given P = 168, Q = 1265, R = 6
  let P := 168
  let Q := 1265
  let R := 6
  -- sorry to skip the actual proof.
  sorry

end percent_of_div_l192_192552


namespace part_i_l192_192128

theorem part_i (n : ℕ) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end part_i_l192_192128


namespace income_of_person_l192_192508

theorem income_of_person (x: ℝ) (h : 9 * x - 8 * x = 2000) : 9 * x = 18000 :=
by
  sorry

end income_of_person_l192_192508


namespace anya_probability_l192_192853

open Finset

def possible_coins := {10, 10, 5, 5, 2}
def target := 19

noncomputable def combinations := (possible_coins.vals.ctype_power 3).val.filter (λ s, Finset.sum s >= target)

noncomputable def probability : ℝ :=
  (combinations.card : ℝ) / (possible_coins.vals.ctype_power 3).card

theorem anya_probability : probability = 0.4 := sorry

end anya_probability_l192_192853


namespace find_xy_l192_192358

theorem find_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 :=
sorry

end find_xy_l192_192358


namespace num_cows_on_farm_l192_192207

variables (D C S : ℕ)

def total_legs : ℕ := 8 * S + 2 * D + 4 * C
def total_heads : ℕ := D + C + S

theorem num_cows_on_farm
  (h1 : S = 2 * D)
  (h2 : total_legs D C S = 2 * total_heads D C S + 72)
  (h3 : D + C + S ≤ 40) :
  C = 30 :=
sorry

end num_cows_on_farm_l192_192207


namespace three_digit_powers_of_two_l192_192906

theorem three_digit_powers_of_two : 
  ∃ (N : ℕ), N = 3 ∧ ∀ (n : ℕ), (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by
  sorry

end three_digit_powers_of_two_l192_192906


namespace car_speed_first_hour_l192_192672

theorem car_speed_first_hour (x : ℕ) (hx : x = 65) : 
  let speed_second_hour := 45 
  let average_speed := 55
  (x + 45) / 2 = 55 
  :=
  by
  sorry

end car_speed_first_hour_l192_192672


namespace line_equation_of_intersection_points_l192_192328

theorem line_equation_of_intersection_points (x y : ℝ) :
  (x^2 + y^2 - 6*x - 7 = 0) ∧ (x^2 + y^2 - 6*y - 27 = 0) → (3*x - 3*y = 10) :=
by
  sorry

end line_equation_of_intersection_points_l192_192328


namespace solve_for_y_l192_192971

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l192_192971


namespace simplify_sum_of_square_roots_l192_192953

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l192_192953


namespace value_of_expression_l192_192318

theorem value_of_expression (a b : ℝ) (h : 2 * a + 4 * b = 3) : 4 * a + 8 * b - 2 = 4 := 
by 
  sorry

end value_of_expression_l192_192318


namespace dot_product_of_a_and_b_is_correct_l192_192775

-- Define vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -1)

-- Define dot product for ℝ × ℝ vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem statement (proof can be omitted with sorry)
theorem dot_product_of_a_and_b_is_correct : dot_product a b = -4 :=
by
  -- proof goes here, omitted for now
  sorry

end dot_product_of_a_and_b_is_correct_l192_192775


namespace smallest_prime_perimeter_l192_192431

def is_prime (n : ℕ) := Nat.Prime n
def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a
def is_scalene (a b c : ℕ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ a ≥ 5
  ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
by
  sorry

end smallest_prime_perimeter_l192_192431


namespace initialPersonsCount_l192_192657

noncomputable def numberOfPersonsInitially (increaseInAverageWeight kg_diff : ℝ) : ℝ :=
  kg_diff / increaseInAverageWeight

theorem initialPersonsCount :
  numberOfPersonsInitially 2.5 20 = 8 := by
  sorry

end initialPersonsCount_l192_192657


namespace modulus_of_z_l192_192010

noncomputable def z : ℂ := (Complex.I / (1 + 2 * Complex.I))

theorem modulus_of_z : Complex.abs z = (Real.sqrt 5) / 5 := by
  sorry

end modulus_of_z_l192_192010


namespace angle_A_measure_l192_192912

variable {a b c A : ℝ}

def vector_m (b c a : ℝ) : ℝ × ℝ := (b, c - a)
def vector_n (b c a : ℝ) : ℝ × ℝ := (b - c, c + a)

theorem angle_A_measure (h_perpendicular : (vector_m b c a).1 * (vector_n b c a).1 + (vector_m b c a).2 * (vector_n b c a).2 = 0) :
  A = 2 * π / 3 := sorry

end angle_A_measure_l192_192912


namespace probability_of_b_l192_192380

noncomputable def P : ℕ → ℝ := sorry

axiom P_a : P 0 = 0.15
axiom P_a_and_b : P 1 = 0.15
axiom P_neither_a_nor_b : P 2 = 0.6

theorem probability_of_b : P 3 = 0.4 := 
by
  sorry

end probability_of_b_l192_192380


namespace sqrt_sum_simplify_l192_192944

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l192_192944


namespace sqrt_sum_simplify_l192_192946

theorem sqrt_sum_simplify :
  Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 :=
sorry

end sqrt_sum_simplify_l192_192946


namespace min_score_needed_l192_192925

-- Definitions of the conditions
def current_scores : List ℤ := [88, 92, 75, 81, 68, 70]
def desired_increase : ℤ := 5
def number_of_tests := current_scores.length
def current_total : ℤ := current_scores.sum
def current_average : ℤ := current_total / number_of_tests
def desired_average : ℤ := current_average + desired_increase 
def new_number_of_tests : ℤ := number_of_tests + 1
def total_required_score : ℤ := desired_average * new_number_of_tests

-- Lean 4 statement (theorem) to prove
theorem min_score_needed : total_required_score - current_total = 114 := by
  sorry

end min_score_needed_l192_192925


namespace sin_geq_tan_minus_half_tan_cubed_l192_192938

theorem sin_geq_tan_minus_half_tan_cubed (x : ℝ) (hx : 0 ≤ x ∧ x < π / 2) :
  Real.sin x ≥ Real.tan x - 1/2 * (Real.tan x) ^ 3 := 
sorry

end sin_geq_tan_minus_half_tan_cubed_l192_192938


namespace triangle_area_scaled_l192_192910

theorem triangle_area_scaled (a b : ℝ) (θ : ℝ) :
  let A := 1/2 * a * b * Real.sin θ
  let a' := 3 * a
  let b' := 2 * b
  let A' := 1/2 * a' * b' * Real.sin θ
  A' = 6 * A := by
  sorry

end triangle_area_scaled_l192_192910


namespace pipe_Q_fill_time_l192_192640

theorem pipe_Q_fill_time (x : ℝ) (h1 : 6 > 0)
    (h2 : 24 > 0)
    (h3 : 3.4285714285714284 > 0)
    (h4 : (1 / 6) + (1 / x) + (1 / 24) = 1 / 3.4285714285714284) :
    x = 8 := by
  sorry

end pipe_Q_fill_time_l192_192640


namespace lily_distance_from_start_l192_192362

open Real

def north_south_net := 40 - 10 -- 30 meters south
def east_west_net := 30 - 15 -- 15 meters east

theorem lily_distance_from_start : 
  ∀ (north_south : ℝ) (east_west : ℝ), 
    north_south = north_south_net → 
    east_west = east_west_net → 
    distance = Real.sqrt ((north_south * north_south) + (east_west * east_west)) → 
    distance = 15 * Real.sqrt 5 :=
by
  intros
  sorry

end lily_distance_from_start_l192_192362


namespace force_on_dam_l192_192725

noncomputable def calculate_force (ρ g a b h : ℝ) :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem force_on_dam :
  let ρ := 1000
  let g := 10
  let a := 6.0
  let b := 9.6
  let h := 4.0
  calculate_force ρ g a b h = 576000 :=
by sorry

end force_on_dam_l192_192725


namespace red_balls_count_after_game_l192_192470

structure BagState :=
  (red : Nat)         -- Number of red balls
  (green : Nat)       -- Number of green balls
  (blue : Nat)        -- Number of blue balls
  (yellow : Nat)      -- Number of yellow balls
  (black : Nat)       -- Number of black balls
  (white : Nat)       -- Number of white balls)

def initialBallCount (totalBalls : Nat) : BagState :=
  let totalRatio := 15 + 13 + 17 + 9 + 7 + 23
  { red := totalBalls * 15 / totalRatio
  , green := totalBalls * 13 / totalRatio
  , blue := totalBalls * 17 / totalRatio
  , yellow := totalBalls * 9 / totalRatio
  , black := totalBalls * 7 / totalRatio
  , white := totalBalls * 23 / totalRatio
  }

def finalBallCount (initialState : BagState) : BagState :=
  { red := initialState.red + 400
  , green := initialState.green - 250
  , blue := initialState.blue
  , yellow := initialState.yellow - 100
  , black := initialState.black + 200
  , white := initialState.white - 500
  }

theorem red_balls_count_after_game :
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  final.red = 2185 :=
by
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  sorry

end red_balls_count_after_game_l192_192470


namespace Patrick_can_play_l192_192056

def friends_prob := 1/2
def participants_needed := 5

noncomputable def binomial (n k: ℕ) := nat.choose n k

theorem Patrick_can_play :
  ∑ k in (finset.range (10 + 1)).filter (λ n, n ≥ participants_needed),
    binomial 10 k * (friends_prob ^ k) * ((1 - friends_prob) ^ (10 - k)) = 319/512 :=
by sorry

end Patrick_can_play_l192_192056


namespace mira_result_l192_192932

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n / 100 * 100 + 100 else n / 100 * 100

theorem mira_result :
  round_to_nearest_hundred ((63 + 48) - 21) = 100 :=
by
  sorry

end mira_result_l192_192932


namespace number_of_terms_is_13_l192_192203

-- Define sum of first three terms
def sum_first_three (a d : ℤ) : ℤ := a + (a + d) + (a + 2 * d)

-- Define sum of last three terms when the number of terms is n
def sum_last_three (a d : ℤ) (n : ℕ) : ℤ := (a + (n - 3) * d) + (a + (n - 2) * d) + (a + (n - 1) * d)

-- Define sum of all terms in the sequence
def sum_all_terms (a d : ℤ) (n : ℕ) : ℤ := n / 2 * (2 * a + (n - 1) * d)

-- Given conditions
def condition_one (a d : ℤ) : Prop := sum_first_three a d = 34
def condition_two (a d : ℤ) (n : ℕ) : Prop := sum_last_three a d n = 146
def condition_three (a d : ℤ) (n : ℕ) : Prop := sum_all_terms a d n = 390

-- Theorem to prove that n = 13
theorem number_of_terms_is_13 (a d : ℤ) (n : ℕ) :
  condition_one a d →
  condition_two a d n →
  condition_three a d n →
  n = 13 :=
by sorry

end number_of_terms_is_13_l192_192203


namespace derivative_at_2_l192_192539

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  a * Real.log x + b / x

theorem derivative_at_2 :
    ∃ (a b : ℝ),
    (f 1 a b = -2) ∧
    (∀ x, deriv (λ x, f x a b) x = (a / x) + (b / (x*x))) ∧
    (deriv (λ x, f x a b) 1 = 0) ∧
    (deriv (λ x, f x a b) 2 = -1/2) :=
begin
  sorry
end

end derivative_at_2_l192_192539


namespace intersection_S_T_l192_192752

def S : Set ℝ := { x | (x - 2) * (x - 3) >= 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T :
  S ∩ T = { x | (0 < x ∧ x <= 2) ∨ (x >= 3) } := by
  sorry

end intersection_S_T_l192_192752


namespace line_canonical_form_l192_192697

theorem line_canonical_form :
  (∀ x y z : ℝ, 4 * x + y - 3 * z + 2 = 0 → 2 * x - y + z - 8 = 0 ↔
    ∃ t : ℝ, x = 1 + -2 * t ∧ y = -6 + -10 * t ∧ z = -6 * t) :=
by
  sorry

end line_canonical_form_l192_192697


namespace probability_of_at_least_one_three_l192_192416

noncomputable def probability_at_least_one_three_tossed : ℚ :=
  10 / 21

theorem probability_of_at_least_one_three 
  (X1 X2 X3 : ℕ) 
  (h1 : 1 ≤ X1 ∧ X1 ≤ 8)
  (h2 : 1 ≤ X2 ∧ X2 ≤ 8)
  (h3 : 1 ≤ X3 ∧ X3 ≤ 8)
  (cond : X1 + X2 = X3 + 1) :
  let event_has_three := (X1 = 3 ∨ X2 = 3 ∨ X3 = 3) in
  (event_has_three ↔ true) → 
  probability_at_least_one_three_tossed = 10 / 21 :=
by sorry

end probability_of_at_least_one_three_l192_192416


namespace germs_left_after_sprays_l192_192286

-- Define the percentages as real numbers
def S1 : ℝ := 0.50 -- 50%
def S2 : ℝ := 0.35 -- 35%
def S3 : ℝ := 0.20 -- 20%
def S4 : ℝ := 0.10 -- 10%

-- Define the overlaps as real numbers
def overlap12 : ℝ := 0.10 -- between S1 and S2
def overlap23 : ℝ := 0.07 -- between S2 and S3
def overlap34 : ℝ := 0.05 -- between S3 and S4
def overlap13 : ℝ := 0.03 -- between S1 and S3
def overlap14 : ℝ := 0.02 -- between S1 and S4

theorem germs_left_after_sprays :
  let total_killed := S1 + S2 + S3 + S4
  let total_overlap := overlap12 + overlap23 + overlap34 + overlap13 + overlap14
  let adjusted_overlap := overlap12 + overlap23 + overlap34
  let effective_killed := total_killed - adjusted_overlap
  let percentage_left := 1.0 - effective_killed
  percentage_left = 0.07 := by
  -- proof steps to be inserted here
  sorry

end germs_left_after_sprays_l192_192286


namespace div_decimal_l192_192163

theorem div_decimal (a b : ℝ)  (h₁ : a = 0.45) (h₂ : b = 0.005):
  a / b = 90 :=
by {
  sorry
}

end div_decimal_l192_192163


namespace solution_set_of_inequality_l192_192601

noncomputable def f : ℝ → ℝ :=
sorry  -- Definition of the function f isn't provided in the original problem.

theorem solution_set_of_inequality (h_deriv : ∀ x, deriv f x < f x)
  (h_even : ∀ x, f (x + 2) = f (2 - x))
  (h_value : f 4 = 1) :
  {x : ℝ | f x < real.exp x} = set.Ioi 0 :=
sorry

end solution_set_of_inequality_l192_192601


namespace find_x_l192_192213

theorem find_x 
  (b : ℤ) (h_b : b = 0) 
  (a z y x w : ℤ)
  (h1 : z + a = 1)
  (h2 : y + z + a = 0)
  (h3 : x + y + z = a)
  (h4 : w + x + y = z)
  :
  x = 2 :=
by {
    sorry
}    

end find_x_l192_192213


namespace eggs_in_basket_empty_l192_192080

theorem eggs_in_basket_empty (a : ℕ) : 
  let remaining_after_first := a - (a / 2 + 1 / 2)
  let remaining_after_second := remaining_after_first - (remaining_after_first / 2 + 1 / 2)
  let remaining_after_third := remaining_after_second - (remaining_after_second / 2 + 1 / 2)
  (remaining_after_first = a / 2 - 1 / 2) → 
  (remaining_after_second = remaining_after_first / 2 - 1 / 2) → 
  (remaining_after_third = remaining_after_second / 2 -1 / 2) → 
  (remaining_after_third = 0) → 
  (a = 7) := sorry

end eggs_in_basket_empty_l192_192080


namespace complex_division_simplification_l192_192181

theorem complex_division_simplification (i : ℂ) (h_i : i * i = -1) : (1 - 3 * i) / (2 - i) = 1 - i := by
  sorry

end complex_division_simplification_l192_192181


namespace find_divisor_l192_192426

variable (Dividend : ℕ) (Quotient : ℕ) (Divisor : ℕ)
variable (h1 : Dividend = 64)
variable (h2 : Quotient = 8)
variable (h3 : Dividend = Divisor * Quotient)

theorem find_divisor : Divisor = 8 := by
  sorry

end find_divisor_l192_192426


namespace cos_double_angle_of_parallel_vectors_l192_192021

theorem cos_double_angle_of_parallel_vectors
  (α : ℝ)
  (a : ℝ × ℝ := (1/3, Real.tan α))
  (b : ℝ × ℝ := (Real.cos α, 2))
  (h_parallel : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  Real.cos (2 * α) = 1 / 9 := 
sorry

end cos_double_angle_of_parallel_vectors_l192_192021


namespace pos_diff_is_multiple_of_9_l192_192545

theorem pos_diff_is_multiple_of_9 
  (q r : ℕ) 
  (h_qr : 10 ≤ q ∧ q < 100 ∧ 10 ≤ r ∧ r < 100 ∧ (q % 10) * 10 + (q / 10) = r)
  (h_max_diff : q - r = 63) : 
  ∃ k : ℕ, q - r = 9 * k :=
by
  sorry

end pos_diff_is_multiple_of_9_l192_192545


namespace max_students_l192_192818

theorem max_students : 
  ∃ x : ℕ, x < 100 ∧ x % 9 = 4 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y < 100 ∧ y % 9 = 4 ∧ y % 7 = 3) → y ≤ x := 
by
  sorry

end max_students_l192_192818


namespace largest_angle_in_triangle_l192_192511

theorem largest_angle_in_triangle (k : ℕ) (h : 3 * k + 4 * k + 5 * k = 180) : 5 * k = 75 :=
  by
  -- This is a placeholder for the proof, which is not required as per instructions
  sorry

end largest_angle_in_triangle_l192_192511


namespace difference_between_length_and_breadth_l192_192262

theorem difference_between_length_and_breadth (L W : ℝ) (h1 : W = 1/2 * L) (h2 : L * W = 800) : L - W = 20 :=
by
  sorry

end difference_between_length_and_breadth_l192_192262


namespace exists_natural_n_l192_192489

theorem exists_natural_n (a b : ℕ) (h1 : b ≥ 2) (h2 : Nat.gcd a b = 1) : ∃ n : ℕ, (n * a) % b = 1 :=
by
  sorry

end exists_natural_n_l192_192489


namespace x0_equals_pm1_l192_192050

-- Define the function f and its second derivative
def f (x : ℝ) : ℝ := x^3
def f'' (x : ℝ) : ℝ := 6 * x

-- Prove that if f''(x₀) = 6 then x₀ = ±1
theorem x0_equals_pm1 (x0 : ℝ) (h : f'' x0 = 6) : x0 = 1 ∨ x0 = -1 :=
by
  sorry

end x0_equals_pm1_l192_192050


namespace pie_eating_contest_l192_192263

theorem pie_eating_contest (B A S : ℕ) (h1 : A = B + 3) (h2 : S = 12) (h3 : B + A + S = 27) :
  S / B = 2 :=
by
  have h4 : B + (B + 3) + 12 = 27 from by rw [h1, h2]; exact h3
  have h5 : 2 * B + 15 = 27 from by linarith
  have h6 : 2 * B = 12 from by linarith
  have h7 : B = 6 from by linarith
  rw [h7, h2]; norm_num

end pie_eating_contest_l192_192263


namespace mean_of_remaining_two_l192_192264

theorem mean_of_remaining_two (a b c d e : ℝ) (h : (a + b + c = 3 * 2010)) : 
  (a + b + c + d + e) / 5 = 2010 → (d + e) / 2 = 2011.5 :=
by
  sorry 

end mean_of_remaining_two_l192_192264


namespace cameron_list_count_l192_192864

theorem cameron_list_count :
  let lower := 100
  let upper := 1000
  let step := 20
  let n_min := lower / step
  let n_max := upper / step
  lower % step = 0 ∧ upper % step = 0 →
  upper ≥ lower →
  n_max - n_min + 1 = 46 :=
by
  sorry

end cameron_list_count_l192_192864


namespace range_of_f_l192_192085

noncomputable def f (x : ℝ) : ℝ :=
  x + Real.sqrt (x - 2)

theorem range_of_f : Set.range f = {y : ℝ | 2 ≤ y} :=
by
  sorry

end range_of_f_l192_192085


namespace divisible_by_56_l192_192937

theorem divisible_by_56 (n : ℕ) (h1 : ∃ k, 3 * n + 1 = k * k) (h2 : ∃ m, 4 * n + 1 = m * m) : 56 ∣ n := 
sorry

end divisible_by_56_l192_192937


namespace playground_total_l192_192516

def boys : ℕ := 44
def girls : ℕ := 53

theorem playground_total : boys + girls = 97 := by
  sorry

end playground_total_l192_192516


namespace pie_difference_l192_192520

theorem pie_difference:
  ∀ (a b c d : ℚ), a = 6 / 7 → b = 3 / 4 → (a - b) = c → c = 3 / 28 :=
by
  sorry

end pie_difference_l192_192520


namespace period_of_f_is_4_and_f_2pow_n_zero_l192_192603

noncomputable def f : ℝ → ℝ := sorry

variables (hf_diff : differentiable ℝ f)
          (hf_nonzero : ∃ x, f x ≠ 0)
          (hf_odd_2 : ∀ x, f (x + 2) = -f (-x - 2))
          (hf_even_2x1 : ∀ x, f (2 * x + 1) = f (-(2 * x + 1)))

theorem period_of_f_is_4_and_f_2pow_n_zero (n : ℕ) (hn : 0 < n) :
  (∀ x, f (x + 4) = f x) ∧ f (2^n) = 0 :=
sorry

end period_of_f_is_4_and_f_2pow_n_zero_l192_192603


namespace correct_mark_l192_192413

theorem correct_mark (x : ℕ) (S_Correct S_Wrong : ℕ) (n : ℕ) :
  n = 26 →
  S_Wrong = S_Correct + (83 - x) →
  (S_Wrong : ℚ) / n = (S_Correct : ℚ) / n + 1 / 2 →
  x = 70 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l192_192413


namespace investment_calculation_l192_192819

noncomputable def calculate_investment_amount (A : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_calculation :
  let A := 80000
  let r := 0.07
  let n := 12
  let t := 7
  let P := calculate_investment_amount A r n t
  abs (P - 46962) < 1 :=
by
  sorry

end investment_calculation_l192_192819


namespace find_f_prime_at_2_l192_192529

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

theorem find_f_prime_at_2 :
  ∃ (a b : ℝ), (f a b 1 = -2 ∧ ∀ x, (1 / x) * (a - 2 / x) + (x - 1) * (a / x ^ 2 + 2 * (1 / x ^ 3)) = 0) → 
  (deriv (f a b) 2 = -1 / 2) :=
by
  -- assume a and b satisfying the given conditions
  use [-2, -2]
  sorry

end find_f_prime_at_2_l192_192529


namespace bond_yield_correct_l192_192491

-- Definitions of the conditions
def number_of_bonds : ℕ := 1000
def holding_period : ℕ := 2
def bond_income : ℚ := 980 - 980 + 1000 * 0.07 * 2
def initial_investment : ℚ := 980000

-- Yield for 2 years
def yield_2_years : ℚ := (number_of_bonds * bond_income) / initial_investment * 100

-- Average annual yield
def avg_annual_yield : ℚ := yield_2_years / holding_period

-- The main theorem to prove
theorem bond_yield_correct :
  yield_2_years = 15.31 ∧ avg_annual_yield = 7.65 :=
by
  sorry

end bond_yield_correct_l192_192491


namespace sum_of_coefficients_binomial_expansion_l192_192151

theorem sum_of_coefficients_binomial_expansion : 
  (Polynomial.sum (\(n : Nat) x => (Polynomial.coeff ((Polynomial.C (1: ℚ) - Polynomial.X)^7) n) x)) = 0 :=
by
  sorry

end sum_of_coefficients_binomial_expansion_l192_192151


namespace solve_for_x_l192_192988

theorem solve_for_x :
  ∃ x : ℝ, (24 / 36) = Real.sqrt (x / 36) ∧ x = 16 :=
by
  use 16
  sorry

end solve_for_x_l192_192988


namespace weeks_to_meet_goal_l192_192719

def hourly_rate : ℕ := 6
def hours_monday : ℕ := 2
def hours_tuesday : ℕ := 3
def hours_wednesday : ℕ := 4
def hours_thursday : ℕ := 2
def hours_friday : ℕ := 3
def helmet_cost : ℕ := 340
def gloves_cost : ℕ := 45
def initial_savings : ℕ := 40
def misc_expenses : ℕ := 20

theorem weeks_to_meet_goal : 
  let total_needed := helmet_cost + gloves_cost + misc_expenses
  let total_deficit := total_needed - initial_savings
  let total_weekly_hours := hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday
  let weekly_earnings := total_weekly_hours * hourly_rate
  let weeks_required := Nat.ceil (total_deficit / weekly_earnings)
  weeks_required = 5 := sorry

end weeks_to_meet_goal_l192_192719


namespace sum_primitive_roots_mod_prime_is_zero_l192_192353

theorem sum_primitive_roots_mod_prime_is_zero
  (p : ℕ)
  (hp : p.prime)
  (hp_odd : p % 2 = 1)
  (S : ℕ)
  (hS : S = ∑ x in primitiveRoots p isPrimitiveRoot x)
  (h_not_squarefree : ∃ k m : ℕ, k > 1 ∧ p - 1 = k^2 * m) :
  S % p = 0 := 
sorry

end sum_primitive_roots_mod_prime_is_zero_l192_192353


namespace ice_cream_remaining_l192_192701

def total_initial_scoops : ℕ := 3 * 10
def ethan_scoops : ℕ := 1 + 1
def lucas_danny_connor_scoops : ℕ := 2 * 3
def olivia_scoops : ℕ := 1 + 1
def shannon_scoops : ℕ := 2 * olivia_scoops
def total_consumed_scoops : ℕ := ethan_scoops + lucas_danny_connor_scoops + olivia_scoops + shannon_scoops
def remaining_scoops : ℕ := total_initial_scoops - total_consumed_scoops

theorem ice_cream_remaining : remaining_scoops = 16 := by
  sorry

end ice_cream_remaining_l192_192701


namespace triangle_angle_sum_l192_192342

theorem triangle_angle_sum (P Q R : ℝ) (h1 : P + Q = 60) (h2 : P + Q + R = 180) : R = 120 := by
  sorry

end triangle_angle_sum_l192_192342


namespace coloring_triangles_l192_192564

theorem coloring_triangles (n : ℕ) (k : ℕ) (h_n : n = 18) (h_k : k = 6) :
  (Nat.choose n k) = 18564 :=
by
  rw [h_n, h_k]
  sorry

end coloring_triangles_l192_192564


namespace range_of_m_l192_192610

theorem range_of_m
  (m : ℝ)
  (h1 : (m - 1) * (3 - m) ≠ 0) 
  (h2 : 3 - m > 0) 
  (h3 : m - 1 > 0) 
  (h4 : 3 - m ≠ m - 1) :
  1 < m ∧ m < 3 ∧ m ≠ 2 :=
sorry

end range_of_m_l192_192610


namespace decreasing_cubic_function_l192_192075

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^3 - x

-- Define the condition that f is decreasing on (-∞, ∞)
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

-- The main theorem that needs to be proven
theorem decreasing_cubic_function (m : ℝ) : is_decreasing (f m) → m < 0 := 
by
  sorry

end decreasing_cubic_function_l192_192075


namespace slower_ball_speed_l192_192519

open Real

variables (v u C : ℝ)

theorem slower_ball_speed :
  (20 * (v - u) = C) → (4 * (v + u) = C) → ((v + u) * 3 = 75) → u = 10 :=
by
  intros h1 h2 h3
  sorry

end slower_ball_speed_l192_192519


namespace find_principal_l192_192827

theorem find_principal
  (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ)
  (hA : A = 896)
  (hr : r = 0.05)
  (ht : t = 12 / 5) :
  P = 800 ↔ A = P * (1 + r * t) :=
by {
  sorry
}

end find_principal_l192_192827


namespace route_difference_is_18_l192_192481

theorem route_difference_is_18 :
  let t_uphill := 6 in
  let t_path1 := 2 * t_uphill in
  let t_final1 := (t_uphill + t_path1)/3 in
  let t_route1 := t_uphill + t_path1 + t_final1 in
  let t_flat := 14 in
  let t_final2 := 2 * t_flat in
  let t_route2 := t_flat + t_final2 in
  t_route2 - t_route1 = 18 :=
by
  let t_uphill := 6
  let t_path1 := 2 * t_uphill
  let t_final1 := (t_uphill + t_path1)/3
  let t_route1 := t_uphill + t_path1 + t_final1
  let t_flat := 14
  let t_final2 := 2 * t_flat
  let t_route2 := t_flat + t_final2
  have h : t_route2 - t_route1 = 18
  sorry

end route_difference_is_18_l192_192481


namespace dante_final_coconuts_l192_192936

theorem dante_final_coconuts
  (Paolo_coconuts : ℕ) (Dante_init_coconuts : ℝ)
  (Bianca_coconuts : ℕ) (Dante_final_coconuts : ℕ):
  Paolo_coconuts = 14 →
  Dante_init_coconuts = 1.5 * Real.sqrt Paolo_coconuts →
  Bianca_coconuts = 2 * (Paolo_coconuts + Int.floor Dante_init_coconuts) →
  Dante_final_coconuts = (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) - 
    (25 * (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) / 100) →
  Dante_final_coconuts = 3 :=
by
  sorry

end dante_final_coconuts_l192_192936


namespace percent_defective_units_shipped_l192_192475

theorem percent_defective_units_shipped :
  let total_units_defective := 6 / 100
  let defective_units_shipped := 4 / 100
  let percent_defective_units_shipped := (total_units_defective * defective_units_shipped) * 100
  percent_defective_units_shipped = 0.24 := by
  sorry

end percent_defective_units_shipped_l192_192475


namespace work_completed_together_in_4_days_l192_192992

/-- A can do the work in 6 days. -/
def A_work_rate : ℚ := 1 / 6

/-- B can do the work in 12 days. -/
def B_work_rate : ℚ := 1 / 12

/-- Combined work rate of A and B working together. -/
def combined_work_rate : ℚ := A_work_rate + B_work_rate

/-- Number of days for A and B to complete the work together. -/
def days_to_complete : ℚ := 1 / combined_work_rate

theorem work_completed_together_in_4_days : days_to_complete = 4 := by
  sorry

end work_completed_together_in_4_days_l192_192992


namespace probability_same_color_l192_192457

-- Define the number of plates of each color
def num_red_plates : ℕ := 7
def num_blue_plates : ℕ := 5
def num_green_plates : ℕ := 3

-- Total number of plates
def total_plates := num_red_plates + num_blue_plates + num_green_plates

-- Function to calculate the number of ways to choose 2 out of n elements
def choose2 (n : ℕ) := n.choose 2

-- Number of ways to choose 2 red plates
def red_pairs := choose2 num_red_plates

-- Number of ways to choose 2 blue plates
def blue_pairs := choose2 num_blue_plates

-- Number of ways to choose 2 green plates
def green_pairs := choose2 num_green_plates

-- Total number of ways to choose any 2 plates
def total_pairs := choose2 total_plates

-- Total number of same-color pairs
def same_color_pairs := red_pairs + blue_pairs + green_pairs

-- Probability that the two plates selected are of the same color
def same_color_probability := (same_color_pairs : ℚ) / total_pairs

-- The theorem to prove
theorem probability_same_color : same_color_probability = 34/105 := sorry

end probability_same_color_l192_192457


namespace carpet_cost_calculation_l192_192414

theorem carpet_cost_calculation
  (length_feet : ℕ)
  (width_feet : ℕ)
  (feet_to_yards : ℕ)
  (cost_per_square_yard : ℕ)
  (h_length : length_feet = 15)
  (h_width : width_feet = 12)
  (h_convert : feet_to_yards = 3)
  (h_cost : cost_per_square_yard = 10) :
  (length_feet / feet_to_yards) *
  (width_feet / feet_to_yards) *
  cost_per_square_yard = 200 := by
  sorry

end carpet_cost_calculation_l192_192414


namespace find_percentage_l192_192129

theorem find_percentage (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 :=
by
  intros h
  sorry

end find_percentage_l192_192129


namespace closest_integer_to_cube_root_of_250_l192_192120

theorem closest_integer_to_cube_root_of_250 : 
  let cbrt250 := real.cbrt 250
  ∃ (n : ℤ), n = 6 ∧ (∀ (m : ℤ), abs (m - cbrt250) ≥ abs (n - cbrt250)) := 
by
  sorry

end closest_integer_to_cube_root_of_250_l192_192120


namespace average_rate_of_change_nonzero_l192_192211

-- Define the conditions related to the average rate of change.
variables {x0 : ℝ} {Δx : ℝ}

-- Define the statement to prove that in the definition of the average rate of change, Δx ≠ 0.
theorem average_rate_of_change_nonzero (h : Δx ≠ 0) : True :=
sorry  -- The proof is omitted as per instruction.

end average_rate_of_change_nonzero_l192_192211


namespace sale_price_of_sarees_after_discounts_l192_192512

theorem sale_price_of_sarees_after_discounts :
  let original_price := 400.0
  let discount_1 := 0.15
  let discount_2 := 0.08
  let discount_3 := 0.07
  let discount_4 := 0.10
  let price_after_first_discount := original_price * (1 - discount_1)
  let price_after_second_discount := price_after_first_discount * (1 - discount_2)
  let price_after_third_discount := price_after_second_discount * (1 - discount_3)
  let final_price := price_after_third_discount * (1 - discount_4)
  final_price = 261.81 := by
    -- Sorry is used to skip the proof
    sorry

end sale_price_of_sarees_after_discounts_l192_192512


namespace distinct_seatings_l192_192786

theorem distinct_seatings : 
  ∃ n : ℕ, (n = 288000) ∧ 
  (∀ (men wives : Fin 6 → ℕ),
  ∃ (f : (Fin 12) → ℕ), 
  (∀ i, f (i + 1) % 12 ≠ f i) ∧
  (∀ i, f i % 2 = 0) ∧
  (∀ j, f (2 * j) = men j ∧ f (2 * j + 1) = wives j)) :=
by
  sorry

end distinct_seatings_l192_192786


namespace solve_for_y_l192_192972

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l192_192972


namespace train_stoppage_time_l192_192583

-- Definitions of the conditions
def speed_excluding_stoppages : ℝ := 48 -- in kmph
def speed_including_stoppages : ℝ := 32 -- in kmph
def time_per_hour : ℝ := 60 -- 60 minutes in an hour

-- The problem statement
theorem train_stoppage_time :
  (speed_excluding_stoppages - speed_including_stoppages) * time_per_hour / speed_excluding_stoppages = 20 :=
by
  -- Initial statement
  sorry

end train_stoppage_time_l192_192583


namespace probability_is_0_4_l192_192855

def coin_values : List ℕ := [10, 10, 5, 5, 2]

def valid_combination (comb : List ℕ) : Prop :=
  comb.sum ≥ 19

def favorable_outcomes : Finset (Finset ℕ) :=
  {s ∈ coin_values.to_finset.powerset.filter (λ s, s.card = 3) | valid_combination s.val.to_list}

def total_outcomes : Finset (Finset ℕ) :=
  coin_values.to_finset.powerset.filter (λ s, s.card = 3)

def probability : ℚ :=
  favorable_outcomes.card / total_outcomes.card

theorem probability_is_0_4 : probability = 2 / 5 :=
by
  -- Proof will go here
  sorry

end probability_is_0_4_l192_192855


namespace least_two_multiples_of_15_gt_450_l192_192523

-- Define a constant for the base multiple
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

-- Define a constant for being greater than 450
def is_greater_than_450 (n : ℕ) : Prop :=
  n > 450

-- Two least positive multiples of 15 greater than 450
theorem least_two_multiples_of_15_gt_450 :
  (is_multiple_of_15 465 ∧ is_greater_than_450 465 ∧
   is_multiple_of_15 480 ∧ is_greater_than_450 480) :=
by
  sorry

end least_two_multiples_of_15_gt_450_l192_192523


namespace max_value_expression_l192_192337

theorem max_value_expression (a b c d : ℤ) (hb_pos : b > 0)
  (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a - 2 * b + 3 * c - 4 * d = -7 := 
sorry

end max_value_expression_l192_192337


namespace new_person_weight_l192_192994

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) : 
    avg_increase = 2.5 ∧ num_persons = 8 ∧ old_weight = 65 → 
    (old_weight + num_persons * avg_increase = 85) :=
by
  intro h
  sorry

end new_person_weight_l192_192994


namespace stack_of_logs_total_l192_192847

-- Define the given conditions as variables and constants in Lean
def bottom_row : Nat := 15
def top_row : Nat := 4
def rows : Nat := bottom_row - top_row + 1
def sum_arithmetic_series (a l n : Nat) : Nat := n * (a + l) / 2

-- Define the main theorem to prove
theorem stack_of_logs_total : sum_arithmetic_series top_row bottom_row rows = 114 :=
by
  -- Here you will normally provide the proof
  sorry

end stack_of_logs_total_l192_192847


namespace lcm_of_numbers_with_ratio_and_hcf_l192_192669

theorem lcm_of_numbers_with_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : Nat.gcd a b = 3) : Nat.lcm a b = 36 := 
  sorry

end lcm_of_numbers_with_ratio_and_hcf_l192_192669


namespace value_of_p_l192_192443

theorem value_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (x1 x2 : ℕ), x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) : p = -2278 :=
by
  sorry

end value_of_p_l192_192443


namespace trig_identity_A_trig_identity_D_l192_192267

theorem trig_identity_A : 
  (Real.tan (25 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) + Real.tan (25 * Real.pi / 180) * Real.tan (20 * Real.pi / 180) = 1) :=
by sorry

theorem trig_identity_D : 
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180) = 4) :=
by sorry

end trig_identity_A_trig_identity_D_l192_192267


namespace green_apples_ordered_l192_192671

-- Definitions based on the conditions
variable (red_apples : Nat := 25)
variable (students : Nat := 10)
variable (extra_apples : Nat := 32)
variable (G : Nat)

-- The mathematical problem to prove
theorem green_apples_ordered :
  red_apples + G - students = extra_apples → G = 17 := by
  sorry

end green_apples_ordered_l192_192671


namespace elena_probability_at_least_one_correct_l192_192169

-- Conditions
def total_questions := 30
def choices_per_question := 4
def guessed_questions := 6
def incorrect_probability_single := 3 / 4

-- Expression for the probability of missing all guessed questions
def probability_all_incorrect := (incorrect_probability_single) ^ guessed_questions

-- Calculation from the solution
def probability_at_least_one_correct := 1 - probability_all_incorrect

-- Problem statement to prove
theorem elena_probability_at_least_one_correct : probability_at_least_one_correct = 3367 / 4096 :=
by sorry

end elena_probability_at_least_one_correct_l192_192169


namespace part1_solution_set_part2_range_a_l192_192077

noncomputable def f (x a : ℝ) := 5 - abs (x + a) - abs (x - 2)

-- Part 1
theorem part1_solution_set (x : ℝ) (a : ℝ) (h : a = 1) :
  (f x a ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 3) := sorry

-- Part 2
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) := sorry

end part1_solution_set_part2_range_a_l192_192077


namespace transistors_in_2010_l192_192933

-- Define initial conditions
def initial_transistors : ℕ := 500000
def years_passed : ℕ := 15
def tripling_period : ℕ := 3
def tripling_factor : ℕ := 3

-- Define the function to compute the number of transistors after a number of years
noncomputable def final_transistors (initial : ℕ) (years : ℕ) (period : ℕ) (factor : ℕ) : ℕ :=
  initial * factor ^ (years / period)

-- State the proposition we aim to prove
theorem transistors_in_2010 : final_transistors initial_transistors years_passed tripling_period tripling_factor = 121500000 := 
by 
  sorry

end transistors_in_2010_l192_192933


namespace min_value_expression_l192_192746

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + 1 / b) * (b + 4 / a) ≥ 9 :=
by
  sorry

end min_value_expression_l192_192746


namespace two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l192_192005

theorem two_pow_add_three_perfect_square (n : ℕ) :
  ∃ k, 2^n + 3 = k^2 ↔ n = 0 :=
by {
  sorry
}

theorem two_pow_add_one_perfect_square (n : ℕ) :
  ∃ k, 2^n + 1 = k^2 ↔ n = 3 :=
by {
  sorry
}

end two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l192_192005


namespace bill_age_l192_192148

theorem bill_age (C : ℕ) (h1 : ∀ B : ℕ, B = 2 * C - 1) (h2 : C + (2 * C - 1) = 26) : 
  ∃ B : ℕ, B = 17 := 
by
  sorry

end bill_age_l192_192148


namespace jenna_discount_l192_192770

def normal_price : ℝ := 50
def tickets_from_website : ℝ := 2 * normal_price
def scalper_initial_price_per_ticket : ℝ := 2.4 * normal_price
def scalper_total_initial : ℝ := 2 * scalper_initial_price_per_ticket
def friend_discounted_ticket : ℝ := 0.6 * normal_price
def total_price_five_tickets : ℝ := tickets_from_website + scalper_total_initial + friend_discounted_ticket
def amount_paid_by_friends : ℝ := 360

theorem jenna_discount : 
    total_price_five_tickets - amount_paid_by_friends = 10 :=
by
  -- The proof would go here, but we leave it as sorry for now.
  sorry

end jenna_discount_l192_192770


namespace oranges_after_eating_l192_192488

def initial_oranges : ℝ := 77.0
def eaten_oranges : ℝ := 2.0
def final_oranges : ℝ := 75.0

theorem oranges_after_eating :
  initial_oranges - eaten_oranges = final_oranges := by
  sorry

end oranges_after_eating_l192_192488


namespace sally_has_18_nickels_and_total_value_98_cents_l192_192615

-- Define the initial conditions
def pennies_initial := 8
def nickels_initial := 7
def nickels_from_dad := 9
def nickels_from_mom := 2

-- Define calculations based on the initial conditions
def total_nickels := nickels_initial + nickels_from_dad + nickels_from_mom
def value_pennies := pennies_initial
def value_nickels := total_nickels * 5
def total_value := value_pennies + value_nickels

-- State the theorem to prove the correct answers
theorem sally_has_18_nickels_and_total_value_98_cents :
  total_nickels = 18 ∧ total_value = 98 := 
by {
  -- Proof goes here
  sorry
}

end sally_has_18_nickels_and_total_value_98_cents_l192_192615


namespace gold_copper_alloy_ratio_l192_192192

theorem gold_copper_alloy_ratio {G C A : ℝ} (hC : C = 9) (hA : A = 18) (hG : 9 < G ∧ G < 18) :
  ∃ x : ℝ, 18 = x * G + (1 - x) * 9 :=
by
  sorry

end gold_copper_alloy_ratio_l192_192192


namespace exists_hamiltonian_cycle_l192_192562

variables (G : SimpleGraph (Fin (2 * n))) (n : ℕ)
  
noncomputable def degree_condition (G : SimpleGraph (Fin (2 * n))) (n : ℕ) : Prop :=
  ∀ k : ℕ, k ∈ Finset.range (n - 1) → G.degree_lt_k k < k
  
theorem exists_hamiltonian_cycle
  (h1 : ∀ v : Fin (2 * n), G.degree v = 2)
  (h2 : degree_condition G n)
  (h3 : 2 ≤ n):
  ¬ G.nonHamiltonian :=
sorry

end exists_hamiltonian_cycle_l192_192562


namespace sum_common_ratios_l192_192357

theorem sum_common_ratios (k p r : ℝ) (h1 : p ≠ r) (h2 : k ≠ 0)
  (h3 : a2 = k * p) (h4 : a3 = k * p^2) (h5 : b2 = k * r) (h6 : b3 = k * r^2)
  (h : a3 - b3 = 3 * (a2 - b2)) : p + r = 3 :=
by 
  have h3 : k * p^2 - k * r^2 = 3 * (k * p - k * r), from h2,
  sorry

end sum_common_ratios_l192_192357


namespace similar_triangle_leg_length_l192_192844

theorem similar_triangle_leg_length (a b c : ℝ) (h0 : a = 12) (h1 : b = 9) (h2 : c = 7.5) :
  ∃ y : ℝ, ((12 / 7.5) = (9 / y) → y = 5.625) :=
by
  use 5.625
  intro h
  linarith

end similar_triangle_leg_length_l192_192844


namespace smallest_bob_number_l192_192145

theorem smallest_bob_number (b : ℕ) (h : ∀ p : ℕ, Prime p → p ∣ 30 → p ∣ b) : 30 ≤ b :=
by {
  sorry
}

end smallest_bob_number_l192_192145


namespace children_selection_l192_192208

-- Conditions and definitions
def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Proof problem statement
theorem children_selection : ∃ r : ℕ, comb 10 r = 210 ∧ r = 4 :=
by
  sorry

end children_selection_l192_192208


namespace insects_total_l192_192515

def total_insects (n_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
                  (n_stones : ℕ) (ants_per_stone : ℕ) 
                  (total_bees : ℕ) (n_flowers : ℕ) : ℕ :=
  let num_ladybugs := n_leaves * ladybugs_per_leaf
  let num_ants := n_stones * ants_per_stone
  let num_bees := total_bees -- already given as total_bees
  num_ladybugs + num_ants + num_bees

theorem insects_total : total_insects 345 267 178 423 498 6 = 167967 :=
  by unfold total_insects; sorry

end insects_total_l192_192515


namespace machine_C_time_l192_192363

theorem machine_C_time (T_c : ℝ) :
  (1 / 4 + 1 / 2 + 1 / T_c = 11 / 12) → T_c = 6 :=
by
  sorry

end machine_C_time_l192_192363


namespace find_other_number_l192_192259

theorem find_other_number (a b : ℕ) (h1 : a + b = 62) (h2 : b - a = 12) (h3 : a = 25) : b = 37 :=
sorry

end find_other_number_l192_192259


namespace problem_b_problem_c_problem_d_l192_192439

variable (a b : ℝ)

theorem problem_b (h : a * b > 0) :
  2 * (a^2 + b^2) ≥ (a + b)^2 :=
sorry

theorem problem_c (h : a * b > 0) :
  (b / a) + (a / b) ≥ 2 :=
sorry

theorem problem_d (h : a * b > 0) :
  (a + 1 / a) * (b + 1 / b) ≥ 4 :=
sorry

end problem_b_problem_c_problem_d_l192_192439


namespace trig_identity_30deg_l192_192728

theorem trig_identity_30deg :
  let t30 := Real.tan (Real.pi / 6)
  let s30 := Real.sin (Real.pi / 6)
  let c30 := Real.cos (Real.pi / 6)
  t30 = (Real.sqrt 3) / 3 ∧ s30 = 1 / 2 ∧ c30 = (Real.sqrt 3) / 2 →
  t30 + 4 * s30 + 2 * c30 = (2 * (Real.sqrt 3) + 3) / 3 := 
by
  intros
  sorry

end trig_identity_30deg_l192_192728


namespace systematic_sampling_condition_l192_192244

theorem systematic_sampling_condition (population sample_size total_removed segments individuals_per_segment : ℕ) 
  (h_population : population = 1650)
  (h_sample_size : sample_size = 35)
  (h_total_removed : total_removed = 5)
  (h_segments : segments = sample_size)
  (h_individuals_per_segment : individuals_per_segment = (population - total_removed) / sample_size)
  (h_modulo : population % sample_size = total_removed)
  :
  total_removed = 5 ∧ segments = 35 ∧ individuals_per_segment = 47 := 
by
  sorry

end systematic_sampling_condition_l192_192244


namespace rectangular_garden_length_l192_192140

theorem rectangular_garden_length (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 900) : l = 300 :=
by
  sorry

end rectangular_garden_length_l192_192140


namespace proposition_A_proposition_B_proposition_C_proposition_D_l192_192397

theorem proposition_A (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a * b < 0 := 
sorry

theorem proposition_B (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬ (a^2 < a * b ∧ a * b < b^2) := 
sorry

theorem proposition_C (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) : ¬ (a / (c - a) < b / (c - b)) := 
sorry

theorem proposition_D (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : a / b > (a + c) / (b + c) := 
sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l192_192397


namespace derivative_at_2_l192_192530

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_2 :
  ∃ a b : ℝ,
    let f := λ x => a * Real.log x + b / x in
    (f 1 = -2) ∧ (∀ x : ℝ, 
      (f : ℝ -> ℝ)' x = (a / x + (2 - a) / x^2)) ∧ (a = -2 / b) →
      (deriv (λ x => a * Real.log x + (b : ℝ) / x) 2 = -1 / 2) := by
  sorry

end derivative_at_2_l192_192530


namespace value_of_f_minus_a_l192_192900

noncomputable def f (x : ℝ) : ℝ := x^3 + x + 1

theorem value_of_f_minus_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by sorry

end value_of_f_minus_a_l192_192900


namespace factorial_equation_solution_l192_192588

theorem factorial_equation_solution (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → (a, b, c) = (3, 3, 4) :=
by
  sorry

end factorial_equation_solution_l192_192588


namespace range_of_a_l192_192887

noncomputable def f (a x : ℝ) := (Real.exp x - a * x^2) 

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 ≤ x → f a x ≥ x + 1) ↔ a ∈ Set.Iic (1/2) :=
by
  sorry

end range_of_a_l192_192887


namespace proposition_A_proposition_B_proposition_C_proposition_D_l192_192394

-- Definitions and conditions for proposition A
def propA_conditions (a b : ℝ) : Prop :=
  a > b ∧ (1 / a) > (1 / b)

def propA (a b : ℝ) : Prop :=
  a * b < 0

-- Definitions and conditions for proposition B
def propB_conditions (a b : ℝ) : Prop :=
  a < b ∧ b < 0

def propB (a b : ℝ) : Prop :=
  a^2 < a * b ∧ a * b < b^2

-- Definitions and conditions for proposition C
def propC_conditions (c a b : ℝ) : Prop :=
  c > a ∧ a > b ∧ b > 0

def propC (c a b : ℝ) : Prop :=
  (a / (c - a)) < (b / (c - b))

-- Definitions and conditions for proposition D
def propD_conditions (a b c : ℝ) : Prop :=
  a > b ∧ b > c ∧ c > 0

def propD (a b c : ℝ) : Prop :=
  (a / b) > ((a + c) / (b + c))

-- The propositions
theorem proposition_A (a b : ℝ) (h : propA_conditions a b) : propA a b := 
sorry

theorem proposition_B (a b : ℝ) (h : propB_conditions a b) : ¬ propB a b :=
sorry

theorem proposition_C (c a b : ℝ) (h : propC_conditions c a b) : ¬ propC c a b :=
sorry

theorem proposition_D (a b c : ℝ) (h : propD_conditions a b c) : propD a b c :=
sorry

end proposition_A_proposition_B_proposition_C_proposition_D_l192_192394


namespace range_of_x_range_of_a_l192_192226

-- Definitions of propositions p and q
def p (a x : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- Question 1
theorem range_of_x (a x : ℝ) : a = 1 → p a x ∧ q x → 2 < x ∧ x < 3 := by
  sorry

-- Question 2
theorem range_of_a (a : ℝ) : (∀ x, ¬p a x → ¬q x) → (∀ x, q x → p a x) → 1 < a ∧ a ≤ 2 := by
  sorry

end range_of_x_range_of_a_l192_192226


namespace yellow_red_chair_ratio_l192_192650

variable (Y B : ℕ)
variable (red_chairs : ℕ := 5)
variable (total_chairs : ℕ := 43)

-- Condition: There are 2 fewer blue chairs than yellow chairs
def blue_chairs_condition : Prop := B = Y - 2

-- Condition: Total number of chairs
def total_chairs_condition : Prop := red_chairs + Y + B = total_chairs

-- Prove the ratio of yellow chairs to red chairs is 4:1
theorem yellow_red_chair_ratio (h1 : blue_chairs_condition Y B) (h2 : total_chairs_condition Y B) :
  (Y / red_chairs) = 4 := 
sorry

end yellow_red_chair_ratio_l192_192650


namespace total_bottles_needed_l192_192631

-- Definitions from conditions
def large_bottle_capacity : ℕ := 450
def small_bottle_capacity : ℕ := 45
def extra_large_bottle_capacity : ℕ := 900

-- Theorem statement
theorem total_bottles_needed :
  ∃ (num_large_bottles num_small_bottles : ℕ), 
    num_large_bottles * large_bottle_capacity + num_small_bottles * small_bottle_capacity = extra_large_bottle_capacity ∧ 
    num_large_bottles + num_small_bottles = 2 :=
by
  sorry

end total_bottles_needed_l192_192631


namespace jane_played_8_rounds_l192_192035

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end jane_played_8_rounds_l192_192035


namespace multiply_negatives_l192_192726

theorem multiply_negatives : (- (1 / 2)) * (- 2) = 1 :=
by
  sorry

end multiply_negatives_l192_192726


namespace num_ways_select_at_least_one_defective_l192_192622

theorem num_ways_select_at_least_one_defective :
  let total_products := 100
  let defective_products := 6
  let selected_products := 3
  finset.card (finset.range (total_products + 1).choose selected_products) - 
  finset.card (finset.range (total_products - defective_products + 1).choose selected_products) 
  = finset.card (finset.range (total_products + 1)).choose selected_products - 
    finset.card (finset.range (total_products - defective_products + 1)).choose selected_products :=
sorry

end num_ways_select_at_least_one_defective_l192_192622


namespace value_of_f_of_1_plus_g_of_2_l192_192355

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := x + 1

theorem value_of_f_of_1_plus_g_of_2 : f (1 + g 2) = 5 :=
by
  sorry

end value_of_f_of_1_plus_g_of_2_l192_192355


namespace min_value_x_plus_9_div_x_l192_192028

theorem min_value_x_plus_9_div_x (x : ℝ) (hx : x > 0) : x + 9 / x ≥ 6 := by
  -- sorry indicates that the proof is omitted.
  sorry

end min_value_x_plus_9_div_x_l192_192028


namespace ratio_of_areas_l192_192821

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ :=
1 / 2 * a * b

theorem ratio_of_areas (a b c x y z : ℝ)
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : x = 9) (h5 : y = 12) (h6 : z = 15)
  (h7 : a^2 + b^2 = c^2) (h8 : x^2 + y^2 = z^2) :
  (area_of_right_triangle a b) / (area_of_right_triangle x y) = 4 / 9 :=
sorry

end ratio_of_areas_l192_192821


namespace manuscript_typing_cost_l192_192401

theorem manuscript_typing_cost 
  (pages_total : ℕ) (pages_first_time : ℕ) (pages_revised_once : ℕ)
  (pages_revised_twice : ℕ) (rate_first_time : ℕ) (rate_revised : ℕ) 
  (cost_total : ℕ) :
  pages_total = 100 →
  pages_first_time = pages_total →
  pages_revised_once = 35 →
  pages_revised_twice = 15 →
  rate_first_time = 6 →
  rate_revised = 4 →
  cost_total = (pages_first_time * rate_first_time) +
              (pages_revised_once * rate_revised) +
              (pages_revised_twice * rate_revised * 2) →
  cost_total = 860 :=
by
  intros htot hfirst hrev1 hrev2 hr1 hr2 hcost
  sorry

end manuscript_typing_cost_l192_192401


namespace parabola_equation_l192_192333

theorem parabola_equation (x y : ℝ) :
  (∃p : ℝ, x = 4 ∧ y = -2 ∧ (x^2 = -2 * p * y ∨ y^2 = 2 * p * x) → (x^2 = -8 * y ∨ y^2 = x)) :=
by
  sorry

end parabola_equation_l192_192333


namespace sqrt_72_plus_sqrt_32_l192_192964

noncomputable def sqrt_simplify (n : ℕ) : ℝ :=
  real.sqrt (n:ℝ)

theorem sqrt_72_plus_sqrt_32 :
  sqrt_simplify 72 + sqrt_simplify 32 = 10 * real.sqrt 2 :=
by {
  have h1 : sqrt_simplify 72 = 6 * real.sqrt 2, sorry,
  have h2 : sqrt_simplify 32 = 4 * real.sqrt 2, sorry,
  rw [h1, h2],
  ring,
}

end sqrt_72_plus_sqrt_32_l192_192964


namespace coordinates_of_B_l192_192012

-- Definitions of the points and vectors are given as conditions.
def A : ℝ × ℝ := (-1, -1)
def a : ℝ × ℝ := (2, 3)

-- Statement of the problem translated to Lean
theorem coordinates_of_B (B : ℝ × ℝ) (h : B = (5, 8)) :
  (B.1 + 1, B.2 + 1) = (3 * a.1, 3 * a.2) :=
sorry

end coordinates_of_B_l192_192012


namespace WallLengthBy40Men_l192_192618

-- Definitions based on the problem conditions
def men1 : ℕ := 20
def length1 : ℕ := 112
def days1 : ℕ := 6

def men2 : ℕ := 40
variable (y : ℕ)  -- given 'y' days

-- Establish the relationship based on the given conditions
theorem WallLengthBy40Men :
  ∃ x : ℕ, x = (men2 / men1) * length1 * (y / days1) :=
by
  sorry

end WallLengthBy40Men_l192_192618


namespace avg_one_fourth_class_l192_192791

variable (N : ℕ) (A : ℕ)
variable (h1 : ((N : ℝ) * 80) = (N / 4) * A + (3 * N / 4) * 76)

theorem avg_one_fourth_class : A = 92 :=
by
  sorry

end avg_one_fourth_class_l192_192791


namespace min_value_quadratic_form_l192_192007

theorem min_value_quadratic_form : ∀ x y : ℝ, ∃ m ∈ set.Iio 1, (m = x^2 - x * y + y^2) :=
by
  intros x y
  use 0
  sorry

end min_value_quadratic_form_l192_192007


namespace fraction_value_l192_192319

variable (a b : ℚ)  -- Variables a and b are rational numbers

theorem fraction_value (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_value_l192_192319


namespace floor_equality_iff_l192_192172

variable (x : ℝ)

theorem floor_equality_iff :
  (⌊3 * x + 4⌋ = ⌊5 * x - 1⌋) ↔
  (11 / 5 ≤ x ∧ x < 7 / 3) ∨
  (12 / 5 ≤ x ∧ x < 13 / 5) ∨
  (17 / 5 ≤ x ∧ x < 18 / 5) := by
  sorry

end floor_equality_iff_l192_192172


namespace sticks_left_is_correct_l192_192368

-- Define the initial conditions
def initial_popsicle_sticks : ℕ := 170
def popsicle_sticks_per_group : ℕ := 15
def number_of_groups : ℕ := 10

-- Define the total number of popsicle sticks given out to the groups
def total_sticks_given : ℕ := popsicle_sticks_per_group * number_of_groups

-- Define the number of popsicle sticks left
def sticks_left : ℕ := initial_popsicle_sticks - total_sticks_given

-- Prove that the number of sticks left is 20
theorem sticks_left_is_correct : sticks_left = 20 :=
by
  sorry

end sticks_left_is_correct_l192_192368


namespace expected_value_is_correct_l192_192842

noncomputable def expected_value_of_heads : ℝ :=
  let penny := 1 / 2 * 1
  let nickel := 1 / 2 * 5
  let dime := 1 / 2 * 10
  let quarter := 1 / 2 * 25
  let half_dollar := 1 / 2 * 50
  (penny + nickel + dime + quarter + half_dollar : ℝ)

theorem expected_value_is_correct : expected_value_of_heads = 45.5 := by
  sorry

end expected_value_is_correct_l192_192842


namespace find_a_9_l192_192604

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (d : ℤ)

-- Assumptions and definitions from the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def condition_one (a : ℕ → ℤ) : Prop := (a 1) + (a 2)^2 = -3
def condition_two (S : ℕ → ℤ) : Prop := S 5 = 10

-- Main theorem statement
theorem find_a_9 (h_arithmetic : arithmetic_sequence a d)
                 (h_sum : sum_of_arithmetic_sequence S a)
                 (h_cond1 : condition_one a)
                 (h_cond2 : condition_two S) : a 9 = 20 := 
sorry

end find_a_9_l192_192604


namespace validate_financial_position_l192_192492

noncomputable def financial_position_start_of_year := 86588
noncomputable def financial_position_end_of_year := 137236
noncomputable def total_tax := 8919
noncomputable def remaining_funds_after_tour := 38817

variables
(father_income monthly: ℝ := 50000)
(mother_income monthly: ℝ := 28000)
(grandmother_pension monthly: ℝ := 15000)
(mikhail_scholarship monthly: ℝ := 3000)
(father_tax_deduction monthly: ℝ := 2800)
(mother_tax_deduction monthly: ℝ := 2800)
(tax_rate: ℝ := 0.13)
(np_father: ℝ := father_income - father_tax_deduction)
(np_mother: ℝ := mother_income - mother_tax_deduction)

def net_father_tax (monthly:ℝ) := np_father * tax_rate
def net_mother_tax (monthly:ℝ) := np_mother * tax_rate

def father_monthly_income_after_tax (monthly:=ℝ) := father_income - net_father_tax
def mother_monthly_income_after_tax (monthly:=ℝ) := mother_income - net_mother_tax

def net_monthly_income (monthly:ℝ) := father_monthly_income_after_tax + mother_monthly_income_after_tax + grandmother_pension + mikhail_scholarship
def annual_net_income (yearly:=ℝ) := net_monthly_income * 12

variables
(financial_safety_cushion: ℝ := 10000 * 12)
(household_expenses: ℝ := (50000 + 15000) * 12)

def net_disposable_income_per_year (net_yearly:= ℝ) := annual_net_income - financial_safety_cushion - household_expenses

variables
(cadastral_value:ℝ := 6240000)
(sq_m:ℝ := 78)
(sq_m_reduction: ℝ := 20)
(rate_property: ℝ := 0.001)

def property_tax := (cadastral_value - sq_m_reduction * (cadastral_value / sq_m)) * rate_property

variables
(lada_prior_hp: ℝ := 106)
(lada_xray_hp: ℝ := 122)
(car_tax_rate: ℝ := 35)
(months_prior:ℝ := 3/12)
(months_xray: ℝ := 8/12)

def total_transport_tax := lada_prior_hp * car_tax_rate * months_prior + lada_xray_hp * car_tax_rate * months_xray

variables
(cadastral_value_land: ℝ := 420300)
(land_are: ℝ := 10)
(tax_rate_land: ℝ := 0.003)
(deducted_land_area:= 6)
 
def land_tax := (cadastral_value_land - (cadastral_value_land / land_are) * deducted_land_area) * tax_rate_land

def total_tax_liability := property_tax + total_transport_tax + land_tax

def after_tax_liquidity (total_tax_yearly:=ℝ) := net_disposable_income_per_year - total_tax_liability

variables
(tour_cost := 17900)
(participants := 5)

def remaining_after_tour := after_tax_liquidity - tour_cost * participants

theorem validate_financial_position :
  financial_position_start_of_year = 86588 ∧
  financial_position_end_of_year = 137236 ∧
  total_tax = 8919 ∧
  remaining_funds_after_tour = 38817 :=
by
  sorry 

end validate_financial_position_l192_192492


namespace total_cost_of_bill_l192_192654

def original_price_curtis := 16.00
def original_price_rob := 18.00
def time_of_meal := 3

def is_early_bird_discount_applicable (time : ℕ) : Prop :=
  2 ≤ time ∧ time ≤ 4

theorem total_cost_of_bill :
  is_early_bird_discount_applicable time_of_meal →
  original_price_curtis / 2 + original_price_rob / 2 = 17.00 :=
by
  sorry

end total_cost_of_bill_l192_192654


namespace limit_perimeters_eq_l192_192166

universe u

noncomputable def limit_perimeters (s : ℝ) : ℝ :=
  let a := 4 * s
  let r := 1 / 2
  a / (1 - r)

theorem limit_perimeters_eq (s : ℝ) : limit_perimeters s = 8 * s := by
  sorry

end limit_perimeters_eq_l192_192166


namespace probability_2_le_ξ_lt_4_l192_192897

-- Let ξ be a random variable following the normal distribution N(μ, σ^2)
variables {μ σ : ℝ}

def ξ : MeasureTheory.ProbabilityTheory.ProbabilityMonadic.random_variable nnreal ℝ :=
  MeasureTheory.ProbabilityTheory.ProbabilityMonadic.normal μ σ

-- Given conditions
axiom h1 : (MeasureTheory.ProbabilityTheory.Probability (ξ < 2)) = 0.15
axiom h2 : (MeasureTheory.ProbabilityTheory.Probability (ξ > 6)) = 0.15

-- We want to prove that P(2 ≤ ξ < 4) = 0.35
theorem probability_2_le_ξ_lt_4 : (MeasureTheory.ProbabilityTheory.Probability (2 ≤ ξ ∧ ξ < 4)) = 0.35 :=
sorry

end probability_2_le_ξ_lt_4_l192_192897


namespace proof_evaluate_expression_l192_192581

def evaluate_expression : Prop :=
  - (18 / 3 * 8 - 72 + 4 * 8) = 8

theorem proof_evaluate_expression : evaluate_expression :=
by 
  sorry

end proof_evaluate_expression_l192_192581


namespace find_a8_l192_192916

variable (a : ℕ → ℝ)
variable (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a8 
  (hq : is_geometric_sequence a q)
  (h1 : a 1 * a 3 = 4)
  (h2 : a 9 = 256) : 
  a 8 = 128 ∨ a 8 = -128 :=
by
  sorry

end find_a8_l192_192916


namespace solution_set_of_inequality_l192_192797

theorem solution_set_of_inequality :
  {x : ℝ | (x-2)*(3-x) > 0} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l192_192797


namespace initial_salt_percentage_l192_192553

theorem initial_salt_percentage (P : ℕ) : 
  let initial_solution := 100 
  let added_salt := 20 
  let final_solution := initial_solution + added_salt 
  (P / 100) * initial_solution + added_salt = (25 / 100) * final_solution → 
  P = 10 := 
by
  sorry

end initial_salt_percentage_l192_192553


namespace min_area_rectangle_l192_192723

theorem min_area_rectangle (l w : ℝ) 
  (hl : 3.5 ≤ l ∧ l ≤ 4.5) 
  (hw : 5.5 ≤ w ∧ w ≤ 6.5) 
  (constraint : l ≥ 2 * w) : 
  l * w = 60.5 := 
sorry

end min_area_rectangle_l192_192723


namespace city_miles_count_l192_192156

-- Defining the variables used in the conditions
def miles_per_gallon_city : ℝ := 30
def miles_per_gallon_highway : ℝ := 40
def highway_miles : ℝ := 200
def cost_per_gallon : ℝ := 3
def total_cost : ℝ := 42

-- Required statement for the proof, statement to prove: count of city miles is 270
theorem city_miles_count : ∃ (C : ℝ), C = 270 ∧
  (total_cost / cost_per_gallon) = ((C / miles_per_gallon_city) + (highway_miles / miles_per_gallon_highway)) :=
by
  sorry

end city_miles_count_l192_192156


namespace area_of_rectangular_field_l192_192297

theorem area_of_rectangular_field (L W A : ℕ) (h1 : L = 10) (h2 : 2 * W + L = 130) :
  A = 600 :=
by
  -- Proof will go here
  sorry

end area_of_rectangular_field_l192_192297


namespace solution1_solution2_l192_192589

-- Definitions of the conditions as inequalities
def condition1 (x : ℝ) : Prop := 2 * x^2 - 5 * x + 3 < 0
def condition2 (x : ℝ) : Prop := (x - 1) / (2 - x) ≤ 1

-- Proposition that these sets are the solution sets to respective inequalities
theorem solution1 :
  { x : ℝ | 1 < x ∧ x < 3 / 2 } = {x : ℝ | condition1 x } :=
by
  sorry

theorem solution2 :
  ({x : ℝ | x ≤ 3 / 2 } ∪ { x : ℝ | x > 2 }) = {x : ℝ | condition2 x } :=
by
  sorry

end solution1_solution2_l192_192589


namespace distance_midpoint_AB_to_line_l192_192179

noncomputable def tan_alpha_eq_neg2 (α : ℝ) : Prop := Real.tan α = -2

noncomputable def focus_F (α : ℝ) : (ℝ × ℝ) := (-Real.sin α * Real.cos α, 0)

noncomputable def line_l_intersects_AB (x1 x2 : ℝ) : Prop :=
  x1 + x2 + (4 / 5) = 4 ∧ (x2 - x1).abs = 4

theorem distance_midpoint_AB_to_line (α x1 x2 : ℝ) :
  tan_alpha_eq_neg2 α →
  let F := focus_F α in
  line_l_intersects_AB x1 x2 →
  let midpoint_x := (x1 + x2) / 2 in
  (midpoint_x - (-1 / 2)).abs = 21 / 10 :=
by
  sorry

end distance_midpoint_AB_to_line_l192_192179


namespace distance_midpoint_chord_AB_to_y_axis_l192_192335

theorem distance_midpoint_chord_AB_to_y_axis
  (k : ℝ)
  (A B : ℝ × ℝ)
  (hA : A.2 = k * A.1 - k)
  (hB : B.2 = k * B.1 - k)
  (hA_on_parabola : A.2 ^ 2 = 4 * A.1)
  (hB_on_parabola : B.2 ^ 2 = 4 * B.1)
  (h_distance_AB : dist A B = 4) :
  (abs ((A.1 + B.1) / 2)) = 1 :=
by
  sorry

end distance_midpoint_chord_AB_to_y_axis_l192_192335


namespace parabola_focus_coordinates_parabola_distance_to_directrix_l192_192789

-- Define constants and variables
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

noncomputable def focus_coordinates : ℝ × ℝ := (1, 0)

noncomputable def point : ℝ × ℝ := (4, 4)

noncomputable def directrix : ℝ := -1

noncomputable def distance_to_directrix : ℝ := 5

-- Proof statements
theorem parabola_focus_coordinates (x y : ℝ) (h : parabola_equation x y) : 
  focus_coordinates = (1, 0) :=
sorry

theorem parabola_distance_to_directrix (p : ℝ × ℝ) (d : ℝ) (h : p = point) (h_line : d = directrix) : 
  distance_to_directrix = 5 :=
  by
    -- Define and use the distance between point and vertical line formula
    sorry

end parabola_focus_coordinates_parabola_distance_to_directrix_l192_192789


namespace remainder_5x_div_9_l192_192828

theorem remainder_5x_div_9 {x : ℕ} (h : x % 9 = 5) : (5 * x) % 9 = 7 :=
sorry

end remainder_5x_div_9_l192_192828


namespace geometric_sum_S15_l192_192898

noncomputable def S (n : ℕ) : ℝ := sorry  -- Assume S is defined for the sequence sum

theorem geometric_sum_S15 (S_5 S_10 : ℝ) (h1 : S_5 = 5) (h2 : S_10 = 30) : 
    S 15 = 155 := 
by 
  -- Placeholder for geometric sequence proof
  sorry

end geometric_sum_S15_l192_192898


namespace solution_to_prime_equation_l192_192305

theorem solution_to_prime_equation (x y : ℕ) (p : ℕ) (h1 : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 = p * (xy + p) ↔ (x = 8 ∧ y = 1 ∧ p = 19) ∨ (x = 1 ∧ y = 8 ∧ p = 19) ∨ 
              (x = 7 ∧ y = 2 ∧ p = 13) ∨ (x = 2 ∧ y = 7 ∧ p = 13) ∨ 
              (x = 5 ∧ y = 4 ∧ p = 7) ∨ (x = 4 ∧ y = 5 ∧ p = 7) := sorry

end solution_to_prime_equation_l192_192305


namespace rail_elevation_correct_angle_l192_192415

noncomputable def rail_elevation_angle (v : ℝ) (R : ℝ) (g : ℝ) : ℝ :=
  Real.arctan (v^2 / (R * g))

theorem rail_elevation_correct_angle :
  rail_elevation_angle (60 * (1000 / 3600)) 200 9.8 = 8.09 := by
  sorry

end rail_elevation_correct_angle_l192_192415


namespace parabola_directrix_equation_l192_192006

theorem parabola_directrix_equation :
  ∀ (x y : ℝ),
  y = -4 * x^2 - 16 * x + 1 →
  ∃ d : ℝ, d = 273 / 16 ∧ y = d :=
by
  sorry

end parabola_directrix_equation_l192_192006


namespace joey_hourly_wage_l192_192222

def sneakers_cost : ℕ := 92
def mowing_earnings (lawns : ℕ) (rate : ℕ) : ℕ := lawns * rate
def selling_earnings (figures : ℕ) (rate : ℕ) : ℕ := figures * rate
def total_additional_earnings (mowing : ℕ) (selling : ℕ) : ℕ := mowing + selling
def remaining_amount (total_cost : ℕ) (earned : ℕ) : ℕ := total_cost - earned
def hourly_wage (remaining : ℕ) (hours : ℕ) : ℕ := remaining / hours

theorem joey_hourly_wage :
  let total_mowing := mowing_earnings 3 8
  let total_selling := selling_earnings 2 9
  let total_earned := total_additional_earnings total_mowing total_selling
  let remaining := remaining_amount sneakers_cost total_earned
  hourly_wage remaining 10 = 5 :=
by
  sorry

end joey_hourly_wage_l192_192222
