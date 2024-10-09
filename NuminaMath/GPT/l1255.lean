import Mathlib

namespace kit_time_to_ticket_window_l1255_125590

theorem kit_time_to_ticket_window 
  (rate : ℝ)
  (remaining_distance : ℝ)
  (yard_to_feet_conv : ℝ)
  (new_rate : rate = 90 / 30)
  (remaining_distance_in_feet : remaining_distance = 100 * yard_to_feet_conv)
  (yard_to_feet_conv_val : yard_to_feet_conv = 3) :
  (remaining_distance / rate = 100) := 
by 
  simp [new_rate, remaining_distance_in_feet, yard_to_feet_conv_val]
  sorry

end kit_time_to_ticket_window_l1255_125590


namespace football_field_area_l1255_125518

theorem football_field_area (total_fertilizer : ℝ) (partial_fertilizer : ℝ) (partial_area : ℝ) (fertilizer_rate : ℝ) (total_area : ℝ) 
  (h1 : total_fertilizer = 800)
  (h2: partial_fertilizer = 300)
  (h3: partial_area = 3600)
  (h4: fertilizer_rate = partial_fertilizer / partial_area)
  (h5: total_area = total_fertilizer / fertilizer_rate) 
  : total_area = 9600 := 
sorry

end football_field_area_l1255_125518


namespace calculate_down_payment_l1255_125541

theorem calculate_down_payment : 
  let monthly_fee := 12
  let years := 3
  let total_paid := 482
  let num_months := years * 12
  let total_monthly_payments := num_months * monthly_fee
  let down_payment := total_paid - total_monthly_payments
  down_payment = 50 :=
by
  sorry

end calculate_down_payment_l1255_125541


namespace JacobProof_l1255_125514

def JacobLadders : Prop :=
  let costPerRung : ℤ := 2
  let costPer50RungLadder : ℤ := 50 * costPerRung
  let num50RungLadders : ℤ := 10
  let totalPayment : ℤ := 3400
  let cost1 : ℤ := num50RungLadders * costPer50RungLadder
  let remainingAmount : ℤ := totalPayment - cost1
  let numRungs20Ladders : ℤ := remainingAmount / costPerRung
  numRungs20Ladders = 1200

theorem JacobProof : JacobLadders := by
  sorry

end JacobProof_l1255_125514


namespace mean_home_runs_correct_l1255_125531

def mean_home_runs (players: List ℕ) (home_runs: List ℕ) : ℚ :=
  let total_runs := (List.zipWith (· * ·) players home_runs).sum
  let total_players := players.sum
  total_runs / total_players

theorem mean_home_runs_correct :
  mean_home_runs [6, 4, 3, 1, 1, 1] [6, 7, 8, 10, 11, 12] = 121 / 16 :=
by
  -- The proof should go here
  sorry

end mean_home_runs_correct_l1255_125531


namespace proof_third_length_gcd_l1255_125528

/-- Statement: The greatest possible length that can be used to measure the given lengths exactly is 1 cm, 
and the third length is an unspecified number of centimeters that is relatively prime to both 1234 cm and 898 cm. -/
def third_length_gcd (x : ℕ) : Prop := 
  Int.gcd 1234 898 = 1 ∧ Int.gcd (Int.gcd 1234 898) x = 1

noncomputable def greatest_possible_length : ℕ := 1

theorem proof_third_length_gcd (x : ℕ) (h : third_length_gcd x) : greatest_possible_length = 1 := by
  sorry

end proof_third_length_gcd_l1255_125528


namespace fraction_strawberries_remaining_l1255_125571

theorem fraction_strawberries_remaining 
  (baskets : ℕ)
  (strawberries_per_basket : ℕ)
  (hedgehogs : ℕ)
  (strawberries_per_hedgehog : ℕ)
  (h1 : baskets = 3)
  (h2 : strawberries_per_basket = 900)
  (h3 : hedgehogs = 2)
  (h4 : strawberries_per_hedgehog = 1050) :
  (baskets * strawberries_per_basket - hedgehogs * strawberries_per_hedgehog) / (baskets * strawberries_per_basket) = 2 / 9 :=
by
  sorry

end fraction_strawberries_remaining_l1255_125571


namespace jennifer_fifth_score_l1255_125532

theorem jennifer_fifth_score :
  ∀ (x : ℝ), (85 + 90 + 87 + 92 + x) / 5 = 89 → x = 91 :=
by
  sorry

end jennifer_fifth_score_l1255_125532


namespace arithmetic_sequence_properties_l1255_125527

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h2 : d ≠ 0)
  (h3 : ∀ n, S n ≤ S 8) :
  d < 0 ∧ S 17 ≤ 0 := 
sorry

end arithmetic_sequence_properties_l1255_125527


namespace total_cost_of_bill_l1255_125524

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

end total_cost_of_bill_l1255_125524


namespace willy_days_worked_and_missed_l1255_125546

theorem willy_days_worked_and_missed:
  ∃ (x : ℚ), 8 * x = 10 * (30 - x) ∧ x = 50/3 ∧ (30 - x) = 40/3 :=
by
  sorry

end willy_days_worked_and_missed_l1255_125546


namespace average_sales_is_104_l1255_125593

-- Define the sales data for the months January to May
def january_sales : ℕ := 150
def february_sales : ℕ := 90
def march_sales : ℕ := 60
def april_sales : ℕ := 140
def may_sales : ℕ := 100
def may_discount : ℕ := 20

-- Define the adjusted sales for May after applying the discount
def adjusted_may_sales : ℕ := may_sales - (may_sales * may_discount / 100)

-- Define the total sales from January to May
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + adjusted_may_sales

-- Define the number of months
def number_of_months : ℕ := 5

-- Define the average sales per month
def average_sales_per_month : ℕ := total_sales / number_of_months

-- Prove that the average sales per month is equal to 104
theorem average_sales_is_104 : average_sales_per_month = 104 := by
  -- Here, we'd write the proof, but we'll leave it as 'sorry' for now
  sorry

end average_sales_is_104_l1255_125593


namespace m_plus_n_eq_123_l1255_125550

/- Define the smallest prime number -/
def m : ℕ := 2

/- Define the largest integer less than 150 with exactly three positive divisors -/
def n : ℕ := 121

/- Prove that the sum of m and n is 123 -/
theorem m_plus_n_eq_123 : m + n = 123 := by
  -- By definition, m is 2 and n is 121
  -- So, their sum is 123
  rfl

end m_plus_n_eq_123_l1255_125550


namespace intersection_complement_l1255_125589

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 5, 6})
variable (hB : B = {1, 3, 4, 6, 7})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 5} :=
sorry

end intersection_complement_l1255_125589


namespace geometric_series_common_ratio_l1255_125562

theorem geometric_series_common_ratio (a : ℝ) (r : ℝ) (S : ℝ) (h1 : S = a / (1 - r))
  (h2 : S = 16 * (r^2 * S)) : |r| = 1/4 :=
by
  sorry

end geometric_series_common_ratio_l1255_125562


namespace sin_2x_value_l1255_125503

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x)

theorem sin_2x_value (x : ℝ) (h1 : f x = 5 / 3) (h2 : -Real.pi / 6 < x) (h3 : x < Real.pi / 6) :
  Real.sin (2 * x) = (Real.sqrt 3 - 2 * Real.sqrt 2) / 6 := 
sorry

end sin_2x_value_l1255_125503


namespace cost_per_tree_l1255_125581

theorem cost_per_tree
    (initial_temperature : ℝ := 80)
    (final_temperature : ℝ := 78.2)
    (total_cost : ℝ := 108)
    (temperature_drop_per_tree : ℝ := 0.1) :
    total_cost / ((initial_temperature - final_temperature) / temperature_drop_per_tree) = 6 :=
by sorry

end cost_per_tree_l1255_125581


namespace plane_distance_l1255_125506

theorem plane_distance :
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  total_distance_AD = 550 :=
by
  intros
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  sorry

end plane_distance_l1255_125506


namespace rectangle_ratio_l1255_125526

theorem rectangle_ratio (s : ℝ) (h : s > 0) :
    let large_square_side := 3 * s
    let rectangle_length := 3 * s
    let rectangle_width := 2 * s
    rectangle_length / rectangle_width = 3 / 2 := by
  sorry

end rectangle_ratio_l1255_125526


namespace arithmetic_sequence_n_value_l1255_125557

theorem arithmetic_sequence_n_value (a : ℕ → ℤ) (a1 : a 1 = 1) (d : ℤ) (d_def : d = 3) (an : ∃ n, a n = 22) :
  ∃ n, n = 8 :=
by
  -- Assume the general term formula for the arithmetic sequence
  have general_term : ∀ n, a n = a 1 + (n-1) * d := sorry
  -- Use the given conditions
  have a_n_22 : ∃ n, a n = 22 := an
  -- Calculations to derive n = 8, skipped here
  sorry

end arithmetic_sequence_n_value_l1255_125557


namespace curve_transformation_l1255_125509

theorem curve_transformation :
  (∀ (x y : ℝ), 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1) → (∀ (x y : ℝ), 50 * x^2 + 72 * y^2 = 1) :=
by
  intros h x y
  have h1 : 2 * (5 * x)^2 + 8 * (3 * y)^2 = 1 := h x y
  sorry

end curve_transformation_l1255_125509


namespace range_of_a_l1255_125570

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 + 2 * a * x + 2 < 0) ↔ 0 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l1255_125570


namespace max_dist_to_origin_from_curve_l1255_125520

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  let x := 3 + Real.sin θ
  let y := Real.cos θ
  (x, y)

theorem max_dist_to_origin_from_curve :
  ∃ M : ℝ × ℝ, (∃ θ : ℝ, M = curve θ) ∧ Real.sqrt (M.fst^2 + M.snd^2) ≤ 4 :=
by
  sorry

end max_dist_to_origin_from_curve_l1255_125520


namespace product_eq_neg_one_l1255_125579

theorem product_eq_neg_one (m b : ℚ) (hm : m = -2 / 3) (hb : b = 3 / 2) : m * b = -1 :=
by
  rw [hm, hb]
  sorry

end product_eq_neg_one_l1255_125579


namespace determine_sunday_l1255_125561

def Brother := Prop -- A type to represent a brother

variable (A B : Brother)
variable (T D : Brother) -- T representing Tweedledum, D representing Tweedledee

-- Conditions translated into Lean
variable (H1 : (A = T) → (B = D))
variable (H2 : (B = D) → (A = T))

-- Define the day of the week as a proposition
def is_sunday := Prop

-- We want to state that given H1 and H2, it is Sunday
theorem determine_sunday (H1 : (A = T) → (B = D)) (H2 : (B = D) → (A = T)) : is_sunday := sorry

end determine_sunday_l1255_125561


namespace rug_length_l1255_125551

theorem rug_length (d : ℕ) (x y : ℕ) (h1 : x * x + y * y = d * d) (h2 : y / x = 2) (h3 : (x = 25 ∧ y = 50)) : 
  x = 25 := 
sorry

end rug_length_l1255_125551


namespace symmetric_point_x_axis_l1255_125521

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  {x := p.x, y := -p.y, z := -p.z}

theorem symmetric_point_x_axis :
  symmetricWithRespectToXAxis ⟨-1, -2, 3⟩ = ⟨-1, 2, -3⟩ :=
  by
    sorry

end symmetric_point_x_axis_l1255_125521


namespace length_of_one_side_l1255_125566

-- Definitions according to the conditions
def perimeter (nonagon : Type) : ℝ := 171
def sides (nonagon : Type) : ℕ := 9

-- Math proof problem to prove
theorem length_of_one_side (nonagon : Type) : perimeter nonagon / sides nonagon = 19 :=
by
  sorry

end length_of_one_side_l1255_125566


namespace fraction_multiplication_l1255_125559

theorem fraction_multiplication :
  (3 / 4) ^ 5 * (4 / 3) ^ 2 = 8 / 19 :=
by
  sorry

end fraction_multiplication_l1255_125559


namespace abs_a_gt_b_l1255_125572

theorem abs_a_gt_b (a b : ℝ) (h : a > b) : |a| > b :=
sorry

end abs_a_gt_b_l1255_125572


namespace find_x_for_f_eq_f_inv_l1255_125575

def f (x : ℝ) : ℝ := 3 * x - 8

noncomputable def f_inv (x : ℝ) : ℝ := (x + 8) / 3

theorem find_x_for_f_eq_f_inv : ∃ x : ℝ, f x = f_inv x ∧ x = 4 :=
by
  sorry

end find_x_for_f_eq_f_inv_l1255_125575


namespace probability_distribution_correct_l1255_125588

noncomputable def numCombinations (n k : ℕ) : ℕ :=
  (Nat.choose n k)

theorem probability_distribution_correct :
  let totalCombinations := numCombinations 5 2
  let prob_two_red := (numCombinations 3 2 : ℚ) / totalCombinations
  let prob_two_white := (numCombinations 2 2 : ℚ) / totalCombinations
  let prob_one_red_one_white := ((numCombinations 3 1) * (numCombinations 2 1) : ℚ) / totalCombinations
  (prob_two_red, prob_one_red_one_white, prob_two_white) = (0.3, 0.6, 0.1) :=
by
  sorry

end probability_distribution_correct_l1255_125588


namespace arithmetic_sequence_value_l1255_125525

variable (a : ℕ → ℝ)
variable (a₁ d a₇ a₅ : ℝ)
variable (h_seq : ∀ n, a n = a₁ + (n - 1) * d)
variable (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120)

theorem arithmetic_sequence_value :
  a 7 - 1/3 * a 5 = 16 :=
sorry

end arithmetic_sequence_value_l1255_125525


namespace find_a2_l1255_125522

theorem find_a2 
  (a1 a2 a3 : ℝ)
  (h1 : a1 * a2 * a3 = 15)
  (h2 : (3 / (a1 * 3 * a2)) + (15 / (3 * a2 * 5 * a3)) + (5 / (5 * a3 * a1)) = 3 / 5) :
  a2 = 3 :=
sorry

end find_a2_l1255_125522


namespace polynomial_divisibility_l1255_125543

theorem polynomial_divisibility :
  ∃ (p : Polynomial ℤ), (Polynomial.X ^ 2 - Polynomial.X + 2) * p = Polynomial.X ^ 15 + Polynomial.X ^ 2 + 100 :=
by
  sorry

end polynomial_divisibility_l1255_125543


namespace sum_of_3x3_matrix_arithmetic_eq_45_l1255_125553

-- Statement: Prove that the sum of all nine elements of a 3x3 matrix, where each row and each column forms an arithmetic sequence and the middle element a_{22} = 5, is 45
theorem sum_of_3x3_matrix_arithmetic_eq_45 
  (matrix : ℤ → ℤ → ℤ)
  (arithmetic_row : ∀ i, matrix i 0 + matrix i 1 + matrix i 2 = 3 * matrix i 1)
  (arithmetic_col : ∀ j, matrix 0 j + matrix 1 j + matrix 2 j = 3 * matrix 1 j)
  (middle_elem : matrix 1 1 = 5) : 
  (matrix 0 0 + matrix 0 1 + matrix 0 2 + matrix 1 0 + matrix 1 1 + matrix 1 2 + matrix 2 0 + matrix 2 1 + matrix 2 2) = 45 :=
by
  sorry -- proof to be provided

end sum_of_3x3_matrix_arithmetic_eq_45_l1255_125553


namespace thirty_three_and_one_third_percent_of_330_l1255_125576

theorem thirty_three_and_one_third_percent_of_330 :
  (33 + 1 / 3) / 100 * 330 = 110 :=
sorry

end thirty_three_and_one_third_percent_of_330_l1255_125576


namespace solve_inequality_system_l1255_125596

theorem solve_inequality_system (y : ℝ) :
  (2 * (y + 1) < 5 * y - 7) ∧ ((y + 2) / 2 < 5) ↔ (3 < y) ∧ (y < 8) := 
by
  sorry

end solve_inequality_system_l1255_125596


namespace remainder_div_product_l1255_125592

theorem remainder_div_product (P D D' D'' Q R Q' R' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = Q' * D' + R') 
  (h3 : Q' = Q'' * D'' + R'') :
  P % (D * D' * D'') = D * D' * R'' + D * R' + R := 
sorry

end remainder_div_product_l1255_125592


namespace Jennifer_apples_l1255_125537

-- Define the conditions
def initial_apples : ℕ := 7
def found_apples : ℕ := 74

-- The theorem to prove
theorem Jennifer_apples : initial_apples + found_apples = 81 :=
by
  -- proof goes here, but we use sorry to skip the proof step
  sorry

end Jennifer_apples_l1255_125537


namespace ratio_of_ages_l1255_125598

variable (J L M : ℕ)

def louis_age := L = 14
def matilda_age := M = 35
def matilda_older := M = J + 7
def jerica_multiple := ∃ k : ℕ, J = k * L

theorem ratio_of_ages
  (hL : louis_age L)
  (hM : matilda_age M)
  (hMO : matilda_older J M)
  : J / L = 2 :=
by
  sorry

end ratio_of_ages_l1255_125598


namespace area_inner_square_l1255_125568

theorem area_inner_square (ABCD_side : ℝ) (BE : ℝ) (EFGH_area : ℝ) 
  (h1 : ABCD_side = Real.sqrt 50) 
  (h2 : BE = 1) :
  EFGH_area = 36 :=
by
  sorry

end area_inner_square_l1255_125568


namespace circle_equation_of_diameter_l1255_125501

theorem circle_equation_of_diameter (A B : ℝ × ℝ) (hA : A = (-4, -5)) (hB : B = (6, -1)) :
  ∃ h k r : ℝ, (x - h)^2 + (y - k)^2 = r ∧ h = 1 ∧ k = -3 ∧ r = 29 := 
by
  sorry

end circle_equation_of_diameter_l1255_125501


namespace intersecting_line_at_one_point_l1255_125542

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l1255_125542


namespace fraction_equality_l1255_125534

theorem fraction_equality (x y : ℝ) : (-x + y) / (-x - y) = (x - y) / (x + y) :=
by sorry

end fraction_equality_l1255_125534


namespace length_of_second_train_l1255_125523

/-
  Given:
  - l₁ : Length of the first train in meters
  - v₁ : Speed of the first train in km/h
  - v₂ : Speed of the second train in km/h
  - t : Time to cross the second train in seconds

  Prove:
  - l₂ : Length of the second train in meters = 299.9560035197185 meters
-/

variable (l₁ : ℝ) (v₁ : ℝ) (v₂ : ℝ) (t : ℝ) (l₂ : ℝ)

theorem length_of_second_train
  (h₁ : l₁ = 250)
  (h₂ : v₁ = 72)
  (h₃ : v₂ = 36)
  (h₄ : t = 54.995600351971845)
  (h_result : l₂ = 299.9560035197185) :
  (v₁ * 1000 / 3600 - v₂ * 1000 / 3600) * t - l₁ = l₂ := by
  sorry

end length_of_second_train_l1255_125523


namespace isosceles_triangle_perimeter_l1255_125574

theorem isosceles_triangle_perimeter (a b : ℕ) (h_a : a = 8 ∨ a = 9) (h_b : b = 8 ∨ b = 9) 
(h_iso : a = a) (h_tri_ineq : a + a > b ∧ a + b > a ∧ b + a > a) :
  a + a + b = 25 ∨ a + a + b = 26 := 
by
  sorry

end isosceles_triangle_perimeter_l1255_125574


namespace peanut_total_correct_l1255_125563

-- Definitions based on the problem conditions:

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35
def total_peanuts : ℕ := jose_peanuts + kenya_peanuts + malachi_peanuts

-- Statement to be proven:
theorem peanut_total_correct : total_peanuts = 386 :=
by 
  -- The proof would be here, but we skip it according to the instruction
  sorry

end peanut_total_correct_l1255_125563


namespace mooncake_packaging_problem_l1255_125583

theorem mooncake_packaging_problem :
  ∃ x y : ℕ, 9 * x + 4 * y = 35 ∧ x + y = 5 :=
by
  -- Proof is omitted
  sorry

end mooncake_packaging_problem_l1255_125583


namespace scientific_notation_l1255_125508

theorem scientific_notation (n : ℝ) (h1 : n = 17600) : ∃ a b, (a = 1.76) ∧ (b = 4) ∧ n = a * 10^b :=
by {
  sorry
}

end scientific_notation_l1255_125508


namespace empty_with_three_pumps_in_12_minutes_l1255_125512

-- Define the conditions
def conditions (a b x : ℝ) : Prop :=
  x = a + b ∧ 2 * x = 3 * a + b

-- Define the main theorem to prove
theorem empty_with_three_pumps_in_12_minutes (a b x : ℝ) (h : conditions a b x) : 
  (3 * (1 / 5) * x = a + (1 / 5) * b) ∧ ((1 / 5) * 60 = 12) := 
by
  -- Use the given conditions in the proof.
  sorry

end empty_with_three_pumps_in_12_minutes_l1255_125512


namespace length_base_bc_l1255_125584

theorem length_base_bc {A B C D : Type} [Inhabited A]
  (AB AC : ℕ)
  (BD : ℕ → ℕ → ℕ → ℕ) -- function for the median on AC
  (perimeter1 perimeter2 : ℕ)
  (h1 : AB = AC)
  (h2 : perimeter1 = 24 ∨ perimeter2 = 30)
  (AD CD : ℕ) :
  (AD = CD ∧ (∃ ab ad cd, ab + ad = perimeter1 ∧ cd + ad = perimeter2 ∧ ((AB = 2 * AD ∧ BC = 30 - CD) ∨ (AB = 2 * AD ∧ BC = 24 - CD)))) →
  (BC = 22 ∨ BC = 14) := 
sorry

end length_base_bc_l1255_125584


namespace parallel_lines_slope_l1255_125594

theorem parallel_lines_slope (a : ℝ) :
  (∀ (x y : ℝ), x + a * y + 6 = 0 ∧ (a - 2) * x + 3 * y + 2 * a = 0 → (1 / (a - 2) = a / 3)) →
  a = -1 :=
by {
  sorry
}

end parallel_lines_slope_l1255_125594


namespace first_discount_percentage_l1255_125519

theorem first_discount_percentage (original_price final_price : ℝ) (additional_discount : ℝ) (x : ℝ) 
  (h1 : original_price = 400) 
  (h2 : additional_discount = 0.05) 
  (h3 : final_price = 342) 
  (hx : (original_price * (100 - x) / 100) * (1 - additional_discount) = final_price) :
  x = 10 := 
sorry

end first_discount_percentage_l1255_125519


namespace myrtle_hens_l1255_125554

/-- Myrtle has some hens that lay 3 eggs a day. She was gone for 7 days and told her neighbor 
    to take as many as they would like. The neighbor took 12 eggs. Once home, Myrtle collected 
    the remaining eggs, dropping 5 on the way into her house. Myrtle has 46 eggs. Prove 
    that Myrtle has 3 hens. -/
theorem myrtle_hens (eggs_per_hen_per_day hens days neighbor_took dropped remaining_hens_eggs : ℕ) 
    (h1 : eggs_per_hen_per_day = 3) 
    (h2 : days = 7) 
    (h3 : neighbor_took = 12) 
    (h4 : dropped = 5) 
    (h5 : remaining_hens_eggs = 46) : 
    hens = 3 := 
by 
  sorry

end myrtle_hens_l1255_125554


namespace rearrangement_inequality_l1255_125555

theorem rearrangement_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c ∧ c = a ↔ (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2)) :=
by 
  -- Proof omitted
  sorry

end rearrangement_inequality_l1255_125555


namespace real_number_solution_pure_imaginary_solution_zero_solution_l1255_125556

noncomputable def real_number_condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 2 = 0

noncomputable def pure_imaginary_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ ¬(m^2 - 3 * m + 2 = 0)

noncomputable def zero_condition (m : ℝ) : Prop :=
  (2 * m^2 - 3 * m - 2 = 0) ∧ (m^2 - 3 * m + 2 = 0)

theorem real_number_solution (m : ℝ) : real_number_condition m ↔ (m = 1 ∨ m = 2) := 
sorry

theorem pure_imaginary_solution (m : ℝ) : pure_imaginary_condition m ↔ (m = -1 / 2) :=
sorry

theorem zero_solution (m : ℝ) : zero_condition m ↔ (m = 2) :=
sorry

end real_number_solution_pure_imaginary_solution_zero_solution_l1255_125556


namespace tub_drain_time_l1255_125599

theorem tub_drain_time (t : ℝ) (p q : ℝ) (h1 : t = 4) (h2 : p = 5 / 7) (h3 : q = 2 / 7) :
  q * t / p = 1.6 := by
  sorry

end tub_drain_time_l1255_125599


namespace total_miles_traveled_l1255_125511

-- Define the conditions
def travel_time_per_mile (n : ℕ) : ℕ :=
  match n with
  | 0 => 10
  | _ => 10 + 6 * n

def daily_miles (n : ℕ) : ℕ :=
  60 / travel_time_per_mile n

-- Statement of the problem
theorem total_miles_traveled : (daily_miles 0 + daily_miles 1 + daily_miles 2 + daily_miles 3 + daily_miles 4) = 20 := by
  sorry

end total_miles_traveled_l1255_125511


namespace tournament_matches_divisible_by_7_l1255_125548

-- Define the conditions of the chess tournament
def single_elimination_tournament_matches (players byes: ℕ) : ℕ :=
  players - 1

theorem tournament_matches_divisible_by_7 :
  single_elimination_tournament_matches 120 40 = 119 ∧ 119 % 7 = 0 :=
by
  sorry

end tournament_matches_divisible_by_7_l1255_125548


namespace range_of_m_for_inequality_l1255_125515

-- Define the condition
def condition (x : ℝ) := x ∈ Set.Iic (-1)

-- Define the inequality for proving the range of m
def inequality_holds (m x : ℝ) : Prop := (m - m^2) * 4^x + 2^x + 1 > 0

-- Prove the range of m for the given conditions such that the inequality holds
theorem range_of_m_for_inequality :
  (∀ (x : ℝ), condition x → inequality_holds m x) ↔ (-2 < m ∧ m < 3) :=
sorry

end range_of_m_for_inequality_l1255_125515


namespace upper_limit_opinion_l1255_125547

theorem upper_limit_opinion (w : ℝ) 
  (H1 : 61 < w ∧ w < 72) 
  (H2 : 60 < w ∧ w < 70) 
  (H3 : (61 + w) / 2 = 63) : w = 65 := 
by
  sorry

end upper_limit_opinion_l1255_125547


namespace unsuccessful_attempts_124_l1255_125507

theorem unsuccessful_attempts_124 (num_digits: ℕ) (choices_per_digit: ℕ) (total_attempts: ℕ):
  num_digits = 3 → choices_per_digit = 5 → total_attempts = choices_per_digit ^ num_digits →
  total_attempts - 1 = 124 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact sorry

end unsuccessful_attempts_124_l1255_125507


namespace find_number_l1255_125565

noncomputable def number_with_point_one_percent (x : ℝ) : Prop :=
  0.1 * x / 100 = 12.356

theorem find_number :
  ∃ x : ℝ, number_with_point_one_percent x ∧ x = 12356 :=
by
  sorry

end find_number_l1255_125565


namespace greatest_integer_func_l1255_125536

noncomputable def pi_approx : ℝ := 3.14159

theorem greatest_integer_func : (⌊2 * pi_approx - 6⌋ : ℝ) = 0 := 
by
  sorry

end greatest_integer_func_l1255_125536


namespace find_R_l1255_125564

theorem find_R (a b : ℝ) (Q R : ℝ) (hQ : Q = 4)
  (h1 : 1/a + 1/b = Q/(a + b))
  (h2 : a/b + b/a = R) : R = 2 :=
by
  sorry

end find_R_l1255_125564


namespace return_trip_time_l1255_125585

variables (d p w : ℝ)
-- Condition 1: The outbound trip against the wind took 120 minutes.
axiom h1 : d = 120 * (p - w)
-- Condition 2: The return trip with the wind took 15 minutes less than it would in still air.
axiom h2 : d / (p + w) = d / p - 15

-- Translate the conclusion that needs to be proven in Lean 4
theorem return_trip_time (h1 : d = 120 * (p - w)) (h2 : d / (p + w) = d / p - 15) : (d / (p + w) = 15) ∨ (d / (p + w) = 85) :=
sorry

end return_trip_time_l1255_125585


namespace remainder_division_lemma_l1255_125587

theorem remainder_division_lemma (j : ℕ) (hj : 0 < j) (hmod : 132 % (j^2) = 12) : 250 % j = 0 :=
sorry

end remainder_division_lemma_l1255_125587


namespace find_missing_digit_l1255_125533

theorem find_missing_digit (B : ℕ) : 
  (B = 2 ∨ B = 4 ∨ B = 7 ∨ B = 8 ∨ B = 9) → 
  (2 * 1000 + B * 100 + 4 * 10 + 0) % 15 = 0 → 
  B = 7 :=
by 
  intro h1 h2
  sorry

end find_missing_digit_l1255_125533


namespace geom_sequence_eq_l1255_125516

theorem geom_sequence_eq :
  ∀ {a : ℕ → ℝ} {q : ℝ}, (∀ n, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by
  intro a q hgeom hsum hsum_sq
  sorry

end geom_sequence_eq_l1255_125516


namespace isabel_camera_pics_l1255_125586

-- Conditions
def phone_pics := 2
def albums := 3
def pics_per_album := 2

-- Define the total pictures and camera pictures
def total_pics := albums * pics_per_album
def camera_pics := total_pics - phone_pics

theorem isabel_camera_pics : camera_pics = 4 :=
by
  -- The goal is translated from the correct answer in step b)
  sorry

end isabel_camera_pics_l1255_125586


namespace center_of_circle_l1255_125529

theorem center_of_circle : 
  ∀ x y : ℝ, 4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 29 = 0 → (x = -1 ∧ y = 3 / 2) :=
by
  sorry

end center_of_circle_l1255_125529


namespace marble_problem_l1255_125591

theorem marble_problem
  (h1 : ∀ x : ℕ, x > 0 → (x + 2) * ((220 / x) - 1) = 220) :
  ∃ x : ℕ, x > 0 ∧ (x + 2) * ((220 / ↑x) - 1) = 220 ∧ x = 20 :=
by
  sorry

end marble_problem_l1255_125591


namespace no_integer_solution_l1255_125502

theorem no_integer_solution (a b c d : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) :
  ¬ ∃ n : ℤ, n^4 - (a : ℤ)*n^3 - (b : ℤ)*n^2 - (c : ℤ)*n - (d : ℤ) = 0 :=
sorry

end no_integer_solution_l1255_125502


namespace tangent_identity_l1255_125545

theorem tangent_identity :
  Real.tan (55 * Real.pi / 180) * 
  Real.tan (65 * Real.pi / 180) * 
  Real.tan (75 * Real.pi / 180) = 
  Real.tan (85 * Real.pi / 180) :=
sorry

end tangent_identity_l1255_125545


namespace total_number_of_animals_l1255_125538

-- Define the given conditions as hypotheses
def num_horses (T : ℕ) : Prop :=
  ∃ (H x z : ℕ), H + x + z = 75

def cows_vs_horses (T : ℕ) : Prop :=
  ∃ (w z : ℕ),  w = z + 10

-- Define the final conclusion we need to prove
def total_animals (T : ℕ) : Prop :=
  T = 170

-- The main theorem which states the conditions imply the conclusion
theorem total_number_of_animals (T : ℕ) (h1 : num_horses T) (h2 : cows_vs_horses T) : total_animals T :=
by
  -- Proof to be filled in later
  sorry

end total_number_of_animals_l1255_125538


namespace diameterOuterBoundary_l1255_125560

-- Definitions based on the conditions in the problem
def widthWalkingPath : ℝ := 10
def widthGardenRing : ℝ := 12
def diameterPond : ℝ := 16

-- The main theorem that proves the diameter of the circle that forms the outer boundary of the walking path
theorem diameterOuterBoundary : 2 * ((diameterPond / 2) + widthGardenRing + widthWalkingPath) = 60 :=
by
  sorry

end diameterOuterBoundary_l1255_125560


namespace value_of_5_S_3_l1255_125544

def S (a b : ℕ) : ℕ := 4 * a + 6 * b + 1

theorem value_of_5_S_3 : S 5 3 = 39 := by
  sorry

end value_of_5_S_3_l1255_125544


namespace find_cos2α_l1255_125573

noncomputable def cos2α (tanα : ℚ) : ℚ :=
  (1 - tanα^2) / (1 + tanα^2)

theorem find_cos2α (h : tanα = (3 / 4)) : cos2α tanα = (7 / 25) :=
by
  rw [cos2α, h]
  -- here the simplification steps would be performed
  sorry

end find_cos2α_l1255_125573


namespace joshua_total_bottle_caps_l1255_125577

def initial_bottle_caps : ℕ := 40
def bought_bottle_caps : ℕ := 7

theorem joshua_total_bottle_caps : initial_bottle_caps + bought_bottle_caps = 47 := 
by
  sorry

end joshua_total_bottle_caps_l1255_125577


namespace tangent_line_through_origin_l1255_125597

theorem tangent_line_through_origin (x : ℝ) (h₁ : 0 < x) (h₂ : ∀ x, ∃ y, y = 2 * Real.log x) (h₃ : ∀ x, y = 2 * Real.log x) :
  x = Real.exp 1 :=
sorry

end tangent_line_through_origin_l1255_125597


namespace apples_difference_l1255_125549

def jimin_apples : ℕ := 7
def grandpa_apples : ℕ := 13
def younger_brother_apples : ℕ := 8
def younger_sister_apples : ℕ := 5

theorem apples_difference :
  grandpa_apples - younger_sister_apples = 8 :=
by
  sorry

end apples_difference_l1255_125549


namespace average_increase_l1255_125569

-- Definitions
def runs_11 := 90
def avg_11 := 40

-- Conditions
def total_runs_before (A : ℕ) := A * 10
def total_runs_after (runs_11 : ℕ) (total_runs_before : ℕ) := total_runs_before + runs_11
def increased_average (avg_11 : ℕ) (avg_before : ℕ) := avg_11 = avg_before + 5

-- Theorem stating the equivalent proof problem
theorem average_increase
  (A : ℕ)
  (H1 : total_runs_after runs_11 (total_runs_before A) = 40 * 11)
  (H2 : avg_11 = 40) :
  increased_average 40 A := 
sorry

end average_increase_l1255_125569


namespace jerry_age_l1255_125567

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 16) : J = 11 :=
sorry

end jerry_age_l1255_125567


namespace sum_of_first_30_terms_l1255_125552

variable (a : Nat → ℤ)
variable (d : ℤ)
variable (S_30 : ℤ)

-- Conditions from part a)
def condition1 := a 1 + a 2 + a 3 = 3
def condition2 := a 28 + a 29 + a 30 = 165

-- Question translated to Lean 4 statement
theorem sum_of_first_30_terms 
  (h1 : condition1 a)
  (h2 : condition2 a) :
  S_30 = 840 := 
sorry

end sum_of_first_30_terms_l1255_125552


namespace average_annual_percentage_decrease_l1255_125505

theorem average_annual_percentage_decrease (P2018 P2020 : ℝ) (x : ℝ) 
  (h_initial : P2018 = 20000)
  (h_final : P2020 = 16200) :
  P2018 * (1 - x)^2 = P2020 :=
by
  sorry

end average_annual_percentage_decrease_l1255_125505


namespace calculate_total_driving_time_l1255_125535

/--
A rancher needs to transport 400 head of cattle to higher ground 60 miles away.
His truck holds 20 head of cattle and travels at 60 miles per hour.
Prove that the total driving time to transport all cattle is 40 hours.
-/
theorem calculate_total_driving_time
  (total_cattle : Nat)
  (cattle_per_trip : Nat)
  (distance_one_way : Nat)
  (speed : Nat)
  (round_trip_miles : Nat)
  (total_miles : Nat)
  (total_time_hours : Nat)
  (h1 : total_cattle = 400)
  (h2 : cattle_per_trip = 20)
  (h3 : distance_one_way = 60)
  (h4 : speed = 60)
  (h5 : round_trip_miles = 2 * distance_one_way)
  (h6 : total_miles = (total_cattle / cattle_per_trip) * round_trip_miles)
  (h7 : total_time_hours = total_miles / speed) :
  total_time_hours = 40 :=
by
  sorry

end calculate_total_driving_time_l1255_125535


namespace onions_left_l1255_125500

def sallyOnions : ℕ := 5
def fredOnions : ℕ := 9
def onionsGivenToSara : ℕ := 4

theorem onions_left : (sallyOnions + fredOnions) - onionsGivenToSara = 10 := by
  sorry

end onions_left_l1255_125500


namespace area_of_equilateral_triangle_inscribed_in_square_l1255_125530

variables {a : ℝ}

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  a^2 * (2 * Real.sqrt 3 - 3)

theorem area_of_equilateral_triangle_inscribed_in_square (a : ℝ) :
  equilateral_triangle_area a = a^2 * (2 * Real.sqrt 3 - 3) :=
by sorry

end area_of_equilateral_triangle_inscribed_in_square_l1255_125530


namespace num_orders_javier_constraint_l1255_125513

noncomputable def num_valid_orders : ℕ :=
  Nat.factorial 5 / 2

theorem num_orders_javier_constraint : num_valid_orders = 60 := 
by
  sorry

end num_orders_javier_constraint_l1255_125513


namespace packets_in_box_l1255_125580

theorem packets_in_box 
  (coffees_per_day : ℕ) 
  (packets_per_coffee : ℕ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (days : ℕ) 
  (P : ℕ) 
  (h_coffees_per_day : coffees_per_day = 2)
  (h_packets_per_coffee : packets_per_coffee = 1)
  (h_cost_per_box : cost_per_box = 4)
  (h_total_cost : total_cost = 24)
  (h_days : days = 90)
  : P = 30 := 
by
  sorry

end packets_in_box_l1255_125580


namespace silver_medals_count_l1255_125510

def total_medals := 67
def gold_medals := 19
def bronze_medals := 16
def silver_medals := total_medals - gold_medals - bronze_medals

theorem silver_medals_count : silver_medals = 32 := by
  -- Proof goes here
  sorry

end silver_medals_count_l1255_125510


namespace units_digit_of_fraction_example_l1255_125558

def units_digit_of_fraction (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem units_digit_of_fraction_example :
  units_digit_of_fraction (25 * 26 * 27 * 28 * 29 * 30) 1250 = 2 := by
  sorry

end units_digit_of_fraction_example_l1255_125558


namespace find_smaller_number_l1255_125539

noncomputable def smaller_number (x y : ℝ) := y

theorem find_smaller_number 
  (x y : ℝ) 
  (h1 : x - y = 9) 
  (h2 : x + y = 46) :
  smaller_number x y = 18.5 :=
sorry

end find_smaller_number_l1255_125539


namespace ship_lighthouse_distance_l1255_125595

-- Definitions for conditions
def speed : ℝ := 15 -- speed of the ship in km/h
def time : ℝ := 4  -- time the ship sails eastward in hours
def angle_A : ℝ := 60 -- angle at point A in degrees
def angle_C : ℝ := 30 -- angle at point C in degrees

-- Main theorem statement
theorem ship_lighthouse_distance (d_A_C : ℝ) (d_C_B : ℝ) : d_A_C = speed * time → d_C_B = 60 := 
by sorry

end ship_lighthouse_distance_l1255_125595


namespace gas_cost_l1255_125504

theorem gas_cost 
  (x : ℝ)
  (h1 : 5 * (x / 5) = x)
  (h2 : 8 * (x / 8) = x)
  (h3 : (x / 5) - 15.50 = (x / 8)) : 
  x = 206.67 :=
by
  sorry

end gas_cost_l1255_125504


namespace cube_points_l1255_125578

theorem cube_points (A B C D E F : ℕ) 
  (h1 : A + B = 13)
  (h2 : C + D = 13)
  (h3 : E + F = 13)
  (h4 : A + C + E = 16)
  (h5 : B + D + E = 24) :
  F = 6 :=
by
  sorry  -- Proof to be filled in by the user

end cube_points_l1255_125578


namespace prism_surface_area_is_14_l1255_125540

-- Definition of the rectangular prism dimensions
def prism_length : ℕ := 3
def prism_width : ℕ := 1
def prism_height : ℕ := 1

-- Definition of the surface area of the rectangular prism
def surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + w * h + h * l)

-- Theorem statement: The surface area of the resulting prism is 14
theorem prism_surface_area_is_14 : surface_area prism_length prism_width prism_height = 14 :=
  sorry

end prism_surface_area_is_14_l1255_125540


namespace bacteria_elimination_l1255_125517

theorem bacteria_elimination (d N : ℕ) (hN : N = 50 - 6 * (d - 1)) (hCondition : N ≤ 0) : d = 10 :=
by
  -- We can straightforwardly combine the given conditions and derive the required theorem.
  sorry

end bacteria_elimination_l1255_125517


namespace jason_steps_is_8_l1255_125582

-- Definition of the problem conditions
def nancy_steps (jason_steps : ℕ) := 3 * jason_steps -- Nancy steps 3 times as often as Jason

def together_steps (jason_steps nancy_steps : ℕ) := jason_steps + nancy_steps -- Total steps

-- Lean statement of the problem to prove
theorem jason_steps_is_8 (J : ℕ) (h₁ : together_steps J (nancy_steps J) = 32) : J = 8 :=
sorry

end jason_steps_is_8_l1255_125582
