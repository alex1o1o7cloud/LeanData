import Mathlib

namespace incorrect_expressions_l2005_200571

-- Definitions for the conditions
def F : ℝ := sorry   -- F represents a repeating decimal
def X : ℝ := sorry   -- X represents the t digits of F that are non-repeating
def Y : ℝ := sorry   -- Y represents the u digits of F that repeat
def t : ℕ := sorry   -- t is the number of non-repeating digits
def u : ℕ := sorry   -- u is the number of repeating digits

-- Statement that expressions (C) and (D) are incorrect
theorem incorrect_expressions : 
  ¬ (10^(t + 2 * u) * F = X + Y / 10 ^ u) ∧ ¬ (10^t * (10^u - 1) * F = Y * (X - 1)) :=
sorry

end incorrect_expressions_l2005_200571


namespace angle_complement_l2005_200527

-- Conditions: The complement of angle A is 60 degrees
def complement (α : ℝ) : ℝ := 90 - α 

theorem angle_complement (A : ℝ) : complement A = 60 → A = 30 :=
by
  sorry

end angle_complement_l2005_200527


namespace minimum_d_value_l2005_200550

theorem minimum_d_value :
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  distance = 4 * d :=
by
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  sorry

end minimum_d_value_l2005_200550


namespace alpha_beta_inequality_l2005_200526

theorem alpha_beta_inequality (α β : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → x^α * y^β < k * (x + y)) ↔ (0 ≤ α ∧ 0 ≤ β ∧ α + β = 1) :=
by
  sorry

end alpha_beta_inequality_l2005_200526


namespace total_students_shook_hands_l2005_200556

theorem total_students_shook_hands (S3 S2 S1 : ℕ) (h1 : S3 = 200) (h2 : S2 = S3 + 40) (h3 : S1 = 2 * S2) : 
  S1 + S2 + S3 = 920 :=
by
  sorry

end total_students_shook_hands_l2005_200556


namespace unique_positive_a_for_one_solution_l2005_200532

theorem unique_positive_a_for_one_solution :
  ∃ (d : ℝ), d ≠ 0 ∧ (∀ a : ℝ, a > 0 → (∀ x : ℝ, x^2 + (a + 1/a) * x + d = 0 ↔ x^2 + (a + 1/a) * x + d = 0)) ∧ d = 1 := 
by
  sorry

end unique_positive_a_for_one_solution_l2005_200532


namespace least_value_x_y_z_l2005_200501

theorem least_value_x_y_z 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h_eq: 2 * x = 5 * y) 
  (h_eq': 5 * y = 8 * z) : 
  x + y + z = 33 :=
by 
  sorry

end least_value_x_y_z_l2005_200501


namespace inscribed_circle_radius_square_l2005_200525

theorem inscribed_circle_radius_square (ER RF GS SH : ℝ) (r : ℝ) 
  (hER : ER = 23) (hRF : RF = 34) (hGS : GS = 42) (hSH : SH = 28)
  (h_tangent : ∀ t, t = r * r * (70 * t - 87953)) :
  r^2 = 87953 / 70 :=
by
  sorry

end inscribed_circle_radius_square_l2005_200525


namespace sum_is_correct_l2005_200579

noncomputable def calculate_sum : ℚ :=
  (4 / 3) + (13 / 9) + (40 / 27) + (121 / 81) - (8 / 3)

theorem sum_is_correct : calculate_sum = 171 / 81 := 
by {
  sorry
}

end sum_is_correct_l2005_200579


namespace perimeter_rectangles_l2005_200545

theorem perimeter_rectangles (a b : ℕ) (p_rect1 p_rect2 : ℕ) (p_photo : ℕ) (h1 : 2 * (a + b) = p_photo) (h2 : a + b = 10) (h3 : p_rect1 = 40) (h4 : p_rect2 = 44) : 
p_rect1 ≠ p_rect2 -> (p_rect1 = 40 ∧ p_rect2 = 44) := 
by 
  sorry

end perimeter_rectangles_l2005_200545


namespace arithmetic_sequence_a6_l2005_200543

theorem arithmetic_sequence_a6 {a : ℕ → ℤ}
  (h1 : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h2 : a 2 + a 8 = 16)
  (h3 : a 4 = 6) :
  a 6 = 10 :=
by
  sorry

end arithmetic_sequence_a6_l2005_200543


namespace range_of_m_l2005_200519

-- Condition p: The solution set of the inequality x² + mx + 1 < 0 is an empty set
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Condition q: The function y = 4x² + 4(m-1)x + 3 has no extreme value
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 12 * x^2 + 4 * (m - 1) ≥ 0

-- Combined condition: "p or q" is true and "p and q" is false
def combined_condition (m : ℝ) : Prop :=
  (p m ∨ q m) ∧ ¬(p m ∧ q m)

-- The range of values for the real number m
theorem range_of_m (m : ℝ) : combined_condition m → (-2 ≤ m ∧ m < 1) ∨ m > 2 :=
sorry

end range_of_m_l2005_200519


namespace distinct_students_count_l2005_200578

open Set

theorem distinct_students_count 
  (germain_students : ℕ := 15) 
  (newton_students : ℕ := 12) 
  (young_students : ℕ := 9)
  (overlap_students : ℕ := 3) :
  (germain_students + newton_students + young_students - overlap_students) = 33 := 
by
  sorry

end distinct_students_count_l2005_200578


namespace ratio_bob_to_jason_l2005_200552

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := 35

theorem ratio_bob_to_jason : bob_grade / jason_grade = 1 / 2 := by
  sorry

end ratio_bob_to_jason_l2005_200552


namespace hens_count_l2005_200560

theorem hens_count (H C : ℕ) (heads_eq : H + C = 44) (feet_eq : 2 * H + 4 * C = 140) : H = 18 := by
  sorry

end hens_count_l2005_200560


namespace parking_lot_capacity_l2005_200524

-- Definitions based on the conditions
def levels : ℕ := 5
def parkedCars : ℕ := 23
def moreCars : ℕ := 62
def capacityPerLevel : ℕ := parkedCars + moreCars

-- Proof problem statement
theorem parking_lot_capacity : levels * capacityPerLevel = 425 := by
  -- Proof omitted
  sorry

end parking_lot_capacity_l2005_200524


namespace arithmetic_mean_15_23_37_45_l2005_200544

def arithmetic_mean (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem arithmetic_mean_15_23_37_45 :
  arithmetic_mean 15 23 37 45 = 30 :=
by {
  sorry
}

end arithmetic_mean_15_23_37_45_l2005_200544


namespace Karlson_drink_ratio_l2005_200535

noncomputable def conical_glass_volume_ratio (r h : ℝ) : Prop :=
  let V_fuzh := (1 / 3) * Real.pi * r^2 * h
  let V_Mal := (1 / 8) * V_fuzh
  let V_Karlsson := V_fuzh - V_Mal
  (V_Karlsson / V_Mal) = 7

theorem Karlson_drink_ratio (r h : ℝ) : conical_glass_volume_ratio r h := sorry

end Karlson_drink_ratio_l2005_200535


namespace total_fencing_needed_l2005_200549

def width1 : ℕ := 4
def length1 : ℕ := 2 * width1 - 1

def length2 : ℕ := length1 + 3
def width2 : ℕ := width1 - 2

def width3 : ℕ := (width1 + width2) / 2
def length3 : ℚ := (length1 + length2) / 2

def perimeter (w l : ℚ) : ℚ := 2 * (w + l)

def P1 : ℚ := perimeter width1 length1
def P2 : ℚ := perimeter width2 length2
def P3 : ℚ := perimeter width3 length3

def total_fence : ℚ := P1 + P2 + P3

theorem total_fencing_needed : total_fence = 69 := 
  sorry

end total_fencing_needed_l2005_200549


namespace derivative_f_l2005_200564

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by
  sorry

end derivative_f_l2005_200564


namespace max_n_intersection_non_empty_l2005_200514

-- Define the set An
def An (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- State the theorem
theorem max_n_intersection_non_empty : 
  ∃ x, (∀ n, n ≤ 4 → x ∈ An n) ∧ (∀ n, n > 4 → x ∉ An n) :=
by
  sorry

end max_n_intersection_non_empty_l2005_200514


namespace max_xy_l2005_200523

-- Lean statement for the given problem
theorem max_xy (x y : ℝ) (h : x^2 + y^2 = 4) : xy ≤ 2 := sorry

end max_xy_l2005_200523


namespace peter_total_pizza_eaten_l2005_200584

def slices_total : Nat := 16
def peter_slices_eaten_alone : ℚ := 2 / 16
def shared_slice_total : ℚ := 1 / (3 * 16)

theorem peter_total_pizza_eaten : peter_slices_eaten_alone + shared_slice_total = 7 / 48 := by
  sorry

end peter_total_pizza_eaten_l2005_200584


namespace basketball_tournament_l2005_200531

theorem basketball_tournament (x : ℕ) 
  (h1 : ∀ n, ((n * (n - 1)) / 2) = 28 -> n = x) 
  (h2 : (x * (x - 1)) / 2 = 28) : 
  (1 / 2 : ℚ) * x * (x - 1) = 28 :=
by 
  sorry

end basketball_tournament_l2005_200531


namespace rectangular_to_cylindrical_4_neg4_6_l2005_200503

theorem rectangular_to_cylindrical_4_neg4_6 :
  let x := 4
  let y := -4
  let z := 6
  let r := 4 * Real.sqrt 2
  let theta := (7 * Real.pi) / 4
  (r = Real.sqrt (x^2 + y^2)) ∧
  (Real.cos theta = x / r) ∧
  (Real.sin theta = y / r) ∧
  0 ≤ theta ∧ theta < 2 * Real.pi ∧
  z = 6 → 
  (r, theta, z) = (4 * Real.sqrt 2, (7 * Real.pi) / 4, 6) :=
by
  sorry

end rectangular_to_cylindrical_4_neg4_6_l2005_200503


namespace find_m_plus_n_l2005_200518

def probability_no_exact_k_pairs (k n : ℕ) : ℚ :=
  -- A function to calculate the probability
  -- Placeholder definition (details omitted for brevity)
  sorry

theorem find_m_plus_n : ∃ m n : ℕ,
  gcd m n = 1 ∧ 
  (probability_no_exact_k_pairs k n = (97 / 1000) → m + n = 1097) :=
sorry

end find_m_plus_n_l2005_200518


namespace rabbits_ate_27_watermelons_l2005_200513

theorem rabbits_ate_27_watermelons
  (original_watermelons : ℕ)
  (watermelons_left : ℕ)
  (watermelons_eaten : ℕ)
  (h1 : original_watermelons = 35)
  (h2 : watermelons_left = 8)
  (h3 : original_watermelons - watermelons_left = watermelons_eaten) :
  watermelons_eaten = 27 :=
by {
  -- Proof skipped
  sorry
}

end rabbits_ate_27_watermelons_l2005_200513


namespace original_wage_l2005_200587

theorem original_wage (W : ℝ) (h : 1.5 * W = 42) : W = 28 :=
by
  sorry

end original_wage_l2005_200587


namespace greatest_x_value_l2005_200538

theorem greatest_x_value :
  ∃ x : ℝ, (x ≠ 2 ∧ (x^2 - 5 * x - 14) / (x - 2) = 4 / (x + 4)) ∧ x = -2 ∧ 
           ∀ y, (y ≠ 2 ∧ (y^2 - 5 * y - 14) / (y - 2) = 4 / (y + 4)) → y ≤ x :=
by
  sorry

end greatest_x_value_l2005_200538


namespace number_of_good_weeks_l2005_200515

-- Definitions from conditions
def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def tough_weeks : ℕ := 3
def total_money_made : ℕ := 10400
def total_tough_week_sales : ℕ := tough_weeks * tough_week_sales
def total_good_week_sales : ℕ := total_money_made - total_tough_week_sales

-- Question to be proven
theorem number_of_good_weeks (G : ℕ) : 
  (total_good_week_sales = G * good_week_sales) → G = 5 := by
  sorry

end number_of_good_weeks_l2005_200515


namespace polar_to_cartesian_correct_l2005_200575

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_correct : polar_to_cartesian 2 (5 * Real.pi / 6) = (-Real.sqrt 3, 1) :=
by
  sorry -- We are not required to provide the proof here

end polar_to_cartesian_correct_l2005_200575


namespace evaluate_expression_l2005_200508

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a ^ a - a * (a - 2) ^ a) ^ (a + 1) = 14889702426 :=
by
  rw [h]
  sorry

end evaluate_expression_l2005_200508


namespace smallest_prime_with_digit_sum_23_l2005_200558

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l2005_200558


namespace sum_of_roots_eq_neg3_l2005_200554

theorem sum_of_roots_eq_neg3
  (a b c : ℝ)
  (h_eq : 2 * x^2 + 6 * x - 1 = 0)
  (h_a : a = 2)
  (h_b : b = 6) :
  (x1 x2 : ℝ) → x1 + x2 = -b / a :=
by
  sorry

end sum_of_roots_eq_neg3_l2005_200554


namespace factor_difference_of_squares_l2005_200551

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l2005_200551


namespace solution_l2005_200521

noncomputable def problem_statement : Prop :=
  ∃ (A B C D : ℝ) (a b : ℝ) (x : ℝ), 
    (|A - B| = 3) ∧
    (|A - C| = 1) ∧
    (A = Real.pi / 2) ∧  -- This typically signifies angle A is 90 degrees.
    (a > 0) ∧
    (b > 0) ∧
    (a = 1) ∧
    (|A - D| = x) ∧
    (|B - D| = 3 - x) ∧
    (|C - D| = Real.sqrt (x^2 + 1)) ∧
    (Real.sqrt (x^2 + 1) - (3 - x) = 2) ∧
    (|A - D| / |B - D| = 4)

theorem solution : problem_statement :=
sorry

end solution_l2005_200521


namespace floor_plus_self_eq_l2005_200520

theorem floor_plus_self_eq (r : ℝ) (h : ⌊r⌋ + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_self_eq_l2005_200520


namespace medicine_types_count_l2005_200580

theorem medicine_types_count (n : ℕ) (hn : n = 5) : (Nat.choose n 2 = 10) :=
by
  sorry

end medicine_types_count_l2005_200580


namespace increase_80_by_150_percent_l2005_200533

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l2005_200533


namespace domain_of_function_l2005_200548

theorem domain_of_function :
  {x : ℝ | 3 - x > 0 ∧ x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end domain_of_function_l2005_200548


namespace minimum_type_A_tickets_value_of_m_l2005_200596

theorem minimum_type_A_tickets (x : ℕ) (h1 : x + (500 - x) = 500) (h2 : x ≥ 3 * (500 - x)) : x = 375 := by
  sorry

theorem value_of_m (m : ℕ) (h : 500 * (1 + (m + 10) / 100) * (m + 20) = 56000) : m = 50 := by
  sorry

end minimum_type_A_tickets_value_of_m_l2005_200596


namespace value_of_a6_l2005_200505

noncomputable def Sn (n : ℕ) : ℕ := n * 2^(n + 1)
noncomputable def an (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem value_of_a6 : an 6 = 448 := by
  sorry

end value_of_a6_l2005_200505


namespace isosceles_triangle_perimeter_l2005_200590

theorem isosceles_triangle_perimeter 
    (a b : ℕ) (h_iso : a = 3 ∨ a = 5) (h_other : b = 3 ∨ b = 5) 
    (h_distinct : a ≠ b) : 
    ∃ p : ℕ, p = (3 + 3 + 5) ∨ p = (5 + 5 + 3) :=
by
  sorry

end isosceles_triangle_perimeter_l2005_200590


namespace shopkeeper_loss_percentage_l2005_200565

theorem shopkeeper_loss_percentage
    (CP : ℝ) (profit_rate loss_percent : ℝ) 
    (SP : ℝ := CP * (1 + profit_rate)) 
    (value_after_theft : ℝ := SP * (1 - loss_percent)) 
    (goods_loss : ℝ := 100 * (1 - (value_after_theft / CP))) :
    goods_loss = 51.6 :=
by
    sorry

end shopkeeper_loss_percentage_l2005_200565


namespace contradiction_proof_l2005_200572

theorem contradiction_proof :
  ∀ (a b c d : ℝ),
    a + b = 1 →
    c + d = 1 →
    ac + bd > 1 →
    (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) :=
by
  sorry

end contradiction_proof_l2005_200572


namespace percentage_donated_to_orphan_house_l2005_200509

-- Given conditions as definitions in Lean 4
def income : ℝ := 400000
def children_percentage : ℝ := 0.2
def children_count : ℕ := 3
def wife_percentage : ℝ := 0.25
def remaining_after_donation : ℝ := 40000

-- Define the problem as a theorem
theorem percentage_donated_to_orphan_house :
  (children_count * children_percentage + wife_percentage) * income = 0.85 * income →
  (income - 0.85 * income = 60000) →
  remaining_after_donation = 40000 →
  (100 * (60000 - remaining_after_donation) / 60000) = 33.33 := 
by
  intros h1 h2 h3 
  sorry

end percentage_donated_to_orphan_house_l2005_200509


namespace min_max_expression_l2005_200557

theorem min_max_expression (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 19) (h2 : b^2 + b * c + c^2 = 19) :
  ∃ (min_val max_val : ℝ), 
    min_val = 0 ∧ max_val = 57 ∧ 
    (∀ x, x = c^2 + c * a + a^2 → min_val ≤ x ∧ x ≤ max_val) :=
by sorry

end min_max_expression_l2005_200557


namespace christopher_more_money_l2005_200588

-- Define the conditions provided in the problem

def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64
def quarter_value : ℝ := 0.25

-- Define the question as a theorem

theorem christopher_more_money : (christopher_quarters - karen_quarters) * quarter_value = 8 :=
by sorry

end christopher_more_money_l2005_200588


namespace curve_not_parabola_l2005_200567

theorem curve_not_parabola (k : ℝ) : ¬(∃ a b c : ℝ, a ≠ 0 ∧ x^2 + ky^2 = a*x^2 + b*y + c) :=
sorry

end curve_not_parabola_l2005_200567


namespace books_arrangement_count_l2005_200534

noncomputable def arrangement_of_books : ℕ :=
  let total_books := 5
  let identical_books := 2
  Nat.factorial total_books / Nat.factorial identical_books

theorem books_arrangement_count : arrangement_of_books = 60 := by
  sorry

end books_arrangement_count_l2005_200534


namespace boys_in_other_communities_l2005_200594

def percentage_of_other_communities (p_M p_H p_S : ℕ) : ℕ :=
  100 - (p_M + p_H + p_S)

def number_of_boys_other_communities (total_boys : ℕ) (percentage_other : ℕ) : ℕ :=
  (percentage_other * total_boys) / 100

theorem boys_in_other_communities (N p_M p_H p_S : ℕ) (hN : N = 650) (hpM : p_M = 44) (hpH : p_H = 28) (hpS : p_S = 10) :
  number_of_boys_other_communities N (percentage_of_other_communities p_M p_H p_S) = 117 :=
by
  -- Steps to prove the theorem would go here
  sorry

end boys_in_other_communities_l2005_200594


namespace expand_expression_l2005_200553

variable (x y z : ℝ)

theorem expand_expression :
  (x + 8) * (3 * y + 12) * (2 * z + 4) =
  6 * x * y * z + 12 * x * z + 24 * y * z + 12 * x * y + 48 * x + 96 * y + 96 * z + 384 := 
  sorry

end expand_expression_l2005_200553


namespace subcommittee_count_l2005_200540

theorem subcommittee_count :
  (Nat.choose 10 4) * (Nat.choose 8 3) = 11760 :=
by
  sorry

end subcommittee_count_l2005_200540


namespace fraction_of_ripe_oranges_eaten_l2005_200577

theorem fraction_of_ripe_oranges_eaten :
  ∀ (total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges : ℕ),
    total_oranges = 96 →
    ripe_oranges = total_oranges / 2 →
    unripe_oranges = total_oranges / 2 →
    eaten_unripe_oranges = unripe_oranges / 8 →
    uneaten_oranges = 78 →
    eaten_ripe_oranges = (total_oranges - uneaten_oranges) - eaten_unripe_oranges →
    (eaten_ripe_oranges : ℚ) / ripe_oranges = 1 / 4 :=
by
  intros total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges
  intros h_total h_ripe h_unripe h_eaten_unripe h_uneaten h_eaten_ripe
  sorry

end fraction_of_ripe_oranges_eaten_l2005_200577


namespace smallest_m_plus_n_l2005_200581

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > n) (h2 : n ≥ 1) 
(h3 : 1000 ∣ 1978^m - 1978^n) : m + n = 106 :=
sorry

end smallest_m_plus_n_l2005_200581


namespace m_value_for_perfect_square_l2005_200536

theorem m_value_for_perfect_square (m : ℤ) (x y : ℤ) :
  (∃ k : ℤ, 4 * x^2 - m * x * y + 9 * y^2 = k^2) → m = 12 ∨ m = -12 :=
by
  sorry

end m_value_for_perfect_square_l2005_200536


namespace min_value_expression_71_l2005_200528

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  4 * x + 9 * y + 1 / (x - 4) + 1 / (y - 5)

theorem min_value_expression_71 (x y : ℝ) (hx : x > 4) (hy : y > 5) : 
  min_value_expression x y ≥ 71 :=
by
  sorry

end min_value_expression_71_l2005_200528


namespace expression_evaluation_l2005_200574

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end expression_evaluation_l2005_200574


namespace problem_l2005_200592

theorem problem (a b c d : ℝ) (h₁ : a + b = 0) (h₂ : c * d = 1) : 
  (5 * a + 5 * b - 7 * c * d) / (-(c * d) ^ 3) = 7 := 
by
  sorry

end problem_l2005_200592


namespace fixed_point_l2005_200595

theorem fixed_point (m : ℝ) : (2 * m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by {
  sorry
}

end fixed_point_l2005_200595


namespace correct_divisor_l2005_200547

-- Definitions of variables and conditions
variables (X D : ℕ)

-- Stating the theorem
theorem correct_divisor (h1 : X = 49 * 12) (h2 : X = 28 * D) : D = 21 :=
by
  sorry

end correct_divisor_l2005_200547


namespace carrie_profit_l2005_200529

def hours_per_day : ℕ := 2
def days_worked : ℕ := 4
def hourly_rate : ℕ := 22
def cost_of_supplies : ℕ := 54
def total_hours_worked : ℕ := hours_per_day * days_worked
def total_payment : ℕ := hourly_rate * total_hours_worked
def profit : ℕ := total_payment - cost_of_supplies

theorem carrie_profit : profit = 122 := by
  sorry

end carrie_profit_l2005_200529


namespace f_g_g_f_l2005_200539

noncomputable def f (x: ℝ) := 1 - 2 * x
noncomputable def g (x: ℝ) := x^2 + 3

theorem f_g (x : ℝ) : f (g x) = -2 * x^2 - 5 :=
by
  sorry

theorem g_f (x : ℝ) : g (f x) = 4 * x^2 - 4 * x + 4 :=
by
  sorry

end f_g_g_f_l2005_200539


namespace find_ellipse_l2005_200507

-- Define the ellipse and conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the focus points
def focus (a b c : ℝ) : Prop :=
  c^2 = a^2 - b^2

-- Define the range condition
def range_condition (a b c : ℝ) : Prop :=
  let min_val := b^2 - c^2;
  let max_val := a^2 - c^2;
  min_val = -3 ∧ max_val = 3

-- Prove the equation of the ellipse
theorem find_ellipse (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (ellipse a b a_pos b_pos ∧ focus a b c ∧ range_condition a b c) →
  (a^2 = 9 ∧ b^2 = 3) :=
by
  sorry

end find_ellipse_l2005_200507


namespace total_pictures_480_l2005_200559

noncomputable def total_pictures (pictures_per_album : ℕ) (num_albums : ℕ) : ℕ :=
  pictures_per_album * num_albums

theorem total_pictures_480 : total_pictures 20 24 = 480 :=
  by
    sorry

end total_pictures_480_l2005_200559


namespace final_amount_simple_interest_l2005_200537

theorem final_amount_simple_interest (P R T : ℕ) (hP : P = 12500) (hR : R = 6) (hT : T = 4) : 
  P + (P * R * T) / 100 = 13250 :=
by
  rw [hP, hR, hT]
  norm_num
  sorry

end final_amount_simple_interest_l2005_200537


namespace quadratic_max_m_l2005_200563

theorem quadratic_max_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (m * x^2 - 2 * m * x + 2) ≤ 4) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ (m * x^2 - 2 * m * x + 2) = 4) ∧ 
  m ≠ 0 → 
  (m = 2 / 3 ∨ m = -2) := 
by
  sorry

end quadratic_max_m_l2005_200563


namespace median_length_of_right_triangle_l2005_200585

theorem median_length_of_right_triangle (DE EF : ℝ) (hDE : DE = 5) (hEF : EF = 12) :
  let DF := Real.sqrt (DE^2 + EF^2)
  let N := (EF / 2)
  let DN := DF / 2
  DN = 6.5 :=
by
  sorry

end median_length_of_right_triangle_l2005_200585


namespace find_common_difference_l2005_200569

-- Define the arithmetic series sum formula
def arithmetic_series_sum (a₁ : ℕ) (d : ℚ) (n : ℕ) :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

-- Define the first day's production, total days, and total fabric
def first_day := 5
def total_days := 30
def total_fabric := 390

-- The proof statement
theorem find_common_difference : 
  ∃ d : ℚ, arithmetic_series_sum first_day d total_days = total_fabric ∧ d = 16 / 29 :=
by
  sorry

end find_common_difference_l2005_200569


namespace average_speed_is_five_l2005_200589

-- Define the speeds for each segment
def swimming_speed : ℝ := 2 -- km/h
def biking_speed : ℝ := 15 -- km/h
def running_speed : ℝ := 9 -- km/h
def kayaking_speed : ℝ := 6 -- km/h

-- Define the problem to prove the average speed
theorem average_speed_is_five :
  let segments := [swimming_speed, biking_speed, running_speed, kayaking_speed]
  let harmonic_mean (speeds : List ℝ) : ℝ :=
    let n := speeds.length
    n / (speeds.foldl (fun acc s => acc + 1 / s) 0)
  harmonic_mean segments = 5 := by
  sorry

end average_speed_is_five_l2005_200589


namespace carrie_bought_tshirts_l2005_200516

variable (cost_per_tshirt : ℝ) (total_spent : ℝ)

theorem carrie_bought_tshirts (h1 : cost_per_tshirt = 9.95) (h2 : total_spent = 248) :
  ⌊total_spent / cost_per_tshirt⌋ = 24 :=
by
  sorry

end carrie_bought_tshirts_l2005_200516


namespace sub_of_neg_l2005_200541

theorem sub_of_neg : -3 - 2 = -5 :=
by 
  sorry

end sub_of_neg_l2005_200541


namespace area_of_ABCD_l2005_200566

noncomputable def quadrilateral_area (AB BC AD DC : ℝ) : ℝ :=
  let area_ABC := 1 / 2 * AB * BC
  let area_ADC := 1 / 2 * AD * DC
  area_ABC + area_ADC

theorem area_of_ABCD {AB BC AD DC AC : ℝ}
  (h1 : AC = 5)
  (h2 : AB * AB + BC * BC = 25)
  (h3 : AD * AD + DC * DC = 25)
  (h4 : AB ≠ AD)
  (h5 : BC ≠ DC) :
  quadrilateral_area AB BC AD DC = 12 :=
sorry

end area_of_ABCD_l2005_200566


namespace insect_population_calculations_l2005_200582

theorem insect_population_calculations :
  (let ants_1 := 100
   let ants_2 := ants_1 - 20 * ants_1 / 100
   let ants_3 := ants_2 - 25 * ants_2 / 100
   let bees_1 := 150
   let bees_2 := bees_1 - 30 * bees_1 / 100
   let termites_1 := 200
   let termites_2 := termites_1 - 10 * termites_1 / 100
   ants_3 = 60 ∧ bees_2 = 105 ∧ termites_2 = 180) :=
by
  sorry

end insect_population_calculations_l2005_200582


namespace probability_A_fires_proof_l2005_200561

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l2005_200561


namespace value_of_F_l2005_200568

   variables (B G P Q F : ℕ)

   -- Define the main hypothesis stating that the total lengths of the books are equal.
   def fill_shelf := 
     (∃ d a : ℕ, d = B * a + 2 * G * a ∧ d = P * a + 2 * Q * a ∧ d = F * a)

   -- Prove that F equals B + 2G and P + 2Q under the hypothesis.
   theorem value_of_F (h : fill_shelf B G P Q F) : F = B + 2 * G ∧ F = P + 2 * Q :=
   sorry
   
end value_of_F_l2005_200568


namespace count_triples_satisfying_conditions_l2005_200570

theorem count_triples_satisfying_conditions :
  (∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ ab + bc = 72 ∧ ac + bc = 35) → 
  ∃! t : (ℕ × ℕ × ℕ), 0 < t.1 ∧ 0 < t.2.1 ∧ 0 < t.2.2 ∧ 
                     t.1 * t.2.1 + t.2.1 * t.2.2 = 72 ∧ 
                     t.1 * t.2.2 + t.2.1 * t.2.2 = 35 :=
by sorry

end count_triples_satisfying_conditions_l2005_200570


namespace max_single_player_salary_is_426000_l2005_200542

noncomputable def max_single_player_salary (total_salary_cap : ℤ) (min_salary : ℤ) (num_players : ℤ) : ℤ :=
  total_salary_cap - (num_players - 1) * min_salary

theorem max_single_player_salary_is_426000 :
  ∃ y, max_single_player_salary 800000 17000 23 = y ∧ y = 426000 :=
by
  sorry

end max_single_player_salary_is_426000_l2005_200542


namespace votes_lost_by_l2005_200586

theorem votes_lost_by (total_votes : ℕ) (candidate_percentage : ℕ) : total_votes = 20000 → candidate_percentage = 10 → 
  (total_votes * candidate_percentage / 100 - total_votes * (100 - candidate_percentage) / 100 = 16000) :=
by
  intros h_total_votes h_candidate_percentage
  have vote_candidate := total_votes * candidate_percentage / 100
  have vote_rival := total_votes * (100 - candidate_percentage) / 100
  have votes_diff := vote_rival - vote_candidate
  rw [h_total_votes, h_candidate_percentage] at *
  sorry

end votes_lost_by_l2005_200586


namespace impossible_to_achieve_12_percent_return_l2005_200598

-- Define the stock parameters and their individual returns
def stock_A_price : ℝ := 52
def stock_A_dividend_rate : ℝ := 0.09
def stock_A_transaction_fee_rate : ℝ := 0.02

def stock_B_price : ℝ := 80
def stock_B_dividend_rate : ℝ := 0.07
def stock_B_transaction_fee_rate : ℝ := 0.015

def stock_C_price : ℝ := 40
def stock_C_dividend_rate : ℝ := 0.10
def stock_C_transaction_fee_rate : ℝ := 0.01

def tax_rate : ℝ := 0.10
def desired_return : ℝ := 0.12

theorem impossible_to_achieve_12_percent_return :
  false :=
sorry

end impossible_to_achieve_12_percent_return_l2005_200598


namespace drawings_per_neighbor_l2005_200583

theorem drawings_per_neighbor (n_neighbors animals : ℕ) (h1 : n_neighbors = 6) (h2 : animals = 54) : animals / n_neighbors = 9 :=
by
  sorry

end drawings_per_neighbor_l2005_200583


namespace exponentiation_rule_l2005_200555

variable {a b : ℝ}

theorem exponentiation_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end exponentiation_rule_l2005_200555


namespace determine_x_l2005_200546

theorem determine_x (x : ℝ) (hx : 0 < x) (h : x * ⌊x⌋ = 72) : x = 9 :=
sorry

end determine_x_l2005_200546


namespace width_decrease_percentage_l2005_200504

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l2005_200504


namespace parking_cost_per_hour_l2005_200597

theorem parking_cost_per_hour (avg_cost : ℝ) (total_initial_cost : ℝ) (hours_excessive : ℝ) (total_hours : ℝ) (cost_first_2_hours : ℝ)
  (h1 : cost_first_2_hours = 9.00) 
  (h2 : avg_cost = 2.361111111111111)
  (h3 : total_hours = 9) 
  (h4 : hours_excessive = 7):
  (total_initial_cost / total_hours = avg_cost) -> 
  (total_initial_cost = cost_first_2_hours + hours_excessive * x) -> 
  x = 1.75 := 
by
  intros h5 h6
  sorry

end parking_cost_per_hour_l2005_200597


namespace divide_400_l2005_200562

theorem divide_400 (a b c d : ℕ) (h1 : a + b + c + d = 400) 
  (h2 : a + 1 = b - 2) (h3 : a + 1 = 3 * c) (h4 : a + 1 = d / 4) 
  : a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 :=
sorry

end divide_400_l2005_200562


namespace sum_odd_even_50_l2005_200573

def sum_first_n_odd (n : ℕ) : ℕ := n * n

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

theorem sum_odd_even_50 : 
  sum_first_n_odd 50 + sum_first_n_even 50 = 5050 := by
  sorry

end sum_odd_even_50_l2005_200573


namespace women_attended_l2005_200530

theorem women_attended :
  (15 * 4) / 3 = 20 :=
by
  sorry

end women_attended_l2005_200530


namespace john_total_animals_is_114_l2005_200500

  -- Define the entities and their relationships based on the conditions
  def num_snakes : ℕ := 15
  def num_monkeys : ℕ := 2 * num_snakes
  def num_lions : ℕ := num_monkeys - 5
  def num_pandas : ℕ := num_lions + 8
  def num_dogs : ℕ := num_pandas / 3

  -- Define the total number of animals
  def total_animals : ℕ := num_snakes + num_monkeys + num_lions + num_pandas + num_dogs

  -- Prove that the total number of animals is 114
  theorem john_total_animals_is_114 : total_animals = 114 := by
    sorry
  
end john_total_animals_is_114_l2005_200500


namespace crystal_run_final_segment_length_l2005_200522

theorem crystal_run_final_segment_length :
  let north_distance := 2
  let southeast_leg := 1 / Real.sqrt 2
  let southeast_movement_north := -southeast_leg
  let southeast_movement_east := southeast_leg
  let northeast_leg := 2 / Real.sqrt 2
  let northeast_movement_north := northeast_leg
  let northeast_movement_east := northeast_leg
  let total_north_movement := north_distance + northeast_movement_north + southeast_movement_north
  let total_east_movement := southeast_movement_east + northeast_movement_east
  total_north_movement = 2.5 ∧ 
  total_east_movement = 3 * Real.sqrt 2 / 2 ∧ 
  Real.sqrt (total_north_movement^2 + total_east_movement^2) = Real.sqrt 10.75 :=
by
  sorry

end crystal_run_final_segment_length_l2005_200522


namespace inappropriate_expression_is_D_l2005_200591

-- Definitions of each expression as constants
def expr_A : String := "Recently, I have had the honor to read your masterpiece, and I felt enlightened."
def expr_B : String := "Your visit has brought glory to my humble abode."
def expr_C : String := "It's the first time you honor my place with a visit, and I apologize for any lack of hospitality."
def expr_D : String := "My mother has been slightly unwell recently, I hope you won't bother her."

-- Definition of the problem context
def is_inappropriate (expr : String) : Prop := 
  expr = expr_D

-- The theorem statement
theorem inappropriate_expression_is_D : is_inappropriate expr_D := 
by
  sorry

end inappropriate_expression_is_D_l2005_200591


namespace find_a_l2005_200593

theorem find_a (a : ℝ) (k_l : ℝ) (h1 : k_l = -1)
  (h2 : a ≠ 3) 
  (h3 : (2 - (-1)) / (3 - a) * k_l = -1) : a = 6 :=
by
  sorry

end find_a_l2005_200593


namespace terminal_side_in_first_quadrant_l2005_200599

noncomputable def theta := -5

def in_first_quadrant (θ : ℝ) : Prop :=
  by sorry

theorem terminal_side_in_first_quadrant : in_first_quadrant theta := 
  by sorry

end terminal_side_in_first_quadrant_l2005_200599


namespace intersect_sets_l2005_200502

open Set

noncomputable def P : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}

theorem intersect_sets (U : Set ℝ) (P : Set ℝ) (Q : Set ℝ) :
  U = univ → P = {x : ℝ | x^2 - 2 * x ≤ 0} → Q = {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x} →
  P ∩ Q = Icc (0 : ℝ) (2 : ℝ) :=
by
  intros
  sorry

end intersect_sets_l2005_200502


namespace tangent_line_equation_l2005_200512

theorem tangent_line_equation (x y : ℝ) (h : y = x^3 + 1) (t : x = -1) :
  3*x - y + 3 = 0 :=
sorry

end tangent_line_equation_l2005_200512


namespace rate_per_square_meter_is_3_l2005_200576

def floor_painting_rate 
  (length : ℝ) 
  (total_cost : ℝ)
  (length_more_than_breadth_by_percentage : ℝ)
  (expected_rate : ℝ) : Prop :=
  ∃ (breadth : ℝ) (rate : ℝ),
    length = (1 + length_more_than_breadth_by_percentage / 100) * breadth ∧
    total_cost = length * breadth * rate ∧
    rate = expected_rate

-- Given conditions
theorem rate_per_square_meter_is_3 :
  floor_painting_rate 15.491933384829668 240 200 3 :=
by
  sorry

end rate_per_square_meter_is_3_l2005_200576


namespace percent_y_of_x_l2005_200506

-- Definitions and assumptions based on the problem conditions
variables (x y : ℝ)
-- Given: 20% of (x - y) = 14% of (x + y)
axiom h : 0.20 * (x - y) = 0.14 * (x + y)

-- Prove that y is 0.1765 (or 17.65%) of x
theorem percent_y_of_x (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) : 
  y = 0.1765 * x :=
sorry

end percent_y_of_x_l2005_200506


namespace sugar_total_more_than_two_l2005_200517

noncomputable def x (p q : ℝ) : ℝ :=
p / q

noncomputable def y (p q : ℝ) : ℝ :=
q / p

theorem sugar_total_more_than_two (p q : ℝ) (hpq : p ≠ q) :
  x p q + y p q > 2 :=
by sorry

end sugar_total_more_than_two_l2005_200517


namespace apple_slices_per_group_l2005_200511

-- defining the conditions
variables (a g : ℕ)

-- 1. Equal number of apple slices and grapes in groups
def equal_group (a g : ℕ) : Prop := a = g

-- 2. Grapes packed in groups of 9
def grapes_groups_of_9 (g : ℕ) : Prop := ∃ k : ℕ, g = 9 * k

-- 3. Smallest number of grapes is 18
def smallest_grapes (g : ℕ) : Prop := g = 18

-- theorem stating that the number of apple slices per group is 9
theorem apple_slices_per_group : equal_group a g ∧ grapes_groups_of_9 g ∧ smallest_grapes g → a = 9 := by
  sorry

end apple_slices_per_group_l2005_200511


namespace correct_transformation_l2005_200510

theorem correct_transformation (a b : ℝ) (h : a ≠ 0) : 
  (a^2 / (a * b) = a / b) :=
by sorry

end correct_transformation_l2005_200510
