import Mathlib

namespace find_m_l328_32893

theorem find_m (m : ℝ) : (∀ x : ℝ, x^2 - 4 * x + m = 0) → m = 4 :=
by
  intro h
  sorry

end find_m_l328_32893


namespace solve_otimes_n_1_solve_otimes_2005_2_l328_32828

-- Define the operation ⊗
noncomputable def otimes (x y : ℕ) : ℕ :=
sorry -- the definition is abstracted away as per conditions

-- Conditions from the problem
axiom otimes_cond_1 : ∀ x : ℕ, otimes x 0 = x + 1
axiom otimes_cond_2 : ∀ x : ℕ, otimes 0 (x + 1) = otimes 1 x
axiom otimes_cond_3 : ∀ x y : ℕ, otimes (x + 1) (y + 1) = otimes (otimes x (y + 1)) y

-- Prove the required equalities
theorem solve_otimes_n_1 (n : ℕ) : otimes n 1 = n + 2 :=
sorry

theorem solve_otimes_2005_2 : otimes 2005 2 = 4013 :=
sorry

end solve_otimes_n_1_solve_otimes_2005_2_l328_32828


namespace inequality_not_necessarily_hold_l328_32812

theorem inequality_not_necessarily_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) :=
sorry

end inequality_not_necessarily_hold_l328_32812


namespace extra_bananas_each_child_gets_l328_32836

-- Define the total number of students and the number of absent students
def total_students : ℕ := 260
def absent_students : ℕ := 130

-- Define the total number of bananas
variable (B : ℕ)

-- The proof statement
theorem extra_bananas_each_child_gets :
  ∀ B : ℕ, (B / (total_students - absent_students)) = (B / total_students) + (B / total_students) :=
by
  intro B
  sorry

end extra_bananas_each_child_gets_l328_32836


namespace age_proof_l328_32815

theorem age_proof (y d : ℕ)
  (h1 : y = 4 * d)
  (h2 : y - 7 = 11 * (d - 7)) :
  y = 48 ∧ d = 12 :=
by
  -- The proof is omitted
  sorry

end age_proof_l328_32815


namespace value_of_x_l328_32862

theorem value_of_x (w : ℝ) (hw : w = 90) (z : ℝ) (hz : z = 2 / 3 * w) (y : ℝ) (hy : y = 1 / 4 * z) (x : ℝ) (hx : x = 1 / 2 * y) : x = 7.5 :=
by
  -- Proof skipped; conclusion derived from conditions
  sorry

end value_of_x_l328_32862


namespace surface_area_reduction_of_spliced_cuboid_l328_32895

theorem surface_area_reduction_of_spliced_cuboid 
  (initial_faces : ℕ := 12)
  (faces_lost : ℕ := 2)
  (percentage_reduction : ℝ := (2 / 12) * 100) :
  percentage_reduction = 16.7 :=
by
  sorry

end surface_area_reduction_of_spliced_cuboid_l328_32895


namespace gcd_problem_l328_32817

-- Define the variables according to the conditions
def m : ℤ := 123^2 + 235^2 + 347^2
def n : ℤ := 122^2 + 234^2 + 348^2

-- Lean statement for the proof problem
theorem gcd_problem : Int.gcd m n = 1 := sorry

end gcd_problem_l328_32817


namespace domain_of_f_l328_32857

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (4 * x - 3))

theorem domain_of_f :
  {x : ℝ | 4 * x - 3 > 0 ∧ Real.log (4 * x - 3) ≠ 0} = 
  {x : ℝ | x ∈ Set.Ioo (3 / 4) 1 ∪ Set.Ioi 1} :=
by
  sorry

end domain_of_f_l328_32857


namespace distance_to_x_axis_l328_32858

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-3, -2)) : |P.2| = 2 := 
by sorry

end distance_to_x_axis_l328_32858


namespace f_increasing_maximum_b_condition_approximate_ln2_l328_32816

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x ≤ f y := 
sorry

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (2 * x) - 4 * b * f x

theorem maximum_b_condition (x : ℝ) (H : 0 < x): ∃ b, g x b > 0 ∧ b ≤ 2 := 
sorry

theorem approximate_ln2 :
  0.692 ≤ Real.log 2 ∧ Real.log 2 ≤ 0.694 :=
sorry

end f_increasing_maximum_b_condition_approximate_ln2_l328_32816


namespace intersection_of_sets_l328_32885

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 ≥ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x < 2

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by
  sorry

end intersection_of_sets_l328_32885


namespace vip_seat_cost_is_65_l328_32845

noncomputable def cost_of_VIP_seat (G V_T V : ℕ) (cost : ℕ) : Prop :=
  G + V_T = 320 ∧
  (15 * G + V * V_T = cost) ∧
  V_T = G - 212 → V = 65

theorem vip_seat_cost_is_65 :
  ∃ (G V_T V : ℕ), cost_of_VIP_seat G V_T V 7500 :=
  sorry

end vip_seat_cost_is_65_l328_32845


namespace units_digit_m_sq_plus_2_m_l328_32832

def m := 2017^2 + 2^2017

theorem units_digit_m_sq_plus_2_m (m := 2017^2 + 2^2017) : (m^2 + 2^m) % 10 = 3 := 
by
  sorry

end units_digit_m_sq_plus_2_m_l328_32832


namespace range_of_m_l328_32887

open Real

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - m * x + m > 0) ↔ (0 < m ∧ m < 4) :=
by
  sorry

end range_of_m_l328_32887


namespace correct_masks_l328_32800

def elephant_mask := 6
def mouse_mask := 4
def pig_mask := 8
def panda_mask := 1

theorem correct_masks :
  (elephant_mask = 6) ∧
  (mouse_mask = 4) ∧
  (pig_mask = 8) ∧
  (panda_mask = 1) := 
by
  sorry

end correct_masks_l328_32800


namespace area_of_region_eq_24π_l328_32840

theorem area_of_region_eq_24π :
  (∃ R, R > 0 ∧ ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 8 * x + 18 * y + 73 = R ^ 2) →
  ∃ π : ℝ, π > 0 ∧ area = 24 * π :=
by
  sorry

end area_of_region_eq_24π_l328_32840


namespace minimize_squared_distances_l328_32879

variable {P : ℝ}

/-- Points A, B, C, D, E are collinear with distances AB = 3, BC = 3, CD = 5, and DE = 7 -/
def collinear_points : Prop :=
  ∀ (A B C D E : ℝ), B = A + 3 ∧ C = B + 3 ∧ D = C + 5 ∧ E = D + 7

/-- Define the squared distance function -/
def squared_distances (P A B C D E : ℝ) : ℝ :=
  (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2

/-- Statement of the proof problem -/
theorem minimize_squared_distances :
  collinear_points →
  ∀ (A B C D E P : ℝ), 
    squared_distances P A B C D E ≥ 181.2 :=
by
  sorry

end minimize_squared_distances_l328_32879


namespace expression_value_l328_32861

theorem expression_value (x y z : ℕ) (hx : x = 2) (hy : y = 5) (hz : z = 3) :
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  rw [hx, hy, hz]
  sorry

end expression_value_l328_32861


namespace min_value_A_mul_abs_x1_minus_x2_l328_32848

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2017 * x + Real.pi / 6) + Real.cos (2017 * x - Real.pi / 3)

theorem min_value_A_mul_abs_x1_minus_x2 :
  ∃ x1 x2 : ℝ, (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) →
  2 * |x1 - x2| = (2 * Real.pi) / 2017 :=
sorry

end min_value_A_mul_abs_x1_minus_x2_l328_32848


namespace good_students_count_l328_32889

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end good_students_count_l328_32889


namespace remainder_17_pow_2023_mod_28_l328_32809

theorem remainder_17_pow_2023_mod_28 :
  17^2023 % 28 = 17 := 
by sorry

end remainder_17_pow_2023_mod_28_l328_32809


namespace solve_exponents_l328_32859

theorem solve_exponents (x y z : ℕ) (hx : x < y) (hy : y < z) 
  (h : 3^x + 3^y + 3^z = 179415) : x = 4 ∧ y = 7 ∧ z = 11 :=
by sorry

end solve_exponents_l328_32859


namespace q_sufficient_not_necessary_p_l328_32837

theorem q_sufficient_not_necessary_p (x : ℝ) (p : Prop) (q : Prop) :
  (p ↔ |x| < 2) →
  (q ↔ x^2 - x - 2 < 0) →
  (q → p) ∧ (p ∧ ¬q) :=
by
  sorry

end q_sufficient_not_necessary_p_l328_32837


namespace driving_speed_l328_32821

variable (total_distance : ℝ) (break_time : ℝ) (total_trip_time : ℝ)

theorem driving_speed (h1 : total_distance = 480)
                      (h2 : break_time = 1)
                      (h3 : total_trip_time = 9) : 
  (total_distance / (total_trip_time - break_time)) = 60 :=
by
  sorry

end driving_speed_l328_32821


namespace set_operation_result_l328_32818

def M : Set ℕ := {2, 3}

def bin_op (A : Set ℕ) : Set ℕ :=
  {x | ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem set_operation_result : bin_op M = {4, 5, 6} :=
by
  sorry

end set_operation_result_l328_32818


namespace circle_equation_l328_32829

theorem circle_equation :
  ∃ (h k r : ℝ), 
    (∀ (x y : ℝ), (x, y) = (-6, 2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ (∀ (x y : ℝ), (x, y) = (2, -2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ r = 5
    ∧ h - k = -1
    ∧ (x + 3)^2 + (y + 2)^2 = 25 :=
by
  sorry

end circle_equation_l328_32829


namespace inequality_proof_l328_32847

theorem inequality_proof (x : ℝ) (h₁ : 3/2 ≤ x) (h₂ : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 :=
sorry

end inequality_proof_l328_32847


namespace divisors_remainders_l328_32823

theorem divisors_remainders (n : ℕ) (h : ∀ k : ℕ, 1001 ≤ k ∧ k ≤ 2012 → ∃ d : ℕ, d ∣ n ∧ d % 2013 = k) :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 2012 → ∃ d : ℕ, d ∣ n^2 ∧ d % 2013 = m :=
by sorry

end divisors_remainders_l328_32823


namespace sebastian_older_than_jeremy_by_4_l328_32841

def J : ℕ := 40
def So : ℕ := 60 - 3
def sum_ages_in_3_years (S : ℕ) : Prop := (J + 3) + (S + 3) + (So + 3) = 150

theorem sebastian_older_than_jeremy_by_4 (S : ℕ) (h : sum_ages_in_3_years S) : S - J = 4 := by
  -- proof will be filled in
  sorry

end sebastian_older_than_jeremy_by_4_l328_32841


namespace only_composite_positive_integer_with_divisors_form_l328_32874

theorem only_composite_positive_integer_with_divisors_form (n : ℕ) (composite : ¬Nat.Prime n ∧ 1 < n)
  (H : ∀ d ∈ Nat.divisors n, ∃ (a r : ℕ), a ≥ 0 ∧ r ≥ 2 ∧ d = a^r + 1) : n = 10 :=
by
  sorry

end only_composite_positive_integer_with_divisors_form_l328_32874


namespace cos_neg_60_equals_half_l328_32802

  theorem cos_neg_60_equals_half : Real.cos (-60 * Real.pi / 180) = 1 / 2 :=
  by
    sorry
  
end cos_neg_60_equals_half_l328_32802


namespace minimum_value_l328_32864

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2 * y = 3) :
  (1 / x + 1 / y) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end minimum_value_l328_32864


namespace arithmetic_geometric_progression_l328_32873

theorem arithmetic_geometric_progression (a b : ℝ) :
  (b = 2 - a) ∧ (b = 1 / a ∨ b = -1 / a) →
  (a = 1 ∧ b = 1) ∨
  (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
  (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
by
  sorry

end arithmetic_geometric_progression_l328_32873


namespace carol_used_tissue_paper_l328_32811

theorem carol_used_tissue_paper (initial_pieces : ℕ) (remaining_pieces : ℕ) (usage: ℕ)
  (h1 : initial_pieces = 97)
  (h2 : remaining_pieces = 93)
  (h3: usage = initial_pieces - remaining_pieces) : 
  usage = 4 :=
by
  -- We only need to set up the problem; proof can be provided later.
  sorry

end carol_used_tissue_paper_l328_32811


namespace algebraic_expression_value_l328_32870

theorem algebraic_expression_value (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 :=
by
  sorry

end algebraic_expression_value_l328_32870


namespace smallest_rectangle_area_contains_L_shape_l328_32883

-- Condition: Side length of each square
def side_length : ℕ := 8

-- Condition: Number of squares
def num_squares : ℕ := 6

-- The correct answer (to be proven equivalent)
def expected_area : ℕ := 768

-- The main theorem stating the expected proof problem
theorem smallest_rectangle_area_contains_L_shape 
  (side_length : ℕ) (num_squares : ℕ) (h_shape : side_length = 8 ∧ num_squares = 6) : 
  ∃area, area = expected_area :=
by
  sorry

end smallest_rectangle_area_contains_L_shape_l328_32883


namespace maximum_area_of_region_l328_32869

/-- Given four circles with radii 2, 4, 6, and 8, tangent to the same point B 
on a line ℓ, with the two largest circles (radii 6 and 8) on the same side of ℓ,
prove that the maximum possible area of the region consisting of points lying
inside exactly one of these circles is 120π. -/
theorem maximum_area_of_region 
  (radius1 : ℝ) (radius2 : ℝ) (radius3 : ℝ) (radius4 : ℝ)
  (line : ℝ → Prop) (B : ℝ)
  (tangent1 : ∀ x, line x → dist x B = radius1) 
  (tangent2 : ∀ x, line x → dist x B = radius2)
  (tangent3 : ∀ x, line x → dist x B = radius3)
  (tangent4 : ∀ x, line x → dist x B = radius4)
  (side1 : ℕ)
  (side2 : ℕ)
  (equal_side : side1 = side2)
  (r1 : ℝ := 2) 
  (r2 : ℝ := 4)
  (r3 : ℝ := 6) 
  (r4 : ℝ := 8) :
  (π * (radius1 * radius1) + π * (radius2 * radius2) + π * (radius3 * radius3) + π * (radius4 * radius4)) = 120 * π := 
sorry

end maximum_area_of_region_l328_32869


namespace sqrt_3x_eq_5x_largest_value_l328_32888

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l328_32888


namespace faster_train_length_l328_32838

noncomputable def length_of_faster_train 
    (speed_train_1_kmph : ℤ) 
    (speed_train_2_kmph : ℤ) 
    (time_seconds : ℤ) : ℤ := 
    (speed_train_1_kmph + speed_train_2_kmph) * 1000 / 3600 * time_seconds

theorem faster_train_length 
    (speed_train_1_kmph : ℤ)
    (speed_train_2_kmph : ℤ)
    (time_seconds : ℤ)
    (h1 : speed_train_1_kmph = 36)
    (h2 : speed_train_2_kmph = 45)
    (h3 : time_seconds = 12) :
    length_of_faster_train speed_train_1_kmph speed_train_2_kmph time_seconds = 270 :=
by
    sorry

end faster_train_length_l328_32838


namespace rubble_money_left_l328_32814

/-- Rubble has $15 in his pocket. -/
def rubble_initial_amount : ℝ := 15

/-- Each notebook costs $4.00. -/
def notebook_price : ℝ := 4

/-- Each pen costs $1.50. -/
def pen_price : ℝ := 1.5

/-- Rubble needs to buy 2 notebooks. -/
def num_notebooks : ℝ := 2

/-- Rubble needs to buy 2 pens. -/
def num_pens : ℝ := 2

/-- The total cost of the notebooks. -/
def total_notebook_cost : ℝ := num_notebooks * notebook_price

/-- The total cost of the pens. -/
def total_pen_cost : ℝ := num_pens * pen_price

/-- The total amount Rubble spends. -/
def total_spent : ℝ := total_notebook_cost + total_pen_cost

/-- The remaining amount Rubble has after the purchase. -/
def rubble_remaining_amount : ℝ := rubble_initial_amount - total_spent

theorem rubble_money_left :
  rubble_remaining_amount = 4 := 
by
  -- Some necessary steps to complete the proof
  sorry

end rubble_money_left_l328_32814


namespace new_person_weight_l328_32871

theorem new_person_weight (N : ℝ) (h : N - 65 = 22.5) : N = 87.5 :=
by
  sorry

end new_person_weight_l328_32871


namespace least_multiple_of_15_greater_than_520_l328_32875

theorem least_multiple_of_15_greater_than_520 : ∃ n : ℕ, n > 520 ∧ n % 15 = 0 ∧ (∀ m : ℕ, m > 520 ∧ m % 15 = 0 → n ≤ m) ∧ n = 525 := 
by
  sorry

end least_multiple_of_15_greater_than_520_l328_32875


namespace total_decorations_l328_32850

-- Define the conditions
def decorations_per_box := 4 + 1 + 5
def total_boxes := 11 + 1

-- Statement of the problem: Prove that the total number of decorations handed out is 120
theorem total_decorations : total_boxes * decorations_per_box = 120 := by
  sorry

end total_decorations_l328_32850


namespace min_value_zero_l328_32808

noncomputable def f (k x y : ℝ) : ℝ :=
  3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ↔ (k = 3 / 2 ∨ k = -3 / 2) :=
by
  sorry

end min_value_zero_l328_32808


namespace sequence_recurrence_l328_32846

theorem sequence_recurrence (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : a 2 = 2) (h₃ : ∀ n, n ≥ 1 → a (n + 2) / a n = (a (n + 1) ^ 2 + 1) / (a n ^ 2 + 1)):
  (∀ n, a (n + 1) = a n + 1 / a n) ∧ 63 < a 2008 ∧ a 2008 < 78 :=
by
  sorry

end sequence_recurrence_l328_32846


namespace three_card_deal_probability_l328_32801

theorem three_card_deal_probability :
  (4 / 52) * (4 / 51) * (4 / 50) = 16 / 33150 := 
by 
  sorry

end three_card_deal_probability_l328_32801


namespace part1_part2_l328_32834

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a+1)*x + a

theorem part1 (a x : ℝ) :
  (a < 1 ∧ f x a < 0 ↔ a < x ∧ x < 1) ∧
  (a = 1 ∧ ¬(f x a < 0)) ∧
  (a > 1 ∧ f x a < 0 ↔ 1 < x ∧ x < a) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 < x → f x a ≥ -1) → a ≤ 3 :=
sorry

end part1_part2_l328_32834


namespace football_team_selection_l328_32882

theorem football_team_selection :
  let team_members : ℕ := 12
  let offensive_lineman_choices : ℕ := 4
  let tight_end_choices : ℕ := 2
  let players_left_after_offensive : ℕ := team_members - 1
  let players_left_after_tightend : ℕ := players_left_after_offensive - 1
  let quarterback_choices : ℕ := players_left_after_tightend
  let players_left_after_quarterback : ℕ := quarterback_choices - 1
  let running_back_choices : ℕ := players_left_after_quarterback
  let players_left_after_runningback : ℕ := running_back_choices - 1
  let wide_receiver_choices : ℕ := players_left_after_runningback
  offensive_lineman_choices * tight_end_choices * 
  quarterback_choices * running_back_choices * 
  wide_receiver_choices = 5760 := 
by 
  sorry

end football_team_selection_l328_32882


namespace divisibility_of_poly_l328_32892

theorem divisibility_of_poly (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x):
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * (y-z) * (z-x) * (x-y) * k :=
by
  sorry

end divisibility_of_poly_l328_32892


namespace max_rock_value_l328_32898

def rock_value (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  14 * weight_5 + 11 * weight_4 + 2 * weight_1

def total_weight (weight_5 : Nat) (weight_4 : Nat) (weight_1 : Nat) : Nat :=
  5 * weight_5 + 4 * weight_4 + 1 * weight_1

theorem max_rock_value : ∃ (weight_5 weight_4 weight_1 : Nat), 
  total_weight weight_5 weight_4 weight_1 ≤ 18 ∧ 
  rock_value weight_5 weight_4 weight_1 = 50 :=
by
  -- We need to find suitable weight_5, weight_4, and weight_1.
  use 2, 2, 0 -- Example values
  apply And.intro
  -- Prove the total weight condition
  show total_weight 2 2 0 ≤ 18
  sorry
  -- Prove the value condition
  show rock_value 2 2 0 = 50
  sorry

end max_rock_value_l328_32898


namespace part_a_l328_32803

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem part_a :
  ∀ (N : ℕ), (N = (sum_of_digits N) ^ 2) → (N = 1 ∨ N = 81) :=
by
  intros N h
  sorry

end part_a_l328_32803


namespace total_loads_washed_l328_32866

theorem total_loads_washed (a b : ℕ) (h1 : a = 8) (h2 : b = 6) : a + b = 14 :=
by
  sorry

end total_loads_washed_l328_32866


namespace subset_complU_N_l328_32843

variable {U : Type} {M N : Set U}

-- Given conditions
axiom non_empty_M : ∃ x, x ∈ M
axiom non_empty_N : ∃ y, y ∈ N
axiom subset_complU_M : N ⊆ Mᶜ

-- Prove the statement that M is a subset of the complement of N
theorem subset_complU_N : M ⊆ Nᶜ := by
  sorry

end subset_complU_N_l328_32843


namespace average_speed_l328_32872

-- Definitions of conditions
def speed_first_hour : ℝ := 120
def speed_second_hour : ℝ := 60
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := 2

-- Theorem stating the equivalent proof problem
theorem average_speed : total_distance / total_time = 90 := by
  sorry

end average_speed_l328_32872


namespace exists_irreducible_fractions_l328_32851

theorem exists_irreducible_fractions:
  ∃ (f : Fin 2018 → ℚ), 
    (∀ i j : Fin 2018, i ≠ j → (f i).den ≠ (f j).den) ∧ 
    (∀ i j : Fin 2018, i ≠ j → ∀ d : ℚ, d = f i - f j → d ≠ 0 → d.den < (f i).den ∧ d.den < (f j).den) :=
by
  -- proof is omitted
  sorry

end exists_irreducible_fractions_l328_32851


namespace part1_part2_l328_32810

def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

theorem part1 (a : ℝ) :
  (Set.Icc (a - 7) (a - 3)) = (Set.Icc (-5 : ℝ) (-1 : ℝ)) -> a = 2 :=
by
  intro h
  sorry

theorem part2 (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 2 < 4 * m + m^2) -> (m < -5 ∨ m > 1) :=
by
  intro h
  sorry

end part1_part2_l328_32810


namespace trig_functions_symmetry_l328_32878

theorem trig_functions_symmetry :
  ∀ k₁ k₂ : ℤ,
  (∃ x, x = k₁ * π / 2 + π / 3 ∧ x = k₂ * π + π / 3) ∧
  (¬ ∃ x, (x, 0) = (k₁ * π / 2 + π / 12, 0) ∧ (x, 0) = (k₂ * π + 5 * π / 6, 0)) :=
by
  sorry

end trig_functions_symmetry_l328_32878


namespace find_m_l328_32890

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -11.6 :=
sorry

end find_m_l328_32890


namespace rotten_tomatoes_l328_32806

-- Conditions
def weight_per_crate := 20
def num_crates := 3
def total_cost := 330
def selling_price_per_kg := 6
def profit := 12

-- Derived data
def total_weight := num_crates * weight_per_crate
def total_revenue := profit + total_cost
def sold_weight := total_revenue / selling_price_per_kg

-- Proof statement
theorem rotten_tomatoes : total_weight - sold_weight = 3 := by
  sorry

end rotten_tomatoes_l328_32806


namespace tory_toys_sold_is_7_l328_32897

-- Define the conditions as Lean definitions
def bert_toy_phones_sold : Nat := 8
def price_per_toy_phone : Nat := 18
def bert_earnings : Nat := bert_toy_phones_sold * price_per_toy_phone
def tory_earnings : Nat := bert_earnings - 4
def price_per_toy_gun : Nat := 20
def tory_toys_sold := tory_earnings / price_per_toy_gun

-- Prove that the number of toy guns Tory sold is 7
theorem tory_toys_sold_is_7 : tory_toys_sold = 7 :=
by
  sorry

end tory_toys_sold_is_7_l328_32897


namespace height_of_water_in_cylindrical_tank_l328_32894

theorem height_of_water_in_cylindrical_tank :
  let r_cone := 15  -- radius of base of conical tank in cm
  let h_cone := 24  -- height of conical tank in cm
  let r_cylinder := 18  -- radius of base of cylindrical tank in cm
  let V_cone := (1 / 3 : ℝ) * Real.pi * r_cone^2 * h_cone  -- volume of conical tank
  let h_cyl := V_cone / (Real.pi * r_cylinder^2)  -- height of water in cylindrical tank
  h_cyl = 5.56 :=
by
  sorry

end height_of_water_in_cylindrical_tank_l328_32894


namespace calculate_expression_l328_32819

theorem calculate_expression : 7 * (12 + 2 / 5) - 3 = 83.8 :=
by
  sorry

end calculate_expression_l328_32819


namespace number_is_93_75_l328_32807

theorem number_is_93_75 (x : ℝ) (h : 0.16 * (0.40 * x) = 6) : x = 93.75 :=
by
  -- The proof is omitted.
  sorry

end number_is_93_75_l328_32807


namespace placing_2_flowers_in_2_vases_l328_32805

noncomputable def num_ways_to_place_flowers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : ℕ :=
  Nat.choose n k * 2

theorem placing_2_flowers_in_2_vases :
  num_ways_to_place_flowers 5 2 rfl rfl = 20 := 
by
  sorry

end placing_2_flowers_in_2_vases_l328_32805


namespace oil_tank_depth_l328_32880

theorem oil_tank_depth (L r A : ℝ) (h : ℝ) (L_pos : L = 8) (r_pos : r = 2) (A_pos : A = 16) :
  h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

end oil_tank_depth_l328_32880


namespace find_N_l328_32822

/--
If 15% of N is 45% of 2003, then N is 6009.
-/
theorem find_N (N : ℕ) (h : 15 / 100 * N = 45 / 100 * 2003) : 
  N = 6009 :=
sorry

end find_N_l328_32822


namespace total_items_bought_l328_32886

def total_money : ℝ := 40
def sandwich_cost : ℝ := 5
def chip_cost : ℝ := 2
def soft_drink_cost : ℝ := 1.5

/-- Ike and Mike spend their total money on sandwiches, chips, and soft drinks.
  We want to prove that the total number of items bought (sandwiches, chips, and soft drinks)
  is equal to 8. -/
theorem total_items_bought :
  ∃ (s c d : ℝ), (sandwich_cost * s + chip_cost * c + soft_drink_cost * d ≤ total_money) ∧
  (∀x : ℝ, sandwich_cost * s ≤ total_money) ∧ ((s + c + d) = 8) :=
by {
  sorry
}

end total_items_bought_l328_32886


namespace rectangular_solid_width_l328_32891

theorem rectangular_solid_width 
  (l : ℝ) (w : ℝ) (h : ℝ) (S : ℝ)
  (hl : l = 5)
  (hh : h = 1)
  (hs : S = 58) :
  2 * l * w + 2 * l * h + 2 * w * h = S → w = 4 := 
by
  intros h_surface_area 
  sorry

end rectangular_solid_width_l328_32891


namespace line_passing_through_points_l328_32813

-- Definition of points
def point1 : ℝ × ℝ := (1, 0)
def point2 : ℝ × ℝ := (0, -2)

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Theorem statement
theorem line_passing_through_points : 
  line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2 :=
by
  sorry

end line_passing_through_points_l328_32813


namespace necessary_and_sufficient_problem_l328_32842

theorem necessary_and_sufficient_problem : 
  (¬ (∀ x : ℝ, (-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬ (∀ x : ℝ, (|x| > 1) → (-2 < x ∧ x < 1))) :=
by {
  sorry
}

end necessary_and_sufficient_problem_l328_32842


namespace area_of_region_below_and_left_l328_32827

theorem area_of_region_below_and_left (x y : ℝ) :
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 4^2) ∧ y ≤ 0 ∧ y ≤ x - 4 →
  π * 4^2 / 4 = 4 * π :=
by sorry

end area_of_region_below_and_left_l328_32827


namespace total_crayons_in_drawer_l328_32839

-- Definitions of conditions from a)
def initial_crayons : Nat := 9
def additional_crayons : Nat := 3

-- Statement to prove that total crayons in the drawer is 12
theorem total_crayons_in_drawer : initial_crayons + additional_crayons = 12 := sorry

end total_crayons_in_drawer_l328_32839


namespace distance_to_big_rock_l328_32856

variables (D : ℝ) (stillWaterSpeed : ℝ) (currentSpeed : ℝ) (totalTime : ℝ)

-- Define the conditions as constraints
def conditions := 
  stillWaterSpeed = 6 ∧
  currentSpeed = 1 ∧
  totalTime = 1 ∧
  (D / (stillWaterSpeed - currentSpeed) + D / (stillWaterSpeed + currentSpeed) = totalTime)

-- The theorem to prove the distance to Big Rock
theorem distance_to_big_rock (h : conditions D 6 1 1) : D = 35 / 12 :=
sorry

end distance_to_big_rock_l328_32856


namespace ratio_milk_water_larger_vessel_l328_32825

-- Definitions for the conditions given in the problem
def ratio_volume (V1 V2 : ℝ) : Prop := V1 / V2 = 3 / 5
def ratio_milk_water_vessel1 (M1 W1 : ℝ) : Prop := M1 / W1 = 1 / 2
def ratio_milk_water_vessel2 (M2 W2 : ℝ) : Prop := M2 / W2 = 3 / 2

-- The final goal to prove
theorem ratio_milk_water_larger_vessel (V1 V2 M1 W1 M2 W2 : ℝ)
  (h1 : ratio_volume V1 V2) 
  (h2 : V1 = M1 + W1) 
  (h3 : V2 = M2 + W2) 
  (h4 : ratio_milk_water_vessel1 M1 W1) 
  (h5 : ratio_milk_water_vessel2 M2 W2) :
  (M1 + M2) / (W1 + W2) = 1 :=
by
  -- Proof is omitted
  sorry

end ratio_milk_water_larger_vessel_l328_32825


namespace smallest_range_between_allocations_l328_32868

-- Problem statement in Lean
theorem smallest_range_between_allocations :
  ∀ (A B C D E : ℕ), 
  (A = 30000) →
  (B < 18000 ∨ B > 42000) →
  (C < 18000 ∨ C > 42000) →
  (D < 58802 ∨ D > 82323) →
  (E < 58802 ∨ E > 82323) →
  min B (min C (min D E)) = 17999 →
  max B (max C (max D E)) = 82323 →
  82323 - 17999 = 64324 :=
by
  intros A B C D E hA hB hC hD hE hmin hmax
  sorry

end smallest_range_between_allocations_l328_32868


namespace largest_divisor_of_5_consecutive_integers_l328_32835

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end largest_divisor_of_5_consecutive_integers_l328_32835


namespace problem_solution_l328_32824

theorem problem_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end problem_solution_l328_32824


namespace fuel_cost_equation_l328_32884

theorem fuel_cost_equation (x : ℝ) (h : (x / 4) - (x / 6) = 8) : x = 96 :=
sorry

end fuel_cost_equation_l328_32884


namespace interval_length_difference_l328_32865

noncomputable def log2_abs (x : ℝ) : ℝ := |Real.log x / Real.log 2|

theorem interval_length_difference :
  ∀ (a b : ℝ), (∀ x, a ≤ x ∧ x ≤ b → 0 ≤ log2_abs x ∧ log2_abs x ≤ 2) → 
               (b - a = 15 / 4 - 3 / 4) :=
by
  intros a b h
  sorry

end interval_length_difference_l328_32865


namespace find_fraction_sum_l328_32844

theorem find_fraction_sum (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -2) : (1 / x) + (1 / y) = -3 :=
by
  sorry

end find_fraction_sum_l328_32844


namespace unique_integer_n_l328_32852

theorem unique_integer_n (n : ℤ) (h : ⌊(n^2 : ℚ) / 5⌋ - ⌊(n / 2 : ℚ)⌋^2 = 3) : n = 5 :=
  sorry

end unique_integer_n_l328_32852


namespace xiao_ming_min_correct_answers_l328_32876

theorem xiao_ming_min_correct_answers (x : ℕ) : (10 * x - 5 * (20 - x) > 100) → (x ≥ 14) := by
  sorry

end xiao_ming_min_correct_answers_l328_32876


namespace sandy_initial_cost_l328_32899

theorem sandy_initial_cost 
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (gain_percent : ℝ)
  (h1 : repairs_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ P : ℝ, P = 800 :=
by
  -- Proof steps would go here
  sorry

end sandy_initial_cost_l328_32899


namespace third_quadrant_angle_to_fourth_l328_32853

theorem third_quadrant_angle_to_fourth {α : ℝ} (k : ℤ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  -90 - k * 360 < 180 - α ∧ 180 - α < -k * 360 :=
by
  sorry

end third_quadrant_angle_to_fourth_l328_32853


namespace max_f_geq_fraction_3_sqrt3_over_2_l328_32860

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq_fraction_3_sqrt3_over_2 : ∃ x : ℝ, f x ≥ (3 + Real.sqrt 3) / 2 := 
sorry

end max_f_geq_fraction_3_sqrt3_over_2_l328_32860


namespace product_of_functions_l328_32831

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x + 3)
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem product_of_functions (x : ℝ) (hx : x ≠ -3) : f x * g x = x - 3 := by
  -- proof goes here
  sorry

end product_of_functions_l328_32831


namespace contrapositive_statement_l328_32826

theorem contrapositive_statement (m : ℝ) (h : ¬ ∃ x : ℝ, x^2 = m) : m < 0 :=
sorry

end contrapositive_statement_l328_32826


namespace A_in_second_quadrant_l328_32881

-- Define the coordinates of point A
def A_x : ℝ := -2
def A_y : ℝ := 3

-- Define the condition that point A lies in the second quadrant
def is_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State the theorem
theorem A_in_second_quadrant : is_second_quadrant A_x A_y :=
by
  -- The proof will be provided here.
  sorry

end A_in_second_quadrant_l328_32881


namespace three_w_seven_l328_32896

def operation_w (a b : ℤ) : ℤ := b + 5 * a - 3 * a^2

theorem three_w_seven : operation_w 3 7 = -5 :=
by
  sorry

end three_w_seven_l328_32896


namespace parallel_vectors_y_value_l328_32863

theorem parallel_vectors_y_value 
  (y : ℝ) 
  (a : ℝ × ℝ := (6, 2)) 
  (b : ℝ × ℝ := (y, 3)) 
  (h : ∃ k : ℝ, b = k • a) : y = 9 :=
sorry

end parallel_vectors_y_value_l328_32863


namespace at_least_one_not_less_than_one_third_l328_32877

theorem at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 :=
sorry

end at_least_one_not_less_than_one_third_l328_32877


namespace ratio_of_inradii_l328_32867

-- Given triangle XYZ with sides XZ=5, YZ=12, XY=13
-- Let W be on XY such that ZW bisects ∠ YZX
-- The inscribed circles of triangles ZWX and ZWY have radii r_x and r_y respectively
-- Prove the ratio r_x / r_y = 1/6

theorem ratio_of_inradii
  (XZ YZ XY : ℝ)
  (W : ℝ)
  (r_x r_y : ℝ)
  (h1 : XZ = 5)
  (h2 : YZ = 12)
  (h3 : XY = 13)
  (h4 : r_x / r_y = 1/6) :
  r_x / r_y = 1/6 :=
by sorry

end ratio_of_inradii_l328_32867


namespace player_one_wins_l328_32854

theorem player_one_wins (initial_coins : ℕ) (h_initial : initial_coins = 2015) : 
  ∃ first_move : ℕ, (1 ≤ first_move ∧ first_move ≤ 99 ∧ first_move % 2 = 1) ∧ 
  (∀ move : ℕ, (2 ≤ move ∧ move ≤ 100 ∧ move % 2 = 0) → 
   ∃ next_move : ℕ, (1 ≤ next_move ∧ next_move ≤ 99 ∧ next_move % 2 = 1) → 
   initial_coins - first_move - move - next_move < 101) → first_move = 95 :=
by 
  sorry

end player_one_wins_l328_32854


namespace complement_intersection_l328_32804

open Set

theorem complement_intersection (U A B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 5})
  (hB : B = {2, 4}) :
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end complement_intersection_l328_32804


namespace remainder_of_division_l328_32849

noncomputable def P (x : ℝ) := x ^ 888
noncomputable def Q (x : ℝ) := (x ^ 2 - x + 1) * (x + 1)

theorem remainder_of_division :
  ∀ x : ℝ, (P x) % (Q x) = 1 :=
sorry

end remainder_of_division_l328_32849


namespace fraction_defined_l328_32820

theorem fraction_defined (x : ℝ) : (1 - 2 * x ≠ 0) ↔ (x ≠ 1 / 2) :=
by sorry

end fraction_defined_l328_32820


namespace sum_of_powers_of_two_l328_32833

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 = 2^5 :=
by
  sorry

end sum_of_powers_of_two_l328_32833


namespace mike_last_5_shots_l328_32855

theorem mike_last_5_shots :
  let initial_shots := 30
  let initial_percentage := 40 / 100
  let additional_shots_1 := 10
  let new_percentage_1 := 45 / 100
  let additional_shots_2 := 5
  let new_percentage_2 := 46 / 100
  
  let initial_makes := initial_shots * initial_percentage
  let total_shots_after_1 := initial_shots + additional_shots_1
  let makes_after_1 := total_shots_after_1 * new_percentage_1 - initial_makes
  let total_makes_after_1 := initial_makes + makes_after_1
  let total_shots_after_2 := total_shots_after_1 + additional_shots_2
  let final_makes := total_shots_after_2 * new_percentage_2
  let makes_in_last_5 := final_makes - total_makes_after_1
  
  makes_in_last_5 = 2
:=
by
  sorry

end mike_last_5_shots_l328_32855


namespace runners_meet_time_l328_32830

theorem runners_meet_time (t_P t_Q : ℕ) (hP: t_P = 252) (hQ: t_Q = 198) : Nat.lcm t_P t_Q = 2772 :=
by
  rw [hP, hQ]
  -- The proof can be continued by proving the LCM calculation step, which we omit here
  sorry

end runners_meet_time_l328_32830
