import Mathlib

namespace S8_value_l1146_114678

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem S8_value 
  (h_geo : is_geometric_sequence a q)
  (h_S4 : S 4 = 3)
  (h_S12_S8 : S 12 - S 8 = 12) :
  S 8 = 9 := 
sorry

end S8_value_l1146_114678


namespace min_function_value_l1146_114677

theorem min_function_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  (1/3 * x^3 + y^2 + z) = 13/12 :=
sorry

end min_function_value_l1146_114677


namespace det_abs_eq_one_l1146_114691

variable {n : ℕ}
variable {A : Matrix (Fin n) (Fin n) ℤ}
variable {p q r : ℕ}
variable (hpq : p^2 = q^2 + r^2)
variable (hodd : Odd r)
variable (hA : p^2 • A ^ p^2 = q^2 • A ^ q^2 + r^2 • 1)

theorem det_abs_eq_one : |A.det| = 1 := by
  sorry

end det_abs_eq_one_l1146_114691


namespace hyperbola_asymptote_slope_proof_l1146_114697

noncomputable def hyperbola_asymptote_slope : ℝ :=
  let foci_distance := Real.sqrt ((8 - 2)^2 + (3 - 3)^2)
  let c := foci_distance / 2
  let a := 2  -- Given that 2a = 4
  let b := Real.sqrt (c^2 - a^2)
  b / a

theorem hyperbola_asymptote_slope_proof :
  ∀ x y : ℝ, 
  (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) →
  hyperbola_asymptote_slope = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_asymptote_slope_proof_l1146_114697


namespace mathe_matics_equals_2014_l1146_114628

/-- 
Given the following mappings for characters in the word "MATHEMATICS":
M = 1, A = 8, T = 3, E = '+', I = 9, K = '-',
verify that the resulting numerical expression 183 + 1839 - 8 equals 2014.
-/
theorem mathe_matics_equals_2014 :
  183 + 1839 - 8 = 2014 :=
by
  sorry

end mathe_matics_equals_2014_l1146_114628


namespace tracy_additional_miles_l1146_114679

def total_distance : ℕ := 1000
def michelle_distance : ℕ := 294
def twice_michelle_distance : ℕ := 2 * michelle_distance
def katie_distance : ℕ := michelle_distance / 3
def tracy_distance := total_distance - (michelle_distance + katie_distance)
def additional_miles := tracy_distance - twice_michelle_distance

-- The statement to prove:
theorem tracy_additional_miles : additional_miles = 20 := by
  sorry

end tracy_additional_miles_l1146_114679


namespace carmen_sprigs_left_l1146_114617

-- Definitions based on conditions
def initial_sprigs : ℕ := 25
def whole_sprigs_used : ℕ := 8
def half_sprigs_plates : ℕ := 12
def half_sprigs_total_used : ℕ := half_sprigs_plates / 2

-- Total sprigs used
def total_sprigs_used : ℕ := whole_sprigs_used + half_sprigs_total_used

-- Leftover sprigs computation
def sprigs_left : ℕ := initial_sprigs - total_sprigs_used

-- Statement to prove
theorem carmen_sprigs_left : sprigs_left = 11 :=
by
  sorry

end carmen_sprigs_left_l1146_114617


namespace regular_14_gon_inequality_l1146_114614

noncomputable def side_length_of_regular_14_gon : ℝ := 2 * Real.sin (Real.pi / 14)

theorem regular_14_gon_inequality (a : ℝ) (h : a = side_length_of_regular_14_gon) :
  (2 - a) / (2 * a) > Real.sqrt (3 * Real.cos (Real.pi / 7)) :=
by
  sorry

end regular_14_gon_inequality_l1146_114614


namespace slices_leftover_l1146_114644

def total_initial_slices : ℕ := 12 * 2
def bob_slices : ℕ := 12 / 2
def tom_slices : ℕ := 12 / 3
def sally_slices : ℕ := 12 / 6
def jerry_slices : ℕ := 12 / 4
def total_slices_eaten : ℕ := bob_slices + tom_slices + sally_slices + jerry_slices

theorem slices_leftover : total_initial_slices - total_slices_eaten = 9 := by
  sorry

end slices_leftover_l1146_114644


namespace find_angle_B_l1146_114608

theorem find_angle_B (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 45) 
  (h2 : a = 6) 
  (h3 : b = 3 * Real.sqrt 2)
  (h4 : ∀ A' B' C' : ℝ, 
        ∃ a' b' c' : ℝ, 
        (a' = a) ∧ (b' = b) ∧ (A' = A) ∧ 
        (b' < a') → (B' < A') ∧ (A' = 45)) :
  B = 30 :=
by
  sorry

end find_angle_B_l1146_114608


namespace circulation_ratio_l1146_114609

variable (A : ℕ) -- Assuming A to be a natural number for simplicity

theorem circulation_ratio (h : ∀ t : ℕ, t = 1971 → t = 4 * A) : 4 / 13 = 4 / 13 := 
by
  sorry

end circulation_ratio_l1146_114609


namespace neg_p_l1146_114687

theorem neg_p : ∀ (m : ℝ), ∀ (x : ℝ), (x^2 + m*x + 1 ≠ 0) :=
by
  sorry

end neg_p_l1146_114687


namespace weight_12m_rod_l1146_114633

-- Define the weight of a 6 meters long rod
def weight_of_6m_rod : ℕ := 7

-- Given the condition that the weight is proportional to the length
def weight_of_rod (length : ℕ) : ℕ := (length / 6) * weight_of_6m_rod

-- Prove the weight of a 12 meters long rod
theorem weight_12m_rod : weight_of_rod 12 = 14 := by
  -- Calculation skipped, proof required here
  sorry

end weight_12m_rod_l1146_114633


namespace obtuse_is_second_quadrant_l1146_114602

-- Define the boundaries for an obtuse angle.
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define the second quadrant condition.
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The proof problem: Prove that an obtuse angle is a second quadrant angle.
theorem obtuse_is_second_quadrant (θ : ℝ) : is_obtuse θ → is_second_quadrant θ :=
by
  intro h
  sorry

end obtuse_is_second_quadrant_l1146_114602


namespace percentage_relationship_l1146_114673

theorem percentage_relationship (a b : ℝ) (h : a = 1.2 * b) : ¬ (b = 0.8 * a) :=
by
  -- assumption: a = 1.2 * b
  -- goal: ¬ (b = 0.8 * a)
  sorry

end percentage_relationship_l1146_114673


namespace marble_problem_l1146_114661

theorem marble_problem
  (x : ℕ) (h1 : 144 / x = 144 / (x + 2) + 1) :
  x = 16 :=
sorry

end marble_problem_l1146_114661


namespace Andrew_runs_2_miles_each_day_l1146_114690

theorem Andrew_runs_2_miles_each_day
  (A : ℕ)
  (Peter_runs : ℕ := A + 3)
  (total_miles_after_5_days : 5 * (A + Peter_runs) = 35) :
  A = 2 :=
by
  sorry

end Andrew_runs_2_miles_each_day_l1146_114690


namespace mean_greater_than_median_by_six_l1146_114669

theorem mean_greater_than_median_by_six (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 :=
by
  sorry

end mean_greater_than_median_by_six_l1146_114669


namespace probability_of_purple_is_one_fifth_l1146_114619

-- Definitions related to the problem
def total_faces : ℕ := 10
def purple_faces : ℕ := 2
def probability_purple := (purple_faces : ℚ) / (total_faces : ℚ)

theorem probability_of_purple_is_one_fifth : probability_purple = 1 / 5 := 
by
  -- Converting the numbers to rationals explicitly ensures division is defined.
  change (2 : ℚ) / (10 : ℚ) = 1 / 5
  norm_num
  -- sorry (if finishing the proof manually isn't desired)

end probability_of_purple_is_one_fifth_l1146_114619


namespace parentheses_removal_correct_l1146_114682

theorem parentheses_removal_correct (x y : ℝ) : -(x^2 + y^2) = -x^2 - y^2 :=
by
  sorry

end parentheses_removal_correct_l1146_114682


namespace min_value_of_function_l1146_114649

theorem min_value_of_function (p : ℝ) : 
  ∃ x : ℝ, (x^2 - 2 * p * x + 2 * p^2 + 2 * p - 1) = -2 := sorry

end min_value_of_function_l1146_114649


namespace haley_more_than_josh_l1146_114667

-- Definitions of the variables and conditions
variable (H : Nat) -- Number of necklaces Haley has
variable (J : Nat) -- Number of necklaces Jason has
variable (Jos : Nat) -- Number of necklaces Josh has

-- The conditions as assumptions
axiom h1 : H = 25
axiom h2 : H = J + 5
axiom h3 : Jos = J / 2

-- The theorem we want to prove based on these conditions
theorem haley_more_than_josh (H J Jos : Nat) (h1 : H = 25) (h2 : H = J + 5) (h3 : Jos = J / 2) : H - Jos = 15 := 
by 
  sorry

end haley_more_than_josh_l1146_114667


namespace solve_quadratic_eq_l1146_114651

theorem solve_quadratic_eq (x : ℝ) : x^2 = 6 * x ↔ (x = 0 ∨ x = 6) := by
  sorry

end solve_quadratic_eq_l1146_114651


namespace initial_marbles_l1146_114655

theorem initial_marbles (M : ℕ) (h1 : M + 9 = 104) : M = 95 := by
  sorry

end initial_marbles_l1146_114655


namespace insurance_covers_80_percent_of_lenses_l1146_114688

/--
James needs to get a new pair of glasses. 
His frames cost $200 and the lenses cost $500. 
Insurance will cover a certain percentage of the cost of lenses and he has a $50 off coupon for frames. 
Everything costs $250. 
Prove that the insurance covers 80% of the cost of the lenses.
-/

def frames_cost : ℕ := 200
def lenses_cost : ℕ := 500
def total_cost_after_discounts_and_insurance : ℕ := 250
def coupon : ℕ := 50

theorem insurance_covers_80_percent_of_lenses :
  ((frames_cost - coupon + lenses_cost - total_cost_after_discounts_and_insurance) * 100 / lenses_cost) = 80 := 
  sorry

end insurance_covers_80_percent_of_lenses_l1146_114688


namespace angle_terminal_side_on_non_negative_y_axis_l1146_114663

theorem angle_terminal_side_on_non_negative_y_axis (P : ℝ × ℝ) (α : ℝ) (hP : P = (0, 3)) :
  α = some_angle_with_terminal_side_on_non_negative_y_axis := by
  sorry

end angle_terminal_side_on_non_negative_y_axis_l1146_114663


namespace number_of_people_who_didnt_do_both_l1146_114641

def total_graduates : ℕ := 73
def graduates_both : ℕ := 13

theorem number_of_people_who_didnt_do_both : total_graduates - graduates_both = 60 :=
by
  sorry

end number_of_people_who_didnt_do_both_l1146_114641


namespace two_fifths_in_fraction_l1146_114685

theorem two_fifths_in_fraction : 
  (∃ (k : ℚ), k = (9/3) / (2/5) ∧ k = 15/2) :=
by 
  sorry

end two_fifths_in_fraction_l1146_114685


namespace dinner_cost_l1146_114698

theorem dinner_cost (tax_rate : ℝ) (tip_rate : ℝ) (total_amount : ℝ) : 
  tax_rate = 0.12 → 
  tip_rate = 0.18 → 
  total_amount = 30 → 
  (total_amount / (1 + tax_rate + tip_rate)) = 23.08 :=
by
  intros h1 h2 h3
  sorry

end dinner_cost_l1146_114698


namespace find_principal_l1146_114692

noncomputable def principal_amount (P : ℝ) : Prop :=
  let r := 0.05
  let t := 2
  let SI := P * r * t
  let CI := P * (1 + r) ^ t - P
  CI - SI = 15

theorem find_principal : principal_amount 6000 :=
by
  simp [principal_amount]
  sorry

end find_principal_l1146_114692


namespace AM_GM_Inequality_four_vars_l1146_114648

theorem AM_GM_Inequality_four_vars (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

end AM_GM_Inequality_four_vars_l1146_114648


namespace average_salary_technicians_correct_l1146_114699

section
variable (average_salary_all : ℝ)
variable (total_workers : ℕ)
variable (average_salary_rest : ℝ)
variable (num_technicians : ℕ)

noncomputable def average_salary_technicians
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : ℝ :=
  12000

theorem average_salary_technicians_correct
  (h1 : average_salary_all = 8000)
  (h2 : total_workers = 30)
  (h3 : average_salary_rest = 6000)
  (h4 : num_technicians = 10)
  : average_salary_technicians average_salary_all total_workers average_salary_rest num_technicians h1 h2 h3 h4 = 12000 :=
sorry

end

end average_salary_technicians_correct_l1146_114699


namespace interest_rate_bc_l1146_114684

def interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

def gain_b (interest_bc interest_ab : ℝ) : ℝ :=
  interest_bc - interest_ab

theorem interest_rate_bc :
  ∀ (principal : ℝ) (rate_ab rate_bc : ℝ) (time : ℕ) (gain : ℝ),
    principal = 3500 → rate_ab = 0.10 → time = 3 → gain = 525 →
    interest principal rate_ab time = 1050 →
    gain_b (interest principal rate_bc time) (interest principal rate_ab time) = gain →
    rate_bc = 0.15 :=
by
  intros principal rate_ab rate_bc time gain h_principal h_rate_ab h_time h_gain h_interest_ab h_gain_b
  sorry

end interest_rate_bc_l1146_114684


namespace largest_of_five_consecutive_sum_180_l1146_114665

theorem largest_of_five_consecutive_sum_180 (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 180) :
  n + 4 = 38 :=
by
  sorry

end largest_of_five_consecutive_sum_180_l1146_114665


namespace fraction_of_boys_participated_l1146_114662

-- Definitions based on given conditions
def total_students (B G : ℕ) : Prop := B + G = 800
def participating_girls (G : ℕ) : Prop := (3 / 4 : ℚ) * G = 150
def total_participants (P : ℕ) : Prop := P = 550
def participating_girls_count (PG : ℕ) : Prop := PG = 150

-- Definition of the fraction of participating boys
def fraction_participating_boys (X : ℚ) (B : ℕ) (PB : ℕ) : Prop := X * B = PB

-- The problem of proving the fraction of boys who participated
theorem fraction_of_boys_participated (B G PB : ℕ) (X : ℚ)
  (h1 : total_students B G)
  (h2 : participating_girls G)
  (h3 : total_participants 550)
  (h4 : participating_girls_count 150)
  (h5 : PB = 550 - 150) :
  fraction_participating_boys X B PB → X = 2 / 3 := by
  sorry

end fraction_of_boys_participated_l1146_114662


namespace distance_in_interval_l1146_114654

open Set Real

def distance_to_town (d : ℝ) : Prop :=
d < 8 ∧ 7 < d ∧ 6 < d

theorem distance_in_interval (d : ℝ) : distance_to_town d → d ∈ Ioo 7 8 :=
by
  intro h
  have d_in_Ioo_8 := h.left
  have d_in_Ioo_7 := h.right.left
  have d_in_Ioo_6 := h.right.right
  /- The specific steps for combining inequalities aren't needed for the final proof. -/
  sorry

end distance_in_interval_l1146_114654


namespace Lloyd_hourly_rate_is_3_5_l1146_114694

/-!
Lloyd normally works 7.5 hours per day and earns a certain amount per hour.
For each hour he works in excess of 7.5 hours on a given day, he is paid 1.5 times his regular rate.
If Lloyd works 10.5 hours on a given day, he earns $42 for that day.
-/

variable (Lloyd_hourly_rate : ℝ)  -- regular hourly rate

def Lloyd_daily_earnings (total_hours : ℝ) (regular_hours : ℝ) (hourly_rate : ℝ) : ℝ :=
  let excess_hours := total_hours - regular_hours
  let excess_earnings := excess_hours * (1.5 * hourly_rate)
  let regular_earnings := regular_hours * hourly_rate
  excess_earnings + regular_earnings

-- Given conditions
axiom H1 : 7.5 = 7.5
axiom H2 : ∀ R : ℝ, Lloyd_hourly_rate = R
axiom H3 : ∀ R : ℝ, ∀ excess_hours : ℝ, Lloyd_hourly_rate + excess_hours = 1.5 * R
axiom H4 : Lloyd_daily_earnings 10.5 7.5 Lloyd_hourly_rate = 42

-- Prove Lloyd earns $3.50 per hour.
theorem Lloyd_hourly_rate_is_3_5 : Lloyd_hourly_rate = 3.5 :=
sorry

end Lloyd_hourly_rate_is_3_5_l1146_114694


namespace notebook_cost_l1146_114605

theorem notebook_cost
  (n c : ℝ)
  (h1 : n + c = 2.20)
  (h2 : n = c + 2) :
  n = 2.10 :=
by
  sorry

end notebook_cost_l1146_114605


namespace abs_square_implication_l1146_114671

theorem abs_square_implication (a b : ℝ) (h : abs a > abs b) : a^2 > b^2 :=
by sorry

end abs_square_implication_l1146_114671


namespace total_triangles_in_geometric_figure_l1146_114638

noncomputable def numberOfTriangles : ℕ :=
  let smallest_triangles := 3 + 2 + 1
  let medium_triangles := 2
  let large_triangle := 1
  smallest_triangles + medium_triangles + large_triangle

theorem total_triangles_in_geometric_figure : numberOfTriangles = 9 := by
  unfold numberOfTriangles
  sorry

end total_triangles_in_geometric_figure_l1146_114638


namespace floor_sqrt_17_squared_eq_16_l1146_114606

theorem floor_sqrt_17_squared_eq_16 :
  (⌊Real.sqrt 17⌋ : Real)^2 = 16 := by
  sorry

end floor_sqrt_17_squared_eq_16_l1146_114606


namespace remainder_sum_15_div_11_l1146_114624

theorem remainder_sum_15_div_11 :
  let n := 15 
  let a := 1 
  let l := 15 
  let S := (n * (a + l)) / 2
  S % 11 = 10 :=
by
  let n := 15
  let a := 1
  let l := 15
  let S := (n * (a + l)) / 2
  show S % 11 = 10
  sorry

end remainder_sum_15_div_11_l1146_114624


namespace jane_original_number_l1146_114646

theorem jane_original_number (x : ℝ) (h : 5 * (3 * x + 16) = 250) : x = 34 / 3 := 
sorry

end jane_original_number_l1146_114646


namespace trigonometric_identity_l1146_114696

theorem trigonometric_identity :
  (2 * Real.sin (10 * Real.pi / 180) - Real.cos (20 * Real.pi / 180)) / Real.cos (70 * Real.pi / 180) = - Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l1146_114696


namespace required_sand_volume_is_five_l1146_114653

noncomputable def length : ℝ := 10
noncomputable def depth_cm : ℝ := 50
noncomputable def depth_m : ℝ := depth_cm / 100  -- converting cm to m
noncomputable def width : ℝ := 2
noncomputable def total_volume : ℝ := length * depth_m * width
noncomputable def current_volume : ℝ := total_volume / 2
noncomputable def additional_sand : ℝ := total_volume - current_volume

theorem required_sand_volume_is_five : additional_sand = 5 :=
by sorry

end required_sand_volume_is_five_l1146_114653


namespace simplify_expression_l1146_114631

theorem simplify_expression (b : ℝ) : (1 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4) = 360 * b^10 :=
by sorry

end simplify_expression_l1146_114631


namespace find_sum_of_terms_l1146_114695

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def given_conditions (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ (a 4 + a 7 = 2) ∧ (a 5 * a 6 = -8)

theorem find_sum_of_terms (a : ℕ → ℝ) (h : given_conditions a) : a 1 + a 10 = -7 :=
sorry

end find_sum_of_terms_l1146_114695


namespace tan_neq_sqrt3_sufficient_but_not_necessary_l1146_114601

-- Definition of the condition: tan(α) ≠ √3
def condition_tan_neq_sqrt3 (α : ℝ) : Prop := Real.tan α ≠ Real.sqrt 3

-- Definition of the statement: α ≠ π/3
def statement_alpha_neq_pi_div_3 (α : ℝ) : Prop := α ≠ Real.pi / 3

-- The theorem to be proven
theorem tan_neq_sqrt3_sufficient_but_not_necessary {α : ℝ} :
  condition_tan_neq_sqrt3 α → statement_alpha_neq_pi_div_3 α :=
sorry

end tan_neq_sqrt3_sufficient_but_not_necessary_l1146_114601


namespace max_value_of_M_l1146_114627

def J (k : ℕ) := 10^(k + 3) + 256

def M (k : ℕ) := Nat.factors (J k) |>.count 2

theorem max_value_of_M (k : ℕ) (hk : k > 0) :
  M k = 8 := by
  sorry

end max_value_of_M_l1146_114627


namespace cashback_discount_percentage_l1146_114693

noncomputable def iphoneOriginalPrice : ℝ := 800
noncomputable def iwatchOriginalPrice : ℝ := 300
noncomputable def iphoneDiscountRate : ℝ := 0.15
noncomputable def iwatchDiscountRate : ℝ := 0.10
noncomputable def finalPrice : ℝ := 931

noncomputable def iphoneDiscountedPrice : ℝ := iphoneOriginalPrice * (1 - iphoneDiscountRate)
noncomputable def iwatchDiscountedPrice : ℝ := iwatchOriginalPrice * (1 - iwatchDiscountRate)
noncomputable def totalDiscountedPrice : ℝ := iphoneDiscountedPrice + iwatchDiscountedPrice
noncomputable def cashbackAmount : ℝ := totalDiscountedPrice - finalPrice
noncomputable def cashbackRate : ℝ := (cashbackAmount / totalDiscountedPrice) * 100

theorem cashback_discount_percentage : cashbackRate = 2 := by
  sorry

end cashback_discount_percentage_l1146_114693


namespace sum_of_numbers_l1146_114686

theorem sum_of_numbers : 148 + 35 + 17 + 13 + 9 = 222 := 
by
  sorry

end sum_of_numbers_l1146_114686


namespace validity_of_D_l1146_114660

def binary_op (a b : ℕ) : ℕ := a^(b + 1)

theorem validity_of_D (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  binary_op (a^n) b = (binary_op a b)^n := 
by
  sorry

end validity_of_D_l1146_114660


namespace greatest_integer_floor_div_l1146_114681

-- Define the parameters
def a : ℕ := 3^100 + 2^105
def b : ℕ := 3^96 + 2^101

-- Formulate the proof statement
theorem greatest_integer_floor_div (a b : ℕ) : 
  a = 3^100 + 2^105 →
  b = 3^96 + 2^101 →
  (a / b) = 16 := 
by
  intros ha hb
  sorry

end greatest_integer_floor_div_l1146_114681


namespace alex_class_size_l1146_114618

theorem alex_class_size 
  (n : ℕ) 
  (h_top : 30 ≤ n)
  (h_bottom : 30 ≤ n) 
  (h_better : n - 30 > 0)
  (h_worse : n - 30 > 0)
  : n = 59 := 
sorry

end alex_class_size_l1146_114618


namespace right_triangle_AB_is_approximately_8point3_l1146_114625

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem right_triangle_AB_is_approximately_8point3 :
  ∀ (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (BC AB : ℝ),
  angle_A = 40 ∧ angle_B = 90 ∧ BC = 7 →
  AB = 7 / tan_deg 40 →
  abs (AB - 8.3) < 0.1 :=
by
  intros A B C angle_A angle_B BC AB h_cond h_AB
  sorry

end right_triangle_AB_is_approximately_8point3_l1146_114625


namespace larger_number_1655_l1146_114674

theorem larger_number_1655 (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
by sorry

end larger_number_1655_l1146_114674


namespace range_a_empty_intersection_range_a_sufficient_condition_l1146_114659

noncomputable def A (x : ℝ) : Prop := -10 < x ∧ x < 2
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a
noncomputable def A_inter_B_empty (a : ℝ) : Prop := ∀ x : ℝ, A x → ¬ B x a
noncomputable def neg_p (x : ℝ) : Prop := x ≥ 2 ∨ x ≤ -10
noncomputable def neg_p_implies_q (a : ℝ) : Prop := ∀ x : ℝ, neg_p x → B x a

theorem range_a_empty_intersection : (∀ x : ℝ, A x → ¬ B x 11) → 11 ≤ a := by
  sorry

theorem range_a_sufficient_condition : (∀ x : ℝ, neg_p x → B x 1) → 0 < a ∧ a ≤ 1 := by
  sorry

end range_a_empty_intersection_range_a_sufficient_condition_l1146_114659


namespace five_more_than_three_in_pages_l1146_114656

def pages := (List.range 512).map (λ n => n + 1)

def count_digit (d : Nat) (n : Nat) : Nat :=
  if n = 0 then 0
  else if n % 10 = d then 1 + count_digit d (n / 10)
  else count_digit d (n / 10)

def total_digit_count (d : Nat) (l : List Nat) : Nat :=
  l.foldl (λ acc x => acc + count_digit d x) 0

theorem five_more_than_three_in_pages :
  total_digit_count 5 pages - total_digit_count 3 pages = 22 := 
by 
  sorry

end five_more_than_three_in_pages_l1146_114656


namespace f_eq_zero_of_le_zero_l1146_114670

variable {R : Type*} [LinearOrderedField R]
variable {f : R → R}
variable (cond : ∀ x y : R, f (x + y) ≤ y * f x + f (f x))

theorem f_eq_zero_of_le_zero (x : R) (h : x ≤ 0) : f x = 0 :=
sorry

end f_eq_zero_of_le_zero_l1146_114670


namespace find_amplitude_l1146_114616

theorem find_amplitude (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : A = 4 :=
by
  sorry

end find_amplitude_l1146_114616


namespace child_ticket_cost_l1146_114664

theorem child_ticket_cost :
  ∃ x : ℤ, (9 * 11 = 7 * x + 50) ∧ x = 7 :=
by
  sorry

end child_ticket_cost_l1146_114664


namespace prove_statements_l1146_114611

theorem prove_statements (x y z : ℝ) (h : x + y + z = x * y * z) :
  ( (∀ (x y : ℝ), x + y = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → z = 0))
  ∧ (∀ (x y : ℝ), x = 0 → (∃ (z : ℝ), (x + y + z = x * y * z) → y = -z))
  ∧ z = (x + y) / (x * y - 1) ) :=
by
  sorry

end prove_statements_l1146_114611


namespace base6_addition_problem_l1146_114603

theorem base6_addition_problem (X Y : ℕ) (h1 : 3 * 6^2 + X * 6 + Y + 24 = 6 * 6^2 + 1 * 6 + X) :
  X = 5 ∧ Y = 1 ∧ X + Y = 6 := by
  sorry

end base6_addition_problem_l1146_114603


namespace probability_identical_cubes_l1146_114652

-- Definitions translating given conditions
def total_ways_to_paint_single_cube : Nat := 3^6
def total_ways_to_paint_three_cubes : Nat := total_ways_to_paint_single_cube^3

-- Cases counting identical painting schemes
def identical_painting_schemes : Nat :=
  let case_A := 3
  let case_B := 90
  let case_C := 540
  case_A + case_B + case_C

-- The main theorem stating the desired probability
theorem probability_identical_cubes :
  let total_ways := (387420489 : ℚ) -- 729^3
  let favorable_ways := (633 : ℚ)  -- sum of all cases (3 + 90 + 540)
  favorable_ways / total_ways = (211 / 129140163 : ℚ) :=
by
  sorry

end probability_identical_cubes_l1146_114652


namespace largest_inscribed_equilateral_triangle_area_l1146_114607

theorem largest_inscribed_equilateral_triangle_area 
  (r : ℝ) (h_r : r = 10) : 
  ∃ A : ℝ, 
    A = 100 * Real.sqrt 3 ∧ 
    (∃ s : ℝ, s = 2 * r ∧ A = (Real.sqrt 3 / 4) * s^2) := 
  sorry

end largest_inscribed_equilateral_triangle_area_l1146_114607


namespace correct_oblique_projection_conclusions_l1146_114675

def oblique_projection (shape : Type) : Type := shape

theorem correct_oblique_projection_conclusions :
  (oblique_projection Triangle = Triangle) ∧
  (oblique_projection Parallelogram = Parallelogram) ↔
  (oblique_projection Square ≠ Square) ∧
  (oblique_projection Rhombus ≠ Rhombus) :=
by
  sorry

end correct_oblique_projection_conclusions_l1146_114675


namespace inequality_solution_l1146_114650

theorem inequality_solution (x : ℝ) :
  (x + 2 > 3 * (2 - x) ∧ x < (x + 3) / 2) ↔ 1 < x ∧ x < 3 := sorry

end inequality_solution_l1146_114650


namespace balance_blue_balls_l1146_114666

variable (G Y W B : ℝ)

-- Define the conditions
def condition1 : 4 * G = 8 * B := sorry
def condition2 : 3 * Y = 8 * B := sorry
def condition3 : 4 * B = 3 * W := sorry

-- Prove the required balance of 3G + 4Y + 3W
theorem balance_blue_balls (h1 : 4 * G = 8 * B) (h2 : 3 * Y = 8 * B) (h3 : 4 * B = 3 * W) :
  3 * (2 * B) + 4 * (8 / 3 * B) + 3 * (4 / 3 * B) = 62 / 3 * B := by
  sorry

end balance_blue_balls_l1146_114666


namespace arccos_one_eq_zero_l1146_114613

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  -- the proof will go here
  sorry

end arccos_one_eq_zero_l1146_114613


namespace inequality_1_l1146_114621

theorem inequality_1 (x : ℝ) : (x - 2) * (1 - 3 * x) > 2 → 1 < x ∧ x < 4 / 3 :=
by sorry

end inequality_1_l1146_114621


namespace expr_comparison_l1146_114645

-- Define the given condition
def eight_pow_2001 : ℝ := 8 * (64 : ℝ) ^ 1000

-- State the theorem
theorem expr_comparison : (65 : ℝ) ^ 1000 > eight_pow_2001 := by
  sorry

end expr_comparison_l1146_114645


namespace skateboarder_speed_l1146_114610

-- Defining the conditions
def distance_feet : ℝ := 476.67
def time_seconds : ℝ := 25
def feet_per_mile : ℝ := 5280
def seconds_per_hour : ℝ := 3600

-- Defining the expected speed in miles per hour
def expected_speed_mph : ℝ := 13.01

-- The problem statement: Prove that the skateboarder's speed is 13.01 mph given the conditions
theorem skateboarder_speed : (distance_feet / feet_per_mile) / (time_seconds / seconds_per_hour) = expected_speed_mph := by
  sorry

end skateboarder_speed_l1146_114610


namespace mask_production_rates_l1146_114615

theorem mask_production_rates (x : ℝ) (y : ℝ) :
  (280 / x) - (280 / (1.4 * x)) = 2 →
  x = 40 ∧ y = 1.4 * x →
  y = 56 :=
by {
  sorry
}

end mask_production_rates_l1146_114615


namespace intersection_sums_l1146_114668

theorem intersection_sums :
  (∀ (x y : ℝ), (y = x^3 - 3 * x - 4) → (x + 3 * y = 3) → (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
  (y1 = x1^3 - 3 * x1 - 4) ∧ (x1 + 3 * y1 = 3) ∧
  (y2 = x2^3 - 3 * x2 - 4) ∧ (x2 + 3 * y2 = 3) ∧
  (y3 = x3^3 - 3 * x3 - 4) ∧ (x3 + 3 * y3 = 3) ∧
  x1 + x2 + x3 = 8 / 3 ∧ y1 + y2 + y3 = 19 / 9)) :=
sorry

end intersection_sums_l1146_114668


namespace percentage_tax_raise_expecting_population_l1146_114657

def percentage_affirmative_responses_tax : ℝ := 0.4
def percentage_affirmative_responses_money : ℝ := 0.3
def percentage_affirmative_responses_bonds : ℝ := 0.5
def percentage_affirmative_responses_gold : ℝ := 0.0

def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 1 - fraction_liars

theorem percentage_tax_raise_expecting_population : 
  (percentage_affirmative_responses_tax - fraction_liars) = 0.3 :=
by
  sorry

end percentage_tax_raise_expecting_population_l1146_114657


namespace functional_equation_solution_l1146_114630

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by
  sorry

end functional_equation_solution_l1146_114630


namespace Liam_chapters_in_fourth_week_l1146_114680

noncomputable def chapters_in_first_week (x : ℕ) : ℕ := x
noncomputable def chapters_in_second_week (x : ℕ) : ℕ := x + 3
noncomputable def chapters_in_third_week (x : ℕ) : ℕ := x + 6
noncomputable def chapters_in_fourth_week (x : ℕ) : ℕ := x + 9
noncomputable def total_chapters (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9)

theorem Liam_chapters_in_fourth_week : ∃ x : ℕ, total_chapters x = 50 → chapters_in_fourth_week x = 17 :=
by
  sorry

end Liam_chapters_in_fourth_week_l1146_114680


namespace invitation_methods_l1146_114689

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem invitation_methods (A B : Type) (students : Finset Type) (h : students.card = 10) :
  (∃ s : Finset Type, s.card = 6 ∧ A ∉ s ∧ B ∉ s) ∧ 
  (∃ t : Finset Type, t.card = 6 ∧ (A ∈ t ∨ B ∉ t)) →
  (combination 10 6 - combination 8 4 = 140) :=
by
  sorry

end invitation_methods_l1146_114689


namespace consistent_system_l1146_114612

variable (x y : ℕ)

def condition1 := x + y = 40
def condition2 := 2 * 15 * x = 20 * y

theorem consistent_system :
  condition1 x y ∧ condition2 x y ↔ 
  (x + y = 40 ∧ 2 * 15 * x = 20 * y) :=
by
  sorry

end consistent_system_l1146_114612


namespace pseudocode_output_l1146_114634

theorem pseudocode_output :
  let s := 0
  let t := 1
  let (s, t) := (List.range 3).foldl (fun (s, t) i => (s + (i + 1), t * (i + 1))) (s, t)
  let r := s * t
  r = 36 :=
by
  sorry

end pseudocode_output_l1146_114634


namespace average_speed_monkey_l1146_114643

def monkeyDistance : ℝ := 2160
def monkeyTimeMinutes : ℝ := 30
def monkeyTimeSeconds : ℝ := monkeyTimeMinutes * 60

theorem average_speed_monkey :
  (monkeyDistance / monkeyTimeSeconds) = 1.2 := 
sorry

end average_speed_monkey_l1146_114643


namespace trig_inequality_2016_l1146_114647

theorem trig_inequality_2016 :
  let a := Real.sin (Real.cos (2016 * Real.pi / 180))
  let b := Real.sin (Real.sin (2016 * Real.pi / 180))
  let c := Real.cos (Real.sin (2016 * Real.pi / 180))
  let d := Real.cos (Real.cos (2016 * Real.pi / 180))
  c > d ∧ d > b ∧ b > a := by
  sorry

end trig_inequality_2016_l1146_114647


namespace distance_traveled_by_second_hand_l1146_114636

theorem distance_traveled_by_second_hand (r : ℝ) (minutes : ℝ) (h1 : r = 10) (h2 : minutes = 45) :
  (2 * Real.pi * r) * (minutes / 1) = 900 * Real.pi := by
  -- Given:
  -- r = length of the second hand = 10 cm
  -- minutes = 45
  -- To prove: distance traveled by the tip = 900π cm
  sorry

end distance_traveled_by_second_hand_l1146_114636


namespace complement_union_M_N_l1146_114658

universe u

namespace complement_union

def U : Set (ℝ × ℝ) := { p | true }

def M : Set (ℝ × ℝ) := { p | (p.2 - 3) = (p.1 - 2) }

def N : Set (ℝ × ℝ) := { p | p.2 ≠ (p.1 + 1) }

theorem complement_union_M_N : (U \ (M ∪ N)) = { (2, 3) } := 
by 
  sorry

end complement_union

end complement_union_M_N_l1146_114658


namespace percentage_needed_to_pass_l1146_114622

-- Define conditions
def student_score : ℕ := 80
def marks_shortfall : ℕ := 40
def total_marks : ℕ := 400

-- Theorem statement: The percentage of marks required to pass the test.
theorem percentage_needed_to_pass : (student_score + marks_shortfall) * 100 / total_marks = 30 := by
  sorry

end percentage_needed_to_pass_l1146_114622


namespace mr_johnson_needs_additional_volunteers_l1146_114642

-- Definitions for the given conditions
def math_classes := 5
def students_per_class := 4
def total_students := math_classes * students_per_class

def total_teachers := 10
def carpentry_skilled_teachers := 3

def total_parents := 15
def lighting_sound_experienced_parents := 6

def total_volunteers_needed := 100
def carpentry_volunteers_needed := 8
def lighting_sound_volunteers_needed := 10

-- Total current volunteers
def current_volunteers := total_students + total_teachers + total_parents

-- Volunteers with specific skills
def current_carpentry_skilled := carpentry_skilled_teachers
def current_lighting_sound_experienced := lighting_sound_experienced_parents

-- Additional volunteers needed
def additional_carpentry_needed :=
  carpentry_volunteers_needed - current_carpentry_skilled
def additional_lighting_sound_needed :=
  lighting_sound_volunteers_needed - current_lighting_sound_experienced

-- Total additional volunteer needed
def additional_volunteers_needed :=
  additional_carpentry_needed + additional_lighting_sound_needed

-- The theorem we need to prove:
theorem mr_johnson_needs_additional_volunteers :
  additional_volunteers_needed = 9 := by
  sorry

end mr_johnson_needs_additional_volunteers_l1146_114642


namespace problem_solution_l1146_114637

variable {f : ℕ → ℕ}
variable (h_mul : ∀ a b : ℕ, f (a + b) = f a * f b)
variable (h_one : f 1 = 2)

theorem problem_solution : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) + (f 8 / f 7) + (f 10 / f 9) = 10 :=
by
  sorry

end problem_solution_l1146_114637


namespace cannot_determine_congruency_l1146_114632

-- Define the congruency criteria for triangles
def SSS (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ c1 = c2
def SAS (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2
def ASA (angle1 b1 angle2 angle3 b2 angle4 : ℝ) : Prop := angle1 = angle2 ∧ b1 = b2 ∧ angle3 = angle4
def AAS (angle1 angle2 b1 angle3 angle4 b2 : ℝ) : Prop := angle1 = angle2 ∧ angle3 = angle4 ∧ b1 = b2
def HL (hyp1 leg1 hyp2 leg2 : ℝ) : Prop := hyp1 = hyp2 ∧ leg1 = leg2

-- Define the condition D, which states the equality of two corresponding sides and a non-included angle
def conditionD (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2

-- The theorem to be proven
theorem cannot_determine_congruency (a1 b1 angle1 a2 b2 angle2 : ℝ) :
  conditionD a1 b1 angle1 a2 b2 angle2 → ¬(SSS a1 b1 0 a2 b2 0 ∨ SAS a1 b1 0 a2 b2 0 ∨ ASA 0 b1 0 0 b2 0 ∨ AAS 0 0 b1 0 0 b2 ∨ HL 0 0 0 0) :=
by
  sorry

end cannot_determine_congruency_l1146_114632


namespace solution_set_of_log_inequality_l1146_114604

noncomputable def log_a (a x : ℝ) : ℝ := sorry -- The precise definition of the log base 'a' is skipped for brevity.

theorem solution_set_of_log_inequality (a x : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (h_max : ∃ y, log_a a (y^2 - 2*y + 3) = y):
  log_a a (x - 1) > 0 ↔ (1 < x ∧ x < 2) :=
sorry

end solution_set_of_log_inequality_l1146_114604


namespace correct_multiplication_factor_l1146_114640

theorem correct_multiplication_factor (x : ℕ) : ((139 * x) - 1251 = 139 * 34) → x = 43 := by
  sorry

end correct_multiplication_factor_l1146_114640


namespace certain_number_modulo_l1146_114683

theorem certain_number_modulo (x : ℕ) : (57 * x) % 8 = 7 ↔ x = 1 := by
  sorry

end certain_number_modulo_l1146_114683


namespace vegetarian_family_l1146_114672

theorem vegetarian_family (eat_veg eat_non_veg eat_both : ℕ) (total_veg : ℕ) 
  (h1 : eat_non_veg = 8) (h2 : eat_both = 11) (h3 : total_veg = 26)
  : eat_veg = total_veg - eat_both := by
  sorry

end vegetarian_family_l1146_114672


namespace geometric_progression_sum_l1146_114629

theorem geometric_progression_sum (a q : ℝ) :
  (a + a * q^2 + a * q^4 = 63) →
  (a * q + a * q^3 = 30) →
  (a = 3 ∧ q = 2) ∨ (a = 48 ∧ q = 1 / 2) :=
by
  intro h1 h2
  sorry

end geometric_progression_sum_l1146_114629


namespace coloring_ways_l1146_114600

def num_colorings (total_circles blue_circles green_circles red_circles : ℕ) : ℕ :=
  if total_circles = blue_circles + green_circles + red_circles then
    (Nat.choose total_circles (green_circles + red_circles)) * (Nat.factorial (green_circles + red_circles) / (Nat.factorial green_circles * Nat.factorial red_circles))
  else
    0

theorem coloring_ways :
  num_colorings 6 4 1 1 = 30 :=
by sorry

end coloring_ways_l1146_114600


namespace last_digit_of_7_power_7_power_7_l1146_114623

theorem last_digit_of_7_power_7_power_7 : (7 ^ (7 ^ 7)) % 10 = 3 :=
by
  sorry

end last_digit_of_7_power_7_power_7_l1146_114623


namespace simplify_expression_l1146_114620

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 2) * (5 * x ^ 12 - 3 * x ^ 11 + 2 * x ^ 9 - x ^ 6) =
  15 * x ^ 13 - 19 * x ^ 12 - 6 * x ^ 11 + 6 * x ^ 10 - 4 * x ^ 9 - 3 * x ^ 7 + 2 * x ^ 6 :=
by
  sorry

end simplify_expression_l1146_114620


namespace complex_modulus_squared_l1146_114676

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6 * Complex.I) : Complex.abs z^2 = 13 / 2 :=
by
  sorry

end complex_modulus_squared_l1146_114676


namespace part1_l1146_114626

theorem part1 (a x0 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a ^ x0 = 2) : a ^ (3 * x0) = 8 := by
  sorry

end part1_l1146_114626


namespace new_sequence_after_removal_is_geometric_l1146_114639

theorem new_sequence_after_removal_is_geometric (a : ℕ → ℝ) (a₁ q : ℝ) (k : ℕ)
  (h_geo : ∀ n, a n = a₁ * q ^ n) :
  ∀ n, (a (n + k)) = a₁ * q ^ (n + k) :=
by
  sorry

end new_sequence_after_removal_is_geometric_l1146_114639


namespace andy_cavities_l1146_114635

def candy_canes_from_parents : ℕ := 2
def candy_canes_per_teacher : ℕ := 3
def number_of_teachers : ℕ := 4
def fraction_to_buy : ℚ := 1 / 7
def cavities_per_candies : ℕ := 4

theorem andy_cavities : (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers 
                         + (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers) * fraction_to_buy)
                         / cavities_per_candies = 4 := by
  sorry

end andy_cavities_l1146_114635
