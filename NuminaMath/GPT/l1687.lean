import Mathlib

namespace tan_105_eq_neg2_sub_sqrt3_l1687_168711

-- Define the main theorem to be proven
theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l1687_168711


namespace max_correct_answers_l1687_168785

theorem max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 25) 
  (h2 : 5 * c - 2 * w = 60) : 
  c ≤ 14 := 
sorry

end max_correct_answers_l1687_168785


namespace n_minus_m_eq_zero_l1687_168743

-- Definitions based on the conditions
def m : ℝ := sorry
def n : ℝ := sorry
def i := Complex.I
def condition : Prop := m + i = (1 + 2 * i) - n * i

-- The theorem stating the equivalence proof problem
theorem n_minus_m_eq_zero (h : condition) : n - m = 0 :=
sorry

end n_minus_m_eq_zero_l1687_168743


namespace more_wrappers_than_bottle_caps_at_park_l1687_168797

-- Define the number of bottle caps and wrappers found at the park.
def bottle_caps_found : ℕ := 11
def wrappers_found : ℕ := 28

-- State the theorem to prove the number of more wrappers than bottle caps found at the park is 17.
theorem more_wrappers_than_bottle_caps_at_park : wrappers_found - bottle_caps_found = 17 :=
by
  -- proof goes here
  sorry

end more_wrappers_than_bottle_caps_at_park_l1687_168797


namespace foil_covered_prism_width_l1687_168792

def inner_prism_length (l : ℝ) := l
def inner_prism_width (l : ℝ) := 2 * l
def inner_prism_height (l : ℝ) := l
def inner_prism_volume (l : ℝ) := l * (2 * l) * l

theorem foil_covered_prism_width :
  (∃ l : ℝ, inner_prism_volume l = 128) → (inner_prism_width l + 2 = 8) := by
sorry

end foil_covered_prism_width_l1687_168792


namespace find_larger_number_l1687_168740

-- Define the conditions
variables (L S : ℕ)

theorem find_larger_number (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l1687_168740


namespace least_m_lcm_l1687_168705

theorem least_m_lcm (m : ℕ) (h : m > 0) : Nat.lcm 15 m = Nat.lcm 42 m → m = 70 := by
  sorry

end least_m_lcm_l1687_168705


namespace triangle_side_ratio_l1687_168706

theorem triangle_side_ratio (a b c: ℝ) (A B C: ℝ) (h1: b * Real.cos C + c * Real.cos B = 2 * b) :
  a / b = 2 :=
sorry

end triangle_side_ratio_l1687_168706


namespace distance_between_first_and_last_tree_l1687_168720

theorem distance_between_first_and_last_tree
  (n : ℕ) (d_1_5 : ℝ) (h1 : n = 8) (h2 : d_1_5 = 100) :
  let interval_distance := d_1_5 / 4
  let total_intervals := n - 1
  let total_distance := interval_distance * total_intervals
  total_distance = 175 :=
by
  sorry

end distance_between_first_and_last_tree_l1687_168720


namespace least_common_multiple_of_20_45_75_l1687_168779

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l1687_168779


namespace start_page_day2_correct_l1687_168754

variables (total_pages : ℕ) (percentage_read_day1 : ℝ) (start_page_day2 : ℕ)

theorem start_page_day2_correct
  (h1 : total_pages = 200)
  (h2 : percentage_read_day1 = 0.2)
  : start_page_day2 = total_pages * percentage_read_day1 + 1 :=
by
  sorry

end start_page_day2_correct_l1687_168754


namespace miles_driven_l1687_168733

def total_miles : ℕ := 1200
def remaining_miles : ℕ := 432

theorem miles_driven : total_miles - remaining_miles = 768 := by
  sorry

end miles_driven_l1687_168733


namespace seventh_fisherman_right_neighbor_l1687_168780

theorem seventh_fisherman_right_neighbor (f1 f2 f3 f4 f5 f6 f7 : ℕ) (L1 L2 L3 L4 L5 L6 L7 : ℕ) :
  (L2 * f1 = 12 ∨ L3 * f2 = 12 ∨ L4 * f3 = 12 ∨ L5 * f4 = 12 ∨ L6 * f5 = 12 ∨ L7 * f6 = 12 ∨ L1 * f7 = 12) → 
  (L2 * f1 = 14 ∨ L3 * f2 = 18 ∨ L4 * f3 = 32 ∨ L5 * f4 = 48 ∨ L6 * f5 = 70 ∨ L7 * f6 = x ∨ L1 * f7 = 12) →
  (12 * 12 * 20 * 24 * 32 * 42 * 56) / (12 * 14 * 18 * 32 * 48 * 70) = x :=
by
  sorry

end seventh_fisherman_right_neighbor_l1687_168780


namespace max_area_parabola_l1687_168761

open Real

noncomputable def max_area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem max_area_parabola (a b c : ℝ) 
  (ha : a^2 = (a * a))
  (hb : b^2 = (b * b))
  (hc : c^2 = (c * c))
  (centroid_cond1 : (a + b + c) = 4)
  (centroid_cond2 : (a^2 + b^2 + c^2) = 6)
  : max_area_of_triangle (a^2, a) (b^2, b) (c^2, c) = (sqrt 3) / 9 := 
sorry

end max_area_parabola_l1687_168761


namespace adjacent_sum_constant_l1687_168769

theorem adjacent_sum_constant (x y : ℤ) (k : ℤ) (h1 : 2 + x = k) (h2 : x + y = k) (h3 : y + 5 = k) : x - y = 3 := 
by 
  sorry

end adjacent_sum_constant_l1687_168769


namespace joshua_finishes_after_malcolm_l1687_168717

-- Definitions based on conditions.
def malcolm_speed : ℕ := 6 -- Malcolm's speed in minutes per mile
def joshua_speed : ℕ := 8 -- Joshua's speed in minutes per mile
def race_distance : ℕ := 10 -- Race distance in miles

-- Theorem: How many minutes after Malcolm crosses the finish line will Joshua cross the finish line?
theorem joshua_finishes_after_malcolm :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 20 :=
by
  -- sorry is a placeholder for the proof
  sorry

end joshua_finishes_after_malcolm_l1687_168717


namespace min_fraction_sum_l1687_168762

theorem min_fraction_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  (∀ x, x = 1 / m + 2 / n → x ≥ 8) :=
  sorry

end min_fraction_sum_l1687_168762


namespace find_C_D_l1687_168766

theorem find_C_D : ∃ C D, 
  (∀ x, x ≠ 3 → x ≠ 5 → (6*x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) ∧ 
  C = -15/2 ∧ D = 27/2 := by
  sorry

end find_C_D_l1687_168766


namespace proposition_false_at_6_l1687_168772

variable (P : ℕ → Prop)

theorem proposition_false_at_6 (h1 : ∀ k : ℕ, 0 < k → P k → P (k + 1)) (h2 : ¬P 7): ¬P 6 :=
by
  sorry

end proposition_false_at_6_l1687_168772


namespace quadratic_solution_range_l1687_168793

theorem quadratic_solution_range :
  ∃ x : ℝ, x^2 + 12 * x - 15 = 0 ∧ 1.1 < x ∧ x < 1.2 :=
sorry

end quadratic_solution_range_l1687_168793


namespace gp_sum_l1687_168732

theorem gp_sum (x : ℕ) (h : (30 + x) / (10 + x) = (60 + x) / (30 + x)) :
  x = 30 ∧ (10 + x) + (30 + x) + (60 + x) + (120 + x) = 340 :=
by {
  sorry
}

end gp_sum_l1687_168732


namespace one_statement_is_true_l1687_168708

theorem one_statement_is_true :
  ∃ (S1 S2 S3 S4 S5 : Prop),
    ((S1 ↔ (¬S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5)) ∧
     (S2 ↔ (¬S1 ∧ ¬S2 ∧ S3 ∧ S4 ∧ ¬S5)) ∧
     (S3 ↔ (¬S1 ∧ S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S4 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ S4 ∧ ¬S5)) ∧
     (S5 ↔ (¬S1 ∧ ¬S2 ∧ ¬S3 ∧ ¬S4 ∧ ¬S5))) ∧
    (S2) ∧ (¬S1) ∧ (¬S3) ∧ (¬S4) ∧ (¬S5) :=
by
  -- Proof goes here
  sorry

end one_statement_is_true_l1687_168708


namespace overhead_cost_calculation_l1687_168755

-- Define the production cost per performance
def production_cost_performance : ℕ := 7000

-- Define the revenue per sold-out performance
def revenue_per_soldout_performance : ℕ := 16000

-- Define the number of performances needed to break even
def break_even_performances : ℕ := 9

-- Prove the overhead cost
theorem overhead_cost_calculation (O : ℕ) :
  (O + break_even_performances * production_cost_performance = break_even_performances * revenue_per_soldout_performance) →
  O = 81000 :=
by
  sorry

end overhead_cost_calculation_l1687_168755


namespace neg_exists_is_forall_l1687_168782

theorem neg_exists_is_forall: 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by
  sorry

end neg_exists_is_forall_l1687_168782


namespace normal_distribution_interval_probability_l1687_168787

noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ :=
sorry

theorem normal_distribution_interval_probability
  (σ : ℝ) (hσ : σ > 0)
  (hprob : normal_cdf 1 σ 2 - normal_cdf 1 σ 0 = 0.8) :
  (normal_cdf 1 σ 2 - normal_cdf 1 σ 1) = 0.4 :=
sorry

end normal_distribution_interval_probability_l1687_168787


namespace students_per_class_l1687_168775

-- Define the conditions
variables (c : ℕ) (h_c : c ≥ 1) (s : ℕ)

-- Define the total number of books read by one student per year
def books_per_student_per_year := 5 * 12

-- Define the total number of students
def total_number_of_students := c * s

-- Define the total number of books read by the entire student body
def total_books_read := total_number_of_students * books_per_student_per_year

-- The given condition that the entire student body reads 60 books in one year
axiom total_books_eq_60 : total_books_read = 60

theorem students_per_class (h_c : c ≥ 1) : s = 1 / c :=
by sorry

end students_per_class_l1687_168775


namespace necessary_but_not_sufficient_l1687_168713

def p (x : ℝ) : Prop := x ^ 2 = 3 * x + 4
def q (x : ℝ) : Prop := x = Real.sqrt (3 * x + 4)

theorem necessary_but_not_sufficient (x : ℝ) : (p x → q x) ∧ ¬ (q x → p x) := by
  sorry

end necessary_but_not_sufficient_l1687_168713


namespace f_eq_f_inv_l1687_168707

noncomputable def f (x : ℝ) : ℝ := 3 * x - 7

noncomputable def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv (x : ℝ) : f x = f_inv x ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_l1687_168707


namespace daphney_potatoes_l1687_168739

theorem daphney_potatoes (cost_per_2kg : ℕ) (total_paid : ℕ) (amount_per_kg : ℕ) (kg_bought : ℕ) 
  (h1 : cost_per_2kg = 6) (h2 : total_paid = 15) (h3 : amount_per_kg = cost_per_2kg / 2) 
  (h4 : kg_bought = total_paid / amount_per_kg) : kg_bought = 5 :=
by
  sorry

end daphney_potatoes_l1687_168739


namespace find_number_l1687_168778

theorem find_number (x : ℤ) : (150 - x = x + 68) → x = 41 :=
by
  intro h
  sorry

end find_number_l1687_168778


namespace minimum_distance_l1687_168723

def curve1 (x y : ℝ) : Prop := y^2 - 9 + 2*y*x - 12*x - 3*x^2 = 0
def curve2 (x y : ℝ) : Prop := y^2 + 3 - 4*x - 2*y + x^2 = 0

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem minimum_distance 
  (A B : ℝ × ℝ) 
  (hA : curve1 A.1 A.2) 
  (hB : curve2 B.1 B.2) : 
  ∃ d, d = 2 * Real.sqrt 2 ∧ (∀ P Q : ℝ × ℝ, curve1 P.1 P.2 → curve2 Q.1 Q.2 → distance P.1 P.2 Q.1 Q.2 ≥ d) :=
sorry

end minimum_distance_l1687_168723


namespace difference_in_surface_areas_l1687_168727

-- Define the conditions: volumes and number of cubes
def V_large : ℕ := 343
def n : ℕ := 343
def V_small : ℕ := 1

-- Define the function to calculate the side length of a cube given its volume
def side_length (V : ℕ) : ℕ := V^(1/3 : ℕ)

-- Specify the side lengths of the larger and smaller cubes
def s_large : ℕ := side_length V_large
def s_small : ℕ := side_length V_small

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- Specify the surface areas of the larger cube and the total of the smaller cubes
def SA_large : ℕ := surface_area s_large
def SA_small_total : ℕ := n * surface_area s_small

-- State the theorem to prove
theorem difference_in_surface_areas : SA_small_total - SA_large = 1764 :=
by {
  -- Intentionally omit proof, as per instructions
  sorry
}

end difference_in_surface_areas_l1687_168727


namespace alan_total_payment_l1687_168746

-- Define the costs of CDs
def cost_AVN : ℝ := 12
def cost_TheDark : ℝ := 2 * cost_AVN
def cost_TheDark_total : ℝ := 2 * cost_TheDark
def cost_other_CDs : ℝ := cost_AVN + cost_TheDark_total
def cost_90s : ℝ := 0.4 * cost_other_CDs
def total_cost : ℝ := cost_AVN + cost_TheDark_total + cost_90s

-- Formulate the main statement
theorem alan_total_payment :
  total_cost = 84 := by
  sorry

end alan_total_payment_l1687_168746


namespace expression_evaluation_l1687_168768

-- Define the property to be proved:
theorem expression_evaluation (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by sorry

end expression_evaluation_l1687_168768


namespace log_expression_value_l1687_168777

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_value :
  log10 8 + 3 * log10 4 - 2 * log10 2 + 4 * log10 25 + log10 16 = 11 := by
  sorry

end log_expression_value_l1687_168777


namespace sym_diff_A_B_l1687_168726

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- Definition of the symmetric difference
def sym_diff (A B : Set ℕ) : Set ℕ := {x | (x ∈ A ∨ x ∈ B) ∧ x ∉ (A ∩ B)}

theorem sym_diff_A_B : sym_diff A B = {0, 3} := 
by 
  sorry

end sym_diff_A_B_l1687_168726


namespace no_solutions_in_naturals_l1687_168747

theorem no_solutions_in_naturals (n k : ℕ) : ¬ (n ≤ n! - k^n ∧ n! - k^n ≤ k * n) :=
sorry

end no_solutions_in_naturals_l1687_168747


namespace growth_rate_equation_l1687_168774

-- Given conditions
def revenue_january : ℕ := 36
def revenue_march : ℕ := 48

-- Problem statement
theorem growth_rate_equation (x : ℝ) 
  (h_january : revenue_january = 36)
  (h_march : revenue_march = 48) :
  36 * (1 + x) ^ 2 = 48 :=
sorry

end growth_rate_equation_l1687_168774


namespace bananas_used_l1687_168750

-- Define the conditions
def bananas_per_loaf := 4
def loaves_monday := 3
def loaves_tuesday := 2 * loaves_monday

-- Define the total bananas used
def bananas_monday := loaves_monday * bananas_per_loaf
def bananas_tuesday := loaves_tuesday * bananas_per_loaf
def total_bananas := bananas_monday + bananas_tuesday

-- Theorem statement to prove the total bananas used is 36
theorem bananas_used : total_bananas = 36 := by
  sorry

end bananas_used_l1687_168750


namespace arithmetic_sequence_sum_l1687_168703

theorem arithmetic_sequence_sum {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h₀ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h₁ : S 9 = 27) :
  (a 4 + a 6) = 6 :=
sorry

end arithmetic_sequence_sum_l1687_168703


namespace hexagon_piece_area_l1687_168791

theorem hexagon_piece_area (A : ℝ) (n : ℕ) (h1 : A = 21.12) (h2 : n = 6) : 
  A / n = 3.52 :=
by
  -- The proof will go here
  sorry

end hexagon_piece_area_l1687_168791


namespace coordinates_satisfy_l1687_168722

theorem coordinates_satisfy (x y : ℝ) : y * (x + 1) = x^2 - 1 ↔ (x = -1 ∨ y = x - 1) :=
by
  sorry

end coordinates_satisfy_l1687_168722


namespace min_slope_at_a_half_l1687_168715

theorem min_slope_at_a_half (a : ℝ) (h : 0 < a) :
  (∀ b : ℝ, 0 < b → 4 * b + 1 / b ≥ 4) → (4 * a + 1 / a = 4) → a = 1 / 2 :=
by
  sorry

end min_slope_at_a_half_l1687_168715


namespace sum_of_integers_l1687_168734

theorem sum_of_integers (n : ℤ) (h : n * (n + 2) = 20400) : n + (n + 2) = 286 ∨ n + (n + 2) = -286 :=
by
  sorry

end sum_of_integers_l1687_168734


namespace taxi_ride_cost_l1687_168748

-- Define the fixed cost
def fixed_cost : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the number of miles traveled
def miles_traveled : ℝ := 7.0

-- Define the total cost calculation
def total_cost : ℝ := fixed_cost + (cost_per_mile * miles_traveled)

-- Theorem: Prove the total cost of a 7-mile taxi ride is $4.10
theorem taxi_ride_cost : total_cost = 4.10 := by
  sorry

end taxi_ride_cost_l1687_168748


namespace mr_johnson_pill_intake_l1687_168728

theorem mr_johnson_pill_intake (total_days : ℕ) (remaining_pills : ℕ) (fraction : ℚ) (dose : ℕ)
  (h1 : total_days = 30)
  (h2 : remaining_pills = 12)
  (h3 : fraction = 4 / 5) :
  dose = 2 :=
by
  sorry

end mr_johnson_pill_intake_l1687_168728


namespace evaluate_g_at_neg_one_l1687_168701

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 - 3 * x + 9

theorem evaluate_g_at_neg_one : g (-1) = 7 :=
by 
  -- lean proof here
  sorry

end evaluate_g_at_neg_one_l1687_168701


namespace petya_catch_bus_l1687_168788

theorem petya_catch_bus 
    (v_p v_b d : ℝ) 
    (h1 : v_b = 5 * v_p)
    (h2 : ∀ t : ℝ, 5 * v_p * t ≤ 0.6) 
    : d = 0.12 := 
sorry

end petya_catch_bus_l1687_168788


namespace no_int_solutions_p_mod_4_neg_1_l1687_168784

theorem no_int_solutions_p_mod_4_neg_1 :
  ∀ (p n : ℕ), (p % 4 = 3) → (∀ x y : ℕ, x^2 + y^2 ≠ p^n) :=
by
  intros
  sorry

end no_int_solutions_p_mod_4_neg_1_l1687_168784


namespace angle_in_second_quadrant_l1687_168702

theorem angle_in_second_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
    α ∈ Set.Ioo (π / 2) π := 
    sorry

end angle_in_second_quadrant_l1687_168702


namespace total_crayons_l1687_168731

def original_crayons := 41
def added_crayons := 12

theorem total_crayons : original_crayons + added_crayons = 53 := by
  sorry

end total_crayons_l1687_168731


namespace difference_between_numbers_l1687_168795

variable (x y : ℕ)

theorem difference_between_numbers (h1 : x + y = 34) (h2 : y = 22) : y - x = 10 := by
  sorry

end difference_between_numbers_l1687_168795


namespace candidate_valid_vote_percentage_l1687_168741

theorem candidate_valid_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_votes : ℕ) 
  (valid_percentage : ℚ)
  (total_votes_eq : total_votes = 560000)
  (invalid_percentage_eq : invalid_percentage = 15 / 100)
  (candidate_votes_eq : candidate_votes = 357000)
  (valid_percentage_eq : valid_percentage = 85 / 100) :
  (candidate_votes / (total_votes * valid_percentage)) * 100 = 75 := 
by
  sorry

end candidate_valid_vote_percentage_l1687_168741


namespace smallest_number_divisible_l1687_168771

theorem smallest_number_divisible (n : ℤ) : 
  (n + 7) % 25 = 0 ∧
  (n + 7) % 49 = 0 ∧
  (n + 7) % 15 = 0 ∧
  (n + 7) % 21 = 0 ↔ n = 3668 :=
by 
 sorry

end smallest_number_divisible_l1687_168771


namespace radical_product_l1687_168719

theorem radical_product :
  (64 ^ (1 / 3) * 16 ^ (1 / 4) * 64 ^ (1 / 6) = 16) :=
by
  sorry

end radical_product_l1687_168719


namespace probability_not_all_same_color_l1687_168729

def num_colors := 3
def draws := 3
def total_outcomes := num_colors ^ draws

noncomputable def prob_same_color : ℚ := (3 / total_outcomes)
noncomputable def prob_not_same_color : ℚ := 1 - prob_same_color

theorem probability_not_all_same_color :
  prob_not_same_color = 8 / 9 :=
by
  sorry

end probability_not_all_same_color_l1687_168729


namespace probability_prime_sum_l1687_168730

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end probability_prime_sum_l1687_168730


namespace top_angle_is_70_l1687_168710

theorem top_angle_is_70
  (sum_angles : ℝ)
  (left_angle : ℝ)
  (right_angle : ℝ)
  (top_angle : ℝ)
  (h1 : sum_angles = 250)
  (h2 : left_angle = 2 * right_angle)
  (h3 : right_angle = 60)
  (h4 : sum_angles = left_angle + right_angle + top_angle) :
  top_angle = 70 :=
by
  sorry

end top_angle_is_70_l1687_168710


namespace fish_added_l1687_168749

theorem fish_added (x : ℕ) (hx : x + (x - 4) = 20) : x - 4 = 8 := by
  sorry

end fish_added_l1687_168749


namespace Robert_more_than_Claire_l1687_168725

variable (Lisa Claire Robert : ℕ)

theorem Robert_more_than_Claire (h1 : Lisa = 3 * Claire) (h2 : Claire = 10) (h3 : Robert > Claire) :
  Robert > 10 :=
by
  rw [h2] at h3
  assumption

end Robert_more_than_Claire_l1687_168725


namespace rectangle_area_ratio_l1687_168789

theorem rectangle_area_ratio (length width diagonal : ℝ) (h_ratio : length / width = 5 / 2) (h_diagonal : diagonal = 13) :
    ∃ k : ℝ, (length * width) = k * diagonal^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_ratio_l1687_168789


namespace visited_neither_l1687_168767

theorem visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) 
  (h1 : total = 100) 
  (h2 : iceland = 55) 
  (h3 : norway = 43) 
  (h4 : both = 61) : 
  (total - (iceland + norway - both)) = 63 := 
by 
  sorry

end visited_neither_l1687_168767


namespace area_difference_l1687_168738

-- Definitions of the conditions
def length_rect := 60 -- length of the rectangular garden in feet
def width_rect := 20 -- width of the rectangular garden in feet

-- Compute the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Compute the perimeter of the rectangular garden
def perimeter_rect := 2 * (length_rect + width_rect)

-- Compute the side length of the square garden from the same perimeter
def side_square := perimeter_rect / 4

-- Compute the area of the square garden
def area_square := side_square * side_square

-- The goal is to prove the area difference
theorem area_difference : area_square - area_rect = 400 := by
  sorry -- Proof to be completed

end area_difference_l1687_168738


namespace probability_age_20_to_40_l1687_168745

theorem probability_age_20_to_40 
    (total_people : ℕ) (aged_20_to_30 : ℕ) (aged_30_to_40 : ℕ) 
    (h_total : total_people = 350) 
    (h_aged_20_to_30 : aged_20_to_30 = 105) 
    (h_aged_30_to_40 : aged_30_to_40 = 85) : 
    (190 / 350 : ℚ) = 19 / 35 := 
by 
  sorry

end probability_age_20_to_40_l1687_168745


namespace range_of_m_l1687_168751

theorem range_of_m (m : ℝ) (p : Prop) (q : Prop)
  (hp : (2 * m)^2 - 4 ≥ 0 ↔ p)
  (hq : 1 < (Real.sqrt (5 + m)) / (Real.sqrt 5) ∧ (Real.sqrt (5 + m)) / (Real.sqrt 5) < 2 ↔ q)
  (hnq : ¬q = False)
  (hpq : (p ∧ q) = False) :
  0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l1687_168751


namespace reciprocal_of_3_div_2_l1687_168735

def reciprocal (a : ℚ) : ℚ := a⁻¹

theorem reciprocal_of_3_div_2 : reciprocal (3 / 2) = 2 / 3 :=
by
  -- proof would go here
  sorry

end reciprocal_of_3_div_2_l1687_168735


namespace rationalize_denominator_l1687_168712

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end rationalize_denominator_l1687_168712


namespace line_intersects_ellipse_possible_slopes_l1687_168724

theorem line_intersects_ellipse_possible_slopes :
  ∀ m : ℝ, (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔
    (m ≤ -Real.sqrt (1 / 20) ∨ m ≥ Real.sqrt (1 / 20)) :=
by
  sorry

end line_intersects_ellipse_possible_slopes_l1687_168724


namespace arithmetic_square_root_l1687_168700

theorem arithmetic_square_root (n : ℝ) (h : (-5)^2 = n) : Real.sqrt n = 5 :=
by
  sorry

end arithmetic_square_root_l1687_168700


namespace problem_l1687_168709

theorem problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (17 * x)) (x : ℝ) :
  f (Real.cos x) ^ 2 + f (Real.sin x) ^ 2 = 1 :=
sorry

end problem_l1687_168709


namespace average_price_per_book_l1687_168783

theorem average_price_per_book 
  (amount1 : ℝ)
  (books1 : ℕ)
  (amount2 : ℝ)
  (books2 : ℕ)
  (h1 : amount1 = 581)
  (h2 : books1 = 27)
  (h3 : amount2 = 594)
  (h4 : books2 = 20) :
  (amount1 + amount2) / (books1 + books2) = 25 := 
by
  sorry

end average_price_per_book_l1687_168783


namespace additional_discount_percentage_l1687_168752

def initial_price : ℝ := 2000
def gift_cards : ℝ := 200
def initial_discount_rate : ℝ := 0.15
def final_price : ℝ := 1330

theorem additional_discount_percentage :
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  additional_discount_percentage = 11.33 :=
by
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  show additional_discount_percentage = 11.33
  sorry

end additional_discount_percentage_l1687_168752


namespace total_gas_cost_l1687_168753

theorem total_gas_cost 
  (x : ℝ)
  (cost_per_person_initial : ℝ := x / 5)
  (cost_per_person_new : ℝ := x / 8)
  (cost_difference : cost_per_person_initial - cost_per_person_new = 15) :
  x = 200 :=
sorry

end total_gas_cost_l1687_168753


namespace sandy_marbles_correct_l1687_168794

namespace MarbleProblem

-- Define the number of dozens Jessica has
def jessica_dozens : ℕ := 3

-- Define the conversion from dozens to individual marbles
def dozens_to_marbles (d : ℕ) : ℕ := 12 * d

-- Calculate the number of marbles Jessica has
def jessica_marbles : ℕ := dozens_to_marbles jessica_dozens

-- Define the multiplier for Sandy's marbles
def sandy_multiplier : ℕ := 4

-- Define the number of marbles Sandy has
def sandy_marbles : ℕ := sandy_multiplier * jessica_marbles

theorem sandy_marbles_correct : sandy_marbles = 144 :=
by
  sorry

end MarbleProblem

end sandy_marbles_correct_l1687_168794


namespace intersection_A_B_l1687_168721

def A : Set ℤ := { x | (2 * x + 3) * (x - 4) < 0 }
def B : Set ℝ := { x | 0 < x ∧ x ≤ Real.exp 1 }

theorem intersection_A_B :
  { x : ℤ | x ∈ A ∧ (x : ℝ) ∈ B } = {1, 2} :=
by
  sorry

end intersection_A_B_l1687_168721


namespace ratio_of_roses_l1687_168796

-- Definitions for conditions
def roses_two_days_ago : ℕ := 50
def roses_yesterday : ℕ := roses_two_days_ago + 20
def roses_total : ℕ := 220
def roses_today : ℕ := roses_total - roses_two_days_ago - roses_yesterday

-- Lean statement to prove the ratio of roses planted today to two days ago is 2
theorem ratio_of_roses :
  roses_today / roses_two_days_ago = 2 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_roses_l1687_168796


namespace south_120_meters_l1687_168781

-- Define the directions
inductive Direction
| North
| South

-- Define the movement function
def movement (dir : Direction) (distance : Int) : Int :=
  match dir with
  | Direction.North => distance
  | Direction.South => -distance

-- Statement to prove
theorem south_120_meters : movement Direction.South 120 = -120 := 
by
  sorry

end south_120_meters_l1687_168781


namespace No_response_percentage_l1687_168799

theorem No_response_percentage (total_guests : ℕ) (yes_percentage : ℕ) (non_respondents : ℕ) (yes_guests := total_guests * yes_percentage / 100) (no_guests := total_guests - yes_guests - non_respondents) (no_percentage := no_guests * 100 / total_guests) :
  total_guests = 200 → yes_percentage = 83 → non_respondents = 16 → no_percentage = 9 :=
by
  sorry

end No_response_percentage_l1687_168799


namespace ratio_of_areas_of_similar_triangles_l1687_168786

-- Define the variables and conditions
variables {ABC DEF : Type} 
variables (hABCDEF : Similar ABC DEF) 
variables (perimeterABC perimeterDEF : ℝ)
variables (hpABC : perimeterABC = 3)
variables (hpDEF : perimeterDEF = 1)

-- The theorem statement
theorem ratio_of_areas_of_similar_triangles :
  (perimeterABC / perimeterDEF) ^ 2 = 9 :=
by
  sorry

end ratio_of_areas_of_similar_triangles_l1687_168786


namespace corrected_observations_mean_l1687_168765

noncomputable def corrected_mean (mean incorrect correct: ℚ) (n: ℕ) : ℚ :=
  let S_incorrect := mean * n
  let Difference := correct - incorrect
  let S_corrected := S_incorrect + Difference
  S_corrected / n

theorem corrected_observations_mean:
  corrected_mean 36 23 34 50 = 36.22 := by
  sorry

end corrected_observations_mean_l1687_168765


namespace value_of_expression_l1687_168759

theorem value_of_expression (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : 
  (x - 2 * y + 3 * z) / (x + y + z) = 8 / 9 := 
  sorry

end value_of_expression_l1687_168759


namespace cannot_obtain_100_pieces_l1687_168776

theorem cannot_obtain_100_pieces : ¬ ∃ n : ℕ, 1 + 2 * n = 100 := by
  sorry

end cannot_obtain_100_pieces_l1687_168776


namespace average_minutes_run_l1687_168773

-- Definitions
def third_graders (fi : ℕ) : ℕ := 6 * fi
def fourth_graders (fi : ℕ) : ℕ := 2 * fi
def fifth_graders (fi : ℕ) : ℕ := fi

-- Number of minutes run by each grade
def third_graders_minutes : ℕ := 10
def fourth_graders_minutes : ℕ := 18
def fifth_graders_minutes : ℕ := 8

-- Main theorem
theorem average_minutes_run 
  (fi : ℕ) 
  (t := third_graders fi) 
  (fr := fourth_graders fi) 
  (f := fifth_graders fi) 
  (minutes_total := 10 * t + 18 * fr + 8 * f) 
  (students_total := t + fr + f) :
  (students_total > 0) →
  (minutes_total : ℚ) / students_total = 104 / 9 :=
by
  sorry

end average_minutes_run_l1687_168773


namespace algebraic_expression_value_l1687_168758

theorem algebraic_expression_value (a b : ℤ) (h : 2 * (-3) - a + 2 * b = 0) : 2 * a - 4 * b + 1 = -11 := 
by {
  sorry
}

end algebraic_expression_value_l1687_168758


namespace shoveling_time_l1687_168704

theorem shoveling_time :
  let kevin_time := 12
  let dave_time := 8
  let john_time := 6
  let allison_time := 4
  let kevin_rate := 1 / kevin_time
  let dave_rate := 1 / dave_time
  let john_rate := 1 / john_time
  let allison_rate := 1 / allison_time
  let combined_rate := kevin_rate + dave_rate + john_rate + allison_rate
  let total_minutes := 60
  let combined_rate_per_minute := combined_rate / total_minutes
  (1 / combined_rate_per_minute = 96) := 
  sorry

end shoveling_time_l1687_168704


namespace int_solution_l1687_168756

theorem int_solution (n : ℕ) (h1 : n ≥ 1) (h2 : n^2 ∣ 2^n + 1) : n = 1 ∨ n = 3 :=
by
  sorry

end int_solution_l1687_168756


namespace stock_percentage_change_l1687_168718

theorem stock_percentage_change :
  let initial_value := 100
  let value_after_first_day := initial_value * (1 - 0.25)
  let value_after_second_day := value_after_first_day * (1 + 0.35)
  let final_value := value_after_second_day * (1 - 0.15)
  let overall_percentage_change := ((final_value - initial_value) / initial_value) * 100
  overall_percentage_change = -13.9375 := 
by
  sorry

end stock_percentage_change_l1687_168718


namespace least_side_is_8_l1687_168716

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l1687_168716


namespace number_of_pages_l1687_168736

-- Define the conditions
def rate_of_printer_A (P : ℕ) : ℕ := P / 60
def rate_of_printer_B (P : ℕ) : ℕ := (P / 60) + 6

-- Define the combined rate condition
def combined_rate (P : ℕ) (R_A R_B : ℕ) : Prop := (R_A + R_B) = P / 24

-- The main theorem to prove
theorem number_of_pages :
  ∃ (P : ℕ), combined_rate P (rate_of_printer_A P) (rate_of_printer_B P) ∧ P = 720 := by
  sorry

end number_of_pages_l1687_168736


namespace complex_expression_equality_l1687_168760

open Complex

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := 
sorry

end complex_expression_equality_l1687_168760


namespace number_of_valid_three_digit_numbers_l1687_168757

def three_digit_numbers_count : Nat :=
  let count_numbers (last_digit : Nat) (remaining_digits : List Nat) : Nat :=
    remaining_digits.length * (remaining_digits.erase last_digit).length

  let count_when_last_digit_is_0 :=
    count_numbers 0 [1, 2, 3, 4, 5, 6, 7, 8, 9]

  let count_when_last_digit_is_5 :=
    count_numbers 5 [0, 1, 2, 3, 4, 6, 7, 8, 9]

  count_when_last_digit_is_0 + count_when_last_digit_is_5

theorem number_of_valid_three_digit_numbers : three_digit_numbers_count = 136 := by
  sorry

end number_of_valid_three_digit_numbers_l1687_168757


namespace tangent_line_through_origin_l1687_168763

theorem tangent_line_through_origin (f : ℝ → ℝ) (x : ℝ) (H1 : ∀ x < 0, f x = Real.log (-x))
  (H2 : ∀ x < 0, DifferentiableAt ℝ f x) (H3 : ∀ (x₀ : ℝ), x₀ < 0 → x₀ = -Real.exp 1 → deriv f x₀ = -1 / Real.exp 1)
  : ∀ x, -Real.exp 1 = x → ∀ y, y = -1 / Real.exp 1 * x → y = 0 → y = -1 / Real.exp 1 * x :=
by
  sorry

end tangent_line_through_origin_l1687_168763


namespace hoseok_value_l1687_168742

theorem hoseok_value (x : ℕ) (h : x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end hoseok_value_l1687_168742


namespace find_X_l1687_168714

theorem find_X 
  (X Y : ℕ)
  (h1 : 6 + X = 13)
  (h2 : Y = 7) :
  X = 7 := by
  sorry

end find_X_l1687_168714


namespace geometric_progression_l1687_168744

theorem geometric_progression (b q : ℝ) :
  (b + b*q + b*q^2 + b*q^3 = -40) ∧ 
  (b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280) →
  (b = 2 ∧ q = -3) ∨ (b = -54 ∧ q = -1/3) :=
by sorry

end geometric_progression_l1687_168744


namespace intersection_of_A_and_B_l1687_168798

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, -1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_A_and_B_l1687_168798


namespace calculate_division_l1687_168770

theorem calculate_division : 
  (- (1 / 28)) / ((1 / 2) - (1 / 4) + (1 / 7) - (1 / 14)) = - (1 / 9) :=
by
  sorry

end calculate_division_l1687_168770


namespace reduced_price_is_16_l1687_168764

noncomputable def reduced_price_per_kg (P : ℝ) (r : ℝ) : ℝ :=
  0.9 * (P * (1 + r))

theorem reduced_price_is_16 (P r : ℝ) (h₀ : (0.9 : ℝ) * (P * (1 + r)) = 16) : 
  reduced_price_per_kg P r = 16 :=
by
  -- We have the hypothesis and we need to prove the result
  exact h₀

end reduced_price_is_16_l1687_168764


namespace factor_expression_l1687_168790

theorem factor_expression (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) :=
by
  sorry

end factor_expression_l1687_168790


namespace original_average_speed_l1687_168737

theorem original_average_speed :
  ∀ (D : ℝ),
  (V = D / (5 / 6)) ∧ (60 = D / (2 / 3)) → V = 48 :=
by
  sorry

end original_average_speed_l1687_168737
