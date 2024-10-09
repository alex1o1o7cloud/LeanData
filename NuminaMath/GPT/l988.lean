import Mathlib

namespace product_roots_example_l988_98891

def cubic_eq (a b c d : ℝ) (x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

noncomputable def product_of_roots (a b c d : ℝ) : ℝ := -d / a

theorem product_roots_example : product_of_roots 4 (-2) (-25) 36 = -9 := by
  sorry

end product_roots_example_l988_98891


namespace hyperbola_condition_sufficiency_l988_98861

theorem hyperbola_condition_sufficiency (k : ℝ) :
  (k > 3) → (∃ x y : ℝ, (x^2)/(3-k) + (y^2)/(k-1) = 1) :=
by
  sorry

end hyperbola_condition_sufficiency_l988_98861


namespace Jackson_missed_one_wednesday_l988_98858

theorem Jackson_missed_one_wednesday (weeks total_sandwiches missed_fridays sandwiches_eaten : ℕ) 
  (h1 : weeks = 36)
  (h2 : total_sandwiches = 2 * weeks)
  (h3 : missed_fridays = 2)
  (h4 : sandwiches_eaten = 69) :
  (total_sandwiches - missed_fridays - sandwiches_eaten) / 2 = 1 :=
by
  -- sorry to skip the proof.
  sorry

end Jackson_missed_one_wednesday_l988_98858


namespace paint_cost_per_quart_l988_98821

-- Definitions of conditions
def edge_length (cube_edge_length : ℝ) : Prop := cube_edge_length = 10
def surface_area (s_area : ℝ) : Prop := s_area = 6 * (10^2)
def coverage_per_quart (coverage : ℝ) : Prop := coverage = 120
def total_cost (cost : ℝ) : Prop := cost = 16
def required_quarts (quarts : ℝ) : Prop := quarts = 600 / 120
def cost_per_quart (cost : ℝ) (quarts : ℝ) (price_per_quart : ℝ) : Prop := price_per_quart = cost / quarts

-- Main theorem statement translating the problem into Lean
theorem paint_cost_per_quart {cube_edge_length s_area coverage cost quarts price_per_quart : ℝ} :
  edge_length cube_edge_length →
  surface_area s_area →
  coverage_per_quart coverage →
  total_cost cost →
  required_quarts quarts →
  quarts = s_area / coverage →
  cost_per_quart cost quarts 3.20 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- proof will go here
  sorry

end paint_cost_per_quart_l988_98821


namespace range_x_minus_q_l988_98856

theorem range_x_minus_q (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) : x - q < 3 - 2*q :=
by
  sorry

end range_x_minus_q_l988_98856


namespace cost_per_steak_knife_l988_98854

theorem cost_per_steak_knife
  (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ)
  (h1 : sets = 2) (h2 : knives_per_set = 4) (h3 : cost_per_set = 80) :
  (cost_per_set * sets) / (sets * knives_per_set) = 20 := by
  sorry

end cost_per_steak_knife_l988_98854


namespace coat_price_reduction_l988_98882

theorem coat_price_reduction:
  ∀ (original_price reduction_amount : ℕ),
  original_price = 500 →
  reduction_amount = 350 →
  (reduction_amount : ℝ) / original_price * 100 = 70 :=
by
  intros original_price reduction_amount h1 h2
  sorry

end coat_price_reduction_l988_98882


namespace length_BD_l988_98800

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

end length_BD_l988_98800


namespace number_of_true_propositions_l988_98884

open Classical

-- Define each proposition as a term or lemma in Lean
def prop1 : Prop := ∀ x : ℝ, x^2 + 1 > 0
def prop2 : Prop := ∀ x : ℕ, x^4 ≥ 1
def prop3 : Prop := ∃ x : ℤ, x^3 < 1
def prop4 : Prop := ∀ x : ℚ, x^2 ≠ 2

-- The main theorem statement that the number of true propositions is 3 given the conditions
theorem number_of_true_propositions : (prop1 ∧ prop3 ∧ prop4) ∧ ¬prop2 → 3 = 3 := by
  sorry

end number_of_true_propositions_l988_98884


namespace paul_lost_crayons_l988_98811

theorem paul_lost_crayons :
  ∀ (initial_crayons given_crayons left_crayons lost_crayons : ℕ),
    initial_crayons = 1453 →
    given_crayons = 563 →
    left_crayons = 332 →
    lost_crayons = (initial_crayons - given_crayons) - left_crayons →
    lost_crayons = 558 :=
by
  intros initial_crayons given_crayons left_crayons lost_crayons
  intros h_initial h_given h_left h_lost
  sorry

end paul_lost_crayons_l988_98811


namespace linear_coefficient_l988_98819

theorem linear_coefficient (a b c : ℤ) (h : a = 1 ∧ b = -2 ∧ c = -1) :
    b = -2 := 
by
  -- Use the given hypothesis directly
  exact h.2.1

end linear_coefficient_l988_98819


namespace simplify_expression_l988_98881

theorem simplify_expression :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
  sorry

end simplify_expression_l988_98881


namespace cooperative_payment_divisibility_l988_98878

theorem cooperative_payment_divisibility (T_old : ℕ) (N : ℕ) 
  (hN : N = 99 * T_old / 100) : 99 ∣ N :=
by
  sorry

end cooperative_payment_divisibility_l988_98878


namespace sum_of_reciprocals_of_geometric_sequence_is_two_l988_98817

theorem sum_of_reciprocals_of_geometric_sequence_is_two
  (a1 q : ℝ)
  (pos_terms : 0 < a1)
  (S P M : ℝ)
  (sum_eq : S = 9)
  (product_eq : P = 81 / 4)
  (sum_of_terms : S = a1 * (1 - q^4) / (1 - q))
  (product_of_terms : P = a1 * a1 * q * q * (a1*q*q) * (q*a1) )
  (sum_of_reciprocals : M = (q^4 - 1) / (a1 * (q^4 - q^3)))
  : M = 2 :=
sorry

end sum_of_reciprocals_of_geometric_sequence_is_two_l988_98817


namespace sale_in_fifth_month_l988_98837

theorem sale_in_fifth_month (s1 s2 s3 s4 s5 s6 : ℤ) (avg_sale : ℤ) (h1 : s1 = 6435) (h2 : s2 = 6927)
  (h3 : s3 = 6855) (h4 : s4 = 7230) (h6 : s6 = 7391) (h_avg_sale : avg_sale = 6900) :
    (s1 + s2 + s3 + s4 + s5 + s6) / 6 = avg_sale → s5 = 6562 :=
by
  sorry

end sale_in_fifth_month_l988_98837


namespace failed_english_is_45_l988_98803

-- Definitions of the given conditions
def total_students : ℝ := 1 -- representing 100%
def failed_hindi : ℝ := 0.35
def failed_both : ℝ := 0.2
def passed_both : ℝ := 0.4

-- The goal is to prove that the percentage of students who failed in English is 45%

theorem failed_english_is_45 :
  let failed_at_least_one := total_students - passed_both
  let failed_english := failed_at_least_one - failed_hindi + failed_both
  failed_english = 0.45 :=
by
  -- The steps and manipulation will go here, but for now we skip with sorry
  sorry

end failed_english_is_45_l988_98803


namespace candy_sharing_l988_98805

theorem candy_sharing (Hugh_candy Tommy_candy Melany_candy shared_candy : ℕ) 
  (h1 : Hugh_candy = 8) (h2 : Tommy_candy = 6) (h3 : shared_candy = 7) :
  Hugh_candy + Tommy_candy + Melany_candy = 3 * shared_candy →
  Melany_candy = 7 :=
by
  intro h
  sorry

end candy_sharing_l988_98805


namespace find_x_solution_l988_98880

noncomputable def find_x (x y : ℝ) (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : Prop := 
  x = (3 + Real.sqrt 17) / 2

theorem find_x_solution (x y : ℝ) 
(h1 : x - y^2 = 3) 
(h2 : x^2 + y^4 = 13) 
(hx_pos : 0 < x) 
(hy_pos : 0 < y) : 
  find_x x y h1 h2 :=
sorry

end find_x_solution_l988_98880


namespace area_under_parabola_l988_98857

-- Define the function representing the parabola
def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- State the theorem about the area under the curve
theorem area_under_parabola : (∫ x in (1 : ℝ)..3, parabola x) = 4 / 3 :=
by
  -- Proof goes here
  sorry

end area_under_parabola_l988_98857


namespace multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l988_98866

variable (a b : ℤ)
variable (h1 : a % 4 = 0) 
variable (h2 : b % 8 = 0)

theorem multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : b % 4 = 0 := by
  sorry

theorem diff_multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 4 = 0 := by
  sorry

theorem diff_multiple_of_two (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 2 = 0 := by
  sorry

end multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l988_98866


namespace sum_of_squares_eq_229_l988_98836

-- The conditions
variables (x y : ℤ)
axiom diff_eq_221 : x^2 - y^2 = 221

-- The proof goal
theorem sum_of_squares_eq_229 : x^2 - y^2 = 221 → ∃ x y : ℤ, x^2 + y^2 = 229 :=
by
  sorry

end sum_of_squares_eq_229_l988_98836


namespace max_f_l988_98859

theorem max_f (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x : ℝ, (-1 < x) →  ∀ y : ℝ, (y > -1) → ((1 + y)^a - a*y ≤ 1) :=
sorry

end max_f_l988_98859


namespace arithmetic_geom_seq_l988_98818

noncomputable def geom_seq (a q : ℝ) : ℕ → ℝ 
| 0     => a
| (n+1) => q * (geom_seq a q n)

theorem arithmetic_geom_seq
  (a q : ℝ)
  (h_arith : 2 * geom_seq a q 1 = 1 + (geom_seq a q 2 - 1))
  (h_q : q = 2) :
  (geom_seq a q 2 + geom_seq a q 3) / (geom_seq a q 4 + geom_seq a q 5) = 1 / 4 :=
by
  sorry

end arithmetic_geom_seq_l988_98818


namespace angle_A_in_triangle_l988_98868

theorem angle_A_in_triangle :
  ∀ (A B C : ℝ) (a b c : ℝ),
  a = 2 * Real.sqrt 3 → b = 2 * Real.sqrt 2 → B = π / 4 → 
  (A = π / 3 ∨ A = 2 * π / 3) :=
by
  intros A B C a b c ha hb hB
  sorry

end angle_A_in_triangle_l988_98868


namespace find_number_l988_98863

theorem find_number (x : ℤ) (h : 45 - (28 - (37 - (x - 18))) = 57) : x = 15 :=
by
  sorry

end find_number_l988_98863


namespace find_interest_rate_l988_98802

theorem find_interest_rate (P r : ℝ) 
  (h1 : 100 = P * (1 + 2 * r)) 
  (h2 : 200 = P * (1 + 6 * r)) : 
  r = 0.5 :=
sorry

end find_interest_rate_l988_98802


namespace transformed_curve_eq_l988_98826

/-- Given the initial curve equation and the scaling transformation,
    prove that the resulting curve has the transformed equation. -/
theorem transformed_curve_eq 
  (x y x' y' : ℝ)
  (h_curve : x^2 + 9*y^2 = 9)
  (h_transform_x : x' = x)
  (h_transform_y : y' = 3*y) :
  (x')^2 + y'^2 = 9 := 
sorry

end transformed_curve_eq_l988_98826


namespace remainder_of_3_pow_19_div_10_l988_98874

def w : ℕ := 3 ^ 19

theorem remainder_of_3_pow_19_div_10 : w % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_div_10_l988_98874


namespace notebooks_ratio_l988_98886

variable (C N : Nat)

theorem notebooks_ratio (h1 : 512 = C * N)
  (h2 : 512 = 16 * (C / 2)) :
  N = C / 8 :=
by
  sorry

end notebooks_ratio_l988_98886


namespace joan_missed_games_l988_98851

variable (total_games : ℕ) (night_games : ℕ) (attended_games : ℕ)

theorem joan_missed_games (h1 : total_games = 864) (h2 : night_games = 128) (h3 : attended_games = 395) : 
  total_games - attended_games = 469 :=
  by
    sorry

end joan_missed_games_l988_98851


namespace arnold_plates_count_l988_98889

def arnold_barbell := 45
def mistaken_weight := 600
def actual_weight := 470
def weight_difference_per_plate := 10

theorem arnold_plates_count : 
  ∃ n : ℕ, mistaken_weight - actual_weight = n * weight_difference_per_plate ∧ n = 13 := 
sorry

end arnold_plates_count_l988_98889


namespace fraction_of_women_married_l988_98828

theorem fraction_of_women_married (total : ℕ) (women men married: ℕ) (h1 : total = women + men)
(h2 : women = 76 * total / 100) (h3 : married = 60 * total / 100) (h4 : 2 * (men - married) = 3 * men):
 (married - (total - women - married) * 1 / 3) = 13 * women / 19 :=
sorry

end fraction_of_women_married_l988_98828


namespace meters_of_cloth_sold_l988_98888

-- Definitions based on conditions
def total_selling_price : ℕ := 8925
def profit_per_meter : ℕ := 20
def cost_price_per_meter : ℕ := 85
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- The proof statement
theorem meters_of_cloth_sold : ∃ x : ℕ, selling_price_per_meter * x = total_selling_price ∧ x = 85 := by
  sorry

end meters_of_cloth_sold_l988_98888


namespace problem_l988_98867

theorem problem (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5) : 5 * x ^ 2 + 8 * x * y + 5 * y ^ 2 = 41 := 
by 
  sorry

end problem_l988_98867


namespace neg_mod_eq_1998_l988_98829

theorem neg_mod_eq_1998 {a : ℤ} (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end neg_mod_eq_1998_l988_98829


namespace mary_daily_tasks_l988_98873

theorem mary_daily_tasks :
  ∃ (x y : ℕ), (x + y = 15) ∧ (4 * x + 7 * y = 85) ∧ (y = 8) :=
by
  sorry

end mary_daily_tasks_l988_98873


namespace relationship_between_n_and_m_l988_98890

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def geometric_sequence (a q : ℝ) (m : ℕ) : ℝ :=
  a * q ^ (m - 1)

theorem relationship_between_n_and_m
  (a d q : ℝ) (n m : ℕ)
  (h_d_ne_zero : d ≠ 0)
  (h1 : arithmetic_sequence a d 1 = geometric_sequence a q 1)
  (h2 : arithmetic_sequence a d 3 = geometric_sequence a q 3)
  (h3 : arithmetic_sequence a d 7 = geometric_sequence a q 5)
  (q_pos : 0 < q) (q_sqrt2 : q^2 = 2)
  :
  n = 2 ^ ((m + 1) / 2) - 1 := sorry

end relationship_between_n_and_m_l988_98890


namespace frac_equiv_l988_98846

theorem frac_equiv (a b : ℚ) (h : a / b = 3 / 4) : (a - b) / (a + b) = -1 / 7 := by
  sorry

end frac_equiv_l988_98846


namespace min_value_tan_product_l988_98844

theorem min_value_tan_product (A B C : ℝ) (h : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (sin_eq : Real.sin A = 3 * Real.sin B * Real.sin C) :
  ∃ t : ℝ, t = Real.tan A * Real.tan B * Real.tan C ∧ t = 12 :=
sorry

end min_value_tan_product_l988_98844


namespace union_M_N_eq_interval_l988_98877

variable {α : Type*} [PartialOrder α]

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem union_M_N_eq_interval :
  M ∪ N = {x | -1/2 < x ∧ x ≤ 1} :=
by
  sorry

end union_M_N_eq_interval_l988_98877


namespace problem1_problem2_l988_98887

theorem problem1 (x y : ℝ) : (x + y) * (x - y) + y * (y - 2) = x^2 - 2 * y :=
by 
  sorry

theorem problem2 (m : ℝ) (h : m ≠ 2) : (1 - m / (m + 2)) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 2 / (m - 2) :=
by 
  sorry

end problem1_problem2_l988_98887


namespace simplify_fraction_l988_98860

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : (2 + 4 * i) / (1 - 5 * i) = (-9 / 13) + (7 / 13) * i :=
by sorry

end simplify_fraction_l988_98860


namespace percent_increase_l988_98894

def initial_price : ℝ := 15
def final_price : ℝ := 16

theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 6.67 :=
by
  sorry

end percent_increase_l988_98894


namespace visual_range_percent_increase_l988_98895

-- Define the original and new visual ranges
def original_range : ℝ := 90
def new_range : ℝ := 150

-- Define the desired percent increase as a real number
def desired_percent_increase : ℝ := 66.67

-- The theorem to prove that the visual range is increased by the desired percentage
theorem visual_range_percent_increase :
  ((new_range - original_range) / original_range) * 100 = desired_percent_increase := 
sorry

end visual_range_percent_increase_l988_98895


namespace total_payment_correct_l988_98875

-- Define the conditions for each singer
def firstSingerPayment : ℝ := 2 * 25
def secondSingerPayment : ℝ := 3 * 35
def thirdSingerPayment : ℝ := 4 * 20
def fourthSingerPayment : ℝ := 2.5 * 30

def firstSingerTip : ℝ := 0.15 * firstSingerPayment
def secondSingerTip : ℝ := 0.20 * secondSingerPayment
def thirdSingerTip : ℝ := 0.25 * thirdSingerPayment
def fourthSingerTip : ℝ := 0.18 * fourthSingerPayment

def firstSingerTotal : ℝ := firstSingerPayment + firstSingerTip
def secondSingerTotal : ℝ := secondSingerPayment + secondSingerTip
def thirdSingerTotal : ℝ := thirdSingerPayment + thirdSingerTip
def fourthSingerTotal : ℝ := fourthSingerPayment + fourthSingerTip

-- Define the total amount paid
def totalPayment : ℝ := firstSingerTotal + secondSingerTotal + thirdSingerTotal + fourthSingerTotal

-- The proof problem: Prove the total amount paid
theorem total_payment_correct : totalPayment = 372 := by
  sorry

end total_payment_correct_l988_98875


namespace tan_double_angle_l988_98820

theorem tan_double_angle (α : ℝ) (h1 : Real.cos (Real.pi - α) = 4 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan (2 * α) = 24 / 7 := 
sorry

end tan_double_angle_l988_98820


namespace WalterWorksDaysAWeek_l988_98831

theorem WalterWorksDaysAWeek (hourlyEarning : ℕ) (hoursPerDay : ℕ) (schoolAllocationFraction : ℚ) (schoolAllocation : ℕ) 
  (dailyEarning : ℕ) (weeklyEarning : ℕ) (daysWorked : ℕ) :
  hourlyEarning = 5 →
  hoursPerDay = 4 →
  schoolAllocationFraction = 3 / 4 →
  schoolAllocation = 75 →
  dailyEarning = hourlyEarning * hoursPerDay →
  weeklyEarning = (schoolAllocation : ℚ) / schoolAllocationFraction →
  daysWorked = weeklyEarning / dailyEarning →
  daysWorked = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end WalterWorksDaysAWeek_l988_98831


namespace restore_triangle_ABC_l988_98839

-- let I be the incenter of triangle ABC
variable (I : Point)
-- let Ic be the C-excenter of triangle ABC
variable (I_c : Point)
-- let H be the foot of the altitude from vertex C to side AB
variable (H : Point)

-- Claim: Given I, I_c, H, we can recover the original triangle ABC
theorem restore_triangle_ABC (I I_c H : Point) : ExistsTriangleABC :=
sorry

end restore_triangle_ABC_l988_98839


namespace min_value_of_x2_y2_z2_l988_98870

noncomputable def min_square_sum (x y z k : ℝ) : ℝ :=
  x^2 + y^2 + z^2

theorem min_value_of_x2_y2_z2 (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = k) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (x y z k : ℝ), (x^3 + y^3 + z^3 - 3 * x * y * z = k ∧ k ≥ -1) -> min_square_sum x y z k ≥ min_val :=
by
  sorry

end min_value_of_x2_y2_z2_l988_98870


namespace intersection_eq_l988_98814

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem intersection_eq : M ∩ N = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_eq_l988_98814


namespace remainder_2011_2015_mod_23_l988_98876

theorem remainder_2011_2015_mod_23 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := 
by
  sorry

end remainder_2011_2015_mod_23_l988_98876


namespace machineB_produces_100_parts_in_40_minutes_l988_98838

-- Define the given conditions
def machineA_rate := 50 / 10 -- Machine A's rate in parts per minute
def machineB_rate := machineA_rate / 2 -- Machine B's rate in parts per minute

-- Machine A produces 50 parts in 10 minutes
def machineA_50_parts_time : ℝ := 10

-- Machine B's time to produce 100 parts (The question)
def machineB_100_parts_time : ℝ := 40

-- Proving that Machine B takes 40 minutes to produce 100 parts
theorem machineB_produces_100_parts_in_40_minutes :
    machineB_100_parts_time = 40 :=
by
  sorry

end machineB_produces_100_parts_in_40_minutes_l988_98838


namespace shortest_side_length_rectangular_solid_geometric_progression_l988_98852

theorem shortest_side_length_rectangular_solid_geometric_progression
  (b s : ℝ)
  (h1 : (b^3 / s) = 512)
  (h2 : 2 * ((b^2 / s) + (b^2 * s) + b^2) = 384)
  : min (b / s) (min b (b * s)) = 8 := 
sorry

end shortest_side_length_rectangular_solid_geometric_progression_l988_98852


namespace error_percent_in_area_l988_98864

theorem error_percent_in_area
  (L W : ℝ)
  (hL : L > 0)
  (hW : W > 0) :
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 0.8 := by
  sorry

end error_percent_in_area_l988_98864


namespace inequality_for_a_ne_1_l988_98849

theorem inequality_for_a_ne_1 (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3 * (1 + a^2 + a^4) :=
sorry

end inequality_for_a_ne_1_l988_98849


namespace ordering_of_variables_l988_98833

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem ordering_of_variables 
  (a b c : ℝ)
  (ha : a - 2 = Real.log (a / 2))
  (hb : b - 3 = Real.log (b / 3))
  (hc : c - 3 = Real.log (c / 2))
  (ha_pos : 0 < a) (ha_lt_one : a < 1)
  (hb_pos : 0 < b) (hb_lt_one : b < 1)
  (hc_pos : 0 < c) (hc_lt_one : c < 1) :
  c < b ∧ b < a :=
sorry

end ordering_of_variables_l988_98833


namespace geometric_sequence_fifth_term_l988_98883

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 16) (h2 : a * r^6 = 2) : a * r^4 = 2 :=
sorry

end geometric_sequence_fifth_term_l988_98883


namespace inscribed_circle_radius_l988_98804

theorem inscribed_circle_radius (a b c r : ℝ) (h : a^2 + b^2 = c^2) (h' : r = (a + b - c) / 2) : r = (a + b - c) / 2 :=
by
  sorry

end inscribed_circle_radius_l988_98804


namespace positive_integer_prime_condition_l988_98897

theorem positive_integer_prime_condition (n : ℕ) 
  (h1 : 0 < n)
  (h2 : ∀ (k : ℕ), k < n → Nat.Prime (4 * k^2 + n)) : 
  n = 3 ∨ n = 7 := 
sorry

end positive_integer_prime_condition_l988_98897


namespace minimum_guests_economical_option_l988_98830

theorem minimum_guests_economical_option :
  ∀ (x : ℕ), (150 + 20 * x > 300 + 15 * x) → x > 30 :=
by 
  intro x
  sorry

end minimum_guests_economical_option_l988_98830


namespace hyperbola_focus_l988_98834

theorem hyperbola_focus (m : ℝ) (h : (0, 5) = (0, 5)) : 
  (∀ x y : ℝ, (y^2 / m - x^2 / 9 = 1) → m = 16) :=
sorry

end hyperbola_focus_l988_98834


namespace sum_powers_eq_34_over_3_l988_98847

theorem sum_powers_eq_34_over_3 (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6):
  a^4 + b^4 + c^4 = 34 / 3 :=
by
  sorry

end sum_powers_eq_34_over_3_l988_98847


namespace find_other_number_l988_98853

theorem find_other_number (y : ℕ) : Nat.lcm 240 y = 5040 ∧ Nat.gcd 240 y = 24 → y = 504 :=
by
  sorry

end find_other_number_l988_98853


namespace solve_for_a_and_b_range_of_f_when_x_lt_zero_l988_98809

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1 + a * (2 ^ x)) / (2 ^ x + b)

theorem solve_for_a_and_b (a b : ℝ) :
  f a b 1 = 3 ∧
  f a b (-1) = -3 →
  a = 1 ∧ b = -1 :=
by
  sorry

theorem range_of_f_when_x_lt_zero (x : ℝ) :
  ∀ x < 0, f 1 (-1) x < -1 :=
by 
  sorry

end solve_for_a_and_b_range_of_f_when_x_lt_zero_l988_98809


namespace prove_percentage_cats_adopted_each_month_l988_98885

noncomputable def percentage_cats_adopted_each_month
    (initial_dogs : ℕ)
    (initial_cats : ℕ)
    (initial_lizards : ℕ)
    (adopted_dogs_percent : ℕ)
    (adopted_lizards_percent : ℕ)
    (new_pets_each_month : ℕ)
    (total_pets_after_month : ℕ)
    (adopted_cats_percent : ℕ) : Prop :=
  initial_dogs = 30 ∧
  initial_cats = 28 ∧
  initial_lizards = 20 ∧
  adopted_dogs_percent = 50 ∧
  adopted_lizards_percent = 20 ∧
  new_pets_each_month = 13 ∧
  total_pets_after_month = 65 →
  adopted_cats_percent = 25

-- The condition to prove
theorem prove_percentage_cats_adopted_each_month :
  percentage_cats_adopted_each_month 30 28 20 50 20 13 65 25 :=
by 
  sorry

end prove_percentage_cats_adopted_each_month_l988_98885


namespace exam_students_count_l988_98845

theorem exam_students_count (n : ℕ) (T : ℕ) (h1 : T = 90 * n) 
                            (h2 : (T - 90) / (n - 2) = 95) : n = 20 :=
by {
  sorry
}

end exam_students_count_l988_98845


namespace probability_bernardo_larger_l988_98893

-- Define the sets from which Bernardo and Silvia are picking numbers
def set_B : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def set_S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to calculate the probability as described in the problem statement
def bernardo_larger_probability : ℚ := sorry -- The step by step calculations will be inserted here

-- Main theorem stating what needs to be proved
theorem probability_bernardo_larger : bernardo_larger_probability = 61 / 80 := 
sorry

end probability_bernardo_larger_l988_98893


namespace greatest_possible_perimeter_l988_98850

theorem greatest_possible_perimeter (x : ℤ) (hx1 : 3 * x > 17) (hx2 : 17 > x) : 
  (3 * x + 17 ≤ 65) :=
by
  have Hx : x ≤ 16 := sorry -- Derived from inequalities hx1 and hx2
  have Hx_ge_6 : x ≥ 6 := sorry -- Derived from integer constraint and hx1, hx2
  sorry -- Show 3 * x + 17 has maximum value 65 when x = 16

end greatest_possible_perimeter_l988_98850


namespace triangle_area_is_60_l988_98827

noncomputable def triangle_area (P r : ℝ) : ℝ :=
  (r * P) / 2

theorem triangle_area_is_60 (hP : 48 = 48) (hr : 2.5 = 2.5) : triangle_area 48 2.5 = 60 := by
  sorry

end triangle_area_is_60_l988_98827


namespace total_customers_is_40_l988_98824

-- The number of tables the waiter is attending
def num_tables : ℕ := 5

-- The number of women at each table
def women_per_table : ℕ := 5

-- The number of men at each table
def men_per_table : ℕ := 3

-- The total number of customers at each table
def customers_per_table : ℕ := women_per_table + men_per_table

-- The total number of customers the waiter has
def total_customers : ℕ := num_tables * customers_per_table

theorem total_customers_is_40 : total_customers = 40 :=
by
  -- Proof goes here
  sorry

end total_customers_is_40_l988_98824


namespace amount_of_rice_distributed_in_first_5_days_l988_98848

-- Definitions from conditions
def workers_day (d : ℕ) : ℕ := if d = 1 then 64 else 64 + 7 * (d - 1)

-- The amount of rice each worker receives per day
def rice_per_worker : ℕ := 3

-- Total workers dispatched in the first 5 days
def total_workers_first_5_days : ℕ := (workers_day 1 + workers_day 2 + workers_day 3 + workers_day 4 + workers_day 5)

-- Given these definitions, we now state the theorem to prove
theorem amount_of_rice_distributed_in_first_5_days : total_workers_first_5_days * rice_per_worker = 1170 :=
by
  sorry

end amount_of_rice_distributed_in_first_5_days_l988_98848


namespace required_integer_l988_98801

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 + d2 + d3 + d4 = sum

def middle_digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  d2 + d3 = sum

def thousands_minus_units (n : ℕ) (diff : ℕ) : Prop :=
  let d1 := n / 1000
  let d4 := n % 10
  d1 - d4 = diff

def divisible_by (n : ℕ) (d : ℕ) : Prop :=
  n % d = 0

theorem required_integer : 
  ∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    digits_sum_to n 18 ∧ 
    middle_digits_sum_to n 9 ∧ 
    thousands_minus_units n 3 ∧ 
    divisible_by n 9 ∧ 
    n = 6453 :=
by
  sorry

end required_integer_l988_98801


namespace binom_10_3_l988_98823

theorem binom_10_3 : Nat.choose 10 3 = 120 := 
by
  sorry

end binom_10_3_l988_98823


namespace smallest_int_with_18_divisors_l988_98841

theorem smallest_int_with_18_divisors : ∃ n : ℕ, (∀ d : ℕ, 0 < d ∧ d ≤ n → d = 288) ∧ (∃ a1 a2 a3 : ℕ, a1 + 1 * a2 + 1 * a3 + 1 = 18) := 
by 
  sorry

end smallest_int_with_18_divisors_l988_98841


namespace number_of_cheeses_per_pack_l988_98813

-- Definitions based on the conditions
def packs : ℕ := 3
def cost_per_cheese : ℝ := 0.10
def total_amount_paid : ℝ := 6

-- Theorem statement to prove the number of string cheeses in each pack
theorem number_of_cheeses_per_pack : 
  (total_amount_paid / (packs : ℝ)) / cost_per_cheese = 20 :=
sorry

end number_of_cheeses_per_pack_l988_98813


namespace sin_value_l988_98896

open Real

-- Define the given conditions
variables (x : ℝ) (h1 : cos (π + x) = 3 / 5) (h2 : π < x) (h3 : x < 2 * π)

-- State the problem to be proved
theorem sin_value : sin x = - 4 / 5 :=
by
  sorry

end sin_value_l988_98896


namespace JohnsonsYield_l988_98825

def JohnsonYieldPerTwoMonths (J : ℕ) : Prop :=
  ∀ (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ),
    neighbor_hectares = 2 →
    neighbor_yield_per_hectare = 2 * J →
    total_yield_six_months = 1200 →
    3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months →
    J = 80

theorem JohnsonsYield
  (J : ℕ)
  (neighbor_hectares neighbor_yield_per_hectare total_yield_six_months : ℕ)
  (h1 : neighbor_hectares = 2)
  (h2 : neighbor_yield_per_hectare = 2 * J)
  (h3 : total_yield_six_months = 1200)
  (h4 : 3 * J + 3 * (neighbor_hectares * neighbor_yield_per_hectare) = total_yield_six_months) :
  J = 80 :=
by
  sorry

end JohnsonsYield_l988_98825


namespace cost_of_whitewashing_l988_98842

-- Definitions of the dimensions
def length_room : ℝ := 25.0
def width_room : ℝ := 15.0
def height_room : ℝ := 12.0

def dimensions_door : (ℝ × ℝ) := (6.0, 3.0)
def dimensions_window : (ℝ × ℝ) := (4.0, 3.0)
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 6.0

-- Definition of areas and costs
def area_wall (a b : ℝ) : ℝ := 2 * (a * b)
def area_door : ℝ := (dimensions_door.1 * dimensions_door.2)
def area_window : ℝ := (dimensions_window.1 * dimensions_window.2) * (num_windows)
def total_area_walls : ℝ := (area_wall length_room height_room) + (area_wall width_room height_room)
def area_to_paint : ℝ := total_area_walls - (area_door + area_window)
def total_cost : ℝ := area_to_paint * cost_per_sqft

-- Proof statement
theorem cost_of_whitewashing : total_cost = 5436 := by
  sorry

end cost_of_whitewashing_l988_98842


namespace coin_toss_sequences_count_l988_98892

theorem coin_toss_sequences_count :
  (∃ (seq : List Char), 
    seq.length = 15 ∧ 
    (seq == ['H', 'H']) = 5 ∧ 
    (seq == ['H', 'T']) = 3 ∧ 
    (seq == ['T', 'H']) = 2 ∧ 
    (seq == ['T', 'T']) = 4) → 
  (count_sequences == 775360) :=
by
  sorry

end coin_toss_sequences_count_l988_98892


namespace find_alpha_l988_98898

def point (α : ℝ) : Prop := 3^α = Real.sqrt 3

theorem find_alpha (α : ℝ) (h : point α) : α = 1/2 := 
by 
  sorry

end find_alpha_l988_98898


namespace percentage_reduced_l988_98806

theorem percentage_reduced (P : ℝ) (h : (85 * P / 100) - 11 = 23) : P = 40 :=
by 
  sorry

end percentage_reduced_l988_98806


namespace prism_volume_l988_98865

theorem prism_volume (a b c : ℝ) (h1 : a * b = 12) (h2 : b * c = 8) (h3 : a * c = 4) : a * b * c = 8 * Real.sqrt 6 :=
by 
  sorry

end prism_volume_l988_98865


namespace intersection_with_y_axis_l988_98869

theorem intersection_with_y_axis (x y : ℝ) : (x + y - 3 = 0 ∧ x = 0) → (x = 0 ∧ y = 3) :=
by {
  sorry
}

end intersection_with_y_axis_l988_98869


namespace find_x_for_condition_l988_98815

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x_for_condition :
  (2 * f 1 - 16 = f (1 - 6)) :=
by
  sorry

end find_x_for_condition_l988_98815


namespace students_in_neither_l988_98810

def total_students := 60
def students_in_art := 40
def students_in_music := 30
def students_in_both := 15

theorem students_in_neither : total_students - (students_in_art - students_in_both + students_in_music - students_in_both + students_in_both) = 5 :=
by
  sorry

end students_in_neither_l988_98810


namespace value_of_a_minus_b_l988_98835

variables (a b : ℚ)

theorem value_of_a_minus_b (h1 : |a| = 5) (h2 : |b| = 2) (h3 : |a + b| = a + b) : a - b = 3 ∨ a - b = 7 :=
sorry

end value_of_a_minus_b_l988_98835


namespace find_k_l988_98879

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x = -2 ∧ x^2 - k * x + 2 = 0) : k = -3 := by
  sorry

end find_k_l988_98879


namespace correct_answer_l988_98862

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + m * x - 1

theorem correct_answer (m : ℝ) : 
  (∀ x₁ x₂, 1 < x₁ → 1 < x₂ → (f x₁ m - f x₂ m) / (x₁ - x₂) > 0) → m ≥ -4 :=
by
  sorry

end correct_answer_l988_98862


namespace isosceles_triangle_sine_base_angle_l988_98808

theorem isosceles_triangle_sine_base_angle (m : ℝ) (θ : ℝ) 
  (h1 : m > 0)
  (h2 : θ > 0 ∧ θ < π / 2)
  (h_base_height : m * (Real.sin θ) = (m * 2 * (Real.sin θ) * (Real.cos θ))) :
  Real.sin θ = (Real.sqrt 15) / 4 := 
sorry

end isosceles_triangle_sine_base_angle_l988_98808


namespace proof_remove_terms_sum_is_one_l988_98812

noncomputable def remove_terms_sum_is_one : Prop :=
  let initial_sum := (1/2) + (1/4) + (1/6) + (1/8) + (1/10) + (1/12)
  let terms_to_remove := (1/8) + (1/10)
  initial_sum - terms_to_remove = 1

theorem proof_remove_terms_sum_is_one : remove_terms_sum_is_one :=
by
  -- proof will go here but is not required
  sorry

end proof_remove_terms_sum_is_one_l988_98812


namespace minimum_connected_components_l988_98840

/-- We start with two points A, B on a 6*7 lattice grid. We say two points 
  X, Y are connected if one can reflect several times with respect to points A, B 
  and reach from X to Y. Prove that the minimum number of connected components 
  over all choices of A, B is 8. -/
theorem minimum_connected_components (A B : ℕ × ℕ) 
  (hA : A.1 < 6 ∧ A.2 < 7) (hB : B.1 < 6 ∧ B.2 < 7) :
  ∃ k, k = 8 :=
sorry

end minimum_connected_components_l988_98840


namespace first_term_of_geometric_sequence_l988_98816

theorem first_term_of_geometric_sequence (a r : ℚ) 
  (h1 : a * r = 18) 
  (h2 : a * r^2 = 24) : 
  a = 27 / 2 := 
sorry

end first_term_of_geometric_sequence_l988_98816


namespace goods_train_speed_l988_98843

noncomputable def passenger_train_speed := 64 -- in km/h
noncomputable def passing_time := 18 -- in seconds
noncomputable def goods_train_length := 420 -- in meters
noncomputable def relative_speed_kmh := 84 -- in km/h (derived from solution)

theorem goods_train_speed :
  (∃ V_g, relative_speed_kmh = V_g + passenger_train_speed) →
  (goods_train_length / (passing_time / 3600): ℝ) = relative_speed_kmh →
  V_g = 20 :=
by
  intro h1 h2
  sorry

end goods_train_speed_l988_98843


namespace geometric_series_sum_example_l988_98832

-- Define the finite geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- State the theorem
theorem geometric_series_sum_example :
  geometric_series_sum (1/2) (1/2) 8 = 255 / 256 :=
by
  sorry

end geometric_series_sum_example_l988_98832


namespace same_oxidation_state_HNO3_N2O5_l988_98822

def oxidation_state_HNO3 (H O: Int) : Int := 1 + 1 + (3 * (-2))
def oxidation_state_N2O5 (H O: Int) : Int := (2 * 1) + (5 * (-2))
def oxidation_state_substances_equal : Prop :=
  oxidation_state_HNO3 1 (-2) = oxidation_state_N2O5 1 (-2)

theorem same_oxidation_state_HNO3_N2O5 : oxidation_state_substances_equal :=
  by
  sorry

end same_oxidation_state_HNO3_N2O5_l988_98822


namespace total_bulbs_is_118_l988_98855

-- Define the number of medium lights
def medium_lights : Nat := 12

-- Define the number of large and small lights based on the given conditions
def large_lights : Nat := 2 * medium_lights
def small_lights : Nat := medium_lights + 10

-- Define the number of bulbs required for each type of light
def bulbs_needed_for_medium : Nat := 2 * medium_lights
def bulbs_needed_for_large : Nat := 3 * large_lights
def bulbs_needed_for_small : Nat := 1 * small_lights

-- Define the total number of bulbs needed
def total_bulbs_needed : Nat := bulbs_needed_for_medium + bulbs_needed_for_large + bulbs_needed_for_small

-- The theorem that represents the proof problem
theorem total_bulbs_is_118 : total_bulbs_needed = 118 := by 
  sorry

end total_bulbs_is_118_l988_98855


namespace angle_C_in_triangle_l988_98807

theorem angle_C_in_triangle (A B C : ℝ) 
  (hA : A = 90) (hB : B = 50) : (A + B + C = 180) → C = 40 :=
by
  intro hSum
  rw [hA, hB] at hSum
  linarith

end angle_C_in_triangle_l988_98807


namespace find_angle_A_find_area_l988_98872

-- Define the geometric and trigonometric conditions of the triangle
def triangle (A B C a b c : ℝ) :=
  a = 4 * Real.sqrt 3 ∧ b + c = 8 ∧
  2 * Real.sin A * Real.cos B + Real.sin B = 2 * Real.sin C

-- Prove angle A is 60 degrees
theorem find_angle_A (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : A = Real.pi / 3 := sorry

-- Prove the area of triangle ABC is 4 * sqrt(3) / 3
theorem find_area (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : 
  (1 / 2) * (a * b * Real.sin C) = (4 * Real.sqrt 3) / 3 := sorry

end find_angle_A_find_area_l988_98872


namespace find_a7_l988_98899

variable (a : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = r * a n

axiom a3_eq_1 : a 3 = 1
axiom det_eq_0 : a 6 * a 8 - 8 * 8 = 0

theorem find_a7 (h_geom : geometric_sequence a) : a 7 = 8 :=
  sorry

end find_a7_l988_98899


namespace proof_problem_l988_98871

theorem proof_problem (a b c : ℝ) (h1 : 4 * a - 2 * b + c > 0) (h2 : a + b + c < 0) : b^2 > a * c :=
sorry

end proof_problem_l988_98871
