import Mathlib

namespace NUMINAMATH_GPT_average_score_in_all_matches_l1998_199868

theorem average_score_in_all_matches (runs_match1_match2 : ℤ) (runs_other_matches : ℤ) (total_matches : ℤ) 
  (average1 : ℤ) (average2 : ℤ)
  (h1 : average1 = 40) (h2 : average2 = 10) (h3 : runs_match1_match2 = 2 * average1)
  (h4 : runs_other_matches = 3 * average2) (h5 : total_matches = 5) :
  ((runs_match1_match2 + runs_other_matches) / total_matches) = 22 := 
by
  sorry

end NUMINAMATH_GPT_average_score_in_all_matches_l1998_199868


namespace NUMINAMATH_GPT_compute_avg_interest_rate_l1998_199873

variable (x : ℝ)

/-- The total amount of investment is $5000 - x at 3% and x at 7%. The incomes are equal 
thus we are asked to compute the average rate of interest -/
def avg_interest_rate : Prop :=
  let i_3 := 0.03 * (5000 - x)
  let i_7 := 0.07 * x
  i_3 = i_7 ∧
  (2 * i_3) / 5000 = 0.042

theorem compute_avg_interest_rate 
  (condition : ∃ x : ℝ, 0.03 * (5000 - x) = 0.07 * x) :
  avg_interest_rate x :=
by
  sorry

end NUMINAMATH_GPT_compute_avg_interest_rate_l1998_199873


namespace NUMINAMATH_GPT_solve_equation_l1998_199846

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2 / 3 → (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) → (x = 1 / 3) ∨ (x = -3)) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1998_199846


namespace NUMINAMATH_GPT_soup_problem_l1998_199879

def cans_needed_for_children (children : ℕ) (children_per_can : ℕ) : ℕ :=
  children / children_per_can

def remaining_cans (initial_cans used_cans : ℕ) : ℕ :=
  initial_cans - used_cans

def half_cans (cans : ℕ) : ℕ :=
  cans / 2

def adults_fed (cans : ℕ) (adults_per_can : ℕ) : ℕ :=
  cans * adults_per_can

theorem soup_problem
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (reserved_fraction : ℕ)
  (hreserved : reserved_fraction = 2)
  (hintial : initial_cans = 8)
  (hchildren : children_fed = 24)
  (hchildren_per_can : children_per_can = 6)
  (hadults_per_can : adults_per_can = 4) :
  adults_fed (half_cans (remaining_cans initial_cans (cans_needed_for_children children_fed children_per_can))) adults_per_can = 8 :=
by
  sorry

end NUMINAMATH_GPT_soup_problem_l1998_199879


namespace NUMINAMATH_GPT_simplify_expression_l1998_199893

theorem simplify_expression (w : ℝ) : 2 * w + 4 * w + 6 * w + 8 * w + 10 * w + 12 = 30 * w + 12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1998_199893


namespace NUMINAMATH_GPT_conditions_for_star_commute_l1998_199861

-- Define the operation star
def star (a b : ℝ) : ℝ := a^3 * b^2 - a * b^3

-- Theorem stating the equivalence
theorem conditions_for_star_commute :
  ∀ (x y : ℝ), (star x y = star y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
sorry

end NUMINAMATH_GPT_conditions_for_star_commute_l1998_199861


namespace NUMINAMATH_GPT_find_number_of_10_bills_from_mother_l1998_199872

variable (m10 : ℕ)  -- number of $10 bills given by Luke's mother

def mother_total : ℕ := 50 + 2*20 + 10*m10
def father_total : ℕ := 4*50 + 20 + 10
def total : ℕ := mother_total m10 + father_total

theorem find_number_of_10_bills_from_mother
  (fee : ℕ := 350)
  (m10 : ℕ) :
  total m10 = fee → m10 = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_number_of_10_bills_from_mother_l1998_199872


namespace NUMINAMATH_GPT_transformation_g_from_f_l1998_199800

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (8 * x + 3 * Real.pi / 2)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem transformation_g_from_f :
  (∀ x, g x = f (x + Real.pi / 4) * 2) ∨ (∀ x, g x = f (x - Real.pi / 4) * 2) := 
by
  sorry

end NUMINAMATH_GPT_transformation_g_from_f_l1998_199800


namespace NUMINAMATH_GPT_hogwarts_school_students_l1998_199898

def total_students_at_school (participants boys : ℕ) (boy_participants girl_non_participants : ℕ) : Prop :=
  participants = 246 ∧ boys = 255 ∧ boy_participants = girl_non_participants + 11 → (boys + (participants - boy_participants + girl_non_participants)) = 490

theorem hogwarts_school_students : total_students_at_school 246 255 (boy_participants) girl_non_participants := 
 sorry

end NUMINAMATH_GPT_hogwarts_school_students_l1998_199898


namespace NUMINAMATH_GPT_determine_f_5_l1998_199842

theorem determine_f_5 (f : ℝ → ℝ) (h1 : f 1 = 3) 
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) : f 5 = 45 :=
sorry

end NUMINAMATH_GPT_determine_f_5_l1998_199842


namespace NUMINAMATH_GPT_slope_of_chord_l1998_199895

theorem slope_of_chord (x y : ℝ) (h : (x^2 / 16) + (y^2 / 9) = 1) (h_midpoint : (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 4)) :
  ∃ k : ℝ, k = -9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_chord_l1998_199895


namespace NUMINAMATH_GPT_exists_nat_concat_is_perfect_square_l1998_199823

theorem exists_nat_concat_is_perfect_square :
  ∃ A : ℕ, ∃ n : ℕ, ∃ B : ℕ, (B * B = (10^n + 1) * A) :=
by sorry

end NUMINAMATH_GPT_exists_nat_concat_is_perfect_square_l1998_199823


namespace NUMINAMATH_GPT_base8_to_base10_l1998_199818

theorem base8_to_base10 :
  ∀ (n : ℕ), (n = 2 * 8^2 + 4 * 8^1 + 3 * 8^0) → (n = 163) :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_base8_to_base10_l1998_199818


namespace NUMINAMATH_GPT_evaluate_expression_at_minus_half_l1998_199841

noncomputable def complex_expression (x : ℚ) : ℚ :=
  (x - 3)^2 + (x + 3) * (x - 3) - 2 * x * (x - 2) + 1

theorem evaluate_expression_at_minus_half :
  complex_expression (-1 / 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_minus_half_l1998_199841


namespace NUMINAMATH_GPT_fraction_addition_l1998_199838

theorem fraction_addition : (1 + 3 + 5)/(2 + 4 + 6) + (2 + 4 + 6)/(1 + 3 + 5) = 25/12 := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l1998_199838


namespace NUMINAMATH_GPT_union_A_B_complement_intersection_A_B_l1998_199888

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}

def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem union_A_B : A ∪ B = { x | x ≥ 3 } := 
by
  sorry

theorem complement_intersection_A_B : (A ∩ B)ᶜ = { x | x < 4 } ∪ { x | x ≥ 10 } := 
by
  sorry

end NUMINAMATH_GPT_union_A_B_complement_intersection_A_B_l1998_199888


namespace NUMINAMATH_GPT_relationship_f_3x_ge_f_2x_l1998_199871

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0, and
    satisfying the symmetry condition f(1-x) = f(1+x) for any x ∈ ℝ,
    the relationship f(3^x) ≥ f(2^x) holds. -/
theorem relationship_f_3x_ge_f_2x (a b c : ℝ) (h_a : a > 0) (symm_cond : ∀ x : ℝ, (a * (1 - x)^2 + b * (1 - x) + c) = (a * (1 + x)^2 + b * (1 + x) + c)) :
  ∀ x : ℝ, (a * (3^x)^2 + b * 3^x + c) ≥ (a * (2^x)^2 + b * 2^x + c) :=
sorry

end NUMINAMATH_GPT_relationship_f_3x_ge_f_2x_l1998_199871


namespace NUMINAMATH_GPT_determine_height_impossible_l1998_199808

-- Definitions used in the conditions
def shadow_length_same (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) : Prop :=
  xiao_ming_height / xiao_ming_distance = xiao_qiang_height / xiao_qiang_distance

-- The proof problem: given that the shadow lengths are the same under the same street lamp,
-- prove that it is impossible to determine who is taller.
theorem determine_height_impossible (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) :
  shadow_length_same xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance →
  ¬ (xiao_ming_height ≠ xiao_qiang_height ↔ true) :=
by
  intro h
  sorry -- Proof not required as per instructions

end NUMINAMATH_GPT_determine_height_impossible_l1998_199808


namespace NUMINAMATH_GPT_total_cost_of_roads_l1998_199816

/-- A rectangular lawn with dimensions 150 m by 80 m with two roads running 
through the middle, one parallel to the length and one parallel to the breadth. 
The first road has a width of 12 m, a base cost of Rs. 4 per sq m, and an additional section 
through a hill costing 25% more for a section of length 60 m. The second road has a width 
of 8 m and a cost of Rs. 5 per sq m. Prove that the total cost for both roads is Rs. 14000. -/
theorem total_cost_of_roads :
  let lawn_length := 150
  let lawn_breadth := 80
  let road1_width := 12
  let road2_width := 8
  let road1_base_cost := 4
  let road1_hill_length := 60
  let road1_hill_cost := road1_base_cost + (road1_base_cost / 4)
  let road2_cost := 5
  let road1_length := lawn_length
  let road2_length := lawn_breadth

  let road1_area_non_hill := road1_length * road1_width
  let road1_area_hill := road1_hill_length * road1_width
  let road1_cost_non_hill := road1_area_non_hill * road1_base_cost
  let road1_cost_hill := road1_area_hill * road1_hill_cost

  let total_road1_cost := road1_cost_non_hill + road1_cost_hill

  let road2_area := road2_length * road2_width
  let road2_total_cost := road2_area * road2_cost

  let total_cost := total_road1_cost + road2_total_cost

  total_cost = 14000 := by sorry

end NUMINAMATH_GPT_total_cost_of_roads_l1998_199816


namespace NUMINAMATH_GPT_find_f_log_l1998_199830

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom f_def : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2

-- Theorem to be proved
theorem find_f_log : f (Real.log 6 / Real.log (1/2)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_log_l1998_199830


namespace NUMINAMATH_GPT_farmer_total_acres_l1998_199853

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end NUMINAMATH_GPT_farmer_total_acres_l1998_199853


namespace NUMINAMATH_GPT_number_of_arrangements_l1998_199844

theorem number_of_arrangements :
  ∃ (n k : ℕ), n = 10 ∧ k = 5 ∧ Nat.choose n k = 252 := by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l1998_199844


namespace NUMINAMATH_GPT_rational_root_of_factors_l1998_199801

theorem rational_root_of_factors (p : ℕ) (a : ℚ) (hprime : Nat.Prime p) 
  (f : Polynomial ℚ) (hf : f = Polynomial.X ^ p - Polynomial.C a)
  (hfactors : ∃ g h : Polynomial ℚ, f = g * h ∧ 1 ≤ g.degree ∧ 1 ≤ h.degree) : 
  ∃ r : ℚ, Polynomial.eval r f = 0 :=
sorry

end NUMINAMATH_GPT_rational_root_of_factors_l1998_199801


namespace NUMINAMATH_GPT_range_of_a_l1998_199874

noncomputable def A : Set ℝ := { x : ℝ | x > 5 }
noncomputable def B (a : ℝ) : Set ℝ := { x : ℝ | x > a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a < 5 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1998_199874


namespace NUMINAMATH_GPT_pipe_A_fill_time_l1998_199811

theorem pipe_A_fill_time 
  (t : ℝ)
  (ht : (1 / t - 1 / 6) = 4 / 15.000000000000005) : 
  t = 30 / 13 :=  
sorry

end NUMINAMATH_GPT_pipe_A_fill_time_l1998_199811


namespace NUMINAMATH_GPT_candies_per_basket_l1998_199843

noncomputable def chocolate_bars : ℕ := 5
noncomputable def mms : ℕ := 7 * chocolate_bars
noncomputable def marshmallows : ℕ := 6 * mms
noncomputable def total_candies : ℕ := chocolate_bars + mms + marshmallows
noncomputable def baskets : ℕ := 25

theorem candies_per_basket : total_candies / baskets = 10 :=
by
  sorry

end NUMINAMATH_GPT_candies_per_basket_l1998_199843


namespace NUMINAMATH_GPT_num_six_digit_asc_digits_l1998_199839

theorem num_six_digit_asc_digits : 
  ∃ n : ℕ, n = (Nat.choose 9 3) ∧ n = 84 := 
by
  sorry

end NUMINAMATH_GPT_num_six_digit_asc_digits_l1998_199839


namespace NUMINAMATH_GPT_jessica_deposit_fraction_l1998_199840

-- Definitions based on conditions
variable (initial_balance : ℝ)
variable (fraction_withdrawn : ℝ) (withdrawn_amount : ℝ)
variable (final_balance remaining_balance fraction_deposit : ℝ)

-- Conditions
def conditions := 
  fraction_withdrawn = 2 / 5 ∧
  withdrawn_amount = 400 ∧
  remaining_balance = initial_balance - withdrawn_amount ∧
  remaining_balance = initial_balance * (1 - fraction_withdrawn) ∧
  final_balance = 750 ∧
  final_balance = remaining_balance + fraction_deposit * remaining_balance

-- The proof problem
theorem jessica_deposit_fraction : 
  conditions initial_balance fraction_withdrawn withdrawn_amount final_balance remaining_balance fraction_deposit →
  fraction_deposit = 1 / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_jessica_deposit_fraction_l1998_199840


namespace NUMINAMATH_GPT_floor_e_minus_3_eq_negative_one_l1998_199899

theorem floor_e_minus_3_eq_negative_one 
  (e : ℝ) 
  (h : 2 < e ∧ e < 3) : 
  (⌊e - 3⌋ = -1) :=
by
  sorry

end NUMINAMATH_GPT_floor_e_minus_3_eq_negative_one_l1998_199899


namespace NUMINAMATH_GPT_mass_of_1m3_l1998_199827

/-- The volume of 1 gram of the substance in cubic centimeters cms_per_gram is 1.3333333333333335 cm³. -/
def cms_per_gram : ℝ := 1.3333333333333335

/-- There are 1,000,000 cubic centimeters in 1 cubic meter. -/
def cm3_per_m3 : ℕ := 1000000

/-- Given the volume of 1 gram of the substance, find the mass of 1 cubic meter of the substance. -/
theorem mass_of_1m3 (h1 : cms_per_gram = 1.3333333333333335) (h2 : cm3_per_m3 = 1000000) :
  ∃ m : ℝ, m = 750 :=
by
  sorry

end NUMINAMATH_GPT_mass_of_1m3_l1998_199827


namespace NUMINAMATH_GPT_divides_two_pow_n_minus_one_l1998_199886

theorem divides_two_pow_n_minus_one {n : ℕ} (h : n > 0) (divides : n ∣ 2^n - 1) : n = 1 :=
sorry

end NUMINAMATH_GPT_divides_two_pow_n_minus_one_l1998_199886


namespace NUMINAMATH_GPT_pirate_coins_l1998_199854

theorem pirate_coins (x : ℕ) (hn : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 → ∃ y : ℕ, y = (2 * k * x) / 15) : 
  ∃ y : ℕ, y = 630630 :=
by sorry

end NUMINAMATH_GPT_pirate_coins_l1998_199854


namespace NUMINAMATH_GPT_optionB_is_a9_l1998_199820

-- Definitions of the expressions
def optionA (a : ℤ) : ℤ := a^3 + a^6
def optionB (a : ℤ) : ℤ := a^3 * a^6
def optionC (a : ℤ) : ℤ := a^10 - a
def optionD (a α : ℤ) : ℤ := α^12 / a^2

-- Theorem stating which option equals a^9
theorem optionB_is_a9 (a α : ℤ) : optionA a ≠ a^9 ∧ optionB a = a^9 ∧ optionC a ≠ a^9 ∧ optionD a α ≠ a^9 :=
by
  sorry

end NUMINAMATH_GPT_optionB_is_a9_l1998_199820


namespace NUMINAMATH_GPT_a_2016_mod_2017_l1998_199836

-- Defining the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧
  a 1 = 2 ∧
  ∀ n, a (n + 2) = 2 * a (n + 1) + 41 * a n

theorem a_2016_mod_2017 (a : ℕ → ℕ) (h : seq a) : 
  a 2016 % 2017 = 0 := 
sorry

end NUMINAMATH_GPT_a_2016_mod_2017_l1998_199836


namespace NUMINAMATH_GPT_class_boys_count_l1998_199875

theorem class_boys_count
    (x y : ℕ)
    (h1 : x + y = 20)
    (h2 : (1 / 3 : ℚ) * x = (1 / 2 : ℚ) * y) :
    x = 12 :=
by
  sorry

end NUMINAMATH_GPT_class_boys_count_l1998_199875


namespace NUMINAMATH_GPT_number_of_teams_l1998_199837

theorem number_of_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l1998_199837


namespace NUMINAMATH_GPT_min_disks_required_l1998_199885

def num_files : ℕ := 35
def disk_size : ℕ := 2
def file_size_0_9 : ℕ := 4
def file_size_0_8 : ℕ := 15
def file_size_0_5 : ℕ := num_files - file_size_0_9 - file_size_0_8

-- Prove the minimum number of disks required to store all files.
theorem min_disks_required 
  (n : ℕ) 
  (disk_storage : ℕ)
  (num_files_0_9 : ℕ)
  (num_files_0_8 : ℕ)
  (num_files_0_5 : ℕ) :
  n = num_files → disk_storage = disk_size → num_files_0_9 = file_size_0_9 → num_files_0_8 = file_size_0_8 → num_files_0_5 = file_size_0_5 → 
  ∃ (d : ℕ), d = 15 :=
by 
  intros H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_min_disks_required_l1998_199885


namespace NUMINAMATH_GPT_find_x_l1998_199810

variable (x : ℝ)  -- Current distance Teena is behind Loe in miles
variable (t : ℝ) -- Time period in hours
variable (speed_teena : ℝ) -- Speed of Teena in miles per hour
variable (speed_loe : ℝ) -- Speed of Loe in miles per hour
variable (d_ahead : ℝ) -- Distance Teena will be ahead of Loe in 1.5 hours

axiom conditions : speed_teena = 55 ∧ speed_loe = 40 ∧ t = 1.5 ∧ d_ahead = 15

theorem find_x : (speed_teena * t - speed_loe * t = x + d_ahead) → x = 7.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l1998_199810


namespace NUMINAMATH_GPT_sin_squared_sum_eq_one_l1998_199851

theorem sin_squared_sum_eq_one (α β γ : ℝ) 
  (h₁ : 0 ≤ α ∧ α ≤ π/2) 
  (h₂ : 0 ≤ β ∧ β ≤ π/2) 
  (h₃ : 0 ≤ γ ∧ γ ≤ π/2) 
  (h₄ : Real.sin α + Real.sin β + Real.sin γ = 1)
  (h₅ : Real.sin α * Real.cos (2 * α) + Real.sin β * Real.cos (2 * β) + Real.sin γ * Real.cos (2 * γ) = -1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1 := 
sorry

end NUMINAMATH_GPT_sin_squared_sum_eq_one_l1998_199851


namespace NUMINAMATH_GPT_ratio_of_costs_l1998_199832

-- Definitions based on conditions
def quilt_length : Nat := 16
def quilt_width : Nat := 20
def patch_area : Nat := 4
def first_10_patch_cost : Nat := 10
def total_cost : Nat := 450

-- Theorem we need to prove
theorem ratio_of_costs : (total_cost - 10 * first_10_patch_cost) / (10 * first_10_patch_cost) = 7 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_costs_l1998_199832


namespace NUMINAMATH_GPT_length_of_train_l1998_199831

-- We state the problem as a theorem in Lean
theorem length_of_train (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ)
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 32)
  (h_train_speed_kmh : train_speed_kmh = 45) :
  ∃ (train_length : ℝ), train_length = 250 := 
by
  -- We assume the necessary conditions as given
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  have total_distance : ℝ := train_speed_ms * crossing_time
  have train_length : ℝ := total_distance - bridge_length
  -- Conclude the length of the train is 250
  use train_length
  -- The proof steps are skipped using 'sorry'
  sorry

end NUMINAMATH_GPT_length_of_train_l1998_199831


namespace NUMINAMATH_GPT_positive_number_solution_l1998_199882

theorem positive_number_solution (x : ℚ) (hx : 0 < x) (h : x * x^2 * (1 / x) = 100 / 81) : x = 10 / 9 :=
sorry

end NUMINAMATH_GPT_positive_number_solution_l1998_199882


namespace NUMINAMATH_GPT_range_of_m_l1998_199881

theorem range_of_m (x y : ℝ) (m : ℝ) (h1 : x^2 + y^2 = 9) (h2 : |x| + |y| ≥ m) :
    m ≤ 3 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1998_199881


namespace NUMINAMATH_GPT_find_y_l1998_199809

def diamond (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y : ∃ y : ℝ, diamond 4 y = 44 ∧ y = 48 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1998_199809


namespace NUMINAMATH_GPT_n_is_power_of_p_l1998_199891

-- Given conditions as definitions
variables {x y p n k l : ℕ}
variables (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < p) (h4 : 0 < n) (h5 : 0 < k)
variables (h6 : x^n + y^n = p^k) (h7 : odd n) (h8 : n > 1) (h9 : prime p) (h10 : odd p)

-- The theorem to be proved
theorem n_is_power_of_p : ∃ l : ℕ, n = p^l :=
  sorry

end NUMINAMATH_GPT_n_is_power_of_p_l1998_199891


namespace NUMINAMATH_GPT_tina_pink_pens_l1998_199859

def number_pink_pens (P G B : ℕ) : Prop :=
  G = P - 9 ∧
  B = P - 6 ∧
  P + G + B = 21

theorem tina_pink_pens :
  ∃ (P G B : ℕ), number_pink_pens P G B ∧ P = 12 :=
by
  sorry

end NUMINAMATH_GPT_tina_pink_pens_l1998_199859


namespace NUMINAMATH_GPT_check_basis_l1998_199866

structure Vector2D :=
  (x : ℤ)
  (y : ℤ)

def are_collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y - v2.x * v1.y = 0

def can_be_basis (v1 v2 : Vector2D) : Prop :=
  ¬ are_collinear v1 v2

theorem check_basis :
  can_be_basis ⟨-1, 2⟩ ⟨5, 7⟩ ∧
  ¬ can_be_basis ⟨0, 0⟩ ⟨1, -2⟩ ∧
  ¬ can_be_basis ⟨3, 5⟩ ⟨6, 10⟩ ∧
  ¬ can_be_basis ⟨2, -3⟩ ⟨(1 : ℤ)/2, -(3 : ℤ)/4⟩ :=
by
  sorry

end NUMINAMATH_GPT_check_basis_l1998_199866


namespace NUMINAMATH_GPT_f_a_minus_2_lt_0_l1998_199877

theorem f_a_minus_2_lt_0 (f : ℝ → ℝ) (m a : ℝ) (h1 : ∀ x, f x = (m + 1 - x) * (x - m + 1)) (h2 : f a > 0) : f (a - 2) < 0 := 
sorry

end NUMINAMATH_GPT_f_a_minus_2_lt_0_l1998_199877


namespace NUMINAMATH_GPT_opposite_of_neg_three_fifths_l1998_199860

theorem opposite_of_neg_three_fifths :
  -(-3 / 5) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_fifths_l1998_199860


namespace NUMINAMATH_GPT_car_new_speed_l1998_199876

theorem car_new_speed (original_speed : ℝ) (supercharge_percent : ℝ) (weight_cut_speed_increase : ℝ) :
  original_speed = 150 → supercharge_percent = 0.30 → weight_cut_speed_increase = 10 → 
  original_speed * (1 + supercharge_percent) + weight_cut_speed_increase = 205 :=
by
  intros h_orig h_supercharge h_weight
  rw [h_orig, h_supercharge]
  sorry

end NUMINAMATH_GPT_car_new_speed_l1998_199876


namespace NUMINAMATH_GPT_hexagon_perimeter_is_42_l1998_199855

-- Define the side length of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) : ℕ :=
  num_sides * side_length

-- The theorem to prove
theorem hexagon_perimeter_is_42 : hexagon_perimeter side_length num_sides = 42 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_is_42_l1998_199855


namespace NUMINAMATH_GPT_ABD_collinear_l1998_199815

noncomputable def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (p2.1 - p1.1) * k = p3.1 - p1.1 ∧ (p2.2 - p1.2) * k = p3.2 - p1.2

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

variables {a b : ℝ × ℝ}
variables {A B C D : ℝ × ℝ}

axiom a_ne_zero : a ≠ (0, 0)
axiom b_ne_zero : b ≠ (0, 0)
axiom a_b_not_collinear : ∀ k : ℝ, a ≠ k • b
axiom AB_def : B = (A.1 + a.1 + b.1, A.2 + a.2 + b.2)
axiom BC_def : C = (B.1 + a.1 + 10 * b.1, B.2 + a.2 + 10 * b.2)
axiom CD_def : D = (C.1 + 3 * (a.1 - 2 * b.1), C.2 + 3 * (a.2 - 2 * b.2))

theorem ABD_collinear : collinear A B D :=
by
  sorry

end NUMINAMATH_GPT_ABD_collinear_l1998_199815


namespace NUMINAMATH_GPT_minimum_value_of_fraction_sum_l1998_199835

open Real

theorem minimum_value_of_fraction_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 2) : 
    6 ≤ (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) := by 
  sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_sum_l1998_199835


namespace NUMINAMATH_GPT_count_integers_between_cubes_l1998_199804

theorem count_integers_between_cubes (a b : ℝ) (h1 : a = 10.5) (h2 : b = 10.6) : 
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  (last_integer - first_integer + 1) = 33 :=
by
  -- Definitions for clarity
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_count_integers_between_cubes_l1998_199804


namespace NUMINAMATH_GPT_parcel_post_cost_l1998_199833

def indicator (P : ℕ) : ℕ := if P >= 5 then 1 else 0

theorem parcel_post_cost (P : ℕ) : 
  P ≥ 0 →
  (C : ℕ) = 15 + 5 * (P - 1) - 8 * indicator P :=
sorry

end NUMINAMATH_GPT_parcel_post_cost_l1998_199833


namespace NUMINAMATH_GPT_total_distance_traveled_l1998_199806

def distance_from_earth_to_planet_x : ℝ := 0.5
def distance_from_planet_x_to_planet_y : ℝ := 0.1
def distance_from_planet_y_to_earth : ℝ := 0.1

theorem total_distance_traveled : 
  distance_from_earth_to_planet_x + distance_from_planet_x_to_planet_y + distance_from_planet_y_to_earth = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l1998_199806


namespace NUMINAMATH_GPT_quadratic_roots_difference_square_l1998_199825

theorem quadratic_roots_difference_square (a b : ℝ) (h : 2 * a^2 - 8 * a + 6 = 0 ∧ 2 * b^2 - 8 * b + 6 = 0) :
  (a - b) ^ 2 = 4 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_difference_square_l1998_199825


namespace NUMINAMATH_GPT_parabola_symmetric_points_l1998_199892

theorem parabola_symmetric_points (a : ℝ) (h : 0 < a) :
  (∃ (P Q : ℝ × ℝ), (P ≠ Q) ∧ ((P.fst + P.snd = 0) ∧ (Q.fst + Q.snd = 0)) ∧
    (P.snd = a * P.fst ^ 2 - 1) ∧ (Q.snd = a * Q.fst ^ 2 - 1)) ↔ (3 / 4 < a) := 
sorry

end NUMINAMATH_GPT_parabola_symmetric_points_l1998_199892


namespace NUMINAMATH_GPT_taxi_fare_charge_l1998_199852

theorem taxi_fare_charge :
  let initial_fee := 2.25
  let total_distance := 3.6
  let total_charge := 4.95
  let increments := total_distance / (2 / 5)
  let distance_charge := total_charge - initial_fee
  let charge_per_increment := distance_charge / increments
  charge_per_increment = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_taxi_fare_charge_l1998_199852


namespace NUMINAMATH_GPT_calculation_result_l1998_199856

theorem calculation_result : (4^2)^3 - 4 = 4092 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l1998_199856


namespace NUMINAMATH_GPT_speed_of_man_l1998_199896

theorem speed_of_man :
  let L := 500 -- Length of the train in meters
  let t := 29.997600191984642 -- Time in seconds
  let V_train_kmh := 63 -- Speed of train in km/hr
  let V_train := (63 * 1000) / 3600 -- Speed of train converted to m/s
  let V_relative := L / t -- Relative speed of train w.r.t man
  
  V_train - V_relative = 0.833 := by
  sorry

end NUMINAMATH_GPT_speed_of_man_l1998_199896


namespace NUMINAMATH_GPT_inequality_abc_l1998_199845

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) (h5 : 0 ≤ c) (h6 : c ≤ 1) :
  (a / (b * c + 1)) + (b / (a * c + 1)) + (c / (a * b + 1)) ≤ 2 := by
  sorry

end NUMINAMATH_GPT_inequality_abc_l1998_199845


namespace NUMINAMATH_GPT_sample_size_correct_l1998_199817

-- Definitions following the conditions in the problem
def total_products : Nat := 80
def sample_products : Nat := 10

-- Statement of the proof problem
theorem sample_size_correct : sample_products = 10 :=
by
  -- The proof is replaced with a placeholder sorry to skip the proof step
  sorry

end NUMINAMATH_GPT_sample_size_correct_l1998_199817


namespace NUMINAMATH_GPT_solution_system_eq_l1998_199864

theorem solution_system_eq (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4 ∧ y = -1) :=
by sorry

end NUMINAMATH_GPT_solution_system_eq_l1998_199864


namespace NUMINAMATH_GPT_factored_quadratic_even_b_l1998_199865

theorem factored_quadratic_even_b
  (c d e f y : ℤ)
  (h1 : c * e = 45)
  (h2 : d * f = 45) 
  (h3 : ∃ b, 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) :
  ∃ b, (45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) ∧ (b % 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_factored_quadratic_even_b_l1998_199865


namespace NUMINAMATH_GPT_solve_for_nabla_l1998_199857

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_nabla_l1998_199857


namespace NUMINAMATH_GPT_unwilted_roses_proof_l1998_199834

-- Conditions
def initial_roses : Nat := 2 * 12
def traded_roses : Nat := 12
def first_day_roses (r: Nat) : Nat := r / 2
def second_day_roses (r: Nat) : Nat := r / 2

-- Initial number of roses
def total_roses : Nat := initial_roses + traded_roses

-- Number of unwilted roses after two days
def unwilted_roses : Nat := second_day_roses (first_day_roses total_roses)

-- Formal statement to prove
theorem unwilted_roses_proof : unwilted_roses = 9 := by
  sorry

end NUMINAMATH_GPT_unwilted_roses_proof_l1998_199834


namespace NUMINAMATH_GPT_intersection_correct_l1998_199812

def A : Set ℝ := { x | 0 < x ∧ x < 3 }
def B : Set ℝ := { x | x^2 ≥ 4 }
def intersection : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

theorem intersection_correct : A ∩ B = intersection := by
  sorry

end NUMINAMATH_GPT_intersection_correct_l1998_199812


namespace NUMINAMATH_GPT_function_is_zero_l1998_199803

theorem function_is_zero (f : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_function_is_zero_l1998_199803


namespace NUMINAMATH_GPT_value_of_expression_l1998_199878

-- Let's define the sequences and sums based on the conditions in a)
def sum_of_evens (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_of_multiples_of_three (p : ℕ) : ℕ :=
  3 * (p * (p + 1)) / 2

def sum_of_odds (m : ℕ) : ℕ :=
  m * m

-- Now let's formulate the problem statement as a theorem.
theorem value_of_expression : 
  sum_of_evens 200 - sum_of_multiples_of_three 100 - sum_of_odds 148 = 3146 :=
  by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1998_199878


namespace NUMINAMATH_GPT_find_number_l1998_199805

theorem find_number (x : ℝ) : x + 5 * 12 / (180 / 3) = 61 ↔ x = 60 := by
  sorry

end NUMINAMATH_GPT_find_number_l1998_199805


namespace NUMINAMATH_GPT_general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l1998_199828

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Noncomputable sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}
variable (c : ℤ)

axiom h1 : is_arithmetic_sequence a d
axiom h2 : d > 0
axiom h3 : a 1 * a 2 = 45
axiom h4 : a 0 + a 4 = 18

-- General formula for the nth term
theorem general_formula_for_nth_term :
  ∃ a1 d, a 0 = a1 ∧ d > 0 ∧ (∀ n, a n = a1 + n * d) :=
sorry

-- Arithmetic sequence from Sn/(n+c)
theorem exists_c_makes_bn_arithmetic :
  ∃ (c : ℤ), c ≠ 0 ∧ (∀ n, n > 0 → (arithmetic_sum a n) / (n + c) - (arithmetic_sum a (n - 1)) / (n - 1 + c) = d) :=
sorry

end NUMINAMATH_GPT_general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l1998_199828


namespace NUMINAMATH_GPT_shaded_region_area_l1998_199858

open Real

noncomputable def area_of_shaded_region (r : ℝ) (s : ℝ) (d : ℝ) : ℝ := 
  (1/4) * π * r^2 + (1/2) * (d - s)^2

theorem shaded_region_area :
  let r := 3
  let s := 2
  let d := sqrt 5
  area_of_shaded_region r s d = 9 * π / 4 + (9 - 4 * sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1998_199858


namespace NUMINAMATH_GPT_gain_percent_is_25_l1998_199883

theorem gain_percent_is_25 (C S : ℝ) (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 :=
  sorry

end NUMINAMATH_GPT_gain_percent_is_25_l1998_199883


namespace NUMINAMATH_GPT_eq_970299_l1998_199850

theorem eq_970299 : 98^3 + 3 * 98^2 + 3 * 98 + 1 = 970299 :=
by
  sorry

end NUMINAMATH_GPT_eq_970299_l1998_199850


namespace NUMINAMATH_GPT_fraction_of_budget_is_31_percent_l1998_199848

def coffee_pastry_cost (B : ℝ) (c : ℝ) (p : ℝ) :=
  c = 0.25 * (B - p) ∧ p = 0.10 * (B - c)

theorem fraction_of_budget_is_31_percent (B c p : ℝ) (h : coffee_pastry_cost B c p) :
  c + p = 0.31 * B :=
sorry

end NUMINAMATH_GPT_fraction_of_budget_is_31_percent_l1998_199848


namespace NUMINAMATH_GPT_infinite_k_Q_ineq_l1998_199814

def Q (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem infinite_k_Q_ineq :
  ∃ᶠ k in at_top, Q (3 ^ k) > Q (3 ^ (k + 1)) := sorry

end NUMINAMATH_GPT_infinite_k_Q_ineq_l1998_199814


namespace NUMINAMATH_GPT_problem_statement_l1998_199824

theorem problem_statement (a b c : ℝ) (h₀ : 4 * a - 4 * b + c > 0) (h₁ : a + 2 * b + c < 0) : b^2 > a * c :=
sorry

end NUMINAMATH_GPT_problem_statement_l1998_199824


namespace NUMINAMATH_GPT_solve_inequality_l1998_199884

theorem solve_inequality (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x → x < -3 ∨ x > 5 / 3 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1998_199884


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1998_199802

variable (x y : ℝ)

theorem sum_of_reciprocals (h1 : x + y = 10) (h2 : x * y = 20) : 1 / x + 1 / y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1998_199802


namespace NUMINAMATH_GPT_elevator_height_after_20_seconds_l1998_199847

-- Conditions
def starting_height : ℕ := 120
def descending_speed : ℕ := 4
def time_elapsed : ℕ := 20

-- Statement to prove
theorem elevator_height_after_20_seconds : 
  starting_height - descending_speed * time_elapsed = 40 := 
by 
  sorry

end NUMINAMATH_GPT_elevator_height_after_20_seconds_l1998_199847


namespace NUMINAMATH_GPT_distinct_integer_roots_l1998_199887

theorem distinct_integer_roots (a : ℤ) : 
  (∃ u v : ℤ, u ≠ v ∧ (u + v = -a) ∧ (u * v = 2 * a)) ↔ a = -1 ∨ a = 9 :=
by
  sorry

end NUMINAMATH_GPT_distinct_integer_roots_l1998_199887


namespace NUMINAMATH_GPT_find_angle_l1998_199870

-- Given the complement condition
def complement_condition (x : ℝ) : Prop :=
  x + 2 * (4 * x + 10) = 90

-- Proving the degree measure of the angle
theorem find_angle (x : ℝ) : complement_condition x → x = 70 / 9 := by
  intro hc
  sorry

end NUMINAMATH_GPT_find_angle_l1998_199870


namespace NUMINAMATH_GPT_one_third_eleven_y_plus_three_l1998_199821

theorem one_third_eleven_y_plus_three (y : ℝ) : 
  (1/3) * (11 * y + 3) = 11 * y / 3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_one_third_eleven_y_plus_three_l1998_199821


namespace NUMINAMATH_GPT_area_of_common_region_l1998_199826

noncomputable def common_area (length : ℝ) (width : ℝ) (radius : ℝ) : ℝ :=
  let pi := Real.pi
  let sector_area := (pi * radius^2 / 4) * 4
  let triangle_area := (1 / 2) * (width / 2) * (length / 2) * 4
  sector_area - triangle_area

theorem area_of_common_region :
  common_area 10 (Real.sqrt 18) 3 = 9 * (Real.pi) - 9 :=
by
  sorry

end NUMINAMATH_GPT_area_of_common_region_l1998_199826


namespace NUMINAMATH_GPT_f_1996x_eq_1996_f_x_l1998_199890

theorem f_1996x_eq_1996_f_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by
  sorry

end NUMINAMATH_GPT_f_1996x_eq_1996_f_x_l1998_199890


namespace NUMINAMATH_GPT_ElaCollected13Pounds_l1998_199867

def KimberleyCollection : ℕ := 10
def HoustonCollection : ℕ := 12
def TotalCollection : ℕ := 35

def ElaCollection : ℕ := TotalCollection - KimberleyCollection - HoustonCollection

theorem ElaCollected13Pounds : ElaCollection = 13 := sorry

end NUMINAMATH_GPT_ElaCollected13Pounds_l1998_199867


namespace NUMINAMATH_GPT_find_a_b_l1998_199869

def satisfies_digit_conditions (n a b : ℕ) : Prop :=
  n = 2000 + 100 * a + 90 + b ∧
  n / 1000 % 10 = 2 ∧
  n / 100 % 10 = a ∧
  n / 10 % 10 = 9 ∧
  n % 10 = b

theorem find_a_b : ∃ (a b : ℕ), 2^a * 9^b = 2000 + 100*a + 90 + b ∧ satisfies_digit_conditions (2^a * 9^b) a b :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l1998_199869


namespace NUMINAMATH_GPT_sum_of_integers_l1998_199819

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 15 :=
by {
    sorry
}

end NUMINAMATH_GPT_sum_of_integers_l1998_199819


namespace NUMINAMATH_GPT_find_a_for_square_binomial_l1998_199829

theorem find_a_for_square_binomial (a r s : ℝ) 
  (h1 : ax^2 + 18 * x + 9 = (r * x + s)^2)
  (h2 : a = r^2)
  (h3 : 2 * r * s = 18)
  (h4 : s^2 = 9) : 
  a = 9 := 
by sorry

end NUMINAMATH_GPT_find_a_for_square_binomial_l1998_199829


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l1998_199849

theorem molecular_weight_of_compound (C H O n : ℕ) 
    (atomic_weight_C : ℝ) (atomic_weight_H : ℝ) (atomic_weight_O : ℝ) 
    (total_weight : ℝ) 
    (h_C : C = 2) (h_H : H = 4) 
    (h_atomic_weight_C : atomic_weight_C = 12.01) 
    (h_atomic_weight_H : atomic_weight_H = 1.008) 
    (h_atomic_weight_O : atomic_weight_O = 16.00) 
    (h_total_weight : total_weight = 60) : 
    C * atomic_weight_C + H * atomic_weight_H + n * atomic_weight_O = total_weight → 
    n = 2 := 
sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l1998_199849


namespace NUMINAMATH_GPT_marble_221_is_green_l1998_199822

def marble_sequence_color (n : ℕ) : String :=
  let cycle_length := 15
  let red_count := 6
  let green_start := red_count + 1
  let green_end := red_count + 5
  let position := n % cycle_length
  if position ≠ 0 then
    let cycle_position := position
    if cycle_position <= red_count then "red"
    else if cycle_position <= green_end then "green"
    else "blue"
  else "blue"

theorem marble_221_is_green : marble_sequence_color 221 = "green" :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_marble_221_is_green_l1998_199822


namespace NUMINAMATH_GPT_unique_solution_7tuples_l1998_199863

theorem unique_solution_7tuples : 
  ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1/8 :=
sorry

end NUMINAMATH_GPT_unique_solution_7tuples_l1998_199863


namespace NUMINAMATH_GPT_total_distance_hiked_l1998_199889

-- Defining the distances Terrell hiked on Saturday and Sunday
def distance_Saturday : Real := 8.2
def distance_Sunday : Real := 1.6

-- Stating the theorem to prove the total distance
theorem total_distance_hiked : distance_Saturday + distance_Sunday = 9.8 := by
  sorry

end NUMINAMATH_GPT_total_distance_hiked_l1998_199889


namespace NUMINAMATH_GPT_inequality_proof_l1998_199894

theorem inequality_proof (a b : ℝ) (h : a - |b| > 0) : b + a > 0 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1998_199894


namespace NUMINAMATH_GPT_dot_product_a_b_l1998_199807

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (-1, 2)

theorem dot_product_a_b : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 := by
  sorry

end NUMINAMATH_GPT_dot_product_a_b_l1998_199807


namespace NUMINAMATH_GPT_arith_prog_sum_eq_l1998_199897

variable (a d : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n / 2) * (2 * a 1 + (n - 1) * d 1)

theorem arith_prog_sum_eq (n : ℕ) : 
  S a d (n + 3) - 3 * S a d (n + 2) + 3 * S a d (n + 1) - S a d n = 0 := 
sorry

end NUMINAMATH_GPT_arith_prog_sum_eq_l1998_199897


namespace NUMINAMATH_GPT_verify_monomial_properties_l1998_199862

def monomial : ℚ := -3/5 * (1:ℚ)^1 * (2:ℚ)^2

def coefficient (m : ℚ) : ℚ := -3/5  -- The coefficient of the monomial
def degree (m : ℚ) : ℕ := 3          -- The degree of the monomial

theorem verify_monomial_properties :
  coefficient monomial = -3/5 ∧ degree monomial = 3 :=
by
  sorry

end NUMINAMATH_GPT_verify_monomial_properties_l1998_199862


namespace NUMINAMATH_GPT_quadratic_roots_l1998_199813

theorem quadratic_roots (m : ℝ) (h_eq : ∃ α β : ℝ, (α + β = -4) ∧ (α * β = m) ∧ (|α - β| = 2)) : m = 5 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_l1998_199813


namespace NUMINAMATH_GPT_value_of_fraction_l1998_199880

variable {x y : ℝ}

theorem value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_value_of_fraction_l1998_199880
