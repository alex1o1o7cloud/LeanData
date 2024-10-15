import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_b_general_term_a_l88_8817

-- Definitions of sequences and given conditions
def a (n : ℕ) : ℕ := sorry -- The sequence a_n
def S (n : ℕ) : ℕ := sorry -- The sum of the first n terms S_n

axiom a1_condition : a 1 = 2
axiom recursion_formula (n : ℕ): S (n+1) = 4 * a n + 2

def b (n : ℕ) : ℕ := a (n+1) - 2 * a n -- Definition of b_n

-- Theorem 1: Prove that b_n is a geometric sequence
theorem geometric_sequence_b (n : ℕ) : ∃ q, ∀ m, b (m+1) = q * b m :=
  sorry

-- Theorem 2: Find the general term formula for a_n
theorem general_term_a (n : ℕ) : a n = n * 2^n :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_general_term_a_l88_8817


namespace NUMINAMATH_GPT_non_poli_sci_gpa_below_or_eq_3_is_10_l88_8816

-- Definitions based on conditions
def total_applicants : ℕ := 40
def poli_sci_majors : ℕ := 15
def gpa_above_3 : ℕ := 20
def poli_sci_gpa_above_3 : ℕ := 5

-- Derived conditions from the problem
def poli_sci_gpa_below_or_eq_3 : ℕ := poli_sci_majors - poli_sci_gpa_above_3
def total_gpa_below_or_eq_3 : ℕ := total_applicants - gpa_above_3
def non_poli_sci_gpa_below_or_eq_3 : ℕ := total_gpa_below_or_eq_3 - poli_sci_gpa_below_or_eq_3

-- Statement to be proven
theorem non_poli_sci_gpa_below_or_eq_3_is_10 : non_poli_sci_gpa_below_or_eq_3 = 10 := by
  sorry

end NUMINAMATH_GPT_non_poli_sci_gpa_below_or_eq_3_is_10_l88_8816


namespace NUMINAMATH_GPT_units_digit_3542_pow_876_l88_8832

theorem units_digit_3542_pow_876 : (3542 ^ 876) % 10 = 6 := by 
  sorry

end NUMINAMATH_GPT_units_digit_3542_pow_876_l88_8832


namespace NUMINAMATH_GPT_rope_length_loss_l88_8868

theorem rope_length_loss
  (stories_needed : ℕ)
  (feet_per_story : ℕ)
  (pieces_of_rope : ℕ)
  (feet_per_rope : ℕ)
  (total_feet_needed : ℕ)
  (total_feet_bought : ℕ)
  (percentage_lost : ℕ) :
  
  stories_needed = 6 →
  feet_per_story = 10 →
  pieces_of_rope = 4 →
  feet_per_rope = 20 →
  total_feet_needed = stories_needed * feet_per_story →
  total_feet_bought = pieces_of_rope * feet_per_rope →
  total_feet_needed <= total_feet_bought →
  percentage_lost = ((total_feet_bought - total_feet_needed) * 100) / total_feet_bought →
  percentage_lost = 25 :=
by
  intros h_stories h_feet_story h_pieces h_feet_rope h_total_needed h_total_bought h_needed_bought h_percentage
  sorry

end NUMINAMATH_GPT_rope_length_loss_l88_8868


namespace NUMINAMATH_GPT_arithmetic_sequence_k_value_l88_8874

theorem arithmetic_sequence_k_value (a_1 d : ℕ) (h1 : a_1 = 1) (h2 : d = 2) (k : ℕ) (S : ℕ → ℕ) (h_sum : ∀ n, S n = n * (2 * a_1 + (n - 1) * d) / 2) (h_condition : S (k + 2) - S k = 24) : k = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_k_value_l88_8874


namespace NUMINAMATH_GPT_angle_B_range_l88_8845

theorem angle_B_range (A B C : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : A + B + C = 180) (h4 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_range_l88_8845


namespace NUMINAMATH_GPT_red_marbles_count_l88_8842

theorem red_marbles_count (R : ℕ) (h1 : 48 - R > 0) (h2 : ((48 - R) / 48 : ℚ) * ((48 - R) / 48) = 9 / 16) : R = 12 :=
sorry

end NUMINAMATH_GPT_red_marbles_count_l88_8842


namespace NUMINAMATH_GPT_total_angles_sum_l88_8870

variables (A B C D E : Type)
variables (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ)

-- Conditions about the geometry
axiom angle_triangle_ABC : angle1 + angle2 + angle3 = 180
axiom angle_triangle_BDE : angle7 + angle4 + angle5 = 180
axiom shared_angle_B : angle2 + angle7 = 180 -- since they form a straight line at vertex B

-- Proof statement
theorem total_angles_sum (A B C D E : Type) (angle1 angle2 angle3 angle4 angle5 angle7 : ℝ) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle7 - 180 = 180 :=
by
  sorry

end NUMINAMATH_GPT_total_angles_sum_l88_8870


namespace NUMINAMATH_GPT_water_hydrogen_oxygen_ratio_l88_8830

/-- In a mixture of water with a total mass of 171 grams, 
    where 19 grams are hydrogen, the ratio of hydrogen to oxygen by mass is 1:8. -/
theorem water_hydrogen_oxygen_ratio 
  (h_total_mass : ℝ) 
  (h_mass : ℝ) 
  (o_mass : ℝ) 
  (h_condition : h_total_mass = 171) 
  (h_hydrogen_mass : h_mass = 19) 
  (h_oxygen_mass : o_mass = h_total_mass - h_mass) :
  h_mass / o_mass = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_water_hydrogen_oxygen_ratio_l88_8830


namespace NUMINAMATH_GPT_simplify_expr_l88_8833

-- Define the condition on b
def condition (b : ℚ) : Prop :=
  b ≠ -1 / 2

-- Define the expression to be evaluated
def expression (b : ℚ) : ℚ :=
  1 - 1 / (1 + b / (1 + b))

-- Define the simplified form
def simplified_expr (b : ℚ) : ℚ :=
  b / (1 + 2 * b)

-- The theorem statement showing the equivalence
theorem simplify_expr (b : ℚ) (h : condition b) : expression b = simplified_expr b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l88_8833


namespace NUMINAMATH_GPT_jacob_three_heads_probability_l88_8853

noncomputable section

def probability_three_heads_after_two_tails : ℚ := 1 / 96

theorem jacob_three_heads_probability :
  let p := (1 / 2) ^ 4 * (1 / 6)
  p = probability_three_heads_after_two_tails := by
sorry

end NUMINAMATH_GPT_jacob_three_heads_probability_l88_8853


namespace NUMINAMATH_GPT_min_hypotenuse_l88_8835

theorem min_hypotenuse {a b : ℝ} (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 10) :
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ c ≥ 5 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_hypotenuse_l88_8835


namespace NUMINAMATH_GPT_fractional_part_exceeds_bound_l88_8803

noncomputable def x (a b : ℕ) : ℝ := Real.sqrt a + Real.sqrt b

theorem fractional_part_exceeds_bound
  (a b : ℕ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hx_not_int : ¬ (∃ n : ℤ, x a b = n))
  (hx_lt : x a b < 1976) :
    x a b % 1 > 3.24e-11 :=
sorry

end NUMINAMATH_GPT_fractional_part_exceeds_bound_l88_8803


namespace NUMINAMATH_GPT_smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l88_8854

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def has_prime_factor_less_than (n k : ℕ) : Prop := ∃ p : ℕ, p < k ∧ is_prime p ∧ p ∣ n

theorem smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60 :
  ∃ m : ℕ, 
    m = 4091 ∧ 
    ¬is_prime m ∧ 
    ¬is_square m ∧ 
    ¬has_prime_factor_less_than m 60 ∧ 
    (∀ n : ℕ, ¬is_prime n ∧ ¬is_square n ∧ ¬has_prime_factor_less_than n 60 → 4091 ≤ n) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l88_8854


namespace NUMINAMATH_GPT_scenario1_winner_scenario2_winner_l88_8827

def optimal_play_winner1 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 6 = 0 then "Balázs"
  else "Anna"

def optimal_play_winner2 (n : ℕ) (start_by_anna : Bool := true) : String :=
  if n % 4 = 0 then "Balázs"
  else "Anna"

theorem scenario1_winner:
  optimal_play_winner1 39 true = "Balázs" :=
by 
  sorry

theorem scenario2_winner:
  optimal_play_winner2 39 true = "Anna" :=
by
  sorry

end NUMINAMATH_GPT_scenario1_winner_scenario2_winner_l88_8827


namespace NUMINAMATH_GPT_clothes_donation_l88_8875

variable (initial_clothes : ℕ)
variable (clothes_thrown_away : ℕ)
variable (final_clothes : ℕ)
variable (x : ℕ)

theorem clothes_donation (h1 : initial_clothes = 100) 
                        (h2 : clothes_thrown_away = 15) 
                        (h3 : final_clothes = 65) 
                        (h4 : 4 * x = initial_clothes - final_clothes - clothes_thrown_away) :
  x = 5 := by
  sorry

end NUMINAMATH_GPT_clothes_donation_l88_8875


namespace NUMINAMATH_GPT_volunteer_hours_per_year_l88_8801

def volunteer_sessions_per_month := 2
def hours_per_session := 3
def months_per_year := 12

theorem volunteer_hours_per_year : 
  (volunteer_sessions_per_month * months_per_year * hours_per_session) = 72 := 
by
  sorry

end NUMINAMATH_GPT_volunteer_hours_per_year_l88_8801


namespace NUMINAMATH_GPT_max_sin_a_given_sin_a_plus_b_l88_8883

theorem max_sin_a_given_sin_a_plus_b (a b : ℝ) (sin_add : Real.sin (a + b) = Real.sin a + Real.sin b) : 
  Real.sin a ≤ 1 := 
sorry

end NUMINAMATH_GPT_max_sin_a_given_sin_a_plus_b_l88_8883


namespace NUMINAMATH_GPT_find_a_values_l88_8828

theorem find_a_values (a x₁ x₂ : ℝ) (h1 : x^2 + a * x - 2 = 0)
                      (h2 : x₁ ≠ x₂)
                      (h3 : x₁^3 + 22 / x₂ = x₂^3 + 22 / x₁) :
                      a = 3 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_values_l88_8828


namespace NUMINAMATH_GPT_strictly_increasing_difference_l88_8847

variable {a b : ℝ}
variable {f g : ℝ → ℝ}

theorem strictly_increasing_difference
  (h_diff : ∀ x ∈ Set.Icc a b, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ g x)
  (h_eq : f a = g a)
  (h_diff_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x : ℝ) > (deriv g x : ℝ)) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end NUMINAMATH_GPT_strictly_increasing_difference_l88_8847


namespace NUMINAMATH_GPT_triangle_right_if_angle_difference_l88_8800

noncomputable def is_right_triangle (A B C : ℝ) : Prop := 
  A = 90

theorem triangle_right_if_angle_difference (A B C : ℝ) (h : A - B = C) (sum_angles : A + B + C = 180) :
  is_right_triangle A B C :=
  sorry

end NUMINAMATH_GPT_triangle_right_if_angle_difference_l88_8800


namespace NUMINAMATH_GPT_feasible_stations_l88_8822

theorem feasible_stations (n : ℕ) (h: n > 0) 
  (pairings : ∀ (i j : ℕ), i ≠ j → i < n → j < n → ∃ k, (i+k) % n = j ∨ (j+k) % n = i) : n = 4 :=
sorry

end NUMINAMATH_GPT_feasible_stations_l88_8822


namespace NUMINAMATH_GPT_racket_price_l88_8856

theorem racket_price (cost_sneakers : ℕ) (cost_outfit : ℕ) (total_spent : ℕ) 
  (h_sneakers : cost_sneakers = 200) 
  (h_outfit : cost_outfit = 250) 
  (h_total : total_spent = 750) : 
  (total_spent - cost_sneakers - cost_outfit) = 300 :=
sorry

end NUMINAMATH_GPT_racket_price_l88_8856


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l88_8829

-- Define the type for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetric point function with respect to the x-axis
def symmetricPointWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Define the specific point
def givenPoint : Point3D := { x := 2, y := 3, z := 4 }

-- State the theorem to be proven
theorem symmetric_point_coordinates : 
  symmetricPointWithRespectToXAxis givenPoint = { x := 2, y := -3, z := -4 } :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l88_8829


namespace NUMINAMATH_GPT_intersection_S_T_eq_T_l88_8846

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end NUMINAMATH_GPT_intersection_S_T_eq_T_l88_8846


namespace NUMINAMATH_GPT_students_in_miss_evans_class_l88_8852

theorem students_in_miss_evans_class
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (contribution_per_student : ℕ)
  (remaining_contribution : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : contribution_per_student = 4)
  (h4 : remaining_contribution = total_contribution - class_funds)
  (h5 : num_students = remaining_contribution / contribution_per_student)
  : num_students = 19 :=
sorry

end NUMINAMATH_GPT_students_in_miss_evans_class_l88_8852


namespace NUMINAMATH_GPT_log_sum_eq_five_l88_8879

variable {a : ℕ → ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 3 * a n 

def sum_condition (a : ℕ → ℝ) : Prop :=
a 2 + a 4 + a 9 = 9

-- The mathematical statement to prove
theorem log_sum_eq_five (h1 : geometric_seq a) (h2 : sum_condition a) :
  Real.logb 3 (a 5 + a 7 + a 9) = 5 := 
sorry

end NUMINAMATH_GPT_log_sum_eq_five_l88_8879


namespace NUMINAMATH_GPT_chairs_bought_l88_8823

theorem chairs_bought (C : ℕ) (tables chairs total_time time_per_furniture : ℕ)
  (h1 : tables = 4)
  (h2 : time_per_furniture = 6)
  (h3 : total_time = 48)
  (h4 : total_time = time_per_furniture * (tables + chairs)) :
  C = 4 :=
by
  -- proof steps are omitted
  sorry

end NUMINAMATH_GPT_chairs_bought_l88_8823


namespace NUMINAMATH_GPT_files_more_than_apps_l88_8896

def initial_apps : ℕ := 11
def initial_files : ℕ := 3
def remaining_apps : ℕ := 2
def remaining_files : ℕ := 24

theorem files_more_than_apps : remaining_files - remaining_apps = 22 :=
by
  sorry

end NUMINAMATH_GPT_files_more_than_apps_l88_8896


namespace NUMINAMATH_GPT_exists_h_not_divisible_l88_8862

noncomputable def h : ℝ := 1969^2 / 1968

theorem exists_h_not_divisible (h := 1969^2 / 1968) :
  ∃ (h : ℝ), ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
by
  use h
  intro n
  sorry

end NUMINAMATH_GPT_exists_h_not_divisible_l88_8862


namespace NUMINAMATH_GPT_geometric_sequence_solve_a1_l88_8873

noncomputable def geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
    (h2 : a 2 = 1) (h3 : a 3 * a 9 = 2 * (a 5 ^ 2)) :=
  a 1 = (Real.sqrt 2) / 2

-- Define the main statement
theorem geometric_sequence_solve_a1 (a : ℕ → ℝ) (q : ℝ)
    (hq : 0 < q) (ha2 : a 2 = 1) (ha3_ha9 : a 3 * a 9 = 2 * (a 5 ^ 2)) :
    a 1 = (Real.sqrt 2) / 2 :=
sorry  -- The proof will be written here

end NUMINAMATH_GPT_geometric_sequence_solve_a1_l88_8873


namespace NUMINAMATH_GPT_rose_bushes_after_work_l88_8899

def initial_rose_bushes := 2
def planned_rose_bushes := 4
def planting_rate := 3
def removed_rose_bushes := 5

theorem rose_bushes_after_work :
  initial_rose_bushes + (planned_rose_bushes * planting_rate) - removed_rose_bushes = 9 :=
by
  sorry

end NUMINAMATH_GPT_rose_bushes_after_work_l88_8899


namespace NUMINAMATH_GPT_sum_first_75_odd_numbers_l88_8865

theorem sum_first_75_odd_numbers : (75^2) = 5625 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_75_odd_numbers_l88_8865


namespace NUMINAMATH_GPT_fish_per_bowl_l88_8857

theorem fish_per_bowl : 6003 / 261 = 23 := by
  sorry

end NUMINAMATH_GPT_fish_per_bowl_l88_8857


namespace NUMINAMATH_GPT_certain_number_sixth_powers_l88_8820

theorem certain_number_sixth_powers :
  ∃ N, (∀ n : ℕ, n < N → ∃ a : ℕ, n = a^6) ∧
       (∃ m ≤ N, (∀ n < m, ∃ k : ℕ, n = k^6) ∧ ¬ ∃ k : ℕ, m = k^6) :=
sorry

end NUMINAMATH_GPT_certain_number_sixth_powers_l88_8820


namespace NUMINAMATH_GPT_fixed_point_l88_8867

theorem fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : (1, 4) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 3)} :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_l88_8867


namespace NUMINAMATH_GPT_complex_inv_condition_l88_8837

theorem complex_inv_condition (i : ℂ) (h : i^2 = -1) : (i - 2 * i⁻¹)⁻¹ = -i / 3 :=
by
  sorry

end NUMINAMATH_GPT_complex_inv_condition_l88_8837


namespace NUMINAMATH_GPT_inequality_1_inequality_3_l88_8871

variable (a b : ℝ)
variable (hab : a > b ∧ b ≥ 2)

theorem inequality_1 (hab : a > b ∧ b ≥ 2) : b ^ 2 > 3 * b - a :=
by sorry

theorem inequality_3 (hab : a > b ∧ b ≥ 2) : a * b > a + b :=
by sorry

end NUMINAMATH_GPT_inequality_1_inequality_3_l88_8871


namespace NUMINAMATH_GPT_geo_sequence_necessity_l88_8815

theorem geo_sequence_necessity (a1 a2 a3 a4 : ℝ) (h_non_zero: a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧ a4 ≠ 0) :
  (a1 * a4 = a2 * a3) → (∀ r : ℝ, (a2 = a1 * r) ∧ (a3 = a2 * r) ∧ (a4 = a3 * r)) → False :=
sorry

end NUMINAMATH_GPT_geo_sequence_necessity_l88_8815


namespace NUMINAMATH_GPT_problem_I_problem_II_l88_8895

-- Definitions
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (a - 2) * x - Real.log x

-- Problem (I)
theorem problem_I (a : ℝ) (h_min : ∀ x : ℝ, function_f a 1 ≤ function_f a x) :
  a = 1 ∧ (∀ x : ℝ, 0 < x ∧ x < 1 → (function_f a x < function_f a 1)) ∧ (∀ x : ℝ, x > 1 → (function_f a x > function_f a 1)) :=
sorry

-- Problem (II)
theorem problem_II (a x0 : ℝ) (h_a_gt_1 : a > 1) (h_x0_pos : 0 < x0) (h_x0_lt_1 : x0 < 1)
    (h_min : ∀ x : ℝ, function_f a (1/a) ≤ function_f a x) :
  ∀ x : ℝ, function_f a 0 > 0
:= sorry

end NUMINAMATH_GPT_problem_I_problem_II_l88_8895


namespace NUMINAMATH_GPT_mike_net_spending_l88_8807

-- Definitions for given conditions
def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84

-- Theorem stating the result
theorem mike_net_spending : trumpet_cost - song_book_revenue = 139.32 :=
by 
  sorry

end NUMINAMATH_GPT_mike_net_spending_l88_8807


namespace NUMINAMATH_GPT_hannah_total_spent_l88_8806

-- Definitions based on conditions
def sweatshirts_bought : ℕ := 3
def t_shirts_bought : ℕ := 2
def cost_per_sweatshirt : ℕ := 15
def cost_per_t_shirt : ℕ := 10

-- Definition of the theorem that needs to be proved
theorem hannah_total_spent : 
  (sweatshirts_bought * cost_per_sweatshirt + t_shirts_bought * cost_per_t_shirt) = 65 :=
by
  sorry

end NUMINAMATH_GPT_hannah_total_spent_l88_8806


namespace NUMINAMATH_GPT_Janet_saves_154_minutes_per_week_l88_8898

-- Definitions for the time spent on each activity daily
def timeLookingForKeys := 8 -- minutes
def timeComplaining := 3 -- minutes
def timeSearchingForPhone := 5 -- minutes
def timeLookingForWallet := 4 -- minutes
def timeSearchingForSunglasses := 2 -- minutes

-- Total time spent daily on these activities
def totalDailyTime := timeLookingForKeys + timeComplaining + timeSearchingForPhone + timeLookingForWallet + timeSearchingForSunglasses
-- Time savings calculation for a week
def weeklySaving := totalDailyTime * 7

-- The proof statement that Janet will save 154 minutes every week
theorem Janet_saves_154_minutes_per_week : weeklySaving = 154 := by
  sorry

end NUMINAMATH_GPT_Janet_saves_154_minutes_per_week_l88_8898


namespace NUMINAMATH_GPT_price_of_battery_l88_8844

def cost_of_tire : ℕ := 42
def cost_of_tires (num_tires : ℕ) : ℕ := num_tires * cost_of_tire
def total_cost : ℕ := 224
def num_tires : ℕ := 4
def cost_of_battery : ℕ := total_cost - cost_of_tires num_tires

theorem price_of_battery : cost_of_battery = 56 := by
  sorry

end NUMINAMATH_GPT_price_of_battery_l88_8844


namespace NUMINAMATH_GPT_profit_calculation_l88_8838

-- Define conditions based on investments
def JohnInvestment := 700
def MikeInvestment := 300

-- Define the equality condition where John received $800 more than Mike
theorem profit_calculation (P : ℝ) 
  (h1 : (P / 6 + (7 / 10) * (2 * P / 3)) - (P / 6 + (3 / 10) * (2 * P / 3)) = 800) : 
  P = 3000 := 
sorry

end NUMINAMATH_GPT_profit_calculation_l88_8838


namespace NUMINAMATH_GPT_total_black_balls_l88_8848

-- Conditions
def number_of_white_balls (B : ℕ) : ℕ := 6 * B

def total_balls (B : ℕ) : ℕ := B + number_of_white_balls B

-- Theorem to prove
theorem total_black_balls (h : total_balls B = 56) : B = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_black_balls_l88_8848


namespace NUMINAMATH_GPT_total_painted_surface_area_l88_8834

-- Defining the conditions
def num_cubes := 19
def top_layer := 1
def middle_layer := 5
def bottom_layer := 13
def exposed_faces_top_layer := 5
def exposed_faces_middle_corner := 3
def exposed_faces_middle_center := 1
def exposed_faces_bottom_layer := 1

-- Question: How many square meters are painted?
theorem total_painted_surface_area : 
  let top_layer_area := top_layer * exposed_faces_top_layer
  let middle_layer_area := (4 * exposed_faces_middle_corner) + exposed_faces_middle_center
  let bottom_layer_area := bottom_layer * exposed_faces_bottom_layer
  top_layer_area + middle_layer_area + bottom_layer_area = 31 :=
by
  sorry

end NUMINAMATH_GPT_total_painted_surface_area_l88_8834


namespace NUMINAMATH_GPT_name_tag_area_l88_8880

-- Define the side length of the square
def side_length : ℕ := 11

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- State the theorem: the area of a square with side length of 11 cm is 121 cm²
theorem name_tag_area : square_area side_length = 121 :=
by
  sorry

end NUMINAMATH_GPT_name_tag_area_l88_8880


namespace NUMINAMATH_GPT_max_side_length_l88_8889

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_max_side_length_l88_8889


namespace NUMINAMATH_GPT_coordinates_of_P_l88_8810

open Real

theorem coordinates_of_P (P : ℝ × ℝ) (h1 : P.1 = 2 * cos (2 * π / 3)) (h2 : P.2 = 2 * sin (2 * π / 3)) :
  P = (-1, sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l88_8810


namespace NUMINAMATH_GPT_exterior_angle_BAC_l88_8866

theorem exterior_angle_BAC (angle_octagon angle_rectangle : ℝ) (h_oct_135 : angle_octagon = 135) (h_rec_90 : angle_rectangle = 90) :
  360 - (angle_octagon + angle_rectangle) = 135 := 
by
  simp [h_oct_135, h_rec_90]
  sorry

end NUMINAMATH_GPT_exterior_angle_BAC_l88_8866


namespace NUMINAMATH_GPT_sunny_lead_l88_8891

-- Define the context of the race
variables {s m : ℝ}  -- s: Sunny's speed, m: Misty's speed
variables (distance_first : ℝ) (distance_ahead_first : ℝ)
variables (additional_distance_sunny_second : ℝ) (correct_answer : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first = 400 ∧
  distance_ahead_first = 20 ∧
  additional_distance_sunny_second = 40 ∧
  correct_answer = 20 

-- The math proof problem in Lean 4
theorem sunny_lead (h : conditions distance_first distance_ahead_first additional_distance_sunny_second correct_answer) :
  ∀ s m : ℝ, s / m = (400 / 380 : ℝ) → 
  (s / m) * 400 + additional_distance_sunny_second = (m / s) * 440 + correct_answer :=
sorry

end NUMINAMATH_GPT_sunny_lead_l88_8891


namespace NUMINAMATH_GPT_pet_store_profit_is_205_l88_8849

def brandon_selling_price : ℤ := 100
def pet_store_selling_price : ℤ := 5 + 3 * brandon_selling_price
def pet_store_profit : ℤ := pet_store_selling_price - brandon_selling_price

theorem pet_store_profit_is_205 :
  pet_store_profit = 205 := by
  sorry

end NUMINAMATH_GPT_pet_store_profit_is_205_l88_8849


namespace NUMINAMATH_GPT_binom_12_9_eq_220_l88_8893

open Nat

theorem binom_12_9_eq_220 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_GPT_binom_12_9_eq_220_l88_8893


namespace NUMINAMATH_GPT_tiger_time_to_pass_specific_point_l88_8884

theorem tiger_time_to_pass_specific_point :
  ∀ (distance_tree : ℝ) (time_tree : ℝ) (length_tiger : ℝ),
  distance_tree = 20 →
  time_tree = 5 →
  length_tiger = 5 →
  (length_tiger / (distance_tree / time_tree)) = 1.25 :=
by
  intros distance_tree time_tree length_tiger h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_tiger_time_to_pass_specific_point_l88_8884


namespace NUMINAMATH_GPT_cheryl_walking_speed_l88_8885

theorem cheryl_walking_speed (H : 12 = 6 * v) : v = 2 := 
by
  -- proof here
  sorry

end NUMINAMATH_GPT_cheryl_walking_speed_l88_8885


namespace NUMINAMATH_GPT_owen_work_hours_l88_8881

def total_hours := 24
def chores_hours := 7
def sleep_hours := 11

theorem owen_work_hours : total_hours - chores_hours - sleep_hours = 6 := by
  sorry

end NUMINAMATH_GPT_owen_work_hours_l88_8881


namespace NUMINAMATH_GPT_integer_inequality_l88_8860

theorem integer_inequality (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := 
  sorry

end NUMINAMATH_GPT_integer_inequality_l88_8860


namespace NUMINAMATH_GPT_number_for_B_expression_l88_8882

-- Define the number for A as a variable
variable (a : ℤ)

-- Define the number for B in terms of a
def number_for_B (a : ℤ) : ℤ := 2 * a - 1

-- Statement to prove
theorem number_for_B_expression (a : ℤ) : number_for_B a = 2 * a - 1 := by
  sorry

end NUMINAMATH_GPT_number_for_B_expression_l88_8882


namespace NUMINAMATH_GPT_division_problem_l88_8825

theorem division_problem
  (R : ℕ) (D : ℕ) (Q : ℕ) (Div : ℕ)
  (hR : R = 5)
  (hD1 : D = 3 * Q)
  (hD2 : D = 3 * R + 3) :
  Div = D * Q + R :=
by
  have hR : R = 5 := hR
  have hD2 := hD2
  have hDQ := hD1
  -- Proof continues with steps leading to the final desired conclusion
  sorry

end NUMINAMATH_GPT_division_problem_l88_8825


namespace NUMINAMATH_GPT_greatest_possible_integer_l88_8836

theorem greatest_possible_integer (m : ℕ) (h1 : m < 150) (h2 : ∃ a : ℕ, m = 10 * a - 2) (h3 : ∃ b : ℕ, m = 9 * b - 4) : m = 68 := 
  by sorry

end NUMINAMATH_GPT_greatest_possible_integer_l88_8836


namespace NUMINAMATH_GPT_simplify_cube_root_18_24_30_l88_8869

noncomputable def cube_root_simplification (a b c : ℕ) : ℕ :=
  let sum_cubes := a^3 + b^3 + c^3
  36

theorem simplify_cube_root_18_24_30 : 
  cube_root_simplification 18 24 30 = 36 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_simplify_cube_root_18_24_30_l88_8869


namespace NUMINAMATH_GPT_Jia_age_is_24_l88_8814

variable (Jia Yi Bing Ding : ℕ)

theorem Jia_age_is_24
  (h1 : (Jia + Yi + Bing) / 3 = (Jia + Yi + Bing + Ding) / 4 + 1)
  (h2 : (Jia + Yi) / 2 = (Jia + Yi + Bing) / 3 + 1)
  (h3 : Jia = Yi + 4)
  (h4 : Ding = 17) :
  Jia = 24 :=
by
  sorry

end NUMINAMATH_GPT_Jia_age_is_24_l88_8814


namespace NUMINAMATH_GPT_helen_chocolate_chip_cookies_l88_8839

theorem helen_chocolate_chip_cookies :
  let cookies_yesterday := 527
  let cookies_morning := 554
  cookies_yesterday + cookies_morning = 1081 :=
by
  let cookies_yesterday := 527
  let cookies_morning := 554
  show cookies_yesterday + cookies_morning = 1081
  -- The proof is omitted according to the provided instructions 
  sorry

end NUMINAMATH_GPT_helen_chocolate_chip_cookies_l88_8839


namespace NUMINAMATH_GPT_loan_repayment_l88_8890

open Real

theorem loan_repayment
  (a r : ℝ) (h_r : 0 ≤ r) :
  ∃ x : ℝ, 
    x = (a * r * (1 + r)^5) / ((1 + r)^5 - 1) :=
sorry

end NUMINAMATH_GPT_loan_repayment_l88_8890


namespace NUMINAMATH_GPT_total_number_of_chips_l88_8897

theorem total_number_of_chips 
  (viviana_chocolate : ℕ) (susana_chocolate : ℕ) (viviana_vanilla : ℕ) (susana_vanilla : ℕ)
  (manuel_vanilla : ℕ) (manuel_chocolate : ℕ)
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : susana_chocolate = 25)
  (h5 : manuel_vanilla = 2 * susana_vanilla)
  (h6 : manuel_chocolate = viviana_chocolate / 2) :
  viviana_chocolate + susana_chocolate + manuel_chocolate + viviana_vanilla + susana_vanilla + manuel_vanilla = 135 :=
sorry

end NUMINAMATH_GPT_total_number_of_chips_l88_8897


namespace NUMINAMATH_GPT_avg_zits_per_kid_mr_jones_class_l88_8805

-- Define the conditions
def avg_zits_ms_swanson_class := 5
def num_kids_ms_swanson_class := 25
def num_kids_mr_jones_class := 32
def extra_zits_mr_jones_class := 67

-- Define the total number of zits in Ms. Swanson's class
def total_zits_ms_swanson_class := avg_zits_ms_swanson_class * num_kids_ms_swanson_class

-- Define the total number of zits in Mr. Jones' class
def total_zits_mr_jones_class := total_zits_ms_swanson_class + extra_zits_mr_jones_class

-- Define the problem statement to prove: the average number of zits per kid in Mr. Jones' class
theorem avg_zits_per_kid_mr_jones_class : 
  total_zits_mr_jones_class / num_kids_mr_jones_class = 6 := by
  sorry

end NUMINAMATH_GPT_avg_zits_per_kid_mr_jones_class_l88_8805


namespace NUMINAMATH_GPT_number_of_white_tiles_l88_8886

theorem number_of_white_tiles (n : ℕ) : 
  ∃ a_n : ℕ, a_n = 4 * n + 2 :=
sorry

end NUMINAMATH_GPT_number_of_white_tiles_l88_8886


namespace NUMINAMATH_GPT_area_of_region_l88_8851

def plane_region (x y : ℝ) : Prop := |x| ≤ 1 ∧ |y| ≤ 1

def inequality_holds (a b : ℝ) : Prop := ∀ x y : ℝ, plane_region x y → a * x - 2 * b * y ≤ 2

theorem area_of_region (a b : ℝ) (h : inequality_holds a b) : 
  (-2 ≤ a ∧ a ≤ 2) ∧ (-1 ≤ b ∧ b ≤ 1) ∧ (4 * 2 = 8) :=
sorry

end NUMINAMATH_GPT_area_of_region_l88_8851


namespace NUMINAMATH_GPT_rectangles_with_one_gray_cell_l88_8818

/- Definitions from conditions -/
def total_gray_cells : ℕ := 40
def blue_cells : ℕ := 36
def red_cells : ℕ := 4

/- The number of rectangles containing exactly one gray cell is the proof goal -/
theorem rectangles_with_one_gray_cell :
  (blue_cells * 4 + red_cells * 8) = 176 :=
sorry

end NUMINAMATH_GPT_rectangles_with_one_gray_cell_l88_8818


namespace NUMINAMATH_GPT_problem_7_sqrt_13_l88_8831

theorem problem_7_sqrt_13 : 
  let m := Int.floor (Real.sqrt 13)
  let n := 10 - Real.sqrt 13 - Int.floor (10 - Real.sqrt 13)
  m + n = 7 - Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_problem_7_sqrt_13_l88_8831


namespace NUMINAMATH_GPT_TylerWeightDifference_l88_8858

-- Define the problem conditions
def PeterWeight : ℕ := 65
def SamWeight : ℕ := 105
def TylerWeight := 2 * PeterWeight

-- State the theorem
theorem TylerWeightDifference : (TylerWeight - SamWeight = 25) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_TylerWeightDifference_l88_8858


namespace NUMINAMATH_GPT_total_seashells_l88_8859

-- Conditions
def sam_seashells : Nat := 18
def mary_seashells : Nat := 47

-- Theorem stating the question and the final answer
theorem total_seashells : sam_seashells + mary_seashells = 65 :=
by
  sorry

end NUMINAMATH_GPT_total_seashells_l88_8859


namespace NUMINAMATH_GPT_parabola_equation_l88_8812

theorem parabola_equation (p : ℝ) (h : 2 * p = 8) :
  ∃ (a : ℝ), a = 8 ∧ (y^2 = a * x ∨ y^2 = -a * x) :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l88_8812


namespace NUMINAMATH_GPT_all_visitors_can_buy_ticket_l88_8813

-- Define the coin types
inductive Coin
  | Three
  | Five

-- Define a function to calculate the total money from a list of coins
def totalMoney (coins : List Coin) : Int :=
  coins.foldr (fun c acc => acc + (match c with | Coin.Three => 3 | Coin.Five => 5)) 0

-- Define the initial state: each person has 22 tugriks in some combination of 3 and 5 tugrik coins
def initial_money := 22
def ticket_cost := 4

-- Each visitor and the cashier has 22 tugriks initially
axiom visitor_money_all_22 (n : Nat) : n ≤ 200 → totalMoney (List.replicate 2 Coin.Five ++ List.replicate 4 Coin.Three) = initial_money

-- We want to prove that all visitors can buy a ticket
theorem all_visitors_can_buy_ticket :
  ∀ n, n ≤ 200 → ∃ coins: List Coin, totalMoney coins = initial_money ∧ totalMoney coins ≥ ticket_cost := by
    sorry -- Proof goes here

end NUMINAMATH_GPT_all_visitors_can_buy_ticket_l88_8813


namespace NUMINAMATH_GPT_population_net_increase_one_day_l88_8841

-- Define the given rates and constants
def birth_rate := 10 -- people per 2 seconds
def death_rate := 2 -- people per 2 seconds
def seconds_per_day := 24 * 60 * 60 -- seconds

-- Define the expected net population increase per second
def population_increase_per_sec := (birth_rate / 2) - (death_rate / 2)

-- Define the expected net population increase per day
def expected_population_increase_per_day := population_increase_per_sec * seconds_per_day

theorem population_net_increase_one_day :
  expected_population_increase_per_day = 345600 := by
  -- This will skip the proof implementation.
  sorry

end NUMINAMATH_GPT_population_net_increase_one_day_l88_8841


namespace NUMINAMATH_GPT_intersection_eq_union_eq_l88_8864

def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | 1 < x ∧ x ≤ 4 }

theorem intersection_eq : A ∩ B = { x : ℝ | 2 ≤ x ∧ x ≤ 4 } :=
by sorry

theorem union_eq : A ∪ B = { x : ℝ | 1 < x } :=
by sorry

end NUMINAMATH_GPT_intersection_eq_union_eq_l88_8864


namespace NUMINAMATH_GPT_opposite_of_two_thirds_l88_8843

theorem opposite_of_two_thirds : - (2/3) = -2/3 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_two_thirds_l88_8843


namespace NUMINAMATH_GPT_min_value_expression_l88_8826

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ (m : ℝ), m = 3 / 2 ∧ ∀ t > 0, (2 * x / (x + 2 * y) + y / x) ≥ m :=
by
  use 3 / 2
  sorry

end NUMINAMATH_GPT_min_value_expression_l88_8826


namespace NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l88_8819

theorem solve_first_equation (x : ℝ) : (8 * x = -2 * (x + 5)) → (x = -1) :=
by
  intro h
  sorry

theorem solve_second_equation (x : ℝ) : ((x - 1) / 4 = (5 * x - 7) / 6 + 1) → (x = -1 / 7) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_first_equation_solve_second_equation_l88_8819


namespace NUMINAMATH_GPT_num_parallel_edge_pairs_correct_l88_8850

-- Define a rectangular prism with given dimensions
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

-- Function to count the number of pairs of parallel edges
def num_parallel_edge_pairs (p : RectangularPrism) : ℕ :=
  4 * ((p.length + p.width + p.height) - 3)

-- Given conditions
def given_prism : RectangularPrism := { length := 4, width := 3, height := 2 }

-- Main theorem statement
theorem num_parallel_edge_pairs_correct :
  num_parallel_edge_pairs given_prism = 12 :=
by
  -- Skipping proof steps
  sorry

end NUMINAMATH_GPT_num_parallel_edge_pairs_correct_l88_8850


namespace NUMINAMATH_GPT_derivative_of_x_log_x_l88_8892

noncomputable def y (x : ℝ) := x * Real.log x

theorem derivative_of_x_log_x (x : ℝ) (hx : 0 < x) :
  (deriv y x) = Real.log x + 1 :=
sorry

end NUMINAMATH_GPT_derivative_of_x_log_x_l88_8892


namespace NUMINAMATH_GPT_inequality_solution_l88_8811

theorem inequality_solution (x : ℝ) (hx : x > 0) : 
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l88_8811


namespace NUMINAMATH_GPT_geese_population_1996_l88_8861

theorem geese_population_1996 (k x : ℝ) 
  (h1 : x - 39 = k * 60) 
  (h2 : 123 - 60 = k * x) : 
  x = 84 := 
by
  sorry

end NUMINAMATH_GPT_geese_population_1996_l88_8861


namespace NUMINAMATH_GPT_Kendall_dimes_l88_8824

theorem Kendall_dimes (total_value : ℝ) (quarters : ℝ) (dimes : ℝ) (nickels : ℝ) 
  (num_quarters : ℕ) (num_nickels : ℕ) 
  (total_amount : total_value = 4)
  (quarter_amount : quarters = num_quarters * 0.25)
  (num_quarters_val : num_quarters = 10)
  (nickel_amount : nickels = num_nickels * 0.05) 
  (num_nickels_val : num_nickels = 6) :
  dimes = 12 := by
  sorry

end NUMINAMATH_GPT_Kendall_dimes_l88_8824


namespace NUMINAMATH_GPT_quadratic_root_property_l88_8876

theorem quadratic_root_property (m p : ℝ) 
  (h1 : (p^2 - 2 * p + m - 1 = 0)) 
  (h2 : (p^2 - 2 * p + 3) * (m + 4) = 7)
  (h3 : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 - 2 * r1 + m - 1 = 0 ∧ r2^2 - 2 * r2 + m - 1 = 0) : 
  m = -3 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_root_property_l88_8876


namespace NUMINAMATH_GPT_minimum_n_minus_m_abs_l88_8840

theorem minimum_n_minus_m_abs (f g : ℝ → ℝ)
  (hf : ∀ x, f x = Real.exp x + 2 * x)
  (hg : ∀ x, g x = 4 * x)
  (m n : ℝ)
  (h_cond : f m = g n) :
  |n - m| = (1 / 2) - (1 / 2) * Real.log 2 := 
sorry

end NUMINAMATH_GPT_minimum_n_minus_m_abs_l88_8840


namespace NUMINAMATH_GPT_restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l88_8877

-- Let P be the original price of the jacket
variable (P : ℝ)

-- The price of the jacket after successive reductions
def price_after_discount (P : ℝ) : ℝ := 0.60 * P

-- The price of the jacket after all discounts including the limited-time offer
def price_after_full_discount (P : ℝ) : ℝ := 0.54 * P

-- Prove that to restore 0.60P back to P a 66.67% increase is needed
theorem restore_to_original_without_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.60 * P) * (1 + 66.67 / 100) = P :=
by sorry

-- Prove that to restore 0.54P back to P an 85.19% increase is needed
theorem restore_to_original_with_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.54 * P) * (1 + 85.19 / 100) = P :=
by sorry

end NUMINAMATH_GPT_restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l88_8877


namespace NUMINAMATH_GPT_find_pq_l88_8894

theorem find_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hline : ∀ x y : ℝ, px + qy = 24) 
  (harea : (1 / 2) * (24 / p) * (24 / q) = 48) : p * q = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_pq_l88_8894


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l88_8855

variable (x a : ℝ)

-- Condition 1: For all x in [1, 2], x^2 - a ≥ 0
def condition1 (x a : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition 2: There exists an x in ℝ such that x^2 + 2ax + 2 - a = 0
def condition2 (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Proof problem: The necessary and sufficient condition for p ∧ q is a ≤ -2 ∨ a = 1
theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) ↔ (a ≤ -2 ∨ a = 1) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l88_8855


namespace NUMINAMATH_GPT_perpendicular_line_through_circle_center_l88_8887

theorem perpendicular_line_through_circle_center :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 2*x - 8 = 0 → x + 2*y = 0 → a * x + b * y + c = 0) ∧
  a = 2 ∧ b = -1 ∧ c = -2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_through_circle_center_l88_8887


namespace NUMINAMATH_GPT_perpendicular_vectors_dot_product_zero_l88_8809

theorem perpendicular_vectors_dot_product_zero (m : ℝ) :
  let a := (1, 2)
  let b := (m + 1, -m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 1 :=
by
  intros a b h_eq
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_dot_product_zero_l88_8809


namespace NUMINAMATH_GPT_cuboid_surface_area_l88_8802

/--
Given a cuboid with length 10 cm, breadth 8 cm, and height 6 cm, the surface area is 376 cm².
-/
theorem cuboid_surface_area 
  (length : ℝ) 
  (breadth : ℝ) 
  (height : ℝ) 
  (h_length : length = 10) 
  (h_breadth : breadth = 8) 
  (h_height : height = 6) : 
  2 * (length * height + length * breadth + breadth * height) = 376 := 
by 
  -- Replace these placeholders with the actual proof steps.
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_l88_8802


namespace NUMINAMATH_GPT_inversely_proportional_rs_l88_8821

theorem inversely_proportional_rs (r s : ℝ) (k : ℝ) 
(h_invprop : r * s = k) 
(h1 : r = 40) (h2 : s = 5) 
(h3 : s = 8) : r = 25 := by
  sorry

end NUMINAMATH_GPT_inversely_proportional_rs_l88_8821


namespace NUMINAMATH_GPT_cube_surface_area_increase_l88_8804

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end NUMINAMATH_GPT_cube_surface_area_increase_l88_8804


namespace NUMINAMATH_GPT_perp_DM_PN_l88_8888

-- Definitions of the triangle and its elements
variables {A B C M N P D : Point}
variables (triangle_incircle_touch : ∀ (A B C : Point) (triangle : Triangle ABC),
  touches_incircle_at triangle B C M ∧ 
  touches_incircle_at triangle C A N ∧ 
  touches_incircle_at triangle A B P)
variables (point_D : lies_on_segment D N P)
variables {BD CD DP DN : ℝ}
variables (ratio_condition : DP / DN = BD / CD)

-- The theorem statement
theorem perp_DM_PN 
  (h1 : triangle_incircle_touch A B C) 
  (h2 : point_D)
  (h3 : ratio_condition) : 
  is_perpendicular D M P N := 
sorry

end NUMINAMATH_GPT_perp_DM_PN_l88_8888


namespace NUMINAMATH_GPT_commute_times_variance_l88_8863

theorem commute_times_variance (x y : ℝ) :
  (x + y + 10 + 11 + 9) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2 →
  |x - y| = 4 :=
by
  sorry

end NUMINAMATH_GPT_commute_times_variance_l88_8863


namespace NUMINAMATH_GPT_difference_of_numbers_l88_8872

theorem difference_of_numbers (x y : ℕ) (h₁ : x + y = 50) (h₂ : Nat.gcd x y = 5) :
  (x - y = 20 ∨ y - x = 20 ∨ x - y = 40 ∨ y - x = 40) :=
sorry

end NUMINAMATH_GPT_difference_of_numbers_l88_8872


namespace NUMINAMATH_GPT_ratio_of_S_to_R_l88_8878

noncomputable def find_ratio (total_amount : ℕ) (diff_SP : ℕ) (n : ℕ) (k : ℕ) (P : ℕ) (Q : ℕ) (R : ℕ) (S : ℕ) (ratio_SR : ℕ) :=
  Q = n ∧ R = n ∧ P = k * n ∧ S = ratio_SR * n ∧ P + Q + R + S = total_amount ∧ S - P = diff_SP

theorem ratio_of_S_to_R :
  ∃ n k ratio_SR, k = 2 ∧ ratio_SR = 4 ∧ 
  find_ratio 1000 250 n k 250 125 125 500 ratio_SR :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_S_to_R_l88_8878


namespace NUMINAMATH_GPT_increase_is_50_percent_l88_8808

theorem increase_is_50_percent (original new : ℕ) (h1 : original = 60) (h2 : new = 90) :
  ((new - original) * 100 / original) = 50 :=
by
  -- Proof can be filled here.
  sorry

end NUMINAMATH_GPT_increase_is_50_percent_l88_8808
