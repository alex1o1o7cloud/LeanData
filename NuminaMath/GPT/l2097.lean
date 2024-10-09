import Mathlib

namespace min_value_fraction_l2097_209769

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y, y = x - 4 ∧ (x + 11) / Real.sqrt (x - 4) = 2 * Real.sqrt 15 := by
  sorry

end min_value_fraction_l2097_209769


namespace relationship_y1_y2_y3_l2097_209779

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l2097_209779


namespace rectangle_A_plus_P_ne_162_l2097_209741

theorem rectangle_A_plus_P_ne_162 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ) 
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : A + P ≠ 162 :=
by
  sorry

end rectangle_A_plus_P_ne_162_l2097_209741


namespace people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l2097_209737

def f (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
  else if 9 ≤ n ∧ n ≤ 32 then 360 * 3 ^ ((n - 8) / 12) + 3000
  else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
  else 0 -- default case for unsupported values

def g (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 18 then 0
  else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
  else if 33 ≤ n ∧ n ≤ 45 then 8800
  else 0 -- default case for unsupported values

theorem people_entering_2pm_to_3pm :
  f 21 + f 22 + f 23 + f 24 = 17460 := sorry

theorem people_leaving_2pm_to_3pm :
  g 21 + g 22 + g 23 + g 24 = 9000 := sorry

theorem peak_visitors_time :
  ∀ n, 1 ≤ n ∧ n ≤ 45 → 
    (n = 28 ↔ ∀ m, 1 ≤ m ∧ m ≤ 45 → f m - g m ≤ f 28 - g 28) := sorry

end people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l2097_209737


namespace negation_proof_l2097_209793

theorem negation_proof :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
by
  -- proof goes here
  sorry

end negation_proof_l2097_209793


namespace blocks_added_l2097_209723

theorem blocks_added (a b : Nat) (h₁ : a = 86) (h₂ : b = 95) : b - a = 9 :=
by
  sorry

end blocks_added_l2097_209723


namespace min_sum_first_n_terms_l2097_209726

variable {a₁ d c : ℝ} (n : ℕ)

noncomputable def sum_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem min_sum_first_n_terms (h₁ : ∀ x, 1/3 ≤ x ∧ x ≤ 4/5 → a₁ * x^2 + (d/2 - a₁) * x + c ≥ 0)
                              (h₂ : a₁ = -15/4 * d)
                              (h₃ : d > 0) :
                              ∃ n : ℕ, n > 0 ∧ sum_first_n_terms a₁ d n ≤ sum_first_n_terms a₁ d 4 :=
by
  use 4
  sorry

end min_sum_first_n_terms_l2097_209726


namespace digit_difference_one_l2097_209787

variable (d C D : ℕ)

-- Assumptions
variables (h1 : d > 8)
variables (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3)

theorem digit_difference_one (h1 : d > 8) (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3) :
  C - D = 1 :=
by
  sorry

end digit_difference_one_l2097_209787


namespace distance_between_trees_l2097_209707

theorem distance_between_trees (num_trees: ℕ) (total_length: ℕ) (trees_at_end: ℕ) 
(h1: num_trees = 26) (h2: total_length = 300) (h3: trees_at_end = 2) :
  total_length / (num_trees - 1) = 12 :=
by sorry

end distance_between_trees_l2097_209707


namespace students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l2097_209740

noncomputable def numStudentsKnowingSecret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem students_on_seventh_day :
  (numStudentsKnowingSecret 7) = 3280 :=
by
  sorry

theorem day_of_week (n : ℕ) : String :=
  if n % 7 = 0 then "Monday" else
  if n % 7 = 1 then "Tuesday" else
  if n % 7 = 2 then "Wednesday" else
  if n % 7 = 3 then "Thursday" else
  if n % 7 = 4 then "Friday" else
  if n % 7 = 5 then "Saturday" else
  "Sunday"

theorem day_when_3280_students_know_secret :
  day_of_week 7 = "Sunday" :=
by
  sorry

end students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l2097_209740


namespace average_length_tapes_l2097_209782

def lengths (l1 l2 l3 l4 l5 : ℝ) : Prop :=
  l1 = 35 ∧ l2 = 29 ∧ l3 = 35.5 ∧ l4 = 36 ∧ l5 = 30.5

theorem average_length_tapes
  (l1 l2 l3 l4 l5 : ℝ)
  (h : lengths l1 l2 l3 l4 l5) :
  (l1 + l2 + l3 + l4 + l5) / 5 = 33.2 := 
by
  sorry

end average_length_tapes_l2097_209782


namespace fraction_work_completed_by_third_group_l2097_209770

def working_speeds (name : String) : ℚ :=
  match name with
  | "A"  => 1
  | "B"  => 2
  | "C"  => 1.5
  | "D"  => 2.5
  | "E"  => 3
  | "F"  => 2
  | "W1" => 1
  | "W2" => 1.5
  | "W3" => 1
  | "W4" => 1
  | "W5" => 0.5
  | "W6" => 1
  | "W7" => 1.5
  | "W8" => 1
  | _    => 0

def work_done_per_hour (workers : List String) : ℚ :=
  workers.map working_speeds |>.sum

def first_group : List String := ["A", "B", "C", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
def second_group : List String := ["A", "B", "C", "D", "E", "F", "W1", "W2"]
def third_group : List String := ["A", "B", "C", "D", "E", "W1", "W2"]

theorem fraction_work_completed_by_third_group :
  (work_done_per_hour third_group) / (work_done_per_hour second_group) = 25 / 29 :=
by
  sorry

end fraction_work_completed_by_third_group_l2097_209770


namespace sum_remainders_mod_15_l2097_209777

theorem sum_remainders_mod_15 (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
    (a + b + c) % 15 = 6 :=
by
  sorry

end sum_remainders_mod_15_l2097_209777


namespace tan_phi_eq_sqrt3_l2097_209789

theorem tan_phi_eq_sqrt3
  (φ : ℝ)
  (h1 : Real.cos (Real.pi / 2 - φ) = Real.sqrt 3 / 2)
  (h2 : abs φ < Real.pi / 2) :
  Real.tan φ = Real.sqrt 3 :=
sorry

end tan_phi_eq_sqrt3_l2097_209789


namespace polar_distance_l2097_209730

theorem polar_distance {r1 θ1 r2 θ2 : ℝ} (A : r1 = 1 ∧ θ1 = π/6) (B : r2 = 3 ∧ θ2 = 5*π/6) : 
  (r1^2 + r2^2 - 2*r1*r2 * Real.cos (θ2 - θ1)) = 13 :=
  sorry

end polar_distance_l2097_209730


namespace part_a_contradiction_l2097_209754

theorem part_a_contradiction :
  ¬ (225 / 25 + 75 = 100 - 16 → 25 * (9 / (1 + 3)) = 84) :=
by
  sorry

end part_a_contradiction_l2097_209754


namespace man_age_twice_son_age_in_two_years_l2097_209763

theorem man_age_twice_son_age_in_two_years :
  ∀ (S M X : ℕ), S = 30 → M = S + 32 → (M + X = 2 * (S + X)) → X = 2 :=
by
  intros S M X hS hM h
  sorry

end man_age_twice_son_age_in_two_years_l2097_209763


namespace BD_length_l2097_209773

theorem BD_length
  (A B C D : Type)
  (dist_AC : ℝ := 10)
  (dist_BC : ℝ := 10)
  (dist_AD : ℝ := 12)
  (dist_CD : ℝ := 5) : (BD : ℝ) = 95 / 12 :=
by
  sorry

end BD_length_l2097_209773


namespace complement_U_M_correct_l2097_209705

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}
def complement_U_M : Set ℕ := {1, 2, 3}

theorem complement_U_M_correct : U \ M = complement_U_M :=
  by sorry

end complement_U_M_correct_l2097_209705


namespace ratio_ab_bd_l2097_209783

-- Definitions based on the given conditions
def ab : ℝ := 4
def bc : ℝ := 8
def cd : ℝ := 5
def bd : ℝ := bc + cd

-- Theorem statement
theorem ratio_ab_bd :
  ((ab / bd) = (4 / 13)) :=
by
  -- Proof goes here
  sorry

end ratio_ab_bd_l2097_209783


namespace sin_neg_p_l2097_209724

theorem sin_neg_p (a : ℝ) : (¬ ∃ x : ℝ, Real.sin x > a) → (a ≥ 1) := 
by
  sorry

end sin_neg_p_l2097_209724


namespace product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l2097_209756

theorem product_of_three_divisors_of_5_pow_4_eq_5_pow_4 (a b c : ℕ) (h1 : a * b * c = 5^4) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c) : a + b + c = 131 :=
sorry

end product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l2097_209756


namespace point_on_line_l2097_209781

theorem point_on_line : 
  ∃ t : ℚ, (3 * t + 1 = 0) ∧ ((2 - 4) / (t - 1) = (7 - 4) / (3 - 1)) :=
by
  sorry

end point_on_line_l2097_209781


namespace train_length_150_m_l2097_209721

def speed_in_m_s (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

def length_of_train (speed_in_m_s : ℕ) (time_s : ℕ) : ℕ :=
  speed_in_m_s * time_s

theorem train_length_150_m (speed_kmh : ℕ) (time_s : ℕ) (speed_m_s : speed_in_m_s speed_kmh = 15) (time_pass_pole : time_s = 10) : length_of_train (speed_in_m_s speed_kmh) time_s = 150 := by
  sorry

end train_length_150_m_l2097_209721


namespace digits_in_base_5_l2097_209702

theorem digits_in_base_5 (n : ℕ) (h : n = 1234) (h_largest_power : 5^4 < n ∧ n < 5^5) : 
  ∃ digits : ℕ, digits = 5 := 
sorry

end digits_in_base_5_l2097_209702


namespace frequency_rate_identity_l2097_209738

theorem frequency_rate_identity (n : ℕ) : 
  (36 : ℕ) / (n : ℕ) = (0.25 : ℝ) → 
  n = 144 := by
  sorry

end frequency_rate_identity_l2097_209738


namespace product_of_x_and_y_l2097_209767

theorem product_of_x_and_y (x y a b : ℝ)
  (h1 : x = b^(3/2))
  (h2 : y = a)
  (h3 : a + a = b^2)
  (h4 : y = b)
  (h5 : a + a = b^(3/2))
  (h6 : b = 3) :
  x * y = 9 * Real.sqrt 3 := 
  sorry

end product_of_x_and_y_l2097_209767


namespace percentage_increase_chef_vs_dishwasher_l2097_209780

variables 
  (manager_wage chef_wage dishwasher_wage : ℝ)
  (h_manager_wage : manager_wage = 8.50)
  (h_chef_wage : chef_wage = manager_wage - 3.315)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)

theorem percentage_increase_chef_vs_dishwasher :
  ((chef_wage - dishwasher_wage) / dishwasher_wage) * 100 = 22 :=
by
  sorry

end percentage_increase_chef_vs_dishwasher_l2097_209780


namespace region_area_l2097_209719

theorem region_area (x y : ℝ) : 
  (|2 * x - 16| + |3 * y + 9| ≤ 6) → ∃ A, A = 72 :=
sorry

end region_area_l2097_209719


namespace tail_length_l2097_209709

theorem tail_length {length body tail : ℝ} (h1 : length = 30) (h2 : tail = body / 2) (h3 : length = body) : tail = 15 := by
  sorry

end tail_length_l2097_209709


namespace simple_interest_principal_l2097_209732

theorem simple_interest_principal (A r t : ℝ) (ht_pos : t > 0) (hr_pos : r > 0) (hA_pos : A > 0) :
  (A = 1120) → (r = 0.08) → (t = 2.4) → ∃ (P : ℝ), abs (P - 939.60) < 0.01 :=
by
  intros hA hr ht
  -- Proof would go here
  sorry

end simple_interest_principal_l2097_209732


namespace sum_first_13_terms_l2097_209795

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (ha : a 2 + a 5 + a 9 + a 12 = 60)

theorem sum_first_13_terms :
  S 13 = 195 := sorry

end sum_first_13_terms_l2097_209795


namespace inequality_holds_l2097_209743

theorem inequality_holds (a : ℝ) : 3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
by
  sorry

end inequality_holds_l2097_209743


namespace minimum_grade_Ahmed_l2097_209733

theorem minimum_grade_Ahmed (assignments : ℕ) (Ahmed_grade : ℕ) (Emily_grade : ℕ) (final_assignment_grade_Emily : ℕ) 
  (sum_grades_Emily : ℕ) (sum_grades_Ahmed : ℕ) (total_points_Ahmed : ℕ) (total_points_Emily : ℕ) :
  assignments = 9 →
  Ahmed_grade = 91 →
  Emily_grade = 92 →
  final_assignment_grade_Emily = 90 →
  sum_grades_Emily = 828 →
  sum_grades_Ahmed = 819 →
  total_points_Ahmed = sum_grades_Ahmed + 100 →
  total_points_Emily = sum_grades_Emily + final_assignment_grade_Emily →
  total_points_Ahmed > total_points_Emily :=
by
  sorry

end minimum_grade_Ahmed_l2097_209733


namespace difference_of_squares_401_399_l2097_209778

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 :=
by
  sorry

end difference_of_squares_401_399_l2097_209778


namespace david_more_pushups_than_zachary_l2097_209798

theorem david_more_pushups_than_zachary :
  ∀ (zachary_pushups zachary_crunches david_crunches : ℕ),
    zachary_pushups = 34 →
    zachary_crunches = 62 →
    david_crunches = 45 →
    david_crunches + 17 = zachary_crunches →
    david_crunches + 17 - zachary_pushups = 17 :=
by
  intros zachary_pushups zachary_crunches david_crunches
  intros h1 h2 h3 h4
  sorry

end david_more_pushups_than_zachary_l2097_209798


namespace negation_of_prop_l2097_209701

variable (x : ℝ)
def prop (x : ℝ) := x ∈ Set.Ici 0 → Real.exp x ≥ 1

theorem negation_of_prop :
  (¬ ∀ x ∈ Set.Ici 0, Real.exp x ≥ 1) = ∃ x ∈ Set.Ici 0, Real.exp x < 1 :=
by
  sorry

end negation_of_prop_l2097_209701


namespace values_of_n_l2097_209725

/-
  Given a natural number n and a target sum 100,
  we need to find if there exists a combination of adding and subtracting 1 through n
  such that the sum equals 100.

- A value k is representable as a sum or difference of 1 through n if the sum of the series
  can be manipulated to produce k.
- The sum of the first n natural numbers S_n = n * (n + 1) / 2 must be even and sufficiently large.
- The specific values that satisfy the conditions are of the form n = 15 + 4 * k or n = 16 + 4 * k.
-/

def exists_sum_to_100 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k

theorem values_of_n (n : ℕ) : exists_sum_to_100 n ↔ (∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k) :=
by { sorry }

end values_of_n_l2097_209725


namespace kvass_affordability_l2097_209753

theorem kvass_affordability (x y : ℚ) (hx : x + y = 1) (hxy : 1.2 * (0.5 * x + y) = 1) : 1.44 * y ≤ 1 :=
by
  -- Placeholder for proof
  sorry

end kvass_affordability_l2097_209753


namespace mul_97_103_l2097_209792

theorem mul_97_103 : (97:ℤ) = 100 - 3 → (103:ℤ) = 100 + 3 → 97 * 103 = 9991 := by
  intros h1 h2
  sorry

end mul_97_103_l2097_209792


namespace john_hourly_wage_with_bonus_l2097_209749

structure JohnJob where
  daily_wage : ℕ
  work_hours : ℕ
  bonus_amount : ℕ
  extra_hours : ℕ

def total_daily_wage (job : JohnJob) : ℕ :=
  job.daily_wage + job.bonus_amount

def total_work_hours (job : JohnJob) : ℕ :=
  job.work_hours + job.extra_hours

def hourly_wage (job : JohnJob) : ℕ :=
  total_daily_wage job / total_work_hours job

noncomputable def johns_job : JohnJob :=
  { daily_wage := 80, work_hours := 8, bonus_amount := 20, extra_hours := 2 }

theorem john_hourly_wage_with_bonus :
  hourly_wage johns_job = 10 :=
by
  sorry

end john_hourly_wage_with_bonus_l2097_209749


namespace polygon_area_144_l2097_209717

-- Given definitions
def polygon (n : ℕ) : Prop := -- definition to capture n squares arrangement
  n = 36

def is_perpendicular (sides : ℕ) : Prop := -- every pair of adjacent sides is perpendicular
  sides = 4

def all_sides_congruent (length : ℕ) : Prop := -- all sides have the same length
  true

def total_perimeter (perimeter : ℕ) : Prop := -- total perimeter of the polygon
  perimeter = 72

-- The side length s leading to polygon's perimeter
def side_length (s perimeter : ℕ) : Prop :=
  perimeter = 36 * s / 2 

-- Prove the area of polygon is 144
theorem polygon_area_144 (n sides length perimeter s: ℕ) 
    (h1 : polygon n) 
    (h2 : is_perpendicular sides) 
    (h3 : all_sides_congruent length) 
    (h4 : total_perimeter perimeter) 
    (h5 : side_length s perimeter) : 
    n * s * s = 144 := 
sorry

end polygon_area_144_l2097_209717


namespace kyle_age_l2097_209761

theorem kyle_age :
  ∃ (kyle shelley julian frederick tyson casey : ℕ),
    shelley = kyle - 3 ∧ 
    shelley = julian + 4 ∧
    julian = frederick - 20 ∧
    frederick = 2 * tyson ∧
    tyson = 2 * casey ∧
    casey = 15 ∧ 
    kyle = 47 :=
by
  sorry

end kyle_age_l2097_209761


namespace smallest_n_divisible_11_remainder_1_l2097_209757

theorem smallest_n_divisible_11_remainder_1 :
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 1) ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 7 = 1) ∧ (n % 11 = 0) ∧ 
    (∀ m : ℕ, (m % 2 = 1) ∧ (m % 3 = 1) ∧ (m % 4 = 1) ∧ (m % 5 = 1) ∧ (m % 7 = 1) ∧ (m % 11 = 0) → 2521 ≤ m) :=
by
  sorry

end smallest_n_divisible_11_remainder_1_l2097_209757


namespace max_elements_in_set_l2097_209728

theorem max_elements_in_set (S : Finset ℕ) (hS : ∀ (a b : ℕ), a ≠ b → a ∈ S → b ∈ S → 
  ∃ (k : ℕ) (c d : ℕ), c < d ∧ c ∈ S ∧ d ∈ S ∧ a + b = c^k * d) :
  S.card ≤ 48 :=
sorry

end max_elements_in_set_l2097_209728


namespace Vanya_correct_answers_l2097_209704

theorem Vanya_correct_answers (x : ℕ) (h : 7 * x = 3 * (50 - x)) : x = 15 := by
  sorry

end Vanya_correct_answers_l2097_209704


namespace nearest_whole_number_l2097_209786

theorem nearest_whole_number (x : ℝ) (h : x = 7263.4987234) : Int.floor (x + 0.5) = 7263 := by
  sorry

end nearest_whole_number_l2097_209786


namespace avg_tickets_sold_by_males_100_l2097_209739

theorem avg_tickets_sold_by_males_100 
  (female_avg : ℕ := 70) 
  (nonbinary_avg : ℕ := 50) 
  (overall_avg : ℕ := 66) 
  (male_ratio : ℕ := 2) 
  (female_ratio : ℕ := 3) 
  (nonbinary_ratio : ℕ := 5) : 
  ∃ (male_avg : ℕ), male_avg = 100 := 
by 
  sorry

end avg_tickets_sold_by_males_100_l2097_209739


namespace minimum_perimeter_l2097_209745

def fractional_part (x : ℚ) : ℚ := x - x.floor

-- Define l, m, n being sides of the triangle with l > m > n
variables (l m n : ℤ)

-- Defining conditions as Lean predicates
def triangle_sides (l m n : ℤ) : Prop := l > m ∧ m > n

def fractional_part_condition (l m n : ℤ) : Prop :=
  fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧
  fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)

-- Prove the minimum perimeter is 3003 given above conditions
theorem minimum_perimeter (l m n : ℤ) :
  triangle_sides l m n →
  fractional_part_condition l m n →
  l + m + n = 3003 :=
by
  intros h_sides h_fractional
  sorry

end minimum_perimeter_l2097_209745


namespace is_linear_equation_with_one_var_l2097_209714

-- Definitions
def eqA := ∀ (x : ℝ), x^2 + 1 = 5
def eqB := ∀ (x y : ℝ), x + 2 = y - 3
def eqC := ∀ (x : ℝ), 1 / (2 * x) = 10
def eqD := ∀ (x : ℝ), x = 4

-- Theorem stating which equation represents a linear equation in one variable
theorem is_linear_equation_with_one_var : eqD :=
by
  -- Proof skipped
  sorry

end is_linear_equation_with_one_var_l2097_209714


namespace wire_length_after_cuts_l2097_209742

-- Given conditions as parameters
def initial_length_cm : ℝ := 23.3
def first_cut_mm : ℝ := 105
def second_cut_cm : ℝ := 4.6

-- Final statement to be proved
theorem wire_length_after_cuts (ell : ℝ) (c1 : ℝ) (c2 : ℝ) : (ell = 23.3) → (c1 = 105) → (c2 = 4.6) → 
  (ell * 10 - c1 - c2 * 10 = 82) := sorry

end wire_length_after_cuts_l2097_209742


namespace updated_mean_l2097_209722

theorem updated_mean
  (n : ℕ) (obs_mean : ℝ) (decrement : ℝ)
  (h1 : n = 50) (h2 : obs_mean = 200) (h3 : decrement = 47) :
  (obs_mean - decrement) = 153 := by
  sorry

end updated_mean_l2097_209722


namespace eval_expression_l2097_209736

theorem eval_expression : 4 * (8 - 3) - 6 = 14 :=
by
  sorry

end eval_expression_l2097_209736


namespace probability_at_least_one_l2097_209746

theorem probability_at_least_one (
    pA pB pC : ℝ
) (hA : pA = 0.9) (hB : pB = 0.8) (hC : pC = 0.7) (independent : true) : 
    (1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.994 := 
by
  rw [hA, hB, hC]
  sorry

end probability_at_least_one_l2097_209746


namespace sine_of_pi_minus_alpha_l2097_209710

theorem sine_of_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 3) : Real.sin (π - α) = 1 / 3 :=
by
  sorry

end sine_of_pi_minus_alpha_l2097_209710


namespace maximize_container_volume_l2097_209771

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ ∀ y : ℝ, 0 < y ∧ y < 24 → 
  ( (48 - 2 * x)^2 * x ≥ (48 - 2 * y)^2 * y ) ∧ x = 8 :=
sorry

end maximize_container_volume_l2097_209771


namespace batsman_average_increases_l2097_209785

theorem batsman_average_increases
  (score_17th: ℕ)
  (avg_increase: ℕ)
  (initial_avg: ℕ)
  (final_avg: ℕ)
  (initial_innings: ℕ):
  score_17th = 74 →
  avg_increase = 3 →
  initial_innings = 16 →
  initial_avg = 23 →
  final_avg = initial_avg + avg_increase →
  (final_avg * (initial_innings + 1) = score_17th + (initial_avg * initial_innings)) →
  final_avg = 26 :=
by
  sorry

end batsman_average_increases_l2097_209785


namespace find_value_of_B_l2097_209775

theorem find_value_of_B (B : ℚ) (h : 4 * B + 4 = 33) : B = 29 / 4 :=
by
  sorry

end find_value_of_B_l2097_209775


namespace average_weight_of_a_b_c_l2097_209752

theorem average_weight_of_a_b_c (A B C : ℕ) 
  (h1 : (A + B) / 2 = 25) 
  (h2 : (B + C) / 2 = 28) 
  (hB : B = 16) : 
  (A + B + C) / 3 = 30 := 
by 
  sorry

end average_weight_of_a_b_c_l2097_209752


namespace second_investment_amount_l2097_209703

/-
A $500 investment and another investment have a combined yearly return of 8.5 percent of the total of the two investments.
The $500 investment has a yearly return of 7 percent.
The other investment has a yearly return of 9 percent.
Prove that the amount of the second investment is $1500.
-/

theorem second_investment_amount :
  ∃ x : ℝ, 35 + 0.09 * x = 0.085 * (500 + x) → x = 1500 :=
by
  sorry

end second_investment_amount_l2097_209703


namespace validCardSelections_l2097_209797

def numberOfValidSelections : ℕ :=
  let totalCards := 12
  let redCards := 4
  let otherColors := 8 -- 4 yellow + 4 blue
  let totalSelections := Nat.choose totalCards 3
  let nonRedSelections := Nat.choose otherColors 3
  let oneRedSelections := Nat.choose redCards 1 * Nat.choose otherColors 2
  let sameColorSelections := 3 * Nat.choose 4 3 -- 3 colors, 4 cards each, selecting 3
  (nonRedSelections + oneRedSelections)

theorem validCardSelections : numberOfValidSelections = 160 := by
  sorry

end validCardSelections_l2097_209797


namespace find_pairs_l2097_209734

theorem find_pairs (n k : ℕ) (h1 : (10^(k-1) ≤ n^n) ∧ (n^n < 10^k)) (h2 : (10^(n-1) ≤ k^k) ∧ (k^k < 10^n)) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) := by
  sorry

end find_pairs_l2097_209734


namespace mia_study_time_l2097_209700

theorem mia_study_time 
  (T : ℕ)
  (watching_tv_exercise_social_media : T = 1440 ∧ 
    ∃ study_time : ℚ, 
    (study_time = (1 / 4) * 
      (((27 / 40) * T - (9 / 80) * T) / 
        (T * 1 / 40 - (1 / 5) * T - (1 / 8) * T))
    )) :
  T = 1440 → study_time = 202.5 := 
by
  sorry

end mia_study_time_l2097_209700


namespace cory_needs_22_weeks_l2097_209706

open Nat

def cory_birthday_money : ℕ := 100 + 45 + 20
def bike_cost : ℕ := 600
def weekly_earning : ℕ := 20

theorem cory_needs_22_weeks : ∃ x : ℕ, cory_birthday_money + x * weekly_earning ≥ bike_cost ∧ x = 22 := by
  sorry

end cory_needs_22_weeks_l2097_209706


namespace Powerjet_pumps_250_gallons_in_30_minutes_l2097_209744

theorem Powerjet_pumps_250_gallons_in_30_minutes :
  let r := 500 -- Pump rate in gallons per hour
  let t := 1 / 2 -- Time in hours (30 minutes)
  r * t = 250 := by
  -- proof steps will go here
  sorry

end Powerjet_pumps_250_gallons_in_30_minutes_l2097_209744


namespace slope_of_line_l2097_209772

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line l intersecting the parabola C at points A and B
def line (k x : ℝ) : ℝ := k * (x - 1)

-- Condition based on the intersection and the given relationship 2 * (BF) = FA
def intersection_condition (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ x1 x2 y1 y2,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    (y1 = line k x1) ∧ (y2 = line k x2) ∧
    2 * (dist (x2, y2) focus) = dist focus (x1, y1)

-- The main theorem to be proven
theorem slope_of_line (k : ℝ) (A B : ℝ × ℝ) :
  intersection_condition k A B → k = 2 * Real.sqrt 2 :=
sorry

end slope_of_line_l2097_209772


namespace quadratic_real_root_exists_l2097_209751

theorem quadratic_real_root_exists :
  ¬ (∃ x : ℝ, x^2 + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 + x + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 - x + 1 = 0) ∧
  (∃ x : ℝ, x^2 - x - 1 = 0) :=
by
  sorry

end quadratic_real_root_exists_l2097_209751


namespace smallest_n_conditions_l2097_209764

theorem smallest_n_conditions (n : ℕ) : 
  (∃ k m : ℕ, 4 * n = k^2 ∧ 5 * n = m^5 ∧ ∀ n' : ℕ, (∃ k' m' : ℕ, 4 * n' = k'^2 ∧ 5 * n' = m'^5) → n ≤ n') → 
  n = 625 :=
by
  intro h
  sorry

end smallest_n_conditions_l2097_209764


namespace ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l2097_209708

-- Problem 1
theorem ab_eq_6_pos_or_neg (a b : ℚ) (h : a * b = 6) : a + b > 0 ∨ a + b < 0 := sorry

-- Problem 2
theorem max_ab_when_sum_neg5 (a b : ℤ) (h : a + b = -5) : a * b ≤ 6 := sorry

-- Problem 3
theorem ab_lt_0_sign_of_sum (a b : ℚ) (h : a * b < 0) : (a + b > 0 ∨ a + b = 0 ∨ a + b < 0) := sorry

end ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l2097_209708


namespace smallest_m_n_sum_l2097_209799

noncomputable def smallestPossibleSum (m n : ℕ) : ℕ :=
  m + n

theorem smallest_m_n_sum :
  ∃ (m n : ℕ), (m > 1) ∧ (m * n * (2021 * (m^2 - 1)) = 2021 * m * m * n) ∧ smallestPossibleSum m n = 4323 :=
by
  sorry

end smallest_m_n_sum_l2097_209799


namespace find_special_four_digit_number_l2097_209715

theorem find_special_four_digit_number :
  ∃ (N : ℕ), 
  (N % 131 = 112) ∧ 
  (N % 132 = 98) ∧ 
  (1000 ≤ N) ∧ 
  (N < 10000) ∧ 
  (N = 1946) :=
sorry

end find_special_four_digit_number_l2097_209715


namespace exactly_one_negative_x_or_y_l2097_209794

theorem exactly_one_negative_x_or_y
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (x1_ne_zero : x1 ≠ 0) (x2_ne_zero : x2 ≠ 0) (x3_ne_zero : x3 ≠ 0)
  (y1_ne_zero : y1 ≠ 0) (y2_ne_zero : y2 ≠ 0) (y3_ne_zero : y3 ≠ 0)
  (h1 : x1 * x2 * x3 = - y1 * y2 * y3)
  (h2 : x1^2 + x2^2 + x3^2 = y1^2 + y2^2 + y3^2)
  (h3 : x1 + y1 + x2 + y2 ≥ x3 + y3 ∧ x2 + y2 + x3 + y3 ≥ x1 + y1 ∧ x3 + y3 + x1 + y1 ≥ x2 + y2)
  (h4 : (x1 + y1)^2 + (x2 + y2)^2 ≥ (x3 + y3)^2 ∧ (x2 + y2)^2 + (x3 + y3)^2 ≥ (x1 + y1)^2 ∧ (x3 + y3)^2 + (x1 + y1)^2 ≥ (x2 + y2)^2) :
  ∃! (a : ℝ), (a = x1 ∨ a = x2 ∨ a = x3 ∨ a = y1 ∨ a = y2 ∨ a = y3) ∧ a < 0 :=
sorry

end exactly_one_negative_x_or_y_l2097_209794


namespace question1_question2_l2097_209755

theorem question1 :
  (1:ℝ) * (Real.sqrt 12 + Real.sqrt 20) + (Real.sqrt 3 - Real.sqrt 5) = 3 * Real.sqrt 3 + Real.sqrt 5 := 
by sorry

theorem question2 :
  (4 * Real.sqrt 2 - 3 * Real.sqrt 6) / (2 * Real.sqrt 2) - (Real.sqrt 8 + Real.pi)^0 = 1 - 3 * Real.sqrt 3 / 2 :=
by sorry

end question1_question2_l2097_209755


namespace plan_b_more_cost_effective_l2097_209788

noncomputable def fare (x : ℝ) : ℝ :=
if x < 3 then 5
else if x <= 10 then 1.2 * x + 1.4
else 1.8 * x - 4.6

theorem plan_b_more_cost_effective :
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  plan_a > plan_b :=
by
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  sorry

end plan_b_more_cost_effective_l2097_209788


namespace cubic_identity_l2097_209716

theorem cubic_identity (x : ℝ) (h : x + 1/x = -6) : x^3 + 1/x^3 = -198 := 
by
  sorry

end cubic_identity_l2097_209716


namespace total_oranges_in_box_l2097_209762

def initial_oranges_in_box : ℝ := 55.0
def oranges_added_by_susan : ℝ := 35.0

theorem total_oranges_in_box :
  initial_oranges_in_box + oranges_added_by_susan = 90.0 := by
  sorry

end total_oranges_in_box_l2097_209762


namespace minimum_value_expression_l2097_209747

theorem minimum_value_expression 
  (a b c d : ℝ)
  (h1 : (2 * a^2 - Real.log a) / b = 1)
  (h2 : (3 * c - 2) / d = 1) :
  ∃ min_val : ℝ, min_val = (a - c)^2 + (b - d)^2 ∧ min_val = 1 / 10 :=
by {
  sorry
}

end minimum_value_expression_l2097_209747


namespace range_of_a_l2097_209774

theorem range_of_a (h : ¬ ∃ x : ℝ, x < 2023 ∧ x > a) : a ≥ 2023 := 
sorry

end range_of_a_l2097_209774


namespace price_reduction_correct_l2097_209760

theorem price_reduction_correct (P : ℝ) : 
  let first_reduction := 0.92 * P
  let second_reduction := first_reduction * 0.90
  second_reduction = 0.828 * P := 
by 
  sorry

end price_reduction_correct_l2097_209760


namespace intersection_complement_l2097_209776

def M : Set ℝ := { x | x^2 - x - 6 ≥ 0 }
def N : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def neg_R (A : Set ℝ) : Set ℝ := { x | x ∉ A }

theorem intersection_complement (N : Set ℝ) (M : Set ℝ) :
  N ∩ (neg_R M) = { x | -2 < x ∧ x ≤ 1 } := 
by {
  -- Proof goes here
  sorry
}

end intersection_complement_l2097_209776


namespace good_goods_not_cheap_is_sufficient_condition_l2097_209727

theorem good_goods_not_cheap_is_sufficient_condition
  (goods_good : Prop)
  (goods_not_cheap : Prop)
  (h : goods_good → goods_not_cheap) :
  (goods_good → goods_not_cheap) :=
by
  exact h

end good_goods_not_cheap_is_sufficient_condition_l2097_209727


namespace find_a_l2097_209765

theorem find_a (a n : ℕ) (h1 : (2 : ℕ) ^ n = 32) (h2 : (a + 1) ^ n = 243) : a = 2 := by
  sorry

end find_a_l2097_209765


namespace oak_grove_libraries_total_books_l2097_209758

theorem oak_grove_libraries_total_books :
  let publicLibraryBooks := 1986
  let schoolLibrariesBooks := 5106
  let communityCollegeLibraryBooks := 3294.5
  let medicalLibraryBooks := 1342.25
  let lawLibraryBooks := 2785.75
  publicLibraryBooks + schoolLibrariesBooks + communityCollegeLibraryBooks + medicalLibraryBooks + lawLibraryBooks = 15514.5 :=
by
  sorry

end oak_grove_libraries_total_books_l2097_209758


namespace prime_sum_20_to_30_l2097_209720

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_sum : ℕ := 23 + 29

theorem prime_sum_20_to_30 :
  (∀ p, 20 < p ∧ p < 30 → is_prime p → p = 23 ∨ p = 29) →
  prime_sum = 52 :=
by
  intros
  unfold prime_sum
  rfl

end prime_sum_20_to_30_l2097_209720


namespace find_quotient_l2097_209784

-- Define the problem variables and conditions
def larger_number : ℕ := 1620
def smaller_number : ℕ := larger_number - 1365
def remainder : ℕ := 15

-- Define the proof problem
theorem find_quotient :
  larger_number = smaller_number * 6 + remainder :=
sorry

end find_quotient_l2097_209784


namespace sum_largest_smallest_ABC_l2097_209711

def hundreds (n : ℕ) : ℕ := n / 100
def units (n : ℕ) : ℕ := n % 10
def tens (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_largest_smallest_ABC : 
  (hundreds 297 = 2) ∧ (units 297 = 7) ∧ (hundreds 207 = 2) ∧ (units 207 = 7) →
  (297 + 207 = 504) :=
by
  sorry

end sum_largest_smallest_ABC_l2097_209711


namespace last_digit_of_one_over_729_l2097_209729

def last_digit_of_decimal_expansion (n : ℕ) : ℕ := (n % 10)

theorem last_digit_of_one_over_729 : last_digit_of_decimal_expansion (1 / 729) = 9 :=
sorry

end last_digit_of_one_over_729_l2097_209729


namespace correct_calculation_l2097_209759

theorem correct_calculation :
    (1 + Real.sqrt 2)^2 = 3 + 2 * Real.sqrt 2 :=
sorry

end correct_calculation_l2097_209759


namespace block_wall_min_blocks_l2097_209712

theorem block_wall_min_blocks :
  ∃ n,
    n = 648 ∧
    ∀ (row_height wall_height block1_length block2_length wall_length: ℕ),
    row_height = 1 ∧
    wall_height = 8 ∧
    block1_length = 1 ∧
    block2_length = 3/2 ∧
    wall_length = 120 ∧
    (∀ i : ℕ, i < wall_height → ∃ k m : ℕ, k * block1_length + m * block2_length = wall_length) →
    n = (wall_height * (1 + 2 * 79))
:= by sorry

end block_wall_min_blocks_l2097_209712


namespace unique_solution_of_function_eq_l2097_209748

theorem unique_solution_of_function_eq (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) : f = id := 
sorry

end unique_solution_of_function_eq_l2097_209748


namespace area_per_cabbage_is_one_l2097_209718

noncomputable def area_per_cabbage (x y : ℕ) : ℕ :=
  let num_cabbages_this_year : ℕ := 10000
  let increase_in_cabbages : ℕ := 199
  let area_this_year : ℕ := y^2
  let area_last_year : ℕ := x^2
  let area_per_cabbage : ℕ := area_this_year / num_cabbages_this_year
  area_per_cabbage

theorem area_per_cabbage_is_one (x y : ℕ) (hx : y^2 = 10000) (hy : y^2 = x^2 + 199) : area_per_cabbage x y = 1 :=
by 
  sorry

end area_per_cabbage_is_one_l2097_209718


namespace inclination_angle_of_line_l2097_209766

def line_equation (x y : ℝ) : Prop := x * (Real.tan (Real.pi / 3)) + y + 2 = 0

theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : 
  ∃ α : ℝ, α = 2 * Real.pi / 3 ∧ 0 ≤ α ∧ α < Real.pi := by
  sorry

end inclination_angle_of_line_l2097_209766


namespace find_x_l2097_209750

theorem find_x (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z)
(h₄ : x^2 / y = 3) (h₅ : y^2 / z = 4) (h₆ : z^2 / x = 5) : 
  x = (6480 : ℝ)^(1/7 : ℝ) :=
by 
  sorry

end find_x_l2097_209750


namespace john_uses_six_pounds_of_vegetables_l2097_209790

-- Define the given conditions:
def pounds_of_beef_bought : ℕ := 4
def pounds_beef_used_in_soup := pounds_of_beef_bought - 1
def pounds_of_vegetables_used := 2 * pounds_beef_used_in_soup

-- Statement to prove:
theorem john_uses_six_pounds_of_vegetables : pounds_of_vegetables_used = 6 :=
by
  sorry

end john_uses_six_pounds_of_vegetables_l2097_209790


namespace pyarelal_loss_l2097_209768

theorem pyarelal_loss (total_loss : ℝ) (P : ℝ) (Ashok_capital : ℝ) (ratio_Ashok_Pyarelal : ℝ) :
  total_loss = 670 →
  Ashok_capital = P / 9 →
  ratio_Ashok_Pyarelal = 1 / 9 →
  Pyarelal_loss = 603 :=
by
  intro total_loss_eq Ashok_capital_eq ratio_eq
  sorry

end pyarelal_loss_l2097_209768


namespace initial_pieces_l2097_209735

-- Define the conditions
def pieces_used : ℕ := 156
def pieces_left : ℕ := 744

-- Define the total number of pieces of paper Isabel bought initially
def total_pieces : ℕ := pieces_used + pieces_left

-- State the theorem that we need to prove
theorem initial_pieces (h1 : pieces_used = 156) (h2 : pieces_left = 744) : total_pieces = 900 :=
by
  sorry

end initial_pieces_l2097_209735


namespace put_balls_in_boxes_l2097_209796

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l2097_209796


namespace volume_of_parallelepiped_l2097_209791

theorem volume_of_parallelepiped 
  (m n Q : ℝ) 
  (ratio_positive : 0 < m ∧ 0 < n)
  (Q_positive : 0 < Q)
  (h_square_area : ∃ a b : ℝ, a / b = m / n ∧ (a^2 + b^2) = Q) :
  ∃ (V : ℝ), V = (m * n * Q * Real.sqrt Q) / (m^2 + n^2) :=
sorry

end volume_of_parallelepiped_l2097_209791


namespace randi_has_6_more_nickels_than_peter_l2097_209731

def ray_initial_cents : Nat := 175
def cents_given_peter : Nat := 30
def cents_given_randi : Nat := 2 * cents_given_peter
def nickel_worth : Nat := 5

def nickels (cents : Nat) : Nat :=
  cents / nickel_worth

def randi_more_nickels_than_peter : Prop :=
  nickels cents_given_randi - nickels cents_given_peter = 6

theorem randi_has_6_more_nickels_than_peter :
  randi_more_nickels_than_peter :=
sorry

end randi_has_6_more_nickels_than_peter_l2097_209731


namespace rth_term_of_arithmetic_progression_l2097_209713

noncomputable def Sn (n : ℕ) : ℕ := 2 * n + 3 * n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) : 
  (Sn r - Sn (r - 1)) = 3 * r^2 + 5 * r - 2 :=
by sorry

end rth_term_of_arithmetic_progression_l2097_209713
