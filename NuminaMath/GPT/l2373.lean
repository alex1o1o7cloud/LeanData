import Mathlib

namespace gina_college_expenses_l2373_237346

theorem gina_college_expenses
  (credits : ℕ)
  (cost_per_credit : ℕ)
  (num_textbooks : ℕ)
  (cost_per_textbook : ℕ)
  (facilities_fee : ℕ)
  (H_credits : credits = 14)
  (H_cost_per_credit : cost_per_credit = 450)
  (H_num_textbooks : num_textbooks = 5)
  (H_cost_per_textbook : cost_per_textbook = 120)
  (H_facilities_fee : facilities_fee = 200)
  : (credits * cost_per_credit) + (num_textbooks * cost_per_textbook) + facilities_fee = 7100 := by
  sorry

end gina_college_expenses_l2373_237346


namespace arrange_chairs_and_stools_l2373_237348

-- Definition of the mathematical entities based on the conditions
def num_ways_to_arrange (women men : ℕ) : ℕ :=
  let total := women + men
  (total.factorial) / (women.factorial * men.factorial)

-- Prove that the arrangement yields the correct number of ways
theorem arrange_chairs_and_stools :
  num_ways_to_arrange 7 3 = 120 := by
  -- The specific definitions and steps are not to be included in the Lean statement;
  -- hence, adding a placeholder for the proof.
  sorry

end arrange_chairs_and_stools_l2373_237348


namespace smallest_a1_l2373_237395

noncomputable def a_seq (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 13 * a (n - 1) - 2 * n

noncomputable def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ i, a i > 0

theorem smallest_a1 : ∃ a : ℕ → ℝ, a_seq a ∧ positive_sequence a ∧ a 1 = 13 / 36 :=
by
  sorry

end smallest_a1_l2373_237395


namespace surface_is_plane_l2373_237393

-- Define cylindrical coordinates
structure CylindricalCoordinate where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the property for a constant θ
def isConstantTheta (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  coord.θ = c

-- Define the plane in cylindrical coordinates
def isPlane (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  isConstantTheta c coord

-- Theorem: The surface described by θ = c in cylindrical coordinates is a plane.
theorem surface_is_plane (c : ℝ) (coord : CylindricalCoordinate) :
    isPlane c coord ↔ isConstantTheta c coord := sorry

end surface_is_plane_l2373_237393


namespace mono_increasing_necessary_not_sufficient_problem_statement_l2373_237304

-- Define the function
def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- Define the first condition of p: f(x) is monotonically increasing in (-∞, +∞)
def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Define the second condition q: m > 4/3
def m_gt_4_over_3 (m : ℝ) : Prop := m > 4/3

-- State the theorem: 
theorem mono_increasing_necessary_not_sufficient (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) := 
by
  sorry

-- Main theorem tying the conditions to the conclusion
theorem problem_statement (m : ℝ):
  is_monotonically_increasing (f x) → m_gt_4_over_3 m → 
  (is_monotonically_increasing (f x) ↔ m ≥ 4/3) ∧ (¬ is_monotonically_increasing (f x) → m > 4/3) :=
  by sorry

end mono_increasing_necessary_not_sufficient_problem_statement_l2373_237304


namespace total_coins_l2373_237321

def piles_of_quarters : Nat := 5
def piles_of_dimes : Nat := 5
def coins_per_pile : Nat := 3

theorem total_coins :
  (piles_of_quarters * coins_per_pile) + (piles_of_dimes * coins_per_pile) = 30 := by
  sorry

end total_coins_l2373_237321


namespace smallest_four_digit_divisible_by_3_5_7_11_l2373_237399

theorem smallest_four_digit_divisible_by_3_5_7_11 : 
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 
          n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 1155 :=
by
  sorry

end smallest_four_digit_divisible_by_3_5_7_11_l2373_237399


namespace parabola_relationship_l2373_237363

theorem parabola_relationship 
  (c : ℝ) (y1 y2 y3 : ℝ) 
  (h1 : y1 = 2*(-2 - 1)^2 + c) 
  (h2 : y2 = 2*(0 - 1)^2 + c) 
  (h3 : y3 = 2*((5:ℝ)/3 - 1)^2 + c):
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_relationship_l2373_237363


namespace sum_of_sequence_l2373_237376

variable (S a b : ℝ)

theorem sum_of_sequence :
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 :=
by
  intros h1 h2 h3
  sorry

end sum_of_sequence_l2373_237376


namespace infinitely_many_solutions_l2373_237389

theorem infinitely_many_solutions (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := sorry

end infinitely_many_solutions_l2373_237389


namespace unknown_sum_of_digits_l2373_237364

theorem unknown_sum_of_digits 
  (A B C D : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h2 : D = 1)
  (h3 : (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D) : 
  A + B = 0 := 
sorry

end unknown_sum_of_digits_l2373_237364


namespace bell_ratio_l2373_237320

theorem bell_ratio :
  ∃ (B3 B2 : ℕ), 
  B2 = 2 * 50 ∧ 
  50 + B2 + B3 = 550 ∧ 
  (B3 / B2 = 4) := 
sorry

end bell_ratio_l2373_237320


namespace total_carrot_sticks_l2373_237333

-- Define the number of carrot sticks James ate before and after dinner
def carrot_sticks_before_dinner : Nat := 22
def carrot_sticks_after_dinner : Nat := 15

-- Prove that the total number of carrot sticks James ate is 37
theorem total_carrot_sticks : carrot_sticks_before_dinner + carrot_sticks_after_dinner = 37 :=
  by sorry

end total_carrot_sticks_l2373_237333


namespace find_m_l2373_237313

theorem find_m (a b m : ℝ) :
  (∀ x : ℝ, (x^2 - b * x + b^2) / (a * x^2 - b^2) = (m - 1) / (m + 1) → (∀ y : ℝ, x = y ∧ x = -y)) →
  c = b^2 →
  m = (a - 1) / (a + 1) :=
by
  sorry

end find_m_l2373_237313


namespace area_of_right_triangle_l2373_237388

theorem area_of_right_triangle
  (BC AC : ℝ)
  (h1 : BC * AC = 16) : 
  0.5 * BC * AC = 8 := by 
  sorry

end area_of_right_triangle_l2373_237388


namespace longer_subsegment_of_YZ_l2373_237331

/-- In triangle XYZ with sides in the ratio 3:4:5, and side YZ being 12 cm.
    The angle bisector XW divides side YZ into segments YW and ZW.
    Prove that the length of ZW is 48/7 cm. --/
theorem longer_subsegment_of_YZ (YZ : ℝ) (hYZ : YZ = 12)
    (XY XZ : ℝ) (hRatio : XY / XZ = 3 / 4) : 
    ∃ ZW : ℝ, ZW = 48 / 7 :=
by
  -- We would provide proof here
  sorry

end longer_subsegment_of_YZ_l2373_237331


namespace weight_of_pecans_l2373_237384

theorem weight_of_pecans (total_weight_of_nuts almonds_weight pecans_weight : ℝ)
  (h1 : total_weight_of_nuts = 0.52)
  (h2 : almonds_weight = 0.14)
  (h3 : pecans_weight = total_weight_of_nuts - almonds_weight) :
  pecans_weight = 0.38 :=
  by
    sorry

end weight_of_pecans_l2373_237384


namespace volleyball_club_girls_l2373_237308

theorem volleyball_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 3 : ℝ) * G + ↑B = 20) : G = 18 := 
by
  sorry

end volleyball_club_girls_l2373_237308


namespace division_of_negatives_l2373_237375

theorem division_of_negatives (x y : Int) (h1 : y ≠ 0) (h2 : -x = 150) (h3 : -y = 25) : (-150) / (-25) = 6 :=
by
  sorry

end division_of_negatives_l2373_237375


namespace trigonometric_sum_l2373_237329

theorem trigonometric_sum (θ : ℝ) (h_tan_θ : Real.tan θ = 5 / 12) (h_range : π ≤ θ ∧ θ ≤ 3 * π / 2) : 
  Real.cos θ + Real.sin θ = -17 / 13 :=
by
  sorry

end trigonometric_sum_l2373_237329


namespace multiplication_result_l2373_237350

theorem multiplication_result :
  121 * 54 = 6534 := by
  sorry

end multiplication_result_l2373_237350


namespace cars_with_both_features_l2373_237341

theorem cars_with_both_features (T P_s P_w N B : ℕ)
  (hT : T = 65) 
  (hPs : P_s = 45) 
  (hPw : P_w = 25) 
  (hN : N = 12) 
  (h_equation : P_s + P_w - B + N = T) :
  B = 17 :=
by
  sorry

end cars_with_both_features_l2373_237341


namespace cylinder_cone_volume_ratio_l2373_237338

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ)
  (hcylinder_csa : π * r_cylinder^2 = π * r_cone^2 / 4):
  (π * r_cylinder^2 * h) / (1 / 3 * π * r_cone^2 * h) = 3 / 4 :=
by
  sorry

end cylinder_cone_volume_ratio_l2373_237338


namespace range_of_a_l2373_237342

-- Given definitions from the problem
def p (a : ℝ) : Prop :=
  (4 - 4 * a) > 0

def q (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

-- The theorem we want to prove
theorem range_of_a (a : ℝ) : ¬ (p a ∨ q a) ↔ a ≥ 3 := 
by sorry

end range_of_a_l2373_237342


namespace range_of_a_l2373_237385

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x^2 + 2 * |x - a| ≥ a^2) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l2373_237385


namespace quadratic_rewrite_sum_l2373_237373

theorem quadratic_rewrite_sum :
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  a + b + c = 143.25 :=
by 
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  sorry

end quadratic_rewrite_sum_l2373_237373


namespace mina_crafts_total_l2373_237324

theorem mina_crafts_total :
  let a₁ := 3
  let d := 4
  let n := 10
  let crafts_sold_on_day (d: ℕ) := a₁ + (d - 1) * d
  let S (n: ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
  S n = 210 :=
by
  sorry

end mina_crafts_total_l2373_237324


namespace heather_blocks_l2373_237326

theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 86 → shared_blocks = 41 → remaining_blocks = initial_blocks - shared_blocks → remaining_blocks = 45 :=
by
  sorry

end heather_blocks_l2373_237326


namespace initial_girls_l2373_237357

theorem initial_girls (G : ℕ) (h : G + 682 = 1414) : G = 732 := 
by
  sorry

end initial_girls_l2373_237357


namespace arithmetic_sequence_sum_l2373_237303

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2)
  (h : a 3 = 20 - a 6) : S 8 = 80 :=
sorry

end arithmetic_sequence_sum_l2373_237303


namespace jane_mean_score_l2373_237307

-- Define Jane's scores as a list
def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

-- Define the total number of quizzes
def total_quizzes : ℕ := 6

-- Define the sum of Jane's scores
def sum_scores : ℕ := 95 + 88 + 94 + 86 + 92 + 91

-- Define the mean score calculation
def mean_score : ℕ := sum_scores / total_quizzes

-- The theorem to state Jane's mean score
theorem jane_mean_score : mean_score = 91 := by
  -- This theorem statement correctly reflects the mathematical problem provided.
  sorry

end jane_mean_score_l2373_237307


namespace interval_of_x₀_l2373_237328

-- Definition of the problem
variable (x₀ : ℝ)

-- Conditions
def condition_1 := x₀ > 0 ∧ x₀ < Real.pi
def condition_2 := Real.sin x₀ + Real.cos x₀ = 2 / 3

-- Proof problem statement
theorem interval_of_x₀ 
  (h1 : condition_1 x₀)
  (h2 : condition_2 x₀) : 
  x₀ > 7 * Real.pi / 12 ∧ x₀ < 3 * Real.pi / 4 := 
sorry

end interval_of_x₀_l2373_237328


namespace determine_number_of_solutions_l2373_237334

noncomputable def num_solutions_eq : Prop :=
  let f (x : ℝ) := (3 * x ^ 2 - 15 * x) / (x ^ 2 - 7 * x + 10)
  let g (x : ℝ) := x - 4
  ∃ S : Finset ℝ, 
    (∀ x ∈ S, (x ≠ 2 ∧ x ≠ 5) ∧ f x = g x) ∧
    S.card = 2

theorem determine_number_of_solutions : num_solutions_eq :=
  by
  sorry

end determine_number_of_solutions_l2373_237334


namespace triangle_perimeter_l2373_237397

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 19)
  (ineq1 : a + b > c) (ineq2 : a + c > b) (ineq3 : b + c > a) : a + b + c = 44 :=
by
  -- Proof omitted
  sorry

end triangle_perimeter_l2373_237397


namespace interest_rate_l2373_237378

theorem interest_rate (total_investment : ℝ) (investment1 : ℝ) (investment2 : ℝ) (rate2 : ℝ) (interest1 : ℝ → ℝ) (interest2 : ℝ → ℝ) :
  (total_investment = 5400) →
  (investment1 = 3000) →
  (investment2 = total_investment - investment1) →
  (rate2 = 0.10) →
  (interest1 investment1 = investment1 * (interest1 1)) →
  (interest2 investment2 = investment2 * rate2) →
  interest1 investment1 = interest2 investment2 →
  interest1 1 = 0.08 :=
by
  intros
  sorry

end interest_rate_l2373_237378


namespace plane_equation_l2373_237322

theorem plane_equation (x y z : ℝ)
  (h₁ : ∃ t : ℝ, x = 2 * t + 1 ∧ y = -3 * t ∧ z = 3 - t)
  (h₂ : ∃ (t₁ t₂ : ℝ), 4 * t₁ + 5 * t₂ - 3 = 0 ∧ 2 * t₁ + t₂ + 2 * t₂ = 0) : 
  2*x - y + 7*z - 23 = 0 :=
sorry

end plane_equation_l2373_237322


namespace robin_extra_drinks_l2373_237362

-- Conditions
def initial_sodas : ℕ := 22
def initial_energy_drinks : ℕ := 15
def initial_smoothies : ℕ := 12
def drank_sodas : ℕ := 6
def drank_energy_drinks : ℕ := 9
def drank_smoothies : ℕ := 2

-- Total drinks bought
def total_drinks_bought : ℕ :=
  initial_sodas + initial_energy_drinks + initial_smoothies
  
-- Total drinks consumed
def total_drinks_consumed : ℕ :=
  drank_sodas + drank_energy_drinks + drank_smoothies

-- Number of extra drinks
def extra_drinks : ℕ :=
  total_drinks_bought - total_drinks_consumed

-- Theorem to prove
theorem robin_extra_drinks : extra_drinks = 32 :=
  by
  -- skipping the proof
  sorry

end robin_extra_drinks_l2373_237362


namespace fraction_solution_l2373_237374

theorem fraction_solution (x : ℝ) (h1 : (x - 4) / (x^2) = 0) (h2 : x ≠ 0) : x = 4 :=
sorry

end fraction_solution_l2373_237374


namespace rest_area_milepost_l2373_237315

theorem rest_area_milepost (milepost_first : ℕ) (milepost_seventh : ℕ) (h_first : milepost_first = 20) (h_seventh : milepost_seventh = 140) : 
  ∃ milepost_rest : ℕ, milepost_rest = (milepost_first + milepost_seventh) / 2 ∧ milepost_rest = 80 :=
by
  sorry

end rest_area_milepost_l2373_237315


namespace polar_to_rectangular_l2373_237311

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), r = 2 ∧ θ = 2 * Real.pi / 3 → 
  (r * Real.cos θ, r * Real.sin θ) = (-1, Real.sqrt 3) := by
  sorry

end polar_to_rectangular_l2373_237311


namespace initial_blue_balls_l2373_237345

theorem initial_blue_balls (B : ℕ) 
  (h1 : 18 - 3 = 15) 
  (h2 : (B - 3) / 15 = 1 / 5) : 
  B = 6 :=
by sorry

end initial_blue_balls_l2373_237345


namespace triangle_CD_length_l2373_237309

noncomputable def triangle_AB_values : ℝ := 4024
noncomputable def triangle_AC_values : ℝ := 4024
noncomputable def triangle_BC_values : ℝ := 2012
noncomputable def CD_value : ℝ := 504.5

theorem triangle_CD_length 
  (AB AC : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (h1 : AB = triangle_AB_values)
  (h2 : AC = triangle_AC_values)
  (h3 : BC = triangle_BC_values) :
  CD = CD_value := by
  sorry

end triangle_CD_length_l2373_237309


namespace magnitude_of_Z_l2373_237314

-- Define the complex number Z
def Z : ℂ := 3 - 4 * Complex.I

-- Define the theorem to prove the magnitude of Z
theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end magnitude_of_Z_l2373_237314


namespace inverse_proportionality_ratio_l2373_237351

variable {x y k x1 x2 y1 y2 : ℝ}

theorem inverse_proportionality_ratio
  (h1 : x * y = k)
  (hx1 : x1 ≠ 0)
  (hx2 : x2 ≠ 0)
  (hy1 : y1 ≠ 0)
  (hy2 : y2 ≠ 0)
  (hx_ratio : x1 / x2 = 3 / 4)
  (hxy1 : x1 * y1 = k)
  (hxy2 : x2 * y2 = k) :
  y1 / y2 = 4 / 3 := by
  sorry

end inverse_proportionality_ratio_l2373_237351


namespace least_of_consecutive_odds_l2373_237396

noncomputable def average_of_consecutive_odds (n : ℕ) (start : ℤ) : ℤ :=
start + (2 * (n - 1))

theorem least_of_consecutive_odds
    (n : ℕ)
    (mean : ℤ)
    (h : n = 30 ∧ mean = 526) : 
    average_of_consecutive_odds 1 (mean * 2 - (n - 1)) = 497 :=
by
  sorry

end least_of_consecutive_odds_l2373_237396


namespace num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l2373_237371

def total_students : ℕ := 800

def percentage_blue_shirts : ℕ := 45
def percentage_red_shirts : ℕ := 23
def percentage_green_shirts : ℕ := 15

def percentage_black_pants : ℕ := 30
def percentage_khaki_pants : ℕ := 25
def percentage_jeans_pants : ℕ := 10

def percentage_white_shoes : ℕ := 40
def percentage_black_shoes : ℕ := 20
def percentage_brown_shoes : ℕ := 15

def students_other_color_shirts : ℕ :=
  total_students * (100 - (percentage_blue_shirts + percentage_red_shirts + percentage_green_shirts)) / 100

def students_other_types_pants : ℕ :=
  total_students * (100 - (percentage_black_pants + percentage_khaki_pants + percentage_jeans_pants)) / 100

def students_other_color_shoes : ℕ :=
  total_students * (100 - (percentage_white_shoes + percentage_black_shoes + percentage_brown_shoes)) / 100

theorem num_students_other_color_shirts : students_other_color_shirts = 136 := by
  sorry

theorem num_students_other_types_pants : students_other_types_pants = 280 := by
  sorry

theorem num_students_other_color_shoes : students_other_color_shoes = 200 := by
  sorry

end num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l2373_237371


namespace charlyn_viewable_area_l2373_237366

noncomputable def charlyn_sees_area (side_length viewing_distance : ℝ) : ℝ :=
  let inner_viewable_area := (side_length^2 - (side_length - 2 * viewing_distance)^2)
  let rectangular_area := 4 * (side_length * viewing_distance)
  let circular_corner_area := 4 * ((viewing_distance^2 * Real.pi) / 4)
  inner_viewable_area + rectangular_area + circular_corner_area

theorem charlyn_viewable_area :
  let side_length := 7
  let viewing_distance := 1.5
  charlyn_sees_area side_length viewing_distance = 82 := 
by
  sorry

end charlyn_viewable_area_l2373_237366


namespace necessary_but_not_sufficient_l2373_237377

-- Define the necessary conditions
variables {a b c d : ℝ}

-- State the main theorem
theorem necessary_but_not_sufficient (h₁ : a > b) (h₂ : c > d) : (a + c > b + d) :=
by
  -- Placeholder for the proof (insufficient as per the context problem statement)
  sorry

end necessary_but_not_sufficient_l2373_237377


namespace num_three_digit_numbers_l2373_237353

theorem num_three_digit_numbers (a b c : ℕ) :
  a ≠ 0 →
  b = (a + c) / 2 →
  c = a - b →
  ∃ n1 n2 n3 : ℕ, 
    (n1 = 100 * 3 + 10 * 2 + 1) ∧
    (n2 = 100 * 9 + 10 * 6 + 3) ∧
    (n3 = 100 * 6 + 10 * 4 + 2) ∧ 
    3 = 3 := 
sorry  

end num_three_digit_numbers_l2373_237353


namespace solutions_to_cube_eq_27_l2373_237358

theorem solutions_to_cube_eq_27 (z : ℂ) : 
  (z^3 = 27) ↔ (z = 3 ∨ z = (Complex.mk (-3 / 2) (3 * Real.sqrt 3 / 2)) ∨ z = (Complex.mk (-3 / 2) (-3 * Real.sqrt 3 / 2))) :=
by sorry

end solutions_to_cube_eq_27_l2373_237358


namespace probability_not_siblings_l2373_237386

noncomputable def num_individuals : ℕ := 6
noncomputable def num_pairs : ℕ := num_individuals / 2
noncomputable def total_pairs : ℕ := num_individuals * (num_individuals - 1) / 2
noncomputable def sibling_pairs : ℕ := num_pairs
noncomputable def non_sibling_pairs : ℕ := total_pairs - sibling_pairs

theorem probability_not_siblings :
  (non_sibling_pairs : ℚ) / total_pairs = 4 / 5 := 
by sorry

end probability_not_siblings_l2373_237386


namespace arithmetic_sequence_sum_l2373_237344

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (h_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 5) (h_a5 : a 5 = 9) :
  S 7 = 49 :=
sorry

end arithmetic_sequence_sum_l2373_237344


namespace johns_contribution_correct_l2373_237391

noncomputable def average_contribution_before : Real := sorry
noncomputable def total_contributions_by_15 : Real := 15 * average_contribution_before
noncomputable def new_average_contribution : Real := 150
noncomputable def johns_contribution : Real := average_contribution_before * 15 + 1377.3

-- The theorem we want to prove
theorem johns_contribution_correct :
  (new_average_contribution = (total_contributions_by_15 + johns_contribution) / 16) ∧
  (new_average_contribution = 2.2 * average_contribution_before) :=
sorry

end johns_contribution_correct_l2373_237391


namespace ticket_cost_is_correct_l2373_237394

-- Conditions
def total_amount_raised : ℕ := 620
def number_of_tickets_sold : ℕ := 155

-- Definition of cost per ticket
def cost_per_ticket : ℕ := total_amount_raised / number_of_tickets_sold

-- The theorem to be proven
theorem ticket_cost_is_correct : cost_per_ticket = 4 :=
by
  sorry

end ticket_cost_is_correct_l2373_237394


namespace avg_values_l2373_237390

theorem avg_values (z : ℝ) : (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end avg_values_l2373_237390


namespace coeff_sum_eq_neg_two_l2373_237337

theorem coeff_sum_eq_neg_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^10 + x^4 + 1) = a + a₁ * (x+1) + a₂ * (x+1)^2 + a₃ * (x+1)^3 + a₄ * (x+1)^4 
   + a₅ * (x+1)^5 + a₆ * (x+1)^6 + a₇ * (x+1)^7 + a₈ * (x+1)^8 + a₉ * (x+1)^9 + a₁₀ * (x+1)^10) 
  → (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2) := 
by sorry

end coeff_sum_eq_neg_two_l2373_237337


namespace no_real_roots_of_geom_seq_l2373_237382

theorem no_real_roots_of_geom_seq (a b c : ℝ) (h_geom_seq : b^2 = a * c) : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  -- You can assume the steps of proving here
  sorry

end no_real_roots_of_geom_seq_l2373_237382


namespace total_water_in_heaters_l2373_237383

theorem total_water_in_heaters (wallace_capacity : ℕ) (catherine_capacity : ℕ) 
(wallace_water : ℕ) (catherine_water : ℕ) :
  wallace_capacity = 40 →
  (wallace_water = (3 * wallace_capacity) / 4) →
  wallace_capacity = 2 * catherine_capacity →
  (catherine_water = (3 * catherine_capacity) / 4) →
  wallace_water + catherine_water = 45 :=
by
  sorry

end total_water_in_heaters_l2373_237383


namespace number_of_valid_pairs_l2373_237340

theorem number_of_valid_pairs :
  (∃ (count : ℕ), count = 280 ∧
    (∃ (m n : ℕ),
      1 ≤ m ∧ m ≤ 2899 ∧
      5^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 5^(n+1))) :=
sorry

end number_of_valid_pairs_l2373_237340


namespace part_one_part_two_l2373_237361

noncomputable def f (x a: ℝ) : ℝ := abs (x - 1) + abs (x + a)
noncomputable def g (a : ℝ) : ℝ := a^2 - a - 2

theorem part_one (x : ℝ) : f x 3 > g 3 + 2 ↔ x < -4 ∨ x > 2 := by
  sorry

theorem part_two (a : ℝ) :
  (∀ x : ℝ, -a ≤ x ∧ x ≤ 1 → f x a ≤ g a) ↔ a ≥ 3 := by
  sorry

end part_one_part_two_l2373_237361


namespace find_x_squared_plus_y_squared_l2373_237381

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l2373_237381


namespace problem1_problem2_l2373_237325

theorem problem1 (n : ℕ) : 2 ≤ (1 + 1 / n) ^ n ∧ (1 + 1 / n) ^ n < 3 :=
sorry

theorem problem2 (n : ℕ) : (n / 3) ^ n < n! :=
sorry

end problem1_problem2_l2373_237325


namespace initial_jelly_beans_l2373_237335

theorem initial_jelly_beans (total_children : ℕ) (percentage : ℕ) (jelly_per_child : ℕ) (remaining_jelly : ℕ) :
  (percentage = 80) → (total_children = 40) → (jelly_per_child = 2) → (remaining_jelly = 36) →
  (total_children * percentage / 100 * jelly_per_child + remaining_jelly = 100) :=
by
  intros h1 h2 h3 h4
  sorry

end initial_jelly_beans_l2373_237335


namespace number_of_children_is_30_l2373_237323

-- Informal statements
def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def children_after_adding_10 := children + 10

-- Formal proof statement
theorem number_of_children_is_30 :
  children_after_adding_10 = 30 := by
  sorry

end number_of_children_is_30_l2373_237323


namespace actual_cost_before_decrease_l2373_237379

theorem actual_cost_before_decrease (x : ℝ) (h : 0.76 * x = 1064) : x = 1400 :=
by
  sorry

end actual_cost_before_decrease_l2373_237379


namespace rational_numbers_cubic_sum_l2373_237318

theorem rational_numbers_cubic_sum
  (a b c : ℚ)
  (h1 : a - b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3) :
  a^3 + b^3 + c^3 = 1 :=
by
  sorry

end rational_numbers_cubic_sum_l2373_237318


namespace expand_expression_l2373_237380

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 :=
by
  sorry

end expand_expression_l2373_237380


namespace find_t_l2373_237360

theorem find_t (t : ℝ) (h : (1 / (t+3) + 3 * t / (t+3) - 4 / (t+3)) = 5) : t = -9 :=
by
  sorry

end find_t_l2373_237360


namespace division_problem_l2373_237370

-- Define the involved constants and operations
def expr1 : ℚ := 5 / 2 * 3
def expr2 : ℚ := 100 / expr1

-- Formulate the final equality
theorem division_problem : expr2 = 40 / 3 :=
  by sorry

end division_problem_l2373_237370


namespace light_stripes_total_area_l2373_237327

theorem light_stripes_total_area (x : ℝ) (h : 45 * x = 135) :
  2 * x + 4 * x + 6 * x + 8 * x = 60 := 
sorry

end light_stripes_total_area_l2373_237327


namespace find_ratio_of_constants_l2373_237317

theorem find_ratio_of_constants (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h₁ : 8 * x - 6 * y = c) (h₂ : 12 * y - 18 * x = d) : c / d = -4 / 9 := 
sorry

end find_ratio_of_constants_l2373_237317


namespace find_C_coordinates_l2373_237356

variables {A B M L C : ℝ × ℝ}

def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def on_line_bisector (L B : ℝ × ℝ) : Prop :=
  B.1 = 6  -- Vertical line through B

theorem find_C_coordinates
  (A := (2, 8))
  (M := (4, 11))
  (L := (6, 6))
  (hM : is_midpoint M A B)
  (hL : on_line_bisector L B) :
  C = (6, 14) :=
sorry

end find_C_coordinates_l2373_237356


namespace tiffany_lives_after_game_l2373_237312

/-- Tiffany's initial number of lives -/
def initial_lives : ℕ := 43

/-- Lives Tiffany loses in the hard part of the game -/
def lost_lives : ℕ := 14

/-- Lives Tiffany gains in the next level -/
def gained_lives : ℕ := 27

/-- Calculate the total lives Tiffany has after losing and gaining lives -/
def total_lives : ℕ := (initial_lives - lost_lives) + gained_lives

-- Prove that the total number of lives Tiffany has is 56
theorem tiffany_lives_after_game : total_lives = 56 := by
  -- This is where the proof would go
  sorry

end tiffany_lives_after_game_l2373_237312


namespace divide_0_24_by_0_004_l2373_237310

theorem divide_0_24_by_0_004 : 0.24 / 0.004 = 60 := by
  sorry

end divide_0_24_by_0_004_l2373_237310


namespace calculate_expression_l2373_237332

theorem calculate_expression :
  2⁻¹ + (3 - Real.pi)^0 + abs (2 * Real.sqrt 3 - Real.sqrt 2) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 12 = 3 / 2 :=
sorry

end calculate_expression_l2373_237332


namespace find_m_l2373_237306

-- Definitions of the given vectors a, b, and c
def vec_a (m : ℝ) : ℝ × ℝ := (1, m)
def vec_b : ℝ × ℝ := (2, 5)
def vec_c (m : ℝ) : ℝ × ℝ := (m, 3)

-- Definition of vector addition and subtraction
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Parallel vectors condition: the ratio of their components must be equal
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- The main theorem stating the desired result
theorem find_m (m : ℝ) :
  parallel (vec_add (vec_a m) (vec_c m)) (vec_sub (vec_a m) vec_b) ↔ 
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 :=
by
  sorry

end find_m_l2373_237306


namespace community_group_loss_l2373_237302

def cookies_bought : ℕ := 800
def cost_per_4_cookies : ℚ := 3 -- dollars per 4 cookies
def sell_per_3_cookies : ℚ := 2 -- dollars per 3 cookies

def cost_per_cookie : ℚ := cost_per_4_cookies / 4
def sell_per_cookie : ℚ := sell_per_3_cookies / 3

def total_cost (n : ℕ) (cost_per_cookie : ℚ) : ℚ := n * cost_per_cookie
def total_revenue (n : ℕ) (sell_per_cookie : ℚ) : ℚ := n * sell_per_cookie

def loss (n : ℕ) (cost_per_cookie sell_per_cookie : ℚ) : ℚ := 
  total_cost n cost_per_cookie - total_revenue n sell_per_cookie

theorem community_group_loss : loss cookies_bought cost_per_cookie sell_per_cookie = 64 := by
  sorry

end community_group_loss_l2373_237302


namespace solve_equation_l2373_237305

theorem solve_equation :
  ∀ y : ℤ, 4 * (y - 1) = 1 - 3 * (y - 3) → y = 2 :=
by
  intros y h
  sorry

end solve_equation_l2373_237305


namespace xiaoxia_exceeds_xiaoming_l2373_237365

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  52 + 15 * n > 70 + 12 * n := 
sorry

end xiaoxia_exceeds_xiaoming_l2373_237365


namespace price_of_item_a_l2373_237349

theorem price_of_item_a : 
  let coins_1000 := 7
  let coins_100 := 4
  let coins_10 := 5
  let price_1000 := coins_1000 * 1000
  let price_100 := coins_100 * 100
  let price_10 := coins_10 * 10
  let total_price := price_1000 + price_100 + price_10
  total_price = 7450 := by
    sorry

end price_of_item_a_l2373_237349


namespace no_food_dogs_l2373_237392

theorem no_food_dogs (total_dogs watermelon_liking salmon_liking chicken_liking ws_liking sc_liking wc_liking wsp_liking : ℕ) 
    (h_total : total_dogs = 100)
    (h_watermelon : watermelon_liking = 20) 
    (h_salmon : salmon_liking = 70) 
    (h_chicken : chicken_liking = 10) 
    (h_ws : ws_liking = 10) 
    (h_sc : sc_liking = 5) 
    (h_wc : wc_liking = 3) 
    (h_wsp : wsp_liking = 2) :
    (total_dogs - ((watermelon_liking - ws_liking - wc_liking + wsp_liking) + 
    (salmon_liking - ws_liking - sc_liking + wsp_liking) + 
    (chicken_liking - sc_liking - wc_liking + wsp_liking) + 
    (ws_liking - wsp_liking) + 
    (sc_liking - wsp_liking) + 
    (wc_liking - wsp_liking) + wsp_liking)) = 28 :=
  by sorry

end no_food_dogs_l2373_237392


namespace edward_cards_l2373_237355

noncomputable def num_cards_each_binder : ℝ := (7496.5 + 27.7) / 23
noncomputable def num_cards_fewer_binder : ℝ := num_cards_each_binder - 27.7

theorem edward_cards : 
  (⌊num_cards_each_binder + 0.5⌋ = 327) ∧ (⌊num_cards_fewer_binder + 0.5⌋ = 299) :=
by
  sorry

end edward_cards_l2373_237355


namespace geometric_sequence_b_value_l2373_237354

theorem geometric_sequence_b_value (b : ℝ) (h1 : 25 * b = b^2) (h2 : b * (1 / 4) = b / 4) :
  b = 5 / 2 :=
sorry

end geometric_sequence_b_value_l2373_237354


namespace problem_statement_l2373_237387

theorem problem_statement (x Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
    10 * (6 * x + 14 * Real.pi) = 4 * Q := 
sorry

end problem_statement_l2373_237387


namespace problem_solution_l2373_237347

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log (2 - x) / Real.log 2 else 2 ^ (x - 1)

theorem problem_solution : f (-2) + f (Real.log 12 / Real.log 2) = 9 := by
  sorry

end problem_solution_l2373_237347


namespace problem_EF_fraction_of_GH_l2373_237352

theorem problem_EF_fraction_of_GH (E F G H : Type) 
  (GE EH GH GF FH EF : ℝ) 
  (h1 : GE = 3 * EH) 
  (h2 : GF = 8 * FH)
  (h3 : GH = GE + EH)
  (h4 : GH = GF + FH) : 
  EF = 5 / 36 * GH :=
by
  sorry

end problem_EF_fraction_of_GH_l2373_237352


namespace sum_of_remainders_mod_11_l2373_237398

theorem sum_of_remainders_mod_11
    (a b c d : ℤ)
    (h₁ : a % 11 = 2)
    (h₂ : b % 11 = 4)
    (h₃ : c % 11 = 6)
    (h₄ : d % 11 = 8) :
    (a + b + c + d) % 11 = 9 :=
by
  sorry

end sum_of_remainders_mod_11_l2373_237398


namespace find_constant_term_l2373_237339

-- Definitions based on conditions:
def sum_of_coeffs (n : ℕ) : ℕ := 4 ^ n
def sum_of_binom_coeffs (n : ℕ) : ℕ := 2 ^ n
def P_plus_Q_equals (n : ℕ) : Prop := sum_of_coeffs n + sum_of_binom_coeffs n = 272

-- Constant term in the binomial expansion:
def constant_term (n r : ℕ) : ℕ := Nat.choose n r * (3 ^ (n - r))

-- The proof statement
theorem find_constant_term : 
  ∃ n r : ℕ, P_plus_Q_equals n ∧ n = 4 ∧ r = 1 ∧ constant_term n r = 108 :=
by {
  sorry
}

end find_constant_term_l2373_237339


namespace philip_school_trip_days_l2373_237336

-- Define the distances for the trips
def school_trip_one_way_miles : ℝ := 2.5
def market_trip_one_way_miles : ℝ := 2

-- Define the number of times he makes the trips in a day and in a week
def school_round_trips_per_day : ℕ := 2
def market_round_trips_per_week : ℕ := 1

-- Define the total mileage in a week
def weekly_mileage : ℕ := 44

-- Define the equation based on the given conditions
def weekly_school_trip_distance (d : ℕ) : ℝ :=
  (school_trip_one_way_miles * 2 * school_round_trips_per_day) * d

def weekly_market_trip_distance : ℝ :=
  (market_trip_one_way_miles * 2) * market_round_trips_per_week

-- Define the main theorem to be proved
theorem philip_school_trip_days :
  ∃ d : ℕ, weekly_school_trip_distance d + weekly_market_trip_distance = weekly_mileage ∧ d = 4 :=
by
  sorry

end philip_school_trip_days_l2373_237336


namespace election_total_polled_votes_l2373_237369

theorem election_total_polled_votes (V : ℝ) (invalid_votes : ℝ) (candidate_votes : ℝ) (margin : ℝ)
  (h1 : candidate_votes = 0.3 * V)
  (h2 : margin = 5000)
  (h3 : V = 0.3 * V + (0.3 * V + margin))
  (h4 : invalid_votes = 100) :
  V + invalid_votes = 12600 :=
by
  sorry

end election_total_polled_votes_l2373_237369


namespace factor_x10_minus_1296_l2373_237319

theorem factor_x10_minus_1296 (x : ℝ) : (x^10 - 1296) = (x^5 + 36) * (x^5 - 36) :=
  by
  sorry

end factor_x10_minus_1296_l2373_237319


namespace determine_parabola_coefficients_l2373_237300

noncomputable def parabola_coefficients (a b c : ℚ) : Prop :=
  ∀ (x y : ℚ), 
      (y = a * x^2 + b * x + c) ∧
      (
        ((4, 5) = (x, y)) ∧
        ((2, 3) = (x, y))
      )

theorem determine_parabola_coefficients :
  parabola_coefficients (-1/2) 4 (-3) :=
by
  sorry

end determine_parabola_coefficients_l2373_237300


namespace problem1_problem2_l2373_237368

theorem problem1 (x : ℝ) : (5 - 2 * x) ^ 2 - 16 = 0 ↔ x = 1 / 2 ∨ x = 9 / 2 := 
by 
  sorry

theorem problem2 (x : ℝ) : 2 * (x - 3) = x^2 - 9 ↔ x = 3 ∨ x = -1 := 
by 
  sorry

end problem1_problem2_l2373_237368


namespace problem_statement_l2373_237301

theorem problem_statement (a b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 2) 
  (h3 : ∀ n : ℕ, a (n + 2) = a n)
  (h_b : ∀ n : ℕ, b (n + 1) - b n = a n)
  (h_repeat : ∀ k : ℕ, ∃ m : ℕ, (b (2 * m) / a m) = k)
  : b 1 = 2 :=
sorry

end problem_statement_l2373_237301


namespace range_of_a_l2373_237343

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) ↔ a ≤ 3 := by
  sorry

end range_of_a_l2373_237343


namespace side_length_of_square_l2373_237367

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end side_length_of_square_l2373_237367


namespace angle_CBD_is_4_l2373_237372

theorem angle_CBD_is_4 (angle_ABC : ℝ) (angle_ABD : ℝ) (h₁ : angle_ABC = 24) (h₂ : angle_ABD = 20) : angle_ABC - angle_ABD = 4 :=
by 
  sorry

end angle_CBD_is_4_l2373_237372


namespace find_missing_number_l2373_237330

theorem find_missing_number (x : ℤ) (h : (4 + 3) + (8 - x - 1) = 11) : x = 3 :=
sorry

end find_missing_number_l2373_237330


namespace tourist_group_people_count_l2373_237316

def large_room_people := 3
def small_room_people := 2
def small_rooms_rented := 1
def people_in_small_room := small_rooms_rented * small_room_people

theorem tourist_group_people_count : 
  ∀ x : ℕ, x ≥ 1 ∧ (x + small_rooms_rented) = (people_in_small_room + x * large_room_people) → 
  (people_in_small_room + x * large_room_people) = 5 := 
  by
  sorry

end tourist_group_people_count_l2373_237316


namespace original_pencils_l2373_237359

-- Define the conditions
def pencils_added : ℕ := 30
def total_pencils_now : ℕ := 71

-- Define the theorem to prove the original number of pencils
theorem original_pencils (original_pencils : ℕ) :
  total_pencils_now = original_pencils + pencils_added → original_pencils = 41 :=
by
  intros h
  sorry

end original_pencils_l2373_237359
