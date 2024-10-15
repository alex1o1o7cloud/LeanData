import Mathlib

namespace NUMINAMATH_GPT_max_sin_B_l2267_226783

theorem max_sin_B (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (AB BC : ℝ)
    (hAB : AB = 25) (hBC : BC = 20) :
    ∃ sinB : ℝ, sinB = 3 / 5 := sorry

end NUMINAMATH_GPT_max_sin_B_l2267_226783


namespace NUMINAMATH_GPT_find_first_term_of_arithmetic_progression_l2267_226795

-- Definitions for the proof
def arithmetic_progression_first_term (L n d : ℕ) : ℕ :=
  L - (n - 1) * d

-- Theorem stating the proof problem
theorem find_first_term_of_arithmetic_progression (L n d : ℕ) (hL : L = 62) (hn : n = 31) (hd : d = 2) :
  arithmetic_progression_first_term L n d = 2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_first_term_of_arithmetic_progression_l2267_226795


namespace NUMINAMATH_GPT_num_supermarkets_in_US_l2267_226746

theorem num_supermarkets_in_US (U C : ℕ) (h1 : U + C = 420) (h2 : U = C + 56) : U = 238 :=
by
  sorry

end NUMINAMATH_GPT_num_supermarkets_in_US_l2267_226746


namespace NUMINAMATH_GPT_problem1_problem2_l2267_226796

-- Define a and b as real numbers
variables (a b : ℝ)

-- Problem 1: Prove (a-2b)^2 - (b-a)(a+b) = 2a^2 - 4ab + 3b^2
theorem problem1 : (a - 2 * b) ^ 2 - (b - a) * (a + b) = 2 * a ^ 2 - 4 * a * b + 3 * b ^ 2 :=
sorry

-- Problem 2: Prove (2a-b)^2 \cdot (2a+b)^2 = 16a^4 - 8a^2b^2 + b^4
theorem problem2 : (2 * a - b) ^ 2 * (2 * a + b) ^ 2 = 16 * a ^ 4 - 8 * a ^ 2 * b ^ 2 + b ^ 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2267_226796


namespace NUMINAMATH_GPT_more_birds_than_storks_l2267_226770

def initial_storks : ℕ := 5
def initial_birds : ℕ := 3
def additional_birds : ℕ := 4

def total_birds : ℕ := initial_birds + additional_birds

def stork_vs_bird_difference : ℕ := total_birds - initial_storks

theorem more_birds_than_storks : stork_vs_bird_difference = 2 := by
  sorry

end NUMINAMATH_GPT_more_birds_than_storks_l2267_226770


namespace NUMINAMATH_GPT_sum_of_three_numbers_l2267_226713

noncomputable def lcm_three_numbers (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem sum_of_three_numbers 
  (a b c : ℕ)
  (x : ℕ)
  (h1 : lcm_three_numbers a b c = 180)
  (h2 : a = 2 * x)
  (h3 : b = 3 * x)
  (h4 : c = 5 * x) : a + b + c = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l2267_226713


namespace NUMINAMATH_GPT_distance_from_town_l2267_226787

theorem distance_from_town (d : ℝ) :
  (7 < d ∧ d < 8) ↔ (d < 8 ∧ d > 7 ∧ d > 6 ∧ d ≠ 9) :=
by sorry

end NUMINAMATH_GPT_distance_from_town_l2267_226787


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2267_226700

def mutually_exclusive (A1 A2 : Prop) : Prop := (A1 ∧ A2) → False
def complementary (A1 A2 : Prop) : Prop := (A1 ∨ A2) ∧ ¬(A1 ∧ A2)

theorem necessary_but_not_sufficient {A1 A2 : Prop}: 
  mutually_exclusive A1 A2 → complementary A1 A2 → (¬(mutually_exclusive A1 A2 → complementary A1 A2) ∧ (complementary A1 A2 → mutually_exclusive A1 A2)) := 
  by
    sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2267_226700


namespace NUMINAMATH_GPT_exponent_inequality_l2267_226775

theorem exponent_inequality (a b c : ℝ) (h1 : a ≠ 1) (h2 : b ≠ 1) (h3 : c ≠ 1) (h4 : a > b) (h5 : b > c) (h6 : c > 0) : a ^ b > c ^ b :=
  sorry

end NUMINAMATH_GPT_exponent_inequality_l2267_226775


namespace NUMINAMATH_GPT_friends_payment_l2267_226702

theorem friends_payment
  (num_friends : ℕ) (num_bread : ℕ) (cost_bread : ℕ) 
  (num_hotteok : ℕ) (cost_hotteok : ℕ) (total_cost : ℕ)
  (cost_per_person : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_bread = 5)
  (h3 : cost_bread = 200)
  (h4 : num_hotteok = 7)
  (h5 : cost_hotteok = 800)
  (h6 : total_cost = num_bread * cost_bread + num_hotteok * cost_hotteok)
  (h7 : cost_per_person = total_cost / num_friends) :
  cost_per_person = 1650 := by
  sorry

end NUMINAMATH_GPT_friends_payment_l2267_226702


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2267_226720

-- Definitions
variable (f : ℝ → ℝ)

-- Condition that we need to prove
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

def is_symmetric_about_origin (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -g (-x)

-- Necessary and sufficient condition
theorem necessary_but_not_sufficient_condition : 
  (∀ x, |f x| = |f (-x)|) ↔ (∀ x, f x = -f (-x)) ∧ ¬(∀ x, |f x| = |f (-x)| → f x = -f (-x)) := by 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2267_226720


namespace NUMINAMATH_GPT_power_decomposition_l2267_226789

theorem power_decomposition (n m : ℕ) (h1 : n ≥ 2) 
  (h2 : n * n = 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) 
  (h3 : Nat.succ 19 = 21) 
  : m + n = 15 := sorry

end NUMINAMATH_GPT_power_decomposition_l2267_226789


namespace NUMINAMATH_GPT_reciprocal_of_neg_three_l2267_226739

theorem reciprocal_of_neg_three : -3 * (-1 / 3) = 1 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_three_l2267_226739


namespace NUMINAMATH_GPT_sum_of_digits_l2267_226793

def S (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits :
  (Finset.range 2013).sum S = 28077 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_l2267_226793


namespace NUMINAMATH_GPT_part_two_l2267_226725

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - 2 * x + a * Real.log x

theorem part_two (a : ℝ) (h : a = 4) (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h_cond : (f m a + f n a) / (m^2 * n^2) = 1) : m + n ≥ 3 :=
sorry

end NUMINAMATH_GPT_part_two_l2267_226725


namespace NUMINAMATH_GPT_seth_spent_more_l2267_226759

def cost_ice_cream (cartons : ℕ) (price : ℕ) := cartons * price
def cost_yogurt (cartons : ℕ) (price : ℕ) := cartons * price
def amount_spent (cost_ice : ℕ) (cost_yog : ℕ) := cost_ice - cost_yog

theorem seth_spent_more :
  amount_spent (cost_ice_cream 20 6) (cost_yogurt 2 1) = 118 := by
  sorry

end NUMINAMATH_GPT_seth_spent_more_l2267_226759


namespace NUMINAMATH_GPT_point_D_eq_1_2_l2267_226765

-- Definitions and conditions
def point : Type := ℝ × ℝ

def A : point := (-1, 4)
def B : point := (-4, -1)
def C : point := (4, 7)

-- Translate function
def translate (p : point) (dx dy : ℝ) := (p.1 + dx, p.2 + dy)

-- The translation distances found from A to C
def dx := C.1 - A.1
def dy := C.2 - A.2

-- The point D
def D : point := translate B dx dy

-- Proof objective
theorem point_D_eq_1_2 : D = (1, 2) := by
  sorry

end NUMINAMATH_GPT_point_D_eq_1_2_l2267_226765


namespace NUMINAMATH_GPT_solve_quadratic_eq_l2267_226757

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 4 * x - 1 = 0) : x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l2267_226757


namespace NUMINAMATH_GPT_jerry_apples_l2267_226760

theorem jerry_apples (J : ℕ) (h1 : 20 + 60 + J = 3 * 2 * 20):
  J = 40 :=
sorry

end NUMINAMATH_GPT_jerry_apples_l2267_226760


namespace NUMINAMATH_GPT_cookies_baked_l2267_226754

noncomputable def total_cookies (irin ingrid nell : ℚ) (percentage_ingrid : ℚ) : ℚ :=
  let total_ratio := irin + ingrid + nell
  let proportion_ingrid := ingrid / total_ratio
  let total_cookies := ingrid / (percentage_ingrid / 100)
  total_cookies

theorem cookies_baked (h_ratio: 9.18 + 5.17 + 2.05 = 16.4)
                      (h_percentage : 31.524390243902438 = 31.524390243902438) : 
  total_cookies 9.18 5.17 2.05 31.524390243902438 = 52 :=
by
  -- Placeholder for the proof.
  sorry

end NUMINAMATH_GPT_cookies_baked_l2267_226754


namespace NUMINAMATH_GPT_scientific_notation_of_viewers_l2267_226799

def million : ℝ := 10^6
def viewers : ℝ := 70.62 * million

theorem scientific_notation_of_viewers : viewers = 7.062 * 10^7 := by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_viewers_l2267_226799


namespace NUMINAMATH_GPT_find_decreased_amount_l2267_226764

variables (x y : ℝ)

axiom h1 : 0.20 * x - y = 6
axiom h2 : x = 50.0

theorem find_decreased_amount : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_decreased_amount_l2267_226764


namespace NUMINAMATH_GPT_mike_total_cards_l2267_226797

variable (original_cards : ℕ) (birthday_cards : ℕ)

def initial_cards : ℕ := 64
def received_cards : ℕ := 18

theorem mike_total_cards :
  original_cards = 64 →
  birthday_cards = 18 →
  original_cards + birthday_cards = 82 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mike_total_cards_l2267_226797


namespace NUMINAMATH_GPT_yanni_money_left_in_cents_l2267_226715

-- Define the constants based on the conditions
def initial_amount := 0.85
def mother_amount := 0.40
def found_amount := 0.50
def toy_cost := 1.60

-- Function to calculate the total amount
def total_amount := initial_amount + mother_amount + found_amount

-- Function to calculate the money left
def money_left := total_amount - toy_cost

-- Convert the remaining money from dollars to cents
def money_left_in_cents := money_left * 100

-- The theorem to prove
theorem yanni_money_left_in_cents : money_left_in_cents = 15 := by
  -- placeholder for proof, sorry used to skip the proof
  sorry

end NUMINAMATH_GPT_yanni_money_left_in_cents_l2267_226715


namespace NUMINAMATH_GPT_collinear_points_l2267_226782

variables (a b : ℝ × ℝ) (A B C D : ℝ × ℝ)

-- Define the vectors
noncomputable def vec_AB : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
noncomputable def vec_BC : ℝ × ℝ := (2 * a.1 + 8 * b.1, 2 * a.2 + 8 * b.2)
noncomputable def vec_CD : ℝ × ℝ := (3 * (a.1 - b.1), 3 * (a.2 - b.2))

-- Define the collinearity condition
def collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Translate the problem statement into Lean
theorem collinear_points (h₀ : a ≠ (0, 0)) (h₁ : b ≠ (0, 0)) (h₂ : ¬ (a.1 * b.2 - a.2 * b.1 = 0)):
  collinear (6 * (a.1 + b.1), 6 * (a.2 + b.2)) (5 * (a.1 + b.1, a.2 + b.2)) :=
sorry

end NUMINAMATH_GPT_collinear_points_l2267_226782


namespace NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l2267_226741

theorem first_term_of_arithmetic_sequence (T : ℕ → ℝ) (b : ℝ) 
  (h1 : ∀ n : ℕ, T n = (n * (2 * b + (n - 1) * 4)) / 2) 
  (h2 : ∃ d : ℝ, ∀ n : ℕ, T (4 * n) / T n = d) :
  b = 2 :=
by
  sorry

end NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l2267_226741


namespace NUMINAMATH_GPT_books_in_either_but_not_both_l2267_226769

theorem books_in_either_but_not_both (shared_books alice_books bob_unique_books : ℕ) 
    (h1 : shared_books = 12) 
    (h2 : alice_books = 26)
    (h3 : bob_unique_books = 8) : 
    (alice_books - shared_books) + bob_unique_books = 22 :=
by
  sorry

end NUMINAMATH_GPT_books_in_either_but_not_both_l2267_226769


namespace NUMINAMATH_GPT_multiple_of_first_number_l2267_226707

theorem multiple_of_first_number (F S M : ℕ) (hF : F = 15) (hS : S = 55) (h_relation : S = M * F + 10) : M = 3 :=
by
  -- We are given that F = 15, S = 55 and the relation S = M * F + 10
  -- We need to prove that M = 3
  sorry

end NUMINAMATH_GPT_multiple_of_first_number_l2267_226707


namespace NUMINAMATH_GPT_simplify_and_evaluate_at_3_l2267_226788

noncomputable def expression (x : ℝ) : ℝ := 
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1))

theorem simplify_and_evaluate_at_3 : expression 3 = -5 := 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_at_3_l2267_226788


namespace NUMINAMATH_GPT_intersection_result_complement_union_result_l2267_226752

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_result : A ∩ B = {x | 0 < x ∧ x < 2} :=
by
  sorry

theorem complement_union_result : (compl B) ∪ A = {x | x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_result_complement_union_result_l2267_226752


namespace NUMINAMATH_GPT_conference_center_distance_l2267_226786

theorem conference_center_distance
  (d : ℝ)  -- total distance to the conference center
  (t : ℝ)  -- total on-time duration
  (h1 : d = 40 * (t + 1.5))  -- condition from initial speed and late time
  (h2 : d - 40 = 60 * (t - 1.75))  -- condition from increased speed and early arrival
  : d = 310 := 
sorry

end NUMINAMATH_GPT_conference_center_distance_l2267_226786


namespace NUMINAMATH_GPT_simplify_expression_l2267_226728

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2267_226728


namespace NUMINAMATH_GPT_total_students_in_school_district_l2267_226740

def CampusA_students : Nat :=
  let students_per_grade : Nat := 100
  let num_grades : Nat := 5
  let special_education : Nat := 30
  (students_per_grade * num_grades) + special_education

def CampusB_students : Nat :=
  let students_per_grade : Nat := 120
  let num_grades : Nat := 5
  students_per_grade * num_grades

def CampusC_students : Nat :=
  let students_per_grade : Nat := 150
  let num_grades : Nat := 2
  let international_program : Nat := 50
  (students_per_grade * num_grades) + international_program

def total_students : Nat :=
  CampusA_students + CampusB_students + CampusC_students

theorem total_students_in_school_district : total_students = 1480 := by
  sorry

end NUMINAMATH_GPT_total_students_in_school_district_l2267_226740


namespace NUMINAMATH_GPT_value_of_I_l2267_226718

variables (T H I S : ℤ)

theorem value_of_I :
  H = 10 →
  T + H + I + S = 50 →
  H + I + T = 35 →
  S + I + T = 40 →
  I = 15 :=
  by
  sorry

end NUMINAMATH_GPT_value_of_I_l2267_226718


namespace NUMINAMATH_GPT_larger_square_area_multiple_l2267_226726

theorem larger_square_area_multiple (a b : ℕ) (h : a = 4 * b) :
  (a ^ 2) = 16 * (b ^ 2) :=
sorry

end NUMINAMATH_GPT_larger_square_area_multiple_l2267_226726


namespace NUMINAMATH_GPT_radius_excircle_ABC_l2267_226738

variables (A B C P Q : Point)
variables (r_ABP r_APQ r_AQC : ℝ) (re_ABP re_APQ re_AQC : ℝ)
variable (r_ABC : ℝ)

-- Conditions
-- Radii of the incircles of triangles ABP, APQ, and AQC are all equal to 1
axiom incircle_ABP : r_ABP = 1
axiom incircle_APQ : r_APQ = 1
axiom incircle_AQC : r_AQC = 1

-- Radii of the corresponding excircles opposite A for ABP, APQ, and AQC are 3, 6, and 5 respectively
axiom excircle_ABP : re_ABP = 3
axiom excircle_APQ : re_APQ = 6
axiom excircle_AQC : re_AQC = 5

-- Radius of the incircle of triangle ABC is 3/2
axiom incircle_ABC : r_ABC = 3 / 2

-- Theorem stating the radius of the excircle of triangle ABC opposite A is 135
theorem radius_excircle_ABC (r_ABC : ℝ) : r_ABC = 3 / 2 → ∀ (re_ABC : ℝ), re_ABC = 135 := 
by
  intros 
  sorry

end NUMINAMATH_GPT_radius_excircle_ABC_l2267_226738


namespace NUMINAMATH_GPT_tangerines_left_proof_l2267_226771

-- Define the number of tangerines Jimin ate
def tangerinesJiminAte : ℕ := 7

-- Define the total number of tangerines
def totalTangerines : ℕ := 12

-- Define the number of tangerines left
def tangerinesLeft : ℕ := totalTangerines - tangerinesJiminAte

-- Theorem stating the number of tangerines left equals 5
theorem tangerines_left_proof : tangerinesLeft = 5 := 
by
  sorry

end NUMINAMATH_GPT_tangerines_left_proof_l2267_226771


namespace NUMINAMATH_GPT_product_not_zero_l2267_226714

theorem product_not_zero (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) : (x - 2) * (x - 5) ≠ 0 := 
by 
  sorry

end NUMINAMATH_GPT_product_not_zero_l2267_226714


namespace NUMINAMATH_GPT_trivia_team_l2267_226763

theorem trivia_team (total_students groups students_per_group students_not_picked : ℕ) (h1 : total_students = 65)
  (h2 : groups = 8) (h3 : students_per_group = 6) (h4 : students_not_picked = total_students - groups * students_per_group) :
  students_not_picked = 17 :=
sorry

end NUMINAMATH_GPT_trivia_team_l2267_226763


namespace NUMINAMATH_GPT_unique_solution_of_system_l2267_226733

noncomputable def solve_system_of_equations (x1 x2 x3 x4 x5 x6 x7 : ℝ) : Prop :=
  10 * x1 + 3 * x2 + 4 * x3 + x4 + x5 = 0 ∧
  11 * x2 + 2 * x3 + 2 * x4 + 3 * x5 + x6 = 0 ∧
  15 * x3 + 4 * x4 + 5 * x5 + 4 * x6 + x7 = 0 ∧
  2 * x1 + x2 - 3 * x3 + 12 * x4 - 3 * x5 + x6 + x7 = 0 ∧
  6 * x1 - 5 * x2 + 3 * x3 - x4 + 17 * x5 + x6 = 0 ∧
  3 * x1 + 2 * x2 - 3 * x3 + 4 * x4 + x5 - 16 * x6 + 2 * x7 = 0 ∧
  4 * x1 - 8 * x2 + x3 + x4 - 3 * x5 + 19 * x7 = 0

theorem unique_solution_of_system :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℝ),
    solve_system_of_equations x1 x2 x3 x4 x5 x6 x7 →
    x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0 ∧ x6 = 0 ∧ x7 = 0 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h
  sorry

end NUMINAMATH_GPT_unique_solution_of_system_l2267_226733


namespace NUMINAMATH_GPT_spotlight_distance_l2267_226777

open Real

-- Definitions for the ellipsoid parameters
def ellipsoid_parameters (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ a - c = 1.5

-- Given conditions as input parameters
variables (a b c : ℝ)
variables (h_a : a = 2.7) -- semi-major axis half length
variables (h_c : c = 1.5) -- focal point distance

-- Prove that the distance from F2 to F1 is 12 cm
theorem spotlight_distance (h : ellipsoid_parameters a b c) : 2 * a - (a - c) = 12 :=
by sorry

end NUMINAMATH_GPT_spotlight_distance_l2267_226777


namespace NUMINAMATH_GPT_winnie_keeps_balloons_l2267_226758

theorem winnie_keeps_balloons : 
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  remainder = 4 :=
by
  let red := 20
  let white := 40
  let green := 70
  let yellow := 90
  let total_balloons := red + white + green + yellow
  let friends := 9
  let remainder := total_balloons % friends
  show remainder = 4
  sorry

end NUMINAMATH_GPT_winnie_keeps_balloons_l2267_226758


namespace NUMINAMATH_GPT_c_work_time_l2267_226784

theorem c_work_time (A B C : ℝ) 
  (h1 : A + B = 1/10) 
  (h2 : B + C = 1/5) 
  (h3 : C + A = 1/15) : 
  C = 1/12 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_c_work_time_l2267_226784


namespace NUMINAMATH_GPT_count_perfect_cubes_l2267_226729

theorem count_perfect_cubes (a b : ℕ) (h₁ : 200 < a) (h₂ : a < 1500) (h₃ : b = 6^3) :
  (∃! n : ℕ, 200 < n^3 ∧ n^3 < 1500) :=
sorry

end NUMINAMATH_GPT_count_perfect_cubes_l2267_226729


namespace NUMINAMATH_GPT_sin_70_eq_1_minus_2k_squared_l2267_226766

theorem sin_70_eq_1_minus_2k_squared (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * k^2 :=
by
  sorry

end NUMINAMATH_GPT_sin_70_eq_1_minus_2k_squared_l2267_226766


namespace NUMINAMATH_GPT_boat_speed_is_13_l2267_226736

noncomputable def boatSpeedStillWater : ℝ := 
  let Vs := 6 -- Speed of the stream in km/hr
  let time := 3.6315789473684212 -- Time taken in hours to travel 69 km downstream
  let distance := 69 -- Distance traveled in km
  (distance - Vs * time) / time

theorem boat_speed_is_13 : boatSpeedStillWater = 13 := by
  sorry

end NUMINAMATH_GPT_boat_speed_is_13_l2267_226736


namespace NUMINAMATH_GPT_find_fg3_l2267_226712

def f (x : ℝ) : ℝ := 4 * x - 3

def g (x : ℝ) : ℝ := (x + 2)^2 - 4 * x

theorem find_fg3 : f (g 3) = 49 :=
by
  sorry

end NUMINAMATH_GPT_find_fg3_l2267_226712


namespace NUMINAMATH_GPT_cost_per_unit_l2267_226743

theorem cost_per_unit 
  (units_per_month : ℕ := 400)
  (selling_price_per_unit : ℝ := 440)
  (profit_requirement : ℝ := 40000)
  (C : ℝ) :
  profit_requirement ≤ (units_per_month * selling_price_per_unit) - (units_per_month * C) → C ≤ 340 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_unit_l2267_226743


namespace NUMINAMATH_GPT_gcd_140_396_is_4_l2267_226723

def gcd_140_396 : ℕ := Nat.gcd 140 396

theorem gcd_140_396_is_4 : gcd_140_396 = 4 :=
by
  unfold gcd_140_396
  sorry

end NUMINAMATH_GPT_gcd_140_396_is_4_l2267_226723


namespace NUMINAMATH_GPT_percent_increase_is_fifteen_l2267_226794

noncomputable def percent_increase_from_sale_price_to_regular_price (P : ℝ) : ℝ :=
  ((P - (0.87 * P)) / (0.87 * P)) * 100

theorem percent_increase_is_fifteen (P : ℝ) (h : P > 0) :
  percent_increase_from_sale_price_to_regular_price P = 15 :=
by
  -- The proof is not required, so we use sorry.
  sorry

end NUMINAMATH_GPT_percent_increase_is_fifteen_l2267_226794


namespace NUMINAMATH_GPT_degree_to_radian_l2267_226735

theorem degree_to_radian (h : 1 = (π / 180)) : 60 = π * (1 / 3) := 
sorry

end NUMINAMATH_GPT_degree_to_radian_l2267_226735


namespace NUMINAMATH_GPT_geom_prog_all_integers_l2267_226773

theorem geom_prog_all_integers (b : ℕ) (r : ℚ) (a c : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, b * r ^ n = a * n + c) ∧ ∃ b_1 : ℤ, b = b_1 →
  (∀ n : ℕ, ∃ b_n : ℤ, b * r ^ n = b_n) :=
by
  sorry

end NUMINAMATH_GPT_geom_prog_all_integers_l2267_226773


namespace NUMINAMATH_GPT_min_initial_questionnaires_l2267_226710

theorem min_initial_questionnaires 
(N : ℕ) 
(h1 : 0.60 * (N:ℝ) + 0.60 * (N:ℝ) * 0.80 + 0.60 * (N:ℝ) * (0.80^2) ≥ 750) : 
  N ≥ 513 := sorry

end NUMINAMATH_GPT_min_initial_questionnaires_l2267_226710


namespace NUMINAMATH_GPT_sequence_sum_equality_l2267_226780

theorem sequence_sum_equality {a_n : ℕ → ℕ} (S_n : ℕ → ℕ) (n : ℕ) (h : n > 0) 
  (h1 : ∀ n, 3 * a_n n = 2 * S_n n + n) : 
  S_n n = (3^((n:ℕ)+1) - 2 * n) / 4 := 
sorry

end NUMINAMATH_GPT_sequence_sum_equality_l2267_226780


namespace NUMINAMATH_GPT_contrapositive_l2267_226737

theorem contrapositive (m : ℝ) :
  (∀ m > 0, ∃ x : ℝ, x^2 + x - m = 0) ↔ (∀ m ≤ 0, ∀ x : ℝ, x^2 + x - m ≠ 0) := by
  sorry

end NUMINAMATH_GPT_contrapositive_l2267_226737


namespace NUMINAMATH_GPT_solutionTriangle_l2267_226747

noncomputable def solveTriangle (a b : ℝ) (B : ℝ) : (ℝ × ℝ × ℝ) :=
  let A := 30
  let C := 30
  let c := 2
  (A, C, c)

theorem solutionTriangle :
  solveTriangle 2 (2 * Real.sqrt 3) 120 = (30, 30, 2) :=
by
  sorry

end NUMINAMATH_GPT_solutionTriangle_l2267_226747


namespace NUMINAMATH_GPT_divisibility_condition_l2267_226776

theorem divisibility_condition (a p q : ℕ) (hp : p > 0) (ha : a > 0) (hq : q > 0) (h : p ≤ q) :
  (p ∣ a^p ↔ p ∣ a^q) :=
sorry

end NUMINAMATH_GPT_divisibility_condition_l2267_226776


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l2267_226791

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2821) : 
  n + 6 = 406 := 
by
  -- Proof steps can be added here
  sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l2267_226791


namespace NUMINAMATH_GPT_zoey_finished_on_monday_l2267_226730

def total_days_read (n : ℕ) : ℕ :=
  2 * ((2^n) - 1)

def day_of_week_finished (start_day : ℕ) (total_days : ℕ) : ℕ :=
  (start_day + total_days) % 7

theorem zoey_finished_on_monday :
  day_of_week_finished 1 (total_days_read 18) = 1 :=
by
  sorry

end NUMINAMATH_GPT_zoey_finished_on_monday_l2267_226730


namespace NUMINAMATH_GPT_f_7_eq_neg3_l2267_226727

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_interval  : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4

theorem f_7_eq_neg3 : f 7 = -3 :=
  sorry

end NUMINAMATH_GPT_f_7_eq_neg3_l2267_226727


namespace NUMINAMATH_GPT_extra_bananas_each_child_gets_l2267_226762

theorem extra_bananas_each_child_gets
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (absent_children : ℕ)
  (present_children : ℕ)
  (total_bananas : ℕ)
  (bananas_each_present_child_gets : ℕ)
  (extra_bananas : ℕ) :
  total_children = 840 ∧
  bananas_per_child = 2 ∧
  absent_children = 420 ∧
  present_children = total_children - absent_children ∧
  total_bananas = total_children * bananas_per_child ∧
  bananas_each_present_child_gets = total_bananas / present_children ∧
  extra_bananas = bananas_each_present_child_gets - bananas_per_child →
  extra_bananas = 2 :=
by
  sorry

end NUMINAMATH_GPT_extra_bananas_each_child_gets_l2267_226762


namespace NUMINAMATH_GPT_dealer_profit_percentage_l2267_226778

-- Definitions of conditions
def cost_price (C : ℝ) : ℝ := C
def list_price (C : ℝ) : ℝ := 1.5 * C
def discount_rate : ℝ := 0.1
def discounted_price (C : ℝ) : ℝ := (1 - discount_rate) * list_price C
def price_for_45_articles (C : ℝ) : ℝ := 45 * discounted_price C
def cost_for_40_articles (C : ℝ) : ℝ := 40 * cost_price C

-- Statement of the problem
theorem dealer_profit_percentage (C : ℝ) (h₀ : C > 0) :
  (price_for_45_articles C - cost_for_40_articles C) / cost_for_40_articles C * 100 = 35 :=  
sorry

end NUMINAMATH_GPT_dealer_profit_percentage_l2267_226778


namespace NUMINAMATH_GPT_snail_reaches_tree_l2267_226716

theorem snail_reaches_tree
  (l1 l2 s : ℝ) 
  (h_l1 : l1 = 4) 
  (h_l2 : l2 = 3) 
  (h_s : s = 40) : 
  ∃ n : ℕ, n = 37 ∧ s - n*(l1 - l2) ≤ l1 :=
  by
    sorry

end NUMINAMATH_GPT_snail_reaches_tree_l2267_226716


namespace NUMINAMATH_GPT_median_hypotenuse_right_triangle_l2267_226706

/-- Prove that in a right triangle with legs of lengths 5 and 12,
  the median on the hypotenuse can be either 6 or 6.5. -/
theorem median_hypotenuse_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) :
  ∃ c : ℝ, (c = 6 ∨ c = 6.5) :=
sorry

end NUMINAMATH_GPT_median_hypotenuse_right_triangle_l2267_226706


namespace NUMINAMATH_GPT_well_depth_and_rope_length_l2267_226781

variables (x y : ℝ)

theorem well_depth_and_rope_length :
  (y = x / 4 - 3) ∧ (y = x / 5 + 1) → y = 17 ∧ x = 80 :=
by
  sorry
 
end NUMINAMATH_GPT_well_depth_and_rope_length_l2267_226781


namespace NUMINAMATH_GPT_function_properties_l2267_226731

theorem function_properties (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 25 ≠ 0 ∧ x^2 - (k - 6) * x + 16 ≠ 0) → 
  (-2 < k ∧ k < 10) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_function_properties_l2267_226731


namespace NUMINAMATH_GPT_total_reading_materials_l2267_226774

theorem total_reading_materials 
  (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (newspapers_per_shelf : ℕ) (graphic_novels_per_shelf : ℕ) 
  (bookshelves : ℕ)
  (h_books : books_per_shelf = 23) 
  (h_magazines : magazines_per_shelf = 61) 
  (h_newspapers : newspapers_per_shelf = 17) 
  (h_graphic_novels : graphic_novels_per_shelf = 29) 
  (h_bookshelves : bookshelves = 37) : 
  (books_per_shelf * bookshelves + magazines_per_shelf * bookshelves + newspapers_per_shelf * bookshelves + graphic_novels_per_shelf * bookshelves) = 4810 := 
by {
  -- Condition definitions are already given; the proof is omitted here.
  sorry
}

end NUMINAMATH_GPT_total_reading_materials_l2267_226774


namespace NUMINAMATH_GPT_determine_y_l2267_226755

theorem determine_y (x y : ℤ) (h1 : x^2 + 4 * x - 1 = y - 2) (h2 : x = -3) : y = -2 := by
  intros
  sorry

end NUMINAMATH_GPT_determine_y_l2267_226755


namespace NUMINAMATH_GPT_min_cylinder_volume_eq_surface_area_l2267_226749

theorem min_cylinder_volume_eq_surface_area (r h V S : ℝ) (hr : r > 0) (hh : h > 0)
  (hV : V = π * r^2 * h) (hS : S = 2 * π * r^2 + 2 * π * r * h) (heq : V = S) :
  V = 54 * π :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_min_cylinder_volume_eq_surface_area_l2267_226749


namespace NUMINAMATH_GPT_calories_per_person_l2267_226767

theorem calories_per_person (oranges : ℕ) (pieces_per_orange : ℕ) (people : ℕ) (calories_per_orange : ℕ) :
  oranges = 5 →
  pieces_per_orange = 8 →
  people = 4 →
  calories_per_orange = 80 →
  (oranges * pieces_per_orange) / people * ((oranges * calories_per_orange) / (oranges * pieces_per_orange)) = 100 :=
by
  intros h_oranges h_pieces_per_orange h_people h_calories_per_orange
  sorry

end NUMINAMATH_GPT_calories_per_person_l2267_226767


namespace NUMINAMATH_GPT_jerry_games_before_birthday_l2267_226732

def num_games_before (current received : ℕ) : ℕ :=
  current - received

theorem jerry_games_before_birthday : 
  ∀ (current received before : ℕ), current = 9 → received = 2 → before = num_games_before current received → before = 7 :=
by
  intros current received before h_current h_received h_before
  rw [h_current, h_received] at h_before
  exact h_before

end NUMINAMATH_GPT_jerry_games_before_birthday_l2267_226732


namespace NUMINAMATH_GPT_rectangular_solid_depth_l2267_226701

def SurfaceArea (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth
  (l w A : ℝ)
  (hl : l = 10)
  (hw : w = 9)
  (hA : A = 408) :
  ∃ h : ℝ, SurfaceArea l w h = A ∧ h = 6 :=
by
  use 6
  sorry

end NUMINAMATH_GPT_rectangular_solid_depth_l2267_226701


namespace NUMINAMATH_GPT_combined_tax_rate_l2267_226705

theorem combined_tax_rate
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h_john_income : john_income = 58000)
  (h_john_tax_rate : john_tax_rate = 0.30)
  (h_ingrid_income : ingrid_income = 72000)
  (h_ingrid_tax_rate : ingrid_tax_rate = 0.40) :
  ((john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income)) = 0.3553846154 :=
by
  sorry

end NUMINAMATH_GPT_combined_tax_rate_l2267_226705


namespace NUMINAMATH_GPT_common_chord_equation_l2267_226724

-- Definitions of the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 14*y + 15 = 0

-- Definition of the common chord line
def common_chord_line (x y : ℝ) : Prop := 6*x + 8*y - 3 = 0

-- The theorem to be proved
theorem common_chord_equation :
  (∀ x y, circle1 x y → circle2 x y → common_chord_line x y) :=
by sorry

end NUMINAMATH_GPT_common_chord_equation_l2267_226724


namespace NUMINAMATH_GPT_domain_of_f_of_f_l2267_226734

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (3 + x)

theorem domain_of_f_of_f :
  {x : ℝ | x ≠ -3 ∧ x ≠ -8 / 5} =
  {x : ℝ | ∃ y : ℝ, f x = y ∧ y ≠ -3 ∧ x ≠ -3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_of_f_l2267_226734


namespace NUMINAMATH_GPT_uniformity_of_scores_l2267_226798

/- Problem statement:
  Randomly select 10 students from class A and class B to participate in an English oral test. 
  The variances of their test scores are S1^2 = 13.2 and S2^2 = 26.26, respectively. 
  Then, we show that the scores of the 10 students from class A are more uniform than 
  those of the 10 students from class B.
-/

theorem uniformity_of_scores (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : 
    13.2 < 26.26 := 
by 
  sorry

end NUMINAMATH_GPT_uniformity_of_scores_l2267_226798


namespace NUMINAMATH_GPT_necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l2267_226717

theorem necessary_ab_given_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 4) : 
  a + b ≥ 4 :=
sorry

theorem not_sufficient_ab_given_a_b : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4 :=
sorry

end NUMINAMATH_GPT_necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l2267_226717


namespace NUMINAMATH_GPT_inequality_solution_reciprocal_inequality_l2267_226711

-- Proof Problem (1)
theorem inequality_solution (x : ℝ) : |x-1| + (1/2)*|x-3| < 2 ↔ (1 < x ∧ x < 3) :=
sorry

-- Proof Problem (2)
theorem reciprocal_inequality (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 2) : 
  (1/a) + (1/b) + (1/c) ≥ 9/2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_reciprocal_inequality_l2267_226711


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2267_226779

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l2267_226779


namespace NUMINAMATH_GPT_camera_pictures_olivia_camera_pictures_l2267_226745

theorem camera_pictures (phone_pics : Nat) (albums : Nat) (pics_per_album : Nat) (total_pics : Nat) : Prop :=
  phone_pics = 5 →
  albums = 8 →
  pics_per_album = 5 →
  total_pics = albums * pics_per_album →
  total_pics - phone_pics = 35

-- Here's the statement of the theorem followed by a sorry to indicate that the proof is not provided
theorem olivia_camera_pictures (phone_pics albums pics_per_album total_pics : Nat) (h1 : phone_pics = 5) (h2 : albums = 8) (h3 : pics_per_album = 5) (h4 : total_pics = albums * pics_per_album) : total_pics - phone_pics = 35 :=
by
  sorry

end NUMINAMATH_GPT_camera_pictures_olivia_camera_pictures_l2267_226745


namespace NUMINAMATH_GPT_geometric_series_first_term_l2267_226790

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_series_first_term_l2267_226790


namespace NUMINAMATH_GPT_infinite_superset_of_infinite_subset_l2267_226709

theorem infinite_superset_of_infinite_subset {A B : Set ℕ} (h_subset : B ⊆ A) (h_infinite : Infinite B) : Infinite A := 
sorry

end NUMINAMATH_GPT_infinite_superset_of_infinite_subset_l2267_226709


namespace NUMINAMATH_GPT_scientific_notation_384000_l2267_226772

theorem scientific_notation_384000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 384000 = a * 10 ^ n ∧ 
  a = 3.84 ∧ n = 5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_384000_l2267_226772


namespace NUMINAMATH_GPT_area_of_triangle_bounded_by_lines_l2267_226753

def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := - x + 5

theorem area_of_triangle_bounded_by_lines :
  let x_intercept_line1 := -3 / 2
  let x_intercept_line2 := 5
  let base := x_intercept_line2 - x_intercept_line1
  let intersection_x := 2 / 3
  let intersection_y := line1 intersection_x
  let height := intersection_y
  let area := (1 / 2) * base * height
  area = 169 / 12 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_bounded_by_lines_l2267_226753


namespace NUMINAMATH_GPT_find_omega_increasing_intervals_l2267_226708

noncomputable def f (ω x : ℝ) : ℝ :=
  (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * (Real.cos (ω * x))^2

noncomputable def g (x : ℝ) : ℝ :=
  let ω := 3/2
  f ω (x - (Real.pi / 2))

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : ∀ x : ℝ, f ω (x + 2*Real.pi / (2*ω)) = f ω x) :
  ω = 3/2 :=
  sorry

theorem increasing_intervals (k : ℤ) :
  ∃ a b, 
  a = (2/3 * k * Real.pi + Real.pi / 4) ∧ 
  b = (2/3 * k * Real.pi + 7 * Real.pi / 12) ∧
  ∀ x, a ≤ x ∧ x ≤ b → g x < g (x + 1) :=
  sorry

end NUMINAMATH_GPT_find_omega_increasing_intervals_l2267_226708


namespace NUMINAMATH_GPT_consecutive_even_product_l2267_226785

-- Define that there exist three consecutive even numbers such that the product equals 87526608.
theorem consecutive_even_product (a : ℤ) : 
  (a - 2) * a * (a + 2) = 87526608 → ∃ b : ℤ, b = a - 2 ∧ b % 2 = 0 ∧ ∃ c : ℤ, c = a ∧ c % 2 = 0 ∧ ∃ d : ℤ, d = a + 2 ∧ d % 2 = 0 :=
sorry

end NUMINAMATH_GPT_consecutive_even_product_l2267_226785


namespace NUMINAMATH_GPT_point_B_l2267_226721

-- Define constants for perimeter and speed factor
def perimeter : ℕ := 24
def speed_factor : ℕ := 2

-- Define the speeds of Jane and Hector
def hector_speed (s : ℕ) : ℕ := s
def jane_speed (s : ℕ) : ℕ := speed_factor * s

-- Define the times until they meet
def time_until_meeting (s : ℕ) : ℚ := perimeter / (hector_speed s + jane_speed s)

-- Distances walked by Hector and Jane upon meeting
noncomputable def hector_distance (s : ℕ) : ℚ := hector_speed s * time_until_meeting s
noncomputable def jane_distance (s : ℕ) : ℚ := jane_speed s * time_until_meeting s

-- Map the perimeter position to a point
def position_on_track (d : ℚ) : ℚ := d % perimeter

-- When they meet
theorem point_B (s : ℕ) (h₀ : 0 < s) : position_on_track (hector_distance s) = position_on_track (jane_distance s) → 
                          position_on_track (hector_distance s) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_point_B_l2267_226721


namespace NUMINAMATH_GPT_smallest_integer_representable_l2267_226768

theorem smallest_integer_representable (a b : ℕ) (h₁ : 3 < a) (h₂ : 3 < b)
    (h₃ : a + 3 = 3 * b + 1) : 13 = min (a + 3) (3 * b + 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_representable_l2267_226768


namespace NUMINAMATH_GPT_correct_option_is_D_l2267_226761

def p : Prop := 3 ≥ 3
def q : Prop := 3 > 4

theorem correct_option_is_D (hp : p) (hq : ¬ q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬ ¬ p :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_D_l2267_226761


namespace NUMINAMATH_GPT_three_same_colored_balls_l2267_226756

theorem three_same_colored_balls (balls : ℕ) (color_count : ℕ) (balls_per_color : ℕ) (h1 : balls = 60) (h2 : color_count = balls / balls_per_color) (h3 : balls_per_color = 6) :
  ∃ n, n = 21 ∧ (∀ picks : ℕ, picks ≥ n → ∃ c, ∃ k ≥ 3, k ≤ balls_per_color ∧ (c < color_count) ∧ (picks / c = k)) :=
sorry

end NUMINAMATH_GPT_three_same_colored_balls_l2267_226756


namespace NUMINAMATH_GPT_solve_for_q_l2267_226742

theorem solve_for_q :
  ∀ (k l q : ℚ),
    (3 / 4 = k / 108) →
    (3 / 4 = (l + k) / 126) →
    (3 / 4 = (q - l) / 180) →
    q = 148.5 :=
by
  intros k l q hk hl hq
  sorry

end NUMINAMATH_GPT_solve_for_q_l2267_226742


namespace NUMINAMATH_GPT_distinct_units_digits_of_cube_l2267_226703

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end NUMINAMATH_GPT_distinct_units_digits_of_cube_l2267_226703


namespace NUMINAMATH_GPT_enrico_earnings_l2267_226719

theorem enrico_earnings : 
  let price_per_kg := 0.50
  let weight_rooster1 := 30
  let weight_rooster2 := 40
  let total_earnings := price_per_kg * weight_rooster1 + price_per_kg * weight_rooster2
  total_earnings = 35 := 
by
  sorry

end NUMINAMATH_GPT_enrico_earnings_l2267_226719


namespace NUMINAMATH_GPT_impossible_fifty_pieces_l2267_226750

open Nat

theorem impossible_fifty_pieces :
  ¬ ∃ (m : ℕ), 1 + 3 * m = 50 :=
by
  sorry

end NUMINAMATH_GPT_impossible_fifty_pieces_l2267_226750


namespace NUMINAMATH_GPT_slope_of_line_l2267_226792

theorem slope_of_line (x y : ℝ) (h : x + 2 * y + 1 = 0) : y = - (1 / 2) * x - (1 / 2) :=
by
  sorry -- The solution would be filled in here

#check slope_of_line -- additional check to ensure theorem implementation is correct

end NUMINAMATH_GPT_slope_of_line_l2267_226792


namespace NUMINAMATH_GPT_min_calls_required_l2267_226751

-- Define the set of people involved in the communication
inductive Person
| A | B | C | D | E | F

-- Function to calculate the minimum number of calls for everyone to know all pieces of gossip
def minCalls : ℕ :=
  9

-- Theorem stating the minimum number of calls required
theorem min_calls_required : minCalls = 9 := by
  sorry

end NUMINAMATH_GPT_min_calls_required_l2267_226751


namespace NUMINAMATH_GPT_Kylie_coins_left_l2267_226744

-- Definitions based on given conditions
def piggyBank := 30
def brother := 26
def father := 2 * brother
def sofa := 15
def totalCoins := piggyBank + brother + father + sofa
def coinsGivenToLaura := totalCoins / 2
def coinsLeft := totalCoins - coinsGivenToLaura

-- Theorem statement
theorem Kylie_coins_left : coinsLeft = 62 := by sorry

end NUMINAMATH_GPT_Kylie_coins_left_l2267_226744


namespace NUMINAMATH_GPT_daniel_earnings_l2267_226722

def fabric_monday := 20
def yarn_monday := 15

def fabric_tuesday := 2 * fabric_monday
def yarn_tuesday := yarn_monday + 10

def fabric_wednesday := fabric_tuesday / 4
def yarn_wednesday := yarn_tuesday / 2

def price_per_yard_fabric := 2
def price_per_yard_yarn := 3

def total_fabric := fabric_monday + fabric_tuesday + fabric_wednesday
def total_yarn := yarn_monday + yarn_tuesday + yarn_wednesday

def earnings_fabric := total_fabric * price_per_yard_fabric
def earnings_yarn := total_yarn * price_per_yard_yarn

def total_earnings := earnings_fabric + earnings_yarn

theorem daniel_earnings :
  total_earnings = 299 := by
  sorry

end NUMINAMATH_GPT_daniel_earnings_l2267_226722


namespace NUMINAMATH_GPT_half_percent_of_160_l2267_226704

theorem half_percent_of_160 : (1 / 2 / 100) * 160 = 0.8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_half_percent_of_160_l2267_226704


namespace NUMINAMATH_GPT_cindy_correct_answer_l2267_226748

noncomputable def cindy_number (x : ℝ) : Prop :=
  (x - 10) / 5 = 40

theorem cindy_correct_answer (x : ℝ) (h : cindy_number x) : (x - 4) / 10 = 20.6 :=
by
  -- The proof is omitted as instructed
  sorry

end NUMINAMATH_GPT_cindy_correct_answer_l2267_226748
