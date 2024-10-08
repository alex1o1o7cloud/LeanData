import Mathlib

namespace angles_sum_eq_l185_185835

variables {a b c : ℝ} {A B C : ℝ}

theorem angles_sum_eq {a b c : ℝ} {A B C : ℝ}
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)
  (h7 : A + B + C = π)
  (h8 : (a + c - b) * (a + c + b) = 3 * a * c) :
  A + C = 2 * π / 3 :=
sorry

end angles_sum_eq_l185_185835


namespace find_cos_A_l185_185431

variable {A : Real}

theorem find_cos_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.tan A = 2 / 3) : Real.cos A = 3 * Real.sqrt 13 / 13 :=
by
  sorry

end find_cos_A_l185_185431


namespace bob_correct_answer_l185_185621

theorem bob_correct_answer (y : ℕ) (h : (y - 7) / 5 = 47) : (y - 5) / 7 = 33 :=
by 
  -- assumption h and the statement to prove
  sorry

end bob_correct_answer_l185_185621


namespace compare_cubic_terms_l185_185790

theorem compare_cubic_terms (a b : ℝ) :
    (a ≥ b → a^3 - b^3 ≥ a * b^2 - a^2 * b) ∧
    (a < b → a^3 - b^3 ≤ a * b^2 - a^2 * b) :=
by sorry

end compare_cubic_terms_l185_185790


namespace solve_cubic_equation_l185_185622

theorem solve_cubic_equation : ∀ x : ℝ, (x^3 - 5*x^2 + 6*x - 2 = 0) → (x = 2) :=
by
  intro x
  intro h
  sorry

end solve_cubic_equation_l185_185622


namespace find_B_l185_185581

theorem find_B (A B C : ℝ) (h : ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -1 → 
    2 / ((x-7)*(x+1)^2) = A / (x-7) + B / (x+1) + C / (x+1)^2) : 
  B = 1 / 16 :=
sorry

end find_B_l185_185581


namespace monotone_intervals_max_floor_a_l185_185307

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + a

theorem monotone_intervals (a : ℝ) (h : a = 1) :
  (∀ x, 0 < x ∧ x < 1 → deriv (λ x => f x 1) x > 0) ∧
  (∀ x, 1 ≤ x → deriv (λ x => f x 1) x < 0) :=
by
  sorry

theorem max_floor_a (a : ℝ) (h : ∀ x > 0, f x a ≤ x) : ⌊a⌋ = 1 :=
by
  sorry

end monotone_intervals_max_floor_a_l185_185307


namespace incorrect_statement_C_l185_185873

theorem incorrect_statement_C 
  (x y : ℝ)
  (n : ℕ)
  (data : Fin n → (ℝ × ℝ))
  (h : ∀ (i : Fin n), (x, y) = data i)
  (reg_eq : ∀ (x : ℝ), 0.85 * x - 85.71 = y) :
  ¬ (forall (x : ℝ), x = 160 → ∀ (y : ℝ), y = 50.29) := 
sorry

end incorrect_statement_C_l185_185873


namespace hyperbola_eccentricity_squared_l185_185638

/-- Given that F is the right focus of the hyperbola 
    \( C: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1 \) with \( a > 0 \) and \( b > 0 \), 
    a line perpendicular to the x-axis is drawn through point F, 
    intersecting one asymptote of the hyperbola at point M. 
    If \( |FM| = 2a \), denote the eccentricity of the hyperbola as \( e \). 
    Prove that \( e^2 = \frac{1 + \sqrt{17}}{2} \).
 -/
theorem hyperbola_eccentricity_squared (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3: c^2 = a^2 + b^2) (h4: b * c = 2 * a^2) : 
  (c / a)^2 = (1 + Real.sqrt 17) / 2 := 
sorry

end hyperbola_eccentricity_squared_l185_185638


namespace arithmetic_geometric_sequence_l185_185271

theorem arithmetic_geometric_sequence (a d : ℤ) (h1 : ∃ a d, (a - d) * a * (a + d) = 1000)
  (h2 : ∃ a d, a^2 = 2 * (a - d) * ((a + d) + 7)) :
  d = 8 ∨ d = -15 :=
by sorry

end arithmetic_geometric_sequence_l185_185271


namespace graph_of_equation_l185_185846

theorem graph_of_equation (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := 
by 
  sorry

end graph_of_equation_l185_185846


namespace range_x_minus_y_compare_polynomials_l185_185734

-- Proof Problem 1: Range of x - y
theorem range_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) : 
  -4 < x - y ∧ x - y < 2 := 
  sorry

-- Proof Problem 2: Comparison of polynomials
theorem compare_polynomials (x : ℝ) : 
  (x - 1) * (x^2 + x + 1) < (x + 1) * (x^2 - x + 1) := 
  sorry

end range_x_minus_y_compare_polynomials_l185_185734


namespace ratio_of_adult_to_kid_charge_l185_185795

variable (A : ℝ)  -- Charge for adults

-- Conditions
def kids_charge : ℝ := 3
def num_kids_per_day : ℝ := 8
def num_adults_per_day : ℝ := 10
def weekly_earnings : ℝ := 588
def days_per_week : ℝ := 7

-- Hypothesis for the relationship between charges and total weekly earnings
def total_weekly_earnings_eq : Prop :=
  days_per_week * (num_kids_per_day * kids_charge + num_adults_per_day * A) = weekly_earnings

-- Statement to be proved
theorem ratio_of_adult_to_kid_charge (h : total_weekly_earnings_eq A) : (A / kids_charge) = 2 := 
by 
  sorry

end ratio_of_adult_to_kid_charge_l185_185795


namespace find_Finley_age_l185_185087

variable (Roger Jill Finley : ℕ)
variable (Jill_age : Jill = 20)
variable (Roger_age : Roger = 2 * Jill + 5)
variable (Finley_condition : 15 + (Roger - Jill) = Finley - 30)

theorem find_Finley_age : Finley = 55 :=
by
  sorry

end find_Finley_age_l185_185087


namespace seats_in_row_l185_185879

theorem seats_in_row (y : ℕ → ℕ) (k b : ℕ) :
  (∀ x, y x = k * x + b) →
  y 1 = 20 →
  y 19 = 56 →
  y 26 = 70 :=
by
  intro h1 h2 h3
  -- Additional constraints to prove the given requirements
  sorry

end seats_in_row_l185_185879


namespace probability_each_mailbox_has_at_least_one_letter_l185_185146

noncomputable def probability_mailbox (total_letters : ℕ) (mailboxes : ℕ) : ℚ := 
  let total_ways := mailboxes ^ total_letters
  let favorable_ways := Nat.choose total_letters (mailboxes - 1) * (mailboxes - 1).factorial
  favorable_ways / total_ways

theorem probability_each_mailbox_has_at_least_one_letter :
  probability_mailbox 3 2 = 3 / 4 := by
  sorry

end probability_each_mailbox_has_at_least_one_letter_l185_185146


namespace smallest_non_factor_l185_185661

-- Definitions of the conditions
def isFactorOf (m n : ℕ) : Prop := n % m = 0
def distinct (a b : ℕ) : Prop := a ≠ b

-- The main statement we need to prove.
theorem smallest_non_factor (a b : ℕ) (h_distinct : distinct a b)
  (h_a_factor : isFactorOf a 48) (h_b_factor : isFactorOf b 48)
  (h_not_factor : ¬ isFactorOf (a * b) 48) :
  a * b = 32 := 
sorry

end smallest_non_factor_l185_185661


namespace intersection_nonempty_implies_m_eq_zero_l185_185294

theorem intersection_nonempty_implies_m_eq_zero (m : ℤ) (P Q : Set ℝ)
  (hP : P = { -1, ↑m } ) (hQ : Q = { x : ℝ | -1 < x ∧ x < 3/4 }) (h : (P ∩ Q).Nonempty) :
  m = 0 :=
by
  sorry

end intersection_nonempty_implies_m_eq_zero_l185_185294


namespace gcd_of_840_and_1764_l185_185356

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l185_185356


namespace root_equation_solution_l185_185055

-- Given conditions from the problem
def is_root_of_quadratic (m : ℝ) : Prop :=
  m^2 - m - 110 = 0

-- Statement of the proof problem
theorem root_equation_solution (m : ℝ) (h : is_root_of_quadratic m) : (m - 1)^2 + m = 111 := 
sorry

end root_equation_solution_l185_185055


namespace average_condition_l185_185165

theorem average_condition (x : ℝ) :
  (1275 + x) / 51 = 80 * x → x = 1275 / 4079 :=
by
  sorry

end average_condition_l185_185165


namespace total_num_problems_eq_30_l185_185590

-- Define the conditions
def test_points : ℕ := 100
def points_per_3_point_problem : ℕ := 3
def points_per_4_point_problem : ℕ := 4
def num_4_point_problems : ℕ := 10

-- Define the number of 3-point problems
def num_3_point_problems : ℕ :=
  (test_points - num_4_point_problems * points_per_4_point_problem) / points_per_3_point_problem

-- Prove the total number of problems is 30
theorem total_num_problems_eq_30 :
  num_3_point_problems + num_4_point_problems = 30 := 
sorry

end total_num_problems_eq_30_l185_185590


namespace largest_rectangle_area_l185_185221

theorem largest_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
by
  sorry

end largest_rectangle_area_l185_185221


namespace smallest_diameter_of_tablecloth_l185_185497

theorem smallest_diameter_of_tablecloth (a : ℝ) (h : a = 1) : ∃ d : ℝ, d = Real.sqrt 2 ∧ (∀ (x : ℝ), x < d → ¬(∀ (y : ℝ), (y^2 + y^2 = x^2) → y ≤ a)) :=
by 
  sorry

end smallest_diameter_of_tablecloth_l185_185497


namespace monotonic_increasing_intervals_max_min_values_l185_185806

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x - Real.pi / 3)

theorem monotonic_increasing_intervals (k : ℤ) :
  ∃ (a b : ℝ), a = k * Real.pi - Real.pi / 12 ∧ b = k * Real.pi + 5 * Real.pi / 12 ∧
    ∀ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → f x₁ ≤ f x₂ :=
sorry

theorem max_min_values : ∃ (xmin xmax : ℝ) (fmin fmax : ℝ),
  xmin = 0 ∧ fmin = f 0 ∧ fmin = - Real.sqrt 3 / 2 ∧
  xmax = 5 * Real.pi / 12 ∧ fmax = f (5 * Real.pi / 12) ∧ fmax = 1 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 →
    fmin ≤ f x ∧ f x ≤ fmax :=
sorry

end monotonic_increasing_intervals_max_min_values_l185_185806


namespace pills_per_day_l185_185500

theorem pills_per_day (total_days : ℕ) (prescription_days_frac : ℚ) (remaining_pills : ℕ) (days_taken : ℕ) (remaining_days : ℕ) (pills_per_day : ℕ)
  (h1 : total_days = 30)
  (h2 : prescription_days_frac = 4/5)
  (h3 : remaining_pills = 12)
  (h4 : days_taken = prescription_days_frac * total_days)
  (h5 : remaining_days = total_days - days_taken)
  (h6 : pills_per_day = remaining_pills / remaining_days) :
  pills_per_day = 2 := by
  sorry

end pills_per_day_l185_185500


namespace man_older_than_son_l185_185859

variables (S M : ℕ)

theorem man_older_than_son (h1 : S = 32) (h2 : M + 2 = 2 * (S + 2)) : M - S = 34 :=
by
  sorry

end man_older_than_son_l185_185859


namespace polynomial_identity_l185_185400

theorem polynomial_identity 
  (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  a^2 * ((x - b) * (x - c) / ((a - b) * (a - c))) +
  b^2 * ((x - c) * (x - a) / ((b - c) * (b - a))) +
  c^2 * ((x - a) * (x - b) / ((c - a) * (c - b))) = x^2 :=
by
  sorry

end polynomial_identity_l185_185400


namespace john_candies_on_fourth_day_l185_185426

theorem john_candies_on_fourth_day (c : ℕ) (h1 : 5 * c + 80 = 150) : c + 24 = 38 :=
by 
  -- Placeholder for proof
  sorry

end john_candies_on_fourth_day_l185_185426


namespace geom_seq_b_value_l185_185382

variable (r : ℝ) (b : ℝ)

-- b is the second term of the geometric sequence with first term 180 and third term 36/25
-- condition 1
def geom_sequence_cond1 := 180 * r = b
-- condition 2
def geom_sequence_cond2 := b * r = 36 / 25

-- Prove b = 16.1 given the conditions
theorem geom_seq_b_value (hb_pos : b > 0) (h1 : geom_sequence_cond1 r b) (h2 : geom_sequence_cond2 r b) : b = 16.1 :=
by sorry

end geom_seq_b_value_l185_185382


namespace linda_original_savings_l185_185525

theorem linda_original_savings :
  ∃ S : ℝ, 
    (5 / 8) * S + (1 / 4) * S = 400 ∧
    (1 / 8) * S = 600 ∧
    S = 4800 :=
by
  sorry

end linda_original_savings_l185_185525


namespace selection_count_l185_185852

def choose (n k : ℕ) : ℕ := -- Binomial coefficient definition
  if h : 0 ≤ k ∧ k ≤ n then
    Nat.choose n k
  else
    0

theorem selection_count : choose 9 5 - choose 6 5 = 120 := by
  sorry

end selection_count_l185_185852


namespace vitamin_d3_total_days_l185_185070

def vitamin_d3_days (capsules_per_bottle : ℕ) (daily_serving_size : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / daily_serving_size) * bottles_needed

theorem vitamin_d3_total_days :
  vitamin_d3_days 60 2 6 = 180 :=
by
  sorry

end vitamin_d3_total_days_l185_185070


namespace money_initial_amounts_l185_185289

theorem money_initial_amounts (x : ℕ) (A B : ℕ) 
  (h1 : A = 8 * x) 
  (h2 : B = 5 * x) 
  (h3 : (A - 50) = 4 * (B + 100) / 5) : 
  A = 800 ∧ B = 500 := 
sorry

end money_initial_amounts_l185_185289


namespace kai_marbles_over_200_l185_185526

theorem kai_marbles_over_200 (marbles_on_day : Nat → Nat)
  (h_initial : marbles_on_day 0 = 4)
  (h_growth : ∀ n, marbles_on_day (n + 1) = 3 * marbles_on_day n) :
  ∃ k, marbles_on_day k > 200 ∧ k = 4 := by
  sorry

end kai_marbles_over_200_l185_185526


namespace sum_ratios_eq_l185_185944

-- Define points A, B, C, D, E, and G as well as their relationships
variables {A B C D E G : Type}

-- Given conditions
axiom BD_2DC : ∀ {BD DC : ℝ}, BD = 2 * DC
axiom AE_3EB : ∀ {AE EB : ℝ}, AE = 3 * EB
axiom AG_2GD : ∀ {AG GD : ℝ}, AG = 2 * GD

-- Mass assumptions for the given problem
noncomputable def mC := 1
noncomputable def mB := 2
noncomputable def mD := mB + 2 * mC  -- mD = B's mass + 2*C's mass
noncomputable def mA := 1
noncomputable def mE := 3 * mA + mB  -- mE = 3A's mass + B's mass
noncomputable def mG := 2 * mA + mD  -- mG = 2A's mass + D's mass

-- Ratios defined according to the problem statement
noncomputable def ratio1 := (1 : ℝ) / mE
noncomputable def ratio2 := mD / mA
noncomputable def ratio3 := mD / mG

-- The Lean theorem to state the problem and correct answer
theorem sum_ratios_eq : ratio1 + ratio2 + ratio3 = (73 / 15 : ℝ) :=
by
  unfold ratio1 ratio2 ratio3
  sorry

end sum_ratios_eq_l185_185944


namespace part_a_proof_part_b_proof_l185_185611

-- Part (a) statement
def part_a_statement (n : ℕ) : Prop :=
  ∀ (m : ℕ), m = 9 → (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 12 ∨ n = 18)

theorem part_a_proof (n : ℕ) (m : ℕ) (h : m = 9) : part_a_statement n :=
  sorry

-- Part (b) statement
def part_b_statement (n m : ℕ) : Prop :=
  (n ≤ m) ∨ (n > m ∧ ∃ d : ℕ, d ∣ m ∧ n = m + d)

theorem part_b_proof (n m : ℕ) : part_b_statement n m :=
  sorry

end part_a_proof_part_b_proof_l185_185611


namespace correct_average_weight_is_58_6_l185_185052

noncomputable def initial_avg_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def incorrect_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 60
noncomputable def correct_avg_weight := (initial_avg_weight * num_boys + (correct_weight - incorrect_weight)) / num_boys

theorem correct_average_weight_is_58_6 :
  correct_avg_weight = 58.6 :=
sorry

end correct_average_weight_is_58_6_l185_185052


namespace outer_boundary_diameter_l185_185231

def width_jogging_path : ℝ := 10
def width_vegetable_garden : ℝ := 12
def diameter_pond : ℝ := 20

theorem outer_boundary_diameter :
  2 * (diameter_pond / 2 + width_vegetable_garden + width_jogging_path) = 64 := by
  sorry

end outer_boundary_diameter_l185_185231


namespace set_intersection_is_result_l185_185875

def set_A := {x : ℝ | 1 < x^2 ∧ x^2 < 4 }
def set_B := {x : ℝ | x ≥ 1}
def result_set := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_is_result : (set_A ∩ set_B) = result_set :=
by sorry

end set_intersection_is_result_l185_185875


namespace value_of_z_l185_185075

theorem value_of_z :
  let mean_of_4_16_20 := (4 + 16 + 20) / 3
  let mean_of_8_z := (8 + z) / 2
  ∀ z : ℚ, mean_of_4_16_20 = mean_of_8_z → z = 56 / 3 := 
by
  intro z mean_eq
  sorry

end value_of_z_l185_185075


namespace reflect_center_of_circle_l185_185967

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

theorem reflect_center_of_circle :
  reflect_point (3, -7) = (7, -3) :=
by
  sorry

end reflect_center_of_circle_l185_185967


namespace problem_system_of_equations_l185_185163

-- Define the problem as a theorem in Lean 4
theorem problem_system_of_equations (x y c d : ℝ) (h1 : 4 * x + 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 :=
by
  -- The proof is omitted
  sorry

end problem_system_of_equations_l185_185163


namespace car_speed_car_speed_correct_l185_185648

theorem car_speed (d t s : ℝ) (hd : d = 810) (ht : t = 5) : s = d / t := 
by
  sorry

theorem car_speed_correct (d t : ℝ) (hd : d = 810) (ht : t = 5) : d / t = 162 :=
by
  sorry

end car_speed_car_speed_correct_l185_185648


namespace incorrect_statement_d_l185_185908

theorem incorrect_statement_d :
  (¬(abs 2 = -2)) :=
by sorry

end incorrect_statement_d_l185_185908


namespace quadratic_vertex_transform_l185_185594

theorem quadratic_vertex_transform {p q r m k : ℝ} (h : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - h)^2 + k) →
  h = -3 :=
by
  intros h1 h2
  -- The actual proof goes here
  sorry

end quadratic_vertex_transform_l185_185594


namespace value_of_f_at_2_l185_185317

theorem value_of_f_at_2 (a b : ℝ) (h : (a + -b + 8) = (9 * a + 3 * b + 8)) :
  (a * 2 ^ 2 + b * 2 + 8) = 8 := 
by
  sorry

end value_of_f_at_2_l185_185317


namespace find_y_l185_185601

theorem find_y
  (XYZ_is_straight_line : XYZ_is_straight_line)
  (angle_XYZ : ℝ)
  (angle_YWZ : ℝ)
  (y : ℝ)
  (exterior_angle_theorem : angle_XYZ = y + angle_YWZ)
  (h1 : angle_XYZ = 150)
  (h2 : angle_YWZ = 58) :
  y = 92 :=
by
  sorry

end find_y_l185_185601


namespace proof_of_k_values_l185_185962

noncomputable def problem_statement : Prop :=
  ∀ k : ℝ,
    (∃ a b : ℝ, (6 * a^2 + 5 * a + k = 0 ∧ 6 * b^2 + 5 * b + k = 0 ∧ a ≠ b ∧
    |a - b| = 3 * (a^2 + b^2))) ↔ (k = 1 ∨ k = -20.717)

theorem proof_of_k_values : problem_statement :=
by sorry

end proof_of_k_values_l185_185962


namespace locus_of_point_P_l185_185857

/-- Given three points in the coordinate plane A(0,3), B(-√3, 0), and C(√3, 0), 
    and a point P on the coordinate plane such that PA = PB + PC, 
    determine the equation of the locus of point P. -/
noncomputable def locus_equation : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | (P.1^2 + (P.2 - 1)^2 = 4) ∧ (P.2 ≤ 0)}

theorem locus_of_point_P :
  ∀ (P : ℝ × ℝ),
  (∃ A B C : ℝ × ℝ, A = (0, 3) ∧ B = (-Real.sqrt 3, 0) ∧ C = (Real.sqrt 3, 0) ∧ 
     dist P A = dist P B + dist P C) →
  P ∈ locus_equation :=
by
  intros P hp
  sorry

end locus_of_point_P_l185_185857


namespace gcf_lcm_problem_l185_185823

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_problem :
  GCF (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end gcf_lcm_problem_l185_185823


namespace solve_for_x_l185_185516

theorem solve_for_x (x : ℚ) : (x = 70 / (8 - 3 / 4)) → (x = 280 / 29) :=
by
  intro h
  -- Proof to be provided here
  sorry

end solve_for_x_l185_185516


namespace arithmetic_sequence_properties_l185_185226

noncomputable def general_term_formula (a₁ : ℕ) (S₃ : ℕ) (n : ℕ) (d : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def sum_of_double_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (2 * (a₁ + (n - 1) * d)) * n / 2

theorem arithmetic_sequence_properties
  (a₁ : ℕ) (S₃ : ℕ)
  (h₁ : a₁ = 2)
  (h₂ : S₃ = 9) :
  general_term_formula a₁ S₃ n (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) = n + 1 ∧
  sum_of_double_sequence a₁ (a₁ + 2 * ((S₃ - 3 * a₁) / 3)) n = 2^(n+2) - 4 :=
by
  sorry

end arithmetic_sequence_properties_l185_185226


namespace committee_count_l185_185161

theorem committee_count :
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  eligible_owners.choose committee_size = 65780 := by
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  have lean_theorem : eligible_owners.choose committee_size = 65780 := sorry
  exact lean_theorem

end committee_count_l185_185161


namespace sum_of_distinct_elements_not_square_l185_185866

open Set

noncomputable def setS : Set ℕ := { n | ∃ k : ℕ, n = 2^(2*k+1) }

theorem sum_of_distinct_elements_not_square (s : Finset ℕ) (hs: ∀ x ∈ s, x ∈ setS) :
  ¬∃ k : ℕ, s.sum id = k^2 :=
sorry

end sum_of_distinct_elements_not_square_l185_185866


namespace total_boxes_l185_185088

variable (N_initial : ℕ) (N_nonempty : ℕ) (N_new_boxes : ℕ)

theorem total_boxes (h_initial : N_initial = 7) 
                     (h_nonempty : N_nonempty = 10)
                     (h_new_boxes : N_new_boxes = N_nonempty * 7) :
  N_initial + N_new_boxes = 77 :=
by 
  have : N_initial = 7 := h_initial
  have : N_new_boxes = N_nonempty * 7 := h_new_boxes
  have : N_nonempty = 10 := h_nonempty
  sorry

end total_boxes_l185_185088


namespace remainder_when_n_plus_5040_divided_by_7_l185_185911

theorem remainder_when_n_plus_5040_divided_by_7 (n : ℤ) (h: n % 7 = 2) : (n + 5040) % 7 = 2 :=
by
  sorry

end remainder_when_n_plus_5040_divided_by_7_l185_185911


namespace boxed_meals_solution_count_l185_185773

theorem boxed_meals_solution_count :
  ∃ n : ℕ, n = 4 ∧ 
  ∃ x y z : ℕ, 
      x + y + z = 22 ∧ 
      10 * x + 8 * y + 5 * z = 183 ∧ 
      x > 0 ∧ y > 0 ∧ z > 0 :=
sorry

end boxed_meals_solution_count_l185_185773


namespace sin_minus_cos_eq_one_sol_l185_185456

theorem sin_minus_cos_eq_one_sol (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) :
  x = Real.pi / 2 ∨ x = Real.pi :=
sorry

end sin_minus_cos_eq_one_sol_l185_185456


namespace probability_not_both_ends_l185_185646

theorem probability_not_both_ends :
  let total_arrangements := 120
  let both_ends_arrangements := 12
  let favorable_arrangements := total_arrangements - both_ends_arrangements
  let probability := favorable_arrangements / total_arrangements
  total_arrangements = 120 ∧ both_ends_arrangements = 12 ∧ favorable_arrangements = 108 ∧ probability = 0.9 :=
by
  sorry

end probability_not_both_ends_l185_185646


namespace angle_C_is_120_l185_185533

theorem angle_C_is_120 (C L U A : ℝ)
  (H1 : C = L)
  (H2 : L = U)
  (H3 : A = L)
  (H4 : A + L = 180)
  (H5 : 6 * C = 720) : C = 120 :=
by
  sorry

end angle_C_is_120_l185_185533


namespace min_value_of_quadratic_fun_min_value_is_reached_l185_185996

theorem min_value_of_quadratic_fun (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1) :
  (3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 ≥ (15 / 782)) :=
sorry

theorem min_value_is_reached (a b c d : ℝ)
  (h : 5 * a + 6 * b - 7 * c + 4 * d = 1)
  (h2 : 3 * a ^ 2 + 2 * b ^ 2 + 5 * c ^ 2 + d ^ 2 = (15 / 782)) :
  true :=
sorry

end min_value_of_quadratic_fun_min_value_is_reached_l185_185996


namespace length_of_diagonal_l185_185321

theorem length_of_diagonal (h1 h2 area : ℝ) (h1_val : h1 = 7) (h2_val : h2 = 3) (area_val : area = 50) :
  ∃ d : ℝ, d = 10 :=
by
  sorry

end length_of_diagonal_l185_185321


namespace find_N_l185_185005

theorem find_N (N : ℕ) (hN : N > 1) (h1 : 2019 ≡ 1743 [MOD N]) (h2 : 3008 ≡ 2019 [MOD N]) : N = 23 :=
by
  sorry

end find_N_l185_185005


namespace find_a_of_complex_eq_l185_185401

theorem find_a_of_complex_eq (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * (⟨1, -a⟩ : ℂ) = 2) : a = 1 :=
by
  sorry

end find_a_of_complex_eq_l185_185401


namespace sandy_walks_before_meet_l185_185979

/-
Sandy leaves her home and walks toward Ed's house.
Two hours later, Ed leaves his home and walks toward Sandy's house.
The distance between their homes is 52 kilometers.
Sandy's walking speed is 6 km/h.
Ed's walking speed is 4 km/h.
Prove that Sandy will walk 36 kilometers before she meets Ed.
-/

theorem sandy_walks_before_meet
    (distance_between_homes : ℕ)
    (sandy_speed ed_speed : ℕ)
    (sandy_start_time ed_start_time : ℕ)
    (time_to_meet : ℕ) :
  distance_between_homes = 52 →
  sandy_speed = 6 →
  ed_speed = 4 →
  sandy_start_time = 2 →
  ed_start_time = 0 →
  time_to_meet = 4 →
  (sandy_start_time * sandy_speed + time_to_meet * sandy_speed) = 36 := 
by
  sorry

end sandy_walks_before_meet_l185_185979


namespace unique_solution_m_l185_185342

theorem unique_solution_m :
  ∃! m : ℝ, ∀ x y : ℝ, (y = x^2 ∧ y = 4*x + m) → m = -4 :=
by 
  sorry

end unique_solution_m_l185_185342


namespace unique_real_solution_l185_185205

theorem unique_real_solution :
  ∀ (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) →
    (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) :=
by
  intro x y z w
  intros h
  have h1 : x = z + w + Real.sqrt (z * w * x) := h.1
  have h2 : y = w + x + Real.sqrt (w * x * y) := h.2.1
  have h3 : z = x + y + Real.sqrt (x * y * z) := h.2.2.1
  have h4 : w = y + z + Real.sqrt (y * z * w) := h.2.2.2
  sorry

end unique_real_solution_l185_185205


namespace jellybeans_needed_l185_185028

-- Define the initial conditions as constants
def jellybeans_per_large_glass := 50
def jellybeans_per_small_glass := jellybeans_per_large_glass / 2
def number_of_large_glasses := 5
def number_of_small_glasses := 3

-- Calculate the total number of jellybeans needed
def total_jellybeans : ℕ :=
  (number_of_large_glasses * jellybeans_per_large_glass) + 
  (number_of_small_glasses * jellybeans_per_small_glass)

-- Prove that the total number of jellybeans needed is 325
theorem jellybeans_needed : total_jellybeans = 325 :=
sorry

end jellybeans_needed_l185_185028


namespace maximal_product_at_12_l185_185436

noncomputable def geometric_sequence (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
a₁ * q^(n - 1)

noncomputable def product_first_n_terms (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
(a₁ ^ n) * (q ^ ((n - 1) * n / 2))

theorem maximal_product_at_12 :
  ∀ (a₁ : ℕ) (q : ℚ), 
  a₁ = 1536 → 
  q = -1/2 → 
  ∀ (n : ℕ), n ≠ 12 → 
  (product_first_n_terms a₁ q 12) > (product_first_n_terms a₁ q n) :=
by
  sorry

end maximal_product_at_12_l185_185436


namespace multiplier_for_second_part_l185_185197

theorem multiplier_for_second_part {x y k : ℝ} (h1 : x + y = 52) (h2 : 10 * x + k * y = 780) (hy : y = 30.333333333333332) (hx : x = 21.666666666666668) :
  k = 18.571428571428573 :=
by
  sorry

end multiplier_for_second_part_l185_185197


namespace fundraiser_full_price_revenue_l185_185041

theorem fundraiser_full_price_revenue :
  ∃ (f h p : ℕ), f + h = 200 ∧ 
                f * p + h * (p / 2) = 2700 ∧ 
                f * p = 600 :=
by 
  sorry

end fundraiser_full_price_revenue_l185_185041


namespace watermelon_vendor_profit_l185_185907

theorem watermelon_vendor_profit 
  (purchase_price : ℝ) (selling_price_initial : ℝ) (initial_quantity_sold : ℝ) 
  (decrease_factor : ℝ) (additional_quantity_per_decrease : ℝ) (fixed_cost : ℝ) 
  (desired_profit : ℝ) 
  (x : ℝ)
  (h_purchase : purchase_price = 2)
  (h_selling_initial : selling_price_initial = 3)
  (h_initial_quantity : initial_quantity_sold = 200)
  (h_decrease_factor : decrease_factor = 0.1)
  (h_additional_quantity : additional_quantity_per_decrease = 40)
  (h_fixed_cost : fixed_cost = 24)
  (h_desired_profit : desired_profit = 200) :
  (x = 2.8 ∨ x = 2.7) ↔ 
  ((x - purchase_price) * (initial_quantity_sold + additional_quantity_per_decrease / decrease_factor * (selling_price_initial - x)) - fixed_cost = desired_profit) :=
by sorry

end watermelon_vendor_profit_l185_185907


namespace set_union_intersection_l185_185246

-- Definitions
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}
def C : Set ℤ := {1, 2}

-- Theorem statement
theorem set_union_intersection : (A ∩ B ∪ C) = {0, 1, 2} :=
by
  sorry

end set_union_intersection_l185_185246


namespace length_of_overlapping_part_l185_185062

theorem length_of_overlapping_part
  (l_p : ℕ)
  (n : ℕ)
  (total_length : ℕ)
  (l_o : ℕ) :
  n = 3 →
  l_p = 217 →
  total_length = 627 →
  3 * l_p - 2 * l_o = total_length →
  l_o = 12 := by
  intros n_eq l_p_eq total_length_eq equation
  sorry

end length_of_overlapping_part_l185_185062


namespace sequence_formula_l185_185232

def seq (n : ℕ) : ℕ := 
  match n with
  | 0     => 1
  | (n+1) => 2 * seq n + 3

theorem sequence_formula (n : ℕ) (h1 : n ≥ 1) : 
  seq n = 2^n + 1 - 3 :=
sorry

end sequence_formula_l185_185232


namespace runners_meet_fractions_l185_185394

theorem runners_meet_fractions (l V₁ V₂ : ℝ)
  (h1 : l / V₂ - l / V₁ = 10)
  (h2 : 720 * V₁ - 720 * V₂ = l) :
  (1 / V₁ = 1 / 80 ∧ 1 / V₂ = 1 / 90) ∨ (1 / V₁ = 1 / 90 ∧ 1 / V₂ = 1 / 80) :=
sorry

end runners_meet_fractions_l185_185394


namespace christian_age_in_eight_years_l185_185429

theorem christian_age_in_eight_years (b c : ℕ)
  (h1 : c = 2 * b)
  (h2 : b + 8 = 40) :
  c + 8 = 72 :=
sorry

end christian_age_in_eight_years_l185_185429


namespace percentage_non_defective_l185_185124

theorem percentage_non_defective :
  let total_units : ℝ := 100
  let M1_percentage : ℝ := 0.20
  let M2_percentage : ℝ := 0.25
  let M3_percentage : ℝ := 0.30
  let M4_percentage : ℝ := 0.15
  let M5_percentage : ℝ := 0.10
  let M1_defective_percentage : ℝ := 0.02
  let M2_defective_percentage : ℝ := 0.04
  let M3_defective_percentage : ℝ := 0.05
  let M4_defective_percentage : ℝ := 0.07
  let M5_defective_percentage : ℝ := 0.08

  let M1_total := total_units * M1_percentage
  let M2_total := total_units * M2_percentage
  let M3_total := total_units * M3_percentage
  let M4_total := total_units * M4_percentage
  let M5_total := total_units * M5_percentage

  let M1_defective := M1_total * M1_defective_percentage
  let M2_defective := M2_total * M2_defective_percentage
  let M3_defective := M3_total * M3_defective_percentage
  let M4_defective := M4_total * M4_defective_percentage
  let M5_defective := M5_total * M5_defective_percentage

  let total_defective := M1_defective + M2_defective + M3_defective + M4_defective + M5_defective
  let total_non_defective := total_units - total_defective
  let percentage_non_defective := (total_non_defective / total_units) * 100

  percentage_non_defective = 95.25 := by
  sorry

end percentage_non_defective_l185_185124


namespace opposite_of_2021_l185_185192

theorem opposite_of_2021 : -(2021) = -2021 := 
sorry

end opposite_of_2021_l185_185192


namespace trigonometric_identity_l185_185211

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi + α) - Real.cos (Real.pi - α)) = 3 :=
by
  sorry

end trigonometric_identity_l185_185211


namespace minimum_w_value_l185_185983

theorem minimum_w_value : 
  (∀ x y : ℝ, w = 2*x^2 + 3*y^2 - 12*x + 9*y + 35) → 
  ∃ w_min : ℝ, w_min = 41 / 4 ∧ 
  (∀ x y : ℝ, 2*x^2 + 3*y^2 - 12*x + 9*y + 35 ≥ w_min) :=
by
  sorry

end minimum_w_value_l185_185983


namespace revenue_increase_l185_185740

open Real

theorem revenue_increase
  (P Q : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q) :
  let R := P * Q
  let P_new := P * 1.60
  let Q_new := Q * 0.65
  let R_new := P_new * Q_new
  (R_new - R) / R * 100 = 4 := by
sorry

end revenue_increase_l185_185740


namespace Mary_age_is_10_l185_185337

-- Define the parameters for the ages of Rahul and Mary
variables (Rahul Mary : ℕ)

-- Conditions provided in the problem
def condition1 := Rahul = Mary + 30
def condition2 := Rahul + 20 = 2 * (Mary + 20)

-- Stating the theorem to be proved
theorem Mary_age_is_10 (Rahul Mary : ℕ) 
  (h1 : Rahul = Mary + 30) 
  (h2 : Rahul + 20 = 2 * (Mary + 20)) : 
  Mary = 10 :=
by 
  sorry

end Mary_age_is_10_l185_185337


namespace Janet_horses_l185_185929

theorem Janet_horses (acres : ℕ) (gallons_per_acre : ℕ) (spread_acres_per_day : ℕ) (total_days : ℕ)
  (gallons_per_day_per_horse : ℕ) (total_gallons_needed : ℕ) (total_gallons_spread : ℕ) (horses : ℕ) :
  acres = 20 ->
  gallons_per_acre = 400 ->
  spread_acres_per_day = 4 ->
  total_days = 25 ->
  gallons_per_day_per_horse = 5 ->
  total_gallons_needed = acres * gallons_per_acre ->
  total_gallons_spread = spread_acres_per_day * gallons_per_acre * total_days ->
  horses = total_gallons_needed / (gallons_per_day_per_horse * total_days) ->
  horses = 64 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Janet_horses_l185_185929


namespace olivia_total_earnings_l185_185709

variable (rate : ℕ) (hours_monday : ℕ) (hours_wednesday : ℕ) (hours_friday : ℕ)

def olivia_earnings : ℕ := hours_monday * rate + hours_wednesday * rate + hours_friday * rate

theorem olivia_total_earnings :
  rate = 9 → hours_monday = 4 → hours_wednesday = 3 → hours_friday = 6 → olivia_earnings rate hours_monday hours_wednesday hours_friday = 117 :=
by
  sorry

end olivia_total_earnings_l185_185709


namespace nth_term_correct_l185_185921

noncomputable def term_in_sequence (n : ℕ) : ℚ :=
  2^n / (2^n + 3)

theorem nth_term_correct (n : ℕ) : term_in_sequence n = 2^n / (2^n + 3) :=
by
  sorry

end nth_term_correct_l185_185921


namespace ratio_of_hardback_books_is_two_to_one_l185_185549

noncomputable def ratio_of_hardback_books : ℕ :=
  let sarah_paperbacks := 6
  let sarah_hardbacks := 4
  let brother_paperbacks := sarah_paperbacks / 3
  let total_books_brother := 10
  let brother_hardbacks := total_books_brother - brother_paperbacks
  brother_hardbacks / sarah_hardbacks

theorem ratio_of_hardback_books_is_two_to_one : 
  ratio_of_hardback_books = 2 :=
by
  sorry

end ratio_of_hardback_books_is_two_to_one_l185_185549


namespace exist_pos_integers_m_n_l185_185812

def d (n : ℕ) : ℕ :=
  -- Number of divisors of n
  sorry 

theorem exist_pos_integers_m_n :
  ∃ (m n : ℕ), (m > 0) ∧ (n > 0) ∧ (m = 24) ∧ 
  ((∃ (triples : Finset (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ triples ↔ (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c ≤ m) ∧ (d (n + a) * d (n + b) * d (n + c)) % (a * b * c) = 0) ∧ 
    (triples.card = 2024))) :=
sorry

end exist_pos_integers_m_n_l185_185812


namespace find_x_l185_185240

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 156) (h2 : x ≥ 0) : x = 12 :=
sorry

end find_x_l185_185240


namespace even_function_sum_eval_l185_185281

variable (v : ℝ → ℝ)

theorem even_function_sum_eval (h_even : ∀ x : ℝ, v x = v (-x)) :
    v (-2.33) + v (-0.81) + v (0.81) + v (2.33) = 2 * (v 2.33 + v 0.81) :=
by
  sorry

end even_function_sum_eval_l185_185281


namespace product_of_two_numbers_l185_185370

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
by
  -- proof goes here
  sorry

end product_of_two_numbers_l185_185370


namespace number_of_pairs_l185_185094

theorem number_of_pairs (n : ℕ) (h : n = 2835) :
  ∃ (count : ℕ), count = 20 ∧
  (∀ (x y : ℕ), (0 < x ∧ 0 < y ∧ x < y ∧ (x^2 + y^2) % (x + y) = 0 ∧ (x^2 + y^2) / (x + y) ∣ n) → count = 20) := 
sorry

end number_of_pairs_l185_185094


namespace compute_xy_l185_185286

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := 
sorry

end compute_xy_l185_185286


namespace work_duration_l185_185947

theorem work_duration (X_full_days : ℕ) (Y_full_days : ℕ) (Y_worked_days : ℕ) (R : ℚ) :
  X_full_days = 18 ∧ Y_full_days = 15 ∧ Y_worked_days = 5 ∧ R = (2 / 3) →
  (R / (1 / X_full_days)) = 12 :=
by
  intros h
  sorry

end work_duration_l185_185947


namespace distance_home_to_school_l185_185560

-- Define the variables and conditions
variables (D T : ℝ)
def boy_travel_5km_hr_late := 5 * (T + 5 / 60) = D
def boy_travel_10km_hr_early := 10 * (T - 10 / 60) = D

-- State the theorem to prove
theorem distance_home_to_school 
    (H1 : boy_travel_5km_hr_late D T) 
    (H2 : boy_travel_10km_hr_early D T) : 
  D = 2.5 :=
by
  sorry

end distance_home_to_school_l185_185560


namespace number_of_friends_l185_185615

def total_gold := 100
def lost_gold := 20
def gold_per_friend := 20

theorem number_of_friends :
  (total_gold - lost_gold) / gold_per_friend = 4 := by
  sorry

end number_of_friends_l185_185615


namespace expression_evaluation_l185_185939

theorem expression_evaluation :
  (-2: ℤ)^3 + ((36: ℚ) / (3: ℚ)^2 * (-1 / 2: ℚ)) + abs (-5: ℤ) = -5 :=
by
  sorry

end expression_evaluation_l185_185939


namespace calculate_expression_l185_185849

theorem calculate_expression : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 :=
by
  sorry

end calculate_expression_l185_185849


namespace solve_for_x_l185_185609

theorem solve_for_x (x : ℝ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  sorry

end solve_for_x_l185_185609


namespace doctor_lawyer_ratio_l185_185898

variables {d l : ℕ} -- Number of doctors and lawyers

-- Conditions
def avg_age_group (d l : ℕ) : Prop := (40 * d + 55 * l) / (d + l) = 45

-- Theorem: Given the conditions, the ratio of doctors to lawyers is 2:1.
theorem doctor_lawyer_ratio (hdl : avg_age_group d l) : d / l = 2 :=
sorry

end doctor_lawyer_ratio_l185_185898


namespace Douglas_won_in_county_Y_l185_185408

def total_percentage (x y t r : ℝ) : Prop :=
  (0.74 * 2 + y * 1 = 0.66 * (2 + 1))

theorem Douglas_won_in_county_Y :
  ∀ (x y t r : ℝ), x = 0.74 → t = 0.66 → r = 2 →
  total_percentage x y t r → y = 0.50 := 
by
  intros x y t r hx ht hr H
  rw [hx, hr, ht] at H
  sorry

end Douglas_won_in_county_Y_l185_185408


namespace sum_quotient_dividend_divisor_l185_185034

theorem sum_quotient_dividend_divisor (n : ℕ) (d : ℕ) (h : n = 45) (h1 : d = 3) : 
  (n / d) + n + d = 63 :=
by
  sorry

end sum_quotient_dividend_divisor_l185_185034


namespace min_digits_fraction_l185_185511

theorem min_digits_fraction : 
  let num := 987654321
  let denom := 2^27 * 5^3
  ∃ (digits : ℕ), (10^digits * num = 987654321 * 2^27 * 5^3) ∧ digits = 27 := 
by
  sorry

end min_digits_fraction_l185_185511


namespace no_solution_ineq_l185_185153

theorem no_solution_ineq (m : ℝ) : 
  (∀ x : ℝ, x - m ≥ 0 → ¬(0.5 * x + 0.5 < 2)) → m ≥ 3 :=
by
  sorry

end no_solution_ineq_l185_185153


namespace correct_operation_l185_185131

theorem correct_operation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = - (a^2 * b) :=
by
  sorry

end correct_operation_l185_185131


namespace system_solution_fraction_l185_185527

theorem system_solution_fraction (x y z : ℝ) (h1 : x + (-95/9) * y + 4 * z = 0)
  (h2 : 4 * x + (-95/9) * y - 3 * z = 0) (h3 : 3 * x + 5 * y - 4 * z = 0) (hx_ne_zero : x ≠ 0) 
  (hy_ne_zero : y ≠ 0) (hz_ne_zero : z ≠ 0) : 
  (x * z) / (y ^ 2) = 20 :=
sorry

end system_solution_fraction_l185_185527


namespace hire_charges_paid_by_B_l185_185793

theorem hire_charges_paid_by_B (total_cost : ℝ) (hours_A hours_B hours_C : ℝ) (b_payment : ℝ) :
  total_cost = 720 ∧ hours_A = 9 ∧ hours_B = 10 ∧ hours_C = 13 ∧ b_payment = (total_cost / (hours_A + hours_B + hours_C)) * hours_B → b_payment = 225 :=
by
  sorry

end hire_charges_paid_by_B_l185_185793


namespace birdseed_mix_percentage_l185_185349

theorem birdseed_mix_percentage (x : ℝ) :
  (0.40 * x + 0.65 * (100 - x) = 50) → x = 60 :=
by
  sorry

end birdseed_mix_percentage_l185_185349


namespace stratified_sampling_l185_185222

-- Define the known quantities
def total_products := 2000
def sample_size := 200
def workshop_production := 250

-- Define the main theorem to prove
theorem stratified_sampling:
  (workshop_production / total_products) * sample_size = 25 := by
  sorry

end stratified_sampling_l185_185222


namespace benny_has_24_books_l185_185177

def books_sandy : ℕ := 10
def books_tim : ℕ := 33
def total_books : ℕ := 67

def books_benny : ℕ := total_books - (books_sandy + books_tim)

theorem benny_has_24_books : books_benny = 24 := by
  unfold books_benny
  unfold total_books
  unfold books_sandy
  unfold books_tim
  sorry

end benny_has_24_books_l185_185177


namespace arithmetic_sequence_15th_term_l185_185711

theorem arithmetic_sequence_15th_term : 
  let a₁ := 3
  let d := 4
  let n := 15
  a₁ + (n - 1) * d = 59 :=
by
  let a₁ := 3
  let d := 4
  let n := 15
  sorry

end arithmetic_sequence_15th_term_l185_185711


namespace number_of_cherries_l185_185890

-- Definitions for the problem conditions
def total_fruits : ℕ := 580
def raspberries (b : ℕ) : ℕ := 2 * b
def grapes (c : ℕ) : ℕ := 3 * c
def cherries (r : ℕ) : ℕ := 3 * r

-- Theorem to prove the number of cherries
theorem number_of_cherries (b r g c : ℕ) 
  (H1 : b + r + g + c = total_fruits)
  (H2 : r = raspberries b)
  (H3 : g = grapes c)
  (H4 : c = cherries r) :
  c = 129 :=
by sorry

end number_of_cherries_l185_185890


namespace ball_hits_ground_time_l185_185405

theorem ball_hits_ground_time (t : ℚ) :
  (-4.9 * (t : ℝ)^2 + 5 * (t : ℝ) + 10 = 0) → t = 10 / 7 :=
sorry

end ball_hits_ground_time_l185_185405


namespace x_pow_y_equals_nine_l185_185710

theorem x_pow_y_equals_nine (x y : ℝ) (h : (|x + 3| * (y - 2)^2 < 0)) : x^y = 9 :=
sorry

end x_pow_y_equals_nine_l185_185710


namespace new_point_in_fourth_quadrant_l185_185656

-- Define the initial point P with coordinates (-3, 2)
def P : ℝ × ℝ := (-3, 2)

-- Define the move operation: 4 units to the right and 6 units down
def move (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 4, p.2 - 6)

-- Define the new point after the move operation
def P' : ℝ × ℝ := move P

-- Prove that the new point P' is in the fourth quadrant
theorem new_point_in_fourth_quadrant (x y : ℝ) (h : P' = (x, y)) : x > 0 ∧ y < 0 :=
by
  sorry

end new_point_in_fourth_quadrant_l185_185656


namespace product_of_solutions_t_squared_eq_49_l185_185556

theorem product_of_solutions_t_squared_eq_49 (t : ℝ) (h1 : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_solutions_t_squared_eq_49_l185_185556


namespace tan_product_l185_185658

open Real

theorem tan_product (x y : ℝ) 
(h1 : sin x * sin y = 24 / 65) 
(h2 : cos x * cos y = 48 / 65) :
tan x * tan y = 1 / 2 :=
by
  sorry

end tan_product_l185_185658


namespace walking_east_of_neg_west_l185_185280

-- Define the representation of directions
def is_walking_west (d : ℕ) (x : ℤ) : Prop := x = d
def is_walking_east (d : ℕ) (x : ℤ) : Prop := x = -d

-- Given the condition and states the relationship is the proposition to prove.
theorem walking_east_of_neg_west (d : ℕ) (x : ℤ) (h : is_walking_west 2 2) : is_walking_east 5 (-5) :=
by
  sorry

end walking_east_of_neg_west_l185_185280


namespace tom_profit_l185_185749

-- Define the initial conditions
def initial_investment : ℕ := 20 * 3
def revenue_from_selling : ℕ := 10 * 4
def value_of_remaining_shares : ℕ := 10 * 6
def total_amount : ℕ := revenue_from_selling + value_of_remaining_shares

-- We claim that the profit Tom makes is 40 dollars
theorem tom_profit : (total_amount - initial_investment) = 40 := by
  sorry

end tom_profit_l185_185749


namespace g_domain_l185_185306

noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arcsin (x ^ 3))

theorem g_domain : {x : ℝ | -1 < x ∧ x < 1} = Set {x | ∃ y, g x = y} :=
by
  sorry

end g_domain_l185_185306


namespace discriminant_of_quad_eq_l185_185078

def a : ℕ := 5
def b : ℕ := 8
def c : ℤ := -6

def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

theorem discriminant_of_quad_eq : discriminant 5 8 (-6) = 184 :=
by
  -- The proof is skipped
  sorry

end discriminant_of_quad_eq_l185_185078


namespace lego_set_cost_l185_185781

-- Definitions and conditions
def price_per_car := 5
def cars_sold := 3
def action_figures_sold := 2
def total_earnings := 120

-- Derived prices
def price_per_action_figure := 2 * price_per_car
def price_per_board_game := price_per_action_figure + price_per_car

-- Total cost of sold items (cars, action figures, and board game)
def total_cost_of_sold_items := 
  (cars_sold * price_per_car) + 
  (action_figures_sold * price_per_action_figure) + 
  price_per_board_game

-- Cost of Lego set
theorem lego_set_cost : 
  total_earnings - total_cost_of_sold_items = 70 :=
by
  -- Proof omitted
  sorry

end lego_set_cost_l185_185781


namespace extra_mangoes_l185_185909

-- Definitions of the conditions
def original_price_per_mango := 433.33 / 130
def new_price_per_mango := original_price_per_mango - 0.10 * original_price_per_mango
def mangoes_at_original_price := 360 / original_price_per_mango
def mangoes_at_new_price := 360 / new_price_per_mango

-- Statement to be proved
theorem extra_mangoes : mangoes_at_new_price - mangoes_at_original_price = 12 := 
by {
  sorry
}

end extra_mangoes_l185_185909


namespace woman_needs_butter_l185_185258

noncomputable def butter_needed (cost_package : ℝ) (cost_8oz : ℝ) (cost_4oz : ℝ) 
                                (discount : ℝ) (lowest_price : ℝ) : ℝ :=
  if lowest_price = cost_8oz + 2 * (cost_4oz * discount / 100) then 8 + 2 * 4 else 0

theorem woman_needs_butter 
  (cost_single_package : ℝ := 7) 
  (cost_8oz_package : ℝ := 4) 
  (cost_4oz_package : ℝ := 2)
  (discount_4oz_package : ℝ := 50) 
  (lowest_price_payment : ℝ := 6) :
  butter_needed cost_single_package cost_8oz_package cost_4oz_package discount_4oz_package lowest_price_payment = 16 := 
by
  sorry

end woman_needs_butter_l185_185258


namespace amount_C_l185_185539

-- Define the variables and conditions.
variables (A B C : ℝ)
axiom h1 : A = (2 / 3) * B
axiom h2 : B = (1 / 4) * C
axiom h3 : A + B + C = 544

-- State the theorem.
theorem amount_C (A B C : ℝ) (h1 : A = (2 / 3) * B) (h2 : B = (1 / 4) * C) (h3 : A + B + C = 544) : C = 384 := 
sorry

end amount_C_l185_185539


namespace problem_solution_exists_l185_185235

theorem problem_solution_exists {x : ℝ} :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 =
    a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 +
    a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
    a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7)
  → a_2 = 56 :=
sorry

end problem_solution_exists_l185_185235


namespace part1_part2_l185_185464

noncomputable def a (n : ℕ) : ℤ :=
  15 * n + 2 + (15 * n - 32) * 16^(n-1)

theorem part1 (n : ℕ) : 15^3 ∣ (a n) := by
  sorry

-- Correct answer for part (2) bundled in a formal statement:
theorem part2 (n k : ℕ) : 1991 ∣ (a n) ∧ 1991 ∣ (a (n + 1)) ∧
    1991 ∣ (a (n + 2)) ↔ n = 89595 * k := by
  sorry

end part1_part2_l185_185464


namespace y_percent_of_x_l185_185189

theorem y_percent_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.20 * (x + y)) : y / x = 0.5 :=
sorry

end y_percent_of_x_l185_185189


namespace train_length_correct_l185_185345

noncomputable def length_of_train (train_speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * cross_time_s
  total_distance - bridge_length_m

theorem train_length_correct :
  length_of_train 45 30 205 = 170 :=
by
  sorry

end train_length_correct_l185_185345


namespace log_base_change_l185_185599

theorem log_base_change (a b : ℝ) (h₁ : Real.log 5 / Real.log 3 = a) (h₂ : Real.log 7 / Real.log 3 = b) :
    Real.log 35 / Real.log 15 = (a + b) / (1 + a) :=
by
  sorry

end log_base_change_l185_185599


namespace value_of_k_parallel_vectors_l185_185377

theorem value_of_k_parallel_vectors :
  (a : ℝ × ℝ) → (b : ℝ × ℝ) → (k : ℝ) →
  a = (2, 1) → b = (-1, k) → 
  (a.1 * b.2 - a.2 * b.1 = 0) →
  k = -(1/2) :=
by
  intros a b k ha hb hab_det
  sorry

end value_of_k_parallel_vectors_l185_185377


namespace secret_known_on_monday_l185_185419

def students_know_secret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem secret_known_on_monday :
  ∃ n : ℕ, students_know_secret n = 3280 ∧ (n + 1) % 7 = 0 :=
by
  sorry

end secret_known_on_monday_l185_185419


namespace root_in_interval_l185_185425

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 2 / x

variable (h_monotonic : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
variable (h_f_half : f (1 / 2) < 0)
variable (h_f_one : f 1 < 0)
variable (h_f_three_half : f (3 / 2) < 0)
variable (h_f_two : f 2 > 0)

theorem root_in_interval : ∃ c : ℝ, c ∈ Set.Ioo (3 / 2) 2 ∧ f c = 0 :=
sorry

end root_in_interval_l185_185425


namespace pow_mod_eq_l185_185838

theorem pow_mod_eq : (17 ^ 2001) % 23 = 11 := 
by {
  sorry
}

end pow_mod_eq_l185_185838


namespace initial_value_divisible_by_456_l185_185114

def initial_value := 374
def to_add := 82
def divisor := 456

theorem initial_value_divisible_by_456 : (initial_value + to_add) % divisor = 0 := by
  sorry

end initial_value_divisible_by_456_l185_185114


namespace quadratic_inequality_false_iff_l185_185672

theorem quadratic_inequality_false_iff (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 - 3 * a * x + 9 < 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
by sorry

end quadratic_inequality_false_iff_l185_185672


namespace evaluate_expression_l185_185891

theorem evaluate_expression : 68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by
  sorry

end evaluate_expression_l185_185891


namespace water_remaining_45_days_l185_185440

-- Define the initial conditions and the evaporation rate
def initial_volume : ℕ := 400
def evaporation_rate : ℕ := 1
def days : ℕ := 45

-- Define a function to compute the remaining water volume
def remaining_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_volume - (evaporation_rate * days)

-- Theorem stating that the water remaining after 45 days is 355 gallons
theorem water_remaining_45_days : remaining_volume 400 1 45 = 355 :=
by
  -- proof goes here
  sorry

end water_remaining_45_days_l185_185440


namespace problem_statement_l185_185025

-- Define y as the sum of the given terms
def y : ℤ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

-- The theorem to prove that y is a multiple of 8, 16, 32, and 64
theorem problem_statement : 
  (8 ∣ y) ∧ (16 ∣ y) ∧ (32 ∣ y) ∧ (64 ∣ y) :=
by sorry

end problem_statement_l185_185025


namespace find_white_towels_l185_185690

variable (W : ℕ) -- Let W be the number of white towels Maria bought

def green_towels : ℕ := 40
def towels_given : ℕ := 65
def towels_left : ℕ := 19

theorem find_white_towels :
  green_towels + W - towels_given = towels_left →
  W = 44 :=
by
  intro h
  sorry

end find_white_towels_l185_185690


namespace distinct_digits_and_difference_is_945_l185_185768

theorem distinct_digits_and_difference_is_945 (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_difference : 10 * (100 * a + 10 * b + c) + 2 - (2000 + 100 * a + 10 * b + c) = 945) :
  (100 * a + 10 * b + c) = 327 :=
by
  sorry

end distinct_digits_and_difference_is_945_l185_185768


namespace min_value_inverse_sum_l185_185223

variable {x y : ℝ}

theorem min_value_inverse_sum (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : (1/x + 1/y) ≥ 1 :=
  sorry

end min_value_inverse_sum_l185_185223


namespace fraction_juniors_study_Japanese_l185_185986

-- Define the size of the junior and senior classes
variable (J S : ℕ)

-- Condition 1: The senior class is twice the size of the junior class
axiom senior_twice_junior : S = 2 * J

-- The fraction of the seniors studying Japanese
noncomputable def fraction_seniors_study_Japanese : ℚ := 3 / 8

-- The total fraction of students in both classes that study Japanese
noncomputable def fraction_total_study_Japanese : ℚ := 1 / 3

-- Define the unknown fraction of juniors studying Japanese
variable (x : ℚ)

-- The proof problem transformed from the questions and the correct answer
theorem fraction_juniors_study_Japanese :
  (fraction_seniors_study_Japanese * ↑S + x * ↑J = fraction_total_study_Japanese * (↑J + ↑S)) → (x = 1 / 4) :=
by
  -- We use the given conditions and solve for x
  sorry

end fraction_juniors_study_Japanese_l185_185986


namespace area_triangle_QDA_l185_185234

-- Define the points
def Q : ℝ × ℝ := (0, 15)
def A (q : ℝ) : ℝ × ℝ := (q, 15)
def D (p : ℝ) : ℝ × ℝ := (0, p)

-- Define the conditions
variable (q : ℝ) (p : ℝ)
variable (hq : q > 0) (hp : p < 15)

-- Theorem stating the area of the triangle QDA in terms of q and p
theorem area_triangle_QDA : 
  1 / 2 * q * (15 - p) = 1 / 2 * q * (15 - p) :=
by sorry

end area_triangle_QDA_l185_185234


namespace kyle_car_payment_l185_185457

theorem kyle_car_payment (income rent utilities retirement groceries insurance miscellaneous gas x : ℕ)
  (h_income : income = 3200)
  (h_rent : rent = 1250)
  (h_utilities : utilities = 150)
  (h_retirement : retirement = 400)
  (h_groceries : groceries = 300)
  (h_insurance : insurance = 200)
  (h_miscellaneous : miscellaneous = 200)
  (h_gas : gas = 350)
  (h_expenses : rent + utilities + retirement + groceries + insurance + miscellaneous + gas + x = income) :
  x = 350 :=
by sorry

end kyle_car_payment_l185_185457


namespace integer_add_results_in_perfect_square_l185_185674

theorem integer_add_results_in_perfect_square (x a b : ℤ) :
  (x + 100 = a^2 ∧ x + 164 = b^2) → (x = 125 ∨ x = -64 ∨ x = -100) :=
by
  intros h
  sorry

end integer_add_results_in_perfect_square_l185_185674


namespace sum_of_transformed_numbers_l185_185992

theorem sum_of_transformed_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := 
by
  sorry

end sum_of_transformed_numbers_l185_185992


namespace golden_section_MP_length_l185_185379

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem golden_section_MP_length (MN : ℝ) (hMN : MN = 2) (P : ℝ) 
  (hP : P > 0 ∧ P < MN ∧ P / (MN - P) = (MN - P) / P)
  (hMP_NP : MN - P < P) :
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_MP_length_l185_185379


namespace Betty_flies_caught_in_morning_l185_185792

-- Definitions from the conditions
def total_flies_needed_in_a_week : ℕ := 14
def flies_eaten_per_day : ℕ := 2
def days_in_a_week : ℕ := 7
def flies_caught_in_morning (X : ℕ) : ℕ := X
def flies_caught_in_afternoon : ℕ := 6
def flies_escaped : ℕ := 1
def flies_short : ℕ := 4

-- Given statement in Lean 4
theorem Betty_flies_caught_in_morning (X : ℕ) 
  (h1 : flies_caught_in_morning X + flies_caught_in_afternoon - flies_escaped = total_flies_needed_in_a_week - flies_short) : 
  X = 5 :=
by
  sorry

end Betty_flies_caught_in_morning_l185_185792


namespace compound_interest_rate_l185_185745

theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (CI r : ℝ)
  (hP : P = 1200)
  (hCI : CI = 1785.98)
  (ht : t = 5)
  (hn : n = 1)
  (hA : A = P * (1 + r/n)^(n * t)) :
  A = P + CI → 
  r = 0.204 :=
by
  sorry

end compound_interest_rate_l185_185745


namespace explicit_expression_for_f_l185_185522

variable (f : ℕ → ℕ)

-- Define the condition
axiom h : ∀ x : ℕ, f (x + 1) = 3 * x + 2

-- State the theorem
theorem explicit_expression_for_f (x : ℕ) : f x = 3 * x - 1 :=
by {
  sorry
}

end explicit_expression_for_f_l185_185522


namespace percent_of_sales_not_pens_pencils_erasers_l185_185487

theorem percent_of_sales_not_pens_pencils_erasers :
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  percent_total - (percent_pens + percent_pencils + percent_erasers) = 25 :=
by
  -- definitions and assumptions
  let percent_pens := 25
  let percent_pencils := 30
  let percent_erasers := 20
  let percent_total := 100
  sorry

end percent_of_sales_not_pens_pencils_erasers_l185_185487


namespace complement_A_in_U_l185_185166

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_A_in_U : (U \ A) = {3, 9} := 
by sorry

end complement_A_in_U_l185_185166


namespace find_values_of_pqr_l185_185551

def A (p : ℝ) := {x : ℝ | x^2 + p * x - 2 = 0}
def B (q r : ℝ) := {x : ℝ | x^2 + q * x + r = 0}
def A_union_B (p q r : ℝ) := A p ∪ B q r = {-2, 1, 5}
def A_intersect_B (p q r : ℝ) := A p ∩ B q r = {-2}

theorem find_values_of_pqr (p q r : ℝ) :
  A_union_B p q r → A_intersect_B p q r → p = -1 ∧ q = -3 ∧ r = -10 :=
by
  sorry

end find_values_of_pqr_l185_185551


namespace optimal_purchase_interval_discount_advantage_l185_185093

/- The functions and assumptions used here. -/
def purchase_feed_days (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) : ℕ :=
-- Implementation omitted
sorry

def should_use_discount (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) : Prop :=
-- Implementation omitted
sorry

/- Conditions -/
def conditions : Prop :=
  let feed_per_day := 200
  let price_per_kg := 1.8
  let storage_cost_per_kg_per_day := 0.03
  let transportation_fee := 300
  let discount_threshold := 5000 -- in kg, since 5 tons = 5000 kg
  let discount_rate := 0.85
  True -- We apply these values in the proofs below.

/- Main statements -/
theorem optimal_purchase_interval : conditions → 
  purchase_feed_days 200 1.8 0.03 300 = 10 :=
by
  intros
  -- Proof is omitted.
  sorry

theorem discount_advantage : conditions →
  should_use_discount 200 1.8 0.03 300 5000 0.85 :=
by
  intros
  -- Proof is omitted.
  sorry

end optimal_purchase_interval_discount_advantage_l185_185093


namespace how_many_bones_in_adult_woman_l185_185201

-- Define the conditions
def numSkeletons : ℕ := 20
def halfSkeletons : ℕ := 10
def numAdultWomen : ℕ := 10
def numMenAndChildren : ℕ := 10
def numAdultMen : ℕ := 5
def numChildren : ℕ := 5
def totalBones : ℕ := 375

-- Define the proof statement
theorem how_many_bones_in_adult_woman (W : ℕ) (H : 10 * W + 5 * (W + 5) + 5 * (W / 2) = 375) : W = 20 :=
sorry

end how_many_bones_in_adult_woman_l185_185201


namespace additional_charge_per_minute_atlantic_call_l185_185476

def base_rate_U : ℝ := 11.0
def rate_per_minute_U : ℝ := 0.25
def base_rate_A : ℝ := 12.0
def call_duration : ℝ := 20.0
variable (rate_per_minute_A : ℝ)

theorem additional_charge_per_minute_atlantic_call :
  base_rate_U + rate_per_minute_U * call_duration = base_rate_A + rate_per_minute_A * call_duration →
  rate_per_minute_A = 0.20 := by
  sorry

end additional_charge_per_minute_atlantic_call_l185_185476


namespace circle_properties_radius_properties_l185_185046

theorem circle_properties (m x y : ℝ) :
  (x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) ↔
    (-((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :=
sorry

theorem radius_properties (m : ℝ) (h : -((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :
  ∃ r : ℝ, (0 < r ∧ r ≤ (4 / Real.sqrt 7)) :=
sorry

end circle_properties_radius_properties_l185_185046


namespace rick_division_steps_l185_185061

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end rick_division_steps_l185_185061


namespace polynomial_identity_l185_185026

theorem polynomial_identity (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 15) 
  (h3 : a^3 + b^3 + c^3 = 47) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) = 625 := 
by 
  sorry

end polynomial_identity_l185_185026


namespace A_alone_finishes_work_in_30_days_l185_185328

noncomputable def work_rate_A (B : ℝ) : ℝ := 2 * B

noncomputable def total_work (B : ℝ) : ℝ := 60 * B

theorem A_alone_finishes_work_in_30_days (B : ℝ) : (total_work B) / (work_rate_A B) = 30 := by
  sorry

end A_alone_finishes_work_in_30_days_l185_185328


namespace speed_of_third_part_l185_185276

theorem speed_of_third_part (d : ℝ) (v : ℝ)
  (h1 : 3 * d = 3.000000000000001)
  (h2 : d / 3 + d / 4 + d / v = 47/60) :
  v = 5 := by
  sorry

end speed_of_third_part_l185_185276


namespace Rockets_won_38_games_l185_185017

-- Definitions for each team and their respective wins
variables (Sharks Dolphins Rockets Wolves Comets : ℕ)
variables (wins : Finset ℕ)
variables (shArks_won_more_than_Dolphins : Sharks > Dolphins)
variables (rockets_won_more_than_Wolves : Rockets > Wolves)
variables (rockets_won_fewer_than_Comets : Rockets < Comets)
variables (Wolves_won_more_than_25_games : Wolves > 25)
variables (possible_wins : wins = {28, 33, 38, 43})

-- Statement that the Rockets won 38 games given the conditions
theorem Rockets_won_38_games
  (shArks_won_more_than_Dolphins : Sharks > Dolphins)
  (rockets_won_more_than_Wolves : Rockets > Wolves)
  (rockets_won_fewer_than_Comets : Rockets < Comets)
  (Wolves_won_more_than_25_games : Wolves > 25)
  (possible_wins : wins = {28, 33, 38, 43}) :
  Rockets = 38 :=
sorry

end Rockets_won_38_games_l185_185017


namespace triangle_angles_median_bisector_altitude_l185_185925

theorem triangle_angles_median_bisector_altitude {α β γ : ℝ} 
  (h : α + β + γ = 180) 
  (median_angle_condition : α / 4 + β / 4 + γ / 4 = 45) -- Derived from 90/4 = 22.5
  (median_from_C : 4 * α = γ) -- Given condition that angle is divided into 4 equal parts
  (median_angle_C : γ = 90) -- Derived that angle @ C must be right angle (90°)
  (sum_angles_C : α + β = 90) : 
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end triangle_angles_median_bisector_altitude_l185_185925


namespace inequality_proof_l185_185635

variable {x₁ x₂ x₃ x₄ : ℝ}

theorem inequality_proof
  (h₁ : x₁ ≥ x₂) (h₂ : x₂ ≥ x₃) (h₃ : x₃ ≥ x₄) (h₄ : x₄ ≥ 2)
  (h₅ : x₂ + x₃ + x₄ ≥ x₁) 
  : (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := 
by {
  sorry
}

end inequality_proof_l185_185635


namespace min_value_condition_l185_185974

theorem min_value_condition {a b c d e f g h : ℝ} (h1 : a * b * c * d = 16) (h2 : e * f * g * h = 25) :
  (a^2 * e^2 + b^2 * f^2 + c^2 * g^2 + d^2 * h^2) ≥ 160 :=
  sorry

end min_value_condition_l185_185974


namespace f_2015_equals_2_l185_185247

noncomputable def f : ℝ → ℝ :=
sorry

theorem f_2015_equals_2 (f_even : ∀ x : ℝ, f (-x) = f x)
    (f_shift : ∀ x : ℝ, f (-x) = f (2 + x))
    (f_log : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = Real.log (3 * x + 1) / Real.log 2) :
    f 2015 = 2 :=
sorry

end f_2015_equals_2_l185_185247


namespace evaluate_fraction_l185_185420

theorem evaluate_fraction : (3 : ℚ) / (2 - (3 / 4)) = (12 / 5) := 
by
  sorry

end evaluate_fraction_l185_185420


namespace neg_P_4_of_P_implication_and_neg_P_5_l185_185993

variable (P : ℕ → Prop)

theorem neg_P_4_of_P_implication_and_neg_P_5
  (h1 : ∀ k : ℕ, 0 < k → (P k → P (k+1)))
  (h2 : ¬ P 5) :
  ¬ P 4 :=
by
  sorry

end neg_P_4_of_P_implication_and_neg_P_5_l185_185993


namespace quadratic_inequality_l185_185604

theorem quadratic_inequality (m y1 y2 y3 : ℝ)
  (h1 : m < -2)
  (h2 : y1 = (m-1)^2 - 2*(m-1))
  (h3 : y2 = m^2 - 2*m)
  (h4 : y3 = (m+1)^2 - 2*(m+1)) :
  y3 < y2 ∧ y2 < y1 :=
by
  sorry

end quadratic_inequality_l185_185604


namespace bud_age_uncle_age_relation_l185_185627

variable (bud_age uncle_age : Nat)

theorem bud_age_uncle_age_relation (h : bud_age = 8) (h0 : bud_age = uncle_age / 3) : uncle_age = 24 := by
  sorry

end bud_age_uncle_age_relation_l185_185627


namespace complex_i_power_l185_185623

theorem complex_i_power (i : ℂ) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : i^2015 = -i := 
by
  sorry

end complex_i_power_l185_185623


namespace absolute_value_inequality_solution_l185_185063

theorem absolute_value_inequality_solution (x : ℝ) :
  |x - 2| + |x - 4| ≤ 3 ↔ (3 / 2 ≤ x ∧ x < 4) :=
by
  sorry

end absolute_value_inequality_solution_l185_185063


namespace cost_of_each_item_l185_185137

theorem cost_of_each_item (initial_order items : ℕ) (price per_item_reduction additional_orders : ℕ) (reduced_order total_order reduced_price profit_per_item : ℕ) 
  (h1 : initial_order = 60)
  (h2 : price = 100)
  (h3 : per_item_reduction = 1)
  (h4 : additional_orders = 3)
  (h5 : reduced_price = price - price * 4 / 100)
  (h6 : total_order = initial_order + additional_orders * (price * 4 / 100))
  (h7 : reduced_order = total_order)
  (h8 : profit_per_item = price - per_item_reduction )
  (h9 : profit_per_item = 24)
  (h10 : items * profit_per_item = reduced_order * (profit_per_item - per_item_reduction)) :
  (price - profit_per_item = 76) :=
by sorry

end cost_of_each_item_l185_185137


namespace painted_cube_faces_l185_185966

theorem painted_cube_faces (a : ℕ) (h : 2 < a) :
  ∃ (one_face two_faces three_faces : ℕ),
  (one_face = 6 * (a - 2) ^ 2) ∧
  (two_faces = 12 * (a - 2)) ∧
  (three_faces = 8) := by
  sorry

end painted_cube_faces_l185_185966


namespace transform_negation_l185_185387

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end transform_negation_l185_185387


namespace calculate_total_calories_l185_185216

-- Definition of variables and conditions
def total_calories (C : ℝ) : Prop :=
  let FDA_recommended_intake := 25
  let consumed_calories := FDA_recommended_intake + 5
  (3 / 4) * C = consumed_calories

-- Theorem statement
theorem calculate_total_calories : ∃ C : ℝ, total_calories C ∧ C = 40 :=
by
  sorry  -- Proof will be provided here

end calculate_total_calories_l185_185216


namespace max_value_of_polynomial_l185_185780

theorem max_value_of_polynomial :
  ∃ x : ℝ, (x = -1) ∧ ∀ y : ℝ, -3 * y^2 - 6 * y + 12 ≤ -3 * (-1)^2 - 6 * (-1) + 12 := by
  sorry

end max_value_of_polynomial_l185_185780


namespace max_a_b_c_d_l185_185220

theorem max_a_b_c_d (a c d b : ℤ) (hb : b > 0) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) 
: a + b + c + d = -5 :=
by
  sorry

end max_a_b_c_d_l185_185220


namespace sequence_infinite_divisibility_l185_185195

theorem sequence_infinite_divisibility :
  ∃ (u : ℕ → ℤ), (∀ n, u (n + 2) = u (n + 1) ^ 2 - u n) ∧ u 1 = 39 ∧ u 2 = 45 ∧ (∀ N, ∃ k ≥ N, 1986 ∣ u k) := 
by
  sorry

end sequence_infinite_divisibility_l185_185195


namespace fraction_computation_l185_185990

noncomputable def compute_fraction : ℚ :=
  (64^4 + 324) * (52^4 + 324) * (40^4 + 324) * (28^4 + 324) * (16^4 + 324) /
  (58^4 + 324) * (46^4 + 324) * (34^4 + 324) * (22^4 + 324) * (10^4 + 324)

theorem fraction_computation :
  compute_fraction = 137 / 1513 :=
by sorry

end fraction_computation_l185_185990


namespace lcm_60_30_40_eq_120_l185_185105

theorem lcm_60_30_40_eq_120 : (Nat.lcm (Nat.lcm 60 30) 40) = 120 := 
sorry

end lcm_60_30_40_eq_120_l185_185105


namespace kitty_cleaning_time_l185_185388

def weekly_cleaning_time (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust

def total_cleaning_time (weeks: ℕ) (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  weeks * weekly_cleaning_time pick_up vacuum clean_windows dust

theorem kitty_cleaning_time :
  total_cleaning_time 4 5 20 15 10 = 200 := by
  sorry

end kitty_cleaning_time_l185_185388


namespace compound_interest_years_is_four_l185_185718
noncomputable def compoundInterestYears (P : ℝ) (r : ℝ) (n : ℕ) (CI : ℝ) : ℕ :=
  let A := P + CI
  let factor := (1 + r / n)
  let log_A_P := Real.log (A / P)
  let log_factor := Real.log factor
  Nat.floor (log_A_P / log_factor)

theorem compound_interest_years_is_four :
  compoundInterestYears 1200 0.20 1 1288.32 = 4 :=
by
  sorry

end compound_interest_years_is_four_l185_185718


namespace num_fish_when_discovered_l185_185112

open Nat

/-- Definition of the conditions given in the problem --/
def initial_fish := 60
def fish_per_day_eaten := 2
def additional_fish := 8
def weeks_before_addition := 2
def extra_week := 1

/-- The proof problem statement --/
theorem num_fish_when_discovered : 
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  final_fish = 26 := 
by
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  have h : final_fish = 26 := sorry
  exact h

end num_fish_when_discovered_l185_185112


namespace average_first_set_eq_3_more_than_second_set_l185_185004

theorem average_first_set_eq_3_more_than_second_set (x : ℤ) :
  let avg_first_set := (14 + 32 + 53) / 3
  let avg_second_set := (x + 47 + 22) / 3
  avg_first_set = avg_second_set + 3 → x = 21 := by
  sorry

end average_first_set_eq_3_more_than_second_set_l185_185004


namespace actual_time_when_watch_reads_11_pm_is_correct_l185_185935

-- Define the conditions
def noon := 0 -- Time when Cassandra sets her watch to the correct time
def actual_time_2_pm := 120 -- 2:00 PM in minutes
def watch_time_2_pm := 113.2 -- 1:53 PM and 12 seconds in minutes (113 minutes + 0.2 minutes)

-- Define the goal
def actual_time_watch_reads_11_pm := 731.25 -- 12:22 PM and 15 seconds in minutes from noon

-- Provide the theorem statement without proof
theorem actual_time_when_watch_reads_11_pm_is_correct :
  actual_time_watch_reads_11_pm = 731.25 :=
sorry

end actual_time_when_watch_reads_11_pm_is_correct_l185_185935


namespace convert_fraction_to_decimal_l185_185329

theorem convert_fraction_to_decimal : (3 / 40 : ℝ) = 0.075 := 
by
  sorry

end convert_fraction_to_decimal_l185_185329


namespace meeting_time_when_speeds_doubled_l185_185466

noncomputable def meeting_time (x y z : ℝ) : ℝ :=
  2 * 91

theorem meeting_time_when_speeds_doubled
  (x y z : ℝ)
  (h1 : 2 * z * (x + y) = (2 * z - 56) * (2 * x + y))
  (h2 : 2 * z * (x + y) = (2 * z - 65) * (x + 2 * y))
  : meeting_time x y z = 182 := 
sorry

end meeting_time_when_speeds_doubled_l185_185466


namespace bill_face_value_l185_185565

theorem bill_face_value
  (TD : ℝ) (T : ℝ) (r : ℝ) (FV : ℝ)
  (h1 : TD = 210)
  (h2 : T = 0.75)
  (h3 : r = 0.16) :
  FV = 1960 :=
by 
  sorry

end bill_face_value_l185_185565


namespace speed_conversion_l185_185068

-- Define the conversion factor
def conversion_factor := 3.6

-- Define the given speed in meters per second
def speed_mps := 16.668

-- Define the expected speed in kilometers per hour
def expected_speed_kmph := 60.0048

-- The theorem to prove that the given speed in m/s converts to the expected speed in km/h
theorem speed_conversion : speed_mps * conversion_factor = expected_speed_kmph := 
  by
    sorry

end speed_conversion_l185_185068


namespace simplify_expression_l185_185677

theorem simplify_expression : |(-4 : Int)^2 - (3 : Int)^2 + 2| = 9 := by
  sorry

end simplify_expression_l185_185677


namespace find_minuend_l185_185818

variable (x y : ℕ)

-- Conditions
axiom h1 : x - y = 8008
axiom h2 : x - 10 * y = 88

-- Theorem statement
theorem find_minuend : x = 8888 :=
by
  sorry

end find_minuend_l185_185818


namespace remainder_of_m_l185_185418

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end remainder_of_m_l185_185418


namespace num_pos_int_values_l185_185432

theorem num_pos_int_values
  (N : ℕ) 
  (h₀ : 0 < N)
  (h₁ : ∃ (k : ℕ), 0 < k ∧ 48 = k * (N + 3)) :
  ∃ (n : ℕ), n = 7 :=
sorry

end num_pos_int_values_l185_185432


namespace max_students_can_be_equally_distributed_l185_185353

def num_pens : ℕ := 2730
def num_pencils : ℕ := 1890

theorem max_students_can_be_equally_distributed : Nat.gcd num_pens num_pencils = 210 := by
  sorry

end max_students_can_be_equally_distributed_l185_185353


namespace no_real_solutions_eqn_l185_185196

theorem no_real_solutions_eqn : ∀ x : ℝ, (2 * x - 4 * x + 7)^2 + 1 ≠ -|x^2 - 1| :=
by
  intro x
  sorry

end no_real_solutions_eqn_l185_185196


namespace possible_birches_l185_185118

theorem possible_birches (N B L : ℕ) (hN : N = 130) (h_sum : B + L = 130)
  (h_linden_false : ∀ l, l < L → (∀ b, b < B → b + l < N → b < B → False))
  (h_birch_false : ∃ b, b < B ∧ (∀ l, l < L → l + b < N → l + b = 2 * B))
  : B = 87 :=
sorry

end possible_birches_l185_185118


namespace zoo_charge_for_child_l185_185736

theorem zoo_charge_for_child (charge_adult : ℕ) (total_people total_bill children : ℕ) (charge_child : ℕ) : 
  charge_adult = 8 → total_people = 201 → total_bill = 964 → children = 161 → 
  total_bill - (total_people - children) * charge_adult = children * charge_child → 
  charge_child = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end zoo_charge_for_child_l185_185736


namespace ending_number_of_range_l185_185869

/-- The sum of the first n consecutive odd integers is n^2. -/
def sum_first_n_odd : ℕ → ℕ 
| 0       => 0
| (n + 1) => (2 * n + 1) + sum_first_n_odd n

/-- The sum of all odd integers between 11 and the ending number is 416. -/
def sum_odd_integers (a b : ℕ) : ℕ :=
  let s := (1 + b) / 2 - (1 + a) / 2 + 1
  sum_first_n_odd s

theorem ending_number_of_range (n : ℕ) (h1 : sum_first_n_odd n = n^2) 
  (h2 : sum_odd_integers 11 n = 416) : 
  n = 67 :=
sorry

end ending_number_of_range_l185_185869


namespace question1_1_question1_2_question2_l185_185142

open Set

noncomputable def universal_set : Set ℝ := univ

def setA : Set ℝ := { x | x^2 - 9 * x + 18 ≥ 0 }

def setB : Set ℝ := { x | -2 < x ∧ x < 9 }

def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem question1_1 : ∀ x, x ∈ setA ∨ x ∈ setB :=
by sorry

theorem question1_2 : ∀ x, x ∈ (universal_set \ setA) ∩ setB ↔ (3 < x ∧ x < 6) :=
by sorry

theorem question2 (a : ℝ) (h : setC a ⊆ setB) : -2 ≤ a ∧ a ≤ 8 :=
by sorry

end question1_1_question1_2_question2_l185_185142


namespace root_value_algebraic_expression_l185_185678

theorem root_value_algebraic_expression {a : ℝ} (h : a^2 + 3 * a + 2 = 0) : a^2 + 3 * a = -2 :=
by
  sorry

end root_value_algebraic_expression_l185_185678


namespace equation_has_two_distinct_real_roots_l185_185409

open Real

theorem equation_has_two_distinct_real_roots (m : ℝ) :
  (∃ (x1 x2 : ℝ), 0 < x1 ∧ x1 < 16 ∧ 0 < x2 ∧ x2 < 16 ∧ x1 ≠ x2 ∧ exp (m * x1) = x1^2 ∧ exp (m * x2) = x2^2) ↔
  (log 2 / 2 < m ∧ m < 2 / exp 1) :=
by sorry

end equation_has_two_distinct_real_roots_l185_185409


namespace rectangular_field_area_l185_185427

noncomputable def length (c : ℚ) : ℚ := 3 * c / 2
noncomputable def width (c : ℚ) : ℚ := 4 * c / 2
noncomputable def area (c : ℚ) : ℚ := (length c) * (width c)
noncomputable def field_area (c1 : ℚ) (c2 : ℚ) : ℚ :=
  let l := length c1
  let w := width c1
  if 25 * c2 = 101.5 * 100 then
    area c1
  else
    0

theorem rectangular_field_area :
  ∃ (c : ℚ), field_area c 25 = 10092 := by
  sorry

end rectangular_field_area_l185_185427


namespace no_distinct_triple_exists_for_any_quadratic_trinomial_l185_185707

theorem no_distinct_triple_exists_for_any_quadratic_trinomial (f : ℝ → ℝ) 
    (hf : ∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) :
    ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = b ∧ f b = c ∧ f c = a := 
by 
  sorry

end no_distinct_triple_exists_for_any_quadratic_trinomial_l185_185707


namespace ants_in_park_l185_185713

theorem ants_in_park:
  let width_meters := 100
  let length_meters := 130
  let cm_per_meter := 100
  let ants_per_sq_cm := 1.2
  let width_cm := width_meters * cm_per_meter
  let length_cm := length_meters * cm_per_meter
  let area_sq_cm := width_cm * length_cm
  let total_ants := ants_per_sq_cm * area_sq_cm
  total_ants = 156000000 := by
  sorry

end ants_in_park_l185_185713


namespace calculate_binary_expr_l185_185512

theorem calculate_binary_expr :
  let a := 0b11001010
  let b := 0b11010
  let c := 0b100
  (a * b) / c = 0b1001110100 := by
sorry

end calculate_binary_expr_l185_185512


namespace jenny_cases_l185_185260

theorem jenny_cases (total_boxes cases_per_box : ℕ) (h1 : total_boxes = 24) (h2 : cases_per_box = 8) :
  total_boxes / cases_per_box = 3 := by
  sorry

end jenny_cases_l185_185260


namespace taxi_fare_for_100_miles_l185_185472

theorem taxi_fare_for_100_miles
  (base_fare : ℝ := 10)
  (proportional_fare : ℝ := 140 / 80)
  (fare_for_80_miles : ℝ := 150)
  (distance_80 : ℝ := 80)
  (distance_100 : ℝ := 100) :
  let additional_fare := proportional_fare * distance_100
  let total_fare_for_100_miles := base_fare + additional_fare
  total_fare_for_100_miles = 185 :=
by
  sorry

end taxi_fare_for_100_miles_l185_185472


namespace range_of_a_l185_185698

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) : (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 := by
  intro h
  sorry

end range_of_a_l185_185698


namespace min_value_when_a_is_negative_one_max_value_bounds_l185_185108

-- Conditions
def f (a x : ℝ) : ℝ := a * x^2 + x
def a1 : ℝ := -1
def a : ℝ := -2
def a_lower_bound : ℝ := -2
def a_upper_bound : ℝ := 0
def interval : Set ℝ := Set.Icc 0 2

-- Part I: Minimum value when a = -1
theorem min_value_when_a_is_negative_one : 
  ∃ x ∈ interval, f a1 x = -2 := 
by
  sorry

-- Part II: Maximum value criterions
theorem max_value_bounds (a : ℝ) (H : a ∈ Set.Icc a_lower_bound a_upper_bound) :
  (∀ x ∈ interval, 
    (a ≥ -1/4 → f a ( -1 / (2 * a) ) = -1 / (4 * a)) 
    ∧ (a < -1/4 → f a 2 = 4 * a + 2 )) :=
by
  sorry

end min_value_when_a_is_negative_one_max_value_bounds_l185_185108


namespace necessary_but_not_sufficient_condition_l185_185012

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x ^ 2

theorem necessary_but_not_sufficient_condition :
  (∀ x, q x → p x) ∧ (¬ ∀ x, p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_condition_l185_185012


namespace exists_positive_ℓ_l185_185729

theorem exists_positive_ℓ (k : ℕ) (h_prime: 0 < k) :
  ∃ ℓ : ℕ, 0 < ℓ ∧ 
  (∀ m n : ℕ, m > 0 → n > 0 → Nat.gcd m ℓ = 1 → Nat.gcd n ℓ = 1 →  m ^ m % ℓ = n ^ n % ℓ → m % k = n % k) :=
sorry

end exists_positive_ℓ_l185_185729


namespace coaching_fee_correct_l185_185885

noncomputable def total_coaching_fee : ℝ :=
  let daily_fee : ℝ := 39
  let discount_threshold : ℝ := 50
  let discount_rate : ℝ := 0.10
  let total_days : ℝ := 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 3 -- non-leap year days count up to Nov 3
  let discount_days : ℝ := total_days - discount_threshold
  let discounted_fee : ℝ := daily_fee * (1 - discount_rate)
  let fee_before_discount : ℝ := discount_threshold * daily_fee
  let fee_after_discount : ℝ := discount_days * discounted_fee
  fee_before_discount + fee_after_discount

theorem coaching_fee_correct :
  total_coaching_fee = 10967.7 := by
  sorry

end coaching_fee_correct_l185_185885


namespace gift_wrapping_combinations_l185_185767

theorem gift_wrapping_combinations :
  (10 * 5 * 6 * 2 = 600) :=
by
  sorry

end gift_wrapping_combinations_l185_185767


namespace eggs_remaining_l185_185095

-- Assign the given constants
def hens : ℕ := 3
def eggs_per_hen_per_day : ℕ := 3
def days_gone : ℕ := 7
def eggs_taken_by_neighbor : ℕ := 12
def eggs_dropped_by_myrtle : ℕ := 5

-- Calculate the expected number of eggs Myrtle should have
noncomputable def total_eggs :=
  hens * eggs_per_hen_per_day * days_gone - eggs_taken_by_neighbor - eggs_dropped_by_myrtle

-- Prove that the total number of eggs equals the correct answer
theorem eggs_remaining : total_eggs = 46 :=
by
  sorry

end eggs_remaining_l185_185095


namespace bookshop_shipment_correct_l185_185117

noncomputable def bookshop_shipment : ℕ :=
  let Initial_books := 743
  let Saturday_instore_sales := 37
  let Saturday_online_sales := 128
  let Sunday_instore_sales := 2 * Saturday_instore_sales
  let Sunday_online_sales := Saturday_online_sales + 34
  let books_sold := Saturday_instore_sales + Saturday_online_sales + Sunday_instore_sales + Sunday_online_sales
  let Final_books := 502
  Final_books - (Initial_books - books_sold)

theorem bookshop_shipment_correct : bookshop_shipment = 160 := by
  sorry

end bookshop_shipment_correct_l185_185117


namespace geom_seq_val_l185_185534

noncomputable def is_geom_seq (a : ℕ → ℝ) : Prop :=
∃ q b, ∀ n, a n = b * q^n

variables (a : ℕ → ℝ)

axiom a_5_a_7 : a 5 * a 7 = 2
axiom a_2_plus_a_10 : a 2 + a 10 = 3

theorem geom_seq_val (a_geom : is_geom_seq a) :
  (a 12) / (a 4) = 2 ∨ (a 12) / (a 4) = 1 / 2 :=
sorry

end geom_seq_val_l185_185534


namespace ellipse_equation_and_line_intersection_unique_l185_185882

-- Definitions from conditions
def ellipse (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1
def line (x0 y0 x y : ℝ) : Prop := 3*x0*x + 4*y0*y - 12 = 0
def on_ellipse (x0 y0 : ℝ) : Prop := ellipse x0 y0

theorem ellipse_equation_and_line_intersection_unique :
  ∀ (x0 y0 : ℝ), on_ellipse x0 y0 → ∀ (x y : ℝ), line x0 y0 x y → ellipse x y → x = x0 ∧ y = y0 :=
by
  sorry

end ellipse_equation_and_line_intersection_unique_l185_185882


namespace calc_factorial_sum_l185_185050

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end calc_factorial_sum_l185_185050


namespace max_value_f_on_interval_l185_185720

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (x^2 - 4) * (x - a)

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  3 * x^2 - 2 * a * x - 4

theorem max_value_f_on_interval :
  f' (-1) (1 / 2) = 0 →
  ∃ max_f, max_f = 42 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x (1 / 2) ≤ max_f :=
by
  sorry

end max_value_f_on_interval_l185_185720


namespace lemon_ratio_l185_185147

variable (Levi Jayden Eli Ian : ℕ)

theorem lemon_ratio (h1: Levi = 5)
    (h2: Jayden = Levi + 6)
    (h3: Jayden = Eli / 3)
    (h4: Levi + Jayden + Eli + Ian = 115) :
    Eli = Ian / 2 :=
by
  sorry

end lemon_ratio_l185_185147


namespace alice_bob_meet_l185_185775

/--
Alice and Bob play a game on a circle divided into 18 equally-spaced points.
Alice moves 7 points clockwise per turn, and Bob moves 13 points counterclockwise.
Prove that they will meet at the same point after 9 turns.
-/
theorem alice_bob_meet : ∃ k : ℕ, k = 9 ∧ (7 * k) % 18 = (18 - 13 * k) % 18 :=
by
  sorry

end alice_bob_meet_l185_185775


namespace sum_of_integers_l185_185051

theorem sum_of_integers (a b : ℕ) (h1 : a * a + b * b = 585) (h2 : Nat.gcd a b + Nat.lcm a b = 87) : a + b = 33 := 
sorry

end sum_of_integers_l185_185051


namespace simplify_T_l185_185368

noncomputable def T (x : ℝ) : ℝ :=
  (x+1)^4 - 4*(x+1)^3 + 6*(x+1)^2 - 4*(x+1) + 1

theorem simplify_T (x : ℝ) : T x = x^4 :=
  sorry

end simplify_T_l185_185368


namespace roots_product_l185_185887

theorem roots_product {a b : ℝ} (h1 : a^2 - a - 2 = 0) (h2 : b^2 - b - 2 = 0) 
(roots : a ≠ b ∧ ∀ x, x^2 - x - 2 = 0 ↔ (x = a ∨ x = b)) : (a - 1) * (b - 1) = -2 := by
  -- proof
  sorry

end roots_product_l185_185887


namespace tate_total_years_l185_185224

-- Define the conditions
def high_school_years : Nat := 3
def gap_years : Nat := 2
def bachelor_years : Nat := 2 * high_school_years
def certification_years : Nat := 1
def work_experience_years : Nat := 1
def master_years : Nat := bachelor_years / 2
def phd_years : Nat := 3 * (high_school_years + bachelor_years + master_years)

-- Define the total years Tate spent
def total_years : Nat :=
  high_school_years + gap_years +
  bachelor_years + certification_years +
  work_experience_years + master_years + phd_years

-- State the theorem
theorem tate_total_years : total_years = 52 := by
  sorry

end tate_total_years_l185_185224


namespace fewest_trips_l185_185603

theorem fewest_trips (total_objects : ℕ) (capacity : ℕ) (h_objects : total_objects = 17) (h_capacity : capacity = 3) : 
  (total_objects + capacity - 1) / capacity = 6 :=
by
  sorry

end fewest_trips_l185_185603


namespace complete_square_eq_l185_185924

theorem complete_square_eq (x : ℝ) : (x^2 - 6 * x - 5 = 0) -> (x - 3)^2 = 14 :=
by
  intro h
  sorry

end complete_square_eq_l185_185924


namespace atomic_weight_S_is_correct_l185_185167

-- Conditions
def molecular_weight_BaSO4 : Real := 233
def atomic_weight_Ba : Real := 137.33
def atomic_weight_O : Real := 16
def num_O_in_BaSO4 : Nat := 4

-- Definition of total weight of Ba and O
def total_weight_Ba_O := atomic_weight_Ba + num_O_in_BaSO4 * atomic_weight_O

-- Expected atomic weight of S
def atomic_weight_S : Real := molecular_weight_BaSO4 - total_weight_Ba_O

-- Theorem to prove that the atomic weight of S is 31.67
theorem atomic_weight_S_is_correct : atomic_weight_S = 31.67 := by
  -- placeholder for the proof
  sorry

end atomic_weight_S_is_correct_l185_185167


namespace onions_total_l185_185325

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ)
  (h1 : Sara_onions = 4) (h2 : Sally_onions = 5) (h3 : Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 := by
  sorry

end onions_total_l185_185325


namespace fixed_point_of_f_l185_185958

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.logb a (|x + 1|)

theorem fixed_point_of_f (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 0 = 1 :=
by
  sorry

end fixed_point_of_f_l185_185958


namespace eq_a_sub_b_l185_185708

theorem eq_a_sub_b (a b : ℝ) (i : ℂ) (hi : i * i = -1) (h1 : (a + 4 * i) * i = b + i) : a - b = 5 :=
by
  have := hi
  have := h1
  sorry

end eq_a_sub_b_l185_185708


namespace robin_gum_count_l185_185733

theorem robin_gum_count (initial_gum : ℝ) (additional_gum : ℝ) (final_gum : ℝ) 
  (h1 : initial_gum = 18.0) (h2 : additional_gum = 44.0) : final_gum = 62.0 :=
by {
  sorry
}

end robin_gum_count_l185_185733


namespace settle_debt_using_coins_l185_185651

theorem settle_debt_using_coins :
  ∃ n m : ℕ, 49 * n - 99 * m = 1 :=
sorry

end settle_debt_using_coins_l185_185651


namespace each_serving_requires_1_5_apples_l185_185724

theorem each_serving_requires_1_5_apples 
  (guest_count : ℕ) (pie_count : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℝ) 
  (h_guest_count : guest_count = 12)
  (h_pie_count : pie_count = 3)
  (h_servings_per_pie : servings_per_pie = 8)
  (h_apples_per_guest : apples_per_guest = 3) :
  (apples_per_guest * guest_count) / (pie_count * servings_per_pie) = 1.5 :=
by
  sorry

end each_serving_requires_1_5_apples_l185_185724


namespace tan_45_eq_1_l185_185706

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l185_185706


namespace identify_incorrect_propositions_l185_185830

-- Definitions for parallel lines and planes
def line := Type -- Define a line type
def plane := Type -- Define a plane type
def parallel_to (l1 l2 : line) : Prop := sorry -- Assume a definition for parallel lines
def parallel_to_plane (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line parallel to a plane
def contained_in (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line contained in a plane

theorem identify_incorrect_propositions (a b : line) (α : plane) :
  (parallel_to_plane a α ∧ parallel_to_plane b α → ¬parallel_to a b) ∧
  (parallel_to_plane a α ∧ contained_in b α → ¬parallel_to a b) ∧
  (parallel_to a b ∧ contained_in b α → ¬parallel_to_plane a α) ∧
  (parallel_to a b ∧ parallel_to_plane b α → ¬parallel_to_plane a α) :=
by
  sorry -- The proof is not required

end identify_incorrect_propositions_l185_185830


namespace hyperbola_symmetric_asymptotes_l185_185502

noncomputable def M : ℝ := 225 / 16

theorem hyperbola_symmetric_asymptotes (M_val : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = x * (4 / 3) ∨ y = -x * (4 / 3))
  ∧ (y^2 / 25 - x^2 / M_val = 1 → y = x * (5 / Real.sqrt M_val) ∨ y = -x * (5 / Real.sqrt M_val)))
  → M_val = M := by
  sorry

end hyperbola_symmetric_asymptotes_l185_185502


namespace problem_statement_l185_185092

variable {x y : ℝ}

theorem problem_statement 
  (h1 : y > x)
  (h2 : x > 0)
  (h3 : x + y = 1) :
  x < 2 * x * y ∧ 2 * x * y < (x + y) / 2 ∧ (x + y) / 2 < y := by
  sorry

end problem_statement_l185_185092


namespace largest_number_l185_185660

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = 0) (h3 : c = 1) (h4 : d = -9) :
  max (max a b) (max c d) = c :=
by
  sorry

end largest_number_l185_185660


namespace large_square_area_l185_185386

theorem large_square_area (a b c : ℕ) (h1 : 4 * a < b) (h2 : c^2 = a^2 + b^2 + 10) : c^2 = 36 :=
  sorry

end large_square_area_l185_185386


namespace min_value_f_l185_185183

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (15 - 12 * cos x) + 
  sqrt (4 - 2 * sqrt 3 * sin x) +
  sqrt (7 - 4 * sqrt 3 * sin x) +
  sqrt (10 - 4 * sqrt 3 * sin x - 6 * cos x)

theorem min_value_f : ∃ x : ℝ, f x = 6 := 
sorry

end min_value_f_l185_185183


namespace sum_of_first_90_terms_l185_185899

def arithmetic_progression_sum (n : ℕ) (a d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_90_terms (a d : ℚ) :
  (arithmetic_progression_sum 15 a d = 150) →
  (arithmetic_progression_sum 75 a d = 75) →
  (arithmetic_progression_sum 90 a d = -112.5) :=
by
  sorry

end sum_of_first_90_terms_l185_185899


namespace shekar_math_marks_l185_185735

variable (science socialStudies english biology average : ℕ)

theorem shekar_math_marks 
  (h1 : science = 65)
  (h2 : socialStudies = 82)
  (h3 : english = 67)
  (h4 : biology = 95)
  (h5 : average = 77) :
  ∃ M, average = (science + socialStudies + english + biology + M) / 5 ∧ M = 76 :=
by
  sorry

end shekar_math_marks_l185_185735


namespace sales_tax_difference_l185_185327

def item_price : ℝ := 20
def sales_tax_rate1 : ℝ := 0.065
def sales_tax_rate2 : ℝ := 0.06

theorem sales_tax_difference :
  (item_price * sales_tax_rate1) - (item_price * sales_tax_rate2) = 0.1 := 
by
  sorry

end sales_tax_difference_l185_185327


namespace range_of_sum_of_zeros_l185_185762

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x else 1 - x / 2

noncomputable def F (x : ℝ) (m : ℝ) : ℝ :=
  f (f x + 1) + m

def has_zeros (F : ℝ → ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ F x₁ m = 0 ∧ F x₂ m = 0

theorem range_of_sum_of_zeros (m : ℝ) :
  has_zeros F m →
  ∃ (x₁ x₂ : ℝ), F x₁ m = 0 ∧ F x₂ m = 0 ∧ (x₁ + x₂) ≥ 4 - 2 * Real.log 2 := sorry

end range_of_sum_of_zeros_l185_185762


namespace find_side_length_l185_185015

theorem find_side_length (a : ℝ) (b : ℝ) (A B : ℝ) (ha : a = 4) (hA : A = 45) (hB : B = 60) :
    b = 2 * Real.sqrt 6 := by
  sorry

end find_side_length_l185_185015


namespace sufficient_not_necessary_condition_l185_185079

variables (a b c : ℝ)

theorem sufficient_not_necessary_condition (h1 : c < b) (h2 : b < a) :
  (ac < 0 → ab > ac) ∧ (ab > ac → ac < 0) → false :=
sorry

end sufficient_not_necessary_condition_l185_185079


namespace problem_solution_l185_185850

theorem problem_solution (a b : ℤ) (h1 : 6 * b + 4 * a = -50) (h2 : a * b = -84) : a + 2 * b = -17 := 
  sorry

end problem_solution_l185_185850


namespace find_x_base_l185_185255

open Nat

def is_valid_digit (n : ℕ) : Prop := n < 10

def interpret_base (digits : ℕ → ℕ) (n : ℕ) : ℕ :=
  digits 2 * n^2 + digits 1 * n + digits 0

theorem find_x_base (a b c : ℕ)
  (ha : is_valid_digit a)
  (hb : is_valid_digit b)
  (hc : is_valid_digit c)
  (h : interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 20 = 2 * interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 13) :
  100 * a + 10 * b + c = 198 :=
by
  sorry

end find_x_base_l185_185255


namespace polynomial_distinct_positive_roots_l185_185150

theorem polynomial_distinct_positive_roots (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^3 + a * x^2 + b * x - 1) 
(hroots : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ P x1 = 0 ∧ P x2 = 0 ∧ P x3 = 0) : 
  P (-1) < -8 := 
by
  sorry

end polynomial_distinct_positive_roots_l185_185150


namespace smallest_odd_digit_n_l185_185508

theorem smallest_odd_digit_n {n : ℕ} (h : n > 1) : 
  (∀ d ∈ (Nat.digits 10 (9997 * n)), d % 2 = 1) → n = 3335 :=
sorry

end smallest_odd_digit_n_l185_185508


namespace scientific_notation_86400_l185_185302

theorem scientific_notation_86400 : 86400 = 8.64 * 10^4 :=
by
  sorry

end scientific_notation_86400_l185_185302


namespace proof_l185_185376

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x ≥ 1
def q : Prop := ∀ x : ℝ, 0 < x → Real.exp x > Real.log x

-- The theorem statement
theorem proof : p ∧ q := by sorry

end proof_l185_185376


namespace keegan_total_school_time_l185_185120

-- Definition of the conditions
def keegan_classes : Nat := 7
def history_and_chemistry_time : ℝ := 1.5
def other_class_time : ℝ := 1.2

-- The theorem stating that given these conditions, Keegan spends 7.5 hours a day in school.
theorem keegan_total_school_time : 
  (history_and_chemistry_time + 5 * other_class_time) = 7.5 := 
by
  sorry

end keegan_total_school_time_l185_185120


namespace tangent_line_to_circle_l185_185406

open Real

theorem tangent_line_to_circle (x y : ℝ) :
  ((x - 2) ^ 2 + (y + 1) ^ 2 = 9) ∧ ((x = -1) → (x = -1 ∧ y = 3) ∨ (y = (37 - 8*x) / 15)) :=
by {
  sorry
}

end tangent_line_to_circle_l185_185406


namespace sum_of_values_of_n_l185_185764

theorem sum_of_values_of_n (n : ℚ) (h : |3 * n - 4| = 6) : 
  (n = 10 / 3 ∨ n = -2 / 3) → (10 / 3 + -2 / 3 = 8 / 3) :=
sorry

end sum_of_values_of_n_l185_185764


namespace total_carriages_in_towns_l185_185981

noncomputable def total_carriages (euston norfolk norwich flyingScotsman victoria waterloo : ℕ) : ℕ :=
  euston + norfolk + norwich + flyingScotsman + victoria + waterloo

theorem total_carriages_in_towns :
  let euston := 130
  let norfolk := euston - (20 * euston / 100)
  let norwich := 100
  let flyingScotsman := 3 * norwich / 2
  let victoria := euston - (15 * euston / 100)
  let waterloo := 2 * norwich
  total_carriages euston norfolk norwich flyingScotsman victoria waterloo = 794 :=
by
  sorry

end total_carriages_in_towns_l185_185981


namespace range_of_x_when_a_is_1_range_of_a_for_necessity_l185_185855

-- Define the statements p and q based on the conditions
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- (1) Prove the range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_1 {x : ℝ} (h1 : ∀ x, p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- (2) Prove the range of a for p to be necessary but not sufficient for q
theorem range_of_a_for_necessity : ∀ a, (∀ x, p x a → q x) → (1 ≤ a ∧ a ≤ 2) :=
  sorry

end range_of_x_when_a_is_1_range_of_a_for_necessity_l185_185855


namespace jogging_distance_apart_l185_185194

theorem jogging_distance_apart
  (alice_speed : ℝ)
  (bob_speed : ℝ)
  (time_in_minutes : ℝ)
  (distance_apart : ℝ)
  (h1 : alice_speed = 1 / 12)
  (h2 : bob_speed = 3 / 40)
  (h3 : time_in_minutes = 120)
  (h4 : distance_apart = alice_speed * time_in_minutes + bob_speed * time_in_minutes) :
  distance_apart = 19 := by
  sorry

end jogging_distance_apart_l185_185194


namespace percentage_games_won_l185_185774

theorem percentage_games_won 
  (P_first : ℝ)
  (P_remaining : ℝ)
  (total_games : ℕ)
  (H1 : P_first = 0.7)
  (H2 : P_remaining = 0.5)
  (H3 : total_games = 100) :
  True :=
by
  -- To prove the percentage of games won is 70%
  have percentage_won : ℝ := P_first
  have : percentage_won * 100 = 70 := by sorry
  trivial

end percentage_games_won_l185_185774


namespace quotient_of_even_and_odd_composites_l185_185747

theorem quotient_of_even_and_odd_composites:
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = 512 / 28525 := by
sorry

end quotient_of_even_and_odd_composites_l185_185747


namespace original_cost_of_article_l185_185950

theorem original_cost_of_article : ∃ C : ℝ, 
  (∀ S : ℝ, S = 1.35 * C) ∧
  (∀ C_new : ℝ, C_new = 0.75 * C) ∧
  (∀ S_new : ℝ, (S_new = 1.35 * C - 25) ∧ (S_new = 1.0875 * C)) ∧
  (C = 95.24) :=
sorry

end original_cost_of_article_l185_185950


namespace find_functions_l185_185964

variable (f : ℝ → ℝ)

theorem find_functions (h : ∀ x y : ℝ, f (x + f y) = f x + f y ^ 2 + 2 * x * f y) :
  ∃ c : ℝ, (∀ x, f x = x ^ 2 + c) ∨ (∀ x, f x = 0) :=
by
  sorry

end find_functions_l185_185964


namespace maximize_container_volume_l185_185031

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → (90 - 2*y) * (48 - 2*y) * y ≤ (90 - 2*x) * (48 - 2*x) * x) ∧ x = 10 :=
sorry

end maximize_container_volume_l185_185031


namespace expected_digits_fair_icosahedral_die_l185_185110

noncomputable def expected_number_of_digits : ℝ :=
  let one_digit_count := 9
  let two_digit_count := 11
  let total_faces := 20
  let prob_one_digit := one_digit_count / total_faces
  let prob_two_digit := two_digit_count / total_faces
  (prob_one_digit * 1) + (prob_two_digit * 2)

theorem expected_digits_fair_icosahedral_die :
  expected_number_of_digits = 1.55 :=
by
  sorry

end expected_digits_fair_icosahedral_die_l185_185110


namespace fifth_bowler_points_l185_185283

variable (P1 P2 P3 P4 P5 : ℝ)
variable (h1 : P1 = (5 / 12) * P3)
variable (h2 : P2 = (5 / 3) * P3)
variable (h3 : P4 = (5 / 3) * P3)
variable (h4 : P5 = (50 / 27) * P3)
variable (h5 : P3 ≤ 500)
variable (total_points : P1 + P2 + P3 + P4 + P5 = 2000)

theorem fifth_bowler_points : P5 = 561 :=
  sorry

end fifth_bowler_points_l185_185283


namespace value_of_six_inch_cube_l185_185107

-- Defining the conditions
def original_cube_weight : ℝ := 5 -- in pounds
def original_cube_value : ℝ := 600 -- in dollars
def original_cube_side : ℝ := 4 -- in inches

def new_cube_side : ℝ := 6 -- in inches

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

-- Statement of the theorem
theorem value_of_six_inch_cube :
  cube_volume new_cube_side / cube_volume original_cube_side * original_cube_value = 2025 :=
by
  -- Here goes the proof
  sorry

end value_of_six_inch_cube_l185_185107


namespace problem_statement_l185_185626

-- Define a multiple of 6 and a multiple of 9
variables (a b : ℤ)
variable (ha : ∃ k, a = 6 * k)
variable (hb : ∃ k, b = 9 * k)

-- Prove that a + b is a multiple of 3
theorem problem_statement : 
  (∃ k, a + b = 3 * k) ∧ 
  ¬((∀ m n, a = 6 * m ∧ b = 9 * n → (a + b = odd))) ∧ 
  ¬(∃ k, a + b = 6 * k) ∧ 
  ¬(∃ k, a + b = 9 * k) :=
by
  sorry

end problem_statement_l185_185626


namespace hexagon_angle_sum_l185_185006

theorem hexagon_angle_sum (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) :
  a1 + a2 + a3 + a4 = 360 ∧ b1 + b2 + b3 + b4 = 360 → 
  a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 = 720 :=
by
  sorry

end hexagon_angle_sum_l185_185006


namespace degree_of_divisor_l185_185577

theorem degree_of_divisor (f q r d : Polynomial ℝ)
  (h_f : f.degree = 15)
  (h_q : q.degree = 9)
  (h_r : r = Polynomial.C 5 * X^4 + Polynomial.C 3 * X^3 - Polynomial.C 2 * X^2 + Polynomial.C 9 * X - Polynomial.C 7)
  (h_div : f = d * q + r) :
  d.degree = 6 :=
by sorry

end degree_of_divisor_l185_185577


namespace no_integer_n_satisfies_conditions_l185_185644

theorem no_integer_n_satisfies_conditions :
  ¬ ∃ n : ℕ, 0 < n ∧ 1000 ≤ n / 5 ∧ n / 5 ≤ 9999 ∧ 1000 ≤ 5 * n ∧ 5 * n ≤ 9999 :=
by
  sorry

end no_integer_n_satisfies_conditions_l185_185644


namespace sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l185_185912

-- Define the real interval [0, π/2]
def interval_0_pi_over_2 (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi / 2

-- Define the proposition to be proven
theorem sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq (a b : ℝ) 
  (ha : interval_0_pi_over_2 a) (hb : interval_0_pi_over_2 b) :
  (Real.sin a)^6 + 3 * (Real.sin a)^2 * (Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b :=
by
  sorry

end sin6_add_3sin2_cos2_add_cos6_eq_one_iff_eq_l185_185912


namespace F5_div_641_Fermat_rel_prime_l185_185569

def Fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

theorem F5_div_641 : Fermat_number 5 % 641 = 0 := 
  sorry

theorem Fermat_rel_prime (k n : ℕ) (hk: k ≠ n) : Nat.gcd (Fermat_number k) (Fermat_number n) = 1 :=
  sorry

end F5_div_641_Fermat_rel_prime_l185_185569


namespace smallest_k_l185_185693

theorem smallest_k (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * max m n + min m n - 1 ∧ 
  (∀ (persons : Finset ℕ),
    persons.card ≥ k →
    (∃ (acquainted : Finset (ℕ × ℕ)), acquainted.card = m ∧ 
      (∀ (x y : ℕ), (x, y) ∈ acquainted → (x ∈ persons ∧ y ∈ persons))) ∨
    (∃ (unacquainted : Finset (ℕ × ℕ)), unacquainted.card = n ∧ 
      (∀ (x y : ℕ), (x, y) ∈ unacquainted → (x ∈ persons ∧ y ∈ persons ∧ x ≠ y)))) :=
sorry

end smallest_k_l185_185693


namespace find_single_digit_A_l185_185787

theorem find_single_digit_A (A : ℕ) (h1 : 0 ≤ A) (h2 : A < 10) (h3 : (10 * A + A) * (10 * A + A) = 5929) : A = 7 :=
sorry

end find_single_digit_A_l185_185787


namespace product_of_y_coordinates_l185_185029

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem product_of_y_coordinates : 
  let P1 := (1, 2 + 4 * Real.sqrt 2)
  let P2 := (1, 2 - 4 * Real.sqrt 2)
  distance (5, 2) P1 = 12 ∧ distance (5, 2) P2 = 12 →
  (P1.2 * P2.2 = -28) :=
by
  intros
  sorry

end product_of_y_coordinates_l185_185029


namespace ring_width_l185_185238

noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def outerCircumference : ℝ := 528 / 7

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

theorem ring_width :
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  r_outer - r_inner = 4 :=
by
  -- Definitions for inner and outer radius
  let r_inner := radius innerCircumference
  let r_outer := radius outerCircumference
  -- Proof goes here
  sorry

end ring_width_l185_185238


namespace population_initial_count_l185_185888

theorem population_initial_count
  (P : ℕ)
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℝ := 1.2) :
  36 = (net_growth_rate / 100) * P ↔ P = 3000 :=
by sorry

end population_initial_count_l185_185888


namespace sculptures_not_on_display_approx_400_l185_185568

theorem sculptures_not_on_display_approx_400 (A : ℕ) (hA : A = 900) :
  (2 / 3 * A - 2 / 9 * A) = 400 := by
  sorry

end sculptures_not_on_display_approx_400_l185_185568


namespace final_answer_correct_l185_185490

-- Define the initial volume V0
def V0 := 1

-- Define the volume increment ratio for new tetrahedra
def volume_ratio := (1 : ℚ) / 27

-- Define the recursive volume increments
def ΔP1 := 4 * volume_ratio
def ΔP2 := 16 * volume_ratio
def ΔP3 := 64 * volume_ratio
def ΔP4 := 256 * volume_ratio

-- Define the total volume V4
def V4 := V0 + ΔP1 + ΔP2 + ΔP3 + ΔP4

-- The target volume as a rational number
def target_volume := 367 / 27

-- Define the fraction components
def m := 367
def n := 27

-- Define the final answer
def final_answer := m + n

-- Proof statement to verify the final answer
theorem final_answer_correct :
  V4 = target_volume ∧ (Nat.gcd m n = 1) ∧ final_answer = 394 :=
by
  -- The specifics of the proof are omitted
  sorry

end final_answer_correct_l185_185490


namespace intersection_M_N_l185_185098

def M : Set ℝ := {x | x < 2016}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l185_185098


namespace midpoint_AB_is_correct_l185_185467

/--
In the Cartesian coordinate system, given points A (-1, 2) and B (3, 0), prove that the coordinates of the midpoint of segment AB are (1, 1).
-/
theorem midpoint_AB_is_correct :
  let A := (-1, 2)
  let B := (3, 0)
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1 := 
by {
  let A := (-1, 2)
  let B := (3, 0)
  sorry -- this part is omitted as no proof is needed
}

end midpoint_AB_is_correct_l185_185467


namespace evaluate_expression_l185_185336

theorem evaluate_expression : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := 
by
  sorry

end evaluate_expression_l185_185336


namespace eq_correct_l185_185896

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end eq_correct_l185_185896


namespace locus_of_P_l185_185033

-- Definitions based on conditions
def F : ℝ × ℝ := (2, 0)
def Q (k : ℝ) : ℝ × ℝ := (0, -2 * k)
def T (k : ℝ) : ℝ × ℝ := (-2 * k^2, 0)
def P (k : ℝ) : ℝ × ℝ := (2 * k^2, -4 * k)

-- Theorem statement based on the proof problem
theorem locus_of_P (x y : ℝ) (k : ℝ) (hf : F = (2, 0)) (hq : Q k = (0, -2 * k))
  (ht : T k = (-2 * k^2, 0)) (hp : P k = (2 * k^2, -4 * k)) :
  y^2 = 8 * x :=
sorry

end locus_of_P_l185_185033


namespace points_per_draw_l185_185620

-- Definitions based on conditions
def total_games : ℕ := 20
def wins : ℕ := 14
def losses : ℕ := 2
def total_points : ℕ := 46
def points_per_win : ℕ := 3
def points_per_loss : ℕ := 0

-- Calculation of the number of draws and points per draw
def draws : ℕ := total_games - wins - losses
def points_wins : ℕ := wins * points_per_win
def points_draws : ℕ := total_points - points_wins

-- Theorem statement
theorem points_per_draw : points_draws / draws = 1 := by
  sorry

end points_per_draw_l185_185620


namespace expense_recording_l185_185559

-- Define the recording of income and expenses
def record_income (amount : Int) : Int := amount
def record_expense (amount : Int) : Int := -amount

-- Given conditions
def income_example := record_income 500
def expense_example := record_expense 400

-- Prove that an expense of 400 yuan is recorded as -400 yuan
theorem expense_recording : record_expense 400 = -400 :=
  by sorry

end expense_recording_l185_185559


namespace c_10_value_l185_185606

def c : ℕ → ℤ
| 0 => 3
| 1 => 9
| (n + 1) => c n * c (n - 1)

theorem c_10_value : c 10 = 3^89 :=
by
  sorry

end c_10_value_l185_185606


namespace fraction_first_to_second_l185_185783

def digit_fraction_proof_problem (a b c d : ℕ) (number : ℕ) :=
  number = 1349 ∧
  a = b / 3 ∧
  c = a + b ∧
  d = 3 * b

theorem fraction_first_to_second (a b c d : ℕ) (number : ℕ) :
  digit_fraction_proof_problem a b c d number → a / b = 1 / 3 :=
by
  intro problem
  sorry

end fraction_first_to_second_l185_185783


namespace total_selling_price_is_correct_l185_185688

-- Define the given constants
def meters_of_cloth : ℕ := 85
def profit_per_meter : ℕ := 10
def cost_price_per_meter : ℕ := 95

-- Compute the selling price per meter
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- Calculate the total selling price
def total_selling_price : ℕ := selling_price_per_meter * meters_of_cloth

-- The theorem statement
theorem total_selling_price_is_correct : total_selling_price = 8925 := by
  sorry

end total_selling_price_is_correct_l185_185688


namespace fraction_draw_l185_185243

theorem fraction_draw (john_wins : ℚ) (mike_wins : ℚ) (h_john : john_wins = 4 / 9) (h_mike : mike_wins = 5 / 18) :
    1 - (john_wins + mike_wins) = 5 / 18 :=
by
    rw [h_john, h_mike]
    sorry

end fraction_draw_l185_185243


namespace exists_m_n_for_any_d_l185_185895

theorem exists_m_n_for_any_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) :=
by
  sorry

end exists_m_n_for_any_d_l185_185895


namespace sqrt_sixteen_equals_four_l185_185991

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l185_185991


namespace range_of_y_l185_185980

noncomputable def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_y :
  (∀ x : ℝ, operation (x - y) (x + y) < 1) ↔ - (1 : ℝ) / 2 < y ∧ y < (3 : ℝ) / 2 :=
by
  sorry

end range_of_y_l185_185980


namespace distance_to_focus_2_l185_185348

-- Definition of the ellipse and the given distance to one focus
def ellipse (P : ℝ × ℝ) : Prop := (P.1^2)/25 + (P.2^2)/16 = 1
def distance_to_focus_1 (P : ℝ × ℝ) : Prop := dist P (5, 0) = 3

-- Proof problem statement
theorem distance_to_focus_2 (P : ℝ × ℝ) (h₁ : ellipse P) (h₂ : distance_to_focus_1 P) :
  dist P (-5, 0) = 7 :=
sorry

end distance_to_focus_2_l185_185348


namespace roots_polynomial_sum_products_l185_185499

theorem roots_polynomial_sum_products (p q r : ℂ)
  (h : 6 * p^3 - 5 * p^2 + 13 * p - 10 = 0)
  (h' : 6 * q^3 - 5 * q^2 + 13 * q - 10 = 0)
  (h'' : 6 * r^3 - 5 * r^2 + 13 * r - 10 = 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) :
  p * q + q * r + r * p = 13 / 6 := 
sorry

end roots_polynomial_sum_products_l185_185499


namespace exists_a_lt_0_l185_185423

noncomputable def f : ℝ → ℝ :=
sorry

theorem exists_a_lt_0 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (Real.sqrt (x * y)) = (f x + f y) / 2)
  (h2 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :
  ∃ a : ℝ, 0 < a ∧ f a < 0 :=
sorry

end exists_a_lt_0_l185_185423


namespace house_painting_cost_l185_185831

theorem house_painting_cost :
  let judson_contrib := 500.0
  let kenny_contrib_euros := judson_contrib * 1.2 / 1.1
  let camilo_contrib_pounds := (kenny_contrib_euros * 1.1 + 200.0) / 1.3
  let camilo_contrib_usd := camilo_contrib_pounds * 1.3
  judson_contrib + kenny_contrib_euros * 1.1 + camilo_contrib_usd = 2020.0 := 
by {
  sorry
}

end house_painting_cost_l185_185831


namespace lcm_of_three_numbers_is_180_l185_185741

-- Define the three numbers based on the ratio and HCF condition
def a : ℕ := 2 * 6
def b : ℕ := 3 * 6
def c : ℕ := 5 * 6

-- State the theorem regarding the LCM
theorem lcm_of_three_numbers_is_180 : Nat.lcm (Nat.lcm a b) c = 180 :=
by
  sorry

end lcm_of_three_numbers_is_180_l185_185741


namespace scientific_notation_correct_l185_185036

theorem scientific_notation_correct : 657000 = 6.57 * 10^5 :=
by
  sorry

end scientific_notation_correct_l185_185036


namespace percentage_exceeds_l185_185524

theorem percentage_exceeds (x y : ℝ) (h₁ : x < y) (h₂ : y = x + 0.35 * x) : ((y - x) / x) * 100 = 35 :=
by sorry

end percentage_exceeds_l185_185524


namespace max_angle_MPN_is_pi_over_2_l185_185513

open Real

noncomputable def max_angle_MPN (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : ℝ :=
  sorry

theorem max_angle_MPN_is_pi_over_2 (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : 
  max_angle_MPN θ P hP = π / 2 :=
sorry

end max_angle_MPN_is_pi_over_2_l185_185513


namespace triangle_inequality_x_values_l185_185961

theorem triangle_inequality_x_values :
  {x : ℕ | 1 ≤ x ∧ x < 14} = {x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12 ∨ x = 13} :=
  by
    sorry

end triangle_inequality_x_values_l185_185961


namespace area_of_trapezium_l185_185381

/-- Two parallel sides of a trapezium are 4 cm and 5 cm respectively. 
    The perpendicular distance between the parallel sides is 6 cm.
    Prove that the area of the trapezium is 27 cm². -/
theorem area_of_trapezium (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) : 
  (1/2) * (a + b) * h = 27 := 
by 
  sorry

end area_of_trapezium_l185_185381


namespace gcd_problem_l185_185913

variable (A B : ℕ)
variable (hA : A = 2 * 3 * 5)
variable (hB : B = 2 * 2 * 5 * 7)

theorem gcd_problem : Nat.gcd A B = 10 :=
by
  -- Proof is omitted.
  sorry

end gcd_problem_l185_185913


namespace product_of_consecutive_integers_plus_one_l185_185492

theorem product_of_consecutive_integers_plus_one (n : ℤ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 := 
sorry

end product_of_consecutive_integers_plus_one_l185_185492


namespace find_k_l185_185295

open BigOperators

noncomputable
def hyperbola_property (k : ℝ) (x a b c : ℝ) : Prop :=
  k > 0 ∧
  (a / 2, b / 2) = (a / 2, k / a / 2) ∧ -- midpoint condition
  abs (a * b) / 2 = 3 ∧                -- area condition
  b = k / a                            -- point B on the hyperbola

theorem find_k (k : ℝ) (x a b c : ℝ) : hyperbola_property k x a b c → k = 2 :=
by
  sorry

end find_k_l185_185295


namespace other_number_remainder_l185_185066

theorem other_number_remainder (x : ℕ) (k n : ℤ) (hx : x > 0) (hk : 200 = k * x + 2) (hnk : n ≠ k) : ∃ m : ℤ, (n * ↑x + 2) = m * ↑x + 2 ∧ (n * ↑x + 2) % x = 2 := 
by
  sorry

end other_number_remainder_l185_185066


namespace sqrt_expr_equals_sum_l185_185557

theorem sqrt_expr_equals_sum :
  ∃ x y z : ℤ,
    (x + y * Int.sqrt z = Real.sqrt (77 + 28 * Real.sqrt 3)) ∧
    (x^2 + y^2 * z = 77) ∧
    (2 * x * y = 28) ∧
    (x + y + z = 16) :=
by
  sorry

end sqrt_expr_equals_sum_l185_185557


namespace ivan_total_pay_l185_185662

theorem ivan_total_pay (cost_per_card : ℕ) (number_of_cards : ℕ) (discount_per_card : ℕ) :
  cost_per_card = 12 → number_of_cards = 10 → discount_per_card = 2 →
  (number_of_cards * (cost_per_card - discount_per_card)) = 100 :=
by
  intro h1 h2 h3
  sorry

end ivan_total_pay_l185_185662


namespace ratio_of_areas_is_two_thirds_l185_185727

noncomputable def PQ := 10
noncomputable def PR := 6
noncomputable def QR := 4
noncomputable def r_PQ := PQ / 2
noncomputable def r_PR := PR / 2
noncomputable def r_QR := QR / 2
noncomputable def area_semi_PQ := (1 / 2) * Real.pi * r_PQ^2
noncomputable def area_semi_PR := (1 / 2) * Real.pi * r_PR^2
noncomputable def area_semi_QR := (1 / 2) * Real.pi * r_QR^2
noncomputable def shaded_area := (area_semi_PQ - area_semi_PR) + area_semi_QR
noncomputable def total_area_circle := Real.pi * r_PQ^2
noncomputable def unshaded_area := total_area_circle - shaded_area
noncomputable def ratio := shaded_area / unshaded_area

theorem ratio_of_areas_is_two_thirds : ratio = 2 / 3 := by
  sorry

end ratio_of_areas_is_two_thirds_l185_185727


namespace circles_intersect_iff_l185_185218

-- Definitions of the two circles and their parameters
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9

def circle2 (x y r : ℝ) : Prop := x^2 + y^2 + 8 * x - 6 * y + 25 - r^2 = 0

-- Lean statement to prove the range of r
theorem circles_intersect_iff (r : ℝ) (hr : 0 < r) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y r) ↔ (2 < r ∧ r < 8) :=
by
  sorry

end circles_intersect_iff_l185_185218


namespace parallel_lines_solution_l185_185007

theorem parallel_lines_solution (m : ℝ) :
  (∀ x y : ℝ, (x + (1 + m) * y + (m - 2) = 0) → (m * x + 2 * y + 8 = 0)) → m = 1 :=
by
  sorry

end parallel_lines_solution_l185_185007


namespace geom_seq_min_m_l185_185103

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def annual_payment (t : ℝ) : Prop := t ≤ 2500
def capital_remaining (aₙ : ℕ → ℝ) (n : ℕ) (t : ℝ) : ℝ := aₙ n * (1 + growth_rate) - t

theorem geom_seq (aₙ : ℕ → ℝ) (t : ℝ) (h₁ : annual_payment t) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (t ≠ 2500) →
  ∃ r : ℝ, ∀ n, aₙ n - 2 * t = (aₙ 0 - 2 * t) * r ^ n :=
sorry

theorem min_m (t : ℝ) (h₁ : t = 1500) (aₙ : ℕ → ℝ) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (aₙ 0 = initial_capital * (1 + growth_rate) - t) →
  ∃ m : ℕ, aₙ m > 21000 ∧ ∀ k < m, aₙ k ≤ 21000 :=
sorry

end geom_seq_min_m_l185_185103


namespace jake_steps_per_second_l185_185537

/-
Conditions:
1. Austin and Jake start descending from the 9th floor at the same time.
2. The stairs have 30 steps across each floor.
3. The elevator takes 1 minute (60 seconds) to reach the ground floor.
4. Jake reaches the ground floor 30 seconds after Austin.
5. Jake descends 8 floors to reach the ground floor.
-/

def floors : ℕ := 8
def steps_per_floor : ℕ := 30
def time_elevator : ℕ := 60 -- in seconds
def additional_time_jake : ℕ := 30 -- in seconds

def total_time_jake := time_elevator + additional_time_jake -- in seconds
def total_steps := floors * steps_per_floor

def steps_per_second_jake := (total_steps : ℚ) / (total_time_jake : ℚ)

theorem jake_steps_per_second :
  steps_per_second_jake = 2.67 := by
  sorry

end jake_steps_per_second_l185_185537


namespace find_f_l185_185751

theorem find_f {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2*x := 
by
  sorry

end find_f_l185_185751


namespace fourth_number_is_8_l185_185862

theorem fourth_number_is_8 (a b c : ℕ) (mean : ℕ) (h_mean : mean = 20) (h_a : a = 12) (h_b : b = 24) (h_c : c = 36) :
  ∃ d : ℕ, mean * 4 = a + b + c + d ∧ (∃ x : ℕ, d = x^2) ∧ d = 8 := by
sorry

end fourth_number_is_8_l185_185862


namespace sufficient_but_not_necessary_l185_185290

theorem sufficient_but_not_necessary (x : ℝ) : (x < -2 → x ≤ 0) → ¬(x ≤ 0 → x < -2) :=
by
  sorry

end sufficient_but_not_necessary_l185_185290


namespace sum_of_10th_degree_polynomials_is_no_higher_than_10_l185_185458

-- Given definitions of two 10th-degree polynomials
def polynomial1 := ∃p : Polynomial ℝ, p.degree = 10
def polynomial2 := ∃p : Polynomial ℝ, p.degree = 10

-- Statement to prove
theorem sum_of_10th_degree_polynomials_is_no_higher_than_10 :
  ∀ (p q : Polynomial ℝ), p.degree = 10 → q.degree = 10 → (p + q).degree ≤ 10 := by
  sorry

end sum_of_10th_degree_polynomials_is_no_higher_than_10_l185_185458


namespace simplify_sqrt_l185_185965

theorem simplify_sqrt (x : ℝ) (h : x < 2) : Real.sqrt (x^2 - 4*x + 4) = 2 - x :=
by
  sorry

end simplify_sqrt_l185_185965


namespace solve_natural_numbers_system_l185_185848

theorem solve_natural_numbers_system :
  ∃ a b c : ℕ, (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (a + b + c)) ∧
  ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end solve_natural_numbers_system_l185_185848


namespace perimeter_C_l185_185752

def is_square (n : ℕ) : Prop := n > 0 ∧ ∃ s : ℕ, s * s = n

variable (A B C : ℕ) -- Defining the squares
variable (sA sB sC : ℕ) -- Defining the side lengths

-- Conditions as definitions
axiom square_figures : is_square A ∧ is_square B ∧ is_square C 
axiom perimeter_A : 4 * sA = 20
axiom perimeter_B : 4 * sB = 40
axiom side_length_C : sC = 2 * (sA + sB)

-- The equivalent proof problem statement
theorem perimeter_C : 4 * sC = 120 :=
by
  -- Proof will go here
  sorry

end perimeter_C_l185_185752


namespace new_weight_l185_185054

-- Conditions
def avg_weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase
def weight_replacement (initial_weight : ℝ) (total_increase : ℝ) : ℝ := initial_weight + total_increase

-- Problem Statement: Proving the weight of the new person
theorem new_weight {n : ℕ} {avg_increase initial_weight W : ℝ} 
  (h_n : n = 8) (h_avg_increase : avg_increase = 2.5) (h_initial_weight : initial_weight = 65) (h_W : W = 85) :
  weight_replacement initial_weight (avg_weight_increase n avg_increase) = W :=
by 
  rw [h_n, h_avg_increase, h_initial_weight, h_W]
  sorry

end new_weight_l185_185054


namespace find_value_b_in_geometric_sequence_l185_185697

theorem find_value_b_in_geometric_sequence
  (b : ℝ)
  (h1 : 15 ≠ 0) -- to ensure division by zero does not occur
  (h2 : b ≠ 0)  -- to ensure division by zero does not occur
  (h3 : 15 * (b / 15) = b) -- 15 * r = b
  (h4 : b * (b / 15) = 45 / 4) -- b * r = 45 / 4
  : b = 15 * Real.sqrt 3 / 2 :=
sorry

end find_value_b_in_geometric_sequence_l185_185697


namespace rectangle_area_inscribed_circle_l185_185159

theorem rectangle_area_inscribed_circle (r : ℝ) (h : r = 7) (ratio : ℝ) (hratio : ratio = 3) : 
  (2 * r) * (ratio * (2 * r)) = 588 :=
by
  rw [h, hratio]
  sorry

end rectangle_area_inscribed_circle_l185_185159


namespace find_angle_C_find_side_c_l185_185085

variable {A B C a b c : ℝ}
variable {AD CD area_ABD : ℝ}

-- Conditions for question 1
variable (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))

-- Conditions for question 2
variable (h2 : AD = 4)
variable (h3 : CD = 4)
variable (h4 : area_ABD = 8 * Real.sqrt 3)
variable (h5 : C = Real.pi / 3)

-- Lean 4 statement for both parts of the problem
theorem find_angle_C (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A)) : 
  C = Real.pi / 3 :=
sorry

theorem find_side_c (h2 : AD = 4) (h3 : CD = 4) (h4 : area_ABD = 8 * Real.sqrt 3) (h5 : C = Real.pi / 3) : 
  c = 4 * Real.sqrt 7 :=
sorry

end find_angle_C_find_side_c_l185_185085


namespace symmetric_graph_increasing_interval_l185_185535

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_graph_increasing_interval :
  (∀ x : ℝ, f (-x) = -f x) → -- f is odd
  (∀ x y : ℝ, 3 ≤ x → x < y → y ≤ 7 → f x < f y) → -- f is increasing in [3,7]
  (∀ x : ℝ, 3 ≤ x → x ≤ 7 → f x ≤ 5) → -- f has a maximum value of 5 in [3,7]
  (∀ x y : ℝ, -7 ≤ x → x < y → y ≤ -3 → f x < f y) ∧ -- f is increasing in [-7,-3]
  (∀ x : ℝ, -7 ≤ x → x ≤ -3 → f x ≥ -5) -- f has a minimum value of -5 in [-7,-3]
:= sorry

end symmetric_graph_increasing_interval_l185_185535


namespace tom_hours_per_week_l185_185335

-- Define the conditions
def summer_hours_per_week := 40
def summer_weeks := 8
def summer_total_earnings := 3200
def semester_weeks := 24
def semester_total_earnings := 2400
def hourly_wage := summer_total_earnings / (summer_hours_per_week * summer_weeks)
def total_hours_needed := semester_total_earnings / hourly_wage

-- Define the theorem to prove
theorem tom_hours_per_week :
  (total_hours_needed / semester_weeks) = 10 :=
sorry

end tom_hours_per_week_l185_185335


namespace least_m_plus_n_l185_185975

theorem least_m_plus_n (m n : ℕ) (hmn : Nat.gcd (m + n) 330 = 1) (hm_multiple : m^m % n^n = 0) (hm_not_multiple : ¬ (m % n = 0)) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  m + n = 119 :=
sorry

end least_m_plus_n_l185_185975


namespace mr_mcpherson_needs_to_raise_840_l185_185902

def total_rent : ℝ := 1200
def mrs_mcpherson_contribution : ℝ := 0.30 * total_rent
def mr_mcpherson_contribution : ℝ := total_rent - mrs_mcpherson_contribution

theorem mr_mcpherson_needs_to_raise_840 :
  mr_mcpherson_contribution = 840 := 
by
  sorry

end mr_mcpherson_needs_to_raise_840_l185_185902


namespace trapezoid_distances_l185_185802

-- Define the problem parameters
variables (AB CD AD BC : ℝ)
-- Assume given conditions
axiom h1 : AD > BC
noncomputable def k := AD / BC

-- Formalizing the proof problem in Lean 4
theorem trapezoid_distances (M : Type) (BM AM CM DM : ℝ) :
  BM = AB * BC / (AD - BC) →
  AM = AB * AD / (AD - BC) →
  CM = CD * BC / (AD - BC) →
  DM = CD * AD / (AD - BC) →
  true :=
sorry

end trapezoid_distances_l185_185802


namespace calculate_percentage_l185_185040

theorem calculate_percentage :
  let total_students := 40
  let A_on_both := 4
  let B_on_both := 6
  let C_on_both := 3
  let D_on_Test1_C_on_Test2 := 2
  let valid_students := A_on_both + B_on_both + C_on_both + D_on_Test1_C_on_Test2
  (valid_students / total_students) * 100 = 37.5 :=
by
  sorry

end calculate_percentage_l185_185040


namespace determine_teeth_l185_185013

theorem determine_teeth (x V : ℝ) (h1 : V = 63 * x / (x + 10)) (h2 : V = 28 * (x + 10)) :
  x = 20 ∧ (x + 10) = 30 :=
by
  sorry

end determine_teeth_l185_185013


namespace shorts_more_than_checkered_l185_185362

noncomputable def total_students : ℕ := 81

noncomputable def striped_shirts : ℕ := (2 * total_students) / 3

noncomputable def checkered_shirts : ℕ := total_students - striped_shirts

noncomputable def shorts : ℕ := striped_shirts - 8

theorem shorts_more_than_checkered :
  shorts - checkered_shirts = 19 :=
by
  sorry

end shorts_more_than_checkered_l185_185362


namespace noah_garden_larger_by_75_l185_185931

-- Define the dimensions of Liam's garden
def length_liam : ℕ := 30
def width_liam : ℕ := 50

-- Define the dimensions of Noah's garden
def length_noah : ℕ := 35
def width_noah : ℕ := 45

-- Define the areas of the gardens
def area_liam : ℕ := length_liam * width_liam
def area_noah : ℕ := length_noah * width_noah

theorem noah_garden_larger_by_75 :
  area_noah - area_liam = 75 :=
by
  -- The proof goes here
  sorry

end noah_garden_larger_by_75_l185_185931


namespace sales_tax_percentage_l185_185020

theorem sales_tax_percentage 
  (total_spent : ℝ)
  (tip_percent : ℝ)
  (food_price : ℝ) 
  (total_with_tip : total_spent = food_price * (1 + tip_percent / 100))
  (sales_tax_percent : ℝ) 
  (total_paid : total_spent = food_price * (1 + sales_tax_percent / 100) * (1 + tip_percent / 100)) :
  sales_tax_percent = 10 :=
by sorry

end sales_tax_percentage_l185_185020


namespace part_a_l185_185209

theorem part_a (m : ℕ) (A B : ℕ) (hA : A = (10^(2 * m) - 1) / 9) (hB : B = 4 * ((10^m - 1) / 9)) :
  ∃ k : ℕ, A + B + 1 = k^2 :=
sorry

end part_a_l185_185209


namespace final_price_jacket_l185_185372

-- Defining the conditions as per the problem
def original_price : ℚ := 250
def first_discount_rate : ℚ := 0.40
def second_discount_rate : ℚ := 0.15
def tax_rate : ℚ := 0.05

-- Defining the calculation steps
def first_discounted_price : ℚ := original_price * (1 - first_discount_rate)
def second_discounted_price : ℚ := first_discounted_price * (1 - second_discount_rate)
def final_price_inclusive_tax : ℚ := second_discounted_price * (1 + tax_rate)

-- The proof problem statement
theorem final_price_jacket : final_price_inclusive_tax = 133.88 := sorry

end final_price_jacket_l185_185372


namespace solve_k_n_l185_185926
-- Import the entire Mathlib

-- Define the theorem statement
theorem solve_k_n (k n : ℕ) (hk : k > 0) (hn : n > 0) : k^2 - 2016 = 3^n ↔ k = 45 ∧ n = 2 :=
  by sorry

end solve_k_n_l185_185926


namespace infinite_geometric_subsequence_exists_l185_185701

theorem infinite_geometric_subsequence_exists
  (a : ℕ) (d : ℕ) (h_d_pos : d > 0)
  (a_n : ℕ → ℕ)
  (h_arith_prog : ∀ n, a_n n = a + n * d) :
  ∃ (g : ℕ → ℕ), (∀ m n, m < n → g m < g n) ∧ (∃ r : ℕ, ∀ n, g (n+1) = g n * r) ∧ (∀ n, ∃ m, a_n m = g n) :=
sorry

end infinite_geometric_subsequence_exists_l185_185701


namespace f_of_2_l185_185704

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_value : f (-2) = 11)

-- The theorem we want to prove
theorem f_of_2 : f 2 = -11 :=
by 
  sorry

end f_of_2_l185_185704


namespace curve_is_parabola_l185_185391

theorem curve_is_parabola (r θ : ℝ) : (r = 1 / (1 - Real.cos θ)) ↔ ∃ x y : ℝ, y^2 = 2 * x + 1 :=
by 
  sorry

end curve_is_parabola_l185_185391


namespace player2_winning_strategy_l185_185977

-- Definitions of the game setup
def initial_position_player1 := (1, 1)
def initial_position_player2 := (998, 1998)

def adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 - 1 ∨ p1.2 = p2.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 - 1 ∨ p1.1 = p2.1 + 1))

-- A function defining the winning condition for Player 2
def player2_wins (p1 p2 : ℕ × ℕ) : Prop :=
  p1 = p2 ∨ p1.1 = (initial_position_player2.1)

-- Theorem stating the pair (998, 1998) guarantees a win for Player 2
theorem player2_winning_strategy : player2_wins (998, 0) (998, 1998) :=
sorry

end player2_winning_strategy_l185_185977


namespace find_multiple_l185_185827

theorem find_multiple (x m : ℤ) (hx : x = 13) (h : x + x + 2 * x + m * x = 104) : m = 4 :=
by
  -- Proof to be provided
  sorry

end find_multiple_l185_185827


namespace gcd_372_684_l185_185210

theorem gcd_372_684 : Int.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l185_185210


namespace maximize_sector_area_l185_185824

noncomputable def max_area_sector_angle (r : ℝ) (l := 36 - 2 * r) (α := l / r) : ℝ :=
  α

theorem maximize_sector_area (h : ∀ r : ℝ, 2 * r + 36 - 2 * r = 36) :
  max_area_sector_angle 9 = 2 :=
by
  sorry

end maximize_sector_area_l185_185824


namespace greatest_n_le_5_value_ge_2525_l185_185808

theorem greatest_n_le_5_value_ge_2525 (n : ℤ) (V : ℤ) 
  (h1 : 101 * n^2 ≤ V) 
  (h2 : ∀ k : ℤ, (101 * k^2 ≤ V) → (k ≤ 5)) : 
  V ≥ 2525 := 
sorry

end greatest_n_le_5_value_ge_2525_l185_185808


namespace exponential_inequality_l185_185291

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) :=
sorry

end exponential_inequality_l185_185291


namespace M_gt_N_l185_185671

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2 * (x + y - 1)

theorem M_gt_N : M x y > N x y := sorry

end M_gt_N_l185_185671


namespace math_problem_l185_185462

theorem math_problem (a b c : ℝ) (h1 : (a + b) / 2 = 30) (h2 : (b + c) / 2 = 60) (h3 : c - a = 60) : c - a = 60 :=
by
  -- Insert proof steps here
  sorry

end math_problem_l185_185462


namespace relationship_f_neg2_f_expr_l185_185798

noncomputable def f : ℝ → ℝ := sorry  -- f is some function ℝ → ℝ, the exact definition is not provided

axiom even_function : ∀ x : ℝ, f (-x) = f x -- f is an even function
axiom increasing_on_negatives : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y -- f is increasing on (-∞, 0)

theorem relationship_f_neg2_f_expr (a : ℝ) : f (-2) ≥ f (a^2 - 4 * a + 6) := by
  -- proof omitted
  sorry

end relationship_f_neg2_f_expr_l185_185798


namespace mechanic_worked_hours_l185_185077

theorem mechanic_worked_hours (total_spent : ℕ) (cost_per_part : ℕ) (labor_cost_per_minute : ℚ) (parts_needed : ℕ) :
  total_spent = 220 → cost_per_part = 20 → labor_cost_per_minute = 0.5 → parts_needed = 2 →
  (total_spent - cost_per_part * parts_needed) / labor_cost_per_minute / 60 = 6 := by
  -- Proof will be inserted here
  sorry

end mechanic_worked_hours_l185_185077


namespace average_page_count_l185_185175

theorem average_page_count 
  (n1 n2 n3 n4 : ℕ)
  (p1 p2 p3 p4 total_students : ℕ)
  (h1 : n1 = 8)
  (h2 : p1 = 3)
  (h3 : n2 = 10)
  (h4 : p2 = 5)
  (h5 : n3 = 7)
  (h6 : p3 = 2)
  (h7 : n4 = 5)
  (h8 : p4 = 4)
  (h9 : total_students = 30) :
  (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) / total_students = 36 / 10 := 
sorry

end average_page_count_l185_185175


namespace overall_profit_no_discount_l185_185297

theorem overall_profit_no_discount:
  let C_b := 100
  let C_p := 100
  let C_n := 100
  let profit_b := 42.5 / 100
  let profit_p := 35 / 100
  let profit_n := 20 / 100
  let S_b := C_b + (C_b * profit_b)
  let S_p := C_p + (C_p * profit_p)
  let S_n := C_n + (C_n * profit_n)
  let TCP := C_b + C_p + C_n
  let TSP := S_b + S_p + S_n
  let OverallProfit := TSP - TCP
  let OverallProfitPercentage := (OverallProfit / TCP) * 100
  OverallProfitPercentage = 32.5 :=
by sorry

end overall_profit_no_discount_l185_185297


namespace group_photo_arrangements_l185_185219

theorem group_photo_arrangements :
  ∃ (arrangements : ℕ), arrangements = 36 ∧
    ∀ (M G H P1 P2 : ℕ),
    (M = G + 1 ∨ M + 1 = G) ∧ (M ≠ H - 1 ∧ M ≠ H + 1) →
    arrangements = 36 :=
by {
  sorry
}

end group_photo_arrangements_l185_185219


namespace total_number_of_students_l185_185634

/-- The total number of high school students in the school given sampling constraints. -/
theorem total_number_of_students (F1 F2 F3 : ℕ) (sample_size : ℕ) (consistency_ratio : ℕ) :
  F2 = 300 ∧ sample_size = 45 ∧ (F1 / F3) = 2 ∧ 
  (20 + 10 + (sample_size - 30)) = sample_size → F1 + F2 + F3 = 900 :=
by
  sorry

end total_number_of_students_l185_185634


namespace units_digit_6_pow_6_l185_185579

theorem units_digit_6_pow_6 : (6 ^ 6) % 10 = 6 := 
by {
  sorry
}

end units_digit_6_pow_6_l185_185579


namespace remainder_when_13_plus_y_divided_by_31_l185_185877

theorem remainder_when_13_plus_y_divided_by_31
  (y : ℕ)
  (hy : 7 * y % 31 = 1) :
  (13 + y) % 31 = 22 :=
sorry

end remainder_when_13_plus_y_divided_by_31_l185_185877


namespace fixed_fee_rental_l185_185027

theorem fixed_fee_rental (F C h : ℕ) (hC : C = F + 7 * h) (hC80 : C = 80) (hh9 : h = 9) : F = 17 :=
by
  sorry

end fixed_fee_rental_l185_185027


namespace intersection_M_N_l185_185042

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x * x = x}

theorem intersection_M_N :
  M ∩ N = {0, 1} :=
sorry

end intersection_M_N_l185_185042


namespace matrix_solution_l185_185501

-- Define the 2x2 matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := 
  ![ ![30 / 7, -13 / 7], 
     ![-6 / 7, -10 / 7] ]

-- Define the vectors
def vec1 : Fin 2 → ℚ := ![2, 3]
def vec2 : Fin 2 → ℚ := ![4, -1]

-- Expected results
def result1 : Fin 2 → ℚ := ![3, -6]
def result2 : Fin 2 → ℚ := ![19, -2]

-- The proof statement
theorem matrix_solution : (N.mulVec vec1 = result1) ∧ (N.mulVec vec2 = result2) :=
  by sorry

end matrix_solution_l185_185501


namespace problem_l185_185584

variable {a b c : ℝ} -- Introducing variables a, b, c as real numbers

-- Conditions:
-- a, b, c are distinct positive real numbers
def distinct_pos (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a 

theorem problem (h : distinct_pos a b c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
sorry 

end problem_l185_185584


namespace exists_function_f_l185_185443

theorem exists_function_f (f : ℕ → ℕ) : (∀ n : ℕ, f (f n) = n^2) → ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 :=
sorry

end exists_function_f_l185_185443


namespace houses_in_block_l185_185179

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) (h1 : junk_mail_per_house = 2) (h2 : total_junk_mail = 14) :
  total_junk_mail / junk_mail_per_house = 7 := by
  sorry

end houses_in_block_l185_185179


namespace total_campers_rowing_and_hiking_l185_185135

def campers_morning_rowing : ℕ := 41
def campers_morning_hiking : ℕ := 4
def campers_afternoon_rowing : ℕ := 26

theorem total_campers_rowing_and_hiking :
  campers_morning_rowing + campers_morning_hiking + campers_afternoon_rowing = 71 :=
by
  -- We are skipping the proof since instructions specify only the statement is needed
  sorry

end total_campers_rowing_and_hiking_l185_185135


namespace g_triple_of_10_l185_185357

def g (x : Int) : Int :=
  if x < 4 then x^2 - 9 else x + 7

theorem g_triple_of_10 : g (g (g 10)) = 31 := by
  sorry

end g_triple_of_10_l185_185357


namespace parallel_lines_slope_l185_185655

theorem parallel_lines_slope (b : ℚ) :
  (∀ x y : ℚ, 3 * y + x - 1 = 0 → 2 * y + b * x - 4 = 0 ∨
    3 * y + x - 1 = 0 ∧ 2 * y + b * x - 4 = 0) →
  b = 2 / 3 :=
by
  intro h
  sorry

end parallel_lines_slope_l185_185655


namespace engineers_to_designers_ratio_l185_185589

-- Define the given conditions for the problem
variables (e d : ℕ) -- e is the number of engineers, d is the number of designers
variables (h1 : (48 * e + 60 * d) / (e + d) = 52)

-- Theorem statement: The ratio of the number of engineers to the number of designers is 2:1
theorem engineers_to_designers_ratio (h1 : (48 * e + 60 * d) / (e + d) = 52) : e = 2 * d :=
by {
  sorry  
}

end engineers_to_designers_ratio_l185_185589


namespace blue_red_area_ratio_l185_185415

theorem blue_red_area_ratio (d_small d_large : ℕ) (h1 : d_small = 2) (h2 : d_large = 6) :
    let r_small := d_small / 2
    let r_large := d_large / 2
    let A_red := Real.pi * (r_small : ℝ) ^ 2
    let A_large := Real.pi * (r_large : ℝ) ^ 2
    let A_blue := A_large - A_red
    A_blue / A_red = 8 :=
by
  sorry

end blue_red_area_ratio_l185_185415


namespace minimum_value_g_l185_185378

noncomputable def g (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem minimum_value_g : ∀ x : ℝ, x > 0 → g x ≥ 7 :=
by
  intros x hx
  sorry

end minimum_value_g_l185_185378


namespace stone_solution_l185_185971

noncomputable def stone_problem : Prop :=
  ∃ y : ℕ, (∃ x z : ℕ, x + y + z = 100 ∧ x + 10 * y + 50 * z = 500) ∧
    ∀ y1 y2 : ℕ, (∃ x1 z1 : ℕ, x1 + y1 + z1 = 100 ∧ x1 + 10 * y1 + 50 * z1 = 500) ∧
                (∃ x2 z2 : ℕ, x2 + y2 + z2 = 100 ∧ x2 + 10 * y2 + 50 * z2 = 500) →
                y1 = y2

theorem stone_solution : stone_problem :=
sorry

end stone_solution_l185_185971


namespace painted_prisms_l185_185586

theorem painted_prisms (n : ℕ) (h : n > 2) :
  2 * ((n - 2) * (n - 1) + (n - 2) * n + (n - 1) * n) = (n - 2) * (n - 1) * n ↔ n = 7 :=
by sorry

end painted_prisms_l185_185586


namespace eraser_crayon_difference_l185_185339

def initial_crayons : Nat := 601
def initial_erasers : Nat := 406
def final_crayons : Nat := 336
def final_erasers : Nat := initial_erasers

theorem eraser_crayon_difference :
  final_erasers - final_crayons = 70 :=
by
  sorry

end eraser_crayon_difference_l185_185339


namespace temperature_on_tuesday_l185_185503

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th) / 3 = 45 →
  (W + Th + F) / 3 = 50 →
  F = 53 →
  T = 38 :=
by 
  intros h1 h2 h3
  sorry

end temperature_on_tuesday_l185_185503


namespace quadratic_inequality_solution_l185_185465

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + m > 0) ↔ m > 1 :=
by
  sorry

end quadratic_inequality_solution_l185_185465


namespace solve_for_x_l185_185310

theorem solve_for_x : (∃ x : ℝ, (x / 18) * (x / 72) = 1) → ∃ x : ℝ, x = 36 :=
by
  sorry

end solve_for_x_l185_185310


namespace exists_consecutive_divisible_by_cube_l185_185596

theorem exists_consecutive_divisible_by_cube (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ j : ℕ, j < k → ∃ m : ℕ, 1 < m ∧ (n + j) % (m^3) = 0 := 
sorry

end exists_consecutive_divisible_by_cube_l185_185596


namespace remainder_is_23_l185_185759

def number_remainder (n : ℤ) : ℤ :=
  n % 36

theorem remainder_is_23 (n : ℤ) (h1 : n % 4 = 3) (h2 : n % 9 = 5) :
  number_remainder n = 23 :=
by
  sorry

end remainder_is_23_l185_185759


namespace mans_speed_against_current_l185_185141

theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (h1 : speed_with_current = 25)
  (h2 : speed_of_current = 2.5) :
  speed_with_current - 2 * speed_of_current = 20 := 
by
  sorry

end mans_speed_against_current_l185_185141


namespace side_length_of_square_l185_185785

theorem side_length_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s = 2 ∧ d = s * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l185_185785


namespace cube_inequality_contradiction_l185_185760

variable {x y : ℝ}

theorem cube_inequality_contradiction (h : x < y) (hne : x^3 ≥ y^3) : false :=
by 
  sorry

end cube_inequality_contradiction_l185_185760


namespace masha_guessed_number_l185_185564

theorem masha_guessed_number (a b : ℕ) (h1 : a + b = 2002 ∨ a * b = 2002)
  (h2 : ∀ x y, x + y = 2002 → x ≠ 1001 → y ≠ 1001)
  (h3 : ∀ x y, x * y = 2002 → x ≠ 1001 → y ≠ 1001) :
  b = 1001 :=
by {
  sorry
}

end masha_guessed_number_l185_185564


namespace goal_amount_is_correct_l185_185694

def earnings_three_families : ℕ := 3 * 10
def earnings_fifteen_families : ℕ := 15 * 5
def total_earned : ℕ := earnings_three_families + earnings_fifteen_families
def goal_amount : ℕ := total_earned + 45

theorem goal_amount_is_correct : goal_amount = 150 :=
by
  -- We are aware of the proof steps but they are not required here
  sorry

end goal_amount_is_correct_l185_185694


namespace div_of_power_diff_div_l185_185338

theorem div_of_power_diff_div (a b n : ℕ) (h : a ≠ b) (h₀ : n ∣ (a^n - b^n)) : n ∣ (a^n - b^n) / (a - b) :=
  sorry

end div_of_power_diff_div_l185_185338


namespace rosie_pies_l185_185269

-- Definition of known conditions
def apples_per_pie (apples_pies_ratio : ℕ × ℕ) : ℕ :=
  apples_pies_ratio.1 / apples_pies_ratio.2

def pies_from_apples (total_apples : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total_apples / apples_per_pie

-- Theorem statement
theorem rosie_pies (apples_pies_ratio : ℕ × ℕ) (total_apples : ℕ) :
  apples_pies_ratio = (12, 3) →
  total_apples = 36 →
  pies_from_apples total_apples (apples_per_pie apples_pies_ratio) = 9 :=
by
  intros h_ratio h_apples
  rw [h_ratio, h_apples]
  sorry

end rosie_pies_l185_185269


namespace carlton_school_earnings_l185_185645

theorem carlton_school_earnings :
  let students_days_adams := 8 * 4
  let students_days_byron := 5 * 6
  let students_days_carlton := 6 * 10
  let total_wages := 1092
  students_days_adams + students_days_byron = 62 → 
  62 * (2 * x) + students_days_carlton * x = total_wages → 
  x = (total_wages : ℝ) / 184 → 
  (students_days_carlton : ℝ) * x = 356.09 := 
by
  intros _ _ _ 
  sorry

end carlton_school_earnings_l185_185645


namespace tan_435_eq_2_plus_sqrt3_l185_185758

open Real

theorem tan_435_eq_2_plus_sqrt3 : tan (435 * (π / 180)) = 2 + sqrt 3 :=
  sorry

end tan_435_eq_2_plus_sqrt3_l185_185758


namespace agatha_bike_budget_l185_185089

def total_initial : ℕ := 60
def cost_frame : ℕ := 15
def cost_front_wheel : ℕ := 25
def total_spent : ℕ := cost_frame + cost_front_wheel
def total_left : ℕ := total_initial - total_spent

theorem agatha_bike_budget : total_left = 20 := by
  sorry

end agatha_bike_budget_l185_185089


namespace rectangle_perimeter_l185_185169

variable (a b : ℕ)

theorem rectangle_perimeter (h1 : a ≠ b) (h2 : ab = 8 * (a + b)) : 
  2 * (a + b) = 66 := 
sorry

end rectangle_perimeter_l185_185169


namespace ratio_of_Lev_to_Akeno_l185_185789

theorem ratio_of_Lev_to_Akeno (L : ℤ) (A : ℤ) (Ambrocio : ℤ) :
  A = 2985 ∧ Ambrocio = L - 177 ∧ A = L + Ambrocio + 1172 → L / A = 1 / 3 :=
by
  intro h
  sorry

end ratio_of_Lev_to_Akeno_l185_185789


namespace number_of_doubles_players_l185_185482

theorem number_of_doubles_players (x y : ℕ) 
  (h1 : x + y = 13) 
  (h2 : 4 * x - 2 * y = 4) : 
  4 * x = 20 :=
by sorry

end number_of_doubles_players_l185_185482


namespace smallest_digit_for_divisibility_by_9_l185_185488

theorem smallest_digit_for_divisibility_by_9 : 
  ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ (18 + d) % 9 = 0 ∧ ∀ d' : ℕ, (0 ≤ d' ∧ d' ≤ 9 ∧ (18 + d') % 9 = 0) → d' ≥ d :=
sorry

end smallest_digit_for_divisibility_by_9_l185_185488


namespace crews_complete_job_l185_185138

-- Define the productivity rates for each crew
variables (x y z : ℝ)

-- Define the conditions derived from the problem
def condition1 : Prop := 1/(x + y) = 1/z - 3/5
def condition2 : Prop := 1/(x + z) = 1/y
def condition3 : Prop := 1/(y + z) = 2/(7 * x)

-- Target proof: the combined time for all three crews
def target_proof : Prop := 1/(x + y + z) = 4/3

-- Final Lean 4 statement combining all conditions and proof requirement
theorem crews_complete_job (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : target_proof x y z :=
sorry

end crews_complete_job_l185_185138


namespace point_A_coordinates_l185_185435

-- Given conditions
def point_A (a : ℝ) : ℝ × ℝ := (a + 1, a^2 - 4)
def negative_half_x_axis (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 = 0

-- Theorem statement
theorem point_A_coordinates (a : ℝ) (h : negative_half_x_axis (point_A a)) :
  point_A a = (-1, 0) :=
sorry

end point_A_coordinates_l185_185435


namespace rectangle_area_l185_185090

theorem rectangle_area (L W : ℕ) (h1 : 2 * L + 2 * W = 280) (h2 : L = 5 * (W / 2)) : L * W = 4000 :=
sorry

end rectangle_area_l185_185090


namespace eval_expr_at_sqrt3_minus_3_l185_185257

noncomputable def expr (a : ℝ) : ℝ :=
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2))

theorem eval_expr_at_sqrt3_minus_3 : expr (Real.sqrt 3 - 3) = -Real.sqrt 3 / 6 := 
  by sorry

end eval_expr_at_sqrt3_minus_3_l185_185257


namespace vector_difference_perpendicular_l185_185550

/-- Proof that the vector difference a - b is perpendicular to b given specific vectors a and b -/
theorem vector_difference_perpendicular {a b : ℝ × ℝ} (h_a : a = (2, 0)) (h_b : b = (1, 1)) :
  (a - b) • b = 0 :=
by
  sorry

end vector_difference_perpendicular_l185_185550


namespace min_folds_exceed_12mm_l185_185666

theorem min_folds_exceed_12mm : ∃ n : ℕ, 0.1 * (2: ℝ)^n > 12 ∧ ∀ m < n, 0.1 * (2: ℝ)^m ≤ 12 := 
by
  sorry

end min_folds_exceed_12mm_l185_185666


namespace probability_of_event_l185_185799

-- Definitions for the problem setup

-- Box C and its range
def boxC := {i : ℕ | 1 ≤ i ∧ i ≤ 30}

-- Box D and its range
def boxD := {i : ℕ | 21 ≤ i ∧ i ≤ 50}

-- Condition for a tile from box C being less than 20
def tile_from_C_less_than_20 (i : ℕ) : Prop := i ∈ boxC ∧ i < 20

-- Condition for a tile from box D being odd or greater than 45
def tile_from_D_odd_or_greater_than_45 (i : ℕ) : Prop := i ∈ boxD ∧ (i % 2 = 1 ∨ i > 45)

-- Main statement
theorem probability_of_event :
  (19 / 30 : ℚ) * (17 / 30 : ℚ) = (323 / 900 : ℚ) :=
by sorry

end probability_of_event_l185_185799


namespace cylinder_ratio_l185_185008

theorem cylinder_ratio
  (V : ℝ) (r h : ℝ)
  (h_volume : π * r^2 * h = V)
  (h_surface_area : 2 * π * r * h = 2 * (V / r)) :
  h / r = 2 :=
sorry

end cylinder_ratio_l185_185008


namespace alpha3_plus_8beta_plus_6_eq_30_l185_185273

noncomputable def alpha_beta_quad_roots (α β : ℝ) : Prop :=
  α^2 - 2 * α - 4 = 0 ∧ β^2 - 2 * β - 4 = 0

theorem alpha3_plus_8beta_plus_6_eq_30 (α β : ℝ) (h : alpha_beta_quad_roots α β) : 
  α^3 + 8 * β + 6 = 30 :=
sorry

end alpha3_plus_8beta_plus_6_eq_30_l185_185273


namespace eval_expression_eq_54_l185_185396

theorem eval_expression_eq_54 : (3 * 4 * 6) * ((1/3 : ℚ) + 1/4 + 1/6) = 54 := 
by
  sorry

end eval_expression_eq_54_l185_185396


namespace scientific_notation_4040000_l185_185788

theorem scientific_notation_4040000 :
  (4040000 : ℝ) = 4.04 * (10 : ℝ)^6 :=
by
  sorry

end scientific_notation_4040000_l185_185788


namespace eval_expression_l185_185496

theorem eval_expression {p q r s : ℝ} 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := 
by 
  sorry

end eval_expression_l185_185496


namespace condition_B_is_necessary_but_not_sufficient_l185_185126

-- Definitions of conditions A and B
def condition_A (x : ℝ) : Prop := 0 < x ∧ x < 5
def condition_B (x : ℝ) : Prop := abs (x - 2) < 3

-- The proof problem statement
theorem condition_B_is_necessary_but_not_sufficient : 
∀ x, condition_A x → condition_B x ∧ ¬(∀ x, condition_B x → condition_A x) := 
sorry

end condition_B_is_necessary_but_not_sufficient_l185_185126


namespace max_z_val_l185_185417

theorem max_z_val (x y : ℝ) (h1 : x + y ≤ 4) (h2 : y - 2 * x + 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ x y, z = x + 2 * y ∧ z = 6 :=
by
  sorry

end max_z_val_l185_185417


namespace food_drive_ratio_l185_185282

/-- Mark brings in 4 times as many cans as Jaydon,
Jaydon brings in 5 more cans than a certain multiple of the amount of cans that Rachel brought in,
There are 135 cans total, and Mark brought in 100 cans.
Prove that the ratio of the number of cans Jaydon brought in to the number of cans Rachel brought in is 5:2. -/
theorem food_drive_ratio (J R : ℕ) (k : ℕ)
  (h1 : 4 * J = 100)
  (h2 : J = k * R + 5)
  (h3 : 100 + J + R = 135) :
  J / Nat.gcd J R = 5 ∧ R / Nat.gcd J R = 2 := by
  sorry

end food_drive_ratio_l185_185282


namespace games_bought_at_garage_sale_l185_185742

theorem games_bought_at_garage_sale (G : ℕ)
  (h1 : 2 + G - 2  = 2) :
  G = 2 :=
by {
  sorry
}

end games_bought_at_garage_sale_l185_185742


namespace value_of_b_l185_185202

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end value_of_b_l185_185202


namespace sum_ABC_eq_7_base_8_l185_185738

/-- Lean 4 statement for the problem.

A, B, C: are distinct non-zero digits less than 8 in base 8, and
A B C_8 + B C_8 = A C A_8 holds true.
-/
theorem sum_ABC_eq_7_base_8 :
  ∃ (A B C : ℕ), A < 8 ∧ B < 8 ∧ C < 8 ∧ 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (A * 64 + B * 8 + C) + (B * 8 + C) = A * 64 + C * 8 + A ∧
  A + B + C = 7 :=
by { sorry }

end sum_ABC_eq_7_base_8_l185_185738


namespace Derek_more_than_Zoe_l185_185100

-- Define the variables for the number of books Emily, Derek, and Zoe have
variables (E : ℝ)

-- Condition: Derek has 75% more books than Emily
def Derek_books : ℝ := 1.75 * E

-- Condition: Zoe has 50% more books than Emily
def Zoe_books : ℝ := 1.5 * E

-- Statement asserting that Derek has 16.67% more books than Zoe
theorem Derek_more_than_Zoe (hD: Derek_books E = 1.75 * E) (hZ: Zoe_books E = 1.5 * E) :
  (Derek_books E - Zoe_books E) / Zoe_books E = 0.1667 :=
by
  sorry

end Derek_more_than_Zoe_l185_185100


namespace new_students_count_l185_185170

theorem new_students_count (O N : ℕ) (avg_class_age avg_new_students_age avg_decrease original_strength : ℕ)
  (h1 : avg_class_age = 40)
  (h2 : avg_new_students_age = 32)
  (h3 : avg_decrease = 4)
  (h4 : original_strength = 8)
  (total_age_class : ℕ := avg_class_age * original_strength)
  (new_avg_age : ℕ := avg_class_age - avg_decrease)
  (total_age_new_students : ℕ := avg_new_students_age * N)
  (total_students : ℕ := original_strength + N)
  (new_total_age : ℕ := total_age_class + total_age_new_students)
  (new_avg_class_age : ℕ := new_total_age / total_students)
  (h5 : new_avg_class_age = new_avg_age) : N = 8 :=
by
  sorry

end new_students_count_l185_185170


namespace compute_fraction_product_l185_185217

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l185_185217


namespace parallel_heater_time_l185_185871

theorem parallel_heater_time (t1 t2 : ℕ) (R1 R2 : ℝ) (t : ℕ) (I : ℝ) (Q : ℝ) (h₁ : t1 = 3) 
  (h₂ : t2 = 6) (hq1 : Q = I^2 * R1 * t1) (hq2 : Q = I^2 * R2 * t2) :
  t = (t1 * t2) / (t1 + t2) := by
  sorry

end parallel_heater_time_l185_185871


namespace total_candies_l185_185782

variable (Adam James Rubert : Nat)
variable (Adam_has_candies : Adam = 6)
variable (James_has_candies : James = 3 * Adam)
variable (Rubert_has_candies : Rubert = 4 * James)

theorem total_candies : Adam + James + Rubert = 96 :=
by
  sorry

end total_candies_l185_185782


namespace cos_double_angle_l185_185067

theorem cos_double_angle (α : ℝ) (h : Real.sin α = (Real.sqrt 3) / 2) : 
  Real.cos (2 * α) = -1 / 2 :=
by
  sorry

end cos_double_angle_l185_185067


namespace min_speed_x_l185_185393

theorem min_speed_x (V_X : ℝ) : 
  let relative_speed_xy := V_X + 40;
  let relative_speed_xz := V_X - 30;
  (500 / relative_speed_xy) > (300 / relative_speed_xz) → 
  V_X ≥ 136 :=
by
  intros;
  sorry

end min_speed_x_l185_185393


namespace sum_of_areas_of_squares_l185_185919

def is_right_angle (a b c : ℝ) : Prop := (a^2 + b^2 = c^2)

def isSquare (side : ℝ) : Prop := (side > 0)

def area_of_square (side : ℝ) : ℝ := side^2

theorem sum_of_areas_of_squares 
  (P Q R S X Y : ℝ) 
  (h1 : is_right_angle P Q R)
  (h2 : PR = 15)
  (h3 : isSquare PR)
  (h4 : isSquare PQ) :
  area_of_square PR + area_of_square PQ = 450 := 
sorry


end sum_of_areas_of_squares_l185_185919


namespace infinite_series_sum_l185_185080

theorem infinite_series_sum :
  ∑' (k : ℕ), (k + 1) / 4^(k + 1) = 4 / 9 :=
sorry

end infinite_series_sum_l185_185080


namespace min_cos_beta_l185_185044

open Real

theorem min_cos_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin (2 * α + β) = (3 / 2) * sin β) :
  cos β = sqrt 5 / 3 := 
sorry

end min_cos_beta_l185_185044


namespace initial_percentage_decrease_l185_185121

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₀ : P > 0)
  (initial_decrease : ∀ (x : ℝ), P * (1 - x / 100) * 1.3 = P * 1.04) :
  x = 20 :=
by 
  sorry

end initial_percentage_decrease_l185_185121


namespace total_tiles_l185_185696

-- Define the dimensions
def length : ℕ := 16
def width : ℕ := 12

-- Define the number of 1-foot by 1-foot tiles for the border
def tiles_border : ℕ := (2 * length + 2 * width - 4)

-- Define the inner dimensions
def inner_length : ℕ := length - 2
def inner_width : ℕ := width - 2

-- Define the number of 2-foot by 2-foot tiles for the interior
def tiles_interior : ℕ := (inner_length * inner_width) / 4

-- Prove that the total number of tiles is 87
theorem total_tiles : tiles_border + tiles_interior = 87 := by
  sorry

end total_tiles_l185_185696


namespace unique_a_for_set_A_l185_185344

def A (a : ℝ) : Set ℝ := {a^2, 2 - a, 4}

theorem unique_a_for_set_A (a : ℝ) : A a = {x : ℝ // x = a^2 ∨ x = 2 - a ∨ x = 4} → a = -1 :=
by
  sorry

end unique_a_for_set_A_l185_185344


namespace balls_is_perfect_square_l185_185731

open Classical -- Open classical logic for nonconstructive proofs

-- Define a noncomputable function to capture the main proof argument
noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem balls_is_perfect_square {a v : ℕ} (h : (2 * a * v) = (a + v) * (a + v - 1))
  : is_perfect_square (a + v) :=
sorry

end balls_is_perfect_square_l185_185731


namespace marina_total_cost_l185_185275

theorem marina_total_cost (E P R X : ℕ) 
    (h1 : 15 + E + P = 47)
    (h2 : 15 + R + X = 58) :
    15 + E + P + R + X = 90 :=
by
  -- The proof will go here
  sorry

end marina_total_cost_l185_185275


namespace cannot_tile_regular_pentagon_l185_185949

theorem cannot_tile_regular_pentagon :
  ¬ (∃ n : ℕ, 360 % (180 - (360 / 5 : ℕ)) = 0) :=
by sorry

end cannot_tile_regular_pentagon_l185_185949


namespace length_of_train_l185_185343

theorem length_of_train (speed_kmh : ℕ) (time_s : ℕ) (length_bridge_m : ℕ) (length_train_m : ℕ) :
  speed_kmh = 45 → time_s = 30 → length_bridge_m = 275 → length_train_m = 475 :=
by
  intros h1 h2 h3
  sorry

end length_of_train_l185_185343


namespace ellen_dinner_calories_proof_l185_185923

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l185_185923


namespace tourists_went_free_l185_185000

theorem tourists_went_free (x : ℕ) : 
  (13 + 4 * x = x + 100) → x = 29 :=
by
  intros h
  sorry

end tourists_went_free_l185_185000


namespace calculate_expression_l185_185692

theorem calculate_expression : 
  - 3 ^ 2 + (-12) * abs (-1/2) - 6 / (-1) = -9 := 
by 
  sorry

end calculate_expression_l185_185692


namespace odd_expression_divisible_by_48_l185_185933

theorem odd_expression_divisible_by_48 (x : ℤ) (h : Odd x) : 48 ∣ (x^3 + 3*x^2 - x - 3) :=
  sorry

end odd_expression_divisible_by_48_l185_185933


namespace total_amount_from_grandparents_l185_185424

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end total_amount_from_grandparents_l185_185424


namespace area_of_ABCD_l185_185585

theorem area_of_ABCD (area_AMOP area_CNOQ : ℝ) 
  (h1: area_AMOP = 8) (h2: area_CNOQ = 24.5) : 
  ∃ (area_ABCD : ℝ), area_ABCD = 60.5 :=
by
  sorry

end area_of_ABCD_l185_185585


namespace ben_chairs_in_10_days_l185_185689

def number_of_chairs (days hours_per_shift hours_rocking_chair hours_dining_chair hours_armchair : ℕ) : ℕ × ℕ × ℕ :=
  let rocking_chairs_per_day := hours_per_shift / hours_rocking_chair
  let remaining_hours_after_rocking_chairs := hours_per_shift % hours_rocking_chair
  let dining_chairs_per_day := remaining_hours_after_rocking_chairs / hours_dining_chair
  let remaining_hours_after_dining_chairs := remaining_hours_after_rocking_chairs % hours_dining_chair
  if remaining_hours_after_dining_chairs >= hours_armchair then
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, days * (remaining_hours_after_dining_chairs / hours_armchair))
  else
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, 0)

theorem ben_chairs_in_10_days :
  number_of_chairs 10 8 5 3 6 = (10, 10, 0) :=
by 
  sorry

end ben_chairs_in_10_days_l185_185689


namespace vaccination_target_failure_l185_185065

noncomputable def percentage_vaccination_target_failed (original_target : ℕ) (first_year : ℕ) (second_year_increase_rate : ℚ) (third_year : ℕ) : ℚ :=
  let second_year := first_year + second_year_increase_rate * first_year
  let total_vaccinated := first_year + second_year + third_year
  let shortfall := original_target - total_vaccinated
  (shortfall / original_target) * 100

theorem vaccination_target_failure :
  percentage_vaccination_target_failed 720 60 (65/100 : ℚ) 150 = 57.11 := 
  by sorry

end vaccination_target_failure_l185_185065


namespace total_amount_spent_by_jim_is_50_l185_185151

-- Definitions for conditions
def cost_per_gallon_nc : ℝ := 2.00  -- Cost per gallon in North Carolina
def gallons_nc : ℕ := 10  -- Gallons bought in North Carolina
def additional_cost_per_gallon_va : ℝ := 1.00  -- Additional cost per gallon in Virginia
def gallons_va : ℕ := 10  -- Gallons bought in Virginia

-- Definition for total cost in North Carolina
def total_cost_nc : ℝ := gallons_nc * cost_per_gallon_nc

-- Definition for cost per gallon in Virginia
def cost_per_gallon_va : ℝ := cost_per_gallon_nc + additional_cost_per_gallon_va

-- Definition for total cost in Virginia
def total_cost_va : ℝ := gallons_va * cost_per_gallon_va

-- Definition for total amount spent
def total_spent : ℝ := total_cost_nc + total_cost_va

-- Theorem to prove
theorem total_amount_spent_by_jim_is_50 : total_spent = 50.00 :=
by
  -- Place proof here
  sorry

end total_amount_spent_by_jim_is_50_l185_185151


namespace parallel_or_identical_lines_l185_185542

theorem parallel_or_identical_lines (a b c d e f : ℝ) :
  2 * b - 3 * a = 15 → 4 * d - 6 * c = 18 → (b ≠ d → a = c) :=
by
  intros h1 h2 hneq
  sorry

end parallel_or_identical_lines_l185_185542


namespace tyrone_gave_25_marbles_l185_185624

/-- Given that Tyrone initially had 97 marbles and Eric had 11 marbles, and after
    giving some marbles to Eric, Tyrone ended with twice as many marbles as Eric,
    we need to find the number of marbles Tyrone gave to Eric. -/
theorem tyrone_gave_25_marbles (x : ℕ) (t0 e0 : ℕ)
  (hT0 : t0 = 97)
  (hE0 : e0 = 11)
  (hT_end : (t0 - x) = 2 * (e0 + x)) :
  x = 25 := 
  sorry

end tyrone_gave_25_marbles_l185_185624


namespace probability_of_yellow_light_l185_185315

def time_red : ℕ := 30
def time_green : ℕ := 25
def time_yellow : ℕ := 5
def total_cycle_time : ℕ := time_red + time_green + time_yellow

theorem probability_of_yellow_light :
  (time_yellow : ℚ) / (total_cycle_time : ℚ) = 1 / 12 :=
by
  sorry

end probability_of_yellow_light_l185_185315


namespace range_of_a_l185_185821

def solution_set_non_empty (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - 3| + |x - 4| < a

theorem range_of_a (a : ℝ) : solution_set_non_empty a ↔ a > 1 := sorry

end range_of_a_l185_185821


namespace max_number_of_band_members_l185_185058

-- Conditions definitions
def num_band_members (r x : ℕ) : ℕ := r * x + 3

def num_band_members_new (r x : ℕ) : ℕ := (r - 1) * (x + 2)

-- The main statement
theorem max_number_of_band_members :
  ∃ (r x : ℕ), num_band_members r x = 231 ∧ num_band_members_new r x = 231 
  ∧ ∀ (r' x' : ℕ), (num_band_members r' x' < 120 ∧ num_band_members_new r' x' = num_band_members r' x') → (num_band_members r' x' ≤ 231) :=
sorry

end max_number_of_band_members_l185_185058


namespace pyramid_surface_area_l185_185905

theorem pyramid_surface_area (base_edge volume : ℝ)
  (h_base_edge : base_edge = 1)
  (h_volume : volume = 1) :
  let height := 3
  let slant_height := Real.sqrt (9.25)
  let base_area := base_edge * base_edge
  let lateral_area := 4 * (1 / 2 * base_edge * slant_height)
  let total_surface_area := base_area + lateral_area
  total_surface_area = 7.082 :=
by
  sorry

end pyramid_surface_area_l185_185905


namespace sum_of_remainders_mod_30_l185_185287

theorem sum_of_remainders_mod_30 (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 11) (h3 : c % 30 = 19) :
  (a + b + c) % 30 = 14 :=
by
  sorry

end sum_of_remainders_mod_30_l185_185287


namespace correct_propositions_l185_185794

def line (P: Type) := P → P → Prop  -- A line is a relation between points in a plane

variables (plane1 plane2: Type) -- Define two types representing two planes
variables (P1 P2: plane1) -- Points in plane1
variables (Q1 Q2: plane2) -- Points in plane2

axiom perpendicular_planes : ¬∃ l1 : line plane1, ∀ l2 : line plane2, ¬ (∀ p1 p2, l1 p1 p2 ∧ ∀ q1 q2, l2 q1 q2)

theorem correct_propositions : 3 = 3 := by
  sorry

end correct_propositions_l185_185794


namespace line_perpendicular_passing_through_point_l185_185725

theorem line_perpendicular_passing_through_point :
  ∃ (a b c : ℝ), (∀ (x y : ℝ), 2 * x + y - 2 = 0 ↔ a * x + b * y + c = 0) ∧ 
                (a, b) ≠ (0, 0) ∧ 
                (a * -1 + b * 4 + c = 0) ∧ 
                (a * 1/2 + b * (-2) ≠ -4) :=
by { sorry }

end line_perpendicular_passing_through_point_l185_185725


namespace cube_root_of_64_eq_two_pow_m_l185_185251

theorem cube_root_of_64_eq_two_pow_m (m : ℕ) (h : (64 : ℝ) ^ (1 / 3) = (2 : ℝ) ^ m) : m = 2 := 
sorry

end cube_root_of_64_eq_two_pow_m_l185_185251


namespace negation_of_existence_l185_185957

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 > 2) ↔ ∀ x : ℝ, x^2 ≤ 2 :=
by
  sorry

end negation_of_existence_l185_185957


namespace equivalent_problem_l185_185822

noncomputable def problem_statement : Prop :=
  ∀ (a b c d : ℝ), a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ∀ (ω : ℂ), ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / (1 + ω)) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2)

#check problem_statement

-- Expected output for type checking without providing the proof
theorem equivalent_problem : problem_statement :=
  sorry

end equivalent_problem_l185_185822


namespace hyperbola_midpoint_l185_185300

theorem hyperbola_midpoint (x1 y1 x2 y2 : ℝ) :
  (x1^2 - y1^2 / 9 = 1) →
  (x2^2 - y2^2 / 9 = 1) →
  ((x1 + x2) / 2 = -1) →
  ((y1 + y2) / 2 = -4) →
  True :=
by
  intro h1 h2 hx hy
  sorry

end hyperbola_midpoint_l185_185300


namespace gymnast_score_difference_l185_185154

theorem gymnast_score_difference 
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x2 + x3 + x4 + x5 = 36)
  (h2 : x1 + x2 + x3 + x4 = 36.8) :
  x1 - x5 = 0.8 :=
by sorry

end gymnast_score_difference_l185_185154


namespace factorization_correct_l185_185844

theorem factorization_correct (a x y : ℝ) : a * x - a * y = a * (x - y) := by sorry

end factorization_correct_l185_185844


namespace peanuts_remaining_l185_185363

theorem peanuts_remaining (initial_peanuts brock_ate bonita_ate brock_fraction : ℕ) (h_initial : initial_peanuts = 148) (h_brock_fraction : brock_fraction = 4) (h_brock_ate : brock_ate = initial_peanuts / brock_fraction) (h_bonita_ate : bonita_ate = 29) :
  (initial_peanuts - brock_ate - bonita_ate) = 82 :=
by
  sorry

end peanuts_remaining_l185_185363


namespace num_digits_abc_l185_185208

theorem num_digits_abc (a b c : ℕ) (n : ℕ) (h_a : 10^(n-1) ≤ a ∧ a < 10^n) (h_b : 10^(n-1) ≤ b ∧ b < 10^n) (h_c : 10^(n-1) ≤ c ∧ c < 10^n) :
  ¬ ((Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 1) ∧
     (Int.natAbs ((10^(n-1) : ℕ) * (10^(n-1) : ℕ) * (10^(n-1) : ℕ)) + 1 = 3*n - 2)) :=
sorry

end num_digits_abc_l185_185208


namespace higher_selling_price_is_463_l185_185820

-- Definitions and conditions
def cost_price : ℝ := 400
def selling_price_340 : ℝ := 340
def loss_340 : ℝ := selling_price_340 - cost_price
def gain_percent : ℝ := 0.05
def additional_gain : ℝ := gain_percent * -loss_340
def expected_gain := -loss_340 + additional_gain

-- Theorem to prove that the higher selling price is 463
theorem higher_selling_price_is_463 : ∃ P : ℝ, P = cost_price + expected_gain ∧ P = 463 :=
by
  sorry

end higher_selling_price_is_463_l185_185820


namespace cristina_catches_nicky_l185_185889

-- Definitions from the conditions
def cristina_speed : ℝ := 4 -- meters per second
def nicky_speed : ℝ := 3 -- meters per second
def nicky_head_start : ℝ := 36 -- meters

-- The proof to find the time 't'
theorem cristina_catches_nicky (t : ℝ) : cristina_speed * t = nicky_head_start + nicky_speed * t -> t = 36 := by
  intros h
  sorry

end cristina_catches_nicky_l185_185889


namespace necessary_but_not_sufficient_condition_l185_185461

theorem necessary_but_not_sufficient_condition (p q : ℝ → Prop)
    (h₁ : ∀ x k, p x ↔ x ≥ k) 
    (h₂ : ∀ x, q x ↔ 3 / (x + 1) < 1) 
    (h₃ : ∃ k : ℝ, ∀ x, p x → q x ∧ ¬ (q x → p x)) :
  ∃ k, k > 2 :=
by
  sorry

end necessary_but_not_sufficient_condition_l185_185461


namespace problem_1_problem_2_l185_185239

-- Statements for our proof problems
theorem problem_1 (a b : ℝ) : a^2 + b^2 ≥ 2 * (2 * a - b) - 5 :=
sorry

theorem problem_2 (a b : ℝ) :
  a^a * b^b ≥ (a * b)^((a + b) / 2) ∧ (a = b ↔ a^a * b^b = (a * b)^((a + b) / 2)) :=
sorry

end problem_1_problem_2_l185_185239


namespace evaluate_product_l185_185936

theorem evaluate_product (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5 * n^3 + 4 * n^2 + 4 * n := 
by
  -- Omitted proof steps
  sorry

end evaluate_product_l185_185936


namespace find_x_l185_185312

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ↔ x = 80 / 3 := by
  sorry

end find_x_l185_185312


namespace find_constants_l185_185099

noncomputable def f (x m n : ℝ) := (m * x + 1) / (x + n)

theorem find_constants (m n : ℝ) (h_symm : ∀ x y, f x m n = y → f (4 - x) m n = 8 - y) : 
  m = 4 ∧ n = -2 := 
by
  sorry

end find_constants_l185_185099


namespace calc_molecular_weight_l185_185695

/-- Atomic weights in g/mol -/
def atomic_weight (e : String) : Float :=
  match e with
  | "Ca"   => 40.08
  | "O"    => 16.00
  | "H"    => 1.01
  | "Al"   => 26.98
  | "S"    => 32.07
  | "K"    => 39.10
  | "N"    => 14.01
  | _      => 0.0

/-- Molecular weight calculation for specific compounds -/
def molecular_weight (compound : String) : Float :=
  match compound with
  | "Ca(OH)2"     => atomic_weight "Ca" + 2 * atomic_weight "O" + 2 * atomic_weight "H"
  | "Al2(SO4)3"   => 2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")
  | "KNO3"        => atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"
  | _             => 0.0

/-- Given moles of different compounds, calculate the total molecular weight -/
def total_molecular_weight (moles : List (String × Float)) : Float :=
  moles.foldl (fun acc (compound, n) => acc + n * molecular_weight compound) 0.0

/-- The given problem -/
theorem calc_molecular_weight :
  total_molecular_weight [("Ca(OH)2", 4), ("Al2(SO4)3", 2), ("KNO3", 3)] = 1284.07 :=
by
  sorry

end calc_molecular_weight_l185_185695


namespace least_number_with_remainders_l185_185614

theorem least_number_with_remainders :
  ∃ x, (x ≡ 4 [MOD 5]) ∧ (x ≡ 4 [MOD 6]) ∧ (x ≡ 4 [MOD 9]) ∧ (x ≡ 4 [MOD 18]) ∧ x = 94 := 
by 
  sorry

end least_number_with_remainders_l185_185614


namespace assignment_ways_l185_185970

-- Definitions
def graduates := 5
def companies := 3

-- Statement to be proven
theorem assignment_ways :
  ∃ (ways : ℕ), ways = 150 :=
sorry

end assignment_ways_l185_185970


namespace redesigned_lock_additional_combinations_l185_185625

-- Definitions for the problem conditions
def original_combinations : ℕ := Nat.choose 10 5
def total_new_combinations : ℕ := (Finset.range 10).sum (λ k => Nat.choose 10 (k + 1)) 
def additional_combinations := total_new_combinations - original_combinations - 2 -- subtract combinations for 0 and 10

-- Statement of the theorem
theorem redesigned_lock_additional_combinations : additional_combinations = 770 :=
by
  -- Proof omitted (insert 'sorry' to indicate incomplete proof state)
  sorry

end redesigned_lock_additional_combinations_l185_185625


namespace base_conversion_b_eq_3_l185_185546

theorem base_conversion_b_eq_3 (b : ℕ) (hb : b > 0) :
  (3 * 6^1 + 5 * 6^0 = 23) →
  (1 * b^2 + 3 * b + 2 = 23) →
  b = 3 :=
by {
  sorry
}

end base_conversion_b_eq_3_l185_185546


namespace min_total_books_l185_185772

-- Definitions based on conditions
variables (P C B : ℕ)

-- Condition 1: Ratio of physics to chemistry books is 3:2
def ratio_physics_chemistry := 3 * C = 2 * P

-- Condition 2: Ratio of chemistry to biology books is 4:3
def ratio_chemistry_biology := 4 * B = 3 * C

-- Condition 3: Total number of books is 3003
def total_books := P + C + B = 3003

-- The theorem to prove
theorem min_total_books (h1 : ratio_physics_chemistry P C) (h2 : ratio_chemistry_biology C B) (h3: total_books P C B) :
  3003 = 3003 :=
by
  sorry

end min_total_books_l185_185772


namespace minimum_distance_from_parabola_to_circle_l185_185715

noncomputable def minimum_distance_sum : ℝ :=
  let focus : ℝ × ℝ := (1, 0)
  let center : ℝ × ℝ := (0, 4)
  let radius : ℝ := 1
  let distance_from_focus_to_center : ℝ := Real.sqrt ((focus.1 - center.1)^2 + (focus.2 - center.2)^2)
  distance_from_focus_to_center - radius

theorem minimum_distance_from_parabola_to_circle : minimum_distance_sum = Real.sqrt 17 - 1 := by
  sorry

end minimum_distance_from_parabola_to_circle_l185_185715


namespace problem_solution_l185_185361

open Real

def system_satisfied (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    (a (2 * k + 1) = (1 / a (2 * (k + n) - 1) + 1 / a (2 * k + 2))) ∧ 
    (a (2 * k + 2) = a (2 * k + 1) + a (2 * k + 3))

theorem problem_solution (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 4)
  (h2 : ∀ k, 0 ≤ k → k < 2 * n → a k > 0)
  (h3 : system_satisfied a n) :
  ∀ k, 0 ≤ k ∧ k < n → a (2 * k + 1) = 1 ∧ a (2 * k + 2) = 2 :=
sorry

end problem_solution_l185_185361


namespace find_nat_numbers_l185_185272

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

theorem find_nat_numbers (n : ℕ) :
  (n + sum_of_digits n = 2021) ↔ (n = 2014 ∨ n = 1996) :=
by
  sorry

end find_nat_numbers_l185_185272


namespace call_center_agents_ratio_l185_185298

noncomputable def fraction_of_agents (calls_A calls_B total_agents total_calls : ℕ) : ℚ :=
  let calls_A_per_agent := calls_A / total_agents
  let calls_B_per_agent := calls_B / total_agents
  let ratio_calls_A_B := (3: ℚ) / 5
  let fraction_calls_B := (8: ℚ) / 11
  let fraction_calls_A := (3: ℚ) / 11
  let ratio_of_agents := (5: ℚ) / 11
  if (calls_A_per_agent * fraction_calls_A = ratio_calls_A_B * calls_B_per_agent) then ratio_of_agents else 0

theorem call_center_agents_ratio (calls_A calls_B total_agents total_calls agents_A agents_B : ℕ) :
  (calls_A : ℚ) / (calls_B : ℚ) = (3 / 5) →
  (calls_B : ℚ) = (8 / 11) * total_calls →
  (agents_A : ℚ) = (5 / 11) * (agents_B : ℚ) :=
sorry

end call_center_agents_ratio_l185_185298


namespace hyperbola_focal_length_l185_185279

theorem hyperbola_focal_length (m : ℝ) (h_eq : m * x^2 + 2 * y^2 = 2) (h_imag_axis : -2 / m = 4) : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 := 
sorry

end hyperbola_focal_length_l185_185279


namespace decision_represented_by_D_l185_185074

-- Define the basic symbols in the flowchart
inductive BasicSymbol
| Start
| Process
| Decision
| End

open BasicSymbol

-- Define the meaning of each basic symbol
def meaning_of (sym : BasicSymbol) : String :=
  match sym with
  | Start => "start"
  | Process => "process"
  | Decision => "decision"
  | End => "end"

-- The theorem stating that the Decision symbol represents a decision
theorem decision_represented_by_D : meaning_of Decision = "decision" :=
by sorry

end decision_represented_by_D_l185_185074


namespace lcm_210_297_l185_185113

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := 
by sorry

end lcm_210_297_l185_185113


namespace cone_height_ratio_l185_185860

theorem cone_height_ratio (C : ℝ) (h₁ : ℝ) (V₂ : ℝ) (r : ℝ) (h₂ : ℝ) :
  C = 20 * Real.pi → 
  h₁ = 40 →
  V₂ = 400 * Real.pi →
  2 * Real.pi * r = 20 * Real.pi →
  V₂ = (1 / 3) * Real.pi * r^2 * h₂ →
  h₂ / h₁ = (3 / 10) := by
sorry

end cone_height_ratio_l185_185860


namespace coeff_x4_expansion_l185_185430

def binom_expansion (a : ℚ) : ℚ :=
  let term1 : ℚ := a * 28
  let term2 : ℚ := -56
  term1 + term2

theorem coeff_x4_expansion (a : ℚ) : (binom_expansion a = -42) → a = 1/2 := 
by 
  intro h
  -- continuation of proof will go here.
  sorry

end coeff_x4_expansion_l185_185430


namespace ana_multiplied_numbers_l185_185203

theorem ana_multiplied_numbers (x : ℕ) (y : ℕ) 
    (h_diff : y = x + 202) 
    (h_mistake : x * y - 1000 = 288 * x + 67) :
    x = 97 ∧ y = 299 :=
sorry

end ana_multiplied_numbers_l185_185203


namespace find_ordered_pair_l185_185746

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 3 * x - 18 * y = 2) 
  (h2 : 4 * y - x = 6) :
  x = -58 / 3 ∧ y = -10 / 3 :=
sorry

end find_ordered_pair_l185_185746


namespace quadratic_always_positive_l185_185091

theorem quadratic_always_positive (x : ℝ) : x^2 + x + 1 > 0 :=
sorry

end quadratic_always_positive_l185_185091


namespace solve_inequality_l185_185230

theorem solve_inequality (x : ℝ) : 
  (0 < (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6))) ↔ 
  (x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x) :=
by 
  sorry

end solve_inequality_l185_185230


namespace calculate_expression_l185_185351
open Complex

-- Define the given values for a and b
def a := 3 + 2 * Complex.I
def b := 2 - 3 * Complex.I

-- Define the target expression
def target := 3 * a + 4 * b

-- The statement asserts that the target expression equals the expected result
theorem calculate_expression : target = 17 - 6 * Complex.I := by
  sorry

end calculate_expression_l185_185351


namespace difference_between_numbers_l185_185148

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 24365) (h2 : a % 5 = 0) (h3 : (a / 10) = 2 * b) : a - b = 19931 :=
by sorry

end difference_between_numbers_l185_185148


namespace jerry_age_l185_185570

theorem jerry_age (M J : ℤ) (h1 : M = 16) (h2 : M = 2 * J - 8) : J = 12 :=
by
  sorry

end jerry_age_l185_185570


namespace proof_x_exists_l185_185699

noncomputable def find_x : ℝ := 33.33

theorem proof_x_exists (A B C : ℝ) (h1 : A = (1 + find_x / 100) * B) (h2 : C = 0.75 * A) (h3 : A > C) (h4 : C > B) :
  find_x = 33.33 := 
by
  -- Proof steps
  sorry

end proof_x_exists_l185_185699


namespace find_xy_l185_185563

theorem find_xy (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 :=
by
  sorry

end find_xy_l185_185563


namespace find_abc_l185_185847

theorem find_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : a^3 + b^3 + c^3 = 2001 → (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) := 
sorry

end find_abc_l185_185847


namespace expression_evaluation_l185_185130

theorem expression_evaluation : 
  (1 : ℝ)^(6 * z - 3) / (7⁻¹ + 4⁻¹) = 28 / 11 :=
by
  sorry

end expression_evaluation_l185_185130


namespace total_amount_spent_l185_185825

def speakers : ℝ := 118.54
def new_tires : ℝ := 106.33
def window_tints : ℝ := 85.27
def seat_covers : ℝ := 79.99
def scheduled_maintenance : ℝ := 199.75
def steering_wheel_cover : ℝ := 15.63
def air_fresheners_set : ℝ := 12.96
def car_wash : ℝ := 25.0

theorem total_amount_spent :
  speakers + new_tires + window_tints + seat_covers + scheduled_maintenance + steering_wheel_cover + air_fresheners_set + car_wash = 643.47 :=
by
  sorry

end total_amount_spent_l185_185825


namespace karl_drove_420_miles_l185_185474

theorem karl_drove_420_miles :
  ∀ (car_mileage_per_gallon : ℕ)
    (tank_capacity : ℕ)
    (initial_drive_miles : ℕ)
    (gas_purchased : ℕ)
    (destination_tank_fraction : ℚ),
    car_mileage_per_gallon = 30 →
    tank_capacity = 16 →
    initial_drive_miles = 420 →
    gas_purchased = 10 →
    destination_tank_fraction = 3 / 4 →
    initial_drive_miles + (destination_tank_fraction * tank_capacity - (tank_capacity - (initial_drive_miles / car_mileage_per_gallon)) + gas_purchased) * car_mileage_per_gallon = 420 :=
by
  intros car_mileage_per_gallon tank_capacity initial_drive_miles gas_purchased destination_tank_fraction
  intro h1 -- car_mileage_per_gallon = 30
  intro h2 -- tank_capacity = 16
  intro h3 -- initial_drive_miles = 420
  intro h4 -- gas_purchased = 10
  intro h5 -- destination_tank_fraction = 3 / 4
  sorry

end karl_drove_420_miles_l185_185474


namespace isosceles_right_triangle_measure_l185_185204

theorem isosceles_right_triangle_measure (a XY YZ : ℝ) 
    (h1 : XY > YZ) 
    (h2 : a^2 = 25 / (1/2)) : XY = 10 :=
by
  sorry

end isosceles_right_triangle_measure_l185_185204


namespace construction_days_behind_without_additional_workers_l185_185313

-- Definitions for initial and additional workers and their respective efficiencies and durations.
def initial_workers : ℕ := 100
def initial_worker_efficiency : ℕ := 1
def total_days : ℕ := 150

def additional_workers_1 : ℕ := 50
def additional_worker_efficiency_1 : ℕ := 2
def additional_worker_start_day_1 : ℕ := 30

def additional_workers_2 : ℕ := 25
def additional_worker_efficiency_2 : ℕ := 3
def additional_worker_start_day_2 : ℕ := 45

def additional_workers_3 : ℕ := 15
def additional_worker_efficiency_3 : ℕ := 4
def additional_worker_start_day_3 : ℕ := 75

-- Define the total additional work units done by the extra workers.
def total_additional_work_units : ℕ := 
  (additional_workers_1 * additional_worker_efficiency_1 * (total_days - additional_worker_start_day_1)) +
  (additional_workers_2 * additional_worker_efficiency_2 * (total_days - additional_worker_start_day_2)) +
  (additional_workers_3 * additional_worker_efficiency_3 * (total_days - additional_worker_start_day_3))

-- Define the days the initial workers would have taken to do the additional work.
def initial_days_for_additional_work : ℕ := 
  (total_additional_work_units + (initial_workers * initial_worker_efficiency) - 1) / (initial_workers * initial_worker_efficiency)

-- Define the total days behind schedule.
def days_behind_schedule : ℕ := (total_days + initial_days_for_additional_work) - total_days

-- Define the theorem to prove.
theorem construction_days_behind_without_additional_workers : days_behind_schedule = 244 := 
  by 
  -- This translates to manually verifying the outcome.
  -- A detailed proof can be added later.
  sorry

end construction_days_behind_without_additional_workers_l185_185313


namespace sarith_laps_l185_185392

theorem sarith_laps 
  (k_speed : ℝ) (s_speed : ℝ) (k_laps : ℝ) (s_laps : ℝ) (distance_ratio : ℝ) :
  k_speed = 3 * s_speed →
  distance_ratio = 1 / 2 →
  k_laps = 12 →
  s_laps = (k_laps * 2 / 3) →
  s_laps = 8 :=
by
  intros
  sorry

end sarith_laps_l185_185392


namespace crackers_eaten_l185_185160

-- Define the number of packs and their respective number of crackers
def num_packs_8 : ℕ := 5
def num_packs_10 : ℕ := 10
def num_packs_12 : ℕ := 7
def num_packs_15 : ℕ := 3

def crackers_per_pack_8 : ℕ := 8
def crackers_per_pack_10 : ℕ := 10
def crackers_per_pack_12 : ℕ := 12
def crackers_per_pack_15 : ℕ := 15

-- Calculate the total number of animal crackers
def total_crackers : ℕ :=
  (num_packs_8 * crackers_per_pack_8) +
  (num_packs_10 * crackers_per_pack_10) +
  (num_packs_12 * crackers_per_pack_12) +
  (num_packs_15 * crackers_per_pack_15)

-- Define the number of students who didn't eat their crackers and the respective number of crackers per pack
def num_students_not_eaten : ℕ := 4
def different_crackers_not_eaten : List ℕ := [8, 10, 12, 15]

-- Calculate the total number of crackers not eaten by adding those packs.
def total_crackers_not_eaten : ℕ := different_crackers_not_eaten.sum

-- Theorem to prove the total number of crackers eaten.
theorem crackers_eaten : total_crackers - total_crackers_not_eaten = 224 :=
by
  -- Total crackers: 269
  -- Subtract crackers not eaten: 8 + 10 + 12 + 15 = 45
  -- Therefore: 269 - 45 = 224
  sorry

end crackers_eaten_l185_185160


namespace find_cost_price_l185_185438

theorem find_cost_price
  (cost_price : ℝ)
  (increase_rate : ℝ := 0.2)
  (decrease_rate : ℝ := 0.1)
  (profit : ℝ := 8):
  (1 + increase_rate) * cost_price * (1 - decrease_rate) - cost_price = profit → 
  cost_price = 100 := 
by 
  sorry

end find_cost_price_l185_185438


namespace permutation_six_two_l185_185019

-- Definition for permutation
def permutation (n k : ℕ) : ℕ := n * (n - 1)

-- Theorem stating that the permutation of 6 taken 2 at a time is 30
theorem permutation_six_two : permutation 6 2 = 30 :=
by
  -- proof will be filled here
  sorry

end permutation_six_two_l185_185019


namespace slow_population_growth_before_ir_l185_185322

-- Define the conditions
def low_level_social_productivity_before_ir : Prop := sorry
def high_birth_rate_before_ir : Prop := sorry
def high_mortality_rate_before_ir : Prop := sorry

-- The correct answer
def low_natural_population_growth_rate_before_ir : Prop := sorry

-- The theorem to prove
theorem slow_population_growth_before_ir 
  (h1 : low_level_social_productivity_before_ir) 
  (h2 : high_birth_rate_before_ir) 
  (h3 : high_mortality_rate_before_ir) : low_natural_population_growth_rate_before_ir := 
sorry

end slow_population_growth_before_ir_l185_185322


namespace simplify_and_evaluate_l185_185686

variable (x y : ℚ)
variable (expr : ℚ := 3 * x * y^2 - (x * y - 2 * (2 * x * y - 3 / 2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y)

theorem simplify_and_evaluate (h1 : x = 3) (h2 : y = -1 / 3) : expr = -3 :=
by
  sorry

end simplify_and_evaluate_l185_185686


namespace cards_given_by_Dan_l185_185484

def initial_cards : Nat := 27
def bought_cards : Nat := 20
def total_cards : Nat := 88

theorem cards_given_by_Dan :
  ∃ (cards_given : Nat), cards_given = total_cards - bought_cards - initial_cards :=
by
  use 41
  sorry

end cards_given_by_Dan_l185_185484


namespace complement_of_A_in_U_l185_185721

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def complementA : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = complementA :=
by
  sorry

end complement_of_A_in_U_l185_185721


namespace area_of_triangle_with_given_sides_l185_185053

variable (a b c : ℝ)
variable (s : ℝ := (a + b + c) / 2)
variable (area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem area_of_triangle_with_given_sides (ha : a = 65) (hb : b = 60) (hc : c = 25) :
  area = 750 := by
  sorry

end area_of_triangle_with_given_sides_l185_185053


namespace probability_snow_once_first_week_l185_185723

theorem probability_snow_once_first_week :
  let p_first_two_days := (3 / 4) * (3 / 4)
  let p_next_three_days := (1 / 2) * (1 / 2) * (1 / 2)
  let p_last_two_days := (2 / 3) * (2 / 3)
  let p_no_snow := p_first_two_days * p_next_three_days * p_last_two_days
  let p_at_least_once := 1 - p_no_snow
  p_at_least_once = 31 / 32 :=
by
  sorry

end probability_snow_once_first_week_l185_185723


namespace relationship_among_a_b_c_l185_185839

noncomputable def a : ℝ := (1 / 2) ^ (3 / 4)
noncomputable def b : ℝ := (3 / 4) ^ (1 / 2)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_among_a_b_c : a < b ∧ b < c := 
by
  -- Skipping the proof steps
  sorry

end relationship_among_a_b_c_l185_185839


namespace cuboid_length_l185_185973

theorem cuboid_length (SA w h : ℕ) (h_SA : SA = 700) (h_w : w = 14) (h_h : h = 7) 
  (h_surface_area : SA = 2 * l * w + 2 * l * h + 2 * w * h) : l = 12 :=
by
  intros
  sorry

end cuboid_length_l185_185973


namespace arithmetic_sequence_general_term_l185_185333

theorem arithmetic_sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = 5 * n^2 + 3 * n)
  (hS₁ : a 1 = S 1)
  (hS₂ : ∀ n, a (n + 1) = S (n + 1) - S n) :
  ∀ n, a n = 10 * n - 2 :=
by
  sorry

end arithmetic_sequence_general_term_l185_185333


namespace probability_A_B_C_adjacent_l185_185168

theorem probability_A_B_C_adjacent (students : Fin 5 → Prop) (A B C : Fin 5) :
  (students A ∧ students B ∧ students C) →
  (∃ n m : ℕ, n = 48 ∧ m = 12 ∧ m / n = (1 : ℚ) / 4) :=
by
  sorry

end probability_A_B_C_adjacent_l185_185168


namespace circle_center_l185_185323

theorem circle_center : 
  ∃ (h k : ℝ), (h, k) = (1, -2) ∧ 
    ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - h)^2 + (y - k)^2 = 9 :=
by
  sorry

end circle_center_l185_185323


namespace adjacent_product_negative_l185_185491

-- Define the sequence
def a (n : ℕ) : ℤ := 2*n - 17

-- Define the claim about the product of adjacent terms being negative
theorem adjacent_product_negative : a 8 * a 9 < 0 :=
by sorry

end adjacent_product_negative_l185_185491


namespace three_integers_sum_of_consecutive_odds_l185_185248

theorem three_integers_sum_of_consecutive_odds :
  {N : ℕ | N ≤ 500 ∧ (∃ j n, N = j * (2 * n + j) ∧ j ≥ 1) ∧
                   (∃! j1 j2 j3, ∃ n1 n2 n3, N = j1 * (2 * n1 + j1) ∧ N = j2 * (2 * n2 + j2) ∧ N = j3 * (2 * n3 + j3) ∧ j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3)} = {16, 18, 50} :=
by
  sorry

end three_integers_sum_of_consecutive_odds_l185_185248


namespace f_even_l185_185434

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_const : ¬ (∀ x y : ℝ, f x = f y)
axiom f_equiv1 : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_equiv2 : ∀ x : ℝ, f (1 + x) = -f x

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  sorry

end f_even_l185_185434


namespace expression_equals_4096_l185_185076

noncomputable def calculate_expression : ℕ :=
  ((16^15 / 16^14)^3 * 8^3) / 2^9

theorem expression_equals_4096 : calculate_expression = 4096 :=
by {
  -- proof would go here
  sorry
}

end expression_equals_4096_l185_185076


namespace paul_initial_books_l185_185364

theorem paul_initial_books (sold_books : ℕ) (left_books : ℕ) (initial_books : ℕ) 
  (h_sold_books : sold_books = 109)
  (h_left_books : left_books = 27)
  (h_initial_books_formula : initial_books = sold_books + left_books) : 
  initial_books = 136 :=
by
  rw [h_sold_books, h_left_books] at h_initial_books_formula
  exact h_initial_books_formula

end paul_initial_books_l185_185364


namespace find_e_l185_185404

def P (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

-- Conditions
variables (d e f : ℝ)
-- Mean of zeros, twice product of zeros, and sum of coefficients are equal
variables (mean_of_zeros equals twice_product_of_zeros equals sum_of_coefficients equals: ℝ)
-- y-intercept is 9
axiom intercept_eq_nine : f = 9

-- Vieta's formulas for cubic polynomial
axiom product_of_zeros : twice_product_of_zeros = 2 * (- (f / 3))
axiom mean_of_zeros_sum : mean_of_zeros = -18/3  -- 3 times the mean of the zeros
axiom sum_of_coef : 3 + d + e + f = sum_of_coefficients

-- All these quantities are equal to the same value
axiom triple_equality : mean_of_zeros = twice_product_of_zeros
axiom triple_equality_coefs : mean_of_zeros = sum_of_coefficients

-- Lean statement we need to prove
theorem find_e : e = -72 :=
by
  sorry

end find_e_l185_185404


namespace total_cost_eq_l185_185236

noncomputable def total_cost : Real :=
  let os_overhead := 1.07
  let cost_per_millisecond := 0.023
  let tape_mounting_cost := 5.35
  let cost_per_megabyte := 0.15
  let cost_per_kwh := 0.02
  let technician_rate_per_hour := 50.0
  let minutes_to_milliseconds := 60000
  let gb_to_mb := 1024

  -- Define program specifics
  let computer_time_minutes := 45.0
  let memory_gb := 3.5
  let electricity_kwh := 2.0
  let technician_time_minutes := 20.0

  -- Calculate costs
  let computer_time_cost := (computer_time_minutes * minutes_to_milliseconds * cost_per_millisecond)
  let memory_cost := (memory_gb * gb_to_mb * cost_per_megabyte)
  let electricity_cost := (electricity_kwh * cost_per_kwh)
  let technician_time_total_hours := (technician_time_minutes * 2 / 60.0)
  let technician_cost := (technician_time_total_hours * technician_rate_per_hour)

  os_overhead + computer_time_cost + tape_mounting_cost + memory_cost + electricity_cost + technician_cost

theorem total_cost_eq : total_cost = 62677.39 := by
  sorry

end total_cost_eq_l185_185236


namespace parallel_line_slope_l185_185318

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 9) : ∃ (m : ℝ), m = 1 / 2 := 
sorry

end parallel_line_slope_l185_185318


namespace decreasing_number_4312_max_decreasing_number_divisible_by_9_l185_185314

-- Definitions and conditions
def is_decreasing_number (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d4 ≠ 0 ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  (10 * d1 + d2 - (10 * d2 + d3) = 10 * d3 + d4)

def is_divisible_by_9 (n m : ℕ) : Prop :=
  (n + m) % 9 = 0

-- Theorem Statements
theorem decreasing_number_4312 : 
  is_decreasing_number 4312 :=
sorry

theorem max_decreasing_number_divisible_by_9 : 
  ∀ n, is_decreasing_number n ∧ is_divisible_by_9 (n / 10) (n % 1000) → n ≤ 8165 :=
sorry

end decreasing_number_4312_max_decreasing_number_divisible_by_9_l185_185314


namespace compare_y_values_l185_185617

-- Define the quadratic function y = x^2 + 2x + c
def quadratic (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * x + c

-- Points A, B, and C on the quadratic function
variables 
  (c : ℝ) 
  (y1 y2 y3 : ℝ) 
  (hA : y1 = quadratic (-3) c) 
  (hB : y2 = quadratic (-2) c) 
  (hC : y3 = quadratic 2 c)

theorem compare_y_values :
  y3 > y1 ∧ y1 > y2 :=
by sorry

end compare_y_values_l185_185617


namespace problem_statement_l185_185816

-- Definitions for the conditions in the problem
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p
def has_three_divisors (k : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ k = p^2

-- Given conditions
def m : ℕ := 3 -- the smallest odd prime
def n : ℕ := 49 -- the largest integer less than 50 with exactly three positive divisors

-- The proof statement
theorem problem_statement : m + n = 52 :=
by sorry

end problem_statement_l185_185816


namespace even_sine_function_phi_eq_pi_div_2_l185_185959
open Real

theorem even_sine_function_phi_eq_pi_div_2 (φ : ℝ) (h : 0 ≤ φ ∧ φ ≤ π)
    (even_f : ∀ x : ℝ, sin (x + φ) = sin (-x + φ)) : φ = π / 2 :=
sorry

end even_sine_function_phi_eq_pi_div_2_l185_185959


namespace problem_l185_185514

def P (x : ℝ) : Prop := x^2 - 2*x + 1 > 0

theorem problem (h : ¬ ∀ x : ℝ, P x) : ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0 :=
by {
  sorry
}

end problem_l185_185514


namespace number_of_sevens_l185_185545

theorem number_of_sevens (n : ℕ) : ∃ (k : ℕ), k < n ∧ ∃ (f : ℕ → ℕ), (∀ i, f i = 7) ∧ (7 * ((77 - 7) / 7) ^ 14 - 1) / (7 + (7 + 7)/7) = 7^(f k) :=
by sorry

end number_of_sevens_l185_185545


namespace smallest_number_of_students_l185_185032

theorem smallest_number_of_students (n : ℕ) :
  (n % 3 = 2) ∧
  (n % 5 = 3) ∧
  (n % 8 = 5) →
  n = 53 :=
by
  intro h
  sorry

end smallest_number_of_students_l185_185032


namespace value_of_squares_l185_185450

-- Define the conditions
variables (p q : ℝ)

-- State the theorem with the given conditions and the proof goal
theorem value_of_squares (h1 : p * q = 12) (h2 : p + q = 8) : p ^ 2 + q ^ 2 = 40 :=
sorry

end value_of_squares_l185_185450


namespace friend_P_distance_l185_185340

theorem friend_P_distance (v t : ℝ) (hv : v > 0)
  (distance_trail : 22 = (1.20 * v * t) + (v * t))
  (h_t : t = 22 / (2.20 * v)) : 
  (1.20 * v * t = 12) :=
by
  sorry

end friend_P_distance_l185_185340


namespace problem_a_problem_b_l185_185330

-- Part (a)
theorem problem_a (n: Nat) : ∃ k: ℤ, (32^ (3 * n) - 1312^ n) = 1966 * k := sorry

-- Part (b)
theorem problem_b (n: Nat) : ∃ m: ℤ, (843^ (2 * n + 1) - 1099^ (2 * n + 1) + 16^ (4 * n + 2)) = 1967 * m := sorry

end problem_a_problem_b_l185_185330


namespace range_of_a_l185_185106

noncomputable def set_A (a : ℝ) : Set ℝ := {x | x < a}
noncomputable def set_B : Set ℝ := {x | 1 < x ∧ x < 2}
noncomputable def complement_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2 }

theorem range_of_a (a : ℝ) : (set_A a ∪ complement_B) = Set.univ ↔ 2 ≤ a := 
by 
  sorry

end range_of_a_l185_185106


namespace total_shoes_count_l185_185229

-- Define the concepts and variables related to the conditions
def num_people := 10
def num_people_regular_shoes := 4
def num_people_sandals := 3
def num_people_slippers := 3
def num_shoes_regular := 2
def num_shoes_sandals := 1
def num_shoes_slippers := 1

-- Goal: Prove that the total number of shoes kept outside is 20
theorem total_shoes_count :
  (num_people_regular_shoes * num_shoes_regular) +
  (num_people_sandals * num_shoes_sandals * 2) +
  (num_people_slippers * num_shoes_slippers * 2) = 20 :=
by
  sorry

end total_shoes_count_l185_185229


namespace sum_of_reciprocals_six_l185_185414

theorem sum_of_reciprocals_six {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x) + (1 / y) = 6 :=
by
  sorry

end sum_of_reciprocals_six_l185_185414


namespace selling_price_l185_185495

-- Definitions
def price_coffee_A : ℝ := 10
def price_coffee_B : ℝ := 12
def weight_coffee_A : ℝ := 240
def weight_coffee_B : ℝ := 240
def total_weight : ℝ := 480
def total_cost : ℝ := (weight_coffee_A * price_coffee_A) + (weight_coffee_B * price_coffee_B)

-- Theorem
theorem selling_price (h_total_weight : total_weight = weight_coffee_A + weight_coffee_B) :
  total_cost / total_weight = 11 :=
by
  sorry

end selling_price_l185_185495


namespace polynomial_simplification_l185_185784

theorem polynomial_simplification (x : ℝ) : 
  (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2 * x^3 - 4 * x^2 + 10 * x + 1 :=
by
  sorry

end polynomial_simplification_l185_185784


namespace eating_time_correct_l185_185607

-- Define the rates at which each individual eats cereal
def rate_fat : ℚ := 1 / 20
def rate_thin : ℚ := 1 / 30
def rate_medium : ℚ := 1 / 15

-- Define the combined rate of eating cereal together
def combined_rate : ℚ := rate_fat + rate_thin + rate_medium

-- Define the total pounds of cereal
def total_cereal : ℚ := 5

-- Define the time taken by everyone to eat the cereal
def time_taken : ℚ := total_cereal / combined_rate

-- Proof statement
theorem eating_time_correct :
  time_taken = 100 / 3 :=
by sorry

end eating_time_correct_l185_185607


namespace mean_age_Mendez_children_l185_185016

def Mendez_children_ages : List ℕ := [5, 5, 10, 12, 15]

theorem mean_age_Mendez_children : 
  (5 + 5 + 10 + 12 + 15) / 5 = 9.4 := 
by
  sorry

end mean_age_Mendez_children_l185_185016


namespace find_pairs_of_positive_integers_l185_185951

theorem find_pairs_of_positive_integers (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  3 * 2^m + 1 = n^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) :=
sorry

end find_pairs_of_positive_integers_l185_185951


namespace count_digit_2_in_range_1_to_1000_l185_185296

theorem count_digit_2_in_range_1_to_1000 :
  let count_digit_occur (digit : ℕ) (range_end : ℕ) : ℕ :=
    (range_end + 1).digits 10
    |>.count digit
  count_digit_occur 2 1000 = 300 :=
by
  sorry

end count_digit_2_in_range_1_to_1000_l185_185296


namespace find_maximum_marks_l185_185398

variable (percent_marks : ℝ := 0.92)
variable (obtained_marks : ℝ := 368)
variable (max_marks : ℝ := obtained_marks / percent_marks)

theorem find_maximum_marks : max_marks = 400 := by
  sorry

end find_maximum_marks_l185_185398


namespace sin_cos_sum_l185_185171

theorem sin_cos_sum (α : ℝ) (h : ∃ (c : ℝ), Real.sin α = -1 / c ∧ Real.cos α = 2 / c ∧ c = Real.sqrt 5) :
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 :=
by sorry

end sin_cos_sum_l185_185171


namespace ab_bc_cd_da_leq_1_over_4_l185_185184

theorem ab_bc_cd_da_leq_1_over_4 (a b c d : ℝ) (h : a + b + c + d = 1) : 
  a * b + b * c + c * d + d * a ≤ 1 / 4 := 
sorry

end ab_bc_cd_da_leq_1_over_4_l185_185184


namespace probability_same_class_l185_185249

-- Define the problem conditions
def num_classes : ℕ := 3
def total_scenarios : ℕ := num_classes * num_classes
def same_class_scenarios : ℕ := num_classes

-- Formulate the proof problem
theorem probability_same_class :
  (same_class_scenarios : ℚ) / total_scenarios = 1 / 3 :=
sorry

end probability_same_class_l185_185249


namespace interval_contains_root_l185_185663

theorem interval_contains_root :
  (∃ c, (0 < c ∧ c < 1) ∧ (2^c + c - 2 = 0) ∧ 
        (∀ x1 x2, x1 < x2 → 2^x1 + x1 - 2 < 2^x2 + x2 - 2) ∧ 
        (0 < 1) ∧ 
        ((2^0 + 0 - 2) = -1) ∧ 
        ((2^1 + 1 - 2) = 1)) := 
by 
  sorry

end interval_contains_root_l185_185663


namespace integer_triples_soln_l185_185158

theorem integer_triples_soln (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3*x*y*z = 2003) ↔ ( (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) ) := 
by
  sorry

end integer_triples_soln_l185_185158


namespace geometric_sequence_sum_l185_185868

theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (n : ℕ) (hS : ∀ n, S n = t - 3 * 2^n) (h_geom : ∀ n, a (n + 1) = a n * r) :
  t = 3 :=
by
  sorry

end geometric_sequence_sum_l185_185868


namespace number_of_members_l185_185416

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l185_185416


namespace mode_of_shoe_sizes_is_25_5_l185_185116

def sales_data := [(24, 2), (24.5, 5), (25, 3), (25.5, 6), (26, 4)]

theorem mode_of_shoe_sizes_is_25_5 
  (h : ∀ x ∈ sales_data, 2 ≤ x.1 ∧ 
        (∀ y ∈ sales_data, x.2 ≤ y.2 → x.1 = 25.5 ∨ x.2 < 6)) : 
  (∃ s, s ∈ sales_data ∧ s.1 = 25.5 ∧ s.2 = 6) :=
sorry

end mode_of_shoe_sizes_is_25_5_l185_185116


namespace quadratic_range_l185_185413

theorem quadratic_range (x y : ℝ) (h1 : y = -(x - 5) ^ 2 + 1) (h2 : 2 < x ∧ x < 6) :
  -8 < y ∧ y ≤ 1 := 
sorry

end quadratic_range_l185_185413


namespace tank_fill_time_l185_185174

theorem tank_fill_time (R1 R2 t_required : ℝ) (hR1: R1 = 1 / 8) (hR2: R2 = 1 / 12) (hT : t_required = 4.8) :
  t_required = 1 / (R1 + R2) :=
by 
  -- Proof goes here
  sorry

end tank_fill_time_l185_185174


namespace julia_shortfall_l185_185145

-- Definitions based on the problem conditions
def rock_and_roll_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def julia_money : ℕ := 75

-- Proof problem: Prove that Julia is short $25
theorem julia_shortfall : (quantity * rock_and_roll_price + quantity * pop_price + quantity * dance_price + quantity * country_price) - julia_money = 25 := by
  sorry

end julia_shortfall_l185_185145


namespace dot_product_of_a_b_l185_185753

theorem dot_product_of_a_b 
  (a b : ℝ)
  (θ : ℝ)
  (ha : a = 2 * Real.sin (15 * Real.pi / 180))
  (hb : b = 4 * Real.cos (15 * Real.pi / 180))
  (hθ : θ = 30 * Real.pi / 180) :
  (a * b * Real.cos θ) = Real.sqrt 3 := by
  sorry

end dot_product_of_a_b_l185_185753


namespace find_a_l185_185817

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l185_185817


namespace cubes_sum_eq_zero_l185_185976

theorem cubes_sum_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 :=
by
  sorry

end cubes_sum_eq_zero_l185_185976


namespace trey_total_time_is_two_hours_l185_185998

-- Define the conditions
def num_cleaning_tasks := 7
def num_shower_tasks := 1
def num_dinner_tasks := 4
def time_per_task := 10 -- in minutes
def minutes_per_hour := 60

-- Total tasks
def total_tasks := num_cleaning_tasks + num_shower_tasks + num_dinner_tasks

-- Total time in minutes
def total_time_minutes := total_tasks * time_per_task

-- Total time in hours
def total_time_hours := total_time_minutes / minutes_per_hour

-- Prove that the total time Trey will need to complete his list is 2 hours
theorem trey_total_time_is_two_hours : total_time_hours = 2 := by
  sorry

end trey_total_time_is_two_hours_l185_185998


namespace smallest_number_remainder_l185_185918

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end smallest_number_remainder_l185_185918


namespace angle_is_120_degrees_l185_185952

-- Define the magnitudes of vectors a and b and their dot product
def magnitude_a : ℝ := 10
def magnitude_b : ℝ := 12
def dot_product_ab : ℝ := -60

-- Define the angle between vectors a and b
def angle_between_vectors (θ : ℝ) : Prop :=
  magnitude_a * magnitude_b * Real.cos θ = dot_product_ab

-- Prove that the angle θ is 120 degrees
theorem angle_is_120_degrees : angle_between_vectors (2 * Real.pi / 3) :=
by 
  unfold angle_between_vectors
  sorry

end angle_is_120_degrees_l185_185952


namespace pasha_mistake_l185_185390

theorem pasha_mistake :
  ¬ (∃ (K R O S C T P : ℕ), K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ P < 10 ∧
    K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ P ∧
    R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ P ∧
    O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ P ∧
    S ≠ C ∧ S ≠ T ∧ S ≠ P ∧
    C ≠ T ∧ C ≠ P ∧ T ≠ P ∧
    10000 * K + 1000 * R + 100 * O + 10 * S + S + 2011 = 10000 * C + 1000 * T + 100 * A + 10 * P + T) :=
sorry

end pasha_mistake_l185_185390


namespace competition_participants_solved_all_three_l185_185308

theorem competition_participants_solved_all_three
  (p1 p2 p3 : ℕ → Prop)
  (total_participants : ℕ)
  (h1 : ∃ n, n = 85 * total_participants / 100 ∧ ∀ k, k < n → p1 k)
  (h2 : ∃ n, n = 80 * total_participants / 100 ∧ ∀ k, k < n → p2 k)
  (h3 : ∃ n, n = 75 * total_participants / 100 ∧ ∀ k, k < n → p3 k) :
  ∃ n, n ≥ 40 * total_participants / 100 ∧ ∀ k, k < n → p1 k ∧ p2 k ∧ p3 k :=
by
  sorry

end competition_participants_solved_all_three_l185_185308


namespace most_stable_performance_l185_185927

theorem most_stable_performance 
    (S_A S_B S_C S_D : ℝ)
    (h_A : S_A = 0.54) 
    (h_B : S_B = 0.61) 
    (h_C : S_C = 0.7) 
    (h_D : S_D = 0.63) :
    S_A <= S_B ∧ S_A <= S_C ∧ S_A <= S_D :=
by {
  sorry
}

end most_stable_performance_l185_185927


namespace squirrel_spiral_path_height_l185_185803

-- Define the conditions
def spiralPath (circumference rise totalDistance : ℝ) : Prop :=
  ∃ (numberOfCircuits : ℝ), numberOfCircuits = totalDistance / circumference ∧ numberOfCircuits * rise = totalDistance

-- Define the height of the post proof
theorem squirrel_spiral_path_height : 
  let circumference := 2 -- feet
  let rise := 4 -- feet
  let totalDistance := 8 -- feet
  let height := 16 -- feet
  spiralPath circumference rise totalDistance → height = (totalDistance / circumference) * rise :=
by
  intro h
  sorry

end squirrel_spiral_path_height_l185_185803


namespace sum_of_numbers_in_ratio_l185_185619

theorem sum_of_numbers_in_ratio 
  (x : ℕ)
  (h : 5 * x = 560) : 
  2 * x + 3 * x + 4 * x + 5 * x = 1568 := 
by 
  sorry

end sum_of_numbers_in_ratio_l185_185619


namespace find_angle_A_l185_185763

theorem find_angle_A (a b : ℝ) (A B : ℝ) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = Real.pi / 3) :
  A = Real.pi / 4 :=
by
  -- This is a placeholder for the proof
  sorry

end find_angle_A_l185_185763


namespace chef_earns_less_than_manager_l185_185532

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.22

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 :=
by
  sorry

end chef_earns_less_than_manager_l185_185532


namespace cost_price_article_l185_185761

theorem cost_price_article (x : ℝ) (h : 56 - x = x - 42) : x = 49 :=
by sorry

end cost_price_article_l185_185761


namespace toy_car_production_l185_185819

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l185_185819


namespace amount_transferred_l185_185132

def original_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : original_balance - remaining_balance = 69 :=
by
  sorry

end amount_transferred_l185_185132


namespace inscribed_circle_radius_of_rhombus_l185_185483

theorem inscribed_circle_radius_of_rhombus (d1 d2 : ℝ) (a r : ℝ) : 
  d1 = 15 → d2 = 24 → a = Real.sqrt ((15 / 2)^2 + (24 / 2)^2) → 
  (d1 * d2) / 2 = 2 * a * r → 
  r = 60.07 / 13 :=
by
  intros h1 h2 h3 h4
  sorry

end inscribed_circle_radius_of_rhombus_l185_185483


namespace lassis_from_mangoes_l185_185903

theorem lassis_from_mangoes (mangoes lassis mangoes' lassis' : ℕ) 
  (h1 : lassis = (8 * mangoes) / 3)
  (h2 : mangoes = 15) :
  lassis = 40 :=
by
  sorry

end lassis_from_mangoes_l185_185903


namespace stephanie_total_remaining_bills_l185_185097

-- Conditions
def electricity_bill : ℕ := 60
def electricity_paid : ℕ := electricity_bill
def gas_bill : ℕ := 40
def gas_paid : ℕ := (3 * gas_bill) / 4 + 5
def water_bill : ℕ := 40
def water_paid : ℕ := water_bill / 2
def internet_bill : ℕ := 25
def internet_payment : ℕ := 5
def internet_paid : ℕ := 4 * internet_payment

-- Define
def remaining_electricity : ℕ := electricity_bill - electricity_paid
def remaining_gas : ℕ := gas_bill - gas_paid
def remaining_water : ℕ := water_bill - water_paid
def remaining_internet : ℕ := internet_bill - internet_paid

def total_remaining : ℕ := remaining_electricity + remaining_gas + remaining_water + remaining_internet

-- Problem Statement
theorem stephanie_total_remaining_bills :
  total_remaining = 30 :=
by
  -- proof goes here (not required as per the instructions)
  sorry

end stephanie_total_remaining_bills_l185_185097


namespace B_subset_A_l185_185520

def A (x : ℝ) : Prop := abs (2 * x - 3) > 1
def B (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem B_subset_A : ∀ x, B x → A x := sorry

end B_subset_A_l185_185520


namespace correct_proposition_l185_185553

-- Definitions based on conditions
def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0
def not_p : Prop := ∀ x : ℝ, x^2 + 2 * x + 2016 > 0

-- Proof statement
theorem correct_proposition : p → not_p :=
by sorry

end correct_proposition_l185_185553


namespace minimum_species_l185_185470

theorem minimum_species (n : ℕ) (h : n = 2021) 
  (even_separation : ∀ (a b : ℕ), a ≠ b → (a ≠ b) → (a % 2 = 0)) : 
  ∃ (s : ℕ), s = 1011 :=
by
  sorry

end minimum_species_l185_185470


namespace find_x_l185_185475

theorem find_x 
  (x : ℝ) 
  (h1 : 0 < x)
  (h2 : x < π / 2)
  (h3 : 1 / (Real.sin x) = 1 / (Real.sin (2 * x)) + 1 / (Real.sin (4 * x)) + 1 / (Real.sin (8 * x))) : 
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 :=
by
  sorry

end find_x_l185_185475


namespace system_solution_5_3_l185_185582

variables (x y : ℤ)

theorem system_solution_5_3 :
  (x = 5) ∧ (y = 3) → (2 * x - 3 * y = 1) :=
by intros; sorry

end system_solution_5_3_l185_185582


namespace factorization_solution_l185_185861

def factorization_problem : Prop :=
  ∃ (a b c : ℤ), (∀ (x : ℤ), x^2 + 17 * x + 70 = (x + a) * (x + b)) ∧ 
                 (∀ (x : ℤ), x^2 - 18 * x + 80 = (x - b) * (x - c)) ∧ 
                 (a + b + c = 28)

theorem factorization_solution : factorization_problem :=
sorry

end factorization_solution_l185_185861


namespace sunflower_seeds_contest_l185_185717

theorem sunflower_seeds_contest 
  (first_player_seeds : ℕ) (second_player_seeds : ℕ) (total_seeds : ℕ) 
  (third_player_seeds : ℕ) (third_more : ℕ) 
  (h1 : first_player_seeds = 78) 
  (h2 : second_player_seeds = 53) 
  (h3 : total_seeds = 214) 
  (h4 : first_player_seeds + second_player_seeds + third_player_seeds = total_seeds) 
  (h5 : third_more = third_player_seeds - second_player_seeds) : 
  third_more = 30 :=
by
  sorry

end sunflower_seeds_contest_l185_185717


namespace trigonometric_identity_l185_185865

theorem trigonometric_identity 
  (α : ℝ)
  (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l185_185865


namespace tree_planting_growth_rate_l185_185680

theorem tree_planting_growth_rate {x : ℝ} :
  400 * (1 + x) ^ 2 = 625 :=
sorry

end tree_planting_growth_rate_l185_185680


namespace min_rho_squared_l185_185536

noncomputable def rho_squared (x t : ℝ) : ℝ :=
  (x - t)^2 + (x^2 - 4 * x + 7 + t)^2

theorem min_rho_squared : 
  ∃ (x t : ℝ), x = 3/2 ∧ t = -7/8 ∧ 
  ∀ (x' t' : ℝ), rho_squared x' t' ≥ rho_squared (3/2) (-7/8) :=
by
  sorry

end min_rho_squared_l185_185536


namespace triangle_area_MEQF_l185_185331

theorem triangle_area_MEQF
  (radius_P : ℝ)
  (chord_EF : ℝ)
  (par_EF_MN : Prop)
  (MQ : ℝ)
  (collinear_MQPN : Prop)
  (P MEF : ℝ × ℝ)
  (segment_P_Q : ℝ)
  (EF_length : ℝ)
  (radius_value : radius_P = 10)
  (EF_value : chord_EF = 12)
  (MQ_value : MQ = 20)
  (MN_parallel : par_EF_MN)
  (collinear : collinear_MQPN) :
  ∃ (area : ℝ), area = 48 := 
sorry

end triangle_area_MEQF_l185_185331


namespace range_of_a_l185_185081

def is_ellipse (a : ℝ) : Prop :=
  2 * a > 0 ∧ 3 * a - 6 > 0 ∧ 2 * a < 3 * a - 6

def discriminant_neg (a : ℝ) : Prop :=
  a^2 + 8 * a - 48 < 0

def p (a : ℝ) : Prop := is_ellipse a
def q (a : ℝ) : Prop := discriminant_neg a

theorem range_of_a (a : ℝ) : p a ∧ q a → 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l185_185081


namespace reinforcement_arrival_l185_185471

theorem reinforcement_arrival (x : ℕ) :
  (2000 * 40) = (2000 * x + 4000 * 10) → x = 20 :=
by
  sorry

end reinforcement_arrival_l185_185471


namespace triangle_cosine_sine_inequality_l185_185530

theorem triangle_cosine_sine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hA_lt_pi : A < Real.pi)
  (hB_lt_pi : B < Real.pi)
  (hC_lt_pi : C < Real.pi) :
  Real.cos A * (Real.sin B + Real.sin C) ≥ -2 * Real.sqrt 6 / 9 := 
by
  sorry

end triangle_cosine_sine_inequality_l185_185530


namespace determine_q_l185_185324

-- Lean 4 statement
theorem determine_q (a : ℝ) (q : ℝ → ℝ) :
  (∀ x, q x = a * (x + 2) * (x - 3)) ∧ q 1 = 8 →
  q x = - (4 / 3) * x ^ 2 + (4 / 3) * x + 8 := 
sorry

end determine_q_l185_185324


namespace volume_tetrahedron_ABCD_l185_185972

noncomputable def volume_of_tetrahedron (AB CD distance angle : ℝ) : ℝ :=
  (1 / 3) * ((1 / 2) * AB * CD * Real.sin angle) * distance

theorem volume_tetrahedron_ABCD :
  volume_of_tetrahedron 1 (Real.sqrt 3) 2 (Real.pi / 3) = 1 / 2 :=
by
  unfold volume_of_tetrahedron
  sorry

end volume_tetrahedron_ABCD_l185_185972


namespace inequality_does_not_hold_l185_185200

theorem inequality_does_not_hold (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by {
  sorry
}

end inequality_does_not_hold_l185_185200


namespace f_2007_eq_0_l185_185367

-- Define even function and odd function properties
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define functions f and g
variables (f g : ℝ → ℝ)

-- Assume the given conditions
axiom even_f : is_even f
axiom odd_g : is_odd g
axiom g_def : ∀ x, g x = f (x - 1)

-- Prove that f(2007) = 0
theorem f_2007_eq_0 : f 2007 = 0 :=
sorry

end f_2007_eq_0_l185_185367


namespace houses_per_block_correct_l185_185538

-- Define the conditions
def total_mail_per_block : ℕ := 32
def mail_per_house : ℕ := 8

-- Define the correct answer
def houses_per_block : ℕ := 4

-- Theorem statement
theorem houses_per_block_correct (total_mail_per_block mail_per_house : ℕ) : 
  total_mail_per_block = 32 →
  mail_per_house = 8 →
  total_mail_per_block / mail_per_house = houses_per_block :=
by
  intros h1 h2
  sorry

end houses_per_block_correct_l185_185538


namespace problem_1_problem_2_l185_185506

-- Problem 1
theorem problem_1 :
  -((1 / 2) / 3) * (3 - (-3)^2) = 1 :=
by
  sorry

-- Problem 2
theorem problem_2 {x : ℝ} (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 * x) / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) :=
by
  sorry

end problem_1_problem_2_l185_185506


namespace yellow_lights_count_l185_185739

theorem yellow_lights_count (total_lights : ℕ) (red_lights : ℕ) (blue_lights : ℕ) (yellow_lights : ℕ) :
  total_lights = 95 → red_lights = 26 → blue_lights = 32 → yellow_lights = total_lights - (red_lights + blue_lights) → yellow_lights = 37 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end yellow_lights_count_l185_185739


namespace greatest_value_a2_b2_c2_d2_l185_185256

theorem greatest_value_a2_b2_c2_d2 :
  ∃ (a b c d : ℝ), a + b = 12 ∧ ab + c + d = 54 ∧ ad + bc = 105 ∧ cd = 50 ∧ a^2 + b^2 + c^2 + d^2 = 124 := by
  sorry

end greatest_value_a2_b2_c2_d2_l185_185256


namespace pictures_per_album_l185_185837

-- Define the conditions
def uploaded_pics_phone : ℕ := 22
def uploaded_pics_camera : ℕ := 2
def num_albums : ℕ := 4

-- Define the total pictures uploaded
def total_pictures : ℕ := uploaded_pics_phone + uploaded_pics_camera

-- Define the target statement as the theorem
theorem pictures_per_album : (total_pictures / num_albums) = 6 := by
  sorry

end pictures_per_album_l185_185837


namespace available_milk_for_me_l185_185722

def initial_milk_litres : ℝ := 1
def myeongseok_milk_litres : ℝ := 0.1
def mingu_milk_litres : ℝ := myeongseok_milk_litres + 0.2
def minjae_milk_litres : ℝ := 0.3

theorem available_milk_for_me :
  initial_milk_litres - (myeongseok_milk_litres + mingu_milk_litres + minjae_milk_litres) = 0.3 :=
by sorry

end available_milk_for_me_l185_185722


namespace total_travel_time_l185_185573

-- Define the given conditions
def speed_jogging : ℝ := 5
def speed_bus : ℝ := 30
def distance_to_school : ℝ := 6.857142857142858

-- State the theorem to prove
theorem total_travel_time :
  (distance_to_school / speed_jogging) + (distance_to_school / speed_bus) = 1.6 :=
by
  sorry

end total_travel_time_l185_185573


namespace ticket_cost_l185_185225

open Real

-- Variables for ticket prices
variable (A C S : ℝ)

-- Given conditions
def cost_condition : Prop :=
  C = A / 2 ∧ S = A - 1.50 ∧ 6 * A + 5 * C + 3 * S = 40.50

-- The goal is to prove that the total cost for 10 adult tickets, 8 child tickets,
-- and 4 senior tickets is 64.38
theorem ticket_cost (h : cost_condition A C S) : 10 * A + 8 * C + 4 * S = 64.38 :=
by
  -- Implementation of the proof would go here
  sorry

end ticket_cost_l185_185225


namespace smallest_d_for_inverse_l185_185876

noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse : ∃ d : ℝ, (∀ x1 x2, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = 3 := 
sorry

end smallest_d_for_inverse_l185_185876


namespace tetrahedron_max_volume_l185_185043

noncomputable def tetrahedron_volume (AC AB BD CD : ℝ) : ℝ :=
  let x := (2 : ℝ) * (Real.sqrt 3) / 3
  let m := Real.sqrt (1 - x^2 / 4)
  let α := Real.pi / 2 -- Maximize with sin α = 1
  x * m^2 * Real.sin α / 6

theorem tetrahedron_max_volume : ∀ (AC AB BD CD : ℝ),
  AC = 1 → AB = 1 → BD = 1 → CD = 1 →
  tetrahedron_volume AC AB BD CD = 2 * Real.sqrt 3 / 27 :=
by
  intros AC AB BD CD hAC hAB hBD hCD
  rw [hAC, hAB, hBD, hCD]
  dsimp [tetrahedron_volume]
  norm_num
  sorry

end tetrahedron_max_volume_l185_185043


namespace probability_square_area_l185_185119

theorem probability_square_area (AB : ℝ) (M : ℝ) (h1 : AB = 12) (h2 : 0 ≤ M) (h3 : M ≤ AB) :
  (∃ (AM : ℝ), (AM = M) ∧ (36 ≤ AM^2 ∧ AM^2 ≤ 81)) → 
  (∃ (p : ℝ), p = 1/4) :=
by
  sorry

end probability_square_area_l185_185119


namespace complex_division_l185_185728

theorem complex_division :
  (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = (⟨3, 2⟩ : ℂ) :=
sorry

end complex_division_l185_185728


namespace neg_exists_equiv_forall_l185_185616

theorem neg_exists_equiv_forall (p : ∃ n : ℕ, 2^n > 1000) :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ ∀ n : ℕ, 2^n ≤ 1000 := 
sorry

end neg_exists_equiv_forall_l185_185616


namespace cos_225_l185_185293

theorem cos_225 (h : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2) :
    Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_l185_185293


namespace correct_completion_l185_185442

theorem correct_completion (A B C D : String) : C = "None" :=
by
  let sentence := "Did you have any trouble with the customs officer? " ++ C ++ " to speak of."
  let correct_sentence := "Did you have any trouble with the customs officer? None to speak of."
  sorry

end correct_completion_l185_185442


namespace zero_in_interval_l185_185111

theorem zero_in_interval (a b : ℝ) (ha : 1 < a) (hb : 0 < b ∧ b < 1) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ (a^x + x - b = 0) :=
by {
  sorry
}

end zero_in_interval_l185_185111


namespace pedro_more_squares_l185_185199

theorem pedro_more_squares
  (jesus_squares : ℕ)
  (linden_squares : ℕ)
  (pedro_squares : ℕ)
  (jesus_linden_combined : jesus_squares + linden_squares = 135)
  (pedro_total : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 :=
by
  sorry

end pedro_more_squares_l185_185199


namespace leading_digits_sum_l185_185268

-- Define the conditions
def M : ℕ := (888888888888888888888888888888888888888888888888888888888888888888888888888888) -- define the 400-digit number
-- Assume the function g(r) which finds the leading digit of the r-th root of M

/-- 
  Function g(r) definition:
  It extracts the leading digit of the r-th root of the given number M.
-/
noncomputable def g (r : ℕ) : ℕ := sorry

-- Define the problem statement in Lean 4
theorem leading_digits_sum :
  g 3 + g 4 + g 5 + g 6 + g 7 = 8 :=
sorry

end leading_digits_sum_l185_185268


namespace quadratic_decreasing_conditions_l185_185122

theorem quadratic_decreasing_conditions (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → ∃ y : ℝ, y = ax^2 + 4*(a+1)*x - 3 ∧ (∀ z : ℝ, z ≥ x → y ≥ (ax^2 + 4*(a+1)*z - 3))) ↔ a ∈ Set.Iic (-1 / 2) :=
sorry

end quadratic_decreasing_conditions_l185_185122


namespace jane_output_increase_l185_185437

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l185_185437


namespace bus_speed_excluding_stoppages_l185_185057

theorem bus_speed_excluding_stoppages 
  (v_s : ℕ) -- Speed including stoppages in kmph
  (stop_duration_minutes : ℕ) -- Duration of stoppages in minutes per hour
  (stop_duration_fraction : ℚ := stop_duration_minutes / 60) -- Fraction of hour stopped
  (moving_fraction : ℚ := 1 - stop_duration_fraction) -- Fraction of hour moving
  (distance_per_hour : ℚ := v_s) -- Distance traveled per hour including stoppages
  (v : ℚ) -- Speed excluding stoppages
  
  (h1 : v_s = 50)
  (h2 : stop_duration_minutes = 10)
  
  -- Equation representing the total distance equals the distance traveled moving
  (h3 : v * moving_fraction = distance_per_hour)
: v = 60 := sorry

end bus_speed_excluding_stoppages_l185_185057


namespace pairs_divisible_by_4_l185_185642

-- Define the set of valid pairs of digits from 00 to 99
def valid_pairs : List (Fin 100) := List.filter (λ n => n % 4 = 0) (List.range 100)

-- State the theorem
theorem pairs_divisible_by_4 : valid_pairs.length = 25 := by
  sorry

end pairs_divisible_by_4_l185_185642


namespace one_greater_than_one_l185_185943

theorem one_greater_than_one (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∨ b > 1 ∨ c > 1 :=
by
  sorry

end one_greater_than_one_l185_185943


namespace problem_statement_l185_185636

noncomputable def f (x : ℝ) : ℝ := 3^x + 3^(-x)

noncomputable def g (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) :=
by {
  sorry
}

end problem_statement_l185_185636


namespace knight_reachability_l185_185916

theorem knight_reachability (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) :
  (p + q) % 2 = 1 ∧ Nat.gcd p q = 1 ↔
  ∀ x y x' y', ∃ k h n m, x' = x + k * p + h * q ∧ y' = y + n * p + m * q :=
by
  sorry

end knight_reachability_l185_185916


namespace sushi_eating_orders_l185_185422

/-- Define a 2 x 3 grid with sushi pieces being distinguishable -/
inductive SushiPiece : Type
| A | B | C | D | E | F

open SushiPiece

/-- A function that counts the valid orders to eat sushi pieces satisfying the given conditions -/
noncomputable def countValidOrders : Nat :=
  sorry -- This is where the proof would go, stating the number of valid orders

theorem sushi_eating_orders :
  countValidOrders = 360 :=
sorry -- Skipping proof details

end sushi_eating_orders_l185_185422


namespace jana_distance_travel_in_20_minutes_l185_185640

theorem jana_distance_travel_in_20_minutes :
  ∀ (usual_pace half_pace double_pace : ℚ)
    (first_15_minutes_distance second_5_minutes_distance total_distance : ℚ),
  usual_pace = 1 / 30 →
  half_pace = usual_pace / 2 →
  double_pace = usual_pace * 2 →
  first_15_minutes_distance = 15 * half_pace →
  second_5_minutes_distance = 5 * double_pace →
  total_distance = first_15_minutes_distance + second_5_minutes_distance →
  total_distance = 7 / 12 := 
by
  intros
  sorry

end jana_distance_travel_in_20_minutes_l185_185640


namespace g_g_x_has_exactly_4_distinct_real_roots_l185_185574

noncomputable def g (d x : ℝ) : ℝ := x^2 + 8*x + d

theorem g_g_x_has_exactly_4_distinct_real_roots (d : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ g d (g d x1) = 0 ∧ g d (g d x2) = 0 ∧ g d (g d x3) = 0 ∧ g d (g d x4) = 0) ↔ d < 4 := by {
  sorry
}

end g_g_x_has_exactly_4_distinct_real_roots_l185_185574


namespace simplify_fraction_l185_185883

variable (c : ℝ)

theorem simplify_fraction :
  (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := 
by 
  sorry

end simplify_fraction_l185_185883


namespace lyka_saving_per_week_l185_185653

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end lyka_saving_per_week_l185_185653


namespace apples_purchased_l185_185743

variable (A : ℕ) -- Let A be the number of kg of apples purchased.

-- Conditions
def cost_of_apples (A : ℕ) : ℕ := 70 * A
def cost_of_mangoes : ℕ := 45 * 9
def total_amount_paid : ℕ := 965

-- Theorem to prove that A == 8
theorem apples_purchased
  (h : cost_of_apples A + cost_of_mangoes = total_amount_paid) :
  A = 8 := by
sorry

end apples_purchased_l185_185743


namespace juliette_and_marco_money_comparison_l185_185679

noncomputable def euro_to_dollar (eur : ℝ) : ℝ := eur * 1.5

theorem juliette_and_marco_money_comparison :
  (600 - euro_to_dollar 350) / 600 * 100 = 12.5 := by
sorry

end juliette_and_marco_money_comparison_l185_185679


namespace emily_workers_needed_l185_185566

noncomputable def least_workers_needed
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ) :
  ℕ :=
  (remaining_work / remaining_days) / (work_done / initial_days / total_workers) * total_workers

theorem emily_workers_needed 
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 40)
  (h2 : initial_days = 10)
  (h3 : total_workers = 12)
  (h4 : work_done = 40)
  (h5 : remaining_work = 60)
  (h6 : remaining_days = 30) :
  least_workers_needed total_days initial_days total_workers work_done remaining_work remaining_days = 6 := 
sorry

end emily_workers_needed_l185_185566


namespace perpendicular_tangent_line_l185_185263

theorem perpendicular_tangent_line :
  ∃ m : ℝ, ∃ x₀ : ℝ, y₀ = x₀ ^ 3 + 3 * x₀ ^ 2 - 1 ∧ y₀ = -3 * x₀ + m ∧ 
  (∀ x, x ≠ x₀ → x ^ 3 + 3 * x ^ 2 - 1 ≠ -3 * x + m) ∧ m = -2 := 
sorry

end perpendicular_tangent_line_l185_185263


namespace barbi_monthly_loss_l185_185938

variable (x : Real)

theorem barbi_monthly_loss : 
  (∃ x : Real, 12 * x = 99 - 81) → x = 1.5 :=
by
  sorry

end barbi_monthly_loss_l185_185938


namespace prime_sum_diff_l185_185473

open Nat

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- The problem statement
theorem prime_sum_diff (p : ℕ) (q s r t : ℕ) :
  is_prime p → is_prime q → is_prime s → is_prime r → is_prime t →
  p = q + s → p = r - t → p = 5 :=
by
  sorry

end prime_sum_diff_l185_185473


namespace abs_sum_condition_l185_185187

theorem abs_sum_condition (a b : ℝ) (h₁ : |a| = 2) (h₂ : b = -1) : |a + b| = 1 ∨ |a + b| = 3 :=
by
  sorry

end abs_sum_condition_l185_185187


namespace union_eq_M_l185_185213

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def S : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem union_eq_M : M ∪ S = M := by
  /- this part is for skipping the proof -/
  sorry

end union_eq_M_l185_185213


namespace calculate_expression_l185_185096

theorem calculate_expression : (3072 - 2993) ^ 2 / 121 = 49 :=
by
  sorry

end calculate_expression_l185_185096


namespace range_of_objective_function_l185_185548

def objective_function (x y : ℝ) : ℝ := 3 * x - y

theorem range_of_objective_function (x y : ℝ) 
  (h1 : x + 2 * y ≥ 2)
  (h2 : 2 * x + y ≤ 4)
  (h3 : 4 * x - y ≥ -1)
  : - 3 / 2 ≤ objective_function x y ∧ objective_function x y ≤ 6 := 
sorry

end range_of_objective_function_l185_185548


namespace total_number_of_balls_l185_185003

def number_of_yellow_balls : Nat := 6
def probability_yellow_ball : Rat := 1 / 9

theorem total_number_of_balls (N : Nat) (h1 : number_of_yellow_balls = 6) (h2 : probability_yellow_ball = 1 / 9) :
    6 / N = 1 / 9 → N = 54 := 
by
  sorry

end total_number_of_balls_l185_185003


namespace complementary_event_A_l185_185851

def EventA (n : ℕ) := n ≥ 2

def ComplementaryEventA (n : ℕ) := n ≤ 1

theorem complementary_event_A (n : ℕ) : ComplementaryEventA n ↔ ¬ EventA n := by
  sorry

end complementary_event_A_l185_185851


namespace incorrect_solution_among_four_l185_185494

theorem incorrect_solution_among_four 
  (x y : ℤ) 
  (h1 : 2 * x - 3 * y = 5) 
  (h2 : 3 * x - 2 * y = 7) : 
  ¬ ((2 * (2 * x - 3 * y) - ((-3) * (3 * x - 2 * y))) = (2 * 5 - (-3) * 7)) :=
sorry

end incorrect_solution_among_four_l185_185494


namespace find_solutions_l185_185060

theorem find_solutions (k : ℤ) (x y : ℤ) (h : x^2 - 2*y^2 = k) :
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ (t = x + 2*y ∨ t = x - 2*y) ∧ (u = x + y ∨ u = x - y) :=
sorry

end find_solutions_l185_185060


namespace solve_system_l185_185730

theorem solve_system :
  ∃ x y : ℝ, x - y = 1 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 := by
  sorry

end solve_system_l185_185730


namespace find_x_l185_185380

-- Defining the conditions
def angle_PQR : ℝ := 180
def angle_PQS : ℝ := 125
def angle_QSR (x : ℝ) : ℝ := x
def SQ_eq_SR : Prop := true -- Assuming an isosceles triangle where SQ = SR.

-- The theorem to be proved
theorem find_x (x : ℝ) :
  angle_PQR = 180 → angle_PQS = 125 → SQ_eq_SR → angle_QSR x = 70 :=
by
  intros _ _ _
  sorry

end find_x_l185_185380


namespace second_polygon_sides_l185_185940

theorem second_polygon_sides (a b n m : ℕ) (s : ℝ) 
  (h1 : a = 45) 
  (h2 : b = 3 * s)
  (h3 : n * b = m * s)
  (h4 : n = 45) : m = 135 := 
by
  sorry

end second_polygon_sides_l185_185940


namespace pyramid_volume_inequality_l185_185477

theorem pyramid_volume_inequality
  (k : ℝ)
  (OA1 OB1 OC1 OA2 OB2 OC2 OA3 OB3 OC3 OB2 : ℝ)
  (V1 := k * |OA1| * |OB1| * |OC1|)
  (V2 := k * |OA2| * |OB2| * |OC2|)
  (V3 := k * |OA3| * |OB3| * |OC3|)
  (V := k * |OA1| * |OB2| * |OC3|) :
  V ≤ (V1 + V2 + V3) / 3 := 
  sorry

end pyramid_volume_inequality_l185_185477


namespace suma_work_rate_l185_185102

theorem suma_work_rate (r s : ℝ) (hr : r = 1 / 5) (hrs : r + s = 1 / 4) : 1 / s = 20 := by
  sorry

end suma_work_rate_l185_185102


namespace line_tangent_to_parabola_l185_185963

theorem line_tangent_to_parabola (k : ℝ) (x₀ y₀ : ℝ) 
  (h₁ : y₀ = k * x₀ - 2) 
  (h₂ : x₀^2 = 4 * y₀) 
  (h₃ : ∀ x y, (x = x₀ ∧ y = y₀) → (k = (1/2) * x₀)) :
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := 
sorry

end line_tangent_to_parabola_l185_185963


namespace subset_implies_value_l185_185301

theorem subset_implies_value (a : ℝ) : (∀ x ∈ ({0, -a} : Set ℝ), x ∈ ({1, -1, 2 * a - 2} : Set ℝ)) → a = 1 := by
  sorry

end subset_implies_value_l185_185301


namespace convert_deg_to_min_compare_negatives_l185_185180

theorem convert_deg_to_min : (0.3 : ℝ) * 60 = 18 :=
by sorry

theorem compare_negatives : -2 > -3 :=
by sorry

end convert_deg_to_min_compare_negatives_l185_185180


namespace cumulus_to_cumulonimbus_ratio_l185_185136

theorem cumulus_to_cumulonimbus_ratio (cirrus cumulonimbus cumulus : ℕ) (x : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = x * cumulonimbus)
  (h3 : cumulonimbus = 3)
  (h4 : cirrus = 144) :
  x = 12 := by
  sorry

end cumulus_to_cumulonimbus_ratio_l185_185136


namespace find_n_l185_185264

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
sorry

end find_n_l185_185264


namespace cost_price_correct_l185_185547

open Real

-- Define the cost price of the table
def cost_price (C : ℝ) : ℝ := C

-- Define the marked price
def marked_price (C : ℝ) : ℝ := 1.30 * C

-- Define the discounted price
def discounted_price (C : ℝ) : ℝ := 0.85 * (marked_price C)

-- Define the final price after sales tax
def final_price (C : ℝ) : ℝ := 1.12 * (discounted_price C)

-- Given that the final price is 9522.84
axiom final_price_value : final_price 9522.84 = 1.2376 * 7695

-- Main theorem stating the problem to prove
theorem cost_price_correct (C : ℝ) : final_price C = 9522.84 -> C = 7695 := by
  sorry

end cost_price_correct_l185_185547


namespace sum_six_smallest_multiples_of_eleven_l185_185073

theorem sum_six_smallest_multiples_of_eleven : 
  (11 + 22 + 33 + 44 + 55 + 66) = 231 :=
by
  sorry

end sum_six_smallest_multiples_of_eleven_l185_185073


namespace a_n_strictly_monotonic_increasing_l185_185049

noncomputable def a_n (n : ℕ) : ℝ := 
  2 * ((1 + 1 / (n : ℝ)) ^ (2 * n + 1)) / (((1 + 1 / (n : ℝ)) ^ n) + ((1 + 1 / (n : ℝ)) ^ (n + 1)))

theorem a_n_strictly_monotonic_increasing : ∀ n : ℕ, a_n (n + 1) > a_n n :=
sorry

end a_n_strictly_monotonic_increasing_l185_185049


namespace fresh_fruit_sold_l185_185346

-- Define the conditions
def total_fruit_sold : ℕ := 9792
def frozen_fruit_sold : ℕ := 3513

-- Define what we need to prove
theorem fresh_fruit_sold : (total_fruit_sold - frozen_fruit_sold = 6279) := by
  sorry

end fresh_fruit_sold_l185_185346


namespace remainder_of_polynomial_l185_185134

theorem remainder_of_polynomial :
  ∀ (x : ℂ), (x^4 + x^3 + x^2 + x + 1 = 0) → (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^4 + x^3 + x^2 + x + 1) = 2 :=
by
  intro x hx
  sorry

end remainder_of_polynomial_l185_185134


namespace solve_arithmetic_sequence_sum_l185_185253

noncomputable def arithmetic_sequence_sum : ℕ :=
  let a : ℕ := 3
  let b : ℕ := 10
  let c : ℕ := 17
  let e : ℕ := 32
  let d := b - a
  let c_term := c + d
  let d_term := c_term + d
  c_term + d_term

theorem solve_arithmetic_sequence_sum : arithmetic_sequence_sum = 55 :=
by
  sorry

end solve_arithmetic_sequence_sum_l185_185253


namespace cyclists_travel_same_distance_l185_185447

-- Define constants for speeds
def v1 := 12   -- speed of the first cyclist in km/h
def v2 := 16   -- speed of the second cyclist in km/h
def v3 := 24   -- speed of the third cyclist in km/h

-- Define the known total time
def total_time := 3  -- total time in hours

-- Hypothesis: Prove that the distance traveled by each cyclist is 16 km
theorem cyclists_travel_same_distance (d : ℚ) : 
  (v1 * (total_time * 3 / 13)) = d ∧
  (v2 * (total_time * 4 / 13)) = d ∧
  (v3 * (total_time * 6 / 13)) = d ∧
  d = 16 :=
by
  sorry

end cyclists_travel_same_distance_l185_185447


namespace total_height_of_pipes_l185_185375

theorem total_height_of_pipes 
  (diameter : ℝ) (radius : ℝ) (total_pipes : ℕ) (first_row_pipes : ℕ) (second_row_pipes : ℕ) 
  (h : ℝ) 
  (h_diam : diameter = 10)
  (h_radius : radius = 5)
  (h_total_pipes : total_pipes = 5)
  (h_first_row : first_row_pipes = 2)
  (h_second_row : second_row_pipes = 3) :
  h = 10 + 5 * Real.sqrt 3 := 
sorry

end total_height_of_pipes_l185_185375


namespace percentage_is_26_53_l185_185181

noncomputable def percentage_employees_with_six_years_or_more (y: ℝ) : ℝ :=
  let total_employees := 10*y + 4*y + 6*y + 5*y + 8*y + 3*y + 5*y + 4*y + 2*y + 2*y
  let employees_with_six_years_or_more := 5*y + 4*y + 2*y + 2*y
  (employees_with_six_years_or_more / total_employees) * 100

theorem percentage_is_26_53 (y: ℝ) (hy: y ≠ 0): percentage_employees_with_six_years_or_more y = 26.53 :=
by
  sorry

end percentage_is_26_53_l185_185181


namespace min_green_beads_l185_185125

theorem min_green_beads (B R G : ℕ) (h : B + R + G = 80)
  (hB : ∀ i j : ℕ, (i < j ∧ j ≤ B → ∃ k, i < k ∧ k < j ∧ k ≤ R)) 
  (hR : ∀ i j : ℕ, (i < j ∧ j ≤ R → ∃ k, i < k ∧ k < j ∧ k ≤ G)) :
  G >= 27 := 
sorry

end min_green_beads_l185_185125


namespace two_numbers_ratio_l185_185610

theorem two_numbers_ratio (A B : ℕ) (h_lcm : Nat.lcm A B = 30) (h_sum : A + B = 25) :
  ∃ x y : ℕ, x = 2 ∧ y = 3 ∧ A / B = x / y := 
sorry

end two_numbers_ratio_l185_185610


namespace path_count_l185_185732

theorem path_count :
  let is_valid_path (path : List (ℕ × ℕ)) : Prop :=
    ∃ (n : ℕ), path = List.range n    -- This is a simplification for definition purposes
  let count_paths_outside_square (start finish : (ℤ × ℤ)) (steps : ℕ) : ℕ :=
    43826                              -- Hardcoded the result as this is the correct answer
  ∀ start finish : (ℤ × ℤ),
    start = (-5, -5) → 
    finish = (5, 5) → 
    count_paths_outside_square start finish 20 = 43826
:= 
sorry

end path_count_l185_185732


namespace minimum_value_l185_185448

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem minimum_value (x : ℝ) (h : x > 10) : (∃ y : ℝ, (∀ x' : ℝ, x' > 10 → f x' ≥ y) ∧ y = 40) := 
sorry

end minimum_value_l185_185448


namespace problem_I_problem_II_l185_185359

-- Define the function f(x) = |x+1| + |x+m+1|
def f (x : ℝ) (m : ℝ) : ℝ := |x+1| + |x+(m+1)|

-- Define the problem (Ⅰ): f(x) ≥ |m-2| for all x implies m ≥ 1
theorem problem_I (m : ℝ) (h : ∀ x : ℝ, f x m ≥ |m-2|) : m ≥ 1 := sorry

-- Define the problem (Ⅱ): Find the solution set for f(-x) < 2m
theorem problem_II (m : ℝ) :
  (m ≤ 0 → ∀ x : ℝ, ¬ (f (-x) m < 2 * m)) ∧
  (m > 0 → ∀ x : ℝ, (1 - m / 2 < x ∧ x < 3 * m / 2 + 1) ↔ f (-x) m < 2 * m) := sorry

end problem_I_problem_II_l185_185359


namespace probability_pair_tile_l185_185023

def letters_in_word : List Char := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
def target_letters : List Char := ['P', 'A', 'I', 'R']

def num_favorable_outcomes : Nat :=
  -- Count occurrences of letters in target_letters within letters_in_word
  List.count 'P' letters_in_word +
  List.count 'A' letters_in_word +
  List.count 'I' letters_in_word + 
  List.count 'R' letters_in_word

def total_outcomes : Nat := letters_in_word.length

theorem probability_pair_tile :
  (num_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

end probability_pair_tile_l185_185023


namespace ellipse_iff_constant_sum_l185_185185

-- Let F_1 and F_2 be two fixed points in the plane.
variables (F1 F2 : Point)
-- Let d be a constant.
variable (d : ℝ)

-- A point M in a plane
variable (M : Point)

-- Define the distance function between two points.
def dist (P Q : Point) : ℝ := sorry

-- Definition: M is on an ellipse with foci F1 and F2
def on_ellipse (M F1 F2 : Point) (d : ℝ) : Prop :=
  dist M F1 + dist M F2 = d

-- Proof that shows the two parts of the statement
theorem ellipse_iff_constant_sum :
  (∀ M, on_ellipse M F1 F2 d) ↔ (∀ M, dist M F1 + dist M F2 = d) ∧ d > dist F1 F2 :=
sorry

end ellipse_iff_constant_sum_l185_185185


namespace pentagon_angles_l185_185832

def is_point_in_convex_pentagon (O A B C D E : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry -- Assume definition of angle in radians

theorem pentagon_angles (O A B C D E: Point) (hO : is_point_in_convex_pentagon O A B C D E)
  (h1: angle A O B = angle B O C) (h2: angle B O C = angle C O D)
  (h3: angle C O D = angle D O E) (h4: angle D O E = angle E O A) :
  (angle E O A = angle A O B) ∨ (angle E O A + angle A O B = π) :=
sorry

end pentagon_angles_l185_185832


namespace simplify_complex_expression_l185_185637

noncomputable def i : ℂ := Complex.I

theorem simplify_complex_expression : 
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i := by 
  sorry

end simplify_complex_expression_l185_185637


namespace probability_forming_more_from_remont_probability_forming_papa_from_papaha_l185_185881

-- Definition for part (a)
theorem probability_forming_more_from_remont : 
  (6 * 5 * 4 * 3 = 360) ∧ (1 / 360 = 0.00278) :=
by
  sorry

-- Definition for part (b)
theorem probability_forming_papa_from_papaha : 
  (6 * 5 * 4 * 3 = 360) ∧ (12 / 360 = 0.03333) :=
by
  sorry

end probability_forming_more_from_remont_probability_forming_papa_from_papaha_l185_185881


namespace initial_percentage_increase_l185_185786

variable (P : ℝ) (x : ℝ)

theorem initial_percentage_increase :
  (P * (1 + x / 100) * 1.3 = P * 1.625) → (x = 25) := by
  sorry

end initial_percentage_increase_l185_185786


namespace yellow_peaches_l185_185261

theorem yellow_peaches (red_peaches green_peaches total_green_yellow_peaches : ℕ)
  (h1 : red_peaches = 5)
  (h2 : green_peaches = 6)
  (h3 : total_green_yellow_peaches = 20) :
  (total_green_yellow_peaches - green_peaches) = 14 :=
by
  sorry

end yellow_peaches_l185_185261


namespace exists_colored_subset_l185_185994

theorem exists_colored_subset (n : ℕ) (h_positive : n > 0) (colors : ℕ → ℕ) (h_colors : ∀ a b : ℕ, a < b → a + b ≤ n → 
  (colors a = colors b ∨ colors b = colors (a + b) ∨ colors a = colors (a + b))) :
  ∃ c, ∃ s : Finset ℕ, s.card ≥ (2 * n / 5) ∧ ∀ x ∈ s, colors x = c :=
sorry

end exists_colored_subset_l185_185994


namespace student_failed_by_40_marks_l185_185133

theorem student_failed_by_40_marks (total_marks : ℕ) (passing_percentage : ℝ) (marks_obtained : ℕ) (h1 : total_marks = 500) (h2 : passing_percentage = 33) (h3 : marks_obtained = 125) :
  ((passing_percentage / 100) * total_marks - marks_obtained : ℝ) = 40 :=
sorry

end student_failed_by_40_marks_l185_185133


namespace simplify_and_evaluate_at_x_eq_4_l185_185157

noncomputable def simplify_and_evaluate (x : ℚ) : ℚ :=
  (x - 1 - (3 / (x + 1))) / ((x^2 - 2*x) / (x + 1))

theorem simplify_and_evaluate_at_x_eq_4 : simplify_and_evaluate 4 = 3 / 2 := by
  sorry

end simplify_and_evaluate_at_x_eq_4_l185_185157


namespace A_and_D_mut_exclusive_not_complementary_l185_185334

-- Define the events based on the conditions
inductive Die
| one | two | three | four | five | six

def is_odd (d : Die) : Prop :=
  d = Die.one ∨ d = Die.three ∨ d = Die.five

def is_even (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_multiple_of_2 (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_two_or_four (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four

-- Define the predicate for mutually exclusive but not complementary
def mutually_exclusive_but_not_complementary (P Q : Die → Prop) : Prop :=
  (∀ d, ¬ (P d ∧ Q d)) ∧ ¬ (∀ d, P d ∨ Q d)

-- Verify that "A and D" are mutually exclusive but not complementary
theorem A_and_D_mut_exclusive_not_complementary :
  mutually_exclusive_but_not_complementary is_odd is_two_or_four :=
  by
    sorry

end A_and_D_mut_exclusive_not_complementary_l185_185334


namespace solve_inequality_l185_185212

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem solve_inequality (x : ℝ) : (otimes (x-2) (x+2) < 2) ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 1 :=
by
  sorry

end solve_inequality_l185_185212


namespace sum_of_consecutive_integers_sqrt_28_l185_185657

theorem sum_of_consecutive_integers_sqrt_28 (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 28) (h3 : Real.sqrt 28 < b) : a + b = 11 :=
by 
    sorry

end sum_of_consecutive_integers_sqrt_28_l185_185657


namespace nurses_quit_count_l185_185439

-- Initial Definitions
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def doctors_quit : ℕ := 5
def total_remaining_staff : ℕ := 22

-- Remaining Doctors Calculation
def remaining_doctors : ℕ := initial_doctors - doctors_quit

-- Theorem to prove the number of nurses who quit
theorem nurses_quit_count : initial_nurses - (total_remaining_staff - remaining_doctors) = 2 := by
  sorry

end nurses_quit_count_l185_185439


namespace orangeade_price_per_glass_l185_185299

theorem orangeade_price_per_glass (O : ℝ) (W : ℝ) (P : ℝ) (price_1_day : ℝ) 
    (h1 : W = O) (h2 : price_1_day = 0.30) (revenue_equal : 2 * O * price_1_day = 3 * O * P) :
  P = 0.20 :=
by
  sorry

end orangeade_price_per_glass_l185_185299


namespace Amanda_lost_notebooks_l185_185934

theorem Amanda_lost_notebooks (initial_notebooks ordered additional_notebooks remaining_notebooks : ℕ)
  (h1 : initial_notebooks = 10)
  (h2 : ordered = 6)
  (h3 : remaining_notebooks = 14) :
  initial_notebooks + ordered - remaining_notebooks = 2 := by
sorry

end Amanda_lost_notebooks_l185_185934


namespace exercise_felt_weight_l185_185571

variable (n w : ℕ)
variable (p : ℝ)

def total_weight (n : ℕ) (w : ℕ) : ℕ := n * w

def felt_weight (total_weight : ℕ) (p : ℝ) : ℝ := total_weight * (1 + p)

theorem exercise_felt_weight (h1 : n = 10) (h2 : w = 30) (h3 : p = 0.20) : 
  felt_weight (total_weight n w) p = 360 :=
by 
  sorry

end exercise_felt_weight_l185_185571


namespace polyhedra_impossible_l185_185128

noncomputable def impossible_polyhedra_projections (p1_outer : List (ℝ × ℝ)) (p1_inner : List (ℝ × ℝ))
                                                  (p2_outer : List (ℝ × ℝ)) (p2_inner : List (ℝ × ℝ)) : Prop :=
  -- Add definitions for the vertices labeling here 
  let vertices_outer := ["A", "B", "C", "D"]
  let vertices_inner := ["A1", "B1", "C1", "D1"]
  -- Add the conditions for projection (a) and (b) 
  p1_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p1_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] ∧
  p2_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p2_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] →
  -- Prove that the polyhedra corresponding to these projections are impossible.
  false

-- Now let's state the theorem
theorem polyhedra_impossible : impossible_polyhedra_projections [(0,0), (1,0), (1,1), (0,1)] 
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)]
                                                                [(0,0), (1,0), (1,1), (0,1)]
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] := 
by {
  sorry
}

end polyhedra_impossible_l185_185128


namespace peter_drew_8_pictures_l185_185770

theorem peter_drew_8_pictures : 
  ∃ (P : ℕ), ∀ (Q R : ℕ), Q = P + 20 → R = 5 → R + P + Q = 41 → P = 8 :=
by
  sorry

end peter_drew_8_pictures_l185_185770


namespace algorithm_comparable_to_euclidean_l185_185776

-- Define the conditions
def ancient_mathematics_world_leading : Prop := 
  True -- Placeholder representing the historical condition

def song_yuan_algorithm : Prop :=
  True -- Placeholder representing the algorithmic condition

-- The main theorem representing the problem statement
theorem algorithm_comparable_to_euclidean :
  ancient_mathematics_world_leading → song_yuan_algorithm → 
  True :=  -- Placeholder representing that the algorithm is the method of successive subtraction
by 
  intro h1 h2 
  sorry

end algorithm_comparable_to_euclidean_l185_185776


namespace votes_ratio_l185_185703

theorem votes_ratio (V : ℝ) 
  (counted_fraction : ℝ := 2/9) 
  (favor_fraction : ℝ := 3/4) 
  (against_fraction_remaining : ℝ := 0.7857142857142856) :
  let counted := counted_fraction * V
  let favor_counted := favor_fraction * counted
  let remaining := V - counted
  let against_remaining := against_fraction_remaining * remaining
  let against_counted := (1 - favor_fraction) * counted
  let total_against := against_counted + against_remaining
  let total_favor := favor_counted
  (total_against / total_favor) = 4 :=
by
  sorry

end votes_ratio_l185_185703


namespace inscribed_pentagon_angles_sum_l185_185373

theorem inscribed_pentagon_angles_sum (α β γ δ ε : ℝ) (h1 : α + β + γ + δ + ε = 360) 
(h2 : α / 2 + β / 2 + γ / 2 + δ / 2 + ε / 2 = 180) : 
(α / 2) + (β / 2) + (γ / 2) + (δ / 2) + (ε / 2) = 180 :=
by
  sorry

end inscribed_pentagon_angles_sum_l185_185373


namespace six_divisors_third_seven_times_second_fourth_ten_more_than_third_l185_185528

theorem six_divisors_third_seven_times_second_fourth_ten_more_than_third (n : ℕ) :
  (∀ d : ℕ, d ∣ n ↔ d ∈ [1, d2, d3, d4, d5, n]) ∧ 
  (d3 = 7 * d2) ∧ 
  (d4 = d3 + 10) → 
  n = 2891 :=
by
  sorry

end six_divisors_third_seven_times_second_fourth_ten_more_than_third_l185_185528


namespace total_number_of_gifts_l185_185480

/-- Number of gifts calculation, given the distribution conditions with certain children -/
theorem total_number_of_gifts
  (n : ℕ) -- the total number of children
  (h1 : 2 * 4 + (n - 2) * 3 + 11 = 3 * n + 13) -- first scenario equation
  (h2 : 4 * 3 + (n - 4) * 6 + 10 = 6 * n - 2) -- second scenario equation
  : 3 * n + 13 = 28 := 
by 
  sorry

end total_number_of_gifts_l185_185480


namespace brick_width_l185_185544

-- Define the dimensions of the wall
def L_wall : Real := 750 -- length in cm
def W_wall : Real := 600 -- width in cm
def H_wall : Real := 22.5 -- height in cm

-- Define the dimensions of the bricks
def L_brick : Real := 25 -- length in cm
def H_brick : Real := 6 -- height in cm

-- Define the number of bricks needed
def n_bricks : Nat := 6000

-- Define the total volume of the wall
def V_wall : Real := L_wall * W_wall * H_wall

-- Define the volume of one brick
def V_brick (W : Real) : Real := L_brick * W * H_brick

-- Statement to prove
theorem brick_width : 
  ∃ W : Real, V_wall = V_brick W * (n_bricks : Real) ∧ W = 11.25 := by 
  sorry

end brick_width_l185_185544


namespace area_of_abs_inequality_l185_185407

theorem area_of_abs_inequality :
  ∀ (x y : ℝ), |x + 2 * y| + |2 * x - y| ≤ 6 → 
  ∃ (area : ℝ), area = 12 := 
by
  -- This skips the proofs
  sorry

end area_of_abs_inequality_l185_185407


namespace correct_operation_l185_185358

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^3 ≠ 2 * a^5) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  (a^3 * a^5 ≠ a^15) ∧
  ((ab^2)^2 = a^2 * b^4) :=
by
  sorry

end correct_operation_l185_185358


namespace option_D_correct_l185_185826

theorem option_D_correct (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 :=
by sorry

end option_D_correct_l185_185826


namespace cylinder_height_l185_185284

theorem cylinder_height (OA OB : ℝ) (h_OA : OA = 7) (h_OB : OB = 2) :
  ∃ (h_cylinder : ℝ), h_cylinder = 3 * Real.sqrt 5 :=
by
  use (Real.sqrt (OA^2 - OB^2))
  rw [h_OA, h_OB]
  norm_num
  sorry

end cylinder_height_l185_185284


namespace olivia_wallet_after_shopping_l185_185766

variable (initial_wallet : ℝ := 200) 
variable (groceries : ℝ := 65)
variable (shoes_original_price : ℝ := 75)
variable (shoes_discount_rate : ℝ := 0.15)
variable (belt : ℝ := 25)

theorem olivia_wallet_after_shopping :
  initial_wallet - (groceries + (shoes_original_price - shoes_original_price * shoes_discount_rate) + belt) = 46.25 := by
  sorry

end olivia_wallet_after_shopping_l185_185766


namespace cost_of_each_card_is_2_l185_185115

-- Define the conditions
def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def total_spent : ℝ := 70

-- Define the total number of cards
def total_cards : ℕ := christmas_cards + birthday_cards

-- Define the cost per card
noncomputable def cost_per_card : ℝ := total_spent / total_cards

-- The theorem
theorem cost_of_each_card_is_2 : cost_per_card = 2 := by
  sorry

end cost_of_each_card_is_2_l185_185115


namespace M_subset_N_l185_185719

open Set

noncomputable def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
noncomputable def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := 
sorry

end M_subset_N_l185_185719


namespace part1_part2_l185_185320

-- Condition for exponents of x to be equal
def condition1 (a : ℤ) : Prop := (3 : ℤ) = 2 * a - 3

-- Condition for exponents of y to be equal
def condition2 (b : ℤ) : Prop := b = 1

noncomputable def a_value : ℤ := 3
noncomputable def b_value : ℤ := 1

-- Theorem for part (1): values of a and b
theorem part1 : condition1 3 ∧ condition2 1 :=
by
  have ha : condition1 3 := by sorry
  have hb : condition2 1 := by sorry
  exact And.intro ha hb

-- Theorem for part (2): value of (7a - 22)^2024 given a = 3
theorem part2 : (7 * a_value - 22) ^ 2024 = 1 :=
by
  have hx : 7 * a_value - 22 = -1 := by sorry
  have hres : (-1) ^ 2024 = 1 := by sorry
  exact Eq.trans (congrArg (fun x => x ^ 2024) hx) hres

end part1_part2_l185_185320


namespace angle_C_in_triangle_l185_185270

open Real

noncomputable def determine_angle_C (A B C: ℝ) (AB BC: ℝ) : Prop :=
  A = (3 * π) / 4 ∧ BC = sqrt 2 * AB → C = π / 6

-- Define the theorem to state the problem
theorem angle_C_in_triangle (A B C : ℝ) (AB BC : ℝ) :
  determine_angle_C A B C AB BC := 
by
  -- Step to indicate where the proof would be
  sorry

end angle_C_in_triangle_l185_185270


namespace candy_weight_reduction_l185_185045

theorem candy_weight_reduction:
  ∀ (W P : ℝ), (33.333333333333314 / 100) * (P / W) = (P / (W - (1/4) * W)) →
  (1 - (W - (1/4) * W) / W) * 100 = 25 :=
by
  intros W P h
  sorry

end candy_weight_reduction_l185_185045


namespace evaluate_f_at_3_l185_185682

def f (x : ℤ) : ℤ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem evaluate_f_at_3 : f 3 = 181 := by
  sorry

end evaluate_f_at_3_l185_185682


namespace find_original_number_l185_185178

theorem find_original_number (x : ℕ) (h1 : 10 * x + 9 + 2 * x = 633) : x = 52 :=
by
  sorry

end find_original_number_l185_185178


namespace line_passes_through_center_l185_185754

-- Define the equation of the circle as given in the problem.
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the center of the circle.
def center_of_circle (x y : ℝ) : Prop := x = 1 ∧ y = -3

-- Define the equation of the line.
def line_equation (x y : ℝ) : Prop := 2*x + y + 1 = 0

-- The theorem to prove.
theorem line_passes_through_center :
  (∃ x y, circle_equation x y ∧ center_of_circle x y) →
  (∃ x y, center_of_circle x y ∧ line_equation x y) :=
by
  sorry

end line_passes_through_center_l185_185754


namespace solve_for_a_and_b_l185_185639

noncomputable def A := {x : ℝ | (-2 < x ∧ x < -1) ∨ (x > 1)}
noncomputable def B (a b : ℝ) := {x : ℝ | a ≤ x ∧ x < b}

theorem solve_for_a_and_b (a b : ℝ) :
  (A ∪ B a b = {x : ℝ | x > -2}) ∧ (A ∩ B a b = {x : ℝ | 1 < x ∧ x < 3}) →
  a = -1 ∧ b = 3 :=
by
  sorry

end solve_for_a_and_b_l185_185639


namespace restaurant_pizzas_more_than_hotdogs_l185_185453

theorem restaurant_pizzas_more_than_hotdogs
  (H P : ℕ) 
  (h1 : H = 60)
  (h2 : 30 * (P + H) = 4800) :
  P - H = 40 :=
by
  sorry

end restaurant_pizzas_more_than_hotdogs_l185_185453


namespace part_I_part_II_l185_185374

-- Part I
theorem part_I :
  ∀ (x_0 y_0 : ℝ),
  (x_0 ^ 2 + y_0 ^ 2 = 8) ∧
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) →
  ∃ a b : ℝ, (a = 2 ∧ b = 2) →
  (∀ x y : ℝ, (x - 2) ^ 2 + (y - 2) ^ 2 = 8) :=
by 
sorry

-- Part II
theorem part_II :
  ¬ ∃ (x_0 y_0 k_1 k_2 : ℝ),
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) ∧
  (k_1k_2 = (y_0^2 - 4) / (x_0^2 - 4)) ∧
  (k_1 + k_2 = 2 * x_0 * y_0 / (x_0^2 - 4)) ∧
  (k_1k_2 - (k_1 + k_2) / (x_0 * y_0) + 1 = 0) :=
by 
sorry

end part_I_part_II_l185_185374


namespace modular_inverse_of_31_mod_35_is_1_l185_185591

theorem modular_inverse_of_31_mod_35_is_1 :
  ∃ a : ℕ, 0 ≤ a ∧ a < 35 ∧ 31 * a % 35 = 1 := sorry

end modular_inverse_of_31_mod_35_is_1_l185_185591


namespace exclude_chairs_l185_185587

-- Definitions
def total_chairs : ℕ := 10000
def perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Statement
theorem exclude_chairs (n : ℕ) (h₁ : n = total_chairs) :
  perfect_square n → (n - total_chairs) = 0 := 
sorry

end exclude_chairs_l185_185587


namespace distance_from_P_to_origin_l185_185486

open Real -- This makes it easier to use real number functions and constants.

noncomputable def hyperbola := { P : ℝ × ℝ // (P.1^2 / 9) - (P.2^2 / 7) = 1 }

theorem distance_from_P_to_origin 
  (P : ℝ × ℝ) 
  (hP : (P.1^2 / 9) - (P.2^2 / 7) = 1)
  (d_right_focus : P.1 - 4 = -1) : 
  dist P (0, 0) = 3 :=
sorry

end distance_from_P_to_origin_l185_185486


namespace math_problem_l185_185182

open Real

variables {a b c d e f : ℝ}

theorem math_problem 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hcond : abs (sqrt (a * b) - sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) :=
sorry

end math_problem_l185_185182


namespace base_conversion_sum_l185_185397

def A := 10
def B := 11

def convert_base11_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 11^2
  let d1 := (n % 11^2) / 11
  let d0 := n % 11
  d2 * 11^2 + d1 * 11 + d0

def convert_base12_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 12^2
  let d1 := (n % 12^2) / 12
  let d0 := n % 12
  d2 * 12^2 + d1 * 12 + d0

def n1 := 2 * 11^2 + 4 * 11 + 9    -- = 249_11 in base 10
def n2 := 3 * 12^2 + A * 12 + B   -- = 3AB_12 in base 10

theorem base_conversion_sum :
  (convert_base11_to_base10 294 + convert_base12_to_base10 563 = 858) := by
  sorry

end base_conversion_sum_l185_185397


namespace bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l185_185631

def bernardo (x : ℕ) : ℕ := 2 * x
def silvia (x : ℕ) : ℕ := x + 30

theorem bernardo_winning_N_initial (N : ℕ) :
  (∃ k : ℕ, (bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia $ bernardo $ silvia N) = k
  ∧ 950 ≤ k ∧ k ≤ 999)
  → 34 ≤ N ∧ N ≤ 35 :=
by
  sorry

theorem bernardo_smallest_N (N : ℕ) (h : 34 ≤ N ∧ N ≤ 35) :
  (N = 34) :=
by
  sorry

theorem sum_of_digits_34 :
  (3 + 4 = 7) :=
by
  sorry

end bernardo_winning_N_initial_bernardo_smallest_N_sum_of_digits_34_l185_185631


namespace matrix_inverse_l185_185254

variable (N : Matrix (Fin 2) (Fin 2) ℚ) 
variable (I : Matrix (Fin 2) (Fin 2) ℚ)
variable (c d : ℚ)

def M1 : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def M2 : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem matrix_inverse (hN : N = M1) 
                       (hI : I = M2) 
                       (hc : c = 1/12) 
                       (hd : d = 1/12) :
                       N⁻¹ = c • N + d • I := by
  sorry

end matrix_inverse_l185_185254


namespace distinct_sequences_count_l185_185311

def letters := ["E", "Q", "U", "A", "L", "S"]

noncomputable def count_sequences : Nat :=
  let remaining_letters := ["E", "Q", "U", "A"] -- 'L' and 'S' are already considered
  3 * (4 * 3) -- as analyzed: (LS__) + (L_S_) + (L__S)

theorem distinct_sequences_count : count_sequences = 36 := 
  by
    unfold count_sequences
    sorry

end distinct_sequences_count_l185_185311


namespace impossible_sum_of_two_smaller_angles_l185_185984

theorem impossible_sum_of_two_smaller_angles
  {α β γ : ℝ}
  (h1 : α + β + γ = 180)
  (h2 : 0 < α + β ∧ α + β < 180) :
  α + β ≠ 130 :=
sorry

end impossible_sum_of_two_smaller_angles_l185_185984


namespace necessary_and_sufficient_for_perpendicular_l185_185629

theorem necessary_and_sufficient_for_perpendicular (a : ℝ) :
  (a = -2) ↔ (∀ (x y : ℝ), x + 2 * y = 0 → ax + y = 1 → false) :=
by
  sorry

end necessary_and_sufficient_for_perpendicular_l185_185629


namespace bike_shop_profit_l185_185460

theorem bike_shop_profit :
  let tire_repair_charge := 20
  let tire_repair_cost := 5
  let tire_repairs_per_month := 300
  let complex_repair_charge := 300
  let complex_repair_cost := 50
  let complex_repairs_per_month := 2
  let retail_profit := 2000
  let fixed_expenses := 4000
  let total_tire_profit := tire_repairs_per_month * (tire_repair_charge - tire_repair_cost)
  let total_complex_profit := complex_repairs_per_month * (complex_repair_charge - complex_repair_cost)
  let total_income := total_tire_profit + total_complex_profit + retail_profit
  let final_profit := total_income - fixed_expenses
  final_profit = 3000 :=
by
  sorry

end bike_shop_profit_l185_185460


namespace solve_problem_l185_185139

theorem solve_problem (a : ℝ) (x : ℝ) (h1 : 3 * x + |a - 2| = -3) (h2 : 3 * x + 4 = 0) :
  (a = 3 ∨ a = 1) → ((a - 2) ^ 2010 - 2 * a + 1 = -4 ∨ (a - 2) ^ 2010 - 2 * a + 1 = 0) :=
by {
  sorry
}

end solve_problem_l185_185139


namespace circle_symmetric_equation_l185_185389

noncomputable def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

noncomputable def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

noncomputable def symmetric_condition (x y : ℝ) (L : ℝ × ℝ → Prop) : Prop := 
  L (y + 1, x - 1)

theorem circle_symmetric_equation :
  ∀ (x y : ℝ),
  circle1 (y + 1) (x - 1) →
  (x-2)^2 + (y+2)^2 = 1 :=
by
  intros x y h
  sorry

end circle_symmetric_equation_l185_185389


namespace edge_length_of_cube_l185_185588

theorem edge_length_of_cube {V_cube V_cuboid : ℝ} (base_area : ℝ) (height : ℝ)
  (h1 : base_area = 10) (h2 : height = 73) (h3 : V_cube = V_cuboid - 1)
  (h4 : V_cuboid = base_area * height) :
  ∃ (a : ℝ), a^3 = V_cube ∧ a = 9 :=
by
  /- The proof is omitted -/
  sorry

end edge_length_of_cube_l185_185588


namespace nontrivial_solution_exists_l185_185836

theorem nontrivial_solution_exists
  (a b c : ℝ) :
  (∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ 
    a * x + b * y + c * z = 0 ∧ 
    b * x + c * y + a * z = 0 ∧ 
    c * x + a * y + b * z = 0) ↔ (a + b + c = 0 ∨ a = b ∧ b = c) := 
sorry

end nontrivial_solution_exists_l185_185836


namespace train_length_is_approx_l185_185303

noncomputable def train_length : ℝ :=
  let speed_kmh : ℝ := 54
  let conversion_factor : ℝ := 1000 / 3600
  let speed_ms : ℝ := speed_kmh * conversion_factor
  let time_seconds : ℝ := 11.999040076793857
  speed_ms * time_seconds

theorem train_length_is_approx : abs (train_length - 179.99) < 0.001 := 
by
  sorry

end train_length_is_approx_l185_185303


namespace not_divisible_by_5_l185_185567

theorem not_divisible_by_5 (n : ℤ) : ¬ (n^2 - 8) % 5 = 0 :=
by sorry

end not_divisible_by_5_l185_185567


namespace count_valid_ways_l185_185691

theorem count_valid_ways (n : ℕ) (h1 : n = 6) : 
  ∀ (library : ℕ), (1 ≤ library) → (library ≤ 5) → ∃ (checked_out : ℕ), 
  (checked_out = n - library) := 
sorry

end count_valid_ways_l185_185691


namespace correct_multiplier_l185_185960

theorem correct_multiplier
  (x : ℕ)
  (incorrect_multiplier : ℕ := 34)
  (difference : ℕ := 1215)
  (number_to_be_multiplied : ℕ := 135) :
  number_to_be_multiplied * x - number_to_be_multiplied * incorrect_multiplier = difference →
  x = 43 :=
  sorry

end correct_multiplier_l185_185960


namespace graph_forms_l185_185987

theorem graph_forms (x y : ℝ) :
  x^3 * (2 * x + 2 * y + 3) = y^3 * (2 * x + 2 * y + 3) →
  (∀ x y : ℝ, y ≠ x → y = -x - 3 / 2) ∨ (y = x) :=
sorry

end graph_forms_l185_185987


namespace find_a_from_log_condition_l185_185371

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_a_from_log_condition (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
  (h₂ : f a 9 = 2) : a = 3 :=
by
  sorry

end find_a_from_log_condition_l185_185371


namespace possible_values_of_x_and_factors_l185_185892

theorem possible_values_of_x_and_factors (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x = p^5 ∧ (∀ (d : ℕ), d ∣ x → d = p^0 ∨ d = p^1 ∨ d = p^2 ∨ d = p^3 ∨ d = p^4 ∨ d = p^5) ∧ Nat.divisors x ≠ ∅ ∧ (Nat.divisors x).card = 6 := 
  by 
    sorry

end possible_values_of_x_and_factors_l185_185892


namespace probability_correct_l185_185265

-- Defining the values on the spinner
inductive SpinnerValue
| Bankrupt
| Thousand
| EightHundred
| FiveThousand
| Thousand'

open SpinnerValue

-- Function to get value in number from SpinnerValue
def value (v : SpinnerValue) : ℕ :=
  match v with
  | Bankrupt => 0
  | Thousand => 1000
  | EightHundred => 800
  | FiveThousand => 5000
  | Thousand' => 1000

-- Total number of spins
def total_spins : ℕ := 3

-- Total possible outcomes
def total_outcomes : ℕ := (5 : ℕ) ^ total_spins

-- Number of favorable outcomes (count of permutations summing to 5800)
def favorable_outcomes : ℕ :=
  12  -- This comes from solution steps

-- The probability as a ratio of favorable outcomes to total outcomes
def probability_of_5800_in_three_spins : ℚ :=
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_5800_in_three_spins = 12 / 125 := by
  sorry

end probability_correct_l185_185265


namespace exists_factorial_with_first_digits_2015_l185_185084

theorem exists_factorial_with_first_digits_2015 : ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 2015 * (10^k) ≤ n! ∧ n! < 2016 * (10^k)) :=
sorry

end exists_factorial_with_first_digits_2015_l185_185084


namespace eval_expression_l185_185643

theorem eval_expression : (Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6)) = 4 * Real.sqrt 6 :=
by
  sorry

end eval_expression_l185_185643


namespace determine_e_l185_185059

-- Define the polynomial Q(x)
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the problem statement
theorem determine_e (d e f : ℝ)
  (h1 : f = 9)
  (h2 : (d * (d + 9)) - 168 = 0)
  (h3 : d^2 - 6 * e = 12 + d + e)
  : e = -24 ∨ e = 20 :=
by
  sorry

end determine_e_l185_185059


namespace graph_not_passing_through_origin_l185_185854

theorem graph_not_passing_through_origin (m : ℝ) (h : 3 * m^2 - 2 * m ≠ 0) : m = -(1 / 3) :=
sorry

end graph_not_passing_through_origin_l185_185854


namespace find_original_price_l185_185779

def initial_price (P : ℝ) : Prop :=
  let first_discount := P * 0.76
  let second_discount := first_discount * 0.85
  let final_price := second_discount * 1.10
  final_price = 532

theorem find_original_price : ∃ P : ℝ, initial_price P :=
sorry

end find_original_price_l185_185779


namespace unique_function_l185_185156

theorem unique_function (f : ℝ → ℝ) 
  (H : ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y) : 
  ∀ x : ℝ, f x = 3 * x :=
by 
  sorry

end unique_function_l185_185156


namespace Dan_speed_must_exceed_45_mph_l185_185444

theorem Dan_speed_must_exceed_45_mph : 
  ∀ (distance speed_Cara time_lag time_required speed_Dan : ℝ),
    distance = 180 →
    speed_Cara = 30 →
    time_lag = 2 →
    time_required = 4 →
    (distance / speed_Cara) = 6 →
    (∀ t, t = distance / speed_Dan → t < time_required) →
    speed_Dan > 45 :=
by
  intro distance speed_Cara time_lag time_required speed_Dan
  intro h1 h2 h3 h4 h5 h6
  sorry

end Dan_speed_must_exceed_45_mph_l185_185444


namespace chocolate_syrup_per_glass_l185_185262

-- Definitions from the conditions
def each_glass_volume : ℝ := 8
def milk_per_glass : ℝ := 6.5
def total_milk : ℝ := 130
def total_chocolate_syrup : ℝ := 60
def total_chocolate_milk : ℝ := 160

-- Proposition and statement to prove
theorem chocolate_syrup_per_glass : 
  (total_chocolate_milk / each_glass_volume) * milk_per_glass = total_milk → 
  (each_glass_volume - milk_per_glass = 1.5) := 
by 
  sorry

end chocolate_syrup_per_glass_l185_185262


namespace geometric_sequence_q_and_an_l185_185874

theorem geometric_sequence_q_and_an
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : q > 0)
  (h2_eq : a 2 = 1)
  (h2_h6_eq_9h4 : a 2 * a 6 = 9 * a 4) :
  q = 3 ∧ ∀ n, a n = 3^(n - 2) := by
sorry

end geometric_sequence_q_and_an_l185_185874


namespace symmetry_graph_l185_185632

theorem symmetry_graph (θ:ℝ) (hθ: θ > 0):
  (∀ k: ℤ, 2 * (3 * Real.pi / 4) + (Real.pi / 3) - 2 * θ = k * Real.pi + Real.pi / 2) 
  → θ = Real.pi / 6 :=
by 
  sorry

end symmetry_graph_l185_185632


namespace domain_of_function_l185_185369

theorem domain_of_function :
  {x : ℝ | x > 4 ∧ x ≠ 5} = (Set.Ioo 4 5 ∪ Set.Ioi 5) :=
by
  sorry

end domain_of_function_l185_185369


namespace problem_1_problem_2_problem_3_l185_185880

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5 / 2}

theorem problem_1 : A ∩ B = {x | -1 < x ∧ x < 2} := sorry

theorem problem_2 : compl B ∪ P = {x | x ≤ 0 ∨ x ≥ 5 / 2} := sorry

theorem problem_3 : (A ∩ B) ∩ compl P = {x | 0 < x ∧ x < 2} := sorry

end problem_1_problem_2_problem_3_l185_185880


namespace sum_a3_a4_eq_14_l185_185479

open Nat

-- Define variables
def S (n : ℕ) : ℕ := n^2 + n
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem sum_a3_a4_eq_14 : a 3 + a 4 = 14 := by
  sorry

end sum_a3_a4_eq_14_l185_185479


namespace cos_neg_300_l185_185071

theorem cos_neg_300 : Real.cos (-(300 : ℝ) * Real.pi / 180) = 1 / 2 :=
by
  -- Proof goes here
  sorry

end cos_neg_300_l185_185071


namespace inequality_abs_l185_185995

noncomputable def f (x : ℝ) : ℝ := abs (x - 1/2) + abs (x + 1/2)

def M : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem inequality_abs (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + a * b| := 
by
  sorry

end inequality_abs_l185_185995


namespace mental_math_competition_l185_185778

theorem mental_math_competition :
  -- The number of teams that participated is 4
  (∃ (teams : ℕ) (numbers : List ℕ),
     -- Each team received a number that can be written as 15M + 11m where M is the largest odd divisor
     -- and m is the smallest odd divisor greater than 1.
     teams = 4 ∧ 
     numbers = [528, 880, 1232, 1936] ∧
     ∀ n ∈ numbers,
       ∃ M m, M > 1 ∧ m > 1 ∧
       M % 2 = 1 ∧ m % 2 = 1 ∧
       (∀ d, d ∣ n → (d % 2 = 1 → M ≥ d)) ∧ 
       (∀ d, d ∣ n → (d % 2 = 1 ∧ d > 1 → m ≤ d)) ∧
       n = 15 * M + 11 * m) :=
sorry

end mental_math_competition_l185_185778


namespace number_of_pieces_sold_on_third_day_l185_185241

variable (m : ℕ)

def first_day_sales : ℕ := m
def second_day_sales : ℕ := (m / 2) - 3
def third_day_sales : ℕ := second_day_sales m + 5

theorem number_of_pieces_sold_on_third_day :
  third_day_sales m = (m / 2) + 2 := by sorry

end number_of_pieces_sold_on_third_day_l185_185241


namespace problem_statement_l185_185968

/-- Definition of the function f that relates the input n with floor functions -/
def f (n : ℕ) : ℤ :=
  n + ⌊(n : ℤ) / 6⌋ - ⌊(n : ℤ) / 2⌋ - ⌊2 * (n : ℤ) / 3⌋

/-- Prove the main statement -/
theorem problem_statement (n : ℕ) (hpos : 0 < n) :
  f n = 0 ↔ ∃ k : ℕ, n = 6 * k + 1 :=
sorry -- Proof goes here.

end problem_statement_l185_185968


namespace quadratic_inequality_solution_l185_185366

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3 * x - 18 < 0 ↔ -6 < x ∧ x < 3 := 
sorry

end quadratic_inequality_solution_l185_185366


namespace pair_exists_l185_185198

def exists_pair (a b : ℕ → ℕ) : Prop :=
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q

theorem pair_exists (a b : ℕ → ℕ) : exists_pair a b :=
sorry

end pair_exists_l185_185198


namespace people_later_than_yoongi_l185_185595

variable (total_students : ℕ) (people_before_yoongi : ℕ)

theorem people_later_than_yoongi
    (h1 : total_students = 20)
    (h2 : people_before_yoongi = 11) :
    total_students - (people_before_yoongi + 1) = 8 := 
sorry

end people_later_than_yoongi_l185_185595


namespace second_smallest_packs_of_hot_dogs_l185_185493

theorem second_smallest_packs_of_hot_dogs 
  (n : ℕ) 
  (h1 : ∃ (k : ℕ), n = 2 * k + 2)
  (h2 : 12 * n ≡ 6 [MOD 8]) : 
  n = 4 :=
by
  sorry

end second_smallest_packs_of_hot_dogs_l185_185493


namespace part1_part2_l185_185593

-- Part (1): Prove k = 3 given x = -1 is a solution
theorem part1 (k : ℝ) (h : k * (-1)^2 + 4 * (-1) + 1 = 0) : k = 3 := 
sorry

-- Part (2): Prove k ≤ 4 and k ≠ 0 for the quadratic equation to have two real roots
theorem part2 (k : ℝ) (h : 16 - 4 * k ≥ 0) : k ≤ 4 ∧ k ≠ 0 :=
sorry

end part1_part2_l185_185593


namespace fraction_transformation_correct_l185_185193

theorem fraction_transformation_correct
  {a b : ℝ} (hb : b ≠ 0) : 
  (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_transformation_correct_l185_185193


namespace integer_triangle_600_integer_triangle_144_l185_185897

-- Problem Part I
theorem integer_triangle_600 :
  ∃ (a b c : ℕ), a * b * c = 600 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 26 :=
by {
  sorry
}

-- Problem Part II
theorem integer_triangle_144 :
  ∃ (a b c : ℕ), a * b * c = 144 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 16 :=
by {
  sorry
}

end integer_triangle_600_integer_triangle_144_l185_185897


namespace percentage_of_men_is_55_l185_185411

-- Define the percentage of men among all employees
def percent_of_men (M : ℝ) := M

-- Define the percentage of women among all employees
def percent_of_women (M : ℝ) := 1 - M

-- Define the contribution to picnic attendance by men
def attendance_by_men (M : ℝ) := 0.20 * M

-- Define the contribution to picnic attendance by women
def attendance_by_women (M : ℝ) := 0.40 * (percent_of_women M)

-- Define the total attendance
def total_attendance (M : ℝ) := attendance_by_men M + attendance_by_women M

theorem percentage_of_men_is_55 : ∀ M : ℝ, total_attendance M = 0.29 → M = 0.55 :=
by
  intro M
  intro h
  sorry

end percentage_of_men_is_55_l185_185411


namespace find_y_l185_185523

theorem find_y : 
  (6 + 10 + 14 + 22) / 4 = (15 + y) / 2 → y = 11 :=
by
  intros h
  sorry

end find_y_l185_185523


namespace construct_one_degree_l185_185953

theorem construct_one_degree (theta : ℝ) (h : theta = 19) : 1 = 19 * theta - 360 :=
by
  -- Proof here will be filled
  sorry

end construct_one_degree_l185_185953


namespace total_flowers_collected_l185_185750

/- Definitions for the given conditions -/
def maxFlowers : ℕ := 50
def arwenTulips : ℕ := 20
def arwenRoses : ℕ := 18
def arwenSunflowers : ℕ := 6

def elrondTulips : ℕ := 2 * arwenTulips
def elrondRoses : ℕ := if 3 * arwenRoses + elrondTulips > maxFlowers then maxFlowers - elrondTulips else 3 * arwenRoses

def galadrielTulips : ℕ := if 3 * elrondTulips > maxFlowers then maxFlowers else 3 * elrondTulips
def galadrielRoses : ℕ := if 2 * arwenRoses + galadrielTulips > maxFlowers then maxFlowers - galadrielTulips else 2 * arwenRoses

def galadrielSunflowers : ℕ := 0 -- she didn't pick any sunflowers
def legolasSunflowers : ℕ := arwenSunflowers + galadrielSunflowers
def legolasRemaining : ℕ := maxFlowers - legolasSunflowers
def legolasRosesAndTulips : ℕ := legolasRemaining / 2
def legolasTulips : ℕ := legolasRosesAndTulips
def legolasRoses : ℕ := legolasRosesAndTulips

def arwenTotal : ℕ := arwenTulips + arwenRoses + arwenSunflowers
def elrondTotal : ℕ := elrondTulips + elrondRoses
def galadrielTotal : ℕ := galadrielTulips + galadrielRoses + galadrielSunflowers
def legolasTotal : ℕ := legolasTulips + legolasRoses + legolasSunflowers

def totalFlowers : ℕ := arwenTotal + elrondTotal + galadrielTotal + legolasTotal

theorem total_flowers_collected : totalFlowers = 194 := by
  /- This will be where the proof goes, but we leave it as a placeholder. -/
  sorry

end total_flowers_collected_l185_185750


namespace combined_population_l185_185360

theorem combined_population (W PP LH : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : PP = LH + 800) : 
  (PP + LH) = 11800 :=
by
  sorry

end combined_population_l185_185360


namespace perpendicular_lines_foot_of_perpendicular_l185_185630

theorem perpendicular_lines_foot_of_perpendicular 
  (m n p : ℝ) 
  (h1 : 2 * 2 + 3 * p - 1 = 0)
  (h2 : 3 * 2 - 2 * p + n = 0)
  (h3 : - (2 / m) * (3 / 2) = -1) 
  : p - m - n = 4 := 
by
  sorry

end perpendicular_lines_foot_of_perpendicular_l185_185630


namespace perpendicular_vectors_solution_l185_185932

theorem perpendicular_vectors_solution (m : ℝ) (a : ℝ × ℝ := (m-1, 2)) (b : ℝ × ℝ := (m, -3)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : m = 3 ∨ m = -2 :=
by sorry

end perpendicular_vectors_solution_l185_185932


namespace cone_volume_l185_185469

theorem cone_volume (S : ℝ) (hPos : S > 0) : 
  let R := Real.sqrt (S / 7)
  let H := Real.sqrt (5 * S)
  let V := (π * S * (Real.sqrt (5 * S))) / 21
  (π * R * R * H / 3) = V := 
sorry

end cone_volume_l185_185469


namespace cube_identity_simplification_l185_185123

theorem cube_identity_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3 * x * y * z) / (x * y * z) = 6 :=
by
  sorry

end cube_identity_simplification_l185_185123


namespace rahul_batting_average_l185_185242

theorem rahul_batting_average:
  ∃ (A : ℝ), A = 46 ∧
  (∀ (R : ℝ), R = 138 → R = 54 * 4 - 78 → A = R / 3) ∧
  ∃ (n_matches : ℕ), n_matches = 3 :=
by
  sorry

end rahul_batting_average_l185_185242


namespace integral_cosine_l185_185937

noncomputable def a : ℝ := 2 * Real.pi / 3

theorem integral_cosine (ha : a = 2 * Real.pi / 3) :
  ∫ x in -a..a, Real.cos x = Real.sqrt 3 := 
sorry

end integral_cosine_l185_185937


namespace percentage_increase_is_50_l185_185402

def initial : ℝ := 110
def final : ℝ := 165

theorem percentage_increase_is_50 :
  ((final - initial) / initial) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l185_185402


namespace rabbit_carrots_l185_185613

theorem rabbit_carrots (r f : ℕ) (hr : 3 * r = 5 * f) (hf : f = r - 6) : 3 * r = 45 :=
by
  sorry

end rabbit_carrots_l185_185613


namespace proof_of_problem_l185_185997

def problem_statement : Prop :=
  2 * Real.cos (Real.pi / 4) + abs (Real.sqrt 2 - 3)
  - (1 / 3) ^ (-2 : ℤ) + (2021 - Real.pi) ^ 0 = -5

theorem proof_of_problem : problem_statement :=
by
  sorry

end proof_of_problem_l185_185997


namespace oranges_equiv_frac_bananas_l185_185326

theorem oranges_equiv_frac_bananas :
  (3 / 4) * 16 * (1 / 3) * 9 = (3 / 2) * 6 :=
by
  sorry

end oranges_equiv_frac_bananas_l185_185326


namespace number_of_candidates_l185_185451

theorem number_of_candidates
  (P : ℕ) (A_c A_p A_f : ℕ)
  (h_p : P = 100)
  (h_ac : A_c = 35)
  (h_ap : A_p = 39)
  (h_af : A_f = 15) :
  ∃ T : ℕ, T = 120 := 
by
  sorry

end number_of_candidates_l185_185451


namespace Krishan_has_4046_l185_185712

variable (Ram Gopal Krishan : ℕ) -- Define the variables

-- Conditions given in the problem
axiom ratio_Ram_Gopal : Ram * 17 = Gopal * 7
axiom ratio_Gopal_Krishan : Gopal * 17 = Krishan * 7
axiom Ram_value : Ram = 686

-- This is the goal to prove
theorem Krishan_has_4046 : Krishan = 4046 :=
by
  -- Here is where the proof would go
  sorry

end Krishan_has_4046_l185_185712


namespace cashier_five_dollar_bills_l185_185670

-- Define the conditions as a structure
structure CashierBills (x y : ℕ) : Prop :=
(total_bills : x + y = 126)
(total_value : 5 * x + 10 * y = 840)

-- State the theorem that we need to prove
theorem cashier_five_dollar_bills (x y : ℕ) (h : CashierBills x y) : x = 84 :=
sorry

end cashier_five_dollar_bills_l185_185670


namespace new_ratio_first_term_l185_185292

theorem new_ratio_first_term (x : ℕ) (r1 r2 : ℕ) (new_r1 : ℕ) :
  r1 = 4 → r2 = 15 → x = 29 → new_r1 = r1 + x → new_r1 = 33 :=
by
  intros h_r1 h_r2 h_x h_new_r1
  rw [h_r1, h_x] at h_new_r1
  exact h_new_r1

end new_ratio_first_term_l185_185292


namespace regression_equation_pos_corr_l185_185600

noncomputable def linear_regression (x y : ℝ) : ℝ := 0.4 * x + 2.5

theorem regression_equation_pos_corr (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (mean_x : ℝ := 2.5) (mean_y : ℝ := 3.5)
    (pos_corr : x * y > 0)
    (cond1 : mean_x = 2.5)
    (cond2 : mean_y = 3.5) :
    linear_regression mean_x mean_y = mean_y :=
by
  sorry

end regression_equation_pos_corr_l185_185600


namespace remainder_is_three_l185_185421

def P (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem remainder_is_three : P 1 = 3 :=
by
  -- Proof goes here.
  sorry

end remainder_is_three_l185_185421


namespace horner_v3_value_l185_185001

-- Define constants
def a_n : ℤ := 2 -- Leading coefficient of x^5
def a_3 : ℤ := -3 -- Coefficient of x^3
def a_2 : ℤ := 5 -- Coefficient of x^2
def a_0 : ℤ := -4 -- Constant term
def x : ℤ := 2 -- Given value of x

-- Horner's method sequence for the coefficients
def v_0 : ℤ := a_n -- Initial value v_0
def v_1 : ℤ := v_0 * x -- Calculated as v_0 * x
def v_2 : ℤ := v_1 * x + a_3 -- Calculated as v_1 * x + a_3 (coefficient of x^3)
def v_3 : ℤ := v_2 * x + a_2 -- Calculated as v_2 * x + a_2 (coefficient of x^2)

theorem horner_v3_value : v_3 = 15 := 
by
  -- Formal proof would go here, skipped due to problem specifications
  sorry

end horner_v3_value_l185_185001


namespace domain_h_l185_185930

def domain_f : Set ℝ := Set.Icc (-12) 6
def h (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-3*x)

theorem domain_h {f : ℝ → ℝ} (hf : ∀ x, x ∈ domain_f → f x ∈ Set.univ) {x : ℝ} :
  h f x ∈ Set.univ ↔ x ∈ Set.Icc (-2) 4 :=
by
  sorry

end domain_h_l185_185930


namespace last_digit_product_3_2001_7_2002_13_2003_l185_185543

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_product_3_2001_7_2002_13_2003 :
  last_digit (3^2001 * 7^2002 * 13^2003) = 9 :=
by
  sorry

end last_digit_product_3_2001_7_2002_13_2003_l185_185543


namespace percent_fair_hair_l185_185605

theorem percent_fair_hair 
  (total_employees : ℕ) 
  (percent_women_fair_hair : ℝ) 
  (percent_fair_hair_women : ℝ)
  (total_women_fair_hair : ℕ)
  (total_fair_hair : ℕ)
  (h1 : percent_women_fair_hair = 30 / 100)
  (h2 : percent_fair_hair_women = 40 / 100)
  (h3 : total_women_fair_hair = percent_women_fair_hair * total_employees)
  (h4 : percent_fair_hair_women * total_fair_hair = total_women_fair_hair)
  : total_fair_hair = 75 / 100 * total_employees := 
by
  sorry

end percent_fair_hair_l185_185605


namespace players_without_cautions_l185_185684

theorem players_without_cautions (Y N : ℕ) (h1 : Y + N = 11) (h2 : Y = 6) : N = 5 :=
by
  sorry

end players_without_cautions_l185_185684


namespace solve_for_x_l185_185675

theorem solve_for_x (x : ℝ) : 5 + 3.4 * x = 2.1 * x - 30 → x = -26.923 := 
by 
  sorry

end solve_for_x_l185_185675


namespace speed_ratio_l185_185022

theorem speed_ratio :
  ∀ (v_A v_B : ℝ), (v_A / v_B = 3 / 2) ↔ (v_A = 3 * v_B / 2) :=
by
  intros
  sorry

end speed_ratio_l185_185022


namespace units_digit_n_l185_185104

theorem units_digit_n (m n : ℕ) (h1 : m * n = 14^5) (h2 : m % 10 = 8) : n % 10 = 3 :=
sorry

end units_digit_n_l185_185104


namespace min_bottles_required_l185_185872

theorem min_bottles_required (bottle_ounces : ℕ) (total_ounces : ℕ) (h : bottle_ounces = 15) (ht : total_ounces = 150) :
  ∃ (n : ℕ), n * bottle_ounces >= total_ounces ∧ n = 10 :=
by
  sorry

end min_bottles_required_l185_185872


namespace steps_probability_to_point_3_3_l185_185552

theorem steps_probability_to_point_3_3 : 
  let a := 35
  let b := 4096
  a + b = 4131 :=
by {
  sorry
}

end steps_probability_to_point_3_3_l185_185552


namespace quadratic_negative_root_l185_185452

theorem quadratic_negative_root (m : ℝ) : (∃ x : ℝ, (m * x^2 + 2 * x + 1 = 0 ∧ x < 0)) ↔ (m ≤ 1) :=
by
  sorry

end quadratic_negative_root_l185_185452


namespace tan_triple_angle_l185_185870

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 1/3) : Real.tan (3 * θ) = 13/9 :=
by
  sorry

end tan_triple_angle_l185_185870


namespace real_set_x_eq_l185_185521

theorem real_set_x_eq :
  {x : ℝ | ⌊x * ⌊x⌋⌋ = 45} = {x : ℝ | 7.5 ≤ x ∧ x < 7.6667} :=
by
  -- The proof would be provided here, but we're skipping it with sorry
  sorry

end real_set_x_eq_l185_185521


namespace ryan_fraction_l185_185304

-- Define the total amount of money
def total_money : ℕ := 48

-- Define that Ryan owns a fraction R of the total money
variable {R : ℚ}

-- Define the debts
def ryan_owes_leo : ℕ := 10
def leo_owes_ryan : ℕ := 7

-- Define the final amount Leo has after settling the debts
def leo_final_amount : ℕ := 19

-- Define the condition that Leo and Ryan together have $48
def leo_plus_ryan (leo_amount ryan_amount : ℚ) : Prop := 
  leo_amount + ryan_amount = total_money

-- Define Ryan's amount as a fraction R of the total money
def ryan_amount (R : ℚ) : ℚ := R * total_money

-- Define Leo's amount before debts were settled
def leo_amount_before_debts : ℚ := (leo_final_amount : ℚ) + leo_owes_ryan

-- Define the equation after settling debts
def leo_final_eq (leo_amount_before_debts : ℚ) : Prop :=
  (leo_amount_before_debts - ryan_owes_leo = leo_final_amount)

-- The Lean theorem that needs to be proved
theorem ryan_fraction :
  ∃ (R : ℚ), leo_plus_ryan (leo_amount_before_debts - ryan_owes_leo) (ryan_amount R)
  ∧ leo_final_eq leo_amount_before_debts
  ∧ R = 11 / 24 :=
sorry

end ryan_fraction_l185_185304


namespace unique_nat_pair_l185_185245

theorem unique_nat_pair (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n + 1 / m : ℚ) ∧ ∀ (n' m' : ℕ), 
  n' ≠ m' ∧ (2 / p : ℚ) = (1 / n' + 1 / m' : ℚ) → (n', m') = (n, m) ∨ (n', m') = (m, n) :=
by
  sorry

end unique_nat_pair_l185_185245


namespace exp_base_lt_imp_cube_l185_185410

theorem exp_base_lt_imp_cube (a x y : ℝ) (h_a : 0 < a) (h_a1 : a < 1) (h_exp : a^x > a^y) : x^3 < y^3 :=
by
  sorry

end exp_base_lt_imp_cube_l185_185410


namespace infinitely_many_singular_pairs_l185_185252

def largestPrimeFactor (n : ℕ) : ℕ := sorry -- definition of largest prime factor

def isSingularPair (p q : ℕ) : Prop :=
  p ≠ q ∧ ∀ (n : ℕ), n ≥ 2 → largestPrimeFactor n * largestPrimeFactor (n + 1) ≠ p * q

theorem infinitely_many_singular_pairs : ∃ (S : ℕ → (ℕ × ℕ)), ∀ i, isSingularPair (S i).1 (S i).2 :=
sorry

end infinitely_many_singular_pairs_l185_185252


namespace flowchart_output_proof_l185_185901

def flowchart_output (x : ℕ) : ℕ :=
  let x := x + 2
  let x := x + 2
  let x := x + 2
  x

theorem flowchart_output_proof :
  flowchart_output 10 = 16 := by
  -- Assume initial value of x is 10
  let x0 := 10
  -- First iteration
  let x1 := x0 + 2
  -- Second iteration
  let x2 := x1 + 2
  -- Third iteration
  let x3 := x2 + 2
  -- Final value of x
  have hx_final : x3 = 16 := by rfl
  -- The result should be 16
  have h_result : flowchart_output 10 = x3 := by rfl
  rw [hx_final] at h_result
  exact h_result

end flowchart_output_proof_l185_185901


namespace no_solution_ineq_system_l185_185319

def inequality_system (x : ℝ) : Prop :=
  (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
  (x + 9 / 2 > x / 8) ∧
  (11 / 3 - x / 6 < (34 - 3 * x) / 5)

theorem no_solution_ineq_system : ¬ ∃ x : ℝ, inequality_system x :=
  sorry

end no_solution_ineq_system_l185_185319


namespace correct_addition_result_l185_185515

theorem correct_addition_result (x : ℚ) (h : x - 13/5 = 9/7) : x + 13/5 = 227/35 := 
by sorry

end correct_addition_result_l185_185515


namespace hare_overtakes_tortoise_l185_185922

noncomputable def hare_distance (t: ℕ) : ℕ := 
  if t ≤ 5 then 10 * t
  else if t ≤ 20 then 50
  else 50 + 20 * (t - 20)

noncomputable def tortoise_distance (t: ℕ) : ℕ :=
  2 * t

theorem hare_overtakes_tortoise : 
  ∃ t : ℕ, t ≤ 60 ∧ hare_distance t = tortoise_distance t ∧ 60 - t = 22 :=
sorry

end hare_overtakes_tortoise_l185_185922


namespace M_is_subset_of_N_l185_185468

theorem M_is_subset_of_N : 
  ∀ (x y : ℝ), (|x| + |y| < 1) → 
    (Real.sqrt ((x - 1/2)^2 + (y + 1/2)^2) + Real.sqrt ((x + 1/2)^2 + (y - 1/2)^2) < 2 * Real.sqrt 2) :=
by
  intro x y h
  sorry

end M_is_subset_of_N_l185_185468


namespace ratio_of_numbers_l185_185554

theorem ratio_of_numbers (A B : ℕ) (h_lcm : Nat.lcm A B = 48) (h_hcf : Nat.gcd A B = 4) : A / 4 = 3 ∧ B / 4 = 4 :=
sorry

end ratio_of_numbers_l185_185554


namespace jury_concludes_you_are_not_guilty_l185_185517

def criminal_is_a_liar : Prop := sorry -- The criminal is a liar, known.
def you_are_a_liar : Prop := sorry -- You are a liar, unknown.
def you_are_not_guilty : Prop := sorry -- You are not guilty.

theorem jury_concludes_you_are_not_guilty :
  criminal_is_a_liar → you_are_a_liar → you_are_not_guilty → "I am guilty" = "You are not guilty" :=
by
  -- Proof construct omitted as per problem requirements
  sorry

end jury_concludes_you_are_not_guilty_l185_185517


namespace profit_relationship_max_profit_l185_185904

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
else if h : 2 < x ∧ x ≤ 5 then 50 * x / (1 + x)
else 0

noncomputable def f (x : ℝ) : ℝ :=
15 * W x - 10 * x - 20 * x

theorem profit_relationship:
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = 75 * x^2 - 30 * x + 225) ∧
  (∀ x, 2 < x ∧ x ≤ 5 → f x = (750 * x)/(1 + x) - 30 * x) :=
by
  -- to be proven
  sorry

theorem max_profit:
  ∃ x, 0 ≤ x ∧ x ≤ 5 ∧ f x = 480 ∧ 10 * x = 40 :=
by
  -- to be proven
  sorry

end profit_relationship_max_profit_l185_185904


namespace range_of_a1_l185_185969

noncomputable def sequence_a (n : ℕ) : ℤ := sorry
noncomputable def sum_S (n : ℕ) : ℤ := sorry

theorem range_of_a1 :
  (∀ n : ℕ, n > 0 → sum_S n + sum_S (n+1) = 2 * n^2 + n) ∧
  (∀ n : ℕ, n > 0 → sequence_a n < sequence_a (n+1)) →
  -1/4 < sequence_a 1 ∧ sequence_a 1 < 3/4 := sorry

end range_of_a1_l185_185969


namespace smallest_possible_value_abs_sum_l185_185144

theorem smallest_possible_value_abs_sum :
  ∃ x : ℝ, (∀ y : ℝ, abs (y + 3) + abs (y + 5) + abs (y + 7) ≥ abs (x + 3) + abs (x + 5) + abs (x + 7))
  ∧ (abs (x + 3) + abs (x + 5) + abs (x + 7) = 4) := by
  sorry

end smallest_possible_value_abs_sum_l185_185144


namespace converse_not_true_prop_B_l185_185608

noncomputable def line_in_plane (b : Type) (α : Type) : Prop := sorry
noncomputable def perp_line_plane (b : Type) (β : Type) : Prop := sorry
noncomputable def perp_planes (α : Type) (β : Type) : Prop := sorry
noncomputable def parallel_planes (α : Type) (β : Type) : Prop := sorry

variables (a b c : Type) (α β : Type)

theorem converse_not_true_prop_B :
  (line_in_plane b α) → (perp_planes α β) → ¬ (perp_line_plane b β) :=
sorry

end converse_not_true_prop_B_l185_185608


namespace convert_angle_degrees_to_radians_l185_185395

theorem convert_angle_degrees_to_radians :
  ∃ (k : ℤ) (α : ℝ), -1125 * (Real.pi / 180) = 2 * k * Real.pi + α ∧ 0 ≤ α ∧ α < 2 * Real.pi ∧ (-8 * Real.pi + 7 * Real.pi / 4) = 2 * k * Real.pi + α :=
by {
  sorry
}

end convert_angle_degrees_to_radians_l185_185395


namespace sqrt_sum_abs_eq_l185_185681

theorem sqrt_sum_abs_eq (x : ℝ) :
    (Real.sqrt (x^2 + 6 * x + 9) + Real.sqrt (x^2 - 6 * x + 9)) = (|x - 3| + |x + 3|) := 
by 
  sorry

end sqrt_sum_abs_eq_l185_185681


namespace min_value_point_on_line_l185_185412

theorem min_value_point_on_line (m n : ℝ) (h : m + 2 * n = 1) : 
  2^m + 4^n ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_point_on_line_l185_185412


namespace no_integer_solution_for_Px_eq_x_l185_185592

theorem no_integer_solution_for_Px_eq_x (P : ℤ → ℤ) (hP_int_coeff : ∀ n : ℤ, ∃ k : ℤ, P n = k * n + k) 
  (hP3 : P 3 = 4) (hP4 : P 4 = 3) :
  ¬ ∃ x : ℤ, P x = x := 
by 
  sorry

end no_integer_solution_for_Px_eq_x_l185_185592


namespace device_works_probability_l185_185288

theorem device_works_probability (p_comp_damaged : ℝ) (two_components : Bool) :
  p_comp_damaged = 0.1 → two_components = true → (0.9 * 0.9 = 0.81) :=
by
  intros h1 h2
  sorry

end device_works_probability_l185_185288


namespace cost_of_each_item_l185_185206

theorem cost_of_each_item 
  (x y z : ℝ) 
  (h1 : 3 * x + 5 * y + z = 32)
  (h2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 :=
by 
  sorry

end cost_of_each_item_l185_185206


namespace find_number_l185_185805

theorem find_number (x : ℝ) (h : 50 + 5 * 12 / (x / 3) = 51) : x = 180 := 
by 
  sorry

end find_number_l185_185805


namespace sum_of_roots_combined_eq_five_l185_185801

noncomputable def sum_of_roots_poly1 : ℝ :=
-(-9/3)

noncomputable def sum_of_roots_poly2 : ℝ :=
-(-8/4)

theorem sum_of_roots_combined_eq_five :
  sum_of_roots_poly1 + sum_of_roots_poly2 = 5 :=
by
  sorry

end sum_of_roots_combined_eq_five_l185_185801


namespace fraction_equality_l185_185676

theorem fraction_equality (p q x y : ℚ) (hpq : p / q = 4 / 5) (hx : x / y + (2 * q - p) / (2 * q + p) = 1) :
  x / y = 4 / 7 :=
by {
  sorry
}

end fraction_equality_l185_185676


namespace exist_one_common_ball_l185_185355

theorem exist_one_common_ball (n : ℕ) (h_n : 5 ≤ n) (A : Fin (n+1) → Finset (Fin n))
  (hA_card : ∀ i, (A i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
sorry

end exist_one_common_ball_l185_185355


namespace chain_of_tangent_circles_exists_iff_integer_angle_multiple_l185_185164

noncomputable def angle_between_tangent_circles (R₁ R₂ : Circle) (line : Line) : ℝ :=
-- the definition should specify how we get the angle between the tangent circles
sorry

def n_tangent_circles_exist (R₁ R₂ : Circle) (n : ℕ) : Prop :=
-- the definition should specify the existence of a chain of n tangent circles
sorry

theorem chain_of_tangent_circles_exists_iff_integer_angle_multiple 
  (R₁ R₂ : Circle) (n : ℕ) (line : Line) : 
  n_tangent_circles_exist R₁ R₂ n ↔ ∃ k : ℤ, angle_between_tangent_circles R₁ R₂ line = k * (360 / n) :=
sorry

end chain_of_tangent_circles_exists_iff_integer_angle_multiple_l185_185164


namespace sectors_containing_all_numbers_l185_185155

theorem sectors_containing_all_numbers (n : ℕ) (h : 0 < n) :
  ∃ (s : Finset (Fin (2 * n))), (s.card = n) ∧ (∀ i : Fin n, ∃ j : Fin (2 * n), j ∈ s ∧ (j.val % n) + 1 = i.val) :=
  sorry

end sectors_containing_all_numbers_l185_185155


namespace find_angle_A_find_side_a_l185_185531

variable {A B C a b c : Real}
variable {area : Real}
variable (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
variable (h2 : b = 2)
variable (h3 : area = Real.sqrt 3)
variable (h4 : area = 1 / 2 * b * c * Real.sin A)

theorem find_angle_A (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A) : A = Real.pi / 3 :=
  sorry

theorem find_side_a (h4 : area = 1 / 2 * b * c * Real.sin A) (h2 : b = 2) (h3 : area = Real.sqrt 3) : a = 2 :=
  sorry

end find_angle_A_find_side_a_l185_185531


namespace range_of_a_l185_185274

def p (a : ℝ) := 0 < a ∧ a < 1
def q (a : ℝ) := a > 5 / 2 ∨ 0 < a ∧ a < 1 / 2

theorem range_of_a (a : ℝ) :
  (a > 0) ∧ (a ≠ 1) ∧ (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ (1 / 2 ≤ a ∧ a < 1) ∨ (a > 5 / 2) :=
sorry

end range_of_a_l185_185274


namespace find_k_l185_185602

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (u v : V)

theorem find_k (h : ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ u + t • (v - u) = k • u + (5 / 8) • v) :
  k = 3 / 8 := sorry

end find_k_l185_185602


namespace find_x_eq_nine_fourths_l185_185578

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l185_185578


namespace infinite_squares_and_circles_difference_l185_185598

theorem infinite_squares_and_circles_difference 
  (side_length : ℝ)
  (h₁ : side_length = 1)
  (square_area_sum : ℝ)
  (circle_area_sum : ℝ)
  (h_square_area : square_area_sum = (∑' n : ℕ, (side_length / 2^n)^2))
  (h_circle_area : circle_area_sum = (∑' n : ℕ, π * (side_length / 2^(n+1))^2 ))
  : square_area_sum - circle_area_sum = 2 - (π / 2) :=
by 
  sorry 

end infinite_squares_and_circles_difference_l185_185598


namespace line_does_not_intersect_circle_l185_185633

theorem line_does_not_intersect_circle (a : ℝ) : 
  (a > 1 ∨ a < -1) → ¬ ∃ (x y : ℝ), (x + y = a) ∧ (x^2 + y^2 = 1) :=
by
  sorry

end line_does_not_intersect_circle_l185_185633


namespace sum_of_solutions_l185_185800

theorem sum_of_solutions (s : Finset ℝ) :
  (∀ x ∈ s, |x^2 - 16 * x + 60| = 4) →
  s.sum id = 24 := 
by
  sorry

end sum_of_solutions_l185_185800


namespace exponential_inequality_l185_185056

theorem exponential_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) : 
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ :=
sorry

end exponential_inequality_l185_185056


namespace log_base_2_of_7_l185_185893

variable (m n : ℝ)

theorem log_base_2_of_7 (h1 : Real.log 5 = m) (h2 : Real.log 7 = n) : Real.logb 2 7 = n / (1 - m) :=
by
  sorry

end log_base_2_of_7_l185_185893


namespace expression_eqn_l185_185744

theorem expression_eqn (a : ℝ) (E : ℝ → ℝ)
  (h₁ : -6 * a^2 = 3 * (E a + 2))
  (h₂ : a = 1) : E a = -2 * a^2 - 2 :=
by
  sorry

end expression_eqn_l185_185744


namespace point_on_line_y_coordinate_l185_185928

variables (m b x : ℝ)

def line_equation := m * x + b

theorem point_on_line_y_coordinate : m = 4 → b = 4 → x = 199 → line_equation m b x = 800 :=
by 
  intros h_m h_b h_x
  unfold line_equation
  rw [h_m, h_b, h_x]
  norm_num
  done

end point_on_line_y_coordinate_l185_185928


namespace specific_five_card_order_probability_l185_185403

open Classical

noncomputable def prob_five_cards_specified_order : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49) * (9 / 48)

theorem specific_five_card_order_probability :
  prob_five_cards_specified_order = 2304 / 31187500 :=
by
  sorry

end specific_five_card_order_probability_l185_185403


namespace point_comparison_on_inverse_proportion_l185_185906

theorem point_comparison_on_inverse_proportion :
  (∃ y1 y2, (y1 = 2 / 1) ∧ (y2 = 2 / 2) ∧ y1 > y2) :=
by
  use 2
  use 1
  sorry

end point_comparison_on_inverse_proportion_l185_185906


namespace converse_proposition_l185_185383

theorem converse_proposition (x : ℝ) (h : x = 1 → x^2 = 1) : x^2 = 1 → x = 1 :=
by
  sorry

end converse_proposition_l185_185383


namespace jean_candy_count_l185_185035

theorem jean_candy_count : ∃ C : ℕ, 
  C - 7 = 16 ∧ 
  (C - 7 + 7 = C) ∧ 
  (C - 7 = 16) ∧ 
  (C + 0 = C) ∧
  (C - 7 = 16) :=
by 
  sorry 

end jean_candy_count_l185_185035


namespace pints_in_two_liters_nearest_tenth_l185_185673

def liters_to_pints (liters : ℝ) : ℝ :=
  2.1 * liters

theorem pints_in_two_liters_nearest_tenth :
  liters_to_pints 2 = 4.2 :=
by
  sorry

end pints_in_two_liters_nearest_tenth_l185_185673


namespace tan_of_tan_squared_2025_deg_l185_185558

noncomputable def tan_squared (x : ℝ) : ℝ := (Real.tan x) ^ 2

theorem tan_of_tan_squared_2025_deg : 
  Real.tan (tan_squared (2025 * Real.pi / 180)) = Real.tan (Real.pi / 180) :=
by
  sorry

end tan_of_tan_squared_2025_deg_l185_185558


namespace total_waiting_time_difference_l185_185845

theorem total_waiting_time_difference :
  let n_swings := 6
  let n_slide := 4 * n_swings
  let t_swings := 3.5 * 60
  let t_slide := 45
  let T_swings := n_swings * t_swings
  let T_slide := n_slide * t_slide
  let T_difference := T_swings - T_slide
  T_difference = 180 :=
by
  sorry

end total_waiting_time_difference_l185_185845


namespace ingrid_tax_rate_proof_l185_185813

namespace TaxProblem

-- Define the given conditions
def john_income : ℝ := 56000
def ingrid_income : ℝ := 72000
def combined_income := john_income + ingrid_income

def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35625

-- Calculate John's tax
def john_tax := john_tax_rate * john_income

-- Calculate total tax paid
def total_tax_paid := combined_tax_rate * combined_income

-- Calculate Ingrid's tax
def ingrid_tax := total_tax_paid - john_tax

-- Prove Ingrid's tax rate
theorem ingrid_tax_rate_proof (r : ℝ) :
  (ingrid_tax / ingrid_income) * 100 = 40 :=
  by sorry

end TaxProblem

end ingrid_tax_rate_proof_l185_185813


namespace trigonometric_identity_l185_185498

variable (α : Real)

theorem trigonometric_identity :
  (Real.tan (α - Real.pi / 4) = 1 / 2) →
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) :=
by
  intro h
  sorry

end trigonometric_identity_l185_185498


namespace ping_pong_ball_probability_l185_185463
open Nat 

def total_balls : ℕ := 70

def multiples_of_4_count : ℕ := 17
def multiples_of_9_count : ℕ := 7
def multiples_of_4_and_9_count : ℕ := 1

def inclusion_exclusion_principle : ℕ :=
  multiples_of_4_count + multiples_of_9_count - multiples_of_4_and_9_count

def desired_outcomes_count : ℕ := inclusion_exclusion_principle

def probability : ℚ := desired_outcomes_count / total_balls

theorem ping_pong_ball_probability : probability = 23 / 70 :=
  sorry

end ping_pong_ball_probability_l185_185463


namespace percentage_change_area_right_triangle_l185_185316

theorem percentage_change_area_right_triangle
  (b h : ℝ)
  (hb : b = 0.5 * h)
  (A_original A_new : ℝ)
  (H_original : A_original = (1 / 2) * b * h)
  (H_new : A_new = (1 / 2) * (1.10 * b) * (1.10 * h)) :
  ((A_new - A_original) / A_original) * 100 = 21 := by
  sorry

end percentage_change_area_right_triangle_l185_185316


namespace find_seventh_number_l185_185540

-- Let's denote the 10 numbers as A1, A2, A3, A4, A5, A6, A7, A8, A9, A10.
variables {A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ}

-- The average of all 10 numbers is 60.
def avg_10 (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10) / 10 = 60

-- The average of the first 6 numbers is 68.
def avg_first_6 (A1 A2 A3 A4 A5 A6 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6) / 6 = 68

-- The average of the last 6 numbers is 75.
def avg_last_6 (A5 A6 A7 A8 A9 A10 : ℝ) := (A5 + A6 + A7 + A8 + A9 + A10) / 6 = 75

-- Proving that the 7th number (A7) is 192.
theorem find_seventh_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) 
  (h1 : avg_10 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10) 
  (h2 : avg_first_6 A1 A2 A3 A4 A5 A6) 
  (h3 : avg_last_6 A5 A6 A7 A8 A9 A10) :
  A7 = 192 :=
by
  sorry

end find_seventh_number_l185_185540


namespace range_of_m_l185_185250

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ (-4 ≤ m ∧ m ≤ 0) := 
by sorry

end range_of_m_l185_185250


namespace meetings_percentage_l185_185259

def workday_hours := 10
def first_meeting_minutes := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_workday_minutes := workday_hours * 60
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

theorem meetings_percentage :
    (total_meeting_minutes / total_workday_minutes) * 100 = 40 :=
by
  sorry

end meetings_percentage_l185_185259


namespace monotonically_increasing_interval_l185_185227

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

theorem monotonically_increasing_interval : 
  ∀ x ∈ Set.Icc (-Real.pi) 0, 
  x ∈ Set.Icc (-Real.pi/6) 0 ↔ deriv f x = 0 := sorry

end monotonically_increasing_interval_l185_185227


namespace water_overflow_amount_l185_185561

-- Declare the conditions given in the problem
def tap_production_per_hour : ℕ := 200
def tap_run_duration_in_hours : ℕ := 24
def tank_capacity_in_ml : ℕ := 4000

-- Define the total water produced by the tap
def total_water_produced : ℕ := tap_production_per_hour * tap_run_duration_in_hours

-- Define the amount of water that overflows
def water_overflowed : ℕ := total_water_produced - tank_capacity_in_ml

-- State the theorem to prove the amount of overflowing water
theorem water_overflow_amount : water_overflowed = 800 :=
by
  -- Placeholder for the proof
  sorry

end water_overflow_amount_l185_185561


namespace tadpole_catch_l185_185082

variable (T : ℝ) (H1 : T * 0.25 = 45)

theorem tadpole_catch (T : ℝ) (H1 : T * 0.25 = 45) : T = 180 :=
sorry

end tadpole_catch_l185_185082


namespace flower_bouquet_violets_percentage_l185_185129

theorem flower_bouquet_violets_percentage
  (total_flowers yellow_flowers purple_flowers : ℕ)
  (yellow_daisies yellow_tulips purple_violets : ℕ)
  (h_yellow_flowers : yellow_flowers = (total_flowers / 2))
  (h_purple_flowers : purple_flowers = (total_flowers / 2))
  (h_yellow_daisies : yellow_daisies = (yellow_flowers / 5))
  (h_yellow_tulips : yellow_tulips = yellow_flowers - yellow_daisies)
  (h_purple_violets : purple_violets = (purple_flowers / 2)) :
  ((purple_violets : ℚ) / total_flowers) * 100 = 25 :=
by
  sorry

end flower_bouquet_violets_percentage_l185_185129


namespace vertical_asymptote_l185_185214

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 - 6*x + 8)

theorem vertical_asymptote (c : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → ((x^2 - x + c) ≠ 0)) ∨
  (∀ x : ℝ, ((x^2 - x + c) = 0) ↔ (x = 2) ∨ (x = 4)) →
  c = -2 ∨ c = -12 :=
sorry

end vertical_asymptote_l185_185214


namespace find_a_l185_185652

variables (x y : ℝ) (a : ℝ)

-- Condition 1: Original profit equation
def original_profit := y - x = x * (a / 100)

-- Condition 2: New profit equation with 5% cost decrease
def new_profit := y - 0.95 * x = 0.95 * x * ((a + 15) / 100)

theorem find_a (h1 : original_profit x y a) (h2 : new_profit x y a) : a = 185 :=
sorry

end find_a_l185_185652


namespace panthers_score_l185_185668

theorem panthers_score (P : ℕ) (wildcats_score : ℕ := 36) (score_difference : ℕ := 19) (h : wildcats_score = P + score_difference) : P = 17 := by
  sorry

end panthers_score_l185_185668


namespace well_diameter_l185_185853

theorem well_diameter (V h : ℝ) (pi : ℝ) (r : ℝ) :
  h = 8 ∧ V = 25.132741228718345 ∧ pi = 3.141592653589793 ∧ V = pi * r^2 * h → 2 * r = 2 :=
by
  sorry

end well_diameter_l185_185853


namespace probability_three_dice_less_than_seven_l185_185176

open Nat

def probability_of_exactly_three_less_than_seven (dice_count : ℕ) (sides : ℕ) (target_faces : ℕ) : ℚ :=
  let p : ℚ := target_faces / sides
  let q : ℚ := 1 - p
  (Nat.choose dice_count (dice_count / 2)) * (p^(dice_count / 2)) * (q^(dice_count / 2))

theorem probability_three_dice_less_than_seven :
  probability_of_exactly_three_less_than_seven 6 12 6 = 5 / 16 := by
  sorry

end probability_three_dice_less_than_seven_l185_185176


namespace total_amount_spent_l185_185485

def cost_per_dozen_apples : ℕ := 40
def cost_per_dozen_pears : ℕ := 50
def dozens_apples : ℕ := 14
def dozens_pears : ℕ := 14

theorem total_amount_spent : (dozens_apples * cost_per_dozen_apples + dozens_pears * cost_per_dozen_pears) = 1260 := 
  by
  sorry

end total_amount_spent_l185_185485


namespace raise_salary_to_original_l185_185454

/--
The salary of a person was reduced by 25%. By what percent should his reduced salary be raised
so as to bring it at par with his original salary?
-/
theorem raise_salary_to_original (S : ℝ) (h : S > 0) :
  ∃ P : ℝ, 0.75 * S * (1 + P / 100) = S ∧ P = 33.333333333333336 :=
sorry

end raise_salary_to_original_l185_185454


namespace squirrels_acorns_l185_185152

theorem squirrels_acorns (x : ℕ) : 
    (5 * (x - 15) = 575) → 
    x = 130 := 
by 
  intros h
  sorry

end squirrels_acorns_l185_185152


namespace cube_problem_l185_185191

theorem cube_problem (n : ℕ) (H1 : 6 * n^2 = 1 / 3 * 6 * n^3) : n = 3 :=
sorry

end cube_problem_l185_185191


namespace distinguishable_arrangements_l185_185305

-- Define number of each type of tiles
def brown_tiles := 2
def purple_tiles := 1
def green_tiles := 3
def yellow_tiles := 2
def total_tiles := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem distinguishable_arrangements :
  (Nat.factorial total_tiles) / 
  ((Nat.factorial green_tiles) * 
   (Nat.factorial brown_tiles) * 
   (Nat.factorial yellow_tiles) * 
   (Nat.factorial purple_tiles)) = 1680 := by
  sorry

end distinguishable_arrangements_l185_185305


namespace hyperbola_asymptotes_equation_l185_185756

noncomputable def hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (e : ℝ)
  (h_eq : e = 5 / 3)
  (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1) :
  String :=
by
  sorry

theorem hyperbola_asymptotes_equation : 
  ∀ a b : ℝ, ∀ ha : a > 0, ∀ hb : b > 0, ∀ e : ℝ,
  e = 5 / 3 →
  (∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1) →
  ( ∀ (x : ℝ), x ≠ 0 → y = (4/3)*x ∨ y = -(4/3)*x
  )
  :=
by
  intros _
  sorry

end hyperbola_asymptotes_equation_l185_185756


namespace investment_compound_half_yearly_l185_185037

theorem investment_compound_half_yearly
  (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (h1 : P = 6000) 
  (h2 : r = 0.10) 
  (h3 : n = 2) 
  (h4 : A = 6615) :
  t = 1 :=
by
  sorry

end investment_compound_half_yearly_l185_185037


namespace sum_of_pos_real_solutions_l185_185278

open Real

noncomputable def cos_equation_sum_pos_real_solutions : ℝ := 1082 * π

theorem sum_of_pos_real_solutions :
  ∃ x : ℝ, (0 < x) ∧ 
    (∀ x, 2 * cos (2 * x) * (cos (2 * x) - cos ((2016 * π ^ 2) / x)) = cos (6 * x) - 1) → 
      x = cos_equation_sum_pos_real_solutions :=
sorry

end sum_of_pos_real_solutions_l185_185278


namespace part_I_part_II_l185_185771

noncomputable def f (x a : ℝ) : ℝ := |2 * x + 1| - |x - a|

-- Problem (I)
theorem part_I (x : ℝ) : 
  (f x 4) > 2 ↔ (x < -7 ∨ x > 5 / 3) :=
sorry

-- Problem (II)
theorem part_II (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x a ≥ |x - 4|) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

end part_I_part_II_l185_185771


namespace find_number_l185_185791

theorem find_number (some_number : ℤ) (h : some_number + 9 = 54) : some_number = 45 :=
sorry

end find_number_l185_185791


namespace remainder_div_72_l185_185489

theorem remainder_div_72 (x : ℤ) (h : x % 8 = 3) : x % 72 = 3 :=
sorry

end remainder_div_72_l185_185489


namespace radius_of_tangent_intersection_l185_185188

variable (x y : ℝ)

def circle_eq : Prop := x^2 + y^2 = 25

def tangent_condition : Prop := y = 5 ∧ x = 0

theorem radius_of_tangent_intersection (h1 : circle_eq x y) (h2 : tangent_condition x y) : ∃r : ℝ, r = 5 :=
by sorry

end radius_of_tangent_intersection_l185_185188


namespace fraction_eval_l185_185797

theorem fraction_eval :
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25) :=
sorry

end fraction_eval_l185_185797


namespace lines_parallel_l185_185597

-- Define line l1 and line l2
def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

-- Prove that l1 is parallel to l2
theorem lines_parallel : ∀ x : ℝ, (l1 x - l2 x) = -4 := by
  sorry

end lines_parallel_l185_185597


namespace Alan_eggs_count_l185_185687

theorem Alan_eggs_count (Price_per_egg Chickens_bought Price_per_chicken Total_spent : ℕ)
  (h1 : Price_per_egg = 2) (h2 : Chickens_bought = 6) (h3 : Price_per_chicken = 8) (h4 : Total_spent = 88) :
  ∃ E : ℕ, 2 * E + Chickens_bought * Price_per_chicken = Total_spent ∧ E = 20 :=
by
  sorry

end Alan_eggs_count_l185_185687


namespace total_customers_served_l185_185069

-- Definitions for the hours worked by Ann, Becky, and Julia
def hours_ann : ℕ := 8
def hours_becky : ℕ := 8
def hours_julia : ℕ := 6

-- Definition for the number of customers served per hour
def customers_per_hour : ℕ := 7

-- Total number of customers served by Ann, Becky, and Julia
def total_customers : ℕ :=
  (hours_ann * customers_per_hour) + 
  (hours_becky * customers_per_hour) + 
  (hours_julia * customers_per_hour)

theorem total_customers_served : total_customers = 154 :=
  by 
    -- This is where the proof would go, but we'll use sorry to indicate it's incomplete
    sorry

end total_customers_served_l185_185069


namespace correct_operation_l185_185285

variable {a : ℝ}

theorem correct_operation : a^4 / (-a)^2 = a^2 := by
  sorry

end correct_operation_l185_185285


namespace radius_increase_125_surface_area_l185_185726

theorem radius_increase_125_surface_area (r r' : ℝ) 
(increase_surface_area : 4 * π * (r'^2) = 2.25 * 4 * π * r^2) : r' = 1.5 * r :=
by 
  sorry

end radius_increase_125_surface_area_l185_185726


namespace triangle_ABC_perimeter_l185_185173

noncomputable def triangle_perimeter (A B C D : Type) (AD BC AC AB : ℝ) : ℝ :=
  AD + BC + AC + AB

theorem triangle_ABC_perimeter (A B C D : Type) (AD BC : ℝ) (cos_BDC : ℝ) (angle_sum : ℝ) (AC : ℝ) (AB : ℝ) :
  AD = 3 → BC = 2 → cos_BDC = 13 / 20 → angle_sum = 180 → 
  (triangle_perimeter A B C D AD BC AC AB = 11) :=
by
  sorry

end triangle_ABC_perimeter_l185_185173


namespace g_ln_1_div_2017_l185_185572

open Real

-- Define the functions fulfilling the given conditions
variables (f g : ℝ → ℝ) (a : ℝ)

-- Define assumptions as required by the conditions
axiom f_property : ∀ m n : ℝ, f (m + n) = f m + f n - 1
axiom g_def : ∀ x : ℝ, g x = f x + a^x / (a^x + 1)
axiom a_property : a > 0 ∧ a ≠ 1
axiom g_ln_2017 : g (log 2017) = 2018

-- The theorem to prove
theorem g_ln_1_div_2017 : g (log (1 / 2017)) = -2015 := by
  sorry

end g_ln_1_div_2017_l185_185572


namespace stratified_sampling_school_C_l185_185649

theorem stratified_sampling_school_C 
  (teachers_A : ℕ) 
  (teachers_B : ℕ) 
  (teachers_C : ℕ) 
  (total_teachers : ℕ)
  (total_drawn : ℕ)
  (hA : teachers_A = 180)
  (hB : teachers_B = 140)
  (hC : teachers_C = 160)
  (hTotal : total_teachers = teachers_A + teachers_B + teachers_C)
  (hDraw : total_drawn = 60) :
  (total_drawn * teachers_C / total_teachers) = 20 := 
by
  sorry

end stratified_sampling_school_C_l185_185649


namespace calculation_A_correct_l185_185064

theorem calculation_A_correct : (-1: ℝ)^4 * (-1: ℝ)^3 = 1 := by
  sorry

end calculation_A_correct_l185_185064


namespace correct_calculated_value_l185_185190

theorem correct_calculated_value (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 :=
by 
  sorry

end correct_calculated_value_l185_185190


namespace solve_equation_l185_185455

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 :=
by
  sorry

end solve_equation_l185_185455


namespace outfit_combinations_l185_185352

theorem outfit_combinations 
  (shirts : Fin 5)
  (pants : Fin 6)
  (restricted_shirt : Fin 1)
  (restricted_pants : Fin 2) :
  ∃ total_combinations : ℕ, total_combinations = 28 :=
sorry

end outfit_combinations_l185_185352


namespace sequence_general_term_l185_185504

theorem sequence_general_term (a : ℕ → ℤ) (h₀ : a 0 = 1) (hstep : ∀ n, a (n + 1) = if a n = 1 then 0 else 1) :
  ∀ n, a n = (1 + (-1)^(n + 1)) / 2 :=
sorry

end sequence_general_term_l185_185504


namespace p_sufficient_not_necessary_for_q_l185_185143

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l185_185143


namespace propositions_p_q_l185_185021

theorem propositions_p_q
  (p q : Prop)
  (h : ¬(p ∧ q) = False) : p ∧ q :=
by
  sorry

end propositions_p_q_l185_185021


namespace paul_final_balance_l185_185101

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end paul_final_balance_l185_185101


namespace valid_word_combinations_l185_185840

-- Definition of valid_combination based on given conditions
def valid_combination : ℕ :=
  26 * 5 * 26

-- Statement to prove the number of valid four-letter combinations is 3380
theorem valid_word_combinations : valid_combination = 3380 := by
  sorry

end valid_word_combinations_l185_185840


namespace cube_surface_area_difference_l185_185014

theorem cube_surface_area_difference :
  let large_cube_volume := 8
  let small_cube_volume := 1
  let num_small_cubes := 8
  let large_cube_side := (large_cube_volume : ℝ) ^ (1 / 3)
  let small_cube_side := (small_cube_volume : ℝ) ^ (1 / 3)
  let large_cube_surface_area := 6 * (large_cube_side ^ 2)
  let small_cube_surface_area := 6 * (small_cube_side ^ 2)
  let total_small_cubes_surface_area := num_small_cubes * small_cube_surface_area
  total_small_cubes_surface_area - large_cube_surface_area = 24 :=
by
  sorry

end cube_surface_area_difference_l185_185014


namespace roots_quadratic_identity_l185_185884

theorem roots_quadratic_identity (p q : ℝ) (r s : ℝ) (h1 : r + s = 3 * p) (h2 : r * s = 2 * q) :
  r^2 + s^2 = 9 * p^2 - 4 * q := 
by 
  sorry

end roots_quadratic_identity_l185_185884


namespace second_number_in_first_set_l185_185810

theorem second_number_in_first_set :
  ∃ (x : ℝ), (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 ∧ x = 40 :=
by
  use 40
  sorry

end second_number_in_first_set_l185_185810


namespace find_exponent_l185_185814

theorem find_exponent (n : ℝ) (hn: (3:ℝ)^n = Real.sqrt 3) : n = 1 / 2 :=
by sorry

end find_exponent_l185_185814


namespace hex_B2F_to_dec_l185_185843

theorem hex_B2F_to_dec : 
  let A := 10
  let B := 11
  let C := 12
  let D := 13
  let E := 14
  let F := 15
  let base := 16
  let b2f := B * base^2 + 2 * base^1 + F * base^0
  b2f = 2863 :=
by {
  sorry
}

end hex_B2F_to_dec_l185_185843


namespace solve_inequality_l185_185667

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

theorem solve_inequality (x : ℝ) (hx : x ≠ 0 ∧ 0 < x) :
  (64 + (log_b (1/5) (x^2))^3) / (log_b (1/5) (x^6) * log_b 5 (x^2) + 5 * log_b 5 (x^6) + 14 * log_b (1/5) (x^2) + 2) ≤ 0 ↔
  (x ∈ Set.Icc (-25 : ℝ) (- Real.sqrt 5)) ∨
  (x ∈ Set.Icc (- (Real.exp (Real.log 5 / 3))) 0) ∨
  (x ∈ Set.Icc 0 (Real.exp (Real.log 5 / 3))) ∨
  (x ∈ Set.Icc (Real.sqrt 5) 25) :=
by 
  sorry

end solve_inequality_l185_185667


namespace original_number_is_80_l185_185354

theorem original_number_is_80 (x : ℝ) (h1 : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end original_number_is_80_l185_185354


namespace checkerboards_that_cannot_be_covered_l185_185575

-- Define the dimensions of the checkerboards
def checkerboard_4x6 := (4, 6)
def checkerboard_3x7 := (3, 7)
def checkerboard_5x5 := (5, 5)
def checkerboard_7x4 := (7, 4)
def checkerboard_5x6 := (5, 6)

-- Define a function to calculate the number of squares
def num_squares (dims : Nat × Nat) : Nat := dims.1 * dims.2

-- Define a function to check if a board can be exactly covered by dominoes
def can_be_covered_by_dominoes (dims : Nat × Nat) : Bool := (num_squares dims) % 2 == 0

-- Statement to be proven
theorem checkerboards_that_cannot_be_covered :
  ¬ can_be_covered_by_dominoes checkerboard_3x7 ∧ ¬ can_be_covered_by_dominoes checkerboard_5x5 :=
by
  sorry

end checkerboards_that_cannot_be_covered_l185_185575


namespace smallest_angle_terminal_side_l185_185149

theorem smallest_angle_terminal_side (θ : ℝ) (H : θ = 2011) :
  ∃ φ : ℝ, 0 ≤ φ ∧ φ < 360 ∧ (∃ k : ℤ, φ = θ - 360 * k) ∧ φ = 211 :=
by
  sorry

end smallest_angle_terminal_side_l185_185149


namespace negation_of_P_is_true_l185_185755

theorem negation_of_P_is_true :
  ¬ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by sorry

end negation_of_P_is_true_l185_185755


namespace shirt_cost_is_ten_l185_185365

theorem shirt_cost_is_ten (S J : ℝ) (h1 : J = 2 * S) 
    (h2 : 20 * S + 10 * J = 400) : S = 10 :=
by
  -- proof skipped
  sorry

end shirt_cost_is_ten_l185_185365


namespace determine_a_l185_185347

theorem determine_a :
  ∃ a : ℝ, (∀ x : ℝ, y = -((x - a) / (x - a - 1)) ↔ x = (3 - a) / (3 - a - 1)) → a = 2 :=
sorry

end determine_a_l185_185347


namespace annual_increase_of_chickens_l185_185705

theorem annual_increase_of_chickens 
  (chickens_now : ℕ)
  (chickens_after_9_years : ℕ)
  (years : ℕ)
  (chickens_now_eq : chickens_now = 550)
  (chickens_after_9_years_eq : chickens_after_9_years = 1900)
  (years_eq : years = 9)
  : ((chickens_after_9_years - chickens_now) / years) = 150 :=
by
  sorry

end annual_increase_of_chickens_l185_185705


namespace range_of_a_l185_185505

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l185_185505


namespace two_numbers_solution_l185_185233

noncomputable def a := 8 + Real.sqrt 58
noncomputable def b := 8 - Real.sqrt 58

theorem two_numbers_solution : 
  (Real.sqrt (a * b) = Real.sqrt 6) ∧ ((2 * a * b) / (a + b) = 3 / 4) → 
  (a = 8 + Real.sqrt 58 ∧ b = 8 - Real.sqrt 58) ∨ (a = 8 - Real.sqrt 58 ∧ b = 8 + Real.sqrt 58) := 
by
  sorry

end two_numbers_solution_l185_185233


namespace sum_of_products_is_70_l185_185030

theorem sum_of_products_is_70 (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 149) (h2 : a + b + c = 17) :
  a * b + b * c + c * a = 70 :=
by
  sorry 

end sum_of_products_is_70_l185_185030


namespace least_positive_integer_not_representable_as_fraction_l185_185955

theorem least_positive_integer_not_representable_as_fraction : 
  ¬ ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ (2^a - 2^b) / (2^c - 2^d) = 11 :=
sorry

end least_positive_integer_not_representable_as_fraction_l185_185955


namespace find_three_digit_number_l185_185024

def is_three_digit_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c

theorem find_three_digit_number : 
  ∃ n : ℕ, is_three_digit_number n ∧ n^2 = (digits_sum n)^5 ∧ n = 243 :=
sorry

end find_three_digit_number_l185_185024


namespace math_proof_problem_l185_185509

noncomputable def problemStatement : Prop :=
  ∃ (α : ℝ), 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180) ∧ 
  (Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2)

theorem math_proof_problem : problemStatement := 
by 
  sorry

end math_proof_problem_l185_185509


namespace geometric_sequence_sum_l185_185945

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : q = 2) (h3 : a 0 + a 1 + a 2 = 21) : 
  a 2 + a 3 + a 4 = 84 :=
sorry

end geometric_sequence_sum_l185_185945


namespace complement_U_A_is_singleton_one_l185_185777

-- Define the universe and subset
def U : Set ℝ := Set.Icc 0 1
def A : Set ℝ := Set.Ico 0 1

-- Define the complement of A relative to U
def complement_U_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_U_A_is_singleton_one : complement_U_A = {1} := by
  sorry

end complement_U_A_is_singleton_one_l185_185777


namespace sqrt_fraction_difference_l185_185716

theorem sqrt_fraction_difference : 
  (Real.sqrt (16 / 9) - Real.sqrt (9 / 16)) = 7 / 12 :=
by
  sorry

end sqrt_fraction_difference_l185_185716


namespace y_intercept_line_l185_185920

theorem y_intercept_line : 
  ∃ m b : ℝ, 
  (2 * m + b = -3) ∧ 
  (6 * m + b = 5) ∧ 
  b = -7 :=
by 
  sorry

end y_intercept_line_l185_185920


namespace positive_difference_abs_eq_l185_185641

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l185_185641


namespace konjok_gorbunok_should_act_l185_185384

def magical_power_retention (eat : ℕ → Prop) (sleep : ℕ → Prop) (seven_days : ℕ) : Prop :=
  ∀ t : ℕ, (0 ≤ t ∧ t ≤ seven_days) → ¬(eat t ∨ sleep t)

def retains_power (need_action : Prop) : Prop :=
  need_action

theorem konjok_gorbunok_should_act
  (eat : ℕ → Prop) (sleep : ℕ → Prop)
  (seven_days : ℕ)
  (h : magical_power_retention eat sleep seven_days)
  (before_start : ℕ → Prop) :
  retains_power (before_start seven_days) :=
by
  sorry

end konjok_gorbunok_should_act_l185_185384


namespace percentage_change_difference_l185_185612

theorem percentage_change_difference (total_students : ℕ) (initial_enjoy : ℕ) (initial_not_enjoy : ℕ) (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  total_students = 100 →
  initial_enjoy = 40 →
  initial_not_enjoy = 60 →
  final_enjoy = 80 →
  final_not_enjoy = 20 →
  (40 ≤ y ∧ y ≤ 80) ∧ (40 - 40 = 0) ∧ (80 - 40 = 40) ∧ (80 - 40 = 40) :=
by
  sorry

end percentage_change_difference_l185_185612


namespace number_of_pickers_is_221_l185_185519
-- Import necessary Lean and math libraries

/--
Given the conditions:
1. The number of pickers fills 100 drums of raspberries per day.
2. The number of pickers fills 221 drums of grapes per day.
3. In 77 days, the pickers would fill 17017 drums of grapes.
Prove that the number of pickers is 221.
-/
theorem number_of_pickers_is_221
  (P : ℕ)
  (d1 : P * 100 = 100 * P)
  (d2 : P * 221 = 221 * P)
  (d17 : P * 221 * 77 = 17017) : 
  P = 221 := 
sorry

end number_of_pickers_is_221_l185_185519


namespace triangle_inequality_sqrt_sides_l185_185910

theorem triangle_inequality_sqrt_sides {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b):
  (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) 
  ∧ (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) = Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_sqrt_sides_l185_185910


namespace find_math_books_l185_185842

theorem find_math_books 
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) : 
  M = 10 := 
by 
  sorry

end find_math_books_l185_185842


namespace math_problem_l185_185341

theorem math_problem 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (h_def : ∀ x, f x = 2 * Real.sin (2 * x + phi) + 1)
  (h_point : f 0 = 0)
  (h_phi_range : -Real.pi / 2 < phi ∧ phi < 0) : 
  (phi = -Real.pi / 6) ∧ (∃ k : ℤ, ∀ x, f x = 3 ↔ x = k * Real.pi + 2 * Real.pi / 3) :=
sorry

end math_problem_l185_185341


namespace exists_fraction_x_only_and_f_of_1_is_0_l185_185815

theorem exists_fraction_x_only_and_f_of_1_is_0 : ∃ f : ℚ → ℚ, (∀ x : ℚ, f x = (x - 1) / x) ∧ f 1 = 0 := 
by
  sorry

end exists_fraction_x_only_and_f_of_1_is_0_l185_185815


namespace inlet_pipe_rate_16_liters_per_minute_l185_185665

noncomputable def rate_of_inlet_pipe : ℝ :=
  let capacity := 21600 -- litres
  let outlet_time_alone := 10 -- hours
  let outlet_time_with_inlet := 18 -- hours
  let outlet_rate := capacity / outlet_time_alone
  let combined_rate := capacity / outlet_time_with_inlet
  let inlet_rate := outlet_rate - combined_rate
  inlet_rate / 60 -- converting litres/hour to litres/min

theorem inlet_pipe_rate_16_liters_per_minute : rate_of_inlet_pipe = 16 :=
by
  sorry

end inlet_pipe_rate_16_liters_per_minute_l185_185665


namespace single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l185_185982

section transmission_scheme

variables (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1)

-- Part A
theorem single_transmission_probability :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β) ^ 2 :=
by sorry

-- Part B
theorem triple_transmission_probability :
  (1 - β) * β * (1 - β) = β * (1 - β) ^ 2 :=
by sorry

-- Part C
theorem triple_transmission_decoding :
  (3 * β * (1 - β) ^ 2) + (1 - β) ^ 3 = β * (1 - β) ^ 2 + (1 - β) ^ 3 :=
by sorry

-- Part D
theorem decoding_comparison (h : 0 < α ∧ α < 0.5) :
  (1 - α) < (3 * α * (1 - α) ^ 2 + (1 - α) ^ 3) :=
by sorry

end transmission_scheme

end single_transmission_probability_triple_transmission_probability_triple_transmission_decoding_decoding_comparison_l185_185982


namespace find_total_values_l185_185769

theorem find_total_values (n : ℕ) (S : ℝ) 
  (h1 : S / n = 150) 
  (h2 : (S + 25) / n = 151.25) 
  (h3 : 25 = 160 - 135) : n = 20 :=
by
  sorry

end find_total_values_l185_185769


namespace maximum_value_of_a_l185_185555

theorem maximum_value_of_a
  (a b c d : ℝ)
  (h1 : b + c + d = 3 - a)
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) :
  a ≤ 2 := by
  sorry

end maximum_value_of_a_l185_185555


namespace vendor_sells_50_percent_on_first_day_l185_185829

variables (A : ℝ) (S : ℝ)

theorem vendor_sells_50_percent_on_first_day 
  (h : 0.2 * A * (1 - S) + 0.5 * A * (1 - S) * 0.8 = 0.3 * A) : S = 0.5 :=
  sorry

end vendor_sells_50_percent_on_first_day_l185_185829


namespace tangent_line_through_point_l185_185659

theorem tangent_line_through_point (x y : ℝ) (tangent f : ℝ → ℝ) (M : ℝ × ℝ) :
  M = (1, 1) →
  f x = x^3 + 1 →
  tangent x = 3 * x^2 →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ ∀ x0 y0 : ℝ, (y0 = f x0) → (y - y0 = tangent x0 * (x - x0))) ∧
  (x, y) = M →
  (a = 0 ∧ b = 1 ∧ c = -1) ∨ (a = 27 ∧ b = -4 ∧ c = -23) :=
by
  sorry

end tangent_line_through_point_l185_185659


namespace eldest_age_l185_185172

theorem eldest_age (A B C : ℕ) (x : ℕ) 
  (h1 : A = 5 * x)
  (h2 : B = 7 * x)
  (h3 : C = 8 * x)
  (h4 : (5 * x - 7) + (7 * x - 7) + (8 * x - 7) = 59) :
  C = 32 := 
by 
  sorry

end eldest_age_l185_185172


namespace dot_product_vec_a_vec_b_l185_185072

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem dot_product_vec_a_vec_b : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 3 := by
  sorry

end dot_product_vec_a_vec_b_l185_185072


namespace age_of_oldest_sibling_l185_185215

theorem age_of_oldest_sibling (Kay_siblings : ℕ) (Kay_age : ℕ) (youngest_sibling_age : ℕ) (oldest_sibling_age : ℕ) 
  (h1 : Kay_siblings = 14) (h2 : Kay_age = 32) (h3 : youngest_sibling_age = Kay_age / 2 - 5) 
  (h4 : oldest_sibling_age = 4 * youngest_sibling_age) : oldest_sibling_age = 44 := 
sorry

end age_of_oldest_sibling_l185_185215


namespace probability_at_least_four_same_face_l185_185700

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_four_same_face_l185_185700


namespace percentage_increase_l185_185576

variables (A B C D E : ℝ)
variables (A_inc B_inc C_inc D_inc E_inc : ℝ)

-- Conditions
def conditions (A_inc B_inc C_inc D_inc E_inc : ℝ) :=
  A_inc = 0.1 * A ∧
  B_inc = (1/15) * B ∧
  C_inc = 0.05 * C ∧
  D_inc = 0.04 * D ∧
  E_inc = (1/30) * E ∧
  B = 1.5 * A ∧
  C = 2 * A ∧
  D = 2.5 * A ∧
  E = 3 * A

-- Theorem to prove
theorem percentage_increase (A B C D E : ℝ) (A_inc B_inc C_inc D_inc E_inc : ℝ) :
  conditions A B C D E A_inc B_inc C_inc D_inc E_inc →
  (A_inc + B_inc + C_inc + D_inc + E_inc) / (A + B + C + D + E) = 0.05 :=
by
  sorry

end percentage_increase_l185_185576


namespace course_selection_plans_l185_185900

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_plans :
  let A_courses := C 4 2
  let B_courses := C 4 3
  let C_courses := C 4 3
  A_courses * B_courses * C_courses = 96 :=
by
  sorry

end course_selection_plans_l185_185900


namespace f_m_plus_1_positive_l185_185863

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + x + a

theorem f_m_plus_1_positive {m a : ℝ} (h_a_pos : a > 0) (h_f_m_neg : f m a < 0) : f (m + 1) a > 0 := by
  sorry

end f_m_plus_1_positive_l185_185863


namespace cost_of_paving_floor_l185_185886

-- Define the constants given in the problem
def length1 : ℝ := 5.5
def width1 : ℝ := 3.75
def length2 : ℝ := 4
def width2 : ℝ := 3
def cost_per_sq_meter : ℝ := 800

-- Define the areas of the two rectangles
def area1 : ℝ := length1 * width1
def area2 : ℝ := length2 * width2

-- Define the total area of the floor
def total_area : ℝ := area1 + area2

-- Define the total cost of paving the floor
def total_cost : ℝ := total_area * cost_per_sq_meter

-- The statement to prove: the total cost equals 26100 Rs
theorem cost_of_paving_floor : total_cost = 26100 := by
  -- Proof skipped
  sorry

end cost_of_paving_floor_l185_185886


namespace portion_divided_equally_for_efforts_l185_185441

-- Definitions of conditions
def tom_investment : ℝ := 700
def jerry_investment : ℝ := 300
def tom_more_than_jerry : ℝ := 800
def total_profit : ℝ := 3000

-- Theorem stating what we need to prove
theorem portion_divided_equally_for_efforts (T J R E : ℝ) 
  (h1 : T = tom_investment)
  (h2 : J = jerry_investment)
  (h3 : total_profit = R)
  (h4 : (E / 2) + (7 / 10) * (R - E) - (E / 2 + (3 / 10) * (R - E)) = tom_more_than_jerry) 
  : E = 1000 :=
by
  sorry

end portion_divided_equally_for_efforts_l185_185441


namespace minimum_value_m_ineq_proof_l185_185757

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem minimum_value_m (x₀ : ℝ) (m : ℝ) (hx : f x₀ ≤ m) : 4 ≤ m := by
  sorry

theorem ineq_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 4) : 3 ≤ 3 / b + 1 / a := by
  sorry

end minimum_value_m_ineq_proof_l185_185757


namespace product_floor_ceil_sequence_l185_185999

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem product_floor_ceil_sequence :
    (floor (-6 - 0.5) * ceil (6 + 0.5)) *
    (floor (-5 - 0.5) * ceil (5 + 0.5)) *
    (floor (-4 - 0.5) * ceil (4 + 0.5)) *
    (floor (-3 - 0.5) * ceil (3 + 0.5)) *
    (floor (-2 - 0.5) * ceil (2 + 0.5)) *
    (floor (-1 - 0.5) * ceil (1 + 0.5)) *
    (floor (-0.5) * ceil (0.5)) = -25401600 :=
by
  sorry

end product_floor_ceil_sequence_l185_185999


namespace not_factorable_l185_185628

-- Define the quartic polynomial P(x)
def P (x : ℤ) : ℤ := x^4 + 2 * x^2 + 2 * x + 2

-- Define the quadratic polynomials with integer coefficients
def Q₁ (a b x : ℤ) : ℤ := x^2 + a * x + b
def Q₂ (c d x : ℤ) : ℤ := x^2 + c * x + d

-- Define the condition for factorization, and the theorem to be proven
theorem not_factorable :
  ¬ ∃ (a b c d : ℤ), ∀ x : ℤ, P x = (Q₁ a b x) * (Q₂ c d x) := by
  sorry

end not_factorable_l185_185628


namespace total_peaches_l185_185267

theorem total_peaches (initial_peaches_Audrey : ℕ) (multiplier_Audrey : ℕ)
                      (initial_peaches_Paul : ℕ) (multiplier_Paul : ℕ)
                      (initial_peaches_Maya : ℕ) (additional_peaches_Maya : ℕ) :
                      initial_peaches_Audrey = 26 →
                      multiplier_Audrey = 3 →
                      initial_peaches_Paul = 48 →
                      multiplier_Paul = 2 →
                      initial_peaches_Maya = 57 →
                      additional_peaches_Maya = 20 →
                      (initial_peaches_Audrey + multiplier_Audrey * initial_peaches_Audrey) +
                      (initial_peaches_Paul + multiplier_Paul * initial_peaches_Paul) +
                      (initial_peaches_Maya + additional_peaches_Maya) = 325 :=
by
  sorry

end total_peaches_l185_185267


namespace p_necessary_not_sufficient_for_q_l185_185127

def p (x : ℝ) : Prop := abs x = -x
def q (x : ℝ) : Prop := x^2 ≥ -x

theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l185_185127


namespace half_AB_equals_l185_185650

-- Define vectors OA and OB
def vector_OA : ℝ × ℝ := (3, 2)
def vector_OB : ℝ × ℝ := (4, 7)

-- Prove that (1 / 2) * (OB - OA) = (1 / 2, 5 / 2)
theorem half_AB_equals :
  (1 / 2 : ℝ) • ((vector_OB.1 - vector_OA.1), (vector_OB.2 - vector_OA.2)) = (1 / 2, 5 / 2) := 
  sorry

end half_AB_equals_l185_185650


namespace minimum_guests_l185_185048

theorem minimum_guests (total_food : ℤ) (max_food_per_guest : ℤ) (food_bound : total_food = 325) (guest_bound : max_food_per_guest = 2) : (⌈total_food / max_food_per_guest⌉ : ℤ) = 163 :=
by {
  sorry 
}

end minimum_guests_l185_185048


namespace max_sum_m_n_l185_185140

noncomputable def ellipse_and_hyperbola_max_sum : Prop :=
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ (∃ x y : ℝ, (x^2 / 25 + y^2 / m^2 = 1 ∧ x^2 / 7 - y^2 / n^2 = 1)) ∧
  (25 - m^2 = 7 + n^2) ∧ (m + n = 6)

theorem max_sum_m_n : ellipse_and_hyperbola_max_sum :=
  sorry

end max_sum_m_n_l185_185140


namespace cost_of_largest_pot_equals_229_l185_185748

-- Define the conditions
variables (total_cost : ℝ) (num_pots : ℕ) (cost_diff : ℝ)

-- Assume given conditions
axiom h1 : num_pots = 6
axiom h2 : total_cost = 8.25
axiom h3 : cost_diff = 0.3

-- Define the function for the cost of the smallest pot and largest pot
noncomputable def smallest_pot_cost : ℝ :=
  (total_cost - (num_pots - 1) * cost_diff) / num_pots

noncomputable def largest_pot_cost : ℝ :=
  smallest_pot_cost total_cost num_pots cost_diff + (num_pots - 1) * cost_diff

-- Prove the cost of the largest pot equals 2.29
theorem cost_of_largest_pot_equals_229 (h1 : num_pots = 6) (h2 : total_cost = 8.25) (h3 : cost_diff = 0.3) :
  largest_pot_cost total_cost num_pots cost_diff = 2.29 :=
  by sorry

end cost_of_largest_pot_equals_229_l185_185748


namespace range_of_k_is_l185_185664

noncomputable def range_of_k (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : Set ℝ :=
{k : ℝ | ∀ x : ℝ, a^x + 4 * a^(-x) - k > 0}

theorem range_of_k_is (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  range_of_k a h₁ h₂ = { k : ℝ | k < 4 ∧ k ≠ 3 } :=
sorry

end range_of_k_is_l185_185664


namespace rice_in_first_5_days_l185_185948

-- Define the arithmetic sequence for number of workers dispatched each day
def num_workers (n : ℕ) : ℕ := 64 + (n - 1) * 7

-- Function to compute the sum of the first n terms of the arithmetic sequence
def sum_workers (n : ℕ) : ℕ := n * 64 + (n * (n - 1)) / 2 * 7

-- Given the rice distribution conditions
def rice_per_worker : ℕ := 3

-- Given the problem specific conditions
def total_rice_distributed_first_5_days : ℕ := 
  rice_per_worker * (sum_workers 1 + sum_workers 2 + sum_workers 3 + sum_workers 4 + sum_workers 5)
  
-- Proof goal
theorem rice_in_first_5_days : total_rice_distributed_first_5_days = 3300 :=
  by
  sorry

end rice_in_first_5_days_l185_185948


namespace cost_difference_l185_185714

-- Given conditions
def first_present_cost : ℕ := 18
def third_present_cost : ℕ := first_present_cost - 11
def total_cost : ℕ := 50

-- denoting costs of the second present via variable
def second_present_cost (x : ℕ) : Prop :=
  first_present_cost + x + third_present_cost = total_cost

-- Goal statement
theorem cost_difference (x : ℕ) (h : second_present_cost x) : x - first_present_cost = 7 :=
  sorry

end cost_difference_l185_185714


namespace annual_population_growth_l185_185446

noncomputable def annual_percentage_increase := 
  let P0 := 15000
  let P2 := 18150  
  exists (r : ℝ), (P0 * (1 + r)^2 = P2) ∧ (r = 0.1)

theorem annual_population_growth : annual_percentage_increase :=
by
  -- Placeholder proof
  sorry

end annual_population_growth_l185_185446


namespace max_a_monotonic_f_l185_185978

theorem max_a_monotonic_f {a : ℝ} (h1 : 0 < a)
  (h2 : ∀ x ≥ 1, 0 ≤ (3 * x^2 - a)) : a ≤ 3 := by
  -- Proof to be provided
  sorry

end max_a_monotonic_f_l185_185978


namespace f_at_2023_l185_185086

noncomputable def f (a x : ℝ) : ℝ := (a - x) / (a + 2 * x)

noncomputable def g (a x : ℝ) : ℝ := (f a (x - 2023)) + (1 / 2)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

variable (a : ℝ)
variable (h_a : a ≠ 0)
variable (h_odd : is_odd (g a))

theorem f_at_2023 : f a 2023 = 1 / 4 :=
sorry

end f_at_2023_l185_185086


namespace find_blue_shirts_l185_185083

-- Statements of the problem conditions
def total_shirts : ℕ := 23
def green_shirts : ℕ := 17

-- Definition that we want to prove
def blue_shirts : ℕ := total_shirts - green_shirts

-- Proof statement (no need to include the proof itself)
theorem find_blue_shirts : blue_shirts = 6 := by
  sorry

end find_blue_shirts_l185_185083


namespace origin_movement_by_dilation_l185_185002

/-- Given a dilation of the plane that maps a circle with radius 4 centered at (3,3) 
to a circle of radius 6 centered at (7,9), calculate the distance the origin (0,0)
moves under this transformation to be 0.5 * sqrt(10). -/
theorem origin_movement_by_dilation :
  let B := (3, 3)
  let B' := (7, 9)
  let radius_B := 4
  let radius_B' := 6
  let dilation_factor := radius_B' / radius_B
  let center_of_dilation := (-1, -3)
  let initial_distance := Real.sqrt ((-1)^2 + (-3)^2) 
  let moved_distance := dilation_factor * initial_distance
  moved_distance - initial_distance = 0.5 * Real.sqrt (10) := 
by
  sorry

end origin_movement_by_dilation_l185_185002


namespace basketball_team_initial_games_l185_185867

theorem basketball_team_initial_games (G W : ℝ) 
  (h1 : W = 0.70 * G) 
  (h2 : W + 2 = 0.60 * (G + 10)) : 
  G = 40 :=
by
  sorry

end basketball_team_initial_games_l185_185867


namespace compute_100a_b_l185_185510

theorem compute_100a_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) * (x + 10) = 0 ↔ x = -a ∨ x = -b ∨ x = -10)
  (h2 : a ≠ -4 ∧ b ≠ -4 ∧ 10 ≠ -4)
  (h3 : ∀ x : ℝ, (x + 2 * a) * (x + 5) * (x + 8) = 0 ↔ x = -5)
  (hb : b = 8)
  (ha : 2 * a = 5) :
  100 * a + b = 258 := 
sorry

end compute_100a_b_l185_185510


namespace min_value_of_one_over_a_and_one_over_b_l185_185047

noncomputable def minValue (a b : ℝ) : ℝ :=
  if 2 * a + 3 * b = 1 then 1 / a + 1 / b else 0

theorem min_value_of_one_over_a_and_one_over_b :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 1 ∧ minValue a b = 65 / 6 :=
by
  sorry

end min_value_of_one_over_a_and_one_over_b_l185_185047


namespace solve_quadratic_l185_185038

theorem solve_quadratic : ∀ (x : ℝ), x^2 - 5 * x + 1 = 0 →
  (x = (5 + Real.sqrt 21) / 2) ∨ (x = (5 - Real.sqrt 21) / 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l185_185038


namespace B_subset_A_l185_185207

variable {α : Type*}
variable (A B : Set α)

def A_def : Set ℝ := { x | x ≥ 1 }
def B_def : Set ℝ := { x | x > 2 }

theorem B_subset_A : B_def ⊆ A_def :=
sorry

end B_subset_A_l185_185207


namespace days_y_needs_l185_185228

theorem days_y_needs
  (d : ℝ)
  (h1 : (1:ℝ) / 21 * 14 = 1 - 5 * (1 / d)) :
  d = 10 :=
sorry

end days_y_needs_l185_185228


namespace hannahs_peppers_total_weight_l185_185858

theorem hannahs_peppers_total_weight:
  let green := 0.3333333333333333
  let red := 0.3333333333333333
  let yellow := 0.25
  let orange := 0.5
  green + red + yellow + orange = 1.4166666666666665 :=
by
  repeat { sorry } -- Placeholder for the actual proof

end hannahs_peppers_total_weight_l185_185858


namespace vec_op_not_comm_l185_185833

open Real

-- Define the operation ⊙
def vec_op (a b: ℝ × ℝ) : ℝ :=
  (a.1 * b.2) - (a.2 * b.1)

-- Define a predicate to check if two vectors are collinear
def collinear (a b: ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Define the proof theorem
theorem vec_op_not_comm (a b: ℝ × ℝ) : vec_op a b ≠ vec_op b a :=
by
  -- The contents of the proof will go here. Insert 'sorry' to skip.
  sorry

end vec_op_not_comm_l185_185833


namespace cost_of_toast_l185_185162

theorem cost_of_toast (egg_cost : ℕ) (toast_cost : ℕ)
  (dale_toasts : ℕ) (dale_eggs : ℕ)
  (andrew_toasts : ℕ) (andrew_eggs : ℕ)
  (total_cost : ℕ)
  (h1 : egg_cost = 3)
  (h2 : dale_toasts = 2)
  (h3 : dale_eggs = 2)
  (h4 : andrew_toasts = 1)
  (h5 : andrew_eggs = 2)
  (h6 : 2 * toast_cost + dale_eggs * egg_cost 
        + andrew_toasts * toast_cost + andrew_eggs * egg_cost = total_cost) :
  total_cost = 15 → toast_cost = 1 :=
by
  -- Proof not needed
  sorry

end cost_of_toast_l185_185162


namespace rope_length_third_post_l185_185009

theorem rope_length_third_post (total first second fourth : ℕ) (h_total : total = 70) 
    (h_first : first = 24) (h_second : second = 20) (h_fourth : fourth = 12) : 
    (total - first - second - fourth) = 14 :=
by
  -- Proof is skipped, but we can state that the theorem should follow from the given conditions.
  sorry

end rope_length_third_post_l185_185009


namespace folded_rectangle_perimeter_l185_185478

theorem folded_rectangle_perimeter (l : ℝ) (w : ℝ) (h_diag : ℝ)
  (h_l : l = 20) (h_w : w = 12)
  (h_diag : h_diag = Real.sqrt (l^2 + w^2)) :
  2 * (l + w) = 64 :=
by
  rw [h_l, h_w]
  simp only [mul_add, mul_two, add_mul] at *
  norm_num


end folded_rectangle_perimeter_l185_185478


namespace solve_fraction_equation_l185_185702

theorem solve_fraction_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : x = -2 / 3 :=
by
  sorry

end solve_fraction_equation_l185_185702


namespace water_volume_in_B_when_A_is_0_point_4_l185_185942

noncomputable def pool_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

noncomputable def valve_rate (volume time : ℝ) : ℝ :=
  volume / time

theorem water_volume_in_B_when_A_is_0_point_4 :
  ∀ (length width depth : ℝ)
    (time_A_fill time_A_to_B : ℝ)
    (depth_A_target : ℝ),
    length = 3 → width = 2 → depth = 1.2 →
    time_A_fill = 18 → time_A_to_B = 24 →
    depth_A_target = 0.4 →
    pool_volume length width depth = 7.2 →
    valve_rate 7.2 time_A_fill = 0.4 →
    valve_rate 7.2 time_A_to_B = 0.3 →
    ∃ (time_required : ℝ),
    time_required = 24 →
    (valve_rate 7.2 time_A_to_B * time_required = 7.2) :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end water_volume_in_B_when_A_is_0_point_4_l185_185942


namespace slope_of_line_l185_185828

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (- (4 : ℝ) / 7) = -4 / 7 :=
by
  -- Sorry for the proof for completeness
  sorry

end slope_of_line_l185_185828


namespace square_field_side_length_l185_185428

theorem square_field_side_length (t : ℕ) (v : ℕ) 
  (run_time : t = 56) 
  (run_speed : v = 9) : 
  ∃ l : ℝ, l = 35 := 
sorry

end square_field_side_length_l185_185428


namespace smallest_consecutive_sum_l185_185804

theorem smallest_consecutive_sum (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by 
  sorry

end smallest_consecutive_sum_l185_185804


namespace response_rate_increase_l185_185010

theorem response_rate_increase :
  let original_customers := 70
  let original_responses := 7
  let redesigned_customers := 63
  let redesigned_responses := 9
  let original_response_rate := (original_responses : ℝ) / original_customers
  let redesigned_response_rate := (redesigned_responses : ℝ) / redesigned_customers
  let percentage_increase := ((redesigned_response_rate - original_response_rate) / original_response_rate) * 100
  abs (percentage_increase - 42.86) < 0.01 :=
by
  sorry

end response_rate_increase_l185_185010


namespace initial_candies_equal_twenty_l185_185809

-- Definitions based on conditions
def friends : ℕ := 6
def candies_per_friend : ℕ := 4
def total_needed_candies : ℕ := friends * candies_per_friend
def additional_candies : ℕ := 4

-- Main statement
theorem initial_candies_equal_twenty :
  (total_needed_candies - additional_candies) = 20 := by
  sorry

end initial_candies_equal_twenty_l185_185809


namespace angle_in_second_quadrant_l185_185237

theorem angle_in_second_quadrant (α : ℝ) (h₁ : -2 * Real.pi < α) (h₂ : α < -Real.pi) : 
  α = -4 → (α > -3 * Real.pi / 2 ∧ α < -Real.pi / 2) :=
by
  intros hα
  sorry

end angle_in_second_quadrant_l185_185237


namespace zach_needs_more_money_l185_185894

noncomputable def cost_of_bike : ℕ := 100
noncomputable def weekly_allowance : ℕ := 5
noncomputable def mowing_income : ℕ := 10
noncomputable def babysitting_rate_per_hour : ℕ := 7
noncomputable def initial_savings : ℕ := 65
noncomputable def hours_babysitting : ℕ := 2

theorem zach_needs_more_money : 
  cost_of_bike - (initial_savings + weekly_allowance + mowing_income + (babysitting_rate_per_hour * hours_babysitting)) = 6 :=
by
  sorry

end zach_needs_more_money_l185_185894


namespace intersection_of_A_and_B_l185_185807

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}
noncomputable def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_of_A_and_B_l185_185807


namespace intersection_M_N_l185_185350

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l185_185350


namespace number_of_Cl_atoms_l185_185917

def atomic_weight_H : ℝ := 1
def atomic_weight_Cl : ℝ := 35.5
def atomic_weight_O : ℝ := 16

def H_atoms : ℕ := 1
def O_atoms : ℕ := 2
def total_molecular_weight : ℝ := 68

theorem number_of_Cl_atoms :
  (total_molecular_weight - (H_atoms * atomic_weight_H + O_atoms * atomic_weight_O)) / atomic_weight_Cl = 1 :=
by
  -- proof to show this holds
  sorry

end number_of_Cl_atoms_l185_185917


namespace peanut_mixture_l185_185685

-- Definitions of given conditions
def virginia_peanuts_weight : ℝ := 10
def virginia_peanuts_cost_per_pound : ℝ := 3.50
def spanish_peanuts_cost_per_pound : ℝ := 3.00
def texan_peanuts_cost_per_pound : ℝ := 4.00
def desired_cost_per_pound : ℝ := 3.60

-- Definitions of unknowns S (Spanish peanuts) and T (Texan peanuts)
variable (S T : ℝ)

-- Equation derived from given conditions
theorem peanut_mixture :
  (0.40 * T) - (0.60 * S) = 1 := sorry

end peanut_mixture_l185_185685


namespace system_of_equations_solution_l185_185985

theorem system_of_equations_solution (x y : ℚ) :
  (x / 3 + y / 4 = 4 ∧ 2 * x - 3 * y = 12) → (x = 10 ∧ y = 8 / 3) :=
by
  sorry

end system_of_equations_solution_l185_185985


namespace david_marks_physics_l185_185541

def marks_english := 96
def marks_math := 95
def marks_chemistry := 97
def marks_biology := 95
def average_marks := 93
def number_of_subjects := 5

theorem david_marks_physics : 
  let total_marks := average_marks * number_of_subjects 
  let total_known_marks := marks_english + marks_math + marks_chemistry + marks_biology
  let marks_physics := total_marks - total_known_marks
  marks_physics = 82 :=
by
  sorry

end david_marks_physics_l185_185541


namespace ant_climbing_floors_l185_185988

theorem ant_climbing_floors (time_per_floor : ℕ) (total_time : ℕ) (floors_climbed : ℕ) :
  time_per_floor = 15 →
  total_time = 105 →
  floors_climbed = total_time / time_per_floor + 1 →
  floors_climbed = 8 :=
by
  intros
  sorry

end ant_climbing_floors_l185_185988


namespace remainder_of_polynomial_division_l185_185956

theorem remainder_of_polynomial_division :
  Polynomial.eval 2 (8 * X^3 - 22 * X^2 + 30 * X - 45) = -9 :=
by {
  sorry
}

end remainder_of_polynomial_division_l185_185956


namespace leftmost_three_nonzero_digits_of_arrangements_l185_185811

-- Definitions based on the conditions
def num_rings := 10
def chosen_rings := 6
def num_fingers := 5

-- Calculate the possible arrangements
def arrangements : ℕ := Nat.choose num_rings chosen_rings * Nat.factorial chosen_rings * Nat.choose (chosen_rings + (num_fingers - 1)) (num_fingers - 1)

-- Find the leftmost three nonzero digits
def leftmost_three_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.reverse.takeWhile (· > 0)).reverse.take 3
  |> List.foldl (· + · * 10) 0
  
-- The main theorem to prove
theorem leftmost_three_nonzero_digits_of_arrangements :
  leftmost_three_nonzero_digits arrangements = 317 :=
by
  sorry

end leftmost_three_nonzero_digits_of_arrangements_l185_185811


namespace cubic_sum_l185_185445

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := 
sorry

end cubic_sum_l185_185445


namespace find_number_l185_185878

theorem find_number (n : ℕ) (some_number : ℕ) 
  (h : (1/5 : ℝ)^n * (1/4 : ℝ)^(18 : ℕ) = 1 / (2 * (some_number : ℝ)^n))
  (hn : n = 35) : some_number = 10 := 
by 
  sorry

end find_number_l185_185878


namespace complement_of_A_with_respect_to_U_l185_185618

open Set

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}
def complement_U_A : Set ℕ := {4, 6}

theorem complement_of_A_with_respect_to_U :
  U \ A = complement_U_A := by
  sorry

end complement_of_A_with_respect_to_U_l185_185618


namespace roots_sum_prod_eq_l185_185449

theorem roots_sum_prod_eq (p q : ℤ) (h1 : p / 3 = 9) (h2 : q / 3 = 20) : p + q = 87 :=
by
  sorry

end roots_sum_prod_eq_l185_185449


namespace positive_difference_between_solutions_l185_185796

theorem positive_difference_between_solutions : 
  let f (x : ℝ) := (5 - (x^2 / 3 : ℝ))^(1 / 3 : ℝ)
  let a := 4 * Real.sqrt 6
  let b := -4 * Real.sqrt 6
  |a - b| = 8 * Real.sqrt 6 := 
by 
  sorry

end positive_difference_between_solutions_l185_185796


namespace find_fraction_of_number_l185_185834

theorem find_fraction_of_number (N : ℚ) (h : (3/10 : ℚ) * N - 8 = 12) :
  (1/5 : ℚ) * N = 40 / 3 :=
by
  sorry

end find_fraction_of_number_l185_185834


namespace factor_x4_minus_81_l185_185683

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end factor_x4_minus_81_l185_185683


namespace largest_perimeter_triangle_l185_185481

theorem largest_perimeter_triangle :
  ∃ (y : ℤ), 4 < y ∧ y < 20 ∧ 8 + 12 + y = 39 :=
by {
  -- we'll skip the proof steps
  sorry 
}

end largest_perimeter_triangle_l185_185481


namespace number_of_rectangles_l185_185864

-- Definition of the problem: We have 12 equally spaced points on a circle.
def points_on_circle : ℕ := 12

-- The number of diameters is half the number of points, as each diameter involves two points.
def diameters (n : ℕ) : ℕ := n / 2

-- The number of ways to choose 2 diameters out of n/2 is given by the binomial coefficient.
noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Prove the number of rectangles that can be formed is 15.
theorem number_of_rectangles :
  binomial_coefficient (diameters points_on_circle) 2 = 15 := by
  sorry

end number_of_rectangles_l185_185864


namespace correct_exponentiation_calculation_l185_185941

theorem correct_exponentiation_calculation (a : ℝ) : a^2 * a^6 = a^8 :=
by sorry

end correct_exponentiation_calculation_l185_185941


namespace farmer_planning_problem_l185_185018

theorem farmer_planning_problem
  (A : ℕ) (D : ℕ)
  (h1 : A = 120 * D)
  (h2 : ∀ t : ℕ, t = 85 * (D + 5) + 40)
  (h3 : 85 * (D + 5) + 40 = 120 * D) : 
  A = 1560 ∧ D = 13 := 
by
  sorry

end farmer_planning_problem_l185_185018


namespace min_value_frac_l185_185039

theorem min_value_frac (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  ∃ (x : ℝ), x = 1/m + 4/n ∧ x ≥ 9 :=
by
  sorry

end min_value_frac_l185_185039


namespace complement_intersection_eq_l185_185915

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3} :=
by
  sorry

end complement_intersection_eq_l185_185915


namespace man_age_twice_son_age_l185_185529

-- Definitions based on conditions
def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

-- Definition of the main statement to be proven
theorem man_age_twice_son_age (Y : ℕ) : man_age + Y = 2 * (son_age + Y) → Y = 2 :=
by sorry

end man_age_twice_son_age_l185_185529


namespace coefficient_x3_y7_expansion_l185_185518

theorem coefficient_x3_y7_expansion : 
  let n := 10
  let a := (2 : ℚ) / 3
  let b := -(3 : ℚ) / 5
  let k := 3
  let binom := Nat.choose n k
  let term := binom * (a ^ k) * (b ^ (n - k))
  term = -(256 : ℚ) / 257 := 
by
  -- Proof omitted
  sorry

end coefficient_x3_y7_expansion_l185_185518


namespace expression_value_l185_185385

theorem expression_value (a b c : ℝ) (h : a + b + c = 0) : (a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b)) = 3 := 
by 
  sorry

end expression_value_l185_185385


namespace no_member_of_T_is_divisible_by_4_l185_185989

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4 : ∀ n : ℤ, ¬ (sum_of_squares_of_four_consecutive_integers n % 4 = 0) := by
  intro n
  sorry

end no_member_of_T_is_divisible_by_4_l185_185989


namespace max_value_of_x2_plus_y2_l185_185946

theorem max_value_of_x2_plus_y2 {x y : ℝ} 
  (h1 : x ≥ 1)
  (h2 : y ≥ x)
  (h3 : x - 2 * y + 3 ≥ 0) : 
  x^2 + y^2 ≤ 18 :=
sorry

end max_value_of_x2_plus_y2_l185_185946


namespace diameter_in_scientific_notation_l185_185244

def diameter : ℝ := 0.00000011
def scientific_notation (d : ℝ) : Prop := d = 1.1e-7

theorem diameter_in_scientific_notation : scientific_notation diameter :=
by
  sorry

end diameter_in_scientific_notation_l185_185244


namespace sasha_questions_per_hour_l185_185309

-- Define the total questions and the time she worked, and the remaining questions
def total_questions : ℕ := 60
def time_worked : ℕ := 2
def remaining_questions : ℕ := 30

-- Define the number of questions she completed
def questions_completed := total_questions - remaining_questions

-- Define the rate at which she completes questions per hour
def questions_per_hour := questions_completed / time_worked

-- The theorem to prove
theorem sasha_questions_per_hour : questions_per_hour = 15 := 
by
  -- Here we would prove the theorem, but we're using sorry to skip the proof for now
  sorry

end sasha_questions_per_hour_l185_185309


namespace smallest_n_for_roots_of_unity_l185_185669

theorem smallest_n_for_roots_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ (n : ℕ), n = 18 ∧ z^n = 1 :=
by {
  sorry
}

end smallest_n_for_roots_of_unity_l185_185669


namespace range_of_a_l185_185433

theorem range_of_a (a : ℝ) :
  (1 < a ∧ a < 8 ∧ a ≠ 4) ↔
  (a > 1 ∧ a < 8) ∧ (a > -4 ∧ a ≠ 4) :=
by sorry

end range_of_a_l185_185433


namespace larger_number_of_hcf_23_lcm_factors_13_15_l185_185654

theorem larger_number_of_hcf_23_lcm_factors_13_15 :
  ∃ A B, (Nat.gcd A B = 23) ∧ (A * B = 23 * 13 * 15) ∧ (A = 345 ∨ B = 345) := sorry

end larger_number_of_hcf_23_lcm_factors_13_15_l185_185654


namespace salary_increase_difference_l185_185186

structure Person where
  name : String
  salary : ℕ
  raise_percent : ℕ
  investment_return : ℕ

def hansel := Person.mk "Hansel" 30000 10 5
def gretel := Person.mk "Gretel" 30000 15 4
def rapunzel := Person.mk "Rapunzel" 40000 8 6
def rumpelstiltskin := Person.mk "Rumpelstiltskin" 35000 12 7
def cinderella := Person.mk "Cinderella" 45000 7 8
def jack := Person.mk "Jack" 50000 6 10

def salary_increase (p : Person) : ℕ := p.salary * p.raise_percent / 100
def investment_return (p : Person) : ℕ := salary_increase p * p.investment_return / 100
def total_increase  (p : Person) : ℕ := salary_increase p + investment_return p

def problem_statement : Prop :=
  let hansel_increase := total_increase hansel
  let gretel_increase := total_increase gretel
  let rapunzel_increase := total_increase rapunzel
  let rumpelstiltskin_increase := total_increase rumpelstiltskin
  let cinderella_increase := total_increase cinderella
  let jack_increase := total_increase jack

  let highest_increase := max gretel_increase (max rumpelstiltskin_increase (max cinderella_increase (max rapunzel_increase (max jack_increase hansel_increase))))
  let lowest_increase := min gretel_increase (min rumpelstiltskin_increase (min cinderella_increase (min rapunzel_increase (min jack_increase hansel_increase))))

  highest_increase - lowest_increase = 1530

theorem salary_increase_difference : problem_statement := by
  sorry

end salary_increase_difference_l185_185186


namespace football_throwing_distance_l185_185583

theorem football_throwing_distance 
  (T : ℝ)
  (yards_per_throw_at_T : ℝ)
  (yards_per_throw_at_80 : ℝ)
  (throws_on_Saturday : ℕ)
  (throws_on_Sunday : ℕ)
  (saturday_distance sunday_distance : ℝ)
  (total_distance : ℝ) :
  yards_per_throw_at_T = 20 →
  yards_per_throw_at_80 = 40 →
  throws_on_Saturday = 20 →
  throws_on_Sunday = 30 →
  saturday_distance = throws_on_Saturday * yards_per_throw_at_T →
  sunday_distance = throws_on_Sunday * yards_per_throw_at_80 →
  total_distance = saturday_distance + sunday_distance →
  total_distance = 1600 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end football_throwing_distance_l185_185583


namespace eggs_in_second_tree_l185_185856

theorem eggs_in_second_tree
  (nests_in_first_tree : ℕ)
  (eggs_per_nest : ℕ)
  (eggs_in_front_yard : ℕ)
  (total_eggs : ℕ)
  (eggs_in_second_tree : ℕ)
  (h1 : nests_in_first_tree = 2)
  (h2 : eggs_per_nest = 5)
  (h3 : eggs_in_front_yard = 4)
  (h4 : total_eggs = 17)
  (h5 : nests_in_first_tree * eggs_per_nest + eggs_in_front_yard + eggs_in_second_tree = total_eggs) :
  eggs_in_second_tree = 3 :=
sorry

end eggs_in_second_tree_l185_185856


namespace christine_amount_l185_185737

theorem christine_amount (S C : ℕ) 
  (h1 : S + C = 50)
  (h2 : C = S + 30) :
  C = 40 :=
by
  -- Proof goes here.
  -- This part should be filled in to complete the proof.
  sorry

end christine_amount_l185_185737


namespace aba_div_by_7_l185_185011

theorem aba_div_by_7 (a b : ℕ) (h : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := 
sorry

end aba_div_by_7_l185_185011


namespace angle_symmetry_l185_185765

theorem angle_symmetry (α β : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) (hβ : 0 < β ∧ β < 2 * Real.pi) (h_symm : α = 2 * Real.pi - β) : α + β = 2 * Real.pi := 
by 
  sorry

end angle_symmetry_l185_185765


namespace jonah_walked_8_miles_l185_185580

def speed : ℝ := 4
def time : ℝ := 2
def distance (s t : ℝ) : ℝ := s * t

theorem jonah_walked_8_miles : distance speed time = 8 := sorry

end jonah_walked_8_miles_l185_185580


namespace construct_pairwise_tangent_circles_l185_185332

-- Define the three points A, B, and C in a 2D plane.
variables (A B C : EuclideanSpace ℝ (Fin 2))

/--
  Given three points A, B, and C in the plane, 
  it is possible to construct three circles that are pairwise tangent at these points.
-/
theorem construct_pairwise_tangent_circles (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃ (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) (r1 r2 r3 : ℝ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    dist O1 O2 = r1 + r2 ∧
    dist O2 O3 = r2 + r3 ∧
    dist O3 O1 = r3 + r1 ∧
    dist O1 A = r1 ∧ dist O2 B = r2 ∧ dist O3 C = r3 :=
sorry

end construct_pairwise_tangent_circles_l185_185332


namespace div_by_squares_l185_185266

variables {R : Type*} [CommRing R] (a b c x y z : R)

theorem div_by_squares (a b c x y z : R) :
  (a * y - b * x) ^ 2 + (b * z - c * y) ^ 2 + (c * x - a * z) ^ 2 + (a * x + b * y + c * z) ^ 2 =
    (a ^ 2 + b ^ 2 + c ^ 2) * (x ^ 2 + y ^ 2 + z ^ 2) := sorry

end div_by_squares_l185_185266


namespace compare_abc_l185_185841

noncomputable def a := Real.exp (Real.sqrt 2)
noncomputable def b := 2 + Real.sqrt 2
noncomputable def c := Real.log (12 + 6 * Real.sqrt 2)

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end compare_abc_l185_185841


namespace log_expression_equals_l185_185954

noncomputable def expression (x y : ℝ) : ℝ :=
  (Real.log x^2) / (Real.log y^10) *
  (Real.log y^3) / (Real.log x^7) *
  (Real.log x^4) / (Real.log y^8) *
  (Real.log y^6) / (Real.log x^9) *
  (Real.log x^11) / (Real.log y^5)

theorem log_expression_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  expression x y = (1 / 15) * Real.log y / Real.log x :=
sorry

end log_expression_equals_l185_185954


namespace flagstaff_height_l185_185399

theorem flagstaff_height 
  (s1 : ℝ) (s2 : ℝ) (hb : ℝ) (h : ℝ)
  (H1 : s1 = 40.25) (H2 : s2 = 28.75) (H3 : hb = 12.5) 
  (H4 : h / s1 = hb / s2) : 
  h = 17.5 :=
by
  sorry

end flagstaff_height_l185_185399


namespace inequality_one_inequality_system_l185_185647

-- Definition for the first problem
theorem inequality_one (x : ℝ) : 3 * x > 2 * (1 - x) ↔ x > 2 / 5 :=
by
  sorry

-- Definitions for the second problem
theorem inequality_system (x : ℝ) : 
  (3 * x - 7) / 2 ≤ x - 2 ∧ 4 * (x - 1) > 4 ↔ 2 < x ∧ x ≤ 3 :=
by
  sorry

end inequality_one_inequality_system_l185_185647


namespace mary_initially_selected_10_l185_185277

-- Definitions based on the conditions
def price_apple := 40
def price_orange := 60
def avg_price_initial := 54
def avg_price_after_putting_back := 48
def num_oranges_put_back := 5

-- Definition of Mary_initially_selected as the total number of pieces of fruit initially selected by Mary
def Mary_initially_selected (A O : ℕ) := A + O

-- Theorem statement
theorem mary_initially_selected_10 (A O : ℕ) 
  (h1 : (price_apple * A + price_orange * O) / (A + O) = avg_price_initial)
  (h2 : (price_apple * A + price_orange * (O - num_oranges_put_back)) / (A + O - num_oranges_put_back) = avg_price_after_putting_back) : 
  Mary_initially_selected A O = 10 := 
sorry

end mary_initially_selected_10_l185_185277


namespace max_profit_at_90_l185_185562

-- Definitions for conditions
def fixed_cost : ℝ := 5
def price_per_unit : ℝ := 100

noncomputable def variable_cost (x : ℕ) : ℝ :=
  if h : x < 80 then
    0.5 * x^2 + 40 * x
  else
    101 * x + 8100 / x - 2180

-- Definition of the profit function
noncomputable def profit (x : ℕ) : ℝ :=
  if h : x < 80 then
    -0.5 * x^2 + 60 * x - fixed_cost
  else
    1680 - x - 8100 / x

-- Maximum profit occurs at x = 90
theorem max_profit_at_90 : ∀ x : ℕ, profit 90 ≥ profit x := 
by {
  sorry
}

end max_profit_at_90_l185_185562


namespace trig_identity_l185_185459

open Real

theorem trig_identity : sin (20 * π / 180) * cos (10 * π / 180) - cos (160 * π / 180) * sin (170 * π / 180) = 1 / 2 := 
by
  sorry

end trig_identity_l185_185459


namespace binary_digit_one_l185_185109
-- We import the necessary libraries

-- Define the problem and prove the statement as follows
def fractional_part_in_binary (x : ℝ) : ℕ → ℕ := sorry

def sqrt_fractional_binary (k : ℕ) (i : ℕ) : ℕ :=
  fractional_part_in_binary (Real.sqrt ((k : ℝ) * (k + 1))) i

theorem binary_digit_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  ∃ i, n + 1 ≤ i ∧ i ≤ 2 * n + 1 ∧ sqrt_fractional_binary k i = 1 :=
sorry

end binary_digit_one_l185_185109


namespace no_tiling_triminos_l185_185914

theorem no_tiling_triminos (board_size : ℕ) (trimino_size : ℕ) (remaining_squares : ℕ) 
  (H_board : board_size = 8) (H_trimino : trimino_size = 3) (H_remaining : remaining_squares = 63) : 
  ¬ ∃ (triminos : ℕ), triminos * trimino_size = remaining_squares :=
by {
  sorry
}

end no_tiling_triminos_l185_185914


namespace round_robin_tournament_l185_185507

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end round_robin_tournament_l185_185507
