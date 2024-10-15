import Mathlib

namespace NUMINAMATH_GPT_max_squared_sum_of_sides_l1735_173518

variable {R : ℝ}
variable {O A B C : EucSpace} -- O is the center, A, B, and C are vertices
variable (a b c : ℝ)  -- Position vectors corresponding to vertices A, B, C

-- Hypotheses based on the problem conditions:
variable (h1 : ‖a‖ = R)
variable (h2 : ‖b‖ = R)
variable (h3 : ‖c‖ = R)
variable (hSumZero : a + b + c = 0)

theorem max_squared_sum_of_sides 
  {AB BC CA : ℝ} -- Side lengths
  (hAB : AB = ‖a - b‖)
  (hBC : BC = ‖b - c‖)
  (hCA : CA = ‖c - a‖) :
  AB^2 + BC^2 + CA^2 = 9 * R^2 :=
sorry

end NUMINAMATH_GPT_max_squared_sum_of_sides_l1735_173518


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1735_173532

-- Define the expression
def expression (x : ℝ) := -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2)

-- State the theorem
theorem simplify_and_evaluate : expression (-2) = -10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1735_173532


namespace NUMINAMATH_GPT_f_satisfies_condition_l1735_173538

noncomputable def f (x : ℝ) : ℝ := 2^x

-- Prove that f(x + 1) = 2 * f(x) for the defined function f.
theorem f_satisfies_condition (x : ℝ) : f (x + 1) = 2 * f x := by
  show 2^(x + 1) = 2 * 2^x
  sorry

end NUMINAMATH_GPT_f_satisfies_condition_l1735_173538


namespace NUMINAMATH_GPT_full_price_tickets_revenue_l1735_173557

-- Define the conditions and then prove the statement
theorem full_price_tickets_revenue (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (p / 3) = 3000) : f * p = 1500 := by
  sorry

end NUMINAMATH_GPT_full_price_tickets_revenue_l1735_173557


namespace NUMINAMATH_GPT_find_expression_roots_l1735_173577

-- Define the roots of the given quadratic equation
def is_root (α : ℝ) : Prop := α ^ 2 - 2 * α - 1 = 0

-- Define the main statement to be proven
theorem find_expression_roots (α β : ℝ) (hα : is_root α) (hβ : is_root β) :
  5 * α ^ 4 + 12 * β ^ 3 = 169 := sorry

end NUMINAMATH_GPT_find_expression_roots_l1735_173577


namespace NUMINAMATH_GPT_smallest_number_l1735_173546

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l1735_173546


namespace NUMINAMATH_GPT_magic_box_problem_l1735_173504

theorem magic_box_problem (m : ℝ) :
  (m^2 - 2*m - 1 = 2) → (m = 3 ∨ m = -1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_magic_box_problem_l1735_173504


namespace NUMINAMATH_GPT_simplify_4sqrt2_minus_sqrt2_l1735_173568

/-- Prove that 4 * sqrt 2 - sqrt 2 = 3 * sqrt 2 given standard mathematical rules -/
theorem simplify_4sqrt2_minus_sqrt2 : 4 * Real.sqrt 2 - Real.sqrt 2 = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_simplify_4sqrt2_minus_sqrt2_l1735_173568


namespace NUMINAMATH_GPT_school_student_count_l1735_173597

-- Definition of the conditions
def students_in_school (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧
  n % 8 = 2 ∧
  n % 9 = 3

-- The main proof statement
theorem school_student_count : ∃ n, students_in_school n ∧ n = 265 :=
by
  sorry  -- Proof would go here

end NUMINAMATH_GPT_school_student_count_l1735_173597


namespace NUMINAMATH_GPT_function_is_zero_l1735_173569

-- Define the condition that for any three points A, B, and C forming an equilateral triangle,
-- the sum of their function values is zero.
def has_equilateral_property (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ (A B C : ℝ × ℝ), dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1 → 
  f A + f B + f C = 0

-- Define the theorem that states that a function with the equilateral property is identically zero.
theorem function_is_zero {f : ℝ × ℝ → ℝ} (h : has_equilateral_property f) : 
  ∀ (x : ℝ × ℝ), f x = 0 := 
by
  sorry

end NUMINAMATH_GPT_function_is_zero_l1735_173569


namespace NUMINAMATH_GPT_inequality_px_qy_l1735_173566

theorem inequality_px_qy 
  (p q x y : ℝ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hpq : p + q < 1) 
  : (p * x + q * y) ^ 2 ≤ p * x ^ 2 + q * y ^ 2 := 
sorry

end NUMINAMATH_GPT_inequality_px_qy_l1735_173566


namespace NUMINAMATH_GPT_outfit_combinations_l1735_173515

theorem outfit_combinations :
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6 -- 5 ties + no tie option
  let num_belts := 3 -- 2 belts + no belt option
  num_shirts * num_pants * num_ties * num_belts = 360 :=
by
  let num_shirts := 5
  let num_pants := 4
  let num_ties := 6
  let num_belts := 3
  show num_shirts * num_pants * num_ties * num_belts = 360
  sorry

end NUMINAMATH_GPT_outfit_combinations_l1735_173515


namespace NUMINAMATH_GPT_mixture_correct_l1735_173584

def water_amount : ℚ := (3/5) * 20
def vinegar_amount : ℚ := (5/6) * 18
def mixture_amount : ℚ := water_amount + vinegar_amount

theorem mixture_correct : mixture_amount = 27 := 
by
  -- Here goes the proof steps
  sorry

end NUMINAMATH_GPT_mixture_correct_l1735_173584


namespace NUMINAMATH_GPT_longest_segment_in_cylinder_l1735_173571

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end NUMINAMATH_GPT_longest_segment_in_cylinder_l1735_173571


namespace NUMINAMATH_GPT_number_of_foxes_l1735_173539

-- Define the conditions as given in the problem
def num_cows : ℕ := 20
def num_sheep : ℕ := 20
def total_animals : ℕ := 100
def num_zebras (F : ℕ) := 3 * F

-- The theorem we want to prove based on the conditions
theorem number_of_foxes (F : ℕ) :
  num_cows + num_sheep + F + num_zebras F = total_animals → F = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_foxes_l1735_173539


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l1735_173528

theorem triangle_angle_contradiction (A B C : ℝ) (h_sum : A + B + C = 180) (h_lt_60 : A < 60 ∧ B < 60 ∧ C < 60) : false := 
sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l1735_173528


namespace NUMINAMATH_GPT_probability_uniform_same_color_l1735_173554

noncomputable def probability_same_color (choices : List String) (athleteA: ℕ) (athleteB: ℕ) : ℚ :=
  if choices.length = 3 ∧ athleteA ∈ [0,1,2] ∧ athleteB ∈ [0,1,2] then
    1 / 3
  else
    0

theorem probability_uniform_same_color :
  probability_same_color ["red", "white", "blue"] 0 1 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_uniform_same_color_l1735_173554


namespace NUMINAMATH_GPT_interval_between_births_l1735_173535

def youngest_child_age : ℕ := 6

def sum_of_ages (I : ℝ) : ℝ :=
  youngest_child_age + (youngest_child_age + I) + (youngest_child_age + 2 * I) + (youngest_child_age + 3 * I) + (youngest_child_age + 4 * I)

theorem interval_between_births : ∃ (I : ℝ), sum_of_ages I = 60 ∧ I = 3.6 := 
by
  sorry

end NUMINAMATH_GPT_interval_between_births_l1735_173535


namespace NUMINAMATH_GPT_arrangements_no_adjacent_dances_arrangements_alternating_order_l1735_173595

-- Part (1)
theorem arrangements_no_adjacent_dances (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 43200 := 
by sorry

-- Part (2)
theorem arrangements_alternating_order (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 2880 := 
by sorry

end NUMINAMATH_GPT_arrangements_no_adjacent_dances_arrangements_alternating_order_l1735_173595


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1735_173545

theorem inscribed_circle_radius (R : ℝ) (h : 0 < R) : 
  ∃ x : ℝ, (x = R / 3) :=
by
  -- Given conditions
  have h1 : R > 0 := h

  -- Mathematical proof statement derived from conditions
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1735_173545


namespace NUMINAMATH_GPT_smallest_b_in_AP_l1735_173527

theorem smallest_b_in_AP (a b c : ℝ) (d : ℝ) (ha : a = b - d) (hc : c = b + d) (habc : a * b * c = 125) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) : 
    b = 5 :=
by
  -- Proof needed here
  sorry

end NUMINAMATH_GPT_smallest_b_in_AP_l1735_173527


namespace NUMINAMATH_GPT_find_a_and_b_l1735_173552

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

theorem find_a_and_b :
  (∃ a b : ℝ, a ≠ 0 ∧
   (∀ x, -1 ≤ x ∧ x ≤ 2 → f a b x ≤ 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = -29)
  ) → ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
sorry

end NUMINAMATH_GPT_find_a_and_b_l1735_173552


namespace NUMINAMATH_GPT_tom_final_amount_l1735_173534

-- Conditions and definitions from the problem
def initial_amount : ℝ := 74
def spent_percentage : ℝ := 0.15
def earnings : ℝ := 86
def share_percentage : ℝ := 0.60

-- Lean proof statement
theorem tom_final_amount :
  (initial_amount - (spent_percentage * initial_amount)) + (share_percentage * earnings) = 114.5 :=
by
  sorry

end NUMINAMATH_GPT_tom_final_amount_l1735_173534


namespace NUMINAMATH_GPT_age_of_30th_employee_l1735_173563

theorem age_of_30th_employee :
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  age_30th_employee = 25 :=
by
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  have h : age_30th_employee = 25 := sorry
  exact h

end NUMINAMATH_GPT_age_of_30th_employee_l1735_173563


namespace NUMINAMATH_GPT_polygon_area_is_1008_l1735_173589

variables (vertices : List (ℕ × ℕ)) (units : ℕ)

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
sorry -- The function would compute the area based on vertices.

theorem polygon_area_is_1008 :
  vertices = [(0, 0), (12, 0), (24, 12), (24, 0), (36, 0), (36, 24), (24, 36), (12, 36), (0, 36), (0, 24), (0, 0)] →
  units = 1 →
  polygon_area vertices = 1008 :=
sorry

end NUMINAMATH_GPT_polygon_area_is_1008_l1735_173589


namespace NUMINAMATH_GPT_rectangle_length_is_16_l1735_173506

-- Define the conditions
def side_length_square : ℕ := 8
def width_rectangle : ℕ := 4
def area_square : ℕ := side_length_square ^ 2  -- Area of the square
def area_rectangle (length : ℕ) : ℕ := width_rectangle * length  -- Area of the rectangle

-- Lean 4 statement
theorem rectangle_length_is_16 (L : ℕ) (h : area_square = area_rectangle L) : L = 16 :=
by
  /- Proof will be inserted here -/
  sorry

end NUMINAMATH_GPT_rectangle_length_is_16_l1735_173506


namespace NUMINAMATH_GPT_surface_area_of_sphere_l1735_173505

-- Define the conditions from the problem.

variables (r R : ℝ) -- r is the radius of the cross-section, R is the radius of the sphere.
variables (π : ℝ := Real.pi) -- Define π using the real pi constant.
variables (h_dist : 1 = 1) -- Distance from the plane to the center is 1 unit.
variables (h_area_cross_section : π = π * r^2) -- Area of the cross-section is π.

-- State to prove the surface area of the sphere is 8π.
theorem surface_area_of_sphere :
    ∃ (R : ℝ), (R^2 = 2) → (4 * π * R^ 2 = 8 * π) := sorry

end NUMINAMATH_GPT_surface_area_of_sphere_l1735_173505


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1735_173562

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ ∃ x : ℝ, |x - 1| - |x + 1| > 3 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1735_173562


namespace NUMINAMATH_GPT_proposition_true_iff_l1735_173537

theorem proposition_true_iff :
  (∀ x y : ℝ, (xy = 1 → x = 1 / y ∧ y = 1 / x) → (x = 1 / y ∧ y = 1 / x → xy = 1)) ∧
  (∀ (A B : Set ℝ), (A ∩ B = B → A ⊆ B) → (A ⊆ B → A ∩ B = B)) ∧
  (∀ m : ℝ, (m > 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0) → (¬(∃ x : ℝ, x^2 - 2 * x + m = 0) → m ≤ 1)) :=
by
  sorry

end NUMINAMATH_GPT_proposition_true_iff_l1735_173537


namespace NUMINAMATH_GPT_month_days_l1735_173502

theorem month_days (letters_per_day packages_per_day total_mail six_months : ℕ) (h1 : letters_per_day = 60) (h2 : packages_per_day = 20) (h3 : total_mail = 14400) (h4 : six_months = 6) : 
  total_mail / (letters_per_day + packages_per_day) / six_months = 30 :=
by sorry

end NUMINAMATH_GPT_month_days_l1735_173502


namespace NUMINAMATH_GPT_A_wins_one_prob_A_wins_at_least_2_of_3_prob_l1735_173599

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Definition of the independent events for A and B
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- The probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * prob_B_incorrect

-- Proof (statement) that A's probability of winning one activity is 1/3
theorem A_wins_one_prob :
  prob_A_wins_one = 1/3 :=
sorry

-- Binomial coefficient for choosing 2 wins out of 3 activities
def binom_coeff_n_2 (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Probability of A winning exactly 2 out of 3 activities
def prob_A_wins_exactly_2_of_3 : ℚ :=
  binom_coeff_n_2 3 2 * prob_A_wins_one^2 * (1 - prob_A_wins_one)

-- Probability of A winning all 3 activities
def prob_A_wins_all_3 : ℚ :=
  prob_A_wins_one^3

-- The probability of A winning at least 2 out of 3 activities
def prob_A_wins_at_least_2_of_3 : ℚ :=
  prob_A_wins_exactly_2_of_3 + prob_A_wins_all_3

-- Proof (statement) that A's probability of winning at least 2 out of 3 activities is 7/27
theorem A_wins_at_least_2_of_3_prob :
  prob_A_wins_at_least_2_of_3 = 7/27 :=
sorry

end NUMINAMATH_GPT_A_wins_one_prob_A_wins_at_least_2_of_3_prob_l1735_173599


namespace NUMINAMATH_GPT_discount_difference_l1735_173524

-- Definitions based on given conditions
def original_bill : ℝ := 8000
def single_discount_rate : ℝ := 0.30
def first_successive_discount_rate : ℝ := 0.26
def second_successive_discount_rate : ℝ := 0.05

-- Calculations based on conditions
def single_discount_final_amount := original_bill * (1 - single_discount_rate)
def first_successive_discount_final_amount := original_bill * (1 - first_successive_discount_rate)
def complete_successive_discount_final_amount := 
  first_successive_discount_final_amount * (1 - second_successive_discount_rate)

-- Proof statement
theorem discount_difference :
  single_discount_final_amount - complete_successive_discount_final_amount = 24 := 
  by
    -- Proof to be provided
    sorry

end NUMINAMATH_GPT_discount_difference_l1735_173524


namespace NUMINAMATH_GPT_infinite_series_sum_l1735_173543

noncomputable def S : ℝ :=
∑' n, (if n % 3 == 0 then 1 / (3 ^ (n / 3)) else if n % 3 == 1 then -1 / (3 ^ (n / 3 + 1)) else -1 / (3 ^ (n / 3 + 2)))

theorem infinite_series_sum : S = 15 / 26 := by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1735_173543


namespace NUMINAMATH_GPT_find_A_l1735_173555

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 100 * A + 78 - (200 + B) = 364) : A = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l1735_173555


namespace NUMINAMATH_GPT_shadow_area_l1735_173536

theorem shadow_area (y : ℝ) (cube_side : ℝ) (shadow_excl_area : ℝ) 
  (h₁ : cube_side = 2) 
  (h₂ : shadow_excl_area = 200)
  (h₃ : ((14.28 - 2) / 2 = y)) :
  ⌊1000 * y⌋ = 6140 :=
by
  sorry

end NUMINAMATH_GPT_shadow_area_l1735_173536


namespace NUMINAMATH_GPT_faster_train_speed_l1735_173592

theorem faster_train_speed (V_s : ℝ) (t : ℝ) (l : ℝ) (V_f : ℝ) : 
  V_s = 36 → t = 20 → l = 200 → V_f = V_s + (l / t) * 3.6 → V_f = 72 
  := by
    intros _ _ _ _
    sorry

end NUMINAMATH_GPT_faster_train_speed_l1735_173592


namespace NUMINAMATH_GPT_find_number_l1735_173541

theorem find_number (x : ℝ) : 
  (x + 72 = (2 * x) / (2 / 3)) → x = 36 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l1735_173541


namespace NUMINAMATH_GPT_find_smallest_number_l1735_173585

theorem find_smallest_number (x y n a : ℕ) (h1 : x + y = 2014) (h2 : 3 * n = y + 6) (h3 : x = 100 * n + a) (ha : a < 100) : min x y = 51 :=
sorry

end NUMINAMATH_GPT_find_smallest_number_l1735_173585


namespace NUMINAMATH_GPT_simplify_expression_l1735_173574

variable (x : ℝ)

theorem simplify_expression :
  2 * x - 3 * (2 - x) + 4 * (1 + 3 * x) - 5 * (1 - x^2) = -5 * x^2 + 17 * x - 7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1735_173574


namespace NUMINAMATH_GPT_num_triples_l1735_173581

/-- Theorem statement:
There are exactly 2 triples of positive integers (a, b, c) satisfying the conditions:
1. ab + ac = 60
2. bc + ac = 36
3. ab + bc = 48
--/
theorem num_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (ab + ac = 60) → (bc + ac = 36) → (ab + bc = 48) → 
  (a, b, c) ∈ [(1, 4, 8), (1, 12, 3)] →
  ∃! (a b c : ℕ), (ab + ac = 60) ∧ (bc + ac = 36) ∧ (ab + bc = 48) :=
sorry

end NUMINAMATH_GPT_num_triples_l1735_173581


namespace NUMINAMATH_GPT_afternoon_sales_l1735_173570

theorem afternoon_sales :
  ∀ (morning_sold afternoon_sold total_sold : ℕ),
    afternoon_sold = 2 * morning_sold ∧
    total_sold = morning_sold + afternoon_sold ∧
    total_sold = 510 →
    afternoon_sold = 340 :=
by
  intros morning_sold afternoon_sold total_sold h
  sorry

end NUMINAMATH_GPT_afternoon_sales_l1735_173570


namespace NUMINAMATH_GPT_factorial_expression_l1735_173544

namespace FactorialProblem

-- Definition of factorial function.
def factorial : ℕ → ℕ 
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Theorem stating the problem equivalently.
theorem factorial_expression : (factorial 12 - factorial 10) / factorial 8 = 11790 := by
  sorry

end FactorialProblem

end NUMINAMATH_GPT_factorial_expression_l1735_173544


namespace NUMINAMATH_GPT_woman_lawyer_probability_l1735_173551

-- Defining conditions
def total_members : ℝ := 100
def percent_women : ℝ := 0.90
def percent_women_lawyers : ℝ := 0.60

-- Calculating numbers based on the percentages
def number_women : ℝ := percent_women * total_members
def number_women_lawyers : ℝ := percent_women_lawyers * number_women

-- Statement of the problem in Lean 4
theorem woman_lawyer_probability :
  (number_women_lawyers / total_members) = 0.54 :=
by sorry

end NUMINAMATH_GPT_woman_lawyer_probability_l1735_173551


namespace NUMINAMATH_GPT_five_times_seven_divided_by_ten_l1735_173598

theorem five_times_seven_divided_by_ten : (5 * 7 : ℝ) / 10 = 3.5 := 
by 
  sorry

end NUMINAMATH_GPT_five_times_seven_divided_by_ten_l1735_173598


namespace NUMINAMATH_GPT_inequality_proof_l1735_173572

theorem inequality_proof 
  (a b c x y z : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h_sum : a + b + c = 1) : 
  (x^2 + y^2 + z^2) * 
  (a^3 / (x^2 + 2 * y^2) + b^3 / (y^2 + 2 * z^2) + c^3 / (z^2 + 2 * x^2)) 
  ≥ 1 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1735_173572


namespace NUMINAMATH_GPT_fraction_position_1991_1949_l1735_173516

theorem fraction_position_1991_1949 :
  ∃ (row position : ℕ), 
    ∀ (i j : ℕ), 
      (∃ k : ℕ, k = i + j - 1 ∧ k = 3939) ∧
      (∃ p : ℕ, p = j ∧ p = 1949) → 
      row = 3939 ∧ position = 1949 := 
sorry

end NUMINAMATH_GPT_fraction_position_1991_1949_l1735_173516


namespace NUMINAMATH_GPT_part1_part2_part3_l1735_173540

noncomputable def seq (a : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else (1 - a) / n

theorem part1 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (a1_eq : seq a 1 = 1 / 2) (a2_eq : seq a 2 = 1 / 4) : true :=
by trivial

theorem part2 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : 0 < seq a n ∧ seq a n < 1 :=
sorry

theorem part3 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : seq a n > seq a (n + 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1735_173540


namespace NUMINAMATH_GPT_quadratic_expression_representation_quadratic_expression_integer_iff_l1735_173531

theorem quadratic_expression_representation (A B C : ℤ) :
  ∃ (k l m : ℤ), 
    (k = 2 * A) ∧ 
    (l = A + B) ∧ 
    (m = C) ∧ 
    (∀ x : ℤ, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) := 
sorry

theorem quadratic_expression_integer_iff (A B C : ℤ) :
  (∀ x : ℤ, ∃ k l m : ℤ, (k = 2 * A) ∧ (l = A + B) ∧ (m = C) ∧ (A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m)) ↔ 
  (A % 1 = 0 ∧ B % 1 = 0 ∧ C % 1 = 0) := 
sorry

end NUMINAMATH_GPT_quadratic_expression_representation_quadratic_expression_integer_iff_l1735_173531


namespace NUMINAMATH_GPT_cubic_equation_real_root_l1735_173508

theorem cubic_equation_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x + 25 = 0 := 
sorry

end NUMINAMATH_GPT_cubic_equation_real_root_l1735_173508


namespace NUMINAMATH_GPT_not_speaking_hindi_is_32_l1735_173512

-- Definitions and conditions
def total_diplomats : ℕ := 120
def spoke_french : ℕ := 20
def percent_neither : ℝ := 0.20
def percent_both : ℝ := 0.10

-- Number of diplomats who spoke neither French nor Hindi
def neither_french_nor_hindi := (percent_neither * total_diplomats : ℝ)

-- Number of diplomats who spoke both French and Hindi
def both_french_and_hindi := (percent_both * total_diplomats : ℝ)

-- Number of diplomats who spoke only French
def only_french := (spoke_french - both_french_and_hindi : ℝ)

-- Number of diplomats who did not speak Hindi
def not_speaking_hindi := (only_french + neither_french_nor_hindi : ℝ)

theorem not_speaking_hindi_is_32 :
  not_speaking_hindi = 32 :=
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_not_speaking_hindi_is_32_l1735_173512


namespace NUMINAMATH_GPT_equation_of_line_l_l1735_173564

theorem equation_of_line_l
  (a : ℝ)
  (l_intersects_circle : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + a = 0)
  (midpoint_chord : ∃ C : ℝ × ℝ, C = (-2, 3) ∧ ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + B.1) / 2 = C.1 ∧ (A.2 + B.2) / 2 = C.2) :
  a < 3 →
  ∃ l : ℝ × ℝ → Prop, (∀ x y : ℝ, l (x, y) ↔ x - y + 5 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_equation_of_line_l_l1735_173564


namespace NUMINAMATH_GPT_fifth_number_in_ninth_row_l1735_173517

theorem fifth_number_in_ninth_row :
  ∃ (n : ℕ), n = 61 ∧ ∀ (i : ℕ), i = 9 → (7 * i - 2 = n) :=
by
  sorry

end NUMINAMATH_GPT_fifth_number_in_ninth_row_l1735_173517


namespace NUMINAMATH_GPT_coordinates_of_E_l1735_173522

theorem coordinates_of_E :
  let A := (-2, 1)
  let B := (1, 4)
  let C := (4, -3)
  let ratio_AB := (1, 2)
  let ratio_CE_ED := (1, 4)
  let D := ( (ratio_AB.1 * B.1 + ratio_AB.2 * A.1) / (ratio_AB.1 + ratio_AB.2),
             (ratio_AB.1 * B.2 + ratio_AB.2 * A.2) / (ratio_AB.1 + ratio_AB.2) )
  let E := ( (ratio_CE_ED.1 * C.1 - ratio_CE_ED.2 * D.1) / (ratio_CE_ED.1 - ratio_CE_ED.2),
             (ratio_CE_ED.1 * C.2 - ratio_CE_ED.2 * D.2) / (ratio_CE_ED.1 - ratio_CE_ED.2) )
  E = (-8 / 3, 11 / 3) := by
  sorry

end NUMINAMATH_GPT_coordinates_of_E_l1735_173522


namespace NUMINAMATH_GPT_year_population_below_five_percent_l1735_173582

def population (P0 : ℕ) (years : ℕ) : ℕ :=
  P0 / 2^years

theorem year_population_below_five_percent (P0 : ℕ) :
  ∃ n, population P0 n < P0 / 20 ∧ (2005 + n) = 2010 := 
by {
  sorry
}

end NUMINAMATH_GPT_year_population_below_five_percent_l1735_173582


namespace NUMINAMATH_GPT_minimum_value_expression_l1735_173567

open Real

theorem minimum_value_expression (α β : ℝ) :
  ∃ x y : ℝ, x = 3 * cos α + 4 * sin β ∧ y = 3 * sin α + 4 * cos β ∧
    ((x - 7) ^ 2 + (y - 12) ^ 2) = 242 - 14 * sqrt 193 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l1735_173567


namespace NUMINAMATH_GPT_find_d_squared_plus_e_squared_l1735_173553

theorem find_d_squared_plus_e_squared {a b c d e : ℕ} 
  (h1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (h2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (h3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1)
  : d ^ 2 + e ^ 2 = 146 := 
sorry

end NUMINAMATH_GPT_find_d_squared_plus_e_squared_l1735_173553


namespace NUMINAMATH_GPT_probability_of_Z_l1735_173513

/-
  Given: 
  - P(W) = 3 / 8
  - P(X) = 1 / 4
  - P(Y) = 1 / 8

  Prove: 
  - P(Z) = 1 / 4 when P(Z) = 1 - (P(W) + P(X) + P(Y))
-/

theorem probability_of_Z (P_W P_X P_Y P_Z : ℚ) (h_W : P_W = 3 / 8) (h_X : P_X = 1 / 4) (h_Y : P_Y = 1 / 8) (h_Z : P_Z = 1 - (P_W + P_X + P_Y)) : 
  P_Z = 1 / 4 :=
by
  -- We can write the whole Lean Math proof here. However, per the instructions, we'll conclude with sorry.
  sorry

end NUMINAMATH_GPT_probability_of_Z_l1735_173513


namespace NUMINAMATH_GPT_solve_equation_l1735_173514

theorem solve_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x * (x - 1)) = 0) ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1735_173514


namespace NUMINAMATH_GPT_point_translation_proof_l1735_173519

def Point := (ℝ × ℝ)

def translate_right (p : Point) (d : ℝ) : Point := (p.1 + d, p.2)

theorem point_translation_proof :
  let A : Point := (1, 2)
  let A' := translate_right A 2
  A' = (3, 2) :=
by
  let A : Point := (1, 2)
  let A' := translate_right A 2
  show A' = (3, 2)
  sorry

end NUMINAMATH_GPT_point_translation_proof_l1735_173519


namespace NUMINAMATH_GPT_net_percentage_error_in_volume_l1735_173501

theorem net_percentage_error_in_volume
  (a : ℝ)
  (side_error : ℝ := 0.03)
  (height_error : ℝ := -0.04)
  (depth_error : ℝ := 0.02) :
  ((1 + side_error) * (1 + height_error) * (1 + depth_error) - 1) * 100 = 0.8656 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_net_percentage_error_in_volume_l1735_173501


namespace NUMINAMATH_GPT_correct_phrase_l1735_173542

-- Define statements representing each option
def option_A : String := "as twice much"
def option_B : String := "much as twice"
def option_C : String := "twice as much"
def option_D : String := "as much twice"

-- The correct option
def correct_option : String := "twice as much"

-- The main theorem statement
theorem correct_phrase : option_C = correct_option :=
by
  sorry

end NUMINAMATH_GPT_correct_phrase_l1735_173542


namespace NUMINAMATH_GPT_greatest_possible_x_l1735_173586

-- Define the conditions and the target proof in Lean 4
theorem greatest_possible_x 
  (x : ℤ)  -- x is an integer
  (h : 2.134 * (10:ℝ)^x < 21000) : 
  x ≤ 3 :=
sorry

end NUMINAMATH_GPT_greatest_possible_x_l1735_173586


namespace NUMINAMATH_GPT_square_side_length_on_hexagon_l1735_173583

noncomputable def side_length_of_square (s : ℝ) : Prop :=
  let hexagon_side := 1
  let internal_angle := 120
  ((s * (1 + 1 / Real.sqrt 3)) = 2) → s = (3 - Real.sqrt 3)

theorem square_side_length_on_hexagon : ∃ s : ℝ, side_length_of_square s :=
by
  use 3 - Real.sqrt 3
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_square_side_length_on_hexagon_l1735_173583


namespace NUMINAMATH_GPT_possible_values_x_plus_y_l1735_173511

theorem possible_values_x_plus_y (x y : ℝ) (h1 : x = y * (3 - y)^2) (h2 : y = x * (3 - x)^2) :
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end NUMINAMATH_GPT_possible_values_x_plus_y_l1735_173511


namespace NUMINAMATH_GPT_statement_books_per_shelf_l1735_173510

/--
A store initially has 40.0 coloring books.
Acquires 20.0 more books.
Uses 15 shelves to store the books equally.
-/
def initial_books : ℝ := 40.0
def acquired_books : ℝ := 20.0
def total_shelves : ℝ := 15.0

/-- 
Theorem statement: The number of coloring books on each shelf.
-/
theorem books_per_shelf : (initial_books + acquired_books) / total_shelves = 4.0 := by
  sorry

end NUMINAMATH_GPT_statement_books_per_shelf_l1735_173510


namespace NUMINAMATH_GPT_arithmetic_sequence_k_value_l1735_173521

theorem arithmetic_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ)
  (S_pos : S 2016 > 0) (S_neg : S 2017 < 0)
  (H : ∀ n, |a n| ≥ |a 1009| ): k = 1009 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_value_l1735_173521


namespace NUMINAMATH_GPT_total_bins_correct_l1735_173507

def total_bins (soup vegetables pasta : ℝ) : ℝ :=
  soup + vegetables + pasta

theorem total_bins_correct : total_bins 0.12 0.12 0.5 = 0.74 :=
  by
    sorry

end NUMINAMATH_GPT_total_bins_correct_l1735_173507


namespace NUMINAMATH_GPT_ratio_of_radii_l1735_173594

-- Given conditions
variables {b a c : ℝ}
variables (h1 : π * b^2 - π * c^2 = 2 * π * a^2)
variables (h2 : c = 1.5 * a)

-- Define and prove the ratio
theorem ratio_of_radii (h1: π * b^2 - π * c^2 = 2 * π * a^2) (h2: c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 :=
sorry

end NUMINAMATH_GPT_ratio_of_radii_l1735_173594


namespace NUMINAMATH_GPT_Sandy_tokens_difference_l1735_173596

theorem Sandy_tokens_difference :
  let total_tokens : ℕ := 1000000
  let siblings : ℕ := 4
  let Sandy_tokens : ℕ := total_tokens / 2
  let sibling_tokens : ℕ := Sandy_tokens / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  sorry

end NUMINAMATH_GPT_Sandy_tokens_difference_l1735_173596


namespace NUMINAMATH_GPT_tangent_line_eq_l1735_173500

theorem tangent_line_eq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) : 
  3 * x - y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_l1735_173500


namespace NUMINAMATH_GPT_natural_numbers_pq_equal_l1735_173559

theorem natural_numbers_pq_equal (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q :=
sorry

end NUMINAMATH_GPT_natural_numbers_pq_equal_l1735_173559


namespace NUMINAMATH_GPT_bottle_count_l1735_173565

theorem bottle_count :
  ∃ N x : ℕ, 
    N = x^2 + 36 ∧ N = (x + 1)^2 + 3 :=
by 
  sorry

end NUMINAMATH_GPT_bottle_count_l1735_173565


namespace NUMINAMATH_GPT_evaluate_g_at_2_l1735_173587

def g (x : ℝ) : ℝ := x^3 + x^2 - 1

theorem evaluate_g_at_2 : g 2 = 11 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_2_l1735_173587


namespace NUMINAMATH_GPT_carnival_friends_l1735_173576

theorem carnival_friends (F : ℕ) (h1 : 865 % F ≠ 0) (h2 : 873 % F = 0) : F = 3 :=
by
  -- proof is not required
  sorry

end NUMINAMATH_GPT_carnival_friends_l1735_173576


namespace NUMINAMATH_GPT_max_area_height_l1735_173530

theorem max_area_height (h : ℝ) (x : ℝ) 
  (right_trapezoid : True) 
  (angle_30_deg : True) 
  (perimeter_eq_6 : 3 * (x + h) = 6) : 
  h = 1 :=
by 
  sorry

end NUMINAMATH_GPT_max_area_height_l1735_173530


namespace NUMINAMATH_GPT_raft_sticks_total_l1735_173525

theorem raft_sticks_total : 
  let S := 45 
  let G := (3/5 * 45 : ℝ)
  let M := 45 + G + 15
  let D := 2 * M - 7
  S + G + M + D = 326 := 
by
  sorry

end NUMINAMATH_GPT_raft_sticks_total_l1735_173525


namespace NUMINAMATH_GPT_negation_one_zero_l1735_173578

theorem negation_one_zero (a b : ℝ) (h : a ≠ 0):
  ¬ (∃! x : ℝ, a * x + b = 0) ↔ (¬ ∃ x : ℝ, a * x + b = 0 ∨ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ + b = 0 ∧ a * x₂ + b = 0) := by
sorry

end NUMINAMATH_GPT_negation_one_zero_l1735_173578


namespace NUMINAMATH_GPT_range_of_a_l1735_173550

noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → a < g x) : a < 0 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l1735_173550


namespace NUMINAMATH_GPT_units_digit_of_8_pow_120_l1735_173561

theorem units_digit_of_8_pow_120 : (8 ^ 120) % 10 = 6 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_of_8_pow_120_l1735_173561


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1735_173503

theorem sufficient_but_not_necessary_condition (a b : ℝ) : 
  (a ≥ 1 ∧ b ≥ 1) → (a + b ≥ 2) ∧ ¬((a + b ≥ 2) → (a ≥ 1 ∧ b ≥ 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1735_173503


namespace NUMINAMATH_GPT_square_area_proof_square_area_square_area_final_square_area_correct_l1735_173520

theorem square_area_proof (x : ℝ) (s1 : ℝ) (s2 : ℝ) (A : ℝ)
  (h1 : s1 = 5 * x - 20)
  (h2 : s2 = 25 - 2 * x)
  (h3 : s1 = s2) :
  A = (s1 * s1) := by
  -- We need to prove A = s1 * s1
  sorry

theorem square_area (x : ℝ) (s : ℝ) (h : s = 85 / 7) :
  s ^ 2 = 7225 / 49 := by
  -- We need to prove s^2 = 7225 / 49
  sorry

theorem square_area_final (x : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (A : ℝ) :
  A = (85 / 7) ^ 2 := by
  -- We need to prove A = (85 / 7) ^ 2
  sorry

theorem square_area_correct (x : ℝ)
  (A : ℝ)
  (h1 : 5 * x - 20 = 25 - 2 * x)
  (h2 : A = (85 / 7) ^ 2) :
  A = 7225 / 49 := by
  -- We need to prove A = 7225 / 49
  sorry

end NUMINAMATH_GPT_square_area_proof_square_area_square_area_final_square_area_correct_l1735_173520


namespace NUMINAMATH_GPT_ratio_of_square_areas_l1735_173558

theorem ratio_of_square_areas (y : ℝ) (hy : y > 0) : 
  (y^2 / (3 * y)^2) = 1 / 9 :=
sorry

end NUMINAMATH_GPT_ratio_of_square_areas_l1735_173558


namespace NUMINAMATH_GPT_total_volume_of_five_cubes_l1735_173579

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end NUMINAMATH_GPT_total_volume_of_five_cubes_l1735_173579


namespace NUMINAMATH_GPT_total_time_to_virgo_l1735_173548

def train_ride : ℝ := 5
def first_layover : ℝ := 1.5
def bus_ride : ℝ := 4
def second_layover : ℝ := 0.5
def first_flight : ℝ := 6
def third_layover : ℝ := 2
def second_flight : ℝ := 3 * bus_ride
def fourth_layover : ℝ := 3
def car_drive : ℝ := 3.5
def first_boat_ride : ℝ := 1.5
def fifth_layover : ℝ := 0.75
def second_boat_ride : ℝ := 2 * first_boat_ride - 0.5
def final_walk : ℝ := 1.25

def total_time : ℝ := train_ride + first_layover + bus_ride + second_layover + first_flight + third_layover + second_flight + fourth_layover + car_drive + first_boat_ride + fifth_layover + second_boat_ride + final_walk

theorem total_time_to_virgo : total_time = 44 := by
  simp [train_ride, first_layover, bus_ride, second_layover, first_flight, third_layover, second_flight, fourth_layover, car_drive, first_boat_ride, fifth_layover, second_boat_ride, final_walk, total_time]
  sorry

end NUMINAMATH_GPT_total_time_to_virgo_l1735_173548


namespace NUMINAMATH_GPT_original_number_is_115_l1735_173575

-- Define the original number N, the least number to be subtracted (given), and the divisor
variable (N : ℤ) (k : ℤ)

-- State the condition based on the problem's requirements
def least_number_condition := ∃ k : ℤ, N - 28 = 87 * k

-- State the proof problem: Given the condition, prove the original number
theorem original_number_is_115 (h : least_number_condition N) : N = 115 := 
by
  sorry

end NUMINAMATH_GPT_original_number_is_115_l1735_173575


namespace NUMINAMATH_GPT_equal_donations_amount_l1735_173509

def raffle_tickets_sold := 25
def cost_per_ticket := 2
def total_raised := 100
def single_donation := 20
def amount_equal_donations (D : ℕ) : Prop := 2 * D + single_donation = total_raised - (raffle_tickets_sold * cost_per_ticket)

theorem equal_donations_amount (D : ℕ) (h : amount_equal_donations D) : D = 15 :=
  sorry

end NUMINAMATH_GPT_equal_donations_amount_l1735_173509


namespace NUMINAMATH_GPT_diagonal_length_of_cuboid_l1735_173549

theorem diagonal_length_of_cuboid
  (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : c * a = Real.sqrt 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_diagonal_length_of_cuboid_l1735_173549


namespace NUMINAMATH_GPT_solve_inequality_when_a_is_one_range_of_values_for_a_l1735_173523

open Real

-- Part (1) Statement
theorem solve_inequality_when_a_is_one (a x : ℝ) (h : a = 1) : 
  |x - a| + |x + 2| ≤ 5 → -3 ≤ x ∧ x ≤ 2 := 
by sorry

-- Part (2) Statement
theorem range_of_values_for_a (a : ℝ) : 
  (∃ x_0 : ℝ, |x_0 - a| + |x_0 + 2| ≤ |2 * a + 1|) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_when_a_is_one_range_of_values_for_a_l1735_173523


namespace NUMINAMATH_GPT_problem_inequality_1_problem_inequality_2_l1735_173529

theorem problem_inequality_1 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : 0 < α ∧ α < 1) : 
  (1 + x) ^ α ≤ 1 + α * x :=
sorry

theorem problem_inequality_2 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : α < 0 ∨ α > 1) : 
  (1 + x) ^ α ≥ 1 + α * x :=
sorry

end NUMINAMATH_GPT_problem_inequality_1_problem_inequality_2_l1735_173529


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1735_173526

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h4 : ∀ n, a n ≤ a (n + 1)) :
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1735_173526


namespace NUMINAMATH_GPT_percent_equivalence_l1735_173590

variable (x : ℝ)
axiom condition : 0.30 * 0.15 * x = 18

theorem percent_equivalence :
  0.15 * 0.30 * x = 18 := sorry

end NUMINAMATH_GPT_percent_equivalence_l1735_173590


namespace NUMINAMATH_GPT_all_buses_have_same_stoppage_time_l1735_173573

-- Define the constants for speeds without and with stoppages
def speed_without_stoppage_bus1 := 50
def speed_without_stoppage_bus2 := 60
def speed_without_stoppage_bus3 := 70

def speed_with_stoppage_bus1 := 40
def speed_with_stoppage_bus2 := 48
def speed_with_stoppage_bus3 := 56

-- Stating the stoppage time per hour for each bus
def stoppage_time_per_hour (speed_without : ℕ) (speed_with : ℕ) : ℚ :=
  1 - (speed_with : ℚ) / (speed_without : ℚ)

-- Theorem to prove the stoppage time correctness
theorem all_buses_have_same_stoppage_time :
  stoppage_time_per_hour speed_without_stoppage_bus1 speed_with_stoppage_bus1 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus2 speed_with_stoppage_bus2 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus3 speed_with_stoppage_bus3 = 0.2 :=
by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_all_buses_have_same_stoppage_time_l1735_173573


namespace NUMINAMATH_GPT_decreasing_geometric_sums_implications_l1735_173591

variable (X : Type)
variable (a1 q : ℝ)
variable (S : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) :=
∀ n : ℕ, a (n + 1) = a1 * q^n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
S 0 = a 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

def is_decreasing_sequence (S : ℕ → ℝ) :=
∀ n : ℕ, S (n + 1) < S n

theorem decreasing_geometric_sums_implications (a1 q : ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 1) < S n) → a1 < 0 ∧ q > 0 := 
by 
  sorry

end NUMINAMATH_GPT_decreasing_geometric_sums_implications_l1735_173591


namespace NUMINAMATH_GPT_union_A_B_l1735_173533

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def C : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_A_B : A ∪ B = C := 
by sorry

end NUMINAMATH_GPT_union_A_B_l1735_173533


namespace NUMINAMATH_GPT_bucket_full_weight_l1735_173593

theorem bucket_full_weight (c d : ℝ) (x y : ℝ) (h1 : x + (1 / 4) * y = c) (h2 : x + (3 / 4) * y = d) : 
  x + y = (3 * d - 3 * c) / 2 :=
by
  sorry

end NUMINAMATH_GPT_bucket_full_weight_l1735_173593


namespace NUMINAMATH_GPT_distinct_numbers_in_list_l1735_173556

def count_distinct_floors (l : List ℕ) : ℕ :=
  l.eraseDups.length

def generate_list : List ℕ :=
  List.map (λ n => Nat.floor ((n * n : ℚ) / 2000)) (List.range' 1 2000)

theorem distinct_numbers_in_list : count_distinct_floors generate_list = 1501 :=
by
  sorry

end NUMINAMATH_GPT_distinct_numbers_in_list_l1735_173556


namespace NUMINAMATH_GPT_incorrect_proposition_statement_l1735_173560

theorem incorrect_proposition_statement (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := 
sorry

end NUMINAMATH_GPT_incorrect_proposition_statement_l1735_173560


namespace NUMINAMATH_GPT_stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l1735_173547

-- Definition of the speeds
def speed_excluding_stoppages_A : ℕ := 60
def speed_including_stoppages_A : ℕ := 48
def speed_excluding_stoppages_B : ℕ := 75
def speed_including_stoppages_B : ℕ := 60
def speed_excluding_stoppages_C : ℕ := 90
def speed_including_stoppages_C : ℕ := 72

-- Theorem to prove the stopped time per hour for each bus
theorem stopped_time_per_hour_A : (speed_excluding_stoppages_A - speed_including_stoppages_A) * 60 / speed_excluding_stoppages_A = 12 := sorry

theorem stopped_time_per_hour_B : (speed_excluding_stoppages_B - speed_including_stoppages_B) * 60 / speed_excluding_stoppages_B = 12 := sorry

theorem stopped_time_per_hour_C : (speed_excluding_stoppages_C - speed_including_stoppages_C) * 60 / speed_excluding_stoppages_C = 12 := sorry

end NUMINAMATH_GPT_stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l1735_173547


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1735_173580

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set_of_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | Real.sqrt 10 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1735_173580


namespace NUMINAMATH_GPT_prove_weight_of_a_l1735_173588

noncomputable def weight_proof (A B C D : ℝ) : Prop :=
  (A + B + C) / 3 = 60 ∧
  50 ≤ A ∧ A ≤ 80 ∧
  50 ≤ B ∧ B ≤ 80 ∧
  50 ≤ C ∧ C ≤ 80 ∧
  60 ≤ D ∧ D ≤ 90 ∧
  (A + B + C + D) / 4 = 65 ∧
  70 ≤ D + 3 ∧ D + 3 ≤ 100 ∧
  (B + C + D + (D + 3)) / 4 = 64 → 
  A = 87

-- Adding a theorem statement to make it clear we need to prove this.
theorem prove_weight_of_a (A B C D : ℝ) : weight_proof A B C D :=
sorry

end NUMINAMATH_GPT_prove_weight_of_a_l1735_173588
