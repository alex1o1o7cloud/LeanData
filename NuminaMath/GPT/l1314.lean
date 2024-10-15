import Mathlib

namespace NUMINAMATH_GPT_find_function_satisfaction_l1314_131410

theorem find_function_satisfaction :
  ∃ (a b : ℚ) (f : ℚ × ℚ → ℚ), (∀ (x y z : ℚ),
  f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)) ∧ 
  (∀ (x y : ℚ), f (x, y) = a * y^2 + 2 * a * x * y + b * y) := sorry

end NUMINAMATH_GPT_find_function_satisfaction_l1314_131410


namespace NUMINAMATH_GPT_eldest_child_age_l1314_131472

variable (y m e : Nat)

theorem eldest_child_age :
  (m - y = 3) →
  (e = 3 * y) →
  (e = y + m + 2) →
  (e = 15) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_eldest_child_age_l1314_131472


namespace NUMINAMATH_GPT_derivative_ln_div_x_l1314_131487

noncomputable def f (x : ℝ) := (Real.log x) / x

theorem derivative_ln_div_x (x : ℝ) (h : x ≠ 0) : deriv f x = (1 - Real.log x) / (x^2) :=
by
  sorry

end NUMINAMATH_GPT_derivative_ln_div_x_l1314_131487


namespace NUMINAMATH_GPT_ellipse_problem_l1314_131498

-- Definitions of conditions from the problem
def F1 := (0, 0)
def F2 := (6, 0)
def ellipse_equation (x y h k a b : ℝ) := ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

-- The main statement to be proved
theorem ellipse_problem :
  let h := 3
  let k := 0
  let a := 5
  let c := 3
  let b := Real.sqrt (a^2 - c^2)
  h + k + a + b = 12 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_ellipse_problem_l1314_131498


namespace NUMINAMATH_GPT_max_truthful_students_l1314_131461

def count_students (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem max_truthful_students : count_students 2015 = 2031120 :=
by sorry

end NUMINAMATH_GPT_max_truthful_students_l1314_131461


namespace NUMINAMATH_GPT_original_number_l1314_131460

theorem original_number (x : ℝ) (h1 : 268 * x = 19832) (h2 : 2.68 * x = 1.9832) : x = 74 :=
sorry

end NUMINAMATH_GPT_original_number_l1314_131460


namespace NUMINAMATH_GPT_evaluate_expression_l1314_131409

theorem evaluate_expression :
  3 * 307 + 4 * 307 + 2 * 307 + 307 * 307 = 97012 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1314_131409


namespace NUMINAMATH_GPT_mary_needs_more_sugar_l1314_131452

def recipe_sugar := 14
def sugar_already_added := 2
def sugar_needed := recipe_sugar - sugar_already_added

theorem mary_needs_more_sugar : sugar_needed = 12 := by
  sorry

end NUMINAMATH_GPT_mary_needs_more_sugar_l1314_131452


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1314_131405

theorem quadratic_no_real_roots (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - a = 0 → a < -1 :=
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1314_131405


namespace NUMINAMATH_GPT_square_fold_distance_l1314_131479

noncomputable def distance_from_A (area : ℝ) (visible_equal : Bool) : ℝ :=
  if area = 18 ∧ visible_equal then 2 * Real.sqrt 6 else 0

theorem square_fold_distance (area : ℝ) (visible_equal : Bool) :
  area = 18 → visible_equal → distance_from_A area visible_equal = 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_square_fold_distance_l1314_131479


namespace NUMINAMATH_GPT_number_of_parrots_in_each_cage_l1314_131488

theorem number_of_parrots_in_each_cage (num_cages : ℕ) (total_birds : ℕ) (parrots_per_cage parakeets_per_cage : ℕ)
    (h1 : num_cages = 9)
    (h2 : parrots_per_cage = parakeets_per_cage)
    (h3 : total_birds = 36)
    (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) :
  parrots_per_cage = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_parrots_in_each_cage_l1314_131488


namespace NUMINAMATH_GPT_calculate_final_amount_l1314_131491

def calculate_percentage (percentage : ℝ) (amount : ℝ) : ℝ :=
  percentage * amount

theorem calculate_final_amount :
  let A := 3000
  let B := 0.20
  let C := 0.35
  let D := 0.05
  D * (C * (B * A)) = 10.50 := by
    sorry

end NUMINAMATH_GPT_calculate_final_amount_l1314_131491


namespace NUMINAMATH_GPT_math_problem_l1314_131408

theorem math_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p + q + r = 0) :
    (p^2 * q^2 / ((p^2 - q * r) * (q^2 - p * r)) +
    p^2 * r^2 / ((p^2 - q * r) * (r^2 - p * q)) +
    q^2 * r^2 / ((q^2 - p * r) * (r^2 - p * q))) = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1314_131408


namespace NUMINAMATH_GPT_simplify_expression_l1314_131459

variable (x : ℝ)

theorem simplify_expression :
  2 * x * (4 * x^2 - 3 * x + 1) - 4 * (2 * x^2 - 3 * x + 5) =
  8 * x^3 - 14 * x^2 + 14 * x - 20 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1314_131459


namespace NUMINAMATH_GPT_average_percentage_decrease_l1314_131406

theorem average_percentage_decrease (x : ℝ) (h : 0 < x ∧ x < 1) :
  (800 * (1 - x)^2 = 578) → x = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_decrease_l1314_131406


namespace NUMINAMATH_GPT_mr_green_expected_produce_l1314_131421

noncomputable def total_produce_yield (steps_length : ℕ) (steps_width : ℕ) (step_length : ℝ)
                                      (yield_carrots : ℝ) (yield_potatoes : ℝ): ℝ :=
  let length_feet := steps_length * step_length
  let width_feet := steps_width * step_length
  let area := length_feet * width_feet
  let yield_carrots_total := area * yield_carrots
  let yield_potatoes_total := area * yield_potatoes
  yield_carrots_total + yield_potatoes_total

theorem mr_green_expected_produce:
  total_produce_yield 18 25 3 0.4 0.5 = 3645 := by
  sorry

end NUMINAMATH_GPT_mr_green_expected_produce_l1314_131421


namespace NUMINAMATH_GPT_sum_in_base5_correct_l1314_131489

-- Defining the integers
def num1 : ℕ := 210
def num2 : ℕ := 72

-- Summing the integers
def sum : ℕ := num1 + num2

-- Converting the resulting sum to base 5
def to_base5 (n : ℕ) : String :=
  let rec aux (n : ℕ) (acc : List Char) : List Char :=
    if n < 5 then Char.ofNat (n + 48) :: acc
    else aux (n / 5) (Char.ofNat (n % 5 + 48) :: acc)
  String.mk (aux n [])

-- The expected sum in base 5
def expected_sum_base5 : String := "2062"

-- The Lean theorem to be proven
theorem sum_in_base5_correct : to_base5 sum = expected_sum_base5 :=
by
  sorry

end NUMINAMATH_GPT_sum_in_base5_correct_l1314_131489


namespace NUMINAMATH_GPT_divisibility_by_10_l1314_131411

theorem divisibility_by_10 (a : ℤ) (n : ℕ) (h : n ≥ 2) : 
  (a^(2^n + 1) - a) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisibility_by_10_l1314_131411


namespace NUMINAMATH_GPT_necessary_sufficient_condition_l1314_131444

theorem necessary_sufficient_condition 
  (a b : ℝ) : 
  a * |a + b| < |a| * (a + b) ↔ (a < 0 ∧ b > -a) :=
sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_l1314_131444


namespace NUMINAMATH_GPT_sum_a3_a7_l1314_131416

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_a3_a7 (a : ℕ → ℝ)
  (h₁ : arithmetic_sequence a)
  (h₂ : a 1 + a 9 + a 2 + a 8 = 20) :
  a 3 + a 7 = 10 :=
sorry

end NUMINAMATH_GPT_sum_a3_a7_l1314_131416


namespace NUMINAMATH_GPT_proof_problem_l1314_131433

-- Proposition B: ∃ x ∈ ℝ, x^2 - 3*x + 3 < 0
def propB : Prop := ∃ x : ℝ, x^2 - 3 * x + 3 < 0

-- Proposition D: ∀ x ∈ ℝ, x^2 - a*x + 1 = 0 has real solutions
def propD (a : ℝ) : Prop := ∀ x : ℝ, ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0

-- Negation of Proposition B: ∀ x ∈ ℝ, x^2 - 3 * x + 3 ≥ 0
def neg_propB : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 ≥ 0

-- Negation of Proposition D: ∃ a ∈ ℝ, ∃ x ∈ ℝ, ∄ (x1 x2 : ℝ), x^2 - a * x + 1 = 0
def neg_propD : Prop := ∃ a : ℝ, ∀ x : ℝ, ¬ ∃ (x1 x2 : ℝ), x^2 - a * x + 1 = 0 

-- The main theorem combining the results based on the solutions.
theorem proof_problem : neg_propB ∧ neg_propD :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1314_131433


namespace NUMINAMATH_GPT_least_prime_P_with_integer_roots_of_quadratic_l1314_131496

theorem least_prime_P_with_integer_roots_of_quadratic :
  ∃ P : ℕ, P.Prime ∧ (∃ m : ℤ,  m^2 = 12 * P + 60) ∧ P = 7 :=
by
  sorry

end NUMINAMATH_GPT_least_prime_P_with_integer_roots_of_quadratic_l1314_131496


namespace NUMINAMATH_GPT_correct_limiting_reagent_and_yield_l1314_131435

noncomputable def balanced_reaction_theoretical_yield : Prop :=
  let Fe2O3_initial : ℕ := 4
  let CaCO3_initial : ℕ := 10
  let moles_Fe2O3_needed_for_CaCO3 := Fe2O3_initial * (6 / 2)
  let limiting_reagent := if CaCO3_initial < moles_Fe2O3_needed_for_CaCO3 then true else false
  let theoretical_yield := (CaCO3_initial * (3 / 6))
  limiting_reagent = true ∧ theoretical_yield = 5

theorem correct_limiting_reagent_and_yield : balanced_reaction_theoretical_yield :=
by
  sorry

end NUMINAMATH_GPT_correct_limiting_reagent_and_yield_l1314_131435


namespace NUMINAMATH_GPT_slant_height_base_plane_angle_l1314_131440

noncomputable def angle_between_slant_height_and_base_plane (R : ℝ) : ℝ :=
  Real.arcsin ((Real.sqrt 13 - 1) / 3)

theorem slant_height_base_plane_angle (R : ℝ) (h : R = R) : angle_between_slant_height_and_base_plane R = Real.arcsin ((Real.sqrt 13 - 1) / 3) :=
by
  -- Here we assume that the mathematical conditions and transformations hold true.
  -- According to the solution steps provided:
  -- We found that γ = arcsin ((sqrt(13) - 1) / 3)
  sorry

end NUMINAMATH_GPT_slant_height_base_plane_angle_l1314_131440


namespace NUMINAMATH_GPT_finance_charge_rate_l1314_131485

theorem finance_charge_rate (original_balance total_payment finance_charge_rate : ℝ)
    (h1 : original_balance = 150)
    (h2 : total_payment = 153)
    (h3 : finance_charge_rate = ((total_payment - original_balance) / original_balance) * 100) :
    finance_charge_rate = 2 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_finance_charge_rate_l1314_131485


namespace NUMINAMATH_GPT_range_of_a_l1314_131462

variable (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 - a ≥ 0
def q : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0

theorem range_of_a (hpq : p a ∨ q a) (hpnq : ¬p a ∧ ¬q a) : 
  (-1 ≤ a ∧ a ≤ 1) ∨ (a > 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1314_131462


namespace NUMINAMATH_GPT_final_statement_l1314_131495

variable (x : ℝ)

def seven_elevenths_of_five_thirteenths_eq_48 (x : ℝ) :=
  (7/11 : ℝ) * (5/13 : ℝ) * x = 48

def solve_for_x (x : ℝ) : Prop :=
  seven_elevenths_of_five_thirteenths_eq_48 x → x = 196

def calculate_315_percent_of_x (x : ℝ) : Prop :=
  solve_for_x x → 3.15 * x = 617.4

theorem final_statement : calculate_315_percent_of_x x :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_final_statement_l1314_131495


namespace NUMINAMATH_GPT_sum_of_first_ten_primes_ending_in_3_is_671_l1314_131474

noncomputable def sum_of_first_ten_primes_ending_in_3 : ℕ :=
  3 + 13 + 23 + 43 + 53 + 73 + 83 + 103 + 113 + 163

theorem sum_of_first_ten_primes_ending_in_3_is_671 :
  sum_of_first_ten_primes_ending_in_3 = 671 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_ten_primes_ending_in_3_is_671_l1314_131474


namespace NUMINAMATH_GPT_negation_of_p_l1314_131478

-- Define the proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Goal is to show the negation of p
theorem negation_of_p : (¬ p) = (∀ n : ℕ, 2^n ≤ 100) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1314_131478


namespace NUMINAMATH_GPT_amc_inequality_l1314_131402

theorem amc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (a / (b + c^2) + b / (c + a^2) + c / (a + b^2)) ≥ (9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_amc_inequality_l1314_131402


namespace NUMINAMATH_GPT_part1_extreme_value_part2_range_of_a_l1314_131414

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem part1_extreme_value :
  ∃ x : ℝ, f x = -1 :=
  sorry

theorem part2_range_of_a :
  ∀ x > 0, ∃ a : ℝ, f x ≥ x + Real.log x + a + 1 → a ≤ 1 :=
  sorry

end NUMINAMATH_GPT_part1_extreme_value_part2_range_of_a_l1314_131414


namespace NUMINAMATH_GPT_baby_plants_produced_l1314_131483

theorem baby_plants_produced (baby_plants_per_time: ℕ) (times_per_year: ℕ) (years: ℕ) (total_babies: ℕ) :
  baby_plants_per_time = 2 ∧ times_per_year = 2 ∧ years = 4 ∧ total_babies = baby_plants_per_time * times_per_year * years → 
  total_babies = 16 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end NUMINAMATH_GPT_baby_plants_produced_l1314_131483


namespace NUMINAMATH_GPT_students_selected_milk_is_54_l1314_131407

-- Define the parameters.
variable (total_students : ℕ)
variable (students_selected_soda students_selected_milk : ℕ)

-- Given conditions.
axiom h1 : students_selected_soda = 90
axiom h2 : students_selected_soda = (1 / 2) * total_students
axiom h3 : students_selected_milk = (3 / 5) * students_selected_soda

-- Prove that the number of students who selected milk is equal to 54.
theorem students_selected_milk_is_54 : students_selected_milk = 54 :=
by
  sorry

end NUMINAMATH_GPT_students_selected_milk_is_54_l1314_131407


namespace NUMINAMATH_GPT_ganesh_ram_together_l1314_131451

theorem ganesh_ram_together (G R S : ℝ) (h1 : G + R + S = 1 / 16) (h2 : S = 1 / 48) : (G + R) = 1 / 24 :=
by
  sorry

end NUMINAMATH_GPT_ganesh_ram_together_l1314_131451


namespace NUMINAMATH_GPT_rectangle_side_greater_than_twelve_l1314_131426

theorem rectangle_side_greater_than_twelve (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 6 * (a + b)) : a > 12 ∨ b > 12 :=
sorry

end NUMINAMATH_GPT_rectangle_side_greater_than_twelve_l1314_131426


namespace NUMINAMATH_GPT_popsicle_melting_ratio_l1314_131404

theorem popsicle_melting_ratio (S : ℝ) (r : ℝ) (h : r^5 = 32) : r = 2 :=
by
  sorry

end NUMINAMATH_GPT_popsicle_melting_ratio_l1314_131404


namespace NUMINAMATH_GPT_solutions_equation1_solutions_equation2_l1314_131437

-- Definition for the first equation
def equation1 (x : ℝ) : Prop := 4 * x^2 - 9 = 0

-- Definition for the second equation
def equation2 (x : ℝ) : Prop := 2 * x^2 - 3 * x - 5 = 0

theorem solutions_equation1 (x : ℝ) :
  equation1 x ↔ (x = 3 / 2 ∨ x = -3 / 2) := 
  by sorry

theorem solutions_equation2 (x : ℝ) :
  equation2 x ↔ (x = 1 ∨ x = 5 / 2) := 
  by sorry

end NUMINAMATH_GPT_solutions_equation1_solutions_equation2_l1314_131437


namespace NUMINAMATH_GPT_domain_transformation_l1314_131417

variable {α : Type*}
variable {f : α → α}
variable {x y : α}
variable (h₁ : ∀ x, -1 < x ∧ x < 1)

theorem domain_transformation (h₁ : ∀ x, -1 < x ∧ x < 1) : ∀ x, 0 < x ∧ x < 1 →
  ((-1 < (2 * x - 1) ∧ (2 * x - 1) < 1)) :=
by
  intro x
  intro h
  have h₂ : -1 < 2 * x - 1 := sorry
  have h₃ : 2 * x - 1 < 1 := sorry
  exact ⟨h₂, h₃⟩

end NUMINAMATH_GPT_domain_transformation_l1314_131417


namespace NUMINAMATH_GPT_find_x_l1314_131465

def custom_op (a b : ℤ) : ℤ := 2 * a + 3 * b

theorem find_x : ∃ x : ℤ, custom_op 5 (custom_op 7 x) = -4 ∧ x = -56 / 9 := by
  sorry

end NUMINAMATH_GPT_find_x_l1314_131465


namespace NUMINAMATH_GPT_determine_f_value_l1314_131442

-- Define initial conditions
def parabola_eqn (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f
def vertex : (ℝ × ℝ) := (2, -3)
def point_on_parabola : (ℝ × ℝ) := (7, 0)

-- Prove that f = 7 given the conditions
theorem determine_f_value (d e f : ℝ) :
  (parabola_eqn d e f (vertex.snd) = vertex.fst) ∧
  (parabola_eqn d e f (point_on_parabola.snd) = point_on_parabola.fst) →
  f = 7 := 
by
  sorry 

end NUMINAMATH_GPT_determine_f_value_l1314_131442


namespace NUMINAMATH_GPT_required_additional_amount_l1314_131422

noncomputable def ryan_order_total : ℝ := 15.80 + 8.20 + 10.50 + 6.25 + 9.15
def minimum_free_delivery : ℝ := 50
def discount_threshold : ℝ := 30
def discount_rate : ℝ := 0.10

theorem required_additional_amount : 
  ∃ X : ℝ, ryan_order_total + X - discount_rate * (ryan_order_total + X) = minimum_free_delivery :=
sorry

end NUMINAMATH_GPT_required_additional_amount_l1314_131422


namespace NUMINAMATH_GPT_distance_between_centers_l1314_131480

-- Declare radii of the circles and the shortest distance between points on the circles
def R := 28
def r := 12
def d := 10

-- Define the problem to prove the distance between the centers
theorem distance_between_centers (R r d : ℝ) (hR : R = 28) (hr : r = 12) (hd : d = 10) : 
  ∀ OO1 : ℝ, OO1 = 6 :=
by sorry

end NUMINAMATH_GPT_distance_between_centers_l1314_131480


namespace NUMINAMATH_GPT_find_b_eq_neg_three_l1314_131468

theorem find_b_eq_neg_three (b : ℝ) (h : (2 - b) / 5 = -(2 * b + 1) / 5) : b = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_eq_neg_three_l1314_131468


namespace NUMINAMATH_GPT_total_ticket_sales_l1314_131450

def ticket_price : Type := 
  ℕ → ℕ

def total_individual_sales (student_count adult_count child_count senior_count : ℕ) (prices : ticket_price) : ℝ :=
  (student_count * prices 6 + adult_count * prices 8 + child_count * prices 4 + senior_count * prices 7)

def total_group_sales (group_student_count group_adult_count group_child_count group_senior_count : ℕ) (prices : ticket_price) : ℝ :=
  let total_price := (group_student_count * prices 6 + group_adult_count * prices 8 + group_child_count * prices 4 + group_senior_count * prices 7)
  if (group_student_count + group_adult_count + group_child_count + group_senior_count) > 10 then 
    total_price - 0.10 * total_price 
  else 
    total_price

theorem total_ticket_sales
  (prices : ticket_price)
  (student_count adult_count child_count senior_count : ℕ)
  (group_student_count group_adult_count group_child_count group_senior_count : ℕ)
  (total_sales : ℝ) :
  student_count = 20 →
  adult_count = 12 →
  child_count = 15 →
  senior_count = 10 →
  group_student_count = 5 →
  group_adult_count = 8 →
  group_child_count = 10 →
  group_senior_count = 9 →
  prices 6 = 6 →
  prices 8 = 8 →
  prices 4 = 4 →
  prices 7 = 7 →
  total_sales = (total_individual_sales student_count adult_count child_count senior_count prices) + (total_group_sales group_student_count group_adult_count group_child_count group_senior_count prices) →
  total_sales = 523.30 := by
  sorry

end NUMINAMATH_GPT_total_ticket_sales_l1314_131450


namespace NUMINAMATH_GPT_find_value_l1314_131427

theorem find_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) : 2 * Real.sin x + 3 * Real.cos x = -7 / 3 := 
sorry

end NUMINAMATH_GPT_find_value_l1314_131427


namespace NUMINAMATH_GPT_range_of_a_minimum_value_of_b_l1314_131469

def is_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop := f x₀ = x₀

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (2 * b - 1) * x + b - 2
noncomputable def g (a x : ℝ) : ℝ := -x + a / (3 * a^2 - 2 * a + 1)

theorem range_of_a (h : ∀ b : ℝ, ∃ x1 x2 : ℝ, is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) : 0 < a ∧ a < 4 :=
sorry

theorem minimum_value_of_b (hx1 : is_fixed_point (f a b) x₁) (hx2 : is_fixed_point (f a b) x₂)
  (hm : g a ((x₁ + x₂) / 2) = (x₁ + x₂) / 2) (ha : 0 < a ∧ a < 4) : b ≥ 3/4 :=
sorry

end NUMINAMATH_GPT_range_of_a_minimum_value_of_b_l1314_131469


namespace NUMINAMATH_GPT_problem_statement_l1314_131475

theorem problem_statement (f : ℝ → ℝ) (a b c m : ℝ)
  (h_cond1 : ∀ x, f x = -x^2 + a * x + b)
  (h_range : ∀ y, y ∈ Set.range f ↔ y ≤ 0)
  (h_ineq_sol : ∀ x, ((-x^2 + a * x + b > c - 1) ↔ (m - 4 < x ∧ x < m + 1))) :
  (b = -(1/4) * (2 * m - 3)^2) ∧ (c = -(21 / 4)) := sorry

end NUMINAMATH_GPT_problem_statement_l1314_131475


namespace NUMINAMATH_GPT_factor_poly_l1314_131443

theorem factor_poly (x : ℝ) : (75 * x^3 - 300 * x^7) = 75 * x^3 * (1 - 4 * x^4) :=
by sorry

end NUMINAMATH_GPT_factor_poly_l1314_131443


namespace NUMINAMATH_GPT_work_problem_l1314_131434

theorem work_problem (x : ℝ) (h1 : x > 0) 
                      (h2 : (2 * (1 / 4 + 1 / x) + 2 * (1 / x) = 1)) : 
                      x = 8 := sorry

end NUMINAMATH_GPT_work_problem_l1314_131434


namespace NUMINAMATH_GPT_rational_sum_l1314_131492

theorem rational_sum (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : x + y = 7 ∨ x + y = 3 := 
sorry

end NUMINAMATH_GPT_rational_sum_l1314_131492


namespace NUMINAMATH_GPT_selling_price_A_count_purchasing_plans_refund_amount_l1314_131476

-- Problem 1
theorem selling_price_A (last_revenue this_revenue last_price this_price cars_sold : ℝ) 
    (last_revenue_eq : last_revenue = 1) (this_revenue_eq : this_revenue = 0.9)
    (diff_eq : last_price = this_price + 1)
    (same_cars : cars_sold ≠ 0) :
    this_price = 9 := by
  sorry

-- Problem 2
theorem count_purchasing_plans (cost_A cost_B total_cars min_cost max_cost : ℝ)
    (cost_A_eq : cost_A = 0.75) (cost_B_eq : cost_B = 0.6)
    (total_cars_eq : total_cars = 15) (min_cost_eq : min_cost = 0.99)
    (max_cost_eq : max_cost = 1.05) :
    ∃ n : ℕ, n = 5 := by
  sorry

-- Problem 3
theorem refund_amount (refund_A refund_B revenue_A revenue_B cost_A cost_B total_profits a : ℝ)
    (revenue_B_eq : revenue_B = 0.8) (cost_A_eq : cost_A = 0.75)
    (cost_B_eq : cost_B = 0.6) (total_profits_eq : total_profits = 30 - 15 * a) :
    a = 0.5 := by
  sorry

end NUMINAMATH_GPT_selling_price_A_count_purchasing_plans_refund_amount_l1314_131476


namespace NUMINAMATH_GPT_cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l1314_131481

theorem cos_alpha_minus_11pi_div_12_eq_neg_2_div_3
  (α : ℝ)
  (h : Real.sin (7 * Real.pi / 12 + α) = 2 / 3) :
  Real.cos (α - 11 * Real.pi / 12) = -(2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_11pi_div_12_eq_neg_2_div_3_l1314_131481


namespace NUMINAMATH_GPT_area_of_square_l1314_131424

-- Conditions: Points A (5, -2) and B (5, 3) are adjacent corners of a square.
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (5, 3)

-- The statement to prove that the area of the square formed by these points is 25.
theorem area_of_square : (∃ s : ℝ, s = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) → s^2 = 25 :=
sorry

end NUMINAMATH_GPT_area_of_square_l1314_131424


namespace NUMINAMATH_GPT_modulus_of_z_l1314_131473

-- Definitions of the problem conditions
def z := Complex.mk 1 (-1)

-- Statement of the math proof problem
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_modulus_of_z_l1314_131473


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1314_131438

-- Define the arithmetic sequence and the given conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values for the sequence a_1 = 2 and a_2 + a_3 = 13
variables {a : ℕ → ℤ} (d : ℤ)
axiom h1 : a 1 = 2
axiom h2 : a 2 + a 3 = 13

-- Conclude the value of a_4 + a_5 + a_6
theorem arithmetic_sequence_sum : a 4 + a 5 + a 6 = 42 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1314_131438


namespace NUMINAMATH_GPT_problem_1_problem_2_l1314_131497

-- Definitions for the sets A and B:

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < 1 + m }

-- Problem 1: When m = -2, find A ∪ B
theorem problem_1 : set_A ∪ set_B (-2) = { x | -5 < x ∧ x ≤ 4 } :=
sorry

-- Problem 2: If A ∩ B = B, find the range of the real number m
theorem problem_2 : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≥ -1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1314_131497


namespace NUMINAMATH_GPT_x_is_perfect_square_l1314_131499

theorem x_is_perfect_square (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (hdiv : 2 * x * y ∣ x^2 + y^2 - x) : ∃ (n : ℕ), x = n^2 :=
by
  sorry

end NUMINAMATH_GPT_x_is_perfect_square_l1314_131499


namespace NUMINAMATH_GPT_expression_eval_l1314_131431

theorem expression_eval :
  -14 - (-2) ^ 3 * (1 / 4) - 16 * (1 / 2 - 1 / 4 + 3 / 8) = -22 := by
  sorry

end NUMINAMATH_GPT_expression_eval_l1314_131431


namespace NUMINAMATH_GPT_sum_of_drawn_numbers_is_26_l1314_131419

theorem sum_of_drawn_numbers_is_26 :
  ∃ A B : ℕ, A > 1 ∧ A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B ∧ Prime B ∧
           (150 * B + A = k^2) ∧ 1 ≤ B ∧ (B > 1 → A > 1 ∧ B = 2) ∧ A + B = 26 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_drawn_numbers_is_26_l1314_131419


namespace NUMINAMATH_GPT_find_x1_l1314_131418

variable (x1 x2 x3 : ℝ)

theorem find_x1 (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 0.8)
    (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
    x1 = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_find_x1_l1314_131418


namespace NUMINAMATH_GPT_coin_prob_not_unique_l1314_131428

theorem coin_prob_not_unique (p : ℝ) (w : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : w = 144 / 625) :
  ¬ ∃! p, (∃ w, w = 10 * p^3 * (1 - p)^2 ∧ w = 144 / 625) :=
by
  sorry

end NUMINAMATH_GPT_coin_prob_not_unique_l1314_131428


namespace NUMINAMATH_GPT_percentage_error_in_area_l1314_131457

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := 1.02 * s
  let A := s ^ 2
  let A' := s' ^ 2
  let error := A' - A
  let percent_error := (error / A) * 100
  percent_error = 4.04 := by
  sorry

end NUMINAMATH_GPT_percentage_error_in_area_l1314_131457


namespace NUMINAMATH_GPT_expected_value_of_8_sided_die_l1314_131445

-- Define the expected value function for the given win calculation rule.
def expected_value := (1/8 : ℚ) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

-- Formal statement of the proof problem.
theorem expected_value_of_8_sided_die : 
  expected_value = 3.50 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_8_sided_die_l1314_131445


namespace NUMINAMATH_GPT_commission_rate_l1314_131454

theorem commission_rate (old_salary new_base_salary sale_amount : ℝ) (required_sales : ℕ) (condition: (old_salary = 75000) ∧ (new_base_salary = 45000) ∧ (sale_amount = 750) ∧ (required_sales = 267)) :
  ∃ commission_rate : ℝ, abs (commission_rate - 0.14981) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_commission_rate_l1314_131454


namespace NUMINAMATH_GPT_binom_1300_2_eq_l1314_131415

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end NUMINAMATH_GPT_binom_1300_2_eq_l1314_131415


namespace NUMINAMATH_GPT_angle_at_3_15_l1314_131432

-- Define the measurements and conditions
def hour_hand_position (hour min : ℕ) : ℝ := 
  30 * hour + 0.5 * min

def minute_hand_position (min : ℕ) : ℝ := 
  6 * min

def angle_between_hands (hour min : ℕ) : ℝ := 
  abs (minute_hand_position min - hour_hand_position hour min)

-- Theorem statement in Lean 4
theorem angle_at_3_15 : angle_between_hands 3 15 = 7.5 :=
by sorry

end NUMINAMATH_GPT_angle_at_3_15_l1314_131432


namespace NUMINAMATH_GPT_distance_between_A_and_mrs_A_l1314_131425

-- Define the initial conditions
def speed_mr_A : ℝ := 30 -- Mr. A's speed in kmph
def speed_mrs_A : ℝ := 10 -- Mrs. A's speed in kmph
def speed_bee : ℝ := 60 -- The bee's speed in kmph
def distance_bee_traveled : ℝ := 180 -- Distance traveled by the bee in km

-- Define the proven statement
theorem distance_between_A_and_mrs_A : 
  distance_bee_traveled / speed_bee * (speed_mr_A + speed_mrs_A) = 120 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_A_and_mrs_A_l1314_131425


namespace NUMINAMATH_GPT_inequality_solution_l1314_131449

variable {a b : ℝ}

theorem inequality_solution
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) :
  ab > ab^2 ∧ ab^2 > a := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1314_131449


namespace NUMINAMATH_GPT_quadrant_iv_l1314_131420

theorem quadrant_iv (x y : ℚ) (h1 : x = 1) (h2 : x - y = 12 / 5) (h3 : 6 * x + 5 * y = -1) :
  x = 1 ∧ y = -7 / 5 ∧ (12 / 5 > 0 ∧ -7 / 5 < 0) :=
by
  sorry

end NUMINAMATH_GPT_quadrant_iv_l1314_131420


namespace NUMINAMATH_GPT_max_rectangle_area_l1314_131464

-- Define the perimeter and side lengths of the rectangle
def perimeter := 60
def L (x : ℝ) := x
def W (x : ℝ) := 30 - x

-- Define the area of the rectangle
def area (x : ℝ) := L x * W x

-- State the theorem to prove the maximum area is 225 square feet
theorem max_rectangle_area : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ area x = 225 :=
by
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1314_131464


namespace NUMINAMATH_GPT_tan_C_in_triangle_l1314_131429

theorem tan_C_in_triangle
  (A B C : ℝ)
  (cos_A : Real.cos A = 4/5)
  (tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 := 
sorry

end NUMINAMATH_GPT_tan_C_in_triangle_l1314_131429


namespace NUMINAMATH_GPT_triangle_tan_inequality_l1314_131400

theorem triangle_tan_inequality 
  {A B C : ℝ} 
  (h1 : π / 2 ≠ A) 
  (h2 : A ≥ B) 
  (h3 : B ≥ C) : 
  |Real.tan A| ≥ Real.tan B ∧ Real.tan B ≥ Real.tan C := 
  by
    sorry

end NUMINAMATH_GPT_triangle_tan_inequality_l1314_131400


namespace NUMINAMATH_GPT_evaluate_expression_l1314_131412

variable (a b : ℤ)

-- Define the main expression
def main_expression (a b : ℤ) : ℤ :=
  (a - b)^2 + (a + 3 * b) * (a - 3 * b) - a * (a - 2 * b)

theorem evaluate_expression : main_expression (-1) 2 = -31 := by
  -- substituting the value and solving it in the proof block
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1314_131412


namespace NUMINAMATH_GPT_polynomial_value_at_2_l1314_131447

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

-- Define the transformation rules for each v_i according to Horner's Rule
def v0 : ℝ := 1
def v1 (x : ℝ) : ℝ := (v0 * x) - 12
def v2 (x : ℝ) : ℝ := (v1 x * x) + 60
def v3 (x : ℝ) : ℝ := (v2 x * x) - 160

-- State the theorem to be proven
theorem polynomial_value_at_2 : v3 2 = -80 := 
by 
  -- Since this is just a Lean 4 statement, we include sorry to defer proof
  sorry

end NUMINAMATH_GPT_polynomial_value_at_2_l1314_131447


namespace NUMINAMATH_GPT_garden_snake_is_10_inches_l1314_131467

-- Define the conditions from the problem statement
def garden_snake_length (garden_snake boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 7 * garden_snake

def boa_constrictor_length (boa_constrictor : ℝ) : Prop :=
  boa_constrictor = 70

-- Prove the length of the garden snake
theorem garden_snake_is_10_inches : ∃ (garden_snake : ℝ), garden_snake_length garden_snake 70 ∧ garden_snake = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_garden_snake_is_10_inches_l1314_131467


namespace NUMINAMATH_GPT_negation_of_proposition_l1314_131439

variable (x : ℝ)

theorem negation_of_proposition (h : ∃ x : ℝ, x^2 + x - 1 < 0) : ¬ (∀ x : ℝ, x^2 + x - 1 ≥ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1314_131439


namespace NUMINAMATH_GPT_max_diff_x_y_l1314_131493

theorem max_diff_x_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  x - y ≤ Real.sqrt (4 / 3) := 
by
  sorry

end NUMINAMATH_GPT_max_diff_x_y_l1314_131493


namespace NUMINAMATH_GPT_group_size_l1314_131471

noncomputable def total_cost : ℤ := 13500
noncomputable def cost_per_person : ℤ := 900

theorem group_size : total_cost / cost_per_person = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_group_size_l1314_131471


namespace NUMINAMATH_GPT_san_antonio_bus_passes_4_austin_buses_l1314_131484

theorem san_antonio_bus_passes_4_austin_buses :
  ∀ (hourly_austin_buses : ℕ → ℕ) (every_50_minute_san_antonio_buses : ℕ → ℕ) (trip_time : ℕ),
    (∀ h : ℕ, hourly_austin_buses (h) = (h * 60)) →
    (∀ m : ℕ, every_50_minute_san_antonio_buses (m) = (m * 60 + 50)) →
    trip_time = 240 →
    ∃ num_buses_passed : ℕ, num_buses_passed = 4 :=
by
  sorry

end NUMINAMATH_GPT_san_antonio_bus_passes_4_austin_buses_l1314_131484


namespace NUMINAMATH_GPT_smaller_circle_radius_l1314_131423

theorem smaller_circle_radius :
  ∀ (R r : ℝ), R = 10 ∧ (4 * r = 2 * R) → r = 5 :=
by
  intro R r
  intro h
  have h1 : R = 10 := h.1
  have h2 : 4 * r = 2 * R := h.2
  -- Use the conditions to eventually show r = 5
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l1314_131423


namespace NUMINAMATH_GPT_part1_part2_l1314_131494

-- Definitions from condition part
def f (a x : ℝ) := a * x^2 + (1 + a) * x + a

-- Part (1) Statement
theorem part1 (a : ℝ) : 
  (a ≥ -1/3) → (∀ x : ℝ, f a x ≥ 0) :=
sorry

-- Part (2) Statement
theorem part2 (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, f a x < a - 1) → 
  ((0 < a ∧ a < 1) → (-1/a < x ∧ x < -1) ∨ 
   (a = 1) → False ∨
   (a > 1) → (-1 < x ∧ x < -1/a)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1314_131494


namespace NUMINAMATH_GPT_cookies_with_five_cups_l1314_131482

theorem cookies_with_five_cups (cookies_per_four_cups : ℕ) (flour_for_four_cups : ℕ) (flour_for_five_cups : ℕ) (h : 24 / 4 = cookies_per_four_cups / 5) :
  cookies_per_four_cups = 30 :=
by
  sorry

end NUMINAMATH_GPT_cookies_with_five_cups_l1314_131482


namespace NUMINAMATH_GPT_find_circle_equation_l1314_131466

-- Define the conditions on the circle
def passes_through_points (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, (c = center ∧ r = radius) ∧ 
  dist (0, 2) c = r ∧ dist (0, 4) c = r

def lies_on_line (center : ℝ × ℝ) : Prop :=
  2 * center.1 - center.2 - 1 = 0

-- Define the problem
theorem find_circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  passes_through_points center radius ∧ lies_on_line center ∧ 
  (∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius^2 
  ↔ (x - 2)^2 + (y - 3)^2 = 5) :=
sorry

end NUMINAMATH_GPT_find_circle_equation_l1314_131466


namespace NUMINAMATH_GPT_frank_initial_boxes_l1314_131401

theorem frank_initial_boxes (filled left : ℕ) (h_filled : filled = 8) (h_left : left = 5) : 
  filled + left = 13 := by
  sorry

end NUMINAMATH_GPT_frank_initial_boxes_l1314_131401


namespace NUMINAMATH_GPT_minimum_students_ans_q1_correctly_l1314_131463

variable (Total Students Q1 Q2 Q1_and_Q2 : ℕ)
variable (did_not_take_test: Student → Bool)

-- Given Conditions
def total_students := 40
def students_ans_q2_correctly := 29
def students_not_taken_test := 10
def students_ans_both_correctly := 29

theorem minimum_students_ans_q1_correctly (H1: Q2 - students_not_taken_test == 30)
                                           (H2: Q1_and_Q2 + students_not_taken_test == total_students)
                                           (H3: Q1_and_Q2 == students_ans_q2_correctly):
  Q1 ≥ 29 := by
  sorry

end NUMINAMATH_GPT_minimum_students_ans_q1_correctly_l1314_131463


namespace NUMINAMATH_GPT_inequality_for_positive_n_and_x_l1314_131477

theorem inequality_for_positive_n_and_x (n : ℕ) (x : ℝ) (hn : n > 0) (hx : x > 0) :
  (x^(2 * n - 1) - 1) / (2 * n - 1) ≤ (x^(2 * n) - 1) / (2 * n) :=
by sorry

end NUMINAMATH_GPT_inequality_for_positive_n_and_x_l1314_131477


namespace NUMINAMATH_GPT_robert_saves_5_dollars_l1314_131490

theorem robert_saves_5_dollars :
  let original_price := 50
  let promotion_c_discount (price : ℕ) := price * 20 / 100
  let promotion_d_discount (price : ℕ) := 15
  let cost_promotion_c := original_price + (original_price - promotion_c_discount original_price)
  let cost_promotion_d := original_price + (original_price - promotion_d_discount original_price)
  (cost_promotion_c - cost_promotion_d) = 5 :=
by
  sorry

end NUMINAMATH_GPT_robert_saves_5_dollars_l1314_131490


namespace NUMINAMATH_GPT_star_eq_zero_iff_x_eq_5_l1314_131456

/-- Define the operation * on real numbers -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Proposition stating that x = 5 is the solution to (x - 4) * 1 = 0 -/
theorem star_eq_zero_iff_x_eq_5 (x : ℝ) : (star (x-4) 1 = 0) ↔ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_star_eq_zero_iff_x_eq_5_l1314_131456


namespace NUMINAMATH_GPT_pipe_p_fills_cistern_in_12_minutes_l1314_131470

theorem pipe_p_fills_cistern_in_12_minutes :
  (∃ (t : ℝ), 
    ∀ (q_fill_rate p_fill_rate : ℝ), 
      q_fill_rate = 1 / 15 ∧ 
      t > 0 ∧ 
      (4 * (1 / t + q_fill_rate) + 6 * q_fill_rate = 1) → t = 12) :=
sorry

end NUMINAMATH_GPT_pipe_p_fills_cistern_in_12_minutes_l1314_131470


namespace NUMINAMATH_GPT_negation_P_l1314_131448

-- Define the condition that x is a real number
variable (x : ℝ)

-- Define the proposition P
def P := ∀ (x : ℝ), x ≥ 2

-- Define the negation of P
def not_P := ∃ (x : ℝ), x < 2

-- Theorem stating the equivalence of the negation of P
theorem negation_P : ¬P ↔ not_P := by
  sorry

end NUMINAMATH_GPT_negation_P_l1314_131448


namespace NUMINAMATH_GPT_barge_arrives_at_B_at_2pm_l1314_131413

noncomputable def barge_arrival_time
  (constant_barge_speed : ℝ)
  (river_current_speed : ℝ)
  (distance_AB : ℝ)
  (time_depart_A : ℕ)
  (wait_time_B : ℝ)
  (time_return_A : ℝ) :
  ℝ := by
  sorry

theorem barge_arrives_at_B_at_2pm :
  ∀ (constant_barge_speed : ℝ), 
    (river_current_speed = 3) →
    (distance_AB = 60) →
    (time_depart_A = 9) →
    (wait_time_B = 2) →
    (time_return_A = 19 + 20 / 60) →
    barge_arrival_time constant_barge_speed river_current_speed distance_AB time_depart_A wait_time_B time_return_A = 14 := by
  sorry

end NUMINAMATH_GPT_barge_arrives_at_B_at_2pm_l1314_131413


namespace NUMINAMATH_GPT_snow_volume_l1314_131436

-- Define the dimensions of the sidewalk and the snow depth
def length : ℝ := 20
def width : ℝ := 2
def depth : ℝ := 0.5

-- Define the volume calculation
def volume (l w d : ℝ) : ℝ := l * w * d

-- The theorem to prove
theorem snow_volume : volume length width depth = 20 := 
by
  sorry

end NUMINAMATH_GPT_snow_volume_l1314_131436


namespace NUMINAMATH_GPT_circle_tangent_to_x_axis_at_origin_l1314_131455

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h : ∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → 
      (x = 0 → y = 0)) :
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end NUMINAMATH_GPT_circle_tangent_to_x_axis_at_origin_l1314_131455


namespace NUMINAMATH_GPT_symmetric_line_x_axis_l1314_131403

theorem symmetric_line_x_axis (x y : ℝ) : 
  let P := (x, y)
  let P' := (x, -y)
  (3 * x - 4 * y + 5 = 0) →  
  (3 * x + 4 * -y + 5 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_line_x_axis_l1314_131403


namespace NUMINAMATH_GPT_cone_surface_area_volume_ineq_l1314_131446

theorem cone_surface_area_volume_ineq
  (A V r a m : ℝ)
  (hA : A = π * r * (r + a))
  (hV : V = (1/3) * π * r^2 * m)
  (hPythagoras : a^2 = r^2 + m^2) :
  A^3 ≥ 72 * π * V^2 := 
by
  sorry

end NUMINAMATH_GPT_cone_surface_area_volume_ineq_l1314_131446


namespace NUMINAMATH_GPT_diving_competition_scores_l1314_131441

theorem diving_competition_scores (A B C D E : ℝ) (hA : 1 ≤ A ∧ A ≤ 10)
  (hB : 1 ≤ B ∧ B ≤ 10) (hC : 1 ≤ C ∧ C ≤ 10) (hD : 1 ≤ D ∧ D ≤ 10) 
  (hE : 1 ≤ E ∧ E ≤ 10) (degree_of_difficulty : ℝ) (h_diff : degree_of_difficulty = 3.2)
  (point_value : ℝ) (h_point_value : point_value = 79.36) :
  A = max A (max B (max C (max D E))) →
  E = min A (min B (min C (min D E))) →
  (B + C + D) = (point_value / degree_of_difficulty) :=
by sorry

end NUMINAMATH_GPT_diving_competition_scores_l1314_131441


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l1314_131453

open Real

-- Definitions based on the given conditions
def asymptote1 (x : ℝ) : ℝ := x + 3
def asymptote2 (x : ℝ) : ℝ := -x + 5
def hyperbola_passes_through (x y : ℝ) : Prop := x = 4 ∧ y = 6
noncomputable def hyperbola_centre : (ℝ × ℝ) := (1, 4)

-- Definition of the hyperbola and the proof problem
theorem distance_between_foci_of_hyperbola (x y : ℝ) (hx : asymptote1 x = y) (hy : asymptote2 x = y) (hpass : hyperbola_passes_through 4 6) :
  2 * (sqrt (5 + 5)) = 2 * sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l1314_131453


namespace NUMINAMATH_GPT_age_equation_correct_l1314_131486

-- Define the main theorem
theorem age_equation_correct (x : ℕ) (h1 : ∀ (b : ℕ), b = 2 * x) (h2 : ∀ (b4 s4 : ℕ), b4 = b - 4 ∧ s4 = x - 4 ∧ b4 = 3 * s4) : 
  2 * x - 4 = 3 * (x - 4) :=
by
  sorry

end NUMINAMATH_GPT_age_equation_correct_l1314_131486


namespace NUMINAMATH_GPT_taxi_fare_l1314_131430

theorem taxi_fare (x : ℝ) (h : x > 6) : 
  let starting_price := 6
  let mid_distance_fare := (6 - 2) * 2.4
  let long_distance_fare := (x - 6) * 3.6
  let total_fare := starting_price + mid_distance_fare + long_distance_fare
  total_fare = 3.6 * x - 6 :=
by
  sorry

end NUMINAMATH_GPT_taxi_fare_l1314_131430


namespace NUMINAMATH_GPT_find_minimum_value_l1314_131458

open Real

noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

theorem find_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 4 := 
sorry

end NUMINAMATH_GPT_find_minimum_value_l1314_131458
