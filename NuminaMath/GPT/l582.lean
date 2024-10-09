import Mathlib

namespace monotonic_intervals_of_f_l582_58265

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

-- Define the derivative f'
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - 1

-- Prove the monotonicity intervals of the function f
theorem monotonic_intervals_of_f :
  (∀ x : ℝ, x < 0 → f' x < 0) ∧ (∀ x : ℝ, 0 < x → f' x > 0) :=
by
  sorry

end monotonic_intervals_of_f_l582_58265


namespace passing_probability_l582_58292

theorem passing_probability :
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  probability = 44 / 45 :=
by
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  have p_eq : probability = 44 / 45 := sorry
  exact p_eq

end passing_probability_l582_58292


namespace geom_arith_sequence_l582_58219

theorem geom_arith_sequence (a b c m n : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : m = (a + b) / 2) 
  (h3 : n = (b + c) / 2) : 
  a / m + c / n = 2 := 
by 
  sorry

end geom_arith_sequence_l582_58219


namespace factor_polynomial_l582_58254

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l582_58254


namespace train_length_l582_58218

theorem train_length (L V : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 175 = V * 39) : 
  L = 150 := 
by 
  -- proof omitted 
  sorry

end train_length_l582_58218


namespace cost_per_spool_l582_58250

theorem cost_per_spool
  (p : ℕ) (f : ℕ) (y : ℕ) (t : ℕ) (n : ℕ)
  (hp : p = 15) (hf : f = 24) (hy : y = 5) (ht : t = 141) (hn : n = 2) :
  (t - (p + y * f)) / n = 3 :=
by sorry

end cost_per_spool_l582_58250


namespace no_nonzero_integer_solution_l582_58290

theorem no_nonzero_integer_solution (m n p : ℤ) :
  (m + n * Real.sqrt 2 + p * Real.sqrt 3 = 0) → (m = 0 ∧ n = 0 ∧ p = 0) :=
by sorry

end no_nonzero_integer_solution_l582_58290


namespace sum_and_product_l582_58203

theorem sum_and_product (c d : ℝ) (h1 : 2 * c = -8) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end sum_and_product_l582_58203


namespace evaluate_x_squared_minus_y_squared_l582_58229

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l582_58229


namespace smallest_difference_l582_58288

noncomputable def triangle_lengths (DE EF FD : ℕ) : Prop :=
  DE < EF ∧ EF ≤ FD ∧ DE + EF + FD = 3010 ∧ DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference :
  ∃ (DE EF FD : ℕ), triangle_lengths DE EF FD ∧ EF - DE = 1 :=
by
  sorry

end smallest_difference_l582_58288


namespace initial_population_is_3162_l582_58279

noncomputable def initial_population (P : ℕ) : Prop :=
  let after_bombardment := 0.95 * (P : ℝ)
  let after_fear := 0.85 * after_bombardment
  after_fear = 2553

theorem initial_population_is_3162 : initial_population 3162 :=
  by
    -- By our condition setup, we need to prove:
    -- let after_bombardment := 0.95 * 3162
    -- let after_fear := 0.85 * after_bombardment
    -- after_fear = 2553

    -- This can be directly stated and verified through concrete calculations as in the problem steps.
    sorry

end initial_population_is_3162_l582_58279


namespace fraction_to_percentage_l582_58209

theorem fraction_to_percentage (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) = 0.65 * y :=
by
  -- the proof steps will go here
  sorry

end fraction_to_percentage_l582_58209


namespace triangle_angle_distance_l582_58212

noncomputable def triangle_properties (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) : Prop :=
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R

theorem triangle_angle_distance (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) :
  triangle_properties ABC P Q R angle dist →
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R :=
by intros; sorry

end triangle_angle_distance_l582_58212


namespace min_score_needed_l582_58226

-- Definitions of the conditions
def current_scores : List ℤ := [88, 92, 75, 81, 68, 70]
def desired_increase : ℤ := 5
def number_of_tests := current_scores.length
def current_total : ℤ := current_scores.sum
def current_average : ℤ := current_total / number_of_tests
def desired_average : ℤ := current_average + desired_increase 
def new_number_of_tests : ℤ := number_of_tests + 1
def total_required_score : ℤ := desired_average * new_number_of_tests

-- Lean 4 statement (theorem) to prove
theorem min_score_needed : total_required_score - current_total = 114 := by
  sorry

end min_score_needed_l582_58226


namespace f_at_2_lt_e6_l582_58258

variable (f : ℝ → ℝ)

-- Specify the conditions
axiom derivable_f : Differentiable ℝ f
axiom condition_3f_gt_fpp : ∀ x : ℝ, 3 * f x > (deriv (deriv f)) x
axiom f_at_1 : f 1 = Real.exp 3

-- Conclusion to prove
theorem f_at_2_lt_e6 : f 2 < Real.exp 6 :=
sorry

end f_at_2_lt_e6_l582_58258


namespace floor_sufficient_but_not_necessary_l582_58206

theorem floor_sufficient_but_not_necessary {x y : ℝ} : 
  (∀ x y : ℝ, (⌊x⌋₊ = ⌊y⌋₊) → abs (x - y) < 1) ∧ 
  ¬ (∀ x y : ℝ, abs (x - y) < 1 → (⌊x⌋₊ = ⌊y⌋₊)) :=
by
  sorry

end floor_sufficient_but_not_necessary_l582_58206


namespace Karen_packs_piece_of_cake_days_l582_58235

theorem Karen_packs_piece_of_cake_days 
(Total Ham_Days : ℕ) (Ham_probability Cake_probability : ℝ) 
  (H_Total : Total = 5) 
  (H_Ham_Days : Ham_Days = 3) 
  (H_Ham_probability : Ham_probability = (3 / 5)) 
  (H_Cake_probability : Ham_probability * (Cake_probability / 5) = 0.12) : 
  Cake_probability = 1 := 
by
  sorry

end Karen_packs_piece_of_cake_days_l582_58235


namespace positive_number_is_49_l582_58211

theorem positive_number_is_49 (a : ℝ) (x : ℝ) (h₁ : (3 - a) * (3 - a) = x) (h₂ : (2 * a + 1) * (2 * a + 1) = x) :
  x = 49 :=
sorry

end positive_number_is_49_l582_58211


namespace initial_flowers_per_bunch_l582_58202

theorem initial_flowers_per_bunch (x : ℕ) (h₁: 8 * x = 72) : x = 9 :=
  by
  sorry

end initial_flowers_per_bunch_l582_58202


namespace length_difference_squares_l582_58230

theorem length_difference_squares (A B : ℝ) (hA : A^2 = 25) (hB : B^2 = 81) : B - A = 4 :=
by
  sorry

end length_difference_squares_l582_58230


namespace grid_area_l582_58276

-- Definitions based on problem conditions
def num_lines : ℕ := 36
def perimeter : ℕ := 72
def side_length : ℕ := perimeter / num_lines

-- Problem statement
theorem grid_area (h : num_lines = 36) (p : perimeter = 72)
  (s : side_length = 2) :
  let n_squares := (8 - 1) * (4 - 1)
  let area_square := side_length ^ 2
  let total_area := n_squares * area_square
  total_area = 84 :=
by {
  -- Skipping proof
  sorry
}

end grid_area_l582_58276


namespace delta_k_f_l582_58294

open Nat

-- Define the function
def f (n : ℕ) : ℕ := 3^n

-- Define the discrete difference operator
def Δ (g : ℕ → ℕ) (n : ℕ) : ℕ := g (n + 1) - g n

-- Define the k-th discrete difference
def Δk (g : ℕ → ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  if k = 0 then g n else Δk (Δ g) (k - 1) n

-- State the theorem
theorem delta_k_f (k : ℕ) (n : ℕ) (h : k ≥ 1) : Δk f k n = 2^k * 3^n := by
  sorry

end delta_k_f_l582_58294


namespace max_path_length_CQ_D_l582_58282

noncomputable def maxCQDPathLength (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) : ℝ :=
  let r := dAB / 2
  let dCD := dAB - dAC - dBD
  2 * Real.sqrt (r^2 - (dCD / 2)^2)

theorem max_path_length_CQ_D 
  (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) (r := dAB / 2) (dCD := dAB - dAC - dBD) :
  dAB = 16 ∧ dAC = 3 ∧ dBD = 5 ∧ r = 8 ∧ dCD = 8
  → maxCQDPathLength 16 3 5 = 8 * Real.sqrt 3 :=
by
  intros h
  cases h
  sorry

end max_path_length_CQ_D_l582_58282


namespace quadratic_solution_unique_l582_58204

theorem quadratic_solution_unique (b : ℝ) (hb : b ≠ 0) (hdisc : 30 * 30 - 4 * b * 10 = 0) :
  ∃ x : ℝ, bx ^ 2 + 30 * x + 10 = 0 ∧ x = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l582_58204


namespace number_of_correct_statements_l582_58266

def is_opposite (a b : ℤ) : Prop := a + b = 0

def statement1 : Prop := ∀ a b : ℤ, (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → is_opposite a b
def statement2 : Prop := ∀ n : ℤ, n = -n → n < 0
def statement3 : Prop := ∀ a b : ℤ, is_opposite a b → a + b = 0
def statement4 : Prop := ∀ a b : ℤ, is_opposite a b → (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)

theorem number_of_correct_statements : (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ↔ (∃n : ℕ, n = 1) :=
by
  sorry

end number_of_correct_statements_l582_58266


namespace arithmetic_sequence_sum_l582_58214

theorem arithmetic_sequence_sum (a d : ℚ) (a1 : a = 1 / 2) 
(S : ℕ → ℚ) (Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) 
(S2_eq_a3 : S 2 = a + 2 * d) :
  ∀ n, S n = (1 / 4 : ℚ) * n^2 + (1 / 4 : ℚ) * n :=
by
  intros n
  sorry

end arithmetic_sequence_sum_l582_58214


namespace total_number_of_squares_is_13_l582_58227

-- Define the vertices of the region
def region_condition (x y : ℕ) : Prop :=
  y ≤ x ∧ y ≤ 4 ∧ x ≤ 4

-- Define the type of squares whose vertices have integer coordinates
def square (n : ℕ) (x y : ℕ) : Prop :=
  region_condition x y ∧ region_condition (x - n) y ∧ 
  region_condition x (y - n) ∧ region_condition (x - n) (y - n)

-- Count the number of squares of each size within the region
def number_of_squares (size : ℕ) : ℕ :=
  match size with
  | 1 => 10 -- number of 1x1 squares
  | 2 => 3  -- number of 2x2 squares
  | _ => 0  -- there are no larger squares in this context

-- Prove the total number of squares is 13
theorem total_number_of_squares_is_13 : number_of_squares 1 + number_of_squares 2 = 13 :=
by
  sorry

end total_number_of_squares_is_13_l582_58227


namespace eval_g_at_8_l582_58208

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem eval_g_at_8 : g 8 = 13 / 3 := by
  sorry

end eval_g_at_8_l582_58208


namespace eval_expression_l582_58237

variable {x : ℝ}

theorem eval_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 8 * x + 2 :=
by
  sorry

end eval_expression_l582_58237


namespace factorization_difference_of_squares_l582_58261

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l582_58261


namespace tom_total_trip_cost_is_correct_l582_58287

noncomputable def Tom_total_cost : ℝ :=
  let cost_vaccines := 10 * 45
  let cost_doctor := 250
  let total_medical := cost_vaccines + cost_doctor
  
  let insurance_coverage := 0.8 * total_medical
  let out_of_pocket_medical := total_medical - insurance_coverage
  
  let cost_flight := 1200

  let cost_lodging := 7 * 150
  let cost_transportation := 200
  let cost_food := 7 * 60
  let total_local_usd := cost_lodging + cost_transportation + cost_food
  let total_local_bbd := total_local_usd * 2

  let conversion_fee_bbd := 0.03 * total_local_bbd
  let conversion_fee_usd := conversion_fee_bbd / 2

  out_of_pocket_medical + cost_flight + total_local_usd + conversion_fee_usd

theorem tom_total_trip_cost_is_correct : Tom_total_cost = 3060.10 :=
  by
    -- Proof skipped
    sorry

end tom_total_trip_cost_is_correct_l582_58287


namespace line_length_limit_l582_58274

theorem line_length_limit : 
  ∑' n : ℕ, 1 / ((3 : ℝ) ^ n) + (1 / (3 ^ (n + 1))) * (Real.sqrt 3) = (3 + Real.sqrt 3) / 2 :=
sorry

end line_length_limit_l582_58274


namespace square_of_real_is_positive_or_zero_l582_58224

def p (x : ℝ) : Prop := x^2 > 0
def q (x : ℝ) : Prop := x^2 = 0

theorem square_of_real_is_positive_or_zero (x : ℝ) : (p x ∨ q x) :=
by
  sorry

end square_of_real_is_positive_or_zero_l582_58224


namespace factorization_a_minus_b_l582_58256

-- Define the problem in Lean 4
theorem factorization_a_minus_b (a b : ℤ) : 
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b))
  → a - b = 7 := 
by 
  sorry

end factorization_a_minus_b_l582_58256


namespace trucks_initial_count_l582_58255

theorem trucks_initial_count (x : ℕ) (h : x - 13 = 38) : x = 51 :=
by sorry

end trucks_initial_count_l582_58255


namespace exponential_function_condition_l582_58289

theorem exponential_function_condition (a : ℝ) (x : ℝ) 
  (h1 : a^2 - 5 * a + 5 = 1) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) : 
  a = 4 := 
sorry

end exponential_function_condition_l582_58289


namespace sum_of_coordinates_l582_58257

variable (f : ℝ → ℝ)

/-- Given that the point (2, 3) is on the graph of y = f(x) / 3,
    show that (9, 2/3) must be on the graph of y = f⁻¹(x) / 3 and the
    sum of its coordinates is 29/3. -/
theorem sum_of_coordinates (h : 3 = f 2 / 3) : (9 : ℝ) + (2 / 3 : ℝ) = 29 / 3 :=
by
  have h₁ : f 2 = 9 := by
    linarith
    
  have h₂ : f⁻¹ 9 = 2 := by
    -- We assume that f has an inverse and it is well-defined
    sorry

  have point_on_graph : (9, (2 / 3)) ∈ { p : ℝ × ℝ | p.2 = f⁻¹ p.1 / 3 } := by
    sorry

  show 9 + 2 / 3 = 29 / 3
  norm_num

end sum_of_coordinates_l582_58257


namespace probability_red_blue_green_l582_58217

def total_marbles : ℕ := 5 + 4 + 3 + 6
def favorable_marbles : ℕ := 5 + 4 + 3

theorem probability_red_blue_green : 
  (favorable_marbles : ℚ) / total_marbles = 2 / 3 := 
by 
  sorry

end probability_red_blue_green_l582_58217


namespace day_53_days_from_friday_l582_58252

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l582_58252


namespace ellipse_equation_l582_58242

-- Definitions from conditions
def ecc (e : ℝ) := e = Real.sqrt 3 / 2
def parabola_focus (c : ℝ) (a : ℝ) := c = Real.sqrt 3 ∧ a = 2
def b_val (b a c : ℝ) := b = Real.sqrt (a^2 - c^2)

-- Main problem statement
theorem ellipse_equation (e a b c : ℝ) (x y : ℝ) :
  ecc e → parabola_focus c a → b_val b a c → (x^2 + y^2 / 4 = 1) := 
by
  intros h1 h2 h3
  sorry

end ellipse_equation_l582_58242


namespace bill_money_left_l582_58295

def bill_remaining_money (merchantA_qty : Int) (merchantA_rate : Int) 
                        (merchantB_qty : Int) (merchantB_rate : Int)
                        (fine : Int) (merchantC_qty : Int) (merchantC_rate : Int) 
                        (protection_costs : Int) (passerby_qty : Int) 
                        (passerby_rate : Int) : Int :=
let incomeA := merchantA_qty * merchantA_rate
let incomeB := merchantB_qty * merchantB_rate
let incomeC := merchantC_qty * merchantC_rate
let incomeD := passerby_qty * passerby_rate
let total_income := incomeA + incomeB + incomeC + incomeD
let total_expenses := fine + protection_costs
total_income - total_expenses

theorem bill_money_left 
    (merchantA_qty : Int := 8) 
    (merchantA_rate : Int := 9) 
    (merchantB_qty : Int := 15) 
    (merchantB_rate : Int := 11) 
    (fine : Int := 80)
    (merchantC_qty : Int := 25) 
    (merchantC_rate : Int := 8) 
    (protection_costs : Int := 30) 
    (passerby_qty : Int := 12) 
    (passerby_rate : Int := 7) : 
    bill_remaining_money merchantA_qty merchantA_rate 
                         merchantB_qty merchantB_rate 
                         fine merchantC_qty merchantC_rate 
                         protection_costs passerby_qty 
                         passerby_rate = 411 := by 
  sorry

end bill_money_left_l582_58295


namespace problem_l582_58210

noncomputable def x : ℝ := 123.75
noncomputable def y : ℝ := 137.5
noncomputable def original_value : ℝ := 125

theorem problem (y_more : y = original_value + 0.1 * original_value) (x_less : x = y * 0.9) : y = 137.5 :=
by
  sorry

end problem_l582_58210


namespace range_of_x_l582_58244

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 2))) ↔ x > 2 :=
by
  sorry

end range_of_x_l582_58244


namespace inequality_solution_set_l582_58200

theorem inequality_solution_set (m : ℝ) : 
  (∀ (x : ℝ), m * x^2 - (1 - m) * x + m ≥ 0) ↔ m ≥ 1/3 := 
sorry

end inequality_solution_set_l582_58200


namespace square_of_volume_of_rect_box_l582_58286

theorem square_of_volume_of_rect_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 18) 
  (h3 : z * x = 10) : (x * y * z) ^ 2 = 2700 :=
sorry

end square_of_volume_of_rect_box_l582_58286


namespace batch_preparation_l582_58220

theorem batch_preparation (total_students cupcakes_per_student cupcakes_per_batch percent_not_attending : ℕ)
    (hlt1 : total_students = 150)
    (hlt2 : cupcakes_per_student = 3)
    (hlt3 : cupcakes_per_batch = 20)
    (hlt4 : percent_not_attending = 20)
    : (total_students * (80 / 100) * cupcakes_per_student) / cupcakes_per_batch = 18 := by
  sorry

end batch_preparation_l582_58220


namespace concert_attendance_difference_l582_58233

theorem concert_attendance_difference :
  let first_concert := 65899
  let second_concert := 66018
  second_concert - first_concert = 119 :=
by
  sorry

end concert_attendance_difference_l582_58233


namespace eggs_leftover_l582_58297

theorem eggs_leftover (d e f : ℕ) (total_eggs_per_carton : ℕ) 
  (h_d : d = 53) (h_e : e = 65) (h_f : f = 26) (h_carton : total_eggs_per_carton = 15) : (d + e + f) % total_eggs_per_carton = 9 :=
by {
  sorry
}

end eggs_leftover_l582_58297


namespace luna_total_monthly_budget_l582_58264

theorem luna_total_monthly_budget
  (H F phone_bill : ℝ)
  (h1 : F = 0.60 * H)
  (h2 : H + F = 240)
  (h3 : phone_bill = 0.10 * F) :
  H + F + phone_bill = 249 :=
by sorry

end luna_total_monthly_budget_l582_58264


namespace total_books_l582_58216

variable (M K G : ℕ)

-- Conditions
def Megan_books := 32
def Kelcie_books := Megan_books / 4
def Greg_books := 2 * Kelcie_books + 9

-- Theorem to prove
theorem total_books : Megan_books + Kelcie_books + Greg_books = 65 := by
  unfold Megan_books Kelcie_books Greg_books
  sorry

end total_books_l582_58216


namespace find_a1_general_term_sum_of_terms_l582_58262

-- Given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom h_condition : ∀ n, S n = (3 / 2) * a n - (1 / 2)

-- Specific condition for finding a1
axiom h_S1_eq_1 : S 1 = 1

-- Prove statements
theorem find_a1 : a 1 = 1 :=
by
  sorry

theorem general_term (n : ℕ) : n ≥ 1 → a n = 3 ^ (n - 1) :=
by
  sorry

theorem sum_of_terms (n : ℕ) : n ≥ 1 → S n = (3 ^ n - 1) / 2 :=
by
  sorry

end find_a1_general_term_sum_of_terms_l582_58262


namespace valid_n_values_l582_58231

variables (n x y : ℕ)

theorem valid_n_values :
  (n * (x - 3) = y + 3) ∧ (x + n = 3 * (y - n)) →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by
  sorry

end valid_n_values_l582_58231


namespace total_number_of_coins_l582_58215

theorem total_number_of_coins (x : ℕ) (h : 1 * x + 5 * x + 10 * x + 50 * x + 100 * x = 332) : 5 * x = 10 :=
by {
  sorry
}

end total_number_of_coins_l582_58215


namespace price_per_glass_on_second_day_l582_58267

 -- Definitions based on the conditions
def orangeade_first_day (O: ℝ) : ℝ := 2 * O -- Total volume on first day, O + O
def orangeade_second_day (O: ℝ) : ℝ := 3 * O -- Total volume on second day, O + 2O
def revenue_first_day (O: ℝ) (price_first_day: ℝ) : ℝ := 2 * O * price_first_day -- Revenue on first day
def revenue_second_day (O: ℝ) (P: ℝ) : ℝ := 3 * O * P -- Revenue on second day
def price_first_day: ℝ := 0.90 -- Given price per glass on the first day

 -- Statement to be proved
theorem price_per_glass_on_second_day (O: ℝ) (P: ℝ) (h: revenue_first_day O price_first_day = revenue_second_day O P) :
  P = 0.60 :=
by
  sorry

end price_per_glass_on_second_day_l582_58267


namespace bus_stop_time_l582_58280

-- Usual time to walk to the bus stop
def usual_time (T : ℕ) := T

-- Usual speed
def usual_speed (S : ℕ) := S

-- New speed when walking at 4/5 of usual speed
def new_speed (S : ℕ) := (4 * S) / 5

-- Time relationship when walking at new speed
def time_relationship (T : ℕ) (S : ℕ) := (S / ((4 * S) / 5)) = (T + 10) / T

-- Prove the usual time T is 40 minutes
theorem bus_stop_time (T S : ℕ) (h1 : time_relationship T S) : T = 40 :=
by
  sorry

end bus_stop_time_l582_58280


namespace chess_tournament_l582_58234

def number_of_players := 30

def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament : total_games number_of_players = 435 := by
  sorry

end chess_tournament_l582_58234


namespace restore_original_salary_l582_58228

theorem restore_original_salary (orig_salary : ℝ) (reducing_percent : ℝ) (increasing_percent : ℝ) :
  reducing_percent = 20 → increasing_percent = 25 →
  (orig_salary * (1 - reducing_percent / 100)) * (1 + increasing_percent / 100 / (1 - reducing_percent / 100)) = orig_salary
:= by
  intros
  sorry

end restore_original_salary_l582_58228


namespace smallest_x_y_sum_l582_58273

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l582_58273


namespace trigonometric_identity_l582_58213

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 4) :
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
sorry

end trigonometric_identity_l582_58213


namespace stratified_sampling_third_year_students_l582_58281

theorem stratified_sampling_third_year_students 
  (total_students : ℕ)
  (sample_size : ℕ)
  (ratio_1st : ℕ)
  (ratio_2nd : ℕ)
  (ratio_3rd : ℕ)
  (ratio_4th : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 200)
  (h3 : ratio_1st = 4)
  (h4 : ratio_2nd = 3)
  (h5 : ratio_3rd = 2)
  (h6 : ratio_4th = 1) :
  (ratio_3rd : ℚ) / (ratio_1st + ratio_2nd + ratio_3rd + ratio_4th : ℚ) * sample_size = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l582_58281


namespace polynomial_identity_l582_58275

open Function

-- Define the polynomial terms
def f1 (x : ℝ) := 2*x^5 + 4*x^3 + 3*x + 4
def f2 (x : ℝ) := x^4 - 2*x^3 + 3
def g (x : ℝ) := -2*x^5 + x^4 - 6*x^3 - 3*x - 1

-- Lean theorem statement
theorem polynomial_identity :
  ∀ x : ℝ, f1 x + g x = f2 x :=
by
  intros x
  sorry

end polynomial_identity_l582_58275


namespace laptop_weight_difference_is_3_67_l582_58238

noncomputable def karen_tote_weight : ℝ := 8
noncomputable def kevin_empty_briefcase_weight : ℝ := karen_tote_weight / 2
noncomputable def umbrella_weight : ℝ := kevin_empty_briefcase_weight / 2
noncomputable def briefcase_full_weight_rainy_day : ℝ := 2 * karen_tote_weight
noncomputable def work_papers_weight : ℝ := (briefcase_full_weight_rainy_day - umbrella_weight) / 6
noncomputable def laptop_weight : ℝ := briefcase_full_weight_rainy_day - umbrella_weight - work_papers_weight
noncomputable def weight_difference : ℝ := laptop_weight - karen_tote_weight

theorem laptop_weight_difference_is_3_67 : weight_difference = 3.67 := by
  sorry

end laptop_weight_difference_is_3_67_l582_58238


namespace func_has_extrema_l582_58249

theorem func_has_extrema (a b c : ℝ) (h_a_nonzero : a ≠ 0) (h_discriminant_positive : b^2 + 8 * a * c > 0) 
    (h_pos_sum_roots : b / a > 0) (h_pos_product_roots : -2 * c / a > 0) : 
    (a * b > 0) ∧ (a * c < 0) :=
by 
  -- Proof skipped.
  sorry

end func_has_extrema_l582_58249


namespace greatest_common_divisor_sum_arithmetic_sequence_l582_58278

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l582_58278


namespace katie_clock_l582_58241

theorem katie_clock (t_clock t_actual : ℕ) :
  t_clock = 540 →
  t_actual = (540 * 60) / 37 →
  8 * 60 + 875 = 22 * 60 + 36 :=
by
  intros h1 h2
  have h3 : 875 = (540 * 60 / 37) := sorry
  have h4 : 8 * 60 + 875 = 480 + 875 := sorry
  have h5 : 480 + 875 = 22 * 60 + 36 := sorry
  exact h5

end katie_clock_l582_58241


namespace f_2010_eq_0_l582_58269

theorem f_2010_eq_0 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, f (x + 2) = f x) : 
  f 2010 = 0 :=
by sorry

end f_2010_eq_0_l582_58269


namespace jellybeans_to_buy_l582_58223

-- Define the conditions: a minimum of 150 jellybeans and a remainder of 15 when divided by 17.
def condition (n : ℕ) : Prop :=
  n ≥ 150 ∧ n % 17 = 15

-- Define the main statement to prove: if condition holds, then n is 151
theorem jellybeans_to_buy (n : ℕ) (h : condition n) : n = 151 :=
by
  -- Proof is skipped with sorry
  sorry

end jellybeans_to_buy_l582_58223


namespace octagon_reflected_arcs_area_l582_58248

theorem octagon_reflected_arcs_area :
  let s := 2
  let θ := 45
  let r := 2 / Real.sqrt (2 - Real.sqrt (2))
  let sector_area := θ / 360 * Real.pi * r^2
  let total_arc_area := 8 * sector_area
  let circle_area := Real.pi * r^2
  let bounded_region_area := 8 * (circle_area - 2 * Real.sqrt (2) * 1 / 2)
  bounded_region_area = (16 * Real.sqrt 2 / 3 - Real.pi)
:= sorry

end octagon_reflected_arcs_area_l582_58248


namespace num_positive_integers_l582_58222

theorem num_positive_integers (m : ℕ) : 
  (∃ n, m^2 - 2 = n ∧ n ∣ 2002) ↔ (m = 2 ∨ m = 3 ∨ m = 4) :=
by
  sorry

end num_positive_integers_l582_58222


namespace find_number_l582_58245

theorem find_number (x : ℝ) (h : 0.9 * x = 0.0063) : x = 0.007 := 
by {
  sorry
}

end find_number_l582_58245


namespace eval_expression_pow_i_l582_58205

theorem eval_expression_pow_i :
  i^(12345 : ℤ) + i^(12346 : ℤ) + i^(12347 : ℤ) + i^(12348 : ℤ) = (0 : ℂ) :=
by
  -- Since this statement doesn't need the full proof, we use sorry to leave it open 
  sorry

end eval_expression_pow_i_l582_58205


namespace initial_number_is_nine_l582_58221

theorem initial_number_is_nine (x : ℝ) (h : 3 * (2 * x + 13) = 93) : x = 9 :=
sorry

end initial_number_is_nine_l582_58221


namespace servings_left_proof_l582_58293

-- Define the number of servings prepared
def total_servings : ℕ := 61

-- Define the number of guests
def total_guests : ℕ := 8

-- Define the fraction of servings the first 3 guests shared
def first_three_fraction : ℚ := 2 / 5

-- Define the fraction of servings the next 4 guests shared
def next_four_fraction : ℚ := 1 / 4

-- Define the number of servings consumed by the 8th guest
def eighth_guest_servings : ℕ := 5

-- Total consumed servings by the first three guests (rounded down)
def first_three_consumed := (first_three_fraction * total_servings).floor

-- Total consumed servings by the next four guests (rounded down)
def next_four_consumed := (next_four_fraction * total_servings).floor

-- Total consumed servings in total
def total_consumed := first_three_consumed + next_four_consumed + eighth_guest_servings

-- The number of servings left unconsumed
def servings_left_unconsumed := total_servings - total_consumed

-- The theorem stating there are 17 servings left unconsumed
theorem servings_left_proof : servings_left_unconsumed = 17 := by
  sorry

end servings_left_proof_l582_58293


namespace find_vector_v_l582_58232

def vector3 := ℝ × ℝ × ℝ

def cross_product (u v : vector3) : vector3 :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1  - u.1   * v.2.2,
   u.1   * v.2.1 - u.2.1 * v.1)

def a : vector3 := (1, 2, 1)
def b : vector3 := (2, 0, -1)
def v : vector3 := (3, 2, 0)
def b_cross_a : vector3 := (2, 3, 4)
def a_cross_b : vector3 := (-2, 3, -4)

theorem find_vector_v :
  cross_product v a = b_cross_a ∧ cross_product v b = a_cross_b :=
sorry

end find_vector_v_l582_58232


namespace andrey_travel_distance_l582_58240

theorem andrey_travel_distance:
  ∃ s t: ℝ, 
    (s = 60 * (t + 4/3) + 20  ∧ s = 90 * (t - 1/3) + 60) ∧ s = 180 :=
by
  sorry

end andrey_travel_distance_l582_58240


namespace expected_number_of_digits_is_1_55_l582_58299

/-- Brent rolls a fair icosahedral die with numbers 1 through 20 on its faces -/
noncomputable def expectedNumberOfDigits : ℚ :=
  let P_one_digit := 9 / 20
  let P_two_digit := 11 / 20
  (P_one_digit * 1) + (P_two_digit * 2)

/-- The expected number of digits Brent will roll is 1.55 -/
theorem expected_number_of_digits_is_1_55 : expectedNumberOfDigits = 1.55 := by
  sorry

end expected_number_of_digits_is_1_55_l582_58299


namespace trig_intersection_identity_l582_58207

theorem trig_intersection_identity (x0 : ℝ) (hx0 : x0 ≠ 0) (htan : -x0 = Real.tan x0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
sorry

end trig_intersection_identity_l582_58207


namespace class_mean_calculation_correct_l582_58263

variable (s1 s2 : ℕ) (mean1 mean2 : ℕ)
variable (n : ℕ) (mean_total : ℕ)

def overall_class_mean (s1 s2 mean1 mean2 : ℕ) : ℕ :=
  let total_score := (s1 * mean1) + (s2 * mean2)
  total_score / (s1 + s2)

theorem class_mean_calculation_correct
  (h1 : s1 = 40)
  (h2 : s2 = 10)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : n = 50)
  (h6 : mean_total = 82) :
  overall_class_mean s1 s2 mean1 mean2 = mean_total :=
  sorry

end class_mean_calculation_correct_l582_58263


namespace merchant_articles_l582_58270

theorem merchant_articles (N CP SP : ℝ) 
  (h1 : N * CP = 16 * SP)
  (h2 : SP = CP * 1.375) : 
  N = 22 :=
by
  sorry

end merchant_articles_l582_58270


namespace cost_per_charge_l582_58271

theorem cost_per_charge
  (charges : ℕ) (budget left : ℝ) (cost_per_charge : ℝ)
  (charges_eq : charges = 4)
  (budget_eq : budget = 20)
  (left_eq : left = 6) :
  cost_per_charge = (budget - left) / charges :=
by
  apply sorry

end cost_per_charge_l582_58271


namespace sin_inequality_iff_angle_inequality_l582_58283

section
variables {A B : ℝ} {a b : ℝ} (R : ℝ) (hA : A = Real.sin a) (hB : B = Real.sin b)

theorem sin_inequality_iff_angle_inequality (A B : ℝ) :
  (A > B) ↔ (Real.sin A > Real.sin B) :=
sorry
end

end sin_inequality_iff_angle_inequality_l582_58283


namespace OilBillJanuary_l582_58296

theorem OilBillJanuary (J F : ℝ) (h1 : F / J = 5 / 4) (h2 : (F + 30) / J = 3 / 2) : J = 120 := by
  sorry

end OilBillJanuary_l582_58296


namespace range_of_a_l582_58284

theorem range_of_a (a : ℝ) : 
  (∀ P Q : ℝ × ℝ, P ≠ Q ∧ P.snd = a * P.fst ^ 2 - 1 ∧ Q.snd = a * Q.fst ^ 2 - 1 ∧ 
  P.fst + P.snd = -(Q.fst + Q.snd)) →
  a > 3 / 4 :=
by
  sorry

end range_of_a_l582_58284


namespace pencils_left_l582_58277

-- Define initial count of pencils
def initial_pencils : ℕ := 20

-- Define pencils misplaced
def misplaced_pencils : ℕ := 7

-- Define pencils broken and thrown away
def broken_pencils : ℕ := 3

-- Define pencils found
def found_pencils : ℕ := 4

-- Define pencils bought
def bought_pencils : ℕ := 2

-- Define the final number of pencils
def final_pencils: ℕ := initial_pencils - misplaced_pencils - broken_pencils + found_pencils + bought_pencils

-- Prove that the final number of pencils is 16
theorem pencils_left : final_pencils = 16 :=
by
  -- The proof steps are omitted here
  sorry

end pencils_left_l582_58277


namespace periodic_function_with_period_sqrt2_l582_58285

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Definition of symmetry about x = sqrt(2)/2
def is_symmetric_about_line (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

-- Main theorem to prove
theorem periodic_function_with_period_sqrt2 (f : ℝ → ℝ) :
  is_even_function f → is_symmetric_about_line f (Real.sqrt 2 / 2) → ∃ T, T = Real.sqrt 2 ∧ ∀ x, f (x + T) = f x :=
by
  sorry

end periodic_function_with_period_sqrt2_l582_58285


namespace percentage_error_in_calculated_area_l582_58201

theorem percentage_error_in_calculated_area
  (a : ℝ)
  (measured_side_length : ℝ := 1.025 * a) :
  (measured_side_length ^ 2 - a ^ 2) / (a ^ 2) * 100 = 5.0625 :=
by 
  sorry

end percentage_error_in_calculated_area_l582_58201


namespace plate_arrangement_l582_58268

def arrangements_without_restriction : Nat :=
  Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 3)

def arrangements_adjacent_green : Nat :=
  (Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3)) * Nat.factorial 3

def allowed_arrangements : Nat :=
  arrangements_without_restriction - arrangements_adjacent_green

theorem plate_arrangement : 
  allowed_arrangements = 2520 := 
by
  sorry

end plate_arrangement_l582_58268


namespace chicken_burger_cost_l582_58247

namespace BurgerCost

variables (C B : ℕ)

theorem chicken_burger_cost (h1 : B = C + 300) 
                            (h2 : 3 * B + 3 * C = 21000) : 
                            C = 3350 := 
sorry

end BurgerCost

end chicken_burger_cost_l582_58247


namespace parrots_fraction_l582_58291

variable (P T : ℚ) -- P: fraction of parrots, T: fraction of toucans

def fraction_parrots (P T : ℚ) : Prop :=
  P + T = 1 ∧
  (2 / 3) * P + (1 / 4) * T = 0.5

theorem parrots_fraction (P T : ℚ) (h : fraction_parrots P T) : P = 3 / 5 :=
by
  sorry

end parrots_fraction_l582_58291


namespace second_machine_finishes_in_10_minutes_l582_58236

-- Definitions for the conditions:
def time_to_clear_by_first_machine (t : ℝ) : Prop := t = 1
def time_to_clear_by_second_machine (t : ℝ) : Prop := t = 3 / 4
def time_first_machine_works (t : ℝ) : Prop := t = 1 / 3
def remaining_time (t : ℝ) : Prop := t = 1 / 6

-- Theorem statement:
theorem second_machine_finishes_in_10_minutes (t₁ t₂ t₃ t₄ : ℝ) 
  (h₁ : time_to_clear_by_first_machine t₁) 
  (h₂ : time_to_clear_by_second_machine t₂) 
  (h₃ : time_first_machine_works t₃) 
  (h₄ : remaining_time t₄) 
  : t₄ = 1 / 6 → t₄ * 60 = 10 := 
by
  -- here we can provide the proof steps, but the task does not require the proof
  sorry

end second_machine_finishes_in_10_minutes_l582_58236


namespace employee_selection_l582_58298

theorem employee_selection
  (total_employees : ℕ)
  (under_35 : ℕ)
  (between_35_and_49 : ℕ)
  (over_50 : ℕ)
  (selected_employees : ℕ) :
  total_employees = 500 →
  under_35 = 125 →
  between_35_and_49 = 280 →
  over_50 = 95 →
  selected_employees = 100 →
  (under_35 * selected_employees / total_employees = 25) ∧
  (between_35_and_49 * selected_employees / total_employees = 56) ∧
  (over_50 * selected_employees / total_employees = 19) := by
  intros h1 h2 h3 h4 h5
  sorry

end employee_selection_l582_58298


namespace final_price_is_correct_l582_58259

-- Define the original price and the discount rate
variable (a : ℝ)

-- The final price of the product after two 10% discounts
def final_price_after_discounts (a : ℝ) : ℝ :=
  a * (0.9 ^ 2)

-- Theorem stating the final price after two consecutive 10% discounts
theorem final_price_is_correct (a : ℝ) :
  final_price_after_discounts a = a * (0.9 ^ 2) :=
by sorry

end final_price_is_correct_l582_58259


namespace lcm_of_times_l582_58272

-- Define the times each athlete takes to complete one lap
def time_A : Nat := 4
def time_B : Nat := 5
def time_C : Nat := 6

-- Prove that the LCM of 4, 5, and 6 is 60
theorem lcm_of_times : Nat.lcm time_A (Nat.lcm time_B time_C) = 60 := by
  sorry

end lcm_of_times_l582_58272


namespace distinct_real_roots_of_quadratic_l582_58225

variable (m : ℝ)

theorem distinct_real_roots_of_quadratic (h1 : 4 + 4 * m > 0) (h2 : m ≠ 0) : m = 1 :=
by
  sorry

end distinct_real_roots_of_quadratic_l582_58225


namespace min_value_of_mn_squared_l582_58243

theorem min_value_of_mn_squared 
  (a b c : ℝ) 
  (h : a^2 + b^2 = c^2) 
  (m n : ℝ) 
  (h_point : a * m + b * n + 2 * c = 0) : 
  m^2 + n^2 = 4 :=
sorry

end min_value_of_mn_squared_l582_58243


namespace eval_expression_l582_58246

theorem eval_expression :
  16^3 + 3 * (16^2) * 2 + 3 * 16 * (2^2) + 2^3 = 5832 :=
by
  sorry

end eval_expression_l582_58246


namespace percent_less_than_m_plus_d_l582_58239

-- Define the given conditions
variables (m d : ℝ) (distribution : ℝ → ℝ)

-- Assume the distribution is symmetric about the mean m
axiom symmetric_distribution :
  ∀ x, distribution (m + x) = distribution (m - x)

-- 84 percent of the distribution lies within one standard deviation d of the mean
axiom within_one_sd :
  ∫ x in -d..d, distribution (m + x) = 0.84

-- The goal is to prove that 42 percent of the distribution is less than m + d
theorem percent_less_than_m_plus_d : 
  ( ∫ x in -d..0, distribution (m + x) ) = 0.42 :=
by 
  sorry

end percent_less_than_m_plus_d_l582_58239


namespace possible_n_values_l582_58253

theorem possible_n_values (x y n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : n > 0)
  (top_box_eq : x * y * n^2 = 720) :
  ∃ k : ℕ,  k = 6 :=
by 
  sorry

end possible_n_values_l582_58253


namespace kim_monthly_revenue_l582_58251

-- Define the cost to open the store
def initial_cost : ℤ := 25000

-- Define the monthly expenses
def monthly_expenses : ℤ := 1500

-- Define the number of months
def months : ℕ := 10

-- Define the revenue per month
def revenue_per_month (total_revenue : ℤ) (months : ℕ) : ℤ := total_revenue / months

theorem kim_monthly_revenue :
  ∃ r, revenue_per_month r months = 4000 :=
by 
  let total_expenses := monthly_expenses * months
  let total_revenue := initial_cost + total_expenses
  use total_revenue
  unfold revenue_per_month
  sorry

end kim_monthly_revenue_l582_58251


namespace camel_steps_divisibility_l582_58260

variables (A B : Type) (p q : ℕ)

-- Description of the conditions
-- let A, B be vertices
-- p and q be the steps to travel from A to B in different paths

theorem camel_steps_divisibility (h1: ∃ r : ℕ, p + r ≡ 0 [MOD 3])
                                  (h2: ∃ r : ℕ, q + r ≡ 0 [MOD 3]) : (p - q) % 3 = 0 := by
  sorry

end camel_steps_divisibility_l582_58260
