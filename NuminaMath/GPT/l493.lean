import Mathlib

namespace part_a_part_b_l493_49316

-- Definition for the number of triangles when the n-gon is divided using non-intersecting diagonals
theorem part_a (n : ℕ) (h : n ≥ 3) : 
  ∃ k, k = n - 2 := 
sorry

-- Definition for the number of diagonals when the n-gon is divided using non-intersecting diagonals
theorem part_b (n : ℕ) (h : n ≥ 3) : 
  ∃ l, l = n - 3 := 
sorry

end part_a_part_b_l493_49316


namespace dealer_profit_percentage_l493_49327

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℝ) (sp_total : ℝ) (sp_count : ℝ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  let profit_percentage := (profit_per_article / cp_per_article) * 100
  profit_percentage

theorem dealer_profit_percentage :
  profit_percentage 25 15 38 12 = 89.99 := by
  sorry

end dealer_profit_percentage_l493_49327


namespace ratio_u_v_l493_49330

variables {u v : ℝ}
variables (u_lt_v : u < v)
variables (h_triangle : triangle 15 12 9)
variables (inscribed_circle : is_inscribed_circle 15 12 9 u v)

theorem ratio_u_v : u / v = 1 / 2 :=
sorry

end ratio_u_v_l493_49330


namespace finite_fraction_n_iff_l493_49389

theorem finite_fraction_n_iff (n : ℕ) (h_pos : 0 < n) :
  (∃ (a b : ℕ), n * (n + 1) = 2^a * 5^b) ↔ (n = 1 ∨ n = 4) :=
by
  sorry

end finite_fraction_n_iff_l493_49389


namespace exists_polynomial_distinct_powers_of_2_l493_49349

open Polynomial

variable (n : ℕ) (hn : n > 0)

theorem exists_polynomial_distinct_powers_of_2 :
  ∃ P : Polynomial ℤ, P.degree = n ∧ (∃ (k : Fin (n + 1) → ℕ), ∀ i j : Fin (n + 1), i ≠ j → 2 ^ k i ≠ 2 ^ k j ∧ (∀ i, P.eval i.val = 2 ^ k i)) :=
sorry

end exists_polynomial_distinct_powers_of_2_l493_49349


namespace find_u_minus_v_l493_49374

theorem find_u_minus_v (u v : ℚ) (h1 : 5 * u - 6 * v = 31) (h2 : 3 * u + 5 * v = 4) : u - v = 5.3 := by
  sorry

end find_u_minus_v_l493_49374


namespace value_of_x_in_logarithm_equation_l493_49343

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_x_in_logarithm_equation (n : ℝ) (h1 : n = 343) : 
  ∃ (x : ℝ), log_base x n + log_base 7 n = log_base 1 n :=
by
  sorry

end value_of_x_in_logarithm_equation_l493_49343


namespace number_proportion_l493_49341

theorem number_proportion (number : ℚ) :
  (number : ℚ) / 12 = 9 / 360 →
  number = 0.3 :=
by
  intro h
  sorry

end number_proportion_l493_49341


namespace cylinder_volume_in_sphere_l493_49331

theorem cylinder_volume_in_sphere 
  (h_c : ℝ) (d_s : ℝ) : 
  (h_c = 1) → (d_s = 2) → 
  (π * (d_s / 2)^2 * (h_c / 2) = π / 2) :=
by 
  intros h_c_eq h_s_eq
  sorry

end cylinder_volume_in_sphere_l493_49331


namespace no_intersection_l493_49344

def M := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
def N (a : ℝ) := { p : ℝ × ℝ | abs (p.1 - 1) + abs (p.2 - 1) = a }

theorem no_intersection (a : ℝ) : M ∩ (N a) = ∅ ↔ a ∈ (Set.Ioo (2-Real.sqrt 2) (2+Real.sqrt 2)) := 
by 
  sorry

end no_intersection_l493_49344


namespace sequence_a_n_l493_49304

-- Given conditions from the problem
variable {a : ℕ → ℕ}
variable (S : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequence is given by S_n
axiom sum_Sn : ∀ n : ℕ, n > 0 → S n = 2 * n * n

-- Definition of a_n, the nth term of the sequence
def a_n (n : ℕ) : ℕ :=
  if n = 1 then
    S 1
  else
    S n - S (n - 1)

-- Prove that a_n = 4n - 2 for all n > 0.
theorem sequence_a_n (n : ℕ) (h : n > 0) : a_n S n = 4 * n - 2 :=
by
  sorry

end sequence_a_n_l493_49304


namespace largest_divisor_of_product_of_5_consecutive_integers_l493_49398

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l493_49398


namespace CoastalAcademy_absent_percentage_l493_49307

theorem CoastalAcademy_absent_percentage :
  ∀ (total_students boys girls : ℕ) (absent_boys_ratio absent_girls_ratio : ℚ),
    total_students = 120 →
    boys = 70 →
    girls = 50 →
    absent_boys_ratio = 1/7 →
    absent_girls_ratio = 1/5 →
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    absent_percentage = 16.67 :=
  by
    intros total_students boys girls absent_boys_ratio absent_girls_ratio
           h1 h2 h3 h4 h5
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    sorry

end CoastalAcademy_absent_percentage_l493_49307


namespace exponent_proof_l493_49393

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l493_49393


namespace gas_pressure_in_final_container_l493_49314

variable (k : ℝ) (p_initial p_second p_final : ℝ) (v_initial v_second v_final v_half : ℝ)

theorem gas_pressure_in_final_container 
  (h1 : v_initial = 3.6)
  (h2 : p_initial = 6)
  (h3 : v_second = 7.2)
  (h4 : v_final = 3.6)
  (h5 : v_half = v_second / 2)
  (h6 : p_initial * v_initial = k)
  (h7 : p_second * v_second = k)
  (h8 : p_final * v_final = k) :
  p_final = 6 := 
sorry

end gas_pressure_in_final_container_l493_49314


namespace only_solution_is_two_l493_49386

theorem only_solution_is_two :
  ∀ n : ℕ, (Nat.Prime (n^n + 1) ∧ Nat.Prime ((2*n)^(2*n) + 1)) → n = 2 :=
by
  sorry

end only_solution_is_two_l493_49386


namespace complex_prod_eq_l493_49306

theorem complex_prod_eq (x y z : ℂ) (h1 : x * y + 6 * y = -24) (h2 : y * z + 6 * z = -24) (h3 : z * x + 6 * x = -24) :
  x * y * z = 144 :=
by
  sorry

end complex_prod_eq_l493_49306


namespace number_of_sad_children_l493_49397

-- Definitions of the given conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20

-- The main statement to be proved
theorem number_of_sad_children : 
  total_children - happy_children - neither_happy_nor_sad_children = 10 := 
by 
  sorry

end number_of_sad_children_l493_49397


namespace sheilas_hours_mwf_is_24_l493_49383

-- Define Sheila's earning conditions and working hours
def sheilas_hours_mwf (H : ℕ) : Prop :=
  let hours_tu_th := 6 * 2
  let earnings_tu_th := hours_tu_th * 14
  let earnings_mwf := 504 - earnings_tu_th
  H = earnings_mwf / 14

-- The theorem to state that Sheila works 24 hours on Monday, Wednesday, and Friday
theorem sheilas_hours_mwf_is_24 : sheilas_hours_mwf 24 :=
by
  -- Proof is omitted
  sorry

end sheilas_hours_mwf_is_24_l493_49383


namespace lloyd_total_hours_worked_l493_49381

-- Conditions
def regular_hours_per_day : ℝ := 7.5
def regular_rate : ℝ := 4.5
def overtime_multiplier : ℝ := 2.5
def total_earnings : ℝ := 67.5

-- Proof problem
theorem lloyd_total_hours_worked :
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_earnings := regular_hours_per_day * regular_rate
  let earnings_from_overtime := total_earnings - regular_earnings
  let hours_of_overtime := earnings_from_overtime / overtime_rate
  let total_hours := regular_hours_per_day + hours_of_overtime
  total_hours = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l493_49381


namespace neg_forall_sin_gt_zero_l493_49347

theorem neg_forall_sin_gt_zero :
  ¬ (∀ x : ℝ, Real.sin x > 0) ↔ ∃ x : ℝ, Real.sin x ≤ 0 := 
sorry

end neg_forall_sin_gt_zero_l493_49347


namespace compute_HHHH_of_3_l493_49356

def H (x : ℝ) : ℝ := -0.5 * x^2 + 3 * x

theorem compute_HHHH_of_3 :
  H (H (H (H 3))) = 2.689453125 := by
  sorry

end compute_HHHH_of_3_l493_49356


namespace perimeter_rectangle_l493_49326

-- Defining the width and length of the rectangle based on the conditions
def width (a : ℝ) := a
def length (a : ℝ) := 2 * a + 1

-- Statement of the problem: proving the perimeter
theorem perimeter_rectangle (a : ℝ) :
  let W := width a
  let L := length a
  2 * W + 2 * L = 6 * a + 2 :=
by
  sorry

end perimeter_rectangle_l493_49326


namespace range_of_a_l493_49358

variable {x : ℝ} {a : ℝ}

theorem range_of_a (h : ∀ x : ℝ, ¬ (x^2 - 5*x + (5/4)*a > 0)) : 5 < a :=
by
  sorry

end range_of_a_l493_49358


namespace profit_percentage_is_60_l493_49318

variable (SellingPrice CostPrice : ℝ)

noncomputable def Profit : ℝ := SellingPrice - CostPrice

noncomputable def ProfitPercentage : ℝ := (Profit SellingPrice CostPrice / CostPrice) * 100

theorem profit_percentage_is_60
  (h1 : SellingPrice = 400)
  (h2 : CostPrice = 250) :
  ProfitPercentage SellingPrice CostPrice = 60 := by
  sorry

end profit_percentage_is_60_l493_49318


namespace set_intersection_complement_l493_49388

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}
def complement_B : Set ℝ := U \ B
def expected_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem set_intersection_complement :
  A ∩ complement_B = expected_set :=
by
  sorry

end set_intersection_complement_l493_49388


namespace area_of_triangle_is_23_over_10_l493_49322

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  1/2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_triangle_is_23_over_10 :
  let A : ℚ × ℚ := (3, 3)
  let B : ℚ × ℚ := (5, 3)
  let C : ℚ × ℚ := (21 / 5, 19 / 5)
  area_of_triangle A.1 A.2 B.1 B.2 C.1 C.2 = 23 / 10 :=
by
  sorry

end area_of_triangle_is_23_over_10_l493_49322


namespace f_shift_l493_49368

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem f_shift (x : ℝ) : f (x - 1) = x^2 - 4 * x + 3 := by
  sorry

end f_shift_l493_49368


namespace domain_f_l493_49321

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (x + 1) + (x - 2) ^ 0

theorem domain_f :
  (∃ x : ℝ, f x = f x) ↔ (∀ x, (x > -1 ∧ x ≠ 2) ↔ (x ∈ Ioo (-1 : ℝ) 2 ∨ x ∈ Ioi 2)) :=
by
  sorry

end domain_f_l493_49321


namespace simplify_expression_l493_49333

theorem simplify_expression : 1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_expression_l493_49333


namespace greatest_divisor_of_arithmetic_sequence_l493_49340

theorem greatest_divisor_of_arithmetic_sequence (x c : ℤ) (h_odd : x % 2 = 1) (h_even : c % 2 = 0) :
  15 ∣ (15 * (x + 7 * c)) :=
sorry

end greatest_divisor_of_arithmetic_sequence_l493_49340


namespace geometric_sequence_sixth_term_l493_49354

theorem geometric_sequence_sixth_term (a : ℕ) (a2 : ℝ) (aₖ : ℕ → ℝ) (r : ℝ) (k : ℕ) (h1 : a = 3) (h2 : a2 = -1/6) (h3 : ∀ n, aₖ n = a * r^(n-1)) (h4 : r = a2 / a) (h5 : k = 6) :
  aₖ k = -1 / 629856 :=
by sorry

end geometric_sequence_sixth_term_l493_49354


namespace part_I_solution_part_II_solution_l493_49335

-- Definitions for the problem
def f (x a : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I: When a = 2, solve the inequality f(x) < 4
theorem part_I_solution (x : ℝ) : f x 2 < 4 ↔ x > -1/2 ∧ x < 7/2 :=
by sorry

-- Part II: Range of values for a such that f(x) ≥ 2 for all x
theorem part_II_solution (a : ℝ) : (∀ x, f x a ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end part_I_solution_part_II_solution_l493_49335


namespace xiao_li_profit_l493_49378

noncomputable def original_price_per_share : ℝ := 21 / 1.05
noncomputable def closing_price_first_day : ℝ := original_price_per_share * 0.94
noncomputable def selling_price_second_day : ℝ := closing_price_first_day * 1.10
noncomputable def total_profit : ℝ := (selling_price_second_day - 21) * 5000

theorem xiao_li_profit :
  total_profit = 600 := sorry

end xiao_li_profit_l493_49378


namespace find_larger_number_l493_49396

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
sorry

end find_larger_number_l493_49396


namespace evaluate_expression_l493_49312

theorem evaluate_expression : 
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  a * b = 63 := 
by
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  sorry

end evaluate_expression_l493_49312


namespace meaningful_expr_l493_49338

theorem meaningful_expr (x : ℝ) : 
    (x + 1 ≥ 0 ∧ x - 2 ≠ 0) → (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end meaningful_expr_l493_49338


namespace smallest_divisible_12_13_14_l493_49399

theorem smallest_divisible_12_13_14 :
  ∃ n : ℕ, n > 0 ∧ (n % 12 = 0) ∧ (n % 13 = 0) ∧ (n % 14 = 0) ∧ n = 1092 := by
  sorry

end smallest_divisible_12_13_14_l493_49399


namespace global_chess_tournament_total_games_global_chess_tournament_player_wins_l493_49305

theorem global_chess_tournament_total_games (num_players : ℕ) (h200 : num_players = 200) :
  (num_players * (num_players - 1)) / 2 = 19900 := by
  sorry

theorem global_chess_tournament_player_wins (num_players losses : ℕ) 
  (h200 : num_players = 200) (h30 : losses = 30) :
  (num_players - 1) - losses = 169 := by
  sorry

end global_chess_tournament_total_games_global_chess_tournament_player_wins_l493_49305


namespace min_square_value_l493_49342

theorem min_square_value (a b : ℤ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ r : ℤ, r^2 = 15 * a + 16 * b)
  (h2 : ∃ s : ℤ, s^2 = 16 * a - 15 * b) : 
  231361 ≤ min (15 * a + 16 * b) (16 * a - 15 * b) :=
sorry

end min_square_value_l493_49342


namespace age_ratio_in_years_l493_49394

variable (s d x : ℕ)

theorem age_ratio_in_years (h1 : s - 3 = 2 * (d - 3)) (h2 : s - 7 = 3 * (d - 7)) (hx : (s + x) = 3 * (d + x) / 2) : x = 5 := sorry

end age_ratio_in_years_l493_49394


namespace range_of_a_l493_49375

-- Definitions of sets and the problem conditions
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}
def condition (a : ℝ) : Prop := P ∪ M a = P

-- The theorem stating what needs to be proven
theorem range_of_a (a : ℝ) (h : condition a) : -1 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l493_49375


namespace meeting_time_coincides_l493_49363

variables (distance_ab : ℕ) (speed_train_a : ℕ) (start_time_train_a : ℕ) (distance_at_9am : ℕ) (speed_train_b : ℕ) (start_time_train_b : ℕ)

def total_distance_ab := 465
def train_a_speed := 60
def train_b_speed := 75
def start_time_a := 8
def start_time_b := 9
def distance_train_a_by_9am := train_a_speed * (start_time_b - start_time_a)
def remaining_distance := total_distance_ab - distance_train_a_by_9am
def relative_speed := train_a_speed + train_b_speed
def time_to_meet := remaining_distance / relative_speed

theorem meeting_time_coincides :
  time_to_meet = 3 → (start_time_b + time_to_meet = 12) :=
by
  sorry

end meeting_time_coincides_l493_49363


namespace plant_ways_count_l493_49380

theorem plant_ways_count :
  ∃ (solutions : Finset (Fin 7 → ℕ)), 
    (∀ x ∈ solutions, (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 = 10) ∧ 
                       (100 * x 0 + 200 * x 1 + 300 * x 2 + 150 * x 3 + 125 * x 4 + 125 * x 5 = 2500)) ∧
    (solutions.card = 8) :=
sorry

end plant_ways_count_l493_49380


namespace problem_statement_l493_49339

noncomputable def a (n : ℕ) := n^2

theorem problem_statement (x : ℝ) (hx : x > 0) (n : ℕ) (hn : n > 0) :
  x + a n / x ^ n ≥ n + 1 :=
sorry

end problem_statement_l493_49339


namespace john_new_total_lifting_capacity_is_correct_l493_49348

def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50

def new_clean_and_jerk : ℕ := 2 * initial_clean_and_jerk
def new_snatch : ℕ := initial_snatch + (initial_snatch * 8 / 10)

def new_combined_total_capacity : ℕ := new_clean_and_jerk + new_snatch

theorem john_new_total_lifting_capacity_is_correct : 
  new_combined_total_capacity = 250 := by
  sorry

end john_new_total_lifting_capacity_is_correct_l493_49348


namespace subtract_base8_l493_49352

theorem subtract_base8 (a b : ℕ) (h₁ : a = 0o2101) (h₂ : b = 0o1245) :
  a - b = 0o634 := sorry

end subtract_base8_l493_49352


namespace skyscraper_anniversary_l493_49309

theorem skyscraper_anniversary (current_year_event future_happens_year target_anniversary_year : ℕ) :
  current_year_event + future_happens_year = target_anniversary_year - 5 →
  target_anniversary_year > current_year_event →
  future_happens_year = 95 := 
by
  sorry

-- Definitions for conditions:
def current_year_event := 100
def future_happens_year := 95
def target_anniversary_year := 200

end skyscraper_anniversary_l493_49309


namespace other_root_correct_l493_49324

noncomputable def other_root (p : ℝ) : ℝ :=
  let a := 3
  let c := -2
  let root1 := -1
  (-c / a) / root1

theorem other_root_correct (p : ℝ) (h_eq : 3 * (-1) ^ 2 + p * (-1) = 2) : other_root p = 2 / 3 :=
  by
    unfold other_root
    sorry

end other_root_correct_l493_49324


namespace inequality_solution_l493_49353

noncomputable def ratFunc (x : ℝ) : ℝ := 
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7))

theorem inequality_solution (x : ℝ) : 
  (ratFunc x > 0) ↔ 
  ((x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ (7 < x)) := 
by
  sorry

end inequality_solution_l493_49353


namespace joan_writing_time_l493_49337

theorem joan_writing_time
  (total_time : ℕ)
  (time_piano : ℕ)
  (time_reading : ℕ)
  (time_exerciser : ℕ)
  (h1 : total_time = 120)
  (h2 : time_piano = 30)
  (h3 : time_reading = 38)
  (h4 : time_exerciser = 27) : 
  total_time - (time_piano + time_reading + time_exerciser) = 25 :=
by
  sorry

end joan_writing_time_l493_49337


namespace number_of_teachers_l493_49367

theorem number_of_teachers (total_people : ℕ) (sampled_individuals : ℕ) (sampled_students : ℕ) 
    (school_total : total_people = 2400) 
    (sample_total : sampled_individuals = 160) 
    (sample_students : sampled_students = 150) : 
    ∃ teachers : ℕ, teachers = 150 := 
by
  -- Proof omitted
  sorry

end number_of_teachers_l493_49367


namespace fraction_of_loss_is_correct_l493_49370

-- Definitions based on the conditions
def selling_price : ℕ := 18
def cost_price : ℕ := 19

-- Calculating the loss
def loss : ℕ := cost_price - selling_price

-- Fraction of the loss compared to the cost price
def fraction_of_loss : ℚ := loss / cost_price

-- The theorem we want to prove
theorem fraction_of_loss_is_correct : fraction_of_loss = 1 / 19 := by
  sorry

end fraction_of_loss_is_correct_l493_49370


namespace satisfies_conditions_l493_49391

open Real

def point_P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

def condition1 (a : ℝ) : Prop := (point_P a).fst = 0

def condition2 (a : ℝ) : Prop := (point_P a).snd = 5

def condition3 (a : ℝ) : Prop := abs ((point_P a).fst) = abs ((point_P a).snd)

theorem satisfies_conditions :
  ∃ P : ℝ × ℝ, P = (12, 12) ∨ P = (-12, -12) ∨ P = (4, -4) ∨ P = (-4, 4) :=
by
  sorry

end satisfies_conditions_l493_49391


namespace sum_of_squares_eq_two_l493_49366

theorem sum_of_squares_eq_two {a b : ℝ} (h : (a^2 + b^2) * (a^2 + b^2 + 4) = 12) : a^2 + b^2 = 2 := sorry

end sum_of_squares_eq_two_l493_49366


namespace value_of_m_l493_49376

theorem value_of_m 
  (m : ℝ)
  (h1 : |m - 1| = 1)
  (h2 : m - 2 ≠ 0) : 
  m = 0 :=
  sorry

end value_of_m_l493_49376


namespace find_x_plus_y_l493_49365

theorem find_x_plus_y
  (x y : ℝ)
  (hx : x^3 - 3 * x^2 + 5 * x - 17 = 0)
  (hy : y^3 - 3 * y^2 + 5 * y + 11 = 0) :
  x + y = 2 := 
sorry

end find_x_plus_y_l493_49365


namespace smallest_y_absolute_value_equation_l493_49317

theorem smallest_y_absolute_value_equation :
  ∃ y : ℚ, (|5 * y - 9| = 55) ∧ y = -46 / 5 :=
by
  sorry

end smallest_y_absolute_value_equation_l493_49317


namespace opposite_of_6_is_neg_6_l493_49345

theorem opposite_of_6_is_neg_6 : -6 = -6 := by
  sorry

end opposite_of_6_is_neg_6_l493_49345


namespace johns_friends_count_l493_49300

-- Define the conditions
def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

-- Define the theorem to prove the number of friends John is going with
theorem johns_friends_count (total_cost cost_per_person : ℕ) (h1 : total_cost = 12100) (h2 : cost_per_person = 1100) : (total_cost / cost_per_person) - 1 = 10 := by
  -- Providing the proof is not required, so we use sorry to skip it
  sorry

end johns_friends_count_l493_49300


namespace correct_subtraction_l493_49359

/-- Given a number n where subtracting 63 results in 8,
we aim to find the result of subtracting 36 from n
and proving that the result is 35. -/
theorem correct_subtraction (n : ℕ) (h : n - 63 = 8) : n - 36 = 35 :=
by
  sorry

end correct_subtraction_l493_49359


namespace value_of_6z_l493_49308

theorem value_of_6z (x y z : ℕ) (h1 : 6 * z = 2 * x) (h2 : x + y + z = 26) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 6 * z = 36 :=
by
  sorry

end value_of_6z_l493_49308


namespace range_of_m_plus_n_l493_49329

theorem range_of_m_plus_n (m n : ℝ)
  (tangent_condition : (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 = 1)) :
  m + n ∈ (Set.Iic (2 - 2*Real.sqrt 2) ∪ Set.Ici (2 + 2*Real.sqrt 2)) :=
sorry

end range_of_m_plus_n_l493_49329


namespace triple_g_eq_nineteen_l493_49355

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 3 else 2 * n + 1

theorem triple_g_eq_nineteen : g (g (g 1)) = 19 := by
  sorry

end triple_g_eq_nineteen_l493_49355


namespace students_tried_out_l493_49332

theorem students_tried_out (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ)
  (h1 : not_picked = 36) (h2 : groups = 4) (h3 : students_per_group = 7) :
  not_picked + groups * students_per_group = 64 :=
by
  sorry

end students_tried_out_l493_49332


namespace work_efficiency_ratio_l493_49301

-- Define the problem conditions and the ratio we need to prove.
theorem work_efficiency_ratio :
  (∃ (a b : ℝ), b = 1 / 18 ∧ (a + b) = 1 / 12 ∧ (a / b) = 1 / 2) :=
by {
  -- Definitions and variables can be listed if necessary
  -- a : ℝ
  -- b : ℝ
  -- Assume conditions
  sorry
}

end work_efficiency_ratio_l493_49301


namespace sum_of_squares_of_roots_eq_14_l493_49336

theorem sum_of_squares_of_roots_eq_14 {α β γ : ℝ}
  (h1: ∀ x: ℝ, (x^3 - 6*x^2 + 11*x - 6 = 0) → (x = α ∨ x = β ∨ x = γ)) :
  α^2 + β^2 + γ^2 = 14 :=
by
  sorry

end sum_of_squares_of_roots_eq_14_l493_49336


namespace find_subtracted_number_l493_49315

theorem find_subtracted_number (x y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  have hx : 2 * 129 - y = 110 := by
    rw [h1] at h2
    exact h2
  linarith

end find_subtracted_number_l493_49315


namespace even_function_iff_b_zero_l493_49360

theorem even_function_iff_b_zero (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c) = ((-x)^2 + b * (-x) + c)) ↔ b = 0 :=
by
  sorry

end even_function_iff_b_zero_l493_49360


namespace rotated_squares_overlap_area_l493_49387

noncomputable def total_overlap_area (side_length : ℝ) : ℝ :=
  let base_area := side_length ^ 2
  3 * base_area

theorem rotated_squares_overlap_area : total_overlap_area 8 = 192 := by
  sorry

end rotated_squares_overlap_area_l493_49387


namespace value_of_f_at_neg_one_l493_49302

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (x : ℝ) (h : x ≠ 0) : ℝ := (2 - 3 * x^2) / x^2

theorem value_of_f_at_neg_one : f (-1) (by norm_num) = -1 := 
sorry

end value_of_f_at_neg_one_l493_49302


namespace delta_max_success_ratio_l493_49320

theorem delta_max_success_ratio (y w x z : ℤ) (h1 : 360 + 240 = 600)
  (h2 : 0 < x ∧ x < y ∧ z < w)
  (h3 : y + w = 600)
  (h4 : (x : ℚ) / y < (200 : ℚ) / 360)
  (h5 : (z : ℚ) / w < (160 : ℚ) / 240)
  (h6 : (360 : ℚ) / 600 = 3 / 5)
  (h7 : (x + z) < 166) :
  (x + z : ℚ) / 600 ≤ 166 / 600 := 
sorry

end delta_max_success_ratio_l493_49320


namespace smallest_n_satisfies_conditions_l493_49385

/-- 
There exists a smallest positive integer n such that 5n is a perfect square 
and 3n is a perfect cube, and that n is 1125.
-/
theorem smallest_n_satisfies_conditions :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 5 * n = k^2) ∧ (∃ m : ℕ, 3 * n = m^3) ∧ n = 1125 := 
by
  sorry

end smallest_n_satisfies_conditions_l493_49385


namespace f_15_equals_227_l493_49361

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem f_15_equals_227 : f 15 = 227 := by
  sorry

end f_15_equals_227_l493_49361


namespace evaluate_expression_l493_49390

theorem evaluate_expression : 2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end evaluate_expression_l493_49390


namespace arctan_sum_property_l493_49357

open Real

theorem arctan_sum_property (x y z : ℝ) :
  arctan x + arctan y + arctan z = π / 2 → x * y + y * z + x * z = 1 :=
by
  sorry

end arctan_sum_property_l493_49357


namespace jane_payment_per_bulb_l493_49362

theorem jane_payment_per_bulb :
  let tulip_bulbs := 20
  let iris_bulbs := tulip_bulbs / 2
  let daffodil_bulbs := 30
  let crocus_bulbs := 3 * daffodil_bulbs
  let total_bulbs := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let total_earned := 75
  let payment_per_bulb := total_earned / total_bulbs
  payment_per_bulb = 0.50 := 
by
  sorry

end jane_payment_per_bulb_l493_49362


namespace max_value_200_max_value_attained_l493_49310

noncomputable def max_value (X Y Z : ℕ) : ℕ := 
  X * Y * Z + X * Y + Y * Z + Z * X

theorem max_value_200 (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  max_value X Y Z ≤ 200 :=
sorry

theorem max_value_attained (X Y Z : ℕ) (h : X = 5) (h1 : Y = 5) (h2 : Z = 5) : 
  max_value X Y Z = 200 :=
sorry

end max_value_200_max_value_attained_l493_49310


namespace smallest_positive_integer_l493_49313

theorem smallest_positive_integer (n : ℕ) : 
  (∃ m : ℕ, (4410 * n = m^2)) → n = 10 := 
by
  sorry

end smallest_positive_integer_l493_49313


namespace uncle_jerry_total_tomatoes_l493_49392

def tomatoes_reaped_yesterday : ℕ := 120
def tomatoes_reaped_more_today : ℕ := 50

theorem uncle_jerry_total_tomatoes : 
  tomatoes_reaped_yesterday + (tomatoes_reaped_yesterday + tomatoes_reaped_more_today) = 290 :=
by 
  sorry

end uncle_jerry_total_tomatoes_l493_49392


namespace total_cookies_l493_49382

theorem total_cookies (x y : Nat) (h1 : x = 137) (h2 : y = 251) : x * y = 34387 := by
  sorry

end total_cookies_l493_49382


namespace set_intersection_l493_49384

theorem set_intersection :
  {x : ℝ | -4 < x ∧ x < 2} ∩ {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l493_49384


namespace maximum_height_l493_49379

noncomputable def h (t : ℝ) : ℝ :=
  -20 * t ^ 2 + 100 * t + 30

theorem maximum_height : 
  ∃ t : ℝ, h t = 155 ∧ ∀ t' : ℝ, h t' ≤ 155 := 
sorry

end maximum_height_l493_49379


namespace alyosha_possible_l493_49334

theorem alyosha_possible (current_date : ℕ) (day_before_yesterday_age current_year_age next_year_age : ℕ) : 
  (next_year_age = 12 ∧ day_before_yesterday_age = 9 ∧ current_year_age = 12 - 1)
  → (current_date = 1 ∧ current_year_age = 11 → (∃ bday : ℕ, bday = 31)) := 
by
  sorry

end alyosha_possible_l493_49334


namespace find_second_bag_weight_l493_49364

variable (initialWeight : ℕ) (firstBagWeight : ℕ) (totalWeight : ℕ)

theorem find_second_bag_weight 
  (h1: initialWeight = 15)
  (h2: firstBagWeight = 15)
  (h3: totalWeight = 40) :
  totalWeight - (initialWeight + firstBagWeight) = 10 :=
  sorry

end find_second_bag_weight_l493_49364


namespace car_catches_truck_in_7_hours_l493_49319

-- Definitions based on the conditions
def initial_distance := 175 -- initial distance in kilometers
def truck_speed := 40 -- speed of the truck in km/h
def car_initial_speed := 50 -- initial speed of the car in km/h
def car_speed_increase := 5 -- speed increase per hour for the car in km/h

-- The main statement to prove
theorem car_catches_truck_in_7_hours :
  ∃ n : ℕ, (n ≥ 0) ∧ 
  (car_initial_speed - truck_speed) * n + (car_speed_increase * n * (n - 1) / 2) = initial_distance :=
by
  existsi 7
  -- Check the equation for n = 7
  -- Simplify: car initial extra speed + sum of increase terms should equal initial distance
  -- (50 - 40) * 7 + 5 * 7 * 6 / 2 = 175
  -- (10) * 7 + 35 * 3 / 2 = 175
  -- 70 + 105 = 175
  sorry

end car_catches_truck_in_7_hours_l493_49319


namespace range_of_k_l493_49311

theorem range_of_k (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + k * x + 1 / 2 ≥ 0) → k ∈ Set.Ioc 0 4 := 
by 
  sorry

end range_of_k_l493_49311


namespace initial_ratio_l493_49369

-- Define the initial number of horses and cows
def initial_horses (H : ℕ) : Prop := H = 120
def initial_cows (C : ℕ) : Prop := C = 20

-- Define the conditions of the problem
def condition1 (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)
def condition2 (H C : ℕ) : Prop := H - 15 = C + 15 + 70

-- The statement that initial ratio is 6:1
theorem initial_ratio (H C : ℕ) (h1 : condition1 H C) (h2 : condition2 H C) : 
  H = 6 * C :=
by {
  sorry
}

end initial_ratio_l493_49369


namespace james_beats_per_week_l493_49328

def beats_per_minute := 200
def hours_per_day := 2
def days_per_week := 7

def beats_per_week (beats_per_minute: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  (beats_per_minute * hours_per_day * 60) * days_per_week

theorem james_beats_per_week : beats_per_week beats_per_minute hours_per_day days_per_week = 168000 := by
  sorry

end james_beats_per_week_l493_49328


namespace sqrt_last_digit_l493_49373

-- Definitions related to the problem
def is_p_adic_number (α : ℕ) (p : ℕ) := true -- assume this definition captures p-adic number system

-- Problem statement in Lean 4
theorem sqrt_last_digit (p α a1 b1 : ℕ) (hα : is_p_adic_number α p) (h_last_digit_α : α % p = a1)
  (h_sqrt : (b1 * b1) % p = α % p) :
  (b1 * b1) % p = a1 :=
by sorry

end sqrt_last_digit_l493_49373


namespace ship_with_highest_no_car_round_trip_percentage_l493_49371

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l493_49371


namespace k_ge_a_l493_49377

theorem k_ge_a (a k : ℕ) (h_pos_a : 0 < a) (h_pos_k : 0 < k) 
  (h_div : (a ^ 2 + k) ∣ (a - 1) * a * (a + 1)) : k ≥ a := 
sorry

end k_ge_a_l493_49377


namespace garden_ratio_length_to_width_l493_49346

theorem garden_ratio_length_to_width (width length : ℕ) (area : ℕ) 
  (h1 : area = 507) 
  (h2 : width = 13) 
  (h3 : length * width = area) :
  length / width = 3 :=
by
  -- Proof to be filled in.
  sorry

end garden_ratio_length_to_width_l493_49346


namespace rate_of_current_l493_49350

theorem rate_of_current (c : ℝ) : 
  (∀ t : ℝ, t = 0.4 → ∀ d : ℝ, d = 9.6 → ∀ b : ℝ, b = 20 →
  d = (b + c) * t → c = 4) :=
sorry

end rate_of_current_l493_49350


namespace evens_minus_odds_equal_40_l493_49325

-- Define the sum of even integers from 2 to 80
def sum_evens : ℕ := (List.range' 2 40).sum

-- Define the sum of odd integers from 1 to 79
def sum_odds : ℕ := (List.range' 1 40).sum

-- Define the main theorem to prove
theorem evens_minus_odds_equal_40 : sum_evens - sum_odds = 40 := by
  -- Proof will go here
  sorry

end evens_minus_odds_equal_40_l493_49325


namespace minimum_pieces_for_K_1997_l493_49303

-- Definitions provided by the conditions in the problem.
def is_cube_shaped (n : ℕ) := ∃ (a : ℕ), n = a^3

def has_chocolate_coating (surface_area : ℕ) (n : ℕ) := 
  surface_area = 6 * n^2

def min_pieces (n K : ℕ) := n^3 / K

-- Expressing the proof problem in Lean 4.
theorem minimum_pieces_for_K_1997 {n : ℕ} (h_n : n = 1997) (H : ∀ (K : ℕ), K = 1997 ∧ K > 0) 
  (h_cube : is_cube_shaped n) (h_chocolate : has_chocolate_coating 6 n) :
  min_pieces 1997 1997 = 1997^3 :=
by
  sorry

end minimum_pieces_for_K_1997_l493_49303


namespace system_consistent_and_solution_l493_49395

theorem system_consistent_and_solution (a x : ℝ) : 
  (a = -10 ∧ x = -1/3) ∨ (a = -8 ∧ x = -1) ∨ (a = 4 ∧ x = -2) ↔ 
  3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0 := by
  sorry

end system_consistent_and_solution_l493_49395


namespace final_quarters_l493_49351

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 760
def first_spent : ℕ := 418
def second_spent : ℕ := 192

-- Define the final amount of quarters Sally should have
theorem final_quarters (initial_quarters first_spent second_spent : ℕ) : initial_quarters - first_spent - second_spent = 150 :=
by
  sorry

end final_quarters_l493_49351


namespace class_average_l493_49323

theorem class_average (n : ℕ) (h₁ : n = 100) (h₂ : 25 ≤ n) 
  (h₃ : 50 ≤ n) (h₄ : 25 * 80 + 50 * 65 + (n - 75) * 90 = 7500) :
  (25 * 80 + 50 * 65 + (n - 75) * 90) / n = 75 := 
by
  sorry

end class_average_l493_49323


namespace wax_current_amount_l493_49372

theorem wax_current_amount (wax_needed wax_total : ℕ) (h : wax_needed + 11 = wax_total) : 11 = wax_total - wax_needed :=
by
  sorry

end wax_current_amount_l493_49372
