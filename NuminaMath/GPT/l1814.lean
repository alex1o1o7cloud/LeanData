import Mathlib

namespace box_volume_correct_l1814_181422

variables (length width height : ℕ)

def volume_of_box (length width height : ℕ) : ℕ :=
  length * width * height

theorem box_volume_correct :
  volume_of_box 20 15 10 = 3000 :=
by
  -- This is where the proof would go
  sorry 

end box_volume_correct_l1814_181422


namespace tan_monotonic_increasing_interval_l1814_181410

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | 2 * k * Real.pi - (5 * Real.pi) / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3 }

theorem tan_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (y = Real.tan ((x / 2) + (Real.pi / 3))) → 
           x ∈ monotonic_increasing_interval k :=
sorry

end tan_monotonic_increasing_interval_l1814_181410


namespace algebraic_expression_value_l1814_181424

variable (a : ℝ)

theorem algebraic_expression_value (h : a = Real.sqrt 2) :
  (a / (a - 1)^2) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 :=
by
  sorry

end algebraic_expression_value_l1814_181424


namespace solve_for_a_l1814_181421

theorem solve_for_a (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (x^3) = Real.log x / Real.log a)
  (h2 : f 8 = 1) :
  a = 2 :=
sorry

end solve_for_a_l1814_181421


namespace divisor_of_99_l1814_181462

def reverse_digits (n : ℕ) : ℕ :=
  -- We assume a placeholder definition for reversing the digits of a number
  sorry

theorem divisor_of_99 (k : ℕ) (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : k ∣ 99 :=
  sorry

end divisor_of_99_l1814_181462


namespace ancient_chinese_wine_problem_l1814_181453

theorem ancient_chinese_wine_problem:
  ∃ x: ℝ, 10 * x + 3 * (5 - x) = 30 :=
by
  sorry

end ancient_chinese_wine_problem_l1814_181453


namespace find_polynomial_h_l1814_181482

theorem find_polynomial_h (f h : ℝ → ℝ) (hf : ∀ x, f x = x^2) (hh : ∀ x, f (h x) = 9 * x^2 + 6 * x + 1) : 
  (∀ x, h x = 3 * x + 1) ∨ (∀ x, h x = -3 * x - 1) :=
by
  sorry

end find_polynomial_h_l1814_181482


namespace product_mnp_l1814_181417

theorem product_mnp (m n p : ℕ) (b x z c : ℂ) (h1 : b^8 * x * z - b^7 * z - b^6 * x = b^5 * (c^5 - 1)) 
  (h2 : (b^m * x - b^n) * (b^p * z - b^3) = b^5 * c^5) : m * n * p = 30 :=
sorry

end product_mnp_l1814_181417


namespace minimum_spend_on_boxes_l1814_181414

noncomputable def box_length : ℕ := 20
noncomputable def box_width : ℕ := 20
noncomputable def box_height : ℕ := 12
noncomputable def cost_per_box : ℝ := 0.40
noncomputable def total_volume : ℕ := 2400000

theorem minimum_spend_on_boxes : 
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 200 :=
by
  sorry

end minimum_spend_on_boxes_l1814_181414


namespace a_pow_11_b_pow_11_l1814_181476

theorem a_pow_11_b_pow_11 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end a_pow_11_b_pow_11_l1814_181476


namespace walther_janous_inequality_equality_condition_l1814_181485

theorem walther_janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ x / y = 2 ∧ y = z :=
sorry

end walther_janous_inequality_equality_condition_l1814_181485


namespace value_at_minus_two_l1814_181426

def f (x : ℝ) : ℝ := x^2 + 3 * x - 5

theorem value_at_minus_two : f (-2) = -7 := by
  sorry

end value_at_minus_two_l1814_181426


namespace evaluate_expression_l1814_181447

theorem evaluate_expression :
  (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 = 8 :=
by
  sorry

end evaluate_expression_l1814_181447


namespace log_x3y2_value_l1814_181458

open Real

noncomputable def log_identity (x y : ℝ) : Prop :=
  log (x * y^4) = 1 ∧ log (x^3 * y) = 1

theorem log_x3y2_value (x y : ℝ) (h : log_identity x y) : log (x^3 * y^2) = 13 / 11 :=
  by
  sorry

end log_x3y2_value_l1814_181458


namespace power_sum_evaluation_l1814_181491

theorem power_sum_evaluation :
  (-1)^(4^3) + 2^(3^2) = 513 :=
by
  sorry

end power_sum_evaluation_l1814_181491


namespace mean_exterior_angles_l1814_181433

theorem mean_exterior_angles (a b c : ℝ) (ha : a = 45) (hb : b = 75) (hc : c = 60) :
  (180 - a + 180 - b + 180 - c) / 3 = 120 :=
by 
  sorry

end mean_exterior_angles_l1814_181433


namespace number_of_boxes_l1814_181467

def magazines : ℕ := 63
def magazines_per_box : ℕ := 9

theorem number_of_boxes : magazines / magazines_per_box = 7 :=
by 
  sorry

end number_of_boxes_l1814_181467


namespace number_solution_l1814_181499

theorem number_solution (x : ℝ) : (x / 5 + 4 = x / 4 - 4) → x = 160 := by
  intros h
  sorry

end number_solution_l1814_181499


namespace population_sampling_precision_l1814_181454

theorem population_sampling_precision (sample_size : ℕ → Prop) 
    (A : Prop) (B : Prop) (C : Prop) (D : Prop)
    (condition_A : A = (∀ n : ℕ, sample_size n → false))
    (condition_B : B = (∀ n : ℕ, sample_size n → n > 0 → true))
    (condition_C : C = (∀ n : ℕ, sample_size n → false))
    (condition_D : D = (∀ n : ℕ, sample_size n → false)) :
  B :=
by sorry

end population_sampling_precision_l1814_181454


namespace evaluate_expression_l1814_181437

theorem evaluate_expression (a b c : ℝ)
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := 
sorry

end evaluate_expression_l1814_181437


namespace square_cookie_cutters_count_l1814_181484

def triangles_sides : ℕ := 6 * 3
def hexagons_sides : ℕ := 2 * 6
def total_sides : ℕ := 46
def sides_from_squares (S : ℕ) : ℕ := S * 4

theorem square_cookie_cutters_count (S : ℕ) :
  triangles_sides + hexagons_sides + sides_from_squares S = total_sides → S = 4 :=
by
  sorry

end square_cookie_cutters_count_l1814_181484


namespace school_election_votes_l1814_181419

theorem school_election_votes (E S R L : ℕ)
  (h1 : E = 2 * S)
  (h2 : E = 4 * R)
  (h3 : S = 5 * R)
  (h4 : S = 3 * L)
  (h5 : R = 16) :
  E = 64 ∧ S = 80 ∧ R = 16 ∧ L = 27 := by
  sorry

end school_election_votes_l1814_181419


namespace measure_of_angle_Q_l1814_181474

theorem measure_of_angle_Q (a b c d e Q : ℝ)
  (ha : a = 138) (hb : b = 85) (hc : c = 130) (hd : d = 120) (he : e = 95)
  (h_hex : a + b + c + d + e + Q = 720) : 
  Q = 152 :=
by
  rw [ha, hb, hc, hd, he] at h_hex
  linarith

end measure_of_angle_Q_l1814_181474


namespace sequence_general_term_l1814_181496

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l1814_181496


namespace quadratic_expression_l1814_181411

-- Definitions of roots and their properties
def quadratic_roots (r s : ℚ) : Prop :=
  (r + s = 5 / 3) ∧ (r * s = -8 / 3)

theorem quadratic_expression (r s : ℚ) (h : quadratic_roots r s) :
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
by
  sorry

end quadratic_expression_l1814_181411


namespace percent_birth_month_in_march_l1814_181460

theorem percent_birth_month_in_march (total_people : ℕ) (march_births : ℕ) (h1 : total_people = 100) (h2 : march_births = 8) : (march_births * 100 / total_people) = 8 := by
  sorry

end percent_birth_month_in_march_l1814_181460


namespace intersection_A_B_l1814_181472

-- Define set A
def A : Set Int := { x | x^2 - x - 2 ≤ 0 }

-- Define set B
def B : Set Int := { x | x < 1 }

-- Define the intersection set
def intersection_AB : Set Int := { -1, 0 }

-- Formalize the proof statement
theorem intersection_A_B : (A ∩ B) = intersection_AB :=
by sorry

end intersection_A_B_l1814_181472


namespace pavan_travel_distance_l1814_181457

theorem pavan_travel_distance (t : ℝ) (v1 v2 : ℝ) (D : ℝ) (h₁ : t = 15) (h₂ : v1 = 30) (h₃ : v2 = 25):
  (D / 2) / v1 + (D / 2) / v2 = t → D = 2250 / 11 :=
by
  intro h
  rw [h₁, h₂, h₃] at h
  sorry

end pavan_travel_distance_l1814_181457


namespace lunch_break_duration_l1814_181402

theorem lunch_break_duration (m a : ℝ) (L : ℝ) :
  (9 - L) * (m + a) = 0.6 → 
  (7 - L) * a = 0.3 → 
  (5 - L) * m = 0.1 → 
  L = 42 / 60 :=
by sorry

end lunch_break_duration_l1814_181402


namespace find_coordinates_of_A_l1814_181493

theorem find_coordinates_of_A (x : ℝ) :
  let A := (x, 1, 2)
  let B := (2, 3, 4)
  (Real.sqrt ((x - 2)^2 + (1 - 3)^2 + (2 - 4)^2) = 2 * Real.sqrt 6) →
  (x = 6 ∨ x = -2) := 
by
  intros
  sorry

end find_coordinates_of_A_l1814_181493


namespace intersection_of_A_and_B_l1814_181405

def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := 
by sorry

end intersection_of_A_and_B_l1814_181405


namespace hypotenuse_length_l1814_181431

theorem hypotenuse_length (a b c : ℝ) (h_right_angled : c^2 = a^2 + b^2) (h_sum_of_squares : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l1814_181431


namespace amelia_jet_bars_l1814_181471

theorem amelia_jet_bars
    (required : ℕ) (sold_monday : ℕ) (sold_tuesday_less : ℕ) (total_sold : ℕ) (remaining : ℕ) :
    required = 90 →
    sold_monday = 45 →
    sold_tuesday_less = 16 →
    total_sold = sold_monday + (sold_monday - sold_tuesday_less) →
    remaining = required - total_sold →
    remaining = 16 :=
by
  intros
  sorry

end amelia_jet_bars_l1814_181471


namespace FDI_in_rural_AndhraPradesh_l1814_181466

-- Definitions from conditions
def total_FDI : ℝ := 300 -- Total FDI calculated
def FDI_Gujarat : ℝ := 0.30 * total_FDI
def FDI_Gujarat_Urban : ℝ := 0.80 * FDI_Gujarat
def FDI_AndhraPradesh : ℝ := 0.20 * total_FDI
def FDI_AndhraPradesh_Rural : ℝ := 0.50 * FDI_AndhraPradesh 

-- Given the conditions, prove the size of FDI in rural Andhra Pradesh is 30 million
theorem FDI_in_rural_AndhraPradesh :
  FDI_Gujarat_Urban = 72 → FDI_AndhraPradesh_Rural = 30 :=
by
  sorry

end FDI_in_rural_AndhraPradesh_l1814_181466


namespace medians_square_sum_l1814_181440

theorem medians_square_sum (a b c : ℝ) (ha : a = 13) (hb : b = 13) (hc : c = 10) :
  let m_a := (1 / 2 * (2 * b^2 + 2 * c^2 - a^2))^(1/2)
  let m_b := (1 / 2 * (2 * c^2 + 2 * a^2 - b^2))^(1/2)
  let m_c := (1 / 2 * (2 * a^2 + 2 * b^2 - c^2))^(1/2)
  m_a^2 + m_b^2 + m_c^2 = 432 :=
by
  sorry

end medians_square_sum_l1814_181440


namespace cupcakes_total_l1814_181404

theorem cupcakes_total (initially_made : ℕ) (sold : ℕ) (newly_made : ℕ) (initially_made_eq : initially_made = 42) (sold_eq : sold = 22) (newly_made_eq : newly_made = 39) : initially_made - sold + newly_made = 59 :=
by
  sorry

end cupcakes_total_l1814_181404


namespace moles_of_HCl_formed_l1814_181435

theorem moles_of_HCl_formed
  (C2H6_initial : Nat)
  (Cl2_initial : Nat)
  (HCl_expected : Nat)
  (balanced_reaction : C2H6_initial + Cl2_initial = C2H6_initial + HCl_expected):
  C2H6_initial = 2 → Cl2_initial = 2 → HCl_expected = 2 :=
by
  intros
  sorry

end moles_of_HCl_formed_l1814_181435


namespace soybeans_in_jar_l1814_181469

theorem soybeans_in_jar
  (totalRedBeans : ℕ)
  (sampleSize : ℕ)
  (sampleRedBeans : ℕ)
  (totalBeans : ℕ)
  (proportion : sampleRedBeans / sampleSize = totalRedBeans / totalBeans)
  (h1 : totalRedBeans = 200)
  (h2 : sampleSize = 60)
  (h3 : sampleRedBeans = 5) :
  totalBeans = 2400 :=
by
  sorry

end soybeans_in_jar_l1814_181469


namespace find_a4_l1814_181403

theorem find_a4 (a : ℕ → ℕ) 
  (h1 : ∀ n, (a n + 1) / (a (n + 1) + 1) = 1 / 2) 
  (h2 : a 2 = 2) : 
  a 4 = 11 :=
sorry

end find_a4_l1814_181403


namespace staircase_problem_l1814_181400

theorem staircase_problem :
  ∃ (n : ℕ), (n > 20) ∧ (n % 5 = 4) ∧ (n % 6 = 3) ∧ (n % 7 = 5) ∧ n = 159 :=
by sorry

end staircase_problem_l1814_181400


namespace sum_of_remainders_mod_15_l1814_181420

theorem sum_of_remainders_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) :
  (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_mod_15_l1814_181420


namespace problem_l1814_181498

theorem problem
  (r s t : ℝ)
  (h₀ : r^3 - 15 * r^2 + 13 * r - 8 = 0)
  (h₁ : s^3 - 15 * s^2 + 13 * s - 8 = 0)
  (h₂ : t^3 - 15 * t^2 + 13 * t - 8 = 0) :
  (r / (1 / r + s * t) + s / (1 / s + t * r) + t / (1 / t + r * s) = 199 / 9) :=
sorry

end problem_l1814_181498


namespace lines_through_same_quadrants_l1814_181406

theorem lines_through_same_quadrants (k b : ℝ) (hk : k ≠ 0):
    ∃ n, n ≥ 7 ∧ ∀ (f : Fin n → ℝ × ℝ), ∃ (i j : Fin n), i ≠ j ∧ 
    ((f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0) :=
by sorry

end lines_through_same_quadrants_l1814_181406


namespace quadrilateral_area_lt_one_l1814_181430

theorem quadrilateral_area_lt_one 
  (a b c d : ℝ) 
  (h_a : a < 1) 
  (h_b : b < 1) 
  (h_c : c < 1) 
  (h_d : d < 1) 
  (h_pos_a : 0 ≤ a)
  (h_pos_b : 0 ≤ b)
  (h_pos_c : 0 ≤ c)
  (h_pos_d : 0 ≤ d) :
  ∃ (area : ℝ), area < 1 :=
by
  sorry

end quadrilateral_area_lt_one_l1814_181430


namespace waiter_tables_l1814_181490

theorem waiter_tables (w m : ℝ) (avg_customers_per_table : ℝ) (total_customers : ℝ) (t : ℝ)
  (hw : w = 7.0)
  (hm : m = 3.0)
  (havg : avg_customers_per_table = 1.111111111)
  (htotal : total_customers = w + m)
  (ht : t = total_customers / avg_customers_per_table) :
  t = 90 :=
by
  -- Proof would be inserted here
  sorry

end waiter_tables_l1814_181490


namespace cube_surface_area_l1814_181438

theorem cube_surface_area (side_length : ℝ) (h : side_length = 8) : 6 * side_length^2 = 384 :=
by
  rw [h]
  sorry

end cube_surface_area_l1814_181438


namespace flour_needed_correct_l1814_181448

-- Define the total flour required and the flour already added
def total_flour : ℕ := 8
def flour_already_added : ℕ := 2

-- Define the equation to determine the remaining flour needed
def flour_needed : ℕ := total_flour - flour_already_added

-- Prove that the flour needed to be added is 6 cups
theorem flour_needed_correct : flour_needed = 6 := by
  sorry

end flour_needed_correct_l1814_181448


namespace range_of_k_l1814_181452

theorem range_of_k (k : ℝ) : ((∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0)) :=
sorry

end range_of_k_l1814_181452


namespace PeytonManning_total_distance_l1814_181443

noncomputable def PeytonManning_threw_distance : Prop :=
  let throw_distance_50 := 20
  let throw_times_sat := 20
  let throw_times_sun := 30
  let total_distance := 1600
  ∃ R : ℚ, 
    let throw_distance_80 := R * throw_distance_50
    let distance_sat := throw_distance_50 * throw_times_sat
    let distance_sun := throw_distance_80 * throw_times_sun
    distance_sat + distance_sun = total_distance

theorem PeytonManning_total_distance :
  PeytonManning_threw_distance := by
  sorry

end PeytonManning_total_distance_l1814_181443


namespace roots_of_quadratic_l1814_181480

theorem roots_of_quadratic (x : ℝ) : x^2 - 5 * x = 0 ↔ (x = 0 ∨ x = 5) := by 
  sorry

end roots_of_quadratic_l1814_181480


namespace tickets_difference_l1814_181459

def number_of_tickets_for_toys := 31
def number_of_tickets_for_clothes := 14

theorem tickets_difference : number_of_tickets_for_toys - number_of_tickets_for_clothes = 17 := by
  sorry

end tickets_difference_l1814_181459


namespace cost_price_of_table_l1814_181477

theorem cost_price_of_table (CP SP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3000) : CP = 2500 := by
    sorry

end cost_price_of_table_l1814_181477


namespace even_gt_one_square_gt_l1814_181497

theorem even_gt_one_square_gt (m : ℕ) (h_even : ∃ k : ℕ, m = 2 * k) (h_gt_one : m > 1) : m < m * m :=
by
  sorry

end even_gt_one_square_gt_l1814_181497


namespace complement_of_A_in_U_l1814_181445

-- Given definitions from the problem
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}

-- The theorem to be proven
theorem complement_of_A_in_U : U \ A = {1, 3, 6, 7} :=
by
  sorry

end complement_of_A_in_U_l1814_181445


namespace negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l1814_181475

theorem negation_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem converse_of_p (π : ℝ) (a b c d : ℚ) (h : a = c ∧ b = d) : a * π + b = c * π + d :=
  sorry

theorem inverse_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b ≠ c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem contrapositive_of_p (π : ℝ) (a b c d : ℚ) (h : a ≠ c ∨ b ≠ d) : a * π + b ≠ c * π + d :=
  sorry

theorem original_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a = c ∧ b = d :=
  sorry

end negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l1814_181475


namespace iterate_g_eq_2_l1814_181413

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then n^2 - 2*n + 2 else 2*n

theorem iterate_g_eq_2 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 100): 
  (∃ m : ℕ, (Nat.iterate g m n) = 2) ↔ n = 1 :=
by
sorry

end iterate_g_eq_2_l1814_181413


namespace sin_theta_minus_cos_theta_l1814_181439

theorem sin_theta_minus_cos_theta (θ : ℝ) (b : ℝ) (hθ_acute : 0 < θ ∧ θ < π / 2) (h_cos2θ : Real.cos (2 * θ) = b) :
  ∃ x, x = Real.sin θ - Real.cos θ ∧ (x = Real.sqrt b ∨ x = -Real.sqrt b) := 
by
  sorry

end sin_theta_minus_cos_theta_l1814_181439


namespace sum_of_a_c_l1814_181436

theorem sum_of_a_c (a b c d : ℝ) (h1 : -2 * abs (1 - a) + b = 7) (h2 : 2 * abs (1 - c) + d = 7)
    (h3 : -2 * abs (11 - a) + b = -1) (h4 : 2 * abs (11 - c) + d = -1) : a + c = 12 := by
  -- Definitions for conditions
  -- h1: intersection at (1, 7) for first graph
  -- h2: intersection at (1, 7) for second graph
  -- h3: intersection at (11, -1) for first graph
  -- h4: intersection at (11, -1) for second graph
  sorry

end sum_of_a_c_l1814_181436


namespace sufficient_but_not_necessary_condition_for_q_l1814_181415

def proposition_p (a : ℝ) := (1 / a) > (1 / 4)
def proposition_q (a : ℝ) := ∀ x : ℝ, (a * x^2 + a * x + 1) > 0

theorem sufficient_but_not_necessary_condition_for_q (a : ℝ) :
  proposition_p a → proposition_q a → (∃ a : ℝ, 0 < a ∧ a < 4) ∧ (∃ a : ℝ, 0 < a ∧ a < 4 ∧ ¬ proposition_p a) 
  := sorry

end sufficient_but_not_necessary_condition_for_q_l1814_181415


namespace distance_between_feet_of_perpendiculars_eq_area_over_radius_l1814_181427
noncomputable def area (ABC : Type) : ℝ := sorry
noncomputable def circumradius (ABC : Type) : ℝ := sorry

theorem distance_between_feet_of_perpendiculars_eq_area_over_radius
  (ABC : Type)
  (area_ABC : ℝ)
  (R : ℝ)
  (h_area : area ABC = area_ABC)
  (h_radius : circumradius ABC = R) :
  ∃ (m : ℝ), m = area_ABC / R := sorry

end distance_between_feet_of_perpendiculars_eq_area_over_radius_l1814_181427


namespace find_slope_l1814_181483

theorem find_slope (k : ℝ) :
  (∀ x y : ℝ, y = -2 * x + 3 → y = k * x + 4 → (x, y) = (1, 1)) → k = -3 :=
by
  sorry

end find_slope_l1814_181483


namespace John_pays_amount_l1814_181487

/-- Prove the amount John pays given the conditions -/
theorem John_pays_amount
  (total_candies : ℕ)
  (candies_paid_by_dave : ℕ)
  (cost_per_candy : ℚ)
  (candies_paid_by_john := total_candies - candies_paid_by_dave)
  (total_cost_paid_by_john := candies_paid_by_john * cost_per_candy) :
  total_candies = 20 →
  candies_paid_by_dave = 6 →
  cost_per_candy = 1.5 →
  total_cost_paid_by_john = 21 := 
by
  intros h1 h2 h3
  -- Proof is skipped
  sorry

end John_pays_amount_l1814_181487


namespace coeff_x3_in_binom_expansion_l1814_181450

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the coefficient function for x^k in the binomial expansion of (x + 1)^n
def binom_coeff (n k : ℕ) : ℕ := binom n k

-- The theorem to prove that the coefficient of x^3 in the expansion of (x + 1)^36 is 7140
theorem coeff_x3_in_binom_expansion : binom_coeff 36 3 = 7140 :=
by
  sorry

end coeff_x3_in_binom_expansion_l1814_181450


namespace a2b2_div_ab1_is_square_l1814_181486

theorem a2b2_div_ab1_is_square (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, (a^2 + b^2) / (ab + 1) = k^2 :=
sorry

end a2b2_div_ab1_is_square_l1814_181486


namespace vehicle_worth_l1814_181449

-- Definitions from the conditions
def monthlyEarnings : ℕ := 4000
def savingFraction : ℝ := 0.5
def savingMonths : ℕ := 8

-- Theorem statement
theorem vehicle_worth : (monthlyEarnings * savingFraction * savingMonths : ℝ) = 16000 := 
by
  sorry

end vehicle_worth_l1814_181449


namespace gain_percentage_is_five_percent_l1814_181429

variables (CP SP New_SP Loss Loss_Percentage Gain Gain_Percentage : ℝ)
variables (H1 : Loss_Percentage = 10)
variables (H2 : CP = 933.33)
variables (H3 : Loss = (Loss_Percentage / 100) * CP)
variables (H4 : SP = CP - Loss)
variables (H5 : New_SP = SP + 140)
variables (H6 : Gain = New_SP - CP)
variables (H7 : Gain_Percentage = (Gain / CP) * 100)

theorem gain_percentage_is_five_percent :
  Gain_Percentage = 5 :=
by
  -- Proof goes here
  sorry

end gain_percentage_is_five_percent_l1814_181429


namespace total_potatoes_now_l1814_181456

def initial_potatoes : ℕ := 8
def uneaten_new_potatoes : ℕ := 3

theorem total_potatoes_now : initial_potatoes + uneaten_new_potatoes = 11 := by
  sorry

end total_potatoes_now_l1814_181456


namespace people_in_room_l1814_181446

theorem people_in_room (total_chairs seated_chairs total_people : ℕ) 
  (h1 : 3 * total_people = 5 * seated_chairs)
  (h2 : 4 * total_chairs = 5 * seated_chairs) 
  (h3 : total_chairs - seated_chairs = 8) : 
  total_people = 54 :=
by
  sorry

end people_in_room_l1814_181446


namespace catriona_total_fish_eq_44_l1814_181479

-- Definitions based on conditions
def goldfish : ℕ := 8
def angelfish : ℕ := goldfish + 4
def guppies : ℕ := 2 * angelfish
def total_fish : ℕ := goldfish + angelfish + guppies

-- The theorem we need to prove
theorem catriona_total_fish_eq_44 : total_fish = 44 :=
by
  -- We are skipping the proof steps with 'sorry' for now
  sorry

end catriona_total_fish_eq_44_l1814_181479


namespace calculate_perimeter_l1814_181416

-- Definitions based on conditions
def num_posts : ℕ := 36
def post_width : ℕ := 2
def gap_width : ℕ := 4
def sides : ℕ := 4

-- Computations inferred from the conditions (not using solution steps directly)
def posts_per_side : ℕ := num_posts / sides
def gaps_per_side : ℕ := posts_per_side - 1
def side_length : ℕ := posts_per_side * post_width + gaps_per_side * gap_width

-- Theorem statement, proving the perimeter is 200 feet
theorem calculate_perimeter : 4 * side_length = 200 := by
  sorry

end calculate_perimeter_l1814_181416


namespace find_a_l1814_181442

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem find_a (a : ℝ) (h : binomial_coefficient 4 2 + 4 * a = 10) : a = 1 :=
by
  sorry

end find_a_l1814_181442


namespace stingrays_count_l1814_181408

theorem stingrays_count (Sh S : ℕ) (h1 : Sh = 2 * S) (h2 : S + Sh = 84) : S = 28 :=
by
  -- Proof will be filled here
  sorry

end stingrays_count_l1814_181408


namespace total_sandwiches_l1814_181478

theorem total_sandwiches :
  let billy := 49
  let katelyn := billy + 47
  let chloe := katelyn / 4
  billy + katelyn + chloe = 169 :=
by
  sorry

end total_sandwiches_l1814_181478


namespace ruth_started_with_89_apples_l1814_181412

theorem ruth_started_with_89_apples 
  (initial_apples : ℕ)
  (shared_apples : ℕ)
  (remaining_apples : ℕ)
  (h1 : shared_apples = 5)
  (h2 : remaining_apples = 84)
  (h3 : remaining_apples = initial_apples - shared_apples) : 
  initial_apples = 89 :=
by
  sorry

end ruth_started_with_89_apples_l1814_181412


namespace calc_expression_l1814_181455

theorem calc_expression : (2019 / 2018) - (2018 / 2019) = 4037 / 4036 := 
by sorry

end calc_expression_l1814_181455


namespace m_n_sum_l1814_181432

theorem m_n_sum (m n : ℝ) (h : ∀ x : ℝ, x^2 + m * x + 6 = (x - 2) * (x - n)) : m + n = -2 :=
by
  sorry

end m_n_sum_l1814_181432


namespace sum_of_solutions_l1814_181468

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end sum_of_solutions_l1814_181468


namespace paula_remaining_money_l1814_181461

theorem paula_remaining_money 
  (M : Int) (C_s : Int) (N_s : Int) (C_p : Int) (N_p : Int)
  (h1 : M = 250) 
  (h2 : C_s = 15) 
  (h3 : N_s = 5) 
  (h4 : C_p = 25) 
  (h5 : N_p = 3) : 
  M - (C_s * N_s + C_p * N_p) = 100 := 
by
  sorry

end paula_remaining_money_l1814_181461


namespace runs_twice_l1814_181464

-- Definitions of the conditions
def game_count : ℕ := 6
def runs_one : ℕ := 1
def runs_five : ℕ := 5
def average_runs : ℕ := 4

-- Assuming the number of runs scored twice is x
variable (x : ℕ)

-- Definition of total runs scored based on the conditions
def total_runs : ℕ := runs_one + 2 * x + 3 * runs_five

-- Statement to prove the number of runs scored twice
theorem runs_twice :
  (total_runs x) / game_count = average_runs → x = 4 :=
by
  sorry

end runs_twice_l1814_181464


namespace p_at_5_l1814_181434

noncomputable def p (x : ℝ) : ℝ :=
  sorry

def p_cond (n : ℝ) : Prop :=
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → p n = 1 / n^3

theorem p_at_5 : (∀ n, p_cond n) → p 5 = -149 / 1500 :=
by
  intros
  sorry

end p_at_5_l1814_181434


namespace part1_part2_l1814_181451

-- Part (1)
theorem part1 (B : ℝ) (b : ℝ) (S : ℝ) (a c : ℝ) (B_eq : B = Real.pi / 3) 
  (b_eq : b = Real.sqrt 7) (S_eq : S = (3 * Real.sqrt 3) / 2) :
  a + c = 5 := 
sorry

-- Part (2)
theorem part2 (C : ℝ) (c : ℝ) (dot_BA_BC AB_AC : ℝ) 
  (C_cond : 2 * Real.cos C * (dot_BA_BC + AB_AC) = c^2) :
  C = Real.pi / 3 := 
sorry

end part1_part2_l1814_181451


namespace find_a_and_solve_inequality_l1814_181401

theorem find_a_and_solve_inequality :
  (∀ x : ℝ, |x^2 - 4 * x + a| + |x - 3| ≤ 5 → x ≤ 3) →
  a = 8 :=
by
  sorry

end find_a_and_solve_inequality_l1814_181401


namespace range_of_solutions_l1814_181481

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l1814_181481


namespace apples_not_sold_correct_l1814_181428

-- Define the constants and conditions
def boxes_ordered_per_week : ℕ := 10
def apples_per_box : ℕ := 300
def fraction_sold : ℚ := 3 / 4

-- Define the total number of apples ordered in a week
def total_apples_ordered : ℕ := boxes_ordered_per_week * apples_per_box

-- Define the total number of apples sold in a week
def apples_sold : ℚ := fraction_sold * total_apples_ordered

-- Define the total number of apples not sold in a week
def apples_not_sold : ℚ := total_apples_ordered - apples_sold

-- Lean statement to prove the total number of apples not sold is 750
theorem apples_not_sold_correct :
  apples_not_sold = 750 := 
sorry

end apples_not_sold_correct_l1814_181428


namespace income_on_fifth_day_l1814_181470

-- Define the incomes for the first four days
def income_day1 := 600
def income_day2 := 250
def income_day3 := 450
def income_day4 := 400

-- Define the average income
def average_income := 500

-- Define the length of days
def days := 5

-- Define the total income for the 5 days
def total_income : ℕ := days * average_income

-- Define the total income for the first 4 days
def total_income_first4 := income_day1 + income_day2 + income_day3 + income_day4

-- Define the income on the fifth day
def income_day5 := total_income - total_income_first4

-- The theorem to prove the income of the fifth day is $800
theorem income_on_fifth_day : income_day5 = 800 := by
  -- proof is not required, so we leave the proof section with sorry
  sorry

end income_on_fifth_day_l1814_181470


namespace num_employees_is_143_l1814_181463

def b := 143
def is_sol (b : ℕ) := 80 < b ∧ b < 150 ∧ b % 4 = 3 ∧ b % 5 = 3 ∧ b % 7 = 4

theorem num_employees_is_143 : is_sol b :=
by
  -- This is where the proof would be written
  sorry

end num_employees_is_143_l1814_181463


namespace area_of_triangle_8_9_9_l1814_181418

noncomputable def triangle_area (a b c : ℕ) : Real :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_8_9_9 : triangle_area 8 9 9 = 4 * Real.sqrt 65 :=
by
  sorry

end area_of_triangle_8_9_9_l1814_181418


namespace tan_beta_is_six_over_seventeen_l1814_181465
-- Import the Mathlib library

-- Define the problem in Lean
theorem tan_beta_is_six_over_seventeen
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 4 / 5)
  (h2 : Real.tan (α - β) = 2 / 3) :
  Real.tan β = 6 / 17 := 
by
  sorry

end tan_beta_is_six_over_seventeen_l1814_181465


namespace price_of_pants_l1814_181473

theorem price_of_pants
  (P S H : ℝ)
  (h1 : P + S + H = 340)
  (h2 : S = (3 / 4) * P)
  (h3 : H = P + 10) :
  P = 120 :=
by
  sorry

end price_of_pants_l1814_181473


namespace swan_populations_after_10_years_l1814_181409

noncomputable def swan_population_rita (R : ℝ) : ℝ :=
  480 * (1 - R / 100) ^ 10

noncomputable def swan_population_sarah (S : ℝ) : ℝ :=
  640 * (1 - S / 100) ^ 10

noncomputable def swan_population_tom (T : ℝ) : ℝ :=
  800 * (1 - T / 100) ^ 10

theorem swan_populations_after_10_years 
  (R S T : ℝ) :
  swan_population_rita R = 480 * (1 - R / 100) ^ 10 ∧
  swan_population_sarah S = 640 * (1 - S / 100) ^ 10 ∧
  swan_population_tom T = 800 * (1 - T / 100) ^ 10 := 
by sorry

end swan_populations_after_10_years_l1814_181409


namespace tan_alpha_plus_beta_l1814_181492

open Real

theorem tan_alpha_plus_beta (A alpha beta : ℝ) (h1 : sin alpha = A * sin (alpha + beta)) (h2 : abs A > 1) :
  tan (alpha + beta) = sin beta / (cos beta - A) :=
by
  sorry

end tan_alpha_plus_beta_l1814_181492


namespace num_values_satisfying_g_g_x_eq_4_l1814_181495

def g (x : ℝ) : ℝ := sorry

theorem num_values_satisfying_g_g_x_eq_4 
  (h1 : g (-2) = 4)
  (h2 : g (2) = 4)
  (h3 : g (4) = 4)
  (h4 : ∀ x, g (x) ≠ -2)
  (h5 : ∃! x, g (x) = 2) 
  (h6 : ∃! x, g (x) = 4) 
  : ∃! x1 x2, g (g x1) = 4 ∧ g (g x2) = 4 ∧ x1 ≠ x2 :=
by
  sorry

end num_values_satisfying_g_g_x_eq_4_l1814_181495


namespace jen_profit_is_960_l1814_181423

def buying_price : ℕ := 80
def selling_price : ℕ := 100
def num_candy_bars_bought : ℕ := 50
def num_candy_bars_sold : ℕ := 48

def profit_per_candy_bar := selling_price - buying_price
def total_profit := profit_per_candy_bar * num_candy_bars_sold

theorem jen_profit_is_960 : total_profit = 960 := by
  sorry

end jen_profit_is_960_l1814_181423


namespace solve_log_equation_l1814_181489

theorem solve_log_equation :
  ∀ x : ℝ, 
  5 * Real.logb x (x / 9) + Real.logb (x / 9) x^3 + 8 * Real.logb (9 * x^2) (x^2) = 2
  → (x = 3 ∨ x = Real.sqrt 3) := by
  sorry

end solve_log_equation_l1814_181489


namespace pentagonal_grid_toothpicks_l1814_181425

theorem pentagonal_grid_toothpicks :
  ∀ (base toothpicks per sides toothpicks per joint : ℕ),
    base = 10 → 
    sides = 4 → 
    toothpicks_per_side = 8 → 
    joints = 5 → 
    toothpicks_per_joint = 1 → 
    (base + sides * toothpicks_per_side + joints * toothpicks_per_joint = 47) :=
by
  intros base sides toothpicks_per_side joints toothpicks_per_joint
  sorry

end pentagonal_grid_toothpicks_l1814_181425


namespace part1_part2_l1814_181441

open Real

variables {a b c : ℝ}

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    a + 4 * b + 9 * c ≥ 36 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    (b + c) / sqrt a + (a + c) / sqrt b + (a + b) / sqrt c ≥ 2 * sqrt (a * b * c) :=
sorry

end part1_part2_l1814_181441


namespace new_quadratic_coeff_l1814_181494

theorem new_quadratic_coeff (r s p q : ℚ) 
  (h1 : 3 * r^2 + 4 * r + 2 = 0)
  (h2 : 3 * s^2 + 4 * s + 2 = 0)
  (h3 : r + s = -4 / 3)
  (h4 : r * s = 2 / 3) 
  (h5 : r^3 + s^3 = - p) :
  p = 16 / 27 :=
by
  sorry

end new_quadratic_coeff_l1814_181494


namespace factorial_division_l1814_181407

open Nat

theorem factorial_division : 12! / 11! = 12 := sorry

end factorial_division_l1814_181407


namespace multiply_base5_234_75_l1814_181444

def to_base5 (n : ℕ) : ℕ := 
  let rec helper (n : ℕ) (acc : ℕ) (multiplier : ℕ) : ℕ := 
    if n = 0 then acc
    else
      let d := n % 5
      let q := n / 5
      helper q (acc + d * multiplier) (multiplier * 10)
  helper n 0 1

def base5_multiplication (a b : ℕ) : ℕ :=
  to_base5 ((a * b : ℕ))

theorem multiply_base5_234_75 : base5_multiplication 234 75 = 450620 := 
  sorry

end multiply_base5_234_75_l1814_181444


namespace multiply_divide_repeating_decimals_l1814_181488

theorem multiply_divide_repeating_decimals :
  (8 * (1 / 3) / 1) = 8 / 3 := by
  sorry

end multiply_divide_repeating_decimals_l1814_181488
