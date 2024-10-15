import Mathlib

namespace NUMINAMATH_GPT_count_multiples_l1896_189663

theorem count_multiples (n : ℕ) : 
  n = 1 ↔ ∃ k : ℕ, k < 500 ∧ k > 0 ∧ k % 4 = 0 ∧ k % 5 = 0 ∧ k % 6 = 0 ∧ k % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_l1896_189663


namespace NUMINAMATH_GPT_gcd_consecutive_term_max_l1896_189653

def b (n : ℕ) : ℕ := n.factorial + 2^n + n 

theorem gcd_consecutive_term_max (n : ℕ) (hn : n ≥ 0) :
  ∃ m ≤ (n : ℕ), (m = 2) := sorry

end NUMINAMATH_GPT_gcd_consecutive_term_max_l1896_189653


namespace NUMINAMATH_GPT_intersection_M_N_l1896_189683

-- Define set M
def M : Set Int := {-2, -1, 0, 1}

-- Define set N using the given condition
def N : Set Int := {n : Int | -1 <= n ∧ n <= 3}

-- State that the intersection of M and N is the set {-1, 0, 1}
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1896_189683


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1896_189621

theorem cone_lateral_surface_area (a : ℝ) (π : ℝ) (sqrt_3 : ℝ) 
  (h₁ : 0 < a)
  (h_area : (1 / 2) * a^2 * (sqrt_3 / 2) = sqrt_3) :
  π * 1 * 2 = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1896_189621


namespace NUMINAMATH_GPT_BoatsRUs_canoes_l1896_189691

theorem BoatsRUs_canoes :
  let a := 6
  let r := 3
  let n := 5
  let S := a * (r^n - 1) / (r - 1)
  S = 726 := by
  -- Proof
  sorry

end NUMINAMATH_GPT_BoatsRUs_canoes_l1896_189691


namespace NUMINAMATH_GPT_candy_distribution_proof_l1896_189620

theorem candy_distribution_proof :
  ∀ (candy_total Kate Robert Bill Mary : ℕ),
  candy_total = 20 →
  Kate = 4 →
  Robert = Kate + 2 →
  Bill = Mary - 6 →
  Kate = Bill + 2 →
  Mary > Robert →
  (Mary - Robert = 2) :=
by
  intros candy_total Kate Robert Bill Mary h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_candy_distribution_proof_l1896_189620


namespace NUMINAMATH_GPT_minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l1896_189622

theorem minimum_additional_games_to_reach_90_percent (N : ℕ) : 
  (2 + N) * 10 ≥ (5 + N) * 9 ↔ N ≥ 25 := 
sorry

-- An alternative approach to assert directly as exactly 25 by using the condition’s natural number ℕ could be as follows:
theorem hawks_minimum_games_needed_to_win (N : ℕ) : 
  ∀ N, (2 + N) * 10 / (5 + N) ≥ 9 / 10 → N ≥ 25 := 
sorry

end NUMINAMATH_GPT_minimum_additional_games_to_reach_90_percent_hawks_minimum_games_needed_to_win_l1896_189622


namespace NUMINAMATH_GPT_paul_taxes_and_fees_l1896_189627

theorem paul_taxes_and_fees 
  (hourly_wage: ℝ) 
  (hours_worked : ℕ)
  (spent_on_gummy_bears_percentage : ℝ)
  (final_amount : ℝ)
  (gross_earnings := hourly_wage * hours_worked)
  (taxes_and_fees := gross_earnings - final_amount / (1 - spent_on_gummy_bears_percentage)):
  hourly_wage = 12.50 →
  hours_worked = 40 →
  spent_on_gummy_bears_percentage = 0.15 →
  final_amount = 340 →
  taxes_and_fees / gross_earnings = 0.20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_paul_taxes_and_fees_l1896_189627


namespace NUMINAMATH_GPT_number_of_C_atoms_in_compound_is_4_l1896_189623

def atomic_weight_C : ℕ := 12
def atomic_weight_H : ℕ := 1
def atomic_weight_O : ℕ := 16

def molecular_weight : ℕ := 65

def weight_contributed_by_H_O : ℕ := atomic_weight_H + atomic_weight_O -- 17 amu

def weight_contributed_by_C : ℕ := molecular_weight - weight_contributed_by_H_O -- 48 amu

def number_of_C_atoms := weight_contributed_by_C / atomic_weight_C -- The quotient of 48 amu divided by 12 amu per C atom

theorem number_of_C_atoms_in_compound_is_4 : number_of_C_atoms = 4 :=
by
  sorry -- This is where the proof would go, but it's omitted as per instructions.

end NUMINAMATH_GPT_number_of_C_atoms_in_compound_is_4_l1896_189623


namespace NUMINAMATH_GPT_jane_change_l1896_189617

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end NUMINAMATH_GPT_jane_change_l1896_189617


namespace NUMINAMATH_GPT_concatenated_number_not_power_of_two_l1896_189695

theorem concatenated_number_not_power_of_two :
  ∀ (N : ℕ), (∀ i, 11111 ≤ i ∧ i ≤ 99999) →
  (N ≡ 0 [MOD 11111]) → ¬ ∃ k, N = 2^k :=
by
  sorry

end NUMINAMATH_GPT_concatenated_number_not_power_of_two_l1896_189695


namespace NUMINAMATH_GPT_hannah_age_is_48_l1896_189682

-- Define the ages of the brothers
def num_brothers : ℕ := 3
def age_each_brother : ℕ := 8

-- Define the sum of brothers' ages
def sum_brothers_ages : ℕ := num_brothers * age_each_brother

-- Define the age of Hannah
def hannah_age : ℕ := 2 * sum_brothers_ages

-- The theorem to prove Hannah's age is 48 years
theorem hannah_age_is_48 : hannah_age = 48 := by
  sorry

end NUMINAMATH_GPT_hannah_age_is_48_l1896_189682


namespace NUMINAMATH_GPT_problem_solution_l1896_189671

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def multiplicative_inverse (a m : ℕ) (inv : ℕ) : Prop := 
  (a * inv) % m = 1

theorem problem_solution :
  is_right_triangle 60 144 156 ∧ multiplicative_inverse 300 3751 3618 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1896_189671


namespace NUMINAMATH_GPT_divisibility_theorem_l1896_189674

theorem divisibility_theorem (n : ℕ) (h1 : n > 0) (h2 : ¬(2 ∣ n)) (h3 : ¬(3 ∣ n)) (k : ℤ) :
  (k + 1) ^ n - k ^ n - 1 ∣ k ^ 2 + k + 1 :=
sorry

end NUMINAMATH_GPT_divisibility_theorem_l1896_189674


namespace NUMINAMATH_GPT_set_C_cannot_form_right_triangle_l1896_189676

theorem set_C_cannot_form_right_triangle :
  ¬(5^2 + 2^2 = 5^2) :=
by
  sorry

end NUMINAMATH_GPT_set_C_cannot_form_right_triangle_l1896_189676


namespace NUMINAMATH_GPT_number_of_comedies_rented_l1896_189694

noncomputable def comedies_rented (r : ℕ) (a : ℕ) : ℕ := 3 * a

theorem number_of_comedies_rented (a : ℕ) (h : a = 5) : comedies_rented 3 a = 15 := by
  rw [h]
  exact rfl

end NUMINAMATH_GPT_number_of_comedies_rented_l1896_189694


namespace NUMINAMATH_GPT_order_of_numbers_l1896_189649

theorem order_of_numbers (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : 
  -m > n ∧ n > -n ∧ -n > m := 
by
  sorry

end NUMINAMATH_GPT_order_of_numbers_l1896_189649


namespace NUMINAMATH_GPT_freshmen_count_l1896_189655

theorem freshmen_count (n : ℕ) : n < 600 ∧ n % 25 = 24 ∧ n % 19 = 10 ↔ n = 574 := 
by sorry

end NUMINAMATH_GPT_freshmen_count_l1896_189655


namespace NUMINAMATH_GPT_find_xsq_plus_inv_xsq_l1896_189609

theorem find_xsq_plus_inv_xsq (x : ℝ) (h : 35 = x^6 + 1/(x^6)) : x^2 + 1/(x^2) = 37 :=
sorry

end NUMINAMATH_GPT_find_xsq_plus_inv_xsq_l1896_189609


namespace NUMINAMATH_GPT_wrappers_after_collection_l1896_189642

theorem wrappers_after_collection (caps_found : ℕ) (wrappers_found : ℕ) (current_caps : ℕ) (initial_caps : ℕ) : 
  caps_found = 22 → wrappers_found = 30 → current_caps = 17 → initial_caps = 0 → 
  wrappers_found ≥ 30 := 
by 
  intros h1 h2 h3 h4
  -- Solution steps are omitted on purpose
  --- This is where the proof is written
  sorry

end NUMINAMATH_GPT_wrappers_after_collection_l1896_189642


namespace NUMINAMATH_GPT_minimum_value_of_a_l1896_189625

theorem minimum_value_of_a (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2 - b * c) 
  (h2 : (1/2) * b * c * (Real.sin A) = (3 * Real.sqrt 3) / 4)
  (h3 : A = Real.arccos (1/2)) :
  a ≥ Real.sqrt 3 := sorry

end NUMINAMATH_GPT_minimum_value_of_a_l1896_189625


namespace NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l1896_189679

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l1896_189679


namespace NUMINAMATH_GPT_sequence_sum_l1896_189680

theorem sequence_sum (P Q R S T U V : ℕ) (h1 : S = 7)
  (h2 : P + Q + R = 21) (h3 : Q + R + S = 21)
  (h4 : R + S + T = 21) (h5 : S + T + U = 21)
  (h6 : T + U + V = 21) : P + V = 14 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l1896_189680


namespace NUMINAMATH_GPT_matrix_solution_correct_l1896_189630

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -7/3], ![4, -1/3]]

def v1 : Fin 2 → ℚ := ![4, 0]
def v2 : Fin 2 → ℚ := ![2, 3]

def result1 : Fin 2 → ℚ := ![12, 16]
def result2 : Fin 2 → ℚ := ![-1, 7]

theorem matrix_solution_correct :
  (mulVec N v1 = result1) ∧ 
  (mulVec N v2 = result2) := by
  sorry

end NUMINAMATH_GPT_matrix_solution_correct_l1896_189630


namespace NUMINAMATH_GPT_distance_between_towns_l1896_189615

theorem distance_between_towns 
  (rate1 rate2 : ℕ) (time : ℕ) (distance : ℕ)
  (h_rate1 : rate1 = 48)
  (h_rate2 : rate2 = 42)
  (h_time : time = 5)
  (h_distance : distance = rate1 * time + rate2 * time) : 
  distance = 450 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_towns_l1896_189615


namespace NUMINAMATH_GPT_log_comparison_l1896_189605

theorem log_comparison :
  (Real.log 80 / Real.log 20) < (Real.log 640 / Real.log 80) :=
by
  sorry

end NUMINAMATH_GPT_log_comparison_l1896_189605


namespace NUMINAMATH_GPT_find_positive_integer_M_l1896_189672

theorem find_positive_integer_M (M : ℕ) (h : 36^2 * 81^2 = 18^2 * M^2) : M = 162 := by
  sorry

end NUMINAMATH_GPT_find_positive_integer_M_l1896_189672


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1896_189666

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_q : q = -1 / 2) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1896_189666


namespace NUMINAMATH_GPT_binary_to_decimal_101101_l1896_189610

theorem binary_to_decimal_101101 : 
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 :=
by
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  have h : (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 := sorry
  exact h

end NUMINAMATH_GPT_binary_to_decimal_101101_l1896_189610


namespace NUMINAMATH_GPT_tan_ratio_proof_l1896_189602

theorem tan_ratio_proof (α : ℝ) (h : 5 * Real.sin (2 * α) = Real.sin 2) : 
  Real.tan (α + 1 * Real.pi / 180) / Real.tan (α - 1 * Real.pi / 180) = - 3 / 2 := 
sorry

end NUMINAMATH_GPT_tan_ratio_proof_l1896_189602


namespace NUMINAMATH_GPT_geometric_sum_5_l1896_189636

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = a n * r ∧ a (m + 1) = a m * r

theorem geometric_sum_5 (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) (h3 : ∀ n, 0 < a n) :
  a 3 + a 5 = 5 :=
sorry

end NUMINAMATH_GPT_geometric_sum_5_l1896_189636


namespace NUMINAMATH_GPT_students_at_end_of_year_l1896_189657

def students_start : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0
def students_end : ℝ := 28.0

theorem students_at_end_of_year :
  students_start - students_left - students_transferred = students_end := by
  sorry

end NUMINAMATH_GPT_students_at_end_of_year_l1896_189657


namespace NUMINAMATH_GPT_skittles_left_l1896_189675

theorem skittles_left (initial_skittles : ℕ) (skittles_given : ℕ) (final_skittles : ℕ) :
  initial_skittles = 50 → skittles_given = 7 → final_skittles = initial_skittles - skittles_given → final_skittles = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_skittles_left_l1896_189675


namespace NUMINAMATH_GPT_point_on_circle_l1896_189640

theorem point_on_circle (t : ℝ) : 
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  x^2 + y^2 = 1 :=
by
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  sorry

end NUMINAMATH_GPT_point_on_circle_l1896_189640


namespace NUMINAMATH_GPT_find_z_given_x4_l1896_189650

theorem find_z_given_x4 (k : ℝ) (z : ℝ) (x : ℝ) :
  (7 * 4 = k / 2^3) → (7 * z = k / x^3) → (x = 4) → (z = 0.5) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_z_given_x4_l1896_189650


namespace NUMINAMATH_GPT_coopers_daily_pie_count_l1896_189677

-- Definitions of conditions
def total_pies_made_per_day (x : ℕ) : ℕ := x
def days := 12
def pies_eaten_by_ashley := 50
def remaining_pies := 34

-- Lean 4 statement of the problem to prove
theorem coopers_daily_pie_count (x : ℕ) : 
  12 * total_pies_made_per_day x - pies_eaten_by_ashley = remaining_pies → 
  x = 7 := 
by
  intro h
  -- Solution steps (not included in the theorem)
  -- Given proof follows from the Lean 4 statement
  sorry

end NUMINAMATH_GPT_coopers_daily_pie_count_l1896_189677


namespace NUMINAMATH_GPT_xy_solution_l1896_189690

theorem xy_solution (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end NUMINAMATH_GPT_xy_solution_l1896_189690


namespace NUMINAMATH_GPT_female_adults_present_l1896_189692

variable (children : ℕ) (male_adults : ℕ) (total_people : ℕ)
variable (children_count : children = 80) (male_adults_count : male_adults = 60) (total_people_count : total_people = 200)

theorem female_adults_present : ∃ (female_adults : ℕ), 
  female_adults = total_people - (children + male_adults) ∧ 
  female_adults = 60 :=
by
  sorry

end NUMINAMATH_GPT_female_adults_present_l1896_189692


namespace NUMINAMATH_GPT_quadratic_polynomial_discriminant_l1896_189662

theorem quadratic_polynomial_discriminant (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : ∃ x : ℝ, a * x^2 + b * x + c = x - 2 ∧ (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h₃ : ∃ x : ℝ, a * x^2 + b * x + c = 1 - x / 2 ∧ (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
sorry

end NUMINAMATH_GPT_quadratic_polynomial_discriminant_l1896_189662


namespace NUMINAMATH_GPT_total_volume_tetrahedra_l1896_189698

theorem total_volume_tetrahedra (side_length : ℝ) (x : ℝ) (sqrt_2 : ℝ := Real.sqrt 2) 
  (cube_to_octa_length : x = 2 * (sqrt_2 - 1)) 
  (volume_of_one_tetra : ℝ := ((6 - 4 * sqrt_2) * (3 - sqrt_2)) / 6) :
  side_length = 2 → 
  8 * volume_of_one_tetra = (104 - 72 * sqrt_2) / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_volume_tetrahedra_l1896_189698


namespace NUMINAMATH_GPT_root_monotonicity_l1896_189651

noncomputable def f (x : ℝ) := 3^x + 2 / (1 - x)

theorem root_monotonicity
  (x0 : ℝ) (H_root : f x0 = 0)
  (x1 x2 : ℝ) (H1 : x1 > 1) (H2 : x1 < x0) (H3 : x2 > x0) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_root_monotonicity_l1896_189651


namespace NUMINAMATH_GPT_smallest_sum_l1896_189638

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end NUMINAMATH_GPT_smallest_sum_l1896_189638


namespace NUMINAMATH_GPT_transport_connectivity_l1896_189633

-- Define the condition that any two cities are connected by either an air route or a canal.
-- We will formalize this with an inductive type to represent the transport means: AirRoute or Canal.
inductive TransportMeans
| AirRoute : TransportMeans
| Canal : TransportMeans

open TransportMeans

-- Represent cities as a type 'City'
universe u
variable (City : Type u)

-- Connect any two cities by a transport means
variable (connected : City → City → TransportMeans)

-- We want to prove that for any set of cities, 
-- there exists a means of transport such that starting from any city,
-- it is possible to reach any other city using only that means of transport.
theorem transport_connectivity (n : ℕ) (h2 : n ≥ 2) : 
  ∃ (T : TransportMeans), ∀ (c1 c2 : City), connected c1 c2 = T :=
by
  sorry

end NUMINAMATH_GPT_transport_connectivity_l1896_189633


namespace NUMINAMATH_GPT_find_smallest_M_l1896_189614

/-- 
Proof of the smallest real number M such that 
for all real numbers a, b, and c, the following inequality holds:
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)|
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2. 
-/
theorem find_smallest_M (a b c : ℝ) : 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_M_l1896_189614


namespace NUMINAMATH_GPT_problem_inequality_l1896_189603

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem problem_inequality (a : ℝ) (m n : ℝ) 
  (h1 : m ∈ Set.Icc 0 2) (h2 : n ∈ Set.Icc 0 2) 
  (h3 : |m - n| ≥ 1) 
  (h4 : f m a / f n a = 1) : 
  1 ≤ a / (Real.exp 1 - 1) ∧ a / (Real.exp 1 - 1) ≤ Real.exp 1 :=
by sorry

end NUMINAMATH_GPT_problem_inequality_l1896_189603


namespace NUMINAMATH_GPT_parameterization_of_line_l1896_189647

theorem parameterization_of_line : 
  ∀ (r k : ℝ),
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (r, 2) + t • (3, k)) → y = 2 * x - 6) → (r = 4 ∧ k = 6) :=
by
  sorry

end NUMINAMATH_GPT_parameterization_of_line_l1896_189647


namespace NUMINAMATH_GPT_pebbles_sum_at_12_days_l1896_189658

def pebbles_collected (n : ℕ) : ℕ :=
  if n = 0 then 0 else n + pebbles_collected (n - 1)

theorem pebbles_sum_at_12_days : pebbles_collected 12 = 78 := by
  -- This would be the place for the proof, but adding sorry as instructed.
  sorry

end NUMINAMATH_GPT_pebbles_sum_at_12_days_l1896_189658


namespace NUMINAMATH_GPT_inequality_ge_zero_l1896_189635

theorem inequality_ge_zero (x y z : ℝ) : 
  4 * x * (x + y) * (x + z) * (x + y + z) + y^2 * z^2 ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_ge_zero_l1896_189635


namespace NUMINAMATH_GPT_part1_part2_l1896_189665

-- Statements derived from Step c)
theorem part1 {m : ℝ} (h : ∃ x : ℝ, m - |5 - 2 * x| - |2 * x - 1| = 0) : 4 ≤ m := by
  sorry

theorem part2 {x : ℝ} (hx : |x - 3| + |x + 4| ≤ 8) : -9 / 2 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1896_189665


namespace NUMINAMATH_GPT_correct_option_l1896_189652

theorem correct_option : (-1 - 3 = -4) ∧ ¬(-2 + 8 = 10) ∧ ¬(-2 * 2 = 4) ∧ ¬(-8 / -1 = -1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l1896_189652


namespace NUMINAMATH_GPT_difference_in_perimeters_of_rectangles_l1896_189628

theorem difference_in_perimeters_of_rectangles 
  (l h : ℝ) (hl : l ≥ 0) (hh : h ≥ 0) :
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  difference = 24 :=
by
  let length_outer := 7
  let height_outer := 5
  let perimeter_outer := 2 * (length_outer + height_outer)
  let perimeter_inner := 2 * (l + h)
  let difference := perimeter_outer - perimeter_inner
  sorry

end NUMINAMATH_GPT_difference_in_perimeters_of_rectangles_l1896_189628


namespace NUMINAMATH_GPT_digit_solve_l1896_189619

theorem digit_solve : ∀ (D : ℕ), D < 10 → (D * 9 + 6 = D * 10 + 3) → D = 3 :=
by
  intros D hD h
  sorry

end NUMINAMATH_GPT_digit_solve_l1896_189619


namespace NUMINAMATH_GPT_smallest_solution_l1896_189688

theorem smallest_solution (x : ℝ) : (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) → x = 4 - (Real.sqrt 15) / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_l1896_189688


namespace NUMINAMATH_GPT_is_condition_B_an_algorithm_l1896_189685

-- Definitions of conditions A, B, C, D
def condition_A := "At home, it is generally the mother who cooks"
def condition_B := "The steps to cook rice include washing the pot, rinsing the rice, adding water, and heating"
def condition_C := "Cooking outdoors is called camping cooking"
def condition_D := "Rice is necessary for cooking"

-- Definition of being considered an algorithm
def is_algorithm (s : String) : Prop :=
  s = condition_B  -- Based on the analysis that condition_B meets the criteria of an algorithm

-- The proof statement to show that condition_B can be considered an algorithm
theorem is_condition_B_an_algorithm : is_algorithm condition_B :=
by
  sorry

end NUMINAMATH_GPT_is_condition_B_an_algorithm_l1896_189685


namespace NUMINAMATH_GPT_first_issue_pages_l1896_189604

-- Define the conditions
def total_pages := 220
def pages_third_issue (x : ℕ) := x + 4

-- Statement of the problem
theorem first_issue_pages (x : ℕ) (hx : 3 * x + 4 = total_pages) : x = 72 :=
sorry

end NUMINAMATH_GPT_first_issue_pages_l1896_189604


namespace NUMINAMATH_GPT_C_share_l1896_189654

theorem C_share (a b c : ℕ) (h1 : a + b + c = 1010)
                (h2 : ∃ k : ℕ, a = 3 * k + 25 ∧ b = 2 * k + 10 ∧ c = 5 * k + 15) : c = 495 :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_C_share_l1896_189654


namespace NUMINAMATH_GPT_tunnel_digging_duration_l1896_189669

theorem tunnel_digging_duration (daily_progress : ℕ) (total_length_km : ℕ) 
    (meters_per_km : ℕ) (days_per_year : ℕ) : 
    daily_progress = 5 → total_length_km = 2 → meters_per_km = 1000 → days_per_year = 365 → 
    total_length_km * meters_per_km / daily_progress > 365 :=
by
  intros hprog htunnel hmeters hdays
  /- ... proof steps will go here -/
  sorry

end NUMINAMATH_GPT_tunnel_digging_duration_l1896_189669


namespace NUMINAMATH_GPT_largest_package_markers_l1896_189696

def Alex_markers : ℕ := 36
def Becca_markers : ℕ := 45
def Charlie_markers : ℕ := 60

theorem largest_package_markers (d : ℕ) :
  d ∣ Alex_markers ∧ d ∣ Becca_markers ∧ d ∣ Charlie_markers → d ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_package_markers_l1896_189696


namespace NUMINAMATH_GPT_initial_percentage_of_chemical_x_l1896_189684

theorem initial_percentage_of_chemical_x (P : ℝ) (h1 : 20 + 80 * P = 44) : P = 0.3 :=
by sorry

end NUMINAMATH_GPT_initial_percentage_of_chemical_x_l1896_189684


namespace NUMINAMATH_GPT_trig_problem_l1896_189659

variables (θ : ℝ)

theorem trig_problem (h : Real.sin (2 * θ) = 1 / 2) : 
  Real.tan θ + 1 / Real.tan θ = 4 :=
sorry

end NUMINAMATH_GPT_trig_problem_l1896_189659


namespace NUMINAMATH_GPT_sum_of_ages_is_29_l1896_189618

theorem sum_of_ages_is_29 (age1 age2 age3 : ℕ) (h1 : age1 = 9) (h2 : age2 = 9) (h3 : age3 = 11) :
  age1 + age2 + age3 = 29 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_29_l1896_189618


namespace NUMINAMATH_GPT_find_a_in_third_quadrant_l1896_189670

theorem find_a_in_third_quadrant :
  ∃ a : ℝ, a < 0 ∧ 3 * a^2 + 4 * a^2 = 28 ∧ a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_in_third_quadrant_l1896_189670


namespace NUMINAMATH_GPT_value_of_a0_plus_a8_l1896_189637

/-- Theorem stating the value of a0 + a8 from the given polynomial equation -/
theorem value_of_a0_plus_a8 (a_0 a_8 : ℤ) :
  (∀ x : ℤ, (1 + x) ^ 10 = a_0 + a_1 * (1 - x) + a_2 * (1 - x) ^ 2 + 
              a_3 * (1 - x) ^ 3 + a_4 * (1 - x) ^ 4 + a_5 * (1 - x) ^ 5 +
              a_6 * (1 - x) ^ 6 + a_7 * (1 - x) ^ 7 + a_8 * (1 - x) ^ 8 + 
              a_9 * (1 - x) ^ 9 + a_10 * (1 - x) ^ 10) →
  a_0 + a_8 = 1204 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a0_plus_a8_l1896_189637


namespace NUMINAMATH_GPT_nested_series_sum_l1896_189631

theorem nested_series_sum : 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))) = 126 :=
by
  sorry

end NUMINAMATH_GPT_nested_series_sum_l1896_189631


namespace NUMINAMATH_GPT_total_legs_l1896_189648

-- Define the number of each type of animal
def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goat : ℕ := 1

-- Define the number of legs per animal
def legs_per_animal : ℕ := 4

-- Define the total number of legs for each type of animal
def horse_legs : ℕ := num_horses * legs_per_animal
def dog_legs : ℕ := num_dogs * legs_per_animal
def cat_legs : ℕ := num_cats * legs_per_animal
def turtle_legs : ℕ := num_turtles * legs_per_animal
def goat_legs : ℕ := num_goat * legs_per_animal

-- Define the problem statement
theorem total_legs : horse_legs + dog_legs + cat_legs + turtle_legs + goat_legs = 72 := by
  -- Sum up all the leg counts
  sorry

end NUMINAMATH_GPT_total_legs_l1896_189648


namespace NUMINAMATH_GPT_worker_late_time_l1896_189667

noncomputable def usual_time : ℕ := 60
noncomputable def speed_factor : ℚ := 4 / 5

theorem worker_late_time (T T_new : ℕ) (S : ℚ) :
  T = usual_time →
  T = 60 →
  T_new = (5 / 4) * T →
  T_new - T = 15 :=
by
  intros
  subst T
  sorry

end NUMINAMATH_GPT_worker_late_time_l1896_189667


namespace NUMINAMATH_GPT_polynomial_div_remainder_l1896_189645

theorem polynomial_div_remainder (x : ℝ) : 
  (x^4 % (x^2 + 7*x + 2)) = -315*x - 94 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_div_remainder_l1896_189645


namespace NUMINAMATH_GPT_find_certain_number_l1896_189689

theorem find_certain_number (x : ℝ) (h : 25 * x = 675) : x = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_certain_number_l1896_189689


namespace NUMINAMATH_GPT_number_of_students_l1896_189678

theorem number_of_students (groups : ℕ) (students_per_group : ℕ) (minutes_per_student : ℕ) (minutes_per_group : ℕ) :
    groups = 3 →
    minutes_per_student = 4 →
    minutes_per_group = 24 →
    minutes_per_group = students_per_group * minutes_per_student →
    18 = groups * students_per_group :=
by
  intros h_groups h_minutes_per_student h_minutes_per_group h_relation
  sorry

end NUMINAMATH_GPT_number_of_students_l1896_189678


namespace NUMINAMATH_GPT_a_eq_3x_or_neg2x_l1896_189624

theorem a_eq_3x_or_neg2x (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 19 * x^3) (h3 : a - b = x) :
    a = 3 * x ∨ a = -2 * x :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_a_eq_3x_or_neg2x_l1896_189624


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1896_189600

theorem arithmetic_sequence_fifth_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 6) (h3 : a 3 = 2) (h_arith_seq : ∀ n, a (n + 1) = a n + d) : a 5 = -2 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1896_189600


namespace NUMINAMATH_GPT_Pascal_remaining_distance_l1896_189616

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Pascal_remaining_distance_l1896_189616


namespace NUMINAMATH_GPT_laser_beam_total_distance_l1896_189643

theorem laser_beam_total_distance :
  let A := (3, 5)
  let D := (7, 5)
  let D'' := (-7, -5)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  distance A D'' = 10 * Real.sqrt 2 :=
by
  -- definitions and conditions are captured
  sorry -- the proof goes here, no proof is required as per instructions

end NUMINAMATH_GPT_laser_beam_total_distance_l1896_189643


namespace NUMINAMATH_GPT_factorization1_factorization2_l1896_189693

-- Definitions for the first problem
def expr1 (x : ℝ) := 3 * x^2 - 12
def factorized_form1 (x : ℝ) := 3 * (x + 2) * (x - 2)

-- Theorem for the first problem
theorem factorization1 (x : ℝ) : expr1 x = factorized_form1 x :=
  sorry

-- Definitions for the second problem
def expr2 (a x y : ℝ) := a * x^2 - 4 * a * x * y + 4 * a * y^2
def factorized_form2 (a x y : ℝ) := a * (x - 2 * y) * (x - 2 * y)

-- Theorem for the second problem
theorem factorization2 (a x y : ℝ) : expr2 a x y = factorized_form2 a x y :=
  sorry

end NUMINAMATH_GPT_factorization1_factorization2_l1896_189693


namespace NUMINAMATH_GPT_function_range_is_correct_l1896_189606

noncomputable def function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.log (x^2 - 6 * x + 17) }

theorem function_range_is_correct : function_range = {x : ℝ | x ≤ Real.log 8} :=
by
  sorry

end NUMINAMATH_GPT_function_range_is_correct_l1896_189606


namespace NUMINAMATH_GPT_didi_total_fund_l1896_189601

-- Define the conditions
def cakes : ℕ := 10
def slices_per_cake : ℕ := 8
def price_per_slice : ℕ := 1
def first_business_owner_donation_per_slice : ℚ := 0.5
def second_business_owner_donation_per_slice : ℚ := 0.25

-- Define the proof problem statement
theorem didi_total_fund (h1 : cakes * slices_per_cake = 80)
    (h2 : (80 : ℕ) * price_per_slice = 80)
    (h3 : (80 : ℕ) * first_business_owner_donation_per_slice = 40)
    (h4 : (80 : ℕ) * second_business_owner_donation_per_slice = 20) : 
    (80 : ℕ) + 40 + 20 = 140 := by
  -- The proof itself will be constructed here
  sorry

end NUMINAMATH_GPT_didi_total_fund_l1896_189601


namespace NUMINAMATH_GPT_scale_model_height_l1896_189673

theorem scale_model_height :
  let scale_ratio : ℚ := 1 / 25
  let actual_height : ℚ := 151
  let model_height : ℚ := actual_height * scale_ratio
  round model_height = 6 :=
by
  sorry

end NUMINAMATH_GPT_scale_model_height_l1896_189673


namespace NUMINAMATH_GPT_f_log2_9_eq_neg_16_div_9_l1896_189641

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x - 2) = f x
axiom f_range_0_1 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2 ^ x

theorem f_log2_9_eq_neg_16_div_9 : f (Real.log 9 / Real.log 2) = -16 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_f_log2_9_eq_neg_16_div_9_l1896_189641


namespace NUMINAMATH_GPT_area_of_triangle_BXC_l1896_189612

-- Define a trapezoid ABCD with given conditions
structure Trapezoid :=
  (A B C D X : Type)
  (AB CD : ℝ)
  (area_ABCD : ℝ)
  (intersect_at_X : Prop)

theorem area_of_triangle_BXC (t : Trapezoid) (h1 : t.AB = 24) (h2 : t.CD = 40)
  (h3 : t.area_ABCD = 480) (h4 : t.intersect_at_X) : 
  ∃ (area_BXC : ℝ), area_BXC = 120 :=
by {
  -- skip the proof here by using sorry
  sorry
}

end NUMINAMATH_GPT_area_of_triangle_BXC_l1896_189612


namespace NUMINAMATH_GPT_ratio_mercedes_jonathan_l1896_189660

theorem ratio_mercedes_jonathan (M : ℝ) (J : ℝ) (D : ℝ) 
  (h1 : J = 7.5) 
  (h2 : D = M + 2) 
  (h3 : M + D = 32) : M / J = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_mercedes_jonathan_l1896_189660


namespace NUMINAMATH_GPT_factorize_expression_l1896_189646

theorem factorize_expression (x : ℝ) : x^2 - 2023 * x = x * (x - 2023) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l1896_189646


namespace NUMINAMATH_GPT_discriminant_zero_l1896_189686

theorem discriminant_zero (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -2) (h₃ : c = 1) :
  (b^2 - 4 * a * c) = 0 :=
by
  sorry

end NUMINAMATH_GPT_discriminant_zero_l1896_189686


namespace NUMINAMATH_GPT_distance_AD_btw_41_and_42_l1896_189632

noncomputable def distance_between (x y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

theorem distance_AD_btw_41_and_42 :
  let A := (0, 0)
  let B := (15, 0)
  let C := (15, 5 * Real.sqrt 3)
  let D := (15, 5 * Real.sqrt 3 + 30)

  41 < distance_between A D ∧ distance_between A D < 42 :=
by
  sorry

end NUMINAMATH_GPT_distance_AD_btw_41_and_42_l1896_189632


namespace NUMINAMATH_GPT_sqrt_90000_eq_300_l1896_189613

theorem sqrt_90000_eq_300 : Real.sqrt 90000 = 300 := by
  sorry

end NUMINAMATH_GPT_sqrt_90000_eq_300_l1896_189613


namespace NUMINAMATH_GPT_min_cans_needed_l1896_189681

theorem min_cans_needed (oz_per_can : ℕ) (total_oz_needed : ℕ) (H1 : oz_per_can = 15) (H2 : total_oz_needed = 150) :
  ∃ n : ℕ, 15 * n ≥ 150 ∧ ∀ m : ℕ, 15 * m ≥ 150 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_min_cans_needed_l1896_189681


namespace NUMINAMATH_GPT_find_interest_rate_l1896_189634

-- Conditions
def principal1 : ℝ := 100
def rate1 : ℝ := 0.05
def time1 : ℕ := 48

def principal2 : ℝ := 600
def time2 : ℕ := 4

-- The given interest produced by the first amount
def interest1 : ℝ := principal1 * rate1 * time1

-- The interest produced by the second amount should be the same
def interest2 (rate2 : ℝ) : ℝ := principal2 * rate2 * time2

-- The interest rate to prove
def rate2_correct : ℝ := 0.1

theorem find_interest_rate :
  ∃ rate2 : ℝ, interest2 rate2 = interest1 ∧ rate2 = rate2_correct :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l1896_189634


namespace NUMINAMATH_GPT_fill_in_the_blank_l1896_189639

theorem fill_in_the_blank (x : ℕ) (h : (x - x) + x * x + x / x = 50) : x = 7 :=
sorry

end NUMINAMATH_GPT_fill_in_the_blank_l1896_189639


namespace NUMINAMATH_GPT_find_common_ratio_l1896_189656

theorem find_common_ratio (a_1 q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS1 : S 1 = a_1)
  (hS2 : S 2 = a_1 * (1 + q))
  (hS3 : S 3 = a_1 * (1 + q + q^2))
  (ha2 : a 2 = a_1 * q)
  (ha3 : a 3 = a_1 * q^2)
  (hcond : 2 * (S 1 + 2 * a 2) = S 3 + a 3 + S 2 + a 2) :
  q = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l1896_189656


namespace NUMINAMATH_GPT_sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l1896_189608

theorem sum_of_roots_eq_zero (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 + x2 = 0 :=
by
  sorry

theorem product_of_roots_eq_neg_twentyfive (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 * x2 = -25 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l1896_189608


namespace NUMINAMATH_GPT_speaker_discounted_price_correct_l1896_189644

-- Define the initial price and the discount
def initial_price : ℝ := 475.00
def discount : ℝ := 276.00

-- Define the discounted price
def discounted_price : ℝ := initial_price - discount

-- The theorem to prove that the discounted price is 199.00
theorem speaker_discounted_price_correct : discounted_price = 199.00 :=
by
  -- Proof is omitted here, adding sorry to indicate it.
  sorry

end NUMINAMATH_GPT_speaker_discounted_price_correct_l1896_189644


namespace NUMINAMATH_GPT_reduced_price_per_dozen_l1896_189664

variables {P R : ℝ}

theorem reduced_price_per_dozen
  (H1 : R = 0.6 * P)
  (H2 : 40 / P - 40 / R = 64) :
  R = 3 := 
sorry

end NUMINAMATH_GPT_reduced_price_per_dozen_l1896_189664


namespace NUMINAMATH_GPT_smallest_a_divisible_by_1984_l1896_189629

theorem smallest_a_divisible_by_1984 :
  ∃ a : ℕ, (∀ n : ℕ, n % 2 = 1 → 1984 ∣ (47^n + a * 15^n)) ∧ a = 1055 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_a_divisible_by_1984_l1896_189629


namespace NUMINAMATH_GPT_reciprocal_of_sum_l1896_189697

theorem reciprocal_of_sum : (1 / (1 / 3 + 1 / 4)) = 12 / 7 := 
by sorry

end NUMINAMATH_GPT_reciprocal_of_sum_l1896_189697


namespace NUMINAMATH_GPT_highway_extension_l1896_189611

theorem highway_extension 
  (current_length : ℕ) 
  (desired_length : ℕ) 
  (first_day_miles : ℕ) 
  (miles_needed : ℕ) 
  (second_day_miles : ℕ) 
  (h1 : current_length = 200) 
  (h2 : desired_length = 650) 
  (h3 : first_day_miles = 50) 
  (h4 : miles_needed = 250) 
  (h5 : second_day_miles = desired_length - current_length - miles_needed - first_day_miles) :
  second_day_miles / first_day_miles = 3 := 
sorry

end NUMINAMATH_GPT_highway_extension_l1896_189611


namespace NUMINAMATH_GPT_exists_n_satisfying_conditions_l1896_189661

open Nat

-- Define that n satisfies the given conditions
theorem exists_n_satisfying_conditions :
  ∃ (n : ℤ), (∃ (k : ℤ), 2 * n + 1 = (2 * k + 1) ^ 2) ∧ 
            (∃ (h : ℤ), 3 * n + 1 = (2 * h + 1) ^ 2) ∧ 
            (40 ∣ n) := by
  sorry

end NUMINAMATH_GPT_exists_n_satisfying_conditions_l1896_189661


namespace NUMINAMATH_GPT_problem_statement_l1896_189668

-- Definitions of parallel and perpendicular predicates (should be axioms or definitions in the context)
-- For simplification, assume we have a space with lines and planes, with corresponding relations.

axiom Line : Type
axiom Plane : Type
axiom parallel : Line → Line → Prop
axiom perpendicular : Line → Plane → Prop
axiom subset : Line → Plane → Prop

-- Assume the necessary conditions: m and n are lines, a and b are planes, with given relationships.
variables (m n : Line) (a b : Plane)

-- The conditions given.
variables (m_parallel_n : parallel m n)
variables (m_perpendicular_a : perpendicular m a)

-- The proposition to prove: If m parallel n and m perpendicular to a, then n is perpendicular to a.
theorem problem_statement : perpendicular n a :=
sorry

end NUMINAMATH_GPT_problem_statement_l1896_189668


namespace NUMINAMATH_GPT_number_of_candies_picked_up_l1896_189626

-- Definitions of the conditions
def num_sides_decagon := 10
def diagonals_from_one_vertex (n : Nat) : Nat := n - 3

-- The theorem stating the number of candies Hyeonsu picked up
theorem number_of_candies_picked_up : diagonals_from_one_vertex num_sides_decagon = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_candies_picked_up_l1896_189626


namespace NUMINAMATH_GPT_container_capacity_l1896_189687

-- Define the given conditions
def initially_full (x : ℝ) : Prop := (1 / 4) * x + 300 = (3 / 4) * x

-- Define the proof problem to show that the total capacity is 600 liters
theorem container_capacity : ∃ x : ℝ, initially_full x → x = 600 := sorry

end NUMINAMATH_GPT_container_capacity_l1896_189687


namespace NUMINAMATH_GPT_discount_on_pickles_l1896_189607

theorem discount_on_pickles :
  ∀ (meat_weight : ℝ) (meat_price_per_pound : ℝ) (bun_price : ℝ) (lettuce_price : ℝ)
    (tomato_weight : ℝ) (tomato_price_per_pound : ℝ) (pickles_price : ℝ) (total_paid : ℝ) (change : ℝ),
  meat_weight = 2 ∧
  meat_price_per_pound = 3.50 ∧
  bun_price = 1.50 ∧
  lettuce_price = 1.00 ∧
  tomato_weight = 1.5 ∧
  tomato_price_per_pound = 2.00 ∧
  pickles_price = 2.50 ∧
  total_paid = 20.00 ∧
  change = 6 →
  pickles_price - (total_paid - change - (meat_weight * meat_price_per_pound + tomato_weight * tomato_price_per_pound + bun_price + lettuce_price)) = 1 := 
by
  -- Begin the proof here (not required for this task)
  sorry

end NUMINAMATH_GPT_discount_on_pickles_l1896_189607


namespace NUMINAMATH_GPT_angle_B_is_180_l1896_189699

variables {l k : Line} {A B C: Point}

def parallel (l k : Line) : Prop := sorry 
def angle (A B C : Point) : ℝ := sorry

theorem angle_B_is_180 (h1 : parallel l k) (h2 : angle A = 110) (h3 : angle C = 70) :
  angle B = 180 := 
by
  sorry

end NUMINAMATH_GPT_angle_B_is_180_l1896_189699
