import Mathlib

namespace NUMINAMATH_GPT_range_of_k_l1904_190409

def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem range_of_k (k : ℝ) : (∀ x : ℝ, tensor k x > 0) ↔ (0 < k ∧ k < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1904_190409


namespace NUMINAMATH_GPT_trajectory_of_centroid_l1904_190400

def foci (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (0, 1) ∧ F2 = (0, -1)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 3) + (P.2^2 / 4) = 1

def centroid_eq (G : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, on_ellipse P ∧ 
  foci (0, 1) (0, -1) ∧ 
  G = (P.1 / 3, (1 + -1 + P.2) / 3)

theorem trajectory_of_centroid :
  ∀ G : ℝ × ℝ, (centroid_eq G → 3 * G.1^2 + (9 * G.2^2) / 4 = 1 ∧ G.1 ≠ 0) :=
by 
  intros G h
  sorry

end NUMINAMATH_GPT_trajectory_of_centroid_l1904_190400


namespace NUMINAMATH_GPT_symmetric_point_coords_l1904_190445

def pointA : ℝ × ℝ := (1, 2)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def pointB : ℝ × ℝ := translate_left pointA 2

def pointC : ℝ × ℝ := reflect_origin pointB

theorem symmetric_point_coords :
  pointC = (1, -2) :=
by
  -- Proof omitted as instructed
  sorry

end NUMINAMATH_GPT_symmetric_point_coords_l1904_190445


namespace NUMINAMATH_GPT_men_work_problem_l1904_190435

theorem men_work_problem (x : ℕ) (h1 : x * 70 = 40 * 63) : x = 36 := 
by
  sorry

end NUMINAMATH_GPT_men_work_problem_l1904_190435


namespace NUMINAMATH_GPT_check_conditions_l1904_190429

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) := n * a + (n * (n - 1) / 2) * d

theorem check_conditions {a d : ℤ}
  (S6 S7 S5 : ℤ)
  (h1 : S6 = sum_of_first_n_terms a d 6)
  (h2 : S7 = sum_of_first_n_terms a d 7)
  (h3 : S5 = sum_of_first_n_terms a d 5)
  (h : S6 > S7 ∧ S7 > S5) :
  d < 0 ∧
  sum_of_first_n_terms a d 11 > 0 ∧
  sum_of_first_n_terms a d 13 < 0 ∧
  sum_of_first_n_terms a d 9 > sum_of_first_n_terms a d 3 := 
sorry

end NUMINAMATH_GPT_check_conditions_l1904_190429


namespace NUMINAMATH_GPT_n_is_prime_l1904_190419

theorem n_is_prime (p : ℕ) (h : ℕ) (n : ℕ)
  (hp : Nat.Prime p)
  (hh : h < p)
  (hn : n = p * h + 1)
  (div_n : n ∣ (2^(n-1) - 1))
  (not_div_n : ¬ n ∣ (2^h - 1)) : Nat.Prime n := sorry

end NUMINAMATH_GPT_n_is_prime_l1904_190419


namespace NUMINAMATH_GPT_Connie_correct_number_l1904_190493

theorem Connie_correct_number (x : ℤ) (h : x + 2 = 80) : x - 2 = 76 := by
  sorry

end NUMINAMATH_GPT_Connie_correct_number_l1904_190493


namespace NUMINAMATH_GPT_nat_divisibility_l1904_190415

theorem nat_divisibility {n : ℕ} : (n + 1 ∣ n^2 + 1) ↔ (n = 0 ∨ n = 1) := 
sorry

end NUMINAMATH_GPT_nat_divisibility_l1904_190415


namespace NUMINAMATH_GPT_total_blocks_l1904_190479

def initial_blocks := 2
def multiplier := 3
def father_blocks := multiplier * initial_blocks

theorem total_blocks :
  initial_blocks + father_blocks = 8 :=
by 
  -- skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_total_blocks_l1904_190479


namespace NUMINAMATH_GPT_elephant_distribution_l1904_190414

theorem elephant_distribution (unions nonunions : ℕ) (elephants : ℕ) :
  unions = 28 ∧ nonunions = 37 ∧ (∀ k : ℕ, elephants = 28 * k ∨ elephants = 37 * k) ∧ (∀ k : ℕ, ((28 * k ≤ elephants) ∧ (37 * k ≤ elephants))) → 
  elephants = 2072 :=
by
  sorry

end NUMINAMATH_GPT_elephant_distribution_l1904_190414


namespace NUMINAMATH_GPT_numerology_eq_l1904_190404

theorem numerology_eq : 2222 - 222 + 22 - 2 = 2020 :=
by
  sorry

end NUMINAMATH_GPT_numerology_eq_l1904_190404


namespace NUMINAMATH_GPT_exists_j_half_for_all_j_l1904_190424

def is_j_half (n j : ℕ) : Prop := 
  ∃ (q : ℕ), n = (2 * j + 1) * q + j

theorem exists_j_half_for_all_j (k : ℕ) : 
  ∃ n : ℕ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ k → is_j_half n j :=
by
  sorry

end NUMINAMATH_GPT_exists_j_half_for_all_j_l1904_190424


namespace NUMINAMATH_GPT_prove_heron_formula_prove_S_squared_rrarc_l1904_190471

variables {r r_a r_b r_c p a b c S : ℝ}

-- Problem 1: Prove Heron's Formula
theorem prove_heron_formula (h1 : r * p = r_a * (p - a))
                            (h2 : r * r_a = (p - b) * (p - c))
                            (h3 : r_b * r_c = p * (p - a)) :
  S^2 = p * (p - a) * (p - b) * (p - c) :=
sorry

-- Problem 2: Prove S^2 = r * r_a * r_b * r_c
theorem prove_S_squared_rrarc (h1 : r * p = r_a * (p - a))
                              (h2 : r * r_a = (p - b) * (p - c))
                              (h3 : r_b * r_c = p * (p - a)) :
  S^2 = r * r_a * r_b * r_c :=
sorry

end NUMINAMATH_GPT_prove_heron_formula_prove_S_squared_rrarc_l1904_190471


namespace NUMINAMATH_GPT_find_p_q_sum_l1904_190498

noncomputable def roots (r1 r2 r3 : ℝ) := (r1 + r2 + r3 = 11 ∧ r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧ 
                                         (∀ x : ℝ, x^3 - 11*x^2 + (r1 * r2 + r2 * r3 + r3 * r1) * x - r1 * r2 * r3 = 0)

theorem find_p_q_sum : ∃ (p q : ℝ), roots 2 4 5 → p + q = 78 :=
by
  sorry

end NUMINAMATH_GPT_find_p_q_sum_l1904_190498


namespace NUMINAMATH_GPT_triangle_area_is_zero_l1904_190495

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D := {
  x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z
}

def scalar_vector_mult (k : ℝ) (v : Point3D) : Point3D := {
  x := k * v.x,
  y := k * v.y,
  z := k * v.z
}

theorem triangle_area_is_zero : 
  let u := Point3D.mk 2 1 (-1)
  let v := Point3D.mk 5 4 1
  let w := Point3D.mk 11 10 5
  vector_sub w u = scalar_vector_mult 3 (vector_sub v u) →
-- If the points u, v, w are collinear, the area of the triangle formed by these points is zero:
  ∃ area : ℝ, area = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_area_is_zero_l1904_190495


namespace NUMINAMATH_GPT_original_average_age_l1904_190486

variable (A : ℕ)
variable (N : ℕ := 2)
variable (new_avg_age : ℕ := 32)
variable (age_decrease : ℕ := 4)

theorem original_average_age :
  (A * N + new_avg_age * 2) / (N + 2) = A - age_decrease → A = 40 := 
by
  sorry

end NUMINAMATH_GPT_original_average_age_l1904_190486


namespace NUMINAMATH_GPT_frank_bakes_for_5_days_l1904_190446

variable (d : ℕ) -- The number of days Frank bakes cookies

def cookies_baked_per_day : ℕ := 2 * 12
def cookies_eaten_per_day : ℕ := 1

-- Total cookies baked over d days minus the cookies Frank eats each day
def cookies_remaining_before_ted (d : ℕ) : ℕ :=
  d * (cookies_baked_per_day - cookies_eaten_per_day)

-- Ted eats 4 cookies on the last day, so we add that back to get total before Ted ate
def total_cookies_before_ted (d : ℕ) : ℕ :=
  cookies_remaining_before_ted d + 4

-- After Ted's visit, there are 134 cookies left
axiom ted_leaves_134_cookies : total_cookies_before_ted d = 138

-- Prove that Frank bakes cookies for 5 days
theorem frank_bakes_for_5_days : d = 5 := by
  sorry

end NUMINAMATH_GPT_frank_bakes_for_5_days_l1904_190446


namespace NUMINAMATH_GPT_altitude_triangle_eq_2w_l1904_190413

theorem altitude_triangle_eq_2w (l w h : ℕ) (h₀ : w ≠ 0) (h₁ : l ≠ 0)
    (h_area_rect : l * w = (1 / 2) * l * h) : h = 2 * w :=
by
  -- Consider h₀ (w is not zero) and h₁ (l is not zero)
  -- We need to prove h = 2w given l * w = (1 / 2) * l * h
  sorry

end NUMINAMATH_GPT_altitude_triangle_eq_2w_l1904_190413


namespace NUMINAMATH_GPT_initial_investment_amount_l1904_190434

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment_amount (P A r t : ℝ) (n : ℕ) (hA : A = 992.25) 
  (hr : r = 0.10) (hn : n = 2) (ht : t = 1) : P = 900 :=
by
  have h : compoundInterest P r n t = A := by sorry
  rw [hA, hr, hn, ht] at h
  simp at h
  exact sorry

end NUMINAMATH_GPT_initial_investment_amount_l1904_190434


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1904_190410

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 + a 13 = 10) 
  (h2 : ∀ n m : ℕ, a (n + 1) = a n + d) : a 3 + a 5 + a 7 + a 9 + a 11 = 25 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1904_190410


namespace NUMINAMATH_GPT_radical_product_l1904_190492

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end NUMINAMATH_GPT_radical_product_l1904_190492


namespace NUMINAMATH_GPT_pentagon_AEDCB_area_l1904_190457

-- Definitions based on the given conditions
def rectangle_ABCD (AB BC : ℕ) : Prop :=
AB = 12 ∧ BC = 10

def triangle_ADE (AE ED : ℕ) : Prop :=
AE = 9 ∧ ED = 6 ∧ AE * ED ≠ 0 ∧ (AE^2 + ED^2 = (AE^2 + ED^2))

def area_of_rectangle (AB BC : ℕ) : ℕ :=
AB * BC

def area_of_triangle (AE ED : ℕ) : ℕ :=
(AE * ED) / 2

-- The theorem to be proved
theorem pentagon_AEDCB_area (AB BC AE ED : ℕ) (h_rect : rectangle_ABCD AB BC) (h_tri : triangle_ADE AE ED) :
  area_of_rectangle AB BC - area_of_triangle AE ED = 93 :=
sorry

end NUMINAMATH_GPT_pentagon_AEDCB_area_l1904_190457


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1904_190497

theorem solution_set_of_inequality :
  {x : ℝ | (x^2 - 2*x - 3) * (x^2 + 1) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1904_190497


namespace NUMINAMATH_GPT_max_xy_l1904_190444

variable {x y : ℝ}

theorem max_xy (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 * x + 8 * y = 48) : x * y ≤ 24 :=
sorry

end NUMINAMATH_GPT_max_xy_l1904_190444


namespace NUMINAMATH_GPT_find_uv_l1904_190423

def mat_eqn (u v : ℝ) : Prop :=
  (3 + 8 * u = -3 * v) ∧ (-1 - 6 * u = 1 + 4 * v)

theorem find_uv : ∃ (u v : ℝ), mat_eqn u v ∧ u = -6/7 ∧ v = 5/7 := 
by
  sorry

end NUMINAMATH_GPT_find_uv_l1904_190423


namespace NUMINAMATH_GPT_unique_f_l1904_190402

def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 10^10 }

noncomputable def f : ℕ → ℕ := sorry

axiom f_cond (x : ℕ) (hx : x ∈ S) :
  f (x + 1) % (10^10) = (f (f x) + 1) % (10^10)

axiom f_boundary :
  f (10^10 + 1) % (10^10) = f 1

theorem unique_f (x : ℕ) (hx : x ∈ S) :
  f x % (10^10) = x % (10^10) :=
sorry

end NUMINAMATH_GPT_unique_f_l1904_190402


namespace NUMINAMATH_GPT_gcd_divisibility_and_scaling_l1904_190468

theorem gcd_divisibility_and_scaling (a b n : ℕ) (c : ℕ) (h₁ : a ≠ 0) (h₂ : c > 0) (d : ℕ := Nat.gcd a b) :
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧ Nat.gcd (a * c) (b * c) = c * d :=
by 
  sorry

end NUMINAMATH_GPT_gcd_divisibility_and_scaling_l1904_190468


namespace NUMINAMATH_GPT_calculate_expression_l1904_190461

noncomputable def expr : ℚ := (5 - 2 * (3 - 6 : ℚ)⁻¹ ^ 2)⁻¹

theorem calculate_expression :
  expr = (9 / 43 : ℚ) := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1904_190461


namespace NUMINAMATH_GPT_bees_count_l1904_190452

theorem bees_count (x : ℕ) (h1 : (1/5 : ℚ) * x + (1/3 : ℚ) * x + 
    3 * ((1/3 : ℚ) * x - (1/5 : ℚ) * x) + 1 = x) : x = 15 := 
sorry

end NUMINAMATH_GPT_bees_count_l1904_190452


namespace NUMINAMATH_GPT_factorization_problem_l1904_190447

theorem factorization_problem (a b c x : ℝ) :
  ¬(2 * a^2 - b^2 = (a + b) * (a - b) + a^2) ∧
  ¬(2 * a * (b + c) = 2 * a * b + 2 * a * c) ∧
  (x^3 - 2 * x^2 + x = x * (x - 1)^2) ∧
  ¬ (x^2 + x = x^2 * (1 + 1 / x)) :=
by
  sorry

end NUMINAMATH_GPT_factorization_problem_l1904_190447


namespace NUMINAMATH_GPT_fran_speed_calculation_l1904_190403

theorem fran_speed_calculation:
  let Joann_speed := 15
  let Joann_time := 5
  let Fran_time := 4
  let Fran_speed := (Joann_speed * Joann_time) / Fran_time
  Fran_speed = 18.75 := by
  sorry

end NUMINAMATH_GPT_fran_speed_calculation_l1904_190403


namespace NUMINAMATH_GPT_total_weight_of_compound_l1904_190463

variable (molecular_weight : ℕ) (moles : ℕ)

theorem total_weight_of_compound (h1 : molecular_weight = 72) (h2 : moles = 4) :
  moles * molecular_weight = 288 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_compound_l1904_190463


namespace NUMINAMATH_GPT_determine_x_value_l1904_190428

variable {a b x r : ℝ}
variable (b_nonzero : b ≠ 0)

theorem determine_x_value (h1 : r = (3 * a)^(3 * b)) (h2 : r = a^b * x^b) : x = 27 * a^2 :=
by
  sorry

end NUMINAMATH_GPT_determine_x_value_l1904_190428


namespace NUMINAMATH_GPT_negation_of_proposition_l1904_190477

theorem negation_of_proposition (p : ∀ (x : ℝ), x^2 + 1 > 0) :
  ∃ (x : ℝ), x^2 + 1 ≤ 0 ↔ ¬ (∀ (x : ℝ), x^2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1904_190477


namespace NUMINAMATH_GPT_trip_total_charge_l1904_190427

noncomputable def initial_fee : ℝ := 2.25
noncomputable def additional_charge_per_increment : ℝ := 0.25
noncomputable def increment_length : ℝ := 2 / 5
noncomputable def trip_length : ℝ := 3.6

theorem trip_total_charge :
  initial_fee + (trip_length / increment_length) * additional_charge_per_increment = 4.50 :=
by
  sorry

end NUMINAMATH_GPT_trip_total_charge_l1904_190427


namespace NUMINAMATH_GPT_bob_spending_over_limit_l1904_190488

theorem bob_spending_over_limit : 
  ∀ (necklace_price book_price limit total_cost amount_over_limit : ℕ),
  necklace_price = 34 →
  book_price = necklace_price + 5 →
  limit = 70 →
  total_cost = necklace_price + book_price →
  amount_over_limit = total_cost - limit →
  amount_over_limit = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bob_spending_over_limit_l1904_190488


namespace NUMINAMATH_GPT_curler_ratio_l1904_190406

theorem curler_ratio
  (total_curlers : ℕ)
  (pink_curlers : ℕ)
  (blue_curlers : ℕ)
  (green_curlers : ℕ)
  (h1 : total_curlers = 16)
  (h2 : blue_curlers = 2 * pink_curlers)
  (h3 : green_curlers = 4) :
  pink_curlers / total_curlers = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_curler_ratio_l1904_190406


namespace NUMINAMATH_GPT_factorization_correct_l1904_190433

theorem factorization_correct : ∀ (x : ℕ), x^2 - x = x * (x - 1) :=
by
  intro x
  -- We know the problem reduces to algebraic identity proof
  sorry

end NUMINAMATH_GPT_factorization_correct_l1904_190433


namespace NUMINAMATH_GPT_surface_area_increase_l1904_190442

noncomputable def percent_increase_surface_area (s p : ℝ) : ℝ :=
  let new_edge_length := s * (1 + p / 100)
  let new_surface_area := 6 * (new_edge_length)^2
  let original_surface_area := 6 * s^2
  let percent_increase := (new_surface_area / original_surface_area - 1) * 100
  percent_increase

theorem surface_area_increase (s p : ℝ) :
  percent_increase_surface_area s p = 2 * p + p^2 / 100 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_increase_l1904_190442


namespace NUMINAMATH_GPT_helicopter_rental_cost_l1904_190412

theorem helicopter_rental_cost :
  let hours_per_day := 2
  let days := 3
  let rate_first_day := 85
  let rate_second_day := 75
  let rate_third_day := 65
  let total_cost_before_discount := hours_per_day * rate_first_day + hours_per_day * rate_second_day + hours_per_day * rate_third_day
  let discount := 0.05
  let discounted_amount := total_cost_before_discount * discount
  let total_cost_after_discount := total_cost_before_discount - discounted_amount
  total_cost_after_discount = 427.50 :=
by
  sorry

end NUMINAMATH_GPT_helicopter_rental_cost_l1904_190412


namespace NUMINAMATH_GPT_two_A_minus_B_l1904_190431

theorem two_A_minus_B (A B : ℝ) 
  (h1 : Real.tan (A - B - Real.pi) = 1 / 2) 
  (h2 : Real.tan (3 * Real.pi - B) = 1 / 7) : 
  2 * A - B = -3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_two_A_minus_B_l1904_190431


namespace NUMINAMATH_GPT_total_legs_correct_l1904_190480

def num_ants : ℕ := 12
def num_spiders : ℕ := 8
def legs_per_ant : ℕ := 6
def legs_per_spider : ℕ := 8
def total_legs := num_ants * legs_per_ant + num_spiders * legs_per_spider

theorem total_legs_correct : total_legs = 136 :=
by
  sorry

end NUMINAMATH_GPT_total_legs_correct_l1904_190480


namespace NUMINAMATH_GPT_remainder_product_modulo_17_l1904_190421

theorem remainder_product_modulo_17 :
  (1234 % 17) = 5 ∧ (1235 % 17) = 6 ∧ (1236 % 17) = 7 ∧ (1237 % 17) = 8 ∧ (1238 % 17) = 9 →
  ((1234 * 1235 * 1236 * 1237 * 1238) % 17) = 9 :=
by
  sorry

end NUMINAMATH_GPT_remainder_product_modulo_17_l1904_190421


namespace NUMINAMATH_GPT_find_ab_l1904_190401

-- Define the conditions and the goal
theorem find_ab (a b : ℝ) (h1 : a^2 + b^2 = 26) (h2 : a + b = 7) : ab = 23 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_ab_l1904_190401


namespace NUMINAMATH_GPT_minimum_toothpicks_removal_l1904_190420

theorem minimum_toothpicks_removal
  (total_toothpicks : ℕ)
  (grid_size : ℕ)
  (toothpicks_per_square : ℕ)
  (shared_sides : ℕ)
  (interior_toothpicks : ℕ) 
  (diagonal_toothpicks : ℕ)
  (min_removal : ℕ) 
  (no_squares_or_triangles : Bool)
  (h1 : total_toothpicks = 40)
  (h2 : grid_size = 3)
  (h3 : toothpicks_per_square = 4)
  (h4 : shared_sides = 16)
  (h5 : interior_toothpicks = 16) 
  (h6 : diagonal_toothpicks = 12)
  (h7 : min_removal = 16)
: no_squares_or_triangles := 
sorry

end NUMINAMATH_GPT_minimum_toothpicks_removal_l1904_190420


namespace NUMINAMATH_GPT_find_b_l1904_190473

def h (x : ℝ) : ℝ := 5 * x + 7

theorem find_b (b : ℝ) : h b = 0 ↔ b = -7 / 5 := by
  sorry

end NUMINAMATH_GPT_find_b_l1904_190473


namespace NUMINAMATH_GPT_gcf_180_240_300_l1904_190411

theorem gcf_180_240_300 : Nat.gcd (Nat.gcd 180 240) 300 = 60 := sorry

end NUMINAMATH_GPT_gcf_180_240_300_l1904_190411


namespace NUMINAMATH_GPT_jennifer_initial_oranges_l1904_190467

theorem jennifer_initial_oranges (O : ℕ) : 
  ∀ (pears apples remaining_fruits : ℕ),
    pears = 10 →
    apples = 2 * pears →
    remaining_fruits = pears - 2 + apples - 2 + O - 2 →
    remaining_fruits = 44 →
    O = 20 :=
by
  intros pears apples remaining_fruits h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jennifer_initial_oranges_l1904_190467


namespace NUMINAMATH_GPT_range_of_x_l1904_190465

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) : 
  12 ≤ x := 
sorry

end NUMINAMATH_GPT_range_of_x_l1904_190465


namespace NUMINAMATH_GPT_measure_85_liters_l1904_190489

theorem measure_85_liters (C1 C2 C3 : ℕ) (capacity : ℕ) : 
  (C1 = 0 ∧ C2 = 0 ∧ C3 = 1 ∧ capacity = 85) → 
  (∃ weighings : ℕ, weighings ≤ 8 ∧ C1 = 85 ∨ C2 = 85 ∨ C3 = 85) :=
by 
  sorry

end NUMINAMATH_GPT_measure_85_liters_l1904_190489


namespace NUMINAMATH_GPT_ball_rebound_percentage_l1904_190422

theorem ball_rebound_percentage (P : ℝ) 
  (h₁ : 100 + 2 * 100 * P + 2 * 100 * P^2 = 250) : P = 0.5 := 
by 
  sorry

end NUMINAMATH_GPT_ball_rebound_percentage_l1904_190422


namespace NUMINAMATH_GPT_remainder_product_mod_5_l1904_190469

theorem remainder_product_mod_5 : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_product_mod_5_l1904_190469


namespace NUMINAMATH_GPT_find_a_l1904_190460

def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_a : ∃ a : ℝ, f a + g a = 0 ∧ a = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1904_190460


namespace NUMINAMATH_GPT_rate_of_change_l1904_190448

noncomputable def radius : ℝ := 12
noncomputable def θ (t : ℝ) : ℝ := (38 + 5 * t) * (Real.pi / 180)
noncomputable def area (t : ℝ) : ℝ := (1/2) * radius^2 * θ t

theorem rate_of_change (t : ℝ) : deriv area t = 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_rate_of_change_l1904_190448


namespace NUMINAMATH_GPT_collinear_points_sum_l1904_190456

-- Points in 3-dimensional space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition of collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ k : ℝ,
    k ≠ 0 ∧
    (p2.x - p1.x) * k = (p3.x - p1.x) ∧
    (p2.y - p1.y) * k = (p3.y - p1.y) ∧
    (p2.z - p1.z) * k = (p3.z - p1.z)

-- Main statement
theorem collinear_points_sum {a b : ℝ} :
  collinear (Point3D.mk 2 a b) (Point3D.mk a 3 b) (Point3D.mk a b 4) → a + b = 6 :=
by
  sorry

end NUMINAMATH_GPT_collinear_points_sum_l1904_190456


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_equation3_solution_l1904_190430

theorem equation1_solution :
  ∀ x : ℝ, x^2 + 4 * x = 0 ↔ x = 0 ∨ x = -4 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, 2 * (x - 1) + x * (x - 1) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 4 = 0 ↔ x = (1 + Real.sqrt 13) / 3 ∨ x = (1 - Real.sqrt 13) / 3 :=
by
  sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_equation3_solution_l1904_190430


namespace NUMINAMATH_GPT_bouncy_ball_pack_count_l1904_190408

theorem bouncy_ball_pack_count
  (x : ℤ)  -- Let x be the number of bouncy balls in each pack
  (r : ℤ := 7 * x)  -- Total number of red bouncy balls
  (y : ℤ := 6 * x)  -- Total number of yellow bouncy balls
  (h : r = y + 18)  -- Condition: 7x = 6x + 18
  : x = 18 := sorry

end NUMINAMATH_GPT_bouncy_ball_pack_count_l1904_190408


namespace NUMINAMATH_GPT_pi_over_2_irrational_l1904_190472

def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ is_rational x

theorem pi_over_2_irrational : is_irrational (Real.pi / 2) :=
by sorry

end NUMINAMATH_GPT_pi_over_2_irrational_l1904_190472


namespace NUMINAMATH_GPT_dot_product_in_triangle_l1904_190487

noncomputable def ab := 3
noncomputable def ac := 2
noncomputable def bc := Real.sqrt 10

theorem dot_product_in_triangle : 
  let AB := ab
  let AC := ac
  let BC := bc
  (AB = 3) → (AC = 2) → (BC = Real.sqrt 10) → 
  ∃ cosA, (cosA = (AB^2 + AC^2 - BC^2) / (2 * AB * AC)) →
  ∃ dot_product, (dot_product = AB * AC * cosA) ∧ dot_product = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_in_triangle_l1904_190487


namespace NUMINAMATH_GPT_speed_of_sound_l1904_190494

theorem speed_of_sound (time_heard : ℕ) (time_occured : ℕ) (distance : ℝ) : 
  time_heard = 30 * 60 + 20 → 
  time_occured = 30 * 60 → 
  distance = 6600 → 
  (distance / ((time_heard - time_occured) / 3600)) / 3600 = 330 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_speed_of_sound_l1904_190494


namespace NUMINAMATH_GPT_percentage_reduction_in_oil_price_l1904_190454

theorem percentage_reduction_in_oil_price (R : ℝ) (P : ℝ) (hR : R = 48) (h_quantity : (800/R) - (800/P) = 5) : 
    ((P - R) / P) * 100 = 30 := 
    sorry

end NUMINAMATH_GPT_percentage_reduction_in_oil_price_l1904_190454


namespace NUMINAMATH_GPT_Jack_goal_l1904_190440

-- Define the amounts Jack made from brownies and lemon squares
def brownies (n : ℕ) (price : ℕ) : ℕ := n * price
def lemonSquares (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the amount Jack needs to make from cookies
def cookies (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the total goal for Jack
def totalGoal (browniesCount : ℕ) (browniesPrice : ℕ) 
              (lemonSquaresCount : ℕ) (lemonSquaresPrice : ℕ) 
              (cookiesCount : ℕ) (cookiesPrice: ℕ) : ℕ :=
  brownies browniesCount browniesPrice + lemonSquares lemonSquaresCount lemonSquaresPrice + cookies cookiesCount cookiesPrice

theorem Jack_goal : totalGoal 4 3 5 2 7 4 = 50 :=
by
  -- Adding up the different components of the total earnings
  let totalFromBrownies := brownies 4 3
  let totalFromLemonSquares := lemonSquares 5 2
  let totalFromCookies := cookies 7 4
  -- Summing up the amounts
  have step1 : totalFromBrownies = 12 := rfl
  have step2 : totalFromLemonSquares = 10 := rfl
  have step3 : totalFromCookies = 28 := rfl
  have step4 : totalGoal 4 3 5 2 7 4 = totalFromBrownies + totalFromLemonSquares + totalFromCookies := rfl
  have step5 : totalFromBrownies + totalFromLemonSquares + totalFromCookies = 12 + 10 + 28 := by rw [step1, step2, step3]
  have step6 : 12 + 10 + 28 = 50 := by norm_num
  exact step4 ▸ (step5 ▸ step6)

end NUMINAMATH_GPT_Jack_goal_l1904_190440


namespace NUMINAMATH_GPT_quadratic_m_ge_neg2_l1904_190416

-- Define the quadratic equation and condition for real roots
def quadratic_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, (x + 2) ^ 2 = m + 2

-- The theorem to prove
theorem quadratic_m_ge_neg2 (m : ℝ) (h : quadratic_has_real_roots m) : m ≥ -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_m_ge_neg2_l1904_190416


namespace NUMINAMATH_GPT_connie_remaining_marbles_l1904_190443

def initial_marbles : ℕ := 73
def marbles_given : ℕ := 70

theorem connie_remaining_marbles : initial_marbles - marbles_given = 3 := by
  sorry

end NUMINAMATH_GPT_connie_remaining_marbles_l1904_190443


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1904_190405

def M : Set ℤ := {1, 2}
def N (a : ℤ) : Set ℤ := {a^2}

theorem sufficient_not_necessary (a : ℤ) :
  (a = 1 → N a ⊆ M) ∧ (N a ⊆ M → a = 1) = false :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1904_190405


namespace NUMINAMATH_GPT_max_min_value_of_product_l1904_190432

theorem max_min_value_of_product (x y : ℝ) (h : x ^ 2 + y ^ 2 = 1) :
  (1 + x * y) * (1 - x * y) ≤ 1 ∧ (1 + x * y) * (1 - x * y) ≥ 3 / 4 :=
by sorry

end NUMINAMATH_GPT_max_min_value_of_product_l1904_190432


namespace NUMINAMATH_GPT_sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l1904_190418

-- Definitions based on conditions
def sum_arithmetic (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Theorem statements based on the correct answers
theorem sum_first_twelve_multiples_17 : 
  17 * sum_arithmetic 12 = 1326 := 
by
  sorry

theorem sum_squares_first_twelve_multiples_17 : 
  17^2 * sum_squares 12 = 187850 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l1904_190418


namespace NUMINAMATH_GPT_local_language_letters_l1904_190475

theorem local_language_letters (n : ℕ) (h : 1 + 2 * n = 139) : n = 69 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_local_language_letters_l1904_190475


namespace NUMINAMATH_GPT_min_value_expression_l1904_190436

theorem min_value_expression (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  ∃ x : ℝ, x = 1 ∧ x = (3 * a - 2 * b + c) / (b - a) := 
  sorry

end NUMINAMATH_GPT_min_value_expression_l1904_190436


namespace NUMINAMATH_GPT_inequality_holds_l1904_190417

theorem inequality_holds (a b : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (hb : 0 ≤ b) (hb' : b ≤ 1) : 
  a^5 + b^3 + (a - b)^2 ≤ 2 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1904_190417


namespace NUMINAMATH_GPT_div_poly_l1904_190437

theorem div_poly (m n p : ℕ) : 
  (X^2 + X + 1) ∣ (X^(3*m) + X^(3*n + 1) + X^(3*p + 2)) := 
sorry

end NUMINAMATH_GPT_div_poly_l1904_190437


namespace NUMINAMATH_GPT_max_value_of_f_product_of_zeros_l1904_190474

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b
 
theorem max_value_of_f (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) : f (1 / a) a b = -Real.log a - 1 + b :=
by
  sorry

theorem product_of_zeros (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) (hx_ne : x1 ≠ x2) : x1 * x2 < 1 / (a * a) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_product_of_zeros_l1904_190474


namespace NUMINAMATH_GPT_weight_of_mixture_is_correct_l1904_190407

noncomputable def weight_mixture_kg (weight_per_liter_a weight_per_liter_b ratio_a ratio_b total_volume_liters : ℕ) : ℝ :=
  let volume_a := (ratio_a * total_volume_liters) / (ratio_a + ratio_b)
  let volume_b := (ratio_b * total_volume_liters) / (ratio_a + ratio_b)
  let weight_a := (volume_a * weight_per_liter_a) 
  let weight_b := (volume_b * weight_per_liter_b) 
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_mixture_kg 900 700 3 2 4 = 3.280 := 
sorry

end NUMINAMATH_GPT_weight_of_mixture_is_correct_l1904_190407


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1904_190491

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 3 ∨ a = 7)) (h2 : (b = 3 ∨ b = 7)) (h3 : a ≠ b) : 
  ∃ (c : ℕ), (a = 7 ∧ b = 3 ∧ c = 17) ∨ (a = 3 ∧ b = 7 ∧ c = 17) := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1904_190491


namespace NUMINAMATH_GPT_radius_of_circle_l1904_190438

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1904_190438


namespace NUMINAMATH_GPT_uniquePlantsTotal_l1904_190459

-- Define the number of plants in each bed
def numPlantsInA : ℕ := 600
def numPlantsInB : ℕ := 500
def numPlantsInC : ℕ := 400

-- Define the number of shared plants between beds
def sharedPlantsAB : ℕ := 60
def sharedPlantsAC : ℕ := 120
def sharedPlantsBC : ℕ := 80
def sharedPlantsABC : ℕ := 30

-- Prove that the total number of unique plants in the garden is 1270
theorem uniquePlantsTotal : 
  numPlantsInA + numPlantsInB + numPlantsInC 
  - sharedPlantsAB - sharedPlantsAC - sharedPlantsBC 
  + sharedPlantsABC = 1270 := 
by sorry

end NUMINAMATH_GPT_uniquePlantsTotal_l1904_190459


namespace NUMINAMATH_GPT_find_common_difference_l1904_190499

theorem find_common_difference 
  (a : ℕ → ℝ)
  (a1 : a 1 = 5)
  (a25 : a 25 = 173)
  (h : ∀ n : ℕ, a (n+1) = a 1 + n * (a 2 - a 1)) : 
  a 2 - a 1 = 7 :=
by 
  sorry

end NUMINAMATH_GPT_find_common_difference_l1904_190499


namespace NUMINAMATH_GPT_given_condition_implies_result_l1904_190439

theorem given_condition_implies_result (a : ℝ) (h : a ^ 2 + 2 * a = 1) : 2 * a ^ 2 + 4 * a + 1 = 3 :=
sorry

end NUMINAMATH_GPT_given_condition_implies_result_l1904_190439


namespace NUMINAMATH_GPT_option_C_correct_l1904_190455

theorem option_C_correct (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_option_C_correct_l1904_190455


namespace NUMINAMATH_GPT_triangle_identity_proof_l1904_190426

variables (r r_a r_b r_c R S p : ℝ)
-- assume necessary properties for valid triangle (not explicitly given in problem but implied)
-- nonnegativity, relations between inradius, exradii and circumradius, etc.

theorem triangle_identity_proof
  (h_r_pos : 0 < r)
  (h_ra_pos : 0 < r_a)
  (h_rb_pos : 0 < r_b)
  (h_rc_pos : 0 < r_c)
  (h_R_pos : 0 < R)
  (h_S_pos : 0 < S)
  (h_p_pos : 0 < p)
  (h_area : S = r * p) :
  (1 / r^3) - (1 / r_a^3) - (1 / r_b^3) - (1 / r_c^3) = (12 * R) / (S^2) :=
sorry

end NUMINAMATH_GPT_triangle_identity_proof_l1904_190426


namespace NUMINAMATH_GPT_each_vaccine_costs_45_l1904_190485

theorem each_vaccine_costs_45
    (num_vaccines : ℕ)
    (doctor_visit_cost : ℝ)
    (insurance_coverage : ℝ)
    (trip_cost : ℝ)
    (total_payment : ℝ) :
    num_vaccines = 10 ->
    doctor_visit_cost = 250 ->
    insurance_coverage = 0.80 ->
    trip_cost = 1200 ->
    total_payment = 1340 ->
    (∃ (vaccine_cost : ℝ), vaccine_cost = 45) :=
by {
    sorry
}

end NUMINAMATH_GPT_each_vaccine_costs_45_l1904_190485


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1904_190478

theorem quadratic_inequality_solution (m: ℝ) (h: m > 1) :
  { x : ℝ | x^2 + (m - 1) * x - m ≥ 0 } = { x | x ≤ -m ∨ x ≥ 1 } :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1904_190478


namespace NUMINAMATH_GPT_present_age_of_son_l1904_190481

theorem present_age_of_son :
  (∃ (S F : ℕ), F = S + 22 ∧ (F + 2) = 2 * (S + 2)) → ∃ (S : ℕ), S = 20 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l1904_190481


namespace NUMINAMATH_GPT_regular_polygon_sides_l1904_190449

theorem regular_polygon_sides (n : ℕ) (h : ∀ (θ : ℝ), θ = 36 → θ = 360 / n) : n = 10 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1904_190449


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1904_190476

theorem solve_quadratic_equation (x : ℝ) : x^2 - 4*x + 3 = 0 ↔ (x = 1 ∨ x = 3) := 
by 
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1904_190476


namespace NUMINAMATH_GPT_parabola_standard_equation_l1904_190450

theorem parabola_standard_equation (x y : ℝ) :
  (3 * x - 4 * y - 12 = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
sorry

end NUMINAMATH_GPT_parabola_standard_equation_l1904_190450


namespace NUMINAMATH_GPT_total_enemies_l1904_190458

theorem total_enemies (n : ℕ) : (n - 3) * 9 = 72 → n = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_enemies_l1904_190458


namespace NUMINAMATH_GPT_distinct_real_numbers_proof_l1904_190425

variables {a b c : ℝ}

theorem distinct_real_numbers_proof (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
  (h₄ : (a / (b - c) + b / (c - a) + c / (a - b)) = -1) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) = 0 :=
sorry

end NUMINAMATH_GPT_distinct_real_numbers_proof_l1904_190425


namespace NUMINAMATH_GPT_find_base_a_l1904_190483

theorem find_base_a 
  (a : ℕ)
  (C_a : ℕ := 12) :
  (3 * a^2 + 4 * a + 7) + (5 * a^2 + 7 * a + 9) = 9 * a^2 + 2 * a + C_a →
  a = 14 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_base_a_l1904_190483


namespace NUMINAMATH_GPT_ned_mowed_in_summer_l1904_190451

def mowed_in_summer (total_mows spring_mows summer_mows : ℕ) : Prop :=
  total_mows = spring_mows + summer_mows

theorem ned_mowed_in_summer :
  ∀ (total_mows spring_mows summer_mows : ℕ),
  total_mows = 11 →
  spring_mows = 6 →
  mowed_in_summer total_mows spring_mows summer_mows →
  summer_mows = 5 :=
by
  intros total_mows spring_mows summer_mows h_total h_spring h_mowed
  sorry

end NUMINAMATH_GPT_ned_mowed_in_summer_l1904_190451


namespace NUMINAMATH_GPT_number_of_girls_and_boys_l1904_190453

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) := g = 4 * (g + b) / 7 ∧ b = 3 * (g + b) / 7
def total_students (g b : ℕ) := g + b = 56

-- The main proof statement
theorem number_of_girls_and_boys (g b : ℕ) 
  (h_ratio : ratio_girls_to_boys g b)
  (h_total : total_students g b) : 
  g = 32 ∧ b = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_girls_and_boys_l1904_190453


namespace NUMINAMATH_GPT_conference_attendees_l1904_190490

theorem conference_attendees (w m : ℕ) (h1 : w + m = 47) (h2 : 16 + (w - 1) = m) : w = 16 ∧ m = 31 :=
by
  sorry

end NUMINAMATH_GPT_conference_attendees_l1904_190490


namespace NUMINAMATH_GPT_inappropriate_character_choice_l1904_190496

-- Definitions and conditions
def is_main_character (c : String) : Prop := 
  c = "Gryphon" ∨ c = "Mock Turtle"

def characters : List String := ["Lobster", "Gryphon", "Mock Turtle"]

-- Theorem statement
theorem inappropriate_character_choice : 
  ¬ is_main_character "Lobster" :=
by 
  sorry

end NUMINAMATH_GPT_inappropriate_character_choice_l1904_190496


namespace NUMINAMATH_GPT_cubic_equation_roots_l1904_190441

theorem cubic_equation_roots (a b c d r s t : ℝ) (h_eq : a ≠ 0) 
(ht1 : a * r^3 + b * r^2 + c * r + d = 0)
(ht2 : a * s^3 + b * s^2 + c * s + d = 0)
(ht3 : a * t^3 + b * t^2 + c * t + d = 0)
(h1 : r * s = 3) 
(h2 : r * t = 3) 
(h3 : s * t = 3) : 
c = 3 * a := 
sorry

end NUMINAMATH_GPT_cubic_equation_roots_l1904_190441


namespace NUMINAMATH_GPT_count_valid_tuples_l1904_190464

variable {b_0 b_1 b_2 b_3 : ℕ}

theorem count_valid_tuples : 
  (∃ b_0 b_1 b_2 b_3 : ℕ, 
    0 ≤ b_0 ∧ b_0 ≤ 99 ∧ 
    0 ≤ b_1 ∧ b_1 ≤ 99 ∧ 
    0 ≤ b_2 ∧ b_2 ≤ 99 ∧ 
    0 ≤ b_3 ∧ b_3 ≤ 99 ∧ 
    5040 = b_3 * 10^3 + b_2 * 10^2 + b_1 * 10 + b_0) ∧ 
    ∃ (M : ℕ), 
    M = 504 :=
sorry

end NUMINAMATH_GPT_count_valid_tuples_l1904_190464


namespace NUMINAMATH_GPT_license_plates_count_l1904_190466

theorem license_plates_count :
  (20 * 6 * 20 * 10 * 26 = 624000) :=
by
  sorry

end NUMINAMATH_GPT_license_plates_count_l1904_190466


namespace NUMINAMATH_GPT_relationship_a_b_l1904_190470

theorem relationship_a_b
  (m a b : ℝ)
  (h1 : ∃ m, ∀ x, -2 * x + m = y)
  (h2 : ∃ x₁ y₁, (x₁ = -2) ∧ (y₁ = a) ∧ (-2 * x₁ + m = y₁))
  (h3 : ∃ x₂ y₂, (x₂ = 2) ∧ (y₂ = b) ∧ (-2 * x₂ + m = y₂)) :
  a > b :=
sorry

end NUMINAMATH_GPT_relationship_a_b_l1904_190470


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_l1904_190484

theorem shorter_leg_of_right_triangle {a b : ℕ} (hypotenuse : ℕ) (h : hypotenuse = 41) (h_right_triangle : a^2 + b^2 = hypotenuse^2) (h_ineq : a < b) : a = 9 :=
by {
  -- proof to be filled in 
  sorry
}

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_l1904_190484


namespace NUMINAMATH_GPT_find_u_value_l1904_190482

theorem find_u_value (u : ℤ) : ∀ (y : ℤ → ℤ), 
  (y 2 = 8) → (y 4 = 14) → (y 6 = 20) → 
  (∀ x, (x % 2 = 0) → (y (x + 2) = y x + 6)) → 
  y 18 = u → u = 56 :=
by
  intros y h2 h4 h6 pattern h18
  sorry

end NUMINAMATH_GPT_find_u_value_l1904_190482


namespace NUMINAMATH_GPT_compare_squares_l1904_190462

theorem compare_squares (a b : ℝ) : a^2 + b^2 ≥ ab + a + b - 1 :=
by
  sorry

end NUMINAMATH_GPT_compare_squares_l1904_190462
