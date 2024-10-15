import Mathlib

namespace NUMINAMATH_GPT_rainfall_ratio_l434_43458

theorem rainfall_ratio (S M T : ℝ) (h1 : M = S + 3) (h2 : S = 4) (h3 : S + M + T = 25) : T / M = 2 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_ratio_l434_43458


namespace NUMINAMATH_GPT_percent_increase_l434_43409

variable (E : ℝ)

-- Given conditions
def enrollment_1992 := 1.20 * E
def enrollment_1993 := 1.26 * E

-- Theorem to prove
theorem percent_increase :
  ((enrollment_1993 E - enrollment_1992 E) / enrollment_1992 E) * 100 = 5 := by
  sorry

end NUMINAMATH_GPT_percent_increase_l434_43409


namespace NUMINAMATH_GPT_P_zero_eq_zero_l434_43457

open Polynomial

noncomputable def P (x : ℝ) : ℝ := sorry

axiom distinct_roots : ∃ y : Fin 17 → ℝ, Function.Injective y ∧ ∀ i, P (y i ^ 2) = 0

theorem P_zero_eq_zero : P 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_P_zero_eq_zero_l434_43457


namespace NUMINAMATH_GPT_find_x_l434_43447

theorem find_x (x y z : ℚ) (h1 : (x * y) / (x + y) = 4) (h2 : (x * z) / (x + z) = 5) (h3 : (y * z) / (y + z) = 6) : x = 40 / 9 :=
by
  -- Structure the proof here
  sorry

end NUMINAMATH_GPT_find_x_l434_43447


namespace NUMINAMATH_GPT_find_interest_rate_l434_43464

-- Define the conditions
def total_amount : ℝ := 2500
def second_part_rate : ℝ := 0.06
def annual_income : ℝ := 145
def first_part_amount : ℝ := 500.0000000000002
noncomputable def interest_rate (r : ℝ) : Prop :=
  first_part_amount * r + (total_amount - first_part_amount) * second_part_rate = annual_income

-- State the theorem
theorem find_interest_rate : interest_rate 0.05 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l434_43464


namespace NUMINAMATH_GPT_area_of_complex_polygon_l434_43415

-- Defining the problem
def area_of_polygon (side1 side2 side3 : ℝ) (rot1 rot2 : ℝ) : ℝ :=
  -- This is a placeholder definition.
  -- In a complete proof, here we would calculate the area based on the input conditions.
  sorry

-- Main theorem statement
theorem area_of_complex_polygon :
  area_of_polygon 4 5 6 (π / 4) (-π / 6) = 72 :=
by sorry

end NUMINAMATH_GPT_area_of_complex_polygon_l434_43415


namespace NUMINAMATH_GPT_men_became_absent_l434_43495

theorem men_became_absent (original_men planned_days actual_days : ℕ) (h1 : original_men = 48) (h2 : planned_days = 15) (h3 : actual_days = 18) :
  ∃ x : ℕ, 48 * 15 = (48 - x) * 18 ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_men_became_absent_l434_43495


namespace NUMINAMATH_GPT_probability_product_even_gt_one_fourth_l434_43446

def n := 100
def is_even (x : ℕ) : Prop := x % 2 = 0
def is_odd (x : ℕ) : Prop := ¬ is_even x

theorem probability_product_even_gt_one_fourth :
  (∃ (p : ℝ), p > 0 ∧ p = 1 - (50 * 49 * 48 : ℝ) / (100 * 99 * 98) ∧ p > 1 / 4) :=
sorry

end NUMINAMATH_GPT_probability_product_even_gt_one_fourth_l434_43446


namespace NUMINAMATH_GPT_wall_length_proof_l434_43454

noncomputable def volume_of_brick (length width height : ℝ) : ℝ := length * width * height

noncomputable def total_volume (brick_volume num_of_bricks : ℝ) : ℝ := brick_volume * num_of_bricks

theorem wall_length_proof
  (height_of_wall : ℝ) (width_of_walls : ℝ) (num_of_bricks : ℝ)
  (length_of_brick width_of_brick height_of_brick : ℝ)
  (total_volume_of_bricks : ℝ) :
  total_volume (volume_of_brick length_of_brick width_of_brick height_of_brick) num_of_bricks = total_volume_of_bricks →
  volume_of_brick length_of_wall height_of_wall width_of_walls = total_volume_of_bricks →
  height_of_wall = 600 →
  width_of_walls = 2 →
  num_of_bricks = 2909.090909090909 →
  length_of_brick = 5 →
  width_of_brick = 11 →
  height_of_brick = 6 →
  total_volume_of_bricks = 960000 →
  length_of_wall = 800 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_wall_length_proof_l434_43454


namespace NUMINAMATH_GPT_ratio_saturday_friday_l434_43411

variable (S : ℕ)
variable (soldOnFriday : ℕ := 30)
variable (soldOnSunday : ℕ := S - 15)
variable (totalSold : ℕ := 135)

theorem ratio_saturday_friday (h1 : soldOnFriday = 30)
                              (h2 : totalSold = 135)
                              (h3 : soldOnSunday = S - 15)
                              (h4 : soldOnFriday + S + soldOnSunday = totalSold) :
  (S / soldOnFriday) = 2 :=
by
  -- Prove the theorem here...
  sorry

end NUMINAMATH_GPT_ratio_saturday_friday_l434_43411


namespace NUMINAMATH_GPT_compounding_frequency_l434_43430

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compounding_frequency (P A r t n : ℝ) 
  (principal : P = 6000) 
  (amount : A = 6615)
  (rate : r = 0.10)
  (time : t = 1) 
  (comp_freq : n = 2) :
  compound_interest P r n t = A := 
by 
  simp [compound_interest, principal, rate, time, comp_freq, amount]
  -- calculations and proof omitted
  sorry

end NUMINAMATH_GPT_compounding_frequency_l434_43430


namespace NUMINAMATH_GPT_mutual_fund_share_increase_l434_43402

theorem mutual_fund_share_increase (P : ℝ) (h1 : (P * 1.20) = 1.20 * P) (h2 : (1.20 * P) * (1 / 3) = 0.40 * P) :
  ((1.60 * P) = (P * 1.60)) :=
by
  sorry

end NUMINAMATH_GPT_mutual_fund_share_increase_l434_43402


namespace NUMINAMATH_GPT_find_constants_l434_43420

-- Given definitions based on the conditions and conjecture
def S (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | 2 => 5
  | 3 => 15
  | 4 => 34
  | 5 => 65
  | _ => 0

noncomputable def conjecture_S (n a b c : ℤ) := (2 * n - 1) * (a * n^2 + b * n + c)

theorem find_constants (a b c : ℤ) (h1 : conjecture_S 1 a b c = 1) (h2 : conjecture_S 2 a b c = 5) (h3 : conjecture_S 3 a b c = 15) : 3 * a + b = 4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_constants_l434_43420


namespace NUMINAMATH_GPT_lena_more_than_nicole_l434_43450

theorem lena_more_than_nicole :
  ∀ (L K N : ℝ),
    L = 37.5 →
    (L + 9.5) = 5 * K →
    K = N - 8.5 →
    (L - N) = 19.6 :=
by
  intros L K N hL hLK hK
  sorry

end NUMINAMATH_GPT_lena_more_than_nicole_l434_43450


namespace NUMINAMATH_GPT_angle_QRS_determination_l434_43403

theorem angle_QRS_determination (PQ_parallel_RS : ∀ (P Q R S T : Type) 
  (angle_PTQ : ℝ) (angle_SRT : ℝ), 
  PQ_parallel_RS → (angle_PTQ = angle_SRT) → (angle_PTQ = 4 * angle_SRT - 120)) 
  (angle_SRT : ℝ) (angle_QRS : ℝ) 
  (h : angle_SRT = 4 * angle_SRT - 120) : angle_QRS = 40 :=
by 
  sorry

end NUMINAMATH_GPT_angle_QRS_determination_l434_43403


namespace NUMINAMATH_GPT_smaller_cube_volume_l434_43433

theorem smaller_cube_volume
  (V_L : ℝ) (N : ℝ) (SA_diff : ℝ) 
  (h1 : V_L = 8)
  (h2 : N = 8)
  (h3 : SA_diff = 24) :
  (∀ V_S : ℝ, V_L = N * V_S → V_S = 1) :=
by
  sorry

end NUMINAMATH_GPT_smaller_cube_volume_l434_43433


namespace NUMINAMATH_GPT_smallest_value_of_c_l434_43410

def bound_a (a b : ℝ) : Prop := 1 + a ≤ b
def bound_inv (a b c : ℝ) : Prop := (1 / a) + (1 / b) ≤ (1 / c)

theorem smallest_value_of_c (a b c : ℝ) (ha : 1 < a) (hb : a < b) 
  (hc : b < c) (h_ab : bound_a a b) (h_inv : bound_inv a b c) : 
  c ≥ (3 + Real.sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_c_l434_43410


namespace NUMINAMATH_GPT_curves_intersect_at_three_points_l434_43417

theorem curves_intersect_at_three_points :
  (∀ x y a : ℝ, (x^2 + y^2 = 4 * a^2) ∧ (y = x^2 - 2 * a) → a = 1) := sorry

end NUMINAMATH_GPT_curves_intersect_at_three_points_l434_43417


namespace NUMINAMATH_GPT_trigonometric_identity_l434_43462

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l434_43462


namespace NUMINAMATH_GPT_find_p_l434_43456

theorem find_p (m n p : ℝ)
  (h1 : m = 4 * n + 5)
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_p_l434_43456


namespace NUMINAMATH_GPT_gardening_project_total_cost_l434_43441

noncomputable def cost_gardening_project : ℕ := 
  let number_rose_bushes := 20
  let cost_per_rose_bush := 150
  let cost_fertilizer_per_bush := 25
  let gardener_work_hours := [6, 5, 4, 7]
  let gardener_hourly_rate := 30
  let soil_amount := 100
  let cost_per_cubic_foot := 5

  let cost_roses := number_rose_bushes * cost_per_rose_bush
  let cost_fertilizer := number_rose_bushes * cost_fertilizer_per_bush
  let total_work_hours := List.sum gardener_work_hours
  let cost_labor := total_work_hours * gardener_hourly_rate
  let cost_soil := soil_amount * cost_per_cubic_foot

  cost_roses + cost_fertilizer + cost_labor + cost_soil

theorem gardening_project_total_cost : cost_gardening_project = 4660 := by
  sorry

end NUMINAMATH_GPT_gardening_project_total_cost_l434_43441


namespace NUMINAMATH_GPT_range_of_c_l434_43424

-- Definitions of the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∃ x : ℝ, x^2 - c^2 ≤ - (1 / 16)

-- Main theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : p c) (h3 : q c) : c ≥ 1 / 4 ∧ c < 1 :=
  sorry

end NUMINAMATH_GPT_range_of_c_l434_43424


namespace NUMINAMATH_GPT_max_students_distributing_pens_and_pencils_l434_43472

theorem max_students_distributing_pens_and_pencils :
  Nat.gcd 1001 910 = 91 :=
by
  -- remaining proof required
  sorry

end NUMINAMATH_GPT_max_students_distributing_pens_and_pencils_l434_43472


namespace NUMINAMATH_GPT_farm_distance_is_6_l434_43482

noncomputable def distance_to_farm (initial_gallons : ℕ) 
  (consumption_rate : ℕ) (supermarket_distance : ℕ) 
  (outbound_distance : ℕ) (remaining_gallons : ℕ) : ℕ :=
initial_gallons * consumption_rate - 
  (2 * supermarket_distance + 2 * outbound_distance - remaining_gallons * consumption_rate)

theorem farm_distance_is_6 : 
  distance_to_farm 12 2 5 2 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_farm_distance_is_6_l434_43482


namespace NUMINAMATH_GPT_product_mod_25_l434_43419

def remainder_when_divided_by_25 (n : ℕ) : ℕ := n % 25

theorem product_mod_25 (a b c d : ℕ) 
  (h1 : a = 1523) (h2 : b = 1857) (h3 : c = 1919) (h4 : d = 2012) :
  remainder_when_divided_by_25 (a * b * c * d) = 8 :=
by
  sorry

end NUMINAMATH_GPT_product_mod_25_l434_43419


namespace NUMINAMATH_GPT_mary_investment_amount_l434_43455

theorem mary_investment_amount
  (A : ℝ := 100000) -- Future value in dollars
  (r : ℝ := 0.08) -- Annual interest rate
  (n : ℕ := 12) -- Compounded monthly
  (t : ℝ := 10) -- Time in years
  : (⌈A / (1 + r / n) ^ (n * t)⌉₊ = 45045) :=
by
  sorry

end NUMINAMATH_GPT_mary_investment_amount_l434_43455


namespace NUMINAMATH_GPT_largest_divisor_product_of_consecutive_odds_l434_43465

theorem largest_divisor_product_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) : 
  15 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) :=
sorry

end NUMINAMATH_GPT_largest_divisor_product_of_consecutive_odds_l434_43465


namespace NUMINAMATH_GPT_number_of_tables_l434_43440

/-- Problem Statement
  In a hall used for a conference, each table is surrounded by 8 stools and 4 chairs. Each stool has 3 legs,
  each chair has 4 legs, and each table has 4 legs. If the total number of legs for all tables, stools, and chairs is 704,
  the number of tables in the hall is 16. -/
theorem number_of_tables (legs_per_stool legs_per_chair legs_per_table total_legs t : ℕ) 
  (Hstools : ∀ tables, stools = 8 * tables)
  (Hchairs : ∀ tables, chairs = 4 * tables)
  (Hlegs : 3 * stools + 4 * chairs + 4 * t = total_legs)
  (Hleg_values : legs_per_stool = 3 ∧ legs_per_chair = 4 ∧ legs_per_table = 4)
  (Htotal_legs : total_legs = 704) :
  t = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_tables_l434_43440


namespace NUMINAMATH_GPT_largest_angle_in_convex_pentagon_l434_43468

theorem largest_angle_in_convex_pentagon (x : ℕ) (h : (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 540) : 
  x + 2 = 110 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_convex_pentagon_l434_43468


namespace NUMINAMATH_GPT_infinite_n_multiples_of_six_available_l434_43444

theorem infinite_n_multiples_of_six_available :
  ∃ (S : Set ℕ), (∀ n ∈ S, ∃ (A : Matrix (Fin 3) (Fin (n : ℕ)) Nat),
    (∀ (i : Fin n), (A 0 i + A 1 i + A 2 i) % 6 = 0) ∧ 
    (∀ (i : Fin 3), (Finset.univ.sum (λ j => A i j)) % 6 = 0)) ∧
  Set.Infinite S :=
sorry

end NUMINAMATH_GPT_infinite_n_multiples_of_six_available_l434_43444


namespace NUMINAMATH_GPT_problem_a4_inv_a4_l434_43413

theorem problem_a4_inv_a4 (a : ℝ) (h : (a + 1/a)^4 = 16) : (a^4 + 1/a^4) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_problem_a4_inv_a4_l434_43413


namespace NUMINAMATH_GPT_general_term_seq_l434_43429

open Nat

-- Definition of the sequence given conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n, n ≥ 1 → (n - 1) * a (n + 1) - n * a n + 1 = 0

-- To prove that the general term is a_n = n
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end NUMINAMATH_GPT_general_term_seq_l434_43429


namespace NUMINAMATH_GPT_age_difference_is_eight_l434_43406

theorem age_difference_is_eight (A B k : ℕ)
  (h1 : A = B + k)
  (h2 : A - 1 = 3 * (B - 1))
  (h3 : A = 2 * B + 3) :
  k = 8 :=
by sorry

end NUMINAMATH_GPT_age_difference_is_eight_l434_43406


namespace NUMINAMATH_GPT_solve_equation_in_natural_numbers_l434_43483

-- Define the main theorem
theorem solve_equation_in_natural_numbers :
  (∃ (x y z : ℕ), 2^x + 5^y + 63 = z! ∧ ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6))) :=
sorry

end NUMINAMATH_GPT_solve_equation_in_natural_numbers_l434_43483


namespace NUMINAMATH_GPT_possible_values_l434_43445

theorem possible_values (a b : ℕ → ℕ) (h1 : ∀ n, a n < (a (n + 1)))
  (h2 : ∀ n, b n < (b (n + 1)))
  (h3 : a 10 = b 10)
  (h4 : a 10 < 2017)
  (h5 : ∀ n, a (n + 2) = a (n + 1) + a n)
  (h6 : ∀ n, b (n + 1) = 2 * b n) :
  ∃ (a1 b1 : ℕ), (a 1 = a1) ∧ (b 1 = b1) ∧ (a1 + b1 = 13 ∨ a1 + b1 = 20) := sorry

end NUMINAMATH_GPT_possible_values_l434_43445


namespace NUMINAMATH_GPT_least_number_to_add_l434_43469

theorem least_number_to_add (n : ℕ) (h₁ : n = 1054) :
  ∃ k : ℕ, (n + k) % 23 = 0 ∧ k = 4 :=
by
  use 4
  have h₂ : n % 23 = 19 := by sorry
  have h₃ : (n + 4) % 23 = 0 := by sorry
  exact ⟨h₃, rfl⟩

end NUMINAMATH_GPT_least_number_to_add_l434_43469


namespace NUMINAMATH_GPT_B_work_days_l434_43460

/-- 
  A and B undertake to do a piece of work for $500.
  A alone can do it in 5 days while B alone can do it in a certain number of days.
  With the help of C, they finish it in 2 days. C's share is $200.
  Prove B alone can do the work in 10 days.
-/
theorem B_work_days (x : ℕ) (h1 : (1/5 : ℝ) + (1/x : ℝ) = 3/10) : x = 10 := 
  sorry

end NUMINAMATH_GPT_B_work_days_l434_43460


namespace NUMINAMATH_GPT_specialPermutationCount_l434_43481

def countSpecialPerms (n : ℕ) : ℕ := 2 ^ (n - 1)

theorem specialPermutationCount (n : ℕ) : 
  (countSpecialPerms n = 2 ^ (n - 1)) := 
by 
  sorry

end NUMINAMATH_GPT_specialPermutationCount_l434_43481


namespace NUMINAMATH_GPT_N_perfect_square_l434_43470

theorem N_perfect_square (N : ℕ) (hN_pos : N > 0) 
  (h_pairs : ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 2005 ∧ 
  ∀ p ∈ pairs, (1 : ℚ) / (p.1 : ℚ) + (1 : ℚ) / (p.2 : ℚ) = (1 : ℚ) / N ∧ p.1 > 0 ∧ p.2 > 0) : 
  ∃ k : ℕ, N = k^2 := 
sorry

end NUMINAMATH_GPT_N_perfect_square_l434_43470


namespace NUMINAMATH_GPT_option_C_incorrect_l434_43478

variable (a b : ℝ)

theorem option_C_incorrect : ((-a^3)^2 * (-b^2)^3) ≠ (a^6 * b^6) :=
by {
  sorry
}

end NUMINAMATH_GPT_option_C_incorrect_l434_43478


namespace NUMINAMATH_GPT_assignment_schemes_correct_l434_43474

-- Define the total number of students
def total_students : ℕ := 6

-- Define the total number of tasks
def total_tasks : ℕ := 4

-- Define a predicate that checks if a student can be assigned to task A
def can_assign_to_task_A (student : ℕ) : Prop := student ≠ 1 ∧ student ≠ 2

-- Calculate the total number of unrestricted assignments
def total_unrestricted_assignments : ℕ := 6 * 5 * 4 * 3

-- Calculate the restricted number of assignments if student A or B is assigned to task A
def restricted_assignments : ℕ := 2 * 5 * 4 * 3

-- Define the problem statement
def number_of_assignment_schemes : ℕ :=
  total_unrestricted_assignments - restricted_assignments

-- The theorem to prove
theorem assignment_schemes_correct :
  number_of_assignment_schemes = 240 :=
by
  -- We acknowledge the problem statement is correct
  sorry

end NUMINAMATH_GPT_assignment_schemes_correct_l434_43474


namespace NUMINAMATH_GPT_infinite_primes_of_form_m2_mn_n2_l434_43437

theorem infinite_primes_of_form_m2_mn_n2 : ∀ m n : ℤ, ∃ p : ℕ, ∃ k : ℕ, (p = k^2 + k * m + n^2) ∧ Prime k :=
sorry

end NUMINAMATH_GPT_infinite_primes_of_form_m2_mn_n2_l434_43437


namespace NUMINAMATH_GPT_evaporation_period_l434_43426

theorem evaporation_period
  (total_water : ℕ)
  (daily_evaporation_rate : ℝ)
  (percentage_evaporated : ℝ)
  (evaporation_period_days : ℕ)
  (h_total_water : total_water = 10)
  (h_daily_evaporation_rate : daily_evaporation_rate = 0.006)
  (h_percentage_evaporated : percentage_evaporated = 0.03)
  (h_evaporation_period_days : evaporation_period_days = 50):
  (percentage_evaporated * total_water) / daily_evaporation_rate = evaporation_period_days := by
  sorry

end NUMINAMATH_GPT_evaporation_period_l434_43426


namespace NUMINAMATH_GPT_find_A_l434_43477

theorem find_A (
  A B C A' r : ℕ
) (hA : A = 312) (hB : B = 270) (hC : C = 211)
  (hremA : A % A' = 4 * r)
  (hremB : B % A' = 2 * r)
  (hremC : C % A' = r) :
  A' = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l434_43477


namespace NUMINAMATH_GPT_second_percentage_increase_l434_43436

theorem second_percentage_increase 
  (P : ℝ) 
  (x : ℝ) 
  (h1: 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := 
  sorry

end NUMINAMATH_GPT_second_percentage_increase_l434_43436


namespace NUMINAMATH_GPT_four_times_num_mod_nine_l434_43442

theorem four_times_num_mod_nine (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 :=
sorry

end NUMINAMATH_GPT_four_times_num_mod_nine_l434_43442


namespace NUMINAMATH_GPT_inequality_solution_l434_43448

noncomputable def inequality_proof (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 2) : Prop :=
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a)) ≥ (27 / 13)

theorem inequality_solution (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 2) : 
  inequality_proof a b c h_positive h_sum :=
sorry

end NUMINAMATH_GPT_inequality_solution_l434_43448


namespace NUMINAMATH_GPT_residue_of_neg_1001_mod_37_l434_43453

theorem residue_of_neg_1001_mod_37 : (-1001 : ℤ) % 37 = 35 :=
by
  sorry

end NUMINAMATH_GPT_residue_of_neg_1001_mod_37_l434_43453


namespace NUMINAMATH_GPT_Jeff_Jogging_Extra_Friday_l434_43496

theorem Jeff_Jogging_Extra_Friday :
  let planned_daily_minutes := 60
  let days_in_week := 5
  let planned_weekly_minutes := days_in_week * planned_daily_minutes
  let thursday_cut_short := 20
  let actual_weekly_minutes := 290
  let thursday_run := planned_daily_minutes - thursday_cut_short
  let other_four_days_minutes := actual_weekly_minutes - thursday_run
  let mondays_to_wednesdays_run := 3 * planned_daily_minutes
  let friday_run := other_four_days_minutes - mondays_to_wednesdays_run
  let extra_run_on_friday := friday_run - planned_daily_minutes
  extra_run_on_friday = 10 := by trivial

end NUMINAMATH_GPT_Jeff_Jogging_Extra_Friday_l434_43496


namespace NUMINAMATH_GPT_bonnie_egg_count_indeterminable_l434_43438

theorem bonnie_egg_count_indeterminable
    (eggs_Kevin : ℕ)
    (eggs_George : ℕ)
    (eggs_Cheryl : ℕ)
    (diff_Cheryl_combined : ℕ)
    (c1 : eggs_Kevin = 5)
    (c2 : eggs_George = 9)
    (c3 : eggs_Cheryl = 56)
    (c4 : diff_Cheryl_combined = 29)
    (h₁ : eggs_Cheryl = diff_Cheryl_combined + (eggs_Kevin + eggs_George + some_children)) :
    ∀ (eggs_Bonnie : ℕ), ∃ some_children : ℕ, eggs_Bonnie = eggs_Bonnie :=
by
  -- The proof is omitted here
  sorry

end NUMINAMATH_GPT_bonnie_egg_count_indeterminable_l434_43438


namespace NUMINAMATH_GPT_simplify_expression_l434_43488

theorem simplify_expression (x : ℝ) : (x + 1) ^ 2 + x * (x - 2) = 2 * x ^ 2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l434_43488


namespace NUMINAMATH_GPT_multiplication_addition_example_l434_43467

theorem multiplication_addition_example :
  469138 * 9999 + 876543 * 12345 = 15512230997 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_addition_example_l434_43467


namespace NUMINAMATH_GPT_fair_die_proba_l434_43425
noncomputable def probability_of_six : ℚ := 1 / 6

theorem fair_die_proba : 
  (1 / 6 : ℚ) = probability_of_six :=
by
  sorry

end NUMINAMATH_GPT_fair_die_proba_l434_43425


namespace NUMINAMATH_GPT_log9_log11_lt_one_l434_43435

theorem log9_log11_lt_one (log9_pos : 0 < Real.log 9) (log11_pos : 0 < Real.log 11) : 
  Real.log 9 * Real.log 11 < 1 :=
by
  sorry

end NUMINAMATH_GPT_log9_log11_lt_one_l434_43435


namespace NUMINAMATH_GPT_identity_x_squared_minus_y_squared_l434_43414

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end NUMINAMATH_GPT_identity_x_squared_minus_y_squared_l434_43414


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l434_43427
-- Import the required Mathlib library in Lean 4

-- State the equivalent proof problem
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (|a| ≤ 1 → a ≤ 1) ∧ ¬ (a ≤ 1 → |a| ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l434_43427


namespace NUMINAMATH_GPT_ted_gathered_10_blue_mushrooms_l434_43473

noncomputable def blue_mushrooms_ted_gathered : ℕ :=
  let bill_red_mushrooms := 12
  let bill_brown_mushrooms := 6
  let ted_green_mushrooms := 14
  let total_white_spotted_mushrooms := 17
  
  let bill_white_spotted_red_mushrooms := bill_red_mushrooms / 2
  let bill_white_spotted_brown_mushrooms := bill_brown_mushrooms

  let total_bill_white_spotted_mushrooms := bill_white_spotted_red_mushrooms + bill_white_spotted_brown_mushrooms
  let ted_white_spotted_mushrooms := total_white_spotted_mushrooms - total_bill_white_spotted_mushrooms

  ted_white_spotted_mushrooms * 2

theorem ted_gathered_10_blue_mushrooms :
  blue_mushrooms_ted_gathered = 10 :=
by
  sorry

end NUMINAMATH_GPT_ted_gathered_10_blue_mushrooms_l434_43473


namespace NUMINAMATH_GPT_algebra_problem_l434_43404

theorem algebra_problem
  (x : ℝ)
  (h : 59 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = Real.sqrt 61 :=
sorry

end NUMINAMATH_GPT_algebra_problem_l434_43404


namespace NUMINAMATH_GPT_geese_in_marsh_l434_43492

theorem geese_in_marsh (number_of_ducks : ℕ) (total_number_of_birds : ℕ) (number_of_geese : ℕ) (h1 : number_of_ducks = 37) (h2 : total_number_of_birds = 95) : 
  number_of_geese = 58 := 
by
  sorry

end NUMINAMATH_GPT_geese_in_marsh_l434_43492


namespace NUMINAMATH_GPT_parallel_lines_count_l434_43416

theorem parallel_lines_count (n : ℕ) (h : 7 * (n - 1) = 588) : n = 85 :=
sorry

end NUMINAMATH_GPT_parallel_lines_count_l434_43416


namespace NUMINAMATH_GPT_sum_of_possible_values_of_N_l434_43461

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_N_l434_43461


namespace NUMINAMATH_GPT_samantha_born_in_1979_l434_43439

-- Condition definitions
def first_AMC8_year := 1985
def annual_event (n : ℕ) : ℕ := first_AMC8_year + n
def seventh_AMC8_year := annual_event 6

variable (Samantha_age_in_seventh_AMC8 : ℕ)
def Samantha_age_when_seventh_AMC8 := 12
def Samantha_birth_year := seventh_AMC8_year - Samantha_age_when_seventh_AMC8

-- Proof statement
theorem samantha_born_in_1979 : Samantha_birth_year = 1979 :=
by
  sorry

end NUMINAMATH_GPT_samantha_born_in_1979_l434_43439


namespace NUMINAMATH_GPT_radius_range_of_sector_l434_43408

theorem radius_range_of_sector (a : ℝ) (h : a > 0) :
  ∃ (R : ℝ), (a / (2 * (1 + π)) < R ∧ R < a / 2) :=
sorry

end NUMINAMATH_GPT_radius_range_of_sector_l434_43408


namespace NUMINAMATH_GPT_negation_of_universal_prop_l434_43479

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l434_43479


namespace NUMINAMATH_GPT_intersection_of_asymptotes_l434_43463

theorem intersection_of_asymptotes :
  ∃ x y : ℝ, (y = 1) ∧ (x = 3) ∧ (y = (x^2 - 6*x + 8) / (x^2 - 6*x + 9)) := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_asymptotes_l434_43463


namespace NUMINAMATH_GPT_train_speed_l434_43459

theorem train_speed (length_bridge : ℕ) (time_total : ℕ) (time_on_bridge : ℕ) (speed_of_train : ℕ) 
  (h1 : length_bridge = 800)
  (h2 : time_total = 60)
  (h3 : time_on_bridge = 40)
  (h4 : length_bridge + (time_total - time_on_bridge) * speed_of_train = time_total * speed_of_train) :
  speed_of_train = 20 := sorry

end NUMINAMATH_GPT_train_speed_l434_43459


namespace NUMINAMATH_GPT_train_speed_is_260_kmph_l434_43449

-- Define the conditions: length of the train and time to cross the pole
def length_of_train : ℝ := 130
def time_to_cross_pole : ℝ := 9

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the expected speed in kilometers per hour
def expected_speed_kmph : ℝ := 260

-- The theorem statement
theorem train_speed_is_260_kmph :
  (length_of_train / time_to_cross_pole) * conversion_factor = expected_speed_kmph :=
sorry

end NUMINAMATH_GPT_train_speed_is_260_kmph_l434_43449


namespace NUMINAMATH_GPT_xyz_value_l434_43490

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
  sorry

end NUMINAMATH_GPT_xyz_value_l434_43490


namespace NUMINAMATH_GPT_work_fraction_completed_after_first_phase_l434_43498

-- Definitions based on conditions
def total_work := 1 -- Assume total work as 1 unit
def initial_days := 100
def initial_people := 10
def first_phase_days := 20
def fired_people := 2
def remaining_days := 75
def remaining_people := initial_people - fired_people

-- Hypothesis about the rate of work initially and after firing people
def initial_rate := total_work / initial_days
def first_phase_work := first_phase_days * initial_rate
def remaining_work := total_work - first_phase_work
def remaining_rate := remaining_work / remaining_days

-- Proof problem statement: 
theorem work_fraction_completed_after_first_phase :
  (first_phase_work / total_work) = (15 / 64) :=
by
  -- This is the place where the actual formal proof should be written.
  sorry

end NUMINAMATH_GPT_work_fraction_completed_after_first_phase_l434_43498


namespace NUMINAMATH_GPT_quadratic_inequality_l434_43487

theorem quadratic_inequality (a b c : ℝ) (h : a^2 + a * b + a * c < 0) : b^2 > 4 * a * c := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_l434_43487


namespace NUMINAMATH_GPT_marks_lost_per_incorrect_sum_l434_43452

variables (marks_per_correct : ℕ) (total_attempts total_marks correct_sums : ℕ)
variable (marks_per_incorrect : ℕ)
variable (incorrect_sums : ℕ)

def calc_marks_per_incorrect_sum : Prop :=
  marks_per_correct = 3 ∧ 
  total_attempts = 30 ∧ 
  total_marks = 50 ∧ 
  correct_sums = 22 ∧ 
  incorrect_sums = total_attempts - correct_sums ∧ 
  (marks_per_correct * correct_sums) - (marks_per_incorrect * incorrect_sums) = total_marks ∧ 
  marks_per_incorrect = 2

theorem marks_lost_per_incorrect_sum : calc_marks_per_incorrect_sum 3 30 50 22 2 (30 - 22) :=
sorry

end NUMINAMATH_GPT_marks_lost_per_incorrect_sum_l434_43452


namespace NUMINAMATH_GPT_sin_x_eq_x_has_unique_root_in_interval_l434_43476

theorem sin_x_eq_x_has_unique_root_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧ x = Real.sin x :=
sorry

end NUMINAMATH_GPT_sin_x_eq_x_has_unique_root_in_interval_l434_43476


namespace NUMINAMATH_GPT_batch_production_equation_l434_43422

theorem batch_production_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 20) :
  (500 / x) = (300 / (x - 20)) :=
sorry

end NUMINAMATH_GPT_batch_production_equation_l434_43422


namespace NUMINAMATH_GPT_math_problem_l434_43418

-- Define the main variables a and b
def a : ℕ := 312
def b : ℕ := 288

-- State the main theorem to be proved
theorem math_problem : (a^2 - b^2) / 24 + 50 = 650 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l434_43418


namespace NUMINAMATH_GPT_weight_of_replaced_person_l434_43443

theorem weight_of_replaced_person
  (avg_increase : ∀ W : ℝ, W + 8 * 2.5 = W - X + 80)
  (new_person_weight : 80 = 80):
  X = 60 := by
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l434_43443


namespace NUMINAMATH_GPT_range_of_m_l434_43499

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l434_43499


namespace NUMINAMATH_GPT_find_xy_l434_43421

theorem find_xy (x y : ℝ) (h1 : (x / 6) * 12 = 11) (h2 : 4 * (x - y) + 5 = 11) : 
  x = 5.5 ∧ y = 4 :=
sorry

end NUMINAMATH_GPT_find_xy_l434_43421


namespace NUMINAMATH_GPT_law_of_sines_proof_l434_43493

noncomputable def law_of_sines (a b c α β γ : ℝ) :=
  (a / Real.sin α = b / Real.sin β) ∧
  (b / Real.sin β = c / Real.sin γ) ∧
  (α + β + γ = Real.pi)

theorem law_of_sines_proof (a b c α β γ : ℝ) (h : law_of_sines a b c α β γ) :
  (a = b * Real.cos γ + c * Real.cos β) ∧
  (b = c * Real.cos α + a * Real.cos γ) ∧
  (c = a * Real.cos β + b * Real.cos α) :=
sorry

end NUMINAMATH_GPT_law_of_sines_proof_l434_43493


namespace NUMINAMATH_GPT_total_students_l434_43471

-- Definitions extracted from the conditions 
def ratio_boys_girls := 8 / 5
def number_of_boys := 128

-- Theorem to prove the total number of students
theorem total_students : 
  (128 + (5 / 8) * 128 = 208) ∧ ((128 : ℝ) * (13 / 8) = 208) :=
by
  sorry

end NUMINAMATH_GPT_total_students_l434_43471


namespace NUMINAMATH_GPT_john_writing_time_l434_43423

def pages_per_day : ℕ := 20
def pages_per_book : ℕ := 400
def number_of_books : ℕ := 3

theorem john_writing_time : (pages_per_book / pages_per_day) * number_of_books = 60 :=
by
  -- The proof should be placed here.
  sorry

end NUMINAMATH_GPT_john_writing_time_l434_43423


namespace NUMINAMATH_GPT_inequality_transformation_l434_43486

theorem inequality_transformation (x y : ℝ) (h : x > y) : 3 * x > 3 * y :=
by sorry

end NUMINAMATH_GPT_inequality_transformation_l434_43486


namespace NUMINAMATH_GPT_factorization_of_polynomial_l434_43480

theorem factorization_of_polynomial :
  ∀ x : ℝ, (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) = (x - 1)^4 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l434_43480


namespace NUMINAMATH_GPT_inequality_proof_l434_43489

variable {m n : ℝ}

theorem inequality_proof (h1 : m < n) (h2 : n < 0) : (n / m + m / n > 2) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l434_43489


namespace NUMINAMATH_GPT_find_ab_l434_43428

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, (3 * x - a) * (2 * x + 5) - x = 6 * x^2 + 2 * (5 * x - b)) → a = 2 ∧ b = 5 :=
by
  intro h
  -- We assume the condition holds for all x
  sorry -- Proof not needed as per instructions

end NUMINAMATH_GPT_find_ab_l434_43428


namespace NUMINAMATH_GPT_find_f_inv_486_l434_43484

open Function

noncomputable def f (x : ℕ) : ℕ := sorry -- placeholder for function definition

axiom f_condition1 : f 5 = 2
axiom f_condition2 : ∀ (x : ℕ), f (3 * x) = 3 * f x

theorem find_f_inv_486 : f⁻¹' {486} = {1215} := sorry

end NUMINAMATH_GPT_find_f_inv_486_l434_43484


namespace NUMINAMATH_GPT_number_in_marked_square_is_10_l434_43451

theorem number_in_marked_square_is_10 : 
  ∃ f : ℕ × ℕ → ℕ, 
    (f (0,0) = 5 ∧ f (0,1) = 6 ∧ f (0,2) = 7) ∧ 
    (∀ r c, r > 0 → 
      f (r,c) = f (r-1,c) + f (r-1,c+1)) 
    ∧ f (1, 1) = 13 
    ∧ f (2, 1) = 10 :=
    sorry

end NUMINAMATH_GPT_number_in_marked_square_is_10_l434_43451


namespace NUMINAMATH_GPT_arithmetic_progression_root_difference_l434_43466

theorem arithmetic_progression_root_difference (a b c : ℚ) (h : 81 * a * a * a - 225 * a * a + 164 * a - 30 = 0)
  (hb : b = 5/3) (hprog : ∃ d : ℚ, a = b - d ∧ c = b + d) :
  c - a = 5 / 9 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_root_difference_l434_43466


namespace NUMINAMATH_GPT_dumpling_probability_l434_43412

theorem dumpling_probability :
  let total_dumplings := 15
  let choose4 := Nat.choose total_dumplings 4
  let choose1 := Nat.choose 3 1
  let choose5_2 := Nat.choose 5 2
  let choose5_1 := Nat.choose 5 1
  (choose1 * choose5_2 * choose5_1 * choose5_1) / choose4 = 50 / 91 := by
  sorry

end NUMINAMATH_GPT_dumpling_probability_l434_43412


namespace NUMINAMATH_GPT_compute_x_l434_43494

theorem compute_x 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 0.1)
  (hs1 : ∑' n, 4 * x^n = 4 / (1 - x))
  (hs2 : ∑' n, 4 * (10^n - 1) * x^n = 4 * (4 / (1 - x))) :
  x = 3 / 40 :=
by
  sorry

end NUMINAMATH_GPT_compute_x_l434_43494


namespace NUMINAMATH_GPT_person_A_work_days_l434_43475

theorem person_A_work_days (x : ℝ) (h1 : 0 < x) 
                                 (h2 : ∃ b_work_rate, b_work_rate = 1 / 30) 
                                 (h3 : 5 * (1 / x + 1 / 30) = 0.5) : 
  x = 15 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_person_A_work_days_l434_43475


namespace NUMINAMATH_GPT_Darren_paints_432_feet_l434_43431

theorem Darren_paints_432_feet (t : ℝ) (h : t = 792) (paint_ratio : ℝ) 
  (h_ratio : paint_ratio = 1.20) : 
  let d := t / (1 + paint_ratio)
  let D := d * paint_ratio
  D = 432 :=
by
  sorry

end NUMINAMATH_GPT_Darren_paints_432_feet_l434_43431


namespace NUMINAMATH_GPT_star_operation_possible_l434_43401

noncomputable def star_operation_exists : Prop := 
  ∃ (star : ℤ → ℤ → ℤ), 
  (∀ (a b c : ℤ), star (star a b) c = star a (star b c)) ∧ 
  (∀ (x y : ℤ), star (star x x) y = y ∧ star y (star x x) = y)

theorem star_operation_possible : star_operation_exists :=
sorry

end NUMINAMATH_GPT_star_operation_possible_l434_43401


namespace NUMINAMATH_GPT_nearby_island_banana_production_l434_43407

theorem nearby_island_banana_production
  (x : ℕ)
  (h_prod: 10 * x + x = 99000) :
  x = 9000 :=
sorry

end NUMINAMATH_GPT_nearby_island_banana_production_l434_43407


namespace NUMINAMATH_GPT_michael_total_cost_l434_43432

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def miles_driven : ℕ := 299

def total_cost (rental_fee : ℝ) (charge_per_mile : ℝ) (miles_driven : ℕ) : ℝ :=
  rental_fee + (charge_per_mile * miles_driven)

theorem michael_total_cost :
  total_cost rental_fee charge_per_mile miles_driven = 95.74 :=
by
  sorry

end NUMINAMATH_GPT_michael_total_cost_l434_43432


namespace NUMINAMATH_GPT_train_length_l434_43400

theorem train_length (speed_kmph : ℕ) (time_s : ℕ) (platform_length_m : ℕ) (h1 : speed_kmph = 72) (h2 : time_s = 26) (h3 : platform_length_m = 260) :
  ∃ train_length_m : ℕ, train_length_m = 260 := by
  sorry

end NUMINAMATH_GPT_train_length_l434_43400


namespace NUMINAMATH_GPT_john_and_lisa_meet_at_midpoint_l434_43434

-- Define the conditions
def john_position : ℝ × ℝ := (2, 9)
def lisa_position : ℝ × ℝ := (-6, 1)

-- Assertion for their meeting point
theorem john_and_lisa_meet_at_midpoint :
  ∃ (x y : ℝ), (x, y) = ((john_position.1 + lisa_position.1) / 2,
                         (john_position.2 + lisa_position.2) / 2) :=
sorry

end NUMINAMATH_GPT_john_and_lisa_meet_at_midpoint_l434_43434


namespace NUMINAMATH_GPT_find_nat_int_l434_43405

theorem find_nat_int (x y : ℕ) (h : x^2 = y^2 + 7 * y + 6) : x = 6 ∧ y = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_nat_int_l434_43405


namespace NUMINAMATH_GPT_sum_of_first_5n_l434_43485

theorem sum_of_first_5n (n : ℕ) (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 504) :
  (5 * n * (5 * n + 1)) / 2 = 1035 :=
sorry

end NUMINAMATH_GPT_sum_of_first_5n_l434_43485


namespace NUMINAMATH_GPT_complement_union_l434_43497

open Set

-- Definitions from the given conditions
def U : Set ℕ := {x | x ≤ 9}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Statement of the proof problem
theorem complement_union :
  compl (A ∪ B) = {7, 8, 9} :=
sorry

end NUMINAMATH_GPT_complement_union_l434_43497


namespace NUMINAMATH_GPT_amount_in_cup_after_division_l434_43491

theorem amount_in_cup_after_division (removed remaining cups : ℕ) (h : remaining + removed = 40) : 
  (40 / cups = 8) :=
by
  sorry

end NUMINAMATH_GPT_amount_in_cup_after_division_l434_43491
