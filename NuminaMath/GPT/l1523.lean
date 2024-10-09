import Mathlib

namespace Kvi_wins_race_l1523_152381

/-- Define the frogs and their properties --/
structure Frog :=
  (name : String)
  (jump_distance_in_dm : ℕ) /-- jump distance in decimeters --/
  (jumps_per_cycle : ℕ) /-- number of jumps per cycle (unit time of reference) --/

def FrogKva : Frog := ⟨"Kva", 6, 2⟩
def FrogKvi : Frog := ⟨"Kvi", 4, 3⟩

/-- Define the conditions for the race --/
def total_distance_in_m : ℕ := 40
def total_distance_in_dm := total_distance_in_m * 10

/-- Racing function to determine winner --/
def race_winner (f1 f2 : Frog) (total_distance : ℕ) : String :=
  if (total_distance % (f1.jump_distance_in_dm * f1.jumps_per_cycle) < total_distance % (f2.jump_distance_in_dm * f2.jumps_per_cycle))
  then f1.name
  else f2.name

/-- Proving Kvi wins under the given conditions --/
theorem Kvi_wins_race :
  race_winner FrogKva FrogKvi total_distance_in_dm = "Kvi" :=
by
  sorry

end Kvi_wins_race_l1523_152381


namespace sequence_max_length_l1523_152368

theorem sequence_max_length (x : ℕ) :
  (2000 - 2 * x > 0) ∧ (3 * x - 2000 > 0) ∧ (4000 - 5 * x > 0) ∧ 
  (8 * x - 6000 > 0) ∧ (10000 - 13 * x > 0) ∧ (21 * x - 16000 > 0) → x = 762 :=
by
  sorry

end sequence_max_length_l1523_152368


namespace no_positive_alpha_exists_l1523_152395

theorem no_positive_alpha_exists :
  ¬ ∃ α > 0, ∀ x : ℝ, |Real.cos x| + |Real.cos (α * x)| > Real.sin x + Real.sin (α * x) :=
by
  sorry

end no_positive_alpha_exists_l1523_152395


namespace jeremy_age_l1523_152380

theorem jeremy_age (A J C : ℕ) (h1 : A + J + C = 132) (h2 : A = J / 3) (h3 : C = 2 * A) : J = 66 :=
by
  sorry

end jeremy_age_l1523_152380


namespace chord_length_of_curve_by_line_l1523_152307

theorem chord_length_of_curve_by_line :
  let x (t : ℝ) := 2 + 2 * t
  let y (t : ℝ) := -t
  let curve_eq (θ : ℝ) := 4 * Real.cos θ
  ∃ a b : ℝ, (x a = 2 + 2 * a ∧ y a = -a) ∧ (x b = 2 + 2 * b ∧ y b = -b) ∧
  ((x a - x b)^2 + (y a - y b)^2 = 4^2) :=
by
  sorry

end chord_length_of_curve_by_line_l1523_152307


namespace symmetric_point_proof_l1523_152373

def symmetric_point (P : ℝ × ℝ) (line : ℝ → ℝ) : ℝ × ℝ := sorry

theorem symmetric_point_proof :
  symmetric_point (2, 5) (λ x => 1 - x) = (-4, -1) := sorry

end symmetric_point_proof_l1523_152373


namespace value_of_r6_plus_s6_l1523_152351

theorem value_of_r6_plus_s6 :
  ∀ r s : ℝ, (r^2 - 2 * r + Real.sqrt 2 = 0) ∧ (s^2 - 2 * s + Real.sqrt 2 = 0) →
  (r^6 + s^6 = 904 - 640 * Real.sqrt 2) :=
by
  intros r s h
  -- Proof skipped
  sorry

end value_of_r6_plus_s6_l1523_152351


namespace mean_of_set_eq_10point6_l1523_152356

open Real -- For real number operations

theorem mean_of_set_eq_10point6 (n : ℝ)
  (h : n + 7 = 11) :
  (4 + 7 + 11 + 13 + 18) / 5 = 10.6 :=
by
  have h1 : n = 4 := by linarith
  sorry -- skip the proof part

end mean_of_set_eq_10point6_l1523_152356


namespace time_to_reach_ship_l1523_152329

-- Conditions in Lean 4
def rate : ℕ := 22
def depth : ℕ := 7260

-- The theorem that we want to prove
theorem time_to_reach_ship : depth / rate = 330 := by
  sorry

end time_to_reach_ship_l1523_152329


namespace power_function_passes_through_fixed_point_l1523_152378

theorem power_function_passes_through_fixed_point 
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
  (P : ℝ × ℝ) (hP : P = (4, 2))
  (f : ℝ → ℝ) (hf : f x = x ^ a) : ∀ x, f x = x ^ (1 / 2) :=
by
  sorry

end power_function_passes_through_fixed_point_l1523_152378


namespace part_I_part_II_l1523_152304

-- Define the function f(x) as per the problem's conditions
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

theorem part_I (x : ℝ) (h₁ : 1 ≠ 0) : 
  (f x 1 > 2) ↔ (x < 1 / 2 ∨ x > 5 / 2) :=
by
  sorry

theorem part_II (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f b a ≥ f a a ∧ (f b a = f a a ↔ ((2 * a - b ≥ 0 ∧ b - a ≥ 0) ∨ (2 * a - b ≤ 0 ∧ b - a ≤ 0) ∨ (2 * a - b = 0) ∨ (b - a = 0))) :=
by
  sorry

end part_I_part_II_l1523_152304


namespace ellipse_tangent_line_equation_l1523_152340

variable {r a b x0 y0 x y : ℝ}
variable (h_r_pos : r > 0) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a > b)
variable (ellipse_eq : (x / a)^2 + (y / b)^2 = 1)
variable (tangent_circle_eq : x0 * x / r^2 + y0 * y / r^2 = 1)

theorem ellipse_tangent_line_equation :
  (a > b) → (a > 0) → (b > 0) → (x0 ≠ 0 ∨ y0 ≠ 0) → (x/a)^2 + (y/b)^2 = 1 →
  (x0 * x / a^2 + y0 * y / b^2 = 1) :=
by
  sorry

end ellipse_tangent_line_equation_l1523_152340


namespace percent_of_volume_filled_by_cubes_l1523_152394

theorem percent_of_volume_filled_by_cubes :
  let box_width := 8
  let box_height := 6
  let box_length := 12
  let cube_size := 2
  let box_volume := box_width * box_height * box_length
  let cube_volume := cube_size ^ 3
  let num_cubes := (box_width / cube_size) * (box_height / cube_size) * (box_length / cube_size)
  let cubes_volume := num_cubes * cube_volume
  (cubes_volume / box_volume : ℝ) * 100 = 100 := by
  sorry

end percent_of_volume_filled_by_cubes_l1523_152394


namespace fraction_identity_l1523_152326

variable (a b : ℝ)

theorem fraction_identity (h : a ≠ 0) : 
  (2 * b + a) / a + (a - 2 * b) / a = 2 := 
by
  sorry

end fraction_identity_l1523_152326


namespace min_value_frac_l1523_152336

theorem min_value_frac (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 10) : 
  ∃ x, (x = (1 / m) + (4 / n)) ∧ (∀ y, y = (1 / m) + (4 / n) → y ≥ 9 / 10) :=
sorry

end min_value_frac_l1523_152336


namespace find_m_l1523_152360

open Real

noncomputable def curve_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

noncomputable def line_equation (m t x y : ℝ) : Prop :=
  x = (sqrt 3 / 2) * t + m ∧ y = (1 / 2) * t

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_m (m : ℝ) (h_nonneg : 0 ≤ m) :
  (∀ (t1 t2 : ℝ), (∀ x y, line_equation m t1 x y → curve_equation x y) → 
                   (∀ x y, line_equation m t2 x y → curve_equation x y) →
                   (dist m 0 x1 y1) * (dist m 0 x2 y2) = 1) →
  m = 1 ∨ m = 1 + sqrt 2 :=
sorry

end find_m_l1523_152360


namespace two_fruits_probability_l1523_152371

noncomputable def prob_exactly_two_fruits : ℚ := 10 / 9

theorem two_fruits_probability :
  (∀ (f : ℕ → ℝ), (f 0 = 1/3) ∧ (f 1 = 1/3) ∧ (f 2 = 1/3) ∧
   (∃ f1 f2 f3, f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (f1 + f2 + f3 = prob_exactly_two_fruits))) :=
sorry

end two_fruits_probability_l1523_152371


namespace probability_of_exactly_9_correct_matches_is_zero_l1523_152387

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end probability_of_exactly_9_correct_matches_is_zero_l1523_152387


namespace find_number_l1523_152397

theorem find_number (x : ℤ) (h : (((55 + x) / 7 + 40) * 5 = 555)) : x = 442 :=
sorry

end find_number_l1523_152397


namespace largest_c_l1523_152369

theorem largest_c (c : ℝ) : (∃ x : ℝ, x^2 + 4 * x + c = -3) → c ≤ 1 :=
by
  sorry

end largest_c_l1523_152369


namespace k_values_equation_satisfied_l1523_152321

theorem k_values_equation_satisfied : 
  {k : ℕ | k > 0 ∧ ∃ r s : ℕ, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s} = {2, 3, 4, 8} :=
by
  sorry

end k_values_equation_satisfied_l1523_152321


namespace sin_nine_pi_over_two_plus_theta_l1523_152324

variable (θ : ℝ)

-- Conditions: Point A(4, -3) lies on the terminal side of angle θ
def terminal_point_on_angle (θ : ℝ) : Prop :=
  let x := 4
  let y := -3
  let hypotenuse := Real.sqrt ((x ^ 2) + (y ^ 2))
  hypotenuse = 5 ∧ Real.cos θ = x / hypotenuse

theorem sin_nine_pi_over_two_plus_theta (θ : ℝ) 
  (h : terminal_point_on_angle θ) : 
  Real.sin (9 * Real.pi / 2 + θ) = 4 / 5 :=
sorry

end sin_nine_pi_over_two_plus_theta_l1523_152324


namespace mike_total_payment_l1523_152361

def camera_initial_cost : ℝ := 4000
def camera_increase_rate : ℝ := 0.30
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200
def sales_tax_rate : ℝ := 0.08

def new_camera_cost := camera_initial_cost * (1 + camera_increase_rate)
def discounted_lens_cost := lens_initial_cost - lens_discount
def total_purchase_before_tax := new_camera_cost + discounted_lens_cost
def sales_tax := total_purchase_before_tax * sales_tax_rate
def total_purchase_with_tax := total_purchase_before_tax + sales_tax

theorem mike_total_payment : total_purchase_with_tax = 5832 := by
  sorry

end mike_total_payment_l1523_152361


namespace total_cost_of_topsoil_l1523_152354

def cost_per_cubic_foot : ℝ := 8
def cubic_yards_to_cubic_feet : ℝ := 27
def volume_in_yards : ℝ := 7

theorem total_cost_of_topsoil :
  (cubic_yards_to_cubic_feet * volume_in_yards) * cost_per_cubic_foot = 1512 :=
by
  sorry

end total_cost_of_topsoil_l1523_152354


namespace min_int_solution_inequality_l1523_152328

theorem min_int_solution_inequality : ∃ x : ℤ, 4 * (x + 1) + 2 > x - 1 ∧ ∀ y : ℤ, 4 * (y + 1) + 2 > y - 1 → y ≥ x := 
by 
  sorry

end min_int_solution_inequality_l1523_152328


namespace savings_on_cheapest_flight_l1523_152388

theorem savings_on_cheapest_flight :
  let delta_price := 850
  let delta_discount := 0.20
  let united_price := 1100
  let united_discount := 0.30
  let delta_final_price := delta_price - delta_price * delta_discount
  let united_final_price := united_price - united_price * united_discount
  delta_final_price < united_final_price →
  united_final_price - delta_final_price = 90 :=
by
  sorry

end savings_on_cheapest_flight_l1523_152388


namespace quadratic_inequality_solution_l1523_152379

theorem quadratic_inequality_solution:
  (∃ p : ℝ, ∀ x : ℝ, x^2 + p * x - 6 < 0 ↔ -3 < x ∧ x < 2) → ∃ p : ℝ, p = 1 :=
by
  intro h
  sorry

end quadratic_inequality_solution_l1523_152379


namespace trigonometric_quadrant_l1523_152312

theorem trigonometric_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
  (π / 2 < α) ∧ (α < π) :=
by
  sorry

end trigonometric_quadrant_l1523_152312


namespace fraction_evaluation_l1523_152372

def h (x : ℤ) : ℤ := 3 * x + 4
def k (x : ℤ) : ℤ := 4 * x - 3

theorem fraction_evaluation :
  (h (k (h 3))) / (k (h (k 3))) = 151 / 121 :=
by sorry

end fraction_evaluation_l1523_152372


namespace geometric_sum_S15_l1523_152383

noncomputable def S (n : ℕ) : ℝ := sorry  -- Assume S is defined for the sequence sum

theorem geometric_sum_S15 (S_5 S_10 : ℝ) (h1 : S_5 = 5) (h2 : S_10 = 30) : 
    S 15 = 155 := 
by 
  -- Placeholder for geometric sequence proof
  sorry

end geometric_sum_S15_l1523_152383


namespace two_numbers_sum_gcd_l1523_152348

theorem two_numbers_sum_gcd (x y : ℕ) (h1 : x + y = 432) (h2 : Nat.gcd x y = 36) :
  (x = 36 ∧ y = 396) ∨ (x = 180 ∧ y = 252) ∨ (x = 396 ∧ y = 36) ∨ (x = 252 ∧ y = 180) :=
by
  -- Proof TBD
  sorry

end two_numbers_sum_gcd_l1523_152348


namespace digit_B_divisibility_l1523_152302

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧ (2 * 100 + B * 10 + 9) % 13 = 0 ↔ B = 0 :=
by
  sorry

end digit_B_divisibility_l1523_152302


namespace calculate_product_l1523_152323

variable (EF FG GH HE : ℚ)
variable (x y : ℚ)

-- Conditions
axiom h1 : EF = 110
axiom h2 : FG = 16 * y^3
axiom h3 : GH = 6 * x + 2
axiom h4 : HE = 64
-- Parallelogram properties
axiom h5 : EF = GH
axiom h6 : FG = HE

theorem calculate_product (EF FG GH HE : ℚ) (x y : ℚ)
  (h1 : EF = 110) (h2 : FG = 16 * y ^ 3) (h3 : GH = 6 * x + 2) (h4 : HE = 64) (h5 : EF = GH) (h6 : FG = HE) :
  x * y = 18 * (4) ^ (1/3) := by
  sorry

end calculate_product_l1523_152323


namespace intersection_M_N_l1523_152317

  open Set

  def M : Set ℝ := {x | Real.log x > 0}
  def N : Set ℝ := {x | x^2 ≤ 4}

  theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
  by
    sorry
  
end intersection_M_N_l1523_152317


namespace lower_bound_for_x_l1523_152330

variable {x y : ℝ}  -- declaring x and y as real numbers

theorem lower_bound_for_x 
  (h₁ : 3 < x) (h₂ : x < 6)
  (h₃ : 6 < y) (h₄ : y < 8)
  (h₅ : y - x = 4) : 
  ∃ ε > 0, 3 + ε = x := 
sorry

end lower_bound_for_x_l1523_152330


namespace multiply_decimals_l1523_152389

theorem multiply_decimals :
  0.25 * 0.08 = 0.02 :=
sorry

end multiply_decimals_l1523_152389


namespace marcel_corn_l1523_152337

theorem marcel_corn (C : ℕ) (H1 : ∃ D, D = C / 2) (H2 : 27 = C + C / 2 + 8 + 4) : C = 10 :=
sorry

end marcel_corn_l1523_152337


namespace order_of_values_l1523_152345

noncomputable def a : ℝ := (1 / 5) ^ 2
noncomputable def b : ℝ := 2 ^ (1 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log 2  -- change of base from log base 2 to natural log

theorem order_of_values : c < a ∧ a < b :=
by
  sorry

end order_of_values_l1523_152345


namespace math_problem_l1523_152367

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c + 2 * (a + b + c) = 672 :=
by
  sorry

end math_problem_l1523_152367


namespace sum_of_digits_floor_large_number_div_50_eq_457_l1523_152333

-- Define a helper function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the large number as the sum of its components
def large_number : ℕ :=
  51 * 10^96 + 52 * 10^94 + 53 * 10^92 + 54 * 10^90 + 55 * 10^88 + 56 * 10^86 + 
  57 * 10^84 + 58 * 10^82 + 59 * 10^80 + 60 * 10^78 + 61 * 10^76 + 62 * 10^74 + 
  63 * 10^72 + 64 * 10^70 + 65 * 10^68 + 66 * 10^66 + 67 * 10^64 + 68 * 10^62 + 
  69 * 10^60 + 70 * 10^58 + 71 * 10^56 + 72 * 10^54 + 73 * 10^52 + 74 * 10^50 + 
  75 * 10^48 + 76 * 10^46 + 77 * 10^44 + 78 * 10^42 + 79 * 10^40 + 80 * 10^38 + 
  81 * 10^36 + 82 * 10^34 + 83 * 10^32 + 84 * 10^30 + 85 * 10^28 + 86 * 10^26 + 
  87 * 10^24 + 88 * 10^22 + 89 * 10^20 + 90 * 10^18 + 91 * 10^16 + 92 * 10^14 + 
  93 * 10^12 + 94 * 10^10 + 95 * 10^8 + 96 * 10^6 + 97 * 10^4 + 98 * 10^2 + 99

-- Define the main statement to be proven
theorem sum_of_digits_floor_large_number_div_50_eq_457 : 
    sum_of_digits (Nat.floor (large_number / 50)) = 457 :=
by
  sorry

end sum_of_digits_floor_large_number_div_50_eq_457_l1523_152333


namespace root_interval_sum_l1523_152393

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) (h1 : b - a = 1) (h2 : ∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) : a + b = -3 :=
by
  sorry

end root_interval_sum_l1523_152393


namespace polynomial_sum_of_squares_l1523_152399

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ (Q R : Polynomial ℝ), P = Q^2 + R^2 :=
sorry

end polynomial_sum_of_squares_l1523_152399


namespace direct_proportion_m_n_l1523_152319

theorem direct_proportion_m_n (m n : ℤ) (h₁ : m - 2 = 1) (h₂ : n + 1 = 0) : m + n = 2 :=
by
  sorry

end direct_proportion_m_n_l1523_152319


namespace problem_proof_l1523_152355

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def y := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem problem_proof :
  is_multiple_of 5 y ∧
  is_multiple_of 10 y ∧
  is_multiple_of 20 y ∧
  is_multiple_of 40 y := 
by
  sorry

end problem_proof_l1523_152355


namespace problem_solution_l1523_152374

-- Define the given circle equation C
def circle_C_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 3 = 0

-- Define the line of symmetry
def line_symmetry_eq (x y : ℝ) : Prop := y = -x - 4

-- Define the symmetric circle equation
def sym_circle_eq (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

theorem problem_solution (x y : ℝ)
  (H1 : circle_C_eq x y)
  (H2 : line_symmetry_eq x y) :
  sym_circle_eq x y :=
sorry

end problem_solution_l1523_152374


namespace gcd_102_238_l1523_152309

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  -- Given conditions as part of proof structure
  have h1 : 238 = 102 * 2 + 34 := by rfl
  have h2 : 102 = 34 * 3 := by rfl
  sorry

end gcd_102_238_l1523_152309


namespace polygon_sides_sum_l1523_152350

theorem polygon_sides_sum (n : ℕ) (x : ℝ) (hx : 0 < x ∧ x < 180) 
  (h_sum : 180 * (n - 2) - x = 2190) : n = 15 :=
sorry

end polygon_sides_sum_l1523_152350


namespace worker_b_time_l1523_152320

theorem worker_b_time (time_A : ℝ) (time_A_B_together : ℝ) (T_B : ℝ) 
  (h1 : time_A = 8) 
  (h2 : time_A_B_together = 4.8) 
  (h3 : (1 / time_A) + (1 / T_B) = (1 / time_A_B_together)) :
  T_B = 12 :=
sorry

end worker_b_time_l1523_152320


namespace total_distance_travelled_l1523_152318

theorem total_distance_travelled (distance_to_market : ℕ) (travel_time_minutes : ℕ) (speed_mph : ℕ) 
  (h1 : distance_to_market = 30) 
  (h2 : travel_time_minutes = 30) 
  (h3 : speed_mph = 20) : 
  (distance_to_market + ((travel_time_minutes / 60) * speed_mph) = 40) :=
by
  sorry

end total_distance_travelled_l1523_152318


namespace find_sum_pqr_l1523_152352

theorem find_sum_pqr (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h : (p + q + r)^3 - p^3 - q^3 - r^3 = 200) : 
  p + q + r = 7 :=
by 
  sorry

end find_sum_pqr_l1523_152352


namespace bus_problem_l1523_152342

theorem bus_problem (x : ℕ) : 50 * x + 10 = 52 * x + 2 := 
sorry

end bus_problem_l1523_152342


namespace time_reduced_fraction_l1523_152396

theorem time_reduced_fraction
  (T : ℝ)
  (V : ℝ)
  (hV : V = 42)
  (D : ℝ)
  (hD_1 : D = V * T)
  (V' : ℝ)
  (hV' : V' = V + 21)
  (T' : ℝ)
  (hD_2 : D = V' * T') :
  (T - T') / T = 1 / 3 :=
by
  -- Proof omitted
  sorry

end time_reduced_fraction_l1523_152396


namespace Annabelle_saved_12_dollars_l1523_152376

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end Annabelle_saved_12_dollars_l1523_152376


namespace sphere_radius_eq_l1523_152313

theorem sphere_radius_eq (h d : ℝ) (r_cylinder : ℝ) (r : ℝ) (pi : ℝ) 
  (h_eq : h = 14) (d_eq : d = 14) (r_cylinder_eq : r_cylinder = d / 2) :
  4 * pi * r^2 = 2 * pi * r_cylinder * h → r = 7 := by
  sorry

end sphere_radius_eq_l1523_152313


namespace total_legs_of_collection_l1523_152391

theorem total_legs_of_collection (spiders ants : ℕ) (legs_per_spider legs_per_ant : ℕ)
  (h_spiders : spiders = 8) (h_ants : ants = 12)
  (h_legs_per_spider : legs_per_spider = 8) (h_legs_per_ant : legs_per_ant = 6) :
  (spiders * legs_per_spider + ants * legs_per_ant) = 136 :=
by
  sorry

end total_legs_of_collection_l1523_152391


namespace students_passed_finals_l1523_152349

def total_students := 180
def students_bombed := 1 / 4 * total_students
def remaining_students_after_bombed := total_students - students_bombed
def students_didnt_show := 1 / 3 * remaining_students_after_bombed
def students_failed_less_than_D := 20

theorem students_passed_finals : 
  total_students - students_bombed - students_didnt_show - students_failed_less_than_D = 70 := 
by 
  -- calculation to derive 70
  sorry

end students_passed_finals_l1523_152349


namespace no_such_function_l1523_152343

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x - y :=
by
  sorry

end no_such_function_l1523_152343


namespace union_A_B_eq_neg2_neg1_0_l1523_152375

def setA : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}
def setB : Set ℤ := {-2, -1}

theorem union_A_B_eq_neg2_neg1_0 : (setA ∪ setB) = ({-2, -1, 0} : Set ℤ) :=
by
  sorry

end union_A_B_eq_neg2_neg1_0_l1523_152375


namespace isosceles_triangle_base_length_l1523_152301

noncomputable def length_of_base (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : ℝ :=
  (12 - 2 * a) / 2

theorem isosceles_triangle_base_length (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : length_of_base a b h_isosceles h_side h_perimeter = 4.5 :=
sorry

end isosceles_triangle_base_length_l1523_152301


namespace trajectory_of_P_below_x_axis_l1523_152359

theorem trajectory_of_P_below_x_axis (x y : ℝ) (P_below_x_axis : y < 0)
    (tangent_to_parabola : ∃ A B: ℝ × ℝ, A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2 ∧ (x^2 + y^2 = 1))
    (AB_tangent_to_circle : ∀ (x0 y0 : ℝ), x0^2 + y0^2 = 1 → x0 * x + y0 * y = 1) :
    y^2 - x^2 = 1 :=
sorry

end trajectory_of_P_below_x_axis_l1523_152359


namespace negation_of_universal_quantifier_proposition_l1523_152316

variable (x : ℝ)

theorem negation_of_universal_quantifier_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1/4 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1/4 < 0) :=
sorry

end negation_of_universal_quantifier_proposition_l1523_152316


namespace diane_age_proof_l1523_152386

noncomputable def diane_age (A Al D : ℕ) : Prop :=
  ((A + (30 - D) = 60) ∧ (Al + (30 - D) = 15) ∧ (A + Al = 47)) → (D = 16)

theorem diane_age_proof : ∃ (D : ℕ), ∃ (A Al : ℕ), diane_age A Al D :=
by {
  sorry
}

end diane_age_proof_l1523_152386


namespace find_a_of_cool_frog_meeting_l1523_152382

-- Question and conditions
def frogs : ℕ := 16
def friend_probability : ℚ := 1 / 2
def cool_condition (f: ℕ → ℕ) : Prop := ∀ i, f i % 4 = 0

-- Example theorem where we need to find 'a'
theorem find_a_of_cool_frog_meeting :
  let a := 1167
  let b := 2 ^ 41
  ∀ (f: ℕ → ℕ), ∀ (p: ℚ) (h: p = friend_probability),
    (cool_condition f) →
    (∃ a b, a / b = p ∧ a % gcd a b = 0 ∧ gcd a b = 1) ∧ a = 1167 :=
by
  sorry

end find_a_of_cool_frog_meeting_l1523_152382


namespace compute_F_2_f_3_l1523_152358

def f (a : ℝ) : ℝ := a^2 - 3 * a + 2
def F (a b : ℝ) : ℝ := b + a^3

theorem compute_F_2_f_3 : F 2 (f 3) = 10 :=
by
  sorry

end compute_F_2_f_3_l1523_152358


namespace charity_distribution_l1523_152370

theorem charity_distribution 
  (X : ℝ) (Y : ℝ) (Z : ℝ) (W : ℝ) (A : ℝ)
  (h1 : X > 0) (h2 : Y > 0) (h3 : Y < 100) (h4 : Z > 0) (h5 : W > 0) (h6 : A > 0)
  (h7 : W * A = X * (100 - Y) / 100) :
  (Y * X) / (100 * Z) = A * W * Y / (100 * Z) :=
by 
  sorry

end charity_distribution_l1523_152370


namespace field_day_difference_l1523_152305

theorem field_day_difference :
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  total_boys - total_girls = 2 :=
by
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  have h1 : total_girls = 46 := rfl
  have h2 : total_boys = 48 := rfl
  have h3 : total_boys - total_girls = 2 := rfl
  exact h3

end field_day_difference_l1523_152305


namespace tg_plus_ctg_l1523_152384

theorem tg_plus_ctg (x : ℝ) (h : 1 / Real.cos x - 1 / Real.sin x = Real.sqrt 15) :
  Real.tan x + (1 / Real.tan x) = -3 ∨ Real.tan x + (1 / Real.tan x) = 5 :=
sorry

end tg_plus_ctg_l1523_152384


namespace shirt_cost_l1523_152335

theorem shirt_cost
  (J S B : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 61)
  (h3 : 3 * J + 3 * S + 2 * B = 90) :
  S = 9 := 
by
  sorry

end shirt_cost_l1523_152335


namespace anne_speed_l1523_152357

-- Definition of distance and time
def distance : ℝ := 6
def time : ℝ := 3

-- Statement to prove
theorem anne_speed : distance / time = 2 := by
  sorry

end anne_speed_l1523_152357


namespace surface_area_of_parallelepiped_l1523_152303

open Real

theorem surface_area_of_parallelepiped 
  (a b c : ℝ)
  (x y z : ℝ)
  (h1: a^2 = x^2 + y^2)
  (h2: b^2 = x^2 + z^2)
  (h3: c^2 = y^2 + z^2) :
  2 * (sqrt ((x * y)) + sqrt ((x * z)) + sqrt ((y * z)))  =
  sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2)) +
  sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2)) +
  sqrt ((a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
by
  sorry

end surface_area_of_parallelepiped_l1523_152303


namespace isoperimetric_inequality_l1523_152344

theorem isoperimetric_inequality (S : ℝ) (P : ℝ) : S ≤ P^2 / (4 * Real.pi) :=
sorry

end isoperimetric_inequality_l1523_152344


namespace find_natural_numbers_l1523_152341

theorem find_natural_numbers (n : ℕ) : 
  (∃ d : ℕ, d ≤ 9 ∧ 10 * n + d = 13 * n) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by {
  sorry
}

end find_natural_numbers_l1523_152341


namespace johns_father_age_l1523_152377

variable {Age : Type} [OrderedRing Age]
variables (J M F : Age)

def john_age := J
def mother_age := M
def father_age := F

def john_younger_than_father (F J : Age) : Prop := F = 2 * J
def father_older_than_mother (F M : Age) : Prop := F = M + 4
def age_difference_between_john_and_mother (M J : Age) : Prop := M = J + 16

-- The question to be proved in Lean:
theorem johns_father_age :
  john_younger_than_father F J →
  father_older_than_mother F M →
  age_difference_between_john_and_mother M J →
  F = 40 := 
by
  intros h1 h2 h3
  sorry

end johns_father_age_l1523_152377


namespace carson_can_ride_giant_slide_exactly_twice_l1523_152363

noncomputable def Carson_Carnival : Prop := 
  let total_time_available := 240
  let roller_coaster_time := 30
  let tilt_a_whirl_time := 60
  let giant_slide_time := 15
  let vortex_time := 45
  let bumper_cars_time := 25
  let roller_coaster_rides := 4
  let tilt_a_whirl_rides := 2
  let vortex_rides := 1
  let bumper_cars_rides := 3

  let total_time_spent := 
    roller_coaster_time * roller_coaster_rides +
    tilt_a_whirl_time * tilt_a_whirl_rides +
    vortex_time * vortex_rides +
    bumper_cars_time * bumper_cars_rides

  total_time_available - (total_time_spent + giant_slide_time * 2) = 0

theorem carson_can_ride_giant_slide_exactly_twice : Carson_Carnival :=
by
  unfold Carson_Carnival
  sorry -- proof will be provided here

end carson_can_ride_giant_slide_exactly_twice_l1523_152363


namespace inscribed_circle_radius_l1523_152327

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 10
noncomputable def c : ℝ := 20

noncomputable def r : ℝ := 1 / (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))

theorem inscribed_circle_radius :
  r = 20 / (3.5 + 2 * Real.sqrt 14) :=
sorry

end inscribed_circle_radius_l1523_152327


namespace cars_meet_in_3_hours_l1523_152339

theorem cars_meet_in_3_hours
  (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (t : ℝ)
  (h_distance: distance = 333)
  (h_speed1: speed1 = 54)
  (h_speed2: speed2 = 57)
  (h_equation: speed1 * t + speed2 * t = distance) :
  t = 3 :=
sorry

end cars_meet_in_3_hours_l1523_152339


namespace extreme_value_point_of_f_l1523_152311

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume the definition of f that derives this f'

def f' (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem extreme_value_point_of_f : (∃ x : ℝ, x = -2 ∧ ∀ y : ℝ, y ≠ -2 → f' y < 0) := sorry

end extreme_value_point_of_f_l1523_152311


namespace ellipse_properties_l1523_152392

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def slopes_condition (x1 y1 x2 y2 : ℝ) (k_ab k_oa k_ob : ℝ) : Prop :=
  (k_ab^2 = k_oa * k_ob)

variables {x y : ℝ}

theorem ellipse_properties :
  (ellipse x y 2 1) ∧ -- Given ellipse equation
  (∃ (x1 y1 x2 y2 k_ab k_oa k_ob : ℝ), slopes_condition x1 y1 x2 y2 k_ab k_oa k_ob) →
  (∃ (OA OB : ℝ), OA^2 + OB^2 = 5) ∧ -- Prove sum of squares is constant
  (∃ (m : ℝ), (m = 1 → ∃ (line_eq : ℝ → ℝ), ∀ x, line_eq x = (1 / 2) * x + m)) -- Maximum area of triangle AOB

:= sorry

end ellipse_properties_l1523_152392


namespace triangle_area_hypotenuse_l1523_152308

-- Definitions of the conditions
def DE : ℝ := 40
def DF : ℝ := 30
def angleD : ℝ := 90

-- Proof statement
theorem triangle_area_hypotenuse :
  let Area : ℝ := 1 / 2 * DE * DF
  let EF : ℝ := Real.sqrt (DE^2 + DF^2)
  Area = 600 ∧ EF = 50 := by
  sorry

end triangle_area_hypotenuse_l1523_152308


namespace tan_a_over_tan_b_plus_tan_b_over_tan_a_l1523_152332

theorem tan_a_over_tan_b_plus_tan_b_over_tan_a {a b : ℝ} 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 44 / 5 :=
sorry

end tan_a_over_tan_b_plus_tan_b_over_tan_a_l1523_152332


namespace part1_part2_l1523_152325

open Real

variable (A B C a b c : ℝ)

-- Conditions
variable (h1 : b * sin A = a * cos B)
variable (h2 : b = 3)
variable (h3 : sin C = 2 * sin A)

theorem part1 : B = π / 4 := 
  sorry

theorem part2 : ∃ a c, c = 2 * a ∧ 9 = a^2 + c^2 - 2 * a * c * cos (π / 4) := 
  sorry

end part1_part2_l1523_152325


namespace arithmetic_sequence_100_l1523_152353

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (S₉ : ℝ) (a₁₀ : ℝ)

theorem arithmetic_sequence_100
  (h1: is_arithmetic_sequence a)
  (h2: S₉ = 27) 
  (h3: a₁₀ = 8): 
  a 100 = 98 := 
sorry

end arithmetic_sequence_100_l1523_152353


namespace find_ab_l1523_152322

variable (a b : ℝ)

theorem find_ab (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 := by
  sorry

end find_ab_l1523_152322


namespace calculate_chord_length_l1523_152366

noncomputable def chord_length_of_tangent (r1 r2 : ℝ) (c : ℝ) : Prop :=
  r1^2 - r2^2 = 18 ∧ (c / 2)^2 = 18

theorem calculate_chord_length (r1 r2 : ℝ) (h : chord_length_of_tangent r1 r2 (6 * Real.sqrt 2)) :
  (6 * Real.sqrt 2) = 6 * Real.sqrt 2 :=
by
  sorry

end calculate_chord_length_l1523_152366


namespace value_of_ab_over_cd_l1523_152362

theorem value_of_ab_over_cd (a b c d : ℚ) (h₁ : a / b = 2 / 3) (h₂ : c / b = 1 / 5) (h₃ : c / d = 7 / 15) : (a * b) / (c * d) = 140 / 9 :=
by
  sorry

end value_of_ab_over_cd_l1523_152362


namespace equivalent_annual_rate_l1523_152315

def quarterly_to_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4 - 1

def to_percentage (rate : ℝ) : ℝ :=
  rate * 100

theorem equivalent_annual_rate (quarterly_rate : ℝ) (annual_rate : ℝ) :
  quarterly_rate = 0.02 →
  annual_rate = quarterly_to_annual_rate quarterly_rate →
  to_percentage annual_rate = 8.24 :=
by
  intros
  sorry

end equivalent_annual_rate_l1523_152315


namespace circle_radius_zero_l1523_152365

theorem circle_radius_zero (x y : ℝ) :
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → ∃ c : ℝ × ℝ, ∃ r : ℝ, (x - c.1)^2 + (y - c.2)^2 = r^2 ∧ r = 0 :=
by
  sorry

end circle_radius_zero_l1523_152365


namespace largest_share_of_partner_l1523_152385

theorem largest_share_of_partner 
    (ratios : List ℕ := [2, 3, 4, 4, 6])
    (total_profit : ℕ := 38000) :
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    largest_share = 12000 :=
by
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    have h1 : total_parts = 19 := by
        sorry
    have h2 : part_value = 2000 := by
        sorry
    have h3 : List.maximum ratios = 6 := by
        sorry
    have h4 : largest_share = 12000 := by
        sorry
    exact h4


end largest_share_of_partner_l1523_152385


namespace arithmetic_sequence_common_difference_l1523_152300

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ)
  (hS2 : S 2 = 4) (hS4 : S 4 = 20)
  (hS_formula : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)) : 
  d = 3 :=
by sorry

end arithmetic_sequence_common_difference_l1523_152300


namespace boxes_given_to_brother_l1523_152390

-- Definitions
def total_boxes : ℝ := 14.0
def pieces_per_box : ℝ := 6.0
def pieces_remaining : ℝ := 42.0

-- Theorem stating the problem
theorem boxes_given_to_brother : 
  (total_boxes * pieces_per_box - pieces_remaining) / pieces_per_box = 7.0 := 
by
  sorry

end boxes_given_to_brother_l1523_152390


namespace union_sets_l1523_152334

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

theorem union_sets :
  M ∪ N = {x | x ≤ 1} :=
by
  sorry

end union_sets_l1523_152334


namespace find_a_plus_b_l1523_152338

def smallest_two_digit_multiple_of_five : ℕ := 10
def smallest_three_digit_multiple_of_seven : ℕ := 105

theorem find_a_plus_b :
  let a := smallest_two_digit_multiple_of_five
  let b := smallest_three_digit_multiple_of_seven
  a + b = 115 := by
  sorry

end find_a_plus_b_l1523_152338


namespace sequence_a6_value_l1523_152314

theorem sequence_a6_value :
  ∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n, a (n + 1) = a n / (2 * a n + 1)) ∧ (a 6 = 1 / 11) :=
by
  sorry

end sequence_a6_value_l1523_152314


namespace fraction_integer_solution_l1523_152347

theorem fraction_integer_solution (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 8) (h₃ : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = -1 := 
sorry

end fraction_integer_solution_l1523_152347


namespace round_trip_time_l1523_152398

theorem round_trip_time (current_speed : ℝ) (boat_speed_still : ℝ) (distance_upstream : ℝ) (total_time : ℝ) :
  current_speed = 4 → 
  boat_speed_still = 18 → 
  distance_upstream = 85.56 →
  total_time = 10 :=
by
  intros h_current h_boat h_distance
  sorry

end round_trip_time_l1523_152398


namespace eval_g_inv_g_inv_14_l1523_152310

variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)

axiom g_def : ∀ x, g x = 3 * x - 4
axiom g_inv_def : ∀ y, g_inv y = (y + 4) / 3

theorem eval_g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by
    sorry

end eval_g_inv_g_inv_14_l1523_152310


namespace brett_red_marbles_l1523_152331

variables (r b : ℕ)

-- Define the conditions
axiom h1 : b = r + 24
axiom h2 : b = 5 * r

theorem brett_red_marbles : r = 6 :=
by
  sorry

end brett_red_marbles_l1523_152331


namespace value_of_expression_l1523_152364

theorem value_of_expression : 8 * (6 - 4) + 2 = 18 := by
  sorry

end value_of_expression_l1523_152364


namespace boys_test_l1523_152306

-- Define the conditions
def passing_time : ℝ := 14
def test_results : List ℝ := [0.6, -1.1, 0, -0.2, 2, 0.5]

-- Define the proof problem
theorem boys_test (number_did_not_pass : ℕ) (fastest_time : ℝ) (average_score : ℝ) :
  passing_time = 14 →
  test_results = [0.6, -1.1, 0, -0.2, 2, 0.5] →
  number_did_not_pass = 3 ∧
  fastest_time = 12.9 ∧
  average_score = 14.3 :=
by
  intros
  sorry

end boys_test_l1523_152306


namespace find_m_l1523_152346

theorem find_m (m : ℝ) (P : Set ℝ) (Q : Set ℝ) (hP : P = {m^2 - 4, m + 1, -3})
  (hQ : Q = {m - 3, 2 * m - 1, 3 * m + 1}) (h_intersect : P ∩ Q = {-3}) :
  m = -4 / 3 :=
by
  sorry

end find_m_l1523_152346
