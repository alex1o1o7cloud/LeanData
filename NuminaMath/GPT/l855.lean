import Mathlib

namespace integer_cube_less_than_triple_unique_l855_85596

theorem integer_cube_less_than_triple_unique (x : ℤ) (h : x^3 < 3*x) : x = 1 :=
sorry

end integer_cube_less_than_triple_unique_l855_85596


namespace simplify_frac_op_l855_85573

-- Definition of the operation *
def frac_op (a b c d : ℚ) : ℚ := (a * c) * (d / (b + 1))

-- Proof problem stating the specific operation result
theorem simplify_frac_op :
  frac_op 5 11 9 4 = 15 :=
by
  sorry

end simplify_frac_op_l855_85573


namespace simplify_polynomial_problem_l855_85505

theorem simplify_polynomial_problem (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) = 2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := 
by
  sorry

end simplify_polynomial_problem_l855_85505


namespace tax_rate_computation_l855_85563

-- Define the inputs
def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 134.4

-- Define the derived taxable amount
def taxable_amount : ℝ := total_value - non_taxable_amount

-- Define the expected tax rate
def expected_tax_rate : ℝ := 0.12

-- State the theorem
theorem tax_rate_computation : 
  (tax_paid / taxable_amount * 100) = expected_tax_rate * 100 := 
by
  sorry

end tax_rate_computation_l855_85563


namespace no_three_digit_numbers_meet_conditions_l855_85549

theorem no_three_digit_numbers_meet_conditions :
  ∀ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (n % 10 = 5) ∧ (n % 10 = 0) → false := 
by {
  sorry
}

end no_three_digit_numbers_meet_conditions_l855_85549


namespace total_rainfall_in_january_l855_85577

theorem total_rainfall_in_january 
  (r1 r2 : ℝ)
  (h1 : r2 = 1.5 * r1)
  (h2 : r2 = 18) : 
  r1 + r2 = 30 := by
  sorry

end total_rainfall_in_january_l855_85577


namespace M_equals_N_l855_85507

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {y | 0 ≤ y}

-- State the main proof goal
theorem M_equals_N : M = N :=
by
  sorry

end M_equals_N_l855_85507


namespace triangle_side_ratio_l855_85512

variables (a b c S : ℝ)
variables (A B C : ℝ)

/-- In triangle ABC, if the sides opposite to angles A, B, and C are a, b, and c respectively,
    and given a=1, B=π/4, and the area S=2, we prove that b / sin(B) = 5√2. -/
theorem triangle_side_ratio (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : b / Real.sin B = 5 * Real.sqrt 2 :=
sorry

end triangle_side_ratio_l855_85512


namespace correct_transformation_l855_85591

theorem correct_transformation (x : ℝ) : x^2 - 10 * x - 1 = 0 → (x - 5)^2 = 26 :=
  sorry

end correct_transformation_l855_85591


namespace alcohol_water_ratio_l855_85536

theorem alcohol_water_ratio (a b : ℚ) (h₁ : a = 3/5) (h₂ : b = 2/5) : a / b = 3 / 2 :=
by
  sorry

end alcohol_water_ratio_l855_85536


namespace paper_area_difference_l855_85503

def sheet1_length : ℕ := 14
def sheet1_width : ℕ := 12
def sheet2_length : ℕ := 9
def sheet2_width : ℕ := 14

def area (length : ℕ) (width : ℕ) : ℕ := length * width

def combined_area (length : ℕ) (width : ℕ) : ℕ := 2 * area length width

theorem paper_area_difference :
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 84 := 
by 
  sorry

end paper_area_difference_l855_85503


namespace range_of_a_l855_85570

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 1 = 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + a > 0

theorem range_of_a (a : ℝ) (hp : proposition_p a) (hq : proposition_q a) : a ≥ 2 :=
sorry

end range_of_a_l855_85570


namespace find_x2_l855_85545

theorem find_x2 (x1 x2 x3 : ℝ) (h1 : x1 + x2 = 14) (h2 : x1 + x3 = 17) (h3 : x2 + x3 = 33) : x2 = 15 :=
by
  sorry

end find_x2_l855_85545


namespace abs_fraction_eq_sqrt_three_over_two_l855_85533

variable (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b)

theorem abs_fraction_eq_sqrt_three_over_two (h : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 10 * a * b) : 
  |(a + b) / (a - b)| = Real.sqrt (3 / 2) := by
  sorry

end abs_fraction_eq_sqrt_three_over_two_l855_85533


namespace minimum_omega_l855_85517

theorem minimum_omega (ω : ℕ) (h_pos : ω ∈ {n : ℕ | n > 0}) (h_cos_center : ∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + π / 2) :
  ω = 2 :=
by { sorry }

end minimum_omega_l855_85517


namespace binary_op_property_l855_85510

variable (X : Type)
variable (star : X → X → X)
variable (h : ∀ x y : X, star (star x y) x = y)

theorem binary_op_property (x y : X) : star x (star y x) = y := 
by 
  sorry

end binary_op_property_l855_85510


namespace sum_of_prime_factors_is_prime_l855_85538

/-- Define the specific number in question -/
def num := 30030

/-- List the prime factors of the number -/
def prime_factors := [2, 3, 5, 7, 11, 13]

/-- Sum of the prime factors -/
def sum_prime_factors := prime_factors.sum

theorem sum_of_prime_factors_is_prime :
  sum_prime_factors = 41 ∧ Prime 41 := 
by
  -- The conditions are encapsulated in the definitions above
  -- Now, establish the required proof goal using these conditions
  sorry

end sum_of_prime_factors_is_prime_l855_85538


namespace arrow_reading_l855_85588

-- Define the interval and values within it
def in_range (x : ℝ) : Prop := 9.75 ≤ x ∧ x ≤ 10.00
def closer_to_990 (x : ℝ) : Prop := |x - 9.90| < |x - 9.875|

-- The main theorem statement expressing the problem
theorem arrow_reading (x : ℝ) (hx1 : in_range x) (hx2 : closer_to_990 x) : x = 9.90 :=
by sorry

end arrow_reading_l855_85588


namespace smallest_n_l855_85558

theorem smallest_n (n : ℕ) : (n > 0) ∧ (2^n % 30 = 1) → n = 4 :=
by
  intro h
  sorry

end smallest_n_l855_85558


namespace haley_money_l855_85550

variable (x : ℕ)

def initial_amount : ℕ := 2
def difference : ℕ := 11
def total_amount (x : ℕ) : ℕ := x

theorem haley_money : total_amount x - initial_amount = difference → total_amount x = 13 := by
  sorry

end haley_money_l855_85550


namespace option_C_is_correct_l855_85559

theorem option_C_is_correct (a b c : ℝ) (h : a > b) : c - a < c - b := 
by
  linarith

end option_C_is_correct_l855_85559


namespace how_many_correct_l855_85501

def calc1 := (2 * Real.sqrt 3) * (3 * Real.sqrt 3) = 6 * Real.sqrt 3
def calc2 := Real.sqrt 2 + Real.sqrt 3 = Real.sqrt 5
def calc3 := (5 * Real.sqrt 5) - (2 * Real.sqrt 2) = 3 * Real.sqrt 3
def calc4 := (Real.sqrt 2) / (Real.sqrt 3) = (Real.sqrt 6) / 3

theorem how_many_correct : (¬ calc1) ∧ (¬ calc2) ∧ (¬ calc3) ∧ calc4 → 1 = 1 :=
by { sorry }

end how_many_correct_l855_85501


namespace multiplication_mistake_l855_85568

theorem multiplication_mistake (x : ℕ) (H : 43 * x - 34 * x = 1215) : x = 135 :=
sorry

end multiplication_mistake_l855_85568


namespace distance_home_to_school_l855_85506

theorem distance_home_to_school :
  ∃ T D : ℝ, 6 * (T + 7/60) = D ∧ 12 * (T - 8/60) = D ∧ 9 * T = D ∧ D = 2.1 :=
by
  sorry

end distance_home_to_school_l855_85506


namespace passes_through_point_P_l855_85574

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 7 + a^(x - 1)

theorem passes_through_point_P
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : f a 1 = 8 :=
by
  -- Proof omitted
  sorry

end passes_through_point_P_l855_85574


namespace trigonometric_values_l855_85584

theorem trigonometric_values (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1 / 3) 
  (h2 : Real.cos x - Real.cos y = 1 / 5) : 
  Real.cos (x + y) = 208 / 225 ∧ Real.sin (x - y) = -15 / 17 := 
by 
  sorry

end trigonometric_values_l855_85584


namespace sandwich_cost_90_cents_l855_85513

def sandwich_cost (bread_cost ham_cost cheese_cost : ℕ) : ℕ :=
  2 * bread_cost + ham_cost + cheese_cost

theorem sandwich_cost_90_cents :
  sandwich_cost 15 25 35 = 90 :=
by
  -- Proof goes here
  sorry

end sandwich_cost_90_cents_l855_85513


namespace cost_of_marker_l855_85524

theorem cost_of_marker (n m : ℝ) (h1 : 3 * n + 2 * m = 7.45) (h2 : 4 * n + 3 * m = 10.40) : m = 1.40 :=
  sorry

end cost_of_marker_l855_85524


namespace ramu_repairs_cost_l855_85560

theorem ramu_repairs_cost :
  ∃ R : ℝ, 64900 - (42000 + R) = (29.8 / 100) * (42000 + R) :=
by
  use 8006.16
  sorry

end ramu_repairs_cost_l855_85560


namespace perpendicular_lines_a_value_l855_85542

theorem perpendicular_lines_a_value :
  ∀ a : ℝ, 
    (∀ x y : ℝ, 2*x + a*y - 7 = 0) → 
    (∀ x y : ℝ, (a-3)*x + y + 4 = 0) → a = 2 :=
by
  sorry

end perpendicular_lines_a_value_l855_85542


namespace earnings_difference_l855_85511

theorem earnings_difference (x y : ℕ) 
  (h1 : 3 * 6 + 4 * 5 + 5 * 4 = 58)
  (h2 : x * y = 12500) 
  (total_earnings : (3 * 6 * x * y / 100 + 4 * 5 * x * y / 100 + 5 * 4 * x * y / 100) = 7250) :
  4 * 5 * x * y / 100 - 3 * 6 * x * y / 100 = 250 := 
by 
  sorry

end earnings_difference_l855_85511


namespace fraction_evaporated_l855_85589

theorem fraction_evaporated (x : ℝ) (h : (1 - x) * (1/4) = 1/6) : x = 1/3 :=
by
  sorry

end fraction_evaporated_l855_85589


namespace polynomial_subtraction_simplify_l855_85578

open Polynomial

noncomputable def p : Polynomial ℚ := 3 * X^2 + 9 * X - 5
noncomputable def q : Polynomial ℚ := 2 * X^2 + 3 * X - 10
noncomputable def result : Polynomial ℚ := X^2 + 6 * X + 5

theorem polynomial_subtraction_simplify : 
  p - q = result :=
by
  sorry

end polynomial_subtraction_simplify_l855_85578


namespace xiaohongs_mother_deposit_l855_85525

theorem xiaohongs_mother_deposit (x : ℝ) :
  x + x * 3.69 / 100 * 3 * (1 - 20 / 100) = 5442.8 :=
by
  sorry

end xiaohongs_mother_deposit_l855_85525


namespace find_pyramid_volume_l855_85595

noncomputable def volume_of_pyramid (α β R : ℝ) : ℝ :=
  (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β

theorem find_pyramid_volume (α β R : ℝ) 
  (base_isosceles : ∀ {a b c : ℝ}, a = b) -- Represents the isosceles triangle condition
  (dihedral_angles_equal : ∀ {angle : ℝ}, angle = β) -- Dihedral angle at the base
  (circumcircle_radius : {radius : ℝ // radius = R}) -- Radius of the circumcircle
  (height_through_point : true) -- Condition: height passes through a point inside the triangle
  :
  volume_of_pyramid α β R = (1/3) * R^3 * (Real.sin α)^2 * Real.cos (α/2) * Real.tan (Real.pi / 4 - α/2) * Real.tan β :=
by {
  sorry
}

end find_pyramid_volume_l855_85595


namespace age_ratio_l855_85522

theorem age_ratio (S : ℕ) (M : ℕ) (h1 : S = 28) (h2 : M = S + 30) : 
  ((M + 2) / (S + 2) = 2) := 
by
  sorry

end age_ratio_l855_85522


namespace min_possible_A_div_C_l855_85556

theorem min_possible_A_div_C (x : ℝ) (A C : ℝ) (h1 : x^2 + (1/x)^2 = A) (h2 : x + 1/x = C) (h3 : 0 < A) (h4 : 0 < C) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ (∀ B, (x^2 + (1/x)^2 = B) → (x + 1/x = C) → (B / C = 0 → B = 0)) :=
by
  sorry

end min_possible_A_div_C_l855_85556


namespace sequence_periodic_mod_l855_85539

-- Define the sequence (u_n) recursively
def sequence_u (a : ℕ) : ℕ → ℕ
  | 0     => a  -- Note: u_1 is defined as the initial term a, treating the starting index as 0 for compatibility with Lean's indexing.
  | (n+1) => a ^ (sequence_u a n)

-- The theorem stating there exist integers k and N such that for all n ≥ N, u_{n+k} ≡ u_n (mod m)
theorem sequence_periodic_mod (a m : ℕ) (hm : 0 < m) (ha : 0 < a) :
  ∃ k N : ℕ, ∀ n : ℕ, N ≤ n → (sequence_u a (n + k) ≡ sequence_u a n [MOD m]) :=
by
  sorry

end sequence_periodic_mod_l855_85539


namespace number_of_faces_of_prism_proof_l855_85592

noncomputable def number_of_faces_of_prism (n : ℕ) : ℕ := 2 + n

theorem number_of_faces_of_prism_proof (n : ℕ) (E_p E_py : ℕ) (h1 : E_p + E_py = 30) (h2 : E_p = 3 * n) (h3 : E_py = 2 * n) :
  number_of_faces_of_prism n = 8 :=
by
  sorry

end number_of_faces_of_prism_proof_l855_85592


namespace cistern_fill_time_l855_85576

theorem cistern_fill_time
  (T : ℝ)
  (H1 : 0 < T)
  (rate_first_tap : ℝ := 1 / T)
  (rate_second_tap : ℝ := 1 / 6)
  (net_rate : ℝ := 1 / 12)
  (H2 : rate_first_tap - rate_second_tap = net_rate) :
  T = 4 :=
sorry

end cistern_fill_time_l855_85576


namespace triangle_ABC_area_l855_85580

-- Define the vertices of the triangle
def A := (-4, 0)
def B := (24, 0)
def C := (0, 2)

-- Function to calculate the determinant, used for the area calculation
def det (x1 y1 x2 y2 x3 y3 : ℝ) :=
  x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

-- Area calculation for triangle given vertices using determinant method
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * |det x1 y1 x2 y2 x3 y3|

-- The goal is to prove that the area of triangle ABC is 14
theorem triangle_ABC_area :
  triangle_area (-4) 0 24 0 0 2 = 14 := sorry

end triangle_ABC_area_l855_85580


namespace all_values_equal_l855_85535

noncomputable def f : ℤ × ℤ → ℕ :=
sorry

theorem all_values_equal (f : ℤ × ℤ → ℕ)
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ x y, f (x, y) = 1/4 * (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1))) :
  ∀ (x1 y1 x2 y2 : ℤ), f (x1, y1) = f (x2, y2) := 
sorry

end all_values_equal_l855_85535


namespace base_seven_to_ten_l855_85587

theorem base_seven_to_ten : 
  (7 * 7^4 + 6 * 7^3 + 5 * 7^2 + 4 * 7^1 + 3 * 7^0) = 19141 := 
by 
  sorry

end base_seven_to_ten_l855_85587


namespace circle_center_eq_l855_85590

theorem circle_center_eq (x y : ℝ) :
    (x^2 + y^2 - 2*x + y + 1/4 = 0) → (x = 1 ∧ y = -1/2) :=
by
  sorry

end circle_center_eq_l855_85590


namespace solution_of_inequality_l855_85518

theorem solution_of_inequality (a b : ℝ) (h : ∀ x : ℝ, (1 < x ∧ x < 3) ↔ (x^2 < a * x + b)) :
  b^a = 81 := 
sorry

end solution_of_inequality_l855_85518


namespace factorial_less_power_l855_85598

open Nat

noncomputable def factorial_200 : ℕ := 200!

noncomputable def power_100_200 : ℕ := 100 ^ 200

theorem factorial_less_power : factorial_200 < power_100_200 :=
by
  -- Proof goes here
  sorry

end factorial_less_power_l855_85598


namespace angle_A_measure_l855_85553

theorem angle_A_measure (A B C D E : ℝ) 
(h1 : A = 3 * B)
(h2 : A = 4 * C)
(h3 : A = 5 * D)
(h4 : A = 6 * E)
(h5 : A + B + C + D + E = 540) : 
A = 277 :=
by
  sorry

end angle_A_measure_l855_85553


namespace packs_needed_l855_85572

def pouches_per_pack : ℕ := 6
def team_members : ℕ := 13
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people : ℕ := team_members + coaches + helpers

theorem packs_needed (people : ℕ) (pouches_per_pack : ℕ) : ℕ :=
  (people + pouches_per_pack - 1) / pouches_per_pack

example : packs_needed total_people pouches_per_pack = 3 :=
by
  have h1 : total_people = 18 := rfl
  have h2 : pouches_per_pack = 6 := rfl
  rw [h1, h2]
  norm_num
  sorry

end packs_needed_l855_85572


namespace α_plus_β_eq_two_l855_85552

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

theorem α_plus_β_eq_two
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 := 
sorry

end α_plus_β_eq_two_l855_85552


namespace angle_of_elevation_proof_l855_85516

noncomputable def height_of_lighthouse : ℝ := 100

noncomputable def distance_between_ships : ℝ := 273.2050807568877

noncomputable def angle_of_elevation_second_ship : ℝ := 45

noncomputable def distance_from_second_ship := height_of_lighthouse

noncomputable def distance_from_first_ship := distance_between_ships - distance_from_second_ship

noncomputable def tanθ := height_of_lighthouse / distance_from_first_ship

noncomputable def angle_of_elevation_first_ship := Real.arctan tanθ

theorem angle_of_elevation_proof :
  angle_of_elevation_first_ship = 30 := by
    sorry

end angle_of_elevation_proof_l855_85516


namespace solve_linear_system_l855_85581

theorem solve_linear_system (x y a : ℝ) (h1 : 4 * x + 3 * y = 1) (h2 : a * x + (a - 1) * y = 3) (hxy : x = y) : a = 11 :=
by
  sorry

end solve_linear_system_l855_85581


namespace volume_calculation_l855_85543

-- Define the dimensions of the rectangular parallelepiped
def a : ℕ := 2
def b : ℕ := 3
def c : ℕ := 4

-- Define the radius for spheres and cylinders
def r : ℝ := 2

theorem volume_calculation : 
  let l := 384
  let o := 140
  let q := 3
  (l + o + q = 527) :=
by
  sorry

end volume_calculation_l855_85543


namespace parabola_focus_l855_85555

theorem parabola_focus (x : ℝ) : ∃ f : ℝ × ℝ, f = (0, 1 / 4) ∧ ∀ y : ℝ, y = x^2 → f = (0, 1 / 4) :=
by
  sorry

end parabola_focus_l855_85555


namespace increase_in_area_l855_85585

theorem increase_in_area (a : ℝ) : 
  let original_radius := 3
  let new_radius := original_radius + a
  let original_area := π * original_radius ^ 2
  let new_area := π * new_radius ^ 2
  new_area - original_area = π * (3 + a) ^ 2 - 9 * π := 
by
  sorry

end increase_in_area_l855_85585


namespace car_meeting_points_l855_85502

-- Define the conditions for the problem
variables {A B : ℝ}
variables {speed_ratio : ℝ} (ratio_pos : speed_ratio = 5 / 4)
variables {T1 T2 : ℝ} (T1_pos : T1 = 145) (T2_pos : T2 = 201)

-- The proof problem statement
theorem car_meeting_points (A B : ℝ) (ratio_pos : speed_ratio = 5 / 4) 
  (T1 T2 : ℝ) (T1_pos : T1 = 145) (T2_pos : T2 = 201) :
  A = 103 ∧ B = 229 :=
sorry

end car_meeting_points_l855_85502


namespace determine_a7_l855_85532

noncomputable def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => a1
| (n+1) => a1 + n * d

noncomputable def sum_arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a1 * (n + 1) + (n * (n + 1) * d) / 2

theorem determine_a7 (a1 d : ℤ) (a2 : a1 + d = 7) (S7 : sum_arithmetic_seq a1 d 7 = -7) : arithmetic_seq a1 d 7 = -13 :=
by
  sorry

end determine_a7_l855_85532


namespace total_cost_of_purchase_l855_85514

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase_l855_85514


namespace shopkeeper_weight_l855_85530

/-- A shopkeeper sells his goods at cost price but uses a certain weight instead of kilogram weight.
    His profit percentage is 25%. Prove that the weight he uses is 0.8 kilograms. -/
theorem shopkeeper_weight (c s p : ℝ) (x : ℝ) (h1 : s = c * (1 + p / 100))
  (h2 : p = 25) (h3 : c = 1) (h4 : s = 1.25) : x = 0.8 :=
by
  sorry

end shopkeeper_weight_l855_85530


namespace students_playing_long_tennis_l855_85504

theorem students_playing_long_tennis (n F B N L : ℕ)
  (h1 : n = 35)
  (h2 : F = 26)
  (h3 : B = 17)
  (h4 : N = 6)
  (h5 : L = (n - N) - (F - B)) :
  L = 20 :=
by
  sorry

end students_playing_long_tennis_l855_85504


namespace MutualExclusivity_Of_A_C_l855_85526

-- Definitions of events using conditions from a)
def EventA (products : List Bool) : Prop :=
  products.all (λ p => p = true)

def EventB (products : List Bool) : Prop :=
  products.all (λ p => p = false)

def EventC (products : List Bool) : Prop :=
  products.any (λ p => p = false)

-- The main theorem using correct answer from b)
theorem MutualExclusivity_Of_A_C (products : List Bool) :
  EventA products → ¬ EventC products :=
by
  sorry

end MutualExclusivity_Of_A_C_l855_85526


namespace customer_initial_amount_l855_85537

theorem customer_initial_amount (d c : ℕ) (h1 : c = 100 * d) (h2 : c = 2 * d) : d = 0 ∧ c = 0 := by
  sorry

end customer_initial_amount_l855_85537


namespace nearest_integer_to_expr_l855_85569

theorem nearest_integer_to_expr : 
  let a := 3 + Real.sqrt 5
  let b := (a)^6
  abs (b - 2744) < 1
:= sorry

end nearest_integer_to_expr_l855_85569


namespace sequence_geometric_and_sum_l855_85528

variables {S : ℕ → ℝ} (a1 : S 1 = 1)
variable (n : ℕ)
def a := (S (n+1) - 2 * S n, S n)
def b := (2, n)

/-- Prove that the sequence {S n / n} is a geometric sequence 
with first term 1 and common ratio 2, and find the sum of the first 
n terms of the sequence {S n} -/
theorem sequence_geometric_and_sum {S : ℕ → ℝ} (a1 : S 1 = 1)
  (n : ℕ)
  (parallel : ∀ n, n * (S (n + 1) - 2 * S n) = 2 * S n) :
  ∃ r : ℝ, r = 2 ∧ ∃ T : ℕ → ℝ, T n = (n-1)*2^n + 1 :=
by
  sorry

end sequence_geometric_and_sum_l855_85528


namespace sarahs_monthly_fee_l855_85520

noncomputable def fixed_monthly_fee (x y : ℝ) : Prop :=
  x + 4 * y = 30.72 ∧ 1.1 * x + 8 * y = 54.72

theorem sarahs_monthly_fee : ∃ x y : ℝ, fixed_monthly_fee x y ∧ x = 7.47 :=
by
  sorry

end sarahs_monthly_fee_l855_85520


namespace spherical_to_rectangular_correct_l855_85554

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_correct : spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) :=
by
  sorry

end spherical_to_rectangular_correct_l855_85554


namespace solve_inequality_l855_85515

theorem solve_inequality (x : ℝ) : 
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2 / 3 ∨ x > 1) := 
by
  sorry

end solve_inequality_l855_85515


namespace juan_speed_l855_85567

theorem juan_speed (J : ℝ) :
  (∀ (time : ℝ) (distance : ℝ) (peter_speed : ℝ),
    time = 1.5 →
    distance = 19.5 →
    peter_speed = 5 →
    distance = J * time + peter_speed * time) →
  J = 8 :=
by
  intro h
  sorry

end juan_speed_l855_85567


namespace difference_between_possible_values_of_x_l855_85547

noncomputable def difference_of_roots (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ℝ :=
  let sol1 := 11  -- First root
  let sol2 := -11 -- Second root
  sol1 - sol2

theorem difference_between_possible_values_of_x (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) :
  difference_of_roots x h = 22 :=
sorry

end difference_between_possible_values_of_x_l855_85547


namespace first_term_geometric_series_l855_85575

variable (a : ℝ)
variable (r : ℝ := 1/4)
variable (S : ℝ := 80)

theorem first_term_geometric_series 
  (h1 : r = 1/4) 
  (h2 : S = 80)
  : a = 60 :=
by 
  sorry

end first_term_geometric_series_l855_85575


namespace lily_typing_break_time_l855_85548

theorem lily_typing_break_time :
  ∃ t : ℝ, (15 * t + 15 * t = 255) ∧ (19 = 2 * t + 2) ∧ (t = 8) := 
sorry

end lily_typing_break_time_l855_85548


namespace non_arithmetic_sequence_l855_85561

theorem non_arithmetic_sequence (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) :
    (∀ n, S_n n = n^2 + 2 * n - 1) →
    (∀ n, a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1)) →
    ¬(∀ d, ∀ n, a_n (n+1) = a_n n + d) :=
by
  intros hS ha
  sorry

end non_arithmetic_sequence_l855_85561


namespace num_int_values_N_l855_85527

theorem num_int_values_N (N : ℕ) : 
  (∃ M, M ∣ 72 ∧ M > 3 ∧ N = M - 3) ↔ N ∈ ({1, 3, 5, 6, 9, 15, 21, 33, 69} : Finset ℕ) :=
by
  sorry

end num_int_values_N_l855_85527


namespace digitalEarthFunctions_l855_85586

axiom OptionA (F : Type) : Prop
axiom OptionB (F : Type) : Prop
axiom OptionC (F : Type) : Prop
axiom OptionD (F : Type) : Prop

axiom isRemoteSensing (F : Type) : OptionA F
axiom isGIS (F : Type) : OptionB F
axiom isGPS (F : Type) : OptionD F

theorem digitalEarthFunctions {F : Type} : OptionC F :=
sorry

end digitalEarthFunctions_l855_85586


namespace group_elements_eq_one_l855_85579
-- Import the entire math library

-- Define the main theorem
theorem group_elements_eq_one 
  {G : Type*} [Group G] 
  (a b : G) 
  (h1 : a * b^2 = b^3 * a) 
  (h2 : b * a^2 = a^3 * b) : 
  a = 1 ∧ b = 1 := 
  by 
  sorry

end group_elements_eq_one_l855_85579


namespace find_k_l855_85582

noncomputable def parabola_k : ℝ := 4

theorem find_k (k : ℝ) (h1 : ∀ x, y = k^2 - x^2) (h2 : k > 0)
    (h3 : ∀ A D : (ℝ × ℝ), A = (-k, 0) ∧ D = (k, 0))
    (h4 : ∀ V : (ℝ × ℝ), V = (0, k^2))
    (h5 : 2 * (2 * k + k^2) = 48) : k = 4 :=
  sorry

end find_k_l855_85582


namespace kim_pairs_of_shoes_l855_85544

theorem kim_pairs_of_shoes : ∃ n : ℕ, 2 * n + 1 = 14 ∧ (1 : ℚ) / (2 * n - 1) = (0.07692307692307693 : ℚ) :=
by
  sorry

end kim_pairs_of_shoes_l855_85544


namespace focus_of_parabola_l855_85529

-- Problem statement
theorem focus_of_parabola (x y : ℝ) : (2 * x^2 = -y) → (focus_coordinates = (0, -1 / 8)) :=
by
  sorry

end focus_of_parabola_l855_85529


namespace satisfy_conditions_l855_85531

variable (x : ℝ)

theorem satisfy_conditions :
  (3 * x^2 + 4 * x - 9 < 0) ∧ (x ≥ -2) ↔ (-2 ≤ x ∧ x < 1) := by
  sorry

end satisfy_conditions_l855_85531


namespace last_three_digits_of_5_power_odd_l855_85565

theorem last_three_digits_of_5_power_odd (n : ℕ) (h : n % 2 = 1) : (5 ^ n) % 1000 = 125 :=
sorry

end last_three_digits_of_5_power_odd_l855_85565


namespace cover_rectangle_with_polyomino_l855_85519

-- Defining the conditions under which the m x n rectangle can be covered by the given polyomino
theorem cover_rectangle_with_polyomino (m n : ℕ) :
  (6 ∣ (m * n)) →
  (m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) →
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) →
  ((3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ (m * n))) :=
sorry

end cover_rectangle_with_polyomino_l855_85519


namespace number_of_rabbits_l855_85551

theorem number_of_rabbits (C D : ℕ) (hC : C = 49) (hD : D = 37) (h : D + R = C + 9) :
  R = 21 :=
by
    sorry

end number_of_rabbits_l855_85551


namespace recurring_decimal_fraction_l855_85597

theorem recurring_decimal_fraction (h54 : (0.54 : ℝ) = 54 / 99) (h18 : (0.18 : ℝ) = 18 / 99) :
    (0.54 / 0.18 : ℝ) = 3 := 
by
  sorry

end recurring_decimal_fraction_l855_85597


namespace books_sold_l855_85508

theorem books_sold (original_books : ℕ) (remaining_books : ℕ) (sold_books : ℕ) 
  (h1 : original_books = 51) 
  (h2 : remaining_books = 6) 
  (h3 : sold_books = original_books - remaining_books) : 
  sold_books = 45 :=
by 
  sorry

end books_sold_l855_85508


namespace tens_of_80_tens_of_190_l855_85594

def tens_place (n : Nat) : Nat :=
  (n / 10) % 10

theorem tens_of_80 : tens_place 80 = 8 := 
  by
  sorry

theorem tens_of_190 : tens_place 190 = 9 := 
  by
  sorry

end tens_of_80_tens_of_190_l855_85594


namespace even_function_a_eq_neg1_l855_85583

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * (Real.exp x + a * Real.exp (-x))

/-- Given that the function f(x) = x(e^x + a e^{-x}) is an even function, prove that a = -1. -/
theorem even_function_a_eq_neg1 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : a = -1 :=
sorry

end even_function_a_eq_neg1_l855_85583


namespace negation_proposition_l855_85500

theorem negation_proposition (p : Prop) : 
  (∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ ¬ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by
  sorry

end negation_proposition_l855_85500


namespace tiffany_total_bags_l855_85593

def initial_bags : ℕ := 10
def found_on_tuesday : ℕ := 3
def found_on_wednesday : ℕ := 7
def total_bags : ℕ := 20

theorem tiffany_total_bags (initial_bags : ℕ) (found_on_tuesday : ℕ) (found_on_wednesday : ℕ) (total_bags : ℕ) :
    initial_bags + found_on_tuesday + found_on_wednesday = total_bags :=
by
  sorry

end tiffany_total_bags_l855_85593


namespace part_I_part_II_l855_85557

def f (x a : ℝ) : ℝ := |x - 4 * a| + |x|

theorem part_I (a : ℝ) (h : -4 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, f x a ≥ a^2 := 
sorry

theorem part_II (x y z : ℝ) (h : 4 * x + 2 * y + z = 4) :
  (x + y)^2 + y^2 + z^2 ≥ 16 / 21 :=
sorry

end part_I_part_II_l855_85557


namespace find_two_digit_integers_l855_85546

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem find_two_digit_integers
    (a b : ℕ) :
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 
    (a = b + 12 ∨ b = a + 12) ∧
    (a / 10 = b / 10 ∨ a % 10 = b % 10) ∧
    (sum_of_digits a = sum_of_digits b + 3 ∨ sum_of_digits b = sum_of_digits a + 3) :=
sorry

end find_two_digit_integers_l855_85546


namespace mary_received_more_l855_85521

theorem mary_received_more (investment_Mary investment_Harry profit : ℤ)
  (one_third_profit divided_equally remaining_profit : ℤ)
  (total_Mary total_Harry difference : ℤ)
  (investment_ratio_Mary investment_ratio_Harry : ℚ) :
  investment_Mary = 700 →
  investment_Harry = 300 →
  profit = 3000 →
  one_third_profit = profit / 3 →
  divided_equally = one_third_profit / 2 →
  remaining_profit = profit - one_third_profit →
  investment_ratio_Mary = 7/10 →
  investment_ratio_Harry = 3/10 →
  total_Mary = divided_equally + investment_ratio_Mary * remaining_profit →
  total_Harry = divided_equally + investment_ratio_Harry * remaining_profit →
  difference = total_Mary - total_Harry →
  difference = 800 := by
  sorry

end mary_received_more_l855_85521


namespace min_value_of_function_l855_85564

theorem min_value_of_function (x : ℝ) (h : x > 2) : (x + 1 / (x - 2)) ≥ 4 :=
  sorry

end min_value_of_function_l855_85564


namespace chenny_friends_count_l855_85523

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end chenny_friends_count_l855_85523


namespace exponent_fraction_equals_five_fourths_l855_85509

theorem exponent_fraction_equals_five_fourths :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5 / 4 :=
by
  sorry

end exponent_fraction_equals_five_fourths_l855_85509


namespace packing_objects_in_boxes_l855_85541

theorem packing_objects_in_boxes 
  (n k : ℕ) (n_pos : 0 < n) (k_pos : 0 < k) 
  (objects : Fin (n * k) → Fin k) 
  (boxes : Fin k → Fin n → Fin k) :
  ∃ (pack : Fin (n * k) → Fin k), 
    (∀ i, ∃ c1 c2, 
      ∀ j, pack i = pack j → 
      (objects i = c1 ∨ objects i = c2 ∧
      objects j = c1 ∨ objects j = c2)) := 
sorry

end packing_objects_in_boxes_l855_85541


namespace sum_series_div_3_powers_l855_85566

theorem sum_series_div_3_powers : ∑' k : ℕ, (k+1) / 3^(k+1) = 3 / 4 :=
by 
-- sorry, leveraging only the necessary conditions and focusing on the final correct answer.
sorry

end sum_series_div_3_powers_l855_85566


namespace min_value_of_x_plus_y_l855_85571

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l855_85571


namespace combined_share_is_50000_l855_85540

def profit : ℝ := 80000

def majority_owner_share : ℝ := 0.25 * profit

def remaining_profit : ℝ := profit - majority_owner_share

def partner_share : ℝ := 0.25 * remaining_profit

def combined_share_majority_two_owners : ℝ := majority_owner_share + 2 * partner_share

theorem combined_share_is_50000 :
  combined_share_majority_two_owners = 50000 := 
by 
  sorry

end combined_share_is_50000_l855_85540


namespace roger_initial_money_l855_85562

theorem roger_initial_money (x : ℤ) 
    (h1 : x + 28 - 25 = 19) : 
    x = 16 := 
by 
    sorry

end roger_initial_money_l855_85562


namespace overtaking_time_l855_85599

variable (a_speed b_speed k_speed : ℕ)
variable (b_delay : ℕ) 
variable (t : ℕ)
variable (t_k : ℕ)

theorem overtaking_time (h1 : a_speed = 30)
                        (h2 : b_speed = 40)
                        (h3 : k_speed = 60)
                        (h4 : b_delay = 5)
                        (h5 : 30 * t = 40 * (t - 5))
                        (h6 : 30 * t = 60 * t_k)
                         : k_speed / 3 = 10 :=
by sorry

end overtaking_time_l855_85599


namespace Debby_daily_bottles_is_six_l855_85534

def daily_bottles (total_bottles : ℕ) (total_days : ℕ) : ℕ :=
  total_bottles / total_days

theorem Debby_daily_bottles_is_six : daily_bottles 12 2 = 6 := by
  sorry

end Debby_daily_bottles_is_six_l855_85534
